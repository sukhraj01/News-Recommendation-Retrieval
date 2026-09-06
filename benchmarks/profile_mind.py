#!/usr/bin/env python3
"""Per-stage, per-query latency/throughput profile of the MIND serving path.

Covers every stage the brief names for MIND: tokenization, BM25 index
lookup/scoring, embedding retrieval (the cosine-similarity pass), the hybrid
ranker, and NRMS inference — plus the ranking tie-break and metric
computation, which are part of the real end-to-end cost and would otherwise
disappear into an unattributed remainder.

Methodology (full rationale in ADR-014):

- **Real scale index, sampled queries.** Every stage runs against the complete
  MINDlarge-dev catalog (72,023 articles, real vocabulary, real 384-dim
  embedding matrix). Only the *queries* are sampled, systematically by user so
  the 1.47 impressions/user multiplicity — which governs how often the per-user
  score caches hit — is preserved. See `loaders.py`.
- **Two clocks, cross-checked.** Manual `StageTimer` instrumentation gives the
  stage table; a `cProfile` pass over a sub-batch gives independent
  function-level attribution. If the two disagree about which stage dominates,
  the disagreement is the finding — so both are reported, never just one.
- **Warmup excluded, and said so.** BLAS thread-pool spin-up and the first
  touch of the memory-mapped embedding matrix are one-time costs; the first
  `--warmup` users are timed and discarded.
- **Reconciliation, not replacement.** The script projects a full-split time
  from the sampled per-query costs and prints it next to ADR-006's ~9.8 min and
  ADR-008's ~10.4 min macro projections. Those numbers were measured
  independently; if this profile contradicts them, that is reported as a
  contradiction rather than quietly overwriting them.

NRMS caveat, stated up front because it bounds what these numbers mean: no
trained Candidate J checkpoint exists on this machine (only `config.json` /
`results.json` survived locally — the weights stayed on Ada). The NRMS stage
therefore runs a randomly-initialised `NRMSLite` built with Candidate J's REAL
recorded architecture (embed_dim=300, num_heads=15, max_title_len=20,
max_history_len=50, from `experiments/candidate_j_nrms_lite_ada_mindlarge_
2026-08-25/config.json`). Inference latency is a function of tensor shapes and
op dispatch, not of weight *values*, so the timing is valid; any *accuracy*
number from this path would not be, and none is reported.

Usage:
    poetry run python benchmarks/profile_mind.py
    poetry run python benchmarks/profile_mind.py --n-users 3000 --device cpu
    poetry run python benchmarks/profile_mind.py --bundle small --split dev
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.loaders import (  # noqa: E402
    embeddings_cache_path,
    load_articles,
    load_history,
    sample_impressions_by_user,
)
from benchmarks.timing import (  # noqa: E402
    StageTimer,
    git_dirty,
    git_rev,
    hardware_label,
    hardware_spec,
    markdown_table,
    peak_rss_gb,
    stats_to_rows,
    write_results,
)
from src.evaluation.ranking_metrics import (  # noqa: E402
    mrr,
    ndcg_at_k,
    rank_candidates,
    safe_auc,
)
from src.retrieval.embed import (  # noqa: E402
    DEFAULT_MODEL,
    build_embedding_index,
    build_user_embedding_query,
    model_slug,
)
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import _lookup_scores, score_all
from src.retrieval.tokenize import tokenize

SEED = 0

# Candidate J's REAL deployed architecture, read from its own recorded config
# rather than retyped — a hardcoded copy here would silently drift if the
# experiment record were ever corrected.
_J_CONFIG = _REPO_ROOT / "experiments" / "candidate_j_nrms_lite_ada_mindlarge_2026-08-25" / "config.json"

# ADR-006 / ADR-008's independently-measured macro figures for MINDlarge-dev,
# carried here so every run reconciles against them automatically instead of
# the comparison being done by hand once and then going stale.
MACRO_REFERENCE = {
    "bm25_index_build_s": 1.70,          # ADR-006 addendum, 2026-08-11
    "bm25_full_retrieval_min": 9.8,      # ADR-006 addendum, projected
    "embed_encode_s": 264.4,             # ADR-008 addendum, 72,023 articles
    "embed_full_retrieval_min": 10.4,    # ADR-008 addendum, projected
    "encoder_throughput_articles_s": 373.6,
    "source": "PROJECT_STATE.md 'Part 2 — Benchmark before trusting anything at MINDlarge scale'",
}


def _minmax(x: np.ndarray) -> np.ndarray:
    """Identical to `scripts/run_leakage_ablation.py::_minmax`, which the real
    hybrid arm uses. Duplicated (not imported) only because importing it drags
    in that script's EB-NeRD zip-reading module scope; the function is four
    lines and is covered by its own unit test there."""
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _load_nrms_config() -> dict:
    import json

    if _J_CONFIG.exists():
        cfg = json.loads(_J_CONFIG.read_text())
        return {
            "embed_dim": cfg["embed_dim"],
            "num_heads": cfg["num_heads"],
            "max_title_len": cfg["max_title_len"],
            "max_history_len": cfg["max_history_len"],
            "recorded_vocab_size": cfg["vocab_size"],
            "source": str(_J_CONFIG.relative_to(_REPO_ROOT)),
        }
    return {
        "embed_dim": 300, "num_heads": 15, "max_title_len": 20,
        "max_history_len": 50, "recorded_vocab_size": None,
        "source": "defaults (candidate J config.json not found)",
    }


def build_stack(articles: pd.DataFrame, history: pd.DataFrame, dataset: str,
                bundle: str, split: str, device: str, timer: StageTimer) -> dict:
    """One-time setup: BM25 index, embedding index, NRMS model + title matrix.

    Timed and reported, but deliberately EXCLUDED from the per-query
    percentage denominator — a fixed startup cost amortised over an entire
    split is a different engineering quantity from marginal per-query latency,
    and folding it in would make every per-query stage look artificially small.
    """
    import torch

    from src.retrieval.nrms import NRMSLite, build_vocab
    from src.retrieval.nrms_training import NRMSLiteScorer, build_title_matrix

    setup: dict = {}

    t0 = time.perf_counter()
    bm25 = build_index(articles)
    setup["bm25_index_build_s"] = time.perf_counter() - t0
    setup["bm25_vocab_size"] = len(bm25.vocab)
    setup["bm25_weights_nnz"] = int(bm25.weights_t.nnz)
    setup["bm25_weights_mb"] = round(
        (bm25.weights_t.data.nbytes + bm25.weights_t.indices.nbytes
         + bm25.weights_t.indptr.nbytes) / 1e6, 1
    )

    cache = embeddings_cache_path(dataset, bundle, split, model_slug(DEFAULT_MODEL))
    t0 = time.perf_counter()
    emb = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache)
    setup["embed_index_load_s"] = time.perf_counter() - t0
    setup["embed_from_cache"] = cache.exists()
    setup["embed_matrix_mb"] = round(emb.vectors.nbytes / 1e6, 1)
    setup["embed_dim"] = int(emb.vectors.shape[1])

    text_lookup = dict(zip(
        articles["article_id"],
        articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
    ))
    vector_lookup = dict(zip(emb.article_ids, emb.vectors))

    ncfg = _load_nrms_config()
    t0 = time.perf_counter()
    word2id = build_vocab(articles["title"].fillna("").tolist(), min_freq=2)
    setup["nrms_vocab_build_s"] = time.perf_counter() - t0
    setup["nrms_vocab_size"] = len(word2id)
    setup["nrms_config"] = ncfg

    t0 = time.perf_counter()
    title_matrix, news_id2row, pad_news_row = build_title_matrix(
        articles, word2id, ncfg["max_title_len"]
    )
    torch.manual_seed(SEED)
    model = NRMSLite(
        vocab_size=len(word2id), embed_dim=ncfg["embed_dim"], num_heads=ncfg["num_heads"]
    )
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)
    title_matrix = title_matrix.to(dev)
    setup["nrms_model_init_s"] = time.perf_counter() - t0
    setup["nrms_params"] = int(sum(p.numel() for p in model.parameters()))
    setup["nrms_device"] = str(dev)

    nrms_scorer = NRMSLiteScorer(
        model, news_id2row, pad_news_row, title_matrix, ncfg["max_history_len"], dev
    )

    hist_by_user = dict(zip(history["user_id"], history["article_ids"]))

    return {
        "setup": setup, "bm25": bm25, "emb": emb, "text_lookup": text_lookup,
        "vector_lookup": vector_lookup, "nrms": nrms_scorer, "hist_by_user": hist_by_user,
    }


def run_batch(stack: dict, rows: pd.DataFrame, timer: StageTimer,
              users: list[str] | None = None) -> dict:
    """Drive the full per-query path for every impression of every user.

    Structured exactly like `scripts/run_ranking_eval.py`'s loop — per-user
    query construction and full-corpus scoring hoisted out of the
    per-impression loop, because that is where the deployed harness does them.
    Measuring a different loop nesting would produce a table that describes
    software this project does not run.
    """
    grouped = rows.groupby("user_id", sort=False)
    counts = {"users": 0, "impressions": 0, "candidate_rows": 0, "history_articles": 0}

    for user_id, user_rows in grouped:
        if users is not None and user_id not in users:
            continue
        hist = list(stack["hist_by_user"].get(user_id, []))
        counts["users"] += 1
        counts["history_articles"] += len(hist)

        # ---- per-user stages -------------------------------------------
        with timer("1_query_tokenize"):
            bm25_query = build_user_query(hist, stack["text_lookup"])

        with timer("2_bm25_score_all"):
            bm25_full = score_all(stack["bm25"], bm25_query)

        with timer("3_embed_query_build"):
            embed_query = build_user_embedding_query(hist, stack["vector_lookup"])

        with timer("4_embed_score_all"):
            if embed_query is None:
                embed_full = np.zeros(len(stack["emb"].article_ids))
            else:
                embed_full = stack["emb"].vectors @ embed_query

        # ---- per-impression stages -------------------------------------
        for impression_id, imp in user_rows.groupby("impression_id", sort=False):
            cand = imp["article_id"].tolist()
            clicked = imp["clicked"].to_numpy(dtype=bool)
            counts["impressions"] += 1
            counts["candidate_rows"] += len(cand)

            with timer("5_bm25_candidate_lookup"):
                bm25_scores = _lookup_scores(bm25_full, stack["bm25"].id_to_col, cand)

            with timer("6_embed_candidate_lookup"):
                embed_scores = _lookup_scores(embed_full, stack["emb"].id_to_col, cand)

            with timer("7_hybrid_blend"):
                hybrid = 0.5 * _minmax(bm25_scores) + 0.5 * _minmax(embed_scores)

            with timer("8_nrms_inference"):
                stack["nrms"].score(hist, cand)

            with timer("9_rank_tiebreak"):
                order = rank_candidates(hybrid, impression_id, seed=SEED)
                ranked_clicked = clicked[order]

            with timer("10_metrics"):
                safe_auc(hybrid, clicked)
                mrr(ranked_clicked)
                ndcg_at_k(ranked_clicked, 5)
                ndcg_at_k(ranked_clicked, 10)

    return counts


def cprofile_batch(stack: dict, rows: pd.DataFrame, n_users: int, top_n: int = 30) -> dict:
    """Independent function-level attribution over a sub-batch.

    `cProfile` adds real per-call overhead (it instruments every Python call),
    so its ABSOLUTE times are not comparable to the StageTimer table and are
    not reported as latencies. What it is used for is the RELATIVE ranking of
    where cumulative time goes, as a cross-check on the manual instrumentation
    — and, critically, to catch time spent in places the manual stages do not
    name at all.
    """
    users = list(dict.fromkeys(rows["user_id"].tolist()))[:n_users]
    subset = rows[rows["user_id"].isin(set(users))]
    throwaway = StageTimer()

    prof = cProfile.Profile()
    prof.enable()
    run_batch(stack, subset, throwaway)
    prof.disable()

    buf = io.StringIO()
    stats = pstats.Stats(prof, stream=buf).sort_stats("cumulative")
    stats.print_stats(top_n)
    raw = buf.getvalue()

    entries = []
    for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
        entries.append({
            "function": f"{Path(func[0]).name}:{func[1]}({func[2]})",
            "ncalls": nc, "tottime_s": round(tt, 4), "cumtime_s": round(ct, 4),
        })
    entries.sort(key=lambda e: -e["cumtime_s"])

    return {
        "n_users_profiled": len(users),
        "n_impressions_profiled": int(subset["impression_id"].nunique()),
        "note": "cProfile inflates absolute times; use for relative attribution only",
        "top_by_cumtime": entries[:top_n],
        "top_by_tottime": sorted(entries, key=lambda e: -e["tottime_s"])[:top_n],
        "pstats_text": raw,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="large")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n-users", type=int, default=1200,
                    help="users sampled systematically; all their impressions are kept")
    ap.add_argument("--warmup", type=int, default=25,
                    help="users timed then discarded, to exclude one-time warmup costs")
    ap.add_argument("--device", default=None,
                    help="torch device for NRMS (default: auto — cuda > mps > cpu)")
    ap.add_argument("--cprofile-users", type=int, default=150)
    ap.add_argument("--no-cprofile", action="store_true")
    ap.add_argument("--tag", default="", help="free-text label stored in the results JSON")
    args = ap.parse_args()

    from src.retrieval.embed import _default_device

    device = args.device or _default_device()

    print(f"[mind] hardware: {hardware_label()}", flush=True)
    print(f"[mind] loading MIND {args.bundle}/{args.split} …", flush=True)

    t_load0 = time.perf_counter()
    articles = load_articles("mind", args.bundle, args.split)
    rows, sample_meta = sample_impressions_by_user("mind", args.bundle, args.split,
                                                   args.n_users, seed=SEED)
    history = load_history("mind", args.bundle, args.split, rows["user_id"].unique())
    load_s = time.perf_counter() - t_load0
    print(f"[mind] {len(articles):,} articles | {sample_meta['sampled_impressions']:,} "
          f"impressions / {sample_meta['sampled_users']:,} users "
          f"({sample_meta['sampled_impressions_per_user']} imp/user vs. population "
          f"{sample_meta['population_impressions_per_user']}) | {load_s:.1f}s", flush=True)

    timer = StageTimer()
    stack = build_stack(articles, history, "mind", args.bundle, args.split, device, timer)
    print(f"[mind] setup done: bm25 {stack['setup']['bm25_index_build_s']:.2f}s | "
          f"embed {stack['setup']['embed_index_load_s']:.2f}s | "
          f"nrms {stack['setup']['nrms_model_init_s']:.2f}s ({device})", flush=True)

    all_users = list(dict.fromkeys(rows["user_id"].tolist()))
    warm_users, bench_users = set(all_users[:args.warmup]), set(all_users[args.warmup:])

    print(f"[mind] warmup on {len(warm_users)} users …", flush=True)
    run_batch(stack, rows[rows["user_id"].isin(warm_users)], StageTimer())

    print(f"[mind] timing {len(bench_users):,} users …", flush=True)
    t0 = time.perf_counter()
    counts = run_batch(stack, rows[rows["user_id"].isin(bench_users)], timer)
    wall_s = time.perf_counter() - t0

    stats = timer.summary()
    n_imp = counts["impressions"]

    # Amortised per-impression cost: per-USER stages divided by the number of
    # impressions they served, so every row of the table is denominated in the
    # same unit and the percentages compose.
    per_impression_ms = {s.stage: s.total_s * 1e3 / n_imp for s in stats}
    stage_total_s = sum(s.total_s for s in stats)

    # ---- reconciliation against the recorded macro numbers ---------------
    pop_imp = sample_meta["population_impressions"]
    pop_users = sample_meta["population_users"]
    bm25_per_user_s = next(s.total_s for s in stats if s.stage == "2_bm25_score_all") / counts["users"]
    embed_per_user_s = next(s.total_s for s in stats if s.stage == "4_embed_score_all") / counts["users"]
    tok_per_user_s = next(s.total_s for s in stats if s.stage == "1_query_tokenize") / counts["users"]

    reconciliation = {
        "reference": MACRO_REFERENCE,
        "measured_bm25_index_build_s": round(stack["setup"]["bm25_index_build_s"], 3),
        "projected_full_split": {
            # ADR-006's "~9.8 min full retrieval" is the whole-corpus scoring
            # pass over every user — reproduced here as (per-user tokenize +
            # per-user sparse matvec) x all users, the same quantity.
            "bm25_query_plus_score_min": round(
                (tok_per_user_s + bm25_per_user_s) * pop_users / 60, 2),
            "bm25_score_only_min": round(bm25_per_user_s * pop_users / 60, 2),
            "embed_score_only_min": round(embed_per_user_s * pop_users / 60, 2),
            "basis": f"{counts['users']:,} sampled users x {pop_users:,} population users",
        },
        "projected_end_to_end_all_stages_min": round(
            stage_total_s / n_imp * pop_imp / 60, 2),
    }

    payload = {
        "benchmark": "profile_mind",
        "dataset": "mind", "bundle": args.bundle, "split": args.split,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_rev": git_rev(), "git_dirty": git_dirty(), "tag": args.tag,
        "hardware_label": hardware_label(), "hardware": hardware_spec(),
        "sample": sample_meta,
        "counts": counts,
        "corpus": {"n_articles": int(len(articles))},
        "setup": stack["setup"],
        "data_load_s": round(load_s, 2),
        "wall_s": round(wall_s, 3),
        "stage_total_s": round(stage_total_s, 3),
        "unattributed_s": round(wall_s - stage_total_s, 3),
        "per_impression_ms": {k: round(v, 4) for k, v in per_impression_ms.items()},
        "end_to_end_impressions_per_s": round(n_imp / wall_s, 1),
        "end_to_end_ms_per_impression": round(wall_s * 1e3 / n_imp, 3),
        "stages": stats_to_rows(stats),
        "reconciliation": reconciliation,
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }

    if not args.no_cprofile:
        print(f"[mind] cProfile over {args.cprofile_users} users …", flush=True)
        payload["cprofile"] = cprofile_batch(stack, rows, args.cprofile_users)

    path = write_results(f"mind_{args.bundle}_{args.split}_profile", payload)

    print()
    print(markdown_table(stats, unit="call"))
    print(f"end-to-end: {payload['end_to_end_ms_per_impression']} ms/impression "
          f"({payload['end_to_end_impressions_per_s']} impressions/s), "
          f"unattributed {payload['unattributed_s']:.2f}s of {wall_s:.2f}s")
    print(f"peak RSS {payload['peak_rss_gb']} GB")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
