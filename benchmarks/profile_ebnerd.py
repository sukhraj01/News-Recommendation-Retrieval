#!/usr/bin/env python3
"""Per-stage, per-query latency/throughput profile of the EB-NeRD serving path.

The EB-NeRD path is structurally different from MIND's, which is precisely why
the brief asks for a separate table rather than assuming the bottlenecks match.
MIND ranks by scoring a query representation against a corpus-wide index; EB-NeRD's
deployed Candidate K (ADR-013) instead builds a 65-column engineered feature
matrix per impression and runs a trained LightGBM booster over it. There is no
corpus-wide scoring pass at all, and no transformer at inference time.

Stages measured, matching `run_ebnerd_gbdt_experiment.py::stream_validation_metrics`
— the code path that actually produced the 0.7542 Codabench submission:

  setup   article table / user profiles / history popularity / behaviours load
  1       feature frame build       (the 65-feature engineering pass)
  1a        └ short-term features   (sub-stage, timed by wrapping the real call)
  2       LightGBM predict          (booster inference)
  3       rank + ADR-007 tie-break
  4       metrics (AUC / MRR / nDCG)

The booster is the REAL trained Candidate K model
(`experiments/candidate_k_gbdt_ebnerd_small_2026-08-26/model_K_rank.txt`), not a
stand-in — unlike MIND's NRMS stage, whose trained weights never came back from
Ada. Tree count and feature count are recorded in the results JSON so the
latency is interpretable against the model that produced it.

Usage:
    poetry run python benchmarks/profile_ebnerd.py
    poetry run python benchmarks/profile_ebnerd.py --n-impressions 20000
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

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

SEED = 0
MODEL_DIR = _REPO_ROOT / "experiments" / "candidate_k_gbdt_ebnerd_small_2026-08-26"
DEFAULT_MODEL = MODEL_DIR / "model_K_rank.txt"

# ADR-013's own recorded macro timings for this same path, carried here so each
# run reconciles automatically rather than by hand.
MACRO_REFERENCE = {
    "source": "experiments/candidate_k_gbdt_ebnerd_small_2026-08-26/config.json (timings block)",
}


def _load_reference_timings() -> dict:
    cfg = MODEL_DIR / "config.json"
    if not cfg.exists():
        return dict(MACRO_REFERENCE)
    data = json.loads(cfg.read_text())
    out = dict(MACRO_REFERENCE)
    for key in ("timings", "timings_s", "n_train_impressions", "n_validation_impressions"):
        if key in data:
            out[key] = data[key]
    return out


def resolve_model_features(booster, model_path: Path) -> tuple[list[str], list[int] | None, str]:
    """Map the booster's columns onto positions in the CURRENT `FEATURE_NAMES`.

    This is not bookkeeping — getting it wrong feeds the model silently
    misaligned columns and produces plausible, wrong scores.

    Two facts make it non-trivial, both discovered by this benchmark failing
    loudly on Ada rather than quietly succeeding:

    1. `model_K_rank.txt` was saved WITHOUT feature names, so
       `booster.feature_name()` returns `Column_0 … Column_59` — useless for
       matching, and `FEATURE_NAMES.index("Column_0")` raises.
    2. The locally-available booster has **60** features, not the 65 in today's
       `FEATURE_NAMES`. It is the 2026-08-26 `ebnerd_small` model, trained
       BEFORE ADR-013's correction added `context_category_match`,
       `context_topic_overlap`, `context_embed_sim`, `session_position` and
       `session_start_gap_h`. The 65-feature model that actually produced the
       0.7542 Codabench result was trained on `ebnerd_large` on Ada and its
       weights were never brought back — only `config.json`,
       `results.json` and `feature_importance.json` exist locally.

    The authoritative source is therefore the training run's own recorded
    `config.json["feature_names"]`, which preserves the exact training-time
    column ORDER. Falling back to the booster's names is only valid when those
    names are real. If neither resolves, this raises rather than guessing —
    a wrong mapping here is worse than no measurement.
    """
    # Imported here rather than at module scope to match this file's existing
    # convention of keeping the heavy `ebnerd_features` import inside functions.
    from src.retrieval.ebnerd_features import FEATURE_NAMES

    cfg = model_path.parent / "config.json"
    if cfg.exists():
        recorded = json.loads(cfg.read_text()).get("feature_names")
        if recorded and len(recorded) == booster.num_feature():
            missing = [n for n in recorded if n not in FEATURE_NAMES]
            if missing:
                raise SystemExit(
                    f"model was trained on features absent from today's FEATURE_NAMES: {missing}"
                )
            return recorded, [FEATURE_NAMES.index(n) for n in recorded], str(cfg.name)

    names = booster.feature_name()
    if all(n in FEATURE_NAMES for n in names):
        idx = [FEATURE_NAMES.index(n) for n in names]
        return names, (None if idx == list(range(len(FEATURE_NAMES))) else idx), "booster"

    raise SystemExit(
        f"cannot map {booster.num_feature()} booster columns onto FEATURE_NAMES: the model "
        f"has no usable feature names ({names[:3]}…) and {cfg} records none matching. "
        f"Refusing to guess a column order."
    )


def build_stack(zip_path: Path, split: str, emb_npy: Path, emb_json: Path) -> dict:
    """One-time setup, timed stage by stage.

    Deliberately excluded from the per-query denominator for the same reason as
    MIND's index build: this is fixed startup cost amortised across a whole
    split, a different quantity from marginal per-impression latency. It is
    reported separately and in full, because on EB-NeRD it is *large* relative
    to MIND's and that is itself a finding.
    """
    from src.retrieval.ebnerd_features import (
        add_session_position_columns,
        build_history_popularity,
        build_user_profiles,
        load_article_table,
    )
    from run_ebnerd_gbdt_experiment import load_behaviors

    setup: dict = {}

    t0 = time.perf_counter()
    art = load_article_table(zip_path, embeddings_npy=emb_npy, embeddings_json=emb_json)
    setup["article_table_s"] = time.perf_counter() - t0
    setup["n_articles"] = int(len(art.category))

    t0 = time.perf_counter()
    prof = build_user_profiles(zip_path, split, art)
    setup["user_profiles_s"] = time.perf_counter() - t0
    setup["n_profiled_users"] = int(len(prof.id_to_pos))

    t0 = time.perf_counter()
    pop = build_history_popularity(zip_path, split, art)
    setup["history_popularity_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    beh = load_behaviors(zip_path, split)
    # Session columns must be computed on the FULL split before any chunking —
    # a session spanning a chunk boundary would otherwise be miscomputed (see
    # build_feature_frame's docstring). Timed as setup, not per-query, because
    # that is where the real pipeline does it.
    beh = add_session_position_columns(beh)
    setup["behaviors_load_s"] = time.perf_counter() - t0
    setup["n_impressions_in_split"] = int(len(beh))

    return {"setup": setup, "art": art, "prof": prof, "pop": pop, "beh": beh}


def sample_behaviors(beh: pd.DataFrame, n: int, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Systematic sample of whole impressions — same reasoning as
    `loaders.sample_impressions_by_user`, and the same reason
    `run_ebnerd_gbdt_experiment.py --train-sample-impressions` samples whole
    impressions: a partial candidate list would break the group structure the
    ranker is defined over.

    Systematic rather than a prefix specifically because EB-NeRD is known to
    cluster a 200,000-row `is_beyond_accuracy` block, which a prefix sample
    could land entirely inside or entirely outside.
    """
    total = len(beh)
    if n >= total:
        return beh.reset_index(drop=True), {
            "sampled_impressions": total, "population_impressions": total,
            "sampling": "full-split", "sample_seed": seed,
        }
    step = total / n
    offset = int(np.random.default_rng(seed).integers(0, max(1, int(step))))
    idx = np.unique(np.clip(np.floor(np.arange(n) * step).astype(np.int64) + offset, 0, total - 1))
    sub = beh.iloc[idx].reset_index(drop=True)
    return sub, {
        "sampled_impressions": int(len(sub)),
        "population_impressions": int(total),
        "sampling": "systematic-by-impression",
        "sample_seed": seed,
    }


def run_batch(stack: dict, beh: pd.DataFrame, booster, feature_idx, timer: StageTimer,
              chunk_size: int) -> dict:
    """Drive the real serving path in chunks, exactly as
    `stream_validation_metrics` does.

    Chunked rather than per-impression because that IS the deployed shape: the
    feature builder is vectorised across an impression batch, and calling it
    once per impression would measure software this project does not run (and
    would report a per-query cost several times too high). Per-impression
    latency is therefore derived by dividing a chunk's stage time by the
    impressions in it — stated explicitly here so the derivation is auditable
    rather than implied.
    """
    from src.evaluation.ranking_metrics import mrr, ndcg_at_k, per_impression_auc, rank_candidates
    from src.retrieval import ebnerd_features as ef

    counts = {"impressions": 0, "candidate_rows": 0, "chunks": 0}

    # Time the short-term feature sub-stage by wrapping the real function in
    # its own module namespace — `build_feature_frame` resolves it by global
    # lookup, so this measures the genuine call without editing src/.
    original_st = ef.compute_short_term_features

    def timed_st(*a, **kw):
        with timer("1a_short_term_features"):
            return original_st(*a, **kw)

    ef.compute_short_term_features = timed_st
    try:
        for start in range(0, len(beh), chunk_size):
            sub = beh.iloc[start : start + chunk_size]

            with timer("1_feature_frame_build"):
                X, meta = ef.build_feature_frame(sub, stack["art"], stack["prof"],
                                                 stack["pop"], with_labels=True)

            mat = X if feature_idx is None else X[:, feature_idx]

            with timer("2_lightgbm_predict"):
                scores = booster.predict(mat, num_iteration=booster.best_iteration)

            labels, sizes = meta["label"], meta["group_sizes"]
            starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
            imp_ids = meta["impression_id"]
            scores = np.asarray(scores, dtype=np.float64)

            with timer("3_rank_tiebreak"):
                orders = [
                    rank_candidates(scores[s : s + n], str(imp_ids[i]), seed=SEED)
                    for i, (s, n) in enumerate(zip(starts, sizes))
                ]

            with timer("4_metrics"):
                per_impression_auc(scores, labels, sizes)
                for i, (s, n) in enumerate(zip(starts, sizes)):
                    ranked = np.asarray(labels[s : s + n], dtype=bool)[orders[i]]
                    mrr(ranked)
                    ndcg_at_k(ranked, 5)
                    ndcg_at_k(ranked, 10)

            counts["impressions"] += len(sub)
            counts["candidate_rows"] += int(X.shape[0])
            counts["chunks"] += 1
            del X, meta, mat, scores
    finally:
        ef.compute_short_term_features = original_st

    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="small")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-impressions", type=int, default=25_000)
    ap.add_argument("--chunk-size", type=int, default=5_000)
    ap.add_argument("--warmup-chunks", type=int, default=1)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--cprofile-impressions", type=int, default=5_000)
    ap.add_argument("--no-cprofile", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import lightgbm as lgb

    from src.retrieval.ebnerd_features import FEATURE_NAMES

    zip_path = _REPO_ROOT / "data" / "raw" / "ebnerd" / f"ebnerd_{args.bundle}.zip"
    emb_dir = _REPO_ROOT / "data" / "processed" / "ebnerd" / args.bundle / "embeddings"
    slug = "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2"
    emb_npy, emb_json = emb_dir / f"{slug}.npy", emb_dir / f"{slug}.json"

    print(f"[ebnerd] hardware: {hardware_label()}", flush=True)
    print(f"[ebnerd] setup from {zip_path.name} …", flush=True)

    stack = build_stack(zip_path, args.split, emb_npy, emb_json)
    s = stack["setup"]
    print(f"[ebnerd] setup: articles {s['article_table_s']:.1f}s | profiles "
          f"{s['user_profiles_s']:.1f}s | popularity {s['history_popularity_s']:.1f}s | "
          f"behaviours {s['behaviors_load_s']:.1f}s", flush=True)

    booster = lgb.Booster(model_file=str(args.model))
    model_features, feature_idx, feature_source = resolve_model_features(booster, args.model)
    print(f"[ebnerd] model {args.model.name}: {booster.num_trees()} trees, "
          f"{len(model_features)} features (mapped via {feature_source}), "
          f"best_iter={booster.best_iteration}", flush=True)
    if len(model_features) != len(FEATURE_NAMES):
        print(f"[ebnerd] NOTE: this booster uses {len(model_features)} of today's "
              f"{len(FEATURE_NAMES)} features — it predates ADR-013's correction. "
              f"Absent: {[n for n in FEATURE_NAMES if n not in model_features]}", flush=True)

    beh, sample_meta = sample_behaviors(stack["beh"], args.n_impressions)
    print(f"[ebnerd] {sample_meta['sampled_impressions']:,} of "
          f"{sample_meta['population_impressions']:,} impressions", flush=True)

    warm_n = args.warmup_chunks * args.chunk_size
    if warm_n:
        print(f"[ebnerd] warmup on {min(warm_n, len(beh)):,} impressions …", flush=True)
        run_batch(stack, beh.iloc[:warm_n], booster, feature_idx, StageTimer(), args.chunk_size)

    timer = StageTimer()
    print(f"[ebnerd] timing {len(beh) - warm_n:,} impressions …", flush=True)
    t0 = time.perf_counter()
    counts = run_batch(stack, beh.iloc[warm_n:], booster, feature_idx, timer, args.chunk_size)
    wall_s = time.perf_counter() - t0

    # `1a_short_term_features` is a SUB-stage of `1_feature_frame_build`; leaving
    # it in the denominator would double-count its time and make every
    # percentage in the table wrong.
    denom = [st for st in
             ["1_feature_frame_build", "2_lightgbm_predict", "3_rank_tiebreak", "4_metrics"]]
    stats = timer.summary(denominator_stages=denom)
    n_imp = counts["impressions"]
    stage_total_s = timer.total_s(denom)

    payload = {
        "benchmark": "profile_ebnerd",
        "dataset": "ebnerd", "bundle": args.bundle, "split": args.split,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_rev": git_rev(), "git_dirty": git_dirty(), "tag": args.tag,
        "hardware_label": hardware_label(), "hardware": hardware_spec(),
        "sample": sample_meta,
        "counts": counts,
        "chunk_size": args.chunk_size,
        "model": {
            "path": str(args.model.relative_to(_REPO_ROOT)),
            "n_trees": int(booster.num_trees()),
            "n_features": len(model_features),
            "best_iteration": booster.best_iteration,
            "trained": True,
            "feature_name_source": feature_source,
            "features_absent_vs_current": [n for n in FEATURE_NAMES if n not in model_features],
            "caveat": ("locally-available ebnerd_small booster (2026-08-26, 60 features); "
                       "the deployed 65-feature ebnerd_large model that scored 0.7542 on "
                       "Codabench was trained on Ada and its weights are not available"),
        },
        "setup": stack["setup"],
        "wall_s": round(wall_s, 3),
        "stage_total_s": round(stage_total_s, 3),
        "unattributed_s": round(wall_s - stage_total_s, 3),
        "per_impression_ms": {st.stage: round(st.total_s * 1e3 / n_imp, 4) for st in stats},
        "per_candidate_row_us": {
            st.stage: round(st.total_s * 1e6 / counts["candidate_rows"], 3) for st in stats
        },
        "end_to_end_impressions_per_s": round(n_imp / wall_s, 1),
        "end_to_end_ms_per_impression": round(wall_s * 1e3 / n_imp, 4),
        "mean_candidates_per_impression": round(counts["candidate_rows"] / n_imp, 2),
        "stages": stats_to_rows(stats),
        "reconciliation": {
            "reference": _load_reference_timings(),
            "projected_full_split_min": round(
                stage_total_s / n_imp * sample_meta["population_impressions"] / 60, 2),
        },
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }

    if not args.no_cprofile:
        print(f"[ebnerd] cProfile over {args.cprofile_impressions:,} impressions …", flush=True)
        sub = beh.iloc[warm_n : warm_n + args.cprofile_impressions]
        prof_ = cProfile.Profile()
        prof_.enable()
        run_batch(stack, sub, booster, feature_idx, StageTimer(), args.chunk_size)
        prof_.disable()
        buf = io.StringIO()
        pst = pstats.Stats(prof_, stream=buf).sort_stats("cumulative")
        pst.print_stats(30)
        entries = [
            {"function": f"{Path(f[0]).name}:{f[1]}({f[2]})", "ncalls": nc,
             "tottime_s": round(tt, 4), "cumtime_s": round(ct, 4)}
            for f, (cc, nc, tt, ct, _c) in pst.stats.items()
        ]
        entries.sort(key=lambda e: -e["cumtime_s"])
        payload["cprofile"] = {
            "n_impressions_profiled": int(len(sub)),
            "note": "cProfile inflates absolute times; use for relative attribution only",
            "top_by_cumtime": entries[:30],
            "top_by_tottime": sorted(entries, key=lambda e: -e["tottime_s"])[:30],
            "pstats_text": buf.getvalue(),
        }

    path = write_results(f"ebnerd_{args.bundle}_{args.split}_profile", payload)

    print()
    print(markdown_table(stats, unit="call"))
    print(f"end-to-end: {payload['end_to_end_ms_per_impression']} ms/impression "
          f"({payload['end_to_end_impressions_per_s']} impressions/s), "
          f"{payload['mean_candidates_per_impression']} candidates/impression")
    print(f"peak RSS {payload['peak_rss_gb']} GB")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
