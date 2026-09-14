#!/usr/bin/env python3
"""A2 Q2: the literal retrieve-then-rank harness (ADR-015's Q2 section).

Both MIND leaderboard submissions rank each impression's own in-view
candidate list -- that's what the competition scores, and it isn't the
two-stage pipeline A1 built. This script runs the actual pipeline: for a
sample of real dev impressions, A1's own retriever (`src/retrieval/index.py`
BM25 or `embed.py` embeddings) pulls the top-K candidates from the WHOLE
corpus using the user's real click history as the query (`query.py`/
`embed.py`'s query construction, unchanged), and the trained `OfficialNRMS`
(`src/retrieval/nrms_official.py`) re-ranks that retrieved list.

Per ADR-015's Q2 decision, NRMS is the only re-ranker evaluated this way:
Candidate K's GBDT features are impression-conditional (`position_in_view`,
`*_rank_in_imp`, etc.) and undefined for a candidate the retriever
surfaced but the publisher never actually showed.

**The ceiling is measured, not cited.** Recall@K for BM25/embeddings on
MIND was already benchmarked in ADR-006/ADR-008, but that number is
corpus-and-split-specific -- reusing an old citation here would risk a
scale mismatch against whichever split this script actually runs against.
Instead, the hit rate (= recall@K) is computed fresh, on the exact same
sampled impressions the before/after metrics are computed on, via the
same `ranking_metric_ci` machinery as every other metric in this project.

**Before/after, not control/treatment.** "Before" is the retrieved list in
its own retrieval-score order (unlabeled by NRMS at all -- BM25/embedding
score is standing in for "no re-ranking"). "After" is the same K candidates
re-ranked by NRMS. Both are scored with this project's own harness
(`src/evaluation/ranking_metrics.py`): AUC needs both classes present in
the list (`safe_auc` returns NaN and the impression is dropped, exactly
`ranking_metric_ci`'s existing convention, for the ~97% of impressions
where the retriever never surfaces the true click); MRR uses the standard
zero-credit-for-a-miss convention (so its absolute value is honestly
compressed by the same miss rate, not hidden); nDCG is undefined (NaN,
dropped) for a hit-less list, same as AUC. No new statistical machinery:
`paired_metric_diff_ci` (ADR-010's statistic, already used by
`a2_evaluate_scores.py` for the control/treatment A/B) is reused unchanged
for the after-vs-before paired test.

Usage:
    python scripts/a2_q2_retrieve_rerank_eval.py \
        --mind-bundle small --mind-dev-zip data/raw/mind/MINDsmall_dev.zip \
        --checkpoint /tmp/a2_q2_timing_probe/model_weights.pt --abstract-size 0 \
        --retriever bm25 --k 200 --n-impressions 3000 --out results/a2_q2/mind_small.json
"""
import argparse
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.ranking_metrics import (  # noqa: E402
    RankingMetricResult,
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.embed import build_embedding_index, build_user_embedding_query  # noqa: E402
from src.retrieval.index import build_index  # noqa: E402
from src.retrieval.nrms_official import OfficialNRMS, OfficialNRMSScorer  # noqa: E402
from src.retrieval.nrms_official_data import build_mind_news_tokens  # noqa: E402
from src.retrieval.query import build_user_query  # noqa: E402
from src.retrieval.retrieve import embed_retrieve_top_k, retrieve_top_k  # noqa: E402
from src.utils.ids import prefix_id  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_gated_cohort_experiment import paired_metric_diff_ci  # noqa: E402

ACCURACY = ("auc", "mrr", "ndcg5", "ndcg10")


def systematic(n: int, k: int | None) -> np.ndarray:
    """Same deterministic non-prefix sample as `a2_nrms_official_run.py`."""
    if not k or k >= n:
        return np.arange(n)
    return np.arange(0, n, n // k)[:k]


def read_news_tsv_raw(zip_path: Path) -> tuple[list[str], list[str], list[str]]:
    z = zipfile.ZipFile(zip_path)
    member = [m for m in z.namelist() if m.endswith("/news.tsv") or m == "news.tsv"][0]
    with z.open(member) as f:
        rows = [ln.decode("utf-8").strip("\n").split("\t") for ln in f]
    return [r[0] for r in rows], [r[3] for r in rows], [r[4] for r in rows]


def score_one_ranking(labels: np.ndarray, impression_key: str) -> dict:
    """One impression's accuracy metrics for a given (already-ordered-by-
    the-caller) label vector -- `labels[i]` is whether candidate i (in the
    order the caller wants scored) was the real click. Ties within that
    order are broken by `rank_candidates` using the score vector implied by
    position (descending), i.e. position IS the score here."""
    n = len(labels)
    pseudo_scores = np.arange(n, 0, -1, dtype=np.float64)  # already in rank order
    order = rank_candidates(pseudo_scores, impression_key)
    ranked = labels[order]
    return {
        "auc": safe_auc(pseudo_scores, labels) if n > 1 else float("nan"),
        "mrr": mrr(ranked),
        "ndcg5": ndcg_at_k(ranked, 5),
        "ndcg10": ndcg_at_k(ranked, 10),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mind-bundle", choices=["small", "large"], required=True)
    ap.add_argument("--mind-dev-zip", type=Path, required=True, help="raw zip, for NRMS token building")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--abstract-size", type=int, default=0, help="0 control / 50 treatment, per ADR-015")
    ap.add_argument("--title-size", type=int, default=30)
    ap.add_argument("--max-history-len", type=int, default=50)
    ap.add_argument("--retriever", choices=["bm25", "embedding"], default="bm25")
    ap.add_argument("--k", type=int, default=200, help="candidates retrieved per impression")
    ap.add_argument("--n-impressions", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    proc = _REPO_ROOT / "data" / "processed" / "mind" / args.mind_bundle / "dev"
    articles = pd.read_parquet(proc / "articles.parquet")
    impressions = pd.read_parquet(proc / "impressions.parquet")
    history = pd.read_parquet(proc / "user_history.parquet")
    print(f"loaded processed dev: {len(articles):,} articles, {len(impressions):,} candidate rows, "
          f"{len(history):,} users", flush=True)

    text_lookup = dict(zip(articles["article_id"], articles["title"].fillna("") + " " + articles["abstract"].fillna("")))
    hist_by_user = dict(zip(history["user_id"], history["article_ids"]))

    t0 = time.time()
    if args.retriever == "bm25":
        index = build_index(articles)

        def retrieve(query_tokens):
            return retrieve_top_k(index, query_tokens, args.k)

        def build_query(hist_ids):
            return build_user_query(list(hist_ids), text_lookup)
    else:
        cache_path = proc / "embeddings" / "sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2.npy"
        index = build_embedding_index(articles, cache_path=cache_path)
        vector_lookup = dict(zip(index.article_ids, index.vectors))

        def retrieve(query_vec):
            return embed_retrieve_top_k(index, query_vec, args.k)

        def build_query(hist_ids):
            return build_user_embedding_query(list(hist_ids), vector_lookup)
    print(f"{args.retriever} index built in {time.time()-t0:.1f}s over {len(articles):,} docs", flush=True)

    # One row per impression: user_id + the set of really-clicked article_ids.
    clicked = impressions[impressions["clicked"] == True]  # noqa: E712
    clicked_by_imp = clicked.groupby("impression_id")["article_id"].apply(set)
    users_by_imp = impressions.drop_duplicates("impression_id").set_index("impression_id")["user_id"]
    imp_ids = clicked_by_imp.index.to_numpy()
    imp_ids = imp_ids[systematic(len(imp_ids), args.n_impressions)]
    print(f"sampled {len(imp_ids):,} impressions with >=1 real click (of {len(clicked_by_imp):,} total)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    news_ids, titles, abstracts = read_news_tsv_raw(args.mind_dev_zip)
    utils_dir = _REPO_ROOT / "data" / "processed" / "mind" / "mind_utils"
    word_dict = pickle.load(open(utils_dir / "word_dict.pkl", "rb"))
    embedding = np.load(utils_dir / "embedding.npy").astype(np.float32)
    nid2index, tokens = build_mind_news_tokens(news_ids, titles, abstracts, word_dict,
                                               args.title_size, args.abstract_size)
    token_lookup = {prefix_id("mind", aid): tokens[idx] for aid, idx in nid2index.items()}
    print(f"NRMS token catalog: {len(token_lookup):,} articles, width {args.title_size + args.abstract_size}",
          flush=True)

    model = OfficialNRMS(embedding, head_num=20, head_dim=20, attention_hidden_dim=200, dropout=0.2).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)  # raises on shape mismatch
    scorer = OfficialNRMSScorer(model, token_lookup, args.max_history_len, device)
    print(f"loaded checkpoint {args.checkpoint} -- shapes matched exactly", flush=True)

    rows = []
    t0 = time.time()
    for i, imp_id in enumerate(imp_ids):
        user_id = users_by_imp.loc[imp_id]
        hist_ids = hist_by_user.get(user_id, [])
        true_clicked = clicked_by_imp.loc[imp_id]

        query = build_query(hist_ids)
        retrieved = retrieve(query)
        hit = any(a in true_clicked for a in retrieved)

        before_metrics = {"auc": float("nan"), "mrr": 0.0, "ndcg5": float("nan"), "ndcg10": float("nan")}
        after_metrics = dict(before_metrics)
        if retrieved:
            before_labels = np.array([a in true_clicked for a in retrieved], dtype=bool)
            before_metrics = score_one_ranking(before_labels, f"{imp_id}:before")

            nrms_scores = scorer.score(list(hist_ids), retrieved)
            order = rank_candidates(nrms_scores, f"{imp_id}:after")
            after_labels = before_labels[order]
            after_metrics = {
                "auc": safe_auc(nrms_scores, before_labels) if len(retrieved) > 1 else float("nan"),
                "mrr": mrr(after_labels),
                "ndcg5": ndcg_at_k(after_labels, 5),
                "ndcg10": ndcg_at_k(after_labels, 10),
            }

        rows.append({
            "impression_id": str(imp_id), "user_id": str(user_id), "hit": int(hit),
            **{f"before_{k}": v for k, v in before_metrics.items()},
            **{f"after_{k}": v for k, v in after_metrics.items()},
        })
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(imp_ids)} impressions, {el:.0f}s elapsed, ETA {el/(i+1)*(len(imp_ids)-i-1):.0f}s",
                  flush=True)

    df = pd.DataFrame(rows)
    elapsed = time.time() - t0
    print(f"scored {len(df):,} impressions in {elapsed:.0f}s", flush=True)

    def ci_json(r: RankingMetricResult) -> dict:
        return {"value": r.metric, "ci_low": r.ci_low, "ci_high": r.ci_high,
                "n_impressions": r.n_impressions, "n_users": r.n_users, "n_skipped": r.n_skipped}

    out = {
        "config": {"mind_bundle": args.mind_bundle, "retriever": args.retriever, "k": args.k,
                   "checkpoint": str(args.checkpoint), "abstract_size": args.abstract_size,
                   "n_impressions_sampled": len(df), "seed": args.seed},
        "hit_rate": ci_json(ranking_metric_ci(df.assign(value=df["hit"].astype(float)), "value", seed=args.seed)),
        "before": {m: ci_json(ranking_metric_ci(df, f"before_{m}", seed=args.seed)) for m in ACCURACY},
        "after": {m: ci_json(ranking_metric_ci(df, f"after_{m}", seed=args.seed)) for m in ACCURACY},
        "paired_after_minus_before": {},
    }
    for m in ACCURACY:
        point, lo, hi = paired_metric_diff_ci(df, f"after_{m}", f"before_{m}", seed=args.seed)
        out["paired_after_minus_before"][m] = {"diff": point, "ci_low": lo, "ci_high": hi}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
