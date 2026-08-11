#!/usr/bin/env python3
"""Q4 ranking evaluation harness: AUC, MRR, nDCG@5, nDCG@10 (Q4.1),
diversity/novelty/coverage (Q4.2), warm/cold slicing (Q4.3, reusing
ADR-005's `<5` history-length threshold), bootstrap 95% CI (Q4.4).

Structurally different question from `run_bm25_experiment.py`'s recall@K:
that script asks whether the true clicked article is anywhere in a top-K
retrieved from the WHOLE corpus. This script scores and ranks the
candidates ALREADY LISTED in each impression, then evaluates that ranking
against the click labels — see ADR-007 for the tie-breaking rule, the
K=10 diversity/novelty cutoff, and why coverage has no bootstrap CI.

Generic over scoring method: `--method bm25` is implemented here via
`src.retrieval.score.BM25Scorer`; `--method embed` (Phase 4, ADR-008) via
`src.retrieval.score.EmbeddingScorer`. The core loop below (steps 6-8)
never references either method by name outside `_build_method`, which is
the seam Phase 4 plugs into, exactly as this docstring originally promised.

Usage:
    poetry run python scripts/run_ranking_eval.py --dataset mind
    poetry run python scripts/run_ranking_eval.py --dataset ebnerd --bundle small
    poetry run python scripts/run_ranking_eval.py --dataset mind --method embed
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import numpy as np
import pandas as pd

from run_bm25_experiment import COLD_THRESHOLD, DEFAULT_BUNDLE, dataset_paths
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.embed import (
    DEFAULT_MODEL,
    build_embedding_index,
    build_user_embedding_query,
    model_slug,
)
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import BM25Scorer, EmbeddingScorer
from src.utils.config import PROCESSED_DIR

K_DIVERSITY_NOVELTY = 10  # anchored to nDCG@10's cutoff, per ADR-007
N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10", "diversity10", "novelty10"]


def _build_method(method: str, articles: pd.DataFrame, history: pd.DataFrame, paths: dict):
    """Returns `(scorer, query_by_user, history_len_by_user, index_build_seconds)`.

    The only place `run()` knows which retrieval method it's driving —
    everything downstream (the per-impression scoring loop, metrics) is
    written against the `Scorer` Protocol only, per ADR-007's design intent.
    """
    if method == "bm25":
        t0 = time.time()
        index = build_index(articles)
        scorer = BM25Scorer(index)
        build_s = time.time() - t0

        text_lookup = dict(zip(
            articles["article_id"],
            articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
        ))
        query_by_user: dict[str, list[str]] = {}
        history_len_by_user: dict[str, int] = {}
        for row in history.itertuples(index=False):
            query_by_user[row.user_id] = build_user_query(row.article_ids, text_lookup)
            history_len_by_user[row.user_id] = len(row.article_ids)
        return scorer, query_by_user, history_len_by_user, build_s

    if method == "embed":
        cache_path = paths["articles"].parent / "embeddings" / f"{model_slug(DEFAULT_MODEL)}.npy"
        t0 = time.time()
        index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
        scorer = EmbeddingScorer(index)
        build_s = time.time() - t0

        vector_lookup = dict(zip(index.article_ids, index.vectors))
        query_by_user: dict[str, np.ndarray | None] = {}
        history_len_by_user: dict[str, int] = {}
        for row in history.itertuples(index=False):
            query_by_user[row.user_id] = build_user_embedding_query(row.article_ids, vector_lookup)
            history_len_by_user[row.user_id] = len(row.article_ids)
        return scorer, query_by_user, history_len_by_user, build_s

    raise ValueError(f"Unknown method {method!r}")


def _train_impressions_path(dataset: str, bundle: str) -> Path:
    return PROCESSED_DIR / dataset / bundle / "train" / "impressions.parquet"


def run(dataset: str, bundle: str | None = None, method: str = "bm25") -> tuple[dict, dict]:
    if method not in ("bm25", "embed"):
        raise ValueError(f"Unknown method {method!r}")

    bundle = bundle or DEFAULT_BUNDLE[dataset]
    paths = dataset_paths(dataset, bundle)

    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])
    train_impressions = pd.read_parquet(_train_impressions_path(dataset, bundle))

    # Precomputed once per user (not lazily) so the SAME query object is
    # reused across that user's impressions — both scorers' identity caches
    # only hit when the query object is literally the same reference.
    scorer, query_by_user, history_len_by_user, index_build_s = _build_method(
        method, articles, history, paths
    )
    empty_query = [] if method == "bm25" else None

    category_lookup = dict(zip(articles["article_id"], articles["category"]))
    popularity = build_train_popularity(train_impressions, n_catalog=len(articles), alpha=1.0)

    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in history_len_by_user.items()
    }

    # Sort by user_id so a user's impressions are processed consecutively
    # (makes the scorer's per-user cache an actual hit, not just a
    # speculative optimization that never fires).
    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    t0 = time.time()
    rows: list[dict] = []
    top_k_ids_by_cohort: dict[str, list[list[str]]] = {"overall": [], "warm": [], "cold": []}

    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        query = query_by_user.get(user_id, empty_query)
        cohort = cohort_by_user.get(user_id, "cold")

        scores = scorer.score(query, candidate_ids)
        order = rank_candidates(scores, impression_id, seed=SEED)
        ranked_ids = [candidate_ids[i] for i in order]
        ranked_clicked = clicked[order]

        k_eff = min(K_DIVERSITY_NOVELTY, len(ranked_ids))
        top_ids = ranked_ids[:k_eff]
        top_categories = [category_lookup.get(a) for a in top_ids]

        rows.append({
            "impression_id": impression_id,
            "user_id": user_id,
            "cohort": cohort,
            "auc": safe_auc(scores, clicked),
            "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5),
            "ndcg10": ndcg_at_k(ranked_clicked, 10),
            "diversity10": intra_list_diversity(top_categories),
            "novelty10": novelty(top_ids, popularity),
        })
        top_k_ids_by_cohort["overall"].append(top_ids)
        top_k_ids_by_cohort[cohort].append(top_ids)

    eval_s = time.time() - t0

    per_impression = pd.DataFrame(rows)
    catalog_size = len(articles)

    metric_results: dict[str, dict] = {}
    for metric in METRIC_COLUMNS:
        slices = {
            "overall": per_impression,
            "warm": per_impression[per_impression["cohort"] == "warm"],
            "cold": per_impression[per_impression["cohort"] == "cold"],
        }
        metric_results[metric] = {
            name: asdict(ranking_metric_ci(df, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))
            for name, df in slices.items()
        }

    coverage_results = {
        name: (coverage(ids, catalog_size) if ids else float("nan"))
        for name, ids in top_k_ids_by_cohort.items()
    }

    n_cold_users = sum(1 for c in cohort_by_user.values() if c == "cold")
    n_warm_users = len(cohort_by_user) - n_cold_users

    config = {
        "dataset": dataset,
        "bundle": bundle,
        "method": method,
        "split": paths["split_name"],
        "corpus_split": paths["corpus_split"],
        "cold_threshold": COLD_THRESHOLD,
        "diversity_novelty_k": K_DIVERSITY_NOVELTY,
        "n_bootstrap": N_BOOTSTRAP,
        "corpus_size": catalog_size,
        "n_users": len(history),
        "n_warm_users": n_warm_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "index_build_seconds": round(index_build_s, 2),
        "eval_seconds": round(eval_s, 2),
        "date": str(date.today()),
    }

    results = {"metrics": metric_results, "coverage": coverage_results}
    if n_cold_users == 0:
        results["notes"] = [
            f"Zero cold-start users (history length < {COLD_THRESHOLD}) in this split by "
            "construction — see ADR-005. The 'cold' slice below is undefined (NaN), not a "
            "measurement of zero."
        ]

    return config, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mind", "ebnerd"], required=True)
    parser.add_argument("--bundle", default=None,
                         help="Defaults to 'small' for mind, 'demo' for ebnerd.")
    parser.add_argument("--method", choices=["bm25", "embed"], default="bm25")
    args = parser.parse_args()

    bundle = args.bundle or DEFAULT_BUNDLE[args.dataset]
    config, results = run(args.dataset, bundle, args.method)

    dataset_part = args.dataset if bundle == DEFAULT_BUNDLE[args.dataset] else f"{args.dataset}_{bundle}"
    out_dir = _REPO_ROOT / "experiments" / f"ranking_{args.method}_{dataset_part}_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps(config, indent=2))
    for metric, slices in results["metrics"].items():
        print(f"\n{metric}")
        for name, r in slices.items():
            print(f"  {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                  f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")
    print("\ncoverage")
    for name, c in results["coverage"].items():
        print(f"  {name}: {c:.4f}")
