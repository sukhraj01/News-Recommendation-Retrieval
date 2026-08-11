#!/usr/bin/env python3
"""BM25 retrieval benchmark: recall@K overall and by warm/cold-user cohort.

Per ADR-005/ADR-006: query = concatenated title+abstract of a user's click
history (BM25Okapi), retrieved once per user (history is static) against
that split's own article catalog, k in {50, 100, 200}. Cold-start threshold
(history length < 5) is grounded in EB-NeRD's own active-user filter — see
ADR-005 for why EB-NeRD's cold cohort is expected to be empty by construction.

Usage:
    poetry run python scripts/run_bm25_experiment.py --dataset mind
    poetry run python scripts/run_bm25_experiment.py --dataset ebnerd
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.metrics import recall_at_k
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.retrieve import retrieve_top_k
from src.utils.config import PROCESSED_DIR

COLD_THRESHOLD = 5  # history length < this = cold, per ADR-005
K_VALUES = [50, 100, 200]
MAX_K = max(K_VALUES)
N_BOOTSTRAP = 2000

# Default bundle per dataset — preserves prior CLI behavior when --bundle is
# omitted. ebnerd's "demo" (quick iteration) vs "small" (final training,
# ADR-002 addendum) are both valid bundles; MIND only has "small" in scope.
DEFAULT_BUNDLE = {"mind": "small", "ebnerd": "demo"}


def dataset_paths(dataset: str, bundle: str) -> dict:
    if dataset == "mind":
        base = PROCESSED_DIR / "mind" / bundle / "dev"
        return {
            "articles": base / "articles.parquet",
            "history": base / "user_history.parquet",
            "impressions": base / "impressions.parquet",
            "split_name": "dev",
            "corpus_split": "dev",
        }
    # ebnerd
    base = PROCESSED_DIR / "ebnerd" / bundle
    return {
        "articles": base / "articles.parquet",
        "history": base / "validation" / "user_history.parquet",
        "impressions": base / "validation" / "impressions.parquet",
        "split_name": "validation",
        "corpus_split": f"shared (ebnerd_{bundle} articles.parquet is not split-partitioned)",
    }


def run(dataset: str, bundle: str | None = None) -> tuple[dict, dict]:
    bundle = bundle or DEFAULT_BUNDLE[dataset]
    paths = dataset_paths(dataset, bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    t0 = time.time()
    index = build_index(articles)
    index_build_s = time.time() - t0

    text_lookup = dict(zip(
        articles["article_id"],
        articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
    ))

    t0 = time.time()
    top_k_by_user: dict[str, list[str]] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        query = build_user_query(row.article_ids, text_lookup)
        top_k_by_user[row.user_id] = retrieve_top_k(index, query, k=MAX_K)
        history_len_by_user[row.user_id] = len(row.article_ids)
    retrieval_s = time.time() - t0

    positives = impressions.loc[impressions["clicked"], ["user_id", "article_id"]].copy()
    positives["history_len"] = positives["user_id"].map(history_len_by_user).fillna(0).astype(int)
    positives["cohort"] = np.where(positives["history_len"] < COLD_THRESHOLD, "cold", "warm")

    k_results: dict[str, dict] = {}
    for k in K_VALUES:
        positives[f"hit@{k}"] = [
            article_id in top_k_by_user.get(user_id, [])[:k]
            for user_id, article_id in zip(positives["user_id"], positives["article_id"])
        ]
        slices = {"overall": positives, "warm": positives[positives["cohort"] == "warm"],
                  "cold": positives[positives["cohort"] == "cold"]}
        k_results[str(k)] = {
            name: asdict(recall_at_k(df.rename(columns={f"hit@{k}": "hit"}), n_bootstrap=N_BOOTSTRAP))
            for name, df in slices.items()
        }

    n_cold_users = sum(1 for v in history_len_by_user.values() if v < COLD_THRESHOLD)
    n_warm_users = len(history_len_by_user) - n_cold_users

    config = {
        "dataset": dataset,
        "bundle": bundle,
        "split": paths["split_name"],
        "corpus_split": paths["corpus_split"],
        "k_values": K_VALUES,
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "bm25_variant": "BM25Okapi (rank_bm25)",
        "query_construction": "unweighted concatenation of title+abstract over full click history",
        "corpus_size": len(articles),
        "n_users": len(history),
        "n_warm_users": n_warm_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(impressions),
        "n_positive_impressions": int(positives.shape[0]),
        "index_build_seconds": round(index_build_s, 2),
        "retrieval_seconds": round(retrieval_s, 2),
        "date": str(date.today()),
    }

    results = {"k": k_results}
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
                         help="Defaults to 'small' for mind, 'demo' for ebnerd. "
                              "Pass 'small' for ebnerd to use the final-training tier (ADR-002 addendum).")
    args = parser.parse_args()

    bundle = args.bundle or DEFAULT_BUNDLE[args.dataset]
    config, results = run(args.dataset, bundle)

    # Preserve prior folder naming when the default bundle is used
    # (bm25_mind_<date>, bm25_ebnerd_<date> stay stable/reproducible);
    # append the bundle name only when it diverges from that default, so
    # e.g. ebnerd_small doesn't silently overwrite the ebnerd_demo results.
    dataset_part = args.dataset if bundle == DEFAULT_BUNDLE[args.dataset] else f"{args.dataset}_{bundle}"
    out_dir = Path(__file__).resolve().parents[1] / "experiments" / f"bm25_{dataset_part}_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps(config, indent=2))
    for k, slices in results["k"].items():
        print(f"\nk={k}")
        for name, r in slices.items():
            print(f"  {name}: recall={r['recall']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                  f"n_positive={r['n_positive_impressions']}, n_users={r['n_users']}")
