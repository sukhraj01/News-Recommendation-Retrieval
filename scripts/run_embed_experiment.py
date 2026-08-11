#!/usr/bin/env python3
"""Embedding retrieval benchmark: recall@K overall and by warm/cold-user
cohort — the Q3/Q3.4 counterpart to `run_bm25_experiment.py`, structured
identically (same recall@K + bootstrap CI machinery, same warm/cold cohort,
same experiment-artifact naming convention) so the two are directly
comparable per ADR-008/Q3.5.

Per ADR-008: query = mean-pooled, L2-renormalized embedding of a user's
click-history articles (paraphrase-multilingual-MiniLM-L12-v2), retrieved
via brute-force cosine similarity against that split's own article catalog
(mirrors ADR-005's "each split's own catalog is the correct index universe"
decision, reused unchanged for embeddings). Embeddings are disk-cached per
(dataset, bundle, model) under `data/processed/.../embeddings/` — see
`embed.py`'s module docstring for why this deviates from BM25's
rebuild-every-run precedent.

Usage:
    poetry run python scripts/run_embed_experiment.py --dataset mind
    poetry run python scripts/run_embed_experiment.py --dataset ebnerd --bundle small
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from run_bm25_experiment import COLD_THRESHOLD, DEFAULT_BUNDLE, K_VALUES, MAX_K, dataset_paths
from src.evaluation.metrics import recall_at_k
from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, build_user_embedding_query, model_slug
from src.retrieval.retrieve import embed_retrieve_top_k
from src.utils.config import PROCESSED_DIR

N_BOOTSTRAP = 2000


def run(dataset: str, bundle: str | None = None, model_name: str = DEFAULT_MODEL) -> tuple[dict, dict]:
    bundle = bundle or DEFAULT_BUNDLE[dataset]
    paths = dataset_paths(dataset, bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    cache_path = paths["articles"].parent / "embeddings" / f"{model_slug(model_name)}.npy"

    t0 = time.time()
    index = build_embedding_index(articles, model_name=model_name, cache_path=cache_path)
    embedding_s = time.time() - t0

    vector_lookup = dict(zip(index.article_ids, index.vectors))

    t0 = time.time()
    top_k_by_user: dict[str, list[str]] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        query = build_user_embedding_query(row.article_ids, vector_lookup)
        top_k_by_user[row.user_id] = embed_retrieve_top_k(index, query, k=MAX_K)
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
        "embedding_model": model_name,
        "ann_backend": "brute-force cosine similarity (vectorized matrix product) — ADR-008",
        "user_representation": "mean-pooled, L2-renormalized embedding of full click history (Q3)",
        "corpus_size": len(articles),
        "n_users": len(history),
        "n_warm_users": n_warm_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(impressions),
        "n_positive_impressions": int(positives.shape[0]),
        "embedding_seconds": round(embedding_s, 2),
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
                         help="Defaults to 'small' for mind, 'demo' for ebnerd.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    bundle = args.bundle or DEFAULT_BUNDLE[args.dataset]
    config, results = run(args.dataset, bundle, args.model)

    dataset_part = args.dataset if bundle == DEFAULT_BUNDLE[args.dataset] else f"{args.dataset}_{bundle}"
    out_dir = Path(__file__).resolve().parents[1] / "experiments" / f"embed_{dataset_part}_{date.today().isoformat()}"
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
