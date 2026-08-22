#!/usr/bin/env python3
"""Candidate E (2026-08-21 second session, see PROJECT_STATE.md's "Current
Objective"): does ranking candidates purely by train-split article
popularity beat the deployed MIND embedding baseline (AUC 0.6340, 95% CI
0.6319-0.6361) on MINDsmall-dev? A genuinely different mechanism from every
other candidate tried so far — no user history, no query, no per-user
signal at all; every user sees the same candidate ranked the same way
within a given impression's candidate set.

Reuses `ranking_metrics.py::build_train_popularity` unchanged (already
built for Q9's novelty metric, per ADR-007/ADR-009's anti-gaming
requirement: popularity computed from the TRAIN split only, Laplace-
smoothed, never from dev/validation — using dev-split popularity here
would be circular, scoring dev impressions by information only available
after seeing dev's own clicks).

Usage:
    poetry run python scripts/run_popularity_experiment.py
"""
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

from run_bm25_experiment import COLD_THRESHOLD, dataset_paths
from run_ranking_eval import _train_impressions_path
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849


def run(bundle: str = "small") -> tuple[dict, dict]:
    paths = dataset_paths("mind", bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])
    train_impressions = pd.read_parquet(_train_impressions_path("mind", bundle))

    popularity = build_train_popularity(train_impressions, n_catalog=len(articles), alpha=1.0)

    history_len_by_user = {row.user_id: len(row.article_ids) for row in history.itertuples(index=False)}
    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in history_len_by_user.items()
    }

    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    t0 = time.time()
    rows: list[dict] = []
    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        cohort = cohort_by_user.get(user_id, "cold")

        scores = np.array([popularity[cid] for cid in candidate_ids])

        order = rank_candidates(scores, impression_id, seed=SEED)
        ranked_clicked = clicked[order]

        rows.append({
            "impression_id": impression_id,
            "user_id": user_id,
            "cohort": cohort,
            "auc": safe_auc(scores, clicked),
            "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5),
            "ndcg10": ndcg_at_k(ranked_clicked, 10),
        })
    eval_s = time.time() - t0

    per_impression = pd.DataFrame(rows)

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

    n_cold_users = sum(1 for c in cohort_by_user.values() if c == "cold")
    config = {
        "dataset": "mind",
        "bundle": bundle,
        "method": "train_popularity",
        "candidate": "E",
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(history),
        "n_warm_users": len(history) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_e_popularity_mind_small_{date.today().isoformat()}"
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
