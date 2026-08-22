#!/usr/bin/env python3
"""Candidate D (2026-08-21 second session, see PROJECT_STATE.md's "Current
Objective"): does a purely symbolic category/subcategory/entity overlap
score beat the deployed MIND embedding baseline (AUC 0.6340, 95% CI
0.6319-0.6361) on MINDsmall-dev? Genuinely different mechanism from
Candidate A (dense TransE cosine similarity) and from BM25/embeddings
(both text-similarity methods) — this scores identity match only ("does
this candidate share a category/entity with the user's history"), never a
vector distance.

Per-impression, per-candidate score (untuned, unweighted sum of three
symbolic components, same "test whether it helps at all" framing as every
other untuned blend in this project):

    score = category_match_score + subcategory_match_score + log1p(entity_overlap_count)

`category_match_score`/`subcategory_match_score` are already in [0, 1]
(fraction of history sharing that category); `log1p(entity_overlap_count)`
dampens the unbounded raw entity-overlap count onto a roughly comparable
scale without a fitted normalization (fitting one would blur the line with
Candidate F's learned combiner, which is the deliberately-separate next
experiment for a *tuned* combination of exactly these signals).

Usage:
    poetry run python scripts/run_symbolic_overlap_experiment.py
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
from src.evaluation.ranking_metrics import (
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.features import (
    build_history_profile,
    build_lookup_tables,
    category_match_score,
    entity_overlap_count,
    subcategory_match_score,
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

    category_lookup, subcategory_lookup, entity_set_lookup = build_lookup_tables(articles)

    t0 = time.time()
    profile_by_user = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        profile_by_user[row.user_id] = build_history_profile(
            row.article_ids, category_lookup, subcategory_lookup, entity_set_lookup
        )
        history_len_by_user[row.user_id] = len(row.article_ids)
    build_s = time.time() - t0

    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in history_len_by_user.items()
    }

    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    t0 = time.time()
    rows: list[dict] = []
    empty_profile = build_history_profile([], {}, {}, {})
    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        profile = profile_by_user.get(user_id, empty_profile)
        cohort = cohort_by_user.get(user_id, "cold")

        scores = np.array([
            category_match_score(category_lookup.get(cid), profile)
            + subcategory_match_score(subcategory_lookup.get(cid), profile)
            + np.log1p(entity_overlap_count(entity_set_lookup.get(cid, frozenset()), profile))
            for cid in candidate_ids
        ])

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
        "method": "symbolic_overlap",
        "candidate": "D",
        "formula": "category_match_score + subcategory_match_score + log1p(entity_overlap_count)",
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(history),
        "n_warm_users": len(history) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "profile_build_seconds": round(build_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_d_symbolic_overlap_mind_small_{date.today().isoformat()}"
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
