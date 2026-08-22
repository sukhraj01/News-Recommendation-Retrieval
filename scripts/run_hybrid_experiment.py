#!/usr/bin/env python3
"""Candidate B (2026-08-21 local-validation session, see PROJECT_STATE.md's
"Current Objective"): does blending BM25 and MiniLM-embedding scores beat
the deployed embedding-only baseline (AUC 0.6340, 95% CI 0.6319-0.6361) on
MINDsmall-dev?

Same untuned-blend design as `run_leakage_ablation.py` (ADR-009): both raw
scores are min-max normalized *within each impression* (so the blend isn't
dominated by whichever method's raw scale happens to be larger — BM25's
scores and cosine similarities live on unrelated scales), then combined at
a flat, unsearched 50/50 weight. The question is whether blending helps at
all, not what an optimally-tuned weight would score — searching the weight
would answer a different, larger question this cheap screen isn't sized
for. Reuses `_minmax` from `run_leakage_ablation.py` rather than
duplicating it (same function, same contract).

Runs all three arms (bm25-only, embed-only, hybrid) in one pass so they
share the exact same per-impression grouping/tie-break/bootstrap seed —
a fair paired comparison, not three separately-run experiments compared
after the fact.

Usage:
    poetry run python scripts/run_hybrid_experiment.py
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
from run_leakage_ablation import _minmax
from run_ranking_eval import _build_method
from src.evaluation.ranking_metrics import (
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BLEND_WEIGHT_BM25 = 0.5
BLEND_WEIGHT_EMBED = 0.5
ARMS = ("bm25", "embed", "hybrid")


def run(bundle: str = "small") -> tuple[dict, dict]:
    paths = dataset_paths("mind", bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    bm25_scorer, bm25_query_by_user, history_len_by_user, bm25_build_s = _build_method(
        "bm25", articles, history, paths
    )
    embed_scorer, embed_query_by_user, _, embed_build_s = _build_method(
        "embed", articles, history, paths
    )

    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in history_len_by_user.items()
    }

    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    t0 = time.time()
    rows_by_arm: dict[str, list[dict]] = {arm: [] for arm in ARMS}

    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        cohort = cohort_by_user.get(user_id, "cold")

        bm25_query = bm25_query_by_user.get(user_id, [])
        embed_query = embed_query_by_user.get(user_id, None)

        bm25_scores = bm25_scorer.score(bm25_query, candidate_ids)
        embed_scores = embed_scorer.score(embed_query, candidate_ids)
        hybrid_scores = (
            BLEND_WEIGHT_BM25 * _minmax(bm25_scores) + BLEND_WEIGHT_EMBED * _minmax(embed_scores)
        )

        for arm, scores in (("bm25", bm25_scores), ("embed", embed_scores), ("hybrid", hybrid_scores)):
            order = rank_candidates(scores, impression_id, seed=SEED)
            ranked_clicked = clicked[order]
            rows_by_arm[arm].append({
                "impression_id": impression_id,
                "user_id": user_id,
                "cohort": cohort,
                "auc": safe_auc(scores, clicked),
                "mrr": mrr(ranked_clicked),
                "ndcg5": ndcg_at_k(ranked_clicked, 5),
                "ndcg10": ndcg_at_k(ranked_clicked, 10),
            })
    eval_s = time.time() - t0

    def _summarize(rows: list[dict]) -> dict:
        df = pd.DataFrame(rows)
        out = {}
        for metric in METRIC_COLUMNS:
            slices = {
                "overall": df,
                "warm": df[df["cohort"] == "warm"],
                "cold": df[df["cohort"] == "cold"],
            }
            out[metric] = {
                name: asdict(ranking_metric_ci(s, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))
                for name, s in slices.items()
            }
        return out

    n_cold_users = sum(1 for c in cohort_by_user.values() if c == "cold")
    config = {
        "dataset": "mind",
        "bundle": bundle,
        "candidate": "B",
        "arms": list(ARMS),
        "blend_weight_bm25": BLEND_WEIGHT_BM25,
        "blend_weight_embed": BLEND_WEIGHT_EMBED,
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(history),
        "n_warm_users": len(history) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(rows_by_arm["hybrid"]),
        "bm25_index_build_seconds": round(bm25_build_s, 2),
        "embed_index_build_seconds": round(embed_build_s, 2),
        "eval_seconds": round(eval_s, 2),
        "date": str(date.today()),
    }
    results = {arm: _summarize(rows_by_arm[arm]) for arm in ARMS}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_b_hybrid_mind_small_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps(config, indent=2))
    for arm in ARMS:
        print(f"\n=== {arm} ===")
        for metric, slices in results[arm].items():
            print(f"  {metric}")
            for name, r in slices.items():
                print(f"    {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                      f"n_impressions={r['n_impressions']}, n_users={r['n_users']}")
