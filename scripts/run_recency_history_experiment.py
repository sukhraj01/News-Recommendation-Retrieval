#!/usr/bin/env python3
"""Candidate C (2026-08-21 local-validation session, see PROJECT_STATE.md's
"Current Objective"): re-tests ADR-005's rejected recency-weighting option
against MIND's embedding-based user representation (Q3's mean-pooled
history query), not BM25's — the deployed method is embeddings (ADR-008),
so this is the version of the question that actually matters for a real
submission decision. ADR-005 rejected recency weighting for *BM25 query
construction* before benchmarking, purely on the unverified-history-order
risk; this experiment measures it directly instead of continuing to reject
it on the same un-benchmarked grounds, per CLAUDE.md's decision-reversal
guidance — the underlying assumption risk (MIND's history order is
"presumed chronological, not independently verified") still applies and is
carried in every output this script produces, not resolved by running it.

Uses `build_user_embedding_query_recency` (`src/retrieval/embed.py`,
decay=0.9, untuned) in place of the deployed `build_user_embedding_query`;
everything else (index, scorer, eval loop) is identical to the baseline
embed run, so this is a clean paired comparison against the same
MINDsmall-dev baseline (AUC 0.6340, 95% CI 0.6319-0.6361).

Usage:
    poetry run python scripts/run_recency_history_experiment.py
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
from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, build_user_embedding_query_recency
from src.retrieval.score import EmbeddingScorer

N_BOOTSTRAP = 2000
SEED = 0
DECAY = 0.9  # untuned starting point, see module docstring
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849  # embed method, unweighted mean-pool, reconfirmed 2026-08-21


def run(bundle: str = "small") -> tuple[dict, dict]:
    paths = dataset_paths("mind", bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    cache_path = paths["articles"].parent / "embeddings" / f"{DEFAULT_MODEL.replace('/', '__')}.npy"
    t0 = time.time()
    index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
    build_s = time.time() - t0
    scorer = EmbeddingScorer(index)

    vector_lookup = dict(zip(index.article_ids, index.vectors))
    query_by_user: dict[str, np.ndarray | None] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        query_by_user[row.user_id] = build_user_embedding_query_recency(
            row.article_ids, vector_lookup, decay=DECAY
        )
        history_len_by_user[row.user_id] = len(row.article_ids)

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
        query = query_by_user.get(user_id, None)
        cohort = cohort_by_user.get(user_id, "cold")

        scores = scorer.score(query, candidate_ids)
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
        "method": "embed_recency",
        "candidate": "C",
        "decay": DECAY,
        "assumption": (
            "MIND user_history.article_ids is ordered oldest-to-most-recent "
            "(article_ids[-1] = most recent click); unverified against real "
            "timestamps, per ADR-005 — inherited, not resolved, by this experiment"
        ),
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(history),
        "n_warm_users": len(history) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "index_build_seconds": round(build_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed_unweighted": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_c_recency_history_mind_small_{date.today().isoformat()}"
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
