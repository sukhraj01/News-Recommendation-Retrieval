#!/usr/bin/env python3
"""Candidate A (2026-08-21 local-validation session, see PROJECT_STATE.md's
"Current Objective"): does scoring MINDsmall-dev with per-article
knowledge-graph entity vectors (Wikidata `WikidataId` + MIND's own
`entity_embedding.vec`, confidence-weighted pooling — see
`src/retrieval/entities.py`) beat the deployed MiniLM-embedding baseline
(AUC 0.6340, 95% CI 0.6319-0.6361, reconfirmed this session by rerunning
`run_ranking_eval.py --dataset mind --method embed`)?

Article coverage (fraction of the corpus with >=1 resolvable entity vector)
is measured and reported *before* the ranking eval, per the objective's own
ordering — a candidate whose coverage is too low to matter shouldn't be
evaluated as if it were a full replacement.

Same lean metric set as `run_leakage_ablation.py` (AUC/MRR/nDCG@5/nDCG@10,
no diversity/novelty/coverage) — this is a cheap go/no-go screen, not a
full Q4 characterization; a candidate that clears this bar would get the
full harness before any real submission decision.

Usage:
    poetry run python scripts/run_entity_embedding_experiment.py
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
from src.retrieval.embed import build_user_embedding_query
from src.retrieval.entities import build_entity_index, load_entity_vectors
from src.retrieval.score import EmbeddingScorer
from src.utils.config import MIND_RAW_DIR
from src.utils.io import read_zip_member_bytes

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849  # embed method, reconfirmed 2026-08-21 (bit-identical to the 2026-08-10 run)


def run(bundle: str = "small") -> tuple[dict, dict]:
    paths = dataset_paths("mind", bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    entity_vec_bytes = read_zip_member_bytes(
        MIND_RAW_DIR / "MINDsmall_dev.zip", "MINDsmall_dev/entity_embedding.vec"
    )
    entity_vectors = load_entity_vectors(entity_vec_bytes)

    t0 = time.time()
    index, coverage = build_entity_index(articles, entity_vectors)
    build_s = time.time() - t0
    scorer = EmbeddingScorer(index)

    vector_lookup = dict(zip(index.article_ids, index.vectors))
    # Zero-filled (uncovered) rows are still valid dict entries here — an
    # all-zero vector is a legitimate (if uninformative) contribution to a
    # user's mean-pooled query, same as any other row; `build_entity_index`
    # already decided how uncovered *articles* are represented, this reuses
    # that decision unchanged rather than re-filtering it.
    query_by_user: dict[str, np.ndarray | None] = {}
    history_len_by_user: dict[str, int] = {}
    for row in history.itertuples(index=False):
        query_by_user[row.user_id] = build_user_embedding_query(row.article_ids, vector_lookup)
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
        "method": "entity_embed",
        "candidate": "A",
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_articles": len(articles),
        "article_entity_coverage": round(coverage, 4),
        "n_users": len(history),
        "n_warm_users": len(history) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "index_build_seconds": round(build_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_a_entity_embed_mind_small_{date.today().isoformat()}"
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
