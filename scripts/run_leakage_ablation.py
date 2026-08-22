#!/usr/bin/env python3
"""Q9 anti-gaming ablation: Q4 ranking metrics WITH vs WITHOUT the three
EB-NeRD article fields that are unavailable at serving time —
`total_inviews`/`total_pageviews`/`total_read_time` (raw ebnerd_small
articles.parquet; confirmed via grep that `src/datasets/ebnerd.py::parse_ebnerd_articles`
never reads them into the unified schema, so no production code path is
touched by this script — the leak is constructed here only, for measurement).

These three fields are dataset-lifetime aggregates (computed from click/view
activity across the whole corpus, including the future relative to any given
impression), so they are exactly the kind of feature Q9 requires reporting
with-and-without, not just documenting as excluded.

Method: `embed` only (the model actually used for both leaderboard
submissions, so this ablation answers "what would have happened to the
deployed model"). WITHOUT is the unmodified `EmbeddingScorer` score.  WITH
adds a per-impression min-max-normalized "leak score" (mean of min-max-
normalized log1p(total_inviews), log1p(total_pageviews),
log1p(total_read_time); missing values imputed with the corpus median, see
`_build_leak_lookup`) to a per-impression min-max-normalized base score, at
equal (0.5/0.5) weight — an unweighted, untuned blend, deliberately: the
question is whether *any* exposure to these fields moves the metrics, not
what an optimally-tuned leaky model would score.

Usage:
    poetry run python scripts/run_leakage_ablation.py --dataset ebnerd --bundle small
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

from run_bm25_experiment import dataset_paths
from run_ranking_eval import _build_method
from src.evaluation.ranking_metrics import (
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.utils.config import EBNERD_RAW_DIR
from src.utils.ids import prefix_id
from src.utils.io import read_zip_parquet

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
LEAK_FIELDS = ["total_inviews", "total_pageviews", "total_read_time"]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.min(x), np.max(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _compute_leak_scores(raw: pd.DataFrame) -> np.ndarray:
    """One leak score per row of `raw` (must have `LEAK_FIELDS` columns):
    mean of per-field min-max-normalized log1p values, missing values
    imputed with that field's own corpus median. Pure function, no I/O —
    the part of the ablation actually worth unit-testing in isolation
    (see `tests/unit/test_leakage_ablation.py`)."""
    components = []
    for field in LEAK_FIELDS:
        vals = raw[field].to_numpy(dtype=np.float64)
        median = np.nanmedian(vals)
        vals = np.where(np.isnan(vals), median, vals)
        components.append(_minmax(np.log1p(vals)))
    return np.mean(components, axis=0)


def _build_leak_lookup(bundle: str) -> dict[str, float]:
    """article_id -> single leak score, built from the three raw fields
    Q9 names, never touching `src/datasets/ebnerd.py`'s production parse."""
    zip_path = EBNERD_RAW_DIR / f"ebnerd_{bundle}.zip"
    raw = read_zip_parquet(zip_path, "articles.parquet", columns=["article_id"] + LEAK_FIELDS)

    leak_score = _compute_leak_scores(raw)
    article_ids = [prefix_id("ebnerd", i) for i in raw["article_id"]]
    return dict(zip(article_ids, leak_score))


def run(dataset: str, bundle: str) -> tuple[dict, dict]:
    paths = dataset_paths(dataset, bundle)
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    scorer, query_by_user, _, _ = _build_method("embed", articles, history, paths)
    leak_lookup = _build_leak_lookup(bundle)
    leak_coverage = sum(1 for a in articles["article_id"] if a in leak_lookup) / len(articles)

    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")

    t0 = time.time()
    rows_without: list[dict] = []
    rows_with: list[dict] = []

    for (user_id, impression_id), group in impressions_sorted.groupby(
        ["user_id", "impression_id"], sort=False
    ):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        query = query_by_user.get(user_id, None)

        base_scores = scorer.score(query, candidate_ids)
        leak_scores = np.array([leak_lookup.get(c, 0.0) for c in candidate_ids])

        base_norm = _minmax(base_scores)
        leak_norm = _minmax(leak_scores)
        blended = 0.5 * base_norm + 0.5 * leak_norm

        for rows, scores in ((rows_without, base_scores), (rows_with, blended)):
            order = rank_candidates(scores, impression_id, seed=SEED)
            ranked_clicked = clicked[order]
            rows.append({
                "impression_id": impression_id,
                "user_id": user_id,
                "auc": safe_auc(scores, clicked),
                "mrr": mrr(ranked_clicked),
                "ndcg5": ndcg_at_k(ranked_clicked, 5),
                "ndcg10": ndcg_at_k(ranked_clicked, 10),
            })

    eval_s = time.time() - t0

    def _summarize(rows: list[dict]) -> dict:
        df = pd.DataFrame(rows)
        return {
            metric: asdict(ranking_metric_ci(df, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))
            for metric in METRIC_COLUMNS
        }

    config = {
        "dataset": dataset,
        "bundle": bundle,
        "method": "embed",
        "leak_fields": LEAK_FIELDS,
        "leak_field_coverage_in_corpus": round(leak_coverage, 4),
        "blend_weight_base": 0.5,
        "blend_weight_leak": 0.5,
        "n_bootstrap": N_BOOTSTRAP,
        "n_impressions": len(rows_without),
        "n_users": impressions["user_id"].nunique(),
        "eval_seconds": round(eval_s, 2),
        "date": str(date.today()),
    }
    results = {"without_leaky_features": _summarize(rows_without), "with_leaky_features": _summarize(rows_with)}
    return config, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ebnerd"], default="ebnerd",
                         help="Only EB-NeRD carries these three article fields.")
    parser.add_argument("--bundle", default="small")
    args = parser.parse_args()

    config, results = run(args.dataset, args.bundle)

    out_dir = _REPO_ROOT / "experiments" / f"ablation_leaky_features_{args.dataset}_{args.bundle}_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps(config, indent=2))
    for arm in ("without_leaky_features", "with_leaky_features"):
        print(f"\n{arm}")
        for metric, r in results[arm].items():
            print(f"  {metric}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                  f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")
