#!/usr/bin/env python3
"""Candidate L (A2 Q1): does engineered click-history feature engineering,
scored by a cheap CPU-only logistic-regression combiner, beat the deployed
MIND embedding baseline (AUC 0.6340, 95% CI 0.6319-0.6361)?

Scope, deliberately narrow: this is A2 Q1's asked-for feature set — click
count, category-match, recency-weighted history (exponential decay) — and
nothing else. Unlike Candidate F (`run_learned_combiner_experiment.py`),
this candidate does NOT include BM25/embedding retrieval scores or
popularity as combiner inputs; the question is specifically whether
click-history statistics alone carry ranking signal, not whether stacking
everything wins (F already answered that: yes, marginally). The embedding
baseline is used only as the comparison target, scored separately to
produce a genuine PAIRED test (`paired_metric_diff_ci`, imported from
`run_gated_cohort_experiment.py`, not reimplemented) — not as a Candidate L
input feature.

Five features (`src/retrieval/mind_features.py::FEATURE_NAMES`):
`log_click_count`, `category_match`, `subcategory_match`,
`recency_category_affinity`, `recency_subcategory_affinity`. The first two
symbolic-match features are `features.py`'s pre-existing unweighted scores
(Candidate D); the two recency features and click count are new (A2 Q1).

A documented, verified modeling fact, not a bug: `log_click_count` is
identical for every candidate within one impression (it depends only on
the user). A linear combiner's contribution to a per-impression RANKING
metric (AUC/MRR/nDCG@K — all rank-order-only) is provably invariant to an
additive per-impression-constant term. This script verifies that claim
empirically (an ablation arm with the feature zeroed out) rather than
asserting it from theory alone — see `_read_zero_variance_and_ablation`.

Dataset limitation, stated rather than worked around: no freshness feature
is built (MIND has no publish timestamps) and no session feature is built
(MIND has no session boundaries) — both confirmed null for every row by
direct measurement (`src/datasets/mind.py` hardcodes them). EB-NeRD already
has both (ADR-013).

Leakage: reuses the SAME per-user history snapshot every other MIND
candidate in this project reads from `user_history.parquet`
(structurally leak-safe by construction, per ADR-002/ADR-005) and adds a
feature-level regression test
(`tests/integration/test_leakage.py::test_mind_click_history_features_trace_to_one_shared_value_across_real_impressions`).

Usage:
    poetry run python scripts/run_mind_history_features_experiment.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_bm25_experiment import COLD_THRESHOLD, dataset_paths
from run_gated_cohort_experiment import paired_metric_diff_ci
from run_ranking_eval import _train_impressions_path
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, build_user_embedding_query, model_slug
from src.retrieval.features import HistoryProfile, build_history_profile, build_lookup_tables
from src.retrieval.mind_features import FEATURE_NAMES, build_feature_row
from src.retrieval.score import EmbeddingScorer

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849  # ADR-008, MINDsmall-dev embedding baseline


def _train_paths(bundle: str) -> dict:
    from src.utils.config import PROCESSED_DIR

    base = PROCESSED_DIR / "mind" / bundle / "train"
    return {"articles": base / "articles.parquet", "history": base / "user_history.parquet",
            "impressions": base / "impressions.parquet"}


def _build_side(paths: dict) -> dict:
    """Per-split: the embedding baseline's scorer (for the paired comparison
    only, NOT a Candidate L feature), per-user history profiles for the
    symbolic-match features, and the raw article_ids lists the recency
    features need (a `HistoryProfile`'s counters discard position order)."""
    articles = pd.read_parquet(paths["articles"])
    history = pd.read_parquet(paths["history"])
    impressions = pd.read_parquet(paths["impressions"])

    cache_path = paths["articles"].parent / "embeddings" / f"{model_slug(DEFAULT_MODEL)}.npy"
    embed_index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
    embed_scorer = EmbeddingScorer(embed_index)
    vector_lookup = dict(zip(embed_index.article_ids, embed_index.vectors))
    category_lookup, subcategory_lookup, entity_set_lookup = build_lookup_tables(articles)

    embed_query_by_user, profile_by_user, article_ids_by_user, history_len_by_user = {}, {}, {}, {}
    for row in history.itertuples(index=False):
        embed_query_by_user[row.user_id] = build_user_embedding_query(row.article_ids, vector_lookup)
        profile_by_user[row.user_id] = build_history_profile(
            row.article_ids, category_lookup, subcategory_lookup, entity_set_lookup
        )
        article_ids_by_user[row.user_id] = list(row.article_ids)
        history_len_by_user[row.user_id] = len(row.article_ids)

    return {
        "impressions": impressions, "embed_scorer": embed_scorer, "embed_query_by_user": embed_query_by_user,
        "category_lookup": category_lookup, "subcategory_lookup": subcategory_lookup,
        "profile_by_user": profile_by_user, "article_ids_by_user": article_ids_by_user,
        "history_len_by_user": history_len_by_user, "n_articles": len(articles),
    }


def _feature_matrix_for_group(
    candidate_ids: list[str], article_ids: list[str], profile: HistoryProfile, side: dict,
) -> np.ndarray:
    return np.array([
        build_feature_row(c, article_ids, side["category_lookup"], side["subcategory_lookup"], profile)
        for c in candidate_ids
    ], dtype=np.float64)


def _build_training_matrix(side: dict) -> tuple[np.ndarray, np.ndarray]:
    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    n = len(impressions_sorted)
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float64)
    y = np.empty(n, dtype=bool)
    empty_profile = HistoryProfile()
    pos = 0
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        k = len(candidate_ids)
        clicked = group["clicked"].to_numpy(dtype=bool)
        article_ids = side["article_ids_by_user"].get(user_id, [])
        profile = side["profile_by_user"].get(user_id, empty_profile)
        X[pos:pos + k] = _feature_matrix_for_group(candidate_ids, article_ids, profile, side)
        y[pos:pos + k] = clicked
        pos += k
    assert pos == n
    return X, y


def _evaluate(side: dict, coef: np.ndarray, intercept: float, scaler_mean, scaler_scale,
             zero_out_feature: str | None = None) -> pd.DataFrame:
    """Scores Candidate L AND the embedding baseline on the SAME impressions,
    so the two columns can be paired directly. `zero_out_feature`, when set,
    zeroes that column of the SCALED feature matrix before scoring — used
    once, standalone, to verify `log_click_count`'s documented zero-ranking-
    contribution claim empirically."""
    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    cohort_by_user = {
        uid: ("cold" if hlen < COLD_THRESHOLD else "warm")
        for uid, hlen in side["history_len_by_user"].items()
    }
    empty_profile = HistoryProfile()
    zero_idx = FEATURE_NAMES.index(zero_out_feature) if zero_out_feature else None
    rows = []
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        cohort = cohort_by_user.get(user_id, "cold")
        article_ids = side["article_ids_by_user"].get(user_id, [])
        profile = side["profile_by_user"].get(user_id, empty_profile)

        X = _feature_matrix_for_group(candidate_ids, article_ids, profile, side)
        X_scaled = (X - scaler_mean) / scaler_scale
        if zero_idx is not None:
            X_scaled[:, zero_idx] = 0.0
        scores_l = X_scaled @ coef + intercept

        embed_query = side["embed_query_by_user"].get(user_id, None)
        scores_baseline = side["embed_scorer"].score(embed_query, candidate_ids)

        order = rank_candidates(scores_l, impression_id, seed=SEED)
        ranked_clicked = clicked[order]
        order_b = rank_candidates(scores_baseline, impression_id, seed=SEED)
        ranked_clicked_b = clicked[order_b]

        rows.append({
            "impression_id": impression_id, "user_id": user_id, "cohort": cohort,
            "auc": safe_auc(scores_l, clicked), "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5), "ndcg10": ndcg_at_k(ranked_clicked, 10),
            "auc_baseline": safe_auc(scores_baseline, clicked), "mrr_baseline": mrr(ranked_clicked_b),
            "ndcg5_baseline": ndcg_at_k(ranked_clicked_b, 5), "ndcg10_baseline": ndcg_at_k(ranked_clicked_b, 10),
        })
    return pd.DataFrame(rows)


def run(bundle: str = "small") -> tuple[dict, dict]:
    dev_paths = dataset_paths("mind", bundle)
    train_paths = _train_paths(bundle)

    t0 = time.time()
    train_side = _build_side(train_paths)
    dev_side = _build_side(dev_paths)
    build_s = time.time() - t0

    t0 = time.time()
    X_train, y_train = _build_training_matrix(train_side)
    feature_build_s = time.time() - t0

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=0)
    t0 = time.time()
    model.fit(X_train_scaled, y_train)
    fit_s = time.time() - t0
    coef, intercept = model.coef_[0], float(model.intercept_[0])

    t0 = time.time()
    per_impression = _evaluate(dev_side, coef, intercept, scaler.mean_, scaler.scale_)
    eval_s = time.time() - t0

    # --- the empirical check for log_click_count's zero-ranking-contribution claim ---
    ablated = _evaluate(dev_side, coef, intercept, scaler.mean_, scaler.scale_,
                        zero_out_feature="log_click_count")
    click_count_ranking_delta = float(
        (per_impression["auc"].dropna() - ablated["auc"].dropna()).abs().max()
    )

    metric_results = {m: asdict(ranking_metric_ci(per_impression, m)) for m in METRIC_COLUMNS}
    n_cold_users = sum(1 for v in dev_side["history_len_by_user"].values() if v < COLD_THRESHOLD)

    cohort_metrics = {}
    for cohort in ("warm", "cold"):
        sub = per_impression[per_impression["cohort"] == cohort]
        if len(sub):
            cohort_metrics[cohort] = {m: asdict(ranking_metric_ci(sub, m)) for m in METRIC_COLUMNS}

    paired = {}
    d, lo, hi = paired_metric_diff_ci(per_impression, "auc", "auc_baseline", N_BOOTSTRAP, SEED)
    paired["auc"] = {"diff": d, "ci_low": lo, "ci_high": hi}
    for m in ("mrr", "ndcg5", "ndcg10"):
        d, lo, hi = paired_metric_diff_ci(per_impression, m, f"{m}_baseline", N_BOOTSTRAP, SEED)
        paired[m] = {"diff": d, "ci_low": lo, "ci_high": hi}

    coef_by_name = {name: float(c) for name, c in zip(FEATURE_NAMES, coef)}

    config = {
        "candidate": "L_mind_history_features", "adr": "ADR-016", "date": str(date.today()),
        "bundle": bundle, "feature_names": FEATURE_NAMES, "decay": 0.9,
        "cold_threshold": COLD_THRESHOLD, "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
        "coefficients": coef_by_name, "intercept": intercept,
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "baseline_auc_overall": BASELINE_AUC_OVERALL,
        "n_train_rows": int(len(y_train)), "n_train_clicks": int(y_train.sum()),
        "n_dev_impressions": int(len(per_impression)), "n_cold_users": n_cold_users,
        "timings_s": {"build": build_s, "feature_build": feature_build_s, "fit": fit_s, "eval": eval_s},
    }
    results = {
        "metrics": metric_results, "cohort_metrics": cohort_metrics,
        "paired_vs_embed_baseline": paired,
        "log_click_count_max_abs_auc_delta_when_zeroed": click_count_ranking_delta,
    }
    return config, results


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="small")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config, results = run(args.bundle)

    out_dir = _REPO_ROOT / "experiments" / (
        args.out or f"candidate_l_mind_history_features_{args.bundle}_{config['date']}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {k: v for k, v in config.items() if k not in ("scaler_mean", "scaler_scale")}, indent=2
    ))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    m = results["metrics"]
    print(f"Candidate L (MIND history features) on {args.bundle}-dev, n={config['n_dev_impressions']:,}")
    for name in METRIC_COLUMNS:
        r = m[name]
        print(f"  {name:8s} {r['metric']:.4f}  [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print("\npaired vs. embedding baseline (0.6340):")
    for name, p in results["paired_vs_embed_baseline"].items():
        tag = "CI-clear" if (p["ci_low"] > 0 or p["ci_high"] < 0) else "not CI-clear"
        print(f"  {name:8s} {p['diff']:+.4f}  [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]  {tag}")
    print(f"\nlog_click_count zeroed-out: max |AUC delta| = "
          f"{results['log_click_count_max_abs_auc_delta_when_zeroed']:.2e} "
          f"(theory: exactly 0, since it's an additive per-impression constant)")
    print(f"\ncoefficients: {json.dumps(config['coefficients'], indent=2)}")
    print(f"\nwritten -> {out_dir}")


if __name__ == "__main__":
    main()
