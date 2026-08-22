#!/usr/bin/env python3
"""Candidate H (2026-08-22 session, ADR-010 addendum): does a nonlinear
model over Candidate F's own feature set — plus one new feature — beat the
deployed MIND embedding baseline (AUC 0.6340, 95% CI 0.6319-0.6361) on
MINDsmall-dev, where Candidate F's plain `LogisticRegression` combiner
(`run_learned_combiner_experiment.py`) already lost overall (0.6255, CI
0.6234-0.6276)?

Two models, isolating two separate hypotheses for why F might have lost:

- **H1** — `LogisticRegression(class_weight="balanced")`: same linear
  model as F, only the class-imbalance correction changes (F used untuned
  defaults on MINDsmall-train's 4.04% click rate). Isolates whether
  imbalance handling alone explains F's loss.
- **H2** — `sklearn.ensemble.HistGradientBoostingClassifier`: a nonlinear
  histogram-based gradient-boosted-tree model (same family LightGBM
  popularized; chosen over LightGBM itself to avoid a new dependency and
  macOS OpenMP install risk — see this session's `knowledge/ai-usage-log/`
  entry). Isolates whether nonlinear feature interactions/thresholds — not
  representable by F's linear combiner — explain the loss. Uses sklearn's
  own built-in `early_stopping=True`, which reserves a validation slice
  from the TRAINING data only (never touches MINDsmall-dev) — standard
  regularization for a tree ensemble, not a hyperparameter search, so this
  keeps ADR-009/ADR-010's "no hidden tuning" stance for first attempts.

One new feature beyond F's seven: `log_history_length =
log1p(n_articles)`, where `n_articles` is `HistoryProfile.n_articles`
(`src/retrieval/features.py`) — history length is currently only used
externally as a hard cohort-routing threshold (ADR-005's `< 5` cutoff,
Candidate G's routing decision); giving a nonlinear model the raw value
directly lets it learn its own warm/cold boundary (or a smooth continuum)
from data instead of relying on a cutoff chosen for evaluation slicing,
not model input.

Reuses Candidate F's train/dev index-and-query building
(`_build_side`/`_train_paths`) and per-candidate feature computation
(`_impression_feature_matrix`, `FEATURE_NAMES`) unchanged — only the
feature *count* (one appended column) and the *model* differ. Trained on
MINDsmall-**train**, never touching dev labels; evaluated on
MINDsmall-**dev** via the same Q4 harness every other candidate uses.
Compared against the baseline via the **paired** bootstrap
(`run_gated_cohort_experiment.py::paired_metric_diff_ci`, imported not
reimplemented) since H1/H2's dev-set impressions are the same underlying
impressions as the baseline's — the statistically correct test ADR-010
established for this shape of comparison.

Usage:
    poetry run python scripts/run_gbdt_combiner_experiment.py
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_bm25_experiment import COLD_THRESHOLD, dataset_paths
from run_gated_cohort_experiment import paired_metric_diff_ci
from run_learned_combiner_experiment import (
    FEATURE_NAMES,
    _build_side,
    _impression_feature_matrix,
    _train_paths,
)
from run_ranking_eval import _train_impressions_path
from src.evaluation.ranking_metrics import (
    build_train_popularity,
    mrr,
    ndcg_at_k,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)
from src.retrieval.features import HistoryProfile

N_BOOTSTRAP = 2000
SEED = 0
METRIC_COLUMNS = ["auc", "mrr", "ndcg5", "ndcg10"]
BASELINE_AUC_OVERALL = 0.63399337223849
FEATURE_NAMES_H = FEATURE_NAMES + ["log_history_length"]
MODEL_NAMES = ["h1_logreg_balanced", "h2_hist_gbdt"]


def append_history_length_feature(X: np.ndarray, profile: HistoryProfile) -> np.ndarray:
    """Appends one column — `log1p(profile.n_articles)`, repeated for every
    row of `X` (history length is a per-user, not per-candidate, quantity,
    so every candidate in an impression shares the same value) — to an
    existing `(k, len(FEATURE_NAMES))` feature matrix, returning a new
    `(k, len(FEATURE_NAMES_H))` array. `profile` defaults to an empty
    `HistoryProfile()` (n_articles=0) for a user missing from the lookup,
    matching every other feature's "skip what's missing" treatment."""
    k = X.shape[0]
    col = np.full((k, 1), np.log1p(profile.n_articles), dtype=np.float64)
    return np.hstack([X, col])


def _build_training_matrix(side: dict, popularity: dict) -> tuple[np.ndarray, np.ndarray]:
    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    n = len(impressions_sorted)
    X = np.empty((n, len(FEATURE_NAMES_H)), dtype=np.float64)
    y = np.empty(n, dtype=bool)
    empty_profile = HistoryProfile()
    pos = 0
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        k = len(candidate_ids)
        clicked = group["clicked"].to_numpy(dtype=bool)
        bm25q = side["bm25_query_by_user"].get(user_id, [])
        embedq = side["embed_query_by_user"].get(user_id, None)
        recq = side["recency_query_by_user"].get(user_id, None)
        profile = side["profile_by_user"].get(user_id, empty_profile)

        bm25_scores = side["bm25_scorer"].score(bm25q, candidate_ids)
        embed_scores = side["embed_scorer"].score(embedq, candidate_ids)
        recency_scores = side["recency_scorer"].score(recq, candidate_ids)

        X_base = _impression_feature_matrix(
            candidate_ids, bm25_scores, embed_scores, recency_scores, profile,
            side["category_lookup"], side["subcategory_lookup"], side["entity_set_lookup"], popularity,
        )
        X[pos:pos + k] = append_history_length_feature(X_base, profile)
        y[pos:pos + k] = clicked
        pos += k
    assert pos == n
    return X, y


def _evaluate(
    side: dict, popularity: dict,
    h1_scaler_mean, h1_scaler_scale, h1_coef, h1_intercept,
    h2_model, cold_threshold: int,
) -> pd.DataFrame:
    impressions_sorted = side["impressions"].sort_values(["user_id", "impression_id"], kind="stable")
    cohort_by_user = {
        uid: ("cold" if hlen < cold_threshold else "warm")
        for uid, hlen in side["history_len_by_user"].items()
    }
    empty_profile = HistoryProfile()
    rows = []
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        cohort = cohort_by_user.get(user_id, "cold")
        bm25q = side["bm25_query_by_user"].get(user_id, [])
        embedq = side["embed_query_by_user"].get(user_id, None)
        recq = side["recency_query_by_user"].get(user_id, None)
        profile = side["profile_by_user"].get(user_id, empty_profile)

        bm25_scores = side["bm25_scorer"].score(bm25q, candidate_ids)
        embed_scores = side["embed_scorer"].score(embedq, candidate_ids)
        recency_scores = side["recency_scorer"].score(recq, candidate_ids)

        X_base = _impression_feature_matrix(
            candidate_ids, bm25_scores, embed_scores, recency_scores, profile,
            side["category_lookup"], side["subcategory_lookup"], side["entity_set_lookup"], popularity,
        )
        X = append_history_length_feature(X_base, profile)

        h1_scores = ((X - h1_scaler_mean) / h1_scaler_scale) @ h1_coef + h1_intercept
        h2_scores = h2_model.predict_proba(X)[:, 1]

        row = {"impression_id": impression_id, "user_id": user_id, "cohort": cohort}
        for name, scores in (("h1", h1_scores), ("h2", h2_scores)):
            order = rank_candidates(scores, impression_id, seed=SEED)
            ranked_clicked = clicked[order]
            row[f"auc_{name}"] = safe_auc(scores, clicked)
            row[f"mrr_{name}"] = mrr(ranked_clicked)
            row[f"ndcg5_{name}"] = ndcg_at_k(ranked_clicked, 5)
            row[f"ndcg10_{name}"] = ndcg_at_k(ranked_clicked, 10)
        # Same-impression baseline AUC (deployed embedding scorer), for the
        # paired bootstrap — same convention as Candidate G's "auc_baseline".
        row["auc_baseline"] = safe_auc(embed_scores, clicked)
        rows.append(row)
    return pd.DataFrame(rows)


def run(bundle: str = "small") -> tuple[dict, dict]:
    dev_paths = dataset_paths("mind", bundle)
    train_paths = _train_paths(bundle)

    t0 = time.time()
    train_side = _build_side(train_paths)
    dev_side = _build_side(dev_paths)
    build_s = time.time() - t0

    train_impressions_for_popularity = pd.read_parquet(_train_impressions_path("mind", bundle))
    popularity = build_train_popularity(
        train_impressions_for_popularity, n_catalog=train_side["n_articles"], alpha=1.0
    )

    t0 = time.time()
    X_train, y_train = _build_training_matrix(train_side, popularity)
    feature_build_s = time.time() - t0

    # H1 — LogisticRegression, class_weight="balanced" (isolates the
    # imbalance-correction hypothesis from H2's model-class change).
    t0 = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    h1_model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=0)
    h1_model.fit(X_train_scaled, y_train)
    h1_fit_s = time.time() - t0
    h1_coef = h1_model.coef_[0]
    h1_intercept = float(h1_model.intercept_[0])

    # H2 — HistGradientBoostingClassifier, library defaults + built-in
    # early stopping (reserves its own validation slice from X_train only).
    t0 = time.time()
    h2_model = HistGradientBoostingClassifier(early_stopping=True, random_state=0)
    h2_model.fit(X_train, y_train)
    h2_fit_s = time.time() - t0

    t0 = time.time()
    per_impression = _evaluate(
        dev_side, popularity, scaler.mean_, scaler.scale_, h1_coef, h1_intercept, h2_model, COLD_THRESHOLD
    )
    eval_s = time.time() - t0

    metric_results: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        prefix = "h1" if model_name.startswith("h1") else "h2"
        metric_results[model_name] = {}
        for metric in METRIC_COLUMNS:
            col = f"{metric}_{prefix}"
            slices = {
                "overall": per_impression,
                "warm": per_impression[per_impression["cohort"] == "warm"],
                "cold": per_impression[per_impression["cohort"] == "cold"],
            }
            metric_results[model_name][metric] = {
                name: asdict(ranking_metric_ci(df, col, n_bootstrap=N_BOOTSTRAP, seed=SEED))
                for name, df in slices.items()
            }

    paired_results: dict[str, dict] = {}
    for model_name in MODEL_NAMES:
        prefix = "h1" if model_name.startswith("h1") else "h2"
        point, ci_low, ci_high = paired_metric_diff_ci(per_impression, f"auc_{prefix}", "auc_baseline")
        paired_results[model_name] = {
            "metric": f"auc_overall_{model_name}_minus_baseline",
            "point_diff": point,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_excludes_zero": bool(ci_low > 0),
        }

    n_cold_users = sum(1 for c in dev_side["history_len_by_user"].values() if c < COLD_THRESHOLD)
    config = {
        "dataset": "mind",
        "bundle": bundle,
        "method": "gbdt_combiner",
        "candidate": "H",
        "models": {
            "h1_logreg_balanced": "LogisticRegression(class_weight='balanced')",
            "h2_hist_gbdt": "HistGradientBoostingClassifier(early_stopping=True)",
        },
        "feature_names": FEATURE_NAMES_H,
        "n_train_rows": int(len(y_train)),
        "n_train_impressions": int(train_side["impressions"]["impression_id"].nunique()),
        "train_click_rate": float(y_train.mean()),
        "h1_scaler_mean": scaler.mean_.tolist(),
        "h1_scaler_scale": scaler.scale_.tolist(),
        "h1_coefficients": h1_coef.tolist(),
        "h1_intercept": h1_intercept,
        "h2_n_iterations": int(h2_model.n_iter_),
        "cold_threshold": COLD_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "n_users": len(dev_side["history_len_by_user"]),
        "n_warm_users": len(dev_side["history_len_by_user"]) - n_cold_users,
        "n_cold_users": n_cold_users,
        "n_impressions": len(per_impression),
        "index_build_seconds": round(build_s, 2),
        "feature_build_seconds": round(feature_build_s, 2),
        "h1_fit_seconds": round(h1_fit_s, 2),
        "h2_fit_seconds": round(h2_fit_s, 2),
        "eval_seconds": round(eval_s, 2),
        "baseline_auc_overall_embed": BASELINE_AUC_OVERALL,
        "date": str(date.today()),
    }
    results = {"metrics": metric_results, "paired_vs_baseline": paired_results}
    return config, results


if __name__ == "__main__":
    config, results = run()

    out_dir = _REPO_ROOT / "experiments" / f"candidate_h_gbdt_combiner_mind_small_{date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"Wrote {out_dir}/config.json and results.json")
    print(json.dumps(
        {k: v for k, v in config.items() if k not in ("h1_scaler_mean", "h1_scaler_scale")}, indent=2
    ))
    for model_name, slices_by_metric in results["metrics"].items():
        print(f"\n=== {model_name} ===")
        for metric, slices in slices_by_metric.items():
            print(f"\n{metric}")
            for name, r in slices.items():
                print(f"  {name}: {r['metric']:.4f} (95% CI: {r['ci_low']:.4f}-{r['ci_high']:.4f}), "
                      f"n_impressions={r['n_impressions']}, n_users={r['n_users']}, n_skipped={r['n_skipped']}")
        p = results["paired_vs_baseline"][model_name]
        print(f"\n  paired bootstrap ({model_name} - baseline, overall AUC): {p['point_diff']:+.4f} "
              f"(95% CI: {p['ci_low']:+.4f} to {p['ci_high']:+.4f}), CI excludes zero: {p['ci_excludes_zero']}")
