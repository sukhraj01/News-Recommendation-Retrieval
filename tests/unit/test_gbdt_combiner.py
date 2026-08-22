"""Candidate H (2026-08-22 session): unit coverage for
`run_gbdt_combiner_experiment.py`'s own new logic — the history-length
feature-augmentation helper, and a minimal fit/predict smoke test for both
H1 (LogisticRegression, balanced) and H2 (HistGradientBoostingClassifier)
on a tiny synthetic feature matrix. The shared feature-computation/eval
harness (`_impression_feature_matrix`, `ranking_metric_ci`,
`paired_metric_diff_ci`, etc.) is already covered by
`test_features.py`/`test_gated_cohort.py` — not retested here."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_gbdt_combiner_experiment import append_history_length_feature
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.retrieval.features import HistoryProfile


def test_append_history_length_feature_repeats_value_per_candidate():
    X = np.zeros((3, 2))
    profile = HistoryProfile(n_articles=7)
    out = append_history_length_feature(X, profile)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out[:, :2], X)
    np.testing.assert_allclose(out[:, 2], np.log1p(7))


def test_append_history_length_feature_zero_for_empty_profile():
    X = np.ones((2, 1))
    out = append_history_length_feature(X, HistoryProfile())
    np.testing.assert_allclose(out[:, 1], 0.0)


def test_h1_logreg_balanced_fits_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] + rng.normal(scale=0.1, size=200) > 0)
    model = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced", random_state=0)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (200,)
    assert np.all((probs >= 0) & (probs <= 1))


def test_h2_hist_gbdt_fits_and_predicts_probabilities():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8))
    y = (X[:, 0] + rng.normal(scale=0.1, size=500) > 0)
    model = HistGradientBoostingClassifier(early_stopping=True, random_state=0)
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert probs.shape == (500,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert model.n_iter_ >= 1
