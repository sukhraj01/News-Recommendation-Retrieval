"""Candidate G (2026-08-21 second session): unit coverage for
`run_gated_cohort_experiment.py::paired_metric_diff_ci`, the paired
bootstrap used to test "does gating actually beat baseline" on the same
underlying impressions, rather than comparing two independent marginal
CIs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_gated_cohort_experiment import paired_metric_diff_ci


def test_paired_metric_diff_ci_zero_when_columns_identical():
    df = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "auc": [0.6, 0.7, 0.5, 0.9],
        "auc_baseline": [0.6, 0.7, 0.5, 0.9],
    })
    point, ci_low, ci_high = paired_metric_diff_ci(df, "auc", "auc_baseline", n_bootstrap=200)
    assert point == pytest.approx(0.0)
    assert ci_low <= 0.0 <= ci_high


def test_paired_metric_diff_ci_detects_consistent_positive_shift():
    # Every user's "auc" is exactly 0.1 higher than "auc_baseline" - a real,
    # consistent paired improvement with zero within-pair noise, so the CI
    # should sit entirely above zero even with few bootstrap replicates.
    rng = np.random.default_rng(0)
    n_users = 50
    df = pd.DataFrame({
        "user_id": [f"u{i}" for i in range(n_users)],
        "auc_baseline": rng.uniform(0.4, 0.6, size=n_users),
    })
    df["auc"] = df["auc_baseline"] + 0.1
    point, ci_low, ci_high = paired_metric_diff_ci(df, "auc", "auc_baseline", n_bootstrap=500)
    assert point == pytest.approx(0.1, abs=1e-9)
    assert ci_low > 0.0


def test_paired_metric_diff_ci_drops_rows_with_nan_in_either_column():
    df = pd.DataFrame({
        "user_id": ["u1", "u2", "u3"],
        "auc": [0.6, np.nan, 0.7],
        "auc_baseline": [0.5, 0.5, np.nan],
    })
    # Only u1's row is complete in both columns.
    point, ci_low, ci_high = paired_metric_diff_ci(df, "auc", "auc_baseline", n_bootstrap=50)
    assert point == pytest.approx(0.1)
