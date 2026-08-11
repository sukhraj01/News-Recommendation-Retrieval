"""Pinning tests for `recall_at_k`, which previously had no direct unit
test (only exercised indirectly via `scripts/run_bm25_experiment.py`).
Written before extracting the bootstrap logic into `src/evaluation/bootstrap.py`
so the refactor is checked against a real regression guard, not just
"looks equivalent" — the expected values below were computed against the
pre-refactor implementation and must not change afterward.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import recall_at_k


def test_recall_at_k_point_estimate_is_sum_over_count():
    hits = pd.DataFrame({
        "user_id": ["A", "A", "A", "B", "B", "C"],
        "hit": [True, True, False, False, False, True],
    })
    r = recall_at_k(hits, n_bootstrap=2000, seed=0)
    assert r.recall == pytest.approx(0.5)
    assert r.n_positive_impressions == 6
    assert r.n_users == 3
    # 3 users, wide interval expected — pinned exact values (seed=0).
    assert r.ci_low == pytest.approx(0.0)
    assert r.ci_high == pytest.approx(1.0)


def test_recall_at_k_pinned_against_larger_toy_dataset():
    """50 users, variable impression counts, deterministic RNG-generated
    hit pattern — values pinned from the pre-refactor implementation
    (seed=0, n_bootstrap=2000)."""
    rng = np.random.default_rng(42)
    users = [f"u{i}" for i in range(50) for _ in range(rng.integers(1, 4))]
    hit_vals = [bool(rng.integers(0, 2)) for _ in users]
    hits = pd.DataFrame({"user_id": users, "hit": hit_vals})

    r = recall_at_k(hits, n_bootstrap=2000, seed=0)
    assert r.n_positive_impressions == 109
    assert r.n_users == 50
    assert r.recall == pytest.approx(0.5045871559633027)
    assert r.ci_low == pytest.approx(0.42104033448106243)
    assert r.ci_high == pytest.approx(0.5945945945945946)


def test_recall_at_k_reproducible_with_same_seed():
    hits = pd.DataFrame({
        "user_id": ["A", "A", "B", "B", "C"],
        "hit": [True, False, True, True, False],
    })
    r1 = recall_at_k(hits, n_bootstrap=500, seed=0)
    r2 = recall_at_k(hits, n_bootstrap=500, seed=0)
    assert r1 == r2


def test_recall_at_k_empty_hits_returns_nan():
    empty = pd.DataFrame({"user_id": [], "hit": []})
    r = recall_at_k(empty)
    assert np.isnan(r.recall)
    assert np.isnan(r.ci_low)
    assert np.isnan(r.ci_high)
    assert r.n_positive_impressions == 0
    assert r.n_users == 0
