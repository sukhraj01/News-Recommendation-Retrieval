"""Equivalence tests for `ranking_metrics.per_impression_auc` (ADR-013).

`per_impression_auc` exists purely as a batched substitute for the scalar
`safe_auc`-in-a-loop path, which is too slow at `ebnerd_large` scale. A
faster metric that disagrees with the one every other number in this project was
measured against would silently make Candidate K incomparable to Candidates A-J.

So the point of these tests is not "is the AUC plausible" but "is it *the same
number*", including the awkward cases: ties, all-tied impressions, multi-click
impressions, and the degenerate single-class impressions `safe_auc` returns NaN
for.

Importing from `src.evaluation` rather than the experiment script also keeps
lightgbm out of the pytest process entirely — loading it alongside torch
segfaulted the suite deterministically at `test_nrms.py`.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ranking_metrics import per_impression_auc, safe_auc


def _scalar_reference(scores, labels, group_sizes):
    starts = np.concatenate([[0], np.cumsum(group_sizes)[:-1]]).astype(np.int64)
    return np.array(
        [
            safe_auc(scores[s : s + n], np.asarray(labels[s : s + n], dtype=bool))
            for s, n in zip(starts, group_sizes)
        ]
    )


def _assert_matches(scores, labels, sizes):
    fast = per_impression_auc(np.asarray(scores, float), np.asarray(labels), np.asarray(sizes))
    ref = _scalar_reference(np.asarray(scores, float), np.asarray(labels), np.asarray(sizes))
    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_matches_scalar_on_simple_impressions():
    _assert_matches(
        scores=[0.9, 0.1, 0.5, 0.2, 0.8],
        labels=[1, 0, 0, 0, 1],
        sizes=[3, 2],
    )


def test_matches_scalar_when_scores_tie():
    """Ties are where a naive rank implementation diverges from roc_auc_score."""
    _assert_matches(
        scores=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        labels=[1, 0, 0, 0, 1, 0],
        sizes=[3, 3],
    )


def test_matches_scalar_on_partial_ties():
    _assert_matches(
        scores=[0.7, 0.7, 0.2, 0.4, 0.4, 0.4, 0.9],
        labels=[1, 0, 0, 0, 1, 0, 0],
        sizes=[3, 4],
    )


def test_matches_scalar_with_multiple_clicks_in_one_impression():
    """EB-NeRD's mean is 1.004 clicks per impression — a few really do have 2+."""
    _assert_matches(
        scores=[0.9, 0.8, 0.1, 0.2],
        labels=[1, 1, 0, 0],
        sizes=[4],
    )


def test_degenerate_impressions_are_nan_exactly_like_safe_auc():
    sizes = np.array([3, 3])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    all_negative = np.array([0, 0, 0, 1, 0, 0])
    fast = per_impression_auc(scores, all_negative, sizes)
    assert np.isnan(fast[0])          # no positives -> undefined
    assert np.isfinite(fast[1])
    all_positive = np.array([1, 1, 1, 1, 0, 0])
    fast = per_impression_auc(scores, all_positive, sizes)
    assert np.isnan(fast[0])          # no negatives -> undefined


def test_perfect_and_inverted_rankings_are_one_and_zero():
    sizes = np.array([4])
    assert per_impression_auc(np.array([0.9, 0.8, 0.7, 0.6]), np.array([1, 0, 0, 0]), sizes)[0] == 1.0
    assert per_impression_auc(np.array([0.6, 0.8, 0.7, 0.9]), np.array([1, 0, 0, 0]), sizes)[0] == 0.0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_scalar_on_randomised_impressions(seed):
    """Fuzz against the reference across many shapes, tie densities and label counts."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 30, size=200)
    total = int(sizes.sum())
    # Coarse rounding deliberately manufactures ties at a realistic rate.
    scores = np.round(rng.random(total), 2)
    labels = np.zeros(total, dtype=np.int8)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for start, size in zip(starts, sizes):
        n_click = rng.integers(0, min(3, size) + 1)
        if n_click:
            labels[start + rng.choice(size, n_click, replace=False)] = 1
    _assert_matches(scores, labels, sizes)
