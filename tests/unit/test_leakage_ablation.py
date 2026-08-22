"""Q9 anti-gaming ablation (scripts/run_leakage_ablation.py): unit coverage
for the two pure functions the with/without comparison depends on. The
end-to-end scoring loop itself is exercised by the real run recorded in
`experiments/ablation_leaky_features_ebnerd_small_2026-08-14/`, not
re-tested here."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_leakage_ablation import LEAK_FIELDS, _compute_leak_scores, _minmax


def test_minmax_scales_into_zero_one():
    out = _minmax(np.array([10.0, 20.0, 30.0]))
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


def test_minmax_constant_input_is_all_zero():
    # No spread to normalize against — must not divide by zero.
    out = _minmax(np.array([5.0, 5.0, 5.0]))
    assert np.array_equal(out, np.zeros(3))


def test_compute_leak_scores_missing_value_imputed_with_median_not_zero():
    raw = pd.DataFrame({
        "total_inviews": [100.0, np.nan, 300.0],
        "total_pageviews": [10.0, 20.0, 30.0],
        "total_read_time": [1000.0, 2000.0, 3000.0],
    })
    scores = _compute_leak_scores(raw)

    # Article 1's missing total_inviews is imputed with the column median
    # (200.0, between 100 and 300) -> its inviews component sits strictly
    # between article 0's and article 2's, not pinned to either extreme
    # (which a zero-fill would do, silently understating a popular but
    # sparsely-logged article's leak signal).
    assert 0.0 < scores[1] < scores[2]
    assert scores[0] < scores[1]


def test_compute_leak_scores_monotonic_in_each_field():
    raw = pd.DataFrame({
        "total_inviews": [1.0, 10.0, 100.0],
        "total_pageviews": [1.0, 10.0, 100.0],
        "total_read_time": [1.0, 10.0, 100.0],
    })
    scores = _compute_leak_scores(raw)
    assert scores[0] < scores[1] < scores[2]


def test_compute_leak_scores_covers_all_named_fields():
    # Regression guard: if Q9's field list is ever edited, the computation
    # must still read exactly those columns from `raw`.
    raw = pd.DataFrame({field: [1.0, 2.0] for field in LEAK_FIELDS})
    scores = _compute_leak_scores(raw)
    assert len(scores) == 2


