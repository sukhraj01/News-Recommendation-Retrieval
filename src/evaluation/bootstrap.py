"""Shared per-user bootstrap resampling substrate.

Extracted from `metrics.py::recall_at_k`'s originally-inlined logic so
`ranking_metrics.py`'s per-impression metrics (AUC, MRR, nDCG@5, nDCG@10,
diversity, novelty) can reuse the exact same resample-and-reduce pattern
instead of duplicating it, per CLAUDE.md's "extend, don't duplicate"
guidance. `recall_at_k` itself is refactored to call `bootstrap_ratio_ci`
with no change to its public signature or behavior (pinned by
`tests/unit/test_metrics.py`, written before this extraction).
"""
from typing import Callable

import numpy as np


def bootstrap_ci(
    n_units: int,
    stat_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Generic percentile bootstrap: resample `n_units` indices with
    replacement `n_bootstrap` times, apply `stat_fn` to each resampled
    index array, return the (2.5, 97.5) percentile interval."""
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_units, size=n_units)
        boot[i] = stat_fn(idx)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    return float(ci_low), float(ci_high)


def bootstrap_ratio_ci(
    user_sum: np.ndarray,
    user_count: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Resamples users (not impressions) so a user with many impressions
    isn't independently resampled per-impression — each bootstrap replicate
    recomputes sum(user_sum)/sum(user_count) over the resampled users."""
    def stat_fn(idx: np.ndarray) -> float:
        return user_sum[idx].sum() / user_count[idx].sum()

    return bootstrap_ci(len(user_sum), stat_fn, n_bootstrap, seed)
