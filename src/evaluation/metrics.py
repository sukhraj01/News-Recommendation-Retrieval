"""Recall@K with bootstrap confidence intervals, per CLAUDE.md's
benchmarking philosophy (never report an isolated number).

Reusable across retrieval methods (BM25 now, semantic retrieval later,
per ADR-002's forward note) since it only depends on a per-impression
hit/miss boolean and a user_id to resample over.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .bootstrap import bootstrap_ratio_ci


@dataclass
class RecallResult:
    recall: float
    ci_low: float
    ci_high: float
    n_positive_impressions: int
    n_users: int


def recall_at_k(
    hits: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> RecallResult:
    """`hits` has one row per positive (clicked=True) impression, with
    columns `user_id` (str) and `hit` (bool: was the clicked article in
    that user's top-K?). Bootstrap resamples users (not impressions), so a
    user with many positive impressions isn't independently resampled
    per-impression.
    """
    n_positive = len(hits)
    if n_positive == 0:
        return RecallResult(recall=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
                             n_positive_impressions=0, n_users=0)

    point_recall = hits["hit"].mean()

    # Resample per-user (sum hits, count) arrays with numpy fancy indexing
    # rather than pandas .loc per iteration — at MINDsmall-dev scale
    # (~50,000 users x 2,000 resamples) the pandas version is a real
    # bottleneck; this vectorized form runs in low single-digit seconds.
    grouped = hits.groupby("user_id")["hit"].agg(["sum", "count"])
    user_sum = grouped["sum"].to_numpy(dtype=np.float64)
    user_count = grouped["count"].to_numpy(dtype=np.float64)
    n_users = len(user_sum)

    ci_low, ci_high = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap, seed)
    return RecallResult(
        recall=float(point_recall),
        ci_low=ci_low,
        ci_high=ci_high,
        n_positive_impressions=n_positive,
        n_users=n_users,
    )
