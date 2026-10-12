"""Metrics, slices and bootstrap CIs for bandit replays.

Resampling unit: USERS (not impressions), the same unit `bootstrap_ratio_ci`
uses for every other metric in this project. A user contributes many
correlated impressions; resampling impressions independently would shrink the
CIs. Day-block resampling is not used: the held-out period is only four days,
too few blocks for a stable percentile interval.

Every statistic here is a ratio of per-user sums, `sum(num)/sum(den)`, so one
helper (`ratio_ci`) covers CTR (den=1), paired differences (num=r_a-r_b, den=1)
and conditional rates (den = indicator).
"""
from __future__ import annotations

import numpy as np

from src.evaluation.bootstrap import bootstrap_ci

N_BOOTSTRAP = 2000


def factorize_users(user_id: np.ndarray) -> tuple[np.ndarray, int]:
    uniq, codes = np.unique(user_id, return_inverse=True)
    return codes.astype(np.int64), len(uniq)


def ratio_ci(
    num: np.ndarray,
    den: np.ndarray | float,
    user_codes: np.ndarray,
    n_users: int,
    *,
    mask: np.ndarray | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = 0,
) -> tuple[float, float, float, int]:
    """(point, ci_low, ci_high, n_impressions) for sum(num)/sum(den), user-bootstrap."""
    num = np.asarray(num, dtype=np.float64)
    den = np.broadcast_to(np.asarray(den, dtype=np.float64), num.shape)
    if mask is not None:
        num, den, codes = num[mask], den[mask], user_codes[mask]
    else:
        codes = user_codes
    n_imp = int(len(num))
    if n_imp == 0 or den.sum() == 0:
        return float("nan"), float("nan"), float("nan"), n_imp
    u_num = np.bincount(codes, weights=num, minlength=n_users)
    u_den = np.bincount(codes, weights=den, minlength=n_users)
    keep = u_den > 0
    u_num, u_den = u_num[keep], u_den[keep]
    point = float(u_num.sum() / u_den.sum())
    lo, hi = bootstrap_ci(len(u_num), lambda idx: u_num[idx].sum() / u_den[idx].sum(), n_bootstrap, seed)
    return point, lo, hi, n_imp


def final_window_mask(n: int, frac: float = 0.2) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[int(round(n * (1 - frac))):] = True
    return m


def oracle_reward(stream) -> np.ndarray:
    return stream.oracle_reward()


def simulated_pseudo_regret(stream, arm: np.ndarray) -> np.ndarray:
    """Per-impression expected regret under the simulator: max_k p_k - p_chosen."""
    p = stream.sim_prob.astype(np.float64)
    best = np.maximum.reduceat(p, stream.offsets)
    chosen = p[stream.offsets + arm]
    return best - chosen


def slice_masks(stream, *, cold_article_n: int, cold_user_n: int, low_hist_n: int) -> dict[str, np.ndarray]:
    """Boolean masks over the impressions of `stream`. Slices that use the label
    (cold-article is defined on the clicked article) are an evaluation-side
    partition only -- policies never see them."""
    n = len(stream)
    masks: dict[str, np.ndarray] = {}
    clicked_prior = np.where(stream.label > 0, stream.prior_exposure, np.iinfo(np.int32).max)
    min_clicked_prior = np.minimum.reduceat(clicked_prior, stream.offsets)
    cold_art = min_clicked_prior < cold_article_n
    masks[f"cold_article (clicked article <{cold_article_n} prior exposures)"] = cold_art
    masks[f"warm_article (>= {cold_article_n})"] = ~cold_art
    masks[f"cold_user (history < {cold_user_n})"] = stream.history_len < cold_user_n
    masks[f"low_history_user (history < {low_hist_n})"] = stream.history_len < low_hist_n
    masks[f"higher_history_user (history >= {low_hist_n})"] = stream.history_len >= low_hist_n
    day = ((stream.time - stream.time[0]) // 86400).astype(int)
    first, last = day.min(), day.max()
    masks["first_day"] = day == first
    masks["last_day"] = day == last
    med = float(np.median(stream.sizes))
    masks[f"small_candidate_set (<= {med:g})"] = stream.sizes <= med
    masks[f"large_candidate_set (> {med:g})"] = stream.sizes > med
    return masks
