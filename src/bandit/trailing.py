"""Causal trailing article-CTR features under a feedback delay (ADR-018 addendum).

Every candidate row is an observation (article, time, clicked). For a row at time t with feedback
delay D and window W, the statistics are computed ONLY from observations with
        t - D - W  <=  obs_time  <  t - D .
The upper bound is strict on purpose: with D = 0 a non-strict bound would admit the row's own
label (and same-second siblings), a direct leak. For D > 0 the strict/non-strict distinction only
matters for events at exactly t - D (second resolution). Shows and clicks are windowed identically,
so a delayed click is never paired with an undelayed show.

Implementation is offline-vectorised (sort once, two searchsorted per row) but the definition is
purely a function of (article, time) -- the test in tests/unit/test_trailing.py compares it with a
brute-force loop and checks that flipping every label at obs_time >= t - D changes nothing.
"""
from __future__ import annotations

import numpy as np

PAD = 1 << 17           # keeps lower bounds non-negative inside one article's key block
BLOCK = 1 << 23         # time span per article block (about 97 days of seconds)


def _keys(article_idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    return article_idx.astype(np.int64) * BLOCK + (t.astype(np.int64) + PAD)


def trailing_counts(
    article_idx: np.ndarray, t_sec: np.ndarray, label: np.ndarray, *, delay_s: int, window_s: int
) -> tuple[np.ndarray, np.ndarray]:
    """(clicks, shows) per row over [t - D - W, t - D). `t_sec` integer seconds >= 0."""
    t_sec = np.asarray(t_sec, dtype=np.int64)
    assert t_sec.min() >= 0 and t_sec.max() + PAD + delay_s < BLOCK, "time span exceeds the key block"
    assert delay_s + window_s < PAD
    all_keys = np.sort(_keys(article_idx, t_sec))
    click_keys = np.sort(_keys(article_idx[label > 0], t_sec[label > 0]))
    hi = _keys(article_idx, t_sec - delay_s)
    lo = _keys(article_idx, t_sec - delay_s - window_s)
    shows = np.searchsorted(all_keys, hi, "left") - np.searchsorted(all_keys, lo, "left")
    clicks = np.searchsorted(click_keys, hi, "left") - np.searchsorted(click_keys, lo, "left")
    return clicks.astype(np.float64), shows.astype(np.float64)


def smoothed_ctr(clicks: np.ndarray, shows: np.ndarray, prior_ctr: float = 0.1, prior_strength: float = 5.0) -> np.ndarray:
    return (clicks + prior_strength * prior_ctr) / (shows + prior_strength)


WINDOWS_S = {"1h": 3600, "2h": 7200}
FEATURE_NAMES_TRAILING = [
    "trail_ctr_1h", "trail_shows_1h", "trail_ctr_2h", "trail_shows_2h",
    "trail_ctr_1h_rank_in_imp", "trail_ctr_2h_rank_in_imp",
]


def trailing_feature_matrix(article_idx, t_sec, label, group_starts, group_sizes, *, delay_s: int, rank_fn) -> np.ndarray:
    """(R, 6) float32 in `FEATURE_NAMES_TRAILING` order. `rank_fn(values, group_starts, sizes)` is
    ebnerd_features._rank_within_group (injected so this module stays import-light)."""
    cols = {}
    for name, w in WINDOWS_S.items():
        c, s = trailing_counts(article_idx, t_sec, label, delay_s=delay_s, window_s=w)
        cols[f"trail_ctr_{name}"] = smoothed_ctr(c, s)
        cols[f"trail_shows_{name}"] = s
    cols["trail_ctr_1h_rank_in_imp"] = rank_fn(cols["trail_ctr_1h"], group_starts, group_sizes)
    cols["trail_ctr_2h_rank_in_imp"] = rank_fn(cols["trail_ctr_2h"], group_starts, group_sizes)
    return np.stack([cols[n] for n in FEATURE_NAMES_TRAILING], axis=1).astype(np.float32)
