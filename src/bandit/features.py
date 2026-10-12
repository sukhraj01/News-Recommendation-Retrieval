"""Linear-model view of the existing leakage-safe EB-NeRD feature set.

Why a transformer at all: Candidate K's 65 features go into a tree ensemble,
which is indifferent to scale, NaNs and integer-coded categoricals. A shared-
parameter linear bandit is not -- it needs standardised numerics, no NaNs, and
categorical codes expanded to indicators (a linear model reading
`category_code=7` as "seven times sport" would be meaningless).

Leakage discipline (task step 7): this module introduces NO new data source.
Its only input is the matrix `build_feature_frame` already emits, in
`FEATURE_NAMES` order, and `assert_leakage_safe` pins that the columns it reads
are a subset of that audited list and never include the article-lifetime
aggregates / post-click outcomes ADR-009 excluded. Scaling statistics are
fitted on the last 24 h of the TRAIN window -- never on the evaluation stream.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import skew

from src.retrieval.ebnerd_features import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FORBIDDEN_RAW_COLUMNS,
)

# The shipped ranker (K_rank_nopos, ADR-013) withholds these; the bandit does
# too so that "bandit vs LightGBM-greedy" differs in policy, not in information.
POSITION_FEATURES: tuple[str, ...] = ("position_in_view", "relative_position_in_view")
# Item attributes expanded to indicators. device_type / gender are constant
# within an impression, so a *shared* linear score cannot use them to prefer one
# candidate over another -- they are dropped rather than one-hot-expanded.
ONEHOT_FEATURES: tuple[str, ...] = ("category_code", "article_type", "sentiment_label")
USER_CONSTANT_CATEGORICALS: tuple[str, ...] = ("device_type", "gender")

LOG_SKEW_THRESHOLD = 2.0
CLIP_PERCENTILES = (1.0, 99.0)


def assert_leakage_safe(source_names: list[str] | tuple[str, ...]) -> None:
    """Raise unless `source_names` is a subset of the audited feature list that
    excludes every field in `FORBIDDEN_RAW_COLUMNS` (total_inviews /
    total_pageviews / total_read_time / next_* )."""
    names = set(source_names)
    forbidden = names & set(FORBIDDEN_RAW_COLUMNS)
    if forbidden:
        raise AssertionError(f"leakage guard: forbidden fields in bandit features: {sorted(forbidden)}")
    unknown = names - set(FEATURE_NAMES)
    if unknown:
        raise AssertionError(
            f"leakage guard: {sorted(unknown)} are not in the audited ebnerd_features.FEATURE_NAMES"
        )


def bandit_source_features() -> list[str]:
    """Names (in `FEATURE_NAMES` order) of the raw columns the bandit reads."""
    names = [n for n in FEATURE_NAMES if n not in POSITION_FEATURES and n not in USER_CONSTANT_CATEGORICALS]
    assert_leakage_safe(names)
    return names


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class LinearFeatureTransformer:
    """raw 65-col matrix -> standardised numerics + one-hot categoricals + bias."""

    def __init__(self, cardinality: dict[str, int]):
        self.cardinality = dict(cardinality)
        self.source_names = bandit_source_features()
        self.numeric_names = [n for n in self.source_names if n not in CATEGORICAL_FEATURES]
        self.numeric_idx = np.array([FEATURE_NAMES.index(n) for n in self.numeric_names])
        self.onehot_idx = {n: FEATURE_NAMES.index(n) for n in ONEHOT_FEATURES}
        assert set(ONEHOT_FEATURES) <= set(self.source_names)
        self.log_mask: np.ndarray | None = None
        self.lo = self.hi = self.mean = self.std = None

    @property
    def feature_names(self) -> list[str]:
        out = list(self.numeric_names)
        for n in ONEHOT_FEATURES:
            out += [f"{n}={c}" for c in range(self.cardinality[n])]
        return out + ["bias"]

    @property
    def dim(self) -> int:
        return len(self.numeric_names) + sum(self.cardinality[n] for n in ONEHOT_FEATURES) + 1

    def fit(self, X_raw: np.ndarray) -> "LinearFeatureTransformer":
        assert X_raw.shape[1] == len(FEATURE_NAMES)
        num = X_raw[:, self.numeric_idx].astype(np.float64)
        num[~np.isfinite(num)] = np.nan
        k = num.shape[1]
        self.log_mask = np.zeros(k, dtype=bool)
        self.lo, self.hi = np.zeros(k), np.zeros(k)
        self.mean, self.std = np.zeros(k), np.ones(k)
        for j in range(k):
            col = num[:, j]
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            if col.size > 2 and np.ptp(col) > 0 and skew(col) > LOG_SKEW_THRESHOLD:
                self.log_mask[j] = True
                col = _signed_log1p(col)
            lo, hi = np.percentile(col, CLIP_PERCENTILES)
            clipped = np.clip(col, lo, hi)
            self.lo[j], self.hi[j] = lo, hi
            self.mean[j] = clipped.mean()
            self.std[j] = clipped.std() if clipped.std() > 1e-9 else 1.0
        return self

    def transform(self, X_raw: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("fit() first")
        assert X_raw.shape[1] == len(FEATURE_NAMES)
        num = X_raw[:, self.numeric_idx].astype(np.float64)
        num[~np.isfinite(num)] = np.nan
        num = np.where(self.log_mask[None, :], _signed_log1p(num), num)
        num = (np.clip(num, self.lo, self.hi) - self.mean) / self.std
        num = np.nan_to_num(num, nan=0.0)  # NaN -> train mean -> 0 after centring
        parts = [num]
        for n in ONEHOT_FEATURES:
            codes = X_raw[:, self.onehot_idx[n]]
            codes = np.where(np.isfinite(codes), codes, -1).astype(np.int64)
            oh = np.zeros((X_raw.shape[0], self.cardinality[n]))
            ok = (codes >= 0) & (codes < self.cardinality[n])
            oh[np.flatnonzero(ok), codes[ok]] = 1.0
            parts.append(oh)
        parts.append(np.ones((X_raw.shape[0], 1)))
        out = np.concatenate(parts, axis=1).astype(np.float32)
        assert out.shape[1] == self.dim and np.isfinite(out).all()
        return out

    def state(self) -> dict:
        return {
            "cardinality": self.cardinality, "source_names": self.source_names,
            "log_mask": self.log_mask.tolist(), "lo": self.lo.tolist(), "hi": self.hi.tolist(),
            "mean": self.mean.tolist(), "std": self.std.tolist(), "dim": self.dim,
        }


# ---- hybrid (LightGBM-score-as-feature) columns ---------------------------

def segment_softmax(scores: np.ndarray, sizes: np.ndarray, tau: float) -> np.ndarray:
    """Softmax of `scores / tau` inside each impression (contiguous groups)."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    z = scores.astype(np.float64) / tau
    z = z - np.repeat(np.maximum.reduceat(z, starts), sizes)
    e = np.exp(z)
    return e / np.repeat(np.add.reduceat(e, starts), sizes)


def fit_softmax_temperature(scores: np.ndarray, labels: np.ndarray, sizes: np.ndarray) -> float:
    """MLE temperature so that softmax(score/tau) is calibrated against clicks.

    Lambdarank scores are only defined up to a monotone map; the hybrid bandit
    wants a quantity on the click-probability scale so a prior weight of 1.0 on
    it is meaningful. Fitted on held-out TRAIN-window impressions only.
    """
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    n_click = np.add.reduceat(labels.astype(np.float64), starts)

    def nll(log_tau: float) -> float:
        p = segment_softmax(scores, sizes, float(np.exp(log_tau)))
        lp = np.log(np.maximum(p, 1e-12)) * labels
        return -float((np.add.reduceat(lp, starts) / np.maximum(n_click, 1))[n_click > 0].mean())

    res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded")
    return float(np.exp(res.x))


def hybrid_columns(lgb_score: np.ndarray, sizes: np.ndarray, tau: float, sigma_c: float) -> np.ndarray:
    """(R, 2): [calibrated softmax probability, impression-centred score / sigma_c]."""
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    centred = lgb_score.astype(np.float64) - np.repeat(np.add.reduceat(lgb_score.astype(np.float64), starts) / sizes, sizes)
    p = segment_softmax(lgb_score, sizes, tau)
    return np.stack([p, centred / sigma_c], axis=1).astype(np.float32)
