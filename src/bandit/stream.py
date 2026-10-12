"""Chronological impression stream + the bandit-feedback environment.

The stream is a flat candidate-row layout (same as `build_feature_frame`):
impression `i` owns rows `offsets[i] : offsets[i] + sizes[i]`.

`BanditEnvironment` is the only door to the labels. A policy (and the runner
loop) can ask "what reward did *this* arm get on *this* impression" and
nothing else -- the label array is name-mangled and there is no accessor for
other candidates or other impressions. The evaluation code (oracle, regret)
reads `stream.label` directly, after the run, never during it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Context:
    """What is known about the user/request when the decision is made."""
    index: int
    user_id: int
    time: float  # epoch seconds
    history_len: float
    n_candidates: int


@dataclass(frozen=True)
class Candidates:
    """The in-view set. `features` has `base_dim + 2` columns: the first
    `base_dim` are the leakage-safe linear features, the last two are the
    hybrid (LightGBM-score) columns. Policies take the slice they need."""
    article_ids: np.ndarray
    features: np.ndarray
    lgb_score: np.ndarray


@dataclass(frozen=True)
class Arm:
    """The chosen candidate, as handed back to `Policy.update`."""
    article_id: int
    features: np.ndarray
    lgb_score: float
    time: float


class ImpressionStream:
    def __init__(
        self,
        *,
        features: np.ndarray,
        lgb_score: np.ndarray,
        label: np.ndarray,
        sizes: np.ndarray,
        time: np.ndarray,
        user_id: np.ndarray,
        article_id: np.ndarray,
        history_len: np.ndarray,
        base_dim: int,
        sim_prob: np.ndarray | None = None,
        prior_exposure: np.ndarray | None = None,
    ):
        n = len(sizes)
        assert features.shape[0] == lgb_score.shape[0] == label.shape[0] == article_id.shape[0] == int(sizes.sum())
        assert features.shape[1] == base_dim + 2
        assert len(time) == len(user_id) == len(history_len) == n
        assert np.all(np.diff(time) >= 0), "stream must be chronologically ordered"
        assert np.isfinite(features).all(), "NaN/inf in stream features"
        self.features, self.lgb_score = features, lgb_score
        self.sizes = sizes.astype(np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)[:-1]]).astype(np.int64)
        self.time, self.user_id, self.article_id = time, user_id, article_id
        self.history_len, self.base_dim = history_len, base_dim
        self.label = label  # read by evaluation only; policies go through the environment
        self.sim_prob = sim_prob
        self.prior_exposure = prior_exposure  # per-row count of earlier in-view appearances (slicing only)

    def __len__(self) -> int:
        return len(self.sizes)

    def rows(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i] + self.sizes[i]))

    def context(self, i: int) -> Context:
        return Context(i, int(self.user_id[i]), float(self.time[i]), float(self.history_len[i]), int(self.sizes[i]))

    def candidates(self, i: int) -> Candidates:
        r = self.rows(i)
        return Candidates(self.article_id[r], self.features[r], self.lgb_score[r])

    def arm(self, i: int, k: int) -> Arm:
        r = self.rows(i)
        return Arm(int(self.article_id[r][k]), self.features[r][k], float(self.lgb_score[r][k]), float(self.time[i]))

    # ---- evaluation-side (never called from a policy or the runner loop) ----
    def oracle_reward(self) -> np.ndarray:
        return np.maximum.reduceat(self.label, self.offsets).astype(np.float64)

    def lgb_top1(self) -> np.ndarray:
        """Index (within the impression) of the LightGBM argmax; first index wins ties."""
        out = np.empty(len(self), dtype=np.int64)
        for i in range(len(self)):
            out[i] = int(np.argmax(self.lgb_score[self.rows(i)]))
        return out

    def window(self, lo: int, hi: int) -> "ImpressionStream":
        """Sub-stream of impressions [lo, hi) as a new, self-contained stream."""
        a = int(self.offsets[lo]); b = int(self.offsets[hi - 1] + self.sizes[hi - 1])
        return ImpressionStream(
            features=self.features[a:b], lgb_score=self.lgb_score[a:b], label=self.label[a:b],
            sizes=self.sizes[lo:hi], time=self.time[lo:hi], user_id=self.user_id[lo:hi],
            article_id=self.article_id[a:b], history_len=self.history_len[lo:hi],
            base_dim=self.base_dim,
            sim_prob=None if self.sim_prob is None else self.sim_prob[a:b],
            prior_exposure=None if self.prior_exposure is None else self.prior_exposure[a:b],
        )


class BanditEnvironment:
    """Reveals the reward of the chosen arm only (log-replay or simulated)."""

    def __init__(self, stream: ImpressionStream, *, simulated: bool = False, seed: int = 0):
        self.__label = stream.label
        self.__offsets = stream.offsets
        self.__sizes = stream.sizes
        self.__prob = stream.sim_prob
        self.__simulated = simulated
        self.__rng = np.random.default_rng(seed)
        if simulated and self.__prob is None:
            raise ValueError("simulated environment needs stream.sim_prob")
        self.n_pulls = 0

    def pull(self, i: int, k: int) -> float:
        assert 0 <= k < self.__sizes[i]
        self.n_pulls += 1
        row = int(self.__offsets[i]) + int(k)
        if self.__simulated:
            return float(self.__rng.random() < self.__prob[row])
        return float(self.__label[row])
