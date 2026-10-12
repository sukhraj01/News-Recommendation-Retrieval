"""Bandit policies behind one interface: `select(context, candidates) -> arm`,
`update(arm, reward)`.

Why a SHARED parameter vector for the linear bandits (not one model per arm):
the arm set is the article catalogue (20,738 articles in ebnerd_small, ~4.7k of
them ever shown in the validation window, and each is born and dies within days
-- `article_age_h` is the dominant feature). Disjoint LinUCB would keep a
d x d matrix per article (4.7k x 100^2 x 8 B = 380 MB, far more at
ebnerd_large's catalogue) and every article would start from zero data, so the
"cold" case would be the *only* case. A shared theta over joint
(user, article, context) features transfers across articles: a brand-new
article is scored from its metadata immediately. Hybrid-LinUCB (Li et al. 2010,
Alg. 2) is the same idea with an extra per-arm term; per-article terms would
again be cold-start-starved, so the shared form is used throughout.

Policies hold no reference to labels. They see a reward only through `update`,
for the arm they chose.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque

import numpy as np

from .stream import Arm, Candidates, Context


class Policy(ABC):
    name: str = "policy"
    stochastic: bool = False

    @abstractmethod
    def select(self, context: Context, candidates: Candidates) -> int:
        """Index into `candidates` of the chosen arm."""

    def update(self, arm: Arm, reward: float) -> None:  # noqa: B027 - no-op by default
        """Learn from the one observed (arm, reward). Static policies ignore it."""


class RandomPolicy(Policy):
    name = "random"
    stochastic = True

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select(self, context: Context, candidates: Candidates) -> int:
        return int(self.rng.integers(len(candidates.article_ids)))


class LightGBMGreedy(Policy):
    """Frozen ranker, top-1. Scores are the booster's output computed upstream
    from the same feature rows (the booster is never retrained or updated)."""
    name = "lgbm_greedy"

    def select(self, context: Context, candidates: Candidates) -> int:
        return int(np.argmax(candidates.lgb_score))


class EpsilonGreedy(Policy):
    name = "eps_greedy_lgbm"
    stochastic = True

    def __init__(self, epsilon: float, seed: int = 0):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self, context: Context, candidates: Candidates) -> int:
        n = len(candidates.article_ids)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(n))
        return int(np.argmax(candidates.lgb_score))


class PopularityGreedy(Policy):
    """Trailing-window CTR per article, from this policy's OWN observed pulls.

    Under bandit feedback the only CTR evidence is for arms it has shown, so an
    article's estimate is a Beta-smoothed (prior strength `m`, prior mean = the
    policy's global trailing CTR) clicks/shows over the last `window_s` seconds.
    Ties (all unseen articles) break uniformly at random.
    """
    name = "popularity_greedy"
    stochastic = True

    def __init__(self, window_s: float = 24 * 3600.0, prior_strength: float = 5.0, seed: int = 0):
        self.window_s, self.m = window_s, prior_strength
        self.rng = np.random.default_rng(seed)
        self.events: dict[int, deque] = defaultdict(deque)  # article -> deque[(t, r)]
        self.clicks: dict[int, float] = defaultdict(float)
        self.global_events: deque = deque()
        self.g_clicks = 0.0

    def _prune(self, article: int, now: float) -> None:
        q = self.events[article]
        while q and q[0][0] < now - self.window_s:
            _, r = q.popleft()
            self.clicks[article] -= r

    def select(self, context: Context, candidates: Candidates) -> int:
        now = context.time
        while self.global_events and self.global_events[0][0] < now - self.window_s:
            self.g_clicks -= self.global_events.popleft()[1]
        p0 = self.g_clicks / len(self.global_events) if self.global_events else 0.1
        est = np.empty(len(candidates.article_ids))
        for k, a in enumerate(candidates.article_ids):
            a = int(a)
            self._prune(a, now)
            shows = len(self.events[a])
            est[k] = (self.clicks[a] + self.m * p0) / (shows + self.m)
        best = np.flatnonzero(est >= est.max() - 1e-12)
        return int(best[self.rng.integers(len(best))])

    def update(self, arm: Arm, reward: float) -> None:
        a = arm.article_id
        self.events[a].append((arm.time, reward))
        self.clicks[a] += reward
        self.global_events.append((arm.time, reward))
        self.g_clicks += reward


class _SharedLinear:
    """Ridge posterior with a shared theta; A^-1 maintained by Sherman-Morrison.

    A_{t+1}^-1 = A_t^-1 - (A_t^-1 x)(A_t^-1 x)^T / (1 + x^T A_t^-1 x): O(d^2) per
    observation versus O(d^3) to re-invert. `A` is also kept so (a) tests can
    compare against the direct inverse and (b) `refresh_every` can re-invert to
    cancel floating-point drift accumulated over many rank-1 updates.
    """

    def __init__(self, dim: int, lam: float, prior_theta: np.ndarray | None, refresh_every: int = 20000):
        self.dim, self.lam = dim, lam
        self.A = np.eye(dim) * lam
        self.A_inv = np.eye(dim) / lam
        theta0 = np.zeros(dim) if prior_theta is None else np.asarray(prior_theta, dtype=np.float64)
        assert theta0.shape == (dim,)
        # b = lam * theta0 makes the posterior mean exactly theta0 before any data.
        self.b = lam * theta0
        self.theta = theta0.copy()
        self.n_updates = 0
        self.refresh_every = refresh_every
        self.version = 0

    def update(self, x: np.ndarray, reward: float) -> None:
        x = x.astype(np.float64, copy=False)
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.A_inv = 0.5 * (self.A_inv + self.A_inv.T)
        self.A += np.outer(x, x)
        self.b += reward * x
        self.n_updates += 1
        if self.refresh_every and self.n_updates % self.refresh_every == 0:
            self.A_inv = np.linalg.inv(self.A)
            self.A_inv = 0.5 * (self.A_inv + self.A_inv.T)
        self.theta = self.A_inv @ self.b
        self.version += 1


def hybrid_prior_theta(dim: int, base_dim: int, prior_weight: float) -> np.ndarray:
    """theta0 that puts `prior_weight` on the calibrated LightGBM probability
    column (index `base_dim`) and 0 elsewhere -> before any data the bandit
    ranks exactly like LightGBM-greedy."""
    theta = np.zeros(dim)
    theta[base_dim] = prior_weight
    return theta


class LinUCB(Policy):
    """score = theta^T x + alpha * sqrt(x^T A^-1 x)."""
    name = "linucb"

    def __init__(self, dim: int, alpha: float, lam: float = 1.0, prior_theta: np.ndarray | None = None,
                 name: str | None = None):
        self.dim, self.alpha = dim, alpha
        self.model = _SharedLinear(dim, lam, prior_theta)
        if name:
            self.name = name

    def select(self, context: Context, candidates: Candidates) -> int:
        X = candidates.features[:, : self.dim].astype(np.float64, copy=False)
        mean = X @ self.model.theta
        if self.alpha > 0:
            bonus = np.sqrt(np.maximum(np.einsum("ij,ij->i", X @ self.model.A_inv, X), 0.0))
            mean = mean + self.alpha * bonus
        return int(np.argmax(mean))

    def update(self, arm: Arm, reward: float) -> None:
        self.model.update(arm.features[: self.dim], reward)


class LinThompson(Policy):
    """Gaussian linear Thompson sampling: theta~ ~ N(theta_hat, v^2 A^-1), one
    draw per decision shared by all candidates (Agrawal & Goyal 2013)."""
    name = "lints"
    stochastic = True

    def __init__(self, dim: int, v: float, lam: float = 1.0, prior_theta: np.ndarray | None = None,
                 seed: int = 0, name: str | None = None):
        self.dim, self.v = dim, v
        self.model = _SharedLinear(dim, lam, prior_theta)
        self.rng = np.random.default_rng(seed)
        self._chol_version = -1
        self._chol: np.ndarray | None = None
        if name:
            self.name = name

    def _factor(self) -> np.ndarray:
        if self._chol_version != self.model.version:
            S = self.model.A_inv
            try:
                self._chol = np.linalg.cholesky(S)
            except np.linalg.LinAlgError:
                w, V = np.linalg.eigh(S)
                self._chol = V * np.sqrt(np.maximum(w, 1e-12))
            self._chol_version = self.model.version
        return self._chol

    def select(self, context: Context, candidates: Candidates) -> int:
        z = self.rng.standard_normal(self.dim)
        theta_tilde = self.model.theta + self.v * (self._factor() @ z)
        X = candidates.features[:, : self.dim].astype(np.float64, copy=False)
        return int(np.argmax(X @ theta_tilde))

    def update(self, arm: Arm, reward: float) -> None:
        self.model.update(arm.features[: self.dim], reward)
