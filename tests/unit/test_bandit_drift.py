"""Tests for the drift-aware update rules, feature masks and groups (ADR-017 addendum)."""
import numpy as np
import pytest

from src.bandit.features import LinearFeatureTransformer
from src.bandit.groups import GROUPS, HYBRID_NAMES, assign_groups, group_mask
from src.bandit.policies import LinThompson, LinUCB, _SharedLinear, hybrid_prior_theta
from src.bandit.runner import run_prequential
from tests.unit.test_bandit import D0, make_stream

D, LAM = 12, 3.0


def _data(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, D)), rng.integers(0, 2, n).astype(float)


def test_exponential_forgetting_matches_direct_inverse_without_prior_restore():
    gamma = 0.97
    X, R = _data(300)
    m = _SharedLinear(D, LAM, None, refresh_every=0, gamma=gamma, restore_prior=False)
    A, b = LAM * np.eye(D), np.zeros(D)
    for x, r in zip(X, R):
        m.update(x, r)
        A, b = gamma * A + np.outer(x, x), gamma * b + r * x
        np.testing.assert_allclose(m.A, A, atol=1e-9)
        np.testing.assert_allclose(m.A_inv @ A, np.eye(D), atol=1e-8)
    np.testing.assert_allclose(m.theta, np.linalg.solve(A, b), atol=1e-8)


def test_forgetting_with_prior_restore_tracks_reference_recursion():
    gamma, f = 0.98, 16
    X, R = _data(400, 1)
    th0 = np.linspace(-1, 1, D)
    m = _SharedLinear(D, LAM, th0, refresh_every=0, gamma=gamma, floor_every=f)
    A, b = LAM * np.eye(D), LAM * th0
    for t, (x, r) in enumerate(zip(X, R), start=1):
        m.update(x, r)
        A, b = gamma * A + np.outer(x, x), gamma * b + r * x
        if t % f == 0:
            lost = 1 - gamma ** f
            A, b = A + LAM * lost * np.eye(D), b + LAM * lost * th0
        np.testing.assert_allclose(m.A, A, atol=1e-8)
        np.testing.assert_allclose(m.A_inv @ A, np.eye(D), atol=1e-7)
        np.testing.assert_allclose(m.theta, np.linalg.solve(A, b), atol=1e-7)
    # the prior floor is what keeps the posterior bounded in directions that never see data
    assert np.linalg.eigvalsh(m.A).min() > 0.5 * LAM * (1 - gamma ** f)


def test_forgetting_without_restore_loses_the_prior():
    """Documents why restore_prior exists: the literal rule lets an unobserved direction's precision decay to 0."""
    m = _SharedLinear(D, LAM, None, refresh_every=0, gamma=0.9, restore_prior=False)
    x = np.zeros(D); x[0] = 1.0  # only coordinate 0 is ever observed
    for _ in range(200):
        m.update(x, 1.0)
    assert m.A[1, 1] < 1e-6 * LAM


@pytest.mark.parametrize("window", [1, 7, 50])
def test_sliding_window_matches_direct_solve(window):
    X, R = _data(220, 2)
    th0 = np.linspace(0, 1, D)
    m = _SharedLinear(D, LAM, th0, refresh_every=0, window=window)
    for t in range(len(X)):
        m.update(X[t], R[t])
        lo = max(0, t + 1 - window)
        A = LAM * np.eye(D) + X[lo:t + 1].T @ X[lo:t + 1]
        b = LAM * th0 + X[lo:t + 1].T @ R[lo:t + 1]
        np.testing.assert_allclose(m.A, A, atol=1e-8)
        np.testing.assert_allclose(m.A_inv, np.linalg.inv(A), atol=1e-7)
        np.testing.assert_allclose(m.theta, np.linalg.solve(A, b), atol=1e-7)


def test_periodic_reset_returns_to_prior():
    X, R = _data(100, 3)
    th0 = np.linspace(-1, 1, D)
    m = _SharedLinear(D, LAM, th0, refresh_every=0, reset_every=30)
    for t in range(30):
        m.update(X[t], R[t])
    assert not np.allclose(m.theta, th0)
    m.update(X[30], R[30])  # 31st update: state was reset first, then this single observation applied
    A = LAM * np.eye(D) + np.outer(X[30], X[30])
    np.testing.assert_allclose(m.A, A, atol=1e-9)
    np.testing.assert_allclose(m.theta, np.linalg.solve(A, LAM * th0 + R[30] * X[30]), atol=1e-8)


def test_default_path_unchanged_by_new_options():
    X, R = _data(200, 4)
    a = _SharedLinear(D, LAM, None, refresh_every=0)
    b = _SharedLinear(D, LAM, None, refresh_every=0, gamma=1.0, window=None, reset_every=None)
    for x, r in zip(X, R):
        a.update(x, r); b.update(x, r)
    np.testing.assert_array_equal(a.A_inv, b.A_inv)
    np.testing.assert_array_equal(a.theta, b.theta)


def drift_policies(seed=0):
    th = hybrid_prior_theta(D0 + 2, D0, 1.0)
    kw = dict(prior_theta=th)
    return [
        LinUCB(D0 + 2, 0.05, 5.0, gamma=0.99, floor_every=20, **kw),
        LinUCB(D0 + 2, 0.05, 5.0, window=40, **kw),
        LinUCB(D0 + 2, 0.05, 5.0, reset_every=60, **kw),
        LinThompson(D0 + 2, 0.1, 5.0, seed=seed, gamma=0.99, floor_every=20, **kw),
        LinThompson(D0 + 2, 0.1, 5.0, seed=seed, window=40, **kw),
        LinThompson(D0 + 2, 0.1, 5.0, seed=seed, reset_every=60, **kw),
    ]


@pytest.mark.parametrize("interval", [None, 3600.0])
def test_drift_variants_ignore_future_labels(interval):
    base = make_stream(n_imp=500, seed=5, t_step=29.0)
    k = 250
    rng = np.random.default_rng(99)
    shuffled = base.label.copy()
    for i in range(k + 1, len(base)):
        r = base.rows(i); shuffled[r] = 0; shuffled[r.start + rng.integers(base.sizes[i])] = 1
    alt = make_stream(n_imp=500, seed=5, labels=shuffled, t_step=29.0)
    for p1, p2 in zip(drift_policies(7), drift_policies(7)):
        a = run_prequential(p1, base, update_interval_s=interval).arm
        b = run_prequential(p2, alt, update_interval_s=interval).arm
        np.testing.assert_array_equal(a[: k + 1], b[: k + 1])


def test_forgetting_actually_changes_decisions_and_is_deterministic():
    s = make_stream(n_imp=500, seed=2)
    th = hybrid_prior_theta(D0 + 2, D0, 0.0)
    plain = run_prequential(LinUCB(D0 + 2, 0.0, 1.0, th), s).arm
    forget = run_prequential(LinUCB(D0 + 2, 0.0, 1.0, th, gamma=0.95, floor_every=20), s).arm
    again = run_prequential(LinUCB(D0 + 2, 0.0, 1.0, th, gamma=0.95, floor_every=20), s).arm
    assert (plain != forget).any()
    np.testing.assert_array_equal(forget, again)


def test_feature_mask_removes_a_column_completely():
    s = make_stream(n_imp=200, seed=3)
    mask = np.ones(D0 + 2); mask[1] = 0.0
    a = run_prequential(LinUCB(D0 + 2, 0.1, 1.0, feature_mask=mask), s).arm
    s2 = make_stream(n_imp=200, seed=3)
    s2.features[:, 1] = np.random.default_rng(0).standard_normal(len(s2.features))  # perturb the masked column
    b = run_prequential(LinUCB(D0 + 2, 0.1, 1.0, feature_mask=mask), s2).arm
    np.testing.assert_array_equal(a, b)


def test_groups_partition_every_feature_exactly_once():
    t = LinearFeatureTransformer({"category_code": 26, "article_type": 12, "sentiment_label": 3})
    names = t.feature_names + HYBRID_NAMES
    groups = assign_groups(names)
    flat = [i for idx in groups.values() for i in idx]
    assert len(flat) == len(set(flat)) == len(names) - 1  # everything but bias
    assert names[-3] == "bias"  # bias precedes the two hybrid columns and belongs to no group
    assert len(groups["onehot_categoricals"]) == 41
    m = group_mask(names, drop=["freshness"])
    assert m[names.index("article_age_h")] == 0 and m[names.index("bias")] == 1
    k = group_mask(names, keep_only=["lightgbm_score"])
    assert k.sum() == 3  # lgb_prob, lgb_centred, bias
    assert set(GROUPS["freshness"]) <= set(names)


class _TimeSpy(LinUCB):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = []

    def select(self, context, candidates):
        self.log.append(("s", context.time))
        return super().select(context, candidates)

    def update(self, arm, reward):
        self.log.append(("u", arm.time))
        super().update(arm, reward)


@pytest.mark.parametrize("delay", [0.0, 100.0, 1000.0])
def test_delayed_feedback_only_uses_observations_old_enough(delay):
    s = make_stream(n_imp=300, t_step=29.0)
    spy = _TimeSpy(D0 + 2, 0.1, 1.0)
    run_prequential(spy, s, delay_s=delay)
    now = None
    for kind, t in spy.log:
        if kind == "s":
            now = t
        else:  # an update logged before the next select happens at that select's clock
            pass
    # re-walk: every update must precede (in log order) a select whose time >= obs_time + delay
    pending = []
    for kind, t in spy.log:
        if kind == "u":
            pending.append(t)
        else:
            assert all(o + delay <= t for o in pending)
            pending.clear()
    n_updates = sum(1 for k, _ in spy.log if k == "u")
    assert n_updates <= len(s)


def test_delayed_feedback_ignores_future_labels_and_delay_slows_learning():
    base = make_stream(n_imp=400, seed=5, t_step=29.0)
    k = 200
    rng = np.random.default_rng(3)
    shuffled = base.label.copy()
    for i in range(k + 1, len(base)):
        r = base.rows(i); shuffled[r] = 0; shuffled[r.start + rng.integers(base.sizes[i])] = 1
    alt = make_stream(n_imp=400, seed=5, labels=shuffled, t_step=29.0)
    a = run_prequential(LinUCB(D0 + 2, 0.05, 5.0, gamma=0.99, floor_every=20), base, delay_s=300.0).arm
    b = run_prequential(LinUCB(D0 + 2, 0.05, 5.0, gamma=0.99, floor_every=20), alt, delay_s=300.0).arm
    np.testing.assert_array_equal(a[: k + 1], b[: k + 1])
    c = run_prequential(LinUCB(D0 + 2, 0.05, 5.0, gamma=0.99, floor_every=20), base, delay_s=0.0).arm
    d = run_prequential(LinUCB(D0 + 2, 0.05, 5.0, gamma=0.99, floor_every=20), base, delay_s=3000.0).arm
    assert (c != d).any()
