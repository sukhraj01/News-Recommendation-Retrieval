"""Unit tests for src/bandit (ADR-017).

All synthetic: no real data, no lightgbm/torch import (see src/bandit/__init__.py).
"""
import numpy as np
import pytest

from src.bandit.evaluation import (
    factorize_users, final_window_mask, ratio_ci, simulated_pseudo_regret, slice_masks,
)
from src.bandit.features import (
    LinearFeatureTransformer, assert_leakage_safe, bandit_source_features, fit_softmax_temperature,
    hybrid_columns, segment_softmax, POSITION_FEATURES,
)
from src.bandit.policies import (
    EpsilonGreedy, LightGBMGreedy, LinThompson, LinUCB, PopularityGreedy, RandomPolicy, _SharedLinear,
    hybrid_prior_theta, Policy,
)
from src.bandit.runner import run_prequential
from src.bandit.stream import BanditEnvironment, ImpressionStream
from src.retrieval.ebnerd_features import FEATURE_NAMES

D0 = 6


def make_stream(n_imp=400, seed=0, labels=None, t_step=37.0) -> ImpressionStream:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 12, size=n_imp)
    rows = int(sizes.sum())
    base = rng.standard_normal((rows, D0)).astype(np.float32)
    base[:, -1] = 1.0
    theta = np.array([1.5, -1.0, 0.5, 0.0, 0.0, 0.0])
    score = base @ theta + rng.standard_normal(rows) * 0.5
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    label = np.zeros(rows, dtype=np.int8)
    for o, s in zip(offsets, sizes):
        p = np.exp(score[o:o + s]); p /= p.sum()
        label[o + rng.choice(s, p=p)] = 1
    if labels is not None:
        label = labels
    lgb = (score + rng.standard_normal(rows) * 0.3).astype(np.float32)
    hyb = hybrid_columns(lgb, sizes, tau=1.0, sigma_c=1.0)
    feats = np.concatenate([base, hyb], axis=1)
    return ImpressionStream(
        features=feats, lgb_score=lgb, label=label, sizes=sizes,
        time=np.arange(n_imp) * t_step, user_id=rng.integers(0, 40, size=n_imp),
        article_id=rng.integers(0, 60, size=rows), history_len=rng.integers(1, 100, size=n_imp).astype(float),
        base_dim=D0, sim_prob=np.concatenate([np.exp(score[o:o + s]) / np.exp(score[o:o + s]).sum()
                                                for o, s in zip(offsets, sizes)]).astype(np.float32),
        prior_exposure=rng.integers(0, 500, size=rows).astype(np.int32),
    )


def all_policies(seed=0):
    return [
        RandomPolicy(seed), LightGBMGreedy(), EpsilonGreedy(0.2, seed), PopularityGreedy(3600.0, 2.0, seed),
        LinUCB(D0, 0.1, 5.0), LinThompson(D0, 0.1, 5.0, seed=seed),
        LinUCB(D0 + 2, 0.05, 5.0, hybrid_prior_theta(D0 + 2, D0, 1.0)),
        LinThompson(D0 + 2, 0.05, 5.0, hybrid_prior_theta(D0 + 2, D0, 1.0), seed=seed),
    ]


# ---- Sherman-Morrison -------------------------------------------------------

def test_sherman_morrison_matches_direct_inverse():
    rng = np.random.default_rng(1)
    d, lam = 25, 3.0
    m = _SharedLinear(d, lam, None, refresh_every=0)  # no refresh: test pure rank-1 updates
    xs = rng.standard_normal((600, d)); rs = rng.integers(0, 2, 600).astype(float)
    for x, r in zip(xs, rs):
        m.update(x, r)
    A = lam * np.eye(d) + xs.T @ xs
    np.testing.assert_allclose(m.A, A, atol=1e-9)
    np.testing.assert_allclose(m.A_inv, np.linalg.inv(A), atol=1e-9)
    np.testing.assert_allclose(m.theta, np.linalg.solve(A, xs.T @ rs), atol=1e-9)


def test_prior_theta_is_posterior_mean_before_data():
    th0 = hybrid_prior_theta(D0 + 2, D0, 1.0)
    m = _SharedLinear(D0 + 2, 7.0, th0)
    np.testing.assert_allclose(m.theta, th0)


# ---- determinism ------------------------------------------------------------

@pytest.mark.parametrize("interval", [None, 3600.0])
def test_determinism_under_fixed_seed(interval):
    s = make_stream()
    a = [run_prequential(p, s, update_interval_s=interval).arm for p in all_policies(seed=3)]
    b = [run_prequential(p, s, update_interval_s=interval).arm for p in all_policies(seed=3)]
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def test_different_seed_changes_stochastic_policy():
    s = make_stream()
    a = run_prequential(RandomPolicy(1), s).arm
    b = run_prequential(RandomPolicy(2), s).arm
    assert (a != b).any()


# ---- bandit feedback / no future -------------------------------------------

class SpyPolicy(Policy):
    name = "spy"

    def __init__(self):
        self.events = []

    def select(self, context, candidates):
        self.events.append(("select", context.index, context.time))
        assert not hasattr(candidates, "label") and not hasattr(context, "label")
        return 0

    def update(self, arm, reward):
        self.events.append(("update", arm.time, reward))


def test_reward_revealed_only_for_chosen_arm():
    s = make_stream(n_imp=60)
    spy = SpyPolicy()
    res = run_prequential(spy, s)
    updates = [e for e in spy.events if e[0] == "update"]
    assert len(updates) == len(s)  # exactly one observation per round
    # the reward the policy saw is the label of the arm it chose (index 0), nothing else
    expected = [float(s.label[s.offsets[i]]) for i in range(len(s))]
    assert [u[2] for u in updates] == expected
    np.testing.assert_array_equal(res.reward, expected)
    env = BanditEnvironment(s)
    env.pull(0, 0)
    assert env.n_pulls == 1
    assert not any(a == "label" for a in dir(env))  # no public label accessor


@pytest.mark.parametrize("interval", [None, 3600.0])
def test_updates_only_contain_the_past(interval):
    s = make_stream(n_imp=500, t_step=29.0)  # ~4 simulated hours
    spy = SpyPolicy()
    run_prequential(spy, s, update_interval_s=interval)
    pending = []  # observations applied since the previous decision
    for kind, a, b in spy.events:
        if kind == "update":
            pending.append(a)
            continue
        for obs_time in pending:  # every observation applied before THIS decision (made at time b)
            assert obs_time <= b
            if interval is not None:
                assert obs_time // interval < b // interval  # batch mode: only an earlier clock hour
        pending.clear()


@pytest.mark.parametrize("interval", [None, 3600.0])
def test_no_future_leakage_shuffled_future_labels_dont_change_past_decisions(interval):
    base = make_stream(n_imp=500, seed=5, t_step=29.0)
    k = 250
    rng = np.random.default_rng(99)
    shuffled = base.label.copy()
    cut = int(base.offsets[k + 1])
    # replace every future impression's click by a random candidate of that impression
    for i in range(k + 1, len(base)):
        r = base.rows(i); shuffled[r] = 0; shuffled[r.start + rng.integers(base.sizes[i])] = 1
    assert (shuffled[cut:] != base.label[cut:]).any()
    alt = make_stream(n_imp=500, seed=5, labels=shuffled, t_step=29.0)
    for p_true, p_alt in zip(all_policies(seed=7), all_policies(seed=7)):
        a = run_prequential(p_true, base, update_interval_s=interval).arm
        b = run_prequential(p_alt, alt, update_interval_s=interval).arm
        np.testing.assert_array_equal(a[: k + 1], b[: k + 1], err_msg=p_true.name)


def test_learning_policies_actually_use_feedback():
    """Guard against the vacuous version of the leakage test: changing PAST labels must change later decisions."""
    base = make_stream(n_imp=500, seed=5)
    flipped = base.label.copy()
    for i in range(0, 100):
        r = base.rows(i); flipped[r] = 0; flipped[r.start] = 1
    alt = make_stream(n_imp=500, seed=5, labels=flipped)
    a = run_prequential(LinUCB(D0, 0.05, 1.0), base).arm
    b = run_prequential(LinUCB(D0, 0.05, 1.0), alt).arm
    assert (a[100:] != b[100:]).any()


def test_hybrid_matches_lgbm_greedy_before_any_data():
    s = make_stream(n_imp=50)
    p = LinUCB(D0 + 2, 0.0, 1.0, hybrid_prior_theta(D0 + 2, D0, 1.0))
    for i in range(10):  # no update() calls
        assert p.select(s.context(i), s.candidates(i)) == int(np.argmax(s.lgb_score[s.rows(i)]))


# ---- shapes / NaN / leakage guard ------------------------------------------

def test_stream_rejects_nan_and_unsorted_time():
    s = make_stream(n_imp=20)
    bad = s.features.copy(); bad[3, 2] = np.nan
    with pytest.raises(AssertionError):
        ImpressionStream(features=bad, lgb_score=s.lgb_score, label=s.label, sizes=s.sizes, time=s.time,
                         user_id=s.user_id, article_id=s.article_id, history_len=s.history_len, base_dim=D0)
    with pytest.raises(AssertionError):
        ImpressionStream(features=s.features, lgb_score=s.lgb_score, label=s.label, sizes=s.sizes,
                         time=s.time[::-1].copy(), user_id=s.user_id, article_id=s.article_id,
                         history_len=s.history_len, base_dim=D0)


def test_candidate_shapes_and_finite_scores():
    s = make_stream(n_imp=30)
    for pol in all_policies():
        for i in range(len(s)):
            c = s.candidates(i)
            assert c.features.shape == (s.sizes[i], D0 + 2)
            k = pol.select(s.context(i), c)
            assert 0 <= k < s.sizes[i]
            pol.update(s.arm(i, k), float(s.label[s.offsets[i] + k]))
    lin = LinUCB(D0, 0.1, 1.0)
    assert np.isfinite(lin.model.theta).all() and np.isfinite(lin.model.A_inv).all()


def test_leakage_guard():
    for bad in ("total_inviews", "total_pageviews", "total_read_time", "next_read_time", "next_scroll_percentage"):
        with pytest.raises(AssertionError):
            assert_leakage_safe(["lt_cat_affinity", bad])
    with pytest.raises(AssertionError):
        assert_leakage_safe(["some_new_column"])
    names = bandit_source_features()
    assert set(names) <= set(FEATURE_NAMES)
    assert not set(names) & set(POSITION_FEATURES)
    assert_leakage_safe(names)


def test_transformer_output_is_finite_and_onehot():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, len(FEATURE_NAMES))).astype(np.float32) * 5
    X[:, FEATURE_NAMES.index("article_age_h")] = rng.exponential(50, 500)
    X[rng.random(X.shape) < 0.1] = np.nan
    X[3, 4] = np.inf
    card = {"category_code": 26, "article_type": 12, "sentiment_label": 3}
    for n, c in card.items():
        X[:, FEATURE_NAMES.index(n)] = rng.integers(-1, c, 500)
    t = LinearFeatureTransformer(card).fit(X)
    Z = t.transform(X)
    assert Z.shape == (500, t.dim) and np.isfinite(Z).all() and Z.dtype == np.float32
    assert (Z[:, -1] == 1).all()
    cat0 = len(t.numeric_names)
    assert (Z[:, cat0:cat0 + 26].sum(axis=1) <= 1).all()
    assert len(t.feature_names) == t.dim
    assert not any(n in t.feature_names for n in ("total_inviews", "position_in_view"))


def test_softmax_and_temperature_recovery():
    rng = np.random.default_rng(0)
    sizes = rng.integers(5, 15, 6000); rows = int(sizes.sum())
    score = rng.standard_normal(rows) * 2.0
    p = segment_softmax(score, sizes, 1.0)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    np.testing.assert_allclose(np.add.reduceat(p, starts), 1.0)
    true_tau = 2.5
    q = segment_softmax(score, sizes, true_tau)
    label = np.zeros(rows, dtype=np.int8)
    for o, s in zip(starts, sizes):
        label[o + rng.choice(s, p=q[o:o + s])] = 1
    assert fit_softmax_temperature(score, label, sizes) == pytest.approx(true_tau, rel=0.1)


# ---- evaluation helpers -----------------------------------------------------

def test_ratio_ci_point_and_coverage():
    rng = np.random.default_rng(0)
    users = rng.integers(0, 50, 2000)
    codes, n = factorize_users(users)
    r = (rng.random(2000) < 0.3).astype(float)
    point, lo, hi, cnt = ratio_ci(r, 1.0, codes, n, n_bootstrap=300)
    assert point == pytest.approx(r.mean()) and lo < point < hi and cnt == 2000
    m = np.zeros(2000, dtype=bool); m[:500] = True
    p2, *_ = ratio_ci(r, 1.0, codes, n, mask=m, n_bootstrap=50)
    assert p2 == pytest.approx(r[:500].mean())
    assert np.isnan(ratio_ci(r, 1.0, codes, n, mask=np.zeros(2000, bool), n_bootstrap=5)[0])


def test_pseudo_regret_zero_for_best_arm_and_masks():
    s = make_stream(n_imp=100)
    best = np.array([int(np.argmax(s.sim_prob[s.rows(i)])) for i in range(len(s))])
    assert np.allclose(simulated_pseudo_regret(s, best), 0)
    assert (simulated_pseudo_regret(s, np.zeros(len(s), dtype=int)) >= -1e-12).all()
    masks = slice_masks(s, cold_article_n=100, cold_user_n=5, low_hist_n=20)
    assert (masks["first_day"] | ~masks["first_day"]).all()
    assert final_window_mask(100, 0.2).sum() == 20
    assert int(s.oracle_reward().sum()) == len(s)


def test_leakage_test_has_teeth_future_peeker_is_detected():
    """Negative control: a policy that peeks at the NEXT impression's label must be caught by the
    same shuffled-future comparison the leakage test relies on."""
    base = make_stream(n_imp=300, seed=5)
    k = 150
    shuffled = base.label.copy()
    rng = np.random.default_rng(1)
    for i in range(k + 1, len(base)):
        r = base.rows(i); shuffled[r] = 0; shuffled[r.start + rng.integers(base.sizes[i])] = 1
    alt = make_stream(n_imp=300, seed=5, labels=shuffled)

    class FuturePeeker(Policy):
        name = "cheater"

        def __init__(self, stream):
            self.stream = stream

        def select(self, context, candidates):
            nxt = min(context.index + 1, len(self.stream) - 1)
            return int(np.argmax(self.stream.label[self.stream.rows(nxt)])) % context.n_candidates

    a = run_prequential(FuturePeeker(base), base).arm
    b = run_prequential(FuturePeeker(alt), alt).arm
    assert (a[: k + 1] != b[: k + 1]).any()
