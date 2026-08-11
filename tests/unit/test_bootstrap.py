import numpy as np

from src.evaluation.bootstrap import bootstrap_ci, bootstrap_ratio_ci


def test_bootstrap_ratio_ci_reproducible_with_same_seed():
    user_sum = np.array([2.0, 0.0, 1.0, 3.0, 5.0])
    user_count = np.array([3.0, 2.0, 1.0, 4.0, 5.0])
    ci1 = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap=500, seed=0)
    ci2 = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap=500, seed=0)
    assert ci1 == ci2


def test_bootstrap_ratio_ci_different_seeds_can_differ():
    user_sum = np.array([2.0, 0.0, 1.0, 3.0, 5.0])
    user_count = np.array([3.0, 2.0, 1.0, 4.0, 5.0])
    ci_a = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap=200, seed=0)
    ci_b = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap=200, seed=1)
    assert ci_a != ci_b


def test_bootstrap_ci_narrows_with_more_units():
    rng = np.random.default_rng(7)

    def make_stat_fn(values):
        def stat_fn(idx):
            return values[idx].mean()
        return stat_fn

    small = rng.integers(0, 2, size=10).astype(float)
    large = rng.integers(0, 2, size=2000).astype(float)

    ci_small = bootstrap_ci(len(small), make_stat_fn(small), n_bootstrap=1000, seed=0)
    ci_large = bootstrap_ci(len(large), make_stat_fn(large), n_bootstrap=1000, seed=0)

    width_small = ci_small[1] - ci_small[0]
    width_large = ci_large[1] - ci_large[0]
    assert width_large < width_small


def test_bootstrap_ratio_ci_matches_point_estimate_when_all_users_identical():
    user_sum = np.full(20, 3.0)
    user_count = np.full(20, 5.0)
    ci_low, ci_high = bootstrap_ratio_ci(user_sum, user_count, n_bootstrap=500, seed=0)
    assert ci_low == ci_high == 0.6
