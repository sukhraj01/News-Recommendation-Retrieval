"""No label at or after (impression_time - D) may influence a trailing feature (ADR-018 addendum)."""
import numpy as np
import pytest

from src.bandit.trailing import FEATURE_NAMES_TRAILING, smoothed_ctr, trailing_counts, trailing_feature_matrix


def _events(n=3000, n_art=25, span=20000, seed=0):
    rng = np.random.default_rng(seed)
    art = rng.integers(0, n_art, n)
    t = np.sort(rng.integers(0, span, n))
    lab = (rng.random(n) < 0.2).astype(np.int8)
    return art, t, lab


def brute(art, t, lab, D, W):
    c = np.zeros(len(art)); s = np.zeros(len(art))
    for i in range(len(art)):
        m = (art == art[i]) & (t >= t[i] - D - W) & (t < t[i] - D)
        s[i], c[i] = m.sum(), (lab[m] > 0).sum()
    return c, s


@pytest.mark.parametrize("D", [0, 900, 1800, 3600])
def test_matches_brute_force(D):
    art, t, lab = _events(n=1200)
    c, s = trailing_counts(art, t, lab, delay_s=D, window_s=3600)
    bc, bs = brute(art, t, lab, D, 3600)
    np.testing.assert_array_equal(c, bc)
    np.testing.assert_array_equal(s, bs)


@pytest.mark.parametrize("D", [0, 900, 1800, 3600])
def test_labels_at_or_after_t_minus_D_do_not_influence_features(D):
    """Flip EVERY label whose observation time is >= t_probe - D; the feature at t_probe must not move."""
    art, t, lab = _events(seed=1)
    base_c, base_s = trailing_counts(art, t, lab, delay_s=D, window_s=7200)
    rng = np.random.default_rng(5)
    for probe in rng.choice(len(art), 60, replace=False):
        tp = t[probe]
        flipped = lab.copy()
        late = t >= tp - D                      # includes the probe's own label when D = 0
        flipped[late] = 1 - flipped[late]
        c2, s2 = trailing_counts(art, t, flipped, delay_s=D, window_s=7200)
        assert c2[probe] == base_c[probe] and s2[probe] == base_s[probe]


def test_own_label_never_enters_even_with_zero_delay():
    art = np.array([1, 1, 1]); t = np.array([100, 100, 100]); lab = np.array([1, 0, 1], dtype=np.int8)
    c, s = trailing_counts(art, t, lab, delay_s=0, window_s=3600)
    assert (c == 0).all() and (s == 0).all()  # same-second siblings are excluded too (strict upper bound)


def test_window_lower_bound_and_article_isolation():
    art = np.array([0, 0, 1, 0]); t = np.array([0, 100, 100, 4000]); lab = np.array([1, 1, 1, 0], dtype=np.int8)
    c, s = trailing_counts(art, t, lab, delay_s=0, window_s=3600)
    assert (c[3], s[3]) == (0.0, 0.0)          # both article-0 events are older than 3600 s
    c, s = trailing_counts(art, t, lab, delay_s=0, window_s=5000)
    assert (c[3], s[3]) == (2.0, 2.0)          # window now reaches them; article 1's event is not counted
    assert (c[1], s[1]) == (1.0, 1.0)


def test_delay_hides_recent_events():
    art = np.zeros(3, dtype=np.int64); t = np.array([0, 1000, 1500]); lab = np.ones(3, dtype=np.int8)
    c0, _ = trailing_counts(art, t, lab, delay_s=0, window_s=3600)
    c1, _ = trailing_counts(art, t, lab, delay_s=900, window_s=3600)
    assert c0[2] == 2 and c1[2] == 1          # the click at t=1000 is only 500 s old at t=1500 < D


def test_feature_matrix_shape_names_and_finite():
    art, t, lab = _events(n=400)
    sizes = np.full(100, 4); starts = np.arange(0, 400, 4)

    def rank(v, gs, sz):
        out = np.zeros(len(v), dtype=np.float32)
        for a, b in zip(gs, sz):
            out[a:a + b] = np.argsort(np.argsort(v[a:a + b])) / (b - 1)
        return out

    F = trailing_feature_matrix(art, t, lab, starts, sizes, delay_s=900, rank_fn=rank)
    assert F.shape == (400, len(FEATURE_NAMES_TRAILING)) and np.isfinite(F).all()
    assert (smoothed_ctr(np.zeros(3), np.zeros(3)) == 0.1).all()
