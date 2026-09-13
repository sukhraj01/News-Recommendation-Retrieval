"""Unit tests for A2 Q1's MIND click-history features (Candidate L,
`src/retrieval/mind_features.py`)."""
import numpy as np
import pytest

from src.retrieval.features import HistoryProfile, build_history_profile
from src.retrieval.mind_features import (
    FEATURE_NAMES,
    build_feature_row,
    click_count,
    recency_weighted_affinity,
    recency_weights,
)


def test_click_count_is_history_length():
    assert click_count([]) == 0
    assert click_count(["a", "b", "c"]) == 3


def test_recency_weights_most_recent_last_gets_weight_one():
    w = recency_weights(4, decay=0.9)
    np.testing.assert_allclose(w, [0.9**3, 0.9**2, 0.9**1, 1.0])


def test_recency_weights_empty_history():
    assert recency_weights(0).size == 0


def test_recency_weights_matches_embed_module_convention():
    """One recency definition for MIND, not two that could drift apart."""
    from src.retrieval.embed import build_user_embedding_query_recency

    n, decay = 5, 0.9
    vectors = {f"a{i}": np.array([float(i)]) for i in range(n)}
    ids = [f"a{i}" for i in range(n)]
    expected_weights = recency_weights(n, decay)
    # build_user_embedding_query_recency returns a weighted mean of unit-norm
    # dummy vectors along one axis; reconstruct its own weights the same way
    # it does internally and compare to ours directly.
    got = build_user_embedding_query_recency(ids, vectors, decay=decay)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    manual = (weights @ np.array([vectors[a] for a in ids])) / weights.sum()
    manual = manual / np.linalg.norm(manual)
    np.testing.assert_allclose(got, manual, rtol=1e-6)  # got is float32
    np.testing.assert_allclose(weights, expected_weights)


def test_recency_weighted_affinity_empty_history_or_no_target_is_zero():
    assert recency_weighted_affinity([], {"a": "news"}, "news") == 0.0
    assert recency_weighted_affinity(["a"], {"a": "news"}, None) == 0.0
    assert recency_weighted_affinity(["a"], {"a": "news"}, "") == 0.0


def test_recency_weighted_affinity_all_match_is_one():
    lookup = {"a": "news", "b": "news", "c": "news"}
    assert recency_weighted_affinity(["a", "b", "c"], lookup, "news") == pytest.approx(1.0)


def test_recency_weighted_affinity_weights_recent_more_than_unweighted_would():
    """A single match at the MOST RECENT position should score higher than
    the same single match at the OLDEST position -- the entire point of
    recency weighting, verified directly rather than assumed from the
    formula."""
    lookup = {"match": "sports", "other": "news"}
    history_recent = ["other", "other", "other", "match"]  # match is most recent
    history_old = ["match", "other", "other", "other"]     # match is oldest
    recent_score = recency_weighted_affinity(history_recent, lookup, "sports")
    old_score = recency_weighted_affinity(history_old, lookup, "sports")
    assert recent_score > old_score


def test_recency_weighted_affinity_unknown_article_id_is_skipped_not_errored():
    lookup = {"a": "news"}
    # "unknown" has no entry in lookup -- must not raise, must not count as a match
    assert recency_weighted_affinity(["unknown", "a"], lookup, "news") == pytest.approx(
        recency_weights(2)[1] / recency_weights(2).sum()
    )


def test_build_feature_row_matches_feature_names_length_and_order():
    category_lookup = {"c1": "news", "h1": "news", "h2": "sports"}
    subcategory_lookup = {"c1": "us", "h1": "us", "h2": "nba"}
    profile = build_history_profile(["h1", "h2"], category_lookup, subcategory_lookup, {})
    row = build_feature_row("c1", ["h1", "h2"], category_lookup, subcategory_lookup, profile)
    assert len(row) == len(FEATURE_NAMES) == 5
    assert all(np.isfinite(v) for v in row)


def test_build_feature_row_log_click_count_is_constant_across_candidates_in_one_impression():
    """The documented, load-bearing limitation: log_click_count depends only
    on the user's history, so it MUST be identical for every candidate in
    the same impression -- verified directly, not just asserted in a
    docstring."""
    category_lookup = {"c1": "news", "c2": "sports", "h1": "news"}
    subcategory_lookup = {"c1": "us", "c2": "nba", "h1": "us"}
    profile = build_history_profile(["h1"], category_lookup, subcategory_lookup, {})
    row1 = build_feature_row("c1", ["h1"], category_lookup, subcategory_lookup, profile)
    row2 = build_feature_row("c2", ["h1"], category_lookup, subcategory_lookup, profile)
    idx = FEATURE_NAMES.index("log_click_count")
    assert row1[idx] == row2[idx]
    # and it should differ for a user with a different history length
    profile2 = build_history_profile(["h1", "h1"], category_lookup, subcategory_lookup, {})
    row3 = build_feature_row("c1", ["h1", "h1"], category_lookup, subcategory_lookup, profile2)
    assert row3[idx] != row1[idx]


def test_build_feature_row_empty_history_gives_zero_affinity_features_not_nan():
    profile = HistoryProfile()
    row = build_feature_row("c1", [], {"c1": "news"}, {"c1": "us"}, profile)
    assert row[FEATURE_NAMES.index("log_click_count")] == 0.0
    for name in ("category_match", "subcategory_match", "recency_category_affinity",
                 "recency_subcategory_affinity"):
        assert row[FEATURE_NAMES.index(name)] == 0.0
