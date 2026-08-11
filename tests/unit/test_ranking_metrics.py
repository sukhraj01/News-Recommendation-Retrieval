import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.ranking_metrics import (
    build_train_popularity,
    coverage,
    intra_list_diversity,
    mrr,
    ndcg_at_k,
    novelty,
    rank_candidates,
    ranking_metric_ci,
    safe_auc,
)


# --- nDCG ---

def test_ndcg_at_k_perfect_ranking_is_one():
    ranked = np.array([True, True, False, False])
    assert ndcg_at_k(ranked, 4) == pytest.approx(1.0)


def test_ndcg_at_k_worst_ranking_pinned():
    ranked = np.array([False, False, True, True])
    assert ndcg_at_k(ranked, 4) == pytest.approx(0.5706417189553201)


def test_ndcg_at_k_respects_cutoff():
    ranked = np.array([False, True, True, False])
    assert ndcg_at_k(ranked, 2) == pytest.approx(0.38685280723454163)


def test_ndcg_at_k_zero_positives_is_nan():
    ranked = np.array([False, False, False])
    assert np.isnan(ndcg_at_k(ranked, 3))


def test_ndcg_at_k_k_larger_than_list_uses_full_list():
    ranked = np.array([True, False])
    assert ndcg_at_k(ranked, 10) == ndcg_at_k(ranked, 2)


# --- MRR ---

def test_mrr_first_hit_at_position_two():
    ranked = np.array([False, True, False, True])
    assert mrr(ranked) == pytest.approx(0.5)


def test_mrr_first_hit_at_position_one():
    ranked = np.array([True, False, False])
    assert mrr(ranked) == pytest.approx(1.0)


def test_mrr_no_hits_is_zero():
    ranked = np.array([False, False])
    assert mrr(ranked) == 0.0


def test_mrr_multi_click_uses_first_hit_only():
    # Multi-click impressions are real in this project's data (up to 24
    # clicks in one MIND impression) — MRR must ignore later hits.
    ranked = np.array([False, True, True, True])
    assert mrr(ranked) == pytest.approx(0.5)


# --- AUC ---

def test_safe_auc_matches_sklearn_reference():
    scores = np.array([0.9, 0.1, 0.5, 0.2])
    clicked = np.array([True, False, True, False])
    expected = roc_auc_score(clicked, scores)
    assert safe_auc(scores, clicked) == pytest.approx(expected)


def test_safe_auc_all_clicked_is_nan():
    scores = np.array([0.9, 0.5, 0.2])
    clicked = np.array([True, True, True])
    assert np.isnan(safe_auc(scores, clicked))


def test_safe_auc_none_clicked_is_nan():
    scores = np.array([0.9, 0.5, 0.2])
    clicked = np.array([False, False, False])
    assert np.isnan(safe_auc(scores, clicked))


# --- tie-breaking ---

def test_rank_candidates_deterministic_same_seed_and_impression():
    scores = np.zeros(20)  # total tie -- exactly the cold-start scenario
    order1 = rank_candidates(scores, impression_id="imp-123", seed=0)
    order2 = rank_candidates(scores, impression_id="imp-123", seed=0)
    np.testing.assert_array_equal(order1, order2)


def test_rank_candidates_is_a_valid_permutation():
    scores = np.zeros(10)
    order = rank_candidates(scores, impression_id="imp-x", seed=0)
    assert sorted(order.tolist()) == list(range(10))


def test_rank_candidates_different_impressions_can_differ():
    scores = np.zeros(30)
    order_a = rank_candidates(scores, impression_id="imp-a", seed=0)
    order_b = rank_candidates(scores, impression_id="imp-b", seed=0)
    assert not np.array_equal(order_a, order_b)


def test_rank_candidates_respects_score_order_ignoring_ties():
    scores = np.array([1.0, 5.0, 3.0, 0.0])
    order = rank_candidates(scores, impression_id="imp-y", seed=0)
    assert list(order) == [1, 2, 0, 3]  # descending by score, no ties here


# --- diversity ---

def test_intra_list_diversity_known_pairs():
    # (a,a) same, (a,b) diff, (a,b) diff -> 2 diff / 3 pairs
    assert intra_list_diversity(["a", "a", "b"]) == pytest.approx(2 / 3)


def test_intra_list_diversity_all_same_category_is_zero():
    assert intra_list_diversity(["a", "a", "a"]) == 0.0


def test_intra_list_diversity_single_item_is_nan():
    assert np.isnan(intra_list_diversity(["a"]))


# --- novelty / train-only popularity ---

def test_build_train_popularity_smooths_unseen_article():
    train_impressions = pd.DataFrame({
        "article_id": ["x1", "x1", "x2"],
        "clicked": [True, True, True],
    })
    n_catalog = 10
    pop = build_train_popularity(train_impressions, n_catalog, alpha=1.0)
    total_clicks = 3.0
    denom = total_clicks + 1.0 * n_catalog
    assert pop["x1"] == pytest.approx((2 + 1.0) / denom)
    assert pop["x2"] == pytest.approx((1 + 1.0) / denom)
    # never seen in train -> smoothed floor, not zero/inf
    floor = 1.0 / denom
    assert pop["never_seen"] == pytest.approx(floor)


def test_novelty_never_seen_article_scores_higher_than_popular_one():
    train_impressions = pd.DataFrame({
        "article_id": ["popular"] * 50,
        "clicked": [True] * 50,
    })
    pop = build_train_popularity(train_impressions, n_catalog=100, alpha=1.0)
    novelty_popular = novelty(["popular"], pop)
    novelty_unseen = novelty(["never_seen_before"], pop)
    assert novelty_unseen > novelty_popular
    assert np.isfinite(novelty_unseen)  # smoothing prevents -log2(0) = inf


def test_novelty_empty_list_is_nan():
    assert np.isnan(novelty([], {}))


# --- coverage ---

def test_coverage_union_over_catalog_size():
    top_k = [["a1", "a2"], ["a2", "a3"], ["a4"]]
    assert coverage(top_k, catalog_size=10) == pytest.approx(4 / 10)


def test_coverage_full_catalog_is_one():
    top_k = [["a1", "a2"], ["a1", "a2"]]
    assert coverage(top_k, catalog_size=2) == pytest.approx(1.0)


# --- ranking_metric_ci (per-impression -> per-user bootstrap) ---

def test_ranking_metric_ci_drops_nan_and_counts_skipped():
    per_impression = pd.DataFrame({
        "user_id": ["A", "A", "B", "C"],
        "auc": [0.8, np.nan, 0.6, 0.9],
    })
    result = ranking_metric_ci(per_impression, "auc", n_bootstrap=200, seed=0)
    assert result.n_skipped == 1
    assert result.n_impressions == 3
    assert result.n_users == 3
    assert result.metric == pytest.approx((0.8 + 0.6 + 0.9) / 3)


def test_ranking_metric_ci_all_nan_returns_nan_result():
    per_impression = pd.DataFrame({
        "user_id": ["A", "B"],
        "auc": [np.nan, np.nan],
    })
    result = ranking_metric_ci(per_impression, "auc")
    assert np.isnan(result.metric)
    assert result.n_skipped == 2
    assert result.n_impressions == 0
