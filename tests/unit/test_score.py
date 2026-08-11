import numpy as np
import pandas as pd

from src.retrieval.index import build_index
from src.retrieval.score import BM25Scorer, score_all


def _index():
    # Same 5-doc corpus as test_retrieval.py (odd doc count avoids the
    # idf==0-at-N/2 coincidence — see that module's docstring).
    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3", "a4", "a5"],
        "title": ["football match today", "election results in", "football league news",
                  "weather forecast", "cooking recipe today"],
        "abstract": ["team wins big game", "president wins vote", "another football story",
                     "rain expected", "delicious pasta dish"],
    })
    return build_index(articles)


def test_id_to_col_matches_article_ids_positions():
    index = _index()
    for i, aid in enumerate(index.article_ids):
        assert index.id_to_col[aid] == i


def test_score_all_empty_query_returns_zero_vector_of_corpus_size():
    index = _index()
    scores = score_all(index, [])
    assert scores.shape == (5,)
    assert np.all(scores == 0)


def test_score_all_no_vocab_overlap_returns_zero_vector():
    index = _index()
    scores = score_all(index, ["zzznotinvocab"])
    assert np.all(scores == 0)


def test_bm25_scorer_subset_matches_score_all_at_same_positions():
    index = _index()
    scorer = BM25Scorer(index)
    query = ["football"]
    full = score_all(index, query)

    candidate_ids = ["a3", "a1", "a4"]  # deliberately out of index order
    subset = scorer.score(query, candidate_ids)

    expected = np.array([full[index.id_to_col[c]] for c in candidate_ids])
    np.testing.assert_array_equal(subset, expected)


def test_bm25_scorer_empty_query_returns_correctly_shaped_zero_array():
    index = _index()
    scorer = BM25Scorer(index)
    candidate_ids = ["a1", "a2", "a5"]
    subset = scorer.score([], candidate_ids)
    assert subset.shape == (3,)
    assert np.all(subset == 0)


def test_bm25_scorer_caches_by_query_identity_not_equality():
    index = _index()
    scorer = BM25Scorer(index)
    query = ["football"]

    first = scorer.score(query, ["a1"])
    cached_scores_obj = scorer._cached_scores
    # Same list object -> cache hit, no recompute (identity check).
    scorer.score(query, ["a3"])
    assert scorer._cached_scores is cached_scores_obj

    # A different (but equal-valued) list object -> cache miss, recompute.
    scorer.score(["football"], ["a1"])
    assert scorer._cached_scores is not cached_scores_obj
    # But the recomputed value is still correct.
    np.testing.assert_array_equal(scorer.score(query, ["a1"]), first)


def test_bm25_scorer_matches_retrieve_top_k_ranking():
    from src.retrieval.retrieve import retrieve_top_k

    index = _index()
    scorer = BM25Scorer(index)
    query = ["football"]

    top2 = retrieve_top_k(index, query, k=2)
    all_candidates = index.article_ids
    scores = scorer.score(query, all_candidates)
    ranked = [aid for aid, _ in sorted(zip(all_candidates, scores), key=lambda x: -x[1])][:2]
    assert set(ranked) == set(top2)
