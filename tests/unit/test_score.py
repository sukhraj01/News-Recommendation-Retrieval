import numpy as np
import pandas as pd

from src.retrieval.embed import EmbeddingIndex
from src.retrieval.index import build_index
from src.retrieval.score import BM25Scorer, EmbeddingScorer, score_all


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


def _embedding_index():
    # 3 docs, 2-dim unit vectors: a1/a2 near-identical direction, a3 orthogonal.
    article_ids = ["a1", "a2", "a3"]
    vectors = np.array([
        [1.0, 0.0],
        [0.9, np.sqrt(1 - 0.9 ** 2)],
        [0.0, 1.0],
    ], dtype=np.float32)
    return EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=vectors,
    )


def test_embedding_scorer_ranks_by_cosine_similarity():
    index = _embedding_index()
    scorer = EmbeddingScorer(index)
    query = np.array([1.0, 0.0], dtype=np.float32)  # matches a1 exactly
    scores = scorer.score(query, ["a1", "a2", "a3"])
    # a1 (identical direction) > a2 (close) > a3 (orthogonal)
    assert scores[0] > scores[1] > scores[2]
    np.testing.assert_allclose(scores[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(scores[2], 0.0, atol=1e-6)


def test_embedding_scorer_subset_matches_full_at_same_positions():
    index = _embedding_index()
    scorer = EmbeddingScorer(index)
    query = np.array([0.0, 1.0], dtype=np.float32)
    full = scorer.score(query, index.article_ids)

    candidate_ids = ["a3", "a1"]  # deliberately out of index order
    subset = scorer.score(query, candidate_ids)
    expected = np.array([full[index.id_to_col[c]] for c in candidate_ids])
    np.testing.assert_array_equal(subset, expected)


def test_embedding_scorer_none_query_returns_zero_tie():
    index = _embedding_index()
    scorer = EmbeddingScorer(index)
    candidate_ids = ["a1", "a2", "a3"]
    scores = scorer.score(None, candidate_ids)
    assert scores.shape == (3,)
    assert np.all(scores == 0)


def test_embedding_scorer_caches_by_query_identity_not_equality():
    index = _embedding_index()
    scorer = EmbeddingScorer(index)
    query = np.array([1.0, 0.0], dtype=np.float32)

    scorer.score(query, ["a1"])
    cached_scores_obj = scorer._cached_scores
    scorer.score(query, ["a2"])
    assert scorer._cached_scores is cached_scores_obj  # same object -> cache hit

    equal_but_different_query = np.array([1.0, 0.0], dtype=np.float32)
    scorer.score(equal_but_different_query, ["a1"])
    assert scorer._cached_scores is not cached_scores_obj  # different object -> recompute


def test_embedding_scorer_none_query_cache_does_not_collide_with_real_query():
    index = _embedding_index()
    scorer = EmbeddingScorer(index)

    none_scores = scorer.score(None, ["a1", "a2", "a3"])
    assert np.all(none_scores == 0)

    real_scores = scorer.score(np.array([1.0, 0.0], dtype=np.float32), ["a1", "a2", "a3"])
    assert not np.all(real_scores == 0)

    # Calling with None again after a real query must recompute the tie,
    # not reuse the real query's cached scores.
    none_again = scorer.score(None, ["a1", "a2", "a3"])
    assert np.all(none_again == 0)
