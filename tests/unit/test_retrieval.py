import numpy as np
import pandas as pd

from src.retrieval.embed import EmbeddingIndex
from src.retrieval.index import build_index
from src.retrieval.retrieve import embed_retrieve_top_k, retrieve_top_k


def _index():
    # 5 docs (not 4): rank_bm25's BM25Okapi uses the ATIRE idf variant with
    # no "+1" smoothing, so a term appearing in exactly N/2 docs gets an
    # idf of exactly zero (log((N-n+0.5)/(n+0.5)) == 0 when n == N/2) and
    # contributes nothing to the score. An odd doc count avoids that
    # coincidence so "football" (in 2 of 5 docs) still has nonzero idf.
    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3", "a4", "a5"],
        "title": ["football match today", "election results in", "football league news",
                  "weather forecast", "cooking recipe today"],
        "abstract": ["team wins big game", "president wins vote", "another football story",
                     "rain expected", "delicious pasta dish"],
    })
    return build_index(articles)


def test_retrieve_top_k_ranks_most_relevant_first():
    index = _index()
    results = retrieve_top_k(index, ["football"], k=2)
    assert set(results) == {"a1", "a3"}


def test_retrieve_top_k_respects_k():
    index = _index()
    assert len(retrieve_top_k(index, ["football"], k=1)) == 1
    assert len(retrieve_top_k(index, ["wins"], k=4)) == 4


def test_retrieve_top_k_empty_query_returns_empty():
    index = _index()
    assert retrieve_top_k(index, [], k=5) == []


def test_retrieve_top_k_larger_than_corpus_returns_all():
    index = _index()
    results = retrieve_top_k(index, ["football"], k=100)
    assert len(results) == 5


def test_retrieve_top_k_zero_or_negative_k_returns_empty():
    index = _index()
    assert retrieve_top_k(index, ["football"], k=0) == []
    assert retrieve_top_k(index, ["football"], k=-1) == []


def _embedding_index():
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


def test_embed_retrieve_top_k_ranks_by_cosine_similarity():
    index = _embedding_index()
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert embed_retrieve_top_k(index, query, k=2) == ["a1", "a2"]


def test_embed_retrieve_top_k_none_query_returns_empty():
    index = _embedding_index()
    assert embed_retrieve_top_k(index, None, k=5) == []


def test_embed_retrieve_top_k_respects_k():
    index = _embedding_index()
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert len(embed_retrieve_top_k(index, query, k=1)) == 1
    assert len(embed_retrieve_top_k(index, query, k=100)) == 3


def test_embed_retrieve_top_k_zero_or_negative_k_returns_empty():
    index = _embedding_index()
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert embed_retrieve_top_k(index, query, k=0) == []
    assert embed_retrieve_top_k(index, query, k=-1) == []
