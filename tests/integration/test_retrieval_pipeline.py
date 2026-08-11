"""End-to-end retrieval: build a real BM25 index from the feature store and
retrieve for a real user history. Runs against the same fast-tier
(MINDsmall + ebnerd_demo) data built by the `processed_dir` fixture in
conftest.py, per ADR-005/ADR-006's decision to index each split's own
article catalog.
"""
import pandas as pd

from src.retrieval.embed import build_embedding_index, build_user_embedding_query
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.retrieve import embed_retrieve_top_k, retrieve_top_k


def _text_lookup(articles: pd.DataFrame) -> dict[str, str]:
    return dict(zip(articles["article_id"], articles["title"].fillna("") + " " + articles["abstract"].fillna("")))


def test_mind_dev_retrieval_returns_valid_article_ids(processed_dir):
    articles = pd.read_parquet(processed_dir / "mind" / "small" / "dev" / "articles.parquet")
    history = pd.read_parquet(processed_dir / "mind" / "small" / "dev" / "user_history.parquet")

    index = build_index(articles)
    lookup = _text_lookup(articles)

    warm_user = history[history["article_ids"].apply(len) >= 5].iloc[0]
    query = build_user_query(warm_user["article_ids"], lookup)
    results = retrieve_top_k(index, query, k=50)

    assert len(results) == 50
    assert set(results).issubset(set(articles["article_id"]))


def test_ebnerd_validation_retrieval_returns_valid_article_ids(processed_dir):
    articles = pd.read_parquet(processed_dir / "ebnerd" / "demo" / "articles.parquet")
    history = pd.read_parquet(processed_dir / "ebnerd" / "demo" / "validation" / "user_history.parquet")

    index = build_index(articles)
    lookup = _text_lookup(articles)

    user = history.iloc[0]
    query = build_user_query(user["article_ids"], lookup)
    results = retrieve_top_k(index, query, k=50)

    assert len(results) == 50
    assert set(results).issubset(set(articles["article_id"]))


def test_cold_start_user_with_empty_history_retrieves_nothing(processed_dir):
    """MIND dev has real zero-history users (per ADR-005's empirical
    evidence) — BM25 structurally cannot retrieve for them."""
    history = pd.read_parquet(processed_dir / "mind" / "small" / "dev" / "user_history.parquet")
    articles = pd.read_parquet(processed_dir / "mind" / "small" / "dev" / "articles.parquet")

    cold_user = history[history["article_ids"].apply(len) == 0].iloc[0]
    index = build_index(articles)
    lookup = _text_lookup(articles)

    query = build_user_query(cold_user["article_ids"], lookup)
    assert query == []
    assert retrieve_top_k(index, query, k=50) == []


def test_retrieval_recovers_topically_similar_article():
    """Small, hand-built sanity check independent of real-data noise: a
    user whose entire history is football articles should retrieve a
    football article over an unrelated weather article."""
    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3"],
        "title": ["Local team wins football championship", "Second football match report", "Weather forecast for the week"],
        "abstract": ["A thrilling football game.", "Another football recap.", "Rain expected all week."],
    })
    index = build_index(articles)
    lookup = _text_lookup(articles)

    query = build_user_query(["a1"], lookup)
    results = retrieve_top_k(index, query, k=2)

    assert results == ["a1", "a2"]


def test_embed_retrieval_end_to_end_on_real_article_sample(processed_dir):
    """Real (not stubbed) encoder, per ADR-008: a small slice of real
    MINDsmall-dev articles (not the full ~42k corpus — that's covered by
    `scripts/run_embed_experiment.py`'s production run, not a unit-speed
    test) to keep this fast while still exercising the real model end to
    end, not a mock."""
    articles = pd.read_parquet(processed_dir / "mind" / "small" / "dev" / "articles.parquet").head(30)

    index = build_embedding_index(articles)
    assert index.vectors.shape[0] == 30

    vector_lookup = dict(zip(index.article_ids, index.vectors))
    history_ids = index.article_ids[:2]
    query = build_user_embedding_query(history_ids, vector_lookup)
    assert query is not None

    results = embed_retrieve_top_k(index, query, k=5)
    assert len(results) == 5
    assert set(results).issubset(set(index.article_ids))


def test_embed_retrieval_cold_start_user_retrieves_nothing():
    articles = pd.DataFrame({
        "article_id": ["a1", "a2"],
        "title": ["Local team wins football championship", "Weather forecast for the week"],
        "abstract": ["A thrilling football game.", "Rain expected all week."],
    })
    index = build_embedding_index(articles)
    query = build_user_embedding_query([], {})
    assert query is None
    assert embed_retrieve_top_k(index, query, k=5) == []
