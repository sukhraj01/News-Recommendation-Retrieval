"""Unit coverage for `scripts/generate_mind_gated_predictions.py`'s
`GatedScorer`/`GatedQuery`/`build_query_by_user` — verified with small
synthetic fixtures before trusting the real ~2hr MINDlarge_test run
(2026-08-21, second MIND Codabench submission)."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from generate_mind_gated_predictions import GatedQuery, GatedScorer, build_query_by_user
from src.retrieval.embed import EmbeddingIndex
from src.retrieval.score import EmbeddingScorer
from src.retrieval.index import build_index
from src.retrieval.score import BM25Scorer


def _make_scorer(coef=None, intercept=0.0):
    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3"],
        "title": ["football match today", "election results in", "cooking recipe today"],
        "abstract": ["team wins big game", "president wins vote", "delicious pasta dish"],
        "category": ["sports", "news", "food"],
        "subcategory": ["football", "politics", "recipes"],
        "entities": ["{}", "{}", "{}"],
    })
    bm25_scorer = BM25Scorer(build_index(articles))

    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    index = EmbeddingIndex(
        article_ids=["a1", "a2", "a3"], id_to_col={"a1": 0, "a2": 1, "a3": 2}, vectors=vectors
    )
    embed_scorer = EmbeddingScorer(index)
    recency_scorer = EmbeddingScorer(index)

    n_features = 7
    coef = coef if coef is not None else np.zeros(n_features)
    scaler_mean = np.zeros(n_features)
    scaler_scale = np.ones(n_features)

    return GatedScorer(
        embed_scorer, bm25_scorer, recency_scorer,
        category_lookup={"a1": "sports", "a2": "news", "a3": "food"},
        subcategory_lookup={"a1": "football", "a2": "politics", "a3": "recipes"},
        entity_set_lookup={"a1": frozenset(), "a2": frozenset(), "a3": frozenset()},
        # Real `build_train_popularity` returns a defaultdict with a
        # Laplace-smoothed floor for unseen article_ids (ADR-007/009) — a
        # plain dict here would raise KeyError on a missing candidate
        # instead of exercising the real fallback behavior.
        popularity=defaultdict(lambda: 0.01, {"a1": 0.1, "a2": 0.2, "a3": 0.3}),
        scaler_mean=scaler_mean, scaler_scale=scaler_scale, coef=coef, intercept=intercept,
    )


def test_warm_query_scores_identically_to_plain_embedding_scorer():
    scorer = _make_scorer()
    embed_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query = GatedQuery(cohort="warm", embed_query=embed_query, bm25_query=[], recency_query=None, profile=None)

    scores = scorer.score(query, ["a1", "a2", "a3"])
    expected = scorer.embed_scorer.score(embed_query, ["a1", "a2", "a3"])
    np.testing.assert_allclose(scores, expected)


def test_cold_query_uses_combiner_not_raw_embedding_score():
    from src.retrieval.features import HistoryProfile

    # A non-zero coefficient vector so the combiner's output provably
    # differs from a plain embedding score.
    coef = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # weight only log_popularity
    scorer = _make_scorer(coef=coef, intercept=0.0)
    embed_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query = GatedQuery(
        cohort="cold", embed_query=embed_query, bm25_query=["football"],
        recency_query=embed_query, profile=HistoryProfile(),
    )

    scores = scorer.score(query, ["a1", "a2", "a3"])
    plain_embed = scorer.embed_scorer.score(embed_query, ["a1", "a2", "a3"])
    assert not np.allclose(scores, plain_embed)
    # With only log_popularity weighted, ranking should follow popularity
    # order (a3 > a2 > a1, per the fixture's popularity dict).
    assert scores[2] > scores[1] > scores[0]


def test_cold_query_missing_candidate_scores_neg_inf_not_nan():
    from src.retrieval.features import HistoryProfile

    # Mixed-sign coefficients, matching Candidate F's real fitted values
    # (bm25 +, embed +, recency -) — the exact shape that produces NaN if
    # a missing candidate's -inf features aren't handled explicitly.
    coef = np.array([0.076, 0.278, -0.014, 0.0, 0.0, 0.0, 0.0])
    scorer = _make_scorer(coef=coef, intercept=0.0)
    embed_query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    query = GatedQuery(
        cohort="cold", embed_query=embed_query, bm25_query=["football"],
        recency_query=embed_query, profile=HistoryProfile(),
    )

    # "missing" isn't in the fixture's 3-article index at all.
    scores = scorer.score(query, ["a1", "missing", "a2"])
    assert not np.isnan(scores).any()
    assert scores[1] == -np.inf
    assert np.isfinite(scores[0])
    assert np.isfinite(scores[2])


def test_build_query_by_user_routes_by_history_length():
    history = pd.DataFrame({
        "user_id": ["u_warm", "u_cold", "u_empty"],
        "article_ids": [["a1", "a2", "a3", "a1", "a2"], ["a1"], []],
    })
    vector_lookup = {
        "a1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "a2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "a3": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    text_lookup = {"a1": "football", "a2": "election", "a3": "cooking"}
    category_lookup = {"a1": "sports", "a2": "news", "a3": "food"}
    subcategory_lookup = {"a1": "football", "a2": "politics", "a3": "recipes"}
    entity_set_lookup = {"a1": frozenset(), "a2": frozenset(), "a3": frozenset()}

    query_by_user = build_query_by_user(
        history, vector_lookup, text_lookup, category_lookup, subcategory_lookup, entity_set_lookup
    )

    assert query_by_user["u_warm"].cohort == "warm"
    assert query_by_user["u_warm"].bm25_query == []  # never built for warm users
    assert query_by_user["u_warm"].recency_query is None

    assert query_by_user["u_cold"].cohort == "cold"
    assert query_by_user["u_cold"].bm25_query != []
    assert query_by_user["u_cold"].recency_query is not None

    assert query_by_user["u_empty"].cohort == "cold"
    assert query_by_user["u_empty"].embed_query is None  # true cold-start, no resolvable history
