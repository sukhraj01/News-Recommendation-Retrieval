"""Candidate A (2026-08-21 local-validation session): unit coverage for
`src/retrieval/entities.py`'s pure functions. End-to-end scoring is
exercised by the real run recorded under `experiments/`, not re-tested
here — same split of responsibility `test_leakage_ablation.py` uses."""
import json

import numpy as np
import pytest

from src.retrieval.entities import (
    build_article_entity_vector,
    build_entity_index,
    load_entity_vectors,
    parse_entity_mentions,
)


def _entities_json(title=None, abstract=None):
    return json.dumps({"title_entities": title or [], "abstract_entities": abstract or []})


def test_parse_entity_mentions_combines_title_and_abstract():
    raw = _entities_json(
        title=[{"WikidataId": "Q1", "Confidence": 1.0}],
        abstract=[{"WikidataId": "Q2", "Confidence": 0.5}],
    )
    assert parse_entity_mentions(raw) == [("Q1", 1.0), ("Q2", 0.5)]


def test_parse_entity_mentions_keeps_duplicate_mentions_as_multiset():
    raw = _entities_json(
        title=[{"WikidataId": "Q1", "Confidence": 1.0}],
        abstract=[{"WikidataId": "Q1", "Confidence": 0.9}],
    )
    assert parse_entity_mentions(raw) == [("Q1", 1.0), ("Q1", 0.9)]


def test_parse_entity_mentions_empty_entities_returns_empty_list():
    assert parse_entity_mentions(_entities_json()) == []


def test_parse_entity_mentions_null_input_returns_empty_list():
    assert parse_entity_mentions(None) == []
    assert parse_entity_mentions(float("nan")) == []


def test_parse_entity_mentions_malformed_json_returns_empty_list_not_raises():
    assert parse_entity_mentions("{not valid json") == []


def test_load_entity_vectors_parses_tab_separated_lines():
    raw = b"Q41\t1.0\t2.0\t3.0\nQ1860\t-1.0\t0.0\t0.5\n"
    vectors = load_entity_vectors(raw)
    assert set(vectors) == {"Q41", "Q1860"}
    np.testing.assert_allclose(vectors["Q41"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vectors["Q1860"], [-1.0, 0.0, 0.5])


def test_load_entity_vectors_tolerates_real_files_trailing_tab():
    # Real MIND `.vec` files end every line with a stray trailing tab
    # before the newline (confirmed against MINDsmall_dev's actual file).
    raw = b"Q41\t1.0\t2.0\t3.0\t\n"
    vectors = load_entity_vectors(raw)
    np.testing.assert_allclose(vectors["Q41"], [1.0, 2.0, 3.0])


def test_build_article_entity_vector_confidence_weighted():
    entity_vectors = {
        "Q1": np.array([1.0, 0.0], dtype=np.float32),
        "Q2": np.array([0.0, 1.0], dtype=np.float32),
    }
    # Q1 weighted 3x more than Q2 -> result should lean toward Q1's direction.
    vec = build_article_entity_vector([("Q1", 0.9), ("Q2", 0.3)], entity_vectors)
    assert vec is not None
    assert vec[0] > vec[1]
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)


def test_build_article_entity_vector_skips_unresolvable_entities():
    entity_vectors = {"Q1": np.array([1.0, 0.0], dtype=np.float32)}
    vec = build_article_entity_vector([("Q1", 1.0), ("Qmissing", 1.0)], entity_vectors)
    np.testing.assert_allclose(vec, [1.0, 0.0], atol=1e-6)


def test_build_article_entity_vector_no_mentions_returns_none():
    assert build_article_entity_vector([], {"Q1": np.array([1.0, 0.0])}) is None


def test_build_article_entity_vector_all_unresolvable_returns_none():
    assert build_article_entity_vector([("Qmissing", 1.0)], {"Q1": np.array([1.0, 0.0])}) is None


def test_build_article_entity_vector_zero_confidence_excluded():
    entity_vectors = {"Q1": np.array([1.0, 0.0], dtype=np.float32)}
    assert build_article_entity_vector([("Q1", 0.0)], entity_vectors) is None


def test_build_entity_index_coverage_and_zero_fill():
    import pandas as pd

    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3"],
        "entities": [
            _entities_json(title=[{"WikidataId": "Q1", "Confidence": 1.0}]),
            _entities_json(),  # no entities at all
            _entities_json(title=[{"WikidataId": "Qmissing", "Confidence": 1.0}]),  # unresolvable
        ],
    })
    entity_vectors = {"Q1": np.array([3.0, 4.0], dtype=np.float32)}

    index, coverage = build_entity_index(articles, entity_vectors)

    assert index.article_ids == ["a1", "a2", "a3"]
    assert coverage == pytest.approx(1 / 3)
    np.testing.assert_allclose(index.vectors[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(index.vectors[1], [0.0, 0.0])
    np.testing.assert_allclose(index.vectors[2], [0.0, 0.0])


def test_build_entity_index_empty_articles_coverage_is_nan():
    import pandas as pd

    articles = pd.DataFrame({"article_id": [], "entities": []})
    index, coverage = build_entity_index(articles, {})
    assert index.vectors.shape[0] == 0
    assert np.isnan(coverage)
