"""Candidates D/F (2026-08-21 second session): unit coverage for
`src/retrieval/features.py`'s pure symbolic-matching functions. End-to-end
scoring is exercised by the real runs recorded under `experiments/`."""
import json

import pandas as pd
import pytest

from src.retrieval.features import (
    HistoryProfile,
    article_entity_set,
    build_history_profile,
    build_lookup_tables,
    category_match_score,
    entity_overlap_count,
    subcategory_match_score,
)


def _entities_json(*wikidata_ids):
    return json.dumps({
        "title_entities": [{"WikidataId": w, "Confidence": 1.0} for w in wikidata_ids],
        "abstract_entities": [],
    })


def test_article_entity_set_deduplicates_repeated_mentions():
    raw = json.dumps({
        "title_entities": [{"WikidataId": "Q1", "Confidence": 1.0}],
        "abstract_entities": [{"WikidataId": "Q1", "Confidence": 0.5}],
    })
    assert article_entity_set(raw) == frozenset({"Q1"})


def test_article_entity_set_empty_returns_empty_frozenset():
    assert article_entity_set(_entities_json()) == frozenset()


def test_build_history_profile_counts_categories_and_entities():
    category_lookup = {"a1": "sports", "a2": "sports", "a3": "news"}
    subcategory_lookup = {"a1": "football", "a2": "football", "a3": "politics"}
    entity_lookup = {
        "a1": frozenset({"Q1", "Q2"}),
        "a2": frozenset({"Q1"}),
        "a3": frozenset(),
    }
    profile = build_history_profile(["a1", "a2", "a3"], category_lookup, subcategory_lookup, entity_lookup)
    assert profile.n_articles == 3
    assert profile.category_counts == {"sports": 2, "news": 1}
    assert profile.subcategory_counts == {"football": 2, "politics": 1}
    assert profile.entity_counts == {"Q1": 2, "Q2": 1}


def test_build_history_profile_skips_missing_lookups_independently():
    # a2 is known to category_lookup but absent from subcategory_lookup and
    # entity_lookup - each dimension degrades independently, not all-or-nothing.
    category_lookup = {"a1": "sports", "a2": "news"}
    subcategory_lookup = {"a1": "football"}
    entity_lookup = {"a1": frozenset({"Q1"})}
    profile = build_history_profile(["a1", "a2"], category_lookup, subcategory_lookup, entity_lookup)
    assert profile.n_articles == 2
    assert profile.category_counts == {"sports": 1, "news": 1}
    assert profile.subcategory_counts == {"football": 1}
    assert profile.entity_counts == {"Q1": 1}


def test_category_match_score_is_fraction_of_history():
    profile = HistoryProfile(category_counts={"sports": 3}, n_articles=4)
    assert category_match_score("sports", profile) == pytest.approx(0.75)
    assert category_match_score("news", profile) == 0.0


def test_category_match_score_empty_history_or_none_category_is_zero():
    assert category_match_score("sports", HistoryProfile()) == 0.0
    assert category_match_score(None, HistoryProfile(category_counts={"sports": 1}, n_articles=1)) == 0.0


def test_subcategory_match_score_is_fraction_of_history():
    profile = HistoryProfile(subcategory_counts={"football": 1}, n_articles=2)
    assert subcategory_match_score("football", profile) == pytest.approx(0.5)


def test_entity_overlap_count_sums_shared_entity_history_counts():
    profile = HistoryProfile(entity_counts={"Q1": 3, "Q2": 1})
    assert entity_overlap_count(frozenset({"Q1", "Q2"}), profile) == 4
    assert entity_overlap_count(frozenset({"Q1", "Q99"}), profile) == 3


def test_entity_overlap_count_no_overlap_is_zero():
    profile = HistoryProfile(entity_counts={"Q1": 3})
    assert entity_overlap_count(frozenset({"Q99"}), profile) == 0
    assert entity_overlap_count(frozenset(), profile) == 0


def test_build_lookup_tables_builds_all_three_from_articles_df():
    articles = pd.DataFrame({
        "article_id": ["a1", "a2"],
        "category": ["sports", "news"],
        "subcategory": ["football", "politics"],
        "entities": [_entities_json("Q1"), _entities_json()],
    })
    category_lookup, subcategory_lookup, entity_set_lookup = build_lookup_tables(articles)
    assert category_lookup == {"a1": "sports", "a2": "news"}
    assert subcategory_lookup == {"a1": "football", "a2": "politics"}
    assert entity_set_lookup == {"a1": frozenset({"Q1"}), "a2": frozenset()}
