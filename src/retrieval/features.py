"""Symbolic (non-vector) candidate-scoring features for MIND, built from
`news.tsv`'s `category`/`subcategory`/entity fields — genuinely distinct
mechanisms from Candidate A's dense TransE cosine similarity (2026-08-21
session's Candidates D/F, see PROJECT_STATE.md's "Current Objective").

Where `entities.py` asks "how similar are these two dense vectors,"
this module asks "does this candidate literally share a category or a
named entity with something in this user's history" — symbolic identity
match, not distance in embedding space. Deliberately reuses
`entities.py::parse_entity_mentions` for entity extraction (same parsing,
different use: identity sets here, not confidence-weighted vectors).
"""
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from .entities import parse_entity_mentions


def article_entity_set(entities_json: str | float) -> frozenset[str]:
    """One article's linked Wikidata entity ids, deduplicated (a mention
    repeated across title/abstract counts once here — unlike
    `entities.py`'s confidence-weighted pooling, this module only asks
    "is this entity present," not "how strongly/how often")."""
    return frozenset(wid for wid, _ in parse_entity_mentions(entities_json))


@dataclass
class HistoryProfile:
    """One user's history, summarized as counts for symbolic matching —
    the counterpart to `embed.py`'s dense mean-pooled vector, built from
    the same history but never touching a vector."""
    category_counts: Counter = field(default_factory=Counter)
    subcategory_counts: Counter = field(default_factory=Counter)
    entity_counts: Counter = field(default_factory=Counter)  # entity_id -> # history articles containing it
    n_articles: int = 0


def build_history_profile(
    article_ids: list[str],
    category_lookup: dict[str, str],
    subcategory_lookup: dict[str, str],
    entity_set_lookup: dict[str, frozenset[str]],
) -> HistoryProfile:
    """`*_lookup` dicts map article_id -> that article's category /
    subcategory / entity set. Unknown article_ids (not in a lookup) are
    skipped for that lookup only, same "skip what's missing" treatment as
    `build_user_query`/`build_user_embedding_query` — a history article
    absent from one lookup doesn't disqualify it from the others.
    """
    profile = HistoryProfile()
    for aid in article_ids:
        profile.n_articles += 1
        if aid in category_lookup:
            profile.category_counts[category_lookup[aid]] += 1
        if aid in subcategory_lookup:
            profile.subcategory_counts[subcategory_lookup[aid]] += 1
        if aid in entity_set_lookup:
            for eid in entity_set_lookup[aid]:
                profile.entity_counts[eid] += 1
    return profile


def category_match_score(category: str | None, profile: HistoryProfile) -> float:
    """Fraction of the user's history sharing `category` — a soft [0, 1]
    match, not a binary in/out-of-set flag, so a category the user reads
    constantly scores higher than one seen once. 0.0 for an empty history
    or unknown category (`None` or absent from history), never NaN/error.
    """
    if not category or profile.n_articles == 0:
        return 0.0
    return profile.category_counts.get(category, 0) / profile.n_articles


def subcategory_match_score(subcategory: str | None, profile: HistoryProfile) -> float:
    if not subcategory or profile.n_articles == 0:
        return 0.0
    return profile.subcategory_counts.get(subcategory, 0) / profile.n_articles


def entity_overlap_count(candidate_entities: frozenset[str], profile: HistoryProfile) -> int:
    """Sum, over entities the candidate shares with the user's history, of
    how many separate history articles mentioned that entity — an entity
    that recurs across many of the user's past clicks counts for more than
    one seen once, while staying a plain symbolic count (no vectors,
    no distance metric). 0 for no shared entities (including an
    entity-free candidate or a history with no linked entities at all).
    """
    return sum(profile.entity_counts.get(eid, 0) for eid in candidate_entities)


def build_lookup_tables(
    articles: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], dict[str, frozenset[str]]]:
    """One pass over `articles` (must have `article_id`/`category`/
    `subcategory`/`entities` columns) building the three lookups
    `build_history_profile` and the per-candidate score functions need.
    Factored out so both the train-side (feature building) and dev-side
    (scoring) callers build these identically, once each, rather than
    duplicating the zip/dict-comprehension logic.
    """
    category_lookup = dict(zip(articles["article_id"], articles["category"]))
    subcategory_lookup = dict(zip(articles["article_id"], articles["subcategory"]))
    entity_set_lookup = {
        aid: article_entity_set(e) for aid, e in zip(articles["article_id"], articles["entities"])
    }
    return category_lookup, subcategory_lookup, entity_set_lookup
