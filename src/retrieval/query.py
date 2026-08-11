"""User query construction from click history, per ADR-005.

Unweighted concatenation of every history article's title+abstract text.
No recency weighting: ADR-002 flags MIND's history order as "presumed
chronological, not independently verified," and a recency-weighted query
would silently break if that assumption is wrong. Empty history (true
cold-start) returns an empty token list rather than raising — retrieval
must handle that explicitly, not treat it as an error.
"""
from .tokenize import tokenize


def build_user_query(article_ids: list[str], text_lookup: dict[str, str]) -> list[str]:
    """`text_lookup` maps article_id -> `title + " " + abstract`."""
    tokens: list[str] = []
    for article_id in article_ids:
        text = text_lookup.get(article_id)
        if text:
            tokens.extend(tokenize(text))
    return tokens
