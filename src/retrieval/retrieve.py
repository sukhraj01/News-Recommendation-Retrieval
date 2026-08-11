"""Top-K BM25 retrieval against a BM25Index (ADR-005/ADR-006).

Scores via `score.py::score_all` (a sparse matrix-vector product against
the precomputed weight matrix — see index.py's module docstring for why:
the rank_bm25-native per-token Python loop is too slow at real
corpus/user-count scale).
"""
import numpy as np

from .index import BM25Index
from .score import score_all


def retrieve_top_k(index: BM25Index, query_tokens: list[str], k: int) -> list[str]:
    """Return up to `k` article_ids ranked by BM25 score, highest first.

    An empty query (true cold-start user, per ADR-005) has no terms to score
    against — returns `[]` rather than producing a degenerate all-zero-score
    ranking. Same for a query with no vocabulary overlap at all (checked by
    membership, not by whether the resulting scores happen to be zero — a
    matched term can still legitimately score exactly 0 under BM25Okapi's
    ATIRE idf when it appears in precisely half the corpus, per
    `tests/unit/test_retrieval.py`'s own docstring; that's a real, scored
    match, not an empty one). This early return is specific to
    `retrieve_top_k`'s "no candidates at all" contract; `score_all` itself
    returns an all-zero vector for the same input, since a fixed-candidate-
    list ranker (Q4's harness) has a different, legitimate use for that
    vector (a resolved tie-break), not an empty result.
    """
    if not query_tokens or k <= 0 or not index.article_ids:
        return []
    if not any(token in index.vocab for token in query_tokens):
        return []

    scores = score_all(index, query_tokens)
    k = min(k, len(scores))
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [index.article_ids[i] for i in top_idx]
