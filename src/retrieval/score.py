"""Generic scoring layer: turn a query representation into per-candidate
scores, for both full-corpus retrieval (`retrieve.py::retrieve_top_k`) and
fixed-candidate-list ranking (Q4's evaluation harness).

`score_all` is extracted from what used to be inlined in `retrieve_top_k` —
"turn query tokens into a dense score-per-doc vector" is needed by both
callers, so it lives here once rather than being duplicated.

The `Scorer` Protocol is the seam a future second retrieval method plugs
into: BM25 is the first consumer, and neither the ranking metrics nor the
harness driver need to know which one they're calling. `query` is
deliberately typed `Any` since a future method's query representation
won't share BM25's token-list shape, only this call *contract*.
"""
from typing import Any, Protocol, Sequence

import numpy as np
import scipy.sparse as sp

from .index import BM25Index


def score_all(index: BM25Index, query_tokens: list[str]) -> np.ndarray:
    """Dense BM25 score for every doc in the index, in `article_ids` order.

    Empty query or no vocabulary overlap (a true cold-start user, per
    ADR-005) returns an all-zero vector, not an error — every candidate
    scoring equally is a legitimate result here (a *ranking* concern for
    whoever consumes this vector, e.g. needing a tie-break — not a scoring
    failure).
    """
    n_docs = len(index.article_ids)
    if n_docs == 0:
        return np.zeros(0)

    counts: dict[int, float] = {}
    for token in query_tokens:
        col = index.vocab.get(token)
        if col is None:
            continue
        counts[col] = counts.get(col, 0.0) + 1.0

    if not counts:
        return np.zeros(n_docs)

    cols = list(counts.keys())
    data = list(counts.values())
    query_vec = sp.csr_matrix((data, ([0] * len(cols), cols)), shape=(1, index.weights_t.shape[0]))
    return (query_vec @ index.weights_t).toarray().ravel()


class Scorer(Protocol):
    def score(self, query: Any, candidate_ids: Sequence[str]) -> np.ndarray:
        """One score per `candidate_id`, aligned by position, same length
        as `candidate_ids`."""
        ...


class BM25Scorer:
    """Scores a fixed candidate subset against a BM25 index.

    Caches the last full-corpus score vector by query-token-list identity
    (not equality — a fresh, cheap check) so that a driver processing a
    user's impressions consecutively (each call passing the *same* query
    list object for that user) scores the whole corpus once per user, then
    reuses it across that user's many impressions. This is a real hit, not
    a speculative optimization, because a user's BM25 score for an article
    is independent of which impression's candidate list the article
    appears in. If a caller doesn't group by user, this merely degrades to
    "no caching," never to incorrect scores.
    """

    def __init__(self, index: BM25Index):
        self.index = index
        self._cached_query: list[str] | None = None
        self._cached_scores: np.ndarray | None = None

    def score(self, query: list[str], candidate_ids: Sequence[str]) -> np.ndarray:
        if query is not self._cached_query:
            self._cached_scores = score_all(self.index, query)
            self._cached_query = query

        full = self._cached_scores
        if full is None or full.size == 0:
            return np.zeros(len(candidate_ids))

        cols = np.fromiter(
            (self.index.id_to_col[c] for c in candidate_ids),
            dtype=np.int64,
            count=len(candidate_ids),
        )
        return full[cols]
