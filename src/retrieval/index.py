"""BM25 index construction over an `articles` table.

Per ADR-006: BM25Okapi is the chosen variant (BM25L/BM25+ solve a
long-document problem that doesn't apply to title+abstract text).

Scoring is precomputed as a sparse (docs x vocab) weight matrix rather than
relying on rank_bm25's own `get_scores`, which loops over query tokens in
pure Python and does an O(corpus_size) numpy pass per token. Measured
directly against MINDsmall-dev (42,416 docs, 50,000 users, ~500-token
average query): the naive per-user `get_scores` loop projected to several
hours. The weight matrix below reproduces rank_bm25's exact formula
(same idf/doc_freqs/doc_len/avgdl/k1/b — this is a performance rewrite, not
a different BM25 variant) so a query becomes one sparse matrix-vector
product instead of a per-token Python loop; see ADR-006's Benchmark Results.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from rank_bm25 import BM25Okapi

from .tokenize import tokenize


@dataclass
class BM25Index:
    article_ids: list[str]  # positional lookup: article_ids[i] <-> weights_t row-space index i
    vocab: dict[str, int]  # term -> column index into the weight matrix
    weights_t: sp.csr_matrix  # (vocab_size, n_docs) — precomputed BM25 doc-term weights, transposed
    id_to_col: dict[str, int]  # article_id -> column index into weights_t (inverse of article_ids)


def build_index(articles: pd.DataFrame) -> BM25Index:
    """Build a BM25 index over `title + " " + abstract` for every article.

    Per ADR-005/ADR-002, `title` and `abstract` are the schema's mandatory
    text fields, present (non-null, possibly empty string) for both datasets.
    """
    text = articles["title"].fillna("") + " " + articles["abstract"].fillna("")
    corpus = [tokenize(t) for t in text]
    bm25 = BM25Okapi(corpus)

    vocab = {term: i for i, term in enumerate(bm25.idf)}
    doc_len = np.array(bm25.doc_len, dtype=np.float64)
    # per-document denominator constant (term-frequency-independent part of
    # Okapi's saturation term), computed once per doc rather than per query
    c = bm25.k1 * (1 - bm25.b + bm25.b * doc_len / bm25.avgdl)

    rows, cols, data = [], [], []
    for d, freqs in enumerate(bm25.doc_freqs):
        for term, tf in freqs.items():
            rows.append(d)
            cols.append(vocab[term])
            data.append(bm25.idf[term] * (tf * (bm25.k1 + 1)) / (tf + c[d]))

    weights = sp.csr_matrix((data, (rows, cols)), shape=(len(corpus), len(vocab)))
    article_ids = articles["article_id"].tolist()
    return BM25Index(
        article_ids=article_ids,
        vocab=vocab,
        weights_t=weights.T.tocsr(),
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
    )
