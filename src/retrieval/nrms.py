"""Candidate J (ADR-012 addendum): NRMS-lite title/history encoders.

H1/H2/I/I-long/I-pop (ADR-010/ADR-011) all scored on top of FROZEN,
pre-computed sentence embeddings with a shallow head on top. This module is
a lightweight, from-scratch reproduction of NRMS's two-encoder design (Wu
et al. 2019, "Neural News Recommendation with Multi-Head Self-Attention"):
a TRAINABLE self-attention encoder over article title text, and a
self-attention encoder over the user's click-history SEQUENCE (not a static
mean-pooled feature) -- the two components every NAML/LSTUR/NRMS model
shares that Candidates H/I never included. See
`decisions/ADR-012_addendum_candidate_J_TEMPLATE.md` for the full rationale.

Training-loop/dataset-construction code stays in
`scripts/mind_nrms_lite_kaggle_run.py` (this project's existing split
between unit-tested `src/retrieval/` model code and `scripts/`-level
training loops, e.g. `rerank.py` vs. `run_attention_reranker_experiment.py`).
"""
import re
from collections import Counter
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str | None) -> list[str]:
    """Lowercase + punctuation-stripped word tokenizer, no external
    tokenizer dependency (matches this project's existing hand-built
    tokenization posture, e.g. `src/retrieval/tokenize.py`'s BM25 tokenizer,
    rather than pulling in a new NLP library this late)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def build_vocab(titles: Iterable[str], min_freq: int = 2) -> dict[str, int]:
    """Vocab built from title text only, no label information -- token
    coverage, not leakage, the same posture BM25's corpus-wide index
    already takes (ADR-012 addendum). `PAD` is always id 0 (relied on by
    `encode_title`'s padding and every model class's `pad_id` default)."""
    counter: Counter = Counter()
    for title in titles:
        counter.update(tokenize(title))
    vocab = [PAD, UNK] + [w for w, c in counter.most_common() if c >= min_freq]
    return {w: i for i, w in enumerate(vocab)}


def encode_title(title: str | None, word2id: dict[str, int], max_len: int) -> list[int]:
    """Fixed-length token-id sequence, truncated/right-padded to `max_len`.
    Unknown tokens map to `UNK`; a `None`/empty title encodes to all-PAD --
    the same all-padded row `NewsEncoder`/`AdditiveAttention` must handle
    without producing NaN (real MIND data has empty titles for some
    articles)."""
    toks = tokenize(title)[:max_len]
    ids = [word2id.get(t, word2id[UNK]) for t in toks]
    ids += [word2id[PAD]] * (max_len - len(ids))
    return ids


def _unmask_one_row(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`mask`: (batch, seq) True = PAD. Returns `(safe_mask, fully_masked)`
    where `safe_mask` unmasks position 0 for any row that was entirely
    masked, and `fully_masked`: (batch,) flags which rows those were.

    This exists because forward-only `nan_to_num` is NOT sufficient for an
    all-True mask row: `softmax(-inf, ..., -inf) = nan` is a real node in
    the autograd graph, and multiplying that NaN by a zero attention
    weight in the backward pass is still NaN (IEEE 754: `0 * nan = nan`) --
    confirmed with a live forward+backward run on this project's installed
    torch (2.13.0; see `tests/unit/test_nrms.py`), not just the numpy
    forward-only simulation this candidate was originally verified with.
    The fix has to prevent the NaN from ever being computed, not clean it
    up afterward: give every row at least one unmasked (fake) key so
    softmax never sees an all -inf row, then zero the RESULT for those
    rows with a plain finite multiply, which has a well-defined (zero)
    gradient."""
    fully_masked = mask.all(dim=-1)  # (batch,)
    if not fully_masked.any():
        return mask, fully_masked
    safe_mask = mask.clone()
    safe_mask[fully_masked, 0] = False
    return safe_mask, fully_masked


class AdditiveAttention(nn.Module):
    """Pools a sequence of vectors into one, weighting by a learned query
    (Wu et al. 2019, Eq. 5-7). Used both to pool title-word vectors into a
    news vector, and to pool history news-vectors into a user vector."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.query = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, seq, dim); key_padding_mask: (batch, seq) True = PAD
        scores = self.query(torch.tanh(self.proj(x))).squeeze(-1)  # (batch, seq)
        fully_masked = None
        if key_padding_mask is not None:
            safe_mask, fully_masked = _unmask_one_row(key_padding_mask)
            scores = scores.masked_fill(safe_mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (batch, dim)
        if fully_masked is not None and fully_masked.any():
            pooled = pooled * (~fully_masked).unsqueeze(-1).to(pooled.dtype)
        return pooled


class NewsEncoder(nn.Module):
    """Title self-attention encoder: word embeddings -> multi-head
    self-attention -> additive-attention pooling into one news vector."""

    def __init__(
        self, vocab_size: int, pad_id: int = 0, embed_dim: int = 128,
        num_heads: int = 8, dropout: float = 0.2,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool = AdditiveAttention(embed_dim)

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        # title_ids: (batch, seq_len)
        pad_mask = title_ids.eq(self.pad_id)  # (batch, seq) True = PAD
        safe_mask, fully_masked = _unmask_one_row(pad_mask)
        x = self.dropout(self.embed(title_ids))
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=safe_mask)
        if fully_masked.any():
            attn_out = attn_out * (~fully_masked).unsqueeze(-1).unsqueeze(-1).to(attn_out.dtype)
        news_vec = self.pool(attn_out, key_padding_mask=pad_mask)
        return news_vec


class UserEncoder(nn.Module):
    """History self-attention encoder: a user's sequence of news vectors ->
    multi-head self-attention -> additive-attention pooling into one user
    vector. Same fully-masked-row risk as `NewsEncoder` -- real here too,
    since zero-history users make `hist_pad_mask` all-True for that row."""

    def __init__(self, news_dim: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(news_dim, num_heads, dropout=dropout, batch_first=True)
        self.pool = AdditiveAttention(news_dim)

    def forward(self, hist_news_vecs: torch.Tensor, hist_pad_mask: torch.Tensor) -> torch.Tensor:
        safe_mask, fully_masked = _unmask_one_row(hist_pad_mask)
        attn_out, _ = self.self_attn(hist_news_vecs, hist_news_vecs, hist_news_vecs,
                                      key_padding_mask=safe_mask)
        if fully_masked.any():
            attn_out = attn_out * (~fully_masked).unsqueeze(-1).unsqueeze(-1).to(attn_out.dtype)
        return self.pool(attn_out, key_padding_mask=hist_pad_mask)


class NRMSLite(nn.Module):
    """Wu et al. 2019's two-encoder design, sized for a single Kaggle GPU
    session rather than the paper's full compute budget (no pretrained word
    embeddings, smaller embed_dim/head count -- see ADR-012 addendum's
    Trade-offs section)."""

    def __init__(
        self, vocab_size: int, pad_id: int = 0, embed_dim: int = 128,
        num_heads: int = 8, dropout: float = 0.2,
    ):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, pad_id, embed_dim, num_heads, dropout)
        self.user_encoder = UserEncoder(embed_dim, num_heads, dropout)

    def forward(
        self, hist_title_ids: torch.Tensor, hist_pad_mask: torch.Tensor, cand_title_ids: torch.Tensor,
    ) -> torch.Tensor:
        # hist_title_ids: (batch, L, T); hist_pad_mask: (batch, L) True=pad
        # cand_title_ids: (batch, C, T)
        b, l, t = hist_title_ids.shape
        hist_news_vecs = self.news_encoder(hist_title_ids.reshape(b * l, t)).reshape(b, l, -1)
        user_vec = self.user_encoder(hist_news_vecs, hist_pad_mask)  # (batch, dim)

        b2, c, t2 = cand_title_ids.shape
        cand_news_vecs = self.news_encoder(cand_title_ids.reshape(b2 * c, t2)).reshape(b2, c, -1)

        scores = torch.bmm(cand_news_vecs, user_vec.unsqueeze(-1)).squeeze(-1)  # (batch, C)
        return scores
