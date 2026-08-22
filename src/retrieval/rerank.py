"""Candidate I (2026-08-22 session, ADR-011): a small, end-to-end-trained
neural re-ranker that attention-pools a user's RAW history embedding
sequence, conditioned per candidate, instead of collapsing it to one fixed
mean-pooled vector (`embed.py::build_user_embedding_query`) or a single
cosine-similarity scalar (Candidates F/H1/H2's `embed_cos` feature).
Genuinely different mechanism from every prior MIND candidate (A-H): those
all either used a fixed similarity function or combined several such fixed
scalars as features; this one *learns*, per candidate, how much weight to
place on each history item.

First neural-network *training* in this project — every other `torch`
usage (`embed.py`) is frozen `sentence-transformers` inference only.
Frozen MiniLM embeddings (the same cached vectors `EmbeddingScorer`
already uses) are the only input; only the attention layer here is
trained. See ADR-011 for the full design rationale (why no padding/
masking, why a dot product over an MLP head, why per-impression forward
passes).
"""
import numpy as np
import torch
import torch.nn as nn


def build_user_history_vectors(
    article_ids: list[str], vector_lookup: dict[str, np.ndarray]
) -> np.ndarray:
    """`vector_lookup` maps article_id -> its L2-normalized embedding
    vector (same lookup `build_user_embedding_query` takes). Unlike that
    function, this returns the RAW per-article sequence, unpooled — shape
    `(n_resolvable, dim)`, in the same order as `article_ids`. Unknown
    article_ids are skipped, same "skip what's missing" convention every
    query-builder in this project uses. Returns an empty `(0, 0)` array
    (never `None`) for a fully-unresolvable/empty history — `AttentionScorer`
    handles a zero-length history explicitly as the cold-start case, so
    this function's contract stays a plain array, not an Optional.
    """
    vecs = [vector_lookup[a] for a in article_ids if a in vector_lookup]
    if not vecs:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack(vecs).astype(np.float32)


class AttentionScorer(nn.Module):
    """Single-head scaled dot-product attention, conditioned per
    candidate: for each candidate, attention-weights the user's history
    items by relevance to THAT candidate (a different pooled "user vector"
    per candidate, not one fixed vector for the whole impression), then
    scores via a dot product between the candidate vector and its own
    candidate-specific pooled user vector.

    Operates on ONE impression at a time (no batch dimension, no padding)
    — `candidate_vecs` is `(K, dim)`, `history_vecs` is `(H, dim)`. Per
    ADR-011, avoiding padding/masking entirely was a deliberate choice on
    this project's first neural training, trading some throughput for
    eliminating an entire class of subtle mask-in-softmax correctness
    bugs.
    """

    def __init__(self, dim: int, d_attn: int = 64):
        super().__init__()
        self.wq = nn.Linear(dim, d_attn, bias=False)
        self.wk = nn.Linear(dim, d_attn, bias=False)
        self.scale = d_attn ** 0.5
        # Always present (harmless if unused): a single learnable scalar
        # weight on a candidate's train-split-only log-popularity, added
        # ADR-011's addendum (2026-08-22) - since AUC (this project's
        # primary metric) is invariant to a per-impression constant shift,
        # no separate bias term is needed for the metric that matters here
        # (a bias would only affect BCE-loss calibration, not ranking).
        self.pop_weight = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, candidate_vecs: torch.Tensor, history_vecs: torch.Tensor,
        log_popularity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns one logit per candidate, shape `(K,)`.

        `log_popularity` (optional, `(K,)`): train-split-only log-popularity
        per candidate (ADR-011 addendum). When given, added to the
        attention-derived score as `pop_weight * log_popularity` — the
        single new learnable parameter this addendum introduces, directly
        analogous to Candidates F/H1's own fitted `log_popularity`
        coefficient.

        `history_vecs.shape[0] == 0` (a true cold-start user, ADR-005/008):
        the attention term is an all-zero vector (the same cold-start
        convention `BM25Scorer`/`EmbeddingScorer` already use for a `None`
        query) — but if `log_popularity` is given, the popularity term is
        still added on top, so a cold user gets a real, learnable,
        non-constant score instead of a flat tie. This is the whole point
        of adding popularity: it is the one signal that needs no history
        at all, matching why it dominated Candidates F/H1's cold-cohort
        wins.
        """
        k = candidate_vecs.shape[0]
        if history_vecs.shape[0] == 0:
            base = torch.zeros(k, dtype=candidate_vecs.dtype)
        else:
            q = self.wq(candidate_vecs)  # (K, d_attn)
            key = self.wk(history_vecs)  # (H, d_attn)
            attn_logits = (q @ key.T) / self.scale  # (K, H)
            attn_weights = torch.softmax(attn_logits, dim=-1)  # (K, H)
            pooled = attn_weights @ history_vecs  # (K, dim) -- candidate-specific user vector
            base = (candidate_vecs * pooled).sum(dim=-1)  # (K,) dot product per candidate

        if log_popularity is not None:
            base = base + self.pop_weight * log_popularity
        return base


class AttentionRerankScorer:
    """Implements `src/retrieval/score.py::Scorer`'s `score(query,
    candidate_ids) -> np.ndarray` contract so the unchanged Q4 harness can
    evaluate a trained `AttentionScorer` exactly like BM25/embedding/F/H.

    `query` is a per-user raw history vector array
    (`build_user_history_vectors`'s output, precomputed once per user —
    same per-user-precompute pattern every other scorer's query-by-user
    dict in this project uses), including the empty-history case.

    Unlike `BM25Scorer`/`EmbeddingScorer`, this does NOT cache a
    full-corpus score vector per query — candidates are scored
    independently given the history (no full-corpus retrieval need here),
    so there is no whole-corpus vector to cache or subset; each call
    scores exactly the `candidate_ids` given. A candidate id absent from
    `vector_lookup` scores `-inf`, matching `score.py::_lookup_scores`'s
    established convention for MIND's `N89741`-style missing-candidate
    quirk (replicated here rather than imported, since that helper's
    full-corpus-array assumption doesn't fit this scorer's shape).
    """

    def __init__(self, model: AttentionScorer, vector_lookup: dict[str, np.ndarray]):
        self.model = model
        self.vector_lookup = vector_lookup
        self.model.eval()

    def score(self, query: np.ndarray, candidate_ids) -> np.ndarray:
        k = len(candidate_ids)
        known_mask = np.fromiter((c in self.vector_lookup for c in candidate_ids), dtype=bool, count=k)
        scores = np.full(k, -np.inf, dtype=np.float64)
        if not known_mask.any():
            return scores

        known_ids = [c for c, m in zip(candidate_ids, known_mask) if m]
        cand_vecs = torch.from_numpy(np.stack([self.vector_lookup[c] for c in known_ids]).astype(np.float32))
        hist_vecs = torch.from_numpy(np.asarray(query, dtype=np.float32))

        with torch.no_grad():
            known_scores = self.model(cand_vecs, hist_vecs).numpy()

        scores[known_mask] = known_scores
        return scores
