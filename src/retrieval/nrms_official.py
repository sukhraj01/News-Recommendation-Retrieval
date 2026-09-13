"""NRMS reimplemented to the official configuration (ADR-015, Option B).

A1's `nrms.py` (Candidate J) is "inspired by" NRMS. This module is a
layer-for-layer PyTorch port of the two official TF/Keras implementations A2
names as the baseline to reproduce:
  - microsoft/recommenders  `newsrec/models/nrms.py` + `layers.py` (MIND)
  - ebnerd-benchmark        `ebrec/models/newsrec/nrms.py`          (EB-NeRD)
Both share the same graph. Every departure from `nrms.py` below is a
deliberate match to that official code, not a style choice:

  | official                                   | nrms.py (J)                    |
  |--------------------------------------------|--------------------------------|
  | Q/K/V projections, NO bias, glorot-uniform | nn.MultiheadAttention (biased) |
  | output = head_num*head_dim (20x20 = 400)   | output = embed_dim (300)       |
  | NO output projection                       | has out_proj                   |
  | NO padding mask (pads attended as zeros)   | key_padding_mask               |
  | AttLayer2: exp(tanh(xW+b)q)/(sum+1e-7)     | softmax, hidden 128            |
  | news enc: emb->drop->selfatt->drop->att    | emb->drop->selfatt->att        |
  | user enc: selfatt->att, no dropout         | same, masked                   |

Why the TF code itself is not run: ADR-015's 2026-09-11 addendum (Ada could
not provision a TF+CUDA env within the engineer's timebox). This port must be
reported as a reimplementation, never as "the official code".

Treatment hook (EB-NeRD, ADR-015): optional per-candidate freshness input,
fused late as `logit = <user, news> + w * age`. `w` is initialised to 0, so
the treatment starts as exactly the control function. No bias term: the
training loss is a softmax over one impression's candidates, which is
invariant to a shared additive constant, so a bias could never be learned.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KERAS_EPSILON = 1e-7  # tf.keras.backend.epsilon(), used by AttLayer2's normaliser


def _glorot_uniform_(t: torch.Tensor) -> torch.Tensor:
    """Keras GlorotUniform for a 2-D (fan_in, fan_out) weight: U(-l, l),
    l = sqrt(6 / (fan_in + fan_out)). Symmetric in the two fans, so torch's
    xavier_uniform_ (which reads fans in the opposite order) gives the same
    limit; written out explicitly so the match is checkable."""
    fan_in, fan_out = t.shape
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        return t.uniform_(-limit, limit)


class KerasSelfAttention(nn.Module):
    """Port of `SelfAttention(head_num, head_dim)` called as `[y, y, y]`
    (no Q_len/V_len, no mask_right): bias-free Q/K/V projections, scaled
    dot-product per head, heads concatenated, no output projection."""

    def __init__(self, in_dim: int, head_num: int, head_dim: int):
        super().__init__()
        self.head_num, self.head_dim = head_num, head_dim
        out_dim = head_num * head_dim
        self.WQ = nn.Parameter(_glorot_uniform_(torch.empty(in_dim, out_dim)))
        self.WK = nn.Parameter(_glorot_uniform_(torch.empty(in_dim, out_dim)))
        self.WV = nn.Parameter(_glorot_uniform_(torch.empty(in_dim, out_dim)))

    @property
    def out_dim(self) -> int:
        return self.head_num * self.head_dim

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.head_num, self.head_dim).transpose(1, 2)  # (b, h, n, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, in_dim) -> (batch, seq, head_num * head_dim)
        q, k, v = self._split(x @ self.WQ), self._split(x @ self.WK), self._split(x @ self.WV)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.head_dim), dim=-1)
        o = (a @ v).transpose(1, 2)  # (b, n, h, d)
        return o.reshape(x.shape[0], x.shape[1], self.out_dim)


class AttLayer2(nn.Module):
    """Port of `AttLayer2(dim)` without a mask: weights are
    exp(tanh(xW + b) q) normalised by (sum + epsilon). Not a softmax: no
    max-subtraction and an epsilon in the denominator. Kept exact, since
    tanh bounds the exponent's argument and so overflow cannot occur."""

    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.W = nn.Parameter(_glorot_uniform_(torch.empty(in_dim, hidden)))
        self.b = nn.Parameter(torch.zeros(hidden))
        self.q = nn.Parameter(_glorot_uniform_(torch.empty(hidden, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, in_dim) -> (batch, in_dim)
        att = torch.exp((torch.tanh(x @ self.W + self.b) @ self.q).squeeze(-1))  # (batch, seq)
        weights = att / (att.sum(dim=-1, keepdim=True) + KERAS_EPSILON)
        return (weights.unsqueeze(-1) * x).sum(dim=1)


class OfficialNewsEncoder(nn.Module):
    """Embedding (pretrained init, trainable) -> Dropout -> SelfAttention ->
    Dropout -> AttLayer2. ebnerd-benchmark's optional dense stack
    (`newsencoder_units_per_layer`) is None in its reproducibility config,
    so it is not ported."""

    def __init__(self, embedding: np.ndarray, head_num: int, head_dim: int,
                 attention_hidden_dim: int, dropout: float):
        super().__init__()
        weight = torch.as_tensor(np.asarray(embedding), dtype=torch.float32)
        self.embed = nn.Embedding.from_pretrained(weight, freeze=False)
        self.drop1 = nn.Dropout(dropout)
        self.self_attn = KerasSelfAttention(weight.shape[1], head_num, head_dim)
        self.drop2 = nn.Dropout(dropout)
        self.pool = AttLayer2(self.self_attn.out_dim, attention_hidden_dim)

    @property
    def out_dim(self) -> int:
        return self.self_attn.out_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, text_len) -> (batch, head_num * head_dim)
        y = self.drop1(self.embed(token_ids))
        y = self.drop2(self.self_attn(y))
        return self.pool(y)


class OfficialUserEncoder(nn.Module):
    """TimeDistributed(news encoder) over history -> SelfAttention ->
    AttLayer2. No dropout here in either official implementation."""

    def __init__(self, news_dim: int, head_num: int, head_dim: int, attention_hidden_dim: int):
        super().__init__()
        self.self_attn = KerasSelfAttention(news_dim, head_num, head_dim)
        self.pool = AttLayer2(self.self_attn.out_dim, attention_hidden_dim)

    def forward(self, hist_news_vecs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.self_attn(hist_news_vecs))


class OfficialNRMS(nn.Module):
    """NRMS to the official configuration. `forward` returns logits
    (batch, n_candidates). The official training model applies softmax and
    categorical cross-entropy, i.e. `F.cross_entropy(logits, target)`; its
    scorer applies a sigmoid, which is monotone, so logits rank identically.

    The news and user encoders must produce the same width, since the score
    is their dot product. Both are head_num * head_dim, which holds by
    construction."""

    def __init__(self, embedding: np.ndarray, head_num: int = 20, head_dim: int = 20,
                 attention_hidden_dim: int = 200, dropout: float = 0.2,
                 use_freshness: bool = False):
        super().__init__()
        self.news_encoder = OfficialNewsEncoder(embedding, head_num, head_dim,
                                                attention_hidden_dim, dropout)
        self.user_encoder = OfficialUserEncoder(self.news_encoder.out_dim, head_num, head_dim,
                                                attention_hidden_dim)
        self.use_freshness = use_freshness
        if use_freshness:
            self.freshness_w = nn.Parameter(torch.zeros(()))  # 0 => starts as the control

    def encode_news(self, token_ids: torch.Tensor) -> torch.Tensor:
        """(..., text_len) -> (..., news_dim). Flattens any leading dims so
        history (b, L, T) and candidates (b, C, T) share one code path."""
        lead = token_ids.shape[:-1]
        vecs = self.news_encoder(token_ids.reshape(-1, token_ids.shape[-1]))
        return vecs.reshape(*lead, -1)

    def forward(self, hist_ids: torch.Tensor, cand_ids: torch.Tensor,
                cand_age: torch.Tensor | None = None) -> torch.Tensor:
        # hist_ids: (b, L, T); cand_ids: (b, C, T); cand_age: (b, C) or None
        user = self.user_encoder(self.encode_news(hist_ids))           # (b, d)
        logits = (self.encode_news(cand_ids) * user.unsqueeze(1)).sum(-1)  # (b, C)
        if self.use_freshness:
            if cand_age is None:
                raise ValueError("use_freshness=True requires cand_age")
            logits = logits + self.freshness_w * cand_age
        return logits


def freshness_feature(impression_time_s: np.ndarray, published_time_s: np.ndarray) -> np.ndarray:
    """log1p(article age in hours) at impression time, clipped at 0 (ADR-015:
    log1p needs age >= 0; the 13 known negative-age EB-NeRD articles,
    0.002% of validation rows, are the only clipped values). A missing
    publish time gives age 0, a neutral value rather than NaN."""
    age_h = (np.asarray(impression_time_s, dtype=np.float64)
             - np.asarray(published_time_s, dtype=np.float64)) / 3600.0
    age_h = np.nan_to_num(age_h, nan=0.0)
    return np.log1p(np.clip(age_h, 0.0, None)).astype(np.float32)


class OfficialNRMSScorer:
    """Implements `src/retrieval/score.py::Scorer`'s `score(query,
    candidate_ids) -> np.ndarray` contract, the same protocol
    `nrms_training.py::NRMSLiteScorer` implements for Candidate J -- so a
    trained `OfficialNRMS` plugs into `src/submission/mind_format.py`'s
    `write_predictions` completely unchanged, the identical real-submission
    path every prior candidate in this project (J, the frozen-embedding
    scorers) used.

    `query` is a user's raw history `article_ids` (prefixed, as stored in
    `user_history.parquet`), truncated/padded to the most-recent 50 inside
    this class -- callers pass the full history, not a pre-truncated one,
    mirroring `NRMSLiteScorer`'s contract exactly. `[]` is the empty-history
    convention every scorer in this project shares.

    A candidate id absent from `token_lookup` (MIND's documented
    `N89741`-style missing-candidate quirk) scores `-inf`, the same
    convention `NRMSLiteScorer`/`score.py::_lookup_scores` already use.
    """

    def __init__(self, model, token_lookup: dict[str, np.ndarray], max_history_len: int,
                device: torch.device):
        self.model = model
        self.token_lookup = token_lookup
        self.max_history_len = max_history_len
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def score(self, query, candidate_ids) -> np.ndarray:
        k = len(candidate_ids)
        known_mask = np.fromiter((c in self.token_lookup for c in candidate_ids), dtype=bool, count=k)
        scores = np.full(k, -np.inf, dtype=np.float64)
        if not known_mask.any():
            return scores

        width = next(iter(self.token_lookup.values())).shape[0]
        pad_row = np.zeros(width, dtype=np.int64)  # official dummy-news row: all zeros
        hist_ids = list(query)[-self.max_history_len:]
        hist_rows = [self.token_lookup.get(a, pad_row) for a in hist_ids]
        hist_rows = [pad_row] * (self.max_history_len - len(hist_rows)) + hist_rows  # left-pad
        hist_t = torch.as_tensor(np.stack(hist_rows), device=self.device).unsqueeze(0)

        known_ids = [c for c, keep in zip(candidate_ids, known_mask) if keep]
        cand_rows = np.stack([self.token_lookup[a] for a in known_ids])
        cand_t = torch.as_tensor(cand_rows, device=self.device).unsqueeze(0)

        known_scores = self.model(hist_t, cand_t).squeeze(0).cpu().numpy()
        scores[known_mask] = known_scores
        return scores


def official_loss(logits: torch.Tensor) -> torch.Tensor:
    """Categorical cross-entropy over [positive, neg_1..neg_npratio], with the
    positive always at index 0 (the label layout both official loaders use)."""
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)
