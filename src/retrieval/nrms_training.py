"""Shared training/eval helpers for Candidate J (NRMS-lite), reused by both
`scripts/mind_nrms_lite_kaggle_run.py` (Kaggle notebook) and
`scripts/mind_nrms_lite_ada_run.py` (Ada/SLURM). Extracted so the two
execution environments don't duplicate the same non-trivial training-loop/
dataset-construction/GloVe-loading logic -- the same anti-duplication
principle this candidate's own integration review already applied to
`src/pipeline`'s MIND loader (see ADR-012 addendum).
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.ranking_metrics import mrr, ndcg_at_k, rank_candidates, safe_auc
from src.retrieval.nrms import encode_title


def build_title_matrix(
    articles: pd.DataFrame, word2id: dict[str, int], max_title_len: int,
) -> tuple[torch.Tensor, dict, int]:
    """Returns `(title_matrix, news_id2row, pad_news_row)`. `title_matrix`
    has one extra all-PAD sentinel row appended at index `len(articles)`,
    used for history padding and any candidate/history id absent from
    `articles` (MIND's documented missing-candidate quirk)."""
    news_ids_all = articles["article_id"].tolist()
    news_id2row = {nid: i for i, nid in enumerate(news_ids_all)}
    pad_news_row = len(news_ids_all)
    title_matrix = torch.tensor(
        [encode_title(t, word2id, max_title_len) for t in articles["title"]], dtype=torch.long
    )
    title_matrix = torch.cat([title_matrix, torch.zeros(1, max_title_len, dtype=torch.long)], dim=0)
    return title_matrix, news_id2row, pad_news_row


def build_training_examples(
    impressions: pd.DataFrame, history_by_user: dict, news_row, neg_k: int, max_history_len: int,
    rng: random.Random, max_examples: int | None = None,
) -> list[tuple[list[int], int, list[int]]]:
    """NRMS-style negative sampling: one example per (impression, clicked
    candidate) pair, `neg_k` non-clicked candidates sampled from the SAME
    impression (with-replacement resampling if fewer than `neg_k` negatives
    exist in that impression). `news_row` maps an article_id to its row in
    a title matrix (`build_title_matrix`'s `news_id2row.get`-style lookup).
    """
    examples: list[tuple[list[int], int, list[int]]] = []
    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")
    for (user_id, _impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        pos_list = [c for c, is_clicked in zip(candidate_ids, clicked) if is_clicked]
        neg_list = [c for c, is_clicked in zip(candidate_ids, clicked) if not is_clicked]
        if not pos_list or not neg_list:
            continue
        hist_ids = list(history_by_user.get(user_id, []))[-max_history_len:]
        hist_rows = [news_row(nid) for nid in hist_ids]
        for pos_id in pos_list:
            sampled_neg = (
                rng.sample(neg_list, neg_k) if len(neg_list) >= neg_k
                else [rng.choice(neg_list) for _ in range(neg_k)]
            )
            examples.append((hist_rows, news_row(pos_id), [news_row(n) for n in sampled_neg]))
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


class NRMSTrainDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list, max_history_len: int, pad_news_row: int):
        self.examples = examples
        self.max_history_len = max_history_len
        self.pad_news_row = pad_news_row

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        hist_rows, pos_row, neg_rows = self.examples[idx]
        hist_padded = hist_rows[-self.max_history_len:]
        hist_len = len(hist_padded)
        hist_padded = hist_padded + [self.pad_news_row] * (self.max_history_len - hist_len)
        cand_rows = [pos_row] + neg_rows  # positive always index 0
        return (
            torch.tensor(hist_padded, dtype=torch.long),
            torch.tensor(hist_len, dtype=torch.long),
            torch.tensor(cand_rows, dtype=torch.long),
        )


def train_one_epoch(
    model, loader, optimizer, device: torch.device, max_history_len: int, title_matrix: torch.Tensor,
) -> float:
    model.train()
    epoch_loss, n_batches = 0.0, 0
    for hist_rows, hist_len, cand_rows in loader:
        hist_rows, hist_len, cand_rows = hist_rows.to(device), hist_len.to(device), cand_rows.to(device)
        hist_t = title_matrix[hist_rows]
        b = hist_rows.shape[0]
        hist_pad_mask = torch.arange(max_history_len, device=device).unsqueeze(0) >= hist_len.unsqueeze(1)
        cand_t = title_matrix[cand_rows]

        scores = model(hist_t, hist_pad_mask, cand_t)
        target = torch.zeros(b, dtype=torch.long, device=device)  # positive is always index 0
        loss = F.cross_entropy(scores, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1
    return epoch_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_impressions(
    model, impressions: pd.DataFrame, history_by_user: dict, news_row, title_matrix: torch.Tensor,
    max_history_len: int, pad_news_row: int, device: torch.device, seed: int = 0,
) -> pd.DataFrame:
    """Per-impression AUC/MRR/nDCG@5/nDCG@10 via this project's own Q4
    ranking-metric functions (`src/evaluation/ranking_metrics.py`) -- same
    methodology as the official `evaluate.py`, reused rather than
    reimplemented (ADR-012 addendum)."""
    model.eval()
    rows = []
    impressions_sorted = impressions.sort_values(["user_id", "impression_id"], kind="stable")
    for (user_id, impression_id), group in impressions_sorted.groupby(["user_id", "impression_id"], sort=False):
        candidate_ids = group["article_id"].tolist()
        clicked = group["clicked"].to_numpy(dtype=bool)
        if clicked.all() or not clicked.any():
            continue  # safe_auc is undefined here too -- skip before building tensors

        hist_ids = list(history_by_user.get(user_id, []))[-max_history_len:]
        hist_rows = [news_row(nid) for nid in hist_ids]
        hist_len = len(hist_rows)
        hist_padded = hist_rows + [pad_news_row] * (max_history_len - hist_len)
        hist_t = title_matrix[torch.tensor(hist_padded, device=device)].unsqueeze(0)
        hist_pad_mask = torch.tensor(
            [[i >= hist_len for i in range(max_history_len)]], dtype=torch.bool, device=device
        )
        cand_rows = torch.tensor([news_row(nid) for nid in candidate_ids], device=device)
        cand_t = title_matrix[cand_rows].unsqueeze(0)

        scores = model(hist_t, hist_pad_mask, cand_t).squeeze(0).cpu().numpy()
        order = rank_candidates(scores, impression_id, seed=seed)
        ranked_clicked = clicked[order]
        rows.append({
            "impression_id": impression_id, "user_id": user_id,
            "auc": safe_auc(scores, clicked),
            "mrr": mrr(ranked_clicked),
            "ndcg5": ndcg_at_k(ranked_clicked, 5),
            "ndcg10": ndcg_at_k(ranked_clicked, 10),
        })
    return pd.DataFrame(rows)


def load_glove_vectors(path: Path, dim: int) -> dict[str, np.ndarray]:
    """Reads a GloVe `.txt` vector file (space-separated: `word v1 v2 ...
    vN`) into a `{word: vector}` dict. Lines that don't split into exactly
    `dim + 1` fields are skipped -- real GloVe release files (e.g.
    `glove.840B.300d.txt`) contain a small number of malformed lines where
    the "word" itself is whitespace, which would otherwise silently corrupt
    a `float()` parse."""
    vectors: dict[str, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != dim + 1:
                continue
            try:
                vec = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            vectors[parts[0]] = vec
    return vectors


def init_pretrained_embeddings(
    embedding: torch.nn.Embedding, glove: dict[str, np.ndarray], word2id: dict[str, int],
) -> int:
    """Overwrites `embedding.weight.data` rows for every `word2id` entry
    found in `glove`; every other row (PAD, UNK, and any real word GloVe
    doesn't cover) keeps its existing (zero / random) init untouched.
    Embeddings stay trainable afterward (`nn.Embedding` requires_grad is
    unaffected) -- this initializes, it does not freeze, matching NRMS's
    own end-to-end fine-tuning of word embeddings. Returns the number of
    rows actually overwritten -- real vocabulary coverage, not assumed,
    since the caller needs this to judge whether the vocab/GloVe match was
    any good."""
    found = 0
    with torch.no_grad():
        for word, idx in word2id.items():
            vec = glove.get(word)
            if vec is not None:
                embedding.weight.data[idx] = torch.from_numpy(vec)
                found += 1
    return found


class NRMSLiteScorer:
    """Implements `src/retrieval/score.py::Scorer`'s `score(query,
    candidate_ids) -> np.ndarray` contract so a trained NRMSLite model can
    plug into `src/submission/mind_format.py::write_predictions` unchanged
    -- the same real-submission path every prior candidate in this project
    used (Candidate G's `GatedScorer`, the original `EmbeddingScorer`).

    `query` is a user's raw history `article_ids` list (as stored in
    `user_history.parquet`, prefixed ids, unpooled) -- not a precomputed
    vector, since this model's history representation is the model's own
    learned attention pooling over titles, not something computable
    outside the model. `[]` is the empty-history convention (matching
    every other scorer's cold-start contract in this project).

    A candidate id absent from `news_id2row` (MIND's documented
    `N89741`-style missing-candidate quirk -- a real, observed gap between
    a split's `behaviors.tsv` candidates and its own `news.tsv`) scores
    `-inf`, the same convention `score.py::_lookup_scores` and
    `AttentionRerankScorer` (`rerank.py`) already use -- replicated here
    rather than imported, since this scorer's per-call shape (one
    impression, model-forward-based) doesn't fit those helpers' full-
    corpus-array assumption.
    """

    def __init__(
        self, model, news_id2row: dict, pad_news_row: int, title_matrix: torch.Tensor,
        max_history_len: int, device: torch.device,
    ):
        self.model = model
        self.news_id2row = news_id2row
        self.pad_news_row = pad_news_row
        self.title_matrix = title_matrix
        self.max_history_len = max_history_len
        self.device = device
        self.model.eval()

    def _news_row(self, article_id) -> int:
        return self.news_id2row.get(article_id, self.pad_news_row)

    @torch.no_grad()
    def score(self, query, candidate_ids) -> np.ndarray:
        k = len(candidate_ids)
        known_mask = np.fromiter(
            (c in self.news_id2row for c in candidate_ids), dtype=bool, count=k
        )
        scores = np.full(k, -np.inf, dtype=np.float64)
        if not known_mask.any():
            return scores

        hist_ids = list(query)[-self.max_history_len:]
        hist_rows = [self._news_row(a) for a in hist_ids]
        hist_len = len(hist_rows)
        hist_padded = hist_rows + [self.pad_news_row] * (self.max_history_len - hist_len)
        hist_t = self.title_matrix[torch.tensor(hist_padded, device=self.device)].unsqueeze(0)
        hist_pad_mask = torch.tensor(
            [[i >= hist_len for i in range(self.max_history_len)]], dtype=torch.bool, device=self.device
        )

        known_ids = [c for c, keep in zip(candidate_ids, known_mask) if keep]
        cand_rows = torch.tensor([self._news_row(a) for a in known_ids], device=self.device)
        cand_t = self.title_matrix[cand_rows].unsqueeze(0)

        known_scores = self.model(hist_t, hist_pad_mask, cand_t).squeeze(0).cpu().numpy()
        scores[known_mask] = known_scores
        return scores
