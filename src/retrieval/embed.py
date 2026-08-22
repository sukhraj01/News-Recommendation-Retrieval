"""Embedding-based (semantic) index construction and query representation.

Phase 4's second `Scorer` consumer, per ADR-008. Mirrors `index.py`/`query.py`'s
BM25 shapes deliberately:

- `EmbeddingIndex` is `BM25Index`'s counterpart — `article_ids`/`id_to_col`
  are identical in spirit, `vectors` (dense, L2-normalized) replaces the
  sparse BM25 weight matrix.
- `build_user_embedding_query` is `build_user_query`'s counterpart —
  mean-pool a history's article vectors instead of concatenating tokens.
  Empty history returns `None` (not an empty array), matching
  `build_user_query`'s empty-list return for the same true-cold-start case
  (ADR-005) — `None` is used here rather than an all-zero vector because a
  zero vector is a valid (if degenerate) point in embedding space, whereas
  "no representation exists" must be structurally distinguishable from it.

Per ADR-008: `title + " " + abstract` is encoded once per article with a
purpose-built multilingual sentence-embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`, chosen over mean-pooled raw
BERT/XLM-R and over `multilingual-e5-small` — see ADR-008's benchmark
evidence), and the result is cached to disk keyed by `(model_name,
article_ids order)`: encoding ~75k articles through a transformer is not
the ~1s BM25 index build was, so unlike `index.py`, rebuilding on every
script invocation would be real, avoidable, repeated cost.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 64


def model_slug(model_name: str) -> str:
    return model_name.replace("/", "__")


@dataclass
class EmbeddingIndex:
    article_ids: list[str]
    id_to_col: dict[str, int]
    vectors: np.ndarray  # (n_docs, dim), L2-normalized rows


def _default_device() -> str:
    """Best available backend, auto-detected rather than hardcoded (2026-08-21
    addendum, MINDlarge Kaggle verification): `device` used to default to
    `"mps"` unconditionally — correct for this project's own Mac development
    machine, but `"mps"` (Apple's GPU backend) doesn't exist at all on
    Kaggle's Linux runners, where it raised `RuntimeError: PyTorch is not
    linked with support for mps devices` on first real use. Preference
    order: CUDA (Kaggle GPU runtime) > MPS (this project's Mac) > CPU
    (always available, the safe universal fallback)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """Isolates the only `sentence_transformers` import in this project,
    mirroring how `index.py` isolates `rank_bm25`. `device=None` (default)
    auto-detects the best available backend — see `_default_device`; pass
    an explicit value to override."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device or _default_device())


def _encode(encoder, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, encoder.get_sentence_embedding_dimension()), dtype=np.float32)
    return encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def build_embedding_index(
    articles: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    cache_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    encoder=None,
) -> EmbeddingIndex:
    """Build (or load-from-cache) an `EmbeddingIndex` over `title + " " +
    abstract` for every article, per ADR-002/ADR-005's mandatory text
    fields.

    `encoder` may be passed directly (used by tests to inject a stub
    encoder and skip a real model load); otherwise `load_encoder(model_name,
    device)` is used.
    """
    article_ids = articles["article_id"].tolist()

    if cache_path is not None:
        cached = _load_cache(cache_path, model_name, article_ids)
        if cached is not None:
            return cached

    text = (articles["title"].fillna("") + " " + articles["abstract"].fillna("")).tolist()
    if encoder is None:
        encoder = load_encoder(model_name, device)
    vectors = _encode(encoder, text, batch_size)

    index = EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=vectors,
    )

    if cache_path is not None:
        _write_cache(cache_path, model_name, index)

    return index


def _load_cache(cache_path: Path, model_name: str, article_ids: list[str]) -> EmbeddingIndex | None:
    sidecar = cache_path.with_suffix(".json")
    if not (cache_path.exists() and sidecar.exists()):
        return None
    meta = json.loads(sidecar.read_text())
    if meta.get("model") != model_name or meta.get("article_ids") != article_ids:
        return None  # stale cache (different model or article set) — recompute, never silently reuse
    vectors = np.load(cache_path)
    return EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=vectors,
    )


def _write_cache(cache_path: Path, model_name: str, index: EmbeddingIndex) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, index.vectors)
    cache_path.with_suffix(".json").write_text(
        json.dumps({"model": model_name, "article_ids": index.article_ids})
    )


def build_user_embedding_query(
    article_ids: list[str], vector_lookup: dict[str, np.ndarray]
) -> np.ndarray | None:
    """`vector_lookup` maps article_id -> its L2-normalized embedding vector.

    Mean-pools a user's history article vectors, then re-normalizes (per
    Q3's "mean-pooled embeddings" spec) so the result is comparable to the
    index's own L2-normalized rows via a plain dot product (cosine
    similarity). Unknown article_ids (not in `vector_lookup`) are skipped,
    same treatment as `build_user_query`'s missing-text-lookup case.

    Returns `None` for an empty/fully-unresolvable history — a true
    cold-start user, per ADR-005/ADR-008 — never a NaN-filled vector.
    """
    vecs = [vector_lookup[a] for a in article_ids if a in vector_lookup]
    if not vecs:
        return None
    mean = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)


def build_user_embedding_query_recency(
    article_ids: list[str], vector_lookup: dict[str, np.ndarray], decay: float = 0.9
) -> np.ndarray | None:
    """Candidate C (2026-08-21 local-validation session): same contract as
    `build_user_embedding_query`, but weights each resolvable history
    vector by exponential decay based on its position in `article_ids`,
    assuming (per MIND's documented convention, but — per ADR-005 — never
    independently verified against real timestamps) that the list is
    ordered oldest-to-most-recent, i.e. `article_ids[-1]` is the most
    recent click. Weight for the i-th *resolvable* entry (0-indexed from
    the end) is `decay ** i`, so the most recent resolvable click gets
    weight 1.0 and older ones decay geometrically — deliberately computed
    over the filtered (resolvable) sequence, not the raw history, so a
    handful of unresolvable ids interspersed in the middle of a history
    don't shift surrounding weights. `decay=0.9` is a single untuned
    starting point (not searched), same "test whether it helps at all"
    framing `run_hybrid_experiment.py`'s untuned 50/50 blend uses for
    Candidate B — a real signal here would justify tuning it later.

    ADR-005 rejected recency weighting for BM25 query construction
    specifically because MIND's history order is unverified — this
    function doesn't resolve that risk, it deliberately re-tests it (this
    is Candidate C's whole point), so any real result from this function
    inherits that same unverified-ordering caveat and must be reported
    with it, not silently.
    """
    resolvable = [vector_lookup[a] for a in article_ids if a in vector_lookup]
    if not resolvable:
        return None
    n = len(resolvable)
    weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float64)
    mean = np.average(np.stack(resolvable), axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)
