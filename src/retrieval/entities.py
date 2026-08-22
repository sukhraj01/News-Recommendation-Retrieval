"""MIND entity-vector index construction (Candidate A, 2026-08-21 local-
validation session — see PROJECT_STATE.md's "Current Objective").

MIND's `news.tsv` links each article's title/abstract to Wikidata entities
(`WikidataId` + a linker `Confidence` in [0, 1]); `entity_embedding.vec`
supplies a pretrained 100-dim TransE vector per `WikidataId`. Neither has a
production code path yet — `src/datasets/mind.py::_combine_mind_entities`
already folds both fields into the unified schema's `entities` column as
`{"title_entities": [...], "abstract_entities": [...]}` JSON, but nothing
downstream reads it (`entities` is schema-only for EB-NeRD/MIND parity, per
ADR-002; MIND's own `topics` column is similarly unused). This module is
the first consumer.

Deliberately reuses `embed.py`'s `EmbeddingIndex` dataclass and
`build_user_embedding_query` unchanged: once an article has a single dense
vector (whatever its source), mean-pooling a user's history and scoring by
cosine similarity is the same operation regardless of whether that vector
came from MiniLM or a knowledge-graph embedding — this is exactly the seam
ADR-008 built `Scorer`/`EmbeddingIndex` for. Only vector *construction* is
new here.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd

from .embed import EmbeddingIndex


def parse_entity_mentions(entities_json: str | float) -> list[tuple[str, float]]:
    """`entities_json` is one article's combined-schema `entities` string
    (`_combine_mind_entities`'s output: `{"title_entities": [...],
    "abstract_entities": [...]}`, each a list of MIND's raw per-mention
    dicts). Returns every `(WikidataId, Confidence)` mention across both
    fields, kept as a multiset (a mention repeated in both title and
    abstract counts twice) — same "repeats carry signal, don't dedupe"
    stance ADR-005 measured for BM25 query tokens, applied here rather than
    re-litigated from scratch. Missing/empty/unparseable input returns `[]`,
    never raises — an article can legitimately have zero linked entities
    (e.g. `N61837`'s empty `title_entities` in the raw sample).
    """
    import json

    if entities_json is None or (isinstance(entities_json, float) and pd.isna(entities_json)):
        return []
    try:
        parsed = json.loads(entities_json)
    except (json.JSONDecodeError, TypeError):
        return []

    mentions: list[tuple[str, float]] = []
    for field in ("title_entities", "abstract_entities"):
        for mention in parsed.get(field, []) or []:
            wikidata_id = mention.get("WikidataId")
            confidence = mention.get("Confidence")
            if wikidata_id and confidence is not None:
                mentions.append((wikidata_id, float(confidence)))
    return mentions


def load_entity_vectors(raw_bytes: bytes) -> dict[str, np.ndarray]:
    """Parses a MIND `entity_embedding.vec` file's raw bytes (tab-separated:
    `WikidataId  v_1 ... v_100`, no header) into `WikidataId -> float32
    vector`. Takes bytes rather than a path so callers can read it straight
    out of the split's zip via `read_zip_member_bytes`, consistent with the
    rest of this project never extracting raw archives to disk.
    """
    vectors: dict[str, np.ndarray] = {}
    for line in io.BytesIO(raw_bytes).readlines():
        # Real MIND `.vec` files carry a trailing tab before the newline on
        # every line (confirmed directly against `MINDsmall_dev`'s file) —
        # a naive split would leave a dangling empty string as the last
        # "value" and fail `float()`. Filtering empty fields handles that
        # trailing tab and any other stray blank fields the same way.
        parts = [p for p in line.decode("utf-8").rstrip("\n").split("\t") if p != ""]
        if len(parts) < 2:
            continue
        vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
    return vectors


def build_article_entity_vector(
    mentions: list[tuple[str, float]], entity_vectors: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Confidence-weighted mean of the mentions' resolvable vectors,
    L2-normalized to match every other row this project scores by cosine
    similarity (`embed.py`'s convention). Unresolvable/absent entities are
    skipped, same "skip what's missing" treatment as
    `build_user_embedding_query`. Returns `None` if nothing resolves (no
    entities, or none of them present in `entity_vectors`) — the caller
    decides how to fill the index row (see `build_entity_index`); kept as
    `None` here rather than an implicit zero vector so "no signal" stays
    structurally distinguishable from "a genuine zero-length vector," same
    reasoning `build_user_embedding_query`'s docstring gives for its own
    `None` return.
    """
    weighted = [
        (entity_vectors[wid], conf)
        for wid, conf in mentions
        if wid in entity_vectors and conf > 0
    ]
    if not weighted:
        return None
    vecs = np.stack([v for v, _ in weighted])
    weights = np.array([c for _, c in weighted], dtype=np.float64)
    mean = np.average(vecs, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    if norm == 0:
        return None
    return (mean / norm).astype(np.float32)


def build_entity_index(
    articles: pd.DataFrame, entity_vectors: dict[str, np.ndarray]
) -> tuple[EmbeddingIndex, float]:
    """Builds one `EmbeddingIndex` row per article in `articles` (must have
    `article_id`/`entities` columns) from `entity_vectors`.

    Articles with no resolvable entity vector get an explicit all-zero row
    — the same "structurally absent signal" contract `EmbeddingScorer`
    already gives a `None` *query* (an all-zero query scores every
    candidate identically, deferring to the tie-break); a zero *row* has
    the same effect from the other side, scoring 0 against every query,
    never spuriously ranking high. This is a genuine behavior difference
    from `build_article_entity_vector`'s own `None` return, which is why
    the zero-fill happens here, at the index-assembly boundary, rather
    than inside that function.

    Returns `(index, coverage)` where `coverage` is the fraction of
    articles that got a real (non-zero-filled) vector — the "measure real
    article coverage first" check the objective calls for, computed once,
    directly from the same pass that builds the vectors (not estimated).
    """
    article_ids = articles["article_id"].tolist()
    dim = next(iter(entity_vectors.values())).shape[0] if entity_vectors else 100

    rows = []
    n_covered = 0
    for entities_json in articles["entities"]:
        mentions = parse_entity_mentions(entities_json)
        vec = build_article_entity_vector(mentions, entity_vectors)
        if vec is None:
            rows.append(np.zeros(dim, dtype=np.float32))
        else:
            rows.append(vec)
            n_covered += 1

    vectors = np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float32)
    coverage = n_covered / len(article_ids) if article_ids else float("nan")

    index = EmbeddingIndex(
        article_ids=article_ids,
        id_to_col={aid: i for i, aid in enumerate(article_ids)},
        vectors=vectors,
    )
    return index, coverage
