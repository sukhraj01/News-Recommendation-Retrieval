"""Official MIND Codabench prediction-format converter (Part 3, ADR-005/006/008's
`Scorer` interface feeding into it).

`evaluation/official/evaluate.py` (the actual script Codabench runs) expects
one line per impression:

    impression_id [rank_1,rank_2,...,rank_N]

where `rank_i` is the 1-indexed rank (1 = best) assigned to the i-th
candidate **in the impression's original order**, exactly as it appears in
the raw `behaviors.tsv` `impressions` column — not a re-sorted order, and
its own parser does a bare `str.split()` on the line, so the JSON list must
contain no spaces (`json.dumps(..., separators=(",", ":"))`).

Why this module reads the raw zip directly rather than the processed
feature store: `src/pipeline/orchestrator.py::_write_table` always sorts
`impressions`/`candidates` by `["impression_id", "article_id"]` before
writing to parquet (ADR-002's deterministic-output requirement) — that sort
is alphabetical on the prefixed article_id string, which has no relationship
to the original within-impression order. Once written, that original order
is gone from the feature store. The raw zip's `behaviors.tsv` is the only
place it still exists, so this module re-reads it directly for ordering
purposes, and still scores through the exact same `Scorer` interface
(`src/retrieval/score.py`) and index (`build_index`/`build_embedding_index`)
everything else in this project already uses.
"""
import json
from pathlib import Path

import numpy as np

from src.datasets.mind import _BEHAVIORS_COLUMNS
from src.evaluation.ranking_metrics import rank_candidates
from src.retrieval.score import Scorer
from src.utils.ids import prefix_id
from src.utils.io import read_zip_tsv

DATASET = "mind"


def read_raw_impressions(zip_path: Path, has_labels: bool) -> list[dict]:
    """One dict per impression, in the exact row order of the raw
    `behaviors.tsv` (Python dict/list ordering, not re-sorted) — this order
    is what Codabench's own ground truth file is assumed to follow, since
    it's the dataset's own native order.

    Each dict: `raw_impression_id` (unprefixed, exactly as MIND's TSV and
    Codabench's truth file key impressions), `user_id` (our prefixed form,
    for query lookup), `article_ids` (our prefixed form, in original
    candidate order), `clicked` (`list[bool]`, same order — only present
    when `has_labels=True`, `None` for the blind test set).
    """
    folder = zip_path.stem
    raw = read_zip_tsv(zip_path, f"{folder}/behaviors.tsv", _BEHAVIORS_COLUMNS)

    rows = []
    for raw_impression_id, raw_user_id, tokens in zip(
        raw["impression_id"], raw["user_id"], raw["impressions"]
    ):
        article_ids = []
        clicked = [] if has_labels else None
        for token in tokens.split(" "):
            if has_labels:
                article_raw, label = token.rsplit("-", 1)
                clicked.append(label == "1")
            else:
                article_raw = token
            article_ids.append(prefix_id(DATASET, article_raw))
        rows.append({
            "raw_impression_id": raw_impression_id,
            "user_id": prefix_id(DATASET, raw_user_id),
            "article_ids": article_ids,
            "clicked": clicked,
        })
    return rows


def ranks_for_impression(scores: np.ndarray, impression_id: str, seed: int = 0) -> list[int]:
    """1-indexed rank (1 = best) per original position — the inverse
    permutation of `rank_candidates`'s score-descending order, so
    `ranks[i]` answers "what rank did the i-th original-order candidate
    get", which is exactly the quantity the official format wants.
    `impression_id` only seeds the tie-break RNG (ADR-007); any stable,
    unique-per-impression string works, so the raw (unprefixed) id is used
    directly rather than round-tripping through `prefix_id`.
    """
    order = rank_candidates(scores, impression_id, seed)
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks.tolist()


def write_predictions(
    out_path: Path,
    zip_path: Path,
    scorer: Scorer,
    query_by_user: dict,
    empty_query,
    has_labels: bool,
    seed: int = 0,
) -> int:
    """Write the official-format predictions file. Returns the number of
    impressions written (== number of lines).

    `query_by_user` maps our prefixed user_id -> that method's query
    representation (token list for BM25, vector-or-None for embeddings);
    `empty_query` is the fallback for a user with no history entry (`[]`
    for BM25, `None` for embeddings, per ADR-005/ADR-008's cold-start
    contract) — passed explicitly rather than guessed from the scorer type,
    since `Scorer` deliberately doesn't expose which method it is.
    """
    rows = read_raw_impressions(zip_path, has_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            query = query_by_user.get(row["user_id"], empty_query)
            scores = scorer.score(query, row["article_ids"])
            ranks = ranks_for_impression(scores, row["raw_impression_id"], seed)
            f.write(f"{row['raw_impression_id']} {json.dumps(ranks, separators=(',', ':'))}\n")
    return len(rows)


def write_truth_file(out_path: Path, zip_path: Path) -> int:
    """Local-only ground-truth file for cross-checking against
    `evaluate.py`, built from a labeled split's own real click labels (dev
    only — Codabench's actual private test-set ground truth is never
    available to us). Same line order/count contract as
    `write_predictions`: one line per impression, `impid [label_1,...]`
    (0/1 per candidate, original order) — `evaluate.py` treats a `[]` label
    list as a masked impression to skip; none are masked here since every
    label is real."""
    rows = read_raw_impressions(zip_path, has_labels=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            labels = [int(c) for c in row["clicked"]]
            f.write(f"{row['raw_impression_id']} {json.dumps(labels, separators=(',', ':'))}\n")
    return len(rows)
