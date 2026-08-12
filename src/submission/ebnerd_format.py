"""Official EB-NeRD Codabench prediction-format converter (Part 1 of the
EB-NeRD submission; see PROJECT_STATE.md's Part 0 for how the format was
confirmed).

Part 0 established, against the real files on Kaggle (not guessed from the
paper's Appendix A schema docs):

- `predictions.txt`: one line per impression, `impression_id
  [rank_1,...,rank_N]` — a permutation matching that impression's
  `article_ids_inview` order. Same shape as MIND's official format (a rank
  permutation, not raw scores), just different source column names
  (`user_id`/`article_ids_inview` vs. MIND's `UserID`/`impressions`).
- `is_beyond_accuracy=True` rows (200,000 of them in the real test set, each
  drawn from one fixed 250-article pool) still need a real submitted
  ranking in the same format — nothing special-cased for them here.

This module is a direct port of `src/submission/mind_format.py`'s design,
not a redesign: same reason for re-reading the raw zip's `behaviors.parquet`
directly rather than the processed feature store — `src/pipeline/
orchestrator.py::_write_table` sorts `impressions` by `["impression_id",
"article_id"]` before writing parquet (ADR-002's deterministic-output
requirement), which destroys `article_ids_inview`'s original order. Scoring
still goes through the same `Scorer` interface (`src/retrieval/score.py`)
and index (`build_index`/`build_embedding_index`) as everything else in
this project.
"""
import json
from pathlib import Path

import numpy as np

from src.evaluation.ranking_metrics import rank_candidates
from src.retrieval.score import Scorer
from src.utils.ids import prefix_id
from src.utils.io import read_zip_parquet

DATASET = "ebnerd"


def read_raw_impressions(zip_path: Path, split: str, has_labels: bool) -> list[dict]:
    """One dict per impression, in the exact row order of the raw
    `{split}/behaviors.parquet` (not re-sorted).

    Each dict: `raw_impression_id` (unprefixed `str(impression_id)`, exactly
    as EB-NeRD's own files and Codabench's truth file key impressions),
    `user_id` (our prefixed form, for query lookup), `article_ids` (our
    prefixed form, in original `article_ids_inview` order), `clicked`
    (`list[bool]`, same order — only present when `has_labels=True`, `None`
    for the blind test set, which drops `article_ids_clicked` entirely).
    """
    behaviors = read_zip_parquet(zip_path, f"{split}/behaviors.parquet")
    clicked_col = behaviors["article_ids_clicked"] if has_labels else [None] * len(behaviors)

    rows = []
    for raw_impression_id, raw_user_id, inview, clicked_ids in zip(
        behaviors["impression_id"], behaviors["user_id"],
        behaviors["article_ids_inview"], clicked_col,
    ):
        article_ids = [prefix_id(DATASET, a) for a in inview]
        clicked = None
        if has_labels:
            clicked_set = set(clicked_ids)
            clicked = [a in clicked_set for a in inview]
        rows.append({
            "raw_impression_id": str(raw_impression_id),
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
    directly rather than round-tripping through `prefix_id`."""
    order = rank_candidates(scores, impression_id, seed)
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks.tolist()


def write_predictions(
    out_path: Path,
    zip_path: Path,
    split: str,
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
    rows = read_raw_impressions(zip_path, split, has_labels)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            query = query_by_user.get(row["user_id"], empty_query)
            scores = scorer.score(query, row["article_ids"])
            ranks = ranks_for_impression(scores, row["raw_impression_id"], seed)
            f.write(f"{row['raw_impression_id']} {json.dumps(ranks, separators=(',', ':'))}\n")
    return len(rows)


def write_truth_file(out_path: Path, zip_path: Path, split: str) -> int:
    """Local-only ground-truth file for cross-checking against
    `evaluation/official/evaluate.py` (confirmed in Part 0 to be a generic,
    format-only script with no MIND-specific logic — it never parses
    anything but the shared `impid [ranks]`/`impid [labels]` line shape, so
    it applies to EB-NeRD's predictions unchanged). Built from a labeled
    split's own real click labels (`ebnerd_small`'s `validation` split
    here; Codabench's actual private test-set ground truth is never
    available to us). Same line order/count contract as
    `write_predictions`: one line per impression, `impid [label_1,...]`
    (0/1 per candidate, original order)."""
    rows = read_raw_impressions(zip_path, split, has_labels=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in rows:
            labels = [int(c) for c in row["clicked"]]
            f.write(f"{row['raw_impression_id']} {json.dumps(labels, separators=(',', ':'))}\n")
    return len(rows)
