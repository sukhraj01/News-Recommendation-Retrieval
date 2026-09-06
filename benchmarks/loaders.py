"""Memory-bounded loaders for benchmarking against the *real* splits.

The brief asks for "a representative batch at real scale, not a toy sample".
Those two requirements pull in opposite directions on an 8GB machine:
MINDlarge-dev's impressions table alone is 14,085,557 candidate rows, and the
project has already lost a session to thrashing at that scale (PROJECT_STATE's
Risk register).

The resolution used throughout `benchmarks/` — and the thing that makes these
numbers honest rather than a toy — is:

    full-scale INDEX (every article, real corpus, real vocabulary)
    x
    sampled QUERIES (a deterministic systematic sample of real impressions)

Per-query latency is a function of the index's scale (corpus size drives the
sparse matvec's cost, the dense matvec's cost, and the vocabulary's size) and
of the query's own shape (history length, candidate count). Sampling *queries*
leaves both intact; shrinking the *index* would not. So the sampled numbers
extrapolate to the full run, and `profile_mind.py` checks that claim directly
by projecting a full-split time from them and reconciling it against ADR-006's
and ADR-008's independently-measured macro figures.

Sampling is systematic over the sorted unique impression ids (every Nth), not
a prefix and not `head()`. A prefix would be biased by whatever ordering the
feature store happened to write — and EB-NeRD in particular is known to
cluster a 200,000-row `is_beyond_accuracy` block, which a prefix sample could
land entirely inside or entirely outside (see PROJECT_STATE's Kaggle Part 2
notes, where the same reasoning drove the same choice).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import PROCESSED_DIR  # noqa: E402

SEED = 0


def _decode_id_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Decode dictionary-encoded id columns to plain strings.

    MIND's processed impressions store `impression_id`/`user_id`/`article_id`
    as Arrow dictionary columns, which pandas surfaces as `category` dtype.
    That is a real correctness trap for a benchmark, not a cosmetic one:
    `groupby` on a categorical defaults to `observed=False` and therefore
    yields a group for every category in the *whole split's* dictionary,
    including the ~254,000 users that were never sampled. Those empty groups
    then hand zero-length score arrays to the stage code — caught here by
    `_minmax` raising on a zero-size reduction, but it would just as easily
    have silently contributed thousands of near-zero timing samples and
    understated every per-query mean.

    Decoding costs little at the sample sizes these loaders return (tens of
    thousands of rows) and removes the trap at the source rather than
    requiring every downstream `groupby` to remember `observed=True`.
    """
    for col in ("impression_id", "user_id", "article_id"):
        if col in rows.columns and isinstance(rows[col].dtype, pd.CategoricalDtype):
            rows[col] = rows[col].astype(str)
    return rows


def split_dir(dataset: str, bundle: str, split: str) -> Path:
    return PROCESSED_DIR / dataset / bundle / split


def articles_path(dataset: str, bundle: str, split: str) -> Path:
    """EB-NeRD's article catalog is bundle-level (one `articles.parquet` shared
    by train/validation); MIND's is per-split. Both conventions already exist
    in `data/processed` — this resolves whichever is present rather than
    assuming one, so the same benchmark code drives both datasets."""
    per_split = split_dir(dataset, bundle, split) / "articles.parquet"
    if per_split.exists():
        return per_split
    return PROCESSED_DIR / dataset / bundle / "articles.parquet"


def load_articles(dataset: str, bundle: str, split: str, with_body: bool = False) -> pd.DataFrame:
    """Full article catalog — never sampled. `body` is excluded by default: no
    retrieval path in this project indexes it (ADR-005 fixes the indexed text
    at `title + " " + abstract`), and on MINDlarge-dev it is the single
    largest column."""
    cols = ["article_id", "title", "abstract", "category", "subcategory"]
    if with_body:
        cols.append("body")
    schema = pq.ParquetFile(articles_path(dataset, bundle, split)).schema_arrow.names
    cols = [c for c in cols if c in schema]
    return pq.read_table(articles_path(dataset, bundle, split), columns=cols).to_pandas()


def sample_impressions(
    dataset: str, bundle: str, split: str, n_impressions: int, seed: int = SEED
) -> tuple[pd.DataFrame, int]:
    """Deterministic systematic sample of `n_impressions` WHOLE impressions.

    Returns `(rows, total_impressions_in_split)`. Whole impressions, never
    individual candidate rows — a partial candidate list would change the
    per-impression scoring shape being measured (and, for the LightGBM arm,
    would break lambdarank's group structure the same way
    `run_ebnerd_gbdt_experiment.py --train-sample-impressions` guards against).

    Two passes over the file, both memory-bounded: the first reads only the
    dictionary-encoded `impression_id` column to enumerate impressions
    (measured: 1.5s / 0.45GB peak on MINDlarge-dev's 14.1M rows), the second
    reads only the sampled rows via a pushed-down predicate.
    """
    path = split_dir(dataset, bundle, split) / "impressions.parquet"
    ids = pc.unique(
        pq.read_table(path, columns=["impression_id"]).column("impression_id").combine_chunks()
    ).to_pylist()
    ids.sort()
    total = len(ids)

    if n_impressions >= total:
        keep = ids
    else:
        # Systematic (every Nth), phase-shifted by a seeded offset so repeated
        # runs at the same n are identical but a different seed genuinely
        # samples elsewhere rather than merely reordering the same rows.
        step = total / n_impressions
        offset = np.random.default_rng(seed).integers(0, max(1, int(step)))
        idx = np.floor(np.arange(n_impressions) * step).astype(np.int64) + int(offset)
        idx = np.unique(np.clip(idx, 0, total - 1))
        keep = [ids[i] for i in idx]

    rows = pq.read_table(
        path,
        columns=["impression_id", "user_id", "article_id", "clicked"],
        filters=[("impression_id", "in", set(keep))],
    ).to_pandas()
    # Same ordering the real harness uses (`run_ranking_eval.py`), so the
    # per-user score caches in BM25Scorer/EmbeddingScorer hit exactly as often
    # here as they do in production. Benchmarking an access pattern the real
    # pipeline never uses would measure the wrong thing.
    rows = _decode_id_columns(rows)
    rows = rows.sort_values(["user_id", "impression_id"], kind="stable").reset_index(drop=True)
    return rows, total


def load_history(dataset: str, bundle: str, split: str, user_ids=None) -> pd.DataFrame:
    """User click histories, optionally restricted to the sampled users.

    Restricting matters on MINDlarge-dev: all 255,990 histories cost real time
    and memory to materialise as Python lists, and a query-sampled benchmark
    only ever reads the sampled users' rows.
    """
    path = split_dir(dataset, bundle, split) / "user_history.parquet"
    filters = [("user_id", "in", set(user_ids))] if user_ids is not None else None
    return pq.read_table(
        path, columns=["user_id", "article_ids"], filters=filters
    ).to_pandas()


def embeddings_cache_path(dataset: str, bundle: str, split: str, model_slug_: str) -> Path:
    """Where `embed.py` already caches this split's article vectors. The
    benchmark reads that same cache rather than re-encoding — re-encoding is a
    separate, one-time stage measured explicitly (ADR-008's 264.4s), not part
    of per-query latency."""
    return articles_path(dataset, bundle, split).parent / "embeddings" / f"{model_slug_}.npy"


def sample_impressions_by_user(
    dataset: str, bundle: str, split: str, n_users: int, seed: int = SEED
) -> tuple[pd.DataFrame, dict]:
    """Systematic sample of `n_users` WHOLE USERS, with every one of their
    impressions kept.

    This, not `sample_impressions`, is the default for the per-query profiles,
    and the reason is a real measurement bug it avoids. `BM25Scorer` and
    `EmbeddingScorer` cache one full-corpus score vector per user and reuse it
    across that user's impressions (see their docstrings), so the *per-
    impression* cost of the expensive `score_all` stage depends on how many
    impressions each sampled user has. MINDlarge-dev has 376,471 impressions
    across 255,990 users — 1.47 impressions/user — but sampling impressions
    independently at n=1,500 drew 1,493 distinct users (1.005 impressions/
    user, measured), which would have inflated the amortised per-impression
    BM25 scoring cost by ~1.47x and pointed the "where should I invest"
    answer at the wrong stage.

    Sampling users preserves the real multiplicity by construction. Returns
    `(rows, meta)` where `meta` records the sampled and population-level
    impressions-per-user so the fidelity of the sample is reported, not
    assumed.
    """
    path = split_dir(dataset, bundle, split) / "impressions.parquet"
    tbl = pq.read_table(path, columns=["impression_id", "user_id"])
    users = pc.unique(tbl.column("user_id").combine_chunks()).to_pylist()
    users.sort()
    total_users = len(users)
    total_impressions = len(pc.unique(tbl.column("impression_id").combine_chunks()))
    del tbl

    if n_users >= total_users:
        keep = users
    else:
        step = total_users / n_users
        offset = np.random.default_rng(seed).integers(0, max(1, int(step)))
        idx = np.floor(np.arange(n_users) * step).astype(np.int64) + int(offset)
        idx = np.unique(np.clip(idx, 0, total_users - 1))
        keep = [users[i] for i in idx]

    rows = pq.read_table(
        path,
        columns=["impression_id", "user_id", "article_id", "clicked"],
        filters=[("user_id", "in", set(keep))],
    ).to_pandas()
    rows = _decode_id_columns(rows)
    rows = rows.sort_values(["user_id", "impression_id"], kind="stable").reset_index(drop=True)

    n_imp = rows["impression_id"].nunique()
    n_usr = rows["user_id"].nunique()
    meta = {
        "sampled_users": int(n_usr),
        "sampled_impressions": int(n_imp),
        "sampled_candidate_rows": int(len(rows)),
        "sampled_impressions_per_user": round(n_imp / n_usr, 4) if n_usr else 0.0,
        "population_users": int(total_users),
        "population_impressions": int(total_impressions),
        "population_impressions_per_user": round(total_impressions / total_users, 4),
        "population_candidate_rows": None,  # filled by callers that know it
        "sample_seed": seed,
        "sampling": "systematic-by-user",
    }
    return rows, meta
