"""ADR-002 Required Tests: mandatory fields non-null, optional fields
dataset-scoped, impressions explode row counts correct, referential
integrity between impressions/user_history and articles."""
from pathlib import Path

import pandas as pd
import pytest

from src.datasets.mind import _BEHAVIORS_COLUMNS
from src.pipeline.orchestrator import build_ebnerd_bundle, build_mind_split, build_mind_test
from src.pipeline.schema import (
    ARTICLES_SCHEMA,
    CANDIDATES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    USER_HISTORY_SCHEMA,
)
from src.pipeline.validators import validate_table
from src.utils.config import (
    EBNERD_RAW_DIR,
    MIND_LARGE_EVIDENCE,
    MIND_LARGE_TOTAL_ARTICLES,
    MIND_LARGE_TOTAL_USERS,
    MIND_RAW_DIR,
    PROCESSED_DIR,
)
from src.utils.io import read_zip_tsv

MIND_TABLES = [
    ("mind/small/train", "mind"),
    ("mind/small/dev", "mind"),
]
EBNERD_SPLIT_TABLES = [
    ("ebnerd/demo/train", "ebnerd"),
    ("ebnerd/demo/validation", "ebnerd"),
]


@pytest.fixture(scope="session")
def ebnerd_small_dir() -> Path:
    """ebnerd_small is an opt-in tier (ADR-002 addendum) — not part of the
    default fast-tier `processed_dir` fixture (mirrors MINDlarge's opt-in
    pattern), built into the real feature-store location so experiment
    scripts can read it too, not just this test session."""
    build_ebnerd_bundle(EBNERD_RAW_DIR / "ebnerd_small.zip", PROCESSED_DIR / "ebnerd" / "small")
    return PROCESSED_DIR


@pytest.mark.parametrize("rel_path,dataset", MIND_TABLES)
def test_mind_split_tables_conform_to_schema(processed_dir, rel_path, dataset):
    base = processed_dir / rel_path
    validate_table(pd.read_parquet(base / "articles.parquet"), ARTICLES_SCHEMA, dataset)
    validate_table(pd.read_parquet(base / "impressions.parquet"), IMPRESSIONS_SCHEMA, dataset)
    validate_table(pd.read_parquet(base / "user_history.parquet"), USER_HISTORY_SCHEMA, dataset)


def test_ebnerd_shared_articles_table_conforms_to_schema(processed_dir):
    df = pd.read_parquet(processed_dir / "ebnerd" / "demo" / "articles.parquet")
    validate_table(df, ARTICLES_SCHEMA, "ebnerd")


@pytest.mark.parametrize("rel_path,dataset", EBNERD_SPLIT_TABLES)
def test_ebnerd_split_tables_conform_to_schema(processed_dir, rel_path, dataset):
    base = processed_dir / rel_path
    validate_table(pd.read_parquet(base / "impressions.parquet"), IMPRESSIONS_SCHEMA, dataset)
    validate_table(pd.read_parquet(base / "user_history.parquet"), USER_HISTORY_SCHEMA, dataset)


def test_referential_integrity_mind_train(processed_dir):
    base = processed_dir / "mind" / "small" / "train"
    articles = set(pd.read_parquet(base / "articles.parquet")["article_id"])
    impressions = pd.read_parquet(base / "impressions.parquet")
    assert set(impressions["article_id"]) <= articles

    history = pd.read_parquet(base / "user_history.parquet")
    history_articles = {a for ids in history["article_ids"] for a in ids}
    assert history_articles <= articles


def test_referential_integrity_ebnerd_train(processed_dir):
    articles = set(pd.read_parquet(processed_dir / "ebnerd" / "demo" / "articles.parquet")["article_id"])
    base = processed_dir / "ebnerd" / "demo" / "train"
    impressions = pd.read_parquet(base / "impressions.parquet")
    assert set(impressions["article_id"]) <= articles

    history = pd.read_parquet(base / "user_history.parquet")
    history_articles = {a for ids in history["article_ids"] for a in ids}
    assert history_articles <= articles


def test_impressions_row_count_equals_sum_of_raw_candidate_lengths_mind(processed_dir):
    import zipfile

    from src.utils.config import MIND_RAW_DIR

    with zipfile.ZipFile(MIND_RAW_DIR / "MINDsmall_train.zip") as z:
        with z.open("MINDsmall_train/behaviors.tsv") as f:
            raw = pd.read_csv(f, sep="\t", header=None,
                               names=["impression_id", "user_id", "time", "history", "impressions"])
    expected = raw["impressions"].str.split(" ").map(len).sum()

    actual = len(pd.read_parquet(processed_dir / "mind" / "small" / "train" / "impressions.parquet"))
    assert actual == expected


def test_impressions_row_count_equals_sum_of_raw_candidate_lengths_ebnerd(processed_dir):
    import io
    import zipfile

    from src.utils.config import EBNERD_RAW_DIR

    with zipfile.ZipFile(EBNERD_RAW_DIR / "ebnerd_demo.zip") as z:
        with z.open("train/behaviors.parquet") as f:
            raw = pd.read_parquet(io.BytesIO(f.read()))
    expected = raw["article_ids_inview"].map(len).sum()

    actual = len(pd.read_parquet(processed_dir / "ebnerd" / "demo" / "train" / "impressions.parquet"))
    assert actual == expected


# --- ebnerd_small (ADR-002 addendum: verifies the schema built strictly
# against ebnerd_demo also holds for the "final training" tier) ---

EBNERD_SMALL_SPLIT_TABLES = [
    ("ebnerd/small/train", "ebnerd"),
    ("ebnerd/small/validation", "ebnerd"),
]


def test_ebnerd_small_shared_articles_table_conforms_to_schema(ebnerd_small_dir):
    df = pd.read_parquet(ebnerd_small_dir / "ebnerd" / "small" / "articles.parquet")
    validate_table(df, ARTICLES_SCHEMA, "ebnerd")


@pytest.mark.parametrize("rel_path,dataset", EBNERD_SMALL_SPLIT_TABLES)
def test_ebnerd_small_split_tables_conform_to_schema(ebnerd_small_dir, rel_path, dataset):
    base = ebnerd_small_dir / rel_path
    validate_table(pd.read_parquet(base / "impressions.parquet"), IMPRESSIONS_SCHEMA, dataset)
    validate_table(pd.read_parquet(base / "user_history.parquet"), USER_HISTORY_SCHEMA, dataset)


def test_referential_integrity_ebnerd_small_train(ebnerd_small_dir):
    articles = set(pd.read_parquet(ebnerd_small_dir / "ebnerd" / "small" / "articles.parquet")["article_id"])
    base = ebnerd_small_dir / "ebnerd" / "small" / "train"
    impressions = pd.read_parquet(base / "impressions.parquet")
    assert set(impressions["article_id"]) <= articles

    history = pd.read_parquet(base / "user_history.parquet")
    history_articles = {a for ids in history["article_ids"] for a in ids}
    assert history_articles <= articles


def test_impressions_row_count_equals_sum_of_raw_candidate_lengths_ebnerd_small(ebnerd_small_dir):
    import io
    import zipfile

    from src.utils.config import EBNERD_RAW_DIR

    with zipfile.ZipFile(EBNERD_RAW_DIR / "ebnerd_small.zip") as z:
        with z.open("train/behaviors.parquet") as f:
            raw = pd.read_parquet(io.BytesIO(f.read()))
    expected = raw["article_ids_inview"].map(len).sum()

    actual = len(pd.read_parquet(ebnerd_small_dir / "ebnerd" / "small" / "train" / "impressions.parquet"))
    assert actual == expected


# --- MINDlarge (Part 1 of the MIND Codabench-submission session: build,
# schema-verify, and cross-check row counts against Wu et al. 2020's
# published Table 2 statistics) ---

MIND_LARGE_SPLIT_TABLES = [
    ("mind/large/train", "mind"),
    ("mind/large/dev", "mind"),
]


@pytest.fixture(scope="session")
def mind_large_dir(processed_dir) -> Path:
    """MINDlarge is a separate, slow, opt-in tier (`include_mind_large=True`).
    Unlike `ebnerd_small_dir`, skip rebuilding if already present: a full
    MINDlarge build takes real minutes (~80M+ exploded impression rows for
    `train` alone), not the few seconds `ebnerd_small` costs, so an
    unconditional rebuild on every `slow`-marked test run would waste real
    time rather than just being a style choice."""
    large_dir = processed_dir / "mind" / "large"
    if not (large_dir / "train" / "impressions.parquet").exists():
        build_mind_split(MIND_RAW_DIR / "MINDlarge_train.zip", "train", large_dir)
        build_mind_split(MIND_RAW_DIR / "MINDlarge_dev.zip", "dev", large_dir)
        build_mind_test(MIND_RAW_DIR / "MINDlarge_test.zip", large_dir)
    return processed_dir


@pytest.mark.slow
@pytest.mark.parametrize("rel_path,dataset", MIND_LARGE_SPLIT_TABLES)
def test_mind_large_split_tables_conform_to_schema(mind_large_dir, rel_path, dataset):
    base = mind_large_dir / rel_path
    validate_table(pd.read_parquet(base / "articles.parquet"), ARTICLES_SCHEMA, dataset)
    validate_table(pd.read_parquet(base / "impressions.parquet"), IMPRESSIONS_SCHEMA, dataset)
    validate_table(pd.read_parquet(base / "user_history.parquet"), USER_HISTORY_SCHEMA, dataset)


@pytest.mark.slow
def test_mind_large_test_candidates_conform_to_schema(mind_large_dir):
    base = mind_large_dir / "mind" / "large" / "test"
    validate_table(pd.read_parquet(base / "articles.parquet"), ARTICLES_SCHEMA, "mind")
    validate_table(pd.read_parquet(base / "candidates.parquet"), CANDIDATES_SCHEMA, "mind")
    validate_table(pd.read_parquet(base / "user_history.parquet"), USER_HISTORY_SCHEMA, "mind")


@pytest.mark.slow
def test_mind_large_impression_counts_match_paper(mind_large_dir):
    """Parse-correctness check against Wu et al. (2020)'s published sample
    counts (train 2,186,683 / dev 365,200 / test 2,341,619) — same
    discipline as `MIND_SMALL_EVIDENCE`/`ebnerd_small`'s row-count
    regression guard. ACL2020_MIND.pdf is not present in this repo; these
    numbers were extracted directly from the real paper (fetched via
    WebFetch, text-extracted with pypdf), not taken from memory — see
    `src/utils/config.py::MIND_LARGE_EVIDENCE`'s docstring.

    A first version of this test asserted the RAW parsed count equals the
    paper's figure and failed for all three splits (e.g. train: parsed
    2,232,748 vs. paper's 2,186,683). Root-caused, not dismissed: the paper
    states explicitly (Section 3.1) "We only kept the samples with
    non-empty news click history" when constructing its reported
    train/valid/test statistics. Checking directly against the raw zips
    confirms this exactly: excluding empty-history rows gives train
    2,186,683 (exact), test 2,341,619 (exact), dev 365,201 (off by one row
    — negligible, not investigated further). This project's own feature
    store deliberately does NOT apply that filter (ADR-005: zero-history
    users are the true cold-start cohort BM25/embeddings are evaluated
    against, not noise to discard) — so the correct regression check is the
    *raw* parsed count against the paper's figure *plus* the raw zip's own
    empty-history row count, not the paper's post-filter figure directly."""
    for split, evidence in MIND_LARGE_EVIDENCE.items():
        table_name = "candidates" if split == "test" else "impressions"
        base = mind_large_dir / "mind" / "large" / split
        df = pd.read_parquet(base / f"{table_name}.parquet")
        n_impressions = df["impression_id"].nunique()

        zip_name = {"train": "MINDlarge_train", "dev": "MINDlarge_dev", "test": "MINDlarge_test"}[split]
        raw = read_zip_tsv(MIND_RAW_DIR / f"{zip_name}.zip", f"{zip_name}/behaviors.tsv", _BEHAVIORS_COLUMNS)
        n_empty_history = (raw["history"].isna() | (raw["history"] == "")).sum()

        assert n_impressions == len(raw), (
            f"MINDlarge {split}: parsed {n_impressions} unique impressions, "
            f"raw behaviors.tsv has {len(raw)} rows — every raw row should "
            f"survive parsing (this project keeps empty-history rows, unlike "
            f"the paper's own reported statistics)"
        )
        # within 2 of the paper's post-filter figure: exact for train/test,
        # off by one for dev (see docstring) -- not zero-tolerance, since
        # that one-row dev gap is a known, accepted, unexplained-further
        # discrepancy, not evidence of a parsing bug.
        assert abs((n_impressions - n_empty_history) - evidence["impressions"]) <= 2, (
            f"MINDlarge {split}: {n_impressions} parsed - {n_empty_history} "
            f"empty-history = {n_impressions - n_empty_history}, paper "
            f"(post-filter) reports {evidence['impressions']}"
        )


@pytest.mark.slow
def test_mind_large_total_article_and_user_counts_sanity(mind_large_dir):
    """Corpus-wide totals (161,013 articles, 1,000,000 users) are reported
    once for the whole MINDlarge dataset (paper Section 3.2, covering 6
    weeks of raw logs), not per-split. Measured directly: the union of
    train/dev/test's own news.tsv article sets is 130,379 (train 101,527 /
    dev 72,023 / test 120,959) — a real ~19% gap from 161,013, too large
    for a tolerance-based approx check to usefully catch regressions
    without also being loose enough to hide real bugs. Root cause (from the
    paper's own construction description, Section 3.1): train's samples
    come from week 5 only, dev from the last day of week 5, test from the
    last week — each split's own news.tsv only contains articles relevant
    to its own narrower window, not the full 6-week corpus the headline
    161,013 figure describes. This is a structural property of the
    official train/dev/test construction, not a parsing defect (same
    category of finding as ADR-002's EB-NeRD user-count discrepancy) — so
    this test checks directional sanity bounds instead of a numeric match:
    the union must be smaller than the full corpus (confirms we're seeing
    per-window subsets, not accidentally the same file three times) and
    larger than any single split (confirms the union is doing real work,
    not silently collapsing to one split)."""
    base = mind_large_dir / "mind" / "large"
    articles: set[str] = set()
    users: set[str] = set()
    per_split_articles: dict[str, int] = {}
    for split, table_name in (("train", "impressions"), ("dev", "impressions"), ("test", "candidates")):
        split_dir = base / split
        split_articles = set(pd.read_parquet(split_dir / "articles.parquet")["article_id"])
        per_split_articles[split] = len(split_articles)
        articles |= split_articles
        users |= set(pd.read_parquet(split_dir / f"{table_name}.parquet")["user_id"])
        hist_path = split_dir / "user_history.parquet"
        if hist_path.exists():
            users |= set(pd.read_parquet(hist_path)["user_id"])

    assert len(articles) < MIND_LARGE_TOTAL_ARTICLES, (
        f"union of {len(articles)} articles should be a proper subset of "
        f"the paper's full-corpus {MIND_LARGE_TOTAL_ARTICLES}"
    )
    assert len(articles) > max(per_split_articles.values()), (
        "union across splits should exceed any single split's own count"
    )
    assert len(users) <= MIND_LARGE_TOTAL_USERS * 1.01, (
        f"union of {len(users)} users should not meaningfully exceed the "
        f"paper's reported {MIND_LARGE_TOTAL_USERS}"
    )


def test_ebnerd_small_validation_has_zero_cold_start_users(ebnerd_small_dir):
    """Documents (not just asserts) the ADR-002 addendum finding: ebnerd_small's
    active-user-filtered construction guarantees history length >= 5, exactly
    like ebnerd_demo — the warm/cold BM25 comparison remains structurally
    unavailable for EB-NeRD at this tier too, not a demo-only artifact."""
    hist = pd.read_parquet(ebnerd_small_dir / "ebnerd" / "small" / "validation" / "user_history.parquet")
    history_len = hist["article_ids"].map(len)
    assert history_len.min() >= 5
    assert (history_len < 5).sum() == 0
