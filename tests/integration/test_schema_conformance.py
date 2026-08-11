"""ADR-002 Required Tests: mandatory fields non-null, optional fields
dataset-scoped, impressions explode row counts correct, referential
integrity between impressions/user_history and articles."""
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.orchestrator import build_ebnerd_bundle
from src.pipeline.schema import ARTICLES_SCHEMA, IMPRESSIONS_SCHEMA, USER_HISTORY_SCHEMA
from src.pipeline.validators import validate_table
from src.utils.config import EBNERD_RAW_DIR, PROCESSED_DIR

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


def test_ebnerd_small_validation_has_zero_cold_start_users(ebnerd_small_dir):
    """Documents (not just asserts) the ADR-002 addendum finding: ebnerd_small's
    active-user-filtered construction guarantees history length >= 5, exactly
    like ebnerd_demo — the warm/cold BM25 comparison remains structurally
    unavailable for EB-NeRD at this tier too, not a demo-only artifact."""
    hist = pd.read_parquet(ebnerd_small_dir / "ebnerd" / "small" / "validation" / "user_history.parquet")
    history_len = hist["article_ids"].map(len)
    assert history_len.min() >= 5
    assert (history_len < 5).sum() == 0
