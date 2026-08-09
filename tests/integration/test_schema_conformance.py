"""ADR-002 Required Tests: mandatory fields non-null, optional fields
dataset-scoped, impressions explode row counts correct, referential
integrity between impressions/user_history and articles."""
import pandas as pd
import pytest

from src.pipeline.schema import ARTICLES_SCHEMA, IMPRESSIONS_SCHEMA, USER_HISTORY_SCHEMA
from src.pipeline.validators import validate_table

MIND_TABLES = [
    ("mind/small/train", "mind"),
    ("mind/small/dev", "mind"),
]
EBNERD_SPLIT_TABLES = [
    ("ebnerd/demo/train", "ebnerd"),
    ("ebnerd/demo/validation", "ebnerd"),
]


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
