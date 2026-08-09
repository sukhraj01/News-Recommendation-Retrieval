"""Builds the full feature store from raw MIND + EB-NeRD zips.

Output layout: data/processed/{dataset}/{bundle}/{split}/{table}.parquet
(bundle dimension included — not just {dataset}/{split} — because MIND
ships multiple bundles, small and large, that both use "train"/"dev" split
names; writing them to the same path would silently overwrite one with the
other). EB-NeRD's articles.parquet is a single file shared across its
splits, per its raw structure, so it's written once at
data/processed/ebnerd/{bundle}/articles.parquet, not duplicated per split.
"""
from pathlib import Path

import pandas as pd

from src.datasets.ebnerd import parse_ebnerd_articles, parse_ebnerd_split
from src.datasets.mind import parse_mind_split, parse_mind_test_candidates
from src.pipeline.download import ensure_raw_data
from src.pipeline.schema import (
    ARTICLES_SCHEMA,
    CANDIDATES_SCHEMA,
    IMPRESSIONS_SCHEMA,
    USER_HISTORY_SCHEMA,
)
from src.pipeline.validators import validate_table
from src.utils.config import EBNERD_RAW_DIR, MIND_RAW_DIR, PROCESSED_DIR, RAW_DIR
from src.utils.io import write_parquet


def _write_table(
    df: pd.DataFrame, schema: dict, dataset: str, path: Path, sort_by: list[str]
) -> None:
    df = df.sort_values(sort_by).reset_index(drop=True)
    validate_table(df, schema, dataset)
    write_parquet(df, path)
    # Round-trip validation catches serialization drift (e.g. list columns
    # silently changing null representation on parquet read-back).
    reread = pd.read_parquet(path)
    validate_table(reread, schema, dataset)


def build_mind_split(zip_path: Path, split: str, out_dir: Path) -> None:
    result = parse_mind_split(zip_path, split)
    _write_table(result["articles"], ARTICLES_SCHEMA, "mind",
                 out_dir / split / "articles.parquet", ["article_id"])
    _write_table(result["impressions"], IMPRESSIONS_SCHEMA, "mind",
                 out_dir / split / "impressions.parquet", ["impression_id", "article_id"])
    _write_table(result["user_history"], USER_HISTORY_SCHEMA, "mind",
                 out_dir / split / "user_history.parquet", ["user_id"])


def build_mind_test(zip_path: Path, out_dir: Path) -> None:
    result = parse_mind_test_candidates(zip_path, "test")
    _write_table(result["articles"], ARTICLES_SCHEMA, "mind",
                 out_dir / "test" / "articles.parquet", ["article_id"])
    _write_table(result["candidates"], CANDIDATES_SCHEMA, "mind",
                 out_dir / "test" / "candidates.parquet", ["impression_id", "article_id"])


def build_ebnerd_bundle(
    zip_path: Path, out_dir: Path, splits: tuple[str, ...] = ("train", "validation")
) -> None:
    articles = parse_ebnerd_articles(zip_path)
    _write_table(articles, ARTICLES_SCHEMA, "ebnerd",
                 out_dir / "articles.parquet", ["article_id"])

    for split in splits:
        result = parse_ebnerd_split(zip_path, split)
        _write_table(result["impressions"], IMPRESSIONS_SCHEMA, "ebnerd",
                     out_dir / split / "impressions.parquet", ["impression_id", "article_id"])
        _write_table(result["user_history"], USER_HISTORY_SCHEMA, "ebnerd",
                     out_dir / split / "user_history.parquet", ["user_id"])


def build_all(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    include_mind_large: bool = False,
) -> None:
    """Rebuild the entire feature store from raw files.

    Default scope (fast tier): MINDsmall (train+dev) + ebnerd_demo
    (train+validation). MINDlarge is a separate, slower tier — pass
    include_mind_large=True to also build it (train+dev+test).
    """
    ensure_raw_data()
    print("✓ Raw data present")

    mind_raw = raw_dir / "mind"
    ebnerd_raw = raw_dir / "ebnerd"
    mind_out = processed_dir / "mind"
    ebnerd_out = processed_dir / "ebnerd"

    build_mind_split(mind_raw / "MINDsmall_train.zip", "train", mind_out / "small")
    build_mind_split(mind_raw / "MINDsmall_dev.zip", "dev", mind_out / "small")

    build_ebnerd_bundle(ebnerd_raw / "ebnerd_demo.zip", ebnerd_out / "demo")

    if include_mind_large:
        build_mind_split(mind_raw / "MINDlarge_train.zip", "train", mind_out / "large")
        build_mind_split(mind_raw / "MINDlarge_dev.zip", "dev", mind_out / "large")
        build_mind_test(mind_raw / "MINDlarge_test.zip", mind_out / "large")

    print("✓ Parsed unified schema")
    print("✓ Temporal split preserved (official train/dev/validation boundaries)")
    print("✓ Feature store built")


if __name__ == "__main__":
    build_all()
