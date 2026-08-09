"""Same code + same data = same results, always (CLAUDE.md).

Compares value-identical re-read DataFrames across two independent build
runs, not raw file-byte-hashes — parquet footers legitimately embed
pyarrow-version metadata that can differ across environments even when the
data values are identical, so byte-identity would be a brittle, wrong
target for "same results."
"""
import json

import pandas as pd
import pytest

from src.datasets.ebnerd import _combine_ebnerd_entities
from src.datasets.mind import _combine_mind_entities
from src.pipeline.orchestrator import build_all

_TABLES = [
    "mind/small/train/articles.parquet",
    "mind/small/train/impressions.parquet",
    "mind/small/train/user_history.parquet",
    "mind/small/dev/articles.parquet",
    "mind/small/dev/impressions.parquet",
    "mind/small/dev/user_history.parquet",
    "ebnerd/demo/articles.parquet",
    "ebnerd/demo/train/impressions.parquet",
    "ebnerd/demo/train/user_history.parquet",
    "ebnerd/demo/validation/impressions.parquet",
    "ebnerd/demo/validation/user_history.parquet",
]


@pytest.fixture(scope="module")
def two_independent_builds(tmp_path_factory):
    dir_a = tmp_path_factory.mktemp("build_a")
    dir_b = tmp_path_factory.mktemp("build_b")
    build_all(processed_dir=dir_a)
    build_all(processed_dir=dir_b)
    return dir_a, dir_b


@pytest.mark.parametrize("rel_path", _TABLES)
def test_repeated_build_is_value_identical(two_independent_builds, rel_path):
    dir_a, dir_b = two_independent_builds
    df_a = pd.read_parquet(dir_a / rel_path)
    df_b = pd.read_parquet(dir_b / rel_path)
    pd.testing.assert_frame_equal(df_a, df_b)


def test_mind_entity_json_serialization_is_deterministic():
    title_entities = json.dumps([{"WikidataId": "Q2", "Label": "B"},
                                  {"WikidataId": "Q1", "Label": "A"}])
    result_a = _combine_mind_entities(title_entities, "[]")
    result_b = _combine_mind_entities(title_entities, "[]")
    assert result_a == result_b


def test_ebnerd_entity_json_serialization_is_deterministic():
    result_a = _combine_ebnerd_entities(["Foo", "Bar"], ["PER", "ORG"])
    result_b = _combine_ebnerd_entities(["Foo", "Bar"], ["PER", "ORG"])
    assert result_a == result_b
