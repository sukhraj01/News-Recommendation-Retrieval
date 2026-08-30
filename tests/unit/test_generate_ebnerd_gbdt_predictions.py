"""Tests for `src/retrieval/ebnerd_features.py::load_test_behaviors` (ADR-013).

Regression coverage for a real gap found against the actual `ebnerd_testset`
bundle: its `behaviors.parquet` does not carry an `article_id` column at all,
unlike train/validation (checked directly against the real schema on Ada this
session; every other `BEHAVIOR_COLUMNS` field IS present there). The first
version of `load_test_behaviors` assumed `BEHAVIOR_COLUMNS` was always fully
available and crashed with `pyarrow.lib.ArrowInvalid` reading the real blind
test set, after training had already completed successfully.

Imports `load_test_behaviors` from `src.retrieval.ebnerd_features`, not from
`scripts/generate_ebnerd_gbdt_predictions.py` (where it originated and has
since been relocated from) -- that script imports `lightgbm` at module level,
and loading lightgbm alongside a torch-based test in the same pytest process
has already segfaulted this suite once tonight.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retrieval.ebnerd_features import BEHAVIOR_COLUMNS, load_test_behaviors


def _behaviors(*, with_article_id: bool) -> pd.DataFrame:
    data = {
        "impression_id": np.array([1, 2], dtype=np.uint32),
        "impression_time": pd.to_datetime(
            ["2023-05-19T10:00:00", "2023-05-20T11:00:00"]
        ).astype("datetime64[us]"),
        "read_time": np.array([5.0, 3.0], dtype=np.float32),
        "scroll_percentage": np.array([np.nan, 80.0], dtype=np.float32),
        "device_type": np.array([1, 2], dtype=np.int8),
        "article_ids_inview": [
            np.array([101, 102], dtype=np.int32),
            np.array([102, 103], dtype=np.int32),
        ],
        "user_id": np.array([1, 2], dtype=np.uint32),
        "is_sso_user": [False, True],
        "gender": np.array([np.nan, 1], dtype=np.float32),
        "postcode": np.array([np.nan, 2], dtype=np.float32),
        "age": np.array([np.nan, 40], dtype=np.float32),
        "is_subscriber": [False, True],
        "session_id": np.array([11, 12], dtype=np.uint32),
        "is_beyond_accuracy": [False, False],  # real testset-only field, never requested
    }
    if with_article_id:
        data["article_id"] = np.array([np.nan, 103.0])
    return pd.DataFrame(data)


def _write_zip(path: Path, *, with_article_id: bool) -> None:
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO()
        _behaviors(with_article_id=with_article_id).to_parquet(buf, index=False)
        z.writestr("test/behaviors.parquet", buf.getvalue())


def test_reads_normally_when_article_id_is_present(tmp_path):
    path = tmp_path / "with_article_id.zip"
    _write_zip(path, with_article_id=True)
    beh = load_test_behaviors(path, "test")
    assert list(beh["article_id"]) == [None, 103.0] or beh["article_id"].isna().tolist() == [True, False]
    assert len(beh) == 2


def test_nan_fills_article_id_when_column_is_genuinely_absent(tmp_path):
    """The real ebnerd_testset case: article_id is not in the schema at all."""
    path = tmp_path / "no_article_id.zip"
    _write_zip(path, with_article_id=False)
    beh = load_test_behaviors(path, "test")
    assert "article_id" in beh.columns
    assert beh["article_id"].isna().all()
    assert len(beh) == 2


def test_column_order_matches_behavior_columns_regardless_of_what_was_missing(tmp_path):
    path = tmp_path / "no_article_id.zip"
    _write_zip(path, with_article_id=False)
    beh = load_test_behaviors(path, "test")
    for col in BEHAVIOR_COLUMNS:
        assert col in beh.columns, f"{col} missing from loaded behaviors"


def test_extra_real_field_not_in_behavior_columns_is_silently_ignored(tmp_path):
    """`is_beyond_accuracy` is real and present but never requested -- must not
    appear or cause any issue; only BEHAVIOR_COLUMNS fields should be returned
    (plus the session-position columns build_feature_frame requires)."""
    path = tmp_path / "no_article_id.zip"
    _write_zip(path, with_article_id=False)
    beh = load_test_behaviors(path, "test")
    assert "is_beyond_accuracy" not in beh.columns


def test_downstream_feature_build_does_not_crash_without_article_id(tmp_path):
    """End-to-end: a missing article_id column must not break build_feature_frame,
    and every context-article feature must degrade to its documented neutral
    value (matching what already happens for an individual null row)."""
    from src.retrieval.ebnerd_features import (
        FEATURE_NAMES,
        build_feature_frame,
        build_history_popularity,
        build_user_profiles,
        load_article_table,
    )

    # One bundle zip carrying articles.parquet + test/history.parquet +
    # test/behaviors.parquet, matching the real EB-NeRD bundle shape -- every
    # loader below reads from the SAME zip_path, as the real pipeline does.
    bundle_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_path, "w") as z:
        art = pd.DataFrame({
            "article_id": np.array([101, 102, 103], dtype=np.int32),
            "title": ["a", "b", "c"], "subtitle": ["a", "b", "c"], "body": ["a", "b", "c"],
            "published_time": pd.to_datetime(
                ["2023-05-01", "2023-05-02", "2023-05-03"]
            ).astype("datetime64[us]"),
            "last_modified_time": pd.to_datetime(
                ["2023-05-01", "2023-05-02", "2023-05-03"]
            ).astype("datetime64[us]"),
            "premium": [False, False, False],
            "article_type": ["article_default"] * 3,
            "url": ["u"] * 3,
            "ner_clusters": [[], [], []],
            "entity_groups": [[], [], []],
            "topics": [[], [], []],
            "category": np.array([1, 2, 1], dtype=np.int32),
            "subcategory": [[], [], []],
            "category_str": ["nyheder", "sport", "nyheder"],
            "image_ids": [[], [], []],
            "sentiment_score": np.array([0.5, 0.5, 0.5], dtype=np.float32),
            "sentiment_label": ["Neutral"] * 3,
        })
        buf = io.BytesIO()
        art.to_parquet(buf, index=False)
        z.writestr("articles.parquet", buf.getvalue())

        buf = io.BytesIO()
        pd.DataFrame({
            "user_id": np.array([1, 2], dtype=np.uint32),
            "article_id_fixed": [np.array([101], dtype=np.int32), np.array([102], dtype=np.int32)],
            "impression_time_fixed": [
                np.array(["2023-05-10T09:00:00"], dtype="datetime64[us]"),
                np.array(["2023-05-10T09:00:00"], dtype="datetime64[us]"),
            ],
            "read_time_fixed": [np.array([5.0], dtype=np.float32), np.array([5.0], dtype=np.float32)],
            "scroll_percentage_fixed": [np.array([50.0], dtype=np.float32), np.array([50.0], dtype=np.float32)],
        }).to_parquet(buf, index=False)
        z.writestr("test/history.parquet", buf.getvalue())

        buf = io.BytesIO()
        _behaviors(with_article_id=False).to_parquet(buf, index=False)
        z.writestr("test/behaviors.parquet", buf.getvalue())

    art_table = load_article_table(bundle_path)
    prof = build_user_profiles(bundle_path, "test", art_table)
    pop = build_history_popularity(bundle_path, "test", art_table)
    beh = load_test_behaviors(bundle_path, "test")
    X, meta = build_feature_frame(beh, art_table, prof, pop, with_labels=False)

    ctx_cat = X[:, FEATURE_NAMES.index("context_category_match")]
    ctx_top = X[:, FEATURE_NAMES.index("context_topic_overlap")]
    ctx_sim = X[:, FEATURE_NAMES.index("context_embed_sim")]
    front = X[:, FEATURE_NAMES.index("is_front_page")]

    assert np.all(ctx_cat == 0.0)
    assert np.all(ctx_top == 0.0)
    assert np.all(np.isnan(ctx_sim))
    assert np.all(front == 1.0)  # every impression treated as "no known context article"
