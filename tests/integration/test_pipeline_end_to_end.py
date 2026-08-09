"""Full build_all() -> expected file layout, run against the fast tier
(MINDsmall + ebnerd_demo). MINDlarge is a separate, slow, manually-run tier
(marked `slow`, skipped by default) — see pyproject.toml's pytest markers."""
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.orchestrator import build_all


def test_expected_file_layout_exists(processed_dir):
    expected = [
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
    for rel in expected:
        assert (processed_dir / rel).exists(), f"missing {rel}"


def test_default_build_does_not_touch_mindlarge(processed_dir):
    """include_mind_large defaults to False — the fast tier must never
    silently pull in the large bundle."""
    assert not (processed_dir / "mind" / "large").exists()


@pytest.mark.slow
def test_mindlarge_test_never_appears_in_impressions(tmp_path):
    """MINDlarge_test's unlabeled candidates must only ever land in
    candidates.parquet, never impressions.parquet (which requires the
    unconditionally-mandatory `clicked` label)."""
    build_all(processed_dir=tmp_path, include_mind_large=True)

    candidates = pd.read_parquet(tmp_path / "mind" / "large" / "test" / "candidates.parquet")
    assert "clicked" not in candidates.columns
    assert not (tmp_path / "mind" / "large" / "test" / "impressions.parquet").exists()
