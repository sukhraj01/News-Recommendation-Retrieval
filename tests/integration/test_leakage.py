"""ADR-001 Required Tests: no future information visible at serving time,
and row counts match the empirically-recorded evidence table (regression
guard against silent parsing bugs)."""
import pandas as pd
import pytest

from src.utils.config import EBNERD_DEMO_EVIDENCE, MIND_SMALL_EVIDENCE


@pytest.mark.parametrize("split", ["train", "validation"])
def test_ebnerd_no_impression_precedes_its_own_history_cutoff(processed_dir, split):
    """EB-NeRD only: MIND provides no per-click timestamps in history, so this
    check cannot be performed for MIND (see skip below) — not silently
    omitted, per CLAUDE.md's "document uncertainty explicitly"."""
    base = processed_dir / "ebnerd" / "demo" / split
    impressions = pd.read_parquet(base / "impressions.parquet")
    history = pd.read_parquet(base / "user_history.parquet")

    impression_min_by_user = impressions.groupby("user_id")["impression_time"].min()

    violations = []
    for _, row in history.iterrows():
        click_times = row["click_times"]
        if len(click_times) == 0:
            continue
        history_cutoff = max(click_times)
        user_id = row["user_id"]
        if user_id not in impression_min_by_user.index:
            continue
        earliest_impression = impression_min_by_user.loc[user_id]
        if history_cutoff >= earliest_impression:
            violations.append((user_id, history_cutoff, earliest_impression))

    assert not violations, (
        f"{len(violations)} user(s) have a history click at/after their own "
        f"earliest impression in split={split!r}: {violations[:5]}"
    )


@pytest.mark.skip(
    reason="MIND provides no per-click timestamps in history (only a static "
    "article-ID list), so there is no way to independently verify that "
    "history entries precede impression timestamps — this is documentation-"
    "tier evidence per ADR-001/ADR-002, not a benchmarkable fact."
)
def test_mind_no_impression_precedes_its_own_history_cutoff():
    pass


def test_static_history_invariant_held_at_full_mindsmall_train_scale(processed_dir):
    """No direct assertion needed: src.datasets.mind._build_user_history
    raises ValueError during the build if any user's history varies across
    their own impression rows. Reaching this point (the session-scoped
    `processed_dir` fixture already built successfully) means the invariant
    held for all 50,000 MINDsmall_train users, not just the 33,617 checked
    manually during ADR-002's original schema inspection."""
    assert (processed_dir / "mind" / "small" / "train" / "user_history.parquet").exists()


@pytest.mark.parametrize("split,expected", MIND_SMALL_EVIDENCE.items())
def test_mind_row_counts_match_adr001_evidence_table(processed_dir, split, expected):
    base = processed_dir / "mind" / "small" / split
    impressions = pd.read_parquet(base / "impressions.parquet")
    user_history = pd.read_parquet(base / "user_history.parquet")

    assert impressions["impression_id"].nunique() == expected["impressions"]
    assert len(user_history) == expected["users"]


@pytest.mark.parametrize("split,expected", EBNERD_DEMO_EVIDENCE.items())
def test_ebnerd_row_counts_match_adr001_evidence_table(processed_dir, split, expected):
    base = processed_dir / "ebnerd" / "demo" / split
    impressions = pd.read_parquet(base / "impressions.parquet")
    user_history = pd.read_parquet(base / "user_history.parquet")

    assert impressions["impression_id"].nunique() == expected["impressions"]
    assert len(user_history) == expected["users"]
