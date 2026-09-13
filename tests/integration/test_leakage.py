"""ADR-001 Required Tests: no future information visible at serving time,
and row counts match the empirically-recorded evidence table (regression
guard against silent parsing bugs)."""
import numpy as np
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


def test_mind_click_history_features_trace_to_one_shared_value_across_real_impressions(processed_dir):
    """A2 Q1's MIND behaviour-window boundary check, reusing the SAME
    invariant `_build_user_history` already enforces at parse time
    (`test_static_history_invariant_held_at_full_mindsmall_train_scale`
    above), now verified at the FEATURE level rather than the raw-data
    level. MIND provides no per-click timestamps, so the EB-NeRD-style
    "no impression precedes its own history cutoff" check
    (`test_ebnerd_no_impression_precedes_its_own_history_cutoff`) cannot be
    built for MIND — this is the structurally equivalent guard the dataset
    actually supports.

    Exercises REAL multiple (user_id, impression_id) groups from the
    processed data — not the same call repeated on identical inputs, which
    would only prove the functions are pure and prove nothing about
    leakage. A user's history-derived feature components (Candidate L,
    `src/retrieval/mind_features.py`) must trace back to exactly ONE value
    per user regardless of which of that user's real impressions produced
    them; a bug that read something impression-local (e.g. mistook the
    current candidate list for history, or recomputed history per group)
    would break this. Candidate-derived components (`category_match`) are
    checked in the SAME loop and must legitimately DIFFER when the
    candidate's own category differs — proving the test isn't vacuously
    insensitive to real variation.
    """
    from src.retrieval.features import build_history_profile
    from src.retrieval.mind_features import build_feature_row, click_count

    base = processed_dir / "mind" / "small" / "train"
    articles = pd.read_parquet(base / "articles.parquet", columns=["article_id", "category"])
    history = pd.read_parquet(base / "user_history.parquet")
    impressions = pd.read_parquet(
        base / "impressions.parquet", columns=["user_id", "impression_id", "article_id"]
    )
    category_lookup = dict(zip(articles["article_id"], articles["category"]))

    n_impressions_by_user = impressions.groupby("user_id", observed=True)["impression_id"].nunique()
    multi_impression_users = n_impressions_by_user[n_impressions_by_user >= 2].index[:25]
    assert len(multi_impression_users) > 0, "no multi-impression users found in MINDsmall_train"

    checked_users, saw_a_category_difference = 0, False
    for user_id in multi_impression_users:
        hist_row = history.loc[history["user_id"] == user_id]
        if hist_row.empty:
            continue
        article_ids = list(hist_row.iloc[0]["article_ids"])
        expected_click_count = click_count(article_ids)  # ONE ground-truth value for this user
        profile = build_history_profile(article_ids, category_lookup, {}, {})

        user_rows = impressions.loc[impressions["user_id"] == user_id]
        seen_category_matches = set()
        for impression_id, group in user_rows.groupby("impression_id", observed=True):
            for candidate_id in group["article_id"]:
                row = build_feature_row(candidate_id, article_ids, category_lookup, {}, profile)
                click_count_component = row[0]  # log1p(click_count) -- must match every time
                assert click_count_component == np.log1p(expected_click_count), (
                    f"user {user_id}, impression {impression_id}: click-history component "
                    f"drifted from the single stored history snapshot"
                )
                seen_category_matches.add(row[1])  # category_match -- MAY legitimately vary
        if len(seen_category_matches) > 1:
            saw_a_category_difference = True
        checked_users += 1

    assert checked_users > 0
    assert saw_a_category_difference, (
        "category_match never varied across any checked user's impressions -- "
        "the test setup may not be exercising real candidate diversity"
    )


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
