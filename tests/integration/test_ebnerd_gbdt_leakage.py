"""Leakage boundary tests for Candidate K's feature pipeline (ADR-013).

`tests/integration/test_leakage.py` already enforces ADR-001's temporal boundary
on the *processed* feature store. Candidate K does not read the processed store
— it reads raw EB-NeRD parquet directly, because the unified schema (ADR-002)
drops the context columns the model needs. That means the existing test does not
cover this code path at all, so the boundary is re-verified here against the raw
bundle the model actually consumes.

These run against the real `ebnerd_small` zip rather than a fixture: a synthetic
fixture can only confirm the code respects a boundary someone constructed, not
that EB-NeRD's real 21-day history window actually closes before its real
behaviors window. That second claim is the one ADR-013's audit table leans on
for 25 of its 60 features, so it is worth testing against real data.

Marked `slow` and skipped when the bundle is absent, matching this project's
existing convention for tests that depend on a large opt-in download.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retrieval.ebnerd_features import (
    BEHAVIOR_COLUMNS,
    FEATURE_NAMES,
    FORBIDDEN_RAW_COLUMNS,
    add_session_position_columns,
    build_feature_frame,
    build_history_popularity,
    build_user_profiles,
    load_article_table,
)
from src.utils.io import read_zip_member_bytes

_BUNDLE = Path("data/raw/ebnerd/ebnerd_small.zip")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _BUNDLE.exists(),
        reason=f"{_BUNDLE} not present (opt-in download, see README)",
    ),
]


@pytest.fixture(scope="module")
def article_table():
    return load_article_table(_BUNDLE)


@pytest.mark.parametrize("split", ["train", "validation"])
def test_history_window_closes_before_behaviors_window_opens(split):
    """The structural claim ADR-013's audit table rests on.

    If this ever fails, every `lt_*` / `st_*` / popularity feature is reading
    information from the future and the whole candidate is invalid.
    """
    hist = pd.read_parquet(
        io.BytesIO(read_zip_member_bytes(_BUNDLE, f"{split}/history.parquet")),
        columns=["impression_time_fixed"],
    )
    beh = pd.read_parquet(
        io.BytesIO(read_zip_member_bytes(_BUNDLE, f"{split}/behaviors.parquet")),
        columns=["impression_time"],
    )
    latest_history_click = max(t[-1] for t in hist["impression_time_fixed"] if len(t))
    earliest_impression = beh["impression_time"].min().to_datetime64()

    assert latest_history_click < earliest_impression, (
        f"split={split}: the latest history click ({latest_history_click}) is not "
        f"strictly before the earliest impression ({earliest_impression})"
    )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_no_user_has_a_history_click_after_their_own_first_impression(split):
    """Per-user form of the same boundary.

    The aggregate check above could pass while an individual user still had a
    history click after their own earliest impression, which is the shape of
    leak that would actually corrupt that user's profile.
    """
    hist = pd.read_parquet(
        io.BytesIO(read_zip_member_bytes(_BUNDLE, f"{split}/history.parquet")),
        columns=["user_id", "impression_time_fixed"],
    )
    beh = pd.read_parquet(
        io.BytesIO(read_zip_member_bytes(_BUNDLE, f"{split}/behaviors.parquet")),
        columns=["user_id", "impression_time"],
    )
    first_impression = beh.groupby("user_id")["impression_time"].min()

    violations = []
    for user_id, times in zip(hist["user_id"], hist["impression_time_fixed"]):
        if len(times) == 0 or user_id not in first_impression.index:
            continue
        if times[-1] >= first_impression.loc[user_id].to_datetime64():
            violations.append((int(user_id), times[-1], first_impression.loc[user_id]))

    assert not violations, (
        f"split={split}: {len(violations)} user(s) have a history click at or after "
        f"their own earliest impression: {violations[:5]}"
    )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_candidate_age_is_overwhelmingly_positive_and_news_fresh(article_table, split):
    """`article_age_h` must be positive for all but a negligible tail.

    Two jobs. First, it is the regression guard for the microsecond/nanosecond
    timestamp bug, which manifested as *every* age being hugely negative
    (median -467,870 h) — a bug that raised nothing and simply made the feature
    a constant.

    Second, it bounds a real EB-NeRD data artifact rather than pretending it
    away. A small number of candidates genuinely carry a `published_time` later
    than the impression that showed them — measured across the full bundle:

        train       641 / 2,585,747 rows (0.025%), 4 distinct articles, worst -0.7h
        validation   61 / 2,928,942 rows (0.002%), 9 distinct articles, worst -43.1h

    The likeliest explanation is that `published_time` reflects a later revision
    or republish for those articles. It is not treated as an exploitable leak,
    on evidence: click-through among those rows is *lower* than baseline (6.40%
    vs 9.04% on train, 1.64% vs 8.39% on validation), i.e. the anomaly is
    anti-correlated with the label, not predictive of it. The values are
    therefore left untransformed rather than silently clipped — clipping to zero
    would make these rows look maximally *fresh*, which is the one direction the
    model would actually reward. See ADR-013.
    """
    beh = pd.read_parquet(
        io.BytesIO(read_zip_member_bytes(_BUNDLE, f"{split}/behaviors.parquet")),
        columns=BEHAVIOR_COLUMNS + ["article_ids_clicked"],
    )
    beh = add_session_position_columns(beh)
    prof = build_user_profiles(_BUNDLE, split, article_table)
    pop = build_history_popularity(_BUNDLE, split, article_table)
    X, _ = build_feature_frame(beh, article_table, prof, pop)

    age = X[:, FEATURE_NAMES.index("article_age_h")]
    finite = age[np.isfinite(age)]
    assert finite.size, "article_age_h produced no finite values at all"

    negative_rate = float((finite < 0).mean())
    assert negative_rate < 0.001, (
        f"split={split}: {negative_rate:.4%} of candidates appear published after the "
        f"impression showing them (was 0.025% train / 0.002% validation when "
        f"characterised). A jump here means either the timestamp scaling regressed "
        f"or the bundle changed."
    )
    # The other side of the same guard: news candidates must be genuinely fresh.
    assert np.median(finite) < 72, (
        f"split={split}: median candidate age is {np.median(finite):.1f}h, implausible "
        f"for a news front page — the time base is probably wrong"
    )


def test_history_popularity_never_exceeds_history_click_volume(article_table):
    """Popularity counts must be bounded by the history window's own clicks.

    A count larger than the history total would mean the behaviors window (the
    labels) had leaked into the popularity feature.
    """
    hist = pd.read_parquet(
        io.BytesIO(read_zip_member_bytes(_BUNDLE, "validation/history.parquet")),
        columns=["article_id_fixed"],
    )
    total_history_clicks = int(sum(len(a) for a in hist["article_id_fixed"]))
    counts, decayed = build_history_popularity(_BUNDLE, "validation", article_table)

    assert counts.sum() <= total_history_clicks
    assert decayed.max() <= counts.max()
    assert counts.min() >= 0


def test_quarantined_columns_are_never_read_from_the_real_bundle():
    """The guard must refuse the real fields, not just fixture stand-ins."""
    from src.retrieval import ebnerd_features as F

    for column in sorted(FORBIDDEN_RAW_COLUMNS):
        with pytest.raises(ValueError, match="serving-time-unavailable"):
            F._read_zip_parquet(_BUNDLE, "articles.parquet", columns=["article_id", column])
