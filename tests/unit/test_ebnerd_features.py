"""Unit tests for `src/retrieval/ebnerd_features.py` (Candidate K, ADR-013).

Three things are worth testing here beyond ordinary correctness, because each
one failed silently rather than loudly during development:

1. **Timestamp scaling.** EB-NeRD stores datetimes at microsecond resolution.
   The reflexive `int64 / 1e9` conversion under-scales by 1000x and does not
   raise — it just turns `article_age_h` into a near-constant, which a model
   happily trains on. `test_epoch_seconds_*` pins the conversion.
2. **The leakage contract.** ADR-009 quarantined the article-lifetime
   aggregates; ADR-013 adds the post-click outcome fields. The guard is only
   worth having if it is enforced at the I/O boundary, so it is tested there.
3. **Row alignment.** The feature matrix, the label vector and the LightGBM
   group vector are three parallel structures built from the same explode. If
   they ever disagree, the model trains against shifted labels and still
   reports a plausible-looking AUC.

The fixture is built locally rather than reusing
`tests/fixtures/ebnerd_demo_sample.zip`, which predates this module and lacks
the context columns it reads (`device_type`, `age`, `sentiment_score`,
`premium`, ...). Extending the shared fixture would touch 46 existing tests for
no benefit.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.retrieval import ebnerd_features as F
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

# Article 101 is published well before every impression; 104 is published
# between the history window and the impressions, i.e. an item-cold-start
# candidate with no history popularity at all.
_PUBLISHED = pd.to_datetime(
    ["2023-04-01 08:00:00", "2023-04-02 08:00:00", "2023-04-03 08:00:00", "2023-05-19 06:00:00"]
).astype("datetime64[us]")


def _articles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": np.array([101, 102, 103, 104], dtype=np.int32),
            "title": ["Title A", "Title BB", "Title CCC", "Title DDDD"],
            "subtitle": ["a", "b", "c", "d"],
            "body": ["Body A", "Body BB", "Body CCC", "Body DDDD"],
            "published_time": _PUBLISHED,
            "last_modified_time": _PUBLISHED,
            "premium": [False, True, False, False],
            "article_type": [
                "article_default",
                "article_default",
                "article_video_standalone",
                "article_default",
            ],
            "url": ["u1", "u2", "u3", "u4"],
            "ner_clusters": [["Foo"], [], ["Bar", "Baz"], ["Foo"]],
            "entity_groups": [["PER"], [], ["ORG", "LOC"], ["PER"]],
            "topics": [["politik"], ["fodbold"], [], ["politik", "krimi"]],
            "category": np.array([1, 2, 1, 1], dtype=np.int32),
            "subcategory": [[1], [2, 3], [1], []],
            "category_str": ["nyheder", "sport", "nyheder", "nyheder"],
            "image_ids": [[1], [2], [3], [4]],
            # Present in the fixture precisely so the guard has something real to refuse.
            "total_inviews": np.array([10, 20, 30, 40], dtype=np.int32),
            "total_pageviews": np.array([1, 2, 3, 4], dtype=np.int32),
            "total_read_time": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "sentiment_score": np.array([0.9, 0.1, 0.5, 0.7], dtype=np.float32),
            "sentiment_label": ["Positive", "Negative", "Neutral", "Positive"],
        }
    )


def _history() -> pd.DataFrame:
    """User 1 has *drifted*; user 2 reads sport only.

    User 1's history is deliberately mixed and time-separated: one old sport
    click, then two recent nyheder clicks. That is what makes the long-term and
    short-term profiles genuinely disagree — a single-category history would
    make them identical and the decomposition untestable.
    """
    return pd.DataFrame(
        {
            "user_id": np.array([1, 2], dtype=np.uint32),
            "article_id_fixed": [
                np.array([102, 101, 101], dtype=np.int32),
                np.array([102], dtype=np.int32),
            ],
            "impression_time_fixed": [
                np.array(
                    ["2023-05-01T09:00:00", "2023-05-16T09:00:00", "2023-05-17T09:00:00"],
                    dtype="datetime64[us]",
                ),
                np.array(["2023-05-03T09:00:00"], dtype="datetime64[us]"),
            ],
            "read_time_fixed": [
                np.array([10.0, 20.0, 30.0], dtype=np.float32),
                np.array([40.0], dtype=np.float32),
            ],
            "scroll_percentage_fixed": [
                np.array([50.0, 100.0, np.nan], dtype=np.float32),
                np.array([80.0], dtype=np.float32),
            ],
        }
    )


def _behaviors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "impression_id": np.array([1, 2], dtype=np.uint32),
            "article_id": np.array([np.nan, 103.0]),  # NaN -> front page
            "impression_time": np.array(
                ["2023-05-19T10:00:00", "2023-05-20T11:00:00"], dtype="datetime64[us]"
            ),
            "read_time": np.array([12.0, 30.0], dtype=np.float32),
            "scroll_percentage": np.array([50.0, np.nan], dtype=np.float32),
            "device_type": np.array([1, 2], dtype=np.int8),
            "article_ids_inview": [
                np.array([101, 102, 104], dtype=np.int32),
                np.array([102, 103], dtype=np.int32),
            ],
            "article_ids_clicked": [
                np.array([102], dtype=np.int32),
                np.array([103], dtype=np.int32),
            ],
            "user_id": np.array([1, 2], dtype=np.uint32),
            "is_sso_user": [False, True],
            "gender": np.array([np.nan, 1.0]),
            "postcode": np.array([np.nan, 2.0]),
            "age": np.array([np.nan, 40.0]),
            "is_subscriber": [False, True],
            "session_id": np.array([11, 12], dtype=np.uint32),
            "next_read_time": np.array([7.0, 8.0], dtype=np.float32),
            "next_scroll_percentage": np.array([22.0, 33.0], dtype=np.float32),
        }
    )


@pytest.fixture(scope="module")
def ebnerd_zip(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("ebnerd_feat") / "ebnerd_fixture.zip"
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO()
        _articles().to_parquet(buf, index=False)
        z.writestr("articles.parquet", buf.getvalue())
        for split in ("train", "validation"):
            buf = io.BytesIO()
            _behaviors().to_parquet(buf, index=False)
            z.writestr(f"{split}/behaviors.parquet", buf.getvalue())
            buf = io.BytesIO()
            _history().to_parquet(buf, index=False)
            z.writestr(f"{split}/history.parquet", buf.getvalue())
    return path


@pytest.fixture(scope="module")
def fake_embeddings(tmp_path_factory, ebnerd_zip) -> tuple[Path, Path]:
    """A deterministic stand-in embedding cache in the real on-disk format.

    Written in the same `.npy` + `.json` shape `src/retrieval/embed.py` caches
    to, with the prefixed string ids it uses, so this exercises the real
    `_load_aligned_embeddings` re-alignment path rather than bypassing it.
    Deliberately stored in a *shuffled* id order so a test would fail if the
    loader ever assumed cache order matches article-table order.
    """
    import json

    d = tmp_path_factory.mktemp("emb")
    ids = [104, 102, 101, 103]  # intentionally not parquet order
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(len(ids), 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    npy, js = d / "fake.npy", d / "fake.json"
    np.save(npy, vecs)
    js.write_text(json.dumps({"model": "fake", "article_ids": [f"ebnerd:{i}" for i in ids]}))
    return npy, js


def test_embeddings_realign_to_article_table_order(ebnerd_zip, fake_embeddings):
    """The cache is keyed by its own id order; vectors must be re-mapped, not zipped."""
    import json

    npy, js = fake_embeddings
    art = load_article_table(ebnerd_zip, embeddings_npy=npy, embeddings_json=js)
    raw = np.load(npy)
    cache_ids = [int(s.split(":")[-1]) for s in json.loads(js.read_text())["article_ids"]]
    for cache_row, article_id in enumerate(cache_ids):
        np.testing.assert_allclose(
            art.embeddings[art.id_to_pos[article_id]], raw[cache_row], rtol=1e-6
        )


@pytest.fixture(scope="module")
def built(ebnerd_zip, fake_embeddings):
    art = load_article_table(
        ebnerd_zip, embeddings_npy=fake_embeddings[0], embeddings_json=fake_embeddings[1]
    )
    prof = build_user_profiles(ebnerd_zip, "train", art)
    pop = build_history_popularity(ebnerd_zip, "train", art)
    with zipfile.ZipFile(ebnerd_zip) as zf:
        beh = pd.read_parquet(
            io.BytesIO(zf.read("train/behaviors.parquet")),
            columns=BEHAVIOR_COLUMNS + ["article_ids_clicked"],
        )
    beh = add_session_position_columns(beh)
    X, meta = build_feature_frame(beh, art, prof, pop)
    return {"art": art, "prof": prof, "pop": pop, "beh": beh, "X": X, "meta": meta}


# --------------------------------------------------------------------------
# 1. Timestamp scaling (regression: microsecond vs nanosecond)
# --------------------------------------------------------------------------


def test_epoch_seconds_converts_microsecond_datetimes_exactly():
    s = pd.Series(np.array(["2023-05-19T10:00:00"], dtype="datetime64[us]"))
    assert s.dtype == np.dtype("datetime64[us]")
    got = F._to_epoch_seconds(s)
    assert got[0] == pytest.approx(pd.Timestamp("2023-05-19 10:00:00").timestamp())


def test_epoch_seconds_is_unit_independent():
    """Same instant at three resolutions must give the same epoch seconds.

    This is the property the original `/1e9` divide violated.
    """
    instant = "2023-05-19T10:00:00"
    vals = [
        F._to_epoch_seconds(pd.Series(np.array([instant], dtype=f"datetime64[{u}]")))[0]
        for u in ("s", "ms", "us")
    ]
    assert vals[0] == vals[1] == vals[2]


def test_epoch_seconds_maps_nat_to_nan():
    s = pd.Series(pd.to_datetime([None, "2023-05-19"]))
    got = F._to_epoch_seconds(s)
    assert np.isnan(got[0]) and np.isfinite(got[1])


def test_article_age_is_positive_and_plausible(built):
    """A candidate cannot be published after the impression that shows it."""
    age = built["X"][:, FEATURE_NAMES.index("article_age_h")]
    assert np.all(age >= 0), "negative article age means the time scaling is wrong again"
    assert np.nanmax(age) < 24 * 365 * 5


def test_hours_since_last_click_is_positive(built):
    v = built["X"][:, FEATURE_NAMES.index("hours_since_last_click")]
    assert np.all(v > 0), "history must precede the impression it feeds"


# --------------------------------------------------------------------------
# 2. Leakage contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("column", sorted(FORBIDDEN_RAW_COLUMNS))
def test_reader_refuses_every_forbidden_column(ebnerd_zip, column):
    with zipfile.ZipFile(ebnerd_zip) as zf:
        with pytest.raises(ValueError, match="serving-time-unavailable"):
            F._read_zip_parquet(zf, "articles.parquet", columns=["article_id", column])


def test_behavior_columns_request_no_forbidden_field():
    assert not FORBIDDEN_RAW_COLUMNS.intersection(BEHAVIOR_COLUMNS)


def test_forbidden_fields_are_absent_from_feature_names():
    """No feature may be named after a quarantined source field."""
    for bad in FORBIDDEN_RAW_COLUMNS:
        assert not any(bad in name for name in FEATURE_NAMES)


def test_popularity_comes_from_history_not_behaviors(built):
    """Article 104 is published after the history window closes.

    It therefore has zero history-window clicks. If popularity were ever
    computed from the behaviors file (the labels), this would be non-zero.
    """
    art, pop = built["art"], built["pop"]
    assert pop[0][art.id_to_pos[104]] == 0.0
    assert pop[1][art.id_to_pos[104]] == 0.0
    assert pop[0][art.id_to_pos[101]] == 2.0  # user 1 clicked 101 twice in history


# --------------------------------------------------------------------------
# 3. Alignment, labels and determinism
# --------------------------------------------------------------------------


def test_matrix_labels_and_groups_are_row_aligned(built):
    X, meta = built["X"], built["meta"]
    assert X.shape[1] == len(FEATURE_NAMES)
    assert X.shape[0] == meta["label"].shape[0] == int(meta["group_sizes"].sum())
    assert meta["group_sizes"].tolist() == [3, 2]


def test_labels_mark_exactly_the_clicked_candidates(built):
    meta = built["meta"]
    cand, label = meta["candidate_article_id"], meta["label"]
    clicked = {int(a) for a, y in zip(cand, label) if y == 1}
    # impression 1 clicked 102, impression 2 clicked 103
    assert clicked == {102, 103}
    assert label.sum() == 2


def test_long_term_affinity_reflects_history_categories(built):
    """User 2's history is sport-only, so a sport candidate must outrank a news one."""
    X, meta = built["X"], built["meta"]
    # impression 2 (rows 3,4) belongs to user 2; candidates are 102 (sport), 103 (nyheder)
    col = FEATURE_NAMES.index("lt_cat_affinity")
    rows = {int(a): X[3 + i, col] for i, a in enumerate(meta["candidate_article_id"][3:])}
    assert rows[102] > rows[103]
    assert rows[103] == pytest.approx(0.0)


def test_short_term_and_long_term_differ_for_drifting_user(built):
    """User 1's most recent click is nyheder(101) but their history spans both.

    The decay-weighted profile must not be identical to the flat one, otherwise
    the long/short-term decomposition carries no independent information.
    """
    X = built["X"]
    lt = X[:, FEATURE_NAMES.index("lt_cat_affinity")]
    st = X[:, FEATURE_NAMES.index("st_cat_affinity")]
    assert not np.allclose(lt, st)


def test_feature_build_is_deterministic(built, ebnerd_zip, fake_embeddings):
    art = load_article_table(
        ebnerd_zip, embeddings_npy=fake_embeddings[0], embeddings_json=fake_embeddings[1]
    )
    prof = build_user_profiles(ebnerd_zip, "train", art)
    pop = build_history_popularity(ebnerd_zip, "train", art)
    X2, meta2 = build_feature_frame(built["beh"], art, prof, pop)
    np.testing.assert_array_equal(np.nan_to_num(built["X"], nan=-999), np.nan_to_num(X2, nan=-999))
    np.testing.assert_array_equal(built["meta"]["label"], meta2["label"])


def test_no_feature_is_entirely_nan(built):
    """An all-NaN column means a feature silently failed to compute."""
    nan_frac = np.isnan(built["X"]).mean(axis=0)
    dead = [FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if nan_frac[i] == 1.0]
    assert not dead, f"features produced no values at all: {dead}"


def test_unseen_category_vocab_is_respected(ebnerd_zip):
    """Forcing a training vocabulary must not silently recode categories."""
    art = load_article_table(ebnerd_zip, category_vocab=["sport"])
    pos_sport = art.id_to_pos[102]
    pos_news = art.id_to_pos[101]
    assert art.category[pos_sport] == 0
    assert art.category[pos_news] == -1  # unseen -> -1, not a shifted valid code


def test_group_sizes_match_inview_lengths(built):
    beh, meta = built["beh"], built["meta"]
    expected = [len(v) for v in beh["article_ids_inview"]]
    assert meta["group_sizes"].tolist() == expected


# --------------------------------------------------------------------------
# 4. Vocabulary persistence (train/inference code agreement)
# --------------------------------------------------------------------------


def test_vocabularies_round_trip_to_identical_codes(ebnerd_zip):
    """Reloading with a persisted vocabulary must reproduce identical codes.

    This is the property that keeps the test-set run honest: the model is
    trained on codes derived from one corpus and applied to another. If the
    second corpus re-derived its own vocabulary, "sport" and "krimi" could swap
    integers and nothing would raise — the AUC would just quietly degrade.
    """
    first = load_article_table(ebnerd_zip)
    vocab = first.vocabularies()
    second = load_article_table(
        ebnerd_zip,
        category_vocab=vocab["category"],
        subcategory_vocab=vocab["subcategory"],
        topic_vocab=vocab["topic"],
        article_type_vocab=vocab["article_type"],
    )
    np.testing.assert_array_equal(first.category, second.category)
    np.testing.assert_array_equal(first.article_type, second.article_type)
    np.testing.assert_array_equal(first.subcategory, second.subcategory)
    np.testing.assert_array_equal(first.topics, second.topics)


def test_vocabularies_are_json_serializable(ebnerd_zip):
    import json

    json.dumps(load_article_table(ebnerd_zip).vocabularies())


def test_unseen_article_type_maps_to_sentinel_not_a_shifted_code(ebnerd_zip):
    art = load_article_table(ebnerd_zip, article_type_vocab=["article_default"])
    assert art.article_type[art.id_to_pos[101]] == 0        # in vocabulary
    assert art.article_type[art.id_to_pos[103]] == -1       # video type, withheld


# --------------------------------------------------------------------------
# 5. Test-set shape tolerance
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zip_without_engagement_columns(tmp_path_factory) -> Path:
    """A bundle whose history carries no read_time/scroll columns.

    Stands in for the blind test set, whose history table is not guaranteed to
    match the training bundles column-for-column. Also nests every member under
    an extra top-level directory, exactly as `ebnerd_testset.zip` does.
    """
    path = tmp_path_factory.mktemp("ebnerd_lean") / "lean.zip"
    hist = _history()[["user_id", "article_id_fixed", "impression_time_fixed"]]
    beh = _behaviors().drop(columns=["article_ids_clicked"])
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO()
        _articles().to_parquet(buf, index=False)
        z.writestr("ebnerd_testset/articles.parquet", buf.getvalue())
        buf = io.BytesIO()
        beh.to_parquet(buf, index=False)
        z.writestr("ebnerd_testset/test/behaviors.parquet", buf.getvalue())
        buf = io.BytesIO()
        hist.to_parquet(buf, index=False)
        z.writestr("ebnerd_testset/test/history.parquet", buf.getvalue())
    return path


def test_reads_through_an_extra_wrapper_directory(zip_without_engagement_columns):
    """ebnerd_testset.zip nests members one level deeper; reads must still resolve."""
    art = load_article_table(zip_without_engagement_columns)
    assert len(art.category) == 4


def test_profiles_build_without_optional_engagement_columns(zip_without_engagement_columns):
    art = load_article_table(zip_without_engagement_columns)
    prof = build_user_profiles(zip_without_engagement_columns, "test", art)
    assert prof.history_len.tolist() == [3, 1]
    assert np.all(np.isnan(prof.mean_read_time))  # absent -> NaN, not zero
    assert np.all(np.isfinite(prof.last_click_at))  # required columns still work


def test_unlabelled_split_produces_full_width_matrix(zip_without_engagement_columns):
    """Feature width must not change between train and test.

    A narrower test matrix would raise inside LightGBM at best, and at worst
    shift every column against the trained model.
    """
    art = load_article_table(zip_without_engagement_columns)
    prof = build_user_profiles(zip_without_engagement_columns, "test", art)
    pop = build_history_popularity(zip_without_engagement_columns, "test", art)
    beh = pd.read_parquet(
        io.BytesIO(F.read_zip_member_bytes(zip_without_engagement_columns, "test/behaviors.parquet")),
        columns=BEHAVIOR_COLUMNS,
    )
    beh = add_session_position_columns(beh)
    X, meta = build_feature_frame(beh, art, prof, pop, with_labels=False)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert "label" not in meta
    assert X.shape[0] == int(meta["group_sizes"].sum())


def test_missing_required_history_column_raises_clearly(tmp_path):
    path = tmp_path / "broken.zip"
    with zipfile.ZipFile(path, "w") as z:
        buf = io.BytesIO()
        _articles().to_parquet(buf, index=False)
        z.writestr("articles.parquet", buf.getvalue())
        buf = io.BytesIO()
        _history()[["user_id"]].to_parquet(buf, index=False)
        z.writestr("test/history.parquet", buf.getvalue())
    art = load_article_table(path)
    with pytest.raises(ValueError, match="missing required column"):
        build_user_profiles(path, "test", art)


# --------------------------------------------------------------------------
# 6. Short-term profile is genuinely per-impression (the bug this fixes)
# --------------------------------------------------------------------------


def test_short_term_affinity_varies_across_a_users_own_impressions(ebnerd_zip, fake_embeddings):
    """Two impressions for the SAME user, days apart, must get different
    short-term affinity for the SAME candidate category -- if they don't,
    the decay reference is still frozen at one split-wide timestamp instead
    of tracking each impression's own time.

    Regression test for the bug documented in `UserHistoryRaw`'s docstring:
    the original implementation computed st_category once per user against a
    single reference (history-window close), so every impression that user
    had in the split got an identical short-term profile regardless of when,
    within the split, it actually occurred.
    """
    art = load_article_table(
        ebnerd_zip, embeddings_npy=fake_embeddings[0], embeddings_json=fake_embeddings[1]
    )
    prof = build_user_profiles(ebnerd_zip, "train", art)

    # User 1's most recent history click (article 101, nyheder) is on
    # 2023-05-17. Two synthetic impressions for user 1, far apart in time,
    # both offering candidate 103 (also nyheder, so it shares user 1's most
    # recent category) -- an early impression close to that click, and a late
    # one three weeks after it. Decay should make the early one's nyheder
    # affinity clearly higher.
    early = pd.Timestamp("2023-05-18 00:00:00")
    late = pd.Timestamp("2023-06-08 00:00:00")
    beh = pd.DataFrame({
        "impression_id": np.array([901, 902], dtype=np.uint32),
        "article_id": np.array([np.nan, np.nan]),
        "impression_time": np.array([early, late], dtype="datetime64[us]"),
        "read_time": np.array([1.0, 1.0], dtype=np.float32),
        "scroll_percentage": np.array([np.nan, np.nan], dtype=np.float32),
        "device_type": np.array([1, 1], dtype=np.int8),
        "article_ids_inview": [np.array([103], dtype=np.int32), np.array([103], dtype=np.int32)],
        "article_ids_clicked": [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
        "user_id": np.array([1, 1], dtype=np.uint32),
        "is_sso_user": [False, False],
        "gender": np.array([np.nan, np.nan]),
        "postcode": np.array([np.nan, np.nan]),
        "age": np.array([np.nan, np.nan]),
        "is_subscriber": [False, False],
        "session_id": np.array([21, 22], dtype=np.uint32),
        "next_read_time": np.array([0.0, 0.0], dtype=np.float32),
        "next_scroll_percentage": np.array([0.0, 0.0], dtype=np.float32),
    })
    beh = add_session_position_columns(beh)
    pop = build_history_popularity(ebnerd_zip, "train", art)
    X, meta = build_feature_frame(beh, art, prof, pop)

    st_col = FEATURE_NAMES.index("st_cat_affinity")
    early_affinity, late_affinity = X[0, st_col], X[1, st_col]
    assert early_affinity > late_affinity, (
        f"short-term affinity did not decay across time for the same user: "
        f"early={early_affinity} late={late_affinity} -- decay reference is "
        f"probably still frozen at a single split-wide timestamp"
    )
    assert not np.isclose(early_affinity, late_affinity), "identical -> reference is static per user"


def test_clicks_last_24h_is_per_impression_not_per_user(ebnerd_zip, fake_embeddings):
    """Same construction as above, isolated to the `user_clicks_last_24h` feature."""
    art = load_article_table(
        ebnerd_zip, embeddings_npy=fake_embeddings[0], embeddings_json=fake_embeddings[1]
    )
    prof = build_user_profiles(ebnerd_zip, "train", art)

    near = pd.Timestamp("2023-05-17 12:00:00")  # within 24h of user 1's last history click
    far = pd.Timestamp("2023-06-08 00:00:00")  # weeks later -> zero clicks in the trailing 24h
    beh = pd.DataFrame({
        "impression_id": np.array([911, 912], dtype=np.uint32),
        "article_id": np.array([np.nan, np.nan]),
        "impression_time": np.array([near, far], dtype="datetime64[us]"),
        "read_time": np.array([1.0, 1.0], dtype=np.float32),
        "scroll_percentage": np.array([np.nan, np.nan], dtype=np.float32),
        "device_type": np.array([1, 1], dtype=np.int8),
        "article_ids_inview": [np.array([103], dtype=np.int32), np.array([103], dtype=np.int32)],
        "article_ids_clicked": [np.array([], dtype=np.int32), np.array([], dtype=np.int32)],
        "user_id": np.array([1, 1], dtype=np.uint32),
        "is_sso_user": [False, False],
        "gender": np.array([np.nan, np.nan]),
        "postcode": np.array([np.nan, np.nan]),
        "age": np.array([np.nan, np.nan]),
        "is_subscriber": [False, False],
        "session_id": np.array([23, 24], dtype=np.uint32),
        "next_read_time": np.array([0.0, 0.0], dtype=np.float32),
        "next_scroll_percentage": np.array([0.0, 0.0], dtype=np.float32),
    })
    beh = add_session_position_columns(beh)
    pop = build_history_popularity(ebnerd_zip, "train", art)
    X, meta = build_feature_frame(beh, art, prof, pop)

    col = FEATURE_NAMES.index("user_clicks_last_24h")
    assert X[0, col] > X[1, col] >= 0


# --------------------------------------------------------------------------
# 7. Context-article match (audit finding: was missing entirely)
# --------------------------------------------------------------------------


def test_context_article_match_detects_shared_category(ebnerd_zip, fake_embeddings):
    """Impression 1's context article is 103 (nyheder); candidate 101 shares
    that category, candidate 102 (sport) does not."""
    art = load_article_table(
        ebnerd_zip, embeddings_npy=fake_embeddings[0], embeddings_json=fake_embeddings[1]
    )
    prof = build_user_profiles(ebnerd_zip, "train", art)
    pop = build_history_popularity(ebnerd_zip, "train", art)
    beh = pd.DataFrame({
        "impression_id": np.array([921], dtype=np.uint32),
        "article_id": np.array([103.0]),  # context: article 103 (nyheder)
        "impression_time": np.array([pd.Timestamp("2023-05-19 10:00:00")], dtype="datetime64[us]"),
        "read_time": np.array([5.0], dtype=np.float32),
        "scroll_percentage": np.array([np.nan], dtype=np.float32),
        "device_type": np.array([1], dtype=np.int8),
        "article_ids_inview": [np.array([101, 102], dtype=np.int32)],  # 101=nyheder, 102=sport
        "article_ids_clicked": [np.array([], dtype=np.int32)],
        "user_id": np.array([1], dtype=np.uint32),
        "is_sso_user": [False],
        "gender": np.array([np.nan]),
        "postcode": np.array([np.nan]),
        "age": np.array([np.nan]),
        "is_subscriber": [False],
        "session_id": np.array([25], dtype=np.uint32),
        "next_read_time": np.array([0.0], dtype=np.float32),
        "next_scroll_percentage": np.array([0.0], dtype=np.float32),
    })
    beh = add_session_position_columns(beh)
    X, meta = build_feature_frame(beh, art, prof, pop)

    col = FEATURE_NAMES.index("context_category_match")
    matches = {int(a): X[i, col] for i, a in enumerate(meta["candidate_article_id"])}
    assert matches[101] == 1.0
    assert matches[102] == 0.0


def test_context_article_match_is_zero_on_front_page(built):
    """Front-page impressions (context article_id null) have no immediate-read signal."""
    X, meta, beh = built["X"], built["meta"], built["beh"]
    col = FEATURE_NAMES.index("context_category_match")
    front_page_imp = beh["article_id"].isna().to_numpy()
    sizes = meta["group_sizes"]
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    for i, is_front in enumerate(front_page_imp):
        if is_front:
            sl = slice(starts[i], starts[i] + sizes[i])
            assert np.all(X[sl, col] == 0.0)


# --------------------------------------------------------------------------
# 8. Causal session features (audit finding: named in brief, never built)
# --------------------------------------------------------------------------


def test_session_position_is_1_indexed_and_causal():
    beh = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "session_id": [10, 10, 10, 20],
        "impression_id": [1, 2, 3, 4],
        "impression_time": pd.to_datetime([
            "2023-05-19 10:00:00", "2023-05-19 10:05:00", "2023-05-19 10:10:00",
            "2023-05-19 11:00:00",
        ]),
    })
    out = add_session_position_columns(beh)
    assert out["session_position"].tolist() == [1, 2, 3, 1]
    np.testing.assert_allclose(out["session_start_gap_h"].tolist(), [0.0, 5 / 60, 10 / 60, 0.0])


def test_session_position_never_uses_a_later_impression_in_the_same_session():
    """Shuffle row order; the CAUSAL position (rank by time) must not change,
    and no impression's gap can be computed from a later impression's time."""
    rng = np.random.default_rng(0)
    times = pd.to_datetime(
        ["2023-05-19 10:00:00", "2023-05-19 10:05:00", "2023-05-19 10:20:00", "2023-05-19 12:00:00"]
    )
    beh = pd.DataFrame({
        "user_id": [7, 7, 7, 7],
        "session_id": [1, 1, 1, 1],
        "impression_id": [10, 11, 12, 13],
        "impression_time": times,
    })
    shuffled = beh.sample(frac=1.0, random_state=0).reset_index(drop=True)
    out = add_session_position_columns(shuffled).sort_values("impression_id")
    assert out["session_position"].tolist() == [1, 2, 3, 4]
    gap_minutes = out["session_start_gap_h"].to_numpy() * 60
    np.testing.assert_allclose(gap_minutes, [0.0, 5.0, 20.0, 120.0], atol=1e-6)


def test_session_features_present_in_build_feature_frame_output(built):
    for name in ("session_position", "session_start_gap_h"):
        col = FEATURE_NAMES.index(name)
        assert np.isfinite(built["X"][:, col]).all()
        assert built["X"][:, col].min() >= 0


def test_build_feature_frame_requires_session_columns(ebnerd_zip, fake_embeddings):
    """Calling without add_session_position_columns must raise, not silently
    produce a feature matrix with a wrong or missing session signal."""
    art = load_article_table(
        ebnerd_zip, embeddings_npy=fake_embeddings[0], embeddings_json=fake_embeddings[1]
    )
    prof = build_user_profiles(ebnerd_zip, "train", art)
    pop = build_history_popularity(ebnerd_zip, "train", art)
    with zipfile.ZipFile(ebnerd_zip) as zf:
        beh = pd.read_parquet(
            io.BytesIO(zf.read("train/behaviors.parquet")),
            columns=BEHAVIOR_COLUMNS + ["article_ids_clicked"],
        )
    with pytest.raises(ValueError, match="session_position"):
        build_feature_frame(beh, art, prof, pop)


# --------------------------------------------------------------------------
# 9. Short-term profile must never admit a click at/after the reference time
# --------------------------------------------------------------------------


def test_short_term_never_counts_a_click_at_or_after_the_reference_time():
    """`compute_short_term_features` must be safe on its own terms.

    In production this can't happen -- `history.parquet` is verified
    (separately) to close strictly before any impression's `behaviors.parquet`
    window opens. But the function used to rely on that external invariant
    without checking it: a synthetic history containing a click AFTER the
    impression's own reference time got counted anyway, with maximal decay
    weight (as if it were the freshest possible click), because
    `np.searchsorted` was only ever used to trim the OLD end of the lookback
    window, never to exclude the future end.

    This constructs exactly that adversarial case directly against the
    function (bypassing real EB-NeRD data, which would never produce it) so
    the function's own correctness doesn't depend on the caller behaving.
    """
    raw = F.UserHistoryRaw(
        kpos=[np.array([0, 1], dtype=np.int64)],
        kts=[np.array([100.0, 500.0], dtype=np.float64)],  # second click is AFTER ref below
    )
    n_cat = 2
    prof = F.UserProfiles(
        id_to_pos={1: 0}, raw=raw,
        lt_category=np.zeros((1, n_cat), dtype=np.float32),
        lt_subcategory=np.zeros((1, 0), dtype=np.float32),
        lt_topics=np.zeros((1, 0), dtype=np.float32), lt_embedding=None,
        history_len=np.array([2]), mean_read_time=np.array([np.nan]),
        median_read_time=np.array([np.nan]), mean_scroll=np.array([np.nan]),
        last_click_at=np.array([500.0]), first_click_at=np.array([100.0]),
        last_category=np.array([0]), recent_categories=np.zeros((1, n_cat), dtype=bool),
        mean_sentiment=np.array([np.nan]), mean_article_age_at_click=np.array([np.nan]),
        distinct_categories=np.array([1]),
    )

    class _FakeArt:
        category = np.array([0, 1], dtype=np.int16)
        category_vocab = ["a", "b"]
        subcategory = np.zeros((2, 0), dtype=bool)
        topics = np.zeros((2, 0), dtype=bool)
        embeddings = None

    art = _FakeArt()
    ref = 200.0  # strictly between the two synthetic clicks (100.0, 500.0)

    out = F.compute_short_term_features(
        prof, art,
        imp_time=np.array([ref]), user_pos=np.array([0]),
        group_starts=np.array([0]), group_sizes=np.array([1]),
        cand_cat=np.array([1]),  # matches the FUTURE click's (t=500) category
        apos_safe=np.array([1]), a_known=np.array([True]),
    )
    assert out["st_cat_affinity"][0] == 0.0, (
        "a history click after the impression's own reference time was counted "
        "toward its short-term affinity"
    )

    # And the past click (t=100, category 0) must still count normally --
    # this isn't just "return zero for everything", the past stays admissible.
    out_past = F.compute_short_term_features(
        prof, art,
        imp_time=np.array([ref]), user_pos=np.array([0]),
        group_starts=np.array([0]), group_sizes=np.array([1]),
        cand_cat=np.array([0]),  # matches the PAST click's (t=100) category
        apos_safe=np.array([0]), a_known=np.array([True]),
    )
    assert out_past["st_cat_affinity"][0] > 0.0


def test_clicks_last_24h_excludes_a_click_at_or_after_the_reference_time():
    raw = F.UserHistoryRaw(
        kpos=[np.array([0], dtype=np.int64)],
        kts=[np.array([500.0], dtype=np.float64)],  # click is AFTER ref
    )
    n_cat = 1
    prof = F.UserProfiles(
        id_to_pos={1: 0}, raw=raw,
        lt_category=np.zeros((1, n_cat), dtype=np.float32),
        lt_subcategory=np.zeros((1, 0), dtype=np.float32),
        lt_topics=np.zeros((1, 0), dtype=np.float32), lt_embedding=None,
        history_len=np.array([1]), mean_read_time=np.array([np.nan]),
        median_read_time=np.array([np.nan]), mean_scroll=np.array([np.nan]),
        last_click_at=np.array([500.0]), first_click_at=np.array([500.0]),
        last_category=np.array([0]), recent_categories=np.zeros((1, n_cat), dtype=bool),
        mean_sentiment=np.array([np.nan]), mean_article_age_at_click=np.array([np.nan]),
        distinct_categories=np.array([1]),
    )

    class _FakeArt:
        category = np.array([0], dtype=np.int16)
        category_vocab = ["a"]
        subcategory = np.zeros((1, 0), dtype=bool)
        topics = np.zeros((1, 0), dtype=bool)
        embeddings = None

    art = _FakeArt()
    out = F.compute_short_term_features(
        prof, art,
        imp_time=np.array([499.9]), user_pos=np.array([0]),  # ref just BEFORE the future click
        group_starts=np.array([0]), group_sizes=np.array([1]),
        cand_cat=np.array([0]), apos_safe=np.array([0]), a_known=np.array([True]),
    )
    assert out["clicks_last_24h"][0] == 0.0
