from pathlib import Path

import pytest

from src.datasets.mind import parse_mind_split, parse_mind_test_candidates

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture(scope="module")
def train_result():
    return parse_mind_split(FIXTURES / "MINDsmall_train_sample.zip", "train")


@pytest.fixture(scope="module")
def test_result():
    return parse_mind_test_candidates(FIXTURES / "MINDlarge_test_sample.zip", "test")


def test_article_ids_are_dataset_prefixed(train_result):
    assert set(train_result["articles"]["article_id"]) == {
        "mind:N1", "mind:N2", "mind:N3", "mind:N4"
    }


def test_empty_abstract_coerced_to_empty_string(train_result):
    row = train_result["articles"].set_index("article_id").loc["mind:N2"]
    assert row["abstract"] == ""
    assert row["abstract"] is not None


def test_entities_combined_as_json_with_both_keys(train_result):
    import json
    row = train_result["articles"].set_index("article_id").loc["mind:N3"]
    payload = json.loads(row["entities"])
    assert set(payload.keys()) == {"title_entities", "abstract_entities"}
    assert payload["title_entities"][0]["WikidataId"] == "Q1"
    assert payload["abstract_entities"] == []


def test_dataset_only_fields_null_for_mind(train_result):
    articles = train_result["articles"]
    assert articles["body"].isna().all()
    assert articles["topics"].isna().all()
    assert articles["article_published_time"].isna().all()


def test_impression_id_is_split_qualified(train_result):
    assert set(train_result["impressions"]["impression_id"]) == {
        "mind:train:1", "mind:train:2", "mind:train:3"
    }


def test_user_id_is_not_split_qualified(train_result):
    """user_id must be a shared identity across splits (real users appear in
    both train and dev), unlike impression_id which genuinely restarts per file."""
    assert set(train_result["impressions"]["user_id"]) == {"mind:U1", "mind:U2"}


def test_clicked_label_parsed_correctly(train_result):
    impressions = train_result["impressions"]
    row = impressions[
        (impressions["impression_id"] == "mind:train:1")
        & (impressions["article_id"] == "mind:N3")
    ].iloc[0]
    assert bool(row["clicked"]) is True

    row = impressions[
        (impressions["impression_id"] == "mind:train:1")
        & (impressions["article_id"] == "mind:N4")
    ].iloc[0]
    assert bool(row["clicked"]) is False


def test_empty_history_string_becomes_empty_list(train_result):
    row = train_result["user_history"].set_index("user_id").loc["mind:U2"]
    assert row["article_ids"] == []


def test_static_history_deduplicated_to_one_row_per_user(train_result):
    """U1 appears in two impression rows (impression 1 and 3) with the same
    history — user_history must have exactly one row for U1, not two."""
    history = train_result["user_history"]
    assert (history["user_id"] == "mind:U1").sum() == 1
    row = history.set_index("user_id").loc["mind:U1"]
    assert row["article_ids"] == ["mind:N1", "mind:N2"]


def test_static_history_invariant_violation_raises():
    import pandas as pd
    from src.datasets.mind import _build_user_history

    behaviors = pd.DataFrame({
        "user_id": ["U1", "U1"],
        "history": ["N1 N2", "N1 N3"],  # varies for the same user -> should raise
    })
    with pytest.raises(ValueError, match="static-history invariant violated"):
        _build_user_history(behaviors)


def test_candidates_table_has_no_clicked_column(test_result):
    assert "clicked" not in test_result["candidates"].columns


def test_candidates_never_produces_an_impressions_table(test_result):
    assert set(test_result.keys()) == {"articles", "candidates", "user_history"}


def test_candidates_still_returns_user_history(test_result):
    """Unlike `clicked`, user_history doesn't depend on labels — the blind
    test split still needs it for query construction."""
    assert len(test_result["user_history"]) > 0
    assert "article_ids" in test_result["user_history"].columns
