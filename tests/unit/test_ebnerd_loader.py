import json
from pathlib import Path

import pytest

from src.datasets.ebnerd import parse_ebnerd_articles, parse_ebnerd_split

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ZIP = FIXTURES / "ebnerd_demo_sample.zip"


@pytest.fixture(scope="module")
def articles():
    return parse_ebnerd_articles(ZIP)


@pytest.fixture(scope="module")
def train_result():
    return parse_ebnerd_split(ZIP, "train")


def test_article_ids_are_dataset_prefixed(articles):
    assert set(articles["article_id"]) == {"ebnerd:101", "ebnerd:102", "ebnerd:103"}


def test_articles_mapped_from_correct_source_columns(articles):
    row = articles.set_index("article_id").loc["ebnerd:101"]
    assert row["abstract"] == "Sub A"       # subtitle -> abstract
    assert row["body"] == "Body A"
    assert row["category"] == "nyheder"     # category_str -> category


def test_entities_combined_as_json_with_both_keys(articles):
    row = articles.set_index("article_id").loc["ebnerd:101"]
    payload = json.loads(row["entities"])
    assert payload == {"ner_clusters": ["Foo"], "entity_groups": ["PER"]}


def test_published_time_present_for_ebnerd(articles):
    assert articles["article_published_time"].notna().all()


def test_impression_id_is_split_qualified(train_result):
    assert set(train_result["impressions"]["impression_id"]) == {
        "ebnerd:train:1", "ebnerd:train:2"
    }


def test_user_id_is_not_split_qualified(train_result):
    assert set(train_result["impressions"]["user_id"]) == {"ebnerd:1", "ebnerd:2"}


def test_is_front_page_derived_from_null_context_article(train_result):
    """Fixture: impression 1 has context article_id=None -> front page;
    impression 2 has context article_id=103 -> not front page."""
    impressions = train_result["impressions"]
    imp1 = impressions[impressions["impression_id"] == "ebnerd:train:1"]
    imp2 = impressions[impressions["impression_id"] == "ebnerd:train:2"]
    assert imp1["is_front_page"].eq(True).all()
    assert imp2["is_front_page"].eq(False).all()


def test_clicked_derived_from_inview_and_clicked_lists(train_result):
    impressions = train_result["impressions"]
    row = impressions[
        (impressions["impression_id"] == "ebnerd:train:1")
        & (impressions["article_id"] == "ebnerd:101")
    ].iloc[0]
    assert bool(row["clicked"]) is True

    row = impressions[
        (impressions["impression_id"] == "ebnerd:train:1")
        & (impressions["article_id"] == "ebnerd:102")
    ].iloc[0]
    assert bool(row["clicked"]) is False


def test_impressions_row_count_equals_sum_of_inview_lengths(train_result):
    # fixture: impression 1 has 2 inview candidates, impression 2 has 2
    assert len(train_result["impressions"]) == 4


def test_user_history_article_ids_prefixed_and_ordered(train_result):
    row = train_result["user_history"].set_index("user_id").loc["ebnerd:1"]
    assert row["article_ids"] == ["ebnerd:101", "ebnerd:102"]


def test_user_history_read_times_present(train_result):
    row = train_result["user_history"].set_index("user_id").loc["ebnerd:1"]
    assert list(row["read_times"]) == [5.0, 6.0]
