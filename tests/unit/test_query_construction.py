from src.retrieval.query import build_user_query
from src.retrieval.tokenize import tokenize


def test_tokenize_lowercases_and_splits_on_non_word_chars():
    assert tokenize("Biden's Climate Plan!") == ["biden", "climate", "plan"]


def test_tokenize_handles_danish_characters():
    # "er" is a Danish stopword ("is") and is filtered, per the module
    # docstring's benchmark-motivated stopword removal.
    assert tokenize("Bidens klimaplan er grøn") == ["bidens", "klimaplan", "grøn"]


def test_tokenize_removes_stopwords():
    assert tokenize("This is not a test of the plan") == ["test", "plan"]


def test_tokenize_empty_string_returns_empty_list():
    assert tokenize("") == []
    assert tokenize(None) == []


def test_query_concatenates_history_article_text():
    lookup = {"a1": "Team wins game", "a2": "Election results in"}
    tokens = build_user_query(["a1", "a2"], lookup)
    assert tokens == ["team", "wins", "game", "election", "results"]


def test_query_empty_history_returns_empty_list():
    assert build_user_query([], {"a1": "Team wins game"}) == []


def test_query_skips_ids_missing_from_lookup():
    """An article_id in history that isn't in the current corpus (e.g.
    pruned/expired) is skipped rather than raising."""
    lookup = {"a1": "Team wins game"}
    tokens = build_user_query(["a1", "unknown"], lookup)
    assert tokens == ["team", "wins", "game"]


def test_query_skips_empty_text_entries():
    lookup = {"a1": ""}
    assert build_user_query(["a1"], lookup) == []
