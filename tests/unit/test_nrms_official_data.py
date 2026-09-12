"""Unit tests: the official NRMS loaders' input semantics (ADR-015, Option B)."""
import random

import numpy as np

from src.retrieval.nrms_official_data import (
    build_mind_news_tokens,
    ebnerd_sample_negatives,
    left_pad_recent,
    mind_encode_text,
    mind_newsample,
    mind_word_tokenize,
)


def test_mind_word_tokenize_matches_official_regex():
    assert mind_word_tokenize("Queen Elizabeth, 93, visits U.S.!") == [
        "queen", "elizabeth", ",", "93", ",", "visits", "u", ".", "s", ".", "!"
    ]
    assert mind_word_tokenize(None) == []
    assert mind_word_tokenize(float("nan")) == []  # pandas NaN abstract


def test_mind_encode_text_truncates_maps_unknown_to_zero_and_right_pads():
    wd = {"a": 1, "b": 2}
    np.testing.assert_array_equal(mind_encode_text("a zz b", wd, 5), [1, 0, 2, 0, 0])
    np.testing.assert_array_equal(mind_encode_text("a b a b a b", wd, 3), [1, 2, 1])


def test_build_mind_news_tokens_row_zero_is_dummy_and_ids_start_at_one():
    nid2index, toks = build_mind_news_tokens(["N1", "N2"], ["a b", "b"], None, {"a": 1, "b": 2}, 3)
    assert nid2index == {"N1": 1, "N2": 2}
    np.testing.assert_array_equal(toks[0], [0, 0, 0])
    np.testing.assert_array_equal(toks[1], [1, 2, 0])


def test_treatment_keeps_control_title_segment_identical():
    wd = {"a": 1, "b": 2, "c": 3}
    _, ctrl = build_mind_news_tokens(["N1"], ["a b"], ["c c"], wd, title_size=3)
    _, treat = build_mind_news_tokens(["N1"], ["a b"], ["c c"], wd, title_size=3, abstract_size=4)
    np.testing.assert_array_equal(treat[:, :3], ctrl)
    np.testing.assert_array_equal(treat[1, 3:], [3, 3, 0, 0])


def test_left_pad_recent_keeps_most_recent_items():
    assert left_pad_recent([1, 2, 3, 4, 5], 3) == [3, 4, 5]
    assert left_pad_recent([7, 8], 4) == [0, 0, 7, 8]
    assert left_pad_recent([], 2) == [0, 0]


def test_mind_newsample_pads_with_zero_instead_of_resampling():
    rng = random.Random(0)
    assert mind_newsample([5, 6], 4, rng) == [5, 6, 0, 0]
    got = mind_newsample([1, 2, 3, 4, 5, 6], 4, rng)
    assert len(got) == 4 and len(set(got)) == 4 and set(got) <= {1, 2, 3, 4, 5, 6}


def test_to_epoch_seconds_is_unit_agnostic():
    """Regression for the 2026-09-12 bug: a datetime64[us] column read via
    astype(int64)/1e9 was 1000x off. Every storage unit must give the same
    seconds."""
    import pandas as pd

    from src.retrieval.nrms_official_data import to_epoch_seconds

    ref = pd.Timestamp("2023-05-18 12:00:00").timestamp()
    for unit in ("s", "ms", "us", "ns"):
        s = pd.Series(pd.to_datetime(["2023-05-18 12:00:00"])).astype(f"datetime64[{unit}]")
        np.testing.assert_allclose(to_epoch_seconds(s), [ref])
    # the exact failure mode: [us] storage, naive int cast, ns assumption
    us = pd.Series(pd.to_datetime(["2023-05-18 12:00:00"])).astype("datetime64[us]")
    assert abs(us.astype("int64").iloc[0] / 1e9 - ref) > 1e8  # the old code was this wrong


def test_ebnerd_sampling_is_with_replacement_and_drops_empty():
    rng = np.random.default_rng(0)
    got = ebnerd_sample_negatives([9], 4, rng)
    assert got == [9, 9, 9, 9]  # with replacement, never padded
    assert ebnerd_sample_negatives([], 4, rng) is None
    many = ebnerd_sample_negatives([1, 2, 3], 1000, rng)
    assert set(many) == {1, 2, 3}
