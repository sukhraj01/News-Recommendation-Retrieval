"""Unit tests for src/submission/ebnerd_format.py (Part 1's official-format
converter). Reuses the committed `ebnerd_demo_sample.zip` fixture rather
than hand-building a new one — it has two hand-verifiable impressions per
split (`train`: impressions 1/2, `user_id` 1/2, `article_ids_inview`
[101,102]/[102,103], `article_ids_clicked` [101]/[103]; `validation`:
impressions 3/4, same shape) covering both the labeled path and (via
`has_labels=False` against the same rows) the code path a real unlabeled
test split takes."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.submission.ebnerd_format import (
    ranks_for_impression,
    read_raw_impressions,
    write_predictions,
    write_truth_file,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DEMO_ZIP = FIXTURES / "ebnerd_demo_sample.zip"


class StubScorer:
    """Deterministic score per candidate_id, independent of query — enough
    to exercise write_predictions' plumbing without a real BM25/embedding
    index."""

    def __init__(self, score_by_article: dict[str, float]):
        self.score_by_article = score_by_article

    def score(self, query, candidate_ids):
        return np.array([self.score_by_article.get(c, 0.0) for c in candidate_ids])


def test_read_raw_impressions_preserves_original_order_labeled():
    rows = read_raw_impressions(DEMO_ZIP, "validation", has_labels=True)
    assert [r["raw_impression_id"] for r in rows] == ["3", "4"]

    row1 = rows[0]
    assert row1["user_id"] == "ebnerd:1"
    assert row1["article_ids"] == ["ebnerd:101", "ebnerd:102"]
    assert row1["clicked"] == [True, False]

    row2 = rows[1]
    assert row2["article_ids"] == ["ebnerd:102", "ebnerd:103"]
    assert row2["clicked"] == [False, True]


def test_read_raw_impressions_unlabeled_has_no_clicked():
    rows = read_raw_impressions(DEMO_ZIP, "validation", has_labels=False)
    assert [r["raw_impression_id"] for r in rows] == ["3", "4"]
    assert rows[0]["article_ids"] == ["ebnerd:101", "ebnerd:102"]
    assert rows[0]["clicked"] is None


def test_ranks_for_impression_best_score_gets_rank_1():
    scores = np.array([5.0, 1.0, 3.0])
    ranks = ranks_for_impression(scores, "impression-x", seed=0)
    assert ranks == [1, 3, 2]
    assert sorted(ranks) == [1, 2, 3]


def test_ranks_for_impression_deterministic_across_calls():
    scores = np.zeros(5)  # total tie -- exercises the seeded tie-break
    r1 = ranks_for_impression(scores, "same-id", seed=0)
    r2 = ranks_for_impression(scores, "same-id", seed=0)
    assert r1 == r2
    assert sorted(r1) == [1, 2, 3, 4, 5]


def test_write_predictions_format_matches_official_parser(tmp_path):
    scorer = StubScorer({"ebnerd:101": 2.0, "ebnerd:102": 1.0, "ebnerd:103": 0.5})
    out_path = tmp_path / "prediction.txt"

    n = write_predictions(
        out_path, DEMO_ZIP, "validation", scorer,
        query_by_user={}, empty_query=[], has_labels=True, seed=0,
    )
    assert n == 2

    lines = out_path.read_text().strip("\n").split("\n")
    assert len(lines) == 2

    # evaluate.py's parse_line does a bare str.split() expecting exactly 2
    # whitespace-separated tokens -- the JSON blob must contain no spaces.
    for line in lines:
        assert " " not in line.split(" ", 1)[1]
        impid, ranks_raw = line.split(" ")
        ranks = json.loads(ranks_raw)
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    impid1, ranks1 = lines[0].split(" ")
    assert impid1 == "3"
    # candidates in original order [101, 102], 101 scores higher -> rank 1
    assert json.loads(ranks1) == [1, 2]


def test_write_truth_file_labels_match_original_order(tmp_path):
    out_path = tmp_path / "truth.txt"
    n = write_truth_file(out_path, DEMO_ZIP, "validation")
    assert n == 2

    lines = out_path.read_text().strip("\n").split("\n")
    impid1, labels1 = lines[0].split(" ")
    assert impid1 == "3"
    assert json.loads(labels1) == [1, 0]  # 101 clicked, 102 not

    impid2, labels2 = lines[1].split(" ")
    assert impid2 == "4"
    assert json.loads(labels2) == [0, 1]  # 103 clicked, 102 not


def test_write_predictions_and_truth_have_matching_line_count(tmp_path):
    """The official scorer reads truth and prediction files in lockstep,
    one readline() per truth line -- line counts must match exactly."""
    scorer = StubScorer({})
    pred_path = tmp_path / "prediction.txt"
    truth_path = tmp_path / "truth.txt"

    n_pred = write_predictions(
        pred_path, DEMO_ZIP, "validation", scorer,
        query_by_user={}, empty_query=[], has_labels=True, seed=0,
    )
    n_truth = write_truth_file(truth_path, DEMO_ZIP, "validation")
    assert n_pred == n_truth
    assert len(pred_path.read_text().strip("\n").split("\n")) == \
        len(truth_path.read_text().strip("\n").split("\n"))
