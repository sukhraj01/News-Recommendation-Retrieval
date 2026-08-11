"""Unit tests for src/submission/mind_format.py (Part 3's official-format
converter). Reuses the committed MIND fixture zips (tests/fixtures/) rather
than hand-building new ones — MINDsmall_train_sample.zip covers the labeled
shape (dev/train), MINDlarge_test_sample.zip covers the unlabeled shape
(test)."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.submission.mind_format import (
    ranks_for_impression,
    read_raw_impressions,
    write_predictions,
    write_truth_file,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LABELED_ZIP = FIXTURES / "MINDsmall_train_sample.zip"
UNLABELED_ZIP = FIXTURES / "MINDlarge_test_sample.zip"


class StubScorer:
    """Deterministic score per candidate_id, independent of query — enough
    to exercise write_predictions' plumbing without a real BM25/embedding
    index."""

    def __init__(self, score_by_article: dict[str, float]):
        self.score_by_article = score_by_article

    def score(self, query, candidate_ids):
        return np.array([self.score_by_article.get(c, 0.0) for c in candidate_ids])


def test_read_raw_impressions_preserves_original_order_labeled():
    rows = read_raw_impressions(LABELED_ZIP, has_labels=True)
    assert [r["raw_impression_id"] for r in rows] == ["1", "2", "3"]

    row1 = rows[0]
    assert row1["user_id"] == "mind:U1"
    # raw tokens were "N3-1 N4-0" -- order must survive exactly, not be
    # re-sorted to N3 < N4 (it happens to already be that way here) or by
    # anything else.
    assert row1["article_ids"] == ["mind:N3", "mind:N4"]
    assert row1["clicked"] == [True, False]

    row3 = rows[2]
    # raw tokens "N4-1 N3-0" -- reversed order vs row1, must stay reversed.
    assert row3["article_ids"] == ["mind:N4", "mind:N3"]
    assert row3["clicked"] == [True, False]


def test_read_raw_impressions_unlabeled_has_no_clicked():
    rows = read_raw_impressions(UNLABELED_ZIP, has_labels=False)
    assert [r["raw_impression_id"] for r in rows] == ["1", "2"]
    assert rows[0]["article_ids"] == ["mind:N1", "mind:N2"]
    assert rows[0]["clicked"] is None


def test_ranks_for_impression_best_score_gets_rank_1():
    scores = np.array([5.0, 1.0, 3.0])
    ranks = ranks_for_impression(scores, "impression-x", seed=0)
    # position 0 (score 5.0, highest) -> rank 1; position 2 (score 3.0,
    # 2nd-highest) -> rank 2; position 1 (score 1.0, lowest) -> rank 3.
    assert ranks == [1, 3, 2]
    assert sorted(ranks) == [1, 2, 3]  # a valid permutation


def test_ranks_for_impression_deterministic_across_calls():
    scores = np.zeros(5)  # total tie -- exercises the seeded tie-break
    r1 = ranks_for_impression(scores, "same-id", seed=0)
    r2 = ranks_for_impression(scores, "same-id", seed=0)
    assert r1 == r2
    assert sorted(r1) == [1, 2, 3, 4, 5]


def test_write_predictions_format_matches_official_parser(tmp_path):
    scorer = StubScorer({"mind:N3": 2.0, "mind:N4": 1.0, "mind:N1": 0.5, "mind:N2": 0.1})
    out_path = tmp_path / "prediction.txt"

    n = write_predictions(
        out_path, LABELED_ZIP, scorer,
        query_by_user={}, empty_query=[], has_labels=True, seed=0,
    )
    assert n == 3

    lines = out_path.read_text().strip("\n").split("\n")
    assert len(lines) == 3

    # Official evaluate.py's parse_line does a bare str.split() expecting
    # exactly 2 whitespace-separated tokens -- the JSON blob must contain
    # no spaces, or that split silently breaks.
    for line in lines:
        assert " " not in line.split(" ", 1)[1]
        impid, ranks_raw = line.split(" ")
        ranks = json.loads(ranks_raw)
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    impid1, ranks1 = lines[0].split(" ")
    assert impid1 == "1"
    # candidates in original order [N3, N4], N3 scores higher -> rank 1
    assert json.loads(ranks1) == [1, 2]


def test_write_truth_file_labels_match_original_order(tmp_path):
    out_path = tmp_path / "truth.txt"
    n = write_truth_file(out_path, LABELED_ZIP)
    assert n == 3

    lines = out_path.read_text().strip("\n").split("\n")
    impid1, labels1 = lines[0].split(" ")
    assert impid1 == "1"
    assert json.loads(labels1) == [1, 0]  # N3 clicked, N4 not -- original order

    impid3, labels3 = lines[2].split(" ")
    assert impid3 == "3"
    assert json.loads(labels3) == [1, 0]  # N4 clicked, N3 not -- reversed order preserved


def test_write_predictions_and_truth_have_matching_line_count(tmp_path):
    """The official scorer reads truth and prediction files in lockstep,
    one readline() per truth line -- line counts must match exactly."""
    scorer = StubScorer({})
    pred_path = tmp_path / "prediction.txt"
    truth_path = tmp_path / "truth.txt"

    n_pred = write_predictions(
        pred_path, LABELED_ZIP, scorer,
        query_by_user={}, empty_query=[], has_labels=True, seed=0,
    )
    n_truth = write_truth_file(truth_path, LABELED_ZIP)
    assert n_pred == n_truth
    assert len(pred_path.read_text().strip("\n").split("\n")) == \
        len(truth_path.read_text().strip("\n").split("\n"))
