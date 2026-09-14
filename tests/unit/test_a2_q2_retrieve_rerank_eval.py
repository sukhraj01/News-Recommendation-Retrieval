"""Unit tests for the pure-logic pieces of the Q2 retrieve-then-rank harness
(`scripts/a2_q2_retrieve_rerank_eval.py`, ADR-015's Q2 section).

Only `systematic` and `score_one_ranking` are tested directly here -- the
rest of the script is I/O (real MIND zips/parquets, a torch checkpoint) and
is instead verified end-to-end against real data (see ADR-015's addendum:
the harness's own freshly-measured MINDsmall-dev hit rate landed inside
ADR-006's independently-measured BM25 recall@200 CI for the same corpus).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from a2_q2_retrieve_rerank_eval import score_one_ranking, systematic  # noqa: E402


def test_systematic_never_a_prefix():
    idx = systematic(1000, 10)
    assert len(idx) == 10
    assert idx[0] == 0
    assert idx[-1] > 500  # spread across the whole range, not clustered at the front


def test_systematic_returns_everything_when_k_none_or_too_big():
    assert list(systematic(50, None)) == list(range(50))
    assert list(systematic(50, 1000)) == list(range(50))


def test_score_one_ranking_click_at_top_is_perfect():
    labels = np.array([True, False, False, False], dtype=bool)
    m = score_one_ranking(labels, "imp1")
    assert m["auc"] == 1.0
    assert m["mrr"] == 1.0
    assert m["ndcg5"] == 1.0
    assert m["ndcg10"] == 1.0


def test_score_one_ranking_click_at_bottom_is_worst():
    labels = np.array([False, False, False, True], dtype=bool)
    m = score_one_ranking(labels, "imp1")
    assert m["auc"] == 0.0
    assert m["mrr"] == 0.25  # reciprocal rank of position 4


def test_score_one_ranking_no_click_is_nan_auc_and_ndcg_zero_mrr():
    """A retrieval miss (true click not in the retrieved K): AUC and nDCG are
    undefined (NaN, per `ranking_metrics.py`'s own convention, dropped by
    `ranking_metric_ci` downstream); MRR uses the standard zero-credit
    convention, which is what lets the harness's unconditional MRR honestly
    reflect the retrieval-miss ceiling rather than hiding it."""
    labels = np.array([False, False, False, False], dtype=bool)
    m = score_one_ranking(labels, "imp1")
    assert np.isnan(m["auc"])
    assert m["mrr"] == 0.0
    assert np.isnan(m["ndcg5"])
    assert np.isnan(m["ndcg10"])


def test_score_one_ranking_single_candidate_auc_is_nan_not_crash():
    labels = np.array([True], dtype=bool)
    m = score_one_ranking(labels, "imp1")
    assert np.isnan(m["auc"])  # n <= 1: no negative to compare against


def test_score_one_ranking_is_deterministic():
    labels = np.array([False, True, False, False, True], dtype=bool)
    a = score_one_ranking(labels, "same-key")
    b = score_one_ranking(labels, "same-key")
    assert a == b
