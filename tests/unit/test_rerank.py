"""Candidate I (2026-08-22 session): unit coverage for
`src/retrieval/rerank.py` — `build_user_history_vectors` (the raw,
unpooled history-sequence builder no other function in this project
provides), `AttentionScorer` (shape/cold-start/learnability), and
`AttentionRerankScorer` (the `Scorer`-protocol eval wrapper). The training
loop itself (`scripts/run_attention_reranker_experiment.py`) is exercised
via its own experiment run, not unit-tested end-to-end here — same split
this project already draws between unit-tested helper functions and
`experiments/` output for F/G/H."""
import numpy as np
import pytest
import torch

from src.retrieval.rerank import (
    AttentionRerankScorer,
    AttentionScorer,
    build_user_history_vectors,
)


def test_build_user_history_vectors_returns_sequence_in_order():
    lookup = {"a1": np.array([1.0, 0.0]), "a2": np.array([0.0, 1.0])}
    out = build_user_history_vectors(["a1", "a2"], lookup)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [1.0, 0.0])
    np.testing.assert_allclose(out[1], [0.0, 1.0])


def test_build_user_history_vectors_skips_missing_ids():
    lookup = {"a1": np.array([1.0, 0.0])}
    out = build_user_history_vectors(["a1", "missing", "a1"], lookup)
    assert out.shape == (2, 2)


def test_build_user_history_vectors_empty_history_returns_empty_array():
    out = build_user_history_vectors([], {"a1": np.array([1.0, 0.0])})
    assert out.shape == (0, 0)


def test_build_user_history_vectors_fully_unresolvable_returns_empty_array():
    out = build_user_history_vectors(["missing1", "missing2"], {"a1": np.array([1.0, 0.0])})
    assert out.shape == (0, 0)


def test_attention_scorer_output_shape():
    torch.manual_seed(0)
    model = AttentionScorer(dim=8, d_attn=4)
    candidates = torch.randn(5, 8)
    history = torch.randn(3, 8)
    out = model(candidates, history)
    assert out.shape == (5,)


def test_attention_scorer_zero_history_returns_zero_scores():
    torch.manual_seed(0)
    model = AttentionScorer(dim=8, d_attn=4)
    candidates = torch.randn(5, 8)
    history = torch.empty(0, 8)
    out = model(candidates, history)
    assert out.shape == (5,)
    assert torch.allclose(out, torch.zeros(5))


def test_attention_scorer_single_history_item_ignores_attention_weighting():
    # With exactly one history item, softmax over a single key is always 1.0
    # regardless of the learned projections, so pooled == that one history
    # vector exactly - a useful sanity check on the weighted-sum mechanics.
    torch.manual_seed(0)
    model = AttentionScorer(dim=4, d_attn=4)
    candidates = torch.randn(3, 4)
    history = torch.randn(1, 4)
    out = model(candidates, history)
    expected = candidates @ history[0]
    assert torch.allclose(out, expected, atol=1e-5)


def test_attention_scorer_zero_history_with_popularity_gives_nonzero_learnable_score():
    # ADR-011 addendum: a cold-start user should still get a real,
    # popularity-driven score (not a flat tie) once pop_weight is nonzero.
    torch.manual_seed(0)
    model = AttentionScorer(dim=4, d_attn=2)
    model.pop_weight.data.fill_(2.0)
    candidates = torch.randn(3, 4)
    history = torch.empty(0, 4)
    log_pop = torch.tensor([1.0, -1.0, 0.5])
    out = model(candidates, history, log_popularity=log_pop)
    assert torch.allclose(out, 2.0 * log_pop)


def test_attention_scorer_popularity_defaults_to_no_effect_when_omitted():
    torch.manual_seed(0)
    model = AttentionScorer(dim=4, d_attn=2)
    model.pop_weight.data.fill_(5.0)  # even a nonzero weight has no effect if not passed
    candidates = torch.randn(3, 4)
    history = torch.randn(2, 4)
    with_pop_omitted = model(candidates, history)
    with_pop_none = model(candidates, history, log_popularity=None)
    assert torch.allclose(with_pop_omitted, with_pop_none)


def test_attention_scorer_gradient_flows_into_pop_weight():
    torch.manual_seed(0)
    model = AttentionScorer(dim=4, d_attn=2)
    candidates = torch.randn(3, 4)
    history = torch.empty(0, 4)
    log_pop = torch.tensor([1.0, -1.0, 0.5])
    labels = torch.tensor([1.0, 0.0, 0.0])

    logits = model(candidates, history, log_popularity=log_pop)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    assert model.pop_weight.grad is not None
    assert model.pop_weight.grad.item() != 0


def test_attention_scorer_gradients_flow_into_projections():
    torch.manual_seed(0)
    model = AttentionScorer(dim=8, d_attn=4)
    candidates = torch.randn(6, 8)
    history = torch.randn(4, 8)
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    logits = model(candidates, history)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()

    assert model.wq.weight.grad is not None
    assert model.wk.weight.grad is not None
    assert torch.any(model.wq.weight.grad != 0)
    assert torch.any(model.wk.weight.grad != 0)


def test_attention_rerank_scorer_unknown_candidate_scores_negative_infinity():
    torch.manual_seed(0)
    model = AttentionScorer(dim=4, d_attn=2)
    lookup = {"a1": np.random.default_rng(0).normal(size=4).astype(np.float32)}
    scorer = AttentionRerankScorer(model, lookup)

    history = build_user_history_vectors(["a1"], lookup)
    scores = scorer.score(history, ["a1", "unknown"])
    assert scores.shape == (2,)
    assert np.isfinite(scores[0])
    assert scores[1] == -np.inf


def test_attention_rerank_scorer_empty_history_gives_zero_score_for_known_candidates():
    torch.manual_seed(0)
    model = AttentionScorer(dim=4, d_attn=2)
    lookup = {"a1": np.random.default_rng(0).normal(size=4).astype(np.float32)}
    scorer = AttentionRerankScorer(model, lookup)

    empty_history = build_user_history_vectors([], lookup)
    scores = scorer.score(empty_history, ["a1"])
    assert scores == pytest.approx([0.0])
