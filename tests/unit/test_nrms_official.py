"""Unit tests for the official-config NRMS port (ADR-015, Option B).

The layer tests check the port against a direct numpy transcription of the
official Keras formulas (recommenders `layers.py`), not against the torch
code's own logic, so a transcription error in either cannot hide.
"""
import math

import numpy as np
import pytest
import torch

from src.retrieval.nrms_official import (
    KERAS_EPSILON,
    AttLayer2,
    KerasSelfAttention,
    OfficialNRMS,
    freshness_feature,
    official_loss,
)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().astype(np.float64)


def test_self_attention_matches_keras_formula():
    torch.manual_seed(0)
    layer = KerasSelfAttention(in_dim=6, head_num=2, head_dim=3)
    x = torch.randn(2, 4, 6)
    out = _np(layer(x))

    xs, WQ, WK, WV = _np(x), _np(layer.WQ), _np(layer.WK), _np(layer.WV)

    def heads(m):  # (b, n, h*d) -> (b, h, n, d)
        return m.reshape(2, 4, 2, 3).transpose(0, 2, 1, 3)

    Q, K, V = heads(xs @ WQ), heads(xs @ WK), heads(xs @ WV)
    A = np.einsum("abij,abkj->abik", Q, K) / math.sqrt(3)
    A = np.exp(A - A.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    expected = np.einsum("abij,abjk->abik", A, V).transpose(0, 2, 1, 3).reshape(2, 4, 6)
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-6)


def test_self_attention_has_no_bias_and_no_output_projection():
    layer = KerasSelfAttention(in_dim=300, head_num=20, head_dim=20)
    names = {n for n, _ in layer.named_parameters()}
    assert names == {"WQ", "WK", "WV"}
    assert layer(torch.randn(1, 5, 300)).shape == (1, 5, 400)  # 20 x 20, not 300


def test_attlayer2_matches_keras_formula_including_epsilon():
    torch.manual_seed(1)
    layer = AttLayer2(in_dim=5, hidden=4)
    with torch.no_grad():
        layer.b.uniform_(-0.5, 0.5)  # exercise the bias path (initialised to zeros)
    x = torch.randn(3, 7, 5)
    out = _np(layer(x))
    xs, W, b, q = _np(x), _np(layer.W), _np(layer.b), _np(layer.q)
    att = np.exp((np.tanh(xs @ W + b) @ q).squeeze(-1))
    w = att / (att.sum(-1, keepdims=True) + KERAS_EPSILON)
    np.testing.assert_allclose(out, (w[..., None] * xs).sum(1), rtol=1e-5, atol=1e-6)


def test_glorot_uniform_limits_respected():
    layer = KerasSelfAttention(in_dim=300, head_num=20, head_dim=20)
    limit = math.sqrt(6.0 / (300 + 400))
    for p in (layer.WQ, layer.WK, layer.WV):
        assert p.abs().max().item() <= limit + 1e-7


def _model(use_freshness=False, seed=0):
    torch.manual_seed(seed)
    emb = np.random.default_rng(seed).normal(size=(50, 8)).astype(np.float32)
    emb[0] = 0.0  # official padding row is a zero vector
    return OfficialNRMS(emb, head_num=2, head_dim=4, attention_hidden_dim=6,
                        dropout=0.2, use_freshness=use_freshness)


def test_forward_shapes_and_official_widths():
    m = _model().eval()
    hist = torch.randint(0, 50, (3, 5, 7))
    cand = torch.randint(0, 50, (3, 4, 7))
    assert m(hist, cand).shape == (3, 4)
    assert m.news_encoder.out_dim == 8  # head_num * head_dim


def test_all_padding_history_is_finite_without_masks():
    """The official graph has no masks: an all-pad history must still give
    finite scores (the zero-history / cold-user case)."""
    m = _model().eval()
    hist = torch.zeros(2, 5, 7, dtype=torch.long)
    cand = torch.randint(1, 50, (2, 3, 7))
    assert torch.isfinite(m(hist, cand)).all()


def test_freshness_treatment_starts_identical_to_control():
    """With w initialised to 0, the treatment and the control compute the same
    function at init when built from the same seed: the ablation isolates what
    training does with the freshness input, nothing else."""
    ctrl, treat = _model(False, seed=3).eval(), _model(True, seed=3).eval()
    hist = torch.randint(0, 50, (2, 5, 7))
    cand = torch.randint(0, 50, (2, 4, 7))
    age = torch.rand(2, 4) * 5
    torch.testing.assert_close(ctrl(hist, cand), treat(hist, cand, age))


def test_freshness_requires_age_input():
    m = _model(True).eval()
    with pytest.raises(ValueError):
        m(torch.zeros(1, 2, 3, dtype=torch.long), torch.zeros(1, 2, 3, dtype=torch.long))


def test_freshness_weight_receives_gradient():
    m = _model(True).train()
    loss = official_loss(m(torch.randint(0, 50, (4, 5, 7)), torch.randint(0, 50, (4, 5, 7)),
                            torch.rand(4, 5)))
    loss.backward()
    assert m.freshness_w.grad is not None and m.freshness_w.grad.abs().item() > 0


def test_embeddings_are_trainable_like_official():
    m = _model()
    assert m.news_encoder.embed.weight.requires_grad


def test_training_step_reduces_loss_on_separable_toy_data():
    torch.manual_seed(0)
    m = _model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    hist = torch.randint(1, 50, (16, 5, 7))
    cand = torch.randint(1, 50, (16, 5, 7))
    cand[:, 0] = hist[:, 0]  # positive shares its tokens with the history
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = official_loss(m(hist, cand))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5


def test_deterministic_given_seed():
    hist = torch.randint(0, 50, (2, 5, 7))
    cand = torch.randint(0, 50, (2, 4, 7))
    a = _model(seed=7).eval()(hist, cand)
    b = _model(seed=7).eval()(hist, cand)
    torch.testing.assert_close(a, b)


def test_freshness_feature_log1p_hours_clipped_and_nan_safe():
    imp = np.array([7200.0, 0.0, 3600.0, 100.0])
    pub = np.array([0.0, 3600.0, np.nan, 100.0])  # 2h, negative, missing, 0h
    got = freshness_feature(imp, pub)
    np.testing.assert_allclose(got, [np.log1p(2.0), 0.0, 0.0, 0.0], rtol=1e-6)
    assert got.dtype == np.float32
