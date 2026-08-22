"""Candidate J (ADR-012 addendum): unit coverage for `src/retrieval/nrms.py`.

The model classes (`NewsEncoder`/`UserEncoder`/`NRMSLite`) were originally
shape-verified by hand and the all-padded-row NaN-masking edge case (a
zero-history user's history rows, an empty-title article) was verified via
a numpy simulation, not a live torch run -- torch wasn't installable in the
sandbox that built the script this was extracted from. This file replaces
that with a real forward+backward pass on tiny fixture data, run against
this project's actual installed torch version, per CLAUDE.md's evidence
hierarchy (a benchmark/test collected in this project outranks a
simulation run elsewhere).
"""
import torch

from src.retrieval.nrms import (
    NRMSLite,
    NewsEncoder,
    UserEncoder,
    build_vocab,
    encode_title,
    tokenize,
)


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("Biden's Climate Plan!") == ["biden", "s", "climate", "plan"]


def test_tokenize_handles_none_and_empty():
    assert tokenize(None) == []
    assert tokenize("") == []


def test_build_vocab_includes_pad_and_unk_first():
    word2id = build_vocab(["a a b", "a b c"], min_freq=1)
    assert word2id["<pad>"] == 0
    assert word2id["<unk>"] == 1


def test_build_vocab_drops_rare_tokens_below_min_freq():
    word2id = build_vocab(["a a a b"], min_freq=2)
    assert "a" in word2id
    assert "b" not in word2id


def test_encode_title_pads_short_titles():
    word2id = build_vocab(["hello world"], min_freq=1)
    ids = encode_title("hello", word2id, max_len=4)
    assert len(ids) == 4
    assert ids[-1] == word2id["<pad>"]


def test_encode_title_truncates_long_titles():
    word2id = build_vocab(["a b c d e"], min_freq=1)
    ids = encode_title("a b c d e", word2id, max_len=3)
    assert len(ids) == 3


def test_encode_title_empty_is_all_pad():
    word2id = build_vocab(["hello"], min_freq=1)
    ids = encode_title(None, word2id, max_len=4)
    assert ids == [word2id["<pad>"]] * 4


def test_encode_title_unknown_word_maps_to_unk():
    word2id = build_vocab(["hello"], min_freq=1)
    ids = encode_title("zzzznotinvocab", word2id, max_len=2)
    assert ids[0] == word2id["<unk>"]


def test_news_encoder_output_shape():
    torch.manual_seed(0)
    encoder = NewsEncoder(vocab_size=6, pad_id=0, embed_dim=8, num_heads=2)
    title_ids = torch.tensor([[2, 3, 4, 0], [2, 0, 0, 0]])
    out = encoder(title_ids)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_news_encoder_all_pad_title_does_not_produce_nan():
    # The empty-title / history-padding-sentinel edge case: every token is
    # PAD, so every key is masked in self-attention -- must not propagate
    # NaN into the pooled vector.
    torch.manual_seed(0)
    encoder = NewsEncoder(vocab_size=6, pad_id=0, embed_dim=8, num_heads=2)
    title_ids = torch.zeros(1, 4, dtype=torch.long)
    out = encoder(title_ids)
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()


def test_user_encoder_all_masked_history_does_not_produce_nan():
    # Zero-history user: hist_pad_mask is all-True for that row.
    torch.manual_seed(0)
    encoder = UserEncoder(news_dim=8, num_heads=2)
    hist_news_vecs = torch.randn(1, 3, 8)
    hist_pad_mask = torch.ones(1, 3, dtype=torch.bool)
    out = encoder(hist_news_vecs, hist_pad_mask)
    assert out.shape == (1, 8)
    assert torch.isfinite(out).all()


def _tiny_batch(batch=2, hist_len=3, title_len=4, n_cand=3):
    torch.manual_seed(0)
    hist_title_ids = torch.randint(0, 6, (batch, hist_len, title_len))
    cand_title_ids = torch.randint(0, 6, (batch, n_cand, title_len))
    hist_pad_mask = torch.zeros(batch, hist_len, dtype=torch.bool)
    return hist_title_ids, hist_pad_mask, cand_title_ids


def test_nrms_lite_forward_output_shape():
    torch.manual_seed(0)
    model = NRMSLite(vocab_size=6, pad_id=0, embed_dim=8, num_heads=2)
    hist_title_ids, hist_pad_mask, cand_title_ids = _tiny_batch()
    scores = model(hist_title_ids, hist_pad_mask, cand_title_ids)
    assert scores.shape == (2, 3)
    assert torch.isfinite(scores).all()


def test_nrms_lite_forward_backward_runs_and_updates_gradients():
    """The real test this module needed: a tiny fixture forward+backward
    pass, matching NRMS's own K+1-way softmax cross-entropy training
    objective (positive candidate always index 0)."""
    torch.manual_seed(0)
    model = NRMSLite(vocab_size=6, pad_id=0, embed_dim=8, num_heads=2)
    hist_title_ids, hist_pad_mask, cand_title_ids = _tiny_batch(batch=2, n_cand=3)

    scores = model(hist_title_ids, hist_pad_mask, cand_title_ids)
    target = torch.zeros(2, dtype=torch.long)  # positive is always index 0
    loss = torch.nn.functional.cross_entropy(scores, target)
    assert torch.isfinite(loss)

    loss.backward()

    grad_found = False
    for name, param in model.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
        if param.grad is not None and torch.any(param.grad != 0):
            grad_found = True
    assert grad_found


def test_nrms_lite_forward_backward_with_zero_history_and_empty_title_batch_element():
    """The specific edge case flagged as numpy-simulated-only, not
    live-torch-verified, before this test existed: one batch element has a
    fully-padded (zero-history) history row AND a fully-padded (empty
    title) candidate row, mixed in the same batch as normal elements.
    Forward+backward must complete without NaN anywhere."""
    torch.manual_seed(0)
    model = NRMSLite(vocab_size=6, pad_id=0, embed_dim=8, num_heads=2)
    batch, hist_len, title_len, n_cand = 2, 3, 4, 2

    hist_title_ids = torch.randint(1, 6, (batch, hist_len, title_len))
    cand_title_ids = torch.randint(1, 6, (batch, n_cand, title_len))
    hist_pad_mask = torch.zeros(batch, hist_len, dtype=torch.bool)

    # Batch element 0: zero-history user (every history slot padded).
    hist_title_ids[0] = 0
    hist_pad_mask[0] = True
    # Batch element 1: one candidate with an empty (all-PAD) title.
    cand_title_ids[1, 0] = 0

    scores = model(hist_title_ids, hist_pad_mask, cand_title_ids)
    assert torch.isfinite(scores).all()

    target = torch.zeros(batch, dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(scores, target)
    assert torch.isfinite(loss)

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is None or torch.isfinite(param.grad).all(), f"non-finite grad in {name}"
