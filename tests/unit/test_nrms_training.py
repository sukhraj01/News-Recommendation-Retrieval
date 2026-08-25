"""Candidate J (ADR-012 addendum): unit coverage for
`src/retrieval/nrms_training.py` -- the shared training/eval/GloVe-loading
helpers reused by both the Kaggle and Ada scripts."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.retrieval.nrms import PAD, NRMSLite, build_vocab
from src.retrieval.nrms_training import (
    NRMSLiteScorer,
    NRMSTrainDataset,
    build_title_matrix,
    build_training_examples,
    init_pretrained_embeddings,
    load_glove_vectors,
)


def test_build_title_matrix_shape_and_pad_sentinel_row():
    articles = pd.DataFrame({"article_id": ["a1", "a2"], "title": ["hello world", "foo"]})
    word2id = build_vocab(articles["title"], min_freq=1)
    title_matrix, news_id2row, pad_row = build_title_matrix(articles, word2id, max_title_len=4)
    assert title_matrix.shape == (3, 4)  # 2 articles + 1 sentinel row
    assert pad_row == 2
    assert news_id2row == {"a1": 0, "a2": 1}
    assert torch.equal(title_matrix[pad_row], torch.zeros(4, dtype=torch.long))


def _tiny_impressions():
    # Two impressions for user "u1": one has a click + negatives (usable),
    # one is a degenerate all-negative impression (must be skipped).
    return pd.DataFrame([
        {"user_id": "u1", "impression_id": "imp1", "article_id": "a1", "clicked": True},
        {"user_id": "u1", "impression_id": "imp1", "article_id": "a2", "clicked": False},
        {"user_id": "u1", "impression_id": "imp1", "article_id": "a3", "clicked": False},
        {"user_id": "u2", "impression_id": "imp2", "article_id": "a1", "clicked": False},
    ])


def test_build_training_examples_skips_impressions_without_both_classes():
    impressions = _tiny_impressions()
    examples = build_training_examples(
        impressions, history_by_user={}, news_row=lambda nid: {"a1": 0, "a2": 1, "a3": 2}[nid],
        neg_k=2, max_history_len=10, rng=random.Random(0),
    )
    assert len(examples) == 1  # only imp1 has a positive
    hist_rows, pos_row, neg_rows = examples[0]
    assert pos_row == 0  # a1
    assert len(neg_rows) == 2
    assert all(n in (1, 2) for n in neg_rows)  # sampled from a2/a3


def test_build_training_examples_resamples_with_replacement_when_negatives_scarce():
    impressions = _tiny_impressions()
    examples = build_training_examples(
        impressions, history_by_user={}, news_row=lambda nid: {"a1": 0, "a2": 1, "a3": 2}[nid],
        neg_k=5, max_history_len=10, rng=random.Random(0),  # only 2 real negatives available
    )
    assert len(examples) == 1
    _, _, neg_rows = examples[0]
    assert len(neg_rows) == 5


def test_build_training_examples_respects_max_examples_cap():
    rows = []
    for i in range(5):
        rows += [
            {"user_id": f"u{i}", "impression_id": f"imp{i}", "article_id": "a1", "clicked": True},
            {"user_id": f"u{i}", "impression_id": f"imp{i}", "article_id": "a2", "clicked": False},
        ]
    impressions = pd.DataFrame(rows)
    examples = build_training_examples(
        impressions, history_by_user={}, news_row=lambda nid: {"a1": 0, "a2": 1}[nid],
        neg_k=1, max_history_len=10, rng=random.Random(0), max_examples=2,
    )
    assert len(examples) == 2


def test_nrms_train_dataset_pads_history_to_fixed_length():
    examples = [([5, 6], 0, [1, 2])]
    ds = NRMSTrainDataset(examples, max_history_len=4, pad_news_row=99)
    hist_padded, hist_len, cand_rows = ds[0]
    assert hist_padded.tolist() == [5, 6, 99, 99]
    assert hist_len.item() == 2
    assert cand_rows.tolist() == [0, 1, 2]  # positive first


def test_load_glove_vectors_parses_valid_lines_and_skips_malformed():
    with_tmp = Path("/tmp/test_glove_fixture.txt")
    with_tmp.write_text("cat 1.0 2.0 3.0\nmalformed line here\ndog 4.0 5.0 6.0\n")
    try:
        vectors = load_glove_vectors(with_tmp, dim=3)
        assert set(vectors.keys()) == {"cat", "dog"}
        np.testing.assert_allclose(vectors["cat"], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(vectors["dog"], [4.0, 5.0, 6.0])
    finally:
        with_tmp.unlink()


def test_init_pretrained_embeddings_overwrites_only_found_words():
    word2id = {"<pad>": 0, "<unk>": 1, "cat": 2, "dog": 3}
    embedding = torch.nn.Embedding(4, 3, padding_idx=0)
    original_unk = embedding.weight.data[1].clone()
    original_dog = embedding.weight.data[3].clone()
    glove = {"cat": np.array([9.0, 9.0, 9.0], dtype=np.float32)}

    found = init_pretrained_embeddings(embedding, glove, word2id)

    assert found == 1
    torch.testing.assert_close(embedding.weight.data[2], torch.tensor([9.0, 9.0, 9.0]))
    torch.testing.assert_close(embedding.weight.data[1], original_unk)  # untouched
    torch.testing.assert_close(embedding.weight.data[3], original_dog)  # untouched (not in glove)


def test_init_pretrained_embeddings_stays_trainable():
    word2id = {"<pad>": 0, "cat": 1}
    embedding = torch.nn.Embedding(2, 2, padding_idx=0)
    init_pretrained_embeddings(embedding, {"cat": np.array([1.0, 1.0], dtype=np.float32)}, word2id)
    assert embedding.weight.requires_grad


def _tiny_scorer_fixture():
    articles = pd.DataFrame({
        "article_id": ["a1", "a2", "a3"],
        "title": ["hello world", "foo bar", "baz qux"],
    })
    word2id = build_vocab(articles["title"], min_freq=1)
    title_matrix, news_id2row, pad_row = build_title_matrix(articles, word2id, max_title_len=4)
    torch.manual_seed(0)
    model = NRMSLite(len(word2id), pad_id=word2id[PAD], embed_dim=8, num_heads=2)
    scorer = NRMSLiteScorer(model, news_id2row, pad_row, title_matrix, max_history_len=5, device=torch.device("cpu"))
    return scorer


def test_nrms_lite_scorer_returns_one_score_per_candidate():
    scorer = _tiny_scorer_fixture()
    scores = scorer.score(["a1"], ["a2", "a3"])
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_nrms_lite_scorer_unknown_candidate_scores_negative_infinity():
    scorer = _tiny_scorer_fixture()
    scores = scorer.score(["a1"], ["a2", "unknown_article"])
    assert np.isfinite(scores[0])
    assert scores[1] == -np.inf


def test_nrms_lite_scorer_all_unknown_candidates_returns_all_negative_infinity():
    scorer = _tiny_scorer_fixture()
    scores = scorer.score(["a1"], ["unknown1", "unknown2"])
    assert scores.tolist() == [-np.inf, -np.inf]


def test_nrms_lite_scorer_empty_history_still_scores_known_candidates():
    scorer = _tiny_scorer_fixture()
    scores = scorer.score([], ["a1", "a2"])
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()


def test_nrms_lite_scorer_history_longer_than_max_gets_truncated_not_erroring():
    scorer = _tiny_scorer_fixture()  # max_history_len=5
    scores = scorer.score(["a1", "a2", "a3", "a1", "a2", "a3", "a1"], ["a1"])
    assert scores.shape == (1,)
    assert np.isfinite(scores).all()
