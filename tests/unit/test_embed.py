import numpy as np
import pandas as pd

from src.retrieval.embed import (
    EmbeddingIndex,
    _default_device,
    build_embedding_index,
    build_user_embedding_query,
    build_user_embedding_query_recency,
)


class _StubEncoder:
    """Deterministic, network-free stand-in for a real SentenceTransformer:
    maps each text to a fixed 3-dim vector by a simple hash, L2-normalized
    (matching real encoder output shape/contract) — lets embed.py's own
    logic (caching, mean-pooling) be tested without downloading a model."""

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(self, texts, batch_size, show_progress_bar, convert_to_numpy, normalize_embeddings):
        vecs = []
        for t in texts:
            h = abs(hash(t))
            v = np.array([(h % 7) - 3, (h // 7 % 7) - 3, (h // 49 % 7) - 3], dtype=np.float32)
            if np.linalg.norm(v) == 0:
                v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            vecs.append(v / np.linalg.norm(v))
        return np.array(vecs, dtype=np.float32)


def _articles():
    return pd.DataFrame({
        "article_id": ["a1", "a2", "a3"],
        "title": ["football match today", "election results in", "cooking recipe today"],
        "abstract": ["team wins big game", "president wins vote", "delicious pasta dish"],
    })


def test_build_embedding_index_shapes_and_normalization():
    index = build_embedding_index(_articles(), encoder=_StubEncoder())
    assert index.article_ids == ["a1", "a2", "a3"]
    assert index.id_to_col == {"a1": 0, "a2": 1, "a3": 2}
    assert index.vectors.shape == (3, 3)
    norms = np.linalg.norm(index.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_build_user_embedding_query_mean_pools_and_renormalizes():
    vector_lookup = {
        "a1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "a2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    query = build_user_embedding_query(["a1", "a2"], vector_lookup)
    assert query is not None
    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    np.testing.assert_allclose(query, expected, atol=1e-6)
    assert np.isclose(np.linalg.norm(query), 1.0)


def test_build_user_embedding_query_single_article_returns_its_own_direction():
    vector_lookup = {"a1": np.array([0.6, 0.8, 0.0], dtype=np.float32)}
    query = build_user_embedding_query(["a1"], vector_lookup)
    np.testing.assert_allclose(query, [0.6, 0.8, 0.0], atol=1e-6)


def test_build_user_embedding_query_empty_history_returns_none():
    assert build_user_embedding_query([], {"a1": np.array([1.0, 0.0])}) is None


def test_build_user_embedding_query_unresolvable_history_returns_none():
    # Every article_id absent from vector_lookup - a true cold-start user
    # whose history doesn't overlap the index (shouldn't normally happen,
    # but must degrade safely, not raise or divide by zero).
    assert build_user_embedding_query(["unknown"], {"a1": np.array([1.0, 0.0])}) is None


def test_build_user_embedding_query_skips_unknown_ids_but_uses_known_ones():
    vector_lookup = {"a1": np.array([1.0, 0.0], dtype=np.float32)}
    query = build_user_embedding_query(["unknown", "a1"], vector_lookup)
    np.testing.assert_allclose(query, [1.0, 0.0], atol=1e-6)


def test_build_embedding_index_cache_round_trip(tmp_path):
    cache_path = tmp_path / "embeddings" / "stub.npy"
    first = build_embedding_index(
        _articles(), model_name="stub-model", cache_path=cache_path, encoder=_StubEncoder()
    )
    assert cache_path.exists()
    assert cache_path.with_suffix(".json").exists()

    # Second call must hit the cache, not re-invoke the encoder: pass an
    # encoder that raises if called, to prove the cache path is taken.
    class _RaisingEncoder:
        def encode(self, *a, **k):
            raise AssertionError("cache should have been used, encoder should not be called")

    second = build_embedding_index(
        _articles(), model_name="stub-model", cache_path=cache_path, encoder=_RaisingEncoder()
    )
    np.testing.assert_array_equal(second.vectors, first.vectors)
    assert second.article_ids == first.article_ids


def test_build_embedding_index_cache_invalidated_by_different_model(tmp_path):
    cache_path = tmp_path / "embeddings" / "stub.npy"
    build_embedding_index(
        _articles(), model_name="model-a", cache_path=cache_path, encoder=_StubEncoder()
    )

    calls = {"n": 0}

    class _CountingEncoder(_StubEncoder):
        def encode(self, *a, **k):
            calls["n"] += 1
            return super().encode(*a, **k)

    build_embedding_index(
        _articles(), model_name="model-b", cache_path=cache_path, encoder=_CountingEncoder()
    )
    assert calls["n"] == 1  # different model name -> cache miss -> real recompute


def test_build_embedding_index_cache_invalidated_by_different_article_set(tmp_path):
    cache_path = tmp_path / "embeddings" / "stub.npy"
    build_embedding_index(
        _articles(), model_name="model-a", cache_path=cache_path, encoder=_StubEncoder()
    )

    different_articles = _articles().iloc[:2].copy()  # a3 dropped
    calls = {"n": 0}

    class _CountingEncoder(_StubEncoder):
        def encode(self, *a, **k):
            calls["n"] += 1
            return super().encode(*a, **k)

    result = build_embedding_index(
        different_articles, model_name="model-a", cache_path=cache_path, encoder=_CountingEncoder()
    )
    assert calls["n"] == 1
    assert result.article_ids == ["a1", "a2"]


def test_build_user_embedding_query_recency_weights_most_recent_entry_highest():
    # a1 = oldest, a3 = most recent (last element). With decay < 1, the
    # result should lean toward a3's direction more than an unweighted
    # mean of the same three orthogonal-ish vectors would.
    vector_lookup = {
        "a1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "a2": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "a3": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    query = build_user_embedding_query_recency(["a1", "a2", "a3"], vector_lookup, decay=0.5)
    assert query is not None
    assert query[2] > query[1] > query[0]


def test_build_user_embedding_query_recency_decay_one_matches_unweighted_mean():
    vector_lookup = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a2": np.array([0.0, 1.0], dtype=np.float32),
    }
    recency_query = build_user_embedding_query_recency(["a1", "a2"], vector_lookup, decay=1.0)
    plain_query = build_user_embedding_query(["a1", "a2"], vector_lookup)
    np.testing.assert_allclose(recency_query, plain_query, atol=1e-6)


def test_build_user_embedding_query_recency_empty_history_returns_none():
    assert build_user_embedding_query_recency([], {"a1": np.array([1.0, 0.0])}) is None


def test_build_user_embedding_query_recency_unresolvable_history_returns_none():
    assert build_user_embedding_query_recency(["unknown"], {"a1": np.array([1.0, 0.0])}) is None


def test_build_user_embedding_query_recency_weighting_skips_unresolvable_ids_by_position():
    # Weights are computed over the FILTERED (resolvable) sequence, so an
    # unresolvable id in the middle doesn't shift surrounding weights —
    # this should behave identically to the same history with the unknown
    # id simply absent.
    vector_lookup = {
        "a1": np.array([1.0, 0.0], dtype=np.float32),
        "a3": np.array([0.0, 1.0], dtype=np.float32),
    }
    with_gap = build_user_embedding_query_recency(["a1", "unknown", "a3"], vector_lookup, decay=0.5)
    without_gap = build_user_embedding_query_recency(["a1", "a3"], vector_lookup, decay=0.5)
    np.testing.assert_allclose(with_gap, without_gap, atol=1e-6)


def test_build_user_embedding_query_recency_single_article_returns_its_own_direction():
    vector_lookup = {"a1": np.array([0.6, 0.8, 0.0], dtype=np.float32)}
    query = build_user_embedding_query_recency(["a1"], vector_lookup, decay=0.5)
    np.testing.assert_allclose(query, [0.6, 0.8, 0.0], atol=1e-6)


def test_default_device_prefers_cuda_when_available(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _default_device() == "cuda"


def test_default_device_falls_back_to_mps_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _default_device() == "mps"


def test_default_device_falls_back_to_cpu_without_cuda_or_mps(monkeypatch):
    # This is the case that broke on Kaggle's Linux runners (2026-08-21):
    # no CUDA visible to this process, and "mps" doesn't exist at all
    # off-Mac — must land on "cpu", the universal fallback, not raise.
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert _default_device() == "cpu"
