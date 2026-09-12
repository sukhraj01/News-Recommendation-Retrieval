"""Corpus and BM25-index statistics for every locally-built corpus.

Answers the "how big is this thing" questions (vocabulary pre/post stopword
removal, total tokens, tokens/doc, index size) with numbers computed by this
project's own `tokenize`/`build_index`, not a reimplementation — so the
figures describe the index the pipeline actually uses. Anchor table:
`docs/corpus_stats.md`.

Usage (repo root):  poetry run python scripts/compute_corpus_stats.py [out.json]

Memory: serial, one corpus at a time; measured peak RSS 1.07GB (MINDlarge-test,
120,959 docs is the largest). Split-size numbers come from parquet metadata
only — no impression table is scanned.
"""
import gc
import json
import pickle
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.retrieval.index import build_index  # noqa: E402
from src.retrieval.tokenize import _TOKEN_RE, STOPWORDS, tokenize  # noqa: E402

PROCESSED = Path("data/processed")
CORPORA = {
    "mind_small_dev": PROCESSED / "mind/small/dev/articles.parquet",
    "mind_large_dev": PROCESSED / "mind/large/dev/articles.parquet",
    "mind_large_test": PROCESSED / "mind/large/test/articles.parquet",
    "ebnerd_demo": PROCESSED / "ebnerd/demo/articles.parquet",
    "ebnerd_small": PROCESSED / "ebnerd/small/articles.parquet",
}


def corpus_stats(path: Path) -> dict:
    arts = pd.read_parquet(path, columns=["article_id", "title", "abstract"])
    # Same text build_index uses; lowercased here because the raw regex split
    # (pre-stopword) must see exactly what tokenize() sees before filtering.
    text = (arts["title"].fillna("") + " " + arts["abstract"].fillna("")).str.lower()
    raw = [_TOKEN_RE.findall(t) for t in text]
    post = [tokenize(t) for t in text]
    raw_len = np.array([len(r) for r in raw])
    post_len = np.array([len(p) for p in post])
    raw_vocab = {w for r in raw for w in r}
    post_vocab = {w for p in post for w in p}
    del raw, post
    gc.collect()

    # tracemalloc sees numpy allocations too, so the retained delta is the
    # index object's real footprint (sparse arrays + vocab/id dicts).
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    t0 = time.perf_counter()
    idx = build_index(arts)
    build_s = time.perf_counter() - t0
    retained = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    w = idx.weights_t
    return {
        "docs": len(arts),
        "vocab_pre_stopword": len(raw_vocab),
        "vocab_post_stopword": len(post_vocab),
        "index_vocab": len(idx.vocab),
        "tokens_pre_stopword": int(raw_len.sum()),
        "tokens_post_stopword": int(post_len.sum()),
        "stopword_token_share": float(1 - post_len.sum() / raw_len.sum()),
        "avg_tokens_per_doc_pre": float(raw_len.mean()),
        "avg_tokens_per_doc_post": float(post_len.mean()),
        "median_tokens_per_doc_post": float(np.median(post_len)),
        "empty_docs_post": int((post_len == 0).sum()),
        "index_nnz": int(w.nnz),
        "sparse_matrix_mb": (w.data.nbytes + w.indices.nbytes + w.indptr.nbytes) / 1e6,
        "index_retained_mem_mb": retained / 1e6,
        "index_pickle_mb": len(pickle.dumps(idx, protocol=pickle.HIGHEST_PROTOCOL)) / 1e6,
        "build_s_under_tracemalloc": build_s,
    }


def main() -> None:
    out: dict = {"stopword_list_size": len(STOPWORDS)}
    for name, path in CORPORA.items():
        out[name] = corpus_stats(path)
        print(name, json.dumps(out[name]), flush=True)
        gc.collect()
    out["parquet_rows"] = {
        str(f.relative_to(PROCESSED)): pq.read_metadata(f).num_rows
        for f in sorted(PROCESSED.rglob("*.parquet"))
        if "embeddings" not in f.parts
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("corpus_stats.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
