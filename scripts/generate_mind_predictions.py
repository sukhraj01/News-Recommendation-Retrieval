#!/usr/bin/env python3
"""Generate official-format MIND Codabench predictions (Part 3/4) for a
labeled split (dev, has real labels — local validation) via
`src.submission.mind_format`. `--method {bm25,embed}` reuses the exact same
`Scorer`/index construction `scripts/run_ranking_eval.py` uses, so a
prediction file generated here and that script's own metrics come from the
identical scoring path.

Usage:
    poetry run python scripts/generate_mind_predictions.py --split dev --method bm25
    poetry run python scripts/generate_mind_predictions.py --split dev --method embed
"""
import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.retrieval.embed import DEFAULT_MODEL, build_embedding_index, build_user_embedding_query, model_slug
from src.retrieval.index import build_index
from src.retrieval.query import build_user_query
from src.retrieval.score import BM25Scorer, EmbeddingScorer
from src.submission.mind_format import write_predictions, write_truth_file
from src.utils.config import MIND_RAW_DIR, PROCESSED_DIR

ZIP_NAME = {"train": "MINDlarge_train", "dev": "MINDlarge_dev", "test": "MINDlarge_test"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument("--method", choices=["bm25", "embed"], required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    has_labels = args.split != "test"
    base = PROCESSED_DIR / "mind" / "large" / args.split
    zip_path = MIND_RAW_DIR / f"{ZIP_NAME[args.split]}.zip"

    articles = pd.read_parquet(base / "articles.parquet")
    history = pd.read_parquet(base / "user_history.parquet")
    print(f"corpus: {len(articles)} articles, {len(history)} users with history", flush=True)

    t0 = time.time()
    if args.method == "bm25":
        index = build_index(articles)
        scorer = BM25Scorer(index)
        text_lookup = dict(zip(
            articles["article_id"],
            articles["title"].fillna("") + " " + articles["abstract"].fillna(""),
        ))
        query_by_user = {
            row.user_id: build_user_query(row.article_ids, text_lookup)
            for row in history.itertuples(index=False)
        }
        empty_query = []
    else:
        cache_path = base / "embeddings" / f"{model_slug(DEFAULT_MODEL)}.npy"
        index = build_embedding_index(articles, model_name=DEFAULT_MODEL, cache_path=cache_path)
        scorer = EmbeddingScorer(index)
        vector_lookup = dict(zip(index.article_ids, index.vectors))
        query_by_user = {
            row.user_id: build_user_embedding_query(row.article_ids, vector_lookup)
            for row in history.itertuples(index=False)
        }
        empty_query = None
    print(f"index + query build: {time.time() - t0:.1f}s", flush=True)

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "submissions" / f"mind_large_{args.split}_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n = write_predictions(
        out_dir / "prediction.txt", zip_path, scorer, query_by_user, empty_query,
        has_labels=has_labels, seed=0,
    )
    print(f"wrote {n} prediction lines in {time.time() - t0:.1f}s -> {out_dir / 'prediction.txt'}", flush=True)

    if has_labels:
        t0 = time.time()
        n_truth = write_truth_file(out_dir / "truth.txt", zip_path)
        print(f"wrote {n_truth} truth lines in {time.time() - t0:.1f}s -> {out_dir / 'truth.txt'}", flush=True)
        assert n == n_truth, f"line count mismatch: {n} predictions vs {n_truth} truth"


if __name__ == "__main__":
    main()
