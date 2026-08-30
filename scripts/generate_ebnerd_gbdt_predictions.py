#!/usr/bin/env python3
"""Score `ebnerd_testset` with a trained Candidate K model and write the
official Codabench submission (ADR-013).

Why this is not `ebnerd_format.write_predictions`
-------------------------------------------------
That function is built around the `Scorer` abstraction: one user-level query
representation, scored against one impression's candidate list at a time. A
GBDT needs a full 60-column feature row per candidate, and calling
`booster.predict` once per impression across 13.5M impressions would dominate
the runtime. This script batches instead — but reuses
`ebnerd_format.ranks_for_impression` unchanged so the official rank encoding
and ADR-007's deterministic tie-break are not reimplemented (and cannot drift).

Scale, and why it is chunked
----------------------------
`ebnerd_testset` holds 13,536,710 impressions. At `ebnerd_small`'s measured
~11.9 candidates per impression that is roughly 160M candidate rows, and at the
measured 240 bytes/row the full feature matrix would be **~37 GB** — before the
behaviors frame itself, which a previous session measured at ~8.5 GB with a
~16 GB materialization spike. Building it in one piece is not viable even on a
120 GB node, so impressions are streamed in chunks and each chunk's matrix is
discarded after scoring. Peak memory is then the behaviors frame plus one
chunk, not the whole split.

Vocabulary agreement
--------------------
The test corpus is `articles_large_only.zip`, a *different* article table from
the one the model trained on. Its categories/subcategories/topics/article-types
are therefore re-derived onto the **training** vocabularies persisted in
`vocabularies.json`. Skipping this would renumber the categorical codes and
degrade the model silently, with nothing raised.

Usage:
    python scripts/generate_ebnerd_gbdt_predictions.py \
        --model-dir ~/ebnerd_gbdt_k_results/candidate_k_gbdt_ebnerd_large \
        --arm K_rank \
        --testset-zip /ssd_scratch/$USER/ebnerd_raw/ebnerd_testset.zip \
        --articles-zip /ssd_scratch/$USER/ebnerd_raw/articles_large_only.zip \
        --embeddings-npy .../MiniLM.npy --embeddings-json .../MiniLM.json \
        --out submissions/ebnerd_testset_gbdt_k
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.retrieval.ebnerd_features import (
    BEHAVIOR_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    build_feature_frame,
    build_history_popularity,
    build_user_profiles,
    load_article_table,
    load_test_behaviors,
)
from src.submission.ebnerd_format import ranks_for_impression
from src.utils.io import read_zip_member_bytes

SEED = 0




def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--arm", default="K_rank")
    ap.add_argument("--testset-zip", required=True)
    ap.add_argument("--articles-zip", required=True)
    ap.add_argument("--embeddings-npy", required=True)
    ap.add_argument("--embeddings-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--chunk-size", type=int, default=200_000,
                    help="Impressions per feature-build batch. 200k ~ 2.4M rows ~ 0.6 GB.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    booster = lgb.Booster(model_file=str(model_dir / f"model_{args.arm}.txt"))
    config = json.loads((model_dir / "config.json").read_text())
    vocab = json.loads((model_dir / "vocabularies.json").read_text())
    print(f"[1/5] loaded {args.arm}: {booster.num_trees()} trees, "
          f"{booster.num_feature()} features")

    # The arm may have been trained with features withheld (the position-bias
    # ablation); score with exactly the columns it was fitted on, in order.
    withheld = set(config.get("arms", {}).get(args.arm, {}).get("withheld", []))
    keep = [i for i, n in enumerate(FEATURE_NAMES) if n not in withheld]
    if booster.num_feature() != len(keep):
        raise SystemExit(
            f"model expects {booster.num_feature()} features but this build "
            f"produces {len(keep)} for arm {args.arm}; refusing to score."
        )

    t0 = time.time()
    art = load_article_table(
        Path(args.articles_zip),
        embeddings_npy=args.embeddings_npy,
        embeddings_json=args.embeddings_json,
        category_vocab=vocab["category"],
        subcategory_vocab=vocab["subcategory"],
        topic_vocab=vocab["topic"],
        article_type_vocab=vocab["article_type"],
    )
    print(f"[2/5] article table: {len(art.category):,} articles on TRAINING "
          f"vocabularies ({time.time()-t0:.0f}s)")

    t0 = time.time()
    prof = build_user_profiles(Path(args.testset_zip), args.split, art)
    pop = build_history_popularity(Path(args.testset_zip), args.split, art)
    print(f"[3/5] user profiles for {len(prof.history_len):,} users + "
          f"history popularity ({time.time()-t0:.0f}s)")

    t0 = time.time()
    beh = load_test_behaviors(Path(args.testset_zip), args.split)
    print(f"[4/5] behaviors: {len(beh):,} impressions ({time.time()-t0:.0f}s)")

    pred_path = out_dir / "predictions.txt"
    n_written = 0
    n_rows = 0
    t0 = time.time()
    with pred_path.open("w") as fh:
        for start in range(0, len(beh), args.chunk_size):
            sub = beh.iloc[start : start + args.chunk_size]
            X, meta = build_feature_frame(sub, art, prof, pop, with_labels=False)
            scores = booster.predict(X[:, keep] if withheld else X)
            sizes = meta["group_sizes"]
            offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
            imp_ids = meta["impression_id"]
            for i, (off, size) in enumerate(zip(offsets, sizes)):
                raw_id = str(imp_ids[i])
                ranks = ranks_for_impression(scores[off : off + size], raw_id, SEED)
                fh.write(f"{raw_id} {json.dumps(ranks, separators=(',', ':'))}\n")
            n_written += len(sub)
            n_rows += X.shape[0]
            del X, meta, scores
            done = start + len(sub)
            rate = done / max(time.time() - t0, 1e-9)
            eta = (len(beh) - done) / max(rate, 1e-9) / 60
            print(f"      {done:,}/{len(beh):,} impressions "
                  f"({100*done/len(beh):.1f}%) {rate:,.0f} imp/s ETA {eta:.0f} min", flush=True)

    if n_written != len(beh):
        raise SystemExit(f"wrote {n_written} lines for {len(beh)} impressions")

    # NAMING GOTCHA, verified against the submissions that actually scored:
    # EB-NeRD's Codabench competition (2469) wants a root-level
    # **`predictions.txt`** inside the zip — that is what submission 888045
    # (Score 0.5404) contained. MIND's competition wants the singular
    # `prediction.txt`, which is also what the bundled
    # `evaluation/official/evaluate.py` opens. The two are not interchangeable,
    # and getting it wrong fails as a scoring error, not a validation error.
    zip_path = out_dir / "prediction.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, arcname="predictions.txt")
    with zipfile.ZipFile(zip_path) as z:
        assert z.namelist() == ["predictions.txt"], (
            f"expected exactly one root-level predictions.txt, got {z.namelist()}"
        )

    print(f"[5/5] wrote {n_written:,} lines over {n_rows:,} candidate rows "
          f"in {(time.time()-t0)/60:.1f} min")
    print(f"      {pred_path}  ({pred_path.stat().st_size/2**20:.0f} MB)")
    print(f"      {zip_path}  ({zip_path.stat().st_size/2**20:.0f} MB)  <- upload this")


if __name__ == "__main__":
    main()
