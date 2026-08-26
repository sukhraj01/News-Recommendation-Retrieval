#!/usr/bin/env python3
"""Generate official-format MINDlarge_test Codabench predictions using
Candidate J's trained NRMS-lite model (ADR-012, real MINDlarge-dev result
0.6579 vs. baseline 0.6335 -- 2026-08-25 addendum). Routed through
`src/submission/mind_format.py::write_predictions` unchanged, the same
module every real submission from this project has used, via
`NRMSLiteScorer` (`src/retrieval/nrms_training.py`) implementing the same
`Scorer` protocol `EmbeddingScorer`/`GatedScorer` already do -- the
output format is identical by construction, not re-derived.

Critical correctness requirement, not a formality: the trained model's
word vocabulary (`word2id`) was never saved to disk during training, only
the model weights were. Scoring test-set titles correctly requires the
EXACT same word2id the model was trained with -- a different mapping
would silently misalign every word embedding (index 47 might mean
"president" during training and "the" here). This script reconstructs it
deterministically from the same train+dev articles `build_vocab` was
originally called on, then verifies the reconstruction two ways before
trusting it: (1) the resulting vocab_size must match `--expected-vocab-
size` (the real recorded size from the training run's own config.json)
exactly; (2) `model.load_state_dict()` itself will raise a shape-mismatch
error if the embedding table size is wrong -- a second, independent,
automatic check, not just trusting (1).

Usage (see the .sbatch file for the full submission wrapper):
    python scripts/generate_mind_nrms_predictions.py \
        --raw-dir /ssd_scratch/$USER/mind_raw_large \
        --data-dir /ssd_scratch/$USER/mind_nrms_lite/data_large \
        --checkpoint $HOME/mind_nrms_lite_large_results/nrms_lite_best.pt \
        --out-dir $HOME/mind_nrms_predictions
"""
import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.pipeline.download import download_mind_bundle  # noqa: E402
from src.pipeline.orchestrator import build_mind_split, build_mind_test  # noqa: E402
from src.retrieval.nrms import PAD, NRMSLite, build_vocab  # noqa: E402
from src.retrieval.nrms_training import NRMSLiteScorer, build_title_matrix  # noqa: E402
from src.submission.mind_format import write_predictions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True,
                         help="dir for MINDlarge_{train,dev,test}.zip -- downloaded here if not present")
    parser.add_argument("--data-dir", type=Path, required=True,
                         help="scratch dir for the built parquet feature store")
    parser.add_argument("--checkpoint", type=Path, required=True,
                         help="trained nrms_lite_best.pt from the real MINDlarge-dev run")
    parser.add_argument("--out-dir", type=Path, required=True,
                         help="where prediction.txt gets written")
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--num-heads", type=int, default=15)
    parser.add_argument("--max-title-len", type=int, default=20)
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--min-word-freq", type=int, default=2)
    parser.add_argument("--expected-vocab-size", type=int, default=26293,
                         help="real vocab_size from the training run's config.json -- "
                              "the reconstructed vocab MUST match this exactly or the run aborts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}, device: {device}", flush=True)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    train_zip = download_mind_bundle("MINDlarge_train.zip", args.raw_dir)
    dev_zip = download_mind_bundle("MINDlarge_dev.zip", args.raw_dir)
    test_zip = download_mind_bundle("MINDlarge_test.zip", args.raw_dir)

    print("Rebuilding train+dev feature store to reconstruct the exact training vocabulary "
          "(not re-parsed by hand -- same real loader as training used)...", flush=True)
    build_mind_split(train_zip, "train", args.data_dir)
    build_mind_split(dev_zip, "dev", args.data_dir)
    train_articles = pd.read_parquet(args.data_dir / "train" / "articles.parquet")
    dev_articles = pd.read_parquet(args.data_dir / "dev" / "articles.parquet")

    train_dev_articles = (
        pd.concat([train_articles[["article_id", "title"]], dev_articles[["article_id", "title"]]])
        .drop_duplicates(subset="article_id").reset_index(drop=True)
    )
    word2id = build_vocab(train_dev_articles["title"], min_freq=args.min_word_freq)
    vocab_size, pad_id = len(word2id), word2id[PAD]
    print(f"Reconstructed vocab_size={vocab_size:,} (expected {args.expected_vocab_size:,})", flush=True)
    if vocab_size != args.expected_vocab_size:
        raise RuntimeError(
            f"Reconstructed vocab_size ({vocab_size}) != expected ({args.expected_vocab_size}). "
            f"This means word2id does NOT match the checkpoint's training run -- scoring would "
            f"silently misalign every word embedding. Stopping rather than proceeding on a guess. "
            f"Check --raw-dir/--data-dir point at the same MINDlarge train+dev data the real "
            f"training run used, and --min-word-freq matches (default 2)."
        )

    print("Building MINDlarge_test candidates/user_history via the real loader "
          "(src/pipeline/orchestrator.py::build_mind_test)...", flush=True)
    build_mind_test(test_zip, args.data_dir)
    test_articles = pd.read_parquet(args.data_dir / "test" / "articles.parquet")
    test_history = pd.read_parquet(args.data_dir / "test" / "user_history.parquet")
    print(f"test: {len(test_articles):,} articles, {len(test_history):,} users with history", flush=True)

    # title_matrix covers TEST'S OWN ARTICLE CATALOG ONLY -- not train+dev+
    # test unioned. This matters, and got it wrong once already (caught via
    # the N89741 spot-check, not code review): every other scorer in this
    # project (EmbeddingScorer/BM25Scorer/GatedScorer) builds its candidate
    # index from the target split's own articles.parquet only, per ADR-005's
    # "each split's own catalog is the query-time universe" convention. A
    # first version of this script unioned in train+dev articles too (the
    # word2id vocabulary genuinely needs train+dev; the article catalog for
    # SCORING does not) -- which "resurrected" MIND's documented N89741
    # missing-candidate quirk (a candidate real in train/dev's news.tsv but
    # absent from test's own news.tsv) with a borrowed title instead of the
    # -inf fallback every other candidate/submission in this project uses
    # for this exact case. Real run confirmed the bug (0/32 known-affected
    # impressions had N89741 correctly ranked last) before this fix.
    # Practical impact of the bug was narrow (32/2,370,727 impressions,
    # ~0.0013%) but the fix is still real, not cosmetic -- consistency with
    # every other submission's handling of this documented edge case.
    title_matrix, news_id2row, pad_news_row = build_title_matrix(
        test_articles[["article_id", "title"]].reset_index(drop=True), word2id, args.max_title_len
    )
    title_matrix = title_matrix.to(device)
    print(f"title_matrix covers {len(test_articles):,} articles (test's own catalog only)", flush=True)

    model = NRMSLite(vocab_size, pad_id=pad_id, embed_dim=args.embed_dim, num_heads=args.num_heads).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)  # raises on any shape mismatch -- a second, independent vocab check
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} -- shapes matched exactly, "
          f"vocab reconstruction verified twice over.", flush=True)

    query_by_user = dict(zip(test_history["user_id"], test_history["article_ids"]))
    empty_query: list = []
    scorer = NRMSLiteScorer(model, news_id2row, pad_news_row, title_matrix, args.max_history_len, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "prediction.txt"
    t0 = time.time()
    n = write_predictions(out_path, test_zip, scorer, query_by_user, empty_query, has_labels=False, seed=args.seed)
    elapsed = time.time() - t0
    print(f"Wrote {n:,} prediction lines in {elapsed:.1f}s ({elapsed/60:.1f}min) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
