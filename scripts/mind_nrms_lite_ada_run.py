#!/usr/bin/env python3
"""Candidate J (NRMS-lite) on IIIT-H's Ada SLURM cluster -- the SLURM-native
counterpart to `scripts/mind_nrms_lite_kaggle_run.py`. Same model
(`src/retrieval/nrms.py`), same data loader (`build_mind_split`), same
training/eval helpers (`src/retrieval/nrms_training.py`) -- only the
execution environment differs (a real argparse script + `sbatch` job
instead of paste-into-notebook-cells), so nothing about the model or
methodology is duplicated or re-derived here. See
`scripts/mind_nrms_lite_ada.sbatch` for the actual job submission script
and setup instructions.

Why this exists: Kaggle's NRMS_EPOCHS default (3) and embed_dim (128) were
explicitly traded down for a single Kaggle GPU session's time budget, and
GloVe pretrained embeddings were skipped for the same reason (see
mind_nrms_lite_kaggle_run.py's 2026-08-22 ADDENDUM docstring). Ada gives a
dedicated GPU (GTX 1080 Ti / RTX 2080 Ti) and a real ~4-day wall-clock
ceiling per job (`research` account, `low`/`medium` QoS) -- this script
carries the same reconsideration (GloVe, embed_dim=300/num_heads=15, real
early stopping instead of a low fixed epoch count) but as a standalone,
resumable-by-checkpoint script rather than a notebook.

Data/storage layout (per Ada's docs -- /home is NFS, 25GB quota;
/scratch and /ssd_scratch are fast but auto-deleted after 7 days):
  - raw MIND zips and the built parquet feature store belong under
    --data-dir (default /ssd_scratch/$USER/mind_nrms_lite/data) -- large,
    disposable, regenerable from the zips.
  - GloVe vectors belong under --glove-dir (same scratch area by default)
    for the same reason.
  - config.json/results.json/model checkpoint are written under
    --results-dir (default $HOME/mind_nrms_lite_results) so they survive
    the scratch 7-day purge without a manual copy step.

Usage (see the .sbatch file for the full submission wrapper):
    python scripts/mind_nrms_lite_ada_run.py \
        --raw-dir /ssd_scratch/$USER/mind_raw \
        --data-dir /ssd_scratch/$USER/mind_nrms_lite/data \
        --glove-dir /ssd_scratch/$USER/glove \
        --results-dir $HOME/mind_nrms_lite_results
"""
import argparse
import json
import random
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.evaluation.ranking_metrics import ranking_metric_ci  # noqa: E402
from src.pipeline.download import download_mind_bundle  # noqa: E402
from src.pipeline.orchestrator import build_mind_split  # noqa: E402
from src.retrieval.nrms import PAD, NRMSLite, build_vocab  # noqa: E402
from src.retrieval.nrms_training import (  # noqa: E402
    NRMSTrainDataset, build_title_matrix, build_training_examples, evaluate_impressions,
    init_pretrained_embeddings, load_glove_vectors, train_one_epoch,
)

BASELINE_LOCAL_Q4_AUC = 0.634
BASELINE_CODABENCH_SCORE = 0.6195
PRIOR_CANDIDATES_AUC = {
    "H1_linear": None,
    "H2_tree": None,
    "I_attention_5ep": 0.6233,
    "I_long_attention_30ep": 0.6214,
    "I_pop_attention_pop_30ep": 0.5133,
}
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def _ensure_glove(glove_dir: Path, embed_dim: int, no_download: bool) -> Path | None:
    """Downloads via `wget -c` (shelled out, not `urllib.request.urlretrieve`)
    specifically because a real run hit `ContentTooShortError` -- the
    connection to Stanford's server dropped partway (160MB/822MB) and
    `urlretrieve` has no retry/resume logic at all, so a single dropped
    connection meant total failure. `wget -c` resumes from a partial file
    and retries automatically -- the actual fix for the failure mode
    observed, not just "try again and hope"."""
    glove_dir.mkdir(parents=True, exist_ok=True)
    target = glove_dir / f"glove.6B.{embed_dim}d.txt"
    if target.exists():
        return target
    if no_download:
        return None
    zip_path = glove_dir / "glove.6B.zip"
    try:
        print(f"Downloading GloVe from {GLOVE_URL} via wget -c (resumable/retrying -- this needs "
              f"internet access from wherever this script runs)...", flush=True)
        subprocess.run(
            ["wget", "-c", "--tries=10", "--waitretry=15", "--timeout=60",
             "--retry-connrefused", GLOVE_URL, "-O", str(zip_path)],
            check=True,
        )
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(target.name, glove_dir)
        return target
    except Exception as exc:  # noqa: BLE001 -- optional enhancement, degrade gracefully
        print(f"GloVe download failed ({exc!r}) -- proceeding with random embedding init. "
              f"To use GloVe, download glove.6B.zip manually (e.g. from the login node) and "
              f"place glove.6B.{embed_dim}d.txt at {target}.", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True,
                         help="dir for MINDsmall_train.zip/MINDsmall_dev.zip -- downloaded here "
                              "automatically if not already present (safe to point at node-local "
                              "scratch storage that may be empty on this run)")
    parser.add_argument("--data-dir", type=Path, required=True,
                         help="scratch dir for the built parquet feature store")
    parser.add_argument("--glove-dir", type=Path, default=None,
                         help="scratch dir for GloVe vectors (default: --data-dir/glove)")
    parser.add_argument("--no-glove", action="store_true", help="skip GloVe entirely, random init")
    parser.add_argument("--no-glove-download", action="store_true",
                         help="use a locally-staged GloVe file only, never attempt a download")
    parser.add_argument("--results-dir", type=Path, required=True,
                         help="persistent dir (NOT scratch) for config/results/checkpoint")
    parser.add_argument("--embed-dim", type=int, default=300)
    parser.add_argument("--num-heads", type=int, default=15)
    parser.add_argument("--max-title-len", type=int, default=20)
    parser.add_argument("--max-history-len", type=int, default=50)
    parser.add_argument("--bundle", choices=["small", "large"], default="small",
                         help="MINDsmall (default) or MINDlarge -- selects which zips to "
                              "download/parse (large is ~14x more train impressions, ~5x more "
                              "dev impressions; verified real ratios, not estimated -- see ADR-012)")
    parser.add_argument("--min-word-freq", type=int, default=2)
    parser.add_argument("--neg-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30, help="ceiling / safety valve, not the real governor")
    parser.add_argument("--patience", type=int, default=3, help="real governor: stop after N non-improving epochs")
    parser.add_argument("--max-train-examples", type=int, default=None,
                         help="cap for a benchmark/smoke-test run before committing to the full job")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch: {torch.__version__}, cuda available: {torch.cuda.is_available()}, device: {device}", flush=True)
    if device.type != "cuda":
        print("WARNING: no GPU detected -- confirm the sbatch script requested --gres=gpu:1 "
              "and that this job actually landed on a GPU node.", flush=True)

    # Auto-download if not already present -- required, not just convenient:
    # /ssd_scratch is LOCAL to each compute node on Ada (confirmed the hard
    # way this session: data staged on one node was invisible to a batch
    # job that landed on a different node), so this script can't assume
    # --raw-dir was pre-populated by an earlier interactive session on a
    # now-irrelevant machine. download_mind_bundle is idempotent (no-ops if
    # the file already exists), so this is safe whether or not the data
    # happens to already be there.
    bundle_prefix = "MINDsmall" if args.bundle == "small" else "MINDlarge"
    train_zip_name, dev_zip_name = f"{bundle_prefix}_train.zip", f"{bundle_prefix}_dev.zip"
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"Ensuring {train_zip_name}/{dev_zip_name} are present under {args.raw_dir} "
          f"(downloads if missing -- do not assume a prior session's node-local scratch persists)...", flush=True)
    train_zip = download_mind_bundle(train_zip_name, args.raw_dir)
    dev_zip = download_mind_bundle(dev_zip_name, args.raw_dir)

    print("Building MIND feature store via src/pipeline/orchestrator.py::build_mind_split "
          "(this project's real MIND loader -- not a standalone re-parse)...", flush=True)
    build_mind_split(train_zip, "train", args.data_dir)
    build_mind_split(dev_zip, "dev", args.data_dir)

    train_articles = pd.read_parquet(args.data_dir / "train" / "articles.parquet")
    train_impressions = pd.read_parquet(args.data_dir / "train" / "impressions.parquet")
    train_history = pd.read_parquet(args.data_dir / "train" / "user_history.parquet")
    dev_articles = pd.read_parquet(args.data_dir / "dev" / "articles.parquet")
    dev_impressions = pd.read_parquet(args.data_dir / "dev" / "impressions.parquet")
    dev_history = pd.read_parquet(args.data_dir / "dev" / "user_history.parquet")
    print(f"train impressions (rows=candidates): {train_impressions.shape}, articles: {train_articles.shape}")
    print(f"dev impressions (rows=candidates): {dev_impressions.shape}, articles: {dev_articles.shape}")

    all_articles = (
        pd.concat([train_articles[["article_id", "title"]], dev_articles[["article_id", "title"]]])
        .drop_duplicates(subset="article_id").reset_index(drop=True)
    )
    word2id = build_vocab(all_articles["title"], min_freq=args.min_word_freq)
    vocab_size, pad_id = len(word2id), word2id[PAD]
    print(f"vocab size (min_freq={args.min_word_freq}): {vocab_size:,}")

    title_matrix, news_id2row, pad_news_row = build_title_matrix(all_articles, word2id, args.max_title_len)
    title_matrix = title_matrix.to(device)

    def news_row(nid) -> int:
        return news_id2row.get(nid, pad_news_row)

    glove_vectors = None
    glove_words_found = 0
    if not args.no_glove:
        glove_dir = args.glove_dir or (args.data_dir / "glove")
        glove_path = _ensure_glove(glove_dir, args.embed_dim, args.no_glove_download)
        if glove_path is not None:
            print(f"Loading GloVe vectors from {glove_path}...", flush=True)
            glove_vectors = load_glove_vectors(glove_path, dim=args.embed_dim)
            print(f"GloVe vocab loaded: {len(glove_vectors):,} words")
        else:
            print("Proceeding WITHOUT pretrained embeddings (random init).", flush=True)

    train_history_by_user = dict(zip(train_history["user_id"], train_history["article_ids"]))
    train_examples = build_training_examples(
        train_impressions, train_history_by_user, news_row, args.neg_k, args.max_history_len,
        rng=random.Random(args.seed), max_examples=args.max_train_examples,
    )
    print(f"train examples: {len(train_examples):,}" +
          (" (capped by --max-train-examples)" if args.max_train_examples else ""))
    if not train_examples:
        raise RuntimeError("Zero training examples built -- check the raw zips are the real "
                            "MINDsmall files, not a fixture/sample.")

    torch.manual_seed(args.seed)
    model = NRMSLite(vocab_size, pad_id=pad_id, embed_dim=args.embed_dim, num_heads=args.num_heads).to(device)
    if glove_vectors is not None:
        glove_words_found = init_pretrained_embeddings(model.news_encoder.embed, glove_vectors, word2id)
        print(f"GloVe-initialized {glove_words_found:,}/{vocab_size:,} vocab words "
              f"({glove_words_found / vocab_size:.1%} coverage); remainder keep random init.", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_ds = NRMSTrainDataset(train_examples, args.max_history_len, pad_news_row)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                                num_workers=4, drop_last=True)
    dev_history_by_user = dict(zip(dev_history["user_id"], dev_history["article_ids"]))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.results_dir / "nrms_lite_best.pt"

    history_log = []
    best_auc = -1.0
    best_dev_df = None
    epochs_without_improvement = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.max_history_len, title_matrix)
        dev_df = evaluate_impressions(
            model, dev_impressions, dev_history_by_user, news_row, title_matrix,
            args.max_history_len, pad_news_row, device, seed=args.seed,
        )
        dev_auc = float(dev_df["auc"].dropna().mean()) if not dev_df.empty else float("nan")
        elapsed = time.time() - t0
        print(f"epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"REAL dev AUC={dev_auc:.4f} n_impressions={len(dev_df)} | elapsed={elapsed/60:.1f}min", flush=True)
        history_log.append({
            "epoch": epoch, "train_loss": train_loss,
            "dev_auc": dev_auc, "n_impressions_scored": len(dev_df), "elapsed_seconds": round(elapsed, 1),
        })

        if dev_auc > best_auc:
            best_auc = dev_auc
            best_dev_df = dev_df
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            # Write results-so-far after every improvement too -- cheap
            # resilience against a killed/preempted job losing everything,
            # without a full checkpoint/resume system this job's expected
            # runtime (well under the 4-day ceiling) doesn't warrant.
            (args.results_dir / "history_log_partial.json").write_text(json.dumps(history_log, indent=2))
            print(f"  -> new best dev AUC, checkpoint saved to {checkpoint_path}", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  -> no improvement for {args.patience} epochs, stopping early "
                      f"(best={best_auc:.4f} at epoch {epoch - args.patience})", flush=True)
                break

    metric_ci = {}
    if best_dev_df is not None and not best_dev_df.empty:
        for metric in ("auc", "mrr", "ndcg5", "ndcg10"):
            metric_ci[metric] = asdict(ranking_metric_ci(best_dev_df, metric, n_bootstrap=2000, seed=args.seed))
    best_epoch = max(history_log, key=lambda h: h["dev_auc"]) if history_log else None

    print("=" * 80)
    print("CANDIDATE J (NRMS-lite, Ada) SUMMARY")
    print("=" * 80)
    print(f"Best dev AUC across {len(history_log)}/{args.epochs} epochs run "
          f"(early-stopped on patience={args.patience}): {best_auc:.4f} "
          f"(epoch {best_epoch['epoch'] if best_epoch else None})")
    if "auc" in metric_ci:
        ci = metric_ci["auc"]
        print(f"Best-epoch dev AUC with bootstrap CI: {ci['metric']:.4f} "
              f"(95% CI {ci['ci_low']:.4f}-{ci['ci_high']:.4f}), n_impressions={ci['n_impressions']}, "
              f"n_users={ci['n_users']}, n_skipped={ci['n_skipped']}")
    print(f"GloVe pretrained init: "
          f"{'yes, ' + format(glove_words_found / vocab_size, '.1%') + ' coverage' if glove_vectors else 'no (random init)'}")
    print(f"vs. deployed baseline (local Q4 AUC): {BASELINE_LOCAL_Q4_AUC:.4f}")
    print(f"vs. deployed baseline (Codabench score): {BASELINE_CODABENCH_SCORE:.4f}")
    print(f"vs. prior neural candidates: {PRIOR_CANDIDATES_AUC}")
    print("Literature band for standard NAML/LSTUR/NRMS on MINDsmall: ~0.64-0.66 AUC")

    config = {
        "candidate": "J_nrms_lite", "environment": "ada_slurm", "date": str(date.today()),
        "vocab_size": vocab_size, "embed_dim": args.embed_dim, "num_heads": args.num_heads,
        "max_title_len": args.max_title_len, "max_history_len": args.max_history_len,
        "neg_k": args.neg_k, "epochs_ceiling": args.epochs, "epochs_run": len(history_log),
        "patience": args.patience, "batch_size": args.batch_size, "lr": args.lr,
        "pretrained_embeddings": glove_vectors is not None, "glove_words_found": glove_words_found,
        "glove_coverage": (glove_words_found / vocab_size) if glove_vectors else None,
        "min_word_freq": args.min_word_freq, "n_train_examples": len(train_examples),
        "max_train_examples_cap": args.max_train_examples, "seed": args.seed,
    }
    results = {
        "per_epoch": history_log, "best_dev_auc": best_auc,
        "best_epoch": best_epoch["epoch"] if best_epoch else None,
        "best_epoch_metric_ci": metric_ci,
        "comparison": {
            "baseline_local_q4_auc": BASELINE_LOCAL_Q4_AUC,
            "baseline_codabench_score": BASELINE_CODABENCH_SCORE,
            "prior_candidates_auc": PRIOR_CANDIDATES_AUC,
            "literature_band_standard_models_mindsmall": [0.64, 0.66],
        },
    }
    (args.results_dir / "config.json").write_text(json.dumps(config, indent=2))
    (args.results_dir / "results.json").write_text(json.dumps(results, indent=2))
    (args.results_dir / "history_log_partial.json").unlink(missing_ok=True)
    print(f"\nWrote {args.results_dir}/config.json, results.json, and nrms_lite_best.pt")
    print(f"Copy experiments/{config['date']}_nrms_lite_mind_small/ locally from {args.results_dir} "
          f"before this job's scratch data (7-day auto-delete) matters -- results.json/config.json/"
          f"the checkpoint are already in --results-dir, not scratch, so no urgency for those "
          f"specifically, but worth pulling down promptly regardless.")


if __name__ == "__main__":
    main()
