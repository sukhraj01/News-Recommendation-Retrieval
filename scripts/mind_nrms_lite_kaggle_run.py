"""
Candidate J — NRMS-lite for MIND (title encoder + click-history encoder)

WHY THIS CANDIDATE EXISTS
--------------------------
Candidates H1 (linear), H2 (tree), I / I-long / I-pop (attention over frozen
sentence embeddings) all stalled at real MINDsmall-dev AUC ~0.61-0.62 --
essentially flat against the already-deployed embedding-similarity baseline
(local Q4 AUC 0.634 on the candidate-list-only slice; Codabench score 0.6195
for that same submission, ID 886468). Checked against literature before
committing to this candidate: published, well-tuned standard architectures
for this exact dataset (NAML/LSTUR/NRMS) score ~0.64-0.66 AUC on MINDsmall,
not the ~0.70 that prompted this round -- so 0.70 itself is being treated
as a suspect target, not a real bar, pending clarification of what metric/
split that number refers to.

The shared root cause across H1/H2/I/I-long/I-pop: every one of them scored
on top of FROZEN, pre-computed sentence embeddings with a shallow head on
top. None of them included the two components that NAML/LSTUR/NRMS all
share and that the literature attributes their AUC to: (1) a TRAINABLE
encoder over the article title text itself, jointly optimized for the click
task, and (2) an explicit model of the user's click-HISTORY SEQUENCE (not a
static mean-pooled/aggregate feature). This script is a lightweight,
from-scratch reproduction of NRMS's two-encoder design (Wu et al., 2019,
"Neural News Recommendation with Multi-Head Self-Attention"), sized to
train on a single Kaggle GPU, not the paper's full compute budget.

Realistic target given the literature check above: closing some of the gap
toward ~0.64-0.66, not 0.70. This is presented as Candidate J -- one more
data point, evaluated with the exact same per-impression-averaged AUC/MRR/
nDCG@5/nDCG@10 methodology as the official `evaluate.py` (so it is directly
comparable to every number already recorded for H1/H2/I and the baseline),
not assumed to be a win before the real GPU run reports back.

WHAT CHANGED FROM THE ORIGINAL DRAFT (repo-integration review)
-----------------------------------------------------------------
The script was originally written outside this repo (no access to
`src/pipeline`), so it re-implemented raw MINDsmall TSV parsing with polars
by hand. That duplicated `src/datasets/mind.py::parse_mind_split` /
`src/pipeline/orchestrator.py::build_mind_split` -- this project's real,
schema-validated MIND loader (the exact equivalent of `build_ebnerd_bundle`
for EB-NeRD), which every other Kaggle notebook in this project already
reuses (see `notebooks/mind_gated_cohort_mindlarge_kaggle_run.py`). Rewired
to call `build_mind_split` directly against the attached raw zips and read
its unified-schema parquet output instead -- no parallel implementation of
the same TSV-parsing logic, and MIND's static-history invariant / schema
validation now run for free as part of loading. The model classes
(`NewsEncoder`/`UserEncoder`/`NRMSLite`) were extracted to
`src/retrieval/nrms.py` (unit-tested in `tests/unit/test_nrms.py`,
mirroring `rerank.py`/`test_rerank.py`'s existing split between
unit-tested `src/retrieval/` model code and `scripts/`-level training
loops) rather than defined inline here. Per-epoch dev evaluation now calls
`src/evaluation/ranking_metrics.py` (`rank_candidates`/`safe_auc`/`mrr`/
`ndcg_at_k`/`ranking_metric_ci`) instead of hand-rolled AUC/MRR/nDCG/CI
code -- this is what actually makes the "same methodology as official
evaluate.py" claim true, rather than merely asserted.

2026-08-22 ADDENDUM (Ada availability -- reconsidering the Kaggle trade-offs)
-----------------------------------------------------------------------------
Three of this candidate's original design choices were explicitly justified
by Kaggle's time/network constraints, not by architecture reasoning -- per
CLAUDE.md's Decision Reversal clause, now that a dedicated-GPU, ~4-day-wall-
clock resource (Ada, this institution's SLURM cluster -- see
`scripts/mind_nrms_lite_ada_run.py`) is available, those three are worth
reconsidering rather than carried forward unexamined:

1. No pretrained word embeddings -- was "none staged in this project's
   Kaggle datasets, downloading a multi-GB file adds real time risk".
   RECONSIDERED: this script now loads GloVe (`glove.6B.300d.txt`, 300-dim,
   the version whose dimensionality matches this addendum's new
   `EMBED_DIM=300`) if available -- see CELL 4 below. Still gracefully
   degrades to random init if no GloVe file is staged/reachable (an
   enhancement, not a new hard dependency).
2. `embed_dim=128` / `num_heads=8`, smaller than commonly-cited NRMS
   reproductions -- was "traded down for Kaggle time budget". RECONSIDERED:
   bumped to `embed_dim=300` (matches GloVe's dimensionality exactly, no
   extra projection layer needed) / `num_heads=15` (20-dim/head, evenly
   divides 300). Flagged honestly: this is an engineering choice for
   divisibility + GloVe-alignment, not a verified reproduction of Wu et
   al.'s own exact head configuration (the original ADR draft's "15-16-head"
   recollection was itself hedged as approximate, not independently checked
   against the paper).
3. `NRMS_EPOCHS` default of 3, "sized to fit a single Kaggle GPU session".
   RECONSIDERED: real early stopping added (`NRMS_PATIENCE`, default 3 --
   stop once real dev AUC hasn't improved in that many epochs), with the
   epoch count now a generous ceiling (`NRMS_EPOCHS` default raised to 20)
   rather than the actual governor. Because this script already checkpoints
   on the REAL MINDsmall-dev AUC every epoch (not an internal holdout, see
   CELL 7's own docstring), running more epochs is safe against the I/I-long
   overfitting trap by construction -- the worst case is wasted compute, not
   a worse reported result, since the best checkpoint is always kept.

One methodological point worth being explicit about, not smoothed over
under "maximize AUC" pressure: checkpointing directly on MINDsmall-dev
(rather than an internal train-side holdout, per Candidate I's stricter
discipline) means the reported dev AUC is mildly optimistic -- the epoch
selected is the one that happened to do best on the exact set being
reported against. This candidate inherited that choice from the original
draft (which explicitly reasoned about the I/I-long trap when making it);
raising the epoch ceiling increases how many "looks" at dev happen, which
mildly widens that same effect. Not disqualifying (every candidate in this
project's search reports this way), but worth remembering when interpreting
Candidate J's final number rather than treating it as a strictly held-out
result.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
- Does not compute a paired bootstrap against the deployed embedding
  baseline the way Candidates I/I-long/I-pop did (that requires building a
  `sentence-transformers` embedding index over the full dev corpus) --
  deliberately deferred, not silently skipped. `ranking_metric_ci` still
  gives real bootstrap CIs on J's own metrics.

Run this as a Kaggle notebook (GPU: T4 x1 or better). Paste each
`# %% CELL N` block into its own notebook cell, in order. See this
project's `knowledge/ai-usage-log/` for the exact Kaggle dataset-attachment
and accelerator-setup steps.
"""

# %% CELL 1 -- unpack the bundled project code, locate MINDsmall_train.zip /
# MINDsmall_dev.zip on Kaggle, confirm GPU
import os
import shutil
import sys
from pathlib import Path

REPO_DIR = "/kaggle/working/repo"
BUNDLE_NAME = "mind_nrms_lite_src_bundle.zip"
REQUIRED_SUBDIRS = {"src", "scripts"}

bundle_zip = None
extracted_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if BUNDLE_NAME in files:
        bundle_zip = os.path.join(root, BUNDLE_NAME)
        break
    if REQUIRED_SUBDIRS.issubset(set(dirs)):
        extracted_dir = root
        break

if not os.path.exists(REPO_DIR):
    if bundle_zip is not None:
        os.makedirs(REPO_DIR)
        shutil.unpack_archive(bundle_zip, REPO_DIR, "zip")
    elif extracted_dir is not None:
        shutil.copytree(extracted_dir, REPO_DIR)
    else:
        raise FileNotFoundError(
            f"Neither {BUNDLE_NAME} nor an extracted src/scripts folder found "
            "under /kaggle/input -- upload the bundle as a Notebook input first "
            "(zip `src/`, `scripts/mind_nrms_lite_kaggle_run.py`, and "
            "`src/retrieval/nrms.py` from the repo root)."
        )
sys.path.insert(0, REPO_DIR)

import torch  # Kaggle GPU images ship torch preinstalled; pandas/numpy too.

print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type != "cuda":
    print("WARNING: no GPU detected -- this will be very slow. Check the Kaggle "
          "notebook's Accelerator setting (Settings -> Accelerator -> GPU T4 x2).")

REAL_TARGETS = ["MINDsmall_train.zip", "MINDsmall_dev.zip"]
found = {}
for search_root in ["/kaggle/input", "/kaggle/working"]:
    for root, _dirs, files in os.walk(search_root):
        for f in files:
            if f in REAL_TARGETS and f not in found:
                found[f] = os.path.join(root, f)
for t in REAL_TARGETS:
    print(f"{t}: {found.get(t, 'NOT FOUND')}")
missing = [t for t in REAL_TARGETS if t not in found]
if missing:
    raise FileNotFoundError(
        f"Missing required inputs: {missing}. Attach the MINDsmall_train and "
        f"MINDsmall_dev zips as a Kaggle dataset input before running."
    )


# %% CELL 2 -- build the real feature store via this project's own MIND
# loader (src/pipeline/orchestrator.py::build_mind_split), not a
# standalone re-parse. Runs the project's real schema validation
# (src/pipeline/validators.py) and MIND's static-history invariant check
# (src/datasets/mind.py) as a side effect -- checks this candidate gets for
# free by reusing the pipeline instead of duplicating it.
from src.pipeline.orchestrator import build_mind_split  # noqa: E402

OUT_DIR = Path("/kaggle/working/data/processed/mind/small")
build_mind_split(Path(found["MINDsmall_train.zip"]), "train", OUT_DIR)
build_mind_split(Path(found["MINDsmall_dev.zip"]), "dev", OUT_DIR)

import pandas as pd  # noqa: E402

train_articles = pd.read_parquet(OUT_DIR / "train" / "articles.parquet")
train_impressions = pd.read_parquet(OUT_DIR / "train" / "impressions.parquet")
train_history = pd.read_parquet(OUT_DIR / "train" / "user_history.parquet")
dev_articles = pd.read_parquet(OUT_DIR / "dev" / "articles.parquet")
dev_impressions = pd.read_parquet(OUT_DIR / "dev" / "impressions.parquet")
dev_history = pd.read_parquet(OUT_DIR / "dev" / "user_history.parquet")
print("train impressions (rows=candidates):", train_impressions.shape, "articles:", train_articles.shape)
print("dev impressions (rows=candidates):", dev_impressions.shape, "articles:", dev_articles.shape)


# %% CELL 3 -- vocabulary + title matrix, built via src/retrieval/nrms.py +
# src/retrieval/nrms_training.py (unit-tested in tests/unit/test_nrms.py and
# test_nrms_training.py). Vocab is built from train+dev TITLES ONLY (no
# label information, so this is token coverage, not leakage -- same
# standard practice as building a BM25 index over the full corpus).
from src.retrieval.nrms import PAD, build_vocab  # noqa: E402
from src.retrieval.nrms_training import build_title_matrix  # noqa: E402

MAX_TITLE_LEN = 20
MAX_HISTORY_LEN = 50
MIN_FREQ = 2
# 300 to match GloVe's dimensionality exactly (CELL 4) -- 128/8 in the
# original draft was traded down for Kaggle's time budget specifically, not
# for an architecture reason; removed now that Ada lifts that constraint
# (see this file's 2026-08-22 ADDENDUM docstring). 15 heads divides 300
# evenly (20-dim/head); not a verified reproduction of Wu et al.'s own head
# count, chosen for divisibility + GloVe alignment.
EMBED_DIM = 300
NUM_HEADS = 15

all_articles = (
    pd.concat([train_articles[["article_id", "title"]], dev_articles[["article_id", "title"]]])
    .drop_duplicates(subset="article_id")
    .reset_index(drop=True)
)
word2id = build_vocab(all_articles["title"], min_freq=MIN_FREQ)
VOCAB_SIZE = len(word2id)
PAD_ID = word2id[PAD]
print(f"vocab size (min_freq={MIN_FREQ}): {VOCAB_SIZE:,}")

title_matrix, news_id2row, PAD_NEWS_ROW = build_title_matrix(all_articles, word2id, MAX_TITLE_LEN)
title_matrix = title_matrix.to(DEVICE)
print("title_matrix:", title_matrix.shape)


def news_row(nid) -> int:
    return news_id2row.get(nid, PAD_NEWS_ROW)


# %% CELL 4 -- GloVe pretrained word-embedding init (2026-08-22 addendum;
# ADR-012's "no pretrained embeddings" trade-off, reconsidered now that Ada
# removes the Kaggle time/network pressure that motivated it). Looks for an
# already-attached Kaggle dataset first (e.g. the public "glove6b300dtxt"
# dataset -- attach it as a Notebook input), then falls back to downloading
# from Stanford's NLP group directly IF the kernel has internet enabled
# (Settings -> Internet -> On; separate from the GPU toggle). Degrades
# gracefully to random init (this candidate's original behavior) if neither
# is available -- an enhancement, not a new hard dependency.
import urllib.request  # noqa: E402
import zipfile as _zipfile  # noqa: E402

from src.retrieval.nrms_training import init_pretrained_embeddings, load_glove_vectors  # noqa: E402

USE_GLOVE = os.environ.get("NRMS_USE_GLOVE", "1") == "1"
GLOVE_FILENAME = f"glove.6B.{EMBED_DIM}d.txt"
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"  # 822MB zip, all 4 dims (50/100/200/300)

glove_path = None
if USE_GLOVE:
    for root, _dirs, files in os.walk("/kaggle/input"):
        if GLOVE_FILENAME in files:
            glove_path = Path(root) / GLOVE_FILENAME
            break
    if glove_path is None:
        cache_path = Path("/kaggle/working") / GLOVE_FILENAME
        if cache_path.exists():
            glove_path = cache_path
        else:
            try:
                print(f"{GLOVE_FILENAME} not found under /kaggle/input -- attempting download "
                      f"(requires kernel internet access to be enabled)...")
                zip_path = Path("/kaggle/working/glove.6B.zip")
                urllib.request.urlretrieve(GLOVE_URL, zip_path)
                with _zipfile.ZipFile(zip_path) as zf:
                    zf.extract(GLOVE_FILENAME, "/kaggle/working")
                glove_path = cache_path
            except Exception as exc:  # noqa: BLE001 -- genuinely optional, log and continue
                print(f"GloVe download failed ({exc!r}) -- proceeding with random embedding init.")
                glove_path = None

GLOVE_WORDS_FOUND = 0
if glove_path is not None:
    print(f"Loading GloVe vectors from {glove_path}...")
    glove_vectors = load_glove_vectors(glove_path, dim=EMBED_DIM)
    print(f"GloVe vocab loaded: {len(glove_vectors):,} words")
else:
    glove_vectors = None
    print("Proceeding WITHOUT pretrained embeddings (random init) -- "
          "see this file's docstring for why this is now optional, not assumed.")


# %% CELL 5 -- build training examples (NRMS-style negative sampling, K=4)
# via src/retrieval/nrms_training.py, from the unified impressions/
# user_history tables (one exploded row per candidate; grouped back into
# per-impression candidate lists the same way
# `scripts/run_attention_reranker_experiment.py::_train_one_epoch` already
# does for Candidate I). Standard NRMS training recipe: for each impression,
# pair the clicked article with K sampled non-clicked articles from the SAME
# impression, train as a K+1-way softmax classification (positive = index 0).
import random  # noqa: E402

from src.retrieval.nrms_training import NRMSTrainDataset, build_training_examples  # noqa: E402

SEED = 0
NEG_K = 4
# Optional cap for a quick benchmark run before committing to the full job
# (CLAUDE.md's benchmarking philosophy -- measure before a multi-hour run,
# the same discipline Candidate I applied via --max-train-impressions).
# Example: `NRMS_MAX_TRAIN_EXAMPLES=3000` for a 1-epoch smoke test.
MAX_TRAIN_EXAMPLES = os.environ.get("NRMS_MAX_TRAIN_EXAMPLES")
MAX_TRAIN_EXAMPLES = int(MAX_TRAIN_EXAMPLES) if MAX_TRAIN_EXAMPLES else None

train_history_by_user = dict(zip(train_history["user_id"], train_history["article_ids"]))
train_examples = build_training_examples(
    train_impressions, train_history_by_user, news_row, NEG_K, MAX_HISTORY_LEN,
    rng=random.Random(SEED), max_examples=MAX_TRAIN_EXAMPLES,
)
print(f"train examples: {len(train_examples):,}" + (" (capped by NRMS_MAX_TRAIN_EXAMPLES)" if MAX_TRAIN_EXAMPLES else ""))


# %% CELL 6 -- model: NRMS-lite, imported from src/retrieval/nrms.py
# (title self-attention encoder + history self-attention encoder). Unit
# coverage, including the all-padded-row NaN-masking edge case (zero-
# history users, empty titles) verified with a real forward+backward pass
# on this project's installed torch, lives in tests/unit/test_nrms.py --
# see that file's docstring for why forward-only nan_to_num was NOT
# sufficient and what the actual fix was.
from src.retrieval.nrms import NRMSLite  # noqa: E402

torch.manual_seed(SEED)
model = NRMSLite(VOCAB_SIZE, pad_id=PAD_ID, embed_dim=EMBED_DIM, num_heads=NUM_HEADS).to(DEVICE)
if glove_vectors is not None:
    GLOVE_WORDS_FOUND = init_pretrained_embeddings(model.news_encoder.embed, glove_vectors, word2id)
    print(f"GloVe-initialized {GLOVE_WORDS_FOUND:,}/{VOCAB_SIZE:,} vocab words "
          f"({GLOVE_WORDS_FOUND / VOCAB_SIZE:.1%} coverage); remainder keep random init. "
          f"Embeddings stay trainable (fine-tuned end-to-end, not frozen).")
    model = model.to(DEVICE)


# %% CELL 7 -- training loop with per-epoch REAL dev evaluation (via
# src/retrieval/nrms_training.py's shared train_one_epoch/
# evaluate_impressions, backed by this project's own Q4 ranking-metric
# functions in src/evaluation/ranking_metrics.py, not hand-rolled AUC/MRR/
# nDCG/tie-break code -- this is what makes the "same methodology as
# official evaluate.py" claim actually true). Per I/I-long's own finding:
# an internal holdout metric can climb while the real MINDsmall-dev metric
# degrades, so this loop evaluates against the REAL dev set every epoch
# (not a train-side holdout) and checkpoints on that number specifically --
# see this file's 2026-08-22 ADDENDUM docstring for the honest caveat that
# comes with checkpointing directly on dev.
#
# 2026-08-22 addendum: real early stopping (NRMS_PATIENCE) replaces the
# original fixed low EPOCHS-as-governor design now that Ada removes the
# Kaggle-session time pressure that motivated a low default.
from dataclasses import asdict  # noqa: E402

from src.evaluation.ranking_metrics import ranking_metric_ci  # noqa: E402
from src.retrieval.nrms_training import evaluate_impressions, train_one_epoch  # noqa: E402

EPOCHS = int(os.environ.get("NRMS_EPOCHS", 20))       # ceiling / safety valve, not the real governor
PATIENCE = int(os.environ.get("NRMS_PATIENCE", 3))    # real governor: stop after this many non-improving epochs
BATCH_SIZE = 64
LR = 1e-3
N_BOOTSTRAP = 2000

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
train_ds = NRMSTrainDataset(train_examples, MAX_HISTORY_LEN, PAD_NEWS_ROW)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                            num_workers=2, drop_last=True)

dev_history_by_user = dict(zip(dev_history["user_id"], dev_history["article_ids"]))

import time  # noqa: E402
import json  # noqa: E402

history_log = []
best_auc = -1.0
best_dev_df = None
epochs_without_improvement = 0
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, MAX_HISTORY_LEN, title_matrix)
    dev_df = evaluate_impressions(
        model, dev_impressions, dev_history_by_user, news_row, title_matrix,
        MAX_HISTORY_LEN, PAD_NEWS_ROW, DEVICE, seed=SEED,
    )
    dev_auc = float(dev_df["auc"].dropna().mean()) if not dev_df.empty else float("nan")
    elapsed = time.time() - t0
    print(f"epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | "
          f"REAL dev AUC={dev_auc:.4f} n_impressions={len(dev_df)} | elapsed={elapsed/60:.1f}min", flush=True)
    history_log.append({
        "epoch": epoch, "train_loss": train_loss,
        "dev_auc": dev_auc, "n_impressions_scored": len(dev_df), "elapsed_seconds": round(elapsed, 1),
    })

    if dev_auc > best_auc:
        best_auc = dev_auc
        best_dev_df = dev_df
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "/kaggle/working/nrms_lite_best.pt")
        print("  -> new best dev AUC, checkpoint saved")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= PATIENCE:
            print(f"  -> no improvement for {PATIENCE} epochs, stopping early "
                  f"(best={best_auc:.4f} at epoch {epoch - PATIENCE})")
            break


# %% CELL 8 -- compare against every prior candidate on the SAME metric
# (via the SAME bootstrap CI code every other candidate's reported CI comes
# from -- src/evaluation/ranking_metrics.py::ranking_metric_ci), save results
BASELINE_LOCAL_Q4_AUC = 0.634       # embed method, local candidate-list AUC (design_note.md 3.2)
BASELINE_CODABENCH_SCORE = 0.6195   # same submission, official leaderboard (submission 886468)
PRIOR_CANDIDATES_AUC = {
    "H1_linear": None,       # fill in from ADR-011 if available
    "H2_tree": None,
    "I_attention_5ep": 0.6233,
    "I_long_attention_30ep": 0.6214,
    "I_pop_attention_pop_30ep": 0.5133,  # confounded (unnormalized feature), not a clean comparison
}

metric_ci = {}
if best_dev_df is not None and not best_dev_df.empty:
    for metric in ("auc", "mrr", "ndcg5", "ndcg10"):
        metric_ci[metric] = asdict(ranking_metric_ci(best_dev_df, metric, n_bootstrap=N_BOOTSTRAP, seed=SEED))

best_epoch = max(history_log, key=lambda h: h["dev_auc"]) if history_log else None

print("=" * 80)
print("CANDIDATE J (NRMS-lite) SUMMARY -- paste this whole block back")
print("=" * 80)
print(f"Best dev AUC across {len(history_log)}/{EPOCHS} epochs run (early-stopped on patience={PATIENCE}): "
      f"{best_auc:.4f} (epoch {best_epoch['epoch'] if best_epoch else None})")
print(f"GloVe pretrained init: {'yes, ' + format(GLOVE_WORDS_FOUND / VOCAB_SIZE, '.1%') + ' coverage' if glove_vectors is not None else 'no (random init)'}")
if "auc" in metric_ci:
    ci = metric_ci["auc"]
    print(f"Best-epoch dev AUC with bootstrap CI: {ci['metric']:.4f} "
          f"(95% CI {ci['ci_low']:.4f}-{ci['ci_high']:.4f}), n_impressions={ci['n_impressions']}, "
          f"n_users={ci['n_users']}, n_skipped={ci['n_skipped']}")
print(f"vs. deployed baseline (local Q4 AUC): {BASELINE_LOCAL_Q4_AUC:.4f}")
print(f"vs. deployed baseline (Codabench score): {BASELINE_CODABENCH_SCORE:.4f}")
print(f"vs. prior neural candidates: {PRIOR_CANDIDATES_AUC}")
print("Literature band for standard NAML/LSTUR/NRMS on MINDsmall: ~0.64-0.66 AUC")
print()
print(json.dumps(history_log, indent=2))

config = {
    "candidate": "J_nrms_lite",
    "date": time.strftime("%Y-%m-%d"),
    "vocab_size": VOCAB_SIZE,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "max_title_len": MAX_TITLE_LEN,
    "max_history_len": MAX_HISTORY_LEN,
    "neg_k": NEG_K,
    "epochs_ceiling": EPOCHS,
    "epochs_run": history_log[-1]["epoch"] if history_log else 0,
    "patience": PATIENCE,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "pretrained_embeddings": glove_vectors is not None,
    "glove_words_found": GLOVE_WORDS_FOUND,
    "glove_coverage": GLOVE_WORDS_FOUND / VOCAB_SIZE if glove_vectors is not None else None,
    "min_word_freq": MIN_FREQ,
    "n_train_examples": len(train_examples),
    "max_train_examples_cap": MAX_TRAIN_EXAMPLES,
    "seed": SEED,
}
results = {
    "per_epoch": history_log,
    "best_dev_auc": best_auc,
    "best_epoch": best_epoch["epoch"] if best_epoch else None,
    "best_epoch_metric_ci": metric_ci,
    "comparison": {
        "baseline_local_q4_auc": BASELINE_LOCAL_Q4_AUC,
        "baseline_codabench_score": BASELINE_CODABENCH_SCORE,
        "prior_candidates_auc": PRIOR_CANDIDATES_AUC,
        "literature_band_standard_models_mindsmall": [0.64, 0.66],
    },
}
out_dir = Path("/kaggle/working") / f"nrms_lite_mind_small_{config['date']}"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "config.json").write_text(json.dumps(config, indent=2))
(out_dir / "results.json").write_text(json.dumps(results, indent=2))
shutil.make_archive("/kaggle/working/nrms_lite_results", "zip", out_dir)
print(f"\nDownload /kaggle/working/nrms_lite_results.zip and place its contents at "
      f"experiments/{out_dir.name}/ locally.")
print("Also download /kaggle/working/nrms_lite_best.pt if you want the trained weights.")
