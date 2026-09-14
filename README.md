# Lexical & Semantic Retrieval on MIND and EB-NeRD

> Assignment 1 — CS4.406 Information Retrieval & Extraction

## Project Overview

This repository implements a reproducible news recommendation pipeline for **Assignment 1 of CS4.406 Information Retrieval & Extraction**.

The system compares **lexical retrieval (BM25)** and **semantic retrieval (embedding-based methods)** on the **MIND** and **EB-NeRD** datasets using a unified evaluation framework. It emphasizes reproducibility, rigorous experimentation, and clean engineering practices while generating leaderboard-ready predictions.

---

# Features

- BM25 lexical retrieval
- Embedding-based semantic retrieval
- Unified data pipeline for MIND and EB-NeRD
- Temporal train/validation/test splitting
- Reproducible experiments
- Automated evaluation and benchmarking
- Codabench submission generation

---

# Repository Structure

```text
.
├── src/
│   ├── pipeline/
│   ├── retrieval/
│   ├── evaluation/
│   ├── datasets/
│   └── utils/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── reproducibility/
├── scripts/
├── experiments/
├── data/
├── decisions/
├── knowledge/
├── configs/
├── submissions/
├── CLAUDE.md
├── PROJECT_STATE.md
├── ARCHITECTURE.md
├── README.md
├── pyproject.toml
├── poetry.lock
└── Makefile
```

---

# Requirements

- Python 3.11+
- Poetry
- GNU Make

---

# Quick Start

## Clone

```bash
git clone <your-repository-url>
cd assignment-1-news-retrieval
```

## Install

```bash
poetry install
```

## Verify Installation

```bash
make test-unit
```

(Not `pytest tests/unit/ -v` directly — `make test-unit` first regenerates
`tests/fixtures/*.zip` via `scripts/generate_test_fixtures.py`. Those fixture
zips are gitignored per Q8's "no `*.zip` in git" policy, so they don't exist
on a fresh clone; `make fixtures` is a real prerequisite of this target, not
just documentation, so a bare `pytest` invocation here would fail with
`FileNotFoundError` on a clean checkout.)

## Build the Data Pipeline

```bash
make data
```

Expected output (verified 2026-08-14 against a real run):

```text
✓ Raw data present
✓ Parsed unified schema
✓ Temporal split preserved (official train/dev/validation boundaries)
✓ Feature store built
```

Builds the fast tier only (MINDsmall + ebnerd_demo) — `include_mind_large`/
`include_ebnerd_small` are opt-in flags on `src.pipeline.orchestrator.build_all`,
not exposed via `make data` (see "Running Experiments" below for how the
MINDlarge/ebnerd_small-scale experiment numbers in this repo were produced).

---

# Getting Started

If you're contributing to the project for the first time:

1. Clone the repository.
2. Run `poetry install`.
3. Run `make data`.
4. Run `make test`.
5. Run the recall@K and Q4 ranking-metrics scripts for BM25 and embeddings on both datasets (see "Running Experiments" below).
6. Read **PROJECT_STATE.md** to understand the current implementation status.
7. Read **ARCHITECTURE.md** to understand the system design.

---

# Project Documentation

| Document | Purpose |
|----------|---------|
| **PROJECT_STATE.md** | Current implementation status and ongoing work |
| **ARCHITECTURE.md** | System architecture and component design |
| **decisions/** | Architecture Decision Records (ADRs) |
| **knowledge/** | Papers, notes, research, and supporting material |
| **CLAUDE.md** | Collaboration contract and engineering standards |

> **Start with `PROJECT_STATE.md` if you're continuing development.**

---

# Testing

Run all tests:

```bash
make test
```

Run individual suites:

```bash
make test-unit            # regenerates tests/fixtures/*.zip first, see below
make test-integration
make test-reproducibility
```

`test-unit` (and `test`, which runs the full suite) depend on a `fixtures`
Make target that regenerates `tests/fixtures/*.zip` from
`scripts/generate_test_fixtures.py` — these zips are gitignored (Q8), not
committed, so a fresh clone has no `tests/fixtures/*.zip` until this runs.
Generation is deterministic and takes under a second, so it's unconditional
rather than timestamp-gated (this repo's `make` is the GNU Make 3.81 macOS
ships by default, which can't cleanly express "these three output files are
one prerequisite" the modern way). `test-integration`/`test-reproducibility`
don't depend on `fixtures` — neither suite reads `tests/fixtures/`.

If you ever need the fixture zips without running tests (e.g. to inspect
them):

```bash
make fixtures
```

Each experiment records:

- Configuration
- Metrics
- Predictions
- Execution log

---

# Running Experiments

All experiment scripts are run directly (there is no `make experiment-*`
wrapper — the `Makefile` only covers `install`, `data`, `test`,
`test-unit`, `test-integration`, `test-reproducibility`, `format`, `lint`,
`clean-data`, and `clean`).

## Recall@K (candidate generation quality)

BM25 (ADR-005/ADR-006):

```bash
poetry run python scripts/run_bm25_experiment.py --dataset mind
poetry run python scripts/run_bm25_experiment.py --dataset ebnerd [--bundle demo|small]
```

Semantic / embedding retrieval (ADR-008):

```bash
poetry run python scripts/run_embed_experiment.py --dataset mind
poetry run python scripts/run_embed_experiment.py --dataset ebnerd [--bundle demo|small]
```

`--bundle` defaults to `small` for `mind`, `demo` for `ebnerd`. Both
scripts write to `experiments/{bm25,embed}_<dataset>_<YYYY-MM-DD>/{config,results}.json`.

## Q4 ranking metrics (AUC/MRR/nDCG@5/nDCG@10/diversity/novelty/coverage, warm/cold, bootstrap CI)

```bash
poetry run python scripts/run_ranking_eval.py --dataset mind --method bm25
poetry run python scripts/run_ranking_eval.py --dataset mind --method embed
poetry run python scripts/run_ranking_eval.py --dataset ebnerd --method bm25 [--bundle demo|small]
poetry run python scripts/run_ranking_eval.py --dataset ebnerd --method embed [--bundle demo|small]
```

`--method` defaults to `bm25`. Writes to
`experiments/ranking_<method>_<dataset>_<YYYY-MM-DD>/{config,results}.json`.

---

# Datasets

## MIND

- Microsoft News Dataset
- English
- MIND-small for rapid iteration
- Uses titles, abstracts, categories, and entity annotations

## EB-NeRD

- Danish news recommendation dataset
- Demo, Small, and Large variants
- Includes pre-computed Word2Vec and BERT embeddings

---

# Common Commands

```bash
# Installation
make install

# Testing
make test
make test-unit
make test-integration
make test-reproducibility

# Formatting
make format
make lint

# Data
make data
make clean-data
make clean
```

Experiments and predictions are run as direct script invocations (see
"Running Experiments" and "Leaderboard Submission" below) — there is no
`make experiment-*`, `make predictions`, or `make view-results` target.

---

# Experiment Results

Experiment artifacts are stored under:

```text
experiments/
```

View the latest experiment:

```bash
cat experiments/$(ls -t experiments/ | head -1)/results.json | python -m json.tool
```

There is no `compare_experiments.py` script — compare two runs directly
with `diff` or by reading both `results.json` files (`python -m json.tool`
on each, or a one-off `python` snippet, as done for the BM25-vs-semantic
comparison writeups in ADR-008 and `PROJECT_STATE.md`'s Benchmarking
Status section).

---

# Leaderboard Submission

Predictions are generated with `scripts/generate_mind_predictions.py` /
`scripts/generate_ebnerd_predictions.py` (`src/submission/{mind,ebnerd}_format.py`,
ADR/Q5), not a `make predictions` target:

```bash
poetry run python scripts/generate_mind_predictions.py --split test --method embed
poetry run python scripts/generate_ebnerd_predictions.py --bundle large --split test --method embed
```

`--method` is `bm25` or `embed`; `--out-dir` overrides the default
(`submissions/mind_large_<split>_<method>/` or
`submissions/ebnerd_<bundle>_<split>_<method>/`). Each run writes
`prediction.txt` (official `impression_id [rank_1,...,rank_N]` format, one
line per impression) plus `truth.txt` when the split has ground truth
(never for `test`, which is blind).

Codabench expects a **zip with `prediction.txt` at the zip root**, not the
raw `.txt` file — create it yourself, e.g.:

```bash
cd submissions/<output_dir> && zip -j prediction.zip prediction.txt
```

Submit `prediction.zip` to:

### MIND Competition

https://www.codabench.org/competitions/13967/

### EB-NeRD (RecSys 2024 Challenge)

https://www.codabench.org/competitions/2469/

The upload itself requires the engineer's own Codabench account/login —
outside what this codebase or Claude Code can do. Both competitions have
real submitted results as of 2026-08-13 (MIND: score 0.6195, EB-NeRD:
score 0.5404) — see `PROJECT_STATE.md`'s Leaderboard Submission row and
`submissions/{mind_large_test_embed,ebnerd_testset_embed}/leaderboard_screenshot_*.png`
for the screenshots (gitignored, kept local as source material for the
design note).

---

# Troubleshooting

## `poetry install` fails

```bash
pip install --upgrade poetry
```

## Dataset download fails

Check your internet connection and retry.

For manual download instructions, see:

```text
scripts/download_data.py
```

## Tests fail

Ensure:

- Python 3.11+
- Poetry dependencies are installed

```bash
poetry install
```

## Experiment results are missing

Recall@K and Q4 ranking-metrics experiments (`experiments/{bm25,embed,ranking_bm25,ranking_embed}_*/`)
each write exactly two files:

```text
config.json
results.json
```

Codabench submission runs (`submissions/*/`) write `prediction.txt` (and
`truth.txt` for non-blind splits) instead — see "Leaderboard Submission"
above.

---

# Assignment 2 — Reproduction, A/B Testing & the Retrieve-then-Rank Pipeline

> Assignment 2 — CS4.406, built on this repository's Assignment 1 pipeline (branch `a2-q3-official-baseline`)

A2 reproduces the official NRMS model (Wu et al. 2019) to the exact published
configuration for both datasets, runs it as an A/B test (control =
reproduced baseline, treatment = one principled change per dataset), and
adds a literal retrieve-then-rank evaluation on top of A1's retriever. See
`decisions/ADR-015-a2-official-baseline-reproduction.md` for the full design
and results, `docs/design_note_a2.tex`/`.pdf` for the write-up, and
`decisions/ADR-016-mind-click-history-features.md` for Q1.

**Environment.** A2's training scripts need `torch`, which is not in this
project's local Poetry environment (`pyproject.toml` — A1 never needed it).
Everything here assumes `pip install torch` (CPU wheel is enough for
`scripts/a2_q2_retrieve_rerank_eval.py` and small local runs; a real GPU is
needed for a full-scale MIND/EB-NeRD training run and was done on Ada, IIIT-H's
SLURM cluster — see `docs/ada_cluster_setup.md` for that environment's own
setup and gotchas, none of which apply to running locally).

## One-command reproduce (Q7): the evaluation half, not the GPU training half

Training the real control/treatment NRMS models end-to-end takes GPU-hours
(MINDlarge: ~34-37h on a single RTX 2080 Ti; see ADR-015). That's not
something a "one command" can honestly paper over, so what *is* one command
is re-running every downstream evaluation from the already-trained models'
output (`scores.parquet`/`results.json`). The per-impression `scores.parquet`
files are **132 MB and deliberately not in git** (Q8's no-large-binaries
policy) — only their sha256 and the metrics/CIs computed from them are
committed, under `results/a2_q3/` (see that directory's own `README.md`).
Re-running this command needs the actual parquet files, either regenerated
by `a2_nrms_official_run.py` (below) or fetched from their durable copy
(`~/a2_model_artifacts/a2_q3_results/` on the machine that trained them, or
Ada's own `$HOME/a2/results/` — matched against the recorded sha256s):

```bash
poetry run python scripts/a2_evaluate_scores.py \
    --control  ~/a2_model_artifacts/a2_q3_results/mind_control_scores.parquet \
    --treatment ~/a2_model_artifacts/a2_q3_results/mind_treatment_scores.parquet \
    --out /tmp/mind_ab.json --dataset mind \
    --source-zip data/raw/mind/MINDlarge_dev.zip \
    --articles data/processed/mind/large/dev/articles.parquet \
    --train-impressions data/processed/mind/large/train/impressions.parquet
```

reproduces Q3's full A/B result table (AUC/MRR/nDCG@5/nDCG@10, paired
bootstrap CI, diversity/novelty/coverage guardrails) exactly — verified
directly, ~4.7 min on a laptop CPU for MINDlarge-dev's 376,471 impressions,
no GPU needed — given the score files. The equivalent EB-NeRD invocation
swaps in the `ebnerd_*` scores/config paths and the `ebnerd_small` processed
dir (244,647 impressions, well under a minute).

## Retraining a model from scratch (the GPU half)

```bash
poetry run python scripts/a2_nrms_official_run.py \
    --dataset mind --arm control --out-dir results/mind_control \
    --mind-train-zip data/raw/mind/MINDlarge_train.zip \
    --mind-dev-zip data/raw/mind/MINDlarge_dev.zip \
    --mind-utils-dir data/processed/mind/mind_utils
# --arm treatment additionally needs --abstract-size 50
```

`--dataset ebnerd` follows the same shape (`--ebnerd-zip`, `--ebnerd-tokens`
from `scripts/a2_ebnerd_prepare_tokens.py`, `--arm treatment` needs no extra
flag — EB-NeRD's treatment is freshness late-fusion, on by construction).
`--max-train-impressions`/`--max-eval-impressions` bound a smoke run (a few
minutes on a laptop CPU); omitting them runs the full official schedule (MIND
10 epochs / EB-NeRD 5 epochs, no subsampling) — that's the multi-hour,
GPU-shaped run. `scripts/a2_nrms_official.sbatch` is the hardened Ada launcher
actually used for every real run in ADR-015 (GPU-capability gate, node
constraint, qos) — a template for anyone repeating this on a different SLURM
cluster, not portable as-is. `scripts/a2_mind_official_nrms.py` /
`a2_ebnerd_official_nrms.py` are the abandoned Option A (official TF/Keras
code) entry points, timeboxed out per ADR-015; kept for the record, not part
of the reproduce path.

## Q1: does click-history feature engineering beat the deployed baseline?

```bash
poetry run python scripts/run_mind_history_features_experiment.py --bundle small
```

Candidate L (ADR-016) — click count, category/subcategory match, and two
recency-weighted affinity features, combined via a cheap logistic regression
and compared against the deployed embedding baseline via the same
`paired_metric_diff_ci` statistic as every other A/B test in this project. A
real, CI-clear **loss**, reported honestly rather than dropped.

## Q2: the literal retrieve-then-rank harness

```bash
poetry run python scripts/a2_q2_retrieve_rerank_eval.py \
    --mind-bundle small --mind-dev-zip data/raw/mind/MINDsmall_dev.zip \
    --checkpoint <model_weights.pt from a2_nrms_official_run.py> \
    --abstract-size 0 --retriever bm25 --k 200 --n-impressions 3000 \
    --out results/a2_q2/mind_small.json
```

Runs A1's own BM25/embedding retriever (`--retriever embedding` reuses the
cached `data/processed/mind/<bundle>/dev/embeddings/`) to pull the top-K
candidates from the **whole corpus** per sampled impression, using the
user's real history as the query — then re-ranks that retrieved list with
the trained `OfficialNRMS` and reports AUC/MRR/nDCG@5/nDCG@10 both before
(retrieval order) and after (NRMS re-rank), paired-bootstrapped, against a
hit rate (= recall@K) computed **fresh on the same sample**, not cited from
an older benchmark run at a possibly different scale. This is deliberately
distinct from the two Codabench submissions, which both rank each
impression's own in-view list (what the competitions actually score) — see
ADR-015's Q2 section for why NRMS is the only re-ranker evaluated this way
(Candidate K's GBDT features are impression-conditional and undefined for a
retrieved-but-never-shown candidate).

Needs a checkpoint from `a2_nrms_official_run.py` (`--checkpoint
.../model_weights.pt`, matching `--abstract-size` to the arm it was trained
with: 0 control / 50 treatment).

## Q3 leaderboard submissions

```bash
poetry run python scripts/a2_generate_mind_test_predictions.py \
    --checkpoint <model_weights.pt> --mind-test-zip data/raw/mind/MINDlarge_test.zip \
    --mind-utils-dir data/processed/mind/mind_utils --abstract-size 50 \
    --out-dir submissions/mind_large_test_nrms_treatment
```

Produces `prediction.zip` in the same official `impression_id
[rank_1,...,rank_N]` format as A1's own submission scripts (reused
unchanged via `src/submission/mind_format.py`). Uploading to Codabench is
always a manual, engineer-only step — see "Leaderboard Submission" above.

## Q4: cost/latency

```bash
poetry run python benchmarks/profile_mind.py    # or profile_ebnerd.py
poetry run python benchmarks/cost_qps.py
```

`cost_qps.py` turns the newest measured latency profile into a p99-SLA/QPS/
cost-per-1000-queries table; its stated pricing/hardware-mapping assumptions
are printed alongside the numbers (see the script's own docstring) — never
report the cost figure without them.

---

# References

## Course

**CS4.406 – Information Retrieval & Extraction**

Assignment 1 — Lexical & Semantic Retrieval
Assignment 2 — Reproduction, A/B Testing & Retrieve-then-Rank

## Datasets

- Wu et al. (2020). *MIND: A Large-scale Dataset for News Recommendation*
- Kruse et al. (2024). *EB-NeRD: Ekstra Bladet News Recommendation Dataset*

## Competitions

- MIND Codabench: https://www.codabench.org/competitions/13967/
- EB-NeRD (RecSys 2024 Challenge): https://www.codabench.org/competitions/2469/