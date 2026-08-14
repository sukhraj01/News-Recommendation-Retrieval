gt # Lexical & Semantic Retrieval on MIND and EB-NeRD

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
poetry run pytest tests/unit/ -v
```

## Build the Data Pipeline

```bash
make data
```

Expected output:

```text
✓ Downloaded datasets
✓ Parsed unified schema
✓ Temporal split created
✓ Feature store built
```

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
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/reproducibility/ -v
```

To verify reproducibility:

```bash
pytest tests/reproducibility/ -v
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

# References

## Course

**CS4.406 – Information Retrieval & Extraction**

Assignment 1 — Lexical & Semantic Retrieval

## Datasets

- Wu et al. (2020). *MIND: A Large-scale Dataset for News Recommendation*
- Kruse et al. (2024). *EB-NeRD: Ekstra Bladet News Recommendation Dataset*

## Competitions

- MIND Codabench: https://www.codabench.org/competitions/13967/
- EB-NeRD (RecSys 2024 Challenge): https://www.codabench.org/competitions/2469/