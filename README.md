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
5. Run `make experiment-all`.
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

Run BM25:

```bash
make experiment-bm25
```

Run semantic retrieval:

```bash
make experiment-semantic
```

Run the complete benchmark suite:

```bash
make experiment-all
```

Experiment outputs are stored under:

```text
experiments/YYYY-MM-DD_<experiment-name>/
```

Each experiment contains:

```text
config.json
results.json
predictions.csv
log.txt
```

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

# Formatting
make format
make lint

# Data
make data
make clean-data

# Experiments
make experiment-bm25
make experiment-semantic
make experiment-all
make view-results

# Submission
make predictions
make clean
```

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

Compare two experiments:

```bash
python scripts/compare_experiments.py \
    experiments/<bm25>/ \
    experiments/<semantic>/
```

---

# Leaderboard Submission

Generate predictions:

```bash
make predictions
```

Generated files:

```text
submissions/mind_predictions.csv
submissions/ebnerd_predictions.csv
```

Submit to:

### MIND Competition

https://www.codabench.org/competitions/13967/

### EB-NeRD (RecSys 2024 Challenge)

https://www.codabench.org/competitions/2469/

Expected CSV format:

```text
user_id,article_id,score
```

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

Verify that the experiment directory contains:

```text
config.json
results.json
predictions.csv
log.txt
```

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