# System Architecture

**Last Updated:** 2026-08-09
**Status:** Under Construction — Data Pipeline / Feature Store implemented (Phase 2); Retrieval / Evaluation not yet started

---

# System Overview

A reproducible pipeline comparing lexical (BM25) and semantic (embedding-based) retrieval for news recommendation, evaluated on MIND and EB-NeRD under a unified schema and shared temporal-split protocol (ADR-001, ADR-002). Data Pipeline and Feature Store are implemented; Retrieval and Evaluation are still design placeholders below.

---

# Design Goals

This architecture is intentionally optimized for:

- Reproducibility
- Modularity
- Comparability of retrieval methods
- Extensibility
- Evaluation rigor
- Engineering clarity
- Ease of experimentation

It is intentionally **not** optimized for:

- Production deployment
- Distributed execution
- Online inference
- Ultra-low latency
- Massive-scale serving

These trade-offs are intentional because they align with the goals of this project.

---

# Architectural Principles

Every architectural decision should follow these principles.

1. Simplicity before optimization.
2. Reproducibility before performance.
3. Evidence before intuition.
4. Modular components with explicit interfaces.
5. Benchmark before replacing an existing component.
6. Minimize coupling between subsystems.
7. Prefer reversible decisions early.
8. Optimize only after bottlenecks are measured.
9. Document decisions before they become tribal knowledge.

---

# System Data Flow

```text
Raw Data (MIND / EB-NeRD)
        │
        ▼
 Download
        │
        ▼
 Parse & Clean
        │
        ▼
 Temporal Split
        │
        ▼
 Feature Store
        │
        ▼
 Index Construction
     ├── BM25
     └── ANN
        │
        ▼
 Query Construction
        │
        ▼
 Retrieval
     ├── Lexical
     └── Semantic
        │
        ▼
 Evaluation
        │
        ├── Metrics
        ├── Experiments
        ├── Benchmarks
        ├── Leaderboard Submission
        └── Design Note
```

---

# Components

## Data Pipeline

### Purpose

Parses raw MIND (TSV, zipped) and EB-NeRD (parquet, zipped) bundles into
ADR-002's unified schema, preserving ADR-001's official train/dev/validation
split boundaries without re-partitioning. Never extracts zips to disk.

### Responsibilities

- Presence-check raw bundles under `data/raw/`, fail loudly with instructions if absent (`src/pipeline/download.py::ensure_raw_data`)
- Parse MIND `news.tsv`/`behaviors.tsv` and EB-NeRD `articles.parquet`/`behaviors.parquet`/`history.parquet` into the unified `articles`/`impressions`/`user_history` shape (`src/datasets/mind.py`, `src/datasets/ebnerd.py`)
- Validate every table against ADR-002's schema contract before and after writing (`src/pipeline/validators.py`)
- Write one parquet file per (dataset, bundle, split, table), deterministically sorted

### Dependencies

- pandas, pyarrow (parquet read/write — see ADR-002 amendment note below)

### Primary Artifacts

- `data/processed/{dataset}/{bundle}/{split}/{table}.parquet` (gitignored, rebuilt by `make data`)
- `data/processed/ebnerd/{bundle}/articles.parquet` (shared across EB-NeRD splits — see Known Limitations)

### Public Interface

- `src.pipeline.orchestrator.build_all(raw_dir, processed_dir, include_mind_large=False) -> None`
- `src.datasets.mind.parse_mind_split(zip_path, split) -> dict[str, DataFrame]`
- `src.datasets.mind.parse_mind_test_candidates(zip_path, split) -> dict[str, DataFrame]`
- `src.datasets.ebnerd.parse_ebnerd_articles(zip_path) -> DataFrame`, `parse_ebnerd_split(zip_path, split) -> dict[str, DataFrame]`
- CLI: `make data` → `scripts/build_feature_store.py`

### Consumers

- Feature Store (same artifacts — the pipeline writes directly to the feature-store layout; there is no separate transformation step between them in the current design)

### Current Assumptions

- Internal MIND zip folder name equals the zip's own filename stem (verified across all 5 MIND bundles on disk)
- `user_id`/`article_id` are stable identities across a dataset's own splits (verified: MIND train/dev share 5,943/50,000 users; EB-NeRD train/validation share 1,217/~1,590) — only `impression_id` is split-qualified, since it genuinely restarts per file
- MIND's per-user `history` field is a static snapshot (0 variance across all rows for a user) — re-asserted at build time, not just trusted from ADR-002's original 33,617-user sample

### Known Limitations

- `ebnerd_small`/`ebnerd_large` unverified against this design (built strictly against `ebnerd_demo`'s confirmed schema)
- EB-NeRD's shared `articles.parquet` (one file, not per-split) is a real asymmetry with MIND's per-split `news.tsv` — referential-integrity checks need dataset-aware logic, not one uniform check
- MINDlarge is a slow tier, excluded from `make data`'s default scope (`include_mind_large=True` required)

### Related Decisions

- ADR-001
- ADR-002 (amendment: `pyarrow` constraint corrected from `^12.0.0` to `^22.0.0` — the original pin predates any Python 3.14 wheel and had been silently dropped from pyproject.toml, which would have broken EB-NeRD parquet parsing entirely; caught and fixed during Phase 2 implementation)

---

## Feature Store

### Purpose

The on-disk parquet layout produced by the Data Pipeline, conforming to ADR-002's `articles`/`impressions`/`user_history` schema. Currently the pipeline writes this layout directly — there is no separate feature-store build step distinct from parsing.

### Responsibilities

- Persist parsed tables in a layout that keeps train/dev/validation/test physically separate (never a single file with a `split` column), so a leakage bug can't merge them silently
- Provide a schema every downstream retrieval/evaluation component can depend on without dataset-specific branching

### Dependencies

- Data Pipeline

### Primary Artifacts

- `data/processed/mind/{small,large}/{train,dev,test}/{articles,impressions,user_history,candidates}.parquet`
- `data/processed/ebnerd/{bundle}/articles.parquet`, `data/processed/ebnerd/{bundle}/{train,validation}/{impressions,user_history}.parquet`

### Public Interface

- Direct `pd.read_parquet()` against the layout above
- `src.pipeline.schema.ARTICLES_SCHEMA` / `IMPRESSIONS_SCHEMA` / `USER_HISTORY_SCHEMA` / `CANDIDATES_SCHEMA` as the schema contract

### Consumers

- Retrieval (BM25/semantic indexing over `articles`)
- Evaluation (labels from `impressions.clicked`)

### Current Assumptions

- Mandatory core fields (title, abstract, category, impression timestamp, clicked label) are sufficient for Phase 2–3 retrieval — verified for BM25/semantic candidate scoring, not yet verified against Q4's diversity/coverage metrics (per ADR-002)

### Known Limitations

- No feature APIs beyond raw parquet reads yet (e.g. no recency-weighted history helper) — deferred to the retrieval phase, where the first consumer determines the actual interface needed

### Related Decisions

- ADR-002

---

## Retrieval

### Purpose

[Purpose]

### Responsibilities

- Lexical retrieval
- Semantic retrieval
- Candidate generation

### Dependencies

- Feature Store
- BM25 Index
- ANN Index

### Primary Artifacts

- BM25 index
- ANN / FAISS index
- Candidate rankings

### Public Interface

- Top-K candidate retrieval

### Consumers

- Evaluation

### Current Assumptions

- ...

### Known Limitations

- ...

### Related Decisions

- ADR-005 (Query Construction)
- ADR-006 (BM25 Variant)

---

## Evaluation

### Purpose

[Purpose]

### Responsibilities

- Metric computation
- Benchmark comparison
- Statistical analysis

### Dependencies

- Retrieval

### Primary Artifacts

- Evaluation metrics
- Benchmark reports
- Experiment summaries

### Public Interface

- Evaluation reports

### Consumers

- Reporting
- Design Notes
- Leaderboard Submission

### Current Assumptions

- ...

### Known Limitations

- ...

### Related Decisions

- ADR-010 (Evaluation Strategy)

---

# Decision Traceability

| Component | Governing ADRs |
|------------|----------------|
| Data Pipeline | ADR-001 (Dataset Processing) |
| Feature Store | ADR-002 (Feature Representation) |
| Retrieval | ADR-005 (Query Construction), ADR-006 (BM25 Variant) |
| Evaluation | ADR-010 (Evaluation Strategy) |

---

# Component Interfaces

## Data Pipeline → Feature Store

| Aspect | Description |
|---------|-------------|
| Data Format | |
| Contract | |
| Assumptions | |

---

## Feature Store → Retrieval

| Aspect | Description |
|---------|-------------|
| Data Format | |
| Contract | |
| Assumptions | |

---

## Retrieval → Evaluation

| Aspect | Description |
|---------|-------------|
| Data Format | |
| Contract | |
| Assumptions | |

---

# Cross-Cutting Concerns

These concerns affect every component.

## Configuration

- Centralized configuration
- Environment-independent execution

## Logging

- Structured logging
- Execution traceability

## Reproducibility

- Random seed management
- Dataset versioning
- Configuration versioning

## Experiment Tracking

- Saved configurations
- Metrics
- Artifacts
- Benchmark history

## Testing

- Unit tests
- Integration tests
- Reproducibility tests

## Benchmarking

- Baseline comparison
- Statistical significance
- Slice evaluation

## Error Handling

- Explicit failures
- Helpful diagnostics
- Graceful recovery

## Documentation

- Architecture
- ADRs
- Project State
- Knowledge Base

---

# Architectural Assumptions

| Assumption | Rationale | Validation | Risk |
|------------|-----------|-----------|------|
| | | | |

---

# Scalability & Known Limitations

## Current Design Limits

Performance

- ...

Memory

- ...

Storage

- ...

Accuracy

- ...

---

## Expected Bottlenecks at 10×

- ...
- ...

---

## Migration Strategy

Monitor

- ...

Trigger

- ...

Potential Replacement

- ...

Related ADR

- ADR-XXX

---

# Architecture Changelog

## 2026-08-09 — Phase 2: Data Pipeline / Feature Store implemented

### What Changed

- Implemented `src/datasets/{mind,ebnerd}.py`, `src/pipeline/{schema,validators,download,orchestrator}.py`, `src/utils/{config,ids,io}.py`
- `make data` now builds the full feature store end-to-end from raw MINDsmall + ebnerd_demo zips
- Added `make test-reproducibility` and `make clean-data` (previously referenced by README but missing)
- Added unit (29), integration (48), and reproducibility test suites

### Why

Turns ADR-001/ADR-002 from paper decisions into a working, tested pipeline, per CLAUDE.md's engineering lifecycle.

### Impact

- Row counts verified to match ADR-001's evidence table exactly (MINDsmall train=156,965/50,000 users, dev=73,152/50,000; ebnerd_demo train=24,724/1,590, validation=25,356/1,562)
- Corrected an in-progress, uncommitted `pyproject.toml` change that had silently dropped `pyarrow` entirely (would have broken all EB-NeRD parsing) — see Data Pipeline's Related Decisions note
- Corrected the implementation plan's original ID scheme mid-build: `user_id` must NOT be split-qualified (verified real user overlap across MIND/EB-NeRD splits), only `impression_id` should be

### Related ADR

- ADR-001, ADR-002

### Benchmark

- Not applicable — this phase is pure ETL correctness/reproducibility, no retrieval-performance claims yet

---

# Related Decisions

| ADR | Status | Impact |
|-----|--------|--------|
| ADR-XXX | Decided | |

---

# Future Evolution

## Known Future Work

- Planned architectural improvements
- Components already scheduled

---

## Research Questions

- Questions requiring experiments
- Competing architectural ideas
- Unknown trade-offs

---

## Technical Debt

- Areas intentionally deferred
- Coupling that should eventually be removed
- Temporary implementations

---

# Related Documentation

## Engineering

- PROJECT_STATE.md
- ENGINEERING_WORKFLOW.md

## Decisions

- decisions/

## Knowledge Base

The project maintains an evolving knowledge repository for research, notes, and references.

```text
knowledge/
├── assignment/
│   ├── assignment.md
│   └── grading_notes.md
│
├── retrieval/
│   ├── bm25.md
│   ├── bm25_variants.md
│   ├── semantic_retrieval.md
│   └── faiss.md
│
├── evaluation/
│   ├── metrics.md
│   ├── benchmarking.md
│   └── statistical_testing.md
│
├── research/
│   ├── papers/
│   └── reading_notes/
│
└── industry/
    ├── production_practices.md
    └── case_studies.md
```

This directory acts as the team's internal engineering wiki. It captures research, benchmarking notes, implementation references, paper summaries, and industry practices so knowledge accumulates throughout the project instead of being rediscovered.

---

# Architecture Health Checklist

When modifying this architecture, verify:

- [ ] Interfaces remain explicit.
- [ ] Components remain loosely coupled.
- [ ] Component interfaces remain backward compatible or are appropriately versioned.
- [ ] New decisions are captured in ADRs.
- [ ] Benchmarks exist for performance-sensitive changes.
- [ ] Assumptions remain valid.
- [ ] Changelog is updated.
- [ ] Documentation reflects the current implementation.