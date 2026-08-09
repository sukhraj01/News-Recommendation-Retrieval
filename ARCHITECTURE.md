# System Architecture

**Last Updated:** YYYY-MM-DD  
**Status:** Design | Under Construction | Stable with Known Limitations

---

# System Overview

[Provide a concise description of the system, its purpose, and the overall architectural approach.]

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

[Purpose]

### Responsibilities

- ...
- ...

### Dependencies

- ...

### Primary Artifacts

- Cleaned datasets
- Temporal splits

### Public Interface

- ...

### Consumers

- Feature Store

### Current Assumptions

- ...

### Known Limitations

- ...

### Related Decisions

- ADR-001

---

## Feature Store

### Purpose

[Purpose]

### Responsibilities

- ...

### Dependencies

- Data Pipeline

### Primary Artifacts

- Feature matrices
- Processed datasets

### Public Interface

- Dataset loaders
- Feature APIs

### Consumers

- Retrieval

### Current Assumptions

- ...

### Known Limitations

- ...

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

## YYYY-MM-DD

### What Changed

-

### Why

-

### Impact

-

### Related ADR

- ADR-XXX

### Benchmark

- experiments/...

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