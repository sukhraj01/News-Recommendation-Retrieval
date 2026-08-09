# Project State: Assignment 1 (Lexical & Semantic Retrieval)

> This document captures the current state of the project. It is updated as implementation progresses and should always reflect the latest engineering status.

**Last Updated:** August 5, 2026 (Project Initialization)

**Current Phase:** Phase 1 — Understanding & Environment Setup

**Current Objective:** Build a strong conceptual foundation and prepare the development environment before beginning system architecture.

---

# Component Status Summary

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| Data Pipeline | ⏳ Not Started | 0% | Design phase |
| Lexical Retrieval (BM25) | ⏳ Not Started | 0% | Awaiting architecture decisions |
| Semantic Retrieval | ⏳ Not Started | 0% | Awaiting embedding model decision |
| Evaluation Harness | ⏳ Not Started | 0% | Design phase |
| Benchmarking Framework | ⏳ Not Started | 0% | Repository structure prepared |
| Leaderboard Submission | ⏳ Not Started | 0% | Planned for final phase |

---

# Progress Summary

## Recently Completed

- ✅ Repository initialized
- ✅ Poetry environment configured
- ✅ Testing framework (pytest) configured
- ✅ Initial documentation system established

## In Progress

- Understanding recommendation systems from first principles
- Verifying local development environment
- Preparing for architecture and design phase

---

# Recent Decisions

| ADR | Title | Status | Notes |
|-----|-------|--------|------|
| — | None yet | Pending | Architectural decisions begin in Phase 2 |

---

# Open Engineering Questions

The following decisions will be resolved during the architecture phase:

- Which BM25 variant should be implemented?
- How should user queries be constructed?
- Which embedding model should be used?
- Which ANN backend should be adopted?
- How should user embeddings be represented?
- What cold-start strategy should be used?
- What temporal split strategy should be adopted?

---

# Current Risks

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|-----------|--------|
| Temporal data leakage | Medium | Critical | Automated leakage tests | Active |
| Submission format mismatch | Low | High | Dry-run before submission | Planned |
| Timeline pressure near deadline | Medium | Medium | Weekly milestone reviews | Monitoring |

---

# Benchmarking Status

**Current Status:** No experiments have been executed yet.

**Planned Baselines**

- BM25 baseline (MIND-small)
- Semantic baseline (MIND-small)
- BM25 vs. Semantic comparison on both datasets

Experiment results will be recorded in the `experiments/` directory as implementation progresses.

---

# Next Actions

## Immediate

1. Complete conceptual understanding of the recommendation problem.
2. Verify the development environment.
3. Begin architecture exploration.

## Upcoming

- Finalize major architectural decisions.
- Create Architecture Decision Records (ADRs).
- Design data pipeline and indexing strategy.
- Design evaluation framework.
- Update `ARCHITECTURE.md`.

---

# Session Notes

## August 5, 2026 — Project Initialization

### Completed

- Repository initialized.
- Documentation structure created.
- Engineering workflow established.
- Project organization designed around:
  - Clear architecture
  - Reproducibility
  - Benchmarking
  - Decision tracking
  - Long-term maintainability

### Key Outcomes

- Documentation is organized so project context can be reconstructed quickly.
- Engineering decisions will be documented through ADRs.
- Benchmarking and reproducibility are first-class parts of the workflow.

### Next Session

- Begin studying recommendation system fundamentals.
- Complete environment verification.
- Prepare for architecture and design.

---

# Important Dates

| Milestone | Target Date | Status |
|-----------|------------|--------|
| Phase 1 Complete | August 12, 2026 | On Track |
| Phase 2 Complete | August 19, 2026 | Planned |
| Pipeline & Retrieval Complete | August 26, 2026 | Planned |
| Leaderboard Submission | August 26, 2026 | Planned |
| Final Assignment Submission | August 27, 2026 | Deadline |