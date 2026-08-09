# Project State: Assignment 1 (Lexical & Semantic Retrieval)

> This document captures the current state of the project. It is updated as implementation progresses and should always reflect the latest engineering status.

**Last Updated:** August 9, 2026 (Phase 1B Complete — Architecture Decisions)

**Current Phase:** Phase 2 — Data Pipeline Implementation

**Current Objective:** Build reproducible data pipeline (download → parse → split → feature store)

---

# Component Status Summary

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| Data Pipeline | 🔄 In Progress | 0% | Implementing per ADR-001 (temporal split) and ADR-002 (unified schema) |
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

# Learning Progress

## Day 1 — Mental Model of Recommendation Systems & News Domain (2026-08-09)

**Sources consulted:** Wu et al. 2020 (MIND, ACL Anthology 2020.acl-main.331), Kruse et al. 2024 (EB-NeRD, arXiv:2410.03432 / ACM RecSys Challenge 2024), plus industry figures on Netflix/Amazon/YouTube recommendation ROI (flagged as community-consensus-tier evidence, not audited).

**1. Why companies invest in recommendation systems**
- Core problem: search-cost / discovery problem at catalog scale no human curation team can solve manually.
- Structural difference from normal software: no correctness oracle. Success is a statistical property over a user population (CTR, retention), not pass/fail. Feedback loops exist (exposure bias) — what's shown changes what's observable next.
- ROI figures (Netflix ~80% of streamed hours + ~$1B/yr churn reduction; Amazon ~35% of sales) are widely cited but not independently audited — useful as directional motivation, not benchmarkable claims.

**2. Why news recommendation is structurally different**
- Framed by item relevance half-life: Amazon (months–years), Netflix (years), YouTube (days–years), News (hours). Item cold-start is the *default state* in news, not an edge case — this is why the assignment centers content-based retrieval (BM25 + embeddings) rather than collaborative filtering.
- EB-NeRD paper names three technical challenges explicitly: continuous publish/expire flow (item cold-start), implicit-only feedback, and mandatory reliance on article content.
- Editorial/normative dimension is unique to news among the compared domains — EB-NeRD paper: recommenders "perform a deeply editorial function." Concrete evidence: submitted models varied from 10.2% to 45.7% category coverage at similar accuracy — direct justification for Q4's mandatory diversity/novelty/coverage metrics, not just AUC/nDCG.
- Temporal (never random) splitting exists to prevent the model from seeing information that wouldn't exist yet at real serving time. MIND splits by date; EB-NeRD uses a fixed 21-day click-history window feeding a 7-day forward impression window, non-overlapping in time.
- Noted discrepancy to double-check later: assignment PDF cites EB-NeRD as ~2.7M users/600M+ impressions; the paper's active-user-filtered subset (5–1,000 clicks, May 18–Jun 8 window) reports ~1M users/37M impressions. Likely full-dataset vs. challenge-scoped-subset, not a contradiction — verify against whichever bundle (demo/small/large) we actually load.

**3. Why lexical (BM25) and semantic (embeddings) are complementary, not redundant**
- BM25 strength: exact/near-exact term specificity (named entities, numbers, proper nouns), fully interpretable, zero training cost, index updates incrementally as new articles land — critical given hourly article churn.
- BM25 failure mode: vocabulary mismatch / synonymy (e.g., Danish "Bidens klimaplan" vs. "Præsidentens grønne udspil" — same story, zero shared tokens).
- Embedding strength: captures conceptual/topical similarity and paraphrase even with no lexical overlap; XLM-R adds cross-lingual generalization.
- Embedding failure mode: over-generalization (blurs distinct entities into the same topic cluster) and no representation for brand-new named entities the model hasn't seen.
- MIND paper's own baselines (NAML, NPA, LSTUR, NRMS — all content-based over title/abstract) substantially outperform pure CF/popularity baselines — empirical basis for why the dataset schema centers article text.
- Working hypothesis (not yet evidence — to be tested via Q4 slicing + bootstrap CIs): BM25 likely stronger on warm users / head or entity-heavy articles; embeddings likely stronger on cold-start users and paraphrase-heavy categories.

**Definition of Done:** met — mental model can be explained without recommender-systems background, news-specific challenges are articulated beyond "recommendations are hard," and lexical/semantic complementarity is grounded in both papers' own evidence rather than asserted.

## August 9, 2026 — Phase 1B: Architecture Exploration Complete (Decisions ADR-001 & ADR-002)

- Explored temporal split (7 vs 14 days) using actual on-disk data inspection
- Decision: adopt official train/val splits (7-day windows) — only feasible option given data budgets
- Inspected schemas for MIND and EB-NeRD (corrected two field-attribution errors in the process)
- Decision: unified three-table schema with mandatory core + dataset-specific optional fields
- Both decisions enable single BM25/semantic code path for cross-dataset Q4 comparison
- Confidence: Medium (verified for retrieval; not yet checked against Q4 diversity/coverage metrics)

---

# Recent Decisions

| ADR | Title | Status | Notes |
|-----|-------|--------|------|
| ADR-001 | Temporal Split Strategy | Decided | Use official train/val splits as-is (7-day windows) |
| ADR-002 | Unified Data Schema | Decided | Three tables (articles, impressions, user_history) with mandatory core + optional fields |

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
- Which EB-NeRD population does the assignment's ~2.7M users / 600M+ impressions figure describe vs. the paper's ~1M / 37M active-user-filtered (5–1,000 clicks, May 18–Jun 8) subset? **Partially resolved in ADR-002 (medium confidence):** assignment figure = full raw traffic log; paper figure = active-user-filtered subset that `ebnerd_large`/`ebnerd_small` are sampled from; `ebnerd_demo` (loaded: 1,590 train-window users / 1,562 validation-window users, 24,724 + 25,356 impressions) structurally matches the paper's filtered-subset methodology (21-day history / 7-day window, verified in ADR-001). Still unverified against `ebnerd_small`/`ebnerd_large` directly — remains open until those bundles are downloaded and inspected.

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

- [ ] Implement data pipeline (download + parse + split + feature store build)
- [ ] Run temporal-split leakage tests
- [ ] Verify unified schema works for both datasets in practice

## Upcoming

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

## August 9, 2026 — Day 1: Mental Model of Recommendation Systems & News Domain

### Completed

- Researched business rationale for recommendation investment (Netflix/Amazon/YouTube), with evidence-tier caveats.
- Read and grounded explanation in MIND (Wu et al., 2020) and EB-NeRD (Kruse et al., 2024) papers directly, not just the assignment PDF's summary.
- Built mental model: why news recommendation differs structurally (item half-life, mandatory content-based cold-start mitigation, editorial/normative dimension), why temporal splitting is non-negotiable, why lexical and semantic retrieval are complementary rather than redundant.
- Logged a data-source discrepancy (assignment PDF vs. paper's active-user-filtered EB-NeRD stats) to verify once we load the actual bundle.

### Key Outcomes

- Full write-up recorded in `# Learning Progress` above.
- Working (unverified) hypothesis for Q3.5: BM25 favors warm users/head or entity-heavy articles, embeddings favor cold-start users/paraphrase-heavy categories — to be tested empirically via Q4 slicing, not assumed.

### Next Session

- Begin Phase 1B: architecture exploration — temporal split strategy and unified schema ADRs first, since they constrain the feature store and both retrieval legs.

---

# Important Dates

| Milestone | Target Date | Status |
|-----------|------------|--------|
| Phase 1 Complete | August 12, 2026 | On Track |
| Phase 2 Complete | August 19, 2026 | Planned |
| Pipeline & Retrieval Complete | August 26, 2026 | Planned |
| Leaderboard Submission | August 26, 2026 | Planned |
| Final Assignment Submission | August 27, 2026 | Deadline |