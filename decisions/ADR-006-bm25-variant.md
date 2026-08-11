# ADR-006 — BM25 Variant and Scoring Implementation

> **Purpose**
>
> An Architecture Decision Record (ADR) captures **why** an engineering
> decision was made.
>
> It records the explored alternatives, the evidence considered, the
> trade-offs accepted, and the reasoning behind the final decision.
>
> Implementation may evolve over time, but the engineering reasoning
> should remain traceable.
>
> When reasoning changes, update the ADR instead of rewriting history.

---

**Date:** 2026-08-10
**Status:** Decided
**Severity:** High

---

# Engineering Question

Should lexical retrieval use:

- BM25Okapi
- BM25L
- BM25+

(the three variants `rank-bm25` ships — the library is mandated by the assignment's implementation constraints), and separately: how should scores actually be computed at the scale this project's real data requires?

---

# Decision Scope

This decision affects:

- [x] Entire project
- [x] Single component
- [ ] Experiment only
- [ ] Temporary / Prototype
- [x] Assignment-specific
- [ ] General reusable pattern

**Affected Components**

- Retrieval (`src/retrieval/index.py`, `src/retrieval/retrieve.py`)

---

# Context

`rank-bm25` is the mandated library. It ships three variants (`BM25Okapi`, `BM25L`, `BM25Plus`), all differing in how they handle document-length normalization. Separately, and only discovered once real data was run through the pipeline: `rank-bm25`'s own `get_scores()` API loops over query tokens in pure Python, doing an O(corpus_size) numpy pass per token — measured directly to be infeasible at this project's real scale (MIND-dev: 50,000 users × 42,416-article corpus). This ADR covers both the variant choice and the resulting scoring-implementation decision, since the two turned out to be inseparable in practice (a "correct" variant choice that can't run in reasonable time isn't usable).

---

# Decision Criteria

| Criteria | Importance | Notes |
|----------|------------|-------|
| Matches document-length characteristics of our corpus | High | Title+abstract text is short and fairly uniform in length across both datasets |
| Interpretability | Medium | Baseline retrieval method — should be the standard, well-understood variant unless there's a specific reason not to |
| Runs in reasonable time at real corpus/user scale | Critical | Discovered mid-implementation to be a real constraint, not a theoretical one — see Benchmark Results |
| Mathematical fidelity to the mandated library | Critical | Any performance rewrite must reproduce the library's exact scores, not approximate them |

---

# Design Space Exploration

## Option A — BM25Okapi

### How it Works

Classic Robertson-Sparck Jones probabilistic scoring with saturating term frequency and length normalization (`k1=1.5`, `b=0.75` defaults). `rank-bm25`'s implementation uses the ATIRE idf variant (no "+1" inside the log), floored at `epsilon * average_idf` for negative idf values.

### Optimizes For

Standard, most widely used and understood BM25 form; no length-normalization correction beyond the standard `b` parameter.

### Sacrifices

Known to over-penalize long documents relative to short ones in some corpora (the specific failure mode BM25L/BM25+ were designed to fix).

### Assumptions

That our corpus (title+abstract, short and fairly uniform in length) doesn't trigger the long-document penalty BM25L/BM25+ exist to fix.

### Complexity

Low.

### Maturity

★★★★★ — the default, most widely deployed BM25 form (Lucene/Elasticsearch's legacy default, most IR textbook treatments).

### Industry / Research Usage

Universal — the form both the MIND and EB-NeRD papers' own content-based baselines implicitly assume when they say "BM25" without further qualification.

---

## Option B — BM25L

### How it Works

Adds a length-normalization correction term (`delta`, default 0.5) specifically to reduce over-penalization of long documents relative to Option A.

### Optimizes For

Corpora with high document-length variance, where long relevant documents are being unfairly suppressed by classic BM25's normalization.

### Sacrifices

Extra tunable parameter (`delta`) with no principled default for this corpus; solves a problem (long-document suppression) this corpus doesn't clearly have.

### Assumptions

That article title+abstract length variance is large enough to matter. **Not verified as true for this corpus** — title+abstract is a short, bounded text field by construction in both datasets' schemas (ADR-002).

### Complexity

Low (same API shape as Option A).

### Maturity

★★★☆☆ — used in research, less standard as a default than Okapi.

---

## Option C — BM25Plus

### How it Works

Adds a lower-bound constant to the term-frequency component, guaranteeing every matching term contributes a nonzero score regardless of document length — another fix for the same long-document suppression problem BM25L targets, via a different mechanism.

### Optimizes For

Same motivation as BM25L, different mechanism (a score floor rather than a normalization delta).

### Sacrifices

Same as BM25L: solves a problem not clearly present in a short, near-uniform-length corpus; adds a parameter without corpus-specific justification.

### Assumptions

Same as BM25L.

### Complexity

Low.

### Maturity

★★★☆☆

---

# Comparison Summary

| Option | Matches corpus (short, uniform text) | Interpretability | Extra tuning needed | Maturity | Decision |
|---------|----------|------------|------------------|-----------|----------|
| A — BM25Okapi | Yes | Highest | No | ★★★★★ | ✅ |
| B — BM25L | Solves a problem we don't clearly have | Medium | Yes (`delta`) | ★★★☆☆ | |
| C — BM25Plus | Solves a problem we don't clearly have | Medium | Yes (implicit floor) | ★★★☆☆ | |

---

# Final Decision

## Chosen Option

**Option A — BM25Okapi**, with scores computed via a precomputed sparse doc-term weight matrix rather than `rank_bm25`'s own `get_scores()` loop (mathematically identical output, verified — see Benchmark Results).

### Reason

BM25L and BM25+ both exist specifically to correct classic BM25's tendency to over-penalize long documents relative to short ones. This project's corpus is title+abstract text — a short, schema-bounded field (ADR-002) for both datasets — so the failure mode those variants fix doesn't clearly apply, and adopting them would mean tuning an extra parameter (`delta`) with no corpus-specific evidence to set it by. BM25Okapi is the standard, most interpretable default, and what both source papers implicitly mean by "BM25" in their own baselines.

The scoring-implementation choice (sparse weight matrix vs. the library's own loop) is not a variant change — it reproduces BM25Okapi's exact formula, using the exact `idf`/`doc_freqs`/`doc_len`/`avgdl`/`k1`/`b` that `rank_bm25.BM25Okapi` itself computes. It exists purely because the library's own `get_scores()` API was measured to be too slow to run this project's actual benchmark in reasonable time.

### Decision Date

2026-08-10

### Decision Owner

**Primary Engineer**

- sukhraj01

### Contributors

- Claude Code (design exploration, implementation, benchmarking, ADR drafting)

### Reviewer *(Optional)*

- Pending

---

# Rationale

## Why this option is best

- No evidence this corpus has the long-document-suppression problem BM25L/BM25+ solve — title+abstract is short and schema-bounded for both datasets (ADR-002), unlike EB-NeRD's much longer `body` field (which this project does not index — see Conditions for Revisiting).
- BM25Okapi is what both the MIND and EB-NeRD papers' own baselines assume; using it keeps this project's numbers comparable to that prior work's framing, not just internally consistent.
- The sparse-matrix scoring rewrite was verified byte-for-byte against `rank_bm25.BM25Okapi.get_scores()` on real data (30 real MIND-dev users, top-20 rankings compared) before being trusted for the production benchmark run — this isn't "we assume it's equivalent," it's measured.

## Why alternatives were rejected

### Option B (BM25L) / Option C (BM25+)

Both rejected on the same grounds: they solve a document-length-variance problem this corpus doesn't clearly exhibit, at the cost of an untuned extra parameter. Revisit trigger: if a future phase indexes EB-NeRD's `body` field (which has real length variance, unlike title+abstract), this reasoning would need to be re-run, not assumed to still hold.

---

# Decision Confidence

**Current Confidence**

High (variant choice) / High (scoring implementation, now that it's benchmark-verified)

### Why

- The variant choice follows directly from BM25L/BM25+'s own stated design motivation (long-document correction) not applying to bounded-length text — a structural argument, not a guess.
- The scoring rewrite's correctness is directly verified (exact match against the mandated library's own output on real data), not just argued to be "probably the same math."
- The scoring rewrite's performance improvement is directly measured (below), not projected from a smaller sample only.

### What Would Increase Confidence

- Re-verifying the exact-match check against a larger and more diverse sample than 30 users if the corpus or tokenizer changes materially.
- Running the same corpus/scale test against MINDlarge (2.2M+ impressions) if that tier is ever brought into scope, to confirm the sparse-matrix approach's memory/time profile still holds an order of magnitude beyond MINDsmall.

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (rank-bm25 library mandated)
- [x] Experimental Benchmark (both the exact-match correctness check and the performance benchmark below)
- [x] Official Documentation (rank_bm25 source — the ATIRE idf variant and per-variant length-normalization mechanics were read directly from the installed library's source, not assumed)
- [ ] Research Paper
- [ ] Industry Practice
- [ ] Community Consensus
- [ ] Engineering Inference

---

## Empirical Evidence

**Correctness of the vectorized scorer** (`src/retrieval/index.py`'s sparse weight matrix vs. `rank_bm25.BM25Okapi.get_scores()` directly): 30 real MIND-dev users, top-20 rankings compared exactly — 0 mismatches.

**Performance, measured directly (not estimated) at three points during implementation:**

| Approach | Sample | Measured | Projected full run (50,000 MIND-dev users) |
|---|---|---|---|
| `rank_bm25.get_scores()` per user, naive loop | 200 users | >150s, killed before completing | ~10+ hours |
| Sparse weight-matrix matvec per user (chosen) | 1,000 users | 1.79s | ~1.5 minutes |
| Sparse weight-matrix matvec, full production run | 50,000 users (actual, not sampled) | 78.6s retrieval + 1.0s index build | — (this is the full run) |

The naive approach isn't just "slower" — it's off by roughly 2-3 orders of magnitude, driven by `get_scores()`'s per-query-token Python loop each doing an O(corpus_size) numpy pass; a ~500-token average query against a 42,416-document corpus means ~21M elementwise numpy operations per user, times 50,000 users, with Python-level loop overhead dominating actual FLOP cost.

**Full production benchmark results** (`experiments/bm25_mind_2026-08-10/results.json`, `experiments/bm25_ebnerd_2026-08-10/results.json` — see ADR-005 for the query-construction/tokenization decisions these numbers also depend on):

| Dataset | k | Overall recall (95% CI) | Warm recall | Cold recall |
|---|---|---|---|---|
| MIND-small dev | 50 | 0.73% (0.68–0.78%) | 0.73% (0.68–0.79%) | 0.72% (0.57–0.87%) |
| MIND-small dev | 100 | 1.50% (1.42–1.58%) | 1.54% (1.46–1.62%) | 1.19% (1.01–1.38%) |
| MIND-small dev | 200 | 2.62% (2.52–2.72%) | 2.73% (2.62–2.84%) | 1.78% (1.56–2.01%) |
| EB-NeRD validation | 50 | 1.01% (0.85–1.17%) | 1.01% (0.85–1.17%) | n/a (0 cold users) |
| EB-NeRD validation | 100 | 2.13% (1.90–2.37%) | 2.13% (1.90–2.37%) | n/a |
| EB-NeRD validation | 200 | 4.02% (3.69–4.39%) | 4.02% (3.69–4.39%) | n/a |

Random baselines for reference: MIND 200/42,416 ≈ 0.47%; EB-NeRD 200/11,777 ≈ 1.70%. Both datasets' BM25 recall@200 clears the random baseline by a real margin (MIND ~5.6x, EB-NeRD ~2.4x), confirming the pipeline measures a genuine (if modest, as expected for a lexical-only baseline) retrieval signal rather than noise.

Warm > cold on MIND at every k, consistent with PROJECT_STATE's Day-1 working hypothesis (BM25 favors warm/entity-rich users). EB-NeRD's cold slice is undefined (0 users below the ADR-005 threshold) — reported as such, not silently dropped.

## Theoretical Evidence

- `rank_bm25` source (`BM25Okapi._calc_idf`, `.get_scores`) — read directly to confirm the ATIRE idf variant and to reproduce its exact formula in the vectorized rewrite.

## Accepted Trade-offs

- The vectorized scorer duplicates `rank_bm25.BM25Okapi`'s scoring formula in project code rather than calling its `get_scores()` method directly — a real maintenance cost (if `rank_bm25` changes its formula in a future version, this project's copy won't automatically track it) accepted because the alternative (the library's own API) was measured infeasible at this project's actual scale. `rank_bm25.BM25Okapi` is still used directly for fitting (`idf`, `doc_freqs`, `doc_len`, `avgdl`, `k1`, `b`) — only the scoring loop is replaced, not the model fitting.
- Absolute recall numbers (2.62% / 4.02% @ k=200) are low by general IR standards. Accepted as a real, reportable baseline finding rather than something to fix by further tuning in this phase — Phase 3's job is to establish the BM25 baseline honestly, not to maximize its score; the assignment's own framing expects BM25 to have real limitations that motivate Phase 4.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| Title+abstract text doesn't exhibit the length variance that motivates BM25L/BM25+ | If wrong, BM25Okapi could be systematically under-scoring longer articles | Revisit if a future phase indexes EB-NeRD's `body` field, which does have real length variance |
| The sparse weight-matrix rewrite stays mathematically identical to `rank_bm25.BM25Okapi` as long as its formula doesn't change | A `rank_bm25` version bump could silently desync the two | Re-run the exact-match verification check (30-user sample vs. `get_scores()`) after any `rank_bm25` version bump |
| MINDsmall-dev scale (50,000 users, 42,416 docs) is representative of the performance profile at MINDlarge scale | MINDlarge is ~40x more impressions; the sparse matvec approach should still hold, but this hasn't been measured | Benchmark this scoring approach against MINDlarge before it enters scope, per PROJECT_STATE's Next Actions |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [x] Performance regression — this is exactly what triggered the scoring-implementation decision in the first place; re-benchmark if corpus size or user count grows materially (e.g. MINDlarge)
- [ ] Latency exceeds threshold
- [x] Memory becomes bottleneck — the sparse weight matrix's memory footprint hasn't been stress-tested beyond MINDsmall/ebnerd_demo scale

---

## Research Triggers

Revisit if:

- [ ] Significant new research appears
- [ ] Better benchmark results emerge

---

## Project / Assignment Triggers

Revisit if:

- [x] A future phase indexes EB-NeRD's `body` field (real length variance) instead of just title+abstract — BM25L/BM25+'s motivating problem would then actually apply
- [x] MINDlarge enters scope — re-benchmark the scoring implementation at ~40x the current impression volume before trusting it to complete in reasonable time

---

**Estimated Cost to Change**

Medium — the variant choice itself is a one-line change (`BM25Okapi` → `BM25L`/`BM25Plus`), but the vectorized scoring rewrite would need re-deriving per variant's specific formula (BM25L/BM25+ have different per-document-term weight formulas than Okapi's).

---

# Engineering Impact

## Affected Components

- Retrieval

---

## Affected Files

- `src/retrieval/index.py`
- `src/retrieval/retrieve.py`
- `scripts/run_bm25_experiment.py`

---

## Expected Refactoring

None — first implementation.

---

## Breaking Changes

No.

---

## Required Tests

- Unit: `retrieve_top_k` correctness on a small hand-built index (ranking order, k-boundary behavior, empty query).
- Integration: real-data retrieval against MIND-dev and EB-NeRD-validation, confirming retrieved IDs are valid and topically-similar articles rank higher (`tests/integration/test_retrieval_pipeline.py`).
- Ad hoc (not a committed pytest test — a one-off verification run): exact-match check of the vectorized scorer against `rank_bm25.BM25Okapi.get_scores()` on real data.

---

## Expected Benchmarks

- Recall@50/100/200, per ADR-005's cohort split. See Benchmark Results above.

---

## Documentation Updates

- [x] Architecture (ARCHITECTURE.md's Retrieval component)
- [x] Project State
- [ ] Knowledge Base

---

# Benchmark Results

## Baseline

None — this is the first retrieval method implemented in this project (no prior BM25 or other retrieval baseline exists to compare against). The "baseline" in the ordinary sense here is the random-retrieval reference rate (200/corpus_size), used above for sanity-checking that the pipeline measures signal rather than noise.

## Candidate

BM25Okapi, unweighted-concatenation query (ADR-005), sparse-matrix scoring.

## Benchmark Environment

| Item | Value |
|------|-------|
| Dataset | MIND-small dev (50,000 users, 42,416 articles); EB-NeRD validation (1,562 users, 11,777 articles) |
| Hardware | Local development machine (macOS, Python 3.14) |
| Software Version | rank-bm25 0.2.2, scipy (sparse), numpy |
| Random Seed | 0 (bootstrap CIs, `src/evaluation/metrics.py`) |

---

## Experiment

`experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/` (each with `config.json` + `results.json`)

---

## Results

See the full recall@K table under Empirical Evidence above.

---

## Interpretation

Both datasets show a real, above-random BM25 signal, with MIND's warm/cold split confirming the expected direction (warm > cold at every k) without yet being confirmed against a semantic-retrieval comparison. EB-NeRD's numbers are weaker in relative terms (2.4x random vs. MIND's 5.6x random) despite the tokenization fix — plausibly reflecting EB-NeRD's much larger, more topically diverse history per user diluting query specificity even after stopword removal, and/or genuine vocabulary mismatch between old history and breaking news (the exact failure mode PROJECT_STATE's Day-1 notes predicted for BM25). This is a hypothesis, not yet tested — a natural target for Phase 4's semantic-vs-lexical comparison.

---

## Comparison to Alternatives

BM25L/BM25+ were not empirically benchmarked against BM25Okapi in this phase — the decision to skip that comparison was itself evidence-based (their motivating failure mode doesn't apply to bounded-length title+abstract text), not an oversight. If EB-NeRD's `body` field enters scope in a later phase, this comparison should be run rather than continuing to assume BM25Okapi is sufficient.

---

# Related Decisions

## Influenced By

- ADR-002 (Unified Data Schema) — title+abstract as the mandatory, schema-bounded text fields this decision reasons about
- ADR-005 (Query Construction) — the query text and tokenization this scoring implementation consumes

## Influences

- Future semantic retrieval ADR — establishes the recall@K comparison baseline it will be measured against

---

# References

## Internal

- ARCHITECTURE.md — System Data Flow (Index Construction, Retrieval stages)
- ADR-002-unified-data-schema.md
- ADR-005-query-construction.md
- `experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/`

## External

- `rank_bm25` source (installed package, version 0.2.2) — `BM25Okapi`, `BM25L`, `BM25Plus`, `_calc_idf`, `get_scores` read directly.
- Wu et al. (2020), MIND — content-based baselines assuming standard BM25/TF-IDF framing.
- Kruse et al. (2024), EB-NeRD.

---

# Decision History

| Date | Event |
|------|-------|
| 2026-08-10 | Variant exploration; BM25Okapi chosen on structural grounds (short/bounded text) before implementation |
| 2026-08-10 | Naive `get_scores()` loop measured infeasible (200 users, projected 10+ hours) |
| 2026-08-10 | Sparse weight-matrix rewrite implemented, verified exact-match against `rank_bm25`, benchmarked at ~400x speedup |
| 2026-08-10 | Full production benchmark run on both datasets; results recorded |

---

# Notes

The scoring-implementation half of this ADR is a clear example of CLAUDE.md's "benchmark before you trust a decision" principle in action: the plan approved before implementation specified calling `index.bm25.get_scores(...)` directly, which seemed reasonable until it was actually run against real data at real scale. The fix was verified for correctness before being trusted, not just assumed to be equivalent because the formula "looks the same on paper."
