# ADR-001 — Temporal Split Strategy

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

**Date:** 2026-08-09
**Status:** Decided
**Severity:** High

---

# Engineering Question

Should the temporal train/test boundary be:

- **Option A** — the official train/dev/validation splits shipped by each dataset, used as-is (7-day forward test windows for both MIND and EB-NeRD), or
- **Option B** — a custom, re-carved split with a longer trailing test window (e.g. 14 days), built from the raw impression logs ourselves, or
- **Option C** — a per-dataset variable boundary (7 days for MIND-small, 14 days for MIND-large)

given:

- the actual data volume present in the downloaded bundles (MINDsmall, MINDlarge, ebnerd_demo),
- the leakage-prevention design already built into EB-NeRD by its creators,
- the assignment's requirement to submit predictions to each dataset's official leaderboard (Codabench), which scores against splits we do not control.

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

- Data Pipeline
- Feature Store
- Evaluation
- Retrieval (indirectly — determines what "seen" vs. "unseen" articles/users means at index-build time)

---

# Context

Every downstream component — feature store, BM25/ANN index construction, evaluation, and Q4's cold-start/temporal slicing — depends on a single upstream answer: **what counts as "past" (available for training/indexing) and what counts as "future" (held out for evaluation)?** Get this wrong and every other number in the project is contaminated, silently.

Two datasets, two native structures:

- **MIND** ships a strict per-day impression log with no article publish timestamps at all. `news.tsv` has 8 fields (id, category, subcategory, title, abstract, url, title_entities, abstract_entities) — no date field. Per-user click history is an unordered list of article IDs with no individual timestamps. Temporal reasoning is only possible at the impression-timestamp level.
- **EB-NeRD** ships both article publish timestamps (`published_time`, range 2000-10-02 to 2023-06-08 in the demo `articles.parquet`) and impression timestamps, plus a fixed-length click history per user with full per-click timestamps.

Both datasets already arrive pre-split by their creators. The question this ADR answers is not "how do we invent a split" but "do we trust the creators' split, or do we override it" — and if we override it, do we actually have the data to do so.

---

# Decision Criteria

| Criteria | Importance | Notes |
|----------|------------|-------|
| Feasibility given on-disk data volume | Critical | A split that doesn't fit the data isn't a real option |
| Leakage prevention | Critical | Temporal leakage silently invalidates every downstream metric |
| Consistency across MIND and EB-NeRD | High | Q4 cross-dataset comparison assumes comparable evaluation protocols |
| Alignment with official leaderboard scoring | High | Final submissions are scored by Codabench against splits we don't control |
| Engineering complexity | Medium | Custom split code is extra surface area to get wrong and test |
| Test-set statistical power | Medium | Smaller test windows → wider confidence intervals |

---

# Design Space Exploration

## Option A — Use official splits as-is (7-day test windows)

### How it Works

Load each dataset's provided files directly: MIND `train/` + `dev/` (MINDsmall) or `train/` + `dev/` + `test/` (MINDlarge); EB-NeRD `train/` + `validation/` (demo/small/large). Treat MIND-dev and EB-NeRD-validation as our local "test" proxy, since the true hidden test sets (MINDlarge test, EB-NeRD challenge test) are leaderboard-scored only and never locally labeled.

### Optimizes For

Correctness by construction — inherits the creators' own leakage-prevention design. Zero custom split logic. Directly comparable to published baselines in both papers.

### Sacrifices

We don't get to choose the test window size; we accept whatever the dataset ships (1 day for MINDsmall-dev, 7 days for MINDlarge-test and EB-NeRD-validation).

### Assumptions

The official split boundaries are leakage-safe (verified for EB-NeRD; taken as documented for MIND — see Assumptions table below).

### Complexity

Low. Direct file loading, no re-splitting code, no new leakage tests to invent beyond verifying the loader respects file boundaries.

### Maturity

★★★★★

### Industry / Research Usage

This is exactly the protocol both the MIND paper (Wu et al., 2020) and the EB-NeRD paper (Kruse et al., 2024) use for their own reported baselines, and the protocol both Codabench leaderboards score against.

---

## Option B — Re-carve a custom, longer test window (e.g. 14 days)

### How it Works

Discard the official split boundaries. Concatenate all available impression logs per dataset and cut our own trailing window (14 days) as test, with everything before it as train.

### Optimizes For

A theoretically larger, higher-power test set, and a single hand-picked number that feels more deliberate than "whatever the download happened to contain."

### Sacrifices

Feasibility. **EB-NeRD demo's entire dataset spans exactly 14 days** (7-day train + 7-day validation). A 14-day test window consumes 100% of it — there is no data left to train on. **MINDsmall's entire logged period spans exactly 7 days** (6-day train + 1-day dev); a 14-day test window doesn't fit at all. Even on MINDlarge (14 days total: 6+1+7), a 14-day test window equals the *entire* dev+test allocation combined, leaving nothing for local training.

It also throws away EB-NeRD's structural leakage guarantee (see Rationale) and requires writing, testing, and maintaining custom split + leakage-verification code where the dataset creators' own split already does this correctly.

### Assumptions

That we have (or will download) enough raw log volume to support a 14-day held-out window with meaningful training data remaining. Not true for any bundle currently on disk.

### Complexity

High. New split logic, new leakage tests, no longer directly comparable to published baselines or the leaderboard's own scoring protocol.

### Maturity

★☆☆☆☆ — not used by either paper, not how the leaderboards score.

### Industry / Research Usage

None found. Neither paper nor either competition uses a 14-day forward test window.

---

## Option C — Per-dataset variable boundary (7d for MIND-small, 14d for MIND-large)

### How it Works

Use 7 days for MIND-small (matching its total budget), but re-carve 14 days for MIND-large where more data exists.

### Optimizes For

Squeezing maximum test-set size out of whichever bundle we're using.

### Sacrifices

Cross-dataset and cross-bundle comparability — Q4's slicing/comparison analysis assumes a consistent evaluation protocol. A 14-day MIND-large test window would also *still* consume the entirety of MIND-large's official dev+test allocation (1+7=8 days ≈ 14 days once rounded up further), pushing into training days and diverging from the official leaderboard's own test definition — the same feasibility and comparability problems as Option B, just deferred to the large bundle.

### Assumptions

That inconsistent test-window lengths across bundles don't compromise the validity of comparing BM25 vs. semantic retrieval across MIND and EB-NeRD in Q4.

### Complexity

Medium-High. Two different split code paths to maintain and test.

### Maturity

★☆☆☆☆ — not used by either paper or leaderboard.

### Industry / Research Usage

None found.

---

# Comparison Summary

| Option | Feasible with current data | Leakage Risk | Complexity | Comparable to leaderboard | Assignment Fit | Decision |
|---------|----------|------------|------------------|-----------|----------------|----------|
| A — Official splits (7d) | Yes | Low (structural, EB-NeRD verified) | Low | Yes | Excellent | ✅ |
| B — Custom 14d test | No (exceeds total data on 2 of 2 bundles inspected) | Medium (custom code, unverified) | High | No | Poor | |
| C — Variable per dataset | Partially | Medium | Medium-High | Partial | Moderate | |

---

# Final Decision

## Chosen Option

**Option A — Adopt official train/validation splits as-is.** Use MIND-dev and EB-NeRD-validation as our local test sets. This yields a 1-day test window for MINDsmall (its native dev), a 7-day test window for MINDlarge-dev-as-test or EB-NeRD-validation, depending on bundle, rather than a single hand-picked "7 vs. 14" number imposed uniformly. Where we do have a real choice — favoring 7-day-scale windows over 14-day — 7 days is the value that is both native to EB-NeRD's design and the only value that fits within any bundle's actual data budget.

### Reason

Every alternative that involves re-carving a longer window fails on feasibility grounds before it even reaches a trade-off discussion: the data to support a 14-day test window does not exist in any bundle currently downloaded (EB-NeRD demo = 14 total days; MINDsmall = 7 total days). Where feasibility isn't the deciding factor, the official splits still win on evidence: they are what the dataset creators — who had access to the full-scale data and explicit consideration of leakage — chose, and they are what the Codabench leaderboards actually score against. Building and testing custom split logic to arrive at a worse-evidenced, worse-fitting alternative is effort spent moving away from correctness, not toward it.

### Decision Date

2026-08-09

### Decision Owner

**Primary Engineer**

- sukhraj01

### Contributors

- Claude Code (analysis, data inspection, ADR drafting)

### Reviewer *(Optional)*

- Pending

---

# Rationale

## Why this option is best

- **Data budget makes Option A the only universally feasible choice.** MINDsmall's entire logged period is 7 days total (6-day train + 1-day dev). EB-NeRD-demo's entire logged period is 14 days total (7-day train + 7-day validation). A 14-day test window is arithmetically impossible on either bundle without eliminating the training set entirely.
- **Structural leakage prevention, empirically confirmed, not assumed.** Inspecting EB-NeRD-demo directly: `train/history.parquet` per-user history ends exactly 21 days before `train/behaviors.parquet`'s impression window opens (Apr 27 → May 18), with zero gap and zero overlap. Validation shows the identical pattern relative to its own window (May 4 → May 25). This is the dataset creators' own leakage-prevention design, verified against the actual bytes on disk rather than taken on the paper's word.
- **Independent convergence is signal, not coincidence.** MIND (Wu et al., 2020) and EB-NeRD (Kruse et al., 2024) are unrelated datasets built by different organizations for different competitions, and both independently converged on ~7-day forward evaluation windows (MINDlarge's official test week is Nov 16–22, exactly 7 days; EB-NeRD's validation window is exactly 7 days). Two independent teams reaching the same design choice under real production/research constraints is stronger evidence than either choice alone.
- **Leaderboard alignment.** The assignment requires submitting predictions to both Codabench competitions. Those leaderboards score against the official test sets. Any local evaluation protocol that diverges from the official split structure risks producing local metrics that don't predict leaderboard behavior — undermining "metric reliability," one of the concerns this ADR was explicitly asked to address.
- **Lower engineering complexity for higher evidence quality.** Option A requires no custom split code and no new leakage tests beyond confirming the data loader respects file/directory boundaries. Option B or C would require writing, testing, and maintaining split logic whose main achievement is deviating from a design two independent research teams and two competition organizers already validated.

## Why alternatives were rejected

### Option B (custom 14-day test window)

Rejected primarily on feasibility: neither bundle currently on disk contains enough total logged days to support a 14-day test window without consuming the entire dataset (EB-NeRD demo) or exceeding the entire dataset (MINDsmall). Even where more data exists (MINDlarge), a 14-day custom window abandons the official test definition the leaderboard scores against, and discards EB-NeRD's built-in, empirically-verified leakage prevention in favor of unverified custom logic — a strictly worse position on every criterion in the comparison table.

### Option C (variable per dataset)

Rejected because it compromises the one thing this project structurally needs across its two datasets: a consistent evaluation protocol for Q4's cross-dataset BM25-vs-semantic comparison. It also doesn't actually solve Option B's feasibility problem on MIND-large — a 14-day carve there still overruns the official dev+test allocation and diverges from the leaderboard's own scoring window.

---

# Decision Confidence

**Current Confidence**

High

### Why

- The feasibility argument is arithmetic, not opinion — verified directly against the byte contents of the downloaded files, not inferred from the papers.
- The leakage-prevention claim for EB-NeRD is empirically confirmed (exact date-boundary check), not merely asserted from the paper.
- Two independent, unrelated organizations converged on the same design under real constraints.

### What Would Increase Confidence

- Downloading MINDlarge (already present as zips, not yet inspected for a full end-to-end run) and confirming the same 7-day test cadence holds when we actually build the pipeline against it.
- Downloading `ebnerd_small` / `ebnerd_large` and confirming they follow the demo bundle's same 21-day-history / 7-day-window cadence rather than something bundle-specific.
- Running an actual leakage test (Q: "does any train article/impression ID leak into the held-out set?") once the data pipeline exists, rather than relying on the manual date-boundary inspection performed for this ADR.

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (leaderboard submission format requires official-comparable evaluation)
- [x] Experimental Benchmark (direct inspection of on-disk data — see below)
- [x] Official Documentation (MIND/EB-NeRD paper split methodology)
- [ ] Research Paper *(see Theoretical Evidence — used as corroboration, not primary evidence)*
- [ ] Industry Practice
- [ ] Community Consensus
- [ ] Engineering Inference

---

## Empirical Evidence

Direct inspection of the actual downloaded files (not paper summaries):

| Bundle | File | Date Range | Span | Impressions | Users |
|---|---|---|---|---|---|
| MINDsmall | `train/behaviors.tsv` | 2019-11-09 → 2019-11-14 | 6 days | 156,965 | 50,000 |
| MINDsmall | `dev/behaviors.tsv` | 2019-11-15 | 1 day | 73,152 | 50,000 |
| MINDlarge | `train/behaviors.tsv` | 2019-11-09 → 2019-11-14 | 6 days | 2,232,748 | — |
| MINDlarge | `dev/behaviors.tsv` | 2019-11-15 | 1 day | 376,471 | — |
| MINDlarge | `test/behaviors.tsv` | 2019-11-16 → 2019-11-22 | 7 days (official test) | 2,370,727 | — |
| ebnerd_demo | `train/behaviors.parquet` | 2023-05-18 → 2023-05-25 | 7 days | 24,724 | 1,590 |
| ebnerd_demo | `validation/behaviors.parquet` | 2023-05-25 → 2023-06-01 | 7 days | 25,356 | 1,562 |
| ebnerd_demo | `train/history.parquet` | ends 2023-04-27 (21d before train start) | 21 days | — | 1,590 |
| ebnerd_demo | `validation/history.parquet` | ends 2023-05-04 (21d before val start) | 21 days | — | 1,562 |

Schema check: `MINDsmall_dev/news.tsv` fields = `id, category, subcategory, title, abstract, url, title_entities, abstract_entities` — confirmed no publish-timestamp field. `ebnerd_demo/articles.parquet` includes `published_time` (range 2000-10-02 → 2023-06-08).

## Theoretical Evidence

- Wu et al. (2020), MIND: describes the train (6-week history + impression logs) / dev / test split used for the official competition.
- Kruse et al. (2024), EB-NeRD: describes the 21-day click-history window feeding a 7-day forward impression window, non-overlapping in time.

## Accepted Trade-offs

- Local test-set size is fixed by whatever the dataset ships (as small as 1 day / 73k impressions for MINDsmall-dev), rather than something we can grow by re-carving. Accepted because the alternative isn't more data — it's less (Option B is infeasible), and a smaller, correctly-bounded test set is more trustworthy than a larger, leakage-risky one.
- We cannot locally validate against MIND's or EB-NeRD's true hidden test sets — only against MIND-dev / EB-NeRD-validation, which stand in as our test proxy until a Codabench submission. Accepted because this is a structural property of both competitions (test labels are never public), not something any split strategy can work around.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| A 7-day (or shorter, for MINDsmall-dev) forward window is sufficient for statistically meaningful offline evaluation | If too small, confidence intervals on Q4 metrics could be too wide to distinguish BM25 from semantic retrieval | Report CIs alongside every metric (per CLAUDE.md benchmarking philosophy); revisit if CIs are uninformatively wide |
| MIND's per-impression click history contains only clicks that occurred before that impression's own timestamp (i.e., MIND's own construction is leakage-safe) | We cannot independently verify this — MIND provides no per-click timestamps to check it against, unlike EB-NeRD | Treat as documented (not benchmarkable) evidence; flag as an open risk in PROJECT_STATE.md; revisit if MIND documentation or later releases provide click-level timestamps |
| `ebnerd_small` and `ebnerd_large` follow the same 21-day-history / 7-day-window cadence as `ebnerd_demo` | This ADR's leakage-prevention evidence was verified only against the demo bundle | Re-run the same date-boundary check against `ebnerd_small`/`ebnerd_large` once downloaded, before relying on them for final leaderboard submission |
| MIND-dev / EB-NeRD-validation are adequate proxies for the true hidden test sets | If the hidden test sets differ systematically (e.g., different time-of-year, different user population), local metrics may not predict leaderboard scores | Compare local validation metrics against actual Codabench leaderboard scores once submitted; document any divergence |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [ ] Performance regression
- [x] Test-set statistical power proves insufficient (CIs too wide to distinguish methods in Q4)
- [ ] Latency exceeds threshold
- [ ] Memory becomes bottleneck

---

## Research Triggers

Revisit if:

- [ ] Significant new research appears
- [x] `ebnerd_small`/`ebnerd_large` inspection reveals a different window structure than the demo bundle
- [ ] Better benchmark results emerge

---

## Project / Assignment Triggers

Revisit if:

- [x] We move to MINDlarge and want to reconsider whether MINDlarge-dev (currently treated as our local test) should instead be folded into training, with MINDlarge-test's (hidden) leaderboard score as the only true test signal
- [ ] Assignment requirements change
- [x] Local validation metrics diverge meaningfully from actual Codabench leaderboard scores after submission

---

**Estimated Cost to Change**

Easy

---

# Engineering Impact

## Affected Components

- Data Pipeline
- Feature Store
- Evaluation

---

## Affected Files

- src/pipeline/ (data loading, split boundaries)
- src/evaluation/ (test-set definition)
- tests/integration/ (leakage tests)

---

## Expected Refactoring

None — this decision precedes implementation. It establishes the contract the data pipeline must implement: load official directory-partitioned splits directly, do not re-partition by date.

---

## Breaking Changes

No — no implementation exists yet to break.

---

## Required Tests

- Integration: confirm no article/user/impression ID present in a held-out set has any impression timestamp preceding the corresponding history cutoff.
- Integration: confirm loader output row counts match the empirical counts recorded in this ADR's Evidence table (regression guard against silent parsing bugs).
- Reproducibility: confirm split boundaries are identical across repeated pipeline runs (they should be, since we load fixed files rather than compute a split).

---

## Expected Benchmarks

Not applicable to this decision — it is determined by dataset structure and feasibility, not by measured retrieval performance. Per CLAUDE.md's benchmarking philosophy: benchmark when decision uncertainty is real. There is no retrieval-performance uncertainty here to resolve; the question is architectural feasibility.

---

## Documentation Updates

- [x] Architecture (ARCHITECTURE.md — Data Pipeline's "Related Decisions" already stubs ADR-001)
- [x] Project State
- [ ] Knowledge Base

---

# Benchmark Results

Not applicable — see "Expected Benchmarks" above.

---

# Related Decisions

## Influenced By

- None (first ADR in the project)

## Influences

- ADR-002 (Unified Data Schema) — the schema must accommodate MIND's lack of article-level timestamps and EB-NeRD's presence of them, given this ADR's decision to use official splits without re-deriving temporal boundaries ourselves

---

# References

## Internal

- ARCHITECTURE.md — System Data Flow (Temporal Split stage)
- PROJECT_STATE.md — Phase 1A Learning Progress (Day 1 notes on temporal splitting rationale)
- `data/raw/mind/`, `data/raw/ebnerd/` — inspected directly for this ADR

## External

- Wu et al. (2020). *MIND: A Large-scale Dataset for News Recommendation.* ACL Anthology 2020.acl-main.331.
- Kruse et al. (2024). *EB-NeRD: A Large-Scale Dataset for News Recommendation.* arXiv:2410.03432 / ACM RecSys Challenge 2024.
- MIND Codabench: https://www.codabench.org/competitions/13967/
- EB-NeRD Codabench: https://www.codabench.org/competitions/2469/

---

# Decision History

| Date | Event |
|------|-------|
| 2026-08-09 | Exploration started (data inspection of MINDsmall, MINDlarge, ebnerd_demo) |
| 2026-08-09 | Alternatives documented |
| 2026-08-09 | Decision made |

---

# Notes

The original framing of this decision was "last 7 days vs. last 14 days as test boundary," implying we would choose a number and re-carve a split ourselves. Data inspection reframed the question: neither MINDsmall nor ebnerd_demo contains enough total logged days to support a 14-day carve at all, and the datasets already ship pre-split by their creators with a real, verifiable leakage-prevention design (EB-NeRD) or a well-documented one (MIND). The effective decision became "trust the official split vs. override it," and the data made that an easy call rather than a close one.
