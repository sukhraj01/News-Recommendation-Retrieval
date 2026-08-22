# ADR-005 — Query Construction for BM25 Retrieval

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

Given a user's `article_ids` click history (ADR-002's `user_history` table), what text should be submitted as the BM25 query, and how should users be tokenized, weighted, and slotted into a warm/cold cohort for evaluation — given:

- ADR-002's explicit warning that MIND's history order is "presumed chronological, not independently verified,"
- the assignment's requirement to compare warm vs. cold users,
- the two datasets' wildly different history-length distributions (MIND median 15 articles; EB-NeRD median 96, max 1,459),
- the constraint to keep one code path for both datasets (ADR-002's governing goal).

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

- Retrieval (`src/retrieval/query.py`, `src/retrieval/tokenize.py`)
- Evaluation (warm/cold cohort assignment consumes this ADR's threshold)

---

# Context

BM25 needs a query — a bag of terms — not a user ID. The feature store gives us a user's ordered `article_ids` list (ADR-002); there is no explicit "user profile" text anywhere upstream. Three sub-decisions are bundled here because they're not independent: how history becomes query text, how that text is tokenized, and how "cold" is defined for the warm/cold comparison the assignment requires — the tokenization choice in particular was not settled until it was benchmarked (see Benchmark Results below), which is why this ADR's confidence section documents a mid-flight correction rather than a single up-front decision.

---

# Decision Criteria

| Criteria | Importance | Notes |
|----------|------------|-------|
| Doesn't rely on unverified assumptions (MIND history order) | Critical | ADR-002 flagged this explicitly; violating it silently would be a correctness risk in exactly the kind of way this project's evidence hierarchy is meant to prevent |
| Recall, measured | Critical | A query construction choice that can't beat a random baseline is not a baseline, it's a bug |
| Single code path across MIND/EB-NeRD | High | Per ADR-002 |
| Simplicity | Medium | This is Phase 3's baseline; more sophisticated query construction (recency weighting, entity boosting) is legitimate future work, not a Phase 3 requirement |
| Cold-start definition must be evidence-grounded, not arbitrary | High | The assignment specifically asks for a warm/cold comparison; an arbitrary threshold would make that comparison undefensible |

---

# Design Space Exploration

## Option A — Unweighted concatenation, full history

### How it Works

Concatenate `title + " " + abstract` for every article in a user's history, in whatever order the history list provides, tokenize the whole thing as one bag of words (a multiset — repeated words across articles are not deduplicated).

### Optimizes For

Simplicity, no dependency on history ordering being correct (doesn't use position at all), single code path.

### Sacrifices

No recency signal — a click from 21 days ago (EB-NeRD's history window) weighs exactly as much as a click from yesterday. Users with very long histories (EB-NeRD, up to 1,459 articles) produce very long queries.

### Assumptions

That term-count mass from a longer history is still net-informative rather than net-noise. **This assumption failed empirically for EB-NeRD when combined with un-filtered tokenization** — see Benchmark Results. It held once stopwords were removed.

### Complexity

Low.

### Maturity

★★★★☆ — "concatenate the profile" is standard practice for content-based bag-of-words user representations; recency/decay weighting is a common refinement, not a prerequisite.

---

## Option B — Recency-weighted history (most-recent-N or exponential decay)

### How it Works

Either truncate to the N most recent history entries, or weight each entry's contribution by recency (e.g. exponential decay), using the history list's order as a proxy for time.

### Optimizes For

Matches the intuition from PROJECT_STATE's Day-1 working hypothesis ("recent clicks are more predictive") and would in principle better model a user's *current* interest rather than their entire multi-week history.

### Sacrifices

**Directly violates ADR-002's own stated monitoring strategy**: "do not build a MIND recency feature that would break silently if [MIND's presumed-chronological history order] is wrong." MIND provides no per-click timestamps to verify order against — this project has never independently confirmed it. EB-NeRD's order *is* verified (`click_times` exists), so this option would also require a dataset-specific branch, undermining ADR-002's single-code-path goal for query construction specifically.

### Assumptions

That MIND's history list order is chronological. Unverified, flagged as a real risk in ADR-002.

### Complexity

Medium — two code paths (order-trusted vs. order-verified), plus decay-parameter tuning.

### Maturity

★★★☆☆ — recency weighting is common in production recommenders, but those systems have verified timestamps to weight by; applying it against an unverified ordering assumption is not standard practice, it's a shortcut.

---

## Option C — Most-recent-K history only (K fixed, e.g. last 20)

### How it Works

Same as Option B without decay — just truncate to the last K entries by list position.

### Optimizes For

Bounds query length (relevant for EB-NeRD's long histories) without needing a decay function.

### Sacrifices

Same core problem as Option B: still depends on MIND's list order being chronological ("most recent" is meaningless if the order isn't temporal), and still needs the ADR-002-flagged unverified assumption to hold.

### Assumptions

Same as Option B.

### Complexity

Low-Medium.

### Maturity

★★★☆☆

---

# Comparison Summary

| Option | Avoids unverified MIND assumption | Single code path | Complexity | Measured to work | Decision |
|---------|----------|------------|------------------|-----------|----------|
| A — Unweighted, full history | Yes | Yes | Low | Yes (after tokenization fix) | ✅ |
| B — Recency-weighted | No | No | Medium | Not tested (rejected before benchmarking) | |
| C — Most-recent-K | No | No | Low-Medium | Not tested (rejected before benchmarking) | |

---

# Final Decision

## Chosen Option

**Option A** — unweighted concatenation of the full history's `title + abstract` text, tokenized as a multiset (repeats kept, not deduplicated), with a **cold-start threshold of history length < 5** for the warm/cold cohort split. Tokenization filters a combined Danish+English stopword list (see Benchmark Results — this was not part of the original decision and was added after measurement showed it necessary).

### Reason

Options B and C both require trusting an assumption ADR-002 explicitly flagged as unverified and told future work not to build on silently. Option A sidesteps that risk entirely by not using history order at all. The remaining question — whether "just concatenate everything" is good enough to be a real baseline rather than noise — was answered empirically, not assumed (see Benchmark Results): it wasn't good enough until stopwords were removed, at which point it clearly and measurably worked.

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

- Structurally cannot violate ADR-002's flagged MIND-history-order risk, because it never reads history order.
- One code path for both datasets — no dataset-specific branch in query construction, consistent with ADR-002's design goal.
- Cold-start threshold (`< 5`) is not an arbitrary round number — it's the EB-NeRD paper's own active-user filter (5–1,000 clicks), already cited in this project's PROJECT_STATE. Applying literature's own threshold, rather than inventing a new one, is stronger evidence than a project-local heuristic.
- The cold/warm split, applied to real data, produced a real and reportable finding rather than a degenerate one: **EB-NeRD-validation has zero users below the threshold (min history = 5, by construction of the active-user-filtered demo bundle)**, while MIND-dev has 8,014/50,000 (16.0%) cold users, including 1,407 users with *literally zero* history. This means MIND is the only dataset in this project where a genuine warm/cold BM25 comparison is possible — documented as a finding, not hidden as a limitation.
- Zero-history users are a clean, structural proof point: they produce an empty query (`build_user_query` returns `[]`), which `retrieve_top_k` short-circuits to `[]` rather than scoring — verified directly: of MIND-dev's 1,407 zero-history users, their 3,415 positive impressions all necessarily miss (recall = 0 for that subgroup by construction, not by measurement). This is real, load-bearing evidence for the "BM25 cannot address true cold-start" argument that motivates Phase 4 (semantic retrieval).

## Why alternatives were rejected

### Option B (recency-weighted)

Rejected primarily because it would silently depend on an assumption this project has already flagged as unverified and explicitly warned against building on (ADR-002). Rejected before benchmarking — this is a case where the assignment's evidence hierarchy (assumptions/risks should be documented and not built on silently) outweighs "maybe it would perform better," which was never tested because the assumption risk made it not worth testing yet.

### Option C (most-recent-K)

Same rejection as Option B — depends on the same unverified MIND ordering assumption, just without a decay function. Rejected for the same reason.

---

# Decision Confidence

**Current Confidence**

Medium-High

### Why

- The core "don't trust MIND's history order" reasoning is high confidence — it's a direct consequence of ADR-002's own documented, unresolved risk.
- The tokenization/stopword-removal component of this decision is now benchmark-verified (see below), not just argued.
- Medium, not High, because: (a) EB-NeRD's absolute recall numbers (4.02% @ k=200) are still low in absolute terms — better than random (1.7%) but this is a weak baseline, consistent with the assignment's own framing of BM25's limitations, not a sign this implementation is optimal; (b) the stopword list is a compact hand-built list, not a validated NLP resource — see Known Limitations.

### What Would Increase Confidence

- Comparing against semantic retrieval (Phase 4) on the same warm/cold split to confirm the expected complementarity pattern (PROJECT_STATE's working hypothesis).
- Testing recency weighting on EB-NeRD only (where history order *is* verified) as a dataset-specific enhancement, to see whether it improves on this baseline where the assumption risk doesn't apply.

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (recall@50/100/200, warm/cold comparison)
- [x] Experimental Benchmark (see below — this ADR was revised mid-flight based on real measurement)
- [ ] Official Documentation
- [x] Research Paper (EB-NeRD's active-user filter threshold, Kruse et al. 2024)
- [ ] Industry Practice
- [ ] Community Consensus
- [x] Engineering Inference (ADR-002's flagged MIND-ordering risk)

---

## Empirical Evidence

**Cold-start threshold, applied to real data:**

| Split | Users | Cold (`<5`) | Warm (`>=5`) | Zero-history |
|---|---|---|---|---|
| MIND dev | 50,000 | 8,014 (16.0%) | 41,986 | 1,407 |
| EB-NeRD validation | 1,562 | **0** | 1,562 | 0 |

**Tokenization variant comparison** (1,500-impression sample, EB-NeRD validation, recall@K measured directly, not estimated):

| Variant | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| Raw multiset, no stopword removal | 0.87% | 1.33% | 2.53% |
| Deduplicated (unique tokens), no stopword removal | 0.13% | 0.47% | 1.80% |
| **Multiset + stopword removal (chosen)** | **1.13%** | **2.33%** | **4.07%** |
| Deduplicated + stopword removal | 0.13% | 0.60% | 2.00% |

Random baseline for reference: 200/11,777 ≈ 1.70%.

Root cause (traced directly, not inferred): EB-NeRD's long per-user histories (up to 1,459 articles) produce queries with thousands of tokens; the ten most frequent tokens in a real sampled query were pure Danish function words (`i`, `er`, `og`, `til`, `at`, `en`, `på`, `det`, `med`, `ikke` — "in/is/and/to/that/a/on/it/with/not"). Their count-weighted mass, even after BM25's own idf downweighting, was large enough to bury genuinely topical terms — a real sampled positive impression's clicked article ranked 9,855th of 11,777 pre-fix. Deduplication was also tested as a fix and made things *worse* (0.13% @ k=50), confirming that repeated terms carry real signal (recurring topics in a user's history) and the problem was specifically high-document-frequency function words, not repetition itself.

Full-dataset production runs (`experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/`) confirm the same pattern at scale — see ADR-006's Benchmark Results for the complete recall table.

## Theoretical Evidence

- Kruse et al. (2024), EB-NeRD — 5–1,000 click active-user filter, used directly as this ADR's cold-start threshold.

## Accepted Trade-offs

- The stopword list (`src/retrieval/tokenize.py::STOPWORDS`) is a compact, hand-built Danish+English list, not a validated linguistic resource (e.g. NLTK's shipped corpora) — accepted because it's demonstrably sufficient (measured 60%+ relative recall improvement) and avoids adding a new dependency/download for a training-corpus-style resource. Flagged as a revisit trigger below.
- EB-NeRD's cold-start comparison is structurally unavailable in this project (demo bundle's active-user filter guarantees `history >= 5`) — accepted because this is a property of the dataset bundle, not something query construction can fix; documented rather than worked around.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| MIND's history list order is not used anywhere in retrieval | If a future change (e.g. recency weighting) reintroduces order-dependence, it inherits ADR-002's unverified-order risk silently | Any future PR touching `query.py` that reads `article_ids` positionally must re-justify against ADR-002 |
| The hand-built stopword list is representative enough of both languages' function words | An incomplete list would under-filter and partially reproduce the pre-fix degradation | Re-run the tokenization variant comparison (table above) if query construction changes; the random-baseline comparison is a cheap sanity check to rerun any time |
| History-length-based cold/warm split is a reasonable proxy for "how much signal BM25 has to work with" | This is the entire basis for the warm/cold comparison the assignment requires | Revisit if Phase 4's semantic retrieval shows a meaningfully different warm/cold pattern under the same threshold |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [x] Performance regression — n/a here (query construction is not the bottleneck; see ADR-006 for the retrieval-scoring performance story)
- [ ] Latency exceeds threshold
- [ ] Memory becomes bottleneck

---

## Research Triggers

Revisit if:

- [x] A validated stopword resource (e.g. a proper Danish NLP library) becomes available and the hand-built list is shown to be insufficient
- [ ] Significant new research appears

---

## Project / Assignment Triggers

Revisit if:

- [x] Phase 4 (semantic retrieval) needs a comparable query representation and the unweighted-concatenation baseline doesn't transfer well
- [x] ~~EB-NeRD `small`/`large` bundles (once downloaded) turn out to include genuinely cold-start users — the "EB-NeRD has no cold users" finding is specific to `ebnerd_demo` and should be re-verified before treating it as a general dataset property~~ **Resolved (2026-08-10, see ADR-002 addendum): `ebnerd_small` validation also has zero users below the threshold (min history = 5) — the finding generalizes beyond `demo`, it's structural to the active-user-filtered bundles; `ebnerd_large` remains unverified.**

---

**Estimated Cost to Change**

Easy — query construction is isolated in `src/retrieval/query.py` and `tokenize.py`; changing it doesn't touch the index-building or retrieval-scoring code.

---

# Engineering Impact

## Affected Components

- Retrieval
- Evaluation

---

## Affected Files

- `src/retrieval/query.py`
- `src/retrieval/tokenize.py`
- `scripts/run_bm25_experiment.py` (cold-threshold constant)
- `tests/unit/test_query_construction.py`

---

## Expected Refactoring

None — first implementation.

---

## Breaking Changes

No.

---

## Required Tests

- Unit: tokenization correctness including stopword removal (`tests/unit/test_query_construction.py`).
- Unit: empty history / missing article_id / empty text all degrade to `[]`, not an exception.
- Integration: a real MIND-dev zero-history user retrieves nothing (`tests/integration/test_retrieval_pipeline.py::test_cold_start_user_with_empty_history_retrieves_nothing`).

---

## Expected Benchmarks

- Recall@50/100/200, overall and by warm/cold cohort, both datasets.

---

## Documentation Updates

- [x] Architecture (ARCHITECTURE.md's Retrieval component)
- [x] Project State
- [ ] Knowledge Base

---

# Benchmark Results

## Baseline

Pre-fix: raw multiset, no stopword removal.

## Candidate

Post-fix: multiset + stopword removal (chosen).

## Benchmark Environment

| Item | Value |
|------|-------|
| Dataset | EB-NeRD validation (tokenization variant test); MIND-small dev + EB-NeRD validation (full runs) |
| Hardware | Local development machine (macOS, Python 3.14) |
| Software Version | rank-bm25 0.2.2 |
| Random Seed | 7 (tokenization variant sample); 0 (bootstrap CIs) |

---

## Experiment

`experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/`

---

## Results

Tokenization variant comparison (1,500-impression EB-NeRD sample) — see Empirical Evidence table above.

Full production runs — see ADR-006's Benchmark Results for the complete recall@K table (this ADR's tokenization decision and ADR-006's BM25-variant decision jointly determine those numbers; splitting the table across both would duplicate it).

---

## Interpretation

Stopword removal is not a nice-to-have here — the un-filtered baseline measured *below* the random baseline at k=50 and k=100 on EB-NeRD, meaning the "baseline" would have been actively misleading if shipped as-is (it would look like BM25 performs worse than picking articles at random, which is not what's actually true; it's an artifact of query construction, not a property of BM25 itself). This is exactly the kind of finding CLAUDE.md's benchmarking philosophy is meant to surface before it ships silently.

---

## Comparison to Alternatives

Deduplicating the query (Option considered, not formally a top-level Option A/B/C but tested as part of the tokenization fix) was directly measured to be worse than keeping the multiset — repeated terms in a user's history are informative, not noise, once function words are removed from contention.

---

# Related Decisions

## Influenced By

- ADR-002 (Unified Data Schema) — the MIND-history-order risk this ADR is designed around
- ADR-001 (Temporal Split Strategy) — determines which split's history/impressions this ADR's evaluation runs against

## Influences

- ADR-006 (BM25 Variant) — consumes this ADR's tokenized query/corpus text
- Future semantic retrieval ADR — will need to decide whether to reuse this query representation or build a different one

---

# References

## Internal

- ARCHITECTURE.md — System Data Flow (Query Construction stage)
- ADR-002-unified-data-schema.md — MIND history-order risk
- `experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/`

## External

- Kruse et al. (2024). *EB-NeRD: A Large-Scale Dataset for News Recommendation.* arXiv:2410.03432 / ACM RecSys Challenge 2024. (Active-user filter: 5–1,000 clicks.)

---

# Decision History

| Date | Event |
|------|-------|
| 2026-08-10 | Exploration started; Option A chosen over B/C on assumption-risk grounds before any benchmarking |
| 2026-08-10 | Full EB-NeRD run showed recall below random baseline; root-caused to stopword-driven query mass, benchmarked 4 tokenization variants |
| 2026-08-10 | Stopword removal adopted after measurement; decision finalized |

---

# Notes

This ADR was written after implementation and benchmarking, not strictly before, because the tokenization defect was only discoverable by running the real pipeline against real data — the failure mode (query mass dominated by function words) doesn't show up in small hand-built unit-test fixtures or in a purely theoretical review of the approach. Per CLAUDE.md's decision-reversal guidance, the original plan's tokenizer design ("lowercase + `\\w+`, no stemming, simplicity as the baseline") is preserved above as Option A's initial form, with the stopword-removal correction documented as a benchmark-driven amendment rather than silently folded in as if it were the plan all along.

---

# Addendum — Recency-Weighting Measured for MIND's Embedding Query (Candidate C, 2026-08-21)

**Status:** Does not reverse this ADR's Option A decision for *BM25 query
construction*, which this addendum doesn't touch. What it adds: Option B
(recency-weighted history) was originally rejected here "before
benchmarking" (Decision History, 2026-08-10) purely on the unverified-
history-order risk, for BM25 specifically. Per PROJECT_STATE.md's "before
committing to a real MIND second-submission attempt" objective and
CLAUDE.md's decision-reversal guidance, that untested rejection was
revisited — not for BM25, but for MIND's *embedding-based* user
representation (Q3's mean-pooled history query, the method actually
deployed per ADR-008), which is the version of "does recency weighting
help" that bears on a real submission decision today.

## What was checked

New `build_user_embedding_query_recency` (`src/retrieval/embed.py`):
same contract as `build_user_embedding_query`, but weights each
*resolvable* history vector by `decay ** i` (i = position counted back
from the most recent resolvable click), `decay=0.9`, untuned — one
reasonable starting point, not searched, same "test whether it helps at
all" framing as the leaky-feature-ablation/hybrid screens' untuned blends.
Explicitly inherits, and does not resolve, this ADR's own flagged risk:
MIND's `article_ids` history order is assumed oldest-to-most-recent per
MIND's documentation, still never independently verified against real
timestamps. (`scripts/run_recency_history_experiment.py`)

## Results

Baseline: MINDsmall-dev, unweighted mean-pool embedding query, AUC 0.6340
(95% CI 0.6319-0.6361, reconfirmed bit-identical this session).

| | Overall AUC | 95% CI | vs. baseline |
|---|---|---|---|
| Recency-weighted (decay=0.9) | 0.6265 | 0.6243-0.6286 | **CI-clear loss** (upper bound 0.6286 < baseline lower bound 0.6319) |

Cohort breakdown: warm also loses CI-clear (0.6351, CI 0.6328-0.6374 vs.
baseline warm 0.6439, CI 0.6416-0.6461); cold is a statistical tie (0.5740,
CI 0.5684-0.5794 vs. baseline cold 0.5737, CI 0.5682-0.5792 — heavily
overlapping), consistent with cold users' short (`<5`) histories leaving
little room for position-based weighting to change anything. Full table in
`experiments/candidate_c_recency_history_mind_small_2026-08-21/`.

## Interpretation

**A real, CI-clear loss, not a wash — recency weighting measurably hurts
here, including for warm users, where it was expected to help most if the
"recent clicks are more predictive" intuition held.** This is evidence
against that intuition for MIND specifically, not just an absence of
evidence for it. Plausible (not isolated by a controlled follow-up):
downweighting older history discards real topical signal that mean-pooling
was already using productively, and MIND's within-user topical variety
(vs. drift toward one recent topic) may be genuinely present enough that
"more history mass" is better than "more recent history mass" for this
dataset — but this is inference from the result's shape, not a separately
tested claim. Separately, and regardless of which direction the metric
moved: the original risk this ADR flagged (unverified history-order
assumption) is now something a real, measured effect depends on for its
validity, not just a theoretical concern — if MIND's list order isn't
actually chronological, this result doesn't mean "recency weighting
doesn't help," it means "weighting by this particular unverified ordering
doesn't help," a narrower and less transferable claim.

## Conditions for Revisiting

- The underlying MIND history-order-is-chronological assumption remains
  unverified — this addendum measures a consequence of trusting it, it
  does not itself verify it. Confirming MIND's real click-order (if such
  data ever becomes available) would meaningfully change how much weight
  to put on this result.
- A decay-parameter sweep was deliberately not done (untuned, single
  starting point, per this ADR's own risk-first stance) — not repeating a
  parameter search here that Option B was already rejected for on
  assumption-risk grounds, now that assumption-risk grounds have also
  produced a real negative result at the one point tested.

## Related

- `src/retrieval/embed.py::build_user_embedding_query_recency`,
  `tests/unit/test_embed.py`
- `scripts/run_recency_history_experiment.py`
- `experiments/candidate_c_recency_history_mind_small_2026-08-21/`
- `knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`
- See ADR-008's own 2026-08-21 addendum for Candidates A/B (MIND entity
  embeddings, BM25+embedding hybrid), tested in the same session against
  the same baseline.
