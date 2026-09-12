# ADR-007 — Q4 Ranking Evaluation Harness Design

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

Q4 requires AUC, MRR, nDCG@5, nDCG@10 (Q4.1), intra-list diversity/novelty/
coverage (Q4.2), warm/cold slicing (Q4.3), and bootstrap 95% CI (Q4.4) —
computed over the candidates *already listed in each impression*, not a
top-K retrieved from the whole corpus (that's recall@K, already built and
governed by ADR-005/ADR-006). Building this harness forces four genuinely
open sub-decisions that aren't independent of each other, so — following
ADR-005's precedent of bundling related sub-decisions into one ADR rather
than four thin ones — this ADR covers all four:

1. How are ties broken when ranking a fixed candidate list (unlike
   recall@K's top-K-of-everything, a full ranking over a small list must
   resolve every position, including ties beyond any cutoff)?
2. What K should diversity/novelty be computed over, given the assignment
   doesn't specify one?
3. How is novelty (and its underlying popularity signal) defined without
   violating Q9's anti-gaming requirement?
4. Should catalog coverage get a bootstrap CI like every other metric, for
   consistency?

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

- Evaluation (`src/evaluation/ranking_metrics.py`, `src/evaluation/bootstrap.py`)
- Retrieval (`src/retrieval/score.py` — the generic scoring interface this
  harness scores against)

---

# Context

Grounding data pulled directly from the processed parquet files before
designing (per CLAUDE.md's evidence hierarchy — not assumed): every
impression in both `MINDsmall-dev` and `ebnerd_demo-validation` currently
has ≥1 clicked and ≥1 unclicked candidate (0 degenerate all-clicked or
all-unclicked impressions found in either dataset), multi-click impressions
are real (up to 24 clicks in one MIND impression), and `category` is 100%
non-null in both datasets' `articles` tables (17 unique in MIND, 25 in
EB-NeRD).

Critically: a true cold-start user (`article_ids=[]` → empty query, per
ADR-005) produces an **all-zero score vector** across every candidate in
every one of their impressions (`src/retrieval/score.py::score_all`'s
documented behavior). recall@K never had to resolve this — ties beyond
position K simply don't matter for a hit/miss boolean. A full ranking over
an impression's small candidate list (MIND-dev median 23/impression,
EB-NeRD median 9–12) does have to resolve it: MRR and nDCG are undefined
without *some* order, and this happens on every cold-user impression, not
as a rare edge case.

---

# Decision Criteria

| Criteria | Importance | Notes |
|----------|------------|-------|
| No silent bias into the cold-user cohort | Critical | Q4.3 exists specifically to characterize cold-start behavior honestly — a biased tie-break would corrupt exactly the result this project cares most about |
| Reproducibility (CLAUDE.md: same code + data + seed = same result) | Critical | Any randomized component must be seeded deterministically |
| No serving-time information leakage (Q9 anti-gaming) | Critical | Novelty/coverage must not reward recommending items whose popularity wasn't knowable at serving time |
| Statistical honesty of reported CIs | High | A CI that looks precise but is a resampling artifact is worse than no CI |
| Consistency with existing metrics' pattern (recall@K's bootstrap) | Medium | Extend, don't duplicate, per CLAUDE.md |
| Anchoring arbitrary constants (e.g. a K value) to something already justified | Medium | Avoid inventing an unexplained second cutoff |

---

# Design Space Exploration

## Sub-decision 1 — Tie-breaking rule

### Option A — Leave candidates in the impression's given order

**How it works:** Rank ties by whatever position the candidate already
occupies in the `impressions` table's rows for that impression.

**Optimizes for:** Zero extra code, "just don't touch the order."

**Sacrifices:** This looks deterministic but is silently non-deterministic
*in effect* — it inherits whatever order MIND's fused `ArticleID-label`
pairs or EB-NeRD's `article_ids_inview` list happen to arrive in upstream,
which this project has never verified is unbiased with respect to click
likelihood. This is the same category of risk ADR-002/ADR-005 already
flagged once for MIND's history-list order (presumed chronological, never
independently verified) — reusing an unverified upstream order a second
time, silently, is the exact anti-pattern this project's evidence
hierarchy exists to catch. If that order has *any* correlation with click
propensity, cold-user MRR/nDCG would look like a real finding but actually
be a data artifact.

**Maturity:** N/A — not a real convention, just "do nothing."

## Option B — Sort by `article_id`

**How it works:** Break ties lexically by the (dataset-prefixed)
`article_id` string.

**Optimizes for:** Fully deterministic, trivial to implement.

**Sacrifices:** Introduces its own systematic bias risk: if `article_id`
correlates with anything relevant (ingestion order, publication recency —
plausible for both datasets' ID schemes), a fixed lexical tie-break would
systematically favor or disfavor the same articles across *every*
cold-user impression. Critically, this bias would be **invisible in the
aggregate metric** — it wouldn't show up as noise, it would show up as a
consistent, undetected offset, and it disproportionately affects exactly
the cold-user cohort Q4.3 is trying to characterize honestly.

**Maturity:** Occasionally seen as a lazy default, not a documented IR
evaluation convention.

## Option C — Deterministic per-impression-seeded pseudo-random permutation (chosen)

**How it works:** For each impression, derive a seed from
`hashlib.sha256(f"{seed}:{impression_id}")` (not Python's built-in
`hash()`, which is salted per-process via `PYTHONHASHSEED` by default and
would silently break cross-run/cross-machine reproducibility), use it to
seed a `numpy` RNG, draw one random tiebreak value per candidate, and
`np.lexsort` on `(tiebreak, -score)` so score is the primary sort key and
the random draw only resolves exact ties.

**Optimizes for:** Zero expected bias toward any candidate — under a total
tie (the cold-user case), a random permutation is exactly the "no
information" result: MRR and nDCG under it are unbiased estimators of what
"no signal" should look like, rather than whatever a fixed rule would
smuggle in. This is also the standard IR-evaluation convention for tied
relevance judgments (`trec_eval`'s own convention). Full reproducibility is
preserved because the randomness is deterministically seeded from
`(seed, impression_id)`, mirroring the same `seed=0` pattern
`metrics.py::recall_at_k` already uses.

**Sacrifices:** Slightly more code than Option A/B; the resulting per-
impression order is not literally reconstructable by eye from the raw
data (has to be recomputed), though it's fully reproducible by rerunning
the code.

**Maturity:** ★★★★☆ — standard practice for resolving tied relevance
judgments in IR evaluation.

---

## Sub-decision 2 — K for diversity/novelty

### Option A — K=10 (chosen)

Anchored to nDCG@10's cutoff — already a required Q4.1 metric, so
diversity/novelty are reported "at the same operating point" as one of the
accuracy metrics, letting the design note say something coherent like
"at top-10, BM25 achieves nDCG@10=X with diversity=Y" instead of two
unrelated cutoffs each needing separate justification.

### Option B — K=5

Matches nDCG@5 too, and is safer against EB-NeRD's minimum candidate count
(5) — no list would ever need truncation below the requested K. Rejected
as primary because 10 is closer to a realistic "above the fold"
recommendation slate size, and anchoring to nDCG@10 specifically (rather
than nDCG@5) avoids the harness needing two separately-justified constants
when one nDCG cutoff already provides a natural anchor.

### Resolution

K=10, with `k_eff = min(10, n_candidates)` for impressions shorter than 10
(EB-NeRD's minimum is 5) — averaged over fewer items in that case, noted
rather than hidden.

---

## Sub-decision 3 — Novelty/popularity source

### Option A — Popularity from the evaluation split itself

**Sacrifices:** Directly violates Q9's anti-gaming requirement — using
validation-set click popularity to score novelty means the metric rewards
recommending items whose popularity wasn't actually knowable at serving
time (the model would effectively be graded using information from the
future relative to when it served the recommendation).

### Option B — Popularity from the train split only (chosen)

`build_train_popularity` counts train-split clicks per article, Laplace
(add-`alpha`) smoothed. Smoothing is required, not cosmetic: ADR-002
measured 32.9% (MIND) / 45.7% (EB-NeRD) of validation-candidate articles
never appear in train at all — unsmoothed popularity would give these
`p=0`, and `novelty = -log2(0) = inf` would poison any average that
includes them. Smoothing gives a small floor probability instead, which
correctly reads as "very novel," not "broken."

---

## Sub-decision 4 — Coverage's bootstrap CI

### Option A — Bootstrap CI, same as every other metric

**Sacrifices:** Coverage is a **set-union** statistic (the union of every
impression's top-K recommendations), not a per-user mean/ratio. Under
with-replacement bootstrap resampling, resampling the same user twice adds
*nothing new* to the union (unlike a mean, where a repeat correctly
reweights the average) — this systematically biases a naive bootstrap CI
toward the ~63.2%-of-panel "effective sample size" with-replacement
resampling produces, understating the true CI width for reasons that are
an artifact of the resampling mechanics, not genuine statistical
uncertainty. A CI computed this way would *look* rigorous while being
actively misleading — exactly the failure mode CLAUDE.md's benchmarking
philosophy exists to prevent ("never report an isolated number" cuts both
ways: don't report a *wrong* interval just to have one).

### Option B — Point estimate only, explicitly documented (chosen)

Report coverage as a single point estimate over the full run (and
separately per warm/cold cohort), with this rationale stated explicitly in
results output and here — not silently omitted, and not silently forced
through machinery that would misrepresent it.

---

# Comparison Summary

| Sub-decision | Chosen | Why |
|---|---|---|
| Tie-break | Seeded pseudo-random per impression | Zero expected bias, matches IR convention, fully reproducible |
| Diversity/novelty K | 10 | Anchored to the already-required nDCG@10 cutoff |
| Novelty popularity source | Train split only, Laplace-smoothed | Q9 anti-gaming requirement; smoothing prevents `-log2(0)=inf` |
| Coverage CI | None (point estimate, documented) | Bootstrap-over-union is mechanically biased, not just noisy |

---

# Final Decision

## Chosen Option

As summarized above — implemented in `src/evaluation/ranking_metrics.py`
(`rank_candidates`, `ndcg_at_k`, `mrr`, `safe_auc`, `intra_list_diversity`,
`build_train_popularity`, `novelty`, `coverage`, `ranking_metric_ci`) and
`scripts/run_ranking_eval.py` (the driver, `K_DIVERSITY_NOVELTY = 10`).

### Decision Date

2026-08-10

### Decision Owner

**Primary Engineer**

- sukhraj01

### Contributors

- Claude Code (design exploration via a dedicated Plan-agent pass grounded
  in real data statistics, implementation, benchmarking, ADR drafting)

### Reviewer *(Optional)*

- Pending

---

# Rationale

## Why this option is best

- Every sub-decision here was forced by a real, measured property of the
  data (cold-user total ties are structural, not rare; the train/
  validation article-set gap is large enough to break unsmoothed novelty;
  coverage's set-union nature is a mathematical fact about bootstrap
  resampling, not a judgment call) — none of the four choices is
  arbitrary, each has a concrete failure mode it avoids.
- The tie-break and novelty decisions both specifically protect the
  cold-user cohort's honesty, which is the same cohort ADR-005 already
  found BM25 structurally cannot serve (zero-history users retrieve
  nothing) — this harness is partly designed to measure *how* well or
  badly, not to accidentally paper over that with an unexamined
  convenience choice.

## Why alternatives were rejected

See each sub-decision's "Sacrifices" section above — each rejected option
was rejected for a specific, stated failure mode (silent bias, information
leakage, or a misleading CI), not preference.

---

# Decision Confidence

**Current Confidence**

Medium-High

### Why

- The tie-break and novelty-leakage reasoning follow directly from
  properties already measured in this project (ADR-002's train/validation
  article-set gap, ADR-005's cold-user zero-query behavior) — not
  speculative.
- The coverage-CI reasoning is a structural fact about bootstrap
  resampling of set-union statistics, not an empirical finding that could
  be wrong.
- Medium, not High: the K=10 choice, while anchored to nDCG@10, is still a
  judgment call the assignment doesn't specify — a different K would
  change diversity/novelty's absolute values (though not the harness's
  correctness).

### What Would Increase Confidence

- Running Phase 4 (semantic retrieval) through the same harness and
  confirming the tie-break/novelty design holds up under a scorer whose
  cold-user behavior differs from BM25's (an embedding-based cold-user
  query is a mean of zero vectors, not necessarily all-zero-score against
  every candidate the way BM25's is — worth re-examining once that
  scorer exists).

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (Q4.1–Q4.4, Q9 anti-gaming)
- [x] Experimental Benchmark (real impression-candidate-count, click-rate,
      and train/validation article-overlap statistics pulled before
      designing — see Context)
- [ ] Official Documentation
- [x] Industry Practice (`trec_eval`'s tied-relevance convention)
- [ ] Community Consensus
- [x] Engineering Inference (bootstrap-over-union bias argument)

---

## Empirical Evidence

- 0 degenerate (all-clicked or all-unclicked) impressions found in either
  `MINDsmall-dev` or `ebnerd_demo-validation` (checked directly against
  the processed parquet files) — `safe_auc`'s NaN-skip path is defensive,
  not currently load-bearing, but necessary since `ebnerd_small` was
  unverified until ADR-002's addendum ran.
- ADR-002: 32.9% (MIND) / 45.7% (EB-NeRD) of validation-candidate articles
  never appear in train — the concrete number `build_train_popularity`'s
  smoothing exists to handle.
- Full harness benchmark results: see Benchmark Results below.

## Theoretical Evidence

- Bootstrap-of-a-union bias: with-replacement resampling of `n` units
  samples on average `1 - (1 - 1/n)^n → 1 - 1/e ≈ 63.2%` of distinct units
  per replicate — a union statistic under this resampling scheme is
  structurally undercounted relative to the true population, independent
  of any assumption about the underlying data.

## Accepted Trade-offs

- The tie-break makes exact per-impression rankings non-deterministic to
  eyeball from raw data (must rerun code to see them) — accepted because
  full reproducibility (same code+data+seed → same result) is preserved,
  and the alternative (a fixed but potentially biased order) is worse in
  a way that wouldn't be visible in the aggregate metrics.
- Coverage has no CI, which makes it look "less rigorous" next to the
  other five metrics at a glance — accepted because a wrong CI is worse
  than no CI, and the rationale is stated explicitly rather than hidden.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|------------|----------------|---------------------|
| A cold user's BM25 scores are uniformly zero across all candidates (total tie) | This is the entire reason the tie-break is mandatory, not optional | Re-verify once Phase 4's embedding scorer exists — its cold-user behavior may not produce total ties the same way |
| `category` remains 100% non-null for any future dataset this harness runs against | `intra_list_diversity` has no null-handling branch | If a future bundle has null categories, this would need a fallback, not silently divide incorrectly |
| K=10 is a reasonable, if not uniquely correct, operating point for diversity/novelty | Absolute diversity/novelty values are K-dependent | Revisit if the design note needs a different K for comparability with a specific baseline |

---

# Conditions for Revisiting

## Technical Triggers

Revisit if:

- [ ] Performance regression
- [ ] Latency exceeds threshold
- [ ] Memory becomes bottleneck

---

## Research Triggers

Revisit if:

- [x] Phase 4's embedding scorer produces a materially different cold-user
      score distribution (not a total tie) — re-examine whether the
      tie-break is still load-bearing the same way
- [ ] Significant new research appears

---

## Project / Assignment Triggers

Revisit if:

- [x] A degenerate (all-clicked/all-unclicked) impression is found in a
      future dataset — confirms `safe_auc`'s defensive NaN-skip path is
      real, not just theoretical
- [ ] Assignment requirements change

---

**Estimated Cost to Change**

Medium — the tie-break and K choice are isolated in `ranking_metrics.py`
and the driver's `K_DIVERSITY_NOVELTY` constant; the novelty-source
decision would require re-running `build_train_popularity` against a
different table if changed.

---

# Engineering Impact

## Affected Components

- Evaluation
- Retrieval (the `Scorer` interface this harness scores against)

---

## Addendum (2026-09-12) — A2 Q5: head-vs-tail slicing, and the slice that reversed a result

A2 Q5 requires the full metric suite with bootstrap CIs **and at least two slices:
cold-start vs warm, head vs tail**. This harness already had warm/cold (Q4.3). Head/tail is
new, and it is implemented in `scripts/a2_evaluate_scores.py` rather than
`run_ranking_eval.py` because A2's arms are scored offline and evaluated from dumped
per-impression scores.

## Definition (a judgement call, so stated plainly)

- **head** = the clicked article's TRAIN-split click count is in the top **20%** of articles
  clicked in train at all; **tail** = everything else. Train-only, reusing the same
  `build_train_popularity` output novelty@10 uses, so no dev clicks leak into the slicing
  (Q9). Articles never clicked in train share the smoothed floor and therefore land in
  **tail** by construction — unseen-in-train is the extreme tail, which is the intended
  reading for news.
- An impression is assigned by its **first clicked** candidate, so each impression falls in
  exactly one bucket. Impressions with no positive are excluded (already NaN for AUC).
- **warm/cold** reuses ADR-005's threshold unchanged (history < 5).
- Sliced paired tests require ≥50 impressions in the slice.
- Four unit tests pin the cut, the unseen-in-train case, `head_frac` widening, and the
  prefixed-id warm/cold join.

## The finding: EB-NeRD's freshness gain reverses on head articles

Paired treatment − control, EB-NeRD (ADR-015's A/B):

| Slice | n impressions | control AUC | paired Δ AUC | 95% CI | |
|---|---:|---:|---:|---|---|
| **head** | 9,541 | 0.6222 | **−0.0217** | −0.0258, −0.0178 | significant **loss** |
| **tail** | 235,106 | 0.5588 | **+0.0076** | +0.0064, +0.0087 | significant gain |
| overall | 244,647 | 0.5613 | +0.0064 | +0.0053, +0.0075 | significant gain |

**The +0.0064 headline is a tail gain diluted by a real head regression.** Reporting only
the aggregate would have hidden a CI-clear loss on 3.9% of impressions. The mechanism is
plausible and worth stating: a freshness-weighted ranker should help least where an article
is popular enough to be clicked regardless of age. This is precisely the case Q5's slicing
requirement exists to expose.

## MIND control, sliced (job 2694501)

| Slice | n | AUC | 95% CI | MRR | nDCG@10 |
|---|---:|---:|---|---:|---:|
| head | 143,164 | 0.7411 | 0.7397–0.7426 | 0.4608 | 0.4976 |
| tail | 233,307 | 0.6474 | 0.6463–0.6486 | 0.3309 | 0.3852 |
| warm | 323,747 | 0.6932 | 0.6922–0.6941 | 0.3861 | 0.4317 |
| cold | 52,724 | 0.6209 | 0.6183–0.6236 | 0.3446 | 0.4050 |

- **Head is easier on both datasets** (MIND 0.7411 vs 0.6474; EB-NeRD 0.6222 vs 0.5588).
- **The reproduced baseline does not close the cold-start gap.** Its warm−cold spread is
  0.0723, against A1's embedding baseline at 0.6431/0.5749 = 0.0682 on the same split. NRMS
  lifts both cohorts by ~0.05 and leaves the gap marginally *wider* — the same conclusion
  ADR-008 reached for embeddings: better models raise the whole curve rather than fixing
  cold start. A history encoder cannot help a user with fewer than five history items.

## Two structural notes

- **Head is 38% of MIND impressions but only 3.9% of EB-NeRD's.** News churn explains it:
  most EB-NeRD clicks land on articles never clicked during the 21-day train window, so they
  are tail by construction. The same definition therefore describes a much rarer population
  on EB-NeRD, and the two head numbers are not directly comparable across datasets.
- **EB-NeRD's warm/cold slice is degenerate** — all 244,647 validation impressions are warm.
  That is not a bug: ADR-002's addendum established `ebnerd_small` has zero users below the
  threshold by construction of its active-user filter. It is reported as degenerate rather
  than silently omitted.

# Affected Files

- `src/evaluation/ranking_metrics.py`
- `src/evaluation/bootstrap.py`
- `src/retrieval/score.py`
- `scripts/run_ranking_eval.py`
- `tests/unit/test_ranking_metrics.py`, `tests/unit/test_bootstrap.py`,
  `tests/unit/test_score.py`
- `tests/integration/test_ranking_eval_pipeline.py`

---

## Expected Refactoring

None — first implementation.

---

## Breaking Changes

No — additive. `src/retrieval/index.py`'s `BM25Index` gained a new field
(`id_to_col`) and `retrieve.py::retrieve_top_k` was refactored to call the
extracted `score_all`, both verified behavior-preserving against existing
tests (`tests/unit/test_retrieval.py` unchanged and passing;
`recall_at_k`'s bootstrap refactor pinned byte-identical by
`tests/unit/test_metrics.py`, written before the refactor).

---

## Required Tests

- Unit: nDCG/MRR/AUC against hand-computed reference values, including a
  multi-click case; tie-break determinism (same seed+impression_id →
  identical order); degenerate AUC → NaN, counted not raised; novelty
  smoothing on a zero-train-click article (`test_ranking_metrics.py`).
- Unit: `BM25Scorer` subset scores match `score_all`'s full vector at the
  same positions; identity-cache behavior (`test_score.py`).
- Unit: bootstrap CI reproducibility and CI-narrows-with-more-units sanity
  check (`test_bootstrap.py`).
- Integration: real EB-NeRD-demo-validation data end-to-end, metric-range
  assertions, warm/cold cohort presence, JSON-serializability
  (`test_ranking_eval_pipeline.py`).

---

## Expected Benchmarks

AUC, MRR, nDCG@5, nDCG@10, diversity@10, novelty@10 (with bootstrap 95%
CI), and coverage@10 (point estimate) — overall/warm/cold — against BM25
on `MINDsmall-dev` and `ebnerd_small-validation`. See Benchmark Results.

---

## Documentation Updates

- [x] Architecture (`ARCHITECTURE.md`'s Evaluation component, filled in
      from placeholder)
- [x] Project State
- [ ] Knowledge Base

---

# Benchmark Results

## Baseline

None — this is the first ranking-metrics harness implemented in this
project (recall@K, ADR-006, measures a structurally different question).

## Candidate

BM25 (ADR-005/ADR-006's existing index/query construction), scored per
impression via `BM25Scorer`, ranked via this ADR's tie-break rule.

## Benchmark Environment

| Item | Value |
|------|-------|
| Dataset | MIND-small dev (73,152 impressions); EB-NeRD-small validation (244,647 impressions) |
| Hardware | Local development machine (macOS, Python 3.14) |
| Software Version | scikit-learn (roc_auc_score), numpy, scipy |
| Random Seed | 0 (tie-break and bootstrap CIs) |

---

## Experiment

`experiments/ranking_bm25_mind_2026-08-10/`,
`experiments/ranking_bm25_ebnerd_small_2026-08-10/`

---

## Results

**MINDsmall-dev** (73,152 impressions, 50,000 users, 42,416-article corpus):

| Metric | Overall | Warm (n=41,986) | Cold (n=8,014) |
|---|---|---|---|
| AUC | 0.5692 (0.5670–0.5714) | 0.5766 (0.5742–0.5789) | 0.5242 (0.5189–0.5292) |
| MRR | 0.3115 (0.3090–0.3140) | 0.3138 (0.3112–0.3167) | 0.2975 (0.2913–0.3040) |
| nDCG@5 | 0.2887 (0.2860–0.2913) | 0.2888 (0.2859–0.2918) | 0.2879 (0.2812–0.2948) |
| nDCG@10 | 0.3486 (0.3460–0.3512) | 0.3485 (0.3459–0.3514) | 0.3490 (0.3425–0.3556) |
| Diversity@10 | 0.8367 (0.8353–0.8382) | 0.8314 (0.8299–0.8331) | 0.8690 (0.8660–0.8718) |
| Novelty@10 | 16.2988 (16.2878–16.3094) | 16.2857 (16.2736–16.2975) | 16.3784 (16.3497–16.4073) |
| Coverage@10 (point est.) | 0.0834 | 0.0804 | 0.0450 |

**EB-NeRD-small validation** (244,647 impressions, 15,342 users, 20,738-article corpus):

| Metric | Overall = Warm (n=15,342) | Cold |
|---|---|---|
| AUC | 0.5288 (0.5272–0.5304) | n/a — 0 cold users |
| MRR | 0.3412 (0.3396–0.3428) | n/a |
| nDCG@5 | 0.3745 (0.3725–0.3764) | n/a |
| nDCG@10 | 0.4543 (0.4526–0.4561) | n/a |
| Diversity@10 | 0.7949 (0.7933–0.7966) | n/a |
| Novelty@10 | 17.1667 (17.1526–17.1801) | n/a |
| Coverage@10 (point est.) | 0.2057 | n/a |

(`experiments/ranking_bm25_mind_2026-08-10/`,
`experiments/ranking_bm25_ebnerd_small_2026-08-10/`; 0 degenerate-AUC
impressions skipped on either dataset, confirming the Context section's
grounding statistic still held at full scale, not just in the sample
checked before designing.)

---

## Interpretation

**Warm > cold on MIND, on AUC specifically** (0.577 vs. 0.524), consistent
with recall@K's existing warm/cold finding (ADR-006) and the Day-1 working
hypothesis (BM25 favors users with more history to build a specific
query from). Notably, **nDCG@5 and nDCG@10 barely differ between warm and
cold** (0.2879 vs 0.2888 at k=5; 0.3490 vs 0.3485 at k=10) — this looks
like a contradiction of the warm>cold story until you account for the
tie-break: a cold user's ranking is a uniformly random permutation
(ADR-007's own tie-break rule) of a candidate list where roughly 1/23
candidates is relevant on average (MIND's median 23 candidates/impression)
— a **random** ranking already gets a non-trivial nDCG@5/@10 by chance
whenever the one relevant item happens to land in the top 5–10 anyway,
which is common with short candidate lists. AUC is a purer discrimination
signal (rank-based over the *entire* list, not just a cutoff) and shows
the warm/cold gap nDCG's cutoff partially masks. This is a genuine,
reportable finding about *why* Q4 mandates multiple metrics rather than
just one — they disagree here, and the disagreement is informative, not
noise.

**Cold diversity is higher than warm on MIND** (0.869 vs. 0.831) for the
same structural reason: a cold user's top-10 is a random draw from
whatever candidates the impression happened to list, which — absent any
topical concentration a real BM25 match would produce — tends toward
higher category diversity by chance, not because the system is doing
anything intentional for cold users.

**EB-NeRD's AUC (0.529) is lower than MIND's (0.569)**, mirroring
ADR-006's recall@K finding that EB-NeRD's BM25 signal is relatively
weaker. But **EB-NeRD's nDCG@5/@10 (0.375/0.454) are substantially higher
than MIND's (0.289/0.349)** — again explained structurally, not by EB-NeRD
having better retrieval: EB-NeRD's median candidate-list length (9–12) is
roughly a third of MIND's (23), so a true positive is combinatorially far
more likely to land inside a top-5/top-10 cutoff regardless of ranking
quality. This is exactly why Q4 mandates AUC (which is list-length-
invariant) alongside cutoff-dependent nDCG — comparing nDCG@10 alone
across datasets with very different candidate-list lengths would produce
a misleading "EB-NeRD retrieval is better" conclusion that AUC directly
contradicts.

**EB-NeRD's coverage (0.206) is over double MIND's (0.083)** — BM25
recommends from a much larger fraction of the EB-NeRD catalog in
aggregate. Plausible driver (not yet isolated): EB-NeRD's per-user
histories are much longer (median 81–93 articles vs. MIND's 15, per
ADR-005), producing more differentiated per-user queries that spread
recommendations across more of the catalog, rather than concentrating on
a smaller popular subset the way MIND's shorter queries might. A
candidate hypothesis for Phase 4 to test, not a settled conclusion.

Both datasets confirm the ADR-002-addendum finding structurally: EB-NeRD's
cold cohort remains empty at the `small` tier (0 users below threshold),
so the warm/cold comparison Q4.3 requires is — again — only meaningful for
MIND with this project's current data.

---

## Comparison to Alternatives

Options A/B for tie-breaking and Option A for coverage-CI were not
empirically benchmarked against the chosen options — both rejections are
structural/statistical arguments (bias risk, resampling-mechanics bias),
not measured performance differences, so a benchmark comparison wouldn't
be meaningful here (there's no "which gives better numbers," only "which
is correct").

---

# Related Decisions

## Influenced By

- ADR-002 (Unified Data Schema) — `category` field used for diversity;
  train/validation article-overlap statistics motivating novelty smoothing
- ADR-005 (Query Construction) — cold-start threshold reused for warm/cold
  slicing; cold-user empty-query behavior is why the tie-break is mandatory
- ADR-006 (BM25 Variant) — the scoring this harness's first consumer uses

## Influences

- Future semantic retrieval ADR (Phase 4) — will plug into
  `src/retrieval/score.py`'s `Scorer` Protocol and reuse this harness
  unchanged, per this ADR's Research Trigger above

---

# References

## Internal

- ARCHITECTURE.md — Evaluation component
- ADR-002-unified-data-schema.md, ADR-005-query-construction.md,
  ADR-006-bm25-variant.md
- `experiments/ranking_bm25_mind_2026-08-10/`,
  `experiments/ranking_bm25_ebnerd_small_2026-08-10/`

## External

- `trec_eval` — standard TREC evaluation tool; tied-relevance handling
  convention referenced for the tie-break rationale.

---

# Decision History

| Date | Event |
|------|-------|
| 2026-08-10 | Real impression-candidate-count/click-rate/train-overlap statistics pulled before design |
| 2026-08-10 | Design explored via a dedicated Plan-agent pass; four sub-decisions identified as bundled, not independent |
| 2026-08-10 | Implemented, unit + integration tested |
| 2026-08-10 | Full benchmark run on MINDsmall-dev and ebnerd_small-validation |

---

# Notes

This ADR bundles four sub-decisions for the same reason ADR-005 bundled
three (query text / tokenization / cold threshold): they aren't
independent choices made in isolation, they're facets of one underlying
question ("how does this harness turn per-impression scores into honest,
reproducible metrics"), and splitting them into four separate ADRs would
scatter reasoning that needs to be read together to make sense.
