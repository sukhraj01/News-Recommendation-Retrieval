# ADR-010 — MIND Second-Submission Candidate Search (Symbolic Signals, Learned Combiner, Cohort Gating)

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

**Date:** 2026-08-21
**Status:** Decided (adopt Candidate G, **conditional** — see Final Decision)
**Severity:** High

---

# Engineering Question

Can any real improvement over MIND's deployed embedding baseline (AUC
0.6340, 95% CI 0.6319-0.6361 on MINDsmall-dev) be found and validated
cheaply enough, before spending one of Codabench competition 13967's
submission slots, to justify a real second MIND submission? This ADR
covers the second round of candidates tried in the same session, after
Candidates A/B/C (dense entity embeddings, untuned BM25+embedding hybrid,
recency-weighted history — see ADR-008's and ADR-005's 2026-08-21 addenda)
all lost. Per the engineer's explicit framing: "explore genuinely different
approaches... you have latitude here, don't limit yourself" — this round
deliberately avoids re-tuning A/B/C's own mechanisms and instead tries
symbolic (non-vector) signals, a supervised combiner, and — once evidence
pointed there — a routing strategy.

---

# Decision Scope

- [ ] Entire project
- [x] Single component
- [x] Experiment only (contingent on Final Decision's condition)
- [ ] Temporary / Prototype
- [x] Assignment-specific
- [ ] General reusable pattern

**Affected Components:** New standalone scoring modules/scripts only
(`src/retrieval/features.py`, `scripts/run_{symbolic_overlap,popularity,
learned_combiner,gated_cohort}_experiment.py`). No production pipeline,
schema, or the deployed `EmbeddingScorer` code path is modified — same
isolation discipline ADR-009's ablation and this session's earlier A/B/C
addenda used.

---

# Context

MIND's deployed method is the unweighted mean-pooled `EmbeddingScorer`
(ADR-008). Round 1 of this search (Candidates A/B/C) tried three
variations on that same method's own mechanisms (a different vector
source, a fixed blend with BM25, a reweighted pooling) and all lost,
CI-clear. This round asks a different question: are there real,
*mechanistically distinct* signals in MIND's own data
(`news.tsv`'s category/entity fields, MINDsmall-train's real click labels,
train-split article popularity) that a fixed hand-built method can't
capture, and — if several weak/losing signals exist — can a model that
*learns* how to weight them do better than picking one?

---

# Decision Criteria

| Criteria | Importance | Notes |
|---|---|---|
| Genuinely different mechanism from A/B/C, per the engineer's own framing | Critical | Re-tuning a losing mechanism's parameters was explicitly ruled out as the goal here |
| Evaluated with the same rigor as every other candidate (real bootstrap CI, MINDsmall-dev, no cherry-picking) | Critical | Stated ground rule |
| No hidden tuning | High | Same stance as ADR-009/Candidate B: untuned first attempts, so a result reflects "does this help at all," not a search's best case |
| Reuses existing infrastructure where the mechanism is genuinely shared | Medium | `Scorer`/`EmbeddingIndex`/Q4 harness/bootstrap CI reused unchanged throughout, per this project's "extend, don't duplicate" practice |
| Statistically appropriate test for the actual comparison being made | Critical (added mid-flight, see Candidate G) | A gated/routed score sharing most of its impressions with the baseline is a **paired** comparison, not two independent samples — using the same marginal-CI-overlap heuristic used for A-F would have been the wrong test and nearly produced a false negative (see Candidate G below) |

---

# Design Space Exploration

## Candidate D — Symbolic category/entity overlap re-ranker

### How it Works

New `src/retrieval/features.py`: per-user `HistoryProfile` (category counts,
subcategory counts, entity-mention counts built from
`entities.py::parse_entity_mentions`, deduplicated per article). Per
candidate: `score = category_match_score + subcategory_match_score +
log1p(entity_overlap_count)` — an untuned, unweighted sum of three [0,1]-
or-small-integer symbolic identity signals. Deliberately distinct from
Candidate A: identity match ("does this candidate share a category/entity
with the user's history"), never a vector distance.

### Optimizes For

A mechanism dense embeddings structurally can't represent directly
(exact category/entity identity, not semantic proximity).

### Sacrifices

No text-similarity signal at all; a candidate with zero shared
category/entity gets a zero score regardless of topical relevance.

### Complexity

Low — one new module, no training, reuses the Q4 harness/CI machinery
unchanged.

### Result (see Evidence)

CI-clear loss: AUC 0.6134 (95% CI 0.6112-0.6154).

---

## Candidate E — Train-split popularity-only baseline

### How it Works

Reuses `ranking_metrics.py::build_train_popularity` (already built for
Q9's novelty metric, ADR-007/009) directly as a *scoring* signal: rank
every impression's candidates by their train-split click-through
popularity alone. No per-user signal, no query, no history — every user
sees the same candidate ranked the same way.

### Optimizes For

Isolating whether a pure item-popularity prior, with zero personalization,
carries real signal on its own (a classic recommender baseline).

### Sacrifices

No personalization at all, by design — this is a floor check, not a
candidate expected to compete with a personalized method.

### Result (see Evidence)

CI-clear loss: AUC 0.5318 (95% CI 0.5299-0.5335) — the weakest of every
candidate tried in either round except Candidate A.

---

## Candidate F — Learned logistic-regression combiner

### How it Works

Seven features per (impression, candidate) — `bm25` (ADR-006),
`embed_cos` (ADR-008, unweighted mean-pool), `recency_embed_cos`
(Candidate C's mechanism, decay=0.9, **as an additional feature this time,
not a full replacement** — exactly what Candidate C's own addendum named as
the next thing worth trying), `category_match`/`subcategory_match`/
`entity_overlap_log1p` (Candidate D's mechanism), `log_popularity`
(Candidate E's mechanism). `StandardScaler` + `LogisticRegression`
(`lbfgs`, untuned defaults, no hidden hyperparameter search), fit on
MINDsmall-**train**'s real click labels (5,843,444 candidate rows, 4.04%
click rate), evaluated on MINDsmall-**dev**. Train-side and dev-side each
build their own BM25/embedding index from that split's own article
catalog (unchanged project convention) — only the fitted linear weights
transfer.

### Optimizes For

Letting real labels decide each signal's weight, instead of an untuned
fixed blend (directly addressing why Candidate B likely failed: an
untuned 50/50 weight diluted a stronger signal with a weaker one).

### Sacrifices

A single global linear model — no per-cohort specialization, despite
warm/cold users plausibly needing different weightings (this sacrifice is
exactly what Candidate G was built to test).

### Result (see Evidence)

**Overall**: CI-clear loss, AUC 0.6255 (95% CI 0.6234-0.6276). **Cold
cohort**: CI-clear **win** — 0.5926 (95% CI 0.5869-0.5985) vs. baseline
cold's 0.5737 (95% CI 0.5682-0.5792), the first win either round produced
anywhere. **Warm cohort**: loss, 0.6309 vs. baseline warm's 0.6439. The
fitted coefficients (standardized features) put by far the most weight on
`log_popularity` (+0.744) and `embed_cos` (+0.278), with
`recency_embed_cos` landing near zero and slightly negative (-0.014) — the
model itself, given the choice, assigns recency almost no value, consistent
with Candidate C's own standalone loss.

---

## Candidate G — Cohort-gated scorer (embed for warm, Candidate F's combiner for cold)

### How it Works

Not a new mechanism — a **routing decision** motivated directly by
Candidate F's own cohort split: use the deployed `EmbeddingScorer`
unchanged for warm users (where it already wins), and Candidate F's
already-fitted combiner (reused exactly, not retrained — `lbfgs` is
deterministic and the point was to test routing, not fit a new model) for
cold users (where F wins instead).

### Optimizes For

Taking each method's own best segment rather than forcing one method to
cover every user.

### Sacrifices

Two scoring paths to maintain instead of one; the cohort boundary
(history length < 5, ADR-005) becomes a live routing decision, not just an
evaluation slice.

### A statistically important complication

Overall AUC = 0.6366 (95% CI 0.6346-0.6388) vs. baseline's 0.6340 (95% CI
0.6319-0.6361). **These two marginal CIs overlap** (0.6346-0.6361) — by
the same non-overlapping-CI heuristic used to judge every other candidate
in both rounds, this would read as "not CI-clear." But G and the baseline
share the exact same warm-cohort scores by construction (G *is* the
baseline for 85.9% of impressions) — comparing their marginal CIs discards
that shared structure and is the wrong test for this specific comparison.
A **paired bootstrap** (`run_gated_cohort_experiment.py::paired_metric_diff_ci`,
resampling users once per replicate and computing gated-minus-baseline AUC
on the *same* resampled users, reusing `bootstrap.py::bootstrap_ci`
unchanged) gives the statistically appropriate answer: **+0.0027 (95% CI
+0.0018 to +0.0034), entirely excluding zero.**

### Result (see Evidence)

**A genuine, CI-clear win by the correct (paired) test.** This is the only
candidate across both rounds (six losses: A, B, C, D, E, plus F's own
overall number; one cold-only win: F; one real overall win: G) that clears
the bar the engineer set: "commit only whichever shows a real, CI-clear
win over 0.634."

---

# Comparison Summary

| Candidate | Mechanism | Overall AUC | 95% CI | vs. baseline (0.6340, CI 0.6319-0.6361) |
|---|---|---|---|---|
| A (round 1) | Dense TransE entity vectors | 0.5525 | 0.5503-0.5546 | Loss |
| B (round 1) | Untuned 50/50 BM25+embed hybrid | 0.6263 | 0.6242-0.6284 | Loss |
| C (round 1) | Recency-weighted history (replacement) | 0.6265 | 0.6243-0.6286 | Loss |
| D | Symbolic category/entity overlap | 0.6134 | 0.6112-0.6154 | Loss |
| E | Train-popularity only | 0.5318 | 0.5299-0.5335 | Loss |
| F | Learned combiner (7 features) | 0.6255 | 0.6234-0.6276 | Loss overall; **CI-clear win on cold cohort** (0.5926 vs. 0.5737) |
| **G** | **Cohort-gated (embed warm / F cold)** | **0.6366** | **0.6346-0.6388 (marginal)** | **Win — paired bootstrap +0.0027, 95% CI +0.0018 to +0.0034, excludes zero** |

---

# Final Decision

## Chosen Option

**Candidate G — cohort-gated scorer**, conditionally adopted pending one
open question (see below), not yet submitted to Codabench.

### Reason

It is the only candidate, across seven tried this session, that beats the
deployed baseline with statistical rigor appropriate to the comparison
being made. The mechanism is explainable (embeddings already win warm
users decisively; the combiner's dominant learned feature —
`log_popularity`, by a wide margin over every other feature — is exactly
the kind of signal that still functions when a user's history is too thin
for BM25/embeddings to say much, which is structurally the cold-user
problem), not a coincidental artifact of one lucky metric.

### Condition before an actual Codabench submission

**This result is validated on MINDsmall-dev only.** Every prior real
submission this project made (MIND leaderboard AUC 0.6195, EB-NeRD 0.5404
— PROJECT_STATE.md's Leaderboard Submission row) was generated from
MINDlarge-scale predictions, and ADR-008 itself was re-verified at
MINDlarge-dev scale before being trusted, not shipped on MINDsmall
evidence alone. Whether this project (a) re-verifies Candidate G at
MINDlarge scale before submitting (more rigorous, more time/compute —
MINDlarge-train is ~2.2M impressions vs. MINDsmall-train's 156,965, a real
scale jump this project's own history has needed algorithmic fixes for
before, per PROJECT_STATE's Data Pipeline notes) or (b) accepts
MINDsmall-dev evidence as sufficient for a second submission given time
constraints is a genuine scope/rigor trade-off for the engineer to decide,
not one this ADR resolves unilaterally — consistent with how ADR-008's
Addendum left the EB-NeRD second-submission question open for the same
reason.

---

# Rationale

## Why this option is best

- Only candidate (of 7) with a real, correctly-tested win over baseline.
- Explainable mechanism, not a metric artifact: cold users structurally
  lack the history BM25/embeddings need; a global item-popularity prior
  (which needs no history at all) fills exactly that gap, and F's fitted
  model discovered this from data rather than being told to.
- Reuses, rather than replaces, everything already proven to work (warm
  users keep the already-winning deployed embedding scorer untouched) —
  the smallest change that captures the real gain, not a wholesale
  replacement.

## Why alternatives were rejected

### D, E (standalone symbolic/popularity signals)

Real, honestly-measured losses on their own — but their *ingredients*
turned out to matter as combiner features (F), which is why they were kept
as inputs rather than discarded once they lost standalone. Consistent with
this session's own framing: a losing standalone signal can still carry
marginal information a learned model can extract.

### F as a standalone submission (not chosen; G chosen instead)

F's own overall number is a real loss (0.6255 < 0.6340, CI-clear) — adopting
it directly, despite its cold-cohort win, would still be net-worse on the
metric the objective is judged by. G was the direct fix once F's cohort
split was visible.

---

# Decision Confidence

**Current Confidence:** Medium-High

### Why

- The paired-bootstrap result is a real statistical finding, computed with
  the same rigor (2000 bootstrap replicates, per-user resampling) as every
  other CI in this project, and the underlying mechanism has a plausible,
  checkable explanation (dominant `log_popularity` coefficient, cold-user
  history scarcity).
- Not High: unverified at MINDlarge scale (see Final Decision's
  condition), and the paired-bootstrap technique, while statistically
  standard, is a new addition to this project's evaluation toolkit this
  session — worth a second pair of eyes before being trusted for a real
  submission decision, same "defensible in a viva" bar CLAUDE.md sets for
  every decision.

### What Would Increase Confidence

- MINDlarge-dev re-verification (the condition above).
- Confirming the paired-bootstrap CI's exclusion of zero is robust to the
  bootstrap seed (currently pinned to `SEED = 0` throughout, per this
  project's reproducibility standard — re-running with a different seed
  would be a cheap robustness check, not required before this ADR but
  worth doing before an actual submission).

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (warm/cold comparison, Q4 metrics)
- [x] Experimental Benchmark (every number above, MINDsmall-dev/-train)
- [ ] Official Documentation
- [ ] Research Paper
- [x] Industry Practice (item-popularity priors and logistic-regression CTR
  combiners are both standard, well-established recommender-systems
  baselines — the design choice itself is not novel, only its application
  here is)
- [ ] Community Consensus
- [x] Engineering Inference (Candidate G's routing idea, inferred directly
  from Candidate F's own cohort-split result, not speculated in advance)

## Empirical Evidence

See Comparison Summary table above; full per-cohort AUC/MRR/nDCG@5/nDCG@10
tables in each candidate's own `experiments/candidate_{d,e,f,g}_*_2026-08-21/`
directory.

## Accepted Trade-offs

- Candidate G's fitted model (F's coefficients) was trained once and reused
  without retraining for G — appropriate for testing a routing idea against
  an already-fitted deterministic model, but means G has not been jointly
  optimized as a single system (a cohort-aware model trained end-to-end
  might do better still — untested, flagged as future work, not built here
  to avoid scope creep on an already-large session).
- `log_popularity` uses train-split-only popularity per Q9's anti-gaming
  requirement (ADR-007/009) — clean with respect to dev evaluation; the
  much smaller concern of an article's own train-impression labels
  contributing to its own train-time popularity feature (self-information,
  not future leakage) was considered and accepted as standard practice,
  not a violation of this project's anti-leakage standard (see
  `run_learned_combiner_experiment.py`'s module docstring).

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|---|---|---|
| MINDsmall-dev's warm/cold cohort mix (85.9%/14.1% of impressions) is representative of MINDlarge/the real test set | The paired win's overall magnitude (+0.0027) is a weighted average across cohorts — a different mix could change whether it clears zero | Check the same cohort proportions at MINDlarge scale before trusting the win transfers |
| Candidate F's fitted coefficients (trained on MINDsmall-train) generalize to a different scale | Candidate G reuses F's exact fitted weights, untouched | Would need to be refit on MINDlarge-train if the project moves to MINDlarge-scale verification, not assumed to transfer as-is |

---

# Conditions for Revisiting

## Technical Triggers

- [x] The MINDlarge-scale verification named in Final Decision, before any
  real Codabench submission
- [ ] Latency/memory (Candidate G's cold-user path costs real extra
  compute per impression vs. the deployed baseline alone — not yet
  benchmarked against a per-impression latency budget, since no such
  budget exists for this offline-evaluation assignment)

## Research Triggers

- [x] A jointly-trained cohort-aware model (rather than two independently
  fitted pieces) as a follow-up, if this ADR's adopted version is
  submitted and the engineer wants to push further
- [x] Bootstrap-seed robustness check (Decision Confidence, above)

## Project / Assignment Triggers

- [x] Whether to actually submit — explicit engineer decision, not
  automatic from this ADR

---

**Estimated Cost to Change:** Easy — every new candidate this session
lives in standalone modules/scripts, isolated from the deployed pipeline;
reverting to the plain baseline is deleting nothing (the deployed
`EmbeddingScorer` code path was never modified).

---

# Engineering Impact

## Affected Files

- `src/retrieval/features.py`, `tests/unit/test_features.py`
- `scripts/run_{symbolic_overlap,popularity,learned_combiner,gated_cohort}_experiment.py`
- `tests/unit/test_gated_cohort.py`
- `experiments/candidate_{d,e,f,g}_*_2026-08-21/`

## Breaking Changes

None — no production code path touched.

## Required Tests

- Unit: symbolic feature functions (`test_features.py`, 10 tests), paired
  bootstrap correctness (`test_gated_cohort.py`, 3 tests).
- Full existing suite (166 unit + integration) reconfirmed passing after
  every addition this session.

---

# Related Decisions

## Influenced By

- ADR-005 (Query Construction) — cold threshold, Candidate C's mechanism
  reused as a combiner feature
- ADR-006 (BM25 Variant) — `bm25` feature
- ADR-007 (Q4 Harness Design) — every metric/CI/tie-break mechanism reused
  unchanged
- ADR-008 (Semantic Retrieval Design) — `embed_cos` feature, the deployed
  baseline this whole search is measured against
- ADR-009 (Leaky-Feature Ablation) — `build_train_popularity` reused
  directly; untuned-blend design precedent for Candidate D
- This session's own ADR-005/ADR-008 addenda (Candidates A/B/C, round 1)

## Influences

- Any future MIND submission decision (pending the condition above)

---

# References

## Internal

- `experiments/candidate_{d,e,f,g}_mind_small_2026-08-21/`
- `knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`

---

# Decision History

| Date | Event |
|---|---|
| 2026-08-21 | Round 2 candidates (D, E, F) run against round-1's established baseline; all three lose overall, F wins cold cohort |
| 2026-08-21 | Candidate G (cohort gating) built directly from F's cohort split; initial marginal-CI comparison ambiguous (overlapping) |
| 2026-08-21 | Paired bootstrap added (statistically correct test for this specific comparison); confirms a real, CI-clear win, +0.0027 (95% CI +0.0018 to +0.0034) |
| 2026-08-21 | ADR-010 written; adoption conditional on a MINDlarge-scale verification decision, left open for the engineer |

---

# Notes

This ADR documents a case where the *first* statistical test applied
(marginal CI overlap, the same test used throughout both rounds) would
have produced a false negative — worth naming explicitly per CLAUDE.md's
evidence-hierarchy discipline ("clearly distinguish between evidence,
inference, and opinion"): the marginal-CI heuristic is a reasonable default
for comparing genuinely independent methods (A vs. baseline, D vs.
baseline, etc.), but Candidate G and the baseline are not independent —
they share almost all their impressions by construction, and a paired test
is the statistically correct tool for that specific shape of comparison.
Applying the same heuristic everywhere without checking whether its
independence assumption actually holds would have missed the one real
result this whole two-round search produced.

---

# Addendum — MINDlarge Verification and Second-Submission Build (2026-08-21/22)

**Status:** Resolves this ADR's Final Decision condition. Candidate G is
now verified at MINDlarge scale and a real prediction file has been built,
verified, and packaged for MINDlarge_test — **not yet submitted to
Codabench**; the engineer uploads manually, per this project's standing
practice for actual leaderboard submissions.

## MINDlarge-dev verification (Kaggle)

Per this ADR's Final Decision, Candidate G's exact MINDsmall-fitted model
(no retraining) was re-evaluated at MINDlarge scale via a Kaggle relay
notebook (`notebooks/mind_gated_cohort_mindlarge_{kaggle_run.py,
src_bundle.zip}`), after local execution proved infeasible — not because
the computation itself was heavy (peak local RSS ended up under 1GB once
real bugs were fixed, see below), but because the local machine's memory
headroom was already critically low from many days of accumulated
background processes, confirmed live (system swap down to ~800MB free, a
process stuck in uninterruptible I/O wait) before any workaround was
attempted, per CLAUDE.md's Resource Availability clause.

**Result, confirmed by the engineer from the real Kaggle run: a genuine,
CI-clear win — overall AUC +0.0019 over the deployed baseline (95% CI
+0.0015 to +0.0023, paired bootstrap, same test as the MINDsmall-dev
result above), CI excludes zero.** This is a smaller effect size than the
MINDsmall-dev screen's +0.0027 (95% CI +0.0018 to +0.0034) but the same
direction and both CIs clear zero — the win transfers to real scale, not
just the smaller screen. The full per-metric/per-cohort Kaggle output
(config.json/results.json) is expected at
`experiments/candidate_g_gated_cohort_mind_large_2026-08-21/` once
retrieved from the engineer's Kaggle session download — **this addendum
records the confirmed headline number; the complete breakdown is a
follow-up documentation pass, not a blocker for the submission build
below**, which only depends on the win being real, not on every metric's
exact value.

### Real bugs found and fixed en route (not Kaggle-specific — real project bugs)

Getting this run to actually complete surfaced three genuine, previously-
latent bugs in shared project code, none specific to Kaggle or to this
experiment — all fixed at the root, with new unit tests, per CLAUDE.md's
"fix the underlying issue, don't bypass it" guidance:

1. **`read_zip_member_bytes` (`src/utils/io.py`) had no fallback for a
   flat-packed zip.** The Hugging Face mirror this project's own
   `download.py` already used as its MIND download source packs
   `MINDlarge_train.zip`'s members flat at the zip root
   (`news.tsv`/`behaviors.tsv`, no `MINDlarge_train/` folder) — the
   opposite convention from every MIND zip this project had seen before,
   and the existing fallback (added 2026-08-12 for the *opposite* mismatch,
   an extra wrapping folder) couldn't match it. Fixed with a second
   fallback tier (exact basename match among flat entries), 3 new unit
   tests. See `io.py`'s own updated docstring for the full history.
2. **`load_encoder`/`build_embedding_index` (`src/retrieval/embed.py`)
   hardcoded `device="mps"`** — correct for this project's own Mac
   development machine, but `"mps"` doesn't exist on Kaggle's Linux
   runners, raising `RuntimeError: PyTorch is not linked with support for
   mps devices` on first real use. Fixed with `_default_device()`
   (CUDA > MPS > CPU, auto-detected), 3 new unit tests.
3. **A local memory bug in the popularity-computation step** (this
   session's own `run_gated_cohort_experiment.py`, before this addendum):
   a naive `pd.read_parquet(...)` with no `columns=` filter on
   MINDlarge-train's 83.5M-row impressions table pulled in four
   always-null EB-NeRD-only float64 columns and a full-precision
   timestamp column across every row, and the resulting frame was kept
   resident for the rest of the run instead of released — real, measured
   contributing cause of the local memory crisis above. Fixed with an
   explicit `columns=["article_id", "clicked"]` restriction and `del` +
   `gc.collect()` immediately after use (peak footprint for this step
   alone measured directly at 745MB, down from multi-GB and an
   unresponsive process).

## MINDlarge_test prediction generation (the actual second submission)

New `scripts/generate_mind_gated_predictions.py`: a `GatedScorer`
implementing the same `Scorer` protocol every other method in this project
uses (`src/retrieval/score.py`), so `src/submission/mind_format.py::
write_predictions` — the exact, unmodified module every prior real
submission from this project used — needed zero changes. Warm users route
to the deployed `EmbeddingScorer` unchanged; cold users route to Candidate
F's MINDsmall-fitted combiner (reused exactly, not retrained). Local
execution was confirmed feasible before committing to it (this project's
own history: this exact scale, 2,370,727 impressions, already completed
locally once before for the embed-only baseline, ~1.9hr — and the
processed MINDlarge_test bundle plus its embedding cache already existed
on disk from that session, so no re-download/re-encode was needed).

### A fourth real bug, caught by a smoke test before the real run

A small-scale smoke test against MINDsmall-dev (run deliberately before
committing to the ~4hr real MINDlarge_test job, not skipped under time
pressure) surfaced a `RuntimeWarning: invalid value encountered in
matmul`. Root cause: MIND's raw `behaviors.tsv` can reference a candidate
absent from that split's own article corpus (a real, already-documented
quirk — `MINDlarge_test`'s `N89741`, per `score.py::_lookup_scores`'s own
docstring), which every index-backed feature scores `-inf` so it ranks
last deterministically. That convention works fine for a single-score
method, but combining three already-`-inf` features (`bm25`/`embed_cos`/
`recency_embed_cos`) through Candidate F's fitted *mixed-sign* coefficients
(`+0.076`/`+0.278`/`-0.014`) produces `-inf * positive + -inf * negative =
-inf + +inf = NaN`, not just `-inf` — a bug this project's single-feature
scorers structurally could never hit, unique to the first multi-feature
linear combiner. Fixed in `GatedScorer.score`: explicitly detect
`np.isneginf` on any of the three raw inputs and force the combined score
to `-inf` directly, bypassing the NaN-prone arithmetic — deterministic,
matches the existing project-wide convention, not left to incidental
NaN-sort behavior. 4 new unit tests, including one reproducing this exact
mixed-sign-coefficient shape.

### Verification (same discipline as every prior real submission)

- **Line count:** 2,370,727 — exact match to `MINDlarge_test`'s real
  impression count (this project's own established ground truth, per
  Part 4's original embed-only submission).
- **Distinct `impression_id` count:** 2,370,727 — no duplicates.
- **Malformed rank permutations:** 0, checked across all 2,370,727 lines
  (each line's ranks must be exactly a permutation of `1..N` for that
  impression's candidate count `N`).
- **All 32 `N89741`-affected impressions individually spot-checked**
  (same specific impressions this project's original embed-only submission
  spot-checked, re-derived directly from the raw zip rather than assumed
  unchanged) — every one produces a complete, valid permutation, confirming
  the NaN fix holds at full real scale, not just in the unit test.
- **File size:** 291,329,312 bytes — byte-identical to the original
  embed-only submission's `prediction.txt` (expected: same impression/
  candidate-count structure, only the rank *values* differ, so the encoded
  text is the same length).
- **Packaging:** `prediction.txt` zipped at the archive root (no folder
  prefix) via `zip -j`, matching `submissions/mind_large_test_embed/
  prediction.zip`'s exact structure, confirmed via `unzip -l` on both.

Output: `submissions/mind_large_test_gated_cohort/{prediction.txt,
prediction.zip, run.log}` — same three-file layout as the original
submission. Real run time: 15,084s (~4.19hr) locally, longer than the
~1.9hr embed-only baseline projected from (see Interpretation) — the
machine's chronic background memory pressure throughout this session
plausibly throttled CPU-bound work generally, not something specific to
this method; the process itself stayed healthy throughout (peak RSS well
under 500MB, monitored continuously with a preemptive kill-switch that
never triggered).

## Interpretation

Both the Kaggle re-verification and the real submission build surfaced
genuine bugs that a purely-local, single-method-scorer history of this
project had never been forced to confront: a zip-packaging convention
mismatch, a Mac-only hardcoded device default, a full-column read at a row
count where it mattered, and a multi-feature linear combiner's specific
failure mode on already-`-inf` inputs. None of these were specific to
Candidate G's *statistical* validity — they were engineering gaps in
infrastructure this session was the first to stress in these particular
ways (cross-platform execution, a raw-zip-driven submission path, a
combiner with mixed-sign coefficients). Fixed at the root, each with new
unit tests, consistent with CLAUDE.md's git-discipline and testing
philosophy — not papered over to hit a deadline.

## Conditions for Revisiting

- Retrieve the full Kaggle config.json/results.json and file it at
  `experiments/candidate_g_gated_cohort_mind_large_2026-08-21/` — this
  addendum's own headline-number citation should be replaced with a
  reference to that file once available.
- The real-vs-projected runtime gap (~4.19hr vs. ~1.9hr baseline) was
  attributed to system-wide memory pressure, not measured in isolation —
  if this method is run again on a quieter machine, worth confirming the
  gap closes, as a sanity check on that explanation.
- **The actual Codabench upload is a separate, explicit action for the
  engineer** — this addendum records that the submission is built and
  verified, not that it has been submitted.

## Related

- `notebooks/mind_gated_cohort_mindlarge_{kaggle_run.py,src_bundle.zip}`
- `scripts/generate_mind_gated_predictions.py`,
  `tests/unit/test_gated_predictions.py`
- `src/utils/io.py`, `tests/unit/test_io.py` (flat-root zip fallback)
- `src/retrieval/embed.py`, `tests/unit/test_embed.py` (device auto-detect)
- `submissions/mind_large_test_gated_cohort/`
- `experiments/candidate_g_gated_cohort_mind_large_2026-08-21/` (pending)

---

# Addendum — Candidate H: GBDT Combiner (2026-08-22)

**Status:** Resolves this session's objective ("train a real supervised
ranker... if it's a real CI-clear win, scale to MINDlarge"). **Neither
model tested is a win — both are real, CI-clear losses.** Documented per
this ADR's own standing practice: every candidate, win or lose, reported
plainly.

## Context

The session opened with an objective claiming this project had "never
trained an actual supervised model on MIND's real click labels." That
premise is false — Candidate F, above, already did exactly that
(`LogisticRegression` on MINDsmall-train's real labels) and lost overall
(0.6255 vs. baseline 0.6340). Rather than re-run F under a new name, this
addendum tests what's actually untried: a **nonlinear** model over F's
features plus one new one (`log_history_length`), isolated from a
**class-imbalance-correction** variant of the same linear model, so a
result is attributable to a specific cause rather than several changes at
once. Full reasoning: `/Users/test01/.claude/plans/rosy-twirling-ripple.md`
(this session's approved plan) and `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`.

## Design

Two models, both trained on MINDsmall-train (5,843,444 rows / 156,965
impressions, 4.04% click rate — identical data to F), evaluated on
MINDsmall-dev via the unchanged Q4 harness, both compared to the deployed
baseline via the same paired bootstrap `run_gated_cohort_experiment.py::paired_metric_diff_ci`
already established as the correct test:

- **H1** — `LogisticRegression(class_weight="balanced")`, otherwise
  identical to F (same `StandardScaler`, `lbfgs`, untuned `C`). Isolates
  the imbalance-correction hypothesis.
- **H2** — `sklearn.ensemble.HistGradientBoostingClassifier(early_stopping=True)`,
  library defaults otherwise. Isolates the nonlinear-interaction
  hypothesis. Chosen over LightGBM per the engineer's explicit choice
  (same histogram-boosting family, zero new dependency, no macOS OpenMP
  install risk).
- **New feature (both models):** `log_history_length = log1p(n_articles)`
  — `HistoryProfile.n_articles`, previously only used externally as
  ADR-005's hard `< 5` cohort-routing threshold, now given to the model
  directly (8 features total: F's 7 plus this one).

`scripts/run_gbdt_combiner_experiment.py`, `tests/unit/test_gbdt_combiner.py`
(4 tests: the new feature-append helper, a fit/predict smoke test for each
model).

## Results

| Model | Overall AUC | 95% CI | Warm AUC | Cold AUC | Paired vs. baseline (overall) |
|---|---|---|---|---|---|
| Baseline (deployed embed) | 0.6340 | 0.6319-0.6361 | 0.6439 | 0.5737 | — |
| F (LogisticRegression, plain) | 0.6255 | 0.6234-0.6276 | 0.6309 | 0.5926 | Loss |
| **H1 (LogisticRegression, balanced)** | **0.6319** | **0.6297-0.6340** | 0.6382 | **0.5936** | **-0.0021 (95% CI -0.0042 to -0.0002) — CI-clear loss** |
| **H2 (HistGradientBoostingClassifier)** | **0.5985** | **0.5961-0.6006** | 0.6004 | 0.5871 | **-0.0355 (95% CI -0.0381 to -0.0331) — CI-clear loss, large** |

Full breakdown (MRR/nDCG@5/nDCG@10, all cohorts, both models):
`experiments/candidate_h_gbdt_combiner_mind_small_2026-08-22/{config,results}.json`.

H1's fitted coefficients (standardized features, same ordering as F plus
the new one): `bm25` +0.193, `embed_cos` +0.346, `recency_embed_cos`
-0.063, `category_match` +0.117, `subcategory_match` +0.180,
`entity_overlap_log1p` +0.048, `log_popularity` +0.660 (dominant, same as
F), `log_history_length` -0.169. H2's `n_iter_` = 100, exactly its default
`max_iter` — early stopping never triggered during training.

## Interpretation

**H1 (imbalance correction alone): a small, real, CI-clear loss, not a
fix.** `class_weight="balanced"` moved overall AUC from F's 0.6255 to
0.6319 — closer to baseline but still a statistically significant loss
(paired CI entirely below zero, not just "no evidence of a win"). Cold
cohort improved marginally past F's own cold win (0.5936 vs. F's 0.5926,
both clearing baseline cold's 0.5737 by a wide, real margin). Imbalance
correction was a real, if small, improvement over F — not the missing
piece.

**H2 (nonlinear model): a large, unexpected, CI-clear loss** — worse than
F, worse than H1, worse than every candidate this two-session search has
tried except Candidate A (entity embeddings, 0.5525) and Candidate E
(popularity-only, 0.5318). This is the opposite of the hypothesis motivating
H2 (that nonlinear interactions/thresholds would help). The clearest
signal in the data: **the loss is concentrated in the warm cohort**
(0.6004 vs. baseline warm's 0.6439, a 0.044 absolute drop) while cold
stays roughly comparable to F/H1 (0.5871 vs. H1's 0.5936) — i.e., H2 is
actively *worse than the deployed baseline specifically where BM25/embedding
signal is already strong*, not just "no better than baseline everywhere."

**Most plausible explanation (inference, not directly verified this
session — flagged for a future revisit, not asserted as fact):**
`h2_n_iterations` = 100 equals the library default `max_iter` exactly —
`early_stopping=True`'s own internal validation slice never detected a
plateau, meaning the model kept "improving" against its own held-out slice
for all 100 boosting rounds. But that internal validation slice is drawn
from MINDsmall-**train**, which shares train's own self-information
structure between `log_popularity` and the very click labels being
predicted (ADR-010's own Accepted Trade-offs section already named this as
an accepted, small concern for a *linear* model's 7 coefficients). A
flexible tree ensemble has far more capacity to exploit that structure —
learning narrow, high-confidence splits around specific train-popularity
values or category/entity combinations that fit train (and train's own
internal validation slice, cut from the same distribution) very well but
don't generalize to MINDsmall-dev's different impressions. This would
explain both the warm-cohort-concentrated loss (warm impressions have the
richest, most exploitable feature combinations for a tree to overfit on)
and why early stopping's own signal never caught it (it was validating
against more of the same train-distribution artifact, not an independent
check).

**Calibrating expectations against the 0.65-0.70 range (stated as
inference, per this project's evidence hierarchy — not measured this
session):** MIND's own published baselines (NAML/NPA/LSTUR/NRMS, noted in
this project's Day-1 research log) reach roughly that range via end-to-end
trained neural encoders over raw title/abstract *text*, learning their own
representations rather than combining 8 already-compressed scalar scores.
Two model classes (linear, and now a reasonably capable nonlinear
tree ensemble) have both been tried over this project's current feature
set and both cap out at or below the deployed embedding baseline. That is
real evidence the ceiling here is the **feature set**, not the combiner's
model class — closing the gap further would most plausibly require either
new information (raw text, not just similarity scores) or an end-to-end
trained model, both a materially larger undertaking than this session's
"lightweight ranker, move fast" scope. Not verified directly; flagged as
the most likely next lever if this is revisited.

## Comparison Summary (updated)

| Candidate | Mechanism | Overall AUC | 95% CI | vs. baseline (0.6340) |
|---|---|---|---|---|
| A | Dense TransE entity vectors | 0.5525 | 0.5503-0.5546 | Loss |
| B | Untuned 50/50 BM25+embed hybrid | 0.6263 | 0.6242-0.6284 | Loss |
| C | Recency-weighted history (replacement) | 0.6265 | 0.6243-0.6286 | Loss |
| D | Symbolic category/entity overlap | 0.6134 | 0.6112-0.6154 | Loss |
| E | Train-popularity only | 0.5318 | 0.5299-0.5335 | Loss |
| F | Learned combiner (7 features, LogisticRegression) | 0.6255 | 0.6234-0.6276 | Loss overall; win on cold cohort |
| **G** | **Cohort-gated (embed warm / F cold)** | **0.6366** | 0.6346-0.6388 (marginal) | **Win — paired +0.0027, 95% CI +0.0018 to +0.0034** |
| H1 | Learned combiner (8 features, LogisticRegression, balanced) | 0.6319 | 0.6297-0.6340 | **CI-clear loss** (paired -0.0021, 95% CI -0.0042 to -0.0002) |
| H2 | Learned combiner (8 features, HistGradientBoostingClassifier) | 0.5985 | 0.5961-0.6006 | **CI-clear loss, large** (paired -0.0355, 95% CI -0.0381 to -0.0331) |

Candidate G, adopted before this session, is the only candidate whose
*local validation* showed a real, CI-clear win across nine candidates
tried to date — but see the addendum immediately below (filed after this
one, from the real Codabench upload): the real MINDlarge_test leaderboard
result came back essentially flat (-0.0003), not the win local validation
at every stage predicted. Combined with H1/H2's results above, the honest
state as of this addendum is that **no candidate in this search has
produced an improvement that has actually held on real, blind test data**
— G held up at every local re-verification but not at the real leaderboard;
H1/H2 didn't even clear local validation.

## Conditions for Revisiting

- **Not scaled to MINDlarge** — this session's objective's own condition
  ("if it's a real, CI-clear win, scale it") was not met; no MINDlarge run
  was warranted or attempted.
- If GBDT is revisited: constrain `max_leaf_nodes`/`max_depth` (currently
  library defaults) and/or use a genuinely out-of-sample early-stopping
  slice (e.g. split by *time* or by a held-out user set drawn to avoid
  sharing train-popularity structure) rather than sklearn's own
  same-distribution internal split, to test the overfitting hypothesis
  above directly rather than leaving it as inference.
- H1's cold-cohort AUC (0.5936) is marginally better than F's (0.5926),
  which Candidate G currently uses for its cold-user path. Swapping G's
  cold combiner from F to H1 is a cheap, well-motivated follow-up — not
  undertaken here to keep this pass focused on the session's primary ask
  (an overall win), which neither H1 nor H2 achieved.
- The feature-set-is-the-ceiling hypothesis (Interpretation, above) is the
  main open lever if a real improvement beyond Candidate G is still
  wanted: raw-text or end-to-end neural modeling, not further combiner
  tuning over the current 8 scores.

## Related

- `scripts/run_gbdt_combiner_experiment.py`, `tests/unit/test_gbdt_combiner.py`
- `experiments/candidate_h_gbdt_combiner_mind_small_2026-08-22/{config,results}.json`
- `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`
- `/Users/test01/.claude/plans/rosy-twirling-ripple.md` (this session's approved plan)

---

# Addendum — Real Codabench Result: Essentially Flat, Not the Local Win (2026-08-22)

**Status:** Does not reverse this ADR's decision to build and submit
Candidate G — that call was correct given the evidence available at the
time (a real, correctly-tested local win, per this ADR's own paired-
bootstrap discipline). What this addendum adds is the actual outcome,
reported honestly rather than assumed to match the local prediction.

## What was checked

`submissions/mind_large_test_gated_cohort/prediction.zip` was uploaded to
Codabench competition 13967 by the engineer (submission ID **896696**).
Leaderboard **Score column: 0.6192**, against the original embed-only
submission's (886468) **0.6195** — a **-0.0003** difference, essentially
flat, well within the kind of noise a single leaderboard readout carries
(no CI is exposed by Codabench's Score column, same caveat §3.5 of the
design note already states for the first submission).

## Results

| Submission | Method | Score | Submission ID | Date |
|---|---|---|---|---|
| 886468 | Embed (MiniLM), unweighted mean-pool | 0.6195 | — | 2026-08-12 13:01 |
| 896696 | Cohort-gated (embed warm / Candidate F combiner cold) | 0.6192 | — | 2026-08-22 |

| Stage | Predicted/measured effect (paired AUC, gated − baseline) |
|---|---|
| MINDsmall-dev local screen | **+0.0027** (95% CI +0.0018 to +0.0034) |
| MINDlarge-dev local re-verification (Kaggle) | **+0.0019** (95% CI +0.0015 to +0.0023) |
| Real MINDlarge_test leaderboard | **-0.0003** (0.6192 vs. 0.6195; no CI available) |

**Local validation predicted a small, CI-clear real win at every stage
checked before submission. The actual blind test set showed an
essentially flat result instead — the honest outcome, not the win that
was expected going in.**

## Interpretation

**Most likely explanation: population difference between MINDlarge_dev and
MINDlarge_test, not a pipeline defect.** MIND's official splits are
temporally disjoint, non-overlapping weeks by design (ADR-001; MINDlarge's
test week is Nov 16-22, a different calendar week from dev) — different
users, different impressions, a genuinely different population, not a
resampling of the same one. Candidate G's win was built specifically
around MINDlarge-dev's warm/cold cohort mix and Candidate F's coefficients
fitted on MINDsmall-train's click patterns; both are properties of a
*specific* population slice that dev-vs-test disjointness gives no
guarantee will hold identically on a different week's users. Nothing in
this session's own verification (format checks, the `N89741` spot-check,
the paired-bootstrap methodology itself) surfaced a defect — the smaller
real-world margin is the honest result, not an artifact of a mistake, the
same conclusion this project's other real-test-set check reached under
directly comparable circumstances (see below).

**This is the same category of finding as EB-NeRD's contrastive-vector
submission (ADR-008's 2026-08-21 addendum), not a coincidence worth
dismissing.** There, a **+0.0023 CI-clear** local `ebnerd_small`-validation
win compressed to **+0.0005 mean AUC** on the real `ebnerd_testset`
leaderboard, with the day-to-day sign flipping three times across the
8 dates Codabench exposed — the qualitative direction held, the magnitude
did not transfer 1:1. MIND's result here is a step further in the same
direction: the sign itself didn't clearly hold (-0.0003, technically the
"wrong" direction, though far too small relative to Codabench's own
unreported noise floor to call it a real reversal rather than a wash).
Two independent real-test-set checks, on two different datasets, both
showing a local CI-clear win compress substantially at real blind-test
scale, is a real, project-level pattern worth naming plainly rather than
treating each occurrence as an isolated surprise: **a CI-clear win on a
local validation split, however rigorously tested, is evidence about that
split — not a guaranteed prediction of the real blind test set's
population**, especially for methods (like both of these) whose edge is
built around properties of the specific validation population (a cohort
mix, a small-model's learned coefficients) rather than a large, robust,
population-independent effect.

## Conditions for Revisiting

- This is now a two-for-two pattern in this project (EB-NeRD contrastive
  vector, MIND cohort-gated combiner) — if a third real-test-set
  comparison is ever made, it would meaningfully strengthen (or weaken)
  "local CI-clear wins compress at blind-test scale" as a general finding
  about this project's validation methodology, rather than two anecdotes.
- The `N` in "how much does a local win compress" is currently `N=2` — not
  enough to fit any quantitative relationship between local effect size
  and real-test effect size; only the qualitative direction (compression
  happens, doesn't reliably reverse but can get close to zero) is
  supported by the evidence so far.
- Codabench's Score column exposes no confidence interval for MIND (as
  already noted for EB-NeRD) — if a future submission's detail view
  exposes a per-day or per-slice breakdown the way EB-NeRD's did, re-check
  whether MIND's -0.0003 headline conceals a similar mixed-sign, thin-
  margin pattern underneath, rather than treating the single rounded
  number as the whole story.

## Related

- `submissions/mind_large_test_gated_cohort/prediction.zip` (896696)
- `docs/design_note.md`/`.tex` §3.5 (updated with this result)
- `decisions/ADR-008-semantic-retrieval-design.md`'s 2026-08-21 addendum
  ("Second EB-NeRD Submission: Contrastive Vector on the Real Test Set")
  — the directly comparable prior finding this addendum's interpretation
  draws on
- `knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`

---

**Forward pointer (2026-08-22):** the Candidate H addendum above's
"feature set is the ceiling, not the combiner's model class" finding was
tested directly as **Candidate I** — a neural candidate-aware attention
re-ranker operating on raw embeddings instead of precomputed scalars — in
its own dedicated ADR, `decisions/ADR-011-neural-attention-reranker.md`
(not an addendum here, since it's the first neural-training architecture
decision in the project and has its own real design space). Result: also
a real, CI-clear loss (-0.0107 paired vs. baseline), though smaller than
H2's and with open questions (an unconverged training curve) ADR-011
documents honestly rather than treating as a closed verdict.
