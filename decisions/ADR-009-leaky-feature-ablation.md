# ADR-009 — Q9 Anti-Gaming Ablation: Serving-Time-Unavailable Features

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

**Date:** 2026-08-14
**Status:** Decided
**Severity:** Medium

---

# Engineering Question

Q9 (the real assignment spec, only seen this session — see design note's
restructure) requires: *"Report metrics with and without features
unavailable at serving time."* ADR-007/ADR-008 already establish that this
project's actual retrieval/ranking code never reads such features (novelty's
popularity signal is train-split-only; BM25/embedding scores are computed
from article text and click history alone) — grepped and confirmed again
this session, `total_inviews`/`total_pageviews`/`total_read_time` (EB-NeRD
article-level lifetime aggregates) do not appear anywhere in
`src/datasets/ebnerd.py::parse_ebnerd_articles` or the unified schema
(`src/pipeline/schema.py`). But documenting *that a leak was avoided* is not
the same evidence as *measuring what the leak would have bought*, which is
what Q9 literally asks for. This ADR covers: how to construct the "with
leaky features" arm without touching the production pipeline, and what it
showed.

---

# Decision Scope

- [ ] Entire project
- [x] Experiment only
- [ ] Temporary / Prototype
- [x] Assignment-specific
- [ ] General reusable pattern

**Affected Components:** Evaluation (new standalone script only —
`src/datasets/ebnerd.py`, `src/pipeline/schema.py`, and every production
scorer are unmodified).

---

# Context

`ebnerd_small.zip`'s `articles.parquet` carries three fields the EB-NeRD
release computes over the article's entire observed lifetime:
`total_inviews`, `total_pageviews`, `total_read_time`. Confirmed directly
(`poetry run python` against the raw zip, this session): present for
20,738 articles, ~48% NaN (articles with too little exposure to have a
logged aggregate — not missing-at-random with respect to popularity, so
zero-filling would systematically understate low-exposure articles' leak
signal — see Design Space below). These are exactly the kind of feature Q9
is concerned with: computed from data that, for a given impression, may not
exist yet at serving time (an article's *total* pageviews includes clicks
that happen after the impression being scored), and — because they're
aggregated over exposure — directly correlated with future popularity in a
way a fair model shouldn't get to see.

---

# Decision Criteria

| Criteria | Importance | Notes |
|---|---|---|
| Never touch the production schema/pipeline | Critical | The leak must not become reachable by any real scoring path as a side effect of measuring it |
| Isolate the effect of the leak itself | High | Blend against the already-deployed method (embeddings), not a new model, so the delta is attributable to the leak alone |
| No hidden tuning | High | An optimally-tuned leaky model would overstate what a *careless* implementation leaks; an untuned blend is the more honest "what does exposure alone buy you" answer |
| Reuse the existing Q4 metric/CI machinery | Medium | Consistency with every other ranking number in this project (ADR-007) |

---

# Design Space Exploration

## Option 1 — Standalone ablation script, untuned 50/50 blend (chosen)

**How it works:** `scripts/run_leakage_ablation.py` reads
`total_inviews`/`total_pageviews`/`total_read_time` directly from the raw
zip (never through `src/datasets/ebnerd.py`), builds one leak score per
article (mean of per-field min-max-normalized `log1p`, NaN imputed with
that field's own corpus median — not zero, see Context), and for each
impression blends it 50/50 with the existing `EmbeddingScorer` score
(both min-max-normalized within the impression first, so the blend isn't
dominated by whichever raw scale happens to be larger). Runs the identical
per-impression AUC/MRR/nDCG@5/nDCG@10 + bootstrap-CI harness ADR-007 built,
side-by-side for "without" (unmodified embedding score) and "with"
(blended score).

**Optimizes for:** Isolation (leak lives only in this script) and honesty
(no weight tuning to search for the largest possible gap).

**Sacrifices:** Not a calibrated estimate of "the best possible leaky
model" — a logistic-regression-calibrated blend would likely show a larger
gap. Deliberately not built (see Rationale).

**Assumptions:** Median imputation for the ~48% NaN rows is a reasonable
default, not a claim that missingness is random (it plausibly correlates
with low exposure, i.e. these articles' *true* leak signal is probably
below the median, meaning this ablation likely **understates** the leak's
real effect, not overstates it — a conservative bias, the right direction
for an anti-gaming argument).

**Complexity:** Low — one new script, two new pure functions
(`_minmax`, `_compute_leak_scores`), reuses every existing metric/scorer.

**Maturity:** N/A (project-internal ablation, not a reusable pattern).

## Option 2 — Popularity-only baseline (no blend with the retrieval score)

**How it works:** Score every candidate by the leak signal alone, ignoring
the retrieval method entirely.

**Sacrifices:** Doesn't answer Q9's actual question — Q9 asks what these
features would do *to the deployed model's metrics*, not what a
pure-popularity baseline scores on its own. A popularity-only number is a
different (also valid, but different) question. Rejected as not matching
the assignment wording.

## Option 3 — Add the fields to the unified schema as optional columns, gate their use behind a flag

**How it works:** Extend `src/pipeline/schema.py`'s EB-NeRD-optional
fields (same mechanism as `read_times`, ADR-002) with the three leak
fields, add a `use_leaky_features` flag to `BM25Scorer`/`EmbeddingScorer`.

**Sacrifices:** Makes the leak *reachable* from the production schema and
every real scorer, permanently, for the sake of one ablation — exactly the
risk Decision Criteria #1 rules out. A future session (or a careless
`generate_ebnerd_predictions.py` flag default) could accidentally wire it
into a real Codabench submission. Rejected: the measurement doesn't need
to live in the same code path as production scoring, and shouldn't.

---

# Comparison Summary

| Option | Answers Q9 | Leak reachable from production | Complexity | Decision |
|---|---|---|---|---|
| 1 — standalone script, untuned blend | Yes | No | Low | ✅ |
| 2 — popularity-only baseline | No (different question) | No | Low | |
| 3 — schema + flag | Yes | Yes (permanently) | Medium | |

---

# Final Decision

## Chosen Option

**Option 1** — standalone ablation script, untuned 50/50 blend against the
deployed embedding model, on `ebnerd_small` validation (the same
split/method ADR-008's leaderboard-cited numbers use, so the "without" arm
is a direct consistency check against already-published numbers).

### Decision Date

2026-08-14

---

# Rationale

## Why this option is best

- Directly answers Q9's literal wording (with vs. without, on the model
  actually used for both leaderboard submissions).
- Zero production-code risk — verified by grep, not just design intent,
  that no scorer or schema file changed.
- The "without" arm reproducing the existing published number exactly
  (0.5430 AUC, see Evidence) is a real correctness check on the harness
  reuse, not just a convenient coincidence.

## Why alternatives were rejected

### Option 2
Doesn't answer the question Q9 asks (with/without the *deployed* model's
metrics, not a new baseline's).

### Option 3
Trades a one-time measurement for a permanent increase in the blast radius
of a future mistake. The cost (leak becomes a reachable code path) is
paid indefinitely for a benefit (schema-level "realism") the ablation
doesn't need — this measurement never needed to be reachable from
`generate_ebnerd_predictions.py` in the first place.

---

# Decision Confidence

**Current Confidence:** Medium-High

### Why
- The direction and non-overlap of the effect (leak helps, and by a CI-
  clear margin) is solid evidence, not noise — see Evidence below.
- The *magnitude* is specific to an untuned 50/50 blend; a different
  weighting or a calibrated model would show a different (plausibly
  larger, see Option 1's Assumptions) number. This ADR does not claim
  0.5662 is *the* ceiling, only *a* real, measured lower bound.

### What Would Increase Confidence
- Repeating on `ebnerd_demo` or MINDlarge-scale-equivalent EB-NeRD bundle
  (not available locally without a Kaggle detour, out of scope this
  session) to check the effect isn't small-sample-specific.
- A calibrated (e.g. logistic-regression-weighted) blend, to bound the
  leak's *maximum* plausible impact rather than only its untuned one.

---

# Evidence

## Evidence Sources

- [x] Assignment Requirements (Q9, verbatim spec pasted this session)
- [x] Experimental Benchmark
- [ ] Official Documentation
- [ ] Research Paper
- [ ] Industry Practice
- [ ] Community Consensus
- [x] Engineering Inference (median-imputation conservative-bias argument)

## Empirical Evidence

Experiment: `experiments/ablation_leaky_features_ebnerd_small_2026-08-14/`
(`config.json`, `results.json`). `ebnerd_small` validation, embed method,
244,647 impressions, 15,342 users, 2000-resample bootstrap 95% CI, seed 0.

| Metric | Without leaky features | With leaky features (50/50 blend) | CI overlap? |
|---|---|---|---|
| AUC | 0.5430 (0.5415–0.5445) | 0.5662 (0.5646–0.5678) | No |
| MRR | 0.3437 (0.3421–0.3452) | 0.3638 (0.3623–0.3653) | No |
| nDCG@5 | 0.3804 (0.3784–0.3823) | 0.4034 (0.4016–0.4051) | No |
| nDCG@10 | 0.4591 (0.4574–0.4607) | 0.4783 (0.4767–0.4799) | No |

The "without" row's AUC (0.5430) matches
`experiments/ranking_embed_ebnerd_small_2026-08-10/results.json` exactly —
the two independent runs (five days apart, this session vs. the original
Q4 harness run) agree, confirming the ablation's reuse of the scoring/eval
code path is faithful, not a divergent reimplementation.

## Interpretation

Even an untuned, equal-weight blend of three serving-time-unavailable
aggregate fields moves every ranking metric by a CI-clear margin (AUC
+0.023 absolute, ~4.3% relative; nDCG@10 +0.019 absolute). None of the
four 95% CIs overlap between arms — this is a real, not noise-level,
effect. Because missingness (~48% of articles) was imputed with the
column median rather than zero, and low-exposure articles plausibly have
below-median true popularity, this measured gap is more likely a
conservative *underestimate* of the leak's real effect than an
overestimate (see Decision Confidence). This is direct, measured
justification — not just a design-intent claim — for why ADR-007's
novelty popularity signal is train-split-only and why these three fields
are absent from the unified schema entirely: a metric built on them would
look meaningfully better while measuring something the system could not
have known at serving time.

---

# Assumptions

| Assumption | Why It Matters | Monitoring Strategy |
|---|---|---|
| Median imputation for missing leak fields approximates, doesn't invert, the true (unobserved) leak signal | If wrong-direction, the ~48%-NaN rows could be overstating rather than understating the effect | Would need EB-NeRD's own exposure-vs-popularity documentation to verify directly; not done this session |
| 50/50 blend weight is representative enough to demonstrate *a* real effect, even if not *the maximal* one | The magnitude, not just the existence, of the reported numbers depends on this | Flagged in Decision Confidence as the main revisit trigger |

---

# Conditions for Revisiting

Revisit if:
- [ ] A calibrated (not untuned) leaky-feature blend is needed to bound
  the maximum plausible gaming effect, not just demonstrate a real one.
- [ ] The same ablation is required for MIND (not applicable — MIND's
  schema carries no equivalent lifetime-aggregate fields; confirmed by
  the same grep this session found nothing in `src/datasets/mind.py`).
- [ ] Assignment requirements around Q9 change.

**Estimated Cost to Change:** Easy — the script is ~150 lines, isolated,
and the blend weight is a single named constant.

---

# Related Decisions

## Influenced By
- ADR-007 (reuses its metric/CI machinery unchanged)
- ADR-008 (ablates against the exact model both leaderboard submissions used)

## Influences
- `docs/design_note.md`'s Anti-Gaming / Q9 section

---

# References

## Internal
- `scripts/run_leakage_ablation.py`
- `tests/unit/test_leakage_ablation.py`
- `experiments/ablation_leaky_features_ebnerd_small_2026-08-14/{config,results}.json`
- ADR-007, ADR-008

---

# Decision History

| Date | Event |
|---|---|
| 2026-08-14 | Real Q9 spec seen for the first time; gap identified (documentation of leak-avoidance existed, with/without measurement did not) |
| 2026-08-14 | Design space explored, Option 1 chosen, script + tests written |
| 2026-08-14 | Ablation run against `ebnerd_small` validation, results recorded above |
| 2026-08-14 | Decided |

---

# Notes

This ablation was triggered by receiving the assignment's real Q7-Q9 text
for the first time this session — `docs/design_note.md`'s prior version
(deleted editorial comment) had inferred the note's structure from ADR
numbering alone, without the literal spec. Q9's exact wording ("with and
without") was more specific than this project's prior anti-gaming posture
(exclusion + documentation only) had satisfied. Per CLAUDE.md's Decision
Reversal principle, this is new evidence changing a prior implicit
assumption ("documenting the exclusion is sufficient"), not a mistake in
ADR-007's original design — ADR-007's exclusion decision was and remains
correct; what changed is that its *justification* now has a measured
number behind it instead of only a stated rule.
