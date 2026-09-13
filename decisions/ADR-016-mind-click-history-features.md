# ADR-016 — A2 Q1: MIND Click-History Feature Engineering (Candidate L)

**Date:** 2026-09-13
**Status:** Decided, and evaluated — a real, CI-clear **loss** against the deployed
embedding baseline. Not adopted. Kept as an honest record, per this project's
established practice of reporting every candidate (A1's design note lists ten
losing MIND candidates alongside the two winners).
**Decision owner:** Engineer (sukhraj01). **Contributors:** Claude Sonnet 5 (design,
implementation, benchmarking, ADR drafting).

---

# Engineering Question

A2 Q1 asks for click-history and session features on both datasets: "user's recent
clicked articles (titles, categories, embeddings), click count, recency-weighted
history (exponential decay)" plus session features (within-session patterns,
position bias). EB-NeRD already has both halves — Candidate K's 65 features
(ADR-013) include long/short-term affinity and freshness, and this assignment's
session addition (`session_position`, `session_start_gap_h`, three `context_*`
match features) closed the session gap there. MIND had neither a recency-weighted
history feature nor a click-count feature; only unweighted symbolic matching
(`features.py`'s `category_match_score`/`subcategory_match_score`, Candidate D)
and frozen-embedding similarity existed.

The question this ADR answers: does adding MIND's missing piece of Q1's feature
list — recency-weighted history affinity plus click count, scored by a cheap
supervised combiner — carry real ranking signal on MIND, the way Candidate K's
freshness feature does on EB-NeRD?

Explicitly out of scope, per the engineer's own scoping instruction: this is
**engineered features only**, evaluated by a **CPU-only logistic-regression
combiner**, not a new NRMS training run — the MIND treatment for Q3 (title+abstract
input) already occupies the one Ada GPU this project has, and this work must not
contend with it.

---

# Context

## What already existed

| Signal | Where | Status before this ADR |
|---|---|---|
| Unweighted category/subcategory match | `features.py` (Candidate D) | Existed, unweighted (every history item counted equally regardless of position) |
| Entity overlap | `features.py`/`entities.py` | Existed |
| Frozen-embedding cosine similarity | `embed.py` | Existed (the deployed MIND baseline, AUC 0.6340) |
| Recency-weighted **embedding query** | `embed.py::build_user_embedding_query_recency` (Candidate C) | Existed, but as a *retrieval query* weighting, not a re-ranking *feature* — and it lost CI-clear for that use (ADR-005 addendum: 0.6265 vs. baseline 0.6340) |
| Recency-weighted **symbolic history feature** | — | **Did not exist.** This ADR's contribution. |
| Click count as an engineered statistic | — | Used only implicitly, as `COLD_THRESHOLD`'s cohort-assignment input, never as a named/tested feature or combiner input |

Candidate C's loss is a real, relevant prior result, but it tested a *different*
mechanism (recency-weighted mean-pooled embeddings replacing the unweighted query
entirely) for a *different* purpose (retrieval-query construction, not a re-ranking
feature alongside other signals). This ADR does not re-litigate Candidate C; it
tests recency weighting in the one form Q1 actually asks for and that has not been
tried: a **feature**, not a query-construction replacement.

## The dataset limitation, confirmed rather than assumed

Q1 also asks for freshness and session features. For MIND, neither is buildable,
and this is a genuine dataset limitation, not something overlooked. Confirmed by
direct measurement against the real processed data (not sampled, not inferred):

| Field | Null count | Total rows | Source of the hardcoding |
|---|---:|---:|---|
| `article_published_time` (articles) | 72,023 | 72,023 | `_parse_news_tsv`: `"article_published_time": pd.NaT` |
| `session_id` (impressions) | 14,085,557 | 14,085,557 | `_explode_impressions`: `df[col] = pd.array([None] * n, ...)` |
| `dwell_time` | 14,085,557 | 14,085,557 | same |
| `scroll_percentage` | 14,085,557 | 14,085,557 | same |
| `is_front_page` | 14,085,557 | 14,085,557 | same |

`src/datasets/mind.py` hardcodes all five to null at parse time because MIND's raw
`news.tsv`/`behaviors.tsv` never carry this information — there is no session
boundary marker, no dwell time, and no publish timestamp anywhere in the source
data. No amount of feature engineering recovers information the dataset never
recorded. This is stated in the design note (§1.1) as a limitation, not left as a
silent gap in the feature list.

---

# Design Space Explored

## Option 1 — Recency-weighted feature via decayed mean-pooled embeddings (rejected, already tried)

Reuse `build_user_embedding_query_recency` directly as a re-ranking *feature*
(cosine similarity between the candidate and the decayed user vector), rather than
as the sole retrieval query. **Rejected as redundant with Candidate F**, which
already includes `recency_embed_cos` (the same function, same decay=0.9) as one of
seven combiner features and found it did not carry the combiner to a win. Testing
the identical feature again in a smaller combiner would not answer a new question.

## Option 2 — Recency-weighted symbolic affinity (chosen)

Extend the *symbolic* (non-vector) match scores — `category_match_score` /
`subcategory_match_score` — with position-based decay, the genuinely untested
combination: recency weighting has been tried for embeddings (Option 1/Candidate
C), and unweighted symbolic matching has been tried (Candidate D), but recency
weighting *of* symbolic matching has not. Uses the exact same decay convention
(`decay ** (n-1-i)`, most-recent-last) as `build_user_embedding_query_recency`, so
this project has one recency definition for MIND, not two that could silently
drift apart.

## Option 3 — GBDT combiner instead of logistic regression

Rejected per the engineer's explicit scoping: "a cheap CPU-only logistic
regression combiner — not a new NRMS run." A GBDT is also unnecessary here — the
feature count (5) is small enough that a linear combiner is the right complexity,
and it keeps this candidate's cost near zero (no contention with the Ada GPU job).

---

# Decision

**Option 2.** Five features (`src/retrieval/mind_features.py::FEATURE_NAMES`):
`log_click_count`, `category_match`, `subcategory_match`,
`recency_category_affinity`, `recency_subcategory_affinity`. Trained on
MINDsmall-**train** (`StandardScaler` + `LogisticRegression`, `lbfgs`, untuned
defaults — same "no hidden tuning" stance as Candidate F/ADR-009), evaluated on
MINDsmall-**dev**, compared against the deployed embedding baseline (AUC 0.6340,
95% CI 0.6319–0.6361) via `paired_metric_diff_ci` (ADR-010's statistic, imported
not reimplemented) computed on the SAME impressions (both the combiner's score and
the baseline's embedding-cosine score are computed per impression, so the
comparison is paired, not merely two marginal CIs — a stronger test than most of
this project's earlier MIND candidates used).

## A modeling fact, verified rather than assumed

`log_click_count` is identical for every candidate within one impression: it
depends only on the user's history length, not on the candidate. A linear
combiner's contribution to a per-impression ranking metric (AUC, MRR, nDCG@K — all
functions of rank order only within that impression) is provably invariant to an
additive per-impression-constant term: shifting every candidate's score by the same
amount cannot change their relative order. The experiment script includes an
ablation (`_evaluate(..., zero_out_feature="log_click_count")`) that zeroes this
column of the scaled feature matrix and re-scores every dev impression, checking
the claim against the real fitted model rather than only asserting it from theory.
Click count is still reported as a feature, per Q1's explicit ask, with this
caveat attached rather than hidden — it may still matter for the *pointwise*
training objective (different users have different base click-rates, which
affects binary cross-entropy loss even though it cannot affect within-impression
ranking), which is a separate, legitimate reason to include it.

## Leakage

MIND provides no per-click timestamps (only a static, per-split, per-user
snapshot), so the EB-NeRD-style temporal-boundary test
(`test_ebnerd_no_impression_precedes_its_own_history_cutoff`) cannot be built for
MIND — this was already true and already documented (the existing
`@pytest.mark.skip`'d `test_mind_no_impression_precedes_its_own_history_cutoff`).
The structurally equivalent guard the dataset *does* support: since a user's
history is one fixed snapshot for the whole split (enforced at parse time by
`_build_user_history`'s invariant check, which raises on any per-user variation),
every history-derived feature must trace back to exactly one value regardless of
which of that user's real impressions it is computed for.

`tests/integration/test_leakage.py::test_mind_click_history_features_trace_to_one_shared_value_across_real_impressions`
verifies this against real MINDsmall-train data — not a synthetic call-the-pure-
function-twice check (which would prove only determinism, not leak-safety), but a
loop over a real user's actual multiple `(user_id, impression_id)` groups, checking
the click-history component of the feature vector against one ground-truth value
computed directly from the stored history, while confirming the candidate-derived
component (`category_match`) legitimately *varies* across the same loop — proving
the test is sensitive to real variation, not vacuously insensitive to everything.

---

# Results (2026-09-13, MINDsmall, real run)

**A real, CI-clear loss.** Trained on 5,843,444 MINDsmall-train candidate rows
(236,344 clicks), evaluated on all 73,152 MINDsmall-dev impressions.

| | AUC | 95% CI | MRR | nDCG@5 | nDCG@10 |
|---|---:|---|---:|---:|---:|
| Candidate L (overall) | **0.6176** | 0.6153–0.6197 | 0.3419 | 0.3251 | 0.3844 |
| warm (n=62,846) | 0.6288 | 0.6265–0.6313 | — | — | — |
| cold (n=10,306, 8,014 users) | 0.5488 | 0.5441–0.5535 | — | — | — |
| Deployed embedding baseline | 0.6340 | 0.6319–0.6361 | — | — | — |

**Paired, treatment (Candidate L) − baseline, same impressions:**

| Metric | Diff | 95% CI | Verdict |
|---|---:|---|---|
| AUC | −0.0164 | −0.0191, −0.0139 | CI-clear loss |
| MRR | −0.0067 | −0.0093, −0.0041 | CI-clear loss |
| nDCG@5 | −0.0063 | −0.0089, −0.0039 | CI-clear loss |
| nDCG@10 | −0.0058 | −0.0081, −0.0035 | CI-clear loss |

Every metric loses, CI-clear. Click-history statistics alone — with or without
recency weighting — do not carry enough per-candidate signal to beat frozen
mean-pooled embedding similarity on MIND. This is a real negative result, not
a tuning failure to be explained away: no hyperparameter search was run (same
"no hidden tuning" stance as Candidate F/ADR-009), and the loss is large enough
(CI half-width ~0.001–0.002 vs. a 0.016 gap) that tuning would not plausibly
close it.

**The click-count ablation confirms the theoretical claim exactly, not just
approximately:** zeroing `log_click_count` and re-scoring all 73,152 dev
impressions changed the maximum absolute AUC by **exactly 0.00e+00** — not "very
small," bit-for-bit zero, matching the proof that an additive per-impression
constant cannot move a within-impression rank order.

**Fitted coefficients** (on standardized features): `subcategory_match` 0.150
and `category_match` 0.094 dominate; `recency_subcategory_affinity` 0.054 is
meaningful; `recency_category_affinity` 0.002 is essentially zero;
`log_click_count` 0.064 is nonzero despite having zero ranking effect —
consistent with the documented mechanism (it can still reduce the *pointwise*
training loss, which differs across users, even though it cannot move any
single impression's internal ranking).

**Reading, not spun:** recency weighting was worth trying (a genuinely
untested combination, per the design-space discussion above) and it did not
help — the two new recency features carry the least weight of the five, and
the unweighted symbolic matches (already known, Candidate D) still dominate
what little signal exists. This is consistent with Candidate D's own earlier
result (0.613, CI-clear loss) and with Candidate F's finding that even a
7-feature combiner including embeddings and popularity only barely lost
overall (0.626) and won solely on the cold cohort. Five history-only features
without any embedding or popularity signal were always a weaker bet than
Candidate F's stack, and the result confirms that rather than surprising it.

Run cost: 775s wall (13 min), 1.13GB peak RSS, no GPU — cheap, as scoped.

**Decision: not adopted.** No further MIND candidate work is planned from this
line; the project's real MIND win remains Candidate J (NRMS-lite) and, per
A2 Q3, the reproduced-then-improved official NRMS.

---

# Affected Files

**New:** `src/retrieval/mind_features.py`, `tests/unit/test_mind_features.py`,
`scripts/run_mind_history_features_experiment.py`, this ADR.
**Modified:** `tests/integration/test_leakage.py` (new invariant test).
**Unmodified:** every existing MIND candidate (A–J), `features.py`, `embed.py`.

---

# Conditions for Revisiting

- [ ] If Candidate L wins CI-clear on any cohort, consider stacking it into
  Candidate F/G's combiner (same pattern as Candidate D's symbolic features were
  absorbed into F) rather than shipping it standalone.
- [ ] If MIND ever gains real per-click timestamps (would require a different
  official data release), replace position-based recency with true time-decay and
  re-test — the position-based assumption ("list is chronological, most-recent-
  last") remains unverified per ADR-002/ADR-005 and is inherited here, not
  re-argued.
