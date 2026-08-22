# AI Usage Log — 2026-08-22 — GBDT Supervised Combiner (Candidate H)

## Prompts (verbatim, in order)

1. > Current Objective: This project has never trained an actual supervised model on MIND's real click labels — every method so far (BM25, embeddings, hybrids, cohort gating) ranks by unsupervised similarity scores alone. Current classmates are scoring 0.65-0.70 on the same assignment; the likely gap is exactly this — build a real supervised ranker now.
>
> Train a lightweight supervised ranker (start with logistic regression, try LightGBM if time allows) on MINDsmall-train's real impression-level labels, using features already computed across this project: BM25 score, embedding cosine similarity, category/entity match, recency — plus anything else cheap and real, not leaky.
> Validate rigorously on MINDsmall-dev with the same bootstrap-CI discipline as everything else — report the real number against the current 0.634 baseline honestly.
> If it's a real, CI-clear win, scale it to MINDlarge the same way every other real result in this project has been scale-verified, then build a genuine improved submission.
> Move fast — this is the highest-value use of the remaining time, prioritize over further tuning of what's already been tried.

2. > try again

(This second prompt followed an `AskUserQuestion` tool call asking Claude
Code to retry a failed subagent research call, which had hit a transient
`UNKNOWN_CERTIFICATE_VERIFICATION_ERROR`. Interpreted as "retry the failed
exploration," not a new instruction.)

## Decision made via UI selection (not a free-text prompt)

Presented with a choice for the gradient-boosted-tree model: LightGBM (as
literally named in prompt 1) vs. scikit-learn's `HistGradientBoostingClassifier`
(same histogram-boosting family, zero new dependency, no macOS OpenMP
install risk). Engineer selected **scikit-learn `HistGradientBoostingClassifier`**.

## Notes on AI vs. human authorship

**Before any implementation:** Claude Code found that prompt 1's premise
("this project has never trained a supervised model on real labels") is
factually contradicted by the project's own record — Candidate F
(`scripts/run_learned_combiner_experiment.py`, ADR-010, 2026-08-21) already
trained a `LogisticRegression` combiner on MINDsmall-train's real click
labels using the same feature families named in prompt 1 (BM25, embedding
cosine, category/entity match, recency, plus popularity), and it lost
overall (AUC 0.6255 vs. baseline 0.6340, CI-clear) — a known negative
result, not an untried idea. This was surfaced to the engineer via text and
a written plan (`/Users/test01/.claude/plans/rosy-twirling-ripple.md`)
before any code was written, per CLAUDE.md's Decision Reversal and
Evidence Hierarchy sections — reusing an already-measured result instead
of silently re-running a known loss under a new name.

**Implementation (this session, human-directed via the approved plan, AI-generated code):**
`scripts/run_gbdt_combiner_experiment.py` (Candidate H: adds one feature —
`log_history_length` — to Candidate F's existing 7, trains two models —
`LogisticRegression(class_weight="balanced")` and
`HistGradientBoostingClassifier` — on MINDsmall-train, evaluates on
MINDsmall-dev via the project's existing Q4 harness and the paired
bootstrap test `run_gated_cohort_experiment.py::paired_metric_diff_ci`
already established as the statistically correct comparison against the
baseline), `tests/unit/test_gbdt_combiner.py`, an ADR-010 addendum, and a
`PROJECT_STATE.md` update — all reusing existing project infrastructure
(`_build_side`/`_impression_feature_matrix`/`FEATURE_NAMES` from Candidate
F, `ranking_metric_ci`/`rank_candidates`/`safe_auc` from the Q4 harness)
rather than duplicating it, per this project's "extend, don't duplicate"
convention.

While Candidate H was training in the background, a separate concurrent
thread (not this one) uploaded Candidate G's real second MIND Codabench
submission — found mid-session when writing the ADR-010 addendum and a
prior version of this file's edit conflicted on disk. Reconciled by
reading that thread's own addendum in full and correcting one now-stale
claim in this session's own addendum rather than overwriting it.

## Prompt 3 (same day, continuing this session)

> Current Objective: Build a real step-change for MIND — a lightweight neural re-ranker with candidate-aware attention pooling over history embeddings (the core NRMS/NAML mechanism), trained end-to-end on real click labels, using the already-computed MiniLM embeddings as frozen input features (not fine-tuning the text encoder itself — that's out of scope for the time left).
>
> Architecture: for each impression, attention-weight the user's history article embeddings by relevance to each candidate (a small learnable attention layer, not a fixed mean-pool), then score via a small MLP or dot product on top. Frozen embeddings as input, only the attention + scoring head is trained.
> Train on MINDsmall-train's real labels. Keep it small and fast to iterate — this needs to train in minutes/hours, not days, given the timeline.
> Validate on MINDsmall-dev with the same bootstrap-CI rigor as every other candidate. This is the checkpoint: if it doesn't show a real, meaningfully-larger win than what H1/H2 got (not just CI-clear, actually substantial — think closer to classmates' range, not +0.002), stop and report back rather than sinking more time into scaling it up.
> If it does show a real substantial win, scale to MINDlarge the same way everything else has been verified, then build the real submission.
> Move fast and report back at the MINDsmall-dev checkpoint before going further — I want to know within a day, not at the deadline, whether this is working.

Planned via a fresh Plan Mode cycle (`/Users/test01/.claude/plans/rosy-twirling-ripple.md`,
overwritten from the Candidate H plan since this is a different task) —
research confirmed this is the first neural-network training this project
has done (torch was previously used only for frozen `sentence-transformers`
inference). Implementation for Candidate I (`src/retrieval/rerank.py`,
`scripts/run_attention_reranker_experiment.py`, `tests/unit/test_rerank.py`,
`decisions/ADR-011-neural-attention-reranker.md`) is AI-generated, following
the approved plan; the architecture/training/evaluation design choices in
that plan (dot-product scoring over an MLP head, no-padding per-impression
training, BCE loss with a data-derived `pos_weight`, a train-internal
95/5 holdout for early stopping) were proposed by Claude Code and approved
by the engineer via the plan-mode review, not separately specified by the
engineer beyond the prompt above.

## Prompt 4 (same day, continuing this session)

> Current Objective: One more bounded check on Candidate I before accepting the result — was 5 epochs simply not enough, and does adding a popularity feature fix the cold-cohort gap?
>
> Re-run with meaningfully more epochs (e.g., 20-30) — same code, no redesign. Watch whether AUC crosses above 0.634 or plateaus below it. Report the full curve, not just the final number.
> If time allows in the same pass, add the popularity feature (train-split-only, same non-leaky construction as everywhere else in this project) as an input alongside the embeddings, since its absence was flagged as the likely cause of the cold-cohort miss.
> Same checkpoint discipline as before: validate on MINDsmall-dev, bootstrap CI, report back before touching MINDlarge or anything submission-related.
> Hard stop after this round regardless of outcome — if it's still a loss or a marginal win, report it plainly as the final state and don't start another candidate. Time left needs to go into finishing the design note and Q7 checklist either way.

Scoped tightly enough (explicit epoch range, one named feature, "same
code, no redesign", an explicit hard stop) that a fresh Plan Mode cycle
was judged unnecessary — implemented directly as a bounded extension of
Candidate I's existing architecture (`src/retrieval/rerank.py`,
`scripts/run_attention_reranker_experiment.py` both extended, not
replaced) and documented as an ADR-011 addendum. AI-generated: the popularity
integration design (a single learnable scalar weight added to the
existing dot-product logit, reusing `build_train_popularity` unchanged;
zero-history impressions get a real, popularity-only score instead of a
constant 0 when the feature is enabled, and are no longer skipped during
training in that case) was Claude Code's own implementation choice within
the engineer's stated scope, not separately specified.
