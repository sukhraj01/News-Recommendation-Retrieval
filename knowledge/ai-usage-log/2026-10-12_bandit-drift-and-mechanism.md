# Session 2026-10-12 (2): Drift hypothesis + mechanism analysis for the hybrid bandit

Branch: `bandit-drift-experiment` (from `bandit-offline-experiment`). Model: Claude Sonnet 5.5.

## Prompts (verbatim, in order)

### Prompt 1

Current Objective: Test the drift hypothesis for the hybrid bandit's shrinking lead (+0.0156 on held-out day 1 → +0.0024 [-0.0004, +0.0056] in the final 20%) with discounted-update variants, and diagnose what the learned correction is actually using. Branch from bandit-offline-experiment; do not touch the 0.7542 artifacts, submissions/, or the training pipeline. Report only measured results, including negative ones.

1. Re-read ADR-017 and the bandit code. Reuse the existing stream, features, frozen LightGBM score, held-out split (days 3-6), user-level bootstrap (n=2000), and the same tuning-days-only protocol. No new data access, no tuning on days 3-6.
2. Implement drift-aware variants of Hybrid LinUCB and Hybrid Thompson:
   a. Exponential forgetting: A <- gamma*A + x x^T, b <- gamma*b + r x (gamma tuned on days 0-2, grid e.g. 0.9999, 0.9995, 0.999, 0.995). Keep an O(d^2) update; derive it from Sherman-Morrison with the scaling, and unit-test it against the direct inverse.
   b. Sliding window of the last W impressions (or last H hours), implemented via rank-1 add and rank-1 remove, also unit-tested against the direct solve.
   c. Periodic reset to the prior, as a crude baseline.
3. Tune gamma/W/reset interval on days 0-2 only. Freeze, then run once on days 3-6. If winners land on a grid edge, widen the grid once and record that in the ADR, as before.
4. Report on days 3-6, each with user-level bootstrap CIs and paired deltas vs. LightGBM-greedy AND vs. the non-forgetting hybrid: overall precision@1, per-day precision@1 for days 3-6, and final-20% precision@1. State plainly whether forgetting recovers the final-window lead.
5. Drift diagnosis, which tests the hypothesis rather than assuming it:
   a. For the non-forgetting hybrid, log the learned weight vector at the end of each held-out day and report how much it moves (cosine similarity between consecutive days, and the top-10 features by absolute change).
   b. Report whether the LightGBM-greedy precision@1 itself changes across days 3-6 (a frozen model going stale), so the shrinking lead can be attributed to either the ranker getting worse, the bandit getting worse, or both.
   c. Slice the final-20% result by article_age_h bucket to check whether the lead disappears on fresh articles.
6. Mechanism analysis, since ADR-017 says about half the gain survives removing static article metadata and the mechanism is unknown: group the 100 linear features into groups (LightGBM score, freshness, user-history, article metadata, one-hot categoricals), then run group ablations (drop one group at a time, retrain on days 0-2 protocol, evaluate on 3-6) and report which groups carry the gain. If no group explains it, say so.
7. Latency check for the forgetting and windowed variants (p50/p99 of select+update) using the existing benchmark harness.
8. Add unit tests for all new update rules and the no-future-leakage test; run the full suite (expect all existing tests still pass).
9. Append the results to ADR-017 (or add ADR-018 if cleaner), including limitations, and update PROJECT_STATE.md. Commit and push to the same branch.
10. Finish with a short summary: did forgetting help, was the drift hypothesis supported, which feature group drives the gain, and 1-2 resume-ready sentences using only measured numbers.

## AI-generated vs. human-written

**AI-generated (Claude Sonnet 5.5), not yet reviewed by the engineer:** the drift-aware update rules and feature-mask/delay/hook
additions in `src/bandit/` (`policies.py`, `runner.py`, new `groups.py`), `tests/unit/test_bandit_drift.py`,
`scripts/run_drift_experiment.py`, `scripts/bandit_realtime_audit.py`, `scripts/drift_make_report.py`, the extended
`benchmarks/profile_bandit.py`, `decisions/ADR-018-bandit-drift-and-mechanism.md`, the addendum to ADR-017, the
PROJECT_STATE.md / ARCHITECTURE.md entries, and all numbers in `results/bandit_ebnerd_small/drift/` (machine-produced).
**Human-written:** none; the task (Prompt 1) is the engineer's.

**Judgement calls not specified by the engineer (review these):**
* Exponential forgetting deviates from the literal rule: the ridge prior is restored every 256 steps (otherwise A^-1 diverges in
  unobserved directions; demonstrated by a test).
* ADR-018 as a separate record rather than an append to ADR-017; ADR-017 got a pointer addendum.
* Tuned only the forgetting parameter and lambda (alpha, v, w held at ADR-017 winners); grid widened once, first pass kept.
* Added analyses the prompt did not ask for, after the large gain appeared (the "suspiciously good" rule carried over from the
  previous brief): label-permutation audit, real-time popularity baseline, feedback-delay sweep, a full-information transfer
  matrix, a follow-up group spec. The delay sweep and popularity baseline are post-hoc.
* The group ablation was run on both the non-forgetting and the forgetting hybrid.
* Branch: new `bandit-drift-experiment` created from `bandit-offline-experiment` (the prompt said both "branch from" and "same branch").

