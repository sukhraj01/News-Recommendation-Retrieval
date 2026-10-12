# Session 2026-10-12: Offline contextual-bandit layer on the EB-NeRD LightGBM ranker

Branch at session start: `a2-q3-official-baseline`. Model: Claude Sonnet 5.5.

## Prompts (verbatim, in order)

### Prompt 1

Current Objective: Build and evaluate an offline contextual-bandit layer on top of the EB-NeRD LightGBM ranker (AUC 0.7542 baseline stays untouched). Deliver measured results, tests, an ADR, and resume-ready numbers. Time-box: ~6 hours total. Do not modify existing leaderboard artifacts, submissions, or the 0.7542 pipeline.

PHASE 0: Orientation (15 min)
1. Read PROJECT_STATE.md, the EB-NeRD feature builder (65 features), the temporal split code, the bootstrap CI utilities, and ADR-011 to ADR-015. Reuse all of them; no re-implementation.
2. Report back (short): the exact feature columns, how candidates per impression are stored (article_ids_inview, clicked list), whether in-view order reflects display position, the split dates, and the dataset size in use. If display position is not recoverable, say so; it determines Step 8.

PHASE 1: Problem formulation (write this into the ADR first)
3. Formulate as a supervised-to-bandit conversion: for each impression in chronological order, the context is the user plus the in-view candidate set. The bandit picks ONE article (arm) and only observes the reward for that pick (1 if clicked, 0 otherwise). It never sees the other labels, so learning is genuinely bandit-feedback. Document the exposure-bias caveat: labels exist only for in-view items, and the production logging policy was not random.
4. Evaluation protocol is prequential: process impressions in time order; the bandit scores, picks, receives the reward, and updates. Update frequency is configurable (per impression and per 1-hour batch, as the latter is closer to production). No update uses any information from the future. Add a test that proves this.

PHASE 2: Implementation (src/bandit/)
5. Policies, all sharing a common interface (select(context, candidates) -> arm, update(arm, reward)):
   a. Random.
   b. Popularity-greedy (trailing-window CTR).
   c. LightGBM-greedy (frozen ranker from the existing pipeline; the strong static baseline).
   d. Epsilon-greedy over the LightGBM score.
   e. LinUCB with a SHARED parameter vector over joint (user, article, context) features (disjoint per-arm models do not scale to the article count; justify this in the ADR). Use Sherman-Morrison rank-1 updates for the inverse so each update is O(d²) with d≈65.
   f. Linear Thompson sampling (Gaussian posterior, same features).
   g. Hybrid warm-start: LinUCB/Thompson where the LightGBM score is an input feature (or the prior mean), so the bandit learns a residual correction on top of the ranker. This is the production-realistic variant.
6. Hyperparameters (alpha for UCB, posterior variance, epsilon, regularization, update interval) are tuned on a validation period ONLY, then frozen and run once on the held-out period. No tuning on the final period.
7. Leakage guard: the bandit features must be the existing leakage-safe set. Do not use the lifetime aggregate fields (total_inviews / total_pageviews / total_read_time) that were excluded earlier. Add an assertion in the feature path.

PHASE 3: Evaluation
8. Primary metrics on the held-out period, each with bootstrap 95% CI (n=2000; resample by user or by day blocks, not by impression, to respect within-user correlation; state which): top-1 CTR (cumulative reward per impression), cumulative regret vs. the oracle (oracle picks a clicked article when one exists), and the final-window CTR (last 20% of the stream) to show learning. Also report the paired bootstrap difference vs. LightGBM-greedy for each bandit.
9. Slices: (a) cold-start articles (fewer than N prior impressions; choose N from corpus_stats and state it), (b) cold-start users (history < 5), (c) the first vs. last day of the stream (drift/freshness behaviour, given article_age_h is the dominant feature), (d) high vs. low candidate-set size.
10. Exploration cost: report how often each bandit's pick differs from the LightGBM top-1 and the CTR of those differing picks vs. the LightGBM pick on the same impressions. This answers "does exploration pay for itself?"
11. Secondary robustness check: build a click simulator from the trained LightGBM click model (probabilities on the held-out set), run all policies against it with synthetic Bernoulli rewards and multiple seeds, and report mean±std cumulative regret. Label the simulator results clearly as simulated, separate from the log-replay results.
12. Optional secondary counterfactual estimate: SNIPS with clipped weights using a propensity model fitted on the logging data (or position-based if Step 2 shows order is display position). Include only if the propensities are defensible; otherwise write one paragraph on why it was skipped.

PHASE 4: Engineering quality
13. Unit tests: determinism under fixed seed; Sherman-Morrison inverse equals the direct inverse within tolerance; no-future-leakage test (shuffled-future labels must not change past decisions); reward revealed only for the chosen arm; shape and NaN checks. Run the full test suite, including the existing tests.
14. Profile: per-decision latency (p50/p99) and throughput for LinUCB/Thompson vs. LightGBM-greedy, using the benchmarks/ harness and the existing format. Report the update cost too.
15. Run a small-subset dry run first, then the full run. If any policy is unstable or results look suspiciously good (e.g., a bandit beating LightGBM-greedy by a large margin), audit for leakage before reporting.

PHASE 5: Documentation and delivery
16. Write docs/ADR-016_bandit_experiment.md: formulation, protocol, policies, hyperparameter selection procedure, all results tables with CIs, slices, exploration cost, simulator results, latency, limitations (exposure bias, non-random logging, label = click only, no real-time feedback), and what an online A/B test would need (traffic split, guardrail metrics: diversity, novelty, coverage, latency). Report every result as measured, including any where the bandits lose to LightGBM-greedy, with your best explanation of why.
17. Update PROJECT_STATE.md, then commit and push to the existing repo.
18. Finish with a summary: a results table, the three most interesting findings, and 2-3 resume bullets that use ONLY numbers measured in this run.

(The prompt was pasted by the engineer inside a `pasted_content` block, together with the CLAUDE.md and memory context the harness attaches automatically.)

## AI-generated vs. human-written

**AI-generated (Claude Sonnet 5.5, this session), all unreviewed by the engineer at time of writing:**
`src/bandit/*` (features, stream, policies, runner, evaluation), `tests/unit/test_bandit.py`,
`scripts/bandit_prepare_stream.py`, `scripts/run_bandit_experiment.py`, `scripts/bandit_leakage_audit.py`,
`scripts/bandit_mechanism_ablation.py`, `scripts/bandit_make_report.py`, `benchmarks/profile_bandit.py`,
`decisions/ADR-017-bandit-experiment.md`, the PROJECT_STATE.md / ARCHITECTURE.md additions, and every
number in `results/bandit_ebnerd_small/` (machine-produced by those scripts; the ADR's tables are rendered
from the JSON by script, the prose findings were written by the AI).

**Human-written / human-edited:** nothing in this session. The task specification (Prompt 1) is the
engineer's. Reused unchanged from earlier human+AI work: the 65-feature builder, the user-bootstrap
substrate (`src/evaluation/bootstrap.py`), the benchmark harness (`benchmarks/timing.py`), and Candidate K's
frozen boosters.

**Judgement calls the AI made that the engineer did not specify (review these):**
ADR numbered 017 in `decisions/` instead of `docs/ADR-016_bandit_experiment.md` (016 was taken); tuning =
stream days 0-2 / held-out = days 3-6 with fresh state; cold-article N = 100; cold-user slice substituted
(history < 20) because history < 5 is empty; d0 = 100 not ~65 (one-hot expansion); grid widened once after
boundary winners; SNIPS skipped; simulator built from the sibling K_cls booster; extra audits/ablation added.

