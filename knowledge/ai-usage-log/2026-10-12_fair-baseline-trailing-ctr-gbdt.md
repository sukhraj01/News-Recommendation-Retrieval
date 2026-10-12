# Session 2026-10-12 (3): Fair baseline for ADR-018 — LightGBM retrained with causal trailing-CTR features

Branch: `bandit-fair-baseline` (from `bandit-drift-experiment`). Model: Claude Sonnet 5.5.

## Prompts (verbatim, in order)

### Prompt 1

Current Objective: Add the fair baseline for ADR-018: LightGBM retrained with causal trailing-CTR features, compared against the forgetting hybrid under identical delay assumptions. Branch from bandit-drift-experiment. No changes to 0.7542 artifacts, submissions/, or the training pipeline.

1. Build trailing article-CTR features (1h, 2h windows; plus smoothed CTR with a prior and an impression-count feature) computed strictly from labels with timestamp <= impression_time - D, where D is the feedback delay. Run D in {0, 15 min, 30 min, 60 min}. Add a test proving no label with a timestamp later than impression_time - D influences a feature.
2. Retrain LightGBM on the training period with these features, using the same hyperparameters and leakage-safe feature set as the existing ranker (no lifetime aggregates). Report stream AUC and top-1 precision on days 3-6 with the same user-level bootstrap (n=2000).
3. Compare, per D, on days 3-6: LightGBM-greedy (original), LightGBM + trailing CTR, popularity blend, forgetting hybrid LinUCB. Give paired deltas with CIs for each against the original and against the forgetting hybrid.
4. Report plainly whether the retrained GBDT matches, beats, or loses to the forgetting hybrid at each D.
5. Update ADR-018 with a new section, add the tests, run the full suite, commit and push. Use SKIP_PERF_SNAPSHOT only if the same unrelated stage flakes again, and record it.
6. Finish with a short summary and one resume-ready sentence using only measured numbers.

## AI-generated vs. human-written

**AI-generated (Claude Sonnet 5.5), not yet reviewed by the engineer:** `src/bandit/trailing.py`, `tests/unit/test_trailing.py`,
`scripts/trailing_ctr_gbdt.py`, `scripts/fair_baseline_compare.py`, `scripts/fair_baseline_audit.py`, `scripts/fair_baseline_report.py`,
ADR-018 Addendum 2 and the superseded-in-part note, the PROJECT_STATE.md entry, and every number in
`results/bandit_ebnerd_small/drift/fair_baseline*.json` / `trailing_gbdt_training.json` (machine-produced). **Human-written:** none; the task is the engineer's.

**Judgement calls not specified (review these):**
* Strict upper bound (`obs_time < t - D`) instead of the prompt's `<=`: with D=0 `<=` admits the row's own label.
* Six trailing features (CTR + count for 1h/2h, plus within-impression rank of each CTR); smoothing prior 0.1, strength 5 fixed a priori.
* Trailing statistics seeded from the end of train (one concatenated timeline); training windows and early stopping unchanged (last 24h of train).
* The forgetting hybrid and the popularity blend were RE-TUNED on days 0-2 under each D (so every contender is tuned under identical delay); this
  changed an ADR-018 number (60-minute row) and the ADR says so.
* Extra, unrequested: end-to-end causality audit on the trained models (permuted future labels, placebo features).
* Not run (flagged): a bandit on top of the retrained ranker.

