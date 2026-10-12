# Performance Log

Rolling record of latency/throughput snapshots, one entry per run of
`benchmarks/snapshot.py`. Written automatically — do not hand-edit entries.

**What these numbers are.** A cheap regression tripwire measured on
**MINDsmall-dev**, not the authoritative full-scale profile. They are meaningful
as a *change* against the previous entry with the same hardware and config; they
are NOT the numbers ADR-014 reports, which come from `profile_mind.py` /
`profile_ebnerd.py` / `ablations.py` at real MINDlarge / ebnerd_small scale.

**The habit.** Any commit touching a performance-relevant path
(`src/retrieval/`, `src/evaluation/ranking_metrics.py`, `src/datasets/`,
`benchmarks/`, `scripts/run_*`, `scripts/generate_*`) gets a before/after pair:

```bash
# before the change
poetry run python benchmarks/snapshot.py --note "before: <what you're about to do>"
# after the change
poetry run python benchmarks/snapshot.py --compare --note "after: <what you did>"
```

`--compare` exits nonzero on a regression past the threshold, so it can gate a
commit rather than merely inform one. See ADR-014 for the methodology and for
why a regression here is worth blocking on.

---

### 2026-09-03T23:56:45 — `ccdebd2` (dirty tree)
**Hardware:** Mac15,12 / 8GB · **Config:** MIND small/dev, 60 users, 100 impressions, 42,416 articles
**Note:** smoke test: baseline

**End-to-end:** 4.8466 ms/impression (206.3 impressions/s) · peak RSS 0.39 GB

| Stage | ms/impression | throughput (calls/s) |
|---|---:|---:|
| `query_tokenize` | 0.2251 | 2,666 |
| `bm25_score_all` | 1.0511 | 571 |
| `embed_query_build` | 0.0432 | 13,893 |
| `embed_score_all` | 1.7971 | 334 |
| `bm25_candidate_lookup` | 0.0391 | 25,578 |
| `embed_candidate_lookup` | 0.0158 | 63,484 |
| `rank_and_auc` | 1.2953 | 772 |

_(no comparable prior snapshot — recorded as a baseline)_

### 2026-09-03T23:56:58 — `ccdebd2` (dirty tree)
**Hardware:** Mac15,12 / 8GB · **Config:** MIND small/dev, 60 users, 100 impressions, 42,416 articles
**Note:** smoke test: compare

**End-to-end:** 4.8358 ms/impression (206.8 impressions/s) · peak RSS 0.43 GB

Compared against `2026-09-03T23-56-45_snapshot.json` (`ccdebd2`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.2251 | 0.2242 | -0.4% | · |
| `bm25_score_all` | 1.0511 | 1.1618 | +10.5% | · |
| `embed_query_build` | 0.0432 | 0.0344 | -20.4% | 🟢 |
| `embed_score_all` | 1.7971 | 1.7862 | -0.6% | · |
| `bm25_candidate_lookup` | 0.0391 | 0.0217 | -44.5% | 🟢 |
| `embed_candidate_lookup` | 0.0158 | 0.0137 | -13.3% | · |
| `rank_and_auc` | 1.2953 | 1.1899 | -8.1% | · |

**Verdict:** no regression

### 2026-09-07T02:56:56 — `8d3e651` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: .gitignore benchmarks/README.md benchmarks/__init__.py benchmarks/ablations.py benchmarks/ada_perf_prep.sbatch benchmarks/ada_perf_profile.sbatch benchmarks/ada_perf_resume.sbatch benchmarks/ada_reconcile.sbatch benchmarks/loaders.py benchmarks/profile_ebnerd.py benchmarks/profile_mind.py benchmarks/reconcile_adr006.py benchmarks/results/2026-09-03T23-17-27_mind_large_dev_profile.json benchmarks/results/2026-09-03T23-17-51_mind_large_dev_profile.json benchmarks/results/2026-09-03T23-19-48_mind_large_dev_profile.json benchmarks/results/ada/2026-09-04T00-26-56_mind_large_dev_profile.json benchmarks/results/ada/2026-09-04T00-29-26_mind_large_dev_profile.json benchmarks/results/ada/2026-09-04T00-32-54_ebnerd_small_validation_profile.json benchmarks/results/ada/2026-09-04T00-40-09_performance_ablations.json benchmarks/results/ada/2026-09-04T00-40-39_performance_ablations.json benchmarks/results/ada/2026-09-04T00-43-04_performance_ablations.json benchmarks/results/ada/2026-09-04T00-45-56_reconcile_adr006.json benchmarks/results/ada/perf-profile_2687421.err benchmarks/results/ada/perf-profile_2687421.out benchmarks/results/ada/perf-resume_2687432.err benchmarks/results/ada/perf-resume_2687432.out benchmarks/snapshot.py benchmarks/timing.py 

**End-to-end:** 10.7329 ms/impression (93.2 impressions/s) · peak RSS 0.36 GB

| Stage | ms/impression | throughput (calls/s) |
|---|---:|---:|
| `query_tokenize` | 0.4394 | 1,665 |
| `bm25_score_all` | 2.6559 | 276 |
| `embed_query_build` | 0.2509 | 2,917 |
| `embed_score_all` | 3.5336 | 207 |
| `bm25_candidate_lookup` | 0.1231 | 8,124 |
| `embed_candidate_lookup` | 0.0387 | 25,839 |
| `rank_and_auc` | 2.4517 | 408 |

_(no comparable prior snapshot — recorded as a baseline)_

### 2026-09-12T12:33:04 — `62aace6` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: scripts/a2_ebnerd_prepare_tokens.py scripts/a2_nrms_official.sbatch scripts/a2_nrms_official_run.py src/retrieval/nrms_official.py src/retrieval/nrms_official_data.py tests/unit/test_nrms_official.py tests/unit/test_nrms_official_data.py 

**End-to-end:** 11.7251 ms/impression (85.3 impressions/s) · peak RSS 0.43 GB

Compared against `2026-09-07T02-56-56_snapshot.json` (`8d3e651`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.4394 | 0.2362 | -46.2% | ~ |
| `bm25_score_all` | 2.6559 | 2.0896 | -21.3% | 🟢 |
| `embed_query_build` | 0.2509 | 0.0763 | -69.6% | ~ |
| `embed_score_all` | 3.5336 | 6.3217 | +78.9% | 🔴 |
| `bm25_candidate_lookup` | 0.1231 | 0.0349 | -71.6% | ~ |
| `embed_candidate_lookup` | 0.0387 | 0.0491 | +26.9% | ~ |
| `rank_and_auc` | 2.4517 | 2.1617 | -11.8% | · |

**Verdict:** REGRESSION DETECTED

### 2026-09-12T12:33:17 — `62aace6` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: scripts/a2_ebnerd_prepare_tokens.py scripts/a2_evaluate_scores.py scripts/a2_nrms_official.sbatch scripts/a2_nrms_official_run.py src/retrieval/nrms_official.py src/retrieval/nrms_official_data.py tests/unit/test_a2_evaluate_scores.py tests/unit/test_nrms_official.py tests/unit/test_nrms_official_data.py 

**End-to-end:** 11.3103 ms/impression (88.4 impressions/s) · peak RSS 0.5 GB

Compared against `2026-09-12T12-33-04_snapshot.json` (`62aace6`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.2362 | 0.223 | -5.6% | · |
| `bm25_score_all` | 2.0896 | 2.3137 | +10.7% | · |
| `embed_query_build` | 0.0763 | 0.097 | +27.1% | ~ |
| `embed_score_all` | 6.3217 | 5.3968 | -14.6% | · |
| `bm25_candidate_lookup` | 0.0349 | 0.0664 | +90.3% | ~ |
| `embed_candidate_lookup` | 0.0491 | 0.0373 | -24.0% | ~ |
| `rank_and_auc` | 2.1617 | 2.5818 | +19.4% | 🔴 |

**Verdict:** REGRESSION DETECTED

### 2026-09-12T12:33:29 — `62aace6` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: knowledge/ai-usage-log/2026-09-11_a2-kickoff-corpus-stats-gap-analysis.md scripts/a2_ebnerd_prepare_tokens.py scripts/a2_evaluate_scores.py scripts/a2_nrms_official.sbatch scripts/a2_nrms_official_run.py src/retrieval/nrms_official.py src/retrieval/nrms_official_data.py tests/unit/test_a2_evaluate_scores.py tests/unit/test_nrms_official.py tests/unit/test_nrms_official_data.py 

**End-to-end:** 10.9217 ms/impression (91.6 impressions/s) · peak RSS 0.47 GB

Compared against `2026-09-12T12-33-17_snapshot.json` (`62aace6`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.223 | 0.233 | +4.5% | · |
| `bm25_score_all` | 2.3137 | 1.9667 | -15.0% | · |
| `embed_query_build` | 0.097 | 0.0453 | -53.3% | ~ |
| `embed_score_all` | 5.3968 | 6.0037 | +11.2% | · |
| `bm25_candidate_lookup` | 0.0664 | 0.0255 | -61.6% | ~ |
| `embed_candidate_lookup` | 0.0373 | 0.0273 | -26.8% | ~ |
| `rank_and_auc` | 2.5818 | 1.9546 | -24.3% | 🟢 |

**Verdict:** no regression

### 2026-09-12T12:40:20 — `5e5d1fb` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** clean serial snapshot after A2 Option B commits (machine idle)

**End-to-end:** 4.3556 ms/impression (229.6 impressions/s) · peak RSS 0.7 GB

Compared against `2026-09-12T12-33-29_snapshot.json` (`62aace6`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.233 | 0.1578 | -32.3% | ~ |
| `bm25_score_all` | 1.9667 | 1.132 | -42.4% | 🟢 |
| `embed_query_build` | 0.0453 | 0.0309 | -31.8% | ~ |
| `embed_score_all` | 6.0037 | 1.2136 | -79.8% | 🟢 |
| `bm25_candidate_lookup` | 0.0255 | 0.0173 | -32.2% | ~ |
| `embed_candidate_lookup` | 0.0273 | 0.0179 | -34.4% | ~ |
| `rank_and_auc` | 1.9546 | 1.547 | -20.9% | 🟢 |

**Verdict:** no regression

### 2026-09-12T17:40:36 — `e5aed04` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: PROJECT_STATE.md benchmarks/cost_qps.py benchmarks/profile_ebnerd.py benchmarks/profile_mind.py benchmarks/results/2026-09-12T17-08-37_ebnerd_small_validation_profile.json benchmarks/results/2026-09-12T17-09-25_ebnerd_small_validation_profile.json benchmarks/results/2026-09-12T17-37-08_mind_large_dev_profile.json benchmarks/results/2026-09-12T17-37-21_cost_qps.json decisions/ADR-014-performance-benchmarking-methodology.md 

**End-to-end:** 3.5361 ms/impression (282.8 impressions/s) · peak RSS 0.55 GB

Compared against `2026-09-12T12-40-20_snapshot.json` (`5e5d1fb`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1578 | 0.15 | -4.9% | · |
| `bm25_score_all` | 1.132 | 1.049 | -7.3% | · |
| `embed_query_build` | 0.0309 | 0.0322 | +4.2% | · |
| `embed_score_all` | 1.2136 | 0.9133 | -24.7% | 🟢 |
| `bm25_candidate_lookup` | 0.0173 | 0.0167 | -3.5% | · |
| `embed_candidate_lookup` | 0.0179 | 0.0125 | -30.2% | ~ |
| `rank_and_auc` | 1.547 | 1.1137 | -28.0% | 🟢 |

**Verdict:** no regression

### 2026-09-12T21:52:05 — `7fa8bce` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: PROJECT_STATE.md decisions/ADR-013-ebnerd-gbdt-ranker.md results/a2_q9/README.md results/a2_q9/q9_ablation_config.json results/a2_q9/q9_ablation_results.json scripts/run_ebnerd_gbdt_experiment.py 

**End-to-end:** 3.6537 ms/impression (273.7 impressions/s) · peak RSS 0.55 GB

Compared against `2026-09-12T17-40-36_snapshot.json` (`e5aed04`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.15 | 0.1424 | -5.1% | · |
| `bm25_score_all` | 1.049 | 1.0577 | +0.8% | · |
| `embed_query_build` | 0.0322 | 0.0318 | -1.2% | · |
| `embed_score_all` | 0.9133 | 1.0383 | +13.7% | · |
| `bm25_candidate_lookup` | 0.0167 | 0.017 | +1.8% | · |
| `embed_candidate_lookup` | 0.0125 | 0.013 | +4.0% | · |
| `rank_and_auc` | 1.1137 | 1.135 | +1.9% | · |

**Verdict:** no regression

### 2026-09-13T07:38:43 — `6d15e50` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: decisions/ADR-016-mind-click-history-features.md scripts/run_mind_history_features_experiment.py src/retrieval/mind_features.py tests/integration/test_leakage.py tests/unit/test_mind_features.py 

**End-to-end:** 3.5795 ms/impression (279.4 impressions/s) · peak RSS 0.6 GB

Compared against `2026-09-12T21-52-05_snapshot.json` (`7fa8bce`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1424 | 0.148 | +3.9% | · |
| `bm25_score_all` | 1.0577 | 1.0729 | +1.4% | · |
| `embed_query_build` | 0.0318 | 0.0354 | +11.3% | · |
| `embed_score_all` | 1.0383 | 1.0134 | -2.4% | · |
| `bm25_candidate_lookup` | 0.017 | 0.0176 | +3.5% | · |
| `embed_candidate_lookup` | 0.013 | 0.0131 | +0.8% | · |
| `rank_and_auc` | 1.135 | 1.0593 | -6.7% | · |

**Verdict:** no regression

### 2026-09-14T02:42:54 — `13564ff` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: PROJECT_STATE.md benchmarks/ada_mind_gpu_per_request.sbatch decisions/ADR-015-a2-official-baseline-reproduction.md docs/design_note_a2.pdf docs/design_note_a2.tex results/a2_q3/README.md results/a2_q3/mind_ab_sliced_eval.json 

**End-to-end:** 3.3728 ms/impression (296.5 impressions/s) · peak RSS 0.52 GB

Compared against `2026-09-13T07-38-43_snapshot.json` (`6d15e50`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.148 | 0.1714 | +15.8% | ~ |
| `bm25_score_all` | 1.0729 | 1.0404 | -3.0% | · |
| `embed_query_build` | 0.0354 | 0.0324 | -8.5% | · |
| `embed_score_all` | 1.0134 | 0.8079 | -20.3% | ~ |
| `bm25_candidate_lookup` | 0.0176 | 0.0166 | -5.7% | · |
| `embed_candidate_lookup` | 0.0131 | 0.0126 | -3.8% | · |
| `rank_and_auc` | 1.0593 | 1.0914 | +3.0% | · |

**Verdict:** no regression

### 2026-10-12T19:22:39 — `f17a1e7` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: ARCHITECTURE.md PROJECT_STATE.md benchmarks/profile_bandit.py benchmarks/results/2026-10-12T19-14-16_ebnerd_small_bandit_profile.json decisions/ADR-017-bandit-experiment.md knowledge/ai-usage-log/2026-10-12_offline-contextual-bandit-layer.md results/bandit_ebnerd_small/heldout.json results/bandit_ebnerd_small/heldout_runs.npz results/bandit_ebnerd_small/leakage_audit.json results/bandit_ebnerd_small/mechanism_ablation.json results/bandit_ebnerd_small/simulated.json results/bandit_ebnerd_small/tables.md results/bandit_ebnerd_small/tune.json results/bandit_ebnerd_small/tune_first_pass.json scripts/bandit_leakage_audit.py scripts/bandit_make_report.py scripts/bandit_mechanism_ablation.py scripts/bandit_prepare_stream.py scripts/run_bandit_experiment.py src/bandit/__init__.py src/bandit/evaluation.py src/bandit/features.py src/bandit/policies.py src/bandit/runner.py src/bandit/stream.py tests/unit/test_bandit.py 

**End-to-end:** 2.9874 ms/impression (334.7 impressions/s) · peak RSS 0.62 GB

Compared against `2026-09-14T02-42-54_snapshot.json` (`13564ff`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1714 | 0.1251 | -27.0% | ~ |
| `bm25_score_all` | 1.0404 | 0.9785 | -5.9% | · |
| `embed_query_build` | 0.0324 | 0.0295 | -9.0% | · |
| `embed_score_all` | 0.8079 | 0.7598 | -6.0% | · |
| `bm25_candidate_lookup` | 0.0166 | 0.0149 | -10.2% | · |
| `embed_candidate_lookup` | 0.0126 | 0.0107 | -15.1% | ~ |
| `rank_and_auc` | 1.0914 | 0.9017 | -17.4% | ~ |

**Verdict:** no regression

### 2026-10-12T22:13:32 — `268f37a` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: ARCHITECTURE.md PROJECT_STATE.md benchmarks/profile_bandit.py benchmarks/results/2026-10-12T22-06-36_ebnerd_small_bandit_profile.json decisions/ADR-017-bandit-experiment.md decisions/ADR-018-bandit-drift-and-mechanism.md knowledge/ai-usage-log/2026-10-12_bandit-drift-and-mechanism.md results/bandit_ebnerd_small/drift/diagnosis.json results/bandit_ebnerd_small/drift/forget_heldout.json results/bandit_ebnerd_small/drift/forget_heldout_runs.npz results/bandit_ebnerd_small/drift/groups_forget_extra_heldout.json results/bandit_ebnerd_small/drift/groups_forget_extra_tune.json results/bandit_ebnerd_small/drift/groups_forget_heldout.json results/bandit_ebnerd_small/drift/groups_forget_tune.json results/bandit_ebnerd_small/drift/groups_plain_heldout.json results/bandit_ebnerd_small/drift/groups_plain_tune.json results/bandit_ebnerd_small/drift/realtime_audit.json results/bandit_ebnerd_small/drift/tables.md results/bandit_ebnerd_small/drift/tune_forget.json results/bandit_ebnerd_small/drift/tune_forget_first_pass.json scripts/bandit_realtime_audit.py scripts/drift_make_report.py scripts/run_bandit_experiment.py scripts/run_drift_experiment.py src/bandit/groups.py src/bandit/policies.py src/bandit/runner.py tests/unit/test_bandit_drift.py 

**End-to-end:** 3.8931 ms/impression (256.9 impressions/s) · peak RSS 0.6 GB

Compared against `2026-10-12T19-22-39_snapshot.json` (`f17a1e7`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1251 | 0.1357 | +8.5% | · |
| `bm25_score_all` | 0.9785 | 1.0386 | +6.1% | · |
| `embed_query_build` | 0.0295 | 0.0347 | +17.6% | ~ |
| `embed_score_all` | 0.7598 | 1.3223 | +74.0% | 🔴 |
| `bm25_candidate_lookup` | 0.0149 | 0.0209 | +40.3% | ~ |
| `embed_candidate_lookup` | 0.0107 | 0.0134 | +25.2% | ~ |
| `rank_and_auc` | 0.9017 | 1.091 | +21.0% | ~ |

**Verdict:** REGRESSION DETECTED

### 2026-10-12T22:14:11 — `268f37a` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** recheck after blocked commit (1)

**End-to-end:** 3.3141 ms/impression (301.7 impressions/s) · peak RSS 0.59 GB

Compared against `2026-10-12T22-13-32_snapshot.json` (`268f37a`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1357 | 0.1642 | +21.0% | ~ |
| `bm25_score_all` | 1.0386 | 1.029 | -0.9% | · |
| `embed_query_build` | 0.0347 | 0.0324 | -6.6% | · |
| `embed_score_all` | 1.3223 | 0.7982 | -39.6% | 🟢 |
| `bm25_candidate_lookup` | 0.0209 | 0.0154 | -26.3% | ~ |
| `embed_candidate_lookup` | 0.0134 | 0.0119 | -11.2% | · |
| `rank_and_auc` | 1.091 | 1.0574 | -3.1% | · |

**Verdict:** no regression

### 2026-10-12T22:14:15 — `268f37a` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** recheck after blocked commit (2)

**End-to-end:** 3.4032 ms/impression (293.8 impressions/s) · peak RSS 0.69 GB

Compared against `2026-10-12T22-14-11_snapshot.json` (`268f37a`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1642 | 0.1403 | -14.6% | · |
| `bm25_score_all` | 1.029 | 1.0282 | -0.1% | · |
| `embed_query_build` | 0.0324 | 0.0298 | -8.0% | · |
| `embed_score_all` | 0.7982 | 0.9243 | +15.8% | ~ |
| `bm25_candidate_lookup` | 0.0154 | 0.0164 | +6.5% | · |
| `embed_candidate_lookup` | 0.0119 | 0.0125 | +5.0% | · |
| `rank_and_auc` | 1.0574 | 1.0553 | -0.2% | · |

**Verdict:** no regression

### 2026-10-12T22:14:28 — `268f37a` (dirty tree)
**Hardware:** Mac15,12 / 8GiB · **Config:** MIND small/dev, 150 users, 205 impressions, 42,416 articles
**Note:** pre-commit: ARCHITECTURE.md PROJECT_STATE.md benchmarks/PERFORMANCE_LOG.md benchmarks/profile_bandit.py benchmarks/results/2026-10-12T22-06-36_ebnerd_small_bandit_profile.json decisions/ADR-017-bandit-experiment.md decisions/ADR-018-bandit-drift-and-mechanism.md knowledge/ai-usage-log/2026-10-12_bandit-drift-and-mechanism.md results/bandit_ebnerd_small/drift/diagnosis.json results/bandit_ebnerd_small/drift/forget_heldout.json results/bandit_ebnerd_small/drift/forget_heldout_runs.npz results/bandit_ebnerd_small/drift/groups_forget_extra_heldout.json results/bandit_ebnerd_small/drift/groups_forget_extra_tune.json results/bandit_ebnerd_small/drift/groups_forget_heldout.json results/bandit_ebnerd_small/drift/groups_forget_tune.json results/bandit_ebnerd_small/drift/groups_plain_heldout.json results/bandit_ebnerd_small/drift/groups_plain_tune.json results/bandit_ebnerd_small/drift/realtime_audit.json results/bandit_ebnerd_small/drift/tables.md results/bandit_ebnerd_small/drift/tune_forget.json results/bandit_ebnerd_small/drift/tune_forget_first_pass.json scripts/bandit_realtime_audit.py scripts/drift_make_report.py scripts/run_bandit_experiment.py scripts/run_drift_experiment.py src/bandit/groups.py src/bandit/policies.py src/bandit/runner.py tests/unit/test_bandit_drift.py 

**End-to-end:** 4.0138 ms/impression (249.1 impressions/s) · peak RSS 0.61 GB

Compared against `2026-10-12T22-14-15_snapshot.json` (`268f37a`):

| Stage | Before (ms/imp) | After (ms/imp) | Change | |
|---|---:|---:|---:|---|
| `query_tokenize` | 0.1403 | 0.135 | -3.8% | · |
| `bm25_score_all` | 1.0282 | 1.1471 | +11.6% | · |
| `embed_query_build` | 0.0298 | 0.036 | +20.8% | ~ |
| `embed_score_all` | 0.9243 | 1.3034 | +41.0% | 🔴 |
| `bm25_candidate_lookup` | 0.0164 | 0.0173 | +5.5% | · |
| `embed_candidate_lookup` | 0.0125 | 0.0133 | +6.4% | · |
| `rank_and_auc` | 1.0553 | 1.1409 | +8.1% | · |

**Verdict:** REGRESSION DETECTED
