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
