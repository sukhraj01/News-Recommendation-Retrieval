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
