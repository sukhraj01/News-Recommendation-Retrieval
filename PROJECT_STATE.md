# Project State: Assignment 1 (Lexical & Semantic Retrieval)

> This document captures the current state of the project. It is updated as implementation progresses and should always reflect the latest engineering status.

**Last Updated:** August 12, 2026 (EB-NeRD Codabench submission — Part 2's Kaggle notebook prepared and ready to run; a real memory-scaling bug found and fixed in the process)

**Current Phase:** MIND Codabench submission (Q5) Parts 1-4 complete on the local side (see prior session notes below). EB-NeRD Codabench submission (competition 2469): Part 0 and Part 1 complete (see the August 12 "Part 0 Resolved + Part 1 Converter" notes below). **Part 2 is now prepped, not yet executed** — `notebooks/ebnerd_part2_kaggle_test_run.py` (a paste-into-Kaggle-cells script, same pattern as Part 0's investigation script) is written, and `notebooks/ebnerd_part2_src_bundle.zip` (this project's own validated `src/` scoring/converter code, minimal subtree, upload as a private Kaggle Dataset) is built and import-verified. Neither has been run on Kaggle yet — that's the engineer's own next step (GPU accelerator + Kaggle account required, same class of action Claude Code cannot perform directly).

**Current Objective:** Hand off `notebooks/ebnerd_part2_kaggle_test_run.py` + `notebooks/ebnerd_part2_src_bundle.zip` to the engineer to run on Kaggle (GPU on). The script covers the full brief: download+unzip, a discovery cell that re-confirms the real format/schema facts directly against the live files (including `test/history.parquet`'s presence, never explicitly checked in Part 0), corpus index build over `articles_large_only`, a real benchmark-then-decide gate (samples the actual file on the actual Kaggle instance, projects full-run time, does not auto-run), the full run itself (embeddings by default, per ebnerd_small's validation result — AUC 0.5430 vs. BM25's 0.5288), line-count validation, and packaging to the confirmed `predictions.txt`-at-zip-root format. Once the engineer relays the printed benchmark output and runs it, the remaining steps are: download only `prediction.zip` back, upload to https://www.codabench.org/competitions/2469/, screenshot the leaderboard result for Q6. Separately, MIND's Q6 design note and the manual Codabench upload/screenshot (`submissions/mind_large_test_embed/prediction.zip`) remain outstanding from a prior session.

---

# Component Status Summary

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| Data Pipeline | ✅ Complete (fast tier + ebnerd_small + MINDlarge) | 100% | `make data` builds MINDsmall + ebnerd_demo end-to-end per ADR-001/ADR-002. `ebnerd_small` built and verified (ADR-002 addendum) via opt-in `include_ebnerd_small=True`. **MINDlarge built and row-count-verified this session** (`include_mind_large=True`) — required three real algorithmic fixes to `src/datasets/mind.py`/`src/pipeline/validators.py` to survive MINDlarge's row counts on this 8GB machine (naive-loop hang, 27GB memory projection, two separate `map_infer_mask` scaling bugs); see Session Notes below. |
| Lexical Retrieval (BM25) | ✅ Complete (fast tier + ebnerd_small + MINDlarge) | 100% | `scripts/run_bm25_experiment.py --dataset {mind,ebnerd} [--bundle {small,demo}]` per ADR-005/ADR-006. Benchmarked against MINDsmall-dev, ebnerd_demo-validation, ebnerd_small-validation, **and MINDlarge-dev this session** (ADR-006 addendum) — stays local, ~10 min projected for the full 255,990-user run. |
| Semantic Retrieval | ✅ Complete (fast tier + MINDlarge) | 100% | `scripts/run_embed_experiment.py --dataset {mind,ebnerd} [--bundle {small,demo}]` per ADR-008. `paraphrase-multilingual-MiniLM-L12-v2` encoder, brute-force cosine ANN, disk-cached embeddings. Benchmarked against the same three corpora BM25 covers **plus MINDlarge-dev this session** (ADR-008 addendum) — stays local, brute-force still sufficient, FAISS still unjustified. |
| Evaluation Harness | ✅ Complete | 100% | `src/evaluation/metrics.py` (recall@K), `src/evaluation/ranking_metrics.py` (Q4: AUC/MRR/nDCG@5/nDCG@10/diversity/novelty/coverage), `src/evaluation/bootstrap.py` (shared CI substrate), `src/retrieval/score.py` (generic `Scorer` interface — BM25 and `EmbeddingScorer` both implemented, exercised through the identical unchanged harness). All bootstrap CI, warm/cold slicing (Q4.3/Q4.4). See ADR-007/ADR-008. |
| Benchmarking Framework | ✅ Complete (BM25 + semantic) | 100% | `experiments/{bm25,embed,ranking_bm25,ranking_embed}_{dataset}_{date}/{config,results}.json` pattern applied to both retrieval methods on all three fast-tier corpora plus MINDlarge-dev |
| Codabench Submission Format | ✅ Complete (MIND: converter + dev-set validation. EB-NeRD: converter + ebnerd_small validation + a real scale fix) | 100% | `src/submission/mind_format.py` — official `impression_id [rank_1,...,rank_N]` format, re-reads the raw zip directly to preserve original candidate order (the processed feature store's deterministic sort destroys it). Validated end-to-end against the real `evaluation/official/evaluate.py` on MINDlarge_dev for both BM25 and embeddings: AUC/nDCG match the project's own `ranking_metrics.py` almost exactly; the one real MRR disagreement is a verified, fully-explained metric-definition difference (official sums 1/rank over all clicked items vs. this project's first-hit-only MRR), not a converter bug. `src/submission/ebnerd_format.py` — direct port of the same design, validated against `ebnerd_small`'s validation split via the same `evaluate.py`. **This session: `read_raw_impressions` (which materialized the whole split as a `list[dict]` before writing anything) was found, by direct measurement, to project to ~16GB at ebnerd_testset's real 13,536,710-impression scale, on top of ~8.5GB for the `behaviors` DataFrame itself (pandas' own `memory_usage(deep=True)` undercounts this >3x for object-dtype list columns) — a real risk MINDlarge_test's 2.37M-impression Part 4 run never surfaced. Fixed at the root: `iter_raw_impressions` is now a generator `write_predictions`/`write_truth_file` consume one row at a time (never materializing the full list), and `read_zip_parquet`/`iter_raw_impressions` now request only the columns actually needed. `read_raw_impressions` kept as `list(iter_raw_impressions(...))` — unchanged contract, all 7 existing unit tests plus 2 new ones (equivalence + laziness) pass, full suite 164/165 (1 pre-existing skip). |
| Leaderboard Submission | 🟡 In progress | ~90% (MIND) / Part 2 prepped, not yet run (EB-NeRD) | MIND: Part 3 (dev-set validation) passing; Part 4's local half done: `submissions/mind_large_test_embed/prediction.zip` generated (embeddings, 2,370,727 lines, row-count- and format-validated against the raw zip). Actual Codabench upload + screenshot still needs the engineer's own account. EB-NeRD: `notebooks/ebnerd_part2_kaggle_test_run.py` + `notebooks/ebnerd_part2_src_bundle.zip` written and import-verified this session, ready for the engineer to run on Kaggle (GPU on) — not yet executed, so no real leaderboard result exists yet (see Deliverables Checklist) |

---

# Deliverables Checklist (Q7)

Honest status against the assignment's four required deliverables, updated at the start of each session rather than assumed complete.

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 1 | Code (GitHub Classroom) | 🟡 In progress | Data pipeline (now including MINDlarge), BM25 retrieval, semantic (embedding) retrieval, Q4 evaluation harness, and both Q5 official-format converters (`src/submission/mind_format.py`, `src/submission/ebnerd_format.py`) all implemented and tested (ADR-005/006/007/008 + this session's addenda). `README.md` documents one-command reproduce (`make data`, `make test`) — still not updated with `scripts/run_embed_experiment.py`/`scripts/generate_mind_predictions.py`/`scripts/generate_ebnerd_predictions.py` usage, flagged again for next session. `.gitignore` verified against Q8's explicit list — MINDlarge's new `data/processed/mind/large/` tree and raw zips confirmed covered by the existing `data/` rule, nothing new needed. **This session: `src/submission/ebnerd_format.py` + `scripts/generate_ebnerd_predictions.py` added and validated** (7 new unit tests, `tests/unit/test_ebnerd_format.py`) — see Session Notes. |
| 2 | Design note (≤4 pages, Moodle) | ⬜ Not started | Was deferred until semantic retrieval produced real numbers to compare against (met, ADR-008) and now also has MINDlarge-scale numbers to draw on — ready to start next session |
| 3 | Leaderboard screenshots (both Codabench competitions) | 🟡 In progress (MIND) / 🟡 In progress (EB-NeRD, Part 2 script ready, not yet run) | MIND: Parts 1-4's local half complete — `submissions/mind_large_test_embed/prediction.zip` ready to upload. **The upload itself needs the engineer's own Codabench account/login, which Claude Code cannot do regardless of local readiness.** EB-NeRD (competition 2469): Part 0/1 complete. **Part 2 this session: `notebooks/ebnerd_part2_kaggle_test_run.py` written (download, discovery/schema re-confirmation, corpus index build, a real benchmark-then-decide gate, the gated full run, line-count validation, packaging) + `notebooks/ebnerd_part2_src_bundle.zip` (this project's validated `src/` code, minimal subtree, import-verified) — both need the engineer to actually run them on Kaggle with the GPU accelerator on, which Claude Code cannot do.** Part 3 (submit + screenshot) follows once Part 2 produces a real `prediction.zip`. |
| 4 | AI usage log (prompts + AI-vs-human marking) | 🟢 Ongoing | `knowledge/ai-usage-log/` exists; one file per session (`YYYY-MM-DD_<topic>.md`), written live per CLAUDE.md's "Prompt & Session Logging" section, not reconstructed after the fact (this session's log: `2026-08-12_ebnerd-codabench-part2-testset-run.md`) |

---

# Progress Summary

## Recently Completed

- ✅ Repository initialized
- ✅ Poetry environment configured
- ✅ Testing framework (pytest) configured
- ✅ Initial documentation system established

## In Progress

- Understanding recommendation systems from first principles
- Verifying local development environment
- Preparing for architecture and design phase

---

# Learning Progress

## Day 1 — Mental Model of Recommendation Systems & News Domain (2026-08-09)

**Sources consulted:** Wu et al. 2020 (MIND, ACL Anthology 2020.acl-main.331), Kruse et al. 2024 (EB-NeRD, arXiv:2410.03432 / ACM RecSys Challenge 2024), plus industry figures on Netflix/Amazon/YouTube recommendation ROI (flagged as community-consensus-tier evidence, not audited).

**1. Why companies invest in recommendation systems**
- Core problem: search-cost / discovery problem at catalog scale no human curation team can solve manually.
- Structural difference from normal software: no correctness oracle. Success is a statistical property over a user population (CTR, retention), not pass/fail. Feedback loops exist (exposure bias) — what's shown changes what's observable next.
- ROI figures (Netflix ~80% of streamed hours + ~$1B/yr churn reduction; Amazon ~35% of sales) are widely cited but not independently audited — useful as directional motivation, not benchmarkable claims.

**2. Why news recommendation is structurally different**
- Framed by item relevance half-life: Amazon (months–years), Netflix (years), YouTube (days–years), News (hours). Item cold-start is the *default state* in news, not an edge case — this is why the assignment centers content-based retrieval (BM25 + embeddings) rather than collaborative filtering.
- EB-NeRD paper names three technical challenges explicitly: continuous publish/expire flow (item cold-start), implicit-only feedback, and mandatory reliance on article content.
- Editorial/normative dimension is unique to news among the compared domains — EB-NeRD paper: recommenders "perform a deeply editorial function." Concrete evidence: submitted models varied from 10.2% to 45.7% category coverage at similar accuracy — direct justification for Q4's mandatory diversity/novelty/coverage metrics, not just AUC/nDCG.
- Temporal (never random) splitting exists to prevent the model from seeing information that wouldn't exist yet at real serving time. MIND splits by date; EB-NeRD uses a fixed 21-day click-history window feeding a 7-day forward impression window, non-overlapping in time.
- Noted discrepancy to double-check later: assignment PDF cites EB-NeRD as ~2.7M users/600M+ impressions; the paper's active-user-filtered subset (5–1,000 clicks, May 18–Jun 8 window) reports ~1M users/37M impressions. Likely full-dataset vs. challenge-scoped-subset, not a contradiction — verify against whichever bundle (demo/small/large) we actually load.

**3. Why lexical (BM25) and semantic (embeddings) are complementary, not redundant**
- BM25 strength: exact/near-exact term specificity (named entities, numbers, proper nouns), fully interpretable, zero training cost, index updates incrementally as new articles land — critical given hourly article churn.
- BM25 failure mode: vocabulary mismatch / synonymy (e.g., Danish "Bidens klimaplan" vs. "Præsidentens grønne udspil" — same story, zero shared tokens).
- Embedding strength: captures conceptual/topical similarity and paraphrase even with no lexical overlap; XLM-R adds cross-lingual generalization.
- Embedding failure mode: over-generalization (blurs distinct entities into the same topic cluster) and no representation for brand-new named entities the model hasn't seen.
- MIND paper's own baselines (NAML, NPA, LSTUR, NRMS — all content-based over title/abstract) substantially outperform pure CF/popularity baselines — empirical basis for why the dataset schema centers article text.
- Working hypothesis (not yet evidence — to be tested via Q4 slicing + bootstrap CIs): BM25 likely stronger on warm users / head or entity-heavy articles; embeddings likely stronger on cold-start users and paraphrase-heavy categories.

**Definition of Done:** met — mental model can be explained without recommender-systems background, news-specific challenges are articulated beyond "recommendations are hard," and lexical/semantic complementarity is grounded in both papers' own evidence rather than asserted.

## August 9, 2026 — Phase 1B: Architecture Exploration Complete (Decisions ADR-001 & ADR-002)

- Explored temporal split (7 vs 14 days) using actual on-disk data inspection
- Decision: adopt official train/val splits (7-day windows) — only feasible option given data budgets
- Inspected schemas for MIND and EB-NeRD (corrected two field-attribution errors in the process)
- Decision: unified three-table schema with mandatory core + dataset-specific optional fields
- Both decisions enable single BM25/semantic code path for cross-dataset Q4 comparison
- Confidence: Medium (verified for retrieval; not yet checked against Q4 diversity/coverage metrics)

---

# Recent Decisions

| ADR | Title | Status | Notes |
|-----|-------|--------|------|
| ADR-001 | Temporal Split Strategy | Decided | Use official train/val splits as-is (7-day windows) |
| ADR-002 | Unified Data Schema | Decided | Three tables (articles, impressions, user_history) with mandatory core + optional fields |
| ADR-005 | Query Construction | Decided | Unweighted concatenation of full history (title+abstract), no recency weighting (avoids MIND's unverified history-order assumption); cold threshold = history length < 5 (EB-NeRD paper's own active-user filter). Stopword removal added mid-flight after benchmarking showed the raw query mass was dominated by function words. |
| ADR-006 | BM25 Variant | Decided | BM25Okapi (title+abstract is short/bounded, doesn't need BM25L/BM25+'s long-document correction). Scoring implemented as a precomputed sparse weight matrix, not `rank_bm25.get_scores()` directly — the latter was measured infeasible at real scale (~10hr projected vs. ~80s actual). |
| ADR-002 (addendum) | ebnerd_small Verification | Decided | Schema identical to `ebnerd_demo`; cold cohort empty by construction at this tier too (structural, not demo-only); recall@200 lower in absolute terms (larger corpus) but higher relative-to-random lift (2.87x vs. 2.37x). |
| ADR-007 | Q4 Ranking Evaluation Harness Design | Decided | Deterministic per-impression-seeded pseudo-random tie-break (cold users produce total ties); diversity/novelty K=10 (anchored to nDCG@10); novelty popularity from train split only (Q9 anti-gaming), Laplace-smoothed; coverage reported as a point estimate, no bootstrap CI (set-union statistics are mechanically biased under with-replacement resampling). |
| ADR-008 | Semantic Retrieval Design | Decided | Compute one embedding model ourselves over both datasets (rejects using EB-NeRD's provided embeddings + a separate MIND model — same single-code-path argument as ADR-002). Encoder: `paraphrase-multilingual-MiniLM-L12-v2`, chosen over `multilingual-e5-small` after a real discrimination-gap benchmark (e5's asymmetric query/passage convention doesn't fit this project's symmetric use case — confirmed empirically, not just argued). ANN: brute-force cosine similarity (0.99ms/query measured at 42,416-doc scale — FAISS unjustified). Cold-start: mirrors ADR-005's reporting posture exactly (`None`/all-zero-tie), no new fallback built. **Addendum (2026-08-11): re-verified at real MINDlarge-dev scale (72,023 articles, 255,990 users) — brute-force stays sufficient, full run projects to ~10.4 min, stays local.** |
| ADR-006 (addendum) | MINDlarge-Scale BM25 Benchmark | Decided | Resolves ADR-006's own flagged revisit trigger. Real MINDlarge-dev numbers (not projected): sparse weight matrix 23.5MB, full 255,990-user retrieval projects to ~9.8 min. Stays local, no Kaggle migration needed. |

---

# Open Engineering Questions

The following decisions will be resolved during the architecture phase:

- ~~Which BM25 variant should be implemented?~~ **Resolved (ADR-006): BM25Okapi.**
- ~~How should user queries be constructed?~~ **Resolved (ADR-005): unweighted concatenation of full history, stopwords removed.**
- Is the hand-built Danish+English stopword list (`src/retrieval/tokenize.py`) complete enough, or would a validated NLP resource change the recall numbers meaningfully? (ADR-005 flags this as a revisit trigger, not yet tested)
- ~~Does `ebnerd_small`/`ebnerd_large` also have zero cold-start users under the `<5` threshold, or is that specific to `ebnerd_demo`'s active-user filtering?~~ **Resolved for `ebnerd_small` (ADR-002 addendum, 2026-08-10): yes, zero cold-start users (min history = 5), same as demo — structural to the active-user-filtered bundle construction, not a demo-only artifact. `ebnerd_large` remains unverified.**
- ~~Which embedding model should be used?~~ **Resolved (ADR-008): `paraphrase-multilingual-MiniLM-L12-v2`, chosen over `multilingual-e5-small` after a real category-based discrimination-gap benchmark on MINDsmall-dev.**
- ~~Which ANN backend should be adopted?~~ **Resolved (ADR-008): brute-force cosine similarity — measured 0.99ms/query at the largest corpus (42,416 docs), FAISS unjustified at this scale.**
- ~~How should user embeddings be represented?~~ **Resolved (ADR-008): mean-pooled, L2-renormalized embedding of full click history, no recency weighting — mirrors ADR-005's BM25 query-construction reasoning exactly.**
- ~~What cold-start strategy should be used?~~ **Resolved (ADR-008): no new strategy — zero-history users produce `None`/an all-zero-tie for embeddings, mirroring BM25's own structural cold-start ceiling (ADR-005), reported the same way rather than papered over with an unbenchmarked fallback.**
- What temporal split strategy should be adopted?
- Which EB-NeRD population does the assignment's ~2.7M users / 600M+ impressions figure describe vs. the paper's ~1M / 37M active-user-filtered (5–1,000 clicks, May 18–Jun 8) subset? **Partially resolved in ADR-002 (medium confidence), now with a second data point:** assignment figure = full raw traffic log; paper figure = active-user-filtered subset that `ebnerd_large`/`ebnerd_small` are sampled from; `ebnerd_demo` (1,590 train-window users / 1,562 validation-window users, 24,724 + 25,356 impressions) and `ebnerd_small` (15,143 train-window users / 15,342 validation-window users, 2,585,747 + 2,928,942 impressions — measured directly, 2026-08-10) both structurally match the paper's filtered-subset methodology (21-day history / 7-day window, verified in ADR-001; same window boundaries confirmed for `ebnerd_small` in the ADR-002 addendum). Still unverified against `ebnerd_large` directly — remains open until that bundle is downloaded and inspected.
- ~~`ebnerd_small`/`ebnerd_large` schema unverified against Phase 2's loaders (built and tested strictly against `ebnerd_demo`)~~ **Resolved for `ebnerd_small` (ADR-002 addendum, 2026-08-10): schema conformance, referential integrity, and row-count checks all pass, identical to `ebnerd_demo`; feature store built at `data/processed/ebnerd/small/`. `ebnerd_large` remains unverified and out of scope for this session.**
- ~~What feature-store read API does Retrieval actually need?~~ **Resolved:** direct `pd.read_parquet()` was sufficient — no loader abstraction was needed in practice (ARCHITECTURE.md updated).

---

# Current Risks

| Risk | Likelihood | Impact | Mitigation | Status |
|------|-----------|--------|-----------|--------|
| Temporal data leakage | Low (EB-NeRD, tested) / Medium (MIND, unverifiable) | Critical | Automated leakage tests (`tests/integration/test_leakage.py`) — EB-NeRD checked directly (no impression precedes its history cutoff, passing); MIND has no per-click timestamps to check, explicitly `skip`-marked rather than silently omitted | Mitigated (EB-NeRD) / Monitoring (MIND) |
| `pyproject.toml` dependency drift on Python 3.14 | Realized once (pyarrow silently dropped, uncommitted) | High | `pyarrow` re-pinned to `^22.0.0` (first cp314 wheel); re-verify any future dependency bump against `poetry lock` succeeding, not just "no error" | Resolved this session, monitor on future bumps |
| Submission format mismatch | Low | High | Dry-run before submission | Planned |
| Timeline pressure near deadline | Medium | Medium | Weekly milestone reviews | Monitoring |
| Naive `rank_bm25.get_scores()` doesn't scale to real user/corpus counts | Realized once (measured ~10hr projected at MINDsmall-dev scale) | High | Replaced with a verified-equivalent sparse-matrix scorer (~80s actual); re-benchmark before MINDlarge enters scope (ADR-006) | Resolved this session, monitor at larger scale |
| Unweighted BM25 query concatenation can silently underperform random retrieval | Realized once (EB-NeRD pre-fix recall@50/100 below random baseline) | High | Stopword removal added and benchmarked (ADR-005); hand-built stopword list not independently validated | Resolved this session, monitor if query construction changes |
| Local machine has 8GB RAM — large-scale runs can thrash/OOM without a pre-flight memory estimate | Medium, grows with scale | High (thrashing/OOM loses in-progress work) | Memory projected from measured per-unit numbers before every new-scale run (CLAUDE.md's Memory Estimation clause); move to Kaggle immediately if projection nears ~8GB, no local attempt first | Ongoing |

---

# Benchmarking Status

**Current Status:** BM25 and semantic (embedding) baselines both complete on all three corpora (fast tier). Full BM25-vs-semantic comparison done — see below and ADR-008.

**BM25 Results** (`experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/`; full detail in ADR-006):

| Dataset | recall@50 | recall@100 | recall@200 | Random baseline @200 |
|---|---|---|---|---|
| MIND-small dev (overall) | 0.73% | 1.50% | 2.62% | 0.47% |
| MIND-small dev (warm, n=41,986) | 0.73% | 1.54% | 2.73% | — |
| MIND-small dev (cold, n=8,014) | 0.72% | 1.19% | 1.78% | — |
| EB-NeRD-demo validation (overall = warm, n=1,562) | 1.01% | 2.13% | 4.02% | 1.70% |
| EB-NeRD-demo validation (cold) | n/a — 0 users below threshold | | | |

Both datasets clear their random baseline by a real margin (MIND ~5.6x, EB-NeRD ~2.4x). MIND shows warm > cold at every k, consistent with the Day-1 working hypothesis; EB-NeRD's cold cohort is structurally empty in the `demo` bundle (min history length = 5, by construction of its active-user filter) so no warm/cold comparison is possible there — see ADR-005.

**`ebnerd_small` verification run** (`experiments/bm25_ebnerd_small_2026-08-10/`; full detail in ADR-002's addendum): overall = warm (n=15,342) recall@50/100/200 = 0.72% / 1.44% / 2.77% (random baseline @200 = 0.96%, 2.87x lift — slightly *higher* relative lift than demo's 2.37x, despite lower absolute recall, because the corpus is 1.76x larger at this tier). Cold cohort is again structurally empty (min history = 5) — confirms this is a property of EB-NeRD's active-user-filtered bundles, not a `demo`-only artifact.

**Q4 Ranking Metrics** (`experiments/ranking_bm25_mind_2026-08-10/`,
`experiments/ranking_bm25_ebnerd_small_2026-08-10/`; full detail + interpretation in ADR-007):

| Metric | MIND overall | MIND warm | MIND cold | EB-NeRD-small overall/warm | EB-NeRD-small cold |
|---|---|---|---|---|---|
| AUC | 0.5692 | 0.5766 | 0.5242 | 0.5288 | n/a (0 cold users) |
| MRR | 0.3115 | 0.3138 | 0.2975 | 0.3412 | n/a |
| nDCG@5 | 0.2887 | 0.2888 | 0.2879 | 0.3745 | n/a |
| nDCG@10 | 0.3486 | 0.3485 | 0.3490 | 0.4543 | n/a |
| Diversity@10 | 0.8367 | 0.8314 | 0.8690 | 0.7949 | n/a |
| Novelty@10 | 16.30 | 16.29 | 16.38 | 17.17 | n/a |
| Coverage@10 (point est.) | 0.0834 | 0.0804 | 0.0450 | 0.2057 | n/a |

Warm > cold on AUC for MIND (confirms recall@K's existing pattern);
nDCG@5/@10 barely differ warm-vs-cold, which looks contradictory until
accounting for the tie-break (cold rankings are random permutations, and a
random ranking already scores non-trivially on nDCG when candidate lists
are short) — a genuine finding about why Q4 mandates multiple metrics, not
a bug. EB-NeRD's nDCG is higher than MIND's despite EB-NeRD's *lower* AUC —
explained by EB-NeRD's much shorter candidate lists (median 9–12 vs.
MIND's 23), not better ranking quality; AUC (list-length-invariant)
resolves the apparent contradiction. See ADR-007's Interpretation for the
full reasoning.

**Semantic (Embedding) Results** (`experiments/embed_mind_2026-08-10/`,
`experiments/embed_ebnerd_2026-08-10/`, `experiments/embed_ebnerd_small_2026-08-10/`;
full detail + interpretation in ADR-008. `paraphrase-multilingual-MiniLM-L12-v2`,
mean-pooled user query, brute-force cosine similarity):

| Dataset | recall@50 | recall@100 | recall@200 |
|---|---|---|---|
| MIND-small dev (overall) | 0.90% | 1.60% | 2.78% |
| MIND-small dev (warm, n=41,986) | 0.93% | 1.67% | 2.92% |
| MIND-small dev (cold, n=8,014) | 0.65% | 1.10% | 1.81% |
| EB-NeRD-demo validation (overall = warm, n=1,562) | 0.32% | 0.94% | 2.57% |
| EB-NeRD-small validation (overall = warm, n=15,342) | 0.14% | 0.43% | 1.21% |

**Q4 Ranking Metrics, embeddings** (`experiments/ranking_embed_mind_2026-08-10/`,
`experiments/ranking_embed_ebnerd_2026-08-10/`,
`experiments/ranking_embed_ebnerd_small_2026-08-10/`):

| Metric | MIND overall | MIND warm | MIND cold | EB-NeRD-demo | EB-NeRD-small |
|---|---|---|---|---|---|
| AUC | 0.6340 | 0.6439 | 0.5737 | 0.5437 | 0.5430 |
| MRR | 0.3486 | 0.3523 | 0.3258 | 0.3433 | 0.3437 |
| nDCG@5 | 0.3314 | 0.3329 | 0.3221 | 0.3804 | 0.3804 |
| nDCG@10 | 0.3903 | 0.3918 | 0.3807 | 0.4587 | 0.4591 |
| Diversity@10 | 0.8269 | 0.8213 | 0.8607 | 0.7893 | 0.7890 |
| Novelty@10 | 16.15 | 16.12 | 16.30 | 14.79 | 17.19 |
| Coverage@10 (point est.) | 0.0782 | 0.0740 | 0.0440 | 0.2158 | 0.2050 |

**BM25 vs. Semantic — Q3.5/Q4.5, the headline comparison (both recall@K and
Q4 metrics, same corpora, same warm/cold split):**

- **MIND: embeddings win outright**, on both recall@K (2.78% vs. 2.62% @
  k=200) and Q4 AUC (0.634 vs. 0.569, +0.065). This does **not** match the
  Day-1 working hypothesis's framing ("BM25 favors warm/entity-heavy
  MIND") — semantic retrieval is ahead on *both* warm and cold cohorts, not
  just cold. The warm/cold AUC gap itself is similar in size for both
  methods (BM25 Δ0.052, embed Δ0.067) — embeddings raise the whole curve,
  they don't specifically close the cold-start gap. Reported as a genuine
  finding against the hypothesis, not smoothed over.
- **EB-NeRD: the two evaluation questions disagree.** BM25 clearly wins
  whole-corpus recall@K (demo: 4.02% vs. 2.57%; small: 2.77% vs. 1.21% —
  roughly 1.6–2.3x higher), but embeddings slightly *edge out* BM25 on Q4's
  already-curated-candidate-list ranking (AUC 0.544 vs. 0.533 demo; 0.543
  vs. 0.529 small). Plausible (not yet isolated) explanation: EB-NeRD's
  long median history (81–93 articles) produces a diffuse mean-pooled query
  that struggles to stand out against the *whole* catalog but still
  discriminates adequately within EB-NeRD's much shorter median candidate
  list (9–12/impression) once that list is already curated upstream.
- **Diversity/novelty/coverage are nearly identical between methods on
  every corpus** — accuracy differs, beyond-accuracy metrics mostly don't.
  This is exactly the kind of pattern ADR-007's multi-metric harness design
  was built to surface, not a null result being glossed over.
- **True (zero-history) cold-start is a shared ceiling, not something
  either method solves.** MIND's cold-cohort recall@200 is nearly identical
  between methods (BM25 1.78% vs. embed 1.81%) — both degrade to a
  structural miss for zero-history users (no vocabulary to match / no
  vector to compute), which is why the aggregate cold numbers converge even
  though the underlying failure mode differs.

Full reasoning, the encoder-selection benchmark that led to
`paraphrase-multilingual-MiniLM-L12-v2` over `multilingual-e5-small`, and
the brute-force-vs-FAISS ANN benchmark are all in ADR-008.

Experiment results are recorded in the `experiments/` directory as implementation progresses.

---

# Next Actions

## Immediate

- [x] Implement data pipeline (download + parse + split + feature store build)
- [x] Run temporal-split leakage tests
- [x] Verify unified schema works for both datasets in practice
- [x] Design BM25 indexing strategy (query construction, variant choice — ADR-005/ADR-006)
- [x] Implement + benchmark BM25 retrieval on MINDsmall-dev and ebnerd_demo-validation
- [x] Verify `ebnerd_small` against Phase 2's loaders; benchmark BM25 against it (ADR-002 addendum)
- [x] Design + implement Q4's ranking evaluation harness (AUC/MRR/nDCG/diversity/novelty/coverage, ADR-007); run against BM25 on both datasets
- [x] Design semantic retrieval (embedding model, ANN backend, user representation, cold-start — ADR-008)
- [x] Implement + benchmark embedding retrieval (recall@K and Q4 harness) on MINDsmall-dev, ebnerd_demo-validation, ebnerd_small-validation
- [x] BM25-vs-semantic comparison (Q3.5/Q4.5), warm/cold sliced where available — see Benchmarking Status above and ADR-008
- [x] Register on Codabench MIND competition (implicit — engineer confirmed session could proceed; EB-NeRD competition registration still separately unconfirmed)
- [x] MINDlarge built, benchmarked, and Q5's format converter validated against real ground truth (Parts 1-3, 2026-08-11 — see Session Notes)
- [x] Part 4 (local half): generate MINDlarge_test (blind) predictions with embeddings, validate row count/format — `submissions/mind_large_test_embed/prediction.zip` ready (2026-08-12 — see Session Notes)
- [ ] Part 4 (manual half): upload `prediction.zip` to the MIND Codabench leaderboard, capture screenshot — needs the engineer's own account/login
- [ ] Q6: write the design note (≤4 pages) — ADR-008's Interpretation section plus MINDlarge-scale findings are now the primary source material
- [x] EB-NeRD's own Codabench submission format: Part 0 resolved — engineer ran `notebooks/ebnerd_part0_kaggle_investigation.py` on Kaggle against the real `predictions_large_random.zip`/`ebnerd_testset.zip`/`articles_large_only.zip` and relayed results (see Session Notes)
- [x] Part 1 (`src/submission/ebnerd_format.py`, validated against ebnerd_small): complete this session — 7 unit tests, end-to-end validation against `evaluation/official/evaluate.py` for both BM25 and embeddings (see Session Notes)
- [x] Part 2 prep: `notebooks/ebnerd_part2_kaggle_test_run.py` + `notebooks/ebnerd_part2_src_bundle.zip` written this session, import-verified, logic-checked locally against `ebnerd_small` (see Session Notes) — a real memory-scaling bug found and fixed in `src/submission/ebnerd_format.py` along the way
- [ ] Part 2 execution (engineer runs the notebook on Kaggle, GPU on, relays the benchmark/discovery output) and Part 3 (submit `prediction.zip` + screenshot): next, blocked on the engineer's own Kaggle/Codabench access
- [ ] Q9 design-note note: confirm in writing that `total_inviews`/`total_pageviews`/`total_read-time` are never read anywhere in this pipeline (grep `src/` before writing that claim, don't assert from memory)

## Upcoming

- Once the engineer relays the Kaggle investigation output: validate it against Table 6-8's documented schema, then implement `src/submission/ebnerd_format.py` against the *confirmed* real format (mirroring `src/submission/mind_format.py`'s pattern — re-read the raw zip for original candidate order, score through the existing `Scorer` interface, no new scoring logic).
- Manual Codabench upload of `submissions/mind_large_test_embed/prediction.zip` + Q6 (design note) remain outstanding from the MIND side, independent of EB-NeRD's progress.
- `README.md` needs a `scripts/run_embed_experiment.py` / `--method embed` usage note, plus `scripts/generate_mind_predictions.py` — not updated yet, flagged again for next.
- `ebnerd_large` remains undownloaded/unverified — lower priority unless the assignment specifically requires the `large` tier for leaderboard submission.
- ADR-008 flags a possible future investigation (not required this session): a curated near-duplicate/paraphrase evaluation set to validate the encoder's discrimination quality more directly than the category-proxy check used here.
- A pre-existing pandas `FutureWarning` (`Index.insert` with object-dtype, inside `validate_table`) surfaced repeatedly in recent sessions — cosmetic, not chased down, worth a quick look eventually.
- BM25 test-split predictions were not generated this session (embeddings won clearly on dev — AUC 0.6335 vs. 0.5699 — so this is the already-justified single-method choice per this session's brief); can be added later if wanted.

---

# Session Notes

## August 12, 2026 (latest) — EB-NeRD Codabench Submission, Part 2: Kaggle Notebook Prepared (Not Yet Run)

### Completed

- **Confirmed no code changes were actually needed to *design* Part 2** — Part 0/1 already settled the format, the article-corpus source, and the beyond-accuracy uniform-treatment rule. What this session found instead was a genuine **scale gap**, the same class of thing MINDlarge's own onboarding session hit (naive loops, memory projections that don't hold at 5-6x the previously-exercised scale): `src/submission/ebnerd_format.py::read_raw_impressions` materializes the entire split as a `list[dict]` before writing a single output line. That was invisible at MINDlarge_test's 2,370,727-impression scale (Part 4, prior session) but ebnerd_testset is 13,536,710 impressions — 5.7x larger, explicitly flagged in this session's brief as new territory needing a real benchmark before committing.
- **Measured the risk directly rather than reasoning about it abstractly**, per CLAUDE.md's Memory Estimation clause: built synthetic rows shaped exactly like `read_raw_impressions`'s real output (~9-15 candidate ids/impression, matching EB-NeRD's real per-impression candidate-count median) and measured actual RSS growth. Result: the `rows` list alone projects to **~16.3GB** at 13,536,710 rows. The underlying `behaviors` DataFrame (built first, stays resident the whole time since the list-building loop iterates over its columns) adds **~8.5GB** more — and pandas' own `df.memory_usage(deep=True)` badly undercounts this (reported ~99MB against a real measured ~321MB on the same 500K-row sample), because it doesn't recurse into the boxed Python ints inside object-dtype list columns. Combined, the naive path risked ~25GB of peak memory before any actual scoring work even started, independent of whatever Kaggle's session RAM ceiling turns out to be.
- **Fixed at the root, not worked around**: added `iter_raw_impressions` (a generator, `src/submission/ebnerd_format.py`) as the real implementation — one row materialized, written, and discarded at a time. `read_raw_impressions` is now `list(iter_raw_impressions(...))`, an unchanged-contract wrapper kept for existing callers/tests that want a list. `write_predictions`/`write_truth_file` now consume the generator directly. Also added an optional `columns=` parameter to `src/utils/io.py::read_zip_parquet` (backward-compatible, default `None` = read everything as before) so `iter_raw_impressions` only parses `impression_id`/`user_id`/`article_ids_inview`(/`article_ids_clicked` when labeled) — not every column the real `behaviors.parquet` happens to ship (e.g. `is_beyond_accuracy`, never read by this module).
- **Verified the fix changes nothing observable**: added `test_iter_raw_impressions_is_lazy_and_matches_read_raw_impressions` and `test_iter_raw_impressions_unlabeled_matches_read_raw_impressions` to `tests/unit/test_ebnerd_format.py` (confirms `iter_raw_impressions` is a real generator, and that streamed output is byte-for-byte identical to the old list-based output on both the labeled and unlabeled/test-shaped code paths). Full suite: 164 passed (up from 162), 1 pre-existing skip, 6 deselected `slow` — no regressions. Beyond the unit tests, ran the *actual* blind-test (`has_labels=False`) code path end-to-end locally against the real `ebnerd_small.zip` validation split (244,647 real impressions, not a fixture): 0 malformed lines, correct packaging (`predictions.txt` alone at the zip root). This is the first time the unlabeled/test path has been exercised against real (not fixture) data for EB-NeRD.
- **Wrote `notebooks/ebnerd_part2_kaggle_test_run.py`** — a paste-into-Kaggle-cells script, same established pattern as `ebnerd_part0_kaggle_investigation.py`. Covers the full brief: Cell 0 `wget`s `ebnerd_testset.zip`/`articles_large_only.zip` (same URLs Part 0 used); Cell 1 discovers inputs and installs `rank_bm25` (not in Kaggle's base image); Cell 3 is a **discovery/re-confirmation cell**, not a trust-the-prior-session cell — it re-derives the real impression count, article count, and (critically) whether `test/history.parquet` actually exists in `ebnerd_testset.zip`, something Part 0's investigation never explicitly checked (it only inspected `behaviors.parquet`) even though every query-construction path in this project depends on per-user history existing; the cell raises and stops rather than guessing if it's absent. Cell 4 builds the BM25 + embedding indexes over the full `articles_large_only` corpus (`device="cuda"` when available — this is the actual GPU-beneficial step the task brief called out). **Cell 6 is a real benchmark-then-decide gate**: it takes a deterministic systematic sample (every Nth impression, not a prefix — avoids bias from the 200,000-row `is_beyond_accuracy` block potentially clustering) of the real file on the real Kaggle instance, measures actual ms/impression for both methods, and prints a projected full-run time — it does **not** auto-proceed. Cell 7 (the actual 13.5M-impression run) stays behind an explicit `RUN_FULL_JOB = True` flag the engineer sets only after reading Cell 6's projection, with a periodic-progress print (every 500K lines, rate + ETA) so a multi-hour run is inspectable rather than a black box — the MINDlarge_test session's own "check progress" prompts made clear that opacity during a long run is a real friction point worth designing around. Cells 8-9 validate line count/format and package to the confirmed `predictions.txt`-at-zip-root structure before anything gets downloaded.
- **Built `notebooks/ebnerd_part2_src_bundle.zip`** — the minimal `src/` subtree Part 2 actually needs (`retrieval/`, `submission/ebnerd_format.py`, `evaluation/ranking_metrics.py`+`bootstrap.py`, `datasets/ebnerd.py`, `utils/`; ~24 files, ~46KB), for the engineer to upload as a private Kaggle Dataset. Chosen over either re-deriving the scoring/converter logic inline in the notebook (risks silent drift from the validated, tested version) or `git clone`ing this repo on Kaggle (no remote is configured locally — `git remote -v` is empty, so that path doesn't exist right now). Import-verified twice (before and after the streaming fix) in an isolated `sys.path` to confirm the bundle is self-contained and has no missing-dependency surprises waiting on Kaggle.
- Confirmed via `experiments/ranking_{bm25,embed}_ebnerd_small_2026-08-10/results.json` that embeddings win on `ebnerd_small`-validation (AUC 0.5430 vs. BM25's 0.5288, non-overlapping 95% CIs) — same direction as MINDlarge-dev's embeddings win (0.6335 vs. 0.5699) — so the notebook defaults `RUN_METHODS = ["embed"]`, with BM25 available as an explicit opt-in if Cell 6's real projection shows there's compute budget for both.

### What Wasn't Done (and why)

- **The notebook has not been run.** This session cannot execute anything on Kaggle (no `kaggle` CLI, no browser/notebook access — same constraint Part 0 hit) or upload to Codabench (needs the engineer's own account). Nothing in this update should be read as "Part 2 is done" — it's prepped and logic-verified everywhere that's possible without Kaggle itself.
- Did not apply the same `iter_raw_impressions`-style streaming fix to `src/submission/mind_format.py`. MINDlarge_test already ran successfully end-to-end at its real 2,370,727-impression scale without hitting this; out of scope for this session's brief and not a demonstrated problem there.

### Next Steps

- Engineer runs `notebooks/ebnerd_part2_kaggle_test_run.py` on Kaggle (GPU accelerator on, `notebooks/ebnerd_part2_src_bundle.zip` uploaded as a Data source) and relays back Cell 3's discovery output, Cell 6's benchmark/projection, and Cell 8's validation result — same relay pattern Part 0 used.
- If Cell 6's projection is comfortably within Kaggle's session/quota limits: flip `RUN_FULL_JOB = True`, let Cell 7 run, then Cells 8-10 validate/package/instruct on the download.
- If the projection is borderline or over budget, or `test/history.parquet` turns out to be missing (Cell 3's hard-stop case): relay back before proceeding — real alternatives (embeddings-only, multi-session chunking, etc.) are named in Cell 6's printed output, not silently chosen.
- Once a real `prediction.zip` exists: download only that file back to `submissions/ebnerd_testset_<method>/`, upload to Codabench competition 2469, screenshot for Q6, then update this file and the AI usage log to reflect the actual (not projected) result, and commit that as the "submission confirmed" checkpoint — separate from this session's own prep-complete commit.

### Related

- `notebooks/ebnerd_part2_kaggle_test_run.py`, `notebooks/ebnerd_part2_src_bundle.zip`
- `src/submission/ebnerd_format.py` (`iter_raw_impressions` addendum), `src/utils/io.py` (`read_zip_parquet`'s new `columns=` parameter)
- `tests/unit/test_ebnerd_format.py` (2 new tests)
- `knowledge/ai-usage-log/2026-08-12_ebnerd-codabench-part2-testset-run.md`

### Addendum (same day, mid-Kaggle-run) — real `ebnerd_testset.zip` packaging quirk found and fixed

The engineer ran Cell 1-3 on Kaggle (GPU on, Tesla T4, 31.3GB RAM) and hit
a real `KeyError` in Cell 3's own diagnostic read: the real
`ebnerd_testset.zip` wraps every member in an extra top-level directory
(`ebnerd_testset/test/behaviors.parquet`, not the flat
`test/behaviors.parquet` every EB-NeRD caller in this project assumed and
was tested against, since `ebnerd_small.zip`/`ebnerd_demo.zip` don't do
this). Cell 1-2's discovery/environment checks all passed cleanly first —
`test/history.parquet` **is confirmed present** (real file:
`ebnerd_testset/test/history.parquet`, 1.16GB), GPU detected correctly
(Tesla T4), 29.6GB RAM available. The crash was purely a path-assumption
bug, not the memory or history-availability risks this session had
already prepared for.

Fixed at the shared IO layer, not just in the notebook: `read_zip_member_
bytes` (`src/utils/io.py`) now tries the exact member name first
(unchanged, fast path — every existing fixture/test still hits this) and
falls back to a suffix search across the real namelist (excluding
`__MACOSX/` junk, raising if the match isn't exactly 1) if that fails.
This fixes every downstream EB-NeRD caller at once (`parse_ebnerd_
articles`, `_parse_history`, `iter_raw_impressions`) without touching
their code — they already went through `read_zip_parquet`. The one place
that didn't was Cell 3's own diagnostic `behaviors_meta` read, which used
a raw `zipfile.ZipFile(...).open(...)` call directly instead of the
project's own utility; switched it to `read_zip_parquet` too. Added 5 new
tests (`tests/unit/test_io.py`: exact-match fast path, wrapped-directory
fallback, `__MACOSX` exclusion, ambiguous-match error, missing-member
error) and verified against a zip built to replicate the real file's
exact reported listing (including the `.DS_Store`/`__MACOSX` junk
entries) before calling it fixed — not just against the synthetic test
fixtures. Full suite: 169 passed (up from 164), 1 pre-existing skip, no
regressions. Rebuilt `notebooks/ebnerd_part2_src_bundle.zip` with the fix
and re-verified its imports in isolation.

**Next:** engineer re-uploads the corrected `ebnerd_part2_src_bundle.zip`
to the same Kaggle Dataset (or a new one) and re-runs from Cell 1. Given
Cell 3 already confirmed `test/history.parquet` exists and the real
resource headroom (T4 GPU, ~30GB RAM), Cells 4 onward should now be
unblocked.

---

## August 12, 2026 — EB-NeRD Codabench Submission, Part 0 Resolved + Part 1 Converter Built & Validated

### Completed

- **Part 0 resolved with real evidence from Kaggle**, relayed by the engineer (per CLAUDE.md's Resource Availability clause — no guessing from the paper's Table 6/7 schema docs, same discipline as MIND's Part 3):
  - `predictions.txt`: one line per impression, `impression_id [rank_1,...,rank_N]`, a permutation matching that impression's `article_ids_inview` order — same shape as MIND's official format, just different source column names.
  - `articles_large_only.zip` alone gives 100% coverage of the real test set's in-view articles (10,451/10,451) — `ebnerd_large.zip` is not needed at all.
  - Exactly 200,000 of the test set's 13,536,710 impressions are flagged `is_beyond_accuracy=True`, every one drawn from one fixed 250-article pool. These still need real submitted rankings in the same format — `predictions.txt`'s expected line count (13,536,710) is an exact match to the full test set, not just the 13,336,710 regular rows, confirming no special-cased handling is needed in the converter.
- **Built `src/submission/ebnerd_format.py`** as a direct port of `mind_format.py`'s design (per the session brief: "a port, not a redesign"), adapted for EB-NeRD's raw shape — reads `{split}/behaviors.parquet` directly from the raw zip (parquet, not TSV; `article_ids_inview`/`article_ids_clicked` list columns rather than MIND's `"N3-1 N4-0"` token strings) for the same reason MIND's converter does: `src/pipeline/orchestrator.py::_write_table` sorts the processed `impressions` table by `["impression_id", "article_id"]`, destroying `article_ids_inview`'s original order. Scoring still goes through the identical `Scorer`/index interfaces (`BM25Scorer`/`EmbeddingScorer`) as every other retrieval path in this project — no new scoring logic. 7 new unit tests (`tests/unit/test_ebnerd_format.py`), reusing the committed `ebnerd_demo_sample.zip` fixture (hand-verifiable: impressions 3/4 in `validation`, articles 101/102/103, known clicks) — mirrors `test_mind_format.py`'s structure exactly.
- **Wrote `scripts/generate_ebnerd_predictions.py`** (port of `generate_mind_predictions.py`) and generated official-format `prediction.txt`/`truth.txt` for `ebnerd_small`'s `validation` split (known labels, 244,647 impressions — line count matches the split's real impression count exactly) with both BM25 and embeddings, reusing the exact same index/query construction `scripts/run_ranking_eval.py` already uses for that split.
- **Validated end-to-end against `evaluation/official/evaluate.py`** — confirmed in Part 0 to be a generic, format-only script (parses only the shared `impid [ranks]`/`impid [labels]` line shape, nothing MIND-specific), so it applies to EB-NeRD's predictions unchanged, the same cross-check class Part 3 used for MIND:

  | Metric | BM25 (ranking_metrics.py) | BM25 (evaluate.py) | Embed (ranking_metrics.py) | Embed (evaluate.py) |
  |---|---|---|---|---|
  | AUC | 0.5288 | 0.5288 | 0.5430 | 0.5430 |
  | nDCG@5 | 0.3745 | 0.3745 | 0.3804 | 0.3804 |
  | nDCG@10 | 0.4543 | 0.4543 | 0.4591 | 0.4591 |
  | MRR | 0.3412 | 0.3407 | 0.3437 | 0.3432 |

  AUC and nDCG@5/@10 match to 4 decimal places for both methods — strong evidence the converter's rank extraction and ordering are correct, same conclusion MIND's Part 3 reached. The MRR gap reproduces the same verified official-vs-project definition difference already documented for MIND (`evaluate.py`'s `mrr_score` sums `1/rank` over every clicked candidate and normalizes by click count; this project's `mrr()` credits only the first hit) — investigated rather than assumed to carry over unchanged: recomputed both formulas directly from the generated `prediction.txt`/`truth.txt` and reproduced both tools' reported numbers exactly (0.34073 official-formula vs. reported 0.3407; 0.34123 first-hit vs. reported 0.3412). Went one step further than MIND's investigation and found a genuine EB-NeRD-specific contributor: of the 1,407 validation impressions with `len(article_ids_clicked) > 1`, 659 are actually a **duplicate entry for the same article** (e.g. `[X, X]`), not two distinct clicks — only the remaining 748 impressions have truly distinct multi-click labels, and 748 is exactly the count of impressions where the two MRR formulas diverge (confirmed by direct recomputation). Not a bug in either tool or the converter — a real data quirk plus the same known metric-definition difference, now fully traced rather than hand-waved.

### Decisions

- No new ADR written for `ebnerd_format.py` — following the precedent set by `mind_format.py` itself (also undocumented in `decisions/`), since this is explicitly a port of an already-justified design (raw-zip-reread reasoning, `Scorer`-interface reuse) rather than a new architectural decision.

### Next

- Part 2: the real Kaggle test-set run (`ebnerd_testset.zip`, `articles_large_only.zip`, same `wget`-into-`/kaggle/working` pattern Part 0 used) — worth doing now that Part 1's converter is validated against real ground truth, not just fixtures.
- Part 3: submit + screenshot (engineer's own Codabench account, same constraint as MIND).
- BM25 vs. embeddings on `ebnerd_small`-validation: embeddings win on every metric here too (AUC 0.5430 vs 0.5288, consistent with the already-established Q4 pattern) — not a new finding, just reconfirmed through this session's independent code path.

## August 12, 2026 (later) — EB-NeRD Codabench Submission, Part 0 Investigation Started

### Completed

- **Resource constraint surfaced before implementation, per CLAUDE.md's Resource Availability clause.** The session brief's Part 0/Part 2 explicitly require Kaggle execution. Checked this environment directly rather than assuming: no `kaggle` CLI, no `~/.kaggle` credentials, no browser/notebook access. This is a hard blocker for the Kaggle-only steps, not a style choice — stopped, named the constraint, and asked the engineer how to proceed (three options: prep-and-relay, configure a Kaggle token here, or skip verification and guess from docs). Engineer chose prep-and-relay, the same pattern already established for the MIND Codabench upload (engineer's own login required).
- **Did everything locally executable first, rather than waiting idle.** Extracted the EB-NeRD paper's Appendix A (Tables 6-8) from `data/ebnerd_paper.pdf` via `pypdf` (ad hoc install into the poetry venv, same one-off pattern as last session's MIND paper extraction) — confirms the documented `behaviors.parquet` test-split schema (drops Article ID/Next read-time/Next scroll percentage/Clicked article IDs, adds `is_beyond_accuracy` across 200,000 samples).
- **Inspected `jppol-ai/ebnerd-benchmark` without a full clone** — a first `git clone --depth 1` attempt timed out twice over a slow connection; switched to the GitHub REST API (`git/trees?recursive=1` + `raw.githubusercontent.com` for specific files) to pull only what was needed, skipping the repo's NRMS/LSTUR/NAML/NPA model code and notebooks entirely (out of scope per the session brief). Found a genuine negative result worth documenting: `codabench/README.md` describes **server-side compute-worker infrastructure** (a Docker setup for running a CodaBench scoring backend on your own VM) — it is not a submission-format spec or a client-side scoring script. This repo does not contain Part 0's ground truth; inspecting the real `predictions_large_random.zip` on Kaggle is the only way to get it, not one option among several. Also tried `WebFetch` against the competition's Submission Guidelines tab (codabench.org/competitions/2469) — returned only the React SPA shell, confirming that route doesn't work without an actual browser session either.
- **Computed the local demo+small article-ID reference set** directly from the already-built feature store (`data/processed/ebnerd/{demo,small}/articles.parquet`): 21,700 unique raw article IDs, published_time up to 2023-07-11. Exported as `notebooks/ebnerd_small_demo_article_ids.csv` so the Kaggle-side coverage check (does `articles_large_only.zip` cover test-period articles that demo/small don't?) doesn't need to re-derive this on Kaggle.
- **Wrote `notebooks/ebnerd_part0_kaggle_investigation.py`** — a paste-into-Kaggle-cells script, auto-discovering input files by filename under `/kaggle/input` (doesn't depend on knowing the engineer's exact dataset slug). Covers all four remaining Part 0 checks: `predictions_large_random.zip`'s literal file layout/line format; `ebnerd_testset.zip`'s real `behaviors.parquet` columns checked against Table 7 (`is_beyond_accuracy` presence + value_counts, the four expected-absent columns, beyond-accuracy rows' fixed-pool structure); and `articles_large_only.zip`'s in-view coverage, cross-checked against both itself and the local demo+small CSV.
- **Committed a small pre-existing housekeeping gap first** (`7423a51`): `CLAUDE.md`'s Memory Estimation clause and its matching `PROJECT_STATE.md` risk-table row were fully written in the working tree from the prior session but never committed. Verified complete and self-contained before committing separately, ahead of this session's own changes.

### Key Outcomes

- Part 0 is genuinely half-done, not stalled: everything answerable without Kaggle access (paper schema, starter-repo scope, WebFetch dead-end) is resolved with real evidence, and the negative result on the starter repo (no format spec there) is itself useful — it rules out a shortcut before the engineer spends Kaggle time.
- No submission-format code has been written yet, deliberately — the session brief explicitly warned against guessing the format from Table 6/7's schema docs, and building `ebnerd_format.py` before Part 0's Kaggle results would be exactly that guess. Part 1 stays blocked until real data confirms the format.
- Reusable pattern establishing itself across both Codabench submissions (MIND and EB-NeRD): Claude Code does all locally-executable investigation and implementation; anything requiring an external account, browser, or platform access is prepped as an exact, ready-to-run artifact (script, upload list, or instructions) and handed to the engineer, who relays results back rather than Claude Code attempting a workaround.

### Next Session

- **Blocking on the engineer:** run `notebooks/ebnerd_part0_kaggle_investigation.py` on Kaggle (upload `notebooks/ebnerd_small_demo_article_ids.csv` alongside the existing dataset), paste the full printed output back.
- Once that lands: validate it against Table 6-8, then implement `src/submission/ebnerd_format.py` (Part 1) against the confirmed real format, generate + cross-check predictions against `ebnerd_small`'s validation split for both BM25 and embeddings, same discipline as MIND's Part 3.
- MIND's Q6 design note and manual Codabench upload/screenshot remain independently outstanding.

---

## August 12, 2026 — Part 4: MINDlarge_test Predictions Generated and Validated

### Completed

- **Framing correction before implementation.** The session brief characterized Part 4 as pure execution on an already-validated pipeline. Reading the actual processed tree before running anything showed that wasn't quite true: Part 4 had never been exercised end-to-end (explicitly deferred every prior session), so two real bugs in code paths only Part 4 touches had never surfaced. Both were found by inspection/reproduction before generating any real predictions, not discovered via a crash in a throwaway run — consistent with CLAUDE.md's "benchmark/verify before trusting" discipline extended to "exercise the code path before trusting it's ready."
- **Bug 1 — `MINDlarge_test`'s `user_history` was computed then discarded.** `src/datasets/mind.py::parse_mind_test_candidates` called the same `_parse_behaviors_tsv` helper train/dev use (which always computes `user_history` regardless of `has_labels`), but discarded the result via `candidates, _user_history = ...`; `src/pipeline/orchestrator.py::build_mind_test` never wrote it. A unit test even asserted the old (incomplete) key set as if this were intentional. Query construction (BM25 or embedding) needs history independent of whether labels exist — the prior "no `clicked` column" framing (a real, correct ADR-002 constraint) had been conflated with "test doesn't need history." Fixed: `parse_mind_test_candidates` now returns `user_history`; `build_mind_test` writes it via the same `_write_table`/`USER_HISTORY_SCHEMA` convention `build_mind_split` already uses. Rebuilt the real `data/processed/mind/large/test/` tree with the fix (702,005 users with history — exactly matching every user referenced in `candidates.parquet`, zero missing). Commit `65f6bfc`.
- **Bug 2 — `Scorer` crashed on a candidate id absent from the corpus.** First real prediction-generation attempt crashed after ~430s with `KeyError: 'mind:N89741'`. Investigated directly against the raw zip rather than guessing: `MINDlarge_test/behaviors.tsv` references `N89741` as a candidate in 32 of 2,370,727 impressions, but that article is genuinely absent from `MINDlarge_test/news.tsv` — confirmed train and dev have zero such gaps, so this is a one-article quirk isolated to the raw test files, not a parser bug. Both `BM25Scorer` and `EmbeddingScorer` shared the same unguarded `id_to_col[c]` lookup in `src/retrieval/score.py`. This is a distinct situation from ADR-005/008's cold-start handling (no *query* → every candidate ties) — here the *item* has no representation. Fixed with a shared `_lookup_scores` helper: an unknown id scores `-inf`, ranking it last deterministically (verified: `N89741` lands at rank 138/138 in its one inspected impression, and strictly last in all 32 affected impressions); the fast vectorized path is unchanged for the common case. Commit `a120c45`.
- Both fixes covered by new unit tests; full fast suite re-verified clean after each (153 passed → 155 passed, up from the prior session's 152).
- **Part 4 (local half) — generated and validated MINDlarge_test predictions.** Ran `scripts/generate_mind_predictions.py --split test --method embed` as a detached (`nohup`+`disown`) background process — embeddings only, per the session brief (won clearly on dev, AUC 0.6335 vs. BM25's 0.5699; BM25 not run this session, already a defensible single-method choice). Encode+index build took ~12s on the restart (articles embedding cache from the crashed first attempt was reused); full scoring of 2,370,727 impressions took ~6,946s (~1.9hr), in line with the session's 1-2hr estimate. Validated before treating the output as upload-ready, same class of check Part 3 used to catch silent truncation: line count matches the raw zip's real impression count exactly (2,370,727); every line parses as a valid rank permutation (0 malformed); the 32 `N89741`-affected impressions individually spot-checked. Packaged as `submissions/mind_large_test_embed/prediction.zip` (`prediction.txt` zipped at the archive root, matching `evaluate.py`'s expected `submit_dir/prediction.txt` layout).

### Key Outcomes

- Part 4's "execution, not discovery" framing was half right: local prediction generation needed no new design decisions, but two real implementation gaps only Part 4 could expose were still hiding in already-committed code, caught and fixed with the same rigor as any other engineering decision (root-caused against real data, tested, documented) rather than patched around or silently absorbed.
- `submissions/mind_large_test_embed/prediction.zip` is ready for upload; the only remaining step is manual (Codabench login/upload/screenshot), which Claude Code cannot perform regardless of local readiness.

### Next Session

- Manual: log into Codabench, upload `submissions/mind_large_test_embed/prediction.zip`, screenshot the leaderboard result for Q6.
- Q6: write the design note (≤4 pages).
- `README.md` still needs the `run_embed_experiment.py`/`generate_mind_predictions.py` usage notes flagged in prior sessions.

---

## August 11, 2026 — MINDlarge Build, Benchmark, and Q5 Dev-Set Validation

### Completed

- **Part 1 — MINDlarge feature store build.** `include_mind_large=True` had never actually been exercised before this session — treated as a real test, per the session's own instructions, and it found real bugs. First attempt hung indefinitely (heavy CPU + swap growth, no progress after 7+ minutes) in `src/datasets/mind.py::_explode_impressions`'s per-token Python loop at MINDlarge_train's real scale (~2.2M impressions x ~37 avg candidates ≈ 80M+ exploded rows) — never a problem at MINDsmall's ~14x-smaller scale. Root-caused and fixed in three separate rounds, each found by directly profiling the actual stuck process (via macOS `sample`) rather than guessing from theory:
  1. Vectorized the explode via `DataFrame.explode` instead of a per-row dict-building loop.
  2. That fix alone projected to **~27GB** in memory (measured directly) because pandas' object dtype repeats each ~37x-duplicated prefixed ID string as a distinct Python object per row. Fixed with `category` dtype for the three ID columns and the four always-null EB-NeRD-only columns (12.5x measured reduction, ~2.2GB projected) — but this exposed a second real bug: `src/pipeline/validators.py::_is_null`'s `series.map(a_python_function)` returned `category`-dtype output for a categorical input in this pandas version, which `.sum()` couldn't reduce. Fixed by switching to plain `series.isna()` (verified to have byte-identical semantics, including the "empty list is never null" case `_is_null` exists for) — this also turned out to be the *actual* runtime bottleneck (a full build hung for over an hour with zero progress; profiling showed nearly all the time inside `map_infer_mask`), not just a dtype bug.
  3. A third, separate `map_infer_mask` bottleneck was found the same way (profiling a build that was still slow after fix #2): `_explode_impressions` was building the full ~40-char prefixed ID string via `+` concatenation *before* converting to `category`, and separately running `.str.rsplit("-", n=1, expand=True)` on the full ~81M-row exploded token Series — both are elementwise Python loops under the hood, not vectorized C ops, paying their cost 81M times instead of the ~2.2M (or fewer) times actually needed. Fixed by categorizing raw (unprefixed) values first and prefixing only the small category array, and by splitting article/label tokens at the raw ~2.2M-row level (before exploding) instead of after.
  - Every fix was verified exact-match against the original loop implementation on real data before being trusted (same discipline as ADR-006's BM25 scoring rewrite), and the full test suite (152 tests) re-verified clean after each round.
  - Final real numbers: MINDlarge_train's ~83.5M-row exploded impressions table builds in ~163s (512K rows/s) at ~5.2GB peak memory (measured, not projected) — the full `build_all(include_mind_large=True)` run completes in well under the time these fixes made obvious was otherwise impossible.
  - Row counts verified against Wu et al. (2020)'s real published Table 2/Section 3.2 statistics — fetched via WebFetch and text-extracted with `pypdf` since `ACL2020_MIND.pdf` isn't present anywhere in this repo or the wider filesystem (flagged explicitly rather than silently using memorized numbers). Found and explained a real discrepancy, not a bug: raw parsed counts (train 2,232,748 / dev 376,471 / test 2,370,727) exceed the paper's reported post-filter counts (2,186,683 / 365,200 / 2,341,619) by exactly the count of empty-history rows in each split (verified directly against the raw zips: train diff = 46,065 = exactly the empty-history row count; test matches exactly; dev matches within 1 row). The paper explicitly states "we only kept the samples with non-empty news click history" when reporting its own statistics — the publicly released files (and this project's parser, deliberately, per ADR-005's cold-start philosophy) keep them. Total union article/user counts across train/dev/test (130,379 articles / see test for users) are meaningfully below the paper's full-corpus 161,013/1,000,000 — explained by the paper's own construction description (each split's own news.tsv only covers its own narrow time window, not the full 6-week raw-log period the headline totals describe), documented as a sanity-bound test rather than a false-precision tolerance check.
  - New tests: `tests/integration/test_schema_conformance.py`'s MINDlarge block (schema conformance, impression-count-vs-paper with the empty-history explanation, article/user-count sanity bounds), all `@pytest.mark.slow`.
  - Confirmed raw zips and the new `data/processed/mind/large/` tree stay out of git — already covered by the existing `.gitignore`'s `data/` rule, nothing new needed.
- **Part 2 — Benchmark before trusting anything at MINDlarge scale.** Both BM25 and embeddings benchmarked directly on MINDlarge_dev's real corpus (72,023 articles — this split's own catalog, not the paper's 161,013 full-corpus figure, per ADR-005's "each split's own catalog is the query-time universe" convention) and real user counts (255,990), not projected from MINDsmall. BM25: index build 1.70s, sparse weight matrix 23.5MB (ADR-006's flagged memory-footprint trigger resolved — nowhere near a bottleneck), full retrieval projects to ~9.8 min. Embeddings: encoder throughput 373.6 articles/s (warm-cache; faster than ADR-008's original 172.4/s, plausibly a warm-vs-cold-cache effect), full encode+cache 264.4s, full retrieval projects to ~10.4 min. **Decision: both stay local** — combined full run is well under 25 minutes, far short of the >2hr/exceeds-RAM threshold that would justify a Kaggle GPU detour under CLAUDE.md's Resource Availability clause. Documented as ADR-006 and ADR-008 addenda (not new ADRs, since the fix is the same kind of "benchmark before trusting" verification each ADR's own decision confidence already called for, not a structurally different decision).
- **Part 3 — Official-format converter, validated against ground truth.** New module `src/submission/mind_format.py`: converts per-impression `Scorer` output into the official `impression_id [rank_1,...,rank_N]` format. The core design problem this module exists to solve: `src/pipeline/orchestrator.py::_write_table` always sorts `impressions`/`candidates` alphabetically by `article_id` before writing to parquet (ADR-002's deterministic-output requirement) — this destroys the original within-impression candidate order the official format needs, so the module re-reads the raw zip directly for ordering (never for scoring, which still goes through the exact same `Scorer`/index interfaces everything else uses). 7 new unit tests (`tests/unit/test_mind_format.py`) using the existing committed MIND fixture zips. Smoke-tested against the real `evaluate.py` on a 3-impression hand-verifiable case first (computed AUC matched a by-hand calculation exactly) before trusting it at real scale.
  - Generated MINDlarge_dev predictions for both BM25 and embeddings (376,471 impressions each, `scripts/generate_mind_predictions.py`), plus a local ground-truth file from dev's own real labels (Codabench's actual private test-set ground truth is never available to us; dev is the only labeled MINDlarge split usable for this kind of validation).
  - Ran the real `evaluation/official/evaluate.py` against both prediction sets and compared against `scripts/run_ranking_eval.py --bundle large --method {bm25,embed}`'s own numbers: **AUC and nDCG@5/@10 match almost exactly for both methods** (largest gap 0.0002) — strong evidence the converter's rank extraction and ordering are correct. **MRR disagrees by a real margin** (BM25: 0.3124 vs 0.2706; embed: 0.3476 vs 0.3036) — investigated rather than dismissed, and fully explained: `evaluate.py`'s `mrr_score` sums `1/rank` over *every* clicked candidate and normalizes by click count, while this project's `mrr()` credits only the first hit (the standard single-hit MRR definition) — verified with certainty by recomputing both formulas directly from the same prediction/truth files (28.72% of MINDlarge-dev impressions are multi-click; the two recomputed values matched each tool's reported number to 4 decimal places). Not a bug — a genuine, now-documented metric-definition difference between the two implementations.
  - **On MINDlarge-dev, embeddings win clearly** (AUC 0.6335 vs BM25's 0.5699, nDCG@10 0.3897 vs 0.3497) — consistent with ADR-008's original MINDsmall finding (embeddings win outright on MIND).
- Two real, unrelated test regressions found and fixed along the way (not silently worked around): `tests/integration/test_pipeline_end_to_end.py::test_default_build_does_not_touch_mindlarge` was checking the shared real `processed_dir` fixture, which now legitimately has `mind/large` built into it — moved to an isolated `tmp_path` build so the test checks `build_all()`'s own default-argument behavior, not incidental ambient state.
- Machine-stability notes for future sessions: this 8GB-RAM machine restarted once mid-session (not just slept) during the heaviest build attempt, and two background jobs run concurrently at MINDlarge scale came within a hair of looking like an OOM kill (later confirmed to be a monitor-script false-positive, not an actual crash — both jobs had completed successfully). Running MINDlarge-scale jobs one at a time, not concurrently, is the safer default on this hardware going forward.

### Key Outcomes

- MINDlarge is no longer an unverified, never-exercised code path — it's built, row-count-verified against the real paper, and every real scaling bug it exposed (three in the data pipeline, none in BM25/embedding scoring itself) is fixed, verified, and documented with the actual measured numbers, not estimates.
- The local-vs-Kaggle resource decision CLAUDE.md's own collaboration model calls for was made with real evidence, not assumption: both retrieval methods comfortably stay local at MINDlarge scale on this machine.
- Q5's format-conversion risk (the assignment's own README originally guessed the wrong CSV format before the real `evaluate.py` script was found) is now fully retired for the labeled dev split — the same code path Part 4 will use for the blind test split has already been proven correct against real ground truth, not just unit-tested against fixtures.

### Next Session

- Part 4: generate MINDlarge_test (blind) predictions — embeddings won clearly on dev, so that's the primary candidate; BM25 also ready if both are wanted. No local scoring is possible for test (no ground truth), so Part 3's already-passing validation is what stands in for it.
- Actual Codabench upload + leaderboard screenshot (Q6) needs the engineer's own account/login — cannot be done by Claude Code regardless of how ready the local artifacts are.
- Q6 design note itself remains not started.
- Minor, non-blocking: a pre-existing pandas `FutureWarning` (`Index.insert` with object-dtype, inside `validate_table`) surfaced during this session's runs — unrelated to this session's fixes, not chased down.

---

## August 10, 2026 — Housekeeping + Phase 4: Semantic Retrieval Complete

### Completed

- **Part 0 — Housekeeping:** Confirmed `knowledge/ai-usage-log/` is live and this session's prompts are being logged verbatim as the session proceeds (`2026-08-10_phase4-semantic-retrieval-design.md`), not reconstructed afterward. Added a Deliverables Checklist (Q7) to this file, honestly flagging that **Codabench registration for both competitions is still outstanding** — this requires the engineer's own account and cannot be done by Claude Code; flagged explicitly rather than silently skipped.
- **Part 1 — ADR-008 design (plan mode):** Bundled four sub-decisions (embedding source, encoder choice, ANN backend, user representation/cold-start), following ADR-005/007's precedent. Chose to compute one embedding model over both datasets rather than use EB-NeRD's provided embeddings + a separate MIND model — same single-code-path argument ADR-002 already established for the schema, reapplied here. Added `sentence-transformers` as a new dependency (verified `poetry lock && poetry install` succeeds, per the pyarrow-incident lesson).
- **Encoder selection, empirically decided, not assumed:** Benchmarked `paraphrase-multilingual-MiniLM-L12-v2` against `multilingual-e5-small` on real MINDsmall-dev articles — both cleared the throughput bar (~6-7 min projected for the full ~75k-article corpus on this machine's MPS backend, no cloud GPU needed), but a same-category-vs-different-category cosine-similarity discrimination check (1,500 real articles, 3,000 sampled pairs) showed MiniLM meaningfully discriminates topically related from unrelated articles (2.4x same/diff ratio, 0.05–0.13 dynamic range) while e5-small — even using its own documented `passage:` prefix convention — compresses nearly everything into a narrow, largely undifferentiated 0.75–0.78 band. This confirmed, with real data, the pre-registered concern that e5's asymmetric retrieval-training objective doesn't fit this project's symmetric user-profile-vs-catalog-article use case. MiniLM chosen.
- **ANN backend:** Measured brute-force cosine similarity at 0.99ms/query against the largest corpus (MIND-dev, 42,416×384) — confirmed FAISS unjustified at this scale, same "benchmark before adding complexity" lesson ADR-006 established for BM25 scoring. `faiss-cpu` stays an unused, documented pyproject dependency.
- **Implementation:** `src/retrieval/embed.py` (`EmbeddingIndex`, disk-cached embedding computation, mean-pooled user query construction), `EmbeddingScorer` added to `src/retrieval/score.py` (mirrors `BM25Scorer`'s identity-caching shape; `None` query → all-zero tie, deliberately resolving ADR-007's own flagged Research Trigger about a non-BM25 scorer's cold-start behavior), `embed_retrieve_top_k` added to `src/retrieval/retrieve.py` (factored a shared `_top_k_from_scores` helper out of `retrieve_top_k` for reuse — a justified small refactor, verified behavior-preserving). `scripts/run_embed_experiment.py` (recall@K, mirrors `run_bm25_experiment.py`'s exact shape) and `scripts/run_ranking_eval.py --method embed` (generalized the previously-hardcoded BM25 dispatch into `_build_method`, re-verified byte-identical against ADR-007's recorded BM25 numbers after the refactor).
- **Benchmarks run on all three corpora BM25 already covers** (MINDsmall-dev, ebnerd_demo-validation, ebnerd_small-validation), both recall@K and the Q4 ranking harness; also backfilled a missing `ranking_bm25_ebnerd_2026-08-10` (demo bundle) run, since ADR-007 had only benchmarked `ebnerd_small` for Q4 ranking, not `demo` — needed for full parity. See Benchmarking Status above and ADR-008 for the complete BM25-vs-semantic comparison.
- 20 new unit tests (`test_embed.py`, plus `EmbeddingScorer`/`embed_retrieve_top_k` cases added to `test_score.py`/`test_retrieval.py`) and 2 new integration tests (real-encoder end-to-end retrieval on a small real MINDsmall-dev sample; cold-start short-circuit) added. Full suite re-verified: 145 passed, 1 skipped (MIND leakage, expected — no per-click timestamps), 1 deselected (`slow` MINDlarge test, expected), no regressions.
- Wrote `decisions/ADR-008-semantic-retrieval-design.md`.

### Key Outcomes

- **The Day-1 working hypothesis does not hold as stated.** On MIND, embeddings beat BM25 on *both* recall@K and Q4 AUC, across *both* warm and cold cohorts — not just cold, as the hypothesis predicted. The warm/cold gap itself is similar in size for both methods; embeddings raise the whole curve rather than specifically closing the cold-start gap. Reported honestly as a finding against the hypothesis, not reframed to fit it.
- **On EB-NeRD, recall@K and Q4 ranking disagree on which method is better** — BM25 clearly wins whole-corpus recall@K (1.6–2.3x higher), embeddings slightly edge out BM25 on ranking the already-curated candidate list. A structurally real result (different evaluation questions), not noise — plausibly explained by EB-NeRD's much longer median history producing a diffuse whole-catalog query that still discriminates adequately within a short, pre-curated candidate list.
- **True (zero-history) cold-start remains a shared ceiling neither method solves** — MIND's cold-cohort recall@200 converges to nearly the same number under both methods (1.78% BM25 vs. 1.81% embed), for different underlying reasons (no vocabulary vs. no vector). Consistent with, not a new instance of, ADR-005's original cold-start finding — reported the same way rather than treated as requiring a new fallback strategy.
- The encoder-selection benchmark is a clean example of this project's evidence hierarchy in practice: theoretical reasoning (the Sentence-BERT paper) correctly picked the right *category* of solution, but a real measurement was needed to pick the right *model within that category* — and it overturned the naive assumption that a retrieval-tuned model would obviously win.

### Next Session

- Q5: generate Codabench prediction files, submit to both leaderboards (blocked on the engineer completing Codabench registration first), capture screenshots.
- Q6: write the design note (≤4 pages) — ADR-008's Interpretation section and this session's comparison table are the primary source material.
- Update `README.md` with `run_embed_experiment.py` usage (flagged, not done this session).

---

## August 10, 2026 — ebnerd_small Verification + Q4 Ranking Evaluation Harness Complete

### Completed

- **Part 1 — `ebnerd_small` verification:** Downloaded `ebnerd_small.zip` (publicly accessible, no registration wall despite `download.py`'s docstring claiming otherwise — flagged as minor documentation drift, not corrected this session), inspected its raw zip structure against `ebnerd_demo`'s before trusting the existing parser, wired it into `orchestrator.build_all(include_ebnerd_small=True)` mirroring the `include_mind_large` opt-in pattern, and built the full feature store (20,738 articles; 15,143/15,342 train/validation users; 2,585,747/2,928,942 impressions). Schema conformance, referential integrity, and row-count regression checks all pass, identical to `ebnerd_demo`. Confirmed the cold-start finding generalizes: `ebnerd_small`'s validation split also has zero users below the `<5` threshold (min history = 5) — structural to the active-user-filtered bundle construction, not a `demo`-only artifact. Re-ran the BM25 benchmark against it (added `--bundle` to `run_bm25_experiment.py`): recall@200 = 2.77% (lower in absolute terms than demo's 4.02%, but the corpus is 1.76x larger; relative-to-random lift is actually higher, 2.87x vs. 2.37x). Documented as an ADR-002 addendum, ticked the relevant "Conditions for Revisiting" checkboxes in both ADR-002 and ADR-005 rather than leaving them stale.
- **Part 2 — Q4 ranking evaluation harness:** Designed via a dedicated Plan-agent pass grounded in real impression-candidate-count/click-rate statistics pulled from the processed data before implementing (median 23 candidates/impression for MIND, 9–12 for EB-NeRD; 0 degenerate all-clicked/all-unclicked impressions found in either dataset; multi-click impressions real, up to 24 in one MIND impression). Extracted `src/retrieval/score.py` (`score_all`, `Scorer` Protocol, `BM25Scorer`) from logic previously inlined in `retrieve_top_k`, adding `id_to_col` to `BM25Index` — this is the generic scoring seam Phase 4's embedding scorer will plug into as a second consumer. Extracted `src/evaluation/bootstrap.py` from `recall_at_k`'s previously-inlined bootstrap logic, pinned behavior-identical by a new test (`test_metrics.py`) written *before* the refactor, per CLAUDE.md's "extend, don't duplicate" guidance. Implemented `src/evaluation/ranking_metrics.py`: AUC (sklearn, degenerate-impression-safe), MRR, nDCG@5/@10 (binary relevance), intra-list diversity (category-based), novelty (train-split-only popularity per Q9's anti-gaming requirement, Laplace-smoothed against ADR-002's measured 32.9%/45.7% train/validation article-set gap), and catalog coverage (deliberately given no bootstrap CI — a set-union statistic is mechanically biased under with-replacement resampling, a structural argument, not a style choice). Built `scripts/run_ranking_eval.py` and ran it against BM25 on both MINDsmall-dev (73,152 impressions) and `ebnerd_small`-validation (244,647 impressions).
- Wrote `decisions/ADR-007-ranking-evaluation-design.md`, bundling four related sub-decisions (tie-break rule, diversity/novelty K=10, novelty's train-only popularity source, coverage's CI omission) the same way ADR-005 bundled three — they're facets of one question, not independent choices.
- 41 new unit/integration tests added (`test_score.py`, `test_bootstrap.py`, `test_metrics.py`, `test_ranking_metrics.py`, `test_ranking_eval_pipeline.py`, plus `ebnerd_small` schema-conformance cases); full existing suite re-verified with no regressions (`retrieve_top_k`'s behavior unchanged after the `score_all` extraction; `recall_at_k`'s bootstrap output byte-identical after the refactor).

### Key Outcomes

- `ebnerd_small` is verified, built, and benchmarked — the last open question ADR-001/ADR-002/ADR-005 all flagged about EB-NeRD's schema/cold-start generalization beyond the `demo` bundle is now resolved with real evidence, not assumption.
- The Q4 ranking harness surfaced a genuine, non-obvious finding rather than just producing numbers: AUC and nDCG *disagree* on which dataset "ranks better" (MIND wins on AUC, EB-NeRD wins on nDCG), and the disagreement is fully explained by candidate-list-length differences between the datasets, not a bug — directly demonstrating why Q4 mandates multiple metrics instead of one. On MIND specifically, warm/cold shows the expected AUC gap but a much smaller nDCG gap, traced to the cold-user tie-break producing near-random (but honestly, not artificially inflated) rankings.
- Both this session's real engineering decisions (score-vs-rank interface split, bootstrap extraction, tie-break rule, novelty/coverage definitions) were made and documented *before* being needed by a second consumer — Phase 4 (semantic retrieval) should be able to plug into `Scorer` and the harness without rewriting either, which was the explicit design goal, not an incidental benefit.

### Next Session

- Begin Phase 4: semantic retrieval design (embedding model choice — provided EB-NeRD embeddings vs. computing MIND's own via BERT/XLM-RoBERTa — and ANN backend, per the assignment's Q3).
- Implement a `Scorer` + query-builder pair for the chosen embedding method; run it through the existing, unchanged Q4 harness for the BM25-vs-semantic comparison this project has been building toward since Day 1's working hypothesis.

---

## August 5, 2026 — Project Initialization

### Completed

- Repository initialized.
- Documentation structure created.
- Engineering workflow established.
- Project organization designed around:
  - Clear architecture
  - Reproducibility
  - Benchmarking
  - Decision tracking
  - Long-term maintainability

### Key Outcomes

- Documentation is organized so project context can be reconstructed quickly.
- Engineering decisions will be documented through ADRs.
- Benchmarking and reproducibility are first-class parts of the workflow.

### Next Session

- Begin studying recommendation system fundamentals.
- Complete environment verification.
- Prepare for architecture and design.

---

## August 9, 2026 — Day 1: Mental Model of Recommendation Systems & News Domain

### Completed

- Researched business rationale for recommendation investment (Netflix/Amazon/YouTube), with evidence-tier caveats.
- Read and grounded explanation in MIND (Wu et al., 2020) and EB-NeRD (Kruse et al., 2024) papers directly, not just the assignment PDF's summary.
- Built mental model: why news recommendation differs structurally (item half-life, mandatory content-based cold-start mitigation, editorial/normative dimension), why temporal splitting is non-negotiable, why lexical and semantic retrieval are complementary rather than redundant.
- Logged a data-source discrepancy (assignment PDF vs. paper's active-user-filtered EB-NeRD stats) to verify once we load the actual bundle.

### Key Outcomes

- Full write-up recorded in `# Learning Progress` above.
- Working (unverified) hypothesis for Q3.5: BM25 favors warm users/head or entity-heavy articles, embeddings favor cold-start users/paraphrase-heavy categories — to be tested empirically via Q4 slicing, not assumed.

### Next Session

- Begin Phase 1B: architecture exploration — temporal split strategy and unified schema ADRs first, since they constrain the feature store and both retrieval legs.

---

## August 9, 2026 — Phase 2: Data Pipeline Implementation Complete

### Completed

- Fixed a pre-existing, uncommitted `pyproject.toml` regression found before implementation started: `pyarrow` had been silently dropped (a prior `poetry lock` on Python 3.14 failed against the original `^12.0.0` pin, which predates any cp314 wheel). Without it, EB-NeRD's parquet files couldn't be read at all. Re-pinned to `^22.0.0` (first version with a cp314 wheel), verified via `poetry lock && poetry install` plus a live parquet read.
- Planned the pipeline architecture via plan mode + a dedicated Plan-agent design pass, grounded directly in ADR-001/ADR-002 and live inspection of both raw bundles' actual file structure (not just the ADRs' summaries).
- Implemented `src/utils/{config,ids,io}.py`, `src/pipeline/{schema,validators,download,orchestrator}.py`, `src/datasets/{mind,ebnerd}.py`, `scripts/{generate_test_fixtures,build_feature_store}.py`; wired `make data`, `make test-reproducibility`, `make clean-data`.
- Corrected the approved plan's ID-prefixing scheme mid-implementation: verified empirically that MIND and EB-NeRD `user_id`s are shared, overlapping identities across train/dev/validation splits (not split-scoped counters like `impression_id`, which genuinely restarts per file) — split-qualifying `user_id` as originally planned would have silently broken Q4's warm/cold-user analysis. Flagged to and confirmed with the user before implementing.
- Sourced MIND's download URLs from the actively-maintained `recommenders-team/recommenders` loader (verified live, not fabricated) rather than guessing.
- Found and fixed two real validator bugs during first real-data run: `pd.NaT` wasn't recognized as null (only `float` NaN was), and empty lists (e.g. a user with no click history) were incorrectly treated as null — both would have produced false failures or false passes on real data.
- `make data` builds MINDsmall (train+dev) + ebnerd_demo (train+validation) end-to-end from raw zips; MINDlarge implemented but excluded from the default (fast) build.
- Full test suite: 47 unit/integration tests + 13 reproducibility tests passing (1 MIND leakage test explicitly skipped — no per-click timestamps exist to check against — 1 MINDlarge `slow`-marked test deselected by default).
- Row counts verified to match ADR-001's evidence table exactly for both datasets and all four splits.
- Two independent `build_all()` runs produce value-identical output (reproducibility test, not just claimed).
- Updated ARCHITECTURE.md's Data Pipeline and Feature Store sections from placeholders to the actual implemented design, including an Architecture Changelog entry.

### Key Outcomes

- Data Pipeline and Feature Store components are implemented and tested for the fast tier (MINDsmall + ebnerd_demo); MINDlarge is a slower, separately-runnable tier.
- Both real engineering issues found this session (the pyarrow regression, the user_id ID-scheme error) were caught by direct verification against real data rather than trusting an unverified assumption — consistent with CLAUDE.md's evidence hierarchy.

### Next Session

- Begin Phase 3: BM25 retrieval design (variant choice, query construction) — first open engineering question in the list above.
- Consider whether semantic retrieval design should proceed in parallel or sequentially after BM25 is benchmarked.

---

## August 10, 2026 — Phase 3: BM25 Lexical Retrieval Complete

### Completed

- Designed query construction and BM25 variant via plan mode before implementing, grounded in ADR-002's flagged MIND-history-order risk (rejected recency weighting on that basis) and the EB-NeRD paper's own active-user filter (5–1,000 clicks, adopted directly as the warm/cold threshold rather than inventing a project-local number).
- Implemented `src/retrieval/{tokenize,index,query,retrieve}.py` and `src/evaluation/metrics.py` (recall@K + bootstrap CI), plus `scripts/run_bm25_experiment.py`.
- Found and fixed two real defects, neither visible without running the real pipeline at real scale (both documented as benchmark-driven ADR amendments, not silently absorbed into the implementation):
  - `rank_bm25.get_scores()`'s per-query-token Python loop projected to 10+ hours at MINDsmall-dev's real scale (50,000 users x 42,416 articles); replaced with a sparse-matrix scorer reproducing the exact same formula (`idf`/`doc_freqs`/`doc_len`/`avgdl`/`k1`/`b` all taken directly from a fitted `rank_bm25.BM25Okapi`), verified byte-for-byte against the library's own output on 30 real users before being trusted, cutting the full run to ~80 seconds.
  - EB-NeRD's long per-user histories (up to 1,459 articles), concatenated unweighted into one query, produced queries whose term-count mass was dominated by high-document-frequency Danish/English function words — measured directly to push recall@50/100 *below* the random-retrieval baseline. Root-caused (not guessed) by inspecting a real query's most frequent tokens, then fixed with stopword removal in the shared tokenizer, verified via a 4-way tokenization variant comparison on a 1,500-impression sample (chosen variant: multiset + stopwords removed, NOT deduplication — deduplication was tested and made things worse).
- Ran full production benchmarks on both datasets (`experiments/bm25_mind_2026-08-10/`, `experiments/bm25_ebnerd_2026-08-10/`): MIND-small dev recall@200 = 2.62% (warm 2.73% / cold 1.78%, 8,014 cold users incl. 1,407 zero-history); EB-NeRD validation recall@200 = 4.02% (cold cohort empty by construction — every ebnerd_demo validation user has history length >= 5). Both clear their random baseline by a real margin (5.6x / 2.4x).
- Wrote ADR-005 (Query Construction) and ADR-006 (BM25 Variant) documenting both the pre-benchmark design reasoning and the mid-flight, evidence-driven corrections above; updated ARCHITECTURE.md's Retrieval component and Feature Store→Retrieval interface from placeholders to the actual implemented design.
- 17 new unit/integration tests added; full existing suite (76 passed, 1 skipped, 1 slow-deselected) re-verified with no regressions.

### Key Outcomes

- BM25 baseline is implemented, correctness-verified against the mandated library, and benchmarked on both datasets — ready to serve as Q4's lexical-retrieval side of the BM25-vs-semantic comparison.
- Both real engineering issues this session (the scoring-performance ceiling, the stopword-mass defect) were caught by benchmarking against real data at real scale, not by code review or small unit-test fixtures — neither would have been visible from the tiny hand-built test fixtures alone, consistent with CLAUDE.md's "benchmark before you trust a decision" principle.
- EB-NeRD demo's structural lack of cold-start users (confirmed: min history = 5) means the warm/cold BM25 comparison the assignment requires is only meaningful for MIND in this project's current data — flagged as an open question for `ebnerd_small`/`ebnerd_large`, not silently glossed over.

### Next Session

- Begin Phase 4: semantic retrieval design (embedding model choice, ANN backend — open questions above).
- Re-benchmark the BM25 sparse-matrix scorer's memory/time profile before MINDlarge enters scope (unmeasured beyond MINDsmall-dev).

---

# Important Dates

| Milestone | Target Date | Status |
|-----------|------------|--------|
| Phase 1 Complete | August 12, 2026 | On Track |
| Phase 2 Complete | August 19, 2026 | Planned |
| Pipeline & Retrieval Complete | August 26, 2026 | Planned |
| Leaderboard Submission | August 26, 2026 | Planned |
| Final Assignment Submission | August 27, 2026 | Deadline |