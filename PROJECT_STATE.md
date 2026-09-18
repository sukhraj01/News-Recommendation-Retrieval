# Project State: Assignment 2 (Learning from Click-Logs), built on Assignment 1

> This document captures the current state of the project. It is updated as implementation progresses and should always reflect the latest engineering status.

## Assignment 2: current state (updated 2026-09-12)

**Brief:** `A2.pdf` (repo root). Due **2026-09-20**. Q1 features · Q2 two-stage re-ranker ·
Q3 reproduce the official baseline, then beat it with an ablation and paired bootstrap CI ·
Q4 serving/scale · Q5 extended eval with slices · Q6 6-page design note · Q7–Q9 deliverables.
Team goal of rank < 10 on both leaderboards is a stretch on top of the graded work, pursued
only once Q1–Q6 are solid.

**Current objective: Q3.** See `decisions/ADR-015-a2-official-baseline-reproduction.md` (canonical).

| Item | Status |
|---|---|
| Corpus-stats anchor table | ✅ `docs/corpus_stats.md` + `scripts/compute_corpus_stats.py` (matches 2026-08-31 exactly) |
| Local env | ✅ Rebuilt: poetry py3.11 venv had vanished; 328 passed / 1 skipped |
| Q3 Option A (official TF code on Ada) | ❌ **Timebox expired.** PyPI is ~0.09–0.37 MB/s from every reachable machine; the TF+CUDA env never finished (job 2694353 held a GPU idle ~4 h). See ADR-015 addendum. |
| Q3 Option B (PyTorch port of the official configs) | ✅ **Complete.** `src/retrieval/nrms_official{,_data}.py`, 26 unit tests, local smoke tests pass, both datasets' control+treatment trained and A/B-verified through to a real MIND leaderboard result (930992, 0.6766) |
| MIND control, job 2694501 | ✅ **Done** (13 h 13 m, 10 epochs). MINDlarge-dev, all 376,471 impressions: **AUC 0.6831** (CI 0.6821–0.6839), MRR 0.3803, nDCG@5 0.3620, nDCG@10 0.4279; guardrails diversity@10 0.8343, novelty@10 17.62, coverage@10 0.0627. Beats Candidate J (0.6579) by +0.0252 on the same split, marginal comparison not paired |
| EB-NeRD control (job 2694962) | ✅ **Done** (26 m, 5 epochs, best at epoch 3). 244,647 validation impressions: **AUC 0.5613** (CI 0.5597–0.5629), MRR 0.3512, nDCG@5 0.3904, nDCG@10 0.4680; guardrails diversity@10 0.7894, novelty@10 17.21, coverage@10 0.2021. **Candidate K scores 0.7588 on the same split** — on EB-NeRD our A1 model beats the official baseline by ~0.20 AUC, the reverse of MIND |
| EB-NeRD treatment (job 2694991, + freshness) | ✅ **Done** (30 m, best at epoch 1, restored freshness weight −0.066 = fresher wins). **AUC 0.5677** |
| **EB-NeRD A/B verdict** | ✅ **Q3 satisfied.** Paired Δ AUC **+0.0064 [+0.0053, +0.0075]**, CI excludes zero; MRR/nDCG@5/nDCG@10 all CI-clear; guardrails diversity +0.0006 and novelty +0.0280 (both improved), coverage −0.0035 (point only, flagged). Caveat: the improved baseline (0.5677) is still far below Candidate K (0.7588) |
| Infrastructure faults survived (all recorded in ADR-015) | 2694505 CUDA OOM (fixed: `--es-batch`, width-aware chunking, expandable segments); 2694963 / 2694964 / 2694986 instant failures on **gnode033** (mismatched driver, `nvidia-smi` exits 18); 2694988 / 2694989 on **gnode012** (**GTX 1080 Ti sm_61**, unsupported by this torch build — `is_available()` returns True, so the old guard missed it). Now: GPU-capability gate with a real CUDA matmul, `--constraint=2080ti`, and `--dependency=afterany` chaining |
| MIND treatment (title + abstract 50), job 2694992 | ✅ **Done** (34.06h, 10 epochs, no errors, checksum-verified). **AUC 0.6868** (CI 0.6859–0.6877) |
| **MIND A/B verdict** | ✅ **Q3 satisfied.** Paired Δ AUC **+0.0037 [+0.0031, +0.0044]**; MRR/nDCG@5/nDCG@10 all CI-clear wins. **Guardrail regression: diversity@10 −0.0060 [−0.0062, −0.0057], CI-clear** — the first guardrail regression in this project's history, disclosed per the professor's framing rather than smoothed over. Novelty improved (+0.0197); coverage flat. Head/tail slicing independently reproduces EB-NeRD's exact pattern (head −0.0063 loss, tail +0.0098 gain); gain is entirely a warm-user effect (cold: +0.0002, not significant) |
| **Q3 status** | ✅ **Complete on both datasets** — reproduced, improved, ablated, paired CI, slicing, both leaderboards' worth of comparison points |
| Candidate K (for Q4/Q5) | ✅ **Retrained 2026-09-12 at 65 features**: K_rank_nopos 0.7588 (CI 0.7571–0.7606), reproducing ADR-013's post-correction band. Stored in 3 places (repo `experiments/`, `~/a2_model_artifacts/`, Ada `$HOME/a2/artifacts/`). `ebnerd_large` retrain deferred: S3 is ~18 KB/s from Ada and `$HOME` has ~4.9 GB free |
| Q2 true-retrieval-ceiling eval | ✅ **Done at full scale** (`scripts/a2_q2_retrieve_rerank_eval.py`, real answer 2026-09-17). **NRMS-only, decided 2026-09-12** — K's features are impression-conditional and undefined for retrieved-but-unshown candidates (ADR-015). Real checkpoint (`mind_treatment_v2`, dev AUC 0.686786, matching the reported 0.6868 to 4 decimals), MINDlarge-dev, 8,000 sampled impressions: hit rate 3.21% [2.83,3.61] (recall@200, measured fresh), and among hits, NRMS re-ranking lifts AUC from 0.5428 (raw BM25 order) to **0.7224** — paired **+0.1796 [+0.1375,+0.2213]**, CI-clear win, MRR/nDCG@5/nDCG@10 all CI-clear too. Supersedes the earlier interim reduced-scale check (paired −0.0464, not significant — confirmed to be an undertrained-checkpoint artifact, not a re-ranking verdict). Full detail + the superseded interim result: `results/a2_q2/README.md` |
| **Q4 serving/scale** | ✅ **Done** (ADR-014 addendum, 2026-09-12). Index memory: MIND **164.0 MB** (BM25 23.5 + embed 110.6 + NRMS 29.9), EB-NeRD **190.5 MB** (articles 108.2 + profiles 82.1 + popularity 0.2). Request p99: MIND **94.67 ms** (1.1× headroom under a 100 ms SLA; max 126 ms already breaches), EB-NeRD **22.74 ms** (4.4×). Cost/1k queries: MIND $0.0031–$0.0004, EB-NeRD $0.0009–$0.0001. 10×: MIND breaks on **latency** (NRMS = 78% of a request), EB-NeRD on **user-profile memory** (5.35 KB/user → ~4.2 GB at `ebnerd_large`'s 791,582 users) |
| Q4 gap — closed | ✅ **MIND GPU per-request latency now measured for real** (job 2695844, once the treatment job released the GPU): **p99 33.15 ms, 3.0× SLA headroom** (vs CPU's 1.1×) — NRMS drops from 78% to 36% of the request, matching ADR-014's training-time 8.98× GPU finding. Caught and fixed a second, mirrored bug in `cost_qps.py` while closing this: it picked "the newest MIND profile" for both instance rows, which would have priced GPU latency onto the CPU-instance row once the GPU profile became newest — fixed with a device filter + 4 regression tests |
| **Q5 extended eval** | ✅ **Complete on both datasets** (ADR-007 addendum + ADR-015's MIND A/B section, 2026-09-12/14): head/tail + warm/cold, per-slice paired CIs. **EB-NeRD's freshness gain reverses on head articles** — head (n=9,541) Δ−0.0217 [−0.0258, −0.0178] vs tail (n=235,106) Δ+0.0076, so the +0.0064 headline is a tail gain diluted by a CI-clear head loss. **MIND's title+abstract treatment independently reproduces the same directional pattern** — head (n=143,164) Δ−0.0063 [−0.0071,−0.0053] CI-clear loss vs. tail (n=233,307) Δ+0.0098 [+0.0090,+0.0108] CI-clear gain — and its gain is entirely a warm-user effect (warm +0.0043 [+0.0036,+0.0050] CI-clear; cold +0.0002 [−0.0020,+0.0022] not significant). Neither dataset's treatment closes the cold-start gap, consistent with every earlier finding in this project |
| **Q9 anti-gaming** | ✅ **Quantified** (ADR-013 addendum, 2026-09-12). New `K_rank_noctx` arm withholds `context_read_time`/`context_scroll_percentage`, the two features ADR-013 flagged as weakest at serving time: **with 0.7581 vs without 0.7570, paired +0.0011 [+0.0008, +0.0015]**. Real but negligible — without them the ranker still beats `embed_sim` by +0.2140. Contrast: withholding position *improves* by 0.0007. Neither flagged group is load-bearing. Leakage tests + behaviour-window enforcement already existed (ADR-009/013) |
| **Q1 MIND history features** | ✅ **Done** (ADR-016, Candidate L) — a real, CI-clear **loss**, reported honestly, not adopted. New `src/retrieval/mind_features.py`: click count + recency-weighted category/subcategory affinity (decay=0.9, Candidate C's convention), stacked with existing unweighted match scores into a 5-feature CPU-only logistic-regression combiner. **AUC 0.6176 (CI 0.6153–0.6197) vs. the embedding baseline's 0.6340 — paired −0.0164 [−0.0191, −0.0139], every metric loses CI-clear.** The click-count ablation confirmed the theoretical zero-ranking-effect claim exactly (max \|ΔAUC\| = 0.00e+00 across 73,152 dev impressions). Dataset limitation confirmed by direct measurement: MIND has no freshness or session data at all (`article_published_time`/`session_id`/`dwell_time`/`scroll_percentage`/`is_front_page` null for every row — hardcoded at parse time). 20 unit/integration tests pass incl. a real-data leakage-boundary test; full suite 327/327 |
| **Q6 design note** | ✅ **Complete** (`docs/design_note_a2.tex`, 6 pages, recompiled 2026-09-18 clean via `tectonic`, page count re-verified via `pypdf` at 6 after the final MIND score landed). Covers Q1, Q2 (the literal harness + its interim result, not just citations), Q3 A/B framing for both datasets, Q5 slicing, Q9 ablation, Q4 serving/scale, and the 10× scaling argument. §2.2's MIND close now cites the real, final Codabench result (930992, 0.6766) in place of the earlier "upload is the engineer's manual step, no score yet" state |
| Q1 session gap (partially addressed above) | See Q1 row |

**Decisions this cycle (the engineer's):**
- Option A with a one-session timebox, falling back to B. The fallback has now been applied.
- Treatments: EB-NeRD = official NRMS + freshness. MIND = official NRMS + title+abstract input.
  The first MIND spec (GloVe vs random init) was re-decided, because the official NRMS is
  already GloVe-initialised.
- Q2: leaderboards keep ranking each impression's real candidate list, plus a separate,
  honestly-reported true-retrieval-ceiling evaluation.

## A2 Deliverables Checklist (Q7–Q9), re-verified 2026-09-17 end-to-end (not just claimed)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | README one-command reproduce | ✅ **Re-run live a second time, 2026-09-17**, after the full-scale Q2 run and every doc change since: `scripts/a2_evaluate_scores.py`'s documented invocation reproduced ADR-015's exact cited numbers again, byte-identical to the 2026-09-14 check (control 0.6831 → treatment 0.6868, diversity −0.0060) |
| 2 | MIND leaderboard screenshot | ✅ **Resolved 2026-09-18 — real, final official score 0.6766 (submission 930992).** 930353 (the original upload) scored 0.5589 — a real pipeline bug, not a genuine result: `a2_generate_mind_test_predictions.py`'s hand-rolled history reader keyed its dict by the raw user_id while the lookup used the prefixed one, so every one of 2,370,727 impressions was scored with empty history (full root-cause writeup: ADR-015's 2026-09-18 addendum). Fixed by retiring the hand-rolled reader for the same tested `build_mind_test`/`user_history.parquet` path Candidate J and Q2 already use, plus regression tests (`test_mind_format.py`) and a new permanent pre-upload gate (`scripts/a2_check_mind_history_coverage.py`) verified against real ground truth (98.77% non-empty-history rate, exact match) before any re-upload. Same checkpoint, no retraining; regenerated via job 2700133, format-validated clean (0 malformed, 0 duplicates). **The engineer uploaded the corrected file as submission 930992 (2026-09-18 06:36), Score 0.6766** — sane, in this project's normal dev-to-test compression range (0.6868 → 0.6766), and a real +0.0304 over the previous official 901961 (0.6462). Confirmed independently on both the participate-tab submission list and the public leaderboard rank table (row 33, team "apollo19", same ID/timestamp/score). **930992 is now the selected official MIND submission, replacing 901961** — screenshots at `submissions/mind_large_test_nrms_official_treatment_fixed/leaderboard_screenshot_{upload,rank}.png` |
| 3 | EB-NeRD leaderboard screenshot | ✅ Re-confirmed present on disk 2026-09-17: `submissions/ebnerd_testset_gbdt_k/leaderboard_screenshot_{upload,rank}.png`, submission 907863, score 0.7542, ADR-013. K's leaderboard entry is unchanged — the 2026-09-12 retrain was a local artifact recovery, not a new submission, per the engineer's explicit instruction |
| 4 | AI usage log current through today | ✅ `knowledge/ai-usage-log/2026-09-11_a2-kickoff-corpus-stats-gap-analysis.md` — Prompts 1–10 verbatim plus the autonomous-execution phase (2026-09-15–17) that shipped the treatment and closed Q2. **2026-09-18: `2026-09-18_mind-submission-bug-investigation-and-fix.md`** — the 930353 root-cause investigation, fix, and the real final 930992/0.6766 result, Prompts 1–10 verbatim |
| 5 | Working tree fully committed and pushed | ✅ Re-verified 2026-09-18: `git status` clean (only the pre-existing, unrelated untracked `.claude/`), `git fetch` shows local `a2-q3-official-baseline` matches `origin/a2-q3-official-baseline` exactly, HEAD at `70242a5` |

**Resolved (2026-09-17), per the decision rule fixed in advance.** `mind_treatment_v2`'s final
(epoch-10) dev monitor AUC: **0.686786** — matching the already-evaluated 0.6868 to four decimal
places, comfortably inside "same ballpark," not a red flag. Proceeded with generating and
validating the submission per the rule set above. Full detail: ADR-015's 2026-09-17 addendum.

**Environment facts that changed since A1:**
- Ada QoS is now `low` (cpu 10, gpu 1, mem 32,000 MB per user). `u22-cpu` is devalab-only.
- `/share1` is not mounted on compute nodes.
- `$HOME` has ~5 GB free of 25.6 GB.
- None of A1's Ada artifacts survive; Candidate K's 65-feature booster must be retrained if
  it is needed.
- ADR-014's claim of "no trained J checkpoint locally" is wrong:
  `experiments/candidate_j_nrms_lite_ada_2026-08-24/nrms_lite_best.pt` (MINDsmall run) exists.

**Open risks:**
- MIND 10-epoch wall time is still unmeasured.
- Option B must be reported as a reimplementation, not "the official code".
- MIND published NRMS (67.76) is above J's 0.6462, so a faithful control should beat J on MIND.

---

# Assignment 1 record (historical, unchanged below)

> **RESOLVED (2026-08-30): Candidate K is a real, confirmed Codabench
> leaderboard win — EB-NeRD's first.** Submission ID **907863**, uploaded
> 2026-08-29 23:01, scored **0.7542** on the real leaderboard (per-date
> breakdown mean: 0.7535) — a genuine **+0.2138** over the previous deployed
> EB-NeRD submission (0.5404) and **+0.1572** over the challenge's own
> popularity baseline (0.5970), which this project's EB-NeRD line had never
> beaten before. Within **0.0157 AUC of the honest, leakage-free literature
> ceiling** (0.7699 — the real winning team's score once the organizers
> stripped features later found to leak future information). **0.80 was not
> reached — stated plainly, not rounded up.** Screenshots:
> `submissions/ebnerd_testset_gbdt_k/leaderboard_screenshot_{upload,rank}.png`.
>
> Full trail: local `ebnerd_small` **0.7514–0.7528** → a targeted audit
> (prompted by the engineer questioning whether recency was used properly)
> found short-term interest was computed once per user against a split-wide
> reference, effectively static rather than per-impression — fixed, plus two
> genuinely missed leak-safe signals added (context-article match, causal
> session position) → corrected local `ebnerd_small` **0.7581–0.7597** → real
> `ebnerd_large` training on Ada (12,063,890 train / 12,566,385 validation
> impressions, full scale) → local `ebnerd_large` validation **0.7590** (95%
> CI 0.7588–0.7593, essentially unmoved by ~10.7x more training data — a real
> null result on scale, not smoothed over) → real Codabench **0.7542**, a
> small honest compression (−0.0048) from local, not the evaporation
> (Candidate G) or near-total thinning (EB-NeRD contrastive vector) every
> other CI-clear local win in this project has shown at real test scale.
>
> Five distinct, real engineering incidents were hit and fixed getting this
> onto Ada, none of them modelling bugs: a genuine SLURM OOM (root-caused to
> two large training-matrix copies never being freed, fixed in code and by
> raising `--mem-per-cpu`); a real missing-column crash in the blind test set
> itself (`ebnerd_testset` carries no `article_id` field, unlike
> train/validation — fixed by extending the same degrade-gracefully pattern
> already used for optional history columns); a second lightgbm+torch
> segfault in the test suite (same root cause as an earlier one this session,
> fixed the same way — relocating the shared function out of the
> lightgbm-importing script); a cluster-side GPU-idle job cancellation
> pattern (fixed by splitting the pipeline into a CPU-only download stage and
> a GPU stage that starts using the GPU within seconds, plus a
> `$HOME`-persistent shared cache); and one `#SBATCH`-ordering bug in a new
> script caught and fixed before it shipped, not after another failed
> submission.
>
> Three findings worth carrying forward regardless of any candidate's score:
> 1. **The project venv was miscomputing.** numpy 1.26.4 had been source-built
>    against CPython 3.14, an unsupported combination with no released wheel,
>    and returned provably wrong array results. Rebuilt on Python 3.11 (which
>    `pyproject.toml` already declares); full suite green afterwards. **It is
>    not known whether any earlier recorded number in this project came from
>    the defective build — they have not been re-verified.**
> 2. **Two literature figures this project was working from were wrong.**
>    BlackPearl scored **88.15**, not 82.20, and no published leakage-free
>    BlackPearl number exists; the organizers ablated only the winner
>    (88.64 → **76.99**). 76.99 is the honest clean anchor. Separately,
>    Candidate H2 used `HistGradientBoostingClassifier`, not LightGBM/CatBoost.
> 3. **The BlackPearl-derived long/short-term hypothesis is falsified**, and
>    more confidently so after the recency fix and at real `ebnerd_large`
>    scale: short-term features contribute **0.8%** of model gain (down from
>    an already-small 1.9%); **freshness contributes 33.3%**, with
>    `article_age_h` the single top feature throughout. What wins on EB-NeRD
>    is *which candidate is freshest relative to the others in the same
>    in-view list* and *what the user is reading right now*
>    (`context_embed_sim`), not hierarchical long/short-term interest
>    modelling.
>
> Full detail in **ADR-013** (canonical, three addenda: 2026-08-27 recency fix
> + dataset audit, 2026-08-29 real `ebnerd_large` training + five Ada
> incidents, 2026-08-30 real Codabench result).

> **RESOLVED (2026-08-26): Candidate J is a real, confirmed Codabench
> leaderboard win.** Real score: **0.6462** (submission ID 901961, rank
> 47/91, uploaded 2026-08-26 06:58) — vs. the original baseline's 0.6195
> and Candidate G's 0.6192. **+0.0267 over the original submission, the
> first real leaderboard win this project's entire MIND candidate search
> has produced.** Full trail: MINDsmall-dev 0.6391 → MINDlarge-dev 0.6579
> (win widened, not compressed — unprecedented in this project) → real
> Codabench 0.6462 (some compression from the dev screen, as expected, but
> stayed a clear win rather than evaporating like Candidate G's did).
> Screenshots and full detail in ADR-012's 2026-08-26 addendum.

**Last Updated:** September 4, 2026 (ADR-014 — pipeline performance profiling: bottleneck named per dataset, per-hardware tables, performance ablations, and a pre-commit benchmarking habit. Previous entry follows.)

**Previously Updated:** August 30, 2026 (Candidate K / ADR-013 — EB-NeRD GBDT ranker, **RESOLVED as a real, confirmed Codabench win: submission 907863, Score 0.7542**, +0.2138 over the previous deployed EB-NeRD submission and +0.1572 over the challenge's own popularity baseline — the first time this project's EB-NeRD line has beaten either. Within 0.0157 of the honest leakage-free literature ceiling (0.7699); 0.80 was not reached. Full trail from local ebnerd_small through a real recency-computation fix, real ebnerd_large training on Ada, and the real leaderboard result is in ADR-013's three addenda. Previous session entry follows — the MIND candidate-search line, explicitly closed on Aug 22 per the engineer's own hard-stop instruction, was reopened after the engineer gained access to Ada (IIIT-H's SLURM HPC cluster) and carried all the way through to a real, confirmed Codabench leaderboard win. **Candidate J (NRMS-lite — trainable title + click-history encoders, Wu et al. 2019, GloVe-initialized): MINDsmall-dev 0.6391, MINDlarge-dev 0.6579 (win widened, not compressed), real Codabench 0.6462 (submission 901961) — a genuine +0.0267 improvement over the original 0.6195 baseline submission, first real leaderboard win this project's MIND search has ever produced.** Along the way: a real multi-hour HPC environment-setup saga (Python 3.6→3.11 via `uv`, a CUDA-version mismatch, node-local `/ssd_scratch` not being cluster-shared, throttled downloads fixed with resumable retry logic in two places), and a real prediction-generation bug (scoring catalog wrongly included train+dev, not test-only) caught by a spot-check and fixed at the root. Initial impact estimate (32/2,370,727, checked against only the one known example) was later corrected once measured directly: real diff is 2,087/2,370,727 (0.088%, ~65x the estimate) — small, but the engineer chose to resubmit the corrected set as a fourth MIND entry rather than let the narrower estimate stand. Full detail in ADR-012 (canonical, four addenda) and ADR-011's addendum (cross-reference). The Aug 22 closure and everything before it (Candidates A-I, I-long, I-pop, all real losses) remains an accurate historical record, not erased — see the Aug 22 entries below.

**Current Phase:** MIND + EB-NeRD Codabench submissions (Q5) and the design note (Q7 deliverable #2) were complete as of Aug 14, but the design note's MIND section (§3.5/§6) is now stale — it describes the candidate search as having found no local/real win, which Candidate J's real 0.6462 leaderboard result contradicts. **Updating the design note to reflect Candidate J is the one clearly remaining task from this whole line of work.** Both MIND submissions are now three: 886468 @ 0.6195, 896696 @ 0.6192, 901961 @ 0.6462 (new best). See ADR-012 (full) and ADR-011's addendum (cross-reference) for detail.

**Current Objective:** **Resolved — Candidate J is a confirmed, real Codabench leaderboard win.** Full real-data comparison, MIND candidate search: G (deployed submission) 0.6192 real leaderboard vs. 0.6195 first submission, flat; H1 0.6319 (-0.0021, loss); H2 0.5985 (-0.0355, loss); I (5 epoch) 0.6233 (-0.0107, loss); I-long (30 epoch) 0.6214 (-0.0126, loss); I-pop 0.5133 (-0.1207, confounded/inconclusive); **J, no GloVe (Ada, first attempt)** 0.6242, statistically flat vs. I; **J, with GloVe, MINDsmall-dev** **0.6391 (95% CI 0.6370-0.6412) vs. baseline 0.6340 — CI-clear win**; **J, with GloVe, MINDlarge-dev** **0.6579 (95% CI 0.6569-0.6588) vs. the real MINDlarge-dev baseline 0.6335 — a +0.0244 win, WIDER than the MINDsmall margin, not compressed**; **J, real Codabench leaderboard (submission 901961, 2026-08-26)** **0.6462 — a real +0.0267 win over the original baseline submission (0.6195) and +0.0270 over Candidate G (0.6192).** Some compression from the MINDlarge-dev screen (0.6579 → 0.6462) occurred, consistent with this project's repeated finding that local numbers don't fully transfer to the real blind test — but unlike Candidate G (whose win evaporated to flat) or the EB-NeRD contrastive result (whose edge thinned to near-nothing), Candidate J's real result stayed a clear, meaningful win. First real leaderboard win this project's entire MIND search has produced, and the first from a genuine architecture rather than a combiner. Full detail in ADR-012 (canonical, three addenda) and ADR-011's addendum (cross-reference). Round-1/round-2 candidate-search history in ADR-010, ADR-008's earlier Addendum (A/B), ADR-005's Addendum (C), and the Aug 22 entries below (still accurate as history).

---

# Component Status Summary

| Component | Status | Progress | Notes |
|-----------|--------|----------|-------|
| Data Pipeline | ✅ Complete (fast tier + ebnerd_small + MINDlarge) | 100% | `make data` builds MINDsmall + ebnerd_demo end-to-end per ADR-001/ADR-002. `ebnerd_small` built and verified (ADR-002 addendum) via opt-in `include_ebnerd_small=True`. **MINDlarge built and row-count-verified this session** (`include_mind_large=True`) — required three real algorithmic fixes to `src/datasets/mind.py`/`src/pipeline/validators.py` to survive MINDlarge's row counts on this 8GB machine (naive-loop hang, 27GB memory projection, two separate `map_infer_mask` scaling bugs); see Session Notes below. |
| Lexical Retrieval (BM25) | ✅ Complete (fast tier + ebnerd_small + MINDlarge) | 100% | `scripts/run_bm25_experiment.py --dataset {mind,ebnerd} [--bundle {small,demo}]` per ADR-005/ADR-006. Benchmarked against MINDsmall-dev, ebnerd_demo-validation, ebnerd_small-validation, **and MINDlarge-dev this session** (ADR-006 addendum) — stays local, ~10 min projected for the full 255,990-user run. |
| Semantic Retrieval | ✅ Complete (fast tier + MINDlarge) | 100% | `scripts/run_embed_experiment.py --dataset {mind,ebnerd} [--bundle {small,demo}]` per ADR-008. `paraphrase-multilingual-MiniLM-L12-v2` encoder, brute-force cosine ANN, disk-cached embeddings. Benchmarked against the same three corpora BM25 covers plus MINDlarge-dev (ADR-008 addendum) — stays local, brute-force still sufficient, FAISS still unjustified. **2026-08-19: isolated comparison against EB-NeRD's provided `contrastive_vector` artifact** (`scripts/run_contrastive_vector_experiment.py`, reusing `EmbeddingIndex`/`EmbeddingScorer`/`build_user_embedding_query`/the Q4 harness completely unchanged — one new function, `load_contrastive_index`) — on `ebnerd_small` validation, the provided artifact wins recall@K (0.58%/1.20%/2.47% vs. MiniLM's 0.14%/0.43%/1.21%) and every Q4 accuracy metric (AUC 0.5453 vs. 0.5430, MRR/nDCG@5/nDCG@10 all higher, all CI-clear) but loses Diversity@10 (0.780 vs. 0.789, CI-clear) and ties Novelty@10 — does not reverse ADR-008's decision (artifact doesn't cover MIND). See ADR-008's Addendum and `experiments/contrastive_vector_ebnerd_small_2026-08-18/`. |
| Evaluation Harness | ✅ Complete | 100% | `src/evaluation/metrics.py` (recall@K), `src/evaluation/ranking_metrics.py` (Q4: AUC/MRR/nDCG@5/nDCG@10/diversity/novelty/coverage), `src/evaluation/bootstrap.py` (shared CI substrate), `src/retrieval/score.py` (generic `Scorer` interface — BM25 and `EmbeddingScorer` both implemented, exercised through the identical unchanged harness). All bootstrap CI, warm/cold slicing (Q4.3/Q4.4). See ADR-007/ADR-008. |
| Benchmarking Framework | ✅ Complete (BM25 + semantic) | 100% | `experiments/{bm25,embed,ranking_bm25,ranking_embed}_{dataset}_{date}/{config,results}.json` pattern applied to both retrieval methods on all three fast-tier corpora plus MINDlarge-dev |
| Codabench Submission Format | ✅ Complete (MIND: converter + dev-set validation. EB-NeRD: converter + ebnerd_small validation + a real scale fix) | 100% | `src/submission/mind_format.py` — official `impression_id [rank_1,...,rank_N]` format, re-reads the raw zip directly to preserve original candidate order (the processed feature store's deterministic sort destroys it). Validated end-to-end against the real `evaluation/official/evaluate.py` on MINDlarge_dev for both BM25 and embeddings: AUC/nDCG match the project's own `ranking_metrics.py` almost exactly; the one real MRR disagreement is a verified, fully-explained metric-definition difference (official sums 1/rank over all clicked items vs. this project's first-hit-only MRR), not a converter bug. `src/submission/ebnerd_format.py` — direct port of the same design, validated against `ebnerd_small`'s validation split via the same `evaluate.py`. **This session: `read_raw_impressions` (which materialized the whole split as a `list[dict]` before writing anything) was found, by direct measurement, to project to ~16GB at ebnerd_testset's real 13,536,710-impression scale, on top of ~8.5GB for the `behaviors` DataFrame itself (pandas' own `memory_usage(deep=True)` undercounts this >3x for object-dtype list columns) — a real risk MINDlarge_test's 2.37M-impression Part 4 run never surfaced. Fixed at the root: `iter_raw_impressions` is now a generator `write_predictions`/`write_truth_file` consume one row at a time (never materializing the full list), and `read_zip_parquet`/`iter_raw_impressions` now request only the columns actually needed. `read_raw_impressions` kept as `list(iter_raw_impressions(...))` — unchanged contract, all 7 existing unit tests plus 2 new ones (equivalence + laziness) pass, full suite 164/165 (1 pre-existing skip). |
| Leaderboard Submission | ✅ Complete (MIND now has 3 submissions, EB-NeRD now has 2 submissions) | 100% | **MIND, NRMS-lite (2026-08-26): submission ID 901961, `submissions/mind_large_test_nrms_lite/prediction.zip`, Score 0.6462** — a real, substantial win over both prior MIND submissions (886468 @ 0.6195, 896696 @ 0.6192), the first real leaderboard win this project's entire MIND candidate search has produced. Full detail in ADR-012's 2026-08-26 addendum. MIND, embed (MiniLM): `submissions/mind_large_test_embed/prediction.zip` uploaded to Codabench — leaderboard **Score column 0.6195**, next three columns 0.3006 / 0.3225 / 0.3785 (submission ID 886468, 2026-08-12 13:01). EB-NeRD (competition 2469), MiniLM: `submissions/ebnerd_testset_embed/prediction.zip` uploaded — leaderboard **Score column 0.5404**, next three columns 0.3447 / 0.3823 / 0.4613 (submission ID 888045, 2026-08-13 23:08). Column headers were cropped out of both screenshots, so the exact metric labels aren't directly confirmed — but the four values line up closely with this project's own locally-measured AUC/MRR/nDCG@5/nDCG@10 for embeddings on `ebnerd_small` (0.5430 / 0.3437 / 0.3804 / 0.4591), in that order, strongly suggesting Score = AUC. **EB-NeRD, contrastive vector (2026-08-21): submission ID 896072, `prediction_contrastive.zip`, also Score 0.5404** — an identical rounded Score to 888045 was verified NOT to mean a duplicate upload (different checksums, 99.48% of 13,536,710 impressions rank differently for the same ID). The per-submission detail view (opened individually per submission ID, since Codabench's detail page itself shows no ID — an initial pass nearly mis-attributed a third-party competitor's stray detail table to one of these two) resolves the tie: mean AUC over the 8-date/50%-of-testset breakdown is 0.5402 (contrastive) vs. 0.5397 (MiniLM), a real but thin +0.0005 edge, far smaller than the ~0.0023 CI-clear gap the same comparison showed on local `ebnerd_small` validation (ADR-008's 2026-08-21 Addendum). **MIND, cohort-gated combiner (2026-08-22): submission ID 896696, `submissions/mind_large_test_gated_cohort/prediction.zip`, Score 0.6192** — essentially flat vs. 886468's 0.6195 (-0.0003), despite CI-clear local wins at both the MINDsmall-dev screen (+0.0027) and a MINDlarge-dev re-verification (+0.0019, ADR-010's Addendum). No pipeline defect found (format re-verified with the same discipline as every other submission). Same category of finding as the EB-NeRD contrastive-vector result immediately above — a CI-clear local win compressing substantially at real blind-test scale — now observed twice, on two different datasets; documented as a real, honest pattern in ADR-010's 2026-08-22 Addendum and the design note's §3.5/§6, not smoothed into a false "it worked." Leaderboard runs on the real held-out test set, so exact match to local validation isn't expected regardless — the agreement (or lack of it) is a coherence check, not a reproduction. Screenshots of both the submission-upload confirmation and the leaderboard rank saved to `submissions/{mind_large_test_embed,ebnerd_testset_embed}/leaderboard_screenshot_{upload,rank}.png` (gitignored along with the rest of `submissions/`, per Q8 — source material for the Q6 design note); the second EB-NeRD and second MIND submissions' own screenshots remain on the Desktop, not yet copied into `submissions/`. |

---

# Candidate K — EB-NeRD GBDT Learning-to-Rank (ADR-013, 2026-08-26 → 2026-08-30, RESOLVED)

**Real Codabench result: submission 907863, Score 0.7542** — see the banner
above and ADR-013's 2026-08-30 addendum for the full comparison. Everything
below this line was the local-validation state partway through the session,
kept as an accurate record of how the candidate got there, not the final word.

| Item | Status |
|---|---|
| Feature module `src/retrieval/ebnerd_features.py` (65 features after the recency fix + audit) | ✅ Built, 39+ unit tests |
| Experiment `scripts/run_ebnerd_gbdt_experiment.py` (3 arms + 3 baselines) | ✅ Run on ebnerd_small and real ebnerd_large |
| Vectorised AUC (`per_impression_auc`, relocated to `src/evaluation/ranking_metrics.py`) | ✅ 11 tests prove exact match to `safe_auc` incl. ties/multi-click/degenerate |
| Streaming validation scoring | ✅ Verified bit-identical to the materialised path; 18 min → 149 s, memory-bounded |
| Leakage audit, all features individually | ✅ ADR-013 table; guard enforced in code + parametrised tests |
| Leakage integration tests on the real bundle | ✅ 8 tests, `slow`-marked |
| Submission generator (chunked) | ✅ Dry-run scored **0.7524 by the official `evaluate.py`**, exact match to the harness; real testset run wrote 13,536,710 lines matching the true count exactly |
| Ada pipeline (`ebnerd_prepare_bundle.py` + `ebnerd_gbdt_ada_{prepare,train,score_only}.sbatch`) | ✅ Run for real on Ada — see ADR-013's 2026-08-29 addendum for five distinct incidents hit and fixed |
| ebnerd_large training | ✅ Real scale: 12,063,890 train / 12,566,385 validation impressions; local validation AUC 0.7590 |
| Codabench submission | ✅ **Submission 907863, Score 0.7542** — arm shipped: **K_rank_nopos** |

**Local result (ebnerd_small validation, 244,647 impressions / 15,342 users):**

| Arm | AUC | 95% CI |
|---|---|---|
| K_cls (binary) | 0.7528 | 0.7511–0.7548 |
| K_rank_nopos (position withheld) — **the arm to ship** | 0.7524 | 0.7507–0.7543 |
| K_rank (lambdarank) | 0.7514 | 0.7496–0.7533 |
| embed_sim (deployed method, same split) | 0.5430 | 0.5415–0.5444 |
| random | 0.4993 | 0.4979–0.5006 |
| history-window popularity | 0.4269 | 0.4252–0.4288 |

**Harness sanity:** `embed_sim` reproduces ADR-008 addendum 2's 0.5430 on this
split to four decimals, and `random` lands at 0.4993 — the evaluation path is
the same instrument every earlier candidate was judged with.

**The popularity row is NOT the challenge's 0.5970 baseline.** It counts clicks
over the 21-day *history* window (the only leak-safe source); history-popular
articles are by construction older, and older is anti-predictive in news. The
two numbers are different constructions and are not compared.

**Position-bias ablation:** withholding `position_in_view` /
`relative_position_in_view` *improved* AUC by a CI-clear +0.0011, so the result
is not "learned Ekstra Bladet's own ranker". K_rank_nopos is quotable without
caveat, which is why it is the arm selected for submission.

**Known data artifact (bounded, not hidden):** 13 articles across 5.5M candidate
rows carry a `published_time` later than an impression showing them (0.025%
train / 0.002% validation). CTR in those rows is *lower* than baseline, so it is
not exploitable; values are left untransformed because clipping to zero would
make exactly those rows look maximally fresh. Bounded by test.


---

# Deliverables Checklist (Q7)

> **Q6 vs. Q7 numbering, resolved (2026-08-30):** these are two different
> sections of the same assignment brief, not a contradiction. The
> verbatim text obtained this session — **"Q6. Design Note (≤4 pages):
> What you built and key design choices / Alternatives considered and why
> / Observations from experiments (lexical vs. semantic, dataset
> differences) / Where your pipeline breaks at 10×scale"** — is Part I's
> content requirement for the note itself. The text captured in this
> session's log on 2026-08-14 (`knowledge/ai-usage-log/2026-08-14_q7-q9-reconciliation-latex-conversion.md`)
> is explicitly headed **"Part II: Deliverables & Policies" → "Q7.
> Deliverables"**, i.e. a later section listing the design note as
> submission item #2 alongside code, screenshots, and the AI usage log.
> The "Part II" heading implies Part I runs at least through Q6, which
> lines up exactly. Both point at the same artifact (`docs/design_note.tex`
> / `.pdf`); the checklist below is kept under its historical "(Q7)"
> heading (submission-logistics framing), while the note's own section
> structure is checked against Q6's four content bullets directly, not
> assumed to already satisfy them. **Not independently confirmed against
> the original assignment PDF** — resolved from internal consistency of
> text already captured in this repo, not a new document; flag to the
> engineer if a copy of the full brief surfaces and contradicts this.

Honest status against the assignment's four required deliverables, updated at the start of each session rather than assumed complete.

| # | Deliverable | Status | Notes |
|---|-------------|--------|-------|
| 1 | Code (GitHub Classroom) | ✅ Complete | Data pipeline (now including MINDlarge), BM25 retrieval, semantic (embedding) retrieval, Q4 evaluation harness, both Q5 official-format converters, and the Q9 leaky-feature ablation (`scripts/run_leakage_ablation.py`, ADR-009) all implemented and tested. `README.md`'s one-command reproduce (`make data`) actually re-run to verify — found and fixed a stale "expected output" block that no longer matched real output, plus a stray leading typo. `.gitignore` re-verified against Q8's explicit list (`*.zip`/`*.pt`/`*.ckpt`/`__pycache__/`/`data/`, all present) and `git rev-list --objects --all`-checked for large blobs across **all of history**, not just the working tree (none found; largest blob ever committed is `poetry.lock` at 340,338 bytes, zero blobs exceed 1MB). **A prior pass this session marked this row Complete prematurely** — a follow-up verification found `tests/fixtures/{MINDsmall_train_sample,MINDlarge_test_sample,ebnerd_demo_sample}.zip` were gitignored and untracked (caught by the blanket `*.zip` rule, no exception for `tests/fixtures/`), so `test_ebnerd_loader.py`/`test_mind_loader.py`/`test_mind_format.py`/`test_ebnerd_format.py` (46 tests) would fail `FileNotFoundError` on a fresh clone. **Fixed, not by a gitignore exception (would violate Q8):** `scripts/generate_test_fixtures.py` (already existed, already produced exactly the right three files at the right paths — confirmed by running it fresh and diffing filenames against what the four test files actually read) is now wired in via a new `fixtures` Make target that `test`/`test-unit` depend on (Makefile, `test-integration`/`test-reproducibility` don't need it — neither reads `tests/fixtures/`). Verified for real, not assumed: deleted all three fixture zips, ran `make test` end to end with them absent beforehand (the actual fresh-clone condition) — **180 passed, 1 expected skip, 6 deselected, zero failures**, fixtures regenerated automatically as a side effect of the `fixtures` prerequisite. README's "Verify Installation" and "Run individual suites" sections, which previously called `pytest` directly (bypassing this dependency), updated to call the `make` targets instead. |
| 2 | Design note (≤4 pages, Moodle) | ✅ Complete | **2026-08-30: brought current with the real, final EB-NeRD result.** The note (structure: What We Built / Choices / Observations §3.1–3.7 / Anti-Gaming \& Leakage / Where It Breaks at 10× / Limitations) covers Q6's four required bullets (what was built + choices, alternatives considered, observations incl. lexical-vs-semantic + dataset differences, 10× breakage) under that section labeling — see the Q6/Q7 numbering note added above the checklist header. Previously the note only reflected Candidate J (MIND); it had **zero content on Candidate K (EB-NeRD's real Codabench win, 0.7542)** despite that being this project's most recent, and highest-value, real result — added this session as new §3.7, a new leaderboard-table row (907863, 0.7542, 2026-08-29 23:01), the real `ebnerd_large` OOM incident in §5, and a Limitations entry stating plainly that the internally-discussed 0.80 target was **not** reached (0.7542, 0.0157 below the honest leakage-free literature ceiling of 76.99) — sourced directly from ADR-013's three addenda, not recalled. Also closed out MIND's corrected-resubmission thread (identical 0.6462, previously described as "still in progress"). `docs/design_note.tex` (article, 10.5pt, 0.75in margins, single column) compiles via `tectonic` to `docs/design_note.pdf` — **re-verified with `pypdf` (freshly installed this session, `poetry add --group dev pypdf`; it was not actually an installed dependency despite the prior entry below claiming a `pypdf`-based check) at exactly 4 pages**, not eyeballed and not carried over from the prior recorded count. Fitting the added content required real trimming (condensed several §3.5 sub-paragraphs, dropped the day-by-day EB-NeRD contrastive table in favor of the mean it already reported in prose) plus a small, standard `\enlargethispage{3\baselineskip}` (~0.5in) to absorb a two-line residual on page 4 — not used to hide a full extra page's worth of content, which was verified by checking the actual per-page character counts before and after. A real numeric-consistency bug was caught and fixed during the trim: an early compressed draft cited "89.24 as submitted → 76.99, an 11.65-point collapse," but 89.24−76.99 is 12.25, not 11.65 — the correct 11.65-point figure is 88.64 (organizers' own ablation-arm-with-features) minus 76.99; fixed to cite 88.64 explicitly rather than the misleading headline figure. A sample of the new content's numbers were grep/direct-execution-verified against source, not just re-quoted from the ADR text: `FEATURE_NAMES` count in `src/retrieval/ebnerd_features.py` is exactly 65 (matches ADR-013); `FORBIDDEN_RAW_COLUMNS` matches the claimed leaky-field list exactly; `total_inviews`/`total_pageviews`/`total_read_time` confirmed absent from `src/datasets/ebnerd.py` by grep (zero matches); `submissions/ebnerd_testset_gbdt_k/prediction.zip` opened directly — contains `predictions.txt` (13,536,710 lines, exact match to the claimed real test-set count) — while both MIND submission zips contain the singular `prediction.txt`, confirming the two competitions' differing filename requirement cited in ADR-013. |
| 3 | Leaderboard screenshots (both Codabench competitions) | ✅ Complete | MIND: uploaded and confirmed — score 0.6195, submission ID 886468, 2026-08-12 13:01. EB-NeRD (competition 2469): uploaded and confirmed — score 0.5404, submission ID 888045, 2026-08-13 23:08. Screenshots saved to `submissions/{mind_large_test_embed,ebnerd_testset_embed}/leaderboard_screenshot_{upload,rank}.png`. **This session: found these were untrackable in git** (blanket `submissions/` gitignore rule) — fixed `.gitignore` (`submissions/*` / `!submissions/*/` / `submissions/*/*` / `!submissions/*/*.png`) so the six screenshot PNGs are trackable while `submissions/`'s large prediction/truth files (up to 278MB) stay ignored; verified with `git check-ignore`/`git add -n`. |
| 4 | AI usage log (prompts + AI-vs-human marking) | 🟢 Ongoing | `knowledge/ai-usage-log/` exists; one file per session (`YYYY-MM-DD_<topic>.md`), written live per CLAUDE.md's "Prompt & Session Logging" section. This session's log: `2026-08-14_q7-q9-reconciliation-latex-conversion.md`. |

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
| ADR-008 (addendum 2, 2026-08-19) | Contrastive-Vector-vs-MiniLM Isolated Comparison | Decided (does not reverse ADR-008) | Measures, for the first time, the accuracy cost of ADR-008's original rejection of EB-NeRD's provided embedding artifacts. On `ebnerd_small` validation, EB-NeRD's provided `contrastive_vector` artifact CI-clear-wins recall@K and every Q4 accuracy metric (AUC 0.5453 vs. 0.5430; MRR/nDCG@5/nDCG@10 all higher) but CI-clear-loses Diversity@10 (0.780 vs. 0.789) and ties Novelty@10 — a real trade-off, not a blanket win. Doesn't reverse ADR-008: the artifact only covers EB-NeRD, so adopting it would still break the single-embedding-space property Q3.5/Q4.5 depends on. Whether to submit a second, EB-NeRD-only leaderboard entry is left open for the engineer. |
| ADR-008 (addendum 3, 2026-08-21) | MIND Entity-Embedding + BM25/Embedding Hybrid Screens | Decided (does not reverse ADR-008) | Pre-submission screens against MIND's deployed baseline (AUC 0.6340, CI 0.6319-0.6361). Entity embeddings (86.1% article coverage, confidence-weighted pooling of MIND's own `entity_embedding.vec`): 0.5525 (CI 0.5503-0.5546) — CI-clear loss. Untuned 50/50 BM25+embedding hybrid: 0.6263 (CI 0.6242-0.6284) — CI-clear loss (blending in BM25's weaker signal dilutes rather than complements). Neither adopted. |
| ADR-005 (addendum, 2026-08-21) | Recency-Weighted MIND Embedding Query (Candidate C) | Decided (does not reverse ADR-005) | Re-tests ADR-005's originally un-benchmarked rejection of recency weighting, this time for MIND's *embedding*-based user representation (decay=0.9, untuned), not BM25. Result: 0.6265 (CI 0.6243-0.6286) vs. baseline 0.6340 (CI 0.6319-0.6361) — CI-clear loss, including for warm users specifically, where the "recent clicks are more predictive" intuition was expected to help most. A real negative result, not just an untested risk anymore — the underlying MIND history-order assumption remains unverified. |
| ADR-010 (2026-08-21) | MIND Second-Submission Candidate Search, Round 2 (Symbolic Overlap, Popularity, Learned Combiner, Cohort Gating) | Decided (adopt Candidate G, conditional) | Symbolic category/entity overlap (D: 0.6134) and train-popularity-only (E: 0.5318) both lose CI-clear. A 7-feature logistic-regression combiner trained on MINDsmall-train (F) loses overall (0.6255) but wins CI-clear on the cold cohort (0.5926 vs. baseline's 0.5737) — the first win either round produced. Cohort-gated routing built directly from that split (G: deployed embed scorer for warm, F's combiner for cold) gets overall AUC 0.6366; marginal CIs overlap the baseline's narrowly, but the statistically correct **paired** bootstrap (G and baseline share almost all impressions) shows +0.0027 (95% CI +0.0018 to +0.0034), excluding zero — a real win. Adoption is conditional on deciding whether to re-verify at MINDlarge scale before an actual Codabench submission (open, for the engineer). |
| ADR-010 (addendum, 2026-08-21/22) | MINDlarge Verification + Second-Submission Build | Decided (condition resolved) | Candidate G re-verified at MINDlarge-dev scale on Kaggle: real, CI-clear paired win, +0.0019 AUC (95% CI +0.0015 to +0.0023) — same direction as the MINDsmall-dev screen, smaller effect size. Getting there fixed three real, previously-latent bugs (flat-packed HF-mirror zip, hardcoded Mac-only `mps` device, a full-column read that drove the local machine into heavy swapping), all root-caused and unit-tested, none Kaggle-specific. `scripts/generate_mind_gated_predictions.py` then generated real MINDlarge_test predictions locally (~4.2hr; feasible since this scale had run locally before and the processed bundle/embedding cache already existed) — a pre-run smoke test caught a fourth bug (multi-feature combiner producing NaN, not -inf, on MIND's documented `N89741`-style missing-candidate quirk), fixed before the real run. Output verified (exact line count, zero malformed permutations, all 32 affected impressions spot-checked) and packaged identically to the first submission at `submissions/mind_large_test_gated_cohort/prediction.zip`. Not yet uploaded — engineer's explicit action. |
| ADR-011 (2026-08-22, + addendum) | Neural Candidate-Aware Attention Re-ranker (Candidates I, I-long, I-pop) | Decided (not adopted) | Attention over frozen MiniLM embeddings (no trainable text encoder). I (5 epoch): 0.6233, CI-clear loss. I-long (30 epoch): 0.6214, CI-clear loss — holdout AUC climbed every epoch (0.6067→0.6372) while real dev AUC got worse, a clean answer that more training wasn't the fix. I-pop (+popularity term): 0.5133, confounded by an unnormalized-feature implementation gap, reported as inconclusive not a clean second negative. 2026-08-24 addendum cross-references ADR-012 (Candidate J), a different architecture that reopened this line with a real win. |
| ADR-012 (2026-08-22 decision, 2026-08-24/25 real results) | NRMS-Lite: Trainable Title + History Encoders (Candidate J) | Decided — real, CI-clear WIN at both scales | Trainable title-self-attention + history-self-attention encoders (Wu et al. 2019), replacing I/I-long/I-pop's frozen-embedding approach. MINDsmall-dev, no GloVe (download failed): 0.6242, flat vs. I. MINDsmall-dev, with GloVe: **0.6391 (CI 0.6370-0.6412) vs. baseline 0.6340 — win.** MINDlarge-dev re-verification (2026-08-25, ADR-010's standing practice before any submission decision): **0.6579 (CI 0.6569-0.6588) vs. the real MINDlarge-dev baseline 0.6335 — +0.0244, WIDER than the MINDsmall margin** — the first candidate in this project where a local win got bigger, not smaller, at larger scale (contrast Candidate G: +0.0027 local → +0.0019 MINDlarge-dev → flat at the real Codabench test). Honest caveat at both scales: checkpoints directly on real dev, not an internal holdout — mildly optimistic, increasingly unlikely to explain the win away given the MINDlarge margin's size. **2026-08-26: real MINDlarge_test predictions generated, packaged, and uploaded.** One real bug found via the established N89741 spot-check (scoring catalog wrongly included train+dev, not test-only) and fixed at the root; initial impact estimate (32/2,370,727, 0.0013%) later corrected via a direct diff once the fix was verified: real measured impact 2,087/2,370,727 (0.088%, ~65x the estimate). **Real Codabench result: submission 901961, Score 0.6462 — a genuine leaderboard win over both the original submission (0.6195) and Candidate G (0.6192).** Given the corrected impact number, the engineer resubmitted the corrected prediction set as a fourth MIND entry — **real score: 0.6462, identical to the original submission**, confirming empirically that the catalog bug never put the submitted result in question. Screenshots and both prediction sets in `submissions/mind_large_test_nrms_lite{,_corrected}/`. **Candidate J is fully closed out: verified at MINDsmall-dev, MINDlarge-dev, and twice on the real Codabench leaderboard.** |

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
- [x] Part 4 (manual half): upload `prediction.zip` to the MIND Codabench leaderboard, capture screenshot — done by the engineer (score 0.6195, ID 886468, 2026-08-12 13:01; screenshots saved this session)
- [x] Q6: write the design note (≤4 pages) — **complete as of 2026-08-30**, now covering both real leaderboard results, Candidate J, and Candidate K (see Deliverables Checklist #2 above)
- [x] EB-NeRD's own Codabench submission format: Part 0 resolved — engineer ran `notebooks/ebnerd_part0_kaggle_investigation.py` on Kaggle against the real `predictions_large_random.zip`/`ebnerd_testset.zip`/`articles_large_only.zip` and relayed results (see Session Notes)
- [x] Part 1 (`src/submission/ebnerd_format.py`, validated against ebnerd_small): complete this session — 7 unit tests, end-to-end validation against `evaluation/official/evaluate.py` for both BM25 and embeddings (see Session Notes)
- [x] Part 2 prep: `notebooks/ebnerd_part2_kaggle_test_run.py` + `notebooks/ebnerd_part2_src_bundle.zip` written this session, import-verified, logic-checked locally against `ebnerd_small` (see Session Notes) — a real memory-scaling bug found and fixed in `src/submission/ebnerd_format.py` along the way
- [x] Part 2 execution (engineer ran the notebook on Kaggle) and Part 3 (submit `prediction.zip` + screenshot): done — `submissions/ebnerd_testset_embed/prediction.zip` uploaded to competition 2469, score 0.5404, ID 888045, 2026-08-13 23:08; screenshots saved this session
- [x] Q9 design-note note: confirmed in writing (2026-08-30, `grep -rn "total_inviews\|total_pageviews\|total_read_time" src/` — the only matches are `src/retrieval/ebnerd_features.py`'s `FORBIDDEN_RAW_COLUMNS` guard and its docstring, which exist specifically to prevent these fields from being read; they are never read anywhere in the pipeline)

## Upcoming

- **New, open (2026-08-26):** Candidate J's real MINDlarge_test submission (`submissions/mind_large_test_nrms_lite/prediction.zip`) is generated, validated, and ready — only the engineer's manual Codabench upload remains. Once uploaded, capture the leaderboard screenshot (same convention as every prior submission) and update this file + ADR-012 with the real score. The design note's §3.5/§6 MIND section also needs updating to reflect Candidate J's result — currently still describes the search as having found no local win.
- **Done (2026-08-21):** the second-submission decision from 2026-08-19 is complete — `notebooks/ebnerd_contrastive_vector_testset_kaggle_run.py` ran on Kaggle, `prediction_contrastive.zip` was submitted to competition 2469 (ID 896072, Score 0.5404, tying the rounded MiniLM score) and verified genuinely distinct from the original submission (checksums, 99.48% of impressions re-ranked, Cell 8's own line-count/malformed check re-run locally and clean). The real-test-set detail-view comparison (mean AUC 0.5402 contrastive vs. 0.5397 MiniLM, +0.0005) is a real but much thinner edge than local `ebnerd_small` validation's CI-clear +0.0023 gap — see ADR-008's 2026-08-21 Addendum ("Second EB-NeRD Submission") and `docs/design_note.md`/`.tex` §3.5 for the full write-up. `prediction_contrastive.zip` currently lives at `~/Downloads/`, not yet moved into `submissions/ebnerd_testset_contrastive/` — a housekeeping step still open.
- Both Codabench leaderboard submissions are now done (see Component Status and Deliverables Checklist above), and the design note (Q6, ≤4 pages) is complete as of 2026-08-30, covering Candidate J's MIND win and Candidate K's EB-NeRD win — no deliverable is blocked on it any longer.
- **Open, for the engineer (2026-08-30):** this repo has no git remote configured (`git remote -v` returns empty) — how/where this repo actually gets submitted (a GitHub Classroom remote to push to, a zip upload, etc.) is unresolved and needs a decision before submission. Also open: a fourth EB-NeRD Codabench submission (`Running prediction.zip`, ID 907906) was mid-flight per ADR-013's 2026-08-30 Addendum and was never followed up on — its outcome, if any, is not recorded anywhere in this repo.
- `README.md` needs a `scripts/run_embed_experiment.py` / `--method embed` usage note, plus `scripts/generate_mind_predictions.py` — not updated yet, flagged again for next.
- `ebnerd_large` remains undownloaded/unverified — lower priority unless the assignment specifically requires the `large` tier for leaderboard submission.
- ADR-008 flags a possible future investigation (not required this session): a curated near-duplicate/paraphrase evaluation set to validate the encoder's discrimination quality more directly than the category-proxy check used here.
- A pre-existing pandas `FutureWarning` (`Index.insert` with object-dtype, inside `validate_table`) surfaced repeatedly in recent sessions — cosmetic, not chased down, worth a quick look eventually.
- BM25 test-split predictions were not generated this session (embeddings won clearly on dev — AUC 0.6335 vs. 0.5699 — so this is the already-justified single-method choice per this session's brief); can be added later if wanted.

---

# Session Notes

## August 21, 2026 — EB-NeRD Second Submission Verified: Contrastive Vector on the Real Test Set (ADR-008 Addendum)

### Context

Objective: the contrastive-vector EB-NeRD submission (896072, prepared
2026-08-19, run on Kaggle and submitted 2026-08-21) scored an identical
rounded Score (0.5404) to the original MiniLM submission (888045) —
surprising, given the contrastive vector clearly beat MiniLM on local
`ebnerd_small` validation (ADR-008's 2026-08-19 Addendum). Before
concluding anything, confirm the two submitted files are actually
different, then pull the real per-day breakdown to see what the rounded
Score column is hiding.

### What was done

- Located both files: `submissions/ebnerd_testset_embed/prediction.zip`
  (888045) and `~/Downloads/prediction_contrastive.zip` (896072, the
  downloaded Kaggle output, not yet moved into `submissions/`).
- **Checksummed and diffed both `predictions.txt` files directly**, rather
  than trusting the identical Score: different SHA-256/CRC-32/MD5;
  line-by-line comparison of all 13,536,710 lines found 0 impression-ID
  misalignments (both walk the real zip in identical order) and 99.48% of
  lines carry a genuinely different ranking for the same impression ID —
  the fingerprint of two independent scoring runs, not a duplicate upload
  or a partial fallback to MiniLM. Confirmed directly in the notebook
  (`notebooks/ebnerd_contrastive_vector_testset_kaggle_run.py`) that
  `load_contrastive_index` was the function actually called, not
  `build_embedding_index`.
- **Re-ran Cell 8's own line-count/malformed-permutation check locally**
  against the downloaded file (13,536,710/13,536,710 lines, 0 malformed) —
  the actual Kaggle-side Cell 8 output was never relayed/logged for this
  run (a real process gap, flagged for next time), so this local re-check
  was the only way to confirm the run completed cleanly rather than
  resuming from a truncated checkpoint.
- **Pulled the real per-day AUC/MRR/nDCG breakdown from Codabench's
  per-submission detail view** — but the attribution (which detail table
  belongs to which submission ID) was genuinely ambiguous at first: the
  detail page shows no submission ID, and three different per-day tables
  turned up across the engineer's screenshots, not two. One (MEAN AUC
  0.5123) matched neither submission and was set aside as a likely stray
  click on an unrelated competitor's row while browsing the public
  leaderboard (which lists every participant, not just this account).
  Rather than guess between the remaining two candidates using screenshot
  timing as a proxy — which pointed the *opposite* direction from the
  engineer's initial recollection — asked the engineer to re-open each
  submission's eye icon individually and confirm. Correct mapping:
  MiniLM (888045) mean AUC 0.5397, contrastive (896072) mean AUC 0.5402.

### Result (see ADR-008's 2026-08-21 Addendum for full detail)

Contrastive leads the mean AUC by **+0.0005** (0.5402 vs. 0.5397, 8-date/
50%-of-testset breakdown) and every non-AUC metric shown (MRR, nDCG@5,
nDCG@10) by a larger relative margin — but the AUC sign flips day-to-day
three times, and the margin is far thinner than the **~0.0023 CI-clear
gap** the same comparison showed on local `ebnerd_small` validation. Real,
same-direction evidence, not confirmation of a large effect — the local
edge was itself small relative to its own CI, so a thinner or noisier
margin on a much larger, differently-distributed real test population is
ordinary sampling behavior, not evidence of a pipeline defect. No bug was
found anywhere in the submission pipeline.

### Updated

- `decisions/ADR-008-semantic-retrieval-design.md` — new Addendum section
  ("Second EB-NeRD Submission"), inserted directly after the 2026-08-19
  Addendum whose own "Conditions for Revisiting" this one resolves;
  history preserved, not overwritten.
- `docs/design_note.md` §3.5 (new submission row, per-day AUC table,
  honest margin comparison) and §6 (the "remains an open decision" bullet
  was stale — updated to point at the now-real result). `docs/design_note.tex`
  mirrored and recompiled — the addition initially pushed the PDF to 5
  real pages (checked via `pypdf`, not assumed); brought back to the
  original 4-page budget by trimming both the new addition and several
  existing Limitations bullets for concision (not by shrinking margins —
  a margin reduction was tried first, produced an 85pt real overfull
  \hbox, and was reverted rather than shipped).
- This file (Leaderboard Submission row, the stale 2026-08-19 "In
  progress" bullet, this entry).

### Genuinely still open

- `prediction_contrastive.zip` is still at `~/Downloads/`, not moved into
  `submissions/ebnerd_testset_contrastive/` alongside the project's other
  submission artifacts — housekeeping, not urgent.
- Only "50% of the testset" is reflected in Codabench's detail view per
  its own footnote; the other half's numbers were never obtained.
- The Kaggle-side Cell 8 output for this specific run was never
  relayed/logged (unlike every prior Kaggle run in this project) — the
  local re-check substituted for it this session, but the gap itself is
  worth closing for future runs.

### Related

- `notebooks/ebnerd_contrastive_vector_testset_kaggle_run.py`,
  `ebnerd_contrastive_vector_testset_src_bundle.zip`
- `submissions/ebnerd_testset_embed/prediction.zip` (888045),
  `~/Downloads/prediction_contrastive.zip` (896072)
- `knowledge/ai-usage-log/2026-08-19_contrastive-vector-testset-submission.md`,
  `knowledge/ai-usage-log/2026-08-21_contrastive-vector-submission-verification.md`

---

## August 26, 2026 (latest) — Candidate J: Real MINDlarge_test Submission Generated (ADR-012 Addendum)

### Context

Direct continuation from the Aug 25 entry below: given the MINDlarge-dev
win widened rather than compressed, the engineer chose to pursue a real
Codabench resubmission rather than stop at local verification.

### What was done

New `NRMSLiteScorer` (`src/retrieval/nrms_training.py`) implements the
`Scorer` protocol so the trained checkpoint plugs into
`src/submission/mind_format.py::write_predictions` unchanged, the same
path every real submission in this project has used. New
`scripts/generate_mind_nrms_predictions.py` reconstructs the model's
word vocabulary deterministically (never saved during training, only the
weights were) and verifies the reconstruction two independent ways
before trusting it. Real run: 2,370,727 predictions in 138.9 minutes,
hitting one environment gap along the way (missing `rank_bm25` in the
minimal Ada venv, pulled in transitively by `mind_format.py`'s `Scorer`
type import — installed).

### Key Finding — a real bug, caught by validation, not review

Full validation (exact line count, zero malformed permutations across
all 2,370,727 lines) passed cleanly. The N89741 spot-check (MIND's
documented missing-candidate quirk, 32 known-affected impressions) did
not: 0/32 had it correctly ranked last. Root cause: the scoring article
catalog wrongly included train+dev articles, not just test's own (every
other scorer in this project uses the target split's own catalog only,
ADR-005's convention) — this "resurrected" the missing candidate with a
borrowed title instead of the established `-inf` fallback. Fixed at the
root. Practical impact assessed directly, not assumed: 32/2,370,727
impressions (0.0013%) — every other candidate has an identical title
regardless of which split's catalog it's read from.

**Engineer's explicit call:** submit the original, otherwise-fully-
validated prediction set as-is rather than block on a multi-hour rerun
for a discrepancy this narrow; a corrected re-run proceeds separately
(`OUT_DIR=$HOME/mind_nrms_predictions_corrected`) for completeness.

### Resolved Same Day: Real Codabench Result

The engineer uploaded `prediction.zip` manually. **Submission ID 901961,
Score 0.6462** — confirmed via both the submission-upload confirmation
and public leaderboard rank screenshots (rank 47/91, username
`apollo19`), both saved to
`submissions/mind_large_test_nrms_lite/leaderboard_screenshot_{upload,rank}.png`.
A real, substantial win: +0.0267 over the original baseline submission
(0.6195), +0.0270 over Candidate G (0.6192) — the first real leaderboard
win this project's entire MIND candidate search has produced. Some
compression from the MINDlarge-dev screen occurred (0.6579 → 0.6462), as
expected given this project's repeated observation that local numbers
don't fully transfer to the real blind test, but unlike Candidate G or
the EB-NeRD contrastive result, the win did not compress away to flat.

### What's Next

**Design note (`docs/design_note.tex`/`.md`) §3.5/§6 need updating** —
they currently describe the MIND candidate search as having found no
real local or leaderboard win, which is now stale. This is the one clear
remaining task before the Aug 27 assignment deadline. The corrected
prediction re-run (fixing the narrow N89741-catalog bug) may still be
running in the background on Ada for methodological completeness — not
blocking, not required before the deadline.

### Related

- `decisions/ADR-012-nrms-lite-candidate-j.md`'s 2026-08-26 addendum (full detail)
- `submissions/mind_large_test_nrms_lite/prediction.zip`
- `src/retrieval/nrms_training.py::NRMSLiteScorer`,
  `scripts/generate_mind_nrms_predictions.py`

---

## August 25, 2026 — Candidate J: MINDlarge-Dev Re-Verification, the Win Widens (ADR-012 Addendum)

### Context

Direct continuation from the Aug 22-24 entry below: per this project's
standing practice (ADR-010) of verifying any local win at MINDlarge scale
before a Codabench submission decision — local wins have compressed at
real scale twice before in this project (Candidate G, EB-NeRD contrastive
vector) — the engineer asked to run that verification for Candidate J.

### What was done

Added `--bundle {small,large}` to `mind_nrms_lite_ada_run.py` (small
stays default) and a new `mind_nrms_lite_ada_large.sbatch`, projecting
real MINDlarge cost from real MINDsmall-Ada timing before committing
(~110 min/epoch projected; ~117 min/epoch measured, within 7%). Two real
infrastructure bugs surfaced and fixed at the root, not worked around:
`src/pipeline/download.py::download_mind_bundle` (this project's shared
MIND downloader, used by `make data` too) had the same no-retry/no-resume
flaw as the earlier GloVe fix, hit for real downloading `MINDlarge_train.zip`
(531MB); fixed with a pure-`urllib` Range-header retry/resume loop
(deliberately not `wget`, which isn't preinstalled on macOS and would
break `make data` there) — new `tests/unit/test_download.py`, 6 tests.
Separately, a node-pin (`--nodelist=gnode007`, meant to reuse a cached
GloVe file and save a ~9h re-download) backfired in practice — the node
stayed busy long enough that the queue wait itself exceeded the
re-download cost — reverted after being tried, not just reasoned about.
The account was also upgraded `low`→`medium` QoS mid-session, breaking
both sbatch scripts (`Invalid qos specification`) until fixed.

### Key Finding

**MINDlarge-dev: 0.6579 (95% CI 0.6569-0.6588) vs. the real MINDlarge-dev
baseline 0.6335 (verified against the official `evaluate.py`, not the
MINDsmall baseline the script's own hardcoded constant printed by
mistake — caught and fixed before it corrupted the write-up) — a +0.0244
win, nearly 5x the MINDsmall-dev margin (+0.0051), not compressed.** This
is the first time in this project's history a local win got *bigger*, not
smaller, when checked at larger/real scale — the opposite of Candidate
G's +0.0027→+0.0019→flat pattern and the EB-NeRD contrastive result.
Working explanation (inference, not proven): a trainable-encoder model
has real capacity to benefit from MINDlarge's ~14x larger training set;
Candidate G's fixed-feature combiner structurally didn't.

### What's Next

**Open, for the engineer, not decided here:** whether to generate real
MINDlarge_test predictions and pursue a Codabench resubmission.

### Related

- `decisions/ADR-012-nrms-lite-candidate-j.md`'s 2026-08-25 addendum (full detail)
- `experiments/candidate_j_nrms_lite_ada_mindlarge_2026-08-25/{config,results}.json`
- `scripts/mind_nrms_lite_ada_large.sbatch`, `src/pipeline/download.py`

---

## August 22-24, 2026 — Candidate J (NRMS-Lite) Reopens the MIND Search With a Real Win (ADR-012)

### Context

Direct continuation from the Aug 22 closure below: the engineer arrived
with a script (`mind_nrms_lite_kaggle_run.py`) and an ADR template
already drafted from outside this repo, targeting Candidate J — a
lightweight, from-scratch NRMS reproduction (Wu et al. 2019) testing
whether H1/H2/I/I-long/I-pop's plateau was a missing-architecture problem
(no trainable title encoder, no click-history sequence model) rather than
a real ceiling. Per the engineer's own instructions, this session first
reviewed and integrated the script into the repo (it had no visibility
into `src/pipeline` when written), then, once the engineer separately
gained access to Ada (IIIT-H's SLURM HPC cluster), reconsidered every
design choice that had been explicitly traded down for Kaggle's time/
network constraints, then ran it for real.

### What was done

- **Integration review** found the script duplicated `src/pipeline`'s real
  MIND loader (standalone polars TSV parsing instead of
  `build_mind_split`) and hand-rolled AUC/MRR/nDCG/CI code instead of
  reusing `src/evaluation/ranking_metrics.py`. Rewired to both — this
  project's usual "one canonical implementation" discipline.
- **The model code review caught a real bug the numpy-only simulation
  couldn't:** a live forward+backward test (`tests/unit/test_nrms.py`)
  found that `torch.nan_to_num` on the forward output does NOT stop
  `nn.MultiheadAttention` from propagating NaN into shared projection
  weights during backward when a batch contains a fully-masked row
  (zero-history user, empty-title article) — `0 * NaN = NaN` under IEEE
  754. Fixed by pre-empting the NaN (guarantee ≥1 unmasked key before
  softmax, zero the result with a plain finite multiply) rather than
  mopping up after the fact. New `src/retrieval/nrms.py` (model classes)
  and `src/retrieval/nrms_training.py` (shared training/eval/GloVe
  helpers, reused by both the Kaggle and Ada scripts — avoids duplicating
  the training loop a second time).
- **Ada reconsideration:** three trade-offs explicitly justified by
  Kaggle's constraints, not architecture reasoning, were revisited once
  Ada removed them — GloVe pretrained embeddings (was: skipped, "no
  multi-GB download this close to the deadline"), `embed_dim`/`num_heads`
  (128/8 → 300/15, matching GloVe's dimensionality), and the fixed
  `NRMS_EPOCHS=3` default (replaced with real early stopping, patience=3,
  against a raised ceiling of 30 — safe against the I/I-long overfitting
  trap since both scripts checkpoint on REAL MINDsmall-dev every epoch).
  One premise corrected before proceeding, not assumed: Ada was described
  as having "no time limit" — the real user guide showed a genuine
  ~4-day wall-clock cap for the `research` account's `low`/`medium` QoS.
- **A real, multi-hour HPC environment saga**, each step diagnosed from
  real error output, not guessed: the login node's system Python (3.6)
  couldn't run this project's type hints; `module avail` errors out on
  this cluster (broken modulefiles); Miniconda's installer refused
  outright (`glibc >=2.28` required, node has 2.17); `uv` (a static
  binary + portable Python builds) worked where both failed;
  `uv pip install` crashed on the LOGIN node specifically (a real memory
  ceiling, not a bug) — fixed by running it inside an `srun` allocation
  instead; the default `pip`-resolved `torch` build required CUDA 13.0,
  too new for the node's actual driver (max CUDA 12.8) — fixed by
  installing from `download.pytorch.org/whl/cu121` specifically;
  `/ssd_scratch` turned out to be **local to each compute node, not
  shared across the cluster** (data staged via one interactive session
  was invisible to the batch job that landed on a different node) — fixed
  by making `mind_nrms_lite_ada_run.py` auto-download the MIND zips
  itself (`src/pipeline/download.py`, idempotent), the same pattern
  GloVe already used; and `downloads.cs.stanford.edu`'s GloVe zip proved
  to be persistently throttled from this cluster (~15-50KB/s, not a
  blip) — the first Ada run's GloVe download failed outright
  (`ContentTooShortError`, `urlretrieve` has no retry/resume), fixed at
  the root by shelling out to `wget -c --tries=10 --retry-connrefused`
  instead, which survived the same throttled connection on the second
  attempt (9h15m, but it completed).
- Real `sinfo`-verified partition name (`u22`, not the Ada wiki's stale
  `long` example) and a real `QOSMaxCpuPerUserLimit` block (an idle
  interactive session was still holding CPUs the queued batch job
  needed) both diagnosed from real SLURM error output along the way.

### Key Findings

- **First Ada run (job 2675355, no GloVe — download failed both in an
  interactive test and the batch job's first attempt):** 0.6242 (95% CI
  0.6220-0.6262), 236,344 real training examples, best at epoch 8/11
  (early-stopped). Statistically flat against Candidate I (0.6233) — a
  real, clean answer that the architecture change ALONE, without
  pretrained embeddings, does not resolve the plateau.
- **Second Ada run (job 2675573, GloVe succeeded — 96.8% vocab
  coverage):** **0.6391 (95% CI 0.6370-0.6412)**, best at epoch 3/6
  (early-stopped). **Real, CI-clear WIN against the deployed baseline**
  (0.6340, CI 0.6319-0.6361) — no CI overlap. This is the first time in
  this project's entire MIND candidate search (A through J) that a
  genuinely new architecture, not a combiner stacking already-deployed
  scorers (which is what Candidate G was), has produced a real win. Lands
  0.0009 below the literature band's low end (0.64) — within noise of
  clearing it, unlike every prior candidate.
- **The no-GloVe-vs-GloVe comparison is itself real, controlled evidence**
  that pretrained embeddings, not the architecture components alone, were
  the dominant missing lever — exactly what this document's own
  pre-registered priority-order guess anticipated before either run
  happened, now confirmed rather than assumed.
- **Honest caveat carried forward, not dropped now that the result is
  good:** both runs checkpoint directly on real MINDsmall-dev every
  epoch (not an internal train-side holdout, Candidate I's stricter
  discipline), which is mildly optimistic. The CI-clear margin is large
  enough that this is unlikely to explain the win away entirely, but it's
  a real, specific reason MINDlarge re-verification matters here, not
  just standing process.

### What's Next

**Open, for the engineer, not decided here:** whether to pursue MINDlarge
re-verification and a possible Codabench resubmission, per this project's
standing practice (ADR-010) for any real local win before deployment.
New/modified this session: `src/retrieval/nrms.py`, `nrms_training.py`,
`tests/unit/test_nrms.py`, `test_nrms_training.py`,
`scripts/mind_nrms_lite_kaggle_run.py`, `mind_nrms_lite_ada_run.py`,
`mind_nrms_lite_ada.sbatch`, `decisions/ADR-012-nrms-lite-candidate-j.md`
(canonical, full detail), `ADR-011`'s new addendum (cross-reference),
`experiments/candidate_j_nrms_lite_ada_2026-08-24/`, this file.

### Related

- `decisions/ADR-012-nrms-lite-candidate-j.md` (canonical — full design,
  environment history, and results)
- `decisions/ADR-011-neural-attention-reranker.md`'s 2026-08-24 addendum
- `experiments/candidate_j_nrms_lite_ada_2026-08-24/{config,results}.json`

---

## August 22, 2026 (latest) — Candidate I Closeout: More Epochs + Popularity Feature (ADR-011 Addendum)

### Context

Direct continuation of the same session, immediately after the Candidate I
checkpoint report (below). New objective (verbatim in
`knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`, prompt
4): one more bounded check before accepting the result — re-run with
20-30 epochs to see if the still-climbing holdout curve meant 5 epochs
wasn't enough, and if time allows, add a popularity feature to test
whether its absence explained the flat cold cohort. Explicit hard stop
after this round regardless of outcome — no further candidates, time
redirected to the design note/Q7 checklist.

Scoped tightly enough (explicit epoch range, one named feature, "same
code, no redesign") that a fresh Plan Mode cycle was judged unnecessary —
implemented directly as a bounded extension.

### What was done

- Extended `AttentionScorer` with an always-present `pop_weight`
  parameter and an optional `log_popularity` argument to `forward()` —
  backward-compatible (defaults to `None`, identical behavior to before
  when omitted). Zero-history users get a real popularity-only score
  instead of a flat 0 tie when popularity is enabled — the actual point of
  the feature, since popularity needs no history at all.
- Extended `run_attention_reranker_experiment.py`: `--use-popularity`
  flag, `build_train_popularity` reused unchanged (train-split-only,
  Laplace-smoothed, same construction as every other candidate), zero-
  history impressions no longer skipped during training when popularity
  is enabled (they now have a real gradient path via `pop_weight`). 3 new
  unit tests (196 total, all passing). A smoke test on a small subset
  caught no new bugs before committing to the real runs.
- Ran **I-long** (`--epochs 30`) and **I-pop** (`--epochs 30
  --use-popularity`) sequentially in the background (~53 min total: I-long
  ~24 min, I-pop ~46 min — slower per-epoch since popularity-enabled
  training no longer skips cold impressions, so more impressions are
  processed per epoch).

### Key Findings

- **I-long: a clean, real negative result.** Holdout AUC climbed every
  single epoch through 30 (0.6067 → 0.6372, never plateaued — by epoch 29
  it exceeds the deployed baseline's own 0.6340). Despite that, the real
  MINDsmall-dev result for the epoch-29 checkpoint (0.6214, paired -0.0126)
  was *worse* than the epoch-4 checkpoint's dev result from the original
  5-epoch run (0.6233, paired -0.0107) — confirmed as a real divergence,
  not run-to-run noise (both runs share an identical seeded trajectory
  through epoch 4, verified by matching holdout AUC to 6 decimal places).
  This directly answers "was 5 epochs enough": yes, and training longer
  made the real result marginally worse, not better — the internal
  train-holdout metric improving does not mean the real dev generalization
  gap is closing. This is the same qualitative pattern this project
  already found twice between local validation and real blind-test
  results (EB-NeRD's contrastive-vector submission, MIND's Candidate G
  leaderboard result) — now observed one level earlier, between an
  internal train-holdout and the officially held-out dev split.
- **I-pop: a dramatic but confounded result — not a clean test of the
  popularity hypothesis.** AUC collapsed to 0.5133 (paired -0.1207, 95% CI
  -0.1237 to -0.1178), worse than nearly every candidate in either search
  round. Investigated rather than taken at face value: holdout AUC was
  already below chance (0.4733) at epoch 0 (not a slow overfit), and the
  fitted `pop_weight` (-0.1047) has the **wrong sign** versus every other
  candidate's fitted popularity coefficient (F: +0.744; H1: +0.660, both
  standardized). Root cause identified: unlike F/H1/H2, which all
  `StandardScaler`-normalize every feature (including popularity) before
  fitting, this implementation fed *raw* `log_popularity` (spanning
  roughly -12.5 to -3 for MINDsmall's Laplace-smoothed values) directly
  into the model, unnormalized, alongside an attention term naturally
  bounded near [-1, 1] — a real, identifiable implementation gap, not
  evidence that popularity itself doesn't help. Documented explicitly as
  **inconclusive**, not as a second clean negative, so a future session
  doesn't mistake a confounded collapse for a settled scientific result.
- Per the engineer's explicit hard-stop instruction, **no fix-and-rerun
  was attempted** even though the root cause (and its fix — z-score
  `log_popularity` before combining) is well-identified and cheap.

### What's Next

**The MIND candidate-search line is closed for this project.** Nine
+ two candidates (A-H, G's real submission, I, I-long, I-pop) have now
been tried across two sessions; the only real win found anywhere (G)
didn't hold at real blind-test scale. Remaining engineering time goes to
the design note and Q7 deliverables checklist, per the engineer's explicit
instruction — not queued as further candidate-search work.
New/modified this session: `src/retrieval/rerank.py` (popularity
extension), `scripts/run_attention_reranker_experiment.py` (popularity
extension), `tests/unit/test_rerank.py` (+3 tests),
`experiments/candidate_i_attention_reranker_mind_small_2026-08-22_{e30,e30_pop}/{config,results}.json`,
`decisions/ADR-011-neural-attention-reranker.md` (addendum),
`knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md` (prompt 4
appended), this file.

---

## August 22, 2026 — Neural Attention Re-ranker Checkpoint: Candidate I (ADR-011)

### Context

Direct continuation of the same session, immediately after Candidate H
(below). New objective (verbatim in `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`,
appended as prompt 3): build a lightweight neural re-ranker with
candidate-aware attention pooling over history embeddings (the core
NRMS/NAML mechanism, minus fine-tuning the text encoder — frozen MiniLM
embeddings as input, only the attention + scoring head trained), on
MINDsmall-train's real labels, with a hard instruction to checkpoint at
MINDsmall-dev and report back — "if it doesn't show a real, meaningfully-
larger win than what H1/H2 got... stop and report back" — rather than
proceeding to MINDlarge automatically.

Planned via a fresh Plan Mode cycle
(`/Users/test01/.claude/plans/rosy-twirling-ripple.md`, overwritten from
the Candidate H plan). Research confirmed this is the first neural-network
*training* this project has done — `torch` was previously used only for
frozen `sentence-transformers` inference.

### What was done

- New `src/retrieval/rerank.py`: `build_user_history_vectors` (the raw,
  unpooled per-article history embedding sequence — no equivalent existed
  before), `AttentionScorer` (single-head scaled dot-product attention,
  candidate-conditioned, per-impression with no padding/masking — a
  deliberate correctness-over-throughput choice for a first neural
  training), `AttentionRerankScorer` (the `Scorer`-protocol eval wrapper).
- New `scripts/run_attention_reranker_experiment.py`: a lean data-side
  builder (embeddings only, deliberately skips BM25/symbolic features
  Candidate H used, per the objective's "frozen embeddings as input"
  scope), BCE training with a data-derived `pos_weight` (23.72, from the
  real 4.04% click rate), Adam (`lr=1e-3`), 64-impression gradient
  accumulation (no padded batching), a 95/5 user-level train-internal
  holdout for epoch-level early stopping (MINDsmall-dev touched exactly
  once, at the end).
- 10 new unit tests (`tests/unit/test_rerank.py`); full suite (193 tests)
  reconfirmed passing.
- **Required benchmark step** (per CLAUDE.md's benchmarking philosophy)
  run before the real job: a 3,000-impression/1-epoch subset run crashed
  with `RuntimeError: element 0 of tensors does not require grad and does
  not have a grad_fn`. Root cause: a zero-history user's score is a
  constant `torch.zeros(...)` under this architecture (the project's
  standard cold-start convention) — no gradient path to the learned
  weights at all, so `backward()` had nothing to differentiate whenever a
  minibatch included one. Fixed by skipping zero-history impressions
  during *training* only (not evaluation, where they're still scored
  normally as a 0/tie) — a correctness fix following directly from the
  architecture, not a workaround, since those rows carry zero gradient
  information for this model either way. Re-benchmarked clean: 1,505
  training impressions/sec, projecting the full 5-epoch job at under 11
  minutes — confirmed before committing to the real run, not assumed.
- Ran the real job (149,107 fit impressions, 5 epochs, ~305s train + 176s
  final dev eval). Holdout AUC climbed every single epoch (0.6067 →
  0.6096 → 0.6137 → 0.6181 → 0.6205) — **not plateaued at the 5-epoch cap**,
  flagged explicitly as an open question in ADR-011 rather than smoothed
  over.

### Key Findings

- **Real, CI-clear loss at the checkpoint — does not clear the "substantial
  win" bar.** Overall AUC 0.6233 (95% CI 0.6212-0.6254), paired vs.
  baseline **-0.0107** (95% CI -0.0126 to -0.0088) — worse than H1's
  -0.0021, smaller than H2's -0.0355. Warm cohort also lost (0.6316 vs.
  baseline's 0.6439); cold cohort came back essentially flat (0.5727 vs.
  baseline's 0.5737) — unlike F/H1's real cold wins, plausibly because
  Candidate I has no `log_popularity` input at all (the single dominant
  fitted feature behind those wins) — inference, documented as such in
  ADR-011, not independently re-verified this session.
- **The still-climbing holdout curve is the single most important
  qualifier on this result.** Because the model had not converged within
  the 5-epoch/report-within-a-day budget, this number is best read as "not
  a substantial win *at this training budget*," not proof the mechanism
  itself is a dead end — an explicit, real distinction, not equivocation
  for its own sake.
- Per the engineer's own explicit instruction, the session stopped here:
  no MINDlarge run, no further epoch tuning, no MLP-head/multi-head
  variant — those are documented as concrete next levers in ADR-011's
  Conditions for Revisiting, not undertaken.

### What's Next

**Reporting back to the engineer at this checkpoint, as instructed.**
Three model classes (linear, tree ensemble, attention-over-raw-embeddings)
have now all been tried on MINDsmall-train/dev and all lost against the
deployed baseline — the honest, currently-complete picture of this
project's MIND candidate search. New/modified this session:
`src/retrieval/rerank.py`, `tests/unit/test_rerank.py`,
`scripts/run_attention_reranker_experiment.py`,
`experiments/candidate_i_attention_reranker_mind_small_2026-08-22/{config,results}.json`,
`decisions/ADR-011-neural-attention-reranker.md` (new),
`decisions/ADR-010-mind-second-submission-candidate-search.md` (forward
pointer), `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`
(prompt 3 appended), this file.

---

## August 22, 2026 — GBDT Supervised-Ranker Screen: Candidate H (ADR-010 Addendum)

### Context

This session opened with an objective claiming the project had "never
trained an actual supervised model on MIND's real click labels" and asking
for one to be built, framed as the likely explanation for classmates
reportedly scoring 0.65-0.70 on the same assignment. Before writing any
code, that premise was checked against the project's own record and found
false: Candidate F (`scripts/run_learned_combiner_experiment.py`,
ADR-010, 2026-08-21) already trained a `LogisticRegression` combiner on
MINDsmall-train's real click labels using the same feature families named
in the new objective (BM25, embedding cosine, category/entity match,
recency, plus popularity) — and it lost overall (AUC 0.6255 vs. baseline
0.6340, CI-clear). This was surfaced to the engineer via a written plan
(`/Users/test01/.claude/plans/rosy-twirling-ripple.md`) before implementation,
per CLAUDE.md's Decision Reversal/Evidence Hierarchy sections, rather than
silently re-running a known negative result under a new name. The
approved plan scoped a genuinely different test: a nonlinear model plus
one new feature, isolated from a class-imbalance-correction variant of the
same linear model, so any result would be attributable to a specific
cause. Full verbatim prompts:
`knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`.

### What was done

- New `scripts/run_gbdt_combiner_experiment.py` (Candidate H), reusing
  Candidate F's index/query building and per-candidate feature computation
  unchanged (`_build_side`, `_impression_feature_matrix`, `FEATURE_NAMES`)
  and the paired-bootstrap comparison Candidate G established
  (`paired_metric_diff_ci`) — no shared code duplicated.
- One new feature: `log_history_length = log1p(n_articles)`, from
  `HistoryProfile.n_articles` (`src/retrieval/features.py`) — previously
  only used externally as ADR-005's hard cohort-routing threshold, now
  given to the model directly.
- Two models, both trained on MINDsmall-train (identical data to F:
  5,843,444 rows / 156,965 impressions, 4.04% click rate), evaluated on
  MINDsmall-dev: **H1** — `LogisticRegression(class_weight="balanced")`,
  isolating the imbalance-correction hypothesis from F's untuned defaults.
  **H2** — `sklearn.ensemble.HistGradientBoostingClassifier(early_stopping=True)`,
  isolating the nonlinear-interaction hypothesis. Chosen over LightGBM
  (the engineer's literal suggestion) per an explicit engineer decision —
  same histogram-boosting family, zero new dependency, no macOS OpenMP
  install risk.
- 4 new unit tests (`tests/unit/test_gbdt_combiner.py`); full existing
  suite (183 tests) reconfirmed passing after the addition.

### Key Findings

- **Both models are real, CI-clear losses — no MINDlarge scale-up was
  warranted** (the session's own objective made scaling conditional on a
  real win). H1: overall AUC 0.6319 (95% CI 0.6297-0.6340), paired vs.
  baseline -0.0021 (95% CI -0.0042 to -0.0002) — closer to baseline than F
  but still a statistically significant loss. H2: overall AUC 0.5985 (95%
  CI 0.5961-0.6006), paired vs. baseline -0.0355 (95% CI -0.0381 to
  -0.0331) — a much larger loss, worse than every candidate tried in this
  search except entity embeddings (A) and popularity-only (E).
- **H2's loss is concentrated in the warm cohort** (0.6004 vs. baseline
  warm's 0.6439) while its cold-cohort number (0.5871) stays roughly
  comparable to H1/F's cold wins — i.e. the nonlinear model is actively
  worse than the baseline specifically where BM25/embedding signal is
  already strong, not just "no better than baseline everywhere." Most
  plausible explanation (inference, flagged as unverified in ADR-010's
  addendum): `n_iter_` = 100 exactly equals the library default
  `max_iter`, meaning early stopping's own internal validation slice
  (drawn from MINDsmall-**train**) never detected a plateau — but that
  slice shares train's own self-information structure between
  `log_popularity` and the labels being predicted, so a flexible tree
  ensemble may be overfitting exactly the pattern its own validation check
  can't see, in a way F's 7-coefficient linear model structurally
  couldn't.
- **Combined with the same-day real-leaderboard result below: two model
  classes (linear and nonlinear) have now both been tried over this
  project's current feature set and neither beats the deployed baseline.**
  Real evidence the ceiling is the feature set (precomputed similarity
  scores), not the combiner's model class — closing the gap toward
  classmates' reported 0.65-0.70 would most plausibly need new information
  (raw text) or an end-to-end trained model, materially larger in scope
  than this session's "lightweight ranker, move fast" framing.
- H1's cold-cohort AUC (0.5936) is marginally better than F's (0.5926,
  currently used by Candidate G's cold-user routing) — a cheap, flagged-
  but-not-undertaken follow-up if this is revisited.

### What's Next

No further action recommended on the MIND supervised-combiner line unless
the feature-set-is-the-ceiling hypothesis above is specifically
challenged (e.g. by adding raw-text features or moving to an end-to-end
model) — documented as an open lever in ADR-010's addendum, not queued as
active work. New/modified this session: `scripts/run_gbdt_combiner_experiment.py`,
`tests/unit/test_gbdt_combiner.py`,
`experiments/candidate_h_gbdt_combiner_mind_small_2026-08-22/{config,results}.json`,
`decisions/ADR-010-mind-second-submission-candidate-search.md` (addendum),
`knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`, this file.

---

## August 22, 2026 — Real MIND Result: Essentially Flat, Not the Local Win (ADR-010 Addendum)

### Context

Objective (verbatim in `knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`):
Candidate G's real Codabench result is in — 0.6192, essentially flat vs.
886468's 0.6195, despite a CI-clear +0.0019 local win — document this
honestly as a validation-to-test transfer finding, not a win, and update
every place the earlier "ready, not yet uploaded" status was recorded.

### What was done

- Wrote a new ADR-010 addendum recording the real result plainly: three
  stages of local validation (MINDsmall-dev +0.0027, MINDlarge-dev
  +0.0019, both CI-clear) followed by a real leaderboard score of 0.6192
  vs. 886468's 0.6195 (-0.0003) — not the win either local stage
  predicted.
- Before writing the "why," checked this project's own records for the
  parallel the engineer referenced (EB-NeRD's contrastive-vector result)
  rather than assuming or inventing numbers — initially found only a
  "prepared, not yet run" status on record and asked the engineer to
  clarify rather than guess; the engineer answered by directly recording
  the real result in ADR-008 (submission 896072: local +0.0023 CI-clear
  win compressed to +0.0005 real mean AUC, day-to-day sign flipping three
  times across 8 dates) — read in full before drawing the parallel.
- Grounded the "population difference" explanation in a real, cited fact
  rather than speculation: MIND's official dev/test splits are
  temporally disjoint, non-overlapping calendar weeks by design (ADR-001;
  MINDlarge's test week is Nov 16-22), not a resampling of the same
  population — checked directly in ADR-001 rather than assumed.
- Updated `docs/design_note.md` **and** `docs/design_note.tex` (both kept
  in sync, not just the Markdown source) — §3.5's leaderboard table gains
  the second MIND row and an honest discussion paragraph mirroring the
  EB-NeRD one already there; §6 gains a new bullet naming the cross-
  dataset pattern (local CI-clear wins compressing at real blind-test
  scale, now N=2). Rebuilt `docs/design_note.pdf` via `tectonic` so the
  PDF deliverable isn't left stale relative to the source.
- Updated PROJECT_STATE.md's Leaderboard Submission row and top summary
  to record both real MIND submissions and retire the earlier "ready, not
  yet uploaded" framing now that the real outcome is known.
- Noticed, but did not touch, unrelated untracked files
  (`scripts/run_gbdt_combiner_experiment.py`,
  `knowledge/ai-usage-log/2026-08-22_gbdt-combiner-candidate-h.md`) that
  appear to be from separate, concurrent work outside this session's scope
  — flagged rather than silently assumed to be this session's own or
  interfered with.

### Key Findings

- **The real result is an honest negative finding, not a bug.** Every
  format/correctness check this project's own discipline requires passed
  cleanly; the gap is real, not a symptom of a mistake.
- **This is now a two-for-two pattern, not an isolated surprise.** Both
  real-test-set checks this project has ever run (EB-NeRD contrastive
  vector, MIND cohort-gated combiner) showed a local CI-clear win compress
  substantially at real blind-test scale — worth treating as a genuine
  property of this project's validation methodology (small local edges
  built on population-specific structure) rather than two unrelated
  coincidences, though `N=2` is too small to fit any quantitative
  relationship between local and real effect sizes.

### What's Next

Both real MIND submissions and the cross-dataset compression pattern are
now fully documented (ADR-010, design note §3.5/§6, PROJECT_STATE). No
further MIND submission work is planned from this thread. If a third
real-test-set comparison is ever made (either dataset), it would
meaningfully strengthen or weaken the two-for-two pattern as a general
finding, per ADR-010's own Conditions for Revisiting.

## August 21-22, 2026 — MINDlarge Verification + Second-Submission Build (ADR-010 Addendum)

### Context

Objective (verbatim in `knowledge/ai-usage-log/`): Candidate G verified at
MINDlarge scale on Kaggle with a real win — build the actual second MIND
Codabench submission using it, verify format like every prior real
submission, and stop short of uploading (engineer does that manually).

### What was done

- Got the Kaggle notebook working end-to-end through a long troubleshooting
  cycle: Kaggle auto-extracting the uploaded zip into a Dataset folder
  instead of keeping it as a `.zip` (Cell 1 fixed to detect either), a
  stale/corrupt partial download silently reused because
  `download_mind_bundle` skips if the file already exists (fixed by
  re-downloading fresh), and two real cross-platform bugs surfaced by
  actually running on Kaggle's infrastructure (not guessed in advance):
  `read_zip_member_bytes` had no fallback for a flat-packed zip (the HF
  MIND mirror packs `MINDlarge_train.zip` flat, opposite of every other
  MIND zip this project had seen), and `load_encoder`/
  `build_embedding_index` hardcoded Mac-only `device="mps"` (Kaggle's
  Linux runners don't have it). Both fixed at the root in `src/`, not
  worked around in the notebook, with new unit tests (6 total).
- Engineer confirmed the real Kaggle result: Candidate G's exact
  MINDsmall-fitted model, evaluated at MINDlarge-dev scale, shows a real
  paired-bootstrap win — AUC +0.0019 (95% CI +0.0015 to +0.0023),
  excludes zero. Same direction as the MINDsmall-dev screen, smaller
  effect size. Full per-metric Kaggle output not yet filed locally
  (pending the engineer's downloaded results package) — the headline
  number is confirmed and sufficient to proceed with the submission build,
  per ADR-010's addendum.
- Built `scripts/generate_mind_gated_predictions.py`: a `GatedScorer`
  implementing the existing `Scorer` protocol so
  `src/submission/mind_format.py::write_predictions` (the same module
  every prior real submission used) needed zero changes. Confirmed local
  execution was feasible before running anything real: this exact scale
  (2,370,727 MINDlarge_test impressions) had already completed locally
  once before for the embed-only baseline (~1.9hr, `submissions/
  mind_large_test_embed/run.log`), and the processed bundle + embedding
  cache already existed on disk — no Kaggle needed for this part.
- A deliberate small-scale smoke test against MINDsmall-dev (run *before*
  committing to the real ~4hr job, not skipped under time pressure)
  caught a real bug: `RuntimeWarning: invalid value encountered in
  matmul`. Root cause: MIND's raw `behaviors.tsv` can reference a
  candidate absent from that split's own corpus (the documented `N89741`
  quirk), which every index-backed feature scores `-inf`; combining three
  already-`-inf` features through Candidate F's *mixed-sign* fitted
  coefficients produces `-inf + +inf = NaN`, not just `-inf` — a failure
  mode no single-feature scorer in this project could ever hit. Fixed by
  explicitly detecting `np.isneginf` on the raw inputs and forcing the
  combined score to `-inf` directly. 4 new unit tests.
- Ran the real job locally as a detached background process with active
  memory monitoring (a preemptive kill-switch, never triggered — peak RSS
  stayed under 500MB throughout). Took 15,084s (~4.19hr), longer than the
  ~1.9hr baseline it was projected from — attributed to this session's
  chronic background memory pressure throttling CPU-bound work generally
  (the process itself never showed memory problems), not confirmed in
  isolation. Survived an accidental laptop-lid-close/sleep mid-run with no
  data loss (macOS suspends and resumes background processes cleanly).
- Verified with the same discipline as every prior real submission: exact
  line count (2,370,727), zero malformed rank permutations checked across
  every line, all 32 `N89741`-affected impressions individually
  spot-checked (not just trusted from the earlier unit test), byte-
  identical file size to the original embed-only submission's
  `prediction.txt` (expected — same structure, only rank values differ).
  Packaged identically (`prediction.txt` zipped at archive root via
  `zip -j`, confirmed via `unzip -l` against the original).

### Key Findings

- **The real win holds up at real scale** — smaller effect size at
  MINDlarge (+0.0019) than the MINDsmall-dev screen (+0.0027), but the
  same direction and both paired-bootstrap CIs exclude zero. This is
  exactly the kind of scale-transfer check ADR-010 flagged as unverified
  and worth doing before trusting a smaller-scale result for a real
  submission — and it transferred, not just assumed to.
- **Every real bug this session hit was invisible from local-only,
  single-method-scorer testing** — a zip-packaging convention only the
  real HF mirror exhibited, a device default only Linux exposed, a memory
  cost only real MINDlarge row counts made material, and a NaN failure
  mode only a multi-feature linear combiner with mixed-sign coefficients
  could produce. None were hypothetical edge cases invented for
  thoroughness — all four were caught by actually running the real thing
  (Kaggle infrastructure, or a deliberate smoke test before the real job),
  consistent with CLAUDE.md's benchmarking philosophy of trusting
  measurement over assumption.

### What's Next

**Ready for the engineer's manual Codabench upload** —
`submissions/mind_large_test_gated_cohort/prediction.zip`, format-verified.
Not uploaded by this session; that action, and reporting the resulting
leaderboard score, is explicitly left to the engineer. Follow-up
documentation task (not blocking): file the full Kaggle
config.json/results.json at
`experiments/candidate_g_gated_cohort_mind_large_2026-08-21/` once
downloaded, and update ADR-010's addendum's citation from the confirmed
headline number to that file. New/modified this session:
`scripts/generate_mind_gated_predictions.py`,
`tests/unit/test_gated_predictions.py`, `src/utils/io.py` (+tests),
`src/retrieval/embed.py` (+tests),
`notebooks/mind_gated_cohort_mindlarge_{kaggle_run.py,src_bundle.zip}`,
`submissions/mind_large_test_gated_cohort/`,
`decisions/ADR-010-mind-second-submission-candidate-search.md` (addendum),
this file.

## August 21, 2026 (round 2) — MIND Candidate Search Round 2: Symbolic Signals, Learned Combiner, Cohort Gating (ADR-010)

### Context

Round 1 (below) found no win among three variations on the deployed
method's own mechanisms. This round's objective (verbatim in
`knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`):
explore genuinely different approaches, not just re-tuning A/B/C — latitude
given explicitly, with a few starting directions (symbolic overlap, a
learned combiner) but "use your judgment" for anything else.

### What was done

- **Candidate D — symbolic overlap:** new `src/retrieval/features.py`
  (`HistoryProfile`, `build_history_profile`, `category_match_score`/
  `subcategory_match_score`/`entity_overlap_count`), 10 new unit tests.
  Untuned score = category match + subcategory match + log1p(entity
  overlap). Result: AUC 0.6134 (CI 0.6112-0.6154) — CI-clear loss, but the
  second-best standalone signal of either round after the deployed
  baseline itself (better than BM25 alone, A, B, or E).
- **Candidate E — train-popularity only:** reused
  `ranking_metrics.py::build_train_popularity` (already built for Q9's
  novelty metric) directly as a scorer — zero personalization, a pure item
  prior. Result: AUC 0.5318 (CI 0.5299-0.5335) — CI-clear loss, weakest
  standalone signal of either round except entity embeddings.
- **Candidate F — learned combiner:** new
  `scripts/run_learned_combiner_experiment.py`. Seven features per
  (impression, candidate): `bm25`, `embed_cos`, `recency_embed_cos`
  (Candidate C's mechanism, now as an additional feature, not a
  replacement), `category_match`, `subcategory_match`,
  `entity_overlap_log1p`, `log_popularity`. `StandardScaler` +
  untuned-default `LogisticRegression`, fit on MINDsmall-train's real click
  labels (5,843,444 candidate rows across 156,965 impressions, 4.04% click
  rate), evaluated on MINDsmall-dev. Fitted coefficients recorded in full
  in `config.json` for reproducibility (largest: `log_popularity` +0.744,
  `embed_cos` +0.278; smallest/negative: `recency_embed_cos` -0.014,
  consistent with Candidate C's own standalone loss). Result: overall AUC
  0.6255 (CI 0.6234-0.6276) — CI-clear loss — **but cold-cohort AUC 0.5926
  (CI 0.5869-0.5985) vs. baseline cold's 0.5737 (CI 0.5682-0.5792) is a
  CI-clear win, the first either round produced anywhere.**
- **Candidate G — cohort-gated scorer:** new
  `scripts/run_gated_cohort_experiment.py`, built directly from F's own
  cohort split — the deployed `EmbeddingScorer` unchanged for warm users,
  F's already-fitted combiner (reused, not retrained) for cold users.
  Overall AUC 0.6366. Its marginal 95% CI (0.6346-0.6388) overlaps the
  baseline's (0.6319-0.6361) by a sliver — by the same test used for every
  other candidate this would read "not CI-clear." **Caught this before
  reporting it as ambiguous:** G shares almost all its impressions with
  the baseline by construction (identical scores for every warm
  impression), so comparing independent marginal CIs discards that shared
  structure and is the wrong test. Added `paired_metric_diff_ci`
  (`run_gated_cohort_experiment.py`, reusing `bootstrap.py::bootstrap_ci`
  unchanged with a custom paired stat function) — resamples users once per
  replicate, computes gated-minus-baseline AUC on the *same* resampled
  users. Result: **+0.0027 (95% CI +0.0018 to +0.0034), entirely excluding
  zero** — a real, correctly-tested win. 3 new unit tests
  (`test_gated_cohort.py`) confirm the paired-diff function is zero for
  identical columns and correctly detects a known constant shift.
- Full unit suite (169 tests) and integration suite (44 passed, 1
  pre-existing skip) both reconfirmed green after every addition.
- Wrote ADR-010 (new, not an addendum — this round's decision space didn't
  fit naturally under ADR-005 or ADR-008) covering all four round-2
  candidates plus the design space/rationale/evidence/conditions-for-
  revisiting, per this project's ADR template.

### Key Findings

- **A real, statistically rigorous win was found this round: Candidate G,
  overall AUC 0.6366 vs. baseline 0.6340 (+0.0027, paired 95% CI
  +0.0018 to +0.0034).** This is the only win, of seven candidates tried
  across both rounds, that clears the objective's own bar ("commit only
  whichever shows a real, CI-clear win").
- **The marginal-CI-overlap heuristic used throughout both rounds nearly
  produced a false negative for the one candidate that actually worked** —
  worth flagging as a general methodology note, not just a Candidate-G
  footnote: that heuristic assumes independence between what's being
  compared, which holds for A-F (each an independently-scored alternative
  method) but not for G (a routing decision sharing most of its scores
  with the baseline by construction). Checking whether that independence
  assumption actually holds, rather than applying one test uniformly, is
  what surfaced the real result here.
- A learned combiner's *cold-cohort* win (F) being real even though its
  *overall* number loses is itself a genuine, explainable finding — cold
  users structurally lack the history BM25/embeddings need, and
  `log_popularity` (which needs no history at all) is the model's
  by-far-dominant learned feature, discovered from data, not asserted in
  advance.

### What's Next

**Not yet submitted to Codabench.** ADR-010's Final Decision names one open
question the engineer needs to decide before an actual submission: this
result is validated on MINDsmall-dev only, and this project's own past
practice (every real MIND leaderboard submission, and ADR-008's own
re-verification) has not trusted MINDsmall-only evidence for a real
submission before — whether to re-verify Candidate G at MINDlarge scale
first, or accept MINDsmall-dev evidence as sufficient given time
constraints, is presented as a decision point, not resolved automatically.
New/modified this session (round 2): `src/retrieval/features.py`,
`scripts/run_{symbolic_overlap,popularity,learned_combiner,gated_cohort}_experiment.py`,
`tests/unit/{test_features.py,test_gated_cohort.py}`,
`experiments/candidate_{d,e,f,g}_*_2026-08-21/`,
`decisions/ADR-010-mind-second-submission-candidate-search.md`, this file,
`knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`.

## August 21, 2026 (round 1) — Three MIND Candidate-Improvement Screens (ADR-005 + ADR-008 Addenda)

### Context

Objective (given verbatim in `knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`):
before attempting a real second MIND Codabench submission, cheaply test
three candidate improvements on MINDsmall-dev (no Kaggle) against the
deployed embedding baseline (AUC 0.634), and commit only whichever shows a
real, CI-clear win.

### What was done

- Checked Codabench competition 13967's real submission limit via its own
  API (`https://www.codabench.org/api/competitions/13967/`) rather than
  assuming: `max_submissions_per_day: 10`, `max_submissions_per_person: 999`,
  identical for both the Development and Official Test phases — generous,
  confirming submission quota was never the binding constraint here.
- Reconfirmed the baseline before testing anything against it: reran
  `run_ranking_eval.py --dataset mind --method embed` and got a
  bit-identical AUC (0.63399337223849) to the 2026-08-10 recorded run —
  the deployed pipeline is deterministic, as CLAUDE.md requires.
- **Candidate A — entity embeddings:** new `src/retrieval/entities.py`
  (`parse_entity_mentions`, `load_entity_vectors`,
  `build_article_entity_vector` — confidence-weighted pooling,
  `build_entity_index`), reusing `EmbeddingIndex`/`EmbeddingScorer`/
  `build_user_embedding_query` from `embed.py`/`score.py` unchanged. Real
  article coverage measured first (86.1% of MINDsmall-dev's 42,416
  articles), per the objective's own ordering, before evaluating. 13 new
  unit tests (`tests/unit/test_entities.py`). Result: AUC 0.5525 (CI
  0.5503-0.5546) — CI-clear loss vs. baseline.
- **Candidate B — BM25+embedding hybrid:** new
  `scripts/run_hybrid_experiment.py`, untuned 50/50 blend of min-max-
  normalized BM25 and embedding scores per impression, same design as
  ADR-009's leaky-feature ablation (`_minmax` reused directly). Result:
  hybrid AUC 0.6263 (CI 0.6242-0.6284) — CI-clear loss vs. embed-only
  baseline; BM25 alone 0.5692, confirming the hybrid dilutes rather than
  complements.
- **Candidate C — recency-weighted history:** new
  `build_user_embedding_query_recency` in `embed.py` (exponential decay by
  position, decay=0.9 untuned, computed over the *resolvable* sequence so
  gaps don't shift surrounding weights), applied to MIND's embedding-based
  user representation specifically (the deployed method), not BM25 (ADR-005
  never tested this for BM25 and still doesn't). 6 new unit tests. Result:
  AUC 0.6265 (CI 0.6243-0.6286) — CI-clear loss, including for warm users.
- Full unit suite (169 tests, up from 156) passes; integration suite
  reconfirmed unaffected (unmodified production code paths — all new logic
  lives in new modules/scripts).
- Documented both real (not merely absence-of-evidence) negative results as
  addenda to the two ADRs they actually bear on: ADR-008 (semantic
  retrieval design — Candidates A/B) and ADR-005 (query/user-representation
  construction — Candidate C's recency-weighting re-test), rather than as
  new standalone ADRs, since neither result changes either ADR's original
  decision.

### Key Findings

- **All three candidates are real, CI-clear losses, not noise or ties** —
  every candidate's 95% CI sits entirely below the baseline's, at overall
  AUC. This is a genuine, informative negative result, not an absence of
  result: it directly answers whether the deployed MiniLM-embedding
  baseline can be cheaply beaten, and the answer (for these three specific,
  untuned implementations) is no.
- Entity embeddings' coverage (86.1%) was high enough that low coverage
  doesn't explain the loss — the more likely explanation is that a
  knowledge-graph (TransE) embedding, trained for entity-relation
  structure, targets a different notion of similarity than MiniLM's
  text-topical one, and discards all non-entity text content besides.
- The hybrid's loss shows an untuned equal-weight blend isn't automatically
  safe even when one input (BM25) is strictly weaker — it can still drag
  the stronger input down rather than leaving it untouched.
- Recency weighting hurt warm users specifically, the cohort where the
  "recent clicks predict better" intuition predicted the *most* benefit —
  a real finding against that intuition for MIND, not just a non-result.

### What's Next

No second MIND Codabench submission from this session's candidates, per
the objective's own decision rule. If any candidate is revisited, each
addendum names a concrete next variant (asymmetric tuned hybrid weight,
entity vectors as an addition to MiniLM rather than a replacement, a decay
sweep for recency) — none of which were in scope for this cheap first
screen. New/modified this session: `src/retrieval/entities.py`,
`src/retrieval/embed.py` (`build_user_embedding_query_recency`),
`scripts/run_{entity_embedding,hybrid,recency_history}_experiment.py`,
`tests/unit/{test_entities.py,test_embed.py}`,
`experiments/candidate_{a,b,c}_*_2026-08-21/`,
`decisions/ADR-{005,008}-*.md`, this file,
`knowledge/ai-usage-log/2026-08-21_mind-candidate-improvements-local-validation.md`.

## August 19, 2026 — EB-NeRD Contrastive-Vector vs. MiniLM (ADR-008 Addendum)

### Context

Objective: measure the accuracy cost of ADR-008's original rejection of
EB-NeRD's provided embedding artifacts — argued at the time, never
measured. Isolated experiment only: `src/retrieval/embed.py`/`score.py`/
`retrieve.py`, `scripts/run_embed_experiment.py`/`run_ranking_eval.py`, and
ADR-008's own decision were explicitly out of scope and were not touched.

### What was done

- Read ADR-008 and the exact existing user-representation/scoring/harness
  code before writing anything, per this project's own workflow.
- Attempted to download `Ekstra_Bladet_contrastive_vector.zip` (341MB)
  directly in-session: real, measured outcome was ~4-170KB/s throughput
  across three attempts, one ending in a mid-transfer connection reset
  after several hours, never completing (confirmed via `unzip -l` failing
  on the truncated file, not just "slow"). Per CLAUDE.md's Resource
  Availability clause, this was surfaced explicitly (not silently retried
  indefinitely or substituted); the engineer chose to move the download to
  Kaggle.
- Confirmed the artifact's real format via `ebnerd-benchmark`'s own
  reproducibility scripts (official documentation) before writing any
  loading code, rather than guessing from the filename: a single
  `Ekstra_Bladet_contrastive_vector/contrastive_vector.parquet`, article-id
  column + vector as the last column. No public documentation of the
  training methodology was found anywhere, including inside the real
  archive once inspected — reported as a real gap, not filled with
  inference.
- Wrote `scripts/run_contrastive_vector_experiment.py` — one new function
  (`load_contrastive_index`), everything else (`build_user_embedding_query`,
  `EmbeddingScorer`, `embed_retrieve_top_k`, `recall_at_k`, every Q4 ranking
  metric + bootstrap CI) imported and reused unchanged.
- **Local smoke test before shipping to Kaggle caught a real bug**: a
  synthetic 90%-coverage artifact run through the actual `run()` end-to-end
  surfaced that `score.py`'s unmodified `_lookup_scores` scores an
  uncovered candidate as `-inf` by design, which crashes
  `sklearn.roc_auc_score` — a path MiniLM's 100%-by-construction coverage
  never exercised. Fixed locally (`_finite_scores_for_auc`, new script
  only, not a change to `score.py`'s contract); re-ran the same smoke test
  and confirmed random vectors correctly produce AUC ≈ 0.50 (no signal),
  validating the harness plumbing itself before trusting real numbers.
- Prepared `notebooks/ebnerd_contrastive_vector_kaggle_run.py` +
  `ebnerd_contrastive_vector_src_bundle.zip` (the project's own unmodified
  `src/pipeline`/`retrieval`/`evaluation` code, bundled) — Kaggle rebuilds
  `ebnerd_small` via this project's own `build_ebnerd_bundle`, guaranteeing
  byte-identical schema to what produced the MiniLM baseline, before
  running the comparison. Mirrors the project's existing Part 0/Part 2
  EB-NeRD Codabench Kaggle-relay pattern.
- Engineer ran the notebook and relayed back
  `experiments/contrastive_vector_ebnerd_small_2026-08-18/{config,results}.json`.
  **Verified directly** (both files parsed, every number checked
  digit-for-digit against the engineer's summary) before writing anything
  downstream — real coverage came back 100% (20,738/20,738 local articles,
  768-dim vectors, artifact's full catalog 125,541 articles), confirming
  the `-inf` fallback path never actually fired on the real data (it would
  have crashed without the earlier fix, so the fix mattered even though it
  wasn't exercised this run).

### Result (see ADR-008's Addendum for full detail)

On `ebnerd_small` validation: contrastive vector wins recall@50/100/200
(0.58%/1.20%/2.47% vs. MiniLM 0.14%/0.43%/1.21%, all CI-clear) and every Q4
accuracy metric (AUC 0.5453 vs. 0.5430 — CI-clear but narrow; MRR/nDCG@5/
nDCG@10 all clearly higher). Diversity@10 is a CI-clear **loss** for the
contrastive vector (0.780 vs. 0.789). Novelty@10 is a **tie**
(`ci_clear_win: null` — MiniLM's point estimate falls inside the
contrastive vector's own CI). Reported as a real trade-off, not a blanket
"the provided artifact is better" — ADR-008's own decision (compute one
model over both datasets) is not reversed, since the artifact doesn't
cover MIND.

### Updated

- `decisions/ADR-008-semantic-retrieval-design.md` — new Addendum section
  (full results table, interpretation, revisit conditions), history
  preserved per CLAUDE.md's decision-reversal guidance, not overwritten.
- `docs/design_note.md` §2 and §6, and `docs/design_note.tex` (recompiled
  to `docs/design_note.pdf`, page-count re-verified at exactly 4 pages via
  `pypdf` — the addition initially pushed it to 5 real pages, caught by
  checking rather than assuming, then brought back under budget by
  trimming both additions and fixing three pre-existing/newly-exposed
  LaTeX overfull-hbox warnings with `sloppypar`, not by silently shrinking
  margins alone without checking for clean output).
- This file (Component Status, Recent Decisions, Next Actions, this entry).

### Genuinely still open

- Whether to submit a second, EB-NeRD-only Codabench leaderboard entry
  using the provided artifact — a real decision for the engineer, not
  resolved by this session.
- The Diversity@10 loss's cause (hypothesis: sharper same/different
  discrimination trading off intra-list variety) was not isolated by a
  controlled follow-up.
- This result is `ebnerd_small`-validation-only; generalization to
  `ebnerd_large` or the real held-out test set is unverified.
- The Aug 14 PROJECT_STATE entry below was not reconciled against Aug
  16/18 work this session (out of scope for this session's brief) — still
  flagged as stale in the summary block at the top of this file.

### Related

- `scripts/run_contrastive_vector_experiment.py`,
  `notebooks/ebnerd_contrastive_vector_{kaggle_run.py,src_bundle.zip}`
- `experiments/contrastive_vector_ebnerd_small_2026-08-18/`
- `knowledge/ai-usage-log/2026-08-18_contrastive-vector-adr008-addendum.md`

## August 14, 2026 — PROJECT_STATE Verification + Both Leaderboard Submissions Confirmed Complete

### Context

Session opened with a task briefing claiming PROJECT_STATE.md was stale at
August 10 with Semantic Retrieval at 0% and Leaderboard Submission "Not
Started." Per CLAUDE.md's evidence-hierarchy and this project's own
"verify before handing back" discipline, every claim was checked against
the real repo state before any edit was made, rather than trusted at face
value.

### What the briefing got wrong

- The doc was already dated **August 12**, not August 10.
- Semantic Retrieval had been ✅ 100% complete since August 10 (ADR-008) —
  confirmed by re-reading the Component Status table and the
  `experiments/embed_*` result files directly.
- Q4's diversity/novelty/coverage metrics were **already implemented and
  computed**, not "the only mandatory eval component still at 0%" —
  `grep` of `src/evaluation/ranking_metrics.py` confirmed
  `intra_list_diversity`, `novelty`, and `coverage` all exist, and
  `experiments/ranking_embed_*/results.json` already carry `coverage`
  keys for every dataset.
- A local scoring harness reproducing `evaluation/official/evaluate.py`
  already exists and was already validated end-to-end against it on
  MINDlarge_dev, for both BM25 and embeddings (documented in the
  Codabench Submission Format row and the August 11 session notes) — the
  one MRR disagreement found was root-caused as a metric-definition
  difference, not a harness bug.
- The full BM25-vs-semantic comparison with warm/cold slicing was already
  written up (see the Benchmarking Status section and ADR-008) with a
  real finding against the Day-1 working hypothesis (embeddings win
  outright on MIND, not just on cold-start).

None of the above was re-implemented this session — doing so would have
overwritten working, already-validated code for no reason. Confirmed with
the engineer before proceeding (rather than assuming) which of the
briefing's five steps were actually still open.

### What was actually open, and what closed it

The one real gap: as of August 12, both Codabench leaderboard submissions
were explicitly *not yet uploaded* (EB-NeRD's Kaggle run hadn't been
executed; MIND's manual upload was pending). The engineer provided four
screenshots of completed submissions from their own Codabench account,
dated after the August 12 session:

- **MIND**: `submissions/mind_large_test_embed/prediction.zip` (already
  generated August 12, per Part 4's local half) uploaded — leaderboard
  Score column 0.6195, submission ID 886468, 2026-08-12 13:01.
- **EB-NeRD** (competition 2469): a fresh `prediction.zip` from the Part 2
  Kaggle run uploaded to `submissions/ebnerd_testset_embed/` — leaderboard
  Score column 0.5404, submission ID 888045, 2026-08-13 23:08.

Verified the screenshots were real files on disk (not just chat-pasted
images) by matching filesystem timestamps on the engineer's Desktop
against the submission times in the images, then copied all four into
`submissions/{mind_large_test_embed,ebnerd_testset_embed}/
leaderboard_screenshot_{upload,rank}.png` as the source material for the
Q6 design note (gitignored along with the rest of `submissions/`, per
Q8's existing policy — no `.gitignore` change needed).

Noted, as inference rather than confirmed fact (column headers were
cropped out of both screenshots): the Score column and the three
following columns line up closely, in order, with this project's own
locally-measured AUC/MRR/nDCG@5/nDCG@10 for embeddings on `ebnerd_small`
(0.5430/0.3437/0.3804/0.4591 local vs. 0.5404/0.3447/0.3823/0.4613
leaderboard) — a real informal coherence check on the submission
pipeline, not a claim that the numbers are defined identically or should
match exactly (leaderboard runs on the true held-out test set).

### Updated

- Component Status Summary: Leaderboard Submission → ✅ Complete (100%).
- Deliverables Checklist #3 (leaderboard screenshots): → ✅ Complete.
- Next Actions: MIND Part 4 manual upload and EB-NeRD Part 2
  execution/Part 3 submission both checked off.
- `Last Updated` / `Current Phase` / `Current Objective` at the top of
  this document.

### Genuinely still open

- Q6: the design note (≤4 pages, Moodle) — now has both real leaderboard
  numbers plus ADR-008's interpretation to draw on, previously deferred
  pending exactly this.
- `README.md`'s reproduce instructions still don't mention
  `scripts/run_embed_experiment.py` / `generate_{mind,ebnerd}_predictions.py`
  — flagged again, still not done.

### AI-generated vs. human

All verification commands (file reads, greps, git log, image comparisons)
and all PROJECT_STATE.md edits this session were AI-generated. The
screenshots themselves and the underlying Codabench uploads are the
engineer's own actions, outside Claude Code's reach by design (per
CLAUDE.md's Resource Availability clause — Codabench login is the
engineer's own account). See
`knowledge/ai-usage-log/2026-08-14_project-state-verification-leaderboard-sync.md`
for the verbatim prompt record.

## August 12, 2026 — EB-NeRD Codabench Submission, Part 2: Kaggle Notebook Prepared (Not Yet Run)

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

### Addendum 2 (same day) — `str`-vs-`Path` bug, then a real ~12GB memory
### risk found and fixed before it could be hit

Re-running Cell 4 after Addendum 1's fix (no bundle re-upload needed — the
zip-packaging bug was in `src/`, this one was notebook-only) hit a second,
unrelated real error: `embed_cache_path` was built via plain string
concatenation, but `build_embedding_index`'s disk-cache helpers call
`cache_path.with_suffix(...)`, which only exists on `Path`. Fixed by
building it as a `Path`. Confirmed by the real Kaggle output that
everything upstream was already healthy: BM25 index built cleanly (4.8s,
1,892,580 nonzeros), and Cell 3 had already surfaced the real corpus/user
scale this run is dealing with — 125,541 articles, 807,677 users with
history (mean 144.6 articles/user, up to 1,530).

That real history-length distribution motivated a deeper look before
telling the engineer to just re-run: Cell 5 (not yet reached) builds a
per-user query dict for **every** method unconditionally, including BM25,
whose query is an unweighted concatenation of a user's *entire* history
(ADR-005 — no dedup). Measured directly (real EB-NeRD text, real
`tokenize()`, realistic history-length sampling): ~17.05 tokens/article
after stopword removal, ~15.9KB/user for a query list — projecting to
**~12.25GB** for all 807,677 test users. Embeddings, by contrast, mean-pool
to one fixed 384-dim vector/user regardless of history length —
**~1.16GB** projected for the same population. A real, measured ~10.6x gap,
not a rounding difference, and it would have landed on top of whatever's
already resident (the engineer's own Cell 2 output showed available RAM
dropping from 29.6GB to 15.5GB just from re-running earlier cells in the
same kernel across attempts — likely allocator fragmentation from
repeated in-place reruns rather than genuine leaked state, but a real
signal that headroom was already tighter than the first successful run
suggested).

Restructured Cells 5-7 around a single `RUN_METHODS` list (now defined in
Cell 5, the config's natural home) that gates query construction,
benchmarking, and the full run alike — BM25 is opt-in, not built by
default, updating this session's earlier "build both, decide at Cell 6"
plan with new evidence (CLAUDE.md's Decision Reversal principle: embeddings
already won on `ebnerd_small`-validation accuracy, and now also costs an
order of magnitude less RAM to even attempt). Also found and removed a
stale duplicate `methods = {...}` block left in Cell 6 from before this
restructuring — it referenced the old `bm25_query_by_user`/`embed_scorer`
variable names directly and would have raised `NameError` immediately;
and removed a variable-shadowing landmine (Cell 6/7's per-method loop
reused the name `query_by_user`, shadowing Cell 5's `{method: {user:
query}}` dict of dicts with a single flat dict — not yet a live bug given
current usage, but fragile, renamed to `q_by_user`).

Before handing this back, ran a full local simulation of the restructured
Cells 4-9 against real data (`ebnerd_small.zip`'s validation split
standing in for the real test split — same code paths, smaller scale) for
**both** `RUN_METHODS` configurations (`["embed"]` and `["bm25",
"embed"]`): index build, gated query construction, benchmark sampling,
the full write loop (244,647 real lines each), validation (0 malformed),
and packaging — all passed cleanly before this was reported fixed, per
this session's own "stop patching reactively, verify before handing back"
correction from the engineer.

**Next:** re-paste Cell 5 through Cell 7 (or the whole file) from the
corrected `notebooks/ebnerd_part2_kaggle_test_run.py` — no dataset
re-upload needed this round, only the notebook script changed. Consider a
kernel restart first given the RAM-drop-across-reruns observation above,
for a clean baseline before trusting Cell 5's printed RAM figures.

### Addendum 3 (same day) — investigated a 10.536ms/impression benchmark
### result (3.5x above ADR-008's corpus-scaled projection); found two real
### bugs in Cell 6's sampling methodology, not a scoring-path regression

Cell 6 (fixed in Addendum 2) ran and reported **10.536ms/impression** for
embeddings — 3.5x above the ~2.9ms ADR-008's 0.99ms/query @ 42,416-doc
benchmark would predict at this corpus's 125,541 docs (2.96x bigger).
Investigated before accepting the resulting ~39.6hr projection, per this
session's own "measure, project, then decide" discipline — profiled the
actual per-impression scoring path directly rather than assuming either
"it's just slower hardware" or "something's unvectorized."

**Confirmed NOT the scoring path itself.** Built a synthetic
`EmbeddingIndex` at the real 125,541×384 scale locally and decomposed the
cost: a single fresh `index.vectors @ query` matvec (the exact operation
`embed_retrieve_top_k` does, which is where ADR-008's 0.99ms number came
from) measured **2.90ms** — matching the corpus-scaled projection almost
exactly. `_lookup_scores` (candidate-subset lookup, ~15 candidates):
0.001ms. `rank_candidates`: 0.01ms. Both negligible. `EmbeddingScorer.
score()` under a forced-cache-miss pattern costs exactly the matvec cost,
no more; under a cache-hit pattern (same query reused, its designed use
case) cost drops to 0.003ms — a 1,036x speedup. So the scoring code is
correctly vectorized and matches MIND's own benchmarked code path exactly
— same `Scorer` interface, same underlying operation, no EB-NeRD-specific
regression.

**Found two real, distinct bugs in Cell 6's *sampling* methodology
instead**, both now fixed:

1. `sample_rows()` wrapped `iter_raw_impressions` in an `if i % STRIDE ==
   0` filter — but that filter runs *after* each row is already fully
   parsed (`prefix_id` calls + list comprehension per row), so it pays
   the full per-row cost for every row of the whole split just to yield a
   subset. Measured directly against real `ebnerd_small` data: walking
   244,647 rows to yield 12,233 (stride 20) took the same wall time
   (0.89s) as parsing all 244,647 — the "sample" bought zero speedup on
   the parsing side. At `ebnerd_testset`'s real 13.5M-row scale this adds
   real, unnecessary wall time and inflates the reported ms/impression by
   folding full-file parsing cost into a denominator of only the sampled
   rows — at this machine's parsing rate, ~0.5ms/impression of the
   observed gap; real Kaggle contribution unknown but plausibly larger
   given the earlier-observed CPU/RAM constraints.
2. More consequential: **real EB-NeRD data has substantial row-to-row
   user locality** — measured directly on `ebnerd_small`'s real
   `behaviors.parquet` (both splits): ~52% of consecutive rows share the
   same `user_id` as the row before, mean consecutive-same-user run
   length ~2.08 (max 33-38). The file is not randomly shuffled.
   `EmbeddingScorer`/`BM25Scorer` cache the last query's full-corpus
   score specifically to exploit this — a user's consecutive impressions
   should reuse one corpus-wide computation instead of recomputing it.
   Cell 6's evenly-STRIDED sample (jump 135 rows every time) is close to
   the worst possible access pattern for that cache: consecutive sampled
   rows essentially never share a user. So the benchmark was measuring a
   near-worst-case "cache always misses" floor (≈ the raw matvec cost,
   2.9ms, consistent with the local finding above), not what Cell 7's
   real *sequential* access would actually experience.

Neither bug fully explains the observed 10.536ms in isolation on this
local machine's numbers (2.9ms raw + ~0.5ms parsing-overhead ≈ 3.4ms, not
10.5ms) — the remaining gap is most plausibly a genuine Kaggle-instance
BLAS/numpy backend difference from the machine ADR-008's 0.99ms was
measured on (`index.vectors @ query` is plain CPU numpy, not
GPU-accelerated, regardless of the encode step's device) — a real
hardware/environment difference, not a code bug, and not something
re-running locally can confirm or rule out. **This is exactly why Cell 6
was redesigned to report a decomposed, isolated per-call cost separately
from the realistic locality-preserving sample**, rather than one opaque
aggregate number — the next real Kaggle run will show directly how much
of the gap is hardware vs. was methodology.

**Fixed:**
- `sample_raw_impressions` (new, `src/submission/ebnerd_format.py`):
  selects row positions at the pandas level first (cheap, vectorized —
  `.iloc[positions]`, no per-row Python work for skipped rows), then
  applies the same per-row transform `iter_raw_impressions` uses (now
  factored into a shared `_row_to_impression` helper so the real run and
  a benchmark sample can never silently diverge in what they compute per
  row). Preserves caller-requested order (doesn't re-sort to file order)
  so a locality-preserving sample stays locality-preserving.
- Cell 6 rewritten: samples 10 contiguous 10,000-row chunks scattered
  across the file (representative across the whole split — the original
  stride design's actual goal — while preserving real within-chunk user
  locality, the actual thing caching needs) instead of one giant
  evenly-strided sample. Adds an isolated single-call diagnostic (forces
  a cache miss via a same-content-different-identity query/token-list
  copy) that measures the raw per-call scorer cost directly, printed
  alongside ADR-008's corpus-scaled projection for direct comparison, and
  alongside the realistic chunked-sample cost so the caching benefit
  itself is visible as a printed multiplier. Decision gate text updated
  to interpret both numbers rather than treat one aggregate as ground
  truth.
- **Separately, as asked**: Cell 9 (and Cell 10) guarded against
  `FileNotFoundError` when `RUN_FULL_JOB` was `False` and no output file
  existed — now prints a clear "nothing to package yet, not an error"
  message and skips, matching Cell 8's existing pattern.

Added 3 new tests (`tests/unit/test_ebnerd_format.py`:
`sample_raw_impressions` matches `read_raw_impressions` at selected
positions, preserves requested order rather than file order, supports
repeated positions). Full suite: 172 passed (up from 169), 1 pre-existing
skip, no regressions. Verified the new Cell 6 methodology end-to-end
against real `ebnerd_small` data before reporting this fixed (per this
session's own "verify before handing back" correction, now also saved as
a standing memory) — sample selection (10,000 rows) took 0.92s vs. the
old method's full-244,647-row walk, and the chunked benchmark showed a
real, visible caching benefit (1.8x for BM25, 3.0x for embeddings at this
smaller corpus scale) rather than the old method's near-zero benefit.
Rebuilt `notebooks/ebnerd_part2_src_bundle.zip` with the fix.

**Not yet known:** how much of the remaining ~7ms gap (10.5ms observed
vs. 2.9ms local-matvec-only) is genuinely BLAS/hardware-driven on Kaggle
specifically — only a real re-run of the corrected Cell 6 on Kaggle can
answer that, and per the task brief's explicit instruction, Cell 7 (the
full run) stays untouched until this is resolved either way.

**Next:** re-paste the corrected `notebooks/ebnerd_part2_kaggle_test_run.py`
(no dataset re-upload needed — only the notebook and, this round, the src
bundle changed; re-upload the bundle too since `ebnerd_format.py`
changed) and re-run Cell 5 through the new Cell 6. Relay back the
isolated-cost and chunked-sample numbers for both — if the isolated cost
is still far above ~2.9ms, that's the real hardware-difference signal; if
the chunked sample now shows a large caching benefit and a much lower
projected full-run time, the original plan may simply proceed.

### Addendum 4 (same day) — the fixed Cell 6 ran for real: confirmed a
### genuine ~2.48x Kaggle-hardware slowdown, projection dropped to 12.80h;
### built cross-session checkpointing since that's borderline against a
### free-tier session cap

Engineer re-ran the corrected notebook on Kaggle (Tesla T4, ~30GB RAM).
Cells 1-5 completed cleanly with the real numbers Addendum 2/3 already
anticipated (807,677 users, 125,541 articles; embedding encode 147.9s at
849.1 articles/s on the T4; embedding query_by_user 105.3s; RAM settled
at 13.3-13.7GB available, comfortable). Cell 6's decomposed diagnostics,
against the real file:

- **Isolated cache-miss cost: 7.198ms/call** (22 candidates) vs. this
  session's local-machine measurement of 2.90ms for the identical
  operation at the identical corpus scale — confirms a real, genuine
  **~2.48x Kaggle-vs-local hardware/BLAS speed difference** for the raw
  `index.vectors @ query` matvec. Not a bug, not fixable in code — numpy's
  matvec is plain CPU regardless of the encode step's GPU device, and
  Kaggle's numpy/BLAS backend is measurably slower at this operation than
  the engineer's Mac's Accelerate/vecLib BLAS.
- **Realistic chunked-sample cost: 3.405ms/impression**, with a real
  **2.1x caching benefit** over the isolated cost — confirms the
  ~52%-locality finding (Addendum 3) holds on the real 13,536,710-row
  file, not just `ebnerd_small`. Projected full run: **12.80 hours**, down
  from the old (methodologically flawed) benchmark's ~39.6-hour
  projection — roughly a 3.1x improvement from fixing the sampling bugs
  alone, layered on top of the genuine ~2.48x hardware factor.

So the original 3.5x gap decomposes cleanly: ~2.5x genuine Kaggle
hardware slowness × a bit under 2x from the two benchmark-methodology
bugs Addendum 3 fixed. Neither number is in question anymore — this is
the real, trustworthy projection.

**Decision point:** 12.80h is genuinely borderline against a free-tier
Kaggle account's commonly-cited ~9-12h session/commit-run cap (confirmed
by the engineer: free tier). Per the session brief's own instruction to
"decide between the checkpointing and background-commit options" once a
legitimate scaling difference is confirmed (not just a code bug) — asked
the engineer directly rather than guessing their account limits; they
confirmed free tier and asked for checkpointing to be built (the more
robust option regardless of exactly where Kaggle's real cap falls, vs.
hoping "Save & Run All (Commit)" happens to have enough headroom).

**Built:**
- `iter_raw_impressions_from` (new, `src/submission/ebnerd_format.py`):
  same per-row transform as `iter_raw_impressions`/`sample_raw_impressions`
  (reuses the shared `_row_to_impression` helper), but does a
  `.iloc[start_row:]` slice first — resuming from row N costs the parquet
  read plus parsing only the *remaining* rows, not a wasted re-parse of
  everything already written.
- Cell 7 rewritten around a self-imposed `MAX_RUNTIME_HOURS` (default
  8.0, deliberately under the ~9-12h real cap) that stops the write loop
  cleanly — current line finished, file flushed — rather than letting
  Kaggle kill the process mid-write and risk a truncated last line. On
  start, searches `/kaggle/working` then `/kaggle/input` for an existing
  `predictions_<method>.txt` (this session's own in-progress file takes
  priority over a re-attached checkpoint from an earlier session, since
  it's always at least as far along); if found, validates only the *last*
  line (the one place a partial write could ever land, since every clean
  stop already flushed a complete line) and discards it if malformed,
  computes `resume_from` from the valid line count, and continues writing
  from there via `iter_raw_impressions_from`. Cross-session workflow
  (download the partial file, upload as a new version of a private
  checkpoint Dataset, re-attach in a fresh session, re-run from Cell 1) is
  documented in Cell 7's own docstring.
- Cell 8 now distinguishes `INCOMPLETE` (valid so far, just not finished —
  expected and routine now) from `FAILED` (malformed content) rather than
  reporting both as one undifferentiated failure.

**Verified before reporting this fixed** (per this session's own standing
"verify before handing back" memory): 3 new unit tests for
`iter_raw_impressions_from` (matches full iteration from row 0, correctly
skips already-written rows, empty result past the end of the file); full
suite 175 passed (up from 172), no regressions. Beyond unit tests, wrote
a standalone harness running Cell 7's *exact* real logic (checkpoint
discovery, last-line truncation validation, resume-from-row) against the
real `ebnerd_small.zip` validation split (244,647 real rows, real BM25
scorer): a full uninterrupted reference run, then a deliberately
interrupted run (stopped at 50,000 rows, last line artificially truncated
to simulate a mid-write kill, checkpoint moved to a separate directory
simulating a fresh session's `/kaggle/input`, resumed to completion) —
**the two outputs are byte-for-byte identical**, confirming truncation
handling, resume-from-row correctness, and that the seeded tie-break in
`ranks_for_impression` (seeded by `impression_id`, not call order) is
correctly insensitive to session boundaries. Rebuilt and re-verified
`notebooks/ebnerd_part2_src_bundle.zip`.

**Next:** engineer creates a private Kaggle Dataset to hold checkpoints
(e.g. "ebnerd-part2-checkpoint"), re-uploads the bundle (changed again
this round), re-pastes the notebook, sets `RUN_FULL_JOB = True`, and lets
Cell 7 run — expect it to stop itself at the `MAX_RUNTIME_HOURS` budget
and print a `CHECKPOINT` message with exact next-session instructions,
likely needing 2 sessions total for the real 12.80h projection against an
8.0h budget. Relay back each session's final printed status (`CHECKPOINT`
or `DONE`) so progress stays visible across the multi-session run.

### Addendum 5 (same day) — switched to a single "Save & Run All (Commit)"
### run: engineer confirmed the real Kaggle limit (~25h) comfortably
### covers the 12.80h projection, no multi-session split actually needed

Engineer confirmed the real Kaggle commit-run limit is ~25h — well above
the 12.80h real projection from Addendum 4, so the multi-session
checkpointing plan (built as a real, verified capability in Addendum 4)
turns out not to be strictly necessary; a single unattended "Save & Run
All (Commit)" pass should cover the whole run.

**Changed in `notebooks/ebnerd_part2_kaggle_test_run.py`** (no `src/`
changes this round — the bundle from Addendum 4 already has everything
this needed):

- `RUN_FULL_JOB` set `True` (was `False`) — the actual green light for
  the committed run, double-confirmed explicit and correctly placed per
  the session brief's own concern (a 12+ hour commit doing nothing
  because this was left `False` would be exactly the failure mode being
  guarded against).
- `MAX_RUNTIME_HOURS` raised `8.0` -> `20.0` — comfortable buffer under
  the confirmed ~25h real limit, well above the ~12.80h this run should
  actually take, so Cell 7 shouldn't self-checkpoint at all in the normal
  case.
- Cell 7's deadline is now anchored to a `NOTEBOOK_START_TIME` captured
  at the very top of Cell 1, not to Cell 7's own start — real, worth
  getting right: in a "Save & Run All (Commit)" run, Cells 0-6's setup
  (encode, query build, the Cell 6 benchmark — real measured cost ~10
  min from Addendum 4's relayed output) happens before Cell 7 even
  begins, and the ~25h limit applies to the whole commit run's wall
  time, not just Cell 7's own portion of it. Anchoring to
  `NOTEBOOK_START_TIME` means the 20h budget is measured from the true
  start, closing a real (if small, ~10-20 min) gap in the original
  Cell-7-local deadline.
- Checkpoint/resume logic itself is unchanged and stays live as the
  safety net (Kaggle killing the run unexpectedly, a shorter-than-
  expected real limit) — per the session brief's explicit instruction.
- **Added a real safety check the session brief's "confirm nothing else
  assumes multi-session behavior" prompted**: this run is now unattended
  (nobody watching "Save & Run All" live), so if Cell 7 happened to find
  a *stale* `predictions_embed.txt` left over from an unrelated earlier
  attempt, it would previously have silently trusted it as a genuine
  resume point — Cell 8's line-count/format check would not catch this,
  since a stale prefix is still well-formed, just wrong. Added a
  cross-check (two cheap single-row lookups via `sample_raw_impressions`)
  comparing the checkpoint's first and last `impression_id` against what
  the real `ebnerd_testset.zip` actually has at those row positions;
  raises `RuntimeError` and stops rather than silently proceeding if they
  don't match.

**Verified all three scenarios against real `ebnerd_small` data** (a
standalone harness running Cell 7's exact current logic, not a
reimplementation) before reporting this ready:
1. A single uninterrupted pass (the now-default commit scenario) writes
   all 244,647 lines correctly.
2. A genuine matching checkpoint (from an interrupted-then-resumed run)
   still resumes correctly and produces output identical to the
   uninterrupted case — the new safety check doesn't break the real
   resume path.
3. A **stale/mismatched checkpoint** (fabricated from `ebnerd_small`'s
   `train` split's row ordering, standing in for an unrelated leftover
   file) is correctly **rejected** with a clear `RuntimeError` rather than
   silently trusted.

Confirmed `RUN_METHODS = ["embed"]` in Cell 5 is unchanged (still the
right choice — accuracy edge on `ebnerd_small`-validation, and the
dramatically lower memory cost vs. BM25's ~12.25GB found earlier). No
`src/` files changed this round, so `notebooks/ebnerd_part2_src_bundle.zip`
does not need re-uploading — only the notebook script itself changed.

**Next:** engineer re-pastes the updated notebook (bundle unchanged,
already uploaded), runs "Save & Run All (Commit)" with `RUN_FULL_JOB`
already `True`, and lets it run unattended. Expect a `DONE` status in
Cell 7's output on completion (~12.80h, plus ~10-15 min of Cells 0-6
setup); the `CHECKPOINT` path should not trigger under normal conditions
but is verified correct if it does. Relay back Cell 7's final status,
Cell 8's validation line, and Cell 9's packaging confirmation once the
commit finishes.

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
| Phase 1 Complete | August 12, 2026 | ✅ Done |
| Phase 2 Complete | August 19, 2026 | ✅ Done |
| Pipeline & Retrieval Complete | August 26, 2026 | ✅ Done |
| Leaderboard Submission | August 26, 2026 | ✅ Done — 3 MIND submissions (best: 901961 @ 0.6462), 2 EB-NeRD submissions |
| Final Assignment Submission | August 27, 2026 | **Deadline — tomorrow. See Deliverables Checklist (Q7) above; design note §3.5/§6 still needs updating for Candidate J.** |

---

## Session Notes — 2026-09-03/04: Pipeline Performance Profiling (ADR-014)

**New requirement from class:** comprehensive latency/throughput profiling, ablations
measured on *performance* rather than accuracy, per-hardware benchmarks, and an
ongoing pre-commit habit. Delivered as ADR-014 plus a `benchmarks/` directory that
mirrors how `experiments/` already captures accuracy work.

### The bottleneck, named

**MIND: NRMS inference — 34.920 ms/impression, 79.9% of serving-path latency on the
local M3** (7.5x the next stage; 90.7% on Ada's Xeon CPU). Confirmed independently by
cProfile (73.0% of `run_batch`).

**EB-NeRD: feature engineering, NOT model inference.** `build_feature_frame` is 63.5%
of the serving path while LightGBM predict is 8.0%. Inside it,
`compute_short_term_features` alone is **47.4% of the whole path** — and contributes
**zero features to the top 20 by gain** (best member ranks #27). Largest cost/value
mismatch found. Not a delete instruction: ADR-013 records fixing this block was part
of the 0.7514 -> 0.7581 gain; the AUC cost of removal is unmeasured.

**The two datasets do not share a bottleneck.** Assuming they did would have sent
effort to the wrong place in one of them.

### Hardware changes the answer (measured, not assumed)

| | NRMS share of serving path |
|---|---:|
| Ada Xeon E5-2640 v4 (CPU) | 90.7% |
| Local Apple M3 (CPU) | 79.9% |
| Ada RTX 2080 Ti (GPU) | **47.1%** |

Same code/data/sample. On the internally-controlled pair (one node, one venv, one
allocation) GPU makes NRMS **8.98x faster** (64.243 -> 7.152 ms); `bm25_score_all`
then becomes a real second bottleneck at 32.3%. Serving throughput: 22.9 imp/s (M3)
/ 14.1 (Ada CPU) / **65.8 (Ada GPU)**.

Two counterintuitive findings: **local MPS is 2.1x SLOWER than the same machine's CPU**
for NRMS at batch=1 (dispatch overhead) even though `embed.py::_default_device` prefers
it; and **several stages are slower on the cluster** (`bm25_score_all` 0.60x, `metrics`
0.40x) because the 2016-era Xeon is a weaker single-thread part than the M3. "Move it to
the HPC cluster" only helps where the GPU does the work.

### Performance ablations (new axis)

| Ablation | Deployed | Alternative | Speedup | Accuracy cost |
|---|---|---|---:|---|
| BM25 scorer | sparse 3.75 ms | `rank_bm25` 9,406.72 ms | **2,507.9x** | none (5.7e-14) |
| ANN | brute numpy 5.51 ms | FAISS IVF 0.74 ms | 7.45x | **recall@100 0.6511** |
| ANN | brute numpy 5.51 ms | FAISS Flat 46.88 ms | **0.12x** | none |
| NRMS cache (CPU) | 9.826 ms | 1.399 ms | **7.02x** | none (exactly 0.0) |
| NRMS cache (GPU) | 3.749 ms | 1.319 ms | 2.84x | none |
| LightGBM | 60 feat 0.0616 ms | 20 feat 0.0569 ms | 1.08x | AUC -0.0023 |

FAISS rejected again, now on accuracy not speed. FAISS Flat being **8.5x slower than a
plain numpy matvec** is the most counterintuitive result. The **NRMS candidate-vector
cache is a free 7.02x on CPU** with numerically identical output and no retraining —
the highest-value unimplemented change this profiling found.

### One recorded figure does NOT reproduce

**ADR-006's ~9.8 min BM25 full-retrieval projection (2.31 ms/user) does not
reproduce**; measured 3.62-4.33 ms/user -> **15.4-21.4 min** on two machines. The
obvious explanation (unrecorded sample-draw method) was **tested and refuted** — a
prefix sample is *slower* than systematic (4.158 vs 3.618 ms, ratio 0.87), so it
cannot produce an under-estimate. Gap unexplained; recorded as an ADR-006 addendum
rather than a silent edit. ADR-006's actual decision is unaffected (21 min is still
far under the >2hr migration threshold). ADR-008's ~10.4 min **does** reproduce
(12.25 min measured, 18% high).

### The ongoing habit

`benchmarks/snapshot.py` (~1.5s tripwire, `--compare` exits nonzero on regression),
`benchmarks/PERFORMANCE_LOG.md` (auto-appended), `.githooks/pre-commit` (fires only on
performance-relevant paths; `make hooks` to install; `SKIP_PERF_SNAPSHOT=1` to bypass),
`make bench-baseline` / `bench-snapshot` / `bench`.

A defect found and fixed while building it: two *identical* runs disagreed by -44.5% on
a 0.02 ms stage, so a pure percentage gate would fire on pure noise. Regression now
requires **both** >15% relative **and** >0.25 ms/impression absolute.

### Ada notes (new, reusable)

Ada is reachable passwordless as `sukhraj.singh@ada.iiit.ac.in`. **Standing rule: never
`scancel` a job on that account that this tooling did not submit.** Jobs submitted this
session: 2687384/2687386/2687388 (prep), 2687421 (profile), 2687432 (resume), 2687446
(reconcile).

Four real environment failures, all now encoded in `benchmarks/ada_*.sbatch`:
- `uv pip install torch` **fails on the login node** (CUDA wheels are mmap'd during
  extraction; per-user memory cap -> `Cannot allocate memory (os error 12)`). Must run
  in a job. `torch` cannot even be **imported** on the login node for the same reason —
  verify a venv from inside a job, never from the login shell.
- `--extra-index-url` is **not sufficient**: the resolver preferred PyPI and picked
  torch 2.14.0 -> CUDA-13 wheels, the documented trap where install succeeds and
  `cuda_available` is silently False. `--index-url` required; prep now hard-fails if
  `torch.version.cuda` is not 12.x.
- `-p u22-cpu` is **unavailable to the `research` account** (`sacctmgr` shows one
  association: research/medium). Use `u22` with no `--gres=gpu`.
- **`quota` exits nonzero even when it prints fine**, so a bare `quota` under
  `set -euo pipefail` marked job 2687388 FAILED *after all real work succeeded*.

**Version parity is imperfect and unavoidable:** cu128's newest cp311 torch is 2.11.0
vs local 2.13.0, and the nodes cap at CUDA 12.8. Cross-machine comparison is confounded;
the GPU-vs-CPU conclusion uses the within-node controlled pair instead.

### Measurement hygiene (learned the hard way)

Running two MINDlarge-scale jobs concurrently on the 8GB machine moved `bm25_score_all`
from **4.33 to 10.9 ms/user** — a 2.5x distortion on identical code and data. The
existing machine-stability note is now a hard requirement for benchmarking, not a
suggestion. Results from the contaminated run were discarded.

### Gaps, stated not papered over

- **Kaggle T4 not measured** — no API credentials on this machine. PROJECT_STATE's
  existing T4 macro figures are not per-stage and are not presented as such.
- **NRMS timings use randomly-initialised weights** (real recorded architecture; no
  trained Candidate J checkpoint exists locally). Valid for latency, no accuracy claim.
- **EB-NeRD used the 60-feature 2026-08-26 booster**, which predates ADR-013's
  correction; the 65-feature `ebnerd_large` model that scored 0.7542 stayed on Ada.
  Inference cost is near-flat in feature count (1.08x), so the conclusion holds.
- **Local EB-NeRD profile and local ablations not yet run** (engineer asked that local
  CPU load stop mid-session). Ada covers both; the local rows can be filled with
  `make bench` when the machine is free.

