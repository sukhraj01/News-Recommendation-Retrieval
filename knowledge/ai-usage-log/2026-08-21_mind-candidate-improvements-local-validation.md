# AI Usage Log — 2026-08-21 — MIND candidate-improvement local validation

## Prompts (verbatim, in order)

1. > Current Objective: Before committing to a real MIND second-submission attempt, cheaply test 3 candidate improvements on local validation (MINDsmall-dev, no Kaggle) against the existing baseline (AUC 0.634 overall) — commit only whichever shows a real, CI-clear win to the actual Codabench submission.
>
> Check MIND's Codabench competition submission limit (competitions/13967) — report the real number, don't assume it's tight or generous.
> Candidate A — entity embeddings: build per-article entity vectors from title_entities/abstract_entities + entity_embedding.vec (confidence-weighted pooling), measure real article coverage first, then evaluate on MINDsmall-dev.
> Candidate B — BM25+embedding hybrid: blend existing BM25 and MiniLM embedding scores (start with an untuned 50/50 blend, same pattern as the leaky-feature ablation), evaluate on MINDsmall-dev.
> Candidate C — recency-weighted history: re-test ADR-005's rejected recency-weighting assumption specifically for MIND's query/user-representation construction, evaluate on MINDsmall-dev.
> Report all three against the 0.634 baseline with bootstrap CIs, plainly — including if none of them show a CI-clear win. Don't recommend committing to a real submission until we see the real numbers.

2. > Current Objective: Candidates A (entity embeddings), B (BM25+embedding hybrid), C (recency-weighted history) all lost against the deployed MIND baseline (AUC 0.6340) on MINDsmall-dev. Rather than just tuning those same three mechanisms further, explore genuinely different approaches for a real second MIND submission — you have latitude here, don't limit yourself to what's listed below.
>
> Some starting directions, not an exhaustive or mandatory list:
>
> Raw entity/category overlap as a re-ranking signal, distinct from A's dense TransE similarity — symbolic match instead of vector distance.
> A small learned combiner (even plain logistic regression) over multiple real features — embedding cosine sim, BM25 score, category match, entity overlap, recency — trained on MINDsmall-train's actual labels, instead of a hand-picked blend weight.
> Anything else in the data that's a plausible, real signal for click likelihood that hasn't been tried — look at what's actually in news.tsv/behaviors.tsv and use your judgment.
>
> Ground rules, same as everything else in this project:
>
> Whatever you try, evaluate it on MINDsmall-dev with the same rigor as A/B/C — real bootstrap CIs, honest reporting, no cherry-picking the best-looking run.
> Codabench's MIND submission limit is generous (10/day, 999 total, confirmed via their API) — you don't need to find one perfect answer, you can try several ideas for real.
> Only commit something to an actual Codabench submission once local validation shows a genuine CI-clear win over 0.6340 — a well-documented negative result is still a fine outcome given the grading note, but don't submit something you already know underperforms.
> Document whatever you try as ADR addenda, win or lose, same as A/B/C.
> Report back plainly with real numbers once you've explored this — I want the actual results, not a progress narration.

3. > Current Objective: Candidate G is verified at MINDlarge scale — real, CI-clear win (+0.0019 AUC, 95% CI +0.0015 to +0.0023 over the deployed baseline). Build the actual second MIND Codabench submission using this method.
>
> Save the results package from this run to experiments/candidate_g_gated_cohort_mind_large_2026-08-21/ as instructed by the notebook output.
> Generate real predictions over MINDlarge_test (not dev) using the same cohort-gated logic: deployed embedding baseline for warm users, the fitted (not retrained) learned combiner for cold users, same cold_threshold=5.
> Confirm whether this needs Kaggle or can run locally — given MINDlarge dev's ~376K impressions took under 2 hours total (index build + eval) on Kaggle, check if that pace makes the full test set feasible locally within a reasonable time before defaulting to another Kaggle round.
> Package and format the predictions exactly like the first MIND submission (ID 886468) did — verify the format before packaging, same discipline as every other submission.
> Tell me when it's ready to upload as the second MIND Codabench entry — I'll do the actual upload once you confirm it's built and verified.
> Document as a new ADR addendum (building on ADR-008/whatever documented Candidates A-G) and update PROJECT_STATE.md.

(Interleaved with this prompt: extensive live troubleshooting of the Kaggle
notebook — wrong-notebook confusion, Kaggle auto-extracting the upload into
a Dataset folder instead of a zip, a stale/corrupt partial download, two
real cross-platform code bugs found and fixed live — and a mid-run laptop
sleep. Not reproduced verbatim here; the real fixes and their reasoning are
recorded in `decisions/ADR-010-mind-second-submission-candidate-search.md`'s
2026-08-21/22 addendum and this file's own commit history.)

4. > Current Objective: Candidate G's real Codabench result is in — 0.6192, essentially flat/slightly below the original submission's 0.6195, despite a CI-clear +0.0019 win on local MINDlarge-dev validation. Document this honestly as a validation-to-test transfer finding, not a win.
>
> Update the ADR-008/candidate addendum with the real Codabench score (0.6192, ID 896696) alongside the local MINDlarge-dev result (+0.0019 CI-clear), stated plainly as: local re-verification predicted a small real win, the actual blind test set showed an essentially flat result instead.
> Add a short, honest discussion of why this gap might exist — most likely explanation is population difference between MINDlarge_dev and MINDlarge_test (different users/day window), not a bug — same category of finding as EB-NeRD's contrastive-vector result compressing between local validation and the real test set.
> Update the design note's leaderboard section (§3.5 or wherever it lives) and PROJECT_STATE.md with both real MIND submissions now on record.
> Confirm the AI usage log captures this session's work.

(Before writing the discussion, checked this repo for an actual EB-NeRD
contrastive-vector test-set Codabench score to cite rather than assume one
— found only a "prepared, not yet run" status on record as of the Aug 19
log, and asked the engineer directly rather than guess or invent a number.
The engineer's first reply ("i dont get what are you saying") indicated
the question was unclear; asked a simpler, more direct version. The
engineer then answered by editing `decisions/ADR-008-semantic-retrieval-
design.md` directly, adding its own real "Second EB-NeRD Submission"
addendum with submission 896072's confirmed result — read in full before
using it as the basis for this file's own parallel discussion, not
paraphrased from memory of the earlier, now-superseded "prepared" status.)

## Notes on AI vs. human authorship

**Round 1 (prompt 1):** new module(s) under `src/retrieval/`, three new
`scripts/run_*_experiment.py` drivers, unit tests, experiment output under
`experiments/` — Claude Code–generated, following this project's existing
patterns (`scripts/run_leakage_ablation.py`'s untuned-blend design,
`scripts/run_ranking_eval.py`'s Q4 harness, `src/retrieval/embed.py`'s
`EmbeddingIndex`/`Scorer` seam). No ADR was written for round 1 at the time
— per that prompt's own framing, ADR-writing was deferred until a candidate
showed a real result worth committing to. None did in round 1, so
ADR-005/ADR-008 addenda (documenting the three losses) were written
instead of new ADRs.

**Round 2 (prompt 2):** `src/retrieval/features.py`, four more
`scripts/run_*_experiment.py` drivers (symbolic overlap, popularity,
learned combiner, cohort gating), their unit tests, experiment output, and
`decisions/ADR-010-mind-second-submission-candidate-search.md` (a new ADR
this time, since round 2 found a real win and the decision space didn't
fit naturally under an existing ADR) — all Claude Code–generated, same
pattern-reuse discipline as round 1. Round 2 also added
`paired_metric_diff_ci` (`run_gated_cohort_experiment.py`), a statistical
technique not used elsewhere in this project before this session, needed
because the marginal-CI heuristic used everywhere else in this project
turned out to be the wrong test for Candidate G's specific (paired)
comparison — flagged explicitly in ADR-010's Notes section, not silently
applied.

**Round 3 (prompt 3):** the Kaggle notebook (`notebooks/mind_gated_cohort_
mindlarge_{kaggle_run.py,src_bundle.zip}`) and its iterative fixes,
`scripts/generate_mind_gated_predictions.py` + its unit tests, the
`src/utils/io.py` flat-root-zip fallback + tests, the
`src/retrieval/embed.py` device-auto-detect fix + tests,
`submissions/mind_large_test_gated_cohort/` (the actual second-submission
prediction file), and ADR-010's 2026-08-21/22 addendum — all Claude
Code–generated. The MINDlarge-dev headline result itself (+0.0019 AUC, 95%
CI +0.0015 to +0.0023) came from the engineer's own real Kaggle run,
reported to Claude Code verbally rather than as a retrieved file — the full
config.json/results.json is a still-pending follow-up (noted in ADR-010's
addendum), not fabricated or estimated. The actual Codabench upload was
explicitly left to the engineer, not performed by Claude Code. Not yet
reviewed/edited by the engineer at time of writing; this line should be
updated once it has been.

**Round 4 (prompt 4):** ADR-010's 2026-08-22 addendum (the real-result
writeup), `docs/design_note.md`/`.tex` §3.5/§6 updates, the rebuilt
`docs/design_note.pdf`, PROJECT_STATE.md's Leaderboard Submission row and
top-summary updates, and this file's own log entries — all Claude
Code–generated. Two real facts this round depends on came from the
engineer directly rather than being derived or assumed by Claude Code: the
real MIND Codabench score (0.6192, submission 896696) and — after Claude
Code asked rather than guessed — the real EB-NeRD contrastive-vector
test-set result (submission 896072, via the engineer's own edit to
ADR-008), both cited as given facts, not re-derived. The MIND↔EB-NeRD
"same category of finding" framing and the MIND dev/test population-
difference explanation are Claude Code's own analysis, grounded in
ADR-001's already-recorded temporal-split facts (checked directly, not
assumed) rather than asserted without support.
