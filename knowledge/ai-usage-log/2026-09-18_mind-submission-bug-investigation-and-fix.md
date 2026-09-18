# AI Usage Log — 2026-09-18 — MIND Submission Collapse: Root Cause, Fix, Final Score

Verbatim record of every engineer prompt this session, in order, plus a note on
what was AI-generated vs. human-written. Per CLAUDE.md's "Prompt & Session
Logging" clause. One prompt (Prompt 7) contained what looked like a real
credential; it is redacted below rather than reproduced verbatim, since this
log is committed to git and later pushed to the remote — reproducing a
credential in permanent, pushed history would itself be a security incident,
and CLAUDE.md's logging clause exists for engineering reproducibility, not to
compel committing a secret. See the note at Prompt 7.

---

## Prompt 1 (verbatim)

> Current Objective: Submission 930353 scored 0.5589 on the real MIND leaderboard — far below the locally-validated 0.6868 AUC, and far outside the compression range every other candidate in this project has shown (hundredths of a point, not 0.13). This is almost certainly a pipeline bug, not a genuine generalization failure. Investigate methodically before resubmitting anything. Submission 901961 (0.6462) remains the selected official entry — don't touch it, don't select 930353 as official.
>
> Confirm the exact checkpoint used to generate prediction.zip for 930353 was job 2695890's actual final checkpoint — not a stale one, not an earlier epoch, not accidentally the wrong job's output. Check the file timestamp/hash against the checkpoint's own hash.
> Check the ranking direction explicitly: confirm higher model scores produce lower (better) rank numbers in the submitted format, matching Codabench's expected convention — a sign flip or reversed argsort would produce exactly this kind of collapse toward random.
> Check candidate/impression alignment: pull 5-10 real impressions from MINDlarge_test, manually verify the candidate order and impression_ids in the submitted predictions match what behaviors.tsv actually specifies for those impressions — not a shuffled or off-by-one mapping.
> Re-run a small manual spot check: score those same 5-10 impressions fresh, independently, outside the submission pipeline, using the same checkpoint, and compare to what's embedded in the committed prediction file. If they disagree, the bug is in the generation path, not the model.
> Report back with the specific root cause found — don't resubmit until you know why, given the file format itself already passed full validation (well-formed isn't the same as correct).

## Prompt 2 (verbatim)

> check agat ada it is free

## Prompt 3 (verbatim)

> no i am talking agat.choubey account

## Prompt 4 (verbatim)

> yes it is mine account or else how would i be able to set up passwordless login it is my alternate username

## Prompt 5 (verbatim)

> try again with ssh -i ~/.ssh/id_ed25519_agat agat.choubey@ada.iiit.ac.in

## Prompt 6 (verbatim)

> ok good but it wont be blocked by harness's permission classfifer now

## Prompt 7 (redacted — see note above)

> [REDACTED: a single token that had the shape of a password/passphrase, sent
> with no other text, immediately after the account-access exchange above.
> Not used for anything; flagged back to the engineer in-session as likely
> sensitive and inadvisable to paste into chat.]

## Prompt 8 (verbatim)

> try agian i have connected to the vpn

## Prompt 9 (verbatim)

> 930992 scored 0.6766 on the real MIND leaderboard — sane, consistent with local validation (0.6868 → 0.6766, normal compression range), and a real +0.0304 over the previous official submission. It's now selected as the official entry on Codabench, replacing 901961.
>
> Final pass:
>
> Update ADR-015 and PROJECT_STATE.md with 0.6766 as the real, final, official MIND score — replace any remaining placeholder/interim reference.
> Update Q6's design note §1.2 (the one section flagged as still tracing back to interim data) with this real number. Recompile and confirm the page count is still within target.
> Take the new leaderboard screenshot showing 930992 as the selected submission, for Q7's deliverable.
> Commit everything in logical groups and push.
>
> Once this lands, give me a final status check across all of Q1-Q9 for both datasets, confirming nothing is still open.
> screenshot is in the desktop folder

## Prompt 10 (verbatim)

> i have added the submission check the latest screenshot and add it

---

## Findings and decisions this session

**Root cause of 930353's 0.5589 collapse (ADR-015's 2026-09-18 addendum, full detail there):**
`a2_generate_mind_test_predictions.py`'s hand-rolled `read_history_raw()` keyed its
`query_by_user` dict by the raw MIND user id, while `mind_format.py::read_raw_impressions()`
looks history up by the `prefix_id`-prefixed id. The keys never matched, so every one of
2,370,727 test impressions was silently scored with empty (all-padding) history. Checkpoint
provenance, file transfer, and ranking direction were each independently verified clean first
(sha256 checks, job-log cross-references on Ada, a diff of every touched source file) before
the bug was found via a 10-impression independent re-scoring spot check that reproduced the
submitted file's exact rank lists only when history was forced empty.

**Decision: retired the hand-rolled reader rather than patch the one key**, routing the script
through `src/pipeline/orchestrator.py::build_mind_test` instead — the same tested path
Candidate J's script and the Q2 harness already use, and the path `OfficialNRMSScorer`'s own
docstring already assumed. Judged lower-risk than a one-line patch with 3 days left: it deletes
duplicated code instead of adding a second bespoke path, and inherits `build_mind_test`'s
existing test coverage plus a static-history invariant check the hand-rolled version lacked.

**New permanent pre-upload gate** (`scripts/a2_check_mind_history_coverage.py`, per the
engineer's explicit request): computes the real non-empty-history rate two independent ways and
fails loudly on divergence. Verified on the full real MINDlarge_test (2,370,727 impressions):
ground truth and the fixed pipeline both give exactly 98.77% (2,341,619/2,370,727) — 0.0000%
difference. Wired into the generation script as a pre-GPU-time preflight and documented in
README.md as a required step before every future MIND upload.

**Regenerated, validated, and (per Prompt 9) uploaded.** Same checkpoint, no retraining;
regenerated via Ada job 2700133 (COMPLETED, 1:56:38), format-validated clean (2,370,727 lines,
0 malformed, 0 duplicates), confirmed different line-by-line from the broken file. Uploaded by
the engineer as submission 930992 — **real Codabench score 0.6766**, inside this project's
normal dev-to-test compression range and a real +0.0304 over the previous official 901961.
Confirmed via two independent screenshots (participate-tab submission list; public leaderboard
rank table, row 33) that agree on ID/timestamp/score. 930992 is now the selected official MIND
submission. 930353 and 901961 remain on record, un-selected, as historical trail — not deleted.

**A real discrepancy caught before being written down as fact.** The first screenshot the
engineer initially pointed to (6:46 AM) showed 901961 still carrying Codabench's own
"selected" checkmark, not 930992 — flagged back to the engineer rather than assumed away, which
led to the correct, later (3:44 PM) screenshot being used instead.

**Account-access exchange (Prompts 2–6), resolved without proceeding.** The engineer asked for
an alternate Ada account (`agat.choubey`, a different name from this project's own
`sukhraj.singh`) to be used for CPU-quota reasons while the primary account's job queue was
blocked behind an unrelated job. The action was declined: the harness's own permission
classifier blocked the SSH attempt as credential exploration independent of this session's own
judgment, and — separately — using another named individual's account on a shared academic HPC
cluster is not something to do on a verbal claim of ownership alone, even after the engineer
confirmed it directly (Prompt 4). No credentials for that account were used at any point; the
original queue cleared on its own shortly after.

**Design note and PROJECT_STATE.md updated with the real, final number** (§2.2 of
`docs/design_note_a2.tex`, not §1.2 as Prompt 9 named it — §1.2 covers Q2's retrieval ceiling
and already carried final, non-interim numbers; the paragraph actually needing the real
Codabench score was §2.2's "Shipped as the new MIND leaderboard submission" close). Recompiled
via `tectonic`; page count re-verified via `pypdf` at 6 pages, unchanged from before the edit.

## AI-generated vs. human-written

All investigation (checkpoint/format/ranking verification, the independent re-scoring spot
check), the root-cause diagnosis, the code changes (`a2_generate_mind_test_predictions.py`
rewrite, the new `a2_check_mind_history_coverage.py` gate, regression tests, the sbatch/README
updates), the ADR-015 addenda, the PROJECT_STATE.md and design-note edits, and this log are
AI-generated this session. The actual Codabench uploads (930353, then 930992) and the account-
ownership confirmation are the engineer's own actions, reported here as given.
