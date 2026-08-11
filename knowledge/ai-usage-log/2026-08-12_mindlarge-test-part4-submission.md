# AI Usage Log — 2026-08-12 — Part 4: MINDlarge_test Predictions + Submission Prep

Per CLAUDE.md's "Prompt & Session Logging" section. Written live as the session
proceeds, not reconstructed afterward.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: Part 4 — Generate MINDlarge_test predictions and submit.

Foundation: Parts 1-3 are complete, committed, and validated end-to-end —
MINDlarge is built, both BM25 and embeddings are benchmarked (stay local,
no Kaggle needed), and the submission-format converter is cross-checked
against the official evaluate.py on MINDlarge_dev's known labels
(embeddings: AUC 0.6335 vs. BM25's 0.5699). Git history is now caught up
and checkpointed through commit 6f940d0.

This is execution on an already-validated pipeline, not discovery — no new
design decisions should be needed here.

1. Generate MINDlarge_test predictions using embeddings (won clearly on
   dev) — run BM25 too only if compute time allows; if not, embeddings
   alone is a defensible, already-justified choice.
2. Long-running (~1-2hrs, ~6x dev's impression count) — run as a
   background/detached process, don't hold the session open polling it.
3. Once complete, validate row count matches the real MINDlarge_test
   candidate count before uploading anything — same class of cheap check
   Part 3 already used to catch silent truncation.
4. Manual: log into Codabench, upload prediction.txt, screenshot the
   leaderboard result for Q6.
5. Update PROJECT_STATE.md and the AI usage log once submission confirms,
   then commit this as its own self-contained checkpoint — don't let it
   accumulate uncommitted like the last stretch did.

Do not re-run or re-verify Parts 1-3 — check PROJECT_STATE.md's session
notes first if unsure what's already done.
```

AI-generated: full session so far (investigation + code).

### Finding before implementation: Part 4 was not actually execution-only

Before generating predictions, read `PROJECT_STATE.md`'s session notes and
inspected `scripts/generate_mind_predictions.py` and the actual processed
MINDlarge test tree. Found a real gap the prompt's framing didn't
anticipate: `data/processed/mind/large/test/` only ever had
`articles.parquet` and `candidates.parquet` — no `user_history.parquet`.

Root cause, traced in `src/datasets/mind.py`:
`parse_mind_test_candidates` calls the same `_parse_behaviors_tsv` helper
train/dev use (which always computes `user_history` regardless of
`has_labels`), but discarded the result via `candidates, _user_history =
...`. `src/pipeline/orchestrator.py::build_mind_test` never wrote a
`user_history.parquet` for the test split. A unit test
(`test_candidates_never_produces_an_impressions_table`) even asserted
`test_result.keys() == {"articles", "candidates"}` as if this were
intentional.

This blocks Part 4 outright: `scripts/generate_mind_predictions.py` builds
each user's query from `user_history.parquet`, which never existed for
`test`. The prior session's "no `clicked` column, never merged into
`impressions`" framing (a real, correct design constraint from ADR-002)
appears to have been conflated with "test doesn't need history" — but
query construction (BM25 or embedding) needs history independently of
whether labels exist. This was never caught earlier because Part 4 was
explicitly deferred every prior session, so this code path was never
actually exercised end-to-end.

Fixed (not silently worked around): `parse_mind_test_candidates` now
returns `user_history` alongside `articles`/`candidates`;
`build_mind_test` writes it via the same `_write_table`/`USER_HISTORY_SCHEMA`
convention `build_mind_split` already uses. Updated the unit test's
assertion set and the schema-conformance integration test to match, then
re-ran `build_mind_test` against the real `MINDlarge_test.zip` to
regenerate the processed tree with the fix in place. Full fast test suite
re-verified clean after the change (153 passed, up from 152). Committed
separately (`65f6bfc`) before starting the long-running generation job, per
this session's own instruction not to let work accumulate uncommitted.

### Second failure: a candidate id genuinely absent from the corpus

First real `generate_mind_predictions.py --split test --method embed` run
crashed ~430s in with `KeyError: 'mind:N89741'`, inside the shared
`Scorer.score()` lookup (`src/retrieval/score.py`) both `BM25Scorer` and
`EmbeddingScorer` use. Investigated against the raw zip directly rather
than assuming a parser bug: `MINDlarge_test/behaviors.tsv` references
`N89741` as a candidate in 32 of 2,370,727 impressions, but that id is
genuinely absent from `MINDlarge_test/news.tsv` itself. Checked train/dev
the same way — zero such gaps in either — so this is a one-article quirk
isolated to the real MINDlarge_test raw files, not something our own
pipeline introduced.

This is a different situation from ADR-005/008's existing cold-start
handling (no *query* → every candidate scores an explicit tie); here the
*candidate* itself has no content to score against. Fixed with a shared
`_lookup_scores` helper in `score.py`: an id missing from `id_to_col`
scores `-inf`, so it's deterministically ranked last rather than crashing
or landing arbitrarily among candidates we do have signal for. Kept the
existing fast vectorized path for the common case (all ids known) and only
falls back to a per-item loop for an impression that actually hits a
missing id. Added unit tests for both scorers; full fast suite
re-verified clean (155 passed, up from 153). Committed separately
(`a120c45`).

Restarted the generation job (embedding cache from the crashed first
attempt was still on disk, so the ~430s encode step was skipped on
restart). Ran to completion: 2,370,727 prediction lines in ~6,946s.

## Prompt 2

```
its been some time just check progress
```

AI-generated: status check only (ran `ps`/log inspection, no code
changes), reported line count (758,033/2,370,727 at that point) and ETA.

## Prompt 3

```
its been some time just check progress
```

AI-generated: status check — this time the job had actually finished
(2,370,727/2,370,727 lines, matching the raw zip's real impression count
exactly). Validated before treating it as upload-ready: every line parses
as a valid rank permutation (0 malformed out of 2,370,727); the 32
`N89741`-affected impressions individually spot-checked (article lands at
rank 138/138, i.e. strictly last, in the one inspected case, and strictly
last in all 32 when checked in aggregate). Packaged as
`submissions/mind_large_test_embed/prediction.zip` — `prediction.txt`
zipped at the archive root (not nested in a folder), matching
`evaluation/official/evaluate.py`'s `submit_dir/prediction.txt` layout,
which is the standard Codabench/CodaLab upload convention this project
hadn't needed to apply until now. Updated `PROJECT_STATE.md` (Current
Phase/Objective, Component Status, Deliverables Checklist, Next Actions,
new Session Notes entry for 2026-08-12) and this log to reflect the
session's real end state, then committed both as their own checkpoint.

### Human vs. AI split, this session

AI-generated: all investigation (reading `PROJECT_STATE.md`/code/raw zip
contents), both root-cause diagnoses, all code changes
(`src/datasets/mind.py`, `src/pipeline/orchestrator.py`,
`src/retrieval/score.py`, associated tests), running the pipeline/tests,
generating and validating the prediction file, packaging the zip, and all
documentation updates (this log, `PROJECT_STATE.md`, commit messages).
Human-written: none this session. Human-in-the-loop: the two "check
progress" prompts above, and the still-outstanding manual Codabench
upload + Q6 design note, which remain the engineer's own steps.
