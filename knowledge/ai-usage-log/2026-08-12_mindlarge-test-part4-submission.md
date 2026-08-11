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
re-verified clean after the change.
</content>
