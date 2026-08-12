# AI Usage Log — 2026-08-12 — EB-NeRD Codabench Submission, Part 2: Real Test-Set Run Prep

Per CLAUDE.md's "Prompt & Session Logging" section. Written live as the session
proceeds, not reconstructed afterward.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: Part 2 — real test-set run against ebnerd_testset.zip on
Kaggle, generate the final submission, upload.

Foundation: Part 1's converter (src/submission/ebnerd_format.py) is built
and validated end-to-end against ebnerd_small's known labels — AUC/nDCG
match ranking_metrics.py to 4 decimal places for both BM25 and embeddings.
Format, article corpus, and beyond-accuracy handling are all confirmed
from Part 0, nothing here should require new discovery.

This session needs GPU (unlike Part 0's investigation) — the embedding
encoding step over articles_large_only's 125,541 articles is the
GPU-beneficial part. Turn the accelerator on this time.

1. In one continuous Edit session: wget ebnerd_testset.zip and
   articles_large_only.zip into /kaggle/working/ (same URLs as before),
   unzip.
2. Before running the full 13,536,710-impression job, benchmark on a small
   sample first — this is ~5.7x MINDlarge_test's scale, new territory even
   though the logic is validated. Same discipline as every large-scale run
   in this project: measure, project, then commit to the full run.
3. Build the BM25 index and embeddings over articles_large_only's corpus.
4. Generate predictions for all of ebnerd_testset/test/behaviors.parquet
   using whichever method performed better on ebnerd_small's validation
   split (or both, if compute allows) — same uniform treatment for
   beyond-accuracy rows as Part 1 established, no special-casing needed.
5. Package to match the real confirmed format exactly: a zip containing a
   single predictions.txt at the root (that's what
   predictions_large_random.zip's structure showed — one entry, no nested
   folders).
6. Validate the output line count is exactly 13,536,710 before downloading
   anything back or uploading.
7. Download only the final packaged prediction zip back to the local
   repo — not the raw test set or article corpus.

Manual, your end: upload to https://www.codabench.org/competitions/2469/,
screenshot the leaderboard result for Q6.

Update PROJECT_STATE.md and the AI usage log, commit as its own checkpoint
once the submission is confirmed.
```

### What was done (AI-driven so far this session)

- Read `PROJECT_STATE.md`'s current state/session notes,
  `notebooks/ebnerd_part0_kaggle_investigation.py` (the established
  paste-into-Kaggle-cells pattern from Part 0), `src/submission/
  ebnerd_format.py`, `src/datasets/ebnerd.py`, `src/retrieval/{index,
  embed,query,score,tokenize}.py`, `src/utils/{ids,io,config}.py`,
  `scripts/generate_ebnerd_predictions.py`/`generate_mind_predictions.py`,
  and the two prior EB-NeRD Codabench session logs, to confirm exactly
  what Part 0/Part 1 already established (real format, 13,536,710 total
  test impressions = 13,336,710 regular + 200,000 beyond-accuracy,
  articles_large_only.zip's 100% in-view coverage) versus what's still
  unconfirmed for the test split specifically (whether `test/
  history.parquet` ships in `ebnerd_testset.zip` — needed for query
  construction, never explicitly checked in Part 0's relayed output).
- Pulled real benchmark numbers to build this session's time/memory
  projections from measured evidence rather than guesses, per CLAUDE.md's
  Evidence Hierarchy: ADR-008's MINDlarge-dev addendum (72,023 articles:
  264.4s full encode+cache, 110.6MB vectors) and the MINDlarge_test Part 4
  session's real generation run (2,370,727 lines in 6,946s = 2.93ms/
  impression, on this project's local Mac, CPU-bound scoring loop).
  Confirmed via `experiments/ranking_{bm25,embed}_ebnerd_small_2026-08-10/
  results.json` that embeddings win on ebnerd_small-validation (AUC 0.5430
  vs. BM25's 0.5288, non-overlapping CIs) — same direction as MINDlarge-dev
  — so embeddings is this session's primary/default method, per the
  prompt's "whichever performed better" instruction.
- Confirmed no git remote is configured locally (`git remote -v` empty),
  so a Kaggle-side `git clone` of this repo isn't available; the project's
  own validated `src/` modules have to reach Kaggle another way to avoid
  reimplementing (and risking silent drift from) the already-tested
  scoring/converter code path.

### A real, measured scale problem — found before it cost any Kaggle time

Before writing the notebook script itself, checked whether the existing
`write_predictions`/`read_raw_impressions` path would actually survive
13,536,710 impressions, rather than assuming "the logic is validated" (the
prompt's own framing) meant "the logic scales." `read_raw_impressions`
materializes the *entire* split as a `list[dict]` before writing a single
output line — invisible at MINDlarge_test's 2,370,727-impression scale
(Part 4), genuinely untested at 5.7x that.

Measured directly rather than reasoning abstractly, per CLAUDE.md's Memory
Estimation clause: built synthetic rows shaped exactly like the real
output (~9-15 candidate ids/impression, matching EB-NeRD's real
per-impression median), measured real RSS growth via
`resource.getrusage`. Result: the `rows` list alone projects to ~16.3GB at
13,536,710 rows; the `behaviors` DataFrame it's built from (stays resident
throughout) adds ~8.5GB more — and pandas' own `memory_usage(deep=True)`
badly undercounts this (~99MB reported vs. ~321MB real RSS on the same
500K-row sample), since it doesn't recurse into the boxed Python ints
inside object-dtype list columns. Combined, ~25GB of peak memory before
any scoring work even starts — a real risk regardless of Kaggle's exact
session RAM ceiling.

Fixed at the root: added `iter_raw_impressions` (a generator) to
`src/submission/ebnerd_format.py` as the real implementation;
`read_raw_impressions` is now `list(iter_raw_impressions(...))`, kept as
an unchanged-contract wrapper for existing callers/tests.
`write_predictions`/`write_truth_file` consume the generator directly.
Added an optional `columns=` parameter to `src/utils/io.py::read_zip_parquet`
(backward-compatible) so only the columns actually needed get parsed.
Added two new tests (`test_iter_raw_impressions_is_lazy_and_matches_
read_raw_impressions`, `test_iter_raw_impressions_unlabeled_matches_
read_raw_impressions`) confirming byte-identical output between the old
and new paths on both the labeled and unlabeled/test-shaped code paths.
Full suite: 164 passed (up from 162), 1 pre-existing skip, no
regressions.

Beyond the unit tests (which use the small `ebnerd_demo_sample.zip`
fixture), ran the actual blind-test (`has_labels=False`) code path
end-to-end locally against the real `ebnerd_small.zip` validation split
(244,647 real impressions) — 0 malformed lines, correct
`predictions.txt`-at-zip-root packaging. This is the first time EB-NeRD's
unlabeled/test path has been exercised against real (not fixture) data.

### Bundle + notebook

Built `notebooks/ebnerd_part2_src_bundle.zip` — the minimal `src/`
subtree Part 2 needs (`retrieval/`, `submission/ebnerd_format.py`,
`evaluation/{ranking_metrics,bootstrap}.py`, `datasets/ebnerd.py`,
`utils/`; ~24 files/46KB), for the engineer to upload as a private Kaggle
Dataset. Chosen over re-deriving the logic inline in the notebook (risks
silent drift from the validated/tested version) or `git clone`ing this
repo on Kaggle (no remote configured locally — `git remote -v` is empty).
Import-verified in an isolated `sys.path` both before and after the
streaming fix.

Wrote `notebooks/ebnerd_part2_kaggle_test_run.py` — paste-into-Kaggle-cells
script, same pattern as `ebnerd_part0_kaggle_investigation.py`. Full
detail in PROJECT_STATE.md's new Session Notes entry; briefly: wget +
unzip, a discovery cell that re-confirms real facts directly against the
live files rather than trusting Part 0's relayed numbers unchanged
(critically: whether `test/history.parquet` exists at all — never
explicitly checked before, and every query-construction path in this
project depends on it), corpus index build over the full
`articles_large_only` corpus with `device="cuda"`, a real
benchmark-then-decide gate (samples the actual file on the actual Kaggle
instance via a deterministic systematic stride, projects full-run time,
does not auto-proceed), the full run gated behind `RUN_FULL_JOB = True`
with periodic progress/ETA printing, then line-count/format validation
and packaging to the confirmed `predictions.txt`-at-zip-root structure.
Defaults to embeddings only (`RUN_METHODS = ["embed"]`) — confirmed via
`experiments/ranking_embed_ebnerd_small_2026-08-10/results.json` vs. the
BM25 equivalent that embeddings win on ebnerd_small-validation (AUC 0.5430
vs. 0.5288, non-overlapping CIs), same direction as MINDlarge-dev.

Updated `PROJECT_STATE.md` (header, Component Status, Deliverables
Checklist, Next Actions, new Session Notes entry) to reflect the real
state honestly: Part 2 is *prepared*, not executed — nothing was run on
Kaggle this session, and the log is explicit that this shouldn't be read
as "Part 2 done."

### Human vs. AI split, this session

AI-generated: all investigation (reading `PROJECT_STATE.md`/prior
session logs/code), the memory-risk measurement and its interpretation,
the `src/submission/ebnerd_format.py`/`src/utils/io.py` fix and its
tests, running the full test suite and the local `ebnerd_small`
logic-check, building and import-verifying the src bundle, writing the
Kaggle notebook script, and all documentation updates (this log,
`PROJECT_STATE.md`). Human-written: none this session. Human-in-the-loop:
the session-start prompt (above), and the still-outstanding manual steps
this session's output cannot substitute for — running the notebook on
Kaggle with the GPU accelerator on, and the Codabench upload/screenshot,
both requiring the engineer's own accounts.
