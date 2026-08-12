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

### Human vs. AI split, this session (through the prep-complete commit)

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

---

## Prompt 2 (mid-Kaggle-run, real output relayed)

```
total 1.7G
-rw-r--r-- 1 root root 143M Apr  4  2024 articles_large_only.zip
-rw-r--r-- 1 root root 1.6G Mar  8  2024 ebnerd_testset.zip
ebnerd_testset.zip: /kaggle/working/ebnerd_testset.zip
articles_large_only.zip: /kaggle/working/articles_large_only.zip
src/ bundle parent dir: /kaggle/input/datasets/apollo19/ebnerd-part2-src-bundle

cuda available: True
device: Tesla T4

imports OK
total RAM: 31.3 GB
available RAM right now: 29.6 GB
CPU count: 4
================================================================================
ebnerd_testset.zip — full member listing
================================================================================
  __MACOSX/._ebnerd_testset  (220 bytes)
  ebnerd_testset/.DS_Store  (6148 bytes)
  __MACOSX/ebnerd_testset/._.DS_Store  (120 bytes)
  ebnerd_testset/articles.parquet  (150841729 bytes)
  __MACOSX/ebnerd_testset/._articles.parquet  (854 bytes)
  ebnerd_testset/test/history.parquet  (1157658806 bytes)
  __MACOSX/ebnerd_testset/test/._history.parquet  (858 bytes)
  ebnerd_testset/test/behaviors.parquet  (567769967 bytes)
  __MACOSX/ebnerd_testset/test/._behaviors.parquet  (860 bytes)

test/history.parquet present: True
[... KeyError traceback: "There is no item named 'test/behaviors.parquet'
in the archive" from pd.read_parquet(zipfile.ZipFile(TESTSET_ZIP).open
("test/behaviors.parquet"), ...) inside Cell 3 ...]
```

AI-generated: diagnosis and fix (see below). Human-written: none — this
is the raw relayed Kaggle cell output, pasted verbatim as instructed.

### What was done

Diagnosed directly from the relayed listing: the real
`ebnerd_testset.zip` wraps every member in an extra top-level
`ebnerd_testset/` directory — `ebnerd_testset/test/behaviors.parquet`,
not the flat `test/behaviors.parquet` every EB-NeRD reader in this
project (`src/datasets/ebnerd.py`, `src/submission/ebnerd_format.py`) was
written and tested against, because `ebnerd_small.zip`/`ebnerd_demo.zip`
(the only files this project had touched directly, as opposed to relayed
investigation, before now) don't do this. Confirmed the crash was
isolated to Cell 3's own diagnostic read (a raw `zipfile.ZipFile(...)
.open(...)` call, bypassing this project's own `read_zip_parquet`
utility) — `_parse_history`/`parse_ebnerd_articles`, called just below the
crash point, already went through `read_zip_parquet` and would have hit
the identical `KeyError` moments later if Cell 3 hadn't crashed first.

Fixed at the shared layer rather than patching only the notebook: `src/
utils/io.py::read_zip_member_bytes` now tries the exact member name
first (unchanged fast path) and falls back to a suffix search across the
zip's real namelist (excluding `__MACOSX/` junk, raising `KeyError` if
the match count isn't exactly 1) — fixes every downstream EB-NeRD caller
at once. Updated Cell 3's own diagnostic read to use `read_zip_parquet`
instead of raw `zipfile`, so it benefits from the same fix rather than
staying a special case. Added 5 tests (`tests/unit/test_io.py`) covering
the exact-match path, the wrapped-directory fallback, `__MACOSX`
exclusion, and both error cases (ambiguous match, no match). Beyond the
unit tests, built a zip in-memory replicating the real file's *exact*
reported listing (including the `.DS_Store`/`__MACOSX` junk entries) and
confirmed `read_zip_parquet`/`_parse_history` both read it correctly —
not just the synthetic test fixtures. Full suite: 169 passed (up from
164), no regressions. Rebuilt `notebooks/ebnerd_part2_src_bundle.zip`
with the fix, re-verified its imports in isolation. Updated
`PROJECT_STATE.md` with an addendum to the same Session Notes entry
(not a new dated entry — this is a direct continuation of the same
session/topic) documenting the real Kaggle environment facts now
confirmed (T4 GPU, ~30GB RAM, `test/history.parquet` present) alongside
the bug/fix.

### Human vs. AI split, this exchange

AI-generated: the diagnosis, the `src/utils/io.py` fix, the new tests
(including the real-listing replica check), the notebook Cell 3 fix, the
bundle rebuild, and the documentation updates. Human-written: none.
Human-in-the-loop: relaying the real Kaggle cell output verbatim (Prompt
2, above) — the only way this bug was discoverable, since it depends on
the real file's actual packaging, not anything derivable from code
review or the fixture files already in this repo.

---

## Prompt 3 (real output relayed, second crash)

```
[relayed Kaggle output — Cell 1-3 now pass cleanly (real counts:
13,536,710 test impressions, 200,000 is_beyond_accuracy, 125,541
articles, 807,677 users with history, mean 144.6 articles/user); BM25
index built cleanly (4.8s); then:
AttributeError: 'str' object has no attribute 'with_suffix'
  at embed_index = build_embedding_index(..., cache_path=embed_cache_path, ...)
```

AI-generated: diagnosis (embed_cache_path built via string concatenation,
but build_embedding_index's cache helpers call .with_suffix(), Path-only)
and fix (wrap in Path(...)). Human-written: none — relayed output again.

## Prompt 4

```
bro make change sinthe fiel we have to submit htat too dont ask me too
patch it
```

Read as: stop handing back single-line patches to paste into a live
Kaggle cell — fix the actual `.py` file completely (it's a submission
artifact itself) so re-pasting from it is the only step needed, instead
of an iterative per-crash patch cycle.

AI-generated response: rather than just applying the one-line Path fix
and calling it done, used the now-known real scale facts (807,677 users,
mean 144.6 articles/user, up to 1,530) to check whether Cell 5 — not yet
reached, but next — would actually survive. Measured directly (real
EB-NeRD article text through the project's own `tokenize()`, realistic
history-length sampling matching the real distribution): BM25's per-user
query dict projects to ~12.25GB for all 807,677 users (unweighted,
non-deduplicated token concatenation of a user's entire history, per
ADR-005) vs. embeddings' ~1.16GB (fixed 384-dim vector regardless of
history length) — a real, measured ~10.6x gap that Cell 5 would have hit
unconditionally (it built both query dicts regardless of which method(s)
were actually going to run). Restructured Cells 5-7 around a single
`RUN_METHODS` list (moved to Cell 5, gating query construction,
benchmarking, and the full run together) with BM25 opt-in rather than
default — updates this session's earlier "build both, decide later" plan
with new evidence, per CLAUDE.md's Decision Reversal principle. Also found
and removed a stale duplicate `methods = {...}` block in Cell 6 (a
leftover from before this restructuring, referencing variable names that
no longer existed — would have raised `NameError` immediately) and a
variable-shadowing landmine (`query_by_user` reused as both Cell 5's
`{method: {user: query}}` dict-of-dicts and a per-method loop variable in
Cells 6/7 — renamed the loop variable to `q_by_user`).

Before reporting this fixed, ran a full local simulation of the
restructured Cells 4-9 against real data (`ebnerd_small.zip`'s validation
split as a stand-in for the real test split) for both `RUN_METHODS`
configurations — index build, gated query construction, benchmark
sampling, the full write loop (244,647 real lines each), 0-malformed
validation, and correct packaging, all passing — specifically because the
engineer's prompt was a direct correction against reactive, unverified
patching.

### Human vs. AI split, this exchange

AI-generated: the `str`-vs-`Path` diagnosis and fix, the memory
measurement and its interpretation, the Cells 5-7 restructuring, the
stale-code/shadowing cleanup, the full local verification pass, and the
documentation updates (this log, `PROJECT_STATE.md`). Human-written: the
relayed Kaggle output (Prompt 3) and the explicit correction (Prompt 4)
that changed how this session verified its own fixes before reporting
them — a real process correction, not just a bug report, saved to
`feedback` memory for future sessions.

---

## Prompt 5

```
Current Objective: Investigate why EB-NeRD embedding scoring measured
10.536ms/impression — 3.5x slower than ADR-008's 0.99ms/query benchmark
would predict at this corpus size (125,541 vs. 42,416 articles, ~2.96x
bigger, should project to ~2.9ms not 10.5ms) — before accepting the 39.6hr
projection and building around it.

Also fix, separately and regardless of the above: the packaging cell in
ebnerd_part2_kaggle_test_run.py crashed with FileNotFoundError instead of
skipping gracefully when RUN_FULL_JOB was False and no output file
existed. Guard that step with an existence check.

Investigate: profile the actual per-impression scoring path in this run's
code — is it batching/vectorizing across impressions the same way MIND's
embedding scorer did, or has something (e.g. filtering each impression
down to its own article_ids_inview subset, or a per-row lookup) been done
in an unvectorized way at this scale? Compare directly against how
src/retrieval's embedding scorer handles MIND's equivalent step, since
that path is the one ADR-008's 0.99ms number came from.

If a real fix drops the projection to something that fits a single Kaggle
session (check actual account session/quota limits, don't assume), great
— proceed with the original plan. If profiling shows this actually is a
legitimate scaling difference (e.g. EB-NeRD's impressions have
meaningfully more candidates per impression than MIND's, which would
genuinely cost more per-impression time), document that finding and then
decide between the checkpointing and background-commit options the
decision gate already laid out — but only after confirming it's not just
an unvectorized loop.

Don't touch Cell 7 or attempt the full run again until this is resolved
either way.
```

AI-generated: the full investigation and fixes below.

### What was done

Profiled the actual scoring path directly rather than assuming either
"slower hardware" or "unvectorized code": built a synthetic
`EmbeddingIndex` at the real 125,541×384 scale locally, decomposed
`EmbeddingScorer.score()`'s cost into its three real components (matvec,
`_lookup_scores`, `rank_candidates`). Raw matvec — the exact operation
`embed_retrieve_top_k` does, the source of ADR-008's 0.99ms number —
measured 2.90ms, matching the user's own corpus-scaled projection almost
exactly; the other two components are negligible (0.001ms, 0.01ms). Under
a forced-cache-miss pattern, `EmbeddingScorer.score()` costs exactly the
matvec cost; under a cache-hit pattern (its designed use case, same query
reused) cost drops 1,036x. Confirmed: not an unvectorized-code regression,
matches MIND's own benchmarked code path exactly (same `Scorer`
interface, same operation).

Found the real cause was in Cell 6's *sampling methodology*, not the
scoring path, via two distinct bugs:
1. `sample_rows()`'s `enumerate(iter_raw_impressions(...))` + `if i %
   STRIDE == 0` filter still runs `iter_raw_impressions`' full per-row
   parsing (two `prefix_id` calls + list comprehension) for every row of
   the whole split before the filter ever gets a chance to skip it —
   confirmed directly against real `ebnerd_small` data (walking 244,647
   rows to yield 12,233 took the same wall time as parsing all 244,647).
2. Measured real EB-NeRD row-to-row user locality directly (`ebnerd_small`,
   both splits): ~52% of consecutive rows share the same `user_id` as the
   row before, mean consecutive-same-user run length ~2.08. The file is
   not randomly shuffled. The old benchmark's evenly-strided sample (jump
   135 rows) is close to the worst possible pattern for
   `EmbeddingScorer`/`BM25Scorer`'s identity-based per-user cache —
   consecutive sampled rows essentially never share a user — so it
   measured a near-worst-case "cache always misses" floor, not what
   Cell 7's real sequential access would experience.

Fixed both: added `sample_raw_impressions` to `src/submission/
ebnerd_format.py` (pandas-level row selection first, no per-row Python
work for skipped rows; factored the per-row transform into a shared
`_row_to_impression` helper so a benchmark sample and the real run can
never silently diverge). Rewrote Cell 6 to sample 10 contiguous
10,000-row chunks scattered across the file (representative, per the
original stride design's actual goal, while preserving real within-chunk
locality) plus an isolated single-call diagnostic (forces a cache miss
deliberately via a same-content-different-identity copy) reported
alongside the realistic chunked-sample cost, so the next Kaggle run gives
a decomposed answer (hardware-driven vs. methodology-driven) instead of
one opaque number. Separately, guarded Cell 9 and Cell 10 against
`FileNotFoundError`/misleading messages when `RUN_FULL_JOB` was `False`.

Added 3 new tests, ran the full suite (172 passed, up from 169, no
regressions), then verified the new Cell 6 methodology end-to-end against
real `ebnerd_small` data before reporting this fixed — sample selection
took 0.92s vs. the old method's full-file walk, and the chunked benchmark
showed a real, visible caching benefit (1.8x BM25, 3.0x embeddings at
this smaller scale) the old method couldn't show. Rebuilt the Kaggle src
bundle. Updated `PROJECT_STATE.md` with the full decomposed finding, and
was explicit that Cell 7 was not touched and the full run was not
attempted, per the prompt's explicit instruction.

### Human vs. AI split, this exchange

AI-generated: all profiling/investigation, the diagnosis (methodology bug
in Cell 6, not a scoring-path regression), the `sample_raw_impressions`
fix, the Cell 6 redesign, the Cell 9/10 guard, the new tests, the local
verification pass, and the documentation updates. Human-written: none —
the session brief (Prompt 5) contained the full investigation scope and
constraints (don't touch Cell 7, confirm before deciding between
checkpointing/background-commit) but no code or prose reused verbatim.

---

## Prompt 6 (real Cell 6 output relayed)

```
[relayed real Kaggle Cell 1-6 output: Cells 1-5 clean (807,677 users,
125,541 articles, encode 147.9s @ 849.1 articles/s, RAM settling at
13.3-13.7GB available); Cell 6:
embed: isolated cache-miss cost = 7.1980 ms/call (22 candidates) --
  compare against ADR-008's 0.99ms/query @ 42,416 docs (this corpus:
  125541 docs, 2.96x -> ~2.93ms projected)
embed: 100000 sampled impressions (locality-preserving chunks) in 340.5s
  (3.405 ms/impression, 2.1x faster than the isolated cache-miss cost
  above -> real caching benefit) -> projected full run: 12.80 hours
RUN_FULL_JOB is False -- not running.
[Cell 9/10 guards both fired correctly, no crash]
```

AI-generated: interpretation and the decision-point framing below.
Human-written: none — relayed real output.

### What was done

Interpreted the decomposed real numbers directly: isolated cache-miss
cost (7.198ms) vs. this session's own local-machine measurement (2.90ms)
for the identical operation at the identical corpus scale confirms a
real, genuine ~2.48x Kaggle-vs-local hardware/BLAS speed difference — not
a bug. The realistic chunked-sample cost (3.405ms/impression, 2.1x
caching benefit) confirms the ~52%-locality finding held on the real
13,536,710-row file. Full-run projection: 12.80h, down from the old
flawed benchmark's ~39.6h — the original 3.5x gap decomposes cleanly into
~2.5x genuine hardware slowness and just under 2x from the two
benchmark-methodology bugs already fixed.

Per the session brief two prompts back ("decide between the checkpointing
and background-commit options" once a legitimate scaling difference is
confirmed, not just a code bug) — this was now confirmed. 12.80h is
genuinely borderline against a free-tier Kaggle account's commonly-cited
~9-12h cap, and this session has no visibility into the engineer's actual
account tier or remaining weekly quota, which directly determines which
path is right. Used `AskUserQuestion` rather than guess: asked (1) the
engineer's real Kaggle tier/limits, (2) which path to build. Answers:
free tier, and build checkpointing (the more robust choice regardless of
exactly where Kaggle's real cap falls, vs. hoping a commit run happens to
have enough headroom).

### Human vs. AI split, this exchange

AI-generated: the interpretation of the relayed numbers. Human-written:
the relayed Cell 1-6 output (Prompt 6) and the two `AskUserQuestion`
answers (free tier; build checkpointing) — a real decision only the
engineer could make, correctly not guessed.

---

## Prompt 7 (implicit — proceed with checkpointing per the AskUserQuestion answers)

AI-generated: the full checkpointing design and implementation described
below, driven by the two answers above rather than a new verbatim prompt.

### What was done

Designed cross-session checkpointing carefully, since correctness bugs
here (resuming from the wrong row, corrupting a partial write, silently
duplicating or dropping rows across a session boundary) are worse than
just being slow:

- Added `iter_raw_impressions_from(zip_path, split, has_labels, start_row)`
  to `src/submission/ebnerd_format.py` — reuses the shared
  `_row_to_impression` transform (added in the Cell 6 fix), does a
  `.iloc[start_row:]` slice first so resuming costs only the remaining
  rows' parsing, not a wasted re-walk of everything already written.
- Rewrote Cell 7 around a self-imposed `MAX_RUNTIME_HOURS` (default 8.0,
  deliberately under the ~9-12h real cap) — stops the write loop cleanly
  (finishes the current line, flushes) rather than risking Kaggle killing
  the process mid-write. On start, searches `/kaggle/working` then
  `/kaggle/input` for an existing `predictions_<method>.txt`; if found,
  validates only the *last* line (the one place a partial write could
  ever land, since a clean stop always flushed a complete line first),
  discards it if malformed, computes `resume_from`, and continues via
  `iter_raw_impressions_from`. Documented the manual cross-session
  workflow (download partial file -> upload as a new Kaggle Dataset
  version -> re-attach in a fresh session -> re-run) directly in Cell 7's
  own docstring, chosen over relying on any implicit Kaggle
  session-persistence behavior this session can't verify with certainty.
- Updated Cell 8 to distinguish `INCOMPLETE` (valid, just not finished —
  now the expected, routine case) from `FAILED` (genuinely malformed).

Verified rigorously before reporting this fixed, per this session's
standing "verify before handing back" memory — this being the
highest-risk new code so far (multi-session correctness):
- 3 new unit tests (`iter_raw_impressions_from` matches full iteration
  from row 0, correctly skips already-written rows, empty past the file
  end). Full suite: 175 passed (up from 172), no regressions.
- A standalone harness running Cell 7's *exact* real logic (copied, not
  reimplemented) against the real `ebnerd_small.zip` validation split
  (244,647 real rows, real `BM25Scorer`): a full uninterrupted reference
  run, then a deliberately interrupted run (stopped at 50,000 rows, last
  line artificially truncated to simulate a mid-write kill, checkpoint
  moved to a directory simulating a fresh session's `/kaggle/input`,
  resumed to completion in a fresh working directory) — the two outputs
  are byte-for-byte identical. Confirms truncation handling,
  resume-from-row correctness, and that `ranks_for_impression`'s seeded
  tie-break (seeded by `impression_id`, not call order) is correctly
  insensitive to session boundaries.

Rebuilt and re-verified `notebooks/ebnerd_part2_src_bundle.zip` (imports
resolve in isolation, including the two new functions). Updated
`PROJECT_STATE.md` with the full decomposed finding and the checkpointing
design/verification.

### Human vs. AI split, this exchange

AI-generated: the checkpointing design, `iter_raw_impressions_from`, the
Cell 7 rewrite, the Cell 8 update, all new tests, the standalone
correctness harness and its interpretation, the bundle rebuild, and the
documentation updates. Human-written: none this exchange — driven by the
`AskUserQuestion` answers from Prompt 6's exchange.
