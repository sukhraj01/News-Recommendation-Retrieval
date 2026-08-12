# AI Usage Log — 2026-08-12 — EB-NeRD Codabench Submission, Part 0 Investigation

Per CLAUDE.md's "Prompt & Session Logging" section. Written live as the session
proceeds, not reconstructed afterward.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: EB-NeRD Codabench submission — format investigation on
Kaggle (files too large for local), build/validate locally against known
labels, run the real test-set pipeline on Kaggle, submit.

Foundation: BM25 and embeddings both already run and are benchmarked
against ebnerd_small/demo (ADR-005/006/008), Q4's ranking harness
(ADR-007) already scores both methods on both. What's missing is the
submission side, and unlike MIND, the real files here are constraints, not
options — ebnerd_testset.zip (1.5GB), ebnerd_large.zip (3.0GB), and the
example submission predictions_large_random.zip (220MB) cannot be
downloaded locally at all. This isn't a "benchmark then decide" situation
like MINDlarge was — go straight to Kaggle for anything touching these
three files, per CLAUDE.md's Resource Availability and Memory Estimation
clauses. Only ebnerd_demo and ebnerd_small are local.

Read first: CLAUDE.md, PROJECT_STATE.md, ARCHITECTURE.md, ADR-005, ADR-006,
ADR-007, ADR-008, and the EB-NeRD paper's Appendix A (Tables 6-8) for the
authoritative schema — already in this conversation's context if picking
up from here, otherwise re-fetch from the paper.

Part 0 — Format investigation, ON KAGGLE, before any local code:
- Download and unzip predictions_large_random.zip there. This is a real,
  confirmed-working submission — inspect its literal structure (file
  layout inside the zip, column names/format) directly. Treat it as the
  ground truth, don't guess from Table 6/7's schema docs.
- Clone https://github.com/jppol-ai/ebnerd-benchmark (small, fine to clone
  anywhere) and look specifically for the CodaBench local-server setup
  guide and any submission-format or scoring code. Skip everything else in
  that repo — the NRMS/LSTUR/NPA/NAML model training code, the `ebrec`
  package, TensorFlow setup — none of that applies here. This project is
  content-based retrieval (Q2/Q3), not a trained neural ranker; don't let
  that repo's scope creep into this one.
- Also on Kaggle: download ebnerd_testset.zip, inspect behaviors.parquet
  directly against the paper's Table 7 — confirm the `is_beyond_accuracy`
  column exists as described, and that Article ID / Next Readtime / Next
  Scroll / Clicked Article IDs are actually absent from test rows.
- Download articles_large_only.zip (140MB) and check whether
  ebnerd_testset's in-view article IDs are covered by ebnerd_small/demo's
  existing article tables or require this larger catalog — the naming
  strongly suggests the latter (test-period articles are newer than what
  demo/small's article tables contain), but verify rather than assume.

Part 1 — Build & validate the converter locally, same discipline as MIND:
- New module (e.g. src/submission/ebnerd_format.py) matching whatever
  Part 0 found, reusing the existing Scorer interface — no new scoring
  logic.
- Generate predictions against ebnerd_small's validation split (known
  labels) using both BM25 and embeddings.
- Cross-check against whatever scoring reference Part 0 turned up (a
  CodaBench scoring script if one exists in the starter repo, or at
  minimum manually verify AUC/MRR/nDCG@5/nDCG@10 match what
  src/evaluation/ranking_metrics.py already reports for the same
  predictions — same cross-check MIND's Part 3 did).
- Handle the is_beyond_accuracy branch explicitly even at this stage if
  ebnerd_small's format hints at it, so the logic isn't first-tested on
  the real blind file.

Part 2 — Real test-set run, ON KAGGLE (do not attempt locally):
- Build the correct article corpus for the test period (articles_large_only
  once Part 0 confirms it's needed) and the BM25/embedding indexes over it.
- Generate predictions for ebnerd_testset — branch correctly between
  ordinary ranking impressions and the beyond-accuracy-flagged ones (the
  latter share one fixed 250-article pool across ~200,000 impressions and
  aren't scored for click-accuracy the same way).
- Download only the final packaged prediction file back to the local repo
  — not the raw test set or article corpus.

Part 3 — Submit:
- Validate the packaged file's structure against predictions_large_random's
  confirmed format before uploading anything.
- Submit to https://www.codabench.org/competitions/2469/, screenshot for
  Q6.

Also, for the design note (Q9): explicitly confirm and state that
total_inviews/total_pageviews/total_read-time are never used anywhere in
this pipeline — the RecSys organizers flagged these as a known leakage
vector and asked participants to report with/without; since this project
never touches them, that's trivially satisfied, but say so outright rather
than leaving it implicit.

Update PROJECT_STATE.md and the AI usage log; commit each part as its own
checkpoint, same discipline as MIND.
```

AI-generated: full session so far (investigation + code).

### Resource constraint identified before any implementation

Checked this environment for Kaggle execution capability before starting
Part 0: no `kaggle` CLI installed, no `~/.kaggle` credentials, no
browser/notebook access. This is a real instance of CLAUDE.md's Resource
Availability clause — Parts 0 and 2 explicitly require Kaggle execution
(unzipping/inspecting `predictions_large_random.zip`, `ebnerd_testset.zip`,
`articles_large_only.zip`; later the real test-set run), which this session
cannot perform directly, the same way last session's MIND Codabench upload
needed the engineer's own login. Stopped and asked rather than silently
guessing the submission format from the paper's schema tables alone (which
the prompt itself explicitly warned against).

## Prompt 2 (AskUserQuestion)

Asked how to handle the Kaggle-only steps: (a) prep exact notebook cells
for the engineer to run and paste results back, (b) configure a Kaggle
API token in this environment so Claude Code can drive it directly, or (c)
skip Kaggle verification and build the converter from the paper's schema
docs alone.

Engineer selected: **(a) — prep, engineer runs Kaggle steps and relays
output.**

### Work completed this session (all locally executable, no Kaggle needed)

- Extracted the EB-NeRD paper's Appendix A (Tables 6-8) directly from
  `data/ebnerd_paper.pdf` via `pypdf` (installed ad hoc into the poetry
  venv, not a pinned dependency — same one-off pattern last session used
  for the MIND paper). Confirms the *documented* schema: `behaviors.parquet`
  test-split rows drop Article ID / Next read-time / Next scroll
  percentage / Clicked article IDs and add `is_beyond_accuracy` across
  200,000 samples — matches the prompt's framing, but per the prompt's own
  caution this is still schema *documentation*, not the real submission
  file, so it doesn't stand in for actually inspecting
  `predictions_large_random.zip`.
- Cloned `jppol-ai/ebnerd-benchmark` — first attempt via `git clone` timed
  out twice over a slow connection; switched to the GitHub REST API
  (`git/trees?recursive=1` + `raw.githubusercontent.com`) to pull only the
  relevant files instead of the full repo (which also carries ~1MB+ of
  NRMS/LSTUR/notebook/image content this project doesn't need). Read
  `codabench/README.md` and the root `README.md` directly: confirmed the
  `codabench/` folder is **server-side compute-worker infrastructure**
  (Docker setup for running a CodaBench scoring backend), not a submission
  format spec or client-side scoring script — so this repo does not
  contain the ground truth Part 0 is looking for. This is a real, useful
  negative result: it confirms inspecting `predictions_large_random.zip`
  directly on Kaggle is the *only* way to ground-truth the format, not one
  option among several.
- Tried `WebFetch` against the CodaBench competition page
  (codabench.org/competitions/2469) to check whether the Submission
  Guidelines tab documents the format in prose — returned only the SPA
  shell (React app, tab content loads client-side), confirms this route
  doesn't work either without an actual browser.
- Computed the local `ebnerd_demo` + `ebnerd_small` article ID union
  directly from the already-built feature store
  (`data/processed/ebnerd/{demo,small}/articles.parquet`): 21,700 unique
  raw article IDs, published_time range up to 2023-07-11. Wrote this out
  as `notebooks/ebnerd_small_demo_article_ids.csv` so the Kaggle-side
  investigation can check test-period article coverage against it without
  needing to re-derive it there.
- Wrote `notebooks/ebnerd_part0_kaggle_investigation.py` — a
  paste-into-Kaggle-cells script (auto-discovers input files by filename
  under `/kaggle/input`, so it doesn't depend on knowing the engineer's
  exact dataset slug) covering all four remaining Part 0 checks:
  `predictions_large_random.zip`'s literal file layout and line format;
  `ebnerd_testset.zip`'s real `behaviors.parquet` columns checked against
  Table 7 (`is_beyond_accuracy` presence/value_counts, the four expected-
  absent columns, and the beyond-accuracy rows' fixed-pool structure); and
  `articles_large_only.zip`'s coverage of the test set's in-view article
  IDs, cross-checked against both `articles_large_only` itself and the
  local demo+small CSV.
- Committed a small housekeeping fix separately first: `CLAUDE.md`'s
  Memory Estimation clause and its matching `PROJECT_STATE.md` risk-table
  row were already fully written in the working tree but never committed
  at the end of the prior session — verified they were complete,
  self-contained, and unrelated to a work-in-progress before committing
  them as their own checkpoint (`7423a51`) ahead of this session's own
  changes.

### Human vs. AI split, this session

AI-generated: all investigation (Kaggle-capability check, PDF extraction,
GitHub API repo inspection, WebFetch attempt), the local article-ID
coverage computation, `notebooks/ebnerd_part0_kaggle_investigation.py`,
`notebooks/ebnerd_small_demo_article_ids.csv`, this log, and the
`PROJECT_STATE.md` update. Human-written: none this session.
Human-in-the-loop: the AskUserQuestion choice on how to handle Kaggle
execution — the one decision this session that was genuinely the
engineer's to make, not inferable from the code or docs.
