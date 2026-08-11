# AI Usage Log — 2026-08-11 — MINDlarge Build, Benchmark, Codabench Submission Prep

Per CLAUDE.md's "Prompt & Session Logging" section. Written live as the session
proceeds, not reconstructed afterward.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: MIND Codabench submission — build MINDlarge, benchmark at
real scale, convert scores to the official rank format, validate against
ground truth, then submit.

Foundation: BM25 (ADR-005/006) and embeddings (ADR-008) are implemented,
benchmarked, and cross-compared via the Q4 harness (ADR-007) — but only at
MINDsmall-dev scale. MINDlarge_train/dev/test.zip are downloaded locally.
scripts/official/mind_evaluate.py is the actual script Codabench runs to
score submissions — format is `impression_id [rank_1,...,rank_N]` per line
(ranks, not scores, in the impression's original candidate order), not the
CSV format the README originally guessed. MINDlarge_dev has labels;
MINDlarge_test is blind.

Read first: CLAUDE.md, PROJECT_STATE.md, ARCHITECTURE.md, ADR-005, ADR-006,
ADR-007, ADR-008, scripts/official/mind_evaluate.py, ACL2020_MIND.pdf's
MINDlarge statistics table, src/pipeline/orchestrator.py's
`include_mind_large` flag.

Work through Parts 1-3 in order before touching Part 4 — do not generate a
real test submission until Part 3's validation passes.

Part 1 — Build & sanity-check MINDlarge
- Run the pipeline with `include_mind_large=True` to build the MINDlarge
  feature store (train/dev/test) from the already-downloaded zips. This
  code path exists but has never actually been exercised — treat the first
  run as a real test.
- Compare resulting article/user/impression counts against MINDlarge's
  published statistics in ACL2020_MIND.pdf as a parse-correctness check,
  same as you did for MINDsmall/ebnerd_small row counts.
- Confirm raw zips and any large intermediate files stay out of git per
  Q8/.gitignore.

Part 2 — Benchmark before trusting anything at this scale
- Before running full BM25 or embeddings over the MINDlarge corpus,
  benchmark both on a small user sample (500-1000) at MINDlarge's real
  corpus size (~161K articles vs. MINDsmall's 42K) — same discipline as the
  original ADR-006 benchmark. Project full-run time/memory from the
  sample, don't guess.
- BM25 sparse weight-matrix: check memory footprint at this vocab/corpus
  size — this is ADR-006's own flagged revisit trigger, now in scope.
- MiniLM embedding encoding: benchmark local CPU throughput over ~161K
  articles. This is the step most likely to actually need GPU.
- Decision point: if either step's projected full run is unreasonable
  locally (name a concrete threshold, e.g. >2hrs or exceeds available RAM),
  move that specific step to a Kaggle GPU notebook per CLAUDE.md's Resource
  Availability clause. Document the decision with the actual measured
  numbers — addendum to ADR-006/ADR-008, not a new ADR unless the fix is
  structurally different.

Part 3 — Official-format converter, validated against ground truth
- New module (e.g. src/submission/mind_format.py): given per-impression
  candidate scores from the existing Scorer interface, emit
  `impression_id [rank_1...rank_N]` lines with candidates in the exact
  order they appear in the source impressions file — this ordering must
  survive the whole pipeline or the truth-file join breaks silently.
- Generate MINDlarge_dev predictions (BM25 and embeddings both — dev has
  labels, so this is your test bed).
- Build the res/ref folder structure mind_evaluate.py expects, run it
  locally, and confirm its AUC/MRR/nDCG output is directionally consistent
  with what src/evaluation/ranking_metrics.py already reports for the same
  predictions. Any disagreement means the converter has a bug — fix it
  before going near the blind test set.

Part 4 — Real submission (only after Part 3 passes)
- Generate MINDlarge_test predictions (blind — no local scoring possible)
  using whichever method performed better on MINDlarge_dev, or both.
- Submit to https://www.codabench.org/competitions/13967/, capture the
  leaderboard screenshot for Q6.

Deliverable: MINDlarge feature store built and count-verified; benchmark
numbers for BM25 and embeddings at MINDlarge scale, with a documented
local-vs-Kaggle decision if the threshold was crossed; mind_format.py
implemented and tested; dev-set cross-check against the official script
passing; MINDlarge_test submission made and screenshotted; PROJECT_STATE.md
and the AI usage log updated.

Stay scoped to MIND this session — EB-NeRD's submission format is separate,
still unverified, and not part of this work.
```

## Notes on deviations from the prompt as work proceeded

- `scripts/official/mind_evaluate.py` does not exist at that path — the
  actual file is `evaluation/official/evaluate.py`. Read directly; format
  matches the prompt's description.
- `ACL2020_MIND.pdf` is not present anywhere in the repo or the wider
  filesystem. Fetched the real paper (Wu et al. 2020, ACL Anthology
  2020.acl-main.331) via WebFetch instead and extracted Table 2 with
  `pypdf` (poppler/pdftoppm unavailable, so a scratch venv + `pypdf` was
  used to extract text rather than render pages as images) — same
  "official documentation" evidence tier, different citable source,
  substitution flagged to the engineer explicitly rather than silently
  treated as equivalent.

## Prompt 2

```
hpw much time shiudl i expect
```

## Prompt 3

```
bro machine did not sleep it restarted
```

## Prompt 4

```
its been more than 5 minutes
```

## Prompt 5

```
its been 15 minutes
```

## AI-generated vs. human-written/edited

All code changes, benchmark scripts, and documentation this session are
AI-generated (Claude Code): `src/datasets/mind.py`'s three explode-related
fixes, `src/pipeline/validators.py`'s `_is_null` fix, `src/submission/`
(new module + tests), `scripts/generate_mind_predictions.py`,
`tests/integration/test_schema_conformance.py`'s MINDlarge block,
`src/utils/config.py`'s `MIND_LARGE_*` constants, ADR-006/ADR-008 addenda,
and all PROJECT_STATE.md/ARCHITECTURE.md updates for this session. Human
input this session was prompts only (above) plus real-time operational
context (the machine-restart correction in Prompt 3, which changed how the
subsequent build-retry investigation was interpreted) — no code was
hand-written or hand-edited by the engineer this session.

## Session summary (for future reference)

Parts 1-3 of the MIND Codabench submission objective completed and
validated: MINDlarge built (after fixing three real scaling bugs found by
profiling, not guessing), BM25/embeddings benchmarked at real MINDlarge
scale (both stay local), and the official-format converter validated
end-to-end against the real Codabench scorer on MINDlarge_dev for both
methods (AUC/nDCG match almost exactly; the one MRR disagreement is a
verified, fully-explained metric-definition difference, not a bug).
Part 4 (blind MINDlarge_test predictions + actual Codabench upload) not
started this session — full detail in PROJECT_STATE.md's 2026-08-11
Session Notes.
