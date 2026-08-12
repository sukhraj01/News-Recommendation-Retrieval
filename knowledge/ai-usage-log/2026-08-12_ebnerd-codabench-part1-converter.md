# AI Usage Log — 2026-08-12 — EB-NeRD Codabench Submission, Part 1: Converter Build + Validation

Per CLAUDE.md's "Prompt & Session Logging" section. Written live as the session
proceeds, not reconstructed afterward.

Tool: Claude Code (Sonnet 5).

---

## Prompt 1 (session start)

```
Current Objective: Part 1 — build src/submission/ebnerd_format.py, validate
locally against ebnerd_small's known labels.

Part 0 is fully resolved and verified on Kaggle, all facts confirmed
directly against real files, not guessed:

- Submission format: predictions.txt, one line per impression,
  `impression_id [rank_1,...,rank_N]` — a permutation matching that
  impression's article_ids_inview order. Same shape as MIND's format
  (rank permutation, not scores), different column names (user_id/
  article_ids_inview vs. MIND's naming).
- Article corpus: articles_large_only.zip alone gives 100% coverage of the
  test set's in-view articles (10,451/10,451 verified) — no need to
  download ebnerd_large.zip at all.
- Beyond-accuracy rows: exactly 200,000 of the test set's 13,536,710 total
  impressions are flagged is_beyond_accuracy=True, every one with
  precisely 250 shared inview articles (a single fixed pool across all
  200,000). Confirmed these need real submitted rankings too, same format
  as every other row — predictions.txt's line count (13,536,710) is an
  exact match to the full test set, not just the 13,336,710 regular rows.
  No special-cased output format needed for these; treat all rows
  uniformly in the converter.

Build the converter by adapting mind_format.py's logic — this is a port,
not a redesign, since the output shape is the same. Generate predictions
against ebnerd_small's validation split (known labels) using both BM25 and
embeddings, cross-check against src/evaluation/ranking_metrics.py the same
way MIND's Part 3 did before trusting it. Only after that validation
passes does Part 2 (the real Kaggle test-set run, reusing
articles_large_only.zip and the same wget approach) become worth doing.

Update PROJECT_STATE.md and the AI usage log; commit as its own checkpoint
once validation passes.
```

### What was done (AI-driven, no further human prompts this session)

- Read `PROJECT_STATE.md`, `src/submission/mind_format.py`,
  `tests/unit/test_mind_format.py`, `src/datasets/ebnerd.py`,
  `src/evaluation/ranking_metrics.py`, `src/retrieval/score.py`,
  `src/utils/ids.py`, `src/utils/io.py`, `src/pipeline/orchestrator.py`,
  `evaluation/official/evaluate.py`, `scripts/generate_mind_predictions.py`,
  and `scripts/run_ranking_eval.py` to understand the exact design being
  ported and confirm the raw-zip-reread requirement (`_write_table` sorts
  `impressions` by `["impression_id", "article_id"]`, destroying
  `article_ids_inview`'s original order — same reason MIND's converter
  exists) also applies to EB-NeRD.
- Inspected the real raw `ebnerd_small.zip`'s `validation/behaviors.parquet`
  directly (`poetry run python3` one-liner) to confirm exact column names/
  dtypes (`impression_id` uint32, `user_id` uint32, `article_ids_inview`/
  `article_ids_clicked` as list columns) before writing any parsing code
  against them, rather than assuming from `ebnerd.py`'s already-processed
  schema.
- Wrote `src/submission/ebnerd_format.py` — `read_raw_impressions`,
  `ranks_for_impression`, `write_predictions`, `write_truth_file` — a
  direct structural port of `mind_format.py`, adapted for EB-NeRD's raw
  shape (parquet list columns vs. MIND's `"N3-1 N4-0"` token strings).
- Wrote `tests/unit/test_ebnerd_format.py` (7 tests) against the existing
  `tests/fixtures/ebnerd_demo_sample.zip` fixture, hand-verified against
  its known contents (validation impressions 3/4, articles 101/102/103,
  known clicks) — same structure as `test_mind_format.py`. All 7 pass.
- Wrote `scripts/generate_ebnerd_predictions.py` (port of
  `generate_mind_predictions.py`), ran it for `ebnerd_small`'s
  `validation` split with both `--method bm25` and `--method embed` —
  244,647 prediction + truth lines each, matching the split's known
  impression count.
- Cross-checked both prediction sets against the real
  `evaluation/official/evaluate.py` (confirmed generic/dataset-agnostic —
  it only parses the shared `impid [ranks]`/`impid [labels]` line shape,
  nothing MIND-specific — so it applies to EB-NeRD unchanged). Ran as a
  detached background process (~2m40s per method on 244,647 impressions)
  and compared its AUC/MRR/nDCG@5/nDCG@10 output against
  `experiments/ranking_{bm25,embed}_ebnerd_small_2026-08-10/results.json`
  (already on disk from a prior session's `run_ranking_eval.py` run).
  AUC/nDCG@5/@10 matched to 4 decimal places for both methods. Investigated
  the small MRR gap rather than assuming it was the same MIND finding
  unchanged: recomputed both the official (sum-over-all-clicks) and
  project (first-hit-only) MRR formulas directly from the generated
  `prediction.txt`/`truth.txt` files, reproduced both tools' numbers
  exactly, then traced *why* only 748 of 1,407 nominally multi-click
  validation impressions actually diverge between the two formulas —
  found that 659 of those 1,407 are a duplicate entry for the *same*
  article within `article_ids_clicked` (not two distinct clicks), leaving
  exactly 748 with genuinely distinct multi-click labels, matching the
  observed divergence count exactly.
- Updated `PROJECT_STATE.md`: header/current-objective, Component Status
  table, Deliverables Checklist, the EB-NeRD checklist items, and a new
  Session Notes entry with the full cross-check table and the
  duplicate-click-id finding.
- Ran the full test suite (`poetry run pytest -q`) before committing: 162
  passed, 1 pre-existing skip, 6 deselected `slow` tests — no regressions.

### Human vs. AI split, this session

AI-generated: `src/submission/ebnerd_format.py`,
`tests/unit/test_ebnerd_format.py`,
`scripts/generate_ebnerd_predictions.py`, the `ebnerd_small` validation
prediction/truth files under `submissions/`, the `evaluate.py` cross-check
and its interpretation (including the duplicate-clicked-id investigation),
the `PROJECT_STATE.md` update, and this log. Human-written: none this
session — the session brief itself (Prompt 1, above) already contained
Part 0's Kaggle-verified facts (relayed by the engineer from the separate
Kaggle investigation session) and the explicit instruction to port rather
than redesign, so no additional human-in-the-loop decision point arose
during Part 1's implementation.
