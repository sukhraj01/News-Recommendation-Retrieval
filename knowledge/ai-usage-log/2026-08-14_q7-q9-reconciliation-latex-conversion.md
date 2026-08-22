# AI Usage Log — 2026-08-14 — Q7-Q9 Reconciliation & LaTeX Conversion

## Prompts (verbatim, in order)

### Prompt 1

> Current Objective: We now have the real Q7-Q9 deliverable spec (pasted
> below). Reconcile the existing design note against it — this changes
> structure and surfaces one real missing experiment, not just formatting
> — then convert to LaTeX under a hard ≤4 page limit.
>
> Part II: Deliverables & Policies
> Q7. Deliverables
> 1. Code (GitHub Classroom): reproducible pipeline, model code, evaluation harness, prediction
> files, README.md with one-command reproduce. No large files — use .gitignore.
> 2. Design note (≤4 pages, Moodle): what you built, choices, observations, where it breaks at
> 10×.
> 3. Leaderboard screenshots from both Codabench competitions.
> 4
> CS4.406: Information Retrieval & Extraction Assignment 1
> 4. AI usage log: all prompts, chat history exports, marking of AI-generated vs. human-written
> code.
> Q8. Git Commit Policy
> • Commit frequently with meaningful messages.
> • No large files in Git — ignore *.zip, *.pt, *.ckpt, __pycache__/, data/.
> • No force-pushes after the deadline.
> Q9. Anti-Gaming
> • Report metrics with and without features unavailable at serving time.
> • Enforce the behaviour-window boundary — no future-click leakage. Include a test asserting this.
> Important
> Grading note: Grading is never on leaderboard rank. Your grade is based on pipeline
> correctness, system design, ablation rigour, scale analysis, and design-note clarity.
>
> Rewrite docs/design_note.md's structure around the actual requirement — what was built, choices
> made, observations, where it breaks at 10x scale — dropping the inferred Q1-Q9 framing and its
> editorial HTML comment now that the real spec is known. Keep every existing cited number; this
> is a restructure, not a rewrite of content.
> Check whether an ablation exists reporting Q4's metrics with vs. without the
> serving-time-unavailable features (total_inviews/total_pageviews/total_read-time). If it doesn't
> exist, that's a real missing experiment per Q9 — run it and cite the results. Don't paper over
> this with prose about leakage prevention alone; Q9 asks for a with/without comparison.
> State plainly in the note's limitations section that the no-future-click-leakage test (Q9)
> exists and passes for EB-NeRD but is explicitly skip-marked for MIND due to no per-click
> timestamps — don't leave this for a grader to find on their own.
> Convert to LaTeX: single-column article class, 1in margins, 11pt. Compile and check the actual
> rendered page count — confirm ≤4 pages for real, don't infer it from the markdown word count.
> Verify Q8's git policy is actually satisfied: confirm .gitignore excludes *.zip, *.pt, *.ckpt,
> __pycache__/, data/, and check no large files have already been committed in this repo's history.
> Confirm Q7's other deliverables are in place: leaderboard screenshots from both Codabench
> competitions saved and trackable, AI usage log up to date, README's one-command reproduce
> actually verified to work.

## What was AI-generated vs. human-written/edited

- **AI-generated, this session:** `scripts/run_leakage_ablation.py` (new
  ablation script) and `tests/unit/test_leakage_ablation.py` (its unit
  tests); `decisions/ADR-009-leaky-feature-ablation.md`; the full restructure
  of `docs/design_note.md`; `docs/design_note.tex` (LaTeX conversion);
  `.gitignore`'s screenshot-tracking fix (`submissions/*` / `!submissions/*/`
  / `submissions/*/*` / `!submissions/*/*.png`); the README fixes (stray
  "gt" typo at the top of the file, and the stale `make data` "Expected
  output" block, replaced with output from a real verification run); this
  log file.
- **AI-verified via real execution, not just claimed:** the leakage
  ablation was actually run against real `ebnerd_small` validation data
  (`experiments/ablation_leaky_features_ebnerd_small_2026-08-14/`); the
  full test suite (180 passed, 1 expected skip) was run after the new
  code landed; `make data` was actually re-run to check the README's
  claimed output against real output (they had diverged); `git rev-list`
  was used to confirm no large blobs exist in this repo's git history;
  `git check-ignore`/`git add -n` were used to confirm the `.gitignore`
  fix tracks exactly the six screenshot PNGs and nothing else under
  `submissions/`.
- **Human-originated:** the real Q7-Q9 assignment text (Prompt 1), every
  number cited from experiments run in prior sessions (kept unchanged
  per the prompt's explicit instruction), and the underlying engineering
  decisions in ADR-001 through ADR-008 this session builds on rather than
  redoes.
