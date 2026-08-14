# AI Usage Log — August 14, 2026: PROJECT_STATE Verification + Leaderboard Sync

Per CLAUDE.md's "Prompt & Session Logging" section — verbatim prompt record,
independent of PROJECT_STATE.md's summarized Session Notes.

## Prompts (verbatim, in order)

### Prompt 1

Four images attached (screenshots of a Codabench-style leaderboard table and
two "Submission upload" pages), followed by this text:

> Current Objective: PROJECT_STATE.md is stale — it's dated August 10, 2026
> and still shows Semantic Retrieval at 0% and Leaderboard Submission as
> "Not Started," but both have actually been completed since. Sync the doc
> to reality first, verify what's actually on disk before writing anything
> down, then close the two evaluation-harness pieces that are genuinely
> still open.
>
> Before editing anything, inspect the actual repo state — check
> experiments/ for any semantic retrieval run directories, check for
> embedding model artifacts, check git log / recent commits, and check for
> the Codabench submission files (predictions/prediction.zip) for MIND and
> EB-NeRD. Confirm semantic retrieval and both leaderboard submissions are
> actually done by looking at real files, not by trusting my say-so.
> Update PROJECT_STATE.md based on what you actually find in step 1: mark
> Semantic Retrieval and Leaderboard Submission complete in the Component
> Status table, update "Last Updated" and "Current Phase," record the real
> benchmark numbers (validation AUC, leaderboard AUC — pull exact values
> from the experiment/results files, don't approximate), and add a session
> note for whatever session did this work if one doesn't already exist.
> Build a local scoring harness in src/evaluation/ that reproduces
> evaluate.py exactly (uploaded separately — reads truth.txt/prediction.txt
> in res/ref dir layout, outputs AUC/MRR/nDCG@5/nDCG@10). Run it against our
> own validation predictions for both BM25 and semantic, both datasets.
> Confirm the AUC it reports matches the real leaderboard score before
> trusting it for anything downstream — if it doesn't match, that's a bug
> to fix, not a rounding difference to ignore.
> Implement Q4's diversity/coverage/novelty metrics in
> src/evaluation/metrics.py — currently the only mandatory eval component
> still at 0%. Ground the definitions in the EB-NeRD paper's own
> category-coverage discussion, not an invented formula.
> With the harness verified and diversity metrics in place, run the full
> BM25-vs-semantic comparison with warm/cold user slicing and bootstrap
> CIs, both datasets. Write it up as a new ADR.
>
> Do not touch retrieval code, query construction, or the BM25/embedding
> implementations in this pass — those are already benchmarked and decided
> (ADR-005, ADR-006). This pass is verification, documentation, and
> evaluation-harness only.

### Prompt 2 (AskUserQuestion response)

Two clarifying questions were asked back (see below); the engineer answered:

- "Given the repo already has semantic retrieval, the Q4
  diversity/coverage/novelty metrics, and the evaluate.py-validated harness
  all done — do you want me to skip re-implementing those and just close
  the genuinely open item (recording the two Codabench leaderboard
  submissions in PROJECT_STATE.md)?" → **"Yes, just close the leaderboard
  gap (Recommended)"**
- "Which screenshot corresponds to which Codabench competition — can you
  confirm the mapping so I record correct dataset/score pairs?" →
  **"886468/0.6195 = MIND, 888045/0.5404 = EB-NeRD (Recommended)"**

## What was AI-generated vs. human-written/edited

- **AI-generated, this session:** all investigation (file reads, greps,
  `git log`, `git status`), the discovery that the task briefing's claims
  about PROJECT_STATE.md's staleness and Semantic Retrieval/Q4 metrics
  status were factually wrong, all PROJECT_STATE.md edits (header, Component
  Status table, Deliverables Checklist, Next Actions, the new August 14
  session note), copying the four screenshot files from the engineer's
  Desktop into `submissions/{mind_large_test_embed,ebnerd_testset_embed}/`,
  and this log file itself.
- **Human-originated, not AI-generated:** the task briefing's prose (Prompt
  1, including its factual errors), the four screenshots and the underlying
  Codabench uploads/leaderboard results they document (the engineer's own
  account, outside Claude Code's reach — see CLAUDE.md's Resource
  Availability clause), and the two clarifying-question answers.
- **Nothing was re-implemented or overwritten this session** — the
  briefing's requests to build a scoring harness, implement diversity
  metrics, and run a fresh BM25-vs-semantic comparison were all confirmed,
  by inspecting the actual files, to already exist and already be complete
  from prior sessions (see `src/evaluation/ranking_metrics.py`,
  `evaluation/official/evaluate.py`, ADR-007, ADR-008), so none of that
  code was touched.
