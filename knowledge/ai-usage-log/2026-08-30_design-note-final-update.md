# AI Usage Log — 2026-08-30 — Final Design Note Update (Pre-Submission)

## Prompts (verbatim, in order)

### Prompt 1

> Current Objective: Produce the final, submission-ready design note (≤4 pages) — this is the last deliverable before the Moodle submission. Pull all real results from the repo itself; don't carry over any numbers discussed in chat (including the 0.80 EB-NeRD target, or any specific score/ID mentioned here) unless they're actually committed in an ADR or results file — re-verify against what was actually achieved, not what was aimed for or recalled.
>
> The official requirement, verbatim, is:
>
>   Q6. Design Note (≤4 pages)
>   Write a concise design note covering:
>   • What you built and key design choices
>   • Alternatives considered and why you chose what you did
>   • Observations from experiments (lexical vs. semantic, dataset differences)
>   • Where your pipeline breaks at 10×scale
>
> Treat this as the authoritative spec for what the note must cover — not a
> paraphrase of it. Note a real discrepancy to resolve, not assume away:
> PROJECT_STATE.md describes the note as having been "restructured around the
> real Q7 spec" (What We Built / Choices / Observations / Anti-Gaming & Leakage
> / Where It Breaks at 10x / Limitations) — but the requirement above is labeled
> Q6. Check the actual assignment numbering (ask the engineer if it can't be
> resolved from repo content) and make sure the note's structure genuinely
> covers all four required bullets above, however the sections end up labeled —
> don't assume the existing structure already satisfies this without checking
> each bullet against it directly.
>
> The submission artifact is docs/design_note.tex (LaTeX, academic-paper format: article class, compiled via tectonic to docs/design_note.pdf) — that PDF is what goes to Moodle. docs/design_note.md is a parallel markdown source that has been kept in sync with it historically (see commit c9497aa, "Update design note for Candidate J's real Codabench win," as precedent for updating both together). Update both consistently; the .tex/.pdf is authoritative for page count and formatting.
>
> Check final state first: git status, git log --oneline -10, and read PROJECT_STATE.md + the relevant ADRs to get the real, final numbers, including anything that fell short of a stated target. Do not reconstruct results from conversation memory — this session has none of the prior conversation's context, which is the point. Specifically read:
> - ADR-012 (MIND Candidate J — NRMS-lite) for the MIND candidate-search line and its real Codabench result.
> - ADR-013 (EB-NeRD Candidate K — GBDT learning-to-rank) for the EB-NeRD line. Note its shape is different from MIND's: not a lettered A–J search, but one candidate (K) with a real per-impression recency bug found and fixed mid-session, three training objectives/arms (K_rank, K_cls, K_rank_nopos), and a real ebnerd_large-scale run on Ada. ADR-013 has three dated addenda — read all three for the full trail, not just the top.
> - PROJECT_STATE.md's top banner and the Deliverables Checklist section for current status of every deliverable, not just the design note.
>
> Update design_note.md and design_note.tex together, mapping content onto the four required bullets (adjust section structure if the current one doesn't cleanly cover all four — don't force a mismatch):
> - What you built / key design choices: verify this is still accurate and current (BM25 + embeddings + unified schema, per ADR-001/002/005/006/008) — likely needs no major change, but confirm rather than assume.
> - Alternatives considered: same check — confirm still accurate.
> - Observations from experiments (lexical vs. semantic, dataset differences) — this is where the results content goes. Condensed MIND candidate-search summary (A–F, H, I/I-long/I-pop, J — one line each: method + result), pointing to ADR-011/ADR-012 for full detail, not exhaustive prose. For EB-NeRD, summarize Candidate K's real trail (the recency fix, the real ebnerd_large scale, the final result) in similarly condensed form, pointing to ADR-013 — don't force it into the same "A–J" shape MIND had. Update the leaderboard results table with every real submission for both competitions — score, submission ID, date — sourced from actual submissions/*/leaderboard_screenshot_*.png files and ADR text, not recalled from memory. Confirm submissions/ebnerd_testset_gbdt_k/ and any MIND submission directories actually contain what the ADRs claim before citing them.
> - Where the pipeline breaks at 10x scale: verify this section is current given everything built since it was last written (MINDlarge memory fixes, ebnerd_testset materialization fix, the real ebnerd_large OOM incident from Candidate K/ADR-013 — that last one is a genuine, real 10x-scale breakage-and-fix worth including if the section doesn't already cover it).
> - Limitations (if kept as a separate section, or folded into observations): state plainly whatever wasn't reached and why (real target vs. real result, e.g. EB-NeRD's result vs. any internally-discussed target) — sourced from ADR-013's own stated comparison table, not re-derived. Don't omit a miss to make the note read cleaner. State the leakage-dependence of the top published literature score for EB-NeRD (the real winning team's score dropped substantially once organizers removed features that leaked future information) as the reason this project's own target was set where it was.
> - Scan the whole file (both .md and .tex) for placeholder, internal-only, or stale text (todo markers, editorial comments, anything referencing conversation-only context) and remove it. Don't assume a specific placeholder exists — check directly; the file may already be clean.
>
> Recompile to PDF and verify page count is ≤4 with pypdf directly — don't eyeball it, don't trust any previously-recorded page count since content changed. Note: pypdf is not currently an installed dependency in this project (checked directly: `poetry show pypdf` → not found) despite an earlier PROJECT_STATE entry describing a pypdf-based page-count check — install it (`poetry add --group dev pypdf` or equivalent) before relying on it, and note in PROJECT_STATE if this was a real gap or if pypdf was available some other way before.
>
> Proofread pass: every number and claim in the note must trace to a specific committed file, in the note's own established citation style (inline path/to/file). Grep-verify a sample of them directly against the source files — don't just skim the note itself.
>
> Confirm git status is clean and everything referenced by the note (ADRs, experiment results, screenshots) is actually committed. On the "pushed" check: this repository currently has no git remote configured at all (`git remote -v` returns empty) — confirm whether that's still true, and if so, treat it as an open question for the engineer (how/where this repo actually gets submitted) rather than something to silently skip or fix unilaterally.
>
> List every file that needs to go to Moodle (design note PDF + whatever else the assignment brief requires, per the Q6 text above and whatever else can be confirmed from repo content). If the numbering discrepancy (Q6 vs Q7) suggests other deliverable requirements exist that aren't yet captured anywhere in the repo, say so plainly as a real gap and ask the engineer rather than guessing.

## What was AI-generated vs. human-written/edited

- **AI-generated, this session:** the EB-NeRD Candidate K section added to
  `docs/design_note.md`/`docs/design_note.tex` (Observations, "Where It
  Breaks at 10x", and Limitations updates covering the real recency-bug
  fix, the real `ebnerd_large` Ada run and OOM incident, and the real
  Codabench result 0.7542 vs. the internally-discussed 0.80 target); the
  leaderboard table update (EB-NeRD Candidate K submission, MIND
  Candidate J corrected-resubmission note); the Q6/Q7 numbering
  resolution note added to `PROJECT_STATE.md`; this log file.
- **AI-verified via direct inspection, not recalled:** every score/ID/date
  cross-checked against `PROJECT_STATE.md`, ADR-012, ADR-013, and the
  actual files under `submissions/*/leaderboard_screenshot_*.png` before
  being written into the note (grep/`ls` run directly, not assumed from
  the ADR text alone); PDF page count re-verified with `pypdf` after
  recompiling with `tectonic`, not eyeballed or carried over from a prior
  recorded count; `poetry show pypdf` confirmed the dependency gap before
  installing it.
- **Human-originated:** the underlying engineering work and results
  themselves (Candidates A–K, the real Codabench submissions, the ADRs)
  were produced in prior sessions; this session only reconciles the
  design note against that already-committed record and does not run any
  new experiments.
