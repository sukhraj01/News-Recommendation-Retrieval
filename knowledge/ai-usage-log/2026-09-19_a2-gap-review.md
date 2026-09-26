# 2026-09-19 — Assignment 2 brief vs. repository gap review

**Engineer:** Neill Rawani
**Tool:** Claude Code (Claude Opus 5)

## Prompts (verbatim, in order)

1. > Go through A2.pdf and look at the state of the project throughly till now, everything code and docs are here, i just need you to tell me if i can make any additions to it (minor) as i also have to show my commits. this was done by my partner and his claude but i should also add a bit if possible

2. > actually wait switch to this branch first (leaving the a2 pdf untracked): a2-q3-official-baseline. then check the project

3. > my partner wont be working on this anymore, from here it is all mine so whatever we do will be on a new branch. you do the changes and i will make the commits accorsingly. you tell me in detail what all you are doing and only do it one by one dont fix everythiing at once. I will tell you or check with you for other things. lay out in a table all minor changes doable before tomorrow ends that we will change and timeline estimate for it and what its impact will be on the project

4. > ok go ahead and get the requirements for the project first in order to run it and then we will start. do the 1 and 2 and instead of commiting yourself send me what to commit and i will do it, you just stage it. add a2pdf in gitignore too tso that the branch  that nis published it doesnt go on git

5. > pushed now lets do the next step and let me know when done and tested. also let me know if any step requires rerunning of the pipeline which requires time as everything is already tested i feel, all we can do is at max update some docs or final report in desiogn note

6. > start change 3

7. > ok, now lets complete the project with the refinements and polish we discussed, tell me whats next (per our plan) and we can get on it and finalize stuff quick. again, step by step only, remove the prompt you added in the unstaged modified file for me creating the testguide as it is not relevant. and start the next step required

8. > [the planned-changes table from an earlier Claude reply, re-pasted by the engineer] I think 1-3 were done, dont waste time installing latex library i will verify stuff on overleaf. just wrote it. and also the final report for this has to be not written by AI so in the end you will help me create a report of my own anyway

9. > is it mentioned in A2 that report needs submission screenshots cause if not i dont wanna add them

10. > ok so i have added the 2 screenshots in the results/ folder, scan through them and rename them and relocate them as needed (name is apollo19 in the screeenshots and you can see which is which via the score). so mention in the report where they are. also i dont think team contributions is really required per the A2 pdf so dont add that either (unless im wrong). just focus on this last step and i will jhandle the report on my own after this

11. > [several turns of report drafting: the engineer wrote each section himself and asked for correction, fact-filling and trimming, then supplied the finished text as `IRE Assignment 2 Final.txt`] ok i have added the IRE assignment 2 final txr file here which has the text of the report, so neatly make it into the report by adding tables where necessary for displaying data and format it perfectly while keeping the sae content and wording (as much as possible) and again make sure nothing is missing per the requirements. ont replace the old design note, maybe rename the old one to something like _old and this one name it proper so i can finally convert to pdf, notice the difference in writing style too if any and if it can be corrected minimally. make the report look appealing not ugly and out of place data etc use good formatting with tables and all as there should be

12. > Ok, so the final file is Design_note_A2 which i just added, I dont want any other a2 design note to be trakced by git (whether pdf or tex) so do that. and then i will commit this design note

## What was produced

- AI-generated: a read-only review mapping the A2 brief (Q1–Q9) against the
  existing code, ADRs, design note, and benchmarks, listing gaps and a set of
  small additions. No code or project docs were changed this session.
- The first review was run against `main`, which turned out to hold only the
  Assignment 1 state. After switching to `a2-q3-official-baseline` (where the
  A2 work lives), the review was redone; most gaps from the first pass were
  already closed on this branch.
- Human-run, AI-driven: a from-scratch environment rebuild on the engineer's
  machine (uv-installed CPython 3.11.15, Poetry as a uv tool, `poetry install`,
  CPU-wheel `torch` 2.13.0 + `sentence-transformers` 5.7.0) and a full test-suite
  run. Results: unit 349 passed; integration 4 passed / 1 skipped / 40 errors and
  reproducibility all-error, every one of them `RawDataMissingError` because the
  raw bundles are not on this machine. No source code was changed.
- AI-generated, human-reviewed: the `.gitignore` rule untracking `A2.pdf`, and
  three README edits (verified fresh-clone setup path, a fresh-clone test-status
  table, and a correction of the false claim that `torch` was not a project
  dependency).
- AI-generated, human-reviewed: the PROJECT_STATE.md handover pass — refreshed
  date and current objective, the handover/branch note, the 2026-09-20 environment
  verification row, a re-assessment of the three open risks (two closed by
  measurement, kept with their resolutions), and two stale deliverable-checklist
  rows.
- AI-generated, human-reviewed: two edits to `docs/design_note_a2.tex` — a
  Limitations bullet giving ADR-007's reason for coverage@10 having no bootstrap
  CI, and a correction to §1.1's claim that EB-NeRD's session/context
  features were added for A2 (git dates them to 2026-08-30, during A1). Not
  recompiled locally by the engineer's instruction; the PDF is rebuilt on
  Overleaf.
- Human-supplied: the two Codabench leaderboard screenshots. AI-generated: their
  renaming/relocation to `results/leaderboard_screenshots/`, the README
  tabulating each row (read off the images themselves and cross-checked against
  ADR-013/ADR-015), and the design-note paragraph pointing at them.
- **Human-written: the A2 report's prose.** The engineer drafted every section
  himself; Claude's role was correcting factual errors, filling numbers he marked
  as placeholders, and cutting detail he judged undefendable. Claude then typeset
  that text as `docs/report_a2.tex` (tables, headings, references) and added the
  Q5 metric/slice tables and the Q9 section needed to meet the brief. The earlier
  draft was renamed `docs/design_note_a2_old.tex`/`.pdf`. At the engineer's
  instruction the repository now tracks only the compiled
  `docs/Design_note_A2.pdf`; the LaTeX source and the earlier draft are
  gitignored so no stale A2 design note is left in the tree.
- AI-generated: this log file.
