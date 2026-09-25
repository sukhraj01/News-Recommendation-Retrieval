# 2026-09-19 — Assignment 2 brief vs. repository gap review

**Engineer:** Neill Rawani
**Tool:** Claude Code (Claude Opus 5)

## Prompts (verbatim, in order)

1. > Go through A2.pdf and look at the state of the project throughly till now, everything code and docs are here, i just need you to tell me if i can make any additions to it (minor) as i also have to show my commits. this was done by my partner and his claude but i should also add a bit if possible

2. > actually wait switch to this branch first (leaving the a2 pdf untracked): a2-q3-official-baseline. then check the project

3. > my partner wont be working on this anymore, from here it is all mine so whatever we do will be on a new branch. you do the changes and i will make the commits accorsingly. you tell me in detail what all you are doing and only do it one by one dont fix everythiing at once. I will tell you or check with you for other things. lay out in a table all minor changes doable before tomorrow ends that we will change and timeline estimate for it and what its impact will be on the project

4. > ok go ahead and get the requirements for the project first in order to run it and then we will start. do the 1 and 2 and instead of commiting yourself send me what to commit and i will do it, you just stage it. add a2pdf in gitignore too tso that the branch  that nis published it doesnt go on git

5. > pushed now lets do the next step and let me know when done and tested. also let me know if any step requires rerunning of the pipeline which requires time as everything is already tested i feel, all we can do is at max update some docs or final report in desiogn note

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
- AI-generated: this log file.
