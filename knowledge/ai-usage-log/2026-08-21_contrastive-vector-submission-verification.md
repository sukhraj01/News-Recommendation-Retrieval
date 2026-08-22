# AI Usage Log — EB-NeRD Second Submission Verification (Contrastive Vector on Real Test Set)

Per CLAUDE.md's "Prompt & Session Logging" — verbatim prompts, in order, plus a note on AI- vs. human-generated content.

## Prompts (verbatim)

1. > Current Objective: Today's EB-NeRD contrastive-vector submission (ID 896072, score 0.5404) scored identically to the original MiniLM submission (ID 888045, also 0.5404) — that shouldn't happen given contrastive clearly beat MiniLM on local validation. Confirm whether the two submitted files are actually different before concluding anything.
>
> Diff or checksum the two local prediction files — the original submissions/ebnerd_testset_embed/prediction.zip against today's prediction_contrastive.zip (wherever it landed after download). If they're byte-identical, that's the answer: the wrong file got uploaded to Codabench.
> If they're genuinely different files, spot-check a handful of lines from prediction_contrastive.zip's predictions.txt against known contrastive-vector output from the local ebnerd_small validation run — confirm the rankings actually look like they came from the contrastive vectors, not accidentally from MiniLM.
> Check the Kaggle run's own Cell 8 validation output (line count, malformed-line check) from when it finished — did it actually report a clean, complete run, or did something get missed in the multi-session resume that only surfaces here?
> Report back plainly: are these two files actually different, and if so, where in the pipeline did it go wrong that a genuinely different (and locally-better) method produced an identical score.

2. > Current Objective: Confirmed — today's contrastive-vector submission (896072) and the original MiniLM submission (888045) are genuinely different, not a duplicate upload. The leaderboard's rounded Score column coincidentally shows 0.5404 for both, but the detailed per-day breakdown shows real differences (contrastive mean 0.5402 vs MiniLM 0.5397, every day's number distinct).
>
> Pull both detailed per-day breakdowns (screenshots attached/described: contrastive mean 0.5402, MiniLM mean 0.5397, both "*50% of testset" per-day tables) into the design note's §3.5 leaderboard section — this is real second-submission evidence, replacing whatever placeholder exists there now.
> Note honestly: the real-test-set margin (~0.0005 mean AUC) is much smaller than local validation's CI-clear ~0.0023 gap on ebnerd_small — say so plainly, don't overstate the win.
> Update ADR-008's addendum and PROJECT_STATE.md to reflect this as the second, real EB-NeRD submission is complete and documented.

3. > [Screenshot: "Submission upload" table, both 896072 and 888045 rows, Score 0.5404 for both]
> look at this wha tyou need more tell me

4. > [Two screenshots: "Ranking Metrics: Grouped by Selected Dates" tables, MEAN AUC 0.5397 and 0.5402 respectively]
> the one screenshot with the first row 2023-06-01 0.5374 is prediction.zip and the other is ccontrastive one

5. > retry

## What was AI-generated vs. human-written/edited

- **AI-generated, verified against real artifacts before being trusted:**
  checksum/CRC comparison of the two `prediction*.zip` files; the full
  13,536,710-line diff (impression-ID alignment check, per-line ranking
  comparison, identical-line rate); a local re-run of Cell 8's own
  line-count/malformed-permutation validation logic against the downloaded
  `predictions.txt`. All read real files on disk, not simulated or assumed
  data.
- **AI-flagged, human-resolved:** the submission-ID-to-detail-table
  attribution. The AI's own read of screenshot timing suggested the
  engineer's initial mapping (contrastive=0.5402, MiniLM=0.5397) might be
  backwards, and — per this project's "verify before handing back"
  discipline — declined to write unverified numbers into the design
  note/ADR/PROJECT_STATE. The engineer re-checked directly on Codabench
  (prompt 4) and confirmed the original mapping was correct; the AI's
  suspicion was reasonable given the available evidence but not directly
  confirmed by that evidence, and screenshot-timing inference does not
  substitute for opening the actual eye-icon detail view per submission ID.
- **AI-generated, human-supplied source data:** the per-day AUC/MRR/nDCG
  table values themselves came from screenshots the engineer captured and
  shared from Codabench's own UI, transcribed exactly as shown — not
  computed or estimated by the AI.
- **AI-generated:** all prose/table edits to `docs/design_note.md`,
  `docs/design_note.tex` (recompiled to `docs/design_note.pdf`, page count
  re-verified at 4 via `pypdf` after trimming to fit — an initial margin
  reduction was tried, produced a real 85pt overfull `\hbox`, and was
  reverted in favor of content trims instead), the new Addendum section in
  `decisions/ADR-008-semantic-retrieval-design.md`, and the updates to
  `PROJECT_STATE.md` (Leaderboard Submission row, the stale 2026-08-19
  "Upcoming" bullet, the new Session Notes entry, this file itself).
- **Human-directed throughout:** which files to diff, what "spot-check"
  should mean given no local per-line contrastive validation output
  existed, and the decision to re-verify the screenshot attribution rather
  than accept either the AI's suspicion or the engineer's first recollection
  at face value.
