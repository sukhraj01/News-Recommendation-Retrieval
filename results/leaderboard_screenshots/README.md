# Leaderboard screenshots — A2 deliverable #3 (Q7)

The two Codabench leaderboard rows for this team's **final, selected**
submissions, one per competition. Captured 2026-09-25 from each competition's
public leaderboard table. Team name on both: **apollo19**.

| File | Competition | Rank | Submission | Uploaded | Score | Next three columns |
|---|---|---:|---:|---|---:|---|
| `mind_codabench_930992_leaderboard.png` | MIND ([13967](https://www.codabench.org/competitions/13967/)) | 33 | **930992** | 2026-09-18 06:36 | **0.6766** | 0.3293 / 0.3575 / 0.4151 |
| `ebnerd_recsys2024_907863_leaderboard.png` | RecSys 2024 / EB-NeRD ([2469](https://www.codabench.org/competitions/2469/)) | 74 | **907863** | 2026-08-29 23:01 | **0.7542** | 0.5323 / 0.5977 / 0.6340 |

Both rows match this repository's own records exactly — submission ID, upload
timestamp and score — for MIND in ADR-015's 2026-09-18 addendum (reproduced
NRMS + title/abstract treatment) and for EB-NeRD in ADR-013's 2026-08-30
addendum (Candidate K, LightGBM `K_rank_nopos`). The three unlabelled columns
after Score line up with this project's locally measured MRR / nDCG@5 / nDCG@10
for the same models, which is why the design note treats Score as AUC; the
column headers are above the visible crop in both captures, so that reading is
inference from the numbers, not a confirmed label.

These are leaderboard **rank** views. The matching upload-confirmation
screenshots, plus the earlier superseded submissions, live beside the
predictions they came from under `submissions/<run>/leaderboard_screenshot_*.png`.
