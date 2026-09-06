# AI Usage Log — 2026-09-07 — GitHub Remote Setup & First Push

Verbatim record of every engineer prompt this session, in order, plus a note on
what was AI-generated vs. human-written. Per CLAUDE.md's "Prompt & Session
Logging" clause.

---

## Prompt 1 (verbatim)

> Current Objective: Set up the new GitHub remote and get all of today's benchmarking work (ADR-014, addenda, benchmarks/, .githooks/, Makefile targets, AI usage log) committed and pushed. Repo was just created empty at https://github.com/sukhraj01/News-Recommendation-Retrieval.git — not yet connected to this local repo.
>
> Check git remote -v first — confirm there's no existing origin already pointing somewhere else before adding this one. If one exists, tell me rather than overwriting it.
> Add the new remote (git remote add origin https://github.com/sukhraj01/News-Recommendation-Retrieval.git), then check whether the GitHub repo already has any commits (e.g. an auto-created README/license) — if so, reconcile that properly (don't force-push over it blindly) rather than assuming it's empty.
> Run make hooks to get the pre-commit performance gate actually wired in before committing this batch of work, since the hook itself is part of what's being committed.
> Commit in logical groups, same discipline as always — real messages explaining why, not one dump: (a) ADR-014 + addenda on ADR-006/008, (b) benchmarks/ (scripts, sbatch files, README, results), (c) .githooks/ + Makefile targets, (d) PROJECT_STATE session notes + the verbatim AI usage log.
> Before pushing: check .gitignore actually excludes what it should (benchmark result blobs, sbatch job outputs, any large/binary files) — a first push to a fresh repo is exactly where an accidentally-committed large file becomes a permanent pain to remove later.
> Push to origin main (confirm the actual default branch name GitHub assigned, don't assume main vs master), then git status and git log --oneline -10 to confirm everything landed and the working tree is clean.
> Leave the two open items (Kaggle T4, local EB-NeRD profile) as PROJECT_STATE's documented next steps — don't block this commit/push on them.

---

## Findings and decisions this session

**Pre-flight checks, all verified rather than assumed:**

- `git remote -v` returned **nothing** — no pre-existing `origin` to overwrite.
  Local branch is already `main`.
- `git ls-remote` against the GitHub URL returned **zero refs** — the remote is
  genuinely empty (no auto-created README or licence), so no reconciliation was
  needed and no force-push was involved at any point.

**`.gitignore` review (ADR-014 additions).** Working tree was measured before
writing any rule: `benchmarks/results/` is **164 KB total**, largest single file
27 KB, nothing binary. Rules added:

- `/*.out`, `/*.err`, `/benchmarks/*.out`, `/benchmarks/*.err` — SLURM writes
  `%x_%j.out` into the submit directory, which would accumulate in the repo root
  on every run. Scoped so the deliberately-fetched Ada job logs under
  `benchmarks/results/ada/` stay tracked; they are the primary record behind
  ADR-014's cited job ids.
- `benchmarks/results/snapshots/` — the pre-commit hook writes one JSON per
  performance-relevant commit, i.e. unbounded growth. `PERFORMANCE_LOG.md` is the
  human-readable deliverable and stays tracked.
- `*.npy *.parquet *.pkl *.joblib *.h5 *.onnx *.safetensors` — none are tracked
  today; a guard for a fresh remote, where a large object stays in history
  permanently even after deletion.

**Deliberately NOT ignored: `benchmarks/results/*.json`** (the full profile and
ablation records). They are small, written rarely, and are the evidence ADR-014
cites by filename — the analogue of `experiments/results.json`, not of the
multi-MB model blobs that got `experiments/` ignored wholesale. Flagged to the
engineer as a reversible call.

Verified with `git ls-files | git check-ignore --stdin --verbose` that **no
currently-tracked file** becomes ignored by the new rules — important because the
repo deliberately tracks `notebooks/*.zip` bundles and `submissions/**/*.png`
leaderboard screenshots via existing negations.

**Pre-commit hook defect found and fixed before first use.** The hook ran
`snapshot.py`, which appends to `benchmarks/PERFORMANCE_LOG.md` — a *tracked*
file — while the commit was being assembled. The log entry would have landed in
the working tree but not in the commit it describes, leaving a stray "update log"
diff after every performance-relevant commit. The hook now `git add`s
`PERFORMANCE_LOG.md` after a passing run, so the record and the change it
records land together.

**Two open items deliberately left open**, per instruction, and carried in
PROJECT_STATE as documented next steps rather than blocking the push: the Kaggle
T4 per-stage row (no API credentials on this machine) and the local EB-NeRD
profile + local ablations (`make bench` when the machine is free).

---

## AI-generated vs. human-written

All commits, commit messages, the `.gitignore` additions, the pre-commit hook fix,
and this log are AI-generated this session. No source under `src/` was modified.
