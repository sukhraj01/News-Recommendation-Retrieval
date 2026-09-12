# Ada (IIIT-H SLURM Cluster) Environment & Job Reference

**Subject:** Real environment and SLURM job setup used to train a LightGBM
learning-to-rank model on the full `ebnerd_large` dataset (~12M train / ~12.6M
validation impressions) — "Candidate K" in the source project.

**Audience:** Written for someone who already knows Ada / SLURM generally but
has no context on this specific project. Every claim below is sourced from
the actual job scripts, ADR addenda, and project log committed alongside this
document — file paths and job IDs are given throughout so each claim can be
re-checked against the primary source rather than taken on faith.

**Primary sources checked directly for this document:**
`scripts/ebnerd_gbdt_ada_prepare.sbatch`, `scripts/ebnerd_gbdt_ada_train.sbatch`,
`scripts/ebnerd_gbdt_ada_score_only.sbatch` (as actually submitted — these are
the real, committed job scripts, not templates), `decisions/ADR-013-ebnerd-gbdt-ranker.md`
(2026-08-29 and 2026-08-30 addenda), `PROJECT_STATE.md`, `decisions/ADR-012-nrms-lite-candidate-j.md`
and `scripts/mind_nrms_lite_ada.sbatch` (for foundational Ada environment
findings established earlier on the same account, which this project's later
GBDT job inherits without re-deriving).

---

## 1. Cluster identity and account

| Item | Value | Source |
|---|---|---|
| Cluster | Ada, IIIT-H's SLURM-managed HPC cluster | all job scripts |
| Account | `-A research` | all `#SBATCH` blocks |
| QoS | `medium` (upgraded from `low` mid-project; broke both `sbatch` scripts outright with `Invalid qos specification` until the flag was updated) | `PROJECT_STATE.md:658-659` |
| Partition | `u22` — real `sinfo` output at submission time: `u22` (default, GPU-bearing), `u22-cpu`, `ihub`, `plafnet2`, `rrc`. **Ada's own wiki example partition name (`long`) does not exist on this cluster** — confirmed via `sinfo`, not assumed from documentation. | `scripts/mind_nrms_lite_ada.sbatch:24-26` |
| GPU request | `--gres=gpu:1` | train and score-only scripts |
| Observed GPU driver (on one allocated node, `gnode007`, same account) | NVIDIA driver 570.211.01, **max CUDA 12.8** — a default `pip`/`uv`-resolved `torch` build targeting CUDA 13.x will install successfully but `torch.cuda.is_available()` silently returns `False` | `scripts/mind_nrms_lite_ada.sbatch:64-72` |
| Wall-clock ceiling | ~4 days for this account's `low`/`medium` QoS (real limit, confirmed against the account's own guide — not "no limit" as originally assumed going in) | `PROJECT_STATE.md:732-734` |

---

## 2. Python environment: what was actually run

**There is no `module load` step anywhere in this pipeline.** That's not an
omission — it's a direct consequence of two broken pieces of Ada's stock
tooling, discovered from real error output on this account:

- The login node's system Python is **3.6**, too old for this project's
  modern type hints (`str | None`-style syntax fails outright).
- `module avail` **errors out** on this cluster — broken/stale modulefiles,
  not just an unhelpful listing.
- The obvious fallback, Miniconda, **refuses to install**: its installer
  requires glibc ≥2.28, and this login node's glibc is 2.17.

**Fix adopted (works around all three, verified working on this account):**
[`uv`](https://astral.sh/uv) — a static binary that ships its own portable
CPython builds, independent of the system's glibc or any module/conda setup.
Used for both venv creation and package installation.

Venv creation, verbatim from `scripts/ebnerd_gbdt_ada_prepare.sbatch`:

```bash
VENV_DIR="$HOME/venvs/ebnerd_gbdt"
uv venv --python 3.11 "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install 'numpy==1.26.4' pandas pyarrow scikit-learn scipy lightgbm \
    sentence-transformers torch rank-bm25
```

- **Python pinned to 3.11 specifically**, not "whatever `uv` defaults to":
  the project's `pyproject.toml` declares `python = "^3.11"`, and
  `numpy==1.26.4` ships an official tested wheel for `cp311`. (A related,
  separate finding on this project's local macOS dev machine — not Ada —
  found the same `numpy==1.26.4` running on CPython 3.14 via a
  source-compiled build produced *silently wrong array results*, i.e. no
  crash, just incorrect output on repeated identical computations. That's
  what hardened the "don't casually bump this Python pin" rule that the Ada
  venv also follows. See ADR-013's "Environment defect found while building
  this" section for the full repro.)
- **`rank-bm25` is in the install line despite not being used directly** by
  this pipeline — `run_ebnerd_gbdt_experiment.py` imports a helper function
  from a module that transitively imports it at module-load time. Missing
  this stalled a real job (2680738) immediately after Step 1's encode
  succeeded. Kept in the pin list specifically so a rebuilt venv never
  re-hits this.
- **No `environment.yml` or `requirements.txt` exists for this venv.**
  Versions beyond numpy's hard pin (`pandas`, `pyarrow`, `scikit-learn`,
  `scipy`, `lightgbm`, `sentence-transformers`, `torch`) are whatever `uv`
  resolves at install time from the bare package names above, plus the
  loose ranges in the project's `pyproject.toml` (Poetry, used for local
  dev, not read by this job script). **This is a real reproducibility gap**
  worth closing before handing this setup to a second project: if the venv
  is ever rebuilt weeks later, the exact `torch`/`lightgbm`/`sentence-transformers`
  versions installed are not guaranteed to match what actually produced the
  results in this document.
- **Thread-pool capping, done in-script rather than via any module or env
  file**, and placed *before* the first `numpy` import:

  ```bash
  export OPENBLAS_NUM_THREADS=8
  export OMP_NUM_THREADS=8
  export MKL_NUM_THREADS=8
  export NUMEXPR_NUM_THREADS=8
  ```

  Left unset, OpenBLAS auto-detects the **physical node's full core count**
  (measured 40 cores on one allocated node) rather than the SLURM
  cgroup/job allocation, and can crash a plain `import numpy` with
  `OpenBLAS blas_thread_init: pthread_create failed ... RLIMIT_NPROC` if the
  process ulimit is tighter than the thread count it tries to spawn — this
  was reproduced directly in an interactive shell on this cluster. The
  thread count is capped to leave headroom under `--cpus-per-task` for the
  main process (8 of 10 allocated CPUs in the training job).
- **A separate, earlier finding on this same account (Candidate J's PyTorch
  job) that affects anyone doing GPU training here, not just this
  project:** the default `pip`/`uv`-resolved `torch` build pulls a
  CUDA-13-bundled wheel, too new for this node's driver. Fix: install
  explicitly from `https://download.pytorch.org/whl/cu121`. This GBDT
  pipeline's own `uv pip install torch` line does **not** pin an index URL —
  its GPU usage is limited to `sentence-transformers` encoding, which was
  not observed to hit this issue, but it is a real risk to check via
  `torch.cuda.is_available()` if this environment is reused for anything
  training-heavy on GPU.

---

## 3. Storage layout and quota gotchas

| Path | Scope | Used for |
|---|---|---|
| `$HOME/venvs/ebnerd_gbdt` | Cluster-shared | The venv itself |
| `$HOME/ebnerd_gbdt_k_results` | Cluster-shared | Trained models, `results.json`, final submission zip |
| `$HOME/ebnerd_shared_cache` | Cluster-shared | Manually-built cache of large downloaded source files (see below) |
| `/ssd_scratch/$USER/ebnerd_raw`, `/ssd_scratch/$USER/ebnerd_embeddings` | **Node-local** | Raw dataset zips and computed embeddings during a job |

**The single most important gotcha for anyone assuming "scratch" behaves like
a shared filesystem:** `/ssd_scratch` is **local to whichever compute node a
job lands on, not shared across the cluster.** A `--dependency=afterok:<id>`
chain only guarantees job *ordering* — SLURM is free to schedule the
dependent job on a completely different node, where `/ssd_scratch` will be
empty even though a prior job just populated it. This was hit concretely
twice on this account, on two different candidates:

- Job 2679593 landed on `gnode004` after the prepare stage had actually run
  on `gnode035` — no data present.
- Job 2679595 landed on `gnode068`, found nothing cached anywhere, and fell
  through to a live S3 fetch on a throttled connection **while holding an
  allocated GPU**, the exact "expensive resource sits idle" pattern the
  whole two-stage job split (Section 4) was built to avoid.
- The same class of bug was found independently on the MIND/Candidate J
  work on this account: data staged via an interactive `srun` session was
  invisible to a later batch job on a different node.

**Fix adopted:** an application-level shared cache under `$HOME`
(`$HOME/ebnerd_shared_cache`), *not* an Ada-provided feature. Every job
checks this cache first, copies from it (fast, same-cluster, no network
fetch) if present, and only falls back to a real S3 download as a last
resort — after which it writes its own copy back to the cache for any later
job. Real size to plan `$HOME` quota headroom for: the three EB-NeRD source
files cached this way total **~4.6GB**.

**Pinning a job to a specific node to dodge the re-download, instead of
building a shared cache, was tried on a related job (Candidate J's MINDlarge
run, same account) and rejected on evidence:** the pinned node stayed busy
long enough that the queue wait alone exceeded the cost of a fresh download
on any free node. Don't assume node-pinning is the cheaper fix without
checking real queue-wait first.

**Login node has its own real memory ceiling**, separate from anything a job
allocation grants: heavy `uv pip install` dependency resolution reliably
crashed there with `memory allocation of N bytes failed`. Installs must run
inside an actual job/`srun` allocation, never on the login node directly —
this pipeline does its install inline at the top of the batch job itself for
exactly this reason.

---

## 4. The SLURM job scripts

The pipeline is split into **three stages**, plus a fourth for cheap reruns.
This split itself is a direct fix for a real incident (Section 5, #1): an
earlier single combined script requested a GPU and then spent hours
downloading data before touching it, and got silently `scancel`'d twice
(job 2679035) — `sacct -j 2679035 --format=JobID,State%30 -X` showed
`CANCELLED by <the account's own uid>`, both times during the idle-GPU
download phase. Consistent with a shared-cluster fair-use reclaim of an
idle GPU, not a script bug. The fix: never hold a GPU while doing nothing
with it.

| Script | Stage | GPU? | `--cpus-per-task` | `--mem-per-cpu` | `--time` | Purpose |
|---|---|---|---|---|---|---|
| `ebnerd_gbdt_ada_prepare.sbatch` | 1 | No | 4 | 4G | 3-00:00:00 | Download `ebnerd_large`, `ebnerd_testset`, `articles_large_only` only. No GPU held. |
| `ebnerd_gbdt_ada_train.sbatch` | 2 | Yes (`--gres=gpu:1`) | 10 | **8G** (raised from 4G — Section 5, #4) | 3-00:00:00 | Encode articles, train all 3 arms, encode test catalog, score + package submission. |
| `ebnerd_gbdt_ada_score_only.sbatch` | rerun | Yes | 10 | 4G | 3-00:00:00 | Skip training entirely; re-score an already-trained model (e.g. after fixing an unrelated scoring bug without re-paying ~75 min of training). |

**Real submission commands, verbatim:**

```bash
sbatch scripts/ebnerd_gbdt_ada_prepare.sbatch
# note the printed job id, then chain the GPU stage on successful completion:
sbatch --dependency=afterok:<prepare-jobid> scripts/ebnerd_gbdt_ada_train.sbatch

# for a cheap rerun once a trained model already exists in $HOME:
sbatch scripts/ebnerd_gbdt_ada_score_only.sbatch
```

**On `--time` sizing:** all three requests are a deliberately generous,
*already-proven-schedulable* 3-day budget rather than a tightly computed
estimate — `qos=medium` on `u22` was already confirmed to accept 3-day
requests by an earlier job on this account (Candidate J's MINDlarge run). A
job that finishes early releases its allocation early regardless of what
`--time` asked for, so the only cost of asking generously is nothing; the
cost of asking too tight is a killed job and lost work (see Section 5, #8).

**A structural detail worth flagging explicitly for anyone copying this
pattern:** all `export OPENBLAS_NUM_THREADS=...`-style lines must come
*after* the `#SBATCH` directive block. SLURM only reads `#SBATCH` lines from
the contiguous comment block at the top of the file, before the first
executable line — see Section 5, #7.

---

## 5. Gotchas: real incidents from this run, not anticipated in advance

Eight distinct, real problems were hit and fixed during this project's Ada
work — five during the `ebnerd_large` training session itself, one during
env setup for the same pipeline, one caught before it ever shipped, and one
during a later score-only rerun. None of these were guessed at in advance;
each is traceable to a specific job ID or a specific caught error.

**1. GPU held idle during a multi-hour download → job silently killed twice
(job 2679035).** Root cause and fix: see Section 4's opening paragraph.

**2. Missing transitive dependency stalled a job after partial success (job
2680738).** `rank-bm25` wasn't installed; the job got through Step 1's
encode, then crashed on an import chain that only manifests inside the
training script. Fixed by adding it to the venv's install line explicitly
(Section 2).

**3. `/ssd_scratch` node-locality caused two separate empty-cache misses
(jobs 2679593, 2679595).** Full detail in Section 3.

**4. A real, SLURM-confirmed OOM (`oom-kill` event) during training (job
2680914).** The most consequential incident — worth walking through in
full:

- **Root cause:** `fit_X`/`stop_X` — the un-split, 65-feature training
  matrix, ~11.5GB combined at the original `--train-sample-impressions
  4000000` sample size — were never freed after being used. One of the
  three arms trained per run (`K_rank_nopos`, which withholds two position
  features) needs a **column-sliced copy** of that matrix; NumPy always
  copies (not views) for fancy-indexed column selection. That copy
  coexisted in memory with the original, unfreed matrix: **~22.7GB
  measured from just those four arrays** (`fit_X`, `stop_X`, and their two
  sliced copies), before LightGBM's own binning overhead, embeddings, user
  profiles, or anything else running alongside.
- **Fix, three parts, all applied together:**
  1. The training script now explicitly frees `fit_X`/`stop_X`/`fit_y`/`stop_y`
     once the arm loop that needs them finishes (confirmed by grep that
     they're never read again afterward).
  2. `--train-sample-impressions` cut **4,000,000 → 2,500,000** (scales the
     ~22.7GB peak down to a projected ~14GB — comfortable even against the
     *original* 40G budget, deliberately more margin than the bare
     minimum, since a second OOM here would have cost ~75 minutes of GPU
     time already sunk into the first two arms for nothing).
  3. `--mem-per-cpu` raised **4G → 8G** (40G → 80G total at
     `--cpus-per-task=10`). `--cpus-per-task` was left unchanged — memory
     and CPU count are independent SLURM resource axes, and only the
     memory axis was the actual constraint here.
- **Verified fixed, not just patched:** the re-run (job 2681044) cleared
  all three arms (`K_rank`, `K_cls`, `K_rank_nopos`) successfully.
- **Why 2.5M and not some other number:** it's still ~10.7x
  `ebnerd_small`'s original 232,887 train impressions, which had already
  produced the local win this run was trying to improve on — a real
  scale-up, not a token one, despite being a cut from the original target.

**5. A missing-column crash in the real blind test set.** `ebnerd_testset`'s
`behaviors.parquet` has **no `article_id` column at all** — checked directly
against the real schema, not assumed; every other expected column is
present. Fixed by degrading gracefully (NaN-fill and continue, the same
pattern already used elsewhere in this codebase for missing optional
columns) rather than crashing, which makes every downstream
context-article feature correctly read as "no known context article." The
fix was verified against the real gap: re-running the actual scoring script
on a smaller split with `article_id` artificially stripped produced a
sensible, only-slightly-lower AUC via the official evaluator, not garbage
output.

**6. A second segfault from a `lightgbm` + `torch` import collision inside
`pytest`,** this time triggered by a new test file importing the scoring
script. Fixed by relocating the shared helper function to a module that
does not import `lightgbm`, so importing it for a test no longer pulls in
the conflicting native library pair.

**7. An `#SBATCH`-ordering bug caught before it ever shipped.** An early
draft of the standalone scoring script placed its `export
OPENBLAS_NUM_THREADS=...`-style lines *before* the `#SBATCH` block. SLURM
only reads `#SBATCH` directives from the contiguous comment block preceding
the first executable line — this would have caused every resource request
in that script to be silently ignored. Caught by explicitly checking
directive placement before submission, not discovered via a failed job.

**8. A too-tight `--time` limit killed a score-only rerun partway through
(job 2681508).** That job requested a 2-hour limit and reached 70.9%
completion before being killed; the real measured full cost was ~2.6 hours.
Fixed the same way as the other three scripts: request the same generous,
already-proven-schedulable 3-day budget instead of computing a tight
estimate from a partial data point.

**Foundational Ada gotchas from earlier work on this same account (not
re-derived for this pipeline, but load-bearing for anyone starting fresh
here):** login node system Python is 3.6; `module avail` is broken on this
cluster; Miniconda's installer refuses on this login node's glibc; default
`torch` installs may resolve a CUDA version too new for the actual node
driver; any large external download from this cluster (GloVe, in an
earlier candidate's case) should be assumed throttled (~15–50KB/s observed,
persistent, not a blip) and done with a resumable client (`wget -c
--tries=10 --retry-connrefused`, not a single-shot `urlretrieve`); the
account's QoS can change mid-project and silently invalidate a previously
working `#SBATCH --qos=...` line; an idle interactive session can hold
CPUs a queued batch job needs via a `QOSMaxCpuPerUserLimit` block. Sources:
`decisions/ADR-012-nrms-lite-candidate-j.md`, `scripts/mind_nrms_lite_ada.sbatch`.

---

## 6. Real scale handled successfully after the fixes

Once the OOM fix (Section 5, #4) landed, the pipeline was run at the real,
full `ebnerd_large` scale — not a projection:

| Metric | Value | How measured |
|---|---|---|
| Train impressions (full split) | **12,063,890** | Measured directly against the real bundle |
| Validation impressions (full split) | **12,566,385** | Measured directly against the real bundle |
| Train impressions actually used for fitting | 2,500,000 (deterministic subsample, whole impressions preserved so every `lambdarank` group stays intact) | `--train-sample-impressions 2500000` |
| Validation impressions scored | **All 12,566,385 — no subsampling** | Streamed/chunked, not loaded whole: `--score-chunk-size 250000`, `--full-metric-impressions 1000000` |

The validation set was deliberately **not** subsampled — it was scored via a
streaming path in fixed-size chunks rather than materializing the whole
matrix at once, so the reported metrics are a real full-scale measurement,
not an estimate extrapolated from a smaller run.

**Real costs measured at this scale:** article-catalog encode for the test
set ~15 min; user-profile build ~13 min across 807,677 real users; the
full scoring loop ~124 min at ~1,780–1,870 impressions/sec; the entire
training run (all 3 arms, post-fix) ~75 minutes elapsed.

**Result at this scale** (the arm actually shipped, `K_rank_nopos`): local
validation AUC **0.7590** (95% CI 0.7588–0.7593, computed over the full
12,566,385-impression validation set — a precise result, not one masked by
sampling noise). This transferred to a real Codabench leaderboard score of
**0.7542** (submission 907863) — a small, honest compression (−0.0048), not
a collapse. This is offered here as evidence the environment and job
configuration actually hold up under real production-scale data volume, not
only in a small local test.

---

## 7. Replicating this setup vs. adapting it for a different workload

**Keep as-is — these fixes are specific to Ada's own broken tooling, not to
this project, and will very likely recur for any workload on this
cluster:**
- `uv`-based venv creation (works around the login node's Python 3.6,
  broken `module avail`, and Miniconda's glibc requirement — all
  cluster-level facts, not project choices).
- The shared-cache-under-`$HOME` pattern for any large downloaded input,
  given `/ssd_scratch`'s node-locality.
- Capping BLAS/OpenMP thread env vars before the first `numpy` import.
- Splitting any "download something large, then use a GPU" job into a
  CPU-only download stage chained via `--dependency=afterok` into a
  GPU stage that starts using the GPU immediately — rather than holding a
  GPU idle through a long download.
- Treating any large external download from this cluster as throttled by
  default and using a resumable fetch method.

**Recompute for a different workload — these numbers are sized for this
specific 65-feature LightGBM job, not a general Ada budget:**
- `--mem-per-cpu` / `--cpus-per-task`: sized here from a *measured* ~22.7GB
  peak for this project's specific feature matrix. Re-derive from your own
  peak-RSS measurement on a small-scale dry run; don't copy `8G`/`10` as a
  default.
- `--time`: the 3-day requests here are "generous, already-proven-schedulable"
  budgets, not computed minimums — actual jobs finished in ~75 minutes
  (training) and ~2.6 hours (scoring). Oversizing costs nothing on this
  cluster since a job releases its allocation as soon as it finishes; a
  too-tight estimate has already caused one real killed job here (Section
  5, #8).
- The `torch` CUDA wheel version, if doing GPU training rather than just
  GPU-accelerated encoding: bound to whichever node's driver you land on
  (max CUDA 12.8 was the observed ceiling here, on one specific node) —
  verify with `nvidia-smi` and `torch.cuda.is_available()` rather than
  assuming this figure still holds.
- Package versions beyond `numpy==1.26.4`: currently unpinned beyond bare
  names in the `uv pip install` line (Section 2). Worth fixing with a real
  `requirements.txt`/lockfile before treating this venv as reproducible
  months later.
