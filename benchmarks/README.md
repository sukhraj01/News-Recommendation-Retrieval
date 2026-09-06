# `benchmarks/` — latency & throughput measurement

Companion to `experiments/`. Where `experiments/` captures config+results for
**accuracy** work, this directory does the same for **performance**: per-stage
per-query latency, throughput, and the ablations that attribute cost to
components. Methodology and findings live in
[`decisions/ADR-014-performance-benchmarking-methodology.md`](../decisions/ADR-014-performance-benchmarking-methodology.md).

## Layout

| File | Role |
|---|---|
| `timing.py` | `StageTimer`, hardware-spec capture, results writer. Every other script builds on it. |
| `loaders.py` | Memory-bounded loaders: **full-scale index, systematically sampled queries**. |
| `profile_mind.py` | MIND serving path: tokenize → BM25 → embedding → hybrid → NRMS → rank → metrics. |
| `profile_ebnerd.py` | EB-NeRD serving path: feature frame → LightGBM → rank → metrics. |
| `ablations.py` | The four performance ablations (BM25 scorer, ANN index, NRMS cache, LightGBM features). |
| `snapshot.py` | Lightweight before/after regression tripwire for routine use. |
| `PERFORMANCE_LOG.md` | Auto-appended log of every snapshot. Do not hand-edit. |
| `results/` | Timestamped JSON, one per run. `results/snapshots/` for snapshot runs. |
| `ada_perf_prep.sbatch` | Builds the Ada venv (must run as a job — see below). |
| `ada_perf_profile.sbatch` | Runs the identical profiles + ablations on an Ada GPU node. |

## The routine (this is the part that matters)

Any commit touching `src/retrieval/`, `src/evaluation/ranking_metrics.py`,
`src/datasets/`, `benchmarks/`, `scripts/run_*` or `scripts/generate_*` gets a
before/after pair:

```bash
make bench-baseline                     # before the change
#   … make the change …
make bench-snapshot                     # after — exits nonzero on a regression
```

Install the gate so it is not optional:

```bash
make hooks          # git config core.hooksPath .githooks
```

The hook runs only when a performance-relevant path is staged, and blocks the
commit on a >15% stage regression. Deliberate, understood slowdowns go through
with `SKIP_PERF_SNAPSHOT=1 git commit …` — and the reason belongs in the commit
message.

## Full-scale measurement

```bash
make bench          # profile_mind + profile_ebnerd + all ablations (minutes)
```

Individual runs:

```bash
poetry run python benchmarks/profile_mind.py   --n-users 1200 --device cpu
poetry run python benchmarks/profile_ebnerd.py --n-impressions 25000
poetry run python benchmarks/ablations.py      --which bm25 ann nrms lgbm
```

## Two things to know before trusting a number from here

**1. Sampling is queries-only, never the index.** Every profile runs against the
complete article catalog (MINDlarge-dev: 72,023 articles; real vocabulary; real
384-dim embedding matrix) and samples only *which queries* are issued —
systematically by user, so the 1.47 impressions/user multiplicity that governs
the per-user score caches is preserved. Shrinking the index would change what is
being measured; shrinking the query set does not. `profile_mind.py` projects a
full-split time from its sample and reconciles it against ADR-006's and
ADR-008's independently-measured macro figures on every run.

**2. Run one job at a time on the 8GB machine.** Measured directly during this
work: the BM25 scoring stage read **4.33 ms/user** run alone and **10.9 ms/user**
with a second MINDlarge-scale job running concurrently — a 2.5x distortion, on
the same code and the same data. PROJECT_STATE's machine-stability note already
said this; it is repeated here because it silently invalidates results rather
than producing an obvious failure.

## Ada (SLURM) notes

```bash
rsync -az --exclude __pycache__ src benchmarks scripts USER@ada.iiit.ac.in:~/perf_bench/
ssh USER@ada.iiit.ac.in 'cd ~/perf_bench && sbatch benchmarks/ada_perf_prep.sbatch'
ssh USER@ada.iiit.ac.in 'cd ~/perf_bench && sbatch benchmarks/ada_perf_profile.sbatch'
```

Three failures encountered building this, all encoded in the sbatch scripts so
they are not rediscovered:

- **`uv pip install torch` cannot run on the login node.** The ~600MB CUDA
  wheels are memory-mapped during extraction and the per-user memory cap kills
  it (`Cannot allocate memory (os error 12)`). It must run inside a job.
- **`--extra-index-url` is not enough.** The resolver still preferred PyPI and
  selected torch 2.14.0, which pulls CUDA-13 wheels — the documented trap on
  these nodes (installs cleanly, then `torch.cuda.is_available()` is silently
  `False`). `--index-url` is required, and the prep job hard-fails if
  `torch.version.cuda` is not 12.x.
- **`torch` cannot even be IMPORTED on the login node**, let alone installed:
  `ImportError: libtorch_cpu.so: failed to map segment from shared object:
  Cannot allocate memory`. Verify a venv from inside a job, never from the
  login shell — a login-node import failure says nothing about the venv.
- **`quota` exits nonzero on this cluster even when it prints correctly**, so a
  bare `quota` under `set -euo pipefail` marks an otherwise-successful job
  FAILED (this happened on job 2687388). Guard it with `|| true`.
- **`-p u22-cpu` is unavailable to the `research` account** (`sbatch` rejects the
  combination; confirmed against `sacctmgr show assoc`). The prep job runs on
  `u22` with no `--gres=gpu` instead, so no GPU idles during a package install.

**Version parity is imperfect and deliberately so:** the cu128 index's newest
cp311 torch is 2.11.0 against the local machine's 2.13.0, and these nodes cap at
CUDA 12.8 so a matching build cannot run. The cross-machine comparison therefore
varies torch version as well as hardware. The GPU-vs-CPU comparison *within* one
Ada node uses one venv on one node and is internally controlled — that is the
pair the "does the bottleneck move with hardware" conclusion rests on.

**Never `scancel` a job on Ada that this tooling did not submit.** The account is
shared with the engineer's own long-running work.
