#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=perf-prep
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Builds the benchmark venv. Split out from the GPU profiling job for two
# real reasons, both learned from this account's own history:
#
#  1. `uv pip install torch` FAILS on the Ada login node — the CUDA wheels
#     (nvidia-cublas-cu12, ~600MB) are memory-mapped during extraction and the
#     login node's per-user memory cap kills it with
#     "Cannot allocate memory (os error 12)". A compute allocation with real
#     memory is required.
#  2. It requests NO `--gres=gpu`: a multi-minute package install holding an
#     idle GPU is exactly the pattern that got a job cluster-side cancelled
#     for GPU idleness earlier in this project (PROJECT_STATE, Ada incident
#     log). The `u22-cpu` partition would be the natural home for this, but
#     the `research` account has no association with it — `sbatch` rejects
#     the combination outright ("Invalid account or account/partition
#     combination"), verified against `sacctmgr show assoc`, so a GPU-less
#     job on `u22` is the available equivalent.
#
# Submit:  sbatch ~/perf_bench/benchmarks/ada_perf_prep.sbatch

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
VENV="$HOME/venvs/perf_bench"

echo "=== node: $(hostname) | job: ${SLURM_JOB_ID:-none} | $(date) ==="
free -g | head -2

rm -rf "$VENV"
uv venv --python 3.11 "$VENV"
source "$VENV/bin/activate"
python -V

# --index-url, NOT --extra-index-url. With `--extra-index-url` the resolver is
# still free to prefer PyPI, and on job 2687384 it did exactly that: it picked
# torch 2.14.0 from PyPI, which pulls `cuda-toolkit` 13.0.3.0 and its CUDA-13
# nvidia-* wheels. That is precisely the documented trap for these nodes
# (driver 570.211.01, max CUDA 12.8) — the install succeeds and
# torch.cuda.is_available() is then silently False. `--index-url` replaces PyPI
# for this resolve, so only cu128-built wheels are reachable.
#
# torch pinned to 2.11.0, and this is a KNOWN, DELIBERATE mismatch with the
# local reference machine's 2.13.0 — not an oversight. The cu128 index's
# newest cp311 build is 2.11.0 (verified directly against
# download.pytorch.org/whl/cu128/torch/: 2.7.0, 2.7.1, 2.8.0, 2.9.0, 2.9.1,
# 2.10.0, 2.11.0). These nodes cap at CUDA 12.8, so a newer cu129/cu130 build
# that would match 2.13.0 cannot run here at all.
#
# What that costs, stated plainly: the M3-vs-Ada comparison varies torch
# version as well as hardware, so a cross-machine NRMS latency ratio is
# confounded. The mitigation is structural, not rhetorical — the profiling job
# runs BOTH cuda and cpu on the SAME Ada node with this SAME venv, and that
# GPU-vs-CPU comparison is internally controlled. The brief's actual question
# ("does the answer to where-should-I-invest change by hardware") is answered
# from that controlled pair, with the cross-machine numbers reported alongside
# and explicitly caveated.
#
# UV_HTTP_TIMEOUT raised because job 2687384 also died on a plain download
# timeout against pypi.nvidia.com (these are ~600MB wheels on a shared link).
export UV_HTTP_TIMEOUT=600
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"

# numpy pinned to 1.26.4 exactly as the previous Ada venv did: it ships a
# tested cp311 wheel, and this project has a recorded case of a
# source-compiled numpy producing SILENTLY WRONG results (no crash) on a
# mismatched Python.
# Versions pinned to the local reference machine's, for the same
# like-for-like reason as the torch pin above.
uv pip install "numpy==1.26.4" "pandas==2.3.3" "scipy==1.17.1" \
    "scikit-learn==1.9.0" "lightgbm==4.7.0" "faiss-cpu==1.15.0" \
    pyarrow sentence-transformers rank-bm25

# rank-bm25 is required even though this pipeline never calls it directly:
# run_ebnerd_gbdt_experiment.py imports a helper from a module that imports it
# at module-load time. Omitting it stalled a real job (2680738) previously.

echo; echo "=== verification ==="
python - <<'PY'
import importlib
for m in ["numpy", "scipy", "pandas", "pyarrow", "sklearn", "lightgbm",
          "torch", "faiss", "rank_bm25", "sentence_transformers"]:
    mod = importlib.import_module(m)
    print(f"  {m:22s} {getattr(mod, '__version__', 'n/a')}")
import torch
print(f"  cuda_available        {torch.cuda.is_available()}  (False is EXPECTED here — no GPU requested)")
print(f"  torch cuda build      {torch.version.cuda}")
# The real gate: this venv must be a CUDA 12.x build. A cu13x build would run
# fine on this CPU node and then silently fail to see the GPU in the profiling
# job — fail loudly here instead, where it is cheap to fix.
major = int((torch.version.cuda or "0").split(".")[0])
if major != 12:
    raise SystemExit(f"FATAL: torch built against CUDA {torch.version.cuda}, need 12.x "
                     f"(nodes cap at 12.8) — see job 2687384's failure")
print("  CUDA-12 build confirmed")
PY

# `|| true` is load-bearing under `set -euo pipefail`: on job 2687388 the venv
# built and verified perfectly, then this line returned nonzero and the whole
# job was recorded as FAILED — a misleading status on a completely successful
# build. `quota` exits nonzero on this cluster even when it prints fine.
echo; echo "=== quota ==="; { quota -s 2>/dev/null | tail -3; } || true
echo "=== done $(date) ==="
