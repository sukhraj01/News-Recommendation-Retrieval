#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=perf-profile
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Per-hardware leg of ADR-014's benchmark: runs the IDENTICAL profiling and
# ablation scripts that produce the local-machine numbers, so the two are
# comparable by construction rather than by argument. Nothing in
# benchmarks/*.py is Ada-specific; only this wrapper is.
#
# `-c 8` is deliberate, not a default: the local reference machine is an
# 8-core Apple M3, and letting Ada's job use every core on a 64-core node
# would make the comparison measure "more cores" rather than "different
# hardware". Recorded in the results JSON via hardware_spec().
#
# Submit from the login node:
#     sbatch ~/perf_bench/benchmarks/ada_perf_profile.sbatch

set -euo pipefail

REPO="$HOME/perf_bench"
VENV="$HOME/venvs/perf_bench"

echo "=== node: $(hostname) | job: ${SLURM_JOB_ID:-none} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

source "$VENV/bin/activate"
cd "$REPO"

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("GPU not visible to torch — refusing to record a 'GPU' benchmark on CPU")
PY

# Thread count pinned to the allocation so numpy/LightGBM don't oversubscribe
# a shared node and quietly distort every timing in the run.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo; echo "=== [1/4] MIND profile — GPU (cuda) ==="
python benchmarks/profile_mind.py --n-users 1200 --warmup 30 \
    --device cuda --cprofile-users 200 --tag "ada-gpu"

echo; echo "=== [2/4] MIND profile — CPU (same node, same allocation) ==="
python benchmarks/profile_mind.py --n-users 1200 --warmup 30 \
    --device cpu --no-cprofile --tag "ada-cpu"

echo; echo "=== [3/4] EB-NeRD profile (LightGBM) ==="
python benchmarks/profile_ebnerd.py --n-impressions 25000 --chunk-size 5000 \
    --cprofile-impressions 5000 --tag "ada-cpu"

echo; echo "=== [4/4] Performance ablations ==="
python benchmarks/ablations.py --which bm25 ann nrms --device cuda --tag "ada-gpu"
python benchmarks/ablations.py --which lgbm --tag "ada-cpu"

echo; echo "=== done $(date) ==="
ls -la benchmarks/results/
