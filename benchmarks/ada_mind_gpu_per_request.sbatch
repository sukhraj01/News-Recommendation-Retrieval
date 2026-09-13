#!/bin/bash
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH --constraint=2080ti
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=31G
#SBATCH --time=01:00:00
#SBATCH --job-name=mind-gpu-preq
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# A2 Q4: the one MIND cost row ADR-014's addendum explicitly excluded rather
# than estimated -- GPU-measured per-request latency, needed because the
# earlier profile_mind.py --per-request run used --device cpu, so pricing it
# on a GPU instance (g4dn.xlarge) would have applied CPU latency to GPU
# hardware. Runs the SAME updated profile_mind.py (with --per-request and
# index_memory, added this session) this account's earlier qos=medium runs
# used, just with --device cuda and qos=low (medium is no longer valid for
# this account -- see ADR-015's incident log for both problems the account's
# qos change has already caused, and the constraint=2080ti fix for the two
# incompatible-node failures already hit this session).
#
# Submit from the login node: sbatch ~/perf_bench/benchmarks/ada_mind_gpu_per_request.sbatch
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
    raise SystemExit("GPU not visible to torch -- refusing to record a 'GPU' benchmark on CPU")
PY

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo; echo "=== MIND profile -- GPU (cuda), per-request p99 for A2 Q4 ==="
python benchmarks/profile_mind.py --n-users 1200 --warmup 30 \
    --device cuda --per-request 500 --no-cprofile --tag "a2q4-gpu"

echo "DONE $(date -Is)"
