#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=perf-resume
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Steps 3-4 only. Job 2687421 completed steps 1-2 (MIND GPU + MIND CPU) and
# then died in step 3 on a feature-name mapping bug, now fixed. Re-running only
# the failed steps rather than the whole job: steps 1-2 produced valid results
# that are already fetched, and re-measuring them would burn a GPU allocation
# to overwrite good numbers.

set -euo pipefail

REPO="$HOME/perf_bench"
VENV="$HOME/venvs/perf_bench"

echo "=== node: $(hostname) | job: ${SLURM_JOB_ID:-none} | $(date) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

source "$VENV/bin/activate"
cd "$REPO"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo; echo "=== [3/4] EB-NeRD profile (LightGBM) ==="
python benchmarks/profile_ebnerd.py --n-impressions 25000 --chunk-size 5000 \
    --cprofile-impressions 5000 --tag "ada-cpu"

echo; echo "=== [4a/4] Ablations: bm25 / ann / nrms (GPU) ==="
python benchmarks/ablations.py --which bm25 ann nrms --device cuda --tag "ada-gpu"

echo; echo "=== [4b/4] Ablations: nrms on CPU (controlled pair) ==="
python benchmarks/ablations.py --which nrms --device cpu --tag "ada-cpu"

echo; echo "=== [4c/4] Ablations: LightGBM feature count ==="
python benchmarks/ablations.py --which lgbm --tag "ada-cpu"

echo; echo "=== done $(date) ==="
