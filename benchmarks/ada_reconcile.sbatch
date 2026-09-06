#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -p u22
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --job-name=perf-reconcile
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -euo pipefail
source "$HOME/venvs/perf_bench/bin/activate"
cd "$HOME/perf_bench"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
echo "=== node: $(hostname) | job: ${SLURM_JOB_ID:-none} ==="
python benchmarks/reconcile_adr006.py
