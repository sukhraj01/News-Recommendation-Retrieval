#!/bin/bash
#SBATCH -J a2-nrms-b
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=31000M
#SBATCH --time=3-00:00:00
#SBATCH -o /home2/sukhraj.singh/a2/logs/%x_%j.out
#SBATCH -e /home2/sukhraj.singh/a2/logs/%x_%j.err
#
# ADR-015 Option B: official-config NRMS (PyTorch port) on Ada.
#
# No installs and no downloads in this job. Every input is already in
# $HOME/a2 (pushed from the local machine and sha256-checked), and the env is
# the torch-cu128 venv ADR-014 proved on these GPU nodes. The GPU is therefore
# used from the first minute: job 2694353 held one idle for ~4h installing
# TF from PyPI (~0.3MB/s from compute nodes).
#
# qos=low: cpu=10, gpu=1, mem=32000M per user; 8 x 4G would be 32768M and is
# rejected (QOSMaxMemoryPerUser), hence --mem=31000M. u22-cpu is devalab-only.
#
# Usage: sbatch scripts/a2_nrms_official.sbatch <mind|ebnerd> <control|treatment> <run-name> [runner args...]
#   e.g. sbatch scripts/a2_nrms_official.sbatch mind treatment mind_treatment --abstract-size 50
set -euo pipefail
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8

P=$HOME/a2
source "$HOME/venvs/perf_bench/bin/activate"
cd "$P/repo"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -c 'import torch; assert torch.cuda.is_available(), "no GPU"; print("torch", torch.__version__, torch.cuda.get_device_name(0))'

DATASET=$1; ARM=$2; NAME=$3; shift 3
case "$DATASET" in
  mind)
    python scripts/a2_nrms_official_run.py --dataset mind --arm "$ARM" --require-gpu \
        --out-dir "$P/results/$NAME" \
        --mind-train-zip "$P/raw/MINDlarge_train.zip" --mind-dev-zip "$P/raw/MINDlarge_dev.zip" \
        --mind-utils-dir "$P/mind_utils" "$@" ;;
  ebnerd)
    python scripts/a2_nrms_official_run.py --dataset ebnerd --arm "$ARM" --require-gpu \
        --out-dir "$P/results/$NAME" \
        --ebnerd-zip "$P/raw/ebnerd_small.zip" \
        --ebnerd-tokens "$P/ebnerd_tokens/ebnerd_small_xlmr30.npz" "$@" ;;
  *) echo "unknown dataset $DATASET"; exit 1 ;;
esac
echo "DONE $(date -Is)"
