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
#
# Exclude known-broken nodes and CHAIN submissions:
#   A=$(sbatch --parsable --exclude=gnode033 ... ebnerd treatment ebnerd_treatment)
#   sbatch --exclude=gnode033 --dependency=afterany:$A ... mind treatment ... --abstract-size 50
# gnode033 killed 2694963/2694964/2694986 at 00:00:00 elapsed: its driver is mismatched
# ("Failed to initialize NVML: Driver/library version mismatch"), nvidia-smi exits 18, and
# set -e ends the script there. sacct's Reason column said QOSMaxCpuPerUserLimit, which was
# the stale pending reason, not the cause -- read the job's own log first.
# The afterany chain also keeps one job eligible at a time under the 1-GPU cap.
set -euo pipefail
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
# Job 2694505 OOM'd with 1.88GB "reserved but unallocated" on an 11GB 2080 Ti;
# expandable segments is the allocator's own advice for that fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

P=$HOME/a2
source "$HOME/venvs/perf_bench/bin/activate"
cd "$P/repo"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
# Non-fatal: on a node with a mismatched driver (gnode033, 2026-09-12) nvidia-smi
# exits 18 and `set -e` would kill the job here with an empty stderr. Let the
# explicit torch.cuda assert below be the authoritative, legible check instead.
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || echo "WARN: nvidia-smi failed on $(hostname) — see the torch CUDA check below"
# A real capability + kernel check, not just is_available(). On gnode012's GTX
# 1080 Ti (sm_61) is_available() returns True, the job passes this gate, and
# then dies 90s later with "no kernel image is available for execution on the
# device" -- this torch build supports sm_75+. Fail in seconds instead, and
# prove a kernel actually runs.
python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch sees no GPU")
cc = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} on {name} (sm_{cc[0]}{cc[1]})", flush=True)
if cc < (7, 5):
    sys.exit(f"FATAL: {name} is sm_{cc[0]}{cc[1]}; this torch build needs sm_75+. "
             "Resubmit excluding this node.")
x = torch.randn(256, 256, device="cuda")
print("cuda matmul OK:", float((x @ x).sum()), flush=True)
PY

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
