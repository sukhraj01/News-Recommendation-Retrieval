#!/bin/bash
#SBATCH -J a2-mind-pred
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=31000M
#SBATCH --time=12:00:00
#SBATCH -o /home2/sukhraj.singh/a2/logs/%x_%j.out
#SBATCH -e /home2/sukhraj.singh/a2/logs/%x_%j.err
#
# ADR-015: generate real MINDlarge_test Codabench predictions from a trained
# OfficialNRMS checkpoint (scripts/a2_generate_mind_test_predictions.py).
#
# Same hardened GPU-capability gate as a2_nrms_official.sbatch (real CUDA
# matmul, not just is_available() -- see that script's comments for why).
# --time=12h is generous, not a tight estimate: Candidate J's equivalent
# full MINDlarge_test scoring run (job 2678461, a comparable-sized NRMS
# variant) took 138.9 minutes: oversizing costs nothing here since a job
# releases its allocation as soon as it finishes (ADR-013's own finding).
#
# Usage: sbatch [--dependency=afterok:<training-job-id>] \
#   scripts/a2_generate_mind_predictions.sbatch <checkpoint-dir-name> <abstract-size>
#   e.g. sbatch --dependency=afterok:2695890 scripts/a2_generate_mind_predictions.sbatch mind_treatment_v2 50
set -euo pipefail
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

P=$HOME/a2
source "$HOME/venvs/perf_bench/bin/activate"
cd "$P/repo"
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
    || echo "WARN: nvidia-smi failed on $(hostname) — see the torch CUDA check below"
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

RUN_NAME=$1; ABSTRACT_SIZE=$2
CKPT="$P/results/$RUN_NAME/model_weights.pt"
if [ ! -f "$CKPT" ]; then
    echo "FATAL: $CKPT does not exist -- training did not finish cleanly or wrote elsewhere"
    exit 1
fi
python scripts/a2_generate_mind_test_predictions.py \
    --checkpoint "$CKPT" \
    --mind-test-zip "$P/raw/MINDlarge_test.zip" \
    --mind-utils-dir "$P/mind_utils" \
    --abstract-size "$ABSTRACT_SIZE" \
    --out-dir "$P/results/${RUN_NAME}_test"
echo "DONE $(date -Is)"
