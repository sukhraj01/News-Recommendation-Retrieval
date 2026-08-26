#!/bin/bash
# Generate real MINDlarge_test predictions for Candidate J (NRMS-lite,
# ADR-012), using the trained checkpoint from the MINDlarge-dev
# verification run (0.6579 AUC, CI 0.6569-0.6588, 2026-08-25 addendum).
#
# Submit with: sbatch scripts/mind_nrms_predictions.sbatch
# Monitor with: squeue -u $USER ; tail -f mind_nrms_predictions_<jobid>.out
#
# Not pinned to a specific node (an earlier pin backfired -- see
# mind_nrms_lite_ada_large.sbatch's own comment). No GloVe needed here --
# the checkpoint already has its embeddings baked in, this script only
# does inference over MINDlarge_test, never touches the GloVe file.
#
# Inference-only (no backward pass) over MINDlarge_test's 2,370,727 real
# impressions (6.3x MINDlarge-dev's 376,471) -- real per-epoch training+
# eval timing from the verification run was ~117 min/epoch combined; this
# is eval-only work at a larger impression count, so genuinely uncertain
# in which direction it nets out. 2-day budget kept for safety margin,
# not a specific measurement.

#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=nrms_predictions
#SBATCH --output=mind_nrms_predictions_%j.out

set -euo pipefail

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/nrms_lite_uv"
RAW_DIR="/ssd_scratch/$USER/mind_raw_large"
DATA_DIR="/ssd_scratch/$USER/mind_nrms_lite/data_large"
CHECKPOINT="$HOME/mind_nrms_lite_large_results/nrms_lite_best.pt"
OUT_DIR="${OUT_DIR:-$HOME/mind_nrms_predictions}"  # override: sbatch --export=ALL,OUT_DIR=/path scripts/mind_nrms_predictions.sbatch

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed -- --gres=gpu:1 may not have been honored, check squeue/job state"

python scripts/generate_mind_nrms_predictions.py \
    --raw-dir "$RAW_DIR" \
    --data-dir "$DATA_DIR" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR" \
    --embed-dim 300 \
    --num-heads 15 \
    --expected-vocab-size 26293

echo "Done. Predictions in $OUT_DIR/prediction.txt."
