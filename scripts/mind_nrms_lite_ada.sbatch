#!/bin/bash
# Candidate J (NRMS-lite) on IIIT-H's Ada cluster. Submit with:
#   sbatch scripts/mind_nrms_lite_ada.sbatch
# Monitor with:
#   squeue -u $USER
#   tail -f mind_nrms_lite_<jobid>.out
#
# Assumes: (1) a venv at $HOME/venvs/nrms_lite already has torch/pandas/
# numpy/scikit-learn/pyarrow installed (build it on the login node first --
# see the chat/session log for the exact commands, since pip installs from
# a compute node aren't confirmed to have internet access); (2)
# MINDsmall_train.zip/MINDsmall_dev.zip are staged under $RAW_DIR below
# (download them via the login node too, same reasoning); (3) this repo is
# checked out at $HOME/assignment-1-news-retrieval (git clone or scp it
# over -- no Kaggle-style bundle-zip step needed here, unlike the Kaggle
# script, since Ada gives a real filesystem + git).
#
# Wall-clock is set to 1 day, well under the real ~4-day ceiling for the
# `research` account's `low`/`medium` QoS (confirmed via the Ada user
# guide, not assumed) -- generous slack for a job whose real per-epoch
# cost hasn't been measured on this hardware yet. Bump --time if a
# benchmark run (see below) projects longer.

#SBATCH -A research
#SBATCH --qos=low
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=nrms_lite_j
#SBATCH --output=mind_nrms_lite_%j.out

set -euo pipefail

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/nrms_lite"
RAW_DIR="/ssd_scratch/$USER/mind_raw"
DATA_DIR="/ssd_scratch/$USER/mind_nrms_lite/data"
GLOVE_DIR="/ssd_scratch/$USER/glove"
RESULTS_DIR="$HOME/mind_nrms_lite_results"

# module load line: VERIFY the actual available module name first with
# `module avail cuda` on the login node -- the one example in the Ada wiki
# (u18/cuda/10.1) is old and may not match what a pip-installed torch 2.x
# wheel needs. pip-installed torch normally bundles its own CUDA runtime,
# so this module load is likely NOT required for the Python side at all --
# only uncomment/adjust if the job fails with a CUDA/driver mismatch error.
# module load u18/cuda/11.8 2>/dev/null || true

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed -- --gres=gpu:1 may not have been honored, check squeue/job state"

python scripts/mind_nrms_lite_ada_run.py \
    --raw-dir "$RAW_DIR" \
    --data-dir "$DATA_DIR" \
    --glove-dir "$GLOVE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --embed-dim 300 \
    --num-heads 15 \
    --epochs 30 \
    --patience 3

echo "Done. Results in $RESULTS_DIR -- copy them to your local machine / this repo's"
echo "experiments/ directory promptly (RESULTS_DIR is under \$HOME, not scratch, so it"
echo "won't be auto-deleted, but pull them down before you forget regardless)."
