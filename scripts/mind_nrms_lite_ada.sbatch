#!/bin/bash
# Candidate J (NRMS-lite) on IIIT-H's Ada cluster. Submit with:
#   sbatch scripts/mind_nrms_lite_ada.sbatch
# Monitor with:
#   squeue -u $USER
#   tail -f mind_nrms_lite_<jobid>.out
#
# Assumes: (1) a venv at $HOME/venvs/nrms_lite_uv already has torch/pandas/
# numpy/scikit-learn/pyarrow installed -- built via `uv`, not plain
# python3/pip: the login node's system Python is 3.6 (too old for this
# project's `str | None`-style type hints) and has no newer Python or
# conda reachable via `module avail` (that command errors out on this
# cluster -- broken/stale modulefiles, not just outdated names). Standard
# Miniconda also fails outright here (`Installer requires GLIBC >=2.28,
# but system has 2.17` -- this login node's glibc predates it). `uv`
# (https://astral.sh/uv) sidesteps both: a static binary plus portable
# Python builds that don't depend on system glibc/conda at all. Verified
# working on this exact account this session, not assumed. (2)
# MINDsmall_train.zip/MINDsmall_dev.zip staged under $RAW_DIR. (3) this
# repo copied to $HOME/assignment-1-news-retrieval (rsync from a local
# machine, in this case -- no GitHub remote was set up, so no `git clone`
# step here; re-rsync manually if the local code changes).
#
# `sinfo`'s real partition list on this cluster is u22 (default,
# GPU nodes)/u22-cpu/ihub/plafnet2/rrc -- NOT the `long` partition the
# Ada wiki's example uses (that name doesn't exist here; the wiki is
# stale for this cluster). Package installs (`uv pip install`) also
# reliably crash on the LOGIN node itself (`memory allocation of N bytes
# failed`) -- the login node enforces a real memory ceiling that heavy
# dependency resolution blows through; run env setup from inside an
# `srun` allocation, not the login node, same as the training job itself.
#
# Wall-clock set to 2 days, well under the real ~4-day ceiling for the
# `research` account's `low`/`medium` QoS. A real benchmark run this
# session (--max-train-examples 3000 --epochs 1, on gnode007 w/ a GTX
# 1080 Ti) measured 7.7 min for one epoch's train+eval step -- almost
# entirely dev evaluation (73,152 real impressions, scored one at a time,
# not batched -- a real GPU-utilization inefficiency, not optimized here
# since the time budget absorbs it comfortably; a legitimate future
# speedup, not a blocker). Scaling training to the real ~146K-example
# train set across up to 30 epochs projects to roughly 10-20 hours total
# -- comfortable margin under 2 days, itself comfortable margin under the
# real account ceiling.

#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=nrms_lite_j
#SBATCH --output=mind_nrms_lite_%j.out

set -euo pipefail

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/nrms_lite_uv"
RAW_DIR="/ssd_scratch/$USER/mind_raw"
DATA_DIR="/ssd_scratch/$USER/mind_nrms_lite/data"
GLOVE_DIR="/ssd_scratch/$USER/glove"
RESULTS_DIR="$HOME/mind_nrms_lite_results"

# No module load needed -- confirmed this session. `module avail cuda`
# errors out on this cluster anyway (broken modulefiles). The real issue
# wasn't a missing module: plain `uv pip install torch` grabbed a
# CUDA-13-bundled build, too new for this node's driver (nvidia-smi:
# driver 570.211.01, max CUDA 12.8) -- `torch.cuda.is_available()`
# silently returned False until torch was reinstalled from
# https://download.pytorch.org/whl/cu121 specifically. That reinstall
# already happened when $VENV_DIR was built; nothing to do here, but
# worth knowing if a fresh venv build ever needs redoing.

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed -- --gres=gpu:1 may not have been honored, check squeue/job state"

# GloVe download included (not --no-glove): downloads.cs.stanford.edu
# measured ~60-80KB/s this session (6.5h ETA for glove.6B.zip) -- too
# slow for a 6-hour interactive srun session, but this batch job gets a
# real 2-day budget, so it can just spend its first ~6.5h downloading via
# _ensure_glove's own fallback before training starts. Confirm the
# interactive session's own background wget (still running in the
# gnode007 srun session) has been stopped first -- two processes writing
# to the same $GLOVE_DIR/glove.6B.zip path risks a corrupt/truncated file.
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
