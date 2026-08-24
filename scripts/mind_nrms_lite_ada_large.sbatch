#!/bin/bash
# Candidate J MINDlarge-dev re-verification, per this project's standing
# practice (ADR-010) of checking any local MINDsmall-dev win at MINDlarge
# scale before considering a real Codabench submission -- local wins have
# compressed at real test scale twice already in this project (Candidate
# G, EB-NeRD contrastive vector).
#
# Submit with: sbatch scripts/mind_nrms_lite_ada_large.sbatch
# Monitor with: squeue -u $USER ; tail -f mind_nrms_lite_large_<jobid>.out
#
# NOT pinned to a specific node (an earlier version pinned --nodelist=
# gnode007 to reuse a cached GloVe file there -- reverted: gnode007 was
# busy long enough that the queue wait itself exceeded the ~9h it would
# take to just re-download GloVe on whatever node is actually free, so
# the pin was net-negative in practice, not just theory). GloVe will
# re-download via _ensure_glove's own resumable `wget -c` retry logic
# (mind_nrms_lite_ada_run.py) if not already present on whichever node
# this lands on -- slower, but bounded and proven to complete (it
# already has, once, over this same throttled connection).
#
# Scale: MINDlarge is ~14.2x MINDsmall's train impressions (2,232,748 vs
# 156,965, both real counts from this project's own prior MINDlarge
# work) and ~5.15x its dev impressions (376,471 vs 73,152). Projected
# from real MINDsmall-Ada per-epoch timing (train ~297s, eval ~458s,
# decomposed from the real 755s/epoch total and the earlier 3000-example
# benchmark): ~70 min/epoch training + ~39 min/epoch eval =~ 110 min/epoch.
# This is a PROJECTION, not a measurement -- run the benchmark step in
# this file's own comment block below before trusting it, same
# benchmark-before-commit discipline used at every new scale in this
# project. Memory bumped (4G/cpu vs MINDsmall's 2G/cpu) since MINDlarge
# has real prior history of memory surprises in this project (a 27GB
# projection on an 8GB machine, before the categorical-dtype fixes
# already baked into src/datasets/mind.py) -- those fixes are already in
# place, but this is still real headroom, not assumed safe.

#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=nrms_lite_j_large
#SBATCH --output=mind_nrms_lite_large_%j.out

set -euo pipefail

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/nrms_lite_uv"
RAW_DIR="/ssd_scratch/$USER/mind_raw_large"
DATA_DIR="/ssd_scratch/$USER/mind_nrms_lite/data_large"
GLOVE_DIR="/ssd_scratch/$USER/glove"          # same path as the MINDsmall run -- already has glove.6B.300d.txt on this node
RESULTS_DIR="$HOME/mind_nrms_lite_large_results"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

echo "=== nvidia-smi ==="
nvidia-smi || echo "nvidia-smi failed -- --gres=gpu:1 may not have been honored, check squeue/job state"

python scripts/mind_nrms_lite_ada_run.py \
    --bundle large \
    --raw-dir "$RAW_DIR" \
    --data-dir "$DATA_DIR" \
    --glove-dir "$GLOVE_DIR" \
    --results-dir "$RESULTS_DIR" \
    --embed-dim 300 \
    --num-heads 15 \
    --epochs 30 \
    --patience 3

echo "Done. Results in $RESULTS_DIR."
