#!/bin/bash
# Candidate K (ADR-013) STAGE 2/2 on Ada: GPU encode + train + score + submit.
#
# Only submit this AFTER scripts/ebnerd_gbdt_ada_prepare.sbatch has completed
# successfully — it assumes ebnerd_large.zip, ebnerd_testset.zip and
# articles_large_only.zip are already sitting in /ssd_scratch. Chain it
# directly off the prepare job so it never even queues until the data is
# there:
#
#   sbatch scripts/ebnerd_gbdt_ada_prepare.sbatch
#   # note the printed job id, then:
#   sbatch --dependency=afterok:<prepare-jobid> scripts/ebnerd_gbdt_ada_train.sbatch
#
# Why this is split from the download at all: this job requests --gres=gpu:1
# and uses the GPU almost immediately (STEP 1 below is the encode step) rather
# than holding it idle through a multi-hour download first — see
# ebnerd_gbdt_ada_prepare.sbatch's header for why that mattered (job 2679035
# was cancelled twice, `sacct` confirmed CANCELLED by the account's own uid,
# both times during the idle-GPU download phase — a shared-cluster fair-use
# pattern, not a script bug).
#
# Same resource shape as the original single-job script (proven schedulable —
# see that history in ebnerd_gbdt_ada_prepare.sbatch's cpus-per-task note).

# --mem-per-cpu bumped 4G -> 8G (40G -> 80G total) after job 2680914 hit a
# real, SLURM-confirmed oom-kill during arm 3's (K_rank_nopos) Dataset
# construction. Root cause: fit_X/stop_X (the un-split 65-feature train
# matrix, ~11.5GB combined at the original 4M-impression sample) were never
# freed, and arm 3 needs a column-sliced COPY (numpy always copies for
# fancy-indexed column selection) that coexists with them -- measured
# combination ~22.7GB just from those four arrays, before LightGBM's own
# binning overhead, embeddings, profiles, and everything else running
# alongside. --cpus-per-task stays 10 (that's the dimension that got a hard
# scheduling rejection earlier at 16 -- see ebnerd_gbdt_ada_prepare.sbatch's
# note); --mem-per-cpu is a different resource axis, worth the higher ask.
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
# Generous, proven-schedulable budget rather than a computed-per-job estimate
# -- see ebnerd_gbdt_ada_score_only.sbatch's matching note. Candidate J's
# mind_nrms_lite_ada_large.sbatch already proved qos=medium/u22 accepts 3 days.
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ebnerd_gbdt_k_train
#SBATCH --output=ebnerd_gbdt_k_train_%j.out

set -euo pipefail

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/ebnerd_gbdt"
RAW_DIR="/ssd_scratch/$USER/ebnerd_raw"
EMB_DIR="/ssd_scratch/$USER/ebnerd_embeddings"
RESULTS_DIR="$HOME/ebnerd_gbdt_k_results"
MODEL_DIR="$RESULTS_DIR/candidate_k_gbdt_ebnerd_large"
SUBMIT_ARM="K_rank_nopos"
SHARED_CACHE="$HOME/ebnerd_shared_cache"

mkdir -p "$RAW_DIR" "$EMB_DIR" "$RESULTS_DIR" "$SHARED_CACHE"

# Cap BLAS/OpenMP thread pools BEFORE anything imports numpy. Left
# unset, OpenBLAS auto-detects the underlying node's full core count
# (measured: 40 on gnode068) and tries to spawn that many threads --
# unrelated to --cpus-per-task, which SLURM enforces via a process/cgroup
# limit, not by capping what a library thinks is "available". That
# mismatch crashed a plain `import numpy` in an interactive shell on this
# same node with "OpenBLAS blas_thread_init: pthread_create failed ...
# RLIMIT_NPROC 200 current" -- i.e. the process ulimit, not this job's own
# resources, but the failure mode (a library grabbing far more threads
# than the environment allows) is exactly the shape of thing that could
# also crash mid-training here after hours of real work, not just an
# import check. Capped to 8 to match LGB_PARAMS' own num_threads (leaves
# 2 of the 10 allocated cpus for the main process/orchestration).
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
python -c "import numpy, sys; print('python', sys.version.split()[0], '| numpy', numpy.__version__)"

# Self-healing rather than hard-failing: /ssd_scratch is node-local (ADR-012),
# so this job landing on a DIFFERENT node than the prepare stage did (job
# 2679593 hit exactly this, confirmed via its own error message -- prepare ran
# on gnode035, this one landed on gnode004) means the data just isn't there
# yet on THIS node, not that anything is actually wrong. Chaining the two
# stages by `--dependency` only guarantees ORDERING, never same-node
# placement -- SLURM is free to schedule them anywhere. Pinning both to one
# node was considered and rejected: ADR-012 already found, during Candidate
# J's GloVe download, that pinning to a specific node cost MORE in queue-wait
# than a fresh re-download would have.
#
# First choice on a miss: copy from $SHARED_CACHE (cluster-shared, unlike
# /ssd_scratch) -- fast regardless of node, since it never touches S3. Falls
# back to a real S3 fetch only if the cache doesn't have it either (e.g. truly
# the first run anywhere). This second fallback is what bit job 2679595: it
# landed on gnode068, found nothing in a (at the time nonexistent) shared
# cache, and had to fetch from S3 directly into a ~0.1 MB/s-throttled window
# while holding the GPU -- the exact multi-hour-idle-GPU pattern this whole
# split was built to avoid. The shared cache exists specifically so that,
# once ANY job has fetched a file once, no later job on any node ever needs
# to hit S3 for it again.
stage_from_cache() {
    local filename="$1"
    if [ -f "$SHARED_CACHE/$filename" ] && [ ! -f "$RAW_DIR/$filename" ]; then
        echo "  found $filename in $SHARED_CACHE -- copying locally (no S3 fetch needed)"
        cp "$SHARED_CACHE/$filename" "$RAW_DIR/$filename"
    fi
}
save_to_cache() {
    local filename="$1"
    if [ -f "$RAW_DIR/$filename" ] && [ ! -f "$SHARED_CACHE/$filename" ]; then
        cp "$RAW_DIR/$filename" "$SHARED_CACHE/$filename"
    fi
}

for bundle_check in "ebnerd_large.zip" "ebnerd_testset.zip" "articles_large_only.zip"; do
    stage_from_cache "$bundle_check"
    if [ ! -f "$RAW_DIR/$bundle_check" ]; then
        echo "$bundle_check missing on this node AND not in shared cache -- fetching from S3 (self-healing)"
    fi
done
if [ ! -f "$RAW_DIR/ebnerd_large.zip" ]; then
    python scripts/ebnerd_prepare_bundle.py \
        --bundle large --raw-dir "$RAW_DIR" --embeddings-dir "$EMB_DIR" \
        --skip-embeddings --scale-report
fi
save_to_cache "ebnerd_large.zip"
if [ ! -f "$RAW_DIR/ebnerd_testset.zip" ] || [ ! -f "$RAW_DIR/articles_large_only.zip" ]; then
    TEST_EMB_DIR="$EMB_DIR/testset"
    mkdir -p "$TEST_EMB_DIR"
    python scripts/ebnerd_prepare_bundle.py \
        --bundle testset --raw-dir "$RAW_DIR" --embeddings-dir "$TEST_EMB_DIR" \
        --skip-embeddings
fi
save_to_cache "ebnerd_testset.zip"
save_to_cache "articles_large_only.zip"

echo "=== nvidia-smi ==="
nvidia-smi || echo "no GPU visible -- the encode step falls back to CPU and will be slow"

MINI="sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2"

echo
echo "############ STEP 1/4: encode ebnerd_large articles (GPU) ############"
python scripts/ebnerd_prepare_bundle.py \
    --bundle large --raw-dir "$RAW_DIR" --embeddings-dir "$EMB_DIR" --batch-size 512

echo
echo "############ STEP 2/4: train + evaluate Candidate K on ebnerd_large ############"
# --train-sample-impressions cut 4,000,000 -> 2,500,000 alongside the memory
# fixes above. At 4M the arm-3 double-copy peak measured ~22.7GB just from
# fit_X/stop_X + their sliced copies; at 2.5M (scales roughly linearly) that
# drops to ~14GB, comfortable even against the ORIGINAL 40G budget, so this
# combined with the 80G bump and the code-level free (run_ebnerd_gbdt_
# experiment.py now deletes fit_X/stop_X/fit_y/stop_y once the arm loop
# finishes) is deliberately more margin than the minimum, not a razor-thin
# recalculation -- a second OOM here would cost another ~75 minutes of
# GPU time already sunk into arms 1-2 for nothing. 2.5M is still ~10.7x
# ebnerd_small's original 232,887 train impressions, which already produced
# the 0.7588 result this run is trying to improve on.
python scripts/run_ebnerd_gbdt_experiment.py \
    --bundle large \
    --zip-path "$RAW_DIR/ebnerd_large.zip" \
    --embeddings-npy "$EMB_DIR/$MINI.npy" \
    --embeddings-json "$EMB_DIR/$MINI.json" \
    --train-sample-impressions 2500000 \
    --score-chunk-size 250000 \
    --full-metric-impressions 1000000 \
    --out "$MODEL_DIR"

echo
echo "############ STEP 3/4: encode ebnerd_testset's article catalog (GPU) ############"
TEST_EMB_DIR="$EMB_DIR/testset"
python scripts/ebnerd_prepare_bundle.py \
    --bundle testset --raw-dir "$RAW_DIR" --embeddings-dir "$TEST_EMB_DIR" --batch-size 512

echo
echo "############ STEP 4/4: score ebnerd_testset with $SUBMIT_ARM ############"
python scripts/generate_ebnerd_gbdt_predictions.py \
    --model-dir "$MODEL_DIR" \
    --arm "$SUBMIT_ARM" \
    --testset-zip "$RAW_DIR/ebnerd_testset.zip" \
    --articles-zip "$RAW_DIR/articles_large_only.zip" \
    --embeddings-npy "$TEST_EMB_DIR/$MINI.npy" \
    --embeddings-json "$TEST_EMB_DIR/$MINI.json" \
    --chunk-size 200000 \
    --out "$RESULTS_DIR/submission"

echo
echo "=== DONE ==="
echo "Validation metrics : $MODEL_DIR/results.json"
echo "Upload to Codabench: $RESULTS_DIR/submission/prediction.zip"
ls -lh "$RESULTS_DIR/submission/" || true
