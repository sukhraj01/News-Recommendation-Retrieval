#!/bin/bash
# Candidate K (ADR-013) SCORE-ONLY job: skips training entirely.
#
# Training already succeeded (job 2681044: K_rank, K_cls, K_rank_nopos all
# trained cleanly after the OOM fix) and its outputs persist in
# $HOME/ebnerd_gbdt_k_results/candidate_k_gbdt_ebnerd_large/ -- models,
# results.json, vocabularies.json, config.json all live in $HOME, so they
# survive regardless of which node this job lands on. That run then hit an
# unrelated bug in STEP 4 (scoring): ebnerd_testset's behaviors.parquet has no
# article_id column, which load_test_behaviors now handles. Re-running the
# full ~75-minute training just to test that one-line fix would be wasteful
# and is exactly the kind of thing worth avoiding -- this script starts
# straight from the already-trained model and only re-does the (cheap, ~2-15
# min) testset encode + scoring steps.
#
# Submit with: sbatch scripts/ebnerd_gbdt_ada_score_only.sbatch
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
# Real measured cost is ~2.6h (job 2681508 got to 70.9% before hitting a too-
# tight 2h limit). But there is no reason to compute a tight estimate here at
# all: the engineer has a 4-day budget for the whole task, not this one job,
# and Candidate J's mind_nrms_lite_ada_large.sbatch already proved qos=medium
# on u22 accepts a 3-day request. A generous, already-proven time budget
# removes an entire class of failure (getting the arithmetic wrong, like the
# 2h limit did) for the cost of nothing -- a job that finishes in 2.6h
# releases its allocation at 2.6h regardless of what --time says.
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ebnerd_gbdt_k_score
#SBATCH --output=ebnerd_gbdt_k_score_%j.out

set -euo pipefail

# Cap BLAS/OpenMP thread pools before anything imports numpy -- same reasoning
# as ebnerd_gbdt_ada_train.sbatch. Must come AFTER the #SBATCH block (SLURM
# only reads #SBATCH directives from the contiguous comment block before the
# first executable line -- an earlier draft of this file put these exports
# BEFORE #SBATCH and would have silently discarded every resource request).
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/ebnerd_gbdt"
RAW_DIR="/ssd_scratch/$USER/ebnerd_raw"
EMB_DIR="/ssd_scratch/$USER/ebnerd_embeddings"
RESULTS_DIR="$HOME/ebnerd_gbdt_k_results"
MODEL_DIR="$RESULTS_DIR/candidate_k_gbdt_ebnerd_large"
SUBMIT_ARM="K_rank_nopos"
SHARED_CACHE="$HOME/ebnerd_shared_cache"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: $MODEL_DIR not found -- training hasn't completed yet." >&2
    echo "Run scripts/ebnerd_gbdt_ada_train.sbatch first." >&2
    exit 1
fi

mkdir -p "$RAW_DIR" "$EMB_DIR" "$RESULTS_DIR" "$SHARED_CACHE"

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"
python -c "import numpy, sys; print('python', sys.version.split()[0], '| numpy', numpy.__version__)"

# Same shared-cache-first, S3-fallback pattern as ebnerd_gbdt_ada_train.sbatch.
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
stage_from_cache "ebnerd_testset.zip"
stage_from_cache "articles_large_only.zip"

echo
echo "=== nvidia-smi ==="
nvidia-smi || echo "no GPU visible -- the encode step falls back to CPU"

MINI="sentence-transformers__paraphrase-multilingual-MiniLM-L12-v2"

echo
echo "############ STEP 1/2: fetch + encode ebnerd_testset's article catalog (GPU) ############"
TEST_EMB_DIR="$EMB_DIR/testset"
mkdir -p "$TEST_EMB_DIR"
python scripts/ebnerd_prepare_bundle.py \
    --bundle testset --raw-dir "$RAW_DIR" --embeddings-dir "$TEST_EMB_DIR" --batch-size 512
save_to_cache "ebnerd_testset.zip"
save_to_cache "articles_large_only.zip"

echo
echo "############ STEP 2/2: score ebnerd_testset with $SUBMIT_ARM ############"
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
echo "Validation metrics (from training) : $MODEL_DIR/results.json"
echo "Upload to Codabench                : $RESULTS_DIR/submission/prediction.zip"
ls -lh "$RESULTS_DIR/submission/" || true
