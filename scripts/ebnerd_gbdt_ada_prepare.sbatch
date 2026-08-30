#!/bin/bash
# Candidate K (ADR-013) STAGE 1/2 on Ada: download only. NO GPU requested.
#
# Split out of the original single-job `ebnerd_gbdt_ada.sbatch` after that job
# got silently `scancel`'d twice in a row (job 2679035, both times during the
# multi-hour throttled download, before any GPU work had started) --
# `sacct -j 2679035 --format=JobID,State%30 -X` showed `CANCELLED by 4127`
# (the account's own uid), which the engineer confirmed they did not do
# themselves. Both cancellations landing squarely in the idle-GPU download
# phase (`--gres=gpu:1` held for hours with 0% utilization) matches the
# textbook shape of a shared-cluster fair-use reclaim. Rather than keep
# guessing at cluster policy, the job is restructured so it simply never holds
# a GPU while doing nothing with it: this stage requests no GPU at all, and
# STAGE 2 (`ebnerd_gbdt_ada_train.sbatch`) only starts once this one finishes.
#
# Submit with:  sbatch scripts/ebnerd_gbdt_ada_prepare.sbatch
# Then chain the GPU stage on successful completion:
#   sbatch --dependency=afterok:<this-jobid> scripts/ebnerd_gbdt_ada_train.sbatch
# (the job ID prints from `sbatch`'s own "Submitted batch job <id>" line)
#
# Also front-loads the ebnerd_testset + articles_large_only downloads (STEP 4
# in the original single-job script) here, since they're pure downloads too --
# no reason to make the GPU stage wait on them either.

#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --partition=u22
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
# Generous, proven-schedulable budget -- see ebnerd_gbdt_ada_score_only.sbatch's
# matching note. Matters here specifically: this stage's download already got
# stuck for 2+ hours once on a throttled connection before being cancelled.
#SBATCH --time=3-00:00:00
#SBATCH --job-name=ebnerd_gbdt_k_prep
#SBATCH --output=ebnerd_gbdt_k_prep_%j.out

set -euo pipefail

# Cap BLAS/OpenMP thread pools before anything imports numpy -- see
# ebnerd_gbdt_ada_train.sbatch's matching note for why this exists: left
# unset, these libraries auto-detect the node's full core count (measured 40
# on one node here) rather than anything scoped to this job's actual
# allocation, and crashed a plain `import numpy` in an interactive shell on
# this cluster via RLIMIT_NPROC. This stage's own numpy usage is lighter
# (parquet reads via pandas), but cheap insurance against the same crash.
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

REPO_DIR="$HOME/assignment-1-news-retrieval"
VENV_DIR="$HOME/venvs/ebnerd_gbdt"
RAW_DIR="/ssd_scratch/$USER/ebnerd_raw"
EMB_DIR="/ssd_scratch/$USER/ebnerd_embeddings"
RESULTS_DIR="$HOME/ebnerd_gbdt_k_results"
# $HOME is cluster-shared (unlike /ssd_scratch, which is node-local — ADR-012).
# A copy staged here survives regardless of which node a LATER job (a GPU
# train job, or a retry of this stage) lands on, so it never has to hit S3's
# unpredictably-throttled connection again for the same bytes. Real cost: the
# three source files total ~4.6GB; check `quota` before assuming headroom if
# $HOME is already tight.
SHARED_CACHE="$HOME/ebnerd_shared_cache"
mkdir -p "$SHARED_CACHE"

# /ssd_scratch is node-local, not cluster-shared (learned the hard way during
# Candidate J — see ADR-012). If STAGE 2 lands on a DIFFERENT node than this
# stage ran on, /ssd_scratch here will be empty and it will need to re-fetch.
# That's a real risk this split accepts in exchange for not holding a GPU idle
# — accept it rather than pin nodes (ADR-012 already found a nodelist pin can
# cost more in queue-wait than it saves).
mkdir -p "$RAW_DIR" "$EMB_DIR" "$RESULTS_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "=== creating venv (uv) ==="
    # uv, not the module system's python: Ada's default is 3.6, far below this
    # project's `python = "^3.11"`. Pinned to 3.11 specifically because numpy
    # 1.26.4 ships an official tested wheel for it — a source build against an
    # unsupported interpreter produced provably wrong array results on the local
    # machine this session (see ADR-013's environment-defect section). Do not
    # "upgrade" this pin casually.
    uv venv --python 3.11 "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    # rank-bm25 isn't used by anything this pipeline needs directly, but
    # run_ebnerd_gbdt_experiment.py imports paired_metric_diff_ci from
    # run_gated_cohort_experiment.py, which imports run_bm25_experiment.py at
    # module load time, which imports src/retrieval/index.py, which imports
    # rank_bm25 -- Python evaluates the whole chain even though only one
    # function from it is actually called. Missing this stalled job 2680738
    # right after STEP 1's encode succeeded; caught and fixed here so a fresh
    # venv build never hits it again.
    uv pip install 'numpy==1.26.4' pandas pyarrow scikit-learn scipy lightgbm \
        sentence-transformers torch rank-bm25
else
    source "$VENV_DIR/bin/activate"
fi

cd "$REPO_DIR"
python -c "import numpy, sys; print('python', sys.version.split()[0], '| numpy', numpy.__version__)"

# If a prior run (this stage or a GPU stage's self-healing fallback) already
# staged the file in the SHARED cache, copy it in locally first -- a same-
# cluster copy, not an S3 fetch, so it's fast regardless of how throttled the
# external connection is on this particular node. download_resumable() then
# sees the destination already complete and skips straight past it.
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

echo
echo "############ STEP 1/3: fetch ebnerd_large + real scale report ############"
stage_from_cache "ebnerd_large.zip"
python scripts/ebnerd_prepare_bundle.py \
    --bundle large --raw-dir "$RAW_DIR" --embeddings-dir "$EMB_DIR" \
    --skip-embeddings --scale-report
save_to_cache "ebnerd_large.zip"

echo
echo "############ STEP 2/3: fetch ebnerd_testset ############"
TEST_EMB_DIR="$EMB_DIR/testset"
mkdir -p "$TEST_EMB_DIR"
stage_from_cache "ebnerd_testset.zip"
stage_from_cache "articles_large_only.zip"
python scripts/ebnerd_prepare_bundle.py \
    --bundle testset --raw-dir "$RAW_DIR" --embeddings-dir "$TEST_EMB_DIR" \
    --skip-embeddings
save_to_cache "ebnerd_testset.zip"

echo
echo "############ STEP 3/3: fetch articles_large_only (testset's article catalog) ############"
# ebnerd_prepare_bundle.py's testset path already fetches this as a side
# effect (needed to build the embedding cache), so by this point it's present;
# this step just confirms and reports its size for the record.
ls -lh "$RAW_DIR/articles_large_only.zip"
save_to_cache "articles_large_only.zip"

echo
echo "=== STAGE 1 DONE — no GPU was requested or held ==="
echo "Next: sbatch --dependency=afterok:\$SLURM_JOB_ID scripts/ebnerd_gbdt_ada_train.sbatch"
echo "(replace \$SLURM_JOB_ID with this job's real ID if running that command outside this job)"
