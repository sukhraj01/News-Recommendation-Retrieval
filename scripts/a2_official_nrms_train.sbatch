#!/bin/bash
#SBATCH -J a2-official-nrms
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
# ADR-015: official NRMS controls (EB-NeRD via ebnerd-benchmark, MIND via
# microsoft/recommenders) in ONE self-provisioning GPU job.
#
# Why one job, set up on node-local scratch (all measured 2026-09-11):
#   - /share1 is NOT mounted on compute nodes (diag job 2694346 on gnode002).
#   - $HOME is cluster-shared but has ~5.8GB free (25.6GB quota) — too small
#     for a ~3.5GB TF+CUDA env plus 2.2GB xlm-roberta plus data.
#   - /ssd_scratch is node-local (563GB free), so a separate prep job would
#     likely land on a different node. Setup happens here instead, with env
#     install and downloads run in parallel from fast sources (HF ~9MB/s) to
#     keep the idle-GPU window to minutes, not the hours that got job
#     2679035 reclaimed in A1.
#   - qos=low caps this account at cpu=10, gpu=1, mem=32000M (sacctmgr).
#     Note 8 x 4G = 32768M EXCEEDS that cap (rejected: QOSMaxMemoryPerUser),
#     hence an explicit --mem=31000M. u22-cpu is devalab-only (scontrol).
#
# Env pins: python 3.11 + tensorflow 2.15.1 satisfies both official repos
# (ebnerd-benchmark: tf<2.22, numpy<1.26.1, polars==0.20.8, transformers<4.37.3;
#  recommenders: tf<2.16). The full freeze is persisted to $P/env_freeze.txt.
#
# Usage:  sbatch scripts/a2_official_nrms_train.sbatch "mind ebnerd"
#         (modes run in order; a failing mode does not block the next)
set -euo pipefail

export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=1 TOKENIZERS_PARALLELISM=false

MODES=${1:-"mind ebnerd"}
P=$HOME/a2                      # persistent + shared: logs, results, small inputs
S=/ssd_scratch/$USER/a2         # node-local: env, caches, data, big weights
mkdir -p "$S"/{cache,raw,data,src,results,venvs} "$P"/results
export UV_CACHE_DIR=$S/cache/uv PIP_CACHE_DIR=$S/cache/pip HF_HOME=$S/cache/hf XDG_CACHE_HOME=$S/cache
UV=$HOME/.local/bin/uv
VENV=$S/venvs/official_nrms
echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h "$S" | tail -1

# ------------------------------------------------------------- setup: env
setup_env() {
    [ -x "$VENV/bin/python" ] || "$UV" venv --python 3.11 "$VENV"
    "$UV" pip install --python "$VENV/bin/python" \
        'tensorflow[and-cuda]==2.15.1' 'tf-slim>=1.1.0' 'numpy==1.26.0' \
        'polars==0.20.8' 'transformers==4.36.2' 'scikit-learn==1.4.0' 'pandas==2.2.3' \
        'pyarrow>=14,<18' pyyaml tqdm requests
    "$UV" pip install --python "$VENV/bin/python" \
        --index-url https://download.pytorch.org/whl/cpu 'torch==2.2.2'
    [ -d "$S/src/ebnerd-benchmark" ] || \
        git clone --depth 1 https://github.com/ebanalyse/ebnerd-benchmark.git "$S/src/ebnerd-benchmark"
    # --no-deps: neither package may re-resolve the pins above.
    "$UV" pip install --python "$VENV/bin/python" --no-deps "$S/src/ebnerd-benchmark"
    "$UV" pip install --python "$VENV/bin/python" --no-deps 'recommenders>=1.2,<1.3'
    { echo "ebnerd-benchmark commit: $(git -C "$S/src/ebnerd-benchmark" rev-parse HEAD)";
      "$UV" pip freeze --python "$VENV/bin/python"; } > "$P/env_freeze.txt"
    HF_HOME=$HF_HOME "$VENV/bin/python" - <<'PY'
from transformers import AutoModel, AutoTokenizer
m = AutoModel.from_pretrained("FacebookAI/xlm-roberta-large")
AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
print("xlm-roberta-large word emb:", tuple(m.embeddings.word_embeddings.weight.shape))
PY
}

# ------------------------------------------------------------- setup: data
fetch() {  # url dest expected_bytes — resumable, size-verified
    local url=$1 dest=$2 want=$3
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$want" ]; then echo "cached $dest"; return; fi
    wget -c --tries=10 --retry-connrefused -q -O "$dest" "$url"
    local got; got=$(stat -c %s "$dest")
    [ "$got" = "$want" ] || { echo "FATAL: $dest is $got bytes, expected $want"; return 3; }
}
setup_data() {
    local HFM=https://huggingface.co/datasets/Recommenders/MIND/resolve/main
    local M=$S/data/mind E=$S/data/ebnerd
    # Stage only what the requested modes need, so an EB-NeRD-only job never
    # blocks on (or fails over) MIND inputs, and vice versa.
    [[ " $MODES " == *" mind "* ]] && setup_mind "$HFM" "$M"
    [[ " $MODES " == *" ebnerd "* ]] && setup_ebnerd "$E"
    return 0
}
setup_mind() {
    local HFM=$1 M=$2
    mkdir -p "$M"   # unzip -d creates only the last path component (job 2694353)
    # Train/dev MUST be the exact files A1 used (J, the embedding baseline),
    # so the control is paired-comparable on the same 376,471 dev impressions.
    # HF's Recommenders/MIND zips are NOT byte-identical (train 531,360,717 vs
    # 530,197,363 bytes; dev 103,592,887 vs 103,456,953), so they are pushed
    # from the local machine into $P/raw and sha256-verified here.
    declare -A SHA=(
        [MINDlarge_train.zip]=120dbabba3c889b47f55070bdfe37ba1384c1aa5a3db20fac468d6c695e46840
        [MINDlarge_dev.zip]=a9ce423cf21b1040e8bafb03ee7283cf5cb170b727e3c49fc054ca0b754e66c8
    )
    for z in MINDlarge_train.zip MINDlarge_dev.zip; do
        [ -f "$S/raw/$z" ] || cp "$P/raw/$z" "$S/raw/$z"
        echo "${SHA[$z]}  $S/raw/$z" | sha256sum -c - || { echo "FATAL: $z sha256 mismatch"; return 3; }
    done
    fetch "$HFM/MINDlarge_utils.zip" "$S/raw/MINDlarge_utils.zip" 150359301
    for s in train dev; do
        if [ ! -f "$M/$s/behaviors.tsv" ]; then
            unzip -oq "$S/raw/MINDlarge_$s.zip" -d "$M/${s}_raw"
            # Zips differ by mirror: some nest under MINDlarge_<s>/, some are flat.
            local d; d=$(dirname "$(find "$M/${s}_raw" -name behaviors.tsv | head -1)")
            [ -f "$d/news.tsv" ] || { echo "FATAL: no behaviors.tsv/news.tsv in MINDlarge_$s.zip"; return 4; }
            mkdir -p "$M/$s"; ln -sf "$d/behaviors.tsv" "$M/$s/behaviors.tsv"; ln -sf "$d/news.tsv" "$M/$s/news.tsv"
        fi
    done
    if [ ! -f "$M/utils/nrms.yaml" ]; then
        unzip -oq "$S/raw/MINDlarge_utils.zip" -d "$M/utils_raw"
        mkdir -p "$M/utils"
        # Flatten whatever directory nesting the utils zip uses.
        for f in nrms.yaml embedding.npy word_dict.pkl uid2index.pkl; do
            src=$(find "$M/utils_raw" -name "$f" | head -1)
            [ -n "$src" ] || { echo "FATAL: $f not in MINDlarge_utils.zip"; return 4; }
            ln -sf "$src" "$M/utils/$f"
        done
    fi
    ls -la "$M" "$M/utils"
}
setup_ebnerd() {
    local E=$1
    # ebnerd_small comes from $P (pushed from local, sha256-checked): S3 is
    # throttled to ~18KB/s from Ada. Official layout: $E/articles.parquet +
    # $E/ebnerd_small/{train,validation}/{behaviors,history}.parquet
    echo "84fc6fc902b8e8dad8383835c90185d54e0916060586d428ced391da4a92bd4f  $P/raw/ebnerd_small.zip" \
        | sha256sum -c - || { echo "FATAL: ebnerd_small.zip sha256 mismatch"; return 3; }
    mkdir -p "$E/ebnerd_small"
    [ -d "$E/ebnerd_small/train" ] || unzip -oq "$P/raw/ebnerd_small.zip" -d "$E/ebnerd_small"
    [ -f "$E/articles.parquet" ] || cp "$E/ebnerd_small/articles.parquet" "$E/articles.parquet"
    ls -la "$E" "$E/ebnerd_small"
}

t0=$(date +%s)
setup_env  > "$P/logs/setup_env_$SLURM_JOB_ID.log"  2>&1 & PID_ENV=$!
setup_data > "$P/logs/setup_data_$SLURM_JOB_ID.log" 2>&1 & PID_DATA=$!
wait $PID_ENV  || { echo "FATAL: env setup failed";  tail -30 "$P/logs/setup_env_$SLURM_JOB_ID.log";  exit 5; }
wait $PID_DATA || { echo "FATAL: data setup failed"; tail -30 "$P/logs/setup_data_$SLURM_JOB_ID.log"; exit 6; }
echo "setup done in $(( $(date +%s) - t0 ))s"

source "$VENV/bin/activate"
# TF 2.15's pip CUDA libs live under site-packages/nvidia/*/lib; expose them
# in case auto-discovery misses on this node (harmless if it does not).
NV=$(python -c 'import nvidia, os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || true)
[ -n "$NV" ] && export LD_LIBRARY_PATH="$(ls -d "$NV"/*/lib | paste -sd:):${LD_LIBRARY_PATH:-}"
python -c 'import tensorflow as tf; g=tf.config.list_physical_devices("GPU"); print("TF", tf.__version__, "GPUs", g); assert g'

REPO=$P/repo
EXAMPLES=$S/src/ebnerd-benchmark/examples/reproducibility_scripts
run() {  # mode name [extra args...]
    local mode=$1 name=$2; shift 2
    local out=$S/results/$name
    case "$mode" in
      mind)   python "$REPO/scripts/a2_mind_official_nrms.py" --data-dir "$S/data/mind" --out-dir "$out" "$@" ;;
      ebnerd) python "$REPO/scripts/a2_ebnerd_official_nrms.py" --official-examples "$EXAMPLES" --out-dir "$out" "$@" ;;
    esac
    # Persist the small outputs; EB-NeRD weights (~1GB+, xlm-roberta vocab)
    # stay on scratch to protect the $HOME quota.
    mkdir -p "$P/results/$name"
    cp "$out"/results.json "$out"/*_scores.parquet "$P/results/$name/" 2>/dev/null || true
    [ "$mode" = mind ] && cp "$out"/*.weights.h5 "$P/results/$name/" 2>/dev/null || true
}

# ------------------------------------------------------------- smoke first
# Tiny end-to-end pass of each mode so a code bug surfaces in minutes.
for m in $MODES; do
    case "$m" in
      mind)   run mind smoke_mind --smoke-lines 3000 --epochs 1 ;;
      ebnerd) run ebnerd smoke_ebnerd --max-val-impressions 2000 --val-chunks 2 -- \
                  --data_path "$S/data/ebnerd" --datasplit ebnerd_small --train_fraction 0.01 --epochs 1 ;;
    esac
    echo "SMOKE OK: $m $(date -Is)"
done

# ------------------------------------------------------------- full controls
for m in $MODES; do
    case "$m" in
      mind)   run mind mind_nrms_control || echo "MODE FAILED: mind" ;;
      ebnerd) run ebnerd ebnerd_nrms_control -- --data_path "$S/data/ebnerd" --datasplit ebnerd_small \
                  || echo "MODE FAILED: ebnerd" ;;
    esac
    echo "FULL DONE: $m $(date -Is)"
done
du -sh "$S"; quota -s | tail -2 || true
echo "DONE $(date -Is)"
