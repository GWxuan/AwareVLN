#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${EVAL_ROOT}/.." && pwd)"

cd "$EVAL_ROOT"

MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/ck/awarevln}"
TOTAL_CHUNKS="${TOTAL_CHUNKS:-4}"
IDX_START="${IDX_START:-0}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS - 1))); do
    CHUNK_IDX=$((IDX + IDX_START))
    echo "Total Chunks: $TOTAL_CHUNKS, Local Chunks: $CHUNKS, Chunk Index: $CHUNK_IDX, GPU: ${GPULIST[$IDX]}"

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python run.py \
        --exp-config vlnce_baselines/config/rxr_baselines/awarevln.yaml \
        --run-type eval \
        --num-chunks "$TOTAL_CHUNKS" \
        --chunk-idx "$CHUNK_IDX" \
        EVAL_CKPT_PATH_DIR "$MODEL_PATH" &
done

wait

RESULTS_DIR="${RESULTS_DIR:-${EVAL_ROOT}/eval_awarevln}"
CKPT_NAME="$(basename "${MODEL_PATH}")"
python scripts/eval_jsons.py \
    "${RESULTS_DIR}/${CKPT_NAME}/RxR-VLN-CE-v1/val_unseen" \
    "$TOTAL_CHUNKS"
