#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

# ---------------------------------------------------------------------------
# Training config (override via env vars)
# ---------------------------------------------------------------------------
OUTPUT="${OUTPUT:-./outputs/awarevln}"
CKPT="${CKPT:-./ck/navila-llama3-8b-8f}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-llava/train/train_mem.py}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-./scripts/zero3.json}"
AWAREVLN_DATA_ROOT="${AWAREVLN_DATA_ROOT:-./data}"

export AWAREVLN_DATA_ROOT

SEED="${SEED:-10}"
DATA_MIXTURE="${DATA_MIXTURE:-r2r+rxr+r2rfollow+rxrfollow+human}"

# ---------------------------------------------------------------------------
# Multi-node distributed config
# Priority:
#   1) Generic env vars: NNODES / NPROC_PER_NODE / RANK / MASTER_ADDR / MASTER_PORT
#   2) Platform env vars: MLP_WORKER_NUM / MLP_ROLE_INDEX / MLP_WORKER_0_HOST / MLP_WORKER_0_PORT
#   3) Defaults for single-node debug
# ---------------------------------------------------------------------------
NNODES="${NNODES:-${MLP_WORKER_NUM:-1}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}"
MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"

TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

echo "============================================================"
echo "  AwareVLN Training"
echo "============================================================"
echo "  PROJECT_ROOT:      $PROJECT_ROOT"
echo "  TRAIN_SCRIPT:      $TRAIN_SCRIPT"
echo "  OUTPUT:            $OUTPUT"
echo "  CKPT:              $CKPT"
echo "  AWAREVLN_DATA_ROOT: $AWAREVLN_DATA_ROOT"
echo "  Nodes:             $NNODES"
echo "  This node rank:    $RANK"
echo "  GPUs/node:         $NPROC_PER_NODE"
echo "  Total GPUs:        $TOTAL_GPUS"
echo "  Master:            $MASTER_ADDR:$MASTER_PORT"
echo "  Data mixture:      $DATA_MIXTURE"
echo "============================================================"

torchrun \
    --nnodes "$NNODES" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --node_rank "$RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "$TRAIN_SCRIPT" \
    --longvila_sampler True \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_name_or_path "$CKPT" \
    --version llama_3 \
    --seed "$SEED" \
    --data_mixture "$DATA_MIXTURE" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --num_video_frames 8 \
    --tune_vision_tower False \
    --tune_mm_projector False \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir "$OUTPUT" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --do_eval False \
    --save_strategy steps \
    --save_steps 500 \
    --fps 0.0 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 32 \
    --lazy_preprocess True
