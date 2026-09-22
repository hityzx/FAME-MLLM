#!/usr/bin/env bash

# Replace the placeholders below before running.
GPU_ID="<GPU_ID>"
TRAIN_JSON="/path/to/train.json"
VAL_JSON="/path/to/validation.json"
FRAMES_ROOT="/path/to/extracted_frames"
CACHE_DIR="/path/to/frame_cache"
OUTPUT_DIR="/path/to/fava_type_rank_outputs"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

python "${SCRIPT_DIR}/train.py" \
    --train_json "${TRAIN_JSON}" \
    --val_json "${VAL_JSON}" \
    --frames_root "${FRAMES_ROOT}" \
    --cache_dir "${CACHE_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs 100 \
    --batch_size 32 \
    --num_workers 4 \
    --lr 2e-4 \
    --amp \
    --rank_weight 0.3 \
    --token_div_weight 0.005 \
    --num_artifact_tokens 128

