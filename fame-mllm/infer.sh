#!/usr/bin/env bash

# Replace the placeholders below before running.
GPU_ID="<GPU_ID>"
CHECKPOINT_DIR="/path/to/training_outputs/v0-xxx/checkpoint-xxx"
INFER_SCRIPT="/path/to/inference_script.py"
VAL_TEST_JSON="/path/to/test.json"
FRAMES_ROOT="/path/to/extracted_frames"
SAVE_JSON="/path/to/inference_results.json"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VIDEO_MAX_TOKEN_NUM=128
export IMAGE_MAX_TOKEN_NUM=128

RUN_DIR=$(dirname "${CHECKPOINT_DIR}")
CKPT_NAME=$(basename "${CHECKPOINT_DIR}")
MERGED_DIR="${RUN_DIR}/${CKPT_NAME}-merged"

if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "Checkpoint directory not found: ${CHECKPOINT_DIR}"
    exit 1
fi

cd "${RUN_DIR}" || exit 1

if [ -d "${MERGED_DIR}" ]; then
    echo "Merged model already exists: ${MERGED_DIR}"
else
    echo "Merging checkpoint: ${CHECKPOINT_DIR}"
    if ! swift export \
        --merge_lora true \
        --adapters "${CKPT_NAME}"; then
        echo "Failed to merge checkpoint: ${CHECKPOINT_DIR}"
        exit 1
    fi
fi

if [ ! -d "${MERGED_DIR}" ]; then
    echo "Merged model directory not found: ${MERGED_DIR}"
    exit 1
fi

echo "Running inference with: ${MERGED_DIR}"
PYTHONWARNINGS="ignore::FutureWarning" python "${INFER_SCRIPT}" \
    --test_json "${VAL_TEST_JSON}" \
    --frames_root "${FRAMES_ROOT}" \
    --torch_dtype bfloat16 \
    --max_new_tokens 1024 \
    --resume \
    --save_interval 1 \
    --model_dir "${MERGED_DIR}" \
    --save_json "${SAVE_JSON}"

status=$?
if [ "${status}" -ne 0 ]; then
    echo "Inference failed."
    exit "${status}"
fi

echo "Inference finished."
echo "Merged model: ${MERGED_DIR}"
echo "Results: ${SAVE_JSON}"

