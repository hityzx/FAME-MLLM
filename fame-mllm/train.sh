#!/usr/bin/env bash

# Replace the placeholders below before running.
GPU_ID="<GPU_ID>"
MODEL_PATH="/path/to/merged_qwen3vl_fava_model"
TRAIN_DATA="/path/to/train_cot.jsonl"
SYSTEM_PROMPT="/path/to/system_prompt.txt"
OUTPUT_DIR="/path/to/training_outputs"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VIDEO_MAX_TOKEN_NUM=128
export IMAGE_MAX_TOKEN_NUM=128

echo "Starting FaVA supervised fine-tuning..."

swift sft \
    --model_type qwen3_vl \
    --model "${MODEL_PATH}" \
    --dataset "${TRAIN_DATA}" \
    --remove_unused_columns False \
    --system "${SYSTEM_PROMPT}" \
    --train_type lora \
    --tuner_backend peft \
    --torch_dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs 6 \
    --save_total_limit 6 \
    --max_length 8192 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --modules_to_save fava_projector cls_head \
    --gradient_checkpointing true \
    --per_device_train_batch_size 1 \
    --weight_decay 0.01 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.03 \
    --save_strategy epoch \
    --logging_steps 10

status=$?
if [ "${status}" -ne 0 ]; then
    echo "Training failed."
    exit "${status}"
fi

echo "Training finished. Outputs: ${OUTPUT_DIR}"

