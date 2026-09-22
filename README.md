# FAME-MLLM

This repository contains the FaVA-T pretraining code and the subsequent FAME-MLLM training and inference pipeline.

## Environment

The main package versions are:

- Python 3.10
- PyTorch 2.8.0
- Transformers 4.57.3
- ms-swift 3.11.1

## Datasets

- FakeSV: [https://github.com/ICTMCG/FakeSV](https://github.com/ICTMCG/FakeSV)
- FakeTT: [https://github.com/ICTMCG/FakingRecipe](https://github.com/ICTMCG/FakingRecipe)

## Workflow

Replace every placeholder path in the scripts and commands before running.

### 1. Train FaVA-T

Configure the dataset and output paths in `fava-t/train.sh`, then run:

```bash
bash fava-t/train.sh
```

The checkpoint used in the merge step is normally:

```text
/path/to/fava_type_rank_outputs/best.pt
```

### 2. Replace the Qwen3-VL implementation

Before merging the model, replace the corresponding Transformers and ms-swift implementations with the files under `fame-mllm/custom`. Set `ENV_PATH` to the target Python environment directory:

```bash
ENV_PATH=/path/to/env_name

cp -a fame-mllm/custom/qwen3_vl/. \
    "${ENV_PATH}/lib/python3.10/site-packages/transformers/models/qwen3_vl/"

cp fame-mllm/custom/qwen.py \
    "${ENV_PATH}/lib/python3.10/site-packages/swift/llm/model/model/qwen.py"
```

Back up the original environment files first if they are still needed.

### 3. Merge FaVA-T into Qwen3-VL

Run the merge script with the base Qwen3-VL model and the FaVA-T checkpoint:

```bash
python fame-mllm/merge_fava_into_qwen3vl.py \
    --base_model_path /path/to/qwen3_vl_model \
    --fava_ckpt_path /path/to/fava_type_rank_outputs/best.pt \
    --output_dir /path/to/merged_qwen3vl_fava_model \
    --torch_dtype bfloat16 \
    --fava_num_tokens 128 \
    --use_cls_head \
    --cls_loss_weight 1.0 \
    --lm_loss_weight 1.0 \
    --use_consistency_loss \
    --consistency_loss_weight 0.5 \
    --consistency_temperature 1.0
```

The answer labels are the default `real` and `fake`, or `真实` and `虚假`.

### 4. Supervised fine-tuning

Set `MODEL_PATH` to the merged model directory and configure the remaining placeholders in `fame-mllm/train.sh`. Then run:

```bash
bash fame-mllm/train.sh
```

### 5. Inference

Choose one training checkpoint and configure the placeholders in `fame-mllm/infer.sh`. Set `INFER_SCRIPT` to the appropriate dataset inference program, such as `infer_sv.py` or `infer_tt.py`. Then run:

```bash
bash fame-mllm/infer.sh
```

The script merges the selected LoRA checkpoint, runs inference once, and writes the predictions and evaluation results to `SAVE_JSON`.
