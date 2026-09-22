# merge_fava_into_qwen3vl.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from transformers import (
    AddedToken,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)


def strip_prefix_if_present(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_fava_state_dict(fava_ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(fava_ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    state_dict = strip_prefix_if_present(state_dict, "module.")
    state_dict = strip_prefix_if_present(state_dict, "model.")
    state_dict = strip_prefix_if_present(state_dict, "fava.")

    return state_dict


def add_fava_token(tokenizer, fava_token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(fava_token)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    token_missing = token_id is None or token_id == unk_id

    if token_missing:
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    AddedToken(fava_token, special=True, normalized=False)
                ]
            },
            replace_additional_special_tokens=False,
        )

    fava_token_id = tokenizer.convert_tokens_to_ids(fava_token)

    if fava_token_id is None or fava_token_id == getattr(tokenizer, "unk_token_id", None):
        raise RuntimeError(f"Failed to add or locate FaVA token: {fava_token}")

    return int(fava_token_id)


def patch_processor_config(output_dir: Path, args, fava_token_id: int) -> None:
    processor_config_path = output_dir / "processor_config.json"

    if processor_config_path.exists():
        with processor_config_path.open("r", encoding="utf-8") as f:
            processor_config = json.load(f)
    else:
        processor_config = {}

    processor_config.update(
        {
            "use_fava": True,
            "fava_token": args.fava_token,
            "fava_token_id": fava_token_id,
            "fava_num_tokens": args.fava_num_tokens,
            "fava_num_frames": args.fava_num_frames,
            "fava_image_size": args.fava_image_size,
        }
    )

    with processor_config_path.open("w", encoding="utf-8") as f:
        json.dump(processor_config, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser("Merge FaVA checkpoint into Qwen3-VL")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--fava_ckpt_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--fava_token", type=str, default="<|fava_pad|>")
    parser.add_argument("--fava_num_frames", type=int, default=16)
    parser.add_argument("--fava_image_size", type=int, default=224)
    parser.add_argument("--fava_patch_size", type=int, default=16)
    parser.add_argument("--fava_num_bands", type=int, default=4)
    parser.add_argument("--fava_token_dim", type=int, default=256)
    parser.add_argument("--fava_num_tokens", type=int, default=128)
    parser.add_argument("--fava_num_query_blocks", type=int, default=2)
    parser.add_argument("--fava_num_heads", type=int, default=8)
    parser.add_argument("--fava_top_k", type=int, default=1024)
    parser.add_argument("--fava_peak_selection", type=str, default="topk")
    parser.add_argument("--fava_projector_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--freeze_fava", action="store_true", default=True)
    parser.add_argument("--unfreeze_fava", action="store_true")
    parser.add_argument("--use_cls_head", action="store_true")
    parser.add_argument("--cls_num_labels", type=int, default=2)
    parser.add_argument("--cls_loss_weight", type=float, default=0.5)
    parser.add_argument("--lm_loss_weight", type=float, default=1.0)
    parser.add_argument("--cls_dropout", type=float, default=0.1)
    parser.add_argument("--cls_pooling", type=str, default="pre_answer")
    parser.add_argument("--answer_real_token_str", type=str, default="real")
    parser.add_argument("--answer_fake_token_str", type=str, default="fake")

    # loss
    parser.add_argument("--use_consistency_loss", action="store_true")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)

    parser.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--safe_serialization", action="store_true", default=True)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    freeze_fava = args.freeze_fava and not args.unfreeze_fava

    print("=" * 80)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )

    fava_token_id = add_fava_token(tokenizer, args.fava_token)
    print(f"FaVA token: {args.fava_token}")
    print(f"FaVA token id: {fava_token_id}")
    print(f"Tokenizer length after adding FaVA token: {len(tokenizer)}")

    print("=" * 80)
    print("Loading and modifying config...")
    config = Qwen3VLConfig.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )

    config.use_fava = True
    config.fava_token = args.fava_token
    config.fava_token_id = fava_token_id
    config.fava_num_frames = args.fava_num_frames
    config.fava_image_size = args.fava_image_size
    config.fava_patch_size = args.fava_patch_size
    config.fava_num_bands = args.fava_num_bands
    config.fava_token_dim = args.fava_token_dim
    config.fava_num_tokens = args.fava_num_tokens
    config.fava_num_query_blocks = args.fava_num_query_blocks
    config.fava_num_heads = args.fava_num_heads
    config.fava_top_k = args.fava_top_k
    config.fava_peak_selection = args.fava_peak_selection
    config.fava_projector_type = args.fava_projector_type
    config.freeze_fava = freeze_fava
    config.use_cls_head = args.use_cls_head
    config.cls_num_labels = args.cls_num_labels
    config.cls_loss_weight = args.cls_loss_weight
    config.lm_loss_weight = args.lm_loss_weight
    config.cls_dropout = args.cls_dropout
    config.cls_pooling = args.cls_pooling

    # loss
    config.use_consistency_loss = args.use_consistency_loss
    config.consistency_loss_weight = args.consistency_loss_weight
    config.consistency_temperature = args.consistency_temperature

    real_ids = tokenizer.encode(args.answer_real_token_str, add_special_tokens=False)
    fake_ids = tokenizer.encode(args.answer_fake_token_str, add_special_tokens=False)

    print("[INFO] real token ids:", real_ids)
    print("[INFO] fake token ids:", fake_ids)

    if len(real_ids) != 1 or len(fake_ids) != 1:
        raise ValueError(
            f"Expected 'real' and 'fake' to be single tokens, "
            f"but got real={real_ids}, fake={fake_ids}. "
            f"Please adjust answer tokenization."
        )

    config.answer_real_token_id = real_ids[0]
    config.answer_fake_token_id = fake_ids[0]

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    print("=" * 80)
    print("Loading Qwen3-VL with FaVA modules enabled...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("Resizing token embeddings...")
    model.resize_token_embeddings(len(tokenizer))

    # Be explicit for saved config.
    model.config.use_fava = True
    model.config.fava_token = args.fava_token
    model.config.fava_token_id = fava_token_id
    model.config.fava_num_tokens = args.fava_num_tokens
    model.config.fava_num_frames = args.fava_num_frames
    model.config.fava_image_size = args.fava_image_size
    model.config.fava_patch_size = args.fava_patch_size
    model.config.fava_num_bands = args.fava_num_bands
    model.config.fava_token_dim = args.fava_token_dim
    model.config.fava_num_query_blocks = args.fava_num_query_blocks
    model.config.fava_num_heads = args.fava_num_heads
    model.config.fava_top_k = args.fava_top_k
    model.config.fava_peak_selection = args.fava_peak_selection
    model.config.fava_projector_type = args.fava_projector_type
    model.config.freeze_fava = freeze_fava
    model.config.use_cls_head = args.use_cls_head
    model.config.cls_num_labels = args.cls_num_labels
    model.config.cls_loss_weight = args.cls_loss_weight
    model.config.lm_loss_weight = args.lm_loss_weight
    model.config.cls_dropout = args.cls_dropout
    model.config.cls_pooling = args.cls_pooling

    # loss
    model.config.use_consistency_loss = config.use_consistency_loss
    model.config.consistency_loss_weight = config.consistency_loss_weight
    model.config.consistency_temperature = config.consistency_temperature
    model.config.answer_real_token_id = config.answer_real_token_id
    model.config.answer_fake_token_id = config.answer_fake_token_id

    # Some models keep vocab size in text_config.
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = len(tokenizer)
    if hasattr(model.config, "vocab_size"):
        model.config.vocab_size = len(tokenizer)

    print("=" * 80)
    print("Loading FaVA checkpoint...")
    fava_state = load_fava_state_dict(args.fava_ckpt_path)

    missing, unexpected = model.model.fava.load_state_dict(fava_state, strict=False)

    print(f"Loaded FaVA checkpoint from: {args.fava_ckpt_path}")
    print(f"Missing keys in FaVA: {missing}")
    print(f"Unexpected keys in FaVA: {unexpected}")

    if freeze_fava:
        for p in model.model.fava.parameters():
            p.requires_grad = False

    print("=" * 80)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
    )

    # Important: replace processor's tokenizer with the updated tokenizer.
    # Otherwise processor.save_pretrained may overwrite the newly saved tokenizer
    # and remove <|fava_pad|>.
    processor.tokenizer = tokenizer

    # These attributes are used by the modified Qwen3VLProcessor.
    processor.use_fava = True
    processor.fava_token = args.fava_token
    processor.fava_token_id = fava_token_id
    processor.fava_num_tokens = args.fava_num_tokens
    processor.fava_num_frames = args.fava_num_frames
    processor.fava_image_size = args.fava_image_size

    print("=" * 80)
    print(f"Saving merged model to: {output_dir}")

    # Save processor first, then save tokenizer again to guarantee that
    # tokenizer files contain <|fava_pad|>.
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    patch_processor_config(output_dir, args, fava_token_id)

    model.save_pretrained(
        output_dir,
        safe_serialization=args.safe_serialization,
    )

    print("=" * 80)
    print("Done.")
    print(f"Merged model path: {output_dir}")
    print(f"FaVA token id: {fava_token_id}")
    print("You can now load it with:")
    print(f"Qwen3VLForConditionalGeneration.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    main()
