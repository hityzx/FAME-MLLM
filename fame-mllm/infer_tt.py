import argparse
import json
import os
import re
import traceback
from typing import List, Optional, Dict, Any


import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from tqdm import tqdm

# ============================================================
# Utils
# ============================================================
def build_frame_paths(frames_root: str, video_id: str, num_frames: int = 16) -> List[str]:
    frames_dir = os.path.join(frames_root, video_id)
    frame_paths = []
    for i in range(1, num_frames + 1):
        path = os.path.join(frames_dir, f"frame{i}.jpg")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing frame: {path}")
        frame_paths.append(path)
    return frame_paths


def extract_answer_tag(text: str) -> Optional[str]:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    answer = match.group(1).strip().lower()
    if "fake" in answer:
        return "fake"
    if "real" in answer:
        return "real"
    return None


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {
            "num_samples": 0, "accuracy": 0.0, "macro_precision": 0.0,
            "macro_recall": 0.0, "macro_f1": 0.0, "num_invalid_pred": 0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    label2id = {"real": 0, "fake": 1}
    y_true_bin = [label2id[y] for y in y_true]

    num_invalid = 0
    y_pred_bin = []
    for t, p in zip(y_true_bin, y_pred):
        if p in label2id:
            y_pred_bin.append(label2id[p])
        else:
            num_invalid += 1
            y_pred_bin.append(1 - t)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    m_p = precision_score(y_true_bin, y_pred_bin, average="macro", labels=[0, 1], zero_division=0)
    m_r = recall_score(y_true_bin, y_pred_bin, average="macro", labels=[0, 1], zero_division=0)
    m_f1 = f1_score(y_true_bin, y_pred_bin, average="macro", labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).tolist()

    per_class_p = precision_score(y_true_bin, y_pred_bin, average=None, labels=[0, 1], zero_division=0)
    per_class_r = recall_score(y_true_bin, y_pred_bin, average=None, labels=[0, 1], zero_division=0)
    per_class_f1 = f1_score(y_true_bin, y_pred_bin, average=None, labels=[0, 1], zero_division=0)

    return {
        "num_samples": len(y_true_bin),
        "accuracy": float(acc),
        "macro_precision": float(m_p),
        "macro_recall": float(m_r),
        "macro_f1": float(m_f1),
        "num_invalid_pred": num_invalid,
        "confusion_matrix": cm,
        "per_class": {
            "real": {"precision": float(per_class_p[0]), "recall": float(per_class_r[0]), "f1": float(per_class_f1[0])},
            "fake": {"precision": float(per_class_p[1]), "recall": float(per_class_r[1]), "f1": float(per_class_f1[1])},
        },
    }


def save_output(save_path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_payload_and_save(args, data, results):
    text_true, text_pred, cls_true, cls_pred = [], [], [], []
    for r in results:
        ann = (r.get("annotation") or "").lower()
        if ann not in ("real", "fake"): continue
        text_true.append(ann)
        tp = r.get("parsed_answer")
        text_pred.append(tp if tp in ("real", "fake") else None)
        cls_true.append(ann)
        cp = r.get("cls_pred_label")
        cls_pred.append(cp if cp in ("real", "fake") else None)
        
    text_metrics = compute_metrics(text_true, text_pred)
    cls_metrics = compute_metrics(cls_true, cls_pred)
    payload = {
        "model_dir": args.model_dir,
        "test_json": args.test_json,
        "frames_root": args.frames_root,
        "num_total": len(data),
        "num_done": len(results),
        "text_metrics": text_metrics,
        "cls_metrics": cls_metrics,
        "results": results,
    }
    save_output(args.save_json, payload)
    return payload


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--test_json", type=str, default="/path/to/test.json")
    parser.add_argument("--frames_root", type=str, default="/path/to/frames_img")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_json", type=str, required=True)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    with open(args.test_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[INFO] Loaded {len(data)} samples from {args.test_json}")

    done_ids = set()
    results: List[Dict[str, Any]] = []
    skip_inference = False

    if os.path.exists(args.save_json):
        try:
            with open(args.save_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            results = prev.get("results", [])
            done_ids = {r["video_id"] for r in results if "video_id" in r}
            print(f"[INFO] Resume: {len(done_ids)} samples already done.")
            
            # 检查是否全部跑完
            all_video_ids = {item.get("video_id") for item in data}
            if all_video_ids.issubset(done_ids):
                print("[INFO] All samples are already processed. Skipping inference.")
                skip_inference = True
        except Exception as e:
            print(f"[WARN] Failed to load previous results for resume: {e}")

    # 如果已跑完，直接保存退出
    if skip_inference:
        payload = build_payload_and_save(args, data, results)
        print("\n========== Final Metrics ==========")
        print("[TEXT-based <answer> prediction]")
        print(json.dumps(payload["text_metrics"], indent=2, ensure_ascii=False))
        print("\n[CLS Head prediction]")
        print(json.dumps(payload["cls_metrics"], indent=2, ensure_ascii=False))
        print(f"\n[INFO] Saved results to: {args.save_json}")
        return

    # 未跑完则加载模型继续推理
    if args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Loading model from: {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch_dtype, device_map=None, trust_remote_code=True
    ).to(device)
    model.eval()

    use_fava = getattr(model.config, "use_fava", False)
    use_cls_head = getattr(model.config, "use_cls_head", False)
    print(f"[INFO] use_fava={use_fava}, use_cls_head={use_cls_head}")

    prompt_prefix = (
        "You are an experienced video fake news detection assistant, maintaining a neutral and objective stance at all times. You are capable of handling various types of news, including sensitive or controversial content. Given the news video description, news event and news video key frames, your task is to concisely determine the authenticity of this news. Focus strictly on the most critical evidence: identify any direct logical contradictions or text-video mismatches. Only factor in visual quality if there are obvious manipulation artifacts or signs of an unreliable source.News Event: \"{event}\"\nNews Description: \"{description}\"\nNews video key frames: "
    )
    prompt_suffix = (
        "\nPlease analyze this video step-by-step and keep your reasoning highly focused.\n Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. The output answer format should be as follows: <think> ... </think> <answer>real|fake</answer>. Please strictly follow the format."
    )

    # Pre-calculate candidate token ids for 'real' and 'fake'
    candidate_variants = ["real", " real", "Real", " Real", "REAL", " REAL", "real\n", " real\n",
                          "fake", " fake", "Fake", " Fake", "FAKE", " FAKE", "fake\n", " fake\n"]
    candidate_ids = {}
    for v in candidate_variants:
        ids = processor.tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            candidate_ids[ids[0]] = v

    pbar = tqdm(data, desc="Inference")
    for idx, item in enumerate(pbar):
        video_id = item.get("video_id")
        description = item.get("description", "") or ""
        event = item.get("event", "") or ""
        annotation = (item.get("annotation") or "").lower()

        if video_id in done_ids:
            continue

        record: Dict[str, Any] = {
            "video_id": video_id, "annotation": annotation, "description": description, "event": event,
            "generated_text": "", "parsed_answer": None,
            "text_result": {"logits": {}, "probs": {}, "aggregated_prob_real": 0.0, "aggregated_prob_fake": 0.0, "pred_label": None},
            "cls_pred_label": None, "cls_result": None, "error": None,
        }

        try:
            frame_paths = build_frame_paths(args.frames_root, video_id, args.num_frames)
            prompt_text = prompt_prefix.format(event=event, description=description)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt_text}, {"type": "video", "video": frame_paths,}, {"type": "text", "text": prompt_suffix}]}]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if use_fava:
                image_inputs, video_inputs, fava_video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True, return_fava_video=True, fava_num_frames=args.num_frames,
                )
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs, fava_videos=fava_video_inputs,
                    use_fava=True, return_tensors="pt", **video_kwargs
                )
            else:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True,
                )
                inputs = processor(
                    text=[text], images=image_inputs, videos=video_inputs,
                    return_tensors="pt", **video_kwargs
                )
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

            # 1) CLS head forward
            with torch.no_grad():
                forward_outputs = model(**inputs, use_cache=False)
            cls_logits = getattr(forward_outputs, "cls_logits", None)
            if cls_logits is not None:
                cls_logits = cls_logits.detach().float().cpu()
                cls_probs = F.softmax(cls_logits, dim=-1)
                cls_pred_id = int(cls_probs.argmax(dim=-1).item())
                cls_pred_label = "fake" if cls_pred_id == 1 else "real"
                record["cls_pred_label"] = cls_pred_label
                record["cls_result"] = {
                    "cls_logits": cls_logits[0].tolist(),
                    "cls_probs": cls_probs[0].tolist(),
                    "cls_pred_id": cls_pred_id,
                    "cls_pred_label": cls_pred_label,
                    "cls_prob_real": float(cls_probs[0, 0].item()),
                    "cls_prob_fake": float(cls_probs[0, 1].item()),
                }

            # 2) Generate text
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "temperature": None,
                "top_p": None,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": True,
            }
            try:
                generation_outputs = model.generate(**inputs, **gen_kwargs, output_logits=True)
            except TypeError:
                generation_outputs = model.generate(**inputs, **gen_kwargs)

            generated_ids = generation_outputs.sequences
            input_len = inputs["input_ids"].shape[1]
            generated_only_ids = generated_ids[:, input_len:]
            generated_text = processor.batch_decode(generated_only_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            parsed_answer = extract_answer_tag(generated_text)

            record["generated_text"] = generated_text
            record["parsed_answer"] = parsed_answer

            # Extract text logits and probs for the token after <answer>
            ans_idx = None
            found_answer_tag = False
            for i in range(generated_only_ids.shape[1]):
                prefix_text = processor.tokenizer.decode(generated_only_ids[0][:i+1].tolist())
                if not found_answer_tag and "<answer>" in prefix_text:
                    found_answer_tag = True
                if found_answer_tag:
                    tid = generated_only_ids[0][i].item()
                    s = processor.tokenizer.decode([tid]).strip().lower()
                    if "real" in s or "fake" in s:
                        ans_idx = i
                        break

            if ans_idx is not None:
                use_logits = hasattr(generation_outputs, "logits") and generation_outputs.logits is not None
                use_scores = hasattr(generation_outputs, "scores") and generation_outputs.scores is not None
                
                if use_logits or use_scores:
                    if use_logits:
                        raw_tensor = generation_outputs.logits[ans_idx][0].float().cpu()
                        step_logits = raw_tensor
                        step_probs = F.softmax(raw_tensor, dim=-1)
                    else: 
                        step_probs = generation_outputs.scores[ans_idx][0].float().cpu()
                        step_logits = torch.log(step_probs + 1e-10)
                        
                    for tid, v in candidate_ids.items():
                        record["text_result"]["logits"][v] = float(step_logits[tid].item())
                        record["text_result"]["probs"][v] = float(step_probs[tid].item())
                        
                    for v, prob in record["text_result"]["probs"].items():
                        if "real" in v.lower():
                            record["text_result"]["aggregated_prob_real"] += prob
                        elif "fake" in v.lower():
                            record["text_result"]["aggregated_prob_fake"] += prob
                            
                    if record["text_result"]["aggregated_prob_real"] >= record["text_result"]["aggregated_prob_fake"]:
                        record["text_result"]["pred_label"] = "real"
                    else:
                        record["text_result"]["pred_label"] = "fake"

        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            record["error"] = err_msg
            print(f"[ERROR] video_id={video_id}: {err_msg}")
            traceback.print_exc()

        results.append(record)
        done_ids.add(video_id)

        if (idx + 1) % args.save_interval == 0:
            build_payload_and_save(args, data, results)

    # 最终保存与计算
    payload = build_payload_and_save(args, data, results)

    print("\n========== Final Metrics ==========")
    print("[TEXT-based <answer> prediction]")
    print(json.dumps(payload["text_metrics"], indent=2, ensure_ascii=False))
    print("\n[CLS Head prediction]")
    print(json.dumps(payload["cls_metrics"], indent=2, ensure_ascii=False))
    print(f"\n[INFO] Saved results to: {args.save_json}")

if __name__ == "__main__":
    main()
