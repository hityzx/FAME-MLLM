# train.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from dataloader import build_fava_type_rank_dataloader
from model import (
    PropagationAwareFaVA,
    compute_fava_type_rank_losses,
)


# ----------------------------------------------------------------------
# Basic utilities
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ----------------------------------------------------------------------
# Scheduler
# ----------------------------------------------------------------------

class WarmupCosineScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.05,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = max(0, int(warmup_steps))
        self.total_steps = max(1, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        lr_mult = self._get_lr_mult(self.step_num)

        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * lr_mult

    def _get_lr_mult(self, step: int) -> float:
        if self.warmup_steps > 0 and step <= self.warmup_steps:
            return step / self.warmup_steps

        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def state_dict(self) -> Dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "base_lrs": self.base_lrs,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.warmup_steps = state["warmup_steps"]
        self.total_steps = state["total_steps"]
        self.min_lr_ratio = state["min_lr_ratio"]
        self.base_lrs = state["base_lrs"]
        self.step_num = state["step_num"]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

@torch.no_grad()
def multilabel_f1_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    logits: [B, V, C] or [N, C]
    labels: [B, V, C] or [N, C]
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    labels = labels.float()

    preds = preds.reshape(-1, preds.shape[-1])
    labels = labels.reshape(-1, labels.shape[-1])

    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1.0 - labels)).sum(dim=0)
    fn = ((1.0 - preds) * labels).sum(dim=0)

    eps = 1e-8

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    support = labels.sum(dim=0)
    valid = support > 0

    if valid.sum() > 0:
        macro_f1 = f1[valid].mean().item()
    else:
        macro_f1 = 0.0

    tp_micro = tp.sum()
    fp_micro = fp.sum()
    fn_micro = fn.sum()

    micro_p = tp_micro / (tp_micro + fp_micro + eps)
    micro_r = tp_micro / (tp_micro + fn_micro + eps)
    micro_f1 = 2.0 * micro_p * micro_r / (micro_p + micro_r + eps)

    exact_match = (preds == labels).all(dim=-1).float().mean().item()
    micro_acc = (preds == labels).float().mean().item()

    return {
        "type_macro_f1": float(macro_f1),
        "type_micro_f1": float(micro_f1.item()),
        "type_micro_acc": float(micro_acc),
        "type_exact_match": float(exact_match),
    }


@torch.no_grad()
def rank_correct_count(
    scores: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[int, int]:
    """
    scores: [B, V]
    labels: [B, V]
    """
    labels_i = labels.unsqueeze(2)
    labels_j = labels.unsqueeze(1)

    scores_i = scores.unsqueeze(2)
    scores_j = scores.unsqueeze(1)

    pair_mask = labels_j > labels_i

    total = int(pair_mask.sum().item())
    if total == 0:
        return 0, 0

    correct = int(((scores_j > scores_i) & pair_mask).sum().item())
    return correct, total


@torch.no_grad()
def token_offdiag_cosine(tokens: torch.Tensor) -> float:
    """
    tokens: [B, V, N, D] or [B, N, D]
    A high value close to 1.0 means token collapse.
    """
    if tokens.dim() == 4:
        b, v, n, d = tokens.shape
        tokens = tokens.reshape(b * v, n, d)
    elif tokens.dim() == 3:
        pass
    else:
        return 0.0

    tokens = torch.nn.functional.normalize(tokens, dim=-1)
    sim = tokens @ tokens.transpose(1, 2)

    n = sim.shape[-1]
    eye = torch.eye(n, device=sim.device, dtype=torch.bool).unsqueeze(0)
    off_diag = sim.masked_select(~eye)

    return float(off_diag.mean().item())


def compute_selection_score(metrics: Dict[str, float]) -> float:
    """Checkpoint score for type-rank training."""
    return 0.7 * metrics["type_macro_f1"] + 0.3 * metrics["rank_acc"]


# ----------------------------------------------------------------------
# Loss weight schedule
# ----------------------------------------------------------------------

def get_epoch_loss_weights(args: argparse.Namespace, epoch: int) -> Dict[str, float]:
    """
    Curriculum schedule.

    The optional warm-up enables rank loss from ``rank_start_epoch``.
    """
    type_weight = args.type_weight

    if epoch >= args.rank_start_epoch:
        rank_weight = args.rank_weight
    else:
        rank_weight = 0.0

    return {
        "type_weight": type_weight,
        "rank_weight": rank_weight,
        "low_weight": args.low_weight,
        "mask_div_weight": args.mask_div_weight,
        "token_div_weight": args.token_div_weight,
    }


# ----------------------------------------------------------------------
# Train / validate
# ----------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()

    meters = {
        "loss": AverageMeter(),
        "type_loss": AverageMeter(),
        "rank_loss": AverageMeter(),
        "low_loss": AverageMeter(),
        "mask_div_loss": AverageMeter(),
        "token_div_loss": AverageMeter(),
    }

    weights = get_epoch_loss_weights(args, epoch)

    start = time.time()

    for step, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(batch["frames"])

            loss_dict = compute_fava_type_rank_losses(
                outputs=outputs,
                batch=batch,
                type_weight=weights["type_weight"],
                rank_weight=weights["rank_weight"],
                low_weight=weights["low_weight"],
                mask_div_weight=weights["mask_div_weight"],
                token_div_weight=weights["token_div_weight"],
                rank_margin=args.rank_margin,
            )

            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected: {loss.item()}")

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        batch_size = batch["frames"].shape[0]

        for k in meters.keys():
            if k in loss_dict:
                meters[k].update(float(loss_dict[k].item()), n=batch_size)

        if step % args.log_interval == 0:
            elapsed = time.time() - start
            msg = (
                f"Epoch [{epoch}/{args.epochs}] "
                f"Step [{step}/{len(loader)}] "
                f"lr={get_lr(optimizer):.3e} "
                f"loss={meters['loss'].avg:.4f} "
                f"type={meters['type_loss'].avg:.4f} "
                f"rank={meters['rank_loss'].avg:.4f} "
                f"time={elapsed:.1f}s"
            )
            print(msg, flush=True)

    return {k: v.avg for k, v in meters.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: Iterable[Dict[str, Any]],
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()

    loss_meter = AverageMeter()

    all_type_logits: List[torch.Tensor] = []
    all_type_labels: List[torch.Tensor] = []

    rank_correct = 0
    rank_total = 0

    token_cos_meter = AverageMeter()

    weights = get_epoch_loss_weights(args, epoch)

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(batch["frames"])

            loss_dict = compute_fava_type_rank_losses(
                outputs=outputs,
                batch=batch,
                type_weight=weights["type_weight"],
                rank_weight=weights["rank_weight"],
                low_weight=weights["low_weight"],
                mask_div_weight=weights["mask_div_weight"],
                token_div_weight=weights["token_div_weight"],
                rank_margin=args.rank_margin,
            )

        batch_size = batch["frames"].shape[0]
        loss_meter.update(float(loss_dict["loss"].item()), n=batch_size)

        all_type_logits.append(outputs["type_logits"].detach().float().cpu())
        all_type_labels.append(batch["type_labels"].detach().float().cpu())

        rc, rt = rank_correct_count(
            outputs["severity_score"].detach().float(),
            batch["severity"].detach().float(),
        )
        rank_correct += rc
        rank_total += rt

        token_cos = token_offdiag_cosine(outputs["tokens"].detach().float())
        token_cos_meter.update(token_cos, n=batch_size)

    type_logits = torch.cat(all_type_logits, dim=0)
    type_labels = torch.cat(all_type_labels, dim=0)

    type_metrics = multilabel_f1_from_logits(
        logits=type_logits,
        labels=type_labels,
        threshold=args.type_threshold,
    )

    rank_acc = rank_correct / rank_total if rank_total > 0 else 0.0
    metrics = {
        "val_loss": loss_meter.avg,
        **type_metrics,
        "rank_acc": float(rank_acc),
        "token_offdiag_cos": token_cos_meter.avg,
    }

    metrics["selection_score"] = compute_selection_score(metrics)

    return metrics


# ----------------------------------------------------------------------
# Checkpointing and logging
# ----------------------------------------------------------------------

def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupCosineScheduler],
    scaler: GradScaler,
    epoch: int,
    best_score: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": epoch,
        "best_score": best_score,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }

    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()

    torch.save(ckpt, save_path)


def load_checkpoint(
    ckpt_path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[WarmupCosineScheduler] = None,
    scaler: Optional[GradScaler] = None,
    load_optimizer: bool = True,
    map_location: str | torch.device = "cpu",
) -> Tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location=map_location)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)

    if len(missing) > 0:
        print(f"[Warning] Missing keys when loading model: {missing}")
    if len(unexpected) > 0:
        print(f"[Warning] Unexpected keys when loading model: {unexpected}")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(ckpt.get("best_score", -1e9))

    if load_optimizer and optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if load_optimizer and scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if load_optimizer and scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    return start_epoch, best_score


def append_csv_log(csv_path: Path, row: Dict[str, Any]) -> None:
    exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_args(args: argparse.Namespace, output_dir: Path) -> None:
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Propagation-aware FaVA type-rank training"
    )

    # Data
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="If set, preprocessed 224x224 frames are cached to/loaded from this directory as .npy files.",
    )
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs/fava_type_rank")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--load_model",
        type=str,
        default="",
        help="Load model weights without optimizer or scheduler state.",
    )

    # Model
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_bands", type=int, default=4)
    parser.add_argument("--token_dim", type=int, default=256)
    parser.add_argument("--num_artifact_tokens", type=int, default=128)
    parser.add_argument("--num_query_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=1024)
    parser.add_argument(
        "--peak_selection",
        type=str,
        default="topk",
        choices=["topk", "soft"],
    )
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")

    # Loss weights
    parser.add_argument("--type_weight", type=float, default=1.0)
    parser.add_argument("--rank_weight", type=float, default=0.3)
    parser.add_argument("--low_weight", type=float, default=0.0)
    parser.add_argument("--mask_div_weight", type=float, default=0.0)
    parser.add_argument("--token_div_weight", type=float, default=0.0)
    parser.add_argument("--rank_margin", type=float, default=0.2)

    # Curriculum
    parser.add_argument(
        "--rank_start_epoch",
        type=int,
        default=0,
        help="Epoch index from which rank loss is enabled.",
    )

    # Validation / logging
    parser.add_argument("--type_threshold", type=float, default=0.5)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=0)

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = ensure_dir(args.output_dir)
    save_args(args, output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_dir = args.cache_dir if args.cache_dir else None

    train_loader = build_fava_type_rank_dataloader(
        json_path=args.train_json,
        frames_root=args.frames_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        num_frames=args.num_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        is_train=True,
        seed=args.seed,
        strict=False,
        include_text=False,
        pin_memory=True,
        cache_dir=cache_dir,
    )

    val_loader = build_fava_type_rank_dataloader(
        json_path=args.val_json,
        frames_root=args.frames_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        num_frames=args.num_frames,
        image_size=args.image_size,
        patch_size=args.patch_size,
        is_train=False,
        seed=args.seed,
        strict=False,
        include_text=False,
        pin_memory=True,
        cache_dir=cache_dir,
    )

    model = PropagationAwareFaVA(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_frames=args.num_frames,
        num_bands=args.num_bands,
        token_dim=args.token_dim,
        num_artifact_tokens=args.num_artifact_tokens,
        num_query_blocks=args.num_query_blocks,
        num_heads=args.num_heads,
        top_k=args.top_k,
        dropout=args.dropout,
        peak_selection=args.peak_selection,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch = 1
    best_score = -1e9

    if args.resume:
        print(f"Resume full checkpoint from: {args.resume}")
        start_epoch, best_score = load_checkpoint(
            ckpt_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            load_optimizer=True,
            map_location=device,
        )
    elif args.load_model:
        print(f"Load model weights only from: {args.load_model}")
        _, _ = load_checkpoint(
            ckpt_path=args.load_model,
            model=model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            load_optimizer=False,
            map_location=device,
        )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples:   {len(val_loader.dataset)}")
    print(f"Start epoch:   {start_epoch}")
    print(f"Output dir:    {output_dir}")

    log_csv = output_dir / "log.csv"
    log_txt = output_dir / "log.txt"

    for epoch in range(start_epoch, args.epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Loss weights: {get_epoch_loss_weights(args, epoch)}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            args=args,
        )

        score = val_metrics["selection_score"]
        is_best = score > best_score

        if is_best:
            best_score = score
            save_checkpoint(
                save_path=output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )

        save_checkpoint(
            save_path=output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_score=best_score,
            args=args,
        )

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(
                save_path=output_dir / f"epoch_{epoch}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_score=best_score,
                args=args,
            )

        row: Dict[str, Any] = {
            "epoch": epoch,
            "lr": get_lr(optimizer),
            "best_score": best_score,
            "is_best": int(is_best),
        }

        for k, v in train_metrics.items():
            row[f"train_{k}"] = v

        for k, v in val_metrics.items():
            row[k] = v

        append_csv_log(log_csv, row)

        log_msg = (
            f"Val: loss={val_metrics['val_loss']:.4f} "
            f"type_macro_f1={val_metrics['type_macro_f1']:.4f} "
            f"type_micro_f1={val_metrics['type_micro_f1']:.4f} "
            f"rank_acc={val_metrics['rank_acc']:.4f} "
            f"token_cos={val_metrics['token_offdiag_cos']:.4f} "
            f"score={score:.4f} "
            f"best={best_score:.4f}"
        )
        print(log_msg)
        with open(log_txt, "a") as f:
            f.write(log_msg + "\n")

        if is_best:
            print(f"New best checkpoint saved to: {output_dir / 'best.pt'}")

    print("Training finished.")
    print(f"Best score: {best_score:.4f}")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
