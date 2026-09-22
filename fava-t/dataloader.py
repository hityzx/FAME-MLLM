# dataloader.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


try:
    BICUBIC = Image.Resampling.BICUBIC
    BILINEAR = Image.Resampling.BILINEAR
except AttributeError:
    BICUBIC = Image.BICUBIC
    BILINEAR = Image.BILINEAR


# Preserve the original six-class output layout for checkpoint compatibility.
# The two local-artifact classes remain negative in type-rank training.
TYPE_NAMES = [
    "jpeg_compression",
    "resize_resample",
    "crop_resize",
    "local_blur",
    "local_recompression",
    "noise",
]


TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPE_NAMES)}


@dataclass
class AugmentResult:
    frames: List[Image.Image]
    type_label: np.ndarray
    severity: float
    transform_names: List[str]


class FakeVideoFramesDataset(Dataset):
    """
    Dataset for propagation-aware FaVA pretraining.

    Expected JSON item:
    {
        "video_id": "...",
        "description": "...",
        "event": "...",
        "path": "...",
        ...
    }

    Expected frames:
        frames_root / video_id / frame1.jpg ... frame16.jpg

    Returned dict:
    {
        "video_id": str,
        "frames": FloatTensor [V, T, 3, H, W], values in [0, 1],
        "type_labels": FloatTensor [V, C],
        "severity": FloatTensor [V],
        "description": Optional[str],
        "event": Optional[str],
    }

    Type-rank mode always returns V = 3: clean, light, heavy.
    """

    def __init__(
        self,
        json_path: Union[str, Path, Sequence[Union[str, Path]]],
        frames_root: Union[str, Path],
        num_frames: int = 16,
        image_size: int = 224,
        patch_size: int = 16,
        is_train: bool = True,
        seed: int = 42,
        strict: bool = False,
        include_text: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size. "
                f"Got image_size={image_size}, patch_size={patch_size}."
            )

        self.items = self._load_json_items(json_path)
        self.frames_root = Path(frames_root)
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.is_train = bool(is_train)
        self.seed = int(seed)
        self.strict = bool(strict)
        self.include_text = bool(include_text)

        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None

        if len(self.items) == 0:
            raise ValueError(f"No items loaded from {json_path}.")

        if self.cache_dir is not None:
            cached = sum(
                1 for item in self.items
                if self._cache_path(str(item.get("video_id", ""))).exists()
            )
            print(
                f"[FrameCache] {cached}/{len(self.items)} videos already cached "
                f"in {self.cache_dir}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]

        video_id = str(item.get("video_id", ""))
        if not video_id:
            raise KeyError(f"Missing video_id in item index={index}.")

        base_frames = self._load_frames(video_id)
        rng = self._get_rng(index)
        variants = [
            self._make_clean_variant(base_frames),
            self._make_light_variant(base_frames, rng),
            self._make_heavy_variant(base_frames, rng),
        ]

        frame_tensors = [self._frames_to_tensor(v.frames) for v in variants]

        frames = torch.stack(frame_tensors, dim=0)  # [V, T, 3, H, W]
        type_labels = torch.tensor(
            np.stack([v.type_label for v in variants], axis=0),
            dtype=torch.float32,
        )
        severity = torch.tensor(
            [v.severity for v in variants],
            dtype=torch.float32,
        )
        output: Dict[str, Any] = {
            "video_id": video_id,
            "frames": frames,
            "type_labels": type_labels,
            "severity": severity,
        }

        if self.include_text:
            output["description"] = str(item.get("description", ""))
            output["event"] = str(item.get("event", ""))

        return output

    # ------------------------------------------------------------------
    # JSON and frame loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_json_items(
        json_path: Union[str, Path, Sequence[Union[str, Path]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(json_path, (str, Path)):
            paths = [json_path]
        else:
            paths = list(json_path)

        all_items: List[Dict[str, Any]] = []

        for p in paths:
            p = Path(p)
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict):
                candidate_keys = ["data", "items", "annotations", "videos"]
                items = None
                for key in candidate_keys:
                    if key in obj and isinstance(obj[key], list):
                        items = obj[key]
                        break
                if items is None:
                    raise ValueError(
                        f"JSON file {p} is a dict but does not contain "
                        f"one of {candidate_keys} as a list."
                    )
            else:
                raise ValueError(f"Unsupported JSON format in {p}.")

            for x in items:
                if isinstance(x, dict):
                    all_items.append(x)

        return all_items

    def _load_frames(self, video_id: str) -> List[Image.Image]:
        if self.cache_dir is not None:
            cached = self._load_frames_from_cache(video_id)
            if cached is not None:
                return cached

        frames = self._load_frames_from_disk(video_id)

        if self.cache_dir is not None:
            self._save_frames_to_cache(video_id, frames)

        return frames

    def _cache_path(self, video_id: str) -> Path:
        return self.cache_dir / f"{video_id}.npy"

    def _load_frames_from_cache(self, video_id: str) -> Optional[List[Image.Image]]:
        cache_path = self._cache_path(video_id)
        if not cache_path.exists():
            return None

        arr = np.load(cache_path)  # [T, H, W, C] uint8
        return [Image.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]

    def _save_frames_to_cache(
        self, video_id: str, frames: List[Image.Image]
    ) -> None:
        arrs = [np.asarray(img, dtype=np.uint8) for img in frames]
        arr = np.stack(arrs, axis=0)  # [T, H, W, C]
        np.save(self._cache_path(video_id), arr)

    def _load_frames_from_disk(self, video_id: str) -> List[Image.Image]:
        video_frame_dir = self.frames_root / video_id

        frames: List[Image.Image] = []
        last_valid: Optional[Image.Image] = None

        for i in range(1, self.num_frames + 1):
            fp = video_frame_dir / f"frame{i}.jpg"

            if fp.exists():
                img = self._open_and_resize(fp)
                last_valid = img
            else:
                if self.strict:
                    raise FileNotFoundError(f"Missing frame: {fp}")

                if last_valid is not None:
                    img = last_valid.copy()
                else:
                    img = Image.new(
                        "RGB",
                        (self.image_size, self.image_size),
                        color=(0, 0, 0),
                    )

            frames.append(img)

        return frames

    def _open_and_resize(self, fp: Path) -> Image.Image:
        with Image.open(fp) as img:
            img = img.convert("RGB").copy()

        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), BICUBIC)

        return img

    def _frames_to_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        arrs = []
        for img in frames:
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]
            arrs.append(arr)

        out = np.stack(arrs, axis=0)  # [T, 3, H, W]
        return torch.from_numpy(out).float()

    def _get_rng(self, index: int) -> random.Random:
        if self.is_train:
            return random
        return random.Random(self.seed + index * 10007)

    # ------------------------------------------------------------------
    # Variant construction
    # ------------------------------------------------------------------

    def _empty_type_label(self) -> np.ndarray:
        return np.zeros(len(TYPE_NAMES), dtype=np.float32)

    def _make_clean_variant(self, frames: List[Image.Image]) -> AugmentResult:
        return AugmentResult(
            frames=[x.copy() for x in frames],
            type_label=self._empty_type_label(),
            severity=0.0,
            transform_names=["clean"],
        )

    def _make_light_variant(
        self,
        frames: List[Image.Image],
        rng: random.Random,
    ) -> AugmentResult:
        frames = [x.copy() for x in frames]
        type_label = self._empty_type_label()
        transform_names: List[str] = []

        op = rng.choice(["jpeg_compression", "resize_resample", "crop_resize"])
        frames = self._apply_global_op(
            frames=frames,
            op=op,
            level="light",
            type_label=type_label,
            rng=rng,
            transform_names=transform_names,
        )

        return AugmentResult(
            frames=frames,
            type_label=type_label,
            severity=1.0,
            transform_names=transform_names,
        )

    def _make_heavy_variant(
        self,
        frames: List[Image.Image],
        rng: random.Random,
    ) -> AugmentResult:
        frames = [x.copy() for x in frames]
        type_label = self._empty_type_label()
        transform_names: List[str] = []

        ops = ["jpeg_compression", "resize_resample", "crop_resize", "noise"]
        selected_ops = rng.sample(ops, k=2)

        for op in selected_ops:
            frames = self._apply_global_op(
                frames=frames,
                op=op,
                level="heavy",
                type_label=type_label,
                rng=rng,
                transform_names=transform_names,
            )

        return AugmentResult(
            frames=frames,
            type_label=type_label,
            severity=2.0,
            transform_names=transform_names,
        )

    # ------------------------------------------------------------------
    # Global propagation transformations
    # ------------------------------------------------------------------

    def _apply_global_op(
        self,
        frames: List[Image.Image],
        op: str,
        level: str,
        type_label: np.ndarray,
        rng: random.Random,
        transform_names: List[str],
    ) -> List[Image.Image]:
        if op == "jpeg_compression":
            if level == "light":
                quality = rng.randint(55, 80)
                repeats = 1
            else:
                quality = rng.randint(18, 45)
                repeats = rng.randint(2, 4)

            out = frames
            for _ in range(repeats):
                out = [self._jpeg_compress(img, quality=quality) for img in out]

            type_label[TYPE_TO_ID["jpeg_compression"]] = 1.0
            transform_names.append(f"jpeg_q{quality}_r{repeats}")
            return out

        if op == "resize_resample":
            if level == "light":
                scale = rng.uniform(0.65, 0.90)
            else:
                scale = rng.uniform(0.35, 0.65)

            out = [self._resize_resample(img, scale=scale) for img in frames]

            type_label[TYPE_TO_ID["resize_resample"]] = 1.0
            transform_names.append(f"resize_scale_{scale:.2f}")
            return out

        if op == "crop_resize":
            if level == "light":
                crop_ratio = rng.uniform(0.82, 0.95)
            else:
                crop_ratio = rng.uniform(0.60, 0.82)

            out = [
                self._crop_and_resize_back(img, crop_ratio=crop_ratio, rng=rng)
                for img in frames
            ]

            type_label[TYPE_TO_ID["crop_resize"]] = 1.0
            transform_names.append(f"crop_ratio_{crop_ratio:.2f}")
            return out

        if op == "noise":
            if level == "light":
                std = rng.uniform(2.0, 5.0)
            else:
                std = rng.uniform(6.0, 14.0)

            out = [self._add_noise(img, std=std, rng=rng) for img in frames]

            type_label[TYPE_TO_ID["noise"]] = 1.0
            transform_names.append(f"noise_std_{std:.1f}")
            return out

        raise ValueError(f"Unsupported global op: {op}")

    @staticmethod
    def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=int(quality), optimize=False)
        buffer.seek(0)
        with Image.open(buffer) as out:
            return out.convert("RGB").copy()

    def _resize_resample(self, img: Image.Image, scale: float) -> Image.Image:
        w, h = img.size
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))

        small = img.resize((new_w, new_h), BILINEAR)
        out = small.resize((w, h), BICUBIC)
        return out.convert("RGB")

    def _crop_and_resize_back(
        self,
        img: Image.Image,
        crop_ratio: float,
        rng: random.Random,
    ) -> Image.Image:
        w, h = img.size
        crop_w = max(8, int(round(w * crop_ratio)))
        crop_h = max(8, int(round(h * crop_ratio)))

        if crop_w >= w or crop_h >= h:
            return img.copy()

        x0 = rng.randint(0, w - crop_w)
        y0 = rng.randint(0, h - crop_h)

        crop = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        out = crop.resize((w, h), BICUBIC)
        return out.convert("RGB")

    @staticmethod
    def _add_noise(
        img: Image.Image,
        std: float,
        rng: random.Random,
    ) -> Image.Image:
        seed = rng.randint(0, 2**31 - 1)
        np_rng = np.random.default_rng(seed)

        arr = np.asarray(img, dtype=np.float32).copy()
        noise = np_rng.normal(loc=0.0, scale=float(std), size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(arr, mode="RGB")

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_fava_type_rank_dataloader(
    json_path: Union[str, Path, Sequence[Union[str, Path]]],
    frames_root: Union[str, Path],
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    num_frames: int = 16,
    image_size: int = 224,
    patch_size: int = 16,
    is_train: bool = True,
    seed: int = 42,
    strict: bool = False,
    include_text: bool = True,
    pin_memory: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> DataLoader:
    dataset = FakeVideoFramesDataset(
        json_path=json_path,
        frames_root=frames_root,
        num_frames=num_frames,
        image_size=image_size,
        patch_size=patch_size,
        is_train=is_train,
        seed=seed,
        strict=strict,
        include_text=include_text,
        cache_dir=cache_dir,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if is_train else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last if is_train else False,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    return loader


    print("transform_names:", batch["transform_names"])
