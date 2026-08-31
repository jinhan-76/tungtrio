"""
Dataset for the AI-image detector.

Expects a directory layout:

    data/
      train/
        real/*.jpg
        fake/*.jpg
      val/
        real/*.jpg
        fake/*.jpg
      test/
        real/*.jpg
        fake/*.jpg

`manifest.csv` (produced by scripts/prepare_dataset.py) additionally records
the generator family for each fake image, used later for per-family error
analysis:

    image_path,label,generator_family
    data/train/fake/0001.jpg,1,sdv1.4
    data/train/real/0002.jpg,0,

Each __getitem__ returns BOTH:
  - `pixel_values`: ImageNet-normalized tensor, for the DINOv2 backbone
  - `raw_pixels`: unnormalized [0,1] tensor, for the SRM branch (which needs
    true pixel statistics — normalizing would distort the residual signal)
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class Sample:
    path: str
    label: int  # 0 = real, 1 = fake (AIGC)
    generator_family: str = ""


def _scan_split_dir(split_dir: Path) -> list[Sample]:
    samples = []
    for label_name, label in [("real", 0), ("fake", 1)]:
        d = split_dir / label_name
        if not d.exists():
            continue
        for p in sorted(d.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append(Sample(path=str(p), label=label))
    return samples


def load_manifest(manifest_path: str) -> dict[str, str]:
    """path -> generator_family lookup, if a manifest.csv is present."""
    lookup = {}
    if not os.path.exists(manifest_path):
        return lookup
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            lookup[row["image_path"]] = row.get("generator_family", "")
    return lookup


class AIImageDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = 224,
        manifest_path: Optional[str] = None,
        online_augment: Optional[Callable] = None,
    ):
        """
        online_augment: optional callable(PIL.Image) -> PIL.Image applied
            BEFORE tensor conversion — used for JPEG/blur/noise/jitter during
            training (see augmentations.py). Leave None for val/test (clean
            eval) or pass a fixed degradation function for robustness eval.
        """
        self.split_dir = Path(data_root) / split
        self.samples = _scan_split_dir(self.split_dir)
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.split_dir}. Did you run "
                f"scripts/prepare_dataset.py first?"
            )

        manifest = load_manifest(manifest_path) if manifest_path else {}
        for s in self.samples:
            s.generator_family = manifest.get(s.path, "")

        self.online_augment = online_augment
        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()  # -> [0,1]
        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")

        if self.online_augment is not None:
            img = self.online_augment(img)

        img = self.resize(img)
        raw = self.to_tensor(img)              # [0,1], for SRM branch
        normalized = self.normalize(raw.clone())  # for DINOv2

        return {
            "pixel_values": normalized,
            "raw_pixels": raw,
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "path": sample.path,
            "generator_family": sample.generator_family,
        }
