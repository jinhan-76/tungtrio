"""
Dataset adapters — a common interface over three ways training data can show up:

  1. HuggingFace Hub dataset  (e.g. saberzl/SID_Set, GenImage-style datasets, CIFAKE)
  2. Local folder tree        (e.g. data/real/*.jpg, data/fake/*.jpg, or
                                 data/<generator_name>/{real,fake}/*.jpg)
  3. CSV manifest             (image_path,label[,source] columns, paths either
                                 absolute or relative to the CSV's directory)

Every adapter yields the same thing: (image, label, source_tag)
  - image: a PIL.Image (already opened/decoded — caller is responsible for
    closing/converting as needed)
  - label: 0 (real) or 1 (fake/AI-generated) — adapters accept a `label_map`
    so callers decide how a dataset's raw label values map to this binary scheme
  - source_tag: a short string recorded in manifest.csv for later per-source
    error-analysis breakdown (e.g. "sid_set:full_synthetic", "genimage:sdv1.4",
    "" for real images). Purely informational — never affects training.

This means scripts/prepare_dataset.py doesn't need to know anything about a
specific dataset's schema; it just calls whichever adapter matches the
`--source-type` flag and gets a uniform stream of (image, label, source_tag).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

from PIL import Image

ImageLabelSource = tuple  # (PIL.Image, int, str)


# ---------------------------------------------------------------------------
# 1. HuggingFace Hub dataset
# ---------------------------------------------------------------------------
def iter_hf_dataset(
    dataset_name: str,
    split: str = "train",
    image_field: str = "image",
    label_field: str = "label",
    label_map: dict | None = None,
    streaming: bool = True,
    shuffle_seed: int = 42,
    shuffle_buffer: int = 10_000,
    config_name: str | None = None,
) -> Iterator[ImageLabelSource]:
    """
    label_map: maps the dataset's raw label values -> (binary_label, source_tag).
        e.g. for SID_Set: {0: (0, ""), 1: (1, "full_synthetic"), 2: (1, "tampered")}
        e.g. for a real/fake-only HF dataset with label 0/1 already binary:
             {0: (0, ""), 1: (1, "")}
        Raw label values not present in label_map are skipped.
    """
    from datasets import load_dataset

    if label_map is None:
        label_map = {0: (0, ""), 1: (1, "")}

    ds = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    if streaming:
        ds = ds.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)

    for example in ds:
        raw_label = example[label_field]
        if raw_label not in label_map:
            continue
        binary_label, source_tag = label_map[raw_label]
        img = example[image_field]
        if not isinstance(img, Image.Image):
            # some HF datasets store a path/bytes dict instead of a decoded image
            img = Image.open(img) if isinstance(img, (str, Path)) else Image.open(img["path"])
        yield img, binary_label, source_tag


# ---------------------------------------------------------------------------
# 2. Local folder tree
# ---------------------------------------------------------------------------
VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_local_folder(
    root: str,
    layout: str = "flat",
) -> Iterator[ImageLabelSource]:
    """
    layout='flat':   root/real/*.jpg, root/fake/*.jpg
    layout='nested': root/<source_name>/real/*.jpg, root/<source_name>/fake/*.jpg
                      (source_name becomes the source_tag for fake images, e.g. a
                       generator name if you've organized data that way)
    """
    root_path = Path(root)

    if layout == "flat":
        for label_name, label in [("real", 0), ("fake", 1)]:
            d = root_path / label_name
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in VALID_EXTS:
                    yield Image.open(p), label, ""

    elif layout == "nested":
        for source_dir in sorted(root_path.iterdir()):
            if not source_dir.is_dir():
                continue
            source_name = source_dir.name
            for label_name, label in [("real", 0), ("fake", 1)]:
                d = source_dir / label_name
                if not d.exists():
                    continue
                for p in sorted(d.iterdir()):
                    if p.suffix.lower() in VALID_EXTS:
                        tag = source_name if label == 1 else ""
                        yield Image.open(p), label, tag
    else:
        raise ValueError(f"Unknown layout: {layout} (expected 'flat' or 'nested')")


# ---------------------------------------------------------------------------
# 3. CSV manifest
# ---------------------------------------------------------------------------
def iter_csv_manifest(
    csv_path: str,
    image_col: str = "image_path",
    label_col: str = "label",
    source_col: str | None = None,
    base_dir: str | None = None,
) -> Iterator[ImageLabelSource]:
    """
    CSV must have at least `image_col` and `label_col`. `label_col` values must
    already be 0 (real) or 1 (fake) — do any raw->binary mapping before writing
    the CSV, or use iter_hf_dataset/iter_local_folder instead if your raw
    labels aren't already binary.

    base_dir: if image_path values are relative, join them to this directory
        (defaults to the CSV's own parent directory).
    """
    csv_file = Path(csv_path)
    base = Path(base_dir) if base_dir else csv_file.parent

    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            img_path = Path(row[image_col])
            if not img_path.is_absolute():
                img_path = base / img_path
            label = int(row[label_col])
            source_tag = row.get(source_col, "") if source_col else ""
            yield Image.open(img_path), label, source_tag
