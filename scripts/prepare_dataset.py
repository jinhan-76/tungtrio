"""
Generalized dataset preparation — works with ANY dataset source via the
adapters in src/data_adapters.py:

  --source-type hf      : any HuggingFace Hub dataset
  --source-type folder   : a local folder tree (real/fake, or nested by source)
  --source-type csv       : a CSV manifest of image_path,label

All three produce the same standard output layout, so train.py, inference.py,
eval_robustness.py, and error_analysis.py never need to change:

  data/{train,val,test}/{real,fake}/*.jpg
  data/manifest.csv   (image_path,label,generator_family)

("generator_family" column holds whatever `source_tag` the adapter reported —
 a generator name for folder/csv sources, or an image_type like
 "full_synthetic"/"tampered" for SID_Set-style HF datasets. It's just a label
 for later error-analysis breakdown, not something training depends on.)

------------------------------------------------------------------------------
EXAMPLES
------------------------------------------------------------------------------

# SID_Set (HuggingFace, gated — see README for huggingface-cli login steps)
python scripts/prepare_dataset.py --source-type hf \\
    --hf-dataset saberzl/SID_Set \\
    --hf-image-field image --hf-label-field label \\
    --hf-label-map "0:0:" "1:1:full_synthetic" "2:1:tampered" \\
    --per-class 3000 --out data/

# A generic HF real/fake dataset with binary labels already 0/1
python scripts/prepare_dataset.py --source-type hf \\
    --hf-dataset some/other-dataset \\
    --per-class 3000 --out data/

# Local folder: data_raw/real/*.jpg, data_raw/fake/*.jpg
python scripts/prepare_dataset.py --source-type folder \\
    --folder-root data_raw --folder-layout flat \\
    --per-class 3000 --out data/

# Local folder organized by generator: data_raw/<generator>/{real,fake}/*.jpg
python scripts/prepare_dataset.py --source-type folder \\
    --folder-root data_raw --folder-layout nested \\
    --per-class 3000 --out data/

# CSV manifest (image_path,label columns, label already 0/1)
python scripts/prepare_dataset.py --source-type csv \\
    --csv-path my_manifest.csv \\
    --per-class 3000 --out data/
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_adapters import iter_csv_manifest, iter_hf_dataset, iter_local_folder


def parse_label_map(entries: list[str]) -> dict:
    """Parses ["0:0:", "1:1:full_synthetic", "2:1:tampered"] into
    {0: (0, ""), 1: (1, "full_synthetic"), 2: (1, "tampered")}.
    Format per entry: "<raw_label>:<binary_label>:<source_tag>"
    """
    label_map = {}
    for entry in entries:
        parts = entry.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad --hf-label-map entry '{entry}', expected raw:binary:tag")
        raw, binary, tag = parts
        label_map[int(raw)] = (int(binary), tag)
    return label_map


def split_and_save(items: list[tuple], split_fracs: list[tuple[str, float]],
                    out_dir: Path, label: int, manifest_rows: list, prefix: str, seed: int):
    random.seed(seed)
    items = list(items)
    random.shuffle(items)
    label_folder = "real" if label == 0 else "fake"

    n = len(items)
    idx = 0
    for i, (split_name, frac) in enumerate(split_fracs):
        count = n - idx if i == len(split_fracs) - 1 else int(n * frac)
        chunk = items[idx: idx + count]
        idx += count

        split_dir = out_dir / split_name / label_folder
        split_dir.mkdir(parents=True, exist_ok=True)
        for j, (img, tag) in enumerate(chunk):
            fname = f"{prefix}_{idx - count + j:06d}.jpg"
            dest = split_dir / fname
            img.convert("RGB").save(dest, format="JPEG", quality=95)
            manifest_rows.append({
                "image_path": str(dest),
                "label": label,
                "generator_family": tag if label == 1 else "",
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-type", required=True, choices=["hf", "folder", "csv"])
    parser.add_argument("--per-class", type=int, default=3000,
                         help="Max real images and max fake images to collect")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    # --source-type hf
    parser.add_argument("--hf-dataset", help="HuggingFace dataset repo id, e.g. saberzl/SID_Set")
    parser.add_argument("--hf-config", default=None, help="HF dataset config name, if any")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--hf-image-field", default="image")
    parser.add_argument("--hf-label-field", default="label")
    parser.add_argument("--hf-label-map", nargs="*", default=None,
                         help='Entries like "0:0:" "1:1:full_synthetic" '
                              '(raw_label:binary_label:source_tag). '
                              'Default: {0: real, 1: fake} with no tag.')
    parser.add_argument("--hf-no-streaming", action="store_true",
                         help="Download the full dataset instead of streaming "
                              "(only use for small datasets)")

    # --source-type folder
    parser.add_argument("--folder-root", type=Path)
    parser.add_argument("--folder-layout", choices=["flat", "nested"], default="flat")

    # --source-type csv
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--csv-image-col", default="image_path")
    parser.add_argument("--csv-label-col", default="label")
    parser.add_argument("--csv-source-col", default=None)

    args = parser.parse_args()

    if args.source_type == "hf":
        if not args.hf_dataset:
            parser.error("--hf-dataset is required for --source-type hf")
        label_map = parse_label_map(args.hf_label_map) if args.hf_label_map else None
        source_iter = iter_hf_dataset(
            dataset_name=args.hf_dataset,
            config_name=args.hf_config,
            split=args.hf_split,
            image_field=args.hf_image_field,
            label_field=args.hf_label_field,
            label_map=label_map,
            streaming=not args.hf_no_streaming,
            shuffle_seed=args.seed,
        )
    elif args.source_type == "folder":
        if not args.folder_root:
            parser.error("--folder-root is required for --source-type folder")
        source_iter = iter_local_folder(str(args.folder_root), layout=args.folder_layout)
    else:  # csv
        if not args.csv_path:
            parser.error("--csv-path is required for --source-type csv")
        source_iter = iter_csv_manifest(
            str(args.csv_path), image_col=args.csv_image_col,
            label_col=args.csv_label_col, source_col=args.csv_source_col,
        )

    print(f"Collecting up to {args.per_class} real + {args.per_class} fake images "
          f"from source-type='{args.source_type}'...")

    collected = {0: [], 1: []}
    tag_counts: dict[str, int] = {}
    for image, label, tag in source_iter:
        if label not in (0, 1):
            continue
        if tag:
            if tag_counts.get(tag, 0) >= args.per_class:
                continue
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        elif len(collected[label]) >= args.per_class:
            continue
        collected[label].append((image, tag))

        if len(collected[0]) >= args.per_class and len(collected[1]) >= args.per_class:
            break

    print(f"Collected: real={len(collected[0])}, fake={len(collected[1])}")
    if collected[1]:
        by_tag: dict[str, int] = {}
        for _, tag in collected[1]:
            by_tag[tag or "(untagged)"] = by_tag.get(tag or "(untagged)", 0) + 1
        print(f"Fake breakdown by source tag: {by_tag}")

    if not collected[0] or not collected[1]:
        print("WARNING: one class is empty — check your --hf-label-map / folder layout / "
              "CSV label column before training on this.")

    splits = [("train", 1 - args.val_frac - args.test_frac),
              ("val", args.val_frac), ("test", args.test_frac)]

    manifest_rows: list = []
    split_and_save(collected[0], splits, args.out, 0, manifest_rows, "real", args.seed)
    split_and_save(collected[1], splits, args.out, 1, manifest_rows, "fake", args.seed)

    manifest_path = args.out / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "generator_family"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote manifest with {len(manifest_rows)} rows to {manifest_path}")
    for split_name, _ in splits:
        n_real = sum(1 for r in manifest_rows if r["label"] == 0 and f"/{split_name}/real/" in r["image_path"])
        n_fake = sum(1 for r in manifest_rows if r["label"] == 1 and f"/{split_name}/fake/" in r["image_path"])
        print(f"  {split_name}: real={n_real} fake={n_fake}")


if __name__ == "__main__":
    main()
