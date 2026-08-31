"""
Generates DDA-aligned synthetic ("fake") counterparts for a folder of real
images — see src/dda.py for the method (VAE reconstruction + frequency
alignment + pixel mixup, from Chen et al. NeurIPS 2025).

Two ways to use this:

  1. STANDALONE: point it at any folder of real images (e.g. MSCOCO) and it
     produces a real/fake pair dataset from scratch — matching the paper's
     setup, which trains exclusively on MSCOCO + its DDA-aligned counterparts
     and needs no other fake-image dataset at all.

  2. AUGMENT existing data: point it at data/train/real (already built by
     prepare_dataset.py from SID_Set or elsewhere) to add DDA-aligned fakes
     into data/train/fake_dda/, then merge into the existing manifest. This
     mixes DDA's shortcut-resistant fakes with your existing (SID_Set, etc.)
     fakes for extra diversity.

Usage:
    # standalone: build a fresh train/val/test split from a real-image folder
    python scripts/generate_dda_dataset.py \\
        --real-images path/to/mscoco_or_any_real_images \\
        --out data_dda/ --max-images 3000

    # augment an existing prepared dataset (adds to data/train/fake_dda/,
    # merges rows into data/manifest.csv with generator_family="dda")
    python scripts/generate_dda_dataset.py \\
        --real-images data/train/real \\
        --out data/ --split train --augment-existing --max-images 3000
"""
from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from tqdm import tqdm

from src.dda import generate_dda_pair, load_vae

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-images", required=True, type=Path,
                         help="Folder of real images to generate DDA counterparts for")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-images", type=int, default=3000)
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse",
                         help="HF repo for the VAE. Default is a publicly-accessible standalone "
                              "kl-f8 VAE (no gating). The paper's own best-performing choice, "
                              "stabilityai/stable-diffusion-2-1 (with --vae-subfolder vae), has "
                              "become gated/inaccessible on the Hub as of this writing — pass it "
                              "explicitly if you have access.")
    parser.add_argument("--vae-subfolder", default=None,
                         help="Set to 'vae' if --vae-model is a full SD pipeline repo "
                              "(e.g. stabilityai/stable-diffusion-2-1) rather than a standalone "
                              "VAE repo (e.g. stabilityai/sd-vae-ft-mse, the default).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--p-freq", type=float, default=0.5,
                         help="Probability of applying JPEG frequency alignment (paper default: 0.5)")
    parser.add_argument("--p-pixel", type=float, default=0.5,
                         help="Probability of applying pixel mixup (paper's Ppixel)")
    parser.add_argument("--r-pixel-max", type=float, default=0.5,
                         help="Upper bound for the pixel mixup ratio (paper's Rpixel)")
    parser.add_argument("--max-dim", type=int, default=512,
                         help="Downsize images so the longer side is at most this before VAE "
                              "encode/decode — bounds GPU memory use regardless of source image "
                              "size. Lower further (e.g. 384) if you still hit CUDA OOM.")
    parser.add_argument("--seed", type=int, default=42)

    # standalone mode: also split the real images into train/val/test
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)

    # augment-existing mode: real images are already inside a prepared split,
    # only write fakes + append manifest rows, don't re-split the reals
    parser.add_argument("--augment-existing", action="store_true",
                         help="--real-images is already a prepared split folder "
                              "(e.g. data/train/real) — only generate the fake "
                              "counterparts and merge into --out/manifest.csv")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                         help="Which split these DDA fakes belong to (augment-existing mode)")

    args = parser.parse_args()
    random.seed(args.seed)

    image_paths = sorted(p for p in args.real_images.iterdir() if p.suffix.lower() in VALID_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.real_images}")
    if len(image_paths) > args.max_images:
        image_paths = random.sample(image_paths, args.max_images)

    print(f"Loading VAE ({args.vae_model}) on {args.device}...")
    vae = load_vae(args.vae_model, subfolder=args.vae_subfolder, device=args.device)

    print(f"Generating DDA-aligned counterparts for {len(image_paths)} images...")

    if args.augment_existing:
        fake_dir = args.out / args.split / "fake_dda"
        fake_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []

        for i, path in enumerate(tqdm(image_paths)):
            real_img = Image.open(path)
            dda_img = generate_dda_pair(
                real_img, vae, device=args.device,
                p_freq=args.p_freq, p_pixel=args.p_pixel, r_pixel_max=args.r_pixel_max,
                seed=args.seed + i, max_dim=args.max_dim,
            )
            dest = fake_dir / f"dda_{i:06d}.jpg"
            dda_img.save(dest, format="JPEG", quality=95)
            manifest_rows.append({"image_path": str(dest), "label": 1, "generator_family": "dda"})

            if (i + 1) % 25 == 0:
                # Belt-and-suspenders against fragmentation across many iterations —
                # vae_reconstruct already frees its own tensors per-call, but a
                # periodic full GC + cache clear catches anything that slips through.
                gc.collect()
                if args.device == "cuda":
                    import torch
                    torch.cuda.empty_cache()

        manifest_path = args.out / "manifest.csv"
        file_exists = manifest_path.exists()
        with open(manifest_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label", "generator_family"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(manifest_rows)

        print(f"\nWrote {len(manifest_rows)} DDA-aligned fakes to {fake_dir}")
        print(f"Appended manifest rows to {manifest_path}")
        print(f"\nNOTE: these are in a separate 'fake_dda/' subfolder, not merged into "
              f"'fake/'. train.py's dataset scanner only reads real/ and fake/ — either "
              f"move these files into {args.out / args.split / 'fake'} (recommended, "
              f"simplest) or extend src/dataset.py's _scan_split_dir to also include "
              f"fake_dda/.")

    else:
        splits = [("train", 1 - args.val_frac - args.test_frac),
                  ("val", args.val_frac), ("test", args.test_frac)]
        random.shuffle(image_paths)

        manifest_rows = []
        idx = 0
        for i, (split_name, frac) in enumerate(splits):
            count = len(image_paths) - idx if i == len(splits) - 1 else int(len(image_paths) * frac)
            chunk = image_paths[idx: idx + count]

            real_dir = args.out / split_name / "real"
            fake_dir = args.out / split_name / "fake"
            real_dir.mkdir(parents=True, exist_ok=True)
            fake_dir.mkdir(parents=True, exist_ok=True)

            for j, path in enumerate(tqdm(chunk, desc=split_name)):
                real_img = Image.open(path).convert("RGB")
                dda_img = generate_dda_pair(
                    real_img, vae, device=args.device,
                    p_freq=args.p_freq, p_pixel=args.p_pixel, r_pixel_max=args.r_pixel_max,
                    seed=args.seed + idx + j, max_dim=args.max_dim,
                )
                real_dest = real_dir / f"real_{idx + j:06d}.jpg"
                fake_dest = fake_dir / f"dda_{idx + j:06d}.jpg"
                real_img.save(real_dest, format="JPEG", quality=95)
                dda_img.save(fake_dest, format="JPEG", quality=95)

                manifest_rows.append({"image_path": str(real_dest), "label": 0, "generator_family": ""})
                manifest_rows.append({"image_path": str(fake_dest), "label": 1, "generator_family": "dda"})

                if (j + 1) % 25 == 0:
                    gc.collect()
                    if args.device == "cuda":
                        import torch
                        torch.cuda.empty_cache()

            idx += count

        manifest_path = args.out / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "label", "generator_family"])
            writer.writeheader()
            writer.writerows(manifest_rows)

        print(f"\nWrote {len(manifest_rows)} rows to {manifest_path}")
        print(f"Standalone real/DDA-fake dataset ready at {args.out}")


if __name__ == "__main__":
    main()
