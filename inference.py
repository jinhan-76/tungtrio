"""
Inference CLI — REQUIRED DELIVERABLE.

Takes a directory of images, outputs a JSON file with a confidence score
per image (probability the image is AI-generated).

Usage:
    python inference.py --input path/to/images --output predictions.json \
        --checkpoint checkpoints/best.pt [--tta] [--config configs/colab_t4.yaml]

Output format (predictions.json):
    [
      {"image_path": "path/to/images/0001.jpg", "pred": 0.93,
       "label": "AI-generated", "confidence_percent": 93.0},
      {"image_path": "path/to/images/0002.jpg", "pred": 0.04,
       "label": "Real", "confidence_percent": 96.0}
    ]

"pred" is the required field for grading (probability the image is
AI-generated, 0-1 — NOT a hard binary call). "label" and "confidence_percent"
are convenience fields derived from "pred" at a 0.5 threshold, for readability
in a demo; they don't add information beyond "pred" and can be ignored by any
downstream consumer that only expects image_path/pred.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import AIImageDetector
from src.utils import load_checkpoint

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def build_transforms(image_size: int):
    resize = transforms.Resize((image_size, image_size))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    return resize, to_tensor, normalize


@torch.no_grad()
def predict_image(model, img_path, resize, to_tensor, normalize, device, tta: bool) -> float:
    img = Image.open(img_path).convert("RGB")
    img = resize(img)
    raw = to_tensor(img).unsqueeze(0).to(device)          # (1,3,H,W) in [0,1]
    normed = normalize(raw.squeeze(0)).unsqueeze(0).to(device)

    logits = model(normed, raw)
    probs = [torch.sigmoid(logits).item()]

    if tta:
        raw_flip = torch.flip(raw, dims=[3])
        normed_flip = normalize(raw_flip.squeeze(0)).unsqueeze(0).to(device)
        logits_flip = model(normed_flip, raw_flip)
        probs.append(torch.sigmoid(logits_flip).item())

    return sum(probs) / len(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory of images to score")
    parser.add_argument("--output", required=True, help="Path to write predictions.json")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", default="configs/colab_t4.yaml",
                         help="Config used to build the model architecture")
    parser.add_argument("--tta", action="store_true",
                         help="Enable horizontal-flip test-time augmentation")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AIImageDetector(
        dinov2_model_name=cfg["model"]["dinov2_model_name"],
        lora_r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        srm_feat_dim=cfg["model"]["srm_feat_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)
    load_checkpoint(args.checkpoint, model, optimizer=None, map_location=device)
    model.eval()

    resize, to_tensor, normalize = build_transforms(cfg["data"]["image_size"])

    image_paths = sorted(
        os.path.join(args.input, f) for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input}")

    results = []
    for path in tqdm(image_paths, desc="scoring"):
        try:
            pred = predict_image(model, path, resize, to_tensor, normalize, device, args.tta)
        except Exception as e:
            print(f"WARNING: failed to score {path}: {e}")
            continue
        results.append({
            "image_path": path,
            "pred": round(pred, 4),
            "label": "AI-generated" if pred >= 0.5 else "Real",
            "confidence_percent": round((pred if pred >= 0.5 else 1 - pred) * 100, 1),
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} predictions to {args.output}")


if __name__ == "__main__":
    main()
