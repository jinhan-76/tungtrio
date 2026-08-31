"""
Train the AI-image detector.

Usage:
    python train.py --config configs/colab_t4.yaml
    python train.py --config configs/workstation.yaml

Resumable by default: if train.<checkpoint_dir>/last.pt exists, training
picks up from the saved epoch/optimizer state automatically. This matters on
Colab, where a session can disconnect mid-run — just re-run the same command.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.augmentations import RequiredTrainingAugmentation
from src.dataset import AIImageDataset
from src.model import AIImageDetector
from src.utils import compute_metrics, load_checkpoint, save_checkpoint, set_seed


def build_dataloaders(cfg):
    aug = RequiredTrainingAugmentation(
        apply_prob=cfg["augmentation"]["apply_prob"],
        max_ops=cfg["augmentation"]["max_ops"],
    )

    train_ds = AIImageDataset(
        cfg["data"]["root"], "train",
        image_size=cfg["data"]["image_size"],
        manifest_path=cfg["data"].get("manifest"),
        online_augment=aug,
    )
    val_ds = AIImageDataset(
        cfg["data"]["root"], "val",
        image_size=cfg["data"]["image_size"],
        manifest_path=cfg["data"].get("manifest"),
        online_augment=None,  # clean eval during training
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["train"]["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


def run_epoch(model, loader, optimizer, scaler, device, cfg, train: bool, epoch: int):
    model.train(mode=train)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels, all_probs = [], []
    grad_accum = cfg["train"]["grad_accum_steps"] if train else 1
    amp_dtype = torch.bfloat16 if cfg["train"]["mixed_precision"] == "bf16" else torch.float16

    if train:
        optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"{'train' if train else 'val'} epoch {epoch}")
    for step, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        raw_pixels = batch["raw_pixels"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                             dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = model(pixel_values, raw_pixels)
            loss = criterion(logits, labels) / grad_accum

        if train:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        all_labels.extend(labels.detach().cpu().tolist())
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())

        if train and step % cfg["train"]["log_every"] == 0:
            pbar.set_postfix(loss=total_loss / (step + 1))

    metrics = compute_metrics(all_labels, all_probs)
    metrics["loss"] = total_loss / len(loader)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_dataloaders(cfg)

    model = AIImageDetector(
        dinov2_model_name=cfg["model"]["dinov2_model_name"],
        lora_r=cfg["model"]["lora_r"],
        lora_alpha=cfg["model"]["lora_alpha"],
        srm_feat_dim=cfg["model"]["srm_feat_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"
                                                 and cfg["train"]["mixed_precision"] == "fp16"))

    last_ckpt = os.path.join(cfg["train"]["checkpoint_dir"], "last.pt")
    best_ckpt = os.path.join(cfg["train"]["checkpoint_dir"], "best.pt")
    start_epoch, best_metric = (0, float("-inf"))
    if cfg["train"].get("resume", True):
        start_epoch, best_metric = load_checkpoint(last_ckpt, model, optimizer,
                                                     map_location=device)
        if start_epoch > 0:
            print(f"Resumed from epoch {start_epoch} (best val F1 so far: {best_metric:.4f})")

    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, cfg,
                                   train=True, epoch=epoch)
        val_metrics = run_epoch(model, val_loader, optimizer, scaler, device, cfg,
                                 train=False, epoch=epoch)

        print(f"[epoch {epoch}] train loss={train_metrics['loss']:.4f} "
              f"acc={train_metrics['accuracy']:.4f} | "
              f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} "
              f"R.Acc={val_metrics['r_acc']:.4f} F.Acc={val_metrics['f_acc']:.4f} "
              f"f1={val_metrics['f1']:.4f} auc={val_metrics['auc']:.4f}")

        save_checkpoint(last_ckpt, model, optimizer, epoch + 1, best_metric)
        if val_metrics["f1"] > best_metric:
            best_metric = val_metrics["f1"]
            save_checkpoint(best_ckpt, model, optimizer, epoch + 1, best_metric)
            print(f"  -> new best (val F1={best_metric:.4f}), saved to {best_ckpt}")


if __name__ == "__main__":
    main()
