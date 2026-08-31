from __future__ import annotations

import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str, model, optimizer, epoch: int, best_metric: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
        },
        path,
    )


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu"):
    """Returns (start_epoch, best_metric). If the file doesn't exist, returns (0, -inf)
    so training can start fresh — this is what makes train.py resumable-by-default:
    just re-run the same command and it'll pick up from checkpoints/last.pt if present.
    """
    if not os.path.exists(path):
        return 0, float("-inf")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("best_metric", float("-inf"))


def compute_metrics(y_true, y_pred_prob, threshold: float = 0.5) -> dict:
    """Includes r_acc/f_acc (accuracy on real images only / fake images only)
    alongside overall accuracy — this decomposition matters specifically
    because overall accuracy can look fine while the detector has actually
    collapsed to "always predict real" under perturbation (R.Acc ~100%,
    F.Acc ~0%). See AIGIBench (Li et al., NeurIPS 2025 D&B) Table 4 for the
    exact failure mode this is designed to catch, and
    src/augmentations.py's module docstring for how this connects to the
    training/eval augmentation design here.
    """
    y_pred = [1 if p >= threshold else 0 for p in y_pred_prob]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = float("nan")  # only one class present in this batch/split

    real_idx = [i for i, l in enumerate(y_true) if l == 0]
    fake_idx = [i for i, l in enumerate(y_true) if l == 1]
    r_acc = (accuracy_score([y_true[i] for i in real_idx], [y_pred[i] for i in real_idx])
             if real_idx else float("nan"))
    f_acc = (accuracy_score([y_true[i] for i in fake_idx], [y_pred[i] for i in fake_idx])
             if fake_idx else float("nan"))

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "r_acc": r_acc,
        "f_acc": f_acc,
    }
