"""
Model architecture: DINOv2 (LoRA-adapted) backbone + fixed SRM high-pass
residual stream, fused into a 2-layer MLP classification head.

The SRM branch is deliberately small and fixed (non-trainable filters) so the
model can't just relearn a low-pass filter and throw away the high-frequency
signal; only the small CNN encoder *after* the fixed filters is trainable.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


# ---------------------------------------------------------------------------
# SRM high-pass filter bank
# ---------------------------------------------------------------------------
# A compact set of classic steganalysis-style high-pass kernels (subset of the
# Fridrich SRM bank). Fixed / non-trainable: these are hand-designed residual
# filters, not learned, so the aux branch is forced to operate on noise
# residuals rather than raw pixel content.
_SRM_KERNELS = [
    # 1st order edge kernels
    [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
    [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
    [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
    # 2nd order
    [[0, 0, 0], [1, -2, 1], [0, 0, 0]],
    [[0, 1, 0], [0, -2, 0], [0, 1, 0]],
    # 3x3 laplacian-like
    [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]],
]


class SRMFilter(nn.Module):
    """Fixed high-pass conv layer applied per RGB channel (depthwise)."""

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(_SRM_KERNELS, dtype=torch.float32)  # (K, 3, 3)
        k = kernels.shape[0]
        # depthwise conv applied independently to each of the 3 RGB channels
        weight = kernels.unsqueeze(1).repeat(3, 1, 1, 1)  # (3*K, 1, 3, 3)
        self.register_buffer("weight", weight)
        self.groups = 3
        self.out_channels = 3 * k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1]
        return F.conv2d(x, self.weight, padding=1, groups=3)


class SRMBranch(nn.Module):
    """SRM filter bank -> small trainable CNN -> pooled feature vector."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.srm = SRMFilter()
        in_ch = self.srm.out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            residual = self.srm(x)
        feat = self.encoder(residual)
        feat = feat.flatten(1)
        return self.proj(feat)


# ---------------------------------------------------------------------------
# DINOv2 backbone with LoRA
# ---------------------------------------------------------------------------
def build_dinov2_lora(model_name: str, lora_r: int = 8, lora_alpha: int = 16,
                       lora_dropout: float = 0.05) -> nn.Module:
    """Loads a DINOv2 backbone and wraps its attention projections with LoRA.

    Base weights are frozen; only LoRA adapters (+ later, our head) train.
    """
    backbone = AutoModel.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "value"],  # attention proj layers
        bias="none",
    )
    backbone = get_peft_model(backbone, lora_config)
    return backbone


class AIImageDetector(nn.Module):
    def __init__(
        self,
        dinov2_model_name: str = "facebook/dinov2-base",
        lora_r: int = 8,
        lora_alpha: int = 16,
        srm_feat_dim: int = 256,
        hidden_dim: int = 256,
        freeze_backbone_lora: bool = False,
    ):
        super().__init__()
        self.backbone = build_dinov2_lora(dinov2_model_name, lora_r, lora_alpha)
        backbone_dim = self.backbone.config.hidden_size

        self.srm_branch = SRMBranch(out_dim=srm_feat_dim)

        fused_dim = backbone_dim + srm_feat_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

        if freeze_backbone_lora:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, pixel_values: torch.Tensor, raw_pixels: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, H, W), ImageNet-normalized (for DINOv2).
        raw_pixels:   (B, 3, H, W), unnormalized [0, 1] range (for the SRM branch,
                      which needs true pixel statistics, not normalized ones).
        Returns raw logits of shape (B,) — apply sigmoid for a probability.
        """
        backbone_out = self.backbone(pixel_values=pixel_values)
        cls_feat = backbone_out.last_hidden_state[:, 0]
        srm_feat = self.srm_branch(raw_pixels)
        fused = torch.cat([cls_feat, srm_feat], dim=1)
        logit = self.head(fused).squeeze(-1)
        return logit

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


if __name__ == "__main__":
    # quick shape sanity check (won't download weights unless run for real)
    model = AIImageDetector()
    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,}")
