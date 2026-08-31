"""
Augmentations matching the hackathon organizer's required spec (six transforms
simulating the real-world image redistribution pipeline: JPEG re-encode,
blur, resize, noise, color jitter, center crop).

HOW they're applied is driven by findings from Li et al., "Is Artificial
Intelligence Generated Image Detection a Solved Problem?" (AIGIBench,
NeurIPS 2025 D&B track):

  1. R.Acc vs F.Acc must be tracked SEPARATELY, not just overall accuracy.
     AIGIBench's central finding: under JPEG/noise, detectors' R.Acc. stays
     near 100% while F.Acc. collapses toward 0% — the detector becomes
     biased toward always predicting "real". Overall accuracy hides this
     completely. (See src/utils.py's compute_metrics, which now reports
     r_acc/f_acc alongside accuracy.)

  2. Training-time augmentation should apply AT MOST ONE transform per
     image, not several stacked together. AIGIBench found that combining
     multiple augmentations "offers no clear advantage and can impair
     performance consistency", particularly for frequency-sensitive
     detectors (relevant here — this project's SRM branch is exactly such
     a component). RequiredTrainingAugmentation below defaults to
     max_ops=1 for this reason — a deliberate, evidence-based choice, not
     a simplification.

  3. Evaluation-time, each transform (and each parameter value within it)
     must be tested in ISOLATION against a clean baseline, not combined —
     otherwise a robustness drop can't be attributed to a specific
     transform. RequiredEvalTransforms.all_conditions() enumerates every
     (transform, parameter) condition from the organizer's table separately,
     mirroring AIGIBench Table 4's per-condition breakdown.

  4. JPEG compression and noise are the most damaging conditions (paper:
     "JPEG Compression is the Most Challenging Perturbation" for F.Acc) —
     these get evaluated across the full organizer-specified quality/sigma
     grid, not just one setting, so the eval isn't hiding a collapse at a
     harsher quality than was tested.

  5. Crop mainly boosts R.Acc, not F.Acc (paper: crop's "improvements are
     primarily driven by gains in R.Acc., while F.Acc. often remains
     unaffected or even degrades"). It's included per the organizer's spec,
     but don't expect it to be a robustness fix on its own — check its
     R.Acc/F.Acc breakdown specifically before crediting it.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# ---------------------------------------------------------------------------
# Exact parameter grids from the organizer's spec
# ---------------------------------------------------------------------------
JPEG_QUALITIES = [90, 70, 50, 30]          # social-media re-encode, messaging
BLUR_SIGMAS = [0.5, 1.0, 2.0]              # out-of-focus, screenshot smoothing
RESIZE_SCALES = [0.5, 0.25]                # thumbnail generation, CDN resize
NOISE_SIGMAS = [0.02, 0.05, 0.10]          # low-light sensor noise (on [0,1]-scaled pixels)
JITTER_STRENGTH = 0.20                     # filter apps, auto-enhance (±20%)
CENTER_CROP_FRAC = 0.80                    # profile-picture cropping, framing


# ---------------------------------------------------------------------------
# Individual transforms — each takes a PIL.Image, returns a PIL.Image.
# `param=None` picks a random value from that transform's grid (training
# use); passing an explicit param is how eval sweeps a specific condition.
# ---------------------------------------------------------------------------
def jpeg_compress(img: Image.Image, quality: int | None = None) -> Image.Image:
    quality = quality if quality is not None else random.choice(JPEG_QUALITIES)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float | None = None) -> Image.Image:
    sigma = sigma if sigma is not None else random.choice(BLUR_SIGMAS)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def resize_down_up(img: Image.Image, scale: float | None = None) -> Image.Image:
    """Downsamples to `scale` of the original size, then upsamples back —
    simulates a thumbnail/CDN resize round-trip, which is lossy in a
    different way than blur (loses high-frequency detail via resampling,
    not smoothing)."""
    scale = scale if scale is not None else random.choice(RESIZE_SCALES)
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def gaussian_noise(img: Image.Image, sigma: float | None = None) -> Image.Image:
    """sigma is on a [0,1]-normalized pixel scale (0.02-0.10), matching the
    organizer's spec — NOT the same scale as 0-255 pixel values."""
    sigma = sigma if sigma is not None else random.choice(NOISE_SIGMAS)
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 1) * 255.0
    return Image.fromarray(arr.astype(np.uint8))


def color_jitter(img: Image.Image, strength: float = JITTER_STRENGTH) -> Image.Image:
    """±strength on brightness, contrast, and saturation (PIL's 'Color' enhancer)."""
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-strength, strength)
        img = enhancer_cls(img).enhance(factor)
    return img


def center_crop(img: Image.Image, frac: float = CENTER_CROP_FRAC) -> Image.Image:
    """Crops to `frac` of width/height, centered. Output is smaller than the
    input — fine, since dataset.py resizes to the model's input size after
    any augmentation runs."""
    w, h = img.size
    new_w, new_h = int(w * frac), int(h * frac)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


TRANSFORM_REGISTRY = {
    "jpeg": jpeg_compress,
    "blur": gaussian_blur,
    "resize": resize_down_up,
    "noise": gaussian_noise,
    "jitter": color_jitter,
    "crop": center_crop,
}


# ---------------------------------------------------------------------------
# Training-time: applied RANDOMLY, at most one transform per image
# ---------------------------------------------------------------------------
class RequiredTrainingAugmentation:
    """Applied randomly during training, per the organizer's instruction.

    `max_ops=1` (default) is a deliberate choice, not a simplification —
    see the module docstring, point 2. AIGIBench found stacking multiple
    augmentations doesn't help and can hurt F.Acc specifically. If you want
    to experiment with stacking anyway, raise max_ops, but check the
    resulting R.Acc/F.Acc breakdown (not just overall accuracy) before
    concluding it helped.
    """

    def __init__(self, transform_names: list[str] | None = None,
                 apply_prob: float = 0.7, max_ops: int = 1, seed: int | None = None):
        self.names = transform_names or list(TRANSFORM_REGISTRY.keys())
        self.apply_prob = apply_prob
        self.max_ops = max_ops
        self._rng = random.Random(seed)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self._rng.random() > self.apply_prob:
            return img  # left clean — keeps some undistorted signal in training
        n = self._rng.randint(1, self.max_ops)
        chosen = self._rng.sample(self.names, k=min(n, len(self.names)))
        for name in chosen:
            img = TRANSFORM_REGISTRY[name](img)
        return img


# ---------------------------------------------------------------------------
# Evaluation-time: each condition tested in ISOLATION against clean baseline
# ---------------------------------------------------------------------------
class RequiredEvalTransforms:
    """Not randomized, not combined — every (transform, parameter) pair from
    the organizer's grid is its own condition, evaluated separately against
    the clean baseline. This is what lets eval_robustness.py report, e.g.,
    "F.Acc collapses specifically at JPEG q=30" rather than a single
    averaged-away robustness number.
    """

    @staticmethod
    def all_conditions() -> list[tuple[str, callable]]:
        conditions: list[tuple[str, callable]] = [("Clean", lambda img: img)]
        for q in JPEG_QUALITIES:
            conditions.append((f"JPEG_q{q}", lambda img, q=q: jpeg_compress(img, q)))
        for s in BLUR_SIGMAS:
            conditions.append((f"Blur_sigma{s}", lambda img, s=s: gaussian_blur(img, s)))
        for sc in RESIZE_SCALES:
            conditions.append((f"Resize_{sc}x", lambda img, sc=sc: resize_down_up(img, sc)))
        for sg in NOISE_SIGMAS:
            conditions.append((f"Noise_sigma{sg}", lambda img, sg=sg: gaussian_noise(img, sg)))
        conditions.append(("ColorJitter_20pct", lambda img: color_jitter(img)))
        conditions.append(("CenterCrop_80pct", lambda img: center_crop(img)))
        return conditions
