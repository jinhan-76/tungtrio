"""
Dual Data Alignment (DDA) — Chen et al., NeurIPS 2025
(https://arxiv.org/abs/2505.14359)

Generates a "hard", shortcut-resistant synthetic counterpart for a real
image via three steps:

  1. VAE reconstruction (pixel alignment):
         x_hat = Decoder(Encoder(x))
     using the Stable Diffusion 2.1 VAE specifically — the paper's ablation
     found SD2.1's VAE outperforms SD1.5/SDXL for this purpose.

  2. Frequency alignment: JPEG-recompress x_hat at the SAME quality factor
     as the original real image (estimated from its quantization tables),
     applied with probability p_freq during training. Without this step,
     the reconstruction retains more high-frequency detail than the real
     image (because the real image lost that detail to compression, but
     the VAE reconstruction didn't) — and a detector trained on unaligned
     pairs learns "high-frequency = fake" as a shortcut, not a real
     forgery signal.

  3. Pixel mixup (further pixel alignment):
         x_mix = r * x_real + (1 - r) * x_freq_aligned,   r ~ U(0, r_pixel_max)
     blends the aligned fake back toward the real image in pixel space.

The result (x_mix) is used as a "fake" training example paired with the
original real image — no labeled multi-generator fake dataset is required;
DDA only needs real images and the SD2.1 VAE.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

# Standard JPEG luminance quantization table at quality=50 (JPEG Annex K).
# Used to estimate a real image's original JPEG quality factor from its
# quantization tables, so the reconstructed counterpart can be recompressed
# at the SAME quality (the frequency-alignment step).
_STD_LUMA_QTABLE = np.array([
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Step 1: VAE reconstruction
# ---------------------------------------------------------------------------
_vae_cache = {}


def load_vae(model_name: str = "stabilityai/sd-vae-ft-mse", subfolder: str | None = None,
             device: str = "cuda"):
    """Loads (and caches) just the VAE — we never need the U-Net/text encoder
    for DDA, only encode/decode.

    Default is `stabilityai/sd-vae-ft-mse` — a standalone, publicly-accessible
    kl-f8 VAE repo (no login/gating required). The paper's own ablation found
    SD2.1's bundled VAE (`stabilityai/stable-diffusion-2-1`, subfolder="vae")
    performs best; if that repo is accessible for you, pass
    model_name="stabilityai/stable-diffusion-2-1", subfolder="vae" instead.
    As of this writing, `stabilityai/stable-diffusion-2-1` has become
    gated/inaccessible on the Hub — check
    https://huggingface.co/stabilityai/stable-diffusion-2-1 directly if you
    want to try requesting access; sd-vae-ft-mse is the fallback that's
    known to work without any extra steps.
    """
    import torch
    from diffusers import AutoencoderKL

    key = (model_name, subfolder, device)
    if key not in _vae_cache:
        kwargs = {"subfolder": subfolder} if subfolder else {}
        vae = AutoencoderKL.from_pretrained(model_name, **kwargs)
        vae = vae.to(device).eval()
        _vae_cache[key] = vae
    return _vae_cache[key]


def _crop_to_multiple_of_8(img: Image.Image, max_dim: int = 512) -> Image.Image:
    """The paper center-crops to the largest size that's a multiple of 8
    before VAE reconstruction, matching the VAE's 8x downsampling factor,
    so the reconstructed image comes back at exactly the same resolution.

    Also downsizes first if the image's longer side exceeds `max_dim` —
    the VAE encode/decode is O(H*W) in GPU memory, and since the model only
    trains at 224x224 anyway (see dataset.py's final resize), reconstructing
    at a source photo's native resolution (e.g. 3000x2000) wastes memory for
    no downstream benefit. 768 keeps memory bounded on a 24GB GPU while
    still being well above the training resolution.
    """
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        w, h = img.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h))


def vae_reconstruct(img: Image.Image, vae, device: str = "cuda", max_dim: int = 512) -> Image.Image:
    """x_hat = Decoder(Encoder(x)) — no diffusion, just the autoencoder pass."""
    import torch

    img = _crop_to_multiple_of_8(img.convert("RGB"), max_dim=max_dim)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0  # [-1, 1], VAE's expected range
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
        recon = vae.decode(latent / vae.config.scaling_factor).sample

    recon = (recon.clamp(-1, 1) + 1) / 2 * 255.0
    recon = recon.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    del tensor, latent  # free GPU memory before returning, rather than waiting for GC
    if device == "cuda":
        torch.cuda.empty_cache()

    return Image.fromarray(recon)


# ---------------------------------------------------------------------------
# Step 2: Frequency alignment (JPEG quality matching)
# ---------------------------------------------------------------------------
def estimate_jpeg_quality(img: Image.Image, default: int = 75) -> int:
    """Estimates a JPEG's original quality factor (1-100) from its
    quantization table, by inverting libjpeg's standard quality->table
    scaling formula. Falls back to `default` if the image has no
    quantization table (e.g. it wasn't actually a JPEG, or PIL couldn't
    read one) — this happens for PNG-sourced real images, in which case
    frequency alignment isn't meaningful and this image should probably
    be skipped for that step (see apply_frequency_alignment).
    """
    qtables = getattr(img, "quantization", None)
    if not qtables or 0 not in qtables:
        return default

    actual = np.array(qtables[0], dtype=np.float64)
    if actual.shape[0] != 64:
        return default

    # Ignore entries clipped at the boundaries (1 or 255) — those don't
    # scale linearly with quality and would bias the estimate.
    mask = (actual > 1) & (actual < 255) & (_STD_LUMA_QTABLE > 0)
    if mask.sum() < 8:
        return default

    ratio = np.mean(actual[mask] / _STD_LUMA_QTABLE[mask])
    scale_factor = ratio * 100

    if scale_factor >= 100:
        quality = 5000 / scale_factor
    else:
        quality = (200 - scale_factor) / 2

    return int(np.clip(round(quality), 1, 100))


def apply_frequency_alignment(recon_img: Image.Image, real_img: Image.Image,
                               p_freq: float = 0.5, rng: np.random.Generator | None = None,
                               default_quality: int = 75) -> Image.Image:
    """Re-JPEG-compresses the VAE reconstruction at the real image's own
    estimated JPEG quality, applied with probability p_freq (matching the
    paper's "50% probability during training" so the model also sees
    unaligned/PNG-format synthetic images and doesn't overfit to the
    presence of JPEG artifacts specifically).
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p_freq:
        return recon_img  # left as PNG-equivalent / uncompressed

    quality = estimate_jpeg_quality(real_img, default=default_quality)
    buf = io.BytesIO()
    recon_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Step 3: Pixel-level mixup
# ---------------------------------------------------------------------------
def pixel_mixup(real_img: Image.Image, freq_aligned_img: Image.Image,
                 p_pixel: float = 0.5, r_pixel_max: float = 0.5,
                 rng: np.random.Generator | None = None) -> Image.Image:
    """x_mix = r * x_real + (1 - r) * x_freq_aligned,  r ~ U(0, r_pixel_max)

    Applied with probability p_pixel (paper's Ppixel). When skipped, the
    frequency-aligned image is returned unchanged (r effectively 0).
    Both images must be the same size — freq_aligned_img comes from a
    VAE-reconstructed (and center-cropped) version of real_img, so resize
    real_img to match before calling this if they've diverged.
    """
    rng = rng or np.random.default_rng()
    if rng.random() > p_pixel:
        return freq_aligned_img

    if real_img.size != freq_aligned_img.size:
        real_img = real_img.resize(freq_aligned_img.size)

    r = rng.uniform(0, r_pixel_max)
    real_arr = np.array(real_img.convert("RGB"), dtype=np.float32)
    syn_arr = np.array(freq_aligned_img.convert("RGB"), dtype=np.float32)
    mixed = r * real_arr + (1 - r) * syn_arr
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def generate_dda_pair(
    real_img: Image.Image,
    vae,
    device: str = "cuda",
    p_freq: float = 0.5,
    p_pixel: float = 0.5,
    r_pixel_max: float = 0.5,
    seed: int | None = None,
    max_dim: int = 512,
) -> Image.Image:
    """Runs all three DDA steps on one real image, returning its DDA-aligned
    synthetic counterpart. Defaults (p_freq=0.5, p_pixel=0.5, r_pixel_max=0.5)
    match the paper's reported sweet spot (Ppixel/Rpixel between 0.2-0.8).
    `max_dim` bounds VAE memory use — see vae_reconstruct's docstring.
    """
    rng = np.random.default_rng(seed)
    recon = vae_reconstruct(real_img, vae, device=device, max_dim=max_dim)
    freq_aligned = apply_frequency_alignment(recon, real_img, p_freq=p_freq, rng=rng)
    mixed = pixel_mixup(real_img, freq_aligned, p_pixel=p_pixel, r_pixel_max=r_pixel_max, rng=rng)
    return mixed
