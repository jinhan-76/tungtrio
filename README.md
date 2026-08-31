# AI-Generated Image Detector

Detects whether an image is AI-generated (AIGC) or authentic, using a
DINOv2-ViT backbone (LoRA fine-tuned) fused with a fixed high-pass (SRM)
residual stream, trained with online JPEG/blur/noise augmentation for
robustness to post-processing.

## Overview

- **Backbone**: `facebook/dinov2-base` (or `dinov2-large`) with LoRA adapters
  (r=8) on the attention `query`/`value` projections. Base weights frozen.
- **Auxiliary stream**: fixed (non-trainable) SRM high-pass filter bank →
  small trainable CNN encoder. Pushes the model toward low-level noise-residual
  statistics (upsampling artifacts, generator fingerprints) instead of
  semantic content, which generalizes better across unseen generators.
- **Head**: concat(backbone CLS feature, SRM branch feature) → 2-layer MLP → logit.
- **Robustness training**: six required transforms (JPEG re-encode, Gaussian
  blur, resize round-trip, Gaussian noise, color jitter, center crop) applied
  randomly during training, matching the hackathon's required transform spec
  — see the "Required Transform Augmentation" section below for exact
  parameters and the reasoning behind how they're applied.

## Repo structure

```
ai-image-detector/
├── configs/
│   ├── colab_t4.yaml        # config for free-tier Colab (T4, 16GB)
│   └── workstation.yaml     # config for local/larger GPU
├── src/
│   ├── model.py              # SRM filters + DINOv2/LoRA + MLP head
│   ├── dataset.py            # dataset loading / caching
│   ├── augmentations.py      # required-transform spec: train (random, 1-op) / eval (isolated sweep)
│   └── utils.py               # checkpointing, seeding, metrics
├── scripts/
│   └── prepare_dataset.py     # subsamples GenImage into balanced splits
├── train.py                   # training loop, resumable from checkpoint
├── inference.py                # image dir -> predictions.json (+ TTA)
├── eval_robustness.py          # clean vs. transformed benchmark table
├── error_analysis.py           # top false positives / false negatives
└── requirements.txt
```

## Required Transform Augmentation

Implements the hackathon's required transform spec (`src/augmentations.py`),
applied per the findings in [Li et al., "Is Artificial Intelligence Generated
Image Detection a Solved Problem?" (AIGIBench, NeurIPS 2025 D&B track)](https://arxiv.org/abs/2505.12335).

| Transform | Parameters | Real-world analog |
|---|---|---|
| JPEG Compression | quality = 90, 70, 50, 30 | Social-media re-encode, messaging |
| Gaussian Blur | sigma = 0.5, 1.0, 2.0 | Out-of-focus, screenshot smoothing |
| Resize | scale 0.5x / 0.25x, then upscale back | Thumbnail generation, CDN resize |
| Gaussian Noise | sigma = 0.02, 0.05, 0.10 (on [0,1] pixels) | Low-light sensor noise |
| Color Jitter | brightness/contrast/saturation ±20% | Filter apps, auto-enhance |
| Center Crop | crop 80% | Profile-picture cropping, framing |

**Training** (`RequiredTrainingAugmentation`): applied randomly, **at most one
transform per image** (`max_ops=1` by default). This isn't a simplification —
AIGIBench found that combining multiple augmentations "offers no clear
advantage and can impair performance consistency," particularly for
frequency-sensitive detectors (relevant here, given this project's SRM
branch).

**Evaluation** (`RequiredEvalTransforms`, driven by `eval_robustness.py`):
every (transform, parameter) pair is evaluated **in isolation** against a
clean baseline — 15 conditions total — rather than combined, so a robustness
drop can be attributed to a specific condition. The resulting table reports
**R.Acc and F.Acc separately** (`src/utils.py`'s `compute_metrics`), not just
overall accuracy — because AIGIBench's central finding is that overall
accuracy can look fine while a detector has actually collapsed to "always
predict real" under perturbation (R.Acc near 100%, F.Acc collapsing toward
0%). Their results show JPEG compression is typically the single most
damaging condition for F.Acc, which is why the full JPEG quality grid
(90/70/50/30) is swept rather than just one setting.

One caveat for the write-up: AIGIBench also found Center Crop's
apparent benefit is "primarily driven by gains in R.Acc., while F.Acc. often
remains unaffected or even degrades" — don't over-credit it as a robustness
fix without checking its F.Acc specifically in your results table.

## Dual Data Alignment (DDA)

Implements [Chen et al., "Dual Data Alignment Makes AI-Generated Image
Detector Easier Generalizable" (NeurIPS 2025)](https://arxiv.org/abs/2505.14359)
(`src/dda.py`, `scripts/generate_dda_dataset.py`).

**Why**: naively reconstructing real images through a VAE to create pixel-aligned
fakes still leaves a shortcut — VAE reconstructions retain more high-frequency
detail than real (JPEG-compressed) images, so a detector trained on such pairs
learns "high-frequency = fake" instead of a real forgery signal. DDA fixes this
with three steps applied to a real image `x`:

1. **VAE reconstruction** (pixel alignment): `x̂ = Decoder(Encoder(x))`, using
   the Stable Diffusion 2.1 VAE specifically (the paper's ablation found SD2.1
   beats SD1.5/SDXL VAEs for this). **Note**: `stabilityai/stable-diffusion-2-1`
   has become gated/inaccessible on HuggingFace as of this writing, so
   `scripts/generate_dda_dataset.py` defaults to `stabilityai/sd-vae-ft-mse`
   instead — a publicly-accessible standalone kl-f8 VAE, no login required.
   If you have access to the original repo, pass
   `--vae-model stabilityai/stable-diffusion-2-1 --vae-subfolder vae` to match
   the paper exactly.
2. **Frequency alignment**: re-JPEG-compress `x̂` at the *same quality factor
   as the original real image* (estimated from its quantization table),
   applied with 50% probability — this is what removes the high-frequency
   shortcut.
3. **Pixel mixup**: `x_mix = r·x_real + (1-r)·x_freq_aligned`, `r ~ U(0, R)` —
   blends the result back toward the real image in pixel space.

Notably, **DDA needs no labeled fake-image dataset at all** — only real
images and the SD2.1 VAE. The paper's own detector is trained exclusively on
MSCOCO + its DDA-aligned counterparts and still beats specialized baselines
across 11 benchmarks, including generators never seen during training.

**Generate DDA fakes from real images:**
```bash
# standalone: build a fresh real/DDA-fake dataset from any folder of real images
python scripts/generate_dda_dataset.py \
  --real-images path/to/real_images --out data_dda/ --max-images 3000

# augment your existing prepared dataset (e.g. from SID_Set) with DDA fakes
python scripts/generate_dda_dataset.py \
  --real-images data/train/real --out data/ --split train \
  --augment-existing --max-images 3000
```
In augment mode, DDA fakes land in `data/train/fake_dda/` — move them into
`data/train/fake/` (the script tells you this) so `train.py`'s dataset
scanner picks them up alongside your existing fakes.

Defaults (`--p-freq 0.5 --p-pixel 0.5 --r-pixel-max 0.5`) match the paper's
reported sweet spot; their ablation shows stable performance for `Ppixel`/`Rpixel`
between 0.2–0.8, with degradation at the extremes (0.0 or 1.0).



```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires a HuggingFace account/token if `facebook/dinov2-*` weights are gated
in your environment (usually not required, they're public).

## Dataset

`scripts/prepare_dataset.py` works with **any** dataset source through a
common adapter interface (`src/data_adapters.py`), so you're not locked into
one dataset:

- `--source-type hf` — any HuggingFace Hub dataset (streams, doesn't download
  everything up front)
- `--source-type folder` — a local folder tree, either flat
  (`root/real/*.jpg`, `root/fake/*.jpg`) or nested by source
  (`root/<generator_name>/{real,fake}/*.jpg`)
- `--source-type csv` — a CSV manifest with `image_path,label` columns

All three write the same standard layout regardless of source, so
`train.py`/`inference.py`/`eval_robustness.py`/`error_analysis.py` never need
to change: `data/{train,val,test}/{real,fake}/*.jpg` + `data/manifest.csv`.

**Example — [SID_Set](https://huggingface.co/datasets/saberzl/SID_Set)**
(300K real/fully-synthetic/tampered images; gated, so
`huggingface-cli login` + accepting the dataset's terms is required first):

```bash
python scripts/prepare_dataset.py --source-type hf \
  --hf-dataset saberzl/SID_Set \
  --hf-image-field image --hf-label-field label \
  --hf-label-map "0:0:" "1:1:full_synthetic" "2:1:tampered" \
  --per-class 3000 --out data/ --val-frac 0.1 --test-frac 0.1
```

`--hf-label-map` entries are `raw_label:binary_label:source_tag` — this is
what makes the script dataset-agnostic: point it at any HF dataset and tell
it how that dataset's label values map to real(0)/fake(1), plus an optional
tag (e.g. generator name) recorded in `manifest.csv` for later per-source
error-analysis breakdown.

**Example — local folder of your own images:**
```bash
python scripts/prepare_dataset.py --source-type folder \
  --folder-root data_raw --folder-layout flat \
  --per-class 3000 --out data/
```

**Example — CSV manifest:**
```bash
python scripts/prepare_dataset.py --source-type csv \
  --csv-path my_manifest.csv --per-class 3000 --out data/
```

See the top of `scripts/prepare_dataset.py` for the full set of options
(each source type has its own flags — `--hf-*`, `--folder-*`, `--csv-*`).

## Training

```bash
python train.py --config configs/colab_t4.yaml       # free Colab T4
# or
python train.py --config configs/workstation.yaml    # local GPU
```

Training checkpoints (model + optimizer + epoch) are written every epoch to
`checkpoints/last.pt` and `checkpoints/best.pt`, and training resumes
automatically from `checkpoints/last.pt` if present — this matters on Colab,
where sessions can disconnect mid-run.

## Reproducing results

1. `python scripts/prepare_dataset.py --source-type hf --hf-dataset saberzl/SID_Set ...` (see Dataset section above for the full command and label-map)
2. `python train.py --config configs/<config>.yaml`
3. `python inference.py --input data/test --output predictions.json --checkpoint checkpoints/best.pt --tta`
4. `python eval_robustness.py --checkpoint checkpoints/best.pt --test-dir-root data`
5. `python error_analysis.py --predictions predictions.json --manifest data/manifest.csv --top-k 10`

## Inference output format

`inference.py` writes a JSON list:

```json
[
  {"image_path": "data/test/fake/0001.jpg", "pred": 0.93},
  {"image_path": "data/test/real/0002.jpg", "pred": 0.04}
]
```

`pred` is the model's confidence (0-1) that the image is AI-generated.

## Robustness evaluation

`eval_robustness.py` runs inference on the test set under all 15 conditions
in the organizer's required-transform grid (clean baseline + every
individual JPEG quality / blur sigma / resize scale / noise sigma / jitter /
crop condition — see "Required Transform Augmentation" above), reporting
R.Acc, F.Acc, accuracy, and AUC per condition, plus which condition is most
damaging to F.Acc specifically. See `results/robustness_table.md` after
running it. Use `--conditions` to run a subset (e.g. `--conditions Clean
JPEG_q30` for a quick check) instead of the full 15-condition sweep.

## Limitations & what we'd improve with more time

- Trained on a subsample of SID_Set (~3,000 images/class); the fully-synthetic
  class is pooled across whatever generator(s) SID_Set used and isn't labeled
  by model, so we can't report per-generator generalization the way a
  GenImage-style multi-generator benchmark would allow. A model trained here
  may not generalize to generator families entirely absent from SID_Set
  (e.g. very recent models released after the dataset was built).
- Tampered (partially-edited) images are held out rather than trained on by
  default — the model may under-perform on partial edits since it never saw
  that failure mode during training. The held-out stress test in step 6
  above quantifies this gap.
- SRM filter bank is a small fixed set (6-8 kernels) rather than the full
  30-filter steganalysis bank, chosen for training speed on limited compute.
- No explicit handling of heavily re-compressed/re-shared social-media
  images (repeated JPEG passes, platform-specific resizing) beyond single-pass
  JPEG augmentation — real-world social media images go through multiple
  transformation passes we don't fully simulate.
- Single-model point estimate; an ensemble (e.g. SRM-branch-only model +
  DINOv2-only model) could give a useful uncertainty signal for borderline cases.
- TTA is currently just horizontal flip averaging; multi-crop or multi-scale
  TTA would likely help images with small/localized artifacts.

## Team

- Jin Han — Model architecture and training. DINOv2 + LoRA backbone,
  SRM residual branch, fusion head, and the training pipeline.
- Jia Jun — Data pipeline and alignment. Dataset adapters, SID_Set
  ingestion, and the Dual Data Alignment implementation.
- Jun Xiang — Evaluation and analysis. Required-transform augmentation
  spec, the 15-condition robustness harness, error analysis, and the
  dataset-bias audit.
