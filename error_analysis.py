"""
Error Analysis Note — REQUIRED DELIVERABLE.

Reads predictions.json (from inference.py) + manifest.csv (from
prepare_dataset.py, for ground-truth labels + generator family), and reports:
  - top-K most confident false positives (real images predicted as fake)
  - top-K most confident false negatives (fake images predicted as real)
  - per-generator-family accuracy breakdown

Usage:
    python error_analysis.py --predictions predictions.json \
        --manifest data/manifest.csv --top-k 10 --output results/error_analysis.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict


def load_ground_truth(manifest_path: str) -> dict[str, dict]:
    gt = {}
    with open(manifest_path, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["image_path"]] = {
                "label": int(row["label"]),
                "generator_family": row.get("generator_family", ""),
            }
    return gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", default="results/error_analysis.md")
    args = parser.parse_args()

    with open(args.predictions) as f:
        preds = json.load(f)
    gt = load_ground_truth(args.manifest)

    joined = []
    for p in preds:
        path = p["image_path"]
        if path not in gt:
            continue  # image not in manifest (e.g. held-out unlabeled set)
        joined.append({
            "path": path,
            "pred": p["pred"],
            "label": gt[path]["label"],
            "generator_family": gt[path]["generator_family"],
        })

    false_positives = [r for r in joined if r["label"] == 0 and r["pred"] >= args.threshold]
    false_negatives = [r for r in joined if r["label"] == 1 and r["pred"] < args.threshold]

    false_positives.sort(key=lambda r: r["pred"], reverse=True)   # most-confidently-wrong first
    false_negatives.sort(key=lambda r: r["pred"])                  # most-confidently-wrong first

    # per-generator-family accuracy (fakes only, since real images have no family)
    family_correct = defaultdict(int)
    family_total = defaultdict(int)
    for r in joined:
        if r["label"] != 1:
            continue
        fam = r["generator_family"] or "unknown"
        family_total[fam] += 1
        if r["pred"] >= args.threshold:
            family_correct[fam] += 1

    lines = ["# Error Analysis", ""]
    lines.append(f"Total scored: {len(joined)} | "
                 f"False positives: {len(false_positives)} | "
                 f"False negatives: {len(false_negatives)}")
    lines.append("")

    lines.append(f"## Top {args.top_k} False Positives (real image, predicted fake)")
    lines.append("")
    lines.append("| image_path | pred |")
    lines.append("|---|---|")
    for r in false_positives[: args.top_k]:
        lines.append(f"| {r['path']} | {r['pred']:.4f} |")
    lines.append("")

    lines.append(f"## Top {args.top_k} False Negatives (fake image, predicted real)")
    lines.append("")
    lines.append("| image_path | pred | generator_family |")
    lines.append("|---|---|---|")
    for r in false_negatives[: args.top_k]:
        lines.append(f"| {r['path']} | {r['pred']:.4f} | {r['generator_family']} |")
    lines.append("")

    lines.append("## Per-Generator-Family Recall (fakes correctly flagged)")
    lines.append("")
    lines.append("| generator_family | recall | n |")
    lines.append("|---|---|---|")
    for fam in sorted(family_total):
        recall = family_correct[fam] / family_total[fam] if family_total[fam] else 0.0
        lines.append(f"| {fam} | {recall:.4f} | {family_total[fam]} |")
    lines.append("")

    lines.append("## Trade-offs")
    lines.append("")
    lines.append("Every choice trades one thing for another. Mapped to this pipeline specifically:")
    lines.append("")
    lines.append("**Robustness vs. clean accuracy.** `RequiredTrainingAugmentation` applies one of "
                 "six degradations (JPEG/blur/resize/noise/jitter/crop) to ~70% of training images "
                 "(`apply_prob=0.7`), which costs some clean-image accuracy in exchange for not "
                 "collapsing under JPEG re-compression or noise at test time — see AIGIBench's finding "
                 "that undegraded training leads to F.Acc near 0% under perturbation. Check "
                 "`results/robustness_table.md`'s Clean row against the worst-case condition: a large "
                 "gap confirms the trade was worth it; a small gap on Clean specifically means the cost "
                 "was low. **[Fill in once robustness_table.md exists: Clean acc = ___, worst-condition "
                 "acc = ___]**")
    lines.append("")
    lines.append("**Generalization vs. specialization.** DDA-aligned fakes (VAE reconstruction of real "
                 "images, generator-agnostic by construction) are used alongside — or instead of — "
                 "SID_Set's specific fake images, trading potential peak accuracy on SID_Set's own test "
                 "distribution for better generalization to generators absent from training entirely "
                 "(this is DDA's central claim: a detector trained exclusively on DDA-aligned data beat "
                 "specialized baselines across 11 unseen-generator benchmarks). Check the per-source "
                 "recall table above: if `dda`-tagged recall is lower than `full_synthetic`-tagged "
                 "recall on this specific test set, that's the specialization cost showing up directly — "
                 "expected, and the point of using DDA in the first place. "
                 "**[Fill in: full_synthetic recall = ___, dda recall = ___, tampered recall = ___]**")
    lines.append("")
    lines.append("**Complexity vs. feasibility.** The architecture is already a 2-branch fusion "
                 "(DINOv2+LoRA backbone + a fixed-filter SRM residual branch) — the SRM branch was kept "
                 "deliberately cheap (a handful of fixed, non-trainable high-pass kernels, no extra "
                 "training cost) rather than a second full trainable backbone, specifically to avoid the "
                 "\"2-branch ensemble wins 1% but costs the demo\" trap. One complexity cost we did accept: "
                 "DDA's VAE-reconstruction preprocessing step adds real wall-clock time to dataset prep. "
                 "One we declined: the DDA paper's custom paired-batch sampler (real + its own DDA "
                 "counterpart in the same batch) was left unimplemented — a real (if secondary) part of "
                 "their reported result, cut here in favor of shipping a working end-to-end pipeline on "
                 "time.")
    lines.append("")
    lines.append("There is no silver bullet here — these are the trade-offs we made explicitly, not ones "
                 "we're unaware of.")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report + "\n")

    print(report)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
