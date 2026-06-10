"""diff_mask.py — compare the SAVED pipeline mask to a recomputed one.

For one frame, runs the brightness algorithm from scratch with the params
you passed to the pipeline and compares to what's in the dynamic_masking_*.npz.
Any difference means the saved mask doesn't match what the algorithm should
produce.

Usage
-----
    python -m piv_pipeline.diff_mask \\
        --output-root "C:\\...\\PipelineTest_final" \\
        --tif-folder  "C:\\...\\100mL-20deg-35cpm" \\
        --frame 1000 \\
        --ksize 0 --keep-pct 68 --uniform-floor 16086 \\
        --open 7 --close 41
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
import skimage.filters
import skimage.io


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", required=True)
    p.add_argument("--tif-folder", required=True)
    p.add_argument("--frame", type=int, default=1000)
    p.add_argument("--method", default="brightness")
    p.add_argument("--ksize", type=int, default=0)
    p.add_argument("--keep-pct", type=float, default=68)
    p.add_argument("--uniform-floor", type=float, default=16086)
    p.add_argument("--open", dest="open_k", type=int, default=7)
    p.add_argument("--close", dest="close_k", type=int, default=41)
    p.add_argument("--out", default="diff_mask.png")
    args = p.parse_args()

    out_root = Path(args.output_root)
    tif_folder = Path(args.tif_folder)

    # Load saved mask
    mask_paths = sorted((out_root / "Dynamic Masks").glob("dynamic_masking_*.npz"))
    if not mask_paths:
        raise SystemExit(f"No dynamic_masking npz in {out_root}/Dynamic Masks")
    saved = np.load(mask_paths[0], allow_pickle=False)
    saved_method = str(saved["method"]) if "method" in saved.files else "?"
    saved_masks = saved["dynamic_masking"]
    print(f"Saved file:    {mask_paths[0].name}")
    print(f"Saved method:  {saved_method}")
    print(f"Saved shape:   {saved_masks.shape}")
    print(f"Saved dtype:   {saved_masks.dtype}")

    # Load TIF — must dedup like the pipeline's _list_tif_files (Windows
    # globs are case-insensitive in older Pythons, so *.tif and *.TIF
    # return the same files and a naive sort gives every file twice).
    _all = list(tif_folder.glob("*.tif")) + list(tif_folder.glob("*.TIF"))
    _seen: set[str] = set()
    tifs: list = []
    for f in _all:
        k = str(f).lower()
        if k in _seen:
            continue
        _seen.add(k)
        tifs.append(f)
    tifs = sorted(tifs)
    idx = max(0, min(args.frame, len(tifs) - 1, saved_masks.shape[2] - 1))
    print(f"TIF count (deduped): {len(tifs)}")
    print(f"Frame index {args.frame} resolves to: {tifs[idx].name}")
    img = skimage.io.imread(str(tifs[idx]))
    if img.ndim == 3:
        img = img.mean(axis=2)
    print(f"Frame {idx}: shape={img.shape}, dtype={img.dtype}, "
          f"range=[{img.min()}, {img.max()}]")

    img_f = img.astype(np.float32)
    frame_std = float(img_f.std())
    print(f"\nframe_std = {frame_std:.1f}")
    print(f"uniform_floor = {args.uniform_floor}")
    triggers = args.uniform_floor > 0 and frame_std < args.uniform_floor
    print(f"safety net triggers: {triggers}")

    # ---- Recompute mask from scratch ------------------------------------
    if triggers:
        mask_bin = np.ones(img.shape[:2], dtype=bool)
    else:
        if args.ksize <= 0:
            smoothed = img_f
        else:
            k = args.ksize | 1
            smoothed = cv2.boxFilter(img_f, -1, (k, k), borderType=cv2.BORDER_REFLECT)
        if args.keep_pct is not None:
            thr = float(np.percentile(smoothed, 100.0 - args.keep_pct))
        else:
            thr = float(skimage.filters.threshold_otsu(smoothed))
        mask_bin = smoothed >= thr
        print(f"threshold = {thr:.1f}")
        print(f"pre-morph kept: {mask_bin.mean()*100:.2f}%")

    # Morph
    m = mask_bin.astype(np.uint8)
    if args.open_k > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.open_k|1, args.open_k|1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    if args.close_k > 0:
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.close_k|1, args.close_k|1))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    # area-open <2
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 2
    m = keep[labels].astype(np.uint8)

    # Resize
    rows, cols = saved_masks.shape[:2]
    resized = cv2.resize(m.astype(np.float32), (cols, rows), interpolation=cv2.INTER_LINEAR)
    out = resized >= 0.5
    # Hole fill
    inv = (~out).astype(np.uint8)
    n_h, lbls, stts, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep_inv = np.zeros(n_h, dtype=bool)
    keep_inv[1:] = stts[1:, cv2.CC_STAT_AREA] >= 200
    recomputed = ~keep_inv[lbls].astype(bool)

    saved_mask = saved_masks[:, :, idx].astype(bool)

    print(f"\nrecomputed:  {recomputed.mean()*100:.2f}% kept")
    print(f"saved:       {saved_mask.mean()*100:.2f}% kept")
    diff = recomputed != saved_mask
    print(f"differing pixels: {diff.sum()} / {diff.size} "
          f"({diff.mean()*100:.2f}%)")
    print(f"equal: {np.array_equal(recomputed, saved_mask)}")

    # ---- Visualize -----------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Frame {idx} raw")
    axes[1].imshow(recomputed, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Recomputed mask ({recomputed.mean()*100:.1f}% kept)")
    axes[2].imshow(saved_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Saved pipeline mask ({saved_mask.mean()*100:.1f}% kept)")
    axes[3].imshow(diff, cmap="hot", vmin=0, vmax=1)
    axes[3].set_title(f"Diff (white = differ)\n{diff.sum()} pixels")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nSaved: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
