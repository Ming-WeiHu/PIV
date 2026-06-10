"""fn_dynamic_masking.py — per-frame binary fluid mask from raw TIF images.

Four methods:
  * 'tophat' — white top-hat (img − opening) catches bright peaks regardless
    of background level. The right tool for SPARSE BRIGHT PARTICLES on a
    varying background — what happens when the dark gaps between particles
    are darker than the smooth no-particle region overall.
  * 'brightness' — smooth image with box filter, threshold above Otsu.
    Works only when particle regions have HIGHER mean intensity than no-
    particle regions (i.e. dense bright particles on a uniformly dark
    background).
  * 'texture' — local standard deviation + Otsu auto-threshold. Robust to
    lighting and works whether flakes are brighter OR darker than background.
    Use when polarity is uncertain or mean intensities overlap.
  * 'intensity' — original MATLAB algorithm. Threshold pixels at
    `lower_bound`, smooth binary + re-threshold. Fails on sparse particles.

All paths use cv2 morphology for speed (~10× faster than skimage on disk(20)).

Output: dynamic_masking_{condition}.npz
    dynamic_masking : (rows, cols, n_frames) bool
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import skimage.filters
import skimage.io


# Standard rec.601 weights — same as MATLAB rgb2gray.
_RGB2GRAY_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)


def _rgb2gray(img: np.ndarray) -> np.ndarray:
    flat = img[..., :3].astype(np.float64) @ _RGB2GRAY_WEIGHTS
    return flat.astype(img.dtype)


def _list_tif_files(img_folder: Path) -> list[Path]:
    candidates = list(img_folder.glob("*.tif")) + list(img_folder.glob("*.TIF"))
    seen: set[str] = set()
    unique: list[Path] = []
    for f in candidates:
        key = str(f).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return sorted(unique)


def _box_or_raw(img_f: np.ndarray, ksize: int) -> np.ndarray:
    """Box-smooth with the given kernel, or return image unchanged when k<=0.

    Also forces the kernel to be odd (cv2 requirement) when smoothing.
    """
    if ksize is None or ksize <= 0:
        return img_f
    k = int(ksize) | 1   # force odd
    return cv2.boxFilter(img_f, ddepth=-1, ksize=(k, k),
                          borderType=cv2.BORDER_REFLECT)


def _local_std(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Local std-dev over a `ksize × ksize` window using cv2 box filter."""
    img_f = img.astype(np.float32)
    mean = cv2.boxFilter(img_f, ddepth=-1, ksize=(ksize, ksize),
                          borderType=cv2.BORDER_REFLECT)
    sq = cv2.boxFilter(img_f * img_f, ddepth=-1, ksize=(ksize, ksize),
                        borderType=cv2.BORDER_REFLECT)
    var = sq - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def _frame_mask_texture(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Texture-based fluid mask: high local std → has features → fluid+flakes."""
    lstd = _local_std(img, ksize=ksize)
    # Otsu over the per-frame local-std map. Separates "smooth" from "textured".
    try:
        thr = skimage.filters.threshold_otsu(lstd)
    except Exception:
        # Fallback if Otsu fails (e.g. uniform field) — use 25th percentile.
        thr = float(np.percentile(lstd, 25))
    return lstd >= thr


def _frame_mask_brightness(img: np.ndarray, ksize: int = 5,
                           keep_pct: float | None = None) -> np.ndarray:
    """Brightness-based fluid mask: smooth image, threshold.

    For data where particles are reliably brighter than the no-particle
    background.

    Parameters
    ----------
    ksize : int
        Box smooth kernel.
    keep_pct : float | None
        None  — auto threshold via Otsu (fails when histogram is unimodal,
                e.g. evenly-distributed particles across the whole frame).
        70    — keep top 70% of pixels (manual percentile cut).
        Use this override on frames where Otsu produces a degenerate split.
    """
    img_f = img.astype(np.float32)
    smoothed = _box_or_raw(img_f, ksize)
    if keep_pct is None:
        try:
            thr = skimage.filters.threshold_otsu(smoothed)
        except Exception:
            thr = float(np.median(smoothed))
    else:
        thr = float(np.percentile(smoothed, 100.0 - float(keep_pct)))
    return smoothed >= thr


def _frame_mask_tophat(img: np.ndarray,
                       particle_ksize: int = 15,
                       region_ksize: int = 31,
                       keep_pct: float | None = None) -> np.ndarray:
    """Top-hat fluid mask: detect bright peaks regardless of background level.

    `particle_ksize` should be slightly LARGER than the size of a typical
    particle (in pixels). The opening removes structures smaller than this,
    so img − opening keeps the particle peaks.

    `region_ksize` smooths the per-pixel top-hat response into a "density
    of bright features" map.

    `keep_pct` controls how much of each frame is kept.
        None  — use Otsu auto-threshold (fragile on uniform / outlier frames).
        50    — keep top 50% (median split).
        70    — keep top 70% (more permissive — for "too destructive" cases).
        30    — keep top 30% (stricter).
    """
    if img.dtype != np.uint8:
        img_u8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_u8 = img

    particle_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (particle_ksize, particle_ksize))
    tophat = cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, particle_kernel)

    # Smooth the top-hat response so it becomes a density map (lots of nearby
    # bright peaks → high response in the smoothed map).
    smoothed = cv2.boxFilter(tophat.astype(np.float32), ddepth=-1,
                              ksize=(region_ksize, region_ksize),
                              borderType=cv2.BORDER_REFLECT)

    if keep_pct is None:
        try:
            thr = skimage.filters.threshold_otsu(smoothed)
        except Exception:
            thr = float(np.percentile(smoothed, 50))
    else:
        # Keep the top `keep_pct` percent → threshold at the (100 - keep_pct) percentile.
        thr = float(np.percentile(smoothed, 100.0 - float(keep_pct)))

    return smoothed >= thr


def _frame_mask_simple(img: np.ndarray, ksize: int = 15,
                       threshold: float = 10.0,
                       direction: str = "above") -> np.ndarray:
    """Pure manual threshold on a box-smoothed image.

    The simplest possible mask. Use when you want to test what a fixed
    intensity cut does without any auto-tuning.

    Parameters
    ----------
    ksize : int
        Box smooth kernel.
    threshold : float
        Threshold value (in pixel intensity units, 0-255 for uint8).
    direction : {'above', 'below'}
        'above' — keep where smoothed >= threshold (kept = brighter than threshold).
        'below' — keep where smoothed <= threshold (kept = darker than threshold).
    """
    img_f = img.astype(np.float32)
    smoothed = _box_or_raw(img_f, ksize)
    if direction == "below":
        return smoothed <= threshold
    return smoothed >= threshold


def _frame_mask_deviation(img: np.ndarray, ksize: int = 15,
                          threshold: float | None = None) -> np.ndarray:
    """Deviation from background median — catches DENSE and SPARSE particles.

    Otsu-based methods fail when dense-particle regions and no-particle
    regions both look uniform but at DIFFERENT brightness levels (a single
    cut can't separate three classes: dense, no-particle, sparse).

    This method computes |smoothed − median| and thresholds the deviation
    map. Anything that deviates from the no-particle baseline — brighter
    (dense particles) OR darker (sparse particles in dark gaps) — gets
    kept. Only the uniform-medium region around the median is excluded.

    Parameters
    ----------
    ksize : int
        Box smooth kernel.
    threshold : float | None
        Manual deviation threshold. If None, uses Otsu on the deviation map.
    """
    img_f = img.astype(np.float32)
    smoothed = _box_or_raw(img_f, ksize)
    med = float(np.median(smoothed))
    deviation = np.abs(smoothed - med)
    if threshold is None:
        try:
            thr = float(skimage.filters.threshold_otsu(deviation))
        except Exception:
            thr = float(np.percentile(deviation, 50))
    else:
        thr = float(threshold)
    return deviation >= thr


def _frame_mask_intensity(img: np.ndarray, lower_bound: float) -> np.ndarray:
    """Original MATLAB algorithm: threshold pixels >= lower_bound, then smooth."""
    mask_orig = (img >= lower_bound).astype(np.float32)
    # 10x10 box smooth + re-threshold (cv2 equivalent of MATLAB conv2 ones/100)
    smooth = cv2.boxFilter(mask_orig, ddepth=-1, ksize=(10, 10),
                            borderType=cv2.BORDER_CONSTANT)
    return smooth >= 0.5


def _morph_clean(mask: np.ndarray,
                 open_ksize: int = 7,
                 close_ksize: int = 41) -> np.ndarray:
    """Morphological cleanup. 0 disables either step.

    Defaults match the MATLAB algorithm (open ≈ disk(3), close ≈ disk(20)).
    Always applies area-open (removes <2-px components).
    """
    m = mask.astype(np.uint8)
    if open_ksize and open_ksize > 0:
        ok = int(open_ksize) | 1   # force odd
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern)
    if close_ksize and close_ksize > 0:
        ck = int(close_ksize) | 1   # force odd
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern)
    # area-open: remove components < 2 px via connectedComponentsWithStats
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 2
    m = keep[labels].astype(np.uint8)
    return m.astype(bool)


def _resize_to_grid(mask: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Bilinear resize to PIV grid, threshold at 0.5, fill small holes."""
    resized = cv2.resize(mask.astype(np.float32), (cols, rows),
                          interpolation=cv2.INTER_LINEAR)
    out = resized >= 0.5
    # Fill holes smaller than 200 px (cv2 equivalent of ~bwareaopen(~m, 200))
    inv = (~out).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    keep_inv = np.zeros(n, dtype=bool)
    keep_inv[1:] = stats[1:, cv2.CC_STAT_AREA] >= 200
    inv_clean = keep_inv[labels].astype(bool)
    return ~inv_clean


def dynamic_masking(
    img_folder: str | Path,
    *,
    condition: str,
    lower_bound: float,
    mask_rows: int,
    mask_cols: int,
    out_dir: str | Path,
    method: str = "tophat",
    texture_ksize: int = 15,
    tophat_particle_ksize: int = 15,
    tophat_region_ksize: int = 31,
    tophat_keep_pct: float | None = None,
    simple_threshold: float = 128.0,
    simple_direction: str = "above",
    deviation_threshold: float | None = None,
    morph_open_ksize: int = 7,
    morph_close_ksize: int = 41,
    uniform_floor: float = 0.0,
) -> Path:
    """Generate the dynamic mask cube and save to disk.

    Parameters
    ----------
    img_folder : path
        Folder containing one .tif per frame.
    condition : str
        Naming string from parse_filename (e.g. '100mL_15-0-0_20deg').
    lower_bound : float
        Used by method='intensity' only. Pixels >= this value are foreground.
    mask_rows, mask_cols : int
        PIV grid size — masks are resized to match.
    out_dir : path
        Folder for the output .npz.
    method : {'tophat', 'brightness', 'texture', 'intensity'}
        Per-frame foreground detector.
            'tophat'     — top-hat + Otsu (recommended for sparse bright particles)
            'brightness' — smooth + Otsu (dense bright particles only)
            'texture'    — local std + Otsu (polarity-agnostic)
            'intensity'  — original MATLAB threshold algorithm
    texture_ksize : int
        Window size for the local-std / box filter (texture & brightness only).
    tophat_particle_ksize, tophat_region_ksize : int
        Particle kernel and region smoothing window for the top-hat method.
    """
    img_folder = Path(img_folder)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_methods = ("tophat", "brightness", "texture", "intensity",
                      "simple", "deviation")
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got {method!r}")

    files = _list_tif_files(img_folder)
    n = len(files)
    if n == 0:
        raise FileNotFoundError(f"No TIF images found in: {img_folder}")

    print(
        f"  [Stage 2] {n} images found. Method={method}. "
        f"Generating masks at [{mask_rows} x {mask_cols}]..."
    )

    masks = np.zeros((mask_rows, mask_cols, n), dtype=bool)

    for i, f in enumerate(tqdm(
        files, desc=f"  Stage 2 — {condition}", unit="frame", leave=False
    )):
        img = skimage.io.imread(str(f))
        if img.ndim == 3 and img.shape[2] >= 3:
            img = _rgb2gray(img)

        # ---- Uniform-frame safety net -----------------------------------
        # If the frame's smoothed std is below `uniform_floor`, there's no
        # spatial contrast for any threshold to key off — particles are
        # everywhere (or nowhere). Skip the mask and keep the whole frame.
        if uniform_floor > 0.0:
            frame_std = float(img.astype(np.float32).std())
            if frame_std < uniform_floor:
                mask_bin = np.ones(img.shape[:2], dtype=bool)
                mask_clean = _morph_clean(
                    mask_bin,
                    open_ksize=morph_open_ksize,
                    close_ksize=morph_close_ksize,
                )
                masks[:, :, i] = _resize_to_grid(mask_clean, mask_rows, mask_cols)
                continue

        if method == "tophat":
            mask_bin = _frame_mask_tophat(
                img,
                particle_ksize=tophat_particle_ksize,
                region_ksize=tophat_region_ksize,
                keep_pct=tophat_keep_pct,
            )
        elif method == "texture":
            mask_bin = _frame_mask_texture(img, ksize=texture_ksize)
        elif method == "brightness":
            mask_bin = _frame_mask_brightness(
                img, ksize=texture_ksize, keep_pct=tophat_keep_pct,
            )
        elif method == "simple":
            mask_bin = _frame_mask_simple(
                img,
                ksize=texture_ksize,
                threshold=simple_threshold,
                direction=simple_direction,
            )
        elif method == "deviation":
            mask_bin = _frame_mask_deviation(
                img,
                ksize=texture_ksize,
                threshold=deviation_threshold,
            )
        else:
            mask_bin = _frame_mask_intensity(img, lower_bound=lower_bound)

        mask_clean = _morph_clean(
            mask_bin,
            open_ksize=morph_open_ksize,
            close_ksize=morph_close_ksize,
        )
        mask_resized = _resize_to_grid(mask_clean, mask_rows, mask_cols)
        masks[:, :, i] = mask_resized

    out_path = out_dir / f"dynamic_masking_{condition}.npz"
    np.savez_compressed(out_path, dynamic_masking=masks,
                         method=np.array(method))
    print(f"  [Stage 2] Saved: {out_path}")
    return out_path
