"""
piv_simple.py — simplified Python PIV processor.

Ported from the algorithms in PIVlab (MATLAB, Shrediquette) and the OpenPIV
Python build bundled in WeheliyeHashi/PIV_Oribiotech_gui. Implements FFT-based
cross-correlation with multi-pass window deformation, three sub-pixel peak
estimators, and the preprocessing pipeline from PIVlab's `Image pre-processing`
panel.

Excluded by design:
  * ensemble correlation (averaging correlation maps across many frame pairs);
  * neural-network wall masking;
  * smoothn smoothing and the full validation pipeline.

The three settings dataclasses mirror the GUI panels one-to-one:

    PreprocSettings      ↔  Image pre-processing (CTRL+I)
    PIVSettings          ↔  PIV settings        (CTRL+S)
    CalibrationSettings  ↔  Calibration         (CTRL+Z)

Public API:

    preprocess(img, settings, background=None) -> float32 image in [0, 1]
    load_mask(path) -> bool array (True = analyse, False = exclude)
    apply_calibration(x, y, u, v, calibration) -> (x_m, y_m, u_ms, v_ms)
    run_piv(frame_a, frame_b, piv_settings, preproc_settings=None,
            background=None, calibration=None, mask=None)
        -> (x, y, u, v) on the final-pass grid; pixels by default,
           or metres / m·s⁻¹ when `calibration` is supplied.
           Vectors over masked-out regions are NaN.

Run on a pair of images from the command line:

    python piv_simple.py frame_a.png frame_b.png [--out vectors.npz]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, median_filter
from scipy.signal import wiener

# Silence libtiff metadata warnings (e.g. "Software" tag with embedded null
# bytes) that OpenCV routes to stderr on every imread. The pixel data is
# unaffected — only ASCII metadata strings get truncated. We still want
# real ERROR-level messages from cv2.
try:
    cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 3))
except AttributeError:
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass


# ───────────────────────────── Settings ───────────────────────────────────────

@dataclass
class PreprocSettings:
    """Mirrors the `Image pre-processing (CTRL+I)` panel."""

    # CLAHE
    enable_clahe: bool = True
    clahe_window_size: int = 64           # px per tile
    clahe_clip_limit: float = 0.01         # PIVlab's adapthisteq 'ClipLimit'

    # Highpass (Gaussian, image - lowpass)
    enable_highpass: bool = False
    highpass_kernel_size: int = 15         # px; sigma ≈ kernel_size / 6

    # Intensity capping at median + N * std
    enable_intensity_capping: bool = False
    intensity_cap_n_std: float = 2.0

    # Wiener2 denoise + low-pass
    enable_wiener2: bool = False
    wiener_window_size: int = 3

    # Auto contrast stretch (imadjust [lo, hi] → [0, 1])
    enable_contrast_stretch: bool = True
    auto_minmax: bool = True               # auto-compute lo/hi from percentiles
    contrast_min: float = 0.0              # used when auto_minmax is False
    contrast_max: float = 1.0
    auto_min_percentile: float = 1.0       # only used when auto_minmax is True
    auto_max_percentile: float = 99.0

    # Background subtraction
    subtract_mean_intensity: bool = False  # subtract image mean (or supplied bg)


@dataclass
class PIVSettings:
    """Mirrors the `PIV settings (CTRL+S)` panel."""

    # PIV algorithm — "ensemble" is intentionally not supported.
    algorithm: str = "fft_windef"          # "fft_windef" | "dcc"

    # Pass interrogation areas and steps, in pixels.
    # Length sets the number of passes. Defaults match the GUI defaults
    # (64/32 → 32/16 → 16/8, with pass 4 disabled).
    window_sizes: Tuple[int, ...] = (64, 32, 16)
    steps: Tuple[int, ...] = (32, 16, 8)

    # "Repeat last pass until quality slope < X"
    repeat_last_pass: bool = False
    quality_slope_threshold: float = 0.025
    repeat_max_iterations: int = 5

    # Sub-pixel estimator
    subpixel_method: str = "gauss2x3"      # "gauss2x3" | "centroid" | "parabolic"

    # When True, skip per-window zero-mean subtraction (so the DC / auto-
    # correlation peak is left in). The GUI checkbox of the same name.
    disable_autocorrelation: bool = False

    # "Standard" uses circular FFT (no padding); "extreme" zero-pads to 2N-1
    # (linear correlation), which kills wrap-around artefacts.
    correlation_robustness: str = "standard"  # "standard" | "extreme"

    # Multi-pass outlier handling between passes — not exposed in the GUI but
    # required for the deformation step to be stable.
    replace_outliers: bool = True
    median_filter_size: int = 3
    outlier_threshold: float = 2.0         # multiples of local median absolute deviation

    # ── Velocity cap (outlier-vector rejection on |v|) ───────────────────────
    # Rejects whole vectors whose VERTICAL speed |v| exceeds a cap, removing
    # the spurious large-displacement tail that cyclic-FFT wrap-around emits
    # (which inflates the vertical mean). When |v| > cap, BOTH u and v at that
    # cell are NaN'd, in the filtered and `_original` fields — the same
    # whole-vector removal the cap-explorer export performed.
    # Three ways to set the cap, in precedence order:
    #   velocity_cap_px         — absolute |v| cap in PIXELS/FRAME
    #   velocity_cap_fraction   — cap = fraction × smallest interrogation window
    #   velocity_cap_percentile — drop the top X% of vectors by |v|
    # Default: absolute 5.4 px/frame (≈428 mm/s at calv=0.0795), the validated
    # global |v| cut from the cap explorer (~0.7% of the 100mL-35cpm data).
    # Set all three to None to disable.
    velocity_cap_px: Optional[float] = 5.4
    velocity_cap_fraction: Optional[float] = None
    velocity_cap_percentile: Optional[float] = None


@dataclass
class CalibrationSettings:
    """Mirrors the `Calibration (CTRL+Z)` panel.

    Pixel ↔ world conversion is:

        m_per_pixel = (real_distance_mm / 1000) / reference_length_px
        m_per_s     = m_per_pixel * 1000 / time_step_ms

    Axis sign flips handle the two dropdowns; offsets shift the world origin.
    Default values match the GUI defaults exactly: 440 px = 70 mm, dt = 2 ms,
    which yields 1 px = 1.5909e-4 m and 1 px/frame = 0.0795 m/s.
    """

    reference_length_px: float = 440.0 #check maybe
    real_distance_mm: float = 70.0
    time_step_ms: float = 2.0
    x_increases_towards: str = "right"      # "right" | "left"
    y_increases_towards: str = "bottom"     # "bottom" | "top"
    x_offset_m: float = 0.0
    y_offset_m: float = 0.0
    optimize_display: bool = True           # for an optional viewer, no-op here
    calibration_image: Optional[str] = None  # path; only stored for reference

    @property
    def m_per_pixel(self) -> float:
        return (self.real_distance_mm / 1000.0) / self.reference_length_px

    @property
    def m_per_second_per_px_per_frame(self) -> float:
        return self.m_per_pixel * 1000.0 / self.time_step_ms

    @property
    def x_sign(self) -> int:
        return +1 if self.x_increases_towards == "right" else -1

    @property
    def y_sign(self) -> int:
        return +1 if self.y_increases_towards == "bottom" else -1


# ─────────────────────────── Pre-processing ───────────────────────────────────

def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        return (img.astype(np.float32) - info.min) / (info.max - info.min)
    return img.astype(np.float32)


def _to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def _renorm01(img: np.ndarray) -> np.ndarray:
    img = img - img.min()
    m = img.max()
    return img / m if m > 0 else img


def apply_clahe(img_u8: np.ndarray, window_size: int = 64,
                clip_limit: float = 0.01) -> np.ndarray:
    """CLAHE with a tile grid sized so each tile spans ~window_size pixels.

    Matches PIVlab's `adapthisteq(..., 'NumTiles', [ny nx], 'ClipLimit', 0.01)`.
    OpenCV's clipLimit is expressed differently; the 0.01 scale tracks PIVlab's
    convention multiplied by 256 to land in OpenCV's domain.
    """
    h, w = img_u8.shape
    n_tiles_y = max(2, h // window_size)
    n_tiles_x = max(2, w // window_size)
    # PIVlab's adapthisteq ClipLimit is a fraction of tile pixels.
    # OpenCV's clipLimit is an absolute count per histogram bin.
    # Correct mapping: opencv_clip = matlab_limit × tile_height × tile_width
    # The old `clip_limit * 256` was tile-size-independent and ~16× too
    # aggressive for typical 64 px tiles, over-enhancing noise.
    tile_h = max(1, h // n_tiles_y)
    tile_w = max(1, w // n_tiles_x)
    opencv_clip = clip_limit * tile_h * tile_w
    clahe = cv2.createCLAHE(
        clipLimit=opencv_clip,
        tileGridSize=(n_tiles_x, n_tiles_y),
    )
    return clahe.apply(img_u8)


def apply_highpass(img: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """Subtract a Gaussian low-pass of `img`. Negatives are clamped to zero
    (matches PIVlab's behaviour after `imfilter(...,'replicate')` subtraction).
    """
    sigma = max(kernel_size / 6.0, 0.5)
    low = gaussian_filter(img.astype(np.float32), sigma=sigma, mode="nearest")
    out = img.astype(np.float32) - low
    return np.clip(out, 0.0, None)


def apply_intensity_capping(img: np.ndarray, n_std: float = 2.0) -> np.ndarray:
    """Cap pixels above `median + n_std * std` (PIVlab's bright-outlier guard)."""
    threshold = float(np.median(img)) + n_std * float(np.std(img))
    return np.minimum(img, threshold)


def apply_wiener2(img: np.ndarray, window: int = 3) -> np.ndarray:
    """Adaptive Wiener filter via `scipy.signal.wiener` (the wiener2 analogue)."""
    # scipy.signal.wiener divides by local variance, which is zero in flat
    # regions; ignore the warning and clean up the NaNs/infs afterwards.
    with np.errstate(divide="ignore", invalid="ignore"):
        out = wiener(img.astype(np.float64), mysize=window)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(img.dtype)


def apply_contrast_stretch(img01: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """imadjust([lo, hi], [0, 1]) — linear stretch with clipping at the ends."""
    if hi <= lo:
        return img01
    return np.clip((img01 - lo) / (hi - lo), 0.0, 1.0)


def auto_contrast_limits(img01: np.ndarray,
                         lo_pct: float = 1.0,
                         hi_pct: float = 99.0) -> Tuple[float, float]:
    return (
        float(np.percentile(img01, lo_pct)),
        float(np.percentile(img01, hi_pct)),
    )


def preprocess(img: np.ndarray,
               settings: PreprocSettings,
               background: Optional[np.ndarray] = None) -> np.ndarray:
    """Run the preprocessing pipeline in PIVlab's order.

    Returns a float32 image in [0, 1].
    """
    work = _to_float01(img)

    if settings.subtract_mean_intensity:
        if background is not None:
            work = work - _to_float01(background)
        else:
            work = work - work.mean()
        work = _renorm01(work)

    if settings.enable_clahe:
        work = _to_float01(apply_clahe(
            _to_uint8(work),
            settings.clahe_window_size,
            settings.clahe_clip_limit,
        ))

    if settings.enable_highpass:
        work = _renorm01(apply_highpass(work, settings.highpass_kernel_size))

    if settings.enable_intensity_capping:
        work = _renorm01(apply_intensity_capping(work, settings.intensity_cap_n_std))

    if settings.enable_wiener2:
        work = apply_wiener2(work, settings.wiener_window_size)

    if settings.enable_contrast_stretch:
        if settings.auto_minmax:
            lo, hi = auto_contrast_limits(
                work, settings.auto_min_percentile, settings.auto_max_percentile,
            )
        else:
            lo, hi = settings.contrast_min, settings.contrast_max
        work = apply_contrast_stretch(work, lo, hi)

    return work.astype(np.float32)


# ───────────────────────────── PIV core ───────────────────────────────────────

def field_shape(image_shape: Tuple[int, int],
                window_size: int, step: int) -> Tuple[int, int]:
    h, w = image_shape
    return (h - window_size) // step + 1, (w - window_size) // step + 1


def window_coordinates(image_shape: Tuple[int, int],
                       window_size: int, step: int
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_centers, y_centers) — pixel coordinates of each interrogation
    window's centre. Shape: (n_rows, n_cols) each.
    """
    n_rows, n_cols = field_shape(image_shape, window_size, step)
    half = window_size / 2.0
    y = np.arange(n_rows) * step + half - 0.5
    x = np.arange(n_cols) * step + half - 0.5
    return np.meshgrid(x, y)


def sliding_windows(image: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Extract overlapping windows. Shape: (n_rows, n_cols, ws, ws)."""
    n_rows, n_cols = field_shape(image.shape, window_size, step)
    # np.lib.stride_tricks.sliding_window_view gives all overlaps; subsample by step.
    all_win = np.lib.stride_tricks.sliding_window_view(
        image, (window_size, window_size),
    )
    return np.ascontiguousarray(all_win[::step, ::step][:n_rows, :n_cols])


def _normalize_per_window(stack: np.ndarray) -> np.ndarray:
    return stack - stack.mean(axis=(-2, -1), keepdims=True)


def fft_correlate(win_a: np.ndarray, win_b: np.ndarray,
                  normalize: bool = True,
                  mode: str = "circular") -> np.ndarray:
    """Cross-correlate a stack of window pairs via FFT.

    `mode = "circular"` — no zero-padding (PIVlab's default, fastest).
    `mode = "linear"`   — zero-pad to 2N-1, eliminating wrap-around.
    """
    if normalize:
        win_a = _normalize_per_window(win_a.astype(np.float32))
        win_b = _normalize_per_window(win_b.astype(np.float32))
    else:
        win_a = win_a.astype(np.float32)
        win_b = win_b.astype(np.float32)

    ws = win_a.shape[-1]
    fft_size = (2 * ws - 1, 2 * ws - 1) if mode == "linear" else (ws, ws)
    fa = np.fft.rfft2(win_a, s=fft_size)
    fb = np.fft.rfft2(win_b, s=fft_size)
    corr = np.fft.irfft2(np.conj(fa) * fb, s=fft_size)
    return np.fft.fftshift(corr, axes=(-2, -1))


def _peak_indices(corr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = corr.shape[-2:]
    flat = corr.reshape(*corr.shape[:-2], -1).argmax(axis=-1)
    return flat // w, flat % w


def find_subpixel_peak(corr: np.ndarray,
                       method: str = "gauss2x3") -> Tuple[np.ndarray, np.ndarray]:
    """Refine the correlation peak to sub-pixel precision.

    Returns (dy, dx): displacement of each correlation map's peak relative to
    its centre, in pixels. Positive dy = downward shift; positive dx = rightward.
    """
    h, w = corr.shape[-2:]
    cy, cx = h // 2, w // 2
    i_peak, j_peak = _peak_indices(corr)

    # Clamp away from the boundary so neighbours are valid; the offset is set
    # to zero on boundary peaks below.
    i_safe = np.clip(i_peak, 1, h - 2)
    j_safe = np.clip(j_peak, 1, w - 2)

    # Gather peak-neighbour values per correlation map.
    if corr.ndim == 2:
        def gather(di, dj):
            return corr[i_safe + di, j_safe + dj]
    else:
        grid = np.indices(i_safe.shape)
        def gather(di, dj):
            return corr[(*grid, i_safe + di, j_safe + dj)]

    c0  = gather(0,  0)
    cN  = gather(-1, 0)   # north (i-1)
    cS  = gather(+1, 0)   # south (i+1)
    cW  = gather(0, -1)   # west  (j-1)
    cE  = gather(0, +1)   # east  (j+1)

    if method == "gauss2x3":
        eps = 1e-8
        c0n  = np.log(np.maximum(c0,  eps))
        cNn  = np.log(np.maximum(cN,  eps))
        cSn  = np.log(np.maximum(cS,  eps))
        cWn  = np.log(np.maximum(cW,  eps))
        cEn  = np.log(np.maximum(cE,  eps))
        denom_y = 2 * (cNn - 2 * c0n + cSn)
        denom_x = 2 * (cWn - 2 * c0n + cEn)
        with np.errstate(divide="ignore", invalid="ignore"):
            sy = np.where(np.abs(denom_y) > eps, (cNn - cSn) / denom_y, 0.0)
            sx = np.where(np.abs(denom_x) > eps, (cWn - cEn) / denom_x, 0.0)
    elif method == "parabolic":
        denom_y = 2 * (cN - 2 * c0 + cS)
        denom_x = 2 * (cW - 2 * c0 + cE)
        with np.errstate(divide="ignore", invalid="ignore"):
            sy = np.where(np.abs(denom_y) > 1e-12, (cN - cS) / denom_y, 0.0)
            sx = np.where(np.abs(denom_x) > 1e-12, (cW - cE) / denom_x, 0.0)
    elif method == "centroid":
        denom_y = cN + c0 + cS
        denom_x = cW + c0 + cE
        with np.errstate(divide="ignore", invalid="ignore"):
            sy = np.where(np.abs(denom_y) > 1e-12, (cS - cN) / denom_y, 0.0)
            sx = np.where(np.abs(denom_x) > 1e-12, (cE - cW) / denom_x, 0.0)
    else:
        raise ValueError(f"unknown subpixel_method: {method!r}")

    # Clamp huge sub-pixel offsets caused by noisy peaks.
    sy = np.clip(sy, -1.0, 1.0)
    sx = np.clip(sx, -1.0, 1.0)

    # Boundary peaks: zero sub-pixel offset.
    on_boundary = (i_peak != i_safe) | (j_peak != j_safe)
    sy = np.where(on_boundary, 0.0, sy)
    sx = np.where(on_boundary, 0.0, sx)

    dy = (i_peak.astype(np.float32) - cy) + sy.astype(np.float32)
    dx = (j_peak.astype(np.float32) - cx) + sx.astype(np.float32)
    return dy, dx


def correlate_pass(frame_a: np.ndarray, frame_b: np.ndarray,
                   window_size: int, step: int,
                   settings: PIVSettings
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One PIV pass on aligned frames. Returns (x_centers, y_centers, u, v)."""
    win_a = sliding_windows(frame_a, window_size, step)
    win_b = sliding_windows(frame_b, window_size, step)
    mode = "linear" if settings.correlation_robustness == "extreme" else "circular"
    normalize = not settings.disable_autocorrelation
    corr = fft_correlate(win_a, win_b, normalize=normalize, mode=mode)
    dy, dx = find_subpixel_peak(corr, settings.subpixel_method)
    x_c, y_c = window_coordinates(frame_a.shape, window_size, step)
    # u is along x (cols), v is along y (rows). Positive v = downward.
    return x_c, y_c, dx, dy


# ──────────────────────── Multi-pass + deformation ────────────────────────────

def _replace_outliers(u: np.ndarray, v: np.ndarray,
                      kernel: int = 3, threshold: float = 2.0
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Replace outlier vectors with the local median (NaN-aware).

    Outliers are detected with a normalised median test (Westerweel & Scarano
    2005). Vectors whose deviation from the local median exceeds
    `threshold` × local MAD are replaced by that median. NaN entries
    (masked-out windows) are preserved as NaN through the call.
    """
    out_u = u.copy()
    out_v = v.copy()
    for arr, out in ((u, out_u), (v, out_v)):
        nan_mask = np.isnan(arr)
        arr_safe = np.where(nan_mask, 0.0, arr)
        med = median_filter(arr_safe, size=kernel, mode="nearest")
        resid = arr_safe - med
        mad = median_filter(np.abs(resid), size=kernel, mode="nearest") + 0.1
        bad = (np.abs(resid) > threshold * mad) & ~nan_mask
        out[bad] = med[bad]
        out[nan_mask] = np.nan
    return out_u, out_v


def load_mask(path: str) -> np.ndarray:
    """Load a mask image (PIVlab convention). Any channel is OK — RGBA masks
    just repeat the same intensity across channels.

    Convention: **bright pixels (>127) are the EXCLUDED region** (drawn
    polygon = mask), dark pixels (≤127) are the analysable ROI. Returns a
    bool array where True = analyse, False = exclude.
    """
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"could not load mask: {path}")
    if raw.ndim == 3:
        # If there's an alpha channel that's all-opaque, drop it; otherwise
        # take the first (R) channel. RGB masks store the same value in all
        # three channels, so this is equivalent to grayscale conversion.
        raw = raw[..., 0]
    return raw <= 127


def _sample_mask_at_centers(mask_keep: np.ndarray,
                            x_c: np.ndarray,
                            y_c: np.ndarray) -> np.ndarray:
    """Sample a per-pixel boolean mask at the (x_c, y_c) window centres.

    Returns a bool array of shape `x_c.shape` — True where the window centre
    falls in the analyse region.
    """
    h, w = mask_keep.shape
    xi = np.clip(np.round(x_c).astype(int), 0, w - 1)
    yi = np.clip(np.round(y_c).astype(int), 0, h - 1)
    return mask_keep[yi, xi]


def _upsample_to_pixel_grid(x_c: np.ndarray, y_c: np.ndarray,
                            u: np.ndarray, v: np.ndarray,
                            image_shape: Tuple[int, int]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Upsample a coarse (u, v) field on (x_c, y_c) to per-pixel grids.

    Uses cv2.remap-friendly bilinear interpolation via cv2.resize after
    extracting the 1-D coordinate axes. The coarse grid is regular by
    construction (`window_coordinates` is a meshgrid).
    """
    h, w = image_shape
    n_rows, n_cols = u.shape

    # X centres are constant along rows of x_c; Y centres are constant along cols.
    xs = x_c[0, :]
    ys = y_c[:, 0]

    # Build per-pixel sample coordinates in the coarse grid's frame.
    px = np.arange(w, dtype=np.float32)
    py = np.arange(h, dtype=np.float32)
    # Map pixel x → fractional column in u/v.
    fx = np.interp(px, xs, np.arange(n_cols, dtype=np.float32))
    fy = np.interp(py, ys, np.arange(n_rows, dtype=np.float32))

    # cv2.remap expects (map_x, map_y) at every output pixel.
    grid_fx, grid_fy = np.meshgrid(fx, fy)
    grid_fx = grid_fx.astype(np.float32)
    grid_fy = grid_fy.astype(np.float32)

    u_dense = cv2.remap(u.astype(np.float32), grid_fx, grid_fy,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    v_dense = cv2.remap(v.astype(np.float32), grid_fx, grid_fy,
                        interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
    return u_dense, v_dense


def _deform_image(image: np.ndarray,
                  u_pixel: np.ndarray, v_pixel: np.ndarray,
                  order: int = 3) -> np.ndarray:
    """Backward-warp `image` so it aligns with the reference frame.

    Each output pixel (y, x) samples the source at (y + v, x + u).
    """
    h, w = image.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    coords = np.stack([yy + v_pixel, xx + u_pixel], axis=0)
    return map_coordinates(image, coords, order=order, mode="nearest")


def multipass_piv(frame_a: np.ndarray, frame_b: np.ndarray,
                  settings: PIVSettings,
                  mask_keep: Optional[np.ndarray] = None,
                  return_originals: bool = False
                  ):
    """Run multi-pass PIV with window deformation.

    Returns (x, y, u, v) on the final pass's grid. `u` is in pixels along the
    image's x-axis (columns), `v` along the y-axis (rows, positive downward).

    If `mask_keep` is supplied (bool, same shape as the frames), masked-out
    pixels (False) are replaced by the local mean before correlation, and
    output vectors whose window centre falls in a masked-out region are set
    to NaN — matching PIVlab's behaviour.

    If `return_originals=True`, returns (x, y, u_filtered, v_filtered,
    u_original, v_original) — the `_original` arrays are the raw correlation
    output of the final pass *before* outlier replacement, matching PIVlab's
    `u_original` / `v_original` session fields.
    """
    if mask_keep is not None:
        if mask_keep.shape != frame_a.shape:
            raise ValueError(
                f"mask shape {mask_keep.shape} != frame shape {frame_a.shape}")
        # Replace excluded pixels with the mean of the keep region so they
        # don't bias the FFT cross-correlation peak.
        fill_a = float(frame_a[mask_keep].mean()) if mask_keep.any() else 0.0
        fill_b = float(frame_b[mask_keep].mean()) if mask_keep.any() else 0.0
        frame_a = np.where(mask_keep, frame_a, fill_a).astype(np.float32)
        frame_b = np.where(mask_keep, frame_b, fill_b).astype(np.float32)

    if settings.algorithm == "dcc":
        x_c, y_c, du, dv = correlate_pass(
            frame_a, frame_b, settings.window_sizes[0], settings.steps[0], settings)
        if mask_keep is not None:
            window_keep = _sample_mask_at_centers(mask_keep, x_c, y_c)
            du = np.where(window_keep, du, np.nan)
            dv = np.where(window_keep, dv, np.nan)
        if return_originals:
            return x_c, y_c, du, dv, du.copy(), dv.copy()
        return x_c, y_c, du, dv

    if settings.algorithm != "fft_windef":
        raise ValueError(
            f"algorithm must be 'fft_windef' or 'dcc' (got {settings.algorithm!r}); "
            "ensemble correlation is intentionally not supported."
        )

    if len(settings.window_sizes) != len(settings.steps):
        raise ValueError("window_sizes and steps must have equal length")

    accumulated_u_pixel: Optional[np.ndarray] = None
    accumulated_v_pixel: Optional[np.ndarray] = None
    x_c = y_c = u = v = None

    u_pre = v_pre = None  # captured at the end of the final pass

    def _run_pass(ws, st, acc_u, acc_v):
        # Deform B with the running estimate. nan_to_num the warp field so
        # masked NaNs in (u, v) don't propagate to the warped pixels.
        if acc_u is None:
            warped_b = frame_b
        else:
            warped_b = _deform_image(
                frame_b,
                np.nan_to_num(acc_u, nan=0.0),
                np.nan_to_num(acc_v, nan=0.0),
            )
        x_c2, y_c2, du, dv = correlate_pass(frame_a, warped_b, ws, st, settings)

        if acc_u is None:
            u_prev = np.zeros_like(du)
            v_prev = np.zeros_like(dv)
        else:
            u_prev = _sample_dense_to_grid(
                np.nan_to_num(acc_u, nan=0.0), x_c2, y_c2)
            v_prev = _sample_dense_to_grid(
                np.nan_to_num(acc_v, nan=0.0), x_c2, y_c2)

        u2 = u_prev + du
        v2 = v_prev + dv

        if mask_keep is not None:
            window_keep = _sample_mask_at_centers(mask_keep, x_c2, y_c2)
            u2 = np.where(window_keep, u2, np.nan)
            v2 = np.where(window_keep, v2, np.nan)

        # Snapshot the pre-filter values (used for u_original / typevector).
        u_unfiltered = u2.copy()
        v_unfiltered = v2.copy()

        if settings.replace_outliers:
            u2, v2 = _replace_outliers(
                u2, v2,
                settings.median_filter_size, settings.outlier_threshold,
            )

        new_acc_u, new_acc_v = _upsample_to_pixel_grid(
            x_c2, y_c2, u2, v2, frame_a.shape)
        return (x_c2, y_c2, u2, v2, u_unfiltered, v_unfiltered,
                new_acc_u, new_acc_v)

    for ws, st in zip(settings.window_sizes, settings.steps):
        (x_c, y_c, u, v, u_pre, v_pre,
         accumulated_u_pixel, accumulated_v_pixel) = _run_pass(
            ws, st, accumulated_u_pixel, accumulated_v_pixel)

    # Repeat-last-pass loop.
    if settings.repeat_last_pass:
        ws, st = settings.window_sizes[-1], settings.steps[-1]
        prev_mean_mag = float(np.nanmean(np.hypot(u, v)))
        for _ in range(settings.repeat_max_iterations):
            (x_c, y_c, u, v, u_pre, v_pre,
             accumulated_u_pixel, accumulated_v_pixel) = _run_pass(
                ws, st, accumulated_u_pixel, accumulated_v_pixel)
            mean_mag = float(np.nanmean(np.hypot(u, v)))
            slope = abs(mean_mag - prev_mean_mag) / max(mean_mag, 1e-6)
            prev_mean_mag = mean_mag
            if slope < settings.quality_slope_threshold:
                break

    # ── Velocity cap — drop whole vectors above the |v| limit ───────────────
    u, v = _apply_velocity_cap(settings, u, v)
    if u_pre is not None:
        u_pre, v_pre = _apply_velocity_cap(settings, u_pre, v_pre)

    if return_originals:
        return x_c, y_c, u, v, u_pre, v_pre
    return x_c, y_c, u, v


def _apply_velocity_cap(settings: PIVSettings,
                        u: np.ndarray, v: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """NaN whole vectors whose vertical speed |v| exceeds the configured cap.

    Matches the cap-explorer export: a vector is removed (both u and v → NaN)
    when |v| > cap. The cap (px/frame) comes from `velocity_cap_px` (absolute),
    else `velocity_cap_fraction` × smallest window, else the per-array
    `velocity_cap_percentile` of |v|. Returns (u, v) with bad cells NaN'd.
    """
    vmag = np.abs(v)
    cap = float("inf")
    if settings.velocity_cap_px is not None:
        cap = float(settings.velocity_cap_px)
    elif settings.velocity_cap_fraction is not None and settings.window_sizes:
        cap = float(settings.velocity_cap_fraction) * float(min(settings.window_sizes))
    elif settings.velocity_cap_percentile is not None:
        finite = vmag[np.isfinite(vmag)]
        if finite.size:
            cap = float(np.percentile(finite, 100.0 - float(settings.velocity_cap_percentile)))
    if not np.isfinite(cap):
        return u, v
    bad = vmag > cap
    return np.where(bad, np.nan, u), np.where(bad, np.nan, v)


def _sample_dense_to_grid(dense: np.ndarray,
                          x_c: np.ndarray, y_c: np.ndarray) -> np.ndarray:
    """Sample a per-pixel field at the (x_c, y_c) window centres."""
    xx = x_c.astype(np.float32)
    yy = y_c.astype(np.float32)
    return cv2.remap(dense.astype(np.float32), xx, yy,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ─────────────────────────── Top-level entry ──────────────────────────────────

def run_piv(frame_a: np.ndarray, frame_b: np.ndarray,
            piv_settings: Optional[PIVSettings] = None,
            preproc_settings: Optional[PreprocSettings] = None,
            background: Optional[np.ndarray] = None,
            calibration: Optional[CalibrationSettings] = None,
            mask: Optional[np.ndarray] = None,
            return_originals: bool = False
            ):
    """End-to-end: preprocess both frames, then run multi-pass PIV.

    Frames can be uint8 or float; colour images are converted to grayscale.
    If `calibration` is provided, output is in world units: x, y in metres
    and u, v in metres per second. Otherwise output is in pixels and px/frame.

    `mask` may be a bool array (True = analyse) or any image which is
    thresholded with the PIVlab convention (bright > 127 = EXCLUDED, dark
    ≤ 127 = analysed). Vectors whose interrogation window centre falls
    outside the keep region are NaN in the output.

    If `return_originals=True`, returns (x, y, u_filtered, v_filtered,
    u_original, v_original) — the `_original` arrays are the raw vectors
    *before* outlier replacement, matching PIVlab's session-save fields.
    """
    piv_settings = piv_settings or PIVSettings()
    preproc_settings = preproc_settings or PreprocSettings()

    a = _ensure_gray(frame_a)
    b = _ensure_gray(frame_b)
    bg = _ensure_gray(background) if background is not None else None

    a = preprocess(a, preproc_settings, bg)
    b = preprocess(b, preproc_settings, bg)

    mask_keep: Optional[np.ndarray] = None
    if mask is not None:
        if mask.dtype == bool:
            mask_keep = mask
        else:
            m = mask
            if m.ndim == 3:
                m = m[..., 0]
            mask_keep = m <= 127  # PIVlab: bright = exclude, dark = analyse
        if mask_keep.shape != a.shape:
            mask_keep = cv2.resize(
                mask_keep.astype(np.uint8), (a.shape[1], a.shape[0]),
                interpolation=cv2.INTER_NEAREST).astype(bool)

    if return_originals:
        x, y, u, v, u_orig, v_orig = multipass_piv(
            a, b, piv_settings, mask_keep, return_originals=True)
        if calibration is not None:
            x, y, u, v = apply_calibration(x, y, u, v, calibration)
            # u_orig / v_orig share the same x, y grid; only the velocity
            # components need to be scaled.
            vs = calibration.m_per_second_per_px_per_frame
            u_orig = u_orig * vs * calibration.x_sign
            v_orig = v_orig * vs * calibration.y_sign
        return x, y, u, v, u_orig, v_orig

    x, y, u, v = multipass_piv(a, b, piv_settings, mask_keep)

    if calibration is not None:
        x, y, u, v = apply_calibration(x, y, u, v, calibration)

    return x, y, u, v


def apply_calibration(x: np.ndarray, y: np.ndarray,
                      u: np.ndarray, v: np.ndarray,
                      calibration: CalibrationSettings
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert pixel-space PIV output to physical units.

    Returns (x_m, y_m, u_ms, v_ms): positions in metres, velocities in m/s.
    Axis-direction dropdowns flip signs; offsets shift the world origin.
    """
    s = calibration.m_per_pixel
    vs = calibration.m_per_second_per_px_per_frame
    x_m = x * s * calibration.x_sign + calibration.x_offset_m
    y_m = y * s * calibration.y_sign + calibration.y_offset_m
    u_ms = u * vs * calibration.x_sign
    v_ms = v * vs * calibration.y_sign
    return x_m, y_m, u_ms, v_ms


def _ensure_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in (3, 4):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"unsupported image shape: {image.shape}")


# ────────────────────────────── CLI ───────────────────────────────────────────

def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def _cli() -> None:
    p = argparse.ArgumentParser(description="Simplified Python PIV processor.")
    p.add_argument("frame_a")
    p.add_argument("frame_b")
    p.add_argument("--background", default=None,
                   help="optional background image for mean-intensity subtraction")
    p.add_argument("--mask", default=None,
                   help="optional binary mask (PIVlab-style); bright (>127) = "
                        "EXCLUDED, dark = analysed (NaN over the excluded "
                        "region in the output)")
    p.add_argument("--out", default="piv_vectors.npz",
                   help="output .npz with arrays x, y, u, v")

    # PIV settings flags — mirroring the GUI panel.
    p.add_argument("--algorithm", default="fft_windef",
                   choices=("fft_windef", "dcc"))
    p.add_argument("--windows", default="64,32,16",
                   help="comma-separated interrogation areas per pass")
    p.add_argument("--steps", default="32,16,8",
                   help="comma-separated steps per pass")
    p.add_argument("--repeat-last-pass", action="store_true")
    p.add_argument("--quality-slope", type=float, default=0.025)
    p.add_argument("--subpixel", default="gauss2x3",
                   choices=("gauss2x3", "centroid", "parabolic"))
    p.add_argument("--disable-autocorrelation", action="store_true")
    p.add_argument("--robustness", default="standard",
                   choices=("standard", "extreme"))

    # Pre-processing flags.
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--clahe-window", type=int, default=64)
    p.add_argument("--highpass", action="store_true")
    p.add_argument("--highpass-kernel", type=int, default=15)
    p.add_argument("--intensity-capping", action="store_true")
    p.add_argument("--wiener", action="store_true")
    p.add_argument("--wiener-window", type=int, default=3)
    p.add_argument("--no-contrast-stretch", action="store_true")
    p.add_argument("--subtract-mean", action="store_true")

    # Calibration flags — defaults match the GUI panel.
    p.add_argument("--calibrate", action="store_true",
                   help="output positions in metres and velocities in m/s")
    p.add_argument("--reference-length-px", type=float, default=440.0)
    p.add_argument("--real-distance-mm", type=float, default=70.0)
    p.add_argument("--time-step-ms", type=float, default=2.0)
    p.add_argument("--x-towards", default="right", choices=("right", "left"))
    p.add_argument("--y-towards", default="bottom", choices=("bottom", "top"))
    p.add_argument("--x-offset-m", type=float, default=0.0)
    p.add_argument("--y-offset-m", type=float, default=0.0)

    args = p.parse_args()

    windows = tuple(int(x) for x in args.windows.split(","))
    steps = tuple(int(x) for x in args.steps.split(","))

    piv_settings = PIVSettings(
        algorithm=args.algorithm,
        window_sizes=windows,
        steps=steps,
        repeat_last_pass=args.repeat_last_pass,
        quality_slope_threshold=args.quality_slope,
        subpixel_method=args.subpixel,
        disable_autocorrelation=args.disable_autocorrelation,
        correlation_robustness=args.robustness,
    )
    preproc_settings = PreprocSettings(
        enable_clahe=not args.no_clahe,
        clahe_window_size=args.clahe_window,
        enable_highpass=args.highpass,
        highpass_kernel_size=args.highpass_kernel,
        enable_intensity_capping=args.intensity_capping,
        enable_wiener2=args.wiener,
        wiener_window_size=args.wiener_window,
        enable_contrast_stretch=not args.no_contrast_stretch,
        subtract_mean_intensity=args.subtract_mean,
    )

    calibration: Optional[CalibrationSettings] = None
    if args.calibrate:
        calibration = CalibrationSettings(
            reference_length_px=args.reference_length_px,
            real_distance_mm=args.real_distance_mm,
            time_step_ms=args.time_step_ms,
            x_increases_towards=args.x_towards,
            y_increases_towards=args.y_towards,
            x_offset_m=args.x_offset_m,
            y_offset_m=args.y_offset_m,
        )

    a = _load_image(args.frame_a)
    b = _load_image(args.frame_b)
    bg = _load_image(args.background) if args.background else None
    mask = load_mask(args.mask) if args.mask else None

    x, y, u, v = run_piv(a, b, piv_settings, preproc_settings, bg,
                         calibration, mask)

    out_path = Path(args.out)
    np.savez(out_path, x=x, y=y, u=u, v=v)
    n_valid = int(np.isfinite(u).sum())
    n_total = u.size
    if calibration is None:
        mean_mag = float(np.nanmean(np.hypot(u, v)))
        print(f"wrote {out_path}  shape={u.shape}  "
              f"valid={n_valid}/{n_total}  "
              f"mean |V|={mean_mag:.3f} px/frame")
    else:
        mean_mag = float(np.nanmean(np.hypot(u, v)))
        print(f"wrote {out_path}  shape={u.shape}  "
              f"valid={n_valid}/{n_total}  "
              f"mean |V|={mean_mag:.4f} m/s  "
              f"(1 px = {calibration.m_per_pixel:.4e} m, "
              f"1 px/frame = {calibration.m_per_second_per_px_per_frame:.4f} m/s)")


if __name__ == "__main__":
    _cli()
