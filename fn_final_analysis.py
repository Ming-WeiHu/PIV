"""fn_final_analysis.py — Python port of fn_final_analysis.m

For each tertiary export computes time-averaged shear metrics and emits:
  - Per-condition smoothed signal plots          ({condition}_signals.png)
  - Per-condition peak-detection plots           ({condition}_peak_detection.png)
  - Summary plots — TimeAvg_mean, Peak_top5pct, Peak_top10pct vs RPM
  - results_rock.csv with all metrics

Metric definitions
------------------
    TimeAvg_mean   : time-average of per-frame spatial MEAN of masked_shear
    TimeAvg_median : time-average of per-frame spatial MEDIAN of masked_shear
    Peak_top5pct   : mean of top  5% temporal points of the p90-spatial signal
    Peak_top10pct  : mean of top 10% temporal points of the p90-spatial signal
    (where "p90-spatial" = per-frame mean of the top 10% of pixel values —
     MATLAB-original naming; not the 90th percentile.)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-GUI backend — safe to use without a display
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm


# Pastel 8-cycle from PIV_Pipeline_Master.m
DEFAULT_PALETTE: list[tuple[float, float, float]] = [
    (0.85, 0.60, 0.60),  # dusty rose
    (0.55, 0.75, 0.90),  # sky blue
    (0.65, 0.88, 0.65),  # sage green
    (0.95, 0.82, 0.58),  # warm peach
    (0.80, 0.68, 0.90),  # lavender
    (0.58, 0.88, 0.88),  # soft cyan
    (0.95, 0.90, 0.58),  # soft yellow
    (0.90, 0.72, 0.58),  # terracotta
]


def _matlab_round(x: float) -> int:
    """MATLAB-style round: round half AWAY from zero (not banker's)."""
    return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))


def _smooth_moving(x: np.ndarray, span: int) -> np.ndarray:
    """Equivalent of MATLAB smooth(x, span, 'moving') — centred moving avg.

    pandas rolling(min_periods=1) handles NaN by ignoring (slightly different
    from MATLAB which propagates NaN, but better-behaved at edges).
    """
    return (
        pd.Series(x)
        .rolling(window=span, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def _parse_rpm(condition: str) -> float:
    """'100mL_15-0-0_20deg' → 15.0; NaN if pattern not present."""
    m = re.search(r"_(\d+)-0-0_", condition)
    return float(m.group(1)) if m else float("nan")


def final_analysis(
    tert_dir: str | Path,
    out_dir: str | Path,
    *,
    span: int = 50,
    palette: list[tuple[float, float, float]] | None = None,
) -> Path:
    """Run final analysis over every *_Tertiary_Export.npz in `tert_dir`.

    Parameters
    ----------
    tert_dir : path
        Folder containing *_Tertiary_Export.npz files.
    out_dir : path
        Folder for plots and the CSV (created if missing).
    span : int
        Smoothing window for moving average. Default 50.
    palette : list of (r, g, b) tuples in [0,1]
        Per-condition colour cycle. Default = MATLAB master pastel 8-cycle.

    Returns
    -------
    csv_path : Path  (results_rock.csv)
    """
    tert_dir = Path(tert_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = palette if palette is not None else DEFAULT_PALETTE

    files = sorted(tert_dir.glob("*_Tertiary_Export.npz"))
    n_files = len(files)
    if n_files == 0:
        raise FileNotFoundError(f"No tertiary exports in: {tert_dir}")

    MTimeAvg_mean = np.full(n_files, np.nan)
    MTimeAvg_median = np.full(n_files, np.nan)
    MPeak_95 = np.full(n_files, np.nan)
    MPeak_90 = np.full(n_files, np.nan)
    Mrpm = np.full(n_files, np.nan)
    Mcondition: list[str] = [""] * n_files

    for i, fp in enumerate(files):
        with np.load(fp, allow_pickle=False) as d:
            masked_shear = d["masked_shear"]
            t = np.asarray(d["t"])
            condition = d["condition"].item()

        Mcondition[i] = condition
        Mrpm[i] = _parse_rpm(condition)

        print(f"  [Stage 4] {condition}  (rpm = {Mrpm[i]:g})")

        n_frames = masked_shear.shape[2]
        meanframe = np.full(n_frames, np.nan)
        medianframe = np.full(n_frames, np.nan)
        mean_p90 = np.full(n_frames, np.nan)

        for k in tqdm(range(n_frames),
                      desc=f"  Stage 4 — {condition}",
                      unit="frame", leave=False):
            vals = masked_shear[:, :, k].ravel()
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            meanframe[k] = vals.mean()
            medianframe[k] = np.median(vals)
            n_top = max(1, _matlab_round(vals.size * 0.1))
            vals_desc = np.sort(vals)[::-1]
            mean_p90[k] = vals_desc[:n_top].mean()

        smooth_mean = _smooth_moving(meanframe, span)
        smooth_median = _smooth_moving(medianframe, span)
        smooth_p90 = _smooth_moving(mean_p90, span)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            MTimeAvg_mean[i] = np.nanmean(smooth_mean)
            MTimeAvg_median[i] = np.nanmean(smooth_median)

        # Peak detection on smooth_p90: top 5% and 10% time points
        valid_mask = ~np.isnan(smooth_p90)
        valid_p90 = smooth_p90[valid_mask]
        valid_idx = np.flatnonzero(valid_mask)
        if valid_p90.size > 0:
            sort_i = np.argsort(valid_p90)[::-1]
            sorted_p90 = valid_p90[sort_i]

            n_top_95 = max(1, _matlab_round(sorted_p90.size * 0.05))
            n_top_90 = max(1, _matlab_round(sorted_p90.size * 0.10))

            MPeak_95[i] = sorted_p90[:n_top_95].mean()
            MPeak_90[i] = sorted_p90[:n_top_90].mean()

            peak_idx_95 = valid_idx[sort_i[:n_top_95]]
            peak_idx_90 = valid_idx[sort_i[:n_top_90]]
        else:
            peak_idx_95 = peak_idx_90 = np.array([], dtype=int)

        ci = i % len(palette)
        col = palette[ci]
        col_dark = tuple(c * 0.75 for c in col)

        # ---- Per-condition: smoothed signals ------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, smooth_mean, color=col, linewidth=1.2,
                label="mean (all pixels)")
        ax.plot(t, smooth_median, color=col_dark, linewidth=1.2,
                linestyle="--", label="median (all pixels)")
        ax.plot(t, smooth_p90, color=(0.35, 0.35, 0.35), linewidth=1.2,
                label="mean of top 10%")
        ax.set_xlabel("t [s]"); ax.set_ylabel("Shear rate [1/s]")
        ax.set_title(condition)
        ax.legend(loc="upper left"); ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{condition}_signals.png", dpi=150)
        plt.close(fig)

        # ---- Per-condition: peak detection --------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(t, smooth_p90, color=(0.3, 0.3, 0.3), linewidth=1.2,
                label="mean p90 spatial")
        if peak_idx_95.size > 0:
            ax.plot(t[peak_idx_95], smooth_p90[peak_idx_95], ".",
                    color=palette[0], markersize=6,
                    label=f"top 5% = {MPeak_95[i]:.4f}")
        if peak_idx_90.size > 0:
            ax.plot(t[peak_idx_90], smooth_p90[peak_idx_90], ".",
                    color=palette[1], markersize=4,
                    label=f"top 10% = {MPeak_90[i]:.4f}")
        if np.isfinite(MTimeAvg_mean[i]):
            ax.axhline(MTimeAvg_mean[i], color=palette[2], linestyle="--",
                       linewidth=1.5,
                       label=f"time-avg mean = {MTimeAvg_mean[i]:.4f}")
        if np.isfinite(MTimeAvg_median[i]):
            ax.axhline(MTimeAvg_median[i], color=palette[3], linestyle="--",
                       linewidth=1.5,
                       label=f"time-avg median = {MTimeAvg_median[i]:.4f}")
        ax.set_xlabel("t [s]"); ax.set_ylabel("Shear rate [1/s]")
        ax.set_title(condition)
        ax.legend(loc="upper left"); ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / f"{condition}_peak_detection.png", dpi=150)
        plt.close(fig)

    # ======================================================================
    # Summary plots: metric vs RPM
    # ======================================================================
    finite_idx = np.flatnonzero(np.isfinite(Mrpm))
    finite_sorted = finite_idx[np.argsort(Mrpm[finite_idx])] if finite_idx.size else finite_idx

    def _summary_plot(metric: np.ndarray, title: str, fname: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        # If no condition parses to an rpm, plot by condition index instead —
        # a "vs RPM" axis is meaningless when comparing methods at one rpm.
        use_rpm = bool(np.isfinite(Mrpm).any())
        for j in range(n_files):
            ci = j % len(palette)
            col = palette[ci]
            edge = tuple(c * 0.65 for c in col)
            x = Mrpm[j] if use_rpm else j
            ax.plot(x, metric[j], "o",
                    markerfacecolor=col, markeredgecolor=edge,
                    markersize=9, label=Mcondition[j])
        if use_rpm:
            if finite_sorted.size > 1:
                ax.plot(Mrpm[finite_sorted], metric[finite_sorted], "-",
                        color=(0.55, 0.55, 0.55), linewidth=1.2)
            ax.set_xlabel("RPM")
        else:
            ax.set_xticks(range(n_files))
            ax.set_xticklabels(Mcondition, rotation=20, ha="right")
            ax.set_xlabel("Condition")
            ax.margins(x=0.2)
        ax.set_ylabel("Shear rate [1/s]")
        ax.set_title(title)
        ax.legend(loc="upper left"); ax.grid(True)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)

    _summary_plot(MTimeAvg_mean,
                  "Time-average shear (spatial mean) vs RPM",
                  "Summary_TimeAvg_mean.png")
    _summary_plot(MPeak_90,
                  "Peak shear (p90 spatial, top 10% temporal) vs RPM",
                  "Summary_Peak_top10pct.png")
    _summary_plot(MPeak_95,
                  "Peak shear (p90 spatial, top 5% temporal) vs RPM",
                  "Summary_Peak_top5pct.png")

    # ---- CSV --------------------------------------------------------------
    df = pd.DataFrame({
        "condition": Mcondition,
        "rpm": Mrpm,
        "TimeAvg_mean": MTimeAvg_mean,
        "TimeAvg_median": MTimeAvg_median,
        "Peak_top5pct": MPeak_95,
        "Peak_top10pct": MPeak_90,
    })
    csv_path = out_dir / "results_rock.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Stage 4] results_rock.csv and summary plots saved to:\n  {out_dir}")
    return csv_path
