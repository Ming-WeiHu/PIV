"""piv_generate_videos.py — Python port of PIV_Generate_Videos.m

Generates an MP4 per condition: shear-rate pcolor with optional streamlines
overlaid. In MATLAB this was a separate script run after the main pipeline;
here it can be invoked standalone OR via piv_pipeline_master --make-videos.

Standalone usage
----------------
    python -m piv_pipeline.piv_generate_videos --tertiary-dir "...Tertiary Exports" \\
        --vid-fps 20 --step 3 --clim-max 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cv2
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm


def _figure_to_bgr(fig, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Render a matplotlib figure to a uint8 BGR ndarray for cv2.VideoWriter.

    If `target_size` (w, h) is given and the figure renders at a different
    size, the result is resized — keeps cv2's fixed-frame-size requirement
    happy across frames.
    """
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if target_size is not None and (w, h) != target_size:
        bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
    return bgr


def _render_frame(
    X_norm: np.ndarray, Y_norm: np.ndarray,
    shear: np.ndarray, uc: np.ndarray, vc: np.ndarray,
    *, t_ms: float, clim_max: float,
    figsize: tuple[float, float] = (8.0, 6.0),
    dpi: int = 100,
):
    """Build the matplotlib figure for one frame. Returns the figure."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Interpolate NaN holes in shear before plotting (masked/capped cells).
    # Only fill *interior* holes — cells enclosed by valid data. The mask
    # interior is `binary_fill_holes(valid)`; everything outside it (the region
    # beyond the mask, and concave notches the convex-hull interpolation would
    # otherwise bleed into) is forced back to NaN so it never extends past the
    # mask boundary.
    shear_plot = shear.copy()
    valid = np.isfinite(shear_plot)
    fill_region = binary_fill_holes(valid)
    nan_mask = ~valid
    if nan_mask.any() and valid.sum() > 10:
        try:
            pts = np.column_stack([X_norm[valid], Y_norm[valid]])
            shear_plot[nan_mask] = griddata(
                pts, shear[valid],
                (X_norm[nan_mask], Y_norm[nan_mask]),
                method="linear",
            )
            # Any remaining NaN inside the mask (outside convex hull) → nearest
            still_nan = ~np.isfinite(shear_plot)
            if still_nan.any():
                shear_plot[still_nan] = griddata(
                    pts, shear[valid],
                    (X_norm[still_nan], Y_norm[still_nan]),
                    method="nearest",
                )
        except Exception:
            pass
    # Clamp back to the mask: drop anything the interpolation pushed past it.
    shear_plot[~fill_region] = np.nan

    # pcolor (shading='gouraud' ≈ MATLAB 'FaceColor', 'interp')
    pcm = ax.pcolormesh(
        X_norm, Y_norm, shear_plot,
        shading="gouraud", cmap="jet",
        vmin=0, vmax=clim_max,
    )
    cb = fig.colorbar(pcm, ax=ax)
    cb.ax.set_title(r"$\dot{\gamma}$ [1/s]", fontsize=12)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"t = {t_ms:.1f} ms")

    # Streamlines from masked velocity — interpolate over masked holes first.
    # Optional: skip silently if it fails for this frame (mirrors MATLAB try/catch).
    vel_valid = np.isfinite(uc) & np.isfinite(vc)
    if vel_valid.sum() > 10:
        try:
            vel_region = binary_fill_holes(vel_valid)
            pts = np.column_stack([X_norm[vel_valid], Y_norm[vel_valid]])
            uc_i = griddata(pts, uc[vel_valid], (X_norm, Y_norm), method="linear")
            vc_i = griddata(pts, vc[vel_valid], (X_norm, Y_norm), method="linear")
            # Keep streamlines inside the mask — NaN elsewhere so streamplot
            # draws nothing past the boundary.
            uc_i[~vel_region] = np.nan
            vc_i[~vel_region] = np.nan

            # streamplot requires strictly increasing 1D x / y.
            if Y_norm[0, 0] > Y_norm[-1, 0]:
                X_s = X_norm[::-1, :]; Y_s = Y_norm[::-1, :]
                u_s = uc_i[::-1, :];   v_s = vc_i[::-1, :]
            else:
                X_s, Y_s, u_s, v_s = X_norm, Y_norm, uc_i, vc_i

            ax.streamplot(
                X_s[0, :], Y_s[:, 0], u_s, v_s,
                density=0.5, color="w", linewidth=0.8,
            )
        except Exception:
            pass

    return fig


def generate_videos(
    tert_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    vid_fps: float = 30,
    step: int = 3,
    clim_max: float = 30,
) -> Path:
    """Render an MP4 per *_Tertiary_Export.npz in `tert_dir`.

    Parameters
    ----------
    tert_dir : path
        Folder containing tertiary exports.
    out_dir : path, optional
        Output folder for MP4s. Default: <tert_dir>/Videos.
    vid_fps : float
        Output video frame rate.
    step : int
        Sample every Nth source frame.
    clim_max : float
        Shear-rate colour bar upper limit (1/s).

    Returns
    -------
    out_dir : Path
    """
    tert_dir = Path(tert_dir)
    out_dir = Path(out_dir) if out_dir else tert_dir / "Videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(tert_dir.glob("*_Tertiary_Export.npz"))
    if not files:
        raise FileNotFoundError(f"No tertiary exports in: {tert_dir}")
    print(f"Found {len(files)} tertiary export(s).")

    for fi, fp in enumerate(files, start=1):
        with np.load(fp, allow_pickle=False) as d:
            masked_shear = d["masked_shear"]
            masked_u = d["masked_u"]
            masked_v = d["masked_v"]
            X_norm = d["X_norm"]
            Y_norm = d["Y_norm"]
            t = np.asarray(d["t"])
            condition = d["condition"].item()

        n_frames = int(masked_shear.shape[2])
        vid_path = out_dir / f"shear_{condition}.mp4"
        frame_idx = list(range(0, n_frames, step))

        print(
            f"\n[{fi}/{len(files)}] {condition}  "
            f"({len(frame_idx)} of {n_frames} frames, step={step})"
        )

        writer: cv2.VideoWriter | None = None
        target_size: tuple[int, int] | None = None
        try:
            for k in tqdm(frame_idx, desc=f"  {condition}", unit="frame", leave=False):
                fig = _render_frame(
                    X_norm, Y_norm,
                    masked_shear[:, :, k],
                    masked_u[:, :, k],
                    masked_v[:, :, k],
                    t_ms=float(t[k]) * 1000.0,
                    clim_max=clim_max,
                )
                bgr = _figure_to_bgr(fig, target_size=target_size)
                plt.close(fig)

                if writer is None:
                    h, w = bgr.shape[:2]
                    target_size = (w, h)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(vid_path), fourcc, vid_fps, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"cv2.VideoWriter failed to open {vid_path}. "
                            f"Check that opencv-python (not -headless) is installed."
                        )
                writer.write(bgr)
        finally:
            if writer is not None:
                writer.release()

        print(f"  Saved: {vid_path}")

    print(f"\nAll videos saved to: {out_dir}")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate shear-rate MP4 videos from tertiary exports.",
    )
    parser.add_argument(
        "--tertiary-dir", required=True,
        help="Folder containing *_Tertiary_Export.npz files.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output folder (default: <tertiary-dir>/Videos).",
    )
    parser.add_argument("--vid-fps", type=float, default=30,
                        help="Output video frame rate (default 30).")
    parser.add_argument("--step", type=int, default=3,
                        help="Sample every Nth source frame (default 3).")
    parser.add_argument("--clim-max", type=float, default=30,
                        help="Shear-rate colour bar upper limit, 1/s (default 30).")
    args = parser.parse_args(argv)

    generate_videos(
        args.tertiary_dir,
        out_dir=args.output_dir,
        vid_fps=args.vid_fps,
        step=args.step,
        clim_max=args.clim_max,
    )


if __name__ == "__main__":
    main()
