"""fn_secondary_export.py — Python port of fn_secondary_export.m

Computes per-frame AA-frame moving averages of u, v, then velocity gradients,
rate-of-strain tensor, S33 (max shear), vorticity, and velocity magnitude.

UNITS
-----
    u, v        : METRES / SECOND       (converted from px/frame using calu/calv when convert_velocity=True)
                   PIXELS / FRAME       (when convert_velocity=False)
    x, y, X, Y  : METRES                (post-calibration)
    dt          : SECONDS               (1/fps)
    delta_x/y   : METRES                (grid spacing)
    velmag      : METRES / SECOND       (sqrt(u^2 + v^2))
    S33, vort   : 1 / SECOND            (gradient of m/s over metres when convert_velocity=True)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import uniform_filter
from tqdm import tqdm


def _fill_nans(field: np.ndarray, region: np.ndarray,
               method: str = "interpolate") -> np.ndarray:
    """Fill NaN cells of `field` before gradients are taken.

    method='zero'        — NaN → 0 everywhere (the old MATLAB-port behaviour;
                           creates artificial 0-edges that can inflate shear).
    method='interpolate' — NaN cells INSIDE `region` (the analysis mask) are
                           filled by iterative normalised-box diffusion from
                           their finite neighbours (PIVlab-style replacement);
                           NaN cells OUTSIDE the region are set to 0 (they are
                           masked out downstream anyway). This avoids the
                           artificial gradient at masked/capped holes.
    """
    out = field.astype(np.float64).copy()
    nan = np.isnan(out)
    if not nan.any():
        return out
    if method == "zero":
        out[nan] = 0.0
        return out

    interior = nan & region          # holes to interpolate
    work = np.where(nan, 0.0, out)
    weight = (~nan).astype(np.float64)   # 1 where a real value is known
    targets = interior.copy()
    for _ in range(100):
        if not targets.any():
            break
        num = uniform_filter(work * weight, size=3, mode="nearest")
        den = uniform_filter(weight, size=3, mode="nearest")
        newly = targets & (den > 1e-9)
        if not newly.any():
            break
        est = np.zeros_like(num)
        np.divide(num, den, out=est, where=den > 1e-9)
        work[newly] = est[newly]
        weight[newly] = 1.0
        targets &= ~newly
    work[np.isnan(work)] = 0.0       # any leftover (incl. exterior) → 0
    return work


def secondary_export(
    primary: dict[str, Any],
    *,
    centre_x: float,
    centre_y: float,
    fps: float,
    AA: int,
    howstupid: float,
    convert_velocity: bool,
    out_dir: str | Path,
    name_info: dict[str, Any],
    nan_fill: str = "interpolate",
) -> tuple[int, int]:
    """Run the secondary export for one primary file.

    Parameters
    ----------
    primary : dict
        Output of `io_loaders.load_primary()`.
    centre_x, centre_y : float
        Vessel centre in PIXELS (from the calibration image).
    fps : float
        Camera frames per second.
    AA : int
        Averaging window in frames.
    howstupid : float
        Scaling correction. 1 normally; 10 for the legacy PIVlab bug.
    convert_velocity : bool
        If True, convert px/frame velocities to m/s before computing gradients.
        If False, use raw px/frame values.
    out_dir : str | Path
        Folder for the output .npz file.
    name_info : dict
        Output of `io_loaders.parse_filename()`.

    Returns
    -------
    (mask_rows, mask_cols) : (int, int)
        Grid size — used by Stage 2 (dynamic masking).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_grid = primary["x"]
    y_grid = primary["y"]
    u_list = primary["u_list"]
    v_list = primary["v_list"]
    typevector = primary["typevector"]
    calxy = primary["calxy"]
    calu = primary.get("calu", 1.0)
    calv = primary.get("calv", 1.0)
    has_calu = primary.get("has_calu", False)
    has_calv = primary.get("has_calv", False)
    n_total = primary["n_pairs"]

    mask_rows, mask_cols = x_grid.shape

    # ---- Coordinate setup (MATLAB lines 25–35) -----------------------------
    xcoord = x_grid * howstupid
    ycoord = y_grid * howstupid

    xmin = xcoord[0, 0]
    X = xcoord - xmin + 10.0 * calxy
    X = X - centre_x * calxy
    Y = -(ycoord - centre_y * calxy)

    delta_x = X[0, 1] - X[0, 0]   # metres per column step
    delta_y = Y[1, 0] - Y[0, 0]   # metres per row step (typically NEGATIVE — Y flipped)

    dt_ms = 1000.0 / fps
    f_loadstep = 1
    img_start = 0
    img_stop = 0

    n_frames = n_total - AA + 1
    if n_frames <= 0:
        raise ValueError(
            f"AA={AA} is larger than n_pairs={n_total}. "
            f"Either reduce AA or load more frames."
        )

    # ---- Preallocate (MATLAB lines 41–49) ----------------------------------
    S33 = np.zeros((mask_rows, mask_cols, n_frames), dtype=np.float32)
    vorticity = np.zeros_like(S33)
    velmag = np.zeros_like(S33)
    avgu = np.zeros_like(S33)
    avgv = np.zeros_like(S33)
    t = np.zeros(n_frames, dtype=np.float64)
    abst = np.zeros(n_frames, dtype=np.float64)

    # ---- Main loop (MATLAB lines 52–92) ------------------------------------
    iterator = tqdm(range(n_frames),
                    desc=f"  Stage 1 — {name_info['condition']}",
                    unit="frame",
                    leave=False)

    print(
        f"  [Stage 1] convert_velocity={convert_velocity} "
        f"has_calu={has_calu} has_calv={has_calv} calu={calu:.6g} calv={calv:.6g} "
        f"calxy={calxy:.6g} fps={fps:.6g}"
    )

    if convert_velocity:
        if has_calu and has_calv:
            u_factor = howstupid * calu
            v_factor = howstupid * calv
            print(
                f"  [Stage 1] Converting velocities with calu={calu:.6g}, calv={calv:.6g}"
            )
        else:
            fallback = calxy * fps
            u_factor = howstupid * fallback
            v_factor = howstupid * fallback
            print(
                f"  [Stage 1] Converting velocities using fallback calxy*fps={fallback:.6g}"
            )
    else:
        u_factor = howstupid
        v_factor = howstupid
        print("  [Stage 1] Velocity conversion disabled; using raw px/frame values.")

    print(f"  [Stage 1] u_factor={u_factor:.6g} v_factor={v_factor:.6g}")
    print(f"  [Stage 1] nan_fill={nan_fill}")

    # Analysis region for NaN interpolation (True = analyse). Outside this the
    # field is masked downstream, so holes there are just zeroed.
    region = np.asarray(typevector).astype(bool)

    for g in iterator:
        # AA-frame cube (MATLAB lines 53–60)
        u_cube = np.stack(u_list[g:g + AA], axis=-1) * u_factor
        v_cube = -np.stack(v_list[g:g + AA], axis=-1) * v_factor   # NB: sign flip on v

        t[g] = (img_start + (g + 1) * f_loadstep - img_stop) * dt_ms / 1000.0
        abst[g] = (g + AA) * dt_ms * f_loadstep / 1000.0

        # nanmean — silence all-NaN-slice RuntimeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            u_avg = np.nanmean(u_cube, axis=2)
            v_avg = np.nanmean(v_cube, axis=2)

        # Fill remaining NaN (fully-masked / capped cells) before gradients.
        #   'interpolate' — PIVlab-style: interpolate holes inside the mask
        #                   from neighbours (avoids artificial 0-edge gradients)
        #   'zero'        — legacy MATLAB-port behaviour (NaN → 0)
        u_avg = _fill_nans(u_avg, region, nan_fill)
        v_avg = _fill_nans(v_avg, region, nan_fill)

        avgu[..., g] = u_avg
        avgv[..., g] = v_avg
        velmag[..., g] = np.hypot(u_avg, v_avg)

        # Velocity gradients
        #   MATLAB:  [dudx, dudy] = gradient(u_avg, delta_x, delta_y)
        #   numpy returns [drow, dcol] when given (dy, dx)
        dudy, dudx = np.gradient(u_avg, delta_y, delta_x)
        dvdy, dvdx = np.gradient(v_avg, delta_y, delta_x)

        # Rate-of-strain tensor (symmetric part of velocity gradient)
        Sxx = dudx
        Syy = dvdy
        Sxy = 0.5 * (dudy + dvdx)

        traceS = Sxx + Syy
        radS = np.sqrt((0.5 * (Sxx - Syy)) ** 2 + Sxy ** 2)
        lam1 = 0.5 * traceS + radS
        lam2 = 0.5 * traceS - radS

        S33[..., g] = 0.5 * (lam1 - lam2)    # max shear rate
        vorticity[..., g] = dvdx - dudy

    iterator.close()

    # ---- Save (MATLAB lines 95–98) -----------------------------------------
    out_path = out_dir / f"{name_info['condition']}_Secondary_Export.npz"
    np.savez_compressed(
        out_path,
        vorticity=vorticity,
        S33=S33,
        calxy=np.float64(calxy),
        AA=np.int64(AA),
        centreY=np.float64(centre_y),
        centreX=np.float64(centre_x),
        typevector=typevector,
        volume=np.array(name_info["volume"]),
        cpm=np.int64(name_info["cpm"]),
        velmag=velmag,
        t=t,
        abst=abst,
        X=X.astype(np.float64),
        Y=Y.astype(np.float64),
        avgu=avgu,
        avgv=avgv,
        condition=np.array(name_info["condition"]),
        convert_velocity=np.bool_(convert_velocity),
        calu=np.float64(calu),
        calv=np.float64(calv),
        has_calu=np.bool_(has_calu),
        has_calv=np.bool_(has_calv),
        nan_fill=np.array(nan_fill),
    )
    print(f"  [Stage 1] Saved: {out_path}")
    return mask_rows, mask_cols
