"""fn_tertiary_export.py — Python port of fn_tertiary_export.m

Applies the dynamic mask to the secondary-export fields. Smooths the
gradient-derived fields (S33, vorticity, velmag) with a 5x5 box filter,
masks out non-fluid pixels (NaN), and normalises X, Y by the vessel radius.

If `mask_file` is None, only the static typevector (from PIVlab / piv_simple's
upstream mask) is used. This is the right mode when the input PIV already had
a mask applied — no separate per-frame dynamic masking needed.

No spatial statistics here — Stage 4 (fn_final_analysis) handles those.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tqdm import tqdm
from scipy.ndimage import uniform_filter


def _load_dynamic_mask(mask_path: Path) -> np.ndarray:
    """Load a dynamic mask from .npz or .mat (v7.3 HDF5 or legacy) file.

    Returns a bool array shaped (rows, cols, frames).
    """
    suffix = mask_path.suffix.lower()

    if suffix == ".npz":
        with np.load(mask_path, allow_pickle=False) as f:
            return f["dynamic_masking"].astype(bool)

    if suffix == ".mat":
        # Try v7.3 HDF5 format first (MATLAB -v7.3 save)
        try:
            import h5py
            with h5py.File(mask_path, "r") as f:
                if "dynamic_masking" not in f:
                    raise KeyError(
                        f"Variable 'dynamic_masking' not found in {mask_path.name}. "
                        f"Keys present: {list(f.keys())}"
                    )
                # h5py reads MATLAB arrays in C order: (frames, cols, rows)
                # Transpose back to (rows, cols, frames) to match secondary arrays
                raw = np.array(f["dynamic_masking"])
                return np.transpose(raw, (2, 1, 0)).astype(bool)
        except (OSError, ImportError):
            pass

        # Fall back to legacy .mat (v4 / v5 / v6) via scipy
        from scipy.io import loadmat
        mat = loadmat(str(mask_path))
        if "dynamic_masking" not in mat:
            raise KeyError(
                f"Variable 'dynamic_masking' not found in {mask_path.name}. "
                f"Keys present: {[k for k in mat if not k.startswith('_')]}"
            )
        return mat["dynamic_masking"].astype(bool)

    raise ValueError(
        f"Unsupported mask file type '{suffix}'. Expected .npz or .mat."
    )


def tertiary_export(
    secondary_file: str | Path,
    mask_file: str | Path | None,
    *,
    R: float,
    out_dir: str | Path,
) -> Path:
    """Run the tertiary export for one secondary export.

    Parameters
    ----------
    secondary_file : path
        {condition}_Secondary_Export.npz produced by Stage 1.
    mask_file : path or None
        dynamic_masking_{condition}.npz produced by Stage 2. Pass None to
        use only the static typevector (e.g. when the upstream PIV already
        applied a mask).
    R : float
        Vessel radius (metres). Used to normalise X, Y.
    out_dir : path
        Folder for the output .npz.

    Returns
    -------
    out_path : Path
    """
    secondary_file = Path(secondary_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not secondary_file.exists():
        raise FileNotFoundError(f"Secondary export not found: {secondary_file}")

    mask_path = Path(mask_file) if mask_file is not None else None
    if mask_path is not None and not mask_path.exists():
        raise FileNotFoundError(f"Dynamic mask file not found: {mask_path}")

    # Load the secondary export first
    with np.load(secondary_file, allow_pickle=False) as sec:
        S33 = sec["S33"]
        vorticity = sec["vorticity"]
        velmag = sec["velmag"]
        avgu = sec["avgu"]
        avgv = sec["avgv"]
        typevector = sec["typevector"].astype(bool)
        X = sec["X"].astype(np.float64)
        Y = sec["Y"].astype(np.float64)
        AA = int(sec["AA"])
        t = sec["t"]
        condition = sec["condition"].item()

        # Materialise arrays before exiting the `with` block
        S33 = np.asarray(S33)
        vorticity = np.asarray(vorticity)
        velmag = np.asarray(velmag)
        avgu = np.asarray(avgu)
        avgv = np.asarray(avgv)
        t = np.asarray(t)

    # Optional dynamic mask (.npz or .mat supported)
    if mask_path is not None:
        dynamic_mask = _load_dynamic_mask(mask_path)
        n_frames = int(min(S33.shape[2], dynamic_mask.shape[2]))
        print(f"  [Stage 3] Applying typevector + dynamic mask to {n_frames} frames...")
    else:
        dynamic_mask = None
        n_frames = int(S33.shape[2])
        print(f"  [Stage 3] Applying typevector-only mask to {n_frames} frames "
              f"(upstream PIV mask).")

    X_norm = X / R
    Y_norm = Y / R
    rows, cols = S33.shape[:2]

    # Preallocate with NaN so unprocessed slices are clearly empty
    masked_shear = np.full((rows, cols, n_frames), np.nan, dtype=np.float32)
    masked_vort = np.full_like(masked_shear, np.nan)
    masked_velmag = np.full_like(masked_shear, np.nan)
    masked_u = np.full_like(masked_shear, np.nan)
    masked_v = np.full_like(masked_shear, np.nan)

    for k in tqdm(range(n_frames),
                  desc=f"  Stage 3 — {condition}",
                  unit="frame", leave=False):
        if dynamic_mask is not None:
            mask_combined = typevector & dynamic_mask[:, :, k]
        else:
            mask_combined = typevector

        # 5x5 box filter — MATLAB imfilter(F, fspecial('average',[5 5]))
        # uses correlation with zero boundary; uniform_filter mode='constant'
        # matches.
        shear_frame = uniform_filter(S33[:, :, k].astype(np.float64),
                                      size=5, mode="constant", cval=0.0)
        vort_frame = uniform_filter(vorticity[:, :, k].astype(np.float64),
                                     size=5, mode="constant", cval=0.0)
        vel_frame = uniform_filter(velmag[:, :, k].astype(np.float64),
                                    size=5, mode="constant", cval=0.0)

        shear_frame[~mask_combined] = np.nan
        vort_frame[~mask_combined] = np.nan
        vel_frame[~mask_combined] = np.nan

        uc = avgu[:, :, k].astype(np.float64)
        vc = avgv[:, :, k].astype(np.float64)
        uc[~mask_combined] = np.nan
        vc[~mask_combined] = np.nan

        masked_shear[:, :, k] = shear_frame
        masked_vort[:, :, k] = vort_frame
        masked_velmag[:, :, k] = vel_frame
        masked_u[:, :, k] = uc
        masked_v[:, :, k] = vc

    t_trimmed = t[:n_frames]

    save_kwargs = dict(
        X_norm=X_norm, Y_norm=Y_norm,
        X=X, Y=Y,
        masked_shear=masked_shear,
        masked_vort=masked_vort,
        masked_velmag=masked_velmag,
        masked_u=masked_u,
        masked_v=masked_v,
        AA=np.int64(AA),
        t=t_trimmed,
        condition=np.array(condition),
    )
    if dynamic_mask is not None:
        save_kwargs["dynamic_masking"] = dynamic_mask

    out_path = out_dir / f"{condition}_Tertiary_Export.npz"
    np.savez_compressed(out_path, **save_kwargs)

    print(f"  [Stage 3] Saved: {out_path}")
    return out_path
