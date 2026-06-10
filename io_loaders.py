"""io_loaders.py — load PIVlab .mat primary exports OR piv_simple .npz files.

Both formats are normalized to the same dict shape so downstream stages don't
care where the data came from.

Normalized output (dict)
------------------------
    x          : (rows, cols) float64  — grid x in METRES (post-calibration)
    y          : (rows, cols) float64  — grid y in METRES
    u_list     : list of (rows, cols) float64  — u per frame, PIXELS/FRAME, NaN over mask
    v_list     : list of (rows, cols) float64  — v per frame, PIXELS/FRAME, NaN over mask
    typevector : (rows, cols) uint8    — 1 = valid, 0 = masked (PIVlab convention)
    calxy      : float                  — metres per pixel
    n_pairs    : int                    — number of frame pairs
    source     : str                    — 'mat' / 'mat-v73' / 'npz'
    path       : Path                   — input file path

NOTES
-----
PIVlab outputs `u_original` / `v_original` in PIXELS / FRAME — that convention
is preserved here (the MATLAB secondary export expects pixels/frame).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


# ─────────────────────────────── dispatch ────────────────────────────────

def load_primary(path: str | Path) -> dict[str, Any]:
    """Auto-detect format from extension and load."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".npz":
        return _load_npz(p)
    if suf == ".mat":
        return _load_mat(p)
    raise ValueError(f"Unknown extension '{suf}' for {p} — expected .mat or .npz")


# ──────────────────────────────────── .npz ────────────────────────────────

def _load_npz(p: Path) -> dict[str, Any]:
    """Load piv_simple.py output. Expects keys: calxy, x, y, u_original, v_original."""
    d = np.load(p, allow_pickle=False)
    required = {"calxy", "x", "y", "u_original", "v_original"}
    missing = required - set(d.files)
    if missing:
        raise KeyError(
            f"{p}: missing keys {sorted(missing)} (found {sorted(d.files)})"
        )

    u = d["u_original"]   # (n_pairs, rows, cols), pixels/frame, NaN over mask
    v = d["v_original"]
    if u.ndim != 3 or v.ndim != 3:
        raise ValueError(f"{p}: u_original/v_original must be 3-D (got {u.shape})")
    if u.shape != v.shape:
        raise ValueError(f"{p}: u/v shape mismatch ({u.shape} vs {v.shape})")

    calxy = float(d["calxy"])
    n_pairs = u.shape[0]

    # piv_simple stores x, y in PIXELS. The downstream code expects metric
    # coords (PIVlab-style), so convert here.
    x = d["x"].astype(np.float64) * calxy
    y = d["y"].astype(np.float64) * calxy

    # Prefer an explicit typevector key (written by _export_npz so that
    # cap-capped cells in frame 0 are not baked in as permanently non-fluid).
    # Fall back to deriving from frame 0 for plain piv_simple outputs where
    # the NaN mask is constant across all frames.
    if "typevector" in d.files:
        typevector = d["typevector"].astype(np.uint8)
    else:
        typevector = np.isfinite(u[0]).astype(np.uint8)

    has_calu = "calu" in d.files
    has_calv = "calv" in d.files
    calu = float(d["calu"]) if has_calu else 1.0
    calv = float(d["calv"]) if has_calv else 1.0
    return {
        "x": x,
        "y": y,
        "u_list": [u[i].astype(np.float64) for i in range(n_pairs)],
        "v_list": [v[i].astype(np.float64) for i in range(n_pairs)],
        "typevector": typevector,
        "calxy": calxy,
        "calu": calu,
        "calv": calv,
        "has_calu": has_calu,
        "has_calv": has_calv,
        "already_calibrated": False,
        "n_pairs": n_pairs,
        "source": "npz",
        "path": p,
    }


# ──────────────────────────────────── .mat ────────────────────────────────

def _load_mat(p: Path) -> dict[str, Any]:
    """Load a PIVlab primary export. Tries scipy.io.loadmat (v6/v7) first;
    falls back to h5py for v7.3 HDF5-format .mat files."""
    try:
        return _load_mat_v7(p)
    except (NotImplementedError, ValueError):
        # scipy raises NotImplementedError for v7.3; some installs raise
        # ValueError. Try the HDF5 path.
        return _load_mat_v73(p)


def _load_mat_v7(p: Path) -> dict[str, Any]:
    """scipy.io.loadmat path — works for v6 and v7 (NOT v7.3)."""
    from scipy.io import loadmat
    raw = loadmat(p, squeeze_me=False)
    if "x" not in raw:
        raise KeyError(
            f"{p}: PIVlab key 'x' not found — is this a primary export?"
        )

    # PIVlab cell arrays come back as object arrays of shape (N, 1).
    x_cell = raw["x"]
    y_cell = raw["y"]
    u_cell = raw["u_original"]
    v_cell = raw["v_original"]
    tv_cell = raw["typevector_original"]
    calxy = float(np.array(raw["calxy"]).squeeze())

    n_pairs = u_cell.shape[0]
    # x, y are the same grid every frame — take the first.
    x = np.asarray(x_cell[0, 0], dtype=np.float64)
    y = np.asarray(y_cell[0, 0], dtype=np.float64)
    u_list = [np.asarray(u_cell[i, 0], dtype=np.float64) for i in range(n_pairs)]
    v_list = [np.asarray(v_cell[i, 0], dtype=np.float64) for i in range(n_pairs)]
    typevector = np.asarray(tv_cell[0, 0], dtype=np.uint8)

    has_calu = "calu" in raw
    has_calv = "calv" in raw
    calu = float(np.array(raw["calu"]).squeeze()) if has_calu else 1.0
    calv = float(np.array(raw["calv"]).squeeze()) if has_calv else 1.0
    # Detect PIVlab "calibrated" export where u/v are already in m/s
    already_calibrated = False
    if "units" in raw:
        try:
            already_calibrated = "m/s" in str(raw["units"]).strip()
        except Exception:
            pass
    return {
        "x": x,
        "y": y,
        "u_list": u_list,
        "v_list": v_list,
        "typevector": typevector,
        "calxy": calxy,
        "calu": calu,
        "calv": calv,
        "has_calu": has_calu,
        "has_calv": has_calv,
        "already_calibrated": already_calibrated,
        "n_pairs": n_pairs,
        "source": "mat",
        "path": p,
    }


def _load_mat_v73(p: Path) -> dict[str, Any]:
    """h5py path for -v7.3 (HDF5) PIVlab files.

    PIVlab v7.3 stores cell arrays as HDF5 object references. We dereference
    each one and transpose because MATLAB writes column-major.
    """
    import h5py

    with h5py.File(p, "r") as f:
        def _read_cell(name: str) -> list[np.ndarray]:
            if name not in f:
                raise KeyError(f"{p}: PIVlab key '{name}' not found in v7.3 file.")
            refs = f[name][()]
            out = []
            for ref in np.asarray(refs).ravel():
                out.append(np.array(f[ref]).T)
            return out

        x_cells = _read_cell("x")
        y_cells = _read_cell("y")
        u_cells = _read_cell("u_original")
        v_cells = _read_cell("v_original")
        tv_cells = _read_cell("typevector_original")
        calxy = float(np.array(f["calxy"]).squeeze())
        has_calu = "calu" in f
        has_calv = "calv" in f
        calu = float(np.array(f["calu"]).squeeze()) if has_calu else 1.0
        calv = float(np.array(f["calv"]).squeeze()) if has_calv else 1.0
        already_calibrated = False
        if "units" in f:
            try:
                u_data = f["units"][()]
                if hasattr(u_data, "dtype") and u_data.dtype.kind in ("u", "i"):
                    u_str = "".join(chr(int(c)) for c in u_data.ravel()
                                    if 0 < int(c) < 0x10000)
                else:
                    u_str = str(u_data)
                already_calibrated = "m/s" in u_str
            except Exception:
                pass

    n_pairs = len(u_cells)
    return {
        "x": x_cells[0].astype(np.float64),
        "y": y_cells[0].astype(np.float64),
        "u_list": [u.astype(np.float64) for u in u_cells],
        "v_list": [v.astype(np.float64) for v in v_cells],
        "typevector": tv_cells[0].astype(np.uint8),
        "calxy": calxy,
        "calu": calu,
        "calv": calv,
        "has_calu": has_calu,
        "has_calv": has_calv,
        "already_calibrated": already_calibrated,
        "n_pairs": n_pairs,
        "source": "mat-v73",
        "path": p,
    }


# ──────────────────────────── filename parsing ────────────────────────────

def parse_filename(stem: str) -> dict[str, Any]:
    """Parse '100mL-20deg-15cpm' → volume, angle, cpm, condition.

    Mirrors the MATLAB `strsplit('-')` logic in fn_secondary_export.m:
        volume    = parts[0]                      # '100mL'
        angle     = parts[1]                      # '20deg'
        cpm_str   = parts[2]                      # '15cpm'
        cpm       = int(cpm_str[:-3])             # 15
        condition = f'{volume}_{cpm}-0-0_{angle}' # '100mL_15-0-0_20deg'

    If the filename doesn't match this convention, falls back to using `stem`
    itself as the condition string (with placeholder volume/angle/cpm). This
    keeps the pipeline runnable on ad-hoc filenames like `firstsave.npz`.
    """
    parts = stem.split("-")
    # Strict convention: 3 dash-separated parts, last ends in 'cpm'
    if len(parts) >= 3 and parts[2].endswith("cpm"):
        try:
            volume = parts[0]
            angle = parts[1]
            cpm_str = parts[2]
            cpm = int(cpm_str[:-3])
            return {
                "volume": volume,
                "angle": angle,
                "cpm": cpm,
                "cpm_str": cpm_str,
                "condition": f"{volume}_{cpm}-0-0_{angle}",
                "stem": stem,
                "strict": True,
            }
        except ValueError:
            pass

    # Lenient fallback: use the stem itself as the condition.
    return {
        "volume": "unknown",
        "angle": "unknown",
        "cpm": -1,
        "cpm_str": "unknowncpm",
        "condition": stem,
        "stem": stem,
        "strict": False,
    }
