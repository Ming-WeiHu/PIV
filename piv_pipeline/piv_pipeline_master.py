"""piv_pipeline_master.py — Python port of PIV_Pipeline_Master.m

Orchestrator for the PIV post-processing pipeline. Stages 1–4 always run;
videos can be opted in with --make-videos (in MATLAB this was a separate
script).

Usage
-----
    # Package style (preferred):
    python -m piv_pipeline.piv_pipeline_master --input-dir "/path/to/Primary Exports"

    # Or run the file directly:
    python piv_pipeline_master.py --input-dir "..."

The config.yaml beside this file replaces the MATLAB inputdlg() prompt. Any
field can be overridden on the CLI (--fps, --centre-x, etc.).

Image folder convention for Stage 2 (mirrors the MATLAB master):
    <input_dir.parent>/<vol>-<angle>-<cpm_str>-imgs/Test1/<vol>-<angle>-<cpm_str>/

Override with --images-base PATH if your layout differs:
    <PATH>/<vol>-<angle>-<cpm_str>-imgs/Test1/<vol>-<angle>-<cpm_str>/

Single-file example (skip the per-frame TIF masking stage):
    python -m piv_pipeline.piv_pipeline_master \
        --input-dir "/path/to/Test1/firstsave.npz" --skip-masking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

# Support both `python -m piv_pipeline.piv_pipeline_master` and
# `python piv_pipeline_master.py` invocation.
try:
    from .io_loaders import load_primary, parse_filename
    from .fn_secondary_export import secondary_export
    from .fn_dynamic_masking import dynamic_masking
    from .fn_tertiary_export import tertiary_export
    from .fn_final_analysis import final_analysis
    from .piv_generate_videos import generate_videos
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from io_loaders import load_primary, parse_filename
    from fn_secondary_export import secondary_export
    from fn_dynamic_masking import dynamic_masking
    from fn_tertiary_export import tertiary_export
    from fn_final_analysis import final_analysis
    from piv_generate_videos import generate_videos


# Files in the primary-exports folder that look like our own outputs — skip.
EXCLUDE_KW = ("Secondary_Export", "Tertiary_Export", "dynamic_masking")


def _gather_inputs(input_dir: Path) -> list[Path]:
    inputs: list[Path] = []
    for ext in (".mat", ".npz"):
        inputs.extend(input_dir.glob(f"*{ext}"))
    return sorted(p for p in inputs if not any(kw in p.name for kw in EXCLUDE_KW))


def _load_config(path: Path | None) -> dict:
    cfg_path = path if path else Path(__file__).with_name("config.yaml")
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    return yaml.safe_load(cfg_path.read_text())


def _find_images_folder(
    input_dir: Path, name_info: dict,
    images_base: Path | None, images_folder: Path | None,
) -> Path:
    """Resolve the TIF folder for Stage 2.

    --images-folder wins if set (direct path, no substructure).
    Otherwise apply the MATLAB convention:
        <base>/<vol-angle-cpm>-imgs/Test1/<vol-angle-cpm>
    where <base> = --images-base if set, else parent of --input-dir.
    """
    if images_folder is not None:
        return images_folder
    base = images_base if images_base else input_dir.parent
    sub = f"{name_info['volume']}-{name_info['angle']}-{name_info['cpm_str']}"
    return base / f"{sub}-imgs" / "Test1" / sub


def _grid_shape_from_secondary(sec_path: Path) -> tuple[int, int]:
    """Pull (rows, cols) from an existing Secondary Export."""
    with np.load(sec_path) as d:
        shape = d["avgu"].shape   # (rows, cols, n_frames)
    return shape[0], shape[1]


def _load_secondary_convert_velocity(sec_path: Path) -> bool | None:
    """Return the stored convert_velocity flag from a Secondary Export."""
    with np.load(sec_path) as d:
        if "convert_velocity" in d:
            return bool(d["convert_velocity"].astype(np.bool_))
    return None


def run_pipeline_for_condition(
    primary_path, *, images_folder, cfg, out_dir, do_masking,
    skip_tertiary: bool = False, force=False,
) -> dict:
    """Run Stage 1 (secondary) → Stage 2 (dynamic mask, optional) → Stage 3
    (tertiary) for ONE primary export. Stage 4 (final_analysis) is run once
    by the caller over all tertiary exports. Returns paths dict.

    When skip_tertiary is True, Stages 1 and 2 run normally but Stage 3 is
    skipped; result["tertiary_path"] is set to None."""
    primary_path = Path(primary_path)
    out_dir = Path(out_dir)
    sec_dir = out_dir / "Secondary Exports"
    masks_dir = out_dir / "Dynamic Masks"
    tert_dir = out_dir / "Tertiary Exports"
    for d in (sec_dir, masks_dir, tert_dir):
        d.mkdir(parents=True, exist_ok=True)

    name_info = parse_filename(primary_path.stem)
    condition = name_info["condition"]
    result = {"condition": condition, "secondary_path": None,
              "mask_path": None, "tertiary_path": None}

    # ---- Stage 1: Secondary Export ----
    sec_path = sec_dir / f"{condition}_Secondary_Export.npz"
    if sec_path.exists() and not force:
        mask_rows, mask_cols = _grid_shape_from_secondary(sec_path)
    else:
        primary = load_primary(primary_path)
        mask_rows, mask_cols = secondary_export(
            primary,
            centre_x=cfg["centre_x"], centre_y=cfg["centre_y"],
            fps=cfg["fps"], AA=cfg["AA"], howstupid=cfg["howstupid"],
            convert_velocity=cfg["convert_velocity"], nan_fill=cfg["nan_fill"],
            out_dir=sec_dir, name_info=name_info,
        )
    result["secondary_path"] = sec_path

    # ---- Stage 2: Dynamic Masking (optional) ----
    mask_path = None
    candidate_mask = masks_dir / f"dynamic_masking_{condition}.npz"
    if do_masking and images_folder is not None and Path(images_folder).is_dir():
        if candidate_mask.exists() and not force:
            mask_path = candidate_mask
        else:
            dynamic_masking(
                Path(images_folder), condition=condition,
                lower_bound=cfg["lower_bound"],
                mask_rows=mask_rows, mask_cols=mask_cols, out_dir=masks_dir,
                method=cfg["mask_method"],
                texture_ksize=int(cfg["mask_particle_ksize"]),
                tophat_particle_ksize=int(cfg["mask_particle_ksize"]),
                tophat_region_ksize=int(cfg["mask_region_ksize"]),
                tophat_keep_pct=cfg["mask_keep_pct"],
                simple_threshold=float(cfg["mask_threshold"]),
                simple_direction=cfg["mask_direction"],
                deviation_threshold=cfg["mask_deviation_threshold"],
                morph_open_ksize=int(cfg["mask_open_ksize"]),
                morph_close_ksize=int(cfg["mask_close_ksize"]),
                uniform_floor=float(cfg["mask_uniform_floor"]),
            )
            mask_path = candidate_mask
    result["mask_path"] = mask_path

    # ---- Stage 3: Tertiary Export ----
    if skip_tertiary:
        print("  [Stage 3] Skipped (skip_tertiary=True).")
        result["tertiary_path"] = None
        return result
    tert_path = tert_dir / f"{condition}_Tertiary_Export.npz"
    if not (tert_path.exists() and not force):
        tertiary_export(sec_path, mask_path, R=cfg["R"], out_dir=tert_dir)
    result["tertiary_path"] = tert_path
    return result


def _apply_cfg_defaults(cfg: dict) -> dict:
    """Fill in defaults for keys that may be absent from config.yaml. Shared by the CLI and the GUI batch worker."""
    cfg.setdefault("span", 50)
    cfg.setdefault("convert_velocity", True)
    cfg.setdefault("nan_fill", "interpolate")
    cfg.setdefault("mask_method", "tophat")
    cfg.setdefault("mask_keep_pct", None)
    cfg.setdefault("mask_particle_ksize", 15)
    cfg.setdefault("mask_region_ksize", 31)
    cfg.setdefault("mask_threshold", 128.0)
    cfg.setdefault("mask_direction", "above")
    cfg.setdefault("mask_open_ksize", 7)
    cfg.setdefault("mask_close_ksize", 41)
    cfg.setdefault("mask_uniform_floor", 0.0)
    cfg.setdefault("mask_deviation_threshold", None)
    return cfg


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PIV post-processing pipeline (Stages 1-4 + opt-in videos)."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Folder of primary exports OR a single .mat / .npz file.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output root folder. Default: --input-dir. Subfolders 'Secondary Exports' "
             "and 'Dynamic Masks' are auto-created beneath it.",
    )
    parser.add_argument(
        "--config", default=None,
        help="YAML config (default: piv_pipeline/config.yaml beside this file).",
    )
    parser.add_argument(
        "--images-base", default=None,
        help="Override base folder for Stage 2 TIF lookup with the "
             "<vol-angle-cpm>-imgs/Test1/<vol-angle-cpm>/ substructure. "
             "Default: parent of --input-dir.",
    )
    parser.add_argument(
        "--images-folder", default=None,
        help="Direct path to a flat folder of TIF images for Stage 2. "
             "Bypasses the MATLAB convention. Used for ALL primary exports "
             "in --input-dir (so only useful for single-condition runs).",
    )
    parser.add_argument(
        "--skip-masking", action="store_true",
        help="Skip Stage 2 (dynamic masking).",
    )
    parser.add_argument(
        "--skip-tertiary", action="store_true",
        help="Skip Stage 3 (tertiary export).",
    )
    parser.add_argument(
        "--skip-final", action="store_true",
        help="Skip Stage 4 (final analysis + summary plots + CSV).",
    )
    parser.add_argument(
        "--make-videos", action="store_true",
        help="After Stage 4, render shear-rate MP4 videos.",
    )
    parser.add_argument("--vid-fps", type=float, default=20,
                        help="--make-videos: output frame rate (default 20).")
    parser.add_argument("--vid-step", type=int, default=3,
                        help="--make-videos: sample every Nth frame (default 3).")
    parser.add_argument("--vid-clim-max", type=float, default=30,
                        help="--make-videos: shear-rate colour bar max (default 30).")
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute even if outputs already exist.",
    )
    parser.add_argument(
        "--convert-velocity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Convert px/frame velocities to m/s before gradient/shear computation.",
    )
    parser.add_argument(
        "--nan-fill", dest="nan_fill", choices=("interpolate", "zero"), default=None,
        help="How to fill masked/capped holes before gradients. "
             "interpolate = PIVlab-style neighbour interpolation (default); "
             "zero = legacy NaN->0.",
    )
    # Per-field overrides
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--centre-x", dest="centre_x", type=float, default=None)
    parser.add_argument("--centre-y", dest="centre_y", type=float, default=None)
    parser.add_argument("--lower-bound", dest="lower_bound", type=float, default=None)
    parser.add_argument("--R", type=float, default=None,
                        help="Vessel radius in metres (overrides config).")
    parser.add_argument("--AA", type=int, default=None)
    parser.add_argument("--span", type=int, default=None,
                        help="Stage 4 smoothing window (frames).")
    parser.add_argument("--howstupid", type=float, default=None)
    # Stage 2 dynamic-mask overrides
    parser.add_argument("--mask-method", dest="mask_method",
                        choices=("tophat", "brightness", "texture", "intensity",
                                 "simple", "deviation"), default=None)
    parser.add_argument("--mask-keep-pct", dest="mask_keep_pct", type=float, default=None)
    parser.add_argument("--mask-particle-ksize", dest="mask_particle_ksize", type=int, default=None)
    parser.add_argument("--mask-region-ksize", dest="mask_region_ksize", type=int, default=None)
    parser.add_argument("--mask-threshold", dest="mask_threshold", type=float, default=None)
    parser.add_argument("--mask-direction", dest="mask_direction",
                        choices=("above", "below"), default=None)
    parser.add_argument("--mask-open-ksize", dest="mask_open_ksize", type=int, default=None)
    parser.add_argument("--mask-close-ksize", dest="mask_close_ksize", type=int, default=None)
    parser.add_argument("--mask-uniform-floor", dest="mask_uniform_floor", type=float, default=None)
    parser.add_argument("--mask-deviation-threshold", dest="mask_deviation_threshold",
                        type=float, default=None)
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.config) if args.config else None)

    # Apply CLI overrides on top of config
    for key in ("fps", "centre_x", "centre_y", "lower_bound", "R", "AA", "span",
                "howstupid", "convert_velocity", "nan_fill",
                "mask_method", "mask_keep_pct", "mask_particle_ksize",
                "mask_region_ksize", "mask_threshold", "mask_direction",
                "mask_open_ksize", "mask_close_ksize", "mask_uniform_floor",
                "mask_deviation_threshold"):
        v = getattr(args, key)
        if v is not None:
            cfg[key] = v
    # Defaults for keys we added later (and Stage 2 mask fallbacks)
    _apply_cfg_defaults(cfg)

    input_path = Path(args.input_dir).resolve()
    if not input_path.exists():
        raise SystemExit(f"Input path does not exist: {input_path}")

    # Accept either a folder (scan for primary exports) or a single file.
    if input_path.is_file():
        if input_path.suffix.lower() not in (".mat", ".npz"):
            raise SystemExit(
                f"Single-file input must be .mat or .npz, got: {input_path.suffix}"
            )
        files = [input_path]
        input_dir = input_path.parent
        default_out = input_dir
    else:
        files = _gather_inputs(input_path)
        input_dir = input_path
        default_out = input_path

    out_root = Path(args.output_dir).resolve() if args.output_dir else default_out
    sec_dir = out_root / "Secondary Exports"
    masks_dir = out_root / "Dynamic Masks"
    tert_dir = out_root / "Tertiary Exports"
    summary_dir = out_root / "Summary"
    sec_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    tert_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    images_base = Path(args.images_base).resolve() if args.images_base else None
    images_folder = Path(args.images_folder).resolve() if args.images_folder else None

    if not files:
        raise SystemExit(f"No primary exports (.mat or .npz) found in:\n  {input_path}")

    print(f"\nPrimary exports:     {input_path}")
    print(f"Output root:         {out_root}")
    print(f"  Secondary Exports: {sec_dir}")
    print(f"  Dynamic Masks:     {masks_dir}")
    print(f"  Tertiary Exports:  {tert_dir}")
    print(f"  Summary:           {summary_dir}\n")
    print(f"Found {len(files)} primary export(s):")
    for f in files:
        print(f"  {f.name}")
    print(f"\nVelocity conversion: {'enabled' if cfg['convert_velocity'] else 'disabled'}")
    print(
        f"\nSettings: fps={cfg['fps']}  centreX={cfg['centre_x']}  "
        f"centreY={cfg['centre_y']}  AA={cfg['AA']}  howstupid={cfg['howstupid']}  "
        f"convert_velocity={cfg['convert_velocity']}\n"
        f"          lower_bound={cfg['lower_bound']}  R={cfg['R']} m\n"
    )

    for i, f in enumerate(files, start=1):
        print(f"========== [{i}/{len(files)}] {f.name} ==========")
        try:
            name_info = parse_filename(f.stem)
        except ValueError as e:
            print(f"  SKIP — {e}")
            continue
        imgs_folder_arg = None
        if not args.skip_masking:
            imgs_folder_arg = _find_images_folder(
                input_dir, name_info, images_base, images_folder
            )
        try:
            run_pipeline_for_condition(
                f, images_folder=imgs_folder_arg, cfg=cfg, out_dir=out_root,
                do_masking=not args.skip_masking,
                skip_tertiary=args.skip_tertiary,
                force=args.force,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"  FAILED — {e}")
            continue
        print()

    # ---- Stage 4: Final Analysis (over all tertiary exports) --------------
    if not args.skip_final:
        print("========== Stage 4: Final Analysis ==========")
        try:
            final_analysis(tert_dir, summary_dir, span=int(cfg["span"]))
        except FileNotFoundError as e:
            print(f"  [Stage 4] FAILED — {e}")
        print()

    # ---- Videos (opt-in) --------------------------------------------------
    if args.make_videos:
        print("========== Generating videos ==========")
        try:
            generate_videos(
                tert_dir,
                out_dir=out_root / "Videos",
                vid_fps=args.vid_fps,
                step=args.vid_step,
                clim_max=args.vid_clim_max,
            )
        except FileNotFoundError as e:
            print(f"  [Videos] FAILED — {e}")
        print()

    print("========== Pipeline run complete ==========")
    print(f"Outputs in: {out_root}")


if __name__ == "__main__":
    main()
    

