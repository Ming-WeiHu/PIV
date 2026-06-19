# PIV Analysis — Oribiotech Bioreactor

Particle Image Velocimetry (PIV) pipeline for analysing fluid flow inside the IRO bioreactor under rocking conditions. Built to replicate and extend the PIVlab MATLAB workflow in Python.

---

## Overview

PIVlab (MATLAB) produces a **Primary Export** (`.mat`). This codebase ports the downstream MATLAB post-processing pipeline into Python and provides interactive GUIs for exploring velocity caps and shear rates.

The pipeline has four stages:

```
Primary Export (.mat / .npz)
  └─ Stage 1: Secondary Export   (AA=10 time-averaged fields, S33 shear)
       └─ Stage 2: Dynamic Masking  (image-based mask from TIFs)
            └─ Stage 3: Tertiary Export  (5×5 smoothed, masked shear)
                 └─ Stage 4: Final Analysis  (span-50 smoothed, TimeAvg_mean)
```

The key output metric is **`TimeAvg_mean`** (time-averaged mean shear rate, 1/s), reported in `results_rock.csv`.

---

## File Structure

### Core Algorithm
- **`piv_simple.py`** — PIV algorithm implementation; outputs `.npz` primaries (px/frame units). Multi-pass window deformation with per-pass vector validation, PIVlab-style `smoothn` field smoothing, and an optional velocity cap.
- **`piv_openpiv.py`** — OpenPIV-based alternative processor
- **`smoothn.py`** — vendored Garcia (2010) robust smoother (faithful port of PIVlab's `smoothn`); used by `piv_simple` for per-pass field smoothing

### GUIs
- **`piv_gui.py`** — Main Tkinter front-end. Runs PIV on an image folder, and also runs the **full 4-stage pipeline end-to-end**: a **batch** mode (point at a parent folder of `vol-deg-cpm`-named condition subfolders → one combined `results_rock.csv`) and a **single-condition** button (runs the currently-loaded folder through PIV → pipeline). Both reuse the same `run_pipeline_for_condition` runner as the CLI, so GUI and CLI stay in lockstep.

### Post-PIVlab Pipeline (`piv_pipeline/`)
| File | Role |
|------|------|
| `piv_pipeline_master.py` | CLI orchestrator — runs all 4 stages. Per-condition Stages 1–3 live in `run_pipeline_for_condition` (shared with the GUI batch); Stage 4 (`final_analysis`) runs once over all conditions. Shared cfg defaults via `_apply_cfg_defaults`. |
| `fn_secondary_export.py` | Stage 1: AA=10 nanmean, S33 shear computation |
| `fn_dynamic_masking.py` | Stage 2: image-based mask from TIF frames |
| `fn_tertiary_export.py` | Stage 3: 5×5 box smooth, apply mask |
| `fn_final_analysis.py` | Stage 4: span-50 smooth, TimeAvg_mean output |
| `io_loaders.py` | Unified loader for `.mat` (v7/v7.3) and `.npz` primaries |
| `piv_generate_videos.py` | Render velocity/shear fields as video |

### Comparison & Validation
- **`compare_tertiary_final.py`** — Python vs MATLAB tertiary export comparison
- **`plot_primary_export.py`**, **`plotsecondarytotert.py`** — diagnostic plots

---

## Quickstart

### Run the pipeline (CLI)

```bash
python piv_pipeline/piv_pipeline_master.py \
    --input path/to/Primary_Export.mat \
    --images-folder path/to/TIF_frames/ \
    --output-dir path/to/output/
```

> **Important:** always pass `--images-folder` pointing at the TIF frames. Without it, Stage 2 dynamic masking silently falls back to typevector-only (no image mask), which inflates shear significantly.

### Run PIV + pipeline from the GUI

```bash
python piv_gui.py
```

- **Batch (multi-condition):** in the *Batch* panel, pick a **parent folder** containing one subfolder per condition (named `50mL-20deg-35cpm` etc., each holding its TIF frames). Leave *Dynamic masking* on (**TIF frames only**), optionally override fps/AA/R/span/centre from `config.yaml`, then **Run batch pipeline**. Output is the shared structure (`Secondary Exports/`, `Dynamic Masks/`, `Tertiary Exports/`, `Summary/results_rock.csv` with one row per condition); `final_analysis` runs once over all conditions. Subfolders whose names don't parse, or with <2 images, are skipped with a warning.
- **Single condition:** load one folder of frames in the normal PIV loader, then click **Run PIV + pipeline (this folder)** — it runs just that folder through PIV → pipeline.

Both paths build each condition's primary `.npz` from PIV, then call the same `run_pipeline_for_condition` the CLI uses.

---

## Key Concepts

### Inputs
- **Primary Export (`.mat`)** — PIVlab output. May store `u`/`v` already in m/s (check the `units` field — the loader handles this automatically via `already_calibrated`).
- **Primary (`.npz`)** — `piv_simple.py` output; always in px/frame.
- **TIF frames** — 16-bit grayscale, standard PIVlab calibration format; required for Stage-2 dynamic masking.

### What drives shear
Shear (`TimeAvg_mean`) is driven by **masking**, not by the `nan_fill` setting. Skipping the image mask (`--skip-masking`) inflates shear by ~50% due to boundary cells and box-filter edge bleed. The `nan_fill` mode (zero vs interpolate) changes the result by less than 0.1%.

Use `nan_fill: interpolate` (default) — this matches PIVlab's own approach.

### Velocity cap
The Python PIV implementation produces a spurious high-velocity tail not present in PIVlab's output — a translation artefact from differences in correlation normalisation and sub-pixel estimation. This tail inflates the shear estimate. The velocity cap is **on by default** to suppress it: it drops the top 0.7% of each component (`|u|` and `|v|` independently, computed **per frame-pair**) and rejects a whole vector when **either** exceeds its threshold. Disable/retune via the GUI checkbox or the CLI flags `--velocity-cap-percentile` / `--velocity-cap-px` / `--no-velocity-cap`.

### Field smoothing (smoothn)
`smoothn.py` is a vendored port of Garcia (2010)'s DCT-based smoother, the same algorithm PIVlab uses per-pass. It is available in `piv_simple.py` via `PIVSettings.enable_smoothn` but is **off by default** — in a heavily masked field (large NaN regions at vessel walls), smoothn extrapolates into the masked boundary and can inflate velocities in adjacent valid cells. Toggle on via the GUI checkbox to experiment; for production runs use the velocity cap instead.

### Python vs MATLAB shear gap
Python pipeline (masked, interpolate, no cap) gives ~7.2 1/s vs MATLAB ~6.55 1/s (~10% gap). The gap is driven by an upper tail of spurious vectors — the velocity cap closes most of it.

---

## Dependencies

Install into the `ORI` conda environment:

```
numpy
scipy
h5py
opencv-python
openpiv
matplotlib
tkinter  (stdlib)
```

