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

- **`piv_simple.py`** — PIV algorithm implementation; outputs `.npz` primaries (px/frame units)
- **`piv_openpiv.py`** — OpenPIV-based alternative processor

### GUIs

- **`piv_gui.py`** — Main Tkinter front-end for running PIV on image pairs
- **`cap_pipeline_finder_gui.py`** — **Primary shear tool.** Runs the real 4-stage pipeline at a chosen velocity cap, reads `TimeAvg_mean` from `results_rock.csv`, and plots TimeAvg_mean vs cap with a ±10% MATLAB baseline band. Also has a 3-panel chart window comparing Python vs MATLAB S33, |v|, and |u| per frame.
- **`cap_explorer_gui.py`** — Lightweight interactive cap slider; shear preview is Stage-1 only (approximate, not pipeline-faithful). Use as a quick sanity check, not for final numbers.
- **`cap_explorer_advanced_gui.py`** — More complex explorer (not the default; use the finder instead).

### Post-PIVlab Pipeline (`piv_pipeline/`)

| File                       | Role                                                         |
| -------------------------- | ------------------------------------------------------------ |
| `piv_pipeline_master.py` | CLI orchestrator — runs all 4 stages                        |
| `fn_secondary_export.py` | Stage 1: AA=10 nanmean, S33 shear computation                |
| `fn_dynamic_masking.py`  | Stage 2: image-based mask from TIF frames                    |
| `fn_tertiary_export.py`  | Stage 3: 5×5 box smooth, apply mask                         |
| `fn_final_analysis.py`   | Stage 4: span-50 smooth, TimeAvg_mean output                 |
| `io_loaders.py`          | Unified loader for `.mat` (v7/v7.3) and `.npz` primaries |
| `piv_generate_videos.py` | Render velocity/shear fields as video                        |
| `tune_mask_gui.py`       | GUI for tuning Stage-2 mask parameters                       |

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

### Find the right velocity cap (GUI)

```bash
python cap_pipeline_finder_gui.py
```

Load your primary `.mat` or `.npz`, set the TIF folder once to build the dynamic mask (saved automatically to `Dynamic Masks/`), then sweep caps. A cap is considered good if its `TimeAvg_mean` falls within ±10% of the MATLAB baseline (default baseline: 6.55 1/s for 100mL-20deg-35cpm).

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

Capping removes fast vectors (typically ~7% on average, up to ~28% at rocking peaks). Because fast vectors tend to be low-shear flow cores, capping *raises* average shear. The cap is applied to both `|u|` and `|v|` independently.

### Python vs MATLAB shear gap

Python pipeline (masked, interpolate, no cap) gives ~7.2 1/s vs MATLAB tertiary 6.55 1/s (~10% gap). Root cause: Python `|v|` is ~3.7× too high relative to MATLAB; `|u|` matches well. This is a known open issue — capping `v` is a workaround, not a fix.

---

## Dependencies

Install into the Python environment:

```
numpy
scipy
h5py
opencv-python
openpiv
matplotlib
tkinter  (stdlib)
```

---
