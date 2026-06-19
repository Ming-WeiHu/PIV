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

### GUI
- **`piv_gui.py`** — Main Tkinter front-end. Runs PIV on an image folder, and now also runs the **full 4-stage pipeline end-to-end**: a **batch** mode (point at a parent folder of `vol-deg-cpm`-named condition subfolders → one combined `results_rock.csv`) and a **single-condition** button (runs the currently-loaded folder through PIV → pipeline). Both reuse the same `run_pipeline_for_condition` runner as the CLI, so GUI and CLI stay in lockstep.

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
| `tune_mask_gui.py` | GUI for tuning Stage-2 mask parameters |

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
The Python PIV implementation produces a spurious high-velocity tail not present in PIVlab's output — a translation artifact from differences in correlation normalisation and sub-pixel estimation. This tail inflates the shear estimate. The velocity cap is on by default to suppress it: it drops the top 0.7% of each component (|u| and |v| independently, computed per frame-pair) and rejects a whole vector when either exceeds its threshold. Disable/retune via the GUI checkbox or the CLI flags --velocity-cap-percentile / --velocity-cap-px / --no-velocity-cap.

### Field smoothing (smoothn)
PIVlab smooths the velocity field at the end of **every pass** (before the next image deformation), and this matches its `+piv/piv_FFTmulti.m`: non-robust Garcia `smoothn` with a fixed `s=4` on intermediate passes and auto-GCV on the last pass (the last-pass smoothed field is what's exported). `piv_simple.py` mirrors this exactly via `smoothn.py`, applied after the per-pass outlier replacement (so it's deliberately **non-robust** — validation is already done upstream). It suppresses spurious high-velocity vectors at their source rather than clipping them like the cap. **On by default** (`PIVSettings.enable_smoothn`), independent of the velocity cap — toggle either (GUI checkbox) to A/B their effect on shear.

### Python vs MATLAB shear gap
Python pipeline (masked, interpolate, no cap) gave ~7.2 1/s vs MATLAB tertiary 6.55 1/s (~10% gap), shrinking to ~10% with the cap on. The gap responds to *capping* — i.e. it's driven by an upper **tail** of spurious vectors (concentrated in `|v|`), not a uniform conversion-factor error. The faithful `smoothn` port above is the principled fix (it's what PIVlab does to clean that tail); the velocity cap remains as a separate, blunter workaround. **A/B not yet measured** — regenerate a primary with smoothn on/off and compare pipeline `TimeAvg_mean` to the 6.55 baseline to confirm.

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

---

