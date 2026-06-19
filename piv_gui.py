"""
piv_gui.py — PIVlab-style folder-batch front-end for piv_simple.

Layout mirrors PIVlab: top menubar (File / Image acquisition / Image settings /
Analysis / Help), large image canvas on the left, scrollable settings sidebar
on the right with three panels stacked (PIV settings / Image pre-processing /
Calibration), log + status bar at the bottom.

Workflow mirrors PIVlab: load a *folder* of images (.tif, .tiff, .png, .jpg,
.bmp), pick a pairing mode (rolling A-B,B-C or consecutive A-B,A-B), batch
process every pair, scrub through results with a frame slider.

Independent from `gui.py` / `particle_testing.py` (the bioreactor analyzer).
Run with:
    python piv_gui.py
"""

from __future__ import annotations

import re
import threading
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk

import piv_simple as ps
from piv_pipeline.io_loaders import parse_filename


# ───────────────────────────── styling ───────────────────────────────────────

PANEL_BG = "#ffffff"
HEADER_FG = "#f60606"
HEADER_FONT = ("Segoe UI", 9, "bold")
GREEN_BG = "#aef0b0"
CANVAS_BG = "#1a1a1a"

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
TIFF_FIRST_FILETYPES = [
    ("TIFF images", "*.tif *.tiff"),
    ("All images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
    ("All files", "*.*"),
]


# ──────────────────────── folder + pair utilities ────────────────────────────

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r"(\d+)", str(s))]


def list_image_folder(folder: str,
                      exts: Tuple[str, ...] = IMAGE_EXTENSIONS) -> List[Path]:
    p = Path(folder)
    if not p.is_dir():
        return []
    files = [f for f in p.iterdir()
             if f.is_file() and f.suffix.lower() in exts]
    return sorted(files, key=lambda f: natural_key(f.name))


def collect_conditions(parent_dir):
    """Return ([{condition,name,folder,images}], skipped_names) for each
    subfolder whose name parses as a PIV condition and holds ≥2 images."""
    parent = Path(parent_dir)
    conds, skipped = [], []
    for sub in sorted(p for p in parent.iterdir() if p.is_dir()):
        info = parse_filename(sub.name)
        if not info.get("strict"):
            skipped.append(sub.name); continue
        imgs = list_image_folder(sub)
        if len(imgs) < 2:
            skipped.append(sub.name); continue
        conds.append({"condition": info["condition"], "name": sub.name,
                      "folder": sub, "images": imgs})
    return conds, skipped


def pair_indices(n: int, mode: str) -> List[Tuple[int, int]]:
    """Return list of (i, j) frame index pairs.

    "A-B,B-C" (rolling)     → (0,1), (1,2), (2,3), …  N-1 pairs.
    "A-B,A-B" (consecutive) → (0,1), (2,3), (4,5), …  N/2 pairs.
    """
    if n < 2:
        return []
    if mode == "A-B,A-B":
        return [(i, i + 1) for i in range(0, n - 1, 2)]
    return [(i, i + 1) for i in range(n - 1)]


def load_image_uint8(path: str, max_dim: int = 1400) -> np.ndarray:
    """Load any supported image type, return a uint8 array suitable for display.

    16-bit TIFFs are stretched to fill 0–255 so fine features remain visible.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not load: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        if img.max() > img.min():
            img = ((img.astype(np.float32) - img.min())
                   / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return img


# ──────────────────────── image loaders for picker ───────────────────────────

def _fmt_dur(seconds: float) -> str:
    """Format seconds as M:SS (or H:MM:SS if > 1 hour)."""
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def load_image_full_uint8(path: str) -> np.ndarray:
    """Load image at full resolution, normalised to uint8. Unlike
    `load_image_uint8`, this does NOT downscale, so click coordinates land in
    the original pixel space.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not load: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        if img.max() > img.min():
            img = ((img.astype(np.float32) - img.min())
                   / (img.max() - img.min()) * 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img


# ──────────────────────── primary-npz I/O ────────────────────────────────────

def save_primary_npz(out_path, x, y, u_orig, v_orig, cal, in_world_units):
    """Write the canonical 7-key PIVlab-style primary .npz (pixel units)."""
    if in_world_units:
        x_px = ((x - cal.x_offset_m) / cal.m_per_pixel) * cal.x_sign
        y_px = ((y - cal.y_offset_m) / cal.m_per_pixel) * cal.y_sign
        uvs = cal.m_per_second_per_px_per_frame
        u = u_orig / uvs * cal.x_sign
        v = v_orig / uvs * cal.y_sign
    else:
        x_px, y_px, u, v = x, y, u_orig, v_orig
    import numpy as _np
    _np.savez(
        out_path,
        calxy=_np.float64(cal.m_per_pixel),
        calu=_np.float64(cal.m_per_second_per_px_per_frame * cal.x_sign),
        calv=_np.float64(cal.m_per_second_per_px_per_frame * cal.y_sign),
        x=x_px.astype(_np.float32),
        y=y_px.astype(_np.float32),
        u_original=u.astype(_np.float32),
        v_original=v.astype(_np.float32),
    )


def run_piv_sequence(file_list, pairs, piv_s, preproc_s, bg, mask, calibration,
                     progress_cb=None, cancel_check=None):
    """Run PIV over the given index pairs; return stacked arrays or None."""
    import numpy as _np
    us, vs, us_o, vs_o = [], [], [], []
    x_grid = y_grid = None
    for k, (i, j) in enumerate(pairs):
        if cancel_check is not None and cancel_check():
            break
        fa = ps._load_image(str(file_list[i]))
        fb = ps._load_image(str(file_list[j]))
        x, y, u, v, u_o, v_o = ps.run_piv(
            fa, fb, piv_s, preproc_s, bg, calibration, mask, return_originals=True)
        if x_grid is None:
            x_grid, y_grid = x, y
        us.append(u); vs.append(v); us_o.append(u_o); vs_o.append(v_o)
        if progress_cb is not None:
            progress_cb(k + 1, len(pairs), i, j)
    if not us:
        return None
    return {
        "x": x_grid, "y": y_grid,
        "u": _np.stack(us, 0), "v": _np.stack(vs, 0),
        "u_original": _np.stack(us_o, 0), "v_original": _np.stack(vs_o, 0),
        "pairs": pairs[:len(us)],
    }


# ─────────────────────────── small UI helpers ────────────────────────────────

def labeled_entry(parent, label, var, width=10):
    row = ttk.Frame(parent, style="Panel.TFrame")
    ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
    ttk.Entry(row, textvariable=var, width=width).pack(side="right", padx=(4, 0))
    return row


def labeled_combo(parent, label, var, values, width=12):
    row = ttk.Frame(parent, style="Panel.TFrame")
    ttk.Label(row, text=label, style="Panel.TLabel").pack(side="left")
    ttk.Combobox(row, textvariable=var, values=values,
                 width=width, state="readonly").pack(side="right", padx=(4, 0))
    return row


# ────────────────────────────── main class ───────────────────────────────────

class PIVGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("PIV Simple")
        root.geometry("1200x820")
        self._init_state()
        self._configure_style()
        self._build_menu()
        self._build_layout()
        self._update_calibration_readout()
        for var in (self.cal_ref_length_px, self.cal_real_distance_mm,
                    self.cal_time_step_ms, self.cal_x_offset_m,
                    self.cal_y_offset_m, self.cal_x_towards, self.cal_y_towards):
            var.trace_add("write", lambda *a: self._update_calibration_readout())

    # ── State ────────────────────────────────────────────────────────────────

    def _init_state(self):
        # Folder + files
        self.image_folder = tk.StringVar()
        self.bg_path = tk.StringVar()
        self.mask_path = tk.StringVar()
        self.calib_image_path = tk.StringVar()
        self.pair_mode = tk.StringVar(value="A-B,B-C")
        self.current_index = tk.IntVar(value=0)
        self.file_list: List[Path] = []

        # PIV settings vars (defaults match the panel screenshot exactly)
        self.algorithm = tk.StringVar(value="fft_windef")
        self.pass1_ia = tk.IntVar(value=64);  self.pass1_step = tk.IntVar(value=32)
        self.pass2_enabled = tk.BooleanVar(value=True)
        self.pass2_ia = tk.IntVar(value=32);  self.pass2_step = tk.IntVar(value=16)
        self.pass3_enabled = tk.BooleanVar(value=True)
        self.pass3_ia = tk.IntVar(value=16);  self.pass3_step = tk.IntVar(value=8)
        self.pass4_enabled = tk.BooleanVar(value=False)
        self.pass4_ia = tk.IntVar(value=32);  self.pass4_step = tk.IntVar(value=16)
        self.repeat_last = tk.BooleanVar(value=False)
        self.quality_slope = tk.DoubleVar(value=0.025)
        self.subpixel = tk.StringVar(value="gauss2x3")
        self.disable_autocorr = tk.BooleanVar(value=False)
        self.robustness = tk.StringVar(value="standard")
        # Velocity cap (vector validation) — ON by default, matching
        # piv_simple's PIVSettings default. Rejects a vector when EITHER
        # component exceeds the cap (|u| OR |v|). Preset = absolute cap of
        # 5.4 px/frame (the validated global cut); the 0.70% percentile is the
        # alternative used when px=0.
        self.cap_enabled = tk.BooleanVar(value=True)
        self.cap_px = tk.DoubleVar(value=5.4)           # absolute |u|/|v| cap, px/frame
        self.cap_percentile = tk.DoubleVar(value=0.70)  # alt: drop top X% of each |u|/|v| (use when px=0)

        # PIVlab-style robust smoothn on the final field (independent of the cap)
        self.smoothn_enabled = tk.BooleanVar(value=False)

        # Batch (multi-condition) vars
        self.batch_parent = tk.StringVar(value="")
        self.batch_out = tk.StringVar(value="")
        self.batch_masking = tk.BooleanVar(value=True)
        self.cfg_fps = tk.StringVar(value="")
        self.cfg_AA = tk.StringVar(value="")
        self.cfg_R = tk.StringVar(value="")
        self.cfg_span = tk.StringVar(value="")
        self.cfg_centre_x = tk.StringVar(value="")
        self.cfg_centre_y = tk.StringVar(value="")

        # Pre-processing vars
        self.pp_clahe = tk.BooleanVar(value=True)
        self.pp_clahe_window = tk.IntVar(value=64)
        self.pp_highpass = tk.BooleanVar(value=False)
        self.pp_highpass_kernel = tk.IntVar(value=15)
        self.pp_intensity_cap = tk.BooleanVar(value=False)
        self.pp_wiener = tk.BooleanVar(value=False)
        self.pp_wiener_window = tk.IntVar(value=3)
        self.pp_contrast = tk.BooleanVar(value=True)
        self.pp_auto_minmax = tk.BooleanVar(value=True)
        self.pp_contrast_min = tk.DoubleVar(value=0.106386)
        self.pp_contrast_max = tk.DoubleVar(value=0.999771)
        self.pp_subtract_mean = tk.BooleanVar(value=False)

        # Calibration vars (defaults from the panel screenshot)
        self.cal_ref_length_px = tk.DoubleVar(value=440.0)
        self.cal_real_distance_mm = tk.DoubleVar(value=70.0)
        self.cal_time_step_ms = tk.DoubleVar(value=2.0)
        self.cal_x_towards = tk.StringVar(value="right")
        self.cal_y_towards = tk.StringVar(value="bottom")
        self.cal_x_offset_m = tk.DoubleVar(value=0.0)
        self.cal_y_offset_m = tk.DoubleVar(value=0.0)
        self.cal_optimize_display = tk.BooleanVar(value=True)
        self.cal_applied = tk.BooleanVar(value=False)
        self.cal_readout = tk.StringVar(value="")

        # Results
        self.results_x: Optional[np.ndarray] = None
        self.results_y: Optional[np.ndarray] = None
        self.results_u: Optional[np.ndarray] = None  # (n_pairs, n_rows, n_cols)
        self.results_v: Optional[np.ndarray] = None
        self.results_u_original: Optional[np.ndarray] = None  # pre-outlier (PIVlab u_original)
        self.results_v_original: Optional[np.ndarray] = None
        self.results_pairs: List[Tuple[int, int]] = []
        self.results_in_world_units = False

        # Worker
        self.cancel_flag = threading.Event()
        self.is_running = False
        self.last_preview_pil: Optional[Image.Image] = None

    # ── Style ────────────────────────────────────────────────────────────────

    def _configure_style(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(background=PANEL_BG)
        s.configure(".", background=PANEL_BG)
        s.configure("TFrame", background=PANEL_BG)
        s.configure("Panel.TFrame", background=PANEL_BG)
        s.configure("TLabel", background=PANEL_BG)
        s.configure("Panel.TLabel", background=PANEL_BG)
        s.configure("TLabelframe", background=PANEL_BG, borderwidth=1, relief="groove")
        s.configure("TLabelframe.Label", background=PANEL_BG,
                    foreground=HEADER_FG, font=HEADER_FONT)
        s.configure("Panel.TLabelframe", background=PANEL_BG,
                    borderwidth=1, relief="groove")
        s.configure("Panel.TLabelframe.Label", background=PANEL_BG,
                    foreground=HEADER_FG, font=HEADER_FONT)
        s.configure("TCheckbutton", background=PANEL_BG)
        s.configure("TRadiobutton", background=PANEL_BG)
        s.configure("Title.TLabel", background=PANEL_BG,
                    foreground=HEADER_FG, font=("Segoe UI", 10, "bold"))

    # ── Menubar ──────────────────────────────────────────────────────────────

    def _build_menu(self):
        mb = tk.Menu(self.root)

        file_m = tk.Menu(mb, tearoff=0)
        file_m.add_command(label="Open image folder…", command=self._browse_image_folder)
        file_m.add_command(label="Open background image…", command=self._browse_background)
        file_m.add_command(label="Open mask image…", command=self._browse_mask)
        file_m.add_command(label="Open calibration image…", command=self._browse_calib_image)
        file_m.add_separator()
        file_m.add_command(label="Save results (.npz)…", command=self._save_results)
        file_m.add_separator()
        file_m.add_command(label="Exit", command=self.root.destroy)
        mb.add_cascade(label="File", menu=file_m)

        acq_m = tk.Menu(mb, tearoff=0)
        acq_m.add_command(label="Live camera (not supported)", state="disabled")
        mb.add_cascade(label="Image acquisition", menu=acq_m)

        img_m = tk.Menu(mb, tearoff=0)
        img_m.add_command(label="Apply pre-processing preview",
                          command=self._preview_preprocessed, accelerator="Ctrl+P")
        mb.add_cascade(label="Image settings", menu=img_m)

        an_m = tk.Menu(mb, tearoff=0)
        an_m.add_command(label="Analyze image sequence",
                         command=self._run_analyze, accelerator="Ctrl+Return")
        an_m.add_command(label="Stop", command=self._cancel_run)
        mb.add_cascade(label="Analysis", menu=an_m)

        help_m = tk.Menu(mb, tearoff=0)
        help_m.add_command(label="About…", command=self._show_about)
        mb.add_cascade(label="Help", menu=help_m)

        self.root.config(menu=mb)
        self.root.bind("<Control-Return>", lambda e: self._run_analyze())
        self.root.bind("<Control-p>",     lambda e: self._preview_preprocessed())

    # ── Layout: image canvas (left) + sidebar (right) + log (bottom) ─────────

    def _build_layout(self):
        # Pack bottom-anchored widgets FIRST so the window manager reserves
        # their vertical slabs before the expanding `top` area grabs the rest.
        # If we packed `top` first with expand=True, it would eat all the
        # vertical space and the log + status would be pushed off-screen
        # when the window shrinks.

        # Status bar — bottom of root, always visible.
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(self.root, textvariable=self.status_var,
                           anchor="w", relief="sunken", padding=(8, 2))
        status.pack(side="bottom", fill="x")

        # Vertical paned window so the user can drag the divider between
        # canvas + log.  The default sash position gives the log ~140 px.
        paned = ttk.PanedWindow(self.root, orient="vertical")
        paned.pack(side="top", fill="both", expand=True)

        # Top section: canvas (left) + sidebar (right)
        top = ttk.Frame(paned)
        paned.add(top, weight=4)
        canvas_area = ttk.Frame(top)
        canvas_area.pack(side="left", fill="both", expand=True)
        self._build_canvas_area(canvas_area)
        self._build_sidebar(top)

        # Bottom section: log (resizable via the sash)
        log_frame = ttk.LabelFrame(paned, text="Log")
        paned.add(log_frame, weight=1)
        self.log = scrolledtext.ScrolledText(log_frame, height=6,
                                             font=("Consolas", 9), bg="#1e1e1e",
                                             fg="#e0e0e0", insertbackground="#e0e0e0")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

        # Keep the window from being shrunk to where things disappear.
        self.root.minsize(960, 640)

    def _build_canvas_area(self, parent):
        # Toolbar: frame slider + pair info
        bar = ttk.Frame(parent)
        bar.pack(fill="x", padx=6, pady=4)
        ttk.Label(bar, text="Frame:", style="Panel.TLabel").pack(side="left")
        self.frame_slider = ttk.Scale(bar, orient="horizontal", from_=0, to=0,
                                      variable=self.current_index,
                                      command=self._on_slider_change)
        self.frame_slider.pack(side="left", fill="x", expand=True, padx=6)
        self.pair_info_var = tk.StringVar(value="—")
        ttk.Label(bar, textvariable=self.pair_info_var,
                  style="Panel.TLabel", width=24).pack(side="left", padx=6)

        # Run controls
        run_bar = ttk.Frame(parent)
        run_bar.pack(fill="x", padx=6, pady=(0, 4))
        self.run_btn = ttk.Button(run_bar, text="▶  Analyze image sequence",
                                  command=self._run_analyze, width=28)
        self.run_btn.pack(side="left", padx=2)
        ttk.Button(run_bar, text="Stop", command=self._cancel_run, width=8).pack(
            side="left", padx=2)
        self.progress = ttk.Progressbar(run_bar, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=6)
        self.progress_pct_var = tk.StringVar(value="")
        ttk.Label(run_bar, textvariable=self.progress_pct_var,
                  width=6, anchor="e").pack(side="left", padx=(0, 6))

        # Time / ETA row underneath progress
        time_row = ttk.Frame(parent)
        time_row.pack(fill="x", padx=6, pady=(0, 4))
        self.progress_time_var = tk.StringVar(value="")
        ttk.Label(time_row, textvariable=self.progress_time_var,
                  foreground="#555").pack(side="left", padx=4)

        # Image canvas
        canvas_frame = ttk.LabelFrame(parent, text="Preview")
        canvas_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self.preview_label = tk.Label(canvas_frame, background=CANVAS_BG)
        self.preview_label.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_sidebar(self, parent):
        sidebar_outer = tk.Frame(parent, bg=PANEL_BG, width=360)
        sidebar_outer.pack(side="right", fill="y")
        sidebar_outer.pack_propagate(False)

        canvas = tk.Canvas(sidebar_outer, bg=PANEL_BG, highlightthickness=0,
                           width=360)
        sb = ttk.Scrollbar(sidebar_outer, orient="vertical",
                           command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        inner = ttk.Frame(canvas, style="Panel.TFrame")

        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        def _on_inner_resize(_e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        def _on_canvas_resize(e):
            canvas.itemconfigure(win_id, width=e.width)
        inner.bind("<Configure>", _on_inner_resize)
        canvas.bind("<Configure>", _on_canvas_resize)
        # Mousewheel only when pointer is in the sidebar
        def _bind_wheel(_e):
            canvas.bind_all("<MouseWheel>",
                            lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units"))
        def _unbind_wheel(_e):
            canvas.unbind_all("<MouseWheel>")
        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        canvas.pack(side="left", fill="y", expand=True)
        sb.pack(side="right", fill="y")

        # Build the four sections inside `inner`.
        self._build_files_section(inner)
        self._build_piv_section(inner)
        self._build_preproc_section(inner)
        self._build_calibration_section(inner)

    # ── Files section (folder picker, background, calibration image) ─────────

    def _build_files_section(self, parent):
        f = ttk.LabelFrame(parent, text="Files")
        f.pack(fill="x", padx=6, pady=6)

        # Image folder
        row = ttk.Frame(f, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Image folder:",
                  style="Panel.TLabel").pack(anchor="w")
        sub = ttk.Frame(f, style="Panel.TFrame")
        sub.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Entry(sub, textvariable=self.image_folder).pack(
            side="left", fill="x", expand=True)
        ttk.Button(sub, text="Browse…", width=10,
                   command=self._browse_image_folder).pack(side="left", padx=4)

        # Pair mode dropdown
        labeled_combo(f, "Pair mode:", self.pair_mode,
                      values=("A-B,B-C", "A-B,A-B"),
                      width=10).pack(fill="x", padx=4, pady=2)

        # Background
        ttk.Label(f, text="Background (optional):",
                  style="Panel.TLabel").pack(anchor="w", padx=4, pady=(6, 0))
        sub = ttk.Frame(f, style="Panel.TFrame")
        sub.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Entry(sub, textvariable=self.bg_path).pack(
            side="left", fill="x", expand=True)
        ttk.Button(sub, text="Browse…", width=10,
                   command=self._browse_background).pack(side="left", padx=4)

        # Mask (PIVlab convention: bright = EXCLUDED, dark = analysed)
        ttk.Label(f, text="Mask (optional, .tif):",
                  style="Panel.TLabel").pack(anchor="w", padx=4, pady=(6, 0))
        sub = ttk.Frame(f, style="Panel.TFrame")
        sub.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Entry(sub, textvariable=self.mask_path).pack(
            side="left", fill="x", expand=True)
        ttk.Button(sub, text="Browse…", width=10,
                   command=self._browse_mask).pack(side="left", padx=4)
        ttk.Button(f, text="View mask",
                   command=self._view_mask).pack(fill="x", padx=4, pady=(0, 4))

        # File count summary
        self.folder_summary_var = tk.StringVar(value="(no folder loaded)")
        ttk.Label(f, textvariable=self.folder_summary_var,
                  style="Panel.TLabel", foreground="#555").pack(
            anchor="w", padx=4, pady=(2, 4))

    # ── PIV settings panel ───────────────────────────────────────────────────

    def _build_piv_section(self, parent):
        outer = ttk.LabelFrame(parent, text="PIV settings (CTRL+S)")
        outer.pack(fill="x", padx=6, pady=6)

        # Algorithm
        algo = ttk.LabelFrame(outer, text="PIV algorithm")
        algo.pack(fill="x", padx=4, pady=4)
        for label, value, state in (
            ("FFT window deformation", "fft_windef", "normal"),
            ("Ensemble correlation",   "ensemble",   "disabled"),
            ("DCC (deprecated)",       "dcc",        "normal"),
        ):
            ttk.Radiobutton(algo, text=label, variable=self.algorithm,
                            value=value, state=state).pack(anchor="w", padx=6)

        # Pass 1
        p1 = ttk.LabelFrame(outer, text="Pass 1")
        p1.pack(fill="x", padx=4, pady=4)
        row = ttk.Frame(p1, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Interrogation area [px]",
                  style="Panel.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.pass1_ia, width=6).pack(side="right")
        row = ttk.Frame(p1, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="Step [px]", style="Panel.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.pass1_step, width=6).pack(side="right")

        # Passes 2…4
        pn = ttk.LabelFrame(outer, text="Pass 2…4")
        pn.pack(fill="x", padx=4, pady=4)
        for label, enabled, ia, step in (
            ("Pass 2", self.pass2_enabled, self.pass2_ia, self.pass2_step),
            ("Pass 3", self.pass3_enabled, self.pass3_ia, self.pass3_step),
            ("Pass 4", self.pass4_enabled, self.pass4_ia, self.pass4_step),
        ):
            row = ttk.Frame(pn, style="Panel.TFrame")
            row.pack(fill="x", padx=4, pady=1)
            ttk.Checkbutton(row, text=label, variable=enabled,
                            width=8).pack(side="left")
            ttk.Entry(row, textvariable=ia, width=6).pack(side="left", padx=(2, 2))
            ttk.Entry(row, textvariable=step, width=6).pack(side="left", padx=(2, 2))
        row = ttk.Frame(pn, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=(4, 2))
        ttk.Checkbutton(row, text="Repeat last pass until quality slope <",
                        variable=self.repeat_last).pack(side="left")
        ttk.Entry(row, textvariable=self.quality_slope, width=6).pack(side="left", padx=4)

        # Sub-pixel
        sp = ttk.LabelFrame(outer, text="Sub-pixel estimator")
        sp.pack(fill="x", padx=4, pady=4)
        ttk.Combobox(sp, textvariable=self.subpixel,
                     values=("gauss2x3", "centroid", "parabolic"),
                     state="readonly", width=18).pack(fill="x", padx=6, pady=4)

        ttk.Checkbutton(outer, text="Disable auto-correlation",
                        variable=self.disable_autocorr).pack(
            anchor="w", padx=8, pady=(2, 2))

        # Robustness
        rb = ttk.LabelFrame(outer, text="Correlation robustness")
        rb.pack(fill="x", padx=4, pady=4)
        ttk.Combobox(rb, textvariable=self.robustness,
                     values=("standard", "extreme"),
                     state="readonly", width=18).pack(fill="x", padx=6, pady=4)

        # Vector validation — velocity cap (drops outlier vectors above a
        # displacement limit; mimics PIVlab's search-area-bounded output).
        vc = ttk.LabelFrame(outer, text="Vector validation (velocity cap)")
        vc.pack(fill="x", padx=4, pady=4)
        ttk.Checkbutton(vc, text="Reject vectors above |u| or |v| cap",
                        variable=self.cap_enabled).pack(anchor="w", padx=6, pady=(4, 2))
        row = ttk.Frame(vc, style="Panel.TFrame"); row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text="max |u|/|v| px/frame (validated 5.4):").pack(side="left")
        ttk.Entry(row, textvariable=self.cap_px, width=6).pack(side="left", padx=4)
        row = ttk.Frame(vc, style="Panel.TFrame"); row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text="or top % of each |u|/|v| (used when px=0):").pack(side="left")
        ttk.Entry(row, textvariable=self.cap_percentile, width=6).pack(side="left", padx=4)

        # Field smoothing — PIVlab-style robust smoothn on the final field
        # (independent of the cap; toggle either to compare their effect).
        sm = ttk.LabelFrame(outer, text="Field smoothing (smoothn)")
        sm.pack(fill="x", padx=4, pady=4)
        ttk.Checkbutton(sm, text="Per-pass smoothn (PIVlab-style: s=4, auto last pass)",
                        variable=self.smoothn_enabled).pack(anchor="w", padx=6, pady=4)

        # Batch (multi-condition pipeline)
        bf = ttk.LabelFrame(outer, text="Batch (multi-condition pipeline)")
        bf.pack(fill="x", padx=4, pady=4)
        row = ttk.Frame(bf, style="Panel.TFrame"); row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text="Parent folder:").pack(side="left")
        ttk.Entry(row, textvariable=self.batch_parent).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse", command=self._browse_batch_parent).pack(side="left")
        row = ttk.Frame(bf, style="Panel.TFrame"); row.pack(fill="x", padx=6, pady=2)
        ttk.Label(row, text="Output root:").pack(side="left")
        ttk.Entry(row, textvariable=self.batch_out).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row, text="Browse", command=self._browse_batch_out).pack(side="left")
        ttk.Checkbutton(bf, text="Dynamic masking (Stage 2) — requires TIF frames",
                        variable=self.batch_masking).pack(anchor="w", padx=6, pady=2)
        ov = ttk.Frame(bf, style="Panel.TFrame"); ov.pack(fill="x", padx=6, pady=2)
        for lbl, var in (("fps", self.cfg_fps), ("AA", self.cfg_AA), ("R", self.cfg_R),
                         ("span", self.cfg_span), ("cx", self.cfg_centre_x), ("cy", self.cfg_centre_y)):
            ttk.Label(ov, text=lbl).pack(side="left")
            ttk.Entry(ov, textvariable=var, width=6).pack(side="left", padx=(2, 8))
        ttk.Label(bf, text="(override fields blank = use config.yaml)").pack(anchor="w", padx=6)
        ttk.Button(bf, text="Run batch pipeline", command=self._run_batch).pack(anchor="w", padx=6, pady=4)
        ttk.Button(bf, text="Run PIV + pipeline (this folder)", command=self._run_single_pipeline).pack(anchor="w", padx=6, pady=4)

    # ── Pre-processing panel ─────────────────────────────────────────────────

    def _build_preproc_section(self, parent):
        outer = ttk.LabelFrame(parent, text="Image pre-processing (CTRL+I)")
        outer.pack(fill="x", padx=6, pady=6)

        # CLAHE
        row = ttk.Frame(outer, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(row, text="Enable CLAHE",
                        variable=self.pp_clahe).pack(side="left")
        ttk.Entry(row, textvariable=self.pp_clahe_window,
                  width=6).pack(side="right")
        ttk.Label(row, text="Window size [px]",
                  style="Panel.TLabel").pack(side="right", padx=4)

        # Highpass
        row = ttk.Frame(outer, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(row, text="Enable highpass",
                        variable=self.pp_highpass).pack(side="left")
        ttk.Entry(row, textvariable=self.pp_highpass_kernel,
                  width=6).pack(side="right")
        ttk.Label(row, text="Kernel size [px]",
                  style="Panel.TLabel").pack(side="right", padx=4)

        # Intensity capping
        ttk.Checkbutton(outer, text="Enable intensity capping",
                        variable=self.pp_intensity_cap).pack(anchor="w",
                                                              padx=6, pady=2)

        # Wiener2
        row = ttk.Frame(outer, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Checkbutton(row, text="Wiener2 denoise and low pass",
                        variable=self.pp_wiener).pack(side="left")
        ttk.Entry(row, textvariable=self.pp_wiener_window,
                  width=6).pack(side="right")

        # Auto contrast stretch
        cs = ttk.LabelFrame(outer, text="Auto contrast stretch")
        cs.pack(fill="x", padx=4, pady=4)
        ttk.Checkbutton(cs, text="Enable",
                        variable=self.pp_contrast).pack(anchor="w", padx=6)
        ttk.Checkbutton(cs, text="Auto-compute min/max from percentiles",
                        variable=self.pp_auto_minmax).pack(anchor="w", padx=6)
        row = ttk.Frame(cs, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="minimum:",
                  style="Panel.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.pp_contrast_min,
                  width=10).pack(side="left", padx=4)
        row = ttk.Frame(cs, style="Panel.TFrame")
        row.pack(fill="x", padx=4, pady=2)
        ttk.Label(row, text="maximum:",
                  style="Panel.TLabel").pack(side="left")
        ttk.Entry(row, textvariable=self.pp_contrast_max,
                  width=10).pack(side="left", padx=4)

        # Background Subtraction
        bg = ttk.LabelFrame(outer, text="Background subtraction")
        bg.pack(fill="x", padx=4, pady=4)
        ttk.Checkbutton(bg, text="Subtract mean intensity",
                        variable=self.pp_subtract_mean).pack(anchor="w",
                                                              padx=6, pady=2)
        ttk.Button(bg, text="View background image",
                   command=self._view_background).pack(fill="x",
                                                        padx=6, pady=4)

        # Action buttons
        ttk.Button(outer, text="Apply and preview current frame",
                   command=self._preview_preprocessed).pack(fill="x",
                                                             padx=6, pady=(8, 2))
        ttk.Button(outer, text="Export preview",
                   command=self._export_preview).pack(fill="x", padx=6, pady=2)

    # ── Calibration panel ────────────────────────────────────────────────────

    def _build_calibration_section(self, parent):
        outer = ttk.LabelFrame(parent, text="Calibration (CTRL+Z)")
        outer.pack(fill="x", padx=6, pady=6)

        ttk.Button(outer, text="Load calibration image (optional)",
                   command=self._browse_calib_image).pack(fill="x",
                                                          padx=6, pady=4)
        ttk.Checkbutton(outer, text="Optimize display",
                        variable=self.cal_optimize_display).pack(
            anchor="w", padx=6)

        # Setup scaling
        sc = ttk.LabelFrame(outer, text="Setup scaling")
        sc.pack(fill="x", padx=4, pady=4)
        ttk.Button(sc, text="Select reference length [px]",
                   command=self._select_reference_length).pack(fill="x",
                                                                padx=6, pady=4)
        labeled_entry(sc, "Reference length [px]",
                      self.cal_ref_length_px).pack(fill="x", padx=6, pady=2)
        labeled_entry(sc, "Real distance [mm]",
                      self.cal_real_distance_mm).pack(fill="x", padx=6, pady=2)
        labeled_entry(sc, "Time step [ms]",
                      self.cal_time_step_ms).pack(fill="x", padx=6, pady=2)

        # Setup offsets
        of = ttk.LabelFrame(outer, text="Setup offsets")
        of.pack(fill="x", padx=4, pady=4)
        labeled_combo(of, "x increases towards the", self.cal_x_towards,
                      values=("right", "left"),
                      width=8).pack(fill="x", padx=6, pady=2)
        labeled_combo(of, "y increases towards the", self.cal_y_towards,
                      values=("bottom", "top"),
                      width=8).pack(fill="x", padx=6, pady=2)
        ttk.Button(of, text="Set x & y offsets",
                   command=self._set_xy_offsets).pack(fill="x", padx=6, pady=4)

        # Green readout
        readout = tk.Frame(outer, bg=GREEN_BG, borderwidth=2, relief="ridge")
        readout.pack(fill="x", padx=4, pady=8)
        tk.Label(readout, textvariable=self.cal_readout, bg=GREEN_BG,
                 justify="left", font=("Consolas", 10)).pack(padx=8, pady=8,
                                                              anchor="w")

        # Apply / Clear
        ttk.Button(outer, text="Apply calibration",
                   command=self._apply_calibration).pack(fill="x", padx=6, pady=2)
        ttk.Button(outer, text="Clear calibration",
                   command=self._clear_calibration).pack(fill="x", padx=6,
                                                          pady=(2, 6))

    # ───────────────────────── browse handlers ───────────────────────────────

    def _browse_image_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        self.image_folder.set(folder)
        files = list_image_folder(folder)
        self.file_list = files
        n = len(files)
        if n == 0:
            self.folder_summary_var.set("(folder is empty)")
            self._log(f"Folder loaded: {folder} — no supported images found.")
            self.frame_slider.configure(to=0)
            return
        ext_counts: dict[str, int] = {}
        for p in files:
            ext_counts[p.suffix.lower()] = ext_counts.get(p.suffix.lower(), 0) + 1
        summary = ", ".join(f"{c}× {e}" for e, c in sorted(ext_counts.items()))
        self.folder_summary_var.set(f"{n} files  ({summary})")
        self.current_index.set(0)
        self.frame_slider.configure(to=n - 1)
        self.pair_info_var.set(f"Frame 1 / {n}  ·  {files[0].name}")
        self._show_preview_path(files[0])
        self._log(f"Folder loaded: {folder}  ({summary})")

    def _browse_background(self):
        p = filedialog.askopenfilename(title="Select background image",
                                       filetypes=TIFF_FIRST_FILETYPES)
        if p:
            self.bg_path.set(p)
            self._log(f"Background: {p}")

    def _browse_mask(self):
        p = filedialog.askopenfilename(title="Select mask image (PIVlab-style)",
                                       filetypes=TIFF_FIRST_FILETYPES)
        if p:
            self.mask_path.set(p)
            try:
                m = ps.load_mask(p)
                self._log(f"Mask: {p}  shape={m.shape}  "
                          f"keep fraction={float(m.mean()):.3f}")
            except Exception as e:
                self._log(f"Mask load failed: {e}")

    def _browse_calib_image(self):
        p = filedialog.askopenfilename(title="Select calibration image",
                                       filetypes=TIFF_FIRST_FILETYPES)
        if p:
            self.calib_image_path.set(p)
            self._show_preview_path(p)
            self._log(f"Calibration image: {p}")

    def _browse_batch_parent(self):
        d = filedialog.askdirectory(title="Parent folder of condition subfolders")
        if d:
            self.batch_parent.set(d)
            if not self.batch_out.get():
                self.batch_out.set(d)

    def _browse_batch_out(self):
        d = filedialog.askdirectory(title="Output root")
        if d:
            self.batch_out.set(d)

    # ─────────────────────── settings → dataclasses ──────────────────────────

    def _piv_settings(self) -> ps.PIVSettings:
        windows = [self.pass1_ia.get()]
        steps   = [self.pass1_step.get()]
        for en, ia, st in ((self.pass2_enabled, self.pass2_ia, self.pass2_step),
                           (self.pass3_enabled, self.pass3_ia, self.pass3_step),
                           (self.pass4_enabled, self.pass4_ia, self.pass4_step)):
            if en.get():
                windows.append(ia.get()); steps.append(st.get())
        # Velocity cap: absolute px wins if > 0; else top-% by speed;
        # disabled entirely if the checkbox is off.
        if self.cap_enabled.get():
            cap_px_val = self.cap_px.get()
            velocity_cap_px = float(cap_px_val) if cap_px_val and cap_px_val > 0 else None
            velocity_cap_percentile = float(self.cap_percentile.get())
        else:
            velocity_cap_px = None
            velocity_cap_percentile = None

        return ps.PIVSettings(
            algorithm=self.algorithm.get(),
            window_sizes=tuple(windows),
            steps=tuple(steps),
            repeat_last_pass=self.repeat_last.get(),
            quality_slope_threshold=self.quality_slope.get(),
            subpixel_method=self.subpixel.get(),
            disable_autocorrelation=self.disable_autocorr.get(),
            correlation_robustness=self.robustness.get(),
            velocity_cap_px=velocity_cap_px,
            velocity_cap_fraction=None,
            velocity_cap_percentile=velocity_cap_percentile,
            enable_smoothn=self.smoothn_enabled.get(),
        )

    def _preproc_settings(self) -> ps.PreprocSettings:
        return ps.PreprocSettings(
            enable_clahe=self.pp_clahe.get(),
            clahe_window_size=self.pp_clahe_window.get(),
            enable_highpass=self.pp_highpass.get(),
            highpass_kernel_size=self.pp_highpass_kernel.get(),
            enable_intensity_capping=self.pp_intensity_cap.get(),
            enable_wiener2=self.pp_wiener.get(),
            wiener_window_size=self.pp_wiener_window.get(),
            enable_contrast_stretch=self.pp_contrast.get(),
            auto_minmax=self.pp_auto_minmax.get(),
            contrast_min=self.pp_contrast_min.get(),
            contrast_max=self.pp_contrast_max.get(),
            subtract_mean_intensity=self.pp_subtract_mean.get(),
        )

    def _calibration_settings(self) -> ps.CalibrationSettings:
        return ps.CalibrationSettings(
            reference_length_px=self.cal_ref_length_px.get(),
            real_distance_mm=self.cal_real_distance_mm.get(),
            time_step_ms=self.cal_time_step_ms.get(),
            x_increases_towards=self.cal_x_towards.get(),
            y_increases_towards=self.cal_y_towards.get(),
            x_offset_m=self.cal_x_offset_m.get(),
            y_offset_m=self.cal_y_offset_m.get(),
            optimize_display=self.cal_optimize_display.get(),
            calibration_image=self.calib_image_path.get() or None,
        )

    # ────────────────────── calibration interactions ─────────────────────────

    def _picking_source_path(self) -> Optional[str]:
        """Image path for click-to-pick: prefers calibration image, else
        the currently-selected frame.
        """
        path = self.calib_image_path.get() or (
            str(self.file_list[self.current_index.get()]) if self.file_list else "")
        if not path:
            messagebox.showwarning("PIV",
                "Load a calibration image or an image folder first.")
            return None
        return path

    def _pick_points_dialog(self, image_path: str, n_points: int,
                            instruction: str,
                            max_display: int = 1100) -> Optional[List[Tuple[int, int]]]:
        """Modal Toplevel that shows an image and captures `n_points` clicks.

        Returns points in ORIGINAL (full-resolution) pixel coordinates, or
        None on cancel. Uses pure Tkinter — no cv2 GUI dependency.
        """
        try:
            full = load_image_full_uint8(image_path)
        except Exception as e:
            messagebox.showerror("PIV", f"Could not load image:\n{e}")
            return None

        h, w = full.shape
        scale = min(max_display / w, max_display / h, 1.0)
        disp_w, disp_h = int(round(w * scale)), int(round(h * scale))
        if scale < 1.0:
            disp_img = cv2.resize(full, (disp_w, disp_h),
                                  interpolation=cv2.INTER_AREA)
        else:
            disp_img = full

        pil = Image.fromarray(disp_img, mode="L")
        photo = ImageTk.PhotoImage(pil)

        top = tk.Toplevel(self.root)
        top.title(instruction)
        top.transient(self.root)

        canvas = tk.Canvas(top, width=disp_w, height=disp_h, bg="black",
                           cursor="crosshair", highlightthickness=0)
        canvas.image = photo  # keep ref alive
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.pack(padx=6, pady=6)

        info = tk.StringVar(value=f"0 / {n_points} points")
        ttk.Label(top, textvariable=info).pack(pady=(0, 2))
        ttk.Label(top, text=instruction).pack(pady=(0, 4))

        state = {"points": [], "result": None}

        def redraw():
            canvas.delete("mark")
            for cx, cy in state["points"]:
                r = 7
                canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                   outline="#00ff66", width=2, tags="mark")
            if n_points == 2 and len(state["points"]) == 2:
                (x1, y1), (x2, y2) = state["points"]
                canvas.create_line(x1, y1, x2, y2,
                                   fill="#ffff00", width=2, tags="mark")
            info.set(f"{len(state['points'])} / {n_points} points")

        def on_click(event):
            if len(state["points"]) >= n_points:
                state["points"] = [(event.x, event.y)]
            else:
                state["points"].append((event.x, event.y))
            redraw()

        def confirm(_=None):
            if len(state["points"]) == n_points:
                state["result"] = [
                    (int(round(x / scale)), int(round(y / scale)))
                    for x, y in state["points"]
                ]
                top.destroy()

        def cancel(_=None):
            state["result"] = None
            top.destroy()

        canvas.bind("<Button-1>", on_click)
        top.bind("<Return>", confirm)
        top.bind("<Escape>", cancel)
        top.protocol("WM_DELETE_WINDOW", cancel)

        bar = ttk.Frame(top)
        bar.pack(pady=(2, 8))
        ttk.Button(bar, text="OK (Enter)", command=confirm).pack(
            side="left", padx=4)
        ttk.Button(bar, text="Cancel (Esc)", command=cancel).pack(
            side="left", padx=4)

        top.grab_set()
        top.focus_set()
        top.wait_window()
        return state["result"]

    def _select_reference_length(self):
        path = self._picking_source_path()
        if path is None:
            return
        pts = self._pick_points_dialog(path, 2,
            "Click two points spanning the known distance, then press Enter")
        if pts is None:
            self._log("Reference length: cancelled."); return
        (x1, y1), (x2, y2) = pts
        dist_px = float(np.hypot(x2 - x1, y2 - y1))
        self.cal_ref_length_px.set(round(dist_px, 2))
        self._log(f"Reference length set: {dist_px:.2f} px "
                  f"({x1},{y1}) → ({x2},{y2})")

    def _set_xy_offsets(self):
        path = self._picking_source_path()
        if path is None:
            return
        pts = self._pick_points_dialog(path, 1,
            "Click the world-origin point, then press Enter")
        if pts is None:
            self._log("Origin: cancelled."); return
        px, py = pts[0]
        cal = self._calibration_settings()
        x_off = -px * cal.m_per_pixel * cal.x_sign
        y_off = -py * cal.m_per_pixel * cal.y_sign
        self.cal_x_offset_m.set(round(x_off, 6))
        self.cal_y_offset_m.set(round(y_off, 6))
        self._log(f"Origin at pixel ({px}, {py}) → "
                  f"x_offset = {x_off:.6f} m, y_offset = {y_off:.6f} m")

    def _apply_calibration(self):
        self.cal_applied.set(True)
        self._update_calibration_readout()
        self._log("Calibration APPLIED — analysis output will be in m / (m/s).")

    def _clear_calibration(self):
        self.cal_applied.set(False)
        self.cal_x_offset_m.set(0.0); self.cal_y_offset_m.set(0.0)
        self._update_calibration_readout()
        self._log("Calibration CLEARED — analysis output will be in px / (px/frame).")

    def _update_calibration_readout(self, *_):
        try:
            cal = self._calibration_settings()
            mpp, mps = cal.m_per_pixel, cal.m_per_second_per_px_per_frame
        except Exception:
            self.cal_readout.set("(enter valid numbers above)"); return
        applied = "  [APPLIED]" if self.cal_applied.get() else "  [not applied]"
        self.cal_readout.set(
            f"  1 px = {mpp:.4e} m\n"
            f"  1 px/frame = {mps:.4f} m/s\n"
            f"  x offset: {cal.x_offset_m:g} m\n"
            f"  y offset: {cal.y_offset_m:g} m{applied}"
        )

    # ──────────────────────── preprocessing preview ──────────────────────────

    def _show_image_toplevel(self, title: str, img_u8: np.ndarray) -> None:
        """Open a non-modal Toplevel showing `img_u8`. Pure Tkinter — no cv2 GUI."""
        if img_u8.ndim == 2:
            pil = Image.fromarray(img_u8, mode="L")
        else:
            pil = Image.fromarray(cv2.cvtColor(img_u8, cv2.COLOR_BGR2RGB))
        pil.thumbnail((1100, 800))
        photo = ImageTk.PhotoImage(pil)
        top = tk.Toplevel(self.root)
        top.title(title)
        lbl = tk.Label(top, image=photo, background="black")
        lbl.image = photo  # keep reference alive
        lbl.pack(padx=4, pady=4)

    def _view_background(self):
        path = self.bg_path.get()
        if not path:
            messagebox.showwarning("PIV", "No background image selected.")
            return
        try:
            img = load_image_uint8(path)
        except Exception as e:
            messagebox.showerror("PIV", str(e)); return
        self._show_image_toplevel("Background", img)
        self._log(f"Background preview: {Path(path).name}")

    def _view_mask(self):
        path = self.mask_path.get()
        if not path:
            messagebox.showwarning("PIV", "No mask image selected.")
            return
        try:
            mk = ps.load_mask(path)
        except Exception as e:
            messagebox.showerror("PIV", str(e)); return
        # Overlay the mask on the current frame (red wash on excluded regions).
        cur = self._current_frame_path()
        if cur is None:
            disp = (mk.astype(np.uint8) * 255)
        else:
            base = load_image_uint8(str(cur))
            mk_resized = cv2.resize(
                mk.astype(np.uint8) * 255,
                (base.shape[1], base.shape[0]),
                interpolation=cv2.INTER_NEAREST)
            disp = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
            disp[mk_resized == 0] = (0.4 * disp[mk_resized == 0]
                                     + 0.6 * np.array([0, 0, 200])).astype(np.uint8)
        self._show_image_toplevel("Mask", disp)
        self._log(f"Mask preview: {Path(path).name}  "
                  f"keep fraction = {float(mk.mean()):.3f}")

    def _current_frame_path(self) -> Optional[Path]:
        if not self.file_list:
            return None
        i = max(0, min(self.current_index.get(), len(self.file_list) - 1))
        return self.file_list[i]

    def _preview_preprocessed(self):
        path = self._current_frame_path()
        if path is None:
            messagebox.showwarning("PIV", "Load an image folder first.")
            return
        try:
            raw = ps._load_image(str(path))
            bg = ps._load_image(self.bg_path.get()) if self.bg_path.get() else None
            raw_g = ps._ensure_gray(raw)
            bg_g = ps._ensure_gray(bg) if bg is not None else None
            out = ps.preprocess(raw_g, self._preproc_settings(), bg_g)
        except Exception:
            self._log("Preview failed:\n" + traceback.format_exc())
            return
        self._show_preview_array((out * 255).astype(np.uint8))
        self._log(f"Preview: {path.name}  range=[{out.min():.3f}, {out.max():.3f}]")

    def _export_preview(self):
        if self.last_preview_pil is None:
            messagebox.showinfo("PIV", "Generate a preview first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save preview", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if out_path:
            self.last_preview_pil.save(out_path)
            self._log(f"Preview saved → {out_path}")

    # ───────────────────────── batch analysis ────────────────────────────────

    def _run_analyze(self):
        if self.is_running:
            return
        if not self.file_list or len(self.file_list) < 2:
            messagebox.showwarning("PIV",
                "Load an image folder with at least 2 images first.")
            return
        pairs = pair_indices(len(self.file_list), self.pair_mode.get())
        if not pairs:
            messagebox.showwarning("PIV", "No valid pairs to process.")
            return
        self.cancel_flag.clear()
        self.is_running = True
        self.run_start_time = time.time()
        self.run_btn.config(state="disabled")
        self.progress.config(maximum=len(pairs), value=0)
        self.progress_pct_var.set("0%")
        self.progress_time_var.set(f"0 / {len(pairs)} pairs  ·  elapsed 0:00  ·  ETA --:--")
        self.status_var.set(f"Running 0 / {len(pairs)}…")
        self._log("─" * 70)
        self._log(f"Starting analysis: {len(pairs)} pairs in mode {self.pair_mode.get()}")
        threading.Thread(target=self._analyze_worker, args=(pairs,),
                         daemon=True).start()

    def _cancel_run(self):
        if self.is_running:
            self.cancel_flag.set()
            self._log("Cancel requested…")

    def _analyze_worker(self, pairs: List[Tuple[int, int]]):
        try:
            piv_s = self._piv_settings()
            preproc_s = self._preproc_settings()
            calibration = self._calibration_settings() if self.cal_applied.get() else None
            bg = ps._load_image(self.bg_path.get()) if self.bg_path.get() else None
            mask = ps.load_mask(self.mask_path.get()) if self.mask_path.get() else None

            self._log(f"PIV passes: windows={piv_s.window_sizes}  "
                      f"steps={piv_s.steps}  algorithm={piv_s.algorithm}")
            if calibration:
                self._log(f"Calibration applied — output in m / (m/s)")
            if mask is not None:
                self._log(f"Mask: keep fraction = {float(mask.mean()):.3f}")

            res = run_piv_sequence(
                self.file_list, pairs, piv_s, preproc_s, bg, mask, calibration,
                progress_cb=lambda d, t, i, j: self.root.after(0, self._update_progress, d, t, i, j),
                cancel_check=self.cancel_flag.is_set)
            if res:
                self.results_x = res["x"]; self.results_y = res["y"]
                self.results_u = res["u"]; self.results_v = res["v"]
                self.results_u_original = res["u_original"]
                self.results_v_original = res["v_original"]
                self.results_pairs = res["pairs"]
                self.results_in_world_units = calibration is not None
                unit = "m/s" if calibration else "px/frame"
                mean_mag = float(np.nanmean(
                    np.hypot(self.results_u, self.results_v)))
                n_valid = int(np.isfinite(self.results_u).sum())
                n_total = self.results_u.size
                self._log(f"Done. {len(self.results_pairs)} pairs processed. "
                          f"Field shape per pair: {self.results_u.shape[1:]}  "
                          f"mean |V| = {mean_mag:.4f} {unit}  "
                          f"valid = {n_valid}/{n_total}")
                self.root.after(0, self._show_pair_overlay, 0)
            if self.cancel_flag.is_set():
                done = len(res["pairs"]) if res else 0
                self._log(f"Cancelled. {done}/{len(pairs)} pairs completed.")
        except Exception:
            self._log("Analysis failed:\n" + traceback.format_exc())
        finally:
            self.root.after(0, self._finish_run)

    def _run_batch(self):
        if self.is_running:
            return
        parent = self.batch_parent.get().strip()
        if not parent or not Path(parent).is_dir():
            messagebox.showwarning("PIV batch", "Pick a valid parent folder.")
            return
        conds, skipped = collect_conditions(parent)
        if not conds:
            messagebox.showwarning("PIV batch", "No valid condition subfolders found.")
            return
        if skipped:
            self._log(f"Skipping (bad name / <2 images): {', '.join(skipped)}")
        self._start_batch(conds)

    def _start_batch(self, conds):
        self.cancel_flag.clear()
        self.is_running = True
        self.run_start_time = time.time()
        self.run_btn.config(state="disabled")
        self.progress.config(maximum=len(conds), value=0)
        self._log("─" * 70)
        self._log(f"Batch: {len(conds)} condition(s)")
        threading.Thread(target=self._batch_worker, args=(conds,), daemon=True).start()

    def _run_single_pipeline(self):
        if self.is_running:
            return
        if not self.file_list or len(self.file_list) < 2:
            messagebox.showwarning("PIV", "Load an image folder with at least 2 images first.")
            return
        folder = self.file_list[0].parent
        name = folder.name
        condition = parse_filename(name)["condition"]
        cond = {"condition": condition, "name": name, "folder": folder, "images": list(self.file_list)}
        if not self.batch_out.get().strip() and not self.batch_parent.get().strip():
            self.batch_out.set(str(folder.parent))
        self._log(f"Single-condition pipeline: {name}")
        self._start_batch([cond])

    def _batch_worker(self, conds):
        from piv_pipeline.piv_pipeline_master import run_pipeline_for_condition, _load_config, _apply_cfg_defaults
        from piv_pipeline.fn_final_analysis import final_analysis
        try:
            cfg = _load_config(None)
            for key, var, cast in (("fps", self.cfg_fps, float), ("AA", self.cfg_AA, int),
                                   ("R", self.cfg_R, float), ("span", self.cfg_span, int),
                                   ("centre_x", self.cfg_centre_x, float),
                                   ("centre_y", self.cfg_centre_y, float)):
                s = var.get().strip()
                if s:
                    cfg[key] = cast(s)
            cfg = _apply_cfg_defaults(cfg)
            out_root = Path(self.batch_out.get().strip() or self.batch_parent.get().strip())

            piv_s = self._piv_settings()
            preproc_s = self._preproc_settings()
            calibration = self._calibration_settings() if self.cal_applied.get() else None
            bg = ps._load_image(self.bg_path.get()) if self.bg_path.get() else None
            mask = ps.load_mask(self.mask_path.get()) if self.mask_path.get() else None

            ok, failed = [], []
            for ci, c in enumerate(conds):
                if self.cancel_flag.is_set():
                    self._log(f"Batch stopped (cancelled). OK={len(ok)} failed={len(failed)}"); break
                self._log(f"[{ci+1}/{len(conds)}] {c['name']}  ({len(c['images'])} frames)")
                try:
                    pairs = pair_indices(len(c["images"]), self.pair_mode.get())
                    res = run_piv_sequence(c["images"], pairs, piv_s, preproc_s,
                                           bg, mask, calibration,
                                           cancel_check=self.cancel_flag.is_set)
                    if not res:
                        failed.append((c["name"], "no pairs")); continue
                    prim = out_root / f"{c['name']}.npz"
                    save_primary_npz(prim, res["x"], res["y"], res["u_original"],
                                     res["v_original"], calibration or self._calibration_settings(),
                                     calibration is not None)
                    run_pipeline_for_condition(
                        prim, images_folder=c["folder"], cfg=cfg, out_dir=out_root,
                        do_masking=self.batch_masking.get(), force=True)
                    ok.append(c["name"])
                except Exception as e:
                    self._log(f"  FAILED {c['name']}: {e}")
                    failed.append((c["name"], str(e)))
                self.root.after(0, lambda v=ci + 1: self.progress.config(value=v))

            if ok and not self.cancel_flag.is_set():
                self._log("Stage 4: final_analysis over all tertiary exports…")
                try:
                    final_analysis(out_root / "Tertiary Exports",
                                   out_root / "Summary", span=int(cfg["span"]))
                except Exception as e:
                    self._log(f"  Stage 4 FAILED: {e}")
            if not self.cancel_flag.is_set():
                self._log(f"Batch done. OK={len(ok)}  failed={len(failed)}")
            for name, why in failed:
                self._log(f"  - {name}: {why}")
        except Exception:
            self._log("Batch failed:\n" + traceback.format_exc())
        finally:
            self.root.after(0, self._finish_run)

    def _update_progress(self, done, total, i, j):
        self.progress.config(value=done)
        pct = 100.0 * done / max(total, 1)
        elapsed = time.time() - getattr(self, "run_start_time", time.time())
        per_pair = elapsed / max(done, 1)
        remaining = per_pair * max(total - done, 0)
        self.progress_pct_var.set(f"{pct:.0f}%")
        self.progress_time_var.set(
            f"{done} / {total} pairs  ·  "
            f"elapsed {_fmt_dur(elapsed)}  ·  "
            f"ETA {_fmt_dur(remaining)}  ·  "
            f"{per_pair:.2f}s/pair"
        )
        self.status_var.set(
            f"Running {done} / {total}  (pair {i}–{j})")

    def _finish_run(self):
        self.is_running = False
        self.run_btn.config(state="normal")
        elapsed = time.time() - getattr(self, "run_start_time", time.time())
        if self.results_u is not None and not self.cancel_flag.is_set():
            self.status_var.set(f"Ready  ·  done in {_fmt_dur(elapsed)}")
            self.progress_pct_var.set("100%")
            self.progress_time_var.set(
                f"Done  ·  total {_fmt_dur(elapsed)}")
        else:
            self.status_var.set("Ready")
            self.progress_time_var.set(
                f"Stopped  ·  elapsed {_fmt_dur(elapsed)}")

    # ──────────────────────── slider + preview ───────────────────────────────

    def _on_slider_change(self, _value):
        # Slider can fire continuously; just update the label + image.
        if not self.file_list:
            return
        i = int(round(self.current_index.get()))
        i = max(0, min(i, len(self.file_list) - 1))
        path = self.file_list[i]
        self.pair_info_var.set(f"Frame {i + 1} / {len(self.file_list)}  ·  {path.name}")
        if self.results_u is not None:
            # If results exist, show vector overlay for the pair starting at i.
            pair_k = self._pair_index_for_frame(i)
            if pair_k is not None:
                self._show_pair_overlay(pair_k); return
        self._show_preview_path(path)

    def _pair_index_for_frame(self, frame_i: int) -> Optional[int]:
        for k, (i, _) in enumerate(self.results_pairs):
            if i == frame_i:
                return k
        return None

    def _show_pair_overlay(self, pair_k: int):
        if self.results_u is None or pair_k >= len(self.results_pairs):
            return
        i, j = self.results_pairs[pair_k]
        path_a = self.file_list[i]
        try:
            base = load_image_uint8(str(path_a))
        except Exception as e:
            self._log(f"Overlay load failed: {e}"); return

        overlay = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        x = self.results_x.copy()
        y = self.results_y.copy()
        u = self.results_u[pair_k].copy()
        v = self.results_v[pair_k].copy()

        # If results are in world units, convert back to display pixels.
        if self.results_in_world_units:
            cal = self._calibration_settings()
            x = ((x - cal.x_offset_m) / cal.m_per_pixel) * cal.x_sign
            y = ((y - cal.y_offset_m) / cal.m_per_pixel) * cal.y_sign
            u = u / cal.m_per_second_per_px_per_frame * cal.x_sign
            v = v / cal.m_per_second_per_px_per_frame * cal.y_sign

        # Account for the display downscaling factor used in load_image_uint8.
        full = cv2.imread(str(path_a), cv2.IMREAD_UNCHANGED)
        if full is None:
            full_h, full_w = base.shape
        else:
            full_h, full_w = full.shape[:2]
        disp_h, disp_w = base.shape
        sx = disp_w / full_w
        sy = disp_h / full_h
        x = x * sx; y = y * sy
        u = u * sx; v = v * sy

        # NaN entries are masked-out windows — skip them in the overlay.
        valid = np.isfinite(u) & np.isfinite(v)
        mag = np.hypot(np.where(valid, u, 0.0), np.where(valid, v, 0.0))
        max_mag = max(float(mag.max()), 1e-6)
        scale = 14.0 / max_mag
        for xi, yi, ui, vi, ok in zip(x.ravel(), y.ravel(),
                                       u.ravel(), v.ravel(), valid.ravel()):
            if not ok:
                continue
            p1 = (int(round(xi)), int(round(yi)))
            p2 = (int(round(xi + ui * scale)), int(round(yi + vi * scale)))
            cv2.arrowedLine(overlay, p1, p2, (0, 255, 255), 1, tipLength=0.3)

        self._show_preview_array(overlay)
        unit = "m/s" if self.results_in_world_units else "px/frame"
        mean_mag = float(np.nanmean(np.hypot(self.results_u[pair_k],
                                              self.results_v[pair_k])))
        n_valid = int(np.isfinite(self.results_u[pair_k]).sum())
        n_total = self.results_u[pair_k].size
        self.pair_info_var.set(
            f"Pair {pair_k + 1}/{len(self.results_pairs)} "
            f"({i + 1}-{j + 1})  mean |V|={mean_mag:.3f} {unit}  "
            f"valid={n_valid}/{n_total}"
        )

    # ────────────────────────── save results ─────────────────────────────────

    def _save_results(self):
        """Write a minimal PIVlab-style results file with just the 7 fields:

            calxy        scalar m/px      length scale
            calu         scalar (m/s)/(px/frame)  x-velocity scale (signed)
            calv         scalar (m/s)/(px/frame)  y-velocity scale (signed)
            x            (n_rows, n_cols) px
            y            (n_rows, n_cols) px
            u_original   (n_pairs, n_rows, n_cols) px/frame
            v_original   (n_pairs, n_rows, n_cols) px/frame

        Positions and velocities are stored in pixels — multiply by calxy
        for metres, by calu / calv for m/s.
        """
        if self.results_u_original is None:
            messagebox.showinfo("PIV", "Run an analysis first.")
            return
        out_path = filedialog.asksaveasfilename(
            title="Save results", defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")])
        if not out_path:
            return

        cal = self._calibration_settings()

        save_primary_npz(out_path, self.results_x, self.results_y,
                         self.results_u_original, self.results_v_original,
                         cal, self.results_in_world_units)

        # Reconstruct x_px / u_orig for the log message.
        if self.results_in_world_units:
            x_px = ((self.results_x - cal.x_offset_m) / cal.m_per_pixel) * cal.x_sign
            uvs = cal.m_per_second_per_px_per_frame
            u_orig = self.results_u_original / uvs * cal.x_sign
        else:
            x_px = self.results_x
            u_orig = self.results_u_original
        self._log(
            f"Saved → {out_path}\n"
            f"  calxy={cal.m_per_pixel:.4e} m/px  "
            f"calu={cal.m_per_second_per_px_per_frame * cal.x_sign:+.4f}  "
            f"calv={cal.m_per_second_per_px_per_frame * cal.y_sign:+.4f}\n"
            f"  x,y shape         : {x_px.shape}\n"
            f"  u/v_original shape: {u_orig.shape}"
        )

    # ───────────────────────── preview rendering ─────────────────────────────

    def _show_preview_path(self, path):
        try:
            img = load_image_uint8(str(path))
            self._show_preview_array(img)
        except Exception as e:
            self._log(f"Preview load failed: {e}")

    def _show_preview_array(self, img):
        if img.ndim == 2:
            pil = Image.fromarray(img, mode="L")
        else:
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # Fit to current label size, with sensible maxima.
        label_w = max(self.preview_label.winfo_width(), 200)
        label_h = max(self.preview_label.winfo_height(), 200)
        pil = pil.copy()
        pil.thumbnail((label_w, label_h))
        self.last_preview_pil = pil
        photo = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # keep ref

    # ───────────────────────────── logging ───────────────────────────────────

    def _log(self, text):
        def append():
            self.log.insert("end", text + "\n")
            self.log.see("end")
        self.root.after(0, append)

    def _show_about(self):
        messagebox.showinfo("PIV Simple",
            "PIV Simple — simplified Python port of PIVlab.\n"
            "Algorithms from PIVlab (Shrediquette) and OpenPIV.\n"
            "Excludes neural-net masking and ensemble correlation.")


def main():
    root = tk.Tk()
    PIVGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
