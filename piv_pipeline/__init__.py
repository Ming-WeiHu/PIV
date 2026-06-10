"""piv_pipeline — Python port of the PIV Flakes post-processing pipeline.

Translated from the MATLAB pipeline (PIV_Pipeline_Master.m):
    Stage 1: fn_secondary_export.m  →  fn_secondary_export.py    [PORTED]
    Stage 2: fn_dynamic_masking.m   →  fn_dynamic_masking.py     [PORTED]
    Stage 3: fn_tertiary_export.m   →  fn_tertiary_export.py     [PORTED]
    Stage 4: fn_final_analysis.m    →  fn_final_analysis.py      [PORTED]
    Videos : PIV_Generate_Videos.m  →  piv_generate_videos.py    [PORTED]

Stays independent from the bioreactor analyzer (gui.py, particle_testing.py)
and from the upstream PIV computation (piv_simple.py, piv_gui.py).
"""

from .io_loaders import load_primary, parse_filename
from .fn_secondary_export import secondary_export
from .fn_dynamic_masking import dynamic_masking
from .fn_tertiary_export import tertiary_export
from .fn_final_analysis import final_analysis
from .piv_generate_videos import generate_videos

__all__ = [
    "load_primary", "parse_filename",
    "secondary_export", "dynamic_masking", "tertiary_export",
    "final_analysis", "generate_videos",
]
