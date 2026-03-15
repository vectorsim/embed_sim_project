"""
fs_electrical_machines
======================
EmbedSim block library for electrical machines and FOC transforms.

Provides
--------
  ClarkeTransformBlock    — [i_a, i_b, i_c]        → [i_alpha, i_beta]
  ParkTransformBlock      — [i_alpha, i_beta, θ_e]  → [i_d, i_q]
  InvParkTransformBlock   — [v_d, v_q, θ_e]         → [v_alpha, v_beta]
  InvClarkeTransformBlock — [v_alpha, v_beta]        → [v_a, v_b, v_c]

Each block has a pure-Python compute_py() fallback and uses the
Cython/C backend (coordinate_transform_wrapper) when built.

Build the C extension
---------------------
    cd fs_electrical_machines\\c_src
    build_coordinate_transform.bat          # Windows
    python setup_coordinate_transform.py build_ext --inplace
"""

from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
    InvClarkeTransformBlock,
)

__all__ = [
    "ClarkeTransformBlock",
    "ParkTransformBlock",
    "InvParkTransformBlock",
    "InvClarkeTransformBlock",
]
