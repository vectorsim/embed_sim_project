"""
setup_coordinate_transform.py  —  fs_electrical_machines/c_src/
================================================================
Builds the coordinate_transform_wrapper Cython extension.

Sources compiled into this extension:
    coordinate_transform_wrapper.pyx  — Cython wrapper
    Coordinate_Transform.c            — Clarke/Park/InvPark/InvClarke
    Matrix.c                          — MatrixFloat type + Q31 infrastructure
                                        shared by all fs_electrical_machines blocks

Usage:
    python setup_coordinate_transform.py build_ext --inplace
"""

import sys
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── resolve paths via _path_utils ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # fs_electrical_machines/c_src/
_PKG  = _HERE.parent                             # fs_electrical_machines/

sys.path.insert(0, str(_PKG))
from _path_utils import get_current_parent

_C_SRC = get_current_parent() / "c_src"         # same as _HERE

# ── extension ─────────────────────────────────────────────────────────────────
ext = Extension(
    name="coordinate_transform_wrapper",
    sources=[
        str(_C_SRC / "coordinate_transform_wrapper.pyx"),
        str(_C_SRC / "embed_sim_coordinate_transform.c"),
        str(_C_SRC / "embed_sim_matrix.c"),
    ],
    include_dirs=[
        str(_C_SRC),          # finds Coordinate_Transform.h, Matrix.h, Sys_Types.h
        np.get_include(),
    ],
    extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O2", "-ffast-math"],
    language="c",
)

setup(
    name="coordinate_transform_wrapper",
    ext_modules=cythonize(
        [ext],
        compiler_directives={"language_level": "3"},
    ),
)
