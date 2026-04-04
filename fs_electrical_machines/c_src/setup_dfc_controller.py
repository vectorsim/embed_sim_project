"""
setup_dfc_controller.py  —  fs_electrical_machines/c_src/
================================================================
Builds the dfc_controller_wrapper Cython extension.

Sources compiled into this extension:
    dfc_controller_wrapper.pyx    — Cython wrapper
    embed_sim_dfc_controller.c    — DFC FOC controller implementation
    embed_sim_coordinate_transform.c — Clarke/Park transforms
    embed_sim_matrix.c            — MatrixFloat type + Q31 infrastructure

Usage:
    python setup_dfc_controller.py build_ext --inplace
    or
    ./build_dfc_controller.sh (Linux)
    build_dfc_controller.bat (Windows)
"""

import sys
import os
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── resolve paths ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # fs_electrical_machines/c_src/
_PKG  = _HERE.parent                             # fs_electrical_machines/

print("=" * 60)
print("EmbedSim Differential Flatness Controller Extension Builder")
print("=" * 60)
print(f"Source directory      : {_HERE}")
print(f"Package directory     : {_PKG}")
print(f"Python version        : {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Define the extension module
ext = Extension(
    name="dfc_controller_wrapper",
    sources=[
        str(_HERE / "dfc_controller_wrapper.pyx"),
        str(_HERE / "embed_sim_dfc_controller.c"),
        str(_HERE / "embed_sim_coordinate_transform.c"),
        str(_HERE / "embed_sim_matrix.c"),
    ],
    include_dirs=[
        str(_HERE),
        np.get_include(),
    ],
    define_macros=[
        ('EMBEDSIM_BUILD', '1'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ],
    extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3", "-ffast-math", "-std=c99"],
    language="c",
)

setup(
    name="dfc_controller_wrapper",
    version="1.0.0",
    description="EmbedSim Differential Flatness FOC Controller Module",
    author="EmbedSim Team",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
        annotate=True,
    ),
)

print("\n" + "=" * 60)
print("Build configuration complete")
print("=" * 60)