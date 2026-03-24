"""
setup_smc_controller.py  —  fs_electrical_machines/c_src/
================================================================
Builds the smc_controller_wrapper Cython extension.

Sources compiled into this extension:
    smc_controller_wrapper.pyx    — Cython wrapper
    SMC_Controller.c              — SMC FOC controller implementation
    Coordinate_Transform.c        — Clarke/Park transforms
    Matrix.c                      — MatrixFloat type + Q31 infrastructure
                                   shared by all fs_electrical_machines blocks

Usage:
    python setup_smc_controller.py build_ext --inplace
"""

import sys
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── resolve paths via _path_utils ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # fs_electrical_machines/c_src/
_PKG  = _HERE.parent                             # fs_electrical_machines/

# Import _path_utils from the package
sys.path.insert(0, str(_PKG))
try:
    from _path_utils import get_current_parent
    _C_SRC = get_current_parent() / "c_src"         # same as _HERE
except ImportError:
    # Fallback if _path_utils not available
    _C_SRC = _HERE

print("=" * 60)
print("EmbedSim SMC Controller Extension Builder")
print("=" * 60)
print(f"Source directory      : {_C_SRC}")
print(f"Package directory     : {_PKG}")
print(f"Python version        : {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Define the extension module with correct name for the output path
ext = Extension(
    name="smc_controller_wrapper",  # Simple name for local build
    sources=[
        str(_C_SRC / "smc_controller_wrapper.pyx"),
        str(_C_SRC / "embed_sim_smc_controller.c"),
        str(_C_SRC / "embed_sim_coordinate_transform.c"),
        str(_C_SRC / "embed_sim_matrix.c"),
    ],
    include_dirs=[
        str(_C_SRC),
        np.get_include(),
    ],
    define_macros=[
        ('EMBEDSIM_BUILD', '1'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ],
    extra_compile_args=["/O2"] if sys.platform == "win32" else ["-O3", "-ffast-math"],
    language="c",
)

setup(
    name="smc_controller_wrapper",
    version="1.0.0",
    description="EmbedSim Sliding Mode FOC Controller Module",
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