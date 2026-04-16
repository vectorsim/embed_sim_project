"""
setup_mpc_controller.py  —  fs_electrical_machines/c_src/
================================================================
Builds the mpc_controller_wrapper Cython extension.

Sources compiled into this extension:
    mpc_controller_wrapper.pyx    — Cython wrapper
    embed_sim_mpc_controller.c    — MPC FOC controller implementation
    embed_sim_coordinate_transform.c — Clarke/Park transforms
    embed_sim_matrix.c            — MatrixFloat type + Q31 infrastructure
                                   shared by all fs_electrical_machines blocks

Usage:
    python setup_mpc_controller.py build_ext --inplace

MISRA C:2012 Compliance:
    - All float literals have 'f' suffix
    - No mixed-mode arithmetic
    - Single return per function
    - All pointer parameters checked for NULL
"""

import sys
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── resolve paths via _path_utils ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # fs_electrical_machines/c_src/
_PKG = _HERE.parent  # fs_electrical_machines/

# Import _path_utils from the package
sys.path.insert(0, str(_PKG))
try:
    from _path_utils import get_current_parent

    _C_SRC = get_current_parent() / "c_src"  # same as _HERE
except ImportError:
    # Fallback if _path_utils not available
    _C_SRC = _HERE

print("=" * 60)
print("EmbedSim MPC Controller Extension Builder")
print("=" * 60)
print(f"Source directory      : {_C_SRC}")
print(f"Package directory     : {_PKG}")
print(f"Python version        : {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Check for required source files
required_files = [
    "mpc_controller_wrapper.pyx",
    "embed_sim_mpc_controller.c",
    "embed_sim_coordinate_transform.c",
    "embed_sim_matrix.c",
]

missing_files = []
for f in required_files:
    if not (_C_SRC / f).exists():
        missing_files.append(f)

if missing_files:
    print("\n⚠️  WARNING: Missing source files:")
    for f in missing_files:
        print(f"    - {f}")
    print("\nEnsure all source files are present in:")
    print(f"    {_C_SRC}")
    print("\nContinuing anyway...\n")

# Define the extension module with correct name for the output path
ext = Extension(
    name="mpc_controller_wrapper",  # Simple name for local build
    sources=[
        str(_C_SRC / "mpc_controller_wrapper.pyx"),
        str(_C_SRC / "embed_sim_mpc_controller.c"),
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
        # MPC-specific defines
        ('MPC_ENABLE_DIAGNOSTICS', '1'),
    ],
    extra_compile_args=["/O2"] if sys.platform == "win32" else [
        "-O3",
        "-ffast-math",
        "-Wno-unused-function",
        "-Wno-unused-variable",
    ],
    language="c",
)

setup(
    name="mpc_controller_wrapper",
    version="1.0.0",
    description="EmbedSim Model Predictive Control FOC Controller Module",
    long_description="""
    MPC FOC Controller for NANOTEC DB42S02 on AURIX TC3xx.

    Features:
        - 3-state receding-horizon MPC (id, iq, omega_m)
        - Analytical closed-form solution O(N)
        - Sliding Mode Observer (SMO) for back-EMF estimation
        - Encoder speed estimation with IIR filtering
        - BEMF feedforward with physical clamp
        - Soft-start current limiting
        - Speed-error integral correction with anti-windup

    MISRA C:2012 compliant C code generation.
    """,
    author="EmbedSim Team",
    author_email="support@embedsim.com",
    url="https://github.com/EmbedSim/fs_electrical_machines",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
            "embedsignature": True,
        },
        annotate=True,  # Generates .html file showing Cython→C translation
    ),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "cython>=0.29.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: C",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)",
    ],
)

print("\n" + "=" * 60)
print("MPC Controller Build Configuration Complete")
print("=" * 60)
print("\nTo build the extension:")
print("    python setup_mpc_controller.py build_ext --inplace")
print("\nAfter successful build, you can import:")
print("    from fs_electrical_machines.mpc_controller_wrapper import MPCControllerWrapper")
print("=" * 60)