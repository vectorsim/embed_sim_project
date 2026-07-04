"""
setup_dfc_controller.py  —  pmsm/c_src/
================================================================
Builds the dfc_controller_wrapper Cython extension (v4, sensorless).

Sources compiled into this extension:
    dfc_controller_wrapper.pyx        — Cython wrapper
    embed_sim_dfc_controller.c        — DFC FOC controller (SVPWM integrated)
    embed_sim_coordinate_transform.c  — Clarke/Park transforms
    embed_sim_sv_pwm.c                — Space Vector PWM (called by DFC_Step)
    embed_sim_matrix.c                — Matrix library

Usage:
    python setup_dfc_controller.py build_ext --inplace
    or
    ./build_dfc_controller.sh (Linux)
    build_dfc_controller.bat (Windows)
"""

import sys
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── resolve paths ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # pmsm/c_src/
_PKG = _HERE.parent                              # pmsm/

print("=" * 60)
print("EmbedSim Differential Flatness Controller Extension Builder")
print("=" * 60)
print(f"Source directory      : {_HERE}")
print(f"Package directory     : {_PKG}")
print(f"Python version        : {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Check if all required source files exist
required_sources = [
    "dfc_controller_wrapper.pyx",
    "embed_sim_dfc_controller.c",
    "embed_sim_coordinate_transform.c",
    "embed_sim_sv_pwm.c",
    "embed_sim_matrix.c",
]

missing_sources = []
for src in required_sources:
    src_path = _HERE / src
    if not src_path.exists():
        missing_sources.append(str(src_path))
        print(f"⚠ WARNING: Missing source file: {src_path}")
    else:
        print(f"✓ Found: {src}")

if missing_sources:
    print("\n" + "!" * 60)
    print("ERROR: Missing required source files. Build will likely fail.")
    print("!" * 60)
    print("Missing files:")
    for f in missing_sources:
        print(f"  - {f}")
    print("!" * 60)

print("=" * 60)

# Detect compiler and set appropriate flags
is_msvc = sys.platform == "win32"

if is_msvc:
    extra_compile_args = [
        "/O2",
        "/fp:fast",
        "/GS-",
        "/Zc:inline",
        "/wd4244",  # conversion loss
        "/wd4018",  # signed/unsigned mismatch
        "/wd4101",  # unreferenced local variable
    ]
    extra_link_args = []
else:
    extra_compile_args = [
        "-O3",
        "-ffast-math",
        "-std=c99",
        "-Wall",
        "-Wextra",
        "-Wno-unused-function",
        "-Wno-unused-variable",
        "-Wno-missing-field-initializers",
        "-Wno-sign-compare",
    ]
    # Don't use -Werror as Cython-generated code may have warnings
    extra_link_args = []

# Define the extension module
ext = Extension(
    name="dfc_controller_wrapper",
    sources=[
        str(_HERE / "dfc_controller_wrapper.pyx"),
        str(_HERE / "embed_sim_dfc_controller.c"),
        str(_HERE / "embed_sim_coordinate_transform.c"),
        str(_HERE / "embed_sim_sv_pwm.c"),
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
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)

# Cython compiler directives
compiler_directives = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "initializedcheck": False,
    "nonecheck": False,
}

setup(
    name="dfc_controller_wrapper",
    version="4.3.0",
    description="EmbedSim Differential Flatness FOC Controller -- Sensorless SMO, integrated SVPWM",
    author="EmbedSim Team",
    ext_modules=cythonize(
        [ext],
        compiler_directives=compiler_directives,
        annotate=True,
        force=True,
    ),
    zip_safe=False,
)

print("\n" + "=" * 60)
print("Build configuration complete")
print("Extension version: 4.3.0 (Sensorless SMO, integrated SVPWM)")
print("=" * 60)
print("\nTo build, run:")
print("  python setup_dfc_controller.py build_ext --inplace")
print("=" * 60)
