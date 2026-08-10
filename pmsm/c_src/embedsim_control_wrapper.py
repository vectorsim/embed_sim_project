"""
embedsim_control_wrapper.py  —  pmsm/c_src/
================================================================
Builds the embedsim_control_wrapper Cython extension.

This is a wrapper for the sensor-based control from
embed_sim_control.c / embed_sim_control.h.

The wrapper exposes:
    1. Control functions: control_init(), control_step()
    2. Transform functions: clarke(), park(), inv_park(), inv_clarke()

Sources:
    embedsim_control_wrapper.pyx   — Cython wrapper
    embed_sim_control.c            — Top-level control (sensor-based)
    embed_sim_dfc_controller.c     — DFC controller
    embed_sim_coordinate_transform.c  — Clarke/Park transforms
    embed_sim_sv_pwm.c             — SVPWM
    embed_sim_matrix.c             — Matrix library

Usage:
    python embedsim_control_wrapper.py build_ext --inplace
"""

import sys
from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── resolve paths ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # pmsm/c_src/
_PKG = _HERE.parent                              # pmsm/
_PROJECT_ROOT = _PKG.parent                      # EMProject/

print("=" * 60)
print("EmbedSim Controller Wrapper Builder (Sensor-Based)")
print("=" * 60)
print(f"Source directory      : {_HERE}")
print(f"Package directory     : {_PKG}")
print(f"Project root          : {_PROJECT_ROOT}")
print(f"Python version        : {sys.version_info.major}.{sys.version_info.minor}")
print("=" * 60)

# Check if all required source files exist
required_sources = [
    "embedsim_control_wrapper.pyx",
    "embed_sim_control.c",
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
    sys.exit(1)

print("=" * 60)

# Detect compiler and set appropriate flags
is_msvc = sys.platform == "win32"
is_linux = sys.platform.startswith("linux")
is_mac = sys.platform == "darwin"

# Include paths
include_dirs = [
    str(_HERE),
    str(_PROJECT_ROOT / "aurix_complex_device_driver" / "EmbedSim_Pmsm_TC3" / "EmbedSim"),
    np.get_include(),
]

print("\nInclude directories:")
for d in include_dirs:
    print(f"  {d}")

# Compiler flags
if is_msvc:
    extra_compile_args = [
        "/O2",
        "/fp:fast",
        "/GS-",
        "/Zc:inline",
        "/wd4244",  # conversion loss
        "/wd4018",  # signed/unsigned mismatch
        "/wd4101",  # unreferenced local variable
        "/wd4127",  # conditional expression is constant
        "/wd4996",  # deprecated POSIX functions
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
        "-Wno-unused-parameter",
        "-Wno-unused-but-set-variable",
    ]
    # Add -march=native for better performance on supported platforms
    if is_linux or is_mac:
        extra_compile_args.append("-march=native")
    extra_link_args = []

# Define the extension module
ext = Extension(
    name="embedsim_control_wrapper",
    sources=[
        str(_HERE / "embedsim_control_wrapper.pyx"),
        str(_HERE / "embed_sim_control.c"),
        str(_HERE / "embed_sim_dfc_controller.c"),
        str(_HERE / "embed_sim_coordinate_transform.c"),
        str(_HERE / "embed_sim_sv_pwm.c"),
        str(_HERE / "embed_sim_matrix.c"),
    ],
    include_dirs=include_dirs,
    define_macros=[
        ('EMBEDSIM_BUILD', '1'),
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ('_USE_MATH_DEFINES', '1'),  # For Windows M_PI
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
    "profile": False,
    "linetrace": False,
}

setup(
    name="embedsim_control_wrapper",
    version="1.0.0",
    description="EmbedSim Control Wrapper -- Sensor-based PMSM FOC",
    long_description="""
    EmbedSim Control Wrapper for sensor-based PMSM control.
    
    Exposes:
      - control_init()  : Initialize the controller
      - control_step()  : Execute one control step
      - clarke()        : UVW -> AlphaBeta transform
      - park()          : AlphaBeta -> DQ transform  
      - inv_park()      : DQ -> AlphaBeta transform
      - inv_clarke()    : AlphaBeta -> UVW transform
    
    All transforms use the same C code that runs on the AURIX TC38x.
    """,
    long_description_content_type="text/plain",
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
print("Extension version: 1.0.0 (Sensor-based Control)")
print("=" * 60)
print("\nTo build, run:")
print("  python embedsim_control_wrapper.py build_ext --inplace")
print("=" * 60)