"""
embedsim_control_wrapper.py
===========================

Build script for the EmbedSim Cython control wrapper.

Location:
    pmsm/c_src/

Build:
    python embedsim_control_wrapper.py build_ext --inplace

The generated ABI-tagged .pyd is produced in this directory.
The Windows batch file is responsible for copying it to:

    pmsm/embedsim_control_wrapper.pyd
"""

import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize


# ============================================================================
# Paths
# ============================================================================

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
_PROJECT_ROOT = _PKG.parent


print("=" * 70)
print("EmbedSim Control Wrapper Builder")
print("=" * 70)
print(f"Source directory : {_HERE}")
print(f"Package directory: {_PKG}")
print(f"Project root     : {_PROJECT_ROOT}")
print(
    f"Python version   : "
    f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
)
print("=" * 70)


# ============================================================================
# Required source files
# ============================================================================

required_sources = [
    "embedsim_control_wrapper.pyx",
    "embed_sim_control.c",
    "embed_sim_dfc_controller.c",
    "embed_sim_coordinate_transform.c",
    "embed_sim_sv_pwm.c",
    "embed_sim_matrix.c",
    "embed_sim_cython_interface.c",
]

required_headers = [
    "embed_sim_control.h",
    "embed_sim_dfc_controller.h",
    "embed_sim_coordinate_transform.h",
    "embed_sim_sv_pwm.h",
    "embed_sim_matrix.h",
    "embed_sim_cython_interface.h",
]


missing = []

print("\nChecking source files:")

for filename in required_sources:
    path = _HERE / filename

    if path.exists():
        print(f"  OK      {filename}")
    else:
        print(f"  MISSING {filename}")
        missing.append(path)


print("\nChecking header files:")

for filename in required_headers:
    path = _HERE / filename

    if path.exists():
        print(f"  OK      {filename}")
    else:
        print(f"  MISSING {filename}")
        missing.append(path)


if missing:
    print("\n" + "=" * 70)
    print("ERROR: Required source/header files are missing.")
    print("=" * 70)

    for path in missing:
        print(f"  {path}")

    sys.exit(1)


# ============================================================================
# Platform
# ============================================================================

is_msvc = sys.platform == "win32"
is_linux = sys.platform.startswith("linux")
is_mac = sys.platform == "darwin"


# ============================================================================
# Include directories
# ============================================================================

include_dirs = [
    str(_HERE),
    str(
        _PROJECT_ROOT
        / "aurix_complex_device_driver"
        / "EmbedSim_Pmsm_TC3"
        / "EmbedSim"
    ),
    np.get_include(),
]


print("\nInclude directories:")

for directory in include_dirs:
    print(f"  {directory}")


# ============================================================================
# Compiler flags
# ============================================================================

if is_msvc:
    extra_compile_args = [
        "/O2",
        "/fp:fast",
        "/GS-",
        "/Zc:inline",
        "/wd4244",
        "/wd4018",
        "/wd4101",
        "/wd4127",
        "/wd4996",
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

    if is_linux or is_mac:
        extra_compile_args.append("-march=native")

    extra_link_args = []


# ============================================================================
# Extension
# ============================================================================

extension = Extension(
    name="embedsim_control_wrapper",
    sources=[
        str(_HERE / "embedsim_control_wrapper.pyx"),

        str(_HERE / "embed_sim_control.c"),
        str(_HERE / "embed_sim_dfc_controller.c"),
        str(_HERE / "embed_sim_coordinate_transform.c"),
        str(_HERE / "embed_sim_sv_pwm.c"),
        str(_HERE / "embed_sim_matrix.c"),

        # IMPORTANT:
        # This was previously checked but not compiled.
        str(_HERE / "embed_sim_cython_interface.c"),
    ],
    include_dirs=include_dirs,
    define_macros=[
        ("EMBEDSIM_BUILD", "1"),
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("_USE_MATH_DEFINES", "1"),
    ],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    language="c",
)


# ============================================================================
# Cython directives
# ============================================================================

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


# ============================================================================
# Setup
# ============================================================================

setup(
    name="embedsim_control_wrapper",
    version="2.1.1",
    description=(
        "EmbedSim sensor-based PMSM DFC control wrapper "
        "with motor state reporting"
    ),
    ext_modules=cythonize(
        [extension],
        compiler_directives=compiler_directives,
        annotate=True,
        force=True,
    ),
    zip_safe=False,
)


print("\n" + "=" * 70)
print("Build configuration complete.")
print("=" * 70)
print()
print("Expected command:")
print("  python embedsim_control_wrapper.py build_ext --inplace")
print()
print("Expected output:")
print("  embedsim_control_wrapper*.pyd")
print("=" * 70)