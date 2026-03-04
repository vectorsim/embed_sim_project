"""
setup_coordinate_transform.py
==============================
Compile the coordinate transform Cython wrapper with matrix operations.

Wraps:
    Coordinate_Transform.c     — Matrix-based Clarke/Park math
    coordinate_transform_wrapper.pyx  — Cython bridge

Usage:
    cd electrical_blocks/c_src
    python setup_coordinate_transform.py build_ext --inplace
"""

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# ── Compiler flags ────────────────────────────────────────────────────────────
if sys.platform == 'win32':
    # MSVC — /O2 optimise, /fp:fast for FPU-friendly float maths
    compile_args = ['/O2', '/fp:fast']
    link_args    = []
    libraries    = []
else:
    # GCC / Clang
    compile_args = ['-O3', '-ffast-math', '-std=c11']
    link_args    = []
    libraries    = ['m']   # libm for cosf/sinf

# ── Extension definition ──────────────────────────────────────────────────────
ext = Extension(
    name    = 'coordinate_transform_wrapper',
    sources = [
        'coordinate_transform_wrapper.pyx',   # Cython bridge
        'Coordinate_Transform.c',             # Matrix-based transforms
    ],
    include_dirs       = [np.get_include(), '.'],
    extra_compile_args = compile_args,
    extra_link_args    = link_args,
    libraries          = libraries,
)

# ── Build ─────────────────────────────────────────────────────────────────────
setup(
    name        = 'coordinate_transform_wrapper',
    ext_modules = cythonize(
        [ext],
        compiler_directives = {
            'language_level': '3',
            'boundscheck'   : False,   # no index checks — MCU-style
            'wraparound'    : False,
            'cdivision'     : True,    # no Python ZeroDivisionError overhead
        },
        annotate = True,               # generates .html annotation file
    ),
)