# =============================================================================
# setup_smo.py
# =============================================================================
# Builds the smo_wrapper Cython extension.
#
# Run from electrical_blocks\c_src\:
#   python setup_smo.py build_ext --inplace
#
# Output:
#   smo_wrapper.cpXXX-win_amd64.pyd   (copied to parent dir by build_all.bat)
#
# C sources compiled:
#   smo.c            — Sliding Mode Observer, Utkin-type (MISRA C:2012, ASIL-D)
#
# Note: smo.c uses atan2f / cosf / sinf from libm — link with /link /DEFAULTLIB
#       or ensure TASKING runtime provides these in embedded build.
# =============================================================================

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name="smo_wrapper",
    sources=[
        "smo_wrapper.pyx",
        "smo.c",
    ],
    include_dirs=[
        HERE,
        np.get_include(),
    ],
    extra_compile_args=[
        "/O2",
        "/W3",
    ],
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ],
    language="c",
)

setup(
    name="smo_wrapper",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
        },
    ),
)
