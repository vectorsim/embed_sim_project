# =============================================================================
# setup_svpwm.py
# =============================================================================
# Builds the svpwm_wrapper Cython extension.
#
# Run from electrical_blocks\c_src\:
#   python setup_svpwm.py build_ext --inplace
#
# Output:
#   svpwm_wrapper.cpXXX-win_amd64.pyd   (copied to parent dir by build_all.bat)
#
# C sources compiled:
#   svpwm.c          — Space Vector PWM modulator (MISRA C:2012, ASIL-D)
# =============================================================================

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name="svpwm_wrapper",
    sources=[
        "svpwm_wrapper.pyx",
        "svpwm.c",
    ],
    include_dirs=[
        HERE,
        np.get_include(),
    ],
    extra_compile_args=[
        "/O2",          # Optimise (MSVC / TASKING-like)
        "/W3",          # Warning level 3
    ],
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
    ],
    language="c",
)

setup(
    name="svpwm_wrapper",
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
