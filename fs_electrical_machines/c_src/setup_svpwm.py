# setup_svpwm.py
# =============================================================================
# Cython build script for the SVPWM extension.
# Run from fs_electrical_machines/foc_generator/c_src/:
#
#   python setup_svpwm.py build_ext --inplace
#
# Or use build_svpwm.bat (Windows).
# =============================================================================

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))

ext = Extension(
    name="svpwm_wrapper",
    sources=[
        os.path.join(HERE, "svpwm_wrapper.pyx"),
        os.path.join(HERE, "svpwm.c"),
    ],
    include_dirs=[
        HERE,
        np.get_include(),
    ],
    extra_compile_args=["/O2"] if os.name == "nt" else ["-O2", "-ffast-math"],
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
