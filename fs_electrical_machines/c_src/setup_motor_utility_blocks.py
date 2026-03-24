# setup_motor_utility_blocks.py
# ==============================
# EmbedSim — NANOTEC DB42S02  Open-loop V/f
# Build script for motor_utility_blocks_wrapper.pyx
#
# Usage (from fs_electrical_machines/c_src/):
#   python setup_motor_utility_blocks.py build_ext --inplace
#
# Produces:
#   motor_utility_blocks_wrapper.cp312-win_amd64.pyd  (Windows)
#   motor_utility_blocks_wrapper.cpython-312-x86_64.so (Linux)
#
# The .pyd/.so is imported by motor_utility_blocks.py (Python block file).

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# MSVC on Windows: /O2   |   GCC/Clang: -O3 -ffast-math
if sys.platform == "win32":
    compile_args = ["/O2"]
else:
    compile_args = ["-O3", "-ffast-math"]

ext = Extension(
    name="motor_utility_blocks_wrapper",
    sources=[
        "motor_utility_blocks_wrapper.pyx",
        "embed_sim_motor_utility_blocks.c",
    ],
    include_dirs=[np.get_include(), "."],
    extra_compile_args=compile_args,
)

setup(
    name="motor_utility_blocks_wrapper",
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            "language_level": "3",
            "boundscheck":    False,
            "wraparound":     False,
            "cdivision":      True,
        },
        annotate=True,
    ),
)
