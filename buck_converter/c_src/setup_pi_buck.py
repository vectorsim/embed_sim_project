"""
setup_pi_buck.py
=================
Compile the PI Buck Controller Cython wrapper.

Usage:
    cd electrical_blocks/c_src
    python setup_pi_buck.py build_ext --inplace
"""

import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

# Compiler flags optimized for performance
if sys.platform == 'win32':
    compile_args = [
        '/O2',                    # Maximize speed
        '/fp:fast',               # Fast floating point model
        '/arch:AVX2',             # Use AVX2 if available
        '/GL',                     # Whole program optimization
        '/GS-',                    # Disable buffer security checks (for speed)
    ]
    link_args = ['/LTCG']          # Link-time code generation
    libraries = []
else:  # Linux/Mac
    compile_args = [
        '-O3',                     # High optimization
        '-march=native',           # Optimize for current CPU
        '-ffast-math',             # Fast math optimizations
        '-funroll-loops',          # Unroll loops
        '-std=c11',                 # C11 standard
    ]
    link_args = []
    libraries = ['m']               # Math library

ext = Extension(
    name='pi_buck_wrapper',
    sources=[
        'pi_buck_wrapper.pyx',
        'pi_buck_controller.c',
    ],
    include_dirs=[
        np.get_include(),           # NumPy headers
        '.',                         # Current directory
        '../..',                     # Project root (for Sys_Types.h)
    ],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
    libraries=libraries,
    define_macros=[
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
    ],
)

setup(
    name='pi_buck_wrapper',
    version='1.0.0',
    description='Cython wrapper for PI Buck Controller',
    author='EmbedSim Framework',
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,      # Disable bounds checking for speed
            'wraparound': False,        # Disable negative indexing
            'cdivision': True,          # Use C division
            'nonecheck': False,         # Disable None checks
            'initializedcheck': False,  # Disable initialized checks
            'embedsignature': True,     # Include signatures in docstrings
        },
        annotate=True,                  # Generate HTML annotation
    ),
    zip_safe=False,
)