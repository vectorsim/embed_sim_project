"""
setup_smc.py
============
Compile the Sliding Mode Controller Cython wrapper.

Usage:
    cd electrical_blocks/c_src
    python setup_smc.py build_ext --inplace
"""

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

if sys.platform == 'win32':
    compile_args = ['/O2', '/fp:fast']
    link_args    = []
    libraries    = []
else:
    compile_args = ['-O3', '-ffast-math', '-std=c11']
    link_args    = []
    libraries    = ['m']

ext = Extension(
    name    = 'smc_wrapper',
    sources = [
        'smc_wrapper.pyx',
        'sliding_mode_controller.c',
    ],
    include_dirs       = [np.get_include(), '.'],
    extra_compile_args = compile_args,
    extra_link_args    = link_args,
    libraries          = libraries,
)

setup(
    name        = 'smc_wrapper',
    ext_modules = cythonize(
        [ext],
        compiler_directives = {
            'language_level': '3',
            'boundscheck'   : False,
            'wraparound'    : False,
            'cdivision'     : True,
        },
        annotate = True,
    ),
)
