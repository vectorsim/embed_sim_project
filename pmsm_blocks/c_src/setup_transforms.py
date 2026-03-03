"""
setup_transforms.py
===================
Compile all four transform Cython wrappers (Clarke, Park, etc.)
from a single transforms_wrapper.pyx + transforms.c pair.

Usage:
    cd pmsm_blocks/c_src
    python setup_transforms.py build_ext --inplace
"""
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

compile_args = ['/O2', '/std:c11'] if sys.platform == 'win32' else ['-O3', '-ffast-math', '-std=c11']

ext = Extension(
    name    = 'transforms_wrapper',
    sources = [
        'transforms_wrapper.pyx',
        'transforms.c',
    ],
    include_dirs       = [np.get_include(), '.'],
    extra_compile_args = compile_args,
    libraries          = ['m'] if sys.platform != 'win32' else [],
)

setup(
    name       = 'transforms_wrapper',
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
