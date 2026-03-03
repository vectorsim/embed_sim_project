"""
setup_pmsm_motor.py
===================
Compile the PMSM motor Cython wrapper.

Usage:
    cd pmsm_blocks/c_src
    python setup_pmsm_motor.py build_ext --inplace
"""
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

compile_args = ['/O2', '/std:c11'] if sys.platform == 'win32' else ['-O3', '-ffast-math', '-std=c11']

ext = Extension(
    name    = 'pmsm_motor_wrapper',
    sources = [
        'pmsm_motor_wrapper.pyx',
        'pmsm_motor.c',
    ],
    include_dirs       = [np.get_include(), '.'],
    extra_compile_args = compile_args,
    libraries          = ['m'] if sys.platform != 'win32' else [],
)

setup(
    name       = 'pmsm_motor_wrapper',
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
