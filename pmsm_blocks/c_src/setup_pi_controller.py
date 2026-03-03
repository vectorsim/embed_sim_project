"""
setup_pi_controller.py
======================
Compile the PI controller Cython wrapper.

Usage:
    cd pmsm_blocks/c_src
    python setup_pi_controller.py build_ext --inplace
"""
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

compile_args = ['/O2', '/std:c11'] if sys.platform == 'win32' else ['-O3', '-ffast-math', '-std=c11']

ext = Extension(
    name    = 'pi_controller_wrapper',
    sources = [
        'pi_controller_wrapper.pyx',
        'pi_controller.c',
    ],
    include_dirs       = [np.get_include(), '.'],
    extra_compile_args = compile_args,
    libraries          = [],
)

setup(
    name       = 'pi_controller_wrapper',
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
