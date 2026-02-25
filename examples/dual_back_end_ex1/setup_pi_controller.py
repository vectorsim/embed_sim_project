# setup_pi_controller.py
# Auto-generated build script for 'pi_controller' Cython wrapper
#
# Usage:
#   python setup_pi_controller.py build_ext --inplace
#
# Prerequisite: pi_controller.c must exist in the same directory.

import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

compile_args = ['/O2'] if sys.platform == 'win32' else ['-O3', '-ffast-math']

ext = Extension(
    name='pi_controller_wrapper',
    sources=[
        'pi_controller_wrapper.pyx',
        'pi_controller.c',  # your C implementation
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
)

setup(
    name='pi_controller_wrapper',
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        },
        annotate=True,
    ),
)