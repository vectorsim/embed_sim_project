# distutils: language = c
# cython: language_level = 3
"""
svpwm_wrapper.pyx  --  pmsm_blocks/c_src/
Cython wrapper for svpwm.c (Space Vector PWM Modulator)
MISRA C:2012 / ASIL-D  --  real32_T interface
"""

import numpy as np
cimport numpy as np
from libc.string cimport memset

# --------------------------------------------------------------------------- #
#  C declarations                                                              #
# --------------------------------------------------------------------------- #
cdef extern from "svpwm.h":
    ctypedef float real32_T

    ctypedef struct SVPWM_Block_T:
        real32_T v_dc

    void SVPWM_Init   (SVPWM_Block_T *blk, real32_T v_dc)
    void SVPWM_Compute(SVPWM_Block_T *blk,
                       real32_T v_a, real32_T v_b, real32_T v_c,
                       real32_T *duty_a, real32_T *duty_b, real32_T *duty_c)

# --------------------------------------------------------------------------- #
#  Python wrapper class                                                        #
# --------------------------------------------------------------------------- #
cdef class SVPWMWrapper:
    """
    Space Vector PWM modulator wrapper.

    Parameters
    ----------
    v_dc : float
        DC-link voltage [V].

    Usage
    -----
    >>> svpwm = SVPWMWrapper(v_dc=48.0)
    >>> duty = svpwm.compute(v_a, v_b, v_c)   # returns float32 ndarray[3]
    """

    cdef SVPWM_Block_T _blk
    cdef float _v_dc

    def __cinit__(self, float v_dc=48.0):
        self._v_dc = v_dc
        SVPWM_Init(&self._blk, v_dc)

    def reinit(self, float v_dc):
        """Re-initialise with a new DC-link voltage."""
        self._v_dc = v_dc
        SVPWM_Init(&self._blk, v_dc)

    def compute(self, float v_a, float v_b, float v_c):
        """
        Compute duty cycles from phase voltages.

        Returns
        -------
        numpy.ndarray, dtype=float32, shape=(3,)
            [duty_a, duty_b, duty_c]  clamped to [0, 1].
        """
        cdef real32_T da, db, dc
        SVPWM_Compute(&self._blk, v_a, v_b, v_c, &da, &db, &dc)
        cdef np.ndarray[np.float32_t, ndim=1] out = np.empty(3, dtype=np.float32)
        out[0] = da
        out[1] = db
        out[2] = dc
        return out

    def compute_array(self,
                      np.ndarray[np.float32_t, ndim=1] vabc not None):
        """
        Convenience overload accepting a length-3 float32 array [v_a, v_b, v_c].
        """
        if vabc.shape[0] < 3:
            raise ValueError("vabc must have at least 3 elements")
        return self.compute(vabc[0], vabc[1], vabc[2])

    @property
    def v_dc(self):
        return self._v_dc
