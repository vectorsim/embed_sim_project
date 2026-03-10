# =============================================================================
# svpwm_wrapper.pyx
# =============================================================================
#
# Cython wrapper for the Space Vector PWM (SVPWM) C block.
#
# Exposes SVPWM_Block_T and SVPWM_Compute() to Python / EmbedSim.
#
# EmbedSim CodeGen attributes
# ----------------------------
#   step_func    : SVPWM_Compute
#   state_struct : SVPWM_Block_T
#   C_SOURCES    : ['svpwm.c']
#   C_HEADERS    : ['svpwm.h', 'Sys_Types.h']
#   NUM_INPUTS   : 1   (port 0: [v_a, v_b, v_c])
#   OUTPUT_SIZE  : 3   ([duty_a, duty_b, duty_c])
#
# =============================================================================

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import  numpy as np
cimport numpy as cnp

cnp.import_array()

# --------------------------------------------------------------------------
# C declarations
# --------------------------------------------------------------------------
cdef extern from "Sys_Types.h":
    ctypedef float real32_T

cdef extern from "svpwm.h":
    ctypedef struct SVPWM_Block_T:
        real32_T duty_a
        real32_T duty_b
        real32_T duty_c
        real32_T v_dc

    void SVPWM_Init   (SVPWM_Block_T *blk, real32_T v_dc)
    void SVPWM_Compute(SVPWM_Block_T *blk,
                       real32_T v_a, real32_T v_b, real32_T v_c,
                       real32_T *duty_a, real32_T *duty_b, real32_T *duty_c)

# --------------------------------------------------------------------------
# Python-visible wrapper class
# --------------------------------------------------------------------------
cdef class SVPWMWrapper:
    """
    Thin Cython wrapper around SVPWM_Block_T.

    Usage
    -----
    >>> w = SVPWMWrapper(v_dc=48.0)
    >>> duty = w.compute(v_a, v_b, v_c)   # returns np.ndarray shape (3,)
    """

    cdef SVPWM_Block_T _blk

    def __cinit__(self, float v_dc = 48.0):
        SVPWM_Init(&self._blk, v_dc)

    def compute(self,
                float v_a,
                float v_b,
                float v_c):
        """
        Compute SVPWM duty cycles.

        Parameters
        ----------
        v_a, v_b, v_c : float
            Phase voltages w.r.t. virtual neutral [V].

        Returns
        -------
        numpy.ndarray, shape (3,), dtype float32
            [duty_a, duty_b, duty_c] in [0, 1].
        """
        cdef real32_T da, db, dc
        SVPWM_Compute(&self._blk, v_a, v_b, v_c, &da, &db, &dc)
        cdef cnp.ndarray[cnp.float32_t, ndim=1] out = np.empty(3, dtype=np.float32)
        out[0] = da
        out[1] = db
        out[2] = dc
        return out

    def compute_array(self,
                      cnp.ndarray[cnp.float32_t, ndim=1] vabc not None):
        """
        Compute from a packed [v_a, v_b, v_c] array.

        Parameters
        ----------
        vabc : np.ndarray, shape (3,), dtype float32

        Returns
        -------
        np.ndarray, shape (3,), dtype float32
        """
        return self.compute(vabc[0], vabc[1], vabc[2])

    @property
    def duty_a(self):
        return self._blk.duty_a

    @property
    def duty_b(self):
        return self._blk.duty_b

    @property
    def duty_c(self):
        return self._blk.duty_c

    @property
    def v_dc(self):
        return self._blk.v_dc

    @v_dc.setter
    def v_dc(self, float value):
        self._blk.v_dc = value
