# svpwm_wrapper.pyx
# =============================================================================
# Cython wrapper for the SVPWM C extension.
# Location: fs_electrical_machines/c_src/svpwm_wrapper.pyx
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# =============================================================================

import numpy as np
cimport numpy as cnp


# ── C declarations ────────────────────────────────────────────────────────────
cdef extern from "svpwm.h":

    ctypedef struct SVPWM_Input:
        float Vref
        float alpha
        float Vdc
        float Ts

    ctypedef struct SVPWM_Output:
        float T1
        float T2
        float T0
        unsigned char sector

    void SVPWM_Init() nogil
    void SVPWM_Step(const SVPWM_Input *u, SVPWM_Output *y) nogil


# ── Cython wrapper class ──────────────────────────────────────────────────────
cdef class SVPWMWrapper:
    """
    Cython wrapper for SVPWM_Step().

    Structs live on the C stack — zero heap allocation on the hot path.

    Usage:
        w = SVPWMWrapper()
        w.set_inputs(Vref=120.0, alpha=0.5, Vdc=400.0, Ts=1e-4)
        w.step()
        T1, T2, T0, sector = w.T1, w.T2, w.T0, w.sector
    """

    cdef SVPWM_Input  _u
    cdef SVPWM_Output _y

    def __cinit__(self):
        self._u.Vref  = 0.0
        self._u.alpha = 0.0
        self._u.Vdc   = 1.0
        self._u.Ts    = 1.0e-4
        self._y.T1    = 0.0
        self._y.T2    = 0.0
        self._y.T0    = 1.0e-4
        self._y.sector = 1

    # ── Input setter ──────────────────────────────────────────────────────────
    cpdef void set_inputs(self,
                          float Vref,
                          float alpha,
                          float Vdc,
                          float Ts):
        self._u.Vref  = Vref
        self._u.alpha = alpha
        self._u.Vdc   = Vdc
        self._u.Ts    = Ts

    # ── Also accept a flat array [Vref, alpha, Vdc] + separate Ts ────────────
    cpdef void set_inputs_array(self,
                                float[::1] u,
                                float Ts):
        """Pack flat input array [Vref, alpha, Vdc] into the input struct."""
        self._u.Vref  = u[0]
        self._u.alpha = u[1]
        self._u.Vdc   = u[2]
        self._u.Ts    = Ts

    # ── C call ────────────────────────────────────────────────────────────────
    cpdef void step(self):
        """Call SVPWM_Step() — GIL released."""
        with nogil:
            SVPWM_Step(&self._u, &self._y)

    # ── Output properties ─────────────────────────────────────────────────────
    @property
    def T1(self) -> float:
        return self._y.T1

    @property
    def T2(self) -> float:
        return self._y.T2

    @property
    def T0(self) -> float:
        return self._y.T0

    @property
    def sector(self) -> int:
        return <int>self._y.sector

    cpdef cnp.ndarray get_outputs(self):
        """Return [T1, T2, T0, sector] as a float32 numpy array."""
        cdef cnp.ndarray y = np.empty(4, dtype=np.float32)
        y[0] = self._y.T1
        y[1] = self._y.T2
        y[2] = self._y.T0
        y[3] = <float>self._y.sector
        return y
