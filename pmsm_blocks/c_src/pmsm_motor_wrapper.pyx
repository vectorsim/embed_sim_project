# pmsm_motor_wrapper.pyx
# =============================================================================
# Cython wrapper for pmsm_motor.c
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# Compile:
#   python setup_pmsm_motor.py build_ext --inplace
# =============================================================================

import numpy as np
cimport numpy as cnp

# --------------------------------------------------------------------------
# C declarations
# --------------------------------------------------------------------------
cdef extern from "pmsm_motor.h":

    ctypedef struct PmsmMotorParams:
        float Rs
        float Ld
        float Lq
        float psi_pm
        float J
        float B
        float dt

    void pmsm_motor_set_params(float Rs, float Ld, float Lq,
                               float psi_pm, float J, float B, float dt)
    void pmsm_motor_compute_flat(const float* in_buf, float* out_buf) nogil


# --------------------------------------------------------------------------
# Cython wrapper class
# --------------------------------------------------------------------------
cdef class PmsmMotorWrapper:
    """
    Cython wrapper for the PMSM motor C model.

    Hot-path usage (called every simulation step):
        wrapper.set_inputs(u)    # u = [v_alpha, v_beta, T_load]
        wrapper.compute()
        y = wrapper.get_outputs()  # [ia, ib, ic, omega, theta]
    """

    cdef float[3] _in_buf
    cdef float[5] _out_buf

    def __cinit__(self):
        cdef int i
        for i in range(3): self._in_buf[i]  = 0.0
        for i in range(5): self._out_buf[i] = 0.0

    cpdef void set_parameters(self,
                               float Rs, float Ld, float Lq,
                               float psi_pm, float J, float B,
                               float dt=0.0001):
        """Push motor parameters to the C static context."""
        pmsm_motor_set_params(Rs, Ld, Lq, psi_pm, J, B, dt)

    def set_inputs(self, u):
        """
        Pack flat input array into internal buffer.

        u[0] = v_alpha [V]
        u[1] = v_beta  [V]
        u[2] = T_load  [N·m]
        """
        self._in_buf[0] = u[0]
        self._in_buf[1] = u[1]
        self._in_buf[2] = u[2]

    cpdef void compute(self):
        """Call C motor model — GIL released."""
        with nogil:
            pmsm_motor_compute_flat(self._in_buf, self._out_buf)

    cpdef cnp.ndarray get_outputs(self):
        """
        Return output buffer as numpy array.

        out[0] = ia    [A]
        out[1] = ib    [A]
        out[2] = ic    [A]
        out[3] = omega [rad/s]
        out[4] = theta [rad]
        """
        cdef cnp.ndarray y = np.empty(5, dtype=np.float32)
        cdef int i
        for i in range(5):
            y[i] = self._out_buf[i]
        return y

    # -- Convenience properties -----------------------------------------------

    @property
    def ia(self) -> float:    return self._out_buf[0]
    @property
    def ib(self) -> float:    return self._out_buf[1]
    @property
    def ic(self) -> float:    return self._out_buf[2]
    @property
    def omega(self) -> float: return self._out_buf[3]
    @property
    def theta(self) -> float: return self._out_buf[4]
