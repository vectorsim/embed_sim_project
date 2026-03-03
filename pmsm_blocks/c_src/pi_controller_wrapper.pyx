# pi_controller_wrapper.pyx
# =============================================================================
# Cython wrapper for pi_controller.c
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# Compile:
#   python setup_pi_controller.py build_ext --inplace
# =============================================================================

import numpy as np
cimport numpy as cnp

# --------------------------------------------------------------------------
# C declarations
# --------------------------------------------------------------------------
cdef extern from "pi_controller.h":

    void pi_controller_set_params(float Kp, float Ki,
                                   float limit, float dt)
    void pi_controller_compute_flat(const float* in_buf,
                                     float*       out_buf) nogil


# --------------------------------------------------------------------------
# Cython wrapper
# --------------------------------------------------------------------------
cdef class PmsmPiWrapper:
    """
    Cython wrapper for the scalar PI controller C implementation.

    One instance corresponds to one PI controller (the C side uses a
    static context, so do NOT share a single PmsmPiWrapper between two
    Python PIControllerBlock instances — create one wrapper per block).

    Usage:
        wrapper.set_parameters(Kp, Ki, limit, dt)
        wrapper.set_inputs(u)    # u = [error]
        wrapper.compute()
        y = wrapper.get_outputs()  # [control_output]
    """

    cdef float[1] _in
    cdef float[1] _out

    def __cinit__(self):
        self._in[0]  = 0.0
        self._out[0] = 0.0

    cpdef void set_parameters(self, float Kp, float Ki,
                               float limit, float dt=0.0001):
        """Push controller parameters to the C static context."""
        pi_controller_set_params(Kp, Ki, limit, dt)

    def set_inputs(self, u):
        """
        Pack input buffer.

        u[0] = error (reference − feedback)
        """
        self._in[0] = u[0]

    cpdef void compute(self):
        """Call C controller — GIL released."""
        with nogil:
            pi_controller_compute_flat(self._in, self._out)

    cpdef cnp.ndarray get_outputs(self):
        """
        Return output as numpy array.

        out[0] = control output (clamped to ±limit)
        """
        cdef cnp.ndarray y = np.empty(1, dtype=np.float32)
        y[0] = self._out[0]
        return y

    @property
    def output(self) -> float:
        return self._out[0]
