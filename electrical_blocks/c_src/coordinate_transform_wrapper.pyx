# coordinate_transform_wrapper.pyx
# =============================================================================
# Cython wrappers for coordinate_transform.c
# (Clarke, InvClarke, Park, InvPark)
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# =============================================================================

import numpy as np
cimport numpy as cnp

# --------------------------------------------------------------------------
# C declarations
# --------------------------------------------------------------------------
cdef extern from "coordinate_transform.h":
    # Clarke transform
    void clarke_transform_compute_flat(const float * in_buf, float * out_buf) nogil

    # Inverse Clarke transform
    void inv_clarke_transform_compute_flat(const float * in_buf, float * out_buf) nogil

    # Park transform
    void park_transform_compute_flat(const float * in_buf, float theta, float * out_buf) nogil

    # Inverse Park transform
    void inv_park_transform_compute_flat(const float * in_buf, float theta, float * out_buf) nogil


# =============================================================================
# Clarke  abc → αβ
# =============================================================================
cdef class ClarkeTransformWrapper:
    """
    Wrapper for clarke_transform_compute_flat.

    set_inputs(u)  — u = [ia, ib, ic]
    compute()
    get_outputs()  — [alpha, beta]
    """
    cdef float[3] _in
    cdef float[2] _out

    def __cinit__(self):
        for i in range(3): self._in[i] = 0.0
        for i in range(2): self._out[i] = 0.0

    def set_inputs(self, u):
        self._in[0] = <float> u[0]
        self._in[1] = <float> u[1]
        self._in[2] = <float> u[2]

    cpdef void compute(self):
        with nogil:
            clarke_transform_compute_flat(self._in, self._out)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(2, dtype=np.float32)
        y[0] = self._out[0]
        y[1] = self._out[1]
        return y

    @property
    def alpha(self) -> float:
        return self._out[0]
    @property
    def beta(self)  -> float:
        return self._out[1]

# =============================================================================
# Inverse Clarke  αβ → abc
# =============================================================================
cdef class InvClarkeTransformWrapper:
    """
    Wrapper for inv_clarke_transform_compute_flat.

    set_inputs(u)  — u = [alpha, beta]
    compute()
    get_outputs()  — [a, b, c]
    """
    cdef float[2] _in
    cdef float[3] _out

    def __cinit__(self):
        for i in range(2): self._in[i] = 0.0
        for i in range(3): self._out[i] = 0.0

    def set_inputs(self, u):
        self._in[0] = <float> u[0]
        self._in[1] = <float> u[1]

    cpdef void compute(self):
        with nogil:
            inv_clarke_transform_compute_flat(self._in, self._out)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(3, dtype=np.float32)
        y[0] = self._out[0]
        y[1] = self._out[1]
        y[2] = self._out[2]
        return y

# =============================================================================
# Park  αβ → dq
# =============================================================================
cdef class ParkTransformWrapper:
    """
    Wrapper for park_transform_compute_flat.

    set_inputs(u)  — u = [alpha, beta]
    set_theta(th)  — electrical angle [rad]
    compute()
    get_outputs()  — [d, q]
    """
    cdef float[2] _in
    cdef float _theta
    cdef float[2] _out

    def __cinit__(self):
        for i in range(2): self._in[i] = 0.0
        self._theta = 0.0
        for i in range(2): self._out[i] = 0.0

    def set_inputs(self, u):
        self._in[0] = <float> u[0]
        self._in[1] = <float> u[1]

    def set_theta(self, theta):
        self._theta = <float> theta

    cpdef void compute(self):
        with nogil:
            park_transform_compute_flat(self._in, self._theta, self._out)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(2, dtype=np.float32)
        y[0] = self._out[0]
        y[1] = self._out[1]
        return y

# =============================================================================
# Inverse Park  dq → αβ
# =============================================================================
cdef class InvParkTransformWrapper:
    """
    Wrapper for inv_park_transform_compute_flat.

    set_inputs(u)  — u = [d, q]
    set_theta(th)  — electrical angle [rad]
    compute()
    get_outputs()  — [alpha, beta]
    """
    cdef float[2] _in
    cdef float _theta
    cdef float[2] _out

    def __cinit__(self):
        for i in range(2): self._in[i] = 0.0
        self._theta = 0.0
        for i in range(2): self._out[i] = 0.0

    def set_inputs(self, u):
        self._in[0] = <float> u[0]
        self._in[1] = <float> u[1]

    def set_theta(self, theta):
        self._theta = <float> theta

    cpdef void compute(self):
        with nogil:
            inv_park_transform_compute_flat(self._in, self._theta, self._out)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(2, dtype=np.float32)
        y[0] = self._out[0]
        y[1] = self._out[1]
        return y


