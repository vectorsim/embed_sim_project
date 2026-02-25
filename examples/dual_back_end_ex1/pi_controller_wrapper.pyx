# pi_controller_wrapper.pyx
# Auto-generated Cython wrapper for EmbedSim block 'pi_controller'
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True

import numpy as np
cimport numpy as cnp

cdef extern from "pi_controller.h":
    ctypedef struct InputSignals:
        float error
    ctypedef struct OutputSignals:
        float control_output
    ctypedef struct StateSignals:
        float integral
    void pi_controller_init(float Kp, float Ki)
    void pi_controller_compute(const InputSignals * inp,
                               OutputSignals * out,
                               StateSignals *  state,
                               float dt) nogil
    void pi_controller_reset(StateSignals * state)

cdef class PIControllerWrapper:
    cdef InputSignals  _in
    cdef OutputSignals _out
    cdef StateSignals  _state
    cdef float         _dt

    def __cinit__(self, float Kp, float Ki):
        pi_controller_init(Kp, Ki)
        self._in.error = 0.0
        self._out.control_output = 0.0
        self._state.integral = 0.0
        self._dt = 0.001

    cpdef void set_inputs(self, float[::1] u):
        self._in.error = u[0]

    cpdef void set_dt(self, float dt):
        self._dt = dt

    cpdef void compute(self):
        with nogil:
            pi_controller_compute(&self._in, &self._out, &self._state, self._dt)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(1, dtype=np.float32)
        y[0] = self._out.control_output
        return y

    cpdef void reset(self):
        pi_controller_reset(&self._state)
