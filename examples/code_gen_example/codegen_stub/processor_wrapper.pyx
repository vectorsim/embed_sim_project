# processor_wrapper.pyx
# =================================================================
# Auto-generated Cython wrapper for ControlForge block 'processor'
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

# -- C declarations -----------------------------------------------
cdef extern from "processor.h":

    ctypedef struct InputSignals:
        float sine_a
        float gain_b

    ctypedef struct OutputSignals:
        float summer

    void processor_compute(
        const InputSignals* inp,
        OutputSignals* out
    ) nogil


# -- Cython wrapper class -----------------------------------------
cdef class ProcessorWrapper:
    """
    Cython wrapper for processor.
    Structs live on the C stack - no heap allocation on the hot path.
    """
    cdef InputSignals  _in
    cdef OutputSignals _out

    def __cinit__(self):
        self._in.sine_a = 0.0
        self._in.gain_b = 0.0
        self._out.summer = 0.0

    cpdef void set_inputs(self, float[::1] u):
        """Pack flat float32 input array into InputSignals struct."""
        self._in.sine_a = u[0]
        self._in.gain_b = u[1]

    cpdef void compute(self):
        """Call C function - GIL released."""
        with nogil:
            processor_compute(&self._in, &self._out)

    cpdef cnp.ndarray get_outputs(self):
        """Return output struct as a flat numpy array."""
        cdef cnp.ndarray y = np.empty(1, dtype=np.float32)
        y[0] = self._out.summer
        return y

    # -- Individual output properties (convenience) ----------------
    @property
    def summer(self) -> float:
        return self._out.summer
