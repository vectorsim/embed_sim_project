# three_phase_processor_wrapper.pyx
# =================================================================
# Auto-generated Cython wrapper for ControlForge block 'three_phase_processor'
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp

# -- C declarations -----------------------------------------------
cdef extern from "three_phase_processor.h":

    ctypedef struct InputSignals:
        double source[3]

    ctypedef struct OutputSignals:
        double gain[3]

    void three_phase_processor_compute(
        const InputSignals* inp,
        OutputSignals* out
    ) nogil


# -- Cython wrapper class -----------------------------------------
cdef class ThreePhaseProcessorWrapper:
    """
    Cython wrapper for three_phase_processor.
    Structs live on the C stack - no heap allocation on the hot path.
    """
    cdef InputSignals  _in
    cdef OutputSignals _out

    def __cinit__(self):
        self._in.source[0] = 0.0
        self._in.source[1] = 0.0
        self._in.source[2] = 0.0
        self._out.gain[0] = 0.0
        self._out.gain[1] = 0.0
        self._out.gain[2] = 0.0

    cpdef void set_inputs(self, double[::1] u):
        """Pack flat input array into InputSignals struct."""
        self._in.source[0] = u[0]
        self._in.source[1] = u[1]
        self._in.source[2] = u[2]

    cpdef void compute(self):
        """Call C function - GIL released."""
        with nogil:
            three_phase_processor_compute(&self._in, &self._out)

    cpdef cnp.ndarray get_outputs(self):
        """Return output struct as a flat numpy array."""
        cdef cnp.ndarray y = np.empty(3, dtype=np.float64)
        y[0] = self._out.gain[0]
        y[1] = self._out.gain[1]
        y[2] = self._out.gain[2]
        return y

    # -- Individual output properties (convenience) ----------------
    @property
    def gain(self):
        return [self._out.gain[i] for i in range(3)]
