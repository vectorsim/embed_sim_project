# coordinate_transform_wrapper.pyx
# ==================================
# Cython wrapper for the MISRA C coordinate transform library.
# Exposes ClarkeWrapper, ParkWrapper, InvParkWrapper, InvClarkeWrapper.
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
#
# FIX 1: cdef extern from "coordinate_transform.h"  (was "Coordinate_Transform.h")
#         Wrong case = silent include failure on Linux / TriCore GCC FS.
#
# FIX 2: step() scalar arguments are already float — no memoryview issue here.
#         No set_inputs/get_outputs pattern used; direct scalar API kept as-is.

import  numpy as np
cimport numpy as cnp

# ---------------------------------------------------------------------------
# C declarations
# ---------------------------------------------------------------------------
cdef extern from "embed_sim_coordinate_transform.h":   # FIX 1: lowercase

    ctypedef struct Clarke_T:
        pass
    ctypedef struct Park_T:
        pass
    ctypedef struct InvPark_T:
        pass
    ctypedef struct InvClarke_T:
        pass

    void Clarke_Init   (Clarke_T*    pS)
    void Clarke_Step   (Clarke_T*    pS, float ia, float ib, float ic,
                        float* alpha_out, float* beta_out)

    void Park_Init     (Park_T*      pS)
    void Park_Step     (Park_T*      pS, float alpha, float beta, float theta,
                        float* d_out, float* q_out)

    void InvPark_Init  (InvPark_T*   pS)
    void InvPark_Step  (InvPark_T*   pS, float d, float q, float theta,
                        float* alpha_out, float* beta_out)

    void InvClarke_Init(InvClarke_T* pS)
    void InvClarke_Step(InvClarke_T* pS, float alpha, float beta,
                        float* va_out, float* vb_out, float* vc_out)


# ---------------------------------------------------------------------------
# ClarkeWrapper
# ---------------------------------------------------------------------------
cdef class ClarkeWrapper:
    cdef Clarke_T _state
    cdef float    _alpha
    cdef float    _beta

    def __cinit__(self):
        Clarke_Init(&self._state)
        self._alpha = 0.0
        self._beta  = 0.0

    def step(self, float ia, float ib, float ic):
        Clarke_Step(&self._state, ia, ib, ic, &self._alpha, &self._beta)

    def get_outputs(self):
        return (self._alpha, self._beta)

    def reset(self):
        Clarke_Init(&self._state)
        self._alpha = 0.0
        self._beta  = 0.0


# ---------------------------------------------------------------------------
# ParkWrapper
# ---------------------------------------------------------------------------
cdef class ParkWrapper:
    cdef Park_T _state
    cdef float  _d
    cdef float  _q

    def __cinit__(self):
        Park_Init(&self._state)
        self._d = 0.0
        self._q = 0.0

    def step(self, float alpha, float beta, float theta):
        Park_Step(&self._state, alpha, beta, theta, &self._d, &self._q)

    def get_outputs(self):
        return (self._d, self._q)

    def reset(self):
        Park_Init(&self._state)
        self._d = 0.0
        self._q = 0.0


# ---------------------------------------------------------------------------
# InvParkWrapper
# ---------------------------------------------------------------------------
cdef class InvParkWrapper:
    cdef InvPark_T _state
    cdef float     _alpha
    cdef float     _beta

    def __cinit__(self):
        InvPark_Init(&self._state)
        self._alpha = 0.0
        self._beta  = 0.0

    def step(self, float d, float q, float theta):
        InvPark_Step(&self._state, d, q, theta, &self._alpha, &self._beta)

    def get_outputs(self):
        return (self._alpha, self._beta)

    def reset(self):
        InvPark_Init(&self._state)
        self._alpha = 0.0
        self._beta  = 0.0


# ---------------------------------------------------------------------------
# InvClarkeWrapper
# ---------------------------------------------------------------------------
cdef class InvClarkeWrapper:
    cdef InvClarke_T _state
    cdef float       _va
    cdef float       _vb
    cdef float       _vc

    def __cinit__(self):
        InvClarke_Init(&self._state)
        self._va = 0.0
        self._vb = 0.0
        self._vc = 0.0

    def step(self, float alpha, float beta):
        InvClarke_Step(&self._state, alpha, beta,
                       &self._va, &self._vb, &self._vc)

    def get_outputs(self):
        return (self._va, self._vb, self._vc)

    def reset(self):
        InvClarke_Init(&self._state)
        self._va = 0.0
        self._vb = 0.0
        self._vc = 0.0
