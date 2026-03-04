# coordinate_transform_wrapper.pyx
# =================================
#
# Cython bridge — WORLD 1 (Python simulation only).
# MATRIX OPTIMIZED VERSION - Wraps matrix-based C transforms.
#
# Exposes the C Clarke/Park functions to Python via thin wrapper classes.
# real32_T is typedefed here to match sys_types.h on the C side.
# On the Aurix target this file is never compiled — C files are used directly.
#
# Author : EmbedSim Framework
# Version: 2.0.0

import numpy as np
cimport numpy as cnp

# ─────────────────────────────────────────────────────────────────────────────
# Mirror real32_T from sys_types.h
# ─────────────────────────────────────────────────────────────────────────────
ctypedef float real32_T

# ─────────────────────────────────────────────────────────────────────────────
# Matrix types from Matrix_Operations.h
# ─────────────────────────────────────────────────────────────────────────────
cdef extern from "Matrix_Operations.h":
    ctypedef struct Matrix3x2_T:
        real32_T M[3][2]

    ctypedef struct Matrix2x3_T:
        real32_T M[2][3]

    ctypedef struct Matrix2x2_T:
        real32_T M[2][2]

    ctypedef struct Matrix4x3_T:
        real32_T M[4][3]

    ctypedef struct Matrix3x4_T:
        real32_T M[3][4]

    ctypedef struct Vector3_T:
        real32_T V[3]

    ctypedef struct Vector2_T:
        real32_T V[2]

    ctypedef struct Vector4_T:
        real32_T V[4]

# ─────────────────────────────────────────────────────────────────────────────
# C struct and function declarations from Coordinate_Transform.h
# ─────────────────────────────────────────────────────────────────────────────
cdef extern from "Coordinate_Transform.h":
    # Signal structs
    ctypedef struct Phase3Signal_T:
        real32_T A
        real32_T B
        real32_T C

    ctypedef struct AlphaBetaSignal_T:
        real32_T Alpha
        real32_T Beta

    ctypedef struct DQSignal_T:
        real32_T D
        real32_T Q

    # Matrix types (aliases from header)
    ctypedef Matrix3x2_T ClarkeMatrix_T
    ctypedef Matrix2x2_T ParkMatrix_T

    # Matrix initialization functions
    void Clarke_InitMatrix(ClarkeMatrix_T* pMatrix)
    void InvClarke_InitMatrix(Matrix3x2_T* pMatrix)
    void Park_InitMatrix(ParkMatrix_T* pMatrix, real32_T theta)
    void InvPark_InitMatrix(ParkMatrix_T* pMatrix, real32_T theta)

    # Transform functions (matrix-based signatures)
    void Clarke_Transform(const ClarkeMatrix_T* pMatrix,
                          const Phase3Signal_T* pPhase3SignalIn,
                          AlphaBetaSignal_T* pAlphaBetaSignalOut)

    void InvClarke_Transform(const Matrix3x2_T* pMatrix,
                             const AlphaBetaSignal_T* pAlphaBetaSignalIn,
                             Phase3Signal_T* pPhase3SignalOut)

    void Park_Transform(const ParkMatrix_T* pMatrix,
                        const AlphaBetaSignal_T* pAlphaBetaSignalIn,
                        DQSignal_T* pDQSignalOut)

    void InvPark_Transform(const ParkMatrix_T* pMatrix,
                           const DQSignal_T* pDQSignalIn,
                           AlphaBetaSignal_T* pAlphaBetaSignalOut)

    void ClarkePark_Transform(const ParkMatrix_T* pParkMatrix,
                              const ClarkeMatrix_T* pClarkeMatrix,
                              const Phase3Signal_T* pPhase3SignalIn,
                              DQSignal_T* pDQSignalOut)

# ─────────────────────────────────────────────────────────────────────────────
# Base class for all transform wrappers
# ─────────────────────────────────────────────────────────────────────────────
cdef class TransformWrapperBase:
    cdef ClarkeMatrix_T _clarke_matrix
    cdef Matrix3x2_T _inv_clarke_matrix

    def __cinit__(self):
        # Initialize constant matrices once
        Clarke_InitMatrix(&self._clarke_matrix)
        InvClarke_InitMatrix(&self._inv_clarke_matrix)

    @staticmethod
    cdef inline void phase3_to_vector(const Phase3Signal_T* pIn, Vector3_T* pOut):
        pOut.V[0] = pIn.A
        pOut.V[1] = pIn.B
        pOut.V[2] = pIn.C

    @staticmethod
    cdef inline void alphabeta_to_vector(const AlphaBetaSignal_T* pIn, Vector2_T* pOut):
        pOut.V[0] = pIn.Alpha
        pOut.V[1] = pIn.Beta

    @staticmethod
    cdef inline void dq_to_vector(const DQSignal_T* pIn, Vector2_T* pOut):
        pOut.V[0] = pIn.D
        pOut.V[1] = pIn.Q

# ─────────────────────────────────────────────────────────────────────────────
# Clarke Transform Wrapper  [A, B, C]  ->  [Alpha, Beta]
# ─────────────────────────────────────────────────────────────────────────────
cdef class ClarkeTransformWrapper(TransformWrapperBase):
    cdef Phase3Signal_T     _in
    cdef AlphaBetaSignal_T  _out

    def __cinit__(self):
        # Initialize structs to zero
        self._in.A = 0.0
        self._in.B = 0.0
        self._in.C = 0.0
        self._out.Alpha = 0.0
        self._out.Beta = 0.0

    def set_inputs_abc(self, real32_T a, real32_T b, real32_T c):
        self._in.A = a
        self._in.B = b
        self._in.C = c

    def set_inputs(self, cnp.ndarray u):
        """Set inputs from numpy array [a, b, c]"""
        if u.shape[0] < 3:
            raise ValueError("Input array must have at least 3 elements")
        self._in.A = <real32_T> u[0]
        self._in.B = <real32_T> u[1]
        self._in.C = <real32_T> u[2]

    def compute(self):
        """Execute the Clarke transform using matrix multiplication"""
        Clarke_Transform(&self._clarke_matrix, &self._in, &self._out)

    def get_outputs_alpha_beta(self):
        """Return outputs as tuple (alpha, beta)"""
        return (self._out.Alpha, self._out.Beta)

    def get_outputs(self):
        """Return outputs as numpy array [alpha, beta]"""
        return np.array([self._out.Alpha, self._out.Beta], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Inverse Clarke Transform Wrapper  [Alpha, Beta]  ->  [A, B, C]
# ─────────────────────────────────────────────────────────────────────────────
cdef class InvClarkeTransformWrapper(TransformWrapperBase):
    cdef AlphaBetaSignal_T  _in
    cdef Phase3Signal_T     _out

    def __cinit__(self):
        self._in.Alpha = 0.0
        self._in.Beta = 0.0
        self._out.A = 0.0
        self._out.B = 0.0
        self._out.C = 0.0

    def set_inputs_alpha_beta(self, real32_T alpha, real32_T beta):
        self._in.Alpha = alpha
        self._in.Beta = beta

    def set_inputs(self, cnp.ndarray u):
        """Set inputs from numpy array [alpha, beta]"""
        if u.shape[0] < 2:
            raise ValueError("Input array must have at least 2 elements")
        self._in.Alpha = <real32_T> u[0]
        self._in.Beta = <real32_T> u[1]

    def compute(self):
        """Execute the inverse Clarke transform using matrix multiplication"""
        InvClarke_Transform(&self._inv_clarke_matrix, &self._in, &self._out)

    def get_outputs_abc(self):
        """Return outputs as tuple (a, b, c)"""
        return (self._out.A, self._out.B, self._out.C)

    def get_outputs(self):
        """Return outputs as numpy array [a, b, c]"""
        return np.array([self._out.A, self._out.B, self._out.C], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Park Transform Wrapper  [Alpha, Beta, Theta]  ->  [D, Q]
# ─────────────────────────────────────────────────────────────────────────────
cdef class ParkTransformWrapper(TransformWrapperBase):
    cdef AlphaBetaSignal_T  _in
    cdef ParkMatrix_T       _park_matrix
    cdef DQSignal_T         _out
    cdef real32_T           _theta

    def __cinit__(self):
        self._in.Alpha = 0.0
        self._in.Beta = 0.0
        self._out.D = 0.0
        self._out.Q = 0.0
        self._theta = 0.0
        Park_InitMatrix(&self._park_matrix, self._theta)

    def set_inputs_alpha_beta(self, real32_T alpha, real32_T beta):
        self._in.Alpha = alpha
        self._in.Beta = beta

    def set_inputs(self, cnp.ndarray u):
        """Set alpha-beta inputs from numpy array [alpha, beta]"""
        if u.shape[0] < 2:
            raise ValueError("Input array must have at least 2 elements")
        self._in.Alpha = <real32_T> u[0]
        self._in.Beta = <real32_T> u[1]

    def set_theta(self, real32_T theta):
        """Set rotor angle in radians and update Park matrix"""
        self._theta = theta
        Park_InitMatrix(&self._park_matrix, self._theta)

    def compute(self):
        """Execute the Park transform using matrix multiplication"""
        Park_Transform(&self._park_matrix, &self._in, &self._out)

    def get_outputs_dq(self):
        """Return outputs as tuple (d, q)"""
        return (self._out.D, self._out.Q)

    def get_outputs(self):
        """Return outputs as numpy array [d, q]"""
        return np.array([self._out.D, self._out.Q], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Inverse Park Transform Wrapper  [D, Q, Theta]  ->  [Alpha, Beta]
# ─────────────────────────────────────────────────────────────────────────────
cdef class InvParkTransformWrapper(TransformWrapperBase):
    cdef DQSignal_T         _in
    cdef ParkMatrix_T       _inv_park_matrix
    cdef AlphaBetaSignal_T  _out
    cdef real32_T           _theta

    def __cinit__(self):
        self._in.D = 0.0
        self._in.Q = 0.0
        self._out.Alpha = 0.0
        self._out.Beta = 0.0
        self._theta = 0.0
        InvPark_InitMatrix(&self._inv_park_matrix, self._theta)

    def set_inputs_dq(self, real32_T d, real32_T q):
        self._in.D = d
        self._in.Q = q

    def set_inputs(self, cnp.ndarray u):
        """Set d-q inputs from numpy array [d, q]"""
        if u.shape[0] < 2:
            raise ValueError("Input array must have at least 2 elements")
        self._in.D = <real32_T> u[0]
        self._in.Q = <real32_T> u[1]

    def set_theta(self, real32_T theta):
        """Set rotor angle in radians and update inverse Park matrix"""
        self._theta = theta
        InvPark_InitMatrix(&self._inv_park_matrix, self._theta)

    def compute(self):
        """Execute the inverse Park transform using matrix multiplication"""
        InvPark_Transform(&self._inv_park_matrix, &self._in, &self._out)

    def get_outputs_alpha_beta(self):
        """Return outputs as tuple (alpha, beta)"""
        return (self._out.Alpha, self._out.Beta)

    def get_outputs(self):
        """Return outputs as numpy array [alpha, beta]"""
        return np.array([self._out.Alpha, self._out.Beta], dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# Combined Clarke-Park Transform Wrapper  [A, B, C, Theta]  ->  [D, Q]
# ─────────────────────────────────────────────────────────────────────────────
cdef class ClarkeParkTransformWrapper(TransformWrapperBase):
    cdef Phase3Signal_T     _in
    cdef ParkMatrix_T       _park_matrix
    cdef DQSignal_T         _out
    cdef real32_T           _theta

    def __cinit__(self):
        self._in.A = 0.0
        self._in.B = 0.0
        self._in.C = 0.0
        self._out.D = 0.0
        self._out.Q = 0.0
        self._theta = 0.0
        Park_InitMatrix(&self._park_matrix, self._theta)

    def set_inputs_abc(self, real32_T a, real32_T b, real32_T c):
        self._in.A = a
        self._in.B = b
        self._in.C = c

    def set_inputs(self, cnp.ndarray u):
        """Set three-phase inputs from numpy array [a, b, c]"""
        if u.shape[0] < 3:
            raise ValueError("Input array must have at least 3 elements")
        self._in.A = <real32_T> u[0]
        self._in.B = <real32_T> u[1]
        self._in.C = <real32_T> u[2]

    def set_theta(self, real32_T theta):
        """Set rotor angle in radians and update Park matrix"""
        self._theta = theta
        Park_InitMatrix(&self._park_matrix, self._theta)

    def compute(self):
        """Execute combined Clarke-Park transform using matrix multiplication"""
        ClarkePark_Transform(&self._park_matrix, &self._clarke_matrix,
                             &self._in, &self._out)

    def get_outputs_dq(self):
        """Return outputs as tuple (d, q)"""
        return (self._out.D, self._out.Q)

    def get_outputs(self):
        """Return outputs as numpy array [d, q]"""
        return np.array([self._out.D, self._out.Q], dtype=np.float32)