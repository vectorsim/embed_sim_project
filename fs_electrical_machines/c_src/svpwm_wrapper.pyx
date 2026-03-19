# svpwm_wrapper.pyx
# =============================================================================
# EmbedSim SVPWM Cython Wrapper for fs_electrical_machines
# =============================================================================
# Cython wrapper for the SVPWM C implementation.
#
# Location: fs_electrical_machines/c_src/svpwm_wrapper.pyx
#
# The compiled module will be available as:
#   from fs_electrical_machines.svpwm_wrapper import EmbedSimSVPWM
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# =============================================================================

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs, sqrt, atan2

# Initialize NumPy C API
cnp.import_array()

# -----------------------------------------------------------------------------
# C declarations from Matrix.h
# FIX #3: Matrix_FloatToQ31 / Matrix_Q31ToFloat belong under Matrix.h,
#         not svpwm.h — they are defined in Matrix.c and declared in Matrix.h.
# -----------------------------------------------------------------------------
cdef extern from "Matrix.h":
    ctypedef int       MatrixElement   # int32_T
    ctypedef float     MatrixFloat

    ctypedef enum MatrixStatus_Type:
        MATRIX_SUCCESS          = 0
        MATRIX_ERROR_NULL_PTR   = 1
        MATRIX_ERROR_DIV_BY_ZERO = 5
        MATRIX_ERROR_OUT_OF_BOUNDS = 8

    MatrixElement Matrix_FloatToQ31(const MatrixFloat value) nogil
    MatrixFloat   Matrix_Q31ToFloat(const MatrixElement value) nogil

# -----------------------------------------------------------------------------
# C declarations from svpwm.h
# FIX #1: Removed all  extern const <macro>  declarations.
#         SVM_2PI_F, SVM_SQRT3_F, SVM_SQRT3_OVER_2_F and Q31_ONE are
#         #define macros — they have no linkage and cannot be declared extern.
#         The literal 6.28318530718f is inlined directly where needed.
# -----------------------------------------------------------------------------
cdef extern from "svpwm.h":

    ctypedef enum SVM_Sector_Type:
        SVM_SECTOR_I   = 0
        SVM_SECTOR_II  = 1
        SVM_SECTOR_III = 2
        SVM_SECTOR_IV  = 3
        SVM_SECTOR_V   = 4
        SVM_SECTOR_VI  = 5

    ctypedef struct SVM_DutyCycle_Type:
        MatrixElement  ta
        MatrixElement  tb
        MatrixElement  tc
        SVM_Sector_Type sector

    MatrixStatus_Type SVM_CalculateDutyCycle(
        const MatrixFloat       modulation_index,
        const MatrixFloat       angle_rad,
        SVM_DutyCycle_Type*     duty) nogil

    MatrixStatus_Type SVM_GetSectorFromDQ(
        const MatrixElement     vd,
        const MatrixElement     vq,
        SVM_Sector_Type*        sector) nogil

    void SVM_GetDutyCyclesFloat(
        const SVM_DutyCycle_Type* duty,
        MatrixFloat*              ta,
        MatrixFloat*              tb,
        MatrixFloat*              tc) nogil


# -----------------------------------------------------------------------------
# Module-level constant (replaces the removed extern macro reference)
# -----------------------------------------------------------------------------
cdef MatrixFloat _TWO_PI = 6.28318530718
cdef MatrixElement _Q31_ONE = 0x7FFFFFFF


# -----------------------------------------------------------------------------
# EmbedSim SVPWM Wrapper Class
# -----------------------------------------------------------------------------
cdef class EmbedSimSVPWM:
    """
    Space Vector PWM Module for Three-Phase Inverters

    Part of the fs_electrical_machines module. Implements SVPWM with Q31
    fixed-point arithmetic for deterministic behaviour in real-time simulations.

    Parameters
    ----------
    timer_period : uint32, optional
        PWM timer period in ticks (default: 10000). Used for converting
        normalised duty cycles to timer compare values.

    Attributes
    ----------
    ta, tb, tc : float
        Phase duty cycles [0, 1]
    sector : int
        Current sector (0-5)
    magnitude : float
        Modulation magnitude (normalised, clamped to 0.95)
    angle : float
        Voltage angle in radians [0, 2π)
    status : int
        Last operation status code (0 = MATRIX_SUCCESS)

    Examples
    --------
    >>> svpwm = EmbedSimSVPWM(timer_period=10000)
    >>> svpwm.calculate(alpha=0.8, beta=0.0)
    >>> print(f"Duty cycles: A={svpwm.ta:.3f}, B={svpwm.tb:.3f}, C={svpwm.tc:.3f}")
    >>> print(f"Sector: {svpwm.sector}")
    """

    cdef:
        SVM_DutyCycle_Type _duty
        MatrixFloat        _magnitude
        MatrixFloat        _angle
        MatrixFloat        _ta_float
        MatrixFloat        _tb_float
        MatrixFloat        _tc_float
        unsigned int       _timer_period   # FIX #2: private C field, safe inside nogil

        # Python-visible read-only attributes
        readonly unsigned int timer_period
        readonly int          sector
        readonly int          status
        readonly float        ta
        readonly float        tb
        readonly float        tc
        readonly float        magnitude
        readonly float        angle

    def __cinit__(self, unsigned int timer_period=10000):
        self._timer_period = timer_period   # C field — nogil safe
        self.timer_period  = timer_period   # Python readonly copy
        self._magnitude    = 0.0
        self._angle        = 0.0
        self._ta_float     = 0.0
        self._tb_float     = 0.0
        self._tc_float     = 0.0
        self.sector        = 0
        self.status        = <int>MATRIX_SUCCESS
        self.ta            = 0.0
        self.tb            = 0.0
        self.tc            = 0.0
        self.magnitude     = 0.0
        self.angle         = 0.0

    # -------------------------------------------------------------------------
    cpdef void calculate(self, float alpha, float beta) except *:
        """
        Calculate duty cycles from alpha-beta voltage components.

        Parameters
        ----------
        alpha : float
            Alpha-axis voltage component (normalised to [-1, 1])
        beta : float
            Beta-axis voltage component (normalised to [-1, 1])
        """
        cdef:
            MatrixFloat       magnitude
            MatrixFloat       angle
            MatrixStatus_Type status

        # Magnitude and angle — pure C math, no Python objects
        magnitude = sqrt(alpha * alpha + beta * beta)
        if magnitude > 0.95:
            magnitude = 0.95

        angle = atan2(beta, alpha)
        if angle < 0.0:
            angle += _TWO_PI

        # Cache for Python attribute access (done outside nogil)
        self._magnitude = magnitude
        self._angle     = angle

        # Core C call — GIL released
        with nogil:
            status = SVM_CalculateDutyCycle(magnitude, angle, &self._duty)

        self.status = <int>status
        self.sector = <int>self._duty.sector

        # Float conversion — GIL released
        with nogil:
            SVM_GetDutyCyclesFloat(
                &self._duty,
                &self._ta_float,
                &self._tb_float,
                &self._tc_float)

        # Update Python-visible readonly floats
        self.ta        = self._ta_float
        self.tb        = self._tb_float
        self.tc        = self._tc_float
        self.magnitude = self._magnitude
        self.angle     = self._angle

    # -------------------------------------------------------------------------
    cpdef void calculate_complex(self, complex voltage) except *:
        """
        Calculate duty cycles from a complex voltage vector.

        Parameters
        ----------
        voltage : complex
            Complex voltage vector where real = alpha, imag = beta
        """
        self.calculate(voltage.real, voltage.imag)

    # -------------------------------------------------------------------------
    cpdef void get_compare_values(self, unsigned int[:] compare) except *:
        """
        Get timer compare values for centre-aligned PWM generation.

        Parameters
        ----------
        compare : unsigned int[3]
            Output array — compare values for phases A, B, C.
            compare[n] = (period - on_time_ticks) >> 1
        """
        cdef:
            unsigned int period   # FIX #2: local C copy — safe inside nogil
            unsigned int ta_ticks
            unsigned int tb_ticks
            unsigned int tc_ticks

        if compare.shape[0] < 3:
            raise ValueError("compare array must have at least 3 elements")

        # FIX #2: copy Python-managed attribute to a C local BEFORE entering nogil
        period = self._timer_period

        with nogil:
            # Q31 → ticks:  ticks = (duty_q31 * period) / Q31_ONE
            ta_ticks = <unsigned int>(
                (<unsigned long long>self._duty.ta * <unsigned long long>period) //
                <unsigned long long>_Q31_ONE)
            tb_ticks = <unsigned int>(
                (<unsigned long long>self._duty.tb * <unsigned long long>period) //
                <unsigned long long>_Q31_ONE)
            tc_ticks = <unsigned int>(
                (<unsigned long long>self._duty.tc * <unsigned long long>period) //
                <unsigned long long>_Q31_ONE)

            # Centre-aligned PWM: compare = (period - on_time) >> 1
            # FIX #2: use >> 1 (explicit C integer shift, unambiguous with cdivision=True)
            compare[0] = (period - ta_ticks) >> 1
            compare[1] = (period - tb_ticks) >> 1
            compare[2] = (period - tc_ticks) >> 1

    # -------------------------------------------------------------------------
    cpdef cnp.ndarray get_outputs(self):
        """
        Return all outputs as a float32 numpy array.

        Returns
        -------
        ndarray, shape (7,), dtype float32
            [ta, tb, tc, sector, status, magnitude, angle]
        """
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y = np.empty(7, dtype=np.float32)

        y[0] = self._ta_float
        y[1] = self._tb_float
        y[2] = self._tc_float
        y[3] = <float>self.sector
        y[4] = <float>self.status
        y[5] = self._magnitude
        y[6] = self._angle
        return y

    # -------------------------------------------------------------------------
    def __repr__(self):
        return (
            f"EmbedSimSVPWM("
            f"ta={self.ta:.3f}, tb={self.tb:.3f}, tc={self.tc:.3f}, "
            f"sector={self.sector}, magnitude={self.magnitude:.3f}, "
            f"angle={self.angle:.4f})"
        )


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
def svpwm_step(float alpha, float beta, unsigned int timer_period=10000):
    """
    Single-step SVPWM calculation (stateless convenience wrapper).

    Parameters
    ----------
    alpha, beta : float
        Alpha/beta voltage components (normalised)
    timer_period : unsigned int
        PWM timer period in ticks

    Returns
    -------
    tuple : (ta, tb, tc, sector)
    """
    cdef EmbedSimSVPWM svpwm = EmbedSimSVPWM(timer_period)
    svpwm.calculate(alpha, beta)
    return svpwm.ta, svpwm.tb, svpwm.tc, svpwm.sector


def svpwm_step_complex(complex voltage, unsigned int timer_period=10000):
    """
    Single-step SVPWM with complex voltage input.

    Parameters
    ----------
    voltage : complex
        Complex voltage vector (real=alpha, imag=beta)
    timer_period : unsigned int
        PWM timer period in ticks

    Returns
    -------
    tuple : (ta, tb, tc, sector)
    """
    return svpwm_step(voltage.real, voltage.imag, timer_period)


# -----------------------------------------------------------------------------
# Version info
# -----------------------------------------------------------------------------
__version__ = "1.1.0"
__author__  = "EmbedSim Team"
