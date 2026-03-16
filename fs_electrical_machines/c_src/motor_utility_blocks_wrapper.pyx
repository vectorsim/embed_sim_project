# motor_utility_blocks_wrapper.pyx
# ==================================
# EmbedSim — NANOTEC DB42S02  Open-loop V/f
# Cython wrapper for motor_utility_blocks.c
#
# Follows the exact pattern of smc_wrapper.pyx / speed_pi_wrapper.pyx
# in fs_electrical_machines/c_src/.
#
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import  numpy as np
cimport numpy as cnp

# ── C declarations ─────────────────────────────────────────────────────────────

cdef extern from "motor_utility_blocks.h":

    # Constants
    float MUB_TWO_PI
    float MUB_SQRT3_2

    # SpeedRamp
    ctypedef struct SpeedRamp_T:
        float ramp_value
        float rate
        float target

    void SpeedRamp_Init(SpeedRamp_T *s,
                        float        omega_target,
                        float        ramp_time) nogil

    void SpeedRamp_Step(SpeedRamp_T *s,
                        float        dt,
                        float       *y) nogil

    # VfAngle
    ctypedef struct VfAngle_T:
        float   theta_e
        float   vf_ratio
        float   v_phase_peak
        unsigned char p_poles

    void VfAngle_Init(VfAngle_T *s,
                      float      vf_ratio,
                      float      v_phase_peak,
                      unsigned char p_poles) nogil

    void VfAngle_Step(VfAngle_T   *s,
                      const float *u,
                      float        dt,
                      float       *y) nogil

    # VfDQ
    ctypedef struct VfDQ_T:
        unsigned char _reserved

    void VfDQ_Init(VfDQ_T *s) nogil

    void VfDQ_Step(VfDQ_T      *s,
                   const float *u,
                   float        dt,
                   float       *y) nogil

    # VfTheta
    ctypedef struct VfTheta_T:
        unsigned char _reserved

    void VfTheta_Init(VfTheta_T *s) nogil

    void VfTheta_Step(VfTheta_T   *s,
                      const float *u,
                      float        dt,
                      float       *y) nogil

    # DutyPack
    ctypedef struct DutyPack_T:
        float v_dc

    void DutyPack_Init(DutyPack_T *s,
                       float       v_dc) nogil

    void DutyPack_Step(DutyPack_T  *s,
                       const float *u,
                       float        dt,
                       float       *y) nogil

    # SVPWMPack
    ctypedef struct SVPWMPack_T:
        float v_dc

    void SVPWMPack_Init(SVPWMPack_T *s,
                        float        v_dc) nogil

    void SVPWMPack_Step(SVPWMPack_T *s,
                        const float *u,
                        float        dt,
                        float       *y) nogil


# ── SpeedRampWrapper ───────────────────────────────────────────────────────────

cdef class SpeedRampWrapper:
    """
    Cython wrapper for SpeedRamp_T / SpeedRamp_Step.
    step_func   = 'SpeedRamp_Step'
    state_struct = 'SpeedRamp_T'
    init_func   = 'SpeedRamp_Init'
    NUM_INPUTS  = 0
    OUTPUT_SIZE = 1
    """
    cdef SpeedRamp_T _state
    cdef float[1]    _y

    def __cinit__(self, float omega_target, float ramp_time):
        SpeedRamp_Init(&self._state, omega_target, ramp_time)
        self._y[0] = 0.0

    # set_inputs / compute / get_outputs — EmbedSim SimBlockBase convention
    cpdef void set_inputs(self, double[::1] u):
        """SpeedRamp has no inputs — no-op for API uniformity."""
        pass

    cpdef void compute(self, float dt):
        with nogil:
            SpeedRamp_Step(&self._state, dt, self._y)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(1, dtype=np.float32)
        y[0] = self._y[0]
        return y

    @property
    def omega_ref(self):
        return self._y[0]


# ── VfAngleWrapper ────────────────────────────────────────────────────────────

cdef class VfAngleWrapper:
    """
    Cython wrapper for VfAngle_T / VfAngle_Step.
    step_func    = 'VfAngle_Step'
    state_struct = 'VfAngle_T'
    init_func    = 'VfAngle_Init'
    NUM_INPUTS   = 1
    OUTPUT_SIZE  = 3
    """
    cdef VfAngle_T _state
    cdef float[1]  _u
    cdef float[3]  _y

    def __cinit__(self, float vf_ratio, float v_phase_peak,
                  unsigned char p_poles):
        VfAngle_Init(&self._state, vf_ratio, v_phase_peak, p_poles)
        self._u[0] = 0.0
        self._y[0] = self._y[1] = self._y[2] = 0.0

    cpdef void set_inputs(self, double[::1] u):
        self._u[0] = <float>u[0]   # omega_m_ref [rad/s]

    cpdef void compute(self, float dt):
        with nogil:
            VfAngle_Step(&self._state, self._u, dt, self._y)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(3, dtype=np.float32)
        y[0] = self._y[0]   # v_d
        y[1] = self._y[1]   # v_q
        y[2] = self._y[2]   # theta_e
        return y

    @property
    def v_d(self):    return self._y[0]
    @property
    def v_q(self):    return self._y[1]
    @property
    def theta_e(self): return self._y[2]


# ── VfDQWrapper ───────────────────────────────────────────────────────────────

cdef class VfDQWrapper:
    """
    Cython wrapper for VfDQ_T / VfDQ_Step.
    step_func    = 'VfDQ_Step'
    state_struct = 'VfDQ_T'
    init_func    = 'VfDQ_Init'
    NUM_INPUTS   = 3
    OUTPUT_SIZE  = 2
    """
    cdef VfDQ_T   _state
    cdef float[3] _u
    cdef float[2] _y

    def __cinit__(self):
        VfDQ_Init(&self._state)
        self._u[0] = self._u[1] = self._u[2] = 0.0
        self._y[0] = self._y[1] = 0.0

    cpdef void set_inputs(self, double[::1] u):
        self._u[0] = <float>u[0]   # v_d
        self._u[1] = <float>u[1]   # v_q
        self._u[2] = <float>u[2]   # theta_e

    cpdef void compute(self, float dt):
        with nogil:
            VfDQ_Step(&self._state, self._u, dt, self._y)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(2, dtype=np.float32)
        y[0] = self._y[0]   # v_d
        y[1] = self._y[1]   # v_q
        return y

    @property
    def v_d(self): return self._y[0]
    @property
    def v_q(self): return self._y[1]


# ── VfThetaWrapper ────────────────────────────────────────────────────────────

cdef class VfThetaWrapper:
    """
    Cython wrapper for VfTheta_T / VfTheta_Step.
    step_func    = 'VfTheta_Step'
    state_struct = 'VfTheta_T'
    init_func    = 'VfTheta_Init'
    NUM_INPUTS   = 3
    OUTPUT_SIZE  = 1
    """
    cdef VfTheta_T _state
    cdef float[3]  _u
    cdef float[1]  _y

    def __cinit__(self):
        VfTheta_Init(&self._state)
        self._u[0] = self._u[1] = self._u[2] = 0.0
        self._y[0] = 0.0

    cpdef void set_inputs(self, double[::1] u):
        self._u[0] = <float>u[0]
        self._u[1] = <float>u[1]
        self._u[2] = <float>u[2]   # theta_e

    cpdef void compute(self, float dt):
        with nogil:
            VfTheta_Step(&self._state, self._u, dt, self._y)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(1, dtype=np.float32)
        y[0] = self._y[0]
        return y

    @property
    def theta_e(self): return self._y[0]


# ── DutyPackWrapper ───────────────────────────────────────────────────────────

cdef class DutyPackWrapper:
    """
    Cython wrapper for DutyPack_T / DutyPack_Step.
    step_func    = 'DutyPack_Step'
    state_struct = 'DutyPack_T'
    init_func    = 'DutyPack_Init'
    NUM_INPUTS   = 2
    OUTPUT_SIZE  = 5
    """
    cdef DutyPack_T _state
    cdef float[2]   _u
    cdef float[5]   _y

    def __cinit__(self, float v_dc):
        DutyPack_Init(&self._state, v_dc)
        self._u[0] = self._u[1] = 0.0
        self._y[0] = self._y[1] = self._y[2] = 0.5
        self._y[3] = v_dc
        self._y[4] = 0.0

    cpdef void set_inputs(self, double[::1] u):
        self._u[0] = <float>u[0]   # v_alpha
        self._u[1] = <float>u[1]   # v_beta

    cpdef void compute(self, float dt):
        with nogil:
            DutyPack_Step(&self._state, self._u, dt, self._y)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(5, dtype=np.float32)
        y[0] = self._y[0]   # duty_a
        y[1] = self._y[1]   # duty_b
        y[2] = self._y[2]   # duty_c
        y[3] = self._y[3]   # V_dc
        y[4] = self._y[4]   # T_load
        return y

    @property
    def duty_a(self): return self._y[0]
    @property
    def duty_b(self): return self._y[1]
    @property
    def duty_c(self): return self._y[2]
    @property
    def v_dc(self):   return self._y[3]


# ── SVPWMPackWrapper ──────────────────────────────────────────────────────────

cdef class SVPWMPackWrapper:
    """
    Cython wrapper for SVPWMPack_T / SVPWMPack_Step.
    step_func    = 'SVPWMPack_Step'
    state_struct = 'SVPWMPack_T'
    init_func    = 'SVPWMPack_Init'
    NUM_INPUTS   = 2   [v_alpha, v_beta]
    OUTPUT_SIZE  = 3   [Vref, alpha_angle, V_dc]
    """
    cdef SVPWMPack_T _state
    cdef float[2]    _u
    cdef float[3]    _y

    def __cinit__(self, float v_dc):
        SVPWMPack_Init(&self._state, v_dc)
        self._u[0] = self._u[1] = 0.0
        self._y[0] = self._y[1] = 0.0
        self._y[2] = v_dc

    cpdef void set_inputs(self, double[::1] u):
        self._u[0] = <float>u[0]   # v_alpha
        self._u[1] = <float>u[1]   # v_beta

    cpdef void compute(self, float dt):
        with nogil:
            SVPWMPack_Step(&self._state, self._u, dt, self._y)

    cpdef cnp.ndarray get_outputs(self):
        cdef cnp.ndarray y = np.empty(3, dtype=np.float32)
        y[0] = self._y[0]   # Vref
        y[1] = self._y[1]   # alpha_angle
        y[2] = self._y[2]   # V_dc
        return y

    @property
    def vref(self):        return self._y[0]
    @property
    def alpha_angle(self): return self._y[1]
    @property
    def v_dc(self):        return self._y[2]
