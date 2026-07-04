# dfc_controller_wrapper.pyx
# =============================================================================
# EmbedSim Differential Flatness Controller -- Cython wrapper
#
# Wraps the sensorless DFC v4 C implementation (embed_sim_dfc_controller.c):
#   SMO angle/speed estimation, ALIGN -> I-f OPENLOOP -> CLOSEDLOOP startup,
#   flatness voltage law, SVPWM integrated into DFC_Step().
#
# Also surfaces the canonical Clarke/Park transforms
# (embed_sim_coordinate_transform.c) as module-level functions so the pure
# Python plant uses the SAME code that runs on the AURIX.
#
# v4.3.0: DFC_LoopOption_T wired through — Option A (DFC_LOOP_OPENLOOP,
#         I-f hold, no SMO handover) / Option B (DFC_LOOP_CLOSEDLOOP, full
#         closed loop).  set_loop_option()/get_loop_option() added; the
#         option is written into DFC_Input_T on every populate.
# =============================================================================

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, atan2, cosf, sinf

cnp.import_array()

# =============================================================================
# C declarations — EmbedSim matrix / FOC types
# =============================================================================
cdef extern from "embed_sim_matrix.h":
    ctypedef float MatrixFloat
    ctypedef enum MatrixStatus_Type:
        MATRIX_SUCCESS

cdef extern from "embed_sim_foc_types.h":
    ctypedef struct FocUvw_T:
        MatrixFloat U
        MatrixFloat V
        MatrixFloat W
    ctypedef struct FocAlphaBeta_T:
        MatrixFloat Alpha
        MatrixFloat Beta
    ctypedef struct FocDq_T:
        MatrixFloat D
        MatrixFloat Q
    ctypedef struct FocAngle_T:
        MatrixFloat ThetaE

cdef extern from "embed_sim_coordinate_transform.h":
    void Transform_Init() nogil
    MatrixStatus_Type Clarke_Transform_Matrix(
        const FocUvw_T* In_P, FocAlphaBeta_T* Out_P) nogil
    MatrixStatus_Type Park_Transform_Matrix(
        const FocAlphaBeta_T* In_P, const FocAngle_T* Angle_P,
        FocDq_T* Out_P) nogil
    MatrixStatus_Type InvPark_Transform_Matrix(
        const FocDq_T* In_P, const FocAngle_T* Angle_P,
        FocAlphaBeta_T* Out_P) nogil
    MatrixStatus_Type InvClarke_Transform_Matrix(
        const FocAlphaBeta_T* In_P, FocUvw_T* Out_P) nogil

cdef extern from "embed_sim_sv_pwm.h":
    void SVM_Init() nogil          # also calls Transform_Init() internally

# =============================================================================
# C declarations — DFC controller (embed_sim_dfc_controller.h, v4.3.0)
#
# Struct declarations are PARTIAL (only the fields accessed from Cython);
# the generated C includes the real header, so layout/size come from there.
# =============================================================================
cdef extern from "embed_sim_dfc_controller.h":
    # Compile-time default gains (embed_sim_dfc_gains.h macros)
    MatrixFloat DFC_KP_SPEED
    MatrixFloat DFC_KP_ID
    MatrixFloat DFC_KP_IQ
    MatrixFloat DFC_KI_ID
    MatrixFloat DFC_REF_WN
    MatrixFloat DFC_REF_ZETA
    MatrixFloat DFC_P_POLES_F

    ctypedef enum DFC_Mode_T:
        DFC_MODE_ALIGN
        DFC_MODE_OPENLOOP
        DFC_MODE_CLOSEDLOOP

    # v4.3.0 — loop option (0 = A: I-f hold, 1 = B: closed loop)
    ctypedef enum DFC_LoopOption_T:
        DFC_LOOP_OPENLOOP
        DFC_LOOP_CLOSEDLOOP

    ctypedef struct DFC_Smo_T:
        MatrixFloat ThetaE
        MatrixFloat OmegaEFilt

    ctypedef struct DFC_GainSet_T:
        MatrixFloat KpSpeed
        MatrixFloat KpId
        MatrixFloat KpIq
        MatrixFloat KiId
        MatrixFloat RefWn
        MatrixFloat RefZeta

    ctypedef struct DFC_State_T:
        DFC_Smo_T     Smo
        DFC_GainSet_T Gains

    ctypedef struct DFC_Input_T:
        MatrixFloat      SpeedRefRpm
        FocUvw_T         PhaseCurrents
        DFC_LoopOption_T LoopOption      # v4.3.0 — APPENDED field

    ctypedef struct DFC_Output_T:
        MatrixFloat Ta
        MatrixFloat Tb
        MatrixFloat Tc
        MatrixFloat AngularVelocity
        MatrixFloat RotorPosition
        FocUvw_T    PhaseCurrents
        DFC_Mode_T  Mode

    ctypedef struct DFC_Diag_T:
        MatrixFloat    OmegaRefF
        MatrixFloat    OmegaMeas
        MatrixFloat    IqRef
        FocDq_T        IdqMeas
        FocDq_T        VDq
        MatrixFloat    TLoadHat

    MatrixStatus_Type DFC_Init(DFC_State_T* State_P) nogil
    MatrixStatus_Type DFC_Step(DFC_State_T* State_P, const DFC_Input_T* In_P,
                               const MatrixFloat Dt, DFC_Output_T* Out_P) nogil
    MatrixStatus_Type DFC_Reset(DFC_State_T* State_P) nogil
    MatrixStatus_Type DFC_GainSet_Apply(DFC_State_T* State_P,
                                        const DFC_GainSet_T* Gains_P) nogil
    MatrixStatus_Type DFC_GetDiagnostics(const DFC_State_T* State_P,
                                         DFC_Diag_T* Diag_P) nogil

# =============================================================================
# Module-level frame transforms — canonical C implementation
# =============================================================================
def clarke(float u, float v, float w):
    """Clarke transform: UVW -> AlphaBeta (C implementation)."""
    cdef FocUvw_T uvw
    cdef FocAlphaBeta_T ab
    cdef MatrixStatus_Type status
    uvw.U = u
    uvw.V = v
    uvw.W = w
    with nogil:
        status = Clarke_Transform_Matrix(&uvw, &ab)
    if status != MATRIX_SUCCESS:
        raise RuntimeError(f"Clarke transform failed: {status}")
    return ab.Alpha, ab.Beta


def park(float alpha, float beta, float theta):
    """Park transform: AlphaBeta -> DQ (C implementation)."""
    cdef FocAlphaBeta_T ab
    cdef FocAngle_T angle
    cdef FocDq_T dq
    cdef MatrixStatus_Type status
    ab.Alpha = alpha
    ab.Beta = beta
    angle.ThetaE = theta
    with nogil:
        status = Park_Transform_Matrix(&ab, &angle, &dq)
    if status != MATRIX_SUCCESS:
        raise RuntimeError(f"Park transform failed: {status}")
    return dq.D, dq.Q


def inv_park(float d, float q, float theta):
    """Inverse Park transform: DQ -> AlphaBeta (C implementation)."""
    cdef FocDq_T dq
    cdef FocAngle_T angle
    cdef FocAlphaBeta_T ab
    cdef MatrixStatus_Type status
    dq.D = d
    dq.Q = q
    angle.ThetaE = theta
    with nogil:
        status = InvPark_Transform_Matrix(&dq, &angle, &ab)
    if status != MATRIX_SUCCESS:
        raise RuntimeError(f"Inverse Park transform failed: {status}")
    return ab.Alpha, ab.Beta


def inv_clarke(float alpha, float beta):
    """Inverse Clarke transform: AlphaBeta -> UVW (C implementation)."""
    cdef FocAlphaBeta_T ab
    cdef FocUvw_T uvw
    cdef MatrixStatus_Type status
    ab.Alpha = alpha
    ab.Beta = beta
    with nogil:
        status = InvClarke_Transform_Matrix(&ab, &uvw)
    if status != MATRIX_SUCCESS:
        raise RuntimeError(f"Inverse Clarke transform failed: {status}")
    return uvw.U, uvw.V, uvw.W


# =============================================================================
# DFC controller wrapper
# =============================================================================
cdef class DFCControllerWrapper:
    """
    Differential Flatness FOC Controller for NANOTEC DB42S02.

    Sensorless v4: SMO angle/speed, ALIGN -> I-f -> CLOSEDLOOP startup,
    SVPWM integrated in DFC_Step().  All state lives in the C DFC_State_T.

    v4.3.0: loop option — 0 = Option A (I-f hold, no SMO handover) |
    1 = Option B (full closed loop, default).
    """
    cdef:
        DFC_State_T    _state
        DFC_Input_T    _input
        DFC_Output_T   _output
        bint           _initialized
        MatrixFloat    _dt
        int            _loop_option     # v4.3.0: 0 = A | 1 = B
        readonly float ta
        readonly float tb
        readonly float tc
        readonly float speed_est
        readonly float rotor_position
        readonly float iq_ref
        readonly float omega_e
        readonly float omega_smo
        readonly int mode
        readonly int status

    def __cinit__(self, float dt_s = 50.0e-6, int loop_option = 1):
        cdef MatrixStatus_Type status
        self._dt = dt_s
        self._initialized = False
        self._loop_option = 1 if loop_option != 0 else 0
        self.ta = 0.5
        self.tb = 0.5
        self.tc = 0.5
        self.speed_est = 0.0
        self.rotor_position = 0.0
        self.iq_ref = 0.0
        self.omega_e = 0.0
        self.omega_smo = 0.0
        self.mode = 0
        self.status = 0
        self._input.SpeedRefRpm = 0.0
        self._input.PhaseCurrents.U = 0.0
        self._input.PhaseCurrents.V = 0.0
        self._input.PhaseCurrents.W = 0.0
        self._input.LoopOption = <DFC_LoopOption_T>self._loop_option
        with nogil:
            SVM_Init()          # initializes SVPWM + Clarke/Park matrices (idempotent)
            status = DFC_Init(&self._state)
        self.status = <int>status
        self._initialized = True

    def get_default_gains(self):
        """Return compile-time default gains read from the C header macros."""
        return (float(DFC_KP_SPEED), float(DFC_KP_ID), float(DFC_KP_IQ),
                float(DFC_KI_ID), float(DFC_REF_WN), float(DFC_REF_ZETA))

    # -------------------------------------------------------------------------
    # v4.3.0 — loop option (mirrors CddGtm_SetDfcLoopOption on the AURIX)
    # -------------------------------------------------------------------------
    def set_loop_option(self, int option):
        """0 = Option A (I-f hold, no handover) | 1 = Option B (closed loop).

        A -> B mid-run is seamless (the handover gate opens on the next step);
        B -> A after handover requires reset() first (restart from ALIGN),
        matching the CddGtm contract on the AURIX.
        """
        self._loop_option = 1 if option != 0 else 0
        self._input.LoopOption = <DFC_LoopOption_T>self._loop_option

    def get_loop_option(self):
        """Return the active loop option (0 = A, 1 = B)."""
        return self._loop_option

    cpdef void set_inputs_individual(
        self,
        float speed_ref_rpm,
        float ia,
        float ib,
        float ic,
    ) except *:
        """Set controller inputs as individual scalars."""
        self._input.SpeedRefRpm = speed_ref_rpm
        self._input.PhaseCurrents.U = ia
        self._input.PhaseCurrents.V = ib
        self._input.PhaseCurrents.W = ic
        self._input.LoopOption = <DFC_LoopOption_T>self._loop_option   # v4.3.0

    cpdef void compute(self, float dt) except *:
        """Execute one FOC step."""
        cdef MatrixFloat c_dt = dt
        cdef MatrixStatus_Type status
        if not self._initialized:
            raise RuntimeError("DFCControllerWrapper not initialised.")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        with nogil:
            status = DFC_Step(&self._state, &self._input, c_dt, &self._output)
        self.status = <int>status
        self.ta = self._output.Ta
        self.tb = self._output.Tb
        self.tc = self._output.Tc
        self.speed_est = self._output.AngularVelocity
        self.rotor_position = self._output.RotorPosition
        self.mode = <int>self._output.Mode
        self._update_diagnostics()

    cdef void _update_diagnostics(self):
        """Internal: update diagnostic fields from C state."""
        cdef DFC_Diag_T diag
        cdef MatrixStatus_Type status
        with nogil:
            status = DFC_GetDiagnostics(&self._state, &diag)
        if status == MATRIX_SUCCESS:
            self.iq_ref = diag.IqRef
            self.omega_e = diag.OmegaMeas * DFC_P_POLES_F
            self.omega_smo = diag.OmegaMeas

    cpdef void apply_gains(
        self,
        float kp_speed,
        float kp_id,
        float kp_iq,
        float ki_id,
        float ref_wn,
        float ref_zeta,
    ) except *:
        """Apply a runtime gain set (C: DFC_GainSet_Apply())."""
        cdef DFC_GainSet_T gains
        cdef MatrixStatus_Type status
        gains.KpSpeed = kp_speed
        gains.KpId = kp_id
        gains.KpIq = kp_iq
        gains.KiId = ki_id
        gains.RefWn = ref_wn
        gains.RefZeta = ref_zeta
        with nogil:
            status = DFC_GainSet_Apply(&self._state, &gains)
        self.status = <int>status

    cpdef cnp.ndarray get_diagnostics(self):
        """Return diagnostic snapshot as float32 array."""
        cdef DFC_Diag_T diag
        cdef MatrixStatus_Type status
        cdef cnp.ndarray[cnp.float32_t, ndim=1] diag_arr
        with nogil:
            status = DFC_GetDiagnostics(&self._state, &diag)
        self.status = <int>status
        diag_arr = np.empty(10, dtype=np.float32)
        diag_arr[0] = self.speed_est
        diag_arr[1] = diag.IqRef
        diag_arr[2] = diag.IdqMeas.Q
        diag_arr[3] = diag.IdqMeas.D
        diag_arr[4] = self._input.SpeedRefRpm
        diag_arr[5] = diag.OmegaMeas
        diag_arr[6] = diag.OmegaRefF
        diag_arr[7] = diag.VDq.D
        diag_arr[8] = diag.VDq.Q
        diag_arr[9] = <float>self.mode
        return diag_arr

    cpdef void reset(self) except *:
        """Reset all integrators and state."""
        cdef MatrixStatus_Type status
        with nogil:
            status = DFC_Reset(&self._state)
        self.status = <int>status
        self.ta = 0.5
        self.tb = 0.5
        self.tc = 0.5
        self.speed_est = 0.0
        self.rotor_position = 0.0
        self.iq_ref = 0.0
        self.omega_e = 0.0
        self.omega_smo = 0.0
        self.mode = 0

    cpdef float get_theta_e(self):
        """Get estimated electrical angle [rad]."""
        return self._state.Smo.ThetaE

    def __repr__(self):
        return (
            f"DFCControllerWrapper("
            f"ta={self.ta:.3f}, tb={self.tb:.3f}, tc={self.tc:.3f}, "
            f"speed={self.speed_est:.2f} rad/s, mode={self.mode}, "
            f"status={self.status})"
        )


# =============================================================================
# Module-level init — transform tables + SVPWM.
# Must run before any module-level transform call or DFC_Step (idempotent,
# so the additional call in __cinit__ is harmless).
# =============================================================================
SVM_Init()

# =============================================================================
__version__ = "4.3.0"
__author__ = "EmbedSim Team"
