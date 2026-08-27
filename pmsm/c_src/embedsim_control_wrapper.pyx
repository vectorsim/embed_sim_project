# =============================================================================
# embedsim_control_wrapper.pyx
# =============================================================================
# EmbedSim Cython Control Wrapper
#
# Native controller:
#   embed_sim_control.c
#   embed_sim_dfc_controller.c
#
# Native transforms:
#   embed_sim_coordinate_transform.c
#
# Native PWM:
#   embed_sim_sv_pwm.c
#
# Cython adapter:
#   embed_sim_cython_interface.c
# =============================================================================

import numpy as np
cimport numpy as cnp

cnp.import_array()


# =============================================================================
# Coordinate transform declarations
# =============================================================================

cdef extern from "embed_sim_coordinate_transform.h":

    ctypedef struct FocUvw_T:
        float U
        float V
        float W

    ctypedef struct FocAlphaBeta_T:
        float Alpha
        float Beta

    ctypedef struct FocDq_T:
        float D
        float Q

    ctypedef struct FocAngle_T:
        float ThetaE

    int Clarke_Transform_Matrix(
        const FocUvw_T* In_P,
        FocAlphaBeta_T* Out_P
    ) noexcept nogil

    int Park_Transform_Matrix(
        const FocAlphaBeta_T* In_P,
        const FocAngle_T* Angle_P,
        FocDq_T* Out_P
    ) noexcept nogil

    int InvPark_Transform_Matrix(
        const FocDq_T* In_P,
        const FocAngle_T* Angle_P,
        FocAlphaBeta_T* Out_P
    ) noexcept nogil

    int InvClarke_Transform_Matrix(
        const FocAlphaBeta_T* In_P,
        FocUvw_T* Out_P
    ) noexcept nogil


# =============================================================================
# Native control interface
# =============================================================================

cdef extern from "embed_sim_cython_interface.h":

    void EmbedSim_CythonControlInit() noexcept nogil

    void EmbedSim_CythonControlStep(
        float Iu,
        float Iv,
        float Iw,
        float RotorPositionSensor,
        float RotorVelocitySensor,
        float AngularVelocityRefRpm,
        float Vdc,
        float SampleTime,
        unsigned int CtrlAlg,
        unsigned int ValidIn,
        float* PwmU,
        float* PwmV,
        float* PwmW,
        unsigned int* ValidOut
    ) noexcept nogil


# =============================================================================
# Native state structures – exactly matching embed_sim_control.h
# =============================================================================

cdef extern from "embed_sim_control.h":

    ctypedef struct EmbedSimMotorState_T:
        # Mechanical
        float SpeedRpm
        float PositionRad

        # PWM
        float DutyU
        float DutyV
        float DutyW
        unsigned int SvmSector

        # Status
        unsigned int Valid
        unsigned long long LoopCounter
        unsigned int SwitchToClosedLoop


cdef extern from "embed_sim_cython_interface.h":

    void EmbedSim_CythonGetMotorState(
        EmbedSimMotorState_T* StatePtr
    ) noexcept nogil


# =============================================================================
# C structure -> Python dictionary
# =============================================================================

cdef dict _motor_state_to_dict(
    EmbedSimMotorState_T* state
):
    return {
        'speed_rpm': float(state.SpeedRpm),
        'position_rad': float(state.PositionRad),
        'duty_u': float(state.DutyU),
        'duty_v': float(state.DutyV),
        'duty_w': float(state.DutyW),
        'svm_sector': int(state.SvmSector),
        'valid': int(state.Valid),
        'loop_counter': int(state.LoopCounter),
        'switch_to_closed_loop': int(state.SwitchToClosedLoop),
    }


# =============================================================================
# Public API
# =============================================================================

def control_init():
    """
    Initialize the native EmbedSim controller.

    Call once before the first control_step().
    """

    with nogil:
        EmbedSim_CythonControlInit()


def control_step(
    ia: float,
    ib: float,
    ic: float,
    rotor_position_rad: float,
    rotor_velocity_rpm: float,
    speed_ref_rpm: float,
    vdc: float = 12.0,
    sample_time: float = 50e-6,
    ctrl_alg: int = 1,
    valid_in: int = 1,
) -> dict:
    """
    Execute one native controller cycle.

    Returns:
        pwm_u
        pwm_v
        pwm_w
        valid_out
        motor_state
    """

    cdef float pwm_u = 0.5
    cdef float pwm_v = 0.5
    cdef float pwm_w = 0.5

    cdef unsigned int valid_out = 0

    cdef unsigned int valid_in_c = \
        <unsigned int>valid_in

    cdef unsigned int ctrl_alg_c = \
        <unsigned int>ctrl_alg

    cdef EmbedSimMotorState_T motor_state

    with nogil:

        EmbedSim_CythonControlStep(
            <float>ia,
            <float>ib,
            <float>ic,
            <float>rotor_position_rad,
            <float>rotor_velocity_rpm,
            <float>speed_ref_rpm,
            <float>vdc,
            <float>sample_time,
            ctrl_alg_c,
            valid_in_c,
            &pwm_u,
            &pwm_v,
            &pwm_w,
            &valid_out
        )

        EmbedSim_CythonGetMotorState(
            &motor_state
        )

    return {
        'pwm_u': float(pwm_u),
        'pwm_v': float(pwm_v),
        'pwm_w': float(pwm_w),
        'valid_out': int(valid_out),
        'motor_state': _motor_state_to_dict(&motor_state),
    }


def get_motor_state() -> dict:
    """
    Return the current native motor state without executing a controller step.
    """

    cdef EmbedSimMotorState_T motor_state

    with nogil:

        EmbedSim_CythonGetMotorState(
            &motor_state
        )

    return _motor_state_to_dict(
        &motor_state
    )


# =============================================================================
# Clarke
# =============================================================================

def clarke(float u, float v, float w):

    cdef FocUvw_T uvw
    cdef FocAlphaBeta_T ab
    cdef int status

    uvw.U = u
    uvw.V = v
    uvw.W = w

    with nogil:

        status = Clarke_Transform_Matrix(
            &uvw,
            &ab
        )

    if status != 0:
        raise RuntimeError(
            f"Clarke failed with status: {status}"
        )

    return (
        float(ab.Alpha),
        float(ab.Beta)
    )


# =============================================================================
# Park
# =============================================================================

def park(
    float alpha,
    float beta,
    float theta
):

    cdef FocAlphaBeta_T ab
    cdef FocAngle_T angle
    cdef FocDq_T dq
    cdef int status

    ab.Alpha = alpha
    ab.Beta = beta

    angle.ThetaE = theta

    with nogil:

        status = Park_Transform_Matrix(
            &ab,
            &angle,
            &dq
        )

    if status != 0:
        raise RuntimeError(
            f"Park failed with status: {status}"
        )

    return (
        float(dq.D),
        float(dq.Q)
    )


# =============================================================================
# Inverse Park
# =============================================================================

def inv_park(
    float d,
    float q,
    float theta
):

    cdef FocDq_T dq
    cdef FocAngle_T angle
    cdef FocAlphaBeta_T ab
    cdef int status

    dq.D = d
    dq.Q = q

    angle.ThetaE = theta

    with nogil:

        status = InvPark_Transform_Matrix(
            &dq,
            &angle,
            &ab
        )

    if status != 0:
        raise RuntimeError(
            f"Inverse Park failed with status: {status}"
        )

    return (
        float(ab.Alpha),
        float(ab.Beta)
    )


# =============================================================================
# Inverse Clarke
# =============================================================================

def inv_clarke(
    float alpha,
    float beta
):

    cdef FocAlphaBeta_T ab
    cdef FocUvw_T uvw
    cdef int status

    ab.Alpha = alpha
    ab.Beta = beta

    with nogil:

        status = InvClarke_Transform_Matrix(
            &ab,
            &uvw
        )

    if status != 0:
        raise RuntimeError(
            f"Inverse Clarke failed with status: {status}"
        )

    return (
        float(uvw.U),
        float(uvw.V),
        float(uvw.W)
    )


# =============================================================================
# Module information
# =============================================================================

__version__ = "2.1.0"

__all__ = [
    'control_init',
    'control_step',
    'get_motor_state',
    'clarke',
    'park',
    'inv_park',
    'inv_clarke',
]