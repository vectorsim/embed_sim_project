# embedsim_control_wrapper.pyx
# =============================================================================
# EmbedSim Control Wrapper -- Sensor-Based PMSM Control
# =============================================================================

import numpy as np
cimport numpy as cnp

cnp.import_array()

# =============================================================================
# C declarations
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
    ) nogil
    int Park_Transform_Matrix(
        const FocAlphaBeta_T* In_P,
        const FocAngle_T* Angle_P,
        FocDq_T* Out_P
    ) nogil
    int InvPark_Transform_Matrix(
        const FocDq_T* In_P,
        const FocAngle_T* Angle_P,
        FocAlphaBeta_T* Out_P
    ) nogil
    int InvClarke_Transform_Matrix(
        const FocAlphaBeta_T* In_P,
        FocUvw_T* Out_P
    ) nogil


cdef extern from "embed_sim_cython_interface.h":
    void EmbedSim_CythonControlInit() nogil
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
    ) nogil


# =============================================================================
# Public API - Control Functions
# =============================================================================

def control_init():
    """
    Initialize C control module. Call ONCE at startup.
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
    Execute one control step using the Cython interface.

    Parameters:
    -----------
    ia : float
        Phase A current [A]
    ib : float
        Phase B current [A]
    ic : float
        Phase C current [A]
    rotor_position_rad : float
        Rotor position sensor reading [RAD]
    rotor_velocity_rpm : float
        Rotor velocity sensor reading [RPM Mechanical]
    speed_ref_rpm : float
        Angular velocity reference [RPM Mechanical]
    vdc : float
        DC bus voltage [V]
    sample_time : float
        Sample time [s]
    ctrl_alg : int
        Control algorithm (0=Open Loop, 1=DFC)
    valid_in : int
        Valid flag (1=valid, 0=invalid)

    Returns:
    --------
    dict:
        - 'pwm_u': float - PWM duty cycle for phase U [0.0 ... 1.0]
        - 'pwm_v': float - PWM duty cycle for phase V [0.0 ... 1.0]
        - 'pwm_w': float - PWM duty cycle for phase W [0.0 ... 1.0]
        - 'valid_out': int - Output valid flag
    """
    cdef float pwm_u = 0.0
    cdef float pwm_v = 0.0
    cdef float pwm_w = 0.0
    cdef unsigned int valid_out = 0
    cdef unsigned int valid_in_c = <unsigned int>valid_in
    cdef unsigned int ctrl_alg_c = <unsigned int>ctrl_alg

    with nogil:
        EmbedSim_CythonControlStep(
            ia,
            ib,
            ic,
            rotor_position_rad,
            rotor_velocity_rpm,
            speed_ref_rpm,
            vdc,
            sample_time,
            ctrl_alg_c,
            valid_in_c,
            &pwm_u,
            &pwm_v,
            &pwm_w,
            &valid_out
        )

    return {
        'pwm_u': float(pwm_u),
        'pwm_v': float(pwm_v),
        'pwm_w': float(pwm_w),
        'valid_out': int(valid_out),
    }


# =============================================================================
# Public API - Transform Functions (for PMSM Python Plant)
# =============================================================================

def clarke(float u, float v, float w):
    """Clarke transform: UVW -> AlphaBeta"""
    cdef FocUvw_T uvw
    cdef FocAlphaBeta_T ab
    cdef int status

    uvw.U = u
    uvw.V = v
    uvw.W = w

    with nogil:
        status = Clarke_Transform_Matrix(&uvw, &ab)

    if status != 0:
        raise RuntimeError(f"Clarke failed with status: {status}")

    return ab.Alpha, ab.Beta


def park(float alpha, float beta, float theta):
    """Park transform: AlphaBeta -> DQ"""
    cdef FocAlphaBeta_T ab
    cdef FocAngle_T angle
    cdef FocDq_T dq
    cdef int status

    ab.Alpha = alpha
    ab.Beta = beta
    angle.ThetaE = theta

    with nogil:
        status = Park_Transform_Matrix(&ab, &angle, &dq)

    if status != 0:
        raise RuntimeError(f"Park failed with status: {status}")

    return dq.D, dq.Q


def inv_park(float d, float q, float theta):
    """Inverse Park transform: DQ -> AlphaBeta"""
    cdef FocDq_T dq
    cdef FocAngle_T angle
    cdef FocAlphaBeta_T ab
    cdef int status

    dq.D = d
    dq.Q = q
    angle.ThetaE = theta

    with nogil:
        status = InvPark_Transform_Matrix(&dq, &angle, &ab)

    if status != 0:
        raise RuntimeError(f"Inverse Park failed with status: {status}")

    return ab.Alpha, ab.Beta


def inv_clarke(float alpha, float beta):
    """Inverse Clarke transform: AlphaBeta -> UVW"""
    cdef FocAlphaBeta_T ab
    cdef FocUvw_T uvw
    cdef int status

    ab.Alpha = alpha
    ab.Beta = beta

    with nogil:
        status = InvClarke_Transform_Matrix(&ab, &uvw)

    if status != 0:
        raise RuntimeError(f"Inverse Clarke failed with status: {status}")

    return uvw.U, uvw.V, uvw.W


# =============================================================================
# EXPORTS - No circular imports!
# =============================================================================

__version__ = "1.0.0"
__all__ = [
    'control_init',
    'control_step',
    'clarke',
    'park',
    'inv_park',
    'inv_clarke',
]