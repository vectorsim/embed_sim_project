# embedsim_control_wrapper.pyx
# =============================================================================
# EmbedSim Control Wrapper -- Sensor-Based PMSM Control
#
# Thin Cython wrapper around embed_sim_control.h:
#   void EmbedSim_ControlInit(void)
#   void EmbedSim_ControlStep(EmbedSimCtrlInput_T*, EmbedSimCtrlOutput_T*)
#
# ALSO exposes transforms and SVPWM directly from C for Python control:
#   clarke(u, v, w) -> (alpha, beta)
#   park(alpha, beta, theta) -> (d, q)
#   inv_park(d, q, theta) -> (alpha, beta)
#   inv_clarke(alpha, beta) -> (u, v, w)
#   svm_calc(v_alpha, v_beta, vdc) -> (ta, tb, tc, sector)
#   svm_calc_dq(v_d, v_q, theta, vdc) -> (ta, tb, tc, sector)
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

    void Transform_Init() nogil
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


cdef extern from "embed_sim_sv_pwm.h":
    ctypedef enum SVM_Sector_T:
        SVM_SECTOR_I
        SVM_SECTOR_II
        SVM_SECTOR_III
        SVM_SECTOR_IV
        SVM_SECTOR_V
        SVM_SECTOR_VI

    ctypedef struct SVM_DutyCycle_T:
        float Ta
        float Tb
        float Tc
        SVM_Sector_T Sector

    void SVM_Init() nogil
    int SVM_CalculateDutyCycleFromAlphaBeta(
        const FocAlphaBeta_T* V_AlphaBeta_P,
        const FocAngle_T* Angle_P,
        float Vdc,
        SVM_DutyCycle_T* DutyOut_P
    ) nogil
    int SVM_CalculateDutyCycleFromDq(
        const FocDq_T* V_Dq_P,
        const FocAngle_T* Angle_P,
        float Vdc,
        SVM_DutyCycle_T* DutyOut_P
    ) nogil


cdef extern from "embed_sim_control.h":
    ctypedef enum EmbedSimCtrl_T:
        SIM_CTRL_OPEN_LOOP
        SIM_CTRL_DFC

    ctypedef struct EmbedSimCtrlInput_T:
        float    AngularVelocityRefRpm
        float    RotorPositionRef
        float    AngularVelocityRef
        float    AngularAccerlerationRef
        float    AngularJerkRef
        float    DutyU
        float    DutyV
        float    DutyW
        float    Iu
        float    Iv
        float    Iw
        float    RotorSpeedSensor
        float    RotorSpeedEst
        float    SampleTime
        float    RotorPositionSensor
        float    RotorPositionEst
        unsigned int   SwitchToClosedLoop
        unsigned int   Valid
        unsigned int   CtrlAlg
        float    Vdc

    ctypedef struct EmbedSimCtrlOutput_T:
        float    DutyU
        float    DutyV
        float    DutyW
        unsigned int   SvmSector
        float    RotorSpeedEst
        float    RotorPositionEst
        unsigned int   Valid

    void EmbedSim_ControlInit() nogil
    void EmbedSim_ControlStep(
        EmbedSimCtrlInput_T* InputPtr,
        EmbedSimCtrlOutput_T* OutputPtr
    ) nogil


# =============================================================================
# Initialize transforms on module load
# =============================================================================

def _init_transforms():
    """Initialize transforms - called on module import"""
    with nogil:
        Transform_Init()
    print("[Wrapper] Transform_Init() called")

# Call initialization when module loads
_init_transforms()


# =============================================================================
# Global initialization state
# =============================================================================

_CONTROL_INIT = False


# =============================================================================
# Transforms - Pure C calls
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
        raise RuntimeError(f"Clarke failed with status: {status}, u={u}, v={v}, w={w}")

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
        raise RuntimeError(f"Park failed with status: {status}, alpha={alpha}, beta={beta}, theta={theta}")

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
        raise RuntimeError(f"Inverse Park failed with status: {status}, d={d}, q={q}, theta={theta}")

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
        raise RuntimeError(f"Inverse Clarke failed with status: {status}, alpha={alpha}, beta={beta}")

    return uvw.U, uvw.V, uvw.W


def svm_calc(float v_alpha, float v_beta, float vdc):
    """SVPWM: AlphaBeta -> duty cycles"""
    cdef FocAlphaBeta_T v_ab
    cdef FocAngle_T angle
    cdef SVM_DutyCycle_T duty
    cdef int status

    v_ab.Alpha = v_alpha
    v_ab.Beta = v_beta
    angle.ThetaE = 0.0

    with nogil:
        status = SVM_CalculateDutyCycleFromAlphaBeta(&v_ab, &angle, vdc, &duty)

    if status != 0:
        raise RuntimeError(f"SVPWM failed with status: {status}, v_alpha={v_alpha}, v_beta={v_beta}, vdc={vdc}")

    return float(duty.Ta), float(duty.Tb), float(duty.Tc), int(duty.Sector) + 1


def svm_calc_dq(float v_d, float v_q, float theta, float vdc):
    """SVPWM: DQ -> duty cycles"""
    cdef FocDq_T v_dq
    cdef FocAngle_T angle
    cdef SVM_DutyCycle_T duty
    cdef int status

    cdef float vmax = vdc / 1.73205080757
    cdef float vmag = (v_d * v_d + v_q * v_q) ** 0.5

    if vmag > vmax and vmax > 0.0:
        scale = vmax / vmag
        v_d = v_d * scale
        v_q = v_q * scale

    v_dq.D = v_d
    v_dq.Q = v_q
    angle.ThetaE = theta

    with nogil:
        status = SVM_CalculateDutyCycleFromDq(&v_dq, &angle, vdc, &duty)

    if status != 0:
        raise RuntimeError(f"SVPWM failed with status: {status}, vd={v_d}, vq={v_q}, theta={theta}, vdc={vdc}")

    return float(duty.Ta), float(duty.Tb), float(duty.Tc), int(duty.Sector) + 1


def control_init():
    """Initialize the control module."""
    global _CONTROL_INIT
    if not _CONTROL_INIT:
        with nogil:
            EmbedSim_ControlInit()
        _CONTROL_INIT = True
        print("[Control] EmbedSim_ControlInit() called")


def control_step(
    speed_ref_rpm: float,
    ia: float,
    ib: float,
    ic: float,
    position_rad: float,
    speed_rpm: float,
    vdc: float = 17.0,
    dt: float = 50e-6,
    ctrl_alg: int = 1,
    valid: int = 1,
) -> dict:
    """Execute one control step"""
    control_init()

    cdef EmbedSimCtrlInput_T inp
    cdef EmbedSimCtrlOutput_T out

    inp.AngularVelocityRefRpm = speed_ref_rpm
    inp.RotorPositionRef = 0.0
    inp.AngularVelocityRef = 0.0
    inp.AngularAccerlerationRef = 0.0
    inp.AngularJerkRef = 0.0
    inp.DutyU = 0.5
    inp.DutyV = 0.5
    inp.DutyW = 0.5
    inp.Iu = ia
    inp.Iv = ib
    inp.Iw = ic
    inp.RotorSpeedSensor = speed_rpm
    inp.RotorSpeedEst = 0.0
    inp.SampleTime = dt
    inp.RotorPositionSensor = position_rad
    inp.RotorPositionEst = 0.0
    inp.SwitchToClosedLoop = 0
    inp.Valid = valid
    inp.CtrlAlg = ctrl_alg
    inp.Vdc = vdc

    with nogil:
        EmbedSim_ControlStep(&inp, &out)

    return {
        'ta': float(out.DutyU),
        'tb': float(out.DutyV),
        'tc': float(out.DutyW),
        'speed_est': float(out.RotorSpeedEst),
        'position_est': float(out.RotorPositionEst),
        'sector': int(out.SvmSector),
        'valid': int(out.Valid),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__version__ = "1.0.0"
__all__ = [
    'clarke', 'park', 'inv_park', 'inv_clarke',
    'svm_calc', 'svm_calc_dq',
    'control_init', 'control_step',
]