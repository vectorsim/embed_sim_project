"""
motor_utility_blocks.py
=======================
EmbedSim — NANOTEC DB42S02  Open-loop V/f controller blocks.

Five VectorBlock subclasses that wrap motor_utility_blocks.c via
motor_utility_blocks_wrapper.pyx.

Each class carries:
    PYX_FILE     — path to the .pyx file (PYXInspector auto-populates
                   step_func / state_struct / init_func / NUM_INPUTS /
                   OUTPUT_SIZE at subclass-definition time via
                   VectorBlock.__init_subclass__)
    C_SOURCES    — .c files required by LoopGenerator
    C_HEADERS    — .h files required by LoopGenerator

Zero C_CUSTOM_EMIT anywhere in this file.
Zero hand-written C strings.

Build the extension once:
    cd fs_electrical_machines/c_src
    python setup_motor_utility_blocks.py build_ext --inplace

Then use:
    from motor_utility_blocks import (
        SpeedRampBlock, VfAngleBlock, VfDQBlock, VfThetaBlock, DutyPackBlock)
"""

from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import List, Optional

# ── Path to this package directory ────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# ── Wrapper import (compiled .pyd / .so) ──────────────────────────────────────
try:
    from motor_utility_blocks_wrapper import (
        SpeedRampWrapper,
        VfAngleWrapper,
        VfDQWrapper,
        VfThetaWrapper,
        DutyPackWrapper,
        SVPWMPackWrapper,
    )
    _WRAPPER_AVAILABLE = True
except ImportError:
    _WRAPPER_AVAILABLE = False
    SpeedRampWrapper = VfAngleWrapper = VfDQWrapper = None
    VfThetaWrapper   = DutyPackWrapper = SVPWMPackWrapper = None


# =============================================================================
# SpeedRampBlock
# =============================================================================
class SpeedRampBlock(VectorBlock):
    """
    Linear speed ramp: 0 → omega_target [rad/s] over ramp_time [s], hold.

    Output: [omega_m_ref]  shape=(1,)

    Backend
    -------
    use_c_backend=False  → compute_py()  (pure Python, Euler ramp)
    use_c_backend=True   → compute_c()   (SpeedRamp_Step via Cython wrapper)
    """
    PYX_FILE     = str(_HERE / "c_src" / "motor_utility_blocks_wrapper.pyx")
    C_SOURCES    = ["Motor_Utility_Blocks.c"]
    C_HEADERS    = ["Motor_Utility_Blocks.h"]
    step_func    = "SpeedRamp_Step"
    state_struct = "SpeedRamp_T"
    init_func    = "SpeedRamp_Init"
    NUM_INPUTS   = 0
    OUTPUT_SIZE  = 1
    # AurixStepGenerator reads these attribute names from the instance
    # and appends their values to the Init call:
    # SpeedRamp_Init(&state, omega_target, ramp_time)
    C_INIT_ARGS  = ["omega_target", "ramp_time"]

    def __init__(self, name: str,
                 omega_target: float,
                 ramp_time: float,
                 use_c_backend: bool = False) -> None:
        super().__init__(name, use_c_backend=use_c_backend)
        self.omega_target  = float(omega_target)
        self.ramp_time     = float(ramp_time)
        self.is_dynamic    = False
        self.output_label  = "ω_ref"
        self._ramp_state   = 0.0   # Python-path state

        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        if not _WRAPPER_AVAILABLE:
            raise ImportError(
                "motor_utility_blocks_wrapper not found. "
                "Build with: python setup_motor_utility_blocks.py build_ext --inplace"
            )
        self._wrapper = SpeedRampWrapper(
            float(self.omega_target), float(self.ramp_time))

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        if self.ramp_time > 0.0:
            self._ramp_state = min(self._ramp_state + (self.omega_target
                                   / self.ramp_time) * dt, self.omega_target)
        else:
            self._ramp_state = self.omega_target
        self.output = VectorSignal(
            np.array([self._ramp_state], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None
                  ) -> VectorSignal:
        self._wrapper.compute(float(dt))
        self.output = VectorSignal(
            self._wrapper.get_outputs().astype(DEFAULT_DTYPE), self.name)
        return self.output

    def reset(self) -> None:
        super().reset()
        self._ramp_state = 0.0
        if self.use_c_backend and self._wrapper is not None:
            self._load_wrapper()   # re-initialise C state


# =============================================================================
# VfAngleBlock
# =============================================================================
class VfAngleBlock(VectorBlock):
    """
    Open-loop V/f integrator + voltage law.

    Input:  [omega_m_ref]       shape=(1,)
    Output: [v_d, v_q, theta_e] shape=(3,)
      v_d     = 0  (always, for open-loop V/f)
      v_q     = vf_ratio * |omega_e|  clamped to v_phase_peak
      theta_e = integral of omega_e, wrapped to [0, 2π)

    Parameters
    ----------
    vf_ratio     : V/f gain [V·s/rad]   default: V_phase_peak / omega_e_rated
    v_phase_peak : peak phase voltage limit [V]
    p_poles      : pole pairs
    v_boost      : zero-speed voltage boost [V] (default 0.0)
                   Compensates stator resistive drop at low speed.
                   Applied as: v_q = vf_ratio * |omega_e| + v_boost,
                   then clamped to v_phase_peak.
                   Typical value: R_stator * I_nominal (e.g. 0.19 * 1.0 = 0.19 V)
    """
    PYX_FILE     = str(_HERE / "c_src" / "motor_utility_blocks_wrapper.pyx")
    C_SOURCES    = ["motor_utility_blocks.c"]
    C_HEADERS    = ["motor_utility_blocks.h"]
    step_func    = "VfAngle_Step"
    state_struct = "VfAngle_T"
    init_func    = "VfAngle_Init"
    NUM_INPUTS   = 1
    OUTPUT_SIZE  = 3
    # VfAngle_Init(&state, vf_ratio, v_phase_peak, v_boost, p_poles)
    C_INIT_ARGS  = ["vf_ratio", "v_phase_peak", "v_boost", "p_poles"]

    def __init__(self, name: str,
                 vf_ratio: float,
                 v_phase_peak: float,
                 p_poles: int,
                 v_boost: float = 0.0,
                 use_c_backend: bool = False) -> None:
        super().__init__(name, use_c_backend=use_c_backend)
        self.vf_ratio      = float(vf_ratio)
        self.v_phase_peak  = float(v_phase_peak)
        self.p_poles       = int(p_poles)
        self.v_boost       = float(v_boost)
        self.is_dynamic    = False
        self.output_label  = "[v_d,v_q,θ_e]"
        self._theta_e      = 0.0   # Python-path state

        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        if not _WRAPPER_AVAILABLE:
            raise ImportError(
                "motor_utility_blocks_wrapper not found. "
                "Build with: python setup_motor_utility_blocks.py build_ext --inplace"
            )
        self._wrapper = VfAngleWrapper(
            float(self.vf_ratio),
            float(self.v_phase_peak),
            float(self.v_boost),
            self.p_poles)

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        omega_m = float(input_values[0].value[0]) if input_values else 0.0
        omega_e = self.p_poles * omega_m
        v_q     = min(self.vf_ratio * abs(omega_e) + self.v_boost, self.v_phase_peak)
        self._theta_e += omega_e * dt
        self._theta_e  = math.fmod(self._theta_e, 2.0 * math.pi)
        if self._theta_e < 0.0:
            self._theta_e += 2.0 * math.pi
        self.output = VectorSignal(
            np.array([0.0, v_q, self._theta_e], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None
                  ) -> VectorSignal:
        u = np.zeros(1, dtype=np.float32)   # FIX: float32 matches float[::1] in wrapper
        if input_values:
            u[0] = float(input_values[0].value[0])
        self._wrapper.set_inputs(u)
        self._wrapper.compute(float(dt))
        self.output = VectorSignal(
            self._wrapper.get_outputs().astype(DEFAULT_DTYPE), self.name)
        return self.output

    def reset(self) -> None:
        super().reset()
        self._theta_e = 0.0
        if self.use_c_backend and self._wrapper is not None:
            self._load_wrapper()


# =============================================================================
# VfDQBlock
# =============================================================================
class VfDQBlock(VectorBlock):
    """
    Extract [v_d, v_q] from VfAngleBlock output[0:2].

    Input:  [v_d, v_q, theta_e]  shape=(3,)
    Output: [v_d, v_q]           shape=(2,)

    Stateless combinatorial pass-through.
    """
    PYX_FILE     = str(_HERE / "c_src" / "motor_utility_blocks_wrapper.pyx")
    C_SOURCES    = ["motor_utility_blocks.c"]
    C_HEADERS    = ["motor_utility_blocks.h"]
    step_func    = "VfDQ_Step"
    state_struct = "VfDQ_T"
    init_func    = "VfDQ_Init"
    NUM_INPUTS   = 3
    OUTPUT_SIZE  = 2

    def __init__(self, name: str,
                 use_c_backend: bool = False) -> None:
        super().__init__(name, use_c_backend=use_c_backend)
        self.output_label = "[v_d,v_q]"

        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        if not _WRAPPER_AVAILABLE:
            raise ImportError(
                "motor_utility_blocks_wrapper not found.")
        self._wrapper = VfDQWrapper()

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        v = (input_values[0].value if input_values
             else np.zeros(3, dtype=DEFAULT_DTYPE))
        self.output = VectorSignal(
            np.array([v[0], v[1]], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None
                  ) -> VectorSignal:
        u = np.zeros(3, dtype=np.float32)   # FIX: float32 matches float[::1] in wrapper
        if input_values:
            u[:3] = input_values[0].value[:3]
        self._wrapper.set_inputs(u)
        self._wrapper.compute(float(dt))
        self.output = VectorSignal(
            self._wrapper.get_outputs().astype(DEFAULT_DTYPE), self.name)
        return self.output


# =============================================================================
# VfThetaBlock
# =============================================================================
class VfThetaBlock(VectorBlock):
    """
    Extract [theta_e] from VfAngleBlock output[2].

    Input:  [v_d, v_q, theta_e]  shape=(3,)
    Output: [theta_e]            shape=(1,)

    Stateless combinatorial pass-through.
    """
    PYX_FILE     = str(_HERE / "c_src" / "motor_utility_blocks_wrapper.pyx")
    C_SOURCES    = ["motor_utility_blocks.c"]
    C_HEADERS    = ["motor_utility_blocks.h"]
    step_func    = "VfTheta_Step"
    state_struct = "VfTheta_T"
    init_func    = "VfTheta_Init"
    NUM_INPUTS   = 3
    OUTPUT_SIZE  = 1

    def __init__(self, name: str,
                 use_c_backend: bool = False) -> None:
        super().__init__(name, use_c_backend=use_c_backend)
        self.output_label = "θ_e"

        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        if not _WRAPPER_AVAILABLE:
            raise ImportError(
                "motor_utility_blocks_wrapper not found.")
        self._wrapper = VfThetaWrapper()

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        v = (input_values[0].value if input_values
             else np.zeros(3, dtype=DEFAULT_DTYPE))
        self.output = VectorSignal(
            np.array([v[2]], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None
                  ) -> VectorSignal:
        u = np.zeros(3, dtype=np.float32)   # FIX: float32 matches float[::1] in wrapper
        if input_values:
            u[:3] = input_values[0].value[:3]
        self._wrapper.set_inputs(u)
        self._wrapper.compute(float(dt))
        self.output = VectorSignal(
            self._wrapper.get_outputs().astype(DEFAULT_DTYPE), self.name)
        return self.output


# =============================================================================
# DutyPackBlock
# =============================================================================
class DutyPackBlock(VectorBlock):
    """
    InvClarke + centred PWM → three phase duty cycles.

    Input  port-0: [v_alpha, v_beta]     shape=(2,)  from InvParkTransformBlock
    Input  port-1: [T1, T2, T0, sector]  shape=(4,)  from SVPWMBlock (sector only)
    Output: [duty_a, duty_b, duty_c, V_dc, T_load]  shape=(5,)

    The C implementation uses only [v_alpha, v_beta] — the SVPWM port
    is wired for topology completeness; sector information is not used
    inside DutyPack_Step.
    """
    PYX_FILE     = str(_HERE / "c_src" / "motor_utility_blocks_wrapper.pyx")
    C_SOURCES    = ["motor_utility_blocks.c"]
    C_HEADERS    = ["motor_utility_blocks.h"]
    step_func    = "DutyPack_Step"
    state_struct = "DutyPack_T"
    init_func    = "DutyPack_Init"
    NUM_INPUTS   = 2
    OUTPUT_SIZE  = 5
    # DutyPack_Init(&state, v_dc)
    C_INIT_ARGS  = ["v_dc"]
    # DutyPack_Step takes [v_alpha, v_beta] — NOT the SVPWM output.
    # SVPWMBlock sits between inv_park and duty_pack for topology purposes only.
    # C_INPUT_MAP pins the generated u_duty_pack[] explicitly to y_inv_park[]
    # so StepGenerator does not auto-wire from the nearest upstream (svpwm).
    C_INPUT_MAP  = [("inv_park", 0), ("inv_park", 1)]
    # Only duty_a/b/c are meaningful outputs for the integration layer.
    # V_dc (index 3) and T_load (index 4) are internal — excluded from
    # EmbedSim_Output_T.  AurixStepGenerator reads these to build the struct.
    OUTPUT_NAMES = ["duty_a", "duty_b", "duty_c"]
    OUTPUT_KEEP  = [0, 1, 2]

    def __init__(self, name: str,
                 v_dc: float,
                 use_c_backend: bool = False) -> None:
        super().__init__(name, use_c_backend=use_c_backend)
        self.v_dc         = float(v_dc)
        self.output_label = "[da,db,dc]"
        self._HSQ3        = math.sqrt(3.0) / 2.0

        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        if not _WRAPPER_AVAILABLE:
            raise ImportError(
                "motor_utility_blocks_wrapper not found.")
        self._wrapper = DutyPackWrapper(float(self.v_dc))

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        v_alpha = v_beta = 0.0
        if input_values and input_values[0] is not None:
            ab = input_values[0].value
            if len(ab) >= 2:
                v_alpha, v_beta = float(ab[0]), float(ab[1])
        va =  v_alpha
        vb = -0.5 * v_alpha + self._HSQ3 * v_beta
        vc = -0.5 * v_alpha - self._HSQ3 * v_beta
        da = max(0.02, min(0.98, 0.5 + va / self.v_dc))
        db = max(0.02, min(0.98, 0.5 + vb / self.v_dc))
        dc = max(0.02, min(0.98, 0.5 + vc / self.v_dc))
        self.output = VectorSignal(
            np.array([da, db, dc, self.v_dc, 0.0], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None
                  ) -> VectorSignal:
        u = np.zeros(2, dtype=np.float32)   # FIX: float32 matches float[::1] in wrapper
        if input_values and input_values[0] is not None:
            ab = input_values[0].value
            if len(ab) >= 2:
                u[0] = float(ab[0])
                u[1] = float(ab[1])
        self._wrapper.set_inputs(u)
        self._wrapper.compute(float(dt))
        self.output = VectorSignal(
            self._wrapper.get_outputs().astype(DEFAULT_DTYPE), self.name)
        return self.output


# =============================================================================
# SVPWMPackBlock
# =============================================================================
class SVPWMPackBlock(VectorBlock):
    """
    Polar adapter: [v_alpha, v_beta] → [Vref, alpha_angle [rad], Vdc].

    Converts InvPark output to the form expected by SVPWMBlock and
    SVM_CalculateDutyCycle():
      Vref        = sqrt(v_alpha^2 + v_beta^2),  clipped to 0.95
      alpha_angle = atan2(v_beta, v_alpha),       wrapped to [0, 2π)
      V_dc        = compile-time constant

    Both the clip and the wrap mirror SpaceVectorModulation1() in the
    AURIX application layer.  SVPWMBlock.compute_py trusts that these
    invariants are already satisfied on its inputs.

    Stateless combinatorial block — same placeholder struct pattern as
    VfDQ and VfTheta.
    """
    PYX_FILE     = str(_HERE / "c_src" / "motor_utility_blocks_wrapper.pyx")
    C_SOURCES    = ["motor_utility_blocks.c"]
    C_HEADERS    = ["motor_utility_blocks.h"]
    step_func    = "SVPWMPack_Step"
    state_struct = "SVPWMPack_T"
    init_func    = "SVPWMPack_Init"
    NUM_INPUTS   = 2
    OUTPUT_SIZE  = 3
    C_INIT_ARGS  = ["v_dc"]

    def __init__(self, name: str,
                 v_dc: float,
                 use_c_backend: bool = False) -> None:
        super().__init__(name, use_c_backend=use_c_backend)
        self.v_dc         = float(v_dc)
        self.output_label = "[Vref,α,Vdc]"

        if use_c_backend:
            self._load_wrapper()

    def _load_wrapper(self) -> None:
        if not _WRAPPER_AVAILABLE:
            raise ImportError(
                "motor_utility_blocks_wrapper not found.")
        self._wrapper = SVPWMPackWrapper(float(self.v_dc))

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:
        va = vb = 0.0
        if input_values and input_values[0] is not None:
            ab = input_values[0].value
            if len(ab) >= 2:
                va, vb = float(ab[0]), float(ab[1])
        Vref  = math.sqrt(va * va + vb * vb)
        Vref  = min(Vref, 0.95)                    # clip — mirrors SpaceVectorModulation1()
        alpha = math.atan2(vb, va)
        if alpha < 0.0:
            alpha += 2.0 * math.pi                 # wrap to [0, 2π) — SVM_CalculateDutyCycle contract
        self.output = VectorSignal(
            np.array([Vref, alpha, self.v_dc], dtype=DEFAULT_DTYPE),
            self.name)
        return self.output

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None
                  ) -> VectorSignal:
        u = np.zeros(2, dtype=np.float32)   # FIX: float32 matches float[::1] in wrapper
        if input_values and input_values[0] is not None:
            ab = input_values[0].value
            if len(ab) >= 2:
                u[0] = float(ab[0])
                u[1] = float(ab[1])
        self._wrapper.set_inputs(u)
        self._wrapper.compute(float(dt))
        self.output = VectorSignal(
            self._wrapper.get_outputs().astype(DEFAULT_DTYPE), self.name)
        return self.output


# =============================================================================
# Module metadata
# =============================================================================
__all__ = [
    "SpeedRampBlock",
    "VfAngleBlock",
    "VfDQBlock",
    "VfThetaBlock",
    "DutyPackBlock",
    "SVPWMPackBlock",
]

__version__ = "1.0.0"
__author__  = "EmbedSim Framework"
