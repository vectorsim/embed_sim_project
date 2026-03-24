# smu_controller_block.py - Using _path_utils
"""
smc_controller_block.py
=======================
Sliding Mode Controller for PMSM with correct torque constant.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from pathlib import Path

# ── Path bootstrap using _path_utils ──────────────────────────────────────────
from _path_utils import get_embedsim_import_path, get_current_parent

# Add the project root to sys.path
_ROOT = get_embedsim_import_path()
_HERE = get_current_parent()
_C_SRC = _HERE / "c_src"

import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_C_SRC) not in sys.path:
    sys.path.insert(0, str(_C_SRC))

# ── EmbedSim imports ──────────────────────────────────────────────────────────
from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorSignal
from pyx_inspector import auto_populate_from_pyx


# ==============================================================================
# CONFIGURATION DATACLASSES
# ==============================================================================

@dataclass
class MotorParams:
    """
    PMSM motor parameters in SI units.

    Torque constant: Kt = (3/2) * p * λpm = 0.0084 Nm/A
    Required current for 20 mNm: I = 0.02 / 0.0084 = 2.38 A
    """
    pole_pairs: float = 4.0
    R_s: float = 0.19
    L_d: float = 0.125e-3
    L_q: float = 0.125e-3
    lambda_pm: float = 0.0014
    J: float = 2.4e-6
    B: float = 7e-5

    @property
    def torque_constant(self) -> float:
        """Kt = (3/2) * p * λpm = 0.0084 Nm/A"""
        return 1.5 * self.pole_pairs * self.lambda_pm


@dataclass
class SMCParams:
    """
    Sliding Mode Controller parameters - tuned for stable operation.
    """
    # Current loop gains
    ks_current: float = 5.0  # Reduced for stability
    phi_current: float = 0.5
    eta_current: float = 0.2

    # Speed loop gains
    ks_speed: float = 0.25  # Reduced for stability
    ki_speed: float = 1.0  # Reduced integral gain
    phi_speed: float = 80.0  # Increased for smoother response
    eta_speed: float = 0.1

    # Limits
    i_max: float = 5.0  # 5A limit (2.38A needed + margin)
    v_max: float = 9.81

    # Current rate limiting
    i_rate_limit: float = 50.0  # A/s - smooth current transitions

    # Observer filters
    load_lpf_hz: float = 10.0
    acc_lpf_hz: float = 20.0
    current_lpf_hz: float = 100.0
    speed_lpf_hz: float = 20.0

    @property
    def phi_speed_rad(self) -> float:
        return self.phi_speed * 2.0 * math.pi / 60.0


# ==============================================================================
# PURE-PYTHON FALLBACK IMPLEMENTATION
# ==============================================================================

class _PySMCCore:
    """
    Pure-Python mirror of the C SMC algorithm.
    """

    def __init__(self, motor: MotorParams, smc: SMCParams, dt: float = 1e-4):
        self.motor = motor
        self.smc = smc
        self.dt = dt

        # State variables
        self.int_speed = 0.0
        self.T_load_est = 0.0
        self.omega_prev = 0.0
        self.domega_filt = 0.0
        self.iq_filtered = 0.0
        self.id_filtered = 0.0
        self.speed_filtered = 0.0

        # Current limiting state
        self._iq_ref_prev = 0.0

        # Filter coefficients
        self.alpha_current = self._get_alpha(smc.current_lpf_hz)
        self.alpha_speed = self._get_alpha(smc.speed_lpf_hz)
        self.alpha_acc = self._get_alpha(smc.acc_lpf_hz)
        self.alpha_load = self._get_alpha(smc.load_lpf_hz)

        # Diagnostics
        self.log_iq_ref = []
        self.log_iq = []
        self.log_id = []
        self.log_speed = []
        self.log_speed_ref = []
        self.log_t = []

    @staticmethod
    def _get_alpha(fc_hz: float, dt: float = 1e-4) -> float:
        if fc_hz <= 0:
            return 1.0
        return 1.0 - math.exp(-2.0 * math.pi * fc_hz * dt)

    @staticmethod
    def _clamp(x: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(max_val, x))

    @staticmethod
    def _sat(x: float, phi: float) -> float:
        if phi <= 0:
            return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)
        r = x / phi
        return max(-1.0, min(1.0, r))

    @staticmethod
    def _lpf(old: float, new: float, alpha: float) -> float:
        return old + alpha * (new - old)

    def _limit_current(self, iq_ref: float, dt: float) -> float:
        """Rate limiting for smooth current transitions"""
        max_delta = self.smc.i_rate_limit * dt
        delta = iq_ref - self._iq_ref_prev
        if abs(delta) > max_delta:
            iq_ref = self._iq_ref_prev + max_delta * (1.0 if delta > 0 else -1.0)
        self._iq_ref_prev = iq_ref
        return self._clamp(iq_ref, -self.smc.i_max, self.smc.i_max)

    def observe_load_torque(self, iq: float, omega: float) -> float:
        T_em = self.motor.torque_constant * iq

        if self.dt > 0:
            domega_raw = (omega - self.omega_prev) / self.dt
            self.domega_filt = self._lpf(self.domega_filt, domega_raw, self.alpha_acc)
        self.omega_prev = omega

        T_acc = self.motor.J * self.domega_filt
        T_fric = self.motor.B * omega

        T_load_raw = T_em - T_acc - T_fric
        T_load_raw = self._clamp(T_load_raw, 0.0, 0.03)

        self.T_load_est = self._lpf(self.T_load_est, T_load_raw, self.alpha_load)
        return self.T_load_est

    def speed_smc(self, omega_ref: float, omega: float, dt: float) -> Tuple[float, float]:
        error = omega_ref - omega
        self.int_speed += error * dt
        int_limit = self.smc.i_max / self.smc.ki_speed
        self.int_speed = self._clamp(self.int_speed, -int_limit, int_limit)

        s_w = error + self.smc.ki_speed * self.int_speed

        ks_adapt = self.smc.ks_speed * (1.0 + min(1.0, abs(s_w) / self.smc.phi_speed_rad))

        iq_smc = ks_adapt * self._sat(s_w, self.smc.phi_speed_rad) + self.smc.eta_speed * s_w
        iq_ff = self.T_load_est / self.motor.torque_constant

        iq_ref = iq_smc + iq_ff
        iq_ref = self._limit_current(iq_ref, dt)

        return iq_ref, s_w

    def current_smc(self, id_ref: float, iq_ref: float,
                    id: float, iq: float, omega_e: float) -> Tuple[float, float, float, float]:
        s_d = id_ref - id
        s_q = iq_ref - iq

        vd_eq = self.motor.R_s * id - omega_e * self.motor.L_q * iq
        vq_eq = self.motor.R_s * iq + omega_e * (self.motor.L_d * id + self.motor.lambda_pm)

        ks_d = self.smc.ks_current * (1.0 + min(1.0, abs(s_d) / self.smc.phi_current))
        ks_q = self.smc.ks_current * (1.0 + min(1.0, abs(s_q) / self.smc.phi_current))

        vd_sw = ks_d * self._sat(s_d, self.smc.phi_current) + self.smc.eta_current * s_d
        vq_sw = ks_q * self._sat(s_q, self.smc.phi_current) + self.smc.eta_current * s_q

        vd = vd_eq + vd_sw
        vq = vq_eq + vq_sw

        v_mag = math.sqrt(vd * vd + vq * vq)
        if v_mag > self.smc.v_max:
            scale = self.smc.v_max / v_mag
            vd *= scale
            vq *= scale

        return vd, vq, s_d, s_q

    def step(self, omega_ref: float, omega_m: float, theta_e: float,
             id_meas: float, iq_meas: float, dt: float) -> Tuple[float, float]:
        # Filter measurements
        self.id_filtered = self._lpf(self.id_filtered, id_meas, self.alpha_current)
        self.iq_filtered = self._lpf(self.iq_filtered, iq_meas, self.alpha_current)
        self.speed_filtered = self._lpf(self.speed_filtered, omega_m, self.alpha_speed)

        id_filt = self.id_filtered
        iq_filt = self.iq_filtered
        omega_filt = self.speed_filtered

        omega_e = self.motor.pole_pairs * omega_filt

        self.observe_load_torque(iq_filt, omega_filt)

        iq_ref, _ = self.speed_smc(omega_ref, omega_filt, dt)

        id_ref = 0.0
        vd, vq, _, _ = self.current_smc(id_ref, iq_ref, id_filt, iq_filt, omega_e)

        return vd, vq

    def reset(self):
        self.int_speed = 0.0
        self.T_load_est = 0.0
        self.omega_prev = 0.0
        self.domega_filt = 0.0
        self.iq_filtered = 0.0
        self.id_filtered = 0.0
        self.speed_filtered = 0.0
        self._iq_ref_prev = 0.0


# ==============================================================================
# SMCControllerBlock - THE MAIN EMBEDSIM BLOCK
# ==============================================================================

class SMCControllerBlock(SimBlockBase):
    """
    Sliding Mode Controller for PMSM.

    Inputs: [omega_ref, omega_m, theta_e, ia, ib, ic]
    Outputs: [v_alpha, v_beta]
    """

    # ── CodeGen attributes ────────────────────────────────────────────────────
    import pathlib as _pl
    PYX_FILE: str = str(_C_SRC / 'smc_controller_wrapper.pyx')

    # These will be auto-populated if the .pyx exists
    step_func: str = ''
    state_struct: str = ''
    NUM_INPUTS: int = 0
    OUTPUT_SIZE: int = 0
    C_SOURCES: list = []
    C_HEADERS: list = []

    OUTPUT_NAMES: list = ["v_alpha", "v_beta"]
    OUTPUT_KEEP: list = [0, 1]
    state_struct: str = 'SMC_Block_T'

    C_CUSTOM_EMIT: str = ''

    @classmethod
    def _build_custom_emit(cls) -> None:
        """Build C_CUSTOM_EMIT if the wrapper exists."""
        if not cls.step_func:
            return
        import re as _re
        fn = cls.step_func
        ss = cls.state_struct

        m = _re.match(r'^(.+?)_(?:Compute|Step|Update)$', fn, _re.IGNORECASE)
        prefix = m.group(1) if m else fn
        in_struct = f"{prefix}_Input_T"
        out_struct = f"{prefix}_Output_T"
        state_var = "smc_state"

        cls.C_CUSTOM_EMIT = (
            f"    /* --- smc_controller ({cls.__name__}) --- */\n"
            f"    {{\n"
            f"        {in_struct}  u_smc;\n"
            f"        {out_struct} y_smc;\n"
            f"        u_smc.omega_ref = u_cg_start[0];\n"
            f"        u_smc.omega_m   = u_cg_start[1];\n"
            f"        u_smc.theta_e   = u_cg_start[2];\n"
            f"        u_smc.ia        = u_cg_start[3];\n"
            f"        u_smc.ib        = u_cg_start[4];\n"
            f"        u_smc.ic        = u_cg_start[5];\n"
            f"        {fn}(&{state_var}, &u_smc, dt, &y_smc);\n"
            f"        out->smc_alpha = y_smc.v_alpha;\n"
            f"        out->smc_beta  = y_smc.v_beta;\n"
            f"    }}"
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, 'PYX_FILE') and cls.PYX_FILE:
            auto_populate_from_pyx(cls, cls.PYX_FILE)
        cls._build_custom_emit()

    def __init__(
            self,
            name: str,
            motor: MotorParams = None,
            smc: SMCParams = None,
            dt: float = 1e-4,
            use_c_backend: bool = False,
            dtype=None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.motor = motor or MotorParams()
        self.smc = smc or SMCParams()
        self.dt = dt

        self.output_label = "[v_alpha, v_beta]"
        self.vector_size = 2
        self.is_dynamic = True

        # RK4 state vector: length 1 = speed integrator
        self.state = np.zeros(1, dtype=np.float32)
        self.k1 = self.k2 = self.k3 = self.k4 = np.zeros(1, dtype=np.float32)

        # Use Python backend by default
        self._impl = _PySMCCore(motor, smc, dt)

        # Diagnostics
        self._log_t = []
        self._log_iq_ref = []
        self._log_iq = []
        self._log_id = []
        self._log_speed = []
        self._log_speed_ref = []
        self._log_next = 0.0
        self._print_count = 0

    def _get_inputs(self, input_values) -> Tuple[float, float, float, float, float, float]:
        omega_ref = omega_m = theta_e = ia = ib = ic = 0.0
        if not input_values:
            return omega_ref, omega_m, theta_e, ia, ib, ic
        if len(input_values) > 0 and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 6:
                omega_ref, omega_m, theta_e, ia, ib, ic = v[0], v[1], v[2], v[3], v[4], v[5]
        return omega_ref, omega_m, theta_e, ia, ib, ic

    def get_derivative(self, t: float,
                       input_values: Optional[List[VectorSignal]] = None
                       ) -> np.ndarray:
        omega_ref, omega_m, _, _, _, _ = self._get_inputs(input_values)
        e = np.float32(omega_ref - omega_m)
        return np.array([e], dtype=np.float32)

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        omega_ref, omega_m, theta_e, ia, ib, ic = self._get_inputs(input_values)

        # Clarke transform
        i_alpha = (2.0 / 3.0) * ia - (1.0 / 3.0) * ib - (1.0 / 3.0) * ic
        i_beta = (ib - ic) / math.sqrt(3.0)

        # Park transform
        cos_theta = math.cos(theta_e)
        sin_theta = math.sin(theta_e)
        id_meas = i_alpha * cos_theta + i_beta * sin_theta
        iq_meas = -i_alpha * sin_theta + i_beta * cos_theta

        # SMC core
        vd, vq = self._impl.step(omega_ref, omega_m, theta_e, id_meas, iq_meas, dt)

        # Inverse Park
        v_alpha = vd * cos_theta - vq * sin_theta
        v_beta = vd * sin_theta + vq * cos_theta

        # Logging
        if t >= self._log_next:
            self._log_t.append(t)
            self._log_speed.append(omega_m * 60.0 / (2.0 * math.pi))
            self._log_speed_ref.append(omega_ref * 60.0 / (2.0 * math.pi))
            self._log_iq_ref.append(self._impl.log_iq_ref[-1] if self._impl.log_iq_ref else 0)
            self._log_iq.append(iq_meas)
            self._log_id.append(id_meas)
            self._log_next += 0.02

            if self._print_count < 30:
                rpm = omega_m * 60.0 / (2.0 * math.pi)
                print(f"[SMC t={t:.2f}] speed={rpm:.0f} RPM, iq={iq_meas:.2f} A")
                self._print_count += 1

        self.output = VectorSignal(np.array([v_alpha, v_beta], dtype=np.float32),
                                   self.name, dtype=self.dtype)
        return self.output

    def reset(self) -> None:
        super().reset()
        self.state = np.zeros(1, dtype=np.float32)
        if hasattr(self, '_impl'):
            self._impl.reset()
        self._log_t.clear()
        self._log_iq_ref.clear()
        self._log_iq.clear()
        self._log_id.clear()
        self._log_speed.clear()
        self._log_speed_ref.clear()
        self._log_next = 0.0
        self._print_count = 0

    @property
    def log_data(self) -> dict:
        return {
            't': np.array(self._log_t, dtype=np.float32),
            'speed': np.array(self._log_speed, dtype=np.float32),
            'speed_ref': np.array(self._log_speed_ref, dtype=np.float32),
            'iq_ref': np.array(self._log_iq_ref, dtype=np.float32),
            'iq': np.array(self._log_iq, dtype=np.float32),
            'id': np.array(self._log_id, dtype=np.float32),
        }

    def __repr__(self) -> str:
        return f"SMCControllerBlock('{self.name}')"


# ── Build C_CUSTOM_EMIT if possible ──────────────────────────────────────────
#SMCControllerBlock._build_custom_emit()
