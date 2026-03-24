<<<<<<< Updated upstream
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
=======
# smc_controller_block.py

"""
smc_controller_block.py
=======================
SMC FOC Controller — transforms delegated to coordinate_transform_blocks.py.

All Clarke / Park / InvPark calculations are performed by
ClarkeTransformBlock, ParkTransformBlock and InvParkTransformBlock from
coordinate_transform_blocks.py — which are the Python mirrors of
Clarke_Step(), Park_Step() and InvPark_Step() in
embed_sim_coordinate_transform.c.

There is no inline transform math in this file.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from pyx_inspector import auto_populate_from_pyx
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)


# =============================================================================
# Motor constants
# =============================================================================

class _DB42S02:
    """NANOTEC DB42S02 motor parameters."""

    # Motor parameters
    SMC_P_POLES = 4
    SMC_R_S = 0.19
    SMC_L_D = 0.125e-3
    SMC_L_Q = 0.125e-3
    SMC_LAMBDA_PM = 0.0014
    SMC_J_ROTOR = 2.4e-6
    SMC_B_FRICTION = 1e-6
    SMC_I_MAX = 3.57
    SMC_V_DC = 17.0
    SMC_V_MAX = SMC_V_DC / math.sqrt(3.0)
    SMC_KT = 1.5 * SMC_P_POLES * SMC_LAMBDA_PM

    # Fixed surface coefficients
    SMC_WC_I = 2.0 * math.pi * 800.0
    SMC_LAMBDA_W = 2.0 * math.pi * 20.0
    SMC_GAMMA_W = 2.0 * math.pi * 5.0

    # Tunable gains
    SMC_KS_I = SMC_L_D * SMC_WC_I
    SMC_PHI_I = 0.5
    SMC_KS_W = 0.035
    SMC_PHI_W = 5.0
    SMC_ETA_W = 0.1


# =============================================================================
# SMCControllerBlock
# =============================================================================
class SMCControllerBlock(VectorBlock):
    """
    Sliding Mode FOC Controller — aligned with embed_sim_smc_controller.c
    and coordinate_transform_blocks.py.

    Clarke (amplitude-invariant) — matches Clarke_Step() in C exactly:
        i_alpha = (2/3)·ia - (1/3)·ib - (1/3)·ic
        i_beta  = (ib - ic) / √3

    Park — matches Park_Step() in C exactly:
        id =  i_alpha·cos θ_e + i_beta·sin θ_e
        iq = -i_alpha·sin θ_e + i_beta·cos θ_e

    Inverse Park — matches InvPark_Step() in C exactly:
        v_alpha = vd·cos θ_e - vq·sin θ_e
        v_beta  = vd·sin θ_e + vq·cos θ_e

    Current SMC switching term sign convention:
        s = i_meas - i_ref   (error surface)
        v_sw = -ks·sat(s/φ)  (negative — Lyapunov stability requirement)
        Matches corrected SMC_CurrentSMC() in embed_sim_smc_controller.c.
    """

    # ── CodeGen ──────────────────────────────────────────────────────────────
    PYX_FILE = str(_C_SRC / "smc_controller_wrapper.pyx")
    C_SOURCES = ["embed_sim_smc_controller.c"]
    C_HEADERS = ["embed_sim_smc_controller.h"]
    state_struct = "SMC_Controller_T"
    step_func = "SMC_Controller_Step"
    init_func = "SMC_Controller_Init"
    C_INIT_ARGS = ["dt_s"]

    C_CUSTOM_EMIT = """\
        /* --- smc_controller (SMCControllerBlock) --- */
        SMC_Input_T   u_smc;
        SMC_Output_T  y_smc_out;
        real32_T      y_smc[2];

        u_smc.omega_ref_mech = in->omega_ref_mech;
        u_smc.theta_m        = in->theta_m;
        u_smc.ia             = in->ia;
        u_smc.ib             = in->ib;
        u_smc.ic             = in->ic;

        SMC_Controller_Step(&smc_state, &u_smc, dt, &y_smc_out);

        y_smc[0] = y_smc_out.v_alpha;
        y_smc[1] = y_smc_out.v_beta;"""

    DIAG_STEPS: int = 200 if os.environ.get("SMC_DBG") == "1" else 20
    _SQRT3 = math.sqrt(3.0)

    def __init__(
            self,
            name: str = "smc",
            SMC_V_DC: float = _DB42S02.SMC_V_DC,
            SMC_P_POLES: int = _DB42S02.SMC_P_POLES,
            SMC_R_S: float = _DB42S02.SMC_R_S,
            SMC_L_D: float = _DB42S02.SMC_L_D,
            SMC_L_Q: float = _DB42S02.SMC_L_Q,
            SMC_LAMBDA_PM: float = _DB42S02.SMC_LAMBDA_PM,
            SMC_J_ROTOR: float = _DB42S02.SMC_J_ROTOR,
            SMC_B_FRICTION: float = _DB42S02.SMC_B_FRICTION,
            SMC_I_MAX: float = _DB42S02.SMC_I_MAX,
            SMC_KS_W: float = _DB42S02.SMC_KS_W,
            SMC_ETA_W: float = _DB42S02.SMC_ETA_W,
            SMC_PHI_W: float = _DB42S02.SMC_PHI_W,
            SMC_KS_I: float = _DB42S02.SMC_KS_I,
            SMC_PHI_I: float = _DB42S02.SMC_PHI_I,
            SMC_LAMBDA_W: float = _DB42S02.SMC_LAMBDA_W,
            SMC_GAMMA_W: float = _DB42S02.SMC_GAMMA_W,
            dt_s: float = 50e-6,
            use_c_backend: bool = False,
            integrator: str = "tustin",
            dtype=None,
    ) -> None:

        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        # Integrator selection
        _valid = ("tustin", "heun", "euler")
        if integrator not in _valid:
            raise ValueError(f"integrator must be one of {_valid}, got {integrator!r}")
        self._integrator: str = integrator

        # Motor parameters
        self.SMC_V_DC = float(SMC_V_DC)
        self.SMC_P_POLES = int(SMC_P_POLES)
        self.SMC_R_S = float(SMC_R_S)
        self.SMC_L_D = float(SMC_L_D)
        self.SMC_L_Q = float(SMC_L_Q)
        self.SMC_LAMBDA_PM = float(SMC_LAMBDA_PM)
        self.SMC_J_ROTOR = float(SMC_J_ROTOR)
        self.SMC_B_FRICTION = float(SMC_B_FRICTION)
        self.SMC_I_MAX = float(SMC_I_MAX)
        self.SMC_V_MAX = self.SMC_V_DC / self._SQRT3

        # Gains
        self.SMC_KS_W = float(SMC_KS_W)
        self.SMC_ETA_W = float(SMC_ETA_W)
        self.SMC_PHI_W = float(SMC_PHI_W)
        self.SMC_KS_I = float(SMC_KS_I)
        self.SMC_PHI_I = float(SMC_PHI_I)
        self.SMC_LAMBDA_W = float(SMC_LAMBDA_W)
        self.SMC_GAMMA_W = float(SMC_GAMMA_W)

        self._dt_s_float = float(dt_s)
        self.dt_s = "EMBEDSIM_DT"
        self.vector_size = 2
        self.output_label = "[v_α,v_β]"
        self.is_dynamic = False

        # Integrator states
        self._int_spd: float = 0.0
        self._int2_spd: float = 0.0
        self._e_prev: float = 0.0
        self._int_spd_prev: float = 0.0

        # Speed estimator state
        self._omega_filt: float = 0.0
        self._last_theta_m: float = 0.0

        # Diagnostic
        self._last_iq_ref: float = 0.0
        self._log_t: list = []
        self._log_spd: list = []
        self._log_sref: list = []
        self._log_iqr: list = []
        self._log_iq: list = []
        self._log_id: list = []
        self._log_next: float = 0.0
        self._diag_count: int = 0

        # C backend wrapper
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

        # ── Transform block instances — canonical, no inline math ────────────
        # ClarkeTransformBlock / ParkTransformBlock / InvParkTransformBlock
        # are imported at the top of this file from coordinate_transform_blocks.
        # Instantiated once here; reused every step at zero allocation cost.
        self._ct_clarke   = ClarkeTransformBlock("_smc_clarke",    use_c_backend=False)
        self._ct_park     = ParkTransformBlock("_smc_park",        use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_smc_inv_park", use_c_backend=False)

        print(f"[SMC] Transforms delegated to coordinate_transform_blocks.py")

    def _load_wrapper(self) -> None:
        try:
            from smc_controller_wrapper import SMCControllerWrapper
            self._wrapper = SMCControllerWrapper(
                self.SMC_V_DC, self.SMC_P_POLES,
                self.SMC_R_S, self.SMC_L_D, self.SMC_L_Q,
                self.SMC_LAMBDA_PM, self.SMC_J_ROTOR, self.SMC_B_FRICTION,
                self.SMC_I_MAX, self._dt_s_float)
        except ImportError as exc:
            raise ImportError(
                "smc_controller_wrapper.pyd not found. Build with:\n"
                "  cd fs_electrical_machines/c_src\n"
                "  python setup_smc_controller.py build_ext --inplace"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"SMCControllerWrapper instantiation failed: {exc}"
            ) from exc

    # ============= STANDARD TEXTBOOK TRANSFORMS =============

    # ── Transforms — delegate to coordinate_transform_blocks ────────────────
    # These are thin wrappers so the SMC never duplicates transform math.
    # The canonical implementations live in coordinate_transform_blocks.py
    # and embed_sim_coordinate_transform.c — one source of truth for both
    # simulation and generated C code.
    #
    # Block instances are created once in __init__ (_ct_clarke, _ct_park,
    # _ct_inv_park) and reused every step — no per-call allocation overhead.

    def _clarke(self, ia: float, ib: float, ic: float) -> tuple:
        """Clarke abc→αβ — delegates to ClarkeTransformBlock.compute_py()."""
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_clarke")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(self, i_alpha: float, i_beta: float, theta_e: float) -> tuple:
        """Park αβ→dq — delegates to ParkTransformBlock.compute_py()."""
        ab  = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_park")
        th  = VectorSignal(np.array([theta_e],         dtype=np.float32), "_park")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(self, vd: float, vq: float, theta_e: float) -> tuple:
        """Inverse Park dq→αβ — delegates to InvParkTransformBlock.compute_py()."""
        dq  = VectorSignal(np.array([vd, vq],   dtype=np.float32), "_inv_park")
        th  = VectorSignal(np.array([theta_e],   dtype=np.float32), "_inv_park")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # NOTE: _clarke_inverse is not used by SMCControllerBlock.
    # Inverse Clarke (αβ→abc) is performed by SVPWMBlock downstream.
    # If ever needed, use InvClarkeTransformBlock from coordinate_transform_blocks.py.

    @staticmethod
    def _sat(x: float, phi: float) -> float:
        if phi <= 0.0:
            return math.copysign(1.0, x) if x != 0.0 else 0.0
        result = x / phi
        return max(-1.0, min(1.0, result))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ── Speed estimation ────────────────────────────────────────────────────
    def _get_speed_from_encoder(self, theta_m: float, dt: float) -> float:
        if dt > 0.0:
            omega_raw = (theta_m - self._last_theta_m) / dt
            self._omega_filt = 0.95 * self._omega_filt + 0.05 * omega_raw
        self._last_theta_m = theta_m
        return self._omega_filt

    # ── Speed SMC ──────────────────────────────────────────────────────────
    def _speed_smc(self, omega_ref: float, omega_m: float, dt: float) -> float:
        e = omega_ref - omega_m
        int_limit = self.SMC_PHI_W / self.SMC_LAMBDA_W
        int2_limit = int_limit / self.SMC_GAMMA_W

        if self._integrator == "tustin":
            half_dt = 0.5 * dt
            new_int_spd = self._int_spd + half_dt * (e + self._e_prev)
            new_int2_spd = self._int2_spd + half_dt * (new_int_spd + self._int_spd)
            self._int_spd = self._clamp(new_int_spd, -int_limit, int_limit)
            self._int2_spd = self._clamp(new_int2_spd, -int2_limit, int2_limit)
            self._int_spd_prev = self._int_spd
            self._e_prev = e
        elif self._integrator == "heun":
            half_dt = 0.5 * dt
            new_int_spd = self._int_spd + half_dt * (self._e_prev + e)
            new_int2_spd = self._int2_spd + half_dt * (self._int_spd_prev + new_int_spd)
            self._int_spd_prev = self._int_spd
            self._int_spd = self._clamp(new_int_spd, -int_limit, int_limit)
            self._int2_spd = self._clamp(new_int2_spd, -int2_limit, int2_limit)
            self._e_prev = e
        else:
            self._int_spd = self._clamp(self._int_spd + dt * e, -int_limit, int_limit)
            self._int2_spd = self._clamp(self._int2_spd + dt * self._int_spd, -int2_limit, int2_limit)

        s_spd = e + self.SMC_LAMBDA_W * self._int_spd + self.SMC_GAMMA_W * self._int2_spd
        iq_ref = (self.SMC_KS_W * self._sat(s_spd, self.SMC_PHI_W) + self.SMC_ETA_W * s_spd)
        return self._clamp(iq_ref, -self.SMC_I_MAX, self.SMC_I_MAX)

    # ── Current SMC ─────────────────────────────────────────────────────────
    def _current_smc(self, id_meas: float, iq_meas: float, id_ref: float, iq_ref: float, omega_e: float) -> tuple:
        s_d = id_meas - id_ref
        s_q = iq_meas - iq_ref

        vd_eq = (self.SMC_R_S * id_meas) - (omega_e * self.SMC_L_Q * iq_meas)
        vq_eq = (self.SMC_R_S * iq_meas + omega_e * (self.SMC_L_D * id_meas + self.SMC_LAMBDA_PM))

        # Switching term: v_sw = -ks·sat(s/φ)
        # s = i_meas - i_ref; Lyapunov requires s·ds/dt < 0 → negative sign.
        # Matches corrected SMC_CurrentSMC() in embed_sim_smc_controller.c.
        vd_sw = -(self.SMC_KS_I * self._sat(s_d, self.SMC_PHI_I))
        vq_sw = -(self.SMC_KS_I * self._sat(s_q, self.SMC_PHI_I))
>>>>>>> Stashed changes

        vd = vd_eq + vd_sw
        vq = vq_eq + vq_sw

<<<<<<< Updated upstream
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
=======
        magnitude = math.sqrt(vd * vd + vq * vq)
        if magnitude > self.SMC_V_MAX:
            scale = self.SMC_V_MAX / magnitude
            vd *= scale
            vq *= scale

        return vd, vq

    # ── compute methods ─────────────────────────────────────────────────────
    def compute_py(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        zero = np.array([0.0, 0.0], dtype=np.float32)

        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        omega_ref_mech = float(u[0])
        theta_m = float(u[1])
        ia = float(u[2])
        ib = float(u[3])
        ic = float(u[4])

        theta_e = float(self.SMC_P_POLES) * theta_m
        omega_m_est = self._get_speed_from_encoder(theta_m, dt)

        # STANDARD transforms
        i_alpha, i_beta = self._clarke(ia, ib, ic)
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        iq_ref = self._speed_smc(omega_ref_mech, omega_m_est, dt)
        self._last_iq_ref = iq_ref

        omega_e = float(self.SMC_P_POLES) * omega_m_est
        vd, vq = self._current_smc(id_meas, iq_meas, 0.0, iq_ref, omega_e)

        # STANDARD inverse Park
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)
>>>>>>> Stashed changes

        # Logging
        if t >= self._log_next:
            self._log_t.append(t)
<<<<<<< Updated upstream
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
=======
            self._log_spd.append(omega_m_est * 60.0 / (2.0 * math.pi))
            self._log_sref.append(omega_ref_mech * 60.0 / (2.0 * math.pi))
            self._log_iqr.append(iq_ref)
            self._log_iq.append(iq_meas)
            self._log_id.append(id_meas)
            self._log_next += 0.001

        self.output = VectorSignal(np.array([v_alpha, v_beta], dtype=np.float32), self.name)
        return self.output

    def compute_c(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        # C backend implementation (keep existing)
        zero = np.array([0.0, 0.0], dtype=np.float32)
        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        inputs = np.zeros(5, dtype=np.float32)
        inputs[:5] = u[:5]

        self._wrapper.set_inputs(inputs)
        self._wrapper.compute(float(dt))
        outputs = self._wrapper.get_outputs()

        self.output = VectorSignal(outputs, self.name)
>>>>>>> Stashed changes
        return self.output

    def reset(self) -> None:
        super().reset()
<<<<<<< Updated upstream
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
=======
        self._int_spd = 0.0
        self._int2_spd = 0.0
        self._last_iq_ref = 0.0
        self._last_theta_m = 0.0
        self._omega_filt = 0.0
        self._e_prev = 0.0
        self._int_spd_prev = 0.0
        self._log_t.clear()
        self._log_spd.clear()
        self._log_sref.clear()
        self._log_iqr.clear()
        self._log_iq.clear()
        self._log_id.clear()
        self._log_next = 0.0
        self._diag_count = 0
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()
        if self._wrapper is not None:
            self._wrapper.reset()
>>>>>>> Stashed changes

    @property
    def log_data(self) -> dict:
        return {
<<<<<<< Updated upstream
            't': np.array(self._log_t, dtype=np.float32),
            'speed': np.array(self._log_speed, dtype=np.float32),
            'speed_ref': np.array(self._log_speed_ref, dtype=np.float32),
            'iq_ref': np.array(self._log_iq_ref, dtype=np.float32),
            'iq': np.array(self._log_iq, dtype=np.float32),
            'id': np.array(self._log_id, dtype=np.float32),
=======
            "t": np.array(self._log_t, dtype=np.float32),
            "speed": np.array(self._log_spd, dtype=np.float32),
            "speed_ref": np.array(self._log_sref, dtype=np.float32),
            "iq_ref": np.array(self._log_iqr, dtype=np.float32),
            "iq": np.array(self._log_iq, dtype=np.float32),
            "id": np.array(self._log_id, dtype=np.float32),
>>>>>>> Stashed changes
        }

    def __repr__(self) -> str:
        return f"SMCControllerBlock('{self.name}')"


<<<<<<< Updated upstream
# ── Build C_CUSTOM_EMIT if possible ──────────────────────────────────────────
#SMCControllerBlock._build_custom_emit()
=======
__all__ = ["SMCControllerBlock", "_DB42S02"]
>>>>>>> Stashed changes
