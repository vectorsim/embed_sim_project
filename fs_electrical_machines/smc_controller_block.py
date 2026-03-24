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

    # Tunable gains — physics-sized for DB42S02 + 20 mN·m load
    # KS_W must cover max load torque:  T_max/KT = 0.020/0.0084 = 2.38 A
    # Set to 3.0 A to include acceleration margin.
    # ETA_W small — only for small-signal damping, not torque production.
    SMC_KS_I = SMC_L_D * SMC_WC_I   # 0.628 V  (current loop — correct)
    SMC_PHI_I = 0.5                  # A
    SMC_KS_W = 3.0                   # A   (was 0.035 — 85× too small for 20 mN·m)
    SMC_PHI_W = 8.0                  # rad/s  (boundary layer width)
    SMC_ETA_W = 0.005                # small damping only


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
        # Gains — enforce physics floor so caller cannot pass an undersized KS_W.
        # KS_W must be ≥ T_load_max/KT to produce torque at steady state (sat=1).
        _KT      = 1.5 * float(SMC_P_POLES) * float(SMC_LAMBDA_PM)
        _KS_W_min = 0.025 / max(_KT, 1e-6)   # 25 mN·m / KT ≈ 2.976 A
        self.SMC_KS_W = max(float(SMC_KS_W), _KS_W_min)
        if float(SMC_KS_W) < _KS_W_min:
            print(f"[SMC] KS_W={SMC_KS_W} promoted to {self.SMC_KS_W:.4f} A (T_max/KT)")
        # ETA_W cap: must stay small — large values cause integrator wind-up
        self.SMC_ETA_W = min(float(SMC_ETA_W), 0.01)
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

        # Current loop voltage integrators (discrete PI, V units)
        self._v_int_d: float = 0.0
        self._v_int_q: float = 0.0

        # Speed estimator state
        # _last_theta_m_unwrapped tracks the continuous (unwrapped) angle so
        # that the 2π resets in theta_m never cause sign-flip spikes in the
        # derivative.  alpha=0.3 → τ ≈ 3 steps (150 µs at 50 µs dt) which is
        # fast enough to track 2000 RPM spin-up without significant lag.
        self._omega_filt: float = 0.0
        self._last_theta_m: float = 0.0
        self._last_theta_m_unwrapped: float = 0.0   # continuous angle [rad]

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
        print(f"[SMC] Speed gains: KS_W={self.SMC_KS_W:.4f} A  PHI_W={self.SMC_PHI_W:.2f} rad/s  ETA_W={self.SMC_ETA_W:.4f}")
        print(f"[SMC] Current gains: KS_I={self.SMC_KS_I:.4f} V  PHI_I={self.SMC_PHI_I:.3f} A  Kp=L/(5dt)={self.SMC_L_D/(5*self._dt_s_float):.2f} V/A")

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
        """
        Finite-difference speed estimator with 2π unwrapping.

        Without unwrapping every electrical cycle wrap (2π/p ≈ 0.79 rad at p=4)
        produces a large negative spike in (theta_m - last_theta_m) / dt that
        drives the IIR negative and collapses omega_m_est to ~0, starving the
        speed SMC of feedback.

        alpha = 0.3  →  τ = dt / alpha = 50µs / 0.3 ≈ 167 µs  (3 steps)
        That is fast enough to track a 2000 RPM ramp without significant lag,
        yet suppresses quantisation noise from the 50 µs difference.
        """
        if dt > 0.0:
            # Unwrap: find the shortest angular step (handles 2π resets)
            delta = theta_m - self._last_theta_m
            # Bring delta into (-π, +π]
            delta = delta - 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
            self._last_theta_m_unwrapped += delta

            omega_raw = delta / dt
            # IIR: alpha=0.3 for fast tracking (τ ≈ 3 steps)
            self._omega_filt = 0.7 * self._omega_filt + 0.3 * omega_raw

        self._last_theta_m = theta_m
        return self._omega_filt

    # ── Speed SMC ──────────────────────────────────────────────────────────
    def _speed_smc(self, omega_ref: float, omega_m: float, dt: float) -> float:
        e = omega_ref - omega_m
        # Integrator limits — generous, sized in physical units.
        # int_spd  [rad]:   max accumulated angle error ≈ 1 full revolution
        # int2_spd [rad·s]: max double-integrated error
        # The final iq_ref is clamped to ±I_MAX anyway, so these just
        # prevent unbounded growth during large transients (startup, load step).
        int_limit  = 10.0                          # rad   (~1.6 revolutions)
        int2_limit = 10.0 / self.SMC_LAMBDA_W      # rad·s

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
        else:  # euler
            self._int_spd = self._clamp(self._int_spd + dt * e, -int_limit, int_limit)
            self._int2_spd = self._clamp(self._int2_spd + dt * self._int_spd, -int2_limit, int2_limit)

        s_spd = e + self.SMC_LAMBDA_W * self._int_spd + self.SMC_GAMMA_W * self._int2_spd
        iq_ref = (self.SMC_KS_W * self._sat(s_spd, self.SMC_PHI_W) + self.SMC_ETA_W * s_spd)
        return self._clamp(iq_ref, -self.SMC_I_MAX, self.SMC_I_MAX)

    # ── Current SMC ─────────────────────────────────────────────────────────
    def _current_smc(self, id_meas: float, iq_meas: float, id_ref: float,
                     iq_ref: float, omega_e: float,
                     phi_i_override: float = None) -> tuple:
        """
        Discrete PI current controller with FOC decoupling feedforward.

        The feedback loop has one step of delay (VectorDelay / motor_delay).
        With delay z⁻¹, the closed-loop pole for a proportional gain Kp is:

            z² − (1 − α)z = 0   where α = Kp·dt/L

        Deadbeat (α=1) gives roots ±j → sustained oscillation.
        α = 0.5  →  Kp = L/(2·dt),  pole at z = 0.5  (well-damped, BW≈4 kHz)

        A small discrete integral term removes steady-state error caused by
        any residual decoupling mismatch:
            v_int(k) = v_int(k-1) + Ki_d · s(k)
            Ki_d = R·dt/L  (one-step integral matching the plant time constant)

        Structure:
            vd = vd_eq  +  Kp·s_d  +  v_int_d  +  Ks·sat(s_d/φ)
            vq = vq_eq  +  Kp·s_q  +  v_int_q  +  Ks·sat(s_q/φ)

        Anti-windup: integrators are frozen when the voltage vector saturates.
        """
        phi_i = phi_i_override if phi_i_override is not None else self.SMC_PHI_I

        s_d = id_ref - id_meas
        s_q = iq_ref - iq_meas

        # FOC decoupling feedforward (handles steady-state exactly)
        vd_eq = self.SMC_R_S * id_meas - omega_e * self.SMC_L_Q * iq_meas
        vq_eq = (self.SMC_R_S * iq_meas
                 + omega_e * (self.SMC_L_D * id_meas + self.SMC_LAMBDA_PM))

        # Proportional: Kp = L/(2·dt) → pole at z=0.5 with one-step delay
        Kp = self.SMC_L_D / (5.0 * max(self._dt_s_float, 1e-7))

        # Discrete integral: Ki_d = R·dt/L  (dimensionless voltage per A·step)
        Ki_step = self.SMC_R_S / 5.0

        # Advance integrators (freeze on saturation via anti-windup flag)
        v_int_d_new = self._v_int_d + Ki_step * s_d
        v_int_q_new = self._v_int_q + Ki_step * s_q

        # Clamp integrators to ±V_MAX to prevent wind-up
        v_int_d_new = self._clamp(v_int_d_new, -self.SMC_V_MAX, self.SMC_V_MAX)
        v_int_q_new = self._clamp(v_int_q_new, -self.SMC_V_MAX, self.SMC_V_MAX)

        # SMC boundary-layer switching
        vd_sw = self.SMC_KS_I * self._sat(s_d, phi_i)
        vq_sw = self.SMC_KS_I * self._sat(s_q, phi_i)

        vd = vd_eq + Kp * s_d + v_int_d_new + vd_sw
        vq = vq_eq + Kp * s_q + v_int_q_new + vq_sw

        # Vector voltage limit with integrator anti-windup (freeze on sat)
        magnitude = math.sqrt(vd * vd + vq * vq)
        if magnitude > self.SMC_V_MAX:
            scale = self.SMC_V_MAX / magnitude
            vd *= scale
            vq *= scale
            # Freeze integrators — do not commit this step's increment
        else:
            self._v_int_d = v_int_d_new
            self._v_int_q = v_int_q_new

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
        theta_m        = float(u[1])
        ia             = float(u[2])
        ib             = float(u[3])
        ic             = float(u[4])

        theta_e    = float(self.SMC_P_POLES) * theta_m
        omega_m_est = self._get_speed_from_encoder(theta_m, dt)

        # Transforms — via coordinate_transform_blocks
        i_alpha, i_beta = self._clarke(ia, ib, ic)
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        iq_ref = self._speed_smc(omega_ref_mech, omega_m_est, dt)
        self._last_iq_ref = iq_ref

        omega_e = float(self.SMC_P_POLES) * omega_m_est
        vd, vq  = self._current_smc(id_meas, iq_meas, 0.0, iq_ref, omega_e)

        # Inverse Park — via coordinate_transform_blocks
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)

        # Logging
        if t >= self._log_next:
            self._log_t.append(t)
            self._log_spd.append(omega_m_est * 60.0 / (2.0 * math.pi))
            self._log_sref.append(omega_ref_mech * 60.0 / (2.0 * math.pi))
            self._log_iqr.append(iq_ref)
            self._log_iq.append(iq_meas)
            self._log_id.append(id_meas)
            self._log_next += 0.001

        self.output = VectorSignal(np.array([v_alpha, v_beta], dtype=np.float32), self.name)
        return self.output

    def compute_c(self, t: float, dt: float, input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
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
        return self.output

    def reset(self) -> None:
        super().reset()
        self._int_spd = 0.0
        self._int2_spd = 0.0
        self._last_iq_ref = 0.0
        self._last_theta_m = 0.0
        self._last_theta_m_unwrapped = 0.0
        self._omega_filt = 0.0
        self._e_prev = 0.0
        self._int_spd_prev = 0.0
        self._v_int_d = 0.0
        self._v_int_q = 0.0
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

    @property
    def log_data(self) -> dict:
        return {
            "t":         np.array(self._log_t,    dtype=np.float32),
            "speed":     np.array(self._log_spd,  dtype=np.float32),
            "speed_ref": np.array(self._log_sref, dtype=np.float32),
            "iq_ref":    np.array(self._log_iqr,  dtype=np.float32),
            "iq":        np.array(self._log_iq,   dtype=np.float32),
            "id":        np.array(self._log_id,   dtype=np.float32),
        }

    def __repr__(self) -> str:
        return f"SMCControllerBlock('{self.name}')"


__all__ = ["SMCControllerBlock", "_DB42S02"]