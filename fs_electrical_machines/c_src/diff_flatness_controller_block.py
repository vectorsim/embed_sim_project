"""
diff_flatness_controller_block.py
==================================
Differential Flatness Controller — PMSM trajectory tracking.

Includes 3-state EKF (id, iq, omega_m) matching C implementation.
"""

import math
import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)

_HERE = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"


# =============================================================================
# SpeedFusion (matches C implementation)
# =============================================================================

class SpeedFusion:
    """
    Speed-dependent complementary filter.

    theta_e = p * theta_m                           encoder always
    omega_e = (1-α)*omega_enc  +  α*omega_smo

    α rises from 0→1 via linear interpolation across [omega_lo, omega_hi].
    Encoder IIR coefficient adapts with speed (heavier smoothing at low speed).

    Matches C implementation in embed_sim_dfc_controller.c:
        - Linear alpha blending (not sigmoid)
        - Adaptive IIR: iir = iir_lo + alpha * (iir_hi - iir_lo)
        - SMO plausibility band: if |omega_smo - omega_enc_e| > PLAUS_BAND,
          use encoder fallback
    """

    def __init__(self,
                 P_POLES: int = 4,
                 omega_lo: float = 50.0,
                 omega_hi: float = 250.0,
                 iir_lo: float = 0.05,
                 iir_hi: float = 0.30,
                 plaus_band: float = 1000.0):
        self.p = float(P_POLES)
        self.omega_lo = omega_lo
        self.omega_hi = omega_hi
        self.iir_lo = iir_lo
        self.iir_hi = iir_hi
        self.plaus_band = plaus_band
        # State
        self._theta_m_prev: float = 0.0
        self._omega_enc_filt: float = 0.0
        self._omega_e_prev: float = 0.0
        # Diagnostics
        self.alpha: float = 0.0
        self.omega_enc: float = 0.0
        self.omega_smo_gated: float = 0.0

    def _alpha(self, omega_abs: float) -> float:
        """Linear alpha blending (matches C implementation)."""
        if omega_abs <= self.omega_lo:
            return 0.0
        if omega_abs >= self.omega_hi:
            return 1.0
        return (omega_abs - self.omega_lo) / (self.omega_hi - self.omega_lo)

    def update(self, theta_m: float, omega_smo_e: float,
               dt: float) -> Tuple[float, float]:
        """
        Returns (theta_e [rad], omega_e [rad/s elec]).

        Matches C implementation DFC_SpeedFusion_Update().
        """
        # Electrical angle directly from encoder
        theta_e = self.p * theta_m

        # Finite-difference mechanical speed from encoder
        delta = theta_m - self._theta_m_prev
        # Wrap to [-pi, pi]
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        omega_raw = delta / dt if dt > 0.0 else 0.0

        # Adaptive IIR smoothing
        alpha = self._alpha(abs(self._omega_e_prev))
        iir_coeff = self.iir_lo + alpha * (self.iir_hi - self.iir_lo)
        self._omega_enc_filt = (1.0 - iir_coeff) * self._omega_enc_filt + iir_coeff * omega_raw

        omega_enc_e = self.p * self._omega_enc_filt

        # SMO plausibility gate: if SMO deviates from encoder by more than
        # plaus_band, substitute encoder value
        if abs(omega_smo_e - omega_enc_e) > self.plaus_band:
            omega_smo_gated = omega_enc_e
        else:
            omega_smo_gated = omega_smo_e

        # Fused electrical speed
        omega_e = (1.0 - alpha) * omega_enc_e + alpha * omega_smo_gated

        # Encoder fallback: when SMO has not yet converged (|omega_smo| < 1)
        # but encoder is above threshold, substitute encoder electrical speed
        if (abs(omega_smo_e) < 1.0 and abs(self._omega_enc_filt) > self.omega_lo):
            omega_e = omega_enc_e

        # Update state
        self._theta_m_prev = theta_m
        self._omega_e_prev = omega_e
        self.alpha = alpha
        self.omega_enc = omega_enc_e
        self.omega_smo_gated = omega_smo_gated

        return theta_e, omega_e

    def reset(self) -> None:
        self._theta_m_prev = 0.0
        self._omega_enc_filt = 0.0
        self._omega_e_prev = 0.0
        self.alpha = 0.0
        self.omega_enc = 0.0
        self.omega_smo_gated = 0.0


# =============================================================================
# SlidingModeObserver (matches C implementation)
# =============================================================================

class SlidingModeObserver:
    """
    Classical SMO — BEMF estimation in αβ frame.

    Matches C implementation DFC_SMO_Step():
        - Linear saturation switch (not tanh)
        - Current observer with Euler integration
        - Back-EMF LPF with time constant tau_e
        - Speed spike clamp at DFC_SMO_OMEGA_MAX (3000 rad/s electrical)
        - Divergence guard: reinitialise from measured current if i_hat exceeds 2*I_MAX
    """

    def __init__(self,
                 L_avg: float = 0.3675e-3, R_S: float = 0.285,
                 k_smo: float = 2.0, tau_e: float = 0.0002,
                 i_max: float = 3.57,
                 omega_max: float = 3000.0,
                 warmup_steps: int = 400):
        self.L_avg = L_avg
        self.R_S = R_S
        self.k_smo = k_smo
        self.tau_e = tau_e
        self.i_max = i_max
        self.omega_max = omega_max
        self.warmup_steps = warmup_steps
        self._warmup_cnt = 0

        # State
        self._i_hat_alpha: float = 0.0
        self._i_hat_beta: float = 0.0
        self._e_hat_alpha: float = 0.0
        self._e_hat_beta: float = 0.0
        self._theta_e_prev: float = 0.0
        self.theta_e_hat: float = 0.0
        self.omega_e_hat: float = 0.0
        self.omega_e_filt: float = 0.0

    def _switch(self, error: float) -> float:
        """
        Smooth sign approximation (linear saturation).
        Matches C implementation DFC_SMOSwitch().
        """
        width = 0.01
        arg = error / width

        if arg > 5.0:
            return 1.0
        elif arg < -5.0:
            return -1.0
        else:
            return arg * 0.2

    def step(self, v_alpha: float, v_beta: float,
             i_alpha: float, i_beta: float, dt: float) -> Tuple[float, float]:
        """
        Returns (theta_e_hat, omega_e_filt).
        Matches C implementation DFC_SMO_Step().
        """
        inv_L = 1.0 / self.L_avg if self.L_avg > 1e-9 else 1.0 / 1e-9
        lpf_alpha = dt / (self.tau_e + dt) if dt > 0.0 else 0.0

        # Divergence guard: if i_hat exceeds 2x physical current limit,
        # reinitialise from measured current
        if (abs(self._i_hat_alpha) > 2.0 * self.i_max or
            abs(self._i_hat_beta) > 2.0 * self.i_max):
            self._i_hat_alpha = i_alpha
            self._i_hat_beta = i_beta
            self._e_hat_alpha = 0.0
            self._e_hat_beta = 0.0
            self.omega_e_hat = 0.0
            self.omega_e_filt = 0.0
            # theta_e_prev preserved intentionally

        # Current errors
        err_alpha = i_alpha - self._i_hat_alpha
        err_beta = i_beta - self._i_hat_beta

        # Switching signals
        sw_alpha = self.k_smo * self._switch(err_alpha)
        sw_beta = self.k_smo * self._switch(err_beta)

        # Current observer (Euler)
        self._i_hat_alpha += dt * inv_L * (v_alpha - self.R_S * self._i_hat_alpha - sw_alpha)
        self._i_hat_beta += dt * inv_L * (v_beta - self.R_S * self._i_hat_beta - sw_beta)

        # Back-EMF LPF
        self._e_hat_alpha += lpf_alpha * (sw_alpha - self._e_hat_alpha)
        self._e_hat_beta += lpf_alpha * (sw_beta - self._e_hat_beta)

        # Angle from back-EMF (sign convention: atan2(e_alpha, -e_beta))
        theta_e_new = math.atan2(self._e_hat_alpha, -self._e_hat_beta)

        # Finite-difference speed
        delta = theta_e_new - self._theta_e_prev
        while delta > math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi

        self._warmup_cnt += 1
        if dt > 0.0 and self._warmup_cnt > self.warmup_steps:
            self.omega_e_hat = delta / dt
            # Spike clamp: discard if exceeds physical ceiling
            if abs(self.omega_e_hat) > self.omega_max:
                self.omega_e_hat = self.omega_e_filt
        else:
            self.omega_e_hat = 0.0

        # LPF on speed estimate
        self.omega_e_filt += lpf_alpha * (self.omega_e_hat - self.omega_e_filt)

        self._theta_e_prev = theta_e_new
        self.theta_e_hat = theta_e_new

        return self.theta_e_hat, self.omega_e_filt

    def reset(self) -> None:
        self._i_hat_alpha = 0.0
        self._i_hat_beta = 0.0
        self._e_hat_alpha = 0.0
        self._e_hat_beta = 0.0
        self._theta_e_prev = 0.0
        self.theta_e_hat = 0.0
        self.omega_e_hat = 0.0
        self.omega_e_filt = 0.0
        self._warmup_cnt = 0


# =============================================================================
# =============================================================================
class DFControllerBlock(VectorBlock):
    """
    Differential Flatness FOC Controller — drop-in for SMCControllerBlock.

    Accepts the 5-element SMC_Input_T bus from CtrlPacker unchanged.
    All outer-loop logic (speed P-loop, theta_ref integration, SpeedFusion)
    lives here so the CodeGen boundary and simulation wiring are identical
    to the SMC simulation.
    """

    # ── EmbedSim CodeGen attributes ──────────────────────────────────────────
    NUM_INPUTS = 1
    OUTPUT_SIZE = 2

    # Matches SMC_Input_T exactly — StepGenerator reads these
    INPUT_NAMES = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP = [0, 1, 2, 3, 4]

    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m": "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia": "Phase-A current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ib": "Phase-B current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ic": "Phase-C current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
    }

    # CodeGen C linkage
    step_func = "DFC_Controller_Step"
    state_struct = "DFC_State_T"
    init_func = "DFC_Controller_Init"
    C_INIT_ARGS = ["dt_s"]
    C_SOURCES = [
        "embed_sim_dfc_controller.c",
    ]
    C_HEADERS = [
        "embed_sim_dfc_controller.h",
    ]

    # Cython wrapper metadata
    PYX_FILE = str(_C_SRC / "dfc_controller_wrapper.pyx")

    # Custom C code emission for code generation
    C_CUSTOM_EMIT = """\
        /* --- dfc_controller (DFControllerBlock) --- */
        /* DFC_Controller_Step() outputs physical voltages [V].                */
        /* The SVPWM block expects normalised [-1,+1] references.              */
        /* The caller must divide by SVPWM_GAIN = V_DC/2 before SVPWM.         */
        DFC_Input_T   u_dfc;
        DFC_Output_T  y_dfc_out;
        real32_T      y_dfc[2];

        u_dfc.omega_ref_mech = in->omega_ref_mech;
        u_dfc.theta_m        = in->theta_m;
        u_dfc.ia             = in->ia;
        u_dfc.ib             = in->ib;
        u_dfc.ic             = in->ic;

        DFC_Controller_Step(&dfc_state, &u_dfc, dt, &y_dfc_out);

        /* Convert physical voltages to normalised for SVPWM */
        y_dfc[0] = y_dfc_out.v_alpha / DFC_V_MAX;
        y_dfc[1] = y_dfc_out.v_beta / DFC_V_MAX;"""

    DIAG_STEPS: int = 200 if os.environ.get("DFC_DBG") == "1" else 20
    _SQRT3 = math.sqrt(3.0)

    # Sensorless cold-start state machine constants (mirror C #defines)

    # Observer modes (matches C enum)
    OBS_MODE_SMO = 0

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(self,
                 name: str = "dfc",
                 # Motor parameters
                 P_POLES: int = 4,
                 R_S: float = 0.285,
                 L_D: float = 0.0003675,
                 L_Q: float = 0.0003675,
                 LAMBDA_PM: float = 0.0014,
                 V_DC: float = 17.0,
                 I_MAX: float = 3.57,
                 dt_s: float = 50e-6,
                 # DFC current feedback gains (matches C: Kp_id, Kp_iq)
                 Kp_id: float = 0.4,
                 Kp_iq: float = 8.0,
                 # Outer speed P-loop (matches C: DFC_KP_SPEED)
                 Kp_speed: float = 0.4,
                 # SMO parameters (matches C)
                 smo_k: float = 2.0,
                 smo_tau: float = 0.0002,
                 smo_omega_max: float = 3000.0,
                 # EKF parameters (matches C defaults)
                 ekf_q_i: float = 1e-4,
                 ekf_q_omega: float = 1.0,       # electrical speed process noise
                 ekf_r_i: float = 1e-4,
                 ekf_p0_i: float = 1.0,
                 ekf_p0_omega: float = 1e6,    # cold-start: speed truly unknown
                 # SpeedFusion parameters (matches C)
                 fusion_omega_lo: float = 50.0,
                 fusion_omega_hi: float = 250.0,
                 fusion_iir_lo: float = 0.05,
                 fusion_iir_hi: float = 0.30,
                 fusion_plaus_band: float = 1000.0,
                 # Current derivative LPF time constant (matches C: DFC_DIQ_TAU)
                 diq_tau: float = 0.001,
                 # Observer mode selection
                 observer_mode: int = 0,  # 0=SMO, 1=EKF, 2=BLEND
                 blend_w: float = 0.5,
                 use_c_backend: bool = False,
                 dtype=None):
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.P_POLES = P_POLES
        self.R_S = R_S
        self.L_D = L_D
        self.L_Q = L_Q
        self.LAMBDA_PM = LAMBDA_PM
        self.V_DC = V_DC
        self.I_MAX = I_MAX
        self.V_MAX = V_DC / self._SQRT3
        self.dt_s = dt_s
        self.Kp_id = Kp_id
        self.Kp_iq = Kp_iq
        self.Kp_speed = Kp_speed
        self.diq_tau = diq_tau

        self.vector_size = 2
        self.output_label = "[v_alpha, v_beta]"
        self.is_dynamic = False

        # Coordinate transforms (Python implementations)
        self._ct_clarke = ClarkeTransformBlock("_dfc_clarke", use_c_backend=False)
        self._ct_park = ParkTransformBlock("_dfc_park", use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_dfc_inv_park", use_c_backend=False)

        # SMO (Python implementation - matches C)
        L_avg = (L_D + L_Q) * 0.5
        self._smo = SlidingModeObserver(
            L_avg=L_avg,
            R_S=R_S,
            k_smo=smo_k,
            tau_e=smo_tau,
            i_max=I_MAX,
            omega_max=smo_omega_max,
            warmup_steps=400,
        )

        # EKF — 4-state sensorless αβ-frame

        # SpeedFusion (Python implementation - matches C)
        self.fusion = SpeedFusion(
            P_POLES=P_POLES,
            omega_lo=fusion_omega_lo,
            omega_hi=fusion_omega_hi,
            iir_lo=fusion_iir_lo,
            iir_hi=fusion_iir_hi,
            plaus_band=fusion_plaus_band,
        )

        # Internal state — mirrored in DFC_State_T on AURIX
        self._v_alpha_prev: float = 0.0
        self._v_beta_prev: float = 0.0
        self._theta_ref: float = 0.0
        self._iq_ref_prev: float = 0.0
        self._diq_filt: float = 0.0

        # Log data — extended with EKF diagnostics
        self.log_data: dict = {
            "t":             [],
            "speed_ref":     [],
            "iq_ref":        [],
            "id":            [],
            "iq":            [],
            "alpha":         [],
            "omega_e":       [],
            "omega_smo":     [],
            "omega_ekf":     [],
        }

        # C backend
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

        print(f"[DFC] Differential Flatness Controller initialized")
        print(f"[DFC] Observer mode: {['SMO', 'EKF', 'BLEND'][observer_mode]}")
        if observer_mode == 2:
            print(f"[DFC] Blend weight: {blend_w:.2f}")
        print(f"[DFC] Speed gains: Kp_speed={self.Kp_speed:.4f} A/(rad/s)")
        print(f"[DFC] Current gains: Kp_id={self.Kp_id:.2f} V/A, Kp_iq={self.Kp_iq:.2f} V/A")
        print(f"[DFC] SMO: k={smo_k:.1f} V, tau={smo_tau * 1000:.1f} ms")
        print(f"[DFC] EKF: 4-state sensorless ab  "
              f"q_i={ekf_q_i:.1e}  q_omega={ekf_q_omega:.1e}  "
              f"r_i={ekf_r_i:.1e}  p0_omega={ekf_p0_omega:.1e}")
        print(f"[DFC] C backend: {'ENABLED' if use_c_backend else 'disabled (using Python)'}")

    # ── C backend loader ──────────────────────────────────────────────────────

    def _load_wrapper(self) -> None:
        """Load the C extension wrapper for DFC controller."""
        try:
            from dfc_controller_wrapper import DFCControllerWrapper
            self._wrapper = DFCControllerWrapper(
                self.V_DC, self.P_POLES,
                self.R_S, self.L_D, self.L_Q,
                self.LAMBDA_PM, self.I_MAX, self.dt_s,
                self.Kp_speed, self.Kp_id, self.Kp_iq)
            # Set observer mode
        except ImportError as exc:
            raise ImportError(
                "dfc_controller_wrapper not found. Build with:\n"
                "  cd fs_electrical_machines/c_src\n"
                "  python setup_dfc_controller.py build_ext --inplace"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"DFCControllerWrapper instantiation failed: {exc}") from exc

    # ── Transform helpers — always delegate to coordinate_transform_blocks ────

    def _clarke(self, ia: float, ib: float, ic: float) -> Tuple[float, float]:
        """Clarke abc→αβ via ClarkeTransformBlock.compute_py()."""
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_clarke")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(self, i_alpha: float, i_beta: float, theta_e: float) -> Tuple[float, float]:
        """Park αβ→dq via ParkTransformBlock.compute_py()."""
        ab = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_park")
        th = VectorSignal(np.array([theta_e], dtype=np.float32), "_park")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(self, vd: float, vq: float, theta_e: float) -> Tuple[float, float]:
        """Inverse Park dq→αβ via InvParkTransformBlock.compute_py()."""
        dq = VectorSignal(np.array([vd, vq], dtype=np.float32), "_inv_park")
        th = VectorSignal(np.array([theta_e], dtype=np.float32), "_inv_park")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # ── flatness voltage law (matches C implementation DFC_VoltageLaw) ───────

    def _df_control(self,
                    iq_ref: float,
                    diq_dt: float,
                    id_meas: float,
                    iq_meas: float,
                    omega_e: float) -> Tuple[float, float]:
        """
        PMSM flatness voltage law.

        Matches C implementation DFC_VoltageLaw():
            vd = -ω_e·L_q·iq_ref + Kp_id·(0 - id_meas)
            vq = R·iq_ref + L_q·diq_dt + ω_e·λ_pm + Kp_iq·(iq_ref - iq_meas)

        Note: id_ref = 0 (MTPA)
        """
        vd = (-omega_e * self.L_Q * iq_ref
              + self.Kp_id * (0.0 - id_meas))

        vq = (self.R_S * iq_ref
              + self.L_Q * diq_dt
              + omega_e * self.LAMBDA_PM
              + self.Kp_iq * (iq_ref - iq_meas))

        # Hexagon voltage saturation
        mag = math.sqrt(vd * vd + vq * vq)
        if mag > self.V_MAX:
            scale = self.V_MAX / mag
            vd *= scale
            vq *= scale

        return vd, vq

    # ── log helper ───────────────────────────────────────────────────────────

    def _log_ekf_step(self, t, omega_ref_mech, iq_ref, id_meas, iq_meas,
                      omega_e, omega_e_smo, omega_ekf_mech, p_omega):
        """Append one row to log_data."""
        self.log_data["t"].append(t)
        self.log_data["speed_ref"].append(omega_ref_mech * 60.0 / (2.0 * math.pi))
        self.log_data["iq_ref"].append(iq_ref)
        self.log_data["id"].append(id_meas)
        self.log_data["iq"].append(iq_meas)
        self.log_data["alpha"].append(self.fusion.alpha)
        self.log_data["omega_e"].append(omega_e)
        self.log_data["omega_smo"].append(omega_e_smo / self.P_POLES)
        self.log_data["omega_ekf"].append(omega_ekf_mech)

    # ── compute_py (Python implementation with EKF) ───────────────────────────

    def compute_py(self,
                   t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None
                   ) -> VectorSignal:

        zero = np.zeros(2, dtype=np.float32)
        if not input_values or len(input_values[0].value) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        _dt = dt if dt > 0.0 else self.dt_s

        omega_ref_mech = float(u[0])
        theta_m = float(u[1])
        ia, ib, ic = float(u[2]), float(u[3]), float(u[4])

        # ── 0. Startup timer ─────────────────────────────────────────────────

        # ── 1. Clarke ─────────────────────────────────────────────────────────
        i_alpha, i_beta = self._clarke(ia, ib, ic)

        # ── 2. SMO (always runs, feeds SpeedFusion) ──────────────────────────
        _, omega_e_smo = self._smo.step(
            self._v_alpha_prev, self._v_beta_prev,
            i_alpha, i_beta, _dt)

        omega_ekf_mech = 0.0  # no EKF

        # ── 4. SpeedFusion → theta_e (encoder), omega_e (complementary) ──────
        theta_e, omega_e = self.fusion.update(theta_m, omega_e_smo, _dt)

        # ── 5. Speed measurement: SpeedFusion (encoder IIR + SMO) ──────────
        omega_meas_mech = self.fusion._omega_enc_filt

        # ── 6. Outer speed P-loop → iq_ref ───────────────────────────────────
            # Force iq_ref=0 during alignment so only id flows
            speed_err = 0.0
            iq_ref    = 0.0
        else:
            speed_err = omega_ref_mech - omega_meas_mech
            iq_ref = self.Kp_speed * speed_err
            iq_ref = max(-self.I_MAX, min(self.I_MAX, iq_ref))

        # ── 7. Current derivative (LPF-filtered finite difference) ───────────
        if _dt > 0.0:
            diq_raw = (iq_ref - self._iq_ref_prev) / _dt
        else:
            diq_raw = 0.0

        lpf_alpha = _dt / (self.diq_tau + _dt) if _dt > 0.0 else 0.0
        self._diq_filt = (1.0 - lpf_alpha) * self._diq_filt + lpf_alpha * diq_raw

        # Clamp: I_MAX / DIQ_TAU ceiling
        diq_ref_dt = max(-self.I_MAX / self.diq_tau,
                         min(self.I_MAX / self.diq_tau, self._diq_filt))
        self._iq_ref_prev = iq_ref

        # ── 8. Park ───────────────────────────────────────────────────────────
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        # ── 9. Flatness voltage law ───────────────────────────────────────────
        vd, vq = self._df_control(
            iq_ref, diq_ref_dt, id_meas, iq_meas, omega_e)

        # ── 10. Inverse Park → αβ ─────────────────────────────────────────────
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)

        # Store voltages for SMO next step (z-1)
        self._v_alpha_prev = v_alpha
        self._v_beta_prev = v_beta

        # ── Log ───────────────────────────────────────────────────────────────
        self._log_ekf_step(t, omega_ref_mech, iq_ref, id_meas, iq_meas,
                           omega_e, omega_e_smo, omega_ekf_mech, _p_omega)

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=np.float32), self.name)
        return self.output

    # ── compute_c (C backend implementation) ──────────────────────────────────

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        C backend compute method for DFControllerBlock.

        Uses the compiled C extension for high-performance execution.
        """
        zero = np.array([0.0, 0.0], dtype=np.float32)
        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        # Copy inputs to float32 array for C wrapper
        inputs = np.zeros(5, dtype=np.float32)
        inputs[0] = float(u[0])  # omega_ref_mech
        inputs[1] = float(u[1])  # theta_m
        inputs[2] = float(u[2])  # ia
        inputs[3] = float(u[3])  # ib
        inputs[4] = float(u[4])  # ic

        # Call C wrapper
        self._wrapper.set_inputs(inputs)
        self._wrapper.compute(float(dt))
        outputs = self._wrapper.get_outputs()

        self.output = VectorSignal(outputs, self.name)
        return self.output

    # ── compute (dispatcher) ──────────────────────────────────────────────────

    def compute(self, t: float, dt: float,
                input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        """
        Dispatch to either Python or C implementation based on use_c_backend.
        """
        if self.use_c_backend and self._wrapper is not None:
            return self.compute_c(t, dt, input_values)
        else:
            return self.compute_py(t, dt, input_values)

    # ── Public API methods ────────────────────────────────────────────────────

    def get_diagnostics(self) -> dict:
        """
        Return current diagnostic values.

        Returns
        -------
        dict with keys:
            speed_est_mech: Mechanical speed estimate [rad/s]
            iq_ref: q-axis current reference [A]
            id_meas: Measured d-axis current [A]
            iq_meas: Measured q-axis current [A]
            fusion_alpha: SpeedFusion weight
            omega_smo: SMO mechanical speed [rad/s]
            omega_ekf: EKF mechanical speed [rad/s]
            observer_mode: Current observer mode
        """
        # Speed estimate from SpeedFusion
        speed_est = self.fusion._omega_enc_filt

        return {
            "speed_est_mech": speed_est,
            "iq_ref": self.log_data["iq_ref"][-1] if self.log_data["iq_ref"] else 0.0,
            "id_meas": self.log_data["id"][-1] if self.log_data["id"] else 0.0,
            "iq_meas": self.log_data["iq"][-1] if self.log_data["iq"] else 0.0,
            "fusion_alpha": self.fusion.alpha,
            "omega_smo": self.log_data["omega_smo"][-1] if self.log_data["omega_smo"] else 0.0,
            "omega_ekf": self.log_data["omega_ekf"][-1] if self.log_data["omega_ekf"] else 0.0,
        }

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset the controller state."""
        super().reset()
        # Reset Python state
        self._v_alpha_prev = 0.0
        self._v_beta_prev = 0.0
        self._theta_ref = 0.0
        self._iq_ref_prev = 0.0
        self._diq_filt = 0.0
        # Reset transform blocks
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()
        # Reset SMO and SpeedFusion
        self._smo.reset()
        self.fusion.reset()
        # Reset EKF
        # Clear log data
        self.log_data = {k: [] for k in self.log_data}
        # Reset C wrapper if present
        if self._wrapper is not None:
            self._wrapper.reset()

    # ── diagnostics properties ────────────────────────────────────────────────

    @property
    def smo_theta_e(self) -> float:
        """SMO estimated electrical angle [rad] (diagnostic only)."""
        return self._smo.theta_e_hat

    @property
    def smo_omega_e(self) -> float:
        """SMO estimated electrical speed [rad/s] (diagnostic only)."""
        return self._smo.omega_e_filt

    @property
    def ekf_omega_m(self) -> float:
        """EKF estimated mechanical speed [rad/s] (diagnostic only)."""
        return self.fusion._omega_enc_filt

    @property
    def ekf_theta_e(self) -> float:
        """EKF estimated electrical angle [rad] (diagnostic only)."""
        return self.fusion.theta_e

    def __repr__(self) -> str:
        return f"DFControllerBlock('{self.name}', mode={mode_str})"


__all__ = ["DFControllerBlock", "SpeedFusion", "SlidingModeObserver"]