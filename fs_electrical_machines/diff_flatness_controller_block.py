"""
diff_flatness_controller_block.py
==================================
Differential Flatness Controller — PMSM trajectory tracking.

Signal sources (AURIX hardware):
  theta_m  : mechanical angle from encoder [rad], unwrapped
  ia, ib, ic : phase currents [A]

Architecture
------------
  theta_e = p * theta_m              exact electrical angle
  omega_m = d(theta_m)/dt + IIR      encoder speed (diagnostics / scheduling)

Flatness Control:
  - Desired trajectory: theta_ref(t), omega_ref(t)
  - Reference currents: id_ref, iq_ref  (from outer-loop planner or fixed)
  - Reference current derivatives: did_ref_dt, diq_ref_dt
  - Feedforward + feedback linearization in dq-frame
  - Park/InvPark delegated to coordinate_transform_blocks.py

SMO:
  - Classical Sliding Mode Observer (Ortega / Boldea form)
  - Runs on iα/iβ; produces estimated θ̂_e, ω̂_e
  - Can replace encoder path for sensorless operation or serve as cross-check

Bug-fixes vs original:
  - alpha_ref was used simultaneously as angular acceleration AND desired currents:
      fixed by splitting into {id_ref, iq_ref, did_ref_dt, diq_ref_dt}
  - Input guard was `< 7` but u[7] (ic) was accessed: guard raised to < 9
  - theta_e / omega_e were recomputed inside _df_control from args already
      available in compute_py; simplified to single computation site
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)

# =============================================================================
# Sliding Mode Observer  (standalone, no VectorBlock overhead)
# =============================================================================

class SlidingModeObserver:
    """
    Classical SMO for PMSM back-EMF estimation.

    Model (stationary frame, surface-wound, Ld=Lq=L):
        L * d(iα)/dt = vα - R*iα - eα
        L * d(iβ)/dt = vβ - R*iβ - eβ

    Sliding surface:
        s = [ĩα, ĩβ]  (ĩ = i_meas - î)

    Injection:
        eα_inj = k_smo * sign(ĩα)   (or sigmoid for chatter reduction)
        eβ_inj = k_smo * sign(ĩβ)

    Back-EMF extraction (low-pass):
        ê_α += Ts/tau * (eα_inj - êα)
        ê_β += Ts/tau * (eβ_inj - êβ)

    Angle extraction:
        θ̂_e = atan2(-êα, êβ)        (sign convention: eα=-ω_e*λpm*sin θ_e)

    Parameters
    ----------
    L_S    : float  — stator inductance [H]  (use mean(Ld,Lq) for IPMSM)
    R_S    : float  — stator resistance [Ω]
    k_smo  : float  — sliding gain (> max|e_BEMF|, start ≈ V_MAX/3)
    tau_e  : float  — BEMF LPF time constant [s]  (≈ 3/omega_e_min)
    sigmoid_w : float — sigmoid width for chatter reduction (0 → hard sign)
    """

    def __init__(self,
                 L_S: float   = 0.3675e-3,
                 R_S: float   = 0.285,
                 k_smo: float = 6.0,
                 tau_e: float = 2e-3,
                 sigmoid_w: float = 5.0):
        self.L_S       = L_S
        self.R_S       = R_S
        self.k_smo     = k_smo
        self.tau_e     = tau_e
        self.sigmoid_w = sigmoid_w

        # Observer states
        self._i_hat_alpha: float = 0.0
        self._i_hat_beta:  float = 0.0
        self._e_hat_alpha: float = 0.0
        self._e_hat_beta:  float = 0.0

        # Output
        self.theta_e_hat: float = 0.0
        self.omega_e_hat: float = 0.0
        self._theta_e_prev: float = 0.0

    # ── injection function ───────────────────────────────────────────────────
    def _inject(self, err: float) -> float:
        """Smooth sigmoid approximation of sign() to reduce chatter."""
        if self.sigmoid_w <= 0.0:
            return math.copysign(1.0, err) if err != 0.0 else 0.0
        return math.tanh(self.sigmoid_w * err)

    # ── step ────────────────────────────────────────────────────────────────
    def step(self,
             v_alpha: float, v_beta: float,
             i_alpha: float, i_beta: float,
             dt: float) -> Tuple[float, float]:
        """
        Advance observer by one time step.

        Parameters
        ----------
        v_alpha, v_beta : stator voltage commands (αβ frame) [V]
        i_alpha, i_beta : measured stator currents (αβ frame) [A]
        dt              : time step [s]

        Returns
        -------
        theta_e_hat : estimated electrical angle [rad]
        omega_e_hat : estimated electrical angular speed [rad/s]
        """
        L, R = self.L_S, self.R_S

        # Current estimation errors (sliding surface)
        err_alpha = i_alpha - self._i_hat_alpha
        err_beta  = i_beta  - self._i_hat_beta

        # Injection signals
        inj_alpha = self.k_smo * self._inject(err_alpha)
        inj_beta  = self.k_smo * self._inject(err_beta)

        # Observer current update (Euler)
        di_hat_alpha = (v_alpha - R * self._i_hat_alpha + inj_alpha) / L
        di_hat_beta  = (v_beta  - R * self._i_hat_beta  + inj_beta)  / L
        self._i_hat_alpha += dt * di_hat_alpha
        self._i_hat_beta  += dt * di_hat_beta

        # BEMF extraction via low-pass (1st-order)
        alpha_lpf = dt / (self.tau_e + dt)
        self._e_hat_alpha += alpha_lpf * (inj_alpha - self._e_hat_alpha)
        self._e_hat_beta  += alpha_lpf * (inj_beta  - self._e_hat_beta)

        # Angle from atan2  (eα = -ω_e·λpm·sin θ_e,  eβ = +ω_e·λpm·cos θ_e)
        theta_new = math.atan2(-self._e_hat_alpha, self._e_hat_beta)

        # Speed from angle difference (unwrap Δθ)
        d_theta = theta_new - self._theta_e_prev
        d_theta -= 2.0 * math.pi * math.floor((d_theta + math.pi) / (2.0 * math.pi))
        self.omega_e_hat = d_theta / dt if dt > 0.0 else 0.0
        self._theta_e_prev = theta_new
        self.theta_e_hat   = theta_new

        return self.theta_e_hat, self.omega_e_hat

    def reset(self) -> None:
        self._i_hat_alpha = 0.0
        self._i_hat_beta  = 0.0
        self._e_hat_alpha = 0.0
        self._e_hat_beta  = 0.0
        self.theta_e_hat  = 0.0
        self.omega_e_hat  = 0.0
        self._theta_e_prev = 0.0


# =============================================================================
# Differential Flatness Controller Block
# =============================================================================

class DFControllerBlock(VectorBlock):
    """
    Differential Flatness FOC Controller for PMSM.

    Input vector (9 elements):
        u[0] : theta_ref   — reference mechanical angle  [rad]
        u[1] : omega_ref   — reference mechanical speed  [rad/s]
        u[2] : id_ref      — reference d-axis current    [A]
        u[3] : iq_ref      — reference q-axis current    [A]
        u[4] : did_ref_dt  — d-axis current derivative   [A/s]
        u[5] : diq_ref_dt  — q-axis current derivative   [A/s]
        u[6] : theta_m     — measured mechanical angle   [rad], unwrapped
        u[7] : ia          — phase-a current             [A]
        u[8] : ib          — phase-b current             [A]
        u[9] : ic          — phase-c current             [A]

    Output: [v_alpha, v_beta]  (αβ-frame voltages for SVPWM)

    Notes
    -----
    The SMO runs every step regardless of mode.  Switch sensorless=True
    to route θ̂_e, ω̂_e into the controller instead of the encoder path.
    When sensorless=False the SMO is diagnostic only.
    """

    NUM_INPUTS  = 1   # single packed VectorSignal with 10 elements
    OUTPUT_SIZE = 2

    def __init__(self,
                 name: str = "dfc",
                 P_POLES: int     = 4,
                 R_S: float       = 0.285,
                 L_D: float       = 0.3675e-3,
                 L_Q: float       = 0.3675e-3,
                 LAMBDA_PM: float = 0.0014,
                 V_DC: float      = 17.0,
                 I_MAX: float     = 3.57,
                 dt_s: float      = 50e-6,
                 Kp_id: float     = 1.0,
                 Kp_iq: float     = 1.0,
                 smo_k: float     = 6.0,
                 smo_tau: float   = 2e-3,
                 smo_sigmoid_w: float = 5.0,
                 sensorless: bool = False,
                 dtype=None):
        super().__init__(name, dtype=dtype)

        # Motor parameters
        self.P_POLES   = P_POLES
        self.R_S       = R_S
        self.L_D       = L_D
        self.L_Q       = L_Q
        self.LAMBDA_PM = LAMBDA_PM
        self.V_DC      = V_DC
        self.I_MAX     = I_MAX
        self.V_MAX     = V_DC / math.sqrt(3.0)
        self.dt_s      = dt_s

        # Feedback gains (proportional only; add integrator externally if needed)
        self.Kp_id = Kp_id
        self.Kp_iq = Kp_iq

        self.sensorless  = sensorless
        self.vector_size = 2
        self.output_label = "[v_alpha, v_beta]"

        # Transform instances
        self._ct_clarke   = ClarkeTransformBlock("_dfc_clarke")
        self._ct_park     = ParkTransformBlock("_dfc_park")
        self._ct_inv_park = InvParkTransformBlock("_dfc_inv_park")

        # SMO instance  (L_S = mean(Ld, Lq) for slightly salient machine)
        self._smo = SlidingModeObserver(
            L_S       = 0.5 * (L_D + L_Q),
            R_S       = R_S,
            k_smo     = smo_k,
            tau_e     = smo_tau,
            sigmoid_w = smo_sigmoid_w,
        )

        # Diagnostics
        self._log_iq: list = []
        self._last_theta_m: float = 0.0
        self._omega_filt:   float = 0.0

        # Last αβ voltage commands fed into SMO (initialised to zero)
        self._v_alpha_prev: float = 0.0
        self._v_beta_prev:  float = 0.0

    # ── encoder speed estimator ──────────────────────────────────────────────
    def _get_speed_from_encoder(self, theta_m: float, dt: float) -> float:
        """IIR-filtered mechanical speed from encoder angle delta."""
        delta = theta_m - self._last_theta_m
        # Unwrap delta to (−π, +π)
        delta -= 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
        omega_raw = delta / dt if dt > 0.0 else 0.0
        alpha = 0.05 if abs(self._omega_filt) < 50.0 else 0.3
        self._omega_filt = (1.0 - alpha) * self._omega_filt + alpha * omega_raw
        self._last_theta_m = theta_m
        return self._omega_filt

    # ── coordinate helpers ───────────────────────────────────────────────────
    def _clarke(self, ia: float, ib: float, ic: float) -> Tuple[float, float]:
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32))
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(self, i_alpha: float, i_beta: float,
              theta_e: float) -> Tuple[float, float]:
        ab = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32))
        th = VectorSignal(np.array([theta_e], dtype=np.float32))
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(self, vd: float, vq: float,
                  theta_e: float) -> Tuple[float, float]:
        dq = VectorSignal(np.array([vd, vq], dtype=np.float32))
        th = VectorSignal(np.array([theta_e], dtype=np.float32))
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # ── flatness-based voltage computation ───────────────────────────────────
    def _df_control(self,
                    id_ref:     float, iq_ref:     float,
                    did_ref_dt: float, diq_ref_dt: float,
                    id_meas:    float, iq_meas:    float,
                    omega_e:    float) -> Tuple[float, float]:
        """
        Voltage commands from flatness-based feedback linearisation.

        PMSM voltage model (dq, generator sign convention omitted):
            vd = R·id + L_d·(d id/dt) − ω_e·L_q·iq
            vq = R·iq + L_q·(d iq/dt) + ω_e·L_d·id + ω_e·λ_pm

        Feedforward uses reference derivatives.
        Feedback uses proportional error (P-only; extend with integrator if
        needed — doing so inside the block risks windup without anti-windup).
        """
        # Feedforward (model inversion)
        vd_ff = self.R_S * id_ref + self.L_D * did_ref_dt - omega_e * self.L_Q * iq_ref
        vq_ff = self.R_S * iq_ref + self.L_Q * diq_ref_dt + omega_e * self.L_D * id_ref \
                + omega_e * self.LAMBDA_PM

        # Proportional feedback on current error
        vd_fb = self.Kp_id * (id_ref - id_meas)
        vq_fb = self.Kp_iq * (iq_ref - iq_meas)

        vd = vd_ff + vd_fb
        vq = vq_ff + vq_fb

        # Voltage magnitude saturation (circular in dq)
        mag = math.sqrt(vd * vd + vq * vq)
        if mag > self.V_MAX:
            scale = self.V_MAX / mag
            vd *= scale
            vq *= scale

        return vd, vq

    # ── compute_py ───────────────────────────────────────────────────────────
    def compute_py(self,
                   t: float,
                   dt: float,
                   input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
        zero = np.zeros(2, dtype=np.float32)
        if not input_values or len(input_values[0].value) < 10:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        theta_ref  = float(u[0])
        omega_ref  = float(u[1])
        id_ref     = float(u[2])
        iq_ref     = float(u[3])
        did_ref_dt = float(u[4])
        diq_ref_dt = float(u[5])
        theta_m    = float(u[6])
        ia         = float(u[7])
        ib         = float(u[8])
        ic         = float(u[9])

        # ── Clarke transform (always) ────────────────────────────────────────
        i_alpha, i_beta = self._clarke(ia, ib, ic)

        # ── SMO step  (receives previous-step voltage commands) ──────────────
        theta_e_smo, omega_e_smo = self._smo.step(
            self._v_alpha_prev, self._v_beta_prev,
            i_alpha, i_beta,
            dt if dt > 0.0 else self.dt_s,
        )

        # ── Select angle / speed source ──────────────────────────────────────
        if self.sensorless:
            theta_e_ctrl = theta_e_smo
            omega_e_ctrl = omega_e_smo
        else:
            omega_m       = self._get_speed_from_encoder(theta_m, dt if dt > 0.0 else self.dt_s)
            theta_e_ctrl  = float(self.P_POLES) * theta_m
            omega_e_ctrl  = float(self.P_POLES) * omega_m

        # ── Park transform ───────────────────────────────────────────────────
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e_ctrl)

        # ── Flatness control ─────────────────────────────────────────────────
        vd, vq = self._df_control(
            id_ref, iq_ref, did_ref_dt, diq_ref_dt,
            id_meas, iq_meas, omega_e_ctrl,
        )

        # ── Inverse Park → αβ ────────────────────────────────────────────────
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e_ctrl)

        # Store for SMO's next step
        self._v_alpha_prev = v_alpha
        self._v_beta_prev  = v_beta

        # Diagnostics
        self._log_iq.append(iq_meas)

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=np.float32), self.name
        )
        return self.output

    # ── public diagnostics ───────────────────────────────────────────────────
    @property
    def smo_theta_e(self) -> float:
        """Most recent SMO electrical angle estimate [rad]."""
        return self._smo.theta_e_hat

    @property
    def smo_omega_e(self) -> float:
        """Most recent SMO electrical speed estimate [rad/s]."""
        return self._smo.omega_e_hat