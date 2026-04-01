# mpc_controller_block.py
"""
Model Predictive Control for PMSM — NANOTEC DB42S02

ARCHITECTURE: True 3-state speed-tracking MPC.

State vector : x = [id, iq, omega_m]
Input vector : u = [vd, vq]
Output       : [v_alpha, v_beta]

Cost function (minimised analytically at each step):
    J = Σ_{k=1}^{N} [ Q_id · id_k²
                     + Q_omega · (omega_k − omega_ref)²
                     + R_vd · vd²  +  R_vq · vq² ]

No separate outer speed loop. Speed tracking is embedded in J via Q_omega.

ANALYTICAL CLOSED-FORM SOLUTION (O(N), no iteration):
─────────────────────────────────────────────────────
Free-run trajectory (u = 0, BEMF handled by feedforward):
    id_free(k+1) = a·id_free + (dt/L)·ωe·L·iq_free
    iq_free(k+1) = a·iq_free − (dt/L)·ωe·L·id_free
    ω_free(k+1)  = ω_free + (dt/J)·(KT·iq_free − B·ω_free)

Step-response coefficients for unit vq:
    bk  = accumulated iq response   (current dynamics)
    ek  = accumulated ω  response   = Σ_{j<k} (dt/J)·KT·b_{j}

Optimal inputs:
    vd_mpc = Q_id · Σ bk·(0 − id_free_k)    /  (Q_id·Σbk² + R_vd)
    vq_mpc = Q_omega · Σ ek·(ωref − ω_free_k) / (Q_omega·Σek² + R_vq)

BEMF feedforward (exact cancellation at every step):
    vd = vd_mpc + ed_hat
    vq = vq_mpc + eq_hat

BEMF clamp (prevents SMO saturation artefacts):
    |ed_hat|, |eq_hat| ≤ ωe · λ_pm   (physical maximum back-EMF)

SMO provides ed_hat, eq_hat for disturbance rejection at speed.
At startup (ωe ≈ 0): clamp → 0, feedforward vanishes (correct physics).
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)


class _DB42S02:
    """NANOTEC DB42S02 motor parameters."""
    P_POLES    = 4
    R_S        = 0.19
    L_D        = 0.125e-3
    L_Q        = 0.125e-3
    LAMBDA_PM  = 0.0014
    J_ROTOR    = 2.4e-6
    B_FRICTION = 1e-6
    I_MAX      = 3.57
    V_DC       = 17.0
    V_MAX      = 17.0 / 2.0        # = V_DC/2 = 8.5 V  (SVPWM normalised limit)
    # Note: hexagon limit is V_DC/√3 = 9.815 V, but SVPWMPackBlock normalises
    # by V_DC/2, so the MPC must clamp to V_DC/2 to keep normalised output ≤ 1.0.
    KT         = 1.5 * 4 * 0.0014

    SMO_K  = 4.68
    SMO_FC = 1000.0


@dataclass
class MPCState:
    """MPC state vector: [id, iq, omega_m]."""
    id:    float = 0.0
    iq:    float = 0.0
    omega: float = 0.0   # mechanical rad/s


class MPCControllerBlock(VectorBlock):
    """
    True 3-state receding-horizon MPC for PMSM speed and current control.

    Input bus  : [omega_ref_mech, theta_m, ia, ib, ic, omega_m_meas]
    Output bus : [v_alpha, v_beta]

    No external speed loop. Speed tracking is inside the MPC cost function.
    """


    # ── CodeGen attributes (EmbedSim StepGenerator) ──────────────────────
    C_SOURCES   = ["embed_sim_mpc_controller.c"]
    C_HEADERS   = ["embed_sim_mpc_controller.h"]
    state_struct = "MPC_Controller_T"
    step_func    = "MPC_Controller_Step"
    init_func    = "MPC_Controller_Init"

    C_CUSTOM_EMIT = (
        "    /* --- mpc (MPCControllerBlock) --- */\n"
        "    {\n"
        "        MPC_Input_T   u_mpc;\n"
        "        MPC_Output_T  y_mpc_out;\n"
        "        real32_T      y_mpc[2];\n"
        "\n"
        "        u_mpc.omega_ref_mech = in->omega_ref_mech;\n"
        "        u_mpc.theta_m        = in->theta_m;\n"
        "        u_mpc.ia             = in->ia;\n"
        "        u_mpc.ib             = in->ib;\n"
        "        u_mpc.ic             = in->ic;\n"
        "\n"
        "        MPC_Controller_Step(&mpc_state, &u_mpc, dt, &y_mpc_out);\n"
        "\n"
        "        y_mpc[0] = y_mpc_out.v_alpha;\n"
        "        y_mpc[1] = y_mpc_out.v_beta;\n"
        "    }"
    )

    def __init__(
            self,
            name:          str   = "mpc",
            P_POLES:       int   = _DB42S02.P_POLES,
            R_S:           float = _DB42S02.R_S,
            L:             float = _DB42S02.L_D,
            LAMBDA_PM:     float = _DB42S02.LAMBDA_PM,
            J:             float = _DB42S02.J_ROTOR,
            B:             float = _DB42S02.B_FRICTION,
            I_MAX:         float = _DB42S02.I_MAX,
            V_MAX:         float = _DB42S02.V_MAX,
            N:             int   = 10,
            Q_id:          float = 10.0,
            Q_iq:          float = 100.0,
            Q_omega:       float = 500.0,
            R_vd:          float = 0.01,
            R_vq:          float = 0.01,
            dt_s:          float = 50e-6,
            SMO_K:         float = _DB42S02.SMO_K,
            SMO_FC:        float = _DB42S02.SMO_FC,
            use_c_backend: bool  = False,
            dtype=None,
    ) -> None:

        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        self.P_POLES   = int(P_POLES)
        self.R_S       = float(R_S)
        self.L         = float(L)
        self.LAMBDA_PM = float(LAMBDA_PM)
        self.J         = float(J)
        self.B         = float(B)
        self.I_MAX     = float(I_MAX)
        self.V_MAX     = float(V_MAX)
        self.KT        = 1.5 * float(P_POLES) * float(LAMBDA_PM)

        # SVPWM_GAIN = V_DC/2: SVPWMPackBlock normalises physical volts by this
        # value to produce duty-cycle references in [-1, +1].
        # V_MAX is already set to V_DC/2 above, so SVPWM_GAIN == V_MAX.
        self.SVPWM_GAIN = self.V_MAX

        self.N      = int(N)
        self.Q_id   = float(Q_id)
        self.Q_iq   = float(Q_iq)
        self.Q_omega = float(Q_omega)
        self.R_vd   = float(R_vd)
        self.R_vq   = float(R_vq)
        self._dt    = float(dt_s)

        self.SMO_K      = float(SMO_K)
        self.SMO_FC     = float(SMO_FC)
        wc_smo          = 2.0 * math.pi * self.SMO_FC
        self._smo_alpha = wc_smo * self._dt / (1.0 + wc_smo * self._dt)

        # SMO state
        self._i_alpha_hat  = 0.0
        self._i_beta_hat   = 0.0
        self._e_alpha_filt = 0.0
        self._e_beta_filt  = 0.0
        self._v_alpha_prev = 0.0
        self._v_beta_prev  = 0.0

        # Speed estimator state
        self._omega_filt   = 0.0
        self._last_theta_m = 0.0

        # Soft-start: ramp iq limit from 0 to I_MAX over SOFTSTART_T seconds
        self._SOFTSTART_T = 0.1
        self._iq_limit    = 0.0

        # Integral correction for MPC steady-state speed offset
        self._speed_err_integral = 0.0

        # Sub-blocks (all Python, no C backend)
        self._clarke   = ClarkeTransformBlock("_clarke",    use_c_backend=False)
        self._park     = ParkTransformBlock("_park",        use_c_backend=False)
        self._inv_park = InvParkTransformBlock("_inv_park", use_c_backend=False)

        # Logging
        self._log_t         = []
        self._log_speed     = []
        self._log_speed_ref = []
        self._log_iq_ref    = []
        self._log_iq        = []
        self._log_id        = []
        self._log_next      = 0.0

        print(f"\n[MPC Controller] Initialized  (3-state speed-tracking MPC)")
        print(f"  Prediction horizon : N={self.N}")
        print(f"  Weights            : Q_id={Q_id:.1f}  Q_iq={Q_iq:.1f}  Q_omega={Q_omega:.1f}")
        print(f"  Control weights    : R_vd={R_vd:.4f}  R_vq={R_vq:.4f}")
        print(f"  SMO                : k={self.SMO_K:.2f} V  fc={self.SMO_FC:.0f} Hz")

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _get_speed(self, theta_m: float, dt: float) -> float:
        """Estimate mechanical speed from encoder angle with IIR filter."""
        if dt <= 0.0:
            return self._omega_filt
        delta = theta_m - self._last_theta_m
        # Unwrap
        delta -= 2.0 * math.pi * math.floor((delta + math.pi) / (2.0 * math.pi))
        omega_raw = delta / dt
        self._omega_filt = 0.8 * self._omega_filt + 0.2 * omega_raw
        self._last_theta_m = theta_m
        return self._omega_filt

    def _smo_step(self, i_alpha: float, i_beta: float,
                  v_alpha: float, v_beta: float, dt: float) -> None:
        """Sliding-mode back-EMF observer (αβ frame)."""
        if dt <= 0.0:
            return
        inv_L = 1.0 / self.L
        k     = self.SMO_K
        alpha = self._smo_alpha

        err_alpha = i_alpha - self._i_alpha_hat
        err_beta  = i_beta  - self._i_beta_hat

        sw_alpha = k * math.tanh(err_alpha / 0.01)
        sw_beta  = k * math.tanh(err_beta  / 0.01)

        self._i_alpha_hat += dt * inv_L * (v_alpha - self.R_S * self._i_alpha_hat - sw_alpha)
        self._i_beta_hat  += dt * inv_L * (v_beta  - self.R_S * self._i_beta_hat  - sw_beta)

        self._e_alpha_filt += alpha * (sw_alpha - self._e_alpha_filt)
        self._e_beta_filt  += alpha * (sw_beta  - self._e_beta_filt)

    # ------------------------------------------------------------------ solver

    def _solve_mpc(self, x0: MPCState, omega_ref: float,
                   ed_hat: float, eq_hat: float, dt: float) -> Tuple[float, float]:
        """
        3-state analytical MPC solver.

        Returns (vd_total, vq_total) including BEMF feedforward.

        Free-run is computed WITHOUT ed/eq_hat (BEMF handled by feedforward).
        This keeps the free-run well-conditioned at all speeds.

        vd_total = vd_mpc + ed_hat
        vq_total = vq_mpc + eq_hat
        """
        omega_e = float(self.P_POLES) * x0.omega   # electrical rad/s
        inv_L   = 1.0 / self.L
        a       = 1.0 - dt * self.R_S * inv_L      # current decay
        b       = dt * inv_L                         # input gain
        dt_J    = dt / self.J                        # speed integration

        id_free    = x0.id
        iq_free    = x0.iq
        omega_free = x0.omega

        bk = 0.0; ek = 0.0

        sum_bk_err_d  = 0.0   # Σ bk·(0 − id_free)           → vd numerator
        sum_bk2       = 0.0   # Σ bk²                         → vd/vq denominator
        sum_ek_err    = 0.0   # Σ ek·(omega_ref − omega_free)  → vq numerator
        sum_ek2       = 0.0   # Σ ek²                         → vq denominator

        for _ in range(self.N):
            # Free-run: cross-coupling only (no BEMF — handled by feedforward)
            f_d     = dt * inv_L * ( omega_e * self.L * iq_free)
            f_q     = dt * inv_L * (-omega_e * self.L * id_free)
            f_omega = dt_J * (self.KT * iq_free - self.B * omega_free)

            id_free    = a * id_free    + f_d
            iq_free    = a * iq_free    + f_q
            omega_free = omega_free     + f_omega

            # Step-response coefficients
            bk = bk * a + b            # iq response to unit vq at step k
            ek += dt_J * self.KT * bk  # omega response: accumulate current bk

            # Gradient accumulation
            sum_bk_err_d  += bk * (0.0       - id_free)
            sum_ek_err    += ek * (omega_ref  - omega_free)
            sum_bk2       += bk * bk
            sum_ek2       += ek * ek

        # Analytical optimal inputs
        # Q_iq acts as extra regularisation on the vq denominator:
        #   it penalises large vq (hence large iq) without creating a
        #   competing numerator term that would fight the speed objective.
        #   This keeps the speed-tracking vq numerator (Q_omega · Σek·err)
        #   in full control while Q_iq simply reduces the gain magnitude.
        denom_d = self.Q_id    * sum_bk2 + self.R_vd
        denom_q = self.Q_omega * sum_ek2 + self.Q_iq * sum_bk2 + self.R_vq

        vd_mpc = self.Q_id    * sum_bk_err_d / denom_d if denom_d > 1e-30 else 0.0
        vq_mpc = self.Q_omega * sum_ek_err   / denom_q if denom_q > 1e-30 else 0.0

        # Add BEMF feedforward and clamp
        vd = self._clamp(vd_mpc + ed_hat, -self.V_MAX, self.V_MAX)
        vq = self._clamp(vq_mpc + eq_hat, -self.V_MAX, self.V_MAX)
        return vd, vq

    # ------------------------------------------------------------------ compute

    def compute_py(self, t: float, dt: float, input_values=None):
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
        # Direct speed measurement from FMU (element [5]) — bypasses the
        # finite-difference estimator which is noisy at startup and stalls
        # the controller during direction changes.
        omega_m_direct = float(u[5]) if len(u) > 5 else None

        # Electrical angle
        theta_e = self.P_POLES * theta_m
        # Use FMU speed directly when available; fall back to estimator otherwise.
        if omega_m_direct is not None:
            omega_m = omega_m_direct
            # Keep estimator state in sync so fallback is seamless
            self._last_theta_m = theta_m
            self._omega_filt   = omega_m_direct
        else:
            omega_m = self._get_speed(theta_m, dt)

        # Clarke transform: [ia,ib,ic] → [i_alpha, i_beta]
        clarke_out = self._clarke.compute_py(
            t, dt,
            [VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "in")])
        i_alpha = float(clarke_out.value[0])
        i_beta  = float(clarke_out.value[1])

        # SMO back-EMF estimation
        self._smo_step(i_alpha, i_beta,
                       self._v_alpha_prev, self._v_beta_prev, dt)

        theta_in = VectorSignal(np.array([theta_e], dtype=np.float32), "in")

        # Park transform: [i_alpha, i_beta] → [id, iq]
        park_out = self._park.compute_py(
            t, dt,
            [VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "in"),
             theta_in])
        id_meas = float(park_out.value[0])
        iq_meas = float(park_out.value[1])

        # Park transform SMO back-EMF into dq frame
        emf_out = self._park.compute_py(
            t, dt,
            [VectorSignal(np.array([self._e_alpha_filt, self._e_beta_filt],
                                    dtype=np.float32), "in"),
             theta_in])
        ed_hat_raw = float(emf_out.value[0])
        eq_hat_raw = float(emf_out.value[1])

        # Physical BEMF clamp: prevents SMO saturation locking up vd/vq.
        # At startup (omega_e≈0): forces ed=eq=0 (BEMF=0 is exact physics).
        # At speed: limits to ωe·λ_pm = max theoretical BEMF.
        omega_e   = float(self.P_POLES) * omega_m
        _bemf_max = abs(omega_e) * self.LAMBDA_PM
        ed_hat    = self._clamp(ed_hat_raw, -_bemf_max, _bemf_max)
        eq_hat    = self._clamp(eq_hat_raw, -_bemf_max, _bemf_max)

        # Soft-start: ramp iq limit from 0 → I_MAX over SOFTSTART_T seconds
        self._iq_limit = min(self.I_MAX,
                             self._iq_limit + self.I_MAX * dt / self._SOFTSTART_T)

        # Clamp measured currents before solver
        id0 = self._clamp(id_meas, -self.I_MAX, self.I_MAX)
        iq0 = self._clamp(iq_meas, -self.I_MAX, self.I_MAX)

        # 3-state MPC: drive id→0 and omega_m→omega_ref simultaneously
        x0 = MPCState(id0, iq0, omega_m)
        vd, vq = self._solve_mpc(x0, omega_ref_mech, ed_hat, eq_hat, dt)

        # Soft-start: limit vq while iq_limit is ramping up
        vq_lim = (self._iq_limit / self.I_MAX) * self.V_MAX
        vq = self._clamp(vq, -vq_lim, vq_lim)

        # Integral correction for steady-state speed error.
        # The finite-horizon MPC cannot fully eliminate speed offset under load
        # because the horizon (0.5ms) is too short to see the slow mechanical
        # dynamics. A vq integrator on speed error fixes this at zero cost to
        # the inner-loop performance — it only adds a small bias to vq.
        # KI_v = 0.03 V/(rad·s): corrects 53 RPM (5.6 rad/s) error in ~1.5s.
        _KI_v   = 0.03
        _sp_err = omega_ref_mech - omega_m
        # Only integrate once soft-start is complete — avoids wind-up while
        # vq is artificially clamped below the MPC's natural output.
        _softstart_done = (self._iq_limit >= self.I_MAX)
        if _softstart_done:
            self._speed_err_integral += _sp_err * dt
        # Anti-windup: keep integral contribution within remaining vq headroom
        _head   = self.V_MAX - abs(vq)
        _intmax = _head / (_KI_v + 1e-30)
        self._speed_err_integral = self._clamp(
            self._speed_err_integral, -_intmax, _intmax)
        vq = self._clamp(vq + _KI_v * self._speed_err_integral,
                         -self.V_MAX, self.V_MAX)

        # InvPark: [vd, vq] → [v_alpha, v_beta]
        inv_out = self._inv_park.compute_py(
            t, dt,
            [VectorSignal(np.array([vd, vq], dtype=np.float32), "in"),
             theta_in])
        v_alpha = float(inv_out.value[0])
        v_beta  = float(inv_out.value[1])

        # Store PHYSICAL volts for SMO (SI units — L and R are in SI).
        self._v_alpha_prev = v_alpha
        self._v_beta_prev  = v_beta

        # Normalise for SVPWMPackBlock: divide by V_DC/2 → range [-1, +1].
        # Matches SMC: y->v_alpha /= SMC_SVPWM_GAIN in embed_sim_smc_controller.c
        v_alpha_out = v_alpha / self.SVPWM_GAIN
        v_beta_out  = v_beta  / self.SVPWM_GAIN

        # Logging (every 1ms, first 500ms detailed)
        if t >= self._log_next:
            self._log_t.append(t)
            self._log_speed.append(omega_m * 60.0 / (2.0 * math.pi))
            self._log_speed_ref.append(omega_ref_mech * 60.0 / (2.0 * math.pi))
            self._log_iq_ref.append(iq_meas)   # log actual iq (no iq_ref)
            self._log_iq.append(iq_meas)
            self._log_id.append(id_meas)
            self._log_next += 0.001

            if t < 0.5 and (len(self._log_t) % 10 == 1):
                print(
                    f"[MPC t={t:.3f}s]  "
                    f"rpm={omega_m*60/(2*math.pi):+8.1f}  "
                    f"id={id_meas:+7.3f}A  iq={iq_meas:+7.3f}A  "
                    f"vd={vd:+6.3f}V  vq={vq:+6.3f}V  "
                    f"|Vab|={math.hypot(v_alpha, v_beta):.3f}V  "
                    f"theta_e={theta_e:.3f}rad"
                )

        self.output = VectorSignal(
            np.array([v_alpha_out, v_beta_out], dtype=np.float32), self.name)
        return self.output

    def compute_c(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)

    def reset(self) -> None:
        super().reset()
        self._i_alpha_hat  = 0.0
        self._i_beta_hat   = 0.0
        self._e_alpha_filt = 0.0
        self._e_beta_filt  = 0.0
        self._v_alpha_prev = 0.0
        self._v_beta_prev  = 0.0
        self._omega_filt   = 0.0
        self._last_theta_m = 0.0
        self._iq_limit          = 0.0
        self._speed_err_integral = 0.0
        self._log_t.clear()
        self._log_speed.clear()
        self._log_speed_ref.clear()
        self._log_iq_ref.clear()
        self._log_iq.clear()
        self._log_id.clear()
        self._log_next = 0.0
        self._clarke.reset()
        self._park.reset()
        self._inv_park.reset()

    @property
    def log_data(self) -> dict:
        return {
            "t":         np.array(self._log_t,         dtype=np.float32),
            "speed":     np.array(self._log_speed,     dtype=np.float32),
            "speed_ref": np.array(self._log_speed_ref, dtype=np.float32),
            "iq_ref":    np.array(self._log_iq_ref,    dtype=np.float32),
            "iq":        np.array(self._log_iq,        dtype=np.float32),
            "id":        np.array(self._log_id,        dtype=np.float32),
        }


__all__ = ["MPCControllerBlock", "_DB42S02"]
