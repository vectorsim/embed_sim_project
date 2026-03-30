# smc_controller_block.py

"""
smc_controller_block.py
=======================
SMC FOC Controller — encoder position + SMO back-EMF filtering.

Signal sources (AURIX hardware):
  theta_m  : mechanical angle from encoder [rad], unwrapped — exact position
  ia,ib,ic : phase currents from ADC [A]   — subject to measurement noise

Architecture
------------
  theta_e = p·theta_m              exact electrical angle  → Park / InvPark
  omega_m = Δtheta_m/dt + IIR     encoder speed           → speed SMC only

  SMO (αβ frame, runs every step):
    Input : measured iα,iβ  +  applied v_α,v_β (previous step)
    Output: ê_α_filt, ê_β_filt  — noise-filtered back-EMF
    Purpose: replaces noisy omega_e·L·i cross-coupling terms in the
             current loop equivalent control with a clean back-EMF estimate.
    NOT used for theta_e or omega_m — the encoder provides those exactly.

  Speed SMC:
    Surface  : s = e + λ·∫e + γ·∫∫e
    Output   : iq_ref = KS_W·sat(s/φ_w) + ETA_W·s

  Current SMC (pure — no PI, no integrator):
    Equivalent control: exact analytical decoupling using encoder omega_e
      vd_eq = R·id − ωe·Lq·iq
      vq_eq = R·iq + ωe·(Ld·id + λpm)
    Switching:  vd = vd_eq + KS_I·sat(s_d/φ_i)
                vq = vq_eq + KS_I·sat(s_q/φ_i)

  SMO (αβ frame, runs every step — diagnostic / future sensorless):
    Estimates back-EMF from applied voltages + measured currents.
    NOT injected into the control loop — encoder gives exact theta_e/omega_m.
    Kept running so the filtered ê_αβ is available for logging/diagnostics.

All Clarke/Park/InvPark are delegated to coordinate_transform_blocks.py,
which mirrors embed_sim_coordinate_transform.c exactly.  No inline
transform math anywhere in this file — canonical single source of truth.
"""

import math
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

_HERE  = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from pyx_inspector import auto_populate_from_pyx
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)


# =============================================================================
# Motor constants — NANOTEC DB42S02
# =============================================================================

class _DB42S02:
    """NANOTEC DB42S02 motor parameters."""

    SMC_P_POLES    = 4
    SMC_R_S        = 0.19          # Ω
    SMC_L_D        = 0.125e-3      # H
    SMC_L_Q        = 0.125e-3      # H
    SMC_LAMBDA_PM  = 0.0014        # Wb
    SMC_J_ROTOR    = 2.4e-6        # kg·m²
    SMC_B_FRICTION = 1e-6          # N·m·s/rad
    SMC_I_MAX      = 3.57          # A
    SMC_V_DC       = 17.0          # V
    SMC_V_MAX      = SMC_V_DC / math.sqrt(3.0)     # V  (hexagon limit)
    SMC_KT         = 1.5 * SMC_P_POLES * SMC_LAMBDA_PM   # 0.0084 N·m/A

    # Sliding surface — first-order: s = e + λ·∫e
    # GAMMA_W = 0.0 disables double integral — was adding phase lag → instability
    SMC_WC_I     = 2.0 * math.pi * 800.0   # current loop bandwidth  [rad/s]
    SMC_LAMBDA_W = 2.0 * math.pi * 20.0    # surface slope [rad/s]
    SMC_GAMMA_W  = 0.0                      # double-integral disabled

    # SMO parameters
    # k must satisfy:  k > |e_back_EMF_max|
    # E_max = lambda_pm * p * omega_e_max = 0.0014 * 4 * 2π * 2000/60 = 1.17 V
    # Use 3× margin:  k = 3.0 * 1.17 = 3.5 V
    # NEVER use k = V_MAX — that injects ±9.8V into the observer at every step,
    # which at 20 kHz / L=125µH gives ΔI = 9.8×50e-6/125e-6 = 3.9 A per step
    # → observer diverges → id=42A → motor stalls.
    SMC_SMO_K  = 2.0                   # V  (1.7× back-EMF margin)
    SMC_SMO_FC = 500.0                 # Hz — diagnostic only; not in control loop

    # Current loop — discrete pole placement at z=0.5:
    #   KS_I = φ_i·L/(2·dt) = 0.5·125e-6/(2·50e-6) = 0.625 V
    #   Slew = KS_I·dt/L = 0.25 A/step < φ_i = 0.5 A  → no overshoot ✓
    #   BW = KS_I/(2π·L·φ_i) = 1592 Hz
    # KS_I sized for actual SVPWM chain gain.
    # SVPWMPackBlock passes Vref=v_ab directly (no normalisation).
    # SVPWM gain = V_DC/2 = 8.5 (confirmed by diagnostic data).
    # For boundary layer: KS_I < phi_i*L/(gain*dt) = 0.5*125e-6/(8.5*50e-6) = 0.147V
    # Use 50% margin: KS_I = 0.0735V -> delta_id = 0.25 A/step < phi_i = 0.5A
    # SVPWM chain gain: SVPWMPackBlock passes Vref=|v_ab| without normalisation.
    # Effective gain = V_DC/2 (confirmed by diagnostic: 0.625V in -> 5.3125V at plant).
    # All controller voltages must be divided by this gain so that after
    # SVPWM amplification the plant sees the intended physical voltages.
    SMC_SVPWM_GAIN = SMC_V_DC / 2.0   # = 8.5
    SMC_KS_I  =   0.058730
    SMC_PHI_I = 0.277341

    # Speed loop:
    #   KS_W ≥ T_load_max/KT = 0.020/0.0084 = 2.381 A → 3.095 A (+30%)
    #   PHI_W: sized so iq_start = I_MAX/3 during ramp
    #     PHI_W = KS_W·e_max/(I_MAX/3) = 3.095·209.4/1.19 = 545 rad/s
    #   LAMBDA_W = 2π×10 Hz = 62.83 rad/s (surface integral slope)
    #   ETA_W: small damping term, keep tiny
    SMC_KS_W  =  5.554994   # A
    SMC_PHI_W = 279.005762    # rad/s
    SMC_ETA_W = 0.001   # —
    SMC_LAMBDA_W = 2.0 * math.pi * 10.0   # 62.83 rad/s


# =============================================================================
# SMCControllerBlock
# =============================================================================

class SMCControllerBlock(VectorBlock):
    """
    Sliding Mode FOC Controller.

    Encoder provides exact theta_e and omega_m.
    SMO provides noise-filtered back-EMF (ê_α_filt, ê_β_filt) for the
    current loop equivalent control — replaces the noisy omega_e·L·i terms
    that would otherwise amplify encoder quantisation noise into the voltages.

    All coordinate transforms delegate to coordinate_transform_blocks.py
    (Clarke, Park, InvPark block instances cached in __init__).
    No inline transform math in this file.
    """

    # ── CodeGen metadata ─────────────────────────────────────────────────────
    PYX_FILE     = str(_C_SRC / "smc_controller_wrapper.pyx")
    C_SOURCES    = ["embed_sim_smc_controller.c"]
    C_HEADERS    = ["embed_sim_smc_controller.h"]
    state_struct = "SMC_Controller_T"
    step_func    = "SMC_Controller_Step"
    init_func    = "SMC_Controller_Init"
    C_INIT_ARGS  = ["dt_s"]

    C_CUSTOM_EMIT = """\
        /* --- smc_controller (SMCControllerBlock) --- */
        /* SMC_Controller_Step() outputs normalised v_alpha/v_beta (÷ V_DC/2)  */
        /* so the SVPWM block receives a reference in [-1,+1].                  */
        /* SMO feedback is stored from the physical InvPark output inside Step. */
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

    # ── Constructor ──────────────────────────────────────────────────────────

    def __init__(
            self,
            name: str             = "smc",
            SMC_V_DC: float       = _DB42S02.SMC_V_DC,
            SMC_P_POLES: int      = _DB42S02.SMC_P_POLES,
            SMC_R_S: float        = _DB42S02.SMC_R_S,
            SMC_L_D: float        = _DB42S02.SMC_L_D,
            SMC_L_Q: float        = _DB42S02.SMC_L_Q,
            SMC_LAMBDA_PM: float  = _DB42S02.SMC_LAMBDA_PM,
            SMC_J_ROTOR: float    = _DB42S02.SMC_J_ROTOR,
            SMC_B_FRICTION: float = _DB42S02.SMC_B_FRICTION,
            SMC_I_MAX: float      = _DB42S02.SMC_I_MAX,
            SMC_KS_W: float       = _DB42S02.SMC_KS_W,
            SMC_ETA_W: float      = _DB42S02.SMC_ETA_W,
            SMC_PHI_W: float      = _DB42S02.SMC_PHI_W,
            SMC_KS_I: float       = _DB42S02.SMC_KS_I,
            SMC_PHI_I: float      = _DB42S02.SMC_PHI_I,
            SMC_LAMBDA_W: float   = _DB42S02.SMC_LAMBDA_W,
            SMC_GAMMA_W: float    = _DB42S02.SMC_GAMMA_W,
            SMC_SMO_K: float      = _DB42S02.SMC_SMO_K,
            SMC_SMO_FC: float     = _DB42S02.SMC_SMO_FC,
            dt_s: float           = 50e-6,
            use_c_backend: bool   = False,
            use_smo_eq_ctrl: bool = False,
            integrator: str       = "tustin",
            dtype                 = None,
    ) -> None:

        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        _valid = ("tustin", "heun", "euler")
        if integrator not in _valid:
            raise ValueError(f"integrator must be one of {_valid}, got {integrator!r}")
        self._integrator: str = integrator
        self._use_smo_eq_ctrl: bool = bool(use_smo_eq_ctrl)

        # Motor parameters
        self.SMC_V_DC       = float(SMC_V_DC)
        self.SMC_P_POLES    = int(SMC_P_POLES)
        self.SMC_R_S        = float(SMC_R_S)
        self.SMC_L_D        = float(SMC_L_D)
        self.SMC_L_Q        = float(SMC_L_Q)
        self.SMC_LAMBDA_PM  = float(SMC_LAMBDA_PM)
        self.SMC_J_ROTOR    = float(SMC_J_ROTOR)
        self.SMC_B_FRICTION = float(SMC_B_FRICTION)
        self.SMC_I_MAX      = float(SMC_I_MAX)
        self.SMC_V_MAX      = self.SMC_V_DC / self._SQRT3

        # Gains
        self.SMC_KS_W  = float(SMC_KS_W)
        self.SMC_ETA_W = min(float(SMC_ETA_W), 0.01)   # hard cap
        self.SMC_PHI_W    = float(SMC_PHI_W)
        self.SMC_SVPWM_GAIN = float(SMC_V_DC) / 2.0   # V_DC/2 = 8.5
        self.SMC_KS_I     = float(SMC_KS_I)
        self.SMC_PHI_I    = float(SMC_PHI_I)
        self.SMC_LAMBDA_W = float(SMC_LAMBDA_W)
        self.SMC_GAMMA_W  = float(SMC_GAMMA_W)

        # SMO parameters
        self.SMC_SMO_K  = float(SMC_SMO_K)
        self.SMC_SMO_FC = float(SMC_SMO_FC)

        self._dt_s_float  = float(dt_s)
        self.dt_s         = "EMBEDSIM_DT"
        self.vector_size  = 2
        self.output_label = "[v_α,v_β]"
        self.is_dynamic   = False

        # Speed SMC integrator states
        self._int_spd: float      = 0.0
        self._int2_spd: float     = 0.0
        self._e_prev: float       = 0.0
        self._int_spd_prev: float = 0.0

        # Encoder speed estimator state (for speed SMC only)
        self._omega_filt: float             = 0.0
        self._last_theta_m: float           = 0.0
        self._last_theta_m_unwrapped: float = 0.0

        # SMO state
        self._i_alpha_hat: float  = 0.0
        self._i_beta_hat: float   = 0.0
        self._e_alpha_filt: float = 0.0
        self._e_beta_filt: float  = 0.0
        self._v_alpha_prev: float = 0.0
        self._v_beta_prev: float  = 0.0
        # LPF coefficient α = ωc·dt / (1 + ωc·dt)
        _wc_smo = 2.0 * math.pi * self.SMC_SMO_FC
        self._smo_lpf_alpha: float = (
            _wc_smo * self._dt_s_float / (1.0 + _wc_smo * self._dt_s_float))

        # ── Soft-start current limit ───────────────────────────────────────────
        # Ramps iq_ref limit from 0 → I_MAX over _SOFTSTART_T seconds.
        # Prevents the first-step voltage spike caused by motor_delay zero-fallback
        # (at t=0 ia=ib=ic=0 for one step → plant integrator overshoot → id spike).
        # 50 ms gives ~1000 steps at 20 kHz — enough to absorb the startup transient.
        # On real AURIX hardware the ADC and encoder are valid from the first ISR
        # tick, so this limit engages naturally and ramps away before the motor
        # builds significant speed.
        self._SOFTSTART_T: float    = 0.05   # [s]  ramp duration
        self._iq_limit: float       = 0.0    # [A]  current soft limit (rises with time)

        # Diagnostics
        self._last_iq_ref: float = 0.0
        self._log_t: list        = []
        self._log_spd: list      = []
        self._log_sref: list     = []
        self._log_iqr: list      = []
        self._log_iq: list       = []
        self._log_id: list       = []
        self._log_next: float    = 0.0
        self._diag_count: int    = 0

        # C backend
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

        # Transform block instances — canonical, no inline math
        # Mirrors Clarke_Step / Park_Step / InvPark_Step in
        # embed_sim_coordinate_transform.c — single source of truth.
        self._ct_clarke   = ClarkeTransformBlock("_smc_clarke",    use_c_backend=False)
        self._ct_park     = ParkTransformBlock("_smc_park",        use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_smc_inv_park", use_c_backend=False)

        print(f"[SMC] Encoder: theta_e=p·theta_m (exact)  omega_m=diff+IIR")
        print(f"[SMC] SMO back-EMF filter: k={self.SMC_SMO_K:.3f} V  "
              f"fc={self.SMC_SMO_FC:.0f} Hz  alpha={self._smo_lpf_alpha:.5f}")
        print(f"[SMC] Speed gains: KS_W={self.SMC_KS_W:.4f} A  "
              f"PHI_W={self.SMC_PHI_W:.2f} rad/s  ETA_W={self.SMC_ETA_W:.4f}")
        print(f"[SMC] Current gains (pure SMC): KS_I={self.SMC_KS_I:.4f} V  "
              f"PHI_I={self.SMC_PHI_I:.3f} A")
        print(f"[SMC] Transforms delegated to coordinate_transform_blocks.py")

    # ── C backend loader ──────────────────────────────────────────────────────

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

    # ── Transform helpers — always delegate to coordinate_transform_blocks ────

    def _clarke(self, ia: float, ib: float, ic: float) -> tuple:
        """Clarke abc→αβ via ClarkeTransformBlock.compute_py()."""
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_clarke")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(self, i_alpha: float, i_beta: float, theta_e: float) -> tuple:
        """Park αβ→dq via ParkTransformBlock.compute_py()."""
        ab  = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_park")
        th  = VectorSignal(np.array([theta_e],         dtype=np.float32), "_park")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(self, vd: float, vq: float, theta_e: float) -> tuple:
        """Inverse Park dq→αβ via InvParkTransformBlock.compute_py()."""
        dq  = VectorSignal(np.array([vd, vq],   dtype=np.float32), "_inv_park")
        th  = VectorSignal(np.array([theta_e],  dtype=np.float32), "_inv_park")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # ── Static helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _sat(x: float, phi: float) -> float:
        """Boundary-layer saturation → [-1, +1]."""
        if phi <= 0.0:
            return math.copysign(1.0, x) if x != 0.0 else 0.0
        return max(-1.0, min(1.0, x / phi))

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    # ── Encoder speed estimator ───────────────────────────────────────────────

    def _get_speed_from_encoder(self, theta_m: float, dt: float) -> float:
        """
        Finite-difference speed estimator with 2π unwrapping + IIR.

        Used ONLY for speed SMC feedback.
        Park/InvPark use theta_e = p·theta_m directly — not this estimate.

        2π unwrap: every electrical wrap (2π/p ≈ 0.79 rad at p=4) would
        cause a large negative spike in the raw diff without unwrapping.
        IIR α=0.3 → τ ≈ 3 steps (150 µs at 50 µs dt).
        """
        if dt > 0.0:
            delta = theta_m - self._last_theta_m
            delta -= 2.0 * math.pi * math.floor(
                (delta + math.pi) / (2.0 * math.pi))
            self._last_theta_m_unwrapped += delta
            omega_raw        = delta / dt
            self._omega_filt = 0.7 * self._omega_filt + 0.3 * omega_raw
        self._last_theta_m = theta_m
        return self._omega_filt

    # ── Sliding Mode Observer ─────────────────────────────────────────────────

    def _smo_step(self, i_alpha: float, i_beta: float,
                  v_alpha: float, v_beta: float, dt: float) -> None:
        """
        One step of the Sliding Mode Observer — αβ frame.

        Purpose
        -------
        Estimate noise-filtered back-EMF (ê_α_filt, ê_β_filt) from noisy ADC
        currents and the known applied voltage from the previous step.
        Replaces the noisy omega_e·L·i cross-coupling terms in the current
        loop equivalent control, preventing ADC/encoder noise amplification.

        This observer does NOT estimate theta_e or omega_m.
        The encoder provides those exactly.

        Observer equations (forward Euler):
            dî_α/dt = (1/L)·(vα - R·î_α - ê_α_sw)
            dî_β/dt = (1/L)·(vβ - R·î_β - ê_β_sw)

        Switching injection:
            ê_sw = k·sign(i - î)    (measured minus estimated — pushes î toward i)

        Back-EMF LPF (fc = SMC_SMO_FC, default 500 Hz):
            ê_filt += α·(ê_sw - ê_filt)

        Outputs stored in self._e_alpha_filt, self._e_beta_filt.
        Used in _current_smc via _park(ê_α_filt, ê_β_filt, theta_e).

        Startup behaviour
        -----------------
        At t=0: _v_alpha_prev = _v_beta_prev = 0, î = 0, ê_filt = 0.
        → ed_hat = eq_hat = 0 → current loop is pure switching only.
        This is correct and stable — no bootstrap needed.
        As back-EMF builds with speed the equivalent control engages smoothly.
        """
        if dt <= 0.0:
            return

        inv_L = 1.0 / self.SMC_L_D
        k     = self.SMC_SMO_K
        alpha = self._smo_lpf_alpha

        # Current estimation errors — sign must be (measured - estimated)
        # so the switching term pushes î toward i, not away from it.
        err_alpha = i_alpha - self._i_alpha_hat
        err_beta  = i_beta  - self._i_beta_hat

        sw_alpha = k * math.tanh(err_alpha / 0.01)
        sw_beta  = k * math.tanh(err_beta  / 0.01)

        self._i_alpha_hat += dt * inv_L * (
            v_alpha - self.SMC_R_S * self._i_alpha_hat - sw_alpha)
        self._i_beta_hat  += dt * inv_L * (
            v_beta  - self.SMC_R_S * self._i_beta_hat  - sw_beta)

        self._e_alpha_filt += alpha * (sw_alpha - self._e_alpha_filt)
        self._e_beta_filt  += alpha * (sw_beta  - self._e_beta_filt)

    # ── Speed SMC ─────────────────────────────────────────────────────────────

    def _speed_smc(self, omega_ref: float, omega_m: float, dt: float,
                   iq_limit: float = None) -> float:
        """
        First-order integral sliding surface speed controller.

        Surface:  s = e + λ·∫e
        Control:  iq_ref = KS_W·sat(s/φ_w) + ETA_W·s
        Clamped to ±iq_limit (or ±I_MAX if not supplied).

        Anti-windup: integrator frozen when output is saturated.
        """
        e   = omega_ref - omega_m
        lim = iq_limit if iq_limit is not None else self.SMC_I_MAX

        # Compute unsaturated output with current integrator state
        s_now    = e + self.SMC_LAMBDA_W * self._int_spd
        iq_unsat = (self.SMC_KS_W * self._sat(s_now, self.SMC_PHI_W)
                    + self.SMC_ETA_W * s_now)

        # Anti-windup: only integrate when not saturated
        if abs(iq_unsat) < lim:
            int_limit = 10.0   # rad
            if self._integrator == "tustin":
                new_int = self._int_spd + 0.5 * dt * (e + self._e_prev)
                self._int_spd = self._clamp(new_int, -int_limit, int_limit)
            elif self._integrator == "heun":
                new_int = self._int_spd + 0.5 * dt * (self._e_prev + e)
                self._int_spd = self._clamp(new_int, -int_limit, int_limit)
            else:
                self._int_spd = self._clamp(
                    self._int_spd + dt * e, -int_limit, int_limit)
        self._e_prev = e

        s_spd  = e + self.SMC_LAMBDA_W * self._int_spd
        iq_ref = (self.SMC_KS_W * self._sat(s_spd, self.SMC_PHI_W)
                  + self.SMC_ETA_W * s_spd)
        return self._clamp(iq_ref, -lim, lim)

    # ── Current SMC (pure) ────────────────────────────────────────────────────

    def _current_smc(self, id_meas: float, iq_meas: float,
                     id_ref: float, iq_ref: float,
                     ed_hat: float, eq_hat: float,
                     phi_i_override: float = None) -> tuple:
        """
        SMC current controller with encoder-based equivalent control.

        Equivalent control (vd_eq, vq_eq) computed from encoder omega_e —
        exact decoupling of cross-coupling terms at all speeds.
        Caller passes pre-computed (ed_hat, eq_hat).

        Switching:
            vd = ed_hat + KS_I·sat(s_d/φ_i)
            vq = eq_hat + KS_I·sat(s_q/φ_i)

        Voltage vector clamped to V_MAX (hexagon limit).
        """
        phi_i = phi_i_override if phi_i_override is not None else self.SMC_PHI_I

        s_d = id_ref - id_meas
        s_q = iq_ref - iq_meas

        vd = ed_hat + self.SMC_KS_I * self._sat(s_d, phi_i)
        vq = eq_hat + self.SMC_KS_I * self._sat(s_q, phi_i)

        magnitude = math.sqrt(vd * vd + vq * vq)
        if magnitude > self.SMC_V_MAX:
            scale = self.SMC_V_MAX / magnitude
            vd   *= scale
            vq   *= scale

        return vd, vq

    # ── compute_py ───────────────────────────────────────────────────────────

    def compute_py(self, t: float, dt: float,
                   input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
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

        # Exact electrical angle from encoder
        theta_e = float(self.SMC_P_POLES) * theta_m

        # Encoder speed — for speed SMC only
        omega_m_est = self._get_speed_from_encoder(theta_m, dt)

        # Clarke: abc → αβ  (canonical)
        i_alpha, i_beta = self._clarke(ia, ib, ic)

        # SMO: update filtered back-EMF estimate (diagnostic — not injected).
        # With an encoder, Park(ê_αβ) ≈ [0, ωe·λpm] in steady state — that BEMF
        # is already inside vq_eq below.  Adding it again would double-count it
        # (≈1.17 V at 2000 RPM = 1.88×KS_I) → speed runaway to ~7000 RPM.
        # SMO runs so ê_αβ is available for diagnostics / sensorless extension.
        self._smo_step(i_alpha, i_beta,
                       self._v_alpha_prev, self._v_beta_prev, dt)

        # Park: αβ → dq  (canonical, exact encoder theta_e)
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        # Soft-start limit ramps 0 → I_MAX over _SOFTSTART_T.
        self._iq_limit = min(
            self.SMC_I_MAX,
            self._iq_limit + self.SMC_I_MAX * dt / self._SOFTSTART_T)

        # Speed SMC → iq_ref
        iq_ref = self._speed_smc(omega_ref_mech, omega_m_est, dt,
                                 iq_limit=self._iq_limit)

        self._last_iq_ref = iq_ref

        # Equivalent control (Utkin 1992 / Krishnan Ch.4)
        # ds_d/dt=0 -> vd_eq = R*id_meas - we*Lq*iq_meas
        # ds_q/dt=0 -> vq_eq = R*iq_meas + we*Ld*id_meas + we*lpm
        omega_e = float(self.SMC_P_POLES) * omega_m_est

        # vd_eq: on the surface s_d=0 means id=0, so R*id_ref=0.
        # Only the q->d cross-coupling term remains. R*id_meas must NOT
        # be used — it creates positive feedback: id large -> vd_eq large
        # -> plant integrates more id -> id grows to 34A.
        # vq_eq: R*iq_meas and omega_e*Ld*id_meas are both bounded and
        # correct — they do not create positive feedback on the q-axis.
        # Equivalent control (Utkin/Krishnan). On the MTPA surface id_ref=0,
        # so omega_e*Ld*id_ref=0 and the cross-coupling in vq_eq vanishes.
        # Using id_meas instead creates positive feedback:
        # id_meas large -> vq_eq large -> iq and omega_e large -> id_meas larger.
        # vd_eq keeps R*id_meas for natural d-axis damping.
        # Equivalent control: textbook Utkin/Krishnan equations,
        # divided by SVPWM_GAIN so that after the SVPWM chain (gain=V_DC/2=8.5)
        # the plant sees the correct physical feedforward voltages.
        # vd_eq_physical = R*id_meas - we*Lq*iq_meas
        # vq_eq_physical = R*iq_meas + we*(Ld*id_meas + lpm)
        # Controller outputs: vd_eq = vd_eq_physical / SVPWM_GAIN
        G = self.SMC_SVPWM_GAIN
        if self._use_smo_eq_ctrl:
            ed_hat, eq_hat = self._park(self._e_alpha_filt, self._e_beta_filt,
                                        theta_e)
            vd_eq = (self.SMC_R_S * id_meas - omega_e * self.SMC_L_Q * iq_meas) / G
            vq_eq = (self.SMC_R_S * iq_meas + omega_e * self.SMC_L_D * id_meas
                     + eq_hat) / G
        else:
            vd_eq = (self.SMC_R_S * id_meas - omega_e * self.SMC_L_Q * iq_meas) / G
            vq_eq = (self.SMC_R_S * iq_meas + omega_e * (self.SMC_L_D * id_meas
                                                           + self.SMC_LAMBDA_PM)) / G

        vd, vq = self._current_smc(id_meas, iq_meas, 0.0, iq_ref,
                                   vd_eq, vq_eq)

        # Inverse Park: dq → αβ  (canonical)
        # vd, vq are already divided by SVPWM_GAIN above, so v_alpha/v_beta
        # here are normalised [-1, +1].  The SMO observer uses L and R in SI
        # units — it must receive the physical voltage actually applied.
        # Physical = normalised × G = v_alpha_norm × (V_DC/2).
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)

        # Store PHYSICAL voltage for SMO next step (undo normalisation).
        self._v_alpha_prev = v_alpha * G
        self._v_beta_prev  = v_beta  * G

        # Logging at 1 kHz
        if t >= self._log_next:
            self._log_t.append(t)
            self._log_spd.append(omega_m_est * 60.0 / (2.0 * math.pi))
            self._log_sref.append(omega_ref_mech * 60.0 / (2.0 * math.pi))
            self._log_iqr.append(iq_ref)
            self._log_iq.append(iq_meas)
            self._log_id.append(id_meas)
            self._log_next += 0.001

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=np.float32), self.name)
        return self.output

    # ── compute_c ─────────────────────────────────────────────────────────────

    def compute_c(self, t: float, dt: float,
                  input_values: Optional[List[VectorSignal]] = None) -> VectorSignal:
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

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        super().reset()
        # Speed SMC integrators
        self._int_spd      = 0.0
        self._int2_spd     = 0.0
        self._e_prev       = 0.0
        self._int_spd_prev = 0.0
        # Encoder speed estimator
        self._omega_filt             = 0.0
        self._last_theta_m           = 0.0
        self._last_theta_m_unwrapped = 0.0
        # SMO
        self._i_alpha_hat  = 0.0
        self._i_beta_hat   = 0.0
        self._e_alpha_filt = 0.0
        self._e_beta_filt  = 0.0
        self._v_alpha_prev = 0.0
        self._v_beta_prev  = 0.0
        # Soft-start
        self._iq_limit     = 0.0
        # Diagnostics
        self._last_iq_ref = 0.0
        self._log_t.clear()
        self._log_spd.clear()
        self._log_sref.clear()
        self._log_iqr.clear()
        self._log_iq.clear()
        self._log_id.clear()
        self._log_next   = 0.0
        self._diag_count = 0
        # Transform blocks
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()
        if self._wrapper is not None:
            self._wrapper.reset()

    # ── log_data ──────────────────────────────────────────────────────────────

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