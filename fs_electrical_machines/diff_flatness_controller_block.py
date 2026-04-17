"""
diff_flatness_controller_block.py
==================================
Differential Flatness FOC Controller block — NANOTEC DB42S02.

EmbedSim VectorBlock wrapper for the C implementation in:
    embed_sim_dfc_controller.c / embed_sim_dfc_controller.h

ARCHITECTURE
============
Three cascaded loops execute in sequence each simulation step,
matching DFC_Controller_Step() in the C implementation exactly:

    OUTER LOOP  — Speed [rad/s mech] -> iq_ref [A]
      iq_ref = Kp_speed * (omega_ref - omega_meas_mech)
      Feedback: SpeedFusion (encoder IIR + SMO blend)

    INNER LOOP D  — id [A] -> vd [V]
      vd = -omega_e * L_Q * iq_ref           [flatness decoupling]
         + Kp_id * (0 - id_meas)             [proportional MTPA]
         + id_integral                       [integral Fix 3: eliminates DC bias]
           id_integral += Ki_id * dt * (0 - id_meas)
           id_integral  = clamp(id_integral, +/-id_int_limit)
           frozen when vd saturated (anti-windup)
      Priority saturation: vd clamped first, vq gets sqrt(V_MAX^2 - vd^2).

    INNER LOOP Q  — iq [A] -> vq [V]
      vq = R_S*iq_ref + L_Q*diq/dt + omega_e*LAMBDA_PM   [flatness feedforward]
         + Kp_iq * (iq_ref - iq_meas)                    [residual correction]

SIGNAL BUS (input_values[0], 5 elements)
=========================================
    u[0]  omega_ref_mech  [rad/s]   Mechanical speed reference
    u[1]  theta_m         [rad]     Encoder mechanical angle (accumulating)
    u[2]  ia              [A]       Phase-A current
    u[3]  ib              [A]       Phase-B current
    u[4]  ic              [A]       Phase-C current

OUTPUT (2 elements)
====================
    y[0]  v_alpha   [V]   Alpha-axis voltage for SVPWM
    y[1]  v_beta    [V]   Beta-axis voltage for SVPWM

C ALIGNMENT
===========
Every class, method, and constant in this file mirrors the corresponding
C construct by name.  Inline ``# C:`` comments give the exact C line.
All default parameter values are numerically identical to the C #defines
in embed_sim_dfc_gains.h and embed_sim_dfc_controller.h.
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

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
_HERE  = Path(__file__).resolve().parent
_C_SRC = _HERE / "c_src"


# ===========================================================================
# SpeedFusion
# Mirrors: DFC_SpeedFusion_T  +  DFC_SpeedFusion_Update()
# ===========================================================================

class SpeedFusion:
    """
    Speed-dependent complementary filter.

    Blends encoder finite-difference speed with the SMO electrical speed
    into a single fused estimate.

    Signal equations (matching DFC_SpeedFusion_Update()):

        theta_e          [rad]      = P_POLES * theta_m
        omega_enc_filt   [rad/s]    = IIR( delta_theta / dt )
        alpha            [-]        = linear ramp in [omega_lo, omega_hi]
        omega_smo_gated  [rad/s]    = omega_smo_e  if |omega_smo_e - omega_enc_e| <= plaus_band
                                    = omega_enc_e  otherwise  (encoder fallback)
        omega_e          [rad/s]    = (1-alpha)*omega_enc_e + alpha*omega_smo_gated

    IIR coefficient adapts with speed:
        iir_coeff = iir_lo + alpha * (iir_hi - iir_lo)

    Encoder fallback when SMO not yet converged:
        if |omega_smo_e| < 1 rad/s  and  |omega_enc_filt| > omega_lo:
            omega_e = omega_enc_e   (override blend)

    Parameters
    ----------
    P_POLES : int
        Number of pole pairs [-].  C: DFC_P_POLES = 4.
    omega_lo : float
        Lower blend threshold [rad/s mech].  C: DFC_FUSION_OMEGA_LO = 50.0.
        Below this speed alpha = 0 (pure encoder).
    omega_hi : float
        Upper blend threshold [rad/s mech].  C: DFC_FUSION_OMEGA_HI = 250.0.
        Above this speed alpha = 1 (pure SMO, subject to plausibility gate).
    iir_lo : float
        Encoder IIR coefficient at low speed [-].  C: DFC_FUSION_IIR_LO = 0.05.
        Heavy smoothing; effective time constant ~ dt*(1-iir_lo)/iir_lo = 950 us.
    iir_hi : float
        Encoder IIR coefficient at high speed [-].  C: DFC_FUSION_IIR_HI = 0.30.
        Lighter smoothing; effective time constant ~ dt*(1-iir_hi)/iir_hi = 117 us.
    plaus_band : float
        SMO plausibility gate [rad/s elec].  C: DFC_SMO_PLAUS_BAND = 1000.0.
        If |omega_smo_e - omega_enc_e| > plaus_band the SMO output is
        replaced by the encoder electrical speed before the blend.
    """

    def __init__(
        self,
        P_POLES:    int   = 4,
        omega_lo:   float = 50.0,
        omega_hi:   float = 250.0,
        iir_lo:     float = 0.05,
        iir_hi:     float = 0.30,
        plaus_band: float = 1000.0,
    ) -> None:
        self.p          = float(P_POLES)   # [-]  pole pairs
        self.omega_lo   = omega_lo         # [rad/s mech]  C: DFC_FUSION_OMEGA_LO
        self.omega_hi   = omega_hi         # [rad/s mech]  C: DFC_FUSION_OMEGA_HI
        self.iir_lo     = iir_lo           # [-]           C: DFC_FUSION_IIR_LO
        self.iir_hi     = iir_hi           # [-]           C: DFC_FUSION_IIR_HI
        self.plaus_band = plaus_band       # [rad/s elec]  C: DFC_SMO_PLAUS_BAND

        # ---- Persistent state  (mirrors DFC_SpeedFusion_T) -----------------
        self._theta_m_prev:   float = 0.0  # [rad]       previous encoder angle
        self._omega_enc_filt: float = 0.0  # [rad/s]     IIR-filtered encoder speed
        self._omega_e_prev:   float = 0.0  # [rad/s elec] previous fused speed (for alpha)

        # ---- Diagnostic attributes (public, read-only by caller) -----------
        self.alpha:           float = 0.0  # [-]          current blend weight
        self.omega_enc:       float = 0.0  # [rad/s elec] encoder electrical speed
        self.omega_smo_gated: float = 0.0  # [rad/s elec] plausibility-gated SMO speed

    # -----------------------------------------------------------------------
    # _alpha  —  C: DFC_FusionAlpha()
    # -----------------------------------------------------------------------
    def _alpha(self, omega_abs: float) -> float:
        """
        Compute blend weight from mechanical speed magnitude.

        Returns alpha in [0.0, 1.0]: 0 = pure encoder, 1 = pure SMO.
        Piecewise-linear ramp between omega_lo and omega_hi.

        C counterpart: DFC_FusionAlpha() in embed_sim_dfc_controller.c.
        """
        if omega_abs <= self.omega_lo:
            return 0.0
        if omega_abs >= self.omega_hi:
            return 1.0
        return (omega_abs - self.omega_lo) / (self.omega_hi - self.omega_lo)

    # -----------------------------------------------------------------------
    # update  —  C: DFC_SpeedFusion_Update()
    # -----------------------------------------------------------------------
    def update(
        self,
        theta_m:     float,  # [rad]       encoder mechanical angle
        omega_smo_e: float,  # [rad/s elec] SMO output from SlidingModeObserver.step()
        dt:          float,  # [s]          step period
    ) -> Tuple[float, float]:
        """
        Execute one SpeedFusion step.

        Returns
        -------
        theta_e : float
            Electrical angle for Park/InvPark [rad].
            Derived from encoder: theta_e = P_POLES * theta_m.  No filtering.
        omega_e : float
            Fused electrical speed [rad/s elec] for _df_control() feedforward.

        Notes
        -----
        self._omega_enc_filt [rad/s mech] is also updated and read by
        compute_py() as omega_meas_mech for the speed P-loop, matching the
        third output pointer of DFC_SpeedFusion_Update() in C.

        C counterpart: DFC_SpeedFusion_Update() in embed_sim_dfc_controller.c.
        """
        # ---- 1. Electrical angle from encoder (exact, no filtering) --------
        # C: *theta_e = (MatrixFloat)DFC_P_POLES * theta_m;
        theta_e: float = self.p * theta_m

        # ---- 2. Encoder finite-difference mechanical speed ------------------
        # Unwrap delta to (-pi, +pi] to prevent 2*pi wrap-around spikes.
        # C: while (delta > DFC_PI_F) delta -= DFC_TWO_PI_F;
        delta: float = theta_m - self._theta_m_prev
        while delta >  math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi
        omega_raw: float = (delta / dt) if dt > 0.0 else 0.0  # [rad/s mech]

        # ---- 3. Adaptive IIR smoothing on encoder speed --------------------
        # alpha from *previous* fused speed — one-step lag breaks algebraic loop.
        # C: alpha     = DFC_FusionAlpha(fabsf(fusion->omega_e_prev));
        #    iir_coeff = DFC_FUSION_IIR_LO + alpha*(DFC_FUSION_IIR_HI - DFC_FUSION_IIR_LO);
        alpha:     float = self._alpha(abs(self._omega_e_prev))
        iir_coeff: float = self.iir_lo + alpha * (self.iir_hi - self.iir_lo)
        self._omega_enc_filt = (
            (1.0 - iir_coeff) * self._omega_enc_filt + iir_coeff * omega_raw
        )

        # ---- 4. SMO plausibility gate + convex blend -----------------------
        # C: omega_enc_e = (MatrixFloat)DFC_P_POLES * fusion->omega_enc_filt;
        omega_enc_e: float = self.p * self._omega_enc_filt  # [rad/s elec]

        # C: if (fabsf(omega_smo_e - omega_enc_e) > DFC_SMO_PLAUS_BAND)
        if abs(omega_smo_e - omega_enc_e) > self.plaus_band:
            omega_smo_gated: float = omega_enc_e   # encoder fallback
        else:
            omega_smo_gated = omega_smo_e           # SMO plausible

        # C: *omega_e = ((1-alpha)*omega_enc_e) + (alpha*omega_smo_gated);
        omega_e: float = (1.0 - alpha) * omega_enc_e + alpha * omega_smo_gated

        # ---- 5. Encoder fallback during SMO warmup -------------------------
        # C: if ((fabsf(omega_smo_e) < DFC_ONE_F) && (...)) *omega_e = omega_enc_e;
        if abs(omega_smo_e) < 1.0 and abs(self._omega_enc_filt) > self.omega_lo:
            omega_e = omega_enc_e

        # ---- 6. Update persistent state ------------------------------------
        self._theta_m_prev   = theta_m
        self._omega_e_prev   = omega_e
        self.alpha           = alpha
        self.omega_enc       = omega_enc_e
        self.omega_smo_gated = omega_smo_gated

        return theta_e, omega_e

    def reset(self) -> None:
        """Reset all state to zero.  C counterpart: init block in DFC_Controller_Init()."""
        self._theta_m_prev   = 0.0
        self._omega_enc_filt = 0.0
        self._omega_e_prev   = 0.0
        self.alpha           = 0.0
        self.omega_enc       = 0.0
        self.omega_smo_gated = 0.0


# ===========================================================================
# SlidingModeObserver
# Mirrors: DFC_SMO_T  +  DFC_SMO_Step()  +  DFC_SMOSwitch()
# ===========================================================================

class SlidingModeObserver:
    """
    Sliding Mode Observer — back-EMF estimation in the stationary αβ frame.

    Estimates electrical angle theta_e_hat [rad] and speed omega_e_filt
    [rad/s elec] from phase voltages and currents without a position sensor.

    Algorithm (matching DFC_SMO_Step()):

        Current observer (Forward Euler):
            i_hat[k+1] = i_hat[k] + dt/L * (v - R*i_hat - K*sat(i - i_hat))

        Back-EMF LPF (exponential IIR, corner ~800 Hz):
            e_hat[k+1] = e_hat[k] + alpha_lpf * (K*sat(err) - e_hat[k])

        Angle from back-EMF (SPMSM geometry):
            theta_e_hat = atan2(e_hat_alpha, -e_hat_beta)

        Speed from finite-difference:
            omega_e_hat  = unwrap(delta_theta) / dt   [clamped to omega_max]
            omega_e_filt += alpha_lpf * (omega_e_hat - omega_e_filt)

    Parameters
    ----------
    L_avg : float
        Average stator inductance [H].  C: (DFC_L_D + DFC_L_Q)*0.5 = 368e-6 H.
    R_S : float
        Stator resistance [Ohm].  C: DFC_R_S = 0.285.
    k_smo : float
        Switching gain [V].  C: DFC_SMO_K = 2.0.
        Must exceed |e_max| = omega_e_max * LAMBDA_PM = 920 * 0.0014 = 1.29 V.
    tau_e : float
        Back-EMF LPF time constant [s].  C: DFC_SMO_TAU_E = 0.0002.
        Corner frequency: 1/(2*pi*tau_e) = 796 Hz.
    i_max : float
        Maximum phase current [A].  C: DFC_I_MAX = 3.57.
        Divergence guard fires at |i_hat| > 2 * i_max.
    omega_max : float
        Speed spike clamp [rad/s elec].  C: DFC_SMO_OMEGA_MAX = 3000.0.
        Samples with |omega_e_hat| > omega_max are discarded; previous
        filtered value is held.
    warmup_steps : int
        Steps before speed output is enabled [-].
        C: DFC_SMO_WARMUP_STEPS = 400 (= 20 ms at 20 kHz).
    """

    def __init__(
        self,
        L_avg:        float = 0.3675e-3,  # [H]
        R_S:          float = 0.285,       # [Ohm]
        k_smo:        float = 2.0,         # [V]
        tau_e:        float = 0.0002,      # [s]
        i_max:        float = 3.57,        # [A]
        omega_max:    float = 3000.0,      # [rad/s elec]
        warmup_steps: int   = 400,         # [-]
    ) -> None:
        self.L_avg        = L_avg
        self.R_S          = R_S
        self.k_smo        = k_smo
        self.tau_e        = tau_e
        self.i_max        = i_max
        self.omega_max    = omega_max
        self.warmup_steps = warmup_steps

        # ---- Persistent state  (mirrors DFC_SMO_T) -------------------------
        self._i_hat_alpha:  float = 0.0  # [A]       estimated alpha current
        self._i_hat_beta:   float = 0.0  # [A]       estimated beta current
        self._e_hat_alpha:  float = 0.0  # [V]       LPF back-EMF alpha
        self._e_hat_beta:   float = 0.0  # [V]       LPF back-EMF beta
        self._theta_e_prev: float = 0.0  # [rad]     previous angle (finite-diff)
        self._warmup_cnt:   int   = 0    # [-]        steps since init

        # ---- Diagnostic attributes (public) --------------------------------
        self.theta_e_hat:   float = 0.0  # [rad]      estimated electrical angle
        self.omega_e_hat:   float = 0.0  # [rad/s]    raw speed before LPF
        self.omega_e_filt:  float = 0.0  # [rad/s]    LPF-smoothed speed (output)

    # -----------------------------------------------------------------------
    # _switch  —  C: DFC_SMOSwitch()
    # -----------------------------------------------------------------------
    def _switch(self, error: float) -> float:
        """
        Smooth sign approximation (linear saturation).

        Replaces pure sign() to eliminate chattering on the current observer.
        Boundary layer width = 0.01 A (~0.28 % of I_MAX).

        Returns value in [-1.0, +1.0].

        C counterpart: DFC_SMOSwitch() in embed_sim_dfc_controller.c.
        """
        width: float = 0.01           # [A]  boundary layer width
        arg:   float = error / width  # [-]  normalised error

        if arg >  5.0:
            return  1.0
        if arg < -5.0:
            return -1.0
        return arg * 0.2  # linear region: effective slope = 1/width, normalised by 5

    # -----------------------------------------------------------------------
    # step  —  C: DFC_SMO_Step()
    # -----------------------------------------------------------------------
    def step(
        self,
        v_alpha:  float,   # [V]   alpha-axis voltage (z-1, previous step)
        v_beta:   float,   # [V]   beta-axis voltage  (z-1, previous step)
        i_alpha:  float,   # [A]   measured alpha current from Clarke
        i_beta:   float,   # [A]   measured beta current from Clarke
        dt:       float,   # [s]   step period
    ) -> Tuple[float, float]:
        """
        Execute one SMO step.

        Parameters
        ----------
        v_alpha, v_beta : float
            Applied voltages from the *previous* ISR step [V].
            The z-1 delay matches C: ADC captures current while the previous
            PWM duty cycle is still active.
        i_alpha, i_beta : float
            Measured stationary-frame currents [A] from Clarke transform.
        dt : float
            Step period [s].

        Returns
        -------
        theta_e_hat : float
            Estimated electrical angle [rad].  Diagnostic; SpeedFusion uses
            the encoder theta_e for Park/InvPark, not this value.
        omega_e_filt : float
            LPF-filtered electrical speed [rad/s elec].
            Passed to SpeedFusion.update() as omega_smo_e.

        C counterpart: DFC_SMO_Step() in embed_sim_dfc_controller.c.
        """
        inv_L:     float = 1.0 / self.L_avg if self.L_avg > 1e-9 else 1e9  # [1/H]
        # Tustin LPF coefficient: alpha = dt / (tau + dt)
        # C: lpf_alpha = dt / (DFC_SMO_TAU_E + dt);
        lpf_alpha: float = (dt / (self.tau_e + dt)) if dt > 0.0 else 0.0

        # ---- Divergence guard ----------------------------------------------
        # Reinitialise i_hat if it has left the physical range.
        # _theta_e_prev preserved intentionally to avoid delta spike.
        # C: if (smo->i_hat_alpha > 2*DFC_I_MAX || ...)
        if (abs(self._i_hat_alpha) > 2.0 * self.i_max or
                abs(self._i_hat_beta) > 2.0 * self.i_max):
            self._i_hat_alpha = i_alpha
            self._i_hat_beta  = i_beta
            self._e_hat_alpha = 0.0
            self._e_hat_beta  = 0.0
            self.omega_e_hat  = 0.0
            self.omega_e_filt = 0.0
            # _theta_e_prev preserved intentionally

        # ---- Current estimation errors [A] ---------------------------------
        err_alpha: float = i_alpha - self._i_hat_alpha
        err_beta:  float = i_beta  - self._i_hat_beta

        # ---- Switching signals [V]: K * sat(error) -------------------------
        # C: sw_alpha = DFC_SMO_K * DFC_SMOSwitch(err_alpha);
        sw_alpha: float = self.k_smo * self._switch(err_alpha)
        sw_beta:  float = self.k_smo * self._switch(err_beta)

        # ---- Current observer — Forward Euler ------------------------------
        # di_hat/dt = (1/L) * (v - R*i_hat - sw)
        # C: smo->i_hat_alpha += dt * inv_L * (v_alpha - R_S*i_hat_alpha - sw_alpha);
        self._i_hat_alpha += dt * inv_L * (v_alpha - self.R_S * self._i_hat_alpha - sw_alpha)
        self._i_hat_beta  += dt * inv_L * (v_beta  - self.R_S * self._i_hat_beta  - sw_beta)

        # ---- Back-EMF LPF --------------------------------------------------
        # In sliding mode sw ≈ e_back; LPF extracts fundamental component.
        # C: smo->e_hat_alpha += lpf_alpha * (sw_alpha - smo->e_hat_alpha);
        self._e_hat_alpha += lpf_alpha * (sw_alpha - self._e_hat_alpha)
        self._e_hat_beta  += lpf_alpha * (sw_beta  - self._e_hat_beta)

        # ---- Electrical angle from back-EMF (SPMSM geometry) ---------------
        # e_alpha = +omega_e * lambda * sin(theta_e)
        # e_beta  = -omega_e * lambda * cos(theta_e)
        # => theta_e = atan2(e_alpha, -e_beta)   (positive for forward CW)
        # C: theta_e_new = atan2f(smo->e_hat_alpha, -smo->e_hat_beta);
        theta_e_new: float = math.atan2(self._e_hat_alpha, -self._e_hat_beta)

        # ---- Angle unwrap for finite-difference ----------------------------
        # C: while (delta > DFC_PI_F) delta -= DFC_TWO_PI_F;
        delta: float = theta_e_new - self._theta_e_prev
        while delta >  math.pi:
            delta -= 2.0 * math.pi
        while delta < -math.pi:
            delta += 2.0 * math.pi

        # ---- Speed from finite-difference (gated by warmup) ----------------
        # Fix 4: increment warmup counter BEFORE the gate test to match the C
        # execution order in DFC_Controller_Step(), which increments
        # smo_warmup_cnt before passing it to DFC_SMO_Step().
        # (Previously the increment was inside the step, giving a 1-step lag.)
        # C: s->smo_warmup_cnt++ [in DFC_Controller_Step];
        #    if ((dt > DFC_ZERO_F) && (warmup_cnt > DFC_SMO_WARMUP_STEPS))
        self._warmup_cnt += 1
        if dt > 0.0 and self._warmup_cnt > self.warmup_steps:
            self.omega_e_hat = delta / dt   # [rad/s elec]
            # Spike clamp: hold last filtered value if atan2 wrap artefact detected
            # C: if (smo->omega_e_hat > DFC_SMO_OMEGA_MAX || ...) hold;
            if abs(self.omega_e_hat) > self.omega_max:
                self.omega_e_hat = self.omega_e_filt
        else:
            # Warmup: suppress output so SpeedFusion encoder fallback applies
            self.omega_e_hat = 0.0

        # ---- LPF on speed estimate -----------------------------------------
        # Same lpf_alpha as back-EMF filter for consistency.
        # C: smo->omega_e_filt += lpf_alpha*(smo->omega_e_hat - smo->omega_e_filt);
        self.omega_e_filt += lpf_alpha * (self.omega_e_hat - self.omega_e_filt)

        # ---- Persist angle state -------------------------------------------
        self._theta_e_prev = theta_e_new
        self.theta_e_hat   = theta_e_new

        return self.theta_e_hat, self.omega_e_filt

    def reset(self) -> None:
        """Reset all state to zero.  C counterpart: init block in DFC_Controller_Init()."""
        self._i_hat_alpha  = 0.0
        self._i_hat_beta   = 0.0
        self._e_hat_alpha  = 0.0
        self._e_hat_beta   = 0.0
        self._theta_e_prev = 0.0
        self._warmup_cnt   = 0
        self.theta_e_hat   = 0.0
        self.omega_e_hat   = 0.0
        self.omega_e_filt  = 0.0


# ===========================================================================
# DFControllerBlock
# Mirrors: DFC_State_T  +  DFC_Controller_Step() / Init() / Reset()
# ===========================================================================

class DFControllerBlock(VectorBlock):
    """
    EmbedSim VectorBlock wrapping the Differential Flatness FOC Controller.

    Accepts the 5-element input bus produced by CtrlPacker (identical to the
    SMCControllerBlock interface) and emits a 2-element [v_alpha, v_beta]
    output in physical volts [V].

    Two execution backends are available:
        Python  — full numerical replication of the C implementation.
                  Use for simulation, gain tuning, and Bayesian optimisation.
        C       — calls the compiled Cython wrapper around the AURIX C code.
                  Use for hardware-in-the-loop verification.

    All default parameter values are numerically identical to the compile-time
    #define constants in embed_sim_dfc_gains.h and embed_sim_dfc_controller.h.
    """

    # ---- EmbedSim CodeGen interface ----------------------------------------
    NUM_INPUTS  = 1
    OUTPUT_SIZE = 2

    # Input bus layout — matches DFC_Input_T and SMC_Input_T field order
    INPUT_NAMES = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP  = [0, 1, 2, 3, 4]

    # C struct field comments emitted by CtrlPacker code generation
    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m":        "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia":             "Phase-A current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ib":             "Phase-B current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
        "ic":             "Phase-C current from ADC [A]; range [-DFC_I_MAX, +DFC_I_MAX]",
    }

    # ---- C code generation linkage -----------------------------------------
    step_func    = "DFC_Controller_Step"   # C: DFC_Controller_Step()
    state_struct = "DFC_State_T"           # C: DFC_State_T
    init_func    = "DFC_Controller_Init"   # C: DFC_Controller_Init()
    C_INIT_ARGS  = ["dt_s"]
    C_SOURCES    = ["embed_sim_dfc_controller.c"]
    C_HEADERS    = ["embed_sim_dfc_controller.h"]

    # Cython wrapper source
    PYX_FILE = str(_C_SRC / "dfc_controller_wrapper.pyx")

    # Custom C snippet emitted into embedsim_loop.c by the code generator.
    # DFC_Controller_Step() outputs physical voltages [V]; SVPWM expects
    # normalised references in [-1, +1] => divide by V_MAX = V_DC / sqrt(3).
    C_CUSTOM_EMIT = """\
        /* --- dfc_controller (DFControllerBlock) --- */
        /* DFC_Controller_Step() outputs physical voltages [V].                */
        /* The SVPWM block expects normalised [-1,+1] references.              */
        /* Divide by DFC_V_MAX = DFC_V_DC / sqrt(3) before SVPWM.             */
        DFC_Input_T   u_dfc;
        DFC_Output_T  y_dfc_out;
        real32_T      y_dfc[2];

        u_dfc.omega_ref_mech = in->omega_ref_mech;
        u_dfc.theta_m        = in->theta_m;
        u_dfc.ia             = in->ia;
        u_dfc.ib             = in->ib;
        u_dfc.ic             = in->ic;

        DFC_Controller_Step(&dfc_state, &u_dfc, dt, &y_dfc_out);

        /* Normalise: physical [V] -> SVPWM [-1, +1] */
        y_dfc[0] = y_dfc_out.v_alpha / DFC_V_MAX;
        y_dfc[1] = y_dfc_out.v_beta  / DFC_V_MAX;"""

    # ---- Class-level constants ---------------------------------------------
    _SQRT3: float = math.sqrt(3.0)  # C: inlined as 1.73205... in DFC_V_MAX definition

    # Observer mode selector (placeholder — only SMO mode currently implemented)
    OBS_MODE_SMO: int = 0

    # Diagnostic log density: every N steps.  Set DFC_DBG=1 for denser logging.
    DIAG_STEPS: int = 200 if os.environ.get("DFC_DBG") == "1" else 20

    # -----------------------------------------------------------------------
    # Constructor  —  C: DFC_Controller_Init()
    # -----------------------------------------------------------------------
    def __init__(
        self,
        name:              str   = "dfc",
        # ---- Motor parameters (C: DFC_MotorParams defgroup) ----------------
        P_POLES:           int   = 4,          # [-]          C: DFC_P_POLES
        R_S:               float = 0.285,       # [Ohm]        C: DFC_R_S
        L_D:               float = 0.0003675,   # [H]          C: DFC_L_D
        L_Q:               float = 0.0003675,   # [H]          C: DFC_L_Q
        LAMBDA_PM:         float = 0.0014,      # [Wb]         C: DFC_LAMBDA_PM
        V_DC:              float = 17.0,        # [V]          C: DFC_V_DC
        I_MAX:             float = 3.57,        # [A]          C: DFC_I_MAX
        dt_s:              float = 50e-6,       # [s]          nominal ISR period
        # ---- Gain constants (C: embed_sim_dfc_gains.h) ---------------------
        Kp_speed:          float = 0.4,         # [A/(rad/s)]  C: DFC_KP_SPEED
        Kp_id:             float = 0.4,         # [V/A]        C: DFC_KP_ID
        Kp_iq:             float = 8.0,         # [V/A]        C: DFC_KP_IQ
        diq_tau:           float = 0.001,       # [s]          C: DFC_DIQ_TAU
        # ---- SMO parameters (C: DFC_SMOParams defgroup) --------------------
        smo_k:             float = 2.0,         # [V]          C: DFC_SMO_K
        smo_tau:           float = 0.0002,      # [s]          C: DFC_SMO_TAU_E
        smo_omega_max:     float = 3000.0,      # [rad/s elec] C: DFC_SMO_OMEGA_MAX
        # ---- SpeedFusion parameters (C: DFC_FusionParams defgroup) ---------
        fusion_omega_lo:   float = 50.0,        # [rad/s mech] C: DFC_FUSION_OMEGA_LO
        fusion_omega_hi:   float = 250.0,       # [rad/s mech] C: DFC_FUSION_OMEGA_HI
        fusion_iir_lo:     float = 0.05,        # [-]          C: DFC_FUSION_IIR_LO
        fusion_iir_hi:     float = 0.30,        # [-]          C: DFC_FUSION_IIR_HI
        fusion_plaus_band: float = 1000.0,      # [rad/s elec] C: DFC_SMO_PLAUS_BAND
        # ---- Backend selection ---------------------------------------------
        use_c_backend:     bool  = False,
        dtype                    = None,
    ) -> None:
        super().__init__(name, use_c_backend=use_c_backend, dtype=dtype)

        # ---- Motor parameters ----------------------------------------------
        self.P_POLES   = P_POLES            # [-]
        self.R_S       = R_S                # [Ohm]
        self.L_D       = L_D                # [H]
        self.L_Q       = L_Q                # [H]
        self.LAMBDA_PM = LAMBDA_PM          # [Wb]
        self.V_DC      = V_DC               # [V]
        self.I_MAX     = I_MAX              # [A]
        self.V_MAX     = V_DC / self._SQRT3  # [V]  inscribed hexagon radius
        self.dt_s      = dt_s               # [s]

        # ---- Controller gains ----------------------------------------------
        self.Kp_speed  = Kp_speed           # [A/(rad/s)]
        self.Kp_id     = Kp_id              # [V/A]
        self.Kp_iq     = Kp_iq              # [V/A]
        self.Ki_id     = Kp_id * 0.30       # [V/(A*s)]  C: DFC_KI_ID = DFC_KP_ID*0.30  Ti~3.3s
        self.id_int_limit = 2.0             # [V]         C: DFC_ID_INT_LIMIT = 2.0
        self.diq_tau   = diq_tau            # [s]

        # ---- VectorBlock metadata ------------------------------------------
        self.vector_size  = 2
        self.output_label = "[v_alpha, v_beta]"
        self.is_dynamic   = False

        # ---- Coordinate transform sub-blocks (always Python) ---------------
        self._ct_clarke   = ClarkeTransformBlock("_dfc_clarke",    use_c_backend=False)
        self._ct_park     = ParkTransformBlock("_dfc_park",        use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_dfc_inv_park",  use_c_backend=False)

        # ---- SMO (mirrors DFC_SMO_T inside DFC_State_T) --------------------
        # C: L_avg = (DFC_L_D + DFC_L_Q) * 0.5f;
        L_avg: float = (L_D + L_Q) * 0.5   # [H]
        self._smo = SlidingModeObserver(
            L_avg        = L_avg,
            R_S          = R_S,
            k_smo        = smo_k,
            tau_e        = smo_tau,
            i_max        = I_MAX,
            omega_max    = smo_omega_max,
            warmup_steps = 400,             # C: DFC_SMO_WARMUP_STEPS
        )

        # ---- SpeedFusion (mirrors DFC_SpeedFusion_T inside DFC_State_T) ----
        self.fusion = SpeedFusion(
            P_POLES    = P_POLES,
            omega_lo   = fusion_omega_lo,
            omega_hi   = fusion_omega_hi,
            iir_lo     = fusion_iir_lo,
            iir_hi     = fusion_iir_hi,
            plaus_band = fusion_plaus_band,
        )

        # ---- Internal state (mirrors DFC_State_T scalar fields) ------------
        self._v_alpha_prev: float = 0.0   # [V]    C: s->v_alpha_prev
        self._v_beta_prev:  float = 0.0   # [V]    C: s->v_beta_prev
        self._iq_ref_prev:  float = 0.0   # [A]    C: s->iq_ref_prev
        self._diq_filt:     float = 0.0   # [A/s]  C: s->diq_filt
        self._id_integral:  float = 0.0   # [V]    C: s->id_integral (Fix 3)

        # ---- Diagnostic log (mirrors DFC_Controller_GetDiagnostics() keys) -
        self.log_data: dict = {
            "t":         [],   # [s]        simulation time
            "speed_ref": [],   # [RPM]      C: s->log_speed_ref
            "iq_ref":    [],   # [A]        C: s->log_iq_ref
            "id":        [],   # [A]        C: s->log_id
            "iq":        [],   # [A]        C: s->log_iq
            "alpha":     [],   # [-]        C: s->log_alpha
            "omega_e":   [],   # [rad/s]    C: s->log_omega_e  (encoder mech speed)
            "omega_smo": [],   # [rad/s]    C: s->log_omega_smo (SMO mech speed)
        }

        # ---- C backend wrapper ---------------------------------------------
        self._wrapper = None
        if use_c_backend:
            self._load_wrapper()

        # ---- Startup diagnostics -------------------------------------------
        print(f"[DFC] Differential Flatness Controller '{name}' initialised")
        print(f"[DFC]   Motor : R={R_S} Ohm, L={L_D*1e3:.4f} mH, "
              f"lambda_pm={LAMBDA_PM*1e3:.2f} mWb, p={P_POLES}")
        print(f"[DFC]   Gains : Kp_speed={Kp_speed:.4f} A/(rad/s), "
              f"Kp_id={Kp_id:.2f} V/A, Ki_id={self.Ki_id:.4f} V/(A*s), "
              f"Kp_iq={Kp_iq:.2f} V/A")
        print(f"[DFC]   SMO   : K={smo_k:.1f} V, tau={smo_tau*1e3:.1f} ms, "
              f"omega_max={smo_omega_max:.0f} rad/s elec")
        print(f"[DFC]   Fusion: omega_lo={fusion_omega_lo:.0f} rad/s, "
              f"omega_hi={fusion_omega_hi:.0f} rad/s, "
              f"plaus_band={fusion_plaus_band:.0f} rad/s elec")
        print(f"[DFC]   Backend: {'C (Cython)' if use_c_backend else 'Python'}")

    # -----------------------------------------------------------------------
    # _load_wrapper  —  C backend initialisation
    # -----------------------------------------------------------------------
    def _load_wrapper(self) -> None:
        """
        Load the Cython extension wrapping the C DFC controller.

        Raises
        ------
        ImportError
            If the .so / .pyd extension has not been built.
        RuntimeError
            If the wrapper object cannot be instantiated.
        """
        try:
            from dfc_controller_wrapper import DFCControllerWrapper
            self._wrapper = DFCControllerWrapper(
                self.V_DC, self.P_POLES,
                self.R_S, self.L_D, self.L_Q,
                self.LAMBDA_PM, self.I_MAX, self.dt_s,
                self.Kp_speed, self.Kp_id, self.Kp_iq,
            )
        except ImportError as exc:
            raise ImportError(
                "dfc_controller_wrapper not found. Build with:\n"
                "  cd fs_electrical_machines/c_src\n"
                "  python setup_dfc_controller.py build_ext --inplace"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"DFCControllerWrapper instantiation failed: {exc}"
            ) from exc

    # -----------------------------------------------------------------------
    # Transform helpers — always delegate to coordinate_transform_blocks
    # -----------------------------------------------------------------------

    def _clarke(self, ia: float, ib: float, ic: float) -> Tuple[float, float]:
        """
        Clarke abc -> αβ.  Returns (i_alpha [A], i_beta [A]).
        C counterpart: Clarke_Step() in embed_sim_coordinate_transform.c.
        """
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_clarke")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(
        self, i_alpha: float, i_beta: float, theta_e: float
    ) -> Tuple[float, float]:
        """
        Park αβ -> dq.  Returns (id_meas [A], iq_meas [A]).
        C counterpart: Park_Step() in embed_sim_coordinate_transform.c.
        """
        ab  = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_park")
        th  = VectorSignal(np.array([theta_e],         dtype=np.float32), "_park")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(
        self, vd: float, vq: float, theta_e: float
    ) -> Tuple[float, float]:
        """
        Inverse Park dq -> αβ.  Returns (v_alpha [V], v_beta [V]).
        C counterpart: InvPark_Step() in embed_sim_coordinate_transform.c.
        """
        dq  = VectorSignal(np.array([vd, vq],  dtype=np.float32), "_inv_park")
        th  = VectorSignal(np.array([theta_e], dtype=np.float32), "_inv_park")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    # -----------------------------------------------------------------------
    # _df_control  —  C: DFC_VoltageLaw()
    # -----------------------------------------------------------------------
    def _df_control(
        self,
        iq_ref:  float,   # [A]        q-axis current reference
        diq_dt:  float,   # [A/s]      LPF-filtered diq_ref/dt
        id_meas: float,   # [A]        measured d-axis current
        iq_meas: float,   # [A]        measured q-axis current
        omega_e: float,   # [rad/s elec] fused electrical speed
        _dt:     float,   # [s]          step period (for integrator update)
    ) -> Tuple[float, float]:
        """
        Compute dq-frame voltage references from the flatness voltage law.

        Equations (matching DFC_VoltageLaw() exactly):

            vd [V] = -omega_e * L_Q * iq_ref       [q->d cross-coupling cancel]
                   + Kp_id * (0 - id_meas)         [id=0 MTPA enforcement]

            vq [V] = R_S * iq_ref                  [resistive drop at iq_ref]
                   + L_Q * diq_dt                  [inductive drop for iq ramp]
                   + omega_e * LAMBDA_PM            [back-EMF cancellation]
                   + Kp_iq * (iq_ref - iq_meas)    [residual error correction]

        VOLTAGE SATURATION — priority-based (vd preserved, vq clipped):
        When ||[vd, vq]|| > V_MAX the inverter is voltage-limited.
        The d-axis (id correction) is given priority: vd is clamped to V_MAX
        first, then vq receives the remaining headroom:
            vq_max = sqrt(V_MAX^2 - vd^2)
        This ensures Kp_id always has full authority to enforce id = 0 (MTPA),
        preventing the load-dependent id bias that arises with proportional
        scaling when vq dominates the voltage vector.

        Returns
        -------
        vd, vq : float
            D- and q-axis voltage commands [V], magnitude-limited to V_MAX.

        C counterpart: DFC_VoltageLaw() in embed_sim_dfc_controller.c.
        """
        # ---- D-axis: decoupling + MTPA enforcement + integral (Fix 3) ------
        # id_ref = 0 A (MTPA for SPMSM with Ld = Lq).
        # iq_ref (not iq_meas) in the decoupling term — iq_meas injects ADC noise
        # through vd -> v_alpha_prev -> SMO -> SpeedFusion (tested, caused collapse).
        # The residual omega_e*Lq*(iq_meas - iq_ref) is a DC disturbance at steady
        # state; the id_integral integrator (Fix 3) is what eliminates it.
        # C: vd_out = -(omega_e*DFC_L_Q*iq_ref) + (DFC_KP_ID*id_error) + id_integral;
        id_error: float = 0.0 - id_meas   # [A]  id_ref = 0
        vd: float = (
            -omega_e * self.L_Q * iq_ref            # [V]  cross-coupling cancel
            + self.Kp_id * id_error                 # [V]  proportional MTPA
            + self._id_integral                     # [V]  integral Fix 3
        )
        vd_unsaturated: float = vd          # save before clamp for anti-windup

        # ---- Q-axis: flatness feedforward + residual feedback --------------
        # C: vq_out = (DFC_R_S*iq_ref) + (DFC_L_Q*diq_dt)
        #           + (omega_e*DFC_LAMBDA_PM) + (DFC_KP_IQ*(iq_ref-iq_meas));
        vq: float = (
              self.R_S    * iq_ref
            + self.L_Q    * diq_dt
            + omega_e     * self.LAMBDA_PM
            + self.Kp_iq  * (iq_ref - iq_meas)
        )

        # ---- Voltage saturation — priority: vd first, vq gets remainder ----
        # Fix 1: d-axis-priority saturation — Kp_id always has full authority.
        # Step 1: clamp vd to V_MAX.
        # Step 2: remaining headroom vq_max = sqrt(V_MAX^2 - vd^2).
        # Step 3: clamp vq to vq_max.
        # C counterpart: DFC_VoltageLaw() priority-clamp block.
        vd = max(-self.V_MAX, min(self.V_MAX, vd))    # [V]  Step 1
        vq_max: float = math.sqrt(
            max(0.0, self.V_MAX * self.V_MAX - vd * vd)
        )                                              # [V]  Step 2
        vq = max(-vq_max, min(vq_max, vq))             # [V]  Step 3

        # ---- d-axis integrator update with conditional anti-windup (Fix 3) --
        # Conditional integration: update only when vd was NOT saturated.
        # C: if (vd_out == vd_unsaturated) { id_integral += KI_ID*dt*id_error; }
        # dt is passed in as _dt from compute_py — use self.dt_s as fallback.
        _dt_int: float = _dt if _dt > 0.0 else self.dt_s
        if vd == vd_unsaturated:                        # not saturated
            self._id_integral += self.Ki_id * _dt_int * id_error
            self._id_integral = max(
                -self.id_int_limit,
                min(self.id_int_limit, self._id_integral)
            )
        else:
            pass   # saturated — freeze integrator (anti-windup)

        return vd, vq

    # -----------------------------------------------------------------------
    # _log_step  —  mirrors DFC_Controller_GetDiagnostics() snapshot fields
    # -----------------------------------------------------------------------
    def _log_step(
        self,
        t:              float,   # [s]
        omega_ref_mech: float,   # [rad/s mech]
        iq_ref:         float,   # [A]
        id_meas:        float,   # [A]
        iq_meas:        float,   # [A]
        omega_e_smo:    float,   # [rad/s elec] raw SMO electrical speed
    ) -> None:
        """
        Append one diagnostic snapshot to log_data.

        Mirrors the DFC_State_T log_* fields updated in DFC_Controller_Step()
        and the values returned by DFC_Controller_GetDiagnostics().

        Units match the C diagnostics API exactly:
            speed_ref  [RPM]        = omega_ref_mech * 60 / (2*pi)
            omega_e    [rad/s mech] = filtered encoder mechanical speed (P-loop)
            omega_smo  [rad/s mech] = omega_e_smo / P_POLES
        """
        self.log_data["t"].append(t)
        # C: s->log_speed_ref = u->omega_ref_mech * 60.0f / DFC_TWO_PI_F;
        self.log_data["speed_ref"].append(omega_ref_mech * 60.0 / (2.0 * math.pi))
        self.log_data["iq_ref"].append(iq_ref)
        self.log_data["id"].append(id_meas)
        self.log_data["iq"].append(iq_meas)
        self.log_data["alpha"].append(self.fusion.alpha)
        # C: s->log_omega_e = omega_meas_mech; (encoder mech speed driving P-loop)
        self.log_data["omega_e"].append(self.fusion._omega_enc_filt)
        # C: s->log_omega_smo = omega_smo_e / (MatrixFloat)DFC_P_POLES;
        self.log_data["omega_smo"].append(omega_e_smo / float(self.P_POLES))

    # -----------------------------------------------------------------------
    # compute_py  —  C: DFC_Controller_Step()  (Python backend)
    # -----------------------------------------------------------------------
    def compute_py(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Execute one DFC step — Python backend.

        Implements the identical execution sequence as DFC_Controller_Step().
        Step numbers in comments match the numbered list in that function's
        docstring in embed_sim_dfc_controller.c.

        Parameters
        ----------
        t : float
            Current simulation time [s].
        dt : float
            Actual step period [s].  Falls back to self.dt_s if zero.
        input_values : list of VectorSignal
            input_values[0].value = [omega_ref_mech, theta_m, ia, ib, ic].

        Returns
        -------
        VectorSignal
            value = [v_alpha [V], v_beta [V]].
        """
        zero = np.zeros(2, dtype=np.float32)

        # Guard: missing or undersized input bus
        if not input_values or len(input_values[0].value) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u   = input_values[0].value
        _dt = dt if dt > 0.0 else self.dt_s   # [s]

        # Unpack input bus — matches DFC_Input_T field order
        omega_ref_mech: float = float(u[0])   # [rad/s]
        theta_m:        float = float(u[1])   # [rad]
        ia:             float = float(u[2])   # [A]
        ib:             float = float(u[3])   # [A]
        ic:             float = float(u[4])   # [A]

        # ---- Step 2: Clarke — abc -> αβ ------------------------------------
        # C: Clarke_Step(&s->clarke_state, u->ia, u->ib, u->ic, &i_alpha, &i_beta);
        i_alpha, i_beta = self._clarke(ia, ib, ic)   # [A], [A]

        # ---- Step 3: SMO — always runs, feeds SpeedFusion ------------------
        # z-1 delay: ADC captures current while previous PWM duty is active.
        # C: DFC_SMO_Step(&s->smo, s->v_alpha_prev, s->v_beta_prev, ...)
        _, omega_e_smo = self._smo.step(
            self._v_alpha_prev, self._v_beta_prev,
            i_alpha, i_beta, _dt,
        )  # omega_e_smo [rad/s elec]

        # ---- Step 4: SpeedFusion — encoder + SMO -> theta_e, omega_e -------
        # C: DFC_SpeedFusion_Update(&s->fusion, u->theta_m, omega_smo_e, ...)
        theta_e, omega_e = self.fusion.update(theta_m, omega_e_smo, _dt)
        # theta_e  [rad]        for Park / InvPark
        # omega_e  [rad/s elec] for _df_control() feedforward

        # ---- Step 5: Speed measurement from SpeedFusion encoder path -------
        # C: omega_meas_mech set by DFC_SpeedFusion_Update() third output ptr
        omega_meas_mech: float = self.fusion._omega_enc_filt   # [rad/s mech]

        # ---- Step 5 (cont.): Speed P-loop -> iq_ref ------------------------
        # C: speed_err = u->omega_ref_mech - omega_meas_mech;
        #    iq_ref    = DFC_KP_SPEED * speed_err;
        #    iq_ref    = DFC_Clamp(iq_ref, DFC_I_MAX);
        speed_err: float = omega_ref_mech - omega_meas_mech    # [rad/s]
        iq_ref:    float = self.Kp_speed * speed_err           # [A]
        iq_ref           = max(-self.I_MAX, min(self.I_MAX, iq_ref))

        # ---- Step 6: Current derivative LPF --------------------------------
        # Tustin: lpf_alpha = dt / (tau + dt)
        # C: lpf_alpha = dt / (diq_tau + dt);
        #    diq_dt    = (iq_ref - s->iq_ref_prev) / dt;
        #    s->diq_filt = (1-alpha)*diq_filt + alpha*diq_dt;
        #    s->diq_filt = DFC_Clamp(s->diq_filt, DFC_I_MAX/DFC_DIQ_TAU);
        lpf_alpha: float = _dt / (self.diq_tau + _dt) if _dt > 0.0 else 0.0
        diq_raw:   float = ((iq_ref - self._iq_ref_prev) / _dt) if _dt > 0.0 else 0.0  # [A/s]
        self._diq_filt   = (1.0 - lpf_alpha) * self._diq_filt + lpf_alpha * diq_raw

        diq_clamp:  float = self.I_MAX / self.diq_tau                         # [A/s]
        diq_ref_dt: float = max(-diq_clamp, min(diq_clamp, self._diq_filt))   # [A/s]
        self._iq_ref_prev = iq_ref

        # ---- Step 7: Park — αβ -> dq ---------------------------------------
        # C: Park_Step(&s->park_state, i_alpha, i_beta, theta_e, &id_meas, &iq_meas);
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)   # [A], [A]

        # ---- Step 8: Flatness voltage law ----------------------------------
        # C: DFC_VoltageLaw(iq_ref, s->diq_filt, id_meas, iq_meas, omega_e, &vd, &vq);
        vd, vq = self._df_control(iq_ref, diq_ref_dt, id_meas, iq_meas, omega_e, _dt)  # [V], [V]

        # ---- Step 9: Inverse Park — dq -> αβ voltage -----------------------
        # C: InvPark_Step(&s->inv_park_state, vd, vq, theta_e, &y->v_alpha, &y->v_beta);
        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)   # [V], [V]

        # ---- Step 10: Latch voltages for next step's SMO (z-1) -------------
        # C: s->v_alpha_prev = y->v_alpha;  s->v_beta_prev = y->v_beta;
        self._v_alpha_prev = v_alpha
        self._v_beta_prev  = v_beta

        # ---- Step 11: Diagnostic log ---------------------------------------
        self._log_step(t, omega_ref_mech, iq_ref, id_meas, iq_meas, omega_e_smo)

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=np.float32), self.name
        )
        return self.output

    # -----------------------------------------------------------------------
    # compute_c  —  C backend via Cython wrapper
    # -----------------------------------------------------------------------
    def compute_c(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """
        Execute one DFC step — C backend (Cython wrapper).

        Calls the compiled DFC_Controller_Step() directly.  All state lives
        inside the C DFC_State_T struct managed by the wrapper; Python-side
        SMO/SpeedFusion state is not used.

        Returns
        -------
        VectorSignal
            value = [v_alpha [V], v_beta [V]].
        """
        zero = np.zeros(2, dtype=np.float32)

        if not input_values or not input_values[0]:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        u = input_values[0].value
        if len(u) < 5:
            self.output = VectorSignal(zero.copy(), self.name)
            return self.output

        # Pack input bus — DFC_Input_T field order
        inputs    = np.zeros(5, dtype=np.float32)
        inputs[0] = float(u[0])   # omega_ref_mech [rad/s]
        inputs[1] = float(u[1])   # theta_m        [rad]
        inputs[2] = float(u[2])   # ia             [A]
        inputs[3] = float(u[3])   # ib             [A]
        inputs[4] = float(u[4])   # ic             [A]

        self._wrapper.set_inputs(inputs)
        self._wrapper.compute(float(dt))
        outputs = self._wrapper.get_outputs()   # [v_alpha [V], v_beta [V]]

        self.output = VectorSignal(outputs, self.name)
        return self.output

    # -----------------------------------------------------------------------
    # compute  —  dispatcher
    # -----------------------------------------------------------------------
    def compute(
        self,
        t:            float,
        dt:           float,
        input_values: Optional[List[VectorSignal]] = None,
    ) -> VectorSignal:
        """Route to compute_c() or compute_py() based on use_c_backend."""
        if self.use_c_backend and self._wrapper is not None:
            return self.compute_c(t, dt, input_values)
        return self.compute_py(t, dt, input_values)

    # -----------------------------------------------------------------------
    # reset  —  C: DFC_Controller_Reset()
    # -----------------------------------------------------------------------
    def reset(self) -> None:
        """
        Reset all controller state to zero.

        Mirrors DFC_Controller_Reset() -> DFC_Controller_Init() in C:
        a single canonical path that zeroes everything, preventing residuals
        from diverging when Reset and Init are maintained separately.

        C counterpart: DFC_Controller_Reset() in embed_sim_dfc_controller.c.
        """
        super().reset()

        # Internal state (mirrors DFC_State_T scalar fields)
        self._v_alpha_prev = 0.0   # [V]
        self._v_beta_prev  = 0.0   # [V]
        self._iq_ref_prev  = 0.0   # [A]
        self._diq_filt     = 0.0   # [A/s]
        self._id_integral  = 0.0   # [V]   C: s->id_integral (Fix 3)

        # Coordinate transform sub-blocks
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()

        # SMO and SpeedFusion
        self._smo.reset()
        self.fusion.reset()

        # Diagnostic log
        self.log_data = {k: [] for k in self.log_data}

        # C backend wrapper
        if self._wrapper is not None:
            self._wrapper.reset()

    # -----------------------------------------------------------------------
    # get_diagnostics  —  C: DFC_Controller_GetDiagnostics()
    # -----------------------------------------------------------------------
    def get_diagnostics(self) -> dict:
        """
        Return the current diagnostic snapshot.

        Mirrors DFC_Controller_GetDiagnostics() in embed_sim_dfc_controller.c.
        All keys and units match the C log_* fields in DFC_State_T.

        Returns
        -------
        dict
            speed_ref_rpm : float   Speed reference [RPM]
            iq_ref        : float   Q-axis current reference [A]
            id_meas       : float   Measured d-axis current [A]
            iq_meas       : float   Measured q-axis current [A]
            fusion_alpha  : float   SpeedFusion blend weight [-]
            omega_e       : float   Filtered encoder mechanical speed [rad/s]
            omega_smo     : float   SMO mechanical speed estimate [rad/s]
        """
        return {
            "speed_ref_rpm": self.log_data["speed_ref"][-1] if self.log_data["speed_ref"] else 0.0,
            "iq_ref":        self.log_data["iq_ref"][-1]    if self.log_data["iq_ref"]    else 0.0,
            "id_meas":       self.log_data["id"][-1]        if self.log_data["id"]        else 0.0,
            "iq_meas":       self.log_data["iq"][-1]        if self.log_data["iq"]        else 0.0,
            "fusion_alpha":  self.fusion.alpha,                                            # [-]
            "omega_e":       self.fusion._omega_enc_filt,                                  # [rad/s mech]
            "omega_smo":     self.log_data["omega_smo"][-1] if self.log_data["omega_smo"] else 0.0,
        }

    # -----------------------------------------------------------------------
    # Diagnostic properties
    # -----------------------------------------------------------------------

    @property
    def smo_theta_e(self) -> float:
        """SMO estimated electrical angle [rad].  Diagnostic; not fed to Park."""
        return self._smo.theta_e_hat

    @property
    def smo_omega_e(self) -> float:
        """SMO LPF-filtered electrical speed [rad/s elec].  Diagnostic."""
        return self._smo.omega_e_filt

    @property
    def smo_omega_m(self) -> float:
        """SMO mechanical speed estimate [rad/s] = smo_omega_e / P_POLES."""
        return self._smo.omega_e_filt / float(self.P_POLES)

    @property
    def enc_omega_m(self) -> float:
        """Encoder IIR-filtered mechanical speed [rad/s].  Speed P-loop feedback signal."""
        return self.fusion._omega_enc_filt

    def __repr__(self) -> str:
        backend = "C" if (self.use_c_backend and self._wrapper) else "Python"
        return (
            f"DFControllerBlock('{self.name}', backend={backend}, "
            f"Kp_speed={self.Kp_speed} A/(rad/s), "
            f"Kp_id={self.Kp_id} V/A, Ki_id={self.Ki_id:.4f} V/(A*s), "
            f"Kp_iq={self.Kp_iq} V/A)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "DFControllerBlock",
    "SpeedFusion",
    "SlidingModeObserver",
]
