"""
maschine_connect.py — Fachschale Elektrische Maschinen: Sensor Interface Library
==================================================================================
Provides :class:`MaschineConnect`, a reusable VectorBlock that replaces the
inline sensor-packing logic in CtrlPacker.  It owns:

  * Speed reference ramp            (rate-limited domega/dt)
  * GPT12 incremental-encoder model (turn-rollover glitch, configurable)
  * ADC phase-current noise model   (Gaussian white noise + OU offset drift)
  * Output bus assembly             -> SMC_Input_T / EmbedSim_Input_T

Noise model parameters (all SI unless stated)
----------------------------------------------
  ADC_NOISE_STD_A    : float  -- 1-sigma white noise on each phase [A]
  ADC_OFFSET_DRIFT_A : float  -- peak DC offset drift (OU random walk) [A]
  ADC_OFFSET_TAU_S   : float  -- drift correlation time [s]  (approx. thermal tau)
  ENC_GLITCH_ENABLE  : bool   -- enable encoder turn-rollover glitch injection
  ENC_GLITCH_PROB    : float  -- probability of glitch at each turn boundary

Noise budget (default, 12-bit ADC @ 17 V rail, +-20 A range)
--------------------------------------------------------------
  LSB   approx. 9.8 mA
  sigma approx. 5 mA    (approx. 0.5 LSB quantisation + thermal)
  Drift approx. 5 mA peak over 500 ms thermal correlation time

Usage
-----
    from maschine_connect import MaschineConnect, MaschineConnectCfg

    cfg = MaschineConnectCfg(
        target_rads_mech   = TARGET_RADS_MECH,
        ramp_time_s        = _RAMP_TIME,
        adc_noise_std_a    = 0.005,
        adc_offset_drift_a = 0.005,
        adc_offset_tau_s   = 0.5,
        enc_glitch_enable  = True,
        enc_glitch_prob    = 0.03,
    )
    packer = MaschineConnect(name="maschine_connect", cfg=cfg)

Input ports  (same contract as CtrlPacker)
------------------------------------------
  [0]  motor feedback bus  [rpm, ia, ib, ic, theta_m, T_em, id, iq]  8 elements
  [1]  speed reference     [rad/s]  scalar

Output bus  (5 elements -- SMC_Input_T)
----------------------------------------
  [0]  omega_ref_mech  [rad/s]   ramp-filtered speed reference
  [1]  theta_m         [rad]     mechanical angle (accumulating, encoder model)
  [2]  ia              [A]       phase-A current  (ADC noise applied)
  [3]  ib              [A]       phase-B current  (ADC noise applied)
  [4]  ic              [A]       phase-C current  (ADC noise applied)
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from _path_utils import get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_embedsim_import_path()
sys.path.insert(0, _ROOT)

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# ---------------------------------------------------------------------------
# Motor feedback bus index map  (matches MotorModel output convention)
# ---------------------------------------------------------------------------
_IDX_RPM   = 0
_IDX_IA    = 1
_IDX_IB    = 2
_IDX_IC    = 3
_IDX_THETA = 4
_IDX_TEM   = 5
_IDX_ID    = 6
_IDX_IQ    = 7
_MOTOR_OUT_SIZE = 8

_TWO_PI = 2.0 * math.pi

# ---------------------------------------------------------------------------
# Scalar defaults
# ---------------------------------------------------------------------------
_DEFAULT_ADC_NOISE_STD   = 0.005   # [A]  approx. 5 mA sigma
_DEFAULT_ADC_DRIFT_PEAK  = 0.005   # [A]  5 mA peak OU amplitude
_DEFAULT_ADC_DRIFT_TAU   = 0.5     # [s]  500 ms thermal correlation time
_DEFAULT_ENC_GLITCH_PROB = 0.03    # probability per turn boundary


# ===========================================================================
# Configuration dataclass
# ===========================================================================

@dataclass
class MaschineConnectCfg:
    """
    Runtime-configurable parameters for MaschineConnect.

    Parameters
    ----------
    target_rads_mech : float
        Nominal target mechanical speed [rad/s].
    ramp_time_s : float
        Time to ramp from 0 to target_rads_mech [s].
        Ramp rate = target_rads_mech / ramp_time_s  [rad/s**2].
    adc_noise_std_a : float
        1-sigma white Gaussian noise per phase [A].  0.0 = disabled.
    adc_offset_drift_a : float
        Steady-state sigma of OU offset drift per phase [A].  0.0 = disabled.
    adc_offset_tau_s : float
        OU drift correlation time [s].
    enc_glitch_enable : bool
        Inject GPT12 turn-rollover glitches.
    enc_glitch_prob : float
        Per-boundary glitch probability [0, 1].
    seed : int
        Base RNG seed.  Each channel gets seed+N for independence.
    """

    target_rads_mech:   float = 0.0
    ramp_time_s:        float = 1.0

    adc_noise_std_a:    float = _DEFAULT_ADC_NOISE_STD
    adc_offset_drift_a: float = _DEFAULT_ADC_DRIFT_PEAK
    adc_offset_tau_s:   float = _DEFAULT_ADC_DRIFT_TAU

    enc_glitch_enable:  bool  = True
    enc_glitch_prob:    float = _DEFAULT_ENC_GLITCH_PROB

    seed: int = 42

    ramp_rate: float = field(init=False)

    def __post_init__(self) -> None:
        if self.ramp_time_s <= 0.0:
            raise ValueError(
                f"MaschineConnectCfg: ramp_time_s must be > 0, got {self.ramp_time_s}"
            )
        self.ramp_rate = self.target_rads_mech / self.ramp_time_s   # [rad/s**2]


# ===========================================================================
# ADC channel noise model
# ===========================================================================

class _AdcChannel:
    """
    Single ADC channel: white Gaussian noise + Ornstein-Uhlenbeck offset drift.

    OU update (Euler-Maruyama, dt-aware):
        dx = -(x / tau) * dt  +  sqrt(2 * sigma**2 / tau) * dW

    Steady-state variance  = sigma**2
    Peak (3-sigma clip)    = 3 * sigma

    The drift models op-amp Vos temperature dependence and PCB self-heating.
    The white noise models ADC quantisation + thermal noise floor.

    Parameters
    ----------
    noise_std  : 1-sigma white noise amplitude [A]
    drift_peak : steady-state sigma of OU process [A]
    drift_tau  : OU correlation time [s]
    rng        : independent numpy Generator for this channel
    """

    def __init__(
        self,
        noise_std:  float,
        drift_peak: float,
        drift_tau:  float,
        rng:        np.random.Generator,
    ) -> None:
        self._noise_std  = float(noise_std)
        self._drift_peak = float(drift_peak)
        self._drift_tau  = max(float(drift_tau), 1e-9)
        self._rng        = rng
        self._offset: float = 0.0   # OU state [A]

    def reset(self) -> None:
        self._offset = 0.0

    def sample(self, true_value: float, dt: float) -> float:
        """Return noise-corrupted ADC reading for one simulation step."""

        # -- Ornstein-Uhlenbeck drift -----------------------------------------
        if self._drift_peak > 0.0:
            diffusion = math.sqrt(2.0 * self._drift_peak ** 2 / self._drift_tau)
            dW = self._rng.standard_normal() * math.sqrt(dt)
            self._offset += (-self._offset / self._drift_tau) * dt + diffusion * dW
            # soft 3-sigma clip -- prevents runaway during cold start
            clip = 3.0 * self._drift_peak
            if self._offset > clip:
                self._offset = clip
            elif self._offset < -clip:
                self._offset = -clip

        # -- white noise ------------------------------------------------------
        white = (
            self._rng.standard_normal() * self._noise_std
            if self._noise_std > 0.0
            else 0.0
        )

        return true_value + self._offset + white


# ===========================================================================
# MaschineConnect -- main VectorBlock
# ===========================================================================

class MaschineConnect(VectorBlock):
    """
    Fachschale Elektrische Maschinen -- Sensor Interface Block.

    Assembles the SMC_Input_T / EmbedSim_Input_T bus from raw motor feedback
    and speed reference, applying hardware-realistic sensor models:

      * Speed reference ramp       -- protects against current-limiter trips
      * GPT12 encoder glitch model -- turn-rollover race condition (AURIX)
      * ADC current noise          -- white noise + OU offset drift per phase

    Drop-in replacement for CtrlPacker.  Identical port contract and output
    bus layout.  All noise can be zeroed via MaschineConnectCfg for clean runs.

    CodeGen note
    ------------
    C_CODEGEN_EXCLUDE = True.  StepGenerator uses INPUT_NAMES and
    C_FIELD_COMMENTS to annotate EmbedSim_Input_T fields only; this block
    does not emit a C translation unit.
    """

    INPUT_NAMES       = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP        = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m":        "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia":             "Phase-A current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
        "ib":             "Phase-B current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
        "ic":             "Phase-C current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
    }

    def __init__(
        self,
        name: str = "maschine_connect",
        cfg:  Optional[MaschineConnectCfg] = None,
        **kw,
    ) -> None:
        super().__init__(name, **kw)
        self.cfg          = cfg if cfg is not None else MaschineConnectCfg()
        self.output_label = "[w_ref,th_m,ia,ib,ic]"

        # -- speed ramp state -------------------------------------------------
        self._omega_ref_filt: float = 0.0

        # -- encoder model state ----------------------------------------------
        self._enc_theta_prev: float = 0.0

        # -- independent RNG streams (seed+N per concern) ---------------------
        s = self.cfg.seed
        self._enc_rng = np.random.default_rng(s)

        self._adc_ia = _AdcChannel(
            noise_std  = self.cfg.adc_noise_std_a,
            drift_peak = self.cfg.adc_offset_drift_a,
            drift_tau  = self.cfg.adc_offset_tau_s,
            rng        = np.random.default_rng(s + 1),
        )
        self._adc_ib = _AdcChannel(
            noise_std  = self.cfg.adc_noise_std_a,
            drift_peak = self.cfg.adc_offset_drift_a,
            drift_tau  = self.cfg.adc_offset_tau_s,
            rng        = np.random.default_rng(s + 2),
        )
        self._adc_ic = _AdcChannel(
            noise_std  = self.cfg.adc_noise_std_a,
            drift_peak = self.cfg.adc_offset_drift_a,
            drift_tau  = self.cfg.adc_offset_tau_s,
            rng        = np.random.default_rng(s + 3),
        )

    # -------------------------------------------------------------------------
    def reset(self) -> None:
        super().reset()
        self._omega_ref_filt = 0.0
        self._enc_theta_prev = 0.0
        self._adc_ia.reset()
        self._adc_ib.reset()
        self._adc_ic.reset()

    # -------------------------------------------------------------------------
    def _apply_speed_ramp(self, omega_target: float, dt: float) -> float:
        """
        Rate-limit speed reference to cfg.ramp_rate [rad/s**2].

        Symmetric clamp on domega/dt prevents both aggressive acceleration
        and sudden deceleration from hitting the current limiter.
        """
        max_step = self.cfg.ramp_rate * dt
        delta    = omega_target - self._omega_ref_filt
        if delta > max_step:
            delta = max_step
        elif delta < -max_step:
            delta = -max_step
        self._omega_ref_filt += delta
        return self._omega_ref_filt

    # -------------------------------------------------------------------------
    def _apply_encoder_model(self, theta_m_true: float) -> float:
        """
        GPT12 incremental encoder model -- turn-rollover glitch injection.

        Models the AURIX IfxGpt12_IncrEnc_getAbsolutePosition race condition:
        at a mechanical turn boundary rawPosition may read 0 before the turn
        counter increments, so reported theta_m is one full turn (2*pi) behind
        for a single sample period.

        After unwrap in the SMO speed estimator this appears as a lost step
        (delta approx. 0) rather than a velocity spike -- consistent with hw.

        Parameters
        ----------
        theta_m_true : float
            True mechanical angle [rad], accumulating (not wrapped).

        Returns
        -------
        float
            Encoder-reported angle [rad] with optional glitch applied.
        """
        if not self.cfg.enc_glitch_enable:
            self._enc_theta_prev = theta_m_true
            return theta_m_true

        turns_prev  = int(self._enc_theta_prev / _TWO_PI)
        turns_now   = int(theta_m_true         / _TWO_PI)
        at_rollover = (turns_now > turns_prev) and (theta_m_true > 0.0)

        if at_rollover and (self._enc_rng.random() < self.cfg.enc_glitch_prob):
            theta_m = theta_m_true - _TWO_PI   # race: turn counter not yet incremented
        else:
            theta_m = theta_m_true

        self._enc_theta_prev = theta_m_true
        return theta_m

    # -------------------------------------------------------------------------
    def compute_py(self, t: float, dt: float, input_values=None):
        """Core compute -- pure Python path."""

        # -- unpack input ports -----------------------------------------------
        m = (
            input_values[0].value
            if input_values and len(input_values) > 0
            else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE)
        )
        r = (
            input_values[1].value
            if input_values and len(input_values) > 1
            else np.zeros(1, dtype=DEFAULT_DTYPE)
        )

        omega_target = float(r[0])              if len(r) > 0          else 0.0
        ia_true      = float(m[_IDX_IA])        if len(m) > _IDX_IA    else 0.0
        ib_true      = float(m[_IDX_IB])        if len(m) > _IDX_IB    else 0.0
        ic_true      = float(m[_IDX_IC])        if len(m) > _IDX_IC    else 0.0
        theta_m_true = float(m[_IDX_THETA])     if len(m) > _IDX_THETA else 0.0

        # -- sensor models ----------------------------------------------------
        omega_ref = self._apply_speed_ramp(omega_target, dt)
        theta_m   = self._apply_encoder_model(theta_m_true)
        ia        = self._adc_ia.sample(ia_true, dt)
        ib        = self._adc_ib.sample(ib_true, dt)
        ic        = self._adc_ic.sample(ic_true, dt)

        # -- assemble output bus ----------------------------------------------
        self.output = VectorSignal(
            np.array([omega_ref, theta_m, ia, ib, ic], dtype=DEFAULT_DTYPE),
            self.name,
        )
        return self.output

    def compute(self, t: float, dt: float, input_values=None):
        return self.compute_py(t, dt, input_values)
