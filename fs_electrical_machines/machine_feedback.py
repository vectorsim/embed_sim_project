"""
machine_feedback.py
===================
EmbedSim — reusable sensor-noise / hardware-artefact library for PMSM
closed-loop simulations.

Motivation
----------
`CtrlPacker` in db42s02_closed_loop_smc_foc_20k.py mixed three concerns:
  1. Bus re-packing (motor feedback → SMC input)
  2. Speed reference ramping
  3. GPT12 encoder roll-over glitch model

This module isolates concern 3 (and extends it) into composable,
independently testable noise-model objects that any CtrlPacker-equivalent
block can call without duplicating physics.

Architecture
------------
Every noise model is a plain Python class (no VectorBlock overhead).
Each exposes a single `.apply(value, t, dt, rng)` method that returns the
corrupted value.  The caller owns the RNG — pass the simulation's
`np.random.default_rng(seed)` so seeds are reproducible and centralised.

Provided models
---------------
  EncoderGlitch      GPT12 turn-rollover race condition (original CtrlPacker logic,
                     isolated and parameterised)
  AdcNoise           Gaussian ADC quantisation noise on phase currents
  AdcOffset          Systematic DC offset per phase (current sensor bias)
  AdcSaturation      Hard-clip at ±I_MAX (ADC rail)
  SpeedIirNoise      Additive white noise on the speed estimate (models
                     resolver/encoder noise floor)
  MachineFeedback    Compositor: applies an ordered pipeline of noise models
                     to the full motor feedback bus

Usage example
-------------
    from machine_feedback import MachineFeedback, EncoderGlitch, AdcNoise

    rng = np.random.default_rng(seed=42)

    fb = MachineFeedback(
        models=[
            EncoderGlitch(glitch_prob=0.15),
            AdcNoise(sigma_a=0.003),
        ]
    )

    # Inside CtrlPacker.compute_py():
    bus_corrupted = fb.apply(motor_bus, t, dt, rng)
    theta_m = bus_corrupted[IDX_THETA_M]
    ia      = bus_corrupted[IDX_IA]
    ...

Motor feedback bus layout (matches DB42S02PlantBlock output)
------------------------------------------------------------
  [0] rpm      [RPM]
  [1] ia       [A]
  [2] ib       [A]
  [3] ic       [A]
  [4] theta_m  [rad]   accumulating, unwrapped
  [5] T_em     [N.m]
  [6] id       [A]
  [7] iq       [A]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np

# Bus index constants — import these in CtrlPacker to avoid magic numbers
IDX_RPM     = 0
IDX_IA      = 1
IDX_IB      = 2
IDX_IC      = 3
IDX_THETA_M = 4
IDX_T_EM    = 5
IDX_ID      = 6
IDX_IQ      = 7
MOTOR_BUS_SIZE = 8


# =============================================================================
# Protocol — every noise model must satisfy this interface
# =============================================================================

class NoiseModel(Protocol):
    """
    Duck-typed interface for noise/artefact models.

    Parameters
    ----------
    bus : np.ndarray  Motor feedback bus (8 elements, float32/float64).
    t   : float       Current simulation time [s].
    dt  : float       Current time step [s].
    rng : np.random.Generator

    Returns
    -------
    np.ndarray  Corrupted bus (same shape, may be a copy or in-place).
    """

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: np.random.Generator) -> np.ndarray:
        ...


# =============================================================================
# EncoderGlitch  —  GPT12 turn-rollover race condition
# =============================================================================

@dataclass
class EncoderGlitch:
    """
    Simulates the GPT12 `IfxGpt12_IncrEnc_getAbsolutePosition` race condition.

    At each mechanical turn boundary the rawPosition register may read 0
    before the turn counter increments, causing the reported theta_m to jump
    back by ~2π for one sample period.  In the downstream speed estimator
    this manifests as a lost step (delta_theta → 0) or a spurious negative
    speed spike.

    Parameters
    ----------
    glitch_prob  : float  Probability [0, 1] that the race condition fires on
                          any given turn boundary crossing.  Hardware default
                          observed on TC3xx: ~0.10–0.20 at 20 kHz / 500 PPR.
    rollback_rad : float  Angular rollback when glitch fires [rad].
                          Default = 2π (one full mechanical turn).
    enabled      : bool   Global on/off switch.
    """

    glitch_prob:  float = 0.15
    rollback_rad: float = 2.0 * math.pi
    enabled:      bool  = True

    # Internal state — not a constructor parameter
    _theta_prev: float = field(default=0.0, init=False, repr=False)

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: np.random.Generator) -> np.ndarray:
        if not self.enabled:
            return bus

        theta_true = float(bus[IDX_THETA_M])

        # Detect turn boundary: integer number of full turns crossed
        turns_prev = int(self._theta_prev / (2.0 * math.pi))
        turns_now  = int(theta_true       / (2.0 * math.pi))
        at_rollover = (turns_now > turns_prev) and (theta_true > 0.0)

        theta_out = theta_true
        if at_rollover and rng.random() < self.glitch_prob:
            theta_out = theta_true - self.rollback_rad

        self._theta_prev = theta_true

        out = bus.copy()
        out[IDX_THETA_M] = theta_out
        return out

    def reset(self) -> None:
        self._theta_prev = 0.0


# =============================================================================
# AdcNoise  —  Gaussian quantisation noise on phase currents
# =============================================================================

@dataclass
class AdcNoise:
    """
    Additive white Gaussian noise on the three phase-current ADC channels.

    Models:
      - ADC quantisation noise (LSB/√12 for uniform distribution)
      - Current sense resistor thermal noise
      - PCB trace pickup

    Parameters
    ----------
    sigma_a, sigma_b, sigma_c : float  [A]  Noise standard deviation per phase.
                                            Set to 0.0 to disable a channel.
    enabled : bool
    """

    sigma_a: float = 0.003   # [A]  ~3 mA rms (typical 12-bit ADC on ±10 A range)
    sigma_b: float = 0.003
    sigma_c: float = 0.003
    enabled: bool  = True

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: np.random.Generator) -> np.ndarray:
        if not self.enabled:
            return bus

        out = bus.copy()
        if self.sigma_a > 0.0:
            out[IDX_IA] += rng.normal(0.0, self.sigma_a)
        if self.sigma_b > 0.0:
            out[IDX_IB] += rng.normal(0.0, self.sigma_b)
        if self.sigma_c > 0.0:
            out[IDX_IC] += rng.normal(0.0, self.sigma_c)
        return out


# =============================================================================
# AdcOffset  —  Systematic DC bias per phase
# =============================================================================

@dataclass
class AdcOffset:
    """
    Static DC offset on each phase-current channel.

    Models op-amp input offset, current-sensor zero-point drift, or
    deliberate imbalance injection for robustness testing.

    Parameters
    ----------
    offset_a, offset_b, offset_c : float [A]  DC offset per phase.
    enabled : bool
    """

    offset_a: float = 0.0
    offset_b: float = 0.0
    offset_c: float = 0.0
    enabled:  bool  = True

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or (
                self.offset_a == 0.0
                and self.offset_b == 0.0
                and self.offset_c == 0.0):
            return bus

        out = bus.copy()
        out[IDX_IA] += self.offset_a
        out[IDX_IB] += self.offset_b
        out[IDX_IC] += self.offset_c
        return out


# =============================================================================
# AdcSaturation  —  Hard-clip at ±I_MAX
# =============================================================================

@dataclass
class AdcSaturation:
    """
    Hard rail clamp on all three phase-current channels.

    Models the ADC full-scale limit.  Should be the *last* stage in the
    noise pipeline so upstream noise and offset are clipped correctly.

    Parameters
    ----------
    i_max : float [A]  Clamp magnitude (applied symmetrically: ±i_max).
    enabled : bool
    """

    i_max:   float = 10.0
    enabled: bool  = True

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: np.random.Generator) -> np.ndarray:
        if not self.enabled:
            return bus

        out = bus.copy()
        for idx in (IDX_IA, IDX_IB, IDX_IC):
            out[idx] = max(-self.i_max, min(self.i_max, float(out[idx])))
        return out


# =============================================================================
# SpeedIirNoise  —  Additive noise on the speed estimate (rpm channel)
# =============================================================================

@dataclass
class SpeedIirNoise:
    """
    White noise on the rpm channel of the motor feedback bus.

    Represents encoder pulse-count jitter, resolver harmonics, or
    digital differentiation noise in the speed estimator.

    Note: the SMC CtrlPacker computes its own IIR speed estimate from
    delta_theta — this model injects noise *before* that computation by
    perturbing theta_m, which is the more physically correct injection
    point.  Use `perturb_theta` to choose:
      True  → noise added to theta_m [rad]  (sigma unit: rad/sample)
      False → noise added to rpm directly   (sigma unit: RPM)

    Parameters
    ----------
    sigma       : float  Noise standard deviation (units depend on perturb_theta).
    perturb_theta : bool  True = inject into theta_m; False = inject into rpm.
    enabled : bool
    """

    sigma:         float = 0.0002   # [rad/sample] ≈ 0.01 deg; realistic for 500 PPR encoder
    perturb_theta: bool  = True
    enabled:       bool  = True

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or self.sigma <= 0.0:
            return bus

        out = bus.copy()
        noise = rng.normal(0.0, self.sigma)
        if self.perturb_theta:
            out[IDX_THETA_M] += noise
        else:
            out[IDX_RPM] += noise
        return out


# =============================================================================
# MachineFeedback  —  compositor / pipeline
# =============================================================================

class MachineFeedback:
    """
    Ordered pipeline of noise/artefact models applied to the motor feedback bus.

    Each model's `.apply()` output is fed as input to the next model —
    i.e. the models compose left-to-right.  Order matters: put
    `AdcSaturation` last so it clips the combined effect of offset + noise.

    Parameters
    ----------
    models   : list of NoiseModel  Pipeline stages.
    rng_seed : int | None          If given, creates an internal RNG with
                                   this seed.  If None, caller must pass rng
                                   to every `.apply()` call.

    Usage
    -----
    Stateless (caller owns RNG) — preferred for reproducibility::

        rng = np.random.default_rng(seed=42)
        fb  = MachineFeedback(models=[EncoderGlitch(), AdcNoise()])
        bus_out = fb.apply(bus_in, t, dt, rng)

    Stateful (internal RNG)::

        fb  = MachineFeedback(models=[EncoderGlitch()], rng_seed=42)
        bus_out = fb.apply(bus_in, t, dt)   # rng optional
    """

    def __init__(self,
                 models:   Optional[List] = None,
                 rng_seed: Optional[int]  = None):
        self.models   = models or []
        self._rng     = np.random.default_rng(seed=rng_seed) if rng_seed is not None else None

    def apply(self,
              bus: np.ndarray,
              t:   float,
              dt:  float,
              rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Run the full noise pipeline.

        Parameters
        ----------
        bus : np.ndarray  Raw motor feedback bus (8 elements).
        t   : float       Simulation time [s].
        dt  : float       Time step [s].
        rng : Generator | None  External RNG.  If None, uses internal RNG
                                (only available if rng_seed was passed to __init__).

        Returns
        -------
        np.ndarray  Corrupted bus (copy — original is never modified).
        """
        _rng = rng if rng is not None else self._rng
        if _rng is None:
            raise ValueError(
                "MachineFeedback: no RNG available.  Pass rng= or set rng_seed= at construction."
            )

        out = bus.copy()
        for model in self.models:
            out = model.apply(out, t, dt, _rng)
        return out

    def reset(self) -> None:
        """Reset all stateful models (e.g. EncoderGlitch._theta_prev)."""
        for m in self.models:
            if hasattr(m, "reset"):
                m.reset()

    def enable_all(self, enabled: bool = True) -> None:
        """Bulk-enable or disable all models (useful for clean baseline runs)."""
        for m in self.models:
            if hasattr(m, "enabled"):
                m.enabled = enabled

    def __repr__(self) -> str:
        names = [type(m).__name__ for m in self.models]
        return f"MachineFeedback([{', '.join(names)}])"


# =============================================================================
# Preset factory — DB42S02 / AURIX TC3xx hardware profile
# =============================================================================

def db42s02_feedback_profile(
        enc_glitch:    bool  = True,
        enc_prob:      float = 0.15,
        adc_noise:     bool  = True,
        adc_sigma:     float = 0.003,
        adc_offset:    bool  = False,
        adc_sat:       bool  = True,
        i_max:         float = 3.57,
        speed_noise:   bool  = False,
        rng_seed:      Optional[int] = None,
) -> MachineFeedback:
    """
    Factory: returns a MachineFeedback configured for the NANOTEC DB42S02
    running on AURIX TC3xx at 20 kHz.

    Pipeline order (correct physical sequence):
      EncoderGlitch → SpeedIirNoise → AdcNoise → AdcOffset → AdcSaturation

    Parameters
    ----------
    enc_glitch  : bool   Enable GPT12 turn-rollover glitch.
    enc_prob    : float  Glitch probability per turn boundary.
    adc_noise   : bool   Enable Gaussian ADC noise on ia/ib/ic.
    adc_sigma   : float  [A] Noise std-dev (same for all three phases).
    adc_offset  : bool   Enable per-phase DC offset (defaults to 0 A — no effect
                         unless you override with .offset_a etc. after construction).
    adc_sat     : bool   Enable hard-rail clip at ±i_max.
    i_max       : float  [A] ADC rail magnitude.
    speed_noise : bool   Enable theta_m encoder jitter.
    rng_seed    : int|None  Internal RNG seed (pass None to use caller's RNG).

    Returns
    -------
    MachineFeedback instance ready to use.
    """
    pipeline = []

    if enc_glitch:
        pipeline.append(EncoderGlitch(glitch_prob=enc_prob))

    if speed_noise:
        pipeline.append(SpeedIirNoise())   # default sigma = 0.0002 rad/sample

    if adc_noise:
        pipeline.append(AdcNoise(sigma_a=adc_sigma,
                                 sigma_b=adc_sigma,
                                 sigma_c=adc_sigma))

    if adc_offset:
        pipeline.append(AdcOffset())       # zero offsets — caller sets .offset_a etc.

    if adc_sat:
        pipeline.append(AdcSaturation(i_max=i_max))

    return MachineFeedback(models=pipeline, rng_seed=rng_seed)
