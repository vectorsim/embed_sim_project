"""
ctrl_packer.py
==============
EmbedSim — CtrlPacker block for db42s02_closed_loop_smc_foc_20k.py.

This is the cleaned version of the inline CtrlPacker class that previously
lived inside the simulation file.  Sensor-noise and hardware-artefact
logic is now delegated to MachineFeedback so the block stays focused on
its single responsibility: bus re-packing + speed ramp.

Drop-in replacement:
  Replace the CtrlPacker class definition in db42s02_closed_loop_smc_foc_20k.py
  with:
      from ctrl_packer import CtrlPacker

  And add to the constants section:
      ENC_GLITCH_ENABLE = True   # or False for clean baseline
      ENC_GLITCH_PROB   = 0.15

  (Both constants were already present in the simulation file — no change needed.)

Block responsibilities
----------------------
  1. Bus re-packing  : motor[8] + speed_ref[1] → SMC_Input_T[5]
  2. Speed ramp      : rate-limits omega_ref to avoid current limiter trip
  3. Noise pipeline  : delegated to MachineFeedback (composable, testable)

Input ports
-----------
  [0] motor feedback bus   [rpm,ia,ib,ic,theta_m,T_em,id,iq]   8 elements
  [1] speed reference      [rad/s]   scalar (VectorStep output)

Output bus (5 elements — SMC_Input_T):
  [0] omega_ref_mech  [rad/s]
  [1] theta_m         [rad]
  [2] ia              [A]
  [3] ib              [A]
  [4] ic              [A]
"""

from __future__ import annotations

import numpy as np

from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from machine_feedback import (
    MachineFeedback,
    IDX_IA, IDX_IB, IDX_IC, IDX_THETA_M,
    MOTOR_BUS_SIZE,
    db42s02_feedback_profile,
)

# These constants are imported from the simulation file at usage time.
# They are declared here only as module-level defaults so CtrlPacker can be
# instantiated stand-alone (e.g. in unit tests) without importing the full sim.
_DEFAULT_TARGET_RADS_MECH = 209.44   # 2000 RPM in rad/s
_DEFAULT_RAMP_TIME        = 0.5      # [s]


class CtrlPacker(VectorBlock):
    """
    Packs motor feedback + speed reference into the SMC input bus.

    Parameters
    ----------
    name              : str          Block name (topology / CodeGen label).
    target_rads_mech  : float        Speed setpoint [rad/s] — used only to
                                     compute ramp rate.  Pass TARGET_RADS_MECH
                                     from the simulation constants.
    ramp_time         : float [s]    Time to ramp from 0 to target_rads_mech.
    feedback          : MachineFeedback | None
                                     Noise pipeline.  If None, a clean
                                     (no-noise) pipeline is used.
    rng_seed          : int | None   RNG seed for the noise pipeline.
                                     Ignored when feedback is provided externally.
    """

    INPUT_NAMES       = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP        = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    # CodeGen field comments — picked up by StepGenerator on the CodeGenStart boundary
    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m":        "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia":             "Phase-A current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
        "ib":             "Phase-B current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
        "ic":             "Phase-C current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
    }

    def __init__(self,
                 name:             str   = "ctrl_packer",
                 target_rads_mech: float = _DEFAULT_TARGET_RADS_MECH,
                 ramp_time:        float = _DEFAULT_RAMP_TIME,
                 feedback:         MachineFeedback | None = None,
                 rng_seed:         int | None             = 42,
                 **kw):
        super().__init__(name, **kw)

        self.output_label = "[w_ref,th_m,ia,ib,ic]"
        self._ramp_rate   = target_rads_mech / ramp_time   # [rad/s²]

        # Noise pipeline — default: DB42S02 hardware profile, glitch enabled
        if feedback is not None:
            self._fb  = feedback
            self._rng = np.random.default_rng(seed=rng_seed)
        else:
            self._fb  = db42s02_feedback_profile(rng_seed=rng_seed)
            self._rng = None   # MachineFeedback owns its own RNG when rng_seed given

        # Internal state
        self._omega_ref_filt: float = 0.0

    # ── public API for noise control ─────────────────────────────────────────

    def set_noise_enabled(self, enabled: bool) -> None:
        """
        Bulk-enable or disable the entire noise pipeline.

        Useful for clean baseline runs without constructing a separate block:
            ctrl.set_noise_enabled(False)   # before sim.run()
        """
        self._fb.enable_all(enabled)

    def reset(self) -> None:
        super().reset()
        self._omega_ref_filt = 0.0
        self._fb.reset()

    # ── compute ──────────────────────────────────────────────────────────────

    def compute_py(self, t, dt, input_values=None):
        # ── Unpack inputs ────────────────────────────────────────────────────
        m = (input_values[0].value
             if input_values and len(input_values) > 0
             else np.zeros(MOTOR_BUS_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value
             if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))

        # ── Speed ramp ───────────────────────────────────────────────────────
        omega_target = float(r[0]) if len(r) > 0 else 0.0
        max_step     = self._ramp_rate * dt
        self._omega_ref_filt += max(
            -max_step,
            min(max_step, omega_target - self._omega_ref_filt))

        # ── Noise pipeline ───────────────────────────────────────────────────
        # MachineFeedback returns a copy; m is never modified.
        m_noisy = self._fb.apply(m, t, dt, self._rng)

        # ── Pack output bus ──────────────────────────────────────────────────
        self.output = VectorSignal(np.array([
            self._omega_ref_filt,
            float(m_noisy[IDX_THETA_M]) if len(m_noisy) > IDX_THETA_M else 0.0,
            float(m_noisy[IDX_IA])      if len(m_noisy) > IDX_IA      else 0.0,
            float(m_noisy[IDX_IB])      if len(m_noisy) > IDX_IB      else 0.0,
            float(m_noisy[IDX_IC])      if len(m_noisy) > IDX_IC      else 0.0,
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)
