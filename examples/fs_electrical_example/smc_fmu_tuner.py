"""
smc_fmu_tuner.py
================
Differential Evolution + Bayesian GP optimiser for SMC gains.
Uses the NANOTEC DB42S02 PMSM FMU plant from db42s02_openloop_fmu.py.

Strategy
--------
  Phase 1 — Differential Evolution (global, immediate updating, single worker)
            Finds the basin.  Fast because each FMU run is ~0.1 s.

  Phase 2 — Gaussian Process Bayesian Opt (local refinement, ~50 evals)
            Polishes inside the basin found by DE.

Parameters tuned  →  embed_sim_smc_controller.h macro names
------------------------------------------------------------
  SMC_KS_W    Speed SMC switching gain      [N·m]
  SMC_ETA_W   Speed SMC linear damping term [—]
  SMC_PHI_W   Speed SMC boundary layer      [rad/s]
  SMC_KS_I    Current SMC switching gain    [V]   ← replaces L_D*WC_I expression
  SMC_PHI_I   Current SMC boundary layer    [A]

Cost function  (ITAE + overshoot + chattering + steady-state iq)
----------------------------------------------------------------
  J = ∫ t·|ω_ref - ω_m| dt          ← ITAE speed error
    + 2·overshoot²                   ← square penalty for overshoot
    + 0.05·∫ (Δiq)² dt              ← chattering penalty
    + 0.1·mean|iq| (last 20%)       ← no-load steady-state current

Output
------
  smc_best_gains.json          — best gains + full history
  embed_sim_smc_controller.h   — header patched in-place with new #defines
  smc_tuner_verify.png         — speed / dq current / torque plot (--verify)

Usage
-----
  python smc_fmu_tuner.py [--rpm 400] [--t_sim 1.0] [--de_iters 50]
                          [--gp_iters 40] [--header path/to/embed_sim_smc_controller.h]
                          [--out gains.json] [--verify]

Paul Abraham / EmbedSim 2025
"""

from __future__ import annotations

import sys
import math
import json
import time
import argparse
import logging
import warnings
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

from scipy.optimize import differential_evolution, minimize
from scipy.linalg import solve

# ── NumPy 2.0 compatibility ───────────────────────────────────────────────────
# np.trapz was removed in NumPy 2.0; np.trapezoid is the replacement.
# We define _trapz once here so every call site is clean.
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid          # NumPy ≥ 2.0
else:
    _trapz = np.trapz              # NumPy < 2.0  (legacy)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SMC_Tuner")


# =============================================================================
#  Gain container  —  field names match embed_sim_smc_controller.h #defines
# =============================================================================

@dataclass
class SMCGains:
    """
    Five tunable SMC gains.

    Field names are IDENTICAL to the #define macros in
    embed_sim_smc_controller.h so that patch_header() can do a
    straight substitution with no translation layer.

    Header macro          │ Field      │ Units    │ Description
    ──────────────────────┼────────────┼──────────┼──────────────────────────
    SMC_KS_W              │ SMC_KS_W   │ N·m      │ Speed switching gain
    SMC_ETA_W             │ SMC_ETA_W  │ —        │ Speed linear damping
    SMC_PHI_W             │ SMC_PHI_W  │ rad/s    │ Speed boundary layer
    SMC_KS_I              │ SMC_KS_I   │ V        │ Current switching gain
    SMC_PHI_I             │ SMC_PHI_I  │ A        │ Current boundary layer
    """
    SMC_KS_W:  float = 0.035    # header default
    SMC_ETA_W: float = 0.1      # header default
    SMC_PHI_W: float = 5.0      # header default  [rad/s]
    SMC_KS_I:  float = 0.6283   # header default  L_D*WC_I = 125e-6*5026.5
    SMC_PHI_I: float = 0.5      # header default  [A]

    # Search bounds — physically motivated
    BOUNDS: List[Tuple[float, float]] = None

    def __post_init__(self):
        self.BOUNDS = [
            ( 0.005,  0.5),    # SMC_KS_W   [N·m]   — J·λ²≈0.038, T_max=0.02
            ( 0.01,   5.0),    # SMC_ETA_W  [—]
            ( 0.5,   50.0),    # SMC_PHI_W  [rad/s] — narrow→chattering, wide→sluggish
            ( 0.1,   10.0),    # SMC_KS_I   [V]     — L_D*WC_i range 0.1–10
            ( 0.05,   2.0),    # SMC_PHI_I  [A]     — fraction of I_MAX=3.57 A
        ]

    def to_array(self) -> np.ndarray:
        return np.array([self.SMC_KS_W, self.SMC_ETA_W,
                         self.SMC_PHI_W, self.SMC_KS_I,
                         self.SMC_PHI_I], dtype=np.float64)

    @classmethod
    def from_array(cls, x: np.ndarray) -> "SMCGains":
        g = cls()
        (g.SMC_KS_W, g.SMC_ETA_W, g.SMC_PHI_W,
         g.SMC_KS_I, g.SMC_PHI_I) = (
            float(x[0]), float(x[1]), float(x[2]),
            float(x[3]), float(x[4]))
        return g

    def normalise(self) -> np.ndarray:
        x = self.to_array()
        return np.array([(x[i] - self.BOUNDS[i][0]) /
                         (self.BOUNDS[i][1] - self.BOUNDS[i][0])
                         for i in range(5)])

    @classmethod
    def denormalise(cls, z: np.ndarray,
                    bounds: List[Tuple[float, float]]) -> "SMCGains":
        x = np.array([z[i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
                      for i in range(5)])
        return cls.from_array(x)

    def __str__(self) -> str:
        return (f"SMC_KS_W={self.SMC_KS_W:.5f}  SMC_ETA_W={self.SMC_ETA_W:.4f}  "
                f"SMC_PHI_W={self.SMC_PHI_W:.3f}  "
                f"SMC_KS_I={self.SMC_KS_I:.4f}  SMC_PHI_I={self.SMC_PHI_I:.4f}")


# =============================================================================
#  Pure-Python SMC step  (mirrors embed_sim_smc_controller.c exactly)
#  Used instead of importing EmbedSim blocks — no FMU wiring needed.
# =============================================================================

class SMCControllerPy:
    """
    Python reimplementation of SMC_Controller_Step.
    Matches embed_sim_smc_controller.c  (Tustin integrator default).

    Sliding surface coefficients come from the header:
      SMC_LAMBDA_W = 2π×20 Hz = 125.664 rad/s
      SMC_GAMMA_W  = 2π×5  Hz =  31.416 rad/s

    Tunable gains (SMCGains fields):
      SMC_KS_W, SMC_ETA_W, SMC_PHI_W  — speed loop
      SMC_KS_I, SMC_PHI_I             — current loop
    """
    # ── DB42S02 motor parameters (from header) ────────────────────────────────
    R_S       = 0.19
    L_D       = 0.125e-3
    L_Q       = 0.125e-3
    LAMBDA_PM = 0.0014
    P_POLES   = 4
    V_MAX     = 17.0 / 1.73205080757   # SMC_V_MAX = V_DC / sqrt(3) = 9.815 V
    I_MAX     = 3.57                   # SMC_I_MAX

    # ── Fixed sliding surface coefficients (from header — NOT tuned) ─────────
    LAMBDA_W  = 125.66370614359172     # SMC_LAMBDA_W = 2π×20 Hz  [rad/s]
    GAMMA_W   =  31.41592653589793     # SMC_GAMMA_W  = 2π×5  Hz  [rad/s]

    def __init__(self, gains: SMCGains):
        self.g = gains
        self.int_e      = 0.0
        self.int_int_e  = 0.0
        self.e_prev     = 0.0
        self.int_e_prev = 0.0
        self.omega_filt     = 0.0
        self.theta_m_prev   = 0.0
        self.id_ref = 0.0
        self.iq_ref = 0.0

        # ── Transform block instances — canonical, no inline math ────────────
        from _path_utils import get_project_root
        _ROOT = get_project_root()
        _FS   = _ROOT / "fs_electrical_machines"
        for _p in (str(_ROOT / "embedsim"), str(_FS), str(_FS / "c_src")):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from coordinate_transform_blocks import (
            ClarkeTransformBlock, ParkTransformBlock, InvParkTransformBlock)
        self._ct_clarke   = ClarkeTransformBlock("_tc_clarke",    use_c_backend=False)
        self._ct_park     = ParkTransformBlock("_tc_park",        use_c_backend=False)
        self._ct_inv_park = InvParkTransformBlock("_tc_inv_park", use_c_backend=False)

    def reset(self):
        self.int_e = self.int_int_e = 0.0
        self.e_prev = self.int_e_prev = 0.0
        self.omega_filt = self.theta_m_prev = 0.0
        self.id_ref = self.iq_ref = 0.0
        self._ct_clarke.reset()
        self._ct_park.reset()
        self._ct_inv_park.reset()

    @staticmethod
    def _clamp(v: float, lim: float) -> float:
        return max(-lim, min(lim, v))

    @staticmethod
    def _sat(x: float, phi: float) -> float:
        if phi <= 0.0:
            return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)
        return max(-1.0, min(1.0, x / phi))

    def _clarke(self, ia, ib, ic):
        """Delegates to ClarkeTransformBlock — canonical formula."""
        import numpy as np
        from embedsim.core_blocks import VectorSignal
        inp = VectorSignal(np.array([ia, ib, ic], dtype=np.float32), "_tc")
        out = self._ct_clarke.compute_py(0.0, 0.0, [inp])
        return float(out.value[0]), float(out.value[1])

    def _park(self, i_alpha, i_beta, theta_e):
        """Delegates to ParkTransformBlock — canonical formula."""
        import numpy as np
        from embedsim.core_blocks import VectorSignal
        ab = VectorSignal(np.array([i_alpha, i_beta], dtype=np.float32), "_tc")
        th = VectorSignal(np.array([theta_e],         dtype=np.float32), "_tc")
        out = self._ct_park.compute_py(0.0, 0.0, [ab, th])
        return float(out.value[0]), float(out.value[1])

    def _inv_park(self, vd, vq, theta_e):
        """Delegates to InvParkTransformBlock — canonical formula."""
        import numpy as np
        from embedsim.core_blocks import VectorSignal
        dq = VectorSignal(np.array([vd, vq],   dtype=np.float32), "_tc")
        th = VectorSignal(np.array([theta_e],   dtype=np.float32), "_tc")
        out = self._ct_inv_park.compute_py(0.0, 0.0, [dq, th])
        return float(out.value[0]), float(out.value[1])

    def step(self, theta_m: float, ia: float, ib: float, ic: float,
             omega_ref: float, dt: float):
        """One control step. Returns (v_alpha, v_beta, iq_ref, id_meas, iq_meas)."""
        g = self.g

        # Speed estimation — Euler differentiator + Tustin LPF ~300 Hz
        # alpha = dt / (dt + 1/(2π·f_c))  — matched to C implementation
        # 300 Hz cutoff gives good noise rejection without excessive lag at 20 kHz
        if dt > 0.0:
            omega_raw = (theta_m - self.theta_m_prev) / dt
            _fc   = 300.0                              # cutoff [Hz]
            _tau  = 1.0 / (2.0 * math.pi * _fc)
            _alpha = dt / (dt + _tau)
            self.omega_filt += _alpha * (omega_raw - self.omega_filt)
        self.theta_m_prev = theta_m
        omega_m = self.omega_filt

        theta_e = self.P_POLES * theta_m

        # Clarke + Park — via coordinate_transform_blocks
        i_alpha, i_beta  = self._clarke(ia, ib, ic)
        id_meas, iq_meas = self._park(i_alpha, i_beta, theta_e)

        # ── Speed SMC (Tustin) ────────────────────────────────────────────────
        e = omega_ref - omega_m
        self.int_e     += dt * 0.5 * (e           + self.e_prev)
        self.int_int_e += dt * 0.5 * (self.int_e  + self.int_e_prev)
        self.e_prev     = e
        self.int_e_prev = self.int_e

        s_spd  = e + self.LAMBDA_W * self.int_e + self.GAMMA_W * self.int_int_e
        iq_ref = (g.SMC_KS_W * self._sat(s_spd, g.SMC_PHI_W)
                  + g.SMC_ETA_W * s_spd)
        iq_ref = self._clamp(iq_ref, self.I_MAX)
        self.iq_ref = iq_ref
        self.id_ref = 0.0   # MTPA

        # ── Current SMC — equivalent control + switching ──────────────────────
        omega_e = self.P_POLES * omega_m
        s_d = id_meas - self.id_ref
        s_q = iq_meas - iq_ref

        vd_eq = self.R_S*id_meas  - omega_e*self.L_Q*iq_meas
        vq_eq = self.R_S*iq_meas  + omega_e*(self.L_D*id_meas + self.LAMBDA_PM)

        # Switching term: v_sw = -ks·sat(s/φ)  — Lyapunov stability, negative sign
        # Matches corrected SMC_CurrentSMC() in embed_sim_smc_controller.c
        vd = vd_eq - g.SMC_KS_I * self._sat(s_d, g.SMC_PHI_I)
        vq = vq_eq - g.SMC_KS_I * self._sat(s_q, g.SMC_PHI_I)

        # Voltage saturation (hexagon limiting)
        mag = math.sqrt(vd*vd + vq*vq)
        if mag > self.V_MAX:
            scale = self.V_MAX / mag
            vd *= scale; vq *= scale

        v_alpha, v_beta = self._inv_park(vd, vq, theta_e)
        return v_alpha, v_beta, iq_ref, id_meas, iq_meas


# =============================================================================
#  Python plant wrapper  (replaces FMUPlant — no FMU, no DASSL)
# =============================================================================

class PythonPlant:
    """
    Thin wrapper around PMSM_Python_Plant for the tuner's run_sim loop.
    Identical interface to the old FMUPlant — drop-in replacement.
    Uses the same RK4 + coordinate_transform_blocks pipeline as the
    full EmbedSim simulation.
    """

    def __init__(self, v_dc: float = 17.0, t_load: float = 0.01):
        from _path_utils import get_project_root
        _ROOT = get_project_root()
        _FS   = _ROOT / "fs_electrical_machines"
        for _p in (str(_ROOT / "embedsim"), str(_FS), str(_FS / "c_src")):
            if _p not in sys.path:
                sys.path.insert(0, _p)

        from pmsm_python_plant import PMSM_Python_Plant
        from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE
        self._VectorSignal = VectorSignal
        self._dtype        = DEFAULT_DTYPE

        self._plant = PMSM_Python_Plant(
            name      = "_tuner_motor",
            R         = 0.19,
            L_d       = 0.125e-3,
            L_q       = 0.125e-3,
            lambda_pm = 0.0014,
            J         = 2.4e-6,
            B_fric    = 1e-6,
            p         = 4.0,
            v_dc      = v_dc,
        )
        self.v_dc      = v_dc
        self.t_load    = t_load
        self.theta_m   = 0.0
        self.omega_m   = 0.0
        self.speed_rpm = 0.0
        self.ia = self.ib = self.ic = 0.0
        self.T_em      = 0.0
        self._t        = 0.0

    def step(self, ta: float, tb: float, tc: float, dt: float):
        inp = self._VectorSignal(
            np.array([ta, tb, tc, self.v_dc, self.t_load],
                     dtype=self._dtype))
        out = self._plant.compute_py(self._t, dt, [inp])
        self._t += dt

        v = out.value
        self.speed_rpm = float(v[0])
        self.ia        = float(v[1])
        self.ib        = float(v[2])
        self.ic        = float(v[3])
        self.theta_m   = float(v[4])   # theta_e / p — unwrapped, direct from plant state
        self.omega_m   = self.speed_rpm * 2.0 * math.pi / 60.0
        self.T_em      = float(v[5])

    def reset(self):
        self._t = 0.0
        self._plant.reset()



# =============================================================================
#  SVPWM  (pure Python — matches SVM_CalculateDutyCycle)
# =============================================================================

def svpwm(v_alpha: float, v_beta: float, v_dc: float = 17.0
          ) -> Tuple[float, float, float, int]:
    """
    Space-vector PWM: (v_alpha, v_beta) → (ta, tb, tc, sector).
    Returns duty cycles in [0,1].
    """
    v_ref = math.sqrt(v_alpha**2 + v_beta**2)
    if v_ref < 1e-9:
        return 0.5, 0.5, 0.5, 0

    m = min(v_ref / (v_dc / math.sqrt(3)), 0.95)   # modulation index, clipped
    angle = math.atan2(v_beta, v_alpha) % (2*math.pi)

    sector = int(angle / (math.pi/3)) % 6           # 0-indexed

    ang_in_sec = angle - sector * math.pi/3

    t1 = m * math.sqrt(3) * math.sin(math.pi/3 - ang_in_sec)
    t2 = m * math.sqrt(3) * math.sin(ang_in_sec)
    t0 = 1.0 - t1 - t2

    if t0 < 0.0:                                    # over-modulation clamp
        t1 /= (t1 + t2); t2 = 1.0 - t1; t0 = 0.0

    half0 = t0 / 2.0
    # Standard symmetric SVM switching sequence per sector
    seq = [
        (half0, half0+t1, half0+t1+t2),   # S1
        (half0+t2, half0, half0+t1+t2),   # S2
        (half0+t1+t2, half0, half0+t1),   # S3
        (half0+t1+t2, half0+t2, half0),   # S4
        (half0+t1, half0+t1+t2, half0),   # S5
        (half0, half0+t1+t2, half0+t1),   # S6
    ]
    ta, tb, tc = seq[sector]
    return (float(np.clip(ta, 0.0, 1.0)),
            float(np.clip(tb, 0.0, 1.0)),
            float(np.clip(tc, 0.0, 1.0)),
            sector + 1)


# =============================================================================
#  Simulation runner
# =============================================================================

def run_sim(gains: SMCGains,
            omega_cmd_rpm: float = 400.0,
            t_sim: float         = 2.0,
            dt: float            = 1e-4,
            ramp_time: float     = 0.5,
            v_dc: float          = 17.0,
            t_load: float        = 0.0,
            _plant_cache: list   = []) -> Optional[Dict]:
    """
    Run one closed-loop simulation with given SMC gains.
    Returns dict of time-series arrays, or None if the sim diverged.

    PythonPlant is cached across calls (created once, reset each call) to
    avoid repeated imports and transform block construction which caused
    severe slowdown in the DE hot loop.
    """
    omega_cmd = omega_cmd_rpm * 2.0*math.pi / 60.0

    # ── Plant cache — create once, reset each evaluation ─────────────────────
    if not _plant_cache:
        try:
            _plant_cache.append(PythonPlant(v_dc=v_dc, t_load=t_load))
        except Exception as exc:
            log.error("Plant load failed: %s", exc)
            return None
    plant = _plant_cache[0]
    plant.t_load = t_load
    plant.v_dc   = v_dc
    plant.reset()

    # ── Controller — update gains and reset state ─────────────────────────────
    # SMCControllerPy constructs three transform block instances in __init__.
    # Cache it and reset instead of reconstructing every evaluation.
    if not hasattr(run_sim, '_ctrl_cache'):
        run_sim._ctrl_cache = SMCControllerPy(gains)
    ctrl = run_sim._ctrl_cache
    ctrl.g = gains
    ctrl.reset()

    n_steps = int(t_sim / dt)
    t_arr   = np.zeros(n_steps, dtype=np.float32)
    spd_arr = np.zeros(n_steps, dtype=np.float32)
    ref_arr = np.zeros(n_steps, dtype=np.float32)
    iq_arr  = np.zeros(n_steps, dtype=np.float32)
    id_arr  = np.zeros(n_steps, dtype=np.float32)
    Tem_arr = np.zeros(n_steps, dtype=np.float32)

    def omega_ref(t_now: float) -> float:
        if t_now < ramp_time:
            return omega_cmd * (t_now / ramp_time)
        return omega_cmd

    for k in range(n_steps):
        t_now = k * dt
        ref   = omega_ref(t_now)

        # Controller step — uses plant.theta_m directly (exact, no drift)
        try:
            v_alpha, v_beta, iq_ref, id_meas, iq_meas = ctrl.step(
                plant.theta_m,
                plant.ia, plant.ib, plant.ic,
                ref, dt)
        except Exception:
            return None

        # SVPWM
        ta, tb, tc, _ = svpwm(v_alpha, v_beta, v_dc)

        # Plant step
        try:
            plant.step(ta, tb, tc, dt)
        except Exception:
            return None

        if not math.isfinite(plant.speed_rpm) or abs(plant.speed_rpm) > 50000:
            return None

        # Early-exit: diverging beyond 2× target speed after warm-up (k>200)
        # or non-finite iq — kills most bad candidates 30-70% earlier than
        # waiting for t_sim to expire.
        if k > 200:
            if abs(plant.speed_rpm) > 2.0 * omega_cmd_rpm:
                return None
            if not math.isfinite(iq_meas):
                return None

        t_arr[k]   = t_now
        spd_arr[k] = plant.speed_rpm
        ref_arr[k] = ref * 60.0 / (2.0*math.pi)
        iq_arr[k]  = iq_meas
        id_arr[k]  = id_meas
        Tem_arr[k] = plant.T_em

    return {
        "t":             t_arr,
        "speed_rpm":     spd_arr,
        "omega_ref_rpm": ref_arr,
        "omega_m":       spd_arr * 2.0*math.pi / 60.0,
        "omega_ref":     ref_arr * 2.0*math.pi / 60.0,
        "iq_meas":       iq_arr,
        "id_meas":       id_arr,
        "T_em":          Tem_arr,
    }


# =============================================================================
#  Cost function
# =============================================================================

def cost_function(x: np.ndarray,
                  omega_cmd_rpm: float,
                  t_sim: float,
                  dt: float) -> float:
    """
    ITAE + overshoot + chattering + steady-state iq penalty.
    Returns large value (1e6) for unstable / divergent runs.
    """
    gains = SMCGains.from_array(np.clip(x,
        [b[0] for b in SMCGains().BOUNDS],
        [b[1] for b in SMCGains().BOUNDS]))

    d = run_sim(gains,
                omega_cmd_rpm=omega_cmd_rpm,
                t_sim=t_sim, dt=dt,
                t_load=0.0)

    if d is None:
        return 1e6

    t        = d["t"].astype(np.float64)
    omega_m  = d["omega_m"].astype(np.float64)
    omega_ref= d["omega_ref"].astype(np.float64)
    iq       = d["iq_meas"].astype(np.float64)

    e = np.abs(omega_ref - omega_m)

    # ITAE
    J_itae = float(_trapz(t * e, t))

    # Overshoot — normalised by omega_ss so penalty is dimensionless fraction
    omega_ss  = omega_ref[-1]
    overshoot = max(0.0, float(np.max(omega_m)) - omega_ss)
    _overshoot_norm = overshoot / (omega_ss + 1e-6)   # fractional overshoot [0,1+]
    J_over    = 2.0 * _overshoot_norm**2 * J_itae     # scaled relative to ITAE

    # Chattering — normalised by n_steps so dt-independent
    diq       = np.diff(iq)
    J_chat    = 0.05 * float(np.mean(diq**2)) * t[-1]   # mean-power × duration → same units as ITAE

    # Steady-state iq (last 20% of sim) — should be near zero at no-load
    ss_start  = int(0.80 * len(iq))
    J_iq_ss   = 0.10 * float(np.mean(np.abs(iq[ss_start:])))

    # id penalty (should stay at 0 — MTPA)
    id_       = d["id_meas"].astype(np.float64)
    J_id      = 0.05 * float(np.mean(np.abs(id_[ss_start:])))

    J = J_itae + J_over + J_chat + J_iq_ss + J_id

    return float(J) if math.isfinite(J) else 1e6


# =============================================================================
#  Gaussian Process Bayesian Optimiser  (no skopt dependency)
# =============================================================================

class GaussianProcessBO:
    """
    Minimal Gaussian Process Bayesian Optimiser.
    Kernel: Matérn 5/2  +  white noise nugget.
    Acquisition: Expected Improvement (EI).
    """

    def __init__(self,
                 bounds: List[Tuple[float,float]],
                 noise: float = 1e-4):
        self.bounds   = np.array(bounds)
        self.noise    = noise
        self.X_obs    : List[np.ndarray] = []
        self.y_obs    : List[float]      = []
        # Initialise length-scales to 0.2 (normalised space) — 1.0 is too wide
        # for parameters with very different physical scales (KS_W vs PHI_W).
        # After each _fit() these are updated from the observed data spread.
        self._l       = np.full(len(bounds), 0.2)  # normalised length-scales
        self._sigma_f = 1.0                         # signal variance

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        lo = self.bounds[:, 0]; hi = self.bounds[:, 1]
        return (X - lo) / (hi - lo)

    def _matern52(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Matérn 5/2 kernel in normalised space."""
        D = X1[:, None, :] - X2[None, :, :]         # (n1, n2, d)
        D = D / self._l[None, None, :]
        r = np.sqrt(np.sum(D**2, axis=-1))           # (n1, n2)
        sqrt5 = math.sqrt(5)
        K = (self._sigma_f**2 *
             (1 + sqrt5*r + (5.0/3.0)*r**2) *
             np.exp(-sqrt5 * r))
        return K

    def _fit(self):
        X = self._normalise(np.array(self.X_obs))    # (n, d)
        y = np.array(self.y_obs)
        # Adapt length-scales to observed data spread in normalised space.
        # std of each dimension + small floor avoids degenerate kernels.
        if len(X) >= 3:
            self._l = np.std(X, axis=0) + 0.05
        K = self._matern52(X, X)
        n = len(y)
        K += self.noise * np.eye(n)
        self._K    = K
        self._X_n  = X
        self._y    = y
        try:
            self._L = np.linalg.cholesky(K)
            self._alpha_vec = np.linalg.solve(
                self._L.T, np.linalg.solve(self._L, y))
        except np.linalg.LinAlgError:
            # Fallback: add more jitter
            self._L = np.linalg.cholesky(K + 1e-3*np.eye(n))
            self._alpha_vec = np.linalg.solve(
                self._L.T, np.linalg.solve(self._L, y))

    def _predict(self, X_new: np.ndarray):
        Xn = self._normalise(X_new)
        Ks = self._matern52(Xn, self._X_n)          # (m, n)
        mu = Ks @ self._alpha_vec
        v  = np.linalg.solve(self._L, Ks.T)
        var = (self._sigma_f**2 -
               np.sum(v**2, axis=0) + self.noise)
        var = np.maximum(var, 1e-10)
        return mu, np.sqrt(var)

    def _ei(self, X_new: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Expected Improvement."""
        from scipy.stats import norm
        mu, sigma = self._predict(X_new)
        y_best = np.min(self._y)
        Z = (y_best - mu - xi) / sigma
        ei = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))
        return ei

    def add_observation(self, x: np.ndarray, y: float):
        self.X_obs.append(x.copy())
        self.y_obs.append(float(y))
        if len(self.X_obs) >= 2:
            self._fit()

    def suggest(self, n_restarts: int = 10) -> np.ndarray:
        """Return next candidate point (maximise EI)."""
        if len(self.X_obs) < 2:
            # Latin Hypercube-ish random before we have data
            lo = self.bounds[:, 0]; hi = self.bounds[:, 1]
            return lo + np.random.rand(len(lo)) * (hi - lo)

        best_ei = -np.inf
        best_x  = None
        lo = self.bounds[:, 0]; hi = self.bounds[:, 1]

        for _ in range(n_restarts):
            x0 = lo + np.random.rand(len(lo)) * (hi - lo)
            res = minimize(
                lambda x: -float(self._ei(x[None, :])[0]),
                x0,
                method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 200, "ftol": 1e-9},
            )
            if res.fun < -best_ei or best_x is None:
                best_ei = -res.fun
                best_x  = res.x

        return best_x


# =============================================================================
#  Main tuner
# =============================================================================

class SMCTuner:

    def __init__(self,
                 omega_cmd_rpm:     float = 400.0,
                 t_sim:             float = 2.0,
                 dt:                float = 1e-4,
                 de_iters:          int   = 80,
                 gp_iters:          int   = 40,
                 workers:           int   = 1,
                 out_file:          str   = "smc_best_gains.json",
                 gains_config_path: str   = ""):
        """
        gains_config_path : path to write smc_gains_config.h after tuning.
                            Empty string (default) → auto-discovered.
                            None → suppressed.
        """
        self.omega_cmd_rpm     = omega_cmd_rpm
        self.t_sim             = t_sim
        self.dt                = dt
        self.de_iters          = de_iters
        self.gp_iters          = gp_iters
        self.workers           = workers
        self.out_file          = out_file
        self.gains_config_path = gains_config_path

        self._init_gains   = SMCGains()
        self._history: List[Dict] = []
        self._best_gains: Optional[SMCGains] = None
        self._best_cost:  float = 1e9

        self._eval_count   = 0
        self._t_start      = time.time()

    def _cost(self, x: np.ndarray) -> float:
        J = cost_function(x,
                          self.omega_cmd_rpm, self.t_sim, self.dt)
        self._eval_count += 1
        elapsed = time.time() - self._t_start

        if J < self._best_cost:
            self._best_cost   = J
            self._best_gains  = SMCGains.from_array(x)
            log.info("  ★ NEW BEST  eval=%3d  J=%.5f  t=%.0fs  │  %s",
                     self._eval_count, J, elapsed, self._best_gains)
        elif self._eval_count % 10 == 0:
            log.info("  · eval=%3d  J=%.5f  best=%.5f  t=%.0fs",
                     self._eval_count, J, self._best_cost, elapsed)

        self._history.append({
            "eval":    self._eval_count,
            "J":       J,
            "best_J":  self._best_cost,
            "elapsed": round(elapsed, 2),
            "x":       x.tolist(),
        })
        return J

    # ── Phase 1: Differential Evolution ──────────────────────────────────────
    def _phase1_de(self) -> SMCGains:
        popsize  = 8                              # 8×5=40 individuals
        n_total  = popsize * 5 * self.de_iters   # rough upper bound
        log.info("=" * 60)
        log.info("  Phase 1 — Differential Evolution  (global search)")
        log.info("  popsize=%d (×5 params = %d individuals)  maxiter=%d",
                 popsize, popsize*5, self.de_iters)
        log.info("  updating=immediate  workers=1  (NumPy 2.0 safe)")
        log.info("=" * 60)

        self._de_gen       = 0
        self._de_gen_best  = 1e9
        self._de_t_gen     = time.time()

        def _callback(xk, convergence=0.0):
            """Called once per generation by scipy DE."""
            self._de_gen += 1
            elapsed = time.time() - self._t_start
            # Build a compact progress bar (50 chars wide)
            frac    = min(self._de_gen / self.de_iters, 1.0)
            filled  = int(frac * 40)
            bar     = "█" * filled + "░" * (40 - filled)
            pct     = frac * 100.0
            gen_dt  = time.time() - self._de_t_gen
            self._de_t_gen = time.time()
            eta_s   = gen_dt * max(0, self.de_iters - self._de_gen)
            log.info(
                "  DE gen %3d/%d  [%s] %5.1f%%  "
                "best_J=%.5f  eval=%d  elapsed=%.0fs  ETA=%.0fs",
                self._de_gen, self.de_iters,
                bar, pct,
                self._best_cost,
                self._eval_count,
                elapsed, eta_s,
            )
            return False   # returning True stops DE early

        bounds = SMCGains().BOUNDS
        result = differential_evolution(
            self._cost,
            bounds,
            maxiter      = self.de_iters,
            popsize      = popsize,
            tol          = 1e-3,           # looser — GP refines later
            mutation     = (0.5, 1.2),
            recombination= 0.75,
            seed         = 42,
            updating     = "immediate",    # greedy: use each improvement now
            workers      = 1,             # single worker — avoids multiprocessing crash
            disp         = False,
            polish       = False,          # GP polishing is better
            callback     = _callback,
        )
        gains = SMCGains.from_array(result.x)
        convergence_msg = result.message if hasattr(result, "message") else "done"
        log.info("Phase 1 done  ->  J=%.5f  convergence=%s  %s",
                 float(result.fun), convergence_msg, gains)
        return gains

    # ── Phase 2: GP Bayesian refinement ──────────────────────────────────────
    def _phase2_gp(self, seed_gains: SMCGains) -> SMCGains:
        log.info("=" * 60)
        log.info("  Phase 2 — Gaussian Process Bayesian refinement")
        log.info("  n_calls=%d  seed: %s", self.gp_iters, seed_gains)
        log.info("=" * 60)

        bounds = SMCGains().BOUNDS
        gp = GaussianProcessBO(bounds)

        # Seed with DE result + small neighbourhood
        np.random.seed(0)
        x_seed   = seed_gains.to_array()
        n_seed   = 6
        n_total  = n_seed + self.gp_iters
        gp_iter  = [0]

        def _gp_eval(x_try: np.ndarray, label: str = "") -> float:
            J = self._cost(x_try)
            gp_iter[0] += 1
            frac   = min(gp_iter[0] / n_total, 1.0)
            filled = int(frac * 40)
            bar    = "█" * filled + "░" * (40 - filled)
            elapsed = time.time() - self._t_start
            log.info(
                "  GP  %3d/%d  [%s] %5.1f%%  "
                "J=%.5f  best_J=%.5f  elapsed=%.0fs  %s",
                gp_iter[0], n_total, bar, frac*100,
                J, self._best_cost, elapsed, label,
            )
            return J

        # Seed neighbourhood
        for i in range(5):
            noise = np.array([
                np.random.uniform(-0.1, 0.1) * (b[1]-b[0])
                for b in bounds])
            x_try = np.clip(x_seed + noise,
                            [b[0] for b in bounds],
                            [b[1] for b in bounds])
            J = _gp_eval(x_try, label=f"seed-noise-{i+1}")
            gp.add_observation(x_try, J)

        J_seed = _gp_eval(x_seed, label="seed-DE")
        gp.add_observation(x_seed, J_seed)

        for i in range(self.gp_iters):
            x_next = gp.suggest(n_restarts=8)
            J_next = _gp_eval(x_next, label=f"GP-EI-{i+1}")
            gp.add_observation(x_next, J_next)

        best_x = gp.X_obs[int(np.argmin(gp.y_obs))]
        gains  = SMCGains.from_array(np.array(best_x))
        log.info("Phase 2 done  →  J=%.5f  %s", min(gp.y_obs), gains)
        return gains

    # ── Run full tuning ───────────────────────────────────────────────────────
    def run(self) -> SMCGains:
        log.info("SMC Gain Tuner — DB42S02  @ %.0f RPM  t_sim=%.1f s",
                 self.omega_cmd_rpm, self.t_sim)

        de_gains = self._phase1_de()
        gp_gains = self._phase2_gp(de_gains)

        best = self._best_gains or gp_gains
        self._save(best)
        self._print_summary(best)
        self._emit_gains_config(best)
        return best

    def _emit_gains_config(self, gains: SMCGains) -> None:
        if self.gains_config_path is None:
            return
        out_path = self.gains_config_path
        if not out_path:
            try:
                from _path_utils import get_project_root
                out_path = str(
                    get_project_root() / "fs_electrical_machines" /
                    "c_src" / _GAINS_CONFIG_FILENAME)
            except Exception:
                log.warning("_emit_gains_config: could not auto-discover path.")
                return
        emit_gains_config(gains, out_path, backup=True)

    def _save(self, gains: SMCGains):
        out = {
            "gains": {
                "SMC_KS_W":  gains.SMC_KS_W,
                "SMC_ETA_W": gains.SMC_ETA_W,
                "SMC_PHI_W": gains.SMC_PHI_W,
                "SMC_KS_I":  gains.SMC_KS_I,
                "SMC_PHI_I": gains.SMC_PHI_I,
            },
            "cost":          self._best_cost,
            "evaluations":   self._eval_count,
            "omega_cmd_rpm": self.omega_cmd_rpm,
            "t_sim":         self.t_sim,
            "history":       self._history[-20:],
        }
        with open(self.out_file, "w") as f:
            json.dump(out, f, indent=2)
        log.info("Gains saved → %s", self.out_file)

    def _print_summary(self, gains: SMCGains):
        log.info("")
        log.info("╔══════════════════════════════════════════════════════╗")
        log.info("║           BEST SMC GAINS FOUND                       ║")
        log.info("╠══════════════════════════════════════════════════════╣")
        log.info("║  SMC_KS_W   = %10.6f  [N·m]  speed switching  ║", gains.SMC_KS_W)
        log.info("║  SMC_ETA_W  = %10.6f  [—]   speed damping     ║", gains.SMC_ETA_W)
        log.info("║  SMC_PHI_W  = %10.6f  [rad/s] speed BL        ║", gains.SMC_PHI_W)
        log.info("║  SMC_KS_I   = %10.6f  [V]   current switching ║", gains.SMC_KS_I)
        log.info("║  SMC_PHI_I  = %10.6f  [A]   current BL        ║", gains.SMC_PHI_I)
        log.info("╠══════════════════════════════════════════════════════╣")
        log.info("║  Best cost J  = %.6f                             ║", self._best_cost)
        log.info("║  Total evals  = %d                                  ║", self._eval_count)
        log.info("║  JSON output  : %-36s ║", self.out_file)
        log.info("╚══════════════════════════════════════════════════════╝")


# =============================================================================
#  smc_gains_config.h emitter  —  writes complete file from scratch, no regex
# =============================================================================

import re  # kept for any other uses in the file

# (field, macro, unit, description) — order matches SMC_GainSet_T
_GAIN_MACROS = [
    ("SMC_KS_W",  "SMC_KS_W",  "N·m",   "Speed SMC switching gain"),
    ("SMC_ETA_W", "SMC_ETA_W", "—",     "Speed SMC linear damping term"),
    ("SMC_PHI_W", "SMC_PHI_W", "rad/s", "Speed boundary layer thickness"),
    ("SMC_KS_I",  "SMC_KS_I",  "V",     "Current SMC switching gain"),
    ("SMC_PHI_I", "SMC_PHI_I", "A",     "Current boundary layer thickness"),
]

_GAINS_CONFIG_FILENAME = "smc_gains_config.h"


def _gains_config_text(gains: SMCGains) -> str:
    """Render the complete content of smc_gains_config.h as a string."""
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    gain_values = {
        "SMC_KS_W":  gains.SMC_KS_W,
        "SMC_ETA_W": gains.SMC_ETA_W,
        "SMC_PHI_W": gains.SMC_PHI_W,
        "SMC_KS_I":  gains.SMC_KS_I,
        "SMC_PHI_I": gains.SMC_PHI_I,
    }

    L = [
        "/**********************************************************************************************************************",
        " * \\file      smc_gains_config.h",
        " * \\brief     SMC tunable gain defaults — NANOTEC DB42S02",
        " *",
        " * Auto-generated by smc_fmu_tuner.py.  DO NOT EDIT — re-run the tuner.",
        f" * Generated : {ts}",
        " *",
        " * THIS IS THE ONLY FILE WRITTEN BY smc_fmu_tuner.py.",
        " * embed_sim_smc_controller.h is never modified by the tuner.",
        " *",
        " * Gain update workflow",
        " * --------------------",
        " *   1. Run smc_fmu_tuner.py  — writes this file with optimised gains.",
        " *   2. Recompile and flash   — SMC_Controller_Init() loads these",
        " *      values into g_smc_gains at startup.",
        " *",
        " * Optional runtime update (no recompile needed):",
        " *   - UDE debugger               : write g_smc_gains members live",
        " *   - UART loader                : smc_uart_loader.py --port COM4",
        " *   - SMC_GainSet_SetFromSchedule(): call from application code",
        " *********************************************************************************************************************/",
        "",
        "#ifndef SMC_GAINS_CONFIG_H_",
        "#define SMC_GAINS_CONFIG_H_",
        "",
        '#include "embed_sim_matrix.h"   /* MatrixFloat (= real32_T) */',
        "",
        "/*********************************************************************************************************************/",
        "/*--------------------------- Tunable gain defaults (written by smc_fmu_tuner.py) ------------------------------------*/",
        "/*********************************************************************************************************************/",
        "",
    ]

    for field, macro, unit, desc in _GAIN_MACROS:
        val = gain_values[field]
        L.append(f"/** \\brief {desc} [{unit}] */")
        L.append(f"#define {macro:<12} ((MatrixFloat){val:.6f}f)")
        L.append("")

    L += [
        "#endif /* SMC_GAINS_CONFIG_H_ */",
        "",
    ]
    return "\n".join(L)


def emit_gains_config(gains: SMCGains,
                      output_path: str,
                      backup: bool = True) -> bool:
    """
    Write smc_gains_config.h to output_path with the given gains.

    Always writes the complete file from scratch — no regex, no in-place
    patching, no comment accumulation across runs.  Creates the file if
    it does not exist yet.

    Returns True on success, False on any error.
    """
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if backup and p.exists():
        bak = p.with_suffix(".h.bak")
        bak.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        log.info("emit_gains_config: backup → %s", bak)

    text = _gains_config_text(gains)
    p.write_text(text, encoding="utf-8")
    log.info("emit_gains_config: wrote %s  (%d bytes)", output_path, len(text))
    for field, macro, unit, _ in _GAIN_MACROS:
        val = getattr(gains, field)
        log.info("  %-12s = %.6f  [%s]", macro, val, unit)
    return True


# Keep patch_header as alias — db42s02_tune_simulate_codegen.py calls it
def patch_header(gains: SMCGains,
                 header_path: str,
                 backup: bool = True) -> bool:
    """Alias for emit_gains_config(). header_path must point to smc_gains_config.h."""
    return emit_gains_config(gains, header_path, backup=backup)



# =============================================================================
#  Verification plot  (run after tuning to visually confirm response)
# =============================================================================

def plot_verification(gains: SMCGains,
                      omega_cmd_rpm: float = 400.0,
                      t_sim: float         = 2.0,
                      dt: float            = 1e-4,
                      path: str            = "smc_tuner_verify.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = run_sim(gains,
                omega_cmd_rpm=omega_cmd_rpm,
                t_sim=t_sim, dt=dt)
    if d is None:
        log.error("Verification sim diverged — cannot plot.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f"SMC Tuner — Verification  "
        f"(cmd {omega_cmd_rpm:.0f} RPM | DB42S02 FMU)",
        fontsize=13, fontweight="bold")

    t = d["t"]

    axes[0].plot(t, d["omega_ref_rpm"], "k--", lw=1.2, label="ω_ref [RPM]")
    axes[0].plot(t, d["speed_rpm"],     "C0",  lw=1.5, label="ω_motor [RPM]")
    axes[0].set_ylabel("Speed [RPM]"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("Motor speed — tuned SMC")

    axes[1].plot(t, d["iq_meas"], "C1", lw=1.0, label="iq [A]")
    axes[1].plot(t, d["id_meas"], "C3", lw=1.0, label="id [A]")
    axes[1].axhline(0, color="gray", lw=0.8, ls="--")
    axes[1].set_ylabel("Current [A]"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("dq currents — id≈0 (MTPA), iq≈0 at no-load steady-state")

    axes[2].plot(t, d["T_em"], "C2", lw=1.0, label="T_em [N·m]")
    axes[2].axhline(0, color="gray", lw=0.8, ls="--")
    axes[2].set_ylabel("Torque [N·m]"); axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Time [s]")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("Electromagnetic torque")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Verification plot → %s", path)


# =============================================================================
#  CLI entry point
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Bayesian + DE SMC gain tuner for EmbedSim DB42S02 (Python plant)")
    p.add_argument("--header",    default="",
                   help="Path to smc_gains_config.h — patched in-place (NOT embed_sim_smc_controller.h)")
    p.add_argument("--rpm",       type=float, default=400.0,
                   help="Target speed [RPM]  (default 400)")
    p.add_argument("--t_sim",     type=float, default=1.0,
                   help="Simulation duration [s]  (default 1.0)")
    p.add_argument("--dt",        type=float, default=50e-6,
                   help="Time step [s]  (default 50e-6 = 20 kHz)")
    p.add_argument("--de_iters",  type=int,   default=50,
                   help="DE max iterations  (default 50)")
    p.add_argument("--gp_iters",  type=int,   default=40,
                   help="GP refinement calls  (default 40)")
    p.add_argument("--out",       default="smc_best_gains.json",
                   help="Output JSON file for best gains")
    p.add_argument("--verify",    action="store_true",
                   help="Plot verification after tuning")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip .bak backup when patching header")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tuner = SMCTuner(
        omega_cmd_rpm     = args.rpm,
        t_sim             = args.t_sim,
        dt                = args.dt,
        de_iters          = args.de_iters,
        gp_iters          = args.gp_iters,
        workers           = 1,
        out_file          = args.out,
        gains_config_path = args.header or "",  # "" → auto-discover in run()
    )

    best = tuner.run()
    # smc_gains_config.h emitted inside tuner.run() → _emit_gains_config()

    if args.verify:
        plot_verification(best,
                          omega_cmd_rpm=args.rpm,
                          t_sim=args.t_sim,
                          dt=args.dt)