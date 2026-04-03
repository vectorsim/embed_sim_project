# db42s02_closed_loop_smc_foc_20k.py
"""
db42s02_closed_loop_smc_foc_20k.py
===================================
EmbedSim  --  Closed-loop SMC FOC  --  NANOTEC DB42S02  --  AURIX TC3xx 20 kHz

Architecture (encoder-based FOC):
  theta_e  = p * theta_m           exact from encoder -> Park / InvPark
  omega_m  = delta_theta_m/dt+IIR  encoder speed      -> speed SMC only
  Speed SMC  -> iq_ref
  Current SMC -> vd, vq -> InvPark -> SVPWM -> ta,tb,tc -> AURIX GTM

Load schedule (simulation only):
  t < T_LOAD_T1   : no load         (0.000 N.m)
  T_LOAD_T1..T2   : light load      (0.005 N.m)
  t >= T_LOAD_T2  : full load       (0.020 N.m)

Tuner (optional - user is asked at startup):
  Latin-Hypercube search over 4 free SMC gains:
    KS_W  [A]     speed switching amplitude
    PHI_W [rad/s] speed boundary layer thickness
    KS_I  [V]     current switching gain (physical, pre-SVPWM)
    PHI_I [A]     current boundary layer thickness
  Physics-derived bounds - see _build_tuner_bounds().
  Best result written to embed_sim_smc_gains.h (MISRA C:2012).

Outputs:
  db42s02_smc_foc_20k_results.png   -- 2x3 plot (baseline + tuned overlay)
  embed_sim_smc_gains.h             -- updated C gain header (if tuned)
  embedsim_gen/embedsim_step.c/.h   -- AURIX code (from tuned or baseline gains)
  db42s02_smc_topology.html         -- block diagram

CodeGen  ->  embedsim_gen/embedsim_step.c / .h
"""

from __future__ import annotations

import sys
import math
import time
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as _animation
from matplotlib.gridspec import GridSpec as _GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from smc_controller_block import SMCControllerBlock, _DB42S02
from pmsm_python_plant import PMSM_Python_Plant


# =============================================================================
# Simulation constants
# =============================================================================

V_DC       = _DB42S02.SMC_V_DC          # [V]     DC bus voltage
TARGET_RPM = 2000.0                      # [RPM]   mechanical speed setpoint
T_SIM      = 5.0                         # [s]     simulation duration
DT         = 50e-6                       # [s]     sample period (20 kHz)
_RAMP_TIME = 0.5                         # [s]     speed ramp duration in CtrlPacker

# =============================================================================
# GPT12 encoder noise model
# =============================================================================
# Simulates the IfxGpt12_IncrEnc_getAbsolutePosition() turn-rollover race
# condition that causes "lost step" glitches on AURIX hardware.
#
# At each turn boundary (rawPosition wraps 3999->0), there is a probability
# ENC_GLITCH_PROB that rawPosition reads 0 before turn increments, producing
# a ~2*pi backward jump in reported theta_m for one step.  After the 2*pi
# unwrap in the speed estimator, this maps to delta=0 (lost step).
# At 2000 RPM: 2000/60 = 33.3 turn-rollovers/second = once per 600 steps.
#
# Set ENC_GLITCH_ENABLE = False for ideal simulation (no hardware noise).
# Set ENC_GLITCH_ENABLE = True  to reproduce AURIX hardware behaviour.
ENC_GLITCH_ENABLE = True           # enable GPT12 rollover glitch model
ENC_GLITCH_PROB   = 0.30           # probability of glitch at each rollover
ENC_RESOLUTION    = 4000           # encoder pulses/rev (x4 quadrature)

# Load-torque schedule breakpoints [s]
T_LOAD_T1    = 0.5                       # [s]  light load applied
T_LOAD_T2    = 1.2                       # [s]  full load applied

# Load-torque levels [N.m]
T_LOAD_ZERO  = 0.000                     # [N.m] no load
T_LOAD_LIGHT = 0.005                     # [N.m] light load
T_LOAD_HEAVY = 0.020                     # [N.m] full load

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0   # [rad/s]
_MOTOR_OUT_SIZE  = 8   # [rpm, ia, ib, ic, theta_m, T_em, id, iq]


# =============================================================================
# Baseline SMC gains  (design-point - matched to embed_sim_smc_gains.h)
# =============================================================================
# KS_W  [A]     Speed SMC switching amplitude.
#               KS_W >= T_load_max / KT = 0.020 / 0.0084 = 2.381 A (+30% margin).
SMC_KS_W_DEFAULT  = _DB42S02.SMC_KS_W        # [A]

# ETA_W [dim-less]  Speed SMC linear damping inside boundary layer.
#               Hard-capped at 0.01 in SMCControllerBlock.  Not a tuning target.
SMC_ETA_W_DEFAULT = _DB42S02.SMC_ETA_W       # [dim-less]

# PHI_W [rad/s] Speed SMC boundary layer thickness.
#               Sized so iq_ref = I_MAX/3 at max ramp error.
SMC_PHI_W_DEFAULT = _DB42S02.SMC_PHI_W       # [rad/s]

# KS_I  [V]     Current SMC switching gain (physical volts, pre-SVPWM).
#               Discrete pole at z=0.5: KS_I = PHI_I * L / (2 * dt).
SMC_KS_I_DEFAULT  = _DB42S02.SMC_KS_I        # [V]

# PHI_I [A]     Current SMC boundary layer thickness.
SMC_PHI_I_DEFAULT = _DB42S02.SMC_PHI_I       # [A]


# =============================================================================
# Tuner cost-function weights
# =============================================================================
# Each weight converts its metric to a dimensionless cost contribution.
# W_ID is largest to enforce MTPA (id = 0) as the primary constraint.

W_SS   = 2.0    # [dim-less]  steady-state speed error weight  (metric in RPM)
W_BUMP = 1.0    # [dim-less]  load-step speed bump weight       (metric in RPM)
W_ID   = 150.0  # [dim-less]  d-axis RMS current weight         (metric in A)
                #             High weight enforces MTPA (id=0) as hard constraint.
                #             At id=1A: contribution=150. At id=0.06A: contribution=9.
W_CHAT = 4.0    # [dim-less]  iq chattering weight              (metric in A, std)


# =============================================================================
# Tuner parameter
# =============================================================================
# Set _TUNER_ROUNDS to the number of simulations you want to run.
# Round 0 is always the baseline design-point gains.
# Rounds 1+ are random Latin-Hypercube samples over the physics-derived bounds.
# Each round is one full T_SIM-second closed-loop simulation (~30 s wall-clock).
# Press Ctrl+C between simulations to stop and keep the best result so far.

_TUNER_ROUNDS = 60   # [dim-less]  total number of tuning simulations


# =============================================================================
# Plant block
# =============================================================================

class DB42S02PlantBlock(PMSM_Python_Plant):
    """
    NANOTEC DB42S02 plant with scheduled load torque.

    Wraps PMSM_Python_Plant to inject the 5-element input bus
    [ta, tb, tc, V_DC, T_load] with T_load chosen from the three levels
    defined above based on simulation time t.

    Output bus (8 elements):
      [0] rpm      [RPM]   mechanical speed
      [1] ia       [A]     phase-A current
      [2] ib       [A]     phase-B current
      [3] ic       [A]     phase-C current
      [4] theta_m  [rad]   mechanical angle (accumulating, unwrapped)
      [5] T_em     [N.m]   electromagnetic torque
      [6] id       [A]     d-axis current
      [7] iq       [A]     q-axis current
    """

    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str, **kwargs):
        super().__init__(
            name      = name,
            R         = _DB42S02.SMC_R_S,
            L_d       = _DB42S02.SMC_L_D,
            L_q       = _DB42S02.SMC_L_Q,
            lambda_pm = _DB42S02.SMC_LAMBDA_PM,
            J         = _DB42S02.SMC_J_ROTOR,
            B_fric    = _DB42S02.SMC_B_FRICTION,
            p         = float(_DB42S02.SMC_P_POLES),
            v_dc      = V_DC,
        )

    def compute_py(self, t, dt, input_values=None):
        if   t < T_LOAD_T1: t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2: t_load = T_LOAD_LIGHT
        else:                t_load = T_LOAD_HEAVY

        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta_in, tb_in, tc_in = float(v[0]), float(v[1]), float(v[2])
                if ta_in != 0.0 or tb_in != 0.0 or tc_in != 0.0:
                    ta, tb, tc = ta_in, tb_in, tc_in

        aug = [VectorSignal(
            np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))]
        return super().compute_py(t, dt, aug)

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# CtrlPacker
# =============================================================================

class CtrlPacker(VectorBlock):
    """
    Packs motor feedback + speed reference into the SMC input bus.

    Input ports:
      [0] motor feedback bus   [rpm,ia,ib,ic,theta_m,T_em,id,iq]   8 elements
      [1] speed reference      [rad/s]   scalar (VectorStep output)

    Output bus (5 elements - SMC_Input_T):
      [0] omega_ref_mech  [rad/s]  ramp-filtered speed reference
      [1] theta_m         [rad]    mechanical angle from encoder
      [2] ia              [A]      phase-A current
      [3] ib              [A]      phase-B current
      [4] ic              [A]      phase-C current

    Speed ramp: limits domega/dt to TARGET_RADS_MECH / _RAMP_TIME [rad/s^2]
    so the motor starts gently without triggering the current limiter.
    """

    INPUT_NAMES       = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP        = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    # Unit / range annotations for EmbedSim_Input_T — picked up by StepGenerator.
    C_FIELD_COMMENTS = {
        "omega_ref_mech": "Mechanical speed reference [rad/s]; range [0, ~314] for 0-3000 RPM",
        "theta_m":        "Mechanical rotor angle [rad]; accumulating (NOT wrapped), from encoder",
        "ia":             "Phase-A current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
        "ib":             "Phase-B current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
        "ic":             "Phase-C current from ADC [A]; range [-SMC_I_MAX, +SMC_I_MAX]",
    }

    _RAMP_RATE = TARGET_RADS_MECH / _RAMP_TIME   # [rad/s^2]

    def __init__(self, name: str = "ctrl_packer", **kw):
        super().__init__(name, **kw)
        self.output_label           = "[w_ref,th_m,ia,ib,ic]"
        self._omega_ref_filt: float = 0.0   # [rad/s] ramped reference state
        self._enc_theta_prev: float = 0.0   # previous true theta_m for rollover detection
        self._enc_rng = np.random.default_rng(seed=42)  # reproducible noise

    def reset(self):
        super().reset()
        self._omega_ref_filt = 0.0
        self._enc_theta_prev = 0.0

    def compute_py(self, t, dt, input_values=None):
        m = (input_values[0].value
             if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value
             if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))

        omega_target = float(r[0]) if len(r) > 0 else 0.0
        max_step     = self._RAMP_RATE * dt
        self._omega_ref_filt += max(
            -max_step,
            min(max_step, omega_target - self._omega_ref_filt))

        # ── GPT12 encoder noise model ──────────────────────────────────────
        # Simulates turn-rollover race condition from IfxGpt12_IncrEnc_getAbsolutePosition.
        # At each mechanical turn boundary, rawPosition may read 0 before
        # turn increments -> reported theta_m jumps back by ~2*pi for one step.
        # After 2*pi unwrap in speed estimator: delta=0 (lost step).
        theta_m_true = float(m[4]) if len(m) > 4 else 0.0
        if ENC_GLITCH_ENABLE:
            # Detect turn boundary: theta_m_true crossed a 2*pi multiple
            turns_prev = int(self._enc_theta_prev / (2.0 * math.pi))
            turns_now  = int(theta_m_true        / (2.0 * math.pi))
            at_rollover = (turns_now > turns_prev) and (theta_m_true > 0.0)
            if at_rollover and self._enc_rng.random() < ENC_GLITCH_PROB:
                # Race condition: report position one turn behind (rawPos=0, turn not incremented)
                theta_m = theta_m_true - 2.0 * math.pi
            else:
                theta_m = theta_m_true
            self._enc_theta_prev = theta_m_true
        else:
            theta_m = theta_m_true
        # ───────────────────────────────────────────────────────────────────

        self.output = VectorSignal(np.array([
            self._omega_ref_filt,
            theta_m,
            float(m[1]) if len(m) > 1 else 0.0,
            float(m[2]) if len(m) > 2 else 0.0,
            float(m[3]) if len(m) > 3 else 0.0,
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Simulation runner - single run with given gains
# =============================================================================

def _run_sim(ks_w: float,
             eta_w: float,
             phi_w: float,
             ks_i: float,
             phi_i: float) -> dict | None:
    """
    Build and run one complete closed-loop simulation.

    Parameters
    ----------
    ks_w    : float [A]        Speed SMC switching amplitude.
    eta_w   : float [dim-less] Speed SMC linear damping (hard-capped at 0.01).
    phi_w   : float [rad/s]    Speed SMC boundary layer thickness.
    ks_i    : float [V]        Current SMC switching gain (physical volts).
    phi_i   : float [A]        Current SMC boundary layer thickness.

    Returns
    -------
    dict with keys:
      t             [s]        time vector (scope rate, ~20 kHz)
      speed_rpm     [RPM]      actual mechanical speed
      omega_ref_rpm [RPM]      speed reference (log rate, 1 kHz)
      iq_ref        [A]        q-axis current reference (1 kHz)
      iq            [A]        q-axis current measured  (1 kHz)
      id            [A]        d-axis current measured  (1 kHz)
      v_alpha       [V]        alpha-axis normalised voltage
      v_beta        [V]        beta-axis normalised voltage
      vref          [dim-less] SVPWM modulation index
      ta, tb, tc    [dim-less] PWM duties [0, 1]
      sector        [dim-less] SVPWM sector index
      torque        [N.m]      electromagnetic torque
      _cg_start               CodeGenStart block (for _run_codegen)
      _cg_end                 CodeGenEnd   block (for _run_codegen)
      _sim                    EmbedSim instance  (for topology export)
    None on simulation failure.
    """
    try:
        cg_start = CodeGenStart("cg_start")

        smc = SMCControllerBlock(
            "smc",
            SMC_V_DC      = V_DC,
            SMC_KS_W      = ks_w,
            SMC_ETA_W     = eta_w,
            SMC_PHI_W     = phi_w,
            SMC_KS_I      = ks_i,
            SMC_PHI_I     = phi_i,
            SMC_SMO_K     = _DB42S02.SMC_SMO_K,
            SMC_SMO_FC    = _DB42S02.SMC_SMO_FC,
            dt_s          = DT,
            use_c_backend = False,  # Changed from True to False (C backend not available)
            integrator    = "tustin",
        )

        svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
        svpwm      = SVPWMBlock("svpwm", use_c_backend=False)
        cg_end     = CodeGenEnd("cg_end")

        speed_ref   = VectorStep("speed_ref", step_time=0.0,
                                 before_value=TARGET_RADS_MECH,
                                 after_value=TARGET_RADS_MECH)
        load_torque = VectorConstant("load_torque", value=T_LOAD_ZERO)
        motor       = DB42S02PlantBlock("motor")
        motor_delay = VectorDelay("motor_delay",
                                  initial=[0.0] * _MOTOR_OUT_SIZE)
        ctrl        = CtrlPacker("ctrl_packer")
        sink        = VectorEnd("sink")
        sink_cg     = VectorEnd("sink_cg")

        cg_start >> smc >> svpwm_pack >> svpwm >> cg_end
        motor >> motor_delay >> ctrl
        speed_ref   >> ctrl
        ctrl        >> cg_start
        cg_end      >> motor
        load_torque >> motor
        motor       >> sink
        cg_end      >> sink_cg

        sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT,
                       solver=ODESolver.EULER)

        sim.scope.add(smc,        indices=[0, 1],                 label="Vab")
        sim.scope.add(svpwm_pack, indices=[0],                    label="Vref")
        sim.scope.add(svpwm,      indices=[0, 1, 2, 3],           label="Duties")
        sim.scope.add(motor,      indices=[0, 1, 2, 3, 5, 6, 7],  label="Motor")

        sim.run()

    except Exception as exc:
        print(f"  [sim error] {exc}")
        return None

    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = smc.log_data
    if len(t) < 100:
        return None

    def _s(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    def _i(key):
        if len(ld["t"]) > 1:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)

    def _m(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    return {
        "t":             t,
        "speed_rpm":     _m(0),
        "omega_ref_rpm": _i("speed_ref"),
        "iq_ref":        _i("iq_ref"),
        "iq":            _i("iq"),
        "id":            _i("id"),
        "v_alpha":       _s("Vab",    0),
        "v_beta":        _s("Vab",    1),
        "vref":          _s("Vref",   0),
        "ta":            _s("Duties", 0),
        "tb":            _s("Duties", 1),
        "tc":            _s("Duties", 2),
        "sector":        _s("Duties", 3),
        "torque":        _m(4),
        # CodeGen handles — reused by _run_codegen(), not re-simulated
        "_cg_start":     cg_start,
        "_cg_end":       cg_end,
        "_sim":          sim,
    }


# =============================================================================
# CodeGen — runs on an already-simulated result dict, no re-simulation
# =============================================================================

_WIRE_LABELS = {
    ("speed_ref",    "ctrl_packer"):  "w_ref [rad/s]",
    ("motor_delay",  "ctrl_packer"):  "[rpm,ia,ib,ic,th_m,Tem,id,iq] z-1",
    ("ctrl_packer",  "cg_start"):     "[w_ref,th_m,ia,ib,ic]",
    ("cg_start",     "smc"):          "[w_ref,th_m,ia,ib,ic]",
    ("smc",          "svpwm_pack"):   "[v_a,v_b]",
    ("svpwm_pack",   "svpwm"):        "[Vref,alpha,Vdc]",
    ("svpwm",        "cg_end"):       "[ta,tb,tc,sector]",
    ("cg_end",       "motor"):        "[ta,tb,tc,sector]",
    ("cg_end",       "sink_cg"):      "[ta,tb,tc,sector]",
    ("motor",        "motor_delay"):  "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
    ("motor",        "sink"):         "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
    ("load_torque",  "motor"):        "T_load [N.m]",
}


def _run_codegen(d: dict) -> None:
    """
    Generate AURIX C code and topology HTML from an already-simulated result.

    Uses the CodeGenStart / CodeGenEnd / EmbedSim objects stored inside *d*
    by _run_sim().  No simulation is run — the wired graph is reused as-is.

    Parameters
    ----------
    d : dict  Result dict returned by _run_sim() (must contain _cg_start,
              _cg_end, _sim keys).
    """
    cg_start = d.get("_cg_start")
    cg_end   = d.get("_cg_end")
    sim      = d.get("_sim")

    if cg_start is None or cg_end is None or sim is None:
        print("  [CodeGen] ERROR: simulation objects missing from result dict.")
        return

    print("\n[Topology]")
    sim.topo.print_console()
    sim.topo.export_html(str(_HERE / "db42s02_smc_topology.html"),
                         wire_labels=_WIRE_LABELS)

    print("\n[CodeGen] Generating AURIX C code ...")
    result = cg_end.generate_step(
        cg_start    = cg_start,
        output_dir  = _ROOT,
        dt_hz       = 1.0 / DT,
        prefix      = "EmbedSim",
        write_files = True,
    )
    if result:
        gen = _ROOT / "embedsim_gen"
        print(f"  {gen}/embedsim_step.h")
        print(f"  {gen}/embedsim_step.c")


# =============================================================================
# Cost metrics
# =============================================================================

def _cost_metrics(d: dict) -> dict | None:
    """
    Compute scalar cost and diagnostic sub-metrics from a simulation result.

    Parameters
    ----------
    d : dict  Simulation result from _run_sim().

    Returns
    -------
    dict with keys:
      cost     [dim-less]  total weighted cost (minimise)
      ss_err   [RPM]       mean absolute speed error in last 20% of T_SIM
      bump     [RPM]       mean load-step speed drop at T_LOAD_T1 and T_LOAD_T2
      id_rms   [A]         RMS d-axis current (MTPA penalty - should be 0)
      iq_chat  [A]         std of iq_ref in steady state (chattering indicator)
    None when the simulation is invalid (diverged or too short).
    """
    if d is None:
        return None

    t   = d["t"]
    rpm = d["speed_rpm"]
    idd = d["id"]
    iqr = d["iq_ref"]
    n   = len(t)

    if n < 200 or float(np.max(np.abs(rpm))) > TARGET_RPM * 3.0:
        return None

    # Hard guard: SVPWM saturation means the controller is clipping — invalid result.
    # vref >= 0.95 in steady state indicates permanent voltage limit operation.
    ss_vref = d["vref"][t > 0.80 * T_SIM]
    if len(ss_vref) > 0 and float(np.percentile(ss_vref, 90)) >= 0.93:
        return None

    # Hard guard: SS error > 200 RPM means the controller failed to track.
    ss = t > 0.80 * T_SIM
    if not np.any(ss):
        return None

    ref_ss = float(np.mean(d["omega_ref_rpm"][ss]))
    ss_err = float(np.mean(np.abs(rpm[ss] - ref_ss)))

    def _bump(t_step: float, window: float = 0.20) -> float:
        """
        Speed drop at a load step.

        Parameters
        ----------
        t_step : float [s]  Time of load application.
        window : float [s]  Window after step to search for minimum.

        Returns
        -------
        float [RPM]  Speed drop (>= 0).
        """
        pre  = rpm[t < t_step]
        post = rpm[(t >= t_step) & (t < t_step + window)]
        if len(pre) < 5 or len(post) < 5:
            return 0.0
        return max(0.0, float(np.mean(pre[-30:])) - float(np.min(post)))

    bump    = (_bump(T_LOAD_T1) + _bump(T_LOAD_T2)) * 0.5
    id_rms  = float(np.sqrt(np.mean(idd[ss] ** 2)))
    iq_chat = float(np.std(iqr[ss]))

    cost = (W_SS   * ss_err
            + W_BUMP * bump
            + W_ID   * id_rms
            + W_CHAT * iq_chat)

    return {
        "cost":    cost,
        "ss_err":  ss_err,
        "bump":    bump,
        "id_rms":  id_rms,
        "iq_chat": iq_chat,
    }


# =============================================================================
# Tuner bounds - physics-derived
# =============================================================================

def _build_tuner_bounds() -> list:
    """
    Return physics-derived search bounds for the 4 free gain parameters.

    Parameter  Units    Lower bound                 Upper bound
    ---------  ------   -----------                 -----------
    KS_W       A        0.7 * T_max/KT              3.5 * T_max/KT
    PHI_W      rad/s    150                         1000
    KS_I       V        R * I_MAX * 0.5             V_MAX * 0.5
    PHI_I      A        0.10                        1.50

    Returns
    -------
    list of (lo, hi) tuples in parameter order [KS_W, PHI_W, KS_I, PHI_I].
    """
    KT    = _DB42S02.SMC_KT           # [N.m/A]
    I_MAX = _DB42S02.SMC_I_MAX        # [A]
    V_MAX = _DB42S02.SMC_V_MAX        # [V]
    R     = _DB42S02.SMC_R_S          # [Ohm]

    return [
        (0.7 * T_LOAD_HEAVY / KT,   3.5 * T_LOAD_HEAVY / KT),   # KS_W  [A]
        (150.0,                      1000.0),                      # PHI_W [rad/s] (lower bound raised from 50 to 150)
        #  At T_LOAD_HEAVY the speed drop is ~75 RPM = 7.85 rad/s;
        #  PHI_W < 150 puts the controller in permanent bang-bang during normal load rejection, driving id large.
        (R * I_MAX * 0.5,            V_MAX * 0.5),                 # KS_I  [V]
        (0.10,                       1.50),                        # PHI_I [A]
    ]


# =============================================================================
# Run tuner  —  Latin-Hypercube random search, plain for-loop
# =============================================================================

def run_tuner() -> tuple:
    """
    Evaluate _TUNER_ROUNDS candidate gain sets and return the best.

    Round 0   : baseline design-point gains (always first).
    Round 1.. : Latin-Hypercube samples over physics-derived bounds.

    Ctrl+C is checked between simulations (not mid-sim — unavoidable).
    PyCharm users: click the red Stop button or press Ctrl+C in the
    Run console; the loop exits after the current simulation finishes.

    Returns
    -------
    tuple (ks_w, phi_w, ks_i, phi_i)  — best gains found.
    """
    bounds     = _build_tuner_bounds()
    lo         = np.array([b[0] for b in bounds])
    hi         = np.array([b[1] for b in bounds])
    rng        = np.random.default_rng(seed=42)

    best_cost   = 1e9
    best_params = [SMC_KS_W_DEFAULT, SMC_PHI_W_DEFAULT,
                   SMC_KS_I_DEFAULT,  SMC_PHI_I_DEFAULT]

    # Latin-Hypercube grid for rounds 1..N-1
    n_lhs  = max(1, _TUNER_ROUNDS - 1)
    cuts   = np.linspace(0.0, 1.0, n_lhs + 1)
    lhs    = np.zeros((n_lhs, len(bounds)))
    for d in range(len(bounds)):
        u = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(u)
        lhs[:, d] = lo[d] + u * (hi[d] - lo[d])

    print("\n" + "=" * 72)
    print("  SMC Gain Tuner  --  Latin-Hypercube Random Search")
    print("=" * 72)
    print(f"  Rounds : {_TUNER_ROUNDS}  (round 0 = baseline, 1..{_TUNER_ROUNDS-1} = LHS samples)")
    print(f"  Params : KS_W [A]  PHI_W [rad/s]  KS_I [V]  PHI_I [A]")
    print(f"  Bounds:")
    _bnames = ["KS_W  [A]    ", "PHI_W [rad/s]", "KS_I  [V]   ", "PHI_I [A]   "]
    for _bn, (_lo, _hi) in zip(_bnames, bounds):
        print(f"    {_bn}  [{_lo:.4f}, {_hi:.4f}]")
    print(f"  Cost weights: W_SS={W_SS}  W_BUMP={W_BUMP}  W_ID={W_ID}  W_CHAT={W_CHAT}")
    print(f"  Ctrl+C stops cleanly after the current simulation finishes.")
    print("=" * 72)

    t0 = time.perf_counter()

    for rnd in range(_TUNER_ROUNDS):

        # Build candidate
        if rnd == 0:
            ks_w  = SMC_KS_W_DEFAULT
            phi_w = SMC_PHI_W_DEFAULT
            ks_i  = SMC_KS_I_DEFAULT
            phi_i = SMC_PHI_I_DEFAULT
            label = "baseline"
        else:
            p     = lhs[rnd - 1]
            ks_w, phi_w, ks_i, phi_i = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            label = f"LHS-{rnd}"

        print(f"\n  [{rnd+1:2d}/{_TUNER_ROUNDS}] {label}"
              f"  KS_W={ks_w:.4f} A  PHI_W={phi_w:.2f} rad/s"  # Fixed: changed "r/s" to "rad/s"
              f"  KS_I={ks_i:.5f} V  PHI_I={phi_i:.3f} A",
              flush=True)

        try:
            d   = _run_sim(ks_w, SMC_ETA_W_DEFAULT, phi_w, ks_i, phi_i)
            met = _cost_metrics(d)
        except KeyboardInterrupt:
            print("\n  [Tuner] Interrupted — returning best so far.")
            break

        if met is None:
            print(f"  --> UNSTABLE")
            continue

        cost = met["cost"]
        star = ""
        if cost < best_cost:
            best_cost   = cost
            best_params = [ks_w, phi_w, ks_i, phi_i]
            star        = "  *** NEW BEST ***"

        print(f"  --> cost={cost:8.2f}  ss={met['ss_err']:6.2f} RPM"
              f"  bump={met['bump']:5.1f} RPM"
              f"  id={met['id_rms']:.3f} A"
              f"  chat={met['iq_chat']:.3f} A{star}")

        # PyCharm-safe Ctrl+C check — KeyboardInterrupt raised here between sims
        try:
            pass
        except KeyboardInterrupt:
            print("\n  [Tuner] Interrupted — returning best so far.")
            break

    elapsed = time.perf_counter() - t0
    ks_w, phi_w, ks_i, phi_i = best_params

    print("\n" + "=" * 72)
    print("  TUNING COMPLETE")
    print("=" * 72)
    print(f"  Elapsed : {elapsed:.1f} s")
    print()
    print(f"  {'Parameter':<18} {'Baseline':>12}  {'Tuned':>12}  {'Delta':>8}")
    print(f"  {'-'*54}")
    for label, base_val, tuned_val in [
        ("KS_W  [A]",     SMC_KS_W_DEFAULT,  ks_w),
        ("PHI_W [rad/s]", SMC_PHI_W_DEFAULT, phi_w),
        ("KS_I  [V]",     SMC_KS_I_DEFAULT,  ks_i),
        ("PHI_I [A]",     SMC_PHI_I_DEFAULT,  phi_i),
    ]:
        delta = (tuned_val - base_val) / (abs(base_val) + 1e-12) * 100.0
        sign  = "UP" if delta > 0.0 else "DN"
        print(f"  {label:<18} {base_val:>12.6f}  {tuned_val:>12.6f}  "
              f"{sign} {abs(delta):5.1f}%")
    print("=" * 72)

    return ks_w, phi_w, ks_i, phi_i


# =============================================================================
# Write embed_sim_smc_gains.h  (MISRA C:2012)
# =============================================================================

def write_smc_gains_h(ks_w: float,
                      phi_w: float,
                      ks_i: float,
                      phi_i: float,
                      path: Path | None = None) -> Path:
    """
    Write embed_sim_smc_gains.h with MISRA C:2012-compliant tuned values.

    MISRA compliance:
      Rule 7.2  : all float literals carry the 'f' suffix.
      Rule 20.10: no token-pasting macros.
      Rule 8.1  : all types explicit (MatrixFloat = real32_T).
    Derivation comments document the physics behind each value so that
    the gain file is self-contained for code review.

    Parameters
    ----------
    ks_w  : float       [A]      Speed SMC switching gain.
    phi_w : float       [rad/s]  Speed SMC boundary layer thickness.
    ks_i  : float       [V]      Current SMC switching gain (physical volts).
    phi_i : float       [A]      Current SMC boundary layer thickness.
    path  : Path | None          Destination file.
                                 Defaults to _FS_ELEC/c_src/embed_sim_smc_gains.h.

    Returns
    -------
    Path  Absolute path of the written file.
    """
    if path is None:
        path = _FS_ELEC / "c_src" / "embed_sim_smc_gains.h"

    KT    = _DB42S02.SMC_KT           # [N.m/A]
    L     = _DB42S02.SMC_L_D          # [H]   d-axis inductance
    slew  = ks_i * DT / L             # [A/step]
    z_pole = 1.0 - ks_i * DT / L     # discrete z-plane pole location
    bounds = _build_tuner_bounds()

    slew_ok = "< PHI_I --> no overshoot" if slew < phi_i else ">= PHI_I --> WARN overshoot"

    content = textwrap.dedent(f"""\
    /*******************************************************************************************************************
     * \\file      embed_sim_smc_gains.h
     * \\brief     SMC tunable gain defaults -- NANOTEC DB42S02
     *
     * Physics-derived / auto-tuned for AURIX TC3xx @ 20 kHz, {V_DC:.1f} V bus,
     * {T_LOAD_HEAVY*1e3:.0f} mN.m max load (T_LOAD_HEAVY).
     *
     * Architecture: encoder-based FOC + classical equivalent control (no SMO in loop).
     *   Speed loop : integral sliding surface  s = e + lambda*integral(e)
     *                iq_ref = KS_W*sat(s/PHI_W) + ETA_W*s
     *   Current loop: equivalent control (full plant ODE cancellation)
     *                ed_hat = R*id_meas - we*Lq*iq_meas
     *                eq_hat = R*iq_meas + we*(Ld*id_meas + lambda_pm)
     *                vd = ed_hat + KS_I*sat(s_d/PHI_I)
     *                vq = eq_hat + KS_I*sat(s_q/PHI_I)
     *
     * Gain derivation (tuned values):
     *   KS_W  >= T_load_max/KT = {T_LOAD_HEAVY:.3f}/{KT:.4f} = {T_LOAD_HEAVY/KT:.3f} A
     *           --> {ks_w:.6f} A  (x{ks_w/(T_LOAD_HEAVY/KT):.2f} margin)
     *   PHI_W   tuned by Latin-Hypercube search
     *           --> {phi_w:.6f} rad/s
     *   KS_I    discrete z-pole at z = {z_pole:.4f}
     *           Current slew = KS_I*dt/L = {slew:.4f} A/step  {slew_ok}
     *           --> {ks_i:.6f} V  (physical; divided by V_DC/2 = {V_DC/2:.1f} inside SMC_Controller_Step)
     *   PHI_I   current boundary layer
     *           --> {phi_i:.6f} A
     *
     * Written by db42s02_closed_loop_smc_foc_20k.py -- do not edit manually.
     * Recompile embed_sim_smc_controller.c after patching this file.
     *
     * MISRA C:2012 compliance:
     *   Rule 7.2  : all float literals carry the f suffix (no implicit double promotion).
     *   Rule 20.10: no token-pasting operators used.
     *   Rule 8.1  : all types explicit via MatrixFloat typedef (= real32_T).
     *******************************************************************************************************************/

    #ifndef EMBED_SIM_SMC_GAINS_H_
    #define EMBED_SIM_SMC_GAINS_H_

    #include "embed_sim_matrix.h"

    /** \\brief Speed SMC switching amplitude [A].
     *
     *  Minimum condition for load rejection (Utkin 1992 s5.3):
     *    KS_W >= T_load_max / KT = {T_LOAD_HEAVY:.3f} N.m / {KT:.4f} N.m/A = {T_LOAD_HEAVY/KT:.3f} A
     *  Tuned value: {ks_w:.6f} A  (x{ks_w/(T_LOAD_HEAVY/KT):.2f} margin).
     *
     *  Units  : A  (q-axis current amplitude)
     *  Range  : [{bounds[0][0]:.3f}, {bounds[0][1]:.3f}] A
     *  Tuned  : Latin-Hypercube search (cost = W_SS*ss_err + W_BUMP*bump + W_ID*id_rms + W_CHAT*chat) */
    #define SMC_KS_W     ((MatrixFloat){ks_w:.6f}f)

    /** \\brief Speed SMC linear damping inside the boundary layer [dimensionless].
     *
     *  Provides smooth proportional action for |s| < PHI_W.
     *  Hard-capped at 0.01 inside SMC_SpeedSMC() -- not a tuning target.
     *  Setting this above 0.01 has no effect.
     *
     *  Units  : dimensionless
     *  Range  : [0.001, 0.010] */
    #define SMC_ETA_W    ((MatrixFloat){SMC_ETA_W_DEFAULT:.6f}f)

    /** \\brief Speed SMC boundary layer thickness [rad/s].
     *
     *  Transition region between switching (bang-bang) and proportional control:
     *    |s| > PHI_W  -->  bang-bang: iq_ref = +/- KS_W
     *    |s| < PHI_W  -->  linear:    iq_ref = KS_W*(s/PHI_W) + ETA_W*s
     *  Larger PHI_W reduces chattering but widens the speed dead-band.
     *  Tuned value: {phi_w:.6f} rad/s.
     *
     *  Units  : rad/s  (mechanical speed error)
     *  Range  : [{bounds[1][0]:.1f}, {bounds[1][1]:.1f}] rad/s
     *  Tuned  : Latin-Hypercube search */
    #define SMC_PHI_W    ((MatrixFloat){phi_w:.6f}f)

    /** \\brief Current SMC switching gain [V] -- physical, before SVPWM normalisation.
     *
     *  SMC_Controller_Step() divides all output voltages by SMC_SVPWM_GAIN = V_DC/2 = {V_DC/2:.1f}
     *  before writing y->v_alpha / y->v_beta, so the SVPWM block receives a normalised
     *  reference in [-1, +1] and the plant sees the correct physical voltages.
     *
     *  Discrete pole placement (Krishnan PMSM Drives, Ch.4):
     *    z-pole = 1 - KS_I*dt/L = 1 - {ks_i:.6f}*{DT:.2e}/{L:.3e} = {z_pole:.4f}
     *  Current slew per sample:
     *    slew = KS_I*dt/L = {slew:.4f} A/step   ({slew_ok})
     *
     *  Units  : V  (physical phase voltage, pre-SVPWM)
     *  Range  : [{bounds[2][0]:.4f}, {bounds[2][1]:.4f}] V
     *  Tuned  : Latin-Hypercube search */
    #define SMC_KS_I     ((MatrixFloat){ks_i:.6f}f)

    /** \\brief Current SMC boundary layer thickness [A].
     *
     *  Controls smooth vs. switching behaviour of the d- and q-axis current loops.
     *  Stability condition: slew = KS_I*dt/L < PHI_I (no inter-sample overshoot).
     *    slew = {slew:.4f} A/step   PHI_I = {phi_i:.4f} A   --> {slew_ok}
     *
     *  Units  : A  (dq current error)
     *  Range  : [{bounds[3][0]:.3f}, {bounds[3][1]:.3f}] A
     *  Tuned  : Latin-Hypercube search */
    #define SMC_PHI_I    ((MatrixFloat){phi_i:.6f}f)

    #endif /* EMBED_SIM_SMC_GAINS_H_ */
    """)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[Gains] Written: {path}")
    return path


# =============================================================================
# Baseline entry point
# =============================================================================

def build_and_run() -> dict:
    """
    Run the baseline simulation with design-point gains.

    Returns
    -------
    dict  Simulation result (see _run_sim docstring for key reference).
    """
    print("=" * 68)
    print("  NANOTEC DB42S02  --  SMC FOC + SMO  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC} V  "
          f"dt={DT*1e6:.0f} us  T_sim={T_SIM} s")
    print(f"  KS_W={SMC_KS_W_DEFAULT:.4f} A  ETA_W={SMC_ETA_W_DEFAULT:.4f}  "
          f"PHI_W={SMC_PHI_W_DEFAULT:.2f} rad/s")
    print(f"  KS_I={SMC_KS_I_DEFAULT:.4f} V  PHI_I={SMC_PHI_I_DEFAULT:.3f} A")
    print(f"  SMO   : k={_DB42S02.SMC_SMO_K:.2f} V  "
          f"fc={_DB42S02.SMC_SMO_FC:.0f} Hz")
    print(f"  Load  : 0 -> {T_LOAD_LIGHT*1e3:.0f} mN.m @ t={T_LOAD_T1}s"
          f"  -> {T_LOAD_HEAVY*1e3:.0f} mN.m @ t={T_LOAD_T2}s")
    print("=" * 68)

    print("\nRunning baseline simulation ...")
    d = _run_sim(SMC_KS_W_DEFAULT, SMC_ETA_W_DEFAULT,
                 SMC_PHI_W_DEFAULT, SMC_KS_I_DEFAULT,
                 SMC_PHI_I_DEFAULT)
    if d is None:
        print("  ERROR: baseline simulation failed.")
        sys.exit(1)

    print(f"  Done -- final speed: {d['speed_rpm'][-1]:.1f} RPM  "
          f"({len(d['t'])} scope samples)")
    return d


# =============================================================================
# Plot
# =============================================================================

def plot_results(base: dict,
                 tuned: dict | None = None,
                 path: str = "db42s02_smc_foc_20k_results.png") -> None:
    """
    Produce a 3x2 diagnostic plot.

    Row 0 left  : Speed tracking  (ref + actual; tuned overlay when available)
    Row 0 right : Speed error     (actual - ref)
    Row 1 left  : dq currents     (iq_ref, iq, id; tuned overlay)
    Row 1 right : id only         (MTPA check: should be 0 A)
    Row 2 left  : PWM duties      (ta, tb, tc; baseline only - visual reference)
    Row 2 right : Electromagnetic torque vs. load levels

    Load-step markers (orange / red dotted verticals) are drawn on all axes
    that carry time on the x-axis.

    Parameters
    ----------
    base   : dict           Baseline simulation result from _run_sim().
    tuned  : dict | None    Tuned simulation result (optional overlay).
    path   : str            Output PNG file path.
    """
    # Plot the FINAL result only: tuned if available, otherwise baseline.
    # No overlays — one clean trace per panel.
    d     = tuned if tuned is not None else base
    t     = d["t"]
    lbl   = "tuned" if tuned is not None else "baseline"
    col   = "C1"   if tuned is not None else "C0"

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(
        f"NANOTEC DB42S02 -- SMC FOC | {TARGET_RPM:.0f} RPM | 20 kHz  [{lbl}]",
        fontsize=12, fontweight="bold")

    def _load_lines(ax):
        ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0, alpha=0.6,
                   label=f"+{T_LOAD_LIGHT*1e3:.0f} mN.m @ {T_LOAD_T1}s")
        ax.axvline(T_LOAD_T2, color="red",    ls=":", lw=1.0, alpha=0.6,
                   label=f"+{T_LOAD_HEAVY*1e3:.0f} mN.m @ {T_LOAD_T2}s")

    # Row 0 left -- Speed tracking
    ax = axes[0, 0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.2, label="ref")
    ax.plot(t, d["speed_rpm"],     col,   lw=1.4, label=lbl)
    _load_lines(ax)
    ax.set_ylabel("Speed [RPM]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"Speed tracking  [{lbl}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 0 right -- Speed error
    ax = axes[0, 1]
    ax.plot(t, d["speed_rpm"] - d["omega_ref_rpm"], col, lw=0.8, label=lbl)
    ax.axhline(0, color="k", lw=0.5)
    _load_lines(ax)
    ax.set_ylabel("Speed error [RPM]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"Speed error  (actual - ref)  [{lbl}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1 left -- dq currents
    ax = axes[1, 0]
    ax.plot(t, d["iq_ref"], "k--", lw=1.0, label="iq_ref")
    ax.plot(t, d["iq"],     col,   lw=0.9, label="iq")
    ax.plot(t, d["id"],     "C5",  lw=0.9, label="id")
    ax.axhline( _DB42S02.SMC_I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-_DB42S02.SMC_I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(0, color="gray", ls="--", lw=0.4)
    ax.set_ylabel("Current [A]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"dq currents  (MTPA: id_ref = 0 A)  [{lbl}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 1 right -- id only (MTPA check)
    ax = axes[1, 1]
    ax.plot(t, d["id"], "C5", lw=0.9, label="id")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("id [A]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"id  (MTPA target: 0 A)  [{lbl}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2 left -- PWM duties
    ax = axes[2, 0]
    ax.plot(t, d["ta"], "C3", lw=0.6, label="ta")
    ax.plot(t, d["tb"], "C2", lw=0.6, label="tb")
    ax.plot(t, d["tc"], "C1", lw=0.6, label="tc")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Duty [dimensionless]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"SVPWM duties  [{lbl}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Row 2 right -- Electromagnetic torque
    ax = axes[2, 1]
    ax.plot(t, d["torque"] * 1e3, col, lw=0.9, label=f"T_em ({lbl})")
    ax.axhline(T_LOAD_LIGHT * 1e3, color="orange", ls=":", lw=1.0, alpha=0.7,
               label=f"T_load light {T_LOAD_LIGHT*1e3:.0f} mN.m")
    ax.axhline(T_LOAD_HEAVY * 1e3, color="red",    ls=":", lw=1.0, alpha=0.7,
               label=f"T_load heavy {T_LOAD_HEAVY*1e3:.0f} mN.m")
    ax.set_ylabel("Torque [mN.m]")
    ax.set_xlabel("t [s]")
    ax.set_title(f"Electromagnetic torque  [{lbl}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Performance summary
# =============================================================================

def print_summary(label: str, d: dict) -> None:
    """
    Print key performance metrics for one simulation result.

    Parameters
    ----------
    label : str   Short descriptor, e.g. 'Baseline' or 'Tuned'.
    d     : dict  Simulation result from _run_sim().
    """
    t   = d["t"]
    rpm = d["speed_rpm"]
    ref = d["omega_ref_rpm"]
    iq  = d["iq"]
    n   = len(t)
    ss  = int(0.80 * n)

    ss_err  = float(np.mean(np.abs(rpm[ss:] - ref[ss:])))
    vref_mx = float(np.max(d["vref"]))
    iq_ss   = float(np.mean(np.abs(iq[ss:])))

    after = t > T_LOAD_T2
    if np.any(after):
        pre     = t < T_LOAD_T2
        spd_pre = float(np.mean(rpm[pre][-50:])) if np.any(pre) else 0.0
        drop    = max(0.0, spd_pre - float(np.mean(rpm[after][:50])))
    else:
        drop = 0.0

    print(f"\n  [{label}]")
    print(f"    Final speed  : {rpm[-1]:8.1f} RPM  (target {TARGET_RPM:.0f})")
    print(f"    SS error     : {ss_err:8.2f} RPM  (last 20% of T_sim)")
    print(f"    Load drop    : {drop:8.1f} RPM  at t={T_LOAD_T2}s")
    print(f"    iq SS mean   : {iq_ss:8.3f} A   "
          f"(expect {T_LOAD_HEAVY/_DB42S02.SMC_KT:.2f} A at full load)")
    print(f"    Vref max     : {vref_mx:8.3f}   (clip 0.95)")


# =============================================================================
# Animation — (id, iq, RPM) manifold  +  s(t)  +  RPM(t)  synchronized
# =============================================================================

_ANIM_R_S  = _DB42S02.SMC_R_S
_ANIM_L_D  = _DB42S02.SMC_L_D
_ANIM_LAM  = _DB42S02.SMC_LAMBDA_PM
_ANIM_P    = int(_DB42S02.SMC_P_POLES)

_C_REACH  = "#ff2222"
_C_SLIDE  = "#ffdd00"
_C_LOCK   = "#22dd22"
_C_TRANS  = "#2288ff"
_S_REACH  = 80.0
_S_SLIDE  = 15.0
_ID_TRANS = 0.4


def _anim_phase_colours(s_arr: np.ndarray, id_arr: np.ndarray) -> list:
    out = []
    for s, id_v in zip(s_arr, id_arr):
        a = abs(s)
        if   a > _S_REACH:          out.append(_C_REACH)
        elif a > _S_SLIDE:          out.append(_C_SLIDE)
        elif abs(id_v) > _ID_TRANS: out.append(_C_TRANS)
        else:                        out.append(_C_LOCK)
    return out


def _build_physics_manifold(id_grid: np.ndarray,
                             iq_grid: np.ndarray) -> tuple:
    """Steady-state PMSM speed surface: v_q = R·iq + ωe·(λ_pm + Ld·id)."""
    ID, IQ    = np.meshgrid(id_grid, iq_grid)
    V_ph      = V_DC / math.sqrt(3.0)
    lam_eff   = np.where(_ANIM_LAM + _ANIM_L_D * ID < 1e-6,
                         1e-6, _ANIM_LAM + _ANIM_L_D * ID)
    v_q_avail = np.maximum(V_ph - _ANIM_R_S * IQ, 0.0)
    RPM_grid  = np.clip(v_q_avail / lam_eff / _ANIM_P * 60.0 / (2.0 * math.pi),
                        0.0, TARGET_RPM * 1.35)
    return ID, IQ, RPM_grid


def _make_anim_dict(d: dict) -> dict:
    """Convert _run_sim() result to slim animation format {t,rpm,ref,id,iq,s}."""
    t   = np.asarray(d["t"],             dtype=np.float64)
    rpm = np.asarray(d["speed_rpm"],     dtype=np.float64)
    ref = np.asarray(d["omega_ref_rpm"], dtype=np.float64)
    lam = float(_DB42S02.SMC_LAMBDA_W)
    err = (ref - rpm) * (2.0 * math.pi / 60.0)
    ie  = np.zeros_like(err)
    for k in range(1, len(err)):
        ie[k] = ie[k-1] + 0.5 * (t[k] - t[k-1]) * (err[k] + err[k-1])
    return {
        "t":   t,
        "rpm": rpm,
        "ref": ref,
        "id":  np.asarray(d["id"], dtype=np.float64),
        "iq":  np.asarray(d["iq"], dtype=np.float64),
        "s":   err + lam * ie,
    }


def make_sync_animation(base_data: dict,
                        tuned_data: dict | None,
                        gif_path: str = "db42s02_smc_sync_animation.gif",
                        png_path: str = "db42s02_smc_sync_animation.png") -> None:
    """
    Three-panel synchronized animation saved as GIF + PNG.
      Left   : (id, iq, RPM) manifold with dynamic trajectory
      Centre : Sliding variable s(t)
      Right  : Speed RPM(t)
    """
    base  = _make_anim_dict(base_data)
    tuned = _make_anim_dict(tuned_data if tuned_data is not None else base_data)

    N        = 500
    t_common = np.linspace(tuned["t"][0], tuned["t"][-1], N)

    id_sync  = np.interp(t_common, tuned["t"], tuned["id"])
    iq_sync  = np.interp(t_common, tuned["t"], tuned["iq"])
    rpm_sync = np.interp(t_common, tuned["t"], tuned["rpm"])
    s_sync   = np.interp(t_common, tuned["t"], tuned["s"])
    ref_sync = np.interp(t_common, tuned["t"], tuned["ref"])

    base_rpm_sync = np.interp(t_common, base["t"], base["rpm"])
    base_s_sync   = np.interp(t_common, base["t"], base["s"])

    cols_all = _anim_phase_colours(tuned["s"], tuned["id"])
    idx_map  = np.searchsorted(tuned["t"], t_common).clip(0, len(cols_all) - 1)
    col_sync = [cols_all[i] for i in idx_map]

    actual_max_rpm  = max(float(np.max(tuned["rpm"])), float(np.max(base["rpm"])))
    target_rpm_max  = min(max(TARGET_RPM * 1.2, actual_max_rpm * 1.05), TARGET_RPM * 1.3)
    rpm_disp        = np.clip(rpm_sync,      0.0, target_rpm_max)
    base_rpm_clipped= np.clip(base_rpm_sync, 0.0, target_rpm_max)

    t0, t1  = t_common[0], t_common[-1]
    s_lim   = np.percentile(np.abs(tuned["s"]), 99) * 1.2
    phi_w   = float(_DB42S02.SMC_PHI_W)
    id_pad  = max(2.0, float(np.max(np.abs(np.concatenate([tuned["id"], base["id"]])))) * 1.5)
    iq_pad  = max(2.5, float(np.max(np.concatenate([tuned["iq"], base["iq"]]))) * 1.3)
    iq_min  = max(-1.0, -iq_pad * 0.2)

    fig = plt.figure(figsize=(26, 11), facecolor="#080808")
    suffix = " [baseline + tuned]" if tuned_data is not None else " [baseline]"
    fig.suptitle(f"SMC FOC | NANOTEC DB42S02 | 20 kHz | Target: {TARGET_RPM:.0f} RPM{suffix}",
                 color="white", fontsize=14, fontweight="bold", y=0.98)

    gs   = _GridSpec(1, 3, left=0.03, right=0.98, bottom=0.08, top=0.94,
                     wspace=0.25, width_ratios=[1.4, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[0], projection="3d", facecolor="#0c0c0c")
    ax_s = fig.add_subplot(gs[1], facecolor="#0f0f0f")
    ax_sp= fig.add_subplot(gs[2], facecolor="#0f0f0f")

    def _style2d(ax, xl, yl, title):
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="#cccccc", labelsize=10)
        ax.xaxis.label.set_color("#ffffff"); ax.yaxis.label.set_color("#ffffff")
        for sp in ax.spines.values(): sp.set_edgecolor("#333333")
        ax.set_xlabel(xl, fontsize=11); ax.set_ylabel(yl, fontsize=11)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=5)
        ax.grid(alpha=0.15, color="#2a2a2a")

    _style2d(ax_s,  "Time [s]", "s  [rad/s]", "Sliding variable s(t)")
    _style2d(ax_sp, "Time [s]", "Speed [RPM]", f"Velocity (Target: {TARGET_RPM:.0f} RPM)")

    ax3d.set_facecolor("#0c0c0c")
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False; pane.set_edgecolor("#181818")
    ax3d.tick_params(colors="#cccccc", labelsize=9)
    for lb in (ax3d.xaxis.label, ax3d.yaxis.label, ax3d.zaxis.label):
        lb.set_color("#ffffff")
    ax3d.grid(alpha=0.08)

    id_grid = np.linspace(-id_pad, id_pad, 60)
    iq_grid = np.linspace(iq_min,  iq_pad, 60)
    ID, IQ, RPM_grid = _build_physics_manifold(id_grid, iq_grid)
    ax3d.plot_surface(ID, IQ, RPM_grid, cmap="inferno", alpha=0.30,
                      edgecolor="none", zorder=1, vmin=0, vmax=TARGET_RPM * 1.1)
    ax3d.contour(ID, IQ, RPM_grid, levels=[TARGET_RPM], zdir="z",
                 offset=TARGET_RPM, colors=["#ff4444"], linewidths=2.0, alpha=0.85, zorder=4)

    Xsm, Ysm = np.meshgrid(np.linspace(-id_pad, id_pad, 8),
                            np.linspace(iq_min, iq_pad, 8))
    ax3d.plot_surface(Xsm, Ysm, np.full_like(Xsm, TARGET_RPM),
                      color="#22ff88", alpha=0.10, edgecolor="#22ff8840",
                      linewidth=0.4, zorder=2)
    ax3d.text(-id_pad * 0.85, iq_pad * 0.85, TARGET_RPM * 1.02,
              "s = 0", color="#22ff88", fontsize=8, alpha=0.85, zorder=12)

    ax3d.plot(np.clip(np.interp(t_common, base["t"], base["id"]), -id_pad, id_pad),
              np.clip(np.interp(t_common, base["t"], base["iq"]),  iq_min, iq_pad),
              base_rpm_clipped, color="#888888", lw=1.6, alpha=0.55, zorder=3, label="baseline")

    _ss     = tuned["t"] > tuned["t"][-1] * 0.85
    _ss_rpm = float(np.mean(np.clip(tuned["rpm"][_ss], 0.0, target_rpm_max)))
    ax3d.scatter([float(np.mean(tuned["id"][_ss]))],
                 [float(np.mean(tuned["iq"][_ss]))],
                 [_ss_rpm], color="#00ffcc", s=180, marker="*", zorder=10,
                 edgecolors="white", linewidths=1.2, label=f"SS: {_ss_rpm:.0f} RPM")

    ax3d.set_xlabel("id [A]", fontsize=10, labelpad=5)
    ax3d.set_ylabel("iq [A]", fontsize=10, labelpad=5)
    ax3d.set_zlabel("RPM",    fontsize=10, labelpad=5)
    ax3d.set_xlim(-id_pad, id_pad); ax3d.set_ylim(iq_min, iq_pad)
    ax3d.set_zlim(0, target_rpm_max)
    ax3d.set_title("(id, iq, RPM) manifold", color="white", fontsize=11,
                   fontweight="bold", pad=6)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.legend(loc="lower left", fontsize=8, labelcolor="white",
                facecolor="#141414", edgecolor="#444444", framealpha=0.85)

    ax_s.axhline(0, color="#555555", lw=0.9, ls="--", zorder=1)
    ax_s.axhspan(-phi_w, phi_w, color="#ffaa0010", zorder=0)
    ax_s.axhline( phi_w, color="#ffaa00", lw=1.0, ls=":", alpha=0.55, zorder=2)
    ax_s.axhline(-phi_w, color="#ffaa00", lw=1.0, ls=":", alpha=0.55, zorder=2,
                 label=f"±PHI_W={phi_w:.0f}")
    ax_s.plot(t_common, base_s_sync, color="#44445a", lw=1.2, alpha=0.45,
              zorder=2, label="baseline s")
    for ax in (ax_s, ax_sp):
        ax.axvline(T_LOAD_T1, color="orange",  ls=":", lw=1.0, alpha=0.4, zorder=1)
        ax.axvline(T_LOAD_T2, color="#ff6666", ls=":", lw=1.0, alpha=0.4, zorder=1)
    ax_s.set_xlim(t0, t1); ax_s.set_ylim(-s_lim, s_lim)
    ax_s.legend(loc="upper right", fontsize=8, labelcolor="white",
                facecolor="#141414", edgecolor="#333333", framealpha=0.85)

    ax_sp.plot(t_common, base_rpm_clipped, color="#777777", lw=1.6,
               alpha=0.55, zorder=2, label="baseline")
    ax_sp.plot(t_common, ref_sync, color="white", lw=1.2, ls="--",
               alpha=0.5, zorder=2, label="ref")
    ax_sp.axhline(TARGET_RPM, color="#ff4444", ls="--", lw=1.5, alpha=0.6,
                  zorder=1, label=f"Target: {TARGET_RPM:.0f} RPM")
    _tol = TARGET_RPM * 0.01
    ax_sp.axhspan(TARGET_RPM - _tol, TARGET_RPM + _tol,
                  color="#44ff44", alpha=0.1, zorder=0)
    ax_sp.set_xlim(t0, t1); ax_sp.set_ylim(0, target_rpm_max)
    ax_sp.legend(loc="lower right", fontsize=8, labelcolor="white",
                 facecolor="#141414", edgecolor="#333333", framealpha=0.85)

    # Animated artists
    path_3d, = ax3d.plot([], [], [], lw=2.8, alpha=0.95, solid_capstyle="round", zorder=7)
    live_dot, = ax3d.plot([], [], [], "o", ms=10, zorder=9,
                          markeredgecolor="white", markeredgewidth=0.8)
    drop_ln,  = ax3d.plot([], [], [], lw=1.0, ls=":", color="#aaaaaa", alpha=0.6, zorder=3)
    s_ln,   = ax_s.plot([], [], lw=2.2, zorder=5)
    s_dot,  = ax_s.plot([], [], "o", ms=8, zorder=8,
                        markeredgecolor="white", markeredgewidth=0.6)
    sp_ln,  = ax_sp.plot([], [], lw=2.5, zorder=4)
    sp_dot, = ax_sp.plot([], [], "o", ms=8, zorder=5,
                         markeredgecolor="white", markeredgewidth=0.6)
    cur_s,  = ax_s.plot([], [], lw=1.8, color="white", ls="--", alpha=0.7, zorder=6)
    cur_sp, = ax_sp.plot([], [], lw=1.8, color="white", ls="--", alpha=0.7, zorder=6)
    phase_txt = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes,
                             fontsize=12, fontweight="bold", color=_C_LOCK, va="top")
    time_txt  = fig.text(0.50, 0.02, "", ha="center", fontsize=11, color="#aaaaaa")

    def _update(frame):
        col = col_sync[frame]
        ih  = id_sync[:frame+1];  iqh = iq_sync[:frame+1]
        rh  = rpm_disp[:frame+1]; th  = t_common[:frame+1]; sh = s_sync[:frame+1]
        path_3d.set_data(ih, iqh);       path_3d.set_3d_properties(rh); path_3d.set_color(col)
        live_dot.set_data([ih[-1]], [iqh[-1]]); live_dot.set_3d_properties([rh[-1]]); live_dot.set_color(col)
        drop_ln.set_data([ih[-1], ih[-1]], [iqh[-1], iqh[-1]]); drop_ln.set_3d_properties([rh[-1], 0.0])
        s_ln.set_data(th, sh);   s_ln.set_color(col)
        s_dot.set_data([th[-1]], [sh[-1]]); s_dot.set_color(col)
        sp_ln.set_data(th, rh);  sp_ln.set_color(col)
        sp_dot.set_data([th[-1]], [rh[-1]]); sp_dot.set_color(col)
        cur_s.set_data([th[-1], th[-1]], [-s_lim, s_lim])
        cur_sp.set_data([th[-1], th[-1]], [0, target_rpm_max])
        if   col == _C_LOCK:  lbl = f"LOCKED   {rh[-1]:.0f} / {TARGET_RPM:.0f} RPM"
        elif col == _C_SLIDE: lbl = f"SLIDING  s={sh[-1]:.1f}"
        elif col == _C_REACH: lbl = f"REACHING s={sh[-1]:.1f}"
        else:                  lbl = "TRANSIENT"
        phase_txt.set_text(lbl); phase_txt.set_color(col)
        time_txt.set_text(f"t = {th[-1]:.3f} s  |  RPM = {rh[-1]:.0f}  ({rh[-1]-TARGET_RPM:+3.0f})")
        return (path_3d, live_dot, drop_ln, s_ln, s_dot,
                sp_ln, sp_dot, cur_s, cur_sp, phase_txt, time_txt)

    print(f"  Building {N}-frame animation ...")
    ani = _animation.FuncAnimation(fig, _update, frames=N, interval=40, blit=False)
    print(f"  Saving {gif_path} ...")
    ani.save(gif_path, writer="pillow", fps=25, dpi=120)
    print(f"  Saved  {gif_path}")
    _update(N - 1)
    fig.savefig(png_path, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved  {png_path}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # 1. Baseline simulation (single run — data + graph objects kept in dict)
    base_data = build_and_run()

    # 2. Ask user
    print()
    try:
        answer = input("Run gain tuner (Latin-Hypercube search)?  [y/N] "
                       ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"

    do_tune = answer in ("y", "yes")

    # 3. Tune (optional)
    if do_tune:
        ks_w_best, phi_w_best, ks_i_best, phi_i_best = run_tuner()

        print("\nRunning tuned simulation ...")
        tuned_data = _run_sim(ks_w_best, SMC_ETA_W_DEFAULT,
                              phi_w_best, ks_i_best, phi_i_best)
        if tuned_data is None:
            print("  ERROR: tuned simulation failed -- CodeGen uses baseline graph.")
            do_tune    = False
            tuned_data = None
            write_smc_gains_h(SMC_KS_W_DEFAULT, SMC_PHI_W_DEFAULT,
                              SMC_KS_I_DEFAULT,  SMC_PHI_I_DEFAULT)
            _run_codegen(base_data)
        else:
            print(f"  Done -- final speed: {tuned_data['speed_rpm'][-1]:.1f} RPM")
            write_smc_gains_h(ks_w_best, phi_w_best, ks_i_best, phi_i_best)
            _run_codegen(tuned_data)
    else:
        tuned_data = None
        # No tuning — write baseline design-point gains so .h is always present
        write_smc_gains_h(SMC_KS_W_DEFAULT, SMC_PHI_W_DEFAULT,
                          SMC_KS_I_DEFAULT,  SMC_PHI_I_DEFAULT)
        _run_codegen(base_data)

    # 4. Plot
    plot_results(base_data, tuned_data)

    # 5. Summary
    print("\n" + "=" * 60)
    print("  SMC FOC -- Performance Summary")
    print("=" * 60)
    print_summary("Baseline", base_data)
    if do_tune and tuned_data is not None:
        print_summary("Tuned   ", tuned_data)

    # 6. Synchronized manifold animation (optional)
    print()
    try:
        ans_anim = input(
            "Build synchronized (id,iq,RPM) manifold animation?  [y/N] "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        ans_anim = "n"

    if ans_anim in ("y", "yes"):
        print("\n[Animation] Building synchronized manifold animation ...")
        make_sync_animation(base_data, tuned_data)
    else:
        print("  [Animation] Skipped.")

    print("\n[Done]")
    print("  db42s02_smc_foc_20k_results.png")
    print("  db42s02_smc_topology.html")
    print("  embedsim_gen/embedsim_step.c   <- flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")
    if do_tune:
        print("  embed_sim_smc_gains.h          <- TUNED gains, recompile")
    else:
        print("  embed_sim_smc_gains.h          <- baseline design-point gains")
    if ans_anim in ("y", "yes"):
        print("  db42s02_smc_sync_animation.gif")
        print("  db42s02_smc_sync_animation.png")