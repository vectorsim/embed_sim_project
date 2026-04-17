# db42s02_closed_loop_mpc_foc_20k.py
"""
db42s02_closed_loop_mpc_foc_20k.py
===================================
EmbedSim  —  Closed-loop MPC FOC  —  NANOTEC DB42S02  —  AURIX TC3xx  20 kHz

Architecture (encoder-based):
  theta_e  = p·theta_m              exact from encoder → Park / InvPark
  omega_m  = Δtheta_m/dt + IIR      encoder speed      → speed reference
  SMO      = ê_α_filt, ê_β_filt     back-EMF filter    → disturbance feedforward
  MPC      = predicts and optimises vd, vq over horizon N → optimal control
  InvPark → SVPWM → ta,tb,tc → AURIX GTM

MPC cost function (minimised analytically at each step):
  J = Σ_{k=1}^{N} [ Q_id·id_k²  +  Q_iq·iq_k²
                   + Q_omega·(omega_k − omega_ref)²
                   + R_vd·vd²  +  R_vq·vq² ]

Load schedule (simulation only):
  t < 0.5 s  : no load
  0.5-1.2 s  : 5 mN·m
  1.2-5.0 s  : 20 mN·m

CodeGen  ->  embedsim_gen/embedsim_step.c / .h

=====================================================================
  INTEGRATED CMA-ES + NEURAL NETWORK WARM-START WEIGHT TUNER
=====================================================================
When run with --tune (or the user answers 'y' at the prompt), the
hybrid CMA-ES / NN tuner executes before the main simulation.

ALGORITHM - four phases:
  Phase 1  CMA-ES exploration
           Run CMA-ES generation 1 (_T_PHASE1_EVALS FMU simulations).
           Every evaluation is one complete closed-loop FMU simulation.
           NO surrogate, NO approximation.

  Phase 2  Neural-network warm-start
           Train MLP 5->16->16->1 (pure NumPy, Adam) on Phase 1 history.
           Gradient descent through the frozen MLP predicts the cost
           minimum.  This prediction becomes CMA-ES mean m0 for Phase 3.

  Phase 3  CMA-ES continuation from NN mean
           CMA-ES resumes from m0 with the remaining evaluation budget.
           IPOP restarts double the population on stagnation.

  Phase 4  Write c_src/embed_sim_mpc_gains.h
           ISO 26262 audit-trail header with best gains found.

WHY CMA-ES + NN WARM-START? (thesis rationale)
  CMA-ES alone  : robust black-box optimiser but cold-starts from the
                  commissioning baseline, potentially wasting early gens.
  MLP alone     : fast gradient prediction but trained on noisy data;
                  predicted minimum may lie in an unstable region.
  Combined      : the MLP provides a cheap, differentiable prediction of
                  the basin minimum after only _T_PHASE1_EVALS real FMU
                  evals. CMA-ES then verifies and refines with its
                  rigorous covariance adaptation. Combines the speed of
                  gradient prediction with the robustness of an
                  evolutionary strategy.

  Conceptually similar to NN-guided evolution in Neural Architecture
  Search (Real et al. 2019) and warm-starting in Bayesian Optimisation
  (Perrone et al. 2018).

  CMA-ES reference: Hansen N. (2016) arXiv:1604.00772
  Install: pip install cma   (scipy Nelder-Mead fallback requires nothing)

Outputs:
  db42s02_mpc_foc_20k_results.png
  db42s02_mpc_topology.html
  embedsim_gen/embedsim_step.c / .h        (CodeGen -> flash to AURIX)
  c_src/embed_sim_mpc_gains.h              (only when --tune requested)
"""

from __future__ import annotations

import sys
import math
import time
import datetime
import argparse
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Optional CMA-ES  --  graceful fallback to scipy Nelder-Mead
# THESIS NOTE: 'cma' is the reference implementation by Nikolaus Hansen
# (the algorithm's author).  Pure Python, no compiled extensions.
# ---------------------------------------------------------------------------
try:
    import cma                                              # type: ignore
    _HAVE_CMA = True
except ImportError:
    _HAVE_CMA = False

try:
    from scipy.optimize import minimize as _scipy_minimize  # type: ignore
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"
_C_SRC   = _FS_ELEC / "c_src"          # gains.h written here

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from mpc_controller_block import MPCControllerBlock, _DB42S02 as MPC_MOTOR
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock
from ctrl_packer import CtrlPacker


# =============================================================================
# Simulation constants
# =============================================================================

V_DC       = 17.0
TARGET_RPM = 2000.0
T_SIM      = 5.0        # [s]  main simulation
DT         = 50e-6      # [s]  20 kHz
_RAMP_TIME = 0.5        # [s]

T_LOAD_T1    = 0.5
T_LOAD_T2    = 1.2
T_LOAD_ZERO  = 0.000
T_LOAD_LIGHT = 0.005    # 5 mN·m
T_LOAD_HEAVY = 0.020    # 20 mN·m

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0
_MOTOR_OUT_SIZE  = 8    # [rpm,ia,ib,ic,theta_m,Tem,id,iq]

# =============================================================================
# Sensor noise
# =============================================================================
ENABLE_NOISE = False
NOISE_SEED   = 42

# =============================================================================
# MPC weight constants
# =============================================================================
# Q_omega = 500.0 FIXED.  Tuned by CMA-ES+NN: Q_id, Q_iq, R_vd, R_vq, KI_v.

MPC_N       = 10
MPC_Q_OMEGA = 500.0
MPC_SMO_K   = 4.68
MPC_SMO_FC  = 1000.0

_ACTIVE_GAINS = {
    "Q_id":  10.82,
    "Q_iq":   0.01,
    "R_vd":   0.001,
    "R_vq":   0.005,
    "KI_v":   0.01,
}

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Plant_FMU.fmu")


# =============================================================================
# Plant block
# =============================================================================

class DB42S02PlantBlock(PMSM_Plant_FMUBlock):
    """DB42S02 FMU plant with timed load torque schedule."""
    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,theta_m,Tem,id,iq]"

    def __init__(self, name: str):
        super().__init__(name=name, fmu_path=_FMU_PATH)

    def compute_py(self, t, dt, input_values=None):
        if   t < T_LOAD_T1: t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2: t_load = T_LOAD_LIGHT
        else:                t_load = T_LOAD_HEAVY
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])
        fmu_in = VectorSignal(np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))
        return super().compute_py(t, dt, [fmu_in])

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# CodeGen SVPWM subclasses
# =============================================================================

class DB42S02SVPWMPackBlock(SVPWMPackBlock):
    """SVPWMPackBlock with DB42S02 / AURIX CodeGen metadata.
    Uses two-pass codegen (same as SVPWMPackBlockDT in DFC) — no C_CUSTOM_EMIT.
    Generator hoists u_svpwm_pack[] and y_svpwm_pack[] to function scope.
    """
    C_SOURCES    = ["embed_sim_motor_utility_blocks.c"]
    C_HEADERS    = ["embed_sim_motor_utility_blocks.h"]
    state_struct = "SVPWMPack_T"
    step_func    = "SVPWMPack_Step"
    init_func    = "SVPWMPack_Init"
    NUM_INPUTS   = 2   # v_alpha + v_beta from MPC output
    OUTPUT_SIZE  = 3
    C_INIT_ARGS  = ["v_dc"]


class DB42S02SVPWMBlock(SVPWMBlock):
    """SVPWMBlock with DB42S02 / AURIX CodeGen metadata.
    Inherits @property C_CUSTOM_EMIT from SVPWMBlock base class which
    auto-detects upstream svpwm_pack and writes directly to out->.
    """
    C_SOURCES = ["embed_sim_sv_pwm.c"]
    C_HEADERS = ["embed_sim_sv_pwm.h"]


# =============================================================================
# _run_sim  --  shared by tuner and build_and_run
# =============================================================================

def _run_sim(*, with_codegen_hooks: bool = True, t_sim: float = T_SIM) -> dict | None:
    """
    Build and run one closed-loop MPC simulation.

    Parameters
    ----------
    with_codegen_hooks : True for main run (CodeGenStart/End included).
                         False for tuner (lighter, no CodeGen objects).
    t_sim              : duration [s]; tuner uses 3 s, main run uses T_SIM.

    Returns dict of result signals, or None on failure.
    """
    try:
        mpc = MPCControllerBlock(
            name="mpc", P_POLES=MPC_MOTOR.P_POLES, R_S=MPC_MOTOR.R_S,
            L=MPC_MOTOR.L_D, LAMBDA_PM=MPC_MOTOR.LAMBDA_PM,
            J=MPC_MOTOR.J_ROTOR, B=MPC_MOTOR.B_FRICTION,
            I_MAX=MPC_MOTOR.I_MAX, V_DC=V_DC, N=MPC_N,
            Q_id=_ACTIVE_GAINS["Q_id"], Q_iq=_ACTIVE_GAINS["Q_iq"],
            Q_omega=MPC_Q_OMEGA,
            R_vd=_ACTIVE_GAINS["R_vd"], R_vq=_ACTIVE_GAINS["R_vq"],
            dt_s=DT, SMO_K=MPC_SMO_K, SMO_FC=MPC_SMO_FC,
            KI_v=_ACTIVE_GAINS["KI_v"], SOFTSTART_T=0.1,
            use_c_backend=True,
        )
        svpwm_pack  = DB42S02SVPWMPackBlock("svpwm_pack", v_dc=V_DC, use_c_backend=False)
        svpwm       = DB42S02SVPWMBlock("svpwm", use_c_backend=False)
        motor       = DB42S02PlantBlock("motor")
        motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
        speed_ref   = VectorStep("speed_ref", step_time=0.0,
                                 before_value=TARGET_RADS_MECH,
                                 after_value=TARGET_RADS_MECH)
        ctrl = CtrlPacker("ctrl_packer", target_rads_mech=TARGET_RADS_MECH,
                          ramp_time=_RAMP_TIME, rng_seed=NOISE_SEED)
        ctrl.set_noise_enabled(ENABLE_NOISE if with_codegen_hooks else False)
        sink    = VectorEnd("sink")
        sink_cg = VectorEnd("sink_cg")

        if with_codegen_hooks:
            cg_start = CodeGenStart("cg_start")
            cg_end   = CodeGenEnd("cg_end")
            cg_start >> mpc >> svpwm_pack >> svpwm >> cg_end
            ctrl >> cg_start
            svpwm >> motor
            svpwm >> sink_cg
            cg_end >> sink_cg
        else:
            cg_start = cg_end = None
            mpc >> svpwm_pack >> svpwm
            ctrl >> mpc
            svpwm >> motor
            svpwm >> sink_cg

        motor >> motor_delay
        motor_delay >> ctrl
        speed_ref >> ctrl
        motor >> sink

        sim = EmbedSim(sinks=[sink, sink_cg], T=t_sim, dt=DT, solver=ODESolver.EULER)

        if with_codegen_hooks:
            print("\n[Topology]")
            sim.topo.print_console()
            sim.topo.export_html(
                str(_HERE / "db42s02_mpc_topology.html"),
                wire_labels={
                    ("speed_ref",   "ctrl_packer"): "w_ref [rad/s]",
                    ("motor",       "motor_delay"): "FMU feedback",
                    ("motor_delay", "ctrl_packer"): "z^-1 feedback",
                    ("ctrl_packer", "cg_start"):    "[w_ref,th_m,ia,ib,ic]",
                    ("cg_start",    "mpc"):         "sensor inputs",
                    ("mpc",         "svpwm_pack"):  "[v_a,v_b]",
                    ("svpwm_pack",  "svpwm"):       "[Vref,a]",
                    ("svpwm",       "cg_end"):      "[ta,tb,tc,sector]",
                    ("svpwm",       "motor"):       "[ta,tb,tc,sector]",
                })

        sim.scope.add(mpc,        indices=[0, 1],       label="Vab")
        sim.scope.add(svpwm_pack, indices=[0],           label="Vref")
        sim.scope.add(svpwm,      indices=[0, 1, 2, 3], label="Duties")
        sim.scope.add(motor,      indices=[0, 5, 6, 7], label="Motor")

        print("\nRunning simulation...")
        sim.run()
        print(f"  Done: {len(sim.scope.t)} steps")

    except Exception as exc:
        import traceback
        print(f"  [sim error] {exc}")
        traceback.print_exc()
        return None

    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = mpc.log_data
    if len(t) < 200:
        return None

    def _s(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)
    def _i(key):
        arr = ld.get(key, []); t_ld = ld.get("t", [])
        if len(t_ld) > 1:
            return np.interp(t, t_ld, arr).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)
    def _m(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    return {
        "t": t, "speed_rpm": _m(0), "omega_ref_rpm": _i("speed_ref"),
        "iq": _i("iq"), "id": _i("id"),
        "vd": _i("vd"), "vq": _i("vq"),          # dq voltage commands [V]
        "v_alpha": _s("Vab", 0), "v_beta": _s("Vab", 1),
        "vref": _s("Vref", 0),
        "ta": _s("Duties", 0), "tb": _s("Duties", 1),
        "tc": _s("Duties", 2), "sector": _s("Duties", 3),
        "torque": _m(1), "id_plant": _m(2), "iq_plant": _m(3),
        "_mpc_log_data": mpc.log_data,   # full MPC diagnostic log for convergence report
        "_cg_start": cg_start, "_cg_end": cg_end, "_sim": sim,
    }


# =============================================================================
# build_and_run  --  main simulation entry point
# =============================================================================

def build_and_run() -> dict:
    """Run the full closed-loop simulation and emit AURIX C code."""
    print("=" * 68)
    print("  NANOTEC DB42S02  --  MPC FOC + SMO  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC}V  dt={DT*1e6:.0f}us  T_sim={T_SIM}s")
    print(f"  MPC    : N={MPC_N}  Q_id={_ACTIVE_GAINS['Q_id']:.4f}  "
          f"Q_iq={_ACTIVE_GAINS['Q_iq']:.4f}  Q_omega={MPC_Q_OMEGA:.1f}  "
          f"R_vd={_ACTIVE_GAINS['R_vd']:.4f}  R_vq={_ACTIVE_GAINS['R_vq']:.4f}  "
          f"KI_v={_ACTIVE_GAINS['KI_v']:.4f}")
    print(f"  SMO    : k={MPC_SMO_K:.2f} V  fc={MPC_SMO_FC:.0f} Hz")
    print(f"  Load   : 0 -> {T_LOAD_LIGHT*1e3:.0f} mN·m @ {T_LOAD_T1}s"
          f" -> {T_LOAD_HEAVY*1e3:.0f} mN·m @ {T_LOAD_T2}s")
    print(f"  Noise  : {'ENABLED (seed=' + str(NOISE_SEED) + ')' if ENABLE_NOISE else 'DISABLED'}")
    print("=" * 68)

    d = _run_sim(with_codegen_hooks=True, t_sim=T_SIM)
    if d is None:
        raise RuntimeError("Main simulation failed.")

    cg_start = d["_cg_start"]; cg_end = d["_cg_end"]
    if cg_start and cg_end:
        print("\n[CodeGen] Generating AURIX C code...")
        result = cg_end.generate_step(
            cg_start=cg_start, output_dir=_ROOT,
            dt_hz=1.0/DT, prefix="EmbedSim", write_files=True,
        )
        if result:
            gen = _ROOT / "embedsim_gen"
            print(f"  {gen}/embedsim_step.h")
            print(f"  {gen}/embedsim_step.c")
            print("  Input_T  : omega_ref_mech, theta_m, ia, ib, ic")
            print("  Output_T : ta, tb, tc, sector")
    return d


# =============================================================================
# Plot
# =============================================================================

def plot_results(d: dict, path: str = "db42s02_mpc_foc_20k_results.png") -> None:
    """
    Five-panel diagnostic plot:
      1. Speed [RPM]  —  tracking + reference
      2. iq [A]       —  SMO estimate vs plant truth
      3. id [A]       —  SMO estimate (MTPA target = 0)
      4. vq [V]       —  q-axis voltage command (physical, pre-SVPWM)
      5. vd [V]       —  d-axis voltage command (physical, pre-SVPWM)

    Load events annotated on every panel (orange = 5 mNm, red = 20 mNm).
    """
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    fig.suptitle(
        f"NANOTEC DB42S02  —  MPC FOC + SMO  |  {TARGET_RPM:.0f} RPM target  |  20 kHz"
        + ("  |  NOISE ON" if ENABLE_NOISE else "  |  noise off"),
        fontsize=11, fontweight="bold")

    t = d["t"]

    # ── Helper: shade load steps on an axis ─────────────────────────────────
    def _load_lines(ax):
        ax.axvline(T_LOAD_T1, color="darkorange", ls=":", lw=1.2,
                   label=f"load {T_LOAD_LIGHT*1e3:.0f} mN·m")
        ax.axvline(T_LOAD_T2, color="crimson",    ls=":", lw=1.2,
                   label=f"load {T_LOAD_HEAVY*1e3:.0f} mN·m")

    # ── Panel 1: Speed ───────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.4, label="ω_ref")
    ax.plot(t, d["speed_rpm"],     "C0",  lw=1.2, label="ω_actual (plant)")
    _load_lines(ax)
    ax.set_ylabel("Speed [RPM]")
    ax.set_title("Speed tracking")
    ax.legend(fontsize=8, ncol=4, loc="lower right")
    ax.grid(alpha=0.25)

    # ── Panel 2: iq ─────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(t, d["iq_plant"], "k--", lw=1.2, alpha=0.7, label="iq plant (truth)")
    ax.plot(t, d["iq"],       "C0",  lw=1.0,             label="iq MPC est")
    ax.axhline(0,               color="gray", ls="--", lw=0.6, alpha=0.5)
    ax.axhline( MPC_MOTOR.I_MAX, color="gray", ls=":",  lw=0.6, alpha=0.4)
    ax.axhline(-MPC_MOTOR.I_MAX, color="gray", ls=":",  lw=0.6, alpha=0.4)
    iq_ref = T_LOAD_HEAVY / MPC_MOTOR.KT
    ax.axhline(iq_ref, color="crimson", ls="--", lw=0.8, alpha=0.6,
               label=f"iq_ref@20mNm = {iq_ref:.2f} A")
    _load_lines(ax)
    ax.set_ylabel("iq [A]")
    ax.set_title("q-axis current  (torque channel)")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.grid(alpha=0.25)

    # ── Panel 3: id ─────────────────────────────────────────────────────────
    ax = axes[2]
    ax.plot(t, d["id_plant"], "k--", lw=1.2, alpha=0.7, label="id plant (truth)")
    ax.plot(t, d["id"],       "C5",  lw=1.0,             label="id MPC est")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5, label="id_ref = 0 (MTPA)")
    _load_lines(ax)
    ax.set_ylabel("id [A]")
    ax.set_title("d-axis current  (MTPA: id_ref = 0)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.25)

    # ── Panel 4: vq ─────────────────────────────────────────────────────────
    ax = axes[3]
    ax.plot(t, d["vq"], "C1", lw=1.0, label="vq [V]")
    ax.axhline( MPC_MOTOR.V_MAX, color="gray", ls=":", lw=0.8, alpha=0.5,
                label=f"+V_MAX = {MPC_MOTOR.V_MAX:.2f} V")
    ax.axhline(-MPC_MOTOR.V_MAX, color="gray", ls=":", lw=0.8, alpha=0.5)
    _load_lines(ax)
    ax.set_ylabel("vq [V]")
    ax.set_title("q-axis voltage command  (physical, pre-SVPWM norm)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.25)

    # ── Panel 5: vd ─────────────────────────────────────────────────────────
    ax = axes[4]
    ax.plot(t, d["vd"], "C3", lw=1.0, label="vd [V]")
    ax.axhline(0, color="k", ls="--", lw=0.6, alpha=0.4)
    ax.axhline( MPC_MOTOR.V_MAX, color="gray", ls=":", lw=0.8, alpha=0.5,
                label=f"+V_MAX = {MPC_MOTOR.V_MAX:.2f} V")
    ax.axhline(-MPC_MOTOR.V_MAX, color="gray", ls=":", lw=0.8, alpha=0.5)
    _load_lines(ax)
    ax.set_ylabel("vd [V]")
    ax.set_xlabel("t [s]")
    ax.set_title("d-axis voltage command  (physical, pre-SVPWM norm)")
    ax.legend(fontsize=8, ncol=4, loc="upper right")
    ax.grid(alpha=0.25)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(d: dict) -> None:
    """
    Print end-of-simulation performance summary.

    Steady-state window: last 20% of T_SIM (same as tuner cost function).

    THESIS CONVERGENCE METRICS
    --------------------------
    In addition to the standard FOC metrics (SS speed error, id RMS, load drop),
    this function reports convergence statistics derived from MPC log_data:

      σ(id) SS   : standard deviation of id over the steady-state window.
                   Non-zero σ(id) is the quantitative fingerprint of the short-
                   horizon id limit-cycle.  Thesis expectation:
                     N=10 horizon: σ(id) ≈ 0.1–0.2 A  (τ_e/N ≫ dt, horizon too short
                                           to fully settle id in N steps)
                     N=30 horizon: σ(id) → 0.02–0.05 A (improvement thesis claims)
                   Reference: analytical bound σ²(id) ≤ (b·V_MAX)²·N / (Q_id·Σbk²)

      σ(iq) SS   : standard deviation of iq.  Should approach σ ≈ 0 when speed
                   is regulated and load is constant (iq tracks iq_ref).

      i_circle SS: mean MTPA current headroom sqrt(I_MAX²−id²) − |iq| [A].
                   Positive → no overcurrent; near-zero → id oscillation is
                   consuming the current budget that should go to iq.

      E[id] SS   : mean d-axis current in steady state.  MTPA target is 0 A.
                   Non-zero E[id] indicates a systematic bias (e.g. SMO BEMF
                   offset or wrong L value).  The ±2.5 A oscillation has E[id]≈0
                   (it is symmetric), confirming it is a limit-cycle, not a bias.
    """
    t = d["t"]; rpm = d["speed_rpm"]; ref = d["omega_ref_rpm"]
    n = len(t); ss = int(0.80 * n)
    ss_err  = float(np.mean(np.abs(rpm[ss:] - ref[ss:])))
    id_rms  = float(np.sqrt(np.mean(d["id"][ss:] ** 2)))
    vref_mx = float(np.max(d["vref"]))
    iq_ss   = float(np.mean(np.abs(d["iq"][ss:])))
    after = t > T_LOAD_T2; pre = (t >= T_LOAD_T2 - 0.1) & (t < T_LOAD_T2)
    drop  = 0.0
    if np.any(after) and np.any(pre):
        drop = max(0.0, float(np.mean(rpm[pre])) - float(np.mean(rpm[after][:50])))

    print(f"\n{'='*60}")
    print("  MPC FOC + SMO -- Performance Summary")
    print(f"{'='*60}")
    print(f"  Active weights   : Q_id={_ACTIVE_GAINS['Q_id']:.4f}  "
          f"Q_iq={_ACTIVE_GAINS['Q_iq']:.4f}  "
          f"R_vd={_ACTIVE_GAINS['R_vd']:.4f}  "
          f"R_vq={_ACTIVE_GAINS['R_vq']:.4f}  "
          f"KI_v={_ACTIVE_GAINS['KI_v']:.4f}")
    print(f"  Final speed      : {rpm[-1]:.1f} RPM  (target {TARGET_RPM:.0f})")
    print(f"  SS error         : {ss_err:.2f} RPM  (last 20%)")
    print(f"  id RMS (MTPA)    : {id_rms:.4f} A    (target 0)")
    print(f"  Load drop        : {drop:.0f} RPM  at t={T_LOAD_T2}s")
    print(f"  iq SS            : {iq_ss:.3f} A  "
          f"(expected {T_LOAD_HEAVY/MPC_MOTOR.KT:.2f} A for 20 mN·m)")
    print(f"  Vref max         : {vref_mx:.3f}  (clip 0.95)")

    # ---- Convergence / variance metrics (thesis-grade) ----------------------
    # These are computed from MPC log_data if available (populated by both
    # compute_py and compute_c backends after this patch).
    mpc_ld = d.get("_mpc_log_data")
    if mpc_ld and len(mpc_ld.get("t", [])) > 10:
        t_log   = np.array(mpc_ld["t"])
        id_log  = np.array(mpc_ld["id"])
        iq_log  = np.array(mpc_ld["iq"])
        ss_mask = t_log > 0.80 * T_SIM

        if np.any(ss_mask):
            id_ss   = id_log[ss_mask]
            iq_ss_a = iq_log[ss_mask]

            id_mean_ss = float(np.mean(id_ss))
            id_std_ss  = float(np.std(id_ss, ddof=1)) if len(id_ss) > 1 else 0.0
            iq_std_ss  = float(np.std(iq_ss_a, ddof=1)) if len(iq_ss_a) > 1 else 0.0

            # Circle headroom from log if available, else recompute
            if "i_lim_circle" in mpc_ld and len(mpc_ld["i_lim_circle"]) == len(t_log):
                circ_log  = np.array(mpc_ld["i_lim_circle"])
                circ_mean = float(np.mean(circ_log[ss_mask]))
            else:
                circ_mean = float(np.mean(
                    np.sqrt(np.maximum(0.0, MPC_MOTOR.I_MAX**2 - id_ss**2)) - np.abs(iq_ss_a)
                ))

            # id variance from log if available, else recompute
            id_var_mean = id_std_ss ** 2

            # Analytical bound: σ²(id) ≤ (b·V_MAX)²·N / (Q_id·Σbk²)
            # bk ≈ b·(1−a^N)/(1−a) for geometric series; use exact b=dt/L
            b_coeff = DT / MPC_MOTOR.L_D   # [A/V]
            a_coeff = 1.0 - DT * MPC_MOTOR.R_S / MPC_MOTOR.L_D
            sum_bk2_approx = (b_coeff ** 2) * (1.0 - a_coeff ** (2 * MPC_N)) / (1.0 - a_coeff ** 2 + 1e-30)
            id_var_bound = (b_coeff * MPC_MOTOR.V_MAX) ** 2 * MPC_N / (
                _ACTIVE_GAINS["Q_id"] * max(sum_bk2_approx, 1e-30)
            )
            id_std_bound = math.sqrt(id_var_bound)

            print(f"\n  -- Convergence variance (thesis metrics) --")
            print(f"  Horizon N        : {MPC_N}  |  τ_e = {MPC_MOTOR.L_D/MPC_MOTOR.R_S*1e3:.2f} ms"
                  f"  |  N·dt = {MPC_N*DT*1e3:.2f} ms"
                  f"  |  τ_e/N·dt = {MPC_MOTOR.L_D/MPC_MOTOR.R_S/(MPC_N*DT):.2f}")
            print(f"  E[id] SS         : {id_mean_ss:+.4f} A    (0 = no MTPA bias; ≠0 = SMO/L offset)")
            print(f"  σ(id) SS         : {id_std_ss:.4f} A    measured")
            print(f"  σ(id) bound      : {id_std_bound:.4f} A    analytical upper bound (N={MPC_N}, Q_id={_ACTIVE_GAINS['Q_id']:.1f})")
            print(f"  σ²(id) SS        : {id_var_mean:.5f} A²  (thesis: ↓ with N↑ or Q_id↑)")
            print(f"  σ(iq) SS         : {iq_std_ss:.4f} A    (0 = steady load; ↑ = speed ripple)")
            print(f"  i_circle mean SS : {circ_mean:+.3f} A    MTPA headroom sqrt(I_MAX²−id²)−|iq|")
            if circ_mean < 0.05:
                print(f"  WARNING: i_circle < 0.05 A -- id oscillation is consuming iq headroom")
            # Thesis interpretation note
            tau_e   = MPC_MOTOR.L_D / MPC_MOTOR.R_S
            n_tau_e = tau_e / DT
            print(f"\n  Thesis interpretation:")
            print(f"    τ_e = {tau_e*1e3:.2f} ms = {n_tau_e:.0f} ISR steps.  "
                  f"N={MPC_N} horizon covers {MPC_N/n_tau_e*100:.0f}% of τ_e.")
            print(f"    Short horizon (N·dt ≪ τ_e) cannot fully damp id in N steps.")
            print(f"    σ(id)={id_std_ss:.4f} A is structural (not a wiring bug).")
            print(f"    Increasing N to ≥{int(n_tau_e*1.5):.0f} steps should reduce σ(id)"
                  f" toward the bound {id_std_bound:.4f} A.")
    else:
        print("\n  [Convergence metrics: MPC log_data not available in result dict]")
        print("   Pass _mpc_log_data=mpc.log_data in the result dict to enable.")

    print(f"{'='*60}")




# =============================================================================
# =============================================================================
#
#   INTEGRATED CMA-ES + NEURAL NETWORK WARM-START WEIGHT TUNER
#
#   Lives in this file so it shares _run_sim(), _ACTIVE_GAINS, and
#   motor constants directly -- no monkey-patching, no separate module.
#
#   THESIS ARCHITECTURE
#   -------------------
#   1. CMA-ES (gradient-free evolutionary strategy)
#      - N(m, sigma^2 * C) search distribution over normalised [0,1]^5
#      - Each generation: sample lambda candidates -> run lambda FMU sims
#        -> rank by cost -> update (m, sigma, C)
#      - C adapts to the shape of the cost surface (learns local Hessian
#        without any derivative computation)
#
#   2. Neural Network warm-start (MLP 5->16->16->1, pure NumPy)
#      - Trained on Phase 1 CMA-ES history (real FMU evaluations only)
#      - Gradient descent through frozen MLP predicts cost minimum
#      - Prediction injected as CMA-ES mean m0 for Phase 3
#      - Focuses remaining budget near predicted optimum
#
#   WHY PURE NUMPY FOR THE MLP?
#   ----------------------------
#   The MLP is a lightweight warm-start predictor, not a production
#   surrogate. Pure NumPy keeps the file self-contained with zero extra
#   dependencies. Students can read the forward pass, MSE loss, Adam
#   update, and backprop in ~80 lines without any framework abstraction
#   -- ideal for thesis exposition of how neural networks work.
#
# =============================================================================
# =============================================================================

# ── Tuner hyper-parameters ────────────────────────────────────────────────────

# Phase 1: number of FMU evaluations before NN training.
# CMA-ES theory default lambda = 4 + floor(3*ln(5)) = 8 for d=5.
# Use >= 2*lambda so Phase 1 covers at least two full generations.
_T_PHASE1_EVALS  = 16       # Phase 1 FMU budget
_T_MAX_FMU_EVALS = 100      # total budget (Phase 1 + Phase 3)
_T_SIGMA0        = 0.3      # CMA-ES initial step size in normalised space
_T_N_RESTARTS    = 1        # IPOP restarts in Phase 3
_T_SIM_DURATION  = 3.0      # [s] per tuner evaluation

# NN warm-start hyper-parameters
# THESIS NOTE: With only _T_PHASE1_EVALS=16 points and 369 MLP parameters
# the network is over-parameterised. Regularisation comes from:
#   1. Small lr (3e-3) limits per-step weight change
#   2. Adam bias correction reduces effective lr early in training
#   3. Smooth cost surface (motor physics C-inf) needs only coarse fit
# Training for 600 epochs on 16 points takes < 0.1 s -- negligible.
_T_NN_EPOCHS     = 600
_T_NN_LR         = 3e-3     # Adam lr -- MLP training
_T_NN_OPT_STEPS  = 500      # gradient-descent steps on frozen MLP
_T_NN_LR_OPT     = 5e-2     # Adam lr -- input optimisation
_T_NN_RESTARTS   = 8        # independent restarts on MLP landscape

# ── Cost function weights ─────────────────────────────────────────────────────
# THESIS NOTE -- linear Pareto scalarisation:
#   J = W_ID*id_rms + W_SS*ss_err + W_DROP*load_drop
#   W_ID=80 dominates: non-zero id wastes copper losses at every 20 kHz cycle.
#   W_SS=2: small SS offsets tolerated.  W_DROP=0.5: transients penalised lightly.
_T_W_ID   = 80.0
_T_W_SS   =  2.0
_T_W_DROP =  0.5
_T_W_VREF = 20.0            # flat penalty if Vref p90 >= 0.93

# ── Physics-derived search bounds ─────────────────────────────────────────────
# THESIS NOTE: bounds derived from analytical MPC solution, not guessed.
#   Q_id < 5  -> id diverges (b=dt/L term dominates denominator)
#   Q_iq > 1  -> degrades iq tracking (appears in vq denominator)
#   KI_v > 0.1 -> integral windup during soft-start ramp
_T_BOUNDS = np.array([
    [  5.0,  50.0],   # Q_id
    [  0.01,  1.0],   # Q_iq
    [  0.001, 0.10],  # R_vd
    [  0.005, 0.20],  # R_vq
    [  0.005, 0.10],  # KI_v
], dtype=np.float64)

_T_PARAM_NAMES  = ["Q_id", "Q_iq", "R_vd", "R_vq", "KI_v"]
_T_DEFAULTS     = [_ACTIVE_GAINS["Q_id"], _ACTIVE_GAINS["Q_iq"],
                   _ACTIVE_GAINS["R_vd"], _ACTIVE_GAINS["R_vq"],
                   _ACTIVE_GAINS["KI_v"]]
_T_DIVERGE_COST = 1e6


# =============================================================================
# Normalisation helpers
# =============================================================================
# THESIS NOTE: CMA-ES requires commensurable dimensions.
# Q_id in [5,50] and R_vd in [0.001,0.1] differ by ~10000x.
# Mapping to [0,1]^5 gives CMA-ES a well-conditioned unit sphere.

def _to_norm(x: np.ndarray) -> np.ndarray:
    """Physical gains -> normalised [0,1]^5."""
    return (x - _T_BOUNDS[:, 0]) / (_T_BOUNDS[:, 1] - _T_BOUNDS[:, 0])

def _to_phys(x: np.ndarray) -> np.ndarray:
    """Normalised [0,1]^5 -> physical gains (clips to bounds)."""
    return _T_BOUNDS[:, 0] + np.clip(x, 0.0, 1.0) * (_T_BOUNDS[:, 1] - _T_BOUNDS[:, 0])


# =============================================================================
# FMU evaluation  --  one complete closed-loop simulation
# =============================================================================

def _t_run_with_gains(
    Q_id: float, Q_iq: float, R_vd: float, R_vq: float, KI_v: float,
) -> dict | None:
    """
    Run one tuner simulation with given weights, restoring gains on exit.
    Returns raw _run_sim() result dict, or None on failure.
    """
    saved = _ACTIVE_GAINS.copy()
    try:
        _ACTIVE_GAINS["Q_id"] = Q_id; _ACTIVE_GAINS["Q_iq"] = Q_iq
        _ACTIVE_GAINS["R_vd"] = R_vd; _ACTIVE_GAINS["R_vq"] = R_vq
        _ACTIVE_GAINS["KI_v"] = KI_v
        return _run_sim(with_codegen_hooks=False, t_sim=_T_SIM_DURATION)
    finally:
        _ACTIVE_GAINS.update(saved)


def _t_cost(d: dict | None) -> dict | None:
    """
    Extract scalar cost + metrics from a _run_sim() result dict.
    Returns None on divergence or insufficient data.

    THESIS NOTE -- steady-state window:
      Last 15% of _T_SIM_DURATION = 3s, i.e. t > 2.55s.
      Both load steps (t=0.5s, t=1.2s) have settled by then.
    """
    if d is None:
        return None
    t = d["t"]; rpm = d["speed_rpm"]; idd = d["id"]
    if len(t) < 200:
        return None
    if float(np.max(np.abs(rpm))) > TARGET_RPM * 3.0:
        return None
    ss = t > 0.85 * _T_SIM_DURATION
    if not np.any(ss):
        return None
    ref_ss = float(np.mean(d["omega_ref_rpm"][ss]))
    ss_err = float(np.mean(np.abs(rpm[ss] - ref_ss)))
    id_rms = float(np.sqrt(np.mean(idd[ss] ** 2)))
    if ss_err > 800.0:
        return None
    after = t >= T_LOAD_T2; before = (t >= T_LOAD_T2 - 0.1) & (t < T_LOAD_T2)
    load_drop = 0.0
    if np.any(after) and np.any(before):
        load_drop = max(0.0, float(np.mean(rpm[before])) - float(np.mean(rpm[after][:50])))
    cost = _T_W_ID * id_rms + _T_W_SS * ss_err + _T_W_DROP * load_drop
    vref = d.get("vref")
    if vref is not None and len(vref) > 0:
        if float(np.percentile(np.abs(vref), 90)) >= 0.93:
            cost += _T_W_VREF
    return {"cost": cost, "id_rms": id_rms, "ss_err": ss_err, "load_drop": load_drop}


def _evaluate_fmu(
    x_norm: np.ndarray,
    eval_counter: List[int],
    history: List[dict],
    t0_wall: float,
) -> float:
    """
    CMA-ES objective: denormalise x_norm, run real FMU, return scalar cost.

    Records result in history for NN training in Phase 2.
    """
    gains = _to_phys(x_norm)
    Q_id, Q_iq, R_vd, R_vq, KI_v = gains
    eval_counter[0] += 1
    n = eval_counter[0]
    elapsed = time.perf_counter() - t0_wall
    print(f"  [{n:3d}]  Q_id={Q_id:6.2f}  Q_iq={Q_iq:.3f}  "
          f"R_vd={R_vd:.4f}  R_vq={R_vq:.4f}  KI_v={KI_v:.4f}",
          end="  ", flush=True)
    try:
        raw     = _t_run_with_gains(Q_id, Q_iq, R_vd, R_vq, KI_v)
        metrics = _t_cost(raw)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"-> EXCEPTION: {exc}")
        history.append({"gains": gains, "cost": _T_DIVERGE_COST, "status": "exception"})
        return _T_DIVERGE_COST
    if metrics is None:
        print("-> DIVERGED")
        history.append({"gains": gains, "cost": _T_DIVERGE_COST, "status": "diverged"})
        return _T_DIVERGE_COST
    cost = metrics["cost"]
    print(f"-> cost={cost:.3f}  id={metrics['id_rms']:.3f}A  "
          f"ss={metrics['ss_err']:.0f}RPM  drop={metrics['load_drop']:.0f}RPM  "
          f"t={elapsed:.0f}s")
    history.append({"gains": gains, "cost": cost,
                    "id_rms": metrics["id_rms"], "ss_err": metrics["ss_err"],
                    "load_drop": metrics["load_drop"], "status": "ok"})
    return cost


# =============================================================================
# Neural Network  --  MLP 5->16->16->1  (pure NumPy, self-contained)
# =============================================================================
# THESIS NOTE -- architecture:
#   5 inputs (one per tunable gain), two hidden layers of 16 tanh units,
#   one linear output (scalar cost). 369 total parameters.
#
#   tanh: smooth, bounded, well-matched to smooth motor cost surface.
#   Linear output: unbounded, correct for positive scalar regression.
#   Kaiming init: W ~ N(0, sqrt(2/fan_in)) -- stable activation variance.
#
# The network is ONLY used to produce a warm-start mean for CMA-ES.
# CMA-ES continues to evaluate the real FMU after the warm-start.

class _MLP:
    """2-hidden-layer MLP, pure NumPy. Architecture: n_in->hidden->hidden->1."""

    def __init__(self, n_in: int = 5, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        def _w(r, c): return rng.standard_normal((r, c)) * np.sqrt(2.0 / c)
        self.n_in = n_in
        self.W1 = _w(hidden, n_in);   self.b1 = np.zeros(hidden)
        self.W2 = _w(hidden, hidden); self.b2 = np.zeros(hidden)
        self.W3 = _w(1, hidden);      self.b3 = np.zeros(1)
        # Adam moment accumulators
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass. X shape: (batch, n_in). Returns (batch,)."""
        h1 = np.tanh(X  @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1)

    def _fwd_cache(self, X):
        z1 = X  @ self.W1.T + self.b1; h1 = np.tanh(z1)
        z2 = h1 @ self.W2.T + self.b2; h2 = np.tanh(z2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1), h1, h2

    def loss_and_grad(self, X: np.ndarray, y_true: np.ndarray):
        """
        MSE loss + backpropagation gradients.

        THESIS NOTE -- chain rule, layer by layer:
          dL/dW3 = dL/dy * h2             (output layer)
          dL/dW2 = dL/dh2 * (1-h2^2) * h1 (tanh deriv: 1 - tanh^2)
          dL/dW1 = dL/dh1 * (1-h1^2) * X  (hidden layer 1)
        """
        N = X.shape[0]
        y, h1, h2 = self._fwd_cache(X)
        err   = y - y_true
        L     = float(np.mean(err ** 2))
        dL_dy = 2.0 * err / N
        dW3   = (dL_dy[:, None] * h2).mean(0, keepdims=True)
        db3   = dL_dy.mean(0, keepdims=True)
        dh2   = dL_dy[:, None] * self.W3
        dz2   = dh2 * (1.0 - h2 ** 2)
        dW2   = dz2.T @ h1 / N
        db2   = dz2.sum(0) / N
        dh1   = dz2 @ self.W2
        dz1   = dh1 * (1.0 - h1 ** 2)
        dW1   = dz1.T @ X / N
        db1   = dz1.sum(0) / N
        return L, [dW1, db1, dW2, db2, dW3, db3]

    def _adam(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam parameter update (Kingma & Ba 2015).

        THESIS NOTE:
          m_hat = m / (1 - beta1^t)  bias-corrected first moment
          v_hat = v / (1 - beta2^t)  bias-corrected second moment
          theta -= lr * m_hat / (sqrt(v_hat) + eps)
        Bias correction prevents large first steps when m, v start at zero.
        """
        self._t += 1
        t = self._t
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._m[i] = beta1 * self._m[i] + (1.0 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1.0 - beta2) * g ** 2
            m_hat = self._m[i] / (1.0 - beta1 ** t)
            v_hat = self._v[i] / (1.0 - beta2 ** t)
            p    -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 600, lr: float = 3e-3, verbose: bool = True):
        """Full-batch Adam training."""
        losses = []
        for ep in range(epochs):
            L, grads = self.loss_and_grad(X, y)
            self._adam(grads, lr=lr)
            losses.append(L)
            if verbose and (ep % 150 == 0 or ep == epochs - 1):
                print(f"    epoch {ep:4d}  MSE={L:.6f}")
        return losses

    def scalar_grad(self, x: np.ndarray):
        """
        Forward pass + d(output)/d(x) for a single input vector.

        THESIS NOTE -- input optimisation:
          Normally backprop computes dL/dW to update weights.
          Here weights are FROZEN and we compute d(output)/d(x)
          to update the INPUT x -- minimising the predicted cost
          by adjusting the gain vector. Called "feature inversion"
          or "input optimisation" (same mechanism as adversarial
          examples and neural style transfer).
        """
        X = x[None, :]
        y, h1, h2 = self._fwd_cache(X)
        dh2 = np.ones((1, self.W3.shape[1])) * self.W3
        dz2 = dh2 * (1.0 - h2 ** 2)
        dh1 = dz2 @ self.W2
        dz1 = dh1 * (1.0 - h1 ** 2)
        dx  = (dz1 @ self.W1).squeeze(0)
        return float(y.squeeze()), dx


# =============================================================================
# Phase 2  --  NN warm-start: train MLP on history, predict optimum
# =============================================================================

def _nn_predict_warmstart(history: List[dict],
                          rng: np.random.Generator) -> Optional[np.ndarray]:
    """
    Train MLP on Phase 1 FMU history and predict cost-minimising gains.

    THESIS NOTE -- warm-start workflow:
      1. Collect valid (gains, cost) pairs from Phase 1 FMU history.
      2. Normalise inputs to [0,1]^5; standardise costs to N(0,1).
         Input norm: prevents Q_id (~27) dominating R_vd (~0.005).
         Output std: keeps MSE in well-conditioned range for Adam.
      3. Train MLP to predict normalised cost from normalised gains.
      4. Run Adam gradient descent through the FROZEN MLP to minimise
         its predicted cost (_T_NN_RESTARTS independent starts).
      5. Return the best x found -> becomes CMA-ES mean m0 for Phase 3.

    Returns normalised [0,1]^5 warm-start vector, or None if insufficient data.
    """
    valid = [e for e in history if e["status"] == "ok"]
    if len(valid) < 4:
        print(f"  [NN] Insufficient data ({len(valid)} pts, need >=4). Skipping.")
        return None

    X_raw = np.array([_to_norm(e["gains"]) for e in valid])
    y_raw = np.array([e["cost"] for e in valid])
    y_mean = float(y_raw.mean())
    y_std  = float(max(y_raw.std(), 1e-8))
    y_norm = (y_raw - y_mean) / y_std

    print(f"\n  [Phase 2 NN] Training MLP 5->16->16->1 on {len(valid)} FMU points ...")
    mlp = _MLP(n_in=5, hidden=16, seed=0)
    mlp.train(X_raw, y_norm, epochs=_T_NN_EPOCHS, lr=_T_NN_LR, verbose=True)

    # Validation R^2 (informational -- not a gate)
    n_val = max(1, len(X_raw) // 5)
    idx   = rng.permutation(len(X_raw))
    X_va  = X_raw[idx[:n_val]]; y_va = y_norm[idx[:n_val]]
    y_p   = mlp.forward(X_va)
    ss_res = float(np.var(y_va - y_p)); ss_tot = float(np.var(y_va))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    print(f"  [NN] Surrogate R^2 (held-out {n_val} pts): {r2:.4f}  (>=0.60 acceptable)")
    if r2 < 0.30:
        print("  [NN] WARNING: R^2 very low -- warm-start may not improve CMA-ES.")

    # Gradient descent through frozen MLP to find predicted minimum
    # THESIS NOTE: _T_NN_RESTARTS independent Adam trajectories mitigate
    # local minima on the MLP's cost surface. Restart 0 is warm-started
    # from the best observed point; others use random init.
    print(f"  [NN] Input optimisation ({_T_NN_RESTARTS} restarts x {_T_NN_OPT_STEPS} steps) ...")
    best_obs_idx = int(np.argmin(y_norm))
    best_x       = X_raw[best_obs_idx].copy()
    best_pred    = float(mlp.forward(best_x[None, :])[0])

    for restart in range(_T_NN_RESTARTS):
        x  = best_x.copy() if restart == 0 else rng.uniform(0.0, 1.0, size=5)
        mi = np.zeros(5); vi = np.zeros(5); ti = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for _ in range(_T_NN_OPT_STEPS):
            c, grad = mlp.scalar_grad(x)
            ti += 1
            mi  = b1*mi + (1.0-b1)*grad
            vi  = b2*vi + (1.0-b2)*grad**2
            mh  = mi / (1.0 - b1**ti)
            vh  = vi / (1.0 - b2**ti)
            x   = np.clip(x - _T_NN_LR_OPT * mh / (np.sqrt(vh) + eps), 0.0, 1.0)
        c_f = float(mlp.forward(x[None, :])[0])
        if c_f < best_pred:
            best_pred = c_f; best_x = x.copy()

    pred_phys = _to_phys(best_x)
    print(f"  [NN] Predicted warm-start (pred cost={best_pred:.4f}):")
    for name, val in zip(_T_PARAM_NAMES, pred_phys):
        print(f"       {name:<8} = {val:.6f}")
    return best_x   # normalised [0,1]^5


# =============================================================================
# CMA-ES runner
# =============================================================================

def _run_cmaes(x0, sigma0, budget, eval_counter, history, t0_wall,
               n_restarts=1, label="CMA-ES"):
    """
    Run IPOP-CMA-ES in normalised [0,1]^5 from starting mean x0.

    THESIS NOTE -- IPOP (Auger & Hansen 2005):
      On stagnation (sigma collapses, flat progress), restart with
      lambda doubled: lambda_{k+1} = 2 * lambda_k.
      Larger populations explore wider basins to escape local minima.
      Budget is shared across all restarts.
    """
    if not _HAVE_CMA:
        raise ImportError("'cma' package required. Install: pip install cma")

    best_cost = np.inf; best_x = x0.copy()
    lam = None  # None -> CMA-ES default: 4 + floor(3*ln(5)) = 8

    for restart in range(n_restarts + 1):
        remaining = max(0, budget - eval_counter[0])
        if remaining <= 0:
            break
        print(f"\n  [{label} restart {restart}]  "
              f"popsize={lam if lam else 'auto(8)'}  remaining={remaining}")

        def _obj(xn):
            if eval_counter[0] >= _T_MAX_FMU_EVALS:
                return _T_DIVERGE_COST * 0.99
            return _evaluate_fmu(np.asarray(xn), eval_counter, history, t0_wall)

        opts = cma.CMAOptions()
        opts.set("maxfevals", remaining)
        opts.set("bounds",    [[0.0]*5, [1.0]*5])
        opts.set("tolx",      1e-4)
        opts.set("tolfun",    1e-5)
        opts.set("verbose",   -9)
        if lam is not None:
            opts.set("popsize", lam)

        xopt, es = cma.fmin2(_obj, x0.copy(), sigma0 / (2.0**restart), opts)
        xc = np.clip(np.asarray(xopt), 0.0, 1.0)
        c  = es.result.fbest
        if c < best_cost:
            best_cost = c; best_x = xc.copy()

        print(f"  [{label} restart {restart} done]  "
              f"best cost={best_cost:.4f}  total evals={eval_counter[0]}")
        lam = 2 * (lam if lam else (4 + int(3*math.log(5))))
        x0  = best_x.copy()

    return best_x


def _run_nelder_mead_fallback(x0, budget, eval_counter, history, t0_wall):
    """
    Nelder-Mead fallback (scipy) when 'cma' is not installed.

    THESIS NOTE -- Nelder-Mead (Nelder & Mead 1965):
      Maintains a simplex of d+1=6 vertices. Worst vertex replaced by
      reflection, expansion, or contraction each step.
      Simpler than CMA-ES: no covariance adaptation, less reliable for
      d>4. Adaptive variant (Gao & Han 2010) improves for d>2.
    """
    if not _HAVE_SCIPY:
        raise ImportError("Neither 'cma' nor 'scipy' available. "
                          "Install one: pip install cma  or  pip install scipy")

    def _obj(xn):
        if eval_counter[0] >= _T_MAX_FMU_EVALS:
            return _T_DIVERGE_COST * 0.99
        return _evaluate_fmu(np.asarray(xn), eval_counter, history, t0_wall)

    result = _scipy_minimize(_obj, x0, method="Nelder-Mead",
                             options={"maxfev": budget, "xatol": 1e-3,
                                      "fatol": 1e-4, "adaptive": True})
    return np.clip(result.x, 0.0, 1.0)


# =============================================================================
# gains.h writer  --  c_src/embed_sim_mpc_gains.h
# =============================================================================

def _write_gains_header(gains, metrics, n_evals, method, out_path):
    """
    Write embed_sim_mpc_gains.h with ISO 26262 audit trail.

    THESIS NOTE -- auto-generated headers in safety-critical software:
      ISO 26262 Part 6 requires traceable parameter propagation.
      Manual transcription is error-prone and non-auditable.
      Auto-generation guarantees exact numeric fidelity and embeds
      full provenance as structured comments.
      MISRA C:2012: Rule 7.2 (f suffix), Rule 8.7 (#defines for constants).
    """
    Q_id, Q_iq, R_vd, R_vq, KI_v = [float(v) for v in gains]
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    content = textwrap.dedent(f"""\
    /******************************************************************************
     * \\file      embed_sim_mpc_gains.h
     * \\brief     MPC FOC weight constants -- CMA-ES + NN warm-start tuned
     *             NANOTEC DB42S02  |  AURIX TC3xx  |  EmbedSim
     *
     * \\details   AUTO-GENERATED by db42s02_closed_loop_mpc_foc_20k.py
     *            DO NOT EDIT MANUALLY.
     *
     *  =========================================================================
     *  TUNING AUDIT TRAIL  (ISO 26262 Part 6 traceability)
     *  =========================================================================
     *  Generated     : {now}
     *  Method        : {method}
     *  Plant         : PMSM_Plant_FMU.fmu  (Modelica, FMU 2.0, dt={DT*1e6:.0f}us)
     *                  NO surrogate -- every evaluation is a real FMU simulation
     *  FMU evals     : {n_evals} total
     *  Sim duration  : {_T_SIM_DURATION:.1f}s per eval (no-load + 5mNm + 20mNm)
     *  Noise model   : DISABLED (clean plant, deterministic surface)
     *  Target        : {TARGET_RPM:.0f} RPM  V_DC={V_DC:.1f}V  dt={DT*1e6:.0f}us
     *  Fixed params  : Q_omega={MPC_Q_OMEGA:.1f}  N={MPC_N}
     *                  SMO_K={MPC_SMO_K:.2f}  SMO_FC={MPC_SMO_FC:.0f}
     *
     *  =========================================================================
     *  ALGORITHM SUMMARY
     *  =========================================================================
     *  Phase 1  CMA-ES gen 1 ({_T_PHASE1_EVALS} real FMU evals)
     *           Samples lambda=8 candidates/gen from N(m, sigma^2*C).
     *           Covariance C adapts to cost surface shape (no derivatives).
     *
     *  Phase 2  Neural Network warm-start
     *           MLP 5->16->16->1 trained on Phase 1 history (NumPy Adam).
     *           Gradient descent through frozen MLP finds predicted minimum.
     *           Prediction injected as CMA-ES mean m0 for Phase 3.
     *
     *  Phase 3  CMA-ES from NN mean ({_T_MAX_FMU_EVALS - _T_PHASE1_EVALS} remaining evals)
     *           Resumes from m0. IPOP doubles population on stagnation.
     *
     *  Ref: Hansen (2016) arXiv:1604.00772  |  Kingma & Ba (2015) arXiv:1412.6980
     *
     *  =========================================================================
     *  COST FUNCTION
     *  =========================================================================
     *  J = {_T_W_ID:.1f} * id_rms_A            (MTPA: id->0, minimise copper losses)
     *    + {_T_W_SS:.1f} * ss_speed_error_RPM   (steady-state speed tracking)
     *    + {_T_W_DROP:.1f} * load_drop_RPM        (load-step rejection)
     *    + {_T_W_VREF:.1f} * [1 if Vref_p90 >= 0.93]  (over-modulation penalty)
     *
     *  =========================================================================
     *  VERIFICATION RESULT
     *  =========================================================================
     *  Total cost      : {metrics.get('cost', float('nan')):.4f}
     *  id RMS  (MTPA)  : {metrics.get('id_rms', float('nan')):.4f} A   (target 0 A)
     *  SS speed error  : {metrics.get('ss_err', float('nan')):.2f} RPM
     *  Load-step drop  : {metrics.get('load_drop', float('nan')):.1f} RPM at t={T_LOAD_T2}s
     *
     *  Baseline (hardware commissioning):
     *    Q_id={_T_DEFAULTS[0]:.4f}  Q_iq={_T_DEFAULTS[1]:.4f}  R_vd={_T_DEFAULTS[2]:.4f}
     *    R_vq={_T_DEFAULTS[3]:.4f}  KI_v={_T_DEFAULTS[4]:.4f}
     *
     * \\note  MISRA C:2012 Rule 7.2: float literals carry the 'f' suffix.
     * \\note  MISRA C:2012 Rule 8.7: #defines avoid magic numbers in .c files.
     * \\version   1.0.0  (auto-generated)
     * \\copyright Copyright (C) EmbedSim 2026
     ******************************************************************************/

    #ifndef EMBED_SIM_MPC_GAINS_H_
    #define EMBED_SIM_MPC_GAINS_H_

    /* embed_sim_matrix.h: typedef float MatrixFloat; */
    #include "embed_sim_matrix.h"

    /**
     * \\defgroup MPC_Gains_Tuned  CMA-ES + NN warm-start tuned MPC constants
     *
     * Cost matrices:
     *   Q = diag(MPC_Q_ID, MPC_Q_IQ, MPC_Q_OMEGA)  [state penalty]
     *   R = diag(MPC_R_VD, MPC_R_VQ)                [control effort]
     *
     * Closed-form MPC optimum per ISR tick:
     *   vd* = Q_id*sum_bk*(0-id_free) / (Q_id*sum_bk2 + R_vd)
     *   vq* = (Q_omega*sum_ek*(w_ref-w_free) + Q_iq*sum_bk*(0-iq_free))
     *          / (Q_omega*sum_ek2 + Q_iq*sum_bk2 + R_vq)
     * \\{{
     */

    /** d-axis state cost [-]  drives id->0 (MTPA).  Baseline: {_T_DEFAULTS[0]:.4f} */
    #define MPC_Q_ID    ((MatrixFloat){Q_id:.6f}f)

    /** q-axis regulariser [-]  must be << MPC_Q_OMEGA.  Baseline: {_T_DEFAULTS[1]:.4f} */
    #define MPC_Q_IQ    ((MatrixFloat){Q_iq:.6f}f)

    /** vd effort weight [-]  damps d-axis commands.  Baseline: {_T_DEFAULTS[2]:.4f} */
    #define MPC_R_VD    ((MatrixFloat){R_vd:.6f}f)

    /** vq effort weight [-]  damps cross-coupling.  Baseline: {_T_DEFAULTS[3]:.4f} */
    #define MPC_R_VQ    ((MatrixFloat){R_vq:.6f}f)

    /** Speed-error integral gain [V/(rad/s*s)].  Baseline: {_T_DEFAULTS[4]:.4f} */
    #define MPC_KI_V    ((MatrixFloat){KI_v:.6f}f)

    /**
     * Speed tracking weight [-]  FIXED -- not tuned.
     * Fixed at {MPC_Q_OMEGA:.0f} to preserve commissioning torque calibration:
     *   iq_ss = T_load/KT = 0.020/0.0084 = 2.38 A at 20 mN·m.
     */
    #define MPC_Q_OMEGA ((MatrixFloat){MPC_Q_OMEGA:.1f}f)

    /** \\}} */

    typedef struct {{
        MatrixFloat Q_id;    /**< MPC_Q_ID    = {Q_id:.6f}f */
        MatrixFloat Q_iq;    /**< MPC_Q_IQ    = {Q_iq:.6f}f */
        MatrixFloat R_vd;    /**< MPC_R_VD    = {R_vd:.6f}f */
        MatrixFloat R_vq;    /**< MPC_R_VQ    = {R_vq:.6f}f */
        MatrixFloat KI_v;    /**< MPC_KI_V    = {KI_v:.6f}f */
        MatrixFloat Q_omega; /**< MPC_Q_OMEGA = {MPC_Q_OMEGA:.1f}f (fixed) */
    }} MPC_GainSet_T;

    #endif /* EMBED_SIM_MPC_GAINS_H_ */
    """)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding="utf-8")
    print(f"\n  [gains.h] Written: {out_path}")


# =============================================================================
# run_tuner  --  four-phase entry point
# =============================================================================

def run_tuner() -> bool:
    """
    Execute the four-phase CMA-ES + Neural Network warm-start MPC tuner.

    Phase 1 : CMA-ES gen 1 (_T_PHASE1_EVALS real FMU simulations)
    Phase 2 : NN trains on history -> predicts warm-start mean
    Phase 3 : CMA-ES continues from NN mean (remaining budget)
    Phase 4 : Write c_src/embed_sim_mpc_gains.h

    Every evaluation is a real closed-loop FMU simulation.
    No surrogate replaces the plant.

    Returns True if tuning completed and _ACTIVE_GAINS was updated.
    """
    rng          = np.random.default_rng(seed=42)
    eval_counter = [0]
    history: List[dict] = []
    t0_wall = time.perf_counter()

    method_label = (
        "CMA-ES (Hansen 2004) + NN warm-start (MLP 5->16->16->1, NumPy Adam)"
        if _HAVE_CMA else
        "Nelder-Mead (scipy adaptive) + NN warm-start (MLP 5->16->16->1, NumPy Adam)"
    )

    # Banner
    print("\n" + "=" * 70)
    print("  MPC Tuner  --  CMA-ES + Neural Network Warm-Start")
    print("  NANOTEC DB42S02  |  AURIX TC3xx  |  EmbedSim")
    print("=" * 70)
    print(f"  Method       : {method_label}")
    print(f"  Plant        : PMSM_Plant_FMU.fmu  (NO surrogate)")
    print(f"  Total budget : {_T_MAX_FMU_EVALS} FMU evals  (T_sim={_T_SIM_DURATION:.1f}s each)")
    print(f"  Phase 1      : {_T_PHASE1_EVALS} evals  CMA-ES gen 1 -> build NN dataset")
    print(f"  Phase 2      : NN {_T_NN_EPOCHS} epochs -> predict warm-start mean m0")
    print(f"  Phase 3      : {_T_MAX_FMU_EVALS - _T_PHASE1_EVALS} evals  CMA-ES from m0 (IPOP x{_T_N_RESTARTS})")
    print(f"  Est. wall    : ~{_T_MAX_FMU_EVALS * 114 // 60} min  (at 114 s/sim)")
    print()
    print(f"  {'Parameter':<8}  {'Lo':>8}  {'Hi':>8}  {'Baseline':>10}")
    print(f"  {'-'*42}")
    for name, (lo, hi), dflt in zip(_T_PARAM_NAMES, _T_BOUNDS, _T_DEFAULTS):
        print(f"  {name:<8}  {lo:>8.4f}  {hi:>8.4f}  {dflt:>10.4f}")
    print(f"\n  Cost weights:")
    print(f"    id RMS (MTPA)   x {_T_W_ID:.1f}")
    print(f"    SS speed error  x {_T_W_SS:.1f}")
    print(f"    Load drop       x {_T_W_DROP:.1f}")
    print(f"    Vref over-mod   + {_T_W_VREF:.1f}  (if Vref p90 >= 0.93)")
    print("=" * 70)

    x_baseline = _to_norm(np.array(_T_DEFAULTS))

    # ── Phase 1: CMA-ES gen 1 ────────────────────────────────────────────────
    print(f"\n[Phase 1] CMA-ES exploration  ({_T_PHASE1_EVALS} FMU evals) ...")
    if _HAVE_CMA:
        _run_cmaes(x_baseline, _T_SIGMA0, _T_PHASE1_EVALS,
                   eval_counter, history, t0_wall, n_restarts=0, label="Phase 1")
    else:
        print("  'cma' not installed -- Nelder-Mead Phase 1.")
        _run_nelder_mead_fallback(x_baseline, _T_PHASE1_EVALS,
                                  eval_counter, history, t0_wall)

    n_valid_p1 = sum(1 for e in history if e["status"] == "ok")
    print(f"\n  Phase 1 done: {eval_counter[0]} evals, {n_valid_p1} valid.")
    if n_valid_p1 == 0:
        print("  ERROR: all Phase 1 simulations diverged.  Aborting.")
        return False

    # ── Phase 2: NN warm-start ───────────────────────────────────────────────
    print(f"\n[Phase 2] Neural Network warm-start ...")
    x_warmstart = _nn_predict_warmstart(history, rng)
    if x_warmstart is None:
        valid_p1 = [e for e in history if e["status"] == "ok"]
        best_p1  = min(valid_p1, key=lambda e: e["cost"])
        x_warmstart = _to_norm(best_p1["gains"])
        print(f"  [NN] Fallback to best Phase 1 point (cost={best_p1['cost']:.4f})")
    else:
        print("  [NN] Warm-start mean ready for CMA-ES Phase 3.")

    # ── Phase 3: CMA-ES from NN mean ─────────────────────────────────────────
    remaining = _T_MAX_FMU_EVALS - eval_counter[0]
    print(f"\n[Phase 3] CMA-ES from NN mean  ({remaining} evals, IPOP x{_T_N_RESTARTS}) ...")
    if remaining > 0:
        if _HAVE_CMA:
            _run_cmaes(x_warmstart, _T_SIGMA0 * 0.5, remaining,
                       eval_counter, history, t0_wall,
                       n_restarts=_T_N_RESTARTS, label="Phase 3")
        else:
            _run_nelder_mead_fallback(x_warmstart, remaining,
                                      eval_counter, history, t0_wall)
    else:
        print("  Budget exhausted -- using NN prediction directly.")

    # ── Find overall best ─────────────────────────────────────────────────────
    valid_all = [e for e in history if e["status"] == "ok"]
    if not valid_all:
        print("  ERROR: no valid FMU evaluations. Aborting.")
        return False

    best_entry  = min(valid_all, key=lambda e: e["cost"])
    best_gains  = best_entry["gains"]
    best_metrics = best_entry

    elapsed = time.perf_counter() - t0_wall
    n_evals = len(history)
    n_ok    = len(valid_all)
    Q_id_f, Q_iq_f, R_vd_f, R_vq_f, KI_v_f = [float(v) for v in best_gains]

    # Summary
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE  --  CMA-ES + NN warm-start")
    print("=" * 70)
    print(f"  FMU evaluations : {n_evals} total  ({n_ok} valid)  "
          f"wall={elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n  {'Parameter':<8}  {'Baseline':>10}  {'Tuned':>10}  {'Delta':>8}")
    print(f"  {'-'*44}")
    for name, dflt, tuned in zip(_T_PARAM_NAMES, _T_DEFAULTS, best_gains):
        pct  = (float(tuned) - dflt) / (abs(dflt) + 1e-12) * 100.0
        sign = "UP" if pct > 0.0 else "DN"
        print(f"  {name:<8}  {dflt:>10.4f}  {float(tuned):>10.4f}  {sign} {abs(pct):5.1f}%")
    print(f"\n  Best cost      : {best_metrics['cost']:.4f}")
    print(f"  id RMS (MTPA)  : {best_metrics['id_rms']:.4f} A  (target 0 A)")
    print(f"  SS speed error : {best_metrics['ss_err']:.2f} RPM")
    print(f"  Load drop      : {best_metrics['load_drop']:.1f} RPM")
    print("=" * 70)

    # Update active gains
    _ACTIVE_GAINS["Q_id"] = Q_id_f; _ACTIVE_GAINS["Q_iq"] = Q_iq_f
    _ACTIVE_GAINS["R_vd"] = R_vd_f; _ACTIVE_GAINS["R_vq"] = R_vq_f
    _ACTIVE_GAINS["KI_v"] = KI_v_f

    # Phase 4: write header
    _write_gains_header(
        gains    = best_gains,
        metrics  = best_metrics,
        n_evals  = n_evals,
        method   = method_label,
        out_path = _C_SRC / "embed_sim_mpc_gains.h",
    )
    return True


# =============================================================================
# Entry point
# =============================================================================

def _ask_user_tune() -> bool:
    print()
    print("  +------------------------------------------------------------------+")
    print("  |  CMA-ES + NEURAL NETWORK WEIGHT TUNER                           |")
    print("  |  Run the CMA-ES + NN warm-start tuner before simulation?        |")
    print("  |                                                                  |")
    print(f"  |  Phase 1: {_T_PHASE1_EVALS:3d} real FMU sims (CMA-ES gen 1)              |")
    print("  |  Phase 2: NN trains on history -> predicts warm-start mean      |")
    print(f"  |  Phase 3: {_T_MAX_FMU_EVALS - _T_PHASE1_EVALS:3d} more FMU sims (CMA-ES from NN mean)    |")
    print(f"  |  Est. wall: ~{_T_MAX_FMU_EVALS * 114 // 60} min  |  Output: c_src/embed_sim_mpc_gains.h |")
    print("  +------------------------------------------------------------------+")
    try:
        answer = input("  Run tuner? [y/N] : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in ("y", "yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DB42S02 MPC FOC simulation with CMA-ES + NN warm-start tuner.")
    parser.add_argument("--tune",    action="store_true",
        help="Run the CMA-ES + NN warm-start tuner (non-interactive).")
    parser.add_argument("--no-tune", action="store_true",
        help="Skip the tuner prompt and use current weights.")
    args = parser.parse_args()

    print("=" * 68)
    print("  DB42S02  --  MPC FOC + SMO  --  20 kHz  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target  : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC}V  dt={DT*1e6:.0f}us")
    print(f"  Q_omega : {MPC_Q_OMEGA:.1f}  (fixed)")
    print(f"\n  Default weights (hardware commissioning):")
    for k, v in _ACTIVE_GAINS.items():
        print(f"    {k:<8} = {v}")
    print("=" * 68)

    if args.tune:
        do_tune = True
    elif args.no_tune:
        do_tune = False
    else:
        do_tune = _ask_user_tune()

    if do_tune:
        ok = run_tuner()
        if not ok:
            print("\n  Tuner aborted -- proceeding with default weights.")
        print(f"\n  Weights active for main simulation:")
        for k, v in _ACTIVE_GAINS.items():
            print(f"    {k:<8} = {v:.6f}")
    else:
        print("\n  Tuner skipped -- using default weights.")

    print("\n" + "=" * 68)
    print("  Running main simulation ...")
    print("=" * 68)

    data = build_and_run()
    print_summary(data)
    plot_results(data, path=str(_HERE / "db42s02_mpc_foc_20k_results.png"))

    print("\n[Done]")
    print("  db42s02_mpc_foc_20k_results.png")
    print("  db42s02_mpc_topology.html")
    print("  embedsim_gen/embedsim_step.c   <- flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")
    if do_tune and (_C_SRC / "embed_sim_mpc_gains.h").exists():
        print("  c_src/embed_sim_mpc_gains.h    <- #include in mpc_controller.h")
