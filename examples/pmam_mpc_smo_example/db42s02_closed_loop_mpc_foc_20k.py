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
  0.5–1.2 s  : 5 mN·m
  1.2–5.0 s  : 20 mN·m

CodeGen  →  embedsim_gen/embedsim_step.c / .h

INTEGRATED NN SURROGATE WEIGHT TUNER
======================================
When run with --tune (or the user answers 'y' at the prompt), the NN
surrogate tuner executes before the main simulation:

    Phase 1  LHS exploration  : _T_N_EXPLORE clean-plant simulations
    Phase 2  MLP training     : 5->16->16->1  (pure NumPy, Adam)
    Phase 3  Surrogate opt.   : gradient descent, _T_N_RESTARTS restarts
    Phase 4  Verification     : one real simulation at recommended weights
    Phase 5  Header write     : embed_sim_mpc_gains_tuned.h

On completion the tuner updates _ACTIVE_GAINS so the subsequent main
simulation uses the tuned weights.

Outputs:
  db42s02_mpc_foc_20k_results.png
  db42s02_mpc_topology.html
  embedsim_gen/embedsim_step.c / .h   (CodeGen)
  embed_sim_mpc_gains_tuned.h         (only when --tune requested)
"""

from __future__ import annotations

import sys
import math
import time
import argparse
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

V_DC       = 17.0       # [V]
TARGET_RPM = 2000.0     # [RPM]
T_SIM      = 5.0        # [s]   main simulation duration
DT         = 50e-6      # [s]   20 kHz — matches AURIX GTM period
_RAMP_TIME = 0.5        # [s]   linear speed ramp

# Load schedule (simulation only — not present on hardware)
T_LOAD_T1    = 0.5     # [s]   light load starts
T_LOAD_T2    = 1.2     # [s]   heavy load starts
T_LOAD_ZERO  = 0.000   # [N·m]
T_LOAD_LIGHT = 0.005   # [N·m]  5 mN·m
T_LOAD_HEAVY = 0.020   # [N·m]  20 mN·m

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0
_MOTOR_OUT_SIZE  = 8    # [rpm(0), ia(1), ib(2), ic(3), theta_m(4), T_em(5), id(6), iq(7)]

# =============================================================================
# Sensor noise configuration
# =============================================================================
# ADC current noise: 12-bit AURIX, ±I_MAX range → LSB ≈ 1.74 mA; sigma=0.01 A
# Encoder quantisation: 1000 PPR × 4 → 4000 cnt/rev; sigma ≈ 0.78 mrad

ENABLE_NOISE = False   # ← set True for Phase 2 / Phase 3 noise validation
NOISE_SEED   = 42      # reproducible runs; set None for random

# =============================================================================
# MPC weight constants  (read by build_and_run() via _ACTIVE_GAINS)
# =============================================================================
#   Q_omega = 500.0 is fixed — it is the dominant speed-tracking weight and
#   its established torque accuracy (iq_ss ≈ 2.41 A for 20 mN·m) must not
#   be disturbed by the tuner.
#
#   Tuned parameters: Q_id, Q_iq, R_vd, R_vq, KI_v
#
#   Q_id    — d-axis state cost (drives id → 0, MTPA)
#   Q_iq    — q-axis regulariser (vq denominator only, must be << Q_omega)
#   R_vd    — vd effort weight  (conservative vd commands)
#   R_vq    — vq effort weight  (damps cross-coupling overshoot)
#   KI_v    — speed-error integral gain (eliminates SS speed offset)

MPC_N       = 10       # prediction horizon (500 µs at 20 kHz)
MPC_Q_OMEGA = 500.0    # speed tracking weight (FIXED — not tuned)
MPC_SMO_K   = 4.68     # SMO switching gain [V]
MPC_SMO_FC  = 1000.0   # SMO back-EMF LPF corner [Hz]

# Hardware-commissioning baseline weights (used when tuner is skipped)
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

        fmu_in = VectorSignal(
            np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))
        return super().compute_py(t, dt, [fmu_in])

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# CodeGen-configured SVPWM subclasses
# =============================================================================

class DB42S02SVPWMPackBlock(SVPWMPackBlock):
    """SVPWMPackBlock with DB42S02 / AURIX CodeGen metadata."""

    C_SOURCES    = ["embed_sim_motor_utility_blocks.c"]
    C_HEADERS    = ["embed_sim_motor_utility_blocks.h"]
    state_struct = "SVPWMPack_T"
    step_func    = "SVPWMPack_Step"
    init_func    = "SVPWMPack_Init"
    C_INIT_ARGS  = ["v_dc"]
    C_CUSTOM_EMIT = (
        "    /* --- svpwm_pack (SVPWMPackBlock) --- */\n"
        "    {\n"
        "        SVPWMPack_T  svpwm_pack_st;\n"
        "        real32_T     y_svpwm_pack[3];\n"
        "        real32_T     u_svpwm_pack[2];\n"
        "        u_svpwm_pack[0] = y_mpc[0];   /* v_alpha */\n"
        "        u_svpwm_pack[1] = y_mpc[1];   /* v_beta  */\n"
        "        SVPWMPack_Init(&svpwm_pack_st, 17.0f);\n"
        "        SVPWMPack_Step(&svpwm_pack_st, u_svpwm_pack, dt, y_svpwm_pack);\n"
        "    }"
    )


class DB42S02SVPWMBlock(SVPWMBlock):
    """SVPWMBlock with DB42S02 / AURIX CodeGen metadata."""

    C_SOURCES = ["embed_sim_sv_pwm.c"]
    C_HEADERS = ["embed_sim_sv_pwm.h"]
    C_CUSTOM_EMIT = (
        "    /* --- svpwm (SVPWMBlock) --- */\n"
        "    {\n"
        "        SVM_DutyCycle_Type  svm_duty;\n"
        "        real32_T            y_svpwm[4];\n"
        "        SVM_CalculateDutyCycle(y_svpwm_pack[0],\n"
        "                               y_svpwm_pack[1],\n"
        "                               &svm_duty);\n"
        "        SVM_GetDutyCyclesFloat(&svm_duty,\n"
        "                               &y_svpwm[0], &y_svpwm[1], &y_svpwm[2]);\n"
        "        y_svpwm[3] = (real32_T)svm_duty.sector;\n"
        "    }"
    )


# =============================================================================
# _run_sim — shared by tuner (no CodeGen) and build_and_run (with CodeGen)
# =============================================================================

def _run_sim(
    *,
    with_codegen_hooks: bool = True,
    t_sim:              float = T_SIM,
) -> dict | None:
    """
    Build and run one closed-loop MPC simulation.

    Parameters
    ----------
    with_codegen_hooks : bool
        True  — include CodeGenStart/End (main run).
        False — lighter wiring without CodeGen objects (tuner).
    t_sim : float
        Simulation duration [s].  Tuner uses 3 s; main run uses T_SIM.

    Returns
    -------
    dict | None
        Result dictionary or None on failure.
    """
    try:
        mpc = MPCControllerBlock(
            name          = "mpc",
            P_POLES       = MPC_MOTOR.P_POLES,
            R_S           = MPC_MOTOR.R_S,
            L             = MPC_MOTOR.L_D,
            LAMBDA_PM     = MPC_MOTOR.LAMBDA_PM,
            J             = MPC_MOTOR.J_ROTOR,
            B             = MPC_MOTOR.B_FRICTION,
            I_MAX         = MPC_MOTOR.I_MAX,
            V_DC          = V_DC,
            N             = MPC_N,
            Q_id          = _ACTIVE_GAINS["Q_id"],
            Q_iq          = _ACTIVE_GAINS["Q_iq"],
            Q_omega       = MPC_Q_OMEGA,
            R_vd          = _ACTIVE_GAINS["R_vd"],
            R_vq          = _ACTIVE_GAINS["R_vq"],
            dt_s          = DT,
            SMO_K         = MPC_SMO_K,
            SMO_FC        = MPC_SMO_FC,
            KI_v          = _ACTIVE_GAINS["KI_v"],
            SOFTSTART_T   = 0.1,
            use_c_backend = False,
        )

        svpwm_pack  = DB42S02SVPWMPackBlock("svpwm_pack", v_dc=V_DC, use_c_backend=False)
        svpwm       = DB42S02SVPWMBlock("svpwm", use_c_backend=False)
        motor       = DB42S02PlantBlock("motor")
        motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
        speed_ref   = VectorStep("speed_ref", step_time=0.0,
                                 before_value=TARGET_RADS_MECH,
                                 after_value=TARGET_RADS_MECH)
        ctrl = CtrlPacker(
            "ctrl_packer",
            target_rads_mech = TARGET_RADS_MECH,
            ramp_time        = _RAMP_TIME,
            rng_seed         = NOISE_SEED,
        )
        ctrl.set_noise_enabled(ENABLE_NOISE if with_codegen_hooks else False)

        sink    = VectorEnd("sink")
        sink_cg = VectorEnd("sink_cg")

        # ── Wiring ─────────────────────────────────────────────────────────
        if with_codegen_hooks:
            cg_start = CodeGenStart("cg_start")
            cg_end   = CodeGenEnd("cg_end")
            cg_start >> mpc >> svpwm_pack >> svpwm >> cg_end
            ctrl     >> cg_start
            svpwm    >> motor
            svpwm    >> sink_cg
            cg_end   >> sink_cg
        else:
            cg_start = cg_end = None
            mpc >> svpwm_pack >> svpwm
            ctrl >> mpc
            svpwm >> motor
            svpwm >> sink_cg

        motor       >> motor_delay
        motor_delay >> ctrl
        speed_ref   >> ctrl
        motor       >> sink

        # ── Simulate ───────────────────────────────────────────────────────
        sim = EmbedSim(sinks=[sink, sink_cg], T=t_sim, dt=DT,
                       solver=ODESolver.RK4)

        if with_codegen_hooks:
            print("\n[Topology]")
            sim.topo.print_console()
            sim.topo.export_html(
                str(_HERE / "db42s02_mpc_topology.html"),
                wire_labels={
                    ("speed_ref",   "ctrl_packer"): "ω_ref [rad/s]",
                    ("motor",       "motor_delay"): "FMU feedback",
                    ("motor_delay", "ctrl_packer"): "z⁻¹ feedback",
                    ("ctrl_packer", "cg_start"):    "[ω_ref,θ_m,ia,ib,ic]",
                    ("cg_start",    "mpc"):         "sensor inputs",
                    ("mpc",         "svpwm_pack"):  "[v_α,v_β]",
                    ("svpwm_pack",  "svpwm"):       "[Vref,α]",
                    ("svpwm",       "cg_end"):      "[ta,tb,tc,sector]",
                    ("svpwm",       "motor"):       "[ta,tb,tc,sector]",
                })

        sim.scope.add(mpc,        indices=[0, 1],       label="Vab")
        sim.scope.add(svpwm_pack, indices=[0],           label="Vref")
        sim.scope.add(svpwm,      indices=[0, 1, 2, 3], label="Duties")
        sim.scope.add(motor,      indices=[0, 5, 6, 7], label="Motor")

        print("\nRunning simulation…")
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
        arr = ld.get(key, [])
        t_ld = ld.get("t", [])
        if len(t_ld) > 1:
            return np.interp(t, t_ld, arr).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)

    def _m(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), dtype=np.float32)

    return {
        "t":             t,
        "speed_rpm":     _m(0),
        "omega_ref_rpm": _i("speed_ref"),
        "iq":            _i("iq"),
        "id":            _i("id"),
        "v_alpha":       _s("Vab",    0),
        "v_beta":        _s("Vab",    1),
        "vref":          _s("Vref",   0),
        "ta":            _s("Duties", 0),
        "tb":            _s("Duties", 1),
        "tc":            _s("Duties", 2),
        "sector":        _s("Duties", 3),
        "torque":        _m(1),
        "id_plant":      _m(2),
        "iq_plant":      _m(3),
        "_cg_start":     cg_start,
        "_cg_end":       cg_end,
        "_sim":          sim,
    }


# =============================================================================
# build_and_run — main simulation entry point (with CodeGen)
# =============================================================================

def build_and_run() -> dict:
    """Run the full closed-loop simulation and emit AURIX C code."""

    print("=" * 68)
    print("  NANOTEC DB42S02  —  MPC FOC + SMO  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC}V  "
          f"dt={DT*1e6:.0f}µs  T_sim={T_SIM}s")
    print(f"  MPC    : N={MPC_N}  Q_id={_ACTIVE_GAINS['Q_id']:.4f}  "
          f"Q_iq={_ACTIVE_GAINS['Q_iq']:.4f}  Q_omega={MPC_Q_OMEGA:.1f}  "
          f"R_vd={_ACTIVE_GAINS['R_vd']:.4f}  R_vq={_ACTIVE_GAINS['R_vq']:.4f}  "
          f"KI_v={_ACTIVE_GAINS['KI_v']:.4f}")
    print(f"  SMO    : k={MPC_SMO_K:.2f} V  fc={MPC_SMO_FC:.0f} Hz")
    print(f"  Load   : 0 → {T_LOAD_LIGHT*1e3:.0f} mN·m @ {T_LOAD_T1}s"
          f" → {T_LOAD_HEAVY*1e3:.0f} mN·m @ {T_LOAD_T2}s")
    print(f"  Noise  : {'ENABLED (seed=' + str(NOISE_SEED) + ')' if ENABLE_NOISE else 'DISABLED'}")
    print("=" * 68)

    d = _run_sim(with_codegen_hooks=True, t_sim=T_SIM)
    if d is None:
        raise RuntimeError("Main simulation failed.")

    # ── CodeGen ────────────────────────────────────────────────────────────────
    cg_start = d["_cg_start"]
    cg_end   = d["_cg_end"]
    if cg_start and cg_end:
        print("\n[CodeGen] Generating AURIX C code…")
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
            print(f"  Input_T  : omega_ref_mech, theta_m, ia, ib, ic")
            print(f"  Output_T : ta, tb, tc, sector")

    return d


# =============================================================================
# Plot
# =============================================================================

def plot_results(d: dict,
                 path: str = "db42s02_mpc_foc_20k_results.png") -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        f"NANOTEC DB42S02 — MPC FOC + SMO  |  {TARGET_RPM:.0f} RPM  |  20 kHz"
        + ("  |  NOISE ON" if ENABLE_NOISE else "  |  no noise"),
        fontsize=12, fontweight="bold")
    t = d["t"]

    ax = axes[0, 0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.5, label="ω_ref")
    ax.plot(t, d["speed_rpm"],     "C0",  lw=1.5, label="ω_actual")
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red",    ls=":", lw=1.0)
    ax.set_ylabel("Speed [RPM]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Speed tracking"); ax.set_xlabel("t [s]")

    ax = axes[0, 1]
    ax.plot(t, d["speed_rpm"] - d["omega_ref_rpm"], "C1", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red",    ls=":", lw=1.0)
    ax.set_ylabel("Error [RPM]"); ax.grid(alpha=0.3)
    ax.set_title("Speed error"); ax.set_xlabel("t [s]")

    ax = axes[1, 0]
    ax.plot(t, d["iq_plant"], "k--", lw=1.2, label="iq (plant)")
    ax.plot(t, d["iq"],       "C0",  lw=1.0, label="iq (SMO est)")
    ax.plot(t, d["id"],       "C5",  lw=1.0, label="id (SMO est)")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline( MPC_MOTOR.I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-MPC_MOTOR.I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_ylabel("Current [A]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("dq currents — plant vs SMO estimate  (id_ref=0 MTPA)")
    ax.set_xlabel("t [s]")

    ax = axes[1, 1]
    ax.plot(t, d["id"], "C5", lw=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("id [A]"); ax.grid(alpha=0.3)
    ax.set_title("id  (should ≈ 0  —  MTPA)"); ax.set_xlabel("t [s]")

    ax = axes[2, 0]
    ax.plot(t, d["v_alpha"], "C0", lw=0.8, label="v_α")
    ax.plot(t, d["v_beta"],  "C1", lw=0.8, label="v_β")
    ax.set_ylabel("Voltage [V]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Stator voltage commands"); ax.set_xlabel("t [s]")

    ax = axes[2, 1]
    ax.plot(t, d["vref"], "C5", lw=0.8)
    ax.axhline(0.95, color="red", ls="--", lw=0.8, alpha=0.7, label="clip=0.95")
    ax.set_ylabel("Vref [norm]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("SVPWM modulation index"); ax.set_xlabel("t [s]")

    ax = axes[3, 0]
    ax.plot(t, d["ta"], "C3", lw=0.7, label="ta")
    ax.plot(t, d["tb"], "C2", lw=0.7, label="tb")
    ax.plot(t, d["tc"], "C1", lw=0.7, label="tc")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("Duty")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("SVPWM duties"); ax.set_xlabel("t [s]")

    ax = axes[3, 1]
    ax.plot(t, d["torque"] * 1000, "C4", lw=0.8, label="T_em")
    ax.axhline(T_LOAD_LIGHT * 1000, color="orange", ls=":", lw=1.0,
               alpha=0.7, label=f"{T_LOAD_LIGHT*1e3:.0f} mN·m")
    ax.axhline(T_LOAD_HEAVY * 1000, color="red",    ls=":", lw=1.0,
               alpha=0.7, label=f"{T_LOAD_HEAVY*1e3:.0f} mN·m")
    ax.set_ylabel("Torque [mN·m]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Electromagnetic torque"); ax.set_xlabel("t [s]")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(d: dict) -> None:
    t   = d["t"]
    rpm = d["speed_rpm"]
    ref = d["omega_ref_rpm"]
    n   = len(t)
    ss  = int(0.80 * n)

    ss_err  = float(np.mean(np.abs(rpm[ss:] - ref[ss:])))
    id_rms  = float(np.sqrt(np.mean(d["id"][ss:] ** 2)))
    vref_mx = float(np.max(d["vref"]))
    iq_ss   = float(np.mean(np.abs(d["iq"][ss:])))

    after = t > T_LOAD_T2
    pre   = (t >= T_LOAD_T2 - 0.1) & (t < T_LOAD_T2)
    drop  = 0.0
    if np.any(after) and np.any(pre):
        drop = max(0.0, float(np.mean(rpm[pre])) - float(np.mean(rpm[after][:50])))

    print(f"\n{'='*60}")
    print("  MPC FOC + SMO — Performance Summary")
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
    print(f"{'='*60}")


# =============================================================================
# =============================================================================
#
#   INTEGRATED NN SURROGATE WEIGHT TUNER
#
#   All tuner code lives here so it can share _run_sim(), _ACTIVE_GAINS,
#   and the motor constants directly — no monkey-patching, no separate module.
#
# =============================================================================
# =============================================================================

# ── Tuner hyper-parameters ────────────────────────────────────────────────────
_T_N_EXPLORE   = 40     # LHS simulations (exploration phase)
_T_N_EPOCHS    = 500    # MLP training epochs
_T_N_RESTARTS  = 8      # gradient-descent restarts on surrogate
_T_LR_SURR     = 3e-3   # Adam lr — surrogate training
_T_LR_OPT      = 5e-2   # Adam lr — surrogate optimisation
_T_N_OPT_STEPS = 400    # steps per optimisation restart
_T_SIM_DURATION = 3.0   # [s] tuner simulation duration (covers both load steps)

# ── Cost weights ──────────────────────────────────────────────────────────────
#   Priority: id_rms (MTPA) >> ss_err (speed) > load_drop (transient)
#   W_ID = 80 ensures id oscillation dominates the cost surface; with the
#   current ±1.1 A id it contributes 80×1.1 = 88 vs 2×27 = 54 for speed.
_T_W_ID   = 80.0   # id RMS in steady state [A]
_T_W_SS   =  2.0   # SS speed error [RPM]
_T_W_DROP =  0.5   # load-step speed drop [RPM]
_T_W_VREF = 20.0   # over-modulation flat penalty

# ── Physics-derived parameter bounds ─────────────────────────────────────────
#   Q_id  : [5, 50]   — below 5 id diverges; above 50 competes with Q_omega
#   Q_iq  : [0.01, 1] — regularisation only; > 1.0 degrades iq tracking
#   R_vd  : [0.001, 0.1]  — conservative vd commands
#   R_vq  : [0.005, 0.2]  — damps cross-coupling overshoot
#   KI_v  : [0.005, 0.1]  — integral correction; > 0.1 risks windup at light load
_T_BOUNDS = np.array([
    [  5.0,  50.0],   # Q_id
    [  0.01,  1.0],   # Q_iq
    [  0.001, 0.10],  # R_vd
    [  0.005, 0.20],  # R_vq
    [  0.005, 0.10],  # KI_v
], dtype=np.float64)

_T_PARAM_NAMES = ["Q_id", "Q_iq", "R_vd", "R_vq", "KI_v"]

# Hardware-commissioning baseline (for before/after summary table)
_T_DEFAULTS = [
    _ACTIVE_GAINS["Q_id"],
    _ACTIVE_GAINS["Q_iq"],
    _ACTIVE_GAINS["R_vd"],
    _ACTIVE_GAINS["R_vq"],
    _ACTIVE_GAINS["KI_v"],
]


# ── Cost function ─────────────────────────────────────────────────────────────

def _t_cost(d: dict | None) -> dict | None:
    """
    Compute scalar cost from a _run_sim() result dict.

    Returns None on divergence or insufficient data.
    """
    if d is None:
        return None

    t   = d["t"]
    rpm = d["speed_rpm"]
    idd = d["id"]

    if len(t) < 200:
        return None

    if float(np.max(np.abs(rpm))) > TARGET_RPM * 3.0:
        return None

    # Steady-state: last 15 % of simulation
    ss = t > 0.85 * _T_SIM_DURATION
    if not np.any(ss):
        return None

    ref_ss = float(np.mean(d["omega_ref_rpm"][ss]))
    ss_err = float(np.mean(np.abs(rpm[ss] - ref_ss)))
    id_rms = float(np.sqrt(np.mean(idd[ss] ** 2)))

    if ss_err > 800.0:
        return None

    # Load-step drop at T_LOAD_T2
    after  = t >= T_LOAD_T2
    before = (t >= T_LOAD_T2 - 0.1) & (t < T_LOAD_T2)
    load_drop = 0.0
    if np.any(after) and np.any(before):
        load_drop = max(0.0, float(np.mean(rpm[before])) - float(np.mean(rpm[after][:50])))

    cost = _T_W_ID * id_rms + _T_W_SS * ss_err + _T_W_DROP * load_drop

    # Over-modulation penalty
    vref = d.get("vref")
    if vref is not None and len(vref) > 0:
        if float(np.percentile(np.abs(vref), 90)) >= 0.93:
            cost += _T_W_VREF

    return {"cost": cost, "id_rms": id_rms, "ss_err": ss_err, "load_drop": load_drop}


def _t_run_with_gains(
    Q_id: float, Q_iq: float, R_vd: float, R_vq: float, KI_v: float,
) -> dict | None:
    """
    Run one tuner simulation with the given weights.

    Temporarily patches _ACTIVE_GAINS so _run_sim() picks them up.
    Restores original gains on exit (even on exception).
    """
    saved = _ACTIVE_GAINS.copy()
    try:
        _ACTIVE_GAINS["Q_id"]  = Q_id
        _ACTIVE_GAINS["Q_iq"]  = Q_iq
        _ACTIVE_GAINS["R_vd"]  = R_vd
        _ACTIVE_GAINS["R_vq"]  = R_vq
        _ACTIVE_GAINS["KI_v"]  = KI_v
        return _t_cost(_run_sim(with_codegen_hooks=False, t_sim=_T_SIM_DURATION))
    finally:
        _ACTIVE_GAINS.update(saved)


# ── Latin-Hypercube sampling ──────────────────────────────────────────────────

def _t_lhs(n: int, bounds: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Latin-Hypercube sample: n points in d dimensions.

    Each dimension is independently stratified into n equal-width bins;
    within each bin one sample is drawn uniformly at random.  The bin
    order is then shuffled independently per dimension so the joint
    distribution has no column correlation.
    """
    d    = bounds.shape[0]
    cuts = np.linspace(0.0, 1.0, n + 1)
    X    = np.zeros((n, d))
    for j in range(d):
        u = rng.uniform(cuts[:-1], cuts[1:])
        rng.shuffle(u)
        X[:, j] = bounds[j, 0] + u * (bounds[j, 1] - bounds[j, 0])
    return X


# ── Minimal MLP surrogate (pure NumPy — no PyTorch / TF dependency) ──────────

class _MLP:
    """
    2-hidden-layer MLP trained with Adam, pure NumPy.

    Architecture: n_in → 16 → 16 → 1   (tanh activations, linear output)

    At n_in=5, hidden=16:
        parameters = 16*5+16 + 16*16+16 + 1*16+1 = 369
    With N_EXPLORE = 40 samples the network is under-determined relative to
    training set size — implicit regularisation via Adam + smooth cost surface.
    Increasing hidden beyond 16 risks memorisation for this budget.
    """

    def __init__(self, n_in: int = 5, hidden: int = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        def _w(r, c): return rng.standard_normal((r, c)) * np.sqrt(2.0 / c)
        self.n_in = n_in
        self.W1 = _w(hidden, n_in);    self.b1 = np.zeros(hidden)
        self.W2 = _w(hidden, hidden);  self.b2 = np.zeros(hidden)
        self.W3 = _w(1, hidden);       self.b3 = np.zeros(1)
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]
        self._t = 0

    def _params(self): return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, X):
        h1 = np.tanh(X  @ self.W1.T + self.b1)
        h2 = np.tanh(h1 @ self.W2.T + self.b2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1)

    def _fwd_cache(self, X):
        z1 = X  @ self.W1.T + self.b1;  h1 = np.tanh(z1)
        z2 = h1 @ self.W2.T + self.b2;  h2 = np.tanh(z2)
        return (h2 @ self.W3.T + self.b3).squeeze(-1), h1, h2

    def loss_and_grad(self, X, y_true):
        N         = X.shape[0]
        y, h1, h2 = self._fwd_cache(X)
        err       = y - y_true
        L         = float(np.mean(err ** 2))
        dL_dy = 2.0 * err / N
        dh2   = dL_dy[:, None] * self.W3
        dW3   = (dL_dy[:, None] * h2).mean(0, keepdims=True)
        db3   = dL_dy.mean(0, keepdims=True)
        dz2 = dh2 * (1.0 - h2 ** 2)
        dW2 = dz2.T @ h1 / N
        db2 = dz2.sum(0) / N
        dh1 = dz2 @ self.W2
        dz1 = dh1 * (1.0 - h1 ** 2)
        dW1 = dz1.T @ X / N
        db1 = dz1.sum(0) / N
        return L, [dW1, db1, dW2, db2, dW3, db3]

    def _adam(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        t = self._t
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._m[i] = beta1 * self._m[i] + (1 - beta1) * g
            self._v[i] = beta2 * self._v[i] + (1 - beta2) * g ** 2
            m_hat = self._m[i] / (1 - beta1 ** t)
            v_hat = self._v[i] / (1 - beta2 ** t)
            p    -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def train(self, X, y, epochs=500, lr=3e-3, verbose=True):
        losses = []
        for ep in range(epochs):
            L, grads = self.loss_and_grad(X, y)
            self._adam(grads, lr=lr)
            losses.append(L)
            if verbose and (ep % 100 == 0 or ep == epochs - 1):
                print(f"    epoch {ep:4d}  MSE={L:.6f}")
        return losses

    def scalar_grad(self, x):
        """Forward + ∂output/∂input (exact backprop).  x shape: (n_in,)."""
        X         = x[None, :]
        y, h1, h2 = self._fwd_cache(X)
        dh2  = np.ones((1, self.W3.shape[1])) * self.W3
        dz2  = dh2 * (1.0 - h2 ** 2)
        dh1  = dz2 @ self.W2
        dz1  = dh1 * (1.0 - h1 ** 2)
        dx   = (dz1 @ self.W1).squeeze(0)
        return float(y.squeeze()), dx


# ── Surrogate optimisation ────────────────────────────────────────────────────

def _t_optimise(mlp: _MLP, X_norm: np.ndarray, y_norm: np.ndarray,
                rng: np.random.Generator) -> np.ndarray:
    """
    Find x in [0,1]^5 that minimises mlp.forward(x).

    Runs _T_N_RESTARTS independent Adam gradient-descent trajectories.
    Restart 0 starts from the best observed point in X_norm.
    Returns the normalised gain vector with the lowest predicted cost.
    """
    n_in      = mlp.n_in
    best_x    = X_norm[int(np.argmin(y_norm))].copy()
    best_cost = float(mlp.forward(best_x[None, :])[0])

    for restart in range(_T_N_RESTARTS):
        x = best_x.copy() if restart == 0 else rng.uniform(0.0, 1.0, size=n_in)
        m_i = np.zeros(n_in); v_i = np.zeros(n_in); t_i = 0
        b1, b2, eps = 0.9, 0.999, 1e-8
        for _ in range(_T_N_OPT_STEPS):
            cost, grad = mlp.scalar_grad(x)
            t_i  += 1
            m_i   = b1 * m_i + (1 - b1) * grad
            v_i   = b2 * v_i + (1 - b2) * grad ** 2
            m_hat = m_i / (1 - b1 ** t_i)
            v_hat = v_i / (1 - b2 ** t_i)
            x     = x - _T_LR_OPT * m_hat / (np.sqrt(v_hat) + eps)
            x     = np.clip(x, 0.0, 1.0)
        c = float(mlp.forward(x[None, :])[0])
        if c < best_cost:
            best_cost = c; best_x = x.copy()

    return best_x


# ── gains.h writer ────────────────────────────────────────────────────────────

def _write_gains_header(
    Q_id: float, Q_iq: float, R_vd: float, R_vq: float, KI_v: float,
    cost: float, id_rms: float, ss_err: float, load_drop: float,
    n_explore: int, out_path: Path,
) -> None:
    """Write embed_sim_mpc_gains_tuned.h with full ISO 26262 audit trail."""
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    content = textwrap.dedent(f"""\
    /**********************************************************************************************************************
     * \\file      embed_sim_mpc_gains_tuned.h
     * \\brief     MPC FOC — NN-tuned weight constants for NANOTEC DB42S02
     *
     * \\details   AUTO-GENERATED by db42s02_closed_loop_mpc_foc_20k.py.  DO NOT EDIT MANUALLY.
     *
     *            TUNING AUDIT TRAIL
     *            ===================
     *            Generated   : {now}
     *            Method      : Latin-Hypercube ({n_explore} samples) + MLP 5->16->16->1
     *                          + Adam gradient descent ({_T_N_RESTARTS} restarts x {_T_N_OPT_STEPS} steps)
     *            Noise model : DISABLED during exploration (clean plant, deterministic surface)
     *            Target      : {TARGET_RPM:.0f} RPM  V_DC={V_DC:.1f}V  dt={DT*1e6:.0f}µs
     *            Fixed params: Q_omega={MPC_Q_OMEGA:.1f}  N={MPC_N}  SMO_K={MPC_SMO_K:.2f}  SMO_FC={MPC_SMO_FC:.0f}
     *
     *            COST FUNCTION
     *            ==============
     *            cost = {_T_W_ID:.1f}  * id_rms_A          (d-axis MTPA, target 0)
     *                 + {_T_W_SS:.1f}  * ss_err_RPM         (speed tracking)
     *                 + {_T_W_DROP:.1f} * load_drop_RPM      (load rejection)
     *                 + {_T_W_VREF:.1f} * [1 if Vref_p90 >= 0.93]  (over-modulation)
     *
     *            VERIFICATION RESULT (clean plant)
     *            ==================================
     *            Total cost     : {cost:.4f}
     *            id RMS (MTPA)  : {id_rms:.4f} A    (target 0)
     *            SS speed error : {ss_err:.2f} RPM
     *            Load drop      : {load_drop:.1f} RPM  at t={T_LOAD_T2}s
     *
     * \\note      MISRA C:2012 — Rule 7.2: float literals carry 'f' suffix.
     *
     * \\version   1.0.0  (auto-generated)
     * \\copyright Copyright (C) EmbedSim 2026
     *********************************************************************************************************************/

    #ifndef EMBED_SIM_MPC_GAINS_TUNED_H_
    #define EMBED_SIM_MPC_GAINS_TUNED_H_

    #include "embed_sim_matrix.h"    /* MatrixFloat = real32_T */

    /** \\defgroup MPC_Gains_Tuned  NN-tuned MPC weight constants  \\{{ */

    #define MPC_Q_ID     ((MatrixFloat){Q_id:.6f}f)   /**< d-axis state cost (was {_T_DEFAULTS[0]:.4f}) */
    #define MPC_Q_IQ     ((MatrixFloat){Q_iq:.6f}f)   /**< q-axis regulariser (was {_T_DEFAULTS[1]:.4f}) */
    #define MPC_R_VD     ((MatrixFloat){R_vd:.6f}f)   /**< vd effort weight  (was {_T_DEFAULTS[2]:.4f}) */
    #define MPC_R_VQ     ((MatrixFloat){R_vq:.6f}f)   /**< vq effort weight  (was {_T_DEFAULTS[3]:.4f}) */
    #define MPC_KI_V     ((MatrixFloat){KI_v:.6f}f)   /**< speed integral    (was {_T_DEFAULTS[4]:.4f}) */
    #define MPC_Q_OMEGA  ((MatrixFloat){MPC_Q_OMEGA:.1f}f)  /**< speed cost (FIXED) */

    /** \\}} */

    typedef struct {{
        MatrixFloat Q_id;     /**< {Q_id:.6f} */
        MatrixFloat Q_iq;     /**< {Q_iq:.6f} */
        MatrixFloat R_vd;     /**< {R_vd:.6f} */
        MatrixFloat R_vq;     /**< {R_vq:.6f} */
        MatrixFloat KI_v;     /**< {KI_v:.6f} */
        MatrixFloat Q_omega;  /**< {MPC_Q_OMEGA:.1f}f (fixed) */
    }} MPC_GainSet_T;

    #endif /* EMBED_SIM_MPC_GAINS_TUNED_H_ */
    """)

    out_path.write_text(content, encoding="utf-8")
    print(f"\n  [gains.h] Written: {out_path}")


# ── Main tuner entry point ────────────────────────────────────────────────────

def run_tuner() -> bool:
    """
    Execute the full NN surrogate MPC weight tuner.

    Returns True if tuning completed and _ACTIVE_GAINS was updated.

    Phases
    ------
    1. LHS exploration  : _T_N_EXPLORE clean-plant simulations (3 s each).
    2. MLP training     : 5->16->16->1, Adam, _T_N_EPOCHS epochs.
    3. Surrogate opt.   : gradient descent, _T_N_RESTARTS restarts.
    4. Verification     : one clean simulation at recommended weights.
    5. Header write     : embed_sim_mpc_gains_tuned.h with audit trail.
    """
    rng   = np.random.default_rng(seed=42)
    n_dim = len(_T_PARAM_NAMES)

    print("\n" + "=" * 70)
    print("  MPC NN Surrogate Weight Tuner  —  NANOTEC DB42S02")
    print("=" * 70)
    print(f"  Phase 1  LHS exploration : {_T_N_EXPLORE} simulations  "
          f"(T_sim={_T_SIM_DURATION:.1f}s, ~{_T_N_EXPLORE*_T_SIM_DURATION/60:.0f} min est.)")
    print(f"  Phase 2  MLP training    : {_T_N_EPOCHS} epochs  ({n_dim}->16->16->1)")
    print(f"  Phase 3  Surrogate opt.  : {_T_N_RESTARTS} restarts x {_T_N_OPT_STEPS} steps")
    print(f"  Phase 4  Verification    : 1 simulation at recommended weights")
    print(f"  Phase 5  Header write    : embed_sim_mpc_gains_tuned.h")
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
    print(f"\n  Noise: DISABLED (clean plant for deterministic cost surface)")
    print("=" * 70)

    # ── Phase 1: LHS exploration ──────────────────────────────────────────────
    print(f"\n[Phase 1] LHS exploration ({_T_N_EXPLORE} simulations) ...")
    X_raw  = _t_lhs(_T_N_EXPLORE, _T_BOUNDS, rng)
    costs: list[float] = []
    valid_results: list[tuple[np.ndarray, dict]] = []

    t0 = time.perf_counter()
    for i, params in enumerate(X_raw):
        q_id, q_iq, r_vd, r_vq, ki_v = params
        print(f"  [{i+1:2d}/{_T_N_EXPLORE}]  "
              f"Q_id={q_id:6.2f}  Q_iq={q_iq:.3f}  "
              f"R_vd={r_vd:.4f}  R_vq={r_vq:.4f}  KI_v={ki_v:.4f}",
              end="  ", flush=True)
        try:
            met = _t_run_with_gains(q_id, q_iq, r_vd, r_vq, ki_v)
        except KeyboardInterrupt:
            print("\n  Interrupted — using data collected so far.")
            X_raw = X_raw[:i]
            break

        if met is None:
            print("-> UNSTABLE")
            costs.append(1e6)
        else:
            print(f"-> cost={met['cost']:.2f}  "
                  f"id={met['id_rms']:.3f}A  "
                  f"ss={met['ss_err']:.0f}RPM  "
                  f"drop={met['load_drop']:.0f}RPM")
            costs.append(met["cost"])
            valid_results.append((params.copy(), met))

    elapsed = time.perf_counter() - t0
    print(f"\n  Exploration done  ({elapsed:.0f} s, "
          f"{elapsed / max(1, len(X_raw)):.1f} s/sim)")

    costs_arr  = np.array(costs, dtype=np.float64)
    valid_mask = costs_arr < 1e5
    n_valid    = int(valid_mask.sum())

    if n_valid < 4:
        print(f"  ERROR: only {n_valid} valid simulations (need ≥ 4).")
        print("  Try widening bounds or increasing _T_N_EXPLORE.")
        return False

    X_valid = X_raw[valid_mask]
    y_valid = costs_arr[valid_mask]

    best_obs_idx   = int(np.argmin(y_valid))
    best_obs_gains = X_valid[best_obs_idx]
    best_obs_cost  = float(y_valid[best_obs_idx])
    best_obs_met   = valid_results[best_obs_idx][1]

    print(f"\n  Best observed:  cost={best_obs_cost:.4f}")
    for name, val in zip(_T_PARAM_NAMES, best_obs_gains):
        print(f"    {name:<8} = {val:.6f}")

    # Save LHS dataset for offline re-analysis
    npz_path = _HERE / "mpc_tuner_exploration.npz"
    np.savez(str(npz_path), X=X_valid, y=y_valid, param_names=_T_PARAM_NAMES)
    print(f"  [dataset] Saved: {npz_path}")

    # ── Phase 2: MLP training ─────────────────────────────────────────────────
    print(f"\n[Phase 2] Training MLP surrogate ({n_dim}->16->16->1) ...")

    X_norm = (X_valid - _T_BOUNDS[:, 0]) / (_T_BOUNDS[:, 1] - _T_BOUNDS[:, 0])
    y_mean = float(y_valid.mean())
    y_std  = float(max(y_valid.std(), 1e-8))
    y_norm = (y_valid - y_mean) / y_std

    n_val  = max(1, len(X_norm) // 5)
    idx    = rng.permutation(len(X_norm))
    X_tr, y_tr = X_norm[idx[n_val:]], y_norm[idx[n_val:]]
    X_va, y_va = X_norm[idx[:n_val]], y_norm[idx[:n_val]]

    mlp = _MLP(n_in=n_dim, hidden=16, seed=0)
    mlp.train(X_tr, y_tr, epochs=_T_N_EPOCHS, lr=_T_LR_SURR, verbose=True)

    y_va_pred = mlp.forward(X_va)
    ss_res    = float(np.var(y_va - y_va_pred))
    ss_tot    = float(np.var(y_va))
    r2_val    = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    print(f"\n  Surrogate R² (held-out {n_val} pts): {r2_val:.4f}  (>0.70 is acceptable)")
    if r2_val < 0.50:
        print("  WARNING: R² < 0.50 — surrogate quality poor.")
        print("  Consider increasing _T_N_EXPLORE before trusting Phase 3.")

    # ── Phase 3: surrogate optimisation ──────────────────────────────────────
    print(f"\n[Phase 3] Gradient descent on surrogate "
          f"({_T_N_RESTARTS} restarts) ...")
    best_norm  = _t_optimise(mlp, X_norm, y_norm, rng)
    gains_pred = _T_BOUNDS[:, 0] + best_norm * (_T_BOUNDS[:, 1] - _T_BOUNDS[:, 0])
    cost_surr  = float(mlp.forward(best_norm[None, :])[0]) * y_std + y_mean
    print(f"  Surrogate minimum: predicted cost = {cost_surr:.4f}")
    for name, val in zip(_T_PARAM_NAMES, gains_pred):
        print(f"    {name:<8} = {val:.6f}")

    # ── Phase 4: verification ─────────────────────────────────────────────────
    print("\n[Phase 4] Verification simulation (clean plant) ...")
    q_id_p, q_iq_p, r_vd_p, r_vq_p, ki_v_p = gains_pred
    met_verify = _t_run_with_gains(q_id_p, q_iq_p, r_vd_p, r_vq_p, ki_v_p)

    if met_verify is None:
        print("  Verification UNSTABLE — returning best observed gains.")
        final_gains = best_obs_gains
        final_met   = best_obs_met
    elif met_verify["cost"] < best_obs_cost:
        final_gains = gains_pred
        final_met   = met_verify
        print(f"  NN gains BETTER: cost {met_verify['cost']:.4f} "
              f"< {best_obs_cost:.4f}")
    else:
        final_gains = best_obs_gains
        final_met   = best_obs_met
        print(f"  Best observed still wins: "
              f"cost {best_obs_cost:.4f} ≤ {met_verify['cost']:.4f}")
        print("  (increase _T_N_EXPLORE for a richer training set)")

    q_id_f, q_iq_f, r_vd_f, r_vq_f, ki_v_f = [float(v) for v in final_gains]

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TUNING COMPLETE")
    print("=" * 70)
    print(f"\n  {'Parameter':<8}  {'Baseline':>10}  {'Tuned':>10}  {'Delta':>8}")
    print(f"  {'-'*44}")
    for name, dflt, tuned in zip(_T_PARAM_NAMES, _T_DEFAULTS, final_gains):
        pct  = (float(tuned) - dflt) / (abs(dflt) + 1e-12) * 100.0
        sign = "UP" if pct > 0.0 else "DN"
        print(f"  {name:<8}  {dflt:>10.4f}  {float(tuned):>10.4f}  "
              f"{sign} {abs(pct):5.1f}%")

    print(f"\n  Best cost      : {final_met['cost']:.4f}")
    print(f"  id RMS (MTPA)  : {final_met['id_rms']:.4f} A")
    print(f"  SS speed error : {final_met['ss_err']:.2f} RPM")
    print(f"  Load drop      : {final_met['load_drop']:.1f} RPM")
    print("=" * 70)

    # ── Phase 5: update active gains + write header ───────────────────────────
    _ACTIVE_GAINS["Q_id"]  = q_id_f
    _ACTIVE_GAINS["Q_iq"]  = q_iq_f
    _ACTIVE_GAINS["R_vd"]  = r_vd_f
    _ACTIVE_GAINS["R_vq"]  = r_vq_f
    _ACTIVE_GAINS["KI_v"]  = ki_v_f

    out_path = _FS_ELEC / "embed_sim_mpc_gains_tuned.h"
    _write_gains_header(
        Q_id      = q_id_f,
        Q_iq      = q_iq_f,
        R_vd      = r_vd_f,
        R_vq      = r_vq_f,
        KI_v      = ki_v_f,
        cost      = final_met["cost"],
        id_rms    = final_met["id_rms"],
        ss_err    = final_met["ss_err"],
        load_drop = final_met["load_drop"],
        n_explore = _T_N_EXPLORE,
        out_path  = out_path,
    )

    return True


# =============================================================================
# Entry point
# =============================================================================

def _ask_user_tune() -> bool:
    """
    Interactively ask whether to run the weight tuner.

    Accepts: y / yes (case-insensitive).
    """
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  WEIGHT TUNER                                               │")
    print("  │  Run the NN surrogate tuner before the main simulation?     │")
    print("  │                                                             │")
    print(f"  │  This will run ~{_T_N_EXPLORE} simulations and may take several minutes.  │")
    print("  │  On completion it writes embed_sim_mpc_gains_tuned.h and   │")
    print("  │  uses the tuned weights for the main simulation.           │")
    print("  └─────────────────────────────────────────────────────────────┘")
    try:
        answer = input("  Run tuner? [y/N] : ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in ("y", "yes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DB42S02 MPC FOC simulation with optional NN weight tuner.")
    parser.add_argument(
        "--tune", action="store_true",
        help="Run the NN surrogate weight tuner before the main simulation "
             "(non-interactive, equivalent to answering 'y' at the prompt).",
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip the tuner prompt and use current weights directly.",
    )
    args = parser.parse_args()

    print("=" * 68)
    print("  DB42S02  —  MPC FOC + SMO  —  20 kHz  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target  : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC}V  dt={DT*1e6:.0f}µs")
    print(f"  Q_omega : {MPC_Q_OMEGA:.1f}  (fixed)")
    print(f"\n  Default weights (hardware commissioning):")
    for k, v in _ACTIVE_GAINS.items():
        print(f"    {k:<8} = {v}")
    print("=" * 68)

    # ── Tuner gate ────────────────────────────────────────────────────────────
    if args.tune:
        do_tune = True
    elif args.no_tune:
        do_tune = False
    else:
        do_tune = _ask_user_tune()

    if do_tune:
        ok = run_tuner()
        if not ok:
            print("\n  Tuner aborted — proceeding with default weights.")
        print(f"\n  Weights active for main simulation:")
        for k, v in _ACTIVE_GAINS.items():
            print(f"    {k:<8} = {v:.6f}")
    else:
        print("\n  Tuner skipped — using default weights.")

    # ── Main simulation ───────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  Running main simulation ...")
    print("=" * 68)

    data = build_and_run()
    print_summary(data)
    plot_results(data, path=str(_HERE / "db42s02_mpc_foc_20k_results.png"))

    print("\n[Done]")
    print("  db42s02_mpc_foc_20k_results.png")
    print("  db42s02_mpc_topology.html")
    print("  embedsim_gen/embedsim_step.c   ← flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")
    if do_tune and (_FS_ELEC / "embed_sim_mpc_gains_tuned.h").exists():
        print("  embed_sim_mpc_gains_tuned.h    ← replace embed_sim_mpc_gains.h")
