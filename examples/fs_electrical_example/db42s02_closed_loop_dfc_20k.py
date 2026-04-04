# db42s02_closed_loop_dfc_20k.py
"""
db42s02_closed_loop_dfc_20k.py
================================
EmbedSim -- Closed-loop Differential Flatness FOC -- NANOTEC DB42S02 -- AURIX TC3xx 20 kHz

Wiring is identical to db42s02_closed_loop_smc_foc_20k.py.
Only SMCControllerBlock is replaced with DFControllerBlock.
CtrlPacker is UNCHANGED — same 5-element bus, same CodeGen boundary.

  cg_start >> dfc >> svpwm_pack >> svpwm >> cg_end

Outputs:
  db42s02_dfc_foc_20k_results.png
  db42s02_dfc_topology.html
  embedsim_gen/embedsim_step.c/.h   (CodeGen)
"""

from __future__ import annotations

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE    = get_current_parent()
_ROOT    = get_project_root()
_FS_ELEC = _ROOT / "fs_electrical_machines"

for _p in (get_embedsim_import_path(), str(_FS_ELEC), str(_FS_ELEC / "c_src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from smc_controller_block import _DB42S02          # motor constants only
from pmsm_python_plant import PMSM_Python_Plant
from ctrl_packer import CtrlPacker                 # UNCHANGED from SMC sim
from machine_feedback import db42s02_feedback_profile
from diff_flatness_controller_block import DFControllerBlock


# =============================================================================
# Simulation constants  (identical to SMC sim)
# =============================================================================

V_DC             = _DB42S02.SMC_V_DC
TARGET_RPM       = 2000.0
T_SIM            = 5.0
DT               = 50e-6
_RAMP_TIME       = 0.5

T_LOAD_T1        = 0.5
T_LOAD_T2        = 1.2
T_LOAD_ZERO      = 0.000
T_LOAD_LIGHT     = 0.005
T_LOAD_HEAVY     = 0.020

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0
_MOTOR_OUT_SIZE  = 8


# =============================================================================
# Plant block  (identical to SMC sim)
# =============================================================================

class DB42S02PlantBlock(PMSM_Python_Plant):
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
        aug = [VectorSignal(np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))]
        return super().compute_py(t, dt, aug)

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Wire labels
# =============================================================================

_WIRE_LABELS = {
    ("speed_ref",   "ctrl_packer"):  "w_ref [rad/s]",
    ("motor_delay", "ctrl_packer"):  "[rpm,ia,ib,ic,th_m,Tem,id,iq] z-1",
    ("ctrl_packer", "cg_start"):     "[w_ref,th_m,ia,ib,ic]",
    ("cg_start",    "dfc"):          "[w_ref,th_m,ia,ib,ic]",
    ("dfc",         "svpwm_pack"):   "[v_alpha,v_beta]",
    ("svpwm_pack",  "svpwm"):        "[Vref,alpha,Vdc]",
    ("svpwm",       "cg_end"):       "[ta,tb,tc,sector]",
    ("cg_end",      "motor"):        "[ta,tb,tc,sector]",
    ("cg_end",      "sink_cg"):      "[ta,tb,tc,sector]",
    ("motor",       "motor_delay"):  "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
    ("motor",       "sink"):         "[rpm,ia,ib,ic,th_m,Tem,id,iq]",
    ("load_torque", "motor"):        "T_load [N.m]",
}


# =============================================================================
# Simulation runner
# =============================================================================

def _run_sim() -> dict | None:
    """
    Identical wiring to SMC sim — only controller block differs.

    cg_start >> dfc >> svpwm_pack >> svpwm >> cg_end
    """
    try:
        cg_start = CodeGenStart("cg_start")

        # ── DFControllerBlock — drop-in for SMCControllerBlock ────────────────
        dfc = DFControllerBlock(
            "dfc",
            P_POLES          = int(_DB42S02.SMC_P_POLES),
            R_S              = _DB42S02.SMC_R_S,
            L_D              = _DB42S02.SMC_L_D,
            L_Q              = _DB42S02.SMC_L_Q,
            LAMBDA_PM        = _DB42S02.SMC_LAMBDA_PM,
            V_DC             = V_DC,
            I_MAX            = _DB42S02.SMC_I_MAX,
            dt_s             = DT,
            Kp_id            = 0.4,
            Kp_iq            = 8.0,
            Kp_speed         = 0.4,
            smo_k            = _DB42S02.SMC_SMO_K,
            smo_tau          = 1.0 / (2.0 * math.pi * _DB42S02.SMC_SMO_FC),
            fusion_omega_lo  = 50.0,
            fusion_omega_hi  = 250.0,
            fusion_gamma     = 2.0,
            fusion_iir_lo    = 0.05,
            fusion_iir_hi    = 0.30,
            use_c_backend    = True,
        )

        svpwm_pack  = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
        svpwm       = SVPWMBlock("svpwm", use_c_backend=False)
        cg_end      = CodeGenEnd("cg_end")

        speed_ref   = VectorStep("speed_ref", step_time=0.0,
                                 before_value=TARGET_RADS_MECH,
                                 after_value=TARGET_RADS_MECH)
        load_torque = VectorConstant("load_torque", value=T_LOAD_ZERO)
        motor       = DB42S02PlantBlock("motor")
        motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)

        # CtrlPacker UNCHANGED — same as SMC sim
        ctrl        = CtrlPacker("ctrl_packer",
                                 target_rads_mech = TARGET_RADS_MECH,
                                 ramp_time        = _RAMP_TIME,
                                 feedback         = db42s02_feedback_profile(
                                     enc_glitch=False,
                                     adc_noise=False,
                                     adc_sat=False,
                                 ))
        sink        = VectorEnd("sink")
        sink_cg     = VectorEnd("sink_cg")

        # ── Wiring — identical to SMC sim ─────────────────────────────────────
        cg_start >> dfc >> svpwm_pack >> svpwm >> cg_end
        motor >> motor_delay >> ctrl
        speed_ref   >> ctrl
        ctrl        >> cg_start
        cg_end      >> motor
        load_torque >> motor
        motor       >> sink
        cg_end      >> sink_cg

        sim = EmbedSim(sinks=[sink, sink_cg], T=T_SIM, dt=DT,
                       solver=ODESolver.EULER)

        sim.scope.add(dfc,        indices=[0, 1],                label="Vab")
        sim.scope.add(svpwm_pack, indices=[0],                   label="Vref")
        sim.scope.add(svpwm,      indices=[0, 1, 2, 3],          label="Duties")
        sim.scope.add(motor,      indices=[0, 1, 2, 3, 5, 6, 7], label="Motor")

        print("  Running DFC FOC simulation ...")
        sim.run()
        print("  Done.")

    except Exception as exc:
        import traceback
        print(f"  [sim error] {exc}")
        traceback.print_exc()
        return None

    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = dfc.log_data
    if len(t) < 100:
        return None

    def _s(label, pos):
        sig = sc.get_signal(label, pos)
        return sig if sig is not None else np.zeros(len(t), np.float32)

    def _i(key):
        if len(ld["t"]) > 1:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), np.float32)

    def _m(pos):
        sig = sc.get_signal("Motor", pos)
        return sig if sig is not None else np.zeros(len(t), np.float32)

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
        "fusion_alpha":  _i("alpha"),
        "_cg_start":     cg_start,
        "_cg_end":       cg_end,
        "_sim":          sim,
    }


# =============================================================================
# CodeGen  (identical call to SMC sim)
# =============================================================================

def _run_codegen(d: dict) -> None:
    cg_start = d.get("_cg_start")
    cg_end   = d.get("_cg_end")
    sim      = d.get("_sim")
    if not all([cg_start, cg_end, sim]):
        print("  [CodeGen] ERROR: missing objects."); return

    print("\n[Topology]")
    sim.topo.print_console()
    sim.topo.export_html(str(_HERE / "db42s02_dfc_topology.html"),
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
# Plot  (same 2×3 layout as SMC sim, extra panel for SpeedFusion α)
# =============================================================================

def plot_results(d: dict,
                 out_path: str = "db42s02_dfc_foc_20k_results.png") -> None:
    t   = d["t"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor="#111111")
    fig.suptitle(
        "DB42S02  —  Differential Flatness FOC  —  20 kHz  AURIX TC3xx",
        color="white", fontsize=13, fontweight="bold")
    axs = axes.flat

    for ax in axs:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888")
        ax.spines[:].set_color("#333")

    def _vlines(ax):
        ax.axvline(T_LOAD_T1, color="orange",  lw=0.8, ls=":", alpha=0.5)
        ax.axvline(T_LOAD_T2, color="#ff6666", lw=0.8, ls=":", alpha=0.5)

    def _leg(ax):
        ax.legend(fontsize=8, facecolor="#222", labelcolor="white", edgecolor="#444")

    def _fmt(ax, ylabel, title):
        ax.set_ylabel(ylabel, color="#888", fontsize=9)
        ax.set_xlabel("t [s]", color="#888", fontsize=8)
        ax.set_title(title,    color="#cccccc", fontsize=9)

    # [0] Speed
    axs[0].plot(t, d["speed_rpm"],     color="#44bbff", lw=1.4, label="actual")
    axs[0].plot(t, d["omega_ref_rpm"], color="white",   lw=1.0, ls="--",
                alpha=0.5, label="ref")
    axs[0].axhline(TARGET_RPM, color="#ff4444", lw=1.0, ls=":", alpha=0.5)
    _vlines(axs[0]); _leg(axs[0])
    _fmt(axs[0], "Speed [RPM]", "Mechanical speed")

    # [1] iq_ref vs iq
    axs[1].plot(t, d["iq_ref"], color="#ff9944", lw=1.2, label="iq_ref")
    axs[1].plot(t, d["iq"],     color="#44ff88", lw=1.2, label="iq_meas")
    axs[1].axhline(0, color="#444", lw=0.7)
    _vlines(axs[1]); _leg(axs[1])
    _fmt(axs[1], "Current [A]", "q-axis current")

    # [2] id  (MTPA: target = 0)
    axs[2].plot(t, d["id"], color="#bb66ff", lw=1.2)
    axs[2].axhline(0, color="#444", lw=0.7, ls="--")
    _vlines(axs[2])
    _fmt(axs[2], "id [A]", "d-axis current  (MTPA target = 0)")

    # [3] Voltages
    axs[3].plot(t, d["v_alpha"], color="#44bbff", lw=0.8, label="vα")
    axs[3].plot(t, d["v_beta"],  color="#ff9944", lw=0.8, label="vβ")
    _vlines(axs[3]); _leg(axs[3])
    _fmt(axs[3], "Voltage [V]", "αβ voltage commands")

    # [4] SVPWM modulation index
    axs[4].plot(t, d["vref"], color="#ffdd44", lw=1.2)
    axs[4].axhline(1.0, color="#ff4444", lw=0.8, ls="--", alpha=0.6)
    _vlines(axs[4])
    _fmt(axs[4], "Vref [0-1]", "SVPWM modulation index")

    # [5] SpeedFusion α  (unique to DFC sim)
    axs[5].plot(t, d["fusion_alpha"], color="#cc88ff", lw=1.8)
    axs[5].fill_between(t, 0, d["fusion_alpha"], color="#cc88ff", alpha=0.15)
    axs[5].axhline(0, color="#444", lw=0.7)
    axs[5].axhline(1, color="#444", lw=0.7)
    axs[5].set_ylim(-0.05, 1.1)
    _fmt(axs[5], "α [0=enc, 1=SMO]", "SpeedFusion weight  (0→encoder  1→SMO)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved {out_path}")


# =============================================================================
# Summary
# =============================================================================

def print_summary(d: dict) -> None:
    t   = d["t"]
    rpm = d["speed_rpm"]
    ss  = t > 0.85 * T_SIM
    if not np.any(ss):
        print("  [summary] insufficient data"); return
    ss_err   = float(np.mean(np.abs(rpm[ss] - TARGET_RPM)))
    id_rms   = float(np.sqrt(np.mean(d["id"][ss] ** 2)))
    iq_chat  = float(np.std(d["iq_ref"][ss]))
    alpha_ss = float(np.mean(d["fusion_alpha"][ss]))
    print(f"\n{'='*55}")
    print("  DFC FOC — Performance Summary")
    print(f"{'='*55}")
    print(f"  SS speed error  : {ss_err:.2f} RPM")
    print(f"  id RMS (MTPA)   : {id_rms:.4f} A    (target 0)")
    print(f"  iq chattering   : {iq_chat:.4f} A   (std in SS)")
    print(f"  SpeedFusion α   : {alpha_ss:.3f}    (1 = full SMO at rated speed)")
    print(f"{'='*55}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  DB42S02 — Differential Flatness FOC — 20 kHz")
    print("=" * 60)

    data = _run_sim()
    if data is None:
        print("  Simulation failed."); import sys; sys.exit(1)

    print_summary(data)
    _run_codegen(data)
    plot_results(data)

    print("\n[Done]")
    print("  db42s02_dfc_foc_20k_results.png")
    print("  db42s02_dfc_topology.html")
    print("  embedsim_gen/embedsim_step.c   <- flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")
