# db42s02_closed_loop_smc_foc_20k.py
"""
db42s02_closed_loop_smc_foc_20k.py
===================================
EmbedSim  --  Closed-loop SMC FOC  --  NANOTEC DB42S02  --  AURIX TC3xx test

Architecture (encoder-based):
  theta_e  = p*theta_m              exact from encoder -> Park / InvPark
  omega_m  = delta_theta_m/dt + IIR encoder speed      -> speed SMC
  Speed SMC   -> iq_ref
  Current SMC -> vd, vq -> InvPark -> SVPWM -> ta,tb,tc -> AURIX GTM

Load schedule (simulation only):
  t < 0.5s  : no load
  0.5-1.2s  : 5 mN.m
  1.2-2.0s  : 20 mN.m

CodeGen  ->  embedsim_gen/embedsim_step.c / .h
"""

from __future__ import annotations

import sys
import math
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
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay
from embedsim.code_generator import CodeGenStart, CodeGenEnd

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from smc_controller_block import SMCControllerBlock, _DB42S02
from pmsm_python_plant import PMSM_Python_Plant


# =============================================================================
# Test parameters
# =============================================================================

V_DC       = _DB42S02.SMC_V_DC   # 17.0 V
TARGET_RPM = 2000.0
T_SIM      = 5.0
DT         = 50e-6                # 20 kHz
_RAMP_TIME = 0.5

T_LOAD_T1    = 0.5
T_LOAD_T2    = 1.2
T_LOAD_ZERO  = 0.000
T_LOAD_LIGHT = 0.005
T_LOAD_HEAVY = 0.020

TARGET_RADS_MECH = TARGET_RPM * 2.0 * math.pi / 60.0
_MOTOR_OUT_SIZE  = 8   # [rpm, ia, ib, ic, theta_m, T_em, id, iq]

SMC_KS_W  = 3.095
SMC_ETA_W = 0.001
SMC_PHI_W = 545.0
SMC_KS_I  = 0.0735
SMC_PHI_I = 0.5


# =============================================================================
# Plant block
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
                ta_in = float(v[0])
                tb_in = float(v[1])
                tc_in = float(v[2])
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

    INPUT_NAMES       = ["omega_ref_mech", "theta_m", "ia", "ib", "ic"]
    INPUT_KEEP        = [0, 1, 2, 3, 4]
    C_CODEGEN_EXCLUDE = True

    _RAMP_RATE = TARGET_RADS_MECH / _RAMP_TIME

    def __init__(self, name: str = "ctrl_packer", **kw):
        super().__init__(name, **kw)
        self.output_label           = "[w_ref,th_m,ia,ib,ic]"
        self._omega_ref_filt: float = 0.0

    def reset(self):
        super().reset()
        self._omega_ref_filt = 0.0

    def compute_py(self, t, dt, input_values=None):
        m = (input_values[0].value
             if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value
             if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))

        omega_target = float(r[0]) if len(r) > 0 else 0.0
        max_step = self._RAMP_RATE * dt
        self._omega_ref_filt += max(
            -max_step,
            min(max_step, omega_target - self._omega_ref_filt))

        theta_m = float(m[4]) if len(m) > 4 else 0.0

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
# Build, simulate, generate code
# =============================================================================

def build_and_run() -> dict:

    print("=" * 68)
    print("  NANOTEC DB42S02  --  SMC FOC + SMO  |  AURIX TC3xx")
    print("=" * 68)
    print(f"  Target : {TARGET_RPM:.0f} RPM  |  Vdc={V_DC}V  "
          f"dt={DT*1e6:.0f}us  T_sim={T_SIM}s")
    print(f"  KS_W={SMC_KS_W:.3f} A  ETA_W={SMC_ETA_W:.4f}  "
          f"PHI_W={SMC_PHI_W:.2f} rad/s  "
          f"KS_I={SMC_KS_I:.4f} V  PHI_I={SMC_PHI_I:.3f} A")
    print(f"  SMO   : k={_DB42S02.SMC_SMO_K:.2f} V  "
          f"fc={_DB42S02.SMC_SMO_FC:.0f} Hz")
    print(f"  Load  : 0 -> {T_LOAD_LIGHT*1e3:.0f} mN.m @ {T_LOAD_T1}s"
          f" -> {T_LOAD_HEAVY*1e3:.0f} mN.m @ {T_LOAD_T2}s")
    print("=" * 68)

    cg_start = CodeGenStart("cg_start")

    smc = SMCControllerBlock(
        "smc",
        SMC_V_DC      = V_DC,
        SMC_KS_W      = SMC_KS_W,
        SMC_ETA_W     = SMC_ETA_W,
        SMC_PHI_W     = SMC_PHI_W,
        SMC_KS_I      = SMC_KS_I,
        SMC_PHI_I     = SMC_PHI_I,
        SMC_SMO_K     = _DB42S02.SMC_SMO_K,
        SMC_SMO_FC    = _DB42S02.SMC_SMO_FC,
        dt_s          = DT,
        use_c_backend = True,
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
    motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
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

    # ── Wire labels — signal semantics declared here, baked into the HTML ──────
    # Pattern: ("src_block_name", "dst_block_name") → "signal label string"
    WIRE_LABELS = {
        ("speed_ref",    "ctrl_packer"):  "ω_ref [rad/s]",
        ("motor_delay",  "ctrl_packer"):  "[rpm,ia,ib,ic,θ_m,Tem,id,iq] z⁻¹",
        ("ctrl_packer",  "cg_start"):     "[ω_ref,θ_m,ia,ib,ic]",
        ("cg_start",     "smc"):          "[ω_ref,θ_m,ia,ib,ic]",
        ("smc",          "svpwm_pack"):   "[v_α,v_β]",
        ("svpwm_pack",   "svpwm"):        "[Vref,α,Vdc]",
        ("svpwm",        "cg_end"):       "[ta,tb,tc,sector]",
        ("cg_end",       "motor"):        "[ta,tb,tc,sector]",
        ("cg_end",       "sink_cg"):      "[ta,tb,tc,sector]",
        ("motor",        "motor_delay"):  "[rpm,ia,ib,ic,θ_m,Tem,id,iq]",
        ("motor",        "sink"):         "[rpm,ia,ib,ic,θ_m,Tem,id,iq]",
        ("load_torque",  "motor"):        "T_load [N·m]",
    }

    print("\n[Topology]")
    sim.topo.print_console()
    sim.topo.export_html(str(_HERE / "db42s02_smc_topology.html"),
                         wire_labels=WIRE_LABELS)

    sim.scope.add(smc,        indices=[0, 1],                label="Vab")
    sim.scope.add(svpwm_pack, indices=[0],                   label="Vref")
    sim.scope.add(svpwm,      indices=[0, 1, 2, 3],          label="Duties")
    sim.scope.add(motor,      indices=[0, 1, 2, 3, 5, 6, 7], label="Motor")

    print("\nRunning simulation...")
    sim.run()
    print(f"  Done: {len(sim.scope.t)} steps")

    print("\n[CodeGen] Generating AURIX C code...")
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

    sc = sim.scope
    t  = np.array(sc.t, dtype=np.float32)
    ld = smc.log_data

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
        "id_plant":      _m(5),
        "iq_plant":      _m(6),
    }


# =============================================================================
# Plot
# =============================================================================

def plot_results(d: dict,
                 path: str = "db42s02_smc_foc_20k_results.png") -> None:
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        f"NANOTEC DB42S02 -- SMC FOC + SMO  |  {TARGET_RPM:.0f} RPM  |  20 kHz",
        fontsize=12, fontweight="bold")
    t = d["t"]

    ax = axes[0, 0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.5, label="w_ref")
    ax.plot(t, d["speed_rpm"],     "C0",  lw=1.5, label="w_actual")
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
    ax.plot(t, d["iq_ref"], "k--", lw=1.2, label="iq_ref")
    ax.plot(t, d["iq"],     "C0",  lw=1.0, label="iq_meas")
    ax.plot(t, d["id"],     "C5",  lw=1.0, label="id_meas")
    ax.axhline(0, color="gray", ls="--", lw=0.5)
    ax.axhline( _DB42S02.SMC_I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-_DB42S02.SMC_I_MAX, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_ylabel("Current [A]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("dq currents (MTPA  id_ref=0)")
    ax.set_xlabel("t [s]")

    ax = axes[1, 1]
    ax.plot(t, d["id"], "C5", lw=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("id [A]"); ax.grid(alpha=0.3)
    ax.set_title("id  (should = 0  --  MTPA)"); ax.set_xlabel("t [s]")

    ax = axes[2, 0]
    ax.plot(t, d["v_alpha"], "C0", lw=0.8, label="v_alpha")
    ax.plot(t, d["v_beta"],  "C1", lw=0.8, label="v_beta")
    ax.set_ylabel("Voltage [V]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Stator voltage commands")
    ax.set_xlabel("t [s]")

    ax = axes[2, 1]
    ax.plot(t, d["vref"], "C5", lw=0.8)
    ax.axhline(0.95, color="red", ls="--", lw=0.8, alpha=0.7, label="clip=0.95")
    ax.set_ylabel("Vref [norm]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("SVPWM modulation index")
    ax.set_xlabel("t [s]")

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
               alpha=0.7, label=f"{T_LOAD_LIGHT*1e3:.0f} mN.m")
    ax.axhline(T_LOAD_HEAVY * 1000, color="red",    ls=":", lw=1.0,
               alpha=0.7, label=f"{T_LOAD_HEAVY*1e3:.0f} mN.m")
    ax.set_ylabel("Torque [mN.m]"); ax.legend(fontsize=8)
    ax.grid(alpha=0.3); ax.set_title("Electromagnetic torque")
    ax.set_xlabel("t [s]")

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

    print("\n" + "=" * 60)
    print("  SMC FOC + SMO -- Performance Summary")
    print("=" * 60)
    print(f"  Final speed    : {rpm[-1]:.1f} RPM  (target {TARGET_RPM:.0f})")
    print(f"  SS error       : {ss_err:.2f} RPM  (last 20%)")
    print(f"  Load drop      : {drop:.0f} RPM  at t={T_LOAD_T2}s")
    print(f"  iq_ref SS      : {iq_ss:.3f} A  "
          f"(expected {T_LOAD_HEAVY/_DB42S02.SMC_KT:.2f} A for 20 mN.m)")
    print(f"  Vref max       : {vref_mx:.3f}  (clip 0.95)")
    print("=" * 60)


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    data = build_and_run()
    plot_results(data)
    print_summary(data)
    print("\n[Done]")
    print("  db42s02_smc_foc_20k_results.png")
    print("  db42s02_smc_topology.html")
    print("  embedsim_gen/embedsim_step.c   <- flash to AURIX")
    print("  embedsim_gen/embedsim_step.h")
