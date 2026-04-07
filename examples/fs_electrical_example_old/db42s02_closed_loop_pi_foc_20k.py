# db42s02_closed_loop_pi_foc_20k.py
"""
NANOTEC DB42S02 — Closed-Loop PI FOC  |  20 kHz
================================================

Wiring is IDENTICAL to the proven db42s02_closed_loop_pi_foc.py reference:
  motor >> motor_delay >> ctrl >> foc >> svpwm_pack >> svpwm >> motor
  speed_ref >> ctrl   (port 1)
  load_torque >> motor (port 1)
  motor >> sink

The only differences from the reference file:
  DT   = 50e-6   (20 kHz, was 1e-4 / 10 kHz)
  T_SIM = 2.0 s
  Load schedule in DB42S02PlantBlock (0 → 5 mN·m → 20 mN·m)
  foc = PIFOCBlock  (carries C metadata; same algorithm as ClosedLoopPIFOC)
  svpwm_pack.NUM_INPUTS = 1  (routing fix)
"""

from __future__ import annotations

import sys, os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
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

from motor_utility_blocks import SVPWMPackBlock
from svpwm_block          import SVPWMBlock
from PMSM_MotorBlock      import PMSM_MotorBlock
from pi_foc_block         import PIFOCBlock, _DB42S02

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Motor.fmu")

# =============================================================================
# Constants
# =============================================================================
V_DC        = _DB42S02.V_dc      # 17.0 V
TARGET_RPM  = 2000.0
TARGET_RADS = TARGET_RPM * 2.0 * math.pi / 60.0

T_SIM       = 2.0
DT          = 50e-6              # 20 kHz

T_LOAD_T1   = 0.5                # s — light load
T_LOAD_T2   = 1.2                # s — heavy load
T_LOAD_ZERO  = 0.000
T_LOAD_LIGHT = 0.005
T_LOAD_HEAVY = 0.020

_MOTOR_OUT_SIZE = 7


# =============================================================================
# DB42S02PlantBlock  (identical to reference, load schedule added)
# =============================================================================
class DB42S02PlantBlock(PMSM_MotorBlock):
    TOPO_CATEGORY     = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label      = "[rpm,ia,ib,ic,θe,ωm,Tem]"

    def __init__(self, name, fmu_path):
        super().__init__(name=name, fmu_path=fmu_path,
                         R=_DB42S02.R_s, L_d=_DB42S02.L_d, L_q=_DB42S02.L_q,
                         lambda_pm=_DB42S02.lam_pm,
                         J=_DB42S02.J, B=_DB42S02.B, p=float(_DB42S02.p))
        self._t_last = -1.0
        self._np = 0
        print(f"[FMU] {fmu_path}")

    def compute_py(self, t, dt, input_values=None):
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        if   t < T_LOAD_T1: t_load = T_LOAD_ZERO
        elif t < T_LOAD_T2: t_load = T_LOAD_LIGHT
        else:                t_load = T_LOAD_HEAVY

        ta = max(0.05, min(0.95, ta))
        tb = max(0.05, min(0.95, tb))
        tc = max(0.05, min(0.95, tc))
        if abs(ta-tb) < 1e-6 and abs(tb-tc) < 1e-6:
            ta += 1e-4

        super().compute_py(t, dt, [VectorSignal(
            np.array([ta, tb, tc, V_DC, t_load], dtype=DEFAULT_DTYPE))])

        self.output = VectorSignal(np.array([
            self.read_speed_rpm(),
            self.read_i_a(), self.read_i_b(), self.read_i_c(),
            self.read_theta_e(),
            self.get_output_by_name("omega_m"),
            self.read_T_em(),
        ], dtype=DEFAULT_DTYPE), self.name)

        if t - self._t_last >= 0.2 and self._np < 15:
            print(f"[PLANT t={t:.2f}s]  rpm={self.read_speed_rpm():.0f}"
                  f"  T_load={t_load*1e3:.0f}mN·m"
                  f"  T_em={self.read_T_em()*1e3:.2f}mN·m")
            self._t_last = t; self._np += 1

        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# CtrlPacker  (identical to reference)
# =============================================================================
class CtrlPacker(VectorBlock):
    """
    port 0: motor_delay  [rpm, ia, ib, ic, theta_e, omega_m, Tem]
    port 1: speed_ref    [omega_ref]
    output: [omega_ref, omega_m, theta_e, ia, ib, ic]
    """
    def __init__(self, name="ctrl_packer", **kw):
        super().__init__(name, **kw)
        self.output_label = "[ω_ref,ω_m,θ_e,ia,ib,ic]"

    def compute_py(self, t, dt, input_values=None):
        m = (input_values[0].value if input_values and len(input_values) > 0
             else np.zeros(_MOTOR_OUT_SIZE, dtype=DEFAULT_DTYPE))
        r = (input_values[1].value if input_values and len(input_values) > 1
             else np.zeros(1, dtype=DEFAULT_DTYPE))
        self.output = VectorSignal(np.array([
            float(r[0])  if len(r) > 0 else 0.0,
            float(m[5])  if len(m) > 5 else 0.0,   # omega_m
            float(m[4])  if len(m) > 4 else 0.0,   # theta_e
            float(m[1])  if len(m) > 1 else 0.0,   # ia
            float(m[2])  if len(m) > 2 else 0.0,   # ib
            float(m[3])  if len(m) > 3 else 0.0,   # ic
        ], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Build & run
# =============================================================================
def build_and_run():
    print("=" * 64)
    print(f"  RUNNING FROM: {os.path.abspath(__file__)}")
    print("=" * 64)
    print("  NANOTEC DB42S02 — Closed-Loop PI FOC  |  20 kHz")
    print(f"  Target: {TARGET_RPM:.0f} RPM  Vdc={V_DC}V  dt={DT*1e6:.0f}µs  T={T_SIM}s")
    print(f"  Kp_i={_DB42S02.KP_I:.4f} V/A   Ki_i={_DB42S02.KI_I:.1f} V/(A·s)  ωc_i=2π×500Hz")
    print(f"  Kp_ω={_DB42S02.KP_SPD:.5f} A·s/rad  Ki_ω={_DB42S02.KI_SPD:.5f} A/rad  Ti=0.1s")
    print(f"  B={_DB42S02.B:.0e} N·m·s/rad (from PMSM_Motor.mo)")
    print(f"  iq_ref at full step ({TARGET_RPM:.0f}RPM) = {_DB42S02.KP_SPD*TARGET_RADS:.2f} A → clamped to {_DB42S02.I_max:.2f} A")
    print(f"  Load: 0→{T_LOAD_LIGHT*1e3:.0f}mN·m@{T_LOAD_T1}s →{T_LOAD_HEAVY*1e3:.0f}mN·m@{T_LOAD_T2}s")
    print("=" * 64)

    speed_ref   = VectorStep("speed_ref", step_time=0.0,
                             before_value=TARGET_RADS, after_value=TARGET_RADS)
    load_torque = VectorConstant("load_torque", value=T_LOAD_LIGHT)

    motor       = DB42S02PlantBlock("motor", fmu_path=_FMU_PATH)
    motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
    ctrl        = CtrlPacker("ctrl_packer")
    foc         = PIFOCBlock("foc", v_dc=V_DC, dt_s=DT)
    svpwm_pack  = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
    svpwm_pack.NUM_INPUTS = 1   # routing fix — prevents engine bypassing svpwm
    svpwm       = SVPWMBlock("svpwm")
    sink        = VectorEnd("sink")

    # Wiring — identical to reference file
    motor       >> motor_delay
    motor_delay >> ctrl         # port 0
    speed_ref   >> ctrl         # port 1

    ctrl        >> foc
    foc         >> svpwm_pack
    svpwm_pack  >> svpwm
    svpwm       >> motor        # port 0: duties
    load_torque >> motor        # port 1: T_load
    motor       >> sink

    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.EULER)

    print("\n[Topology]")
    sim.topo.print_console()

    sim.scope.add(speed_ref,  indices=[0],           label="SpeedRef")
    sim.scope.add(foc,        indices=[0, 1],        label="Vab")
    sim.scope.add(svpwm_pack, indices=[0],           label="Vref")
    sim.scope.add(svpwm,      indices=[0, 1, 2, 3],  label="Duties")

    print("\nRunning simulation …")
    sim.run()
    print(f"  Completed: {len(sim.scope.t)} steps")

    sc  = sim.scope
    t   = np.array(sc.t, dtype=np.float32)
    ld  = foc.log_data

    def gs(label, idx=0):
        v = sc.get_signal(label, idx)
        return v if v is not None else np.zeros(len(t))

    def interp(key):
        if len(ld["t"]) > 1:
            return np.interp(t, ld["t"], ld[key]).astype(np.float32)
        return np.zeros(len(t), dtype=np.float32)

    hist = {
        "t":             t,
        "speed_rpm":     interp("speed"),
        "omega_ref_rpm": interp("speed_ref"),
        "iq_ref":        interp("iq_ref"),
        "iq":            interp("iq"),
        "id":            interp("id"),
        "v_alpha":       gs("Vab", 0),
        "v_beta":        gs("Vab", 1),
        "vref":          gs("Vref"),
        "ta":            gs("Duties", 0),
        "tb":            gs("Duties", 1),
        "tc":            gs("Duties", 2),
        "sector":        gs("Duties", 3).astype(int),
    }
    return hist


# =============================================================================
# Summary + plot
# =============================================================================
def print_summary(d):
    n  = len(d["t"])
    ss = int(0.8 * n)
    err  = float(np.mean(np.abs(d["speed_rpm"][ss:] - d["omega_ref_rpm"][ss:])))
    vrfx = float(np.max(d["vref"]))
    after     = d["t"] > T_LOAD_T2
    avg_after = float(np.mean(d["speed_rpm"][after][-100:])) if np.any(after) else float(d["speed_rpm"][-1])
    drop      = float(np.max(d["speed_rpm"][:int(T_LOAD_T2/DT*1.1)])) - avg_after if np.any(after) else 0.0

    # iq_ref from log — this IS reliable (it's the speed PI output, not sampled dq)
    iqr_ss  = float(np.mean(np.abs(d["iq_ref"][ss:])))

    # Note: id/iq in the log are sampled at 1kHz against 133Hz electrical →
    # apparent RMS is aliased and misleading. Speed tracking is the true metric.
    print("\n" + "=" * 55)
    print("  PI FOC 20 kHz — Performance Summary")
    print("=" * 55)
    print(f"  Final speed     : {d['speed_rpm'][-1]:.1f} RPM  (target {TARGET_RPM:.0f})")
    print(f"  SS speed error  : {err:.2f} RPM  (last 20% of run)")
    print(f"  Speed drop      : {drop:.0f} RPM at {T_LOAD_T2}s load step")
    print(f"  Recovery speed  : {avg_after:.0f} RPM  (avg last 100 pts)")
    print(f"  iq_ref SS mean  : {iqr_ss:.3f} A  (≈ T_load/KT = {T_LOAD_HEAVY/_DB42S02.KT*1000:.0f} mA × KT)")
    print(f"  Vref max        : {vrfx:.3f}  (clip 0.95)")
    print(f"  Note: id/iq log sampled at 1kHz vs 133Hz electrical → apparent")
    print(f"        id RMS includes alias. Speed tracking is primary metric.")
    print("=" * 55)


def plot_results(d, path="db42s02_pi_foc_20k_results.png"):
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle(
        f"NANOTEC DB42S02 — PI FOC  |  {TARGET_RPM:.0f} RPM  |  20 kHz",
        fontsize=12, fontweight="bold")
    t = d["t"]

    ax = axes[0, 0]
    ax.plot(t, d["omega_ref_rpm"], "k--", lw=1.5, label="ω_ref")
    ax.plot(t, d["speed_rpm"],     "C0",  lw=1.5, label="ω_actual")
    ax.axvline(T_LOAD_T1, color="orange", ls=":", lw=1.0)
    ax.axvline(T_LOAD_T2, color="red",    ls=":", lw=1.0)
    ax.set_ylabel("Speed [RPM]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Speed tracking"); ax.set_xlabel("t [s]")

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
    ax.axhline( _DB42S02.I_max, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-_DB42S02.I_max, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.set_ylabel("Current [A]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("dq currents — MTPA (id_ref=0)"); ax.set_xlabel("t [s]")

    ax = axes[1, 1]
    ax.plot(t, d["id"], "C5", lw=0.8)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("id [A]"); ax.grid(alpha=0.3)
    ax.set_title("id (should ≈ 0 — MTPA)"); ax.set_xlabel("t [s]")

    ax = axes[2, 0]
    ax.plot(t, d["v_alpha"], "C0", lw=0.8, label="v_α")
    ax.plot(t, d["v_beta"],  "C1", lw=0.8, label="v_β")
    ax.set_ylabel("Voltage [V]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Stator voltage commands"); ax.set_xlabel("t [s]")

    ax = axes[2, 1]
    ax.plot(t, d["vref"], "C5", lw=0.8)
    ax.axhline(0.95, color="red", ls="--", lw=0.8, alpha=0.7, label="clip=0.95")
    ax.set_ylabel("Vref [norm]"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("SVPWM modulation index"); ax.set_xlabel("t [s]")

    ax = axes[3, 0]
    ax.plot(t, d["ta"], "C3", lw=0.7, label="ta → ATOM CH0")
    ax.plot(t, d["tb"], "C2", lw=0.7, label="tb → ATOM CH2")
    ax.plot(t, d["tc"], "C1", lw=0.7, label="tc → ATOM CH4")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Duty"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("SVPWM duties — GTM ATOM0"); ax.set_xlabel("t [s]")

    ax = axes[3, 1]
    ax.plot(d["speed_rpm"], d["iq"], "b.", markersize=0.5, alpha=0.3)
    ax.axvline(TARGET_RPM, color="k", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Speed [RPM]"); ax.set_ylabel("iq [A]")
    ax.grid(alpha=0.3); ax.set_title("Phase portrait: iq vs speed")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    data = build_and_run()
    plot_results(data)
    print_summary(data)
    print("\n[Done]  db42s02_pi_foc_20k_results.png")
