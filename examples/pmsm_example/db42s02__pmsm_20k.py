"""
db42s02__pmsm_20k.py  -  PMSM Control Simulation - C BACKEND
"""

from __future__ import annotations

import sys
import os
import math
from pathlib import Path

# ================================================================
# Path setup
# ================================================================
_HERE = Path(__file__).resolve().parent
_PMSM = _HERE.parent / "pmsm"
_C_SRC = _PMSM / "c_src"

if str(_C_SRC) not in sys.path:
    sys.path.insert(0, str(_C_SRC))

# ================================================================
# Imports
# ================================================================

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from _path_utils import get_project_root, get_embedsim_import_path

_ROOT = get_project_root()
_PMSM = _ROOT / "pmsm"

for _p in (get_embedsim_import_path(),
           str(_PMSM),
           str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay
from embedsim.plot_helper import create_plotter

from pmsm_python_plant import PMSM_Python_Plant
from embedsim_control_block import EmbedSimControlBlock


# =============================================================================
# Simulation constants
# =============================================================================

P_POLES = 4
R_S = 0.19
L_D = 0.125e-3
L_Q = 0.125e-3
LAMBDA_PM = 0.0014
J_ROTOR = 2.4e-6
B_FRIC = 1.0e-6
I_MAX = 3.57
V_DC = 12.0

TARGET_RPM = 2000.0
T_SIM = 4.0
DT = 50e-6

_MOTOR_OUT_SIZE = 8

VALID_FLAG = 1
INVALID_FLAG = 0
SIM_CTRL_OPEN_LOOP = 0
SIM_CTRL_DFC = 1
HARDWARE_VDC = 12.0


# =============================================================================
# Wire labels
# =============================================================================

_WIRE_LABELS = {
    ("speed_ref", "ctrl_packer"): "rpm_ref [RPM]",
    ("motor_delay", "ctrl_packer"): "[rpm,ia,ib,ic,pos,Tem,id,iq]",
    ("ctrl", "duty_delay"): "[ta,tb,tc]",
    ("duty_delay", "ctrl_packer"): "[dutyU,dutyV,dutyW]",
    ("ctrl_packer", "ctrl"): "ALL vars",
    ("ctrl", "load_adapter"): "[ta,tb,tc]",
    ("load_adapter", "motor"): "[ta,tb,tc,Vdc,Tload]",
    ("motor", "motor_delay"): "[rpm,ia,ib,ic,pos,Tem,id,iq]",
    ("motor", "sink"): "[rpm,ia,ib,ic,pos,Tem,id,iq]",
}


# =============================================================================
# CtrlPacker
# =============================================================================

class CtrlPacker(VectorBlock):
    TOPO_CATEGORY = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label = "[rpm_ref,ia,ib,ic,dutyU,dutyV,dutyW,speed_rpm,pos_rad,vdc,valid]"

    def __init__(self, name="ctrl_packer", monitor=None):
        super().__init__(name)
        self.vector_size = 11
        self._dutyU = 0.5
        self._dutyV = 0.5
        self._dutyW = 0.5
        self._monitor = monitor

    def compute_py(self, t, dt, input_values=None):
        rpm_ref = 0.0
        ia = ib = ic = 0.0
        speed_rpm = 0.0
        position_rad = 0.0
        dutyU = self._dutyU
        dutyV = self._dutyV
        dutyW = self._dutyW

        vdc = V_DC
        valid = VALID_FLAG

        if input_values:
            for sig in input_values:
                if sig is None:
                    continue
                v = np.atleast_1d(sig.value)
                if len(v) >= _MOTOR_OUT_SIZE:
                    speed_rpm = float(v[0])
                    ia = float(v[1])
                    ib = float(v[2])
                    ic = float(v[3])
                    position_rad = float(v[4])
                elif len(v) >= 3:
                    dutyU = float(v[0])
                    dutyV = float(v[1])
                    dutyW = float(v[2])
                    self._dutyU = dutyU
                    self._dutyV = dutyV
                    self._dutyW = dutyW
                elif len(v) >= 1:
                    rpm_ref = float(v[0])

        position_rad = position_rad % (2.0 * math.pi)

        output = VectorSignal(
            np.array([rpm_ref, ia, ib, ic, dutyU, dutyV, dutyW,
                     speed_rpm, position_rad, vdc, valid],
                     dtype=DEFAULT_DTYPE), self.name)

        self.output = output
        return output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Load Adapter
# =============================================================================

class LoadAdapter(VectorBlock):
    TOPO_CATEGORY = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label = "[ta,tb,tc,Vdc,Tload]"

    def __init__(self, name="load_adapter"):
        super().__init__(name)
        self.vector_size = 5

    def compute_py(self, t, dt, input_values=None):
        ta = tb = tc = 0.5
        if input_values and input_values[0] is not None:
            v = np.atleast_1d(input_values[0].value)
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        self.output = VectorSignal(
            np.array([ta, tb, tc, V_DC, 0.0], dtype=DEFAULT_DTYPE), self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Console Monitor
# =============================================================================

class ConsoleMonitor:
    def __init__(self, period_s=0.2):
        self._period = float(period_s)
        self._next_t = 0.0
        self._step = 0

    def tick(self, t, rpm_ref, rpm_plant, ia, ib, ic, id, iq, torque, ta, tb, tc):
        self._step += 1
        if t >= self._next_t:
            self._next_t += self._period
            print(f"\n{'='*80}")
            print(f"  Time: {t:8.3f}s  Step: {self._step:8d}")
            print(f"{'='*80}")
            print(f"  Speed Ref: {rpm_ref:8.1f} RPM")
            print(f"  Speed    : {rpm_plant:8.1f} RPM")
            print(f"  Error    : {rpm_ref - rpm_plant:+8.1f} RPM")
            print(f"\n  ia: {ia:+8.3f}A  ib: {ib:+8.3f}A  ic: {ic:+8.3f}A")
            print(f"  id: {id:+8.3f}A  iq: {iq:+8.3f}A")
            print(f"  Torque: {torque*1e3:+8.2f} mN.m")
            print(f"  ta: {ta:8.3f}  tb: {tb:8.3f}  tc: {tc:8.3f}")
            print(f"{'='*80}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\n[config] target={TARGET_RPM:.0f} RPM  T={T_SIM:.1f}s  dt={DT*1e6:.0f}us")
    print(f"[config] Vdc={V_DC:.1f}V")

    monitor = ConsoleMonitor()

    # ================================================================
    # CONTROLLER: C BACKEND ONLY (WORKING!)
    # ================================================================
    ctrl = EmbedSimControlBlock(
        name="ctrl",
        dt_s=DT,
        ctrl_alg=SIM_CTRL_OPEN_LOOP,  # Open-Loop
        vdc_nom=V_DC,
        use_c_backend=True,            # ← C BACKEND!
    )

    motor = PMSM_Python_Plant(
        name="motor",
        R=R_S,
        L_d=L_D,
        L_q=L_Q,
        lambda_pm=LAMBDA_PM,
        J=J_ROTOR,
        B_fric=B_FRIC,
        p=P_POLES,
        v_dc=V_DC,
    )

    speed_ref = VectorStep("speed_ref", step_time=0.1, before_value=0.0, after_value=TARGET_RPM, dim=1)

    motor_delay = VectorDelay("motor_delay", initial=[0.0] * _MOTOR_OUT_SIZE)
    duty_delay = VectorDelay("duty_delay", initial=[0.5, 0.5, 0.5])

    ctrl_packer = CtrlPacker("ctrl_packer", monitor=monitor)
    load_adapter = LoadAdapter("load_adapter")
    sink = VectorEnd("sink")

    speed_ref >> ctrl_packer
    motor_delay >> ctrl_packer
    ctrl >> duty_delay
    duty_delay >> ctrl_packer
    ctrl_packer >> ctrl
    ctrl >> load_adapter
    load_adapter >> motor
    motor >> motor_delay
    motor >> sink

    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.RK4)

    sim.scope.add(ctrl, indices=[0, 1, 2], label="Duties")
    sim.scope.add(motor, indices=[0, 1, 2, 3, 4, 5, 6, 7], label="Motor")
    sim.scope.add(speed_ref, indices=[0], label="SpeedRef")

    print("\n" + "=" * 60)
    print(" BLOCK DIAGRAM TOPOLOGY")
    print("=" * 60)
    if sim.topo is not None:
        sim.topo.print_console()

    topo_html = _HERE / "pmsm_dfc_topology.html"
    if sim.topo is not None:
        sim.topo.export_html(str(topo_html), wire_labels=_WIRE_LABELS)
        print(f"\n  Topology HTML -> {topo_html}")

    print("\n" + "=" * 60)
    print(" Starting simulation...")
    print("=" * 60)

    sim.run(progress_bar=True)

    ph = create_plotter(sim)

    t = ph.t
    speed_ref_data = ph.sim.scope.get_signal("SpeedRef", 0)
    speed_data = ph.sim.scope.get_signal("Motor", 0)

    fig, ax = plt.subplots(figsize=(12, 6))

    if speed_ref_data is not None:
        ax.plot(t, speed_ref_data, 'r--', linewidth=2, label='Speed Reference (RPM)')

    if speed_data is not None:
        ax.plot(t, speed_data, 'b-', linewidth=2, label='Motor Speed (RPM)')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Speed (RPM)', fontsize=12)
    ax.set_title(f'Motor Speed vs Reference - Target {TARGET_RPM} RPM, Vdc={V_DC:.1f}V', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    ax.axhline(y=TARGET_RPM, color='g', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Target = {TARGET_RPM} RPM')

    if speed_data is not None and len(speed_data) > 0:
        final_speed = speed_data[-1]
        ax.annotate(f'Final Speed: {final_speed:.1f} RPM',
                    xy=(t[-1], final_speed),
                    xytext=(t[-1]*0.7, final_speed*0.8),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='blue'))

    plt.tight_layout()

    save_path = str(_HERE / "pmsm_rpm_plot.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ RPM plot saved to: {save_path}")
    plt.show()

    sc = sim.scope
    speed_data = sc.get_signal("Motor", 0)
    final_speed = speed_data[-1] if speed_data is not None and len(speed_data) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Vdc          : {V_DC:.1f} V")
    print(f"  Target speed : {TARGET_RPM:.0f} RPM")
    print(f"  Final speed  : {final_speed:.1f} RPM")
    print(f"  Error        : {final_speed - TARGET_RPM:+.1f} RPM")

    if abs(final_speed - TARGET_RPM) < 50:
        print("  Status       : SUCCESS")
    elif abs(final_speed) > 100:
        print(f"  Status       : RUNNING at {final_speed:.1f} RPM")
    else:
        print("  Status       : STALLED")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())