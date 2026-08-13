"""
db42s02__pmsm_20k.py  -  PMSM Control Simulation - C BACKEND
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

# ================================================================
# Path setup
# ================================================================
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_PMSM = _ROOT / "pmsm"
_C_SRC = _PMSM / "c_src"

for _p in (get_embedsim_import_path(), str(_PMSM), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ================================================================
# Imports
# ================================================================

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.simulation_engine import VectorDelay          # <--- built-in loop breaker
from embedsim.plot_helper import create_plotter

from pmsm_python_plant import PMSM_Python_Plant
from embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_OPEN_LOOP, SIM_CTRL_DFC

# =============================================================================
# Custom vector delay that inherits from built-in VectorDelay
# =============================================================================

class VectorDelayVector(VectorDelay):
    """
    A VectorDelay that stores and outputs a full vector.
    Inherits from the built-in VectorDelay, so it is automatically
    a loop breaker (is_loop_breaker = True).
    """
    def __init__(self, name="delay", initial=None):
        if initial is None:
            initial = [0.0]
        # Call parent __init__ – it sets is_loop_breaker = True
        # Pass only the first element as the scalar initial value
        super().__init__(name, initial=initial[0])
        # Override the state to store the whole vector
        self._state = np.array(initial, dtype=float)
        self.vector_size = len(self._state)
        self.output = VectorSignal(self._state.copy(), self.name)

    def compute(self, t, dt, input_values=None):
        if input_values and len(input_values) > 0:
            new_val = input_values[0].value
            if len(new_val) == self.vector_size:
                self._state = new_val.copy()
            # else keep old state (should not happen)
        self.output = VectorSignal(self._state.copy(), self.name)
        return self.output

    def reset(self):
        self._state = np.zeros(self.vector_size, dtype=float)
        self.output = VectorSignal(self._state.copy(), self.name)


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
V_DC = 12.0

TARGET_RPM = 1000.0
T_SIM = 4.0
DT = 50e-6

_MOTOR_OUT_SIZE = 8
VALID_FLAG = 1

# =============================================================================
# Debug flag – set to True to see controller inputs and duties
# =============================================================================
DEBUG_CTRL = True


# =============================================================================
# Wire labels
# =============================================================================

_WIRE_LABELS = {
    ("speed_ref", "ctrl_packer"): "rpm_ref [RPM]",
    ("motor_delay", "ctrl_packer"): "[rpm,ia,ib,ic,theta_m,Tem,id,iq]",
    ("ctrl_packer", "ctrl"): "ctrl inputs [10]",
    ("ctrl", "load_adapter"): "[duty_u,duty_v,duty_w,valid]",
    ("load_adapter", "motor"): "[ta,tb,tc,Vdc,Tload]",
    ("motor", "motor_delay"): "[rpm,ia,ib,ic,theta_m,Tem,id,iq]",
    ("motor", "sink"): "[rpm,ia,ib,ic,theta_m,Tem,id,iq]",
}


# =============================================================================
# CtrlPacker - Two explicit ports (no concatenation issues)
# =============================================================================

class CtrlPacker(VectorBlock):
    """Pack speed reference and motor feedback into the 10‑element vector
    expected by the C control block.
    Uses two explicit input ports:
        port 0: speed reference (1 element)
        port 1: motor feedback (8 elements)
    """
    TOPO_CATEGORY = "utility"
    C_CODEGEN_EXCLUDE = True
    NUM_INPUTS = 2                     # two explicit ports
    output_label = "ctrl_inputs[10]"

    def __init__(self, name="ctrl_packer", dt=DT, monitor=None):
        super().__init__(name)
        self.vector_size = 10
        self._dt = dt
        self._monitor = monitor
        self._last_debug_t = -1.0

    def compute_py(self, t, dt, input_values=None):
        # port 0: speed reference
        speed_ref_sig = input_values[0]
        speed_ref_rpm = float(speed_ref_sig.value[0])

        # port 1: motor feedback (8 elements)
        motor_vals = input_values[1].value
        speed_sensor_rpm = float(motor_vals[0])
        ia = float(motor_vals[1])
        ib = float(motor_vals[2])
        ic = float(motor_vals[3])
        position_sensor_rad = float(motor_vals[4]) % (2.0 * math.pi)

        # Debug print every 0.2 s
        if DEBUG_CTRL and (t - self._last_debug_t >= 0.2):
            self._last_debug_t = t
            print(f"\n[CtrlPacker t={t:.2f}s]")
            print(f"  speed_ref       = {speed_ref_rpm:.1f} RPM")
            print(f"  speed_sensor    = {speed_sensor_rpm:.1f} RPM")
            print(f"  theta_m (mech)  = {position_sensor_rad:.4f} rad")
            print(f"  ia={ia:.3f}  ib={ib:.3f}  ic={ic:.3f}")

        # Build the 10‑element output for the control block
        output_array = np.array([
            speed_ref_rpm,
            ia,
            ib,
            ic,
            speed_sensor_rpm,
            dt,                     # sample time
            position_sensor_rad,    # mechanical angle
            VALID_FLAG,
            0.0,                    # unused placeholder
            V_DC,
        ], dtype=DEFAULT_DTYPE)

        self.output = VectorSignal(output_array, self.name)

        # Optional monitor
        if self._monitor:
            self._monitor.tick(t, speed_ref_rpm, speed_sensor_rpm,
                               ia, ib, ic, motor_vals[6], motor_vals[7],
                               motor_vals[5], 0.0, 0.0, 0.0)

        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Load Adapter - Converts duties to motor inputs
# =============================================================================

class LoadAdapter(VectorBlock):
    TOPO_CATEGORY = "utility"
    C_CODEGEN_EXCLUDE = True
    output_label = "[ta,tb,tc,Vdc,Tload]"

    def __init__(self, name="load_adapter", tload=0.0):
        super().__init__(name)
        self.vector_size = 5
        self._tload = float(tload)

    def compute_py(self, t, dt, input_values=None):
        v = input_values[0].value
        ta = float(v[0])
        tb = float(v[1])
        tc = float(v[2])

        self.output = VectorSignal(
            np.array([ta, tb, tc, V_DC, self._tload], dtype=DEFAULT_DTYPE),
            self.name
        )
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
    # Select control mode here
    CONTROL_MODE = SIM_CTRL_OPEN_LOOP  # Change to SIM_CTRL_DFC for closed loop

    mode_name = "DFC" if CONTROL_MODE == SIM_CTRL_DFC else "OPEN_LOOP"

    print(f"\n[config] target={TARGET_RPM:.0f} RPM  T={T_SIM:.1f}s  dt={DT*1e6:.0f}us")
    print(f"[config] Vdc={V_DC:.1f}V")
    print(f"[config] Mode: {mode_name} (C Backend)")
    print(f"[config] Controller expects MECHANICAL angle (from sensor)")

    monitor = ConsoleMonitor(period_s=0.2)

    ctrl = EmbedSimControlBlock(
        name="ctrl",
        dt_s=DT,
        ctrl_alg=CONTROL_MODE,
        vdc_nom=V_DC,
        use_c_backend=True,
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

    speed_ref = VectorStep(
        "speed_ref",
        step_time=0.1,
        before_value=0.0,
        after_value=TARGET_RPM,
        dim=1
    )

    # Use the custom vector delay (inherits loop-breaker from VectorDelay)
    motor_delay = VectorDelayVector(
        name="motor_delay",
        initial=[0.0] * _MOTOR_OUT_SIZE
    )

    sink = VectorEnd("sink")

    ctrl_packer = CtrlPacker("ctrl_packer", dt=DT, monitor=monitor)
    load_adapter = LoadAdapter("load_adapter", tload=0.0)

    # Connections – port 0: speed_ref, port 1: motor_delay
    speed_ref >> ctrl_packer
    motor_delay >> ctrl_packer
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

    topo_html = _HERE / f"pmsm_{mode_name.lower()}_topology.html"
    if sim.topo is not None:
        sim.topo.export_html(str(topo_html), wire_labels=_WIRE_LABELS)
        print(f"\n  Topology HTML -> {topo_html}")

    print("\n" + "=" * 60)
    print(" Starting simulation...")
    print("=" * 60)

    sim.run(progress_bar=True)

    # ------------------------------------------------------------------------
    # Use plot_helper for clean, organised plotting
    # ------------------------------------------------------------------------
    ph = create_plotter(sim)

    # Plot 1: Speed reference vs motor speed
    ph.easyplot(["SpeedRef[0]", "Motor[0]"],
                title="Speed Reference & Motor Speed",
                time_range=(0, T_SIM),
                figsize=(10, 4),
                save_path=str(_HERE / f"speed_{mode_name.lower()}.png"))

    # Plot 2: Phase currents
    ph.easyplot(["Motor[1]", "Motor[2]", "Motor[3]"],
                title="Phase Currents (ia, ib, ic)",
                time_range=(0, T_SIM),
                figsize=(10, 4),
                save_path=str(_HERE / f"currents_{mode_name.lower()}.png"))

    # Plot 3: dq currents
    ph.easyplot(["Motor[6]", "Motor[7]"],
                title="DQ Currents (id, iq)",
                time_range=(0, T_SIM),
                figsize=(10, 4),
                save_path=str(_HERE / f"dq_{mode_name.lower()}.png"))

    # Summary
    sc = sim.scope
    speed_data = sc.get_signal("Motor", 0)
    final_speed = speed_data[-1] if speed_data is not None and len(speed_data) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f" SUMMARY")
    print(f"{'='*60}")
    print(f"  Control Mode : {mode_name}")
    print(f"  Vdc          : {V_DC:.1f} V")
    print(f"  Target speed : {TARGET_RPM:.0f} RPM")
    print(f"  Final speed  : {final_speed:.1f} RPM")
    print(f"  Error        : {final_speed - TARGET_RPM:+.1f} RPM")

    if abs(final_speed - TARGET_RPM) < 50:
        print("  Status       : SUCCESS ✅")
    elif abs(final_speed) > 100:
        print(f"  Status       : RUNNING at {final_speed:.1f} RPM")
    else:
        print("  Status       : STALLED ❌")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())