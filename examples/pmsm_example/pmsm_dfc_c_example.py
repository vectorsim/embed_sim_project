"""
pmsm_dfc_c_example_fixed.py  -  Fixed closed‑loop simulation with C DFC.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

# Path setup
from _path_utils import get_project_root, get_embedsim_import_path, get_current_parent

_HERE = get_current_parent()
_ROOT = get_project_root()
_PMSM = _ROOT / "pmsm"
_C_SRC = _PMSM / "c_src"

for _p in (get_embedsim_import_path(), str(_PMSM), str(_C_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.plot_helper import create_plotter

# Plant – use the RK4 version (original)
from pmsm_python_plant import PMSM_Python_Plant

# Helpers
from embedsim_connections import CtrlPacker, LoadAdapter, MotorVectorDelay

# C controller
from embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_DFC


# =============================================================================
# DEBUG Motor wrapper – prints internal state and input
# =============================================================================

class DebugMotor(PMSM_Python_Plant):
    """
    Subclass that prints debug info without changing the plant logic.
    """
    def compute(self, t, dt, input_values=None):
        # Call parent (RK4 integration)
        result = super().compute(t, dt, input_values)

        # Print internal state every 0.3 s
        if not hasattr(self, '_last_print_t'):
            self._last_print_t = -1.0
        if t - self._last_print_t >= 0.3:
            print(f"\n[Motor] t={t:.3f}  omega_m={self.omega_m:.3f} rad/s  "
                  f"id={self.id:.4f} A  iq={self.iq:.4f} A  "
                  f"theta_m={self.theta_m:.4f} rad")
            # Also print input duty cycles
            if input_values is not None and len(input_values) > 0:
                u = input_values[0].value
                if len(u) >= 3:
                    print(f"[Motor] input duty: {u[0]:.4f}, {u[1]:.4f}, {u[2]:.4f}, Vdc={u[3]:.2f}")
            self._last_print_t = t

        return result


# =============================================================================
# SpeedPrinter as a SINK (parallel tap, no pass‑through)
# =============================================================================

class SpeedPrinter(VectorBlock):
    """
    Sink that prints motor speed every interval – no output.
    """
    NUM_INPUTS = 1
    OUTPUT_SIZE = 0

    def __init__(self, name="speed_printer", print_interval=0.3):
        super().__init__(name, use_c_backend=False, dtype=DEFAULT_DTYPE)
        self._last_print_t = -1.0
        self._print_interval = float(print_interval)

    def compute(self, t, dt, input_values=None):
        if input_values is not None and len(input_values) > 0:
            speed = float(input_values[0].value[0])
            if t - self._last_print_t >= self._print_interval:
                print(f"[SpeedPrinter] t={t:6.2f}s  Speed={speed:7.1f} RPM")
                self._last_print_t = t
        return None


# =============================================================================
# Simulation Parameters
# =============================================================================

T_SIM = 5.0           # shorter for quick test
DT = 100e-6           # 100 µs – faster simulation (still stable)
V_DC = 12.0
TARGET_RPM = 850.0
STEP_TIME = 0.01

# Motor parameters (same as before)
R_S = 0.19
L_D = 0.125e-3
L_Q = 0.125e-3
LAMBDA_PM = 0.0014
J_ROTOR = 2.4e-6
B_FRIC = 1.0e-6
P_POLES = 4.0

# =============================================================================
# Build blocks
# =============================================================================

# Speed reference
speed_ref = VectorStep("speed_ref", step_time=STEP_TIME,
                        before_value=0.0, after_value=TARGET_RPM, dim=1)

# Motor – using debug wrapper to see what happens inside
motor = DebugMotor(
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
motor_out_size = 8

# C DFC controller
ctrl = EmbedSimControlBlock(
    name="ctrl",
    dt_s=DT,
    ctrl_alg=SIM_CTRL_DFC,
    vdc_nom=V_DC,
    use_c_backend=True,
)

# Packer, adapter, delay
ctrl_packer = CtrlPacker("ctrl_packer", vdc=V_DC, valid_flag=1)
load_adapter = LoadAdapter("load_adapter", vdc=V_DC, tload=0.0)
motor_delay = MotorVectorDelay("motor_delay", vector_size=motor_out_size)

# Sinks
sink = VectorEnd("sink")
speed_printer = SpeedPrinter("speed_printer", print_interval=0.3)

# =============================================================================
# Connections – CLEAN TOPOLOGY
# =============================================================================

# Forward path
speed_ref >> ctrl_packer
motor_delay >> ctrl_packer
ctrl_packer >> ctrl
ctrl >> load_adapter
load_adapter >> motor

# Feedback & outputs – parallel taps
motor >> motor_delay   # feedback to controller
motor >> sink          # terminal sink for simulation
motor >> speed_printer # speed printer as a separate tap (sink)

# =============================================================================
# Simulation
# =============================================================================

sim = EmbedSim(
    sinks=[sink, speed_printer],   # both sinks
    T=T_SIM,
    dt=DT,
    solver=ODESolver.EULER,        # Euler is fine with this DT
)

# Topology
sim.print_topology()

# Scope
sim.scope.add(speed_ref, indices=[0], label="SpeedRef")
sim.scope.add(motor, indices=[0], label="Motor")

print("\n" + "="*60)
print(" FIXED CLOSED‑LOOP SIMULATION")
print("="*60)
print(f" Target: {TARGET_RPM} RPM")
print(f" Time: {T_SIM}s, dt={DT*1e6:.0f}µs")
print("="*60 + "\n")

sim.run(progress_bar=True)

print("\n✓ Simulation complete")

# =============================================================================
# Plot
# =============================================================================

ph = create_plotter(sim)
ph.easyplot(
    ["SpeedRef[0]", "Motor[0]"],
    title="Speed Control – Fixed Topology",
    time_range=(0, T_SIM),
    figsize=(10, 4),
    save_path=None
)

# Summary
sc = sim.scope
speed_data = sc.get_signal("Motor", 0)
if speed_data is not None and len(speed_data) > 0:
    final_speed = speed_data[-1]
    steady_start = int(len(speed_data) * 0.8)
    steady_speed = np.mean(speed_data[steady_start:])
    steady_std = np.std(speed_data[steady_start:])
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(f" Final speed: {final_speed:.1f} RPM")
    print(f" Steady-state: {steady_speed:.1f} ± {steady_std:.1f} RPM")
    print(f" Error: {steady_speed - TARGET_RPM:+.1f} RPM")
    print("="*60)

plt.show()