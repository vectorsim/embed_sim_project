"""
pmsm_dfc_c_modelica_fmu_plant_example.py

PMSM DFC (Differential Flatness Control) — C controller + Modelica FMU plant.

To get started:
1. Open the OpenModelica model  pmsm/modelica/PMSM_Motor.mo
2. Compile and export it as FMU  →  PMSM_Plant_FMU.fmu
3. Execute  pmsm/modelica/gen_fmu.py  to generate the EmbedSim wrapper
   class  PMSM_Plant_FMUBlock  used below.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BLOCK DIAGRAM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                            ┌───────────────┐
                            │   speed_ref   │   VectorStep
                            │  0 → 850 rpm  │   (step at t = 0.5 s)
                            └───────┬───────┘
                                    │  ω_ref
                                    ▼
                          ┌───────────────────┐
              ┌──────────►│    ctrl_packer    │  bundles ω_ref + measured
              │           │  (vdc, valid=1)   │  state into one control vector
              │           └─────────┬─────────┘
              │                     │
              │                     ▼
              │           ┌───────────────────┐
              │           │       ctrl        │  C DFC controller
              │           │   (C backend)     │  dq-current + speed loop
              │           └─────────┬─────────┘
              │                     │  duty cycles (da, db, dc)
              │                     ▼
              │           ┌───────────────────┐
              │           │   load_adapter    │  duty → 3-phase voltage
              │           │   (vdc, tload=0)  │
              │           └─────────┬─────────┘
              │                     │  (va, vb, vc)
              │                     ▼
              │           ┌───────────────────┐
              │           │       motor       │  Modelica FMU plant
              │           │  PMSM_Plant_FMU   │  (8 outputs: rpm, ia, ib, ic,
              │           │                   │   theta_m, Tem, id, iq)
              │           └─────────┬─────────┘
              │                     │
              │                     ▼
              │           ┌───────────────────┐
              │           │   motor_delay     │  z⁻¹ loop breaker
              │           │  (vector_size=8)  │  outputs motor(k−1)
              │           └─────────┬─────────┘
              │                     │
              └─────────────────────┤   ← feedback: measured state (k−1)
                                    │
                                    ▼
                          ┌───────────────────┐
                          │   motor_debug     │  SignalPrinter
                          │   (every_n=1e4)   │
                          └─────────┬─────────┘
                                    │
                                    ▼
                          ┌───────────────────┐
                          │       sink        │   VectorEnd (DFS root)
                          └───────────────────┘

  Loop breaker
  ────────────
    motor_delay is a LoopBreaker (z⁻¹).  It breaks the algebraic cycle

        ctrl_packer → ctrl → load_adapter → motor → motor_delay
              ▲                                          │
              └──────────────────────────────────────────┘

    Without motor_delay, EmbedSim's two-pass DFS at __init__ would raise
    ValueError (see  example_algebraic_loop.py  for the full treatment).
    With motor_delay the feedback path carries motor(k−1), so each block
    can be evaluated in a well-defined order.

  Resolved execution order (by DFS from 'sink')
  ─────────────────────────────────────────────
    motor_delay → speed_ref → ctrl_packer → ctrl → load_adapter
                → motor → printer → sink

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS SCRIPT DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • builds the block diagram above (speed step → CtrlPacker → C DFC →
    LoadAdapter → FMU motor → MotorVectorDelay → printer → sink)
  • prints the resolved topology (table + ASCII + interactive HTML)
  • runs the simulation and plots the RPM response
"""

from __future__ import annotations

import os
import sys

# ================================================================
# Path setup
# ================================================================
from _path_utils import (
    get_embedsim_import_path,
    get_pmsm_path,
    get_pmsm_c_src_path,
    get_modelica_path,
    get_current_parent,
)

# Directory of THIS file — used to write the topology HTML export.
_HERE = get_current_parent()

for _p in (get_embedsim_import_path(), str(get_pmsm_path()), str(get_pmsm_c_src_path())):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# FMU path (returned as a string by get_modelica_path)
FMU_PATH = get_modelica_path("PMSM_Plant_FMU.fmu")
FMU_NAME = os.path.basename(FMU_PATH)

# ================================================================
# Imports
# ================================================================
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.source_blocks import VectorStep
from embedsim.plot_helper import create_plotter

# FMU plant
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock

# Helpers
from embedsim_connections import CtrlPacker, LoadAdapter, MotorVectorDelay, SignalPrinter

# C controller
from embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_DFC

# =============================================================================
# Simulation parameters
# =============================================================================
T_SIM = 10.0        # total simulated time          [s]
DT = 50e-6          # integration step              [s]
V_DC = 12.0         # DC-link voltage for the inverter [V]
TARGET_RPM = 850.0  # speed setpoint                [rpm]
STEP_TIME = 0.5     # time at which the step is applied [s]

# =============================================================================
# Build blocks
# =============================================================================

# --- Speed reference: 0 → TARGET_RPM at t = STEP_TIME -----------------------
speed_ref = VectorStep(
    "speed_ref",
    step_time=STEP_TIME,
    before_value=0.0,
    after_value=TARGET_RPM,
    dim=1,
)

# --- FMU plant (Modelica PMSM model exported as FMU) ------------------------
motor = PMSM_Plant_FMUBlock(
    name="motor",
    fmu_path=FMU_PATH,
)
print(f"[Plant] FMU model: {FMU_PATH}")

# --- Debug printer: dumps the motor state every N calls ---------------------
# The FMU emits 8 signals in this order, matching motor_fields below.
motor_fields = ["rpm", "ia", "ib", "ic", "theta_m", "Tem", "id", "iq"]
printer = SignalPrinter(
    name="motor_debug",
    fields=motor_fields,
    print_prefix="Motor State: ",
    every_n=10000,          # print once every 10000 calls (≈ every 0.5 s)
)

# --- C DFC controller (closed loop, C backend) ------------------------------
ctrl = EmbedSimControlBlock(
    name="ctrl",
    dt_s=DT,
    ctrl_alg=SIM_CTRL_DFC,
    vdc_nom=V_DC,
    use_c_backend=True,
)

# --- Utility blocks ---------------------------------------------------------
# CtrlPacker   : bundles the speed reference + measured state into the
#                vector the controller expects (adds Vdc and valid flag).
# LoadAdapter  : converts controller duty cycles into the 3-phase voltage
#                vector that the FMU plant accepts as input.
ctrl_packer = CtrlPacker("ctrl_packer", vdc=V_DC, valid_flag=1)
load_adapter = LoadAdapter("load_adapter", vdc=V_DC, tload=0.0)

# --- Feedback delay (z⁻¹ loop breaker) --------------------------------------
# The FMU plant output feeds back into the controller, which feeds into the
# plant.  Without a delay on the feedback path, this would create an
# algebraic loop (see  example_algebraic_loop.py  for the full explanation).
# MotorVectorDelay holds the previous plant output for one step, breaking
# the cycle.  The FMU exposes 8 output signals.
motor_out_size = 8
motor_delay = MotorVectorDelay("motor_delay", vector_size=motor_out_size)

# Terminal sink (drives DFS traversal for the engine)
sink = VectorEnd("sink")

# =============================================================================
# Connections — the >> operator registers directed edges in the block graph
# =============================================================================
speed_ref   >> ctrl_packer
motor_delay >> ctrl_packer
ctrl_packer >> ctrl >> load_adapter >> motor >> motor_delay >> printer >> sink

# =============================================================================
# Simulation engine
# =============================================================================
# EmbedSim.__init__ runs the two-pass DFS that:
#   1. collects all LoopBreaker blocks (here: motor_delay)
#   2. produces a flat execution order in which each block runs after all
#      of its (non-loop-breaker) dependencies
sim = EmbedSim(
    sinks=[sink],
    T=T_SIM,
    dt=DT,
    solver=ODESolver.EULER,
)

# -----------------------------------------------------------------------------
# Print resolved topology
# -----------------------------------------------------------------------------
# Topology is available immediately after EmbedSim.__init__ — no need to run
# the simulation first.  Three representations are printed:
#   1. print_topology()    — compact execution-order table
#   2. topo.print_console() — full ASCII block diagram
#   3. topo.export_html()   — interactive HTML file written next to this script
print("\n" + "=" * 60)
print(" RESOLVED TOPOLOGY")
print("=" * 60)

sim.print_topology()

if sim.topo is not None:
    print("\nTopology (ASCII):")
    sim.topo.print_console()

    _topo_html = _HERE / "pmsm_dfc_c_fmu_topo.html"
    sim.topo.export_html(str(_topo_html))
    print(f"\nTopology HTML → {_topo_html}")

# -----------------------------------------------------------------------------
# Register signals with the scope for plotting
# -----------------------------------------------------------------------------
sim.scope.add(speed_ref, indices=[0], label=speed_ref.name)   # "speed_ref[0]"
sim.scope.add(motor,     indices=[0], label=motor.name)       # "motor[0]"  (RPM)

# -----------------------------------------------------------------------------
# Banner
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print(" C DFC + FMU PLANT SIMULATION")
print("=" * 60)
print(f" Target:     {TARGET_RPM} RPM")
print(f" Time:       {T_SIM}s, dt={DT*1e6:.0f}µs")
print(f" Controller: C_DFC (use_c_backend=True)")
print(f" Plant:      FMU ({FMU_NAME})")
print("=" * 60 + "\n")

# =============================================================================
# Run
# =============================================================================
sim.run(progress_bar=True)

# =============================================================================
# Plot RPM response
# =============================================================================
ph = create_plotter(sim)
ph.easyplot(
    [f"{speed_ref.name}[0]", f"{motor.name}[0]"],
    title="Speed Control - C DFC (FMU Plant)",
    time_range=(0, T_SIM),
    figsize=(10, 4),
    save_path=None,
)

# =============================================================================
# Quick summary — steady-state metrics over the last 10 % of the run
# =============================================================================
sc = sim.scope
speed_data = sc.get_signal(motor.name, 0)
if speed_data is not None and len(speed_data) > 0:
    final_speed = speed_data[-1]
    steady_start = int(len(speed_data) * 0.9)
    steady_speed = np.mean(speed_data[steady_start:])
    steady_std = np.std(speed_data[steady_start:])
    print("\n" + "=" * 60)
    print(" C DFC SUMMARY (FMU Plant)")
    print("=" * 60)
    print(f" Final speed:  {final_speed:.1f} RPM")
    print(f" Steady-state: {steady_speed:.1f} ± {steady_std:.1f} RPM")
    print(f" Error:        {steady_speed - TARGET_RPM:+.1f} RPM")
    print("=" * 60)

plt.show()