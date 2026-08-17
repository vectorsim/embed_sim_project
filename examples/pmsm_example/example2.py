"""
pmsm_dfc_example.py  -  PMSM Control Example with Mode Switching
                       Now uses the FMU-based motor plant.
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


FMU_PATH = str(_ROOT / "pmsm" / "modelica" / "PMSM_Plant_FMU.fmu")

# ================================================================
# Imports
# ================================================================

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep
from embedsim.plot_helper import create_plotter

# Import the FMU block wrapper
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock

# Existing helpers
from embedsim_generic_control import GenericControlBlock
from embedsim_connections import CtrlPacker, LoadAdapter, MotorVectorDelay

# Python and C controllers
from pmsm_dfc import PythonController
from embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_OPEN_LOOP, SIM_CTRL_DFC


# =============================================================================
# MODE SELECTION - CHANGE THIS TO SWITCH MODES
# =============================================================================

# MODE OPTIONS:
#   "PYTHON_OPEN_LOOP"   - Python open-loop (follows speed_ref)
#   "PYTHON_DFC"         - Python DFC
#   "C_OPEN_LOOP"        - C backend open-loop
#   "C_DFC"              - C backend DFC

MODE = "PYTHON_OPEN_LOOP"  # Change this to switch modes


# =============================================================================
# Main Simulation
# =============================================================================

def main():
    # Simulation parameters
    TARGET_RPM = 850.0
    T_SIM = 1.0
    DT = 50e-6
    V_DC = 12.0
    P_POLES = 4

    # Motor parameters (only used for reference; actual values are inside the FMU)
    R_S = 0.19
    L_D = 0.125e-3
    L_Q = 0.125e-3
    LAMBDA_PM = 0.0014
    J_ROTOR = 2.4e-6
    B_FRIC = 1.0e-6

    _MOTOR_OUT_SIZE = 8  # rpm, ia, ib, ic, theta_m, T_em, id_out, iq_out
    VALID_FLAG = 1

    print(f"\n{'='*60}")
    print(" PMSM CONTROL (FMU Plant)")
    print(f"{'='*60}")
    print(f"  Target: {TARGET_RPM:.0f} RPM")
    print(f"  Time: {T_SIM:.1f}s")
    print(f"  dt: {DT*1e6:.0f}us")
    print(f"  Vdc: {V_DC:.1f}V")
    print(f"  Mode: {MODE}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------
    # 1. Create blocks
    # ------------------------------------------------------------
    speed_ref = VectorStep("speed_ref", step_time=0.1, before_value=0.0, after_value=TARGET_RPM, dim=1)

    # Instantiate the FMU plant
    # Provide the absolute path to the compiled FMU file.
    # For this example, we assume it is located in the same directory as this script.
    # Instantiate the FMU plant
    fmu_path = str(FMU_PATH)  # FMU_PATH is defined at module level
    motor = PMSM_Plant_FMUBlock(
        name="motor",
        fmu_path=fmu_path,
    )
    # Optionally, you can override parameters via the block's `parameters` dict:
    # motor.parameters = {'R': R_S, 'L_d': L_D, ...}  # if needed

    # Select controller based on MODE
    if MODE == "PYTHON_OPEN_LOOP":
        ctrl = PythonController(name="ctrl", dt_s=DT, vdc_nom=V_DC,
                               controller_mode="OPEN_LOOP")
        ctrl_label = "Python_OpenLoop"
    elif MODE == "PYTHON_DFC":
        ctrl = PythonController(name="ctrl", dt_s=DT, vdc_nom=V_DC,
                               controller_mode="DFC")
        ctrl_label = "Python_DFC"
    elif MODE == "C_OPEN_LOOP":
        ctrl = EmbedSimControlBlock(
            name="ctrl",
            dt_s=DT,
            ctrl_alg=SIM_CTRL_OPEN_LOOP,
            vdc_nom=V_DC,
            use_c_backend=True,
        )
        ctrl_label = "C_OpenLoop"
    elif MODE == "C_DFC":
        ctrl = EmbedSimControlBlock(
            name="ctrl",
            dt_s=DT,
            ctrl_alg=SIM_CTRL_DFC,
            vdc_nom=V_DC,
            use_c_backend=True,
        )
        ctrl_label = "C_DFC"
    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    # Helper blocks
    ctrl_packer = CtrlPacker("ctrl_packer", vdc=V_DC, valid_flag=VALID_FLAG)
    load_adapter = LoadAdapter("load_adapter", vdc=V_DC, tload=0.0)
    motor_delay = MotorVectorDelay("motor_delay", vector_size=_MOTOR_OUT_SIZE)
    sink = VectorEnd("sink")

    # ------------------------------------------------------------
    # 2. Connect blocks
    # ------------------------------------------------------------
    speed_ref >> ctrl_packer
    motor_delay >> ctrl_packer
    ctrl_packer >> ctrl
    ctrl >> load_adapter
    load_adapter >> motor   # motor expects inputs: [duty_a, duty_b, duty_c, v_dc, T_load]
    motor >> motor_delay    # motor outputs 8 signals
    motor >> sink

    # ------------------------------------------------------------
    # 3. Create simulation object
    # ------------------------------------------------------------
    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.RK4)

    # ------------------------------------------------------------
    # 4. Add signals to scope for plotting
    # ------------------------------------------------------------
    sim.scope.add(speed_ref, indices=[0], label="SpeedRef")
    # Motor outputs: [rpm, ia, ib, ic, theta_m, T_em, id_out, iq_out]
    sim.scope.add(motor, indices=[0, 1, 2, 3, 4, 5, 6, 7], label="Motor")
    sim.scope.add(ctrl, indices=[0, 1, 2], label="Duties")

    print(" Starting simulation...\n")
    sim.run(progress_bar=True)

    # ------------------------------------------------------------
    # 5. Plot results
    # ------------------------------------------------------------
    ph = create_plotter(sim)

    ph.easyplot(["SpeedRef[0]", "Motor[0]"],
                title=f"Speed Control - {ctrl_label} (FMU Plant)",
                time_range=(0, T_SIM),
                figsize=(10, 4),
                save_path=str(_HERE / f"{ctrl_label.lower()}_fmu_speed.png"))

    # ------------------------------------------------------------
    # 6. Analysis
    # ------------------------------------------------------------
    sc = sim.scope
    speed_data = sc.get_signal("Motor", 0)  # rpm
    final_speed = speed_data[-1] if speed_data is not None and len(speed_data) > 0 else 0.0

    if speed_data is not None and len(speed_data) > 100:
        steady_start = int(len(speed_data) * 0.9)
        steady_speed = np.mean(speed_data[steady_start:])
    else:
        steady_speed = final_speed

    print(f"\n{'='*60}")
    print(f" {ctrl_label} SUMMARY (FMU Plant)")
    print(f"{'='*60}")
    print(f"  Target speed: {TARGET_RPM:.0f} RPM")
    print(f"  Final speed: {final_speed:.1f} RPM")
    print(f"  Steady-state speed: {steady_speed:.1f} RPM")
    print(f"  Steady-state error: {steady_speed - TARGET_RPM:+.1f} RPM")

    if abs(steady_speed - TARGET_RPM) < 50:
        print("  Status: SUCCESS ✅")
    else:
        print(f"  Status: RUNNING at {steady_speed:.1f} RPM")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())