"""
pmsm_dfc_example.py  -  PMSM Control Example with Mode Switching
                       Supports both Python and FMU plant models
                       WITH ENHANCED MOTOR STATE DISPLAY & DIAGNOSTICS
                       CLEANED VERSION - Console + RPM Plot Only
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import Optional, Dict, Any

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

# FMU path
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

# Plant models
from pmsm_python_plant import PMSM_Python_Plant
from PMSM_Plant_FMUBlock import PMSM_Plant_FMUBlock

# Existing helpers
from embedsim_generic_control import GenericControlBlock
from embedsim_connections import CtrlPacker, LoadAdapter, MotorVectorDelay

# Python and C controllers
from pmsm_dfc1 import PythonController
from embedsim_control_block import EmbedSimControlBlock, SIM_CTRL_OPEN_LOOP, SIM_CTRL_DFC

# Try to import motor state reporting
try:
    from embedsim_control_wrapper import get_motor_state, control_init
    HAS_C_WRAPPER = True
except ImportError:
    HAS_C_WRAPPER = False
    print("⚠️ Warning: embedsim_control_wrapper not found")
    print("   C motor state reporting will be unavailable")


# =============================================================================
# CONFIGURATION - CHANGE THESE TO SWITCH MODES
# =============================================================================

# Controller options:
#   "PYTHON_OPEN_LOOP"   - Python open-loop (follows speed_ref)
#   "PYTHON_DFC"         - Python DFC (ALIGNED WITH C)
#   "C_OPEN_LOOP"        - C backend open-loop
#   "C_DFC"              - C backend DFC

CONTROLLER_MODE = "C_DFC"  # Change this to switch controllers

# Plant options:
#   "PYTHON"  - Python PMSM plant model
#   "FMU"     - FMU-based plant model

PLANT_MODE = "FMU"  # Change this to switch plants

# =============================================================================
# Simulation Parameters
# =============================================================================

class SimulationConfig:
    """Configuration container for simulation parameters."""

    def __init__(self):
        # Simulation timing
        self.T_SIM = 10
        self.DT = 50e-6

        # Motor parameters (for Python plant)
        self.R_S = 0.19
        self.L_D = 0.125e-3
        self.L_Q = 0.125e-3
        self.LAMBDA_PM = 0.0014
        self.J_ROTOR = 2.4e-6
        self.B_FRIC = 1.0e-6
        self.P_POLES = 4.0

        # Electrical parameters
        self.V_DC = 12.0

        # Control parameters
        self.TARGET_RPM = 850.0
        self.STEP_TIME = 0.01

        # FMU path
        self.fmu_path = FMU_PATH

        # Output settings
        self.SAVE_PLOT = True
        self.PLOT_DIR = _HERE

        # FMU-specific settings
        self.fmu_settling_time = 2.0
        self.fmu_spinning_past_index = 500
        self.fmu_stopped_past_index = 100

        # State logging
        self.LOG_INTERVAL = 0.1  # Log state every 0.1 seconds


# =============================================================================
# Plant Factory
# =============================================================================

def create_plant(plant_mode: str, config: SimulationConfig):
    """Factory function to create the appropriate plant model."""
    if plant_mode == "PYTHON":
        motor = PMSM_Python_Plant(
            name="motor",
            R=config.R_S,
            L_d=config.L_D,
            L_q=config.L_Q,
            lambda_pm=config.LAMBDA_PM,
            J=config.J_ROTOR,
            B_fric=config.B_FRIC,
            p=config.P_POLES,
            v_dc=config.V_DC,
        )
        motor_out_size = 8
        print(f"[Plant] Python PMSM model")
        print(f"  R={config.R_S:.3f}Ω, Ld={config.L_D*1e3:.3f}mH, Lq={config.L_Q*1e3:.3f}mH")
        print(f"  λpm={config.LAMBDA_PM*1e3:.2f}mWb, J={config.J_ROTOR*1e6:.2f}g·m²")
        print(f"  p={config.P_POLES:.0f}, Vdc={config.V_DC:.1f}V")
        plant_type = "PYTHON"

    elif plant_mode == "FMU":
        motor = PMSM_Plant_FMUBlock(
            name="motor",
            fmu_path=config.fmu_path,
        )
        motor_out_size = 8
        print(f"[Plant] FMU model: {config.fmu_path}")
        print(f"  Note: FMU may have different dynamics than Python plant")
        print(f"  Using FMU-specific settings:")
        print(f"    - Spinning PastIndex: {config.fmu_spinning_past_index}")
        print(f"    - Settling time: {config.fmu_settling_time}s")
        plant_type = "FMU"

    else:
        raise ValueError(f"Unknown PLANT_MODE: {plant_mode}")

    return motor, motor_out_size, plant_type


# =============================================================================
# Controller Factory
# =============================================================================

def create_controller(controller_mode: str, config: SimulationConfig, plant_type: str = "PYTHON"):
    """Factory function to create the appropriate controller."""

    if plant_type == "FMU":
        spinning_past_index = config.fmu_spinning_past_index
        stopped_past_index = config.fmu_stopped_past_index
        print(f"[Controller] Using FMU-specific settings:")
        print(f"  Spinning PastIndex: {spinning_past_index}")
        print(f"  Stopped PastIndex: {stopped_past_index}")
    else:
        spinning_past_index = 89500
        stopped_past_index = 2000

    if controller_mode == "PYTHON_OPEN_LOOP":
        ctrl = PythonController(
            name="ctrl",
            dt_s=config.DT,
            vdc_nom=config.V_DC,
            controller_mode="OPEN_LOOP",
            pole_pairs=config.P_POLES,
            rs=config.R_S,
            ld=config.L_D,
            lq=config.L_Q,
            lambda_pm=config.LAMBDA_PM,
        )
        ctrl_label = "Python_OpenLoop"

    elif controller_mode == "PYTHON_DFC":
        ctrl = PythonController(
            name="ctrl",
            dt_s=config.DT,
            vdc_nom=config.V_DC,
            controller_mode="DFC",
            pole_pairs=config.P_POLES,
            rs=config.R_S,
            ld=config.L_D,
            lq=config.L_Q,
            lambda_pm=config.LAMBDA_PM,
            j=config.J_ROTOR,
            b=config.B_FRIC,
            kp_speed=0.0039,
            ki_speed=0.0002,
            kp_d=0.0001,
            kp_q=0.0195,
            ki_d=0.0005,
            ki_q=0.0002,
            integral_limit=25.0,
            max_current=100.0,
            max_iq_dot=1000.0,
            startup_mod_min=0.05,
            startup_mod_max=0.25,
            startup_increment=0.001,
            spinning_past_index=spinning_past_index,
            stopped_past_index=stopped_past_index,
            use_python=True,
            debug=False
        )
        ctrl_label = "Python_DFC"

    elif controller_mode == "C_OPEN_LOOP":
        ctrl = EmbedSimControlBlock(
            name="ctrl",
            dt_s=config.DT,
            ctrl_alg=SIM_CTRL_OPEN_LOOP,
            vdc_nom=config.V_DC,
            use_c_backend=True,
        )
        ctrl_label = "C_OpenLoop"

    elif controller_mode == "C_DFC":
        ctrl = EmbedSimControlBlock(
            name="ctrl",
            dt_s=config.DT,
            ctrl_alg=SIM_CTRL_DFC,
            vdc_nom=config.V_DC,
            use_c_backend=True,
        )
        ctrl_label = "C_DFC"

    else:
        raise ValueError(f"Unknown CONTROLLER_MODE: {controller_mode}")

    return ctrl, ctrl_label


# =============================================================================
# Motor State Display Function (ENHANCED)
# =============================================================================

def display_motor_state(t: float, state: dict, prefix: str = "", verbose: bool = False):
    """
    Display motor state in a readable format.
    Now includes numeric flags and diagnostic fields.
    """
    if not state:
        return

    # Extract key fields with defaults
    closed_loop = state.get('closed_loop', 0)
    controller_mode = state.get('controller_mode', 0)   # 0=OPEN, 1=DFC (or other)
    speed = state.get('speed_rpm', 0.0)
    speed_ref = state.get('speed_ref_rpm', 0.0)
    id_val = state.get('id', 0.0)
    iq_val = state.get('iq', 0.0)
    duty_u = state.get('duty_u', 0.5)
    duty_v = state.get('duty_v', 0.5)
    duty_w = state.get('duty_w', 0.5)
    torque = state.get('torque_total', 0.0)
    spin_counter = state.get('spinning_counter', 0)
    speed_error = state.get('speed_error_rpm', 0.0)

    # Determine mode string
    mode_str = "CLOSED" if closed_loop else "OPEN"

    # Build basic line
    line = (
        f"{prefix}[{t:6.2f}s] {mode_str}  "
        f"ω={speed:6.1f} RPM  "
        f"ω_ref={speed_ref:6.1f}  "
        f"Id={id_val:6.3f}A  "
        f"Iq={iq_val:6.3f}A  "
        f"T={torque:6.3f}Nm  "
        f"spin={spin_counter:6d}  "
        f"duty=({duty_u:.3f},{duty_v:.3f},{duty_w:.3f})"
    )

    # Append extra diagnostic info if verbose or if closed_loop=0 but speed error is small
    show_diag = verbose or (closed_loop == 0 and abs(speed_error) < 50 and speed > 100)
    if show_diag:
        line += (
            f"  [closed_loop={closed_loop}  ctrl_mode={controller_mode}  "
            f"err={speed_error:6.1f} RPM]"
        )

    print(line)


# =============================================================================
# Plotting Functions - RPM Only
# =============================================================================

def plot_rpm(sim, ctrl_label, plant_mode, config):
    """Plot RPM response only."""
    ph = create_plotter(sim)

    plot_title = f"Speed Control - {ctrl_label} ({plant_mode} Plant)"
    plot_path = str(config.PLOT_DIR / f"{ctrl_label.lower()}_{plant_mode.lower()}_speed.png")

    ph.easyplot(
        ["SpeedRef[0]", "Motor[0]"],
        title=plot_title,
        time_range=(0, config.T_SIM),
        figsize=(10, 4),
        save_path=plot_path if config.SAVE_PLOT else None
    )


# =============================================================================
# Custom simulation runner with state logging
# =============================================================================

def run_simulation_with_logging(sim: EmbedSim, config: SimulationConfig):
    """
    Run the simulation and log C motor state by re-implementing the run loop.
    """
    steps = int(config.T_SIM / config.DT)
    last_log_time = -1.0
    motor_states = []
    total_steps = 0

    print("  Running simulation with step-by-step logging...")

    # Get the blocks in execution order
    blocks = sim.execution_order
    dynamic_blocks = sim.dynamic_blocks

    # Reset all blocks
    for b in blocks:
        b.reset()

    t = 0.0

    # Progress bar
    bar_length = 50

    for step in range(steps):
        # Compute all blocks
        for block in blocks:
            # Get inputs from connected blocks
            if len(block.inputs) > 0:
                input_values = []
                for inp in block.inputs:
                    if inp.output is not None:
                        input_values.append(inp.output)
                    else:
                        input_values.append(VectorSignal([0.0]))
            else:
                input_values = None

            block.compute(t, config.DT, input_values)

        # Record signals
        sim.scope.record(t)

        # Integrate dynamics (Euler for simplicity)
        for b in dynamic_blocks:
            if hasattr(b, 'get_derivative'):
                input_values = [inp.output for inp in b.inputs] if b.inputs else None
                derivative = b.get_derivative(t, input_values)
                if derivative is not None:
                    b.state = b.state + derivative * config.DT

        total_steps += 1

        # Log motor state at interval
        if t - last_log_time >= config.LOG_INTERVAL:
            last_log_time = t

            if HAS_C_WRAPPER:
                try:
                    state = get_motor_state()
                    if state and state.get('valid', 0):
                        motor_states.append((t, state))
                        # Use verbose flag to show diagnostics if closed_loop is 0 but speed is high
                        verbose = (state.get('closed_loop', 0) == 0 and state.get('speed_rpm', 0) > 100)
                        display_motor_state(t, state, verbose=verbose)
                except Exception as e:
                    # Silent fail during simulation
                    pass

            # Update progress
            progress = (step + 1) / steps
            bar = "█" * int(progress * bar_length) + "░" * (bar_length - int(progress * bar_length))
            print(f"\r  Progress: [{bar}] {progress*100:.1f}%", end="", flush=True)

        t += config.DT

    # Final step
    for block in blocks:
        if len(block.inputs) > 0:
            input_values = []
            for inp in block.inputs:
                if inp.output is not None:
                    input_values.append(inp.output)
                else:
                    input_values.append(VectorSignal([0.0]))
        else:
            input_values = None
        block.compute(t, config.DT, input_values)
    sim.scope.record(t)

    print()  # New line after progress bar
    print(f"  ✅ Completed {total_steps} steps")

    return motor_states


# =============================================================================
# Main Simulation
# =============================================================================

def main():
    """Main simulation entry point."""

    config = SimulationConfig()

    print(f"\n{'='*60}")
    print(" PMSM CONTROL SIMULATION")
    print(f"{'='*60}")
    print(f"  Target: {config.TARGET_RPM:.0f} RPM")
    print(f"  Time: {config.T_SIM:.1f}s")
    print(f"  dt: {config.DT*1e6:.0f}us")
    print(f"  Vdc: {config.V_DC:.1f}V")
    print(f"  Controller: {CONTROLLER_MODE}")
    print(f"  Plant: {PLANT_MODE}")
    print(f"  Log interval: {config.LOG_INTERVAL:.1f}s")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------
    # 1. Create blocks
    # ------------------------------------------------------------

    speed_ref = VectorStep(
        "speed_ref",
        step_time=config.STEP_TIME,
        before_value=0.0,
        after_value=config.TARGET_RPM,
        dim=1
    )

    motor, motor_out_size, plant_type = create_plant(PLANT_MODE, config)
    ctrl, ctrl_label = create_controller(CONTROLLER_MODE, config, plant_type)

    valid_flag = 1
    ctrl_packer = CtrlPacker("ctrl_packer", vdc=config.V_DC, valid_flag=valid_flag)
    load_adapter = LoadAdapter("load_adapter", vdc=config.V_DC, tload=0.0)
    motor_delay = MotorVectorDelay("motor_delay", vector_size=motor_out_size)
    sink = VectorEnd("sink")

    # ------------------------------------------------------------
    # 2. Connect blocks
    # ------------------------------------------------------------

    speed_ref >> ctrl_packer
    motor_delay >> ctrl_packer
    ctrl_packer >> ctrl
    ctrl >> load_adapter
    load_adapter >> motor
    motor >> motor_delay
    motor >> sink

    # ------------------------------------------------------------
    # 3. Create simulation object
    # ------------------------------------------------------------

    sim = EmbedSim(
        sinks=[sink],
        T=config.T_SIM,
        dt=config.DT,
        solver=ODESolver.EULER
    )

    # ------------------------------------------------------------
    # 4. Add signals to scope for plotting
    # ------------------------------------------------------------

    sim.scope.add(speed_ref, indices=[0], label="SpeedRef")
    sim.scope.add(motor, indices=[0], label="Motor")  # Only speed (index 0) for RPM plot

    # ------------------------------------------------------------
    # 5. Initialize C controller if using C backend
    # ------------------------------------------------------------

    if CONTROLLER_MODE in ["C_OPEN_LOOP", "C_DFC"] and HAS_C_WRAPPER:
        try:
            control_init()
            print("✅ C controller initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize C controller: {e}")

    # ------------------------------------------------------------
    # 6. Run simulation
    # ------------------------------------------------------------

    print(f" Starting simulation with {PLANT_MODE} plant...\n")

    if CONTROLLER_MODE in ["C_OPEN_LOOP", "C_DFC"] and HAS_C_WRAPPER:
        motor_states = run_simulation_with_logging(sim, config)
    else:
        sim.run(progress_bar=True)
        motor_states = []

    # ------------------------------------------------------------
    # 7. Get final motor state (with full dump)
    # ------------------------------------------------------------

    if CONTROLLER_MODE in ["C_OPEN_LOOP", "C_DFC"] and HAS_C_WRAPPER:
        try:
            state = get_motor_state()
            if state and state.get('valid', 0):
                if not motor_states or motor_states[-1][0] < config.T_SIM:
                    motor_states.append((config.T_SIM, state))
                print("\n📊 Final C Motor State (full):")
                # Print all keys and values for diagnosis
                for key, value in state.items():
                    print(f"    {key:20s} = {value}")
                # Also a compact final display
                print("\n📊 Final Summary:")
                display_motor_state(config.T_SIM, state, verbose=True)
        except Exception as e:
            print(f"⚠️ Could not get motor state: {e}")

    print(f"\n✅ Logged {len(motor_states)} motor states during simulation")

    # ------------------------------------------------------------
    # 8. Plot RPM only
    # ------------------------------------------------------------

    plot_rpm(sim, ctrl_label, PLANT_MODE, config)

    # ------------------------------------------------------------
    # 9. Analysis
    # ------------------------------------------------------------

    sc = sim.scope
    speed_data = sc.get_signal("Motor", 0)

    if speed_data is None or len(speed_data) == 0:
        print("Error: No speed data available")
        return 1

    final_speed = speed_data[-1]

    if len(speed_data) > 100:
        steady_start = int(len(speed_data) * 0.9)
        steady_speed = np.mean(speed_data[steady_start:])
        steady_std = np.std(speed_data[steady_start:])
    else:
        steady_speed = final_speed
        steady_std = 0.0

    switched_to_closed_loop = False
    if motor_states:
        for t, state in motor_states:
            if state.get('closed_loop', 0):
                switched_to_closed_loop = True
                break

    print(f"\n{'='*60}")
    print(f" {ctrl_label} SUMMARY ({PLANT_MODE} Plant)")
    print(f"{'='*60}")
    print(f"  Target speed: {config.TARGET_RPM:.0f} RPM")
    print(f"  Final speed: {final_speed:.1f} RPM")
    print(f"  Steady-state speed: {steady_speed:.1f} ± {steady_std:.1f} RPM")
    print(f"  Steady-state error: {steady_speed - config.TARGET_RPM:+.1f} RPM")
    print(f"  Closed-loop flag: {switched_to_closed_loop}")

    if abs(steady_speed - config.TARGET_RPM) < 50:
        if switched_to_closed_loop:
            print("  Status: SUCCESS ✅ (Closed-loop achieved)")
        else:
            print("  Status: PARTIAL ⚠️ (Speed stable but not in closed-loop)")
    else:
        print(f"  Status: RUNNING at {steady_speed:.1f} RPM (Open-loop)")

    print(f"{'='*60}")

    plt.show()
    return 0


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())