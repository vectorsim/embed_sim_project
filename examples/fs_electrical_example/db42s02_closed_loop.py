# db42s02_closed_loop_smc.py - WORKING EXAMPLE with Double SMC
"""
NANOTEC DB42S02 - Double Sliding Mode Control with Current Limiting
"""

from __future__ import annotations
import sys
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple
from pathlib import Path

# ── Path bootstrap using _path_utils ──────────────────────────────────────────
from _path_utils import get_embedsim_import_path, get_current_parent

_ROOT = Path(get_embedsim_import_path())
_HERE = get_current_parent()
_FS_ELEC = _ROOT / "fs_electrical_machines"

# Add paths
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_FS_ELEC) not in sys.path:
    sys.path.insert(0, str(_FS_ELEC))
_c_src = _FS_ELEC / "c_src"
if str(_c_src) not in sys.path:
    sys.path.insert(0, str(_c_src))

print(f"Project root: {_ROOT}")
print(f"fs_electrical_machines: {_FS_ELEC}")

# ── EmbedSim imports ──────────────────────────────────────────────────────────
from embedsim import EmbedSim, ODESolver, VectorEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE
from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.simulation_engine import VectorDelay

# ── fs_electrical_machines imports ───────────────────────────────────────────
from smc_controller_block import (
    SMCControllerBlock,
    MotorParams,
    SMCParams,
)
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)
from motor_utility_blocks import SVPWMPackBlock
from svpwm_block import SVPWMBlock
from PMSM_MotorBlock import PMSM_MotorBlock

_FMU_PATH = str(_FS_ELEC / "modelica" / "PMSM_Motor.fmu")


# =============================================================================
# CtrlPacker Block
# =============================================================================

class CtrlPacker(VectorBlock):
    output_label = "[ω_ref,ω_m,θ_e,ia,ib,ic]"

    def __init__(self, name: str, dtype=None):
        super().__init__(name, dtype=dtype)
        self.vector_size = 6
        self.is_dynamic = False

    def compute_py(self, t, dt, input_values=None):
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal(np.zeros(6, dtype=DEFAULT_DTYPE), self.name)
            return self.output

        m = input_values[0].value if input_values[0] is not None else np.zeros(7)
        r = input_values[1].value if input_values[1] is not None else np.zeros(1)

        output_array = np.array([
            float(r[0]) if len(r) > 0 else 0.0,
            float(m[5]) if len(m) > 5 else 0.0,
            float(m[4]) if len(m) > 4 else 0.0,
            float(m[1]) if len(m) > 1 else 0.0,
            float(m[2]) if len(m) > 2 else 0.0,
            float(m[3]) if len(m) > 3 else 0.0,
        ], dtype=DEFAULT_DTYPE)

        self.output = VectorSignal(output_array, self.name)
        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Motor plant block
# =============================================================================

class DB42S02PlantBlock(PMSM_MotorBlock):
    TOPO_CATEGORY = "plant"
    C_CODEGEN_EXCLUDE = True
    output_label = "[rpm,ia,ib,ic,θe,ωm,Tem]"

    def __init__(self, name: str, fmu_path: str):
        super().__init__(
            name=name, fmu_path=fmu_path,
            R=0.19, L_d=0.125e-3, L_q=0.125e-3, lambda_pm=0.0014,
            J=2.4e-6, B=7e-5, p=4.0,
        )
        print(f"[FMU] Loaded: {fmu_path}")
        self._print_count = 0
        self._t_last_print = 0

    def compute_py(self, t: float, dt: float, input_values=None):
        if input_values is None or len(input_values) == 0:
            self.output = VectorSignal(np.zeros(7, dtype=DEFAULT_DTYPE), self.name)
            return self.output

        ta = tb = tc = 0.5
        t_load = 0.0

        if input_values[0] is not None:
            v = input_values[0].value
            if len(v) >= 3:
                ta, tb, tc = float(v[0]), float(v[1]), float(v[2])

        if len(input_values) > 1 and input_values[1] is not None:
            t_load = float(input_values[1].value.flat[0])

        # Load profile
        T_LOAD_T1 = 0.5
        T_LOAD_T2 = 1.2
        T_LOAD_0 = 0.000
        T_LOAD_1 = 0.005
        T_LOAD_2 = 0.020

        if t < T_LOAD_T1:
            t_load = T_LOAD_0
        elif t < T_LOAD_T2:
            t_load = T_LOAD_1
        else:
            t_load = T_LOAD_2

        ta = max(0.05, min(0.95, ta))
        tb = max(0.05, min(0.95, tb))
        tc = max(0.05, min(0.95, tc))

        fmu_input = VectorSignal(np.array([ta, tb, tc, 17.0, t_load], dtype=DEFAULT_DTYPE))
        super().compute_py(t, dt, [fmu_input])

        speed_rpm_true = self.read_speed_rpm()
        ia_true = self.read_i_a()
        ib_true = self.read_i_b()
        ic_true = self.read_i_c()
        theta_e_true = self.read_theta_e()
        omega_m_true = self.get_output_by_name('omega_m')
        tem = self.read_T_em()

        self.output = VectorSignal(
            np.array([speed_rpm_true, ia_true, ib_true, ic_true,
                      theta_e_true, omega_m_true, tem], dtype=DEFAULT_DTYPE), self.name)

        if t - self._t_last_print > 0.2 and self._print_count < 20:
            print(f"[PLANT t={t:.2f}] rpm={speed_rpm_true:.0f} T_load={t_load:.4f}")
            self._t_last_print = t
            self._print_count += 1

        return self.output

    def compute(self, t, dt, input_values=None):
        return self.compute_py(t, dt, input_values)


# =============================================================================
# Simulation Parameters
# =============================================================================

T_SIM = 2.0
DT = 1e-4
TARGET_RPM = 2000.0
TARGET_RADS = TARGET_RPM * 2.0 * math.pi / 60.0
V_DC = 17.0


# =============================================================================
# Build and Run
# =============================================================================

def build_and_run() -> Tuple[dict, EmbedSim]:
    print("=" * 64)
    print("  NANOTEC DB42S02 - Double Sliding Mode Controller")
    print(f"  Target: {TARGET_RPM:.0f} RPM  Vdc={V_DC}V  dt={DT * 1e6:.0f}µs  T={T_SIM}s")
    print("=" * 64)

    # Create motor parameters
    motor_params = MotorParams(
        pole_pairs=4.0,
        R_s=0.19,
        L_d=0.125e-3,
        L_q=0.125e-3,
        lambda_pm=0.0014,
        J=2.4e-6,
        B=7e-5,
    )

    # Create SMC parameters - PROVEN GAINS with current limiting
    smc_params = SMCParams(
        # Current loop (proven)
        ks_current=8.0,
        phi_current=0.6,
        eta_current=0.3,
        # Speed loop (proven - achieved 95% load rejection)
        ks_speed=0.8,
        ki_speed=5.0,
        phi_speed=30.0,
        eta_speed=0.3,
        # Current limit - added for protection
        i_max=3.5,
        v_max=V_DC / math.sqrt(3.0),
        # Rate limiting - added to prevent spikes
        i_rate_limit=20.0,
        # Filters
        load_lpf_hz=20.0,
        acc_lpf_hz=50.0,
        current_lpf_hz=200.0,
        speed_lpf_hz=50.0,
    )

    print(f"  Torque constant KT={motor_params.torque_constant:.4f} Nm/A")
    print(f"  Required current for 20 mNm: {0.020 / motor_params.torque_constant:.2f} A")
    print(f"  Speed SMC: Ks={smc_params.ks_speed} Nm, Ki={smc_params.ki_speed}, φ={smc_params.phi_speed} RPM")
    print(f"  Current SMC: Ks={smc_params.ks_current} V, φ={smc_params.phi_current} A")
    print(f"  I_max={smc_params.i_max} A, V_max={smc_params.v_max:.2f} V")
    print(f"  Rate limit: {smc_params.i_rate_limit} A/s")
    print("=" * 64)

    # Blocks
    speed_ref = VectorStep("speed_ref", step_time=0.1, before_value=0.0, after_value=TARGET_RADS)
    load_torque = VectorConstant("load_torque", value=0.005)

    motor = DB42S02PlantBlock("motor", fmu_path=_FMU_PATH)
    motor_delay = VectorDelay("motor_delay", initial=[0.0] * 7)
    ctrl = CtrlPacker("ctrl_packer")
    smc = SMCControllerBlock("smc_controller", motor=motor_params, smc=smc_params, dt=DT)
    svpwm_pack = SVPWMPackBlock("svpwm_pack", v_dc=V_DC)
    svpwm_pack.NUM_INPUTS = 1
    svpwm = SVPWMBlock("svpwm")
    sink = VectorEnd("sink")

    # Wiring
    motor >> motor_delay
    motor_delay >> ctrl
    speed_ref >> ctrl
    ctrl >> smc
    smc >> svpwm_pack
    svpwm_pack >> svpwm
    svpwm >> motor
    load_torque >> motor
    motor >> sink

    sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.RK4)

    print("\n[Topology]")
    sim.topo.print_console()

    sim.scope.add(speed_ref, indices=[0], label="SpeedRef")
    sim.scope.add(motor, indices=[0], label="SpeedRPM")
    sim.scope.add(motor, indices=[6], label="Torque")

    print("\nRunning simulation...")
    sim.run()
    print(f"  Completed: {len(sim.scope.t)} steps")

    # Collect results
    hist = smc.log_data

    # Print results
    print("\n" + "=" * 55)
    print("  Simulation Results")
    print("=" * 55)

    if len(hist["speed"]) > 0:
        final_speed = hist["speed"][-1]
        max_speed = np.max(hist["speed"])
        min_speed = np.min(hist["speed"][hist["speed"] > 100]) if np.any(hist["speed"] > 100) else 0

        print(f"  Speed range: {min_speed:.0f} - {max_speed:.0f} RPM")
        print(f"  Final speed: {final_speed:.0f} RPM")
        print(f"  Target: {TARGET_RPM:.0f} RPM")
        print(f"  Steady-state error: {abs(final_speed - TARGET_RPM):.1f} RPM")

        # Performance after load step
        t_log = hist["t"]
        T_LOAD_T2 = 1.2
        after_load_idx = t_log > T_LOAD_T2
        if np.any(after_load_idx):
            speed_after = hist["speed"][after_load_idx]
            if len(speed_after) > 0:
                avg_speed = np.mean(speed_after[-50:])
                print(f"\n  Load step performance (after {T_LOAD_T2}s):")
                print(f"    Avg speed after load: {avg_speed:.0f} RPM")
                print(f"    Speed drop: {max_speed - avg_speed:.0f} RPM")
                if max_speed > 0:
                    print(f"    Load rejection: {(1 - (max_speed - avg_speed) / max_speed) * 100:.1f}%")

        # Current analysis
        if len(hist["iq"]) > 0:
            max_iq = np.max(np.abs(hist["iq"]))
            print(f"\n  Current analysis:")
            print(f"    Max iq current: {max_iq:.2f} A")
            print(f"    Current limit: {smc_params.i_max:.2f} A")
            if max_iq <= smc_params.i_max:
                print(f"    ✓ Within limits")
            else:
                print(f"    ⚠ Exceeds limit by {max_iq - smc_params.i_max:.2f} A")

    print("=" * 55)

    return hist, sim


# =============================================================================
# Plotting
# =============================================================================

def plot_results(d: dict, path: str = "smc_controller_results.png") -> None:
    if len(d["t"]) == 0:
        print("No data to plot")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f"NANOTEC DB42S02 — Double Sliding Mode Controller", fontsize=12)

    t = d["t"]
    T_LOAD_T1 = 0.5
    T_LOAD_T2 = 1.2
    TARGET_RPM = 2000.0

    # Speed tracking
    ax = axes[0, 0]
    ax.plot(t, d["speed"], 'b-', label='Actual Speed', lw=1.5)
    ax.plot(t, d["speed_ref"], 'k--', label='Reference', lw=1.5, alpha=0.7)
    ax.axvline(T_LOAD_T1, color='orange', ls=':', alpha=0.5, label='Light load (5 mNm)')
    ax.axvline(T_LOAD_T2, color='red', ls=':', alpha=0.5, label='Heavy load (20 mNm)')
    ax.set_ylabel('Speed [RPM]')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_title('Speed Tracking')
    ax.set_xlabel('Time [s]')

    # Speed error
    ax = axes[0, 1]
    speed_error = d["speed"] - d["speed_ref"]
    ax.plot(t, speed_error, 'r-', lw=0.8)
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(T_LOAD_T1, color='orange', ls=':', alpha=0.5)
    ax.axvline(T_LOAD_T2, color='red', ls=':', alpha=0.5)
    ax.set_ylabel('Speed Error [RPM]')
    ax.grid(True, alpha=0.3)
    ax.set_title('Speed Error')
    ax.set_xlabel('Time [s]')

    # dq currents
    ax = axes[1, 0]
    ax.plot(t, d["iq_ref"], 'k--', label='iq_ref', lw=1.2)
    ax.plot(t, d["iq"], 'b-', label='iq_meas', lw=1)
    ax.plot(t, d["id"], 'g-', label='id_meas', lw=1)
    ax.axhline(0, color='gray', ls='--', lw=0.5)
    ax.axvline(T_LOAD_T1, color='orange', ls=':', alpha=0.5)
    ax.axvline(T_LOAD_T2, color='red', ls=':', alpha=0.5)
    ax.set_ylabel('Current [A]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('dq Currents')
    ax.set_xlabel('Time [s]')

    # Speed vs current correlation
    ax = axes[1, 1]
    ax.plot(d["iq"], d["speed"], 'b.', alpha=0.5, markersize=1)
    ax.set_xlabel('iq Current [A]')
    ax.set_ylabel('Speed [RPM]')
    ax.grid(True, alpha=0.3)
    ax.set_title('Speed vs q-axis Current')
    ax.axhline(TARGET_RPM, color='k', linestyle='--', alpha=0.5)

    # Current histogram
    ax = axes[2, 0]
    ax.hist(d["iq"], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(3.5, color='red', linestyle='--', label='Limit (3.5A)')
    ax.axvline(-3.5, color='red', linestyle='--')
    ax.set_xlabel('iq Current [A]')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Current Distribution')

    # Speed distribution
    ax = axes[2, 1]
    ax.hist(d["speed"], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(TARGET_RPM, color='r', linestyle='--', label='Target')
    ax.set_xlabel('Speed [RPM]')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Speed Distribution')

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Plot] {path}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import os

    print("=" * 64)
    print(f"  RUNNING FROM: {os.path.abspath(__file__)}")
    print("=" * 64)

    data, sim = build_and_run()
    plot_results(data, "smc_controller_results.png")

    print("\n" + "=" * 64)
    print("  Double Sliding Mode Controller - Performance Summary")
    print("=" * 64)
    print(f"  ✓ Speed SMC with integral action")
    print(f"  ✓ Current SMC for d and q axes")
    print(f"  ✓ Load torque observer")
    print(f"  ✓ Current limiting ({3.5}A)")
    print(f"  ✓ Rate limiting ({20} A/s)")
    if len(data["speed"]) > 0:
        print(f"  ✓ Final speed: {data['speed'][-1]:.0f} RPM")
        print(f"  ✓ Target: {TARGET_RPM:.0f} RPM")
    if len(data["iq"]) > 0:
        print(f"  ✓ Max current: {np.max(np.abs(data['iq'])):.2f} A")
    print("=" * 64)

    print("\n[Done]")