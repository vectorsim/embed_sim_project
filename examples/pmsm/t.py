# =============================================================================
# pmsm_foc_pwm_smc.py (DIRECT CLOSED-LOOP - NO OPEN-LOOP STARTUP)
# =============================================================================

import sys
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Path bootstrap (same as your original)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.insert(0, _HERE)
    from _path_utils import get_project_root

    _PROJECT_ROOT = str(get_project_root())
    _ELEC_BLOCKS = os.path.join(_PROJECT_ROOT, "electrical_blocks")
except ImportError:
    _PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
    _ELEC_BLOCKS = os.path.abspath(os.path.join(_HERE, "..", "..", "electrical_blocks"))

for _p in [_PROJECT_ROOT, _ELEC_BLOCKS]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# EmbedSim core
from embedsim.code_generator import SimBlockBase, CodeGenStart, CodeGenEnd
from embedsim.core_blocks import VectorBlock, VectorSignal
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver, LoopBreaker, VectorDelay
from embedsim.source_blocks import VectorConstant

# Your existing blocks
from coordinate_transform_blocks import (
    ClarkeTransformBlock,
    InvClarkeTransformBlock,
    ParkTransformBlock,
    InvParkTransformBlock,
)
from speed_pi_block import SpeedPIBlock
from smc_block import SMCBlock
from svpwm_block import SVPWMBlock

# NEW motor model
from PMSM_Motor_WithSensorsBlock import PMSM_Motor_WithSensorsBlock

# ---------------------------------------------------------------------------
# FMU path for the new motor model
# ---------------------------------------------------------------------------
_FMU_PATH = os.path.join(_ELEC_BLOCKS, "modelica", "PMSM_Motor_WithSensors.fmu")

# =============================================================================
# DEBUG FLAG
# =============================================================================
DEBUG = False

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================
V_DC = 48.0
POLE_PAIRS = 2
R_S = 0.5
L_D = 0.005
L_Q = 0.006
LAMBDA_PM = 0.175
T_LOAD = 0.0

SPEED_RPM = 600.0
SPEED_RAD_S = SPEED_RPM * 2.0 * np.pi / 60.0

T_END = 1.0
DT = 1e-4

V_MAX = V_DC / np.sqrt(3.0)

# =============================================================================
# SPEED PI GAINS
# =============================================================================
KP_SPEED = 0.5  # Reduced for smoother startup
KI_SPEED = 5.0  # Reduced for smoother startup
IQ_MAX = 15.0

# =============================================================================
# SMC GAINS
# =============================================================================
LAMBDA_D = 80.0
K_SW_D = V_MAX
PHI_D = 2.0

LAMBDA_Q = 80.0
K_SW_Q = V_MAX
PHI_Q = 2.0

V_SMC_MAX = V_MAX


# =============================================================================
# Simple blocks
# =============================================================================

class MotorOutputBlock(VectorBlock):
    """Read motor outputs and print them for debugging"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 5  # [i_a, i_b, i_c, theta_m, omega_m]
        self.output_label = "[ia,ib,ic,θm,ωm]"

    # FMU OUTPUT_VARS index map (matches PMSM_Motor_WithSensorsBlock.OUTPUT_VARS):
    # 0:i_a  1:i_b  2:i_c  3:theta_m  4:omega_m_out  5:emf_a  6:emf_b  7:emf_c
    # 8:speed_rpm  9:T_em_out  10:i_d  11:i_q
    _FMU_IDX = {'i_a': 0, 'i_b': 1, 'i_c': 2, 'theta_m': 3, 'omega_m_out': 4}

    def compute(self, t, dt, input_values=None):
        out = np.zeros(5, dtype=np.float32)
        if input_values and input_values[0] is not None:
            raw = input_values[0].value

            if isinstance(raw, dict):
                # FMU returned a named dict
                out[0] = float(raw.get('i_a', 0.0))
                out[1] = float(raw.get('i_b', 0.0))
                out[2] = float(raw.get('i_c', 0.0))
                out[3] = float(raw.get('theta_m', 0.0))
                out[4] = float(raw.get('omega_m_out', 0.0))
            elif hasattr(raw, '__len__') and len(raw) >= 5:
                # FMU returned flat numpy array aligned with OUTPUT_VARS order
                out[0] = float(raw[0])  # i_a
                out[1] = float(raw[1])  # i_b
                out[2] = float(raw[2])  # i_c
                out[3] = float(raw[3])  # theta_m
                out[4] = float(raw[4])  # omega_m_out
            elif hasattr(raw, '__len__') and len(raw) == 5:
                # Already packed by a previous MotorOutputBlock pass (VectorDelay)
                out = raw.copy().astype(np.float32)
            # else: size-1 placeholder from engine — leave as zeros(5)

        self.output = VectorSignal(out, self.name)
        return self.output


class MotorCurrentsBlock(VectorBlock):
    """Extract currents from motor outputs"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 3
        self.output_label = "[ia,ib,ic]"

    def compute(self, t, dt, input_values=None):
        currents = np.zeros(3, dtype=np.float32)
        if input_values and input_values[0] is not None:
            motor_out = input_values[0].value
            if hasattr(motor_out, '__len__') and len(motor_out) >= 3:
                currents = np.array(motor_out[:3], dtype=np.float32)
        self.output = VectorSignal(currents, self.name)
        return self.output


class MotorAngleBlock(VectorBlock):
    """Extract angle from motor outputs and convert to electrical angle"""

    def __init__(self, name: str, pole_pairs: int):
        super().__init__(name)
        self.pole_pairs = pole_pairs
        self.vector_size = 1
        self.output_label = "θe"

    def compute(self, t, dt, input_values=None):
        theta_e = 0.0
        if input_values and input_values[0] is not None:
            motor_out = input_values[0].value
            if hasattr(motor_out, '__len__') and len(motor_out) >= 4:
                theta_m = motor_out[3]
                theta_e = float(theta_m) * self.pole_pairs
        self.output = VectorSignal(np.array([theta_e], dtype=np.float32), self.name)
        return self.output


class MotorSpeedBlock(VectorBlock):
    """Extract speed from motor outputs"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 1
        self.output_label = "ωm"

    def compute(self, t, dt, input_values=None):
        omega_m = 0.0
        if input_values and input_values[0] is not None:
            motor_out = input_values[0].value
            if hasattr(motor_out, '__len__') and len(motor_out) >= 5:
                omega_m = float(motor_out[4])
        self.output = VectorSignal(np.array([omega_m], dtype=np.float32), self.name)
        return self.output


class DQRefBlock(VectorBlock):
    """DQ reference (id_ref=0, iq_ref from constant)"""

    def __init__(self, name: str):
        super().__init__(name)
        self.vector_size = 2
        self.output_label = "[id_ref,iq_ref]"

    def compute(self, t, dt, input_values=None):
        iq_ref_val = input_values[0].value[1] if (input_values and len(input_values[0].value) >= 2) else 0.0
        self.output = VectorSignal(np.array([0.0, iq_ref_val], dtype=np.float32), self.name)
        return self.output


# =============================================================================
# Build simulation - NO OPEN-LOOP, DIRECT CLOSED-LOOP FOC
# =============================================================================

def build_simulation():
    """Instantiate all blocks, wire signals — direct closed-loop FOC without open-loop startup."""

    # =========================================================================
    # Motor plant with realistic sensors
    # =========================================================================
    motor = PMSM_Motor_WithSensorsBlock(
        name="motor",
        fmu_path=_FMU_PATH,
        R=R_S,
        L_d=L_D,
        L_q=L_Q,
        lambda_pm=LAMBDA_PM,
        J=0.002,
        B=0.001,
        p=float(POLE_PAIRS),
    )

    # =========================================================================
    # Motor output monitor + signal extractors
    # =========================================================================
    motor_out = MotorOutputBlock("motor_out")
    motor >> motor_out

    # Break the algebraic loop
    motor_fb = VectorDelay("motor_fb", initial=[0.0, 0.0, 0.0, 0.0, 0.0])
    motor_out >> motor_fb

    motor_currents = MotorCurrentsBlock("motor_currents")
    motor_angle = MotorAngleBlock("motor_angle", POLE_PAIRS)
    motor_speed = MotorSpeedBlock("motor_speed")

    motor_fb >> motor_currents
    motor_fb >> motor_angle
    motor_fb >> motor_speed

    # Ensure motor is always in the execution graph
    motor_sink = VectorEnd("motor_sink")
    motor >> motor_sink

    # =========================================================================
    # Closed-loop FOC controller chain - NO OPEN-LOOP STARTUP
    # =========================================================================

    # 1. Speed reference
    speed_ref = VectorConstant("speed_ref", value=[SPEED_RAD_S])

    # 2. Speed PI - enabled from t=0 (no open-loop)
    speed_pi = SpeedPIBlock(
        "speed_pi",
        Kp=KP_SPEED,
        Ki=KI_SPEED,
        i_max=IQ_MAX,
        t_enable=0.0,  # Start immediately
    )
    speed_ref >> speed_pi
    motor_speed >> speed_pi

    # 3. DQ reference
    dq_ref = DQRefBlock("dq_ref")
    speed_pi >> dq_ref

    # 4. Clarke transform
    clarke = ClarkeTransformBlock("clarke")
    motor_currents >> clarke

    # 5. Park transform
    park = ParkTransformBlock("park")
    clarke >> park
    motor_angle >> park

    # 6. SMC - enabled from t=0
    smc = SMCBlock(
        "smc",
        lambda_d=LAMBDA_D, K_sw_d=K_SW_D, phi_d=PHI_D,
        lambda_q=LAMBDA_Q, K_sw_q=K_SW_Q, phi_q=PHI_Q,
        out_min=-V_SMC_MAX, out_max=V_SMC_MAX,
        t_enable=0.0,  # Start immediately
    )
    dq_ref >> smc
    park >> smc

    # 7. Inverse Park
    inv_park = InvParkTransformBlock("inv_park")
    smc >> inv_park
    motor_angle >> inv_park

    # 8. Inverse Clarke
    inv_clarke = InvClarkeTransformBlock("inv_clarke")
    inv_park >> inv_clarke

    # 9. SVPWM (using default parameters - its internal startup will still run)
    svpwm = SVPWMBlock("svpwm", v_dc=V_DC)
    inv_clarke >> svpwm

    # Motor inputs
    vdc_src = VectorConstant("vdc", value=[V_DC])
    tload_src = VectorConstant("tload", value=[T_LOAD])

    # Connect SVPWM directly to motor
    svpwm >> motor  # port 0: duty cycles
    vdc_src >> motor  # port 1: v_dc
    tload_src >> motor  # port 2: T_load

    # =========================================================================
    # Sinks
    # =========================================================================
    sink = VectorEnd("sink")
    motor_out >> sink

    # =========================================================================
    # Simulation engine
    # =========================================================================
    sim = EmbedSim(
        sinks=[sink, motor_sink],
        T=T_END,
        dt=DT,
        solver=ODESolver.RK4,
    )

    # Scope channels
    sim.scope.add(motor_speed, label="speed")
    sim.scope.add(motor_currents, label="currents")
    sim.scope.add(motor_angle, label="angle")
    sim.scope.add(svpwm, label="duties")
    sim.scope.add(park, label="dq_currents")
    sim.scope.add(smc, label="vdq")

    sim.motor = motor
    return sim


# =============================================================================
# Plotting
# =============================================================================

def _s(sim, label, idx=0):
    key = f"{label}[{idx}]"
    d = sim.scope.data.get(key)
    return np.array(d) if d is not None else None


def plot_results(sim, out_path: str):
    t = np.array(sim.scope.t)
    if len(t) == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Speed
    speed = _s(sim, "speed", 0)
    if speed is not None:
        rpm = speed * 60.0 / (2.0 * np.pi)
        ax1.plot(t, rpm, 'b-', lw=1.5)
        ax1.axhline(SPEED_RPM, color='r', linestyle='--', lw=1, label=f"Target {SPEED_RPM} RPM")
        ax1.set_ylabel('RPM')
        ax1.set_title('Motor Speed')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Angle
    angle = _s(sim, "angle", 0)
    if angle is not None:
        ax2.plot(t, np.sin(angle), 'g-', lw=1.5)
        ax2.set_ylabel('sin(θ)')
        ax2.set_title('Electrical Angle')
        ax2.grid(True, alpha=0.3)

    # Currents
    for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['i_a', 'i_b', 'i_c'])):
        current = _s(sim, "currents", i)
        if current is not None:
            ax3.plot(t, current, color=color, lw=1, label=label, alpha=0.7)
    ax3.set_ylabel('Current (A)')
    ax3.set_title('Phase Currents')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # PWM duties
    for i, (color, label) in enumerate(zip(['r', 'g', 'b'], ['d_a', 'd_b', 'd_c'])):
        duty = _s(sim, "duties", i)
        if duty is not None:
            ax4.plot(t, duty, color=color, lw=1, label=label, alpha=0.7)
    ax4.axhline(0.5, color='k', linestyle='--', lw=0.5, alpha=0.5)
    ax4.set_ylabel('Duty')
    ax4.set_title('PWM Duties')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Info panel
    ax5.axis('off')
    ax6.axis('off')

    info_text = f"""
    PMSM FOC - DIRECT CLOSED-LOOP (NO OPEN-LOOP STARTUP)

    Parameters:
    • Target Speed: {SPEED_RPM} RPM
    • DC Voltage: {V_DC} V
    • dt: {DT * 1000:.2f} ms

    Controller:
    • Speed PI: Kp={KP_SPEED}, Ki={KI_SPEED}, Imax={IQ_MAX}A
    • SMC: λ=80, φ=2A, Ksw={V_SMC_MAX:.1f}V

    Note: SVPWM internal startup (0-0.15s) still active
    The motor will start from standstill using closed-loop FOC
    """
    ax5.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', transform=ax5.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("PMSM FOC - DIRECT CLOSED-LOOP (NO OPEN-LOOP STARTUP)")
    print("=" * 70)
    print(f"Target Speed: {SPEED_RPM} RPM")
    print(f"DC Voltage: {V_DC} V")
    print(f"dt: {DT * 1000:.2f} ms")
    print("=" * 70)

    sim = build_simulation()

    print("\nInitializing FMU...")
    sim.motor.initialize_fmu(t_start=0)

    print("\n⚙️  Running simulation...")
    sim.run()
    print("Simulation complete!")

    # Results
    t = np.array(sim.scope.t)
    speed_data = np.array(sim.scope.data.get("speed[0]", [0]))
    final_rpm = speed_data[-1] * 60.0 / (2.0 * np.pi) if len(speed_data) > 0 else 0

    current_data = np.array(sim.scope.data.get("currents[0]", [0]))
    max_current = np.max(np.abs(current_data)) if len(current_data) > 0 else 0

    print(f"\nFinal Speed: {final_rpm:.1f} RPM")
    print(f"Target: {SPEED_RPM} RPM")
    print(f"Error: {abs(final_rpm - SPEED_RPM):.1f} RPM")
    print(f"Max current: {max_current:.2f} A")

    # Plot
    out_png = os.path.join(_HERE, "pmsm_foc_results_direct.png")
    plot_results(sim, out_png)


if __name__ == "__main__":
    main()