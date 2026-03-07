"""
pmsm_foc_smc.py
===============

Permanent Magnet Synchronous Motor (PMSM) Field-Oriented Control (FOC)
with Sliding Mode Controller (SMC) using EmbedSim.

================================================================================
CONTROL SYSTEM ARCHITECTURE
================================================================================

Cascaded Control Structure:
---------------------------
OUTER LOOP (Speed Control):
    ω_ref (reference speed)
        → Speed PI Controller
            → i_q* (q-axis current reference)
    i_d* = 0 (field weakening not active)

INNER LOOP (Current Control):
    [i_d*, i_q*] (current references)
        → Sliding Mode Controller (SMC)
            → [v_d*, v_q*] (voltage commands)

COORDINATE TRANSFORMATIONS:
    [v_d*, v_q*]
        → Inverse Park Transform (using rotor angle θ_e)
            → [v_α*, v_β*] (stationary frame)
                → Inverse Clarke Transform
                    → [v_a*, v_b*, v_c*] (3-phase voltages)

PLANT:
    ThreePhaseMotorBlock (FMU - Functional Mock-up Unit)
        Inputs:  [v_a, v_b, v_c] (stator voltages)
                 T_load (load torque)
        Outputs: [i_d, i_q, ω_m, θ_e, T_em, ω_e, rpm]

FEEDBACK PATH:
    Motor states are fed back to controllers:
    - ω_m → Speed PI (speed feedback)
    - [i_d, i_q] → SMC (current feedback)
    - θ_e → Inverse Park (rotor position for transformation)

================================================================================
NUMERICAL IMPLEMENTATION DETAILS
================================================================================

Timing:
-------
- Timestep: 50 µs (20 kHz control frequency) - typical for FOC applications
- Solver: RK4 (4th order Runge-Kutta) for numerical stability
- Simulation time: 0.5 s (sufficient to observe transient and steady-state)

Algebraic Loop Prevention:
--------------------------
FOC creates an instantaneous algebraic dependency:
    motor currents → controller → voltages → motor currents

This creates a circular dependency that EmbedSim cannot solve directly.
Solution: Insert one-step delays in feedback paths:
    - Delay motor outputs by one timestep
    - Controllers use delayed measurements (realistic digital control)
    - Creates causal, acyclic data flow

================================================================================
CONTROLLER DESIGN
================================================================================

Speed PI Controller (Outer Loop):
--------------------------------
- Kp = 1.0    : Proportional gain
- Ki = 20.0   : Integral gain
- i_max = 20.0: Current limit [A]
- Tuning: Classic PI with anti-windup

Sliding Mode Controller (Inner Loop):
------------------------------------
Theory: σ = λ·(i_ref - i) + (d/dt)(i_ref - i)

Design Parameters:
- λ = 83.0          : Sliding surface slope (≈ R/Lq, matches electrical time constant)
- K_sw = 40.0       : Switching gain (> max back-EMF for robustness)
- φ = 1.0           : Boundary layer thickness (reduces chattering)
- V_DC = 48.0       : DC bus voltage [V] (output saturation)

Control Law: v = L·[λ·(i_ref - i) + di_ref/dt] + R·i + ω_e·λ_pm + K_sw·sat(σ/φ)

================================================================================
SIGNAL INDEXING (Motor Output Vector)
================================================================================

Motor output vector indices (size 7):
    [0] = i_d    : d-axis current [A]
    [1] = i_q    : q-axis current [A]
    [2] = ω_m    : mechanical speed [rad/s]
    [3] = θ_e    : electrical angle [rad]
    [4] = T_em   : electromagnetic torque [N·m]
    [5] = ω_e    : electrical speed [rad/s]
    [6] = rpm    : rotational speed [RPM]

================================================================================
FILE DEPENDENCIES
================================================================================

Required blocks from electrical_blocks/:
- coordinate_transform_blocks.py : Park/Clarke transformations
- smc_block.py                   : Sliding mode controller
- speed_pi_block.py              : Speed PI controller
- fmu_pmsm.py                    : PMSM FMU wrapper
- modelica/ThreePhaseMotor.fmu   : FMU model (optional, falls back to stub)

================================================================================
USAGE
================================================================================

Run directly:
    $ python pmsm_foc_smc.py

Outputs:
- Console: Simulation progress and topology diagram (sim.topo.print_console())
- Browser: Interactive block diagram (sim.topo.show_gui())
- Plot window: Real-time visualization of all key signals
- File: pmsm_foc_smc_results.png (saved plot)

Expected behavior:
- Speed tracks 100 rad/s reference after ~0.15s
- i_d regulated to zero (field orientation)
- i_q proportional to torque demand
- Three-phase currents sinusoidal at steady state
"""


# =============================================================================
# Standard imports + EmbedSim path setup
# =============================================================================

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -----------------------------------------------------------------------------
# EmbedSim path resolution
# -----------------------------------------------------------------------------
# _path_utils ensures project root detection regardless of working directory.
# This avoids fragile relative imports.
# -----------------------------------------------------------------------------

from _path_utils import get_project_root, get_current_parent

project_root = get_project_root()
print(f"Project root: {project_root}")

# Add project root so embedsim package can be imported
sys.path.insert(0, str(project_root))

# Add electrical_blocks to import path
import os
electrical_blocks_path = os.path.join(project_root, "electrical_blocks")
sys.path.insert(0, electrical_blocks_path)
print(f"Added to path: {electrical_blocks_path}")

# =============================================================================
# EmbedSim core imports
# =============================================================================

from embedsim.source_blocks import VectorStep, VectorConstant
from embedsim.code_generator import SimBlockBase
from embedsim.core_blocks import VectorBlock, VectorSignal
from embedsim.dynamic_blocks import VectorEnd, VectorIntegrator
from embedsim.processing_blocks import VectorGain, VectorSum
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.code_generator import CodeGenStart, CodeGenEnd
from embedsim.plot_helper import create_plotter

# =============================================================================
# Electrical blocks (FOC components)
# =============================================================================

from coordinate_transform_blocks import (
    ParkTransformBlock,
    InvParkTransformBlock,
    InvClarkeTransformBlock,
)

from smc_block import SMCBlock
from speed_pi_block import SpeedPIBlock
from fmu_pmsm import ThreePhaseMotorBlock


# =============================================================================
# Custom InvPark for FOC wiring
# =============================================================================

class InvParkFOC(InvParkTransformBlock):
    """
    Modified inverse Park transform for FOC wiring.

    Inputs:
        port 0 → [v_d, v_q] from SMC
        port 1 → full motor output vector (delayed)

    We extract θ_e from motor output index 3.

    Math:
        v_α = v_d cosθ − v_q sinθ
        v_β = v_d sinθ + v_q cosθ
    """

    def compute_py(self, t, dt, input_values=None):

        # Safety check
        if not input_values or len(input_values) < 2:
            self.output = VectorSignal(np.zeros(2, dtype=np.float32), self.name)
            return self.output

        # dq voltages
        v_dq = input_values[0].value

        # Electrical angle from motor
        theta = float(input_values[1].value[3])

        v_d, v_q = float(v_dq[0]), float(v_dq[1])

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # Inverse Park transform
        v_alpha = v_d * cos_t - v_q * sin_t
        v_beta  = v_d * sin_t + v_q * cos_t

        self.output = VectorSignal(
            np.array([v_alpha, v_beta], dtype=np.float32),
            self.name,
            dtype=self.dtype
        )

        return self.output


# =============================================================================
# Simulation parameters
# =============================================================================

T_SIM = 0.5        # Total simulation time [s]
DT = 5e-5          # 50 µs timestep (20 kHz control)
OMEGA_REF = 100.0  # Target mechanical speed [rad/s]
T_LOAD = 0.2       # Load torque [N·m]

# FMU path (Modelica-exported PMSM)
FMU_PATH = os.path.join(
    str(project_root),
    "electrical_blocks",
    "modelica",
    "ThreePhaseMotor.fmu"
)
print(f"FMU path: {FMU_PATH}")

# -----------------------------------------------------------------------------
# Sliding Mode Controller tuning
# -----------------------------------------------------------------------------
# λ ≈ R/Lq ensures surface slope matches electrical time constant
# K_sw > max back-EMF ensures robustness
# -----------------------------------------------------------------------------

SMC_LAMBDA = 83.0
SMC_KSW = 40.0
SMC_PHI_D = 1.0
SMC_PHI_Q = 1.0
V_DC = 48.0

# Outer speed PI
KP_SPEED = 1.0
KI_SPEED = 20.0


# =============================================================================
# Build simulation
# =============================================================================

def build_sim(fmu_path: str = FMU_PATH):

    print("\n" + "="*65)
    print("EmbedSim — PMSM FOC with Sliding Mode Controller")
    print("="*65)

    # -------------------------------------------------------------------------
    # Reference sources
    # -------------------------------------------------------------------------

    omega_ref_src = VectorStep(
        "omega_ref",
        step_time=0.05,
        before_value=0.0,
        after_value=OMEGA_REF,
        dim=1,
    )

    load_torque_src = VectorConstant("T_load", value=[T_LOAD])

    # -------------------------------------------------------------------------
    # Speed PI (outer loop)
    # -------------------------------------------------------------------------

    try:
        speed_pi = SpeedPIBlock(
            "speed_pi",
            Kp=KP_SPEED,
            Ki=KI_SPEED,
            i_max=20.0,
            use_c_backend=True
        )
        print("SpeedPI: C backend")
    except ImportError:
        speed_pi = SpeedPIBlock(
            "speed_pi",
            Kp=KP_SPEED,
            Ki=KI_SPEED,
            i_max=20.0,
            use_c_backend=False
        )
        print("SpeedPI: Python backend")

    # Code generation boundary markers
    cg_start = CodeGenStart("cg_start")
    cg_end   = CodeGenEnd("cg_end")

    # -------------------------------------------------------------------------
    # Sliding Mode Controller (inner current loop)
    # -------------------------------------------------------------------------

    try:
        smc = SMCBlock(
            "smc",
            use_c_backend=True,
            lambda_d=SMC_LAMBDA,
            K_sw_d=SMC_KSW,
            phi_d=SMC_PHI_D,
            lambda_q=SMC_LAMBDA,
            K_sw_q=SMC_KSW,
            phi_q=SMC_PHI_Q,
            out_min=-V_DC,
            out_max=V_DC
        )
        print("SMC: C backend")
    except ImportError:
        smc = SMCBlock(
            "smc",
            use_c_backend=False,
            lambda_d=SMC_LAMBDA,
            K_sw_d=SMC_KSW,
            phi_d=SMC_PHI_D,
            lambda_q=SMC_LAMBDA,
            K_sw_q=SMC_KSW,
            phi_q=SMC_PHI_Q,
            out_min=-V_DC,
            out_max=V_DC
        )
        print("SMC: Python backend")

    # -------------------------------------------------------------------------
    # Coordinate transforms
    # -------------------------------------------------------------------------

    inv_park = InvParkFOC("inv_park", use_c_backend=False)
    inv_clarke = InvClarkeTransformBlock("inv_clarke", use_c_backend=False)

    # -------------------------------------------------------------------------
    # PMSM Plant (FMU or Stub fallback)
    # -------------------------------------------------------------------------

    try:
        motor = ThreePhaseMotorBlock(
            "motor",
            fmu_path=fmu_path,
            R=0.5, L_d=0.005, L_q=0.006,
            lambda_pm=0.175, J=0.002, B=0.001, p=2.0,
        )
        print("Motor FMU loaded")
    except Exception:
        print("FMU not found — using stub motor")
        motor = _StubMotor("motor")

    # -------------------------------------------------------------------------
    # Loop breaking (causal delay insertion)
    # -------------------------------------------------------------------------
    # Why?
    # FOC is inherently algebraic:
    #     motor → currents → SMC → voltages → motor
    #
    # EmbedSim requires acyclic graph.
    # We insert 1-step delays to make system causal.
    # -------------------------------------------------------------------------

    _motor_reg = {'output': np.zeros(7, dtype=np.float32)}

    from embedsim.simulation_engine import VectorDelay, LoopBreaker

    class _MotorCaptureSink(VectorEnd):
        """Captures motor output into shared register."""

        def __init__(self, name, reg):
            super().__init__(name)
            self._reg = reg

        def compute_py(self, t, dt, input_values=None):
            super().compute_py(t, dt, input_values)
            if input_values and input_values[0] is not None:
                v = input_values[0].value
                if len(v) == 7:
                    self._reg['output'] = v.astype(np.float32).copy()
            return self.output

    class _RegDelay(VectorBlock, LoopBreaker):
        """
        LoopBreaker:
        Outputs previous-step motor vector.
        """

        is_loop_breaker = True

        def __init__(self, name, reg):
            super().__init__(name)
            self._reg = reg
            self._held = np.zeros(7, dtype=np.float32)

        def get_loop_breaking_output(self):
            return VectorSignal(self._held.copy(), self.name)

        def compute_py(self, t, dt, input_values=None):
            out = self._held.copy()
            self._held = self._reg['output'].copy()
            self.output = VectorSignal(out, self.name)
            return self.output

    motor_sink = _MotorCaptureSink("motor_sink", _motor_reg)
    delay_omega = _RegDelay("delay_omega", _motor_reg)
    delay_idiq  = _RegDelay("delay_idiq", _motor_reg)
    delay_theta = _RegDelay("delay_theta", _motor_reg)

    # -------------------------------------------------------------------------
    # Signal routing (FOC signal graph)
    # -------------------------------------------------------------------------

    sink = VectorEnd("sink")

    # Speed loop
    omega_ref_src >> cg_start >> speed_pi
    delay_omega >> speed_pi

    # Current loop
    speed_pi >> smc
    delay_idiq >> smc

    # Coordinate transforms
    smc >> inv_park
    delay_theta >> inv_park
    inv_park >> inv_clarke >> cg_end >> sink

    # Plant
    smc >> motor
    load_torque_src >> motor
    motor >> motor_sink

    # -------------------------------------------------------------------------
    # Simulation engine
    # -------------------------------------------------------------------------

    sim = EmbedSim(
        sinks=[sink, motor_sink],
        T=T_SIM,
        dt=DT,
        solver=ODESolver.RK4,
    )

    # Expose CodeGen boundary markers so main() can call generate_loop()
    sim.cg_start = cg_start
    sim.cg_end   = cg_end

    # ── Topology visualisation ────────────────────────────────────────────
    # sim.topo is auto-attached by EmbedSim.__init__ via TopologyPrinter.
    # print_console() renders a clean multi-lane ASCII diagram.
    # show_gui()      opens an interactive SVG diagram in the browser.
    if sim.topo is not None:
        sim.topo.print_console()
    else:
        sim.print_topology_sources2sink()   # legacy fallback

    # Scope signals
    sim.scope.add(omega_ref_src, label="omega_ref")
    sim.scope.add(motor, label="motor")
    sim.scope.add(smc, label="smc")
    sim.scope.add(speed_pi, label="speed_pi")

    return sim


# =============================================================================
# Stub motor (simple physics fallback)
# =============================================================================

class _StubMotor(VectorBlock):
    """
    Simplified PMSM:

        T_em = 1.5 p λ iq
        J ω̇ = T_em − T_load − B ω

    Used when FMU is unavailable.
    """

    def __init__(self, name):
        super().__init__(name)
        self.vector_size = 7
        self.is_dynamic = True

        self.p = 2
        self.lam = 0.175
        self.J = 0.002
        self.B = 0.001
        self.R = 0.5
        self.L = 0.005

        self._omega = 0.0
        self._theta = 0.0
        self._iq = 0.0

    def compute_py(self, t, dt, input_values=None):

        v_q = float(input_values[0].value[1])
        T_load = float(input_values[1].value[0])

        omega_e = self.p * self._omega
        e_q = omega_e * self.lam

        # Electrical dynamic (Euler)
        self._iq += (v_q - self.R*self._iq - e_q)/self.L * dt

        T_em = 1.5*self.p*self.lam*self._iq

        # Mechanical dynamic
        self._omega += (T_em - T_load - self.B*self._omega)/self.J * dt
        self._theta = (self._theta + omega_e*dt) % (2*np.pi)

        rpm = self._omega * 60/(2*np.pi)

        self.output = VectorSignal(
            np.array([0.0, self._iq, self._omega,
                      self._theta, T_em, omega_e, rpm],
                     dtype=np.float32),
            self.name
        )
        return self.output


# =============================================================================
# Main execution
# =============================================================================

def main():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # ── Backend selection ─────────────────────────────────────────────────
    # Try interactive backends in order; fall back to Agg (non-interactive,
    # saves file only) rather than crashing on headless systems.
    for _backend in ('TkAgg', 'Qt5Agg', 'WXAgg', 'Agg'):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue

    from embedsim.plot_helper import create_plotter

    # ------------------------------
    # Build + run simulation
    # ------------------------------
    sim = build_sim()

    # Optional: open browser GUI topology diagram before running
    # Uncomment the line below to launch the interactive SVG viewer:
    # sim.topo.show_gui()

    print("\n⚙️  Running simulation...")
    sim.run()
    print("✅ Simulation complete.")

    # ── CodeGen: emit embedsim_loop.c / embedsim_loop.h ──────────────────
    # Generates the embedded C loop for all blocks inside the
    # CodeGenStart → CodeGenEnd region (speed_pi → smc → inv_park → inv_clarke).
    # Output goes to:  examples/pmsm/embedsim_gen/
    print("\n⚙️  Generating embedded C loop...")
    try:
        loop_files = sim.cg_end.generate_loop(
            sim.cg_start,
            output_dir=os.path.dirname(os.path.abspath(__file__)),
            dt_hz=1.0 / DT,          # 20 kHz  →  #define EMBEDSIM_DT 0.00005f
        )
        print("✅ CodeGen complete.")
        print(f"   embedsim_gen/embedsim_loop.c")
        print(f"   embedsim_gen/embedsim_loop.h")
    except Exception as e:
        print(f"⚠️  CodeGen skipped: {e}")

    # ------------------------------
    # Extract scope signals
    # ------------------------------
    t = np.array(sim.scope.t)
    i_d = np.array(sim.scope.data['motor[0]'])
    i_q = np.array(sim.scope.data['motor[1]'])
    theta = np.array(sim.scope.data['motor[3]'])

    # Reconstruct 3-phase currents (αβ → abc)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    i_alpha = i_d * cos_t - i_q * sin_t
    i_beta = i_d * sin_t + i_q * cos_t
    i_a = i_alpha
    i_b = -0.5 * i_alpha + (np.sqrt(3)/2) * i_beta
    i_c = -0.5 * i_alpha - (np.sqrt(3)/2) * i_beta

    # ------------------------------
    # Plot layout
    # ------------------------------
    fig = plt.figure(figsize=(14, 13))
    fig.suptitle("EmbedSim — PMSM FOC with Sliding Mode Controller",
                 fontsize=13, fontweight='bold')
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])   # Speed
    ax2 = fig.add_subplot(gs[1, 0])   # d-axis current
    ax3 = fig.add_subplot(gs[1, 1])   # q-axis current
    ax4 = fig.add_subplot(gs[2, 0])   # SMC voltages
    ax5 = fig.add_subplot(gs[2, 1])   # Torque
    ax6 = fig.add_subplot(gs[3, :])   # 3-phase currents

    # Helper functions
    def plot2(ax, t, d1, d2, label1, label2, title, ylabel, c1, c2, ls2='--'):
        ax.plot(t, d1, color=c1, lw=1.5, label=label1)
        ax.plot(t, d2, color=c2, lw=1.5, label=label2, linestyle=ls2)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    def plot1(ax, t, d, label, title, ylabel, color):
        ax.plot(t, d, color=color, lw=1.5, label=label)
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    # Plot signals
    plot2(ax1, t,
          np.array(sim.scope.data['omega_ref[0]']),
          np.array(sim.scope.data['motor[2]']),
          'ω_ref', 'ω_m',
          'Speed [rad/s]', 'Speed [rad/s]',
          'tab:blue', 'tab:orange')

    plot2(ax2, t,
          np.array(sim.scope.data['speed_pi[0]']),
          i_d,
          'id_ref', 'id_meas',
          'd-axis Current [A]', 'Current [A]',
          'tab:green', 'tab:red')

    plot2(ax3, t,
          np.array(sim.scope.data['speed_pi[1]']),
          i_q,
          'iq_ref', 'iq_meas',
          'q-axis Current [A]', 'Current [A]',
          'tab:purple', 'tab:brown')

    ax4.plot(t, np.array(sim.scope.data['smc[0]']), color='tab:cyan', lw=1.5, label='v_d')
    ax4.plot(t, np.array(sim.scope.data['smc[1]']), color='tab:red', lw=1.5, label='v_q')
    ax4.set_title('SMC Voltage Commands [V]', fontweight='bold')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Voltage [V]')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')

    plot1(ax5, t, np.array(sim.scope.data['motor[4]']),
          'T_em', 'Electromagnetic Torque [N·m]', 'Torque [N·m]', 'tab:olive')

    ax6.plot(t, i_a, color='tab:blue', lw=1.0, label='i_a', alpha=0.9)
    ax6.plot(t, i_b, color='tab:orange', lw=1.0, label='i_b', alpha=0.9)
    ax6.plot(t, i_c, color='tab:green', lw=1.0, label='i_c', alpha=0.9)
    ax6.set_title('Three-Phase Stator Currents [A]', fontweight='bold')
    ax6.set_xlabel('Time [s]')
    ax6.set_ylabel('Current [A]')
    ax6.legend(fontsize=9, loc='upper right')
    ax6.grid(True, alpha=0.3, linestyle='--')

    # Inset zoom last 50 ms
    axins = inset_axes(ax6, width="30%", height="60%", loc='center right',
                       bbox_to_anchor=(-0.01, 0, 1, 1), bbox_transform=ax6.transAxes)
    mask = t >= T_SIM - 0.05
    axins.plot(t[mask], i_a[mask], color='tab:blue', lw=1.2)
    axins.plot(t[mask], i_b[mask], color='tab:orange', lw=1.2)
    axins.plot(t[mask], i_c[mask], color='tab:green', lw=1.2)
    axins.set_title('Steady state (last 50 ms)', fontsize=8)
    axins.grid(True, alpha=0.3, linestyle='--')
    axins.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig("pmsm_foc_smc_results.png", dpi=120, bbox_inches="tight")
    print("\n💾 Saved plot: pmsm_foc_smc_results.png")

    # Export topology diagram as standalone HTML
    if sim.topo is not None:
        topo_path = sim.topo.export_html("pmsm_foc_smc_topology.html")
        print(f"💾 Saved topology: {topo_path}")

    # ------------------------------
    # Show plot
    # ------------------------------
    plt.show()
    print("✅ Plot displayed successfully.")


if __name__ == "__main__":
    main()