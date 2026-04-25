"""
dc_motor_pid_example.py
=======================

EmbedSim — DC Motor PID Speed Control
--------------------------------------

Demonstrates a complete closed-loop speed-control simulation of a
permanent-magnet DC motor modelled as a Modelica FMU, controlled by a
discrete PID algorithm implemented as an EmbedSim VectorBlock.

Signal diagram
--------------

    [◈ reference]  ──► ω_ref (rad/s) ──►┐
                                          ├──► [⊕ error] ──► e (rad/s) ──► [⚡ pid] ──► u (V) ──► [⚙ dc_motor] ──► ω (rad/s) ──► [■ output]
    [z⁻¹ feedback] ──► ω_fb (rad/s) ──►┘                                                              └──► ω (rad/s) ──► [z⁻¹ feedback]

Motor model (DCMotor.fmu)
-------------------------
    Electrical : R = 1 Ω,  L = 0.5 H,  k_emf = 0.01 V·s/rad
    Mechanical : J = 0.01 kg·m²,  B = 0.1 N·m·s/rad
    Input  : u  — armature voltage [V]
    Output : w  — angular velocity [rad/s]

Reference profile (default)
----------------------------
    t = 0 s  →  ω_ref =   0 rad/s  (hold)
    t = 1 s  →  ω_ref = 100 rad/s  (step up)
    t = 3 s  →  ω_ref =  50 rad/s  (step down)

Default PID gains
-----------------
    Kp = 0.8,  Ki = 3.0,  Kd = 0.05
    Saturation : ±24 V,  Anti-windup : back-calculation
    Derivative filter α = 0.1,  Integral limit = ±100

Outputs
-------
    dc_motor_pid_response.png     — 4-panel diagnostic plot + metrics table
    dc_motor_pid_topology.html    — interactive signal-flow diagram

Run
---
    python dc_motor_pid_example.py

Author  : Paul Abraham (EmbedSim project)
License : MIT
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path bootstrap ─────────────────────────────────────────────────────────────
# _path_utils locates the project root via the .project_root_marker sentinel
# and adds the embedsim package directory to sys.path.
from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.core_blocks       import VectorBlock, VectorSignal
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.processing_blocks import VectorSum
from embedsim.simulation_engine import EmbedSim, VectorDelay, ODESolver
from embedsim.fmu_blocks        import FMUBlock


# =============================================================================
#  Configuration dataclass
#  All tuneable parameters live here — no magic numbers elsewhere in the file.
# =============================================================================

@dataclass
class SimConfig:
    """
    Full simulation configuration for the DC-motor PID example.

    Attributes
    ----------
    t_stop : float
        Simulation end time [s].
    dt : float
        Fixed timestep [s].  Must be << electrical time constant L/R = 0.5 s.
    solver : str
        ODE integration method.  Use ODESolver.RK4 for smooth FMU coupling.

    Kp, Ki, Kd : float
        Proportional, integral and derivative gains.
    u_min, u_max : float
        Actuator saturation limits [V].
    derivative_filter : float
        First-order filter coefficient α ∈ (0, 1] on the derivative term.
        Smaller α → heavier smoothing.
    integral_limit : float
        Symmetric anti-windup clamp on the integral accumulator.

    reference_steps : list of (t, ω_ref) pairs
        Piecewise-constant reference profile.  Must include t=0.

    fmu_path : Path
        Absolute or relative path to DCMotor.fmu.
    fmu_voltage_input : str
        FMU input variable name for armature voltage.
    fmu_speed_output : str
        FMU output variable name for angular speed.

    plot_file : str
        File name for the saved diagnostic plot.
    """

    # Timing
    t_stop:  float = 5.0
    dt:      float = 0.001
    solver:  str   = ODESolver.RK4

    # PID gains
    Kp: float = 0.8
    Ki: float = 3.0
    Kd: float = 0.05

    # Actuator and controller limits
    u_min:             float = -24.0
    u_max:             float =  24.0
    derivative_filter: float =  0.1
    integral_limit:    float =  100.0

    # Reference speed profile  [(t [s], ω_ref [rad/s]), ...]
    reference_steps: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 100.0), (3.0, 50.0)]
    )

    # FMU interface
    fmu_path:          Path = Path(__file__).parent / "modelica" / "DCMotor.fmu"
    fmu_voltage_input: str  = "u"
    fmu_speed_output:  str  = "w"

    # Outputs
    plot_file:     str = "dc_motor_pid_response.png"
    topology_file: str = "dc_motor_pid_topology.html"


# =============================================================================
#  Blocks
#  Each block sets self.output_label so the topology printer can annotate wires.
# =============================================================================

class StepReference(VectorBlock):
    """
    Piecewise-constant angular speed reference generator.

    Parameters
    ----------
    steps : list of (t, ω) tuples
        At each time t the output switches to ω [rad/s].

    Output label : ω_ref (rad/s)
    """

    def __init__(self, name: str, steps: List[Tuple[float, float]]) -> None:
        super().__init__(name)
        self._steps      = sorted(steps, key=lambda s: s[0])
        self.output_label = "ω_ref (rad/s)"

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        value = self._steps[0][1]
        for step_t, step_v in self._steps:
            if t >= step_t:
                value = step_v
            else:
                break
        self.output = VectorSignal([value], self.name)
        return self.output


class PIDController(VectorBlock):
    """
    Discrete-time PID controller with:
        - First-order filtered derivative  (prevents noise amplification)
        - Conditional integration          (prevents windup while saturated)
        - Back-calculation anti-windup

    Input label  : e (rad/s)   — tracking error
    Output label : u (V)       — voltage command

    Parameters
    ----------
    Kp, Ki, Kd : float
        Controller gains.
    u_min, u_max : float
        Output saturation limits [V].
    alpha : float
        Derivative filter coefficient α ∈ (0,1].  1.0 = no filter.
    integral_limit : float
        Symmetric clamp on the raw integral accumulator.
    """

    def __init__(
        self,
        name: str,
        Kp: float,
        Ki: float,
        Kd: float,
        u_min: float            = -24.0,
        u_max: float            =  24.0,
        alpha: float            =  0.1,
        integral_limit: float   =  100.0,
    ) -> None:
        super().__init__(name)
        self.Kp             = Kp
        self.Ki             = Ki
        self.Kd             = Kd
        self.u_min          = u_min
        self.u_max          = u_max
        self.alpha          = alpha
        self.integral_limit = integral_limit

        self._integral:   float = 0.0
        self._prev_error: float = 0.0
        self._prev_deriv: float = 0.0

        # Diagnostic signals — readable by the scope after each step
        self.P = self.I = self.D = 0.0

        self.output_label = "u (V)"

    def reset(self) -> None:
        super().reset()
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_deriv  = 0.0
        self.P = self.I = self.D = 0.0

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        if not input_values:
            raise ValueError(f"{self.name}: requires a connected error signal.")

        error = float(input_values[0].value[0])

        # Proportional
        self.P = self.Kp * error

        # Integral with symmetric clamp (windup guard stage 1)
        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit,
             self.integral_limit,
        )
        self.I = self.Ki * self._integral

        # Derivative with first-order low-pass filter
        raw_d          = (error - self._prev_error) / dt if dt > 1e-12 else 0.0
        filtered_d     = self.alpha * raw_d + (1.0 - self.alpha) * self._prev_deriv
        self.D         = self.Kd * filtered_d

        u_raw = self.P + self.I + self.D
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))

        # Back-calculation anti-windup (stage 2) — only active when saturated
        if self.Ki > 1e-12:
            self._integral += (u_sat - u_raw) / self.Ki * dt

        self._prev_error = error
        self._prev_deriv = filtered_d

        self.output = VectorSignal([u_sat], self.name)
        return self.output


class DCMotorFMU(FMUBlock):
    """
    Thin wrapper around FMUBlock for the DCMotor.fmu plant.

    Input label  : u (V)       — armature voltage from PID
    Output label : ω (rad/s)   — shaft angular velocity

    The FMU implements the following continuous-time model:

        L · di/dt = u − R·i − k·ω
        J · dω/dt = k·i − B·ω

    with parameters:  R=1 Ω, L=0.5 H, k=0.01, J=0.01 kg·m², B=0.1 N·m·s/rad
    """

    def __init__(self, name: str, cfg: SimConfig) -> None:
        super().__init__(
            name         = name,
            fmu_path     = str(cfg.fmu_path),
            input_names  = [cfg.fmu_voltage_input],
            output_names = [cfg.fmu_speed_output],
        )
        self.output_label = "ω (rad/s)"
        self.last_voltage = 0.0
        self.last_speed   = 0.0

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        result = super().compute(t, dt, input_values)
        if input_values:
            self.last_voltage = float(input_values[0].value[0])
        self.last_speed = float(result.value[0])
        return result


# =============================================================================
#  Diagram builder
# =============================================================================

def build_diagram(cfg: SimConfig):
    """
    Instantiate and wire the closed-loop PID diagram.

    Topology
    --------
        reference  ──► ω_ref ──►┐
                                  ├──► error ──► e ──► pid ──► u ──► dc_motor ──► ω ──► output
        feedback   ──► ω_fb ──►┘                                         └──► ω ──► feedback (z⁻¹)

    Returns
    -------
    (reference, motor, pid, error_sum, feedback, sink)
    """

    # Reference generator
    reference             = StepReference("reference", cfg.reference_steps)

    # Error summing junction  (+ω_ref − ω_fb)
    error_sum             = VectorSum("error", signs=[1, -1])
    error_sum.output_label = "e (rad/s)"

    # PID controller
    pid = PIDController(
        "pid",
        cfg.Kp, cfg.Ki, cfg.Kd,
        u_min          = cfg.u_min,
        u_max          = cfg.u_max,
        alpha          = cfg.derivative_filter,
        integral_limit = cfg.integral_limit,
    )

    # DC motor FMU plant
    motor = DCMotorFMU("dc_motor", cfg)

    # One-step feedback delay (algebraic-loop breaker)
    feedback              = VectorDelay("feedback", initial=[0.0])
    feedback.output_label = "ω_fb (rad/s)"

    # Sink (marks the end of the forward signal path for EmbedSim)
    sink = VectorEnd("output")

    # ── Connect ───────────────────────────────────────────────────────────────
    reference >> error_sum      # ω_ref enters summing junction (+ port)
    error_sum >> pid            # tracking error enters PID
    pid       >> motor          # voltage command enters motor FMU
    motor     >> sink           # speed to logging sink
    motor     >> feedback       # speed enters one-step delay
    feedback  >> error_sum      # delayed speed fed back to summing junction (− port)

    return reference, motor, pid, error_sum, feedback, sink


# =============================================================================
#  Step-response analysis
# =============================================================================

@dataclass
class StepMetrics:
    """Performance metrics computed from the first speed step."""
    rise_time:         Optional[float] = None
    overshoot_pct:     float           = 0.0
    settling_time:     Optional[float] = None
    steady_state_err:  float           = 0.0
    peak_voltage:      float           = 0.0
    iae:               float           = 0.0   # Integral Absolute Error
    ise:               float           = 0.0   # Integral Squared Error


def analyse_step(
    t:         np.ndarray,
    response:  np.ndarray,
    reference: np.ndarray,
    voltage:   np.ndarray,
    dt:        float,
    tol:       float = 0.02,
) -> StepMetrics:
    """
    Compute standard step-response KPIs for the first non-zero step.

    Parameters
    ----------
    t, response, reference, voltage : ndarray
        Time vector, speed response, reference, and control voltage arrays.
    dt : float
        Timestep [s] — used only for IAE/ISE consistency.
    tol : float
        Settling band half-width as a fraction of the final value (default 2 %).
    """
    m = StepMetrics()

    # Locate the first step transition in the reference
    step_idx = next(
        (i for i in range(1, len(reference))
         if abs(reference[i] - reference[i - 1]) > 1e-6),
        0,
    )
    final_val = reference[step_idx]
    if abs(final_val) < 1e-9:
        return m                    # no meaningful step; skip

    resp = response[step_idx:]
    t_s  = t[step_idx:]

    # Rise time (10 % → 90 % of final value)
    try:
        i10 = np.where(resp >= 0.10 * final_val)[0][0]
        i90 = np.where(resp >= 0.90 * final_val)[0][0]
        m.rise_time = float(t_s[i90] - t_s[i10])
    except IndexError:
        pass

    # Overshoot
    m.overshoot_pct = float(
        max(0.0, (resp.max() - final_val) / abs(final_val) * 100.0)
    )

    # Settling time — first sample that stays within ±tol of final_val
    band        = tol * abs(final_val)
    in_band     = np.abs(resp - final_val) <= band
    settled_idx = np.where(in_band)[0]
    if len(settled_idx):
        # Confirm it never leaves the band afterwards
        last_exit = -1
        for k in range(len(settled_idx) - 1):
            if settled_idx[k + 1] != settled_idx[k] + 1:
                last_exit = settled_idx[k + 1]
        if last_exit == -1:
            m.settling_time = float(t_s[settled_idx[0]])

    # Steady-state error (mean of last 20 % of the step segment)
    tail = resp[int(0.80 * len(resp)):]
    m.steady_state_err = float(abs(final_val - tail.mean()))

    # Peak actuator effort
    m.peak_voltage = float(np.abs(voltage).max())

    # Integral error norms
    err   = np.abs(reference - response)
    m.iae = float(np.trapezoid(err,    t))
    m.ise = float(np.trapezoid(err**2, t))

    return m


# =============================================================================
#  Plotting
# =============================================================================

def plot_results(
    t:       np.ndarray,
    ref:     np.ndarray,
    speed:   np.ndarray,
    voltage: np.ndarray,
    error:   np.ndarray,
    metrics: StepMetrics,
    cfg:     SimConfig,
) -> None:
    """
    Produce a four-panel diagnostic figure:
        Panel 1 (top, full width)  — speed tracking + reference
        Panel 2                    — control voltage with ±u_max limits
        Panel 3 (left)             — tracking error over time
        Panel 4 (right)            — phase-plane (speed vs error, colour = time)
        Panel 5 (bottom, table)    — step-response KPI table
    """
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"DC Motor PID Speed Control   ·   "
        f"Kp={cfg.Kp}  Ki={cfg.Ki}  Kd={cfg.Kd}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Panel 1 : speed tracking ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(t, ref,   "--", color="#e74c3c", lw=1.5, label="Reference  ω_ref (rad/s)")
    ax1.plot(t, speed, "-",  color="#2980b9", lw=1.8, label="Motor speed ω   (rad/s)")
    ax1.fill_between(t, ref, speed, alpha=0.12, color="#2980b9")
    for st, _ in cfg.reference_steps[1:]:
        ax1.axvline(st, color="#95a5a6", lw=0.8, ls=":", label="_nolegend_")
    ax1.set_ylabel("Speed (rad/s)")
    ax1.set_title("Speed Tracking")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(t[0], t[-1])

    # ── Panel 2 : control voltage ─────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, voltage, color="#27ae60", lw=1.5, label="u (V)")
    ax2.axhline( cfg.u_max, color="#e74c3c", ls="--", lw=0.9, alpha=0.7,
                 label=f"±{cfg.u_max} V limit")
    ax2.axhline(-cfg.u_max, color="#e74c3c", ls="--", lw=0.9, alpha=0.7)
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("Control Voltage")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(t[0], t[-1])

    # ── Panel 3 : tracking error ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(t, error, color="#8e44ad", lw=1.3)
    ax3.axhline(0, color="black", ls="--", lw=0.7, alpha=0.4)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Error (rad/s)")
    ax3.set_title("Tracking Error  e(t)")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(t[0], t[-1])

    # ── Panel 4 : phase plane ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    sc  = ax4.scatter(speed, error, c=t, cmap="viridis", s=2, alpha=0.6)
    plt.colorbar(sc, ax=ax4, label="Time (s)")
    ax4.set_xlabel("Speed ω (rad/s)")
    ax4.set_ylabel("Error e (rad/s)")
    ax4.set_title("Phase Plane   ω vs e(t)")
    ax4.grid(True, alpha=0.3)

    # ── Panel 5 : KPI table ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")

    def _fmt(v, unit="", fmt=".3f"):
        return f"{v:{fmt}} {unit}".strip() if v is not None else "—"

    rows = [
        ["Rise time  (10 → 90 %)",  _fmt(metrics.rise_time,        "s")],
        ["Overshoot",                _fmt(metrics.overshoot_pct,    "%")],
        ["Settling time  (±2 %)",   _fmt(metrics.settling_time,    "s")],
        ["Steady-state error",       _fmt(metrics.steady_state_err, "rad/s")],
        ["Peak voltage",             _fmt(metrics.peak_voltage,     "V")],
        ["IAE",                      _fmt(metrics.iae)],
        ["ISE",                      _fmt(metrics.ise)],
    ]
    tbl = ax5.table(
        cellText  = rows,
        colLabels = ["Metric", "Value"],
        loc       = "center",
        cellLoc   = "left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for col in range(2):
        tbl[(0, col)].set_facecolor("#2c3e50")
        tbl[(0, col)].set_text_props(color="white", fontweight="bold")

    plt.savefig(cfg.plot_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved  →  '{cfg.plot_file}'")


# =============================================================================
#  Utility
# =============================================================================

def _extract(scope, label: str) -> np.ndarray:
    """Return the channel `label[0]` from the scope as a flat ndarray."""
    key = f"{label}[0]"
    if key not in scope.data:
        available = list(scope.data)
        raise KeyError(
            f"Channel '{key}' not found in scope.\n"
            f"  Available channels: {available}"
        )
    return np.asarray(scope.data[key])


# =============================================================================
#  Main entry point
# =============================================================================

def main(cfg: SimConfig | None = None) -> EmbedSim:
    """
    Build, simulate, analyse and plot the DC-motor PID example.

    Parameters
    ----------
    cfg : SimConfig, optional
        Override the default configuration for scripted parameter sweeps.
        If None the default SimConfig() is used.

    Returns
    -------
    EmbedSim
        The completed simulation object (scope data available for inspection).
    """
    cfg = cfg or SimConfig()

    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  EmbedSim — DC Motor PID Speed Control")
    print(SEP)
    print(f"  Duration    : {cfg.t_stop} s   dt = {cfg.dt} s   Solver: {cfg.solver}")
    print(f"  PID gains   : Kp={cfg.Kp}   Ki={cfg.Ki}   Kd={cfg.Kd}")
    print(f"  Saturation  : ±{cfg.u_max} V")
    print(f"  FMU         : {cfg.fmu_path}")
    print(SEP)

    # ── 1. Build the signal-flow diagram ──────────────────────────────────────
    reference, motor, pid, error_sum, feedback, sink = build_diagram(cfg)

    # ── 2. Construct the EmbedSim runner ──────────────────────────────────────
    sim = EmbedSim(sinks=[sink], T=cfg.t_stop, dt=cfg.dt, solver=cfg.solver)

    # ── 3. Print and export the signal topology ────────────────────────────────
    print("\n  Signal topology (output_label shown on each wire):\n")
    sim.topo.print_console()
    sim.topo.export_html(cfg.topology_file)
    print(f"  Topology    →  '{cfg.topology_file}'")

    # ── 4. Register scope channels ────────────────────────────────────────────
    sim.scope.add(reference, label="Reference")        # ω_ref  (rad/s)
    sim.scope.add(motor,     label="Motor Speed")      # ω      (rad/s)
    sim.scope.add(pid,       label="Control Voltage")  # u      (V)
    sim.scope.add(error_sum, label="Error")            # e      (rad/s)

    # ── 5. Run ────────────────────────────────────────────────────────────────
    print("\n  Running simulation …")
    t0 = time.perf_counter()
    sim.run(verbose=True, progress_bar=True)
    elapsed = time.perf_counter() - t0
    print(f"  Wall-clock  : {elapsed:.2f} s")

    # Graceful FMU teardown
    try:
        motor.terminate()
    except Exception:
        pass

    # ── 6. Extract results ────────────────────────────────────────────────────
    t_arr   = np.asarray(sim.scope.t)
    ref_arr = _extract(sim.scope, "Reference")
    spd_arr = _extract(sim.scope, "Motor Speed")
    vol_arr = _extract(sim.scope, "Control Voltage")
    err_arr = _extract(sim.scope, "Error")

    # ── 7. Compute step-response KPIs ─────────────────────────────────────────
    metrics = analyse_step(t_arr, spd_arr, ref_arr, vol_arr, cfg.dt)

    print("\n  ── Step-response KPIs " + "─" * 50)
    rt  = f"{metrics.rise_time:.3f} s"   if metrics.rise_time    else "—"
    st  = f"{metrics.settling_time:.3f} s" if metrics.settling_time else "—"
    print(f"  Rise time  (10 → 90 %) : {rt}")
    print(f"  Overshoot              : {metrics.overshoot_pct:.2f} %")
    print(f"  Settling time  (±2 %) : {st}")
    print(f"  Steady-state error     : {metrics.steady_state_err:.4f} rad/s")
    print(f"  Peak voltage           : {metrics.peak_voltage:.2f} V")
    print(f"  IAE                    : {metrics.iae:.4f}")
    print(f"  ISE                    : {metrics.ise:.4f}")

    # ── 8. Plot ───────────────────────────────────────────────────────────────
    plot_results(t_arr, ref_arr, spd_arr, vol_arr, err_arr, metrics, cfg)

    return sim


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = main()
    print("\n  Scope channels available:", list(sim.scope.data.keys()))
