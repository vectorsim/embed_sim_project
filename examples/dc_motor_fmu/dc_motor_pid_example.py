"""
dc_motor_pid_example.py
=======================

VectorSim Framework — DC Motor PID Control with Step-Response Analysis

Demonstrates closed-loop PID speed control of a DC motor FMU with:
  - Anti-windup + derivative filter PID
  - Multi-step reference profile
  - Full performance metrics (rise time, overshoot, settling, SSE)
  - Four-panel result plot saved to PNG

Usage:
    python dc_motor_pid_example.py

Author: Paul Abraham / ControlForge
Date  : 2026-02-24
Version: 3.0
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – safe everywhere
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Make the local embedsim package importable when running from any directory.
# Adjust this path to match where your embedsim folder lives.
# ---------------------------------------------------------------------------
from pathlib import Path

# Lambda to add N-levels parent to sys.path with print
add_parent_to_syspath = lambda levels=2: (
    print(f"[EmbedSim] Adding to sys.path: {(p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]}"),
    sys.path.insert(0, str((p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]))
)[-1]
add_parent_to_syspath(2)

from embedsim.core_blocks import VectorBlock, VectorSignal
from embedsim.dynamic_blocks import VectorEnd
from embedsim.processing_blocks import VectorSum
from embedsim.simulation_engine import EmbedSim, VectorDelay, ODESolver
from embedsim.fmu_blocks import FMUBlock


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  (change these without touching the rest of the code)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """All tunable simulation parameters in one place."""

    # Timing
    t_start: float = 0.0
    t_stop: float  = 5.0
    dt: float      = 0.001
    solver: str    = ODESolver.RK4

    # PID gains
    Kp: float = 0.8
    Ki: float = 3.0
    Kd: float = 0.05

    # Controller limits
    u_min: float = -24.0
    u_max: float  =  24.0
    derivative_filter: float = 0.1
    integral_limit: float    = 100.0

    # Reference speed profile  [(t_change, speed_rad_s), ...]
    reference_steps: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 100.0), (3.0, 50.0)]
    )

    # FMU
    fmu_path: Path = Path(__file__).parent / "modelica" / "DCMotor.fmu"
    fmu_voltage_input: str = "u"
    fmu_speed_output:  str = "w"

    # Output
    plot_file: str = "dc_motor_pid_response.png"


# ─────────────────────────────────────────────────────────────────────────────
#  Blocks
# ─────────────────────────────────────────────────────────────────────────────

class DCMotorFMU(FMUBlock):
    """FMUBlock specialised for a DC motor speed-control FMU."""

    def __init__(self, name: str, cfg: SimConfig) -> None:
        super().__init__(
            name=name,
            fmu_path=str(cfg.fmu_path),
            input_names=[cfg.fmu_voltage_input],
            output_names=[cfg.fmu_speed_output],
        )
        self._voltage_name = cfg.fmu_voltage_input
        self._speed_name   = cfg.fmu_speed_output
        # Logged for post-simulation diagnostics
        self.last_voltage: float = 0.0
        self.last_speed:   float = 0.0

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        result = super().compute(t, dt, input_values)
        if input_values:
            self.last_voltage = float(input_values[0].value[0])
        self.last_speed = float(result.value[0])
        return result


class PIDController(VectorBlock):
    """
    Discrete PID with:
      - Integral anti-windup (back-calculation)
      - Low-pass derivative filter
      - Symmetric output saturation
    """

    def __init__(
        self,
        name: str,
        Kp: float,
        Ki: float,
        Kd: float,
        u_min: float = -24.0,
        u_max: float =  24.0,
        alpha: float = 0.1,
        integral_limit: float = 100.0,
    ) -> None:
        super().__init__(name)
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_min, self.u_max    = u_min, u_max
        self.alpha                = alpha           # derivative filter coefficient
        self.integral_limit       = integral_limit

        self._integral:   float = 0.0
        self._prev_error: float = 0.0
        self._prev_deriv: float = 0.0

        # Diagnostics
        self.P: float = 0.0
        self.I: float = 0.0
        self.D: float = 0.0

    def reset(self) -> None:
        super().reset()
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_deriv = 0.0
        self.P = self.I = self.D = 0.0

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        if not input_values:
            raise ValueError(f"{self.name}: no input")

        error = float(input_values[0].value[0])

        # P
        self.P = self.Kp * error

        # I  (pre-clamp)
        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit, self.integral_limit
        )
        self.I = self.Ki * self._integral

        # D  (filtered)
        raw_deriv      = (error - self._prev_error) / dt if dt > 0 else 0.0
        filt_deriv     = self.alpha * raw_deriv + (1.0 - self.alpha) * self._prev_deriv
        self.D         = self.Kd * filt_deriv

        u_raw = self.P + self.I + self.D
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))

        # Anti-windup back-calculation
        if self.Ki != 0.0:
            self._integral += (u_sat - u_raw) / self.Ki * dt

        self._prev_error = error
        self._prev_deriv = filt_deriv

        self.output = VectorSignal([u_sat], self.name)
        return self.output


class StepReference(VectorBlock):
    """Piecewise-constant reference generator from a list of (t, value) breakpoints."""

    def __init__(self, name: str, steps: List[Tuple[float, float]]) -> None:
        super().__init__(name)
        self._steps = sorted(steps, key=lambda x: x[0])

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        value = self._steps[0][1]
        for step_t, step_v in self._steps:
            if t >= step_t:
                value = step_v
            else:
                break
        self.output = VectorSignal([value], self.name)
        return self.output


# ─────────────────────────────────────────────────────────────────────────────
#  Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepMetrics:
    rise_time:        Optional[float] = None
    overshoot_pct:    float           = 0.0
    settling_time:    Optional[float] = None
    steady_state_err: float           = 0.0
    peak_voltage:     float           = 0.0
    iae:              float           = 0.0   # Integral Absolute Error
    ise:              float           = 0.0   # Integral Squared Error


def analyse_step(
    time: np.ndarray,
    response: np.ndarray,
    reference: np.ndarray,
    voltage: np.ndarray,
    dt: float,
    tol: float = 0.02,
) -> StepMetrics:
    """Compute standard step-response metrics over the full trace."""
    m = StepMetrics()

    # Find the first step-change in reference
    step_idx = 0
    for i in range(1, len(reference)):
        if abs(reference[i] - reference[i - 1]) > 1e-6:
            step_idx = i
            break

    final_val = reference[step_idx]
    if abs(final_val) < 1e-9:
        return m       # avoid division by zero for zero-target step

    resp = response[step_idx:]
    t_s  = time[step_idx:]

    # Rise time (10 % → 90 %)
    try:
        i10 = np.where(resp >= 0.10 * final_val)[0][0]
        i90 = np.where(resp >= 0.90 * final_val)[0][0]
        m.rise_time = float(t_s[i90] - t_s[i10])
    except IndexError:
        pass

    # Overshoot
    m.overshoot_pct = float(max(0.0, (resp.max() - final_val) / abs(final_val) * 100.0))

    # Settling time
    settled = np.where(np.abs(resp - final_val) <= tol * abs(final_val))[0]
    if len(settled):
        # Confirm it stays settled
        for k in range(len(settled) - 1):
            if settled[k + 1] != settled[k] + 1:
                break
        else:
            m.settling_time = float(t_s[settled[0]])

    # Steady-state error (last 20 % of the step segment)
    tail = resp[int(0.8 * len(resp)):]
    m.steady_state_err = float(abs(final_val - tail.mean()))

    # Controller effort
    m.peak_voltage = float(np.abs(voltage).max())

    # Error integrals over the full trace
    error = np.abs(reference - response)
    m.iae = float(np.trapezoid(error, time))
    m.ise = float(np.trapezoid(error**2, time))

    return m


def _scope_signal(scope, label: str) -> np.ndarray:
    key = f"{label}[0]"
    if key not in scope.data:
        raise KeyError(
            f"Signal '{label}' not in scope.\n"
            f"Available: {list(scope.data.keys())}"
        )
    return np.asarray(scope.data[key])


# ─────────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    time:    np.ndarray,
    ref:     np.ndarray,
    speed:   np.ndarray,
    voltage: np.ndarray,
    error:   np.ndarray,
    metrics: StepMetrics,
    cfg:     SimConfig,
) -> None:
    """Four-panel publication-quality plot."""

    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    gains_str = f"Kp={cfg.Kp}  Ki={cfg.Ki}  Kd={cfg.Kd}"
    fig.suptitle(
        f"DC Motor PID Speed Control   ·   {gains_str}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Speed tracking (wide) ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, ref,   "--", color="#e74c3c", lw=1.5, label="Reference  (rad/s)")
    ax1.plot(time, speed, "-",  color="#2980b9", lw=1.8, label="Motor speed (rad/s)")
    ax1.fill_between(time, ref, speed, alpha=0.12, color="#2980b9")

    # Annotate step changes
    for st, sv in cfg.reference_steps[1:]:
        ax1.axvline(st, color="#95a5a6", lw=0.8, ls=":")

    ax1.set_ylabel("Speed (rad/s)")
    ax1.set_title("Speed Tracking")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(time[0], time[-1])

    # ── Panel 2: Control voltage ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time, voltage, color="#27ae60", lw=1.5)
    ax2.axhline( cfg.u_max, color="#e74c3c", ls="--", lw=0.9, alpha=0.7, label=f"±{cfg.u_max} V limit")
    ax2.axhline(-cfg.u_max, color="#e74c3c", ls="--", lw=0.9, alpha=0.7)
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("Control Voltage")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time[0], time[-1])

    # ── Panel 3: Tracking error ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, error, color="#8e44ad", lw=1.3)
    ax3.axhline(0, color="black", ls="--", lw=0.7, alpha=0.4)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Error (rad/s)")
    ax3.set_title("Tracking Error")
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(time[0], time[-1])

    # ── Panel 4: Phase-plane portrait ────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    sc = ax4.scatter(speed, error, c=time, cmap="viridis", s=2, alpha=0.6)
    plt.colorbar(sc, ax=ax4, label="Time (s)")
    ax4.set_xlabel("Speed (rad/s)")
    ax4.set_ylabel("Error (rad/s)")
    ax4.set_title("Phase Plane")
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: Metrics table ────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")

    def fmt(v, unit="", fmt=".3f") -> str:
        return f"{v:{fmt}} {unit}".strip() if v is not None else "—"

    rows = [
        ["Rise time (10→90 %)",  fmt(metrics.rise_time, "s")],
        ["Overshoot",            fmt(metrics.overshoot_pct, "%")],
        ["Settling time (±2 %)", fmt(metrics.settling_time, "s")],
        ["Steady-state error",   fmt(metrics.steady_state_err, "rad/s")],
        ["Peak voltage",         fmt(metrics.peak_voltage, "V")],
        ["IAE",                  fmt(metrics.iae)],
        ["ISE",                  fmt(metrics.ise)],
    ]
    col_labels = ["Metric", "Value"]
    tbl = ax5.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    plt.savefig(cfg.plot_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → '{cfg.plot_file}'")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def build_diagram(cfg: SimConfig):
    """Instantiate blocks and wire the closed-loop diagram."""

    reference      = StepReference("reference", cfg.reference_steps)
    motor          = DCMotorFMU("dc_motor", cfg)
    pid            = PIDController(
                        "pid",
                        cfg.Kp, cfg.Ki, cfg.Kd,
                        u_min=cfg.u_min, u_max=cfg.u_max,
                        alpha=cfg.derivative_filter,
                        integral_limit=cfg.integral_limit,
                    )
    feedback_delay = VectorDelay("feedback_delay", initial=[0.0])
    error_sum      = VectorSum("error", signs=[1, -1])
    sink           = VectorEnd("output")

    # Signal flow:
    #   reference ──►┐
    #                error ──► pid ──► motor ──► sink
    #   delay ───────┘             └──► delay
    reference      >> error_sum
    error_sum      >> pid
    pid            >> motor
    motor          >> sink
    motor          >> feedback_delay
    feedback_delay >> error_sum

    return reference, motor, pid, error_sum, feedback_delay, sink


def main(cfg: Optional[SimConfig] = None) -> EmbedSim:
    cfg = cfg or SimConfig()

    sep = "=" * 70
    print(sep)
    print("  DC Motor PID Control Simulation  —  ControlForge / VectorSim")
    print(sep)
    print(f"  Duration  : {cfg.t_stop} s   |  dt = {cfg.dt} s   |  Solver: {cfg.solver}")
    print(f"  PID       : Kp={cfg.Kp}  Ki={cfg.Ki}  Kd={cfg.Kd}")
    print(f"  FMU       : {cfg.fmu_path}")
    print(sep + "\n")

    # Build block diagram
    reference, motor, pid, error_sum, feedback_delay, sink = build_diagram(cfg)

    # Create simulation
    sim = EmbedSim(sinks=[sink], T=cfg.t_stop, dt=cfg.dt, solver=cfg.solver)

    # Print topology (both views)
    sim.print_topology()
    sim.print_topology_sources2sink()

    # Register scope signals
    sim.scope.add(reference,  label="Reference")
    sim.scope.add(motor,      label="Motor Speed")
    sim.scope.add(pid,        label="Control Voltage")
    sim.scope.add(error_sum,  label="Error")

    # Run
    t0 = time.perf_counter()
    sim.run(verbose=True, progress_bar=True)
    elapsed = time.perf_counter() - t0
    print(f"\n  Wall-clock time: {elapsed:.2f} s")

    # Clean up FMU
    try:
        motor.terminate()
    except Exception:
        pass

    # Extract signals
    t_arr   = np.asarray(sim.scope.t)
    ref_arr = _scope_signal(sim.scope, "Reference")
    spd_arr = _scope_signal(sim.scope, "Motor Speed")
    vol_arr = _scope_signal(sim.scope, "Control Voltage")
    err_arr = _scope_signal(sim.scope, "Error")

    # Performance analysis
    metrics = analyse_step(t_arr, spd_arr, ref_arr, vol_arr, cfg.dt)

    print("\n  ── Step-Response Metrics ─────────────────────────────────────")
    print(f"  Rise time (10→90 %)   : "
          f"{metrics.rise_time:.3f} s" if metrics.rise_time else "  Rise time: —")
    print(f"  Overshoot             : {metrics.overshoot_pct:.2f} %")
    print(f"  Settling time (±2 %)  : "
          f"{metrics.settling_time:.3f} s" if metrics.settling_time else "  Settling: —")
    print(f"  Steady-state error    : {metrics.steady_state_err:.4f} rad/s")
    print(f"  Peak voltage          : {metrics.peak_voltage:.2f} V")
    print(f"  IAE                   : {metrics.iae:.4f}")
    print(f"  ISE                   : {metrics.ise:.4f}")
    print()

    # Plot
    plot_results(t_arr, ref_arr, spd_arr, vol_arr, err_arr, metrics, cfg)

    return sim


if __name__ == "__main__":
    sim = main()
    print("\n  Simulation data in sim.scope")
    print("  Keys:", list(sim.scope.data.keys())[:6])