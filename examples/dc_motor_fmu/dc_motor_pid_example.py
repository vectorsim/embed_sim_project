"""
dc_motor_pid_example.py
=======================

EmbedSim — DC Motor PID Speed Control
Rewritten to showcase signal labelling via output_label so the topology
printer shows exactly what flows on every wire:

    [◈ reference]  ──► ω_ref (rad/s) ──►┐
                                         ├──► [⊕ error]  ──► e (rad/s)  ──► [⚡ pid] ──► u (V) ──► [⚙ dc_motor] ──► ω (rad/s) ──► [■ output]
    [z⁻¹ feedback] ──► ω_fb (rad/s) ──►┘                                                                 └──► ω (rad/s) ──► [z⁻¹ feedback]

Run:
    python dc_motor_pid_example.py

Author: EmbedSim / Paul Abraham
"""

from __future__ import annotations

import os, sys, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── path bootstrap ────────────────────────────────────────────────────────────
add_parent = lambda lvl=2: sys.path.insert(
    0, str(Path(__file__).resolve().parents[lvl - 1])
)
add_parent(2)

from embedsim.core_blocks       import VectorBlock, VectorSignal
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.processing_blocks import VectorSum
from embedsim.simulation_engine import EmbedSim, VectorDelay, ODESolver
from embedsim.fmu_blocks        import FMUBlock


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimConfig:
    # Timing
    t_stop:  float = 5.0
    dt:      float = 0.001
    solver:  str   = ODESolver.RK4

    # PID gains
    Kp: float = 0.8
    Ki: float = 3.0
    Kd: float = 0.05

    # Limits
    u_min:            float = -24.0
    u_max:            float =  24.0
    derivative_filter: float = 0.1
    integral_limit:   float = 100.0

    # Reference profile  [(t, ω_ref_rad_s), ...]
    reference_steps: List[Tuple[float, float]] = field(
        default_factory=lambda: [(0.0, 0.0), (1.0, 100.0), (3.0, 50.0)]
    )

    # FMU
    fmu_path:          Path = Path(__file__).parent / "modelica" / "DCMotor.fmu"
    fmu_voltage_input: str  = "u"
    fmu_speed_output:  str  = "w"

    # Output
    plot_file: str = "dc_motor_pid_response.png"


# =============================================================================
# Blocks  —  every block carries an output_label describing the signal it emits
# =============================================================================

class StepReference(VectorBlock):
    """
    Piecewise-constant speed reference.

    Output label: ω_ref (rad/s)
    """

    def __init__(self, name: str, steps: List[Tuple[float, float]]) -> None:
        super().__init__(name)
        self._steps = sorted(steps, key=lambda x: x[0])
        # ── signal label shown in topology diagram ────────────────────────
        self.output_label = "ω_ref (rad/s)"

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        value = self._steps[0][1]
        for st, sv in self._steps:
            if t >= st:
                value = sv
            else:
                break
        self.output = VectorSignal([value], self.name)
        return self.output


class PIDController(VectorBlock):
    """
    Discrete PID with anti-windup + derivative filter.

    Input  label: e (rad/s)   ← tracking error from VectorSum
    Output label: u (V)       → voltage command to motor
    """

    def __init__(
        self,
        name: str,
        Kp: float, Ki: float, Kd: float,
        u_min: float = -24.0, u_max: float = 24.0,
        alpha: float = 0.1,
        integral_limit: float = 100.0,
    ) -> None:
        super().__init__(name)
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.u_min, self.u_max    = u_min, u_max
        self.alpha                = alpha
        self.integral_limit       = integral_limit
        self._integral:   float = 0.0
        self._prev_error: float = 0.0
        self._prev_deriv: float = 0.0
        self.P = self.I = self.D = 0.0
        # ── signal labels ─────────────────────────────────────────────────
        self.output_label = "u (V)"

    def reset(self) -> None:
        super().reset()
        self._integral = self._prev_error = self._prev_deriv = 0.0
        self.P = self.I = self.D = 0.0

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        if not input_values:
            raise ValueError(f"{self.name}: no input")
        error = float(input_values[0].value[0])

        self.P = self.Kp * error
        self._integral = np.clip(
            self._integral + error * dt,
            -self.integral_limit, self.integral_limit,
        )
        self.I = self.Ki * self._integral

        raw_d      = (error - self._prev_error) / dt if dt > 0 else 0.0
        filt_d     = self.alpha * raw_d + (1.0 - self.alpha) * self._prev_deriv
        self.D     = self.Kd * filt_d

        u_raw = self.P + self.I + self.D
        u_sat = float(np.clip(u_raw, self.u_min, self.u_max))

        if self.Ki:
            self._integral += (u_sat - u_raw) / self.Ki * dt

        self._prev_error = error
        self._prev_deriv = filt_d
        self.output = VectorSignal([u_sat], self.name)
        return self.output


class DCMotorFMU(FMUBlock):
    """
    DC motor FMU wrapper.

    Input  label: u (V)       ← voltage command from PID
    Output label: ω (rad/s)   → motor shaft speed
    """

    def __init__(self, name: str, cfg: SimConfig) -> None:
        super().__init__(
            name=name,
            fmu_path=str(cfg.fmu_path),
            input_names=[cfg.fmu_voltage_input],
            output_names=[cfg.fmu_speed_output],
        )
        # ── signal labels ─────────────────────────────────────────────────
        self.output_label = "ω (rad/s)"
        self.last_voltage = self.last_speed = 0.0

    def compute(self, t: float, dt: float, input_values=None) -> VectorSignal:
        result = super().compute(t, dt, input_values)
        if input_values:
            self.last_voltage = float(input_values[0].value[0])
        self.last_speed = float(result.value[0])
        return result


# =============================================================================
# Wiring helper  —  annotate intermediate blocks
# =============================================================================

def build_diagram(cfg: SimConfig):
    """
    Wire the closed-loop DC motor PID diagram.

    Signal flow with labels
    -----------------------
    reference ──► ω_ref (rad/s) ──►┐
                                    ├──► error ──► e (rad/s) ──► pid ──► u (V) ──► dc_motor ──► ω (rad/s) ──► output
    feedback  ──► ω_fb (rad/s) ──►┘                                                    └──► ω (rad/s) ──► feedback (z⁻¹)
    """

    # ── Sources ───────────────────────────────────────────────────────────────
    reference = StepReference("reference", cfg.reference_steps)
    # output_label already set in __init__

    # ── Error junction ────────────────────────────────────────────────────────
    error_sum = VectorSum("error", signs=[1, -1])
    error_sum.output_label = "e (rad/s)"          # what leaves the subtractor

    # ── Controller ────────────────────────────────────────────────────────────
    pid = PIDController(
        "pid", cfg.Kp, cfg.Ki, cfg.Kd,
        u_min=cfg.u_min, u_max=cfg.u_max,
        alpha=cfg.derivative_filter,
        integral_limit=cfg.integral_limit,
    )
    # output_label = "u (V)"  already set in __init__

    # ── Plant ─────────────────────────────────────────────────────────────────
    motor = DCMotorFMU("dc_motor", cfg)
    # output_label = "ω (rad/s)"  already set in __init__

    # ── Feedback delay (LoopBreaker) ──────────────────────────────────────────
    feedback = VectorDelay("feedback", initial=[0.0])
    feedback.output_label = "ω_fb (rad/s)"        # fed back to error_sum

    # ── Sink ──────────────────────────────────────────────────────────────────
    sink = VectorEnd("output")

    # ── Connect ───────────────────────────────────────────────────────────────
    #   reference ──► ω_ref ──►┐
    #                           ├──► error ──► e ──► pid ──► u ──► motor ──► ω ──► output
    #   feedback  ──► ω_fb ──►┘                                    └──► ω ──► feedback
    reference >> error_sum
    error_sum >> pid
    pid       >> motor
    motor     >> sink
    motor     >> feedback
    feedback  >> error_sum

    return reference, motor, pid, error_sum, feedback, sink


# =============================================================================
# Analysis
# =============================================================================

@dataclass
class StepMetrics:
    rise_time:        Optional[float] = None
    overshoot_pct:    float           = 0.0
    settling_time:    Optional[float] = None
    steady_state_err: float           = 0.0
    peak_voltage:     float           = 0.0
    iae:              float           = 0.0
    ise:              float           = 0.0


def analyse_step(
    time:      np.ndarray,
    response:  np.ndarray,
    reference: np.ndarray,
    voltage:   np.ndarray,
    dt:        float,
    tol:       float = 0.02,
) -> StepMetrics:
    m = StepMetrics()
    step_idx = next(
        (i for i in range(1, len(reference))
         if abs(reference[i] - reference[i - 1]) > 1e-6),
        0,
    )
    final_val = reference[step_idx]
    if abs(final_val) < 1e-9:
        return m

    resp = response[step_idx:]
    t_s  = time[step_idx:]

    try:
        i10 = np.where(resp >= 0.10 * final_val)[0][0]
        i90 = np.where(resp >= 0.90 * final_val)[0][0]
        m.rise_time = float(t_s[i90] - t_s[i10])
    except IndexError:
        pass

    m.overshoot_pct = float(max(0.0, (resp.max() - final_val) / abs(final_val) * 100.0))

    settled = np.where(np.abs(resp - final_val) <= tol * abs(final_val))[0]
    if len(settled):
        for k in range(len(settled) - 1):
            if settled[k + 1] != settled[k] + 1:
                break
        else:
            m.settling_time = float(t_s[settled[0]])

    tail = resp[int(0.8 * len(resp)):]
    m.steady_state_err = float(abs(final_val - tail.mean()))
    m.peak_voltage     = float(np.abs(voltage).max())
    err = np.abs(reference - response)
    m.iae = float(np.trapezoid(err,    time))
    m.ise = float(np.trapezoid(err**2, time))
    return m


def _sig(scope, label: str) -> np.ndarray:
    key = f"{label}[0]"
    if key not in scope.data:
        raise KeyError(f"'{label}' not in scope. Available: {list(scope.data)}")
    return np.asarray(scope.data[key])


# =============================================================================
# Plotting
# =============================================================================

def plot_results(time, ref, speed, voltage, error, metrics, cfg):
    fig = plt.figure(figsize=(13, 10))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    fig.suptitle(
        f"DC Motor PID Speed Control   ·   Kp={cfg.Kp}  Ki={cfg.Ki}  Kd={cfg.Kd}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, ref,   "--", color="#e74c3c", lw=1.5, label="Reference  (rad/s)")
    ax1.plot(time, speed, "-",  color="#2980b9", lw=1.8, label="Motor speed (rad/s)")
    ax1.fill_between(time, ref, speed, alpha=0.12, color="#2980b9")
    for st, _ in cfg.reference_steps[1:]:
        ax1.axvline(st, color="#95a5a6", lw=0.8, ls=":")
    ax1.set_ylabel("Speed (rad/s)"); ax1.set_title("Speed Tracking")
    ax1.legend(loc="upper right", fontsize=9); ax1.grid(True, alpha=0.3)
    ax1.set_xlim(time[0], time[-1])

    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(time, voltage, color="#27ae60", lw=1.5)
    ax2.axhline( cfg.u_max, color="#e74c3c", ls="--", lw=0.9, alpha=0.7,
                 label=f"±{cfg.u_max} V limit")
    ax2.axhline(-cfg.u_max, color="#e74c3c", ls="--", lw=0.9, alpha=0.7)
    ax2.set_ylabel("Voltage (V)"); ax2.set_title("Control Voltage")
    ax2.legend(loc="upper right", fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time[0], time[-1])

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(time, error, color="#8e44ad", lw=1.3)
    ax3.axhline(0, color="black", ls="--", lw=0.7, alpha=0.4)
    ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Error (rad/s)")
    ax3.set_title("Tracking Error"); ax3.grid(True, alpha=0.3)
    ax3.set_xlim(time[0], time[-1])

    ax4 = fig.add_subplot(gs[2, 1])
    sc = ax4.scatter(speed, error, c=time, cmap="viridis", s=2, alpha=0.6)
    plt.colorbar(sc, ax=ax4, label="Time (s)")
    ax4.set_xlabel("Speed (rad/s)"); ax4.set_ylabel("Error (rad/s)")
    ax4.set_title("Phase Plane"); ax4.grid(True, alpha=0.3)

    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis("off")
    fmt = lambda v, u="", f=".3f": f"{v:{f}} {u}".strip() if v is not None else "—"
    rows = [
        ["Rise time (10→90 %)",  fmt(metrics.rise_time, "s")],
        ["Overshoot",            fmt(metrics.overshoot_pct, "%")],
        ["Settling time (±2 %)", fmt(metrics.settling_time, "s")],
        ["Steady-state error",   fmt(metrics.steady_state_err, "rad/s")],
        ["Peak voltage",         fmt(metrics.peak_voltage, "V")],
        ["IAE",                  fmt(metrics.iae)],
        ["ISE",                  fmt(metrics.ise)],
    ]
    tbl = ax5.table(cellText=rows, colLabels=["Metric", "Value"],
                    loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 1.6)
    for j in range(2):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    plt.savefig(cfg.plot_file, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → '{cfg.plot_file}'")


# =============================================================================
# Main
# =============================================================================

def main(cfg: SimConfig | None = None) -> EmbedSim:
    cfg = cfg or SimConfig()

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("  EmbedSim — DC Motor PID Speed Control")
    print(SEP)
    print(f"  Duration : {cfg.t_stop} s  |  dt = {cfg.dt} s  |  Solver: {cfg.solver}")
    print(f"  PID      : Kp={cfg.Kp}  Ki={cfg.Ki}  Kd={cfg.Kd}")
    print(f"  FMU      : {cfg.fmu_path}")
    print(SEP)

    # ── Build diagram ─────────────────────────────────────────────────────────
    reference, motor, pid, error_sum, feedback, sink = build_diagram(cfg)

    # ── Simulation object ─────────────────────────────────────────────────────
    sim = EmbedSim(sinks=[sink], T=cfg.t_stop, dt=cfg.dt, solver=cfg.solver)

    # ── Topology  ─────────────────────────────────────────────────────────────
    # sim.topo is auto-attached by EmbedSim.__init__
    print("\n📊 Signal topology (output_label shown on each wire):\n")
    sim.topo.print_console()
    sim.topo.export_html("dc_motor_pid_topology.html")

    # Optional: open interactive browser diagram
    # sim.topo.show_gui()

    # ── Scope  ────────────────────────────────────────────────────────────────
    # Labels match the output_label of each block so the legend is self-documenting
    sim.scope.add(reference, label="Reference")         # ω_ref (rad/s)
    sim.scope.add(motor,     label="Motor Speed")       # ω     (rad/s)
    sim.scope.add(pid,       label="Control Voltage")   # u     (V)
    sim.scope.add(error_sum, label="Error")             # e     (rad/s)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\n⚙️  Running simulation …")
    t0 = time.perf_counter()
    sim.run(verbose=True, progress_bar=True)
    print(f"  Wall-clock: {time.perf_counter() - t0:.2f} s")

    try:
        motor.terminate()
    except Exception:
        pass

    # ── Extract & analyse ─────────────────────────────────────────────────────
    t_arr   = np.asarray(sim.scope.t)
    ref_arr = _sig(sim.scope, "Reference")
    spd_arr = _sig(sim.scope, "Motor Speed")
    vol_arr = _sig(sim.scope, "Control Voltage")
    err_arr = _sig(sim.scope, "Error")

    metrics = analyse_step(t_arr, spd_arr, ref_arr, vol_arr, cfg.dt)

    print("\n  ── Step-Response Metrics " + "─" * 46)
    print(f"  Rise time (10→90 %)  : "
          f"{metrics.rise_time:.3f} s" if metrics.rise_time else "  Rise time: —")
    print(f"  Overshoot            : {metrics.overshoot_pct:.2f} %")
    print(f"  Settling time (±2 %) : "
          f"{metrics.settling_time:.3f} s" if metrics.settling_time else "  Settling: —")
    print(f"  Steady-state error   : {metrics.steady_state_err:.4f} rad/s")
    print(f"  Peak voltage         : {metrics.peak_voltage:.2f} V")
    print(f"  IAE                  : {metrics.iae:.4f}")
    print(f"  ISE                  : {metrics.ise:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_results(t_arr, ref_arr, spd_arr, vol_arr, err_arr, metrics, cfg)

    return sim


if __name__ == "__main__":
    sim = main()
    print("\n  Scope keys:", list(sim.scope.data.keys()))
