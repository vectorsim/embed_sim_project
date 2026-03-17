"""
simple_signal_addition.py
=========================
EmbedSim — Example: Signal Summation, Gain, and Integration
============================================================

PURPOSE
-------
Introduces the three standard block categories used in every EmbedSim
simulation and shows how they chain together through the >> operator:

  SOURCE blocks   — generate signals with no upstream inputs
  PROCESSING blocks — transform signals (algebraic, instantaneous)
  DYNAMIC blocks  — carry state across time steps (integrators, delays)

Block graph
-----------

  [const_1  (VectorConstant)]      ──►┐
  [sin_source (SinusoidalGenerator)] ──►┤ [source_sum (VectorSum)]
  [cosine_source (SinusoidalGenerator)]►┘        │
                                                  ▼
                                       [gain (VectorGain)]
                                                  │
                                                  ▼
                                       [integrator (VectorIntegrator)]
                                                  │
                                                  ▼
                                       [output (VectorEnd)]  ← terminal sink

Mathematical description
------------------------
  source_sum(t) = cos(2π·f·t) + sin(2π·f·t) + C       f = FREQ Hz
               = √2 · sin(2π·f·t + π/4)  +  C          (AC + DC)

  gain_out(t)  = GAIN · source_sum(t)

  integrator(t) = ∫₀ᵗ gain_out(τ) dτ
                = GAIN · [ (−cos(2π·f·t) + sin(2π·f·t)) / (2π·f)  +  C·t ]

Topology features demonstrated
-------------------------------
  sim.topo.print_console()      — ASCII signal-flow tree to stdout
  sim.topo.export_html(path)    — interactive pan/zoom HTML diagram
  sim.execution_order           — DFS-sorted block list (sources first)

Run:
    python simple_signal_addition.py
"""

from __future__ import annotations

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")            # safe non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Path bootstrap — use shared utility (safe to call multiple times).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_HERE))   # make local _path_utils importable
from _path_utils import setup_embedsim_path
setup_embedsim_path()

# ---------------------------------------------------------------------------
# EmbedSim imports
# ---------------------------------------------------------------------------
from embedsim.core_blocks       import VectorSignal                    # signal carrier
from embedsim.source_blocks     import VectorConstant, SinusoidalGenerator
from embedsim.processing_blocks import VectorGain, VectorSum
from embedsim.dynamic_blocks    import VectorEnd, VectorIntegrator
from embedsim.simulation_engine import EmbedSim, ODESolver

# =============================================================================
# Simulation parameters
# =============================================================================
FREQ  =  5.0    # Hz   — frequency of sin / cos sources
AMP   =  1.0    # —    — peak amplitude of each sinusoidal source
GAIN  =  3.0    # —    — scalar gain applied before the integrator
C     =  1.3    # —    — DC offset (VectorConstant value)
T_SIM =  4.0    # s    — total simulation time
DT    =  0.01   # s    — fixed time step  (RK4 solver)


# =============================================================================
# Plot
# =============================================================================
def plot_results(sim: EmbedSim, path: str = "simple_signal_addition.png") -> None:
    """
    Four-panel plot:
      Panel 1 — source signals (sin, cos, DC constant)
      Panel 2 — weighted sum out of VectorSum
      Panel 3 — gained sum (VectorGain output)
      Panel 4 — integrated signal (VectorIntegrator state)

    Two separate y-scales are used because the integrator grows linearly
    due to the DC component and would dwarf the source panel if mixed in.
    """
    t    = np.array(sim.scope.t)
    s_c  = sim.scope.get_signal("source_const", index=0)
    s_s  = sim.scope.get_signal("source_sin",   index=0)
    s_co = sim.scope.get_signal("source_cos",   index=0)
    s_sm = sim.scope.get_signal("source_sum",   index=0)
    s_gn = sim.scope.get_signal("gain_out",     index=0)
    s_ig = sim.scope.get_signal("integrator",   index=0)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        f"EmbedSim — Signal Addition + Gain + Integration\n"
        f"f={FREQ} Hz   A={AMP}   k={GAIN}   C={C}   RK4   dt={DT} s",
        fontsize=12, fontweight="bold")

    # ── Panel 1: sources ─────────────────────────────────────────────────────
    axes[0].plot(t, s_s,  color="C3", lw=1.4, label=f"sin  ({FREQ} Hz)")
    axes[0].plot(t, s_co, color="C2", lw=1.4, label=f"cos  ({FREQ} Hz)")
    axes[0].plot(t, s_c,  color="C4", lw=1.4, ls="--", label=f"DC = {C}")
    axes[0].set_ylabel("Amplitude"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("SOURCE blocks — VectorConstant + SinusoidalGenerator × 2")

    # ── Panel 2: sum ─────────────────────────────────────────────────────────
    axes[1].plot(t, s_sm, color="C0", lw=1.4,
                 label="sum = sin + cos + C  =  √2·sin(ωt+π/4) + C")
    axes[1].set_ylabel("Amplitude"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("PROCESSING — VectorSum (signs=[+1,+1,+1])")

    # ── Panel 3: gain ────────────────────────────────────────────────────────
    axes[2].plot(t, s_gn, color="C1", lw=1.4,
                 label=f"gain_out = {GAIN} × sum")
    axes[2].set_ylabel("Amplitude"); axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_title(f"PROCESSING — VectorGain  (k = {GAIN})")

    # ── Panel 4: integrator ──────────────────────────────────────────────────
    axes[3].plot(t, s_ig, color="C5", lw=1.8,
                 label="∫ gain_out dt   (ramps due to DC component)")
    axes[3].set_ylabel("State"); axes[3].legend(fontsize=9)
    axes[3].set_xlabel("Time [s]")
    axes[3].grid(alpha=0.3)
    axes[3].set_title("DYNAMIC — VectorIntegrator  (RK4,  x(0)=0)")

    plt.tight_layout()
    out = str(_HERE / path)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [Plot]  {out}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 68)
    print("  EmbedSim — simple_signal_addition.py")
    print("  Sine + Cosine + DC  →  Sum  →  Gain  →  Integrator")
    print("=" * 68)

    # ── 1. Instantiate source blocks ─────────────────────────────────────────
    #
    #   VectorConstant(name, value_list)
    #     Outputs the same vector at every time step.
    #     Use for DC offsets, reference set-points, parameter injection.
    #
    source_const = VectorConstant("const_1", [C], use_c_backend=False)

    #   SinusoidalGenerator(name, amplitude, freq_hz, phase_rad)
    #     Output: amplitude · sin(2π·freq·t + phase)
    #     source_sin  → phase=0      → A·sin(ωt)
    #     source_cos  → phase=π/2   → A·cos(ωt)
    #
    source_sin = SinusoidalGenerator(
        "sin_source",    AMP, FREQ, 0.0,          use_c_backend=False)
    source_cos = SinusoidalGenerator(
        "cosine_source", AMP, FREQ, np.pi / 2.0,  use_c_backend=False)

    # ── 2. Summation block ────────────────────────────────────────────────────
    #
    #   VectorSum(name, signs)
    #     signs=[+1,+1,+1] → all three inputs added with weight +1.
    #     signs=[-1,+1]    would subtract port 0 from port 1.
    #
    #   Multi-port wiring with >>:
    #     source_cos >> source_sum   → port 0
    #     source_sin >> source_sum   → port 1
    #     source_const >> source_sum → port 2
    #   Port order follows the order of >> calls.
    #
    source_sum = VectorSum("source_sum", [1, 1, 1], use_c_backend=False)
    source_cos   >> source_sum      # port 0
    source_sin   >> source_sum      # port 1
    source_const >> source_sum      # port 2

    # ── 3. Processing + dynamic chain ─────────────────────────────────────────
    #
    #   VectorGain(name, gain)
    #     Instantaneous scalar multiply: output = gain · input
    #
    #   VectorIntegrator(name, initial_state)
    #     Continuous integrator: dx/dt = u
    #     State update uses the solver selected on EmbedSim (RK4 here).
    #     initial_state sets x(0).
    #
    #   VectorEnd  — terminal sink.
    #     EmbedSim walks the graph backward from every VectorEnd to
    #     discover all reachable blocks.
    #
    gain       = VectorGain("gain",       gain=GAIN, use_c_backend=False)
    integrator = VectorIntegrator("integrator", initial_state=[0.0])
    output     = VectorEnd("output")

    # Chain with >>:  sum → gain → integrator → output
    source_sum >> gain >> integrator >> output

    # ── 4. EmbedSim engine ────────────────────────────────────────────────────
    #
    #   ODESolver.RK4 — classic 4th-order Runge-Kutta.
    #   Use RK4 whenever the graph contains VectorIntegrator or other
    #   state-carrying dynamic blocks (better accuracy than Euler at the
    #   same dt).
    #   ODESolver.EULER is sufficient for purely algebraic / source-only graphs.
    #
    sim = EmbedSim(
        sinks  = [output],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.RK4,
    )

    # ── 5. Scope registration ─────────────────────────────────────────────────
    #
    #   MUST happen before sim.run() — the scope allocates its recording
    #   arrays at run-start.
    #
    #   sim.scope.add(block, indices, label)
    #     indices : list of output-vector element indices to record
    #     label   : key used later in sim.scope.get_signal(label, index)
    #
    sim.scope.add(source_const, indices=[0], label="source_const")
    sim.scope.add(source_sin,   indices=[0], label="source_sin")
    sim.scope.add(source_cos,   indices=[0], label="source_cos")
    sim.scope.add(source_sum,   indices=[0], label="source_sum")   # sum node
    sim.scope.add(gain,         indices=[0], label="gain_out")     # after gain
    sim.scope.add(integrator,   indices=[0], label="integrator")

    # ── 6. Topology — console ASCII ───────────────────────────────────────────
    #
    #   print_console() shows the signal-flow tree.
    #   Inspect it to confirm:
    #     • All three sources appear before source_sum
    #     • gain appears after source_sum
    #     • integrator appears after gain
    #   If wiring is wrong (e.g. a port missing) it shows up here,
    #   before any time is wasted running the simulation.
    #
    print("\n[Topology]  Signal-flow diagram (console)")
    print("-" * 50)
    sim.topo.print_console()

    # ── 7. Topology — interactive HTML ────────────────────────────────────────
    _topo_html = str(_HERE / "simple_signal_addition_topology.html")
    sim.topo.export_html(_topo_html)
    print(f"\n[Topology]  Written: {_topo_html}")

    # ── 8. Execution order ────────────────────────────────────────────────────
    #
    #   sim.execution_order is the DFS-topologically-sorted block list.
    #   The engine calls block.compute_py() in exactly this sequence at
    #   every time step.
    #
    #   Expected order for this graph:
    #     sources (const_1, sin_source, cosine_source) — any relative order
    #     source_sum
    #     gain
    #     integrator
    #     output  (VectorEnd — last)
    #
    print("\n[Execution order]  (DFS topological sort)")
    print("-" * 50)
    for step, blk in enumerate(sim.execution_order):
        print(f"  Step {step:>2d} :  {blk.name:<22s}  ({type(blk).__name__})")
    print()

    # ── 9. Run ────────────────────────────────────────────────────────────────
    print("[Run]  Starting simulation …")
    sim.run()
    n_steps = len(sim.scope.t)
    print(f"[Run]  Done — {n_steps} steps  "
          f"(T={T_SIM} s  dt={DT} s  RK4)\n")

    # ── 10. Plot ──────────────────────────────────────────────────────────────
    plot_results(sim)

    print("\n  Output files:")
    print("    simple_signal_addition.png")
    print("    simple_signal_addition_topology.html")
    print("\n[Done]")
