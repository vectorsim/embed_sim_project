"""
example_two_sines_gain.py
=========================
EmbedSim — Introductory example: two sine sources, gain, summer
================================================================

PURPOSE
-------
This file demonstrates the core EmbedSim workflow on the simplest
possible signal-flow graph:

    SineSource("sine_a")  ──────────────────────────────► SumBlock("summer")
    SineSource("sine_b")  ──► GainBlock("gain_b", k=2.0) ──► SumBlock("summer")
                                                                     │
                                                               VectorEnd("sink")

It focuses on two features that are useful for understanding and
debugging any EmbedSim simulation:

  1. Topology Printer  — ASCII console view + interactive HTML export
                         (sim.topo.print_console / sim.topo.export_html)

  2. Execution order   — the DFS-sorted block list that the simulation
                         engine walks at every time step
                         (sim.execution_order)

No FMU, no Cython extension, no CodeGen — just pure Python blocks so
the file runs anywhere with only embedsim + numpy + matplotlib.

WHAT TO LOOK FOR
----------------
Console output after "Run this script":

  [Topology] Signal-flow diagram:
      ┌─ cg_start ─ sine_a ─ ...
      ...
  [Topology] Written: example_signal_flow.html

  [Execution order]  (DFS topological sort — engine walks this list)
      Step 0 : sine_a        SineSource
      Step 1 : sine_b        SineSource
      Step 2 : gain_b        GainBlock
      Step 3 : summer        SumBlock
      Step 4 : sink          VectorEnd
  ...

Run:
    python example_two_sines_gain.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on all machines
import matplotlib.pyplot as plt
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Path bootstrap — use shared utility (safe to call multiple times).
# Walks up to the .project_root_marker file and inserts project root onto
# sys.path so "from embedsim import ..." always resolves.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_HERE))   # make local _path_utils importable
from _path_utils import setup_embedsim_path
setup_embedsim_path()

# ---------------------------------------------------------------------------
# EmbedSim — public surface
# ---------------------------------------------------------------------------
from embedsim import EmbedSim, ODESolver, VectorEnd          # engine + sink
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
T_SIM = 0.10          # total simulation time  [s]
DT    = 1e-4          # fixed time step         [s]  → 10 000 steps

FREQ_A = 50.0         # sine_a frequency  [Hz]
AMP_A  = 1.0          # sine_a amplitude  [V]

FREQ_B = 150.0        # sine_b frequency  [Hz]  (3rd harmonic)
AMP_B  = 0.5          # sine_b amplitude  [V]

GAIN_B = 2.0          # scalar gain applied to sine_b before summation


# =============================================================================
# Block definitions
# =============================================================================

class SineSource(VectorBlock):
    """
    Pure-Python sine generator block.

    Output vector: [sin(2π·freq·t) * amp]   (1 element)

    This is a SOURCE block — it has no upstream inputs.  EmbedSim
    DFS-sorts source blocks to the front of the execution order
    automatically.
    """

    def __init__(self, name: str, freq: float, amp: float = 1.0) -> None:
        super().__init__(name)
        self.freq = freq
        self.amp  = amp
        # No is_dynamic flag needed — stateless block, recomputed each step.

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        """
        Called by the simulation engine once per time step.

        Parameters
        ----------
        t   : current simulation time  [s]
        dt  : fixed step size          [s]  (unused here — stateless)
        input_values : list[VectorSignal | None] — upstream port signals.
                       None for a source block.

        Returns
        -------
        VectorSignal wrapping a 1-element numpy array.
        """
        value = self.amp * math.sin(2.0 * math.pi * self.freq * t)
        self.output = VectorSignal(
            np.array([value], dtype=DEFAULT_DTYPE), self.name)
        return self.output


class GainBlock(VectorBlock):
    """
    Element-wise scalar gain: output = k * input[0]

    This is a PROCESSING block — it sits between SineSource and SumBlock
    in the execution order.  EmbedSim DFS ensures all upstream blocks
    (SineSource here) are computed before this block is called.
    """

    def __init__(self, name: str, k: float = 1.0) -> None:
        super().__init__(name)
        self.k = k

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        """
        Multiply every element of the upstream signal by scalar k.

        input_values[0] is the VectorSignal produced by the immediately
        upstream block.  EmbedSim guarantees it is not None when this
        method is called (the block has one declared input edge).
        """
        upstream = input_values[0].value if input_values else np.zeros(1)
        self.output = VectorSignal(
            (self.k * upstream).astype(DEFAULT_DTYPE), self.name)
        return self.output


class SumBlock(VectorBlock):
    """
    Element-wise sum over all upstream ports.

    This is a SINK-adjacent block.  It merges two incoming signals
    (sine_a and the gained sine_b) into a single output vector.

    Multi-port wiring in EmbedSim:
        sine_a  >> summer   →  port 0
        gain_b  >> summer   →  port 1
    EmbedSim concatenates them as input_values = [sig_port0, sig_port1].
    Both signals must have the same length for element-wise addition.
    """

    def compute_py(self, t: float, dt: float, input_values=None) -> VectorSignal:
        """
        Sum all upstream port signals element-wise.

        Guards against None entries — can happen in unit tests or when a
        port has not yet been connected.
        """
        result = None
        for sig in (input_values or []):
            if sig is None:
                continue
            v = sig.value
            result = v if result is None else result + v

        if result is None:
            result = np.zeros(1, dtype=DEFAULT_DTYPE)

        self.output = VectorSignal(result.astype(DEFAULT_DTYPE), self.name)
        return self.output


# =============================================================================
# build_and_run
# =============================================================================
def build_and_run() -> dict:
    """
    Instantiate blocks, wire them, configure EmbedSim, run the simulation.

    This function demonstrates the complete EmbedSim workflow:

      1.  Instantiate blocks.
      2.  Wire blocks with the >> operator (builds the signal-flow graph).
      3.  Create EmbedSim(sinks=[...]) — triggers DFS topology sort.
      4.  Register scope channels  (MUST be before sim.run()).
      5.  Print topology to console  (ASCII art).
      6.  Export topology to HTML    (interactive pan/zoom).
      7.  Print execution order      (sorted block list).
      8.  sim.run()                  (fixed-step Euler time loop).
      9.  Extract scope data.

    Returns
    -------
    dict with numpy arrays: t, sig_a, sig_b_gained, sig_sum
    """

    # ── 1. Instantiate ───────────────────────────────────────────────────────
    sine_a = SineSource("sine_a", freq=FREQ_A, amp=AMP_A)
    sine_b = SineSource("sine_b", freq=FREQ_B, amp=AMP_B)
    gain_b = GainBlock ("gain_b", k=GAIN_B)
    summer = SumBlock  ("summer")
    sink   = VectorEnd ("sink")

    # ── 2. Wire ──────────────────────────────────────────────────────────────
    #
    #   A >> B   means: "the output of A becomes an input of B"
    #                   and adds the edge A → B to the signal-flow graph.
    #
    #   EmbedSim supports multi-port blocks:
    #       A >> C   →  C.input_values[0] = A.output
    #       B >> C   →  C.input_values[1] = B.output
    #   Ports are assigned in wiring order.
    #
    sine_a >> gain_b    # port 0 of gain_b  (only port)
    sine_a >> summer    # port 0 of summer
    gain_b >> summer    # port 1 of summer
    summer >> sink      # terminal — EmbedSim walks the graph backward from here

    # ── 3. Create EmbedSim ───────────────────────────────────────────────────
    #
    #   sinks  : one or more VectorEnd blocks.  EmbedSim performs a DFS
    #            backward from each sink to discover ALL reachable blocks,
    #            then reverses to produce the execution order (sources first).
    #
    #   solver : ODESolver.EULER — fixed-step forward Euler.
    #            ODESolver.RK4  is available for stiff systems.
    #
    sim = EmbedSim(
        sinks  = [sink],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )

    # ── 4. Register scope channels ───────────────────────────────────────────
    #
    #   sim.scope.add(block, indices, label)
    #     block   : any VectorBlock in the graph
    #     indices : list of output-vector element indices to record
    #     label   : key used later with sim.scope.get_signal(label, index)
    #
    #   IMPORTANT: All scope.add() calls must happen BEFORE sim.run().
    #   The scope allocates its recording arrays when sim.run() is called.
    #
    sim.scope.add(sine_a, indices=[0], label="sine_a")
    sim.scope.add(gain_b, indices=[0], label="gained_b")   # sine_b * GAIN_B
    sim.scope.add(summer, indices=[0], label="sum_out")

    # ── 5. Topology — ASCII console ──────────────────────────────────────────
    #
    #   sim.topo.print_console() walks the sorted block list and prints a
    #   human-readable signal-flow tree to stdout.  Useful for a quick
    #   sanity-check that wiring is correct before a long simulation.
    #
    print("\n" + "=" * 60)
    print("  [Topology]  Signal-flow diagram (console)")
    print("=" * 60)
    sim.topo.print_console()

    # ── 6. Topology — interactive HTML ───────────────────────────────────────
    #
    #   sim.topo.export_html(path) writes a self-contained HTML file with
    #   the block graph rendered as a pannable, zoomable diagram.
    #   Each node shows the block name, class, and wiring.
    #
    _topo_html = str(_HERE / "example_signal_flow.html")
    sim.topo.export_html(_topo_html)
    print(f"\n  [Topology]  Written: {_topo_html}\n")

    # ── 7. Execution order ───────────────────────────────────────────────────
    #
    #   sim.execution_order is the DFS-topologically-sorted list of blocks.
    #   The simulation engine calls block.compute_py() in exactly this order
    #   at every time step.  Inspect it to verify that:
    #     • All source blocks appear before their consumers.
    #     • LoopBreaker nodes are placed correctly in feedback chains.
    #     • No unexpected blocks crept into the graph.
    #
    print("=" * 60)
    print("  [Execution order]  (DFS topological sort — engine walks this list)")
    print("=" * 60)
    # execution_order lives on the topology printer as sorted_blocks
    _order = (
        sim.topo.sorted_blocks
        if hasattr(sim.topo, "sorted_blocks")
        else getattr(sim, "execution_order", [])
    )
    for step, blk in enumerate(_order):
        class_name = type(blk).__name__
        print(f"  Step {step:>2d} :  {blk.name:<20s}  ({class_name})")
    print()

    # ── 8. Run ───────────────────────────────────────────────────────────────
    print("  [Run]  Starting simulation …")
    sim.run()
    print(f"  [Run]  Done — {int(T_SIM / DT)} steps × dt={DT*1e6:.0f} µs\n")

    # ── 9. Extract scope data ─────────────────────────────────────────────────
    sc = sim.scope
    return {
        "t":         np.array(sc.t,                     dtype=np.float32),
        "sig_a":     sc.get_signal("sine_a",   0),      # raw sine_a
        "sig_b_g":   sc.get_signal("gained_b", 0),      # sine_b × GAIN_B
        "sig_sum":   sc.get_signal("sum_out",  0),      # summer output
    }


# =============================================================================
# Plot
# =============================================================================
def plot_results(d: dict, path: str = "example_two_sines_gain.png") -> None:
    """
    Three-panel plot:
      Top    : sine_a  (50 Hz)
      Middle : gained_b  (150 Hz × 2)
      Bottom : sum  (composite waveform)
    """
    t   = d["t"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        f"EmbedSim — Two Sines + Gain  "
        f"(f_a={FREQ_A} Hz, f_b={FREQ_B} Hz, k={GAIN_B})",
        fontsize=12, fontweight="bold")

    axes[0].plot(t * 1e3, d["sig_a"], color="C0", lw=1.2,
                 label=f"sine_a  ({FREQ_A} Hz, A={AMP_A})")
    axes[0].set_ylabel("Amplitude"); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("SineSource: sine_a")

    axes[1].plot(t * 1e3, d["sig_b_g"], color="C1", lw=1.2,
                 label=f"gained_b  ({FREQ_B} Hz, A={AMP_B}×{GAIN_B}={AMP_B*GAIN_B})")
    axes[1].set_ylabel("Amplitude"); axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title(f"SineSource → GainBlock (k={GAIN_B}): sine_b × {GAIN_B}")

    axes[2].plot(t * 1e3, d["sig_sum"], color="C2", lw=1.2,
                 label="sum = sine_a + gained_b")
    axes[2].set_ylabel("Amplitude"); axes[2].legend(fontsize=9)
    axes[2].set_xlabel("Time [ms]")
    axes[2].grid(alpha=0.3)
    axes[2].set_title("SumBlock: composite waveform")

    plt.tight_layout()
    # Save next to the script (original location)
    out = str(_HERE / path)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [Plot]  {out}")


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  EmbedSim — example_two_sines_gain.py")
    print("  Demonstrates:")
    print("    • Block wiring with the >> operator")
    print("    • sim.topo.print_console()  — ASCII topology")
    print("    • sim.topo.export_html()    — interactive HTML")
    print("    • sim.execution_order       — DFS sorted block list")
    print("    • sim.scope                 — signal recording")
    print(f"  T_SIM = {T_SIM*1e3:.0f} ms   DT = {DT*1e6:.0f} µs")
    print("=" * 60)

    data = build_and_run()
    plot_results(data)

    print()
    print("  Output files:")
    print("    example_signal_flow.html       ← interactive topology")
    print("    example_two_sines_gain.png     ← three-panel waveform plot")
    print()
    print("  [Done]")
