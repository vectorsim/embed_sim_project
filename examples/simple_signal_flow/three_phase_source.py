"""
three_phase_source.py
=====================
EmbedSim — Example: Balanced Three-Phase Signal Generation
===========================================================

PURPOSE
-------
Demonstrates the minimal EmbedSim simulation: a single source block
wired to a VectorEnd sink, with topology inspection and scope recording.

This is the "hello world" of EmbedSim.  Use it as the first check
whenever you integrate a new source block — if topology and scope work
here, they will work inside a larger FOC chain.

Block graph
-----------

  [3phase_gen (ThreePhaseGenerator)] ──► [output (VectorEnd)]

  ThreePhaseGenerator outputs a 3-element vector at every time step:

    [u_a(t), u_b(t), u_c(t)]

  where the balanced three-phase voltages are:

    u_a(t) = A · sin(2π·f·t)
    u_b(t) = A · sin(2π·f·t − 2π/3)    (120° lagging)
    u_c(t) = A · sin(2π·f·t + 2π/3)    (120° leading)

  Sum rule (sanity check): u_a + u_b + u_c = 0  at all t.

Topology features demonstrated
-------------------------------
  sim.topo.print_console()           — ASCII signal-flow tree to stdout
  sim.topo.export_html(path)         — interactive pan/zoom HTML diagram
  sim.execution_order                — DFS-sorted block list
  sim.topo.print_sources_to_sink()   — alternative sources→sink view

Scope feature demonstrated
--------------------------
  sim.scope.add(block, label)        — record all output indices by default
  sim.scope.get_signal(label, index) — retrieve recorded array after run

Run:
    python three_phase_source.py
"""

from __future__ import annotations

import sys
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
# EmbedSim imports
# ---------------------------------------------------------------------------
from embedsim.source_blocks     import ThreePhaseGenerator
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver

# =============================================================================
# Simulation parameters
# =============================================================================
AMP    = 10.0      # V     — peak phase voltage
FREQ   = 50.0      # Hz    — fundamental frequency  (1 cycle = 20 ms)
T_SIM  = 0.04      # s     — 2 full cycles  (enough to show symmetry)
DT     = 1e-4      # s     — 0.1 ms step  → 400 steps total


# =============================================================================
# Plot
# =============================================================================
def plot_results(sim: EmbedSim, path: str = "three_phase_source.png") -> None:
    """
    Two-panel plot:
      Top    — full simulation window, all three phases
      Bottom — zoomed to the first cycle (20 ms) for waveform clarity

    The bottom panel also plots the instantaneous sum u_a + u_b + u_c,
    which should be zero at all times for a balanced source — this is a
    useful sanity check after any ThreePhaseGenerator change.
    """
    t  = np.array(sim.scope.t)
    ua = sim.scope.get_signal("3phase", 0)    # phase U  (reference)
    ub = sim.scope.get_signal("3phase", 1)    # phase V  (−120°)
    uc = sim.scope.get_signal("3phase", 2)    # phase W  (+120°)
    s3 = ua + ub + uc                          # balance check — must be ≈ 0

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    fig.suptitle(
        f"EmbedSim — Balanced Three-Phase Source\n"
        f"A={AMP} V   f={FREQ} Hz   dt={DT*1e3:.2f} ms",
        fontsize=12, fontweight="bold")

    # ── Top panel: full window ────────────────────────────────────────────────
    axes[0].plot(t * 1e3, ua, color="C3", lw=1.5, label="u_a  (0°)")
    axes[0].plot(t * 1e3, ub, color="C2", lw=1.5, label="u_b  (−120°)")
    axes[0].plot(t * 1e3, uc, color="C0", lw=1.5, label="u_c  (+120°)")
    axes[0].set_ylabel("Voltage [V]")
    axes[0].legend(fontsize=10, loc="upper right")
    axes[0].grid(alpha=0.3)
    axes[0].set_title(
        "ThreePhaseGenerator — all phases  "
        f"({int(T_SIM * FREQ)} cycles)")

    # ── Bottom panel: first cycle + balance check ─────────────────────────────
    mask = t <= (1.0 / FREQ)      # first full cycle
    axes[1].plot(t[mask] * 1e3, ua[mask], color="C3", lw=1.8, label="u_a")
    axes[1].plot(t[mask] * 1e3, ub[mask], color="C2", lw=1.8, label="u_b")
    axes[1].plot(t[mask] * 1e3, uc[mask], color="C0", lw=1.8, label="u_c")
    axes[1].plot(t[mask] * 1e3, s3[mask], color="k",  lw=1.0,
                 ls="--", label="u_a+u_b+u_c  (balance check ≈ 0)")
    axes[1].set_ylabel("Voltage [V]")
    axes[1].set_xlabel("Time [ms]")
    axes[1].legend(fontsize=9, loc="upper right")
    axes[1].grid(alpha=0.3)
    axes[1].set_title("First cycle  (20 ms)  +  instantaneous sum check")

    plt.tight_layout()
    out = str(_HERE / path)
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [Plot]  {out}")


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    print("\n" + "=" * 60)
    print("  EmbedSim — three_phase_source.py")
    print(f"  A={AMP} V   f={FREQ} Hz   T={T_SIM*1e3:.0f} ms   dt={DT*1e3:.2f} ms")
    print("=" * 60)

    # ── 1. Instantiate ───────────────────────────────────────────────────────
    #
    #   ThreePhaseGenerator(name, amplitude, freq, use_c_backend)
    #     Output vector : [u_a, u_b, u_c]  (3 elements)
    #     u_a = A·sin(ωt)
    #     u_b = A·sin(ωt − 2π/3)
    #     u_c = A·sin(ωt + 2π/3)
    #
    generator = ThreePhaseGenerator(
        "3phase_gen",
        amplitude    = AMP,
        freq         = FREQ,
        use_c_backend= False,
    )

    #   VectorEnd — terminal sink.
    #   EmbedSim performs a DFS backward from every VectorEnd to discover
    #   all reachable blocks.  For this minimal example there is only one
    #   upstream block (generator), so the execution order will be:
    #     Step 0 : 3phase_gen  (ThreePhaseGenerator)
    #     Step 1 : output      (VectorEnd)
    #
    output = VectorEnd("output")

    # ── 2. Wire ──────────────────────────────────────────────────────────────
    #
    #   generator >> output
    #   The entire output vector [u_a, u_b, u_c] flows to output port 0.
    #
    generator >> output

    # ── 3. EmbedSim engine ───────────────────────────────────────────────────
    #
    #   ODESolver.RK4 — selected here to match the pattern used in larger
    #   examples.  For a purely algebraic source graph (no integrators)
    #   ODESolver.EULER gives identical results.
    #
    sim = EmbedSim(
        sinks  = [output],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.RK4,
    )

    # ── 4. Scope registration ─────────────────────────────────────────────────
    #
    #   sim.scope.add(block, label) — without `indices`, all output elements
    #   are recorded automatically.  For the 3-element generator that means
    #   index 0 = u_a, 1 = u_b, 2 = u_c.
    #
    #   Retrieve after sim.run() with:
    #     sim.scope.get_signal("3phase", 0)  → u_a array
    #     sim.scope.get_signal("3phase", 1)  → u_b array
    #     sim.scope.get_signal("3phase", 2)  → u_c array
    #
    sim.scope.add(generator, label="3phase")

    # ── 5. Topology — console ASCII ───────────────────────────────────────────
    #
    #   sim.topo.print_console()
    #     Walks the sorted execution order and prints a signal-flow tree.
    #     For this graph expect:
    #       3phase_gen ──► output
    #
    print("\n[Topology]  Signal-flow diagram (console)")
    print("-" * 44)
    sim.topo.print_console()

    # ── 7. Topology — interactive HTML ────────────────────────────────────────
    _topo_html = str(_HERE / "three_phase_source_topology.html")
    sim.topo.export_html(_topo_html)
    print(f"\n[Topology]  Written: {_topo_html}")

    # ── 8. Execution order ────────────────────────────────────────────────────
    #
    #   For a minimal source → sink graph the execution order is trivial,
    #   but printing it here sets the habit for larger simulations where
    #   it matters (e.g. checking that LoopBreaker nodes are in the right
    #   position inside a feedback chain).
    #
    print("\n[Execution order]  (DFS topological sort)")
    print("-" * 44)
    for step, blk in enumerate(sim.execution_order):
        print(f"  Step {step:>2d} :  {blk.name:<20s}  ({type(blk).__name__})")
    print()

    # ── 9. Run ────────────────────────────────────────────────────────────────
    print("[Run]  Starting simulation …")
    sim.run()
    n_steps = len(sim.scope.t)
    print(f"[Run]  Done — {n_steps} steps  "
          f"(T={T_SIM*1e3:.0f} ms  dt={DT*1e3:.3f} ms  RK4)\n")

    # ── 10. Balance sanity check ──────────────────────────────────────────────
    #
    #   For a balanced three-phase source the instantaneous sum of all
    #   three phases must equal zero at every sample.
    #   If max(|u_a+u_b+u_c|) is not negligibly small there is a bug
    #   in ThreePhaseGenerator or a phase-shift error.
    #
    ua  = sim.scope.get_signal("3phase", 0)
    ub  = sim.scope.get_signal("3phase", 1)
    uc  = sim.scope.get_signal("3phase", 2)
    err = float(np.max(np.abs(ua + ub + uc)))
    print(f"[Sanity]  max|u_a+u_b+u_c| = {err:.2e}  "
          f"({'OK — balanced' if err < 1e-9 else 'WARN — check phases'})")

    # ── 11. Plot ──────────────────────────────────────────────────────────────
    plot_results(sim)

    print("\n  Output files:")
    print("    three_phase_source.png")
    print("    three_phase_source_topology.html")
    print("\n[Done]")


if __name__ == "__main__":
    main()
