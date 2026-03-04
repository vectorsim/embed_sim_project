"""
eb_test.py
==========

Showcase: three-phase generator → Clarke transform → sink.

Author : EmbedSim Framework
"""

# =============================================================================
# Signal flow
# =============================================================================
#
#   ThreePhaseGenerator       ClarkeTransformBlock      VectorEnd
#   block name: "gen"         block name: "clarke"      "sink"
#   scope label: "gen[a,b,c]" scope label: "clarke[alpha,beta]"
#        │                          │
#        ▼                          ▼
#   Phase3Signal         ──►  AlphaBetaSignal       ──►  (recorded)
#   [a, b, c]                 [alpha, beta]
#        Iabc                      Iαβ
#
#   PlotHelper keys:          PlotHelper keys:
#   "gen[a,b,c][0]"  = a      "clarke[alpha,beta][0]" = alpha
#   "gen[a,b,c][1]"  = b      "clarke[alpha,beta][1]" = beta
#   "gen[a,b,c][2]"  = c
#
#   NOTE: block name must be a simple identifier (no brackets).
#         Brackets belong only in the scope label.
# =============================================================================

import sys
import matplotlib.pyplot as plt

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.source_blocks     import ThreePhaseGenerator
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.plot_helper       import create_plotter

from coordinate_transform_blocks import ClarkeTransformBlock


# =============================================================================
# Build simulation
# =============================================================================

def build_sim():

    # ── Source: 50 Hz, amplitude 10 V ────────────────────────────────────────
    generator = ThreePhaseGenerator(
        "gen",
        amplitude     = 10.0,
        freq          = 50.0,
        use_c_backend = False,
    )

    # ── Clarke: Phase3Signal [a,b,c] → AlphaBetaSignal [alpha,beta] ──────────
    try:
        clarke = ClarkeTransformBlock("clarke", use_c_backend=True)
        print("✅ Clarke: C backend (Cython wrapper)")
    except ImportError:
        clarke = ClarkeTransformBlock("clarke", use_c_backend=False)
        print("⚠️  Clarke: Python backend  (run build_all.bat to compile C)")

    # ── Sink ──────────────────────────────────────────────────────────────────
    sink = VectorEnd("sink")

    # ── Connect: generator → clarke → sink ───────────────────────────────────
    generator >> clarke >> sink

    # ── Simulation: 2 cycles of 50 Hz = 40 ms, dt = 0.1 ms ──────────────────
    sim = EmbedSim(
        sinks  = [sink],
        T      = 0.04,
        dt     = 0.0001,
        solver = ODESolver.RK4,
    )

    # ── Scope ─────────────────────────────────────────────────────────────────
    sim.scope.add(generator, label="gen[a,b,c]")
    sim.scope.add(clarke,    label="clarke[alpha,beta]")

    return sim


# =============================================================================
# Run + plot
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  EmbedSim — Three-phase  →  Clarke Transform")
    print("=" * 60)

    sim = build_sim()

    # Topology — signal labels come from block.output_label automatically
    print("\n📊 Topology:")
    sim.print_topology_sources2sink()

    # Run
    print("\n⚙️  Running ...")
    sim.run()
    print(f"✅ Done — {len(sim.scope.t)} samples over "
          f"{sim.scope.t[-1]*1000:.1f} ms")

    # ── PlotHelper ────────────────────────────────────────────────────────────
    plotter = create_plotter(sim)
    plotter.info()

    # Plot 1: three-phase [a,b,c] top, Clarke [alpha,beta] bottom
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("Three-phase  →  Clarke Transform", fontsize=13, fontweight="bold")

    plotter.easyplot(signals="gen[a,b,c]",        title="Input  [a, b, c]",      ax=ax_top)
    plotter.easyplot(signals="clarke[alpha,beta]", title="Clarke  [alpha, beta]", ax=ax_bot)

    ax_bot.set_xlabel("Time [s]")
    plt.tight_layout()
    fig.savefig("clarke_time_traces.png", dpi=120, bbox_inches="tight")
    print("💾  Saved: clarke_time_traces.png")
    plt.show()

    # Plot 2: Clarke locus
    fig2, _ = plotter.xy_plot(
        x_signal   = "clarke[alpha,beta][0]",
        y_signal   = "clarke[alpha,beta][1]",
        title      = "Clarke Locus  (alpha vs beta) — balanced → circle",
        color      = "tab:purple",
        equal_axes = True,
    )
    if fig2:
        fig2.savefig("clarke_locus.png", dpi=120, bbox_inches="tight")
        print("💾  Saved: clarke_locus.png")

    print("=" * 60)


if __name__ == "__main__":
    main()
