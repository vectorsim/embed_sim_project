"""
example_two_sines_gain.py
=========================

EmbedSim — Example: Two Sinusoidal Sources + Gain

Signal flow:

   [sin_a5 (SinusoidalGenerator)] ──►┐
                                      ├──► [sum (VectorSum)] ──► [gain (VectorGain)] ──► [output (VectorEnd)]
   [sin_a2 (SinusoidalGenerator)] ──►┘

Signal Description:
    source1(t) = 5.0 · sin(2π · 15t)
    source2(t) = 2.0 · sin(2π · 15t + 1.56 rad)
    sum(t)     = source1(t) + source2(t)
    output(t)  = 0.5 · sum(t)

Parameters:
    Amplitude 1 : 5.0
    Amplitude 2 : 2.0
    Frequency   : 15 Hz
    Phase       : 1.56 rad  (source2 only)
    Gain        : 0.5
    Duration    : 0.56 s
    Time step   : 0.001 s
    Solver      : RK4

Author : EmbedSim Framework
"""

import sys
import matplotlib.pyplot as plt

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.source_blocks      import SinusoidalGenerator
from embedsim.processing_blocks  import VectorSum, VectorGain
from embedsim.dynamic_blocks     import VectorEnd
from embedsim.simulation_engine  import EmbedSim, ODESolver
from embedsim.plot_helper        import create_plotter


# =============================================================================
# Build simulation
# =============================================================================

GAIN_VALUE = 0.5   # single source of truth — used in both block and label

def build_sim():

    # ── Sources ───────────────────────────────────────────────────────────────
    source1 = SinusoidalGenerator("sin_a5", amplitude=5.0, freq=15.0,
                                  output_label="Ia")
    source2 = SinusoidalGenerator("sin_a2", amplitude=2.0, freq=15.0,
                                  phase=1.56, output_label="Ib")

    # ── Processing ────────────────────────────────────────────────────────────
    adder = VectorSum("sum",   signs=[1, 1],
                      output_label="Ia+Ib")

    gain  = VectorGain("gain", gain=GAIN_VALUE,
                       output_label=f"{GAIN_VALUE}·(Ia+Ib)")

    # ── Sink ──────────────────────────────────────────────────────────────────
    output = VectorEnd("output")

    # ── Connect ───────────────────────────────────────────────────────────────
    source1 >> adder
    source2 >> adder
    adder >> gain >> output

    # ── Simulation ────────────────────────────────────────────────────────────
    sim = EmbedSim(
        sinks  = [output],
        T      = 0.56,
        dt     = 0.001,
        solver = ODESolver.RK4,
    )

    # ── Scope ─────────────────────────────────────────────────────────────────
    sim.scope.add(source1, indices=[0], label="sin_a5")
    sim.scope.add(source2, indices=[0], label="sin_a2")
    sim.scope.add(gain,    indices=[0], label="gain")

    return sim


# =============================================================================
# Run + plot
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  EmbedSim — Two Sine Sources + Gain")
    print("=" * 60)

    sim = build_sim()

    # Topology
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

    plotter.easyplot(
        signals   = ["sin_a5[0]", "sin_a2[0]", "gain[0]"],
        title     = "Two Sine Sources + Gain",
        figsize   = (12, 5),
        save_path = "two_sines_gain.png",
    )

    print("=" * 60)


if __name__ == "__main__":
    main()