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
import numpy as np

from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.source_blocks      import SinusoidalGenerator
from embedsim.processing_blocks  import VectorSum, VectorGain
from embedsim.dynamic_blocks     import VectorEnd
from embedsim.simulation_engine  import EmbedSim, ODESolver
from embedsim.plot_helper        import create_plotter
from embedsim.topology_printer   import TopologyPrinter, attach


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
    sim.scope.add(adder,   indices=[0], label="sum")  # Add the sum signal too

    return sim


# =============================================================================
# Run + plot + visualize topology
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  EmbedSim — Two Sine Sources + Gain")
    print("=" * 70)

    sim = build_sim()

    # ── Attach topology printer ───────────────────────────────────────────────
    # This adds sim.topo and sim.print_topology_sources2sink()
    attach(sim, title="Two Sine Sources + Gain - Topology")

    # Print topology to console
    print("\n📊 Topology (via sim.topo):")
    sim.topo.print_console()

    print("\n📊 Topology (via built-in method):")
    sim.print_topology_sources2sink()

    # Run simulation
    print("\n⚙️  Running simulation...")
    sim.run()
    print(f"✅ Done — {len(sim.scope.t)} samples over "
          f"{sim.scope.t[-1]*1000:.1f} ms")

    # ── Show all recorded signals ─────────────────────────────────────────────
    plotter = create_plotter(sim)
    plotter.info()

    print("\n📋 Available signals:")
    plotter.list_signals()

    # ── Create various plots ──────────────────────────────────────────────────

    # 1. Simple plot with component names (this works!)
    print("\n📊 Creating plot with component names...")
    plotter.easyplot(
        signals   = ["sin_a5[0]", "sin_a2[0]", "gain[0]"],
        title     = "Two Sine Sources + Gain",
        figsize   = (12, 5),
        save_path = "two_sines_gain.png",
    )

    # 2. Enhanced plot with all signals including sum (using plotter's easyplot)
    print("\n📊 Creating enhanced plot with all signals...")
    plotter.easyplot(
        signals   = ["sin_a5[0]", "sin_a2[0]", "sum[0]", "gain[0]"],
        title     = "Two Sine Sources + Gain - All Signals",
        figsize   = (12, 6),
        save_path = "two_sines_gain_all_signals.png",
    )

    # 3. Manual plot with subplots (using correct data access)
    print("\n📊 Creating custom subplot figure...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Convert time to numpy array for mathematical operations
    t = np.array(sim.scope.t)

    # Access data through sim.scope.data dictionary
    signal1 = np.array(sim.scope.data["sin_a5[0]"])
    signal2 = np.array(sim.scope.data["sin_a2[0]"])
    signal_sum = np.array(sim.scope.data["sum[0]"])
    signal_gain = np.array(sim.scope.data["gain[0]"])

    # Top subplot: Individual sources
    ax1 = axes[0]
    ax1.plot(t, signal1, 'b-', linewidth=1.5,
             label=r'sin_a5: $5.0 \cdot \sin(2\pi \cdot 15t)$')
    ax1.plot(t, signal2, 'g-', linewidth=1.5,
             label=r'sin_a2: $2.0 \cdot \sin(2\pi \cdot 15t + 1.56)$')
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Individual Sources', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.56])

    # Bottom subplot: Sum and Gain
    ax2 = axes[1]

    # Calculate theoretical values for verification (using numpy arrays)
    theoretical_sum = 5.0 * np.sin(2 * np.pi * 15 * t) + 2.0 * np.sin(2 * np.pi * 15 * t + 1.56)
    theoretical_gain = GAIN_VALUE * theoretical_sum

    ax2.plot(t, signal_sum, 'm-', linewidth=1.5, alpha=0.7,
             label='sum = sin_a5 + sin_a2 (from scope)')
    ax2.plot(t, theoretical_sum, 'm--', linewidth=1, alpha=0.5,
             label='sum (theoretical)')
    ax2.plot(t, signal_gain, 'r-', linewidth=2,
             label=f'gain = {GAIN_VALUE}·sum (from scope)')
    ax2.plot(t, theoretical_gain, 'k--', linewidth=1, alpha=0.5,
             label='gain (theoretical)')

    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title(f'Sum and Gain (Gain = {GAIN_VALUE})', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.56])

    plt.suptitle('Two Sine Sources + Gain - Detailed Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("two_sines_gain_detailed.png", dpi=150, bbox_inches='tight')
    plt.show()

    # 4. Phase relationship plot (XY plot)
    print("\n📊 Creating XY plot (Lissajous figure)...")
    fig, ax = plotter.xy_plot("sin_a5[0]", "sin_a2[0]",
                               title="Phase Relationship: sin_a5 vs sin_a2",
                               figsize=(7, 7),
                               color='purple')
    if fig:
        fig.savefig("two_sines_gain_xy.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 5. FFT analysis
    print("\n📊 Creating FFT plot...")
    fig, axes = plotter.fft_plot("gain[0]", max_freq=100,
                                  title="Frequency Spectrum of Gain Output",
                                  figsize=(12, 6))
    if fig:
        fig.savefig("two_sines_gain_fft.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 6. Component plot (showing all components of a block)
    print("\n📊 Creating component plot for gain block...")
    fig, axes = plotter.plot_components("gain", figsize=(10, 4))
    if fig:
        fig.savefig("two_sines_gain_component.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Export HTML topology ──────────────────────────────────────────────────
    print("\n📊 Exporting interactive topology visualization...")
    html_path = sim.topo.export_html("two_sines_gain_topology.html")

    # Open in browser (optional - comment out if you don't want browser to open)
    print("   Opening topology in browser...")
    sim.topo.show_gui()  # Opens in browser

    # ── Save all plots ────────────────────────────────────────────────────────
    print("\n💾 Saving all plots with prefix 'two_sines_gain'...")
    plotter.save_all_plots(prefix="two_sines_gain", format="png")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("✅ COMPLETE - All outputs generated:")
    print("=" * 70)
    print("\n📊 Topology:")
    print("   - Console output (above)")
    print(f"   - Interactive HTML: two_sines_gain_topology.html")

    print("\n📈 Plots:")
    print("   - two_sines_gain.png (basic plot with component names)")
    print("   - two_sines_gain_all_signals.png (all signals together)")
    print("   - two_sines_gain_detailed.png (custom subplot with verification)")
    print("   - two_sines_gain_xy.png (Lissajous figure)")
    print("   - two_sines_gain_fft.png (frequency spectrum)")
    print("   - two_sines_gain_component.png (component view)")

    print("\n📊 Signal statistics:")
    for signal in ["sin_a5[0]", "sin_a2[0]", "sum[0]", "gain[0]"]:
        if signal in sim.scope.data:
            data = np.array(sim.scope.data[signal])
            print(f"   {signal:10s}: min={np.min(data):6.3f}, "
                  f"max={np.max(data):6.3f}, mean={np.mean(data):6.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()