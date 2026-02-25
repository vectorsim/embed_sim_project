"""
example_two_sines_gain.py
=========================
ControlForge — Example: Two Sinusoidal Sources + Gain

Category:
    Signal Processing / Time-Domain Simulation

Purpose:
    Demonstrates the combination of two sinusoidal sources using VectorSum,
    scaling the result with VectorGain, and observing the output using VectorEnd.
    Shows how to chain blocks, monitor signals, and run a time-domain simulation.

Demonstrates:
    1. SinusoidalGenerator  — generate two single-channel sine waves
    2. VectorSum            — sum multiple input signals
    3. VectorGain           — apply scalar gain to summed signal
    4. VectorEnd            — sink/output for simulation
    5. VectorSim            — run simulation using RK4 solver

Block Diagram:

   └── ○ output (VectorEnd)
    └── ○ gain (VectorGain)
        └── ○ sum_two_sin_sources  (VectorSum)
            ├── ○ sin_A5 (SinusoidalGenerator)
            └── ○ sin_A1 (SinusoidalGenerator)

   From Source -> Sink

      [sin_a5 (SinusoidalGenerator)] ──►┐[sum_two_sin_sources (VectorSum)] ──► [gain (VectorGain)] ──► [output (VectorEnd)]
                                        │
      [sin_a2 (SinusoidalGenerator)] ──►┘

Signal Description:
    source1(t) = 5.0 · sin(2π · 15 t)
    source2(t) = 2.0 · sin(2π · 15 t + 1.56)

    sum(t)    = source1(t) + source2(t)
    output(t) = 0.5 · sum(t)

Parameters:
    Amplitude 1 = 5.0
    Amplitude 2 = 2.0
    Frequency   = 15 Hz
    Phase       = 1.56 rad
    Gain        = 0.5
    Time step   = 0.001 s
    Duration    = 0.56 s

Expected Behavior:
    • The summed signal oscillates combining both sine waves.
    • Gain reduces amplitude by half.
    • The scope shows both sources and the scaled output.

Run:
    python example_two_sines_gain.py

"""

import sys
import matplotlib.pyplot as plt
import numpy as np

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

from embedsim.source_blocks import SinusoidalGenerator
from embedsim.processing_blocks import VectorSum, VectorGain
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver

# ============================================================================
# MAIN MENU
# ============================================================================

def plot_results(sim):
    """Plot simulation results """

    # Get time
    t = np.array(sim.scope.t)

    # Get signals
    source1_sig = sim.scope.get_signal("sin_a5", 0)
    source2_sig = sim.scope.get_signal("sin_a2", 0)
    gain_sig = sim.scope.get_signal("gain", 0)

    plt.figure(figsize=(10, 5))

    # Plot each phase with labels and colors
    plt.plot(t, source1_sig, label='Source', color='r', linewidth=1.5)
    plt.plot(t, source2_sig, label='Source2', color='g', linewidth=1.5)
    plt.plot(t, gain_sig, label='Gain', color='b', linewidth=1.5)

    # Add title and labels
    plt.title("Two Sin Addition", fontsize=14, fontweight='bold')
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)

    # Add grid with light style
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Display example header
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Two Sine Sources + Gain")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # CREATE SIGNAL SOURCES
    # ------------------------------------------------------------------------

    # First sine wave: amplitude=5, frequency=15Hz
    source1 = SinusoidalGenerator("sin_a5", amplitude=5.0, freq=15.0, use_c_backend=False)

    # Second sine wave: amplitude=2, frequency=15Hz, phase=1.56 rad
    source2 = SinusoidalGenerator("sin_a2", amplitude=2.0, freq=15.0, phase=1.56, use_c_backend=False)

    # ------------------------------------------------------------------------
    # CREATE PROCESSING BLOCKS
    # ------------------------------------------------------------------------

    # Sum block to add multiple input signals (signs=[1,1] adds both)
    adder = VectorSum("sum_two_sin_sources", signs=[1, 1], use_c_backend=False)

    # Gain block to scale summed signal
    gain = VectorGain("gain", gain=0.5,use_c_backend=False)

    # Output sink
    output = VectorEnd("output", use_c_backend=False)

    # ------------------------------------------------------------------------
    # CONNECT BLOCKS (Signal Flow)
    # ------------------------------------------------------------------------

    # Connect both sources to the adder
    # adder.inputs = [source1, source2]
    source1 >> adder
    source2 >> adder

    # Chain blocks: adder → gain → output
    adder >> gain >> output

    # ------------------------------------------------------------------------
    # SETUP SIMULATION
    # ------------------------------------------------------------------------

    # Create simulation object
    sim = EmbedSim(
        sinks=[output],
        T=0.56,  # Total simulation time in seconds
        dt=0.001,  # Time step
        solver=ODESolver.RK4  # 4th-order Runge-Kutta solver
    )

    # Add signals to the scope for plotting/monitoring
    sim.scope.add(source1, indices=[0], label="sin_a5")
    sim.scope.add(source2, indices=[0], label="sin_a2")
    sim.scope.add(gain, indices=[0], label="gain")

    # Print block connection tree (topology) from sink oriented
    sim.print_topology_tree()
    sim.print_topology_sources2sink()


    # Print block connection tree (topology) from sink oriented
    #sim.print_full_ascii_tree()

    # ------------------------------------------------------------------------
    # RUN SIMULATION
    # ------------------------------------------------------------------------

    print("\nRunning simulation...")
    sim.run()

    print("\nGenerating plot...")
    plot_results(sim)

    print("Complete")
