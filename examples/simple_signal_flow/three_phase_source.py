#!/usr/bin/env python3
"""
 Basic Three-Phase Signal Generation
==============================================
Demonstrates the simplest possible simulation with balanced three-phase signals.

Key Concepts:
- Creating source blocks (ThreePhaseGenerator)
- Creating sink blocks (VectorEnd)
- Connecting blocks with >> operator
- Basic simulation setup
- Signal monitoring and plotting

Author: Vector Simulation Framework
"""


# ---------------------------------------------------------------------------
# Make the local embedsim package importable when running from any directory.
# Adjust this path to match where your embedsim folder lives.

import sys
import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------------------------------------
from pathlib import Path
# Lambda to add N-levels parent to sys.path with print
add_parent_to_syspath = lambda levels=2: (
    print(f"[EmbedSim] Adding to sys.path: {(p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]}"),
    sys.path.insert(0, str((p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]))
)[-1]
add_parent_to_syspath(2)

from embedsim.source_blocks import *
from embedsim.dynamic_blocks import *
from embedsim.simulation_engine import *


def plot_results(sim):
    """Plot simulation results with improved clarity."""

    # Get time
    t = np.array(sim.scope.t)

    # Get signals
    phase_u = sim.scope.get_signal("3Phase", 0)
    phase_v = sim.scope.get_signal("3Phase", 1)
    phase_w = sim.scope.get_signal("3Phase", 2)

    plt.figure(figsize=(10, 5))

    # Plot each phase with labels and colors
    plt.plot(t, phase_u, label='Phase U', color='r', linewidth=1.5)
    plt.plot(t, phase_v, label='Phase V', color='g', linewidth=1.5)
    plt.plot(t, phase_w, label='Phase W', color='b', linewidth=1.5)

    # Add title and labels
    plt.title("Three-Phase Signal Generation", fontsize=14, fontweight='bold')
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Amplitude [V]", fontsize=12)

    # Add grid with light style
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='upper right')

    # Optional: limit x-axis to first cycle for clarity
    plt.xlim(0, 0.02)  # first 20 ms = 1 cycle at 50 Hz

    plt.tight_layout()
    plt.show()


def main():
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Three-Phase Signal Generation")
    print("="*70)
    
    # Create a 50 Hz three-phase generator
    generator = ThreePhaseGenerator("3phase_gen", 
                                   amplitude=10.0,  # Peak amplitude: 10
                                   freq=50.0,       # Frequency: 50 Hz
                                   use_c_backend=False)
    
    # Create sink to record output
    output = VectorEnd("output")
    
    # Connect blocks using >> operator
    generator >> output
    
    # Create simulation
    sim = EmbedSim(sinks=[output],
                            T=0.04,          # Total time: 40 ms (2 cycles)
                            dt=0.0001,       # Time step: 0.1 ms
                            solver=ODESolver.RK4)

    # Show block topology
    sim.print_topology()

    # Represent as Tree
    sim.print_topology_sources2sink()
    
    # Add all three phases to scope for monitoring
    sim.scope.add(generator, label="3Phase")
    
    # Run simulation
    sim.run()
    # Plot the signals
    plot_results(sim)


if __name__ == "__main__":
    main()
