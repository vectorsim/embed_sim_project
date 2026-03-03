import sys
import matplotlib.pyplot as plt
import numpy as np
# ------------------------------------------------------------------
# Locate embedsim regardless of working directory
# ------------------------------------------------------------------
from _path_utils import get_embedsim_import_path

sys.path.insert(0, get_embedsim_import_path())

from embedsim.source_blocks import ThreePhaseGenerator
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver

from coordinate_transform_blocks import ClarkeTransformBlock
from embedsim.plot_helper import create_plotter


def main():
    print("\n" + "=" * 70)
    print("Clarke Transform Test with PlotHelper")
    print("=" * 70)

    # Create a 50 Hz three-phase generator
    generator = ThreePhaseGenerator("3phase_gen",
                                    amplitude=10.0,
                                    freq=50.0,
                                    use_c_backend=False)

    # Create Clarke transform block
    try:
        ct = ClarkeTransformBlock(name="clark_transform_block", use_c_backend=True)
        print("✅ Using C backend for Clarke transform")
    except ImportError:
        print("⚠️ C backend not available, falling back to Python")
        ct = ClarkeTransformBlock(name="clark_transform_block", use_c_backend=False)

    # Create ONE sink for Clarke output
    output = VectorEnd("output")

    # Connect blocks: generator → ct → output
    generator >> ct >> output

    # Create simulation with ONE sink
    sim = EmbedSim(sinks=[output],
                   T=0.04,  # 40 ms (2 cycles)
                   dt=0.0001,
                   solver=ODESolver.RK4)

    # Print topology
    print("\n📊 Block Topology:")
    sim.print_topology_tree()
    sim.print_topology_sources2sink()

    # Add signals to scope for monitoring
    sim.scope.add(generator, label="3Phase Input")
    sim.scope.add(ct, label="Clarke Output")

    # Run simulation
    print("\n⚙️ Running simulation...")
    sim.run()
    print("✅ Simulation complete!")

    # Check if we have data
    if len(sim.scope.t) == 0:
        print("❌ No data collected! Check scope configuration.")
        return

    print(f"\n📊 Data collected: {len(sim.scope.t)} samples over {sim.scope.t[-1] * 1000:.1f} ms")

    # ======================================================
    # USE THE PLOTHELPER CLASS
    # ======================================================

    # Create plotter
    plotter = create_plotter(sim)

    # 1. Show information about recorded signals
    plotter.info()

    # 2. List all available signals
    plotter.list_signals()

    # 3. Quick plot of all signals
    print("\n📈 Plotting all signals...")
    plotter.easyplot(title="All Signals - Clarke Transform Test")

    # # 4. Plot Clarke output (both components)
    # print("\n📈 Plotting Clarke Output...")
    # plotter.easyplot('Clarke Output', title="Clarke Transform Output (αβ)")

    # # 5. Plot specific components
    # print("\n📈 Plotting Alpha component only...")
    # plotter.easyplot('Clarke Output[0]', title="Alpha Component")

    # print("\n📈 Plotting Beta component only...")
    # plotter.easyplot('Clarke Output[1]', title="Beta Component")

    # # 6. Plot all three phases in separate subplots
    # print("\n📈 Plotting three-phase components...")
    # plotter.plot_components('3Phase Input', title="Three-Phase Input Components")

    # # 7. Compare Phase A vs Alpha
    # print("\n📈 Comparing Phase A and Alpha...")
    # plotter.compare('3Phase Input[0]', 'Clarke Output[0]',
                    # titles=['Phase A (Input)', 'Alpha (Output)'],
                    # colors=['blue', 'red'])

    # # 8. XY plot (Alpha vs Beta) - should show a circle
    # print("\n📈 Creating XY plot (Lissajous figure)...")
    # plotter.xy_plot('Clarke Output[0]', 'Clarke Output[1]',
                    # title='Alpha vs Beta (Should be a circle)',
                    # color='purple')

    # # 9. FFT analysis of Alpha component
    # print("\n📈 FFT analysis of Alpha component...")
    # plotter.fft_plot('Clarke Output[0]', max_freq=200,
                     # title='Frequency Spectrum of Alpha')

    # # 10. Zoomed view of first cycle
    # print("\n📈 Zoomed view (first cycle)...")
    # plotter.easyplot(['3Phase Input[0]', 'Clarke Output[0]'],
                     # time_range=(0, 0.02),  # First cycle only
                     # title='First Cycle - Phase A and Alpha',
                     # save_path='first_cycle.png')

    # 11. Save all plots to files
    print("\n💾 Saving all plots...")
    #plotter.save_all_plots(prefix='clarke_test', format='png')

    print("\n" + "=" * 70)
    print("✅ All plots generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()