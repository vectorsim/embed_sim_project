import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

# ---------------------------------------------------------------------------
# Make the local embedsim package importable when running from any directory.
# Adjust this path to match where your embedsim folder lives.
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Get project root by looking for a .project_root_marker file.

    Returns:
        Path object representing the project root
    """
    current_path = Path(__file__).resolve()

    # Search through parent directories for marker file
    for parent in current_path.parents:
        if (parent / '.project_root_marker').exists():
            return parent

    # If no marker found, return current file's parent directory
    return current_path.parent

# Add the project root's embedsim directory to Python path
root_path = get_project_root()
sys.path.insert(0, str(root_path / "embedsim"))

from embedsim.core_blocks import VectorSignal, DEFAULT_DTYPE, validate_inputs_exist
from embedsim.source_blocks import ThreePhaseGenerator
from embedsim.processing_blocks import VectorGain
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver, traverse_blocks_from_sinks_with_loops
from embedsim.code_generator import CodeGenStart, CodeGenEnd, SimBlockBase, MCUTarget




def plot_results(sim):
    """Plot simulation results."""

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


def step_1_run_python_version() -> None:
    """Run the pure Python version first"""
    print("\n" + "=" * 70)
    print("RUNNING PURE PYTHON VERSION")
    print("=" * 70)

    # Create blocks
    generator = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0, use_c_backend=False)
    gain = VectorGain("gain", gain=2.0, use_c_backend=False)  # Amplify by factor 2
    sink = VectorEnd("sink", use_c_backend=False)

    # Connect
    generator >> gain
    gain >> sink

    # Run simulation
    sim = EmbedSim(sinks=[sink], T=0.04, dt=0.0001, solver=ODESolver.RK4)
    sim.scope.add(gain, label="3Phase")
    sim.print_topology_sources2sink()

    print("\nRunning Python simulation...")
    sim.run()
    print("\nPlotting results...")
    plot_results(sim)
    print("\n" + "=" * 70)


def step_2_marked_version() -> None:
    """Create version with code generation markers"""
    print("\n" + "=" * 60)
    print("VERSION WITH CODE GENERATION MARKERS")
    print("=" * 60)

    # Create blocks
    generator = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0)

    # MARK THE REGION FOR CODE GENERATION
    cg_start = CodeGenStart("region_start", use_c_backend=False)  # Input boundary
    gain = VectorGain("gain", gain=2.0)  # Block to be replaced
    cg_end = CodeGenEnd("region_end")  # Output boundary

    sink = VectorEnd("sink")

    # Connect with markers
    generator >> cg_start  # Signal enters region
    cg_start >> gain  # Processing inside region
    gain >> cg_end  # Signal leaves region
    cg_end >> sink  # Continue to sink

    print("\nBlock Diagram:")
    print("  [source] → [cg_start] → [gain] → [cg_end] → [sink]")
    print("              <─── CODE GENERATION REGION ───>")

    # Run one step to determine signal sizes
    sim = EmbedSim(sinks=[sink], T=2.001, dt=0.0001, solver=ODESolver.RK4)
    sim.scope.add(cg_end, label="3Phase")
    sim.run(verbose=False)

    sim.print_topology_sources2sink()

    plot_results(sim)


    print("\nInput signals to region:")
    cg_start.print_signal_info()

    print("\nOutput signals from region:")
    cg_end.print_signal_info()

    # Generate code
    print("\nGenerating C code stubs...")
    files = cg_end.generate_pyx_stub(
        cg_start=cg_start,
        block_name="three_phase_processor",
        output_dir=str(root_path / "code_gen"),
        write_files=True
    )

    print("\nGenerated files:")
    for name, content in files.items():
        print(f"  - {name}: {len(content)} bytes")


def step_3_run_python_version() -> None:
    """Run the pure Python version first"""
    print("\n" + "=" * 70)
    print("RUNNING PURE PYTHON VERSION")
    print("=" * 70)

    sys.path.insert(1, str(root_path / "code_gen"))

    from code_gen.three_phase_processor_simblock import  ThreePhaseProcessorSimBlock

    # Create blocks
    generator = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0, use_c_backend=False)
    gain = ThreePhaseProcessorSimBlock("gain", use_c_backend=True)  # Amplify by factor 2
    sink = VectorEnd("sink", use_c_backend=False)

    # Connect
    generator >> gain
    gain >> sink

    # Run simulation
    sim = EmbedSim(sinks=[sink], T=0.04, dt=0.0001, solver=ODESolver.RK4)
    sim.scope.add(gain, label="3Phase")
    sim.print_topology_sources2sink()

    print("\nRunning Python simulation...")
    sim.run()
    print("\nPlotting results...")
    plot_results(sim)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Display example header
    print("\n" + "=" * 70)
    print("Simple Gain Code Generator")
    print("=" * 70)

    # Step1 : Run simulation in Python
    #step_1_run_python_version()

    # Step2 : Mark
    #step_2_marked_version()

    step_3_run_python_version()