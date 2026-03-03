#!/usr/bin/env python3
"""
Three-Phase Signal Processing with Code Generation Demo
=======================================================
This example demonstrates:
1. Creating a three-phase signal processing chain in Python
2. Marking a region for C code generation
3. Generating C code from the marked region
4. Switching between Python and C backends

The processing chain:
    ThreePhaseGenerator → Gain → Sink
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for embedsim import
add_parent_to_syspath = lambda levels=2: (
    print(f"[EmbedSim] Adding to sys.path: {(p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]}"),
    sys.path.insert(0, str((p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]))
)[-1]
add_parent_to_syspath(2)

from embedsim.source_blocks import ThreePhaseGenerator
from embedsim.processing_blocks import VectorGain
from embedsim.dynamic_blocks import VectorEnd
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.code_generator import CodeGenStart, CodeGenEnd, SimBlockBase, MCUTarget


def run_python_version():
    """Run the pure Python version first"""
    print("\n" + "="*70)
    print("RUNNING PURE PYTHON VERSION")
    print("="*70)
    
    # Create blocks
    generator = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0)
    gain = VectorGain("gain", gain=2.0)  # Amplify by factor 2
    sink = VectorEnd("sink")
    
    # Connect
    generator >> gain
    gain >> sink
    
    # Run simulation
    sim = EmbedSim(sinks=[sink], T=0.04, dt=0.0001, solver=ODESolver.RK4)
    sim.scope.add(generator, label="original")
    sim.scope.add(gain, label="amplified")
    
    print("\nRunning Python simulation...")
    sim.run(verbose=True)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    t = np.array(sim.scope.t)
    orig = sim.scope.get_signal("original", 0)
    amplified = sim.scope.get_signal("amplified", 0)
    
    plt.plot(t, orig, 'b-', label="Original (Python)", alpha=0.7)
    plt.plot(t, amplified, 'r--', label="Amplified (Python)", alpha=0.7)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Python Version: Three-Phase Signal Processing")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.04)
    plt.savefig("python_version.png")
    plt.show()
    
    return sim


def create_codegen_version():
    """Create version with code generation markers"""
    print("\n" + "="*70)
    print("CREATING VERSION WITH CODE GENERATION MARKERS")
    print("="*70)
    
    # Create blocks
    generator = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0)
    
    # Mark the beginning of code generation region
    cg_start = CodeGenStart("region_start")
    
    # This block will be replaced by C code
    gain = VectorGain("gain", gain=2.0)
    
    # Mark the end of code generation region
    cg_end = CodeGenEnd("region_end")
    
    sink = VectorEnd("sink")
    
    # Connect with code generation markers
    generator >> cg_start    # Input to region
    cg_start >> gain         # Inside region
    gain >> cg_end           # Output from region
    cg_end >> sink           # Continue to sink
    
    print("\nBlock diagram with code generation region:")
    print("  [source] → [cg_start] → [gain] → [cg_end] → [sink]")
    print("              <───── code generation region ─────>")
    
    # Run once to determine signal sizes
    sim = EmbedSim(sinks=[sink], T=0.001, dt=0.0001, solver=ODESolver.RK4)
    sim.scope.add(generator, label="original")
    sim.scope.add(gain, label="amplified")
    
    print("\nRunning one step to determine signal sizes...")
    sim.run(verbose=False)
    
    return sim, cg_start, cg_end, generator, gain, sink


def generate_c_code(sim, cg_start, cg_end):
    """Generate C code from the marked region"""
    print("\n" + "="*70)
    print("GENERATING C CODE")
    print("="*70)
    
    # Generate C code
    files = cg_end.generate_pyx_stub(
        cg_start=cg_start,
        block_name="three_phase_processor",
        output_dir="./generated_code",
        write_files=True
    )
    
    print("\nGenerated files:")
    for name, content in files.items():
        print(f"  - {name}: {len(content)} bytes")
    
    return files


def create_c_implementation():
    """Create the C implementation file"""
    print("\n" + "="*70)
    print("CREATING C IMPLEMENTATION")
    print("="*70)
    
    c_code = """/*
 * three_phase_processor.c
 * C implementation of three-phase signal processing
 * 
 * Input:  source signal (3 doubles)
 * Output: amplified signal (3 doubles)
 */

#include "three_phase_processor.h"
#include <math.h>

void three_phase_processor_compute(const InputSignals* in, OutputSignals* out) {
    // Simple gain of 2.0 on all three phases
    out->source[0] = in->source[0] * 2.0;
    out->source[1] = in->source[1] * 2.0;
    out->source[2] = in->source[2] * 2.0;
    
    // You could add more complex processing here:
    // - Filtering
    // - Clarke/Park transforms
    // - PI control
    // - Harmonic compensation
}
"""
    
    with open("./generated_code/three_phase_processor.c", "w") as f:
        f.write(c_code)
    
    print("✓ Created C implementation: ./generated_code/three_phase_processor.c")
    return c_code


def create_setup_script():
    """Create setup script for compilation"""
    print("\n" + "="*70)
    print("CREATING SETUP SCRIPT")
    print("="*70)
    
    setup_code = """# setup_three_phase_processor.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Compiler flags
if sys.platform == 'win32':
    compile_args = ['/O2', '/openmp']
else:
    compile_args = ['-O3', '-ffast-math', '-fopenmp']

ext = Extension(
    name='three_phase_processor_wrapper',
    sources=[
        'three_phase_processor_wrapper.pyx',
        'three_phase_processor.c',
    ],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=['-fopenmp'] if sys.platform != 'win32' else [],
)

setup(
    name='three_phase_processor_wrapper',
    ext_modules=cythonize(
        [ext],
        compiler_directives={
            'language_level': '3',
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        },
        annotate=True,
    ),
)
"""
    
    with open("./generated_code/setup_three_phase_processor.py", "w") as f:
        f.write(setup_code)
    
    print("✓ Created setup script: ./generated_code/setup_three_phase_processor.py")


def compile_and_test():
    """Compile and test the generated code"""
    print("\n" + "="*70)
    print("COMPILING AND TESTING C VERSION")
    print("="*70)
    
    import subprocess
    import os
    
    # Change to generated_code directory
    os.chdir("./generated_code")
    
    # Compile the Cython wrapper
    print("\nCompiling Cython wrapper...")
    result = subprocess.run(
        [sys.executable, "setup_three_phase_processor.py", "build_ext", "--inplace"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print("Compilation failed:")
        print(result.stderr)
        os.chdir("..")
        return None
    
    print("✓ Compilation successful!")
    
    # Go back to original directory
    os.chdir("..")
    
    # Test the compiled version
    print("\n" + "="*70)
    print("TESTING C VERSION")
    print("="*70)
    
    from generated_code.three_phase_processor_simblock import ThreePhaseProcessorSimBlock
    
    # Create blocks
    generator = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0)
    
    # Use the generated SimBlock with C backend
    processor = ThreePhaseProcessorSimBlock("processor", use_c_backend=True)
    
    sink = VectorEnd("sink")
    
    # Connect
    generator >> processor
    processor >> sink
    
    # Run simulation
    sim = EmbedSim(sinks=[sink], T=0.04, dt=0.0001, solver=ODESolver.RK4)
    sim.scope.add(generator, label="original")
    sim.scope.add(processor, label="amplified_c")
    
    print("\nRunning C version simulation...")
    sim.run(verbose=True)
    
    # Plot comparison
    plt.figure(figsize=(12, 10))
    
    t = np.array(sim.scope.t)
    orig = sim.scope.get_signal("original", 0)
    amplified_c = sim.scope.get_signal("amplified_c", 0)
    
    # Also run Python version for comparison
    sim_py = EmbedSim(sinks=[sink], T=0.04, dt=0.0001, solver=ODESolver.RK4)
    sim_py.scope.add(generator, label="original")
    
    # Python version with regular gain
    gen_py = ThreePhaseGenerator("source", amplitude=10.0, freq=50.0)
    gain_py = VectorGain("gain", gain=2.0)
    sink_py = VectorEnd("sink")
    gen_py >> gain_py >> sink_py
    
    sim_py = EmbedSim(sinks=[sink_py], T=0.04, dt=0.0001, solver=ODESolver.RK4)
    sim_py.scope.add(gen_py, label="original")
    sim_py.scope.add(gain_py, label="amplified_py")
    sim_py.run(verbose=False)
    
    t_py = np.array(sim_py.scope.t)
    amplified_py = sim_py.scope.get_signal("amplified_py", 0)
    
    # Plot comparison
    plt.subplot(2, 1, 1)
    plt.plot(t, orig, 'b-', label="Original", alpha=0.7)
    plt.plot(t, amplified_c, 'r--', label="Amplified (C)", linewidth=2)
    plt.plot(t_py, amplified_py, 'g:', label="Amplified (Python)", alpha=0.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("C vs Python Implementation Comparison")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.04)
    
    # Plot difference
    plt.subplot(2, 1, 2)
    # Interpolate Python data to match C time points
    from scipy import interpolate
    f = interpolate.interp1d(t_py, amplified_py, kind='linear', 
                             bounds_error=False, fill_value=0)
    amplified_py_interp = f(t)
    
    diff = amplified_c - amplified_py_interp
    plt.plot(t, diff, 'purple', label="Difference (C - Python)")
    plt.xlabel("Time [s]")
    plt.ylabel("Difference")
    plt.title("Numerical Difference (should be near zero)")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.04)
    
    plt.tight_layout()
    plt.savefig("c_vs_python_comparison.png")
    plt.show()
    
    print(f"\nMax difference: {np.max(np.abs(diff)):.2e}")
    if np.max(np.abs(diff)) < 1e-10:
        print("✓ C implementation matches Python exactly!")
    else:
        print("⚠ C implementation differs from Python")
    
    return sim


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("THREE-PHASE SIGNAL PROCESSING WITH CODE GENERATION")
    print("="*70)
    
    # Step 1: Run Python version
    python_sim = run_python_version()
    
    # Step 2: Create version with code generation markers
    sim, cg_start, cg_end, generator, gain, sink = create_codegen_version()
    
    # Step 3: Generate C code
    files = generate_c_code(sim, cg_start, cg_end)
    
    # Step 4: Create C implementation
    c_code = create_c_implementation()
    
    # Step 5: Create setup script
    create_setup_script()
    
    # Step 6: Compile and test
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
To compile and test the C version:

1. Navigate to the generated_code directory:
   cd generated_code

2. Compile the Cython wrapper:
   python setup_three_phase_processor.py build_ext --inplace

3. Run the test script:
   python -c "from three_phase_processor_simblock import ThreePhaseProcessorSimBlock; print('✓ Module loaded successfully')"

4. The C version will be used automatically when use_c_backend=True
    """)
    
    # Ask user if they want to compile now
    response = input("\nDo you want to compile and test the C version now? (y/n): ")
    if response.lower() == 'y':
        compile_and_test()
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print("""
Key takeaways:
1. Python version runs first for development
2. CodeGenStart/End mark the region for C translation
3. generate_pyx_stub() creates all necessary files
4. C implementation must match the expected signature
5. Switch backends with use_c_backend=True
    """)


if __name__ == "__main__":
    main()