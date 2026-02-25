"""
example_validate_02.py
======================
ControlForge — Example: Signal Summation, Gain, and Integration

Demonstrates:
  1. VectorConstant       — DC offset source
  2. SinusoidalGenerator  — single-channel sine / cosine source
  3. VectorSum            — sum multiple signals with configurable signs
  4. VectorGain           — scalar gain applied to a vector signal
  5. VectorIntegrator     — continuous integrator  x_dot = u
  6. EnhancedVectorSim    — simulation engine with RK4 solver

Block diagram:

└── ○ output (VectorEnd)
    └── ⚡ integrator (VectorIntegrator)
        └── ○ gain (VectorGain)
            └── ○ source_sum (VectorSum)
                ├── ○ cosine_source (SinusoidalGenerator)
                ├── ○ sin_source (SinusoidalGenerator)
                └── ○ const_1 (VectorConstant)


 [cosine_source (SinusoidalGenerator)] ──►┐[source_sum (VectorSum)] ──► [gain (VectorGain)] ──► [⚡integrator (VectorIntegrator)]  ──► [output (VectorEnd)]
                                           │
  [sin_source (SinusoidalGenerator)] ─────►┤
                                           │
  [const_1 (VectorConstant)] ─────────────►┘


Mathematical description:
  source_sum(t) = cos(2π·f·t) + sin(2π·f·t) + 1     f = 20 Hz
               = √2 · sin(2π·f·t + π/4)  +  1       (combined AC + DC)

  gain_out(t)  = gain · source_sum(t)

  integrator(t)= ∫₀ᵗ gain_out(τ) dτ
               = gain · [ (1 - cos(2π·f·t) + sin(2π·f·t)) / (2π·f)  +  t ]


Run:
    python simple_signal_addition.py
"""

# ---------------------------------------------------------------------------
# Make the local embedsim package importable when running from any directory.
# Adjust this path to match where your embedsim folder lives.
# ---------------------------------------------------------------------------
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Lambda to add N-levels parent to sys.path with print
add_parent_to_syspath = lambda levels=2: (
    print(f"[EmbedSim] Adding to sys.path: {(p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]}"),
    sys.path.insert(0, str((p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]))
)[-1]
add_parent_to_syspath(2)

from embedsim.core_blocks                import VectorSignal
from embedsim.source_blocks              import VectorConstant, SinusoidalGenerator
from embedsim.processing_blocks          import VectorGain, VectorSum
from embedsim.dynamic_blocks             import VectorEnd, VectorIntegrator
from embedsim.simulation_engine          import EmbedSim, ODESolver


# =============================================================================
# Simulation parameters
# =============================================================================
FREQ   =  5.0       # Hz   frequency of sin / cos sources
AMP    =  1.0       # –    amplitude of each sinusoidal source
GAIN   =  3.0       # –    scalar gain before integrator
T_SIM  =  4.0       # -    total simulation time in s
DT     =  0.01      # -    time step (RK4 solver) in s
C      =  1.3       # -    Constant (DC Value)

# =============================================================================
# Plot function
# =============================================================================
def plot_results(sim):

    t = np.array(sim.scope.t)
    # Signals
    source_const = sim.scope.get_signal("source_const", index=0)
    source_sin   = sim.scope.get_signal("source_sin",   index=0)
    source_cos   = sim.scope.get_signal("source_cos",   index=0)
    integrator   = sim.scope.get_signal("integrator",   index=0)

    plt.figure(figsize=(15, 10))

    # Plot each phase with labels and colors
    plt.plot(t, source_sin,  label='Sin', color='r', linewidth=1.5)
    plt.plot(t, source_cos,  label='Cos', color='g', linewidth=1.5)
    plt.plot(t, source_const, label='Const', color='y', linewidth=1.5)
    plt.plot(t, integrator, label='integreator', color='b', linewidth=2.5)


    # Add title and labels
    plt.title("Simple Signal Addition", fontsize=14, fontweight='bold')
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Amplitude [V]", fontsize=12)

    # Add grid with light style
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend
    plt.legend(loc='upper right')


    plt.tight_layout()
    plt.show()




# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Sine + Cosine + DC  →  Sum  →  Gain  →  Integrator")
    print("=" * 70)

    # ── Sources ───────────────────────────────────────────────────────────────
    # VectorConstant: outputs [1.0] at every time step
    source_const = VectorConstant("const_1", [C], use_c_backend=False)

    # SinusoidalGenerator(name, amplitude, freq_hz, phase_rad)
    # source_sin: A·sin(2π·f·t)   — zero phase
    # source_cos: A·sin(2π·f·t + π/2) = A·cos(2π·f·t)  — 90° phase shift
    source_sin = SinusoidalGenerator("sin_source",   AMP, FREQ, 0.0, use_c_backend=False)
    source_cos = SinusoidalGenerator("cosine_source", AMP, FREQ, np.pi / 2.0)

    # ── Summation ─────────────────────────────────────────────────────────────
    # VectorSum with signs=[+1,+1,+1]: output = cos + sin + const
    # Note: inputs are assigned directly here rather than using >> operator
    # because all three feed into the same sum block simultaneously.
    source_sum = VectorSum("source_sum", [1, 1, 1], use_c_backend=False)
    source_sum.inputs = [source_cos, source_sin, source_const]

    # ── Processing chain ──────────────────────────────────────────────────────
    # VectorGain: scales the sum by 0.005 before integration
    # VectorIntegrator: x_dot = u,  x(0) = 0.0
    # VectorEnd: sink — records the final signal for scope
    gain       = VectorGain("gain",       gain=GAIN, use_c_backend=False)
    integrator = VectorIntegrator("integrator", initial_state=[0.0])
    output     = VectorEnd("output")

    # Signal flow: sum → gain → integrator → output (sink)
    source_sum >> gain >> integrator >> output

    # ── Simulation engine ─────────────────────────────────────────────────────
    # RK4 solver for accurate integration of the continuous integrator state.
    # dt=0.1 s gives 100 steps over 10 s.
    sim = EmbedSim(
        sinks  = [output],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.RK4,
    )

    # ── Scope — which signals to record ──────────────────────────────────────
    # indices=[0] records element 0 of the block's output vector.
    # label is the key used later in sim.scope.get_signal(label, index).
    sim.scope.add(source_const, indices=[0], label="source_const")
    sim.scope.add(source_sin,   indices=[0], label="source_sin")
    sim.scope.add(source_cos,   indices=[0], label="source_cos")
    sim.scope.add(integrator,   indices=[0], label="integrator")

    # ── Print topology ────────────────────────────────────────────────────────
    print("\nBlock diagram topology:")
    sim.print_topology_tree()
    sim.print_topology_sources2sink()

    # ── Run ───────────────────────────────────────────────────────────────────
    print("\nRunning simulation...")
    sim.run()
    print(f"  Completed: {len(sim.scope.t)} time steps recorded")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\nGenerating plot...")
    plot_results(sim)

    print("Complete")