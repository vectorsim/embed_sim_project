"""
CENTER DYNAMICS - COMPLETE WORKING EXAMPLE
===========================================
System: ẋ = A·x,  A = [[0, -2], [2, 0]]
Eigenvalues: λ = ±2i → Center (closed orbits)
"""

import sys
from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

from embedsim.dynamic_blocks import VectorEnd, StateSpaceBlock
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.plot_helper import create_plotter
import numpy as np

# ── 1. System Definition ──────────────────────────────────────
#
#   ẋ = A·x  where:
#   A = [[0, -2],
#        [2,  0]]
#
#   Eigenvalues: λ = ±2i → Center (purely imaginary)
#   Solution: x₁(t) = cos(2t), x₂(t) = sin(2t)
#   Orbits: Closed ellipses around origin
#
A = np.array([[0, -2],
              [2,  0]])

# B = 0 (no input) - autonomous system
B = np.zeros((2, 1))   # 2 states, 1 input (but input is zero)
C = np.eye(2)          # Output both states: y = [x₁, x₂]
D = np.zeros((2, 1))   # No direct feedthrough

# ── 2. Build Blocks ────────────────────────────────────────────
#
#   StateSpaceBlock "center" ──► VectorEnd "output"
#   (No input block needed since B=0)
#
system = StateSpaceBlock(
    "center",
    A, B, C, D,
    initial_state=[1.0, 0.0]   # x(0) = [1, 0]
)

# ONE sink records BOTH states as a vector
sink = VectorEnd("output")
system >> sink

# ── 3. Simulate ─────────────────────────────────────────────────
#
#   T = π ≈ 3.14159... (one full period)
#   dt = 0.001 s (1 ms, 1000 Hz)
#   RK4 solver for high accuracy (error ~5e-6)
#
sim = EmbedSim(
    sinks=[sink],
    T=np.pi,           # Use exact π for better accuracy!
    dt=0.001,
    solver=ODESolver.RK4
)

# Record both states
sim.scope.add(system, label="states")

print("\n" + "="*70)
print("CENTER DYNAMICS - Simulation Starting")
print("="*70)
print(f"  System: ẋ = A·x")
print(f"  A = {A}")
print(f"  x(0) = [1.0, 0.0]")
print(f"  Expected: x₁(t) = cos(2t), x₂(t) = sin(2t)")
print(f"  Period: π = {np.pi:.6f} s")
print("="*70 + "\n")

sim.run()

# ── 4. Topology Views ──────────────────────────────────────────

print("\n" + "="*70)
print("1. sim.print_topology() - EXECUTION ORDER")
print("="*70)
sim.print_topology()

print("\n" + "="*70)
print("2. sim.topo.print_console() - SIGNAL FLOW DIAGRAM")
print("="*70)
sim.topo.print_console()


plotter = create_plotter(sim)
# Quick plot of both states
plotter.easyplot(
    title="Center Dynamics - States x₁(t) and x₂(t)",
    figsize=(12, 5)
)

# Phase portrait (x₁ vs x₂) - shows the closed orbit!
plotter.xy_plot(
    'states[0]',
    'states[1]',
    title='Phase Portrait - Center Dynamics (Closed Orbit)',
    color='purple',
    equal_axes=True
)


print("✅ Done! All outputs generated successfully.")