"""
fs_clarke_park_codegen.py
=========================

Simple test: 3-phase sine-wave generator → Clarke → Park transformation.
Theta is supplied as a VectorConstant (fixed electrical angle).

Runs a short simulation then triggers EmbedSim code generation,
emitting embedsim_loop.c / embedsim_loop.h into embedsim_gen/.

Layout
------
  Va(t) = sin(ωt)
  Vb(t) = sin(ωt - 2π/3)
  Vc(t) = sin(ωt + 2π/3)
  θ     = constant (e.g. 0.5 rad)

  [Va, Vb, Vc] ──► ClarkeBlock ──► [Vα, Vβ]
                                         │
  [θ] ─────────────────────────────────► ParkBlock ──► [Vd, Vq]
"""

import sys
import math
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
from _path_utils import get_embedsim_import_path
sys.path.insert(0, get_embedsim_import_path())

# ── EmbedSim imports ──────────────────────────────────────────────────────────
from embedsim.simulation_engine import SimulationEngine
from embedsim.vector_block     import VectorConstant
from embedsim.script_block     import ScriptBlock          # used for sine gen

# Coordinate transform blocks
from fs_electrical_machines.electrical_blocks.clarke_block import ClarkeBlock
from fs_electrical_machines.electrical_blocks.park_block   import ParkBlock

# ── Parameters ────────────────────────────────────────────────────────────────
F_ELEC   = 50.0          # electrical frequency [Hz]
OMEGA    = 2 * math.pi * F_ELEC
V_PEAK   = 1.0           # normalised peak amplitude
THETA    = 0.5           # fixed electrical angle [rad]

T_END    = 0.04          # 2 full electrical cycles
DT       = 1e-5          # 10 µs step

# ── Build simulation ──────────────────────────────────────────────────────────
eng = SimulationEngine(dt=DT, t_end=T_END)

# 3-phase sine source  (ScriptBlock: outputs [Va, Vb, Vc])
def three_phase_sine(t, inputs):
    va = V_PEAK * math.sin(OMEGA * t)
    vb = V_PEAK * math.sin(OMEGA * t - 2 * math.pi / 3)
    vc = V_PEAK * math.sin(OMEGA * t + 2 * math.pi / 3)
    return np.array([va, vb, vc], dtype=np.float32)

sine_src = ScriptBlock(
    name        = "ThreePhaseSine",
    num_inputs  = 0,
    output_size = 3,
    func        = three_phase_sine,
)

# Fixed electrical angle
theta_src = VectorConstant(
    name  = "Theta",
    value = np.array([THETA], dtype=np.float32),
)

# Clarke transform:  [Va, Vb, Vc] → [Vα, Vβ]
clarke = ClarkeBlock(name="Clarke")

# Park transform:    [Vα, Vβ, θ] → [Vd, Vq]
park = ParkBlock(name="Park")

# ── Wiring ────────────────────────────────────────────────────────────────────
#   sine_src[0:3] → Clarke inputs
sine_src >> clarke

#   Clarke[0:2] → Park inputs 0,1
#   theta_src[0] → Park input 2
clarke  >> park          # Vα, Vβ
theta_src >> park        # θ  (port 2)

# Register blocks
eng.add_blocks([sine_src, theta_src, clarke, park])

# ── Run simulation ────────────────────────────────────────────────────────────
print("=" * 60)
print(" Running Clarke-Park simulation")
print(f"  f_elec = {F_ELEC} Hz,  θ = {THETA:.3f} rad,  T = {T_END*1e3:.1f} ms")
print("=" * 60)

eng.run()

# ── Print final-step results ──────────────────────────────────────────────────
t_final = eng.time_log[-1]
va_f = V_PEAK * math.sin(OMEGA * t_final)
vb_f = V_PEAK * math.sin(OMEGA * t_final - 2 * math.pi / 3)
vc_f = V_PEAK * math.sin(OMEGA * t_final + 2 * math.pi / 3)

v_alpha = clarke.output_signal.value[0]
v_beta  = clarke.output_signal.value[1]
vd      = park.output_signal.value[0]
vq      = park.output_signal.value[1]

print(f"\nFinal step  t = {t_final*1e3:.3f} ms")
print(f"  Va={va_f:+.4f}  Vb={vb_f:+.4f}  Vc={vc_f:+.4f}")
print(f"  Vα={v_alpha:+.4f}  Vβ={v_beta:+.4f}")
print(f"  Vd={vd:+.4f}  Vq={vq:+.4f}")

# ── Code generation ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(" Running code generation  →  embedsim_gen/")
print("=" * 60)

eng.generate_loop()

print("\n[DONE] embedsim_loop.c and embedsim_loop.h written to embedsim_gen/")
