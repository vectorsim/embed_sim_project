"""
fs_clarke_park_codegen.py
=========================

3-phase sine source -> Clarke -> Park transformation.

Wiring for codegen:
  ThreePhaseSine --> cg_start --> Clarke --> Park --> cg_end --> Sink
                                               ^
                                           Theta  (VectorConstant, outside
                                                   codegen region so it is
                                                   not emitted as a block —
                                                   its value becomes the
                                                   THETA_E #define in the
                                                   generated C header)
"""

import sys
import math
import numpy as np
from pathlib import Path

# -- Path setup ----------------------------------------------------------------
from _path_utils import get_project_root, get_embedsim_import_path

_root = get_project_root()

for _p in (
    get_embedsim_import_path(),
    str(_root / "fs_electrical_machines"),
    str(_root / "fs_electrical_machines" / "c_src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# -- EmbedSim imports ----------------------------------------------------------
from embedsim.simulation_engine import EmbedSim, ODESolver
from embedsim.source_blocks     import VectorConstant, ThreePhaseGenerator
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.plot_helper       import create_plotter
from embedsim.code_generator    import CodeGenStart, CodeGenEnd, StepGenerator

# -- Coordinate transform blocks -----------------------------------------------
from coordinate_transform_blocks import (
    ClarkeTransformBlock as ClarkeBlock,
    ParkTransformBlock   as ParkBlock,
)

# -- Parameters ----------------------------------------------------------------
F_ELEC = 50.0
V_PEAK = 1.0
THETA  = 0.5            # fixed electrical angle [rad]

T_END  = 0.04
DT     = 1e-5

# -- Build blocks --------------------------------------------------------------
sine_src  = ThreePhaseGenerator(name="ThreePhaseSine", amplitude=V_PEAK, freq=F_ELEC)

# Theta is outside the codegen region — it feeds Park at simulation time
# but in generated C it is represented by the THETA_E #define below.
theta_src = VectorConstant(name="Theta",
                           value=np.array([THETA], dtype=np.float32))

clarke    = ClarkeBlock(name="Clarke")
park      = ParkBlock(name="Park")

cg_start  = CodeGenStart(name="cg_start")
cg_end    = CodeGenEnd(name="cg_end")
sink      = VectorEnd(name="Sink")

# -- Wiring --------------------------------------------------------------------
#
#  ThreePhaseSine --> cg_start --> Clarke --> Park --> cg_end --> Sink
#                                              ^
#                                           Theta   (outside region)
#
sine_src  >> cg_start
cg_start  >> clarke
clarke    >> park
theta_src >> park          # Theta feeds Park but is outside the codegen region
park      >> cg_end
cg_end    >> sink

# -- Build simulation ----------------------------------------------------------
eng = EmbedSim(sinks=[sink], T=T_END, dt=DT, solver=ODESolver.RK4)

# -- Topology ------------------------------------------------------------------
eng.topo.print_console()

# -- Register signals for recording --------------------------------------------
eng.scope.add(sine_src, label="3phase")
eng.scope.add(clarke,   label="clarke")
eng.scope.add(park,     label="park")

# -- Run -----------------------------------------------------------------------
print("=" * 60)
print(" Running Clarke-Park simulation")
print(f"  f_elec = {F_ELEC} Hz,  theta = {THETA:.3f} rad,  T = {T_END*1e3:.1f} ms")
print("=" * 60)

eng.run()

# -- Print final-step results --------------------------------------------------
t_final = eng.scope.t[-1]
v_alpha = float(clarke.output.value[0])
v_beta  = float(clarke.output.value[1])
vd      = float(park.output.value[0])
vq      = float(park.output.value[1])
omega   = 2 * math.pi * F_ELEC
va_f    = V_PEAK * math.sin(omega * t_final)
vb_f    = V_PEAK * math.sin(omega * t_final - 2 * math.pi / 3)
vc_f    = V_PEAK * math.sin(omega * t_final + 2 * math.pi / 3)

print(f"\nFinal step  t = {t_final*1e3:.3f} ms")
print(f"  Va={va_f:+.4f}  Vb={vb_f:+.4f}  Vc={vc_f:+.4f}")
print(f"  Valpha={v_alpha:+.4f}  Vbeta={v_beta:+.4f}")
print(f"  Vd={vd:+.4f}  Vq={vq:+.4f}")

# -- Plot ----------------------------------------------------------------------
ph = create_plotter(eng)

ph.plot_grid(
    rows=[
        dict(signal="3phase[0]", ylabel="Voltage (V)",
             title="Phase Va",  color="#58a6ff"),
        dict(signal="3phase[1]", ylabel="Voltage (V)",
             title="Phase Vb",  color="#3fb950"),
        dict(signal="3phase[2]", ylabel="Voltage (V)",
             title="Phase Vc",  color="#d2a8ff"),
        dict(signal="clarke[0]", ylabel="Valpha (V)",
             title="Clarke -- Valpha (alpha-axis)", color="#f0883e"),
        dict(signal="clarke[1]", ylabel="Vbeta (V)",
             title="Clarke -- Vbeta (beta-axis)",  color="#ffa657"),
        dict(signal="park[0]", ylabel="Vd (V)",
             title="Park -- Vd (d-axis)", color="#79c0ff",
             ref_val=round(vd, 3), ref_label=f"Vd = {vd:.3f} V"),
        dict(signal="park[1]", ylabel="Vq (V)",
             title="Park -- Vq (q-axis)", color="#56d364",
             ref_val=round(vq, 3), ref_label=f"Vq = {vq:.3f} V"),
    ],
    title=f"Clarke-Park Transform  |  f={F_ELEC} Hz   theta={THETA:.2f} rad",
    figsize=(13, 16),
    time_unit="ms",
    save_path="clarke_park_results.png",
)

# -- Code generation -----------------------------------------------------------
print("\n" + "=" * 60)
print(" Running code generation  ->  embedsim_gen/")
print("=" * 60)

gen = StepGenerator(cg_start, cg_end)
gen.generate(output_dir=_root, dt_hz=1.0 / DT)

print("\n[DONE] embedsim_loop.c and embedsim_loop.h written to embedsim_gen/")
