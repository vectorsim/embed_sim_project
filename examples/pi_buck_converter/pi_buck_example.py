"""
pi_buck_example.py
==================
Complete buck converter simulation with PI control.

Enhanced with:
  - CodeGen  : emits embedsim_loop.c / embedsim_loop.h into embedsim_gen/
  - Topology : ASCII block-diagram printed before the run
  - PlotHelper: formatted 3-subplot figure with annotations

WIRE LABELS
-----------
Signal names on topology arrows are declared in WIRE_LABELS (Section 2).
This dict is the single authoritative place for signal semantics in this
simulation. topology_printer and export_html both receive it so the
console ASCII diagram and the HTML viewer show identical labels.

Pattern:  ("src_block_name", "dst_block_name") → "signal label"

For any new model, copy this file, rebuild the wiring section, and
update WIRE_LABELS to match the new signal names. No framework changes
are needed — the dict is passed through as-is.
"""

import sys
import numpy as np
from pathlib import Path

# ── Project path ─────────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# ── EmbedSim imports ──────────────────────────────────────────────────────────
from embedsim.simulation_engine import EmbedSim, ODESolver, VectorDelay
from embedsim.source_blocks     import VectorStep
from embedsim.dynamic_blocks    import VectorEnd

# CodeGen — inline pass-through boundary markers + loop generator
from embedsim.code_generator import CodeGenStart, CodeGenEnd, LoopGenerator

# Topology printer
from embedsim.topology_printer import TopologyPrinter

# Plot helper
from embedsim.plot_helper import create_plotter

# Buck converter blocks
sys.path.append(str(project_root / "buck_converter"))
from pi_buck_block      import PI_BuckBlock
from BuckConverterBlock import BuckConverterBlock


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  1.  BLOCK CONSTRUCTION                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Reference voltage source — 0 V before t=1 ms, 12 V after
v_ref = VectorStep(
    "vref",
    step_time=0.001,
    before_value=0.0,
    after_value=12.0,
    dim=1,
)

# PI controller (C backend)
pi_controller = PI_BuckBlock(
    name="pi_buck",
    Kp=0.15,
    Ki=8.0,
    duty_max=0.9,
    duty_min=0.1,
    Ts=1e-4,
    use_c_backend=True,     # uses pi_buck_wrapper.pyd
)

# Buck-converter FMU plant
buck_plant = BuckConverterBlock(
    name="buck",
    fmu_path=str(
        project_root / "buck_converter" / "modelica" / "BuckConverter.fmu"
    ),
    L=100e-6,
    C=100e-6,
    R_load=10,
    V_in=24,
    f_sw=100e3,
)

# One-step delay to break the algebraic feedback loop
feedback_delay = VectorDelay("fb_delay", initial=[0.0])

# Sink
sink = VectorEnd("sink")

# CodeGen boundary markers — transparent pass-throughs wired inline with >>.
# cg_start marks the INPUT  boundary of the region to be C-generated.
# cg_end   marks the OUTPUT boundary of the region.
# After sim.run(), call LoopGenerator(cg_start, cg_end).generate() to emit .c/.h
cg_start = CodeGenStart("pi_ctrl_start")
cg_end   = CodeGenEnd("pi_ctrl_end")
codegen_dir = str(project_root / "embedsim_gen")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  2.  WIRING  (>> operator)  +  WIRE LABELS                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Forward path: v_ref → [cg_start] → PI → [cg_end] → buck plant → sink
v_ref         >> cg_start          # V_ref enters codegen region
cg_start      >> pi_controller     # V_ref → PI port 0
pi_controller >> cg_end            # duty exits codegen region
cg_end        >> buck_plant        # duty → FMU input
buck_plant    >> sink              # outputs → sink

# Feedback path: V_out → delay → PI port 1
buck_plant    >> feedback_delay    # V_out → delay
feedback_delay >> pi_controller   # delayed V_meas → PI port 1

print("\n✅ FMU outputs:", buck_plant.OUTPUT_VARS)

# ── Wire labels ───────────────────────────────────────────────────────────────
# Maps (src_block_name, dst_block_name) → signal label shown on the arrow.
#
# This is the only place in the project where signal names are declared.
# It lives here (not in the framework) because only this simulation script
# knows what each wire carries — the framework cannot infer it automatically.
#
# Rules:
#   - Keys must match the block name strings passed to each constructor.
#   - Fan-out wires from the same source get individual entries with
#     different labels (e.g. buck → fb_delay vs buck → sink).
#   - Omitting a wire leaves it unlabelled — no error, just no text on arrow.
#   - For a new model, copy and update this dict; no framework changes needed.
WIRE_LABELS = {
    ("vref",          "pi_ctrl_start"): "[V_ref]",
    ("pi_ctrl_start", "pi_buck"):       "[V_ref]",
    ("pi_buck",       "pi_ctrl_end"):   "[duty]",
    ("pi_ctrl_end",   "buck"):          "[duty]",
    ("buck",          "fb_delay"):      "[V_out]",
    ("fb_delay",      "pi_buck"):       "[V_meas]",
    ("buck",          "sink"):          "[V_out, I_L, I_load]",
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3.  TOPOLOGY PRINTER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

sim = EmbedSim(sinks=[sink], T=0.01, dt=1e-6, solver=ODESolver.RK4)

# wire_labels is passed to both the console printer and the HTML exporter
# so ASCII diagram and HTML viewer show identical signal names on every arrow.
printer = TopologyPrinter(sim, title="Buck Converter — PI Voltage Control",
                          wire_labels=WIRE_LABELS)
printer.print_console()
printer.show_gui()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4.  SIMULATION  + CODEGEN                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

sim.scope.add(v_ref,           label="v_ref")
sim.scope.add(pi_controller,   label="pi_ctrl")
sim.scope.add(buck_plant,      label="buck_out", indices=[0, 1])  # V_out, I_L
sim.scope.add(feedback_delay,  label="fb_delay")

# wire_labels passed here so the exported HTML carries the signal names
sim.topo.export_html("pi_buck.html", wire_labels=WIRE_LABELS)

print("\nRunning simulation …")
sim.run(verbose=True, progress_bar=True)
print("Simulation complete!")

# ── CodeGen ───────────────────────────────────────────────────────────────────
# LoopGenerator walks the sub-graph between cg_start and cg_end (exclusive),
# inspects each SimBlockBase block, and emits:
#   embedsim_gen/embedsim_loop.c   — static C loop (MCU-ready)
#   embedsim_gen/embedsim_loop.h   — extern declarations + includes
#
# BuckConverterBlock (FMU) is outside the region → automatically excluded.
# Only the PI controller C code is emitted.
# dt_hz=1e6 bakes  #define EMBEDSIM_DT (0.0000010000f)  into the header.

gen = LoopGenerator(cg_start, cg_end)
gen.generate(output_dir=codegen_dir, dt_hz=1e6)
print(f"  ↳ C files written to: {codegen_dir}/")
print( "    embedsim_loop.c")
print( "    embedsim_loop.h")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  5.  PLOT HELPER                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

output_png = str(Path(__file__).parent / "pi_buck_response.png")

ph = create_plotter(sim)
ph.plot_grid([
    dict(signal="buck_out[0]", ylabel="Voltage (V)",
         title="Output Voltage  V_out",  color="#58a6ff",
         ylim=(-1, 20), ref_val=12.0, ref_label="V_ref = 12 V",
         step_time=1.0),
    dict(signal="pi_ctrl[0]",  ylabel="Duty cycle",
         title="PI Controller — Duty Cycle", color="#3fb950",
         ylim=(0.0, 1.0)),
    dict(signal="buck_out[1]", ylabel="Current (A)",
         title="Inductor Current  I_L",  color="#d2a8ff",
         ylim=(-12, 12)),
], title="Buck Converter — PI Voltage Control (EmbedSim)",
   save_path=output_png)