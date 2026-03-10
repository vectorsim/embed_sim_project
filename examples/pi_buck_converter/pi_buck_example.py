"""
pi_buck_example.py
==================
Complete buck converter simulation with PI control.

Enhanced with:
  - CodeGen  : emits embedsim_loop.c / embedsim_loop.h into embedsim_gen/
  - Topology : ASCII block-diagram printed before the run
  - PlotHelper: formatted 3-subplot figure with annotations
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
from embedsim.topology_printer import TopologyPrinter       # prints ASCII DAG

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
# ║  2.  WIRING  (>> operator)                                                  ║
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


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  3.  TOPOLOGY PRINTER                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

sim = EmbedSim(sinks=[sink], T=0.01, dt=1e-6, solver=ODESolver.RK4)
printer = TopologyPrinter(sim, title="Buck Converter — PI Voltage Control")
printer.print_console()
printer.show_gui()


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  4.  SIMULATION  + CODEGEN                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

sim.scope.add(v_ref,           label="v_ref")
sim.scope.add(pi_controller,   label="pi_ctrl")
sim.scope.add(buck_plant,      label="buck_out", indices=[0, 1])  # V_out, I_L
sim.scope.add(feedback_delay,  label="fb_delay")

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

class PlotHelper:
    """Lightweight scope-to-figure helper for EmbedSim examples."""

    _STYLE = {
        "axes.facecolor":   "#0d1117",
        "figure.facecolor": "#0d1117",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#c9d1d9",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "text.color":       "#c9d1d9",
    }

    def __init__(self, sim, nrows=3, title="", figsize=(12, 8)):
        matplotlib.rcParams.update(self._STYLE)
        self.sim   = sim
        self.nrows = nrows
        self.fig   = plt.figure(figsize=figsize, constrained_layout=True)
        self.fig.suptitle(title, fontsize=14, color="#58a6ff", fontweight="bold")
        self.gs    = gridspec.GridSpec(nrows, 1, figure=self.fig, hspace=0.35)
        self.axes  = [self.fig.add_subplot(self.gs[i]) for i in range(nrows)]

    def subplot(self, row, label, index=0, *,
                ylabel="", title="", color="#58a6ff",
                ylim=None, ref_val=None, ref_label=None, step_time_ms=None):
        ax = self.axes[row]
        scope = self.sim.scope

        # VectorScope stores time in .t (list) or ._buf_t (pre-allocated ndarray)
        t_raw = scope.t if scope.t else (scope._buf_t[:scope._step_idx]
                                         if hasattr(scope, '_buf_t') and scope._buf_t is not None
                                         else [])
        t = np.asarray(t_raw) * 1e3          # s → ms

        # Data may be keyed as "label[0]", "label[1]" (multi-index) or plain "label"
        key = f"{label}[{index}]"
        if key not in scope.data and label in scope.data:
            # Scalar / single-element signal stored under plain label
            raw = scope.data[label]
            y = np.array([float(v) for v in raw])
        elif key in scope.data:
            raw = scope.data[key]
            y = np.array([float(v) for v in raw])
        else:
            y = np.zeros_like(t)

        ax.plot(t, y, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel or label, fontsize=9)
        ax.set_title(title or label, fontsize=10, color="#58a6ff")
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if ref_val is not None:
            ax.axhline(ref_val, color="#f0883e", linestyle="--",
                       linewidth=1, label=ref_label or f"ref={ref_val}")
            ax.legend(fontsize=8, loc="upper right")
        if step_time_ms is not None:
            ax.axvline(step_time_ms, color="#8b949e", linestyle=":", linewidth=1)
        if len(y):
            ax.annotate(
                f"{y[-1]:.3f}",
                xy=(t[-1], y[-1]),
                xytext=(-6, 4),
                textcoords="offset points",
                fontsize=8,
                color="#c9d1d9",
            )

    def finalize(self, filename="pi_buck_response.png", dpi=150):
        self.axes[-1].set_xlabel("Time (ms)", fontsize=10)
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight",
                         facecolor=self.fig.get_facecolor())
        print(f"\n📊 Plot saved → {filename}")
        plt.show()


ph = PlotHelper(sim, nrows=3, title="Buck Converter — PI Voltage Control (EmbedSim)",
                figsize=(12, 9))

ph.subplot(0, "buck_out", index=0,
           ylabel="Voltage (V)", title="Output Voltage V_out",
           color="#58a6ff",
           ylim=(-1, 20),
           ref_val=12.0, ref_label="V_ref = 12 V",
           step_time_ms=1.0)

ph.subplot(1, "pi_ctrl", index=0,
           ylabel="Duty cycle", title="PI Controller Output — Duty Cycle",
           color="#3fb950",
           ylim=(0.0, 1.0))

ph.subplot(2, "buck_out", index=1,
           ylabel="Current (A)", title="Inductor Current I_L",
           color="#d2a8ff",
           ylim=(-12, 12))

output_png = str(Path(__file__).parent / "pi_buck_response.png")
ph.finalize(output_png)