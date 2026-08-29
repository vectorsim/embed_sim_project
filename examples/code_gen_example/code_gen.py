"""
example_two_sines_gain_with_codegen.py
======================================
EmbedSim — Two sine sources, gain, summer, with CodeGen

Same as the original example, but with:
  - CodeGenStart/End markers around the processing region (gain_b + summer)
  - Modified SumBlock that can sum elements of a single vector input
  - C stub generation via cg_end.generate_pyx_stub()
"""

from __future__ import annotations

import sys
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
sys.path.insert(0, str(_HERE))
from _path_utils import setup_embedsim_path
setup_embedsim_path()

# ---------------------------------------------------------------------------
# EmbedSim imports
# ---------------------------------------------------------------------------
from embedsim import EmbedSim, ODESolver, VectorEnd, CodeGenStart, CodeGenEnd
from embedsim.core_blocks import VectorBlock, VectorSignal, DEFAULT_DTYPE

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
T_SIM = 0.10
DT    = 1e-4
FREQ_A = 50.0
AMP_A  = 1.0
FREQ_B = 150.0
AMP_B  = 0.5
GAIN_B = 2.0

# =============================================================================
# Block definitions (extended from original)
# =============================================================================

class SineSource(VectorBlock):
    def __init__(self, name: str, freq: float, amp: float = 1.0) -> None:
        super().__init__(name)
        self.freq = freq
        self.amp  = amp

    def compute_py(self, t, dt, input_values=None):
        value = self.amp * math.sin(2.0 * math.pi * self.freq * t)
        self.output = VectorSignal(np.array([value], dtype=DEFAULT_DTYPE), self.name)
        return self.output

class GainBlock(VectorBlock):
    def __init__(self, name: str, k: float = 1.0) -> None:
        super().__init__(name)
        self.k = k

    def compute_py(self, t, dt, input_values=None):
        upstream = input_values[0].value if input_values else np.zeros(1)
        self.output = VectorSignal((self.k * upstream).astype(DEFAULT_DTYPE), self.name)
        return self.output

class SumBlock(VectorBlock):
    """
    Extended SumBlock:
      - If given multiple inputs, sums them element-wise.
      - If given a single input (a vector), sums its elements (for CodeGen).
    """
    def compute_py(self, t, dt, input_values=None):
        if input_values is None:
            val = np.zeros(1, dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(val, self.name)
            return self.output

        if len(input_values) == 1:
            # Single input: sum its elements (for CodeGen concat)
            v = input_values[0].value
            val = np.sum(v)
            self.output = VectorSignal(np.array([val], dtype=DEFAULT_DTYPE), self.name)
        else:
            # Original element-wise sum over multiple inputs
            result = None
            for sig in input_values:
                if sig is None:
                    continue
                v = sig.value
                result = v if result is None else result + v
            if result is None:
                result = np.zeros(1, dtype=DEFAULT_DTYPE)
            self.output = VectorSignal(result.astype(DEFAULT_DTYPE), self.name)
        return self.output

# =============================================================================
# Build model with CodeGen markers
# =============================================================================

sine_a = SineSource("sine_a", freq=FREQ_A, amp=AMP_A)
sine_b = SineSource("sine_b", freq=FREQ_B, amp=AMP_B)
gain_b = GainBlock("gain_b", k=GAIN_B)
summer = SumBlock("summer")
sink   = VectorEnd("sink")

# CodeGen markers
cg_start = CodeGenStart("cg_start")
cg_end   = CodeGenEnd("cg_end")

# Wiring:
#   sine_a ──┐
#            ├──► cg_start ──► summer ──► cg_end ──► sink
#   sine_b ──► gain_b ──┘
sine_a >> cg_start
sine_b >> gain_b >> cg_start
cg_start >> summer >> cg_end >> sink

# ---------------------------------------------------------------------------
# Simulation and plotting (same as original)
# ---------------------------------------------------------------------------
sim = EmbedSim(sinks=[sink], T=T_SIM, dt=DT, solver=ODESolver.EULER)

sim.scope.add(sine_a, indices=[0], label="sine_a")
sim.scope.add(gain_b, indices=[0], label="gained_b")
sim.scope.add(summer, indices=[0], label="sum_out")

print("\n" + "=" * 60)
print("  [Topology]  Signal-flow diagram (console)")
print("=" * 60)
sim.topo.print_console()

_topo_html = str(_HERE / "example_signal_flow_codegen.html")
sim.topo.export_html(_topo_html)
print(f"\n  [Topology]  Written: {_topo_html}\n")

print("=" * 60)
print("  [Execution order]  (DFS topological sort)")
print("=" * 60)
_order = sim.topo.sorted_blocks if hasattr(sim.topo, "sorted_blocks") else sim.execution_order
for step, blk in enumerate(_order):
    print(f"  Step {step:>2d} :  {blk.name:<20s}  ({type(blk).__name__})")
print()

print("  [Run]  Starting simulation …")
sim.run()
print(f"  [Run]  Done — {int(T_SIM / DT)} steps × dt={DT*1e6:.0f} µs\n")

# Extract data
sc = sim.scope
data = {
    "t":         np.array(sc.t, dtype=np.float32),
    "sig_a":     sc.get_signal("sine_a", 0),
    "sig_b_g":   sc.get_signal("gained_b", 0),
    "sig_sum":   sc.get_signal("sum_out", 0),
}

# Plot
t = data["t"]
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f"Two Sines + Gain  (f_a={FREQ_A} Hz, f_b={FREQ_B} Hz, k={GAIN_B})")
axes[0].plot(t*1e3, data["sig_a"], label="sine_a")
axes[0].set_ylabel("Amp"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(t*1e3, data["sig_b_g"], label=f"gained_b (×{GAIN_B})")
axes[1].set_ylabel("Amp"); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(t*1e3, data["sig_sum"], label="sum = sine_a + gained_b")
axes[2].set_xlabel("Time [ms]"); axes[2].set_ylabel("Amp")
axes[2].legend(); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(str(_HERE / "two_sines_gain_codegen.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------------------
# Code Generation: generate a single C stub for the region
# ---------------------------------------------------------------------------
print("\n[CodeGen] Generating C stub for processing region (gain_b + summer)")
cg_end.generate_pyx_stub(
    cg_start=cg_start,
    block_name="processor",
    output_dir="./codegen_stub",
    write_files=True,
)

print("\n✅ C stub generated in ./codegen_stub/")
print("   - processor.h          (InputSignals, OutputSignals, processor_compute)")
print("   - processor_wrapper.pyx (Cython wrapper)")
print("   - processor_simblock.py (Python SimBlock)")
print("   - setup_processor.py    (compilation script)")
print("\nImplement ONE C function in processor.c:")
print("   #include \"processor.h\"")
print("   void processor_compute(const InputSignals* in, OutputSignals* out) {")
print("       out->summer = in->sine_a + in->gain_b;")
print("   }")
print("\nCompile: cd codegen_stub && python setup_processor.py build_ext --inplace")
print("Then use ProcessorSimBlock with use_c_backend=True in Python.")

print("\n  Output files:")
print("    example_signal_flow_codegen.html")
print("    two_sines_gain_codegen.png")
print("    ./codegen_stub/")