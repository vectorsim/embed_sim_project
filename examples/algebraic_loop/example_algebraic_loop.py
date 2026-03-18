"""
example_algebraic_loop.py
=========================
EmbedSim — Algebraic Loop: Detection and Resolution

Scenario:
  A sine source feeds a gain block.  The gain output feeds back into
  a sum block (alongside the original sine), creating a feedback loop.

  This example has TWO parts:

  PART 1 — Algebraic loop (broken diagram)
  ─────────────────────────────────────────
  The feedback path connects directly back to the sum with no delay.
  EmbedSim detects the circular dependency at build time and raises a
  ValueError — the simulation never starts.

  Block diagram (broken):
    [sin] ──► [sum] ──► [gain] ──┐
                ▲                │
                └────────────────┘   ← algebraic loop!
                                       gain needs sum output,
                                       sum  needs gain output — undefined order.

  PART 2 — Loop broken by VectorDelay (working diagram)
  ──────────────────────────────────────────────────────
  Insert VectorDelay in the feedback path.
  The delay block outputs the PREVIOUS step's value, breaking the
  circular dependency.  Execution order is now well-defined:

    delay(k-1) → sum(k) → gain(k) → delay stores for step k+1

  Block diagram (fixed):
    [sin] ──► [sum] ──► [gain] ──► [output]
                ▲           │
           [delay] ◄────────┘

  Mathematical description:
    sin_out(t)  = A · sin(2π·f·t)
    sum_out(k)  = sin_out(k) + gain_out(k-1)      ← fed from previous step
    gain_out(k) = K · sum_out(k)

    This is a first-order IIR-like recurrence driven by a sine input.
    With |K| < 1 the system is stable; with |K| ≥ 1 it diverges.

Run:
    python example_algebraic_loop.py
"""

# =============================================================================
# Path bootstrap — _path_utils.py lives alongside this script and resolves
# the project root via .project_root_marker, so the script runs from any CWD.
# =============================================================================
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# Add this folder to sys.path so _path_utils is importable
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _path_utils import get_embedsim_import_path, get_project_root

_embedsim_path = get_embedsim_import_path()
if _embedsim_path not in sys.path:
    sys.path.insert(0, _embedsim_path)
    print(f"[EmbedSim] Added to sys.path: {_embedsim_path}")

_PROJECT_ROOT = get_project_root()

# =============================================================================
# Imports
# =============================================================================
import matplotlib
matplotlib.use("Agg")

from embedsim.source_blocks     import SinusoidalGenerator
from embedsim.processing_blocks import VectorGain, VectorSum
from embedsim.dynamic_blocks    import VectorEnd
from embedsim.simulation_engine import EmbedSim, VectorDelay, ODESolver
from embedsim.plot_helper       import create_plotter


# =============================================================================
# Parameters
# =============================================================================
FREQ  =  2.0    # Hz   sine frequency
AMP   =  1.0    # –    sine amplitude
K     =  0.5    # –    feedback gain  (stable for |K| < 1)
T_SIM =  3.0    # s    simulation duration
DT    =  0.01   # s    time step


# =============================================================================
# PART 1 — Demonstrate the algebraic loop error
# =============================================================================
print("\n" + "=" * 60)
print("PART 1 — Algebraic loop (no delay in feedback)")
print("=" * 60)
print("""
  Block diagram:
    [sin] ──► [sum] ──► [gain] ──┐
                ▲                │
                └────────────────┘

  gain needs sum's output to compute.
  sum  needs gain's output to compute.
  → circular dependency → undefined execution order
""")

sin_src = SinusoidalGenerator("sin",  AMP, FREQ, 0.0)
fb_sum  = VectorSum("sum",   signs=[1, 1])
fb_gain = VectorGain("gain", gain=K)
fb_out  = VectorEnd("output")

sin_src >> fb_sum
fb_sum  >> fb_gain
fb_gain >> fb_sum   # ← closes the algebraic loop
fb_gain >> fb_out

try:
    sim_broken = EmbedSim(
        sinks  = [fb_out],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )
    print("  [UNEXPECTED] No error raised — loop not detected.")

except ValueError as e:
    print(f"  [DETECTED]  EmbedSim raised ValueError:")
    print(f"  {e}")
    print("\n  ✓ The engine correctly refuses to run an algebraically")
    print("    inconsistent diagram.  Insert a VectorDelay to fix it.")


# =============================================================================
# PART 2 — Fix with VectorDelay
# =============================================================================
print("\n" + "=" * 60)
print("PART 2 — Loop broken by VectorDelay")
print("=" * 60)
print(f"""
  Block diagram:

    [sin] ──► [sum] ──► [gain] ──► [output]
                ▲           │
           [delay] ◄────────┘

  VectorDelay outputs the PREVIOUS step value.
  Execution order each step:
    1. delay  outputs  gain(k-1)          ← already known
    2. sum    computes sin(k) + gain(k-1)
    3. gain   computes K · sum(k)
    4. delay  stores  gain(k)  for next step

  Parameters:  f={FREQ} Hz   A={AMP}   K={K}   dt={DT} s   T={T_SIM} s
""")

# ── Build the corrected diagram ───────────────────────────────────────────────
sin_src   = SinusoidalGenerator("sin",   AMP, FREQ, 0.0)
fb_delay  = VectorDelay("delay", initial=[0.0])   # loop breaker
loop_sum  = VectorSum("sum",   signs=[1, 1])
loop_gain = VectorGain("gain", gain=K)
loop_out  = VectorEnd("output")

# Forward path
sin_src >> loop_sum >> loop_gain >> loop_out
# Feedback path (broken by VectorDelay)
loop_gain >> fb_delay >> loop_sum


# ── Simulation ────────────────────────────────────────────────────────────────
sim = EmbedSim(
    sinks  = [loop_out],
    T      = T_SIM,
    dt     = DT,
    solver = ODESolver.EULER,
)
sim.scope.add(sin_src,   label="Sine")
sim.scope.add(loop_sum,  label="Sum")
sim.scope.add(loop_gain, label="Gain_out")
sim.scope.add(fb_delay,  label="Delay")

# ── Topology ──────────────────────────────────────────────────────────────────
print("Block diagram topology:")
sim.print_topology()

print("\nBlock diagram topology (console):")
if sim.topo is not None:
    sim.topo.print_console()

_topo_html = _HERE / "example_algebraic_loop_topo.html"
if sim.topo is not None:
    sim.topo.export_html(str(_topo_html))
    print(f"\n  Topology HTML → {_topo_html}")

# ── Run ───────────────────────────────────────────────────────────────────────
print("\nRunning simulation...")
sim.run(verbose=False, progress_bar=True)
print(f"  Completed: {len(sim.scope.t)} steps\n")


# =============================================================================
# Plot — PlotHelper
# =============================================================================
print("Generating plots...")

ph = create_plotter(sim)
ph.info()

_plot_overview = str(_HERE / "example_algebraic_loop_overview.png")
_plot_grid     = str(_HERE / "example_algebraic_loop.png")

# ── Overview: all four signals on one axes ────────────────────────────────────
ph.easyplot(
    signals   = ["Sine[0]", "Sum[0]", "Gain_out[0]", "Delay[0]"],
    title     = (
        f"EmbedSim — Algebraic Loop: Detection & Resolution\n"
        f"sin(2π·{FREQ}·t)  →  sum  →  gain(K={K})  →  delay  ↩  sum"
    ),
    figsize   = (13, 5),
    save_path = _plot_overview,
    linewidth = 2.0,
)

# ── Grid: one subplot per signal, clean engineering layout ───────────────────
ph.plot_grid(
    rows = [
        dict(
            signal = "Sine[0]",
            ylabel = "Amplitude  [–]",
            title  = f"Sine Source  A·sin(2π·{FREQ}·t)",
            color  = "#9CA3AF",
        ),
        dict(
            signal = "Sum[0]",
            ylabel = "Amplitude  [–]",
            title  = "VectorSum Output  sin(k) + gain(k-1)",
            color  = "#0891B2",
        ),
        dict(
            signal = "Gain_out[0]",
            ylabel = "Amplitude  [–]",
            title  = f"VectorGain Output  K·sum(k)   K = {K}",
            color  = "#2563EB",
        ),
        dict(
            signal    = "Delay[0]",
            ylabel    = "Amplitude  [–]",
            title     = "VectorDelay Output  gain(k-1)  [one step behind]",
            color     = "#EA580C",
        ),
    ],
    title     = (
        f"EmbedSim — Algebraic Loop Fixed by VectorDelay\n"
        f"f = {FREQ} Hz   A = {AMP}   K = {K}   dt = {DT} s   T = {T_SIM} s"
    ),
    figsize   = (13, 10),
    time_unit = "s",
    save_path = _plot_grid,
)

print(f"  Overview plot → {_plot_overview}")
print(f"  Grid plot     → {_plot_grid}")
print("\n" + "=" * 60 + "\nDone.\n")
