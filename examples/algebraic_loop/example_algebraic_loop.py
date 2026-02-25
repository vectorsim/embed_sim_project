"""
example_algebraic_loop.py
=========================
ControlForge — Algebraic Loop: Detection and Resolution

Scenario:
  A sine source feeds a gain block. The gain output feeds back into
  a sum block (alongside the original sine), creating a feedback loop.

  This example has TWO parts:

  PART 1 — Algebraic loop (broken diagram)
  ─────────────────────────────────────────
  The feedback path connects directly back to the sum with no delay.
  ControlForge detects the circular dependency at build time and raises
  a ValueError — the simulation never starts.

  Block diagram (broken):
    [sin] ──► [sum] ──► [gain] ──┐
                ▲                │
                └────────────────┘   ← algebraic loop! gain needs sum,
                                       sum needs gain — undefined order.

  PART 2 — Loop broken by VectorDelayEnhanced (working diagram)
  ──────────────────────────────────────────────────────────────
  Insert VectorDelay in the feedback path.nt
  The delay block outputs the PREVIOUS step's value, breaking the
  circular dependency. Now the execution order is well-defined:
    delay(k-1) → sum(k) → gain(k) → delay stores for step k+1

  Block diagram (fixed):
    [sin] ──► [sum] ──► [gain] ──► [output]
                ▲           │
           [delay] ◄────────┘

  Mathematical description:
    sin_out(t)  = A · sin(2π·f·t)
    sum_out(k)  = sin_out(k) + K · sum_out(k-1)      ← K from previous step
    gain_out(k) = K · sum_out(k)

    This is a first-order IIR-like recurrence driven by a sine input.
    With K < 1 the system is stable; with K >= 1 it diverges.

Run:
    python example_algebraic_loop.py
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Make the local embedsim package importable when running from any directory.
# Adjust this path to match where your embedsim folder lives.
# ---------------------------------------------------------------------------
from pathlib import Path

# Lambda to add N-levels parent to sys.path with print
add_parent_to_syspath = lambda levels=2: (
    print(f"[EmbedSim] Adding to sys.path: {(p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]}"),
    sys.path.insert(0, str((p:=Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()).parents[levels-1]))
)[-1]
add_parent_to_syspath(2)

# ─────────────────────────────────────────────────────────────────────────────

from embedsim.source_blocks              import (SinusoidalGenerator)
from embedsim.processing_blocks          import (VectorGain, VectorSum)
from embedsim.dynamic_blocks             import (VectorEnd)
from embedsim.simulation_engine          import (EmbedSim, VectorDelay, ODESolver)



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

# Build the broken diagram
sin_src  = SinusoidalGenerator("sin", AMP, FREQ, 0.0)
fb_sum   = VectorSum("sum",  signs=[1, 1])
fb_gain  = VectorGain("gain", gain=K)
fb_out   = VectorEnd("output")

sin_src >> fb_sum           # sin feeds into sum
fb_sum  >> fb_gain          # sum feeds into gain
fb_gain >> fb_sum           # gain feeds BACK into sum  ← creates the loop
fb_gain >> fb_out

try:
    # EnhancedVectorSim performs dependency graph analysis at construction time.
    # It will raise ValueError immediately — before run() is even called.
    sim_broken = EmbedSim(
        sinks  = [fb_out],
        T      = T_SIM,
        dt     = DT,
        solver = ODESolver.EULER,
    )
    print("  [UNEXPECTED] No error raised — loop not detected.")

except ValueError as e:
    print(f"  [DETECTED]  ControlForge raised ValueError:")
    print(f"  {e}")
    print("\n  ✓ The engine correctly refuses to run an algebraically")
    print("    inconsistent diagram. Add a VectorDelayEnhanced to fix it.")


# =============================================================================
# PART 2 — Fix with VectorDelayEnhanced
# =============================================================================
print("\n" + "=" * 60)
print("PART 2 — Loop broken by VectorDelay")
print("=" * 60)
print(f"""
  Block diagram:
    
  [sin (SinusoidalGenerator)] ──► [sum (VectorSum)] ──► [gain (VectorGain)] ──► [output (VectorEnd)]
                                  │                                        │
                                  ◄────────────────────────────────────────┘
                                             delay (VectorDelay)
                                                     
   └── ○ output (VectorEnd)
    └── ○ gain (VectorGain)
        └── ○ sum (VectorSum)
            ├── ○ sin (SinusoidalGenerator)
            └── ○ delay (VectorDelay)
                └── ○ gain (see above)


  VectorDelayEnhanced outputs the PREVIOUS step value.
  Execution order each step:
    1. delay  outputs  gain(k-1)          ← already known
    2. sum    computes sin(k) + gain(k-1)
    3. gain   computes K · sum(k)
    4. delay  stores  gain(k)  for next step

  Parameters:  f={FREQ} Hz   A={AMP}   K={K}   dt={DT} s   T={T_SIM} s
""")

# ── Build the corrected diagram ───────────────────────────────────────────────

sin_src = SinusoidalGenerator("sin",   AMP, FREQ, 0.0)

# VectorDelayEnhanced: initial=[0.0] means the feedback signal starts at zero.
# It implements the LoopBreaker interface so the engine knows to use its
# previous-step output when resolving the dependency graph.
fb_delay = VectorDelay("delay", initial=[0.0])

# sum = sin(k) + delay(k-1)
# signs=[1, 1]: both inputs are added with weight +1
loop_sum  = VectorSum("sum",    signs=[1, 1])
loop_gain = VectorGain("gain",  gain=K)
loop_out  = VectorEnd("output")

# Signal flow
# sin_src  >> loop_sum     # forward path: sine into sum
# fb_delay  >> loop_sum     # feedback path: delayed gain into sum (loop breaker)
# loop_sum  >> loop_gain    # sum into gain
# loop_gain >> fb_delay     # gain into delay (closes the loop safely)
# loop_gain >> loop_out     # gain into sink

sin_src >> loop_sum >> loop_gain >> loop_out
loop_gain >> fb_delay >> loop_sum



# ── Simulation ────────────────────────────────────────────────────────────────
sim = VectorSim(
    sinks  = [loop_out],
    T      = T_SIM,
    dt     = DT,
    solver = ODESolver.EULER,
)
sim.scope.add(sin_src,  label="Sine")
sim.scope.add(loop_sum,  label="Sum")
sim.scope.add(loop_gain, label="Gain_out")
sim.scope.add(fb_delay,  label="Delay")

print("Block diagram topology:")
sim.print_topology()

print("Block diagram topology tree:")
sim.print_topology_tree()
sim.print_topology_sources2sink()

print("Running simulation...")
sim.run(verbose=False, progress_bar=True)
print(f"  Completed: {len(sim.scope.t)} steps\n")


# =============================================================================
# Extract signals
# =============================================================================
t_arr    = np.array(sim.scope.t)
sig_sin  = sim.scope.get_signal("Sine",     index=0)
sig_sum  = sim.scope.get_signal("Sum",      index=0)
sig_gain = sim.scope.get_signal("Gain_out", index=0)
sig_dly  = sim.scope.get_signal("Delay",    index=0)


# =============================================================================
# Plot
# =============================================================================
print("Generating plot...")

BG   = "#F8FAFC"
DARK = "#1E293B"
BLUE = "#2563EB"
RED  = "#DC2626"
GRN  = "#16A34A"
ORNG = "#EA580C"
GRAY = "#9CA3AF"
TEAL = "#0891B2"

fig = plt.figure(figsize=(13, 10), facecolor=BG)
gs  = gridspec.GridSpec(
    3, 2, figure=fig,
    hspace=0.52, wspace=0.34,
    left=0.08, right=0.97,
    top=0.92,  bottom=0.07,
)

# ── A: All signals overview (full width) ──────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
ax_a.set_facecolor("white")

ax_a.plot(t_arr, sig_sin,  color=GRAY, lw=1.5, ls="--",
          label=f"Sine source  A·sin(2π·{FREQ}·t)", zorder=2)
ax_a.plot(t_arr, sig_sum,  color=TEAL, lw=1.8,
          label="Sum output  sin(k) + gain(k-1)", zorder=3)
ax_a.plot(t_arr, sig_gain, color=BLUE, lw=2.2,
          label=f"Gain output  K·sum(k)   K={K}", zorder=4)

ax_a.axhline(0, color=GRAY, lw=0.6, ls="-", alpha=0.4)
ax_a.set_ylabel("Amplitude  [–]", fontsize=10)
ax_a.set_title(
    "ControlForge — Algebraic Loop: Detection & Resolution\n"
    f"sin(2π·{FREQ}·t)  →  sum  →  gain(K={K})  →  delay  ↩  sum",
    fontsize=11, fontweight="bold",
)
ax_a.legend(fontsize=9, loc="upper right")
ax_a.grid(True, alpha=0.2)
ax_a.set_xlim(0, T_SIM)

# ── B: Sine source ────────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])
ax_b.set_facecolor("white")
ax_b.plot(t_arr, sig_sin, color=GRAY, lw=1.8, label=f"sin  f={FREQ} Hz  A={AMP}")
ax_b.fill_between(t_arr, 0, sig_sin,
                  where=(sig_sin >= 0), color=BLUE, alpha=0.10)
ax_b.fill_between(t_arr, 0, sig_sin,
                  where=(sig_sin  < 0), color=RED,  alpha=0.10)
ax_b.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
ax_b.set_xlabel("Time  [s]", fontsize=9)
ax_b.set_ylabel("Amplitude  [–]", fontsize=9)
ax_b.set_title("B — Sine Source  (no feedback)", fontsize=9, fontweight="bold")
ax_b.legend(fontsize=8)
ax_b.grid(True, alpha=0.2)
ax_b.set_xlim(0, T_SIM)

# ── C: Gain output vs delayed feedback ───────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])
ax_c.set_facecolor("white")
ax_c.plot(t_arr, sig_gain, color=BLUE, lw=2.0,
          label=f"Gain output  (K={K})")
ax_c.plot(t_arr, sig_dly,  color=ORNG, lw=1.5, ls="-.",
          label="Delay output  gain(k-1)  [one step behind]")
ax_c.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
ax_c.set_xlabel("Time  [s]", fontsize=9)
ax_c.set_ylabel("Amplitude  [–]", fontsize=9)
ax_c.set_title("C — Gain vs Delayed Feedback", fontsize=9, fontweight="bold")
ax_c.legend(fontsize=8)
ax_c.grid(True, alpha=0.2)
ax_c.set_xlim(0, T_SIM)

# ── D: Sum breakdown — show how feedback adds to sine ─────────────────────────
ax_d = fig.add_subplot(gs[2, 0])
ax_d.set_facecolor("white")
ax_d.plot(t_arr, sig_sin, color=GRAY, lw=1.5, ls="--",
          label="Sine alone")
ax_d.plot(t_arr, sig_sum, color=TEAL, lw=2.0,
          label="Sum = sine + feedback")
ax_d.fill_between(t_arr, sig_sin, sig_sum,
                  color=GRN, alpha=0.20,
                  label="Feedback contribution")
ax_d.axhline(0, color=GRAY, lw=0.8, ls="--", alpha=0.5)
ax_d.set_xlabel("Time  [s]", fontsize=9)
ax_d.set_ylabel("Amplitude  [–]", fontsize=9)
ax_d.set_title("D — VectorSum:  Sine  +  Delayed Feedback", fontsize=9,
               fontweight="bold")
ax_d.legend(fontsize=8)
ax_d.grid(True, alpha=0.2)
ax_d.set_xlim(0, T_SIM)

# ── E: Explanation card ───────────────────────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 1])
ax_e.set_facecolor(DARK)
ax_e.set_axis_off()

card = [
    ("Algebraic Loop",                    0.95, 10.5, "#94A3B8", "bold"),
    ("=" * 34,                            0.89,  8.0, "#334155", "normal"),
    ("BROKEN  (Part 1)",                  0.83,  9.5, "#DC2626", "bold"),
    ("  gain needs sum output",           0.77,  9.0, "#E2E8F0", "normal"),
    ("  sum  needs gain output",          0.71,  9.0, "#E2E8F0", "normal"),
    ("  → circular → ValueError",         0.65,  9.0, "#FCA5A5", "normal"),
    ("=" * 34,                            0.59,  8.0, "#334155", "normal"),
    ("FIXED  (Part 2)",                   0.53,  9.5, "#34D399", "bold"),
    ("  VectorDelayEnhanced inserted",    0.47,  9.0, "#E2E8F0", "normal"),
    ("  delay outputs gain(k-1)",         0.41,  9.0, "#E2E8F0", "normal"),
    ("  order: delay→sum→gain→delay",     0.35,  9.0, "#E2E8F0", "normal"),
    ("=" * 34,                            0.29,  8.0, "#334155", "normal"),
    ("PARAMETERS",                        0.23,  9.5, "#FBBF24", "bold"),
    (f"  f={FREQ} Hz   A={AMP}   K={K}", 0.17,  9.0, "#E2E8F0", "normal"),
    (f"  dt={DT} s   T={T_SIM} s",       0.11,  9.0, "#E2E8F0", "normal"),
    (f"  K<1 → stable recurrence",        0.05,  9.0, "#E2E8F0", "normal"),
]

for text, y, fsize, col, weight in card:
    ax_e.text(0.04, y, text,
              transform=ax_e.transAxes,
              fontsize=fsize, color=col,
              fontweight=weight, fontfamily="monospace", va="top")

# ── Save ──────────────────────────────────────────────────────────────────────
plot_path = os.path.join(_HERE, "example_algebraic_loop.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"  Plot saved → {plot_path}")
print("\n" + "=" * 60 + "\nDone.\n")
