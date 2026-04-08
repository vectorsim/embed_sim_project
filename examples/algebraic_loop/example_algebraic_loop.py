"""
example_algebraic_loop.py
=========================
EmbedSim — Algebraic Loop: Detection, Resolution & Engine Internals

This is the canonical introductory example for EmbedSim.  Read it top-to-bottom
once and you will understand how the engine works at every level.

┌─────────────────────────────────────────────────────────────────────────────┐
│  PART 1 — What is an algebraic loop and why does it crash simulation?       │
│  PART 2 — How VectorDelay breaks a loop (z⁻¹ one-step memory)              │
│  PART 3 — Engine internals: DFS traversal, execution order, zero-fallback  │
└─────────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 1 — Algebraic loop (unresolvable circular dependency)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Direct feedback without a delay element creates a circular data dependency:

    [sin] ──► [sum] ──► [gain] ──┐
                ▲                │
                └────────────────┘   ← algebraic loop!

  To compute gain, we need sum's output.
  To compute sum,  we need gain's output.
  There is no valid execution order — both blocks depend on each other
  within the same timestep.

  EmbedSim detects this at __init__ time via a two-pass DFS (described in
  PART 3 below) and raises ValueError before a single timestep executes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 2 — VectorDelay breaks the loop (the z⁻¹ solution)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    [sin] ──► [sum] ──► [gain] ──► [output]
                ▲           │
           [delay] ◄────────┘    (z⁻¹ block — outputs gain from PREVIOUS step)

  VectorDelay holds the value computed at step k−1 and presents it as its
  output at step k.  The circular dependency dissolves because the feedback
  signal is always one step old — already known before the current step begins.

  Execution order each step k:
    1. delay  presents  gain(k−1)     ← held from previous step, 0.0 at k=0
    2. sum    computes  sin(k) + delay_out = sin(k) + gain(k−1)
    3. gain   computes  K · sum(k)
    4. output records   gain(k)
    (delay latches gain(k) internally, making it delay_out for step k+1)

  Mathematical recurrence:
    y(k) = K · [ A·sin(2π·f·k·dt)  +  y(k−1) ]

  This is a first-order IIR filter driven by a sine wave.
  Stability: bounded for |K| < 1, diverges for |K| ≥ 1.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PART 3 — EmbedSim engine internals (read simulation_engine.py alongside this)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Phase 0 — Graph construction (the >> operator)
  ──────────────────────────────────────────────
  The >> operator registers directed edges in each block's successor list.
  No computation happens.  The graph is a pure Python object graph at this
  point — blocks hold references to their input blocks via block.inputs[].

  Phase 1 — EmbedSim.__init__: two-pass DFS (traverse_blocks_from_sinks_with_loops)
  ──────────────────────────────────────────────────────────────────────────────────
  Pass 1 (find_loop_breakers):
    Walks the full graph from every sink, recursively following block.inputs[].
    Any block whose class inherits from LoopBreaker (e.g. VectorDelay) is
    collected in the loop_breakers set.  The LoopBreaker.is_loop_breaker flag
    makes identification O(1) without isinstance on every edge.

  Pass 2 (dfs — post-order):
    Walks again from every sink.  For each block:
      • If the block is in 'visiting' (grey — on the current DFS stack) AND
        is NOT a LoopBreaker → true algebraic loop → ValueError raised.
      • If the block IS a LoopBreaker → inserted into execution order
        immediately but its inputs are NOT followed recursively.
        The back-edge is cut; the cycle ceases to exist from DFS's point of view.
      • Otherwise → recurse into inputs first (post-order), then append block.

    Result: a flat list 'blocks_order' — the safe sequential execution order.
    For our 5-block diagram it produces:  [delay, sin, sum, gain, output]

    Note the counter-intuitive placement of delay FIRST.  This is correct
    because delay is inserted as a loop-breaker when DFS processes sum's
    inputs — before sum itself is finalised.

  Phase 2 — _compute_all_blocks (every timestep)
  ───────────────────────────────────────────────
  Step A — Loop-breaker pre-initialisation (lines 1031-1036 in engine):
    Before the forward pass, any LoopBreaker whose .output is still None
    (first call, t=0) has its output set from get_loop_breaking_output().
    For VectorDelay(initial=[0.0]) this returns VectorSignal([0.0]).
    After this step, delay.output = [0.0]  and downstream blocks can read it.

  Step B — Forward pass (lines 1041-1060 in engine):
    Iterates blocks_order sequentially.  For each block:
      • Collects inp.output for every inp in block.inputs[].
      • If any upstream block's output is still None → zero-fallback warning.
      • Calls block.compute(t, dt, input_values).

    ┌─ Why the zero-fallback warning fires at t=0 (and what it means) ────────┐
    │                                                                          │
    │  Execution order:  [delay, sin, sum, gain, output]                       │
    │                                                                          │
    │  When the forward pass reaches DELAY (position 1 in the list):          │
    │    delay.inputs = [gain]                                                 │
    │    gain.output  = None   ← gain hasn't run yet in this pass             │
    │    → zero-fallback warning fires                                         │
    │    → input_values = [VectorSignal([0.0])]  substituted                  │
    │    → delay.compute_py() runs:                                            │
    │        outputs  last_output = initial = [0.0]   (correct!)              │
    │        latches  input [0.0] as new last_output for next step            │
    │                                                                          │
    │  The warning is technically correct but MISLEADING for a loop-breaker.  │
    │  VectorDelay is DESIGNED to run before its upstream gain.  The zero     │
    │  substitute is never actually used — delay ignores input_values[0] for  │
    │  its output (it returns last_output, not the current input).             │
    │                                                                          │
    │  The warning disappears at step k=1 because gain.output is no longer    │
    │  None — gain ran successfully at step k=0 and stored its result.         │
    │                                                                          │
    │  FIX IN simulation_engine.py — suppress the warning for LoopBreakers:  │
    │    Replace lines 1051-1056 with:                                         │
    │      if inp.output is not None:                                          │
    │          input_values.append(inp.output)                                 │
    │      elif isinstance(block, LoopBreaker):                                │
    │          input_values.append(VectorSignal([0.0]))  # silent: expected   │
    │      else:                                                               │
    │          self.logger.warning(                                            │
    │              f"  zero-fallback input: block '{block.name}' "            │
    │              f"← upstream '{inp.name}' has no output yet at t={t:.6f}"  │
    │          )                                                               │
    │          input_values.append(VectorSignal([0.0]))                        │
    │                                                                          │
    │  With this fix only genuine upstream failures (non-loop-breaker blocks   │
    │  with missing output) produce warnings.  Loop-breaker zero-fallbacks     │
    │  are expected by design and should be silent.                            │
    └──────────────────────────────────────────────────────────────────────────┘

  Phase 3 — Scope recording and ODE integration (after each compute pass)
  ────────────────────────────────────────────────────────────────────────
  After _compute_all_blocks():
    • scope.record(t) snapshots block.output.value for every registered block.
    • _integrate_dynamics_euler() or _integrate_dynamics_rk4() advances
      any dynamic (stateful) blocks — integrators, state-space models, etc.
      VectorDelay is NOT dynamic in the ODE sense; its state update happens
      inside compute_py() itself (it latches input as last_output).

  RK4 note: _integrate_dynamics_rk4() calls _compute_all_blocks() three
  extra times per step (for stages k2, k3, k4).  This re-triggers the
  zero-fallback warning three more times per step when VectorDelay is used
  — another reason the LoopBreaker suppression fix above is important.

Run:
    python example_algebraic_loop.py
"""

# =============================================================================
# Path bootstrap
# =============================================================================
import sys
from _path_utils import get_embedsim_import_path, get_project_root, get_current_parent

_HERE          = get_current_parent()
_embedsim_path = get_embedsim_import_path()
_PROJECT_ROOT  = get_project_root()

if _embedsim_path not in sys.path:
    sys.path.insert(0, _embedsim_path)

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
# PART 1 — Demonstrate algebraic loop detection
# =============================================================================
_SEP = "=" * 60

print(f"\n{_SEP}")
print("PART 1 — Algebraic loop (no delay in feedback)")
print(_SEP)
print("""
  Block diagram:
    [sin] ──► [sum] ──► [gain] ──┐
                ▲                │
                └────────────────┘

  gain needs sum's output to compute.
  sum  needs gain's output to compute.
  → circular dependency → no valid execution order exists.

  EmbedSim runs a two-pass DFS during __init__.
  Pass 1 collects all LoopBreaker blocks.
  Pass 2 detects back-edges with no LoopBreaker on the path → ValueError.
""")

sin_src = SinusoidalGenerator("sin",  AMP, FREQ, 0.0)
fb_sum  = VectorSum("sum",   signs=[1, 1])
fb_gain = VectorGain("gain", gain=K)
fb_out  = VectorEnd("output")

sin_src >> fb_sum
fb_sum  >> fb_gain
fb_gain >> fb_sum   # ← closes the algebraic loop (no LoopBreaker here)
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
    print()
    print("  ✓ The engine correctly refuses to build an algebraically")
    print("    inconsistent diagram.  Insert a VectorDelay to fix it.")


# =============================================================================
# PART 2 — Fix with VectorDelay (z⁻¹ loop breaker)
# =============================================================================
print(f"\n{_SEP}")
print("PART 2 — Loop broken by VectorDelay (z⁻¹)")
print(_SEP)
print(f"""
  Block diagram:

    [sin] ──► [sum] ──► [gain] ──► [output]
                ▲           │
           [delay] ◄────────┘    z⁻¹: outputs gain(k−1)

  VectorDelay(initial=[0.0]) is declared as a LoopBreaker.
  The DFS cuts the back-edge at the delay block and places it first in
  the execution order so its stored value is available when sum runs.

  Execution order resolved by DFS: delay → sin → sum → gain → output

  Each step k:
    1. delay  presents  gain(k−1)   (= 0.0 at k=0, initial value)
    2. sum    computes  sin(k) + gain(k−1)
    3. gain   computes  K · sum(k)
    4. output records   gain(k)
    delay latches gain(k) internally for the next step.

  Recurrence:  y(k) = K · [A·sin(2π·f·k·dt) + y(k−1)]
  Stability:   bounded for |K| < 1, diverges for |K| ≥ 1.
  Parameters:  f={FREQ} Hz   A={AMP}   K={K}   dt={DT} s   T={T_SIM} s
""")

# ── Build the corrected diagram ───────────────────────────────────────────────
sin_src   = SinusoidalGenerator("sin",   AMP, FREQ, 0.0)
fb_delay  = VectorDelay("delay", initial=[0.0])   # ← LoopBreaker, cuts the cycle
loop_sum  = VectorSum("sum",   signs=[1, 1])
loop_gain = VectorGain("gain", gain=K)
loop_out  = VectorEnd("output")

# Forward path
sin_src >> loop_sum >> loop_gain >> loop_out
# Feedback path — VectorDelay breaks the cycle
loop_gain >> fb_delay >> loop_sum

# ── Instantiate engine ────────────────────────────────────────────────────────
#
# NOTE ON THE ZERO-FALLBACK WARNING:
#   The engine will emit one WARNING at t=0.000000:
#     "zero-fallback input: block 'delay' ← upstream 'gain' has no output yet"
#
#   This is expected and BENIGN for a VectorDelay.  The delay block is placed
#   FIRST in execution order (before gain) by design.  It ignores the zero-
#   fallback entirely — it returns last_output = initial = [0.0] regardless
#   of what the upstream gain provides at this point.
#
#   The warning disappears from step k=1 onward because gain.output is then
#   available from the previous step.
#
#   Root cause: _compute_all_blocks() in simulation_engine.py does not
#   distinguish between LoopBreaker blocks (expected upstream None) and
#   ordinary blocks (unexpected upstream None).  The fix is a one-liner in
#   the engine — see the detailed explanation in this file's docstring above
#   (PART 3, "Why the zero-fallback warning fires").
#
#   Until that fix is applied the warning is informational only.  No
#   numerical error results from it.
#
sim = EmbedSim(
    sinks  = [loop_out],
    T      = T_SIM,
    dt     = DT,
    solver = ODESolver.RK4,
)
sim.scope.add(sin_src,   label="Sine")
sim.scope.add(loop_sum,  label="Sum")
sim.scope.add(loop_gain, label="Gain_out")
sim.scope.add(fb_delay,  label="Delay")

# ── Topology ──────────────────────────────────────────────────────────────────
print("Block diagram topology (execution order table):")
sim.print_topology()

print("\nBlock diagram topology (console ASCII):")
if sim.topo is not None:
    sim.topo.print_console()

_topo_html = _HERE / "example_algebraic_loop_topo.html"
if sim.topo is not None:
    sim.topo.export_html(str(_topo_html))
    print(f"\n  Topology HTML → {_topo_html}")


# =============================================================================
# PART 3 — Engine internals insight printed to console
# =============================================================================
print(f"\n{_SEP}")
print("PART 3 — Engine internals: DFS, execution order, zero-fallback")
print(_SEP)
print(f"""
  DFS traversal starting from sink 'output':
    → output needs gain     (recurse into gain)
    → gain   needs sum      (recurse into sum)
    → sum    needs sin      (recurse into sin)
      sin has no inputs → append sin to order  ✓
    → sum    needs delay    (delay IS a LoopBreaker)
      → insert delay immediately, do NOT follow delay.inputs  ✓
    → append sum to order  ✓
    → append gain to order  ✓
    → append output to order  ✓

  Resulting execution order: [delay, sin, sum, gain, output]

  At runtime, _compute_all_blocks() iterates this list left-to-right:

    Step 0 (t=0.000):
      Pre-init:  delay.output ← initial=[0.0]       (get_loop_breaking_output)
      Exec delay: gain.output is None → zero-fallback (BENIGN for LoopBreaker)
                  delay returns last_output=[0.0], latches [0.0] for next step
      Exec sin:   no inputs → computes sin(2π·2·0) = 0.0
      Exec sum:   inputs = [sin.output=0.0, delay.output=0.0] → sum=0.0
      Exec gain:  input  = [sum.output=0.0]  → gain=0.0
      Exec output: records 0.0

    Step 1 (t=0.010):
      Pre-init:  delay.output already set → skipped
      Exec delay: gain.output = 0.0 (from step 0)  → NO WARNING ✓
                  delay returns [0.0], latches [0.0]
      Exec sin:   sin(2π·2·0.01) = sin(0.1257) ≈ 0.1253
      Exec sum:   0.1253 + 0.0 = 0.1253
      Exec gain:  0.5 · 0.1253 = 0.0626
      Exec output: records 0.0626

  Note on RK4 and extra zero-fallback instances:
    RK4 calls _compute_all_blocks() three additional times per step for the
    intermediate k2, k3, k4 derivative evaluations.  Each sub-call at step 0
    also triggers the zero-fallback for delay, since gain.output gets reset
    in between sub-steps.  This accounts for any repeated warnings you may
    see at t=0.  All are benign.

  The single-line engine fix (simulation_engine.py, _compute_all_blocks):
    Change the zero-fallback guard from:
      else:   # inp.output is None
          self.logger.warning(...)
          input_values.append(VectorSignal([0.0]))
    To:
      elif isinstance(block, LoopBreaker):
          input_values.append(VectorSignal([0.0]))   # silent — by design
      else:
          self.logger.warning(...)
          input_values.append(VectorSignal([0.0]))

  This surgically suppresses noise for loop-breakers while preserving the
  warning for all other blocks where a missing upstream IS unexpected.
""")


# ── Run ───────────────────────────────────────────────────────────────────────
print("Running simulation...")
sim.run(verbose=False, progress_bar=True)
print(f"  Completed: {len(sim.scope.t)} steps\n")


# =============================================================================
# Plot
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
        f"y(k) = K·[A·sin(2π·{FREQ}·k·dt) + y(k−1)]   "
        f"K={K}  A={AMP}  f={FREQ} Hz"
    ),
    figsize   = (13, 5),
    save_path = _plot_overview,
    linewidth = 2.0,
)

# ── Grid: one subplot per signal ──────────────────────────────────────────────
ph.plot_grid(
    rows = [
        dict(
            signal = "Sine[0]",
            ylabel = "Amplitude  [–]",
            title  = f"[sin]  Source:  A·sin(2π·{FREQ}·t)    "
                     f"pure sinusoid, no feedback",
            color  = "#9CA3AF",
        ),
        dict(
            signal = "Sum[0]",
            ylabel = "Amplitude  [–]",
            title  = "[sum]  VectorSum:  sin(k) + gain(k−1)    "
                     "mixes current input with delayed feedback",
            color  = "#0891B2",
        ),
        dict(
            signal = "Gain_out[0]",
            ylabel = "Amplitude  [–]",
            title  = f"[gain]  VectorGain:  K·sum(k)   K={K}    "
                     f"scaled output — also feeds the delay",
            color  = "#2563EB",
        ),
        dict(
            signal    = "Delay[0]",
            ylabel    = "Amplitude  [–]",
            title     = "[delay]  VectorDelay:  gain(k−1)    "
                        "one step behind gain — the z⁻¹ loop breaker",
            color     = "#EA580C",
        ),
    ],
    title     = (
        f"EmbedSim — Algebraic Loop Fixed by VectorDelay (z⁻¹)\n"
        f"f={FREQ} Hz   A={AMP}   K={K}   dt={DT} s   T={T_SIM} s   "
        f"solver=RK4"
    ),
    figsize   = (13, 11),
    time_unit = "s",
    save_path = _plot_grid,
)

print(f"  Overview plot → {_plot_overview}")
print(f"  Grid plot     → {_plot_grid}")
print(f"\n{_SEP}\nDone.\n")
