================================================================================
EmbedSim — Algebraic Loop: Detection, Resolution & Engine Internals
================================================================================
Folder:  examples/algebraic_loop/
Script:  example_algebraic_loop.py
Version: EmbedSim 3.1.0


WHAT THIS EXAMPLE TEACHES
──────────────────────────
This is the canonical introductory example for EmbedSim.  It covers the three
concepts every user must understand before building real block diagrams:

  1. What an algebraic loop is and why it makes simulation impossible.
  2. How VectorDelay (z⁻¹) breaks a feedback loop cleanly.
  3. How the EmbedSim engine resolves execution order internally (DFS / SCC).


THE BLOCK DIAGRAM
─────────────────
PART 1 — Broken diagram (algebraic loop, no delay):

    [sin] ──► [sum] ──► [gain] ──┐
                ▲                │
                └────────────────┘   ← circular dependency, no valid order

  Both blocks need each other's output within the same timestep.
  EmbedSim raises ValueError at __init__ time — no simulation step runs.

PART 2 — Fixed diagram (VectorDelay in feedback path):

    [sin] ──► [sum] ──► [gain] ──► [output]
                ▲           │
           [delay] ◄────────┘    z⁻¹ (outputs gain from PREVIOUS step)

  VectorDelay is a LoopBreaker.  The DFS cuts the back-edge at the delay
  block and produces a valid sequential execution order:

    delay → sin → sum → gain → output

  Mathematical recurrence:
    y(k) = K · [ A·sin(2π·f·k·dt)  +  y(k−1) ]

  This is a first-order IIR filter driven by a sine wave.
  Stable for |K| < 1, diverges for |K| ≥ 1.

Parameters:
    f   = 2.0 Hz     sine frequency
    A   = 1.0        sine amplitude
    K   = 0.5        feedback gain  (|K| < 1 → stable)
    dt  = 0.01 s     timestep
    T   = 3.0 s      simulation duration
    ODE = RK4        solver


THE ZERO-FALLBACK WARNING (expected — not a bug)
─────────────────────────────────────────────────
At t = 0 the engine emits:

    WARNING: zero-fallback input: block 'delay' ← upstream 'gain'
             has no output yet at t=0.000000

This is expected and numerically benign.  The delay block is placed FIRST
in the execution order by design (it is a LoopBreaker).  When the forward
pass reaches it, gain.output is still None because gain has not run yet.
The engine substitutes [0.0] as a fallback, but VectorDelay never uses that
value — it returns its stored initial value [0.0] regardless of the input.

ENGINE INTERNALS — DFS EXECUTION ORDER
───────────────────────────────────────
EmbedSim.__init__ calls traverse_blocks_from_sinks_with_loops() which
performs a two-pass Depth-First Search (DFS):

  Pass 1 — find_loop_breakers:
    Walks the entire graph and collects every block that inherits from
    LoopBreaker into the loop_breakers set.

  Pass 2 — post-order DFS:
    Walks from every sink, following block.inputs[] recursively.
    Rules:
      • Block already fully visited (BLACK) → skip.
      • Block on the current DFS stack (GREY) AND not a LoopBreaker
        → true algebraic loop → ValueError raised immediately.
      • Block is a LoopBreaker → insert into order NOW, do NOT recurse
        into its inputs (the back-edge is cut).
      • Otherwise → recurse into inputs first, then append block (post-order).

  For our 5-block diagram the DFS produces:
    [delay, sin, sum, gain, output]

  Execution trace for step k=0:
    Pre-init: delay.output ← VectorSignal([0.0])  (initial value)
    1. delay  → returns [0.0]  (initial), latches [0.0] for next step
    2. sin    → A·sin(2π·2·0) = 0.0
    3. sum    → 0.0 + 0.0 = 0.0
    4. gain   → 0.5 · 0.0 = 0.0
    5. output → records 0.0

  Execution trace for step k=1 (t=0.01 s):
    1. delay  → returns [0.0]  (latched from step 0), latches [0.0]
    2. sin    → A·sin(2π·2·0.01) ≈ 0.1253
    3. sum    → 0.1253 + 0.0 = 0.1253
    4. gain   → 0.5 · 0.1253 = 0.0626
    5. output → records 0.0626


FILES IN THIS FOLDER
────────────────────
  example_algebraic_loop.py        Main example script
  algebraic_loop.sh                Linux/macOS runner (menu shell script)
  algebraic_loop.cmd               Windows runner
  example_algebraic_loop.png       Grid plot  (4 subplots, one per signal)
  example_algebraic_loop_overview.png  Overview plot (all signals overlaid)
  example_algebraic_loop_topo.html     Interactive topology diagram (browser)
  _path_utils.py                   Project-root path resolver (shared utility)
  read_me/read_me.txt              This file


HOW TO RUN
──────────
  Linux / macOS:
    chmod +x algebraic_loop.sh
    ./algebraic_loop.sh

  Windows:
    algebraic_loop.cmd

  Direct (any OS):
    python example_algebraic_loop.py

  The script must be run from the examples/algebraic_loop/ directory,
  OR from any directory — _path_utils.py resolves the project root
  automatically via the .project_root_marker file.


EXPECTED OUTPUT
───────────────
  Console:
    • PART 1 — ValueError message confirming algebraic loop detection
    • PART 2 — Topology table + ASCII diagram
    • PART 3 — DFS step-by-step execution trace (printed)
    • WARNING at t=0 (benign, see explanation above)
    • Simulation progress bar → 100%
    • Signal statistics table

  Files written:
    • example_algebraic_loop_overview.png
    • example_algebraic_loop.png
    • example_algebraic_loop_topo.html

  Performance (typical):
    300 steps  ·  RK4  ·  wall ≈ 0.14 s  ·  avg ≈ 470 µs/step


KEY CONCEPTS — QUICK REFERENCE
───────────────────────────────
  Algebraic loop:   A cycle in the block graph with no LoopBreaker.
                    Makes topological sorting impossible → ValueError.

  LoopBreaker:      Mixin class (simulation_engine.py).  Marks a block as
                    a z⁻¹ element.  DFS cuts back-edges at these blocks.

  VectorDelay:      Concrete LoopBreaker.  y(k) = u(k−1).
                    Stores one timestep of memory.  initial=[] sets y(0).

  VectorScope:      Signal recorder.  scope.add(block, label=...) before
                    run(), then scope.get_signal(label) after run().

  ODESolver.RK4:    Fourth-order Runge-Kutta.  Calls _compute_all_blocks()
                    four times per step.  Use for smooth accuracy.

  ODESolver.EULER:  Forward Euler.  One compute pass per step.  Fast,
                    first-order accurate.  Sufficient for discrete recurrences
                    like the IIR in this example.


AUTHOR / PROJECT
────────────────
  EmbedSim Framework — open-source Python-native simulation and
  MISRA C:2012 code generation for embedded targets.
  GitHub: https://github.com/vectorsim/embed_sim_project

================================================================================
