====================================================================
Buck Converter PI Control — EmbedSim Educational Example
====================================================================
EmbedSim Framework — github.com/vectorsim/embed_sim_project
====================================================================

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUGGESTED LEARNING PATH  (read this first)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If you are new to EmbedSim or closed-loop power electronics simulation,
work through the material in this order:

  1. "WHAT THIS EXAMPLE DOES"   — overall picture (2 min)
  2. "THE PLANT — Buck Converter Physics" — understand what you control
  3. "THE CONTROLLER — PI Design"  — understand the algorithm
  4. "NUMERICAL METHODS"           — understand how the solver works
  5. "SYSTEM ARCHITECTURE"         — understand the software layers
  6. "HOW TO RUN"                  — actually run it
  7. "CODEGEN PIPELINE"            — understand the embedded C output
  8. "MODE 2 — AI TUNER"           — advanced: neural gain scheduling
  9. "DESIGN DECISIONS"            — why the code is structured as it is

Experienced readers can go straight to "HOW TO RUN" and refer to
the other sections as needed.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS EXAMPLE DOES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This example simulates a closed-loop DC-DC buck (step-down) power
converter. The goal: regulate the output voltage to exactly 12 V
even though the input is 24 V, the load may vary, and the control
must run in real time at 100 kHz on embedded hardware.

Closed-loop block diagram (both simulation modes):

  ┌──────────┐  V_ref   ┌──────────────┐  duty   ┌────────────┐
  │ VectorStep│─────────>│ PI Controller│─────────>│  Buck FMU  │──> V_out
  │  (12 V)  │          │  (pi_buck)   │          │  (plant)   │
  └──────────┘          └──────┬───────┘          └─────┬──────┘
                               │ V_meas                  │ V_out
                               └──── [one-step delay] <──┘

  KEY INSIGHT: the controller never "sees" the plant directly.
  It measures V_out (via the feedback delay), computes the error
  e = V_ref - V_out, and adjusts the duty cycle to drive e → 0.
  This is the fundamental principle of proportional-integral (PI)
  feedback control.

Two simulation modes are provided:

  Mode 1 — Fixed-gain PI demo
  ───────────────────────────
  Closed-loop voltage control with hand-tuned gains (Kp=0.15, Ki=8.0).
  Produces a time-domain plot, an interactive topology diagram,
  and auto-generated C code (embedsim_loop.c/.h) ready for AURIX TC38x.

  Mode 2 — FMU-Probed Neural PI Tuner
  ─────────────────────────────────────
  An advanced three-phase pipeline:
    Phase 1 — FMU Prober:
        Systematically sweeps (V_ref, R_load, V_in) × (Kp, Ki) over
        many short simulations to map "which gains are best for which
        operating point?"  Data is collected automatically.
    Phase 2 — Neural Network Training:
        Trains a small MLP  [V_ref, R_load, V_in] → [Kp*, Ki*].
        The network learns the gain-scheduling map from the probe data.
    Phase 3 — Closed-Loop Demonstration:
        Runs the full 10 ms simulation using the trained network to
        select gains in real time, then compares against the fixed-gain
        baseline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE PLANT — BUCK CONVERTER PHYSICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A buck converter is a switching power supply that steps DOWN a DC
voltage. The key components are an inductor (L) and a capacitor (C):

     V_in ─── [switch] ─── L ─── V_out ─── R_load ─── GND
                                  │
                                  C
                                  │
                                 GND

The switch opens/closes at the switching frequency (f_sw = 100 kHz).
The duty cycle d ∈ [0,1] is the fraction of each period the switch
is closed. In an ideal lossless converter, V_out = d × V_in.

AVERAGED CONTINUOUS MODEL (what the FMU simulates)
───────────────────────────────────────────────────
Rather than simulating each switching event (computationally expensive),
the FMU uses the "averaged" model that treats the duty cycle as a
continuous control input d ∈ [0,1]:

    L · dI_L/dt  = d · V_in - V_out       (inductor volt-second balance)
    C · dV_out/dt = I_L - V_out / R_load  (capacitor current balance)
    I_load        = V_out / R_load         (Ohm's law at the load)

  State variables : I_L (inductor current [A]), V_out (output voltage [V])
  Control input   : d (duty cycle [0-1])
  Disturbance     : R_load (load resistance [Ω])

  This is a 2nd-order linear time-invariant (LTI) system — ideal for
  teaching PI control and Laplace/transfer-function analysis.

OPEN-LOOP TRANSFER FUNCTION
────────────────────────────
Taking the Laplace transform with V_in as a scaling factor:

        V_out(s)       V_in
       ────────── = ──────────────────────────
          D(s)       LCs² + (L/R_load)s + 1

This is a second-order system with:
  - Natural frequency:  ω_n = 1/√(LC) = 1/√(100µH × 100µF) = 10,000 rad/s
  - Damping ratio:      ζ  = (1/R_load) × √(L/C) / 2 = 0.05   (lightly damped!)

A damping ratio of 0.05 means the open-loop plant is nearly undamped —
it would ring badly without control. The PI controller must add enough
phase margin to stabilise it. This is why Ki matters so much here.

PLANT DEFAULT PARAMETERS (BuckConverterBlock.DEFAULT_PARAMS)
────────────────────────────────────────────────────────────
  L      = 100 µH   Inductor — stores energy as magnetic field
  C      = 100 µF   Capacitor — stores energy as electric field; filters ripple
  R_load = 10  Ω    Nominal load — draws I_load = V_out / R_load ≈ 1.2 A at 12 V
  V_in   = 24  V    Source voltage — sets the step-down ratio: 12/24 = 0.5
  f_sw   = 100 kHz  Switching frequency (averaged model parameter only)

Inputs  : duty   ∈ [0, 1]    — PWM duty cycle (control signal)
Outputs : V_out  [V]          — output voltage (regulated output)
          I_L    [A]          — inductor current (observable state)
          I_load [A]          — load current (= V_out / R_load)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THE CONTROLLER — PI DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY PI (NOT JUST P)?
────────────────────
A proportional (P) controller alone drives the error toward zero but
cannot eliminate steady-state error when a constant disturbance exists
(e.g., load current). The integral term accumulates the error over time
and keeps pushing until e = 0 exactly — giving zero steady-state error
for constant references and disturbances. This is the "I" in PI.

DISCRETE-TIME PI ALGORITHM  (pi_buck_controller.c)
────────────────────────────────────────────────────
The C implementation uses forward Euler integration of the integrator:

  e(k)     = V_ref - V_meas                               (error)
  σ(k)     = clamp( σ(k-1) + e(k)·Ts,  -σ_lim, +σ_lim ) (integrator with anti-windup)
  duty(k)  = clamp( Kp·e(k) + Ki·σ(k),  duty_min, duty_max )  (output with saturation)

  where  σ_lim = duty_max / Ki   (anti-windup limit, see below)
         Ts    = 100 µs          (sample period = 1/f_sw)

ANTI-WINDUP
────────────
Without anti-windup, the integrator σ continues to accumulate error
even when the actuator is saturated (duty already at its maximum or
minimum). When the reference finally drops, the integrator takes a
long time to "unwind", causing large transients (integrator windup).

Anti-windup here clamps σ so that even at full saturation, the I-term
cannot push the output further past the clamp:

    σ_lim = duty_max / Ki

  This means: if Ki·σ_lim = duty_max, the integrator cannot push
  duty beyond duty_max regardless of how long the error persists.

OUTPUT SATURATION
─────────────────
The duty cycle is hard-clamped:
  duty_min = 0.10  (ensures the inductor current never fully collapses)
  duty_max = 0.90  (ensures the diode always has a chance to conduct)

GAIN TUNING  (for L=100µH, C=100µF, R_load=10Ω, V_in=24V)
────────────────────────────────────────────────────────────
  Kp = 0.15    Proportional gain [duty/V]
               A 1 V error produces a 0.15 duty cycle correction.
               Too high → oscillation. Too low → sluggish response.

  Ki = 8.0     Integral gain [duty/(V·s)]
               Accumulates 8.0 duty/V for every V·s of accumulated error.
               Controls settling time and steady-state error elimination.
               Too high → overshoot or instability.

  Settling time target: V_out reaches 12 V ± 2% within 3 ms after
  the 12 V reference step at t = 1 ms. Overshoot target: < 5%.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMERICAL METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EmbedSim solves the coupled plant+controller ODEs using the classic
4th-order Runge-Kutta (RK4) method with a fixed step dt = 1 µs.

WHY RK4 (NOT EULER)?
─────────────────────
Forward Euler:   x(t+dt) ≈ x(t) + f(x,t)·dt             (1st-order, O(dt²) error)
RK4:             x(t+dt) ≈ x(t) + (k1+2k2+2k3+k4)·dt/6  (4th-order, O(dt⁵) error)

For dt = 1 µs and a 10 ms simulation (10,000 steps), RK4's higher
accuracy means stable integration of the lightly-damped LC plant without
the numerical dissipation or instability that plagues Euler.

RK4 IN THE PI CONTROLLER
─────────────────────────
The PI integrator is a continuous state: dσ/dt = e(t) = V_ref - V_meas.
EmbedSim's RK4 solver calls PI_BuckBlock.get_derivative() to evaluate
k1, k2, k3, k4 for σ. This is more accurate than Euler integration
inside the C function, especially near the reference step at t=1 ms.

The C backend (pi_buck_wrapper.pyd) is kept in sync with the RK4 state
via an explicit two-way synchronisation every step:

    BEFORE compute:  push RK4 σ → C struct  (so C uses correct state)
    AFTER  compute:  pull C struct → RK4 σ  (so RK4 continues correctly)

ONE-STEP FEEDBACK DELAY
────────────────────────
The feedback path includes a ScalarDelay block that delays V_out by
exactly one time step (1 µs). This is intentional: it breaks the
algebraic loop that would arise if PI and plant were computed in the
same RK4 stage (each would need the other's output simultaneously).
In real hardware the feedback A/D conversion introduces a comparable
measurement delay, so this is physically motivated too.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SYSTEM ARCHITECTURE — SOFTWARE LAYERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The implementation has five clear layers. Understanding these layers
is essential for adapting the example to a new plant or controller.

  LAYER 1 — Physics model (Modelica / FMU)
  ─────────────────────────────────────────
  BuckConverter.mo     Averaged ODE model compiled by OpenModelica into
  BuckConverter.fmu    a Functional Mock-up Unit (FMU 2.0).
                       The FMU is a self-contained binary; EmbedSim
                       drives it via FMPy (FMI 2.0 standard API).

  LAYER 2 — Plant wrapper (Python)
  ──────────────────────────────────
  BuckConverterBlock.py    Thin VectorBlock wrapper around the FMU.
                           Declares INPUT_VARS, OUTPUT_VARS, DEFAULT_PARAMS
                           so EmbedSim's topology printer and scope can
                           name all signals automatically.

  LAYER 3 — Controller algorithm (C / Cython)
  ─────────────────────────────────────────────
  pi_buck_controller.c     MISRA C:2012 PI algorithm (bare-metal ready).
  pi_buck_controller.h     Struct definitions: PI_Buck_Block_T,
                           PI_Buck_Input_T, PI_Buck_Output_T.
  pi_buck_wrapper.pyx      Cython bridge: exposes PI_BuckWrapper Python
                           class that calls into the C struct directly.
                           Cython generates C, which is compiled into a
                           .pyd (Windows) / .so (Linux) extension module.

  LAYER 4 — Controller block (Python / EmbedSim)
  ─────────────────────────────────────────────────
  pi_buck_block.py         PI_BuckBlock — the EmbedSim VectorBlock that
                           wires the C algorithm into the simulation graph.
                           Manages RK4 state, dual backend (C/Python),
                           anti-windup, soft-start, and CodeGen attributes.
                           Also contains _PyPI_Buck — a pure-Python fallback
                           that exactly mirrors the C algorithm in float32.

  LAYER 5 — Simulation script
  ─────────────────────────────
  pi_buck_example.py       Instantiates blocks, wires them with >>,
                           runs the simulation, generates plots and C code.

DATA FLOW THROUGH THE LAYERS (one simulation step):

  EmbedSim RK4 engine
    │
    ├─ calls get_derivative()  on PI_BuckBlock  →  dσ/dt = V_ref - V_out
    │
    ├─ updates PI_BuckBlock.state[0] = σ(t+dt)  (RK4 integration)
    │
    └─ calls compute() on PI_BuckBlock:
         │
         ├─ [C backend]   push σ → C struct → PI_Buck_Compute() → pull σ
         └─ [Py backend]  read σ, apply anti-windup, compute duty
              │
              └─> duty → BuckConverterBlock (FMU) → [V_out, I_L, I_load]
                              │
                              └─> ScalarDelay → V_meas → PI_BuckBlock

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIMULATION SETTINGS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  T      =  10 ms    Total duration  (captures full settling transient)
  dt     =   1 µs    Time step       (100× oversampled vs Ts = 100 µs)
  Solver = RK4       4th-order Runge-Kutta (see "NUMERICAL METHODS")
  Step   : V_ref = 0 V → 12 V at t = 1 ms
           (1 ms of simulation at V_ref=0 lets the solver settle
            before the demanding step transient)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE STRUCTURE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  embed_sim_project/
  │
  ├── buck_converter/                     ← plant + controller source
  │   ├── BuckConverter.mo               Modelica averaged plant model
  │   ├── BuckConverterBlock.py          Auto-generated FMU wrapper [LAYER 2]
  │   ├── pi_buck_block.py               EmbedSim PI controller block [LAYER 4]
  │   ├── pyx_inspector.py               CodeGen attribute auto-populator
  │   ├── _path_utils.py                 Project root / import path helpers
  │   │
  │   ├── c_src/                         ← C algorithm [LAYER 3]
  │   │   ├── pi_buck_controller.c       PI algorithm (MISRA C:2012)
  │   │   ├── pi_buck_controller.h       Struct types and prototypes
  │   │   ├── pi_buck_wrapper.pyx        Cython bridge (C ↔ Python)
  │   │   ├── setup_pi_buck.py           Cython build script
  │   │   └── build_pi_buck.bat          Windows build shortcut
  │   │
  │   └── modelica/                      ← FMU artifacts [LAYER 1]
  │       ├── BuckConverter.fmu          Compiled FMU binary (OpenModelica)
  │       ├── BuckConverter.mo           Modelica source (same as above)
  │       └── gen_fmu.py                 Regenerates BuckConverterBlock.py
  │
  └── examples/pi_buck_converter_example/   ← simulation scripts [LAYER 5]
      ├── pi_buck_example.py             Canonical example (Mode 1 + Mode 2)
      ├── pi_buck_response.png           Output plot (generated on run)
      ├── pi_buck.html                   Interactive topology (generated)
      ├── _path_utils.py                 Copy of path helper for this folder
      │
      └── embedsim_gen/  (created on run)
          ├── embedsim_loop.c            Auto-generated C control loop
          └── embedsim_loop.h            Declarations for AURIX TC38x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW TO RUN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 — Compile the Cython wrapper (once, needed for C backend):
────────────────────────────────────────────────────────────────────
  Windows:
    cd buck_converter\c_src
    build_pi_buck.bat          (runs setup_pi_buck.py automatically)

  Linux / macOS:
    cd buck_converter/c_src
    python setup_pi_buck.py build_ext --inplace

  If compilation fails, the simulation still runs using the pure-Python
  backend (_PyPI_Buck). See "DUAL BACKEND" in "DESIGN DECISIONS" below.

Step 2 — Run the simulation:
────────────────────────────
  cd embed_sim_project
  python examples\pi_buck_converter_example\pi_buck_example.py

  You will be prompted to choose:
    [1] Fixed-gain PI demo
    [2] AI-tuned PI demo   (requires PyTorch; takes 1–4 min to probe)
    [3] Both

Step 3 — (Optional) Regenerate BuckConverterBlock.py if .mo changes:
──────────────────────────────────────────────────────────────────────
  python buck_converter\modelica\gen_fmu.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  pi_buck_response.png       3-panel time-domain plot:
                               Panel 1: V_out [V] vs time (with V_ref overlay)
                               Panel 2: duty cycle vs time
                               Panel 3: I_L [A] vs time

  pi_buck.html               Interactive block topology diagram.
                             Open in any browser to trace signal paths.

  embedsim_gen/
    embedsim_loop.c          Auto-generated C control loop for AURIX TC38x.
                             See "CODEGEN PIPELINE" below.
    embedsim_loop.h          Header: struct types, function prototype.

  ai_vs_fixed_comparison.png (Mode 2 only)
                             Side-by-side AI-tuned vs fixed-gain comparison
                             with performance metrics table.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED RESULTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  V_out settles to 12 V within ~3 ms after the 1 ms step (2 ms settling).
  Overshoot:           < 5%    (< 0.6 V above 12 V)
  Steady-state error:  < 10 mV (< 0.083%)
  I_L steady state:    ≈ 1.2 A  (= 12 V / 10 Ω)
  duty cycle steady:   ≈ 0.50   (= 12 V / 24 V)

  If you see oscillations that do not settle, the gains are too high or
  the Cython backend did not compile (check which backend is active).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CODEGEN PIPELINE  (embedsim_loop.c / embedsim_loop.h)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The LoopGenerator (aliased as StepGenerator in the code) walks the
sub-graph bounded by CodeGenStart and CodeGenEnd, and emits ISO C99
code that can be compiled directly by the TASKING CC-TC compiler for
AURIX TC38x.

HOW IT WORKS
────────────

  1. PYXInspector reads pi_buck_wrapper.pyx and extracts:
       step_func   = 'PI_Buck_Compute'
       state_struct= 'PI_Buck_Block_T'
       C_SOURCES   = ['pi_buck_controller.c']
       C_HEADERS   = ['pi_buck_controller.h']
       NUM_INPUTS  = 2  (V_ref, V_meas)
       OUTPUT_SIZE = 1  (duty)

  2. C_CUSTOM_EMIT is built from these names to produce:
       {
           PI_Buck_Input_T  u_pi_buck;
           PI_Buck_Output_T y_pi_buck;
           u_pi_buck.V_ref  = in->vref;
           u_pi_buck.V_meas = in->fb_delay;
           PI_Buck_Compute(&pi_buck_state, &u_pi_buck, dt, &y_pi_buck);
           out->pi_buck = y_pi_buck.duty;
       }

  3. LoopGenerator wraps this in a complete EmbedSim_Step() function
     with proper #includes, EmbedSim_Input_T, EmbedSim_Output_T,
     and the pi_buck_state static variable.

WHY C_CUSTOM_EMIT?
──────────────────
The generic code generator assumes blocks are called with flat arrays:
  step_func(u[0], u[1], ..., y[0], y[1], ...)
But PI_Buck_Compute() uses typed C structs:
  PI_Buck_Compute(PI_Buck_Block_T*, PI_Buck_Input_T*, real32_T, PI_Buck_Output_T*)
C_CUSTOM_EMIT is the escape hatch that allows verbatim struct-based
call emission. The simulation code and the generated embedded code
thus call the SAME pi_buck_controller.c object — zero divergence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE 2 — AI TUNER OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The neural PI tuner treats gain scheduling as a supervised learning
problem: "given an operating point, what gains minimise a cost?"

  Phase 1 — FMU Prober
  ─────────────────────
  For each combination of (V_ref, R_load, V_in):
    For each (Kp, Ki) in a grid:
      → Run a short mini-simulation (no scope, minimal overhead)
      → Evaluate a cost: weighted sum of (settling time, overshoot, SSE)
    → Record best (Kp*, Ki*) for this operating point
  Output: a dataset of (V_ref, R_load, V_in) → (Kp*, Ki*) pairs.

  Operating-point variation is achieved by patching
  BuckConverterBlock.DEFAULT_PARAMS before each FMU instantiation
  (the PMSM pattern: no constructor kwargs required).

  Phase 2 — Neural Network
  ─────────────────────────
  Architecture: MLP with 3 inputs → 64 → 64 → 2 outputs (Kp, Ki)
  Loss:         MSE on normalised gain values
  Training:     ~300 epochs, Adam optimiser

  The network learns a smooth gain surface over operating-point space.
  At runtime it can interpolate to untrained points — something a
  lookup table cannot do.

  Phase 3 — Closed-Loop Demo
  ───────────────────────────
  AI-Tuned PI: at each step, query net.predict(V_ref, R_load, V_in)
  and call pi_block.set_params(Kp=..., Ki=...) to update gains live.
  A load step (R_load: 10Ω → 5Ω at t=5ms) tests adaptation.

  The comparison plot shows settling time, overshoot, and steady-state
  error for AI-tuned vs fixed gains, with a performance metrics table.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DESIGN DECISIONS  (architectural rationale)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DUAL BACKEND (C + Python)
──────────────────────────
  WHY:  The Cython wrapper requires a C compiler and Cython installed.
        Not every user has these. The pure-Python fallback (_PyPI_Buck)
        produces bit-identical results (verified) at ~50× slower speed.
        For 10 ms at dt=1 µs (10,000 steps) the Python backend still
        runs in well under a second — acceptable for learning.
  HOW:  use_c_backend=True / False in PI_BuckBlock.__init__().
        The C backend syncs the RK4 state with the C struct every step
        (two-way push/pull). The Python backend reads the RK4 state
        directly and applies anti-windup inline.

float32 THROUGHOUT
───────────────────
  All computation uses np.float32 to match real32_T in the C struct.
  This ensures Python and C backends give bit-identical results on
  IEEE 754 compliant hardware (AURIX TriCore is IEEE 754).

RK4 STATE OWNERSHIP
─────────────────────
  EmbedSim's RK4 solver is the authoritative owner of PI_BuckBlock.state[0]
  (the integrator σ). The C wrapper (and Python backend) are NOT allowed
  to integrate independently — they only READ state[0] and WRITE duty.
  The C backend explicitly syncs: push σ → C before compute, pull σ ← C
  after compute. Without this sync the C struct would drift over time.

CODEGEN WITHOUT HARDCODING
────────────────────────────
  All names in C_CUSTOM_EMIT (step function, struct types) are derived
  from the .pyx file via PYXInspector at class-definition time. This
  means: if you rename PI_Buck_Compute → MyController_Step in the .pyx
  and .c, the generated C code automatically follows — no manual edits
  needed. This is the "single source of truth" principle.

FMU PARAMETER PATCHING
────────────────────────
  BuckConverterBlock.DEFAULT_PARAMS is patched directly before each
  FMU instantiation in the AI prober. This mirrors the PMSM pattern
  used throughout EmbedSim: plant parameters live in the class dict,
  not in constructor kwargs, so automated sweeps can instantiate
  many variants without changing any function signatures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TARGETS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Simulation : Windows / Linux  (Python 3.12, FMPy 0.3+, EmbedSim v3.1+)
               PyTorch required for Mode 2 only
  Embedded   : Infineon AURIX TC38x — embedsim_loop.c compiled with
               TASKING CC-TC (--iso=99, C99 mode)
               The same pi_buck_controller.c object file is used in
               both simulation and embedded targets.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEPENDENCIES SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Always required:
    Python 3.12+       Runtime
    NumPy              Array math (float32 everywhere)
    Matplotlib         Plotting (pi_buck_response.png)
    FMPy               FMI 2.0 FMU driver (drives BuckConverter.fmu)
    EmbedSim v3.1+     Simulation framework (this repo)

  For C backend (recommended, optional):
    Cython             Compiles pi_buck_wrapper.pyx → .pyd/.so
    C compiler         MSVC / GCC / Clang

  For Mode 2 — AI tuner (optional):
    PyTorch            Neural network training and inference

====================================================================
EmbedSim Framework — github.com/vectorsim/embed_sim_project
====================================================================
