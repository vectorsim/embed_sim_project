================================================================================
  EmbedSim Example — DC Motor PID Speed Control
  examples/dc_motor_fmu/
================================================================================

OVERVIEW
--------
This example demonstrates the complete EmbedSim workflow for a closed-loop
speed-control problem:

    Simulate → Analyse → Plot → Export topology diagram

A brushed DC motor is modelled as a Modelica/FMU plant and driven by a
discrete-time PID controller implemented as a native EmbedSim VectorBlock.
The closed-loop diagram uses a one-step VectorDelay as the algebraic-loop
breaker on the feedback path.


FILES IN THIS DIRECTORY
-----------------------
  dc_motor_pid_example.py   Main simulation script.  Run this directly.
  _path_utils.py            Project-root locator; adds embedsim/ to sys.path.
                            Shared across all examples — do not edit.
  modelica/
    DCMotor.mo              Modelica source model (for reference / re-export).
    DCMotor.fmu             Pre-compiled FMI 2.0 co-simulation unit.
  read_me.txt               This file.


GENERATED OUTPUTS (written to the working directory when the script runs)
------------------------------------------------------------------------
  dc_motor_pid_response.png     Four-panel diagnostic plot + KPI table.
  dc_motor_pid_topology.html    Interactive signal-flow diagram (open in
                                any modern browser — no server required).


MOTOR MODEL  (DCMotor.fmu)
--------------------------
The FMU encapsulates a standard second-order armature-controlled DC motor:

    Electrical:   L · di/dt = u − R·i − k·ω
    Mechanical:   J · dω/dt = k·i − B·ω

Parameter values:
    R    =  1.0  Ω          Armature resistance
    L    =  0.5  H          Armature inductance
    k    =  0.01 V·s/rad    Back-EMF / torque constant
    J    =  0.01 kg·m²      Rotor inertia
    B    =  0.1  N·m·s/rad  Viscous friction

FMU interface:
    Input  : u  [V]       armature voltage
    Output : w  [rad/s]   shaft angular velocity

The electrical time constant  τ_e = L/R = 0.5 s is significantly larger than
the mechanical time constant  τ_m = J/B = 0.1 s, so the full second-order
dynamics are clearly visible in the transient response.


SIGNAL DIAGRAM
--------------

    [◈ reference]  ──► ω_ref (rad/s) ──►┐
                                          ├──► [⊕ error] ──► e (rad/s) ──► [⚡ pid] ──► u (V) ──► [⚙ dc_motor] ──► ω (rad/s) ──► [■ output]
    [z⁻¹ feedback] ──► ω_fb (rad/s) ──►┘                                                              └──► ω (rad/s) ──► [z⁻¹ feedback]

Block roles:
    reference   StepReference     Piecewise-constant ω_ref profile.
    error       VectorSum[+1,−1]  Computes tracking error  e = ω_ref − ω_fb.
    pid         PIDController     Anti-windup PID; outputs clamped voltage u.
    dc_motor    DCMotorFMU        FMUBlock wrapper for the Modelica plant.
    feedback    VectorDelay       One-step z⁻¹ delay; breaks the algebraic loop.
    output      VectorEnd         Marks the end of the forward path (logging sink).


DEFAULT REFERENCE PROFILE
--------------------------
    t = 0.0 s  →  ω_ref =   0 rad/s  (hold at rest)
    t = 1.0 s  →  ω_ref = 100 rad/s  (step up)
    t = 3.0 s  →  ω_ref =  50 rad/s  (step down)

Total simulation duration: 5.0 s,  timestep dt = 0.001 s,  RK4 solver.


DEFAULT PID TUNING
------------------
    Kp = 0.8   Proportional gain
    Ki = 3.0   Integral gain
    Kd = 0.05  Derivative gain

    Saturation          : ±24 V  (matches typical small-motor driver supply)
    Derivative filter α : 0.1   (heavy smoothing — avoids noise on derivative)
    Integral limit      : ±100  (symmetric clamp before Ki multiplication)
    Anti-windup method  : back-calculation  (active only when output is clipped)

These gains were selected empirically to give:
    - Rise time  ≈ 0.08 s  (10 %→ 90 % of 100 rad/s step)
    - Overshoot  < 5 %
    - Settling   < 0.5 s  (±2 % band)
    - Zero steady-state error at both operating points

Re-tuning guidance:
    - Increase Kp to reduce rise time; watch for overshoot.
    - Increase Ki to eliminate any residual steady-state offset.
    - Increase Kd only if overshoot is excessive; always keep α small.


STEP-RESPONSE KPIs  (computed automatically)
--------------------------------------------
The function analyse_step() evaluates:

    Rise time       Time for the response to travel from 10 % to 90 % of
                    the step final value.  Unit: s.

    Overshoot       Peak exceedance above the final value expressed as a
                    percentage of the final value.  Unit: %.

    Settling time   Time at which the response enters and permanently remains
                    within a ±2 % band around the final value.  Unit: s.

    Steady-state    Mean absolute deviation from the reference over the last
    error           20 % of the step segment.  Unit: rad/s.

    Peak voltage    Maximum absolute control effort over the whole run.  Unit: V.

    IAE             Integral Absolute Error  = ∫|e(t)| dt over the full run.
    ISE             Integral Squared Error   = ∫e²(t) dt over the full run.

All metrics are printed to stdout and displayed in the plot's KPI table.


QUICK-START
-----------
Prerequisites (see project pyproject.toml for pinned versions):
    Python 3.10+,  numpy,  matplotlib,  fmpy (or equivalent FMI loader)

Run from the example directory:

    python dc_motor_pid_example.py

Or from the project root:

    python examples/dc_motor_fmu/dc_motor_pid_example.py

Scripted parameter sweep (from another script or Jupyter notebook):

    from dc_motor_pid_example import main, SimConfig

    cfg      = SimConfig()
    cfg.Kp   = 1.2
    cfg.Ki   = 5.0
    cfg.Kd   = 0.02
    sim      = main(cfg)

    # Access raw scope data for further post-processing
    import numpy as np
    speed = np.asarray(sim.scope.data["Motor Speed[0]"])


HOW TO REGENERATE THE FMU
--------------------------
If you have OpenModelica installed:

    omc -s DCMotor.mo            # generates C code + Makefile
    make -f DCMotor.makefile     # compiles to DCMotor.fmu

Alternatively use the EmbedSim utility:

    python utility_functions/gen_fmu.py --model DCMotor.mo

The compiled FMU is already included so this step is only necessary if you
modify the Modelica model or target a different platform.


EXTENDING THIS EXAMPLE
----------------------
1.  Change the motor parameters
        Edit DCMotor.mo, regenerate DCMotor.fmu (see above), re-run the script.

2.  Try a different controller
        Replace PIDController with any VectorBlock that accepts e (rad/s)
        and emits u (V).  The wiring in build_diagram() remains identical.

3.  Add load torque disturbance
        Extend DCMotor.mo with a Modelica.Mechanics.Rotational.Sources.Torque
        block and wire it to the mechanical shaft.  Add a second FMU input
        variable and feed it from a new source block in build_diagram().

4.  Log internal PID signals (P, I, D terms)
        After build_diagram() but before sim.run():
            import functools
            sim.scope.add_custom(
                label  = "P term",
                getter = lambda: pid.P,
            )
        (Requires EmbedSim scope to support custom lambdas — see scope API.)


KNOWN LIMITATIONS
-----------------
  - The FMU uses a fixed-step internal solver; very large dt may cause
    numerical issues.  Keep dt ≤ 0.005 s.
  - DCMotor.fmu is compiled for the platform it was built on.  If you
    encounter a load error on a different OS, regenerate the FMU.
  - The derivative filter assumes constant dt.  Adaptive-step solvers are
    not currently supported in this example.


RELATED EXAMPLES
----------------
  examples/rlc_fmu/                 RLC circuit with PI, LQR, PIR and NN control.
  examples/pmsm_dfc_smo_example/    PMSM with Differential Flatness Controller + SMO.
  examples/pmsm_smc_smo_example/    PMSM with Sliding Mode Controller + SMO.
  examples/pi_buck_converter/       Buck converter with AI-tuned PI gains.


AUTHOR & LICENCE
----------------
  Paul Abraham  (Dipl.-Math.)
  EmbedSim project  —  github.com/vectorsim/embed_sim_project
  Licence: MIT  (see LICENSE in the project root)

================================================================================
