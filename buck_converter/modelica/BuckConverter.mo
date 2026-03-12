model BuckConverter
  /*
  ============================================================
  BuckConverter.mo
  ============================================================

  WHAT IS THIS FILE?
  ------------------
  This is a Modelica source file describing the PHYSICAL PLANT
  of a buck (step-down) DC-DC converter.

  Modelica is an equation-based language — you write physical
  equations directly, and the compiler (OpenModelica) converts
  them into an FMU (Functional Mock-up Unit): a compiled
  binary (.fmu / .dll) that EmbedSim drives step-by-step via
  the FMPy library.

  WHAT IS A BUCK CONVERTER?
  -------------------------
  A buck converter is a switched-mode power supply that steps
  a higher voltage (V_in = 24 V) down to a lower regulated
  voltage (V_out ≈ duty × V_in).

  Ideal continuous averaged circuit:

      V_in ──[switch]──[L]──┬── V_out
                            C      R_load
                            │
                           GND

  The SWITCH is replaced by its averaged model:
      switch_state = duty   (0 to 1 continuous)

  This avoids the high-frequency switching discontinuity and
  gives smooth ODE dynamics suitable for EmbedSim's RK4 solver.

  STATE VARIABLES (continuous dynamic states)
  --------------------------------------------
  Two differential equations define the system:
      I_L(t)   — inductor current [A]      dI_L/dt = (switch_state*V_in - V_out) / L
      V_out(t) — output voltage [V]        dV_out/dt = (I_L - V_out/R_load) / C

  STEADY STATE (at constant duty, after transient settles)
  ---------------------------------------------------------
      V_out = duty × V_in                    e.g. 0.5 × 24 = 12 V
      I_L   = V_out / R_load                 e.g. 12 / 10 = 1.2 A
      I_load = I_L  (lossless model)

  HOW THIS FILE IS USED
  ---------------------
  1. OpenModelica compiles it → BuckConverter.fmu
  2. gen_fmu.py runs mo_to_fmu_client.py on THIS .mo file
     → generates BuckConverterBlock.py (the EmbedSim Python wrapper)
  3. pi_buck_example.py loads BuckConverter.fmu at runtime via FMPy,
     sets parameters (L, C, R_load, ...), and calls do_step() each tick.

  CORRECTNESS: EQUATIONS ARE CORRECT.
  ============================================================
  */
  "Simple buck converter plant model - Clean version without switch_state"

  // ── PARAMETERS ──────────────────────────────────────────────────────────────
  // Parameters are CONSTANT for the duration of a simulation run.
  // They are set by FMPy before the first do_step() call via FMUBlock.__init__.
  // Changing them mid-simulation requires an FMU reset.

  // Inductance: governs the rate of current rise/fall.
  // Larger L → slower current transients, less ripple, slower dynamics.
  // L * dI_L/dt = ... means dI_L/dt = .../L — larger L → smaller derivative.
  parameter Real L = 100e-6 "Inductance [H]";

  // Capacitance: governs the rate of voltage rise/fall.
  // Larger C → smoother V_out, slower step response.
  // C * dV_out/dt = ... means dV_out/dt = .../C — larger C → smaller derivative.
  parameter Real C = 100e-6 "Capacitance [F]";

  // Load resistance: determines steady-state current draw.
  // I_load = V_out / R_load.  Smaller R_load → heavier load → more current.
  parameter Real R_load = 10 "Load resistance [Ω]";

  // Supply voltage: the unregulated DC input rail.
  // Steady-state output: V_out = duty × V_in.
  parameter Real V_in = 24 "Input voltage [V]";

  // Switching frequency: included for completeness and FMU metadata.
  // In this AVERAGED model, the switch is not discretised at f_sw —
  // the continuous approximation (switch_state = duty) is used instead.
  // f_sw would matter in a switched (non-averaged) model.
  parameter Real f_sw = 100e3 "Switching frequency [Hz]";


  // ── INPUTS ──────────────────────────────────────────────────────────────────
  // Inputs are signals written by EmbedSim at the start of each do_step().
  // BuckConverterBlock.INPUT_VARS = ['duty'] matches this declaration.

  // The PI controller output — a value in [0, 1] representing the fraction
  // of time the ideal switch is closed each cycle.
  // 0.0 → switch always open → V_out falls to 0
  // 1.0 → switch always closed → V_out rises toward V_in
  input Real duty "PWM duty cycle [0-1]";


  // ── OUTPUTS ─────────────────────────────────────────────────────────────────
  // Outputs are read by EmbedSim after each do_step() call.
  // BuckConverterBlock.OUTPUT_VARS = ['V_out', 'I_L', 'I_load'] matches these.

  // The regulated output voltage — what the PI controller is trying to
  // hold at a setpoint (e.g., 12 V).
  // This is also an ODE state: C * dV_out/dt = I_L - V_out/R_load.
  output Real V_out "Output voltage [V]";

  // Inductor current — an ODE state.
  // L * dI_L/dt = switch_state * V_in - V_out.
  // In steady state: I_L ≈ V_out / R_load (capacitor carries no net DC).
  output Real I_L "Inductor current [A]";

  // Load current — purely algebraic (no ODE, just Ohm's law).
  // I_load = V_out / R_load, computed instantaneously each step.
  // In this lossless ideal model I_load = I_L in steady state.
  output Real I_load "Load current [A]";


  // ── PROTECTED SECTION ───────────────────────────────────────────────────────
  // Variables declared after 'protected' are INTERNAL to the model.
  // They are NOT exposed as FMU outputs and are NOT accessible via FMPy.
  // MoParser._is_in_protected_section() detects this and skips switch_state
  // when building the output variable list — preventing it from appearing
  // in BuckConverterBlock.OUTPUT_VARS.
protected
  // The averaged switch model maps duty → switch_state continuously.
  // In a real switched model, switch_state would be 0 or 1 (binary).
  // Here it equals duty exactly — this IS the averaging approximation.
  // Protected so the simulation user cannot accidentally read or set it
  // (it has no independent physical meaning beyond duty itself).
  Real switch_state;  // Internal variable - NOT exposed as output


  // ── EQUATIONS ───────────────────────────────────────────────────────────────
  // In Modelica, 'equation' declares mathematical relationships.
  // The compiler determines the solution order automatically — you do NOT
  // write sequential assignments; you write constraints and the tool solves them.
equation

  // IDEAL SWITCH MODEL
  // switch_state = duty implements the continuous averaged approximation.
  // It replaces the real PWM switch with a linear gain.
  // This is valid when the switching frequency (100 kHz) is much higher
  // than the control bandwidth (typically < 1 kHz for this converter).
  switch_state = duty;

  // INDUCTOR DYNAMICS  —  ODE 1 of 2
  // Physical origin: Faraday's law:  V_L = L * dI_L/dt
  // KVL around the switch-inductor-capacitor loop:
  //   V_in * switch_state - V_out = L * dI_L/dt
  // Rearranged to Modelica form:
  //   L * der(I_L) = switch_state * V_in - V_out
  // der(I_L) is Modelica notation for dI_L/dt.
  // When switch_state*V_in > V_out: current rises  (energy stored in L)
  // When switch_state*V_in < V_out: current falls  (L releases energy)
  L * der(I_L) = switch_state * V_in - V_out;

  // CAPACITOR DYNAMICS  —  ODE 2 of 2
  // Physical origin: KCL at the output node:
  //   I_L - I_load = C * dV_out/dt
  //   I_load = V_out / R_load  (Ohm's law on the resistive load)
  // Substituting:
  //   C * der(V_out) = I_L - V_out / R_load
  // When I_L > V_out/R_load: capacitor charges, V_out rises
  // When I_L < V_out/R_load: capacitor discharges, V_out falls
  C * der(V_out) = I_L - V_out / R_load;

  // LOAD CURRENT  —  algebraic equation (no ODE)
  // Ohm's law on the load resistor.
  // OpenModelica solves this instantaneously — no integration needed.
  // In steady state this equals I_L (the capacitor carries no net DC current).
  I_load = V_out / R_load;

  // ── INITIAL CONDITIONS ────────────────────────────────────────────────────
  // 'initial equation' sets the ODE state values at t=0 (cold start).
  // Both states start at zero — the converter begins with no stored energy.
  // FMPy passes these to the FMU during the initialize() call.
  // The PI controller's t_enable parameter in pi_buck_block.py can be used
  // to delay control action until the FMU has settled from t=0 conditions.
  initial equation
    I_L = 0;    // No current in the inductor at t=0
    V_out = 0;  // No voltage on the capacitor at t=0

end BuckConverter;
