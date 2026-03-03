/*
  SpiralGalaxy.mo  — v2  (OpenModelica / Dymola compatible)
  ===========================================================

  Fix applied vs. v1
  ------------------
  OpenModelica error:  "Too few equations — under-determined system.
  Variable r_init[i] / theta[i] / arm_off[i] / omega_c[i] does not
  have any remaining equation to be solved in."

  Root cause
  ----------
  In Modelica every Real variable that is NOT a state variable must have
  one algebraic equation in the 'equation' section.  r_init, theta,
  arm_off, and omega_c were declared as "Real" but assigned only inside
  "initial equation".  OpenModelica correctly counts them as unknowns
  with no continuous-time equation → under-determined DAE.

  Fix
  ---
  Promote the four IC-only helper arrays to "parameter Real".
  Parameters are resolved once at compile/init time — no equation needed
  in the continuous section.  This removes the 4·N = 240 missing equations
  and brings the count back to a well-determined system
  (600 equations, 600 unknowns for N=60).

  Physics corrections (unchanged from v1)
  ----------------------------------------
  1. der(vz): restored sz = sqrt(z²+b²) in denominator  (MN potential).
  2. Initial ω_c: Miyamoto-Nagai formula, not Keplerian sqrt(GMd/r³).
  3. Two clean logarithmic spiral arms via 'pitch' parameter.

  Compilation
  -----------
  OpenModelica GUI:
    File → Open → SpiralGalaxy.mo
    Simulate → Translate (check only)
    FMI → Export FMU (Co-Simulation, FMI 2.0)

  OpenModelica scripting (omc):
    loadFile("SpiralGalaxy.mo");
    buildModelFMU(SpiralGalaxy, version="2.0", fmuType="cs");

  Dymola:
    File → Export → FMU (Co-Simulation, FMI 2.0)
*/

model SpiralGalaxy

  // ── Scalar parameters ────────────────────────────────────────────────────
  parameter Integer N     = 60;
  parameter Real    G     = 4.302e-6;   // kpc·(km/s)²/M_sun
  parameter Real    Md    = 1.0e11;     // M_sun
  parameter Real    a     = 6.0;        // kpc  Miyamoto-Nagai scale length
  parameter Real    b     = 0.5;        // kpc  Miyamoto-Nagai scale height
  parameter Real    pitch = 0.28;       // rad/kpc  logarithmic spiral pitch
  parameter Real    r_min = 3.0;        // kpc  radius of innermost star
  parameter Real    dr    = 0.35;       // kpc  radial spacing between stars

  // ── IC helper arrays  →  declared as 'parameter', not 'Real'  ──────────
  //
  // FIX:  These quantities are used only to compute initial conditions.
  //       Declaring them as "Real" forces OpenModelica to look for a
  //       continuous-time equation for each element → under-determined DAE.
  //       "parameter Real" means: evaluate once at start, no ODE/algebraic
  //       equation needed in the simulation section.
  //
  parameter Real r_init[N]  = {r_min + dr * i          for i in 1:N};
  parameter Real arm_off[N] = {if mod(i,2)==0
                                then Modelica.Constants.pi
                                else 0.0                for i in 1:N};
  parameter Real theta[N]   = {pitch * r_init[i] + arm_off[i]
                                                        for i in 1:N};
  parameter Real omega_c[N] = {sqrt(G * Md /
                                (r_init[i]^2 + (a+b)^2)^1.5)
                                                        for i in 1:N};

  // ── Continuous state variables (ODE) ────────────────────────────────────
  Real x[N](  each start = 0.0);
  Real y[N](  each start = 0.0);
  Real z[N](  each start = 0.0);
  Real vx[N]( each start = 0.0);
  Real vy[N]( each start = 0.0);
  Real vz[N]( each start = 0.0);

  // ── Algebraic helper variables (defined by equations below) ─────────────
  Real r2[N];      // in-plane R²
  Real sz[N];      // sqrt(z² + b²)
  Real B[N];       // a + sz  (Miyamoto-Nagai B factor)
  Real denom[N];   // (R² + B²)^(3/2)

initial equation
  // ── Apply initial conditions ─────────────────────────────────────────────
  // Parameters (r_init, theta, arm_off, omega_c) are already resolved.
  // Only the state variables need explicit initial values.
  for i in 1:N loop
    x[i]  =  r_init[i] * cos(theta[i]);
    y[i]  =  r_init[i] * sin(theta[i]);
    z[i]  =  0.07 * (i - N / 2.0) / N;          // thin-disk z scatter

    vx[i] = -sin(theta[i]) * r_init[i] * omega_c[i];
    vy[i] =  cos(theta[i]) * r_init[i] * omega_c[i];
    vz[i] =  0.0;
  end for;

equation
  for i in 1:N loop
    // ── Algebraic definitions ─────────────────────────────────────────────
    r2[i]    = x[i]^2 + y[i]^2;
    sz[i]    = sqrt(z[i]^2 + b^2);               // sqrt(z² + b²)
    B[i]     = a + sz[i];                         // Miyamoto-Nagai B
    denom[i] = (r2[i] + B[i]^2)^1.5;             // (R² + B²)^(3/2)

    // ── Equations of motion ───────────────────────────────────────────────
    der(x[i])  =  vx[i];
    der(y[i])  =  vy[i];
    der(z[i])  =  vz[i];

    //  ∂Φ/∂x
    der(vx[i]) = -G * Md * x[i] / denom[i];

    //  ∂Φ/∂y
    der(vy[i]) = -G * Md * y[i] / denom[i];

    //  ∂Φ/∂z  — CORRECTED: sz in denominator (was missing in original)
    der(vz[i]) = -G * Md * B[i] * z[i] / (sz[i] * denom[i]);
  end for;

end SpiralGalaxy;
