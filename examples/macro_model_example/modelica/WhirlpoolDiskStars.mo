model SpiralGalaxy

  // ── Scalar parameters ───────────────────────────────────────────────
  parameter Integer N     = 500;   // number of stars
  parameter Integer M     = 4;     // number of spiral arms
  parameter Real    G     = 4.302e-6;   // kpc·(km/s)^2/M_sun
  parameter Real    Md    = 1.0e11;     // M_sun
  parameter Real    a     = 6.0;        // kpc  Miyamoto-Nagai scale length
  parameter Real    b     = 0.5;        // kpc  Miyamoto-Nagai scale height
  parameter Real    pitch = 0.28;       // rad/kpc  logarithmic spiral pitch
  parameter Real    r_min = 3.0;        // kpc  innermost star
  parameter Real    dr    = 0.15;       // kpc  radial spacing

  // ── IC helper arrays ───────────────────────────────────────────────
  parameter Real r_init[N]  = { r_min + dr*i  for i in 1:N };

  // Deterministic offsets for 4 spiral arms
  parameter Real arm_off[N] = {
    0.0 + 2*Modelica.Constants.pi*(i-1)/N* M  for i in 1:N
  };

  parameter Real theta[N]   = { pitch*r_init[i] + arm_off[i] for i in 1:N };
  parameter Real omega_c[N] = { sqrt(G*Md / (r_init[i]^2 + (a+b)^2)^1.5) for i in 1:N };

  // ── Continuous state variables ────────────────────────────────────
  Real x[N](each start=0.0);
  Real y[N](each start=0.0);
  Real z[N](each start=0.0);
  Real vx[N](each start=0.0);
  Real vy[N](each start=0.0);
  Real vz[N](each start=0.0);

  // ── Algebraic helper variables ──────────────────────────────────
  Real r2[N];
  Real sz[N];
  Real B[N];
  Real denom[N];

initial equation
  for i in 1:N loop
    // deterministic “scatter” using a tiny offset factor
    x[i] = r_init[i]*cos(theta[i] + 0.02*(i/N - 0.5));
    y[i] = r_init[i]*sin(theta[i] + 0.02*(i/N - 0.5));
    z[i] = 0.07*(i - N/2)/N + 0.01*(i/N - 0.5);

    vx[i] = -sin(theta[i])*r_init[i]*omega_c[i];
    vy[i] =  cos(theta[i])*r_init[i]*omega_c[i];
    vz[i] = 0.0;
  end for;

equation
  for i in 1:N loop
    r2[i]    = x[i]^2 + y[i]^2;
    sz[i]    = sqrt(z[i]^2 + b^2);
    B[i]     = a + sz[i];
    denom[i] = (r2[i] + B[i]^2)^1.5;

    der(x[i])  = vx[i];
    der(y[i])  = vy[i];
    der(z[i])  = vz[i];

    der(vx[i]) = -G*Md*x[i]/denom[i];
    der(vy[i]) = -G*Md*y[i]/denom[i];
    der(vz[i]) = -G*Md*B[i]*z[i]/(sz[i]*denom[i]);
  end for;

end SpiralGalaxy;
