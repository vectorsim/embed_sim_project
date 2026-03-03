// =============================================================================
// ThreePhaseMotor.mo  -  PMSM (Permanent Magnet Synchronous Motor) Model
// =============================================================================
//
// Suitable for export as FMU Co-Simulation from OpenModelica.
//
// PHYSICS:
//   d/q axis voltage equations (rotating reference frame):
//     v_d = R*i_d + L_d * di_d/dt - omega_e * L_q * i_q
//     v_q = R*i_q + L_q * di_q/dt + omega_e * (L_d * i_d + lambda_pm)
//
//   Electromagnetic torque:
//     T_e = 1.5 * p * (lambda_pm * i_q + (L_d - L_q) * i_d * i_q)
//
//   Mechanical:
//     J * domega_m/dt = T_e - B * omega_m - T_load
//     dtheta_e/dt     = p * omega_m
//
// INPUTS:
//   v_d    [V]   - d-axis voltage command
//   v_q    [V]   - q-axis voltage command
//   T_load [N.m] - load torque
//
// OUTPUTS:
//   i_d       [A]     - d-axis current
//   i_q       [A]     - q-axis current
//   omega_m   [rad/s] - mechanical angular velocity
//   theta_e   [rad]   - electrical rotor angle
//   T_em      [N.m]   - electromagnetic torque
//   speed_rpm [RPM]   - shaft speed
//
// PARAMETERS (all settable as FMU parameters):
//   R        - stator resistance  [Ohm]
//   L_d      - d-axis inductance  [H]
//   L_q      - q-axis inductance  [H]
//   lambda_pm- PM flux linkage    [Wb]
//   J        - rotor inertia      [kg.m²]
//   B        - viscous friction   [N.m.s/rad]
//   p        - pole pairs         [-]
//
// =============================================================================

model ThreePhaseMotor
  "PMSM 3-Phase Motor in dq0 Reference Frame for FMU Co-Simulation"

  // -------------------------------------------------------------------------
  // Parameters
  // -------------------------------------------------------------------------
  parameter Real R        = 0.5    "Stator resistance [Ohm]";
  parameter Real L_d      = 0.005  "d-axis inductance [H]";
  parameter Real L_q      = 0.006  "q-axis inductance [H] (L_q >= L_d for IPMSM)";
  parameter Real lambda_pm = 0.175 "Permanent magnet flux linkage [Wb]";
  parameter Real J        = 0.002  "Rotor inertia [kg.m2]";
  parameter Real B        = 0.001  "Viscous friction coefficient [N.m.s/rad]";
  parameter Real p        = 2.0    "Number of pole pairs [-]";

  // -------------------------------------------------------------------------
  // Inputs (FMU: causality = input)
  // -------------------------------------------------------------------------
  input Real v_d(start = 0.0)    "d-axis stator voltage [V]";
  input Real v_q(start = 0.0)    "q-axis stator voltage [V]";
  input Real T_load(start = 0.0) "Load torque [N.m]";

  // -------------------------------------------------------------------------
  // States
  // -------------------------------------------------------------------------
  Real i_d(start = 0.0, fixed = true)     "d-axis current [A]";
  Real i_q(start = 0.0, fixed = true)     "q-axis current [A]";
  Real omega_m(start = 0.0, fixed = true) "Mechanical angular velocity [rad/s]";
  Real theta_e(start = 0.0, fixed = true) "Electrical rotor angle [rad]";

  // -------------------------------------------------------------------------
  // Algebraic outputs
  // -------------------------------------------------------------------------
  Real T_em      "Electromagnetic torque [N.m]";
  Real omega_e   "Electrical angular velocity [rad/s]";
  Real speed_rpm "Shaft speed [RPM]";

equation
  // Electrical angular velocity
  omega_e = p * omega_m;

  // d-axis voltage equation
  L_d * der(i_d) = v_d - R * i_d + omega_e * L_q * i_q;

  // q-axis voltage equation
  L_q * der(i_q) = v_q - R * i_q - omega_e * (L_d * i_d + lambda_pm);

  // Electromagnetic torque (with reluctance torque for IPMSM)
  T_em = 1.5 * p * (lambda_pm * i_q + (L_d - L_q) * i_d * i_q);

  // Mechanical equation
  J * der(omega_m) = T_em - B * omega_m - T_load;

  // Electrical angle integration (wrap handled by sin/cos usage)
  der(theta_e) = omega_e;

  // RPM conversion
  speed_rpm = omega_m * 60.0 / (2.0 * Modelica.Constants.pi);

  annotation(
    experiment(
      StopTime    = 2.0,
      Interval    = 0.0001,
      Tolerance   = 1e-6
    ),
    Documentation(info = "
<html>
<h2>Three-Phase PMSM Motor Model</h2>
<p>
  Interior Permanent Magnet Synchronous Motor (IPMSM) modelled
  in the d-q rotating reference frame. Suitable for Field-Oriented
  Control (FOC) co-simulation via FMU export.
</p>
<h3>Export as FMU</h3>
<p>
  In OpenModelica (OMEdit or omc):
  <pre>
    loadFile(\"ThreePhaseMotor.mo\");
    buildModelFMU(ThreePhaseMotor,
                  version=\"2.0\",
                  fmuType=\"cs\",
                  fileNamePrefix=\"ThreePhaseMotor\");
  </pre>
  This produces <b>ThreePhaseMotor.fmu</b> ready for use with
  the EmbedSim FMUBlock.
</p>
</html>
    ")
  );

end ThreePhaseMotor;

// =============================================================================
// END OF MODEL
// =============================================================================
