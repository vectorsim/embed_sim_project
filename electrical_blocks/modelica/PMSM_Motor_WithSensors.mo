// =============================================================================
// PMSM_Motor_WithSensors.mo  -  PMSM Motor with Realistic Sensor Outputs
// =============================================================================
//
// This model represents a REAL PMSM motor with physical sensors:
// - Current sensors (i_a, i_b, i_c) - measures actual phase currents
// - Resolver (theta_m, omega_m) - measures actual mechanical position/speed
// - Back-EMF sensors (emf_a, emf_b, emf_c) - measures actual back-EMF voltage
//
// =============================================================================

model PMSM_Motor_WithSensors
  "PMSM with realistic sensor outputs - currents, position, back-EMF"

  // ---------------------------------------------------------------------------
  // Motor Parameters
  // ---------------------------------------------------------------------------
  parameter Real R = 0.5 "Stator phase resistance [Ohm]";
  parameter Real L_d = 0.005 "d-axis inductance [H]";
  parameter Real L_q = 0.006 "q-axis inductance [H]";
  parameter Real lambda_pm = 0.175 "PM flux linkage [Wb]";
  parameter Real J = 0.002 "Rotor inertia [kg.m2]";
  parameter Real B = 0.001 "Viscous friction [N.m.s/rad]";
  parameter Real p = 2.0 "Number of pole pairs [-]";

  // ---------------------------------------------------------------------------
  // Inputs (from controller)
  // ---------------------------------------------------------------------------
  input Real duty_a(start = 0.5) "PWM duty cycle phase A [0..1]";
  input Real duty_b(start = 0.5) "PWM duty cycle phase B [0..1]";
  input Real duty_c(start = 0.5) "PWM duty cycle phase C [0..1]";
  input Real v_dc(start = 48.0) "DC bus voltage [V]";
  input Real T_load(start = 0.0) "Load torque [N.m]";

  // ---------------------------------------------------------------------------
  // States - Actual motor states (true values)
  // ---------------------------------------------------------------------------
  Real i_d(start = 0.0, fixed = true) "d-axis current [A]";
  Real i_q(start = 0.0, fixed = true) "q-axis current [A]";
  Real omega_m(start = 0.0, fixed = true) "Actual mechanical speed [rad/s]";
  Real theta_e(start = 0.0, fixed = true) "Actual electrical angle [rad]";

  // ---------------------------------------------------------------------------
  // OUTPUTS - Actual sensor measurements
  // ---------------------------------------------------------------------------
  // Current sensors (3-phase)
  output Real i_a "Phase A current sensor [A] - ACTUAL";
  output Real i_b "Phase B current sensor [A] - ACTUAL";
  output Real i_c "Phase C current sensor [A] - ACTUAL";

  // Resolver outputs (mechanical position/speed)
  output Real theta_m "Resolver: mechanical angle [rad] - ACTUAL";
  output Real omega_m_out "Resolver: mechanical speed [rad/s] - ACTUAL";
  
  // Back-EMF sensors (phase back-EMF voltages)
  output Real emf_a "Phase A back-EMF [V] - ACTUAL";
  output Real emf_b "Phase B back-EMF [V] - ACTUAL";
  output Real emf_c "Phase C back-EMF [V] - ACTUAL";

  // Diagnostic outputs
  output Real speed_rpm "Shaft speed [RPM] - ACTUAL";
  output Real T_em_out "Electromagnetic torque [N.m] - ACTUAL";

  // ---------------------------------------------------------------------------
  // Internal variables
  // ---------------------------------------------------------------------------
  Real v_a_leg, v_b_leg, v_c_leg;      // Inverter leg voltages
  Real v_neutral;                        // Virtual neutral point
  Real v_a, v_b, v_c;                    // Phase voltages
  Real v_alpha, v_beta;                   // Alpha-beta voltages
  Real v_d, v_q;                           // dq voltages (FIXED: added these)
  Real i_alpha, i_beta;                    // Alpha-beta currents
  Real omega_e;                            // Electrical speed
  Real T_em;                                // Electromagnetic torque

equation
  // ===========================================================================
  // 1. INVERTER MODEL - Convert PWM duties to phase voltages
  // ===========================================================================
  v_a_leg = duty_a * v_dc;
  v_b_leg = duty_b * v_dc;
  v_c_leg = duty_c * v_dc;
  
  // Virtual neutral point (star connection)
  v_neutral = (v_a_leg + v_b_leg + v_c_leg) / 3.0;
  
  // Phase voltages relative to neutral
  v_a = v_a_leg - v_neutral;
  v_b = v_b_leg - v_neutral;
  v_c = v_c_leg - v_neutral;

  // ===========================================================================
  // 2. CLARKE TRANSFORM (abc -> αβ)
  // ===========================================================================
  v_alpha = (2.0/3.0)*(v_a - 0.5*v_b - 0.5*v_c);
  v_beta = (2.0/3.0)*((sqrt(3.0)/2.0)*v_b - (sqrt(3.0)/2.0)*v_c);

  // ===========================================================================
  // 3. PARK TRANSFORM (αβ -> dq) - Using ACTUAL rotor angle
  // ===========================================================================
  v_d =  v_alpha * cos(theta_e) + v_beta * sin(theta_e);
  v_q = -v_alpha * sin(theta_e) + v_beta * cos(theta_e);

  // ===========================================================================
  // 4. ELECTRICAL DYNAMICS - Actual motor physics
  // ===========================================================================
  omega_e = p * omega_m;
  
  // dq voltage equations
  L_d * der(i_d) = v_d - R * i_d + omega_e * L_q * i_q;
  L_q * der(i_q) = v_q - R * i_q - omega_e * (L_d * i_d + lambda_pm);

  // ===========================================================================
  // 5. ELECTROMAGNETIC TORQUE
  // ===========================================================================
  T_em = 1.5 * p * (lambda_pm * i_q + (L_d - L_q) * i_d * i_q);

  // ===========================================================================
  // 6. MECHANICAL DYNAMICS
  // ===========================================================================
  J * der(omega_m) = T_em - B * omega_m - T_load;
  der(theta_e) = omega_e;

  // ===========================================================================
  // 7. INVERSE PARK (dq -> αβ) - For current calculation
  // ===========================================================================
  i_alpha = i_d * cos(theta_e) - i_q * sin(theta_e);
  i_beta  = i_d * sin(theta_e) + i_q * cos(theta_e);

  // ===========================================================================
  // 8. INVERSE CLARKE (αβ -> abc) - Actual phase currents
  // ===========================================================================
  i_a = i_alpha;
  i_b = -0.5 * i_alpha + (sqrt(3.0) / 2.0) * i_beta;
  i_c = -0.5 * i_alpha - (sqrt(3.0) / 2.0) * i_beta;

  // ===========================================================================
  // 9. RESOLVER OUTPUT - Actual mechanical position and speed
  // ===========================================================================
  theta_m = theta_e / p;
  omega_m_out = omega_m;

  // ===========================================================================
  // 10. BACK-EMF OUTPUT - Actual back-EMF voltages
  // ===========================================================================
  emf_a = lambda_pm * omega_e * sin(theta_e);
  emf_b = lambda_pm * omega_e * sin(theta_e - 2.0 * Modelica.Constants.pi / 3.0);
  emf_c = lambda_pm * omega_e * sin(theta_e + 2.0 * Modelica.Constants.pi / 3.0);

  // ===========================================================================
  // 11. DIAGNOSTIC OUTPUTS
  // ===========================================================================
  speed_rpm = omega_m * 60.0 / (2.0 * Modelica.Constants.pi);
  T_em_out = T_em;

  annotation(
    experiment(
      StopTime = 2.0,
      Interval = 0.0001,
      Tolerance = 1e-6
    ),
    Documentation(info = "
<html>
<h2>PMSM Motor with Realistic Sensor Outputs</h2>
<p>This model represents a REAL PMSM motor with physical sensors.</p>
</html>
    ")
  );

end PMSM_Motor_WithSensors;
