model ThreePhaseMotor_PWM_Sensorless
  "PMSM with separate physics and control paths for sensorless testing"

  // Parameters (same as before)
  parameter Real R = 0.5;
  parameter Real L_d = 0.005;
  parameter Real L_q = 0.006;
  parameter Real lambda_pm = 0.175;
  parameter Real J = 0.002;
  parameter Real B = 0.001;
  parameter Real p = 2.0;

  // Inputs
  input Real duty_a;
  input Real duty_b; 
  input Real duty_c;
  input Real v_dc;
  input Real T_load;
  input Real theta_e_est;  // Estimated angle from observer

  // States (true motor physics)
  Real i_d(start=0, fixed=true);
  Real i_q(start=0, fixed=true);
  Real omega_m(start=0, fixed=true);
  Real theta_e(start=0, fixed=true);  // True angle

  // Outputs (sensor measurements)
  output Real i_a;
  output Real i_b;
  output Real i_c;
  output Real theta_m;
  output Real emf_a;
  output Real emf_b;
  output Real emf_c;
  output Real speed_rpm;
  output Real T_em_out;

  // Intermediate
  Real v_a_leg, v_b_leg, v_c_leg;
  Real v_neutral;
  Real v_a, v_b, v_c;
  Real v_alpha, v_beta;
  Real v_d_cmd_ctrl, v_q_cmd_ctrl;  // For controller path
  Real v_d_cmd_actual, v_q_cmd_actual;  // For actual physics
  Real omega_e;
  Real T_em;
  Real i_alpha, i_beta;

equation
  // 1. Inverter model (same)
  v_a_leg = duty_a * v_dc;
  v_b_leg = duty_b * v_dc;
  v_c_leg = duty_c * v_dc;
  v_neutral = (v_a_leg + v_b_leg + v_c_leg) / 3.0;
  v_a = v_a_leg - v_neutral;
  v_b = v_b_leg - v_neutral;
  v_c = v_c_leg - v_neutral;

  // 2. Clarke transform (same)
  v_alpha = v_a;
  v_beta = (v_b - v_c) / sqrt(3.0);

  // 3. Park transforms - TWO different ones!
  //    Controller path: uses ESTIMATED angle
  v_d_cmd_ctrl =  v_alpha * cos(theta_e_est) + v_beta * sin(theta_e_est);
  v_q_cmd_ctrl = -v_alpha * sin(theta_e_est) + v_beta * cos(theta_e_est);
  
  //    Actual physics: uses TRUE angle
  v_d_cmd_actual =  v_alpha * cos(theta_e) + v_beta * sin(theta_e);
  v_q_cmd_actual = -v_alpha * sin(theta_e) + v_beta * cos(theta_e);

  // 4. Electrical dynamics (use ACTUAL voltages)
  omega_e = p * omega_m;
  L_d * der(i_d) = v_d_cmd_actual - R * i_d + omega_e * L_q * i_q;
  L_q * der(i_q) = v_q_cmd_actual - R * i_q - omega_e * (L_d * i_d + lambda_pm);

  // 5. Torque and mechanics (same)
  T_em = 1.5 * p * (lambda_pm * i_q + (L_d - L_q) * i_d * i_q);
  J * der(omega_m) = T_em - B * omega_m - T_load;
  der(theta_e) = omega_e;

  // 6. Current outputs (use TRUE angle for inverse Park)
  i_alpha = i_d * cos(theta_e) - i_q * sin(theta_e);
  i_beta = i_d * sin(theta_e) + i_q * cos(theta_e);
  
  i_a = i_alpha;
  i_b = -0.5 * i_alpha + (sqrt(3.0) / 2.0) * i_beta;
  i_c = -0.5 * i_alpha - (sqrt(3.0) / 2.0) * i_beta;

  // 7. Resolver output
  theta_m = theta_e / p;

  // 8. Back-EMF (based on TRUE angle)
  emf_a = lambda_pm * omega_e * sin(theta_e);
  emf_b = lambda_pm * omega_e * sin(theta_e - 2.0 * Modelica.Constants.pi / 3.0);
  emf_c = lambda_pm * omega_e * sin(theta_e + 2.0 * Modelica.Constants.pi / 3.0);

  // 9. Diagnostics
  speed_rpm = omega_m * 60.0 / (2.0 * Modelica.Constants.pi);
  T_em_out = T_em;

end ThreePhaseMotor_PWM_Sensorless;