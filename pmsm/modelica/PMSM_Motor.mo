model PMSM_Plant_FMU
  parameter Real R = 0.19;
  parameter Real L_d = 0.000125;
  parameter Real L_q = 0.000125;
  parameter Real lambda_pm = 0.0014;
  parameter Real J = 2.4e-6;
  parameter Real B_fric = 1e-6;
  parameter Integer p = 4;
  parameter Real v_dc_nom = 12.0;

  input Real duty_a(start=0.5);
  input Real duty_b(start=0.5);
  input Real duty_c(start=0.5);
  input Real v_dc(start=12.0);
  input Real T_load(start=0.0);

  Real i_d(start=0.0);
  Real i_q(start=0.0);
  Real omega_m(start=0.0);
  Real theta_e(start=0.0);

  output Real rpm;
  output Real ia;
  output Real ib;
  output Real ic;
  output Real theta_m;
  output Real T_em;
  output Real id_out = i_d;   // continuous — EmbedSim controls step rate
  output Real iq_out = i_q;

protected
  Real va_leg, vb_leg, vc_leg, v_neutral;
  Real va, vb, vc;
  Real v_alpha, v_beta, v_d, v_q;
  Real i_alpha, i_beta;
  Real omega_e;

equation
  // Star voltages with neutral subtraction
  va_leg  = duty_a * v_dc;
  vb_leg  = duty_b * v_dc;
  vc_leg  = duty_c * v_dc;
  v_neutral = (va_leg + vb_leg + vc_leg) / 3.0;
  va = va_leg - v_neutral;
  vb = vb_leg - v_neutral;
  vc = vc_leg - v_neutral;

  // Clarke — amplitude-invariant, matches embed_sim_coordinate_transform.c
  v_alpha = (2.0/3.0)*va - (1.0/3.0)*vb - (1.0/3.0)*vc;
  v_beta  = (vb - vc) / sqrt(3.0);

  // Park
  v_d =  v_alpha*cos(theta_e) + v_beta*sin(theta_e);
  v_q = -v_alpha*sin(theta_e) + v_beta*cos(theta_e);

  omega_e = p * omega_m;

  L_d*der(i_d) = v_d - R*i_d + omega_e*L_q*i_q;
  L_q*der(i_q) = v_q - R*i_q - omega_e*(L_d*i_d + lambda_pm);

  T_em = 1.5*p*(lambda_pm*i_q + (L_d - L_q)*i_d*i_q);

  J*der(omega_m) = T_em - B_fric*omega_m - T_load;
  der(theta_e)   = omega_e;

  // Inverse Park
  i_alpha = i_d*cos(theta_e) - i_q*sin(theta_e);
  i_beta  = i_d*sin(theta_e) + i_q*cos(theta_e);

  // Inverse Clarke
  ia =  i_alpha;
  ib = -0.5*i_alpha + (sqrt(3.0)/2.0)*i_beta;
  ic = -0.5*i_alpha - (sqrt(3.0)/2.0)*i_beta;

  theta_m = theta_e / p;
  rpm     = omega_m * 60.0 / (2.0*Modelica.Constants.pi);

end PMSM_Plant_FMU;