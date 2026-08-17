/**********************************************************************************************************************
 * PMSM_Plant_FMU
 *
 * Surface-mounted PMSM with:
 *
 *   - 20 kHz center-aligned PWM
 *   - 3-phase, 2-level inverter
 *   - Configurable inverter dead time
 *   - Floating motor neutral
 *   - Amplitude-invariant Clarke/Park transforms
 *   - PMSM dq electrical model
 *   - Mechanical model
 *
 * FMU INPUTS — unchanged:
 *   duty_a
 *   duty_b
 *   duty_c
 *   v_dc
 *   T_load
 *
 * FMU OUTPUTS — unchanged:
 *   rpm
 *   ia
 *   ib
 *   ic
 *   theta_m
 *   T_em
 *   id_out
 *   iq_out
 *
 * IMPORTANT:
 *   This model contains actual PWM switching rather than an averaged inverter.
 *
 *   For 20 kHz PWM:
 *       T_pwm = 50 us
 *
 *   For accurate switching simulation, use an FMU communication/integration
 *   step significantly smaller than 50 us, preferably 1-2 us.
 *
 *********************************************************************************************************************/

model PMSM_Plant_FMU

  // ========================================================================
  // MOTOR PARAMETERS
  // ========================================================================

  parameter Real R = 0.19
    "Stator resistance [Ohm]";

  parameter Real L_d = 0.000125
    "d-axis inductance [H]";

  parameter Real L_q = 0.000125
    "q-axis inductance [H]";

  parameter Real lambda_pm = 0.0014
    "Permanent magnet flux linkage [Wb]";

  parameter Real J = 2.4e-6
    "Rotor inertia [kg.m2]";

  parameter Real B_fric = 1e-6
    "Viscous friction coefficient [N.m.s]";

  parameter Integer p = 4
    "Number of pole pairs";

  parameter Real v_dc_nom = 12.0
    "Nominal DC bus voltage [V]";


  // ========================================================================
  // INVERTER / PWM PARAMETERS
  // ========================================================================

  parameter Real f_pwm = 20000.0
    "PWM switching frequency [Hz]";

  parameter Real dead_time = 1.0e-6
    "Equivalent inverter dead time [s]";

  parameter Real Ron_inv = 0.005
    "Equivalent inverter ON resistance [Ohm]";

  parameter Boolean use_dead_time = true
    "Enable equivalent dead-time effect";


  // ========================================================================
  // INPUTS — DO NOT CHANGE
  // ========================================================================

  input Real duty_a(start=0.5);
  input Real duty_b(start=0.5);
  input Real duty_c(start=0.5);

  input Real v_dc(start=12.0);

  input Real T_load(start=0.0);


  // ========================================================================
  // STATES
  // ========================================================================

  Real i_d(start=0.0)
    "d-axis current [A]";

  Real i_q(start=0.0)
    "q-axis current [A]";

  Real omega_m(start=0.0)
    "Mechanical angular speed [rad/s]";

  Real theta_e(start=0.0)
    "Electrical rotor angle [rad]";


  // ========================================================================
  // OUTPUTS — EXACTLY THE SAME INTERFACE
  // ========================================================================

  output Real rpm
    "Mechanical speed [RPM]";

  output Real ia
    "Phase A current [A]";

  output Real ib
    "Phase B current [A]";

  output Real ic
    "Phase C current [A]";

  output Real theta_m
    "Mechanical rotor angle [rad]";

  output Real T_em
    "Electromagnetic torque [N.m]";

  output Real id_out = i_d
    "d-axis current [A]";

  output Real iq_out = i_q
    "q-axis current [A]";


  // ========================================================================
  // INTERNAL VARIABLES
  // ========================================================================

protected

  // ------------------------------------------------------------------------
  // PWM
  // ------------------------------------------------------------------------

  Real T_pwm
    "PWM period [s]";

  Real pwm_time
    "Time within PWM period [s]";

  Real carrier
    "Center-aligned triangular carrier [0..1]";


  // ------------------------------------------------------------------------
  // Duty commands
  // ------------------------------------------------------------------------

  Real duty_a_lim;
  Real duty_b_lim;
  Real duty_c_lim;

  Real duty_a_eff;
  Real duty_b_eff;
  Real duty_c_eff;


  // ------------------------------------------------------------------------
  // Gate signals
  // ------------------------------------------------------------------------

  Boolean gate_a_high;
  Boolean gate_b_high;
  Boolean gate_c_high;

  Boolean gate_a_low;
  Boolean gate_b_low;
  Boolean gate_c_low;


  // ------------------------------------------------------------------------
  // Inverter pole voltages
  // ------------------------------------------------------------------------

  Real va_pole;
  Real vb_pole;
  Real vc_pole;


  // ------------------------------------------------------------------------
  // Motor phase voltages
  // ------------------------------------------------------------------------

  Real v_neutral;

  Real va;
  Real vb;
  Real vc;


  // ------------------------------------------------------------------------
  // Clarke transform
  // ------------------------------------------------------------------------

  Real v_alpha;
  Real v_beta;


  // ------------------------------------------------------------------------
  // Park transform
  // ------------------------------------------------------------------------

  Real v_d;
  Real v_q;


  // ------------------------------------------------------------------------
  // Current transforms
  // ------------------------------------------------------------------------

  Real i_alpha;
  Real i_beta;


  // ------------------------------------------------------------------------
  // Electrical speed
  // ------------------------------------------------------------------------

  Real omega_e;


  // ========================================================================
  // EQUATIONS
  // ========================================================================

equation

  // ========================================================================
  // 1. PWM PERIOD
  // ========================================================================

  T_pwm = 1.0 / f_pwm;


  // ========================================================================
  // 2. LIMIT PWM DUTY COMMANDS
  //
  // Protect the inverter model against controller numerical overshoot.
  //
  // 0.0 <= duty <= 1.0
  // ========================================================================

  duty_a_lim = max(0.0, min(1.0, duty_a));
  duty_b_lim = max(0.0, min(1.0, duty_b));
  duty_c_lim = max(0.0, min(1.0, duty_c));


  // ========================================================================
  // 3. PWM TIME
  //
  // 20 kHz:
  //
  //       T_pwm = 50 us
  //
  // ========================================================================

  pwm_time = mod(time, T_pwm);


  // ========================================================================
  // 4. CENTER-ALIGNED TRIANGULAR CARRIER
  //
  //       carrier
  //       1.0       /\
  //                /  \
  //               /    \
  //       0.0 ___/      \___
  //
  // ========================================================================

  if pwm_time < 0.5*T_pwm then

    carrier = 2.0*pwm_time/T_pwm;

  else

    carrier = 2.0*(1.0 - pwm_time/T_pwm);

  end if;


  // ========================================================================
  // 5. EQUIVALENT DEAD-TIME MODEL
  //
  // A real complementary inverter inserts a short period where both
  // switches of a leg are OFF.
  //
  // For this dq-based plant we represent this by shortening each PWM
  // pulse by approximately 2*dead_time.
  //
  // Example:
  //
  //       f_pwm = 20 kHz
  //       T_pwm = 50 us
  //       dead_time = 1 us
  //
  //       duty reduction = 2 us / 50 us = 0.04
  //
  // ========================================================================

  if use_dead_time then

    duty_a_eff =
      max(0.0,
          min(1.0,
              duty_a_lim
              - 2.0*dead_time/T_pwm));

    duty_b_eff =
      max(0.0,
          min(1.0,
              duty_b_lim
              - 2.0*dead_time/T_pwm));

    duty_c_eff =
      max(0.0,
          min(1.0,
              duty_c_lim
              - 2.0*dead_time/T_pwm));

  else

    duty_a_eff = duty_a_lim;
    duty_b_eff = duty_b_lim;
    duty_c_eff = duty_c_lim;

  end if;


  // ========================================================================
  // 6. PWM COMPARATORS
  //
  // Upper transistor:
  //
  //       carrier < duty
  //
  // ========================================================================

  gate_a_high = carrier < duty_a_eff;
  gate_b_high = carrier < duty_b_eff;
  gate_c_high = carrier < duty_c_eff;


  // ========================================================================
  // 7. COMPLEMENTARY LOWER SWITCHES
  // ========================================================================

  gate_a_low = not gate_a_high;
  gate_b_low = not gate_b_high;
  gate_c_low = not gate_c_high;


  // ========================================================================
  // 8. TWO-LEVEL INVERTER
  //
  // Upper switch ON:
  //
  //       pole = +Vdc
  //
  // Lower switch ON:
  //
  //       pole = 0
  //
  // ========================================================================

  va_pole =
    if gate_a_high then
      v_dc
    else
      0.0;

  vb_pole =
    if gate_b_high then
      v_dc
    else
      0.0;

  vc_pole =
    if gate_c_high then
      v_dc
    else
      0.0;


  // ========================================================================
  // 9. FLOATING MOTOR NEUTRAL
  //
  // Three-phase star-connected motor with floating neutral.
  // ========================================================================

  v_neutral =
      (va_pole + vb_pole + vc_pole) / 3.0;


  // ========================================================================
  // 10. MOTOR PHASE VOLTAGES
  // ========================================================================

  va = va_pole - v_neutral;
  vb = vb_pole - v_neutral;
  vc = vc_pole - v_neutral;


  // ========================================================================
  // 11. CLARKE TRANSFORM
  //
  // Amplitude invariant:
  //
  // alpha = 2/3 va - 1/3 vb - 1/3 vc
  //
  // beta  = (vb-vc)/sqrt(3)
  //
  // ========================================================================

  v_alpha =
      (2.0/3.0)*va
      - (1.0/3.0)*vb
      - (1.0/3.0)*vc;

  v_beta =
      (vb - vc) / sqrt(3.0);


  // ========================================================================
  // 12. ELECTRICAL SPEED
  // ========================================================================

  omega_e = p * omega_m;


  // ========================================================================
  // 13. PARK TRANSFORM
  //
  // d = alpha*cos(theta) + beta*sin(theta)
  //
  // q = -alpha*sin(theta) + beta*cos(theta)
  //
  // ========================================================================

  v_d =
      v_alpha*cos(theta_e)
      + v_beta*sin(theta_e);

  v_q =
      -v_alpha*sin(theta_e)
      + v_beta*cos(theta_e);


  // ========================================================================
  // 14. PMSM ELECTRICAL EQUATIONS
  //
  // Ld * did/dt =
  //
  //       vd - R*id + omega_e*Lq*iq
  //
  // Lq * diq/dt =
  //
  //       vq - R*iq
  //          - omega_e*(Ld*id + lambda_pm)
  //
  // ========================================================================

  L_d * der(i_d) =
      v_d
      - R*i_d
      + omega_e*L_q*i_q;

  L_q * der(i_q) =
      v_q
      - R*i_q
      - omega_e*(L_d*i_d + lambda_pm);


  // ========================================================================
  // 15. ELECTROMAGNETIC TORQUE
  //
  // General PMSM equation:
  //
  // T = 1.5*p[
  //          lambda_pm*iq
  //          +(Ld-Lq)*id*iq
  //         ]
  //
  // For surface PMSM:
  //
  // Ld = Lq
  //
  // therefore:
  //
  // T = 1.5*p*lambda_pm*iq
  //
  // ========================================================================

  T_em =
      1.5*p*
      (
        lambda_pm*i_q
        + (L_d - L_q)*i_d*i_q
      );


  // ========================================================================
  // 16. MECHANICAL EQUATION
  //
  // J*domega/dt =
  //
  //       Tem - B*omega - Tload
  //
  // ========================================================================

  J * der(omega_m) =
      T_em
      - B_fric*omega_m
      - T_load;


  // ========================================================================
  // 17. ELECTRICAL ANGLE
  //
  // Do NOT wrap theta_e itself.
  // Keep the state continuous.
  //
  // ========================================================================

  der(theta_e) = omega_e;


  // ========================================================================
  // 18. INVERSE PARK
  //
  // id/iq -> alpha/beta
  // ========================================================================

  i_alpha =
      i_d*cos(theta_e)
      - i_q*sin(theta_e);

  i_beta =
      i_d*sin(theta_e)
      + i_q*cos(theta_e);


  // ========================================================================
  // 19. INVERSE CLARKE
  //
  // alpha/beta -> abc
  // ========================================================================

  ia = i_alpha;

  ib =
      -0.5*i_alpha
      + (sqrt(3.0)/2.0)*i_beta;

  ic =
      -0.5*i_alpha
      - (sqrt(3.0)/2.0)*i_beta;


  // ========================================================================
  // 20. MECHANICAL ANGLE
  //
  // theta_e = p * theta_m
  //
  // ========================================================================

  theta_m = theta_e / p;


  // ========================================================================
  // 21. RPM
  // ========================================================================

  rpm =
      omega_m
      * 60.0
      / (2.0*Modelica.Constants.pi);


end PMSM_Plant_FMU;