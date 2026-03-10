model BuckConverter
  "Simple buck converter plant model - Clean version without switch_state"

  // Parameters
  parameter Real L = 100e-6 "Inductance [H]";
  parameter Real C = 100e-6 "Capacitance [F]";
  parameter Real R_load = 10 "Load resistance [Ω]";
  parameter Real V_in = 24 "Input voltage [V]";
  parameter Real f_sw = 100e3 "Switching frequency [Hz]";

  // Inputs
  input Real duty "PWM duty cycle [0-1]";

  // Outputs
  output Real V_out "Output voltage [V]";
  output Real I_L "Inductor current [A]";
  output Real I_load "Load current [A]";

protected
  Real switch_state;  // Internal variable - NOT exposed as output

equation
  // Ideal switch model (continuous approximation)
  switch_state = duty;

  // Inductor dynamics
  L * der(I_L) = switch_state * V_in - V_out;

  // Capacitor dynamics
  C * der(V_out) = I_L - V_out / R_load;

  // Load current
  I_load = V_out / R_load;

  // Initial conditions
  initial equation
    I_L = 0;
    V_out = 0;

end BuckConverter;