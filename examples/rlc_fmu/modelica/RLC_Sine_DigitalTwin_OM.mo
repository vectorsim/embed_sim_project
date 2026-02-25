model RLC_Sine_DigitalTwin_OM

  import Modelica;
  import Modelica.Blocks;
  import Modelica.Electrical.Analog;

  // PARAMETERS
  parameter Real R = 10;
  parameter Real L = 10e-3;
  parameter Real C = 100e-6;
  parameter Real Vref_ampl = 10;
  parameter Real freq = 50;
  parameter Boolean usePythonControl = true;

  // PLANT - Series RLC: VS -> L1 -> R1 -> C1 -> GND
  Analog.Basic.Resistor  R1(R=R);
  Analog.Basic.Inductor  L1(L=L, i(start=0, fixed=true));
  Analog.Basic.Capacitor C1(C=C, v(start=0, fixed=true));
  Analog.Basic.Ground    GND;

  // VOLTAGE SENSOR across C1
  Analog.Sensors.VoltageSensor Vsense;

  // SIGNAL-DRIVEN VOLTAGE SOURCE
  Analog.Sources.SignalVoltage VS;

  // CONTROL SIGNALS
  Blocks.Interfaces.RealInput  Vcontrol_python "External Python control input";
  Real Vcontrol_internal;
  Real Vcontrol;

  // INTERNAL PI CONTROL
  Blocks.Sources.Sine        sineRef(amplitude=Vref_ampl, f=freq);
  Blocks.Math.Feedback       error_sig;
  Blocks.Continuous.PID      pi(k=5, Ti=0.01, Td=0);

  // OUTPUT
  Blocks.Interfaces.RealOutput Vout;

equation
  // Internal PI loop
  connect(sineRef.y,   error_sig.u1);
  connect(Vsense.v,    error_sig.u2);
  connect(error_sig.y, pi.u);
  Vcontrol_internal = pi.y;

  // Control switching
  Vcontrol = if usePythonControl then Vcontrol_python else Vcontrol_internal;
  VS.v = Vcontrol;

  // Circuit topology: VS -> L1 -> R1 -> C1 -> GND
  connect(VS.p,  L1.p);
  connect(L1.n,  R1.p);
  connect(R1.n,  C1.p);
  connect(C1.n,  GND.p);
  connect(VS.n,  GND.p);

  // Voltage sensor across C1
  connect(Vsense.p, C1.p);
  connect(Vsense.n, C1.n);

  // Output
  Vout = Vsense.v;

end RLC_Sine_DigitalTwin_OM;
