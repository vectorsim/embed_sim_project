within ;
model DCMotor

  // Electrical components
  Modelica.Electrical.Analog.Basic.Resistor resistor(R=1);
  Modelica.Electrical.Analog.Basic.Inductor inductor(L=0.5);
  Modelica.Electrical.Analog.Basic.RotationalEMF emf(k=0.01);
  Modelica.Electrical.Analog.Sources.SignalVoltage voltage;
  Modelica.Electrical.Analog.Basic.Ground ground;

  // Mechanical components
  Modelica.Mechanics.Rotational.Components.Inertia inertia(J=0.01);
  Modelica.Mechanics.Rotational.Components.Damper damper(d=0.1);

  // Input voltage
  Modelica.Blocks.Interfaces.RealInput u;

  // Output speed
  Modelica.Blocks.Interfaces.RealOutput w;

equation
  // Electrical loop
  connect(voltage.p, resistor.p);
  connect(resistor.n, inductor.p);
  connect(inductor.n, emf.p);
  connect(emf.n, voltage.n);
  connect(voltage.n, ground.p);

  // Mechanical side
  connect(emf.flange, inertia.flange_a);
  connect(inertia.flange_b, damper.flange_a);

  // Apply voltage
  voltage.v = u;

  // Output angular velocity
  w = der(inertia.phi);

end DCMotor;
