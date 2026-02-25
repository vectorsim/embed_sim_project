model VehicleKinematicBicycle
  "Simple kinematic bicycle model for vehicle dynamics"
  
  // Parameters
  parameter Real L = 2.5 "Wheelbase length (m)";
  parameter Real x0 = 0.0 "Initial x position (m)";
  parameter Real y0 = 0.0 "Initial y position (m)";
  parameter Real theta0 = 0.0 "Initial heading angle (rad)";
  parameter Real v0 = 0.0 "Initial velocity (m/s)";
  
  // States
  Real x(start=x0, fixed=true) "X position (m)";
  Real y(start=y0, fixed=true) "Y position (m)";
  Real theta(start=theta0, fixed=true) "Heading angle (rad)";
  Real v(start=v0, fixed=true) "Velocity (m/s)";
  
  // Inputs
  input Real delta "Steering angle (rad)";
  input Real a "Acceleration (m/s^2)";
  
  // Outputs (same as states for convenience)
  output Real x_out = x "X position output";
  output Real y_out = y "Y position output";
  output Real theta_out = theta "Heading output";
  output Real v_out = v "Velocity output";
  
equation
  // Kinematic bicycle model
  der(x) = v * cos(theta);
  der(y) = v * sin(theta);
  der(theta) = (v / L) * tan(delta);
  der(v) = a;
  
  annotation(
    Documentation(info="<html>
<p>Kinematic bicycle model for vehicle dynamics.</p>
<p><b>States:</b></p>
<ul>
<li>x, y: Position in 2D plane</li>
<li>theta: Heading angle</li>
<li>v: Velocity</li>
</ul>
<p><b>Inputs:</b></p>
<ul>
<li>delta: Steering angle (rad)</li>
<li>a: Acceleration (m/s^2)</li>
</ul>
<p><b>Dynamics:</b></p>
<pre>
dx/dt = v * cos(theta)
dy/dt = v * sin(theta)
dtheta/dt = (v/L) * tan(delta)
dv/dt = a
</pre>
</html>"));
end VehicleKinematicBicycle;
