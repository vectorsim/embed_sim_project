/* Linearization */
#include "ThreePhaseMotor_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
const char *ThreePhaseMotor_linear_model_frame()
{
  return "model linearized_model \"ThreePhaseMotor\"\n"
  "  parameter Integer n = 4 \"number of states\";\n"
  "  parameter Integer m = 3 \"number of inputs\";\n"
  "  parameter Integer p = 0 \"number of outputs\";\n"
  "\n"
  "  parameter Real x0[n] = %s;\n"
  "  parameter Real u0[m] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] = zeros(p, n);%s\n\n"
  "  parameter Real D[p, m] = zeros(p, m);%s\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "\n"
  "  Real 'x_i_d' = x[1];\n"
  "  Real 'x_i_q' = x[2];\n"
  "  Real 'x_omega_m' = x[3];\n"
  "  Real 'x_theta_e' = x[4];\n"
  "  Real 'u_T_load' = u[1];\n"
  "  Real 'u_v_d' = u[2];\n"
  "  Real 'u_v_q' = u[3];\n"
  "equation\n"
  "  der(x) = A * x + B * u;\n"
  "  y = C * x + D * u;\n"
  "end linearized_model;\n";
}
const char *ThreePhaseMotor_linear_model_datarecovery_frame()
{
  return "model linearized_model \"ThreePhaseMotor\"\n"
  "  parameter Integer n = 4 \"number of states\";\n"
  "  parameter Integer m = 3 \"number of inputs\";\n"
  "  parameter Integer p = 0 \"number of outputs\";\n"
  "  parameter Integer nz = 6 \"data recovery variables\";\n"
  "\n"
  "  parameter Real x0[n] = %s;\n"
  "  parameter Real u0[m] = %s;\n"
  "  parameter Real z0[nz] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] = zeros(p, n);%s\n\n"
  "  parameter Real D[p, m] = zeros(p, m);%s\n\n"
  "  parameter Real Cz[nz, n] =\n\t[%s];\n\n"
  "  parameter Real Dz[nz, m] =\n\t[%s];\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "  output Real z[nz];\n"
  "\n"
  "  Real 'x_i_d' = x[1];\n"
  "  Real 'x_i_q' = x[2];\n"
  "  Real 'x_omega_m' = x[3];\n"
  "  Real 'x_theta_e' = x[4];\n"
  "  Real 'u_T_load' = u[1];\n"
  "  Real 'u_v_d' = u[2];\n"
  "  Real 'u_v_q' = u[3];\n"
  "  Real 'z_T_em' = z[1];\n"
  "  Real 'z_T_load' = z[2];\n"
  "  Real 'z_omega_e' = z[3];\n"
  "  Real 'z_speed_rpm' = z[4];\n"
  "  Real 'z_v_d' = z[5];\n"
  "  Real 'z_v_q' = z[6];\n"
  "equation\n"
  "  der(x) = A * x + B * u;\n"
  "  y = C * x + D * u;\n"
  "  z = Cz * x + Dz * u;\n"
  "end linearized_model;\n";
}
#if defined(__cplusplus)
}
#endif

