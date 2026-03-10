/* Linearization */
#include "BuckConverter_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
const char *BuckConverter_linear_model_frame()
{
  return "model linearized_model \"BuckConverter\"\n"
  "  parameter Integer n = 2 \"number of states\";\n"
  "  parameter Integer m = 1 \"number of inputs\";\n"
  "  parameter Integer p = 3 \"number of outputs\";\n"
  "\n"
  "  parameter Real x0[n] = %s;\n"
  "  parameter Real u0[m] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] =\n\t[%s];\n\n"
  "  parameter Real D[p, m] =\n\t[%s];\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "\n"
  "  Real 'x_$outputAlias_I_L' = x[1];\n"
  "  Real 'x_$outputAlias_V_out' = x[2];\n"
  "  Real 'u_duty' = u[1];\n"
  "  Real 'y_I_L' = y[1];\n"
  "  Real 'y_I_load' = y[2];\n"
  "  Real 'y_V_out' = y[3];\n"
  "equation\n"
  "  der(x) = A * x + B * u;\n"
  "  y = C * x + D * u;\n"
  "end linearized_model;\n";
}
const char *BuckConverter_linear_model_datarecovery_frame()
{
  return "model linearized_model \"BuckConverter\"\n"
  "  parameter Integer n = 2 \"number of states\";\n"
  "  parameter Integer m = 1 \"number of inputs\";\n"
  "  parameter Integer p = 3 \"number of outputs\";\n"
  "  parameter Integer nz = 4 \"data recovery variables\";\n"
  "\n"
  "  parameter Real x0[n] = %s;\n"
  "  parameter Real u0[m] = %s;\n"
  "  parameter Real z0[nz] = %s;\n"
  "\n"
  "  parameter Real A[n, n] =\n\t[%s];\n\n"
  "  parameter Real B[n, m] =\n\t[%s];\n\n"
  "  parameter Real C[p, n] =\n\t[%s];\n\n"
  "  parameter Real D[p, m] =\n\t[%s];\n\n"
  "  parameter Real Cz[nz, n] =\n\t[%s];\n\n"
  "  parameter Real Dz[nz, m] =\n\t[%s];\n\n"
  "\n"
  "  Real x[n](start=x0);\n"
  "  input Real u[m](start=u0);\n"
  "  output Real y[p];\n"
  "  output Real z[nz];\n"
  "\n"
  "  Real 'x_$outputAlias_I_L' = x[1];\n"
  "  Real 'x_$outputAlias_V_out' = x[2];\n"
  "  Real 'u_duty' = u[1];\n"
  "  Real 'y_I_L' = y[1];\n"
  "  Real 'y_I_load' = y[2];\n"
  "  Real 'y_V_out' = y[3];\n"
  "  Real 'z_I_L' = z[1];\n"
  "  Real 'z_I_load' = z[2];\n"
  "  Real 'z_V_out' = z[3];\n"
  "  Real 'z_duty' = z[4];\n"
  "equation\n"
  "  der(x) = A * x + B * u;\n"
  "  y = C * x + D * u;\n"
  "  z = Cz * x + Dz * u;\n"
  "end linearized_model;\n";
}
#if defined(__cplusplus)
}
#endif

