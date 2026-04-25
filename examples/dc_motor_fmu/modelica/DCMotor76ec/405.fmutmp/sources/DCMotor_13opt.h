#if defined(__cplusplus)
  extern "C" {
#endif
  int DCMotor_mayer(DATA* data, modelica_real** res, short*);
  int DCMotor_lagrange(DATA* data, modelica_real** res, short *, short *);
  int DCMotor_getInputVarIndicesInOptimization(DATA* data, int* input_var_indices);
  int DCMotor_pickUpBoundsForInputsInOptimization(DATA* data, modelica_real* min, modelica_real* max, modelica_real*nominal, modelica_boolean *useNominal, char ** name, modelica_real * start, modelica_real * startTimeOpt);
  int DCMotor_setInputData(DATA *data, const modelica_boolean file);
  int DCMotor_getTimeGrid(DATA *data, modelica_integer * nsi, modelica_real**t);
#if defined(__cplusplus)
}
#endif
