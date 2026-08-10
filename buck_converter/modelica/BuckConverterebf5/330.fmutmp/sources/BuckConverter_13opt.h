#if defined(__cplusplus)
  extern "C" {
#endif
  int BuckConverter_mayer(DATA* data, modelica_real** res, short*);
  int BuckConverter_lagrange(DATA* data, modelica_real** res, short *, short *);
  int BuckConverter_getInputVarIndicesInOptimization(DATA* data, int* input_var_indices);
  int BuckConverter_pickUpBoundsForInputsInOptimization(DATA* data, modelica_real* min, modelica_real* max, modelica_real*nominal, modelica_boolean *useNominal, char ** name, modelica_real * start, modelica_real * startTimeOpt);
  int BuckConverter_setInputData(DATA *data, const modelica_boolean file);
  int BuckConverter_getTimeGrid(DATA *data, modelica_integer * nsi, modelica_real**t);
#if defined(__cplusplus)
}
#endif
