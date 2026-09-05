#if defined(__cplusplus)
  extern "C" {
#endif
  int PMSM_Plant_FMU_mayer(DATA* data, modelica_real** res, short*);
  int PMSM_Plant_FMU_lagrange(DATA* data, modelica_real** res, short *, short *);
  int PMSM_Plant_FMU_getInputVarIndicesInOptimization(DATA* data, int* input_var_indices);
  int PMSM_Plant_FMU_pickUpBoundsForInputsInOptimization(DATA* data, modelica_real* min, modelica_real* max, modelica_real*nominal, modelica_boolean *useNominal, char ** name, modelica_real * start, modelica_real * startTimeOpt);
  int PMSM_Plant_FMU_setInputData(DATA *data, const modelica_boolean file);
  int PMSM_Plant_FMU_getTimeGrid(DATA *data, modelica_integer * nsi, modelica_real**t);
#if defined(__cplusplus)
}
#endif
