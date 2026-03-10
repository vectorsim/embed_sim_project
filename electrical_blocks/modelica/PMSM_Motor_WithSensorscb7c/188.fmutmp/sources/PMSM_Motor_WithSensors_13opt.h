#if defined(__cplusplus)
  extern "C" {
#endif
  int PMSM_Motor_WithSensors_mayer(DATA* data, modelica_real** res, short*);
  int PMSM_Motor_WithSensors_lagrange(DATA* data, modelica_real** res, short *, short *);
  int PMSM_Motor_WithSensors_pickUpBoundsForInputsInOptimization(DATA* data, modelica_real* min, modelica_real* max, modelica_real*nominal, modelica_boolean *useNominal, char ** name, modelica_real * start, modelica_real * startTimeOpt);
  int PMSM_Motor_WithSensors_setInputData(DATA *data, const modelica_boolean file);
  int PMSM_Motor_WithSensors_getTimeGrid(DATA *data, modelica_integer * nsi, modelica_real**t);
#if defined(__cplusplus)
}
#endif