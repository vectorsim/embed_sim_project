/* Events: Sample, Zero Crossings, Relations, Discrete Changes */
#include "PMSM_Plant_FMU_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/* Initializes the raw time events of the simulation using the now
   calcualted parameters. */
void PMSM_Plant_FMU_function_initSample(DATA *data, threadData_t *threadData)
{
  long i=0;
}

const char *PMSM_Plant_FMU_zeroCrossingDescription(int i, int **out_EquationIndexes)
{
  static const char *res[] = {"mod(time, T_pwm, 0)",
  "pwm_time < 0.5 * T_pwm",
  "carrier < duty_a_eff",
  "not carrier < duty_a_eff",
  "carrier < duty_b_eff",
  "not carrier < duty_b_eff",
  "carrier < duty_c_eff",
  "not carrier < duty_c_eff"};
  static const int occurEqs0[] = {1,48};
  static const int occurEqs1[] = {1,49};
  static const int occurEqs2[] = {1,56};
  static const int occurEqs3[] = {1,56};
  static const int occurEqs4[] = {1,62};
  static const int occurEqs5[] = {1,62};
  static const int occurEqs6[] = {1,68};
  static const int occurEqs7[] = {1,68};
  static const int *occurEqs[] = {occurEqs0,occurEqs1,occurEqs2,occurEqs3,occurEqs4,occurEqs5,occurEqs6,occurEqs7};
  *out_EquationIndexes = (int*) occurEqs[i];
  return res[i];
}

/* forwarded equations */
extern void PMSM_Plant_FMU_eqFunction_48(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_49(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_52(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_53(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_54(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_58(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_59(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_60(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_61(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_64(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_65(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_66(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_67(DATA* data, threadData_t *threadData);

int PMSM_Plant_FMU_function_ZeroCrossingsEquations(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->callStatistics.functionZeroCrossingsEquations++;

  static void (*const eqFunctions[14])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_48,
    PMSM_Plant_FMU_eqFunction_49,
    PMSM_Plant_FMU_eqFunction_52,
    PMSM_Plant_FMU_eqFunction_53,
    PMSM_Plant_FMU_eqFunction_54,
    PMSM_Plant_FMU_eqFunction_55,
    PMSM_Plant_FMU_eqFunction_58,
    PMSM_Plant_FMU_eqFunction_59,
    PMSM_Plant_FMU_eqFunction_60,
    PMSM_Plant_FMU_eqFunction_61,
    PMSM_Plant_FMU_eqFunction_64,
    PMSM_Plant_FMU_eqFunction_65,
    PMSM_Plant_FMU_eqFunction_66,
    PMSM_Plant_FMU_eqFunction_67
  };
  
  for (int id = 0; id < 14; id++) {
    eqFunctions[id](data, threadData);
  }
  
  return 0;
}

int PMSM_Plant_FMU_function_ZeroCrossings(DATA *data, threadData_t *threadData, double *gout)
{
  const int *equationIndexes = NULL;

  modelica_real tmp0;
  modelica_real tmp1;
  modelica_boolean tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_boolean tmp5;
  modelica_real tmp6;
  modelica_real tmp7;
  modelica_boolean tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_boolean tmp11;
  modelica_real tmp12;
  modelica_real tmp13;
  modelica_boolean tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  modelica_boolean tmp17;
  modelica_real tmp18;
  modelica_real tmp19;
  modelica_boolean tmp20;
  modelica_real tmp21;
  modelica_real tmp22;
  modelica_integer current_index = 0;
  modelica_integer start_index;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ZC);
#endif
  data->simulationInfo->callStatistics.functionZeroCrossings++;

  start_index = current_index;
  tmp0 = floor((data->localData[0]->timeValue) / ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */)));
  tmp1 = floor((data->simulationInfo->mathEventsValuePre[((modelica_integer) 0)]) / (data->simulationInfo->mathEventsValuePre[((modelica_integer) 0)+1]));
  gout[start_index] = tmp0 != tmp1 ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp3 = 1.0;
  tmp4 = 0.5;
  tmp2 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */), (0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */)), tmp3, tmp4, data->simulationInfo->storedRelations[0]);
  gout[start_index] = (tmp2) ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp6 = 1.0;
  tmp7 = 1.0;
  tmp5 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */), tmp6, tmp7, data->simulationInfo->storedRelations[1]);
  gout[start_index] = (tmp5) ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp9 = 1.0;
  tmp10 = 1.0;
  tmp8 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */), tmp9, tmp10, data->simulationInfo->storedRelations[1]);
  gout[start_index] = ((!tmp8)) ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp12 = 1.0;
  tmp13 = 1.0;
  tmp11 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */), tmp12, tmp13, data->simulationInfo->storedRelations[2]);
  gout[start_index] = (tmp11) ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp15 = 1.0;
  tmp16 = 1.0;
  tmp14 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */), tmp15, tmp16, data->simulationInfo->storedRelations[2]);
  gout[start_index] = ((!tmp14)) ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp18 = 1.0;
  tmp19 = 1.0;
  tmp17 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */), tmp18, tmp19, data->simulationInfo->storedRelations[3]);
  gout[start_index] = (tmp17) ? 1 : -1;
  current_index++;

  start_index = current_index;
  tmp21 = 1.0;
  tmp22 = 1.0;
  tmp20 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */), tmp21, tmp22, data->simulationInfo->storedRelations[3]);
  gout[start_index] = ((!tmp20)) ? 1 : -1;
  current_index++;

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ZC);
#endif

  return 0;
}

const char *PMSM_Plant_FMU_relationDescription(int i)
{
  const char *res[] = {"pwm_time < 0.5 * T_pwm",
  "carrier < duty_a_eff",
  "carrier < duty_b_eff",
  "carrier < duty_c_eff"};
  return res[i];
}

int PMSM_Plant_FMU_function_updateRelations(DATA *data, threadData_t *threadData, int evalforZeroCross)
{
  const int *equationIndexes = NULL;

  modelica_boolean tmp23;
  modelica_real tmp24;
  modelica_real tmp25;
  modelica_boolean tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_boolean tmp29;
  modelica_real tmp30;
  modelica_real tmp31;
  modelica_boolean tmp32;
  modelica_real tmp33;
  modelica_real tmp34;
  modelica_integer current_index = 0;
  modelica_integer start_index;
  
  if(evalforZeroCross) {
    start_index = current_index;
    tmp24 = 1.0;
    tmp25 = 0.5;
    tmp23 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */), (0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */)), tmp24, tmp25, data->simulationInfo->storedRelations[0]);
    data->simulationInfo->relations[start_index] = tmp23;
    current_index++;

    start_index = current_index;
    tmp27 = 1.0;
    tmp28 = 1.0;
    tmp26 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */), tmp27, tmp28, data->simulationInfo->storedRelations[1]);
    data->simulationInfo->relations[start_index] = tmp26;
    current_index++;

    start_index = current_index;
    tmp30 = 1.0;
    tmp31 = 1.0;
    tmp29 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */), tmp30, tmp31, data->simulationInfo->storedRelations[2]);
    data->simulationInfo->relations[start_index] = tmp29;
    current_index++;

    start_index = current_index;
    tmp33 = 1.0;
    tmp34 = 1.0;
    tmp32 = LessZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */), tmp33, tmp34, data->simulationInfo->storedRelations[3]);
    data->simulationInfo->relations[start_index] = tmp32;
    current_index++;
  } else {
    start_index = current_index;
    data->simulationInfo->relations[start_index] = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */) < (0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */)));
    current_index++;

    start_index = current_index;
    data->simulationInfo->relations[start_index] = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */) < (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */));
    current_index++;

    start_index = current_index;
    data->simulationInfo->relations[start_index] = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */) < (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */));
    current_index++;

    start_index = current_index;
    data->simulationInfo->relations[start_index] = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */) < (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */));
    current_index++;
  }
  
  return 0;
}

#if defined(__cplusplus)
}
#endif
