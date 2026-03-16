/* Events: Sample, Zero Crossings, Relations, Discrete Changes */
#include "PMSM_Motor_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/* Initializes the raw time events of the simulation using the now
   calcualted parameters. */
void PMSM_Motor_function_initSample(DATA *data, threadData_t *threadData)
{
  long i=0;
}

const char *PMSM_Motor_zeroCrossingDescription(int i, int **out_EquationIndexes)
{
  static const char *res[] = {"P_in > 0.0"};
  static const int occurEqs0[] = {1,63};
  static const int *occurEqs[] = {occurEqs0};
  *out_EquationIndexes = (int*) occurEqs[i];
  return res[i];
}

/* forwarded equations */
extern void PMSM_Motor_eqFunction_37(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_38(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_40(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_41(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_42(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_43(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_44(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_45(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_46(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_47(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_48(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_49(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_50(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_52(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_53(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_54(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_56(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_57(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_58(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_59(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_60(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_61(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_62(DATA* data, threadData_t *threadData);

int PMSM_Motor_function_ZeroCrossingsEquations(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->callStatistics.functionZeroCrossingsEquations++;

  static void (*const eqFunctions[24])(DATA*, threadData_t*) = {
    PMSM_Motor_eqFunction_37,
    PMSM_Motor_eqFunction_38,
    PMSM_Motor_eqFunction_40,
    PMSM_Motor_eqFunction_41,
    PMSM_Motor_eqFunction_42,
    PMSM_Motor_eqFunction_43,
    PMSM_Motor_eqFunction_44,
    PMSM_Motor_eqFunction_45,
    PMSM_Motor_eqFunction_46,
    PMSM_Motor_eqFunction_47,
    PMSM_Motor_eqFunction_48,
    PMSM_Motor_eqFunction_49,
    PMSM_Motor_eqFunction_50,
    PMSM_Motor_eqFunction_52,
    PMSM_Motor_eqFunction_53,
    PMSM_Motor_eqFunction_54,
    PMSM_Motor_eqFunction_55,
    PMSM_Motor_eqFunction_56,
    PMSM_Motor_eqFunction_57,
    PMSM_Motor_eqFunction_58,
    PMSM_Motor_eqFunction_59,
    PMSM_Motor_eqFunction_60,
    PMSM_Motor_eqFunction_61,
    PMSM_Motor_eqFunction_62
  };
  
  for (int id = 0; id < 24; id++) {
    eqFunctions[id](data, threadData);
  }
  
  return 0;
}

int PMSM_Motor_function_ZeroCrossings(DATA *data, threadData_t *threadData, double *gout)
{
  const int *equationIndexes = NULL;

  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  modelica_integer current_index = 0;
  modelica_integer start_index;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ZC);
#endif
  data->simulationInfo->callStatistics.functionZeroCrossings++;

  start_index = current_index;
  tmp1 = 1.0;
  tmp2 = 0.0;
  tmp0 = GreaterZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */), 0.0, tmp1, tmp2, data->simulationInfo->storedRelations[0]);
  gout[start_index] = (tmp0) ? 1 : -1;
  current_index++;

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ZC);
#endif

  return 0;
}

const char *PMSM_Motor_relationDescription(int i)
{
  const char *res[] = {"P_in > 0.0"};
  return res[i];
}

int PMSM_Motor_function_updateRelations(DATA *data, threadData_t *threadData, int evalforZeroCross)
{
  const int *equationIndexes = NULL;

  modelica_boolean tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_integer current_index = 0;
  modelica_integer start_index;
  
  if(evalforZeroCross) {
    start_index = current_index;
    tmp4 = 1.0;
    tmp5 = 0.0;
    tmp3 = GreaterZC((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */), 0.0, tmp4, tmp5, data->simulationInfo->storedRelations[0]);
    data->simulationInfo->relations[start_index] = tmp3;
    current_index++;
  } else {
    start_index = current_index;
    data->simulationInfo->relations[start_index] = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */) > 0.0);
    current_index++;
  }
  
  return 0;
}

#if defined(__cplusplus)
}
#endif
