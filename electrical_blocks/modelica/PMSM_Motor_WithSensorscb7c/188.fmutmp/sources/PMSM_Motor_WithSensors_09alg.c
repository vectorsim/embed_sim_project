/* Algebraic */
#include "PMSM_Motor_WithSensors_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forwarded equations */
extern void PMSM_Motor_WithSensors_eqFunction_32(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_33(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_47(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_54(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_56(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_57(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_58(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_59(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_60(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_61(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_62(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  int id;

  static void (*const eqFunctions[12])(DATA*, threadData_t*) = {
    PMSM_Motor_WithSensors_eqFunction_32,
    PMSM_Motor_WithSensors_eqFunction_33,
    PMSM_Motor_WithSensors_eqFunction_47,
    PMSM_Motor_WithSensors_eqFunction_54,
    PMSM_Motor_WithSensors_eqFunction_55,
    PMSM_Motor_WithSensors_eqFunction_56,
    PMSM_Motor_WithSensors_eqFunction_57,
    PMSM_Motor_WithSensors_eqFunction_58,
    PMSM_Motor_WithSensors_eqFunction_59,
    PMSM_Motor_WithSensors_eqFunction_60,
    PMSM_Motor_WithSensors_eqFunction_61,
    PMSM_Motor_WithSensors_eqFunction_62
  };
  
  static const int eqIndices[12] = {
    32,
    33,
    47,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62
  };
  
  for (id = 0; id < 12; id++) {
    eqFunctions[id](data, threadData);
    threadData->lastEquationSolved = eqIndices[id];
  }
}
/* for continuous time variables */
int PMSM_Motor_WithSensors_functionAlgebraics(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ALGEBRAICS);
#endif
  data->simulationInfo->callStatistics.functionAlgebraics++;

  PMSM_Motor_WithSensors_function_savePreSynchronous(data, threadData);
  
  functionAlg_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ALGEBRAICS);
#endif

  TRACE_POP
  return 0;
}

#ifdef __cplusplus
}
#endif
