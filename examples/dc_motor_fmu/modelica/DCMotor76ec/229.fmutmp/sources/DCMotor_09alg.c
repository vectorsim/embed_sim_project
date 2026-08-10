/* Algebraic */
#include "DCMotor_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forwarded equations */
extern void DCMotor_eqFunction_29(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_30(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_32(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_41(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_43(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_44(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_45(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[7])(DATA*, threadData_t*) = {
    DCMotor_eqFunction_29,
    DCMotor_eqFunction_30,
    DCMotor_eqFunction_32,
    DCMotor_eqFunction_41,
    DCMotor_eqFunction_43,
    DCMotor_eqFunction_44,
    DCMotor_eqFunction_45
  };
  
  for (int id = 0; id < 7; id++) {
    eqFunctions[id](data, threadData);
  }
}
/* for continuous time variables */
int DCMotor_functionAlgebraics(DATA *data, threadData_t *threadData)
{

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ALGEBRAICS);
#endif
  data->simulationInfo->callStatistics.functionAlgebraics++;

  DCMotor_function_savePreSynchronous(data, threadData);
  
  functionAlg_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ALGEBRAICS);
#endif

  return 0;
}

#ifdef __cplusplus
}
#endif
