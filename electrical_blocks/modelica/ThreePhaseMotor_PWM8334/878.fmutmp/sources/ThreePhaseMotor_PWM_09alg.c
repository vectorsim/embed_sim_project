/* Algebraic */
#include "ThreePhaseMotor_PWM_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forwarded equations */
extern void ThreePhaseMotor_PWM_eqFunction_30(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_31(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_50(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_51(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_52(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_53(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_54(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_55(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_56(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_57(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_58(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  int id;

  static void (*const eqFunctions[11])(DATA*, threadData_t*) = {
    ThreePhaseMotor_PWM_eqFunction_30,
    ThreePhaseMotor_PWM_eqFunction_31,
    ThreePhaseMotor_PWM_eqFunction_50,
    ThreePhaseMotor_PWM_eqFunction_51,
    ThreePhaseMotor_PWM_eqFunction_52,
    ThreePhaseMotor_PWM_eqFunction_53,
    ThreePhaseMotor_PWM_eqFunction_54,
    ThreePhaseMotor_PWM_eqFunction_55,
    ThreePhaseMotor_PWM_eqFunction_56,
    ThreePhaseMotor_PWM_eqFunction_57,
    ThreePhaseMotor_PWM_eqFunction_58
  };
  
  static const int eqIndices[11] = {
    30,
    31,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58
  };
  
  for (id = 0; id < 11; id++) {
    eqFunctions[id](data, threadData);
    threadData->lastEquationSolved = eqIndices[id];
  }
}
/* for continuous time variables */
int ThreePhaseMotor_PWM_functionAlgebraics(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ALGEBRAICS);
#endif
  data->simulationInfo->callStatistics.functionAlgebraics++;

  ThreePhaseMotor_PWM_function_savePreSynchronous(data, threadData);
  
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
