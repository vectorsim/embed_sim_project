/* Algebraic */
#include "PMSM_Plant_FMU_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forwarded equations */
extern void PMSM_Plant_FMU_eqFunction_30(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_31(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_45(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_46(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_53(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_54(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_56(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[8])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_30,
    PMSM_Plant_FMU_eqFunction_31,
    PMSM_Plant_FMU_eqFunction_45,
    PMSM_Plant_FMU_eqFunction_46,
    PMSM_Plant_FMU_eqFunction_53,
    PMSM_Plant_FMU_eqFunction_54,
    PMSM_Plant_FMU_eqFunction_55,
    PMSM_Plant_FMU_eqFunction_56
  };
  
  for (int id = 0; id < 8; id++) {
    eqFunctions[id](data, threadData);
  }
}
/* for continuous time variables */
int PMSM_Plant_FMU_functionAlgebraics(DATA *data, threadData_t *threadData)
{

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ALGEBRAICS);
#endif
  data->simulationInfo->callStatistics.functionAlgebraics++;

  PMSM_Plant_FMU_function_savePreSynchronous(data, threadData);
  
  functionAlg_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ALGEBRAICS);
#endif

  return 0;
}

#ifdef __cplusplus
}
#endif
