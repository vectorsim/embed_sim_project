/* Algebraic */
#include "PMSM_Plant_FMU_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forwarded equations */
extern void PMSM_Plant_FMU_eqFunction_42(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_43(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_50(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_51(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_82(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_83(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_84(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_85(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[8])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_42,
    PMSM_Plant_FMU_eqFunction_43,
    PMSM_Plant_FMU_eqFunction_50,
    PMSM_Plant_FMU_eqFunction_51,
    PMSM_Plant_FMU_eqFunction_82,
    PMSM_Plant_FMU_eqFunction_83,
    PMSM_Plant_FMU_eqFunction_84,
    PMSM_Plant_FMU_eqFunction_85
  };
  
  if (data->simulationInfo->evalSelection) {
    for (int i = 0; i < data->simulationInfo->evalSelection->n; i++) {
      int id = data->simulationInfo->evalSelection->idx[i];
      eqFunctions[id](data, threadData);
    }
  } else {
    for (int id = 0; id < 8; id++) {
      eqFunctions[id](data, threadData);
    }
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
