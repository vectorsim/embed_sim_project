/* Algebraic */
#include "PMSM_Motor_model.h"

#ifdef __cplusplus
extern "C" {
#endif

/* forwarded equations */
extern void PMSM_Motor_eqFunction_39(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_40(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_43(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_62(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_63(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_64(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_65(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_66(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_67(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_68(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_69(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_70(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_71(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_72(DATA* data, threadData_t *threadData);

static void functionAlg_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[15])(DATA*, threadData_t*) = {
    PMSM_Motor_eqFunction_39,
    PMSM_Motor_eqFunction_40,
    PMSM_Motor_eqFunction_43,
    PMSM_Motor_eqFunction_55,
    PMSM_Motor_eqFunction_62,
    PMSM_Motor_eqFunction_63,
    PMSM_Motor_eqFunction_64,
    PMSM_Motor_eqFunction_65,
    PMSM_Motor_eqFunction_66,
    PMSM_Motor_eqFunction_67,
    PMSM_Motor_eqFunction_68,
    PMSM_Motor_eqFunction_69,
    PMSM_Motor_eqFunction_70,
    PMSM_Motor_eqFunction_71,
    PMSM_Motor_eqFunction_72
  };
  
  for (int id = 0; id < 15; id++) {
    eqFunctions[id](data, threadData);
  }
}
/* for continuous time variables */
int PMSM_Motor_functionAlgebraics(DATA *data, threadData_t *threadData)
{

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_ALGEBRAICS);
#endif
  data->simulationInfo->callStatistics.functionAlgebraics++;

  PMSM_Motor_function_savePreSynchronous(data, threadData);
  
  functionAlg_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_ALGEBRAICS);
#endif

  return 0;
}

#ifdef __cplusplus
}
#endif
