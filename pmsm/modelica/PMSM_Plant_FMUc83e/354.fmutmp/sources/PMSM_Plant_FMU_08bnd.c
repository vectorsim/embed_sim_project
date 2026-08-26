/* update bound parameters and variable attributes (start, nominal, min, max) */
#include "PMSM_Plant_FMU_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

OMC_DISABLE_OPT
int PMSM_Plant_FMU_updateBoundVariableAttributes(DATA *data, threadData_t *threadData)
{
  /* min ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating min-values");
  messageClose(OMC_LOG_INIT);
  
  /* max ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating max-values");
  messageClose(OMC_LOG_INIT);
  
  /* nominal **************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating nominal-values");
  messageClose(OMC_LOG_INIT);
  
  /* start ****************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating primary start-values");
  messageClose(OMC_LOG_INIT);
  
  return 0;
}

void PMSM_Plant_FMU_updateBoundParameters_0(DATA *data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_20(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void PMSM_Plant_FMU_updateBoundParameters_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[1])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_20
  };
  
  for (int id = 0; id < 1; id++) {
    eqFunctions[id](data, threadData);
  }
}
OMC_DISABLE_OPT
int PMSM_Plant_FMU_updateBoundParameters(DATA *data, threadData_t *threadData)
{
  (data->simulationInfo->booleanParameter[data->simulationInfo->booleanParamsIndex[0]] /* use_dead_time PARAM */) = 1 /* true */;
  data->modelData->booleanParameterData[0].time_unvarying = 1;
  PMSM_Plant_FMU_updateBoundParameters_0(data, threadData);
  return 0;
}

#if defined(__cplusplus)
}
#endif
