/* update bound parameters and variable attributes (start, nominal, min, max) */
#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

OMC_DISABLE_OPT
int SpiralGalaxy_updateBoundVariableAttributes(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  /* min ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating min-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  /* max ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating max-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  /* nominal **************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating nominal-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  /* start ****************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating primary start-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  TRACE_POP
  return 0;
}

void SpiralGalaxy_updateBoundParameters_0(DATA *data, threadData_t *threadData);
void SpiralGalaxy_updateBoundParameters_1(DATA *data, threadData_t *threadData);
void SpiralGalaxy_updateBoundParameters_2(DATA *data, threadData_t *threadData);
void SpiralGalaxy_updateBoundParameters_3(DATA *data, threadData_t *threadData);
OMC_DISABLE_OPT
int SpiralGalaxy_updateBoundParameters(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  (data->simulationInfo->integerParameter[1] /* N PARAM */) = ((modelica_integer) 500);
  data->modelData->integerParameterData[1].time_unvarying = 1;
  SpiralGalaxy_updateBoundParameters_0(data, threadData);
  SpiralGalaxy_updateBoundParameters_1(data, threadData);
  SpiralGalaxy_updateBoundParameters_2(data, threadData);
  SpiralGalaxy_updateBoundParameters_3(data, threadData);
  TRACE_POP
  return 0;
}

#if defined(__cplusplus)
}
#endif

