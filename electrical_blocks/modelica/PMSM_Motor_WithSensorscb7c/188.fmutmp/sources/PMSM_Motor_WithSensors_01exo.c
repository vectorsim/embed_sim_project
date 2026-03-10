/* External objects file */
#include "PMSM_Motor_WithSensors_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

void PMSM_Motor_WithSensors_callExternalObjectDestructors(DATA *data, threadData_t *threadData)
{
  if(data->simulationInfo->extObjs)
  {
    free(data->simulationInfo->extObjs);
    data->simulationInfo->extObjs = 0;
  }
}
#if defined(__cplusplus)
}
#endif

