/* External objects file */
#include "PMSM_Plant_FMU_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

void PMSM_Plant_FMU_callExternalObjectDestructors(DATA *data, threadData_t *threadData)
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
