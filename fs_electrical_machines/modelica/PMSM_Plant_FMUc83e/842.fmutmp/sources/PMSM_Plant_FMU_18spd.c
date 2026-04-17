/* spatialDistribution */
#include "PMSM_Plant_FMU_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

int PMSM_Plant_FMU_function_storeSpatialDistribution(DATA *data, threadData_t *threadData)
{
  int equationIndexes[2] = {1,-1};
  
  return 0;
}

int PMSM_Plant_FMU_function_initSpatialDistribution(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

#if defined(__cplusplus)
}
#endif
