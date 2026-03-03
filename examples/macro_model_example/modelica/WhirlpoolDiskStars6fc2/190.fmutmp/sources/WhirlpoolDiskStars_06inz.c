/* Initialization */
#include "WhirlpoolDiskStars_model.h"
#include "WhirlpoolDiskStars_11mix.h"
#include "WhirlpoolDiskStars_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void WhirlpoolDiskStars_functionInitialEquations_0(DATA *data, threadData_t *threadData);
void WhirlpoolDiskStars_functionInitialEquations_1(DATA *data, threadData_t *threadData);
void WhirlpoolDiskStars_functionInitialEquations_2(DATA *data, threadData_t *threadData);
void WhirlpoolDiskStars_functionInitialEquations_3(DATA *data, threadData_t *threadData);
void WhirlpoolDiskStars_functionInitialEquations_4(DATA *data, threadData_t *threadData);
void WhirlpoolDiskStars_functionInitialEquations_5(DATA *data, threadData_t *threadData);

int WhirlpoolDiskStars_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  WhirlpoolDiskStars_functionInitialEquations_0(data, threadData);
  WhirlpoolDiskStars_functionInitialEquations_1(data, threadData);
  WhirlpoolDiskStars_functionInitialEquations_2(data, threadData);
  WhirlpoolDiskStars_functionInitialEquations_3(data, threadData);
  WhirlpoolDiskStars_functionInitialEquations_4(data, threadData);
  WhirlpoolDiskStars_functionInitialEquations_5(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}

/* No WhirlpoolDiskStars_functionInitialEquations_lambda0 function */

int WhirlpoolDiskStars_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  TRACE_POP
  return 0;
}


#if defined(__cplusplus)
}
#endif

