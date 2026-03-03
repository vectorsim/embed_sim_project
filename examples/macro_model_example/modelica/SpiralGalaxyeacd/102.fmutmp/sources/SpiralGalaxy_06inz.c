/* Initialization */
#include "SpiralGalaxy_model.h"
#include "SpiralGalaxy_11mix.h"
#include "SpiralGalaxy_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void SpiralGalaxy_functionInitialEquations_0(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_1(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_2(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_3(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_4(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_5(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_6(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_7(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_8(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_9(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_10(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_11(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_12(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_13(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_14(DATA *data, threadData_t *threadData);
void SpiralGalaxy_functionInitialEquations_15(DATA *data, threadData_t *threadData);

int SpiralGalaxy_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  SpiralGalaxy_functionInitialEquations_0(data, threadData);
  SpiralGalaxy_functionInitialEquations_1(data, threadData);
  SpiralGalaxy_functionInitialEquations_2(data, threadData);
  SpiralGalaxy_functionInitialEquations_3(data, threadData);
  SpiralGalaxy_functionInitialEquations_4(data, threadData);
  SpiralGalaxy_functionInitialEquations_5(data, threadData);
  SpiralGalaxy_functionInitialEquations_6(data, threadData);
  SpiralGalaxy_functionInitialEquations_7(data, threadData);
  SpiralGalaxy_functionInitialEquations_8(data, threadData);
  SpiralGalaxy_functionInitialEquations_9(data, threadData);
  SpiralGalaxy_functionInitialEquations_10(data, threadData);
  SpiralGalaxy_functionInitialEquations_11(data, threadData);
  SpiralGalaxy_functionInitialEquations_12(data, threadData);
  SpiralGalaxy_functionInitialEquations_13(data, threadData);
  SpiralGalaxy_functionInitialEquations_14(data, threadData);
  SpiralGalaxy_functionInitialEquations_15(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}

/* No SpiralGalaxy_functionInitialEquations_lambda0 function */

int SpiralGalaxy_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
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

