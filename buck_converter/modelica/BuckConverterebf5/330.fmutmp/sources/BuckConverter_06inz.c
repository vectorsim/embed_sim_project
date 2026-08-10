/* Initialization */
#include "BuckConverter_model.h"
#include "BuckConverter_11mix.h"
#include "BuckConverter_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void BuckConverter_functionInitialEquations_0(DATA *data, threadData_t *threadData);

/*
equation index: 1
type: SIMPLE_ASSIGN
V_out = 0.0
*/
void BuckConverter_eqFunction_1(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* V_out variable */) = 0.0;
  threadData->lastEquationSolved = 1;
}

/*
equation index: 2
type: SIMPLE_ASSIGN
$outputAlias_V_out = V_out
*/
void BuckConverter_eqFunction_2(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,2};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* $outputAlias_V_out STATE(1) */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* V_out variable */);
  threadData->lastEquationSolved = 2;
}
extern void BuckConverter_eqFunction_10(DATA *data, threadData_t *threadData);

extern void BuckConverter_eqFunction_8(DATA *data, threadData_t *threadData);


/*
equation index: 5
type: SIMPLE_ASSIGN
I_L = 0.0
*/
void BuckConverter_eqFunction_5(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,5};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* I_L variable */) = 0.0;
  threadData->lastEquationSolved = 5;
}

/*
equation index: 6
type: SIMPLE_ASSIGN
$outputAlias_I_L = I_L
*/
void BuckConverter_eqFunction_6(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,6};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* $outputAlias_I_L STATE(1) */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* I_L variable */);
  threadData->lastEquationSolved = 6;
}
extern void BuckConverter_eqFunction_9(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void BuckConverter_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[7])(DATA*, threadData_t*) = {
    BuckConverter_eqFunction_1,
    BuckConverter_eqFunction_2,
    BuckConverter_eqFunction_10,
    BuckConverter_eqFunction_8,
    BuckConverter_eqFunction_5,
    BuckConverter_eqFunction_6,
    BuckConverter_eqFunction_9
  };
  
  for (int id = 0; id < 7; id++) {
    eqFunctions[id](data, threadData);
  }
}

int BuckConverter_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->discreteCall = 1;
  BuckConverter_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  return 0;
}

/* No BuckConverter_functionInitialEquations_lambda0 function */

int BuckConverter_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  return 0;
}


#if defined(__cplusplus)
}
#endif
