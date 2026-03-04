/* Initialization */
#include "ThreePhaseMotor_model.h"
#include "ThreePhaseMotor_11mix.h"
#include "ThreePhaseMotor_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void ThreePhaseMotor_functionInitialEquations_0(DATA *data, threadData_t *threadData);

/*
equation index: 1
type: SIMPLE_ASSIGN
i_d = $START.i_d
*/
void ThreePhaseMotor_eqFunction_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */) = (data->modelData->realVarsData[0] /* i_d STATE(1) */).attribute .start;
  TRACE_POP
}

/*
equation index: 2
type: SIMPLE_ASSIGN
i_q = $START.i_q
*/
void ThreePhaseMotor_eqFunction_2(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */) = (data->modelData->realVarsData[1] /* i_q STATE(1) */).attribute .start;
  TRACE_POP
}
extern void ThreePhaseMotor_eqFunction_13(DATA *data, threadData_t *threadData);


/*
equation index: 4
type: SIMPLE_ASSIGN
omega_m = $START.omega_m
*/
void ThreePhaseMotor_eqFunction_4(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */) = (data->modelData->realVarsData[2] /* omega_m STATE(1) */).attribute .start;
  TRACE_POP
}
extern void ThreePhaseMotor_eqFunction_12(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_eqFunction_14(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_eqFunction_15(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_eqFunction_18(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_eqFunction_17(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_eqFunction_16(DATA *data, threadData_t *threadData);


/*
equation index: 11
type: SIMPLE_ASSIGN
theta_e = $START.theta_e
*/
void ThreePhaseMotor_eqFunction_11(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) = (data->modelData->realVarsData[3] /* theta_e STATE(1,omega_e) */).attribute .start;
  TRACE_POP
}
OMC_DISABLE_OPT
void ThreePhaseMotor_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  ThreePhaseMotor_eqFunction_1(data, threadData);
  ThreePhaseMotor_eqFunction_2(data, threadData);
  ThreePhaseMotor_eqFunction_13(data, threadData);
  ThreePhaseMotor_eqFunction_4(data, threadData);
  ThreePhaseMotor_eqFunction_12(data, threadData);
  ThreePhaseMotor_eqFunction_14(data, threadData);
  ThreePhaseMotor_eqFunction_15(data, threadData);
  ThreePhaseMotor_eqFunction_18(data, threadData);
  ThreePhaseMotor_eqFunction_17(data, threadData);
  ThreePhaseMotor_eqFunction_16(data, threadData);
  ThreePhaseMotor_eqFunction_11(data, threadData);
  TRACE_POP
}

int ThreePhaseMotor_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  ThreePhaseMotor_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}

/* No ThreePhaseMotor_functionInitialEquations_lambda0 function */

int ThreePhaseMotor_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
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

