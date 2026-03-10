/* Initialization */
#include "PMSM_Motor_WithSensors_model.h"
#include "PMSM_Motor_WithSensors_11mix.h"
#include "PMSM_Motor_WithSensors_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void PMSM_Motor_WithSensors_functionInitialEquations_0(DATA *data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_38(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_39(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_40(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_41(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_42(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_43(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_44(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_45(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_46(DATA *data, threadData_t *threadData);


/*
equation index: 10
type: SIMPLE_ASSIGN
i_d = $START.i_d
*/
void PMSM_Motor_WithSensors_eqFunction_10(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,10};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */) = (data->modelData->realVarsData[0] /* i_d STATE(1) */).attribute .start;
  TRACE_POP
}

/*
equation index: 11
type: SIMPLE_ASSIGN
i_q = $START.i_q
*/
void PMSM_Motor_WithSensors_eqFunction_11(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */) = (data->modelData->realVarsData[1] /* i_q STATE(1) */).attribute .start;
  TRACE_POP
}
extern void PMSM_Motor_WithSensors_eqFunction_34(DATA *data, threadData_t *threadData);


/*
equation index: 13
type: SIMPLE_ASSIGN
omega_m = $START.omega_m
*/
void PMSM_Motor_WithSensors_eqFunction_13(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */) = (data->modelData->realVarsData[2] /* omega_m STATE(1) */).attribute .start;
  TRACE_POP
}
extern void PMSM_Motor_WithSensors_eqFunction_47(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_36(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_37(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_35(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_32(DATA *data, threadData_t *threadData);


/*
equation index: 19
type: SIMPLE_ASSIGN
theta_e = $START.theta_e
*/
void PMSM_Motor_WithSensors_eqFunction_19(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,19};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) = (data->modelData->realVarsData[3] /* theta_e STATE(1,omega_e) */).attribute .start;
  TRACE_POP
}

/*
equation index: 20
type: SIMPLE_ASSIGN
v_d = v_alpha * cos(theta_e) + v_beta * sin(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_20(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_alpha variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_beta variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  TRACE_POP
}
extern void PMSM_Motor_WithSensors_eqFunction_51(DATA *data, threadData_t *threadData);


/*
equation index: 22
type: SIMPLE_ASSIGN
v_q = v_beta * cos(theta_e) - v_alpha * sin(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_22(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,22};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_beta variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_alpha variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  TRACE_POP
}
extern void PMSM_Motor_WithSensors_eqFunction_53(DATA *data, threadData_t *threadData);


/*
equation index: 24
type: SIMPLE_ASSIGN
i_a = i_d * cos(theta_e) - i_q * sin(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_24(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,24};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  TRACE_POP
}

/*
equation index: 25
type: SIMPLE_ASSIGN
i_beta = i_d * sin(theta_e) + i_q * cos(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_25(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,25};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  TRACE_POP
}
extern void PMSM_Motor_WithSensors_eqFunction_56(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_57(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_WithSensors_eqFunction_33(DATA *data, threadData_t *threadData);


/*
equation index: 29
type: SIMPLE_ASSIGN
emf_a = lambda_pm * omega_e * sin(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_29(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,29};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf_a variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  TRACE_POP
}

/*
equation index: 30
type: SIMPLE_ASSIGN
emf_b = lambda_pm * omega_e * sin(theta_e - 2.0943951023931953)
*/
void PMSM_Motor_WithSensors_eqFunction_30(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,30};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf_b variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) - 2.0943951023931953)));
  TRACE_POP
}

/*
equation index: 31
type: SIMPLE_ASSIGN
emf_c = lambda_pm * omega_e * sin(theta_e + 2.0943951023931953)
*/
void PMSM_Motor_WithSensors_eqFunction_31(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* emf_c variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) + 2.0943951023931953)));
  TRACE_POP
}
OMC_DISABLE_OPT
void PMSM_Motor_WithSensors_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  PMSM_Motor_WithSensors_eqFunction_38(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_39(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_40(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_41(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_42(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_43(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_44(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_45(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_46(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_10(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_11(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_34(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_13(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_47(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_36(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_37(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_35(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_32(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_19(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_20(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_51(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_22(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_53(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_24(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_25(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_56(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_57(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_33(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_29(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_30(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_31(data, threadData);
  TRACE_POP
}

int PMSM_Motor_WithSensors_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  PMSM_Motor_WithSensors_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}

/* No PMSM_Motor_WithSensors_functionInitialEquations_lambda0 function */

int PMSM_Motor_WithSensors_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
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

