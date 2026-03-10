/* Initialization */
#include "ThreePhaseMotor_PWM_model.h"
#include "ThreePhaseMotor_PWM_11mix.h"
#include "ThreePhaseMotor_PWM_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void ThreePhaseMotor_PWM_functionInitialEquations_0(DATA *data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_36(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_35(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_34(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_37(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_39(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_40(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_41(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_38(DATA *data, threadData_t *threadData);


/*
equation index: 9
type: SIMPLE_ASSIGN
i_d = $START.i_d
*/
void ThreePhaseMotor_PWM_eqFunction_9(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,9};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */) = (data->modelData->realVarsData[0] /* i_d STATE(1) */).attribute .start;
  TRACE_POP
}

/*
equation index: 10
type: SIMPLE_ASSIGN
i_q = $START.i_q
*/
void ThreePhaseMotor_PWM_eqFunction_10(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,10};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */) = (data->modelData->realVarsData[1] /* i_q STATE(1) */).attribute .start;
  TRACE_POP
}
extern void ThreePhaseMotor_PWM_eqFunction_47(DATA *data, threadData_t *threadData);


/*
equation index: 12
type: SIMPLE_ASSIGN
omega_m = $START.omega_m
*/
void ThreePhaseMotor_PWM_eqFunction_12(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */) = (data->modelData->realVarsData[2] /* omega_m STATE(1) */).attribute .start;
  TRACE_POP
}
extern void ThreePhaseMotor_PWM_eqFunction_44(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_49(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_48(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_58(DATA *data, threadData_t *threadData);


/*
equation index: 17
type: SIMPLE_ASSIGN
theta_e = $START.theta_e
*/
void ThreePhaseMotor_PWM_eqFunction_17(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,17};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) = (data->modelData->realVarsData[3] /* theta_e STATE(1,omega_e) */).attribute .start;
  TRACE_POP
}

/*
equation index: 18
type: SIMPLE_ASSIGN
v_d_cmd = v_alpha * cos(theta_e) + v_beta * sin(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_18(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,18};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_d_cmd variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_alpha variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_beta variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  TRACE_POP
}
extern void ThreePhaseMotor_PWM_eqFunction_45(DATA *data, threadData_t *threadData);


/*
equation index: 20
type: SIMPLE_ASSIGN
v_q_cmd = v_beta * cos(theta_e) - v_alpha * sin(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_20(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_q_cmd variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_beta variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_alpha variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  TRACE_POP
}
extern void ThreePhaseMotor_PWM_eqFunction_46(DATA *data, threadData_t *threadData);


/*
equation index: 22
type: SIMPLE_ASSIGN
i_a = i_d * cos(theta_e) - i_q * sin(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_22(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,22};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  TRACE_POP
}

/*
equation index: 23
type: SIMPLE_ASSIGN
i_beta = i_d * sin(theta_e) + i_q * cos(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_23(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,23};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  TRACE_POP
}
extern void ThreePhaseMotor_PWM_eqFunction_52(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_53(DATA *data, threadData_t *threadData);

extern void ThreePhaseMotor_PWM_eqFunction_54(DATA *data, threadData_t *threadData);


/*
equation index: 27
type: SIMPLE_ASSIGN
emf_a = lambda_pm * omega_e * sin(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_27(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,27};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf_a variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  TRACE_POP
}

/*
equation index: 28
type: SIMPLE_ASSIGN
emf_b = lambda_pm * omega_e * sin(theta_e - 2.0943951023931953)
*/
void ThreePhaseMotor_PWM_eqFunction_28(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,28};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf_b variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) - 2.0943951023931953)));
  TRACE_POP
}

/*
equation index: 29
type: SIMPLE_ASSIGN
emf_c = lambda_pm * omega_e * sin(theta_e + 2.0943951023931953)
*/
void ThreePhaseMotor_PWM_eqFunction_29(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,29};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* emf_c variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) + 2.0943951023931953)));
  TRACE_POP
}
OMC_DISABLE_OPT
void ThreePhaseMotor_PWM_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  ThreePhaseMotor_PWM_eqFunction_36(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_35(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_34(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_37(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_39(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_40(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_41(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_38(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_9(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_10(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_47(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_12(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_44(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_49(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_48(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_58(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_17(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_18(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_45(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_20(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_46(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_22(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_23(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_52(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_53(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_54(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_27(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_28(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_29(data, threadData);
  TRACE_POP
}

int ThreePhaseMotor_PWM_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->discreteCall = 1;
  ThreePhaseMotor_PWM_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  TRACE_POP
  return 0;
}

/* No ThreePhaseMotor_PWM_functionInitialEquations_lambda0 function */

int ThreePhaseMotor_PWM_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
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

