/* Initialization */
#include "PMSM_Motor_model.h"
#include "PMSM_Motor_11mix.h"
#include "PMSM_Motor_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void PMSM_Motor_functionInitialEquations_0(DATA *data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_42(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_43(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_44(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_45(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_46(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_47(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_48(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_49(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_50(DATA *data, threadData_t *threadData);


/*
equation index: 10
type: SIMPLE_ASSIGN
i_d = $START.i_d
*/
void PMSM_Motor_eqFunction_10(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,10};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[0] /* i_d STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 10;
}

/*
equation index: 11
type: SIMPLE_ASSIGN
i_q = $START.i_q
*/
void PMSM_Motor_eqFunction_11(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[1] /* i_q STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 11;
}
extern void PMSM_Motor_eqFunction_37(DATA *data, threadData_t *threadData);


/*
equation index: 13
type: SIMPLE_ASSIGN
omega_m = $START.omega_m
*/
void PMSM_Motor_eqFunction_13(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,13};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[2] /* omega_m STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 13;
}
extern void PMSM_Motor_eqFunction_51(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_40(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_41(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_38(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_35(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_39(DATA *data, threadData_t *threadData);


/*
equation index: 20
type: SIMPLE_ASSIGN
theta_e = $START.theta_e
*/
void PMSM_Motor_eqFunction_20(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) = ((modelica_real *)((data->modelData->realVarsData[3] /* theta_e STATE(1,omega_e) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 20;
}

/*
equation index: 21
type: SIMPLE_ASSIGN
v_d = v_alpha * cos(theta_e) + v_beta * sin(theta_e)
*/
void PMSM_Motor_eqFunction_21(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,21};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_alpha variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_beta variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  threadData->lastEquationSolved = 21;
}
extern void PMSM_Motor_eqFunction_55(DATA *data, threadData_t *threadData);


/*
equation index: 23
type: SIMPLE_ASSIGN
v_q = v_beta * cos(theta_e) - v_alpha * sin(theta_e)
*/
void PMSM_Motor_eqFunction_23(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,23};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_beta variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_alpha variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 23;
}
extern void PMSM_Motor_eqFunction_57(DATA *data, threadData_t *threadData);


/*
equation index: 25
type: SIMPLE_ASSIGN
i_a = i_d * cos(theta_e) - i_q * sin(theta_e)
*/
void PMSM_Motor_eqFunction_25(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,25};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_a variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 25;
}

/*
equation index: 26
type: SIMPLE_ASSIGN
i_beta = i_d * sin(theta_e) + i_q * cos(theta_e)
*/
void PMSM_Motor_eqFunction_26(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,26};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  threadData->lastEquationSolved = 26;
}
extern void PMSM_Motor_eqFunction_60(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_61(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_62(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_63(DATA *data, threadData_t *threadData);

extern void PMSM_Motor_eqFunction_36(DATA *data, threadData_t *threadData);


/*
equation index: 32
type: SIMPLE_ASSIGN
emf_a = lambda_pm * omega_e * sin(theta_e)
*/
void PMSM_Motor_eqFunction_32(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,32};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* emf_a variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 32;
}

/*
equation index: 33
type: SIMPLE_ASSIGN
emf_b = lambda_pm * omega_e * sin(theta_e - 2.0943951023931953)
*/
void PMSM_Motor_eqFunction_33(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* emf_b variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) - 2.0943951023931953)));
  threadData->lastEquationSolved = 33;
}

/*
equation index: 34
type: SIMPLE_ASSIGN
emf_c = lambda_pm * omega_e * sin(theta_e + 2.0943951023931953)
*/
void PMSM_Motor_eqFunction_34(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,34};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* emf_c variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) + 2.0943951023931953)));
  threadData->lastEquationSolved = 34;
}
OMC_DISABLE_OPT
void PMSM_Motor_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[34])(DATA*, threadData_t*) = {
    PMSM_Motor_eqFunction_42,
    PMSM_Motor_eqFunction_43,
    PMSM_Motor_eqFunction_44,
    PMSM_Motor_eqFunction_45,
    PMSM_Motor_eqFunction_46,
    PMSM_Motor_eqFunction_47,
    PMSM_Motor_eqFunction_48,
    PMSM_Motor_eqFunction_49,
    PMSM_Motor_eqFunction_50,
    PMSM_Motor_eqFunction_10,
    PMSM_Motor_eqFunction_11,
    PMSM_Motor_eqFunction_37,
    PMSM_Motor_eqFunction_13,
    PMSM_Motor_eqFunction_51,
    PMSM_Motor_eqFunction_40,
    PMSM_Motor_eqFunction_41,
    PMSM_Motor_eqFunction_38,
    PMSM_Motor_eqFunction_35,
    PMSM_Motor_eqFunction_39,
    PMSM_Motor_eqFunction_20,
    PMSM_Motor_eqFunction_21,
    PMSM_Motor_eqFunction_55,
    PMSM_Motor_eqFunction_23,
    PMSM_Motor_eqFunction_57,
    PMSM_Motor_eqFunction_25,
    PMSM_Motor_eqFunction_26,
    PMSM_Motor_eqFunction_60,
    PMSM_Motor_eqFunction_61,
    PMSM_Motor_eqFunction_62,
    PMSM_Motor_eqFunction_63,
    PMSM_Motor_eqFunction_36,
    PMSM_Motor_eqFunction_32,
    PMSM_Motor_eqFunction_33,
    PMSM_Motor_eqFunction_34
  };
  
  for (int id = 0; id < 34; id++) {
    eqFunctions[id](data, threadData);
  }
}

int PMSM_Motor_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->discreteCall = 1;
  PMSM_Motor_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  return 0;
}

/* No PMSM_Motor_functionInitialEquations_lambda0 function */

int PMSM_Motor_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  return 0;
}


#if defined(__cplusplus)
}
#endif
