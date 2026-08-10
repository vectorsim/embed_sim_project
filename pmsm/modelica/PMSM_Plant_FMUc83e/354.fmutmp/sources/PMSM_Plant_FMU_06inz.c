/* Initialization */
#include "PMSM_Plant_FMU_model.h"
#include "PMSM_Plant_FMU_11mix.h"
#include "PMSM_Plant_FMU_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void PMSM_Plant_FMU_functionInitialEquations_0(DATA *data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_36(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_37(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_38(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_39(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_40(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_41(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_42(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_43(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_44(DATA *data, threadData_t *threadData);


/*
equation index: 10
type: SIMPLE_ASSIGN
i_d = $START.i_d
*/
void PMSM_Plant_FMU_eqFunction_10(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,10};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[0] /* i_d STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 10;
}
extern void PMSM_Plant_FMU_eqFunction_46(DATA *data, threadData_t *threadData);


/*
equation index: 12
type: SIMPLE_ASSIGN
i_q = $START.i_q
*/
void PMSM_Plant_FMU_eqFunction_12(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[1] /* i_q STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 12;
}
extern void PMSM_Plant_FMU_eqFunction_45(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_32(DATA *data, threadData_t *threadData);


/*
equation index: 15
type: SIMPLE_ASSIGN
omega_m = $START.omega_m
*/
void PMSM_Plant_FMU_eqFunction_15(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[2] /* omega_m STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 15;
}
extern void PMSM_Plant_FMU_eqFunction_34(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_35(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_33(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_30(DATA *data, threadData_t *threadData);


/*
equation index: 20
type: SIMPLE_ASSIGN
theta_e = $START.theta_e
*/
void PMSM_Plant_FMU_eqFunction_20(DATA *data, threadData_t *threadData)
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
void PMSM_Plant_FMU_eqFunction_21(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,21};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* v_alpha variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* v_beta variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  threadData->lastEquationSolved = 21;
}
extern void PMSM_Plant_FMU_eqFunction_50(DATA *data, threadData_t *threadData);


/*
equation index: 23
type: SIMPLE_ASSIGN
v_q = v_beta * cos(theta_e) - v_alpha * sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_23(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,23};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* v_beta variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* v_alpha variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 23;
}
extern void PMSM_Plant_FMU_eqFunction_52(DATA *data, threadData_t *threadData);


/*
equation index: 25
type: SIMPLE_ASSIGN
ia = i_d * cos(theta_e) - i_q * sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_25(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,25};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* ia variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 25;
}

/*
equation index: 26
type: SIMPLE_ASSIGN
i_beta = i_d * sin(theta_e) + i_q * cos(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_26(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,26};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  threadData->lastEquationSolved = 26;
}
extern void PMSM_Plant_FMU_eqFunction_55(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_56(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_31(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void PMSM_Plant_FMU_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[29])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_36,
    PMSM_Plant_FMU_eqFunction_37,
    PMSM_Plant_FMU_eqFunction_38,
    PMSM_Plant_FMU_eqFunction_39,
    PMSM_Plant_FMU_eqFunction_40,
    PMSM_Plant_FMU_eqFunction_41,
    PMSM_Plant_FMU_eqFunction_42,
    PMSM_Plant_FMU_eqFunction_43,
    PMSM_Plant_FMU_eqFunction_44,
    PMSM_Plant_FMU_eqFunction_10,
    PMSM_Plant_FMU_eqFunction_46,
    PMSM_Plant_FMU_eqFunction_12,
    PMSM_Plant_FMU_eqFunction_45,
    PMSM_Plant_FMU_eqFunction_32,
    PMSM_Plant_FMU_eqFunction_15,
    PMSM_Plant_FMU_eqFunction_34,
    PMSM_Plant_FMU_eqFunction_35,
    PMSM_Plant_FMU_eqFunction_33,
    PMSM_Plant_FMU_eqFunction_30,
    PMSM_Plant_FMU_eqFunction_20,
    PMSM_Plant_FMU_eqFunction_21,
    PMSM_Plant_FMU_eqFunction_50,
    PMSM_Plant_FMU_eqFunction_23,
    PMSM_Plant_FMU_eqFunction_52,
    PMSM_Plant_FMU_eqFunction_25,
    PMSM_Plant_FMU_eqFunction_26,
    PMSM_Plant_FMU_eqFunction_55,
    PMSM_Plant_FMU_eqFunction_56,
    PMSM_Plant_FMU_eqFunction_31
  };
  
  for (int id = 0; id < 29; id++) {
    eqFunctions[id](data, threadData);
  }
}

int PMSM_Plant_FMU_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->discreteCall = 1;
  PMSM_Plant_FMU_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  return 0;
}

/* No PMSM_Plant_FMU_functionInitialEquations_lambda0 function */

int PMSM_Plant_FMU_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  return 0;
}


#if defined(__cplusplus)
}
#endif
