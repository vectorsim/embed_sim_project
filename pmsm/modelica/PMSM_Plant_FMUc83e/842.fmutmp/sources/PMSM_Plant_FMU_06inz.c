/* Initialization */
#include "PMSM_Plant_FMU_model.h"
#include "PMSM_Plant_FMU_11mix.h"
#include "PMSM_Plant_FMU_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void PMSM_Plant_FMU_functionInitialEquations_0(DATA *data, threadData_t *threadData);

/*
equation index: 1
type: SIMPLE_ASSIGN
duty_c_lim = max(0.0, min(1.0, duty_c))
*/
void PMSM_Plant_FMU_eqFunction_1(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* duty_c_lim variable */) = fmax(0.0,fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* duty_c variable */)));
  threadData->lastEquationSolved = 1;
}

/*
equation index: 2
type: SIMPLE_ASSIGN
duty_b_lim = max(0.0, min(1.0, duty_b))
*/
void PMSM_Plant_FMU_eqFunction_2(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,2};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* duty_b_lim variable */) = fmax(0.0,fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* duty_b variable */)));
  threadData->lastEquationSolved = 2;
}

/*
equation index: 3
type: SIMPLE_ASSIGN
duty_a_lim = max(0.0, min(1.0, duty_a))
*/
void PMSM_Plant_FMU_eqFunction_3(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,3};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* duty_a_lim variable */) = fmax(0.0,fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* duty_a variable */)));
  threadData->lastEquationSolved = 3;
}

/*
equation index: 4
type: SIMPLE_ASSIGN
i_d = $START.i_d
*/
void PMSM_Plant_FMU_eqFunction_4(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,4};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[0] /* i_d STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 4;
}
extern void PMSM_Plant_FMU_eqFunction_51(DATA *data, threadData_t *threadData);


/*
equation index: 6
type: SIMPLE_ASSIGN
i_q = $START.i_q
*/
void PMSM_Plant_FMU_eqFunction_6(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,6};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[1] /* i_q STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 6;
}
extern void PMSM_Plant_FMU_eqFunction_50(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_44(DATA *data, threadData_t *threadData);


/*
equation index: 9
type: SIMPLE_ASSIGN
omega_m = $START.omega_m
*/
void PMSM_Plant_FMU_eqFunction_9(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,9};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[2] /* omega_m STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 9;
}
extern void PMSM_Plant_FMU_eqFunction_46(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_47(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_45(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_42(DATA *data, threadData_t *threadData);


/*
equation index: 14
type: SIMPLE_ASSIGN
theta_e = $START.theta_e
*/
void PMSM_Plant_FMU_eqFunction_14(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,14};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) = ((modelica_real *)((data->modelData->realVarsData[3] /* theta_e STATE(1,omega_e) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 14;
}

/*
equation index: 15
type: SIMPLE_ASSIGN
ia = i_d * cos(theta_e) - i_q * sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_15(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* ia variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 15;
}

/*
equation index: 16
type: SIMPLE_ASSIGN
i_beta = i_d * sin(theta_e) + i_q * cos(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_16(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,16};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  threadData->lastEquationSolved = 16;
}
extern void PMSM_Plant_FMU_eqFunction_84(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_85(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_43(DATA *data, threadData_t *threadData);


/*
equation index: 20
type: SIMPLE_ASSIGN
T_pwm = 1.0 / f_pwm
*/
void PMSM_Plant_FMU_eqFunction_20(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */) = DIVISION_SIM(1.0,(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[7]] /* f_pwm PARAM */),"f_pwm",equationIndexes);
  threadData->lastEquationSolved = 20;
}

/*
equation index: 21
type: SIMPLE_ASSIGN
pwm_time = mod(time, T_pwm)
*/
void PMSM_Plant_FMU_eqFunction_21(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,21};
  modelica_real tmp0;
  tmp0 = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */);
  if (tmp0 == 0) {throwStreamPrint(threadData, "Division by zero %s", "mod(time, T_pwm)");}
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */) = modelica_real_mod(data->localData[0]->timeValue, tmp0);
  threadData->lastEquationSolved = 21;
}
extern void PMSM_Plant_FMU_eqFunction_49(DATA *data, threadData_t *threadData);


/*
equation index: 23
type: SIMPLE_ASSIGN
duty_a_eff = max(0.0, min(1.0, duty_a_lim - 2.0 * dead_time / T_pwm))
*/
void PMSM_Plant_FMU_eqFunction_23(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,23};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */) = fmax(0.0,fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* duty_a_lim variable */) - (DIVISION_SIM((2.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* dead_time PARAM */)),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes))));
  threadData->lastEquationSolved = 23;
}
extern void PMSM_Plant_FMU_eqFunction_56(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_57(DATA *data, threadData_t *threadData);


/*
equation index: 26
type: SIMPLE_ASSIGN
duty_b_eff = max(0.0, min(1.0, duty_b_lim - 2.0 * dead_time / T_pwm))
*/
void PMSM_Plant_FMU_eqFunction_26(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,26};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */) = fmax(0.0,fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* duty_b_lim variable */) - (DIVISION_SIM((2.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* dead_time PARAM */)),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes))));
  threadData->lastEquationSolved = 26;
}
extern void PMSM_Plant_FMU_eqFunction_62(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_63(DATA *data, threadData_t *threadData);


/*
equation index: 29
type: SIMPLE_ASSIGN
duty_c_eff = max(0.0, min(1.0, duty_c_lim - 2.0 * dead_time / T_pwm))
*/
void PMSM_Plant_FMU_eqFunction_29(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,29};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */) = fmax(0.0,fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* duty_c_lim variable */) - (DIVISION_SIM((2.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* dead_time PARAM */)),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes))));
  threadData->lastEquationSolved = 29;
}
extern void PMSM_Plant_FMU_eqFunction_68(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_69(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_70(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_71(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_72(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_73(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_74(DATA *data, threadData_t *threadData);

extern void PMSM_Plant_FMU_eqFunction_75(DATA *data, threadData_t *threadData);


/*
equation index: 38
type: SIMPLE_ASSIGN
v_d = v_alpha * cos(theta_e) + v_beta * sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_38(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_alpha variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_beta variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */)));
  threadData->lastEquationSolved = 38;
}
extern void PMSM_Plant_FMU_eqFunction_79(DATA *data, threadData_t *threadData);


/*
equation index: 40
type: SIMPLE_ASSIGN
v_q = v_beta * cos(theta_e) - v_alpha * sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_40(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[44]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_beta variable */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_alpha variable */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */))));
  threadData->lastEquationSolved = 40;
}
extern void PMSM_Plant_FMU_eqFunction_81(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void PMSM_Plant_FMU_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[41])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_1,
    PMSM_Plant_FMU_eqFunction_2,
    PMSM_Plant_FMU_eqFunction_3,
    PMSM_Plant_FMU_eqFunction_4,
    PMSM_Plant_FMU_eqFunction_51,
    PMSM_Plant_FMU_eqFunction_6,
    PMSM_Plant_FMU_eqFunction_50,
    PMSM_Plant_FMU_eqFunction_44,
    PMSM_Plant_FMU_eqFunction_9,
    PMSM_Plant_FMU_eqFunction_46,
    PMSM_Plant_FMU_eqFunction_47,
    PMSM_Plant_FMU_eqFunction_45,
    PMSM_Plant_FMU_eqFunction_42,
    PMSM_Plant_FMU_eqFunction_14,
    PMSM_Plant_FMU_eqFunction_15,
    PMSM_Plant_FMU_eqFunction_16,
    PMSM_Plant_FMU_eqFunction_84,
    PMSM_Plant_FMU_eqFunction_85,
    PMSM_Plant_FMU_eqFunction_43,
    PMSM_Plant_FMU_eqFunction_20,
    PMSM_Plant_FMU_eqFunction_21,
    PMSM_Plant_FMU_eqFunction_49,
    PMSM_Plant_FMU_eqFunction_23,
    PMSM_Plant_FMU_eqFunction_56,
    PMSM_Plant_FMU_eqFunction_57,
    PMSM_Plant_FMU_eqFunction_26,
    PMSM_Plant_FMU_eqFunction_62,
    PMSM_Plant_FMU_eqFunction_63,
    PMSM_Plant_FMU_eqFunction_29,
    PMSM_Plant_FMU_eqFunction_68,
    PMSM_Plant_FMU_eqFunction_69,
    PMSM_Plant_FMU_eqFunction_70,
    PMSM_Plant_FMU_eqFunction_71,
    PMSM_Plant_FMU_eqFunction_72,
    PMSM_Plant_FMU_eqFunction_73,
    PMSM_Plant_FMU_eqFunction_74,
    PMSM_Plant_FMU_eqFunction_75,
    PMSM_Plant_FMU_eqFunction_38,
    PMSM_Plant_FMU_eqFunction_79,
    PMSM_Plant_FMU_eqFunction_40,
    PMSM_Plant_FMU_eqFunction_81
  };
  
  for (int id = 0; id < 41; id++) {
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
