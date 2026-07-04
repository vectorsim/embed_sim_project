/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "PMSM_Plant_FMU_model.h"
#include "simulation/solver/events.h"
#include "util/real_array.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int PMSM_Plant_FMU_input_function(DATA *data, threadData_t *threadData)
{
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* T_load variable */) = data->simulationInfo->inputVars[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* duty_a variable */) = data->simulationInfo->inputVars[1];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* duty_b variable */) = data->simulationInfo->inputVars[2];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* duty_c variable */) = data->simulationInfo->inputVars[3];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_dc variable */) = data->simulationInfo->inputVars[4];
  
  return 0;
}

int PMSM_Plant_FMU_input_function_init(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[11].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[0] = real_get(data->modelData->realVarsData[11].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[12].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[1] = real_get(data->modelData->realVarsData[12].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[13].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[2] = real_get(data->modelData->realVarsData[13].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[14].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[3] = real_get(data->modelData->realVarsData[14].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[27].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[4] = real_get(data->modelData->realVarsData[27].attribute.start, 0);
  
  return 0;
}

int PMSM_Plant_FMU_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[11].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[0], 0, &data->modelData->realVarsData[11].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[12].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[1], 0, &data->modelData->realVarsData[12].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[13].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[2], 0, &data->modelData->realVarsData[13].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[14].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[3], 0, &data->modelData->realVarsData[14].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[27].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[4], 0, &data->modelData->realVarsData[27].attribute.start);
  
  return 0;
}

int PMSM_Plant_FMU_inputNames(DATA *data, char ** names){
  names[0] = (char *) data->modelData->realVarsData[11].info.name;
  names[1] = (char *) data->modelData->realVarsData[12].info.name;
  names[2] = (char *) data->modelData->realVarsData[13].info.name;
  names[3] = (char *) data->modelData->realVarsData[14].info.name;
  names[4] = (char *) data->modelData->realVarsData[27].info.name;
  
  return 0;
}

int PMSM_Plant_FMU_data_function(DATA *data, threadData_t *threadData)
{
  return 0;
}

int PMSM_Plant_FMU_dataReconciliationInputNames(DATA *data, char ** names){
  
  return 0;
}

int PMSM_Plant_FMU_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  
  return 0;
}

int PMSM_Plant_FMU_output_function(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* T_em variable */);
  data->simulationInfo->outputVars[1] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* ia variable */);
  data->simulationInfo->outputVars[2] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* ib variable */);
  data->simulationInfo->outputVars[3] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* ic variable */);
  data->simulationInfo->outputVars[4] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* id_out variable */);
  data->simulationInfo->outputVars[5] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* iq_out variable */);
  data->simulationInfo->outputVars[6] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* rpm variable */);
  data->simulationInfo->outputVars[7] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* theta_m variable */);
  
  return 0;
}

int PMSM_Plant_FMU_setc_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

int PMSM_Plant_FMU_setb_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}


/*
equation index: 30
type: SIMPLE_ASSIGN
rpm = 9.549296585513721 * omega_m
*/
void PMSM_Plant_FMU_eqFunction_30(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,30};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* rpm variable */) = (9.549296585513721) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 30;
}

/*
equation index: 31
type: SIMPLE_ASSIGN
theta_m = theta_e / (*Real*)(p)
*/
void PMSM_Plant_FMU_eqFunction_31(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* theta_m variable */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */),((modelica_real)(data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* p PARAM */)),"/*Real*/(p)",equationIndexes);
  threadData->lastEquationSolved = 31;
}

/*
equation index: 32
type: SIMPLE_ASSIGN
T_em = 1.5 * (*Real*)(p) * i_q * (lambda_pm + (L_d - L_q) * i_d)
*/
void PMSM_Plant_FMU_eqFunction_32(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,32};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* T_em variable */) = (1.5) * ((((modelica_real)(data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* p PARAM */))) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */) + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)))));
  threadData->lastEquationSolved = 32;
}

/*
equation index: 33
type: SIMPLE_ASSIGN
$DER.omega_m = (T_em + (-B_fric) * omega_m - T_load) / J
*/
void PMSM_Plant_FMU_eqFunction_33(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(omega_m) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* T_em variable */) + ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* B_fric PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* T_load variable */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* J PARAM */),"J",equationIndexes);
  threadData->lastEquationSolved = 33;
}

/*
equation index: 34
type: SIMPLE_ASSIGN
omega_e = (*Real*)(p) * omega_m
*/
void PMSM_Plant_FMU_eqFunction_34(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,34};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* omega_e variable */) = (((modelica_real)(data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* p PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 34;
}

/*
equation index: 35
type: SIMPLE_ASSIGN
$DER.theta_e = omega_e
*/
void PMSM_Plant_FMU_eqFunction_35(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(theta_e) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* omega_e variable */);
  threadData->lastEquationSolved = 35;
}

/*
equation index: 36
type: SIMPLE_ASSIGN
vc_leg = duty_c * v_dc
*/
void PMSM_Plant_FMU_eqFunction_36(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* vc_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* duty_c variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_dc variable */));
  threadData->lastEquationSolved = 36;
}

/*
equation index: 37
type: SIMPLE_ASSIGN
vb_leg = duty_b * v_dc
*/
void PMSM_Plant_FMU_eqFunction_37(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* vb_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* duty_b variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_dc variable */));
  threadData->lastEquationSolved = 37;
}

/*
equation index: 38
type: SIMPLE_ASSIGN
va_leg = duty_a * v_dc
*/
void PMSM_Plant_FMU_eqFunction_38(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* va_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* duty_a variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_dc variable */));
  threadData->lastEquationSolved = 38;
}

/*
equation index: 39
type: SIMPLE_ASSIGN
v_neutral = 0.3333333333333333 * (va_leg + vb_leg + vc_leg)
*/
void PMSM_Plant_FMU_eqFunction_39(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,39};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_neutral variable */) = (0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* va_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* vb_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* vc_leg variable */));
  threadData->lastEquationSolved = 39;
}

/*
equation index: 40
type: SIMPLE_ASSIGN
vb = vb_leg - v_neutral
*/
void PMSM_Plant_FMU_eqFunction_40(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* vb variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* vb_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_neutral variable */);
  threadData->lastEquationSolved = 40;
}

/*
equation index: 41
type: SIMPLE_ASSIGN
vc = vc_leg - v_neutral
*/
void PMSM_Plant_FMU_eqFunction_41(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,41};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* vc variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* vc_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_neutral variable */);
  threadData->lastEquationSolved = 41;
}

/*
equation index: 42
type: SIMPLE_ASSIGN
v_beta = 0.5773502691896258 * (vb - vc)
*/
void PMSM_Plant_FMU_eqFunction_42(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* v_beta variable */) = (0.5773502691896258) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* vb variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* vc variable */));
  threadData->lastEquationSolved = 42;
}

/*
equation index: 43
type: SIMPLE_ASSIGN
va = va_leg - v_neutral
*/
void PMSM_Plant_FMU_eqFunction_43(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* va variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* va_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_neutral variable */);
  threadData->lastEquationSolved = 43;
}

/*
equation index: 44
type: SIMPLE_ASSIGN
v_alpha = 0.6666666666666666 * va + (-0.3333333333333333) * (vb + vc)
*/
void PMSM_Plant_FMU_eqFunction_44(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* v_alpha variable */) = (0.6666666666666666) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* va variable */)) + (-0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* vb variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* vc variable */));
  threadData->lastEquationSolved = 44;
}

/*
equation index: 45
type: SIMPLE_ASSIGN
iq_out = i_q
*/
void PMSM_Plant_FMU_eqFunction_45(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* iq_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */);
  threadData->lastEquationSolved = 45;
}

/*
equation index: 46
type: SIMPLE_ASSIGN
id_out = i_d
*/
void PMSM_Plant_FMU_eqFunction_46(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,46};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* id_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */);
  threadData->lastEquationSolved = 46;
}

/*
equation index: 47
type: SIMPLE_ASSIGN
$cse2 = cos(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_47(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */) = cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  threadData->lastEquationSolved = 47;
}

/*
equation index: 48
type: SIMPLE_ASSIGN
$cse1 = sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_48(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,48};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  threadData->lastEquationSolved = 48;
}

/*
equation index: 49
type: SIMPLE_ASSIGN
v_d = v_alpha * $cse2 + v_beta * $cse1
*/
void PMSM_Plant_FMU_eqFunction_49(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */));
  threadData->lastEquationSolved = 49;
}

/*
equation index: 50
type: SIMPLE_ASSIGN
$DER.i_d = (v_d + omega_e * L_q * i_q - R * i_d) / L_d
*/
void PMSM_Plant_FMU_eqFunction_50(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,50};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(i_d) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* v_d variable */) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */))) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* R PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */))),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */),"L_d",equationIndexes);
  threadData->lastEquationSolved = 50;
}

/*
equation index: 51
type: SIMPLE_ASSIGN
v_q = v_beta * $cse2 - v_alpha * $cse1
*/
void PMSM_Plant_FMU_eqFunction_51(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  threadData->lastEquationSolved = 51;
}

/*
equation index: 52
type: SIMPLE_ASSIGN
$DER.i_q = (v_q + (-R) * i_q - omega_e * (L_d * i_d + lambda_pm)) / L_q
*/
void PMSM_Plant_FMU_eqFunction_52(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(i_q) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_q variable */) + ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* R PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) + (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */))),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */),"L_q",equationIndexes);
  threadData->lastEquationSolved = 52;
}

/*
equation index: 53
type: SIMPLE_ASSIGN
ia = i_d * $cse2 - i_q * $cse1
*/
void PMSM_Plant_FMU_eqFunction_53(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* ia variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  threadData->lastEquationSolved = 53;
}

/*
equation index: 54
type: SIMPLE_ASSIGN
i_beta = i_d * $cse1 + i_q * $cse2
*/
void PMSM_Plant_FMU_eqFunction_54(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */));
  threadData->lastEquationSolved = 54;
}

/*
equation index: 55
type: SIMPLE_ASSIGN
ib = 0.8660254037844386 * i_beta + (-0.5) * ia
*/
void PMSM_Plant_FMU_eqFunction_55(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,55};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* ib variable */) = (0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* i_beta variable */)) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* ia variable */));
  threadData->lastEquationSolved = 55;
}

/*
equation index: 56
type: SIMPLE_ASSIGN
ic = (-0.5) * ia + (-0.8660254037844386) * i_beta
*/
void PMSM_Plant_FMU_eqFunction_56(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,56};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* ic variable */) = (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* ia variable */)) + (-0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* i_beta variable */));
  threadData->lastEquationSolved = 56;
}

OMC_DISABLE_OPT
int PMSM_Plant_FMU_functionDAE(DATA *data, threadData_t *threadData)
{
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  PMSM_Plant_FMU_functionLocalKnownVars(data, threadData);
  static void (*const eqFunctions[27])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_30,
    PMSM_Plant_FMU_eqFunction_31,
    PMSM_Plant_FMU_eqFunction_32,
    PMSM_Plant_FMU_eqFunction_33,
    PMSM_Plant_FMU_eqFunction_34,
    PMSM_Plant_FMU_eqFunction_35,
    PMSM_Plant_FMU_eqFunction_36,
    PMSM_Plant_FMU_eqFunction_37,
    PMSM_Plant_FMU_eqFunction_38,
    PMSM_Plant_FMU_eqFunction_39,
    PMSM_Plant_FMU_eqFunction_40,
    PMSM_Plant_FMU_eqFunction_41,
    PMSM_Plant_FMU_eqFunction_42,
    PMSM_Plant_FMU_eqFunction_43,
    PMSM_Plant_FMU_eqFunction_44,
    PMSM_Plant_FMU_eqFunction_45,
    PMSM_Plant_FMU_eqFunction_46,
    PMSM_Plant_FMU_eqFunction_47,
    PMSM_Plant_FMU_eqFunction_48,
    PMSM_Plant_FMU_eqFunction_49,
    PMSM_Plant_FMU_eqFunction_50,
    PMSM_Plant_FMU_eqFunction_51,
    PMSM_Plant_FMU_eqFunction_52,
    PMSM_Plant_FMU_eqFunction_53,
    PMSM_Plant_FMU_eqFunction_54,
    PMSM_Plant_FMU_eqFunction_55,
    PMSM_Plant_FMU_eqFunction_56
  };
  
  for (int id = 0; id < 27; id++) {
    eqFunctions[id](data, threadData);
  }
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  return 0;
}


int PMSM_Plant_FMU_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

/* forwarded equations */
extern void PMSM_Plant_FMU_eqFunction_32(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_33(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_34(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_35(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_36(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_37(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_38(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_39(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_40(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_41(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_42(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_43(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_44(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_47(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_48(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_49(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_50(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_51(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_52(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[19])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_32,
    PMSM_Plant_FMU_eqFunction_33,
    PMSM_Plant_FMU_eqFunction_34,
    PMSM_Plant_FMU_eqFunction_35,
    PMSM_Plant_FMU_eqFunction_36,
    PMSM_Plant_FMU_eqFunction_37,
    PMSM_Plant_FMU_eqFunction_38,
    PMSM_Plant_FMU_eqFunction_39,
    PMSM_Plant_FMU_eqFunction_40,
    PMSM_Plant_FMU_eqFunction_41,
    PMSM_Plant_FMU_eqFunction_42,
    PMSM_Plant_FMU_eqFunction_43,
    PMSM_Plant_FMU_eqFunction_44,
    PMSM_Plant_FMU_eqFunction_47,
    PMSM_Plant_FMU_eqFunction_48,
    PMSM_Plant_FMU_eqFunction_49,
    PMSM_Plant_FMU_eqFunction_50,
    PMSM_Plant_FMU_eqFunction_51,
    PMSM_Plant_FMU_eqFunction_52
  };
  
  for (int id = 0; id < 19; id++) {
    eqFunctions[id](data, threadData);
  }
}

int PMSM_Plant_FMU_functionODE(DATA *data, threadData_t *threadData)
{
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  PMSM_Plant_FMU_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  return 0;
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);
extern int _main_OptimizationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);

#include "PMSM_Plant_FMU_12jac.h"
#include "PMSM_Plant_FMU_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks PMSM_Plant_FMU_callback = {
  NULL,    /* performSimulation */
  NULL,    /* performQSSSimulation */
  NULL,    /* updateContinuousSystem */
  PMSM_Plant_FMU_callExternalObjectDestructors,    /* callExternalObjectDestructors */
  NULL,    /* initialNonLinearSystem */
  NULL,    /* initialLinearSystem */
  NULL,    /* initialMixedSystem */
  #if !defined(OMC_NO_STATESELECTION)
  PMSM_Plant_FMU_initializeStateSets,
  #else
  NULL,
  #endif    /* initializeStateSets */
  PMSM_Plant_FMU_initializeDAEmodeData,
  PMSM_Plant_FMU_functionODE,
  PMSM_Plant_FMU_functionAlgebraics,
  PMSM_Plant_FMU_functionDAE,
  PMSM_Plant_FMU_functionLocalKnownVars,
  PMSM_Plant_FMU_input_function,
  PMSM_Plant_FMU_input_function_init,
  PMSM_Plant_FMU_input_function_updateStartValues,
  PMSM_Plant_FMU_data_function,
  PMSM_Plant_FMU_output_function,
  PMSM_Plant_FMU_setc_function,
  PMSM_Plant_FMU_setb_function,
  PMSM_Plant_FMU_function_storeDelayed,
  PMSM_Plant_FMU_function_storeSpatialDistribution,
  PMSM_Plant_FMU_function_initSpatialDistribution,
  PMSM_Plant_FMU_updateBoundVariableAttributes,
  PMSM_Plant_FMU_functionInitialEquations,
  GLOBAL_EQUIDISTANT_HOMOTOPY,
  NULL,
  PMSM_Plant_FMU_functionRemovedInitialEquations,
  PMSM_Plant_FMU_updateBoundParameters,
  PMSM_Plant_FMU_checkForAsserts,
  PMSM_Plant_FMU_function_ZeroCrossingsEquations,
  PMSM_Plant_FMU_function_ZeroCrossings,
  PMSM_Plant_FMU_function_updateRelations,
  PMSM_Plant_FMU_zeroCrossingDescription,
  PMSM_Plant_FMU_relationDescription,
  PMSM_Plant_FMU_function_initSample,
  PMSM_Plant_FMU_INDEX_JAC_A,
  PMSM_Plant_FMU_INDEX_JAC_B,
  PMSM_Plant_FMU_INDEX_JAC_C,
  PMSM_Plant_FMU_INDEX_JAC_D,
  PMSM_Plant_FMU_INDEX_JAC_F,
  PMSM_Plant_FMU_INDEX_JAC_H,
  PMSM_Plant_FMU_initialAnalyticJacobianA,
  PMSM_Plant_FMU_initialAnalyticJacobianB,
  PMSM_Plant_FMU_initialAnalyticJacobianC,
  PMSM_Plant_FMU_initialAnalyticJacobianD,
  PMSM_Plant_FMU_initialAnalyticJacobianF,
  PMSM_Plant_FMU_initialAnalyticJacobianH,
  PMSM_Plant_FMU_functionJacA_column,
  PMSM_Plant_FMU_functionJacB_column,
  PMSM_Plant_FMU_functionJacC_column,
  PMSM_Plant_FMU_functionJacD_column,
  PMSM_Plant_FMU_functionJacF_column,
  PMSM_Plant_FMU_functionJacH_column,
  PMSM_Plant_FMU_linear_model_frame,
  PMSM_Plant_FMU_linear_model_datarecovery_frame,
  PMSM_Plant_FMU_mayer,
  PMSM_Plant_FMU_lagrange,
  PMSM_Plant_FMU_getInputVarIndicesInOptimization,
  PMSM_Plant_FMU_pickUpBoundsForInputsInOptimization,
  PMSM_Plant_FMU_setInputData,
  PMSM_Plant_FMU_getTimeGrid,
  PMSM_Plant_FMU_symbolicInlineSystem,
  PMSM_Plant_FMU_function_initSynchronous,
  PMSM_Plant_FMU_function_updateSynchronous,
  PMSM_Plant_FMU_function_equationsSynchronous,
  PMSM_Plant_FMU_inputNames,
  PMSM_Plant_FMU_dataReconciliationInputNames,
  PMSM_Plant_FMU_dataReconciliationUnmeasuredVariables,
  PMSM_Plant_FMU_read_simulation_info,
  PMSM_Plant_FMU_read_input_fmu,
  NULL,
  NULL,
  -1,
  NULL,
  NULL,
  -1

};

#define _OMC_LIT_RESOURCE_0_name_data "Complex"
#define _OMC_LIT_RESOURCE_0_dir_data "C:/Users/CSO212/AppData/Roaming/.openmodelica/libraries/Complex 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,7,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,78,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "Modelica"
#define _OMC_LIT_RESOURCE_1_dir_data "C:/Users/CSO212/AppData/Roaming/.openmodelica/libraries/Modelica 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,8,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,79,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_2_dir_data "C:/Users/CSO212/AppData/Roaming/.openmodelica/libraries/ModelicaServices 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,16,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,87,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "PMSM_Plant_FMU"
#define _OMC_LIT_RESOURCE_3_dir_data "C:/EmbedSimProject/embed_sim_project/pmsm/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,14,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,50,_OMC_LIT_RESOURCE_3_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,8,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir)}};
void PMSM_Plant_FMU_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &PMSM_Plant_FMU_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "PMSM_Plant_FMU";
  data->modelData->modelFilePrefix = "PMSM_Plant_FMU";
  data->modelData->modelFileName = "PMSM_Motor.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "C:/EmbedSimProject/embed_sim_project/pmsm/modelica";
  data->modelData->modelGUID = "{0307ad53-9c7c-4ff7-b149-6daa230dd690}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/PMSM_Plant_FMU_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStatesArray = 4;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesRealArray = 36;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersRealArray = 7;
  data->modelData->nParametersIntegerArray = 1;
  data->modelData->nParametersBooleanArray = 0;
  data->modelData->nParametersStringArray = 0;
  data->modelData->nParametersReal = 7;
  data->modelData->nParametersInteger = 1;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nAliasRealArray = 9;
  data->modelData->nAliasIntegerArray = 0;
  data->modelData->nAliasBooleanArray = 0;
  data->modelData->nAliasStringArray = 0;
  data->modelData->nInputVars = 5;
  data->modelData->nOutputVars = 8;
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 57;
  data->modelData->nMixedSystems = 0;
  data->modelData->nLinearSystems = 0;
  data->modelData->nNonLinearSystems = 0;
  data->modelData->nStateSets = 0;
  data->modelData->nJacobians = 6;
  data->modelData->nOptimizeConstraints = 0;
  data->modelData->nOptimizeFinalConstraints = 0;
  data->modelData->nDelayExpressions = 0;
  data->modelData->nBaseClocks = 0;
  data->modelData->nSpatialDistributions = 0;
  data->modelData->nSensitivityVars = 0;
  data->modelData->nSensitivityParamVars = 0;
  data->modelData->nSetcVars = 0;
  data->modelData->ndataReconVars = 0;
  data->modelData->nSetbVars = 0;
  data->modelData->nRelatedBoundaryConditions = 0;
  data->modelData->linearizationDumpLanguage = OMC_LINEARIZE_DUMP_LANGUAGE_MODELICA;
}

static int rml_execution_failed()
{
  fflush(NULL);
  fprintf(stderr, "Execution failed!\n");
  fflush(NULL);
  return 1;
}

