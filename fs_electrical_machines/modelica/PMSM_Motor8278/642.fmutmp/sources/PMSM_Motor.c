/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "PMSM_Motor_model.h"
#include "simulation/solver/events.h"
#include "util/real_array.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int PMSM_Motor_input_function(DATA *data, threadData_t *threadData)
{
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* T_load variable */) = data->simulationInfo->inputVars[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* duty_a variable */) = data->simulationInfo->inputVars[1];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* duty_b variable */) = data->simulationInfo->inputVars[2];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* duty_c variable */) = data->simulationInfo->inputVars[3];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_dc variable */) = data->simulationInfo->inputVars[4];
  
  return 0;
}

int PMSM_Motor_input_function_init(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[15].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[0] = real_get(data->modelData->realVarsData[15].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[16].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[1] = real_get(data->modelData->realVarsData[16].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[17].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[2] = real_get(data->modelData->realVarsData[17].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[18].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[3] = real_get(data->modelData->realVarsData[18].attribute.start, 0);
  assertStreamPrint(threadData, data->modelData->realVarsData[40].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[4] = real_get(data->modelData->realVarsData[40].attribute.start, 0);
  
  return 0;
}

int PMSM_Motor_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[15].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[0], 0, &data->modelData->realVarsData[15].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[16].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[1], 0, &data->modelData->realVarsData[16].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[17].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[2], 0, &data->modelData->realVarsData[17].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[18].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[3], 0, &data->modelData->realVarsData[18].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[40].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[4], 0, &data->modelData->realVarsData[40].attribute.start);
  
  return 0;
}

int PMSM_Motor_inputNames(DATA *data, char ** names){
  names[0] = (char *) data->modelData->realVarsData[15].info.name;
  names[1] = (char *) data->modelData->realVarsData[16].info.name;
  names[2] = (char *) data->modelData->realVarsData[17].info.name;
  names[3] = (char *) data->modelData->realVarsData[18].info.name;
  names[4] = (char *) data->modelData->realVarsData[40].info.name;
  
  return 0;
}

int PMSM_Motor_data_function(DATA *data, threadData_t *threadData)
{
  return 0;
}

int PMSM_Motor_dataReconciliationInputNames(DATA *data, char ** names){
  
  return 0;
}

int PMSM_Motor_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  
  return 0;
}

int PMSM_Motor_output_function(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */);
  data->simulationInfo->outputVars[1] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* P_out variable */);
  data->simulationInfo->outputVars[2] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* T_em_out variable */);
  data->simulationInfo->outputVars[3] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* efficiency variable */);
  data->simulationInfo->outputVars[4] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* emf_a variable */);
  data->simulationInfo->outputVars[5] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* emf_b variable */);
  data->simulationInfo->outputVars[6] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* emf_c variable */);
  data->simulationInfo->outputVars[7] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_a variable */);
  data->simulationInfo->outputVars[8] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* i_b variable */);
  data->simulationInfo->outputVars[9] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* i_c variable */);
  data->simulationInfo->outputVars[10] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* omega_m_out variable */);
  data->simulationInfo->outputVars[11] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* speed_rpm variable */);
  data->simulationInfo->outputVars[12] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* theta_m variable */);
  
  return 0;
}

int PMSM_Motor_setc_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

int PMSM_Motor_setb_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}


/*
equation index: 35
type: SIMPLE_ASSIGN
speed_rpm = 9.549296585513721 * omega_m
*/
void PMSM_Motor_eqFunction_35(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* speed_rpm variable */) = (9.549296585513721) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 35;
}

/*
equation index: 36
type: SIMPLE_ASSIGN
theta_m = theta_e / p
*/
void PMSM_Motor_eqFunction_36(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* theta_m variable */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* p PARAM */),"p",equationIndexes);
  threadData->lastEquationSolved = 36;
}

/*
equation index: 37
type: SIMPLE_ASSIGN
T_em_out = 1.5 * p * i_q * (lambda_pm + (L_d - L_q) * i_d)
*/
void PMSM_Motor_eqFunction_37(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* T_em_out variable */) = (1.5) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* p PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */) + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)))));
  threadData->lastEquationSolved = 37;
}

/*
equation index: 38
type: SIMPLE_ASSIGN
$DER.omega_m = (T_em_out + (-B) * omega_m - T_load) / J
*/
void PMSM_Motor_eqFunction_38(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(omega_m) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* T_em_out variable */) + ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* B PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* T_load variable */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* J PARAM */),"J",equationIndexes);
  threadData->lastEquationSolved = 38;
}

/*
equation index: 39
type: SIMPLE_ASSIGN
P_out = T_em_out * omega_m
*/
void PMSM_Motor_eqFunction_39(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,39};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* P_out variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* T_em_out variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 39;
}

/*
equation index: 40
type: SIMPLE_ASSIGN
omega_e = p * omega_m
*/
void PMSM_Motor_eqFunction_40(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* p PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 40;
}

/*
equation index: 41
type: SIMPLE_ASSIGN
$DER.theta_e = omega_e
*/
void PMSM_Motor_eqFunction_41(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,41};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(theta_e) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */);
  threadData->lastEquationSolved = 41;
}

/*
equation index: 42
type: SIMPLE_ASSIGN
v_c_leg = duty_c * v_dc
*/
void PMSM_Motor_eqFunction_42(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_c_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* duty_c variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_dc variable */));
  threadData->lastEquationSolved = 42;
}

/*
equation index: 43
type: SIMPLE_ASSIGN
v_b_leg = duty_b * v_dc
*/
void PMSM_Motor_eqFunction_43(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_b_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* duty_b variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_dc variable */));
  threadData->lastEquationSolved = 43;
}

/*
equation index: 44
type: SIMPLE_ASSIGN
v_a_leg = duty_a * v_dc
*/
void PMSM_Motor_eqFunction_44(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_a_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* duty_a variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_dc variable */));
  threadData->lastEquationSolved = 44;
}

/*
equation index: 45
type: SIMPLE_ASSIGN
v_neutral = 0.3333333333333333 * (v_a_leg + v_b_leg + v_c_leg)
*/
void PMSM_Motor_eqFunction_45(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_neutral variable */) = (0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_a_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_b_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_c_leg variable */));
  threadData->lastEquationSolved = 45;
}

/*
equation index: 46
type: SIMPLE_ASSIGN
v_b = v_b_leg - v_neutral
*/
void PMSM_Motor_eqFunction_46(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,46};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_b variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_b_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_neutral variable */);
  threadData->lastEquationSolved = 46;
}

/*
equation index: 47
type: SIMPLE_ASSIGN
v_c = v_c_leg - v_neutral
*/
void PMSM_Motor_eqFunction_47(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_c variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_c_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_neutral variable */);
  threadData->lastEquationSolved = 47;
}

/*
equation index: 48
type: SIMPLE_ASSIGN
v_beta = 0.5773502691896257 * (v_b - v_c)
*/
void PMSM_Motor_eqFunction_48(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,48};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_beta variable */) = (0.5773502691896257) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_b variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_c variable */));
  threadData->lastEquationSolved = 48;
}

/*
equation index: 49
type: SIMPLE_ASSIGN
v_a = v_a_leg - v_neutral
*/
void PMSM_Motor_eqFunction_49(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_a variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_a_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_neutral variable */);
  threadData->lastEquationSolved = 49;
}

/*
equation index: 50
type: SIMPLE_ASSIGN
v_alpha = 0.6666666666666666 * (v_a + (-0.5) * (v_b + v_c))
*/
void PMSM_Motor_eqFunction_50(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,50};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_alpha variable */) = (0.6666666666666666) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_a variable */) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_b variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_c variable */)));
  threadData->lastEquationSolved = 50;
}

/*
equation index: 51
type: SIMPLE_ASSIGN
omega_m_out = omega_m
*/
void PMSM_Motor_eqFunction_51(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* omega_m_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */);
  threadData->lastEquationSolved = 51;
}

/*
equation index: 52
type: SIMPLE_ASSIGN
$cse4 = cos(theta_e)
*/
void PMSM_Motor_eqFunction_52(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */) = cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  threadData->lastEquationSolved = 52;
}

/*
equation index: 53
type: SIMPLE_ASSIGN
$cse3 = sin(theta_e)
*/
void PMSM_Motor_eqFunction_53(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  threadData->lastEquationSolved = 53;
}

/*
equation index: 54
type: SIMPLE_ASSIGN
v_d = v_alpha * $cse4 + v_beta * $cse3
*/
void PMSM_Motor_eqFunction_54(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */));
  threadData->lastEquationSolved = 54;
}

/*
equation index: 55
type: SIMPLE_ASSIGN
$DER.i_d = (v_d + omega_e * L_q * i_q - R * i_d) / L_d
*/
void PMSM_Motor_eqFunction_55(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,55};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(i_d) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_d variable */) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */))) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* R PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */))),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */),"L_d",equationIndexes);
  threadData->lastEquationSolved = 55;
}

/*
equation index: 56
type: SIMPLE_ASSIGN
v_q = v_beta * $cse4 - v_alpha * $cse3
*/
void PMSM_Motor_eqFunction_56(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,56};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  threadData->lastEquationSolved = 56;
}

/*
equation index: 57
type: SIMPLE_ASSIGN
$DER.i_q = (v_q + (-R) * i_q - omega_e * (L_d * i_d + lambda_pm)) / L_q
*/
void PMSM_Motor_eqFunction_57(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,57};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(i_q) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_q variable */) + ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* R PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) + (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */))),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */),"L_q",equationIndexes);
  threadData->lastEquationSolved = 57;
}

/*
equation index: 58
type: SIMPLE_ASSIGN
i_a = i_d * $cse4 - i_q * $cse3
*/
void PMSM_Motor_eqFunction_58(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,58};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_a variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  threadData->lastEquationSolved = 58;
}

/*
equation index: 59
type: SIMPLE_ASSIGN
i_beta = i_d * $cse3 + i_q * $cse4
*/
void PMSM_Motor_eqFunction_59(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,59};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */));
  threadData->lastEquationSolved = 59;
}

/*
equation index: 60
type: SIMPLE_ASSIGN
i_b = 0.8660254037844386 * i_beta + (-0.5) * i_a
*/
void PMSM_Motor_eqFunction_60(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,60};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* i_b variable */) = (0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* i_beta variable */)) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_a variable */));
  threadData->lastEquationSolved = 60;
}

/*
equation index: 61
type: SIMPLE_ASSIGN
i_c = (-0.5) * i_a + (-0.8660254037844386) * i_beta
*/
void PMSM_Motor_eqFunction_61(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,61};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* i_c variable */) = (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_a variable */)) + (-0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* i_beta variable */));
  threadData->lastEquationSolved = 61;
}

/*
equation index: 62
type: SIMPLE_ASSIGN
P_in = v_a * i_a + v_b * i_b + v_c * i_c
*/
void PMSM_Motor_eqFunction_62(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,62};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_a variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_a variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_b variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* i_b variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_c variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* i_c variable */));
  threadData->lastEquationSolved = 62;
}

/*
equation index: 63
type: SIMPLE_ASSIGN
efficiency = if P_in > 0.0 then P_out / P_in * 100.0 else 0.0
*/
void PMSM_Motor_eqFunction_63(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,63};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  tmp1 = 1.0;
  tmp2 = 0.0;
  relationhysteresis(data, &tmp0, (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */), 0.0, tmp1, tmp2, 0, Greater, GreaterZC);
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* efficiency variable */) = (tmp0?(DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* P_out variable */),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* P_in variable */),"P_in",equationIndexes)) * (100.0):0.0);
  threadData->lastEquationSolved = 63;
}

/*
equation index: 64
type: SIMPLE_ASSIGN
emf_a = lambda_pm * omega_e * $cse3
*/
void PMSM_Motor_eqFunction_64(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,64};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* emf_a variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  threadData->lastEquationSolved = 64;
}

/*
equation index: 65
type: SIMPLE_ASSIGN
$cse2 = sin(theta_e - 2.0943951023931953)
*/
void PMSM_Motor_eqFunction_65(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,65};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) - 2.0943951023931953);
  threadData->lastEquationSolved = 65;
}

/*
equation index: 66
type: SIMPLE_ASSIGN
emf_b = lambda_pm * omega_e * $cse2
*/
void PMSM_Motor_eqFunction_66(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,66};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* emf_b variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)));
  threadData->lastEquationSolved = 66;
}

/*
equation index: 67
type: SIMPLE_ASSIGN
$cse1 = sin(theta_e + 2.0943951023931953)
*/
void PMSM_Motor_eqFunction_67(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,67};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) + 2.0943951023931953);
  threadData->lastEquationSolved = 67;
}

/*
equation index: 68
type: SIMPLE_ASSIGN
emf_c = lambda_pm * omega_e * $cse1
*/
void PMSM_Motor_eqFunction_68(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,68};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* emf_c variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  threadData->lastEquationSolved = 68;
}

OMC_DISABLE_OPT
int PMSM_Motor_functionDAE(DATA *data, threadData_t *threadData)
{
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  PMSM_Motor_functionLocalKnownVars(data, threadData);
  static void (*const eqFunctions[34])(DATA*, threadData_t*) = {
    PMSM_Motor_eqFunction_35,
    PMSM_Motor_eqFunction_36,
    PMSM_Motor_eqFunction_37,
    PMSM_Motor_eqFunction_38,
    PMSM_Motor_eqFunction_39,
    PMSM_Motor_eqFunction_40,
    PMSM_Motor_eqFunction_41,
    PMSM_Motor_eqFunction_42,
    PMSM_Motor_eqFunction_43,
    PMSM_Motor_eqFunction_44,
    PMSM_Motor_eqFunction_45,
    PMSM_Motor_eqFunction_46,
    PMSM_Motor_eqFunction_47,
    PMSM_Motor_eqFunction_48,
    PMSM_Motor_eqFunction_49,
    PMSM_Motor_eqFunction_50,
    PMSM_Motor_eqFunction_51,
    PMSM_Motor_eqFunction_52,
    PMSM_Motor_eqFunction_53,
    PMSM_Motor_eqFunction_54,
    PMSM_Motor_eqFunction_55,
    PMSM_Motor_eqFunction_56,
    PMSM_Motor_eqFunction_57,
    PMSM_Motor_eqFunction_58,
    PMSM_Motor_eqFunction_59,
    PMSM_Motor_eqFunction_60,
    PMSM_Motor_eqFunction_61,
    PMSM_Motor_eqFunction_62,
    PMSM_Motor_eqFunction_63,
    PMSM_Motor_eqFunction_64,
    PMSM_Motor_eqFunction_65,
    PMSM_Motor_eqFunction_66,
    PMSM_Motor_eqFunction_67,
    PMSM_Motor_eqFunction_68
  };
  
  for (int id = 0; id < 34; id++) {
    eqFunctions[id](data, threadData);
  }
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  return 0;
}


int PMSM_Motor_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

/* forwarded equations */
extern void PMSM_Motor_eqFunction_37(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_38(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_40(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_41(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_42(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_43(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_44(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_45(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_46(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_47(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_48(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_49(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_50(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_52(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_53(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_54(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_56(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_eqFunction_57(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[19])(DATA*, threadData_t*) = {
    PMSM_Motor_eqFunction_37,
    PMSM_Motor_eqFunction_38,
    PMSM_Motor_eqFunction_40,
    PMSM_Motor_eqFunction_41,
    PMSM_Motor_eqFunction_42,
    PMSM_Motor_eqFunction_43,
    PMSM_Motor_eqFunction_44,
    PMSM_Motor_eqFunction_45,
    PMSM_Motor_eqFunction_46,
    PMSM_Motor_eqFunction_47,
    PMSM_Motor_eqFunction_48,
    PMSM_Motor_eqFunction_49,
    PMSM_Motor_eqFunction_50,
    PMSM_Motor_eqFunction_52,
    PMSM_Motor_eqFunction_53,
    PMSM_Motor_eqFunction_54,
    PMSM_Motor_eqFunction_55,
    PMSM_Motor_eqFunction_56,
    PMSM_Motor_eqFunction_57
  };
  
  for (int id = 0; id < 19; id++) {
    eqFunctions[id](data, threadData);
  }
}

int PMSM_Motor_functionODE(DATA *data, threadData_t *threadData)
{
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  PMSM_Motor_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  return 0;
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);
extern int _main_OptimizationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);

#include "PMSM_Motor_12jac.h"
#include "PMSM_Motor_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks PMSM_Motor_callback = {
  NULL,    /* performSimulation */
  NULL,    /* performQSSSimulation */
  NULL,    /* updateContinuousSystem */
  PMSM_Motor_callExternalObjectDestructors,    /* callExternalObjectDestructors */
  NULL,    /* initialNonLinearSystem */
  NULL,    /* initialLinearSystem */
  NULL,    /* initialMixedSystem */
  #if !defined(OMC_NO_STATESELECTION)
  PMSM_Motor_initializeStateSets,
  #else
  NULL,
  #endif    /* initializeStateSets */
  PMSM_Motor_initializeDAEmodeData,
  PMSM_Motor_functionODE,
  PMSM_Motor_functionAlgebraics,
  PMSM_Motor_functionDAE,
  PMSM_Motor_functionLocalKnownVars,
  PMSM_Motor_input_function,
  PMSM_Motor_input_function_init,
  PMSM_Motor_input_function_updateStartValues,
  PMSM_Motor_data_function,
  PMSM_Motor_output_function,
  PMSM_Motor_setc_function,
  PMSM_Motor_setb_function,
  PMSM_Motor_function_storeDelayed,
  PMSM_Motor_function_storeSpatialDistribution,
  PMSM_Motor_function_initSpatialDistribution,
  PMSM_Motor_updateBoundVariableAttributes,
  PMSM_Motor_functionInitialEquations,
  GLOBAL_EQUIDISTANT_HOMOTOPY,
  NULL,
  PMSM_Motor_functionRemovedInitialEquations,
  PMSM_Motor_updateBoundParameters,
  PMSM_Motor_checkForAsserts,
  PMSM_Motor_function_ZeroCrossingsEquations,
  PMSM_Motor_function_ZeroCrossings,
  PMSM_Motor_function_updateRelations,
  PMSM_Motor_zeroCrossingDescription,
  PMSM_Motor_relationDescription,
  PMSM_Motor_function_initSample,
  PMSM_Motor_INDEX_JAC_A,
  PMSM_Motor_INDEX_JAC_B,
  PMSM_Motor_INDEX_JAC_C,
  PMSM_Motor_INDEX_JAC_D,
  PMSM_Motor_INDEX_JAC_F,
  PMSM_Motor_INDEX_JAC_H,
  PMSM_Motor_initialAnalyticJacobianA,
  PMSM_Motor_initialAnalyticJacobianB,
  PMSM_Motor_initialAnalyticJacobianC,
  PMSM_Motor_initialAnalyticJacobianD,
  PMSM_Motor_initialAnalyticJacobianF,
  PMSM_Motor_initialAnalyticJacobianH,
  PMSM_Motor_functionJacA_column,
  PMSM_Motor_functionJacB_column,
  PMSM_Motor_functionJacC_column,
  PMSM_Motor_functionJacD_column,
  PMSM_Motor_functionJacF_column,
  PMSM_Motor_functionJacH_column,
  PMSM_Motor_linear_model_frame,
  PMSM_Motor_linear_model_datarecovery_frame,
  PMSM_Motor_mayer,
  PMSM_Motor_lagrange,
  PMSM_Motor_getInputVarIndicesInOptimization,
  PMSM_Motor_pickUpBoundsForInputsInOptimization,
  PMSM_Motor_setInputData,
  PMSM_Motor_getTimeGrid,
  PMSM_Motor_symbolicInlineSystem,
  PMSM_Motor_function_initSynchronous,
  PMSM_Motor_function_updateSynchronous,
  PMSM_Motor_function_equationsSynchronous,
  PMSM_Motor_inputNames,
  PMSM_Motor_dataReconciliationInputNames,
  PMSM_Motor_dataReconciliationUnmeasuredVariables,
  PMSM_Motor_read_simulation_info,
  PMSM_Motor_read_input_fmu,
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

#define _OMC_LIT_RESOURCE_3_name_data "PMSM_Motor"
#define _OMC_LIT_RESOURCE_3_dir_data "C:/EmbedSimProject/embed_sim_project/fs_electrical_machines/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,10,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,68,_OMC_LIT_RESOURCE_3_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,8,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir)}};
void PMSM_Motor_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &PMSM_Motor_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "PMSM_Motor";
  data->modelData->modelFilePrefix = "PMSM_Motor";
  data->modelData->modelFileName = "PMSM_Motor.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "C:/EmbedSimProject/embed_sim_project/fs_electrical_machines/modelica";
  data->modelData->modelGUID = "{1d831d67-fdf0-4df4-a4ce-7577bc54d53d}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/PMSM_Motor_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStatesArray = 4;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesRealArray = 43;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersRealArray = 7;
  data->modelData->nParametersIntegerArray = 0;
  data->modelData->nParametersBooleanArray = 0;
  data->modelData->nParametersStringArray = 0;
  data->modelData->nParametersReal = 7;
  data->modelData->nParametersInteger = 0;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nAliasRealArray = 15;
  data->modelData->nAliasIntegerArray = 0;
  data->modelData->nAliasBooleanArray = 0;
  data->modelData->nAliasStringArray = 0;
  data->modelData->nInputVars = 5;
  data->modelData->nOutputVars = 13;
  data->modelData->nZeroCrossings = 1;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 1;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 69;
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

