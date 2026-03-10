/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "PMSM_Motor_WithSensors_model.h"
#include "simulation/solver/events.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int PMSM_Motor_WithSensors_input_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* T_load variable */) = data->simulationInfo->inputVars[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* duty_a variable */) = data->simulationInfo->inputVars[1];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* duty_b variable */) = data->simulationInfo->inputVars[2];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* duty_c variable */) = data->simulationInfo->inputVars[3];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_dc variable */) = data->simulationInfo->inputVars[4];
  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_input_function_init(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->inputVars[0] = data->modelData->realVarsData[13].attribute.start;
  data->simulationInfo->inputVars[1] = data->modelData->realVarsData[14].attribute.start;
  data->simulationInfo->inputVars[2] = data->modelData->realVarsData[15].attribute.start;
  data->simulationInfo->inputVars[3] = data->modelData->realVarsData[16].attribute.start;
  data->simulationInfo->inputVars[4] = data->modelData->realVarsData[37].attribute.start;
  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->modelData->realVarsData[13].attribute.start = data->simulationInfo->inputVars[0];
  data->modelData->realVarsData[14].attribute.start = data->simulationInfo->inputVars[1];
  data->modelData->realVarsData[15].attribute.start = data->simulationInfo->inputVars[2];
  data->modelData->realVarsData[16].attribute.start = data->simulationInfo->inputVars[3];
  data->modelData->realVarsData[37].attribute.start = data->simulationInfo->inputVars[4];
  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_inputNames(DATA *data, char ** names){
  TRACE_PUSH

  names[0] = (char *) data->modelData->realVarsData[13].info.name;
  names[1] = (char *) data->modelData->realVarsData[14].info.name;
  names[2] = (char *) data->modelData->realVarsData[15].info.name;
  names[3] = (char *) data->modelData->realVarsData[16].info.name;
  names[4] = (char *) data->modelData->realVarsData[37].info.name;
  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_data_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_dataReconciliationInputNames(DATA *data, char ** names){
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_output_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* T_em_out variable */);
  data->simulationInfo->outputVars[1] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf_a variable */);
  data->simulationInfo->outputVars[2] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf_b variable */);
  data->simulationInfo->outputVars[3] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* emf_c variable */);
  data->simulationInfo->outputVars[4] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */);
  data->simulationInfo->outputVars[5] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* i_b variable */);
  data->simulationInfo->outputVars[6] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_c variable */);
  data->simulationInfo->outputVars[7] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* omega_m_out variable */);
  data->simulationInfo->outputVars[8] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* speed_rpm variable */);
  data->simulationInfo->outputVars[9] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* theta_m variable */);
  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_setc_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int PMSM_Motor_WithSensors_setb_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/*
equation index: 32
type: SIMPLE_ASSIGN
speed_rpm = 9.549296585513721 * omega_m
*/
void PMSM_Motor_WithSensors_eqFunction_32(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,32};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* speed_rpm variable */) = (9.549296585513721) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  TRACE_POP
}
/*
equation index: 33
type: SIMPLE_ASSIGN
theta_m = theta_e / p
*/
void PMSM_Motor_WithSensors_eqFunction_33(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* theta_m variable */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */),(data->simulationInfo->realParameter[6] /* p PARAM */),"p",equationIndexes);
  TRACE_POP
}
/*
equation index: 34
type: SIMPLE_ASSIGN
T_em_out = 1.5 * p * i_q * (lambda_pm + (L_d - L_q) * i_d)
*/
void PMSM_Motor_WithSensors_eqFunction_34(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,34};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* T_em_out variable */) = (1.5) * (((data->simulationInfo->realParameter[6] /* p PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */) + ((data->simulationInfo->realParameter[2] /* L_d PARAM */) - (data->simulationInfo->realParameter[3] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)))));
  TRACE_POP
}
/*
equation index: 35
type: SIMPLE_ASSIGN
$DER.omega_m = (T_em_out + (-B) * omega_m - T_load) / J
*/
void PMSM_Motor_WithSensors_eqFunction_35(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(omega_m) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* T_em_out variable */) + ((-(data->simulationInfo->realParameter[0] /* B PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* T_load variable */),(data->simulationInfo->realParameter[1] /* J PARAM */),"J",equationIndexes);
  TRACE_POP
}
/*
equation index: 36
type: SIMPLE_ASSIGN
omega_e = p * omega_m
*/
void PMSM_Motor_WithSensors_eqFunction_36(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */) = ((data->simulationInfo->realParameter[6] /* p PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  TRACE_POP
}
/*
equation index: 37
type: SIMPLE_ASSIGN
$DER.theta_e = omega_e
*/
void PMSM_Motor_WithSensors_eqFunction_37(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(theta_e) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */);
  TRACE_POP
}
/*
equation index: 38
type: SIMPLE_ASSIGN
v_c_leg = duty_c * v_dc
*/
void PMSM_Motor_WithSensors_eqFunction_38(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_c_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* duty_c variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_dc variable */));
  TRACE_POP
}
/*
equation index: 39
type: SIMPLE_ASSIGN
v_b_leg = duty_b * v_dc
*/
void PMSM_Motor_WithSensors_eqFunction_39(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,39};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_b_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* duty_b variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_dc variable */));
  TRACE_POP
}
/*
equation index: 40
type: SIMPLE_ASSIGN
v_a_leg = duty_a * v_dc
*/
void PMSM_Motor_WithSensors_eqFunction_40(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_a_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* duty_a variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_dc variable */));
  TRACE_POP
}
/*
equation index: 41
type: SIMPLE_ASSIGN
v_neutral = 0.3333333333333333 * (v_a_leg + v_b_leg + v_c_leg)
*/
void PMSM_Motor_WithSensors_eqFunction_41(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,41};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_neutral variable */) = (0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_a_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_b_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_c_leg variable */));
  TRACE_POP
}
/*
equation index: 42
type: SIMPLE_ASSIGN
v_b = v_b_leg - v_neutral
*/
void PMSM_Motor_WithSensors_eqFunction_42(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_b variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_b_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_neutral variable */);
  TRACE_POP
}
/*
equation index: 43
type: SIMPLE_ASSIGN
v_c = v_c_leg - v_neutral
*/
void PMSM_Motor_WithSensors_eqFunction_43(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_c variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_c_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_neutral variable */);
  TRACE_POP
}
/*
equation index: 44
type: SIMPLE_ASSIGN
v_beta = 0.5773502691896257 * (v_b - v_c)
*/
void PMSM_Motor_WithSensors_eqFunction_44(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_beta variable */) = (0.5773502691896257) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_b variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_c variable */));
  TRACE_POP
}
/*
equation index: 45
type: SIMPLE_ASSIGN
v_a = v_a_leg - v_neutral
*/
void PMSM_Motor_WithSensors_eqFunction_45(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_a variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_a_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* v_neutral variable */);
  TRACE_POP
}
/*
equation index: 46
type: SIMPLE_ASSIGN
v_alpha = 0.6666666666666666 * (v_a + (-0.5) * (v_b + v_c))
*/
void PMSM_Motor_WithSensors_eqFunction_46(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,46};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_alpha variable */) = (0.6666666666666666) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_a variable */) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_b variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_c variable */)));
  TRACE_POP
}
/*
equation index: 47
type: SIMPLE_ASSIGN
omega_m_out = omega_m
*/
void PMSM_Motor_WithSensors_eqFunction_47(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* omega_m_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */);
  TRACE_POP
}
/*
equation index: 48
type: SIMPLE_ASSIGN
$cse4 = cos(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_48(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,48};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */) = cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  TRACE_POP
}
/*
equation index: 49
type: SIMPLE_ASSIGN
$cse3 = sin(theta_e)
*/
void PMSM_Motor_WithSensors_eqFunction_49(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  TRACE_POP
}
/*
equation index: 50
type: SIMPLE_ASSIGN
v_d = v_alpha * $cse4 + v_beta * $cse3
*/
void PMSM_Motor_WithSensors_eqFunction_50(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,50};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */));
  TRACE_POP
}
/*
equation index: 51
type: SIMPLE_ASSIGN
$DER.i_d = (v_d + omega_e * L_q * i_q - R * i_d) / L_d
*/
void PMSM_Motor_WithSensors_eqFunction_51(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(i_d) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_d variable */) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[3] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */))) - (((data->simulationInfo->realParameter[4] /* R PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */))),(data->simulationInfo->realParameter[2] /* L_d PARAM */),"L_d",equationIndexes);
  TRACE_POP
}
/*
equation index: 52
type: SIMPLE_ASSIGN
v_q = v_beta * $cse4 - v_alpha * $cse3
*/
void PMSM_Motor_WithSensors_eqFunction_52(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 53
type: SIMPLE_ASSIGN
$DER.i_q = (v_q + (-R) * i_q - omega_e * (L_d * i_d + lambda_pm)) / L_q
*/
void PMSM_Motor_WithSensors_eqFunction_53(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(i_q) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_q variable */) + ((-(data->simulationInfo->realParameter[4] /* R PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[2] /* L_d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) + (data->simulationInfo->realParameter[5] /* lambda_pm PARAM */))),(data->simulationInfo->realParameter[3] /* L_q PARAM */),"L_q",equationIndexes);
  TRACE_POP
}
/*
equation index: 54
type: SIMPLE_ASSIGN
i_a = i_d * $cse4 - i_q * $cse3
*/
void PMSM_Motor_WithSensors_eqFunction_54(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 55
type: SIMPLE_ASSIGN
i_beta = i_d * $cse3 + i_q * $cse4
*/
void PMSM_Motor_WithSensors_eqFunction_55(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,55};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */));
  TRACE_POP
}
/*
equation index: 56
type: SIMPLE_ASSIGN
i_b = 0.8660254037844386 * i_beta + (-0.5) * i_a
*/
void PMSM_Motor_WithSensors_eqFunction_56(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,56};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* i_b variable */) = (0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */)) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */));
  TRACE_POP
}
/*
equation index: 57
type: SIMPLE_ASSIGN
i_c = (-0.5) * i_a + (-0.8660254037844386) * i_beta
*/
void PMSM_Motor_WithSensors_eqFunction_57(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,57};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_c variable */) = (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */)) + (-0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */));
  TRACE_POP
}
/*
equation index: 58
type: SIMPLE_ASSIGN
emf_a = lambda_pm * omega_e * $cse3
*/
void PMSM_Motor_WithSensors_eqFunction_58(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,58};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf_a variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 59
type: SIMPLE_ASSIGN
$cse2 = sin(theta_e - 2.0943951023931953)
*/
void PMSM_Motor_WithSensors_eqFunction_59(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,59};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) - 2.0943951023931953);
  TRACE_POP
}
/*
equation index: 60
type: SIMPLE_ASSIGN
emf_b = lambda_pm * omega_e * $cse2
*/
void PMSM_Motor_WithSensors_eqFunction_60(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,60};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf_b variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)));
  TRACE_POP
}
/*
equation index: 61
type: SIMPLE_ASSIGN
$cse1 = sin(theta_e + 2.0943951023931953)
*/
void PMSM_Motor_WithSensors_eqFunction_61(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,61};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) + 2.0943951023931953);
  TRACE_POP
}
/*
equation index: 62
type: SIMPLE_ASSIGN
emf_c = lambda_pm * omega_e * $cse1
*/
void PMSM_Motor_WithSensors_eqFunction_62(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,62};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* emf_c variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  TRACE_POP
}

OMC_DISABLE_OPT
int PMSM_Motor_WithSensors_functionDAE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  PMSM_Motor_WithSensors_functionLocalKnownVars(data, threadData);
  PMSM_Motor_WithSensors_eqFunction_32(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_33(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_34(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_35(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_36(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_37(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_38(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_39(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_40(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_41(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_42(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_43(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_44(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_45(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_46(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_47(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_48(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_49(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_50(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_51(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_52(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_53(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_54(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_55(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_56(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_57(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_58(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_59(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_60(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_61(data, threadData);

  PMSM_Motor_WithSensors_eqFunction_62(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  TRACE_POP
  return 0;
}


int PMSM_Motor_WithSensors_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

/* forwarded equations */
extern void PMSM_Motor_WithSensors_eqFunction_34(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_35(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_36(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_37(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_38(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_39(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_40(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_41(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_42(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_43(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_44(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_45(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_46(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_48(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_49(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_50(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_51(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_52(DATA* data, threadData_t *threadData);
extern void PMSM_Motor_WithSensors_eqFunction_53(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  int id;

  static void (*const eqFunctions[19])(DATA*, threadData_t*) = {
    PMSM_Motor_WithSensors_eqFunction_34,
    PMSM_Motor_WithSensors_eqFunction_35,
    PMSM_Motor_WithSensors_eqFunction_36,
    PMSM_Motor_WithSensors_eqFunction_37,
    PMSM_Motor_WithSensors_eqFunction_38,
    PMSM_Motor_WithSensors_eqFunction_39,
    PMSM_Motor_WithSensors_eqFunction_40,
    PMSM_Motor_WithSensors_eqFunction_41,
    PMSM_Motor_WithSensors_eqFunction_42,
    PMSM_Motor_WithSensors_eqFunction_43,
    PMSM_Motor_WithSensors_eqFunction_44,
    PMSM_Motor_WithSensors_eqFunction_45,
    PMSM_Motor_WithSensors_eqFunction_46,
    PMSM_Motor_WithSensors_eqFunction_48,
    PMSM_Motor_WithSensors_eqFunction_49,
    PMSM_Motor_WithSensors_eqFunction_50,
    PMSM_Motor_WithSensors_eqFunction_51,
    PMSM_Motor_WithSensors_eqFunction_52,
    PMSM_Motor_WithSensors_eqFunction_53
  };
  
  static const int eqIndices[19] = {
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    48,
    49,
    50,
    51,
    52,
    53
  };
  
  for (id = 0; id < 19; id++) {
    eqFunctions[id](data, threadData);
    threadData->lastEquationSolved = eqIndices[id];
  }
}

int PMSM_Motor_WithSensors_functionODE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  PMSM_Motor_WithSensors_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  TRACE_POP
  return 0;
}

void PMSM_Motor_WithSensors_computeVarIndices(size_t* realIndex, size_t* integerIndex, size_t* booleanIndex, size_t* stringIndex)
{
  TRACE_PUSH

  size_t i_real = 0;
  size_t i_integer = 0;
  size_t i_boolean = 0;
  size_t i_string = 0;

  realIndex[0] = 0;
  integerIndex[0] = 0;
  booleanIndex[0] = 0;
  stringIndex[0] = 0;

  /* stateVars */
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_d STATE(1) */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_q STATE(1) */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* omega_m STATE(1) */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* theta_e STATE(1,omega_e) */
  
  /* derivativeVars */
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(i_d) STATE_DER */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(i_q) STATE_DER */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(omega_m) STATE_DER */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(theta_e) STATE_DER */
  
  /* algVars */
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse1 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse2 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse3 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse4 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* T_em_out variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* T_load variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* duty_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* duty_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* duty_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* emf_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* emf_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* emf_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_beta variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* omega_e variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* omega_m_out variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* speed_rpm variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* theta_m variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_a_leg variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_alpha variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_b_leg variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_beta variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_c_leg variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_d variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_dc variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_neutral variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_q variable */
  
  /* discreteAlgVars */
  
  /* realOptimizeConstraintsVars */
  
  /* realOptimizeFinalConstraintsVars */
  
  
  /* intAlgVars */
  
  /* boolAlgVars */
  
  /* stringAlgVars */
  
  TRACE_POP
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char**argv, DATA *data, threadData_t *threadData);

#include "PMSM_Motor_WithSensors_12jac.h"
#include "PMSM_Motor_WithSensors_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks PMSM_Motor_WithSensors_callback = {
   NULL,    /* performSimulation */
   NULL,    /* performQSSSimulation */
   NULL,    /* updateContinuousSystem */
   PMSM_Motor_WithSensors_callExternalObjectDestructors,    /* callExternalObjectDestructors */
   NULL,    /* initialNonLinearSystem */
   NULL,    /* initialLinearSystem */
   NULL,    /* initialMixedSystem */
   #if !defined(OMC_NO_STATESELECTION)
   PMSM_Motor_WithSensors_initializeStateSets,
   #else
   NULL,
   #endif    /* initializeStateSets */
   PMSM_Motor_WithSensors_initializeDAEmodeData,
   PMSM_Motor_WithSensors_computeVarIndices,
   PMSM_Motor_WithSensors_functionODE,
   PMSM_Motor_WithSensors_functionAlgebraics,
   PMSM_Motor_WithSensors_functionDAE,
   PMSM_Motor_WithSensors_functionLocalKnownVars,
   PMSM_Motor_WithSensors_input_function,
   PMSM_Motor_WithSensors_input_function_init,
   PMSM_Motor_WithSensors_input_function_updateStartValues,
   PMSM_Motor_WithSensors_data_function,
   PMSM_Motor_WithSensors_output_function,
   PMSM_Motor_WithSensors_setc_function,
   PMSM_Motor_WithSensors_setb_function,
   PMSM_Motor_WithSensors_function_storeDelayed,
   PMSM_Motor_WithSensors_function_storeSpatialDistribution,
   PMSM_Motor_WithSensors_function_initSpatialDistribution,
   PMSM_Motor_WithSensors_updateBoundVariableAttributes,
   PMSM_Motor_WithSensors_functionInitialEquations,
   1, /* useHomotopy - 0: local homotopy (equidistant lambda), 1: global homotopy (equidistant lambda), 2: new global homotopy approach (adaptive lambda), 3: new local homotopy approach (adaptive lambda)*/
   NULL,
   PMSM_Motor_WithSensors_functionRemovedInitialEquations,
   PMSM_Motor_WithSensors_updateBoundParameters,
   PMSM_Motor_WithSensors_checkForAsserts,
   PMSM_Motor_WithSensors_function_ZeroCrossingsEquations,
   PMSM_Motor_WithSensors_function_ZeroCrossings,
   PMSM_Motor_WithSensors_function_updateRelations,
   PMSM_Motor_WithSensors_zeroCrossingDescription,
   PMSM_Motor_WithSensors_relationDescription,
   PMSM_Motor_WithSensors_function_initSample,
   PMSM_Motor_WithSensors_INDEX_JAC_A,
   PMSM_Motor_WithSensors_INDEX_JAC_B,
   PMSM_Motor_WithSensors_INDEX_JAC_C,
   PMSM_Motor_WithSensors_INDEX_JAC_D,
   PMSM_Motor_WithSensors_INDEX_JAC_F,
   PMSM_Motor_WithSensors_INDEX_JAC_H,
   PMSM_Motor_WithSensors_initialAnalyticJacobianA,
   PMSM_Motor_WithSensors_initialAnalyticJacobianB,
   PMSM_Motor_WithSensors_initialAnalyticJacobianC,
   PMSM_Motor_WithSensors_initialAnalyticJacobianD,
   PMSM_Motor_WithSensors_initialAnalyticJacobianF,
   PMSM_Motor_WithSensors_initialAnalyticJacobianH,
   PMSM_Motor_WithSensors_functionJacA_column,
   PMSM_Motor_WithSensors_functionJacB_column,
   PMSM_Motor_WithSensors_functionJacC_column,
   PMSM_Motor_WithSensors_functionJacD_column,
   PMSM_Motor_WithSensors_functionJacF_column,
   PMSM_Motor_WithSensors_functionJacH_column,
   PMSM_Motor_WithSensors_linear_model_frame,
   PMSM_Motor_WithSensors_linear_model_datarecovery_frame,
   PMSM_Motor_WithSensors_mayer,
   PMSM_Motor_WithSensors_lagrange,
   PMSM_Motor_WithSensors_pickUpBoundsForInputsInOptimization,
   PMSM_Motor_WithSensors_setInputData,
   PMSM_Motor_WithSensors_getTimeGrid,
   PMSM_Motor_WithSensors_symbolicInlineSystem,
   PMSM_Motor_WithSensors_function_initSynchronous,
   PMSM_Motor_WithSensors_function_updateSynchronous,
   PMSM_Motor_WithSensors_function_equationsSynchronous,
   PMSM_Motor_WithSensors_inputNames,
   PMSM_Motor_WithSensors_dataReconciliationInputNames,
   PMSM_Motor_WithSensors_dataReconciliationUnmeasuredVariables,
   PMSM_Motor_WithSensors_read_simulation_info,
   PMSM_Motor_WithSensors_read_input_fmu,
   NULL,
   NULL,
   -1,
   NULL,
   NULL,
   -1

};

#define _OMC_LIT_RESOURCE_0_name_data "Complex"
#define _OMC_LIT_RESOURCE_0_dir_data "C:/Users/paul/AppData/Roaming/.openmodelica/libraries/Complex 4.0.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,7,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,76,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "Modelica"
#define _OMC_LIT_RESOURCE_1_dir_data "C:/Users/paul/AppData/Roaming/.openmodelica/libraries/Modelica 4.0.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,8,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,77,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_2_dir_data "C:/Users/paul/AppData/Roaming/.openmodelica/libraries/ModelicaServices 4.0.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,16,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,85,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "PMSM_Motor_WithSensors"
#define _OMC_LIT_RESOURCE_3_dir_data "C:/EmbedSimProject/electrical_blocks/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,22,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,45,_OMC_LIT_RESOURCE_3_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,8,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir)}};
void PMSM_Motor_WithSensors_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &PMSM_Motor_WithSensors_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "PMSM_Motor_WithSensors";
  data->modelData->modelFilePrefix = "PMSM_Motor_WithSensors";
  data->modelData->modelFileName = "PMSM_Motor_WithSensors.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "C:/EmbedSimProject/electrical_blocks/modelica";
  data->modelData->modelGUID = "{33f30a05-f53c-4141-abb4-ba15d014c8ce}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/PMSM_Motor_WithSensors_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStates = 4;
  data->modelData->nVariablesRealArray = 40;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersReal = 7;
  data->modelData->nParametersInteger = 0;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nInputVars = 5;
  data->modelData->nOutputVars = 10;
  data->modelData->nAliasReal = 12;
  data->modelData->nAliasInteger = 0;
  data->modelData->nAliasBoolean = 0;
  data->modelData->nAliasString = 0;
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 63;
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

