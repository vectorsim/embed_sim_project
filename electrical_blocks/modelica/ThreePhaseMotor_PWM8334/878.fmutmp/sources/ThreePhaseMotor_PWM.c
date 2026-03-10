/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "ThreePhaseMotor_PWM_model.h"
#include "simulation/solver/events.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int ThreePhaseMotor_PWM_input_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* T_load variable */) = data->simulationInfo->inputVars[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* duty_a variable */) = data->simulationInfo->inputVars[1];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* duty_b variable */) = data->simulationInfo->inputVars[2];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* duty_c variable */) = data->simulationInfo->inputVars[3];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_dc variable */) = data->simulationInfo->inputVars[4];
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_input_function_init(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->inputVars[0] = data->modelData->realVarsData[13].attribute.start;
  data->simulationInfo->inputVars[1] = data->modelData->realVarsData[14].attribute.start;
  data->simulationInfo->inputVars[2] = data->modelData->realVarsData[15].attribute.start;
  data->simulationInfo->inputVars[3] = data->modelData->realVarsData[16].attribute.start;
  data->simulationInfo->inputVars[4] = data->modelData->realVarsData[35].attribute.start;
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->modelData->realVarsData[13].attribute.start = data->simulationInfo->inputVars[0];
  data->modelData->realVarsData[14].attribute.start = data->simulationInfo->inputVars[1];
  data->modelData->realVarsData[15].attribute.start = data->simulationInfo->inputVars[2];
  data->modelData->realVarsData[16].attribute.start = data->simulationInfo->inputVars[3];
  data->modelData->realVarsData[35].attribute.start = data->simulationInfo->inputVars[4];
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_inputNames(DATA *data, char ** names){
  TRACE_PUSH

  names[0] = (char *) data->modelData->realVarsData[13].info.name;
  names[1] = (char *) data->modelData->realVarsData[14].info.name;
  names[2] = (char *) data->modelData->realVarsData[15].info.name;
  names[3] = (char *) data->modelData->realVarsData[16].info.name;
  names[4] = (char *) data->modelData->realVarsData[35].info.name;
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_data_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_dataReconciliationInputNames(DATA *data, char ** names){
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_output_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* T_em_out variable */);
  data->simulationInfo->outputVars[1] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf_a variable */);
  data->simulationInfo->outputVars[2] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf_b variable */);
  data->simulationInfo->outputVars[3] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* emf_c variable */);
  data->simulationInfo->outputVars[4] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */);
  data->simulationInfo->outputVars[5] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* i_b variable */);
  data->simulationInfo->outputVars[6] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_c variable */);
  data->simulationInfo->outputVars[7] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* speed_rpm variable */);
  data->simulationInfo->outputVars[8] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* theta_m variable */);
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_setc_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_PWM_setb_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/*
equation index: 30
type: SIMPLE_ASSIGN
$cse1 = sin(theta_e + 2.0943951023931953)
*/
void ThreePhaseMotor_PWM_eqFunction_30(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,30};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) + 2.0943951023931953);
  TRACE_POP
}
/*
equation index: 31
type: SIMPLE_ASSIGN
$cse2 = sin(theta_e - 2.0943951023931953)
*/
void ThreePhaseMotor_PWM_eqFunction_31(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */) - 2.0943951023931953);
  TRACE_POP
}
/*
equation index: 32
type: SIMPLE_ASSIGN
$cse3 = sin(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_32(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,32};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  TRACE_POP
}
/*
equation index: 33
type: SIMPLE_ASSIGN
$cse4 = cos(theta_e)
*/
void ThreePhaseMotor_PWM_eqFunction_33(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */) = cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  TRACE_POP
}
/*
equation index: 34
type: SIMPLE_ASSIGN
v_a_leg = duty_a * v_dc
*/
void ThreePhaseMotor_PWM_eqFunction_34(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,34};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_a_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* duty_a variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_dc variable */));
  TRACE_POP
}
/*
equation index: 35
type: SIMPLE_ASSIGN
v_b_leg = duty_b * v_dc
*/
void ThreePhaseMotor_PWM_eqFunction_35(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_b_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* duty_b variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_dc variable */));
  TRACE_POP
}
/*
equation index: 36
type: SIMPLE_ASSIGN
v_c_leg = duty_c * v_dc
*/
void ThreePhaseMotor_PWM_eqFunction_36(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_c_leg variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* duty_c variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* v_dc variable */));
  TRACE_POP
}
/*
equation index: 37
type: SIMPLE_ASSIGN
v_neutral = 0.3333333333333333 * (v_a_leg + v_b_leg + v_c_leg)
*/
void ThreePhaseMotor_PWM_eqFunction_37(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_neutral variable */) = (0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_a_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_b_leg variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_c_leg variable */));
  TRACE_POP
}
/*
equation index: 38
type: SIMPLE_ASSIGN
v_alpha = v_a_leg - v_neutral
*/
void ThreePhaseMotor_PWM_eqFunction_38(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_alpha variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* v_a_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_neutral variable */);
  TRACE_POP
}
/*
equation index: 39
type: SIMPLE_ASSIGN
v_b = v_b_leg - v_neutral
*/
void ThreePhaseMotor_PWM_eqFunction_39(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,39};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_b variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* v_b_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_neutral variable */);
  TRACE_POP
}
/*
equation index: 40
type: SIMPLE_ASSIGN
v_c = v_c_leg - v_neutral
*/
void ThreePhaseMotor_PWM_eqFunction_40(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_c variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* v_c_leg variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* v_neutral variable */);
  TRACE_POP
}
/*
equation index: 41
type: SIMPLE_ASSIGN
v_beta = 0.5773502691896258 * (v_b - v_c)
*/
void ThreePhaseMotor_PWM_eqFunction_41(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,41};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_beta variable */) = (0.5773502691896258) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* v_b variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* v_c variable */));
  TRACE_POP
}
/*
equation index: 42
type: SIMPLE_ASSIGN
v_d_cmd = v_alpha * $cse4 + v_beta * $cse3
*/
void ThreePhaseMotor_PWM_eqFunction_42(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_d_cmd variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */));
  TRACE_POP
}
/*
equation index: 43
type: SIMPLE_ASSIGN
v_q_cmd = v_beta * $cse4 - v_alpha * $cse3
*/
void ThreePhaseMotor_PWM_eqFunction_43(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_q_cmd variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 44
type: SIMPLE_ASSIGN
omega_e = p * omega_m
*/
void ThreePhaseMotor_PWM_eqFunction_44(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */) = ((data->simulationInfo->realParameter[6] /* p PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  TRACE_POP
}
/*
equation index: 45
type: SIMPLE_ASSIGN
$DER.i_d = (v_d_cmd + omega_e * L_q * i_q - R * i_d) / L_d
*/
void ThreePhaseMotor_PWM_eqFunction_45(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(i_d) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* v_d_cmd variable */) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[3] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */))) - (((data->simulationInfo->realParameter[4] /* R PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */))),(data->simulationInfo->realParameter[2] /* L_d PARAM */),"L_d",equationIndexes);
  TRACE_POP
}
/*
equation index: 46
type: SIMPLE_ASSIGN
$DER.i_q = (v_q_cmd + (-R) * i_q - omega_e * (L_d * i_d + lambda_pm)) / L_q
*/
void ThreePhaseMotor_PWM_eqFunction_46(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,46};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(i_q) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* v_q_cmd variable */) + ((-(data->simulationInfo->realParameter[4] /* R PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[2] /* L_d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) + (data->simulationInfo->realParameter[5] /* lambda_pm PARAM */))),(data->simulationInfo->realParameter[3] /* L_q PARAM */),"L_q",equationIndexes);
  TRACE_POP
}
/*
equation index: 47
type: SIMPLE_ASSIGN
T_em_out = 1.5 * p * i_q * (lambda_pm + (L_d - L_q) * i_d)
*/
void ThreePhaseMotor_PWM_eqFunction_47(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* T_em_out variable */) = (1.5) * (((data->simulationInfo->realParameter[6] /* p PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */) + ((data->simulationInfo->realParameter[2] /* L_d PARAM */) - (data->simulationInfo->realParameter[3] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)))));
  TRACE_POP
}
/*
equation index: 48
type: SIMPLE_ASSIGN
$DER.omega_m = (T_em_out + (-B) * omega_m - T_load) / J
*/
void ThreePhaseMotor_PWM_eqFunction_48(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,48};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(omega_m) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* T_em_out variable */) + ((-(data->simulationInfo->realParameter[0] /* B PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* T_load variable */),(data->simulationInfo->realParameter[1] /* J PARAM */),"J",equationIndexes);
  TRACE_POP
}
/*
equation index: 49
type: SIMPLE_ASSIGN
$DER.theta_e = omega_e
*/
void ThreePhaseMotor_PWM_eqFunction_49(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(theta_e) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */);
  TRACE_POP
}
/*
equation index: 50
type: SIMPLE_ASSIGN
i_a = i_d * $cse4 - i_q * $cse3
*/
void ThreePhaseMotor_PWM_eqFunction_50(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,50};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 51
type: SIMPLE_ASSIGN
i_beta = i_d * $cse3 + i_q * $cse4
*/
void ThreePhaseMotor_PWM_eqFunction_51(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */));
  TRACE_POP
}
/*
equation index: 52
type: SIMPLE_ASSIGN
i_b = 0.8660254037844386 * i_beta + (-0.5) * i_a
*/
void ThreePhaseMotor_PWM_eqFunction_52(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* i_b variable */) = (0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */)) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */));
  TRACE_POP
}
/*
equation index: 53
type: SIMPLE_ASSIGN
i_c = (-0.5) * i_a + (-0.8660254037844386) * i_beta
*/
void ThreePhaseMotor_PWM_eqFunction_53(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* i_c variable */) = (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* i_a variable */)) + (-0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* i_beta variable */));
  TRACE_POP
}
/*
equation index: 54
type: SIMPLE_ASSIGN
theta_m = theta_e / p
*/
void ThreePhaseMotor_PWM_eqFunction_54(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* theta_m variable */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */),(data->simulationInfo->realParameter[6] /* p PARAM */),"p",equationIndexes);
  TRACE_POP
}
/*
equation index: 55
type: SIMPLE_ASSIGN
emf_a = lambda_pm * omega_e * $cse3
*/
void ThreePhaseMotor_PWM_eqFunction_55(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,55};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf_a variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */)));
  TRACE_POP
}
/*
equation index: 56
type: SIMPLE_ASSIGN
emf_b = lambda_pm * omega_e * $cse2
*/
void ThreePhaseMotor_PWM_eqFunction_56(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,56};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf_b variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)));
  TRACE_POP
}
/*
equation index: 57
type: SIMPLE_ASSIGN
emf_c = lambda_pm * omega_e * $cse1
*/
void ThreePhaseMotor_PWM_eqFunction_57(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,57};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* emf_c variable */) = ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* omega_e variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  TRACE_POP
}
/*
equation index: 58
type: SIMPLE_ASSIGN
speed_rpm = 9.549296585513721 * omega_m
*/
void ThreePhaseMotor_PWM_eqFunction_58(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,58};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* speed_rpm variable */) = (9.549296585513721) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  TRACE_POP
}

OMC_DISABLE_OPT
int ThreePhaseMotor_PWM_functionDAE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  ThreePhaseMotor_PWM_functionLocalKnownVars(data, threadData);
  ThreePhaseMotor_PWM_eqFunction_30(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_31(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_32(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_33(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_34(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_35(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_36(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_37(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_38(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_39(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_40(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_41(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_42(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_43(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_44(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_45(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_46(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_47(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_48(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_49(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_50(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_51(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_52(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_53(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_54(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_55(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_56(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_57(data, threadData);

  ThreePhaseMotor_PWM_eqFunction_58(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  TRACE_POP
  return 0;
}


int ThreePhaseMotor_PWM_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

/* forwarded equations */
extern void ThreePhaseMotor_PWM_eqFunction_32(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_33(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_34(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_35(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_36(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_37(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_38(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_39(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_40(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_41(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_42(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_43(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_44(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_45(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_46(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_47(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_48(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_PWM_eqFunction_49(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  int id;

  static void (*const eqFunctions[18])(DATA*, threadData_t*) = {
    ThreePhaseMotor_PWM_eqFunction_32,
    ThreePhaseMotor_PWM_eqFunction_33,
    ThreePhaseMotor_PWM_eqFunction_34,
    ThreePhaseMotor_PWM_eqFunction_35,
    ThreePhaseMotor_PWM_eqFunction_36,
    ThreePhaseMotor_PWM_eqFunction_37,
    ThreePhaseMotor_PWM_eqFunction_38,
    ThreePhaseMotor_PWM_eqFunction_39,
    ThreePhaseMotor_PWM_eqFunction_40,
    ThreePhaseMotor_PWM_eqFunction_41,
    ThreePhaseMotor_PWM_eqFunction_42,
    ThreePhaseMotor_PWM_eqFunction_43,
    ThreePhaseMotor_PWM_eqFunction_44,
    ThreePhaseMotor_PWM_eqFunction_45,
    ThreePhaseMotor_PWM_eqFunction_46,
    ThreePhaseMotor_PWM_eqFunction_47,
    ThreePhaseMotor_PWM_eqFunction_48,
    ThreePhaseMotor_PWM_eqFunction_49
  };
  
  static const int eqIndices[18] = {
    32,
    33,
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
    47,
    48,
    49
  };
  
  for (id = 0; id < 18; id++) {
    eqFunctions[id](data, threadData);
    threadData->lastEquationSolved = eqIndices[id];
  }
}

int ThreePhaseMotor_PWM_functionODE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  ThreePhaseMotor_PWM_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  TRACE_POP
  return 0;
}

void ThreePhaseMotor_PWM_computeVarIndices(size_t* realIndex, size_t* integerIndex, size_t* booleanIndex, size_t* stringIndex)
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
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse1 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse2 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse3 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* $cse4 variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* T_em_out variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* T_load variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* duty_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* duty_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* duty_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* emf_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* emf_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* emf_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_a variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_beta variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* omega_e variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* speed_rpm variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* theta_m variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_a_leg variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_alpha variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_b variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_b_leg variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_beta variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_c variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_c_leg variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_d_cmd variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_dc variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_neutral variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_q_cmd variable */
  
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

#include "ThreePhaseMotor_PWM_12jac.h"
#include "ThreePhaseMotor_PWM_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks ThreePhaseMotor_PWM_callback = {
   NULL,    /* performSimulation */
   NULL,    /* performQSSSimulation */
   NULL,    /* updateContinuousSystem */
   ThreePhaseMotor_PWM_callExternalObjectDestructors,    /* callExternalObjectDestructors */
   NULL,    /* initialNonLinearSystem */
   NULL,    /* initialLinearSystem */
   NULL,    /* initialMixedSystem */
   #if !defined(OMC_NO_STATESELECTION)
   ThreePhaseMotor_PWM_initializeStateSets,
   #else
   NULL,
   #endif    /* initializeStateSets */
   ThreePhaseMotor_PWM_initializeDAEmodeData,
   ThreePhaseMotor_PWM_computeVarIndices,
   ThreePhaseMotor_PWM_functionODE,
   ThreePhaseMotor_PWM_functionAlgebraics,
   ThreePhaseMotor_PWM_functionDAE,
   ThreePhaseMotor_PWM_functionLocalKnownVars,
   ThreePhaseMotor_PWM_input_function,
   ThreePhaseMotor_PWM_input_function_init,
   ThreePhaseMotor_PWM_input_function_updateStartValues,
   ThreePhaseMotor_PWM_data_function,
   ThreePhaseMotor_PWM_output_function,
   ThreePhaseMotor_PWM_setc_function,
   ThreePhaseMotor_PWM_setb_function,
   ThreePhaseMotor_PWM_function_storeDelayed,
   ThreePhaseMotor_PWM_function_storeSpatialDistribution,
   ThreePhaseMotor_PWM_function_initSpatialDistribution,
   ThreePhaseMotor_PWM_updateBoundVariableAttributes,
   ThreePhaseMotor_PWM_functionInitialEquations,
   1, /* useHomotopy - 0: local homotopy (equidistant lambda), 1: global homotopy (equidistant lambda), 2: new global homotopy approach (adaptive lambda), 3: new local homotopy approach (adaptive lambda)*/
   NULL,
   ThreePhaseMotor_PWM_functionRemovedInitialEquations,
   ThreePhaseMotor_PWM_updateBoundParameters,
   ThreePhaseMotor_PWM_checkForAsserts,
   ThreePhaseMotor_PWM_function_ZeroCrossingsEquations,
   ThreePhaseMotor_PWM_function_ZeroCrossings,
   ThreePhaseMotor_PWM_function_updateRelations,
   ThreePhaseMotor_PWM_zeroCrossingDescription,
   ThreePhaseMotor_PWM_relationDescription,
   ThreePhaseMotor_PWM_function_initSample,
   ThreePhaseMotor_PWM_INDEX_JAC_A,
   ThreePhaseMotor_PWM_INDEX_JAC_B,
   ThreePhaseMotor_PWM_INDEX_JAC_C,
   ThreePhaseMotor_PWM_INDEX_JAC_D,
   ThreePhaseMotor_PWM_INDEX_JAC_F,
   ThreePhaseMotor_PWM_INDEX_JAC_H,
   ThreePhaseMotor_PWM_initialAnalyticJacobianA,
   ThreePhaseMotor_PWM_initialAnalyticJacobianB,
   ThreePhaseMotor_PWM_initialAnalyticJacobianC,
   ThreePhaseMotor_PWM_initialAnalyticJacobianD,
   ThreePhaseMotor_PWM_initialAnalyticJacobianF,
   ThreePhaseMotor_PWM_initialAnalyticJacobianH,
   ThreePhaseMotor_PWM_functionJacA_column,
   ThreePhaseMotor_PWM_functionJacB_column,
   ThreePhaseMotor_PWM_functionJacC_column,
   ThreePhaseMotor_PWM_functionJacD_column,
   ThreePhaseMotor_PWM_functionJacF_column,
   ThreePhaseMotor_PWM_functionJacH_column,
   ThreePhaseMotor_PWM_linear_model_frame,
   ThreePhaseMotor_PWM_linear_model_datarecovery_frame,
   ThreePhaseMotor_PWM_mayer,
   ThreePhaseMotor_PWM_lagrange,
   ThreePhaseMotor_PWM_pickUpBoundsForInputsInOptimization,
   ThreePhaseMotor_PWM_setInputData,
   ThreePhaseMotor_PWM_getTimeGrid,
   ThreePhaseMotor_PWM_symbolicInlineSystem,
   ThreePhaseMotor_PWM_function_initSynchronous,
   ThreePhaseMotor_PWM_function_updateSynchronous,
   ThreePhaseMotor_PWM_function_equationsSynchronous,
   ThreePhaseMotor_PWM_inputNames,
   ThreePhaseMotor_PWM_dataReconciliationInputNames,
   ThreePhaseMotor_PWM_dataReconciliationUnmeasuredVariables,
   ThreePhaseMotor_PWM_read_simulation_info,
   ThreePhaseMotor_PWM_read_input_fmu,
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

#define _OMC_LIT_RESOURCE_3_name_data "ThreePhaseMotor_PWM"
#define _OMC_LIT_RESOURCE_3_dir_data "C:/EmbedSimProject/electrical_blocks/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,19,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,45,_OMC_LIT_RESOURCE_3_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,8,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir)}};
void ThreePhaseMotor_PWM_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &ThreePhaseMotor_PWM_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "ThreePhaseMotor_PWM";
  data->modelData->modelFilePrefix = "ThreePhaseMotor_PWM";
  data->modelData->modelFileName = "ThreePhaseMotor_PWM.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "C:/EmbedSimProject/electrical_blocks/modelica";
  data->modelData->modelGUID = "{bf10467c-8a3e-43eb-a696-71406c380a9b}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/ThreePhaseMotor_PWM_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStates = 4;
  data->modelData->nVariablesRealArray = 38;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersReal = 7;
  data->modelData->nParametersInteger = 0;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nInputVars = 5;
  data->modelData->nOutputVars = 9;
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
  data->modelData->modelDataXml.nEquations = 59;
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

