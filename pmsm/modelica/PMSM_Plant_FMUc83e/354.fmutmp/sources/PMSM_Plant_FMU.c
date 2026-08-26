/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "PMSM_Plant_FMU_model.h"
#include "simulation/solver/events.h"
#include "simulation/arrayIndex.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int PMSM_Plant_FMU_input_function(DATA *data, threadData_t *threadData)
{
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* T_load variable */) = data->simulationInfo->inputVars[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* duty_a variable */) = data->simulationInfo->inputVars[1];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* duty_b variable */) = data->simulationInfo->inputVars[2];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* duty_c variable */) = data->simulationInfo->inputVars[3];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_dc variable */) = data->simulationInfo->inputVars[4];
  
  return 0;
}

int PMSM_Plant_FMU_input_function_init(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->inputVars[0] = getStartFromScalarIdx(data->simulationInfo, data->modelData, VAR_TYPE_REAL, VAR_KIND_VARIABLE, 17);
  data->simulationInfo->inputVars[1] = getStartFromScalarIdx(data->simulationInfo, data->modelData, VAR_TYPE_REAL, VAR_KIND_VARIABLE, 20);
  data->simulationInfo->inputVars[2] = getStartFromScalarIdx(data->simulationInfo, data->modelData, VAR_TYPE_REAL, VAR_KIND_VARIABLE, 23);
  data->simulationInfo->inputVars[3] = getStartFromScalarIdx(data->simulationInfo, data->modelData, VAR_TYPE_REAL, VAR_KIND_VARIABLE, 26);
  data->simulationInfo->inputVars[4] = getStartFromScalarIdx(data->simulationInfo, data->modelData, VAR_TYPE_REAL, VAR_KIND_VARIABLE, 42);
  
  return 0;
}

int PMSM_Plant_FMU_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[17].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[0], 0, &data->modelData->realVarsData[17].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[20].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[1], 0, &data->modelData->realVarsData[20].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[23].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[2], 0, &data->modelData->realVarsData[23].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[26].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[3], 0, &data->modelData->realVarsData[26].attribute.start);
  assertStreamPrint(threadData, data->modelData->realVarsData[42].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[4], 0, &data->modelData->realVarsData[42].attribute.start);
  
  return 0;
}

int PMSM_Plant_FMU_inputNames(DATA *data, char ** names){
  names[0] = (char *) data->modelData->realVarsData[17].info.name;
  names[1] = (char *) data->modelData->realVarsData[20].info.name;
  names[2] = (char *) data->modelData->realVarsData[23].info.name;
  names[3] = (char *) data->modelData->realVarsData[26].info.name;
  names[4] = (char *) data->modelData->realVarsData[42].info.name;
  
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
  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* T_em variable */);
  data->simulationInfo->outputVars[1] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* ia variable */);
  data->simulationInfo->outputVars[2] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* ib variable */);
  data->simulationInfo->outputVars[3] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* ic variable */);
  data->simulationInfo->outputVars[4] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* id_out variable */);
  data->simulationInfo->outputVars[5] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* iq_out variable */);
  data->simulationInfo->outputVars[6] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* rpm variable */);
  data->simulationInfo->outputVars[7] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* theta_m variable */);
  
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
equation index: 42
type: SIMPLE_ASSIGN
rpm = 9.549296585513721 * omega_m
*/
void PMSM_Plant_FMU_eqFunction_42(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* rpm variable */) = (9.549296585513721) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 42;
}

/*
equation index: 43
type: SIMPLE_ASSIGN
theta_m = theta_e / (*Real*)(p)
*/
void PMSM_Plant_FMU_eqFunction_43(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* theta_m variable */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */),((modelica_real)(data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* p PARAM */)),"/*Real*/(p)",equationIndexes);
  threadData->lastEquationSolved = 43;
}

/*
equation index: 44
type: SIMPLE_ASSIGN
T_em = 1.5 * (*Real*)(p) * i_q * (lambda_pm + (L_d - L_q) * i_d)
*/
void PMSM_Plant_FMU_eqFunction_44(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* T_em variable */) = (1.5) * ((((modelica_real)(data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* p PARAM */))) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[8]] /* lambda_pm PARAM */) + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)))));
  threadData->lastEquationSolved = 44;
}

/*
equation index: 45
type: SIMPLE_ASSIGN
$DER.omega_m = (T_em + (-B_fric) * omega_m - T_load) / J
*/
void PMSM_Plant_FMU_eqFunction_45(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(omega_m) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* T_em variable */) + ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* B_fric PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* T_load variable */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* J PARAM */),"J",equationIndexes);
  threadData->lastEquationSolved = 45;
}

/*
equation index: 46
type: SIMPLE_ASSIGN
omega_e = (*Real*)(p) * omega_m
*/
void PMSM_Plant_FMU_eqFunction_46(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,46};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* omega_e variable */) = (((modelica_real)(data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* p PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  threadData->lastEquationSolved = 46;
}

/*
equation index: 47
type: SIMPLE_ASSIGN
$DER.theta_e = omega_e
*/
void PMSM_Plant_FMU_eqFunction_47(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(theta_e) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* omega_e variable */);
  threadData->lastEquationSolved = 47;
}

/*
equation index: 48
type: SIMPLE_ASSIGN
pwm_time = mod(time, T_pwm, 0)
*/
void PMSM_Plant_FMU_eqFunction_48(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,48};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */) = _event_mod_real(data->localData[0]->timeValue, (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */), ((modelica_integer) 0), data, threadData);
  threadData->lastEquationSolved = 48;
}

/*
equation index: 49
type: SIMPLE_ASSIGN
carrier = if pwm_time < 0.5 * T_pwm then 2.0 * pwm_time / T_pwm else 2.0 * (1.0 - pwm_time / T_pwm)
*/
void PMSM_Plant_FMU_eqFunction_49(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,49};
  modelica_boolean tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  tmp1 = 1.0;
  tmp2 = 0.5;
  relationhysteresis(data, &tmp0, (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */), (0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */)), tmp1, tmp2, 0, Less, LessZC);
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */) = (tmp0?(2.0) * (DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes)):(2.0) * (1.0 - (DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* pwm_time variable */),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes))));
  threadData->lastEquationSolved = 49;
}

/*
equation index: 50
type: SIMPLE_ASSIGN
iq_out = i_q
*/
void PMSM_Plant_FMU_eqFunction_50(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,50};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* iq_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */);
  threadData->lastEquationSolved = 50;
}

/*
equation index: 51
type: SIMPLE_ASSIGN
id_out = i_d
*/
void PMSM_Plant_FMU_eqFunction_51(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* id_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */);
  threadData->lastEquationSolved = 51;
}

/*
equation index: 52
type: SIMPLE_ASSIGN
$cse8 = min(1.0, duty_a)
*/
void PMSM_Plant_FMU_eqFunction_52(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* $cse8 variable */) = fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* duty_a variable */));
  threadData->lastEquationSolved = 52;
}

/*
equation index: 53
type: SIMPLE_ASSIGN
duty_a_lim = max(0.0, $cse8)
*/
void PMSM_Plant_FMU_eqFunction_53(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* duty_a_lim variable */) = fmax(0.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* $cse8 variable */));
  threadData->lastEquationSolved = 53;
}

/*
equation index: 54
type: SIMPLE_ASSIGN
$cse5 = min(1.0, duty_a_lim - 2.0 * dead_time / T_pwm)
*/
void PMSM_Plant_FMU_eqFunction_54(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* $cse5 variable */) = fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* duty_a_lim variable */) - (DIVISION_SIM((2.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* dead_time PARAM */)),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes)));
  threadData->lastEquationSolved = 54;
}

/*
equation index: 55
type: SIMPLE_ASSIGN
duty_a_eff = max(0.0, $cse5)
*/
void PMSM_Plant_FMU_eqFunction_55(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,55};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */) = fmax(0.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* $cse5 variable */));
  threadData->lastEquationSolved = 55;
}

/*
equation index: 56
type: SIMPLE_ASSIGN
gate_a_low = not carrier < duty_a_eff
*/
void PMSM_Plant_FMU_eqFunction_56(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,56};
  modelica_boolean tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  tmp4 = 1.0;
  tmp5 = 1.0;
  relationhysteresis(data, &tmp3, (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* duty_a_eff variable */), tmp4, tmp5, 1, Less, LessZC);
  (data->localData[0]->booleanVars[data->simulationInfo->booleanVarsIndex[0]] /* gate_a_low DISCRETE */) = (!tmp3);
  threadData->lastEquationSolved = 56;
}

/*
equation index: 57
type: SIMPLE_ASSIGN
va_pole = if not gate_a_low then v_dc else 0.0
*/
void PMSM_Plant_FMU_eqFunction_57(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,57};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* va_pole variable */) = ((!(data->localData[0]->booleanVars[data->simulationInfo->booleanVarsIndex[0]] /* gate_a_low DISCRETE */))?(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_dc variable */):0.0);
  threadData->lastEquationSolved = 57;
}

/*
equation index: 58
type: SIMPLE_ASSIGN
$cse7 = min(1.0, duty_b)
*/
void PMSM_Plant_FMU_eqFunction_58(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,58};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* $cse7 variable */) = fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* duty_b variable */));
  threadData->lastEquationSolved = 58;
}

/*
equation index: 59
type: SIMPLE_ASSIGN
duty_b_lim = max(0.0, $cse7)
*/
void PMSM_Plant_FMU_eqFunction_59(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,59};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* duty_b_lim variable */) = fmax(0.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* $cse7 variable */));
  threadData->lastEquationSolved = 59;
}

/*
equation index: 60
type: SIMPLE_ASSIGN
$cse4 = min(1.0, duty_b_lim - 2.0 * dead_time / T_pwm)
*/
void PMSM_Plant_FMU_eqFunction_60(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,60};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */) = fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* duty_b_lim variable */) - (DIVISION_SIM((2.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* dead_time PARAM */)),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes)));
  threadData->lastEquationSolved = 60;
}

/*
equation index: 61
type: SIMPLE_ASSIGN
duty_b_eff = max(0.0, $cse4)
*/
void PMSM_Plant_FMU_eqFunction_61(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,61};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */) = fmax(0.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* $cse4 variable */));
  threadData->lastEquationSolved = 61;
}

/*
equation index: 62
type: SIMPLE_ASSIGN
gate_b_low = not carrier < duty_b_eff
*/
void PMSM_Plant_FMU_eqFunction_62(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,62};
  modelica_boolean tmp6;
  modelica_real tmp7;
  modelica_real tmp8;
  tmp7 = 1.0;
  tmp8 = 1.0;
  relationhysteresis(data, &tmp6, (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* duty_b_eff variable */), tmp7, tmp8, 2, Less, LessZC);
  (data->localData[0]->booleanVars[data->simulationInfo->booleanVarsIndex[1]] /* gate_b_low DISCRETE */) = (!tmp6);
  threadData->lastEquationSolved = 62;
}

/*
equation index: 63
type: SIMPLE_ASSIGN
vb_pole = if not gate_b_low then v_dc else 0.0
*/
void PMSM_Plant_FMU_eqFunction_63(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,63};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* vb_pole variable */) = ((!(data->localData[0]->booleanVars[data->simulationInfo->booleanVarsIndex[1]] /* gate_b_low DISCRETE */))?(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_dc variable */):0.0);
  threadData->lastEquationSolved = 63;
}

/*
equation index: 64
type: SIMPLE_ASSIGN
$cse6 = min(1.0, duty_c)
*/
void PMSM_Plant_FMU_eqFunction_64(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,64};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* $cse6 variable */) = fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* duty_c variable */));
  threadData->lastEquationSolved = 64;
}

/*
equation index: 65
type: SIMPLE_ASSIGN
duty_c_lim = max(0.0, $cse6)
*/
void PMSM_Plant_FMU_eqFunction_65(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,65};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* duty_c_lim variable */) = fmax(0.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* $cse6 variable */));
  threadData->lastEquationSolved = 65;
}

/*
equation index: 66
type: SIMPLE_ASSIGN
$cse3 = min(1.0, duty_c_lim - 2.0 * dead_time / T_pwm)
*/
void PMSM_Plant_FMU_eqFunction_66(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,66};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */) = fmin(1.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* duty_c_lim variable */) - (DIVISION_SIM((2.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* dead_time PARAM */)),(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* T_pwm variable */),"T_pwm",equationIndexes)));
  threadData->lastEquationSolved = 66;
}

/*
equation index: 67
type: SIMPLE_ASSIGN
duty_c_eff = max(0.0, $cse3)
*/
void PMSM_Plant_FMU_eqFunction_67(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,67};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */) = fmax(0.0,(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* $cse3 variable */));
  threadData->lastEquationSolved = 67;
}

/*
equation index: 68
type: SIMPLE_ASSIGN
gate_c_low = not carrier < duty_c_eff
*/
void PMSM_Plant_FMU_eqFunction_68(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,68};
  modelica_boolean tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  tmp10 = 1.0;
  tmp11 = 1.0;
  relationhysteresis(data, &tmp9, (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* carrier variable */), (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* duty_c_eff variable */), tmp10, tmp11, 3, Less, LessZC);
  (data->localData[0]->booleanVars[data->simulationInfo->booleanVarsIndex[2]] /* gate_c_low DISCRETE */) = (!tmp9);
  threadData->lastEquationSolved = 68;
}

/*
equation index: 69
type: SIMPLE_ASSIGN
vc_pole = if not gate_c_low then v_dc else 0.0
*/
void PMSM_Plant_FMU_eqFunction_69(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,69};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[50]] /* vc_pole variable */) = ((!(data->localData[0]->booleanVars[data->simulationInfo->booleanVarsIndex[2]] /* gate_c_low DISCRETE */))?(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* v_dc variable */):0.0);
  threadData->lastEquationSolved = 69;
}

/*
equation index: 70
type: SIMPLE_ASSIGN
v_neutral = 0.3333333333333333 * (va_pole + vb_pole + vc_pole)
*/
void PMSM_Plant_FMU_eqFunction_70(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,70};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[43]] /* v_neutral variable */) = (0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* va_pole variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* vb_pole variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[50]] /* vc_pole variable */));
  threadData->lastEquationSolved = 70;
}

/*
equation index: 71
type: SIMPLE_ASSIGN
va = va_pole - v_neutral
*/
void PMSM_Plant_FMU_eqFunction_71(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,71};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* va variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* va_pole variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[43]] /* v_neutral variable */);
  threadData->lastEquationSolved = 71;
}

/*
equation index: 72
type: SIMPLE_ASSIGN
vb = vb_pole - v_neutral
*/
void PMSM_Plant_FMU_eqFunction_72(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,72};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* vb variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* vb_pole variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[43]] /* v_neutral variable */);
  threadData->lastEquationSolved = 72;
}

/*
equation index: 73
type: SIMPLE_ASSIGN
vc = vc_pole - v_neutral
*/
void PMSM_Plant_FMU_eqFunction_73(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,73};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[49]] /* vc variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[50]] /* vc_pole variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[43]] /* v_neutral variable */);
  threadData->lastEquationSolved = 73;
}

/*
equation index: 74
type: SIMPLE_ASSIGN
v_alpha = 0.6666666666666666 * va + (-0.3333333333333333) * (vb + vc)
*/
void PMSM_Plant_FMU_eqFunction_74(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,74};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_alpha variable */) = (0.6666666666666666) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* va variable */)) + (-0.3333333333333333) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* vb variable */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[49]] /* vc variable */));
  threadData->lastEquationSolved = 74;
}

/*
equation index: 75
type: SIMPLE_ASSIGN
v_beta = 0.5773502691896258 * (vb - vc)
*/
void PMSM_Plant_FMU_eqFunction_75(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,75};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_beta variable */) = (0.5773502691896258) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* vb variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[49]] /* vc variable */));
  threadData->lastEquationSolved = 75;
}

/*
equation index: 76
type: SIMPLE_ASSIGN
$cse2 = cos(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_76(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,76};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */) = cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  threadData->lastEquationSolved = 76;
}

/*
equation index: 77
type: SIMPLE_ASSIGN
$cse1 = sin(theta_e)
*/
void PMSM_Plant_FMU_eqFunction_77(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,77};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */) = sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* theta_e STATE(1,omega_e) */));
  threadData->lastEquationSolved = 77;
}

/*
equation index: 78
type: SIMPLE_ASSIGN
v_d = v_alpha * $cse2 + v_beta * $cse1
*/
void PMSM_Plant_FMU_eqFunction_78(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,78};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_d variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */));
  threadData->lastEquationSolved = 78;
}

/*
equation index: 79
type: SIMPLE_ASSIGN
$DER.i_d = (v_d + omega_e * L_q * i_q - R * i_d) / L_d
*/
void PMSM_Plant_FMU_eqFunction_79(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,79};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(i_d) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* v_d variable */) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */))) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* R PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */))),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */),"L_d",equationIndexes);
  threadData->lastEquationSolved = 79;
}

/*
equation index: 80
type: SIMPLE_ASSIGN
v_q = v_beta * $cse2 - v_alpha * $cse1
*/
void PMSM_Plant_FMU_eqFunction_80(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,80};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[44]] /* v_q variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* v_beta variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* v_alpha variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  threadData->lastEquationSolved = 80;
}

/*
equation index: 81
type: SIMPLE_ASSIGN
$DER.i_q = (v_q + (-R) * i_q - omega_e * (L_d * i_d + lambda_pm)) / L_q
*/
void PMSM_Plant_FMU_eqFunction_81(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,81};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(i_q) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[44]] /* v_q variable */) + ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* R PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* L_d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) + (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[8]] /* lambda_pm PARAM */))),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* L_q PARAM */),"L_q",equationIndexes);
  threadData->lastEquationSolved = 81;
}

/*
equation index: 82
type: SIMPLE_ASSIGN
ia = i_d * $cse2 - i_q * $cse1
*/
void PMSM_Plant_FMU_eqFunction_82(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,82};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* ia variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)));
  threadData->lastEquationSolved = 82;
}

/*
equation index: 83
type: SIMPLE_ASSIGN
i_beta = i_d * $cse1 + i_q * $cse2
*/
void PMSM_Plant_FMU_eqFunction_83(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,83};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* i_beta variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* $cse1 variable */)) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* $cse2 variable */));
  threadData->lastEquationSolved = 83;
}

/*
equation index: 84
type: SIMPLE_ASSIGN
ib = 0.8660254037844386 * i_beta + (-0.5) * ia
*/
void PMSM_Plant_FMU_eqFunction_84(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,84};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* ib variable */) = (0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* i_beta variable */)) + (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* ia variable */));
  threadData->lastEquationSolved = 84;
}

/*
equation index: 85
type: SIMPLE_ASSIGN
ic = (-0.5) * ia + (-0.8660254037844386) * i_beta
*/
void PMSM_Plant_FMU_eqFunction_85(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,85};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* ic variable */) = (-0.5) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* ia variable */)) + (-0.8660254037844386) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* i_beta variable */));
  threadData->lastEquationSolved = 85;
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
  static void (*const eqFunctions[44])(DATA*, threadData_t*) = {
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
    PMSM_Plant_FMU_eqFunction_56,
    PMSM_Plant_FMU_eqFunction_57,
    PMSM_Plant_FMU_eqFunction_58,
    PMSM_Plant_FMU_eqFunction_59,
    PMSM_Plant_FMU_eqFunction_60,
    PMSM_Plant_FMU_eqFunction_61,
    PMSM_Plant_FMU_eqFunction_62,
    PMSM_Plant_FMU_eqFunction_63,
    PMSM_Plant_FMU_eqFunction_64,
    PMSM_Plant_FMU_eqFunction_65,
    PMSM_Plant_FMU_eqFunction_66,
    PMSM_Plant_FMU_eqFunction_67,
    PMSM_Plant_FMU_eqFunction_68,
    PMSM_Plant_FMU_eqFunction_69,
    PMSM_Plant_FMU_eqFunction_70,
    PMSM_Plant_FMU_eqFunction_71,
    PMSM_Plant_FMU_eqFunction_72,
    PMSM_Plant_FMU_eqFunction_73,
    PMSM_Plant_FMU_eqFunction_74,
    PMSM_Plant_FMU_eqFunction_75,
    PMSM_Plant_FMU_eqFunction_76,
    PMSM_Plant_FMU_eqFunction_77,
    PMSM_Plant_FMU_eqFunction_78,
    PMSM_Plant_FMU_eqFunction_79,
    PMSM_Plant_FMU_eqFunction_80,
    PMSM_Plant_FMU_eqFunction_81,
    PMSM_Plant_FMU_eqFunction_82,
    PMSM_Plant_FMU_eqFunction_83,
    PMSM_Plant_FMU_eqFunction_84,
    PMSM_Plant_FMU_eqFunction_85
  };
  
  for (int id = 0; id < 44; id++) {
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
extern void PMSM_Plant_FMU_eqFunction_44(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_45(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_46(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_47(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_48(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_49(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_52(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_53(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_54(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_55(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_56(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_57(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_58(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_59(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_60(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_61(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_62(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_63(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_64(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_65(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_66(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_67(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_68(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_69(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_70(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_71(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_72(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_73(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_74(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_75(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_76(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_77(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_78(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_79(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_80(DATA* data, threadData_t *threadData);
extern void PMSM_Plant_FMU_eqFunction_81(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[36])(DATA*, threadData_t*) = {
    PMSM_Plant_FMU_eqFunction_44,
    PMSM_Plant_FMU_eqFunction_45,
    PMSM_Plant_FMU_eqFunction_46,
    PMSM_Plant_FMU_eqFunction_47,
    PMSM_Plant_FMU_eqFunction_48,
    PMSM_Plant_FMU_eqFunction_49,
    PMSM_Plant_FMU_eqFunction_52,
    PMSM_Plant_FMU_eqFunction_53,
    PMSM_Plant_FMU_eqFunction_54,
    PMSM_Plant_FMU_eqFunction_55,
    PMSM_Plant_FMU_eqFunction_56,
    PMSM_Plant_FMU_eqFunction_57,
    PMSM_Plant_FMU_eqFunction_58,
    PMSM_Plant_FMU_eqFunction_59,
    PMSM_Plant_FMU_eqFunction_60,
    PMSM_Plant_FMU_eqFunction_61,
    PMSM_Plant_FMU_eqFunction_62,
    PMSM_Plant_FMU_eqFunction_63,
    PMSM_Plant_FMU_eqFunction_64,
    PMSM_Plant_FMU_eqFunction_65,
    PMSM_Plant_FMU_eqFunction_66,
    PMSM_Plant_FMU_eqFunction_67,
    PMSM_Plant_FMU_eqFunction_68,
    PMSM_Plant_FMU_eqFunction_69,
    PMSM_Plant_FMU_eqFunction_70,
    PMSM_Plant_FMU_eqFunction_71,
    PMSM_Plant_FMU_eqFunction_72,
    PMSM_Plant_FMU_eqFunction_73,
    PMSM_Plant_FMU_eqFunction_74,
    PMSM_Plant_FMU_eqFunction_75,
    PMSM_Plant_FMU_eqFunction_76,
    PMSM_Plant_FMU_eqFunction_77,
    PMSM_Plant_FMU_eqFunction_78,
    PMSM_Plant_FMU_eqFunction_79,
    PMSM_Plant_FMU_eqFunction_80,
    PMSM_Plant_FMU_eqFunction_81
  };
  
  if (data->simulationInfo->evalSelection) {
    for (int i = 0; i < data->simulationInfo->evalSelection->n; i++) {
      int id = data->simulationInfo->evalSelection->idx[i];
      eqFunctions[id](data, threadData);
    }
  } else {
    for (int id = 0; id < 36; id++) {
      eqFunctions[id](data, threadData);
    }
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

void PMSM_Plant_FMU_ODE_DAG(DATA* data, threadData_t* threadData)
{
  const size_t eqMap[] = {44, 45, 46, 47, 48, 49, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81};
  buildEvalDAG_ODE(data->modelData, sizeof(eqMap)/sizeof(size_t), eqMap);
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
  PMSM_Plant_FMU_ODE_DAG,
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
  PMSM_Plant_FMU_INDEX_JAC_ADJ,
  PMSM_Plant_FMU_INDEX_JAC_B,
  PMSM_Plant_FMU_INDEX_JAC_C,
  PMSM_Plant_FMU_INDEX_JAC_D,
  PMSM_Plant_FMU_INDEX_JAC_F,
  PMSM_Plant_FMU_INDEX_JAC_H,
  PMSM_Plant_FMU_initialAnalyticJacobianA,
  PMSM_Plant_FMU_initialAnalyticJacobianADJ,
  PMSM_Plant_FMU_initialAnalyticJacobianB,
  PMSM_Plant_FMU_initialAnalyticJacobianC,
  PMSM_Plant_FMU_initialAnalyticJacobianD,
  PMSM_Plant_FMU_initialAnalyticJacobianF,
  PMSM_Plant_FMU_initialAnalyticJacobianH,
  PMSM_Plant_FMU_functionJacA_column,
  PMSM_Plant_FMU_functionJacADJ_column,
  PMSM_Plant_FMU_functionJacB_column,
  PMSM_Plant_FMU_functionJacC_column,
  PMSM_Plant_FMU_functionJacD_column,
  PMSM_Plant_FMU_functionJacF_column,
  PMSM_Plant_FMU_functionJacH_column,
  PMSM_Plant_FMU_JacA_DAG,
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
#define _OMC_LIT_RESOURCE_0_dir_data "/home/epl05/.openmodelica/libraries/Complex 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,7,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,58,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "Modelica"
#define _OMC_LIT_RESOURCE_1_dir_data "/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,8,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,59,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_2_dir_data "/home/epl05/.openmodelica/libraries/ModelicaServices 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,16,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,67,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "PMSM_Plant_FMU"
#define _OMC_LIT_RESOURCE_3_dir_data "/home/epl05/EMProject/pmsm/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,14,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,35,_OMC_LIT_RESOURCE_3_dir_data);

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
  data->modelData->modelDir = "/home/epl05/EMProject/pmsm/modelica";
  data->modelData->modelGUID = "{84a1a5e9-d624-4a31-a8f9-ee62a525be00}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/PMSM_Plant_FMU_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStatesArray = 4;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesRealArray = 51;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 3;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersRealArray = 10;
  data->modelData->nParametersIntegerArray = 1;
  data->modelData->nParametersBooleanArray = 1;
  data->modelData->nParametersStringArray = 0;
  data->modelData->nParametersReal = 10;
  data->modelData->nParametersInteger = 1;
  data->modelData->nParametersBoolean = 1;
  data->modelData->nParametersString = 0;
  data->modelData->nAliasRealArray = 9;
  data->modelData->nAliasIntegerArray = 0;
  data->modelData->nAliasBooleanArray = 3;
  data->modelData->nAliasStringArray = 0;
  data->modelData->nInputVars = 5;
  data->modelData->nOutputVars = 8;
  data->modelData->nZeroCrossings = 8;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 4;
  data->modelData->nMathEvents = 3;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 88;
  data->modelData->nMixedSystems = 0;
  data->modelData->nLinearSystems = 0;
  data->modelData->nNonLinearSystems = 0;
  data->modelData->nStateSets = 0;
  data->modelData->nJacobians = 7;
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

