/* update bound parameters and variable attributes (start, nominal, min, max) */
#include "DCMotor_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 46
type: SIMPLE_ASSIGN
damper.phi_rel = if damper.phi_nominal >= 2.220446049250313e-16 then damper.phi_nominal else 1.0
*/
static void DCMotor_eqFunction_46(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,46};
  data->modelData->realVarsData[0].attribute /* damper.phi_rel */.nominal = (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* damper.phi_nominal PARAM */) >= 2.220446049250313e-16)?(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* damper.phi_nominal PARAM */):1.0);
  infoStreamPrint(OMC_LOG_INIT_V, 0, "%s(nominal=%g)", data->modelData->realVarsData[0].info /* damper.phi_rel */.name,
        (modelica_real) data->modelData->realVarsData[0].attribute /* damper.phi_rel */.nominal);
  threadData->lastEquationSolved = 46;
}

OMC_DISABLE_OPT
int DCMotor_updateBoundVariableAttributes(DATA *data, threadData_t *threadData)
{
  /* min ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating min-values");
  messageClose(OMC_LOG_INIT);
  
  /* max ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating max-values");
  messageClose(OMC_LOG_INIT);
  
  /* nominal **************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating nominal-values");
  DCMotor_eqFunction_46(data, threadData);
  messageClose(OMC_LOG_INIT);
  
  /* start ****************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating primary start-values");
  messageClose(OMC_LOG_INIT);
  
  return 0;
}

void DCMotor_updateBoundParameters_0(DATA *data, threadData_t *threadData);

/*
equation index: 47
type: SIMPLE_ASSIGN
resistor.T = resistor.T_ref
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_47(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,47};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[10]] /* resistor.T PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[12]] /* resistor.T_ref PARAM */);
  threadData->lastEquationSolved = 47;
}

/*
equation index: 48
type: SIMPLE_ASSIGN
resistor.T_heatPort = resistor.T
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_48(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,48};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[11]] /* resistor.T_heatPort PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[10]] /* resistor.T PARAM */);
  threadData->lastEquationSolved = 48;
}

/*
equation index: 50
type: SIMPLE_ASSIGN
emf.internalSupport.phi = emf.fixed.phi0
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_50(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,50};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[5]] /* emf.internalSupport.phi PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* emf.fixed.phi0 PARAM */);
  threadData->lastEquationSolved = 50;
}

/*
equation index: 51
type: SIMPLE_ASSIGN
emf.internalSupport.flange.phi = emf.fixed.phi0
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_51(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,51};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[4]] /* emf.internalSupport.flange.phi PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* emf.fixed.phi0 PARAM */);
  threadData->lastEquationSolved = 51;
}

/*
equation index: 52
type: SIMPLE_ASSIGN
emf.fixed.flange.phi = emf.fixed.phi0
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_52(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,52};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* emf.fixed.flange.phi PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* emf.fixed.phi0 PARAM */);
  threadData->lastEquationSolved = 52;
}
extern void DCMotor_eqFunction_22(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_21(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_20(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_14(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_3(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_2(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_1(DATA *data, threadData_t *threadData);


/*
equation index: 66
type: ALGORITHM

  assert(resistor.T_ref >= 0.0, "Variable violating min constraint: 0.0 <= resistor.T_ref, has value: " + String(resistor.T_ref, "g"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_66(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,66};
  modelica_boolean tmp0;
  static const MMC_DEFSTRINGLIT(tmp1,69,"Variable violating min constraint: 0.0 <= resistor.T_ref, has value: ");
  modelica_string tmp2;
  modelica_metatype tmpMeta3;
  static int tmp4 = 0;
  if(!tmp4)
  {
    tmp0 = GreaterEq((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[12]] /* resistor.T_ref PARAM */),0.0);
    if(!tmp0)
    {
      tmp2 = modelica_real_to_modelica_string_format((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[12]] /* resistor.T_ref PARAM */), (modelica_string) mmc_strings_len1[103]);
      tmpMeta3 = stringAppend(MMC_REFSTRINGLIT(tmp1),tmp2);
      {
        const char* assert_cond = "(resistor.T_ref >= 0.0)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Basic/Resistor.mo",5,3,5,64,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta3));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Basic/Resistor.mo",5,3,5,64,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta3));
        }
      }
      tmp4 = 1;
    }
  }
  threadData->lastEquationSolved = 66;
}

/*
equation index: 67
type: ALGORITHM

  assert(resistor.T >= 0.0, "Variable violating min constraint: 0.0 <= resistor.T, has value: " + String(resistor.T, "g"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_67(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,67};
  modelica_boolean tmp5;
  static const MMC_DEFSTRINGLIT(tmp6,65,"Variable violating min constraint: 0.0 <= resistor.T, has value: ");
  modelica_string tmp7;
  modelica_metatype tmpMeta8;
  static int tmp9 = 0;
  if(!tmp9)
  {
    tmp5 = GreaterEq((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[10]] /* resistor.T PARAM */),0.0);
    if(!tmp5)
    {
      tmp7 = modelica_real_to_modelica_string_format((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[10]] /* resistor.T PARAM */), (modelica_string) mmc_strings_len1[103]);
      tmpMeta8 = stringAppend(MMC_REFSTRINGLIT(tmp6),tmp7);
      {
        const char* assert_cond = "(resistor.T >= 0.0)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Interfaces/ConditionalHeatPort.mo",7,3,8,97,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta8));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Interfaces/ConditionalHeatPort.mo",7,3,8,97,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta8));
        }
      }
      tmp9 = 1;
    }
  }
  threadData->lastEquationSolved = 67;
}

/*
equation index: 68
type: ALGORITHM

  assert(resistor.T_heatPort >= 0.0, "Variable violating min constraint: 0.0 <= resistor.T_heatPort, has value: " + String(resistor.T_heatPort, "g"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_68(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,68};
  modelica_boolean tmp10;
  static const MMC_DEFSTRINGLIT(tmp11,74,"Variable violating min constraint: 0.0 <= resistor.T_heatPort, has value: ");
  modelica_string tmp12;
  modelica_metatype tmpMeta13;
  static int tmp14 = 0;
  if(!tmp14)
  {
    tmp10 = GreaterEq((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[11]] /* resistor.T_heatPort PARAM */),0.0);
    if(!tmp10)
    {
      tmp12 = modelica_real_to_modelica_string_format((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[11]] /* resistor.T_heatPort PARAM */), (modelica_string) mmc_strings_len1[103]);
      tmpMeta13 = stringAppend(MMC_REFSTRINGLIT(tmp11),tmp12);
      {
        const char* assert_cond = "(resistor.T_heatPort >= 0.0)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Interfaces/ConditionalHeatPort.mo",14,3,14,54,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta13));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Interfaces/ConditionalHeatPort.mo",14,3,14,54,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta13));
        }
      }
      tmp14 = 1;
    }
  }
  threadData->lastEquationSolved = 68;
}

/*
equation index: 69
type: ALGORITHM

  assert(damper.d >= 0.0, "Variable violating min constraint: 0.0 <= damper.d, has value: " + String(damper.d, "g"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_69(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,69};
  modelica_boolean tmp15;
  static const MMC_DEFSTRINGLIT(tmp16,63,"Variable violating min constraint: 0.0 <= damper.d, has value: ");
  modelica_string tmp17;
  modelica_metatype tmpMeta18;
  static int tmp19 = 0;
  if(!tmp19)
  {
    tmp15 = GreaterEq((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* damper.d PARAM */),0.0);
    if(!tmp15)
    {
      tmp17 = modelica_real_to_modelica_string_format((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* damper.d PARAM */), (modelica_string) mmc_strings_len1[103]);
      tmpMeta18 = stringAppend(MMC_REFSTRINGLIT(tmp16),tmp17);
      {
        const char* assert_cond = "(damper.d >= 0.0)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Components/Damper.mo",5,3,6,23,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta18));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Components/Damper.mo",5,3,6,23,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta18));
        }
      }
      tmp19 = 1;
    }
  }
  threadData->lastEquationSolved = 69;
}

/*
equation index: 70
type: ALGORITHM

  assert(damper.stateSelect >= StateSelect.never and damper.stateSelect <= StateSelect.always, "Variable violating min/max constraint: StateSelect.never <= damper.stateSelect <= StateSelect.always, has value: " + String(damper.stateSelect, "d"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_70(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,70};
  modelica_boolean tmp20;
  modelica_boolean tmp21;
  static const MMC_DEFSTRINGLIT(tmp22,113,"Variable violating min/max constraint: StateSelect.never <= damper.stateSelect <= StateSelect.always, has value: ");
  modelica_string tmp23;
  modelica_metatype tmpMeta24;
  static int tmp25 = 0;
  if(!tmp25)
  {
    tmp20 = GreaterEq((data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* damper.stateSelect PARAM */),1);
    tmp21 = LessEq((data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* damper.stateSelect PARAM */),5);
    if(!(tmp20 && tmp21))
    {
      tmp23 = modelica_integer_to_modelica_string_format((data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* damper.stateSelect PARAM */), (modelica_string) mmc_strings_len1[100]);
      tmpMeta24 = stringAppend(MMC_REFSTRINGLIT(tmp22),tmp23);
      {
        const char* assert_cond = "(damper.stateSelect >= StateSelect.never and damper.stateSelect <= StateSelect.always)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Interfaces/PartialCompliantWithRelativeStates.mo",24,3,26,57,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta24));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Interfaces/PartialCompliantWithRelativeStates.mo",24,3,26,57,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta24));
        }
      }
      tmp25 = 1;
    }
  }
  threadData->lastEquationSolved = 70;
}

/*
equation index: 71
type: ALGORITHM

  assert(damper.phi_nominal >= 0.0, "Variable violating min constraint: 0.0 <= damper.phi_nominal, has value: " + String(damper.phi_nominal, "g"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_71(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,71};
  modelica_boolean tmp26;
  static const MMC_DEFSTRINGLIT(tmp27,73,"Variable violating min constraint: 0.0 <= damper.phi_nominal, has value: ");
  modelica_string tmp28;
  modelica_metatype tmpMeta29;
  static int tmp30 = 0;
  if(!tmp30)
  {
    tmp26 = GreaterEq((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* damper.phi_nominal PARAM */),0.0);
    if(!tmp26)
    {
      tmp28 = modelica_real_to_modelica_string_format((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* damper.phi_nominal PARAM */), (modelica_string) mmc_strings_len1[103]);
      tmpMeta29 = stringAppend(MMC_REFSTRINGLIT(tmp27),tmp28);
      {
        const char* assert_cond = "(damper.phi_nominal >= 0.0)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Interfaces/PartialCompliantWithRelativeStates.mo",20,3,23,40,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta29));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Interfaces/PartialCompliantWithRelativeStates.mo",20,3,23,40,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta29));
        }
      }
      tmp30 = 1;
    }
  }
  threadData->lastEquationSolved = 71;
}

/*
equation index: 72
type: ALGORITHM

  assert(inertia.stateSelect >= StateSelect.never and inertia.stateSelect <= StateSelect.always, "Variable violating min/max constraint: StateSelect.never <= inertia.stateSelect <= StateSelect.always, has value: " + String(inertia.stateSelect, "d"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_72(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,72};
  modelica_boolean tmp31;
  modelica_boolean tmp32;
  static const MMC_DEFSTRINGLIT(tmp33,114,"Variable violating min/max constraint: StateSelect.never <= inertia.stateSelect <= StateSelect.always, has value: ");
  modelica_string tmp34;
  modelica_metatype tmpMeta35;
  static int tmp36 = 0;
  if(!tmp36)
  {
    tmp31 = GreaterEq((data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[1]] /* inertia.stateSelect PARAM */),1);
    tmp32 = LessEq((data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[1]] /* inertia.stateSelect PARAM */),5);
    if(!(tmp31 && tmp32))
    {
      tmp34 = modelica_integer_to_modelica_string_format((data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[1]] /* inertia.stateSelect PARAM */), (modelica_string) mmc_strings_len1[100]);
      tmpMeta35 = stringAppend(MMC_REFSTRINGLIT(tmp33),tmp34);
      {
        const char* assert_cond = "(inertia.stateSelect >= StateSelect.never and inertia.stateSelect <= StateSelect.always)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Components/Inertia.mo",5,3,7,57,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta35));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Components/Inertia.mo",5,3,7,57,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta35));
        }
      }
      tmp36 = 1;
    }
  }
  threadData->lastEquationSolved = 72;
}

/*
equation index: 73
type: ALGORITHM

  assert(inertia.J >= 0.0, "Variable violating min constraint: 0.0 <= inertia.J, has value: " + String(inertia.J, "g"));
*/
OMC_DISABLE_OPT
static void DCMotor_eqFunction_73(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,73};
  modelica_boolean tmp37;
  static const MMC_DEFSTRINGLIT(tmp38,64,"Variable violating min constraint: 0.0 <= inertia.J, has value: ");
  modelica_string tmp39;
  modelica_metatype tmpMeta40;
  static int tmp41 = 0;
  if(!tmp41)
  {
    tmp37 = GreaterEq((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[8]] /* inertia.J PARAM */),0.0);
    if(!tmp37)
    {
      tmp39 = modelica_real_to_modelica_string_format((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[8]] /* inertia.J PARAM */), (modelica_string) mmc_strings_len1[103]);
      tmpMeta40 = stringAppend(MMC_REFSTRINGLIT(tmp38),tmp39);
      {
        const char* assert_cond = "(inertia.J >= 0.0)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Components/Inertia.mo",4,3,4,61,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta40));
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Mechanics/Rotational/Components/Inertia.mo",4,3,4,61,0};
          omc_assert_warning_withEquationIndexes(info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(tmpMeta40));
        }
      }
      tmp41 = 1;
    }
  }
  threadData->lastEquationSolved = 73;
}
OMC_DISABLE_OPT
void DCMotor_updateBoundParameters_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[20])(DATA*, threadData_t*) = {
    DCMotor_eqFunction_47,
    DCMotor_eqFunction_48,
    DCMotor_eqFunction_50,
    DCMotor_eqFunction_51,
    DCMotor_eqFunction_52,
    DCMotor_eqFunction_22,
    DCMotor_eqFunction_21,
    DCMotor_eqFunction_20,
    DCMotor_eqFunction_14,
    DCMotor_eqFunction_3,
    DCMotor_eqFunction_2,
    DCMotor_eqFunction_1,
    DCMotor_eqFunction_66,
    DCMotor_eqFunction_67,
    DCMotor_eqFunction_68,
    DCMotor_eqFunction_69,
    DCMotor_eqFunction_70,
    DCMotor_eqFunction_71,
    DCMotor_eqFunction_72,
    DCMotor_eqFunction_73
  };
  
  for (int id = 0; id < 20; id++) {
    eqFunctions[id](data, threadData);
  }
}
OMC_DISABLE_OPT
int DCMotor_updateBoundParameters(DATA *data, threadData_t *threadData)
{
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* damper.flange_a.tau variable */) = -0.0;
  data->modelData->realVarsData[10].time_unvarying = 1;
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* emf.fixed.phi0 PARAM */) = 0.0;
  data->modelData->realParameterData[3].time_unvarying = 1;
  (data->simulationInfo->booleanParameter[data->simulationInfo->booleanParamsIndex[0]] /* damper.useHeatPort PARAM */) = 0 /* false */;
  data->modelData->booleanParameterData[0].time_unvarying = 1;
  (data->simulationInfo->booleanParameter[data->simulationInfo->booleanParamsIndex[1]] /* emf.useSupport PARAM */) = 0 /* false */;
  data->modelData->booleanParameterData[1].time_unvarying = 1;
  (data->simulationInfo->booleanParameter[data->simulationInfo->booleanParamsIndex[2]] /* resistor.useHeatPort PARAM */) = 0 /* false */;
  data->modelData->booleanParameterData[2].time_unvarying = 1;
  (data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[0]] /* damper.stateSelect PARAM */) = 4;
  data->modelData->integerParameterData[0].time_unvarying = 1;
  (data->simulationInfo->integerParameter[data->simulationInfo->integerParamsIndex[1]] /* inertia.stateSelect PARAM */) = 3;
  data->modelData->integerParameterData[1].time_unvarying = 1;
  DCMotor_updateBoundParameters_0(data, threadData);
  return 0;
}

#if defined(__cplusplus)
}
#endif
