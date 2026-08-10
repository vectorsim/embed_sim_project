/* Initialization */
#include "DCMotor_model.h"
#include "DCMotor_11mix.h"
#include "DCMotor_12jac.h"
#if defined(__cplusplus)
extern "C" {
#endif

void DCMotor_functionInitialEquations_0(DATA *data, threadData_t *threadData);

/*
equation index: 1
type: SIMPLE_ASSIGN
damper.a_rel = 0.0
*/
void DCMotor_eqFunction_1(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* damper.a_rel variable */) = 0.0;
  threadData->lastEquationSolved = 1;
}

/*
equation index: 2
type: SIMPLE_ASSIGN
damper.lossPower = 0.0
*/
void DCMotor_eqFunction_2(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,2};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* damper.lossPower variable */) = 0.0;
  threadData->lastEquationSolved = 2;
}

/*
equation index: 3
type: SIMPLE_ASSIGN
damper.w_rel = 0.0
*/
void DCMotor_eqFunction_3(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,3};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* damper.w_rel DUMMY_STATE */) = 0.0;
  threadData->lastEquationSolved = 3;
}

/*
equation index: 4
type: SIMPLE_ASSIGN
inductor.i = $START.inductor.i
*/
void DCMotor_eqFunction_4(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,4};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* inductor.i STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[1] /* inductor.i STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 4;
}
extern void DCMotor_eqFunction_37(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_38(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_39(DATA *data, threadData_t *threadData);


/*
equation index: 8
type: SIMPLE_ASSIGN
inertia.w = $START.inertia.w
*/
void DCMotor_eqFunction_8(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,8};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* inertia.w STATE(1,inertia.a) */) = ((modelica_real *)((data->modelData->realVarsData[3] /* inertia.w STATE(1,inertia.a) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 8;
}
extern void DCMotor_eqFunction_43(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_40(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_30(DATA *data, threadData_t *threadData);


/*
equation index: 12
type: SIMPLE_ASSIGN
emf.v = emf.k * emf.w
*/
void DCMotor_eqFunction_12(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf.v variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* emf.k PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf.w variable */));
  threadData->lastEquationSolved = 12;
}

/*
equation index: 13
type: SIMPLE_ASSIGN
$DER.emf.phi = emf.w
*/
void DCMotor_eqFunction_13(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,13};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* der(emf.phi) DUMMY_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf.w variable */);
  threadData->lastEquationSolved = 13;
}

/*
equation index: 14
type: SIMPLE_ASSIGN
resistor.R_actual = resistor.R * (1.0 + resistor.alpha * (resistor.T - resistor.T_ref))
*/
void DCMotor_eqFunction_14(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,14};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* resistor.R_actual variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[9]] /* resistor.R PARAM */)) * (1.0 + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[13]] /* resistor.alpha PARAM */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[10]] /* resistor.T PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[12]] /* resistor.T_ref PARAM */)));
  threadData->lastEquationSolved = 14;
}
extern void DCMotor_eqFunction_31(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_33(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_35(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_36(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_32(DATA *data, threadData_t *threadData);


/*
equation index: 20
type: SIMPLE_ASSIGN
ground.p.i = 0.0
*/
void DCMotor_eqFunction_20(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* ground.p.i variable */) = 0.0;
  threadData->lastEquationSolved = 20;
}

/*
equation index: 21
type: SIMPLE_ASSIGN
ground.p.v = 0.0
*/
void DCMotor_eqFunction_21(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,21};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* ground.p.v variable */) = 0.0;
  threadData->lastEquationSolved = 21;
}

/*
equation index: 22
type: SIMPLE_ASSIGN
damper.flange_b.tau = 0.0
*/
void DCMotor_eqFunction_22(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,22};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* damper.flange_b.tau variable */) = 0.0;
  threadData->lastEquationSolved = 22;
}

/*
equation index: 23
type: SIMPLE_ASSIGN
damper.flange_a.tau = 0.0
*/
void DCMotor_eqFunction_23(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,23};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* damper.flange_a.tau variable */) = 0.0;
  threadData->lastEquationSolved = 23;
}

/*
equation index: 24
type: SIMPLE_ASSIGN
damper.phi_rel = $START.damper.phi_rel
*/
void DCMotor_eqFunction_24(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,24};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* damper.phi_rel STATE(1) */) = ((modelica_real *)((data->modelData->realVarsData[0] /* damper.phi_rel STATE(1) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 24;
}

/*
equation index: 25
type: SIMPLE_ASSIGN
inertia.phi = $START.inertia.phi
*/
void DCMotor_eqFunction_25(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,25};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* inertia.phi STATE(1,inertia.w) */) = ((modelica_real *)((data->modelData->realVarsData[2] /* inertia.phi STATE(1,inertia.w) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 25;
}
extern void DCMotor_eqFunction_44(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_41(DATA *data, threadData_t *threadData);

extern void DCMotor_eqFunction_42(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void DCMotor_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[28])(DATA*, threadData_t*) = {
    DCMotor_eqFunction_1,
    DCMotor_eqFunction_2,
    DCMotor_eqFunction_3,
    DCMotor_eqFunction_4,
    DCMotor_eqFunction_37,
    DCMotor_eqFunction_38,
    DCMotor_eqFunction_39,
    DCMotor_eqFunction_8,
    DCMotor_eqFunction_43,
    DCMotor_eqFunction_40,
    DCMotor_eqFunction_30,
    DCMotor_eqFunction_12,
    DCMotor_eqFunction_13,
    DCMotor_eqFunction_14,
    DCMotor_eqFunction_31,
    DCMotor_eqFunction_33,
    DCMotor_eqFunction_35,
    DCMotor_eqFunction_36,
    DCMotor_eqFunction_32,
    DCMotor_eqFunction_20,
    DCMotor_eqFunction_21,
    DCMotor_eqFunction_22,
    DCMotor_eqFunction_23,
    DCMotor_eqFunction_24,
    DCMotor_eqFunction_25,
    DCMotor_eqFunction_44,
    DCMotor_eqFunction_41,
    DCMotor_eqFunction_42
  };
  
  for (int id = 0; id < 28; id++) {
    eqFunctions[id](data, threadData);
  }
}

int DCMotor_functionInitialEquations(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->discreteCall = 1;
  DCMotor_functionInitialEquations_0(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
  return 0;
}

/* No DCMotor_functionInitialEquations_lambda0 function */

int DCMotor_functionRemovedInitialEquations(DATA *data, threadData_t *threadData)
{
  const int *equationIndexes = NULL;
  double res = 0.0;

  
  return 0;
}


#if defined(__cplusplus)
}
#endif
