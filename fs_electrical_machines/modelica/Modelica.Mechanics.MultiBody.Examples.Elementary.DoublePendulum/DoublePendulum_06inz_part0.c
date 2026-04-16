#include "DoublePendulum_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 1
type: SIMPLE_ASSIGN
revolute2.fixed.flange.tau = 0.0
*/
void DoublePendulum_eqFunction_1(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[228]] /* revolute2.fixed.flange.tau variable */) = 0.0;
  threadData->lastEquationSolved = 1;
}

/*
equation index: 2
type: SIMPLE_ASSIGN
world.gravitySphereColor[1] = 0.0
*/
void DoublePendulum_eqFunction_2(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,2};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[24]] /* world.gravitySphereColor[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 2;
}

/*
equation index: 3
type: SIMPLE_ASSIGN
world.gravitySphereColor[2] = 230
*/
void DoublePendulum_eqFunction_3(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,3};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[25]] /* world.gravitySphereColor[2] DISCRETE */) = ((modelica_integer) 230);
  threadData->lastEquationSolved = 3;
}

/*
equation index: 4
type: SIMPLE_ASSIGN
world.gravitySphereColor[3] = 0.0
*/
void DoublePendulum_eqFunction_4(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,4};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[26]] /* world.gravitySphereColor[3] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 4;
}

/*
equation index: 5
type: SIMPLE_ASSIGN
world.groundColor[1] = 200
*/
void DoublePendulum_eqFunction_5(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,5};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[27]] /* world.groundColor[1] DISCRETE */) = ((modelica_integer) 200);
  threadData->lastEquationSolved = 5;
}

/*
equation index: 6
type: SIMPLE_ASSIGN
world.groundColor[2] = 200
*/
void DoublePendulum_eqFunction_6(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,6};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[28]] /* world.groundColor[2] DISCRETE */) = ((modelica_integer) 200);
  threadData->lastEquationSolved = 6;
}

/*
equation index: 7
type: SIMPLE_ASSIGN
world.groundColor[3] = 200
*/
void DoublePendulum_eqFunction_7(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,7};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[29]] /* world.groundColor[3] DISCRETE */) = ((modelica_integer) 200);
  threadData->lastEquationSolved = 7;
}

/*
equation index: 8
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_8(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,8};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[331]] /* world.x_arrowLine.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 8;
}

/*
equation index: 9
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_9(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,9};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[332]] /* world.x_arrowLine.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 9;
}

/*
equation index: 10
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_10(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,10};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[333]] /* world.x_arrowLine.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 10;
}

/*
equation index: 11
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_11(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[334]] /* world.x_arrowLine.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 11;
}

/*
equation index: 12
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_12(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[335]] /* world.x_arrowLine.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 12;
}

/*
equation index: 13
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_13(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,13};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[336]] /* world.x_arrowLine.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 13;
}

/*
equation index: 14
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_14(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,14};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[337]] /* world.x_arrowLine.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 14;
}

/*
equation index: 15
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_15(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[338]] /* world.x_arrowLine.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 15;
}

/*
equation index: 16
type: SIMPLE_ASSIGN
world.x_arrowLine.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_16(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,16};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[339]] /* world.x_arrowLine.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 16;
}

/*
equation index: 17
type: SIMPLE_ASSIGN
world.x_arrowLine.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_17(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,17};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[340]] /* world.x_arrowLine.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 17;
}

/*
equation index: 18
type: SIMPLE_ASSIGN
world.x_arrowLine.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_18(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,18};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[341]] /* world.x_arrowLine.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 18;
}

/*
equation index: 19
type: SIMPLE_ASSIGN
world.x_arrowLine.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_19(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,19};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[342]] /* world.x_arrowLine.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 19;
}

/*
equation index: 20
type: SIMPLE_ASSIGN
world.x_arrowLine.r[1] = 0.0
*/
void DoublePendulum_eqFunction_20(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[347]] /* world.x_arrowLine.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 20;
}

/*
equation index: 21
type: SIMPLE_ASSIGN
world.x_arrowLine.r[2] = 0.0
*/
void DoublePendulum_eqFunction_21(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,21};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[348]] /* world.x_arrowLine.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 21;
}

/*
equation index: 22
type: SIMPLE_ASSIGN
world.x_arrowLine.r[3] = 0.0
*/
void DoublePendulum_eqFunction_22(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,22};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[349]] /* world.x_arrowLine.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 22;
}

/*
equation index: 23
type: SIMPLE_ASSIGN
world.x_arrowLine.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_23(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,23};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[350]] /* world.x_arrowLine.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 23;
}

/*
equation index: 24
type: SIMPLE_ASSIGN
world.x_arrowLine.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_24(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,24};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[351]] /* world.x_arrowLine.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 24;
}

/*
equation index: 25
type: SIMPLE_ASSIGN
world.x_arrowLine.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_25(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,25};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[352]] /* world.x_arrowLine.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 25;
}

/*
equation index: 26
type: SIMPLE_ASSIGN
world.x_arrowLine.lengthDirection[1] = 1.0
*/
void DoublePendulum_eqFunction_26(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,26};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[344]] /* world.x_arrowLine.lengthDirection[1] variable */) = 1.0;
  threadData->lastEquationSolved = 26;
}

/*
equation index: 27
type: SIMPLE_ASSIGN
world.x_arrowLine.lengthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_27(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,27};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[345]] /* world.x_arrowLine.lengthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 27;
}

/*
equation index: 28
type: SIMPLE_ASSIGN
world.x_arrowLine.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_28(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,28};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[346]] /* world.x_arrowLine.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 28;
}

/*
equation index: 29
type: SIMPLE_ASSIGN
world.x_arrowLine.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_29(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,29};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[354]] /* world.x_arrowLine.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 29;
}

/*
equation index: 30
type: SIMPLE_ASSIGN
world.x_arrowLine.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_30(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,30};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[355]] /* world.x_arrowLine.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 30;
}

/*
equation index: 31
type: SIMPLE_ASSIGN
world.x_arrowLine.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_31(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[356]] /* world.x_arrowLine.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 31;
}

/*
equation index: 32
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_32(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,32};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[303]] /* world.x_arrowHead.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 32;
}

/*
equation index: 33
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_33(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[304]] /* world.x_arrowHead.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 33;
}

/*
equation index: 34
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_34(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,34};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[305]] /* world.x_arrowHead.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 34;
}

/*
equation index: 35
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_35(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[306]] /* world.x_arrowHead.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 35;
}

/*
equation index: 36
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_36(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[307]] /* world.x_arrowHead.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 36;
}

/*
equation index: 37
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_37(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[308]] /* world.x_arrowHead.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 37;
}

/*
equation index: 38
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_38(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[309]] /* world.x_arrowHead.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 38;
}

/*
equation index: 39
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_39(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,39};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[310]] /* world.x_arrowHead.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 39;
}

/*
equation index: 40
type: SIMPLE_ASSIGN
world.x_arrowHead.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_40(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[311]] /* world.x_arrowHead.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 40;
}

/*
equation index: 41
type: SIMPLE_ASSIGN
world.x_arrowHead.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_41(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,41};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[312]] /* world.x_arrowHead.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 41;
}

/*
equation index: 42
type: SIMPLE_ASSIGN
world.x_arrowHead.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_42(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[313]] /* world.x_arrowHead.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 42;
}

/*
equation index: 43
type: SIMPLE_ASSIGN
world.x_arrowHead.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_43(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[314]] /* world.x_arrowHead.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 43;
}

/*
equation index: 44
type: SIMPLE_ASSIGN
world.x_arrowHead.r[2] = 0.0
*/
void DoublePendulum_eqFunction_44(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[322]] /* world.x_arrowHead.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 44;
}

/*
equation index: 45
type: SIMPLE_ASSIGN
world.x_arrowHead.r[3] = 0.0
*/
void DoublePendulum_eqFunction_45(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[323]] /* world.x_arrowHead.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 45;
}

/*
equation index: 46
type: SIMPLE_ASSIGN
world.x_arrowHead.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_46(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,46};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[324]] /* world.x_arrowHead.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 46;
}

/*
equation index: 47
type: SIMPLE_ASSIGN
world.x_arrowHead.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_47(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[325]] /* world.x_arrowHead.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 47;
}

/*
equation index: 48
type: SIMPLE_ASSIGN
world.x_arrowHead.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_48(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,48};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[326]] /* world.x_arrowHead.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 48;
}

/*
equation index: 49
type: SIMPLE_ASSIGN
world.x_arrowHead.lengthDirection[1] = 1.0
*/
void DoublePendulum_eqFunction_49(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[319]] /* world.x_arrowHead.lengthDirection[1] variable */) = 1.0;
  threadData->lastEquationSolved = 49;
}

/*
equation index: 50
type: SIMPLE_ASSIGN
world.x_arrowHead.lengthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_50(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,50};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[320]] /* world.x_arrowHead.lengthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 50;
}

/*
equation index: 51
type: SIMPLE_ASSIGN
world.x_arrowHead.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_51(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[321]] /* world.x_arrowHead.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 51;
}

/*
equation index: 52
type: SIMPLE_ASSIGN
world.x_arrowHead.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_52(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[328]] /* world.x_arrowHead.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 52;
}

/*
equation index: 53
type: SIMPLE_ASSIGN
world.x_arrowHead.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_53(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[329]] /* world.x_arrowHead.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 53;
}

/*
equation index: 54
type: SIMPLE_ASSIGN
world.x_arrowHead.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_54(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[330]] /* world.x_arrowHead.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 54;
}

/*
equation index: 55
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_55(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,55};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[394]] /* world.x_label.cylinders[1].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 55;
}

/*
equation index: 56
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_56(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,56};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[395]] /* world.x_label.cylinders[1].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 56;
}

/*
equation index: 57
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_57(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,57};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[396]] /* world.x_label.cylinders[1].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 57;
}

/*
equation index: 58
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_58(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,58};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[400]] /* world.x_label.cylinders[1].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 58;
}

/*
equation index: 59
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_59(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,59};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[401]] /* world.x_label.cylinders[1].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 59;
}

/*
equation index: 60
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_60(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,60};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[402]] /* world.x_label.cylinders[1].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 60;
}

/*
equation index: 61
type: SIMPLE_ASSIGN
world.x_label.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_61(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,61};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[366]] /* world.x_label.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 61;
}

/*
equation index: 62
type: SIMPLE_ASSIGN
world.x_label.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_62(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,62};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[367]] /* world.x_label.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 62;
}

/*
equation index: 63
type: SIMPLE_ASSIGN
world.x_label.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_63(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,63};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[368]] /* world.x_label.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 63;
}

/*
equation index: 64
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_64(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,64};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[397]] /* world.x_label.cylinders[2].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 64;
}

/*
equation index: 65
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_65(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,65};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[398]] /* world.x_label.cylinders[2].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 65;
}

/*
equation index: 66
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_66(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,66};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[399]] /* world.x_label.cylinders[2].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 66;
}

/*
equation index: 67
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_67(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,67};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[403]] /* world.x_label.cylinders[2].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 67;
}

/*
equation index: 68
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_68(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,68};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[404]] /* world.x_label.cylinders[2].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 68;
}

/*
equation index: 69
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_69(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,69};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[405]] /* world.x_label.cylinders[2].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 69;
}

/*
equation index: 70
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_70(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,70};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[452]] /* world.y_arrowLine.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 70;
}

/*
equation index: 71
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_71(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,71};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[453]] /* world.y_arrowLine.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 71;
}

/*
equation index: 72
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_72(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,72};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[454]] /* world.y_arrowLine.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 72;
}

/*
equation index: 73
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_73(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,73};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[455]] /* world.y_arrowLine.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 73;
}

/*
equation index: 74
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_74(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,74};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[456]] /* world.y_arrowLine.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 74;
}

/*
equation index: 75
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_75(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,75};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[457]] /* world.y_arrowLine.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 75;
}

/*
equation index: 76
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_76(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,76};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[458]] /* world.y_arrowLine.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 76;
}

/*
equation index: 77
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_77(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,77};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[459]] /* world.y_arrowLine.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 77;
}

/*
equation index: 78
type: SIMPLE_ASSIGN
world.y_arrowLine.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_78(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,78};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[460]] /* world.y_arrowLine.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 78;
}

/*
equation index: 79
type: SIMPLE_ASSIGN
world.y_arrowLine.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_79(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,79};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[461]] /* world.y_arrowLine.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 79;
}

/*
equation index: 80
type: SIMPLE_ASSIGN
world.y_arrowLine.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_80(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,80};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[462]] /* world.y_arrowLine.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 80;
}

/*
equation index: 81
type: SIMPLE_ASSIGN
world.y_arrowLine.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_81(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,81};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[463]] /* world.y_arrowLine.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 81;
}

/*
equation index: 82
type: SIMPLE_ASSIGN
world.y_arrowLine.r[1] = 0.0
*/
void DoublePendulum_eqFunction_82(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,82};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[468]] /* world.y_arrowLine.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 82;
}

/*
equation index: 83
type: SIMPLE_ASSIGN
world.y_arrowLine.r[2] = 0.0
*/
void DoublePendulum_eqFunction_83(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,83};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[469]] /* world.y_arrowLine.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 83;
}

/*
equation index: 84
type: SIMPLE_ASSIGN
world.y_arrowLine.r[3] = 0.0
*/
void DoublePendulum_eqFunction_84(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,84};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[470]] /* world.y_arrowLine.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 84;
}

/*
equation index: 85
type: SIMPLE_ASSIGN
world.y_arrowLine.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_85(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,85};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[471]] /* world.y_arrowLine.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 85;
}

/*
equation index: 86
type: SIMPLE_ASSIGN
world.y_arrowLine.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_86(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,86};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[472]] /* world.y_arrowLine.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 86;
}

/*
equation index: 87
type: SIMPLE_ASSIGN
world.y_arrowLine.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_87(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,87};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[473]] /* world.y_arrowLine.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 87;
}

/*
equation index: 88
type: SIMPLE_ASSIGN
world.y_arrowLine.lengthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_88(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,88};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[465]] /* world.y_arrowLine.lengthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 88;
}

/*
equation index: 89
type: SIMPLE_ASSIGN
world.y_arrowLine.lengthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_89(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,89};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[466]] /* world.y_arrowLine.lengthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 89;
}

/*
equation index: 90
type: SIMPLE_ASSIGN
world.y_arrowLine.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_90(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,90};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[467]] /* world.y_arrowLine.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 90;
}

/*
equation index: 91
type: SIMPLE_ASSIGN
world.y_arrowLine.widthDirection[1] = 1.0
*/
void DoublePendulum_eqFunction_91(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,91};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[475]] /* world.y_arrowLine.widthDirection[1] variable */) = 1.0;
  threadData->lastEquationSolved = 91;
}

/*
equation index: 92
type: SIMPLE_ASSIGN
world.y_arrowLine.widthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_92(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,92};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[476]] /* world.y_arrowLine.widthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 92;
}

/*
equation index: 93
type: SIMPLE_ASSIGN
world.y_arrowLine.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_93(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,93};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[477]] /* world.y_arrowLine.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 93;
}

/*
equation index: 94
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_94(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,94};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[424]] /* world.y_arrowHead.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 94;
}

/*
equation index: 95
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_95(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,95};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[425]] /* world.y_arrowHead.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 95;
}

/*
equation index: 96
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_96(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,96};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[426]] /* world.y_arrowHead.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 96;
}

/*
equation index: 97
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_97(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,97};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[427]] /* world.y_arrowHead.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 97;
}

/*
equation index: 98
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_98(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,98};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[428]] /* world.y_arrowHead.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 98;
}

/*
equation index: 99
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_99(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,99};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[429]] /* world.y_arrowHead.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 99;
}

/*
equation index: 100
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_100(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,100};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[430]] /* world.y_arrowHead.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 100;
}

/*
equation index: 101
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_101(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[431]] /* world.y_arrowHead.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 101;
}

/*
equation index: 102
type: SIMPLE_ASSIGN
world.y_arrowHead.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_102(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,102};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[432]] /* world.y_arrowHead.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 102;
}

/*
equation index: 103
type: SIMPLE_ASSIGN
world.y_arrowHead.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_103(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,103};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[433]] /* world.y_arrowHead.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 103;
}

/*
equation index: 104
type: SIMPLE_ASSIGN
world.y_arrowHead.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_104(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,104};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[434]] /* world.y_arrowHead.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 104;
}

/*
equation index: 105
type: SIMPLE_ASSIGN
world.y_arrowHead.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_105(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,105};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[435]] /* world.y_arrowHead.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 105;
}

/*
equation index: 106
type: SIMPLE_ASSIGN
world.y_arrowHead.r[1] = 0.0
*/
void DoublePendulum_eqFunction_106(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,106};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[443]] /* world.y_arrowHead.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 106;
}

/*
equation index: 107
type: SIMPLE_ASSIGN
world.y_arrowHead.r[3] = 0.0
*/
void DoublePendulum_eqFunction_107(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,107};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[444]] /* world.y_arrowHead.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 107;
}

/*
equation index: 108
type: SIMPLE_ASSIGN
world.y_arrowHead.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_108(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,108};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[445]] /* world.y_arrowHead.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 108;
}

/*
equation index: 109
type: SIMPLE_ASSIGN
world.y_arrowHead.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_109(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[446]] /* world.y_arrowHead.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 109;
}

/*
equation index: 110
type: SIMPLE_ASSIGN
world.y_arrowHead.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_110(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,110};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[447]] /* world.y_arrowHead.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 110;
}

/*
equation index: 111
type: SIMPLE_ASSIGN
world.y_arrowHead.lengthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_111(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,111};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[440]] /* world.y_arrowHead.lengthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 111;
}

/*
equation index: 112
type: SIMPLE_ASSIGN
world.y_arrowHead.lengthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_112(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,112};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[441]] /* world.y_arrowHead.lengthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 112;
}

/*
equation index: 113
type: SIMPLE_ASSIGN
world.y_arrowHead.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_113(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,113};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[442]] /* world.y_arrowHead.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 113;
}

/*
equation index: 114
type: SIMPLE_ASSIGN
world.y_arrowHead.widthDirection[1] = 1.0
*/
void DoublePendulum_eqFunction_114(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,114};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[449]] /* world.y_arrowHead.widthDirection[1] variable */) = 1.0;
  threadData->lastEquationSolved = 114;
}

/*
equation index: 115
type: SIMPLE_ASSIGN
world.y_arrowHead.widthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_115(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[450]] /* world.y_arrowHead.widthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 115;
}

/*
equation index: 116
type: SIMPLE_ASSIGN
world.y_arrowHead.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_116(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,116};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[451]] /* world.y_arrowHead.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 116;
}

/*
equation index: 117
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_117(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[515]] /* world.y_label.cylinders[1].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 117;
}

/*
equation index: 118
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_118(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,118};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[516]] /* world.y_label.cylinders[1].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 118;
}

/*
equation index: 119
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_119(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,119};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[517]] /* world.y_label.cylinders[1].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 119;
}

/*
equation index: 120
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_120(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,120};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[521]] /* world.y_label.cylinders[1].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 120;
}

/*
equation index: 121
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_121(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,121};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[522]] /* world.y_label.cylinders[1].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 121;
}

/*
equation index: 122
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_122(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,122};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[523]] /* world.y_label.cylinders[1].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 122;
}

/*
equation index: 123
type: SIMPLE_ASSIGN
world.y_label.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_123(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,123};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[487]] /* world.y_label.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 123;
}

/*
equation index: 124
type: SIMPLE_ASSIGN
world.y_label.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_124(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,124};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[488]] /* world.y_label.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 124;
}

/*
equation index: 125
type: SIMPLE_ASSIGN
world.y_label.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_125(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[489]] /* world.y_label.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 125;
}

/*
equation index: 126
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_126(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,126};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[518]] /* world.y_label.cylinders[2].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 126;
}

/*
equation index: 127
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_127(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,127};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[519]] /* world.y_label.cylinders[2].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 127;
}

/*
equation index: 128
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_128(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,128};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[520]] /* world.y_label.cylinders[2].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 128;
}

/*
equation index: 129
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_129(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,129};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[524]] /* world.y_label.cylinders[2].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 129;
}

/*
equation index: 130
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_130(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,130};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[525]] /* world.y_label.cylinders[2].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 130;
}

/*
equation index: 131
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_131(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[526]] /* world.y_label.cylinders[2].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 131;
}

/*
equation index: 132
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_132(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,132};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[576]] /* world.z_arrowLine.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 132;
}

/*
equation index: 133
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_133(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[577]] /* world.z_arrowLine.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 133;
}

/*
equation index: 134
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_134(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,134};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[578]] /* world.z_arrowLine.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 134;
}

/*
equation index: 135
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_135(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,135};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[579]] /* world.z_arrowLine.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 135;
}

/*
equation index: 136
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_136(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,136};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[580]] /* world.z_arrowLine.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 136;
}

/*
equation index: 137
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_137(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,137};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[581]] /* world.z_arrowLine.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 137;
}

/*
equation index: 138
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_138(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,138};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[582]] /* world.z_arrowLine.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 138;
}

/*
equation index: 139
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_139(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,139};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[583]] /* world.z_arrowLine.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 139;
}

/*
equation index: 140
type: SIMPLE_ASSIGN
world.z_arrowLine.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_140(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,140};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[584]] /* world.z_arrowLine.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 140;
}

/*
equation index: 141
type: SIMPLE_ASSIGN
world.z_arrowLine.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_141(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[585]] /* world.z_arrowLine.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 141;
}

/*
equation index: 142
type: SIMPLE_ASSIGN
world.z_arrowLine.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_142(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,142};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[586]] /* world.z_arrowLine.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 142;
}

/*
equation index: 143
type: SIMPLE_ASSIGN
world.z_arrowLine.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_143(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,143};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[587]] /* world.z_arrowLine.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 143;
}

/*
equation index: 144
type: SIMPLE_ASSIGN
world.z_arrowLine.r[1] = 0.0
*/
void DoublePendulum_eqFunction_144(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,144};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[592]] /* world.z_arrowLine.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 144;
}

/*
equation index: 145
type: SIMPLE_ASSIGN
world.z_arrowLine.r[2] = 0.0
*/
void DoublePendulum_eqFunction_145(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,145};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[593]] /* world.z_arrowLine.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 145;
}

/*
equation index: 146
type: SIMPLE_ASSIGN
world.z_arrowLine.r[3] = 0.0
*/
void DoublePendulum_eqFunction_146(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,146};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[594]] /* world.z_arrowLine.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 146;
}

/*
equation index: 147
type: SIMPLE_ASSIGN
world.z_arrowLine.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_147(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[595]] /* world.z_arrowLine.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 147;
}

/*
equation index: 148
type: SIMPLE_ASSIGN
world.z_arrowLine.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_148(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,148};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[596]] /* world.z_arrowLine.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 148;
}

/*
equation index: 149
type: SIMPLE_ASSIGN
world.z_arrowLine.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_149(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[597]] /* world.z_arrowLine.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 149;
}

/*
equation index: 150
type: SIMPLE_ASSIGN
world.z_arrowLine.lengthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_150(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,150};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[589]] /* world.z_arrowLine.lengthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 150;
}

/*
equation index: 151
type: SIMPLE_ASSIGN
world.z_arrowLine.lengthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_151(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,151};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[590]] /* world.z_arrowLine.lengthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 151;
}

/*
equation index: 152
type: SIMPLE_ASSIGN
world.z_arrowLine.lengthDirection[3] = 1.0
*/
void DoublePendulum_eqFunction_152(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,152};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[591]] /* world.z_arrowLine.lengthDirection[3] variable */) = 1.0;
  threadData->lastEquationSolved = 152;
}

/*
equation index: 153
type: SIMPLE_ASSIGN
world.z_arrowLine.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_153(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,153};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[599]] /* world.z_arrowLine.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 153;
}

/*
equation index: 154
type: SIMPLE_ASSIGN
world.z_arrowLine.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_154(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,154};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[600]] /* world.z_arrowLine.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 154;
}

/*
equation index: 155
type: SIMPLE_ASSIGN
world.z_arrowLine.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_155(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,155};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[601]] /* world.z_arrowLine.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 155;
}

/*
equation index: 156
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_156(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,156};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[548]] /* world.z_arrowHead.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 156;
}

/*
equation index: 157
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_157(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[549]] /* world.z_arrowHead.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 157;
}

/*
equation index: 158
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_158(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,158};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[550]] /* world.z_arrowHead.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 158;
}

/*
equation index: 159
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_159(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,159};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[551]] /* world.z_arrowHead.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 159;
}

/*
equation index: 160
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_160(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,160};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[552]] /* world.z_arrowHead.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 160;
}

/*
equation index: 161
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_161(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,161};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[553]] /* world.z_arrowHead.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 161;
}

/*
equation index: 162
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_162(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,162};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[554]] /* world.z_arrowHead.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 162;
}

/*
equation index: 163
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_163(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[555]] /* world.z_arrowHead.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 163;
}

/*
equation index: 164
type: SIMPLE_ASSIGN
world.z_arrowHead.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_164(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,164};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[556]] /* world.z_arrowHead.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 164;
}

/*
equation index: 165
type: SIMPLE_ASSIGN
world.z_arrowHead.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_165(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[557]] /* world.z_arrowHead.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 165;
}

/*
equation index: 166
type: SIMPLE_ASSIGN
world.z_arrowHead.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_166(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,166};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[558]] /* world.z_arrowHead.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 166;
}

/*
equation index: 167
type: SIMPLE_ASSIGN
world.z_arrowHead.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_167(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,167};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[559]] /* world.z_arrowHead.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 167;
}

/*
equation index: 168
type: SIMPLE_ASSIGN
world.z_arrowHead.r[1] = 0.0
*/
void DoublePendulum_eqFunction_168(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,168};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[567]] /* world.z_arrowHead.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 168;
}

/*
equation index: 169
type: SIMPLE_ASSIGN
world.z_arrowHead.r[2] = 0.0
*/
void DoublePendulum_eqFunction_169(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,169};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[568]] /* world.z_arrowHead.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 169;
}

/*
equation index: 170
type: SIMPLE_ASSIGN
world.z_arrowHead.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_170(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,170};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[569]] /* world.z_arrowHead.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 170;
}

/*
equation index: 171
type: SIMPLE_ASSIGN
world.z_arrowHead.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_171(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,171};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[570]] /* world.z_arrowHead.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 171;
}

/*
equation index: 172
type: SIMPLE_ASSIGN
world.z_arrowHead.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_172(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,172};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[571]] /* world.z_arrowHead.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 172;
}

/*
equation index: 173
type: SIMPLE_ASSIGN
world.z_arrowHead.lengthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_173(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[564]] /* world.z_arrowHead.lengthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 173;
}

/*
equation index: 174
type: SIMPLE_ASSIGN
world.z_arrowHead.lengthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_174(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,174};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[565]] /* world.z_arrowHead.lengthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 174;
}

/*
equation index: 175
type: SIMPLE_ASSIGN
world.z_arrowHead.lengthDirection[3] = 1.0
*/
void DoublePendulum_eqFunction_175(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,175};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[566]] /* world.z_arrowHead.lengthDirection[3] variable */) = 1.0;
  threadData->lastEquationSolved = 175;
}

/*
equation index: 176
type: SIMPLE_ASSIGN
world.z_arrowHead.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_176(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,176};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[573]] /* world.z_arrowHead.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 176;
}

/*
equation index: 177
type: SIMPLE_ASSIGN
world.z_arrowHead.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_177(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,177};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[574]] /* world.z_arrowHead.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 177;
}

/*
equation index: 178
type: SIMPLE_ASSIGN
world.z_arrowHead.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_178(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,178};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[575]] /* world.z_arrowHead.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 178;
}

/*
equation index: 179
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_179(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[647]] /* world.z_label.cylinders[1].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 179;
}

/*
equation index: 180
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_180(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,180};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[648]] /* world.z_label.cylinders[1].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 180;
}

/*
equation index: 181
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_181(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[649]] /* world.z_label.cylinders[1].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 181;
}

/*
equation index: 182
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_182(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,182};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[656]] /* world.z_label.cylinders[1].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 182;
}

/*
equation index: 183
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_183(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,183};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[657]] /* world.z_label.cylinders[1].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 183;
}

/*
equation index: 184
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_184(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,184};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[658]] /* world.z_label.cylinders[1].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 184;
}

/*
equation index: 185
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_185(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,185};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[650]] /* world.z_label.cylinders[2].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 185;
}

/*
equation index: 186
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_186(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,186};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[651]] /* world.z_label.cylinders[2].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 186;
}

/*
equation index: 187
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_187(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,187};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[652]] /* world.z_label.cylinders[2].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 187;
}

/*
equation index: 188
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_188(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,188};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[659]] /* world.z_label.cylinders[2].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 188;
}

/*
equation index: 189
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_189(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[660]] /* world.z_label.cylinders[2].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 189;
}

/*
equation index: 190
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_190(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,190};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[661]] /* world.z_label.cylinders[2].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 190;
}

/*
equation index: 191
type: SIMPLE_ASSIGN
world.z_label.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_191(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,191};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[611]] /* world.z_label.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 191;
}

/*
equation index: 192
type: SIMPLE_ASSIGN
world.z_label.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_192(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,192};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[612]] /* world.z_label.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 192;
}

/*
equation index: 193
type: SIMPLE_ASSIGN
world.z_label.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_193(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,193};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[613]] /* world.z_label.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 193;
}

/*
equation index: 194
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_194(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,194};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[653]] /* world.z_label.cylinders[3].r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 194;
}

/*
equation index: 195
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_195(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[654]] /* world.z_label.cylinders[3].r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 195;
}

/*
equation index: 196
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_196(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,196};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[655]] /* world.z_label.cylinders[3].r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 196;
}

/*
equation index: 197
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_197(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[662]] /* world.z_label.cylinders[3].widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 197;
}

/*
equation index: 198
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_198(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,198};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[663]] /* world.z_label.cylinders[3].widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 198;
}

/*
equation index: 199
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_199(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,199};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[664]] /* world.z_label.cylinders[3].widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 199;
}

/*
equation index: 200
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].color[3] = 0.0
*/
void DoublePendulum_eqFunction_200(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,200};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[625]] /* world.z_label.cylinders[3].color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 200;
}

/*
equation index: 201
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].color[2] = 0.0
*/
void DoublePendulum_eqFunction_201(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,201};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[624]] /* world.z_label.cylinders[3].color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 201;
}

/*
equation index: 202
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].color[1] = 0.0
*/
void DoublePendulum_eqFunction_202(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,202};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[623]] /* world.z_label.cylinders[3].color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 202;
}

/*
equation index: 203
type: SIMPLE_ASSIGN
world.z_arrowHead.color[3] = 0.0
*/
void DoublePendulum_eqFunction_203(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,203};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[562]] /* world.z_arrowHead.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 203;
}

/*
equation index: 204
type: SIMPLE_ASSIGN
world.z_arrowHead.color[2] = 0.0
*/
void DoublePendulum_eqFunction_204(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,204};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[561]] /* world.z_arrowHead.color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 204;
}

/*
equation index: 205
type: SIMPLE_ASSIGN
world.z_arrowHead.color[1] = 0.0
*/
void DoublePendulum_eqFunction_205(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[560]] /* world.z_arrowHead.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 205;
}

/*
equation index: 206
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].color[3] = 0.0
*/
void DoublePendulum_eqFunction_206(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,206};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[501]] /* world.y_label.cylinders[2].color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 206;
}

/*
equation index: 207
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].color[2] = 0.0
*/
void DoublePendulum_eqFunction_207(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,207};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[500]] /* world.y_label.cylinders[2].color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 207;
}

/*
equation index: 208
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].color[1] = 0.0
*/
void DoublePendulum_eqFunction_208(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,208};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[499]] /* world.y_label.cylinders[2].color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 208;
}

/*
equation index: 209
type: SIMPLE_ASSIGN
world.y_arrowHead.color[3] = 0.0
*/
void DoublePendulum_eqFunction_209(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,209};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[438]] /* world.y_arrowHead.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 209;
}

/*
equation index: 210
type: SIMPLE_ASSIGN
world.y_arrowHead.color[2] = 0.0
*/
void DoublePendulum_eqFunction_210(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,210};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[437]] /* world.y_arrowHead.color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 210;
}

/*
equation index: 211
type: SIMPLE_ASSIGN
world.y_arrowHead.color[1] = 0.0
*/
void DoublePendulum_eqFunction_211(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[436]] /* world.y_arrowHead.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 211;
}

/*
equation index: 212
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].color[3] = 0.0
*/
void DoublePendulum_eqFunction_212(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,212};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[380]] /* world.x_label.cylinders[2].color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 212;
}

/*
equation index: 213
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].color[2] = 0.0
*/
void DoublePendulum_eqFunction_213(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[379]] /* world.x_label.cylinders[2].color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 213;
}

/*
equation index: 214
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].color[1] = 0.0
*/
void DoublePendulum_eqFunction_214(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,214};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[378]] /* world.x_label.cylinders[2].color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 214;
}

/*
equation index: 215
type: SIMPLE_ASSIGN
world.x_arrowHead.color[3] = 0.0
*/
void DoublePendulum_eqFunction_215(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,215};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[317]] /* world.x_arrowHead.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 215;
}

/*
equation index: 216
type: SIMPLE_ASSIGN
world.x_arrowHead.color[2] = 0.0
*/
void DoublePendulum_eqFunction_216(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,216};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[316]] /* world.x_arrowHead.color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 216;
}

/*
equation index: 217
type: SIMPLE_ASSIGN
world.x_arrowHead.color[1] = 0.0
*/
void DoublePendulum_eqFunction_217(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,217};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[315]] /* world.x_arrowHead.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 217;
}

/*
equation index: 218
type: SIMPLE_ASSIGN
world.axisColor_x[1] = 0.0
*/
void DoublePendulum_eqFunction_218(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,218};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[18]] /* world.axisColor_x[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 218;
}

/*
equation index: 219
type: SIMPLE_ASSIGN
world.axisColor_x[2] = 0.0
*/
void DoublePendulum_eqFunction_219(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,219};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[19]] /* world.axisColor_x[2] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 219;
}

/*
equation index: 220
type: SIMPLE_ASSIGN
world.axisColor_x[3] = 0.0
*/
void DoublePendulum_eqFunction_220(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,220};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[20]] /* world.axisColor_x[3] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 220;
}

/*
equation index: 221
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_221(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[280]] /* world.gravityArrowLine.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 221;
}

/*
equation index: 222
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_222(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,222};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[281]] /* world.gravityArrowLine.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 222;
}

/*
equation index: 223
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_223(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,223};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[282]] /* world.gravityArrowLine.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 223;
}

/*
equation index: 224
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_224(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,224};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[283]] /* world.gravityArrowLine.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 224;
}

/*
equation index: 225
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_225(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,225};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[284]] /* world.gravityArrowLine.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 225;
}

/*
equation index: 226
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_226(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,226};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[285]] /* world.gravityArrowLine.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 226;
}

/*
equation index: 227
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_227(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[286]] /* world.gravityArrowLine.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 227;
}

/*
equation index: 228
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_228(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,228};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[287]] /* world.gravityArrowLine.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 228;
}

/*
equation index: 229
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_229(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[288]] /* world.gravityArrowLine.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 229;
}

/*
equation index: 230
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_230(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,230};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[289]] /* world.gravityArrowLine.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 230;
}

/*
equation index: 231
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_231(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,231};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[290]] /* world.gravityArrowLine.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 231;
}

/*
equation index: 232
type: SIMPLE_ASSIGN
world.gravityArrowLine.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_232(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,232};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[291]] /* world.gravityArrowLine.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 232;
}

/*
equation index: 233
type: SIMPLE_ASSIGN
world.gravityArrowLine.r[1] = 0.0
*/
void DoublePendulum_eqFunction_233(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,233};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[296]] /* world.gravityArrowLine.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 233;
}

/*
equation index: 234
type: SIMPLE_ASSIGN
world.gravityArrowLine.r[2] = 0.0
*/
void DoublePendulum_eqFunction_234(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,234};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[297]] /* world.gravityArrowLine.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 234;
}

/*
equation index: 235
type: SIMPLE_ASSIGN
world.gravityArrowLine.r[3] = 0.0
*/
void DoublePendulum_eqFunction_235(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,235};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[298]] /* world.gravityArrowLine.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 235;
}

/*
equation index: 236
type: SIMPLE_ASSIGN
world.gravityArrowLine.lengthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_236(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,236};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[293]] /* world.gravityArrowLine.lengthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 236;
}

/*
equation index: 237
type: SIMPLE_ASSIGN
world.gravityArrowLine.lengthDirection[2] = -1.0
*/
void DoublePendulum_eqFunction_237(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[294]] /* world.gravityArrowLine.lengthDirection[2] variable */) = -1.0;
  threadData->lastEquationSolved = 237;
}

/*
equation index: 238
type: SIMPLE_ASSIGN
world.gravityArrowLine.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_238(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,238};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[295]] /* world.gravityArrowLine.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 238;
}

/*
equation index: 239
type: SIMPLE_ASSIGN
world.gravityArrowLine.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_239(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,239};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[300]] /* world.gravityArrowLine.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 239;
}

/*
equation index: 240
type: SIMPLE_ASSIGN
world.gravityArrowLine.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_240(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,240};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[301]] /* world.gravityArrowLine.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 240;
}

/*
equation index: 241
type: SIMPLE_ASSIGN
world.gravityArrowLine.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_241(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,241};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[302]] /* world.gravityArrowLine.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 241;
}

/*
equation index: 242
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[1,1] = 1.0
*/
void DoublePendulum_eqFunction_242(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,242};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[253]] /* world.gravityArrowHead.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 242;
}

/*
equation index: 243
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[1,2] = 0.0
*/
void DoublePendulum_eqFunction_243(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[254]] /* world.gravityArrowHead.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 243;
}

/*
equation index: 244
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[1,3] = 0.0
*/
void DoublePendulum_eqFunction_244(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,244};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[255]] /* world.gravityArrowHead.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 244;
}

/*
equation index: 245
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[2,1] = 0.0
*/
void DoublePendulum_eqFunction_245(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[256]] /* world.gravityArrowHead.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 245;
}

/*
equation index: 246
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[2,2] = 1.0
*/
void DoublePendulum_eqFunction_246(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,246};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[257]] /* world.gravityArrowHead.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 246;
}

/*
equation index: 247
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[2,3] = 0.0
*/
void DoublePendulum_eqFunction_247(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,247};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[258]] /* world.gravityArrowHead.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 247;
}

/*
equation index: 248
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[3,1] = 0.0
*/
void DoublePendulum_eqFunction_248(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,248};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[259]] /* world.gravityArrowHead.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 248;
}

/*
equation index: 249
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[3,2] = 0.0
*/
void DoublePendulum_eqFunction_249(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,249};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[260]] /* world.gravityArrowHead.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 249;
}

/*
equation index: 250
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.T[3,3] = 1.0
*/
void DoublePendulum_eqFunction_250(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,250};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[261]] /* world.gravityArrowHead.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 250;
}

/*
equation index: 251
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.w[1] = 0.0
*/
void DoublePendulum_eqFunction_251(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,251};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[262]] /* world.gravityArrowHead.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 251;
}

/*
equation index: 252
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.w[2] = 0.0
*/
void DoublePendulum_eqFunction_252(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,252};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[263]] /* world.gravityArrowHead.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 252;
}

/*
equation index: 253
type: SIMPLE_ASSIGN
world.gravityArrowHead.R.w[3] = 0.0
*/
void DoublePendulum_eqFunction_253(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[264]] /* world.gravityArrowHead.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 253;
}

/*
equation index: 254
type: SIMPLE_ASSIGN
world.gravityArrowHead.r[1] = 0.0
*/
void DoublePendulum_eqFunction_254(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,254};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[272]] /* world.gravityArrowHead.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 254;
}

/*
equation index: 255
type: SIMPLE_ASSIGN
world.gravityArrowHead.r[2] = 0.0
*/
void DoublePendulum_eqFunction_255(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,255};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[273]] /* world.gravityArrowHead.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 255;
}

/*
equation index: 256
type: SIMPLE_ASSIGN
world.gravityArrowHead.r[3] = 0.0
*/
void DoublePendulum_eqFunction_256(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,256};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[274]] /* world.gravityArrowHead.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 256;
}

/*
equation index: 257
type: SIMPLE_ASSIGN
world.gravityArrowHead.r_shape[2] = world.gravityArrowTail[2] - world.gravityLineLength
*/
void DoublePendulum_eqFunction_257(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,257};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[275]] /* world.gravityArrowHead.r_shape[2] variable */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[277]] /* world.gravityArrowTail[2] PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[281]] /* world.gravityLineLength PARAM */);
  threadData->lastEquationSolved = 257;
}

/*
equation index: 258
type: SIMPLE_ASSIGN
world.gravityArrowHead.lengthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_258(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,258};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[269]] /* world.gravityArrowHead.lengthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 258;
}

/*
equation index: 259
type: SIMPLE_ASSIGN
world.gravityArrowHead.lengthDirection[2] = -1.0
*/
void DoublePendulum_eqFunction_259(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[270]] /* world.gravityArrowHead.lengthDirection[2] variable */) = -1.0;
  threadData->lastEquationSolved = 259;
}

/*
equation index: 260
type: SIMPLE_ASSIGN
world.gravityArrowHead.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_260(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,260};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[271]] /* world.gravityArrowHead.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 260;
}

/*
equation index: 261
type: SIMPLE_ASSIGN
world.gravityArrowHead.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_261(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[277]] /* world.gravityArrowHead.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 261;
}

/*
equation index: 262
type: SIMPLE_ASSIGN
world.gravityArrowHead.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_262(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,262};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[278]] /* world.gravityArrowHead.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 262;
}

/*
equation index: 263
type: SIMPLE_ASSIGN
world.gravityArrowHead.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_263(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,263};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[279]] /* world.gravityArrowHead.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 263;
}

/*
equation index: 264
type: SIMPLE_ASSIGN
world.gravityArrowHead.color[3] = 0.0
*/
void DoublePendulum_eqFunction_264(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,264};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[267]] /* world.gravityArrowHead.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 264;
}

/*
equation index: 265
type: SIMPLE_ASSIGN
world.gravityArrowHead.color[2] = 230.0
*/
void DoublePendulum_eqFunction_265(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,265};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[266]] /* world.gravityArrowHead.color[2] variable */) = 230.0;
  threadData->lastEquationSolved = 265;
}

/*
equation index: 266
type: SIMPLE_ASSIGN
world.gravityArrowHead.color[1] = 0.0
*/
void DoublePendulum_eqFunction_266(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,266};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[265]] /* world.gravityArrowHead.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 266;
}

/*
equation index: 267
type: SIMPLE_ASSIGN
world.gravityArrowColor[1] = 0.0
*/
void DoublePendulum_eqFunction_267(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,267};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[21]] /* world.gravityArrowColor[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 267;
}

/*
equation index: 268
type: SIMPLE_ASSIGN
world.gravityArrowColor[2] = 230
*/
void DoublePendulum_eqFunction_268(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,268};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[22]] /* world.gravityArrowColor[2] DISCRETE */) = ((modelica_integer) 230);
  threadData->lastEquationSolved = 268;
}

/*
equation index: 269
type: SIMPLE_ASSIGN
world.gravityArrowColor[3] = 0.0
*/
void DoublePendulum_eqFunction_269(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,269};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[23]] /* world.gravityArrowColor[3] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 269;
}

/*
equation index: 270
type: SIMPLE_ASSIGN
revolute1.cylinder.r_shape[1] = (-revolute1.e[1]) * 0.5 * revolute1.cylinderLength
*/
void DoublePendulum_eqFunction_270(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,270};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[192]] /* revolute1.cylinder.r_shape[1] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[242]] /* revolute1.e[1] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[241]] /* revolute1.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 270;
}

/*
equation index: 271
type: SIMPLE_ASSIGN
revolute1.cylinder.r_shape[2] = (-revolute1.e[2]) * 0.5 * revolute1.cylinderLength
*/
void DoublePendulum_eqFunction_271(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[193]] /* revolute1.cylinder.r_shape[2] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[243]] /* revolute1.e[2] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[241]] /* revolute1.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 271;
}

/*
equation index: 272
type: SIMPLE_ASSIGN
revolute1.cylinder.r_shape[3] = (-revolute1.e[3]) * 0.5 * revolute1.cylinderLength
*/
void DoublePendulum_eqFunction_272(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,272};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[194]] /* revolute1.cylinder.r_shape[3] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[244]] /* revolute1.e[3] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[241]] /* revolute1.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 272;
}

/*
equation index: 273
type: SIMPLE_ASSIGN
revolute1.cylinder.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_273(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[195]] /* revolute1.cylinder.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 273;
}

/*
equation index: 274
type: SIMPLE_ASSIGN
revolute1.cylinder.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_274(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,274};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[196]] /* revolute1.cylinder.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 274;
}

/*
equation index: 275
type: SIMPLE_ASSIGN
revolute1.cylinder.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_275(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[197]] /* revolute1.cylinder.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 275;
}

/*
equation index: 276
type: SIMPLE_ASSIGN
revolute1.cylinder.color[3] = 0.0
*/
void DoublePendulum_eqFunction_276(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,276};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[190]] /* revolute1.cylinder.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 276;
}

/*
equation index: 277
type: SIMPLE_ASSIGN
revolute1.cylinder.color[2] = 0.0
*/
void DoublePendulum_eqFunction_277(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[189]] /* revolute1.cylinder.color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 277;
}

/*
equation index: 278
type: SIMPLE_ASSIGN
revolute1.cylinder.color[1] = 255.0
*/
void DoublePendulum_eqFunction_278(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,278};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[188]] /* revolute1.cylinder.color[1] variable */) = 255.0;
  threadData->lastEquationSolved = 278;
}

/*
equation index: 279
type: SIMPLE_ASSIGN
revolute1.cylinderColor[1] = 255
*/
void DoublePendulum_eqFunction_279(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,279};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[12]] /* revolute1.cylinderColor[1] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 279;
}

/*
equation index: 280
type: SIMPLE_ASSIGN
revolute1.cylinderColor[2] = 0.0
*/
void DoublePendulum_eqFunction_280(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,280};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[13]] /* revolute1.cylinderColor[2] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 280;
}

/*
equation index: 281
type: SIMPLE_ASSIGN
revolute1.cylinderColor[3] = 0.0
*/
void DoublePendulum_eqFunction_281(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,281};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[14]] /* revolute1.cylinderColor[3] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 281;
}

/*
equation index: 282
type: SIMPLE_ASSIGN
boxBody1.body.sphereColor[1] = 0.0
*/
void DoublePendulum_eqFunction_282(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,282};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[0]] /* boxBody1.body.sphereColor[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 282;
}

/*
equation index: 283
type: SIMPLE_ASSIGN
boxBody1.body.sphereColor[2] = 128
*/
void DoublePendulum_eqFunction_283(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,283};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[1]] /* boxBody1.body.sphereColor[2] DISCRETE */) = ((modelica_integer) 128);
  threadData->lastEquationSolved = 283;
}

/*
equation index: 284
type: SIMPLE_ASSIGN
boxBody1.body.sphereColor[3] = 255
*/
void DoublePendulum_eqFunction_284(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,284};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[2]] /* boxBody1.body.sphereColor[3] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 284;
}

/*
equation index: 285
type: SIMPLE_ASSIGN
$START.boxBody1.body.Q[1] = boxBody1.body.Q_start[1]
*/
void DoublePendulum_eqFunction_285(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,285};
  ((modelica_real *)((data->modelData->realVarsData[45] /* boxBody1.body.Q[1] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* boxBody1.body.Q_start[1] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* boxBody1.body.Q[1] variable */) = ((modelica_real *)((data->modelData->realVarsData[45] /* boxBody1.body.Q[1] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[45].info /* boxBody1.body.Q[1] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* boxBody1.body.Q[1] variable */));
  threadData->lastEquationSolved = 285;
}

/*
equation index: 286
type: SIMPLE_ASSIGN
boxBody1.body.Q[1] = 0.0
*/
void DoublePendulum_eqFunction_286(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,286};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* boxBody1.body.Q[1] variable */) = 0.0;
  threadData->lastEquationSolved = 286;
}

/*
equation index: 287
type: SIMPLE_ASSIGN
$START.boxBody1.body.Q[2] = boxBody1.body.Q_start[2]
*/
void DoublePendulum_eqFunction_287(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,287};
  ((modelica_real *)((data->modelData->realVarsData[46] /* boxBody1.body.Q[2] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[40]] /* boxBody1.body.Q_start[2] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* boxBody1.body.Q[2] variable */) = ((modelica_real *)((data->modelData->realVarsData[46] /* boxBody1.body.Q[2] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[46].info /* boxBody1.body.Q[2] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* boxBody1.body.Q[2] variable */));
  threadData->lastEquationSolved = 287;
}

/*
equation index: 288
type: SIMPLE_ASSIGN
boxBody1.body.Q[2] = 0.0
*/
void DoublePendulum_eqFunction_288(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,288};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* boxBody1.body.Q[2] variable */) = 0.0;
  threadData->lastEquationSolved = 288;
}

/*
equation index: 289
type: SIMPLE_ASSIGN
$START.boxBody1.body.Q[3] = boxBody1.body.Q_start[3]
*/
void DoublePendulum_eqFunction_289(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,289};
  ((modelica_real *)((data->modelData->realVarsData[47] /* boxBody1.body.Q[3] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[41]] /* boxBody1.body.Q_start[3] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* boxBody1.body.Q[3] variable */) = ((modelica_real *)((data->modelData->realVarsData[47] /* boxBody1.body.Q[3] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[47].info /* boxBody1.body.Q[3] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* boxBody1.body.Q[3] variable */));
  threadData->lastEquationSolved = 289;
}

/*
equation index: 290
type: SIMPLE_ASSIGN
boxBody1.body.Q[3] = 0.0
*/
void DoublePendulum_eqFunction_290(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,290};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* boxBody1.body.Q[3] variable */) = 0.0;
  threadData->lastEquationSolved = 290;
}

/*
equation index: 291
type: SIMPLE_ASSIGN
$START.boxBody1.body.Q[4] = boxBody1.body.Q_start[4]
*/
void DoublePendulum_eqFunction_291(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,291};
  ((modelica_real *)((data->modelData->realVarsData[48] /* boxBody1.body.Q[4] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[42]] /* boxBody1.body.Q_start[4] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* boxBody1.body.Q[4] variable */) = ((modelica_real *)((data->modelData->realVarsData[48] /* boxBody1.body.Q[4] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[48].info /* boxBody1.body.Q[4] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* boxBody1.body.Q[4] variable */));
  threadData->lastEquationSolved = 291;
}

/*
equation index: 292
type: SIMPLE_ASSIGN
boxBody1.body.Q[4] = 1.0
*/
void DoublePendulum_eqFunction_292(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,292};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* boxBody1.body.Q[4] variable */) = 1.0;
  threadData->lastEquationSolved = 292;
}

/*
equation index: 293
type: SIMPLE_ASSIGN
boxBody1.body.phi[1] = 0.0
*/
void DoublePendulum_eqFunction_293(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[60]] /* boxBody1.body.phi[1] variable */) = 0.0;
  threadData->lastEquationSolved = 293;
}

/*
equation index: 294
type: SIMPLE_ASSIGN
boxBody1.body.phi[2] = 0.0
*/
void DoublePendulum_eqFunction_294(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,294};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[61]] /* boxBody1.body.phi[2] variable */) = 0.0;
  threadData->lastEquationSolved = 294;
}

/*
equation index: 295
type: SIMPLE_ASSIGN
boxBody1.body.phi[3] = 0.0
*/
void DoublePendulum_eqFunction_295(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[62]] /* boxBody1.body.phi[3] variable */) = 0.0;
  threadData->lastEquationSolved = 295;
}

/*
equation index: 296
type: SIMPLE_ASSIGN
boxBody1.body.phi_d[1] = 0.0
*/
void DoublePendulum_eqFunction_296(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,296};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[63]] /* boxBody1.body.phi_d[1] variable */) = 0.0;
  threadData->lastEquationSolved = 296;
}

/*
equation index: 297
type: SIMPLE_ASSIGN
boxBody1.body.phi_d[2] = 0.0
*/
void DoublePendulum_eqFunction_297(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[64]] /* boxBody1.body.phi_d[2] variable */) = 0.0;
  threadData->lastEquationSolved = 297;
}

/*
equation index: 298
type: SIMPLE_ASSIGN
boxBody1.body.phi_d[3] = 0.0
*/
void DoublePendulum_eqFunction_298(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,298};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[65]] /* boxBody1.body.phi_d[3] variable */) = 0.0;
  threadData->lastEquationSolved = 298;
}

/*
equation index: 299
type: SIMPLE_ASSIGN
boxBody1.body.phi_dd[1] = 0.0
*/
void DoublePendulum_eqFunction_299(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,299};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[66]] /* boxBody1.body.phi_dd[1] variable */) = 0.0;
  threadData->lastEquationSolved = 299;
}

/*
equation index: 300
type: SIMPLE_ASSIGN
boxBody1.body.phi_dd[2] = 0.0
*/
void DoublePendulum_eqFunction_300(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,300};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[67]] /* boxBody1.body.phi_dd[2] variable */) = 0.0;
  threadData->lastEquationSolved = 300;
}

/*
equation index: 301
type: SIMPLE_ASSIGN
boxBody1.body.phi_dd[3] = 0.0
*/
void DoublePendulum_eqFunction_301(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[68]] /* boxBody1.body.phi_dd[3] variable */) = 0.0;
  threadData->lastEquationSolved = 301;
}

/*
equation index: 302
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.r_shape[1] = 0.0
*/
void DoublePendulum_eqFunction_302(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,302};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[86]] /* boxBody1.frameTranslation.shape.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 302;
}

/*
equation index: 303
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.r_shape[2] = 0.0
*/
void DoublePendulum_eqFunction_303(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,303};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[87]] /* boxBody1.frameTranslation.shape.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 303;
}

/*
equation index: 304
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.r_shape[3] = 0.0
*/
void DoublePendulum_eqFunction_304(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,304};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[88]] /* boxBody1.frameTranslation.shape.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 304;
}

/*
equation index: 305
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.lengthDirection[1] = 0.5
*/
void DoublePendulum_eqFunction_305(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,305};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[83]] /* boxBody1.frameTranslation.shape.lengthDirection[1] variable */) = 0.5;
  threadData->lastEquationSolved = 305;
}

/*
equation index: 306
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.lengthDirection[2] = 0.0
*/
void DoublePendulum_eqFunction_306(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,306};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[84]] /* boxBody1.frameTranslation.shape.lengthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 306;
}

/*
equation index: 307
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.lengthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_307(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[85]] /* boxBody1.frameTranslation.shape.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 307;
}

/*
equation index: 308
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_308(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,308};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[89]] /* boxBody1.frameTranslation.shape.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 308;
}

/*
equation index: 309
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_309(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[90]] /* boxBody1.frameTranslation.shape.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 309;
}

/*
equation index: 310
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_310(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,310};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[91]] /* boxBody1.frameTranslation.shape.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 310;
}

/*
equation index: 311
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.color[3] = 255.0
*/
void DoublePendulum_eqFunction_311(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[82]] /* boxBody1.frameTranslation.shape.color[3] variable */) = 255.0;
  threadData->lastEquationSolved = 311;
}

/*
equation index: 312
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.color[2] = 128.0
*/
void DoublePendulum_eqFunction_312(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,312};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[81]] /* boxBody1.frameTranslation.shape.color[2] variable */) = 128.0;
  threadData->lastEquationSolved = 312;
}

/*
equation index: 313
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.color[1] = 0.0
*/
void DoublePendulum_eqFunction_313(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[80]] /* boxBody1.frameTranslation.shape.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 313;
}

/*
equation index: 314
type: SIMPLE_ASSIGN
boxBody1.color[1] = 0.0
*/
void DoublePendulum_eqFunction_314(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,314};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[3]] /* boxBody1.color[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 314;
}

/*
equation index: 315
type: SIMPLE_ASSIGN
boxBody1.color[2] = 128
*/
void DoublePendulum_eqFunction_315(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,315};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[4]] /* boxBody1.color[2] DISCRETE */) = ((modelica_integer) 128);
  threadData->lastEquationSolved = 315;
}

/*
equation index: 316
type: SIMPLE_ASSIGN
boxBody1.color[3] = 255
*/
void DoublePendulum_eqFunction_316(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,316};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[5]] /* boxBody1.color[3] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 316;
}

/*
equation index: 317
type: SIMPLE_ASSIGN
revolute2.cylinder.r_shape[1] = (-revolute2.e[1]) * 0.5 * revolute2.cylinderLength
*/
void DoublePendulum_eqFunction_317(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[222]] /* revolute2.cylinder.r_shape[1] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[252]] /* revolute2.e[1] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[251]] /* revolute2.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 317;
}

/*
equation index: 318
type: SIMPLE_ASSIGN
revolute2.cylinder.r_shape[2] = (-revolute2.e[2]) * 0.5 * revolute2.cylinderLength
*/
void DoublePendulum_eqFunction_318(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,318};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[223]] /* revolute2.cylinder.r_shape[2] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[253]] /* revolute2.e[2] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[251]] /* revolute2.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 318;
}

/*
equation index: 319
type: SIMPLE_ASSIGN
revolute2.cylinder.r_shape[3] = (-revolute2.e[3]) * 0.5 * revolute2.cylinderLength
*/
void DoublePendulum_eqFunction_319(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[224]] /* revolute2.cylinder.r_shape[3] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[254]] /* revolute2.e[3] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[251]] /* revolute2.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 319;
}

/*
equation index: 320
type: SIMPLE_ASSIGN
revolute2.cylinder.widthDirection[1] = 0.0
*/
void DoublePendulum_eqFunction_320(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,320};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[225]] /* revolute2.cylinder.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 320;
}

/*
equation index: 321
type: SIMPLE_ASSIGN
revolute2.cylinder.widthDirection[2] = 1.0
*/
void DoublePendulum_eqFunction_321(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[226]] /* revolute2.cylinder.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 321;
}

/*
equation index: 322
type: SIMPLE_ASSIGN
revolute2.cylinder.widthDirection[3] = 0.0
*/
void DoublePendulum_eqFunction_322(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,322};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[227]] /* revolute2.cylinder.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 322;
}

/*
equation index: 323
type: SIMPLE_ASSIGN
revolute2.cylinder.color[3] = 0.0
*/
void DoublePendulum_eqFunction_323(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[220]] /* revolute2.cylinder.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 323;
}

/*
equation index: 324
type: SIMPLE_ASSIGN
revolute2.cylinder.color[2] = 0.0
*/
void DoublePendulum_eqFunction_324(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,324};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[219]] /* revolute2.cylinder.color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 324;
}

/*
equation index: 325
type: SIMPLE_ASSIGN
revolute2.cylinder.color[1] = 255.0
*/
void DoublePendulum_eqFunction_325(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[218]] /* revolute2.cylinder.color[1] variable */) = 255.0;
  threadData->lastEquationSolved = 325;
}

/*
equation index: 326
type: SIMPLE_ASSIGN
revolute2.cylinderColor[1] = 255
*/
void DoublePendulum_eqFunction_326(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,326};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[15]] /* revolute2.cylinderColor[1] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 326;
}

/*
equation index: 327
type: SIMPLE_ASSIGN
revolute2.cylinderColor[2] = 0.0
*/
void DoublePendulum_eqFunction_327(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,327};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[16]] /* revolute2.cylinderColor[2] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 327;
}

/*
equation index: 328
type: SIMPLE_ASSIGN
revolute2.cylinderColor[3] = 0.0
*/
void DoublePendulum_eqFunction_328(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,328};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[17]] /* revolute2.cylinderColor[3] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 328;
}

/*
equation index: 329
type: SIMPLE_ASSIGN
boxBody2.body.sphereColor[1] = 0.0
*/
void DoublePendulum_eqFunction_329(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,329};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[6]] /* boxBody2.body.sphereColor[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 329;
}

/*
equation index: 330
type: SIMPLE_ASSIGN
boxBody2.body.sphereColor[2] = 128
*/
void DoublePendulum_eqFunction_330(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,330};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[7]] /* boxBody2.body.sphereColor[2] DISCRETE */) = ((modelica_integer) 128);
  threadData->lastEquationSolved = 330;
}

/*
equation index: 331
type: SIMPLE_ASSIGN
boxBody2.body.sphereColor[3] = 255
*/
void DoublePendulum_eqFunction_331(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,331};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[8]] /* boxBody2.body.sphereColor[3] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 331;
}

/*
equation index: 332
type: SIMPLE_ASSIGN
$START.boxBody2.body.Q[1] = boxBody2.body.Q_start[1]
*/
void DoublePendulum_eqFunction_332(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,332};
  ((modelica_real *)((data->modelData->realVarsData[110] /* boxBody2.body.Q[1] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[158]] /* boxBody2.body.Q_start[1] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[110]] /* boxBody2.body.Q[1] variable */) = ((modelica_real *)((data->modelData->realVarsData[110] /* boxBody2.body.Q[1] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[110].info /* boxBody2.body.Q[1] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[110]] /* boxBody2.body.Q[1] variable */));
  threadData->lastEquationSolved = 332;
}

/*
equation index: 333
type: SIMPLE_ASSIGN
boxBody2.body.Q[1] = 0.0
*/
void DoublePendulum_eqFunction_333(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[110]] /* boxBody2.body.Q[1] variable */) = 0.0;
  threadData->lastEquationSolved = 333;
}

/*
equation index: 334
type: SIMPLE_ASSIGN
$START.boxBody2.body.Q[2] = boxBody2.body.Q_start[2]
*/
void DoublePendulum_eqFunction_334(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,334};
  ((modelica_real *)((data->modelData->realVarsData[111] /* boxBody2.body.Q[2] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[159]] /* boxBody2.body.Q_start[2] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[111]] /* boxBody2.body.Q[2] variable */) = ((modelica_real *)((data->modelData->realVarsData[111] /* boxBody2.body.Q[2] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[111].info /* boxBody2.body.Q[2] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[111]] /* boxBody2.body.Q[2] variable */));
  threadData->lastEquationSolved = 334;
}

/*
equation index: 335
type: SIMPLE_ASSIGN
boxBody2.body.Q[2] = 0.0
*/
void DoublePendulum_eqFunction_335(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[111]] /* boxBody2.body.Q[2] variable */) = 0.0;
  threadData->lastEquationSolved = 335;
}

/*
equation index: 336
type: SIMPLE_ASSIGN
$START.boxBody2.body.Q[3] = boxBody2.body.Q_start[3]
*/
void DoublePendulum_eqFunction_336(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,336};
  ((modelica_real *)((data->modelData->realVarsData[112] /* boxBody2.body.Q[3] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[160]] /* boxBody2.body.Q_start[3] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[112]] /* boxBody2.body.Q[3] variable */) = ((modelica_real *)((data->modelData->realVarsData[112] /* boxBody2.body.Q[3] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[112].info /* boxBody2.body.Q[3] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[112]] /* boxBody2.body.Q[3] variable */));
  threadData->lastEquationSolved = 336;
}

/*
equation index: 337
type: SIMPLE_ASSIGN
boxBody2.body.Q[3] = 0.0
*/
void DoublePendulum_eqFunction_337(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[112]] /* boxBody2.body.Q[3] variable */) = 0.0;
  threadData->lastEquationSolved = 337;
}

/*
equation index: 338
type: SIMPLE_ASSIGN
$START.boxBody2.body.Q[4] = boxBody2.body.Q_start[4]
*/
void DoublePendulum_eqFunction_338(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,338};
  ((modelica_real *)((data->modelData->realVarsData[113] /* boxBody2.body.Q[4] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[161]] /* boxBody2.body.Q_start[4] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[113]] /* boxBody2.body.Q[4] variable */) = ((modelica_real *)((data->modelData->realVarsData[113] /* boxBody2.body.Q[4] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[113].info /* boxBody2.body.Q[4] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[113]] /* boxBody2.body.Q[4] variable */));
  threadData->lastEquationSolved = 338;
}

/*
equation index: 339
type: SIMPLE_ASSIGN
boxBody2.body.Q[4] = 1.0
*/
void DoublePendulum_eqFunction_339(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[113]] /* boxBody2.body.Q[4] variable */) = 1.0;
  threadData->lastEquationSolved = 339;
}

/*
equation index: 340
type: SIMPLE_ASSIGN
boxBody2.body.phi[1] = 0.0
*/
void DoublePendulum_eqFunction_340(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,340};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[119]] /* boxBody2.body.phi[1] variable */) = 0.0;
  threadData->lastEquationSolved = 340;
}

/*
equation index: 341
type: SIMPLE_ASSIGN
boxBody2.body.phi[2] = 0.0
*/
void DoublePendulum_eqFunction_341(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[120]] /* boxBody2.body.phi[2] variable */) = 0.0;
  threadData->lastEquationSolved = 341;
}

/*
equation index: 342
type: SIMPLE_ASSIGN
boxBody2.body.phi[3] = 0.0
*/
void DoublePendulum_eqFunction_342(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,342};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[121]] /* boxBody2.body.phi[3] variable */) = 0.0;
  threadData->lastEquationSolved = 342;
}

/*
equation index: 343
type: SIMPLE_ASSIGN
boxBody2.body.phi_d[1] = 0.0
*/
void DoublePendulum_eqFunction_343(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[122]] /* boxBody2.body.phi_d[1] variable */) = 0.0;
  threadData->lastEquationSolved = 343;
}

/*
equation index: 344
type: SIMPLE_ASSIGN
boxBody2.body.phi_d[2] = 0.0
*/
void DoublePendulum_eqFunction_344(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,344};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[123]] /* boxBody2.body.phi_d[2] variable */) = 0.0;
  threadData->lastEquationSolved = 344;
}

/*
equation index: 345
type: SIMPLE_ASSIGN
boxBody2.body.phi_d[3] = 0.0
*/
void DoublePendulum_eqFunction_345(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[124]] /* boxBody2.body.phi_d[3] variable */) = 0.0;
  threadData->lastEquationSolved = 345;
}

/*
equation index: 346
type: SIMPLE_ASSIGN
boxBody2.body.phi_dd[1] = 0.0
*/
void DoublePendulum_eqFunction_346(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,346};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[125]] /* boxBody2.body.phi_dd[1] variable */) = 0.0;
  threadData->lastEquationSolved = 346;
}
OMC_DISABLE_OPT
void DoublePendulum_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[346])(DATA*, threadData_t*) = {
    DoublePendulum_eqFunction_1,
    DoublePendulum_eqFunction_2,
    DoublePendulum_eqFunction_3,
    DoublePendulum_eqFunction_4,
    DoublePendulum_eqFunction_5,
    DoublePendulum_eqFunction_6,
    DoublePendulum_eqFunction_7,
    DoublePendulum_eqFunction_8,
    DoublePendulum_eqFunction_9,
    DoublePendulum_eqFunction_10,
    DoublePendulum_eqFunction_11,
    DoublePendulum_eqFunction_12,
    DoublePendulum_eqFunction_13,
    DoublePendulum_eqFunction_14,
    DoublePendulum_eqFunction_15,
    DoublePendulum_eqFunction_16,
    DoublePendulum_eqFunction_17,
    DoublePendulum_eqFunction_18,
    DoublePendulum_eqFunction_19,
    DoublePendulum_eqFunction_20,
    DoublePendulum_eqFunction_21,
    DoublePendulum_eqFunction_22,
    DoublePendulum_eqFunction_23,
    DoublePendulum_eqFunction_24,
    DoublePendulum_eqFunction_25,
    DoublePendulum_eqFunction_26,
    DoublePendulum_eqFunction_27,
    DoublePendulum_eqFunction_28,
    DoublePendulum_eqFunction_29,
    DoublePendulum_eqFunction_30,
    DoublePendulum_eqFunction_31,
    DoublePendulum_eqFunction_32,
    DoublePendulum_eqFunction_33,
    DoublePendulum_eqFunction_34,
    DoublePendulum_eqFunction_35,
    DoublePendulum_eqFunction_36,
    DoublePendulum_eqFunction_37,
    DoublePendulum_eqFunction_38,
    DoublePendulum_eqFunction_39,
    DoublePendulum_eqFunction_40,
    DoublePendulum_eqFunction_41,
    DoublePendulum_eqFunction_42,
    DoublePendulum_eqFunction_43,
    DoublePendulum_eqFunction_44,
    DoublePendulum_eqFunction_45,
    DoublePendulum_eqFunction_46,
    DoublePendulum_eqFunction_47,
    DoublePendulum_eqFunction_48,
    DoublePendulum_eqFunction_49,
    DoublePendulum_eqFunction_50,
    DoublePendulum_eqFunction_51,
    DoublePendulum_eqFunction_52,
    DoublePendulum_eqFunction_53,
    DoublePendulum_eqFunction_54,
    DoublePendulum_eqFunction_55,
    DoublePendulum_eqFunction_56,
    DoublePendulum_eqFunction_57,
    DoublePendulum_eqFunction_58,
    DoublePendulum_eqFunction_59,
    DoublePendulum_eqFunction_60,
    DoublePendulum_eqFunction_61,
    DoublePendulum_eqFunction_62,
    DoublePendulum_eqFunction_63,
    DoublePendulum_eqFunction_64,
    DoublePendulum_eqFunction_65,
    DoublePendulum_eqFunction_66,
    DoublePendulum_eqFunction_67,
    DoublePendulum_eqFunction_68,
    DoublePendulum_eqFunction_69,
    DoublePendulum_eqFunction_70,
    DoublePendulum_eqFunction_71,
    DoublePendulum_eqFunction_72,
    DoublePendulum_eqFunction_73,
    DoublePendulum_eqFunction_74,
    DoublePendulum_eqFunction_75,
    DoublePendulum_eqFunction_76,
    DoublePendulum_eqFunction_77,
    DoublePendulum_eqFunction_78,
    DoublePendulum_eqFunction_79,
    DoublePendulum_eqFunction_80,
    DoublePendulum_eqFunction_81,
    DoublePendulum_eqFunction_82,
    DoublePendulum_eqFunction_83,
    DoublePendulum_eqFunction_84,
    DoublePendulum_eqFunction_85,
    DoublePendulum_eqFunction_86,
    DoublePendulum_eqFunction_87,
    DoublePendulum_eqFunction_88,
    DoublePendulum_eqFunction_89,
    DoublePendulum_eqFunction_90,
    DoublePendulum_eqFunction_91,
    DoublePendulum_eqFunction_92,
    DoublePendulum_eqFunction_93,
    DoublePendulum_eqFunction_94,
    DoublePendulum_eqFunction_95,
    DoublePendulum_eqFunction_96,
    DoublePendulum_eqFunction_97,
    DoublePendulum_eqFunction_98,
    DoublePendulum_eqFunction_99,
    DoublePendulum_eqFunction_100,
    DoublePendulum_eqFunction_101,
    DoublePendulum_eqFunction_102,
    DoublePendulum_eqFunction_103,
    DoublePendulum_eqFunction_104,
    DoublePendulum_eqFunction_105,
    DoublePendulum_eqFunction_106,
    DoublePendulum_eqFunction_107,
    DoublePendulum_eqFunction_108,
    DoublePendulum_eqFunction_109,
    DoublePendulum_eqFunction_110,
    DoublePendulum_eqFunction_111,
    DoublePendulum_eqFunction_112,
    DoublePendulum_eqFunction_113,
    DoublePendulum_eqFunction_114,
    DoublePendulum_eqFunction_115,
    DoublePendulum_eqFunction_116,
    DoublePendulum_eqFunction_117,
    DoublePendulum_eqFunction_118,
    DoublePendulum_eqFunction_119,
    DoublePendulum_eqFunction_120,
    DoublePendulum_eqFunction_121,
    DoublePendulum_eqFunction_122,
    DoublePendulum_eqFunction_123,
    DoublePendulum_eqFunction_124,
    DoublePendulum_eqFunction_125,
    DoublePendulum_eqFunction_126,
    DoublePendulum_eqFunction_127,
    DoublePendulum_eqFunction_128,
    DoublePendulum_eqFunction_129,
    DoublePendulum_eqFunction_130,
    DoublePendulum_eqFunction_131,
    DoublePendulum_eqFunction_132,
    DoublePendulum_eqFunction_133,
    DoublePendulum_eqFunction_134,
    DoublePendulum_eqFunction_135,
    DoublePendulum_eqFunction_136,
    DoublePendulum_eqFunction_137,
    DoublePendulum_eqFunction_138,
    DoublePendulum_eqFunction_139,
    DoublePendulum_eqFunction_140,
    DoublePendulum_eqFunction_141,
    DoublePendulum_eqFunction_142,
    DoublePendulum_eqFunction_143,
    DoublePendulum_eqFunction_144,
    DoublePendulum_eqFunction_145,
    DoublePendulum_eqFunction_146,
    DoublePendulum_eqFunction_147,
    DoublePendulum_eqFunction_148,
    DoublePendulum_eqFunction_149,
    DoublePendulum_eqFunction_150,
    DoublePendulum_eqFunction_151,
    DoublePendulum_eqFunction_152,
    DoublePendulum_eqFunction_153,
    DoublePendulum_eqFunction_154,
    DoublePendulum_eqFunction_155,
    DoublePendulum_eqFunction_156,
    DoublePendulum_eqFunction_157,
    DoublePendulum_eqFunction_158,
    DoublePendulum_eqFunction_159,
    DoublePendulum_eqFunction_160,
    DoublePendulum_eqFunction_161,
    DoublePendulum_eqFunction_162,
    DoublePendulum_eqFunction_163,
    DoublePendulum_eqFunction_164,
    DoublePendulum_eqFunction_165,
    DoublePendulum_eqFunction_166,
    DoublePendulum_eqFunction_167,
    DoublePendulum_eqFunction_168,
    DoublePendulum_eqFunction_169,
    DoublePendulum_eqFunction_170,
    DoublePendulum_eqFunction_171,
    DoublePendulum_eqFunction_172,
    DoublePendulum_eqFunction_173,
    DoublePendulum_eqFunction_174,
    DoublePendulum_eqFunction_175,
    DoublePendulum_eqFunction_176,
    DoublePendulum_eqFunction_177,
    DoublePendulum_eqFunction_178,
    DoublePendulum_eqFunction_179,
    DoublePendulum_eqFunction_180,
    DoublePendulum_eqFunction_181,
    DoublePendulum_eqFunction_182,
    DoublePendulum_eqFunction_183,
    DoublePendulum_eqFunction_184,
    DoublePendulum_eqFunction_185,
    DoublePendulum_eqFunction_186,
    DoublePendulum_eqFunction_187,
    DoublePendulum_eqFunction_188,
    DoublePendulum_eqFunction_189,
    DoublePendulum_eqFunction_190,
    DoublePendulum_eqFunction_191,
    DoublePendulum_eqFunction_192,
    DoublePendulum_eqFunction_193,
    DoublePendulum_eqFunction_194,
    DoublePendulum_eqFunction_195,
    DoublePendulum_eqFunction_196,
    DoublePendulum_eqFunction_197,
    DoublePendulum_eqFunction_198,
    DoublePendulum_eqFunction_199,
    DoublePendulum_eqFunction_200,
    DoublePendulum_eqFunction_201,
    DoublePendulum_eqFunction_202,
    DoublePendulum_eqFunction_203,
    DoublePendulum_eqFunction_204,
    DoublePendulum_eqFunction_205,
    DoublePendulum_eqFunction_206,
    DoublePendulum_eqFunction_207,
    DoublePendulum_eqFunction_208,
    DoublePendulum_eqFunction_209,
    DoublePendulum_eqFunction_210,
    DoublePendulum_eqFunction_211,
    DoublePendulum_eqFunction_212,
    DoublePendulum_eqFunction_213,
    DoublePendulum_eqFunction_214,
    DoublePendulum_eqFunction_215,
    DoublePendulum_eqFunction_216,
    DoublePendulum_eqFunction_217,
    DoublePendulum_eqFunction_218,
    DoublePendulum_eqFunction_219,
    DoublePendulum_eqFunction_220,
    DoublePendulum_eqFunction_221,
    DoublePendulum_eqFunction_222,
    DoublePendulum_eqFunction_223,
    DoublePendulum_eqFunction_224,
    DoublePendulum_eqFunction_225,
    DoublePendulum_eqFunction_226,
    DoublePendulum_eqFunction_227,
    DoublePendulum_eqFunction_228,
    DoublePendulum_eqFunction_229,
    DoublePendulum_eqFunction_230,
    DoublePendulum_eqFunction_231,
    DoublePendulum_eqFunction_232,
    DoublePendulum_eqFunction_233,
    DoublePendulum_eqFunction_234,
    DoublePendulum_eqFunction_235,
    DoublePendulum_eqFunction_236,
    DoublePendulum_eqFunction_237,
    DoublePendulum_eqFunction_238,
    DoublePendulum_eqFunction_239,
    DoublePendulum_eqFunction_240,
    DoublePendulum_eqFunction_241,
    DoublePendulum_eqFunction_242,
    DoublePendulum_eqFunction_243,
    DoublePendulum_eqFunction_244,
    DoublePendulum_eqFunction_245,
    DoublePendulum_eqFunction_246,
    DoublePendulum_eqFunction_247,
    DoublePendulum_eqFunction_248,
    DoublePendulum_eqFunction_249,
    DoublePendulum_eqFunction_250,
    DoublePendulum_eqFunction_251,
    DoublePendulum_eqFunction_252,
    DoublePendulum_eqFunction_253,
    DoublePendulum_eqFunction_254,
    DoublePendulum_eqFunction_255,
    DoublePendulum_eqFunction_256,
    DoublePendulum_eqFunction_257,
    DoublePendulum_eqFunction_258,
    DoublePendulum_eqFunction_259,
    DoublePendulum_eqFunction_260,
    DoublePendulum_eqFunction_261,
    DoublePendulum_eqFunction_262,
    DoublePendulum_eqFunction_263,
    DoublePendulum_eqFunction_264,
    DoublePendulum_eqFunction_265,
    DoublePendulum_eqFunction_266,
    DoublePendulum_eqFunction_267,
    DoublePendulum_eqFunction_268,
    DoublePendulum_eqFunction_269,
    DoublePendulum_eqFunction_270,
    DoublePendulum_eqFunction_271,
    DoublePendulum_eqFunction_272,
    DoublePendulum_eqFunction_273,
    DoublePendulum_eqFunction_274,
    DoublePendulum_eqFunction_275,
    DoublePendulum_eqFunction_276,
    DoublePendulum_eqFunction_277,
    DoublePendulum_eqFunction_278,
    DoublePendulum_eqFunction_279,
    DoublePendulum_eqFunction_280,
    DoublePendulum_eqFunction_281,
    DoublePendulum_eqFunction_282,
    DoublePendulum_eqFunction_283,
    DoublePendulum_eqFunction_284,
    DoublePendulum_eqFunction_285,
    DoublePendulum_eqFunction_286,
    DoublePendulum_eqFunction_287,
    DoublePendulum_eqFunction_288,
    DoublePendulum_eqFunction_289,
    DoublePendulum_eqFunction_290,
    DoublePendulum_eqFunction_291,
    DoublePendulum_eqFunction_292,
    DoublePendulum_eqFunction_293,
    DoublePendulum_eqFunction_294,
    DoublePendulum_eqFunction_295,
    DoublePendulum_eqFunction_296,
    DoublePendulum_eqFunction_297,
    DoublePendulum_eqFunction_298,
    DoublePendulum_eqFunction_299,
    DoublePendulum_eqFunction_300,
    DoublePendulum_eqFunction_301,
    DoublePendulum_eqFunction_302,
    DoublePendulum_eqFunction_303,
    DoublePendulum_eqFunction_304,
    DoublePendulum_eqFunction_305,
    DoublePendulum_eqFunction_306,
    DoublePendulum_eqFunction_307,
    DoublePendulum_eqFunction_308,
    DoublePendulum_eqFunction_309,
    DoublePendulum_eqFunction_310,
    DoublePendulum_eqFunction_311,
    DoublePendulum_eqFunction_312,
    DoublePendulum_eqFunction_313,
    DoublePendulum_eqFunction_314,
    DoublePendulum_eqFunction_315,
    DoublePendulum_eqFunction_316,
    DoublePendulum_eqFunction_317,
    DoublePendulum_eqFunction_318,
    DoublePendulum_eqFunction_319,
    DoublePendulum_eqFunction_320,
    DoublePendulum_eqFunction_321,
    DoublePendulum_eqFunction_322,
    DoublePendulum_eqFunction_323,
    DoublePendulum_eqFunction_324,
    DoublePendulum_eqFunction_325,
    DoublePendulum_eqFunction_326,
    DoublePendulum_eqFunction_327,
    DoublePendulum_eqFunction_328,
    DoublePendulum_eqFunction_329,
    DoublePendulum_eqFunction_330,
    DoublePendulum_eqFunction_331,
    DoublePendulum_eqFunction_332,
    DoublePendulum_eqFunction_333,
    DoublePendulum_eqFunction_334,
    DoublePendulum_eqFunction_335,
    DoublePendulum_eqFunction_336,
    DoublePendulum_eqFunction_337,
    DoublePendulum_eqFunction_338,
    DoublePendulum_eqFunction_339,
    DoublePendulum_eqFunction_340,
    DoublePendulum_eqFunction_341,
    DoublePendulum_eqFunction_342,
    DoublePendulum_eqFunction_343,
    DoublePendulum_eqFunction_344,
    DoublePendulum_eqFunction_345,
    DoublePendulum_eqFunction_346
  };
  
  for (int id = 0; id < 346; id++) {
    eqFunctions[id](data, threadData);
  }
}
#if defined(__cplusplus)
}
#endif