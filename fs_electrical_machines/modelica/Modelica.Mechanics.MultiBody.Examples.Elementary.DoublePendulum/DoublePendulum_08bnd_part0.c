#include "DoublePendulum_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 993
type: SIMPLE_ASSIGN
boxBody2.frameTranslation.shape.shapeType = boxBody2.frameTranslation.shapeType
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_993(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,993};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[2]] /* boxBody2.frameTranslation.shape.shapeType PARAM */) = (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[3]] /* boxBody2.frameTranslation.shapeType PARAM */);
  threadData->lastEquationSolved = 993;
}

/*
equation index: 994
type: SIMPLE_ASSIGN
boxBody2.height = boxBody2.width
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_994(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,994};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[209]] /* boxBody2.height PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[231]] /* boxBody2.width PARAM */);
  threadData->lastEquationSolved = 994;
}

/*
equation index: 995
type: SIMPLE_ASSIGN
boxBody2.frameTranslation.height = boxBody2.height
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_995(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,995};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[194]] /* boxBody2.frameTranslation.height PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[209]] /* boxBody2.height PARAM */);
  threadData->lastEquationSolved = 995;
}

/*
equation index: 996
type: SIMPLE_ASSIGN
boxBody2.frameTranslation.width = boxBody2.width
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_996(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,996};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[205]] /* boxBody2.frameTranslation.width PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[231]] /* boxBody2.width PARAM */);
  threadData->lastEquationSolved = 996;
}

/*
equation index: 997
type: SIMPLE_ASSIGN
boxBody2.frameTranslation.length = boxBody2.length
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_997(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,997};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[195]] /* boxBody2.frameTranslation.length PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */);
  threadData->lastEquationSolved = 997;
}

/*
equation index: 1011
type: SIMPLE_ASSIGN
boxBody2.body.angles_start[3] = boxBody2.angles_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1011(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1011};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[176]] /* boxBody2.body.angles_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[142]] /* boxBody2.angles_start[3] PARAM */);
  threadData->lastEquationSolved = 1011;
}

/*
equation index: 1012
type: SIMPLE_ASSIGN
boxBody2.body.phi_start[3] = boxBody2.body.angles_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1012(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1012};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[181]] /* boxBody2.body.phi_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[176]] /* boxBody2.body.angles_start[3] PARAM */);
  threadData->lastEquationSolved = 1012;
}

/*
equation index: 1013
type: SIMPLE_ASSIGN
boxBody2.body.angles_start[2] = boxBody2.angles_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1013(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1013};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[175]] /* boxBody2.body.angles_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[141]] /* boxBody2.angles_start[2] PARAM */);
  threadData->lastEquationSolved = 1013;
}

/*
equation index: 1014
type: SIMPLE_ASSIGN
boxBody2.body.phi_start[2] = boxBody2.body.angles_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1014(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1014};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[180]] /* boxBody2.body.phi_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[175]] /* boxBody2.body.angles_start[2] PARAM */);
  threadData->lastEquationSolved = 1014;
}

/*
equation index: 1015
type: SIMPLE_ASSIGN
boxBody2.body.angles_start[1] = boxBody2.angles_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1015(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1015};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[174]] /* boxBody2.body.angles_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[140]] /* boxBody2.angles_start[1] PARAM */);
  threadData->lastEquationSolved = 1015;
}

/*
equation index: 1016
type: SIMPLE_ASSIGN
boxBody2.body.phi_start[1] = boxBody2.body.angles_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1016(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1016};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[179]] /* boxBody2.body.phi_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[174]] /* boxBody2.body.angles_start[1] PARAM */);
  threadData->lastEquationSolved = 1016;
}

/*
equation index: 1017
type: ARRAY_CALL_ASSIGN

boxBody2.body.Q_start = {0.0, 0.0, 0.0, 1.0}
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1017(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1017};
  real_array tmp0;
  real_array_create(&tmp0, ((modelica_real*)&((&(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[158]] /* boxBody2.body.Q_start[1] PARAM */))[((modelica_integer) 1) - 1])), 1, (_index_t)4);
  real_array_copy_data(_OMC_LIT5, tmp0);
  threadData->lastEquationSolved = 1017;
}

/*
equation index: 1036
type: SIMPLE_ASSIGN
boxBody2.body.z_0_start[3] = boxBody2.z_0_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1036(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1036};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[191]] /* boxBody2.body.z_0_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[237]] /* boxBody2.z_0_start[3] PARAM */);
  threadData->lastEquationSolved = 1036;
}

/*
equation index: 1037
type: SIMPLE_ASSIGN
boxBody2.body.z_0_start[2] = boxBody2.z_0_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1037(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1037};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[190]] /* boxBody2.body.z_0_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[236]] /* boxBody2.z_0_start[2] PARAM */);
  threadData->lastEquationSolved = 1037;
}

/*
equation index: 1038
type: SIMPLE_ASSIGN
boxBody2.body.z_0_start[1] = boxBody2.z_0_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1038(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1038};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[189]] /* boxBody2.body.z_0_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[235]] /* boxBody2.z_0_start[1] PARAM */);
  threadData->lastEquationSolved = 1038;
}

/*
equation index: 1040
type: SIMPLE_ASSIGN
boxBody2.body.w_0_start[3] = boxBody2.w_0_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1040(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1040};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[188]] /* boxBody2.body.w_0_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[230]] /* boxBody2.w_0_start[3] PARAM */);
  threadData->lastEquationSolved = 1040;
}

/*
equation index: 1041
type: SIMPLE_ASSIGN
boxBody2.body.w_0_start[2] = boxBody2.w_0_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1041(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1041};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[187]] /* boxBody2.body.w_0_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[229]] /* boxBody2.w_0_start[2] PARAM */);
  threadData->lastEquationSolved = 1041;
}

/*
equation index: 1042
type: SIMPLE_ASSIGN
boxBody2.body.w_0_start[1] = boxBody2.w_0_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1042(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1042};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[186]] /* boxBody2.body.w_0_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[228]] /* boxBody2.w_0_start[1] PARAM */);
  threadData->lastEquationSolved = 1042;
}

/*
equation index: 1048
type: SIMPLE_ASSIGN
boxBody2.mo = boxBody2.density * boxBody2.length * boxBody2.width * boxBody2.height
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1048(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1048};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[218]] /* boxBody2.mo PARAM */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[192]] /* boxBody2.density PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[231]] /* boxBody2.width PARAM */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[209]] /* boxBody2.height PARAM */))));
  threadData->lastEquationSolved = 1048;
}

/*
equation index: 1049
type: SIMPLE_ASSIGN
boxBody2.innerHeight = boxBody2.innerWidth
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1049(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1049};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[210]] /* boxBody2.innerHeight PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[211]] /* boxBody2.innerWidth PARAM */);
  threadData->lastEquationSolved = 1049;
}

/*
equation index: 1050
type: SIMPLE_ASSIGN
boxBody2.mi = boxBody2.density * boxBody2.length * boxBody2.innerWidth * boxBody2.innerHeight
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1050(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1050};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[217]] /* boxBody2.mi PARAM */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[192]] /* boxBody2.density PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[211]] /* boxBody2.innerWidth PARAM */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[210]] /* boxBody2.innerHeight PARAM */))));
  threadData->lastEquationSolved = 1050;
}

/*
equation index: 1051
type: SIMPLE_ASSIGN
boxBody2.m = boxBody2.mo - boxBody2.mi
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1051(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1051};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[216]] /* boxBody2.m PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[218]] /* boxBody2.mo PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[217]] /* boxBody2.mi PARAM */);
  threadData->lastEquationSolved = 1051;
}

/*
equation index: 1052
type: SIMPLE_ASSIGN
boxBody2.body.m = boxBody2.m
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1052(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1052};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[178]] /* boxBody2.body.m PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[216]] /* boxBody2.m PARAM */);
  threadData->lastEquationSolved = 1052;
}

/*
equation index: 1055
type: SIMPLE_ASSIGN
boxBody2.r_CM[1] = 0.5 * boxBody2.length
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1055(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1055};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[222]] /* boxBody2.r_CM[1] PARAM */) = (0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */));
  threadData->lastEquationSolved = 1055;
}

/*
equation index: 1056
type: SIMPLE_ASSIGN
boxBody2.body.r_CM[1] = boxBody2.r_CM[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1056(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1056};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[182]] /* boxBody2.body.r_CM[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[222]] /* boxBody2.r_CM[1] PARAM */);
  threadData->lastEquationSolved = 1056;
}

/*
equation index: 1058
type: ARRAY_CALL_ASSIGN

boxBody2.I = {{0.08333333333333333 * (boxBody2.mo * (boxBody2.width ^ 2.0 + boxBody2.height ^ 2.0) - boxBody2.mi * (boxBody2.innerWidth ^ 2.0 + boxBody2.innerHeight ^ 2.0)), 0.0, 0.0}, {0.0, 0.08333333333333333 * (boxBody2.mo * (boxBody2.length ^ 2.0 + boxBody2.height ^ 2.0) - boxBody2.mi * (boxBody2.length ^ 2.0 + boxBody2.innerHeight ^ 2.0)), 0.0}, {0.0, 0.0, 0.08333333333333333 * (boxBody2.mo * (boxBody2.length ^ 2.0 + boxBody2.width ^ 2.0) - boxBody2.mi * (boxBody2.length ^ 2.0 + boxBody2.innerWidth ^ 2.0))}}
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1058(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1058};
  real_array tmp1;
  real_array tmp2;
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  modelica_real tmp6;
  real_array tmp7;
  modelica_real tmp8;
  modelica_real tmp9;
  modelica_real tmp10;
  modelica_real tmp11;
  real_array tmp12;
  modelica_real tmp13;
  modelica_real tmp14;
  modelica_real tmp15;
  modelica_real tmp16;
  real_array tmp17;
  tmp3 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[231]] /* boxBody2.width PARAM */);
  tmp4 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[209]] /* boxBody2.height PARAM */);
  tmp5 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[211]] /* boxBody2.innerWidth PARAM */);
  tmp6 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[210]] /* boxBody2.innerHeight PARAM */);
  array_alloc_scalar_real_array(&tmp2, 3, (modelica_real)(0.08333333333333333) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[218]] /* boxBody2.mo PARAM */)) * ((tmp3 * tmp3) + (tmp4 * tmp4)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[217]] /* boxBody2.mi PARAM */)) * ((tmp5 * tmp5) + (tmp6 * tmp6)))), (modelica_real)0.0, (modelica_real)0.0);
  tmp8 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */);
  tmp9 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[209]] /* boxBody2.height PARAM */);
  tmp10 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */);
  tmp11 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[210]] /* boxBody2.innerHeight PARAM */);
  array_alloc_scalar_real_array(&tmp7, 3, (modelica_real)0.0, (modelica_real)(0.08333333333333333) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[218]] /* boxBody2.mo PARAM */)) * ((tmp8 * tmp8) + (tmp9 * tmp9)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[217]] /* boxBody2.mi PARAM */)) * ((tmp10 * tmp10) + (tmp11 * tmp11)))), (modelica_real)0.0);
  tmp13 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */);
  tmp14 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[231]] /* boxBody2.width PARAM */);
  tmp15 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[212]] /* boxBody2.length PARAM */);
  tmp16 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[211]] /* boxBody2.innerWidth PARAM */);
  array_alloc_scalar_real_array(&tmp12, 3, (modelica_real)0.0, (modelica_real)0.0, (modelica_real)(0.08333333333333333) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[218]] /* boxBody2.mo PARAM */)) * ((tmp13 * tmp13) + (tmp14 * tmp14)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[217]] /* boxBody2.mi PARAM */)) * ((tmp15 * tmp15) + (tmp16 * tmp16)))));
  array_alloc_real_array(&tmp1, 3, tmp2, tmp7, tmp12);
  real_array_create(&tmp17, ((modelica_real*)&((&(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[119]] /* boxBody2.I[1,1] PARAM */))[(((modelica_integer) 1) - 1) * 3 + (((modelica_integer) 1)-1)])), 2, (_index_t)3, (_index_t)3);
  real_array_copy_data(tmp1, tmp17);
  threadData->lastEquationSolved = 1058;
}

/*
equation index: 1100
type: SIMPLE_ASSIGN
revolute2.cylinder.shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1100(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1100};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[5]] /* revolute2.cylinder.shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1100;
}

/*
equation index: 1105
type: SIMPLE_ASSIGN
revolute2.cylinderDiameter = world.defaultJointWidth
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1105(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1105};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[250]] /* revolute2.cylinderDiameter PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[268]] /* world.defaultJointWidth PARAM */);
  threadData->lastEquationSolved = 1105;
}

/*
equation index: 1106
type: SIMPLE_ASSIGN
revolute2.cylinderLength = world.defaultJointLength
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1106(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1106};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[251]] /* revolute2.cylinderLength PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[267]] /* world.defaultJointLength PARAM */);
  threadData->lastEquationSolved = 1106;
}

/*
equation index: 1112
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.shape.shapeType = boxBody1.frameTranslation.shapeType
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1112(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1112};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[0]] /* boxBody1.frameTranslation.shape.shapeType PARAM */) = (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[1]] /* boxBody1.frameTranslation.shapeType PARAM */);
  threadData->lastEquationSolved = 1112;
}

/*
equation index: 1113
type: SIMPLE_ASSIGN
boxBody1.height = boxBody1.width
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1113(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1113};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[90]] /* boxBody1.height PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[112]] /* boxBody1.width PARAM */);
  threadData->lastEquationSolved = 1113;
}

/*
equation index: 1114
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.height = boxBody1.height
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1114(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1114};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[75]] /* boxBody1.frameTranslation.height PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[90]] /* boxBody1.height PARAM */);
  threadData->lastEquationSolved = 1114;
}

/*
equation index: 1115
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.width = boxBody1.width
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1115(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1115};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[86]] /* boxBody1.frameTranslation.width PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[112]] /* boxBody1.width PARAM */);
  threadData->lastEquationSolved = 1115;
}

/*
equation index: 1116
type: SIMPLE_ASSIGN
boxBody1.frameTranslation.length = boxBody1.length
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1116(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1116};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[76]] /* boxBody1.frameTranslation.length PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */);
  threadData->lastEquationSolved = 1116;
}

/*
equation index: 1130
type: SIMPLE_ASSIGN
boxBody1.body.angles_start[3] = boxBody1.angles_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1130(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1130};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[57]] /* boxBody1.body.angles_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[23]] /* boxBody1.angles_start[3] PARAM */);
  threadData->lastEquationSolved = 1130;
}

/*
equation index: 1131
type: SIMPLE_ASSIGN
boxBody1.body.phi_start[3] = boxBody1.body.angles_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1131(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1131};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[62]] /* boxBody1.body.phi_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[57]] /* boxBody1.body.angles_start[3] PARAM */);
  threadData->lastEquationSolved = 1131;
}

/*
equation index: 1132
type: SIMPLE_ASSIGN
boxBody1.body.angles_start[2] = boxBody1.angles_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1132(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1132};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[56]] /* boxBody1.body.angles_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[22]] /* boxBody1.angles_start[2] PARAM */);
  threadData->lastEquationSolved = 1132;
}

/*
equation index: 1133
type: SIMPLE_ASSIGN
boxBody1.body.phi_start[2] = boxBody1.body.angles_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1133(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1133};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[61]] /* boxBody1.body.phi_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[56]] /* boxBody1.body.angles_start[2] PARAM */);
  threadData->lastEquationSolved = 1133;
}

/*
equation index: 1134
type: SIMPLE_ASSIGN
boxBody1.body.angles_start[1] = boxBody1.angles_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1134(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1134};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* boxBody1.body.angles_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[21]] /* boxBody1.angles_start[1] PARAM */);
  threadData->lastEquationSolved = 1134;
}

/*
equation index: 1135
type: SIMPLE_ASSIGN
boxBody1.body.phi_start[1] = boxBody1.body.angles_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1135(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1135};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[60]] /* boxBody1.body.phi_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* boxBody1.body.angles_start[1] PARAM */);
  threadData->lastEquationSolved = 1135;
}

/*
equation index: 1136
type: ARRAY_CALL_ASSIGN

boxBody1.body.Q_start = {0.0, 0.0, 0.0, 1.0}
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1136(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1136};
  real_array tmp18;
  real_array_create(&tmp18, ((modelica_real*)&((&(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* boxBody1.body.Q_start[1] PARAM */))[((modelica_integer) 1) - 1])), 1, (_index_t)4);
  real_array_copy_data(_OMC_LIT5, tmp18);
  threadData->lastEquationSolved = 1136;
}

/*
equation index: 1155
type: SIMPLE_ASSIGN
boxBody1.body.z_0_start[3] = boxBody1.z_0_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1155(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1155};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[72]] /* boxBody1.body.z_0_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[118]] /* boxBody1.z_0_start[3] PARAM */);
  threadData->lastEquationSolved = 1155;
}

/*
equation index: 1156
type: SIMPLE_ASSIGN
boxBody1.body.z_0_start[2] = boxBody1.z_0_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1156(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1156};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[71]] /* boxBody1.body.z_0_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[117]] /* boxBody1.z_0_start[2] PARAM */);
  threadData->lastEquationSolved = 1156;
}

/*
equation index: 1157
type: SIMPLE_ASSIGN
boxBody1.body.z_0_start[1] = boxBody1.z_0_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1157(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1157};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[70]] /* boxBody1.body.z_0_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[116]] /* boxBody1.z_0_start[1] PARAM */);
  threadData->lastEquationSolved = 1157;
}

/*
equation index: 1159
type: SIMPLE_ASSIGN
boxBody1.body.w_0_start[3] = boxBody1.w_0_start[3]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1159(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1159};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[69]] /* boxBody1.body.w_0_start[3] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[111]] /* boxBody1.w_0_start[3] PARAM */);
  threadData->lastEquationSolved = 1159;
}

/*
equation index: 1160
type: SIMPLE_ASSIGN
boxBody1.body.w_0_start[2] = boxBody1.w_0_start[2]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1160(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1160};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[68]] /* boxBody1.body.w_0_start[2] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[110]] /* boxBody1.w_0_start[2] PARAM */);
  threadData->lastEquationSolved = 1160;
}

/*
equation index: 1161
type: SIMPLE_ASSIGN
boxBody1.body.w_0_start[1] = boxBody1.w_0_start[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1161(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1161};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[67]] /* boxBody1.body.w_0_start[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[109]] /* boxBody1.w_0_start[1] PARAM */);
  threadData->lastEquationSolved = 1161;
}

/*
equation index: 1167
type: SIMPLE_ASSIGN
boxBody1.mo = boxBody1.density * boxBody1.length * boxBody1.width * boxBody1.height
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1167(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1167};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* boxBody1.mo PARAM */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[73]] /* boxBody1.density PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[112]] /* boxBody1.width PARAM */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[90]] /* boxBody1.height PARAM */))));
  threadData->lastEquationSolved = 1167;
}

/*
equation index: 1168
type: SIMPLE_ASSIGN
boxBody1.innerHeight = boxBody1.innerWidth
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1168(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1168};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[91]] /* boxBody1.innerHeight PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[92]] /* boxBody1.innerWidth PARAM */);
  threadData->lastEquationSolved = 1168;
}

/*
equation index: 1169
type: SIMPLE_ASSIGN
boxBody1.mi = boxBody1.density * boxBody1.length * boxBody1.innerWidth * boxBody1.innerHeight
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1169(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1169};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[98]] /* boxBody1.mi PARAM */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[73]] /* boxBody1.density PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */)) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[92]] /* boxBody1.innerWidth PARAM */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[91]] /* boxBody1.innerHeight PARAM */))));
  threadData->lastEquationSolved = 1169;
}

/*
equation index: 1170
type: SIMPLE_ASSIGN
boxBody1.m = boxBody1.mo - boxBody1.mi
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1170(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1170};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[97]] /* boxBody1.m PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* boxBody1.mo PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[98]] /* boxBody1.mi PARAM */);
  threadData->lastEquationSolved = 1170;
}

/*
equation index: 1171
type: SIMPLE_ASSIGN
boxBody1.body.m = boxBody1.m
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1171(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1171};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[59]] /* boxBody1.body.m PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[97]] /* boxBody1.m PARAM */);
  threadData->lastEquationSolved = 1171;
}

/*
equation index: 1174
type: SIMPLE_ASSIGN
boxBody1.r_CM[1] = 0.5 * boxBody1.length
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1174(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1174};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[103]] /* boxBody1.r_CM[1] PARAM */) = (0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */));
  threadData->lastEquationSolved = 1174;
}

/*
equation index: 1175
type: SIMPLE_ASSIGN
boxBody1.body.r_CM[1] = boxBody1.r_CM[1]
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1175(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1175};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[63]] /* boxBody1.body.r_CM[1] PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[103]] /* boxBody1.r_CM[1] PARAM */);
  threadData->lastEquationSolved = 1175;
}

/*
equation index: 1177
type: ARRAY_CALL_ASSIGN

boxBody1.I = {{0.08333333333333333 * (boxBody1.mo * (boxBody1.width ^ 2.0 + boxBody1.height ^ 2.0) - boxBody1.mi * (boxBody1.innerWidth ^ 2.0 + boxBody1.innerHeight ^ 2.0)), 0.0, 0.0}, {0.0, 0.08333333333333333 * (boxBody1.mo * (boxBody1.length ^ 2.0 + boxBody1.height ^ 2.0) - boxBody1.mi * (boxBody1.length ^ 2.0 + boxBody1.innerHeight ^ 2.0)), 0.0}, {0.0, 0.0, 0.08333333333333333 * (boxBody1.mo * (boxBody1.length ^ 2.0 + boxBody1.width ^ 2.0) - boxBody1.mi * (boxBody1.length ^ 2.0 + boxBody1.innerWidth ^ 2.0))}}
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1177(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1177};
  real_array tmp19;
  real_array tmp20;
  modelica_real tmp21;
  modelica_real tmp22;
  modelica_real tmp23;
  modelica_real tmp24;
  real_array tmp25;
  modelica_real tmp26;
  modelica_real tmp27;
  modelica_real tmp28;
  modelica_real tmp29;
  real_array tmp30;
  modelica_real tmp31;
  modelica_real tmp32;
  modelica_real tmp33;
  modelica_real tmp34;
  real_array tmp35;
  tmp21 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[112]] /* boxBody1.width PARAM */);
  tmp22 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[90]] /* boxBody1.height PARAM */);
  tmp23 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[92]] /* boxBody1.innerWidth PARAM */);
  tmp24 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[91]] /* boxBody1.innerHeight PARAM */);
  array_alloc_scalar_real_array(&tmp20, 3, (modelica_real)(0.08333333333333333) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* boxBody1.mo PARAM */)) * ((tmp21 * tmp21) + (tmp22 * tmp22)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[98]] /* boxBody1.mi PARAM */)) * ((tmp23 * tmp23) + (tmp24 * tmp24)))), (modelica_real)0.0, (modelica_real)0.0);
  tmp26 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */);
  tmp27 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[90]] /* boxBody1.height PARAM */);
  tmp28 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */);
  tmp29 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[91]] /* boxBody1.innerHeight PARAM */);
  array_alloc_scalar_real_array(&tmp25, 3, (modelica_real)0.0, (modelica_real)(0.08333333333333333) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* boxBody1.mo PARAM */)) * ((tmp26 * tmp26) + (tmp27 * tmp27)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[98]] /* boxBody1.mi PARAM */)) * ((tmp28 * tmp28) + (tmp29 * tmp29)))), (modelica_real)0.0);
  tmp31 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */);
  tmp32 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[112]] /* boxBody1.width PARAM */);
  tmp33 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[93]] /* boxBody1.length PARAM */);
  tmp34 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[92]] /* boxBody1.innerWidth PARAM */);
  array_alloc_scalar_real_array(&tmp30, 3, (modelica_real)0.0, (modelica_real)0.0, (modelica_real)(0.08333333333333333) * (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* boxBody1.mo PARAM */)) * ((tmp31 * tmp31) + (tmp32 * tmp32)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[98]] /* boxBody1.mi PARAM */)) * ((tmp33 * tmp33) + (tmp34 * tmp34)))));
  array_alloc_real_array(&tmp19, 3, tmp20, tmp25, tmp30);
  real_array_create(&tmp35, ((modelica_real*)&((&(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* boxBody1.I[1,1] PARAM */))[(((modelica_integer) 1) - 1) * 3 + (((modelica_integer) 1)-1)])), 2, (_index_t)3, (_index_t)3);
  real_array_copy_data(tmp19, tmp35);
  threadData->lastEquationSolved = 1177;
}

/*
equation index: 1219
type: SIMPLE_ASSIGN
revolute1.cylinder.shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1219(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1219};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[4]] /* revolute1.cylinder.shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1219;
}

/*
equation index: 1224
type: SIMPLE_ASSIGN
revolute1.cylinderDiameter = world.defaultJointWidth
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1224(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1224};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[240]] /* revolute1.cylinderDiameter PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[268]] /* world.defaultJointWidth PARAM */);
  threadData->lastEquationSolved = 1224;
}

/*
equation index: 1225
type: SIMPLE_ASSIGN
revolute1.cylinderLength = world.defaultJointLength
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1225(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1225};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[241]] /* revolute1.cylinderLength PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[267]] /* world.defaultJointLength PARAM */);
  threadData->lastEquationSolved = 1225;
}

/*
equation index: 1231
type: SIMPLE_ASSIGN
world.gravityArrowHead.shapeType = "cone"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1231(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1231};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[6]] /* world.gravityArrowHead.shapeType PARAM */) = _OMC_LIT7;
  threadData->lastEquationSolved = 1231;
}

/*
equation index: 1232
type: SIMPLE_ASSIGN
world.gravityArrowLine.shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1232(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1232};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[7]] /* world.gravityArrowLine.shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1232;
}

/*
equation index: 1233
type: SIMPLE_ASSIGN
world.gravityArrowLength = 0.5 * world.axisLength
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1233(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1233};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[275]] /* world.gravityArrowLength PARAM */) = (0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[260]] /* world.axisLength PARAM */));
  threadData->lastEquationSolved = 1233;
}

/*
equation index: 1234
type: SIMPLE_ASSIGN
world.gravityArrowDiameter = world.gravityArrowLength / world.defaultWidthFraction
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1234(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1234};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[274]] /* world.gravityArrowDiameter PARAM */) = DIVISION_SIM((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[275]] /* world.gravityArrowLength PARAM */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[272]] /* world.defaultWidthFraction PARAM */),"world.defaultWidthFraction",equationIndexes);
  threadData->lastEquationSolved = 1234;
}

/*
equation index: 1235
type: SIMPLE_ASSIGN
world.gravityHeadLength = min(world.gravityArrowLength, world.gravityArrowDiameter * 4.0)
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1235(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1235};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[279]] /* world.gravityHeadLength PARAM */) = fmin((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[275]] /* world.gravityArrowLength PARAM */),((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[274]] /* world.gravityArrowDiameter PARAM */)) * (4.0));
  threadData->lastEquationSolved = 1235;
}

/*
equation index: 1236
type: SIMPLE_ASSIGN
world.gravityLineLength = max(0.0, world.gravityArrowLength - world.gravityHeadLength)
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1236(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1236};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[281]] /* world.gravityLineLength PARAM */) = fmax(0.0,(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[275]] /* world.gravityArrowLength PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[279]] /* world.gravityHeadLength PARAM */));
  threadData->lastEquationSolved = 1236;
}

/*
equation index: 1237
type: SIMPLE_ASSIGN
world.gravityHeadWidth = 3.0 * world.gravityArrowDiameter
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1237(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1237};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[280]] /* world.gravityHeadWidth PARAM */) = (3.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[274]] /* world.gravityArrowDiameter PARAM */));
  threadData->lastEquationSolved = 1237;
}

/*
equation index: 1238
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1238(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1238};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[22]] /* world.z_label.cylinders[3].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1238;
}

/*
equation index: 1239
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1239(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1239};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[21]] /* world.z_label.cylinders[2].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1239;
}

/*
equation index: 1240
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1240(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1240};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[20]] /* world.z_label.cylinders[1].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1240;
}

/*
equation index: 1242
type: SIMPLE_ASSIGN
world.z_arrowHead.shapeType = "cone"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1242(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1242};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[18]] /* world.z_arrowHead.shapeType PARAM */) = _OMC_LIT7;
  threadData->lastEquationSolved = 1242;
}

/*
equation index: 1243
type: SIMPLE_ASSIGN
world.z_arrowLine.shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1243(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1243};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[19]] /* world.z_arrowLine.shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1243;
}

/*
equation index: 1244
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1244(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1244};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[17]] /* world.y_label.cylinders[2].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1244;
}

/*
equation index: 1245
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1245(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1245};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[16]] /* world.y_label.cylinders[1].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1245;
}

/*
equation index: 1247
type: SIMPLE_ASSIGN
world.y_arrowHead.shapeType = "cone"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1247(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1247};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[14]] /* world.y_arrowHead.shapeType PARAM */) = _OMC_LIT7;
  threadData->lastEquationSolved = 1247;
}

/*
equation index: 1248
type: SIMPLE_ASSIGN
world.y_arrowLine.shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1248(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1248};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[15]] /* world.y_arrowLine.shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1248;
}

/*
equation index: 1249
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1249(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1249};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[13]] /* world.x_label.cylinders[2].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1249;
}

/*
equation index: 1250
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1250(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1250};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[12]] /* world.x_label.cylinders[1].shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1250;
}

/*
equation index: 1252
type: SIMPLE_ASSIGN
world.x_arrowHead.shapeType = "cone"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1252(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1252};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[10]] /* world.x_arrowHead.shapeType PARAM */) = _OMC_LIT7;
  threadData->lastEquationSolved = 1252;
}

/*
equation index: 1253
type: SIMPLE_ASSIGN
world.x_arrowLine.shapeType = "cylinder"
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1253(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1253};
  (data->simulationInfo->stringParameter[data->simulationInfo->stringParamsIndex[11]] /* world.x_arrowLine.shapeType PARAM */) = _OMC_LIT6;
  threadData->lastEquationSolved = 1253;
}

/*
equation index: 1254
type: SIMPLE_ASSIGN
world.labelStart = 1.05 * world.axisLength
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1254(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1254};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[290]] /* world.labelStart PARAM */) = (1.05) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[260]] /* world.axisLength PARAM */));
  threadData->lastEquationSolved = 1254;
}

/*
equation index: 1255
type: SIMPLE_ASSIGN
world.axisDiameter = world.axisLength / world.defaultFrameDiameterFraction
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1255(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1255};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[259]] /* world.axisDiameter PARAM */) = DIVISION_SIM((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[260]] /* world.axisLength PARAM */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[266]] /* world.defaultFrameDiameterFraction PARAM */),"world.defaultFrameDiameterFraction",equationIndexes);
  threadData->lastEquationSolved = 1255;
}

/*
equation index: 1256
type: SIMPLE_ASSIGN
world.scaledLabel = 3.0 * world.axisDiameter
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1256(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1256};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[298]] /* world.scaledLabel PARAM */) = (3.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[259]] /* world.axisDiameter PARAM */));
  threadData->lastEquationSolved = 1256;
}

/*
equation index: 1257
type: SIMPLE_ASSIGN
world.lineWidth = world.axisDiameter
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1257(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1257};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[292]] /* world.lineWidth PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[259]] /* world.axisDiameter PARAM */);
  threadData->lastEquationSolved = 1257;
}

/*
equation index: 1258
type: SIMPLE_ASSIGN
world.headLength = min(world.axisLength, world.axisDiameter * 5.0)
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1258(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1258};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[288]] /* world.headLength PARAM */) = fmin((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[260]] /* world.axisLength PARAM */),((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[259]] /* world.axisDiameter PARAM */)) * (5.0));
  threadData->lastEquationSolved = 1258;
}

/*
equation index: 1259
type: SIMPLE_ASSIGN
world.lineLength = max(0.0, world.axisLength - world.headLength)
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1259(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1259};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[291]] /* world.lineLength PARAM */) = fmax(0.0,(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[260]] /* world.axisLength PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[288]] /* world.headLength PARAM */));
  threadData->lastEquationSolved = 1259;
}

/*
equation index: 1260
type: SIMPLE_ASSIGN
world.headWidth = 3.0 * world.axisDiameter
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1260(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1260};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[289]] /* world.headWidth PARAM */) = (3.0) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[259]] /* world.axisDiameter PARAM */));
  threadData->lastEquationSolved = 1260;
}

/*
equation index: 1263
type: SIMPLE_ASSIGN
world.groundLength_v = world.groundLength_u
*/
OMC_DISABLE_OPT
static void DoublePendulum_eqFunction_1263(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,1263};
  (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[287]] /* world.groundLength_v PARAM */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[286]] /* world.groundLength_u PARAM */);
  threadData->lastEquationSolved = 1263;
}
extern void DoublePendulum_eqFunction_531(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_530(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_529(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_528(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_527(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_526(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_525(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_524(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_523(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_522(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_521(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_520(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_519(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_518(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_517(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_516(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_515(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_514(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_513(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_512(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_511(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_510(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_509(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_508(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_507(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_506(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_505(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_504(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_503(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_502(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_501(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_500(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_499(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_498(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_497(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_496(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_495(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_494(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_493(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_492(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_491(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_490(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_489(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_488(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_487(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_486(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_485(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_484(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_483(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_482(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_481(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_480(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_479(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_478(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_477(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_476(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_475(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_474(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_473(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_472(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_471(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_470(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_469(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_468(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_467(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_466(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_465(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_464(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_463(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_462(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_461(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_460(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_459(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_458(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_457(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_456(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_455(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_454(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_453(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_452(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_451(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_450(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_449(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_448(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_447(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_446(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_445(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_444(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_443(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_442(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_441(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_440(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_439(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_438(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_437(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_436(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_408(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_407(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_406(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_435(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_404(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_434(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_433(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_432(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_431(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_430(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_429(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_428(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_427(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_426(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_425(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_424(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_423(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_422(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_421(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_420(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_405(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_409(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_403(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_402(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_401(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_400(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_399(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_398(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_397(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_396(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_395(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_394(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_393(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_392(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_391(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_390(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_389(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_388(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_387(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_386(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_385(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_384(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_383(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_382(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_381(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_380(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_379(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_378(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_377(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_376(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_375(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_374(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_373(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_372(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_371(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_370(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_369(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_368(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_367(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_366(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_365(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_364(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_363(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_362(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_360(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_359(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_358(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_357(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_356(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_355(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_354(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_353(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_352(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_351(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_350(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_349(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_348(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_347(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_346(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_345(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_344(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_343(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_342(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_341(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_340(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_339(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_337(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_335(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_333(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_331(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_330(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_326(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_325(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_324(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_323(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_322(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_321(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_320(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_319(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_318(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_317(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_316(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_315(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_313(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_312(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_311(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_310(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_309(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_308(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_307(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_306(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_305(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_304(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_303(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_302(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_301(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_300(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_299(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_298(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_297(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_296(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_295(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_294(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_293(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_292(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_290(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_288(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_286(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_284(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_283(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_279(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_278(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_277(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_276(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_275(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_274(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_273(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_272(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_271(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_270(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_268(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_266(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_265(DATA *data, threadData_t *threadData);

extern void DoublePendulum_eqFunction_264(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void DoublePendulum_updateBoundParameters_0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[326])(DATA*, threadData_t*) = {
    DoublePendulum_eqFunction_993,
    DoublePendulum_eqFunction_994,
    DoublePendulum_eqFunction_995,
    DoublePendulum_eqFunction_996,
    DoublePendulum_eqFunction_997,
    DoublePendulum_eqFunction_1011,
    DoublePendulum_eqFunction_1012,
    DoublePendulum_eqFunction_1013,
    DoublePendulum_eqFunction_1014,
    DoublePendulum_eqFunction_1015,
    DoublePendulum_eqFunction_1016,
    DoublePendulum_eqFunction_1017,
    DoublePendulum_eqFunction_1036,
    DoublePendulum_eqFunction_1037,
    DoublePendulum_eqFunction_1038,
    DoublePendulum_eqFunction_1040,
    DoublePendulum_eqFunction_1041,
    DoublePendulum_eqFunction_1042,
    DoublePendulum_eqFunction_1048,
    DoublePendulum_eqFunction_1049,
    DoublePendulum_eqFunction_1050,
    DoublePendulum_eqFunction_1051,
    DoublePendulum_eqFunction_1052,
    DoublePendulum_eqFunction_1055,
    DoublePendulum_eqFunction_1056,
    DoublePendulum_eqFunction_1058,
    DoublePendulum_eqFunction_1100,
    DoublePendulum_eqFunction_1105,
    DoublePendulum_eqFunction_1106,
    DoublePendulum_eqFunction_1112,
    DoublePendulum_eqFunction_1113,
    DoublePendulum_eqFunction_1114,
    DoublePendulum_eqFunction_1115,
    DoublePendulum_eqFunction_1116,
    DoublePendulum_eqFunction_1130,
    DoublePendulum_eqFunction_1131,
    DoublePendulum_eqFunction_1132,
    DoublePendulum_eqFunction_1133,
    DoublePendulum_eqFunction_1134,
    DoublePendulum_eqFunction_1135,
    DoublePendulum_eqFunction_1136,
    DoublePendulum_eqFunction_1155,
    DoublePendulum_eqFunction_1156,
    DoublePendulum_eqFunction_1157,
    DoublePendulum_eqFunction_1159,
    DoublePendulum_eqFunction_1160,
    DoublePendulum_eqFunction_1161,
    DoublePendulum_eqFunction_1167,
    DoublePendulum_eqFunction_1168,
    DoublePendulum_eqFunction_1169,
    DoublePendulum_eqFunction_1170,
    DoublePendulum_eqFunction_1171,
    DoublePendulum_eqFunction_1174,
    DoublePendulum_eqFunction_1175,
    DoublePendulum_eqFunction_1177,
    DoublePendulum_eqFunction_1219,
    DoublePendulum_eqFunction_1224,
    DoublePendulum_eqFunction_1225,
    DoublePendulum_eqFunction_1231,
    DoublePendulum_eqFunction_1232,
    DoublePendulum_eqFunction_1233,
    DoublePendulum_eqFunction_1234,
    DoublePendulum_eqFunction_1235,
    DoublePendulum_eqFunction_1236,
    DoublePendulum_eqFunction_1237,
    DoublePendulum_eqFunction_1238,
    DoublePendulum_eqFunction_1239,
    DoublePendulum_eqFunction_1240,
    DoublePendulum_eqFunction_1242,
    DoublePendulum_eqFunction_1243,
    DoublePendulum_eqFunction_1244,
    DoublePendulum_eqFunction_1245,
    DoublePendulum_eqFunction_1247,
    DoublePendulum_eqFunction_1248,
    DoublePendulum_eqFunction_1249,
    DoublePendulum_eqFunction_1250,
    DoublePendulum_eqFunction_1252,
    DoublePendulum_eqFunction_1253,
    DoublePendulum_eqFunction_1254,
    DoublePendulum_eqFunction_1255,
    DoublePendulum_eqFunction_1256,
    DoublePendulum_eqFunction_1257,
    DoublePendulum_eqFunction_1258,
    DoublePendulum_eqFunction_1259,
    DoublePendulum_eqFunction_1260,
    DoublePendulum_eqFunction_1263,
    DoublePendulum_eqFunction_531,
    DoublePendulum_eqFunction_530,
    DoublePendulum_eqFunction_529,
    DoublePendulum_eqFunction_528,
    DoublePendulum_eqFunction_527,
    DoublePendulum_eqFunction_526,
    DoublePendulum_eqFunction_525,
    DoublePendulum_eqFunction_524,
    DoublePendulum_eqFunction_523,
    DoublePendulum_eqFunction_522,
    DoublePendulum_eqFunction_521,
    DoublePendulum_eqFunction_520,
    DoublePendulum_eqFunction_519,
    DoublePendulum_eqFunction_518,
    DoublePendulum_eqFunction_517,
    DoublePendulum_eqFunction_516,
    DoublePendulum_eqFunction_515,
    DoublePendulum_eqFunction_514,
    DoublePendulum_eqFunction_513,
    DoublePendulum_eqFunction_512,
    DoublePendulum_eqFunction_511,
    DoublePendulum_eqFunction_510,
    DoublePendulum_eqFunction_509,
    DoublePendulum_eqFunction_508,
    DoublePendulum_eqFunction_507,
    DoublePendulum_eqFunction_506,
    DoublePendulum_eqFunction_505,
    DoublePendulum_eqFunction_504,
    DoublePendulum_eqFunction_503,
    DoublePendulum_eqFunction_502,
    DoublePendulum_eqFunction_501,
    DoublePendulum_eqFunction_500,
    DoublePendulum_eqFunction_499,
    DoublePendulum_eqFunction_498,
    DoublePendulum_eqFunction_497,
    DoublePendulum_eqFunction_496,
    DoublePendulum_eqFunction_495,
    DoublePendulum_eqFunction_494,
    DoublePendulum_eqFunction_493,
    DoublePendulum_eqFunction_492,
    DoublePendulum_eqFunction_491,
    DoublePendulum_eqFunction_490,
    DoublePendulum_eqFunction_489,
    DoublePendulum_eqFunction_488,
    DoublePendulum_eqFunction_487,
    DoublePendulum_eqFunction_486,
    DoublePendulum_eqFunction_485,
    DoublePendulum_eqFunction_484,
    DoublePendulum_eqFunction_483,
    DoublePendulum_eqFunction_482,
    DoublePendulum_eqFunction_481,
    DoublePendulum_eqFunction_480,
    DoublePendulum_eqFunction_479,
    DoublePendulum_eqFunction_478,
    DoublePendulum_eqFunction_477,
    DoublePendulum_eqFunction_476,
    DoublePendulum_eqFunction_475,
    DoublePendulum_eqFunction_474,
    DoublePendulum_eqFunction_473,
    DoublePendulum_eqFunction_472,
    DoublePendulum_eqFunction_471,
    DoublePendulum_eqFunction_470,
    DoublePendulum_eqFunction_469,
    DoublePendulum_eqFunction_468,
    DoublePendulum_eqFunction_467,
    DoublePendulum_eqFunction_466,
    DoublePendulum_eqFunction_465,
    DoublePendulum_eqFunction_464,
    DoublePendulum_eqFunction_463,
    DoublePendulum_eqFunction_462,
    DoublePendulum_eqFunction_461,
    DoublePendulum_eqFunction_460,
    DoublePendulum_eqFunction_459,
    DoublePendulum_eqFunction_458,
    DoublePendulum_eqFunction_457,
    DoublePendulum_eqFunction_456,
    DoublePendulum_eqFunction_455,
    DoublePendulum_eqFunction_454,
    DoublePendulum_eqFunction_453,
    DoublePendulum_eqFunction_452,
    DoublePendulum_eqFunction_451,
    DoublePendulum_eqFunction_450,
    DoublePendulum_eqFunction_449,
    DoublePendulum_eqFunction_448,
    DoublePendulum_eqFunction_447,
    DoublePendulum_eqFunction_446,
    DoublePendulum_eqFunction_445,
    DoublePendulum_eqFunction_444,
    DoublePendulum_eqFunction_443,
    DoublePendulum_eqFunction_442,
    DoublePendulum_eqFunction_441,
    DoublePendulum_eqFunction_440,
    DoublePendulum_eqFunction_439,
    DoublePendulum_eqFunction_438,
    DoublePendulum_eqFunction_437,
    DoublePendulum_eqFunction_436,
    DoublePendulum_eqFunction_408,
    DoublePendulum_eqFunction_407,
    DoublePendulum_eqFunction_406,
    DoublePendulum_eqFunction_435,
    DoublePendulum_eqFunction_404,
    DoublePendulum_eqFunction_434,
    DoublePendulum_eqFunction_433,
    DoublePendulum_eqFunction_432,
    DoublePendulum_eqFunction_431,
    DoublePendulum_eqFunction_430,
    DoublePendulum_eqFunction_429,
    DoublePendulum_eqFunction_428,
    DoublePendulum_eqFunction_427,
    DoublePendulum_eqFunction_426,
    DoublePendulum_eqFunction_425,
    DoublePendulum_eqFunction_424,
    DoublePendulum_eqFunction_423,
    DoublePendulum_eqFunction_422,
    DoublePendulum_eqFunction_421,
    DoublePendulum_eqFunction_420,
    DoublePendulum_eqFunction_405,
    DoublePendulum_eqFunction_409,
    DoublePendulum_eqFunction_403,
    DoublePendulum_eqFunction_402,
    DoublePendulum_eqFunction_401,
    DoublePendulum_eqFunction_400,
    DoublePendulum_eqFunction_399,
    DoublePendulum_eqFunction_398,
    DoublePendulum_eqFunction_397,
    DoublePendulum_eqFunction_396,
    DoublePendulum_eqFunction_395,
    DoublePendulum_eqFunction_394,
    DoublePendulum_eqFunction_393,
    DoublePendulum_eqFunction_392,
    DoublePendulum_eqFunction_391,
    DoublePendulum_eqFunction_390,
    DoublePendulum_eqFunction_389,
    DoublePendulum_eqFunction_388,
    DoublePendulum_eqFunction_387,
    DoublePendulum_eqFunction_386,
    DoublePendulum_eqFunction_385,
    DoublePendulum_eqFunction_384,
    DoublePendulum_eqFunction_383,
    DoublePendulum_eqFunction_382,
    DoublePendulum_eqFunction_381,
    DoublePendulum_eqFunction_380,
    DoublePendulum_eqFunction_379,
    DoublePendulum_eqFunction_378,
    DoublePendulum_eqFunction_377,
    DoublePendulum_eqFunction_376,
    DoublePendulum_eqFunction_375,
    DoublePendulum_eqFunction_374,
    DoublePendulum_eqFunction_373,
    DoublePendulum_eqFunction_372,
    DoublePendulum_eqFunction_371,
    DoublePendulum_eqFunction_370,
    DoublePendulum_eqFunction_369,
    DoublePendulum_eqFunction_368,
    DoublePendulum_eqFunction_367,
    DoublePendulum_eqFunction_366,
    DoublePendulum_eqFunction_365,
    DoublePendulum_eqFunction_364,
    DoublePendulum_eqFunction_363,
    DoublePendulum_eqFunction_362,
    DoublePendulum_eqFunction_360,
    DoublePendulum_eqFunction_359,
    DoublePendulum_eqFunction_358,
    DoublePendulum_eqFunction_357,
    DoublePendulum_eqFunction_356,
    DoublePendulum_eqFunction_355,
    DoublePendulum_eqFunction_354,
    DoublePendulum_eqFunction_353,
    DoublePendulum_eqFunction_352,
    DoublePendulum_eqFunction_351,
    DoublePendulum_eqFunction_350,
    DoublePendulum_eqFunction_349,
    DoublePendulum_eqFunction_348,
    DoublePendulum_eqFunction_347,
    DoublePendulum_eqFunction_346,
    DoublePendulum_eqFunction_345,
    DoublePendulum_eqFunction_344,
    DoublePendulum_eqFunction_343,
    DoublePendulum_eqFunction_342,
    DoublePendulum_eqFunction_341,
    DoublePendulum_eqFunction_340,
    DoublePendulum_eqFunction_339,
    DoublePendulum_eqFunction_337,
    DoublePendulum_eqFunction_335,
    DoublePendulum_eqFunction_333,
    DoublePendulum_eqFunction_331,
    DoublePendulum_eqFunction_330,
    DoublePendulum_eqFunction_326,
    DoublePendulum_eqFunction_325,
    DoublePendulum_eqFunction_324,
    DoublePendulum_eqFunction_323,
    DoublePendulum_eqFunction_322,
    DoublePendulum_eqFunction_321,
    DoublePendulum_eqFunction_320,
    DoublePendulum_eqFunction_319,
    DoublePendulum_eqFunction_318,
    DoublePendulum_eqFunction_317,
    DoublePendulum_eqFunction_316,
    DoublePendulum_eqFunction_315,
    DoublePendulum_eqFunction_313,
    DoublePendulum_eqFunction_312,
    DoublePendulum_eqFunction_311,
    DoublePendulum_eqFunction_310,
    DoublePendulum_eqFunction_309,
    DoublePendulum_eqFunction_308,
    DoublePendulum_eqFunction_307,
    DoublePendulum_eqFunction_306,
    DoublePendulum_eqFunction_305,
    DoublePendulum_eqFunction_304,
    DoublePendulum_eqFunction_303,
    DoublePendulum_eqFunction_302,
    DoublePendulum_eqFunction_301,
    DoublePendulum_eqFunction_300,
    DoublePendulum_eqFunction_299,
    DoublePendulum_eqFunction_298,
    DoublePendulum_eqFunction_297,
    DoublePendulum_eqFunction_296,
    DoublePendulum_eqFunction_295,
    DoublePendulum_eqFunction_294,
    DoublePendulum_eqFunction_293,
    DoublePendulum_eqFunction_292,
    DoublePendulum_eqFunction_290,
    DoublePendulum_eqFunction_288,
    DoublePendulum_eqFunction_286,
    DoublePendulum_eqFunction_284,
    DoublePendulum_eqFunction_283,
    DoublePendulum_eqFunction_279,
    DoublePendulum_eqFunction_278,
    DoublePendulum_eqFunction_277,
    DoublePendulum_eqFunction_276,
    DoublePendulum_eqFunction_275,
    DoublePendulum_eqFunction_274,
    DoublePendulum_eqFunction_273,
    DoublePendulum_eqFunction_272,
    DoublePendulum_eqFunction_271,
    DoublePendulum_eqFunction_270,
    DoublePendulum_eqFunction_268,
    DoublePendulum_eqFunction_266,
    DoublePendulum_eqFunction_265,
    DoublePendulum_eqFunction_264
  };
  
  for (int id = 0; id < 326; id++) {
    eqFunctions[id](data, threadData);
  }
}
#if defined(__cplusplus)
}
#endif