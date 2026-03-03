#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 14001
type: SIMPLE_ASSIGN
arm_off[500] = 6.270618936565227 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14001(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14001};
  (data->simulationInfo->realParameter[502] /* arm_off[500] PARAM */) = (6.270618936565227) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14002
type: SIMPLE_ASSIGN
theta[500] = pitch * r_init[500] + arm_off[500]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14002(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14002};
  (data->simulationInfo->realParameter[2006] /* theta[500] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */)) + (data->simulationInfo->realParameter[502] /* arm_off[500] PARAM */);
  TRACE_POP
}

/*
equation index: 14003
type: SIMPLE_ASSIGN
arm_off[499] = 6.258052565950868 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14003};
  (data->simulationInfo->realParameter[501] /* arm_off[499] PARAM */) = (6.258052565950868) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14004
type: SIMPLE_ASSIGN
theta[499] = pitch * r_init[499] + arm_off[499]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14004(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14004};
  (data->simulationInfo->realParameter[2005] /* theta[499] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */)) + (data->simulationInfo->realParameter[501] /* arm_off[499] PARAM */);
  TRACE_POP
}

/*
equation index: 14005
type: SIMPLE_ASSIGN
arm_off[498] = 6.245486195336509 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14005};
  (data->simulationInfo->realParameter[500] /* arm_off[498] PARAM */) = (6.245486195336509) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14006
type: SIMPLE_ASSIGN
theta[498] = pitch * r_init[498] + arm_off[498]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14006(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14006};
  (data->simulationInfo->realParameter[2004] /* theta[498] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */)) + (data->simulationInfo->realParameter[500] /* arm_off[498] PARAM */);
  TRACE_POP
}

/*
equation index: 14007
type: SIMPLE_ASSIGN
arm_off[497] = 6.232919824722149 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14007};
  (data->simulationInfo->realParameter[499] /* arm_off[497] PARAM */) = (6.232919824722149) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14008
type: SIMPLE_ASSIGN
theta[497] = pitch * r_init[497] + arm_off[497]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14008(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14008};
  (data->simulationInfo->realParameter[2003] /* theta[497] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */)) + (data->simulationInfo->realParameter[499] /* arm_off[497] PARAM */);
  TRACE_POP
}

/*
equation index: 14009
type: SIMPLE_ASSIGN
arm_off[496] = 6.220353454107791 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14009};
  (data->simulationInfo->realParameter[498] /* arm_off[496] PARAM */) = (6.220353454107791) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14010
type: SIMPLE_ASSIGN
theta[496] = pitch * r_init[496] + arm_off[496]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14010(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14010};
  (data->simulationInfo->realParameter[2002] /* theta[496] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */)) + (data->simulationInfo->realParameter[498] /* arm_off[496] PARAM */);
  TRACE_POP
}

/*
equation index: 14011
type: SIMPLE_ASSIGN
arm_off[495] = 6.2077870834934314 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14011(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14011};
  (data->simulationInfo->realParameter[497] /* arm_off[495] PARAM */) = (6.2077870834934314) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14012
type: SIMPLE_ASSIGN
theta[495] = pitch * r_init[495] + arm_off[495]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14012};
  (data->simulationInfo->realParameter[2001] /* theta[495] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */)) + (data->simulationInfo->realParameter[497] /* arm_off[495] PARAM */);
  TRACE_POP
}

/*
equation index: 14013
type: SIMPLE_ASSIGN
arm_off[494] = 6.195220712879072 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14013};
  (data->simulationInfo->realParameter[496] /* arm_off[494] PARAM */) = (6.195220712879072) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14014
type: SIMPLE_ASSIGN
theta[494] = pitch * r_init[494] + arm_off[494]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14014(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14014};
  (data->simulationInfo->realParameter[2000] /* theta[494] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */)) + (data->simulationInfo->realParameter[496] /* arm_off[494] PARAM */);
  TRACE_POP
}

/*
equation index: 14015
type: SIMPLE_ASSIGN
arm_off[493] = 6.182654342264713 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14015};
  (data->simulationInfo->realParameter[495] /* arm_off[493] PARAM */) = (6.182654342264713) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14016
type: SIMPLE_ASSIGN
theta[493] = pitch * r_init[493] + arm_off[493]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14016(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14016};
  (data->simulationInfo->realParameter[1999] /* theta[493] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */)) + (data->simulationInfo->realParameter[495] /* arm_off[493] PARAM */);
  TRACE_POP
}

/*
equation index: 14017
type: SIMPLE_ASSIGN
arm_off[492] = 6.1700879716503545 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14017(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14017};
  (data->simulationInfo->realParameter[494] /* arm_off[492] PARAM */) = (6.1700879716503545) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14018
type: SIMPLE_ASSIGN
theta[492] = pitch * r_init[492] + arm_off[492]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14018(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14018};
  (data->simulationInfo->realParameter[1998] /* theta[492] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */)) + (data->simulationInfo->realParameter[494] /* arm_off[492] PARAM */);
  TRACE_POP
}

/*
equation index: 14019
type: SIMPLE_ASSIGN
arm_off[491] = 6.157521601035994 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14019};
  (data->simulationInfo->realParameter[493] /* arm_off[491] PARAM */) = (6.157521601035994) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14020
type: SIMPLE_ASSIGN
theta[491] = pitch * r_init[491] + arm_off[491]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14020(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14020};
  (data->simulationInfo->realParameter[1997] /* theta[491] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */)) + (data->simulationInfo->realParameter[493] /* arm_off[491] PARAM */);
  TRACE_POP
}

/*
equation index: 14021
type: SIMPLE_ASSIGN
arm_off[490] = 6.144955230421635 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14021};
  (data->simulationInfo->realParameter[492] /* arm_off[490] PARAM */) = (6.144955230421635) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14022
type: SIMPLE_ASSIGN
theta[490] = pitch * r_init[490] + arm_off[490]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14022(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14022};
  (data->simulationInfo->realParameter[1996] /* theta[490] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */)) + (data->simulationInfo->realParameter[492] /* arm_off[490] PARAM */);
  TRACE_POP
}

/*
equation index: 14023
type: SIMPLE_ASSIGN
arm_off[489] = 6.132388859807277 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14023};
  (data->simulationInfo->realParameter[491] /* arm_off[489] PARAM */) = (6.132388859807277) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14024
type: SIMPLE_ASSIGN
theta[489] = pitch * r_init[489] + arm_off[489]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14024(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14024};
  (data->simulationInfo->realParameter[1995] /* theta[489] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */)) + (data->simulationInfo->realParameter[491] /* arm_off[489] PARAM */);
  TRACE_POP
}

/*
equation index: 14025
type: SIMPLE_ASSIGN
arm_off[488] = 6.1198224891929165 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14025};
  (data->simulationInfo->realParameter[490] /* arm_off[488] PARAM */) = (6.1198224891929165) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14026
type: SIMPLE_ASSIGN
theta[488] = pitch * r_init[488] + arm_off[488]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14026(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14026};
  (data->simulationInfo->realParameter[1994] /* theta[488] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */)) + (data->simulationInfo->realParameter[490] /* arm_off[488] PARAM */);
  TRACE_POP
}

/*
equation index: 14027
type: SIMPLE_ASSIGN
arm_off[487] = 6.107256118578558 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14027(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14027};
  (data->simulationInfo->realParameter[489] /* arm_off[487] PARAM */) = (6.107256118578558) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14028
type: SIMPLE_ASSIGN
theta[487] = pitch * r_init[487] + arm_off[487]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14028};
  (data->simulationInfo->realParameter[1993] /* theta[487] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */)) + (data->simulationInfo->realParameter[489] /* arm_off[487] PARAM */);
  TRACE_POP
}

/*
equation index: 14029
type: SIMPLE_ASSIGN
arm_off[486] = 6.094689747964199 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14029};
  (data->simulationInfo->realParameter[488] /* arm_off[486] PARAM */) = (6.094689747964199) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14030
type: SIMPLE_ASSIGN
theta[486] = pitch * r_init[486] + arm_off[486]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14030(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14030};
  (data->simulationInfo->realParameter[1992] /* theta[486] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */)) + (data->simulationInfo->realParameter[488] /* arm_off[486] PARAM */);
  TRACE_POP
}

/*
equation index: 14031
type: SIMPLE_ASSIGN
arm_off[485] = 6.0821233773498395 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14031};
  (data->simulationInfo->realParameter[487] /* arm_off[485] PARAM */) = (6.0821233773498395) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14032
type: SIMPLE_ASSIGN
theta[485] = pitch * r_init[485] + arm_off[485]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14032(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14032};
  (data->simulationInfo->realParameter[1991] /* theta[485] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */)) + (data->simulationInfo->realParameter[487] /* arm_off[485] PARAM */);
  TRACE_POP
}

/*
equation index: 14033
type: SIMPLE_ASSIGN
arm_off[484] = 6.06955700673548 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14033(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14033};
  (data->simulationInfo->realParameter[486] /* arm_off[484] PARAM */) = (6.06955700673548) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14034
type: SIMPLE_ASSIGN
theta[484] = pitch * r_init[484] + arm_off[484]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14034(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14034};
  (data->simulationInfo->realParameter[1990] /* theta[484] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */)) + (data->simulationInfo->realParameter[486] /* arm_off[484] PARAM */);
  TRACE_POP
}

/*
equation index: 14035
type: SIMPLE_ASSIGN
arm_off[483] = 6.056990636121122 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14035};
  (data->simulationInfo->realParameter[485] /* arm_off[483] PARAM */) = (6.056990636121122) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14036
type: SIMPLE_ASSIGN
theta[483] = pitch * r_init[483] + arm_off[483]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14036(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14036};
  (data->simulationInfo->realParameter[1989] /* theta[483] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */)) + (data->simulationInfo->realParameter[485] /* arm_off[483] PARAM */);
  TRACE_POP
}

/*
equation index: 14037
type: SIMPLE_ASSIGN
arm_off[482] = 6.044424265506762 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14037};
  (data->simulationInfo->realParameter[484] /* arm_off[482] PARAM */) = (6.044424265506762) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14038
type: SIMPLE_ASSIGN
theta[482] = pitch * r_init[482] + arm_off[482]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14038(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14038};
  (data->simulationInfo->realParameter[1988] /* theta[482] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */)) + (data->simulationInfo->realParameter[484] /* arm_off[482] PARAM */);
  TRACE_POP
}

/*
equation index: 14039
type: SIMPLE_ASSIGN
arm_off[481] = 6.031857894892402 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14039};
  (data->simulationInfo->realParameter[483] /* arm_off[481] PARAM */) = (6.031857894892402) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14040
type: SIMPLE_ASSIGN
theta[481] = pitch * r_init[481] + arm_off[481]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14040(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14040};
  (data->simulationInfo->realParameter[1987] /* theta[481] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */)) + (data->simulationInfo->realParameter[483] /* arm_off[481] PARAM */);
  TRACE_POP
}

/*
equation index: 14041
type: SIMPLE_ASSIGN
arm_off[480] = 6.019291524278044 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14041};
  (data->simulationInfo->realParameter[482] /* arm_off[480] PARAM */) = (6.019291524278044) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14042
type: SIMPLE_ASSIGN
theta[480] = pitch * r_init[480] + arm_off[480]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14042(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14042};
  (data->simulationInfo->realParameter[1986] /* theta[480] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */)) + (data->simulationInfo->realParameter[482] /* arm_off[480] PARAM */);
  TRACE_POP
}

/*
equation index: 14043
type: SIMPLE_ASSIGN
arm_off[479] = 6.006725153663684 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14043(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14043};
  (data->simulationInfo->realParameter[481] /* arm_off[479] PARAM */) = (6.006725153663684) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14044
type: SIMPLE_ASSIGN
theta[479] = pitch * r_init[479] + arm_off[479]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14044};
  (data->simulationInfo->realParameter[1985] /* theta[479] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */)) + (data->simulationInfo->realParameter[481] /* arm_off[479] PARAM */);
  TRACE_POP
}

/*
equation index: 14045
type: SIMPLE_ASSIGN
arm_off[478] = 5.9941587830493255 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14045};
  (data->simulationInfo->realParameter[480] /* arm_off[478] PARAM */) = (5.9941587830493255) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14046
type: SIMPLE_ASSIGN
theta[478] = pitch * r_init[478] + arm_off[478]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14046(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14046};
  (data->simulationInfo->realParameter[1984] /* theta[478] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */)) + (data->simulationInfo->realParameter[480] /* arm_off[478] PARAM */);
  TRACE_POP
}

/*
equation index: 14047
type: SIMPLE_ASSIGN
arm_off[477] = 5.981592412434966 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14047};
  (data->simulationInfo->realParameter[479] /* arm_off[477] PARAM */) = (5.981592412434966) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14048
type: SIMPLE_ASSIGN
theta[477] = pitch * r_init[477] + arm_off[477]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14048(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14048};
  (data->simulationInfo->realParameter[1983] /* theta[477] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */)) + (data->simulationInfo->realParameter[479] /* arm_off[477] PARAM */);
  TRACE_POP
}

/*
equation index: 14049
type: SIMPLE_ASSIGN
arm_off[476] = 5.969026041820607 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14049(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14049};
  (data->simulationInfo->realParameter[478] /* arm_off[476] PARAM */) = (5.969026041820607) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14050
type: SIMPLE_ASSIGN
theta[476] = pitch * r_init[476] + arm_off[476]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14050(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14050};
  (data->simulationInfo->realParameter[1982] /* theta[476] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */)) + (data->simulationInfo->realParameter[478] /* arm_off[476] PARAM */);
  TRACE_POP
}

/*
equation index: 14051
type: SIMPLE_ASSIGN
arm_off[475] = 5.956459671206248 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14051};
  (data->simulationInfo->realParameter[477] /* arm_off[475] PARAM */) = (5.956459671206248) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14052
type: SIMPLE_ASSIGN
theta[475] = pitch * r_init[475] + arm_off[475]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14052(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14052};
  (data->simulationInfo->realParameter[1981] /* theta[475] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */)) + (data->simulationInfo->realParameter[477] /* arm_off[475] PARAM */);
  TRACE_POP
}

/*
equation index: 14053
type: SIMPLE_ASSIGN
arm_off[474] = 5.943893300591889 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14053};
  (data->simulationInfo->realParameter[476] /* arm_off[474] PARAM */) = (5.943893300591889) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14054
type: SIMPLE_ASSIGN
theta[474] = pitch * r_init[474] + arm_off[474]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14054(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14054};
  (data->simulationInfo->realParameter[1980] /* theta[474] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */)) + (data->simulationInfo->realParameter[476] /* arm_off[474] PARAM */);
  TRACE_POP
}

/*
equation index: 14055
type: SIMPLE_ASSIGN
arm_off[473] = 5.931326929977529 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14055};
  (data->simulationInfo->realParameter[475] /* arm_off[473] PARAM */) = (5.931326929977529) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14056
type: SIMPLE_ASSIGN
theta[473] = pitch * r_init[473] + arm_off[473]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14056(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14056};
  (data->simulationInfo->realParameter[1979] /* theta[473] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */)) + (data->simulationInfo->realParameter[475] /* arm_off[473] PARAM */);
  TRACE_POP
}

/*
equation index: 14057
type: SIMPLE_ASSIGN
arm_off[472] = 5.918760559363171 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14057};
  (data->simulationInfo->realParameter[474] /* arm_off[472] PARAM */) = (5.918760559363171) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14058
type: SIMPLE_ASSIGN
theta[472] = pitch * r_init[472] + arm_off[472]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14058(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14058};
  (data->simulationInfo->realParameter[1978] /* theta[472] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */)) + (data->simulationInfo->realParameter[474] /* arm_off[472] PARAM */);
  TRACE_POP
}

/*
equation index: 14059
type: SIMPLE_ASSIGN
arm_off[471] = 5.906194188748811 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14059(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14059};
  (data->simulationInfo->realParameter[473] /* arm_off[471] PARAM */) = (5.906194188748811) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14060
type: SIMPLE_ASSIGN
theta[471] = pitch * r_init[471] + arm_off[471]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14060};
  (data->simulationInfo->realParameter[1977] /* theta[471] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */)) + (data->simulationInfo->realParameter[473] /* arm_off[471] PARAM */);
  TRACE_POP
}

/*
equation index: 14061
type: SIMPLE_ASSIGN
arm_off[470] = 5.893627818134451 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14061};
  (data->simulationInfo->realParameter[472] /* arm_off[470] PARAM */) = (5.893627818134451) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14062
type: SIMPLE_ASSIGN
theta[470] = pitch * r_init[470] + arm_off[470]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14062(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14062};
  (data->simulationInfo->realParameter[1976] /* theta[470] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */)) + (data->simulationInfo->realParameter[472] /* arm_off[470] PARAM */);
  TRACE_POP
}

/*
equation index: 14063
type: SIMPLE_ASSIGN
arm_off[469] = 5.881061447520093 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14063};
  (data->simulationInfo->realParameter[471] /* arm_off[469] PARAM */) = (5.881061447520093) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14064
type: SIMPLE_ASSIGN
theta[469] = pitch * r_init[469] + arm_off[469]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14064(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14064};
  (data->simulationInfo->realParameter[1975] /* theta[469] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */)) + (data->simulationInfo->realParameter[471] /* arm_off[469] PARAM */);
  TRACE_POP
}

/*
equation index: 14065
type: SIMPLE_ASSIGN
arm_off[468] = 5.868495076905734 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14065(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14065};
  (data->simulationInfo->realParameter[470] /* arm_off[468] PARAM */) = (5.868495076905734) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14066
type: SIMPLE_ASSIGN
theta[468] = pitch * r_init[468] + arm_off[468]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14066(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14066};
  (data->simulationInfo->realParameter[1974] /* theta[468] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */)) + (data->simulationInfo->realParameter[470] /* arm_off[468] PARAM */);
  TRACE_POP
}

/*
equation index: 14067
type: SIMPLE_ASSIGN
arm_off[467] = 5.855928706291374 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14067};
  (data->simulationInfo->realParameter[469] /* arm_off[467] PARAM */) = (5.855928706291374) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14068
type: SIMPLE_ASSIGN
theta[467] = pitch * r_init[467] + arm_off[467]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14068(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14068};
  (data->simulationInfo->realParameter[1973] /* theta[467] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */)) + (data->simulationInfo->realParameter[469] /* arm_off[467] PARAM */);
  TRACE_POP
}

/*
equation index: 14069
type: SIMPLE_ASSIGN
arm_off[466] = 5.843362335677015 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14069};
  (data->simulationInfo->realParameter[468] /* arm_off[466] PARAM */) = (5.843362335677015) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14070
type: SIMPLE_ASSIGN
theta[466] = pitch * r_init[466] + arm_off[466]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14070(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14070};
  (data->simulationInfo->realParameter[1972] /* theta[466] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */)) + (data->simulationInfo->realParameter[468] /* arm_off[466] PARAM */);
  TRACE_POP
}

/*
equation index: 14071
type: SIMPLE_ASSIGN
arm_off[465] = 5.830795965062657 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14071};
  (data->simulationInfo->realParameter[467] /* arm_off[465] PARAM */) = (5.830795965062657) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14072
type: SIMPLE_ASSIGN
theta[465] = pitch * r_init[465] + arm_off[465]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14072(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14072};
  (data->simulationInfo->realParameter[1971] /* theta[465] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */)) + (data->simulationInfo->realParameter[467] /* arm_off[465] PARAM */);
  TRACE_POP
}

/*
equation index: 14073
type: SIMPLE_ASSIGN
arm_off[464] = 5.8182295944482965 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14073};
  (data->simulationInfo->realParameter[466] /* arm_off[464] PARAM */) = (5.8182295944482965) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14074
type: SIMPLE_ASSIGN
theta[464] = pitch * r_init[464] + arm_off[464]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14074(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14074};
  (data->simulationInfo->realParameter[1970] /* theta[464] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */)) + (data->simulationInfo->realParameter[466] /* arm_off[464] PARAM */);
  TRACE_POP
}

/*
equation index: 14075
type: SIMPLE_ASSIGN
arm_off[463] = 5.805663223833938 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14075(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14075};
  (data->simulationInfo->realParameter[465] /* arm_off[463] PARAM */) = (5.805663223833938) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14076
type: SIMPLE_ASSIGN
theta[463] = pitch * r_init[463] + arm_off[463]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14076};
  (data->simulationInfo->realParameter[1969] /* theta[463] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */)) + (data->simulationInfo->realParameter[465] /* arm_off[463] PARAM */);
  TRACE_POP
}

/*
equation index: 14077
type: SIMPLE_ASSIGN
arm_off[462] = 5.793096853219579 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14077};
  (data->simulationInfo->realParameter[464] /* arm_off[462] PARAM */) = (5.793096853219579) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14078
type: SIMPLE_ASSIGN
theta[462] = pitch * r_init[462] + arm_off[462]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14078(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14078};
  (data->simulationInfo->realParameter[1968] /* theta[462] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */)) + (data->simulationInfo->realParameter[464] /* arm_off[462] PARAM */);
  TRACE_POP
}

/*
equation index: 14079
type: SIMPLE_ASSIGN
arm_off[461] = 5.7805304826052195 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14079};
  (data->simulationInfo->realParameter[463] /* arm_off[461] PARAM */) = (5.7805304826052195) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14080
type: SIMPLE_ASSIGN
theta[461] = pitch * r_init[461] + arm_off[461]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14080(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14080};
  (data->simulationInfo->realParameter[1967] /* theta[461] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */)) + (data->simulationInfo->realParameter[463] /* arm_off[461] PARAM */);
  TRACE_POP
}

/*
equation index: 14081
type: SIMPLE_ASSIGN
arm_off[460] = 5.76796411199086 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14081(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14081};
  (data->simulationInfo->realParameter[462] /* arm_off[460] PARAM */) = (5.76796411199086) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14082
type: SIMPLE_ASSIGN
theta[460] = pitch * r_init[460] + arm_off[460]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14082(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14082};
  (data->simulationInfo->realParameter[1966] /* theta[460] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */)) + (data->simulationInfo->realParameter[462] /* arm_off[460] PARAM */);
  TRACE_POP
}

/*
equation index: 14083
type: SIMPLE_ASSIGN
arm_off[459] = 5.755397741376501 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14083};
  (data->simulationInfo->realParameter[461] /* arm_off[459] PARAM */) = (5.755397741376501) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14084
type: SIMPLE_ASSIGN
theta[459] = pitch * r_init[459] + arm_off[459]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14084(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14084};
  (data->simulationInfo->realParameter[1965] /* theta[459] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */)) + (data->simulationInfo->realParameter[461] /* arm_off[459] PARAM */);
  TRACE_POP
}

/*
equation index: 14085
type: SIMPLE_ASSIGN
arm_off[458] = 5.742831370762142 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14085};
  (data->simulationInfo->realParameter[460] /* arm_off[458] PARAM */) = (5.742831370762142) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14086
type: SIMPLE_ASSIGN
theta[458] = pitch * r_init[458] + arm_off[458]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14086(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14086};
  (data->simulationInfo->realParameter[1964] /* theta[458] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */)) + (data->simulationInfo->realParameter[460] /* arm_off[458] PARAM */);
  TRACE_POP
}

/*
equation index: 14087
type: SIMPLE_ASSIGN
arm_off[457] = 5.730265000147782 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14087};
  (data->simulationInfo->realParameter[459] /* arm_off[457] PARAM */) = (5.730265000147782) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14088
type: SIMPLE_ASSIGN
theta[457] = pitch * r_init[457] + arm_off[457]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14088(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14088};
  (data->simulationInfo->realParameter[1963] /* theta[457] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */)) + (data->simulationInfo->realParameter[459] /* arm_off[457] PARAM */);
  TRACE_POP
}

/*
equation index: 14089
type: SIMPLE_ASSIGN
arm_off[456] = 5.717698629533424 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14089};
  (data->simulationInfo->realParameter[458] /* arm_off[456] PARAM */) = (5.717698629533424) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14090
type: SIMPLE_ASSIGN
theta[456] = pitch * r_init[456] + arm_off[456]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14090(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14090};
  (data->simulationInfo->realParameter[1962] /* theta[456] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */)) + (data->simulationInfo->realParameter[458] /* arm_off[456] PARAM */);
  TRACE_POP
}

/*
equation index: 14091
type: SIMPLE_ASSIGN
arm_off[455] = 5.705132258919064 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14091(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14091};
  (data->simulationInfo->realParameter[457] /* arm_off[455] PARAM */) = (5.705132258919064) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14092
type: SIMPLE_ASSIGN
theta[455] = pitch * r_init[455] + arm_off[455]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14092};
  (data->simulationInfo->realParameter[1961] /* theta[455] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */)) + (data->simulationInfo->realParameter[457] /* arm_off[455] PARAM */);
  TRACE_POP
}

/*
equation index: 14093
type: SIMPLE_ASSIGN
arm_off[454] = 5.6925658883047054 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14093};
  (data->simulationInfo->realParameter[456] /* arm_off[454] PARAM */) = (5.6925658883047054) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14094
type: SIMPLE_ASSIGN
theta[454] = pitch * r_init[454] + arm_off[454]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14094(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14094};
  (data->simulationInfo->realParameter[1960] /* theta[454] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */)) + (data->simulationInfo->realParameter[456] /* arm_off[454] PARAM */);
  TRACE_POP
}

/*
equation index: 14095
type: SIMPLE_ASSIGN
arm_off[453] = 5.679999517690346 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14095};
  (data->simulationInfo->realParameter[455] /* arm_off[453] PARAM */) = (5.679999517690346) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14096
type: SIMPLE_ASSIGN
theta[453] = pitch * r_init[453] + arm_off[453]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14096(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14096};
  (data->simulationInfo->realParameter[1959] /* theta[453] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */)) + (data->simulationInfo->realParameter[455] /* arm_off[453] PARAM */);
  TRACE_POP
}

/*
equation index: 14097
type: SIMPLE_ASSIGN
arm_off[452] = 5.667433147075987 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14097(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14097};
  (data->simulationInfo->realParameter[454] /* arm_off[452] PARAM */) = (5.667433147075987) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14098
type: SIMPLE_ASSIGN
theta[452] = pitch * r_init[452] + arm_off[452]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14098(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14098};
  (data->simulationInfo->realParameter[1958] /* theta[452] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */)) + (data->simulationInfo->realParameter[454] /* arm_off[452] PARAM */);
  TRACE_POP
}

/*
equation index: 14099
type: SIMPLE_ASSIGN
arm_off[451] = 5.654866776461628 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14099};
  (data->simulationInfo->realParameter[453] /* arm_off[451] PARAM */) = (5.654866776461628) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14100
type: SIMPLE_ASSIGN
theta[451] = pitch * r_init[451] + arm_off[451]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14100};
  (data->simulationInfo->realParameter[1957] /* theta[451] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */)) + (data->simulationInfo->realParameter[453] /* arm_off[451] PARAM */);
  TRACE_POP
}

/*
equation index: 14101
type: SIMPLE_ASSIGN
arm_off[450] = 5.642300405847269 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14101};
  (data->simulationInfo->realParameter[452] /* arm_off[450] PARAM */) = (5.642300405847269) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14102
type: SIMPLE_ASSIGN
theta[450] = pitch * r_init[450] + arm_off[450]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14102(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14102};
  (data->simulationInfo->realParameter[1956] /* theta[450] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */)) + (data->simulationInfo->realParameter[452] /* arm_off[450] PARAM */);
  TRACE_POP
}

/*
equation index: 14103
type: SIMPLE_ASSIGN
arm_off[449] = 5.629734035232909 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14103};
  (data->simulationInfo->realParameter[451] /* arm_off[449] PARAM */) = (5.629734035232909) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14104
type: SIMPLE_ASSIGN
theta[449] = pitch * r_init[449] + arm_off[449]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14104(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14104};
  (data->simulationInfo->realParameter[1955] /* theta[449] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */)) + (data->simulationInfo->realParameter[451] /* arm_off[449] PARAM */);
  TRACE_POP
}

/*
equation index: 14105
type: SIMPLE_ASSIGN
arm_off[448] = 5.61716766461855 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14105};
  (data->simulationInfo->realParameter[450] /* arm_off[448] PARAM */) = (5.61716766461855) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14106
type: SIMPLE_ASSIGN
theta[448] = pitch * r_init[448] + arm_off[448]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14106(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14106};
  (data->simulationInfo->realParameter[1954] /* theta[448] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */)) + (data->simulationInfo->realParameter[450] /* arm_off[448] PARAM */);
  TRACE_POP
}

/*
equation index: 14107
type: SIMPLE_ASSIGN
arm_off[447] = 5.604601294004191 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14107};
  (data->simulationInfo->realParameter[449] /* arm_off[447] PARAM */) = (5.604601294004191) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14108
type: SIMPLE_ASSIGN
theta[447] = pitch * r_init[447] + arm_off[447]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14108};
  (data->simulationInfo->realParameter[1953] /* theta[447] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */)) + (data->simulationInfo->realParameter[449] /* arm_off[447] PARAM */);
  TRACE_POP
}

/*
equation index: 14109
type: SIMPLE_ASSIGN
arm_off[446] = 5.592034923389831 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14109};
  (data->simulationInfo->realParameter[448] /* arm_off[446] PARAM */) = (5.592034923389831) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14110
type: SIMPLE_ASSIGN
theta[446] = pitch * r_init[446] + arm_off[446]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14110(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14110};
  (data->simulationInfo->realParameter[1952] /* theta[446] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */)) + (data->simulationInfo->realParameter[448] /* arm_off[446] PARAM */);
  TRACE_POP
}

/*
equation index: 14111
type: SIMPLE_ASSIGN
arm_off[445] = 5.579468552775473 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14111};
  (data->simulationInfo->realParameter[447] /* arm_off[445] PARAM */) = (5.579468552775473) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14112
type: SIMPLE_ASSIGN
theta[445] = pitch * r_init[445] + arm_off[445]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14112(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14112};
  (data->simulationInfo->realParameter[1951] /* theta[445] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */)) + (data->simulationInfo->realParameter[447] /* arm_off[445] PARAM */);
  TRACE_POP
}

/*
equation index: 14113
type: SIMPLE_ASSIGN
arm_off[444] = 5.5669021821611135 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14113};
  (data->simulationInfo->realParameter[446] /* arm_off[444] PARAM */) = (5.5669021821611135) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14114
type: SIMPLE_ASSIGN
theta[444] = pitch * r_init[444] + arm_off[444]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14114(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14114};
  (data->simulationInfo->realParameter[1950] /* theta[444] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */)) + (data->simulationInfo->realParameter[446] /* arm_off[444] PARAM */);
  TRACE_POP
}

/*
equation index: 14115
type: SIMPLE_ASSIGN
arm_off[443] = 5.554335811546754 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14115};
  (data->simulationInfo->realParameter[445] /* arm_off[443] PARAM */) = (5.554335811546754) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14116
type: SIMPLE_ASSIGN
theta[443] = pitch * r_init[443] + arm_off[443]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14116};
  (data->simulationInfo->realParameter[1949] /* theta[443] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */)) + (data->simulationInfo->realParameter[445] /* arm_off[443] PARAM */);
  TRACE_POP
}

/*
equation index: 14117
type: SIMPLE_ASSIGN
arm_off[442] = 5.541769440932395 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14117};
  (data->simulationInfo->realParameter[444] /* arm_off[442] PARAM */) = (5.541769440932395) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14118
type: SIMPLE_ASSIGN
theta[442] = pitch * r_init[442] + arm_off[442]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14118(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14118};
  (data->simulationInfo->realParameter[1948] /* theta[442] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */)) + (data->simulationInfo->realParameter[444] /* arm_off[442] PARAM */);
  TRACE_POP
}

/*
equation index: 14119
type: SIMPLE_ASSIGN
arm_off[441] = 5.529203070318037 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14119};
  (data->simulationInfo->realParameter[443] /* arm_off[441] PARAM */) = (5.529203070318037) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14120
type: SIMPLE_ASSIGN
theta[441] = pitch * r_init[441] + arm_off[441]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14120(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14120};
  (data->simulationInfo->realParameter[1947] /* theta[441] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */)) + (data->simulationInfo->realParameter[443] /* arm_off[441] PARAM */);
  TRACE_POP
}

/*
equation index: 14121
type: SIMPLE_ASSIGN
arm_off[440] = 5.516636699703676 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14121};
  (data->simulationInfo->realParameter[442] /* arm_off[440] PARAM */) = (5.516636699703676) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14122
type: SIMPLE_ASSIGN
theta[440] = pitch * r_init[440] + arm_off[440]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14122(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14122};
  (data->simulationInfo->realParameter[1946] /* theta[440] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */)) + (data->simulationInfo->realParameter[442] /* arm_off[440] PARAM */);
  TRACE_POP
}

/*
equation index: 14123
type: SIMPLE_ASSIGN
arm_off[439] = 5.504070329089318 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14123};
  (data->simulationInfo->realParameter[441] /* arm_off[439] PARAM */) = (5.504070329089318) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14124
type: SIMPLE_ASSIGN
theta[439] = pitch * r_init[439] + arm_off[439]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14124};
  (data->simulationInfo->realParameter[1945] /* theta[439] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */)) + (data->simulationInfo->realParameter[441] /* arm_off[439] PARAM */);
  TRACE_POP
}

/*
equation index: 14125
type: SIMPLE_ASSIGN
arm_off[438] = 5.491503958474959 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14125};
  (data->simulationInfo->realParameter[440] /* arm_off[438] PARAM */) = (5.491503958474959) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14126
type: SIMPLE_ASSIGN
theta[438] = pitch * r_init[438] + arm_off[438]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14126(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14126};
  (data->simulationInfo->realParameter[1944] /* theta[438] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */)) + (data->simulationInfo->realParameter[440] /* arm_off[438] PARAM */);
  TRACE_POP
}

/*
equation index: 14127
type: SIMPLE_ASSIGN
arm_off[437] = 5.478937587860599 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14127};
  (data->simulationInfo->realParameter[439] /* arm_off[437] PARAM */) = (5.478937587860599) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14128
type: SIMPLE_ASSIGN
theta[437] = pitch * r_init[437] + arm_off[437]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14128(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14128};
  (data->simulationInfo->realParameter[1943] /* theta[437] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */)) + (data->simulationInfo->realParameter[439] /* arm_off[437] PARAM */);
  TRACE_POP
}

/*
equation index: 14129
type: SIMPLE_ASSIGN
arm_off[436] = 5.46637121724624 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14129};
  (data->simulationInfo->realParameter[438] /* arm_off[436] PARAM */) = (5.46637121724624) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14130
type: SIMPLE_ASSIGN
theta[436] = pitch * r_init[436] + arm_off[436]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14130(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14130};
  (data->simulationInfo->realParameter[1942] /* theta[436] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */)) + (data->simulationInfo->realParameter[438] /* arm_off[436] PARAM */);
  TRACE_POP
}

/*
equation index: 14131
type: SIMPLE_ASSIGN
arm_off[435] = 5.453804846631881 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14131};
  (data->simulationInfo->realParameter[437] /* arm_off[435] PARAM */) = (5.453804846631881) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14132
type: SIMPLE_ASSIGN
theta[435] = pitch * r_init[435] + arm_off[435]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14132};
  (data->simulationInfo->realParameter[1941] /* theta[435] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */)) + (data->simulationInfo->realParameter[437] /* arm_off[435] PARAM */);
  TRACE_POP
}

/*
equation index: 14133
type: SIMPLE_ASSIGN
arm_off[434] = 5.441238476017522 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14133};
  (data->simulationInfo->realParameter[436] /* arm_off[434] PARAM */) = (5.441238476017522) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14134
type: SIMPLE_ASSIGN
theta[434] = pitch * r_init[434] + arm_off[434]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14134(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14134};
  (data->simulationInfo->realParameter[1940] /* theta[434] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */)) + (data->simulationInfo->realParameter[436] /* arm_off[434] PARAM */);
  TRACE_POP
}

/*
equation index: 14135
type: SIMPLE_ASSIGN
arm_off[433] = 5.428672105403162 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14135};
  (data->simulationInfo->realParameter[435] /* arm_off[433] PARAM */) = (5.428672105403162) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14136
type: SIMPLE_ASSIGN
theta[433] = pitch * r_init[433] + arm_off[433]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14136(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14136};
  (data->simulationInfo->realParameter[1939] /* theta[433] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */)) + (data->simulationInfo->realParameter[435] /* arm_off[433] PARAM */);
  TRACE_POP
}

/*
equation index: 14137
type: SIMPLE_ASSIGN
arm_off[432] = 5.416105734788804 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14137};
  (data->simulationInfo->realParameter[434] /* arm_off[432] PARAM */) = (5.416105734788804) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14138
type: SIMPLE_ASSIGN
theta[432] = pitch * r_init[432] + arm_off[432]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14138(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14138};
  (data->simulationInfo->realParameter[1938] /* theta[432] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */)) + (data->simulationInfo->realParameter[434] /* arm_off[432] PARAM */);
  TRACE_POP
}

/*
equation index: 14139
type: SIMPLE_ASSIGN
arm_off[431] = 5.403539364174444 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14139(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14139};
  (data->simulationInfo->realParameter[433] /* arm_off[431] PARAM */) = (5.403539364174444) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14140
type: SIMPLE_ASSIGN
theta[431] = pitch * r_init[431] + arm_off[431]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14140};
  (data->simulationInfo->realParameter[1937] /* theta[431] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */)) + (data->simulationInfo->realParameter[433] /* arm_off[431] PARAM */);
  TRACE_POP
}

/*
equation index: 14141
type: SIMPLE_ASSIGN
arm_off[430] = 5.390972993560085 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14141};
  (data->simulationInfo->realParameter[432] /* arm_off[430] PARAM */) = (5.390972993560085) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14142
type: SIMPLE_ASSIGN
theta[430] = pitch * r_init[430] + arm_off[430]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14142(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14142};
  (data->simulationInfo->realParameter[1936] /* theta[430] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */)) + (data->simulationInfo->realParameter[432] /* arm_off[430] PARAM */);
  TRACE_POP
}

/*
equation index: 14143
type: SIMPLE_ASSIGN
arm_off[429] = 5.378406622945726 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14143};
  (data->simulationInfo->realParameter[431] /* arm_off[429] PARAM */) = (5.378406622945726) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14144
type: SIMPLE_ASSIGN
theta[429] = pitch * r_init[429] + arm_off[429]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14144(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14144};
  (data->simulationInfo->realParameter[1935] /* theta[429] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */)) + (data->simulationInfo->realParameter[431] /* arm_off[429] PARAM */);
  TRACE_POP
}

/*
equation index: 14145
type: SIMPLE_ASSIGN
arm_off[428] = 5.365840252331367 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14145};
  (data->simulationInfo->realParameter[430] /* arm_off[428] PARAM */) = (5.365840252331367) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14146
type: SIMPLE_ASSIGN
theta[428] = pitch * r_init[428] + arm_off[428]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14146(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14146};
  (data->simulationInfo->realParameter[1934] /* theta[428] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */)) + (data->simulationInfo->realParameter[430] /* arm_off[428] PARAM */);
  TRACE_POP
}

/*
equation index: 14147
type: SIMPLE_ASSIGN
arm_off[427] = 5.353273881717008 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14147};
  (data->simulationInfo->realParameter[429] /* arm_off[427] PARAM */) = (5.353273881717008) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14148
type: SIMPLE_ASSIGN
theta[427] = pitch * r_init[427] + arm_off[427]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14148};
  (data->simulationInfo->realParameter[1933] /* theta[427] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */)) + (data->simulationInfo->realParameter[429] /* arm_off[427] PARAM */);
  TRACE_POP
}

/*
equation index: 14149
type: SIMPLE_ASSIGN
arm_off[426] = 5.340707511102648 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14149};
  (data->simulationInfo->realParameter[428] /* arm_off[426] PARAM */) = (5.340707511102648) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14150
type: SIMPLE_ASSIGN
theta[426] = pitch * r_init[426] + arm_off[426]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14150(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14150};
  (data->simulationInfo->realParameter[1932] /* theta[426] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */)) + (data->simulationInfo->realParameter[428] /* arm_off[426] PARAM */);
  TRACE_POP
}

/*
equation index: 14151
type: SIMPLE_ASSIGN
arm_off[425] = 5.328141140488289 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14151};
  (data->simulationInfo->realParameter[427] /* arm_off[425] PARAM */) = (5.328141140488289) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14152
type: SIMPLE_ASSIGN
theta[425] = pitch * r_init[425] + arm_off[425]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14152(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14152};
  (data->simulationInfo->realParameter[1931] /* theta[425] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */)) + (data->simulationInfo->realParameter[427] /* arm_off[425] PARAM */);
  TRACE_POP
}

/*
equation index: 14153
type: SIMPLE_ASSIGN
arm_off[424] = 5.31557476987393 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14153};
  (data->simulationInfo->realParameter[426] /* arm_off[424] PARAM */) = (5.31557476987393) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14154
type: SIMPLE_ASSIGN
theta[424] = pitch * r_init[424] + arm_off[424]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14154(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14154};
  (data->simulationInfo->realParameter[1930] /* theta[424] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */)) + (data->simulationInfo->realParameter[426] /* arm_off[424] PARAM */);
  TRACE_POP
}

/*
equation index: 14155
type: SIMPLE_ASSIGN
arm_off[423] = 5.303008399259571 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14155};
  (data->simulationInfo->realParameter[425] /* arm_off[423] PARAM */) = (5.303008399259571) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14156
type: SIMPLE_ASSIGN
theta[423] = pitch * r_init[423] + arm_off[423]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14156};
  (data->simulationInfo->realParameter[1929] /* theta[423] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */)) + (data->simulationInfo->realParameter[425] /* arm_off[423] PARAM */);
  TRACE_POP
}

/*
equation index: 14157
type: SIMPLE_ASSIGN
arm_off[422] = 5.290442028645211 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14157};
  (data->simulationInfo->realParameter[424] /* arm_off[422] PARAM */) = (5.290442028645211) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14158
type: SIMPLE_ASSIGN
theta[422] = pitch * r_init[422] + arm_off[422]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14158(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14158};
  (data->simulationInfo->realParameter[1928] /* theta[422] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */)) + (data->simulationInfo->realParameter[424] /* arm_off[422] PARAM */);
  TRACE_POP
}

/*
equation index: 14159
type: SIMPLE_ASSIGN
arm_off[421] = 5.277875658030853 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14159};
  (data->simulationInfo->realParameter[423] /* arm_off[421] PARAM */) = (5.277875658030853) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14160
type: SIMPLE_ASSIGN
theta[421] = pitch * r_init[421] + arm_off[421]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14160(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14160};
  (data->simulationInfo->realParameter[1927] /* theta[421] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */)) + (data->simulationInfo->realParameter[423] /* arm_off[421] PARAM */);
  TRACE_POP
}

/*
equation index: 14161
type: SIMPLE_ASSIGN
arm_off[420] = 5.2653092874164935 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14161};
  (data->simulationInfo->realParameter[422] /* arm_off[420] PARAM */) = (5.2653092874164935) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14162
type: SIMPLE_ASSIGN
theta[420] = pitch * r_init[420] + arm_off[420]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14162(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14162};
  (data->simulationInfo->realParameter[1926] /* theta[420] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */)) + (data->simulationInfo->realParameter[422] /* arm_off[420] PARAM */);
  TRACE_POP
}

/*
equation index: 14163
type: SIMPLE_ASSIGN
arm_off[419] = 5.252742916802134 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14163};
  (data->simulationInfo->realParameter[421] /* arm_off[419] PARAM */) = (5.252742916802134) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14164
type: SIMPLE_ASSIGN
theta[419] = pitch * r_init[419] + arm_off[419]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14164};
  (data->simulationInfo->realParameter[1925] /* theta[419] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */)) + (data->simulationInfo->realParameter[421] /* arm_off[419] PARAM */);
  TRACE_POP
}

/*
equation index: 14165
type: SIMPLE_ASSIGN
arm_off[418] = 5.240176546187775 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14165};
  (data->simulationInfo->realParameter[420] /* arm_off[418] PARAM */) = (5.240176546187775) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14166
type: SIMPLE_ASSIGN
theta[418] = pitch * r_init[418] + arm_off[418]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14166(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14166};
  (data->simulationInfo->realParameter[1924] /* theta[418] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */)) + (data->simulationInfo->realParameter[420] /* arm_off[418] PARAM */);
  TRACE_POP
}

/*
equation index: 14167
type: SIMPLE_ASSIGN
arm_off[417] = 5.227610175573417 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14167};
  (data->simulationInfo->realParameter[419] /* arm_off[417] PARAM */) = (5.227610175573417) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14168
type: SIMPLE_ASSIGN
theta[417] = pitch * r_init[417] + arm_off[417]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14168(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14168};
  (data->simulationInfo->realParameter[1923] /* theta[417] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */)) + (data->simulationInfo->realParameter[419] /* arm_off[417] PARAM */);
  TRACE_POP
}

/*
equation index: 14169
type: SIMPLE_ASSIGN
arm_off[416] = 5.215043804959056 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14169};
  (data->simulationInfo->realParameter[418] /* arm_off[416] PARAM */) = (5.215043804959056) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14170
type: SIMPLE_ASSIGN
theta[416] = pitch * r_init[416] + arm_off[416]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14170(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14170};
  (data->simulationInfo->realParameter[1922] /* theta[416] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */)) + (data->simulationInfo->realParameter[418] /* arm_off[416] PARAM */);
  TRACE_POP
}

/*
equation index: 14171
type: SIMPLE_ASSIGN
arm_off[415] = 5.202477434344697 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14171};
  (data->simulationInfo->realParameter[417] /* arm_off[415] PARAM */) = (5.202477434344697) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14172
type: SIMPLE_ASSIGN
theta[415] = pitch * r_init[415] + arm_off[415]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14172};
  (data->simulationInfo->realParameter[1921] /* theta[415] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */)) + (data->simulationInfo->realParameter[417] /* arm_off[415] PARAM */);
  TRACE_POP
}

/*
equation index: 14173
type: SIMPLE_ASSIGN
arm_off[414] = 5.189911063730339 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14173};
  (data->simulationInfo->realParameter[416] /* arm_off[414] PARAM */) = (5.189911063730339) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14174
type: SIMPLE_ASSIGN
theta[414] = pitch * r_init[414] + arm_off[414]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14174(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14174};
  (data->simulationInfo->realParameter[1920] /* theta[414] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */)) + (data->simulationInfo->realParameter[416] /* arm_off[414] PARAM */);
  TRACE_POP
}

/*
equation index: 14175
type: SIMPLE_ASSIGN
arm_off[413] = 5.177344693115979 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14175};
  (data->simulationInfo->realParameter[415] /* arm_off[413] PARAM */) = (5.177344693115979) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14176
type: SIMPLE_ASSIGN
theta[413] = pitch * r_init[413] + arm_off[413]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14176(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14176};
  (data->simulationInfo->realParameter[1919] /* theta[413] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */)) + (data->simulationInfo->realParameter[415] /* arm_off[413] PARAM */);
  TRACE_POP
}

/*
equation index: 14177
type: SIMPLE_ASSIGN
arm_off[412] = 5.16477832250162 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14177};
  (data->simulationInfo->realParameter[414] /* arm_off[412] PARAM */) = (5.16477832250162) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14178
type: SIMPLE_ASSIGN
theta[412] = pitch * r_init[412] + arm_off[412]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14178(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14178};
  (data->simulationInfo->realParameter[1918] /* theta[412] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */)) + (data->simulationInfo->realParameter[414] /* arm_off[412] PARAM */);
  TRACE_POP
}

/*
equation index: 14179
type: SIMPLE_ASSIGN
arm_off[411] = 5.152211951887261 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14179};
  (data->simulationInfo->realParameter[413] /* arm_off[411] PARAM */) = (5.152211951887261) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14180
type: SIMPLE_ASSIGN
theta[411] = pitch * r_init[411] + arm_off[411]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14180};
  (data->simulationInfo->realParameter[1917] /* theta[411] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */)) + (data->simulationInfo->realParameter[413] /* arm_off[411] PARAM */);
  TRACE_POP
}

/*
equation index: 14181
type: SIMPLE_ASSIGN
arm_off[410] = 5.139645581272902 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14181};
  (data->simulationInfo->realParameter[412] /* arm_off[410] PARAM */) = (5.139645581272902) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14182
type: SIMPLE_ASSIGN
theta[410] = pitch * r_init[410] + arm_off[410]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14182(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14182};
  (data->simulationInfo->realParameter[1916] /* theta[410] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */)) + (data->simulationInfo->realParameter[412] /* arm_off[410] PARAM */);
  TRACE_POP
}

/*
equation index: 14183
type: SIMPLE_ASSIGN
arm_off[409] = 5.127079210658542 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14183};
  (data->simulationInfo->realParameter[411] /* arm_off[409] PARAM */) = (5.127079210658542) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14184
type: SIMPLE_ASSIGN
theta[409] = pitch * r_init[409] + arm_off[409]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14184(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14184};
  (data->simulationInfo->realParameter[1915] /* theta[409] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */)) + (data->simulationInfo->realParameter[411] /* arm_off[409] PARAM */);
  TRACE_POP
}

/*
equation index: 14185
type: SIMPLE_ASSIGN
arm_off[408] = 5.114512840044184 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14185};
  (data->simulationInfo->realParameter[410] /* arm_off[408] PARAM */) = (5.114512840044184) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14186
type: SIMPLE_ASSIGN
theta[408] = pitch * r_init[408] + arm_off[408]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14186(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14186};
  (data->simulationInfo->realParameter[1914] /* theta[408] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */)) + (data->simulationInfo->realParameter[410] /* arm_off[408] PARAM */);
  TRACE_POP
}

/*
equation index: 14187
type: SIMPLE_ASSIGN
arm_off[407] = 5.101946469429824 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14187};
  (data->simulationInfo->realParameter[409] /* arm_off[407] PARAM */) = (5.101946469429824) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14188
type: SIMPLE_ASSIGN
theta[407] = pitch * r_init[407] + arm_off[407]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14188};
  (data->simulationInfo->realParameter[1913] /* theta[407] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */)) + (data->simulationInfo->realParameter[409] /* arm_off[407] PARAM */);
  TRACE_POP
}

/*
equation index: 14189
type: SIMPLE_ASSIGN
arm_off[406] = 5.0893800988154645 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14189};
  (data->simulationInfo->realParameter[408] /* arm_off[406] PARAM */) = (5.0893800988154645) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14190
type: SIMPLE_ASSIGN
theta[406] = pitch * r_init[406] + arm_off[406]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14190(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14190};
  (data->simulationInfo->realParameter[1912] /* theta[406] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */)) + (data->simulationInfo->realParameter[408] /* arm_off[406] PARAM */);
  TRACE_POP
}

/*
equation index: 14191
type: SIMPLE_ASSIGN
arm_off[405] = 5.076813728201106 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14191};
  (data->simulationInfo->realParameter[407] /* arm_off[405] PARAM */) = (5.076813728201106) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14192
type: SIMPLE_ASSIGN
theta[405] = pitch * r_init[405] + arm_off[405]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14192(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14192};
  (data->simulationInfo->realParameter[1911] /* theta[405] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */)) + (data->simulationInfo->realParameter[407] /* arm_off[405] PARAM */);
  TRACE_POP
}

/*
equation index: 14193
type: SIMPLE_ASSIGN
arm_off[404] = 5.064247357586746 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14193};
  (data->simulationInfo->realParameter[406] /* arm_off[404] PARAM */) = (5.064247357586746) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14194
type: SIMPLE_ASSIGN
theta[404] = pitch * r_init[404] + arm_off[404]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14194(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14194};
  (data->simulationInfo->realParameter[1910] /* theta[404] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */)) + (data->simulationInfo->realParameter[406] /* arm_off[404] PARAM */);
  TRACE_POP
}

/*
equation index: 14195
type: SIMPLE_ASSIGN
arm_off[403] = 5.0516809869723875 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14195};
  (data->simulationInfo->realParameter[405] /* arm_off[403] PARAM */) = (5.0516809869723875) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14196
type: SIMPLE_ASSIGN
theta[403] = pitch * r_init[403] + arm_off[403]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14196};
  (data->simulationInfo->realParameter[1909] /* theta[403] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */)) + (data->simulationInfo->realParameter[405] /* arm_off[403] PARAM */);
  TRACE_POP
}

/*
equation index: 14197
type: SIMPLE_ASSIGN
arm_off[402] = 5.039114616358028 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14197};
  (data->simulationInfo->realParameter[404] /* arm_off[402] PARAM */) = (5.039114616358028) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14198
type: SIMPLE_ASSIGN
theta[402] = pitch * r_init[402] + arm_off[402]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14198(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14198};
  (data->simulationInfo->realParameter[1908] /* theta[402] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */)) + (data->simulationInfo->realParameter[404] /* arm_off[402] PARAM */);
  TRACE_POP
}

/*
equation index: 14199
type: SIMPLE_ASSIGN
arm_off[401] = 5.026548245743669 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14199};
  (data->simulationInfo->realParameter[403] /* arm_off[401] PARAM */) = (5.026548245743669) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14200
type: SIMPLE_ASSIGN
theta[401] = pitch * r_init[401] + arm_off[401]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14200(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14200};
  (data->simulationInfo->realParameter[1907] /* theta[401] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */)) + (data->simulationInfo->realParameter[403] /* arm_off[401] PARAM */);
  TRACE_POP
}

/*
equation index: 14201
type: SIMPLE_ASSIGN
arm_off[400] = 5.01398187512931 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14201};
  (data->simulationInfo->realParameter[402] /* arm_off[400] PARAM */) = (5.01398187512931) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14202
type: SIMPLE_ASSIGN
theta[400] = pitch * r_init[400] + arm_off[400]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14202(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14202};
  (data->simulationInfo->realParameter[1906] /* theta[400] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */)) + (data->simulationInfo->realParameter[402] /* arm_off[400] PARAM */);
  TRACE_POP
}

/*
equation index: 14203
type: SIMPLE_ASSIGN
arm_off[399] = 5.001415504514951 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14203};
  (data->simulationInfo->realParameter[401] /* arm_off[399] PARAM */) = (5.001415504514951) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14204
type: SIMPLE_ASSIGN
theta[399] = pitch * r_init[399] + arm_off[399]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14204};
  (data->simulationInfo->realParameter[1905] /* theta[399] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */)) + (data->simulationInfo->realParameter[401] /* arm_off[399] PARAM */);
  TRACE_POP
}

/*
equation index: 14205
type: SIMPLE_ASSIGN
arm_off[398] = 4.988849133900591 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14205};
  (data->simulationInfo->realParameter[400] /* arm_off[398] PARAM */) = (4.988849133900591) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14206
type: SIMPLE_ASSIGN
theta[398] = pitch * r_init[398] + arm_off[398]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14206(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14206};
  (data->simulationInfo->realParameter[1904] /* theta[398] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */)) + (data->simulationInfo->realParameter[400] /* arm_off[398] PARAM */);
  TRACE_POP
}

/*
equation index: 14207
type: SIMPLE_ASSIGN
arm_off[397] = 4.976282763286233 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14207};
  (data->simulationInfo->realParameter[399] /* arm_off[397] PARAM */) = (4.976282763286233) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14208
type: SIMPLE_ASSIGN
theta[397] = pitch * r_init[397] + arm_off[397]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14208(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14208};
  (data->simulationInfo->realParameter[1903] /* theta[397] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */)) + (data->simulationInfo->realParameter[399] /* arm_off[397] PARAM */);
  TRACE_POP
}

/*
equation index: 14209
type: SIMPLE_ASSIGN
arm_off[396] = 4.9637163926718735 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14209};
  (data->simulationInfo->realParameter[398] /* arm_off[396] PARAM */) = (4.9637163926718735) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14210
type: SIMPLE_ASSIGN
theta[396] = pitch * r_init[396] + arm_off[396]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14210(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14210};
  (data->simulationInfo->realParameter[1902] /* theta[396] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */)) + (data->simulationInfo->realParameter[398] /* arm_off[396] PARAM */);
  TRACE_POP
}

/*
equation index: 14211
type: SIMPLE_ASSIGN
arm_off[395] = 4.951150022057513 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14211};
  (data->simulationInfo->realParameter[397] /* arm_off[395] PARAM */) = (4.951150022057513) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14212
type: SIMPLE_ASSIGN
theta[395] = pitch * r_init[395] + arm_off[395]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14212};
  (data->simulationInfo->realParameter[1901] /* theta[395] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */)) + (data->simulationInfo->realParameter[397] /* arm_off[395] PARAM */);
  TRACE_POP
}

/*
equation index: 14213
type: SIMPLE_ASSIGN
arm_off[394] = 4.938583651443155 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14213};
  (data->simulationInfo->realParameter[396] /* arm_off[394] PARAM */) = (4.938583651443155) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14214
type: SIMPLE_ASSIGN
theta[394] = pitch * r_init[394] + arm_off[394]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14214(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14214};
  (data->simulationInfo->realParameter[1900] /* theta[394] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */)) + (data->simulationInfo->realParameter[396] /* arm_off[394] PARAM */);
  TRACE_POP
}

/*
equation index: 14215
type: SIMPLE_ASSIGN
arm_off[393] = 4.926017280828796 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14215};
  (data->simulationInfo->realParameter[395] /* arm_off[393] PARAM */) = (4.926017280828796) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14216
type: SIMPLE_ASSIGN
theta[393] = pitch * r_init[393] + arm_off[393]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14216(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14216};
  (data->simulationInfo->realParameter[1899] /* theta[393] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */)) + (data->simulationInfo->realParameter[395] /* arm_off[393] PARAM */);
  TRACE_POP
}

/*
equation index: 14217
type: SIMPLE_ASSIGN
arm_off[392] = 4.913450910214436 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14217};
  (data->simulationInfo->realParameter[394] /* arm_off[392] PARAM */) = (4.913450910214436) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14218
type: SIMPLE_ASSIGN
theta[392] = pitch * r_init[392] + arm_off[392]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14218(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14218};
  (data->simulationInfo->realParameter[1898] /* theta[392] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */)) + (data->simulationInfo->realParameter[394] /* arm_off[392] PARAM */);
  TRACE_POP
}

/*
equation index: 14219
type: SIMPLE_ASSIGN
arm_off[391] = 4.900884539600077 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14219};
  (data->simulationInfo->realParameter[393] /* arm_off[391] PARAM */) = (4.900884539600077) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14220
type: SIMPLE_ASSIGN
theta[391] = pitch * r_init[391] + arm_off[391]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14220};
  (data->simulationInfo->realParameter[1897] /* theta[391] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */)) + (data->simulationInfo->realParameter[393] /* arm_off[391] PARAM */);
  TRACE_POP
}

/*
equation index: 14221
type: SIMPLE_ASSIGN
arm_off[390] = 4.888318168985719 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14221};
  (data->simulationInfo->realParameter[392] /* arm_off[390] PARAM */) = (4.888318168985719) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14222
type: SIMPLE_ASSIGN
theta[390] = pitch * r_init[390] + arm_off[390]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14222(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14222};
  (data->simulationInfo->realParameter[1896] /* theta[390] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */)) + (data->simulationInfo->realParameter[392] /* arm_off[390] PARAM */);
  TRACE_POP
}

/*
equation index: 14223
type: SIMPLE_ASSIGN
arm_off[389] = 4.8757517983713585 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14223};
  (data->simulationInfo->realParameter[391] /* arm_off[389] PARAM */) = (4.8757517983713585) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14224
type: SIMPLE_ASSIGN
theta[389] = pitch * r_init[389] + arm_off[389]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14224(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14224};
  (data->simulationInfo->realParameter[1895] /* theta[389] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */)) + (data->simulationInfo->realParameter[391] /* arm_off[389] PARAM */);
  TRACE_POP
}

/*
equation index: 14225
type: SIMPLE_ASSIGN
arm_off[388] = 4.863185427757 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14225};
  (data->simulationInfo->realParameter[390] /* arm_off[388] PARAM */) = (4.863185427757) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14226
type: SIMPLE_ASSIGN
theta[388] = pitch * r_init[388] + arm_off[388]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14226(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14226};
  (data->simulationInfo->realParameter[1894] /* theta[388] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */)) + (data->simulationInfo->realParameter[390] /* arm_off[388] PARAM */);
  TRACE_POP
}

/*
equation index: 14227
type: SIMPLE_ASSIGN
arm_off[387] = 4.850619057142641 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14227};
  (data->simulationInfo->realParameter[389] /* arm_off[387] PARAM */) = (4.850619057142641) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14228
type: SIMPLE_ASSIGN
theta[387] = pitch * r_init[387] + arm_off[387]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14228};
  (data->simulationInfo->realParameter[1893] /* theta[387] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */)) + (data->simulationInfo->realParameter[389] /* arm_off[387] PARAM */);
  TRACE_POP
}

/*
equation index: 14229
type: SIMPLE_ASSIGN
arm_off[386] = 4.838052686528282 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14229};
  (data->simulationInfo->realParameter[388] /* arm_off[386] PARAM */) = (4.838052686528282) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14230
type: SIMPLE_ASSIGN
theta[386] = pitch * r_init[386] + arm_off[386]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14230(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14230};
  (data->simulationInfo->realParameter[1892] /* theta[386] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */)) + (data->simulationInfo->realParameter[388] /* arm_off[386] PARAM */);
  TRACE_POP
}

/*
equation index: 14231
type: SIMPLE_ASSIGN
arm_off[385] = 4.825486315913922 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14231};
  (data->simulationInfo->realParameter[387] /* arm_off[385] PARAM */) = (4.825486315913922) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14232
type: SIMPLE_ASSIGN
theta[385] = pitch * r_init[385] + arm_off[385]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14232(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14232};
  (data->simulationInfo->realParameter[1891] /* theta[385] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */)) + (data->simulationInfo->realParameter[387] /* arm_off[385] PARAM */);
  TRACE_POP
}

/*
equation index: 14233
type: SIMPLE_ASSIGN
arm_off[384] = 4.812919945299563 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14233};
  (data->simulationInfo->realParameter[386] /* arm_off[384] PARAM */) = (4.812919945299563) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14234
type: SIMPLE_ASSIGN
theta[384] = pitch * r_init[384] + arm_off[384]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14234(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14234};
  (data->simulationInfo->realParameter[1890] /* theta[384] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */)) + (data->simulationInfo->realParameter[386] /* arm_off[384] PARAM */);
  TRACE_POP
}

/*
equation index: 14235
type: SIMPLE_ASSIGN
arm_off[383] = 4.800353574685204 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14235};
  (data->simulationInfo->realParameter[385] /* arm_off[383] PARAM */) = (4.800353574685204) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14236
type: SIMPLE_ASSIGN
theta[383] = pitch * r_init[383] + arm_off[383]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14236};
  (data->simulationInfo->realParameter[1889] /* theta[383] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */)) + (data->simulationInfo->realParameter[385] /* arm_off[383] PARAM */);
  TRACE_POP
}

/*
equation index: 14237
type: SIMPLE_ASSIGN
arm_off[382] = 4.7877872040708445 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14237};
  (data->simulationInfo->realParameter[384] /* arm_off[382] PARAM */) = (4.7877872040708445) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14238
type: SIMPLE_ASSIGN
theta[382] = pitch * r_init[382] + arm_off[382]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14238(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14238};
  (data->simulationInfo->realParameter[1888] /* theta[382] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */)) + (data->simulationInfo->realParameter[384] /* arm_off[382] PARAM */);
  TRACE_POP
}

/*
equation index: 14239
type: SIMPLE_ASSIGN
arm_off[381] = 4.775220833456486 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14239};
  (data->simulationInfo->realParameter[383] /* arm_off[381] PARAM */) = (4.775220833456486) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14240
type: SIMPLE_ASSIGN
theta[381] = pitch * r_init[381] + arm_off[381]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14240(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14240};
  (data->simulationInfo->realParameter[1887] /* theta[381] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */)) + (data->simulationInfo->realParameter[383] /* arm_off[381] PARAM */);
  TRACE_POP
}

/*
equation index: 14241
type: SIMPLE_ASSIGN
arm_off[380] = 4.762654462842126 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14241};
  (data->simulationInfo->realParameter[382] /* arm_off[380] PARAM */) = (4.762654462842126) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14242
type: SIMPLE_ASSIGN
theta[380] = pitch * r_init[380] + arm_off[380]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14242(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14242};
  (data->simulationInfo->realParameter[1886] /* theta[380] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */)) + (data->simulationInfo->realParameter[382] /* arm_off[380] PARAM */);
  TRACE_POP
}

/*
equation index: 14243
type: SIMPLE_ASSIGN
arm_off[379] = 4.7500880922277675 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14243};
  (data->simulationInfo->realParameter[381] /* arm_off[379] PARAM */) = (4.7500880922277675) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14244
type: SIMPLE_ASSIGN
theta[379] = pitch * r_init[379] + arm_off[379]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14244};
  (data->simulationInfo->realParameter[1885] /* theta[379] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */)) + (data->simulationInfo->realParameter[381] /* arm_off[379] PARAM */);
  TRACE_POP
}

/*
equation index: 14245
type: SIMPLE_ASSIGN
arm_off[378] = 4.737521721613408 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14245};
  (data->simulationInfo->realParameter[380] /* arm_off[378] PARAM */) = (4.737521721613408) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14246
type: SIMPLE_ASSIGN
theta[378] = pitch * r_init[378] + arm_off[378]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14246(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14246};
  (data->simulationInfo->realParameter[1884] /* theta[378] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */)) + (data->simulationInfo->realParameter[380] /* arm_off[378] PARAM */);
  TRACE_POP
}

/*
equation index: 14247
type: SIMPLE_ASSIGN
arm_off[377] = 4.724955350999049 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14247};
  (data->simulationInfo->realParameter[379] /* arm_off[377] PARAM */) = (4.724955350999049) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14248
type: SIMPLE_ASSIGN
theta[377] = pitch * r_init[377] + arm_off[377]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14248(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14248};
  (data->simulationInfo->realParameter[1883] /* theta[377] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */)) + (data->simulationInfo->realParameter[379] /* arm_off[377] PARAM */);
  TRACE_POP
}

/*
equation index: 14249
type: SIMPLE_ASSIGN
arm_off[376] = 4.71238898038469 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14249};
  (data->simulationInfo->realParameter[378] /* arm_off[376] PARAM */) = (4.71238898038469) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14250
type: SIMPLE_ASSIGN
theta[376] = pitch * r_init[376] + arm_off[376]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14250(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14250};
  (data->simulationInfo->realParameter[1882] /* theta[376] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */)) + (data->simulationInfo->realParameter[378] /* arm_off[376] PARAM */);
  TRACE_POP
}

/*
equation index: 14251
type: SIMPLE_ASSIGN
arm_off[375] = 4.699822609770331 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14251};
  (data->simulationInfo->realParameter[377] /* arm_off[375] PARAM */) = (4.699822609770331) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14252
type: SIMPLE_ASSIGN
theta[375] = pitch * r_init[375] + arm_off[375]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14252};
  (data->simulationInfo->realParameter[1881] /* theta[375] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */)) + (data->simulationInfo->realParameter[377] /* arm_off[375] PARAM */);
  TRACE_POP
}

/*
equation index: 14253
type: SIMPLE_ASSIGN
arm_off[374] = 4.687256239155971 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14253};
  (data->simulationInfo->realParameter[376] /* arm_off[374] PARAM */) = (4.687256239155971) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14254
type: SIMPLE_ASSIGN
theta[374] = pitch * r_init[374] + arm_off[374]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14254(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14254};
  (data->simulationInfo->realParameter[1880] /* theta[374] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */)) + (data->simulationInfo->realParameter[376] /* arm_off[374] PARAM */);
  TRACE_POP
}

/*
equation index: 14255
type: SIMPLE_ASSIGN
arm_off[373] = 4.674689868541612 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14255};
  (data->simulationInfo->realParameter[375] /* arm_off[373] PARAM */) = (4.674689868541612) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14256
type: SIMPLE_ASSIGN
theta[373] = pitch * r_init[373] + arm_off[373]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14256(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14256};
  (data->simulationInfo->realParameter[1879] /* theta[373] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */)) + (data->simulationInfo->realParameter[375] /* arm_off[373] PARAM */);
  TRACE_POP
}

/*
equation index: 14257
type: SIMPLE_ASSIGN
arm_off[372] = 4.6621234979272534 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14257};
  (data->simulationInfo->realParameter[374] /* arm_off[372] PARAM */) = (4.6621234979272534) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14258
type: SIMPLE_ASSIGN
theta[372] = pitch * r_init[372] + arm_off[372]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14258(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14258};
  (data->simulationInfo->realParameter[1878] /* theta[372] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */)) + (data->simulationInfo->realParameter[374] /* arm_off[372] PARAM */);
  TRACE_POP
}

/*
equation index: 14259
type: SIMPLE_ASSIGN
arm_off[371] = 4.649557127312893 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14259};
  (data->simulationInfo->realParameter[373] /* arm_off[371] PARAM */) = (4.649557127312893) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14260
type: SIMPLE_ASSIGN
theta[371] = pitch * r_init[371] + arm_off[371]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14260};
  (data->simulationInfo->realParameter[1877] /* theta[371] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */)) + (data->simulationInfo->realParameter[373] /* arm_off[371] PARAM */);
  TRACE_POP
}

/*
equation index: 14261
type: SIMPLE_ASSIGN
arm_off[370] = 4.636990756698535 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14261};
  (data->simulationInfo->realParameter[372] /* arm_off[370] PARAM */) = (4.636990756698535) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14262
type: SIMPLE_ASSIGN
theta[370] = pitch * r_init[370] + arm_off[370]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14262(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14262};
  (data->simulationInfo->realParameter[1876] /* theta[370] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */)) + (data->simulationInfo->realParameter[372] /* arm_off[370] PARAM */);
  TRACE_POP
}

/*
equation index: 14263
type: SIMPLE_ASSIGN
arm_off[369] = 4.624424386084176 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14263};
  (data->simulationInfo->realParameter[371] /* arm_off[369] PARAM */) = (4.624424386084176) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14264
type: SIMPLE_ASSIGN
theta[369] = pitch * r_init[369] + arm_off[369]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14264(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14264};
  (data->simulationInfo->realParameter[1875] /* theta[369] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */)) + (data->simulationInfo->realParameter[371] /* arm_off[369] PARAM */);
  TRACE_POP
}

/*
equation index: 14265
type: SIMPLE_ASSIGN
arm_off[368] = 4.611858015469816 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14265};
  (data->simulationInfo->realParameter[370] /* arm_off[368] PARAM */) = (4.611858015469816) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14266
type: SIMPLE_ASSIGN
theta[368] = pitch * r_init[368] + arm_off[368]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14266(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14266};
  (data->simulationInfo->realParameter[1874] /* theta[368] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */)) + (data->simulationInfo->realParameter[370] /* arm_off[368] PARAM */);
  TRACE_POP
}

/*
equation index: 14267
type: SIMPLE_ASSIGN
arm_off[367] = 4.599291644855457 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14267};
  (data->simulationInfo->realParameter[369] /* arm_off[367] PARAM */) = (4.599291644855457) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14268
type: SIMPLE_ASSIGN
theta[367] = pitch * r_init[367] + arm_off[367]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14268};
  (data->simulationInfo->realParameter[1873] /* theta[367] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */)) + (data->simulationInfo->realParameter[369] /* arm_off[367] PARAM */);
  TRACE_POP
}

/*
equation index: 14269
type: SIMPLE_ASSIGN
arm_off[366] = 4.586725274241099 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14269};
  (data->simulationInfo->realParameter[368] /* arm_off[366] PARAM */) = (4.586725274241099) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14270
type: SIMPLE_ASSIGN
theta[366] = pitch * r_init[366] + arm_off[366]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14270(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14270};
  (data->simulationInfo->realParameter[1872] /* theta[366] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */)) + (data->simulationInfo->realParameter[368] /* arm_off[366] PARAM */);
  TRACE_POP
}

/*
equation index: 14271
type: SIMPLE_ASSIGN
arm_off[365] = 4.5741589036267385 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14271};
  (data->simulationInfo->realParameter[367] /* arm_off[365] PARAM */) = (4.5741589036267385) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14272
type: SIMPLE_ASSIGN
theta[365] = pitch * r_init[365] + arm_off[365]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14272(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14272};
  (data->simulationInfo->realParameter[1871] /* theta[365] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */)) + (data->simulationInfo->realParameter[367] /* arm_off[365] PARAM */);
  TRACE_POP
}

/*
equation index: 14273
type: SIMPLE_ASSIGN
arm_off[364] = 4.56159253301238 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14273};
  (data->simulationInfo->realParameter[366] /* arm_off[364] PARAM */) = (4.56159253301238) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14274
type: SIMPLE_ASSIGN
theta[364] = pitch * r_init[364] + arm_off[364]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14274(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14274};
  (data->simulationInfo->realParameter[1870] /* theta[364] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */)) + (data->simulationInfo->realParameter[366] /* arm_off[364] PARAM */);
  TRACE_POP
}

/*
equation index: 14275
type: SIMPLE_ASSIGN
arm_off[363] = 4.549026162398021 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14275};
  (data->simulationInfo->realParameter[365] /* arm_off[363] PARAM */) = (4.549026162398021) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14276
type: SIMPLE_ASSIGN
theta[363] = pitch * r_init[363] + arm_off[363]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14276};
  (data->simulationInfo->realParameter[1869] /* theta[363] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */)) + (data->simulationInfo->realParameter[365] /* arm_off[363] PARAM */);
  TRACE_POP
}

/*
equation index: 14277
type: SIMPLE_ASSIGN
arm_off[362] = 4.536459791783661 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14277};
  (data->simulationInfo->realParameter[364] /* arm_off[362] PARAM */) = (4.536459791783661) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14278
type: SIMPLE_ASSIGN
theta[362] = pitch * r_init[362] + arm_off[362]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14278(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14278};
  (data->simulationInfo->realParameter[1868] /* theta[362] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */)) + (data->simulationInfo->realParameter[364] /* arm_off[362] PARAM */);
  TRACE_POP
}

/*
equation index: 14279
type: SIMPLE_ASSIGN
arm_off[361] = 4.523893421169302 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14279};
  (data->simulationInfo->realParameter[363] /* arm_off[361] PARAM */) = (4.523893421169302) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14280
type: SIMPLE_ASSIGN
theta[361] = pitch * r_init[361] + arm_off[361]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14280(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14280};
  (data->simulationInfo->realParameter[1867] /* theta[361] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */)) + (data->simulationInfo->realParameter[363] /* arm_off[361] PARAM */);
  TRACE_POP
}

/*
equation index: 14281
type: SIMPLE_ASSIGN
arm_off[360] = 4.511327050554943 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14281};
  (data->simulationInfo->realParameter[362] /* arm_off[360] PARAM */) = (4.511327050554943) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14282
type: SIMPLE_ASSIGN
theta[360] = pitch * r_init[360] + arm_off[360]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14282(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14282};
  (data->simulationInfo->realParameter[1866] /* theta[360] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */)) + (data->simulationInfo->realParameter[362] /* arm_off[360] PARAM */);
  TRACE_POP
}

/*
equation index: 14283
type: SIMPLE_ASSIGN
arm_off[359] = 4.498760679940584 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14283};
  (data->simulationInfo->realParameter[361] /* arm_off[359] PARAM */) = (4.498760679940584) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14284
type: SIMPLE_ASSIGN
theta[359] = pitch * r_init[359] + arm_off[359]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14284};
  (data->simulationInfo->realParameter[1865] /* theta[359] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */)) + (data->simulationInfo->realParameter[361] /* arm_off[359] PARAM */);
  TRACE_POP
}

/*
equation index: 14285
type: SIMPLE_ASSIGN
arm_off[358] = 4.486194309326224 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14285};
  (data->simulationInfo->realParameter[360] /* arm_off[358] PARAM */) = (4.486194309326224) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14286
type: SIMPLE_ASSIGN
theta[358] = pitch * r_init[358] + arm_off[358]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14286(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14286};
  (data->simulationInfo->realParameter[1864] /* theta[358] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */)) + (data->simulationInfo->realParameter[360] /* arm_off[358] PARAM */);
  TRACE_POP
}

/*
equation index: 14287
type: SIMPLE_ASSIGN
arm_off[357] = 4.473627938711866 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14287};
  (data->simulationInfo->realParameter[359] /* arm_off[357] PARAM */) = (4.473627938711866) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14288
type: SIMPLE_ASSIGN
theta[357] = pitch * r_init[357] + arm_off[357]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14288(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14288};
  (data->simulationInfo->realParameter[1863] /* theta[357] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */)) + (data->simulationInfo->realParameter[359] /* arm_off[357] PARAM */);
  TRACE_POP
}

/*
equation index: 14289
type: SIMPLE_ASSIGN
arm_off[356] = 4.461061568097506 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14289};
  (data->simulationInfo->realParameter[358] /* arm_off[356] PARAM */) = (4.461061568097506) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14290
type: SIMPLE_ASSIGN
theta[356] = pitch * r_init[356] + arm_off[356]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14290(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14290};
  (data->simulationInfo->realParameter[1862] /* theta[356] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */)) + (data->simulationInfo->realParameter[358] /* arm_off[356] PARAM */);
  TRACE_POP
}

/*
equation index: 14291
type: SIMPLE_ASSIGN
arm_off[355] = 4.4484951974831475 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14291};
  (data->simulationInfo->realParameter[357] /* arm_off[355] PARAM */) = (4.4484951974831475) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14292
type: SIMPLE_ASSIGN
theta[355] = pitch * r_init[355] + arm_off[355]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14292};
  (data->simulationInfo->realParameter[1861] /* theta[355] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */)) + (data->simulationInfo->realParameter[357] /* arm_off[355] PARAM */);
  TRACE_POP
}

/*
equation index: 14293
type: SIMPLE_ASSIGN
arm_off[354] = 4.435928826868788 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14293};
  (data->simulationInfo->realParameter[356] /* arm_off[354] PARAM */) = (4.435928826868788) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14294
type: SIMPLE_ASSIGN
theta[354] = pitch * r_init[354] + arm_off[354]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14294(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14294};
  (data->simulationInfo->realParameter[1860] /* theta[354] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */)) + (data->simulationInfo->realParameter[356] /* arm_off[354] PARAM */);
  TRACE_POP
}

/*
equation index: 14295
type: SIMPLE_ASSIGN
arm_off[353] = 4.423362456254428 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14295};
  (data->simulationInfo->realParameter[355] /* arm_off[353] PARAM */) = (4.423362456254428) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14296
type: SIMPLE_ASSIGN
theta[353] = pitch * r_init[353] + arm_off[353]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14296(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14296};
  (data->simulationInfo->realParameter[1859] /* theta[353] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */)) + (data->simulationInfo->realParameter[355] /* arm_off[353] PARAM */);
  TRACE_POP
}

/*
equation index: 14297
type: SIMPLE_ASSIGN
arm_off[352] = 4.41079608564007 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14297};
  (data->simulationInfo->realParameter[354] /* arm_off[352] PARAM */) = (4.41079608564007) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14298
type: SIMPLE_ASSIGN
theta[352] = pitch * r_init[352] + arm_off[352]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14298(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14298};
  (data->simulationInfo->realParameter[1858] /* theta[352] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */)) + (data->simulationInfo->realParameter[354] /* arm_off[352] PARAM */);
  TRACE_POP
}

/*
equation index: 14299
type: SIMPLE_ASSIGN
arm_off[351] = 4.39822971502571 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14299};
  (data->simulationInfo->realParameter[353] /* arm_off[351] PARAM */) = (4.39822971502571) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14300
type: SIMPLE_ASSIGN
theta[351] = pitch * r_init[351] + arm_off[351]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14300};
  (data->simulationInfo->realParameter[1857] /* theta[351] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */)) + (data->simulationInfo->realParameter[353] /* arm_off[351] PARAM */);
  TRACE_POP
}

/*
equation index: 14301
type: SIMPLE_ASSIGN
arm_off[350] = 4.385663344411351 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14301};
  (data->simulationInfo->realParameter[352] /* arm_off[350] PARAM */) = (4.385663344411351) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14302
type: SIMPLE_ASSIGN
theta[350] = pitch * r_init[350] + arm_off[350]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14302(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14302};
  (data->simulationInfo->realParameter[1856] /* theta[350] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */)) + (data->simulationInfo->realParameter[352] /* arm_off[350] PARAM */);
  TRACE_POP
}

/*
equation index: 14303
type: SIMPLE_ASSIGN
arm_off[349] = 4.373096973796992 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14303};
  (data->simulationInfo->realParameter[351] /* arm_off[349] PARAM */) = (4.373096973796992) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14304
type: SIMPLE_ASSIGN
theta[349] = pitch * r_init[349] + arm_off[349]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14304(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14304};
  (data->simulationInfo->realParameter[1855] /* theta[349] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */)) + (data->simulationInfo->realParameter[351] /* arm_off[349] PARAM */);
  TRACE_POP
}

/*
equation index: 14305
type: SIMPLE_ASSIGN
arm_off[348] = 4.360530603182633 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14305};
  (data->simulationInfo->realParameter[350] /* arm_off[348] PARAM */) = (4.360530603182633) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14306
type: SIMPLE_ASSIGN
theta[348] = pitch * r_init[348] + arm_off[348]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14306(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14306};
  (data->simulationInfo->realParameter[1854] /* theta[348] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */)) + (data->simulationInfo->realParameter[350] /* arm_off[348] PARAM */);
  TRACE_POP
}

/*
equation index: 14307
type: SIMPLE_ASSIGN
arm_off[347] = 4.347964232568273 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14307};
  (data->simulationInfo->realParameter[349] /* arm_off[347] PARAM */) = (4.347964232568273) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14308
type: SIMPLE_ASSIGN
theta[347] = pitch * r_init[347] + arm_off[347]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14308};
  (data->simulationInfo->realParameter[1853] /* theta[347] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */)) + (data->simulationInfo->realParameter[349] /* arm_off[347] PARAM */);
  TRACE_POP
}

/*
equation index: 14309
type: SIMPLE_ASSIGN
arm_off[346] = 4.335397861953915 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14309};
  (data->simulationInfo->realParameter[348] /* arm_off[346] PARAM */) = (4.335397861953915) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14310
type: SIMPLE_ASSIGN
theta[346] = pitch * r_init[346] + arm_off[346]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14310(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14310};
  (data->simulationInfo->realParameter[1852] /* theta[346] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */)) + (data->simulationInfo->realParameter[348] /* arm_off[346] PARAM */);
  TRACE_POP
}

/*
equation index: 14311
type: SIMPLE_ASSIGN
arm_off[345] = 4.322831491339556 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14311};
  (data->simulationInfo->realParameter[347] /* arm_off[345] PARAM */) = (4.322831491339556) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14312
type: SIMPLE_ASSIGN
theta[345] = pitch * r_init[345] + arm_off[345]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14312(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14312};
  (data->simulationInfo->realParameter[1851] /* theta[345] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */)) + (data->simulationInfo->realParameter[347] /* arm_off[345] PARAM */);
  TRACE_POP
}

/*
equation index: 14313
type: SIMPLE_ASSIGN
arm_off[344] = 4.310265120725196 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14313};
  (data->simulationInfo->realParameter[346] /* arm_off[344] PARAM */) = (4.310265120725196) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14314
type: SIMPLE_ASSIGN
theta[344] = pitch * r_init[344] + arm_off[344]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14314(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14314};
  (data->simulationInfo->realParameter[1850] /* theta[344] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */)) + (data->simulationInfo->realParameter[346] /* arm_off[344] PARAM */);
  TRACE_POP
}

/*
equation index: 14315
type: SIMPLE_ASSIGN
arm_off[343] = 4.297698750110837 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14315};
  (data->simulationInfo->realParameter[345] /* arm_off[343] PARAM */) = (4.297698750110837) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14316
type: SIMPLE_ASSIGN
theta[343] = pitch * r_init[343] + arm_off[343]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14316};
  (data->simulationInfo->realParameter[1849] /* theta[343] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */)) + (data->simulationInfo->realParameter[345] /* arm_off[343] PARAM */);
  TRACE_POP
}

/*
equation index: 14317
type: SIMPLE_ASSIGN
arm_off[342] = 4.285132379496478 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14317};
  (data->simulationInfo->realParameter[344] /* arm_off[342] PARAM */) = (4.285132379496478) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14318
type: SIMPLE_ASSIGN
theta[342] = pitch * r_init[342] + arm_off[342]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14318(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14318};
  (data->simulationInfo->realParameter[1848] /* theta[342] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */)) + (data->simulationInfo->realParameter[344] /* arm_off[342] PARAM */);
  TRACE_POP
}

/*
equation index: 14319
type: SIMPLE_ASSIGN
arm_off[341] = 4.2725660088821185 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14319};
  (data->simulationInfo->realParameter[343] /* arm_off[341] PARAM */) = (4.2725660088821185) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14320
type: SIMPLE_ASSIGN
theta[341] = pitch * r_init[341] + arm_off[341]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14320(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14320};
  (data->simulationInfo->realParameter[1847] /* theta[341] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */)) + (data->simulationInfo->realParameter[343] /* arm_off[341] PARAM */);
  TRACE_POP
}

/*
equation index: 14321
type: SIMPLE_ASSIGN
arm_off[340] = 4.259999638267759 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14321};
  (data->simulationInfo->realParameter[342] /* arm_off[340] PARAM */) = (4.259999638267759) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14322
type: SIMPLE_ASSIGN
theta[340] = pitch * r_init[340] + arm_off[340]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14322(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14322};
  (data->simulationInfo->realParameter[1846] /* theta[340] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */)) + (data->simulationInfo->realParameter[342] /* arm_off[340] PARAM */);
  TRACE_POP
}

/*
equation index: 14323
type: SIMPLE_ASSIGN
arm_off[339] = 4.247433267653401 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14323};
  (data->simulationInfo->realParameter[341] /* arm_off[339] PARAM */) = (4.247433267653401) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14324
type: SIMPLE_ASSIGN
theta[339] = pitch * r_init[339] + arm_off[339]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14324};
  (data->simulationInfo->realParameter[1845] /* theta[339] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */)) + (data->simulationInfo->realParameter[341] /* arm_off[339] PARAM */);
  TRACE_POP
}

/*
equation index: 14325
type: SIMPLE_ASSIGN
arm_off[338] = 4.234866897039041 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14325};
  (data->simulationInfo->realParameter[340] /* arm_off[338] PARAM */) = (4.234866897039041) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14326
type: SIMPLE_ASSIGN
theta[338] = pitch * r_init[338] + arm_off[338]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14326(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14326};
  (data->simulationInfo->realParameter[1844] /* theta[338] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */)) + (data->simulationInfo->realParameter[340] /* arm_off[338] PARAM */);
  TRACE_POP
}

/*
equation index: 14327
type: SIMPLE_ASSIGN
arm_off[337] = 4.222300526424682 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14327};
  (data->simulationInfo->realParameter[339] /* arm_off[337] PARAM */) = (4.222300526424682) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14328
type: SIMPLE_ASSIGN
theta[337] = pitch * r_init[337] + arm_off[337]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14328(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14328};
  (data->simulationInfo->realParameter[1843] /* theta[337] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */)) + (data->simulationInfo->realParameter[339] /* arm_off[337] PARAM */);
  TRACE_POP
}

/*
equation index: 14329
type: SIMPLE_ASSIGN
arm_off[336] = 4.209734155810323 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14329};
  (data->simulationInfo->realParameter[338] /* arm_off[336] PARAM */) = (4.209734155810323) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14330
type: SIMPLE_ASSIGN
theta[336] = pitch * r_init[336] + arm_off[336]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14330(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14330};
  (data->simulationInfo->realParameter[1842] /* theta[336] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */)) + (data->simulationInfo->realParameter[338] /* arm_off[336] PARAM */);
  TRACE_POP
}

/*
equation index: 14331
type: SIMPLE_ASSIGN
arm_off[335] = 4.197167785195964 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14331(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14331};
  (data->simulationInfo->realParameter[337] /* arm_off[335] PARAM */) = (4.197167785195964) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14332
type: SIMPLE_ASSIGN
theta[335] = pitch * r_init[335] + arm_off[335]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14332};
  (data->simulationInfo->realParameter[1841] /* theta[335] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */)) + (data->simulationInfo->realParameter[337] /* arm_off[335] PARAM */);
  TRACE_POP
}

/*
equation index: 14333
type: SIMPLE_ASSIGN
arm_off[334] = 4.184601414581604 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14333};
  (data->simulationInfo->realParameter[336] /* arm_off[334] PARAM */) = (4.184601414581604) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14334
type: SIMPLE_ASSIGN
theta[334] = pitch * r_init[334] + arm_off[334]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14334(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14334};
  (data->simulationInfo->realParameter[1840] /* theta[334] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */)) + (data->simulationInfo->realParameter[336] /* arm_off[334] PARAM */);
  TRACE_POP
}

/*
equation index: 14335
type: SIMPLE_ASSIGN
arm_off[333] = 4.172035043967246 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14335};
  (data->simulationInfo->realParameter[335] /* arm_off[333] PARAM */) = (4.172035043967246) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14336
type: SIMPLE_ASSIGN
theta[333] = pitch * r_init[333] + arm_off[333]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14336(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14336};
  (data->simulationInfo->realParameter[1839] /* theta[333] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */)) + (data->simulationInfo->realParameter[335] /* arm_off[333] PARAM */);
  TRACE_POP
}

/*
equation index: 14337
type: SIMPLE_ASSIGN
arm_off[332] = 4.159468673352886 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14337};
  (data->simulationInfo->realParameter[334] /* arm_off[332] PARAM */) = (4.159468673352886) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14338
type: SIMPLE_ASSIGN
theta[332] = pitch * r_init[332] + arm_off[332]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14338(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14338};
  (data->simulationInfo->realParameter[1838] /* theta[332] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */)) + (data->simulationInfo->realParameter[334] /* arm_off[332] PARAM */);
  TRACE_POP
}

/*
equation index: 14339
type: SIMPLE_ASSIGN
arm_off[331] = 4.146902302738527 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14339};
  (data->simulationInfo->realParameter[333] /* arm_off[331] PARAM */) = (4.146902302738527) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14340
type: SIMPLE_ASSIGN
theta[331] = pitch * r_init[331] + arm_off[331]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14340};
  (data->simulationInfo->realParameter[1837] /* theta[331] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */)) + (data->simulationInfo->realParameter[333] /* arm_off[331] PARAM */);
  TRACE_POP
}

/*
equation index: 14341
type: SIMPLE_ASSIGN
arm_off[330] = 4.134335932124168 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14341};
  (data->simulationInfo->realParameter[332] /* arm_off[330] PARAM */) = (4.134335932124168) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14342
type: SIMPLE_ASSIGN
theta[330] = pitch * r_init[330] + arm_off[330]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14342(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14342};
  (data->simulationInfo->realParameter[1836] /* theta[330] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */)) + (data->simulationInfo->realParameter[332] /* arm_off[330] PARAM */);
  TRACE_POP
}

/*
equation index: 14343
type: SIMPLE_ASSIGN
arm_off[329] = 4.121769561509808 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14343};
  (data->simulationInfo->realParameter[331] /* arm_off[329] PARAM */) = (4.121769561509808) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14344
type: SIMPLE_ASSIGN
theta[329] = pitch * r_init[329] + arm_off[329]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14344(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14344};
  (data->simulationInfo->realParameter[1835] /* theta[329] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */)) + (data->simulationInfo->realParameter[331] /* arm_off[329] PARAM */);
  TRACE_POP
}

/*
equation index: 14345
type: SIMPLE_ASSIGN
arm_off[328] = 4.10920319089545 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14345};
  (data->simulationInfo->realParameter[330] /* arm_off[328] PARAM */) = (4.10920319089545) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14346
type: SIMPLE_ASSIGN
theta[328] = pitch * r_init[328] + arm_off[328]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14346(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14346};
  (data->simulationInfo->realParameter[1834] /* theta[328] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */)) + (data->simulationInfo->realParameter[330] /* arm_off[328] PARAM */);
  TRACE_POP
}

/*
equation index: 14347
type: SIMPLE_ASSIGN
arm_off[327] = 4.09663682028109 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14347(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14347};
  (data->simulationInfo->realParameter[329] /* arm_off[327] PARAM */) = (4.09663682028109) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14348
type: SIMPLE_ASSIGN
theta[327] = pitch * r_init[327] + arm_off[327]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14348};
  (data->simulationInfo->realParameter[1833] /* theta[327] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */)) + (data->simulationInfo->realParameter[329] /* arm_off[327] PARAM */);
  TRACE_POP
}

/*
equation index: 14349
type: SIMPLE_ASSIGN
arm_off[326] = 4.084070449666731 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14349};
  (data->simulationInfo->realParameter[328] /* arm_off[326] PARAM */) = (4.084070449666731) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14350
type: SIMPLE_ASSIGN
theta[326] = pitch * r_init[326] + arm_off[326]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14350(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14350};
  (data->simulationInfo->realParameter[1832] /* theta[326] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */)) + (data->simulationInfo->realParameter[328] /* arm_off[326] PARAM */);
  TRACE_POP
}

/*
equation index: 14351
type: SIMPLE_ASSIGN
arm_off[325] = 4.071504079052372 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14351};
  (data->simulationInfo->realParameter[327] /* arm_off[325] PARAM */) = (4.071504079052372) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14352
type: SIMPLE_ASSIGN
theta[325] = pitch * r_init[325] + arm_off[325]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14352(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14352};
  (data->simulationInfo->realParameter[1831] /* theta[325] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */)) + (data->simulationInfo->realParameter[327] /* arm_off[325] PARAM */);
  TRACE_POP
}

/*
equation index: 14353
type: SIMPLE_ASSIGN
arm_off[324] = 4.0589377084380125 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14353};
  (data->simulationInfo->realParameter[326] /* arm_off[324] PARAM */) = (4.0589377084380125) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14354
type: SIMPLE_ASSIGN
theta[324] = pitch * r_init[324] + arm_off[324]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14354(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14354};
  (data->simulationInfo->realParameter[1830] /* theta[324] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */)) + (data->simulationInfo->realParameter[326] /* arm_off[324] PARAM */);
  TRACE_POP
}

/*
equation index: 14355
type: SIMPLE_ASSIGN
arm_off[323] = 4.046371337823653 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14355};
  (data->simulationInfo->realParameter[325] /* arm_off[323] PARAM */) = (4.046371337823653) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14356
type: SIMPLE_ASSIGN
theta[323] = pitch * r_init[323] + arm_off[323]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14356};
  (data->simulationInfo->realParameter[1829] /* theta[323] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */)) + (data->simulationInfo->realParameter[325] /* arm_off[323] PARAM */);
  TRACE_POP
}

/*
equation index: 14357
type: SIMPLE_ASSIGN
arm_off[322] = 4.033804967209294 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14357};
  (data->simulationInfo->realParameter[324] /* arm_off[322] PARAM */) = (4.033804967209294) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14358
type: SIMPLE_ASSIGN
theta[322] = pitch * r_init[322] + arm_off[322]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14358(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14358};
  (data->simulationInfo->realParameter[1828] /* theta[322] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */)) + (data->simulationInfo->realParameter[324] /* arm_off[322] PARAM */);
  TRACE_POP
}

/*
equation index: 14359
type: SIMPLE_ASSIGN
arm_off[321] = 4.0212385965949355 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14359};
  (data->simulationInfo->realParameter[323] /* arm_off[321] PARAM */) = (4.0212385965949355) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14360
type: SIMPLE_ASSIGN
theta[321] = pitch * r_init[321] + arm_off[321]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14360(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14360};
  (data->simulationInfo->realParameter[1827] /* theta[321] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */)) + (data->simulationInfo->realParameter[323] /* arm_off[321] PARAM */);
  TRACE_POP
}

/*
equation index: 14361
type: SIMPLE_ASSIGN
arm_off[320] = 4.008672225980576 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14361};
  (data->simulationInfo->realParameter[322] /* arm_off[320] PARAM */) = (4.008672225980576) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14362
type: SIMPLE_ASSIGN
theta[320] = pitch * r_init[320] + arm_off[320]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14362(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14362};
  (data->simulationInfo->realParameter[1826] /* theta[320] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */)) + (data->simulationInfo->realParameter[322] /* arm_off[320] PARAM */);
  TRACE_POP
}

/*
equation index: 14363
type: SIMPLE_ASSIGN
arm_off[319] = 3.996105855366217 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14363};
  (data->simulationInfo->realParameter[321] /* arm_off[319] PARAM */) = (3.996105855366217) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14364
type: SIMPLE_ASSIGN
theta[319] = pitch * r_init[319] + arm_off[319]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14364};
  (data->simulationInfo->realParameter[1825] /* theta[319] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */)) + (data->simulationInfo->realParameter[321] /* arm_off[319] PARAM */);
  TRACE_POP
}

/*
equation index: 14365
type: SIMPLE_ASSIGN
arm_off[318] = 3.9835394847518577 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14365};
  (data->simulationInfo->realParameter[320] /* arm_off[318] PARAM */) = (3.9835394847518577) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14366
type: SIMPLE_ASSIGN
theta[318] = pitch * r_init[318] + arm_off[318]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14366(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14366};
  (data->simulationInfo->realParameter[1824] /* theta[318] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */)) + (data->simulationInfo->realParameter[320] /* arm_off[318] PARAM */);
  TRACE_POP
}

/*
equation index: 14367
type: SIMPLE_ASSIGN
arm_off[317] = 3.970973114137499 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14367};
  (data->simulationInfo->realParameter[319] /* arm_off[317] PARAM */) = (3.970973114137499) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14368
type: SIMPLE_ASSIGN
theta[317] = pitch * r_init[317] + arm_off[317]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14368(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14368};
  (data->simulationInfo->realParameter[1823] /* theta[317] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */)) + (data->simulationInfo->realParameter[319] /* arm_off[317] PARAM */);
  TRACE_POP
}

/*
equation index: 14369
type: SIMPLE_ASSIGN
arm_off[316] = 3.9584067435231396 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14369};
  (data->simulationInfo->realParameter[318] /* arm_off[316] PARAM */) = (3.9584067435231396) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14370
type: SIMPLE_ASSIGN
theta[316] = pitch * r_init[316] + arm_off[316]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14370(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14370};
  (data->simulationInfo->realParameter[1822] /* theta[316] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */)) + (data->simulationInfo->realParameter[318] /* arm_off[316] PARAM */);
  TRACE_POP
}

/*
equation index: 14371
type: SIMPLE_ASSIGN
arm_off[315] = 3.94584037290878 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14371};
  (data->simulationInfo->realParameter[317] /* arm_off[315] PARAM */) = (3.94584037290878) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14372
type: SIMPLE_ASSIGN
theta[315] = pitch * r_init[315] + arm_off[315]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14372};
  (data->simulationInfo->realParameter[1821] /* theta[315] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */)) + (data->simulationInfo->realParameter[317] /* arm_off[315] PARAM */);
  TRACE_POP
}

/*
equation index: 14373
type: SIMPLE_ASSIGN
arm_off[314] = 3.933274002294421 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14373};
  (data->simulationInfo->realParameter[316] /* arm_off[314] PARAM */) = (3.933274002294421) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14374
type: SIMPLE_ASSIGN
theta[314] = pitch * r_init[314] + arm_off[314]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14374(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14374};
  (data->simulationInfo->realParameter[1820] /* theta[314] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */)) + (data->simulationInfo->realParameter[316] /* arm_off[314] PARAM */);
  TRACE_POP
}

/*
equation index: 14375
type: SIMPLE_ASSIGN
arm_off[313] = 3.9207076316800618 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14375};
  (data->simulationInfo->realParameter[315] /* arm_off[313] PARAM */) = (3.9207076316800618) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14376
type: SIMPLE_ASSIGN
theta[313] = pitch * r_init[313] + arm_off[313]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14376(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14376};
  (data->simulationInfo->realParameter[1819] /* theta[313] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */)) + (data->simulationInfo->realParameter[315] /* arm_off[313] PARAM */);
  TRACE_POP
}

/*
equation index: 14377
type: SIMPLE_ASSIGN
arm_off[312] = 3.9081412610657025 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14377};
  (data->simulationInfo->realParameter[314] /* arm_off[312] PARAM */) = (3.9081412610657025) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14378
type: SIMPLE_ASSIGN
theta[312] = pitch * r_init[312] + arm_off[312]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14378(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14378};
  (data->simulationInfo->realParameter[1818] /* theta[312] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */)) + (data->simulationInfo->realParameter[314] /* arm_off[312] PARAM */);
  TRACE_POP
}

/*
equation index: 14379
type: SIMPLE_ASSIGN
arm_off[311] = 3.8955748904513436 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14379};
  (data->simulationInfo->realParameter[313] /* arm_off[311] PARAM */) = (3.8955748904513436) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14380
type: SIMPLE_ASSIGN
theta[311] = pitch * r_init[311] + arm_off[311]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14380};
  (data->simulationInfo->realParameter[1817] /* theta[311] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */)) + (data->simulationInfo->realParameter[313] /* arm_off[311] PARAM */);
  TRACE_POP
}

/*
equation index: 14381
type: SIMPLE_ASSIGN
arm_off[310] = 3.8830085198369844 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14381};
  (data->simulationInfo->realParameter[312] /* arm_off[310] PARAM */) = (3.8830085198369844) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14382
type: SIMPLE_ASSIGN
theta[310] = pitch * r_init[310] + arm_off[310]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14382(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14382};
  (data->simulationInfo->realParameter[1816] /* theta[310] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */)) + (data->simulationInfo->realParameter[312] /* arm_off[310] PARAM */);
  TRACE_POP
}

/*
equation index: 14383
type: SIMPLE_ASSIGN
arm_off[309] = 3.870442149222625 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14383};
  (data->simulationInfo->realParameter[311] /* arm_off[309] PARAM */) = (3.870442149222625) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14384
type: SIMPLE_ASSIGN
theta[309] = pitch * r_init[309] + arm_off[309]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14384(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14384};
  (data->simulationInfo->realParameter[1815] /* theta[309] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */)) + (data->simulationInfo->realParameter[311] /* arm_off[309] PARAM */);
  TRACE_POP
}

/*
equation index: 14385
type: SIMPLE_ASSIGN
arm_off[308] = 3.8578757786082662 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14385};
  (data->simulationInfo->realParameter[310] /* arm_off[308] PARAM */) = (3.8578757786082662) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14386
type: SIMPLE_ASSIGN
theta[308] = pitch * r_init[308] + arm_off[308]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14386(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14386};
  (data->simulationInfo->realParameter[1814] /* theta[308] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */)) + (data->simulationInfo->realParameter[310] /* arm_off[308] PARAM */);
  TRACE_POP
}

/*
equation index: 14387
type: SIMPLE_ASSIGN
arm_off[307] = 3.845309407993907 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14387};
  (data->simulationInfo->realParameter[309] /* arm_off[307] PARAM */) = (3.845309407993907) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14388
type: SIMPLE_ASSIGN
theta[307] = pitch * r_init[307] + arm_off[307]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14388};
  (data->simulationInfo->realParameter[1813] /* theta[307] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */)) + (data->simulationInfo->realParameter[309] /* arm_off[307] PARAM */);
  TRACE_POP
}

/*
equation index: 14389
type: SIMPLE_ASSIGN
arm_off[306] = 3.8327430373795477 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14389};
  (data->simulationInfo->realParameter[308] /* arm_off[306] PARAM */) = (3.8327430373795477) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14390
type: SIMPLE_ASSIGN
theta[306] = pitch * r_init[306] + arm_off[306]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14390(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14390};
  (data->simulationInfo->realParameter[1812] /* theta[306] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */)) + (data->simulationInfo->realParameter[308] /* arm_off[306] PARAM */);
  TRACE_POP
}

/*
equation index: 14391
type: SIMPLE_ASSIGN
arm_off[305] = 3.8201766667651884 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14391};
  (data->simulationInfo->realParameter[307] /* arm_off[305] PARAM */) = (3.8201766667651884) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14392
type: SIMPLE_ASSIGN
theta[305] = pitch * r_init[305] + arm_off[305]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14392(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14392};
  (data->simulationInfo->realParameter[1811] /* theta[305] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */)) + (data->simulationInfo->realParameter[307] /* arm_off[305] PARAM */);
  TRACE_POP
}

/*
equation index: 14393
type: SIMPLE_ASSIGN
arm_off[304] = 3.807610296150829 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14393};
  (data->simulationInfo->realParameter[306] /* arm_off[304] PARAM */) = (3.807610296150829) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14394
type: SIMPLE_ASSIGN
theta[304] = pitch * r_init[304] + arm_off[304]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14394(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14394};
  (data->simulationInfo->realParameter[1810] /* theta[304] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */)) + (data->simulationInfo->realParameter[306] /* arm_off[304] PARAM */);
  TRACE_POP
}

/*
equation index: 14395
type: SIMPLE_ASSIGN
arm_off[303] = 3.79504392553647 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14395};
  (data->simulationInfo->realParameter[305] /* arm_off[303] PARAM */) = (3.79504392553647) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14396
type: SIMPLE_ASSIGN
theta[303] = pitch * r_init[303] + arm_off[303]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14396};
  (data->simulationInfo->realParameter[1809] /* theta[303] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */)) + (data->simulationInfo->realParameter[305] /* arm_off[303] PARAM */);
  TRACE_POP
}

/*
equation index: 14397
type: SIMPLE_ASSIGN
arm_off[302] = 3.782477554922111 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14397};
  (data->simulationInfo->realParameter[304] /* arm_off[302] PARAM */) = (3.782477554922111) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14398
type: SIMPLE_ASSIGN
theta[302] = pitch * r_init[302] + arm_off[302]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14398(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14398};
  (data->simulationInfo->realParameter[1808] /* theta[302] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */)) + (data->simulationInfo->realParameter[304] /* arm_off[302] PARAM */);
  TRACE_POP
}

/*
equation index: 14399
type: SIMPLE_ASSIGN
arm_off[301] = 3.7699111843077517 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14399};
  (data->simulationInfo->realParameter[303] /* arm_off[301] PARAM */) = (3.7699111843077517) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14400
type: SIMPLE_ASSIGN
theta[301] = pitch * r_init[301] + arm_off[301]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14400(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14400};
  (data->simulationInfo->realParameter[1807] /* theta[301] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */)) + (data->simulationInfo->realParameter[303] /* arm_off[301] PARAM */);
  TRACE_POP
}

/*
equation index: 14401
type: SIMPLE_ASSIGN
arm_off[300] = 3.7573448136933925 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14401};
  (data->simulationInfo->realParameter[302] /* arm_off[300] PARAM */) = (3.7573448136933925) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14402
type: SIMPLE_ASSIGN
theta[300] = pitch * r_init[300] + arm_off[300]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14402(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14402};
  (data->simulationInfo->realParameter[1806] /* theta[300] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */)) + (data->simulationInfo->realParameter[302] /* arm_off[300] PARAM */);
  TRACE_POP
}

/*
equation index: 14403
type: SIMPLE_ASSIGN
arm_off[299] = 3.7447784430790336 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14403};
  (data->simulationInfo->realParameter[301] /* arm_off[299] PARAM */) = (3.7447784430790336) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14404
type: SIMPLE_ASSIGN
theta[299] = pitch * r_init[299] + arm_off[299]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14404};
  (data->simulationInfo->realParameter[1805] /* theta[299] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */)) + (data->simulationInfo->realParameter[301] /* arm_off[299] PARAM */);
  TRACE_POP
}

/*
equation index: 14405
type: SIMPLE_ASSIGN
arm_off[298] = 3.7322120724646743 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14405};
  (data->simulationInfo->realParameter[300] /* arm_off[298] PARAM */) = (3.7322120724646743) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14406
type: SIMPLE_ASSIGN
theta[298] = pitch * r_init[298] + arm_off[298]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14406(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14406};
  (data->simulationInfo->realParameter[1804] /* theta[298] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */)) + (data->simulationInfo->realParameter[300] /* arm_off[298] PARAM */);
  TRACE_POP
}

/*
equation index: 14407
type: SIMPLE_ASSIGN
arm_off[297] = 3.719645701850315 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14407};
  (data->simulationInfo->realParameter[299] /* arm_off[297] PARAM */) = (3.719645701850315) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14408
type: SIMPLE_ASSIGN
theta[297] = pitch * r_init[297] + arm_off[297]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14408(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14408};
  (data->simulationInfo->realParameter[1803] /* theta[297] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */)) + (data->simulationInfo->realParameter[299] /* arm_off[297] PARAM */);
  TRACE_POP
}

/*
equation index: 14409
type: SIMPLE_ASSIGN
arm_off[296] = 3.7070793312359562 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14409};
  (data->simulationInfo->realParameter[298] /* arm_off[296] PARAM */) = (3.7070793312359562) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14410
type: SIMPLE_ASSIGN
theta[296] = pitch * r_init[296] + arm_off[296]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14410(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14410};
  (data->simulationInfo->realParameter[1802] /* theta[296] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */)) + (data->simulationInfo->realParameter[298] /* arm_off[296] PARAM */);
  TRACE_POP
}

/*
equation index: 14411
type: SIMPLE_ASSIGN
arm_off[295] = 3.694512960621597 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14411};
  (data->simulationInfo->realParameter[297] /* arm_off[295] PARAM */) = (3.694512960621597) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14412
type: SIMPLE_ASSIGN
theta[295] = pitch * r_init[295] + arm_off[295]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14412};
  (data->simulationInfo->realParameter[1801] /* theta[295] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */)) + (data->simulationInfo->realParameter[297] /* arm_off[295] PARAM */);
  TRACE_POP
}

/*
equation index: 14413
type: SIMPLE_ASSIGN
arm_off[294] = 3.6819465900072372 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14413};
  (data->simulationInfo->realParameter[296] /* arm_off[294] PARAM */) = (3.6819465900072372) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14414
type: SIMPLE_ASSIGN
theta[294] = pitch * r_init[294] + arm_off[294]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14414(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14414};
  (data->simulationInfo->realParameter[1800] /* theta[294] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */)) + (data->simulationInfo->realParameter[296] /* arm_off[294] PARAM */);
  TRACE_POP
}

/*
equation index: 14415
type: SIMPLE_ASSIGN
arm_off[293] = 3.6693802193928784 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14415};
  (data->simulationInfo->realParameter[295] /* arm_off[293] PARAM */) = (3.6693802193928784) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14416
type: SIMPLE_ASSIGN
theta[293] = pitch * r_init[293] + arm_off[293]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14416(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14416};
  (data->simulationInfo->realParameter[1799] /* theta[293] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */)) + (data->simulationInfo->realParameter[295] /* arm_off[293] PARAM */);
  TRACE_POP
}

/*
equation index: 14417
type: SIMPLE_ASSIGN
arm_off[292] = 3.656813848778519 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14417};
  (data->simulationInfo->realParameter[294] /* arm_off[292] PARAM */) = (3.656813848778519) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14418
type: SIMPLE_ASSIGN
theta[292] = pitch * r_init[292] + arm_off[292]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14418(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14418};
  (data->simulationInfo->realParameter[1798] /* theta[292] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */)) + (data->simulationInfo->realParameter[294] /* arm_off[292] PARAM */);
  TRACE_POP
}

/*
equation index: 14419
type: SIMPLE_ASSIGN
arm_off[291] = 3.64424747816416 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14419};
  (data->simulationInfo->realParameter[293] /* arm_off[291] PARAM */) = (3.64424747816416) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14420
type: SIMPLE_ASSIGN
theta[291] = pitch * r_init[291] + arm_off[291]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14420};
  (data->simulationInfo->realParameter[1797] /* theta[291] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */)) + (data->simulationInfo->realParameter[293] /* arm_off[291] PARAM */);
  TRACE_POP
}

/*
equation index: 14421
type: SIMPLE_ASSIGN
arm_off[290] = 3.631681107549801 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14421};
  (data->simulationInfo->realParameter[292] /* arm_off[290] PARAM */) = (3.631681107549801) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14422
type: SIMPLE_ASSIGN
theta[290] = pitch * r_init[290] + arm_off[290]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14422(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14422};
  (data->simulationInfo->realParameter[1796] /* theta[290] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */)) + (data->simulationInfo->realParameter[292] /* arm_off[290] PARAM */);
  TRACE_POP
}

/*
equation index: 14423
type: SIMPLE_ASSIGN
arm_off[289] = 3.6191147369354417 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14423};
  (data->simulationInfo->realParameter[291] /* arm_off[289] PARAM */) = (3.6191147369354417) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14424
type: SIMPLE_ASSIGN
theta[289] = pitch * r_init[289] + arm_off[289]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14424(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14424};
  (data->simulationInfo->realParameter[1795] /* theta[289] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */)) + (data->simulationInfo->realParameter[291] /* arm_off[289] PARAM */);
  TRACE_POP
}

/*
equation index: 14425
type: SIMPLE_ASSIGN
arm_off[288] = 3.6065483663210824 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14425};
  (data->simulationInfo->realParameter[290] /* arm_off[288] PARAM */) = (3.6065483663210824) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14426
type: SIMPLE_ASSIGN
theta[288] = pitch * r_init[288] + arm_off[288]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14426(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14426};
  (data->simulationInfo->realParameter[1794] /* theta[288] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */)) + (data->simulationInfo->realParameter[290] /* arm_off[288] PARAM */);
  TRACE_POP
}

/*
equation index: 14427
type: SIMPLE_ASSIGN
arm_off[287] = 3.5939819957067236 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14427};
  (data->simulationInfo->realParameter[289] /* arm_off[287] PARAM */) = (3.5939819957067236) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14428
type: SIMPLE_ASSIGN
theta[287] = pitch * r_init[287] + arm_off[287]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14428};
  (data->simulationInfo->realParameter[1793] /* theta[287] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */)) + (data->simulationInfo->realParameter[289] /* arm_off[287] PARAM */);
  TRACE_POP
}

/*
equation index: 14429
type: SIMPLE_ASSIGN
arm_off[286] = 3.5814156250923643 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14429};
  (data->simulationInfo->realParameter[288] /* arm_off[286] PARAM */) = (3.5814156250923643) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14430
type: SIMPLE_ASSIGN
theta[286] = pitch * r_init[286] + arm_off[286]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14430(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14430};
  (data->simulationInfo->realParameter[1792] /* theta[286] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */)) + (data->simulationInfo->realParameter[288] /* arm_off[286] PARAM */);
  TRACE_POP
}

/*
equation index: 14431
type: SIMPLE_ASSIGN
arm_off[285] = 3.568849254478005 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14431};
  (data->simulationInfo->realParameter[287] /* arm_off[285] PARAM */) = (3.568849254478005) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14432
type: SIMPLE_ASSIGN
theta[285] = pitch * r_init[285] + arm_off[285]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14432(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14432};
  (data->simulationInfo->realParameter[1791] /* theta[285] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */)) + (data->simulationInfo->realParameter[287] /* arm_off[285] PARAM */);
  TRACE_POP
}

/*
equation index: 14433
type: SIMPLE_ASSIGN
arm_off[284] = 3.556282883863646 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14433};
  (data->simulationInfo->realParameter[286] /* arm_off[284] PARAM */) = (3.556282883863646) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14434
type: SIMPLE_ASSIGN
theta[284] = pitch * r_init[284] + arm_off[284]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14434(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14434};
  (data->simulationInfo->realParameter[1790] /* theta[284] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */)) + (data->simulationInfo->realParameter[286] /* arm_off[284] PARAM */);
  TRACE_POP
}

/*
equation index: 14435
type: SIMPLE_ASSIGN
arm_off[283] = 3.5437165132492865 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14435};
  (data->simulationInfo->realParameter[285] /* arm_off[283] PARAM */) = (3.5437165132492865) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14436
type: SIMPLE_ASSIGN
theta[283] = pitch * r_init[283] + arm_off[283]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14436};
  (data->simulationInfo->realParameter[1789] /* theta[283] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */)) + (data->simulationInfo->realParameter[285] /* arm_off[283] PARAM */);
  TRACE_POP
}

/*
equation index: 14437
type: SIMPLE_ASSIGN
arm_off[282] = 3.531150142634927 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14437};
  (data->simulationInfo->realParameter[284] /* arm_off[282] PARAM */) = (3.531150142634927) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14438
type: SIMPLE_ASSIGN
theta[282] = pitch * r_init[282] + arm_off[282]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14438(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14438};
  (data->simulationInfo->realParameter[1788] /* theta[282] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */)) + (data->simulationInfo->realParameter[284] /* arm_off[282] PARAM */);
  TRACE_POP
}

/*
equation index: 14439
type: SIMPLE_ASSIGN
arm_off[281] = 3.5185837720205684 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14439};
  (data->simulationInfo->realParameter[283] /* arm_off[281] PARAM */) = (3.5185837720205684) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14440
type: SIMPLE_ASSIGN
theta[281] = pitch * r_init[281] + arm_off[281]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14440(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14440};
  (data->simulationInfo->realParameter[1787] /* theta[281] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */)) + (data->simulationInfo->realParameter[283] /* arm_off[281] PARAM */);
  TRACE_POP
}

/*
equation index: 14441
type: SIMPLE_ASSIGN
arm_off[280] = 3.506017401406209 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14441};
  (data->simulationInfo->realParameter[282] /* arm_off[280] PARAM */) = (3.506017401406209) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14442
type: SIMPLE_ASSIGN
theta[280] = pitch * r_init[280] + arm_off[280]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14442(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14442};
  (data->simulationInfo->realParameter[1786] /* theta[280] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */)) + (data->simulationInfo->realParameter[282] /* arm_off[280] PARAM */);
  TRACE_POP
}

/*
equation index: 14443
type: SIMPLE_ASSIGN
arm_off[279] = 3.49345103079185 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14443};
  (data->simulationInfo->realParameter[281] /* arm_off[279] PARAM */) = (3.49345103079185) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14444
type: SIMPLE_ASSIGN
theta[279] = pitch * r_init[279] + arm_off[279]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14444};
  (data->simulationInfo->realParameter[1785] /* theta[279] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */)) + (data->simulationInfo->realParameter[281] /* arm_off[279] PARAM */);
  TRACE_POP
}

/*
equation index: 14445
type: SIMPLE_ASSIGN
arm_off[278] = 3.480884660177491 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14445};
  (data->simulationInfo->realParameter[280] /* arm_off[278] PARAM */) = (3.480884660177491) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14446
type: SIMPLE_ASSIGN
theta[278] = pitch * r_init[278] + arm_off[278]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14446(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14446};
  (data->simulationInfo->realParameter[1784] /* theta[278] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */)) + (data->simulationInfo->realParameter[280] /* arm_off[278] PARAM */);
  TRACE_POP
}

/*
equation index: 14447
type: SIMPLE_ASSIGN
arm_off[277] = 3.4683182895631317 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14447};
  (data->simulationInfo->realParameter[279] /* arm_off[277] PARAM */) = (3.4683182895631317) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14448
type: SIMPLE_ASSIGN
theta[277] = pitch * r_init[277] + arm_off[277]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14448(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14448};
  (data->simulationInfo->realParameter[1783] /* theta[277] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */)) + (data->simulationInfo->realParameter[279] /* arm_off[277] PARAM */);
  TRACE_POP
}

/*
equation index: 14449
type: SIMPLE_ASSIGN
arm_off[276] = 3.4557519189487724 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14449};
  (data->simulationInfo->realParameter[278] /* arm_off[276] PARAM */) = (3.4557519189487724) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14450
type: SIMPLE_ASSIGN
theta[276] = pitch * r_init[276] + arm_off[276]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14450(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14450};
  (data->simulationInfo->realParameter[1782] /* theta[276] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */)) + (data->simulationInfo->realParameter[278] /* arm_off[276] PARAM */);
  TRACE_POP
}

/*
equation index: 14451
type: SIMPLE_ASSIGN
arm_off[275] = 3.4431855483344136 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14451};
  (data->simulationInfo->realParameter[277] /* arm_off[275] PARAM */) = (3.4431855483344136) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14452
type: SIMPLE_ASSIGN
theta[275] = pitch * r_init[275] + arm_off[275]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14452};
  (data->simulationInfo->realParameter[1781] /* theta[275] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */)) + (data->simulationInfo->realParameter[277] /* arm_off[275] PARAM */);
  TRACE_POP
}

/*
equation index: 14453
type: SIMPLE_ASSIGN
arm_off[274] = 3.4306191777200543 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14453};
  (data->simulationInfo->realParameter[276] /* arm_off[274] PARAM */) = (3.4306191777200543) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14454
type: SIMPLE_ASSIGN
theta[274] = pitch * r_init[274] + arm_off[274]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14454(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14454};
  (data->simulationInfo->realParameter[1780] /* theta[274] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */)) + (data->simulationInfo->realParameter[276] /* arm_off[274] PARAM */);
  TRACE_POP
}

/*
equation index: 14455
type: SIMPLE_ASSIGN
arm_off[273] = 3.418052807105695 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14455};
  (data->simulationInfo->realParameter[275] /* arm_off[273] PARAM */) = (3.418052807105695) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14456
type: SIMPLE_ASSIGN
theta[273] = pitch * r_init[273] + arm_off[273]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14456(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14456};
  (data->simulationInfo->realParameter[1779] /* theta[273] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */)) + (data->simulationInfo->realParameter[275] /* arm_off[273] PARAM */);
  TRACE_POP
}

/*
equation index: 14457
type: SIMPLE_ASSIGN
arm_off[272] = 3.4054864364913358 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14457};
  (data->simulationInfo->realParameter[274] /* arm_off[272] PARAM */) = (3.4054864364913358) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14458
type: SIMPLE_ASSIGN
theta[272] = pitch * r_init[272] + arm_off[272]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14458(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14458};
  (data->simulationInfo->realParameter[1778] /* theta[272] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */)) + (data->simulationInfo->realParameter[274] /* arm_off[272] PARAM */);
  TRACE_POP
}

/*
equation index: 14459
type: SIMPLE_ASSIGN
arm_off[271] = 3.3929200658769765 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14459};
  (data->simulationInfo->realParameter[273] /* arm_off[271] PARAM */) = (3.3929200658769765) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14460
type: SIMPLE_ASSIGN
theta[271] = pitch * r_init[271] + arm_off[271]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14460};
  (data->simulationInfo->realParameter[1777] /* theta[271] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */)) + (data->simulationInfo->realParameter[273] /* arm_off[271] PARAM */);
  TRACE_POP
}

/*
equation index: 14461
type: SIMPLE_ASSIGN
arm_off[270] = 3.380353695262617 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14461};
  (data->simulationInfo->realParameter[272] /* arm_off[270] PARAM */) = (3.380353695262617) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14462
type: SIMPLE_ASSIGN
theta[270] = pitch * r_init[270] + arm_off[270]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14462(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14462};
  (data->simulationInfo->realParameter[1776] /* theta[270] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */)) + (data->simulationInfo->realParameter[272] /* arm_off[270] PARAM */);
  TRACE_POP
}

/*
equation index: 14463
type: SIMPLE_ASSIGN
arm_off[269] = 3.3677873246482584 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14463};
  (data->simulationInfo->realParameter[271] /* arm_off[269] PARAM */) = (3.3677873246482584) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14464
type: SIMPLE_ASSIGN
theta[269] = pitch * r_init[269] + arm_off[269]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14464(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14464};
  (data->simulationInfo->realParameter[1775] /* theta[269] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */)) + (data->simulationInfo->realParameter[271] /* arm_off[269] PARAM */);
  TRACE_POP
}

/*
equation index: 14465
type: SIMPLE_ASSIGN
arm_off[268] = 3.355220954033899 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14465};
  (data->simulationInfo->realParameter[270] /* arm_off[268] PARAM */) = (3.355220954033899) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14466
type: SIMPLE_ASSIGN
theta[268] = pitch * r_init[268] + arm_off[268]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14466(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14466};
  (data->simulationInfo->realParameter[1774] /* theta[268] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */)) + (data->simulationInfo->realParameter[270] /* arm_off[268] PARAM */);
  TRACE_POP
}

/*
equation index: 14467
type: SIMPLE_ASSIGN
arm_off[267] = 3.34265458341954 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14467};
  (data->simulationInfo->realParameter[269] /* arm_off[267] PARAM */) = (3.34265458341954) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14468
type: SIMPLE_ASSIGN
theta[267] = pitch * r_init[267] + arm_off[267]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14468};
  (data->simulationInfo->realParameter[1773] /* theta[267] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */)) + (data->simulationInfo->realParameter[269] /* arm_off[267] PARAM */);
  TRACE_POP
}

/*
equation index: 14469
type: SIMPLE_ASSIGN
arm_off[266] = 3.330088212805181 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14469};
  (data->simulationInfo->realParameter[268] /* arm_off[266] PARAM */) = (3.330088212805181) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14470
type: SIMPLE_ASSIGN
theta[266] = pitch * r_init[266] + arm_off[266]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14470(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14470};
  (data->simulationInfo->realParameter[1772] /* theta[266] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */)) + (data->simulationInfo->realParameter[268] /* arm_off[266] PARAM */);
  TRACE_POP
}

/*
equation index: 14471
type: SIMPLE_ASSIGN
arm_off[265] = 3.3175218421908217 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14471};
  (data->simulationInfo->realParameter[267] /* arm_off[265] PARAM */) = (3.3175218421908217) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14472
type: SIMPLE_ASSIGN
theta[265] = pitch * r_init[265] + arm_off[265]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14472(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14472};
  (data->simulationInfo->realParameter[1771] /* theta[265] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */)) + (data->simulationInfo->realParameter[267] /* arm_off[265] PARAM */);
  TRACE_POP
}

/*
equation index: 14473
type: SIMPLE_ASSIGN
arm_off[264] = 3.3049554715764624 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14473};
  (data->simulationInfo->realParameter[266] /* arm_off[264] PARAM */) = (3.3049554715764624) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14474
type: SIMPLE_ASSIGN
theta[264] = pitch * r_init[264] + arm_off[264]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14474(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14474};
  (data->simulationInfo->realParameter[1770] /* theta[264] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */)) + (data->simulationInfo->realParameter[266] /* arm_off[264] PARAM */);
  TRACE_POP
}

/*
equation index: 14475
type: SIMPLE_ASSIGN
arm_off[263] = 3.2923891009621036 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14475};
  (data->simulationInfo->realParameter[265] /* arm_off[263] PARAM */) = (3.2923891009621036) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14476
type: SIMPLE_ASSIGN
theta[263] = pitch * r_init[263] + arm_off[263]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14476};
  (data->simulationInfo->realParameter[1769] /* theta[263] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */)) + (data->simulationInfo->realParameter[265] /* arm_off[263] PARAM */);
  TRACE_POP
}

/*
equation index: 14477
type: SIMPLE_ASSIGN
arm_off[262] = 3.279822730347744 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14477};
  (data->simulationInfo->realParameter[264] /* arm_off[262] PARAM */) = (3.279822730347744) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14478
type: SIMPLE_ASSIGN
theta[262] = pitch * r_init[262] + arm_off[262]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14478(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14478};
  (data->simulationInfo->realParameter[1768] /* theta[262] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */)) + (data->simulationInfo->realParameter[264] /* arm_off[262] PARAM */);
  TRACE_POP
}

/*
equation index: 14479
type: SIMPLE_ASSIGN
arm_off[261] = 3.2672563597333846 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14479};
  (data->simulationInfo->realParameter[263] /* arm_off[261] PARAM */) = (3.2672563597333846) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14480
type: SIMPLE_ASSIGN
theta[261] = pitch * r_init[261] + arm_off[261]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14480(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14480};
  (data->simulationInfo->realParameter[1767] /* theta[261] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */)) + (data->simulationInfo->realParameter[263] /* arm_off[261] PARAM */);
  TRACE_POP
}

/*
equation index: 14481
type: SIMPLE_ASSIGN
arm_off[260] = 3.2546899891190257 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14481};
  (data->simulationInfo->realParameter[262] /* arm_off[260] PARAM */) = (3.2546899891190257) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14482
type: SIMPLE_ASSIGN
theta[260] = pitch * r_init[260] + arm_off[260]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14482(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14482};
  (data->simulationInfo->realParameter[1766] /* theta[260] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */)) + (data->simulationInfo->realParameter[262] /* arm_off[260] PARAM */);
  TRACE_POP
}

/*
equation index: 14483
type: SIMPLE_ASSIGN
arm_off[259] = 3.2421236185046665 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14483};
  (data->simulationInfo->realParameter[261] /* arm_off[259] PARAM */) = (3.2421236185046665) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14484
type: SIMPLE_ASSIGN
theta[259] = pitch * r_init[259] + arm_off[259]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14484};
  (data->simulationInfo->realParameter[1765] /* theta[259] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */)) + (data->simulationInfo->realParameter[261] /* arm_off[259] PARAM */);
  TRACE_POP
}

/*
equation index: 14485
type: SIMPLE_ASSIGN
arm_off[258] = 3.229557247890307 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14485};
  (data->simulationInfo->realParameter[260] /* arm_off[258] PARAM */) = (3.229557247890307) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14486
type: SIMPLE_ASSIGN
theta[258] = pitch * r_init[258] + arm_off[258]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14486(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14486};
  (data->simulationInfo->realParameter[1764] /* theta[258] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */)) + (data->simulationInfo->realParameter[260] /* arm_off[258] PARAM */);
  TRACE_POP
}

/*
equation index: 14487
type: SIMPLE_ASSIGN
arm_off[257] = 3.2169908772759483 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14487};
  (data->simulationInfo->realParameter[259] /* arm_off[257] PARAM */) = (3.2169908772759483) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14488
type: SIMPLE_ASSIGN
theta[257] = pitch * r_init[257] + arm_off[257]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14488(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14488};
  (data->simulationInfo->realParameter[1763] /* theta[257] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */)) + (data->simulationInfo->realParameter[259] /* arm_off[257] PARAM */);
  TRACE_POP
}

/*
equation index: 14489
type: SIMPLE_ASSIGN
arm_off[256] = 3.204424506661589 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14489};
  (data->simulationInfo->realParameter[258] /* arm_off[256] PARAM */) = (3.204424506661589) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14490
type: SIMPLE_ASSIGN
theta[256] = pitch * r_init[256] + arm_off[256]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14490(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14490};
  (data->simulationInfo->realParameter[1762] /* theta[256] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */)) + (data->simulationInfo->realParameter[258] /* arm_off[256] PARAM */);
  TRACE_POP
}

/*
equation index: 14491
type: SIMPLE_ASSIGN
arm_off[255] = 3.19185813604723 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14491};
  (data->simulationInfo->realParameter[257] /* arm_off[255] PARAM */) = (3.19185813604723) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14492
type: SIMPLE_ASSIGN
theta[255] = pitch * r_init[255] + arm_off[255]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14492};
  (data->simulationInfo->realParameter[1761] /* theta[255] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */)) + (data->simulationInfo->realParameter[257] /* arm_off[255] PARAM */);
  TRACE_POP
}

/*
equation index: 14493
type: SIMPLE_ASSIGN
arm_off[254] = 3.179291765432871 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14493};
  (data->simulationInfo->realParameter[256] /* arm_off[254] PARAM */) = (3.179291765432871) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14494
type: SIMPLE_ASSIGN
theta[254] = pitch * r_init[254] + arm_off[254]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14494(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14494};
  (data->simulationInfo->realParameter[1760] /* theta[254] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */)) + (data->simulationInfo->realParameter[256] /* arm_off[254] PARAM */);
  TRACE_POP
}

/*
equation index: 14495
type: SIMPLE_ASSIGN
arm_off[253] = 3.1667253948185117 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14495};
  (data->simulationInfo->realParameter[255] /* arm_off[253] PARAM */) = (3.1667253948185117) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14496
type: SIMPLE_ASSIGN
theta[253] = pitch * r_init[253] + arm_off[253]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14496(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14496};
  (data->simulationInfo->realParameter[1759] /* theta[253] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */)) + (data->simulationInfo->realParameter[255] /* arm_off[253] PARAM */);
  TRACE_POP
}

/*
equation index: 14497
type: SIMPLE_ASSIGN
arm_off[252] = 3.1541590242041524 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14497};
  (data->simulationInfo->realParameter[254] /* arm_off[252] PARAM */) = (3.1541590242041524) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14498
type: SIMPLE_ASSIGN
theta[252] = pitch * r_init[252] + arm_off[252]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14498(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14498};
  (data->simulationInfo->realParameter[1758] /* theta[252] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */)) + (data->simulationInfo->realParameter[254] /* arm_off[252] PARAM */);
  TRACE_POP
}

/*
equation index: 14499
type: SIMPLE_ASSIGN
arm_off[251] = 3.141592653589793 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14499};
  (data->simulationInfo->realParameter[253] /* arm_off[251] PARAM */) = (3.141592653589793) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14500
type: SIMPLE_ASSIGN
theta[251] = pitch * r_init[251] + arm_off[251]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14500};
  (data->simulationInfo->realParameter[1757] /* theta[251] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */)) + (data->simulationInfo->realParameter[253] /* arm_off[251] PARAM */);
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_updateBoundParameters_2(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_14001(data, threadData);
  SpiralGalaxy_eqFunction_14002(data, threadData);
  SpiralGalaxy_eqFunction_14003(data, threadData);
  SpiralGalaxy_eqFunction_14004(data, threadData);
  SpiralGalaxy_eqFunction_14005(data, threadData);
  SpiralGalaxy_eqFunction_14006(data, threadData);
  SpiralGalaxy_eqFunction_14007(data, threadData);
  SpiralGalaxy_eqFunction_14008(data, threadData);
  SpiralGalaxy_eqFunction_14009(data, threadData);
  SpiralGalaxy_eqFunction_14010(data, threadData);
  SpiralGalaxy_eqFunction_14011(data, threadData);
  SpiralGalaxy_eqFunction_14012(data, threadData);
  SpiralGalaxy_eqFunction_14013(data, threadData);
  SpiralGalaxy_eqFunction_14014(data, threadData);
  SpiralGalaxy_eqFunction_14015(data, threadData);
  SpiralGalaxy_eqFunction_14016(data, threadData);
  SpiralGalaxy_eqFunction_14017(data, threadData);
  SpiralGalaxy_eqFunction_14018(data, threadData);
  SpiralGalaxy_eqFunction_14019(data, threadData);
  SpiralGalaxy_eqFunction_14020(data, threadData);
  SpiralGalaxy_eqFunction_14021(data, threadData);
  SpiralGalaxy_eqFunction_14022(data, threadData);
  SpiralGalaxy_eqFunction_14023(data, threadData);
  SpiralGalaxy_eqFunction_14024(data, threadData);
  SpiralGalaxy_eqFunction_14025(data, threadData);
  SpiralGalaxy_eqFunction_14026(data, threadData);
  SpiralGalaxy_eqFunction_14027(data, threadData);
  SpiralGalaxy_eqFunction_14028(data, threadData);
  SpiralGalaxy_eqFunction_14029(data, threadData);
  SpiralGalaxy_eqFunction_14030(data, threadData);
  SpiralGalaxy_eqFunction_14031(data, threadData);
  SpiralGalaxy_eqFunction_14032(data, threadData);
  SpiralGalaxy_eqFunction_14033(data, threadData);
  SpiralGalaxy_eqFunction_14034(data, threadData);
  SpiralGalaxy_eqFunction_14035(data, threadData);
  SpiralGalaxy_eqFunction_14036(data, threadData);
  SpiralGalaxy_eqFunction_14037(data, threadData);
  SpiralGalaxy_eqFunction_14038(data, threadData);
  SpiralGalaxy_eqFunction_14039(data, threadData);
  SpiralGalaxy_eqFunction_14040(data, threadData);
  SpiralGalaxy_eqFunction_14041(data, threadData);
  SpiralGalaxy_eqFunction_14042(data, threadData);
  SpiralGalaxy_eqFunction_14043(data, threadData);
  SpiralGalaxy_eqFunction_14044(data, threadData);
  SpiralGalaxy_eqFunction_14045(data, threadData);
  SpiralGalaxy_eqFunction_14046(data, threadData);
  SpiralGalaxy_eqFunction_14047(data, threadData);
  SpiralGalaxy_eqFunction_14048(data, threadData);
  SpiralGalaxy_eqFunction_14049(data, threadData);
  SpiralGalaxy_eqFunction_14050(data, threadData);
  SpiralGalaxy_eqFunction_14051(data, threadData);
  SpiralGalaxy_eqFunction_14052(data, threadData);
  SpiralGalaxy_eqFunction_14053(data, threadData);
  SpiralGalaxy_eqFunction_14054(data, threadData);
  SpiralGalaxy_eqFunction_14055(data, threadData);
  SpiralGalaxy_eqFunction_14056(data, threadData);
  SpiralGalaxy_eqFunction_14057(data, threadData);
  SpiralGalaxy_eqFunction_14058(data, threadData);
  SpiralGalaxy_eqFunction_14059(data, threadData);
  SpiralGalaxy_eqFunction_14060(data, threadData);
  SpiralGalaxy_eqFunction_14061(data, threadData);
  SpiralGalaxy_eqFunction_14062(data, threadData);
  SpiralGalaxy_eqFunction_14063(data, threadData);
  SpiralGalaxy_eqFunction_14064(data, threadData);
  SpiralGalaxy_eqFunction_14065(data, threadData);
  SpiralGalaxy_eqFunction_14066(data, threadData);
  SpiralGalaxy_eqFunction_14067(data, threadData);
  SpiralGalaxy_eqFunction_14068(data, threadData);
  SpiralGalaxy_eqFunction_14069(data, threadData);
  SpiralGalaxy_eqFunction_14070(data, threadData);
  SpiralGalaxy_eqFunction_14071(data, threadData);
  SpiralGalaxy_eqFunction_14072(data, threadData);
  SpiralGalaxy_eqFunction_14073(data, threadData);
  SpiralGalaxy_eqFunction_14074(data, threadData);
  SpiralGalaxy_eqFunction_14075(data, threadData);
  SpiralGalaxy_eqFunction_14076(data, threadData);
  SpiralGalaxy_eqFunction_14077(data, threadData);
  SpiralGalaxy_eqFunction_14078(data, threadData);
  SpiralGalaxy_eqFunction_14079(data, threadData);
  SpiralGalaxy_eqFunction_14080(data, threadData);
  SpiralGalaxy_eqFunction_14081(data, threadData);
  SpiralGalaxy_eqFunction_14082(data, threadData);
  SpiralGalaxy_eqFunction_14083(data, threadData);
  SpiralGalaxy_eqFunction_14084(data, threadData);
  SpiralGalaxy_eqFunction_14085(data, threadData);
  SpiralGalaxy_eqFunction_14086(data, threadData);
  SpiralGalaxy_eqFunction_14087(data, threadData);
  SpiralGalaxy_eqFunction_14088(data, threadData);
  SpiralGalaxy_eqFunction_14089(data, threadData);
  SpiralGalaxy_eqFunction_14090(data, threadData);
  SpiralGalaxy_eqFunction_14091(data, threadData);
  SpiralGalaxy_eqFunction_14092(data, threadData);
  SpiralGalaxy_eqFunction_14093(data, threadData);
  SpiralGalaxy_eqFunction_14094(data, threadData);
  SpiralGalaxy_eqFunction_14095(data, threadData);
  SpiralGalaxy_eqFunction_14096(data, threadData);
  SpiralGalaxy_eqFunction_14097(data, threadData);
  SpiralGalaxy_eqFunction_14098(data, threadData);
  SpiralGalaxy_eqFunction_14099(data, threadData);
  SpiralGalaxy_eqFunction_14100(data, threadData);
  SpiralGalaxy_eqFunction_14101(data, threadData);
  SpiralGalaxy_eqFunction_14102(data, threadData);
  SpiralGalaxy_eqFunction_14103(data, threadData);
  SpiralGalaxy_eqFunction_14104(data, threadData);
  SpiralGalaxy_eqFunction_14105(data, threadData);
  SpiralGalaxy_eqFunction_14106(data, threadData);
  SpiralGalaxy_eqFunction_14107(data, threadData);
  SpiralGalaxy_eqFunction_14108(data, threadData);
  SpiralGalaxy_eqFunction_14109(data, threadData);
  SpiralGalaxy_eqFunction_14110(data, threadData);
  SpiralGalaxy_eqFunction_14111(data, threadData);
  SpiralGalaxy_eqFunction_14112(data, threadData);
  SpiralGalaxy_eqFunction_14113(data, threadData);
  SpiralGalaxy_eqFunction_14114(data, threadData);
  SpiralGalaxy_eqFunction_14115(data, threadData);
  SpiralGalaxy_eqFunction_14116(data, threadData);
  SpiralGalaxy_eqFunction_14117(data, threadData);
  SpiralGalaxy_eqFunction_14118(data, threadData);
  SpiralGalaxy_eqFunction_14119(data, threadData);
  SpiralGalaxy_eqFunction_14120(data, threadData);
  SpiralGalaxy_eqFunction_14121(data, threadData);
  SpiralGalaxy_eqFunction_14122(data, threadData);
  SpiralGalaxy_eqFunction_14123(data, threadData);
  SpiralGalaxy_eqFunction_14124(data, threadData);
  SpiralGalaxy_eqFunction_14125(data, threadData);
  SpiralGalaxy_eqFunction_14126(data, threadData);
  SpiralGalaxy_eqFunction_14127(data, threadData);
  SpiralGalaxy_eqFunction_14128(data, threadData);
  SpiralGalaxy_eqFunction_14129(data, threadData);
  SpiralGalaxy_eqFunction_14130(data, threadData);
  SpiralGalaxy_eqFunction_14131(data, threadData);
  SpiralGalaxy_eqFunction_14132(data, threadData);
  SpiralGalaxy_eqFunction_14133(data, threadData);
  SpiralGalaxy_eqFunction_14134(data, threadData);
  SpiralGalaxy_eqFunction_14135(data, threadData);
  SpiralGalaxy_eqFunction_14136(data, threadData);
  SpiralGalaxy_eqFunction_14137(data, threadData);
  SpiralGalaxy_eqFunction_14138(data, threadData);
  SpiralGalaxy_eqFunction_14139(data, threadData);
  SpiralGalaxy_eqFunction_14140(data, threadData);
  SpiralGalaxy_eqFunction_14141(data, threadData);
  SpiralGalaxy_eqFunction_14142(data, threadData);
  SpiralGalaxy_eqFunction_14143(data, threadData);
  SpiralGalaxy_eqFunction_14144(data, threadData);
  SpiralGalaxy_eqFunction_14145(data, threadData);
  SpiralGalaxy_eqFunction_14146(data, threadData);
  SpiralGalaxy_eqFunction_14147(data, threadData);
  SpiralGalaxy_eqFunction_14148(data, threadData);
  SpiralGalaxy_eqFunction_14149(data, threadData);
  SpiralGalaxy_eqFunction_14150(data, threadData);
  SpiralGalaxy_eqFunction_14151(data, threadData);
  SpiralGalaxy_eqFunction_14152(data, threadData);
  SpiralGalaxy_eqFunction_14153(data, threadData);
  SpiralGalaxy_eqFunction_14154(data, threadData);
  SpiralGalaxy_eqFunction_14155(data, threadData);
  SpiralGalaxy_eqFunction_14156(data, threadData);
  SpiralGalaxy_eqFunction_14157(data, threadData);
  SpiralGalaxy_eqFunction_14158(data, threadData);
  SpiralGalaxy_eqFunction_14159(data, threadData);
  SpiralGalaxy_eqFunction_14160(data, threadData);
  SpiralGalaxy_eqFunction_14161(data, threadData);
  SpiralGalaxy_eqFunction_14162(data, threadData);
  SpiralGalaxy_eqFunction_14163(data, threadData);
  SpiralGalaxy_eqFunction_14164(data, threadData);
  SpiralGalaxy_eqFunction_14165(data, threadData);
  SpiralGalaxy_eqFunction_14166(data, threadData);
  SpiralGalaxy_eqFunction_14167(data, threadData);
  SpiralGalaxy_eqFunction_14168(data, threadData);
  SpiralGalaxy_eqFunction_14169(data, threadData);
  SpiralGalaxy_eqFunction_14170(data, threadData);
  SpiralGalaxy_eqFunction_14171(data, threadData);
  SpiralGalaxy_eqFunction_14172(data, threadData);
  SpiralGalaxy_eqFunction_14173(data, threadData);
  SpiralGalaxy_eqFunction_14174(data, threadData);
  SpiralGalaxy_eqFunction_14175(data, threadData);
  SpiralGalaxy_eqFunction_14176(data, threadData);
  SpiralGalaxy_eqFunction_14177(data, threadData);
  SpiralGalaxy_eqFunction_14178(data, threadData);
  SpiralGalaxy_eqFunction_14179(data, threadData);
  SpiralGalaxy_eqFunction_14180(data, threadData);
  SpiralGalaxy_eqFunction_14181(data, threadData);
  SpiralGalaxy_eqFunction_14182(data, threadData);
  SpiralGalaxy_eqFunction_14183(data, threadData);
  SpiralGalaxy_eqFunction_14184(data, threadData);
  SpiralGalaxy_eqFunction_14185(data, threadData);
  SpiralGalaxy_eqFunction_14186(data, threadData);
  SpiralGalaxy_eqFunction_14187(data, threadData);
  SpiralGalaxy_eqFunction_14188(data, threadData);
  SpiralGalaxy_eqFunction_14189(data, threadData);
  SpiralGalaxy_eqFunction_14190(data, threadData);
  SpiralGalaxy_eqFunction_14191(data, threadData);
  SpiralGalaxy_eqFunction_14192(data, threadData);
  SpiralGalaxy_eqFunction_14193(data, threadData);
  SpiralGalaxy_eqFunction_14194(data, threadData);
  SpiralGalaxy_eqFunction_14195(data, threadData);
  SpiralGalaxy_eqFunction_14196(data, threadData);
  SpiralGalaxy_eqFunction_14197(data, threadData);
  SpiralGalaxy_eqFunction_14198(data, threadData);
  SpiralGalaxy_eqFunction_14199(data, threadData);
  SpiralGalaxy_eqFunction_14200(data, threadData);
  SpiralGalaxy_eqFunction_14201(data, threadData);
  SpiralGalaxy_eqFunction_14202(data, threadData);
  SpiralGalaxy_eqFunction_14203(data, threadData);
  SpiralGalaxy_eqFunction_14204(data, threadData);
  SpiralGalaxy_eqFunction_14205(data, threadData);
  SpiralGalaxy_eqFunction_14206(data, threadData);
  SpiralGalaxy_eqFunction_14207(data, threadData);
  SpiralGalaxy_eqFunction_14208(data, threadData);
  SpiralGalaxy_eqFunction_14209(data, threadData);
  SpiralGalaxy_eqFunction_14210(data, threadData);
  SpiralGalaxy_eqFunction_14211(data, threadData);
  SpiralGalaxy_eqFunction_14212(data, threadData);
  SpiralGalaxy_eqFunction_14213(data, threadData);
  SpiralGalaxy_eqFunction_14214(data, threadData);
  SpiralGalaxy_eqFunction_14215(data, threadData);
  SpiralGalaxy_eqFunction_14216(data, threadData);
  SpiralGalaxy_eqFunction_14217(data, threadData);
  SpiralGalaxy_eqFunction_14218(data, threadData);
  SpiralGalaxy_eqFunction_14219(data, threadData);
  SpiralGalaxy_eqFunction_14220(data, threadData);
  SpiralGalaxy_eqFunction_14221(data, threadData);
  SpiralGalaxy_eqFunction_14222(data, threadData);
  SpiralGalaxy_eqFunction_14223(data, threadData);
  SpiralGalaxy_eqFunction_14224(data, threadData);
  SpiralGalaxy_eqFunction_14225(data, threadData);
  SpiralGalaxy_eqFunction_14226(data, threadData);
  SpiralGalaxy_eqFunction_14227(data, threadData);
  SpiralGalaxy_eqFunction_14228(data, threadData);
  SpiralGalaxy_eqFunction_14229(data, threadData);
  SpiralGalaxy_eqFunction_14230(data, threadData);
  SpiralGalaxy_eqFunction_14231(data, threadData);
  SpiralGalaxy_eqFunction_14232(data, threadData);
  SpiralGalaxy_eqFunction_14233(data, threadData);
  SpiralGalaxy_eqFunction_14234(data, threadData);
  SpiralGalaxy_eqFunction_14235(data, threadData);
  SpiralGalaxy_eqFunction_14236(data, threadData);
  SpiralGalaxy_eqFunction_14237(data, threadData);
  SpiralGalaxy_eqFunction_14238(data, threadData);
  SpiralGalaxy_eqFunction_14239(data, threadData);
  SpiralGalaxy_eqFunction_14240(data, threadData);
  SpiralGalaxy_eqFunction_14241(data, threadData);
  SpiralGalaxy_eqFunction_14242(data, threadData);
  SpiralGalaxy_eqFunction_14243(data, threadData);
  SpiralGalaxy_eqFunction_14244(data, threadData);
  SpiralGalaxy_eqFunction_14245(data, threadData);
  SpiralGalaxy_eqFunction_14246(data, threadData);
  SpiralGalaxy_eqFunction_14247(data, threadData);
  SpiralGalaxy_eqFunction_14248(data, threadData);
  SpiralGalaxy_eqFunction_14249(data, threadData);
  SpiralGalaxy_eqFunction_14250(data, threadData);
  SpiralGalaxy_eqFunction_14251(data, threadData);
  SpiralGalaxy_eqFunction_14252(data, threadData);
  SpiralGalaxy_eqFunction_14253(data, threadData);
  SpiralGalaxy_eqFunction_14254(data, threadData);
  SpiralGalaxy_eqFunction_14255(data, threadData);
  SpiralGalaxy_eqFunction_14256(data, threadData);
  SpiralGalaxy_eqFunction_14257(data, threadData);
  SpiralGalaxy_eqFunction_14258(data, threadData);
  SpiralGalaxy_eqFunction_14259(data, threadData);
  SpiralGalaxy_eqFunction_14260(data, threadData);
  SpiralGalaxy_eqFunction_14261(data, threadData);
  SpiralGalaxy_eqFunction_14262(data, threadData);
  SpiralGalaxy_eqFunction_14263(data, threadData);
  SpiralGalaxy_eqFunction_14264(data, threadData);
  SpiralGalaxy_eqFunction_14265(data, threadData);
  SpiralGalaxy_eqFunction_14266(data, threadData);
  SpiralGalaxy_eqFunction_14267(data, threadData);
  SpiralGalaxy_eqFunction_14268(data, threadData);
  SpiralGalaxy_eqFunction_14269(data, threadData);
  SpiralGalaxy_eqFunction_14270(data, threadData);
  SpiralGalaxy_eqFunction_14271(data, threadData);
  SpiralGalaxy_eqFunction_14272(data, threadData);
  SpiralGalaxy_eqFunction_14273(data, threadData);
  SpiralGalaxy_eqFunction_14274(data, threadData);
  SpiralGalaxy_eqFunction_14275(data, threadData);
  SpiralGalaxy_eqFunction_14276(data, threadData);
  SpiralGalaxy_eqFunction_14277(data, threadData);
  SpiralGalaxy_eqFunction_14278(data, threadData);
  SpiralGalaxy_eqFunction_14279(data, threadData);
  SpiralGalaxy_eqFunction_14280(data, threadData);
  SpiralGalaxy_eqFunction_14281(data, threadData);
  SpiralGalaxy_eqFunction_14282(data, threadData);
  SpiralGalaxy_eqFunction_14283(data, threadData);
  SpiralGalaxy_eqFunction_14284(data, threadData);
  SpiralGalaxy_eqFunction_14285(data, threadData);
  SpiralGalaxy_eqFunction_14286(data, threadData);
  SpiralGalaxy_eqFunction_14287(data, threadData);
  SpiralGalaxy_eqFunction_14288(data, threadData);
  SpiralGalaxy_eqFunction_14289(data, threadData);
  SpiralGalaxy_eqFunction_14290(data, threadData);
  SpiralGalaxy_eqFunction_14291(data, threadData);
  SpiralGalaxy_eqFunction_14292(data, threadData);
  SpiralGalaxy_eqFunction_14293(data, threadData);
  SpiralGalaxy_eqFunction_14294(data, threadData);
  SpiralGalaxy_eqFunction_14295(data, threadData);
  SpiralGalaxy_eqFunction_14296(data, threadData);
  SpiralGalaxy_eqFunction_14297(data, threadData);
  SpiralGalaxy_eqFunction_14298(data, threadData);
  SpiralGalaxy_eqFunction_14299(data, threadData);
  SpiralGalaxy_eqFunction_14300(data, threadData);
  SpiralGalaxy_eqFunction_14301(data, threadData);
  SpiralGalaxy_eqFunction_14302(data, threadData);
  SpiralGalaxy_eqFunction_14303(data, threadData);
  SpiralGalaxy_eqFunction_14304(data, threadData);
  SpiralGalaxy_eqFunction_14305(data, threadData);
  SpiralGalaxy_eqFunction_14306(data, threadData);
  SpiralGalaxy_eqFunction_14307(data, threadData);
  SpiralGalaxy_eqFunction_14308(data, threadData);
  SpiralGalaxy_eqFunction_14309(data, threadData);
  SpiralGalaxy_eqFunction_14310(data, threadData);
  SpiralGalaxy_eqFunction_14311(data, threadData);
  SpiralGalaxy_eqFunction_14312(data, threadData);
  SpiralGalaxy_eqFunction_14313(data, threadData);
  SpiralGalaxy_eqFunction_14314(data, threadData);
  SpiralGalaxy_eqFunction_14315(data, threadData);
  SpiralGalaxy_eqFunction_14316(data, threadData);
  SpiralGalaxy_eqFunction_14317(data, threadData);
  SpiralGalaxy_eqFunction_14318(data, threadData);
  SpiralGalaxy_eqFunction_14319(data, threadData);
  SpiralGalaxy_eqFunction_14320(data, threadData);
  SpiralGalaxy_eqFunction_14321(data, threadData);
  SpiralGalaxy_eqFunction_14322(data, threadData);
  SpiralGalaxy_eqFunction_14323(data, threadData);
  SpiralGalaxy_eqFunction_14324(data, threadData);
  SpiralGalaxy_eqFunction_14325(data, threadData);
  SpiralGalaxy_eqFunction_14326(data, threadData);
  SpiralGalaxy_eqFunction_14327(data, threadData);
  SpiralGalaxy_eqFunction_14328(data, threadData);
  SpiralGalaxy_eqFunction_14329(data, threadData);
  SpiralGalaxy_eqFunction_14330(data, threadData);
  SpiralGalaxy_eqFunction_14331(data, threadData);
  SpiralGalaxy_eqFunction_14332(data, threadData);
  SpiralGalaxy_eqFunction_14333(data, threadData);
  SpiralGalaxy_eqFunction_14334(data, threadData);
  SpiralGalaxy_eqFunction_14335(data, threadData);
  SpiralGalaxy_eqFunction_14336(data, threadData);
  SpiralGalaxy_eqFunction_14337(data, threadData);
  SpiralGalaxy_eqFunction_14338(data, threadData);
  SpiralGalaxy_eqFunction_14339(data, threadData);
  SpiralGalaxy_eqFunction_14340(data, threadData);
  SpiralGalaxy_eqFunction_14341(data, threadData);
  SpiralGalaxy_eqFunction_14342(data, threadData);
  SpiralGalaxy_eqFunction_14343(data, threadData);
  SpiralGalaxy_eqFunction_14344(data, threadData);
  SpiralGalaxy_eqFunction_14345(data, threadData);
  SpiralGalaxy_eqFunction_14346(data, threadData);
  SpiralGalaxy_eqFunction_14347(data, threadData);
  SpiralGalaxy_eqFunction_14348(data, threadData);
  SpiralGalaxy_eqFunction_14349(data, threadData);
  SpiralGalaxy_eqFunction_14350(data, threadData);
  SpiralGalaxy_eqFunction_14351(data, threadData);
  SpiralGalaxy_eqFunction_14352(data, threadData);
  SpiralGalaxy_eqFunction_14353(data, threadData);
  SpiralGalaxy_eqFunction_14354(data, threadData);
  SpiralGalaxy_eqFunction_14355(data, threadData);
  SpiralGalaxy_eqFunction_14356(data, threadData);
  SpiralGalaxy_eqFunction_14357(data, threadData);
  SpiralGalaxy_eqFunction_14358(data, threadData);
  SpiralGalaxy_eqFunction_14359(data, threadData);
  SpiralGalaxy_eqFunction_14360(data, threadData);
  SpiralGalaxy_eqFunction_14361(data, threadData);
  SpiralGalaxy_eqFunction_14362(data, threadData);
  SpiralGalaxy_eqFunction_14363(data, threadData);
  SpiralGalaxy_eqFunction_14364(data, threadData);
  SpiralGalaxy_eqFunction_14365(data, threadData);
  SpiralGalaxy_eqFunction_14366(data, threadData);
  SpiralGalaxy_eqFunction_14367(data, threadData);
  SpiralGalaxy_eqFunction_14368(data, threadData);
  SpiralGalaxy_eqFunction_14369(data, threadData);
  SpiralGalaxy_eqFunction_14370(data, threadData);
  SpiralGalaxy_eqFunction_14371(data, threadData);
  SpiralGalaxy_eqFunction_14372(data, threadData);
  SpiralGalaxy_eqFunction_14373(data, threadData);
  SpiralGalaxy_eqFunction_14374(data, threadData);
  SpiralGalaxy_eqFunction_14375(data, threadData);
  SpiralGalaxy_eqFunction_14376(data, threadData);
  SpiralGalaxy_eqFunction_14377(data, threadData);
  SpiralGalaxy_eqFunction_14378(data, threadData);
  SpiralGalaxy_eqFunction_14379(data, threadData);
  SpiralGalaxy_eqFunction_14380(data, threadData);
  SpiralGalaxy_eqFunction_14381(data, threadData);
  SpiralGalaxy_eqFunction_14382(data, threadData);
  SpiralGalaxy_eqFunction_14383(data, threadData);
  SpiralGalaxy_eqFunction_14384(data, threadData);
  SpiralGalaxy_eqFunction_14385(data, threadData);
  SpiralGalaxy_eqFunction_14386(data, threadData);
  SpiralGalaxy_eqFunction_14387(data, threadData);
  SpiralGalaxy_eqFunction_14388(data, threadData);
  SpiralGalaxy_eqFunction_14389(data, threadData);
  SpiralGalaxy_eqFunction_14390(data, threadData);
  SpiralGalaxy_eqFunction_14391(data, threadData);
  SpiralGalaxy_eqFunction_14392(data, threadData);
  SpiralGalaxy_eqFunction_14393(data, threadData);
  SpiralGalaxy_eqFunction_14394(data, threadData);
  SpiralGalaxy_eqFunction_14395(data, threadData);
  SpiralGalaxy_eqFunction_14396(data, threadData);
  SpiralGalaxy_eqFunction_14397(data, threadData);
  SpiralGalaxy_eqFunction_14398(data, threadData);
  SpiralGalaxy_eqFunction_14399(data, threadData);
  SpiralGalaxy_eqFunction_14400(data, threadData);
  SpiralGalaxy_eqFunction_14401(data, threadData);
  SpiralGalaxy_eqFunction_14402(data, threadData);
  SpiralGalaxy_eqFunction_14403(data, threadData);
  SpiralGalaxy_eqFunction_14404(data, threadData);
  SpiralGalaxy_eqFunction_14405(data, threadData);
  SpiralGalaxy_eqFunction_14406(data, threadData);
  SpiralGalaxy_eqFunction_14407(data, threadData);
  SpiralGalaxy_eqFunction_14408(data, threadData);
  SpiralGalaxy_eqFunction_14409(data, threadData);
  SpiralGalaxy_eqFunction_14410(data, threadData);
  SpiralGalaxy_eqFunction_14411(data, threadData);
  SpiralGalaxy_eqFunction_14412(data, threadData);
  SpiralGalaxy_eqFunction_14413(data, threadData);
  SpiralGalaxy_eqFunction_14414(data, threadData);
  SpiralGalaxy_eqFunction_14415(data, threadData);
  SpiralGalaxy_eqFunction_14416(data, threadData);
  SpiralGalaxy_eqFunction_14417(data, threadData);
  SpiralGalaxy_eqFunction_14418(data, threadData);
  SpiralGalaxy_eqFunction_14419(data, threadData);
  SpiralGalaxy_eqFunction_14420(data, threadData);
  SpiralGalaxy_eqFunction_14421(data, threadData);
  SpiralGalaxy_eqFunction_14422(data, threadData);
  SpiralGalaxy_eqFunction_14423(data, threadData);
  SpiralGalaxy_eqFunction_14424(data, threadData);
  SpiralGalaxy_eqFunction_14425(data, threadData);
  SpiralGalaxy_eqFunction_14426(data, threadData);
  SpiralGalaxy_eqFunction_14427(data, threadData);
  SpiralGalaxy_eqFunction_14428(data, threadData);
  SpiralGalaxy_eqFunction_14429(data, threadData);
  SpiralGalaxy_eqFunction_14430(data, threadData);
  SpiralGalaxy_eqFunction_14431(data, threadData);
  SpiralGalaxy_eqFunction_14432(data, threadData);
  SpiralGalaxy_eqFunction_14433(data, threadData);
  SpiralGalaxy_eqFunction_14434(data, threadData);
  SpiralGalaxy_eqFunction_14435(data, threadData);
  SpiralGalaxy_eqFunction_14436(data, threadData);
  SpiralGalaxy_eqFunction_14437(data, threadData);
  SpiralGalaxy_eqFunction_14438(data, threadData);
  SpiralGalaxy_eqFunction_14439(data, threadData);
  SpiralGalaxy_eqFunction_14440(data, threadData);
  SpiralGalaxy_eqFunction_14441(data, threadData);
  SpiralGalaxy_eqFunction_14442(data, threadData);
  SpiralGalaxy_eqFunction_14443(data, threadData);
  SpiralGalaxy_eqFunction_14444(data, threadData);
  SpiralGalaxy_eqFunction_14445(data, threadData);
  SpiralGalaxy_eqFunction_14446(data, threadData);
  SpiralGalaxy_eqFunction_14447(data, threadData);
  SpiralGalaxy_eqFunction_14448(data, threadData);
  SpiralGalaxy_eqFunction_14449(data, threadData);
  SpiralGalaxy_eqFunction_14450(data, threadData);
  SpiralGalaxy_eqFunction_14451(data, threadData);
  SpiralGalaxy_eqFunction_14452(data, threadData);
  SpiralGalaxy_eqFunction_14453(data, threadData);
  SpiralGalaxy_eqFunction_14454(data, threadData);
  SpiralGalaxy_eqFunction_14455(data, threadData);
  SpiralGalaxy_eqFunction_14456(data, threadData);
  SpiralGalaxy_eqFunction_14457(data, threadData);
  SpiralGalaxy_eqFunction_14458(data, threadData);
  SpiralGalaxy_eqFunction_14459(data, threadData);
  SpiralGalaxy_eqFunction_14460(data, threadData);
  SpiralGalaxy_eqFunction_14461(data, threadData);
  SpiralGalaxy_eqFunction_14462(data, threadData);
  SpiralGalaxy_eqFunction_14463(data, threadData);
  SpiralGalaxy_eqFunction_14464(data, threadData);
  SpiralGalaxy_eqFunction_14465(data, threadData);
  SpiralGalaxy_eqFunction_14466(data, threadData);
  SpiralGalaxy_eqFunction_14467(data, threadData);
  SpiralGalaxy_eqFunction_14468(data, threadData);
  SpiralGalaxy_eqFunction_14469(data, threadData);
  SpiralGalaxy_eqFunction_14470(data, threadData);
  SpiralGalaxy_eqFunction_14471(data, threadData);
  SpiralGalaxy_eqFunction_14472(data, threadData);
  SpiralGalaxy_eqFunction_14473(data, threadData);
  SpiralGalaxy_eqFunction_14474(data, threadData);
  SpiralGalaxy_eqFunction_14475(data, threadData);
  SpiralGalaxy_eqFunction_14476(data, threadData);
  SpiralGalaxy_eqFunction_14477(data, threadData);
  SpiralGalaxy_eqFunction_14478(data, threadData);
  SpiralGalaxy_eqFunction_14479(data, threadData);
  SpiralGalaxy_eqFunction_14480(data, threadData);
  SpiralGalaxy_eqFunction_14481(data, threadData);
  SpiralGalaxy_eqFunction_14482(data, threadData);
  SpiralGalaxy_eqFunction_14483(data, threadData);
  SpiralGalaxy_eqFunction_14484(data, threadData);
  SpiralGalaxy_eqFunction_14485(data, threadData);
  SpiralGalaxy_eqFunction_14486(data, threadData);
  SpiralGalaxy_eqFunction_14487(data, threadData);
  SpiralGalaxy_eqFunction_14488(data, threadData);
  SpiralGalaxy_eqFunction_14489(data, threadData);
  SpiralGalaxy_eqFunction_14490(data, threadData);
  SpiralGalaxy_eqFunction_14491(data, threadData);
  SpiralGalaxy_eqFunction_14492(data, threadData);
  SpiralGalaxy_eqFunction_14493(data, threadData);
  SpiralGalaxy_eqFunction_14494(data, threadData);
  SpiralGalaxy_eqFunction_14495(data, threadData);
  SpiralGalaxy_eqFunction_14496(data, threadData);
  SpiralGalaxy_eqFunction_14497(data, threadData);
  SpiralGalaxy_eqFunction_14498(data, threadData);
  SpiralGalaxy_eqFunction_14499(data, threadData);
  SpiralGalaxy_eqFunction_14500(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif