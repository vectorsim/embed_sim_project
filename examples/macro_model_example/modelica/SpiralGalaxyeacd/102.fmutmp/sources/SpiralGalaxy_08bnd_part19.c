#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 14501
type: SIMPLE_ASSIGN
arm_off[250] = 3.129026282975434 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14501};
  (data->simulationInfo->realParameter[252] /* arm_off[250] PARAM */) = (3.129026282975434) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14502
type: SIMPLE_ASSIGN
theta[250] = pitch * r_init[250] + arm_off[250]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14502(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14502};
  (data->simulationInfo->realParameter[1756] /* theta[250] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */)) + (data->simulationInfo->realParameter[252] /* arm_off[250] PARAM */);
  TRACE_POP
}

/*
equation index: 14503
type: SIMPLE_ASSIGN
arm_off[249] = 3.1164599123610746 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14503};
  (data->simulationInfo->realParameter[251] /* arm_off[249] PARAM */) = (3.1164599123610746) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14504
type: SIMPLE_ASSIGN
theta[249] = pitch * r_init[249] + arm_off[249]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14504(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14504};
  (data->simulationInfo->realParameter[1755] /* theta[249] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */)) + (data->simulationInfo->realParameter[251] /* arm_off[249] PARAM */);
  TRACE_POP
}

/*
equation index: 14505
type: SIMPLE_ASSIGN
arm_off[248] = 3.1038935417467157 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14505};
  (data->simulationInfo->realParameter[250] /* arm_off[248] PARAM */) = (3.1038935417467157) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14506
type: SIMPLE_ASSIGN
theta[248] = pitch * r_init[248] + arm_off[248]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14506(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14506};
  (data->simulationInfo->realParameter[1754] /* theta[248] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */)) + (data->simulationInfo->realParameter[250] /* arm_off[248] PARAM */);
  TRACE_POP
}

/*
equation index: 14507
type: SIMPLE_ASSIGN
arm_off[247] = 3.0913271711323564 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14507};
  (data->simulationInfo->realParameter[249] /* arm_off[247] PARAM */) = (3.0913271711323564) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14508
type: SIMPLE_ASSIGN
theta[247] = pitch * r_init[247] + arm_off[247]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14508};
  (data->simulationInfo->realParameter[1753] /* theta[247] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */)) + (data->simulationInfo->realParameter[249] /* arm_off[247] PARAM */);
  TRACE_POP
}

/*
equation index: 14509
type: SIMPLE_ASSIGN
arm_off[246] = 3.078760800517997 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14509};
  (data->simulationInfo->realParameter[248] /* arm_off[246] PARAM */) = (3.078760800517997) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14510
type: SIMPLE_ASSIGN
theta[246] = pitch * r_init[246] + arm_off[246]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14510(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14510};
  (data->simulationInfo->realParameter[1752] /* theta[246] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */)) + (data->simulationInfo->realParameter[248] /* arm_off[246] PARAM */);
  TRACE_POP
}

/*
equation index: 14511
type: SIMPLE_ASSIGN
arm_off[245] = 3.0661944299036383 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14511};
  (data->simulationInfo->realParameter[247] /* arm_off[245] PARAM */) = (3.0661944299036383) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14512
type: SIMPLE_ASSIGN
theta[245] = pitch * r_init[245] + arm_off[245]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14512(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14512};
  (data->simulationInfo->realParameter[1751] /* theta[245] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */)) + (data->simulationInfo->realParameter[247] /* arm_off[245] PARAM */);
  TRACE_POP
}

/*
equation index: 14513
type: SIMPLE_ASSIGN
arm_off[244] = 3.053628059289279 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14513};
  (data->simulationInfo->realParameter[246] /* arm_off[244] PARAM */) = (3.053628059289279) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14514
type: SIMPLE_ASSIGN
theta[244] = pitch * r_init[244] + arm_off[244]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14514(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14514};
  (data->simulationInfo->realParameter[1750] /* theta[244] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */)) + (data->simulationInfo->realParameter[246] /* arm_off[244] PARAM */);
  TRACE_POP
}

/*
equation index: 14515
type: SIMPLE_ASSIGN
arm_off[243] = 3.0410616886749198 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14515};
  (data->simulationInfo->realParameter[245] /* arm_off[243] PARAM */) = (3.0410616886749198) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14516
type: SIMPLE_ASSIGN
theta[243] = pitch * r_init[243] + arm_off[243]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14516};
  (data->simulationInfo->realParameter[1749] /* theta[243] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */)) + (data->simulationInfo->realParameter[245] /* arm_off[243] PARAM */);
  TRACE_POP
}

/*
equation index: 14517
type: SIMPLE_ASSIGN
arm_off[242] = 3.028495318060561 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14517};
  (data->simulationInfo->realParameter[244] /* arm_off[242] PARAM */) = (3.028495318060561) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14518
type: SIMPLE_ASSIGN
theta[242] = pitch * r_init[242] + arm_off[242]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14518(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14518};
  (data->simulationInfo->realParameter[1748] /* theta[242] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */)) + (data->simulationInfo->realParameter[244] /* arm_off[242] PARAM */);
  TRACE_POP
}

/*
equation index: 14519
type: SIMPLE_ASSIGN
arm_off[241] = 3.015928947446201 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14519};
  (data->simulationInfo->realParameter[243] /* arm_off[241] PARAM */) = (3.015928947446201) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14520
type: SIMPLE_ASSIGN
theta[241] = pitch * r_init[241] + arm_off[241]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14520(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14520};
  (data->simulationInfo->realParameter[1747] /* theta[241] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */)) + (data->simulationInfo->realParameter[243] /* arm_off[241] PARAM */);
  TRACE_POP
}

/*
equation index: 14521
type: SIMPLE_ASSIGN
arm_off[240] = 3.003362576831842 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14521};
  (data->simulationInfo->realParameter[242] /* arm_off[240] PARAM */) = (3.003362576831842) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14522
type: SIMPLE_ASSIGN
theta[240] = pitch * r_init[240] + arm_off[240]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14522(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14522};
  (data->simulationInfo->realParameter[1746] /* theta[240] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */)) + (data->simulationInfo->realParameter[242] /* arm_off[240] PARAM */);
  TRACE_POP
}

/*
equation index: 14523
type: SIMPLE_ASSIGN
arm_off[239] = 2.990796206217483 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14523};
  (data->simulationInfo->realParameter[241] /* arm_off[239] PARAM */) = (2.990796206217483) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14524
type: SIMPLE_ASSIGN
theta[239] = pitch * r_init[239] + arm_off[239]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14524};
  (data->simulationInfo->realParameter[1745] /* theta[239] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */)) + (data->simulationInfo->realParameter[241] /* arm_off[239] PARAM */);
  TRACE_POP
}

/*
equation index: 14525
type: SIMPLE_ASSIGN
arm_off[238] = 2.978229835603124 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14525};
  (data->simulationInfo->realParameter[240] /* arm_off[238] PARAM */) = (2.978229835603124) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14526
type: SIMPLE_ASSIGN
theta[238] = pitch * r_init[238] + arm_off[238]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14526(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14526};
  (data->simulationInfo->realParameter[1744] /* theta[238] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */)) + (data->simulationInfo->realParameter[240] /* arm_off[238] PARAM */);
  TRACE_POP
}

/*
equation index: 14527
type: SIMPLE_ASSIGN
arm_off[237] = 2.9656634649887645 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14527};
  (data->simulationInfo->realParameter[239] /* arm_off[237] PARAM */) = (2.9656634649887645) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14528
type: SIMPLE_ASSIGN
theta[237] = pitch * r_init[237] + arm_off[237]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14528(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14528};
  (data->simulationInfo->realParameter[1743] /* theta[237] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */)) + (data->simulationInfo->realParameter[239] /* arm_off[237] PARAM */);
  TRACE_POP
}

/*
equation index: 14529
type: SIMPLE_ASSIGN
arm_off[236] = 2.9530970943744057 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14529};
  (data->simulationInfo->realParameter[238] /* arm_off[236] PARAM */) = (2.9530970943744057) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14530
type: SIMPLE_ASSIGN
theta[236] = pitch * r_init[236] + arm_off[236]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14530(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14530};
  (data->simulationInfo->realParameter[1742] /* theta[236] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */)) + (data->simulationInfo->realParameter[238] /* arm_off[236] PARAM */);
  TRACE_POP
}

/*
equation index: 14531
type: SIMPLE_ASSIGN
arm_off[235] = 2.9405307237600464 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14531};
  (data->simulationInfo->realParameter[237] /* arm_off[235] PARAM */) = (2.9405307237600464) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14532
type: SIMPLE_ASSIGN
theta[235] = pitch * r_init[235] + arm_off[235]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14532};
  (data->simulationInfo->realParameter[1741] /* theta[235] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */)) + (data->simulationInfo->realParameter[237] /* arm_off[235] PARAM */);
  TRACE_POP
}

/*
equation index: 14533
type: SIMPLE_ASSIGN
arm_off[234] = 2.927964353145687 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14533};
  (data->simulationInfo->realParameter[236] /* arm_off[234] PARAM */) = (2.927964353145687) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14534
type: SIMPLE_ASSIGN
theta[234] = pitch * r_init[234] + arm_off[234]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14534(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14534};
  (data->simulationInfo->realParameter[1740] /* theta[234] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */)) + (data->simulationInfo->realParameter[236] /* arm_off[234] PARAM */);
  TRACE_POP
}

/*
equation index: 14535
type: SIMPLE_ASSIGN
arm_off[233] = 2.9153979825313283 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14535};
  (data->simulationInfo->realParameter[235] /* arm_off[233] PARAM */) = (2.9153979825313283) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14536
type: SIMPLE_ASSIGN
theta[233] = pitch * r_init[233] + arm_off[233]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14536(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14536};
  (data->simulationInfo->realParameter[1739] /* theta[233] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */)) + (data->simulationInfo->realParameter[235] /* arm_off[233] PARAM */);
  TRACE_POP
}

/*
equation index: 14537
type: SIMPLE_ASSIGN
arm_off[232] = 2.902831611916969 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14537};
  (data->simulationInfo->realParameter[234] /* arm_off[232] PARAM */) = (2.902831611916969) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14538
type: SIMPLE_ASSIGN
theta[232] = pitch * r_init[232] + arm_off[232]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14538(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14538};
  (data->simulationInfo->realParameter[1738] /* theta[232] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */)) + (data->simulationInfo->realParameter[234] /* arm_off[232] PARAM */);
  TRACE_POP
}

/*
equation index: 14539
type: SIMPLE_ASSIGN
arm_off[231] = 2.8902652413026098 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14539};
  (data->simulationInfo->realParameter[233] /* arm_off[231] PARAM */) = (2.8902652413026098) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14540
type: SIMPLE_ASSIGN
theta[231] = pitch * r_init[231] + arm_off[231]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14540};
  (data->simulationInfo->realParameter[1737] /* theta[231] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */)) + (data->simulationInfo->realParameter[233] /* arm_off[231] PARAM */);
  TRACE_POP
}

/*
equation index: 14541
type: SIMPLE_ASSIGN
arm_off[230] = 2.8776988706882505 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14541};
  (data->simulationInfo->realParameter[232] /* arm_off[230] PARAM */) = (2.8776988706882505) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14542
type: SIMPLE_ASSIGN
theta[230] = pitch * r_init[230] + arm_off[230]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14542(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14542};
  (data->simulationInfo->realParameter[1736] /* theta[230] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */)) + (data->simulationInfo->realParameter[232] /* arm_off[230] PARAM */);
  TRACE_POP
}

/*
equation index: 14543
type: SIMPLE_ASSIGN
arm_off[229] = 2.865132500073891 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14543};
  (data->simulationInfo->realParameter[231] /* arm_off[229] PARAM */) = (2.865132500073891) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14544
type: SIMPLE_ASSIGN
theta[229] = pitch * r_init[229] + arm_off[229]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14544(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14544};
  (data->simulationInfo->realParameter[1735] /* theta[229] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */)) + (data->simulationInfo->realParameter[231] /* arm_off[229] PARAM */);
  TRACE_POP
}

/*
equation index: 14545
type: SIMPLE_ASSIGN
arm_off[228] = 2.852566129459532 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14545};
  (data->simulationInfo->realParameter[230] /* arm_off[228] PARAM */) = (2.852566129459532) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14546
type: SIMPLE_ASSIGN
theta[228] = pitch * r_init[228] + arm_off[228]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14546(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14546};
  (data->simulationInfo->realParameter[1734] /* theta[228] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */)) + (data->simulationInfo->realParameter[230] /* arm_off[228] PARAM */);
  TRACE_POP
}

/*
equation index: 14547
type: SIMPLE_ASSIGN
arm_off[227] = 2.839999758845173 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14547};
  (data->simulationInfo->realParameter[229] /* arm_off[227] PARAM */) = (2.839999758845173) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14548
type: SIMPLE_ASSIGN
theta[227] = pitch * r_init[227] + arm_off[227]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14548};
  (data->simulationInfo->realParameter[1733] /* theta[227] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */)) + (data->simulationInfo->realParameter[229] /* arm_off[227] PARAM */);
  TRACE_POP
}

/*
equation index: 14549
type: SIMPLE_ASSIGN
arm_off[226] = 2.827433388230814 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14549};
  (data->simulationInfo->realParameter[228] /* arm_off[226] PARAM */) = (2.827433388230814) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14550
type: SIMPLE_ASSIGN
theta[226] = pitch * r_init[226] + arm_off[226]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14550(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14550};
  (data->simulationInfo->realParameter[1732] /* theta[226] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */)) + (data->simulationInfo->realParameter[228] /* arm_off[226] PARAM */);
  TRACE_POP
}

/*
equation index: 14551
type: SIMPLE_ASSIGN
arm_off[225] = 2.8148670176164545 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14551};
  (data->simulationInfo->realParameter[227] /* arm_off[225] PARAM */) = (2.8148670176164545) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14552
type: SIMPLE_ASSIGN
theta[225] = pitch * r_init[225] + arm_off[225]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14552(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14552};
  (data->simulationInfo->realParameter[1731] /* theta[225] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */)) + (data->simulationInfo->realParameter[227] /* arm_off[225] PARAM */);
  TRACE_POP
}

/*
equation index: 14553
type: SIMPLE_ASSIGN
arm_off[224] = 2.8023006470020957 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14553};
  (data->simulationInfo->realParameter[226] /* arm_off[224] PARAM */) = (2.8023006470020957) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14554
type: SIMPLE_ASSIGN
theta[224] = pitch * r_init[224] + arm_off[224]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14554(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14554};
  (data->simulationInfo->realParameter[1730] /* theta[224] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */)) + (data->simulationInfo->realParameter[226] /* arm_off[224] PARAM */);
  TRACE_POP
}

/*
equation index: 14555
type: SIMPLE_ASSIGN
arm_off[223] = 2.7897342763877364 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14555};
  (data->simulationInfo->realParameter[225] /* arm_off[223] PARAM */) = (2.7897342763877364) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14556
type: SIMPLE_ASSIGN
theta[223] = pitch * r_init[223] + arm_off[223]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14556};
  (data->simulationInfo->realParameter[1729] /* theta[223] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */)) + (data->simulationInfo->realParameter[225] /* arm_off[223] PARAM */);
  TRACE_POP
}

/*
equation index: 14557
type: SIMPLE_ASSIGN
arm_off[222] = 2.777167905773377 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14557};
  (data->simulationInfo->realParameter[224] /* arm_off[222] PARAM */) = (2.777167905773377) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14558
type: SIMPLE_ASSIGN
theta[222] = pitch * r_init[222] + arm_off[222]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14558(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14558};
  (data->simulationInfo->realParameter[1728] /* theta[222] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */)) + (data->simulationInfo->realParameter[224] /* arm_off[222] PARAM */);
  TRACE_POP
}

/*
equation index: 14559
type: SIMPLE_ASSIGN
arm_off[221] = 2.7646015351590183 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14559};
  (data->simulationInfo->realParameter[223] /* arm_off[221] PARAM */) = (2.7646015351590183) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14560
type: SIMPLE_ASSIGN
theta[221] = pitch * r_init[221] + arm_off[221]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14560(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14560};
  (data->simulationInfo->realParameter[1727] /* theta[221] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */)) + (data->simulationInfo->realParameter[223] /* arm_off[221] PARAM */);
  TRACE_POP
}

/*
equation index: 14561
type: SIMPLE_ASSIGN
arm_off[220] = 2.752035164544659 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14561};
  (data->simulationInfo->realParameter[222] /* arm_off[220] PARAM */) = (2.752035164544659) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14562
type: SIMPLE_ASSIGN
theta[220] = pitch * r_init[220] + arm_off[220]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14562(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14562};
  (data->simulationInfo->realParameter[1726] /* theta[220] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */)) + (data->simulationInfo->realParameter[222] /* arm_off[220] PARAM */);
  TRACE_POP
}

/*
equation index: 14563
type: SIMPLE_ASSIGN
arm_off[219] = 2.7394687939302993 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14563};
  (data->simulationInfo->realParameter[221] /* arm_off[219] PARAM */) = (2.7394687939302993) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14564
type: SIMPLE_ASSIGN
theta[219] = pitch * r_init[219] + arm_off[219]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14564};
  (data->simulationInfo->realParameter[1725] /* theta[219] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */)) + (data->simulationInfo->realParameter[221] /* arm_off[219] PARAM */);
  TRACE_POP
}

/*
equation index: 14565
type: SIMPLE_ASSIGN
arm_off[218] = 2.7269024233159405 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14565};
  (data->simulationInfo->realParameter[220] /* arm_off[218] PARAM */) = (2.7269024233159405) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14566
type: SIMPLE_ASSIGN
theta[218] = pitch * r_init[218] + arm_off[218]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14566(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14566};
  (data->simulationInfo->realParameter[1724] /* theta[218] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */)) + (data->simulationInfo->realParameter[220] /* arm_off[218] PARAM */);
  TRACE_POP
}

/*
equation index: 14567
type: SIMPLE_ASSIGN
arm_off[217] = 2.714336052701581 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14567};
  (data->simulationInfo->realParameter[219] /* arm_off[217] PARAM */) = (2.714336052701581) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14568
type: SIMPLE_ASSIGN
theta[217] = pitch * r_init[217] + arm_off[217]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14568(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14568};
  (data->simulationInfo->realParameter[1723] /* theta[217] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */)) + (data->simulationInfo->realParameter[219] /* arm_off[217] PARAM */);
  TRACE_POP
}

/*
equation index: 14569
type: SIMPLE_ASSIGN
arm_off[216] = 2.701769682087222 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14569};
  (data->simulationInfo->realParameter[218] /* arm_off[216] PARAM */) = (2.701769682087222) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14570
type: SIMPLE_ASSIGN
theta[216] = pitch * r_init[216] + arm_off[216]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14570(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14570};
  (data->simulationInfo->realParameter[1722] /* theta[216] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */)) + (data->simulationInfo->realParameter[218] /* arm_off[216] PARAM */);
  TRACE_POP
}

/*
equation index: 14571
type: SIMPLE_ASSIGN
arm_off[215] = 2.689203311472863 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14571};
  (data->simulationInfo->realParameter[217] /* arm_off[215] PARAM */) = (2.689203311472863) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14572
type: SIMPLE_ASSIGN
theta[215] = pitch * r_init[215] + arm_off[215]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14572};
  (data->simulationInfo->realParameter[1721] /* theta[215] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */)) + (data->simulationInfo->realParameter[217] /* arm_off[215] PARAM */);
  TRACE_POP
}

/*
equation index: 14573
type: SIMPLE_ASSIGN
arm_off[214] = 2.676636940858504 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14573};
  (data->simulationInfo->realParameter[216] /* arm_off[214] PARAM */) = (2.676636940858504) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14574
type: SIMPLE_ASSIGN
theta[214] = pitch * r_init[214] + arm_off[214]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14574(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14574};
  (data->simulationInfo->realParameter[1720] /* theta[214] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */)) + (data->simulationInfo->realParameter[216] /* arm_off[214] PARAM */);
  TRACE_POP
}

/*
equation index: 14575
type: SIMPLE_ASSIGN
arm_off[213] = 2.6640705702441445 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14575};
  (data->simulationInfo->realParameter[215] /* arm_off[213] PARAM */) = (2.6640705702441445) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14576
type: SIMPLE_ASSIGN
theta[213] = pitch * r_init[213] + arm_off[213]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14576(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14576};
  (data->simulationInfo->realParameter[1719] /* theta[213] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */)) + (data->simulationInfo->realParameter[215] /* arm_off[213] PARAM */);
  TRACE_POP
}

/*
equation index: 14577
type: SIMPLE_ASSIGN
arm_off[212] = 2.6515041996297857 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14577};
  (data->simulationInfo->realParameter[214] /* arm_off[212] PARAM */) = (2.6515041996297857) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14578
type: SIMPLE_ASSIGN
theta[212] = pitch * r_init[212] + arm_off[212]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14578(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14578};
  (data->simulationInfo->realParameter[1718] /* theta[212] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */)) + (data->simulationInfo->realParameter[214] /* arm_off[212] PARAM */);
  TRACE_POP
}

/*
equation index: 14579
type: SIMPLE_ASSIGN
arm_off[211] = 2.6389378290154264 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14579};
  (data->simulationInfo->realParameter[213] /* arm_off[211] PARAM */) = (2.6389378290154264) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14580
type: SIMPLE_ASSIGN
theta[211] = pitch * r_init[211] + arm_off[211]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14580};
  (data->simulationInfo->realParameter[1717] /* theta[211] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */)) + (data->simulationInfo->realParameter[213] /* arm_off[211] PARAM */);
  TRACE_POP
}

/*
equation index: 14581
type: SIMPLE_ASSIGN
arm_off[210] = 2.626371458401067 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14581};
  (data->simulationInfo->realParameter[212] /* arm_off[210] PARAM */) = (2.626371458401067) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14582
type: SIMPLE_ASSIGN
theta[210] = pitch * r_init[210] + arm_off[210]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14582(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14582};
  (data->simulationInfo->realParameter[1716] /* theta[210] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */)) + (data->simulationInfo->realParameter[212] /* arm_off[210] PARAM */);
  TRACE_POP
}

/*
equation index: 14583
type: SIMPLE_ASSIGN
arm_off[209] = 2.6138050877867083 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14583};
  (data->simulationInfo->realParameter[211] /* arm_off[209] PARAM */) = (2.6138050877867083) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14584
type: SIMPLE_ASSIGN
theta[209] = pitch * r_init[209] + arm_off[209]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14584(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14584};
  (data->simulationInfo->realParameter[1715] /* theta[209] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */)) + (data->simulationInfo->realParameter[211] /* arm_off[209] PARAM */);
  TRACE_POP
}

/*
equation index: 14585
type: SIMPLE_ASSIGN
arm_off[208] = 2.6012387171723486 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14585};
  (data->simulationInfo->realParameter[210] /* arm_off[208] PARAM */) = (2.6012387171723486) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14586
type: SIMPLE_ASSIGN
theta[208] = pitch * r_init[208] + arm_off[208]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14586(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14586};
  (data->simulationInfo->realParameter[1714] /* theta[208] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */)) + (data->simulationInfo->realParameter[210] /* arm_off[208] PARAM */);
  TRACE_POP
}

/*
equation index: 14587
type: SIMPLE_ASSIGN
arm_off[207] = 2.5886723465579893 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14587};
  (data->simulationInfo->realParameter[209] /* arm_off[207] PARAM */) = (2.5886723465579893) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14588
type: SIMPLE_ASSIGN
theta[207] = pitch * r_init[207] + arm_off[207]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14588};
  (data->simulationInfo->realParameter[1713] /* theta[207] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */)) + (data->simulationInfo->realParameter[209] /* arm_off[207] PARAM */);
  TRACE_POP
}

/*
equation index: 14589
type: SIMPLE_ASSIGN
arm_off[206] = 2.5761059759436304 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14589};
  (data->simulationInfo->realParameter[208] /* arm_off[206] PARAM */) = (2.5761059759436304) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14590
type: SIMPLE_ASSIGN
theta[206] = pitch * r_init[206] + arm_off[206]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14590(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14590};
  (data->simulationInfo->realParameter[1712] /* theta[206] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */)) + (data->simulationInfo->realParameter[208] /* arm_off[206] PARAM */);
  TRACE_POP
}

/*
equation index: 14591
type: SIMPLE_ASSIGN
arm_off[205] = 2.563539605329271 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14591};
  (data->simulationInfo->realParameter[207] /* arm_off[205] PARAM */) = (2.563539605329271) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14592
type: SIMPLE_ASSIGN
theta[205] = pitch * r_init[205] + arm_off[205]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14592(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14592};
  (data->simulationInfo->realParameter[1711] /* theta[205] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */)) + (data->simulationInfo->realParameter[207] /* arm_off[205] PARAM */);
  TRACE_POP
}

/*
equation index: 14593
type: SIMPLE_ASSIGN
arm_off[204] = 2.550973234714912 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14593};
  (data->simulationInfo->realParameter[206] /* arm_off[204] PARAM */) = (2.550973234714912) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14594
type: SIMPLE_ASSIGN
theta[204] = pitch * r_init[204] + arm_off[204]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14594(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14594};
  (data->simulationInfo->realParameter[1710] /* theta[204] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */)) + (data->simulationInfo->realParameter[206] /* arm_off[204] PARAM */);
  TRACE_POP
}

/*
equation index: 14595
type: SIMPLE_ASSIGN
arm_off[203] = 2.538406864100553 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14595};
  (data->simulationInfo->realParameter[205] /* arm_off[203] PARAM */) = (2.538406864100553) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14596
type: SIMPLE_ASSIGN
theta[203] = pitch * r_init[203] + arm_off[203]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14596};
  (data->simulationInfo->realParameter[1709] /* theta[203] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */)) + (data->simulationInfo->realParameter[205] /* arm_off[203] PARAM */);
  TRACE_POP
}

/*
equation index: 14597
type: SIMPLE_ASSIGN
arm_off[202] = 2.5258404934861938 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14597};
  (data->simulationInfo->realParameter[204] /* arm_off[202] PARAM */) = (2.5258404934861938) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14598
type: SIMPLE_ASSIGN
theta[202] = pitch * r_init[202] + arm_off[202]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14598(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14598};
  (data->simulationInfo->realParameter[1708] /* theta[202] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */)) + (data->simulationInfo->realParameter[204] /* arm_off[202] PARAM */);
  TRACE_POP
}

/*
equation index: 14599
type: SIMPLE_ASSIGN
arm_off[201] = 2.5132741228718345 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14599};
  (data->simulationInfo->realParameter[203] /* arm_off[201] PARAM */) = (2.5132741228718345) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14600
type: SIMPLE_ASSIGN
theta[201] = pitch * r_init[201] + arm_off[201]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14600(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14600};
  (data->simulationInfo->realParameter[1707] /* theta[201] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */)) + (data->simulationInfo->realParameter[203] /* arm_off[201] PARAM */);
  TRACE_POP
}

/*
equation index: 14601
type: SIMPLE_ASSIGN
arm_off[200] = 2.5007077522574757 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14601};
  (data->simulationInfo->realParameter[202] /* arm_off[200] PARAM */) = (2.5007077522574757) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14602
type: SIMPLE_ASSIGN
theta[200] = pitch * r_init[200] + arm_off[200]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14602(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14602};
  (data->simulationInfo->realParameter[1706] /* theta[200] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */)) + (data->simulationInfo->realParameter[202] /* arm_off[200] PARAM */);
  TRACE_POP
}

/*
equation index: 14603
type: SIMPLE_ASSIGN
arm_off[199] = 2.4881413816431164 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14603};
  (data->simulationInfo->realParameter[201] /* arm_off[199] PARAM */) = (2.4881413816431164) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14604
type: SIMPLE_ASSIGN
theta[199] = pitch * r_init[199] + arm_off[199]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14604};
  (data->simulationInfo->realParameter[1705] /* theta[199] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */)) + (data->simulationInfo->realParameter[201] /* arm_off[199] PARAM */);
  TRACE_POP
}

/*
equation index: 14605
type: SIMPLE_ASSIGN
arm_off[198] = 2.4755750110287567 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14605};
  (data->simulationInfo->realParameter[200] /* arm_off[198] PARAM */) = (2.4755750110287567) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14606
type: SIMPLE_ASSIGN
theta[198] = pitch * r_init[198] + arm_off[198]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14606(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14606};
  (data->simulationInfo->realParameter[1704] /* theta[198] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */)) + (data->simulationInfo->realParameter[200] /* arm_off[198] PARAM */);
  TRACE_POP
}

/*
equation index: 14607
type: SIMPLE_ASSIGN
arm_off[197] = 2.463008640414398 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14607};
  (data->simulationInfo->realParameter[199] /* arm_off[197] PARAM */) = (2.463008640414398) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14608
type: SIMPLE_ASSIGN
theta[197] = pitch * r_init[197] + arm_off[197]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14608(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14608};
  (data->simulationInfo->realParameter[1703] /* theta[197] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */)) + (data->simulationInfo->realParameter[199] /* arm_off[197] PARAM */);
  TRACE_POP
}

/*
equation index: 14609
type: SIMPLE_ASSIGN
arm_off[196] = 2.4504422698000385 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14609};
  (data->simulationInfo->realParameter[198] /* arm_off[196] PARAM */) = (2.4504422698000385) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14610
type: SIMPLE_ASSIGN
theta[196] = pitch * r_init[196] + arm_off[196]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14610(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14610};
  (data->simulationInfo->realParameter[1702] /* theta[196] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */)) + (data->simulationInfo->realParameter[198] /* arm_off[196] PARAM */);
  TRACE_POP
}

/*
equation index: 14611
type: SIMPLE_ASSIGN
arm_off[195] = 2.4378758991856793 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14611};
  (data->simulationInfo->realParameter[197] /* arm_off[195] PARAM */) = (2.4378758991856793) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14612
type: SIMPLE_ASSIGN
theta[195] = pitch * r_init[195] + arm_off[195]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14612};
  (data->simulationInfo->realParameter[1701] /* theta[195] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */)) + (data->simulationInfo->realParameter[197] /* arm_off[195] PARAM */);
  TRACE_POP
}

/*
equation index: 14613
type: SIMPLE_ASSIGN
arm_off[194] = 2.4253095285713204 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14613};
  (data->simulationInfo->realParameter[196] /* arm_off[194] PARAM */) = (2.4253095285713204) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14614
type: SIMPLE_ASSIGN
theta[194] = pitch * r_init[194] + arm_off[194]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14614(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14614};
  (data->simulationInfo->realParameter[1700] /* theta[194] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */)) + (data->simulationInfo->realParameter[196] /* arm_off[194] PARAM */);
  TRACE_POP
}

/*
equation index: 14615
type: SIMPLE_ASSIGN
arm_off[193] = 2.412743157956961 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14615};
  (data->simulationInfo->realParameter[195] /* arm_off[193] PARAM */) = (2.412743157956961) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14616
type: SIMPLE_ASSIGN
theta[193] = pitch * r_init[193] + arm_off[193]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14616(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14616};
  (data->simulationInfo->realParameter[1699] /* theta[193] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */)) + (data->simulationInfo->realParameter[195] /* arm_off[193] PARAM */);
  TRACE_POP
}

/*
equation index: 14617
type: SIMPLE_ASSIGN
arm_off[192] = 2.400176787342602 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14617};
  (data->simulationInfo->realParameter[194] /* arm_off[192] PARAM */) = (2.400176787342602) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14618
type: SIMPLE_ASSIGN
theta[192] = pitch * r_init[192] + arm_off[192]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14618(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14618};
  (data->simulationInfo->realParameter[1698] /* theta[192] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */)) + (data->simulationInfo->realParameter[194] /* arm_off[192] PARAM */);
  TRACE_POP
}

/*
equation index: 14619
type: SIMPLE_ASSIGN
arm_off[191] = 2.387610416728243 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14619};
  (data->simulationInfo->realParameter[193] /* arm_off[191] PARAM */) = (2.387610416728243) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14620
type: SIMPLE_ASSIGN
theta[191] = pitch * r_init[191] + arm_off[191]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14620};
  (data->simulationInfo->realParameter[1697] /* theta[191] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */)) + (data->simulationInfo->realParameter[193] /* arm_off[191] PARAM */);
  TRACE_POP
}

/*
equation index: 14621
type: SIMPLE_ASSIGN
arm_off[190] = 2.3750440461138838 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14621};
  (data->simulationInfo->realParameter[192] /* arm_off[190] PARAM */) = (2.3750440461138838) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14622
type: SIMPLE_ASSIGN
theta[190] = pitch * r_init[190] + arm_off[190]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14622(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14622};
  (data->simulationInfo->realParameter[1696] /* theta[190] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */)) + (data->simulationInfo->realParameter[192] /* arm_off[190] PARAM */);
  TRACE_POP
}

/*
equation index: 14623
type: SIMPLE_ASSIGN
arm_off[189] = 2.3624776754995245 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14623};
  (data->simulationInfo->realParameter[191] /* arm_off[189] PARAM */) = (2.3624776754995245) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14624
type: SIMPLE_ASSIGN
theta[189] = pitch * r_init[189] + arm_off[189]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14624(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14624};
  (data->simulationInfo->realParameter[1695] /* theta[189] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */)) + (data->simulationInfo->realParameter[191] /* arm_off[189] PARAM */);
  TRACE_POP
}

/*
equation index: 14625
type: SIMPLE_ASSIGN
arm_off[188] = 2.3499113048851656 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14625};
  (data->simulationInfo->realParameter[190] /* arm_off[188] PARAM */) = (2.3499113048851656) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14626
type: SIMPLE_ASSIGN
theta[188] = pitch * r_init[188] + arm_off[188]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14626(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14626};
  (data->simulationInfo->realParameter[1694] /* theta[188] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */)) + (data->simulationInfo->realParameter[190] /* arm_off[188] PARAM */);
  TRACE_POP
}

/*
equation index: 14627
type: SIMPLE_ASSIGN
arm_off[187] = 2.337344934270806 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14627};
  (data->simulationInfo->realParameter[189] /* arm_off[187] PARAM */) = (2.337344934270806) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14628
type: SIMPLE_ASSIGN
theta[187] = pitch * r_init[187] + arm_off[187]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14628};
  (data->simulationInfo->realParameter[1693] /* theta[187] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */)) + (data->simulationInfo->realParameter[189] /* arm_off[187] PARAM */);
  TRACE_POP
}

/*
equation index: 14629
type: SIMPLE_ASSIGN
arm_off[186] = 2.3247785636564466 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14629};
  (data->simulationInfo->realParameter[188] /* arm_off[186] PARAM */) = (2.3247785636564466) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14630
type: SIMPLE_ASSIGN
theta[186] = pitch * r_init[186] + arm_off[186]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14630(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14630};
  (data->simulationInfo->realParameter[1692] /* theta[186] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */)) + (data->simulationInfo->realParameter[188] /* arm_off[186] PARAM */);
  TRACE_POP
}

/*
equation index: 14631
type: SIMPLE_ASSIGN
arm_off[185] = 2.312212193042088 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14631};
  (data->simulationInfo->realParameter[187] /* arm_off[185] PARAM */) = (2.312212193042088) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14632
type: SIMPLE_ASSIGN
theta[185] = pitch * r_init[185] + arm_off[185]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14632(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14632};
  (data->simulationInfo->realParameter[1691] /* theta[185] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */)) + (data->simulationInfo->realParameter[187] /* arm_off[185] PARAM */);
  TRACE_POP
}

/*
equation index: 14633
type: SIMPLE_ASSIGN
arm_off[184] = 2.2996458224277285 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14633};
  (data->simulationInfo->realParameter[186] /* arm_off[184] PARAM */) = (2.2996458224277285) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14634
type: SIMPLE_ASSIGN
theta[184] = pitch * r_init[184] + arm_off[184]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14634(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14634};
  (data->simulationInfo->realParameter[1690] /* theta[184] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */)) + (data->simulationInfo->realParameter[186] /* arm_off[184] PARAM */);
  TRACE_POP
}

/*
equation index: 14635
type: SIMPLE_ASSIGN
arm_off[183] = 2.2870794518133692 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14635};
  (data->simulationInfo->realParameter[185] /* arm_off[183] PARAM */) = (2.2870794518133692) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14636
type: SIMPLE_ASSIGN
theta[183] = pitch * r_init[183] + arm_off[183]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14636};
  (data->simulationInfo->realParameter[1689] /* theta[183] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */)) + (data->simulationInfo->realParameter[185] /* arm_off[183] PARAM */);
  TRACE_POP
}

/*
equation index: 14637
type: SIMPLE_ASSIGN
arm_off[182] = 2.2745130811990104 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14637};
  (data->simulationInfo->realParameter[184] /* arm_off[182] PARAM */) = (2.2745130811990104) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14638
type: SIMPLE_ASSIGN
theta[182] = pitch * r_init[182] + arm_off[182]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14638(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14638};
  (data->simulationInfo->realParameter[1688] /* theta[182] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */)) + (data->simulationInfo->realParameter[184] /* arm_off[182] PARAM */);
  TRACE_POP
}

/*
equation index: 14639
type: SIMPLE_ASSIGN
arm_off[181] = 2.261946710584651 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14639};
  (data->simulationInfo->realParameter[183] /* arm_off[181] PARAM */) = (2.261946710584651) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14640
type: SIMPLE_ASSIGN
theta[181] = pitch * r_init[181] + arm_off[181]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14640(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14640};
  (data->simulationInfo->realParameter[1687] /* theta[181] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */)) + (data->simulationInfo->realParameter[183] /* arm_off[181] PARAM */);
  TRACE_POP
}

/*
equation index: 14641
type: SIMPLE_ASSIGN
arm_off[180] = 2.249380339970292 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14641};
  (data->simulationInfo->realParameter[182] /* arm_off[180] PARAM */) = (2.249380339970292) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14642
type: SIMPLE_ASSIGN
theta[180] = pitch * r_init[180] + arm_off[180]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14642(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14642};
  (data->simulationInfo->realParameter[1686] /* theta[180] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */)) + (data->simulationInfo->realParameter[182] /* arm_off[180] PARAM */);
  TRACE_POP
}

/*
equation index: 14643
type: SIMPLE_ASSIGN
arm_off[179] = 2.236813969355933 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14643};
  (data->simulationInfo->realParameter[181] /* arm_off[179] PARAM */) = (2.236813969355933) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14644
type: SIMPLE_ASSIGN
theta[179] = pitch * r_init[179] + arm_off[179]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14644};
  (data->simulationInfo->realParameter[1685] /* theta[179] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */)) + (data->simulationInfo->realParameter[181] /* arm_off[179] PARAM */);
  TRACE_POP
}

/*
equation index: 14645
type: SIMPLE_ASSIGN
arm_off[178] = 2.2242475987415737 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14645};
  (data->simulationInfo->realParameter[180] /* arm_off[178] PARAM */) = (2.2242475987415737) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14646
type: SIMPLE_ASSIGN
theta[178] = pitch * r_init[178] + arm_off[178]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14646(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14646};
  (data->simulationInfo->realParameter[1684] /* theta[178] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */)) + (data->simulationInfo->realParameter[180] /* arm_off[178] PARAM */);
  TRACE_POP
}

/*
equation index: 14647
type: SIMPLE_ASSIGN
arm_off[177] = 2.211681228127214 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14647};
  (data->simulationInfo->realParameter[179] /* arm_off[177] PARAM */) = (2.211681228127214) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14648
type: SIMPLE_ASSIGN
theta[177] = pitch * r_init[177] + arm_off[177]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14648(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14648};
  (data->simulationInfo->realParameter[1683] /* theta[177] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */)) + (data->simulationInfo->realParameter[179] /* arm_off[177] PARAM */);
  TRACE_POP
}

/*
equation index: 14649
type: SIMPLE_ASSIGN
arm_off[176] = 2.199114857512855 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14649};
  (data->simulationInfo->realParameter[178] /* arm_off[176] PARAM */) = (2.199114857512855) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14650
type: SIMPLE_ASSIGN
theta[176] = pitch * r_init[176] + arm_off[176]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14650(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14650};
  (data->simulationInfo->realParameter[1682] /* theta[176] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */)) + (data->simulationInfo->realParameter[178] /* arm_off[176] PARAM */);
  TRACE_POP
}

/*
equation index: 14651
type: SIMPLE_ASSIGN
arm_off[175] = 2.186548486898496 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14651};
  (data->simulationInfo->realParameter[177] /* arm_off[175] PARAM */) = (2.186548486898496) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14652
type: SIMPLE_ASSIGN
theta[175] = pitch * r_init[175] + arm_off[175]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14652};
  (data->simulationInfo->realParameter[1681] /* theta[175] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */)) + (data->simulationInfo->realParameter[177] /* arm_off[175] PARAM */);
  TRACE_POP
}

/*
equation index: 14653
type: SIMPLE_ASSIGN
arm_off[174] = 2.1739821162841366 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14653};
  (data->simulationInfo->realParameter[176] /* arm_off[174] PARAM */) = (2.1739821162841366) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14654
type: SIMPLE_ASSIGN
theta[174] = pitch * r_init[174] + arm_off[174]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14654(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14654};
  (data->simulationInfo->realParameter[1680] /* theta[174] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */)) + (data->simulationInfo->realParameter[176] /* arm_off[174] PARAM */);
  TRACE_POP
}

/*
equation index: 14655
type: SIMPLE_ASSIGN
arm_off[173] = 2.161415745669778 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14655};
  (data->simulationInfo->realParameter[175] /* arm_off[173] PARAM */) = (2.161415745669778) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14656
type: SIMPLE_ASSIGN
theta[173] = pitch * r_init[173] + arm_off[173]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14656(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14656};
  (data->simulationInfo->realParameter[1679] /* theta[173] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */)) + (data->simulationInfo->realParameter[175] /* arm_off[173] PARAM */);
  TRACE_POP
}

/*
equation index: 14657
type: SIMPLE_ASSIGN
arm_off[172] = 2.1488493750554185 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14657};
  (data->simulationInfo->realParameter[174] /* arm_off[172] PARAM */) = (2.1488493750554185) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14658
type: SIMPLE_ASSIGN
theta[172] = pitch * r_init[172] + arm_off[172]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14658(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14658};
  (data->simulationInfo->realParameter[1678] /* theta[172] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */)) + (data->simulationInfo->realParameter[174] /* arm_off[172] PARAM */);
  TRACE_POP
}

/*
equation index: 14659
type: SIMPLE_ASSIGN
arm_off[171] = 2.1362830044410592 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14659};
  (data->simulationInfo->realParameter[173] /* arm_off[171] PARAM */) = (2.1362830044410592) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14660
type: SIMPLE_ASSIGN
theta[171] = pitch * r_init[171] + arm_off[171]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14660};
  (data->simulationInfo->realParameter[1677] /* theta[171] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */)) + (data->simulationInfo->realParameter[173] /* arm_off[171] PARAM */);
  TRACE_POP
}

/*
equation index: 14661
type: SIMPLE_ASSIGN
arm_off[170] = 2.1237166338267004 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14661};
  (data->simulationInfo->realParameter[172] /* arm_off[170] PARAM */) = (2.1237166338267004) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14662
type: SIMPLE_ASSIGN
theta[170] = pitch * r_init[170] + arm_off[170]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14662(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14662};
  (data->simulationInfo->realParameter[1676] /* theta[170] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */)) + (data->simulationInfo->realParameter[172] /* arm_off[170] PARAM */);
  TRACE_POP
}

/*
equation index: 14663
type: SIMPLE_ASSIGN
arm_off[169] = 2.111150263212341 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14663};
  (data->simulationInfo->realParameter[171] /* arm_off[169] PARAM */) = (2.111150263212341) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14664
type: SIMPLE_ASSIGN
theta[169] = pitch * r_init[169] + arm_off[169]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14664(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14664};
  (data->simulationInfo->realParameter[1675] /* theta[169] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */)) + (data->simulationInfo->realParameter[171] /* arm_off[169] PARAM */);
  TRACE_POP
}

/*
equation index: 14665
type: SIMPLE_ASSIGN
arm_off[168] = 2.098583892597982 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14665};
  (data->simulationInfo->realParameter[170] /* arm_off[168] PARAM */) = (2.098583892597982) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14666
type: SIMPLE_ASSIGN
theta[168] = pitch * r_init[168] + arm_off[168]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14666(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14666};
  (data->simulationInfo->realParameter[1674] /* theta[168] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */)) + (data->simulationInfo->realParameter[170] /* arm_off[168] PARAM */);
  TRACE_POP
}

/*
equation index: 14667
type: SIMPLE_ASSIGN
arm_off[167] = 2.086017521983623 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14667};
  (data->simulationInfo->realParameter[169] /* arm_off[167] PARAM */) = (2.086017521983623) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14668
type: SIMPLE_ASSIGN
theta[167] = pitch * r_init[167] + arm_off[167]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14668};
  (data->simulationInfo->realParameter[1673] /* theta[167] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */)) + (data->simulationInfo->realParameter[169] /* arm_off[167] PARAM */);
  TRACE_POP
}

/*
equation index: 14669
type: SIMPLE_ASSIGN
arm_off[166] = 2.0734511513692633 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14669};
  (data->simulationInfo->realParameter[168] /* arm_off[166] PARAM */) = (2.0734511513692633) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14670
type: SIMPLE_ASSIGN
theta[166] = pitch * r_init[166] + arm_off[166]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14670(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14670};
  (data->simulationInfo->realParameter[1672] /* theta[166] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */)) + (data->simulationInfo->realParameter[168] /* arm_off[166] PARAM */);
  TRACE_POP
}

/*
equation index: 14671
type: SIMPLE_ASSIGN
arm_off[165] = 2.060884780754904 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14671};
  (data->simulationInfo->realParameter[167] /* arm_off[165] PARAM */) = (2.060884780754904) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14672
type: SIMPLE_ASSIGN
theta[165] = pitch * r_init[165] + arm_off[165]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14672(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14672};
  (data->simulationInfo->realParameter[1671] /* theta[165] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */)) + (data->simulationInfo->realParameter[167] /* arm_off[165] PARAM */);
  TRACE_POP
}

/*
equation index: 14673
type: SIMPLE_ASSIGN
arm_off[164] = 2.048318410140545 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14673};
  (data->simulationInfo->realParameter[166] /* arm_off[164] PARAM */) = (2.048318410140545) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14674
type: SIMPLE_ASSIGN
theta[164] = pitch * r_init[164] + arm_off[164]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14674(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14674};
  (data->simulationInfo->realParameter[1670] /* theta[164] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */)) + (data->simulationInfo->realParameter[166] /* arm_off[164] PARAM */);
  TRACE_POP
}

/*
equation index: 14675
type: SIMPLE_ASSIGN
arm_off[163] = 2.035752039526186 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14675};
  (data->simulationInfo->realParameter[165] /* arm_off[163] PARAM */) = (2.035752039526186) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14676
type: SIMPLE_ASSIGN
theta[163] = pitch * r_init[163] + arm_off[163]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14676};
  (data->simulationInfo->realParameter[1669] /* theta[163] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */)) + (data->simulationInfo->realParameter[165] /* arm_off[163] PARAM */);
  TRACE_POP
}

/*
equation index: 14677
type: SIMPLE_ASSIGN
arm_off[162] = 2.0231856689118266 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14677};
  (data->simulationInfo->realParameter[164] /* arm_off[162] PARAM */) = (2.0231856689118266) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14678
type: SIMPLE_ASSIGN
theta[162] = pitch * r_init[162] + arm_off[162]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14678(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14678};
  (data->simulationInfo->realParameter[1668] /* theta[162] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */)) + (data->simulationInfo->realParameter[164] /* arm_off[162] PARAM */);
  TRACE_POP
}

/*
equation index: 14679
type: SIMPLE_ASSIGN
arm_off[161] = 2.0106192982974678 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14679};
  (data->simulationInfo->realParameter[163] /* arm_off[161] PARAM */) = (2.0106192982974678) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14680
type: SIMPLE_ASSIGN
theta[161] = pitch * r_init[161] + arm_off[161]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14680(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14680};
  (data->simulationInfo->realParameter[1667] /* theta[161] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */)) + (data->simulationInfo->realParameter[163] /* arm_off[161] PARAM */);
  TRACE_POP
}

/*
equation index: 14681
type: SIMPLE_ASSIGN
arm_off[160] = 1.9980529276831085 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14681};
  (data->simulationInfo->realParameter[162] /* arm_off[160] PARAM */) = (1.9980529276831085) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14682
type: SIMPLE_ASSIGN
theta[160] = pitch * r_init[160] + arm_off[160]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14682(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14682};
  (data->simulationInfo->realParameter[1666] /* theta[160] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */)) + (data->simulationInfo->realParameter[162] /* arm_off[160] PARAM */);
  TRACE_POP
}

/*
equation index: 14683
type: SIMPLE_ASSIGN
arm_off[159] = 1.9854865570687494 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14683};
  (data->simulationInfo->realParameter[161] /* arm_off[159] PARAM */) = (1.9854865570687494) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14684
type: SIMPLE_ASSIGN
theta[159] = pitch * r_init[159] + arm_off[159]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14684};
  (data->simulationInfo->realParameter[1665] /* theta[159] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */)) + (data->simulationInfo->realParameter[161] /* arm_off[159] PARAM */);
  TRACE_POP
}

/*
equation index: 14685
type: SIMPLE_ASSIGN
arm_off[158] = 1.97292018645439 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14685};
  (data->simulationInfo->realParameter[160] /* arm_off[158] PARAM */) = (1.97292018645439) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14686
type: SIMPLE_ASSIGN
theta[158] = pitch * r_init[158] + arm_off[158]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14686(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14686};
  (data->simulationInfo->realParameter[1664] /* theta[158] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */)) + (data->simulationInfo->realParameter[160] /* arm_off[158] PARAM */);
  TRACE_POP
}

/*
equation index: 14687
type: SIMPLE_ASSIGN
arm_off[157] = 1.9603538158400309 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14687};
  (data->simulationInfo->realParameter[159] /* arm_off[157] PARAM */) = (1.9603538158400309) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14688
type: SIMPLE_ASSIGN
theta[157] = pitch * r_init[157] + arm_off[157]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14688(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14688};
  (data->simulationInfo->realParameter[1663] /* theta[157] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1162] /* r_init[157] PARAM */)) + (data->simulationInfo->realParameter[159] /* arm_off[157] PARAM */);
  TRACE_POP
}

/*
equation index: 14689
type: SIMPLE_ASSIGN
arm_off[156] = 1.9477874452256718 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14689};
  (data->simulationInfo->realParameter[158] /* arm_off[156] PARAM */) = (1.9477874452256718) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14690
type: SIMPLE_ASSIGN
theta[156] = pitch * r_init[156] + arm_off[156]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14690(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14690};
  (data->simulationInfo->realParameter[1662] /* theta[156] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1161] /* r_init[156] PARAM */)) + (data->simulationInfo->realParameter[158] /* arm_off[156] PARAM */);
  TRACE_POP
}

/*
equation index: 14691
type: SIMPLE_ASSIGN
arm_off[155] = 1.9352210746113125 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14691};
  (data->simulationInfo->realParameter[157] /* arm_off[155] PARAM */) = (1.9352210746113125) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14692
type: SIMPLE_ASSIGN
theta[155] = pitch * r_init[155] + arm_off[155]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14692};
  (data->simulationInfo->realParameter[1661] /* theta[155] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1160] /* r_init[155] PARAM */)) + (data->simulationInfo->realParameter[157] /* arm_off[155] PARAM */);
  TRACE_POP
}

/*
equation index: 14693
type: SIMPLE_ASSIGN
arm_off[154] = 1.9226547039969535 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14693};
  (data->simulationInfo->realParameter[156] /* arm_off[154] PARAM */) = (1.9226547039969535) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14694
type: SIMPLE_ASSIGN
theta[154] = pitch * r_init[154] + arm_off[154]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14694(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14694};
  (data->simulationInfo->realParameter[1660] /* theta[154] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1159] /* r_init[154] PARAM */)) + (data->simulationInfo->realParameter[156] /* arm_off[154] PARAM */);
  TRACE_POP
}

/*
equation index: 14695
type: SIMPLE_ASSIGN
arm_off[153] = 1.9100883333825942 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14695};
  (data->simulationInfo->realParameter[155] /* arm_off[153] PARAM */) = (1.9100883333825942) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14696
type: SIMPLE_ASSIGN
theta[153] = pitch * r_init[153] + arm_off[153]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14696(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14696};
  (data->simulationInfo->realParameter[1659] /* theta[153] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1158] /* r_init[153] PARAM */)) + (data->simulationInfo->realParameter[155] /* arm_off[153] PARAM */);
  TRACE_POP
}

/*
equation index: 14697
type: SIMPLE_ASSIGN
arm_off[152] = 1.897521962768235 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14697};
  (data->simulationInfo->realParameter[154] /* arm_off[152] PARAM */) = (1.897521962768235) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14698
type: SIMPLE_ASSIGN
theta[152] = pitch * r_init[152] + arm_off[152]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14698(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14698};
  (data->simulationInfo->realParameter[1658] /* theta[152] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1157] /* r_init[152] PARAM */)) + (data->simulationInfo->realParameter[154] /* arm_off[152] PARAM */);
  TRACE_POP
}

/*
equation index: 14699
type: SIMPLE_ASSIGN
arm_off[151] = 1.8849555921538759 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14699};
  (data->simulationInfo->realParameter[153] /* arm_off[151] PARAM */) = (1.8849555921538759) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14700
type: SIMPLE_ASSIGN
theta[151] = pitch * r_init[151] + arm_off[151]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14700};
  (data->simulationInfo->realParameter[1657] /* theta[151] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1156] /* r_init[151] PARAM */)) + (data->simulationInfo->realParameter[153] /* arm_off[151] PARAM */);
  TRACE_POP
}

/*
equation index: 14701
type: SIMPLE_ASSIGN
arm_off[150] = 1.8723892215395168 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14701};
  (data->simulationInfo->realParameter[152] /* arm_off[150] PARAM */) = (1.8723892215395168) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14702
type: SIMPLE_ASSIGN
theta[150] = pitch * r_init[150] + arm_off[150]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14702(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14702};
  (data->simulationInfo->realParameter[1656] /* theta[150] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1155] /* r_init[150] PARAM */)) + (data->simulationInfo->realParameter[152] /* arm_off[150] PARAM */);
  TRACE_POP
}

/*
equation index: 14703
type: SIMPLE_ASSIGN
arm_off[149] = 1.8598228509251575 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14703};
  (data->simulationInfo->realParameter[151] /* arm_off[149] PARAM */) = (1.8598228509251575) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14704
type: SIMPLE_ASSIGN
theta[149] = pitch * r_init[149] + arm_off[149]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14704(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14704};
  (data->simulationInfo->realParameter[1655] /* theta[149] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1154] /* r_init[149] PARAM */)) + (data->simulationInfo->realParameter[151] /* arm_off[149] PARAM */);
  TRACE_POP
}

/*
equation index: 14705
type: SIMPLE_ASSIGN
arm_off[148] = 1.8472564803107985 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14705};
  (data->simulationInfo->realParameter[150] /* arm_off[148] PARAM */) = (1.8472564803107985) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14706
type: SIMPLE_ASSIGN
theta[148] = pitch * r_init[148] + arm_off[148]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14706(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14706};
  (data->simulationInfo->realParameter[1654] /* theta[148] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1153] /* r_init[148] PARAM */)) + (data->simulationInfo->realParameter[150] /* arm_off[148] PARAM */);
  TRACE_POP
}

/*
equation index: 14707
type: SIMPLE_ASSIGN
arm_off[147] = 1.8346901096964392 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14707};
  (data->simulationInfo->realParameter[149] /* arm_off[147] PARAM */) = (1.8346901096964392) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14708
type: SIMPLE_ASSIGN
theta[147] = pitch * r_init[147] + arm_off[147]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14708};
  (data->simulationInfo->realParameter[1653] /* theta[147] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1152] /* r_init[147] PARAM */)) + (data->simulationInfo->realParameter[149] /* arm_off[147] PARAM */);
  TRACE_POP
}

/*
equation index: 14709
type: SIMPLE_ASSIGN
arm_off[146] = 1.82212373908208 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14709};
  (data->simulationInfo->realParameter[148] /* arm_off[146] PARAM */) = (1.82212373908208) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14710
type: SIMPLE_ASSIGN
theta[146] = pitch * r_init[146] + arm_off[146]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14710(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14710};
  (data->simulationInfo->realParameter[1652] /* theta[146] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1151] /* r_init[146] PARAM */)) + (data->simulationInfo->realParameter[148] /* arm_off[146] PARAM */);
  TRACE_POP
}

/*
equation index: 14711
type: SIMPLE_ASSIGN
arm_off[145] = 1.8095573684677209 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14711};
  (data->simulationInfo->realParameter[147] /* arm_off[145] PARAM */) = (1.8095573684677209) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14712
type: SIMPLE_ASSIGN
theta[145] = pitch * r_init[145] + arm_off[145]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14712(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14712};
  (data->simulationInfo->realParameter[1651] /* theta[145] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1150] /* r_init[145] PARAM */)) + (data->simulationInfo->realParameter[147] /* arm_off[145] PARAM */);
  TRACE_POP
}

/*
equation index: 14713
type: SIMPLE_ASSIGN
arm_off[144] = 1.7969909978533618 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14713};
  (data->simulationInfo->realParameter[146] /* arm_off[144] PARAM */) = (1.7969909978533618) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14714
type: SIMPLE_ASSIGN
theta[144] = pitch * r_init[144] + arm_off[144]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14714(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14714};
  (data->simulationInfo->realParameter[1650] /* theta[144] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1149] /* r_init[144] PARAM */)) + (data->simulationInfo->realParameter[146] /* arm_off[144] PARAM */);
  TRACE_POP
}

/*
equation index: 14715
type: SIMPLE_ASSIGN
arm_off[143] = 1.7844246272390025 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14715};
  (data->simulationInfo->realParameter[145] /* arm_off[143] PARAM */) = (1.7844246272390025) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14716
type: SIMPLE_ASSIGN
theta[143] = pitch * r_init[143] + arm_off[143]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14716};
  (data->simulationInfo->realParameter[1649] /* theta[143] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1148] /* r_init[143] PARAM */)) + (data->simulationInfo->realParameter[145] /* arm_off[143] PARAM */);
  TRACE_POP
}

/*
equation index: 14717
type: SIMPLE_ASSIGN
arm_off[142] = 1.7718582566246432 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14717};
  (data->simulationInfo->realParameter[144] /* arm_off[142] PARAM */) = (1.7718582566246432) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14718
type: SIMPLE_ASSIGN
theta[142] = pitch * r_init[142] + arm_off[142]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14718(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14718};
  (data->simulationInfo->realParameter[1648] /* theta[142] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1147] /* r_init[142] PARAM */)) + (data->simulationInfo->realParameter[144] /* arm_off[142] PARAM */);
  TRACE_POP
}

/*
equation index: 14719
type: SIMPLE_ASSIGN
arm_off[141] = 1.7592918860102842 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14719};
  (data->simulationInfo->realParameter[143] /* arm_off[141] PARAM */) = (1.7592918860102842) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14720
type: SIMPLE_ASSIGN
theta[141] = pitch * r_init[141] + arm_off[141]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14720(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14720};
  (data->simulationInfo->realParameter[1647] /* theta[141] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1146] /* r_init[141] PARAM */)) + (data->simulationInfo->realParameter[143] /* arm_off[141] PARAM */);
  TRACE_POP
}

/*
equation index: 14721
type: SIMPLE_ASSIGN
arm_off[140] = 1.746725515395925 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14721};
  (data->simulationInfo->realParameter[142] /* arm_off[140] PARAM */) = (1.746725515395925) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14722
type: SIMPLE_ASSIGN
theta[140] = pitch * r_init[140] + arm_off[140]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14722(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14722};
  (data->simulationInfo->realParameter[1646] /* theta[140] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1145] /* r_init[140] PARAM */)) + (data->simulationInfo->realParameter[142] /* arm_off[140] PARAM */);
  TRACE_POP
}

/*
equation index: 14723
type: SIMPLE_ASSIGN
arm_off[139] = 1.7341591447815659 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14723};
  (data->simulationInfo->realParameter[141] /* arm_off[139] PARAM */) = (1.7341591447815659) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14724
type: SIMPLE_ASSIGN
theta[139] = pitch * r_init[139] + arm_off[139]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14724};
  (data->simulationInfo->realParameter[1645] /* theta[139] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1144] /* r_init[139] PARAM */)) + (data->simulationInfo->realParameter[141] /* arm_off[139] PARAM */);
  TRACE_POP
}

/*
equation index: 14725
type: SIMPLE_ASSIGN
arm_off[138] = 1.7215927741672068 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14725};
  (data->simulationInfo->realParameter[140] /* arm_off[138] PARAM */) = (1.7215927741672068) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14726
type: SIMPLE_ASSIGN
theta[138] = pitch * r_init[138] + arm_off[138]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14726(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14726};
  (data->simulationInfo->realParameter[1644] /* theta[138] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1143] /* r_init[138] PARAM */)) + (data->simulationInfo->realParameter[140] /* arm_off[138] PARAM */);
  TRACE_POP
}

/*
equation index: 14727
type: SIMPLE_ASSIGN
arm_off[137] = 1.7090264035528475 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14727};
  (data->simulationInfo->realParameter[139] /* arm_off[137] PARAM */) = (1.7090264035528475) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14728
type: SIMPLE_ASSIGN
theta[137] = pitch * r_init[137] + arm_off[137]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14728(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14728};
  (data->simulationInfo->realParameter[1643] /* theta[137] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1142] /* r_init[137] PARAM */)) + (data->simulationInfo->realParameter[139] /* arm_off[137] PARAM */);
  TRACE_POP
}

/*
equation index: 14729
type: SIMPLE_ASSIGN
arm_off[136] = 1.6964600329384882 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14729};
  (data->simulationInfo->realParameter[138] /* arm_off[136] PARAM */) = (1.6964600329384882) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14730
type: SIMPLE_ASSIGN
theta[136] = pitch * r_init[136] + arm_off[136]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14730(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14730};
  (data->simulationInfo->realParameter[1642] /* theta[136] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1141] /* r_init[136] PARAM */)) + (data->simulationInfo->realParameter[138] /* arm_off[136] PARAM */);
  TRACE_POP
}

/*
equation index: 14731
type: SIMPLE_ASSIGN
arm_off[135] = 1.6838936623241292 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14731};
  (data->simulationInfo->realParameter[137] /* arm_off[135] PARAM */) = (1.6838936623241292) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14732
type: SIMPLE_ASSIGN
theta[135] = pitch * r_init[135] + arm_off[135]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14732};
  (data->simulationInfo->realParameter[1641] /* theta[135] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1140] /* r_init[135] PARAM */)) + (data->simulationInfo->realParameter[137] /* arm_off[135] PARAM */);
  TRACE_POP
}

/*
equation index: 14733
type: SIMPLE_ASSIGN
arm_off[134] = 1.67132729170977 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14733};
  (data->simulationInfo->realParameter[136] /* arm_off[134] PARAM */) = (1.67132729170977) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14734
type: SIMPLE_ASSIGN
theta[134] = pitch * r_init[134] + arm_off[134]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14734(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14734};
  (data->simulationInfo->realParameter[1640] /* theta[134] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1139] /* r_init[134] PARAM */)) + (data->simulationInfo->realParameter[136] /* arm_off[134] PARAM */);
  TRACE_POP
}

/*
equation index: 14735
type: SIMPLE_ASSIGN
arm_off[133] = 1.6587609210954108 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14735};
  (data->simulationInfo->realParameter[135] /* arm_off[133] PARAM */) = (1.6587609210954108) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14736
type: SIMPLE_ASSIGN
theta[133] = pitch * r_init[133] + arm_off[133]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14736(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14736};
  (data->simulationInfo->realParameter[1639] /* theta[133] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1138] /* r_init[133] PARAM */)) + (data->simulationInfo->realParameter[135] /* arm_off[133] PARAM */);
  TRACE_POP
}

/*
equation index: 14737
type: SIMPLE_ASSIGN
arm_off[132] = 1.6461945504810518 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14737};
  (data->simulationInfo->realParameter[134] /* arm_off[132] PARAM */) = (1.6461945504810518) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14738
type: SIMPLE_ASSIGN
theta[132] = pitch * r_init[132] + arm_off[132]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14738(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14738};
  (data->simulationInfo->realParameter[1638] /* theta[132] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1137] /* r_init[132] PARAM */)) + (data->simulationInfo->realParameter[134] /* arm_off[132] PARAM */);
  TRACE_POP
}

/*
equation index: 14739
type: SIMPLE_ASSIGN
arm_off[131] = 1.6336281798666923 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14739};
  (data->simulationInfo->realParameter[133] /* arm_off[131] PARAM */) = (1.6336281798666923) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14740
type: SIMPLE_ASSIGN
theta[131] = pitch * r_init[131] + arm_off[131]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14740};
  (data->simulationInfo->realParameter[1637] /* theta[131] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1136] /* r_init[131] PARAM */)) + (data->simulationInfo->realParameter[133] /* arm_off[131] PARAM */);
  TRACE_POP
}

/*
equation index: 14741
type: SIMPLE_ASSIGN
arm_off[130] = 1.6210618092523332 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14741};
  (data->simulationInfo->realParameter[132] /* arm_off[130] PARAM */) = (1.6210618092523332) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14742
type: SIMPLE_ASSIGN
theta[130] = pitch * r_init[130] + arm_off[130]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14742(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14742};
  (data->simulationInfo->realParameter[1636] /* theta[130] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1135] /* r_init[130] PARAM */)) + (data->simulationInfo->realParameter[132] /* arm_off[130] PARAM */);
  TRACE_POP
}

/*
equation index: 14743
type: SIMPLE_ASSIGN
arm_off[129] = 1.6084954386379742 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14743};
  (data->simulationInfo->realParameter[131] /* arm_off[129] PARAM */) = (1.6084954386379742) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14744
type: SIMPLE_ASSIGN
theta[129] = pitch * r_init[129] + arm_off[129]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14744(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14744};
  (data->simulationInfo->realParameter[1635] /* theta[129] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1134] /* r_init[129] PARAM */)) + (data->simulationInfo->realParameter[131] /* arm_off[129] PARAM */);
  TRACE_POP
}

/*
equation index: 14745
type: SIMPLE_ASSIGN
arm_off[128] = 1.595929068023615 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14745};
  (data->simulationInfo->realParameter[130] /* arm_off[128] PARAM */) = (1.595929068023615) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14746
type: SIMPLE_ASSIGN
theta[128] = pitch * r_init[128] + arm_off[128]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14746(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14746};
  (data->simulationInfo->realParameter[1634] /* theta[128] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1133] /* r_init[128] PARAM */)) + (data->simulationInfo->realParameter[130] /* arm_off[128] PARAM */);
  TRACE_POP
}

/*
equation index: 14747
type: SIMPLE_ASSIGN
arm_off[127] = 1.5833626974092558 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14747};
  (data->simulationInfo->realParameter[129] /* arm_off[127] PARAM */) = (1.5833626974092558) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14748
type: SIMPLE_ASSIGN
theta[127] = pitch * r_init[127] + arm_off[127]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14748};
  (data->simulationInfo->realParameter[1633] /* theta[127] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1132] /* r_init[127] PARAM */)) + (data->simulationInfo->realParameter[129] /* arm_off[127] PARAM */);
  TRACE_POP
}

/*
equation index: 14749
type: SIMPLE_ASSIGN
arm_off[126] = 1.5707963267948966 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14749};
  (data->simulationInfo->realParameter[128] /* arm_off[126] PARAM */) = (1.5707963267948966) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14750
type: SIMPLE_ASSIGN
theta[126] = pitch * r_init[126] + arm_off[126]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14750(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14750};
  (data->simulationInfo->realParameter[1632] /* theta[126] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1131] /* r_init[126] PARAM */)) + (data->simulationInfo->realParameter[128] /* arm_off[126] PARAM */);
  TRACE_POP
}

/*
equation index: 14751
type: SIMPLE_ASSIGN
arm_off[125] = 1.5582299561805373 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14751};
  (data->simulationInfo->realParameter[127] /* arm_off[125] PARAM */) = (1.5582299561805373) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14752
type: SIMPLE_ASSIGN
theta[125] = pitch * r_init[125] + arm_off[125]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14752(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14752};
  (data->simulationInfo->realParameter[1631] /* theta[125] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1130] /* r_init[125] PARAM */)) + (data->simulationInfo->realParameter[127] /* arm_off[125] PARAM */);
  TRACE_POP
}

/*
equation index: 14753
type: SIMPLE_ASSIGN
arm_off[124] = 1.5456635855661782 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14753};
  (data->simulationInfo->realParameter[126] /* arm_off[124] PARAM */) = (1.5456635855661782) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14754
type: SIMPLE_ASSIGN
theta[124] = pitch * r_init[124] + arm_off[124]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14754(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14754};
  (data->simulationInfo->realParameter[1630] /* theta[124] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1129] /* r_init[124] PARAM */)) + (data->simulationInfo->realParameter[126] /* arm_off[124] PARAM */);
  TRACE_POP
}

/*
equation index: 14755
type: SIMPLE_ASSIGN
arm_off[123] = 1.5330972149518192 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14755};
  (data->simulationInfo->realParameter[125] /* arm_off[123] PARAM */) = (1.5330972149518192) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14756
type: SIMPLE_ASSIGN
theta[123] = pitch * r_init[123] + arm_off[123]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14756};
  (data->simulationInfo->realParameter[1629] /* theta[123] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1128] /* r_init[123] PARAM */)) + (data->simulationInfo->realParameter[125] /* arm_off[123] PARAM */);
  TRACE_POP
}

/*
equation index: 14757
type: SIMPLE_ASSIGN
arm_off[122] = 1.5205308443374599 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14757};
  (data->simulationInfo->realParameter[124] /* arm_off[122] PARAM */) = (1.5205308443374599) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14758
type: SIMPLE_ASSIGN
theta[122] = pitch * r_init[122] + arm_off[122]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14758(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14758};
  (data->simulationInfo->realParameter[1628] /* theta[122] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1127] /* r_init[122] PARAM */)) + (data->simulationInfo->realParameter[124] /* arm_off[122] PARAM */);
  TRACE_POP
}

/*
equation index: 14759
type: SIMPLE_ASSIGN
arm_off[121] = 1.5079644737231006 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14759};
  (data->simulationInfo->realParameter[123] /* arm_off[121] PARAM */) = (1.5079644737231006) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14760
type: SIMPLE_ASSIGN
theta[121] = pitch * r_init[121] + arm_off[121]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14760(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14760};
  (data->simulationInfo->realParameter[1627] /* theta[121] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1126] /* r_init[121] PARAM */)) + (data->simulationInfo->realParameter[123] /* arm_off[121] PARAM */);
  TRACE_POP
}

/*
equation index: 14761
type: SIMPLE_ASSIGN
arm_off[120] = 1.4953981031087415 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14761};
  (data->simulationInfo->realParameter[122] /* arm_off[120] PARAM */) = (1.4953981031087415) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14762
type: SIMPLE_ASSIGN
theta[120] = pitch * r_init[120] + arm_off[120]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14762(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14762};
  (data->simulationInfo->realParameter[1626] /* theta[120] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1125] /* r_init[120] PARAM */)) + (data->simulationInfo->realParameter[122] /* arm_off[120] PARAM */);
  TRACE_POP
}

/*
equation index: 14763
type: SIMPLE_ASSIGN
arm_off[119] = 1.4828317324943823 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14763};
  (data->simulationInfo->realParameter[121] /* arm_off[119] PARAM */) = (1.4828317324943823) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14764
type: SIMPLE_ASSIGN
theta[119] = pitch * r_init[119] + arm_off[119]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14764};
  (data->simulationInfo->realParameter[1625] /* theta[119] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1124] /* r_init[119] PARAM */)) + (data->simulationInfo->realParameter[121] /* arm_off[119] PARAM */);
  TRACE_POP
}

/*
equation index: 14765
type: SIMPLE_ASSIGN
arm_off[118] = 1.4702653618800232 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14765};
  (data->simulationInfo->realParameter[120] /* arm_off[118] PARAM */) = (1.4702653618800232) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14766
type: SIMPLE_ASSIGN
theta[118] = pitch * r_init[118] + arm_off[118]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14766(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14766};
  (data->simulationInfo->realParameter[1624] /* theta[118] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1123] /* r_init[118] PARAM */)) + (data->simulationInfo->realParameter[120] /* arm_off[118] PARAM */);
  TRACE_POP
}

/*
equation index: 14767
type: SIMPLE_ASSIGN
arm_off[117] = 1.4576989912656642 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14767};
  (data->simulationInfo->realParameter[119] /* arm_off[117] PARAM */) = (1.4576989912656642) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14768
type: SIMPLE_ASSIGN
theta[117] = pitch * r_init[117] + arm_off[117]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14768(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14768};
  (data->simulationInfo->realParameter[1623] /* theta[117] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1122] /* r_init[117] PARAM */)) + (data->simulationInfo->realParameter[119] /* arm_off[117] PARAM */);
  TRACE_POP
}

/*
equation index: 14769
type: SIMPLE_ASSIGN
arm_off[116] = 1.4451326206513049 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14769};
  (data->simulationInfo->realParameter[118] /* arm_off[116] PARAM */) = (1.4451326206513049) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14770
type: SIMPLE_ASSIGN
theta[116] = pitch * r_init[116] + arm_off[116]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14770(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14770};
  (data->simulationInfo->realParameter[1622] /* theta[116] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1121] /* r_init[116] PARAM */)) + (data->simulationInfo->realParameter[118] /* arm_off[116] PARAM */);
  TRACE_POP
}

/*
equation index: 14771
type: SIMPLE_ASSIGN
arm_off[115] = 1.4325662500369456 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14771};
  (data->simulationInfo->realParameter[117] /* arm_off[115] PARAM */) = (1.4325662500369456) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14772
type: SIMPLE_ASSIGN
theta[115] = pitch * r_init[115] + arm_off[115]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14772};
  (data->simulationInfo->realParameter[1621] /* theta[115] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1120] /* r_init[115] PARAM */)) + (data->simulationInfo->realParameter[117] /* arm_off[115] PARAM */);
  TRACE_POP
}

/*
equation index: 14773
type: SIMPLE_ASSIGN
arm_off[114] = 1.4199998794225865 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14773};
  (data->simulationInfo->realParameter[116] /* arm_off[114] PARAM */) = (1.4199998794225865) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14774
type: SIMPLE_ASSIGN
theta[114] = pitch * r_init[114] + arm_off[114]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14774(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14774};
  (data->simulationInfo->realParameter[1620] /* theta[114] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1119] /* r_init[114] PARAM */)) + (data->simulationInfo->realParameter[116] /* arm_off[114] PARAM */);
  TRACE_POP
}

/*
equation index: 14775
type: SIMPLE_ASSIGN
arm_off[113] = 1.4074335088082273 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14775};
  (data->simulationInfo->realParameter[115] /* arm_off[113] PARAM */) = (1.4074335088082273) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14776
type: SIMPLE_ASSIGN
theta[113] = pitch * r_init[113] + arm_off[113]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14776(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14776};
  (data->simulationInfo->realParameter[1619] /* theta[113] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1118] /* r_init[113] PARAM */)) + (data->simulationInfo->realParameter[115] /* arm_off[113] PARAM */);
  TRACE_POP
}

/*
equation index: 14777
type: SIMPLE_ASSIGN
arm_off[112] = 1.3948671381938682 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14777};
  (data->simulationInfo->realParameter[114] /* arm_off[112] PARAM */) = (1.3948671381938682) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14778
type: SIMPLE_ASSIGN
theta[112] = pitch * r_init[112] + arm_off[112]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14778(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14778};
  (data->simulationInfo->realParameter[1618] /* theta[112] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1117] /* r_init[112] PARAM */)) + (data->simulationInfo->realParameter[114] /* arm_off[112] PARAM */);
  TRACE_POP
}

/*
equation index: 14779
type: SIMPLE_ASSIGN
arm_off[111] = 1.3823007675795091 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14779};
  (data->simulationInfo->realParameter[113] /* arm_off[111] PARAM */) = (1.3823007675795091) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14780
type: SIMPLE_ASSIGN
theta[111] = pitch * r_init[111] + arm_off[111]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14780};
  (data->simulationInfo->realParameter[1617] /* theta[111] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1116] /* r_init[111] PARAM */)) + (data->simulationInfo->realParameter[113] /* arm_off[111] PARAM */);
  TRACE_POP
}

/*
equation index: 14781
type: SIMPLE_ASSIGN
arm_off[110] = 1.3697343969651496 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14781};
  (data->simulationInfo->realParameter[112] /* arm_off[110] PARAM */) = (1.3697343969651496) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14782
type: SIMPLE_ASSIGN
theta[110] = pitch * r_init[110] + arm_off[110]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14782(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14782};
  (data->simulationInfo->realParameter[1616] /* theta[110] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1115] /* r_init[110] PARAM */)) + (data->simulationInfo->realParameter[112] /* arm_off[110] PARAM */);
  TRACE_POP
}

/*
equation index: 14783
type: SIMPLE_ASSIGN
arm_off[109] = 1.3571680263507906 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14783};
  (data->simulationInfo->realParameter[111] /* arm_off[109] PARAM */) = (1.3571680263507906) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14784
type: SIMPLE_ASSIGN
theta[109] = pitch * r_init[109] + arm_off[109]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14784(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14784};
  (data->simulationInfo->realParameter[1615] /* theta[109] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1114] /* r_init[109] PARAM */)) + (data->simulationInfo->realParameter[111] /* arm_off[109] PARAM */);
  TRACE_POP
}

/*
equation index: 14785
type: SIMPLE_ASSIGN
arm_off[108] = 1.3446016557364315 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14785};
  (data->simulationInfo->realParameter[110] /* arm_off[108] PARAM */) = (1.3446016557364315) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14786
type: SIMPLE_ASSIGN
theta[108] = pitch * r_init[108] + arm_off[108]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14786(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14786};
  (data->simulationInfo->realParameter[1614] /* theta[108] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1113] /* r_init[108] PARAM */)) + (data->simulationInfo->realParameter[110] /* arm_off[108] PARAM */);
  TRACE_POP
}

/*
equation index: 14787
type: SIMPLE_ASSIGN
arm_off[107] = 1.3320352851220723 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14787};
  (data->simulationInfo->realParameter[109] /* arm_off[107] PARAM */) = (1.3320352851220723) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14788
type: SIMPLE_ASSIGN
theta[107] = pitch * r_init[107] + arm_off[107]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14788};
  (data->simulationInfo->realParameter[1613] /* theta[107] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1112] /* r_init[107] PARAM */)) + (data->simulationInfo->realParameter[109] /* arm_off[107] PARAM */);
  TRACE_POP
}

/*
equation index: 14789
type: SIMPLE_ASSIGN
arm_off[106] = 1.3194689145077132 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14789};
  (data->simulationInfo->realParameter[108] /* arm_off[106] PARAM */) = (1.3194689145077132) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14790
type: SIMPLE_ASSIGN
theta[106] = pitch * r_init[106] + arm_off[106]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14790(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14790};
  (data->simulationInfo->realParameter[1612] /* theta[106] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1111] /* r_init[106] PARAM */)) + (data->simulationInfo->realParameter[108] /* arm_off[106] PARAM */);
  TRACE_POP
}

/*
equation index: 14791
type: SIMPLE_ASSIGN
arm_off[105] = 1.3069025438933541 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14791};
  (data->simulationInfo->realParameter[107] /* arm_off[105] PARAM */) = (1.3069025438933541) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14792
type: SIMPLE_ASSIGN
theta[105] = pitch * r_init[105] + arm_off[105]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14792(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14792};
  (data->simulationInfo->realParameter[1611] /* theta[105] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1110] /* r_init[105] PARAM */)) + (data->simulationInfo->realParameter[107] /* arm_off[105] PARAM */);
  TRACE_POP
}

/*
equation index: 14793
type: SIMPLE_ASSIGN
arm_off[104] = 1.2943361732789946 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14793};
  (data->simulationInfo->realParameter[106] /* arm_off[104] PARAM */) = (1.2943361732789946) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14794
type: SIMPLE_ASSIGN
theta[104] = pitch * r_init[104] + arm_off[104]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14794(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14794};
  (data->simulationInfo->realParameter[1610] /* theta[104] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1109] /* r_init[104] PARAM */)) + (data->simulationInfo->realParameter[106] /* arm_off[104] PARAM */);
  TRACE_POP
}

/*
equation index: 14795
type: SIMPLE_ASSIGN
arm_off[103] = 1.2817698026646356 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14795};
  (data->simulationInfo->realParameter[105] /* arm_off[103] PARAM */) = (1.2817698026646356) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14796
type: SIMPLE_ASSIGN
theta[103] = pitch * r_init[103] + arm_off[103]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14796};
  (data->simulationInfo->realParameter[1609] /* theta[103] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1108] /* r_init[103] PARAM */)) + (data->simulationInfo->realParameter[105] /* arm_off[103] PARAM */);
  TRACE_POP
}

/*
equation index: 14797
type: SIMPLE_ASSIGN
arm_off[102] = 1.2692034320502765 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14797};
  (data->simulationInfo->realParameter[104] /* arm_off[102] PARAM */) = (1.2692034320502765) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14798
type: SIMPLE_ASSIGN
theta[102] = pitch * r_init[102] + arm_off[102]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14798(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14798};
  (data->simulationInfo->realParameter[1608] /* theta[102] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1107] /* r_init[102] PARAM */)) + (data->simulationInfo->realParameter[104] /* arm_off[102] PARAM */);
  TRACE_POP
}

/*
equation index: 14799
type: SIMPLE_ASSIGN
arm_off[101] = 1.2566370614359172 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14799};
  (data->simulationInfo->realParameter[103] /* arm_off[101] PARAM */) = (1.2566370614359172) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14800
type: SIMPLE_ASSIGN
theta[101] = pitch * r_init[101] + arm_off[101]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14800(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14800};
  (data->simulationInfo->realParameter[1607] /* theta[101] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1106] /* r_init[101] PARAM */)) + (data->simulationInfo->realParameter[103] /* arm_off[101] PARAM */);
  TRACE_POP
}

/*
equation index: 14801
type: SIMPLE_ASSIGN
arm_off[100] = 1.2440706908215582 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14801};
  (data->simulationInfo->realParameter[102] /* arm_off[100] PARAM */) = (1.2440706908215582) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14802
type: SIMPLE_ASSIGN
theta[100] = pitch * r_init[100] + arm_off[100]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14802(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14802};
  (data->simulationInfo->realParameter[1606] /* theta[100] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1105] /* r_init[100] PARAM */)) + (data->simulationInfo->realParameter[102] /* arm_off[100] PARAM */);
  TRACE_POP
}

/*
equation index: 14803
type: SIMPLE_ASSIGN
arm_off[99] = 1.231504320207199 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14803};
  (data->simulationInfo->realParameter[101] /* arm_off[99] PARAM */) = (1.231504320207199) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14804
type: SIMPLE_ASSIGN
theta[99] = pitch * r_init[99] + arm_off[99]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14804};
  (data->simulationInfo->realParameter[1605] /* theta[99] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1104] /* r_init[99] PARAM */)) + (data->simulationInfo->realParameter[101] /* arm_off[99] PARAM */);
  TRACE_POP
}

/*
equation index: 14805
type: SIMPLE_ASSIGN
arm_off[98] = 1.2189379495928396 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14805};
  (data->simulationInfo->realParameter[100] /* arm_off[98] PARAM */) = (1.2189379495928396) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14806
type: SIMPLE_ASSIGN
theta[98] = pitch * r_init[98] + arm_off[98]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14806(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14806};
  (data->simulationInfo->realParameter[1604] /* theta[98] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1103] /* r_init[98] PARAM */)) + (data->simulationInfo->realParameter[100] /* arm_off[98] PARAM */);
  TRACE_POP
}

/*
equation index: 14807
type: SIMPLE_ASSIGN
arm_off[97] = 1.2063715789784806 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14807};
  (data->simulationInfo->realParameter[99] /* arm_off[97] PARAM */) = (1.2063715789784806) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14808
type: SIMPLE_ASSIGN
theta[97] = pitch * r_init[97] + arm_off[97]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14808(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14808};
  (data->simulationInfo->realParameter[1603] /* theta[97] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1102] /* r_init[97] PARAM */)) + (data->simulationInfo->realParameter[99] /* arm_off[97] PARAM */);
  TRACE_POP
}

/*
equation index: 14809
type: SIMPLE_ASSIGN
arm_off[96] = 1.1938052083641215 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14809};
  (data->simulationInfo->realParameter[98] /* arm_off[96] PARAM */) = (1.1938052083641215) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14810
type: SIMPLE_ASSIGN
theta[96] = pitch * r_init[96] + arm_off[96]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14810(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14810};
  (data->simulationInfo->realParameter[1602] /* theta[96] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1101] /* r_init[96] PARAM */)) + (data->simulationInfo->realParameter[98] /* arm_off[96] PARAM */);
  TRACE_POP
}

/*
equation index: 14811
type: SIMPLE_ASSIGN
arm_off[95] = 1.1812388377497622 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14811};
  (data->simulationInfo->realParameter[97] /* arm_off[95] PARAM */) = (1.1812388377497622) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14812
type: SIMPLE_ASSIGN
theta[95] = pitch * r_init[95] + arm_off[95]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14812};
  (data->simulationInfo->realParameter[1601] /* theta[95] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1100] /* r_init[95] PARAM */)) + (data->simulationInfo->realParameter[97] /* arm_off[95] PARAM */);
  TRACE_POP
}

/*
equation index: 14813
type: SIMPLE_ASSIGN
arm_off[94] = 1.168672467135403 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14813};
  (data->simulationInfo->realParameter[96] /* arm_off[94] PARAM */) = (1.168672467135403) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14814
type: SIMPLE_ASSIGN
theta[94] = pitch * r_init[94] + arm_off[94]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14814(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14814};
  (data->simulationInfo->realParameter[1600] /* theta[94] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */)) + (data->simulationInfo->realParameter[96] /* arm_off[94] PARAM */);
  TRACE_POP
}

/*
equation index: 14815
type: SIMPLE_ASSIGN
arm_off[93] = 1.156106096521044 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14815};
  (data->simulationInfo->realParameter[95] /* arm_off[93] PARAM */) = (1.156106096521044) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14816
type: SIMPLE_ASSIGN
theta[93] = pitch * r_init[93] + arm_off[93]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14816(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14816};
  (data->simulationInfo->realParameter[1599] /* theta[93] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */)) + (data->simulationInfo->realParameter[95] /* arm_off[93] PARAM */);
  TRACE_POP
}

/*
equation index: 14817
type: SIMPLE_ASSIGN
arm_off[92] = 1.1435397259066846 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14817};
  (data->simulationInfo->realParameter[94] /* arm_off[92] PARAM */) = (1.1435397259066846) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14818
type: SIMPLE_ASSIGN
theta[92] = pitch * r_init[92] + arm_off[92]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14818(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14818};
  (data->simulationInfo->realParameter[1598] /* theta[92] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */)) + (data->simulationInfo->realParameter[94] /* arm_off[92] PARAM */);
  TRACE_POP
}

/*
equation index: 14819
type: SIMPLE_ASSIGN
arm_off[91] = 1.1309733552923256 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14819};
  (data->simulationInfo->realParameter[93] /* arm_off[91] PARAM */) = (1.1309733552923256) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14820
type: SIMPLE_ASSIGN
theta[91] = pitch * r_init[91] + arm_off[91]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14820};
  (data->simulationInfo->realParameter[1597] /* theta[91] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */)) + (data->simulationInfo->realParameter[93] /* arm_off[91] PARAM */);
  TRACE_POP
}

/*
equation index: 14821
type: SIMPLE_ASSIGN
arm_off[90] = 1.1184069846779665 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14821};
  (data->simulationInfo->realParameter[92] /* arm_off[90] PARAM */) = (1.1184069846779665) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14822
type: SIMPLE_ASSIGN
theta[90] = pitch * r_init[90] + arm_off[90]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14822(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14822};
  (data->simulationInfo->realParameter[1596] /* theta[90] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */)) + (data->simulationInfo->realParameter[92] /* arm_off[90] PARAM */);
  TRACE_POP
}

/*
equation index: 14823
type: SIMPLE_ASSIGN
arm_off[89] = 1.105840614063607 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14823};
  (data->simulationInfo->realParameter[91] /* arm_off[89] PARAM */) = (1.105840614063607) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14824
type: SIMPLE_ASSIGN
theta[89] = pitch * r_init[89] + arm_off[89]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14824(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14824};
  (data->simulationInfo->realParameter[1595] /* theta[89] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */)) + (data->simulationInfo->realParameter[91] /* arm_off[89] PARAM */);
  TRACE_POP
}

/*
equation index: 14825
type: SIMPLE_ASSIGN
arm_off[88] = 1.093274243449248 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14825};
  (data->simulationInfo->realParameter[90] /* arm_off[88] PARAM */) = (1.093274243449248) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14826
type: SIMPLE_ASSIGN
theta[88] = pitch * r_init[88] + arm_off[88]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14826(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14826};
  (data->simulationInfo->realParameter[1594] /* theta[88] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */)) + (data->simulationInfo->realParameter[90] /* arm_off[88] PARAM */);
  TRACE_POP
}

/*
equation index: 14827
type: SIMPLE_ASSIGN
arm_off[87] = 1.080707872834889 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14827};
  (data->simulationInfo->realParameter[89] /* arm_off[87] PARAM */) = (1.080707872834889) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14828
type: SIMPLE_ASSIGN
theta[87] = pitch * r_init[87] + arm_off[87]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14828};
  (data->simulationInfo->realParameter[1593] /* theta[87] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */)) + (data->simulationInfo->realParameter[89] /* arm_off[87] PARAM */);
  TRACE_POP
}

/*
equation index: 14829
type: SIMPLE_ASSIGN
arm_off[86] = 1.0681415022205296 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14829};
  (data->simulationInfo->realParameter[88] /* arm_off[86] PARAM */) = (1.0681415022205296) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14830
type: SIMPLE_ASSIGN
theta[86] = pitch * r_init[86] + arm_off[86]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14830(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14830};
  (data->simulationInfo->realParameter[1592] /* theta[86] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */)) + (data->simulationInfo->realParameter[88] /* arm_off[86] PARAM */);
  TRACE_POP
}

/*
equation index: 14831
type: SIMPLE_ASSIGN
arm_off[85] = 1.0555751316061706 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14831};
  (data->simulationInfo->realParameter[87] /* arm_off[85] PARAM */) = (1.0555751316061706) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14832
type: SIMPLE_ASSIGN
theta[85] = pitch * r_init[85] + arm_off[85]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14832(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14832};
  (data->simulationInfo->realParameter[1591] /* theta[85] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */)) + (data->simulationInfo->realParameter[87] /* arm_off[85] PARAM */);
  TRACE_POP
}

/*
equation index: 14833
type: SIMPLE_ASSIGN
arm_off[84] = 1.0430087609918115 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14833};
  (data->simulationInfo->realParameter[86] /* arm_off[84] PARAM */) = (1.0430087609918115) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14834
type: SIMPLE_ASSIGN
theta[84] = pitch * r_init[84] + arm_off[84]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14834(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14834};
  (data->simulationInfo->realParameter[1590] /* theta[84] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */)) + (data->simulationInfo->realParameter[86] /* arm_off[84] PARAM */);
  TRACE_POP
}

/*
equation index: 14835
type: SIMPLE_ASSIGN
arm_off[83] = 1.030442390377452 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14835};
  (data->simulationInfo->realParameter[85] /* arm_off[83] PARAM */) = (1.030442390377452) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14836
type: SIMPLE_ASSIGN
theta[83] = pitch * r_init[83] + arm_off[83]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14836};
  (data->simulationInfo->realParameter[1589] /* theta[83] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */)) + (data->simulationInfo->realParameter[85] /* arm_off[83] PARAM */);
  TRACE_POP
}

/*
equation index: 14837
type: SIMPLE_ASSIGN
arm_off[82] = 1.017876019763093 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14837};
  (data->simulationInfo->realParameter[84] /* arm_off[82] PARAM */) = (1.017876019763093) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14838
type: SIMPLE_ASSIGN
theta[82] = pitch * r_init[82] + arm_off[82]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14838(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14838};
  (data->simulationInfo->realParameter[1588] /* theta[82] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */)) + (data->simulationInfo->realParameter[84] /* arm_off[82] PARAM */);
  TRACE_POP
}

/*
equation index: 14839
type: SIMPLE_ASSIGN
arm_off[81] = 1.0053096491487339 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14839};
  (data->simulationInfo->realParameter[83] /* arm_off[81] PARAM */) = (1.0053096491487339) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14840
type: SIMPLE_ASSIGN
theta[81] = pitch * r_init[81] + arm_off[81]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14840(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14840};
  (data->simulationInfo->realParameter[1587] /* theta[81] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */)) + (data->simulationInfo->realParameter[83] /* arm_off[81] PARAM */);
  TRACE_POP
}

/*
equation index: 14841
type: SIMPLE_ASSIGN
arm_off[80] = 0.9927432785343747 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14841};
  (data->simulationInfo->realParameter[82] /* arm_off[80] PARAM */) = (0.9927432785343747) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14842
type: SIMPLE_ASSIGN
theta[80] = pitch * r_init[80] + arm_off[80]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14842(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14842};
  (data->simulationInfo->realParameter[1586] /* theta[80] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */)) + (data->simulationInfo->realParameter[82] /* arm_off[80] PARAM */);
  TRACE_POP
}

/*
equation index: 14843
type: SIMPLE_ASSIGN
arm_off[79] = 0.9801769079200154 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14843};
  (data->simulationInfo->realParameter[81] /* arm_off[79] PARAM */) = (0.9801769079200154) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14844
type: SIMPLE_ASSIGN
theta[79] = pitch * r_init[79] + arm_off[79]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14844};
  (data->simulationInfo->realParameter[1585] /* theta[79] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */)) + (data->simulationInfo->realParameter[81] /* arm_off[79] PARAM */);
  TRACE_POP
}

/*
equation index: 14845
type: SIMPLE_ASSIGN
arm_off[78] = 0.9676105373056563 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14845};
  (data->simulationInfo->realParameter[80] /* arm_off[78] PARAM */) = (0.9676105373056563) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14846
type: SIMPLE_ASSIGN
theta[78] = pitch * r_init[78] + arm_off[78]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14846(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14846};
  (data->simulationInfo->realParameter[1584] /* theta[78] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */)) + (data->simulationInfo->realParameter[80] /* arm_off[78] PARAM */);
  TRACE_POP
}

/*
equation index: 14847
type: SIMPLE_ASSIGN
arm_off[77] = 0.9550441666912971 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14847};
  (data->simulationInfo->realParameter[79] /* arm_off[77] PARAM */) = (0.9550441666912971) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14848
type: SIMPLE_ASSIGN
theta[77] = pitch * r_init[77] + arm_off[77]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14848(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14848};
  (data->simulationInfo->realParameter[1583] /* theta[77] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */)) + (data->simulationInfo->realParameter[79] /* arm_off[77] PARAM */);
  TRACE_POP
}

/*
equation index: 14849
type: SIMPLE_ASSIGN
arm_off[76] = 0.9424777960769379 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14849};
  (data->simulationInfo->realParameter[78] /* arm_off[76] PARAM */) = (0.9424777960769379) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14850
type: SIMPLE_ASSIGN
theta[76] = pitch * r_init[76] + arm_off[76]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14850(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14850};
  (data->simulationInfo->realParameter[1582] /* theta[76] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */)) + (data->simulationInfo->realParameter[78] /* arm_off[76] PARAM */);
  TRACE_POP
}

/*
equation index: 14851
type: SIMPLE_ASSIGN
arm_off[75] = 0.9299114254625788 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14851};
  (data->simulationInfo->realParameter[77] /* arm_off[75] PARAM */) = (0.9299114254625788) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14852
type: SIMPLE_ASSIGN
theta[75] = pitch * r_init[75] + arm_off[75]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14852};
  (data->simulationInfo->realParameter[1581] /* theta[75] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */)) + (data->simulationInfo->realParameter[77] /* arm_off[75] PARAM */);
  TRACE_POP
}

/*
equation index: 14853
type: SIMPLE_ASSIGN
arm_off[74] = 0.9173450548482196 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14853};
  (data->simulationInfo->realParameter[76] /* arm_off[74] PARAM */) = (0.9173450548482196) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14854
type: SIMPLE_ASSIGN
theta[74] = pitch * r_init[74] + arm_off[74]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14854(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14854};
  (data->simulationInfo->realParameter[1580] /* theta[74] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */)) + (data->simulationInfo->realParameter[76] /* arm_off[74] PARAM */);
  TRACE_POP
}

/*
equation index: 14855
type: SIMPLE_ASSIGN
arm_off[73] = 0.9047786842338604 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14855};
  (data->simulationInfo->realParameter[75] /* arm_off[73] PARAM */) = (0.9047786842338604) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14856
type: SIMPLE_ASSIGN
theta[73] = pitch * r_init[73] + arm_off[73]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14856(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14856};
  (data->simulationInfo->realParameter[1579] /* theta[73] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */)) + (data->simulationInfo->realParameter[75] /* arm_off[73] PARAM */);
  TRACE_POP
}

/*
equation index: 14857
type: SIMPLE_ASSIGN
arm_off[72] = 0.8922123136195013 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14857};
  (data->simulationInfo->realParameter[74] /* arm_off[72] PARAM */) = (0.8922123136195013) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14858
type: SIMPLE_ASSIGN
theta[72] = pitch * r_init[72] + arm_off[72]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14858(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14858};
  (data->simulationInfo->realParameter[1578] /* theta[72] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */)) + (data->simulationInfo->realParameter[74] /* arm_off[72] PARAM */);
  TRACE_POP
}

/*
equation index: 14859
type: SIMPLE_ASSIGN
arm_off[71] = 0.8796459430051421 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14859};
  (data->simulationInfo->realParameter[73] /* arm_off[71] PARAM */) = (0.8796459430051421) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14860
type: SIMPLE_ASSIGN
theta[71] = pitch * r_init[71] + arm_off[71]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14860};
  (data->simulationInfo->realParameter[1577] /* theta[71] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */)) + (data->simulationInfo->realParameter[73] /* arm_off[71] PARAM */);
  TRACE_POP
}

/*
equation index: 14861
type: SIMPLE_ASSIGN
arm_off[70] = 0.8670795723907829 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14861};
  (data->simulationInfo->realParameter[72] /* arm_off[70] PARAM */) = (0.8670795723907829) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14862
type: SIMPLE_ASSIGN
theta[70] = pitch * r_init[70] + arm_off[70]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14862(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14862};
  (data->simulationInfo->realParameter[1576] /* theta[70] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */)) + (data->simulationInfo->realParameter[72] /* arm_off[70] PARAM */);
  TRACE_POP
}

/*
equation index: 14863
type: SIMPLE_ASSIGN
arm_off[69] = 0.8545132017764238 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14863};
  (data->simulationInfo->realParameter[71] /* arm_off[69] PARAM */) = (0.8545132017764238) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14864
type: SIMPLE_ASSIGN
theta[69] = pitch * r_init[69] + arm_off[69]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14864(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14864};
  (data->simulationInfo->realParameter[1575] /* theta[69] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */)) + (data->simulationInfo->realParameter[71] /* arm_off[69] PARAM */);
  TRACE_POP
}

/*
equation index: 14865
type: SIMPLE_ASSIGN
arm_off[68] = 0.8419468311620646 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14865};
  (data->simulationInfo->realParameter[70] /* arm_off[68] PARAM */) = (0.8419468311620646) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14866
type: SIMPLE_ASSIGN
theta[68] = pitch * r_init[68] + arm_off[68]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14866(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14866};
  (data->simulationInfo->realParameter[1574] /* theta[68] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */)) + (data->simulationInfo->realParameter[70] /* arm_off[68] PARAM */);
  TRACE_POP
}

/*
equation index: 14867
type: SIMPLE_ASSIGN
arm_off[67] = 0.8293804605477054 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14867};
  (data->simulationInfo->realParameter[69] /* arm_off[67] PARAM */) = (0.8293804605477054) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14868
type: SIMPLE_ASSIGN
theta[67] = pitch * r_init[67] + arm_off[67]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14868};
  (data->simulationInfo->realParameter[1573] /* theta[67] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */)) + (data->simulationInfo->realParameter[69] /* arm_off[67] PARAM */);
  TRACE_POP
}

/*
equation index: 14869
type: SIMPLE_ASSIGN
arm_off[66] = 0.8168140899333461 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14869};
  (data->simulationInfo->realParameter[68] /* arm_off[66] PARAM */) = (0.8168140899333461) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14870
type: SIMPLE_ASSIGN
theta[66] = pitch * r_init[66] + arm_off[66]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14870(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14870};
  (data->simulationInfo->realParameter[1572] /* theta[66] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */)) + (data->simulationInfo->realParameter[68] /* arm_off[66] PARAM */);
  TRACE_POP
}

/*
equation index: 14871
type: SIMPLE_ASSIGN
arm_off[65] = 0.8042477193189871 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14871};
  (data->simulationInfo->realParameter[67] /* arm_off[65] PARAM */) = (0.8042477193189871) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14872
type: SIMPLE_ASSIGN
theta[65] = pitch * r_init[65] + arm_off[65]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14872(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14872};
  (data->simulationInfo->realParameter[1571] /* theta[65] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */)) + (data->simulationInfo->realParameter[67] /* arm_off[65] PARAM */);
  TRACE_POP
}

/*
equation index: 14873
type: SIMPLE_ASSIGN
arm_off[64] = 0.7916813487046279 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14873};
  (data->simulationInfo->realParameter[66] /* arm_off[64] PARAM */) = (0.7916813487046279) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14874
type: SIMPLE_ASSIGN
theta[64] = pitch * r_init[64] + arm_off[64]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14874(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14874};
  (data->simulationInfo->realParameter[1570] /* theta[64] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */)) + (data->simulationInfo->realParameter[66] /* arm_off[64] PARAM */);
  TRACE_POP
}

/*
equation index: 14875
type: SIMPLE_ASSIGN
arm_off[63] = 0.7791149780902686 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14875};
  (data->simulationInfo->realParameter[65] /* arm_off[63] PARAM */) = (0.7791149780902686) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14876
type: SIMPLE_ASSIGN
theta[63] = pitch * r_init[63] + arm_off[63]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14876};
  (data->simulationInfo->realParameter[1569] /* theta[63] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */)) + (data->simulationInfo->realParameter[65] /* arm_off[63] PARAM */);
  TRACE_POP
}

/*
equation index: 14877
type: SIMPLE_ASSIGN
arm_off[62] = 0.7665486074759096 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14877};
  (data->simulationInfo->realParameter[64] /* arm_off[62] PARAM */) = (0.7665486074759096) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14878
type: SIMPLE_ASSIGN
theta[62] = pitch * r_init[62] + arm_off[62]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14878(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14878};
  (data->simulationInfo->realParameter[1568] /* theta[62] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */)) + (data->simulationInfo->realParameter[64] /* arm_off[62] PARAM */);
  TRACE_POP
}

/*
equation index: 14879
type: SIMPLE_ASSIGN
arm_off[61] = 0.7539822368615503 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14879};
  (data->simulationInfo->realParameter[63] /* arm_off[61] PARAM */) = (0.7539822368615503) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14880
type: SIMPLE_ASSIGN
theta[61] = pitch * r_init[61] + arm_off[61]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14880(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14880};
  (data->simulationInfo->realParameter[1567] /* theta[61] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */)) + (data->simulationInfo->realParameter[63] /* arm_off[61] PARAM */);
  TRACE_POP
}

/*
equation index: 14881
type: SIMPLE_ASSIGN
arm_off[60] = 0.7414158662471911 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14881};
  (data->simulationInfo->realParameter[62] /* arm_off[60] PARAM */) = (0.7414158662471911) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14882
type: SIMPLE_ASSIGN
theta[60] = pitch * r_init[60] + arm_off[60]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14882(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14882};
  (data->simulationInfo->realParameter[1566] /* theta[60] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */)) + (data->simulationInfo->realParameter[62] /* arm_off[60] PARAM */);
  TRACE_POP
}

/*
equation index: 14883
type: SIMPLE_ASSIGN
arm_off[59] = 0.7288494956328321 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14883};
  (data->simulationInfo->realParameter[61] /* arm_off[59] PARAM */) = (0.7288494956328321) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14884
type: SIMPLE_ASSIGN
theta[59] = pitch * r_init[59] + arm_off[59]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14884};
  (data->simulationInfo->realParameter[1565] /* theta[59] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */)) + (data->simulationInfo->realParameter[61] /* arm_off[59] PARAM */);
  TRACE_POP
}

/*
equation index: 14885
type: SIMPLE_ASSIGN
arm_off[58] = 0.7162831250184728 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14885};
  (data->simulationInfo->realParameter[60] /* arm_off[58] PARAM */) = (0.7162831250184728) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14886
type: SIMPLE_ASSIGN
theta[58] = pitch * r_init[58] + arm_off[58]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14886(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14886};
  (data->simulationInfo->realParameter[1564] /* theta[58] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */)) + (data->simulationInfo->realParameter[60] /* arm_off[58] PARAM */);
  TRACE_POP
}

/*
equation index: 14887
type: SIMPLE_ASSIGN
arm_off[57] = 0.7037167544041136 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14887};
  (data->simulationInfo->realParameter[59] /* arm_off[57] PARAM */) = (0.7037167544041136) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14888
type: SIMPLE_ASSIGN
theta[57] = pitch * r_init[57] + arm_off[57]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14888(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14888};
  (data->simulationInfo->realParameter[1563] /* theta[57] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */)) + (data->simulationInfo->realParameter[59] /* arm_off[57] PARAM */);
  TRACE_POP
}

/*
equation index: 14889
type: SIMPLE_ASSIGN
arm_off[56] = 0.6911503837897546 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14889};
  (data->simulationInfo->realParameter[58] /* arm_off[56] PARAM */) = (0.6911503837897546) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14890
type: SIMPLE_ASSIGN
theta[56] = pitch * r_init[56] + arm_off[56]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14890(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14890};
  (data->simulationInfo->realParameter[1562] /* theta[56] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */)) + (data->simulationInfo->realParameter[58] /* arm_off[56] PARAM */);
  TRACE_POP
}

/*
equation index: 14891
type: SIMPLE_ASSIGN
arm_off[55] = 0.6785840131753953 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14891};
  (data->simulationInfo->realParameter[57] /* arm_off[55] PARAM */) = (0.6785840131753953) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14892
type: SIMPLE_ASSIGN
theta[55] = pitch * r_init[55] + arm_off[55]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14892};
  (data->simulationInfo->realParameter[1561] /* theta[55] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */)) + (data->simulationInfo->realParameter[57] /* arm_off[55] PARAM */);
  TRACE_POP
}

/*
equation index: 14893
type: SIMPLE_ASSIGN
arm_off[54] = 0.6660176425610361 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14893};
  (data->simulationInfo->realParameter[56] /* arm_off[54] PARAM */) = (0.6660176425610361) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14894
type: SIMPLE_ASSIGN
theta[54] = pitch * r_init[54] + arm_off[54]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14894(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14894};
  (data->simulationInfo->realParameter[1560] /* theta[54] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */)) + (data->simulationInfo->realParameter[56] /* arm_off[54] PARAM */);
  TRACE_POP
}

/*
equation index: 14895
type: SIMPLE_ASSIGN
arm_off[53] = 0.6534512719466771 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14895};
  (data->simulationInfo->realParameter[55] /* arm_off[53] PARAM */) = (0.6534512719466771) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14896
type: SIMPLE_ASSIGN
theta[53] = pitch * r_init[53] + arm_off[53]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14896(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14896};
  (data->simulationInfo->realParameter[1559] /* theta[53] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */)) + (data->simulationInfo->realParameter[55] /* arm_off[53] PARAM */);
  TRACE_POP
}

/*
equation index: 14897
type: SIMPLE_ASSIGN
arm_off[52] = 0.6408849013323178 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14897};
  (data->simulationInfo->realParameter[54] /* arm_off[52] PARAM */) = (0.6408849013323178) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14898
type: SIMPLE_ASSIGN
theta[52] = pitch * r_init[52] + arm_off[52]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14898(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14898};
  (data->simulationInfo->realParameter[1558] /* theta[52] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */)) + (data->simulationInfo->realParameter[54] /* arm_off[52] PARAM */);
  TRACE_POP
}

/*
equation index: 14899
type: SIMPLE_ASSIGN
arm_off[51] = 0.6283185307179586 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14899};
  (data->simulationInfo->realParameter[53] /* arm_off[51] PARAM */) = (0.6283185307179586) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14900
type: SIMPLE_ASSIGN
theta[51] = pitch * r_init[51] + arm_off[51]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14900};
  (data->simulationInfo->realParameter[1557] /* theta[51] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */)) + (data->simulationInfo->realParameter[53] /* arm_off[51] PARAM */);
  TRACE_POP
}

/*
equation index: 14901
type: SIMPLE_ASSIGN
arm_off[50] = 0.6157521601035995 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14901};
  (data->simulationInfo->realParameter[52] /* arm_off[50] PARAM */) = (0.6157521601035995) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14902
type: SIMPLE_ASSIGN
theta[50] = pitch * r_init[50] + arm_off[50]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14902(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14902};
  (data->simulationInfo->realParameter[1556] /* theta[50] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */)) + (data->simulationInfo->realParameter[52] /* arm_off[50] PARAM */);
  TRACE_POP
}

/*
equation index: 14903
type: SIMPLE_ASSIGN
arm_off[49] = 0.6031857894892403 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14903};
  (data->simulationInfo->realParameter[51] /* arm_off[49] PARAM */) = (0.6031857894892403) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14904
type: SIMPLE_ASSIGN
theta[49] = pitch * r_init[49] + arm_off[49]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14904(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14904};
  (data->simulationInfo->realParameter[1555] /* theta[49] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */)) + (data->simulationInfo->realParameter[51] /* arm_off[49] PARAM */);
  TRACE_POP
}

/*
equation index: 14905
type: SIMPLE_ASSIGN
arm_off[48] = 0.5906194188748811 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14905};
  (data->simulationInfo->realParameter[50] /* arm_off[48] PARAM */) = (0.5906194188748811) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14906
type: SIMPLE_ASSIGN
theta[48] = pitch * r_init[48] + arm_off[48]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14906(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14906};
  (data->simulationInfo->realParameter[1554] /* theta[48] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */)) + (data->simulationInfo->realParameter[50] /* arm_off[48] PARAM */);
  TRACE_POP
}

/*
equation index: 14907
type: SIMPLE_ASSIGN
arm_off[47] = 0.578053048260522 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14907};
  (data->simulationInfo->realParameter[49] /* arm_off[47] PARAM */) = (0.578053048260522) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14908
type: SIMPLE_ASSIGN
theta[47] = pitch * r_init[47] + arm_off[47]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14908};
  (data->simulationInfo->realParameter[1553] /* theta[47] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */)) + (data->simulationInfo->realParameter[49] /* arm_off[47] PARAM */);
  TRACE_POP
}

/*
equation index: 14909
type: SIMPLE_ASSIGN
arm_off[46] = 0.5654866776461628 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14909};
  (data->simulationInfo->realParameter[48] /* arm_off[46] PARAM */) = (0.5654866776461628) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14910
type: SIMPLE_ASSIGN
theta[46] = pitch * r_init[46] + arm_off[46]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14910(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14910};
  (data->simulationInfo->realParameter[1552] /* theta[46] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */)) + (data->simulationInfo->realParameter[48] /* arm_off[46] PARAM */);
  TRACE_POP
}

/*
equation index: 14911
type: SIMPLE_ASSIGN
arm_off[45] = 0.5529203070318035 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14911};
  (data->simulationInfo->realParameter[47] /* arm_off[45] PARAM */) = (0.5529203070318035) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14912
type: SIMPLE_ASSIGN
theta[45] = pitch * r_init[45] + arm_off[45]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14912(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14912};
  (data->simulationInfo->realParameter[1551] /* theta[45] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */)) + (data->simulationInfo->realParameter[47] /* arm_off[45] PARAM */);
  TRACE_POP
}

/*
equation index: 14913
type: SIMPLE_ASSIGN
arm_off[44] = 0.5403539364174444 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14913};
  (data->simulationInfo->realParameter[46] /* arm_off[44] PARAM */) = (0.5403539364174444) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14914
type: SIMPLE_ASSIGN
theta[44] = pitch * r_init[44] + arm_off[44]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14914(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14914};
  (data->simulationInfo->realParameter[1550] /* theta[44] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */)) + (data->simulationInfo->realParameter[46] /* arm_off[44] PARAM */);
  TRACE_POP
}

/*
equation index: 14915
type: SIMPLE_ASSIGN
arm_off[43] = 0.5277875658030853 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14915};
  (data->simulationInfo->realParameter[45] /* arm_off[43] PARAM */) = (0.5277875658030853) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14916
type: SIMPLE_ASSIGN
theta[43] = pitch * r_init[43] + arm_off[43]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14916};
  (data->simulationInfo->realParameter[1549] /* theta[43] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */)) + (data->simulationInfo->realParameter[45] /* arm_off[43] PARAM */);
  TRACE_POP
}

/*
equation index: 14917
type: SIMPLE_ASSIGN
arm_off[42] = 0.515221195188726 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14917};
  (data->simulationInfo->realParameter[44] /* arm_off[42] PARAM */) = (0.515221195188726) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14918
type: SIMPLE_ASSIGN
theta[42] = pitch * r_init[42] + arm_off[42]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14918(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14918};
  (data->simulationInfo->realParameter[1548] /* theta[42] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */)) + (data->simulationInfo->realParameter[44] /* arm_off[42] PARAM */);
  TRACE_POP
}

/*
equation index: 14919
type: SIMPLE_ASSIGN
arm_off[41] = 0.5026548245743669 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14919};
  (data->simulationInfo->realParameter[43] /* arm_off[41] PARAM */) = (0.5026548245743669) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14920
type: SIMPLE_ASSIGN
theta[41] = pitch * r_init[41] + arm_off[41]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14920(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14920};
  (data->simulationInfo->realParameter[1547] /* theta[41] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */)) + (data->simulationInfo->realParameter[43] /* arm_off[41] PARAM */);
  TRACE_POP
}

/*
equation index: 14921
type: SIMPLE_ASSIGN
arm_off[40] = 0.4900884539600077 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14921};
  (data->simulationInfo->realParameter[42] /* arm_off[40] PARAM */) = (0.4900884539600077) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14922
type: SIMPLE_ASSIGN
theta[40] = pitch * r_init[40] + arm_off[40]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14922(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14922};
  (data->simulationInfo->realParameter[1546] /* theta[40] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */)) + (data->simulationInfo->realParameter[42] /* arm_off[40] PARAM */);
  TRACE_POP
}

/*
equation index: 14923
type: SIMPLE_ASSIGN
arm_off[39] = 0.47752208334564855 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14923};
  (data->simulationInfo->realParameter[41] /* arm_off[39] PARAM */) = (0.47752208334564855) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14924
type: SIMPLE_ASSIGN
theta[39] = pitch * r_init[39] + arm_off[39]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14924};
  (data->simulationInfo->realParameter[1545] /* theta[39] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */)) + (data->simulationInfo->realParameter[41] /* arm_off[39] PARAM */);
  TRACE_POP
}

/*
equation index: 14925
type: SIMPLE_ASSIGN
arm_off[38] = 0.4649557127312894 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14925};
  (data->simulationInfo->realParameter[40] /* arm_off[38] PARAM */) = (0.4649557127312894) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14926
type: SIMPLE_ASSIGN
theta[38] = pitch * r_init[38] + arm_off[38]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14926(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14926};
  (data->simulationInfo->realParameter[1544] /* theta[38] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */)) + (data->simulationInfo->realParameter[40] /* arm_off[38] PARAM */);
  TRACE_POP
}

/*
equation index: 14927
type: SIMPLE_ASSIGN
arm_off[37] = 0.4523893421169302 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14927};
  (data->simulationInfo->realParameter[39] /* arm_off[37] PARAM */) = (0.4523893421169302) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14928
type: SIMPLE_ASSIGN
theta[37] = pitch * r_init[37] + arm_off[37]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14928(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14928};
  (data->simulationInfo->realParameter[1543] /* theta[37] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */)) + (data->simulationInfo->realParameter[39] /* arm_off[37] PARAM */);
  TRACE_POP
}

/*
equation index: 14929
type: SIMPLE_ASSIGN
arm_off[36] = 0.43982297150257105 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14929};
  (data->simulationInfo->realParameter[38] /* arm_off[36] PARAM */) = (0.43982297150257105) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14930
type: SIMPLE_ASSIGN
theta[36] = pitch * r_init[36] + arm_off[36]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14930(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14930};
  (data->simulationInfo->realParameter[1542] /* theta[36] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */)) + (data->simulationInfo->realParameter[38] /* arm_off[36] PARAM */);
  TRACE_POP
}

/*
equation index: 14931
type: SIMPLE_ASSIGN
arm_off[35] = 0.4272566008882119 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14931};
  (data->simulationInfo->realParameter[37] /* arm_off[35] PARAM */) = (0.4272566008882119) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14932
type: SIMPLE_ASSIGN
theta[35] = pitch * r_init[35] + arm_off[35]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14932};
  (data->simulationInfo->realParameter[1541] /* theta[35] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */)) + (data->simulationInfo->realParameter[37] /* arm_off[35] PARAM */);
  TRACE_POP
}

/*
equation index: 14933
type: SIMPLE_ASSIGN
arm_off[34] = 0.4146902302738527 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14933};
  (data->simulationInfo->realParameter[36] /* arm_off[34] PARAM */) = (0.4146902302738527) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14934
type: SIMPLE_ASSIGN
theta[34] = pitch * r_init[34] + arm_off[34]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14934(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14934};
  (data->simulationInfo->realParameter[1540] /* theta[34] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */)) + (data->simulationInfo->realParameter[36] /* arm_off[34] PARAM */);
  TRACE_POP
}

/*
equation index: 14935
type: SIMPLE_ASSIGN
arm_off[33] = 0.40212385965949354 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14935};
  (data->simulationInfo->realParameter[35] /* arm_off[33] PARAM */) = (0.40212385965949354) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14936
type: SIMPLE_ASSIGN
theta[33] = pitch * r_init[33] + arm_off[33]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14936(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14936};
  (data->simulationInfo->realParameter[1539] /* theta[33] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */)) + (data->simulationInfo->realParameter[35] /* arm_off[33] PARAM */);
  TRACE_POP
}

/*
equation index: 14937
type: SIMPLE_ASSIGN
arm_off[32] = 0.3895574890451343 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14937};
  (data->simulationInfo->realParameter[34] /* arm_off[32] PARAM */) = (0.3895574890451343) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14938
type: SIMPLE_ASSIGN
theta[32] = pitch * r_init[32] + arm_off[32]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14938(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14938};
  (data->simulationInfo->realParameter[1538] /* theta[32] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */)) + (data->simulationInfo->realParameter[34] /* arm_off[32] PARAM */);
  TRACE_POP
}

/*
equation index: 14939
type: SIMPLE_ASSIGN
arm_off[31] = 0.37699111843077515 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14939};
  (data->simulationInfo->realParameter[33] /* arm_off[31] PARAM */) = (0.37699111843077515) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14940
type: SIMPLE_ASSIGN
theta[31] = pitch * r_init[31] + arm_off[31]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14940};
  (data->simulationInfo->realParameter[1537] /* theta[31] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */)) + (data->simulationInfo->realParameter[33] /* arm_off[31] PARAM */);
  TRACE_POP
}

/*
equation index: 14941
type: SIMPLE_ASSIGN
arm_off[30] = 0.36442474781641604 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14941};
  (data->simulationInfo->realParameter[32] /* arm_off[30] PARAM */) = (0.36442474781641604) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14942
type: SIMPLE_ASSIGN
theta[30] = pitch * r_init[30] + arm_off[30]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14942(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14942};
  (data->simulationInfo->realParameter[1536] /* theta[30] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */)) + (data->simulationInfo->realParameter[32] /* arm_off[30] PARAM */);
  TRACE_POP
}

/*
equation index: 14943
type: SIMPLE_ASSIGN
arm_off[29] = 0.3518583772020568 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14943};
  (data->simulationInfo->realParameter[31] /* arm_off[29] PARAM */) = (0.3518583772020568) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14944
type: SIMPLE_ASSIGN
theta[29] = pitch * r_init[29] + arm_off[29]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14944(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14944};
  (data->simulationInfo->realParameter[1535] /* theta[29] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */)) + (data->simulationInfo->realParameter[31] /* arm_off[29] PARAM */);
  TRACE_POP
}

/*
equation index: 14945
type: SIMPLE_ASSIGN
arm_off[28] = 0.33929200658769765 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14945};
  (data->simulationInfo->realParameter[30] /* arm_off[28] PARAM */) = (0.33929200658769765) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14946
type: SIMPLE_ASSIGN
theta[28] = pitch * r_init[28] + arm_off[28]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14946(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14946};
  (data->simulationInfo->realParameter[1534] /* theta[28] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */)) + (data->simulationInfo->realParameter[30] /* arm_off[28] PARAM */);
  TRACE_POP
}

/*
equation index: 14947
type: SIMPLE_ASSIGN
arm_off[27] = 0.32672563597333854 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14947};
  (data->simulationInfo->realParameter[29] /* arm_off[27] PARAM */) = (0.32672563597333854) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14948
type: SIMPLE_ASSIGN
theta[27] = pitch * r_init[27] + arm_off[27]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14948};
  (data->simulationInfo->realParameter[1533] /* theta[27] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */)) + (data->simulationInfo->realParameter[29] /* arm_off[27] PARAM */);
  TRACE_POP
}

/*
equation index: 14949
type: SIMPLE_ASSIGN
arm_off[26] = 0.3141592653589793 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14949};
  (data->simulationInfo->realParameter[28] /* arm_off[26] PARAM */) = (0.3141592653589793) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14950
type: SIMPLE_ASSIGN
theta[26] = pitch * r_init[26] + arm_off[26]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14950(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14950};
  (data->simulationInfo->realParameter[1532] /* theta[26] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */)) + (data->simulationInfo->realParameter[28] /* arm_off[26] PARAM */);
  TRACE_POP
}

/*
equation index: 14951
type: SIMPLE_ASSIGN
arm_off[25] = 0.30159289474462014 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14951};
  (data->simulationInfo->realParameter[27] /* arm_off[25] PARAM */) = (0.30159289474462014) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14952
type: SIMPLE_ASSIGN
theta[25] = pitch * r_init[25] + arm_off[25]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14952(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14952};
  (data->simulationInfo->realParameter[1531] /* theta[25] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */)) + (data->simulationInfo->realParameter[27] /* arm_off[25] PARAM */);
  TRACE_POP
}

/*
equation index: 14953
type: SIMPLE_ASSIGN
arm_off[24] = 0.289026524130261 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14953};
  (data->simulationInfo->realParameter[26] /* arm_off[24] PARAM */) = (0.289026524130261) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14954
type: SIMPLE_ASSIGN
theta[24] = pitch * r_init[24] + arm_off[24]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14954(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14954};
  (data->simulationInfo->realParameter[1530] /* theta[24] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */)) + (data->simulationInfo->realParameter[26] /* arm_off[24] PARAM */);
  TRACE_POP
}

/*
equation index: 14955
type: SIMPLE_ASSIGN
arm_off[23] = 0.27646015351590175 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14955};
  (data->simulationInfo->realParameter[25] /* arm_off[23] PARAM */) = (0.27646015351590175) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14956
type: SIMPLE_ASSIGN
theta[23] = pitch * r_init[23] + arm_off[23]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14956};
  (data->simulationInfo->realParameter[1529] /* theta[23] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */)) + (data->simulationInfo->realParameter[25] /* arm_off[23] PARAM */);
  TRACE_POP
}

/*
equation index: 14957
type: SIMPLE_ASSIGN
arm_off[22] = 0.26389378290154264 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14957};
  (data->simulationInfo->realParameter[24] /* arm_off[22] PARAM */) = (0.26389378290154264) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14958
type: SIMPLE_ASSIGN
theta[22] = pitch * r_init[22] + arm_off[22]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14958(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14958};
  (data->simulationInfo->realParameter[1528] /* theta[22] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */)) + (data->simulationInfo->realParameter[24] /* arm_off[22] PARAM */);
  TRACE_POP
}

/*
equation index: 14959
type: SIMPLE_ASSIGN
arm_off[21] = 0.25132741228718347 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14959};
  (data->simulationInfo->realParameter[23] /* arm_off[21] PARAM */) = (0.25132741228718347) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14960
type: SIMPLE_ASSIGN
theta[21] = pitch * r_init[21] + arm_off[21]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14960(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14960};
  (data->simulationInfo->realParameter[1527] /* theta[21] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */)) + (data->simulationInfo->realParameter[23] /* arm_off[21] PARAM */);
  TRACE_POP
}

/*
equation index: 14961
type: SIMPLE_ASSIGN
arm_off[20] = 0.23876104167282428 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14961};
  (data->simulationInfo->realParameter[22] /* arm_off[20] PARAM */) = (0.23876104167282428) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14962
type: SIMPLE_ASSIGN
theta[20] = pitch * r_init[20] + arm_off[20]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14962(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14962};
  (data->simulationInfo->realParameter[1526] /* theta[20] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */)) + (data->simulationInfo->realParameter[22] /* arm_off[20] PARAM */);
  TRACE_POP
}

/*
equation index: 14963
type: SIMPLE_ASSIGN
arm_off[19] = 0.2261946710584651 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14963};
  (data->simulationInfo->realParameter[21] /* arm_off[19] PARAM */) = (0.2261946710584651) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14964
type: SIMPLE_ASSIGN
theta[19] = pitch * r_init[19] + arm_off[19]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14964};
  (data->simulationInfo->realParameter[1525] /* theta[19] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */)) + (data->simulationInfo->realParameter[21] /* arm_off[19] PARAM */);
  TRACE_POP
}

/*
equation index: 14965
type: SIMPLE_ASSIGN
arm_off[18] = 0.21362830044410594 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14965};
  (data->simulationInfo->realParameter[20] /* arm_off[18] PARAM */) = (0.21362830044410594) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14966
type: SIMPLE_ASSIGN
theta[18] = pitch * r_init[18] + arm_off[18]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14966(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14966};
  (data->simulationInfo->realParameter[1524] /* theta[18] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */)) + (data->simulationInfo->realParameter[20] /* arm_off[18] PARAM */);
  TRACE_POP
}

/*
equation index: 14967
type: SIMPLE_ASSIGN
arm_off[17] = 0.20106192982974677 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14967};
  (data->simulationInfo->realParameter[19] /* arm_off[17] PARAM */) = (0.20106192982974677) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14968
type: SIMPLE_ASSIGN
theta[17] = pitch * r_init[17] + arm_off[17]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14968(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14968};
  (data->simulationInfo->realParameter[1523] /* theta[17] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */)) + (data->simulationInfo->realParameter[19] /* arm_off[17] PARAM */);
  TRACE_POP
}

/*
equation index: 14969
type: SIMPLE_ASSIGN
arm_off[16] = 0.18849555921538758 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14969};
  (data->simulationInfo->realParameter[18] /* arm_off[16] PARAM */) = (0.18849555921538758) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14970
type: SIMPLE_ASSIGN
theta[16] = pitch * r_init[16] + arm_off[16]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14970(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14970};
  (data->simulationInfo->realParameter[1522] /* theta[16] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */)) + (data->simulationInfo->realParameter[18] /* arm_off[16] PARAM */);
  TRACE_POP
}

/*
equation index: 14971
type: SIMPLE_ASSIGN
arm_off[15] = 0.1759291886010284 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14971};
  (data->simulationInfo->realParameter[17] /* arm_off[15] PARAM */) = (0.1759291886010284) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14972
type: SIMPLE_ASSIGN
theta[15] = pitch * r_init[15] + arm_off[15]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14972};
  (data->simulationInfo->realParameter[1521] /* theta[15] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */)) + (data->simulationInfo->realParameter[17] /* arm_off[15] PARAM */);
  TRACE_POP
}

/*
equation index: 14973
type: SIMPLE_ASSIGN
arm_off[14] = 0.16336281798666927 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14973};
  (data->simulationInfo->realParameter[16] /* arm_off[14] PARAM */) = (0.16336281798666927) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14974
type: SIMPLE_ASSIGN
theta[14] = pitch * r_init[14] + arm_off[14]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14974(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14974};
  (data->simulationInfo->realParameter[1520] /* theta[14] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */)) + (data->simulationInfo->realParameter[16] /* arm_off[14] PARAM */);
  TRACE_POP
}

/*
equation index: 14975
type: SIMPLE_ASSIGN
arm_off[13] = 0.15079644737231007 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14975};
  (data->simulationInfo->realParameter[15] /* arm_off[13] PARAM */) = (0.15079644737231007) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14976
type: SIMPLE_ASSIGN
theta[13] = pitch * r_init[13] + arm_off[13]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14976(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14976};
  (data->simulationInfo->realParameter[1519] /* theta[13] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */)) + (data->simulationInfo->realParameter[15] /* arm_off[13] PARAM */);
  TRACE_POP
}

/*
equation index: 14977
type: SIMPLE_ASSIGN
arm_off[12] = 0.13823007675795088 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14977};
  (data->simulationInfo->realParameter[14] /* arm_off[12] PARAM */) = (0.13823007675795088) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14978
type: SIMPLE_ASSIGN
theta[12] = pitch * r_init[12] + arm_off[12]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14978(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14978};
  (data->simulationInfo->realParameter[1518] /* theta[12] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */)) + (data->simulationInfo->realParameter[14] /* arm_off[12] PARAM */);
  TRACE_POP
}

/*
equation index: 14979
type: SIMPLE_ASSIGN
arm_off[11] = 0.12566370614359174 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14979};
  (data->simulationInfo->realParameter[13] /* arm_off[11] PARAM */) = (0.12566370614359174) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14980
type: SIMPLE_ASSIGN
theta[11] = pitch * r_init[11] + arm_off[11]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14980};
  (data->simulationInfo->realParameter[1517] /* theta[11] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */)) + (data->simulationInfo->realParameter[13] /* arm_off[11] PARAM */);
  TRACE_POP
}

/*
equation index: 14981
type: SIMPLE_ASSIGN
arm_off[10] = 0.11309733552923255 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14981};
  (data->simulationInfo->realParameter[12] /* arm_off[10] PARAM */) = (0.11309733552923255) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14982
type: SIMPLE_ASSIGN
theta[10] = pitch * r_init[10] + arm_off[10]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14982(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14982};
  (data->simulationInfo->realParameter[1516] /* theta[10] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */)) + (data->simulationInfo->realParameter[12] /* arm_off[10] PARAM */);
  TRACE_POP
}

/*
equation index: 14983
type: SIMPLE_ASSIGN
arm_off[9] = 0.10053096491487339 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14983};
  (data->simulationInfo->realParameter[11] /* arm_off[9] PARAM */) = (0.10053096491487339) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14984
type: SIMPLE_ASSIGN
theta[9] = pitch * r_init[9] + arm_off[9]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14984(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14984};
  (data->simulationInfo->realParameter[1515] /* theta[9] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */)) + (data->simulationInfo->realParameter[11] /* arm_off[9] PARAM */);
  TRACE_POP
}

/*
equation index: 14985
type: SIMPLE_ASSIGN
arm_off[8] = 0.0879645943005142 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14985};
  (data->simulationInfo->realParameter[10] /* arm_off[8] PARAM */) = (0.0879645943005142) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14986
type: SIMPLE_ASSIGN
theta[8] = pitch * r_init[8] + arm_off[8]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14986(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14986};
  (data->simulationInfo->realParameter[1514] /* theta[8] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */)) + (data->simulationInfo->realParameter[10] /* arm_off[8] PARAM */);
  TRACE_POP
}

/*
equation index: 14987
type: SIMPLE_ASSIGN
arm_off[7] = 0.07539822368615504 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14987};
  (data->simulationInfo->realParameter[9] /* arm_off[7] PARAM */) = (0.07539822368615504) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14988
type: SIMPLE_ASSIGN
theta[7] = pitch * r_init[7] + arm_off[7]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14988};
  (data->simulationInfo->realParameter[1513] /* theta[7] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */)) + (data->simulationInfo->realParameter[9] /* arm_off[7] PARAM */);
  TRACE_POP
}

/*
equation index: 14989
type: SIMPLE_ASSIGN
arm_off[6] = 0.06283185307179587 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14989};
  (data->simulationInfo->realParameter[8] /* arm_off[6] PARAM */) = (0.06283185307179587) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14990
type: SIMPLE_ASSIGN
theta[6] = pitch * r_init[6] + arm_off[6]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14990(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14990};
  (data->simulationInfo->realParameter[1512] /* theta[6] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */)) + (data->simulationInfo->realParameter[8] /* arm_off[6] PARAM */);
  TRACE_POP
}

/*
equation index: 14991
type: SIMPLE_ASSIGN
arm_off[5] = 0.05026548245743669 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14991};
  (data->simulationInfo->realParameter[7] /* arm_off[5] PARAM */) = (0.05026548245743669) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14992
type: SIMPLE_ASSIGN
theta[5] = pitch * r_init[5] + arm_off[5]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14992(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14992};
  (data->simulationInfo->realParameter[1511] /* theta[5] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */)) + (data->simulationInfo->realParameter[7] /* arm_off[5] PARAM */);
  TRACE_POP
}

/*
equation index: 14993
type: SIMPLE_ASSIGN
arm_off[4] = 0.03769911184307752 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14993};
  (data->simulationInfo->realParameter[6] /* arm_off[4] PARAM */) = (0.03769911184307752) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14994
type: SIMPLE_ASSIGN
theta[4] = pitch * r_init[4] + arm_off[4]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14994(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14994};
  (data->simulationInfo->realParameter[1510] /* theta[4] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */)) + (data->simulationInfo->realParameter[6] /* arm_off[4] PARAM */);
  TRACE_POP
}

/*
equation index: 14995
type: SIMPLE_ASSIGN
arm_off[3] = 0.025132741228718346 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14995};
  (data->simulationInfo->realParameter[5] /* arm_off[3] PARAM */) = (0.025132741228718346) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14996
type: SIMPLE_ASSIGN
theta[3] = pitch * r_init[3] + arm_off[3]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14996};
  (data->simulationInfo->realParameter[1509] /* theta[3] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */)) + (data->simulationInfo->realParameter[5] /* arm_off[3] PARAM */);
  TRACE_POP
}

/*
equation index: 14997
type: SIMPLE_ASSIGN
arm_off[2] = 0.012566370614359173 * (*Real*)(M)
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14997};
  (data->simulationInfo->realParameter[4] /* arm_off[2] PARAM */) = (0.012566370614359173) * (((modelica_real)(data->simulationInfo->integerParameter[0] /* M PARAM */)));
  TRACE_POP
}

/*
equation index: 14998
type: SIMPLE_ASSIGN
theta[2] = pitch * r_init[2] + arm_off[2]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14998(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14998};
  (data->simulationInfo->realParameter[1508] /* theta[2] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */)) + (data->simulationInfo->realParameter[4] /* arm_off[2] PARAM */);
  TRACE_POP
}

/*
equation index: 14999
type: SIMPLE_ASSIGN
theta[1] = pitch * r_init[1] + arm_off[1]
*/
OMC_DISABLE_OPT
static void SpiralGalaxy_eqFunction_14999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14999};
  (data->simulationInfo->realParameter[1507] /* theta[1] PARAM */) = ((data->simulationInfo->realParameter[1005] /* pitch PARAM */)) * ((data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */)) + (data->simulationInfo->realParameter[3] /* arm_off[1] PARAM */);
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_updateBoundParameters_3(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_14501(data, threadData);
  SpiralGalaxy_eqFunction_14502(data, threadData);
  SpiralGalaxy_eqFunction_14503(data, threadData);
  SpiralGalaxy_eqFunction_14504(data, threadData);
  SpiralGalaxy_eqFunction_14505(data, threadData);
  SpiralGalaxy_eqFunction_14506(data, threadData);
  SpiralGalaxy_eqFunction_14507(data, threadData);
  SpiralGalaxy_eqFunction_14508(data, threadData);
  SpiralGalaxy_eqFunction_14509(data, threadData);
  SpiralGalaxy_eqFunction_14510(data, threadData);
  SpiralGalaxy_eqFunction_14511(data, threadData);
  SpiralGalaxy_eqFunction_14512(data, threadData);
  SpiralGalaxy_eqFunction_14513(data, threadData);
  SpiralGalaxy_eqFunction_14514(data, threadData);
  SpiralGalaxy_eqFunction_14515(data, threadData);
  SpiralGalaxy_eqFunction_14516(data, threadData);
  SpiralGalaxy_eqFunction_14517(data, threadData);
  SpiralGalaxy_eqFunction_14518(data, threadData);
  SpiralGalaxy_eqFunction_14519(data, threadData);
  SpiralGalaxy_eqFunction_14520(data, threadData);
  SpiralGalaxy_eqFunction_14521(data, threadData);
  SpiralGalaxy_eqFunction_14522(data, threadData);
  SpiralGalaxy_eqFunction_14523(data, threadData);
  SpiralGalaxy_eqFunction_14524(data, threadData);
  SpiralGalaxy_eqFunction_14525(data, threadData);
  SpiralGalaxy_eqFunction_14526(data, threadData);
  SpiralGalaxy_eqFunction_14527(data, threadData);
  SpiralGalaxy_eqFunction_14528(data, threadData);
  SpiralGalaxy_eqFunction_14529(data, threadData);
  SpiralGalaxy_eqFunction_14530(data, threadData);
  SpiralGalaxy_eqFunction_14531(data, threadData);
  SpiralGalaxy_eqFunction_14532(data, threadData);
  SpiralGalaxy_eqFunction_14533(data, threadData);
  SpiralGalaxy_eqFunction_14534(data, threadData);
  SpiralGalaxy_eqFunction_14535(data, threadData);
  SpiralGalaxy_eqFunction_14536(data, threadData);
  SpiralGalaxy_eqFunction_14537(data, threadData);
  SpiralGalaxy_eqFunction_14538(data, threadData);
  SpiralGalaxy_eqFunction_14539(data, threadData);
  SpiralGalaxy_eqFunction_14540(data, threadData);
  SpiralGalaxy_eqFunction_14541(data, threadData);
  SpiralGalaxy_eqFunction_14542(data, threadData);
  SpiralGalaxy_eqFunction_14543(data, threadData);
  SpiralGalaxy_eqFunction_14544(data, threadData);
  SpiralGalaxy_eqFunction_14545(data, threadData);
  SpiralGalaxy_eqFunction_14546(data, threadData);
  SpiralGalaxy_eqFunction_14547(data, threadData);
  SpiralGalaxy_eqFunction_14548(data, threadData);
  SpiralGalaxy_eqFunction_14549(data, threadData);
  SpiralGalaxy_eqFunction_14550(data, threadData);
  SpiralGalaxy_eqFunction_14551(data, threadData);
  SpiralGalaxy_eqFunction_14552(data, threadData);
  SpiralGalaxy_eqFunction_14553(data, threadData);
  SpiralGalaxy_eqFunction_14554(data, threadData);
  SpiralGalaxy_eqFunction_14555(data, threadData);
  SpiralGalaxy_eqFunction_14556(data, threadData);
  SpiralGalaxy_eqFunction_14557(data, threadData);
  SpiralGalaxy_eqFunction_14558(data, threadData);
  SpiralGalaxy_eqFunction_14559(data, threadData);
  SpiralGalaxy_eqFunction_14560(data, threadData);
  SpiralGalaxy_eqFunction_14561(data, threadData);
  SpiralGalaxy_eqFunction_14562(data, threadData);
  SpiralGalaxy_eqFunction_14563(data, threadData);
  SpiralGalaxy_eqFunction_14564(data, threadData);
  SpiralGalaxy_eqFunction_14565(data, threadData);
  SpiralGalaxy_eqFunction_14566(data, threadData);
  SpiralGalaxy_eqFunction_14567(data, threadData);
  SpiralGalaxy_eqFunction_14568(data, threadData);
  SpiralGalaxy_eqFunction_14569(data, threadData);
  SpiralGalaxy_eqFunction_14570(data, threadData);
  SpiralGalaxy_eqFunction_14571(data, threadData);
  SpiralGalaxy_eqFunction_14572(data, threadData);
  SpiralGalaxy_eqFunction_14573(data, threadData);
  SpiralGalaxy_eqFunction_14574(data, threadData);
  SpiralGalaxy_eqFunction_14575(data, threadData);
  SpiralGalaxy_eqFunction_14576(data, threadData);
  SpiralGalaxy_eqFunction_14577(data, threadData);
  SpiralGalaxy_eqFunction_14578(data, threadData);
  SpiralGalaxy_eqFunction_14579(data, threadData);
  SpiralGalaxy_eqFunction_14580(data, threadData);
  SpiralGalaxy_eqFunction_14581(data, threadData);
  SpiralGalaxy_eqFunction_14582(data, threadData);
  SpiralGalaxy_eqFunction_14583(data, threadData);
  SpiralGalaxy_eqFunction_14584(data, threadData);
  SpiralGalaxy_eqFunction_14585(data, threadData);
  SpiralGalaxy_eqFunction_14586(data, threadData);
  SpiralGalaxy_eqFunction_14587(data, threadData);
  SpiralGalaxy_eqFunction_14588(data, threadData);
  SpiralGalaxy_eqFunction_14589(data, threadData);
  SpiralGalaxy_eqFunction_14590(data, threadData);
  SpiralGalaxy_eqFunction_14591(data, threadData);
  SpiralGalaxy_eqFunction_14592(data, threadData);
  SpiralGalaxy_eqFunction_14593(data, threadData);
  SpiralGalaxy_eqFunction_14594(data, threadData);
  SpiralGalaxy_eqFunction_14595(data, threadData);
  SpiralGalaxy_eqFunction_14596(data, threadData);
  SpiralGalaxy_eqFunction_14597(data, threadData);
  SpiralGalaxy_eqFunction_14598(data, threadData);
  SpiralGalaxy_eqFunction_14599(data, threadData);
  SpiralGalaxy_eqFunction_14600(data, threadData);
  SpiralGalaxy_eqFunction_14601(data, threadData);
  SpiralGalaxy_eqFunction_14602(data, threadData);
  SpiralGalaxy_eqFunction_14603(data, threadData);
  SpiralGalaxy_eqFunction_14604(data, threadData);
  SpiralGalaxy_eqFunction_14605(data, threadData);
  SpiralGalaxy_eqFunction_14606(data, threadData);
  SpiralGalaxy_eqFunction_14607(data, threadData);
  SpiralGalaxy_eqFunction_14608(data, threadData);
  SpiralGalaxy_eqFunction_14609(data, threadData);
  SpiralGalaxy_eqFunction_14610(data, threadData);
  SpiralGalaxy_eqFunction_14611(data, threadData);
  SpiralGalaxy_eqFunction_14612(data, threadData);
  SpiralGalaxy_eqFunction_14613(data, threadData);
  SpiralGalaxy_eqFunction_14614(data, threadData);
  SpiralGalaxy_eqFunction_14615(data, threadData);
  SpiralGalaxy_eqFunction_14616(data, threadData);
  SpiralGalaxy_eqFunction_14617(data, threadData);
  SpiralGalaxy_eqFunction_14618(data, threadData);
  SpiralGalaxy_eqFunction_14619(data, threadData);
  SpiralGalaxy_eqFunction_14620(data, threadData);
  SpiralGalaxy_eqFunction_14621(data, threadData);
  SpiralGalaxy_eqFunction_14622(data, threadData);
  SpiralGalaxy_eqFunction_14623(data, threadData);
  SpiralGalaxy_eqFunction_14624(data, threadData);
  SpiralGalaxy_eqFunction_14625(data, threadData);
  SpiralGalaxy_eqFunction_14626(data, threadData);
  SpiralGalaxy_eqFunction_14627(data, threadData);
  SpiralGalaxy_eqFunction_14628(data, threadData);
  SpiralGalaxy_eqFunction_14629(data, threadData);
  SpiralGalaxy_eqFunction_14630(data, threadData);
  SpiralGalaxy_eqFunction_14631(data, threadData);
  SpiralGalaxy_eqFunction_14632(data, threadData);
  SpiralGalaxy_eqFunction_14633(data, threadData);
  SpiralGalaxy_eqFunction_14634(data, threadData);
  SpiralGalaxy_eqFunction_14635(data, threadData);
  SpiralGalaxy_eqFunction_14636(data, threadData);
  SpiralGalaxy_eqFunction_14637(data, threadData);
  SpiralGalaxy_eqFunction_14638(data, threadData);
  SpiralGalaxy_eqFunction_14639(data, threadData);
  SpiralGalaxy_eqFunction_14640(data, threadData);
  SpiralGalaxy_eqFunction_14641(data, threadData);
  SpiralGalaxy_eqFunction_14642(data, threadData);
  SpiralGalaxy_eqFunction_14643(data, threadData);
  SpiralGalaxy_eqFunction_14644(data, threadData);
  SpiralGalaxy_eqFunction_14645(data, threadData);
  SpiralGalaxy_eqFunction_14646(data, threadData);
  SpiralGalaxy_eqFunction_14647(data, threadData);
  SpiralGalaxy_eqFunction_14648(data, threadData);
  SpiralGalaxy_eqFunction_14649(data, threadData);
  SpiralGalaxy_eqFunction_14650(data, threadData);
  SpiralGalaxy_eqFunction_14651(data, threadData);
  SpiralGalaxy_eqFunction_14652(data, threadData);
  SpiralGalaxy_eqFunction_14653(data, threadData);
  SpiralGalaxy_eqFunction_14654(data, threadData);
  SpiralGalaxy_eqFunction_14655(data, threadData);
  SpiralGalaxy_eqFunction_14656(data, threadData);
  SpiralGalaxy_eqFunction_14657(data, threadData);
  SpiralGalaxy_eqFunction_14658(data, threadData);
  SpiralGalaxy_eqFunction_14659(data, threadData);
  SpiralGalaxy_eqFunction_14660(data, threadData);
  SpiralGalaxy_eqFunction_14661(data, threadData);
  SpiralGalaxy_eqFunction_14662(data, threadData);
  SpiralGalaxy_eqFunction_14663(data, threadData);
  SpiralGalaxy_eqFunction_14664(data, threadData);
  SpiralGalaxy_eqFunction_14665(data, threadData);
  SpiralGalaxy_eqFunction_14666(data, threadData);
  SpiralGalaxy_eqFunction_14667(data, threadData);
  SpiralGalaxy_eqFunction_14668(data, threadData);
  SpiralGalaxy_eqFunction_14669(data, threadData);
  SpiralGalaxy_eqFunction_14670(data, threadData);
  SpiralGalaxy_eqFunction_14671(data, threadData);
  SpiralGalaxy_eqFunction_14672(data, threadData);
  SpiralGalaxy_eqFunction_14673(data, threadData);
  SpiralGalaxy_eqFunction_14674(data, threadData);
  SpiralGalaxy_eqFunction_14675(data, threadData);
  SpiralGalaxy_eqFunction_14676(data, threadData);
  SpiralGalaxy_eqFunction_14677(data, threadData);
  SpiralGalaxy_eqFunction_14678(data, threadData);
  SpiralGalaxy_eqFunction_14679(data, threadData);
  SpiralGalaxy_eqFunction_14680(data, threadData);
  SpiralGalaxy_eqFunction_14681(data, threadData);
  SpiralGalaxy_eqFunction_14682(data, threadData);
  SpiralGalaxy_eqFunction_14683(data, threadData);
  SpiralGalaxy_eqFunction_14684(data, threadData);
  SpiralGalaxy_eqFunction_14685(data, threadData);
  SpiralGalaxy_eqFunction_14686(data, threadData);
  SpiralGalaxy_eqFunction_14687(data, threadData);
  SpiralGalaxy_eqFunction_14688(data, threadData);
  SpiralGalaxy_eqFunction_14689(data, threadData);
  SpiralGalaxy_eqFunction_14690(data, threadData);
  SpiralGalaxy_eqFunction_14691(data, threadData);
  SpiralGalaxy_eqFunction_14692(data, threadData);
  SpiralGalaxy_eqFunction_14693(data, threadData);
  SpiralGalaxy_eqFunction_14694(data, threadData);
  SpiralGalaxy_eqFunction_14695(data, threadData);
  SpiralGalaxy_eqFunction_14696(data, threadData);
  SpiralGalaxy_eqFunction_14697(data, threadData);
  SpiralGalaxy_eqFunction_14698(data, threadData);
  SpiralGalaxy_eqFunction_14699(data, threadData);
  SpiralGalaxy_eqFunction_14700(data, threadData);
  SpiralGalaxy_eqFunction_14701(data, threadData);
  SpiralGalaxy_eqFunction_14702(data, threadData);
  SpiralGalaxy_eqFunction_14703(data, threadData);
  SpiralGalaxy_eqFunction_14704(data, threadData);
  SpiralGalaxy_eqFunction_14705(data, threadData);
  SpiralGalaxy_eqFunction_14706(data, threadData);
  SpiralGalaxy_eqFunction_14707(data, threadData);
  SpiralGalaxy_eqFunction_14708(data, threadData);
  SpiralGalaxy_eqFunction_14709(data, threadData);
  SpiralGalaxy_eqFunction_14710(data, threadData);
  SpiralGalaxy_eqFunction_14711(data, threadData);
  SpiralGalaxy_eqFunction_14712(data, threadData);
  SpiralGalaxy_eqFunction_14713(data, threadData);
  SpiralGalaxy_eqFunction_14714(data, threadData);
  SpiralGalaxy_eqFunction_14715(data, threadData);
  SpiralGalaxy_eqFunction_14716(data, threadData);
  SpiralGalaxy_eqFunction_14717(data, threadData);
  SpiralGalaxy_eqFunction_14718(data, threadData);
  SpiralGalaxy_eqFunction_14719(data, threadData);
  SpiralGalaxy_eqFunction_14720(data, threadData);
  SpiralGalaxy_eqFunction_14721(data, threadData);
  SpiralGalaxy_eqFunction_14722(data, threadData);
  SpiralGalaxy_eqFunction_14723(data, threadData);
  SpiralGalaxy_eqFunction_14724(data, threadData);
  SpiralGalaxy_eqFunction_14725(data, threadData);
  SpiralGalaxy_eqFunction_14726(data, threadData);
  SpiralGalaxy_eqFunction_14727(data, threadData);
  SpiralGalaxy_eqFunction_14728(data, threadData);
  SpiralGalaxy_eqFunction_14729(data, threadData);
  SpiralGalaxy_eqFunction_14730(data, threadData);
  SpiralGalaxy_eqFunction_14731(data, threadData);
  SpiralGalaxy_eqFunction_14732(data, threadData);
  SpiralGalaxy_eqFunction_14733(data, threadData);
  SpiralGalaxy_eqFunction_14734(data, threadData);
  SpiralGalaxy_eqFunction_14735(data, threadData);
  SpiralGalaxy_eqFunction_14736(data, threadData);
  SpiralGalaxy_eqFunction_14737(data, threadData);
  SpiralGalaxy_eqFunction_14738(data, threadData);
  SpiralGalaxy_eqFunction_14739(data, threadData);
  SpiralGalaxy_eqFunction_14740(data, threadData);
  SpiralGalaxy_eqFunction_14741(data, threadData);
  SpiralGalaxy_eqFunction_14742(data, threadData);
  SpiralGalaxy_eqFunction_14743(data, threadData);
  SpiralGalaxy_eqFunction_14744(data, threadData);
  SpiralGalaxy_eqFunction_14745(data, threadData);
  SpiralGalaxy_eqFunction_14746(data, threadData);
  SpiralGalaxy_eqFunction_14747(data, threadData);
  SpiralGalaxy_eqFunction_14748(data, threadData);
  SpiralGalaxy_eqFunction_14749(data, threadData);
  SpiralGalaxy_eqFunction_14750(data, threadData);
  SpiralGalaxy_eqFunction_14751(data, threadData);
  SpiralGalaxy_eqFunction_14752(data, threadData);
  SpiralGalaxy_eqFunction_14753(data, threadData);
  SpiralGalaxy_eqFunction_14754(data, threadData);
  SpiralGalaxy_eqFunction_14755(data, threadData);
  SpiralGalaxy_eqFunction_14756(data, threadData);
  SpiralGalaxy_eqFunction_14757(data, threadData);
  SpiralGalaxy_eqFunction_14758(data, threadData);
  SpiralGalaxy_eqFunction_14759(data, threadData);
  SpiralGalaxy_eqFunction_14760(data, threadData);
  SpiralGalaxy_eqFunction_14761(data, threadData);
  SpiralGalaxy_eqFunction_14762(data, threadData);
  SpiralGalaxy_eqFunction_14763(data, threadData);
  SpiralGalaxy_eqFunction_14764(data, threadData);
  SpiralGalaxy_eqFunction_14765(data, threadData);
  SpiralGalaxy_eqFunction_14766(data, threadData);
  SpiralGalaxy_eqFunction_14767(data, threadData);
  SpiralGalaxy_eqFunction_14768(data, threadData);
  SpiralGalaxy_eqFunction_14769(data, threadData);
  SpiralGalaxy_eqFunction_14770(data, threadData);
  SpiralGalaxy_eqFunction_14771(data, threadData);
  SpiralGalaxy_eqFunction_14772(data, threadData);
  SpiralGalaxy_eqFunction_14773(data, threadData);
  SpiralGalaxy_eqFunction_14774(data, threadData);
  SpiralGalaxy_eqFunction_14775(data, threadData);
  SpiralGalaxy_eqFunction_14776(data, threadData);
  SpiralGalaxy_eqFunction_14777(data, threadData);
  SpiralGalaxy_eqFunction_14778(data, threadData);
  SpiralGalaxy_eqFunction_14779(data, threadData);
  SpiralGalaxy_eqFunction_14780(data, threadData);
  SpiralGalaxy_eqFunction_14781(data, threadData);
  SpiralGalaxy_eqFunction_14782(data, threadData);
  SpiralGalaxy_eqFunction_14783(data, threadData);
  SpiralGalaxy_eqFunction_14784(data, threadData);
  SpiralGalaxy_eqFunction_14785(data, threadData);
  SpiralGalaxy_eqFunction_14786(data, threadData);
  SpiralGalaxy_eqFunction_14787(data, threadData);
  SpiralGalaxy_eqFunction_14788(data, threadData);
  SpiralGalaxy_eqFunction_14789(data, threadData);
  SpiralGalaxy_eqFunction_14790(data, threadData);
  SpiralGalaxy_eqFunction_14791(data, threadData);
  SpiralGalaxy_eqFunction_14792(data, threadData);
  SpiralGalaxy_eqFunction_14793(data, threadData);
  SpiralGalaxy_eqFunction_14794(data, threadData);
  SpiralGalaxy_eqFunction_14795(data, threadData);
  SpiralGalaxy_eqFunction_14796(data, threadData);
  SpiralGalaxy_eqFunction_14797(data, threadData);
  SpiralGalaxy_eqFunction_14798(data, threadData);
  SpiralGalaxy_eqFunction_14799(data, threadData);
  SpiralGalaxy_eqFunction_14800(data, threadData);
  SpiralGalaxy_eqFunction_14801(data, threadData);
  SpiralGalaxy_eqFunction_14802(data, threadData);
  SpiralGalaxy_eqFunction_14803(data, threadData);
  SpiralGalaxy_eqFunction_14804(data, threadData);
  SpiralGalaxy_eqFunction_14805(data, threadData);
  SpiralGalaxy_eqFunction_14806(data, threadData);
  SpiralGalaxy_eqFunction_14807(data, threadData);
  SpiralGalaxy_eqFunction_14808(data, threadData);
  SpiralGalaxy_eqFunction_14809(data, threadData);
  SpiralGalaxy_eqFunction_14810(data, threadData);
  SpiralGalaxy_eqFunction_14811(data, threadData);
  SpiralGalaxy_eqFunction_14812(data, threadData);
  SpiralGalaxy_eqFunction_14813(data, threadData);
  SpiralGalaxy_eqFunction_14814(data, threadData);
  SpiralGalaxy_eqFunction_14815(data, threadData);
  SpiralGalaxy_eqFunction_14816(data, threadData);
  SpiralGalaxy_eqFunction_14817(data, threadData);
  SpiralGalaxy_eqFunction_14818(data, threadData);
  SpiralGalaxy_eqFunction_14819(data, threadData);
  SpiralGalaxy_eqFunction_14820(data, threadData);
  SpiralGalaxy_eqFunction_14821(data, threadData);
  SpiralGalaxy_eqFunction_14822(data, threadData);
  SpiralGalaxy_eqFunction_14823(data, threadData);
  SpiralGalaxy_eqFunction_14824(data, threadData);
  SpiralGalaxy_eqFunction_14825(data, threadData);
  SpiralGalaxy_eqFunction_14826(data, threadData);
  SpiralGalaxy_eqFunction_14827(data, threadData);
  SpiralGalaxy_eqFunction_14828(data, threadData);
  SpiralGalaxy_eqFunction_14829(data, threadData);
  SpiralGalaxy_eqFunction_14830(data, threadData);
  SpiralGalaxy_eqFunction_14831(data, threadData);
  SpiralGalaxy_eqFunction_14832(data, threadData);
  SpiralGalaxy_eqFunction_14833(data, threadData);
  SpiralGalaxy_eqFunction_14834(data, threadData);
  SpiralGalaxy_eqFunction_14835(data, threadData);
  SpiralGalaxy_eqFunction_14836(data, threadData);
  SpiralGalaxy_eqFunction_14837(data, threadData);
  SpiralGalaxy_eqFunction_14838(data, threadData);
  SpiralGalaxy_eqFunction_14839(data, threadData);
  SpiralGalaxy_eqFunction_14840(data, threadData);
  SpiralGalaxy_eqFunction_14841(data, threadData);
  SpiralGalaxy_eqFunction_14842(data, threadData);
  SpiralGalaxy_eqFunction_14843(data, threadData);
  SpiralGalaxy_eqFunction_14844(data, threadData);
  SpiralGalaxy_eqFunction_14845(data, threadData);
  SpiralGalaxy_eqFunction_14846(data, threadData);
  SpiralGalaxy_eqFunction_14847(data, threadData);
  SpiralGalaxy_eqFunction_14848(data, threadData);
  SpiralGalaxy_eqFunction_14849(data, threadData);
  SpiralGalaxy_eqFunction_14850(data, threadData);
  SpiralGalaxy_eqFunction_14851(data, threadData);
  SpiralGalaxy_eqFunction_14852(data, threadData);
  SpiralGalaxy_eqFunction_14853(data, threadData);
  SpiralGalaxy_eqFunction_14854(data, threadData);
  SpiralGalaxy_eqFunction_14855(data, threadData);
  SpiralGalaxy_eqFunction_14856(data, threadData);
  SpiralGalaxy_eqFunction_14857(data, threadData);
  SpiralGalaxy_eqFunction_14858(data, threadData);
  SpiralGalaxy_eqFunction_14859(data, threadData);
  SpiralGalaxy_eqFunction_14860(data, threadData);
  SpiralGalaxy_eqFunction_14861(data, threadData);
  SpiralGalaxy_eqFunction_14862(data, threadData);
  SpiralGalaxy_eqFunction_14863(data, threadData);
  SpiralGalaxy_eqFunction_14864(data, threadData);
  SpiralGalaxy_eqFunction_14865(data, threadData);
  SpiralGalaxy_eqFunction_14866(data, threadData);
  SpiralGalaxy_eqFunction_14867(data, threadData);
  SpiralGalaxy_eqFunction_14868(data, threadData);
  SpiralGalaxy_eqFunction_14869(data, threadData);
  SpiralGalaxy_eqFunction_14870(data, threadData);
  SpiralGalaxy_eqFunction_14871(data, threadData);
  SpiralGalaxy_eqFunction_14872(data, threadData);
  SpiralGalaxy_eqFunction_14873(data, threadData);
  SpiralGalaxy_eqFunction_14874(data, threadData);
  SpiralGalaxy_eqFunction_14875(data, threadData);
  SpiralGalaxy_eqFunction_14876(data, threadData);
  SpiralGalaxy_eqFunction_14877(data, threadData);
  SpiralGalaxy_eqFunction_14878(data, threadData);
  SpiralGalaxy_eqFunction_14879(data, threadData);
  SpiralGalaxy_eqFunction_14880(data, threadData);
  SpiralGalaxy_eqFunction_14881(data, threadData);
  SpiralGalaxy_eqFunction_14882(data, threadData);
  SpiralGalaxy_eqFunction_14883(data, threadData);
  SpiralGalaxy_eqFunction_14884(data, threadData);
  SpiralGalaxy_eqFunction_14885(data, threadData);
  SpiralGalaxy_eqFunction_14886(data, threadData);
  SpiralGalaxy_eqFunction_14887(data, threadData);
  SpiralGalaxy_eqFunction_14888(data, threadData);
  SpiralGalaxy_eqFunction_14889(data, threadData);
  SpiralGalaxy_eqFunction_14890(data, threadData);
  SpiralGalaxy_eqFunction_14891(data, threadData);
  SpiralGalaxy_eqFunction_14892(data, threadData);
  SpiralGalaxy_eqFunction_14893(data, threadData);
  SpiralGalaxy_eqFunction_14894(data, threadData);
  SpiralGalaxy_eqFunction_14895(data, threadData);
  SpiralGalaxy_eqFunction_14896(data, threadData);
  SpiralGalaxy_eqFunction_14897(data, threadData);
  SpiralGalaxy_eqFunction_14898(data, threadData);
  SpiralGalaxy_eqFunction_14899(data, threadData);
  SpiralGalaxy_eqFunction_14900(data, threadData);
  SpiralGalaxy_eqFunction_14901(data, threadData);
  SpiralGalaxy_eqFunction_14902(data, threadData);
  SpiralGalaxy_eqFunction_14903(data, threadData);
  SpiralGalaxy_eqFunction_14904(data, threadData);
  SpiralGalaxy_eqFunction_14905(data, threadData);
  SpiralGalaxy_eqFunction_14906(data, threadData);
  SpiralGalaxy_eqFunction_14907(data, threadData);
  SpiralGalaxy_eqFunction_14908(data, threadData);
  SpiralGalaxy_eqFunction_14909(data, threadData);
  SpiralGalaxy_eqFunction_14910(data, threadData);
  SpiralGalaxy_eqFunction_14911(data, threadData);
  SpiralGalaxy_eqFunction_14912(data, threadData);
  SpiralGalaxy_eqFunction_14913(data, threadData);
  SpiralGalaxy_eqFunction_14914(data, threadData);
  SpiralGalaxy_eqFunction_14915(data, threadData);
  SpiralGalaxy_eqFunction_14916(data, threadData);
  SpiralGalaxy_eqFunction_14917(data, threadData);
  SpiralGalaxy_eqFunction_14918(data, threadData);
  SpiralGalaxy_eqFunction_14919(data, threadData);
  SpiralGalaxy_eqFunction_14920(data, threadData);
  SpiralGalaxy_eqFunction_14921(data, threadData);
  SpiralGalaxy_eqFunction_14922(data, threadData);
  SpiralGalaxy_eqFunction_14923(data, threadData);
  SpiralGalaxy_eqFunction_14924(data, threadData);
  SpiralGalaxy_eqFunction_14925(data, threadData);
  SpiralGalaxy_eqFunction_14926(data, threadData);
  SpiralGalaxy_eqFunction_14927(data, threadData);
  SpiralGalaxy_eqFunction_14928(data, threadData);
  SpiralGalaxy_eqFunction_14929(data, threadData);
  SpiralGalaxy_eqFunction_14930(data, threadData);
  SpiralGalaxy_eqFunction_14931(data, threadData);
  SpiralGalaxy_eqFunction_14932(data, threadData);
  SpiralGalaxy_eqFunction_14933(data, threadData);
  SpiralGalaxy_eqFunction_14934(data, threadData);
  SpiralGalaxy_eqFunction_14935(data, threadData);
  SpiralGalaxy_eqFunction_14936(data, threadData);
  SpiralGalaxy_eqFunction_14937(data, threadData);
  SpiralGalaxy_eqFunction_14938(data, threadData);
  SpiralGalaxy_eqFunction_14939(data, threadData);
  SpiralGalaxy_eqFunction_14940(data, threadData);
  SpiralGalaxy_eqFunction_14941(data, threadData);
  SpiralGalaxy_eqFunction_14942(data, threadData);
  SpiralGalaxy_eqFunction_14943(data, threadData);
  SpiralGalaxy_eqFunction_14944(data, threadData);
  SpiralGalaxy_eqFunction_14945(data, threadData);
  SpiralGalaxy_eqFunction_14946(data, threadData);
  SpiralGalaxy_eqFunction_14947(data, threadData);
  SpiralGalaxy_eqFunction_14948(data, threadData);
  SpiralGalaxy_eqFunction_14949(data, threadData);
  SpiralGalaxy_eqFunction_14950(data, threadData);
  SpiralGalaxy_eqFunction_14951(data, threadData);
  SpiralGalaxy_eqFunction_14952(data, threadData);
  SpiralGalaxy_eqFunction_14953(data, threadData);
  SpiralGalaxy_eqFunction_14954(data, threadData);
  SpiralGalaxy_eqFunction_14955(data, threadData);
  SpiralGalaxy_eqFunction_14956(data, threadData);
  SpiralGalaxy_eqFunction_14957(data, threadData);
  SpiralGalaxy_eqFunction_14958(data, threadData);
  SpiralGalaxy_eqFunction_14959(data, threadData);
  SpiralGalaxy_eqFunction_14960(data, threadData);
  SpiralGalaxy_eqFunction_14961(data, threadData);
  SpiralGalaxy_eqFunction_14962(data, threadData);
  SpiralGalaxy_eqFunction_14963(data, threadData);
  SpiralGalaxy_eqFunction_14964(data, threadData);
  SpiralGalaxy_eqFunction_14965(data, threadData);
  SpiralGalaxy_eqFunction_14966(data, threadData);
  SpiralGalaxy_eqFunction_14967(data, threadData);
  SpiralGalaxy_eqFunction_14968(data, threadData);
  SpiralGalaxy_eqFunction_14969(data, threadData);
  SpiralGalaxy_eqFunction_14970(data, threadData);
  SpiralGalaxy_eqFunction_14971(data, threadData);
  SpiralGalaxy_eqFunction_14972(data, threadData);
  SpiralGalaxy_eqFunction_14973(data, threadData);
  SpiralGalaxy_eqFunction_14974(data, threadData);
  SpiralGalaxy_eqFunction_14975(data, threadData);
  SpiralGalaxy_eqFunction_14976(data, threadData);
  SpiralGalaxy_eqFunction_14977(data, threadData);
  SpiralGalaxy_eqFunction_14978(data, threadData);
  SpiralGalaxy_eqFunction_14979(data, threadData);
  SpiralGalaxy_eqFunction_14980(data, threadData);
  SpiralGalaxy_eqFunction_14981(data, threadData);
  SpiralGalaxy_eqFunction_14982(data, threadData);
  SpiralGalaxy_eqFunction_14983(data, threadData);
  SpiralGalaxy_eqFunction_14984(data, threadData);
  SpiralGalaxy_eqFunction_14985(data, threadData);
  SpiralGalaxy_eqFunction_14986(data, threadData);
  SpiralGalaxy_eqFunction_14987(data, threadData);
  SpiralGalaxy_eqFunction_14988(data, threadData);
  SpiralGalaxy_eqFunction_14989(data, threadData);
  SpiralGalaxy_eqFunction_14990(data, threadData);
  SpiralGalaxy_eqFunction_14991(data, threadData);
  SpiralGalaxy_eqFunction_14992(data, threadData);
  SpiralGalaxy_eqFunction_14993(data, threadData);
  SpiralGalaxy_eqFunction_14994(data, threadData);
  SpiralGalaxy_eqFunction_14995(data, threadData);
  SpiralGalaxy_eqFunction_14996(data, threadData);
  SpiralGalaxy_eqFunction_14997(data, threadData);
  SpiralGalaxy_eqFunction_14998(data, threadData);
  SpiralGalaxy_eqFunction_14999(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif