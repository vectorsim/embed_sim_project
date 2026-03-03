#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 6501
type: SIMPLE_ASSIGN
x[407] = r_init[407] * cos(theta[407] + 0.006279999999999999)
*/
void SpiralGalaxy_eqFunction_6501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1906]] /* x[407] STATE(1,vx[407]) */) = ((data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */)) * (cos((data->simulationInfo->realParameter[1913] /* theta[407] PARAM */) + 0.006279999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12066(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12067(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12070(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12069(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12068(DATA *data, threadData_t *threadData);


/*
equation index: 6507
type: SIMPLE_ASSIGN
vx[407] = (-sin(theta[407])) * r_init[407] * omega_c[407]
*/
void SpiralGalaxy_eqFunction_6507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6507};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[406]] /* vx[407] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1913] /* theta[407] PARAM */)))) * (((data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */)) * ((data->simulationInfo->realParameter[911] /* omega_c[407] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12063(DATA *data, threadData_t *threadData);


/*
equation index: 6509
type: SIMPLE_ASSIGN
vy[407] = cos(theta[407]) * r_init[407] * omega_c[407]
*/
void SpiralGalaxy_eqFunction_6509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[906]] /* vy[407] STATE(1) */) = (cos((data->simulationInfo->realParameter[1913] /* theta[407] PARAM */))) * (((data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */)) * ((data->simulationInfo->realParameter[911] /* omega_c[407] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12062(DATA *data, threadData_t *threadData);


/*
equation index: 6511
type: SIMPLE_ASSIGN
vz[407] = 0.0
*/
void SpiralGalaxy_eqFunction_6511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6511};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1406]] /* vz[407] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12061(DATA *data, threadData_t *threadData);


/*
equation index: 6513
type: SIMPLE_ASSIGN
z[408] = 0.02528
*/
void SpiralGalaxy_eqFunction_6513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6513};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2907]] /* z[408] STATE(1,vz[408]) */) = 0.02528;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12074(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12075(DATA *data, threadData_t *threadData);


/*
equation index: 6516
type: SIMPLE_ASSIGN
y[408] = r_init[408] * sin(theta[408] + 0.006319999999999999)
*/
void SpiralGalaxy_eqFunction_6516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6516};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2407]] /* y[408] STATE(1,vy[408]) */) = ((data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */)) * (sin((data->simulationInfo->realParameter[1914] /* theta[408] PARAM */) + 0.006319999999999999));
  TRACE_POP
}

/*
equation index: 6517
type: SIMPLE_ASSIGN
x[408] = r_init[408] * cos(theta[408] + 0.006319999999999999)
*/
void SpiralGalaxy_eqFunction_6517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1907]] /* x[408] STATE(1,vx[408]) */) = ((data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */)) * (cos((data->simulationInfo->realParameter[1914] /* theta[408] PARAM */) + 0.006319999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12076(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12077(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12080(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12079(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12078(DATA *data, threadData_t *threadData);


/*
equation index: 6523
type: SIMPLE_ASSIGN
vx[408] = (-sin(theta[408])) * r_init[408] * omega_c[408]
*/
void SpiralGalaxy_eqFunction_6523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6523};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[407]] /* vx[408] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1914] /* theta[408] PARAM */)))) * (((data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */)) * ((data->simulationInfo->realParameter[912] /* omega_c[408] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12073(DATA *data, threadData_t *threadData);


/*
equation index: 6525
type: SIMPLE_ASSIGN
vy[408] = cos(theta[408]) * r_init[408] * omega_c[408]
*/
void SpiralGalaxy_eqFunction_6525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[907]] /* vy[408] STATE(1) */) = (cos((data->simulationInfo->realParameter[1914] /* theta[408] PARAM */))) * (((data->simulationInfo->realParameter[1413] /* r_init[408] PARAM */)) * ((data->simulationInfo->realParameter[912] /* omega_c[408] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12072(DATA *data, threadData_t *threadData);


/*
equation index: 6527
type: SIMPLE_ASSIGN
vz[408] = 0.0
*/
void SpiralGalaxy_eqFunction_6527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1407]] /* vz[408] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12071(DATA *data, threadData_t *threadData);


/*
equation index: 6529
type: SIMPLE_ASSIGN
z[409] = 0.02544
*/
void SpiralGalaxy_eqFunction_6529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6529};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2908]] /* z[409] STATE(1,vz[409]) */) = 0.02544;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12084(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12085(DATA *data, threadData_t *threadData);


/*
equation index: 6532
type: SIMPLE_ASSIGN
y[409] = r_init[409] * sin(theta[409] + 0.006359999999999999)
*/
void SpiralGalaxy_eqFunction_6532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6532};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2408]] /* y[409] STATE(1,vy[409]) */) = ((data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */)) * (sin((data->simulationInfo->realParameter[1915] /* theta[409] PARAM */) + 0.006359999999999999));
  TRACE_POP
}

/*
equation index: 6533
type: SIMPLE_ASSIGN
x[409] = r_init[409] * cos(theta[409] + 0.006359999999999999)
*/
void SpiralGalaxy_eqFunction_6533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1908]] /* x[409] STATE(1,vx[409]) */) = ((data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */)) * (cos((data->simulationInfo->realParameter[1915] /* theta[409] PARAM */) + 0.006359999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12086(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12087(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12090(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12089(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12088(DATA *data, threadData_t *threadData);


/*
equation index: 6539
type: SIMPLE_ASSIGN
vx[409] = (-sin(theta[409])) * r_init[409] * omega_c[409]
*/
void SpiralGalaxy_eqFunction_6539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6539};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[408]] /* vx[409] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1915] /* theta[409] PARAM */)))) * (((data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */)) * ((data->simulationInfo->realParameter[913] /* omega_c[409] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12083(DATA *data, threadData_t *threadData);


/*
equation index: 6541
type: SIMPLE_ASSIGN
vy[409] = cos(theta[409]) * r_init[409] * omega_c[409]
*/
void SpiralGalaxy_eqFunction_6541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[908]] /* vy[409] STATE(1) */) = (cos((data->simulationInfo->realParameter[1915] /* theta[409] PARAM */))) * (((data->simulationInfo->realParameter[1414] /* r_init[409] PARAM */)) * ((data->simulationInfo->realParameter[913] /* omega_c[409] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12082(DATA *data, threadData_t *threadData);


/*
equation index: 6543
type: SIMPLE_ASSIGN
vz[409] = 0.0
*/
void SpiralGalaxy_eqFunction_6543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6543};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1408]] /* vz[409] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12081(DATA *data, threadData_t *threadData);


/*
equation index: 6545
type: SIMPLE_ASSIGN
z[410] = 0.025600000000000005
*/
void SpiralGalaxy_eqFunction_6545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6545};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2909]] /* z[410] STATE(1,vz[410]) */) = 0.025600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12094(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12095(DATA *data, threadData_t *threadData);


/*
equation index: 6548
type: SIMPLE_ASSIGN
y[410] = r_init[410] * sin(theta[410] + 0.0063999999999999994)
*/
void SpiralGalaxy_eqFunction_6548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6548};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2409]] /* y[410] STATE(1,vy[410]) */) = ((data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */)) * (sin((data->simulationInfo->realParameter[1916] /* theta[410] PARAM */) + 0.0063999999999999994));
  TRACE_POP
}

/*
equation index: 6549
type: SIMPLE_ASSIGN
x[410] = r_init[410] * cos(theta[410] + 0.0063999999999999994)
*/
void SpiralGalaxy_eqFunction_6549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1909]] /* x[410] STATE(1,vx[410]) */) = ((data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */)) * (cos((data->simulationInfo->realParameter[1916] /* theta[410] PARAM */) + 0.0063999999999999994));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12096(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12097(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12100(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12099(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12098(DATA *data, threadData_t *threadData);


/*
equation index: 6555
type: SIMPLE_ASSIGN
vx[410] = (-sin(theta[410])) * r_init[410] * omega_c[410]
*/
void SpiralGalaxy_eqFunction_6555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6555};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[409]] /* vx[410] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1916] /* theta[410] PARAM */)))) * (((data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */)) * ((data->simulationInfo->realParameter[914] /* omega_c[410] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12093(DATA *data, threadData_t *threadData);


/*
equation index: 6557
type: SIMPLE_ASSIGN
vy[410] = cos(theta[410]) * r_init[410] * omega_c[410]
*/
void SpiralGalaxy_eqFunction_6557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[909]] /* vy[410] STATE(1) */) = (cos((data->simulationInfo->realParameter[1916] /* theta[410] PARAM */))) * (((data->simulationInfo->realParameter[1415] /* r_init[410] PARAM */)) * ((data->simulationInfo->realParameter[914] /* omega_c[410] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12092(DATA *data, threadData_t *threadData);


/*
equation index: 6559
type: SIMPLE_ASSIGN
vz[410] = 0.0
*/
void SpiralGalaxy_eqFunction_6559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6559};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1409]] /* vz[410] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12091(DATA *data, threadData_t *threadData);


/*
equation index: 6561
type: SIMPLE_ASSIGN
z[411] = 0.02576
*/
void SpiralGalaxy_eqFunction_6561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6561};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2910]] /* z[411] STATE(1,vz[411]) */) = 0.02576;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12104(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12105(DATA *data, threadData_t *threadData);


/*
equation index: 6564
type: SIMPLE_ASSIGN
y[411] = r_init[411] * sin(theta[411] + 0.0064399999999999995)
*/
void SpiralGalaxy_eqFunction_6564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6564};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2410]] /* y[411] STATE(1,vy[411]) */) = ((data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */)) * (sin((data->simulationInfo->realParameter[1917] /* theta[411] PARAM */) + 0.0064399999999999995));
  TRACE_POP
}

/*
equation index: 6565
type: SIMPLE_ASSIGN
x[411] = r_init[411] * cos(theta[411] + 0.0064399999999999995)
*/
void SpiralGalaxy_eqFunction_6565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1910]] /* x[411] STATE(1,vx[411]) */) = ((data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */)) * (cos((data->simulationInfo->realParameter[1917] /* theta[411] PARAM */) + 0.0064399999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12106(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12107(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12110(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12109(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12108(DATA *data, threadData_t *threadData);


/*
equation index: 6571
type: SIMPLE_ASSIGN
vx[411] = (-sin(theta[411])) * r_init[411] * omega_c[411]
*/
void SpiralGalaxy_eqFunction_6571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6571};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[410]] /* vx[411] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1917] /* theta[411] PARAM */)))) * (((data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */)) * ((data->simulationInfo->realParameter[915] /* omega_c[411] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12103(DATA *data, threadData_t *threadData);


/*
equation index: 6573
type: SIMPLE_ASSIGN
vy[411] = cos(theta[411]) * r_init[411] * omega_c[411]
*/
void SpiralGalaxy_eqFunction_6573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[910]] /* vy[411] STATE(1) */) = (cos((data->simulationInfo->realParameter[1917] /* theta[411] PARAM */))) * (((data->simulationInfo->realParameter[1416] /* r_init[411] PARAM */)) * ((data->simulationInfo->realParameter[915] /* omega_c[411] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12102(DATA *data, threadData_t *threadData);


/*
equation index: 6575
type: SIMPLE_ASSIGN
vz[411] = 0.0
*/
void SpiralGalaxy_eqFunction_6575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6575};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1410]] /* vz[411] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12101(DATA *data, threadData_t *threadData);


/*
equation index: 6577
type: SIMPLE_ASSIGN
z[412] = 0.025920000000000002
*/
void SpiralGalaxy_eqFunction_6577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6577};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2911]] /* z[412] STATE(1,vz[412]) */) = 0.025920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12114(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12115(DATA *data, threadData_t *threadData);


/*
equation index: 6580
type: SIMPLE_ASSIGN
y[412] = r_init[412] * sin(theta[412] + 0.00648)
*/
void SpiralGalaxy_eqFunction_6580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6580};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2411]] /* y[412] STATE(1,vy[412]) */) = ((data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */)) * (sin((data->simulationInfo->realParameter[1918] /* theta[412] PARAM */) + 0.00648));
  TRACE_POP
}

/*
equation index: 6581
type: SIMPLE_ASSIGN
x[412] = r_init[412] * cos(theta[412] + 0.00648)
*/
void SpiralGalaxy_eqFunction_6581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1911]] /* x[412] STATE(1,vx[412]) */) = ((data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */)) * (cos((data->simulationInfo->realParameter[1918] /* theta[412] PARAM */) + 0.00648));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12116(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12117(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12120(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12119(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12118(DATA *data, threadData_t *threadData);


/*
equation index: 6587
type: SIMPLE_ASSIGN
vx[412] = (-sin(theta[412])) * r_init[412] * omega_c[412]
*/
void SpiralGalaxy_eqFunction_6587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6587};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[411]] /* vx[412] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1918] /* theta[412] PARAM */)))) * (((data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */)) * ((data->simulationInfo->realParameter[916] /* omega_c[412] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12113(DATA *data, threadData_t *threadData);


/*
equation index: 6589
type: SIMPLE_ASSIGN
vy[412] = cos(theta[412]) * r_init[412] * omega_c[412]
*/
void SpiralGalaxy_eqFunction_6589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[911]] /* vy[412] STATE(1) */) = (cos((data->simulationInfo->realParameter[1918] /* theta[412] PARAM */))) * (((data->simulationInfo->realParameter[1417] /* r_init[412] PARAM */)) * ((data->simulationInfo->realParameter[916] /* omega_c[412] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12112(DATA *data, threadData_t *threadData);


/*
equation index: 6591
type: SIMPLE_ASSIGN
vz[412] = 0.0
*/
void SpiralGalaxy_eqFunction_6591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6591};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1411]] /* vz[412] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12111(DATA *data, threadData_t *threadData);


/*
equation index: 6593
type: SIMPLE_ASSIGN
z[413] = 0.026080000000000002
*/
void SpiralGalaxy_eqFunction_6593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6593};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2912]] /* z[413] STATE(1,vz[413]) */) = 0.026080000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12124(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12125(DATA *data, threadData_t *threadData);


/*
equation index: 6596
type: SIMPLE_ASSIGN
y[413] = r_init[413] * sin(theta[413] + 0.006519999999999999)
*/
void SpiralGalaxy_eqFunction_6596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6596};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2412]] /* y[413] STATE(1,vy[413]) */) = ((data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */)) * (sin((data->simulationInfo->realParameter[1919] /* theta[413] PARAM */) + 0.006519999999999999));
  TRACE_POP
}

/*
equation index: 6597
type: SIMPLE_ASSIGN
x[413] = r_init[413] * cos(theta[413] + 0.006519999999999999)
*/
void SpiralGalaxy_eqFunction_6597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1912]] /* x[413] STATE(1,vx[413]) */) = ((data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */)) * (cos((data->simulationInfo->realParameter[1919] /* theta[413] PARAM */) + 0.006519999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12126(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12127(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12130(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12129(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12128(DATA *data, threadData_t *threadData);


/*
equation index: 6603
type: SIMPLE_ASSIGN
vx[413] = (-sin(theta[413])) * r_init[413] * omega_c[413]
*/
void SpiralGalaxy_eqFunction_6603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6603};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[412]] /* vx[413] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1919] /* theta[413] PARAM */)))) * (((data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */)) * ((data->simulationInfo->realParameter[917] /* omega_c[413] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12123(DATA *data, threadData_t *threadData);


/*
equation index: 6605
type: SIMPLE_ASSIGN
vy[413] = cos(theta[413]) * r_init[413] * omega_c[413]
*/
void SpiralGalaxy_eqFunction_6605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[912]] /* vy[413] STATE(1) */) = (cos((data->simulationInfo->realParameter[1919] /* theta[413] PARAM */))) * (((data->simulationInfo->realParameter[1418] /* r_init[413] PARAM */)) * ((data->simulationInfo->realParameter[917] /* omega_c[413] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12122(DATA *data, threadData_t *threadData);


/*
equation index: 6607
type: SIMPLE_ASSIGN
vz[413] = 0.0
*/
void SpiralGalaxy_eqFunction_6607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6607};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1412]] /* vz[413] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12121(DATA *data, threadData_t *threadData);


/*
equation index: 6609
type: SIMPLE_ASSIGN
z[414] = 0.02624
*/
void SpiralGalaxy_eqFunction_6609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6609};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2913]] /* z[414] STATE(1,vz[414]) */) = 0.02624;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12134(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12135(DATA *data, threadData_t *threadData);


/*
equation index: 6612
type: SIMPLE_ASSIGN
y[414] = r_init[414] * sin(theta[414] + 0.006559999999999999)
*/
void SpiralGalaxy_eqFunction_6612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6612};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2413]] /* y[414] STATE(1,vy[414]) */) = ((data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */)) * (sin((data->simulationInfo->realParameter[1920] /* theta[414] PARAM */) + 0.006559999999999999));
  TRACE_POP
}

/*
equation index: 6613
type: SIMPLE_ASSIGN
x[414] = r_init[414] * cos(theta[414] + 0.006559999999999999)
*/
void SpiralGalaxy_eqFunction_6613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1913]] /* x[414] STATE(1,vx[414]) */) = ((data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */)) * (cos((data->simulationInfo->realParameter[1920] /* theta[414] PARAM */) + 0.006559999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12136(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12137(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12140(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12139(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12138(DATA *data, threadData_t *threadData);


/*
equation index: 6619
type: SIMPLE_ASSIGN
vx[414] = (-sin(theta[414])) * r_init[414] * omega_c[414]
*/
void SpiralGalaxy_eqFunction_6619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6619};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[413]] /* vx[414] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1920] /* theta[414] PARAM */)))) * (((data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */)) * ((data->simulationInfo->realParameter[918] /* omega_c[414] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12133(DATA *data, threadData_t *threadData);


/*
equation index: 6621
type: SIMPLE_ASSIGN
vy[414] = cos(theta[414]) * r_init[414] * omega_c[414]
*/
void SpiralGalaxy_eqFunction_6621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[913]] /* vy[414] STATE(1) */) = (cos((data->simulationInfo->realParameter[1920] /* theta[414] PARAM */))) * (((data->simulationInfo->realParameter[1419] /* r_init[414] PARAM */)) * ((data->simulationInfo->realParameter[918] /* omega_c[414] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12132(DATA *data, threadData_t *threadData);


/*
equation index: 6623
type: SIMPLE_ASSIGN
vz[414] = 0.0
*/
void SpiralGalaxy_eqFunction_6623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6623};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1413]] /* vz[414] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12131(DATA *data, threadData_t *threadData);


/*
equation index: 6625
type: SIMPLE_ASSIGN
z[415] = 0.026400000000000003
*/
void SpiralGalaxy_eqFunction_6625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6625};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2914]] /* z[415] STATE(1,vz[415]) */) = 0.026400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12144(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12145(DATA *data, threadData_t *threadData);


/*
equation index: 6628
type: SIMPLE_ASSIGN
y[415] = r_init[415] * sin(theta[415] + 0.006599999999999999)
*/
void SpiralGalaxy_eqFunction_6628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6628};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2414]] /* y[415] STATE(1,vy[415]) */) = ((data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */)) * (sin((data->simulationInfo->realParameter[1921] /* theta[415] PARAM */) + 0.006599999999999999));
  TRACE_POP
}

/*
equation index: 6629
type: SIMPLE_ASSIGN
x[415] = r_init[415] * cos(theta[415] + 0.006599999999999999)
*/
void SpiralGalaxy_eqFunction_6629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1914]] /* x[415] STATE(1,vx[415]) */) = ((data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */)) * (cos((data->simulationInfo->realParameter[1921] /* theta[415] PARAM */) + 0.006599999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12146(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12147(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12150(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12149(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12148(DATA *data, threadData_t *threadData);


/*
equation index: 6635
type: SIMPLE_ASSIGN
vx[415] = (-sin(theta[415])) * r_init[415] * omega_c[415]
*/
void SpiralGalaxy_eqFunction_6635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6635};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[414]] /* vx[415] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1921] /* theta[415] PARAM */)))) * (((data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */)) * ((data->simulationInfo->realParameter[919] /* omega_c[415] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12143(DATA *data, threadData_t *threadData);


/*
equation index: 6637
type: SIMPLE_ASSIGN
vy[415] = cos(theta[415]) * r_init[415] * omega_c[415]
*/
void SpiralGalaxy_eqFunction_6637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[914]] /* vy[415] STATE(1) */) = (cos((data->simulationInfo->realParameter[1921] /* theta[415] PARAM */))) * (((data->simulationInfo->realParameter[1420] /* r_init[415] PARAM */)) * ((data->simulationInfo->realParameter[919] /* omega_c[415] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12142(DATA *data, threadData_t *threadData);


/*
equation index: 6639
type: SIMPLE_ASSIGN
vz[415] = 0.0
*/
void SpiralGalaxy_eqFunction_6639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6639};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1414]] /* vz[415] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12141(DATA *data, threadData_t *threadData);


/*
equation index: 6641
type: SIMPLE_ASSIGN
z[416] = 0.026560000000000004
*/
void SpiralGalaxy_eqFunction_6641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6641};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2915]] /* z[416] STATE(1,vz[416]) */) = 0.026560000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12154(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12155(DATA *data, threadData_t *threadData);


/*
equation index: 6644
type: SIMPLE_ASSIGN
y[416] = r_init[416] * sin(theta[416] + 0.006639999999999999)
*/
void SpiralGalaxy_eqFunction_6644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6644};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2415]] /* y[416] STATE(1,vy[416]) */) = ((data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */)) * (sin((data->simulationInfo->realParameter[1922] /* theta[416] PARAM */) + 0.006639999999999999));
  TRACE_POP
}

/*
equation index: 6645
type: SIMPLE_ASSIGN
x[416] = r_init[416] * cos(theta[416] + 0.006639999999999999)
*/
void SpiralGalaxy_eqFunction_6645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1915]] /* x[416] STATE(1,vx[416]) */) = ((data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */)) * (cos((data->simulationInfo->realParameter[1922] /* theta[416] PARAM */) + 0.006639999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12156(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12157(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12160(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12159(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12158(DATA *data, threadData_t *threadData);


/*
equation index: 6651
type: SIMPLE_ASSIGN
vx[416] = (-sin(theta[416])) * r_init[416] * omega_c[416]
*/
void SpiralGalaxy_eqFunction_6651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6651};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[415]] /* vx[416] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1922] /* theta[416] PARAM */)))) * (((data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */)) * ((data->simulationInfo->realParameter[920] /* omega_c[416] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12153(DATA *data, threadData_t *threadData);


/*
equation index: 6653
type: SIMPLE_ASSIGN
vy[416] = cos(theta[416]) * r_init[416] * omega_c[416]
*/
void SpiralGalaxy_eqFunction_6653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[915]] /* vy[416] STATE(1) */) = (cos((data->simulationInfo->realParameter[1922] /* theta[416] PARAM */))) * (((data->simulationInfo->realParameter[1421] /* r_init[416] PARAM */)) * ((data->simulationInfo->realParameter[920] /* omega_c[416] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12152(DATA *data, threadData_t *threadData);


/*
equation index: 6655
type: SIMPLE_ASSIGN
vz[416] = 0.0
*/
void SpiralGalaxy_eqFunction_6655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6655};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1415]] /* vz[416] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12151(DATA *data, threadData_t *threadData);


/*
equation index: 6657
type: SIMPLE_ASSIGN
z[417] = 0.02672
*/
void SpiralGalaxy_eqFunction_6657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6657};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2916]] /* z[417] STATE(1,vz[417]) */) = 0.02672;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12164(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12165(DATA *data, threadData_t *threadData);


/*
equation index: 6660
type: SIMPLE_ASSIGN
y[417] = r_init[417] * sin(theta[417] + 0.006679999999999999)
*/
void SpiralGalaxy_eqFunction_6660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6660};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2416]] /* y[417] STATE(1,vy[417]) */) = ((data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */)) * (sin((data->simulationInfo->realParameter[1923] /* theta[417] PARAM */) + 0.006679999999999999));
  TRACE_POP
}

/*
equation index: 6661
type: SIMPLE_ASSIGN
x[417] = r_init[417] * cos(theta[417] + 0.006679999999999999)
*/
void SpiralGalaxy_eqFunction_6661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1916]] /* x[417] STATE(1,vx[417]) */) = ((data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */)) * (cos((data->simulationInfo->realParameter[1923] /* theta[417] PARAM */) + 0.006679999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12166(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12167(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12170(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12169(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12168(DATA *data, threadData_t *threadData);


/*
equation index: 6667
type: SIMPLE_ASSIGN
vx[417] = (-sin(theta[417])) * r_init[417] * omega_c[417]
*/
void SpiralGalaxy_eqFunction_6667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6667};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[416]] /* vx[417] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1923] /* theta[417] PARAM */)))) * (((data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */)) * ((data->simulationInfo->realParameter[921] /* omega_c[417] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12163(DATA *data, threadData_t *threadData);


/*
equation index: 6669
type: SIMPLE_ASSIGN
vy[417] = cos(theta[417]) * r_init[417] * omega_c[417]
*/
void SpiralGalaxy_eqFunction_6669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[916]] /* vy[417] STATE(1) */) = (cos((data->simulationInfo->realParameter[1923] /* theta[417] PARAM */))) * (((data->simulationInfo->realParameter[1422] /* r_init[417] PARAM */)) * ((data->simulationInfo->realParameter[921] /* omega_c[417] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12162(DATA *data, threadData_t *threadData);


/*
equation index: 6671
type: SIMPLE_ASSIGN
vz[417] = 0.0
*/
void SpiralGalaxy_eqFunction_6671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6671};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1416]] /* vz[417] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12161(DATA *data, threadData_t *threadData);


/*
equation index: 6673
type: SIMPLE_ASSIGN
z[418] = 0.02688
*/
void SpiralGalaxy_eqFunction_6673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6673};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2917]] /* z[418] STATE(1,vz[418]) */) = 0.02688;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12174(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12175(DATA *data, threadData_t *threadData);


/*
equation index: 6676
type: SIMPLE_ASSIGN
y[418] = r_init[418] * sin(theta[418] + 0.006719999999999999)
*/
void SpiralGalaxy_eqFunction_6676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6676};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2417]] /* y[418] STATE(1,vy[418]) */) = ((data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */)) * (sin((data->simulationInfo->realParameter[1924] /* theta[418] PARAM */) + 0.006719999999999999));
  TRACE_POP
}

/*
equation index: 6677
type: SIMPLE_ASSIGN
x[418] = r_init[418] * cos(theta[418] + 0.006719999999999999)
*/
void SpiralGalaxy_eqFunction_6677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1917]] /* x[418] STATE(1,vx[418]) */) = ((data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */)) * (cos((data->simulationInfo->realParameter[1924] /* theta[418] PARAM */) + 0.006719999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12176(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12177(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12180(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12179(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12178(DATA *data, threadData_t *threadData);


/*
equation index: 6683
type: SIMPLE_ASSIGN
vx[418] = (-sin(theta[418])) * r_init[418] * omega_c[418]
*/
void SpiralGalaxy_eqFunction_6683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6683};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[417]] /* vx[418] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1924] /* theta[418] PARAM */)))) * (((data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */)) * ((data->simulationInfo->realParameter[922] /* omega_c[418] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12173(DATA *data, threadData_t *threadData);


/*
equation index: 6685
type: SIMPLE_ASSIGN
vy[418] = cos(theta[418]) * r_init[418] * omega_c[418]
*/
void SpiralGalaxy_eqFunction_6685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[917]] /* vy[418] STATE(1) */) = (cos((data->simulationInfo->realParameter[1924] /* theta[418] PARAM */))) * (((data->simulationInfo->realParameter[1423] /* r_init[418] PARAM */)) * ((data->simulationInfo->realParameter[922] /* omega_c[418] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12172(DATA *data, threadData_t *threadData);


/*
equation index: 6687
type: SIMPLE_ASSIGN
vz[418] = 0.0
*/
void SpiralGalaxy_eqFunction_6687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6687};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1417]] /* vz[418] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12171(DATA *data, threadData_t *threadData);


/*
equation index: 6689
type: SIMPLE_ASSIGN
z[419] = 0.027040000000000005
*/
void SpiralGalaxy_eqFunction_6689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6689};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2918]] /* z[419] STATE(1,vz[419]) */) = 0.027040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12184(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12185(DATA *data, threadData_t *threadData);


/*
equation index: 6692
type: SIMPLE_ASSIGN
y[419] = r_init[419] * sin(theta[419] + 0.0067599999999999995)
*/
void SpiralGalaxy_eqFunction_6692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6692};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2418]] /* y[419] STATE(1,vy[419]) */) = ((data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */)) * (sin((data->simulationInfo->realParameter[1925] /* theta[419] PARAM */) + 0.0067599999999999995));
  TRACE_POP
}

/*
equation index: 6693
type: SIMPLE_ASSIGN
x[419] = r_init[419] * cos(theta[419] + 0.0067599999999999995)
*/
void SpiralGalaxy_eqFunction_6693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1918]] /* x[419] STATE(1,vx[419]) */) = ((data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */)) * (cos((data->simulationInfo->realParameter[1925] /* theta[419] PARAM */) + 0.0067599999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12186(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12187(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12190(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12189(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12188(DATA *data, threadData_t *threadData);


/*
equation index: 6699
type: SIMPLE_ASSIGN
vx[419] = (-sin(theta[419])) * r_init[419] * omega_c[419]
*/
void SpiralGalaxy_eqFunction_6699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6699};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[418]] /* vx[419] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1925] /* theta[419] PARAM */)))) * (((data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */)) * ((data->simulationInfo->realParameter[923] /* omega_c[419] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12183(DATA *data, threadData_t *threadData);


/*
equation index: 6701
type: SIMPLE_ASSIGN
vy[419] = cos(theta[419]) * r_init[419] * omega_c[419]
*/
void SpiralGalaxy_eqFunction_6701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[918]] /* vy[419] STATE(1) */) = (cos((data->simulationInfo->realParameter[1925] /* theta[419] PARAM */))) * (((data->simulationInfo->realParameter[1424] /* r_init[419] PARAM */)) * ((data->simulationInfo->realParameter[923] /* omega_c[419] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12182(DATA *data, threadData_t *threadData);


/*
equation index: 6703
type: SIMPLE_ASSIGN
vz[419] = 0.0
*/
void SpiralGalaxy_eqFunction_6703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6703};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1418]] /* vz[419] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12181(DATA *data, threadData_t *threadData);


/*
equation index: 6705
type: SIMPLE_ASSIGN
z[420] = 0.027200000000000002
*/
void SpiralGalaxy_eqFunction_6705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6705};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2919]] /* z[420] STATE(1,vz[420]) */) = 0.027200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12194(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12195(DATA *data, threadData_t *threadData);


/*
equation index: 6708
type: SIMPLE_ASSIGN
y[420] = r_init[420] * sin(theta[420] + 0.0068)
*/
void SpiralGalaxy_eqFunction_6708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6708};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2419]] /* y[420] STATE(1,vy[420]) */) = ((data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */)) * (sin((data->simulationInfo->realParameter[1926] /* theta[420] PARAM */) + 0.0068));
  TRACE_POP
}

/*
equation index: 6709
type: SIMPLE_ASSIGN
x[420] = r_init[420] * cos(theta[420] + 0.0068)
*/
void SpiralGalaxy_eqFunction_6709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1919]] /* x[420] STATE(1,vx[420]) */) = ((data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */)) * (cos((data->simulationInfo->realParameter[1926] /* theta[420] PARAM */) + 0.0068));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12196(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12197(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12200(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12199(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12198(DATA *data, threadData_t *threadData);


/*
equation index: 6715
type: SIMPLE_ASSIGN
vx[420] = (-sin(theta[420])) * r_init[420] * omega_c[420]
*/
void SpiralGalaxy_eqFunction_6715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6715};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[419]] /* vx[420] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1926] /* theta[420] PARAM */)))) * (((data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */)) * ((data->simulationInfo->realParameter[924] /* omega_c[420] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12193(DATA *data, threadData_t *threadData);


/*
equation index: 6717
type: SIMPLE_ASSIGN
vy[420] = cos(theta[420]) * r_init[420] * omega_c[420]
*/
void SpiralGalaxy_eqFunction_6717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[919]] /* vy[420] STATE(1) */) = (cos((data->simulationInfo->realParameter[1926] /* theta[420] PARAM */))) * (((data->simulationInfo->realParameter[1425] /* r_init[420] PARAM */)) * ((data->simulationInfo->realParameter[924] /* omega_c[420] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12192(DATA *data, threadData_t *threadData);


/*
equation index: 6719
type: SIMPLE_ASSIGN
vz[420] = 0.0
*/
void SpiralGalaxy_eqFunction_6719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6719};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1419]] /* vz[420] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12191(DATA *data, threadData_t *threadData);


/*
equation index: 6721
type: SIMPLE_ASSIGN
z[421] = 0.027360000000000002
*/
void SpiralGalaxy_eqFunction_6721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6721};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2920]] /* z[421] STATE(1,vz[421]) */) = 0.027360000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12204(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12205(DATA *data, threadData_t *threadData);


/*
equation index: 6724
type: SIMPLE_ASSIGN
y[421] = r_init[421] * sin(theta[421] + 0.00684)
*/
void SpiralGalaxy_eqFunction_6724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6724};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2420]] /* y[421] STATE(1,vy[421]) */) = ((data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */)) * (sin((data->simulationInfo->realParameter[1927] /* theta[421] PARAM */) + 0.00684));
  TRACE_POP
}

/*
equation index: 6725
type: SIMPLE_ASSIGN
x[421] = r_init[421] * cos(theta[421] + 0.00684)
*/
void SpiralGalaxy_eqFunction_6725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1920]] /* x[421] STATE(1,vx[421]) */) = ((data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */)) * (cos((data->simulationInfo->realParameter[1927] /* theta[421] PARAM */) + 0.00684));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12206(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12207(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12210(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12209(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12208(DATA *data, threadData_t *threadData);


/*
equation index: 6731
type: SIMPLE_ASSIGN
vx[421] = (-sin(theta[421])) * r_init[421] * omega_c[421]
*/
void SpiralGalaxy_eqFunction_6731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6731};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[420]] /* vx[421] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1927] /* theta[421] PARAM */)))) * (((data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */)) * ((data->simulationInfo->realParameter[925] /* omega_c[421] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12203(DATA *data, threadData_t *threadData);


/*
equation index: 6733
type: SIMPLE_ASSIGN
vy[421] = cos(theta[421]) * r_init[421] * omega_c[421]
*/
void SpiralGalaxy_eqFunction_6733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[920]] /* vy[421] STATE(1) */) = (cos((data->simulationInfo->realParameter[1927] /* theta[421] PARAM */))) * (((data->simulationInfo->realParameter[1426] /* r_init[421] PARAM */)) * ((data->simulationInfo->realParameter[925] /* omega_c[421] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12202(DATA *data, threadData_t *threadData);


/*
equation index: 6735
type: SIMPLE_ASSIGN
vz[421] = 0.0
*/
void SpiralGalaxy_eqFunction_6735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6735};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1420]] /* vz[421] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12201(DATA *data, threadData_t *threadData);


/*
equation index: 6737
type: SIMPLE_ASSIGN
z[422] = 0.02752
*/
void SpiralGalaxy_eqFunction_6737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6737};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2921]] /* z[422] STATE(1,vz[422]) */) = 0.02752;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12214(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12215(DATA *data, threadData_t *threadData);


/*
equation index: 6740
type: SIMPLE_ASSIGN
y[422] = r_init[422] * sin(theta[422] + 0.00688)
*/
void SpiralGalaxy_eqFunction_6740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6740};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2421]] /* y[422] STATE(1,vy[422]) */) = ((data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */)) * (sin((data->simulationInfo->realParameter[1928] /* theta[422] PARAM */) + 0.00688));
  TRACE_POP
}

/*
equation index: 6741
type: SIMPLE_ASSIGN
x[422] = r_init[422] * cos(theta[422] + 0.00688)
*/
void SpiralGalaxy_eqFunction_6741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1921]] /* x[422] STATE(1,vx[422]) */) = ((data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */)) * (cos((data->simulationInfo->realParameter[1928] /* theta[422] PARAM */) + 0.00688));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12216(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12217(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12220(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12219(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12218(DATA *data, threadData_t *threadData);


/*
equation index: 6747
type: SIMPLE_ASSIGN
vx[422] = (-sin(theta[422])) * r_init[422] * omega_c[422]
*/
void SpiralGalaxy_eqFunction_6747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6747};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[421]] /* vx[422] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1928] /* theta[422] PARAM */)))) * (((data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */)) * ((data->simulationInfo->realParameter[926] /* omega_c[422] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12213(DATA *data, threadData_t *threadData);


/*
equation index: 6749
type: SIMPLE_ASSIGN
vy[422] = cos(theta[422]) * r_init[422] * omega_c[422]
*/
void SpiralGalaxy_eqFunction_6749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[921]] /* vy[422] STATE(1) */) = (cos((data->simulationInfo->realParameter[1928] /* theta[422] PARAM */))) * (((data->simulationInfo->realParameter[1427] /* r_init[422] PARAM */)) * ((data->simulationInfo->realParameter[926] /* omega_c[422] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12212(DATA *data, threadData_t *threadData);


/*
equation index: 6751
type: SIMPLE_ASSIGN
vz[422] = 0.0
*/
void SpiralGalaxy_eqFunction_6751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6751};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1421]] /* vz[422] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12211(DATA *data, threadData_t *threadData);


/*
equation index: 6753
type: SIMPLE_ASSIGN
z[423] = 0.027680000000000003
*/
void SpiralGalaxy_eqFunction_6753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6753};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2922]] /* z[423] STATE(1,vz[423]) */) = 0.027680000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12224(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12225(DATA *data, threadData_t *threadData);


/*
equation index: 6756
type: SIMPLE_ASSIGN
y[423] = r_init[423] * sin(theta[423] + 0.00692)
*/
void SpiralGalaxy_eqFunction_6756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6756};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2422]] /* y[423] STATE(1,vy[423]) */) = ((data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */)) * (sin((data->simulationInfo->realParameter[1929] /* theta[423] PARAM */) + 0.00692));
  TRACE_POP
}

/*
equation index: 6757
type: SIMPLE_ASSIGN
x[423] = r_init[423] * cos(theta[423] + 0.00692)
*/
void SpiralGalaxy_eqFunction_6757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1922]] /* x[423] STATE(1,vx[423]) */) = ((data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */)) * (cos((data->simulationInfo->realParameter[1929] /* theta[423] PARAM */) + 0.00692));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12226(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12227(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12230(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12229(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12228(DATA *data, threadData_t *threadData);


/*
equation index: 6763
type: SIMPLE_ASSIGN
vx[423] = (-sin(theta[423])) * r_init[423] * omega_c[423]
*/
void SpiralGalaxy_eqFunction_6763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6763};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[422]] /* vx[423] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1929] /* theta[423] PARAM */)))) * (((data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */)) * ((data->simulationInfo->realParameter[927] /* omega_c[423] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12223(DATA *data, threadData_t *threadData);


/*
equation index: 6765
type: SIMPLE_ASSIGN
vy[423] = cos(theta[423]) * r_init[423] * omega_c[423]
*/
void SpiralGalaxy_eqFunction_6765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[922]] /* vy[423] STATE(1) */) = (cos((data->simulationInfo->realParameter[1929] /* theta[423] PARAM */))) * (((data->simulationInfo->realParameter[1428] /* r_init[423] PARAM */)) * ((data->simulationInfo->realParameter[927] /* omega_c[423] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12222(DATA *data, threadData_t *threadData);


/*
equation index: 6767
type: SIMPLE_ASSIGN
vz[423] = 0.0
*/
void SpiralGalaxy_eqFunction_6767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6767};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1422]] /* vz[423] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12221(DATA *data, threadData_t *threadData);


/*
equation index: 6769
type: SIMPLE_ASSIGN
z[424] = 0.027840000000000004
*/
void SpiralGalaxy_eqFunction_6769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6769};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2923]] /* z[424] STATE(1,vz[424]) */) = 0.027840000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12234(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12235(DATA *data, threadData_t *threadData);


/*
equation index: 6772
type: SIMPLE_ASSIGN
y[424] = r_init[424] * sin(theta[424] + 0.00696)
*/
void SpiralGalaxy_eqFunction_6772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6772};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2423]] /* y[424] STATE(1,vy[424]) */) = ((data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */)) * (sin((data->simulationInfo->realParameter[1930] /* theta[424] PARAM */) + 0.00696));
  TRACE_POP
}

/*
equation index: 6773
type: SIMPLE_ASSIGN
x[424] = r_init[424] * cos(theta[424] + 0.00696)
*/
void SpiralGalaxy_eqFunction_6773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1923]] /* x[424] STATE(1,vx[424]) */) = ((data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */)) * (cos((data->simulationInfo->realParameter[1930] /* theta[424] PARAM */) + 0.00696));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12236(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12237(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12240(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12239(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12238(DATA *data, threadData_t *threadData);


/*
equation index: 6779
type: SIMPLE_ASSIGN
vx[424] = (-sin(theta[424])) * r_init[424] * omega_c[424]
*/
void SpiralGalaxy_eqFunction_6779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6779};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[423]] /* vx[424] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1930] /* theta[424] PARAM */)))) * (((data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */)) * ((data->simulationInfo->realParameter[928] /* omega_c[424] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12233(DATA *data, threadData_t *threadData);


/*
equation index: 6781
type: SIMPLE_ASSIGN
vy[424] = cos(theta[424]) * r_init[424] * omega_c[424]
*/
void SpiralGalaxy_eqFunction_6781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[923]] /* vy[424] STATE(1) */) = (cos((data->simulationInfo->realParameter[1930] /* theta[424] PARAM */))) * (((data->simulationInfo->realParameter[1429] /* r_init[424] PARAM */)) * ((data->simulationInfo->realParameter[928] /* omega_c[424] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12232(DATA *data, threadData_t *threadData);


/*
equation index: 6783
type: SIMPLE_ASSIGN
vz[424] = 0.0
*/
void SpiralGalaxy_eqFunction_6783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6783};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1423]] /* vz[424] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12231(DATA *data, threadData_t *threadData);


/*
equation index: 6785
type: SIMPLE_ASSIGN
z[425] = 0.028000000000000004
*/
void SpiralGalaxy_eqFunction_6785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6785};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2924]] /* z[425] STATE(1,vz[425]) */) = 0.028000000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12244(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12245(DATA *data, threadData_t *threadData);


/*
equation index: 6788
type: SIMPLE_ASSIGN
y[425] = r_init[425] * sin(theta[425] + 0.006999999999999999)
*/
void SpiralGalaxy_eqFunction_6788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6788};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2424]] /* y[425] STATE(1,vy[425]) */) = ((data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */)) * (sin((data->simulationInfo->realParameter[1931] /* theta[425] PARAM */) + 0.006999999999999999));
  TRACE_POP
}

/*
equation index: 6789
type: SIMPLE_ASSIGN
x[425] = r_init[425] * cos(theta[425] + 0.006999999999999999)
*/
void SpiralGalaxy_eqFunction_6789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1924]] /* x[425] STATE(1,vx[425]) */) = ((data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */)) * (cos((data->simulationInfo->realParameter[1931] /* theta[425] PARAM */) + 0.006999999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12246(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12247(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12250(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12249(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12248(DATA *data, threadData_t *threadData);


/*
equation index: 6795
type: SIMPLE_ASSIGN
vx[425] = (-sin(theta[425])) * r_init[425] * omega_c[425]
*/
void SpiralGalaxy_eqFunction_6795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6795};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[424]] /* vx[425] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1931] /* theta[425] PARAM */)))) * (((data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */)) * ((data->simulationInfo->realParameter[929] /* omega_c[425] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12243(DATA *data, threadData_t *threadData);


/*
equation index: 6797
type: SIMPLE_ASSIGN
vy[425] = cos(theta[425]) * r_init[425] * omega_c[425]
*/
void SpiralGalaxy_eqFunction_6797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[924]] /* vy[425] STATE(1) */) = (cos((data->simulationInfo->realParameter[1931] /* theta[425] PARAM */))) * (((data->simulationInfo->realParameter[1430] /* r_init[425] PARAM */)) * ((data->simulationInfo->realParameter[929] /* omega_c[425] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12242(DATA *data, threadData_t *threadData);


/*
equation index: 6799
type: SIMPLE_ASSIGN
vz[425] = 0.0
*/
void SpiralGalaxy_eqFunction_6799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6799};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1424]] /* vz[425] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12241(DATA *data, threadData_t *threadData);


/*
equation index: 6801
type: SIMPLE_ASSIGN
z[426] = 0.02816
*/
void SpiralGalaxy_eqFunction_6801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6801};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2925]] /* z[426] STATE(1,vz[426]) */) = 0.02816;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12254(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12255(DATA *data, threadData_t *threadData);


/*
equation index: 6804
type: SIMPLE_ASSIGN
y[426] = r_init[426] * sin(theta[426] + 0.007039999999999999)
*/
void SpiralGalaxy_eqFunction_6804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6804};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2425]] /* y[426] STATE(1,vy[426]) */) = ((data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */)) * (sin((data->simulationInfo->realParameter[1932] /* theta[426] PARAM */) + 0.007039999999999999));
  TRACE_POP
}

/*
equation index: 6805
type: SIMPLE_ASSIGN
x[426] = r_init[426] * cos(theta[426] + 0.007039999999999999)
*/
void SpiralGalaxy_eqFunction_6805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1925]] /* x[426] STATE(1,vx[426]) */) = ((data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */)) * (cos((data->simulationInfo->realParameter[1932] /* theta[426] PARAM */) + 0.007039999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12256(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12257(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12260(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12259(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12258(DATA *data, threadData_t *threadData);


/*
equation index: 6811
type: SIMPLE_ASSIGN
vx[426] = (-sin(theta[426])) * r_init[426] * omega_c[426]
*/
void SpiralGalaxy_eqFunction_6811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6811};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[425]] /* vx[426] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1932] /* theta[426] PARAM */)))) * (((data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */)) * ((data->simulationInfo->realParameter[930] /* omega_c[426] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12253(DATA *data, threadData_t *threadData);


/*
equation index: 6813
type: SIMPLE_ASSIGN
vy[426] = cos(theta[426]) * r_init[426] * omega_c[426]
*/
void SpiralGalaxy_eqFunction_6813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[925]] /* vy[426] STATE(1) */) = (cos((data->simulationInfo->realParameter[1932] /* theta[426] PARAM */))) * (((data->simulationInfo->realParameter[1431] /* r_init[426] PARAM */)) * ((data->simulationInfo->realParameter[930] /* omega_c[426] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12252(DATA *data, threadData_t *threadData);


/*
equation index: 6815
type: SIMPLE_ASSIGN
vz[426] = 0.0
*/
void SpiralGalaxy_eqFunction_6815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6815};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1425]] /* vz[426] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12251(DATA *data, threadData_t *threadData);


/*
equation index: 6817
type: SIMPLE_ASSIGN
z[427] = 0.028319999999999998
*/
void SpiralGalaxy_eqFunction_6817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6817};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2926]] /* z[427] STATE(1,vz[427]) */) = 0.028319999999999998;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12264(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12265(DATA *data, threadData_t *threadData);


/*
equation index: 6820
type: SIMPLE_ASSIGN
y[427] = r_init[427] * sin(theta[427] + 0.0070799999999999995)
*/
void SpiralGalaxy_eqFunction_6820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6820};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2426]] /* y[427] STATE(1,vy[427]) */) = ((data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */)) * (sin((data->simulationInfo->realParameter[1933] /* theta[427] PARAM */) + 0.0070799999999999995));
  TRACE_POP
}

/*
equation index: 6821
type: SIMPLE_ASSIGN
x[427] = r_init[427] * cos(theta[427] + 0.0070799999999999995)
*/
void SpiralGalaxy_eqFunction_6821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1926]] /* x[427] STATE(1,vx[427]) */) = ((data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */)) * (cos((data->simulationInfo->realParameter[1933] /* theta[427] PARAM */) + 0.0070799999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12266(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12267(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12270(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12269(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12268(DATA *data, threadData_t *threadData);


/*
equation index: 6827
type: SIMPLE_ASSIGN
vx[427] = (-sin(theta[427])) * r_init[427] * omega_c[427]
*/
void SpiralGalaxy_eqFunction_6827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6827};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[426]] /* vx[427] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1933] /* theta[427] PARAM */)))) * (((data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */)) * ((data->simulationInfo->realParameter[931] /* omega_c[427] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12263(DATA *data, threadData_t *threadData);


/*
equation index: 6829
type: SIMPLE_ASSIGN
vy[427] = cos(theta[427]) * r_init[427] * omega_c[427]
*/
void SpiralGalaxy_eqFunction_6829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[926]] /* vy[427] STATE(1) */) = (cos((data->simulationInfo->realParameter[1933] /* theta[427] PARAM */))) * (((data->simulationInfo->realParameter[1432] /* r_init[427] PARAM */)) * ((data->simulationInfo->realParameter[931] /* omega_c[427] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12262(DATA *data, threadData_t *threadData);


/*
equation index: 6831
type: SIMPLE_ASSIGN
vz[427] = 0.0
*/
void SpiralGalaxy_eqFunction_6831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6831};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1426]] /* vz[427] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12261(DATA *data, threadData_t *threadData);


/*
equation index: 6833
type: SIMPLE_ASSIGN
z[428] = 0.028480000000000002
*/
void SpiralGalaxy_eqFunction_6833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6833};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2927]] /* z[428] STATE(1,vz[428]) */) = 0.028480000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12274(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12275(DATA *data, threadData_t *threadData);


/*
equation index: 6836
type: SIMPLE_ASSIGN
y[428] = r_init[428] * sin(theta[428] + 0.00712)
*/
void SpiralGalaxy_eqFunction_6836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6836};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2427]] /* y[428] STATE(1,vy[428]) */) = ((data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */)) * (sin((data->simulationInfo->realParameter[1934] /* theta[428] PARAM */) + 0.00712));
  TRACE_POP
}

/*
equation index: 6837
type: SIMPLE_ASSIGN
x[428] = r_init[428] * cos(theta[428] + 0.00712)
*/
void SpiralGalaxy_eqFunction_6837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1927]] /* x[428] STATE(1,vx[428]) */) = ((data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */)) * (cos((data->simulationInfo->realParameter[1934] /* theta[428] PARAM */) + 0.00712));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12276(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12277(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12280(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12279(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12278(DATA *data, threadData_t *threadData);


/*
equation index: 6843
type: SIMPLE_ASSIGN
vx[428] = (-sin(theta[428])) * r_init[428] * omega_c[428]
*/
void SpiralGalaxy_eqFunction_6843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6843};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[427]] /* vx[428] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1934] /* theta[428] PARAM */)))) * (((data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */)) * ((data->simulationInfo->realParameter[932] /* omega_c[428] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12273(DATA *data, threadData_t *threadData);


/*
equation index: 6845
type: SIMPLE_ASSIGN
vy[428] = cos(theta[428]) * r_init[428] * omega_c[428]
*/
void SpiralGalaxy_eqFunction_6845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[927]] /* vy[428] STATE(1) */) = (cos((data->simulationInfo->realParameter[1934] /* theta[428] PARAM */))) * (((data->simulationInfo->realParameter[1433] /* r_init[428] PARAM */)) * ((data->simulationInfo->realParameter[932] /* omega_c[428] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12272(DATA *data, threadData_t *threadData);


/*
equation index: 6847
type: SIMPLE_ASSIGN
vz[428] = 0.0
*/
void SpiralGalaxy_eqFunction_6847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6847};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1427]] /* vz[428] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12271(DATA *data, threadData_t *threadData);


/*
equation index: 6849
type: SIMPLE_ASSIGN
z[429] = 0.028640000000000002
*/
void SpiralGalaxy_eqFunction_6849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6849};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2928]] /* z[429] STATE(1,vz[429]) */) = 0.028640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12284(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12285(DATA *data, threadData_t *threadData);


/*
equation index: 6852
type: SIMPLE_ASSIGN
y[429] = r_init[429] * sin(theta[429] + 0.00716)
*/
void SpiralGalaxy_eqFunction_6852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6852};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2428]] /* y[429] STATE(1,vy[429]) */) = ((data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */)) * (sin((data->simulationInfo->realParameter[1935] /* theta[429] PARAM */) + 0.00716));
  TRACE_POP
}

/*
equation index: 6853
type: SIMPLE_ASSIGN
x[429] = r_init[429] * cos(theta[429] + 0.00716)
*/
void SpiralGalaxy_eqFunction_6853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1928]] /* x[429] STATE(1,vx[429]) */) = ((data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */)) * (cos((data->simulationInfo->realParameter[1935] /* theta[429] PARAM */) + 0.00716));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12286(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12287(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12290(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12289(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12288(DATA *data, threadData_t *threadData);


/*
equation index: 6859
type: SIMPLE_ASSIGN
vx[429] = (-sin(theta[429])) * r_init[429] * omega_c[429]
*/
void SpiralGalaxy_eqFunction_6859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6859};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[428]] /* vx[429] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1935] /* theta[429] PARAM */)))) * (((data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */)) * ((data->simulationInfo->realParameter[933] /* omega_c[429] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12283(DATA *data, threadData_t *threadData);


/*
equation index: 6861
type: SIMPLE_ASSIGN
vy[429] = cos(theta[429]) * r_init[429] * omega_c[429]
*/
void SpiralGalaxy_eqFunction_6861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[928]] /* vy[429] STATE(1) */) = (cos((data->simulationInfo->realParameter[1935] /* theta[429] PARAM */))) * (((data->simulationInfo->realParameter[1434] /* r_init[429] PARAM */)) * ((data->simulationInfo->realParameter[933] /* omega_c[429] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12282(DATA *data, threadData_t *threadData);


/*
equation index: 6863
type: SIMPLE_ASSIGN
vz[429] = 0.0
*/
void SpiralGalaxy_eqFunction_6863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6863};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1428]] /* vz[429] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12281(DATA *data, threadData_t *threadData);


/*
equation index: 6865
type: SIMPLE_ASSIGN
z[430] = 0.028800000000000003
*/
void SpiralGalaxy_eqFunction_6865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6865};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2929]] /* z[430] STATE(1,vz[430]) */) = 0.028800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12294(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12295(DATA *data, threadData_t *threadData);


/*
equation index: 6868
type: SIMPLE_ASSIGN
y[430] = r_init[430] * sin(theta[430] + 0.0072)
*/
void SpiralGalaxy_eqFunction_6868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6868};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2429]] /* y[430] STATE(1,vy[430]) */) = ((data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */)) * (sin((data->simulationInfo->realParameter[1936] /* theta[430] PARAM */) + 0.0072));
  TRACE_POP
}

/*
equation index: 6869
type: SIMPLE_ASSIGN
x[430] = r_init[430] * cos(theta[430] + 0.0072)
*/
void SpiralGalaxy_eqFunction_6869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1929]] /* x[430] STATE(1,vx[430]) */) = ((data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */)) * (cos((data->simulationInfo->realParameter[1936] /* theta[430] PARAM */) + 0.0072));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12296(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12297(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12300(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12299(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12298(DATA *data, threadData_t *threadData);


/*
equation index: 6875
type: SIMPLE_ASSIGN
vx[430] = (-sin(theta[430])) * r_init[430] * omega_c[430]
*/
void SpiralGalaxy_eqFunction_6875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6875};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[429]] /* vx[430] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1936] /* theta[430] PARAM */)))) * (((data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */)) * ((data->simulationInfo->realParameter[934] /* omega_c[430] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12293(DATA *data, threadData_t *threadData);


/*
equation index: 6877
type: SIMPLE_ASSIGN
vy[430] = cos(theta[430]) * r_init[430] * omega_c[430]
*/
void SpiralGalaxy_eqFunction_6877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[929]] /* vy[430] STATE(1) */) = (cos((data->simulationInfo->realParameter[1936] /* theta[430] PARAM */))) * (((data->simulationInfo->realParameter[1435] /* r_init[430] PARAM */)) * ((data->simulationInfo->realParameter[934] /* omega_c[430] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12292(DATA *data, threadData_t *threadData);


/*
equation index: 6879
type: SIMPLE_ASSIGN
vz[430] = 0.0
*/
void SpiralGalaxy_eqFunction_6879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6879};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1429]] /* vz[430] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12291(DATA *data, threadData_t *threadData);


/*
equation index: 6881
type: SIMPLE_ASSIGN
z[431] = 0.028960000000000007
*/
void SpiralGalaxy_eqFunction_6881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6881};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2930]] /* z[431] STATE(1,vz[431]) */) = 0.028960000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12304(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12305(DATA *data, threadData_t *threadData);


/*
equation index: 6884
type: SIMPLE_ASSIGN
y[431] = r_init[431] * sin(theta[431] + 0.00724)
*/
void SpiralGalaxy_eqFunction_6884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6884};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2430]] /* y[431] STATE(1,vy[431]) */) = ((data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */)) * (sin((data->simulationInfo->realParameter[1937] /* theta[431] PARAM */) + 0.00724));
  TRACE_POP
}

/*
equation index: 6885
type: SIMPLE_ASSIGN
x[431] = r_init[431] * cos(theta[431] + 0.00724)
*/
void SpiralGalaxy_eqFunction_6885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1930]] /* x[431] STATE(1,vx[431]) */) = ((data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */)) * (cos((data->simulationInfo->realParameter[1937] /* theta[431] PARAM */) + 0.00724));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12306(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12307(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12310(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12309(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12308(DATA *data, threadData_t *threadData);


/*
equation index: 6891
type: SIMPLE_ASSIGN
vx[431] = (-sin(theta[431])) * r_init[431] * omega_c[431]
*/
void SpiralGalaxy_eqFunction_6891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6891};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[430]] /* vx[431] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1937] /* theta[431] PARAM */)))) * (((data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */)) * ((data->simulationInfo->realParameter[935] /* omega_c[431] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12303(DATA *data, threadData_t *threadData);


/*
equation index: 6893
type: SIMPLE_ASSIGN
vy[431] = cos(theta[431]) * r_init[431] * omega_c[431]
*/
void SpiralGalaxy_eqFunction_6893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[930]] /* vy[431] STATE(1) */) = (cos((data->simulationInfo->realParameter[1937] /* theta[431] PARAM */))) * (((data->simulationInfo->realParameter[1436] /* r_init[431] PARAM */)) * ((data->simulationInfo->realParameter[935] /* omega_c[431] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12302(DATA *data, threadData_t *threadData);


/*
equation index: 6895
type: SIMPLE_ASSIGN
vz[431] = 0.0
*/
void SpiralGalaxy_eqFunction_6895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6895};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1430]] /* vz[431] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12301(DATA *data, threadData_t *threadData);


/*
equation index: 6897
type: SIMPLE_ASSIGN
z[432] = 0.029120000000000004
*/
void SpiralGalaxy_eqFunction_6897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6897};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2931]] /* z[432] STATE(1,vz[432]) */) = 0.029120000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12314(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12315(DATA *data, threadData_t *threadData);


/*
equation index: 6900
type: SIMPLE_ASSIGN
y[432] = r_init[432] * sin(theta[432] + 0.00728)
*/
void SpiralGalaxy_eqFunction_6900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6900};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2431]] /* y[432] STATE(1,vy[432]) */) = ((data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */)) * (sin((data->simulationInfo->realParameter[1938] /* theta[432] PARAM */) + 0.00728));
  TRACE_POP
}

/*
equation index: 6901
type: SIMPLE_ASSIGN
x[432] = r_init[432] * cos(theta[432] + 0.00728)
*/
void SpiralGalaxy_eqFunction_6901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1931]] /* x[432] STATE(1,vx[432]) */) = ((data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */)) * (cos((data->simulationInfo->realParameter[1938] /* theta[432] PARAM */) + 0.00728));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12316(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12317(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12320(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12319(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12318(DATA *data, threadData_t *threadData);


/*
equation index: 6907
type: SIMPLE_ASSIGN
vx[432] = (-sin(theta[432])) * r_init[432] * omega_c[432]
*/
void SpiralGalaxy_eqFunction_6907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6907};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[431]] /* vx[432] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1938] /* theta[432] PARAM */)))) * (((data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */)) * ((data->simulationInfo->realParameter[936] /* omega_c[432] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12313(DATA *data, threadData_t *threadData);


/*
equation index: 6909
type: SIMPLE_ASSIGN
vy[432] = cos(theta[432]) * r_init[432] * omega_c[432]
*/
void SpiralGalaxy_eqFunction_6909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[931]] /* vy[432] STATE(1) */) = (cos((data->simulationInfo->realParameter[1938] /* theta[432] PARAM */))) * (((data->simulationInfo->realParameter[1437] /* r_init[432] PARAM */)) * ((data->simulationInfo->realParameter[936] /* omega_c[432] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12312(DATA *data, threadData_t *threadData);


/*
equation index: 6911
type: SIMPLE_ASSIGN
vz[432] = 0.0
*/
void SpiralGalaxy_eqFunction_6911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6911};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1431]] /* vz[432] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12311(DATA *data, threadData_t *threadData);


/*
equation index: 6913
type: SIMPLE_ASSIGN
z[433] = 0.02928
*/
void SpiralGalaxy_eqFunction_6913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6913};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2932]] /* z[433] STATE(1,vz[433]) */) = 0.02928;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12324(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12325(DATA *data, threadData_t *threadData);


/*
equation index: 6916
type: SIMPLE_ASSIGN
y[433] = r_init[433] * sin(theta[433] + 0.00732)
*/
void SpiralGalaxy_eqFunction_6916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6916};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2432]] /* y[433] STATE(1,vy[433]) */) = ((data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */)) * (sin((data->simulationInfo->realParameter[1939] /* theta[433] PARAM */) + 0.00732));
  TRACE_POP
}

/*
equation index: 6917
type: SIMPLE_ASSIGN
x[433] = r_init[433] * cos(theta[433] + 0.00732)
*/
void SpiralGalaxy_eqFunction_6917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1932]] /* x[433] STATE(1,vx[433]) */) = ((data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */)) * (cos((data->simulationInfo->realParameter[1939] /* theta[433] PARAM */) + 0.00732));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12326(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12327(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12330(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12329(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12328(DATA *data, threadData_t *threadData);


/*
equation index: 6923
type: SIMPLE_ASSIGN
vx[433] = (-sin(theta[433])) * r_init[433] * omega_c[433]
*/
void SpiralGalaxy_eqFunction_6923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6923};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[432]] /* vx[433] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1939] /* theta[433] PARAM */)))) * (((data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */)) * ((data->simulationInfo->realParameter[937] /* omega_c[433] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12323(DATA *data, threadData_t *threadData);


/*
equation index: 6925
type: SIMPLE_ASSIGN
vy[433] = cos(theta[433]) * r_init[433] * omega_c[433]
*/
void SpiralGalaxy_eqFunction_6925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[932]] /* vy[433] STATE(1) */) = (cos((data->simulationInfo->realParameter[1939] /* theta[433] PARAM */))) * (((data->simulationInfo->realParameter[1438] /* r_init[433] PARAM */)) * ((data->simulationInfo->realParameter[937] /* omega_c[433] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12322(DATA *data, threadData_t *threadData);


/*
equation index: 6927
type: SIMPLE_ASSIGN
vz[433] = 0.0
*/
void SpiralGalaxy_eqFunction_6927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6927};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1432]] /* vz[433] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12321(DATA *data, threadData_t *threadData);


/*
equation index: 6929
type: SIMPLE_ASSIGN
z[434] = 0.02944
*/
void SpiralGalaxy_eqFunction_6929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6929};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2933]] /* z[434] STATE(1,vz[434]) */) = 0.02944;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12334(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12335(DATA *data, threadData_t *threadData);


/*
equation index: 6932
type: SIMPLE_ASSIGN
y[434] = r_init[434] * sin(theta[434] + 0.00736)
*/
void SpiralGalaxy_eqFunction_6932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6932};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2433]] /* y[434] STATE(1,vy[434]) */) = ((data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */)) * (sin((data->simulationInfo->realParameter[1940] /* theta[434] PARAM */) + 0.00736));
  TRACE_POP
}

/*
equation index: 6933
type: SIMPLE_ASSIGN
x[434] = r_init[434] * cos(theta[434] + 0.00736)
*/
void SpiralGalaxy_eqFunction_6933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1933]] /* x[434] STATE(1,vx[434]) */) = ((data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */)) * (cos((data->simulationInfo->realParameter[1940] /* theta[434] PARAM */) + 0.00736));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12336(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12337(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12340(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12339(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12338(DATA *data, threadData_t *threadData);


/*
equation index: 6939
type: SIMPLE_ASSIGN
vx[434] = (-sin(theta[434])) * r_init[434] * omega_c[434]
*/
void SpiralGalaxy_eqFunction_6939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6939};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[433]] /* vx[434] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1940] /* theta[434] PARAM */)))) * (((data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */)) * ((data->simulationInfo->realParameter[938] /* omega_c[434] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12333(DATA *data, threadData_t *threadData);


/*
equation index: 6941
type: SIMPLE_ASSIGN
vy[434] = cos(theta[434]) * r_init[434] * omega_c[434]
*/
void SpiralGalaxy_eqFunction_6941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[933]] /* vy[434] STATE(1) */) = (cos((data->simulationInfo->realParameter[1940] /* theta[434] PARAM */))) * (((data->simulationInfo->realParameter[1439] /* r_init[434] PARAM */)) * ((data->simulationInfo->realParameter[938] /* omega_c[434] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12332(DATA *data, threadData_t *threadData);


/*
equation index: 6943
type: SIMPLE_ASSIGN
vz[434] = 0.0
*/
void SpiralGalaxy_eqFunction_6943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6943};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1433]] /* vz[434] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12331(DATA *data, threadData_t *threadData);


/*
equation index: 6945
type: SIMPLE_ASSIGN
z[435] = 0.0296
*/
void SpiralGalaxy_eqFunction_6945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6945};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2934]] /* z[435] STATE(1,vz[435]) */) = 0.0296;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12344(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12345(DATA *data, threadData_t *threadData);


/*
equation index: 6948
type: SIMPLE_ASSIGN
y[435] = r_init[435] * sin(theta[435] + 0.0074)
*/
void SpiralGalaxy_eqFunction_6948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6948};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2434]] /* y[435] STATE(1,vy[435]) */) = ((data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */)) * (sin((data->simulationInfo->realParameter[1941] /* theta[435] PARAM */) + 0.0074));
  TRACE_POP
}

/*
equation index: 6949
type: SIMPLE_ASSIGN
x[435] = r_init[435] * cos(theta[435] + 0.0074)
*/
void SpiralGalaxy_eqFunction_6949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1934]] /* x[435] STATE(1,vx[435]) */) = ((data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */)) * (cos((data->simulationInfo->realParameter[1941] /* theta[435] PARAM */) + 0.0074));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12346(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12347(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12350(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12349(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12348(DATA *data, threadData_t *threadData);


/*
equation index: 6955
type: SIMPLE_ASSIGN
vx[435] = (-sin(theta[435])) * r_init[435] * omega_c[435]
*/
void SpiralGalaxy_eqFunction_6955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6955};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[434]] /* vx[435] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1941] /* theta[435] PARAM */)))) * (((data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */)) * ((data->simulationInfo->realParameter[939] /* omega_c[435] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12343(DATA *data, threadData_t *threadData);


/*
equation index: 6957
type: SIMPLE_ASSIGN
vy[435] = cos(theta[435]) * r_init[435] * omega_c[435]
*/
void SpiralGalaxy_eqFunction_6957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[934]] /* vy[435] STATE(1) */) = (cos((data->simulationInfo->realParameter[1941] /* theta[435] PARAM */))) * (((data->simulationInfo->realParameter[1440] /* r_init[435] PARAM */)) * ((data->simulationInfo->realParameter[939] /* omega_c[435] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12342(DATA *data, threadData_t *threadData);


/*
equation index: 6959
type: SIMPLE_ASSIGN
vz[435] = 0.0
*/
void SpiralGalaxy_eqFunction_6959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6959};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1434]] /* vz[435] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12341(DATA *data, threadData_t *threadData);


/*
equation index: 6961
type: SIMPLE_ASSIGN
z[436] = 0.029760000000000005
*/
void SpiralGalaxy_eqFunction_6961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6961};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2935]] /* z[436] STATE(1,vz[436]) */) = 0.029760000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12354(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12355(DATA *data, threadData_t *threadData);


/*
equation index: 6964
type: SIMPLE_ASSIGN
y[436] = r_init[436] * sin(theta[436] + 0.00744)
*/
void SpiralGalaxy_eqFunction_6964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6964};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2435]] /* y[436] STATE(1,vy[436]) */) = ((data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */)) * (sin((data->simulationInfo->realParameter[1942] /* theta[436] PARAM */) + 0.00744));
  TRACE_POP
}

/*
equation index: 6965
type: SIMPLE_ASSIGN
x[436] = r_init[436] * cos(theta[436] + 0.00744)
*/
void SpiralGalaxy_eqFunction_6965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1935]] /* x[436] STATE(1,vx[436]) */) = ((data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */)) * (cos((data->simulationInfo->realParameter[1942] /* theta[436] PARAM */) + 0.00744));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12356(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12357(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12360(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12359(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12358(DATA *data, threadData_t *threadData);


/*
equation index: 6971
type: SIMPLE_ASSIGN
vx[436] = (-sin(theta[436])) * r_init[436] * omega_c[436]
*/
void SpiralGalaxy_eqFunction_6971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6971};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[435]] /* vx[436] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1942] /* theta[436] PARAM */)))) * (((data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */)) * ((data->simulationInfo->realParameter[940] /* omega_c[436] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12353(DATA *data, threadData_t *threadData);


/*
equation index: 6973
type: SIMPLE_ASSIGN
vy[436] = cos(theta[436]) * r_init[436] * omega_c[436]
*/
void SpiralGalaxy_eqFunction_6973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[935]] /* vy[436] STATE(1) */) = (cos((data->simulationInfo->realParameter[1942] /* theta[436] PARAM */))) * (((data->simulationInfo->realParameter[1441] /* r_init[436] PARAM */)) * ((data->simulationInfo->realParameter[940] /* omega_c[436] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12352(DATA *data, threadData_t *threadData);


/*
equation index: 6975
type: SIMPLE_ASSIGN
vz[436] = 0.0
*/
void SpiralGalaxy_eqFunction_6975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6975};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1435]] /* vz[436] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12351(DATA *data, threadData_t *threadData);


/*
equation index: 6977
type: SIMPLE_ASSIGN
z[437] = 0.029920000000000002
*/
void SpiralGalaxy_eqFunction_6977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6977};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2936]] /* z[437] STATE(1,vz[437]) */) = 0.029920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12364(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12365(DATA *data, threadData_t *threadData);


/*
equation index: 6980
type: SIMPLE_ASSIGN
y[437] = r_init[437] * sin(theta[437] + 0.0074800000000000005)
*/
void SpiralGalaxy_eqFunction_6980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6980};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2436]] /* y[437] STATE(1,vy[437]) */) = ((data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */)) * (sin((data->simulationInfo->realParameter[1943] /* theta[437] PARAM */) + 0.0074800000000000005));
  TRACE_POP
}

/*
equation index: 6981
type: SIMPLE_ASSIGN
x[437] = r_init[437] * cos(theta[437] + 0.0074800000000000005)
*/
void SpiralGalaxy_eqFunction_6981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1936]] /* x[437] STATE(1,vx[437]) */) = ((data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */)) * (cos((data->simulationInfo->realParameter[1943] /* theta[437] PARAM */) + 0.0074800000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12366(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12367(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12370(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12369(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12368(DATA *data, threadData_t *threadData);


/*
equation index: 6987
type: SIMPLE_ASSIGN
vx[437] = (-sin(theta[437])) * r_init[437] * omega_c[437]
*/
void SpiralGalaxy_eqFunction_6987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6987};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[436]] /* vx[437] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1943] /* theta[437] PARAM */)))) * (((data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */)) * ((data->simulationInfo->realParameter[941] /* omega_c[437] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12363(DATA *data, threadData_t *threadData);


/*
equation index: 6989
type: SIMPLE_ASSIGN
vy[437] = cos(theta[437]) * r_init[437] * omega_c[437]
*/
void SpiralGalaxy_eqFunction_6989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[936]] /* vy[437] STATE(1) */) = (cos((data->simulationInfo->realParameter[1943] /* theta[437] PARAM */))) * (((data->simulationInfo->realParameter[1442] /* r_init[437] PARAM */)) * ((data->simulationInfo->realParameter[941] /* omega_c[437] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12362(DATA *data, threadData_t *threadData);


/*
equation index: 6991
type: SIMPLE_ASSIGN
vz[437] = 0.0
*/
void SpiralGalaxy_eqFunction_6991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6991};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1436]] /* vz[437] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12361(DATA *data, threadData_t *threadData);


/*
equation index: 6993
type: SIMPLE_ASSIGN
z[438] = 0.030080000000000003
*/
void SpiralGalaxy_eqFunction_6993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6993};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2937]] /* z[438] STATE(1,vz[438]) */) = 0.030080000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12374(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12375(DATA *data, threadData_t *threadData);


/*
equation index: 6996
type: SIMPLE_ASSIGN
y[438] = r_init[438] * sin(theta[438] + 0.00752)
*/
void SpiralGalaxy_eqFunction_6996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6996};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2437]] /* y[438] STATE(1,vy[438]) */) = ((data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */)) * (sin((data->simulationInfo->realParameter[1944] /* theta[438] PARAM */) + 0.00752));
  TRACE_POP
}

/*
equation index: 6997
type: SIMPLE_ASSIGN
x[438] = r_init[438] * cos(theta[438] + 0.00752)
*/
void SpiralGalaxy_eqFunction_6997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1937]] /* x[438] STATE(1,vx[438]) */) = ((data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */)) * (cos((data->simulationInfo->realParameter[1944] /* theta[438] PARAM */) + 0.00752));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12376(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12377(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12380(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_13(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_6501(data, threadData);
  SpiralGalaxy_eqFunction_12066(data, threadData);
  SpiralGalaxy_eqFunction_12067(data, threadData);
  SpiralGalaxy_eqFunction_12070(data, threadData);
  SpiralGalaxy_eqFunction_12069(data, threadData);
  SpiralGalaxy_eqFunction_12068(data, threadData);
  SpiralGalaxy_eqFunction_6507(data, threadData);
  SpiralGalaxy_eqFunction_12063(data, threadData);
  SpiralGalaxy_eqFunction_6509(data, threadData);
  SpiralGalaxy_eqFunction_12062(data, threadData);
  SpiralGalaxy_eqFunction_6511(data, threadData);
  SpiralGalaxy_eqFunction_12061(data, threadData);
  SpiralGalaxy_eqFunction_6513(data, threadData);
  SpiralGalaxy_eqFunction_12074(data, threadData);
  SpiralGalaxy_eqFunction_12075(data, threadData);
  SpiralGalaxy_eqFunction_6516(data, threadData);
  SpiralGalaxy_eqFunction_6517(data, threadData);
  SpiralGalaxy_eqFunction_12076(data, threadData);
  SpiralGalaxy_eqFunction_12077(data, threadData);
  SpiralGalaxy_eqFunction_12080(data, threadData);
  SpiralGalaxy_eqFunction_12079(data, threadData);
  SpiralGalaxy_eqFunction_12078(data, threadData);
  SpiralGalaxy_eqFunction_6523(data, threadData);
  SpiralGalaxy_eqFunction_12073(data, threadData);
  SpiralGalaxy_eqFunction_6525(data, threadData);
  SpiralGalaxy_eqFunction_12072(data, threadData);
  SpiralGalaxy_eqFunction_6527(data, threadData);
  SpiralGalaxy_eqFunction_12071(data, threadData);
  SpiralGalaxy_eqFunction_6529(data, threadData);
  SpiralGalaxy_eqFunction_12084(data, threadData);
  SpiralGalaxy_eqFunction_12085(data, threadData);
  SpiralGalaxy_eqFunction_6532(data, threadData);
  SpiralGalaxy_eqFunction_6533(data, threadData);
  SpiralGalaxy_eqFunction_12086(data, threadData);
  SpiralGalaxy_eqFunction_12087(data, threadData);
  SpiralGalaxy_eqFunction_12090(data, threadData);
  SpiralGalaxy_eqFunction_12089(data, threadData);
  SpiralGalaxy_eqFunction_12088(data, threadData);
  SpiralGalaxy_eqFunction_6539(data, threadData);
  SpiralGalaxy_eqFunction_12083(data, threadData);
  SpiralGalaxy_eqFunction_6541(data, threadData);
  SpiralGalaxy_eqFunction_12082(data, threadData);
  SpiralGalaxy_eqFunction_6543(data, threadData);
  SpiralGalaxy_eqFunction_12081(data, threadData);
  SpiralGalaxy_eqFunction_6545(data, threadData);
  SpiralGalaxy_eqFunction_12094(data, threadData);
  SpiralGalaxy_eqFunction_12095(data, threadData);
  SpiralGalaxy_eqFunction_6548(data, threadData);
  SpiralGalaxy_eqFunction_6549(data, threadData);
  SpiralGalaxy_eqFunction_12096(data, threadData);
  SpiralGalaxy_eqFunction_12097(data, threadData);
  SpiralGalaxy_eqFunction_12100(data, threadData);
  SpiralGalaxy_eqFunction_12099(data, threadData);
  SpiralGalaxy_eqFunction_12098(data, threadData);
  SpiralGalaxy_eqFunction_6555(data, threadData);
  SpiralGalaxy_eqFunction_12093(data, threadData);
  SpiralGalaxy_eqFunction_6557(data, threadData);
  SpiralGalaxy_eqFunction_12092(data, threadData);
  SpiralGalaxy_eqFunction_6559(data, threadData);
  SpiralGalaxy_eqFunction_12091(data, threadData);
  SpiralGalaxy_eqFunction_6561(data, threadData);
  SpiralGalaxy_eqFunction_12104(data, threadData);
  SpiralGalaxy_eqFunction_12105(data, threadData);
  SpiralGalaxy_eqFunction_6564(data, threadData);
  SpiralGalaxy_eqFunction_6565(data, threadData);
  SpiralGalaxy_eqFunction_12106(data, threadData);
  SpiralGalaxy_eqFunction_12107(data, threadData);
  SpiralGalaxy_eqFunction_12110(data, threadData);
  SpiralGalaxy_eqFunction_12109(data, threadData);
  SpiralGalaxy_eqFunction_12108(data, threadData);
  SpiralGalaxy_eqFunction_6571(data, threadData);
  SpiralGalaxy_eqFunction_12103(data, threadData);
  SpiralGalaxy_eqFunction_6573(data, threadData);
  SpiralGalaxy_eqFunction_12102(data, threadData);
  SpiralGalaxy_eqFunction_6575(data, threadData);
  SpiralGalaxy_eqFunction_12101(data, threadData);
  SpiralGalaxy_eqFunction_6577(data, threadData);
  SpiralGalaxy_eqFunction_12114(data, threadData);
  SpiralGalaxy_eqFunction_12115(data, threadData);
  SpiralGalaxy_eqFunction_6580(data, threadData);
  SpiralGalaxy_eqFunction_6581(data, threadData);
  SpiralGalaxy_eqFunction_12116(data, threadData);
  SpiralGalaxy_eqFunction_12117(data, threadData);
  SpiralGalaxy_eqFunction_12120(data, threadData);
  SpiralGalaxy_eqFunction_12119(data, threadData);
  SpiralGalaxy_eqFunction_12118(data, threadData);
  SpiralGalaxy_eqFunction_6587(data, threadData);
  SpiralGalaxy_eqFunction_12113(data, threadData);
  SpiralGalaxy_eqFunction_6589(data, threadData);
  SpiralGalaxy_eqFunction_12112(data, threadData);
  SpiralGalaxy_eqFunction_6591(data, threadData);
  SpiralGalaxy_eqFunction_12111(data, threadData);
  SpiralGalaxy_eqFunction_6593(data, threadData);
  SpiralGalaxy_eqFunction_12124(data, threadData);
  SpiralGalaxy_eqFunction_12125(data, threadData);
  SpiralGalaxy_eqFunction_6596(data, threadData);
  SpiralGalaxy_eqFunction_6597(data, threadData);
  SpiralGalaxy_eqFunction_12126(data, threadData);
  SpiralGalaxy_eqFunction_12127(data, threadData);
  SpiralGalaxy_eqFunction_12130(data, threadData);
  SpiralGalaxy_eqFunction_12129(data, threadData);
  SpiralGalaxy_eqFunction_12128(data, threadData);
  SpiralGalaxy_eqFunction_6603(data, threadData);
  SpiralGalaxy_eqFunction_12123(data, threadData);
  SpiralGalaxy_eqFunction_6605(data, threadData);
  SpiralGalaxy_eqFunction_12122(data, threadData);
  SpiralGalaxy_eqFunction_6607(data, threadData);
  SpiralGalaxy_eqFunction_12121(data, threadData);
  SpiralGalaxy_eqFunction_6609(data, threadData);
  SpiralGalaxy_eqFunction_12134(data, threadData);
  SpiralGalaxy_eqFunction_12135(data, threadData);
  SpiralGalaxy_eqFunction_6612(data, threadData);
  SpiralGalaxy_eqFunction_6613(data, threadData);
  SpiralGalaxy_eqFunction_12136(data, threadData);
  SpiralGalaxy_eqFunction_12137(data, threadData);
  SpiralGalaxy_eqFunction_12140(data, threadData);
  SpiralGalaxy_eqFunction_12139(data, threadData);
  SpiralGalaxy_eqFunction_12138(data, threadData);
  SpiralGalaxy_eqFunction_6619(data, threadData);
  SpiralGalaxy_eqFunction_12133(data, threadData);
  SpiralGalaxy_eqFunction_6621(data, threadData);
  SpiralGalaxy_eqFunction_12132(data, threadData);
  SpiralGalaxy_eqFunction_6623(data, threadData);
  SpiralGalaxy_eqFunction_12131(data, threadData);
  SpiralGalaxy_eqFunction_6625(data, threadData);
  SpiralGalaxy_eqFunction_12144(data, threadData);
  SpiralGalaxy_eqFunction_12145(data, threadData);
  SpiralGalaxy_eqFunction_6628(data, threadData);
  SpiralGalaxy_eqFunction_6629(data, threadData);
  SpiralGalaxy_eqFunction_12146(data, threadData);
  SpiralGalaxy_eqFunction_12147(data, threadData);
  SpiralGalaxy_eqFunction_12150(data, threadData);
  SpiralGalaxy_eqFunction_12149(data, threadData);
  SpiralGalaxy_eqFunction_12148(data, threadData);
  SpiralGalaxy_eqFunction_6635(data, threadData);
  SpiralGalaxy_eqFunction_12143(data, threadData);
  SpiralGalaxy_eqFunction_6637(data, threadData);
  SpiralGalaxy_eqFunction_12142(data, threadData);
  SpiralGalaxy_eqFunction_6639(data, threadData);
  SpiralGalaxy_eqFunction_12141(data, threadData);
  SpiralGalaxy_eqFunction_6641(data, threadData);
  SpiralGalaxy_eqFunction_12154(data, threadData);
  SpiralGalaxy_eqFunction_12155(data, threadData);
  SpiralGalaxy_eqFunction_6644(data, threadData);
  SpiralGalaxy_eqFunction_6645(data, threadData);
  SpiralGalaxy_eqFunction_12156(data, threadData);
  SpiralGalaxy_eqFunction_12157(data, threadData);
  SpiralGalaxy_eqFunction_12160(data, threadData);
  SpiralGalaxy_eqFunction_12159(data, threadData);
  SpiralGalaxy_eqFunction_12158(data, threadData);
  SpiralGalaxy_eqFunction_6651(data, threadData);
  SpiralGalaxy_eqFunction_12153(data, threadData);
  SpiralGalaxy_eqFunction_6653(data, threadData);
  SpiralGalaxy_eqFunction_12152(data, threadData);
  SpiralGalaxy_eqFunction_6655(data, threadData);
  SpiralGalaxy_eqFunction_12151(data, threadData);
  SpiralGalaxy_eqFunction_6657(data, threadData);
  SpiralGalaxy_eqFunction_12164(data, threadData);
  SpiralGalaxy_eqFunction_12165(data, threadData);
  SpiralGalaxy_eqFunction_6660(data, threadData);
  SpiralGalaxy_eqFunction_6661(data, threadData);
  SpiralGalaxy_eqFunction_12166(data, threadData);
  SpiralGalaxy_eqFunction_12167(data, threadData);
  SpiralGalaxy_eqFunction_12170(data, threadData);
  SpiralGalaxy_eqFunction_12169(data, threadData);
  SpiralGalaxy_eqFunction_12168(data, threadData);
  SpiralGalaxy_eqFunction_6667(data, threadData);
  SpiralGalaxy_eqFunction_12163(data, threadData);
  SpiralGalaxy_eqFunction_6669(data, threadData);
  SpiralGalaxy_eqFunction_12162(data, threadData);
  SpiralGalaxy_eqFunction_6671(data, threadData);
  SpiralGalaxy_eqFunction_12161(data, threadData);
  SpiralGalaxy_eqFunction_6673(data, threadData);
  SpiralGalaxy_eqFunction_12174(data, threadData);
  SpiralGalaxy_eqFunction_12175(data, threadData);
  SpiralGalaxy_eqFunction_6676(data, threadData);
  SpiralGalaxy_eqFunction_6677(data, threadData);
  SpiralGalaxy_eqFunction_12176(data, threadData);
  SpiralGalaxy_eqFunction_12177(data, threadData);
  SpiralGalaxy_eqFunction_12180(data, threadData);
  SpiralGalaxy_eqFunction_12179(data, threadData);
  SpiralGalaxy_eqFunction_12178(data, threadData);
  SpiralGalaxy_eqFunction_6683(data, threadData);
  SpiralGalaxy_eqFunction_12173(data, threadData);
  SpiralGalaxy_eqFunction_6685(data, threadData);
  SpiralGalaxy_eqFunction_12172(data, threadData);
  SpiralGalaxy_eqFunction_6687(data, threadData);
  SpiralGalaxy_eqFunction_12171(data, threadData);
  SpiralGalaxy_eqFunction_6689(data, threadData);
  SpiralGalaxy_eqFunction_12184(data, threadData);
  SpiralGalaxy_eqFunction_12185(data, threadData);
  SpiralGalaxy_eqFunction_6692(data, threadData);
  SpiralGalaxy_eqFunction_6693(data, threadData);
  SpiralGalaxy_eqFunction_12186(data, threadData);
  SpiralGalaxy_eqFunction_12187(data, threadData);
  SpiralGalaxy_eqFunction_12190(data, threadData);
  SpiralGalaxy_eqFunction_12189(data, threadData);
  SpiralGalaxy_eqFunction_12188(data, threadData);
  SpiralGalaxy_eqFunction_6699(data, threadData);
  SpiralGalaxy_eqFunction_12183(data, threadData);
  SpiralGalaxy_eqFunction_6701(data, threadData);
  SpiralGalaxy_eqFunction_12182(data, threadData);
  SpiralGalaxy_eqFunction_6703(data, threadData);
  SpiralGalaxy_eqFunction_12181(data, threadData);
  SpiralGalaxy_eqFunction_6705(data, threadData);
  SpiralGalaxy_eqFunction_12194(data, threadData);
  SpiralGalaxy_eqFunction_12195(data, threadData);
  SpiralGalaxy_eqFunction_6708(data, threadData);
  SpiralGalaxy_eqFunction_6709(data, threadData);
  SpiralGalaxy_eqFunction_12196(data, threadData);
  SpiralGalaxy_eqFunction_12197(data, threadData);
  SpiralGalaxy_eqFunction_12200(data, threadData);
  SpiralGalaxy_eqFunction_12199(data, threadData);
  SpiralGalaxy_eqFunction_12198(data, threadData);
  SpiralGalaxy_eqFunction_6715(data, threadData);
  SpiralGalaxy_eqFunction_12193(data, threadData);
  SpiralGalaxy_eqFunction_6717(data, threadData);
  SpiralGalaxy_eqFunction_12192(data, threadData);
  SpiralGalaxy_eqFunction_6719(data, threadData);
  SpiralGalaxy_eqFunction_12191(data, threadData);
  SpiralGalaxy_eqFunction_6721(data, threadData);
  SpiralGalaxy_eqFunction_12204(data, threadData);
  SpiralGalaxy_eqFunction_12205(data, threadData);
  SpiralGalaxy_eqFunction_6724(data, threadData);
  SpiralGalaxy_eqFunction_6725(data, threadData);
  SpiralGalaxy_eqFunction_12206(data, threadData);
  SpiralGalaxy_eqFunction_12207(data, threadData);
  SpiralGalaxy_eqFunction_12210(data, threadData);
  SpiralGalaxy_eqFunction_12209(data, threadData);
  SpiralGalaxy_eqFunction_12208(data, threadData);
  SpiralGalaxy_eqFunction_6731(data, threadData);
  SpiralGalaxy_eqFunction_12203(data, threadData);
  SpiralGalaxy_eqFunction_6733(data, threadData);
  SpiralGalaxy_eqFunction_12202(data, threadData);
  SpiralGalaxy_eqFunction_6735(data, threadData);
  SpiralGalaxy_eqFunction_12201(data, threadData);
  SpiralGalaxy_eqFunction_6737(data, threadData);
  SpiralGalaxy_eqFunction_12214(data, threadData);
  SpiralGalaxy_eqFunction_12215(data, threadData);
  SpiralGalaxy_eqFunction_6740(data, threadData);
  SpiralGalaxy_eqFunction_6741(data, threadData);
  SpiralGalaxy_eqFunction_12216(data, threadData);
  SpiralGalaxy_eqFunction_12217(data, threadData);
  SpiralGalaxy_eqFunction_12220(data, threadData);
  SpiralGalaxy_eqFunction_12219(data, threadData);
  SpiralGalaxy_eqFunction_12218(data, threadData);
  SpiralGalaxy_eqFunction_6747(data, threadData);
  SpiralGalaxy_eqFunction_12213(data, threadData);
  SpiralGalaxy_eqFunction_6749(data, threadData);
  SpiralGalaxy_eqFunction_12212(data, threadData);
  SpiralGalaxy_eqFunction_6751(data, threadData);
  SpiralGalaxy_eqFunction_12211(data, threadData);
  SpiralGalaxy_eqFunction_6753(data, threadData);
  SpiralGalaxy_eqFunction_12224(data, threadData);
  SpiralGalaxy_eqFunction_12225(data, threadData);
  SpiralGalaxy_eqFunction_6756(data, threadData);
  SpiralGalaxy_eqFunction_6757(data, threadData);
  SpiralGalaxy_eqFunction_12226(data, threadData);
  SpiralGalaxy_eqFunction_12227(data, threadData);
  SpiralGalaxy_eqFunction_12230(data, threadData);
  SpiralGalaxy_eqFunction_12229(data, threadData);
  SpiralGalaxy_eqFunction_12228(data, threadData);
  SpiralGalaxy_eqFunction_6763(data, threadData);
  SpiralGalaxy_eqFunction_12223(data, threadData);
  SpiralGalaxy_eqFunction_6765(data, threadData);
  SpiralGalaxy_eqFunction_12222(data, threadData);
  SpiralGalaxy_eqFunction_6767(data, threadData);
  SpiralGalaxy_eqFunction_12221(data, threadData);
  SpiralGalaxy_eqFunction_6769(data, threadData);
  SpiralGalaxy_eqFunction_12234(data, threadData);
  SpiralGalaxy_eqFunction_12235(data, threadData);
  SpiralGalaxy_eqFunction_6772(data, threadData);
  SpiralGalaxy_eqFunction_6773(data, threadData);
  SpiralGalaxy_eqFunction_12236(data, threadData);
  SpiralGalaxy_eqFunction_12237(data, threadData);
  SpiralGalaxy_eqFunction_12240(data, threadData);
  SpiralGalaxy_eqFunction_12239(data, threadData);
  SpiralGalaxy_eqFunction_12238(data, threadData);
  SpiralGalaxy_eqFunction_6779(data, threadData);
  SpiralGalaxy_eqFunction_12233(data, threadData);
  SpiralGalaxy_eqFunction_6781(data, threadData);
  SpiralGalaxy_eqFunction_12232(data, threadData);
  SpiralGalaxy_eqFunction_6783(data, threadData);
  SpiralGalaxy_eqFunction_12231(data, threadData);
  SpiralGalaxy_eqFunction_6785(data, threadData);
  SpiralGalaxy_eqFunction_12244(data, threadData);
  SpiralGalaxy_eqFunction_12245(data, threadData);
  SpiralGalaxy_eqFunction_6788(data, threadData);
  SpiralGalaxy_eqFunction_6789(data, threadData);
  SpiralGalaxy_eqFunction_12246(data, threadData);
  SpiralGalaxy_eqFunction_12247(data, threadData);
  SpiralGalaxy_eqFunction_12250(data, threadData);
  SpiralGalaxy_eqFunction_12249(data, threadData);
  SpiralGalaxy_eqFunction_12248(data, threadData);
  SpiralGalaxy_eqFunction_6795(data, threadData);
  SpiralGalaxy_eqFunction_12243(data, threadData);
  SpiralGalaxy_eqFunction_6797(data, threadData);
  SpiralGalaxy_eqFunction_12242(data, threadData);
  SpiralGalaxy_eqFunction_6799(data, threadData);
  SpiralGalaxy_eqFunction_12241(data, threadData);
  SpiralGalaxy_eqFunction_6801(data, threadData);
  SpiralGalaxy_eqFunction_12254(data, threadData);
  SpiralGalaxy_eqFunction_12255(data, threadData);
  SpiralGalaxy_eqFunction_6804(data, threadData);
  SpiralGalaxy_eqFunction_6805(data, threadData);
  SpiralGalaxy_eqFunction_12256(data, threadData);
  SpiralGalaxy_eqFunction_12257(data, threadData);
  SpiralGalaxy_eqFunction_12260(data, threadData);
  SpiralGalaxy_eqFunction_12259(data, threadData);
  SpiralGalaxy_eqFunction_12258(data, threadData);
  SpiralGalaxy_eqFunction_6811(data, threadData);
  SpiralGalaxy_eqFunction_12253(data, threadData);
  SpiralGalaxy_eqFunction_6813(data, threadData);
  SpiralGalaxy_eqFunction_12252(data, threadData);
  SpiralGalaxy_eqFunction_6815(data, threadData);
  SpiralGalaxy_eqFunction_12251(data, threadData);
  SpiralGalaxy_eqFunction_6817(data, threadData);
  SpiralGalaxy_eqFunction_12264(data, threadData);
  SpiralGalaxy_eqFunction_12265(data, threadData);
  SpiralGalaxy_eqFunction_6820(data, threadData);
  SpiralGalaxy_eqFunction_6821(data, threadData);
  SpiralGalaxy_eqFunction_12266(data, threadData);
  SpiralGalaxy_eqFunction_12267(data, threadData);
  SpiralGalaxy_eqFunction_12270(data, threadData);
  SpiralGalaxy_eqFunction_12269(data, threadData);
  SpiralGalaxy_eqFunction_12268(data, threadData);
  SpiralGalaxy_eqFunction_6827(data, threadData);
  SpiralGalaxy_eqFunction_12263(data, threadData);
  SpiralGalaxy_eqFunction_6829(data, threadData);
  SpiralGalaxy_eqFunction_12262(data, threadData);
  SpiralGalaxy_eqFunction_6831(data, threadData);
  SpiralGalaxy_eqFunction_12261(data, threadData);
  SpiralGalaxy_eqFunction_6833(data, threadData);
  SpiralGalaxy_eqFunction_12274(data, threadData);
  SpiralGalaxy_eqFunction_12275(data, threadData);
  SpiralGalaxy_eqFunction_6836(data, threadData);
  SpiralGalaxy_eqFunction_6837(data, threadData);
  SpiralGalaxy_eqFunction_12276(data, threadData);
  SpiralGalaxy_eqFunction_12277(data, threadData);
  SpiralGalaxy_eqFunction_12280(data, threadData);
  SpiralGalaxy_eqFunction_12279(data, threadData);
  SpiralGalaxy_eqFunction_12278(data, threadData);
  SpiralGalaxy_eqFunction_6843(data, threadData);
  SpiralGalaxy_eqFunction_12273(data, threadData);
  SpiralGalaxy_eqFunction_6845(data, threadData);
  SpiralGalaxy_eqFunction_12272(data, threadData);
  SpiralGalaxy_eqFunction_6847(data, threadData);
  SpiralGalaxy_eqFunction_12271(data, threadData);
  SpiralGalaxy_eqFunction_6849(data, threadData);
  SpiralGalaxy_eqFunction_12284(data, threadData);
  SpiralGalaxy_eqFunction_12285(data, threadData);
  SpiralGalaxy_eqFunction_6852(data, threadData);
  SpiralGalaxy_eqFunction_6853(data, threadData);
  SpiralGalaxy_eqFunction_12286(data, threadData);
  SpiralGalaxy_eqFunction_12287(data, threadData);
  SpiralGalaxy_eqFunction_12290(data, threadData);
  SpiralGalaxy_eqFunction_12289(data, threadData);
  SpiralGalaxy_eqFunction_12288(data, threadData);
  SpiralGalaxy_eqFunction_6859(data, threadData);
  SpiralGalaxy_eqFunction_12283(data, threadData);
  SpiralGalaxy_eqFunction_6861(data, threadData);
  SpiralGalaxy_eqFunction_12282(data, threadData);
  SpiralGalaxy_eqFunction_6863(data, threadData);
  SpiralGalaxy_eqFunction_12281(data, threadData);
  SpiralGalaxy_eqFunction_6865(data, threadData);
  SpiralGalaxy_eqFunction_12294(data, threadData);
  SpiralGalaxy_eqFunction_12295(data, threadData);
  SpiralGalaxy_eqFunction_6868(data, threadData);
  SpiralGalaxy_eqFunction_6869(data, threadData);
  SpiralGalaxy_eqFunction_12296(data, threadData);
  SpiralGalaxy_eqFunction_12297(data, threadData);
  SpiralGalaxy_eqFunction_12300(data, threadData);
  SpiralGalaxy_eqFunction_12299(data, threadData);
  SpiralGalaxy_eqFunction_12298(data, threadData);
  SpiralGalaxy_eqFunction_6875(data, threadData);
  SpiralGalaxy_eqFunction_12293(data, threadData);
  SpiralGalaxy_eqFunction_6877(data, threadData);
  SpiralGalaxy_eqFunction_12292(data, threadData);
  SpiralGalaxy_eqFunction_6879(data, threadData);
  SpiralGalaxy_eqFunction_12291(data, threadData);
  SpiralGalaxy_eqFunction_6881(data, threadData);
  SpiralGalaxy_eqFunction_12304(data, threadData);
  SpiralGalaxy_eqFunction_12305(data, threadData);
  SpiralGalaxy_eqFunction_6884(data, threadData);
  SpiralGalaxy_eqFunction_6885(data, threadData);
  SpiralGalaxy_eqFunction_12306(data, threadData);
  SpiralGalaxy_eqFunction_12307(data, threadData);
  SpiralGalaxy_eqFunction_12310(data, threadData);
  SpiralGalaxy_eqFunction_12309(data, threadData);
  SpiralGalaxy_eqFunction_12308(data, threadData);
  SpiralGalaxy_eqFunction_6891(data, threadData);
  SpiralGalaxy_eqFunction_12303(data, threadData);
  SpiralGalaxy_eqFunction_6893(data, threadData);
  SpiralGalaxy_eqFunction_12302(data, threadData);
  SpiralGalaxy_eqFunction_6895(data, threadData);
  SpiralGalaxy_eqFunction_12301(data, threadData);
  SpiralGalaxy_eqFunction_6897(data, threadData);
  SpiralGalaxy_eqFunction_12314(data, threadData);
  SpiralGalaxy_eqFunction_12315(data, threadData);
  SpiralGalaxy_eqFunction_6900(data, threadData);
  SpiralGalaxy_eqFunction_6901(data, threadData);
  SpiralGalaxy_eqFunction_12316(data, threadData);
  SpiralGalaxy_eqFunction_12317(data, threadData);
  SpiralGalaxy_eqFunction_12320(data, threadData);
  SpiralGalaxy_eqFunction_12319(data, threadData);
  SpiralGalaxy_eqFunction_12318(data, threadData);
  SpiralGalaxy_eqFunction_6907(data, threadData);
  SpiralGalaxy_eqFunction_12313(data, threadData);
  SpiralGalaxy_eqFunction_6909(data, threadData);
  SpiralGalaxy_eqFunction_12312(data, threadData);
  SpiralGalaxy_eqFunction_6911(data, threadData);
  SpiralGalaxy_eqFunction_12311(data, threadData);
  SpiralGalaxy_eqFunction_6913(data, threadData);
  SpiralGalaxy_eqFunction_12324(data, threadData);
  SpiralGalaxy_eqFunction_12325(data, threadData);
  SpiralGalaxy_eqFunction_6916(data, threadData);
  SpiralGalaxy_eqFunction_6917(data, threadData);
  SpiralGalaxy_eqFunction_12326(data, threadData);
  SpiralGalaxy_eqFunction_12327(data, threadData);
  SpiralGalaxy_eqFunction_12330(data, threadData);
  SpiralGalaxy_eqFunction_12329(data, threadData);
  SpiralGalaxy_eqFunction_12328(data, threadData);
  SpiralGalaxy_eqFunction_6923(data, threadData);
  SpiralGalaxy_eqFunction_12323(data, threadData);
  SpiralGalaxy_eqFunction_6925(data, threadData);
  SpiralGalaxy_eqFunction_12322(data, threadData);
  SpiralGalaxy_eqFunction_6927(data, threadData);
  SpiralGalaxy_eqFunction_12321(data, threadData);
  SpiralGalaxy_eqFunction_6929(data, threadData);
  SpiralGalaxy_eqFunction_12334(data, threadData);
  SpiralGalaxy_eqFunction_12335(data, threadData);
  SpiralGalaxy_eqFunction_6932(data, threadData);
  SpiralGalaxy_eqFunction_6933(data, threadData);
  SpiralGalaxy_eqFunction_12336(data, threadData);
  SpiralGalaxy_eqFunction_12337(data, threadData);
  SpiralGalaxy_eqFunction_12340(data, threadData);
  SpiralGalaxy_eqFunction_12339(data, threadData);
  SpiralGalaxy_eqFunction_12338(data, threadData);
  SpiralGalaxy_eqFunction_6939(data, threadData);
  SpiralGalaxy_eqFunction_12333(data, threadData);
  SpiralGalaxy_eqFunction_6941(data, threadData);
  SpiralGalaxy_eqFunction_12332(data, threadData);
  SpiralGalaxy_eqFunction_6943(data, threadData);
  SpiralGalaxy_eqFunction_12331(data, threadData);
  SpiralGalaxy_eqFunction_6945(data, threadData);
  SpiralGalaxy_eqFunction_12344(data, threadData);
  SpiralGalaxy_eqFunction_12345(data, threadData);
  SpiralGalaxy_eqFunction_6948(data, threadData);
  SpiralGalaxy_eqFunction_6949(data, threadData);
  SpiralGalaxy_eqFunction_12346(data, threadData);
  SpiralGalaxy_eqFunction_12347(data, threadData);
  SpiralGalaxy_eqFunction_12350(data, threadData);
  SpiralGalaxy_eqFunction_12349(data, threadData);
  SpiralGalaxy_eqFunction_12348(data, threadData);
  SpiralGalaxy_eqFunction_6955(data, threadData);
  SpiralGalaxy_eqFunction_12343(data, threadData);
  SpiralGalaxy_eqFunction_6957(data, threadData);
  SpiralGalaxy_eqFunction_12342(data, threadData);
  SpiralGalaxy_eqFunction_6959(data, threadData);
  SpiralGalaxy_eqFunction_12341(data, threadData);
  SpiralGalaxy_eqFunction_6961(data, threadData);
  SpiralGalaxy_eqFunction_12354(data, threadData);
  SpiralGalaxy_eqFunction_12355(data, threadData);
  SpiralGalaxy_eqFunction_6964(data, threadData);
  SpiralGalaxy_eqFunction_6965(data, threadData);
  SpiralGalaxy_eqFunction_12356(data, threadData);
  SpiralGalaxy_eqFunction_12357(data, threadData);
  SpiralGalaxy_eqFunction_12360(data, threadData);
  SpiralGalaxy_eqFunction_12359(data, threadData);
  SpiralGalaxy_eqFunction_12358(data, threadData);
  SpiralGalaxy_eqFunction_6971(data, threadData);
  SpiralGalaxy_eqFunction_12353(data, threadData);
  SpiralGalaxy_eqFunction_6973(data, threadData);
  SpiralGalaxy_eqFunction_12352(data, threadData);
  SpiralGalaxy_eqFunction_6975(data, threadData);
  SpiralGalaxy_eqFunction_12351(data, threadData);
  SpiralGalaxy_eqFunction_6977(data, threadData);
  SpiralGalaxy_eqFunction_12364(data, threadData);
  SpiralGalaxy_eqFunction_12365(data, threadData);
  SpiralGalaxy_eqFunction_6980(data, threadData);
  SpiralGalaxy_eqFunction_6981(data, threadData);
  SpiralGalaxy_eqFunction_12366(data, threadData);
  SpiralGalaxy_eqFunction_12367(data, threadData);
  SpiralGalaxy_eqFunction_12370(data, threadData);
  SpiralGalaxy_eqFunction_12369(data, threadData);
  SpiralGalaxy_eqFunction_12368(data, threadData);
  SpiralGalaxy_eqFunction_6987(data, threadData);
  SpiralGalaxy_eqFunction_12363(data, threadData);
  SpiralGalaxy_eqFunction_6989(data, threadData);
  SpiralGalaxy_eqFunction_12362(data, threadData);
  SpiralGalaxy_eqFunction_6991(data, threadData);
  SpiralGalaxy_eqFunction_12361(data, threadData);
  SpiralGalaxy_eqFunction_6993(data, threadData);
  SpiralGalaxy_eqFunction_12374(data, threadData);
  SpiralGalaxy_eqFunction_12375(data, threadData);
  SpiralGalaxy_eqFunction_6996(data, threadData);
  SpiralGalaxy_eqFunction_6997(data, threadData);
  SpiralGalaxy_eqFunction_12376(data, threadData);
  SpiralGalaxy_eqFunction_12377(data, threadData);
  SpiralGalaxy_eqFunction_12380(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif