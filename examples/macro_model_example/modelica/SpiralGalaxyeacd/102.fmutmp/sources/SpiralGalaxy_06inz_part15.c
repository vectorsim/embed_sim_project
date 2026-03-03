#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 7501
type: SIMPLE_ASSIGN
vy[469] = cos(theta[469]) * r_init[469] * omega_c[469]
*/
void SpiralGalaxy_eqFunction_7501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[968]] /* vy[469] STATE(1) */) = (cos((data->simulationInfo->realParameter[1975] /* theta[469] PARAM */))) * (((data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */)) * ((data->simulationInfo->realParameter[973] /* omega_c[469] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12682(DATA *data, threadData_t *threadData);


/*
equation index: 7503
type: SIMPLE_ASSIGN
vz[469] = 0.0
*/
void SpiralGalaxy_eqFunction_7503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7503};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1468]] /* vz[469] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12681(DATA *data, threadData_t *threadData);


/*
equation index: 7505
type: SIMPLE_ASSIGN
z[470] = 0.0352
*/
void SpiralGalaxy_eqFunction_7505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7505};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2969]] /* z[470] STATE(1,vz[470]) */) = 0.0352;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12694(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12695(DATA *data, threadData_t *threadData);


/*
equation index: 7508
type: SIMPLE_ASSIGN
y[470] = r_init[470] * sin(theta[470] + 0.008799999999999999)
*/
void SpiralGalaxy_eqFunction_7508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7508};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2469]] /* y[470] STATE(1,vy[470]) */) = ((data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */)) * (sin((data->simulationInfo->realParameter[1976] /* theta[470] PARAM */) + 0.008799999999999999));
  TRACE_POP
}

/*
equation index: 7509
type: SIMPLE_ASSIGN
x[470] = r_init[470] * cos(theta[470] + 0.008799999999999999)
*/
void SpiralGalaxy_eqFunction_7509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1969]] /* x[470] STATE(1,vx[470]) */) = ((data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */)) * (cos((data->simulationInfo->realParameter[1976] /* theta[470] PARAM */) + 0.008799999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12696(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12697(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12700(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12699(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12698(DATA *data, threadData_t *threadData);


/*
equation index: 7515
type: SIMPLE_ASSIGN
vx[470] = (-sin(theta[470])) * r_init[470] * omega_c[470]
*/
void SpiralGalaxy_eqFunction_7515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7515};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[469]] /* vx[470] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1976] /* theta[470] PARAM */)))) * (((data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */)) * ((data->simulationInfo->realParameter[974] /* omega_c[470] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12693(DATA *data, threadData_t *threadData);


/*
equation index: 7517
type: SIMPLE_ASSIGN
vy[470] = cos(theta[470]) * r_init[470] * omega_c[470]
*/
void SpiralGalaxy_eqFunction_7517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[969]] /* vy[470] STATE(1) */) = (cos((data->simulationInfo->realParameter[1976] /* theta[470] PARAM */))) * (((data->simulationInfo->realParameter[1475] /* r_init[470] PARAM */)) * ((data->simulationInfo->realParameter[974] /* omega_c[470] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12692(DATA *data, threadData_t *threadData);


/*
equation index: 7519
type: SIMPLE_ASSIGN
vz[470] = 0.0
*/
void SpiralGalaxy_eqFunction_7519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7519};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1469]] /* vz[470] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12691(DATA *data, threadData_t *threadData);


/*
equation index: 7521
type: SIMPLE_ASSIGN
z[471] = 0.03536
*/
void SpiralGalaxy_eqFunction_7521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7521};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2970]] /* z[471] STATE(1,vz[471]) */) = 0.03536;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12704(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12705(DATA *data, threadData_t *threadData);


/*
equation index: 7524
type: SIMPLE_ASSIGN
y[471] = r_init[471] * sin(theta[471] + 0.008839999999999999)
*/
void SpiralGalaxy_eqFunction_7524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7524};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2470]] /* y[471] STATE(1,vy[471]) */) = ((data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */)) * (sin((data->simulationInfo->realParameter[1977] /* theta[471] PARAM */) + 0.008839999999999999));
  TRACE_POP
}

/*
equation index: 7525
type: SIMPLE_ASSIGN
x[471] = r_init[471] * cos(theta[471] + 0.008839999999999999)
*/
void SpiralGalaxy_eqFunction_7525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1970]] /* x[471] STATE(1,vx[471]) */) = ((data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */)) * (cos((data->simulationInfo->realParameter[1977] /* theta[471] PARAM */) + 0.008839999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12706(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12707(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12710(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12709(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12708(DATA *data, threadData_t *threadData);


/*
equation index: 7531
type: SIMPLE_ASSIGN
vx[471] = (-sin(theta[471])) * r_init[471] * omega_c[471]
*/
void SpiralGalaxy_eqFunction_7531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7531};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[470]] /* vx[471] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1977] /* theta[471] PARAM */)))) * (((data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */)) * ((data->simulationInfo->realParameter[975] /* omega_c[471] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12703(DATA *data, threadData_t *threadData);


/*
equation index: 7533
type: SIMPLE_ASSIGN
vy[471] = cos(theta[471]) * r_init[471] * omega_c[471]
*/
void SpiralGalaxy_eqFunction_7533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[970]] /* vy[471] STATE(1) */) = (cos((data->simulationInfo->realParameter[1977] /* theta[471] PARAM */))) * (((data->simulationInfo->realParameter[1476] /* r_init[471] PARAM */)) * ((data->simulationInfo->realParameter[975] /* omega_c[471] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12702(DATA *data, threadData_t *threadData);


/*
equation index: 7535
type: SIMPLE_ASSIGN
vz[471] = 0.0
*/
void SpiralGalaxy_eqFunction_7535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7535};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1470]] /* vz[471] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12701(DATA *data, threadData_t *threadData);


/*
equation index: 7537
type: SIMPLE_ASSIGN
z[472] = 0.03552
*/
void SpiralGalaxy_eqFunction_7537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7537};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2971]] /* z[472] STATE(1,vz[472]) */) = 0.03552;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12714(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12715(DATA *data, threadData_t *threadData);


/*
equation index: 7540
type: SIMPLE_ASSIGN
y[472] = r_init[472] * sin(theta[472] + 0.008879999999999999)
*/
void SpiralGalaxy_eqFunction_7540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7540};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2471]] /* y[472] STATE(1,vy[472]) */) = ((data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */)) * (sin((data->simulationInfo->realParameter[1978] /* theta[472] PARAM */) + 0.008879999999999999));
  TRACE_POP
}

/*
equation index: 7541
type: SIMPLE_ASSIGN
x[472] = r_init[472] * cos(theta[472] + 0.008879999999999999)
*/
void SpiralGalaxy_eqFunction_7541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1971]] /* x[472] STATE(1,vx[472]) */) = ((data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */)) * (cos((data->simulationInfo->realParameter[1978] /* theta[472] PARAM */) + 0.008879999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12716(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12717(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12720(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12719(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12718(DATA *data, threadData_t *threadData);


/*
equation index: 7547
type: SIMPLE_ASSIGN
vx[472] = (-sin(theta[472])) * r_init[472] * omega_c[472]
*/
void SpiralGalaxy_eqFunction_7547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7547};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[471]] /* vx[472] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1978] /* theta[472] PARAM */)))) * (((data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */)) * ((data->simulationInfo->realParameter[976] /* omega_c[472] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12713(DATA *data, threadData_t *threadData);


/*
equation index: 7549
type: SIMPLE_ASSIGN
vy[472] = cos(theta[472]) * r_init[472] * omega_c[472]
*/
void SpiralGalaxy_eqFunction_7549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[971]] /* vy[472] STATE(1) */) = (cos((data->simulationInfo->realParameter[1978] /* theta[472] PARAM */))) * (((data->simulationInfo->realParameter[1477] /* r_init[472] PARAM */)) * ((data->simulationInfo->realParameter[976] /* omega_c[472] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12712(DATA *data, threadData_t *threadData);


/*
equation index: 7551
type: SIMPLE_ASSIGN
vz[472] = 0.0
*/
void SpiralGalaxy_eqFunction_7551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7551};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1471]] /* vz[472] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12711(DATA *data, threadData_t *threadData);


/*
equation index: 7553
type: SIMPLE_ASSIGN
z[473] = 0.03568
*/
void SpiralGalaxy_eqFunction_7553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7553};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2972]] /* z[473] STATE(1,vz[473]) */) = 0.03568;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12724(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12725(DATA *data, threadData_t *threadData);


/*
equation index: 7556
type: SIMPLE_ASSIGN
y[473] = r_init[473] * sin(theta[473] + 0.008919999999999999)
*/
void SpiralGalaxy_eqFunction_7556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7556};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2472]] /* y[473] STATE(1,vy[473]) */) = ((data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */)) * (sin((data->simulationInfo->realParameter[1979] /* theta[473] PARAM */) + 0.008919999999999999));
  TRACE_POP
}

/*
equation index: 7557
type: SIMPLE_ASSIGN
x[473] = r_init[473] * cos(theta[473] + 0.008919999999999999)
*/
void SpiralGalaxy_eqFunction_7557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1972]] /* x[473] STATE(1,vx[473]) */) = ((data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */)) * (cos((data->simulationInfo->realParameter[1979] /* theta[473] PARAM */) + 0.008919999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12726(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12727(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12730(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12729(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12728(DATA *data, threadData_t *threadData);


/*
equation index: 7563
type: SIMPLE_ASSIGN
vx[473] = (-sin(theta[473])) * r_init[473] * omega_c[473]
*/
void SpiralGalaxy_eqFunction_7563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7563};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[472]] /* vx[473] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1979] /* theta[473] PARAM */)))) * (((data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */)) * ((data->simulationInfo->realParameter[977] /* omega_c[473] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12723(DATA *data, threadData_t *threadData);


/*
equation index: 7565
type: SIMPLE_ASSIGN
vy[473] = cos(theta[473]) * r_init[473] * omega_c[473]
*/
void SpiralGalaxy_eqFunction_7565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[972]] /* vy[473] STATE(1) */) = (cos((data->simulationInfo->realParameter[1979] /* theta[473] PARAM */))) * (((data->simulationInfo->realParameter[1478] /* r_init[473] PARAM */)) * ((data->simulationInfo->realParameter[977] /* omega_c[473] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12722(DATA *data, threadData_t *threadData);


/*
equation index: 7567
type: SIMPLE_ASSIGN
vz[473] = 0.0
*/
void SpiralGalaxy_eqFunction_7567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7567};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1472]] /* vz[473] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12721(DATA *data, threadData_t *threadData);


/*
equation index: 7569
type: SIMPLE_ASSIGN
z[474] = 0.035840000000000004
*/
void SpiralGalaxy_eqFunction_7569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7569};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2973]] /* z[474] STATE(1,vz[474]) */) = 0.035840000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12734(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12735(DATA *data, threadData_t *threadData);


/*
equation index: 7572
type: SIMPLE_ASSIGN
y[474] = r_init[474] * sin(theta[474] + 0.00896)
*/
void SpiralGalaxy_eqFunction_7572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7572};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2473]] /* y[474] STATE(1,vy[474]) */) = ((data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */)) * (sin((data->simulationInfo->realParameter[1980] /* theta[474] PARAM */) + 0.00896));
  TRACE_POP
}

/*
equation index: 7573
type: SIMPLE_ASSIGN
x[474] = r_init[474] * cos(theta[474] + 0.00896)
*/
void SpiralGalaxy_eqFunction_7573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1973]] /* x[474] STATE(1,vx[474]) */) = ((data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */)) * (cos((data->simulationInfo->realParameter[1980] /* theta[474] PARAM */) + 0.00896));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12736(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12737(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12740(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12739(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12738(DATA *data, threadData_t *threadData);


/*
equation index: 7579
type: SIMPLE_ASSIGN
vx[474] = (-sin(theta[474])) * r_init[474] * omega_c[474]
*/
void SpiralGalaxy_eqFunction_7579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7579};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[473]] /* vx[474] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1980] /* theta[474] PARAM */)))) * (((data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */)) * ((data->simulationInfo->realParameter[978] /* omega_c[474] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12733(DATA *data, threadData_t *threadData);


/*
equation index: 7581
type: SIMPLE_ASSIGN
vy[474] = cos(theta[474]) * r_init[474] * omega_c[474]
*/
void SpiralGalaxy_eqFunction_7581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[973]] /* vy[474] STATE(1) */) = (cos((data->simulationInfo->realParameter[1980] /* theta[474] PARAM */))) * (((data->simulationInfo->realParameter[1479] /* r_init[474] PARAM */)) * ((data->simulationInfo->realParameter[978] /* omega_c[474] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12732(DATA *data, threadData_t *threadData);


/*
equation index: 7583
type: SIMPLE_ASSIGN
vz[474] = 0.0
*/
void SpiralGalaxy_eqFunction_7583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7583};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1473]] /* vz[474] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12731(DATA *data, threadData_t *threadData);


/*
equation index: 7585
type: SIMPLE_ASSIGN
z[475] = 0.036
*/
void SpiralGalaxy_eqFunction_7585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7585};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2974]] /* z[475] STATE(1,vz[475]) */) = 0.036;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12744(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12745(DATA *data, threadData_t *threadData);


/*
equation index: 7588
type: SIMPLE_ASSIGN
y[475] = r_init[475] * sin(theta[475] + 0.009)
*/
void SpiralGalaxy_eqFunction_7588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7588};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2474]] /* y[475] STATE(1,vy[475]) */) = ((data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */)) * (sin((data->simulationInfo->realParameter[1981] /* theta[475] PARAM */) + 0.009));
  TRACE_POP
}

/*
equation index: 7589
type: SIMPLE_ASSIGN
x[475] = r_init[475] * cos(theta[475] + 0.009)
*/
void SpiralGalaxy_eqFunction_7589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1974]] /* x[475] STATE(1,vx[475]) */) = ((data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */)) * (cos((data->simulationInfo->realParameter[1981] /* theta[475] PARAM */) + 0.009));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12746(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12747(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12750(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12749(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12748(DATA *data, threadData_t *threadData);


/*
equation index: 7595
type: SIMPLE_ASSIGN
vx[475] = (-sin(theta[475])) * r_init[475] * omega_c[475]
*/
void SpiralGalaxy_eqFunction_7595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7595};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[474]] /* vx[475] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1981] /* theta[475] PARAM */)))) * (((data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */)) * ((data->simulationInfo->realParameter[979] /* omega_c[475] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12743(DATA *data, threadData_t *threadData);


/*
equation index: 7597
type: SIMPLE_ASSIGN
vy[475] = cos(theta[475]) * r_init[475] * omega_c[475]
*/
void SpiralGalaxy_eqFunction_7597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[974]] /* vy[475] STATE(1) */) = (cos((data->simulationInfo->realParameter[1981] /* theta[475] PARAM */))) * (((data->simulationInfo->realParameter[1480] /* r_init[475] PARAM */)) * ((data->simulationInfo->realParameter[979] /* omega_c[475] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12742(DATA *data, threadData_t *threadData);


/*
equation index: 7599
type: SIMPLE_ASSIGN
vz[475] = 0.0
*/
void SpiralGalaxy_eqFunction_7599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7599};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1474]] /* vz[475] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12741(DATA *data, threadData_t *threadData);


/*
equation index: 7601
type: SIMPLE_ASSIGN
z[476] = 0.03616
*/
void SpiralGalaxy_eqFunction_7601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7601};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2975]] /* z[476] STATE(1,vz[476]) */) = 0.03616;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12754(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12755(DATA *data, threadData_t *threadData);


/*
equation index: 7604
type: SIMPLE_ASSIGN
y[476] = r_init[476] * sin(theta[476] + 0.00904)
*/
void SpiralGalaxy_eqFunction_7604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7604};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2475]] /* y[476] STATE(1,vy[476]) */) = ((data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */)) * (sin((data->simulationInfo->realParameter[1982] /* theta[476] PARAM */) + 0.00904));
  TRACE_POP
}

/*
equation index: 7605
type: SIMPLE_ASSIGN
x[476] = r_init[476] * cos(theta[476] + 0.00904)
*/
void SpiralGalaxy_eqFunction_7605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1975]] /* x[476] STATE(1,vx[476]) */) = ((data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */)) * (cos((data->simulationInfo->realParameter[1982] /* theta[476] PARAM */) + 0.00904));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12756(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12757(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12760(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12759(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12758(DATA *data, threadData_t *threadData);


/*
equation index: 7611
type: SIMPLE_ASSIGN
vx[476] = (-sin(theta[476])) * r_init[476] * omega_c[476]
*/
void SpiralGalaxy_eqFunction_7611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7611};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[475]] /* vx[476] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1982] /* theta[476] PARAM */)))) * (((data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */)) * ((data->simulationInfo->realParameter[980] /* omega_c[476] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12753(DATA *data, threadData_t *threadData);


/*
equation index: 7613
type: SIMPLE_ASSIGN
vy[476] = cos(theta[476]) * r_init[476] * omega_c[476]
*/
void SpiralGalaxy_eqFunction_7613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[975]] /* vy[476] STATE(1) */) = (cos((data->simulationInfo->realParameter[1982] /* theta[476] PARAM */))) * (((data->simulationInfo->realParameter[1481] /* r_init[476] PARAM */)) * ((data->simulationInfo->realParameter[980] /* omega_c[476] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12752(DATA *data, threadData_t *threadData);


/*
equation index: 7615
type: SIMPLE_ASSIGN
vz[476] = 0.0
*/
void SpiralGalaxy_eqFunction_7615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7615};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1475]] /* vz[476] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12751(DATA *data, threadData_t *threadData);


/*
equation index: 7617
type: SIMPLE_ASSIGN
z[477] = 0.036320000000000005
*/
void SpiralGalaxy_eqFunction_7617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7617};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2976]] /* z[477] STATE(1,vz[477]) */) = 0.036320000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12764(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12765(DATA *data, threadData_t *threadData);


/*
equation index: 7620
type: SIMPLE_ASSIGN
y[477] = r_init[477] * sin(theta[477] + 0.00908)
*/
void SpiralGalaxy_eqFunction_7620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7620};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2476]] /* y[477] STATE(1,vy[477]) */) = ((data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */)) * (sin((data->simulationInfo->realParameter[1983] /* theta[477] PARAM */) + 0.00908));
  TRACE_POP
}

/*
equation index: 7621
type: SIMPLE_ASSIGN
x[477] = r_init[477] * cos(theta[477] + 0.00908)
*/
void SpiralGalaxy_eqFunction_7621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1976]] /* x[477] STATE(1,vx[477]) */) = ((data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */)) * (cos((data->simulationInfo->realParameter[1983] /* theta[477] PARAM */) + 0.00908));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12766(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12767(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12770(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12769(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12768(DATA *data, threadData_t *threadData);


/*
equation index: 7627
type: SIMPLE_ASSIGN
vx[477] = (-sin(theta[477])) * r_init[477] * omega_c[477]
*/
void SpiralGalaxy_eqFunction_7627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7627};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[476]] /* vx[477] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1983] /* theta[477] PARAM */)))) * (((data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */)) * ((data->simulationInfo->realParameter[981] /* omega_c[477] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12763(DATA *data, threadData_t *threadData);


/*
equation index: 7629
type: SIMPLE_ASSIGN
vy[477] = cos(theta[477]) * r_init[477] * omega_c[477]
*/
void SpiralGalaxy_eqFunction_7629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[976]] /* vy[477] STATE(1) */) = (cos((data->simulationInfo->realParameter[1983] /* theta[477] PARAM */))) * (((data->simulationInfo->realParameter[1482] /* r_init[477] PARAM */)) * ((data->simulationInfo->realParameter[981] /* omega_c[477] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12762(DATA *data, threadData_t *threadData);


/*
equation index: 7631
type: SIMPLE_ASSIGN
vz[477] = 0.0
*/
void SpiralGalaxy_eqFunction_7631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7631};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1476]] /* vz[477] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12761(DATA *data, threadData_t *threadData);


/*
equation index: 7633
type: SIMPLE_ASSIGN
z[478] = 0.036480000000000005
*/
void SpiralGalaxy_eqFunction_7633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7633};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2977]] /* z[478] STATE(1,vz[478]) */) = 0.036480000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12774(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12775(DATA *data, threadData_t *threadData);


/*
equation index: 7636
type: SIMPLE_ASSIGN
y[478] = r_init[478] * sin(theta[478] + 0.00912)
*/
void SpiralGalaxy_eqFunction_7636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7636};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2477]] /* y[478] STATE(1,vy[478]) */) = ((data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */)) * (sin((data->simulationInfo->realParameter[1984] /* theta[478] PARAM */) + 0.00912));
  TRACE_POP
}

/*
equation index: 7637
type: SIMPLE_ASSIGN
x[478] = r_init[478] * cos(theta[478] + 0.00912)
*/
void SpiralGalaxy_eqFunction_7637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1977]] /* x[478] STATE(1,vx[478]) */) = ((data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */)) * (cos((data->simulationInfo->realParameter[1984] /* theta[478] PARAM */) + 0.00912));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12776(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12777(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12780(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12779(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12778(DATA *data, threadData_t *threadData);


/*
equation index: 7643
type: SIMPLE_ASSIGN
vx[478] = (-sin(theta[478])) * r_init[478] * omega_c[478]
*/
void SpiralGalaxy_eqFunction_7643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7643};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[477]] /* vx[478] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1984] /* theta[478] PARAM */)))) * (((data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */)) * ((data->simulationInfo->realParameter[982] /* omega_c[478] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12773(DATA *data, threadData_t *threadData);


/*
equation index: 7645
type: SIMPLE_ASSIGN
vy[478] = cos(theta[478]) * r_init[478] * omega_c[478]
*/
void SpiralGalaxy_eqFunction_7645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[977]] /* vy[478] STATE(1) */) = (cos((data->simulationInfo->realParameter[1984] /* theta[478] PARAM */))) * (((data->simulationInfo->realParameter[1483] /* r_init[478] PARAM */)) * ((data->simulationInfo->realParameter[982] /* omega_c[478] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12772(DATA *data, threadData_t *threadData);


/*
equation index: 7647
type: SIMPLE_ASSIGN
vz[478] = 0.0
*/
void SpiralGalaxy_eqFunction_7647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7647};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1477]] /* vz[478] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12771(DATA *data, threadData_t *threadData);


/*
equation index: 7649
type: SIMPLE_ASSIGN
z[479] = 0.036640000000000006
*/
void SpiralGalaxy_eqFunction_7649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7649};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2978]] /* z[479] STATE(1,vz[479]) */) = 0.036640000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12784(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12785(DATA *data, threadData_t *threadData);


/*
equation index: 7652
type: SIMPLE_ASSIGN
y[479] = r_init[479] * sin(theta[479] + 0.00916)
*/
void SpiralGalaxy_eqFunction_7652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7652};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2478]] /* y[479] STATE(1,vy[479]) */) = ((data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */)) * (sin((data->simulationInfo->realParameter[1985] /* theta[479] PARAM */) + 0.00916));
  TRACE_POP
}

/*
equation index: 7653
type: SIMPLE_ASSIGN
x[479] = r_init[479] * cos(theta[479] + 0.00916)
*/
void SpiralGalaxy_eqFunction_7653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1978]] /* x[479] STATE(1,vx[479]) */) = ((data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */)) * (cos((data->simulationInfo->realParameter[1985] /* theta[479] PARAM */) + 0.00916));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12786(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12787(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12790(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12789(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12788(DATA *data, threadData_t *threadData);


/*
equation index: 7659
type: SIMPLE_ASSIGN
vx[479] = (-sin(theta[479])) * r_init[479] * omega_c[479]
*/
void SpiralGalaxy_eqFunction_7659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7659};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[478]] /* vx[479] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1985] /* theta[479] PARAM */)))) * (((data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */)) * ((data->simulationInfo->realParameter[983] /* omega_c[479] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12783(DATA *data, threadData_t *threadData);


/*
equation index: 7661
type: SIMPLE_ASSIGN
vy[479] = cos(theta[479]) * r_init[479] * omega_c[479]
*/
void SpiralGalaxy_eqFunction_7661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[978]] /* vy[479] STATE(1) */) = (cos((data->simulationInfo->realParameter[1985] /* theta[479] PARAM */))) * (((data->simulationInfo->realParameter[1484] /* r_init[479] PARAM */)) * ((data->simulationInfo->realParameter[983] /* omega_c[479] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12782(DATA *data, threadData_t *threadData);


/*
equation index: 7663
type: SIMPLE_ASSIGN
vz[479] = 0.0
*/
void SpiralGalaxy_eqFunction_7663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7663};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1478]] /* vz[479] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12781(DATA *data, threadData_t *threadData);


/*
equation index: 7665
type: SIMPLE_ASSIGN
z[480] = 0.0368
*/
void SpiralGalaxy_eqFunction_7665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7665};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2979]] /* z[480] STATE(1,vz[480]) */) = 0.0368;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12794(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12795(DATA *data, threadData_t *threadData);


/*
equation index: 7668
type: SIMPLE_ASSIGN
y[480] = r_init[480] * sin(theta[480] + 0.0092)
*/
void SpiralGalaxy_eqFunction_7668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7668};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2479]] /* y[480] STATE(1,vy[480]) */) = ((data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */)) * (sin((data->simulationInfo->realParameter[1986] /* theta[480] PARAM */) + 0.0092));
  TRACE_POP
}

/*
equation index: 7669
type: SIMPLE_ASSIGN
x[480] = r_init[480] * cos(theta[480] + 0.0092)
*/
void SpiralGalaxy_eqFunction_7669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1979]] /* x[480] STATE(1,vx[480]) */) = ((data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */)) * (cos((data->simulationInfo->realParameter[1986] /* theta[480] PARAM */) + 0.0092));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12796(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12797(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12800(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12799(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12798(DATA *data, threadData_t *threadData);


/*
equation index: 7675
type: SIMPLE_ASSIGN
vx[480] = (-sin(theta[480])) * r_init[480] * omega_c[480]
*/
void SpiralGalaxy_eqFunction_7675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7675};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[479]] /* vx[480] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1986] /* theta[480] PARAM */)))) * (((data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */)) * ((data->simulationInfo->realParameter[984] /* omega_c[480] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12793(DATA *data, threadData_t *threadData);


/*
equation index: 7677
type: SIMPLE_ASSIGN
vy[480] = cos(theta[480]) * r_init[480] * omega_c[480]
*/
void SpiralGalaxy_eqFunction_7677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[979]] /* vy[480] STATE(1) */) = (cos((data->simulationInfo->realParameter[1986] /* theta[480] PARAM */))) * (((data->simulationInfo->realParameter[1485] /* r_init[480] PARAM */)) * ((data->simulationInfo->realParameter[984] /* omega_c[480] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12792(DATA *data, threadData_t *threadData);


/*
equation index: 7679
type: SIMPLE_ASSIGN
vz[480] = 0.0
*/
void SpiralGalaxy_eqFunction_7679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7679};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1479]] /* vz[480] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12791(DATA *data, threadData_t *threadData);


/*
equation index: 7681
type: SIMPLE_ASSIGN
z[481] = 0.03696
*/
void SpiralGalaxy_eqFunction_7681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7681};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2980]] /* z[481] STATE(1,vz[481]) */) = 0.03696;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12804(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12805(DATA *data, threadData_t *threadData);


/*
equation index: 7684
type: SIMPLE_ASSIGN
y[481] = r_init[481] * sin(theta[481] + 0.00924)
*/
void SpiralGalaxy_eqFunction_7684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7684};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2480]] /* y[481] STATE(1,vy[481]) */) = ((data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */)) * (sin((data->simulationInfo->realParameter[1987] /* theta[481] PARAM */) + 0.00924));
  TRACE_POP
}

/*
equation index: 7685
type: SIMPLE_ASSIGN
x[481] = r_init[481] * cos(theta[481] + 0.00924)
*/
void SpiralGalaxy_eqFunction_7685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1980]] /* x[481] STATE(1,vx[481]) */) = ((data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */)) * (cos((data->simulationInfo->realParameter[1987] /* theta[481] PARAM */) + 0.00924));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12806(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12807(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12810(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12809(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12808(DATA *data, threadData_t *threadData);


/*
equation index: 7691
type: SIMPLE_ASSIGN
vx[481] = (-sin(theta[481])) * r_init[481] * omega_c[481]
*/
void SpiralGalaxy_eqFunction_7691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7691};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[480]] /* vx[481] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1987] /* theta[481] PARAM */)))) * (((data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */)) * ((data->simulationInfo->realParameter[985] /* omega_c[481] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12803(DATA *data, threadData_t *threadData);


/*
equation index: 7693
type: SIMPLE_ASSIGN
vy[481] = cos(theta[481]) * r_init[481] * omega_c[481]
*/
void SpiralGalaxy_eqFunction_7693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[980]] /* vy[481] STATE(1) */) = (cos((data->simulationInfo->realParameter[1987] /* theta[481] PARAM */))) * (((data->simulationInfo->realParameter[1486] /* r_init[481] PARAM */)) * ((data->simulationInfo->realParameter[985] /* omega_c[481] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12802(DATA *data, threadData_t *threadData);


/*
equation index: 7695
type: SIMPLE_ASSIGN
vz[481] = 0.0
*/
void SpiralGalaxy_eqFunction_7695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7695};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1480]] /* vz[481] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12801(DATA *data, threadData_t *threadData);


/*
equation index: 7697
type: SIMPLE_ASSIGN
z[482] = 0.03712
*/
void SpiralGalaxy_eqFunction_7697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7697};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2981]] /* z[482] STATE(1,vz[482]) */) = 0.03712;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12814(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12815(DATA *data, threadData_t *threadData);


/*
equation index: 7700
type: SIMPLE_ASSIGN
y[482] = r_init[482] * sin(theta[482] + 0.00928)
*/
void SpiralGalaxy_eqFunction_7700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7700};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2481]] /* y[482] STATE(1,vy[482]) */) = ((data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */)) * (sin((data->simulationInfo->realParameter[1988] /* theta[482] PARAM */) + 0.00928));
  TRACE_POP
}

/*
equation index: 7701
type: SIMPLE_ASSIGN
x[482] = r_init[482] * cos(theta[482] + 0.00928)
*/
void SpiralGalaxy_eqFunction_7701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1981]] /* x[482] STATE(1,vx[482]) */) = ((data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */)) * (cos((data->simulationInfo->realParameter[1988] /* theta[482] PARAM */) + 0.00928));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12816(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12817(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12820(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12819(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12818(DATA *data, threadData_t *threadData);


/*
equation index: 7707
type: SIMPLE_ASSIGN
vx[482] = (-sin(theta[482])) * r_init[482] * omega_c[482]
*/
void SpiralGalaxy_eqFunction_7707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7707};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[481]] /* vx[482] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1988] /* theta[482] PARAM */)))) * (((data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */)) * ((data->simulationInfo->realParameter[986] /* omega_c[482] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12813(DATA *data, threadData_t *threadData);


/*
equation index: 7709
type: SIMPLE_ASSIGN
vy[482] = cos(theta[482]) * r_init[482] * omega_c[482]
*/
void SpiralGalaxy_eqFunction_7709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[981]] /* vy[482] STATE(1) */) = (cos((data->simulationInfo->realParameter[1988] /* theta[482] PARAM */))) * (((data->simulationInfo->realParameter[1487] /* r_init[482] PARAM */)) * ((data->simulationInfo->realParameter[986] /* omega_c[482] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12812(DATA *data, threadData_t *threadData);


/*
equation index: 7711
type: SIMPLE_ASSIGN
vz[482] = 0.0
*/
void SpiralGalaxy_eqFunction_7711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7711};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1481]] /* vz[482] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12811(DATA *data, threadData_t *threadData);


/*
equation index: 7713
type: SIMPLE_ASSIGN
z[483] = 0.03728
*/
void SpiralGalaxy_eqFunction_7713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7713};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2982]] /* z[483] STATE(1,vz[483]) */) = 0.03728;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12824(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12825(DATA *data, threadData_t *threadData);


/*
equation index: 7716
type: SIMPLE_ASSIGN
y[483] = r_init[483] * sin(theta[483] + 0.00932)
*/
void SpiralGalaxy_eqFunction_7716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7716};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2482]] /* y[483] STATE(1,vy[483]) */) = ((data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */)) * (sin((data->simulationInfo->realParameter[1989] /* theta[483] PARAM */) + 0.00932));
  TRACE_POP
}

/*
equation index: 7717
type: SIMPLE_ASSIGN
x[483] = r_init[483] * cos(theta[483] + 0.00932)
*/
void SpiralGalaxy_eqFunction_7717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1982]] /* x[483] STATE(1,vx[483]) */) = ((data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */)) * (cos((data->simulationInfo->realParameter[1989] /* theta[483] PARAM */) + 0.00932));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12826(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12827(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12830(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12829(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12828(DATA *data, threadData_t *threadData);


/*
equation index: 7723
type: SIMPLE_ASSIGN
vx[483] = (-sin(theta[483])) * r_init[483] * omega_c[483]
*/
void SpiralGalaxy_eqFunction_7723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7723};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[482]] /* vx[483] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1989] /* theta[483] PARAM */)))) * (((data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */)) * ((data->simulationInfo->realParameter[987] /* omega_c[483] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12823(DATA *data, threadData_t *threadData);


/*
equation index: 7725
type: SIMPLE_ASSIGN
vy[483] = cos(theta[483]) * r_init[483] * omega_c[483]
*/
void SpiralGalaxy_eqFunction_7725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[982]] /* vy[483] STATE(1) */) = (cos((data->simulationInfo->realParameter[1989] /* theta[483] PARAM */))) * (((data->simulationInfo->realParameter[1488] /* r_init[483] PARAM */)) * ((data->simulationInfo->realParameter[987] /* omega_c[483] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12822(DATA *data, threadData_t *threadData);


/*
equation index: 7727
type: SIMPLE_ASSIGN
vz[483] = 0.0
*/
void SpiralGalaxy_eqFunction_7727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7727};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1482]] /* vz[483] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12821(DATA *data, threadData_t *threadData);


/*
equation index: 7729
type: SIMPLE_ASSIGN
z[484] = 0.03744
*/
void SpiralGalaxy_eqFunction_7729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7729};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2983]] /* z[484] STATE(1,vz[484]) */) = 0.03744;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12834(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12835(DATA *data, threadData_t *threadData);


/*
equation index: 7732
type: SIMPLE_ASSIGN
y[484] = r_init[484] * sin(theta[484] + 0.00936)
*/
void SpiralGalaxy_eqFunction_7732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7732};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2483]] /* y[484] STATE(1,vy[484]) */) = ((data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */)) * (sin((data->simulationInfo->realParameter[1990] /* theta[484] PARAM */) + 0.00936));
  TRACE_POP
}

/*
equation index: 7733
type: SIMPLE_ASSIGN
x[484] = r_init[484] * cos(theta[484] + 0.00936)
*/
void SpiralGalaxy_eqFunction_7733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1983]] /* x[484] STATE(1,vx[484]) */) = ((data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */)) * (cos((data->simulationInfo->realParameter[1990] /* theta[484] PARAM */) + 0.00936));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12836(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12837(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12840(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12839(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12838(DATA *data, threadData_t *threadData);


/*
equation index: 7739
type: SIMPLE_ASSIGN
vx[484] = (-sin(theta[484])) * r_init[484] * omega_c[484]
*/
void SpiralGalaxy_eqFunction_7739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7739};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[483]] /* vx[484] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1990] /* theta[484] PARAM */)))) * (((data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */)) * ((data->simulationInfo->realParameter[988] /* omega_c[484] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12833(DATA *data, threadData_t *threadData);


/*
equation index: 7741
type: SIMPLE_ASSIGN
vy[484] = cos(theta[484]) * r_init[484] * omega_c[484]
*/
void SpiralGalaxy_eqFunction_7741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[983]] /* vy[484] STATE(1) */) = (cos((data->simulationInfo->realParameter[1990] /* theta[484] PARAM */))) * (((data->simulationInfo->realParameter[1489] /* r_init[484] PARAM */)) * ((data->simulationInfo->realParameter[988] /* omega_c[484] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12832(DATA *data, threadData_t *threadData);


/*
equation index: 7743
type: SIMPLE_ASSIGN
vz[484] = 0.0
*/
void SpiralGalaxy_eqFunction_7743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7743};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1483]] /* vz[484] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12831(DATA *data, threadData_t *threadData);


/*
equation index: 7745
type: SIMPLE_ASSIGN
z[485] = 0.03760000000000001
*/
void SpiralGalaxy_eqFunction_7745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7745};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2984]] /* z[485] STATE(1,vz[485]) */) = 0.03760000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12844(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12845(DATA *data, threadData_t *threadData);


/*
equation index: 7748
type: SIMPLE_ASSIGN
y[485] = r_init[485] * sin(theta[485] + 0.0094)
*/
void SpiralGalaxy_eqFunction_7748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7748};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2484]] /* y[485] STATE(1,vy[485]) */) = ((data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */)) * (sin((data->simulationInfo->realParameter[1991] /* theta[485] PARAM */) + 0.0094));
  TRACE_POP
}

/*
equation index: 7749
type: SIMPLE_ASSIGN
x[485] = r_init[485] * cos(theta[485] + 0.0094)
*/
void SpiralGalaxy_eqFunction_7749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1984]] /* x[485] STATE(1,vx[485]) */) = ((data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */)) * (cos((data->simulationInfo->realParameter[1991] /* theta[485] PARAM */) + 0.0094));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12846(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12847(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12850(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12849(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12848(DATA *data, threadData_t *threadData);


/*
equation index: 7755
type: SIMPLE_ASSIGN
vx[485] = (-sin(theta[485])) * r_init[485] * omega_c[485]
*/
void SpiralGalaxy_eqFunction_7755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7755};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[484]] /* vx[485] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1991] /* theta[485] PARAM */)))) * (((data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */)) * ((data->simulationInfo->realParameter[989] /* omega_c[485] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12843(DATA *data, threadData_t *threadData);


/*
equation index: 7757
type: SIMPLE_ASSIGN
vy[485] = cos(theta[485]) * r_init[485] * omega_c[485]
*/
void SpiralGalaxy_eqFunction_7757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[984]] /* vy[485] STATE(1) */) = (cos((data->simulationInfo->realParameter[1991] /* theta[485] PARAM */))) * (((data->simulationInfo->realParameter[1490] /* r_init[485] PARAM */)) * ((data->simulationInfo->realParameter[989] /* omega_c[485] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12842(DATA *data, threadData_t *threadData);


/*
equation index: 7759
type: SIMPLE_ASSIGN
vz[485] = 0.0
*/
void SpiralGalaxy_eqFunction_7759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7759};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1484]] /* vz[485] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12841(DATA *data, threadData_t *threadData);


/*
equation index: 7761
type: SIMPLE_ASSIGN
z[486] = 0.03776000000000001
*/
void SpiralGalaxy_eqFunction_7761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7761};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2985]] /* z[486] STATE(1,vz[486]) */) = 0.03776000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12854(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12855(DATA *data, threadData_t *threadData);


/*
equation index: 7764
type: SIMPLE_ASSIGN
y[486] = r_init[486] * sin(theta[486] + 0.00944)
*/
void SpiralGalaxy_eqFunction_7764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7764};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2485]] /* y[486] STATE(1,vy[486]) */) = ((data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */)) * (sin((data->simulationInfo->realParameter[1992] /* theta[486] PARAM */) + 0.00944));
  TRACE_POP
}

/*
equation index: 7765
type: SIMPLE_ASSIGN
x[486] = r_init[486] * cos(theta[486] + 0.00944)
*/
void SpiralGalaxy_eqFunction_7765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1985]] /* x[486] STATE(1,vx[486]) */) = ((data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */)) * (cos((data->simulationInfo->realParameter[1992] /* theta[486] PARAM */) + 0.00944));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12856(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12857(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12860(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12859(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12858(DATA *data, threadData_t *threadData);


/*
equation index: 7771
type: SIMPLE_ASSIGN
vx[486] = (-sin(theta[486])) * r_init[486] * omega_c[486]
*/
void SpiralGalaxy_eqFunction_7771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7771};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[485]] /* vx[486] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1992] /* theta[486] PARAM */)))) * (((data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */)) * ((data->simulationInfo->realParameter[990] /* omega_c[486] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12853(DATA *data, threadData_t *threadData);


/*
equation index: 7773
type: SIMPLE_ASSIGN
vy[486] = cos(theta[486]) * r_init[486] * omega_c[486]
*/
void SpiralGalaxy_eqFunction_7773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[985]] /* vy[486] STATE(1) */) = (cos((data->simulationInfo->realParameter[1992] /* theta[486] PARAM */))) * (((data->simulationInfo->realParameter[1491] /* r_init[486] PARAM */)) * ((data->simulationInfo->realParameter[990] /* omega_c[486] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12852(DATA *data, threadData_t *threadData);


/*
equation index: 7775
type: SIMPLE_ASSIGN
vz[486] = 0.0
*/
void SpiralGalaxy_eqFunction_7775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7775};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1485]] /* vz[486] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12851(DATA *data, threadData_t *threadData);


/*
equation index: 7777
type: SIMPLE_ASSIGN
z[487] = 0.03792
*/
void SpiralGalaxy_eqFunction_7777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7777};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2986]] /* z[487] STATE(1,vz[487]) */) = 0.03792;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12864(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12865(DATA *data, threadData_t *threadData);


/*
equation index: 7780
type: SIMPLE_ASSIGN
y[487] = r_init[487] * sin(theta[487] + 0.00948)
*/
void SpiralGalaxy_eqFunction_7780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7780};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2486]] /* y[487] STATE(1,vy[487]) */) = ((data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */)) * (sin((data->simulationInfo->realParameter[1993] /* theta[487] PARAM */) + 0.00948));
  TRACE_POP
}

/*
equation index: 7781
type: SIMPLE_ASSIGN
x[487] = r_init[487] * cos(theta[487] + 0.00948)
*/
void SpiralGalaxy_eqFunction_7781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1986]] /* x[487] STATE(1,vx[487]) */) = ((data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */)) * (cos((data->simulationInfo->realParameter[1993] /* theta[487] PARAM */) + 0.00948));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12866(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12867(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12870(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12869(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12868(DATA *data, threadData_t *threadData);


/*
equation index: 7787
type: SIMPLE_ASSIGN
vx[487] = (-sin(theta[487])) * r_init[487] * omega_c[487]
*/
void SpiralGalaxy_eqFunction_7787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7787};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[486]] /* vx[487] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1993] /* theta[487] PARAM */)))) * (((data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */)) * ((data->simulationInfo->realParameter[991] /* omega_c[487] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12863(DATA *data, threadData_t *threadData);


/*
equation index: 7789
type: SIMPLE_ASSIGN
vy[487] = cos(theta[487]) * r_init[487] * omega_c[487]
*/
void SpiralGalaxy_eqFunction_7789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[986]] /* vy[487] STATE(1) */) = (cos((data->simulationInfo->realParameter[1993] /* theta[487] PARAM */))) * (((data->simulationInfo->realParameter[1492] /* r_init[487] PARAM */)) * ((data->simulationInfo->realParameter[991] /* omega_c[487] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12862(DATA *data, threadData_t *threadData);


/*
equation index: 7791
type: SIMPLE_ASSIGN
vz[487] = 0.0
*/
void SpiralGalaxy_eqFunction_7791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7791};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1486]] /* vz[487] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12861(DATA *data, threadData_t *threadData);


/*
equation index: 7793
type: SIMPLE_ASSIGN
z[488] = 0.03808
*/
void SpiralGalaxy_eqFunction_7793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7793};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2987]] /* z[488] STATE(1,vz[488]) */) = 0.03808;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12874(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12875(DATA *data, threadData_t *threadData);


/*
equation index: 7796
type: SIMPLE_ASSIGN
y[488] = r_init[488] * sin(theta[488] + 0.009519999999999999)
*/
void SpiralGalaxy_eqFunction_7796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7796};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2487]] /* y[488] STATE(1,vy[488]) */) = ((data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */)) * (sin((data->simulationInfo->realParameter[1994] /* theta[488] PARAM */) + 0.009519999999999999));
  TRACE_POP
}

/*
equation index: 7797
type: SIMPLE_ASSIGN
x[488] = r_init[488] * cos(theta[488] + 0.009519999999999999)
*/
void SpiralGalaxy_eqFunction_7797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1987]] /* x[488] STATE(1,vx[488]) */) = ((data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */)) * (cos((data->simulationInfo->realParameter[1994] /* theta[488] PARAM */) + 0.009519999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12876(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12877(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12880(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12879(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12878(DATA *data, threadData_t *threadData);


/*
equation index: 7803
type: SIMPLE_ASSIGN
vx[488] = (-sin(theta[488])) * r_init[488] * omega_c[488]
*/
void SpiralGalaxy_eqFunction_7803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7803};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[487]] /* vx[488] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1994] /* theta[488] PARAM */)))) * (((data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */)) * ((data->simulationInfo->realParameter[992] /* omega_c[488] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12873(DATA *data, threadData_t *threadData);


/*
equation index: 7805
type: SIMPLE_ASSIGN
vy[488] = cos(theta[488]) * r_init[488] * omega_c[488]
*/
void SpiralGalaxy_eqFunction_7805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[987]] /* vy[488] STATE(1) */) = (cos((data->simulationInfo->realParameter[1994] /* theta[488] PARAM */))) * (((data->simulationInfo->realParameter[1493] /* r_init[488] PARAM */)) * ((data->simulationInfo->realParameter[992] /* omega_c[488] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12872(DATA *data, threadData_t *threadData);


/*
equation index: 7807
type: SIMPLE_ASSIGN
vz[488] = 0.0
*/
void SpiralGalaxy_eqFunction_7807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7807};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1487]] /* vz[488] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12871(DATA *data, threadData_t *threadData);


/*
equation index: 7809
type: SIMPLE_ASSIGN
z[489] = 0.03824
*/
void SpiralGalaxy_eqFunction_7809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7809};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2988]] /* z[489] STATE(1,vz[489]) */) = 0.03824;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12884(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12885(DATA *data, threadData_t *threadData);


/*
equation index: 7812
type: SIMPLE_ASSIGN
y[489] = r_init[489] * sin(theta[489] + 0.009559999999999999)
*/
void SpiralGalaxy_eqFunction_7812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7812};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2488]] /* y[489] STATE(1,vy[489]) */) = ((data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */)) * (sin((data->simulationInfo->realParameter[1995] /* theta[489] PARAM */) + 0.009559999999999999));
  TRACE_POP
}

/*
equation index: 7813
type: SIMPLE_ASSIGN
x[489] = r_init[489] * cos(theta[489] + 0.009559999999999999)
*/
void SpiralGalaxy_eqFunction_7813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1988]] /* x[489] STATE(1,vx[489]) */) = ((data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */)) * (cos((data->simulationInfo->realParameter[1995] /* theta[489] PARAM */) + 0.009559999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12886(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12887(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12890(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12889(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12888(DATA *data, threadData_t *threadData);


/*
equation index: 7819
type: SIMPLE_ASSIGN
vx[489] = (-sin(theta[489])) * r_init[489] * omega_c[489]
*/
void SpiralGalaxy_eqFunction_7819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7819};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[488]] /* vx[489] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1995] /* theta[489] PARAM */)))) * (((data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */)) * ((data->simulationInfo->realParameter[993] /* omega_c[489] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12883(DATA *data, threadData_t *threadData);


/*
equation index: 7821
type: SIMPLE_ASSIGN
vy[489] = cos(theta[489]) * r_init[489] * omega_c[489]
*/
void SpiralGalaxy_eqFunction_7821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[988]] /* vy[489] STATE(1) */) = (cos((data->simulationInfo->realParameter[1995] /* theta[489] PARAM */))) * (((data->simulationInfo->realParameter[1494] /* r_init[489] PARAM */)) * ((data->simulationInfo->realParameter[993] /* omega_c[489] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12882(DATA *data, threadData_t *threadData);


/*
equation index: 7823
type: SIMPLE_ASSIGN
vz[489] = 0.0
*/
void SpiralGalaxy_eqFunction_7823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7823};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1488]] /* vz[489] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12881(DATA *data, threadData_t *threadData);


/*
equation index: 7825
type: SIMPLE_ASSIGN
z[490] = 0.038400000000000004
*/
void SpiralGalaxy_eqFunction_7825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7825};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2989]] /* z[490] STATE(1,vz[490]) */) = 0.038400000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12894(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12895(DATA *data, threadData_t *threadData);


/*
equation index: 7828
type: SIMPLE_ASSIGN
y[490] = r_init[490] * sin(theta[490] + 0.0096)
*/
void SpiralGalaxy_eqFunction_7828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7828};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2489]] /* y[490] STATE(1,vy[490]) */) = ((data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */)) * (sin((data->simulationInfo->realParameter[1996] /* theta[490] PARAM */) + 0.0096));
  TRACE_POP
}

/*
equation index: 7829
type: SIMPLE_ASSIGN
x[490] = r_init[490] * cos(theta[490] + 0.0096)
*/
void SpiralGalaxy_eqFunction_7829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1989]] /* x[490] STATE(1,vx[490]) */) = ((data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */)) * (cos((data->simulationInfo->realParameter[1996] /* theta[490] PARAM */) + 0.0096));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12896(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12897(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12900(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12899(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12898(DATA *data, threadData_t *threadData);


/*
equation index: 7835
type: SIMPLE_ASSIGN
vx[490] = (-sin(theta[490])) * r_init[490] * omega_c[490]
*/
void SpiralGalaxy_eqFunction_7835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7835};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[489]] /* vx[490] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1996] /* theta[490] PARAM */)))) * (((data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */)) * ((data->simulationInfo->realParameter[994] /* omega_c[490] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12893(DATA *data, threadData_t *threadData);


/*
equation index: 7837
type: SIMPLE_ASSIGN
vy[490] = cos(theta[490]) * r_init[490] * omega_c[490]
*/
void SpiralGalaxy_eqFunction_7837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[989]] /* vy[490] STATE(1) */) = (cos((data->simulationInfo->realParameter[1996] /* theta[490] PARAM */))) * (((data->simulationInfo->realParameter[1495] /* r_init[490] PARAM */)) * ((data->simulationInfo->realParameter[994] /* omega_c[490] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12892(DATA *data, threadData_t *threadData);


/*
equation index: 7839
type: SIMPLE_ASSIGN
vz[490] = 0.0
*/
void SpiralGalaxy_eqFunction_7839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7839};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1489]] /* vz[490] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12891(DATA *data, threadData_t *threadData);


/*
equation index: 7841
type: SIMPLE_ASSIGN
z[491] = 0.03856
*/
void SpiralGalaxy_eqFunction_7841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7841};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2990]] /* z[491] STATE(1,vz[491]) */) = 0.03856;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12904(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12905(DATA *data, threadData_t *threadData);


/*
equation index: 7844
type: SIMPLE_ASSIGN
y[491] = r_init[491] * sin(theta[491] + 0.00964)
*/
void SpiralGalaxy_eqFunction_7844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7844};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2490]] /* y[491] STATE(1,vy[491]) */) = ((data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */)) * (sin((data->simulationInfo->realParameter[1997] /* theta[491] PARAM */) + 0.00964));
  TRACE_POP
}

/*
equation index: 7845
type: SIMPLE_ASSIGN
x[491] = r_init[491] * cos(theta[491] + 0.00964)
*/
void SpiralGalaxy_eqFunction_7845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1990]] /* x[491] STATE(1,vx[491]) */) = ((data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */)) * (cos((data->simulationInfo->realParameter[1997] /* theta[491] PARAM */) + 0.00964));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12906(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12907(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12910(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12909(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12908(DATA *data, threadData_t *threadData);


/*
equation index: 7851
type: SIMPLE_ASSIGN
vx[491] = (-sin(theta[491])) * r_init[491] * omega_c[491]
*/
void SpiralGalaxy_eqFunction_7851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7851};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[490]] /* vx[491] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1997] /* theta[491] PARAM */)))) * (((data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */)) * ((data->simulationInfo->realParameter[995] /* omega_c[491] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12903(DATA *data, threadData_t *threadData);


/*
equation index: 7853
type: SIMPLE_ASSIGN
vy[491] = cos(theta[491]) * r_init[491] * omega_c[491]
*/
void SpiralGalaxy_eqFunction_7853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[990]] /* vy[491] STATE(1) */) = (cos((data->simulationInfo->realParameter[1997] /* theta[491] PARAM */))) * (((data->simulationInfo->realParameter[1496] /* r_init[491] PARAM */)) * ((data->simulationInfo->realParameter[995] /* omega_c[491] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12902(DATA *data, threadData_t *threadData);


/*
equation index: 7855
type: SIMPLE_ASSIGN
vz[491] = 0.0
*/
void SpiralGalaxy_eqFunction_7855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7855};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1490]] /* vz[491] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12901(DATA *data, threadData_t *threadData);


/*
equation index: 7857
type: SIMPLE_ASSIGN
z[492] = 0.03872
*/
void SpiralGalaxy_eqFunction_7857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7857};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2991]] /* z[492] STATE(1,vz[492]) */) = 0.03872;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12914(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12915(DATA *data, threadData_t *threadData);


/*
equation index: 7860
type: SIMPLE_ASSIGN
y[492] = r_init[492] * sin(theta[492] + 0.00968)
*/
void SpiralGalaxy_eqFunction_7860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7860};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2491]] /* y[492] STATE(1,vy[492]) */) = ((data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */)) * (sin((data->simulationInfo->realParameter[1998] /* theta[492] PARAM */) + 0.00968));
  TRACE_POP
}

/*
equation index: 7861
type: SIMPLE_ASSIGN
x[492] = r_init[492] * cos(theta[492] + 0.00968)
*/
void SpiralGalaxy_eqFunction_7861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1991]] /* x[492] STATE(1,vx[492]) */) = ((data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */)) * (cos((data->simulationInfo->realParameter[1998] /* theta[492] PARAM */) + 0.00968));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12916(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12917(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12920(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12919(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12918(DATA *data, threadData_t *threadData);


/*
equation index: 7867
type: SIMPLE_ASSIGN
vx[492] = (-sin(theta[492])) * r_init[492] * omega_c[492]
*/
void SpiralGalaxy_eqFunction_7867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7867};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[491]] /* vx[492] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1998] /* theta[492] PARAM */)))) * (((data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */)) * ((data->simulationInfo->realParameter[996] /* omega_c[492] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12913(DATA *data, threadData_t *threadData);


/*
equation index: 7869
type: SIMPLE_ASSIGN
vy[492] = cos(theta[492]) * r_init[492] * omega_c[492]
*/
void SpiralGalaxy_eqFunction_7869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[991]] /* vy[492] STATE(1) */) = (cos((data->simulationInfo->realParameter[1998] /* theta[492] PARAM */))) * (((data->simulationInfo->realParameter[1497] /* r_init[492] PARAM */)) * ((data->simulationInfo->realParameter[996] /* omega_c[492] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12912(DATA *data, threadData_t *threadData);


/*
equation index: 7871
type: SIMPLE_ASSIGN
vz[492] = 0.0
*/
void SpiralGalaxy_eqFunction_7871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7871};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1491]] /* vz[492] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12911(DATA *data, threadData_t *threadData);


/*
equation index: 7873
type: SIMPLE_ASSIGN
z[493] = 0.03888
*/
void SpiralGalaxy_eqFunction_7873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7873};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2992]] /* z[493] STATE(1,vz[493]) */) = 0.03888;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12924(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12925(DATA *data, threadData_t *threadData);


/*
equation index: 7876
type: SIMPLE_ASSIGN
y[493] = r_init[493] * sin(theta[493] + 0.00972)
*/
void SpiralGalaxy_eqFunction_7876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7876};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2492]] /* y[493] STATE(1,vy[493]) */) = ((data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */)) * (sin((data->simulationInfo->realParameter[1999] /* theta[493] PARAM */) + 0.00972));
  TRACE_POP
}

/*
equation index: 7877
type: SIMPLE_ASSIGN
x[493] = r_init[493] * cos(theta[493] + 0.00972)
*/
void SpiralGalaxy_eqFunction_7877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1992]] /* x[493] STATE(1,vx[493]) */) = ((data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */)) * (cos((data->simulationInfo->realParameter[1999] /* theta[493] PARAM */) + 0.00972));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12926(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12927(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12930(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12929(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12928(DATA *data, threadData_t *threadData);


/*
equation index: 7883
type: SIMPLE_ASSIGN
vx[493] = (-sin(theta[493])) * r_init[493] * omega_c[493]
*/
void SpiralGalaxy_eqFunction_7883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7883};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[492]] /* vx[493] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1999] /* theta[493] PARAM */)))) * (((data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */)) * ((data->simulationInfo->realParameter[997] /* omega_c[493] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12923(DATA *data, threadData_t *threadData);


/*
equation index: 7885
type: SIMPLE_ASSIGN
vy[493] = cos(theta[493]) * r_init[493] * omega_c[493]
*/
void SpiralGalaxy_eqFunction_7885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[992]] /* vy[493] STATE(1) */) = (cos((data->simulationInfo->realParameter[1999] /* theta[493] PARAM */))) * (((data->simulationInfo->realParameter[1498] /* r_init[493] PARAM */)) * ((data->simulationInfo->realParameter[997] /* omega_c[493] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12922(DATA *data, threadData_t *threadData);


/*
equation index: 7887
type: SIMPLE_ASSIGN
vz[493] = 0.0
*/
void SpiralGalaxy_eqFunction_7887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7887};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1492]] /* vz[493] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12921(DATA *data, threadData_t *threadData);


/*
equation index: 7889
type: SIMPLE_ASSIGN
z[494] = 0.039040000000000005
*/
void SpiralGalaxy_eqFunction_7889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7889};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2993]] /* z[494] STATE(1,vz[494]) */) = 0.039040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12934(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12935(DATA *data, threadData_t *threadData);


/*
equation index: 7892
type: SIMPLE_ASSIGN
y[494] = r_init[494] * sin(theta[494] + 0.00976)
*/
void SpiralGalaxy_eqFunction_7892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7892};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2493]] /* y[494] STATE(1,vy[494]) */) = ((data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */)) * (sin((data->simulationInfo->realParameter[2000] /* theta[494] PARAM */) + 0.00976));
  TRACE_POP
}

/*
equation index: 7893
type: SIMPLE_ASSIGN
x[494] = r_init[494] * cos(theta[494] + 0.00976)
*/
void SpiralGalaxy_eqFunction_7893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1993]] /* x[494] STATE(1,vx[494]) */) = ((data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */)) * (cos((data->simulationInfo->realParameter[2000] /* theta[494] PARAM */) + 0.00976));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12936(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12937(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12940(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12939(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12938(DATA *data, threadData_t *threadData);


/*
equation index: 7899
type: SIMPLE_ASSIGN
vx[494] = (-sin(theta[494])) * r_init[494] * omega_c[494]
*/
void SpiralGalaxy_eqFunction_7899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7899};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[493]] /* vx[494] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2000] /* theta[494] PARAM */)))) * (((data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */)) * ((data->simulationInfo->realParameter[998] /* omega_c[494] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12933(DATA *data, threadData_t *threadData);


/*
equation index: 7901
type: SIMPLE_ASSIGN
vy[494] = cos(theta[494]) * r_init[494] * omega_c[494]
*/
void SpiralGalaxy_eqFunction_7901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[993]] /* vy[494] STATE(1) */) = (cos((data->simulationInfo->realParameter[2000] /* theta[494] PARAM */))) * (((data->simulationInfo->realParameter[1499] /* r_init[494] PARAM */)) * ((data->simulationInfo->realParameter[998] /* omega_c[494] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12932(DATA *data, threadData_t *threadData);


/*
equation index: 7903
type: SIMPLE_ASSIGN
vz[494] = 0.0
*/
void SpiralGalaxy_eqFunction_7903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7903};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1493]] /* vz[494] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12931(DATA *data, threadData_t *threadData);


/*
equation index: 7905
type: SIMPLE_ASSIGN
z[495] = 0.039200000000000006
*/
void SpiralGalaxy_eqFunction_7905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7905};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2994]] /* z[495] STATE(1,vz[495]) */) = 0.039200000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12944(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12945(DATA *data, threadData_t *threadData);


/*
equation index: 7908
type: SIMPLE_ASSIGN
y[495] = r_init[495] * sin(theta[495] + 0.0098)
*/
void SpiralGalaxy_eqFunction_7908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7908};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2494]] /* y[495] STATE(1,vy[495]) */) = ((data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */)) * (sin((data->simulationInfo->realParameter[2001] /* theta[495] PARAM */) + 0.0098));
  TRACE_POP
}

/*
equation index: 7909
type: SIMPLE_ASSIGN
x[495] = r_init[495] * cos(theta[495] + 0.0098)
*/
void SpiralGalaxy_eqFunction_7909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1994]] /* x[495] STATE(1,vx[495]) */) = ((data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */)) * (cos((data->simulationInfo->realParameter[2001] /* theta[495] PARAM */) + 0.0098));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12946(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12947(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12950(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12949(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12948(DATA *data, threadData_t *threadData);


/*
equation index: 7915
type: SIMPLE_ASSIGN
vx[495] = (-sin(theta[495])) * r_init[495] * omega_c[495]
*/
void SpiralGalaxy_eqFunction_7915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7915};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[494]] /* vx[495] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2001] /* theta[495] PARAM */)))) * (((data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */)) * ((data->simulationInfo->realParameter[999] /* omega_c[495] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12943(DATA *data, threadData_t *threadData);


/*
equation index: 7917
type: SIMPLE_ASSIGN
vy[495] = cos(theta[495]) * r_init[495] * omega_c[495]
*/
void SpiralGalaxy_eqFunction_7917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[994]] /* vy[495] STATE(1) */) = (cos((data->simulationInfo->realParameter[2001] /* theta[495] PARAM */))) * (((data->simulationInfo->realParameter[1500] /* r_init[495] PARAM */)) * ((data->simulationInfo->realParameter[999] /* omega_c[495] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12942(DATA *data, threadData_t *threadData);


/*
equation index: 7919
type: SIMPLE_ASSIGN
vz[495] = 0.0
*/
void SpiralGalaxy_eqFunction_7919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7919};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1494]] /* vz[495] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12941(DATA *data, threadData_t *threadData);


/*
equation index: 7921
type: SIMPLE_ASSIGN
z[496] = 0.039360000000000006
*/
void SpiralGalaxy_eqFunction_7921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7921};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2995]] /* z[496] STATE(1,vz[496]) */) = 0.039360000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12954(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12955(DATA *data, threadData_t *threadData);


/*
equation index: 7924
type: SIMPLE_ASSIGN
y[496] = r_init[496] * sin(theta[496] + 0.00984)
*/
void SpiralGalaxy_eqFunction_7924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7924};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2495]] /* y[496] STATE(1,vy[496]) */) = ((data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */)) * (sin((data->simulationInfo->realParameter[2002] /* theta[496] PARAM */) + 0.00984));
  TRACE_POP
}

/*
equation index: 7925
type: SIMPLE_ASSIGN
x[496] = r_init[496] * cos(theta[496] + 0.00984)
*/
void SpiralGalaxy_eqFunction_7925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1995]] /* x[496] STATE(1,vx[496]) */) = ((data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */)) * (cos((data->simulationInfo->realParameter[2002] /* theta[496] PARAM */) + 0.00984));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12956(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12957(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12960(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12959(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12958(DATA *data, threadData_t *threadData);


/*
equation index: 7931
type: SIMPLE_ASSIGN
vx[496] = (-sin(theta[496])) * r_init[496] * omega_c[496]
*/
void SpiralGalaxy_eqFunction_7931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7931};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[495]] /* vx[496] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2002] /* theta[496] PARAM */)))) * (((data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */)) * ((data->simulationInfo->realParameter[1000] /* omega_c[496] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12953(DATA *data, threadData_t *threadData);


/*
equation index: 7933
type: SIMPLE_ASSIGN
vy[496] = cos(theta[496]) * r_init[496] * omega_c[496]
*/
void SpiralGalaxy_eqFunction_7933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[995]] /* vy[496] STATE(1) */) = (cos((data->simulationInfo->realParameter[2002] /* theta[496] PARAM */))) * (((data->simulationInfo->realParameter[1501] /* r_init[496] PARAM */)) * ((data->simulationInfo->realParameter[1000] /* omega_c[496] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12952(DATA *data, threadData_t *threadData);


/*
equation index: 7935
type: SIMPLE_ASSIGN
vz[496] = 0.0
*/
void SpiralGalaxy_eqFunction_7935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7935};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1495]] /* vz[496] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12951(DATA *data, threadData_t *threadData);


/*
equation index: 7937
type: SIMPLE_ASSIGN
z[497] = 0.039520000000000007
*/
void SpiralGalaxy_eqFunction_7937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7937};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2996]] /* z[497] STATE(1,vz[497]) */) = 0.039520000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12964(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12965(DATA *data, threadData_t *threadData);


/*
equation index: 7940
type: SIMPLE_ASSIGN
y[497] = r_init[497] * sin(theta[497] + 0.00988)
*/
void SpiralGalaxy_eqFunction_7940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7940};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2496]] /* y[497] STATE(1,vy[497]) */) = ((data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */)) * (sin((data->simulationInfo->realParameter[2003] /* theta[497] PARAM */) + 0.00988));
  TRACE_POP
}

/*
equation index: 7941
type: SIMPLE_ASSIGN
x[497] = r_init[497] * cos(theta[497] + 0.00988)
*/
void SpiralGalaxy_eqFunction_7941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1996]] /* x[497] STATE(1,vx[497]) */) = ((data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */)) * (cos((data->simulationInfo->realParameter[2003] /* theta[497] PARAM */) + 0.00988));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12966(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12967(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12970(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12969(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12968(DATA *data, threadData_t *threadData);


/*
equation index: 7947
type: SIMPLE_ASSIGN
vx[497] = (-sin(theta[497])) * r_init[497] * omega_c[497]
*/
void SpiralGalaxy_eqFunction_7947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7947};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[496]] /* vx[497] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2003] /* theta[497] PARAM */)))) * (((data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */)) * ((data->simulationInfo->realParameter[1001] /* omega_c[497] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12963(DATA *data, threadData_t *threadData);


/*
equation index: 7949
type: SIMPLE_ASSIGN
vy[497] = cos(theta[497]) * r_init[497] * omega_c[497]
*/
void SpiralGalaxy_eqFunction_7949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[996]] /* vy[497] STATE(1) */) = (cos((data->simulationInfo->realParameter[2003] /* theta[497] PARAM */))) * (((data->simulationInfo->realParameter[1502] /* r_init[497] PARAM */)) * ((data->simulationInfo->realParameter[1001] /* omega_c[497] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12962(DATA *data, threadData_t *threadData);


/*
equation index: 7951
type: SIMPLE_ASSIGN
vz[497] = 0.0
*/
void SpiralGalaxy_eqFunction_7951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7951};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1496]] /* vz[497] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12961(DATA *data, threadData_t *threadData);


/*
equation index: 7953
type: SIMPLE_ASSIGN
z[498] = 0.03968000000000001
*/
void SpiralGalaxy_eqFunction_7953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7953};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2997]] /* z[498] STATE(1,vz[498]) */) = 0.03968000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12974(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12975(DATA *data, threadData_t *threadData);


/*
equation index: 7956
type: SIMPLE_ASSIGN
y[498] = r_init[498] * sin(theta[498] + 0.00992)
*/
void SpiralGalaxy_eqFunction_7956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7956};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2497]] /* y[498] STATE(1,vy[498]) */) = ((data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */)) * (sin((data->simulationInfo->realParameter[2004] /* theta[498] PARAM */) + 0.00992));
  TRACE_POP
}

/*
equation index: 7957
type: SIMPLE_ASSIGN
x[498] = r_init[498] * cos(theta[498] + 0.00992)
*/
void SpiralGalaxy_eqFunction_7957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1997]] /* x[498] STATE(1,vx[498]) */) = ((data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */)) * (cos((data->simulationInfo->realParameter[2004] /* theta[498] PARAM */) + 0.00992));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12976(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12977(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12980(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12979(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12978(DATA *data, threadData_t *threadData);


/*
equation index: 7963
type: SIMPLE_ASSIGN
vx[498] = (-sin(theta[498])) * r_init[498] * omega_c[498]
*/
void SpiralGalaxy_eqFunction_7963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7963};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[497]] /* vx[498] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2004] /* theta[498] PARAM */)))) * (((data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */)) * ((data->simulationInfo->realParameter[1002] /* omega_c[498] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12973(DATA *data, threadData_t *threadData);


/*
equation index: 7965
type: SIMPLE_ASSIGN
vy[498] = cos(theta[498]) * r_init[498] * omega_c[498]
*/
void SpiralGalaxy_eqFunction_7965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[997]] /* vy[498] STATE(1) */) = (cos((data->simulationInfo->realParameter[2004] /* theta[498] PARAM */))) * (((data->simulationInfo->realParameter[1503] /* r_init[498] PARAM */)) * ((data->simulationInfo->realParameter[1002] /* omega_c[498] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12972(DATA *data, threadData_t *threadData);


/*
equation index: 7967
type: SIMPLE_ASSIGN
vz[498] = 0.0
*/
void SpiralGalaxy_eqFunction_7967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7967};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1497]] /* vz[498] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12971(DATA *data, threadData_t *threadData);


/*
equation index: 7969
type: SIMPLE_ASSIGN
z[499] = 0.03984000000000001
*/
void SpiralGalaxy_eqFunction_7969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7969};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2998]] /* z[499] STATE(1,vz[499]) */) = 0.03984000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12984(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12985(DATA *data, threadData_t *threadData);


/*
equation index: 7972
type: SIMPLE_ASSIGN
y[499] = r_init[499] * sin(theta[499] + 0.00996)
*/
void SpiralGalaxy_eqFunction_7972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7972};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2498]] /* y[499] STATE(1,vy[499]) */) = ((data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */)) * (sin((data->simulationInfo->realParameter[2005] /* theta[499] PARAM */) + 0.00996));
  TRACE_POP
}

/*
equation index: 7973
type: SIMPLE_ASSIGN
x[499] = r_init[499] * cos(theta[499] + 0.00996)
*/
void SpiralGalaxy_eqFunction_7973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1998]] /* x[499] STATE(1,vx[499]) */) = ((data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */)) * (cos((data->simulationInfo->realParameter[2005] /* theta[499] PARAM */) + 0.00996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12986(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12987(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12990(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12989(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12988(DATA *data, threadData_t *threadData);


/*
equation index: 7979
type: SIMPLE_ASSIGN
vx[499] = (-sin(theta[499])) * r_init[499] * omega_c[499]
*/
void SpiralGalaxy_eqFunction_7979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7979};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[498]] /* vx[499] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2005] /* theta[499] PARAM */)))) * (((data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */)) * ((data->simulationInfo->realParameter[1003] /* omega_c[499] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12983(DATA *data, threadData_t *threadData);


/*
equation index: 7981
type: SIMPLE_ASSIGN
vy[499] = cos(theta[499]) * r_init[499] * omega_c[499]
*/
void SpiralGalaxy_eqFunction_7981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[998]] /* vy[499] STATE(1) */) = (cos((data->simulationInfo->realParameter[2005] /* theta[499] PARAM */))) * (((data->simulationInfo->realParameter[1504] /* r_init[499] PARAM */)) * ((data->simulationInfo->realParameter[1003] /* omega_c[499] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12982(DATA *data, threadData_t *threadData);


/*
equation index: 7983
type: SIMPLE_ASSIGN
vz[499] = 0.0
*/
void SpiralGalaxy_eqFunction_7983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7983};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1498]] /* vz[499] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12981(DATA *data, threadData_t *threadData);


/*
equation index: 7985
type: SIMPLE_ASSIGN
z[500] = 0.04
*/
void SpiralGalaxy_eqFunction_7985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7985};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2999]] /* z[500] STATE(1,vz[500]) */) = 0.04;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12994(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12995(DATA *data, threadData_t *threadData);


/*
equation index: 7988
type: SIMPLE_ASSIGN
y[500] = r_init[500] * sin(theta[500] + 0.01)
*/
void SpiralGalaxy_eqFunction_7988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7988};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2499]] /* y[500] STATE(1,vy[500]) */) = ((data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */)) * (sin((data->simulationInfo->realParameter[2006] /* theta[500] PARAM */) + 0.01));
  TRACE_POP
}

/*
equation index: 7989
type: SIMPLE_ASSIGN
x[500] = r_init[500] * cos(theta[500] + 0.01)
*/
void SpiralGalaxy_eqFunction_7989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1999]] /* x[500] STATE(1,vx[500]) */) = ((data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */)) * (cos((data->simulationInfo->realParameter[2006] /* theta[500] PARAM */) + 0.01));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12996(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12997(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_13000(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12999(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12998(DATA *data, threadData_t *threadData);


/*
equation index: 7995
type: SIMPLE_ASSIGN
vx[500] = (-sin(theta[500])) * r_init[500] * omega_c[500]
*/
void SpiralGalaxy_eqFunction_7995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7995};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[499]] /* vx[500] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[2006] /* theta[500] PARAM */)))) * (((data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */)) * ((data->simulationInfo->realParameter[1004] /* omega_c[500] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12993(DATA *data, threadData_t *threadData);


/*
equation index: 7997
type: SIMPLE_ASSIGN
vy[500] = cos(theta[500]) * r_init[500] * omega_c[500]
*/
void SpiralGalaxy_eqFunction_7997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[999]] /* vy[500] STATE(1) */) = (cos((data->simulationInfo->realParameter[2006] /* theta[500] PARAM */))) * (((data->simulationInfo->realParameter[1505] /* r_init[500] PARAM */)) * ((data->simulationInfo->realParameter[1004] /* omega_c[500] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12992(DATA *data, threadData_t *threadData);


/*
equation index: 7999
type: SIMPLE_ASSIGN
vz[500] = 0.0
*/
void SpiralGalaxy_eqFunction_7999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7999};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1499]] /* vz[500] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12991(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_15(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_7501(data, threadData);
  SpiralGalaxy_eqFunction_12682(data, threadData);
  SpiralGalaxy_eqFunction_7503(data, threadData);
  SpiralGalaxy_eqFunction_12681(data, threadData);
  SpiralGalaxy_eqFunction_7505(data, threadData);
  SpiralGalaxy_eqFunction_12694(data, threadData);
  SpiralGalaxy_eqFunction_12695(data, threadData);
  SpiralGalaxy_eqFunction_7508(data, threadData);
  SpiralGalaxy_eqFunction_7509(data, threadData);
  SpiralGalaxy_eqFunction_12696(data, threadData);
  SpiralGalaxy_eqFunction_12697(data, threadData);
  SpiralGalaxy_eqFunction_12700(data, threadData);
  SpiralGalaxy_eqFunction_12699(data, threadData);
  SpiralGalaxy_eqFunction_12698(data, threadData);
  SpiralGalaxy_eqFunction_7515(data, threadData);
  SpiralGalaxy_eqFunction_12693(data, threadData);
  SpiralGalaxy_eqFunction_7517(data, threadData);
  SpiralGalaxy_eqFunction_12692(data, threadData);
  SpiralGalaxy_eqFunction_7519(data, threadData);
  SpiralGalaxy_eqFunction_12691(data, threadData);
  SpiralGalaxy_eqFunction_7521(data, threadData);
  SpiralGalaxy_eqFunction_12704(data, threadData);
  SpiralGalaxy_eqFunction_12705(data, threadData);
  SpiralGalaxy_eqFunction_7524(data, threadData);
  SpiralGalaxy_eqFunction_7525(data, threadData);
  SpiralGalaxy_eqFunction_12706(data, threadData);
  SpiralGalaxy_eqFunction_12707(data, threadData);
  SpiralGalaxy_eqFunction_12710(data, threadData);
  SpiralGalaxy_eqFunction_12709(data, threadData);
  SpiralGalaxy_eqFunction_12708(data, threadData);
  SpiralGalaxy_eqFunction_7531(data, threadData);
  SpiralGalaxy_eqFunction_12703(data, threadData);
  SpiralGalaxy_eqFunction_7533(data, threadData);
  SpiralGalaxy_eqFunction_12702(data, threadData);
  SpiralGalaxy_eqFunction_7535(data, threadData);
  SpiralGalaxy_eqFunction_12701(data, threadData);
  SpiralGalaxy_eqFunction_7537(data, threadData);
  SpiralGalaxy_eqFunction_12714(data, threadData);
  SpiralGalaxy_eqFunction_12715(data, threadData);
  SpiralGalaxy_eqFunction_7540(data, threadData);
  SpiralGalaxy_eqFunction_7541(data, threadData);
  SpiralGalaxy_eqFunction_12716(data, threadData);
  SpiralGalaxy_eqFunction_12717(data, threadData);
  SpiralGalaxy_eqFunction_12720(data, threadData);
  SpiralGalaxy_eqFunction_12719(data, threadData);
  SpiralGalaxy_eqFunction_12718(data, threadData);
  SpiralGalaxy_eqFunction_7547(data, threadData);
  SpiralGalaxy_eqFunction_12713(data, threadData);
  SpiralGalaxy_eqFunction_7549(data, threadData);
  SpiralGalaxy_eqFunction_12712(data, threadData);
  SpiralGalaxy_eqFunction_7551(data, threadData);
  SpiralGalaxy_eqFunction_12711(data, threadData);
  SpiralGalaxy_eqFunction_7553(data, threadData);
  SpiralGalaxy_eqFunction_12724(data, threadData);
  SpiralGalaxy_eqFunction_12725(data, threadData);
  SpiralGalaxy_eqFunction_7556(data, threadData);
  SpiralGalaxy_eqFunction_7557(data, threadData);
  SpiralGalaxy_eqFunction_12726(data, threadData);
  SpiralGalaxy_eqFunction_12727(data, threadData);
  SpiralGalaxy_eqFunction_12730(data, threadData);
  SpiralGalaxy_eqFunction_12729(data, threadData);
  SpiralGalaxy_eqFunction_12728(data, threadData);
  SpiralGalaxy_eqFunction_7563(data, threadData);
  SpiralGalaxy_eqFunction_12723(data, threadData);
  SpiralGalaxy_eqFunction_7565(data, threadData);
  SpiralGalaxy_eqFunction_12722(data, threadData);
  SpiralGalaxy_eqFunction_7567(data, threadData);
  SpiralGalaxy_eqFunction_12721(data, threadData);
  SpiralGalaxy_eqFunction_7569(data, threadData);
  SpiralGalaxy_eqFunction_12734(data, threadData);
  SpiralGalaxy_eqFunction_12735(data, threadData);
  SpiralGalaxy_eqFunction_7572(data, threadData);
  SpiralGalaxy_eqFunction_7573(data, threadData);
  SpiralGalaxy_eqFunction_12736(data, threadData);
  SpiralGalaxy_eqFunction_12737(data, threadData);
  SpiralGalaxy_eqFunction_12740(data, threadData);
  SpiralGalaxy_eqFunction_12739(data, threadData);
  SpiralGalaxy_eqFunction_12738(data, threadData);
  SpiralGalaxy_eqFunction_7579(data, threadData);
  SpiralGalaxy_eqFunction_12733(data, threadData);
  SpiralGalaxy_eqFunction_7581(data, threadData);
  SpiralGalaxy_eqFunction_12732(data, threadData);
  SpiralGalaxy_eqFunction_7583(data, threadData);
  SpiralGalaxy_eqFunction_12731(data, threadData);
  SpiralGalaxy_eqFunction_7585(data, threadData);
  SpiralGalaxy_eqFunction_12744(data, threadData);
  SpiralGalaxy_eqFunction_12745(data, threadData);
  SpiralGalaxy_eqFunction_7588(data, threadData);
  SpiralGalaxy_eqFunction_7589(data, threadData);
  SpiralGalaxy_eqFunction_12746(data, threadData);
  SpiralGalaxy_eqFunction_12747(data, threadData);
  SpiralGalaxy_eqFunction_12750(data, threadData);
  SpiralGalaxy_eqFunction_12749(data, threadData);
  SpiralGalaxy_eqFunction_12748(data, threadData);
  SpiralGalaxy_eqFunction_7595(data, threadData);
  SpiralGalaxy_eqFunction_12743(data, threadData);
  SpiralGalaxy_eqFunction_7597(data, threadData);
  SpiralGalaxy_eqFunction_12742(data, threadData);
  SpiralGalaxy_eqFunction_7599(data, threadData);
  SpiralGalaxy_eqFunction_12741(data, threadData);
  SpiralGalaxy_eqFunction_7601(data, threadData);
  SpiralGalaxy_eqFunction_12754(data, threadData);
  SpiralGalaxy_eqFunction_12755(data, threadData);
  SpiralGalaxy_eqFunction_7604(data, threadData);
  SpiralGalaxy_eqFunction_7605(data, threadData);
  SpiralGalaxy_eqFunction_12756(data, threadData);
  SpiralGalaxy_eqFunction_12757(data, threadData);
  SpiralGalaxy_eqFunction_12760(data, threadData);
  SpiralGalaxy_eqFunction_12759(data, threadData);
  SpiralGalaxy_eqFunction_12758(data, threadData);
  SpiralGalaxy_eqFunction_7611(data, threadData);
  SpiralGalaxy_eqFunction_12753(data, threadData);
  SpiralGalaxy_eqFunction_7613(data, threadData);
  SpiralGalaxy_eqFunction_12752(data, threadData);
  SpiralGalaxy_eqFunction_7615(data, threadData);
  SpiralGalaxy_eqFunction_12751(data, threadData);
  SpiralGalaxy_eqFunction_7617(data, threadData);
  SpiralGalaxy_eqFunction_12764(data, threadData);
  SpiralGalaxy_eqFunction_12765(data, threadData);
  SpiralGalaxy_eqFunction_7620(data, threadData);
  SpiralGalaxy_eqFunction_7621(data, threadData);
  SpiralGalaxy_eqFunction_12766(data, threadData);
  SpiralGalaxy_eqFunction_12767(data, threadData);
  SpiralGalaxy_eqFunction_12770(data, threadData);
  SpiralGalaxy_eqFunction_12769(data, threadData);
  SpiralGalaxy_eqFunction_12768(data, threadData);
  SpiralGalaxy_eqFunction_7627(data, threadData);
  SpiralGalaxy_eqFunction_12763(data, threadData);
  SpiralGalaxy_eqFunction_7629(data, threadData);
  SpiralGalaxy_eqFunction_12762(data, threadData);
  SpiralGalaxy_eqFunction_7631(data, threadData);
  SpiralGalaxy_eqFunction_12761(data, threadData);
  SpiralGalaxy_eqFunction_7633(data, threadData);
  SpiralGalaxy_eqFunction_12774(data, threadData);
  SpiralGalaxy_eqFunction_12775(data, threadData);
  SpiralGalaxy_eqFunction_7636(data, threadData);
  SpiralGalaxy_eqFunction_7637(data, threadData);
  SpiralGalaxy_eqFunction_12776(data, threadData);
  SpiralGalaxy_eqFunction_12777(data, threadData);
  SpiralGalaxy_eqFunction_12780(data, threadData);
  SpiralGalaxy_eqFunction_12779(data, threadData);
  SpiralGalaxy_eqFunction_12778(data, threadData);
  SpiralGalaxy_eqFunction_7643(data, threadData);
  SpiralGalaxy_eqFunction_12773(data, threadData);
  SpiralGalaxy_eqFunction_7645(data, threadData);
  SpiralGalaxy_eqFunction_12772(data, threadData);
  SpiralGalaxy_eqFunction_7647(data, threadData);
  SpiralGalaxy_eqFunction_12771(data, threadData);
  SpiralGalaxy_eqFunction_7649(data, threadData);
  SpiralGalaxy_eqFunction_12784(data, threadData);
  SpiralGalaxy_eqFunction_12785(data, threadData);
  SpiralGalaxy_eqFunction_7652(data, threadData);
  SpiralGalaxy_eqFunction_7653(data, threadData);
  SpiralGalaxy_eqFunction_12786(data, threadData);
  SpiralGalaxy_eqFunction_12787(data, threadData);
  SpiralGalaxy_eqFunction_12790(data, threadData);
  SpiralGalaxy_eqFunction_12789(data, threadData);
  SpiralGalaxy_eqFunction_12788(data, threadData);
  SpiralGalaxy_eqFunction_7659(data, threadData);
  SpiralGalaxy_eqFunction_12783(data, threadData);
  SpiralGalaxy_eqFunction_7661(data, threadData);
  SpiralGalaxy_eqFunction_12782(data, threadData);
  SpiralGalaxy_eqFunction_7663(data, threadData);
  SpiralGalaxy_eqFunction_12781(data, threadData);
  SpiralGalaxy_eqFunction_7665(data, threadData);
  SpiralGalaxy_eqFunction_12794(data, threadData);
  SpiralGalaxy_eqFunction_12795(data, threadData);
  SpiralGalaxy_eqFunction_7668(data, threadData);
  SpiralGalaxy_eqFunction_7669(data, threadData);
  SpiralGalaxy_eqFunction_12796(data, threadData);
  SpiralGalaxy_eqFunction_12797(data, threadData);
  SpiralGalaxy_eqFunction_12800(data, threadData);
  SpiralGalaxy_eqFunction_12799(data, threadData);
  SpiralGalaxy_eqFunction_12798(data, threadData);
  SpiralGalaxy_eqFunction_7675(data, threadData);
  SpiralGalaxy_eqFunction_12793(data, threadData);
  SpiralGalaxy_eqFunction_7677(data, threadData);
  SpiralGalaxy_eqFunction_12792(data, threadData);
  SpiralGalaxy_eqFunction_7679(data, threadData);
  SpiralGalaxy_eqFunction_12791(data, threadData);
  SpiralGalaxy_eqFunction_7681(data, threadData);
  SpiralGalaxy_eqFunction_12804(data, threadData);
  SpiralGalaxy_eqFunction_12805(data, threadData);
  SpiralGalaxy_eqFunction_7684(data, threadData);
  SpiralGalaxy_eqFunction_7685(data, threadData);
  SpiralGalaxy_eqFunction_12806(data, threadData);
  SpiralGalaxy_eqFunction_12807(data, threadData);
  SpiralGalaxy_eqFunction_12810(data, threadData);
  SpiralGalaxy_eqFunction_12809(data, threadData);
  SpiralGalaxy_eqFunction_12808(data, threadData);
  SpiralGalaxy_eqFunction_7691(data, threadData);
  SpiralGalaxy_eqFunction_12803(data, threadData);
  SpiralGalaxy_eqFunction_7693(data, threadData);
  SpiralGalaxy_eqFunction_12802(data, threadData);
  SpiralGalaxy_eqFunction_7695(data, threadData);
  SpiralGalaxy_eqFunction_12801(data, threadData);
  SpiralGalaxy_eqFunction_7697(data, threadData);
  SpiralGalaxy_eqFunction_12814(data, threadData);
  SpiralGalaxy_eqFunction_12815(data, threadData);
  SpiralGalaxy_eqFunction_7700(data, threadData);
  SpiralGalaxy_eqFunction_7701(data, threadData);
  SpiralGalaxy_eqFunction_12816(data, threadData);
  SpiralGalaxy_eqFunction_12817(data, threadData);
  SpiralGalaxy_eqFunction_12820(data, threadData);
  SpiralGalaxy_eqFunction_12819(data, threadData);
  SpiralGalaxy_eqFunction_12818(data, threadData);
  SpiralGalaxy_eqFunction_7707(data, threadData);
  SpiralGalaxy_eqFunction_12813(data, threadData);
  SpiralGalaxy_eqFunction_7709(data, threadData);
  SpiralGalaxy_eqFunction_12812(data, threadData);
  SpiralGalaxy_eqFunction_7711(data, threadData);
  SpiralGalaxy_eqFunction_12811(data, threadData);
  SpiralGalaxy_eqFunction_7713(data, threadData);
  SpiralGalaxy_eqFunction_12824(data, threadData);
  SpiralGalaxy_eqFunction_12825(data, threadData);
  SpiralGalaxy_eqFunction_7716(data, threadData);
  SpiralGalaxy_eqFunction_7717(data, threadData);
  SpiralGalaxy_eqFunction_12826(data, threadData);
  SpiralGalaxy_eqFunction_12827(data, threadData);
  SpiralGalaxy_eqFunction_12830(data, threadData);
  SpiralGalaxy_eqFunction_12829(data, threadData);
  SpiralGalaxy_eqFunction_12828(data, threadData);
  SpiralGalaxy_eqFunction_7723(data, threadData);
  SpiralGalaxy_eqFunction_12823(data, threadData);
  SpiralGalaxy_eqFunction_7725(data, threadData);
  SpiralGalaxy_eqFunction_12822(data, threadData);
  SpiralGalaxy_eqFunction_7727(data, threadData);
  SpiralGalaxy_eqFunction_12821(data, threadData);
  SpiralGalaxy_eqFunction_7729(data, threadData);
  SpiralGalaxy_eqFunction_12834(data, threadData);
  SpiralGalaxy_eqFunction_12835(data, threadData);
  SpiralGalaxy_eqFunction_7732(data, threadData);
  SpiralGalaxy_eqFunction_7733(data, threadData);
  SpiralGalaxy_eqFunction_12836(data, threadData);
  SpiralGalaxy_eqFunction_12837(data, threadData);
  SpiralGalaxy_eqFunction_12840(data, threadData);
  SpiralGalaxy_eqFunction_12839(data, threadData);
  SpiralGalaxy_eqFunction_12838(data, threadData);
  SpiralGalaxy_eqFunction_7739(data, threadData);
  SpiralGalaxy_eqFunction_12833(data, threadData);
  SpiralGalaxy_eqFunction_7741(data, threadData);
  SpiralGalaxy_eqFunction_12832(data, threadData);
  SpiralGalaxy_eqFunction_7743(data, threadData);
  SpiralGalaxy_eqFunction_12831(data, threadData);
  SpiralGalaxy_eqFunction_7745(data, threadData);
  SpiralGalaxy_eqFunction_12844(data, threadData);
  SpiralGalaxy_eqFunction_12845(data, threadData);
  SpiralGalaxy_eqFunction_7748(data, threadData);
  SpiralGalaxy_eqFunction_7749(data, threadData);
  SpiralGalaxy_eqFunction_12846(data, threadData);
  SpiralGalaxy_eqFunction_12847(data, threadData);
  SpiralGalaxy_eqFunction_12850(data, threadData);
  SpiralGalaxy_eqFunction_12849(data, threadData);
  SpiralGalaxy_eqFunction_12848(data, threadData);
  SpiralGalaxy_eqFunction_7755(data, threadData);
  SpiralGalaxy_eqFunction_12843(data, threadData);
  SpiralGalaxy_eqFunction_7757(data, threadData);
  SpiralGalaxy_eqFunction_12842(data, threadData);
  SpiralGalaxy_eqFunction_7759(data, threadData);
  SpiralGalaxy_eqFunction_12841(data, threadData);
  SpiralGalaxy_eqFunction_7761(data, threadData);
  SpiralGalaxy_eqFunction_12854(data, threadData);
  SpiralGalaxy_eqFunction_12855(data, threadData);
  SpiralGalaxy_eqFunction_7764(data, threadData);
  SpiralGalaxy_eqFunction_7765(data, threadData);
  SpiralGalaxy_eqFunction_12856(data, threadData);
  SpiralGalaxy_eqFunction_12857(data, threadData);
  SpiralGalaxy_eqFunction_12860(data, threadData);
  SpiralGalaxy_eqFunction_12859(data, threadData);
  SpiralGalaxy_eqFunction_12858(data, threadData);
  SpiralGalaxy_eqFunction_7771(data, threadData);
  SpiralGalaxy_eqFunction_12853(data, threadData);
  SpiralGalaxy_eqFunction_7773(data, threadData);
  SpiralGalaxy_eqFunction_12852(data, threadData);
  SpiralGalaxy_eqFunction_7775(data, threadData);
  SpiralGalaxy_eqFunction_12851(data, threadData);
  SpiralGalaxy_eqFunction_7777(data, threadData);
  SpiralGalaxy_eqFunction_12864(data, threadData);
  SpiralGalaxy_eqFunction_12865(data, threadData);
  SpiralGalaxy_eqFunction_7780(data, threadData);
  SpiralGalaxy_eqFunction_7781(data, threadData);
  SpiralGalaxy_eqFunction_12866(data, threadData);
  SpiralGalaxy_eqFunction_12867(data, threadData);
  SpiralGalaxy_eqFunction_12870(data, threadData);
  SpiralGalaxy_eqFunction_12869(data, threadData);
  SpiralGalaxy_eqFunction_12868(data, threadData);
  SpiralGalaxy_eqFunction_7787(data, threadData);
  SpiralGalaxy_eqFunction_12863(data, threadData);
  SpiralGalaxy_eqFunction_7789(data, threadData);
  SpiralGalaxy_eqFunction_12862(data, threadData);
  SpiralGalaxy_eqFunction_7791(data, threadData);
  SpiralGalaxy_eqFunction_12861(data, threadData);
  SpiralGalaxy_eqFunction_7793(data, threadData);
  SpiralGalaxy_eqFunction_12874(data, threadData);
  SpiralGalaxy_eqFunction_12875(data, threadData);
  SpiralGalaxy_eqFunction_7796(data, threadData);
  SpiralGalaxy_eqFunction_7797(data, threadData);
  SpiralGalaxy_eqFunction_12876(data, threadData);
  SpiralGalaxy_eqFunction_12877(data, threadData);
  SpiralGalaxy_eqFunction_12880(data, threadData);
  SpiralGalaxy_eqFunction_12879(data, threadData);
  SpiralGalaxy_eqFunction_12878(data, threadData);
  SpiralGalaxy_eqFunction_7803(data, threadData);
  SpiralGalaxy_eqFunction_12873(data, threadData);
  SpiralGalaxy_eqFunction_7805(data, threadData);
  SpiralGalaxy_eqFunction_12872(data, threadData);
  SpiralGalaxy_eqFunction_7807(data, threadData);
  SpiralGalaxy_eqFunction_12871(data, threadData);
  SpiralGalaxy_eqFunction_7809(data, threadData);
  SpiralGalaxy_eqFunction_12884(data, threadData);
  SpiralGalaxy_eqFunction_12885(data, threadData);
  SpiralGalaxy_eqFunction_7812(data, threadData);
  SpiralGalaxy_eqFunction_7813(data, threadData);
  SpiralGalaxy_eqFunction_12886(data, threadData);
  SpiralGalaxy_eqFunction_12887(data, threadData);
  SpiralGalaxy_eqFunction_12890(data, threadData);
  SpiralGalaxy_eqFunction_12889(data, threadData);
  SpiralGalaxy_eqFunction_12888(data, threadData);
  SpiralGalaxy_eqFunction_7819(data, threadData);
  SpiralGalaxy_eqFunction_12883(data, threadData);
  SpiralGalaxy_eqFunction_7821(data, threadData);
  SpiralGalaxy_eqFunction_12882(data, threadData);
  SpiralGalaxy_eqFunction_7823(data, threadData);
  SpiralGalaxy_eqFunction_12881(data, threadData);
  SpiralGalaxy_eqFunction_7825(data, threadData);
  SpiralGalaxy_eqFunction_12894(data, threadData);
  SpiralGalaxy_eqFunction_12895(data, threadData);
  SpiralGalaxy_eqFunction_7828(data, threadData);
  SpiralGalaxy_eqFunction_7829(data, threadData);
  SpiralGalaxy_eqFunction_12896(data, threadData);
  SpiralGalaxy_eqFunction_12897(data, threadData);
  SpiralGalaxy_eqFunction_12900(data, threadData);
  SpiralGalaxy_eqFunction_12899(data, threadData);
  SpiralGalaxy_eqFunction_12898(data, threadData);
  SpiralGalaxy_eqFunction_7835(data, threadData);
  SpiralGalaxy_eqFunction_12893(data, threadData);
  SpiralGalaxy_eqFunction_7837(data, threadData);
  SpiralGalaxy_eqFunction_12892(data, threadData);
  SpiralGalaxy_eqFunction_7839(data, threadData);
  SpiralGalaxy_eqFunction_12891(data, threadData);
  SpiralGalaxy_eqFunction_7841(data, threadData);
  SpiralGalaxy_eqFunction_12904(data, threadData);
  SpiralGalaxy_eqFunction_12905(data, threadData);
  SpiralGalaxy_eqFunction_7844(data, threadData);
  SpiralGalaxy_eqFunction_7845(data, threadData);
  SpiralGalaxy_eqFunction_12906(data, threadData);
  SpiralGalaxy_eqFunction_12907(data, threadData);
  SpiralGalaxy_eqFunction_12910(data, threadData);
  SpiralGalaxy_eqFunction_12909(data, threadData);
  SpiralGalaxy_eqFunction_12908(data, threadData);
  SpiralGalaxy_eqFunction_7851(data, threadData);
  SpiralGalaxy_eqFunction_12903(data, threadData);
  SpiralGalaxy_eqFunction_7853(data, threadData);
  SpiralGalaxy_eqFunction_12902(data, threadData);
  SpiralGalaxy_eqFunction_7855(data, threadData);
  SpiralGalaxy_eqFunction_12901(data, threadData);
  SpiralGalaxy_eqFunction_7857(data, threadData);
  SpiralGalaxy_eqFunction_12914(data, threadData);
  SpiralGalaxy_eqFunction_12915(data, threadData);
  SpiralGalaxy_eqFunction_7860(data, threadData);
  SpiralGalaxy_eqFunction_7861(data, threadData);
  SpiralGalaxy_eqFunction_12916(data, threadData);
  SpiralGalaxy_eqFunction_12917(data, threadData);
  SpiralGalaxy_eqFunction_12920(data, threadData);
  SpiralGalaxy_eqFunction_12919(data, threadData);
  SpiralGalaxy_eqFunction_12918(data, threadData);
  SpiralGalaxy_eqFunction_7867(data, threadData);
  SpiralGalaxy_eqFunction_12913(data, threadData);
  SpiralGalaxy_eqFunction_7869(data, threadData);
  SpiralGalaxy_eqFunction_12912(data, threadData);
  SpiralGalaxy_eqFunction_7871(data, threadData);
  SpiralGalaxy_eqFunction_12911(data, threadData);
  SpiralGalaxy_eqFunction_7873(data, threadData);
  SpiralGalaxy_eqFunction_12924(data, threadData);
  SpiralGalaxy_eqFunction_12925(data, threadData);
  SpiralGalaxy_eqFunction_7876(data, threadData);
  SpiralGalaxy_eqFunction_7877(data, threadData);
  SpiralGalaxy_eqFunction_12926(data, threadData);
  SpiralGalaxy_eqFunction_12927(data, threadData);
  SpiralGalaxy_eqFunction_12930(data, threadData);
  SpiralGalaxy_eqFunction_12929(data, threadData);
  SpiralGalaxy_eqFunction_12928(data, threadData);
  SpiralGalaxy_eqFunction_7883(data, threadData);
  SpiralGalaxy_eqFunction_12923(data, threadData);
  SpiralGalaxy_eqFunction_7885(data, threadData);
  SpiralGalaxy_eqFunction_12922(data, threadData);
  SpiralGalaxy_eqFunction_7887(data, threadData);
  SpiralGalaxy_eqFunction_12921(data, threadData);
  SpiralGalaxy_eqFunction_7889(data, threadData);
  SpiralGalaxy_eqFunction_12934(data, threadData);
  SpiralGalaxy_eqFunction_12935(data, threadData);
  SpiralGalaxy_eqFunction_7892(data, threadData);
  SpiralGalaxy_eqFunction_7893(data, threadData);
  SpiralGalaxy_eqFunction_12936(data, threadData);
  SpiralGalaxy_eqFunction_12937(data, threadData);
  SpiralGalaxy_eqFunction_12940(data, threadData);
  SpiralGalaxy_eqFunction_12939(data, threadData);
  SpiralGalaxy_eqFunction_12938(data, threadData);
  SpiralGalaxy_eqFunction_7899(data, threadData);
  SpiralGalaxy_eqFunction_12933(data, threadData);
  SpiralGalaxy_eqFunction_7901(data, threadData);
  SpiralGalaxy_eqFunction_12932(data, threadData);
  SpiralGalaxy_eqFunction_7903(data, threadData);
  SpiralGalaxy_eqFunction_12931(data, threadData);
  SpiralGalaxy_eqFunction_7905(data, threadData);
  SpiralGalaxy_eqFunction_12944(data, threadData);
  SpiralGalaxy_eqFunction_12945(data, threadData);
  SpiralGalaxy_eqFunction_7908(data, threadData);
  SpiralGalaxy_eqFunction_7909(data, threadData);
  SpiralGalaxy_eqFunction_12946(data, threadData);
  SpiralGalaxy_eqFunction_12947(data, threadData);
  SpiralGalaxy_eqFunction_12950(data, threadData);
  SpiralGalaxy_eqFunction_12949(data, threadData);
  SpiralGalaxy_eqFunction_12948(data, threadData);
  SpiralGalaxy_eqFunction_7915(data, threadData);
  SpiralGalaxy_eqFunction_12943(data, threadData);
  SpiralGalaxy_eqFunction_7917(data, threadData);
  SpiralGalaxy_eqFunction_12942(data, threadData);
  SpiralGalaxy_eqFunction_7919(data, threadData);
  SpiralGalaxy_eqFunction_12941(data, threadData);
  SpiralGalaxy_eqFunction_7921(data, threadData);
  SpiralGalaxy_eqFunction_12954(data, threadData);
  SpiralGalaxy_eqFunction_12955(data, threadData);
  SpiralGalaxy_eqFunction_7924(data, threadData);
  SpiralGalaxy_eqFunction_7925(data, threadData);
  SpiralGalaxy_eqFunction_12956(data, threadData);
  SpiralGalaxy_eqFunction_12957(data, threadData);
  SpiralGalaxy_eqFunction_12960(data, threadData);
  SpiralGalaxy_eqFunction_12959(data, threadData);
  SpiralGalaxy_eqFunction_12958(data, threadData);
  SpiralGalaxy_eqFunction_7931(data, threadData);
  SpiralGalaxy_eqFunction_12953(data, threadData);
  SpiralGalaxy_eqFunction_7933(data, threadData);
  SpiralGalaxy_eqFunction_12952(data, threadData);
  SpiralGalaxy_eqFunction_7935(data, threadData);
  SpiralGalaxy_eqFunction_12951(data, threadData);
  SpiralGalaxy_eqFunction_7937(data, threadData);
  SpiralGalaxy_eqFunction_12964(data, threadData);
  SpiralGalaxy_eqFunction_12965(data, threadData);
  SpiralGalaxy_eqFunction_7940(data, threadData);
  SpiralGalaxy_eqFunction_7941(data, threadData);
  SpiralGalaxy_eqFunction_12966(data, threadData);
  SpiralGalaxy_eqFunction_12967(data, threadData);
  SpiralGalaxy_eqFunction_12970(data, threadData);
  SpiralGalaxy_eqFunction_12969(data, threadData);
  SpiralGalaxy_eqFunction_12968(data, threadData);
  SpiralGalaxy_eqFunction_7947(data, threadData);
  SpiralGalaxy_eqFunction_12963(data, threadData);
  SpiralGalaxy_eqFunction_7949(data, threadData);
  SpiralGalaxy_eqFunction_12962(data, threadData);
  SpiralGalaxy_eqFunction_7951(data, threadData);
  SpiralGalaxy_eqFunction_12961(data, threadData);
  SpiralGalaxy_eqFunction_7953(data, threadData);
  SpiralGalaxy_eqFunction_12974(data, threadData);
  SpiralGalaxy_eqFunction_12975(data, threadData);
  SpiralGalaxy_eqFunction_7956(data, threadData);
  SpiralGalaxy_eqFunction_7957(data, threadData);
  SpiralGalaxy_eqFunction_12976(data, threadData);
  SpiralGalaxy_eqFunction_12977(data, threadData);
  SpiralGalaxy_eqFunction_12980(data, threadData);
  SpiralGalaxy_eqFunction_12979(data, threadData);
  SpiralGalaxy_eqFunction_12978(data, threadData);
  SpiralGalaxy_eqFunction_7963(data, threadData);
  SpiralGalaxy_eqFunction_12973(data, threadData);
  SpiralGalaxy_eqFunction_7965(data, threadData);
  SpiralGalaxy_eqFunction_12972(data, threadData);
  SpiralGalaxy_eqFunction_7967(data, threadData);
  SpiralGalaxy_eqFunction_12971(data, threadData);
  SpiralGalaxy_eqFunction_7969(data, threadData);
  SpiralGalaxy_eqFunction_12984(data, threadData);
  SpiralGalaxy_eqFunction_12985(data, threadData);
  SpiralGalaxy_eqFunction_7972(data, threadData);
  SpiralGalaxy_eqFunction_7973(data, threadData);
  SpiralGalaxy_eqFunction_12986(data, threadData);
  SpiralGalaxy_eqFunction_12987(data, threadData);
  SpiralGalaxy_eqFunction_12990(data, threadData);
  SpiralGalaxy_eqFunction_12989(data, threadData);
  SpiralGalaxy_eqFunction_12988(data, threadData);
  SpiralGalaxy_eqFunction_7979(data, threadData);
  SpiralGalaxy_eqFunction_12983(data, threadData);
  SpiralGalaxy_eqFunction_7981(data, threadData);
  SpiralGalaxy_eqFunction_12982(data, threadData);
  SpiralGalaxy_eqFunction_7983(data, threadData);
  SpiralGalaxy_eqFunction_12981(data, threadData);
  SpiralGalaxy_eqFunction_7985(data, threadData);
  SpiralGalaxy_eqFunction_12994(data, threadData);
  SpiralGalaxy_eqFunction_12995(data, threadData);
  SpiralGalaxy_eqFunction_7988(data, threadData);
  SpiralGalaxy_eqFunction_7989(data, threadData);
  SpiralGalaxy_eqFunction_12996(data, threadData);
  SpiralGalaxy_eqFunction_12997(data, threadData);
  SpiralGalaxy_eqFunction_13000(data, threadData);
  SpiralGalaxy_eqFunction_12999(data, threadData);
  SpiralGalaxy_eqFunction_12998(data, threadData);
  SpiralGalaxy_eqFunction_7995(data, threadData);
  SpiralGalaxy_eqFunction_12993(data, threadData);
  SpiralGalaxy_eqFunction_7997(data, threadData);
  SpiralGalaxy_eqFunction_12992(data, threadData);
  SpiralGalaxy_eqFunction_7999(data, threadData);
  SpiralGalaxy_eqFunction_12991(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif