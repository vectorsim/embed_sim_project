#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 5501
type: SIMPLE_ASSIGN
vy[344] = cos(theta[344]) * r_init[344] * omega_c[344]
*/
void SpiralGalaxy_eqFunction_5501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[843]] /* vy[344] STATE(1) */) = (cos((data->simulationInfo->realParameter[1850] /* theta[344] PARAM */))) * (((data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */)) * ((data->simulationInfo->realParameter[848] /* omega_c[344] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11432(DATA *data, threadData_t *threadData);


/*
equation index: 5503
type: SIMPLE_ASSIGN
vz[344] = 0.0
*/
void SpiralGalaxy_eqFunction_5503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5503};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1343]] /* vz[344] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11431(DATA *data, threadData_t *threadData);


/*
equation index: 5505
type: SIMPLE_ASSIGN
z[345] = 0.0152
*/
void SpiralGalaxy_eqFunction_5505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5505};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2844]] /* z[345] STATE(1,vz[345]) */) = 0.0152;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11444(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11445(DATA *data, threadData_t *threadData);


/*
equation index: 5508
type: SIMPLE_ASSIGN
y[345] = r_init[345] * sin(theta[345] + 0.003799999999999999)
*/
void SpiralGalaxy_eqFunction_5508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5508};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2344]] /* y[345] STATE(1,vy[345]) */) = ((data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */)) * (sin((data->simulationInfo->realParameter[1851] /* theta[345] PARAM */) + 0.003799999999999999));
  TRACE_POP
}

/*
equation index: 5509
type: SIMPLE_ASSIGN
x[345] = r_init[345] * cos(theta[345] + 0.003799999999999999)
*/
void SpiralGalaxy_eqFunction_5509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1844]] /* x[345] STATE(1,vx[345]) */) = ((data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */)) * (cos((data->simulationInfo->realParameter[1851] /* theta[345] PARAM */) + 0.003799999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11446(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11447(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11450(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11449(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11448(DATA *data, threadData_t *threadData);


/*
equation index: 5515
type: SIMPLE_ASSIGN
vx[345] = (-sin(theta[345])) * r_init[345] * omega_c[345]
*/
void SpiralGalaxy_eqFunction_5515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5515};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[344]] /* vx[345] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1851] /* theta[345] PARAM */)))) * (((data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */)) * ((data->simulationInfo->realParameter[849] /* omega_c[345] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11443(DATA *data, threadData_t *threadData);


/*
equation index: 5517
type: SIMPLE_ASSIGN
vy[345] = cos(theta[345]) * r_init[345] * omega_c[345]
*/
void SpiralGalaxy_eqFunction_5517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[844]] /* vy[345] STATE(1) */) = (cos((data->simulationInfo->realParameter[1851] /* theta[345] PARAM */))) * (((data->simulationInfo->realParameter[1350] /* r_init[345] PARAM */)) * ((data->simulationInfo->realParameter[849] /* omega_c[345] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11442(DATA *data, threadData_t *threadData);


/*
equation index: 5519
type: SIMPLE_ASSIGN
vz[345] = 0.0
*/
void SpiralGalaxy_eqFunction_5519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5519};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1344]] /* vz[345] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11441(DATA *data, threadData_t *threadData);


/*
equation index: 5521
type: SIMPLE_ASSIGN
z[346] = 0.01536
*/
void SpiralGalaxy_eqFunction_5521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5521};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2845]] /* z[346] STATE(1,vz[346]) */) = 0.01536;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11454(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11455(DATA *data, threadData_t *threadData);


/*
equation index: 5524
type: SIMPLE_ASSIGN
y[346] = r_init[346] * sin(theta[346] + 0.0038399999999999992)
*/
void SpiralGalaxy_eqFunction_5524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5524};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2345]] /* y[346] STATE(1,vy[346]) */) = ((data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */)) * (sin((data->simulationInfo->realParameter[1852] /* theta[346] PARAM */) + 0.0038399999999999992));
  TRACE_POP
}

/*
equation index: 5525
type: SIMPLE_ASSIGN
x[346] = r_init[346] * cos(theta[346] + 0.0038399999999999992)
*/
void SpiralGalaxy_eqFunction_5525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1845]] /* x[346] STATE(1,vx[346]) */) = ((data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */)) * (cos((data->simulationInfo->realParameter[1852] /* theta[346] PARAM */) + 0.0038399999999999992));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11456(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11457(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11460(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11459(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11458(DATA *data, threadData_t *threadData);


/*
equation index: 5531
type: SIMPLE_ASSIGN
vx[346] = (-sin(theta[346])) * r_init[346] * omega_c[346]
*/
void SpiralGalaxy_eqFunction_5531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5531};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[345]] /* vx[346] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1852] /* theta[346] PARAM */)))) * (((data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */)) * ((data->simulationInfo->realParameter[850] /* omega_c[346] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11453(DATA *data, threadData_t *threadData);


/*
equation index: 5533
type: SIMPLE_ASSIGN
vy[346] = cos(theta[346]) * r_init[346] * omega_c[346]
*/
void SpiralGalaxy_eqFunction_5533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[845]] /* vy[346] STATE(1) */) = (cos((data->simulationInfo->realParameter[1852] /* theta[346] PARAM */))) * (((data->simulationInfo->realParameter[1351] /* r_init[346] PARAM */)) * ((data->simulationInfo->realParameter[850] /* omega_c[346] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11452(DATA *data, threadData_t *threadData);


/*
equation index: 5535
type: SIMPLE_ASSIGN
vz[346] = 0.0
*/
void SpiralGalaxy_eqFunction_5535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5535};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1345]] /* vz[346] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11451(DATA *data, threadData_t *threadData);


/*
equation index: 5537
type: SIMPLE_ASSIGN
z[347] = 0.01552
*/
void SpiralGalaxy_eqFunction_5537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5537};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2846]] /* z[347] STATE(1,vz[347]) */) = 0.01552;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11464(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11465(DATA *data, threadData_t *threadData);


/*
equation index: 5540
type: SIMPLE_ASSIGN
y[347] = r_init[347] * sin(theta[347] + 0.003879999999999999)
*/
void SpiralGalaxy_eqFunction_5540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5540};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2346]] /* y[347] STATE(1,vy[347]) */) = ((data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */)) * (sin((data->simulationInfo->realParameter[1853] /* theta[347] PARAM */) + 0.003879999999999999));
  TRACE_POP
}

/*
equation index: 5541
type: SIMPLE_ASSIGN
x[347] = r_init[347] * cos(theta[347] + 0.003879999999999999)
*/
void SpiralGalaxy_eqFunction_5541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1846]] /* x[347] STATE(1,vx[347]) */) = ((data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */)) * (cos((data->simulationInfo->realParameter[1853] /* theta[347] PARAM */) + 0.003879999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11466(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11467(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11470(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11469(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11468(DATA *data, threadData_t *threadData);


/*
equation index: 5547
type: SIMPLE_ASSIGN
vx[347] = (-sin(theta[347])) * r_init[347] * omega_c[347]
*/
void SpiralGalaxy_eqFunction_5547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5547};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[346]] /* vx[347] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1853] /* theta[347] PARAM */)))) * (((data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */)) * ((data->simulationInfo->realParameter[851] /* omega_c[347] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11463(DATA *data, threadData_t *threadData);


/*
equation index: 5549
type: SIMPLE_ASSIGN
vy[347] = cos(theta[347]) * r_init[347] * omega_c[347]
*/
void SpiralGalaxy_eqFunction_5549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[846]] /* vy[347] STATE(1) */) = (cos((data->simulationInfo->realParameter[1853] /* theta[347] PARAM */))) * (((data->simulationInfo->realParameter[1352] /* r_init[347] PARAM */)) * ((data->simulationInfo->realParameter[851] /* omega_c[347] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11462(DATA *data, threadData_t *threadData);


/*
equation index: 5551
type: SIMPLE_ASSIGN
vz[347] = 0.0
*/
void SpiralGalaxy_eqFunction_5551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5551};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1346]] /* vz[347] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11461(DATA *data, threadData_t *threadData);


/*
equation index: 5553
type: SIMPLE_ASSIGN
z[348] = 0.01568
*/
void SpiralGalaxy_eqFunction_5553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5553};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2847]] /* z[348] STATE(1,vz[348]) */) = 0.01568;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11474(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11475(DATA *data, threadData_t *threadData);


/*
equation index: 5556
type: SIMPLE_ASSIGN
y[348] = r_init[348] * sin(theta[348] + 0.003919999999999999)
*/
void SpiralGalaxy_eqFunction_5556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5556};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2347]] /* y[348] STATE(1,vy[348]) */) = ((data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */)) * (sin((data->simulationInfo->realParameter[1854] /* theta[348] PARAM */) + 0.003919999999999999));
  TRACE_POP
}

/*
equation index: 5557
type: SIMPLE_ASSIGN
x[348] = r_init[348] * cos(theta[348] + 0.003919999999999999)
*/
void SpiralGalaxy_eqFunction_5557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1847]] /* x[348] STATE(1,vx[348]) */) = ((data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */)) * (cos((data->simulationInfo->realParameter[1854] /* theta[348] PARAM */) + 0.003919999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11476(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11477(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11480(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11479(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11478(DATA *data, threadData_t *threadData);


/*
equation index: 5563
type: SIMPLE_ASSIGN
vx[348] = (-sin(theta[348])) * r_init[348] * omega_c[348]
*/
void SpiralGalaxy_eqFunction_5563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5563};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[347]] /* vx[348] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1854] /* theta[348] PARAM */)))) * (((data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */)) * ((data->simulationInfo->realParameter[852] /* omega_c[348] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11473(DATA *data, threadData_t *threadData);


/*
equation index: 5565
type: SIMPLE_ASSIGN
vy[348] = cos(theta[348]) * r_init[348] * omega_c[348]
*/
void SpiralGalaxy_eqFunction_5565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[847]] /* vy[348] STATE(1) */) = (cos((data->simulationInfo->realParameter[1854] /* theta[348] PARAM */))) * (((data->simulationInfo->realParameter[1353] /* r_init[348] PARAM */)) * ((data->simulationInfo->realParameter[852] /* omega_c[348] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11472(DATA *data, threadData_t *threadData);


/*
equation index: 5567
type: SIMPLE_ASSIGN
vz[348] = 0.0
*/
void SpiralGalaxy_eqFunction_5567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5567};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1347]] /* vz[348] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11471(DATA *data, threadData_t *threadData);


/*
equation index: 5569
type: SIMPLE_ASSIGN
z[349] = 0.01584
*/
void SpiralGalaxy_eqFunction_5569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5569};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2848]] /* z[349] STATE(1,vz[349]) */) = 0.01584;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11484(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11485(DATA *data, threadData_t *threadData);


/*
equation index: 5572
type: SIMPLE_ASSIGN
y[349] = r_init[349] * sin(theta[349] + 0.003959999999999999)
*/
void SpiralGalaxy_eqFunction_5572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5572};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2348]] /* y[349] STATE(1,vy[349]) */) = ((data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */)) * (sin((data->simulationInfo->realParameter[1855] /* theta[349] PARAM */) + 0.003959999999999999));
  TRACE_POP
}

/*
equation index: 5573
type: SIMPLE_ASSIGN
x[349] = r_init[349] * cos(theta[349] + 0.003959999999999999)
*/
void SpiralGalaxy_eqFunction_5573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1848]] /* x[349] STATE(1,vx[349]) */) = ((data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */)) * (cos((data->simulationInfo->realParameter[1855] /* theta[349] PARAM */) + 0.003959999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11486(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11487(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11490(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11489(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11488(DATA *data, threadData_t *threadData);


/*
equation index: 5579
type: SIMPLE_ASSIGN
vx[349] = (-sin(theta[349])) * r_init[349] * omega_c[349]
*/
void SpiralGalaxy_eqFunction_5579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5579};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[348]] /* vx[349] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1855] /* theta[349] PARAM */)))) * (((data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */)) * ((data->simulationInfo->realParameter[853] /* omega_c[349] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11483(DATA *data, threadData_t *threadData);


/*
equation index: 5581
type: SIMPLE_ASSIGN
vy[349] = cos(theta[349]) * r_init[349] * omega_c[349]
*/
void SpiralGalaxy_eqFunction_5581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[848]] /* vy[349] STATE(1) */) = (cos((data->simulationInfo->realParameter[1855] /* theta[349] PARAM */))) * (((data->simulationInfo->realParameter[1354] /* r_init[349] PARAM */)) * ((data->simulationInfo->realParameter[853] /* omega_c[349] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11482(DATA *data, threadData_t *threadData);


/*
equation index: 5583
type: SIMPLE_ASSIGN
vz[349] = 0.0
*/
void SpiralGalaxy_eqFunction_5583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5583};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1348]] /* vz[349] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11481(DATA *data, threadData_t *threadData);


/*
equation index: 5585
type: SIMPLE_ASSIGN
z[350] = 0.016
*/
void SpiralGalaxy_eqFunction_5585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5585};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2849]] /* z[350] STATE(1,vz[350]) */) = 0.016;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11494(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11495(DATA *data, threadData_t *threadData);


/*
equation index: 5588
type: SIMPLE_ASSIGN
y[350] = r_init[350] * sin(theta[350] + 0.003999999999999999)
*/
void SpiralGalaxy_eqFunction_5588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5588};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2349]] /* y[350] STATE(1,vy[350]) */) = ((data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */)) * (sin((data->simulationInfo->realParameter[1856] /* theta[350] PARAM */) + 0.003999999999999999));
  TRACE_POP
}

/*
equation index: 5589
type: SIMPLE_ASSIGN
x[350] = r_init[350] * cos(theta[350] + 0.003999999999999999)
*/
void SpiralGalaxy_eqFunction_5589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1849]] /* x[350] STATE(1,vx[350]) */) = ((data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */)) * (cos((data->simulationInfo->realParameter[1856] /* theta[350] PARAM */) + 0.003999999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11496(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11497(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11500(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11499(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11498(DATA *data, threadData_t *threadData);


/*
equation index: 5595
type: SIMPLE_ASSIGN
vx[350] = (-sin(theta[350])) * r_init[350] * omega_c[350]
*/
void SpiralGalaxy_eqFunction_5595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5595};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[349]] /* vx[350] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1856] /* theta[350] PARAM */)))) * (((data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */)) * ((data->simulationInfo->realParameter[854] /* omega_c[350] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11493(DATA *data, threadData_t *threadData);


/*
equation index: 5597
type: SIMPLE_ASSIGN
vy[350] = cos(theta[350]) * r_init[350] * omega_c[350]
*/
void SpiralGalaxy_eqFunction_5597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[849]] /* vy[350] STATE(1) */) = (cos((data->simulationInfo->realParameter[1856] /* theta[350] PARAM */))) * (((data->simulationInfo->realParameter[1355] /* r_init[350] PARAM */)) * ((data->simulationInfo->realParameter[854] /* omega_c[350] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11492(DATA *data, threadData_t *threadData);


/*
equation index: 5599
type: SIMPLE_ASSIGN
vz[350] = 0.0
*/
void SpiralGalaxy_eqFunction_5599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5599};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1349]] /* vz[350] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11491(DATA *data, threadData_t *threadData);


/*
equation index: 5601
type: SIMPLE_ASSIGN
z[351] = 0.01616
*/
void SpiralGalaxy_eqFunction_5601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5601};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2850]] /* z[351] STATE(1,vz[351]) */) = 0.01616;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11504(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11505(DATA *data, threadData_t *threadData);


/*
equation index: 5604
type: SIMPLE_ASSIGN
y[351] = r_init[351] * sin(theta[351] + 0.004039999999999999)
*/
void SpiralGalaxy_eqFunction_5604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5604};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2350]] /* y[351] STATE(1,vy[351]) */) = ((data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */)) * (sin((data->simulationInfo->realParameter[1857] /* theta[351] PARAM */) + 0.004039999999999999));
  TRACE_POP
}

/*
equation index: 5605
type: SIMPLE_ASSIGN
x[351] = r_init[351] * cos(theta[351] + 0.004039999999999999)
*/
void SpiralGalaxy_eqFunction_5605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1850]] /* x[351] STATE(1,vx[351]) */) = ((data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */)) * (cos((data->simulationInfo->realParameter[1857] /* theta[351] PARAM */) + 0.004039999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11506(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11507(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11510(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11509(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11508(DATA *data, threadData_t *threadData);


/*
equation index: 5611
type: SIMPLE_ASSIGN
vx[351] = (-sin(theta[351])) * r_init[351] * omega_c[351]
*/
void SpiralGalaxy_eqFunction_5611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5611};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[350]] /* vx[351] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1857] /* theta[351] PARAM */)))) * (((data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */)) * ((data->simulationInfo->realParameter[855] /* omega_c[351] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11503(DATA *data, threadData_t *threadData);


/*
equation index: 5613
type: SIMPLE_ASSIGN
vy[351] = cos(theta[351]) * r_init[351] * omega_c[351]
*/
void SpiralGalaxy_eqFunction_5613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[850]] /* vy[351] STATE(1) */) = (cos((data->simulationInfo->realParameter[1857] /* theta[351] PARAM */))) * (((data->simulationInfo->realParameter[1356] /* r_init[351] PARAM */)) * ((data->simulationInfo->realParameter[855] /* omega_c[351] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11502(DATA *data, threadData_t *threadData);


/*
equation index: 5615
type: SIMPLE_ASSIGN
vz[351] = 0.0
*/
void SpiralGalaxy_eqFunction_5615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5615};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1350]] /* vz[351] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11501(DATA *data, threadData_t *threadData);


/*
equation index: 5617
type: SIMPLE_ASSIGN
z[352] = 0.01632
*/
void SpiralGalaxy_eqFunction_5617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5617};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2851]] /* z[352] STATE(1,vz[352]) */) = 0.01632;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11514(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11515(DATA *data, threadData_t *threadData);


/*
equation index: 5620
type: SIMPLE_ASSIGN
y[352] = r_init[352] * sin(theta[352] + 0.004079999999999999)
*/
void SpiralGalaxy_eqFunction_5620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5620};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2351]] /* y[352] STATE(1,vy[352]) */) = ((data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */)) * (sin((data->simulationInfo->realParameter[1858] /* theta[352] PARAM */) + 0.004079999999999999));
  TRACE_POP
}

/*
equation index: 5621
type: SIMPLE_ASSIGN
x[352] = r_init[352] * cos(theta[352] + 0.004079999999999999)
*/
void SpiralGalaxy_eqFunction_5621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1851]] /* x[352] STATE(1,vx[352]) */) = ((data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */)) * (cos((data->simulationInfo->realParameter[1858] /* theta[352] PARAM */) + 0.004079999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11516(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11517(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11520(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11519(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11518(DATA *data, threadData_t *threadData);


/*
equation index: 5627
type: SIMPLE_ASSIGN
vx[352] = (-sin(theta[352])) * r_init[352] * omega_c[352]
*/
void SpiralGalaxy_eqFunction_5627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5627};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[351]] /* vx[352] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1858] /* theta[352] PARAM */)))) * (((data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */)) * ((data->simulationInfo->realParameter[856] /* omega_c[352] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11513(DATA *data, threadData_t *threadData);


/*
equation index: 5629
type: SIMPLE_ASSIGN
vy[352] = cos(theta[352]) * r_init[352] * omega_c[352]
*/
void SpiralGalaxy_eqFunction_5629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[851]] /* vy[352] STATE(1) */) = (cos((data->simulationInfo->realParameter[1858] /* theta[352] PARAM */))) * (((data->simulationInfo->realParameter[1357] /* r_init[352] PARAM */)) * ((data->simulationInfo->realParameter[856] /* omega_c[352] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11512(DATA *data, threadData_t *threadData);


/*
equation index: 5631
type: SIMPLE_ASSIGN
vz[352] = 0.0
*/
void SpiralGalaxy_eqFunction_5631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5631};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1351]] /* vz[352] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11511(DATA *data, threadData_t *threadData);


/*
equation index: 5633
type: SIMPLE_ASSIGN
z[353] = 0.01648
*/
void SpiralGalaxy_eqFunction_5633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5633};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2852]] /* z[353] STATE(1,vz[353]) */) = 0.01648;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11524(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11525(DATA *data, threadData_t *threadData);


/*
equation index: 5636
type: SIMPLE_ASSIGN
y[353] = r_init[353] * sin(theta[353] + 0.0041199999999999995)
*/
void SpiralGalaxy_eqFunction_5636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5636};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2352]] /* y[353] STATE(1,vy[353]) */) = ((data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */)) * (sin((data->simulationInfo->realParameter[1859] /* theta[353] PARAM */) + 0.0041199999999999995));
  TRACE_POP
}

/*
equation index: 5637
type: SIMPLE_ASSIGN
x[353] = r_init[353] * cos(theta[353] + 0.0041199999999999995)
*/
void SpiralGalaxy_eqFunction_5637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1852]] /* x[353] STATE(1,vx[353]) */) = ((data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */)) * (cos((data->simulationInfo->realParameter[1859] /* theta[353] PARAM */) + 0.0041199999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11526(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11527(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11530(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11529(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11528(DATA *data, threadData_t *threadData);


/*
equation index: 5643
type: SIMPLE_ASSIGN
vx[353] = (-sin(theta[353])) * r_init[353] * omega_c[353]
*/
void SpiralGalaxy_eqFunction_5643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5643};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[352]] /* vx[353] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1859] /* theta[353] PARAM */)))) * (((data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */)) * ((data->simulationInfo->realParameter[857] /* omega_c[353] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11523(DATA *data, threadData_t *threadData);


/*
equation index: 5645
type: SIMPLE_ASSIGN
vy[353] = cos(theta[353]) * r_init[353] * omega_c[353]
*/
void SpiralGalaxy_eqFunction_5645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[852]] /* vy[353] STATE(1) */) = (cos((data->simulationInfo->realParameter[1859] /* theta[353] PARAM */))) * (((data->simulationInfo->realParameter[1358] /* r_init[353] PARAM */)) * ((data->simulationInfo->realParameter[857] /* omega_c[353] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11522(DATA *data, threadData_t *threadData);


/*
equation index: 5647
type: SIMPLE_ASSIGN
vz[353] = 0.0
*/
void SpiralGalaxy_eqFunction_5647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5647};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1352]] /* vz[353] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11521(DATA *data, threadData_t *threadData);


/*
equation index: 5649
type: SIMPLE_ASSIGN
z[354] = 0.016640000000000002
*/
void SpiralGalaxy_eqFunction_5649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5649};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2853]] /* z[354] STATE(1,vz[354]) */) = 0.016640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11534(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11535(DATA *data, threadData_t *threadData);


/*
equation index: 5652
type: SIMPLE_ASSIGN
y[354] = r_init[354] * sin(theta[354] + 0.00416)
*/
void SpiralGalaxy_eqFunction_5652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5652};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2353]] /* y[354] STATE(1,vy[354]) */) = ((data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */)) * (sin((data->simulationInfo->realParameter[1860] /* theta[354] PARAM */) + 0.00416));
  TRACE_POP
}

/*
equation index: 5653
type: SIMPLE_ASSIGN
x[354] = r_init[354] * cos(theta[354] + 0.00416)
*/
void SpiralGalaxy_eqFunction_5653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1853]] /* x[354] STATE(1,vx[354]) */) = ((data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */)) * (cos((data->simulationInfo->realParameter[1860] /* theta[354] PARAM */) + 0.00416));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11536(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11537(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11540(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11539(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11538(DATA *data, threadData_t *threadData);


/*
equation index: 5659
type: SIMPLE_ASSIGN
vx[354] = (-sin(theta[354])) * r_init[354] * omega_c[354]
*/
void SpiralGalaxy_eqFunction_5659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5659};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[353]] /* vx[354] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1860] /* theta[354] PARAM */)))) * (((data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */)) * ((data->simulationInfo->realParameter[858] /* omega_c[354] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11533(DATA *data, threadData_t *threadData);


/*
equation index: 5661
type: SIMPLE_ASSIGN
vy[354] = cos(theta[354]) * r_init[354] * omega_c[354]
*/
void SpiralGalaxy_eqFunction_5661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[853]] /* vy[354] STATE(1) */) = (cos((data->simulationInfo->realParameter[1860] /* theta[354] PARAM */))) * (((data->simulationInfo->realParameter[1359] /* r_init[354] PARAM */)) * ((data->simulationInfo->realParameter[858] /* omega_c[354] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11532(DATA *data, threadData_t *threadData);


/*
equation index: 5663
type: SIMPLE_ASSIGN
vz[354] = 0.0
*/
void SpiralGalaxy_eqFunction_5663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5663};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1353]] /* vz[354] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11531(DATA *data, threadData_t *threadData);


/*
equation index: 5665
type: SIMPLE_ASSIGN
z[355] = 0.016800000000000002
*/
void SpiralGalaxy_eqFunction_5665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5665};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2854]] /* z[355] STATE(1,vz[355]) */) = 0.016800000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11544(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11545(DATA *data, threadData_t *threadData);


/*
equation index: 5668
type: SIMPLE_ASSIGN
y[355] = r_init[355] * sin(theta[355] + 0.0042)
*/
void SpiralGalaxy_eqFunction_5668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5668};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2354]] /* y[355] STATE(1,vy[355]) */) = ((data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */)) * (sin((data->simulationInfo->realParameter[1861] /* theta[355] PARAM */) + 0.0042));
  TRACE_POP
}

/*
equation index: 5669
type: SIMPLE_ASSIGN
x[355] = r_init[355] * cos(theta[355] + 0.0042)
*/
void SpiralGalaxy_eqFunction_5669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1854]] /* x[355] STATE(1,vx[355]) */) = ((data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */)) * (cos((data->simulationInfo->realParameter[1861] /* theta[355] PARAM */) + 0.0042));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11546(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11547(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11550(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11549(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11548(DATA *data, threadData_t *threadData);


/*
equation index: 5675
type: SIMPLE_ASSIGN
vx[355] = (-sin(theta[355])) * r_init[355] * omega_c[355]
*/
void SpiralGalaxy_eqFunction_5675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5675};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[354]] /* vx[355] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1861] /* theta[355] PARAM */)))) * (((data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */)) * ((data->simulationInfo->realParameter[859] /* omega_c[355] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11543(DATA *data, threadData_t *threadData);


/*
equation index: 5677
type: SIMPLE_ASSIGN
vy[355] = cos(theta[355]) * r_init[355] * omega_c[355]
*/
void SpiralGalaxy_eqFunction_5677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[854]] /* vy[355] STATE(1) */) = (cos((data->simulationInfo->realParameter[1861] /* theta[355] PARAM */))) * (((data->simulationInfo->realParameter[1360] /* r_init[355] PARAM */)) * ((data->simulationInfo->realParameter[859] /* omega_c[355] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11542(DATA *data, threadData_t *threadData);


/*
equation index: 5679
type: SIMPLE_ASSIGN
vz[355] = 0.0
*/
void SpiralGalaxy_eqFunction_5679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5679};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1354]] /* vz[355] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11541(DATA *data, threadData_t *threadData);


/*
equation index: 5681
type: SIMPLE_ASSIGN
z[356] = 0.016960000000000003
*/
void SpiralGalaxy_eqFunction_5681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5681};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2855]] /* z[356] STATE(1,vz[356]) */) = 0.016960000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11554(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11555(DATA *data, threadData_t *threadData);


/*
equation index: 5684
type: SIMPLE_ASSIGN
y[356] = r_init[356] * sin(theta[356] + 0.00424)
*/
void SpiralGalaxy_eqFunction_5684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5684};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2355]] /* y[356] STATE(1,vy[356]) */) = ((data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */)) * (sin((data->simulationInfo->realParameter[1862] /* theta[356] PARAM */) + 0.00424));
  TRACE_POP
}

/*
equation index: 5685
type: SIMPLE_ASSIGN
x[356] = r_init[356] * cos(theta[356] + 0.00424)
*/
void SpiralGalaxy_eqFunction_5685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1855]] /* x[356] STATE(1,vx[356]) */) = ((data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */)) * (cos((data->simulationInfo->realParameter[1862] /* theta[356] PARAM */) + 0.00424));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11556(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11557(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11560(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11559(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11558(DATA *data, threadData_t *threadData);


/*
equation index: 5691
type: SIMPLE_ASSIGN
vx[356] = (-sin(theta[356])) * r_init[356] * omega_c[356]
*/
void SpiralGalaxy_eqFunction_5691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5691};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[355]] /* vx[356] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1862] /* theta[356] PARAM */)))) * (((data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */)) * ((data->simulationInfo->realParameter[860] /* omega_c[356] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11553(DATA *data, threadData_t *threadData);


/*
equation index: 5693
type: SIMPLE_ASSIGN
vy[356] = cos(theta[356]) * r_init[356] * omega_c[356]
*/
void SpiralGalaxy_eqFunction_5693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[855]] /* vy[356] STATE(1) */) = (cos((data->simulationInfo->realParameter[1862] /* theta[356] PARAM */))) * (((data->simulationInfo->realParameter[1361] /* r_init[356] PARAM */)) * ((data->simulationInfo->realParameter[860] /* omega_c[356] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11552(DATA *data, threadData_t *threadData);


/*
equation index: 5695
type: SIMPLE_ASSIGN
vz[356] = 0.0
*/
void SpiralGalaxy_eqFunction_5695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5695};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1355]] /* vz[356] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11551(DATA *data, threadData_t *threadData);


/*
equation index: 5697
type: SIMPLE_ASSIGN
z[357] = 0.017120000000000003
*/
void SpiralGalaxy_eqFunction_5697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5697};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2856]] /* z[357] STATE(1,vz[357]) */) = 0.017120000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11564(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11565(DATA *data, threadData_t *threadData);


/*
equation index: 5700
type: SIMPLE_ASSIGN
y[357] = r_init[357] * sin(theta[357] + 0.004279999999999999)
*/
void SpiralGalaxy_eqFunction_5700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5700};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2356]] /* y[357] STATE(1,vy[357]) */) = ((data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */)) * (sin((data->simulationInfo->realParameter[1863] /* theta[357] PARAM */) + 0.004279999999999999));
  TRACE_POP
}

/*
equation index: 5701
type: SIMPLE_ASSIGN
x[357] = r_init[357] * cos(theta[357] + 0.004279999999999999)
*/
void SpiralGalaxy_eqFunction_5701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1856]] /* x[357] STATE(1,vx[357]) */) = ((data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */)) * (cos((data->simulationInfo->realParameter[1863] /* theta[357] PARAM */) + 0.004279999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11566(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11567(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11570(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11569(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11568(DATA *data, threadData_t *threadData);


/*
equation index: 5707
type: SIMPLE_ASSIGN
vx[357] = (-sin(theta[357])) * r_init[357] * omega_c[357]
*/
void SpiralGalaxy_eqFunction_5707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5707};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[356]] /* vx[357] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1863] /* theta[357] PARAM */)))) * (((data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */)) * ((data->simulationInfo->realParameter[861] /* omega_c[357] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11563(DATA *data, threadData_t *threadData);


/*
equation index: 5709
type: SIMPLE_ASSIGN
vy[357] = cos(theta[357]) * r_init[357] * omega_c[357]
*/
void SpiralGalaxy_eqFunction_5709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[856]] /* vy[357] STATE(1) */) = (cos((data->simulationInfo->realParameter[1863] /* theta[357] PARAM */))) * (((data->simulationInfo->realParameter[1362] /* r_init[357] PARAM */)) * ((data->simulationInfo->realParameter[861] /* omega_c[357] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11562(DATA *data, threadData_t *threadData);


/*
equation index: 5711
type: SIMPLE_ASSIGN
vz[357] = 0.0
*/
void SpiralGalaxy_eqFunction_5711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5711};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1356]] /* vz[357] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11561(DATA *data, threadData_t *threadData);


/*
equation index: 5713
type: SIMPLE_ASSIGN
z[358] = 0.01728
*/
void SpiralGalaxy_eqFunction_5713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5713};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2857]] /* z[358] STATE(1,vz[358]) */) = 0.01728;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11574(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11575(DATA *data, threadData_t *threadData);


/*
equation index: 5716
type: SIMPLE_ASSIGN
y[358] = r_init[358] * sin(theta[358] + 0.004319999999999999)
*/
void SpiralGalaxy_eqFunction_5716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5716};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2357]] /* y[358] STATE(1,vy[358]) */) = ((data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */)) * (sin((data->simulationInfo->realParameter[1864] /* theta[358] PARAM */) + 0.004319999999999999));
  TRACE_POP
}

/*
equation index: 5717
type: SIMPLE_ASSIGN
x[358] = r_init[358] * cos(theta[358] + 0.004319999999999999)
*/
void SpiralGalaxy_eqFunction_5717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1857]] /* x[358] STATE(1,vx[358]) */) = ((data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */)) * (cos((data->simulationInfo->realParameter[1864] /* theta[358] PARAM */) + 0.004319999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11576(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11577(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11580(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11579(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11578(DATA *data, threadData_t *threadData);


/*
equation index: 5723
type: SIMPLE_ASSIGN
vx[358] = (-sin(theta[358])) * r_init[358] * omega_c[358]
*/
void SpiralGalaxy_eqFunction_5723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5723};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[357]] /* vx[358] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1864] /* theta[358] PARAM */)))) * (((data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */)) * ((data->simulationInfo->realParameter[862] /* omega_c[358] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11573(DATA *data, threadData_t *threadData);


/*
equation index: 5725
type: SIMPLE_ASSIGN
vy[358] = cos(theta[358]) * r_init[358] * omega_c[358]
*/
void SpiralGalaxy_eqFunction_5725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[857]] /* vy[358] STATE(1) */) = (cos((data->simulationInfo->realParameter[1864] /* theta[358] PARAM */))) * (((data->simulationInfo->realParameter[1363] /* r_init[358] PARAM */)) * ((data->simulationInfo->realParameter[862] /* omega_c[358] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11572(DATA *data, threadData_t *threadData);


/*
equation index: 5727
type: SIMPLE_ASSIGN
vz[358] = 0.0
*/
void SpiralGalaxy_eqFunction_5727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5727};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1357]] /* vz[358] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11571(DATA *data, threadData_t *threadData);


/*
equation index: 5729
type: SIMPLE_ASSIGN
z[359] = 0.01744
*/
void SpiralGalaxy_eqFunction_5729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5729};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2858]] /* z[359] STATE(1,vz[359]) */) = 0.01744;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11584(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11585(DATA *data, threadData_t *threadData);


/*
equation index: 5732
type: SIMPLE_ASSIGN
y[359] = r_init[359] * sin(theta[359] + 0.004359999999999999)
*/
void SpiralGalaxy_eqFunction_5732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5732};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2358]] /* y[359] STATE(1,vy[359]) */) = ((data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */)) * (sin((data->simulationInfo->realParameter[1865] /* theta[359] PARAM */) + 0.004359999999999999));
  TRACE_POP
}

/*
equation index: 5733
type: SIMPLE_ASSIGN
x[359] = r_init[359] * cos(theta[359] + 0.004359999999999999)
*/
void SpiralGalaxy_eqFunction_5733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1858]] /* x[359] STATE(1,vx[359]) */) = ((data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */)) * (cos((data->simulationInfo->realParameter[1865] /* theta[359] PARAM */) + 0.004359999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11586(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11587(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11590(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11589(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11588(DATA *data, threadData_t *threadData);


/*
equation index: 5739
type: SIMPLE_ASSIGN
vx[359] = (-sin(theta[359])) * r_init[359] * omega_c[359]
*/
void SpiralGalaxy_eqFunction_5739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5739};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[358]] /* vx[359] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1865] /* theta[359] PARAM */)))) * (((data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */)) * ((data->simulationInfo->realParameter[863] /* omega_c[359] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11583(DATA *data, threadData_t *threadData);


/*
equation index: 5741
type: SIMPLE_ASSIGN
vy[359] = cos(theta[359]) * r_init[359] * omega_c[359]
*/
void SpiralGalaxy_eqFunction_5741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[858]] /* vy[359] STATE(1) */) = (cos((data->simulationInfo->realParameter[1865] /* theta[359] PARAM */))) * (((data->simulationInfo->realParameter[1364] /* r_init[359] PARAM */)) * ((data->simulationInfo->realParameter[863] /* omega_c[359] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11582(DATA *data, threadData_t *threadData);


/*
equation index: 5743
type: SIMPLE_ASSIGN
vz[359] = 0.0
*/
void SpiralGalaxy_eqFunction_5743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5743};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1358]] /* vz[359] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11581(DATA *data, threadData_t *threadData);


/*
equation index: 5745
type: SIMPLE_ASSIGN
z[360] = 0.0176
*/
void SpiralGalaxy_eqFunction_5745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5745};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2859]] /* z[360] STATE(1,vz[360]) */) = 0.0176;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11594(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11595(DATA *data, threadData_t *threadData);


/*
equation index: 5748
type: SIMPLE_ASSIGN
y[360] = r_init[360] * sin(theta[360] + 0.004399999999999999)
*/
void SpiralGalaxy_eqFunction_5748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5748};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2359]] /* y[360] STATE(1,vy[360]) */) = ((data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */)) * (sin((data->simulationInfo->realParameter[1866] /* theta[360] PARAM */) + 0.004399999999999999));
  TRACE_POP
}

/*
equation index: 5749
type: SIMPLE_ASSIGN
x[360] = r_init[360] * cos(theta[360] + 0.004399999999999999)
*/
void SpiralGalaxy_eqFunction_5749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1859]] /* x[360] STATE(1,vx[360]) */) = ((data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */)) * (cos((data->simulationInfo->realParameter[1866] /* theta[360] PARAM */) + 0.004399999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11596(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11597(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11600(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11599(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11598(DATA *data, threadData_t *threadData);


/*
equation index: 5755
type: SIMPLE_ASSIGN
vx[360] = (-sin(theta[360])) * r_init[360] * omega_c[360]
*/
void SpiralGalaxy_eqFunction_5755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5755};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[359]] /* vx[360] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1866] /* theta[360] PARAM */)))) * (((data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */)) * ((data->simulationInfo->realParameter[864] /* omega_c[360] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11593(DATA *data, threadData_t *threadData);


/*
equation index: 5757
type: SIMPLE_ASSIGN
vy[360] = cos(theta[360]) * r_init[360] * omega_c[360]
*/
void SpiralGalaxy_eqFunction_5757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[859]] /* vy[360] STATE(1) */) = (cos((data->simulationInfo->realParameter[1866] /* theta[360] PARAM */))) * (((data->simulationInfo->realParameter[1365] /* r_init[360] PARAM */)) * ((data->simulationInfo->realParameter[864] /* omega_c[360] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11592(DATA *data, threadData_t *threadData);


/*
equation index: 5759
type: SIMPLE_ASSIGN
vz[360] = 0.0
*/
void SpiralGalaxy_eqFunction_5759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5759};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1359]] /* vz[360] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11591(DATA *data, threadData_t *threadData);


/*
equation index: 5761
type: SIMPLE_ASSIGN
z[361] = 0.01776
*/
void SpiralGalaxy_eqFunction_5761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5761};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2860]] /* z[361] STATE(1,vz[361]) */) = 0.01776;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11604(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11605(DATA *data, threadData_t *threadData);


/*
equation index: 5764
type: SIMPLE_ASSIGN
y[361] = r_init[361] * sin(theta[361] + 0.0044399999999999995)
*/
void SpiralGalaxy_eqFunction_5764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5764};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2360]] /* y[361] STATE(1,vy[361]) */) = ((data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */)) * (sin((data->simulationInfo->realParameter[1867] /* theta[361] PARAM */) + 0.0044399999999999995));
  TRACE_POP
}

/*
equation index: 5765
type: SIMPLE_ASSIGN
x[361] = r_init[361] * cos(theta[361] + 0.0044399999999999995)
*/
void SpiralGalaxy_eqFunction_5765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1860]] /* x[361] STATE(1,vx[361]) */) = ((data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */)) * (cos((data->simulationInfo->realParameter[1867] /* theta[361] PARAM */) + 0.0044399999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11606(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11607(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11610(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11609(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11608(DATA *data, threadData_t *threadData);


/*
equation index: 5771
type: SIMPLE_ASSIGN
vx[361] = (-sin(theta[361])) * r_init[361] * omega_c[361]
*/
void SpiralGalaxy_eqFunction_5771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5771};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[360]] /* vx[361] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1867] /* theta[361] PARAM */)))) * (((data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */)) * ((data->simulationInfo->realParameter[865] /* omega_c[361] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11603(DATA *data, threadData_t *threadData);


/*
equation index: 5773
type: SIMPLE_ASSIGN
vy[361] = cos(theta[361]) * r_init[361] * omega_c[361]
*/
void SpiralGalaxy_eqFunction_5773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[860]] /* vy[361] STATE(1) */) = (cos((data->simulationInfo->realParameter[1867] /* theta[361] PARAM */))) * (((data->simulationInfo->realParameter[1366] /* r_init[361] PARAM */)) * ((data->simulationInfo->realParameter[865] /* omega_c[361] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11602(DATA *data, threadData_t *threadData);


/*
equation index: 5775
type: SIMPLE_ASSIGN
vz[361] = 0.0
*/
void SpiralGalaxy_eqFunction_5775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5775};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1360]] /* vz[361] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11601(DATA *data, threadData_t *threadData);


/*
equation index: 5777
type: SIMPLE_ASSIGN
z[362] = 0.017920000000000002
*/
void SpiralGalaxy_eqFunction_5777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5777};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2861]] /* z[362] STATE(1,vz[362]) */) = 0.017920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11614(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11615(DATA *data, threadData_t *threadData);


/*
equation index: 5780
type: SIMPLE_ASSIGN
y[362] = r_init[362] * sin(theta[362] + 0.00448)
*/
void SpiralGalaxy_eqFunction_5780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5780};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2361]] /* y[362] STATE(1,vy[362]) */) = ((data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */)) * (sin((data->simulationInfo->realParameter[1868] /* theta[362] PARAM */) + 0.00448));
  TRACE_POP
}

/*
equation index: 5781
type: SIMPLE_ASSIGN
x[362] = r_init[362] * cos(theta[362] + 0.00448)
*/
void SpiralGalaxy_eqFunction_5781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1861]] /* x[362] STATE(1,vx[362]) */) = ((data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */)) * (cos((data->simulationInfo->realParameter[1868] /* theta[362] PARAM */) + 0.00448));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11616(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11617(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11620(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11619(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11618(DATA *data, threadData_t *threadData);


/*
equation index: 5787
type: SIMPLE_ASSIGN
vx[362] = (-sin(theta[362])) * r_init[362] * omega_c[362]
*/
void SpiralGalaxy_eqFunction_5787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5787};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[361]] /* vx[362] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1868] /* theta[362] PARAM */)))) * (((data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */)) * ((data->simulationInfo->realParameter[866] /* omega_c[362] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11613(DATA *data, threadData_t *threadData);


/*
equation index: 5789
type: SIMPLE_ASSIGN
vy[362] = cos(theta[362]) * r_init[362] * omega_c[362]
*/
void SpiralGalaxy_eqFunction_5789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[861]] /* vy[362] STATE(1) */) = (cos((data->simulationInfo->realParameter[1868] /* theta[362] PARAM */))) * (((data->simulationInfo->realParameter[1367] /* r_init[362] PARAM */)) * ((data->simulationInfo->realParameter[866] /* omega_c[362] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11612(DATA *data, threadData_t *threadData);


/*
equation index: 5791
type: SIMPLE_ASSIGN
vz[362] = 0.0
*/
void SpiralGalaxy_eqFunction_5791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5791};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1361]] /* vz[362] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11611(DATA *data, threadData_t *threadData);


/*
equation index: 5793
type: SIMPLE_ASSIGN
z[363] = 0.01808
*/
void SpiralGalaxy_eqFunction_5793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5793};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2862]] /* z[363] STATE(1,vz[363]) */) = 0.01808;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11624(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11625(DATA *data, threadData_t *threadData);


/*
equation index: 5796
type: SIMPLE_ASSIGN
y[363] = r_init[363] * sin(theta[363] + 0.00452)
*/
void SpiralGalaxy_eqFunction_5796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5796};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2362]] /* y[363] STATE(1,vy[363]) */) = ((data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */)) * (sin((data->simulationInfo->realParameter[1869] /* theta[363] PARAM */) + 0.00452));
  TRACE_POP
}

/*
equation index: 5797
type: SIMPLE_ASSIGN
x[363] = r_init[363] * cos(theta[363] + 0.00452)
*/
void SpiralGalaxy_eqFunction_5797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1862]] /* x[363] STATE(1,vx[363]) */) = ((data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */)) * (cos((data->simulationInfo->realParameter[1869] /* theta[363] PARAM */) + 0.00452));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11626(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11627(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11630(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11629(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11628(DATA *data, threadData_t *threadData);


/*
equation index: 5803
type: SIMPLE_ASSIGN
vx[363] = (-sin(theta[363])) * r_init[363] * omega_c[363]
*/
void SpiralGalaxy_eqFunction_5803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5803};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[362]] /* vx[363] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1869] /* theta[363] PARAM */)))) * (((data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */)) * ((data->simulationInfo->realParameter[867] /* omega_c[363] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11623(DATA *data, threadData_t *threadData);


/*
equation index: 5805
type: SIMPLE_ASSIGN
vy[363] = cos(theta[363]) * r_init[363] * omega_c[363]
*/
void SpiralGalaxy_eqFunction_5805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[862]] /* vy[363] STATE(1) */) = (cos((data->simulationInfo->realParameter[1869] /* theta[363] PARAM */))) * (((data->simulationInfo->realParameter[1368] /* r_init[363] PARAM */)) * ((data->simulationInfo->realParameter[867] /* omega_c[363] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11622(DATA *data, threadData_t *threadData);


/*
equation index: 5807
type: SIMPLE_ASSIGN
vz[363] = 0.0
*/
void SpiralGalaxy_eqFunction_5807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5807};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1362]] /* vz[363] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11621(DATA *data, threadData_t *threadData);


/*
equation index: 5809
type: SIMPLE_ASSIGN
z[364] = 0.018240000000000003
*/
void SpiralGalaxy_eqFunction_5809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5809};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2863]] /* z[364] STATE(1,vz[364]) */) = 0.018240000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11634(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11635(DATA *data, threadData_t *threadData);


/*
equation index: 5812
type: SIMPLE_ASSIGN
y[364] = r_init[364] * sin(theta[364] + 0.00456)
*/
void SpiralGalaxy_eqFunction_5812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5812};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2363]] /* y[364] STATE(1,vy[364]) */) = ((data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */)) * (sin((data->simulationInfo->realParameter[1870] /* theta[364] PARAM */) + 0.00456));
  TRACE_POP
}

/*
equation index: 5813
type: SIMPLE_ASSIGN
x[364] = r_init[364] * cos(theta[364] + 0.00456)
*/
void SpiralGalaxy_eqFunction_5813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1863]] /* x[364] STATE(1,vx[364]) */) = ((data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */)) * (cos((data->simulationInfo->realParameter[1870] /* theta[364] PARAM */) + 0.00456));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11636(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11637(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11640(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11639(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11638(DATA *data, threadData_t *threadData);


/*
equation index: 5819
type: SIMPLE_ASSIGN
vx[364] = (-sin(theta[364])) * r_init[364] * omega_c[364]
*/
void SpiralGalaxy_eqFunction_5819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5819};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[363]] /* vx[364] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1870] /* theta[364] PARAM */)))) * (((data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */)) * ((data->simulationInfo->realParameter[868] /* omega_c[364] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11633(DATA *data, threadData_t *threadData);


/*
equation index: 5821
type: SIMPLE_ASSIGN
vy[364] = cos(theta[364]) * r_init[364] * omega_c[364]
*/
void SpiralGalaxy_eqFunction_5821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[863]] /* vy[364] STATE(1) */) = (cos((data->simulationInfo->realParameter[1870] /* theta[364] PARAM */))) * (((data->simulationInfo->realParameter[1369] /* r_init[364] PARAM */)) * ((data->simulationInfo->realParameter[868] /* omega_c[364] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11632(DATA *data, threadData_t *threadData);


/*
equation index: 5823
type: SIMPLE_ASSIGN
vz[364] = 0.0
*/
void SpiralGalaxy_eqFunction_5823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5823};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1363]] /* vz[364] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11631(DATA *data, threadData_t *threadData);


/*
equation index: 5825
type: SIMPLE_ASSIGN
z[365] = 0.0184
*/
void SpiralGalaxy_eqFunction_5825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5825};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2864]] /* z[365] STATE(1,vz[365]) */) = 0.0184;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11644(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11645(DATA *data, threadData_t *threadData);


/*
equation index: 5828
type: SIMPLE_ASSIGN
y[365] = r_init[365] * sin(theta[365] + 0.0046)
*/
void SpiralGalaxy_eqFunction_5828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5828};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2364]] /* y[365] STATE(1,vy[365]) */) = ((data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */)) * (sin((data->simulationInfo->realParameter[1871] /* theta[365] PARAM */) + 0.0046));
  TRACE_POP
}

/*
equation index: 5829
type: SIMPLE_ASSIGN
x[365] = r_init[365] * cos(theta[365] + 0.0046)
*/
void SpiralGalaxy_eqFunction_5829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1864]] /* x[365] STATE(1,vx[365]) */) = ((data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */)) * (cos((data->simulationInfo->realParameter[1871] /* theta[365] PARAM */) + 0.0046));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11646(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11647(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11650(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11649(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11648(DATA *data, threadData_t *threadData);


/*
equation index: 5835
type: SIMPLE_ASSIGN
vx[365] = (-sin(theta[365])) * r_init[365] * omega_c[365]
*/
void SpiralGalaxy_eqFunction_5835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5835};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[364]] /* vx[365] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1871] /* theta[365] PARAM */)))) * (((data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */)) * ((data->simulationInfo->realParameter[869] /* omega_c[365] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11643(DATA *data, threadData_t *threadData);


/*
equation index: 5837
type: SIMPLE_ASSIGN
vy[365] = cos(theta[365]) * r_init[365] * omega_c[365]
*/
void SpiralGalaxy_eqFunction_5837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[864]] /* vy[365] STATE(1) */) = (cos((data->simulationInfo->realParameter[1871] /* theta[365] PARAM */))) * (((data->simulationInfo->realParameter[1370] /* r_init[365] PARAM */)) * ((data->simulationInfo->realParameter[869] /* omega_c[365] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11642(DATA *data, threadData_t *threadData);


/*
equation index: 5839
type: SIMPLE_ASSIGN
vz[365] = 0.0
*/
void SpiralGalaxy_eqFunction_5839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5839};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1364]] /* vz[365] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11641(DATA *data, threadData_t *threadData);


/*
equation index: 5841
type: SIMPLE_ASSIGN
z[366] = 0.01856
*/
void SpiralGalaxy_eqFunction_5841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5841};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2865]] /* z[366] STATE(1,vz[366]) */) = 0.01856;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11654(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11655(DATA *data, threadData_t *threadData);


/*
equation index: 5844
type: SIMPLE_ASSIGN
y[366] = r_init[366] * sin(theta[366] + 0.00464)
*/
void SpiralGalaxy_eqFunction_5844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5844};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2365]] /* y[366] STATE(1,vy[366]) */) = ((data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */)) * (sin((data->simulationInfo->realParameter[1872] /* theta[366] PARAM */) + 0.00464));
  TRACE_POP
}

/*
equation index: 5845
type: SIMPLE_ASSIGN
x[366] = r_init[366] * cos(theta[366] + 0.00464)
*/
void SpiralGalaxy_eqFunction_5845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1865]] /* x[366] STATE(1,vx[366]) */) = ((data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */)) * (cos((data->simulationInfo->realParameter[1872] /* theta[366] PARAM */) + 0.00464));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11656(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11657(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11660(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11659(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11658(DATA *data, threadData_t *threadData);


/*
equation index: 5851
type: SIMPLE_ASSIGN
vx[366] = (-sin(theta[366])) * r_init[366] * omega_c[366]
*/
void SpiralGalaxy_eqFunction_5851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5851};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[365]] /* vx[366] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1872] /* theta[366] PARAM */)))) * (((data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */)) * ((data->simulationInfo->realParameter[870] /* omega_c[366] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11653(DATA *data, threadData_t *threadData);


/*
equation index: 5853
type: SIMPLE_ASSIGN
vy[366] = cos(theta[366]) * r_init[366] * omega_c[366]
*/
void SpiralGalaxy_eqFunction_5853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[865]] /* vy[366] STATE(1) */) = (cos((data->simulationInfo->realParameter[1872] /* theta[366] PARAM */))) * (((data->simulationInfo->realParameter[1371] /* r_init[366] PARAM */)) * ((data->simulationInfo->realParameter[870] /* omega_c[366] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11652(DATA *data, threadData_t *threadData);


/*
equation index: 5855
type: SIMPLE_ASSIGN
vz[366] = 0.0
*/
void SpiralGalaxy_eqFunction_5855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5855};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1365]] /* vz[366] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11651(DATA *data, threadData_t *threadData);


/*
equation index: 5857
type: SIMPLE_ASSIGN
z[367] = 0.01872
*/
void SpiralGalaxy_eqFunction_5857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5857};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2866]] /* z[367] STATE(1,vz[367]) */) = 0.01872;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11664(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11665(DATA *data, threadData_t *threadData);


/*
equation index: 5860
type: SIMPLE_ASSIGN
y[367] = r_init[367] * sin(theta[367] + 0.00468)
*/
void SpiralGalaxy_eqFunction_5860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5860};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2366]] /* y[367] STATE(1,vy[367]) */) = ((data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */)) * (sin((data->simulationInfo->realParameter[1873] /* theta[367] PARAM */) + 0.00468));
  TRACE_POP
}

/*
equation index: 5861
type: SIMPLE_ASSIGN
x[367] = r_init[367] * cos(theta[367] + 0.00468)
*/
void SpiralGalaxy_eqFunction_5861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1866]] /* x[367] STATE(1,vx[367]) */) = ((data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */)) * (cos((data->simulationInfo->realParameter[1873] /* theta[367] PARAM */) + 0.00468));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11666(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11667(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11670(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11669(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11668(DATA *data, threadData_t *threadData);


/*
equation index: 5867
type: SIMPLE_ASSIGN
vx[367] = (-sin(theta[367])) * r_init[367] * omega_c[367]
*/
void SpiralGalaxy_eqFunction_5867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5867};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[366]] /* vx[367] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1873] /* theta[367] PARAM */)))) * (((data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */)) * ((data->simulationInfo->realParameter[871] /* omega_c[367] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11663(DATA *data, threadData_t *threadData);


/*
equation index: 5869
type: SIMPLE_ASSIGN
vy[367] = cos(theta[367]) * r_init[367] * omega_c[367]
*/
void SpiralGalaxy_eqFunction_5869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[866]] /* vy[367] STATE(1) */) = (cos((data->simulationInfo->realParameter[1873] /* theta[367] PARAM */))) * (((data->simulationInfo->realParameter[1372] /* r_init[367] PARAM */)) * ((data->simulationInfo->realParameter[871] /* omega_c[367] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11662(DATA *data, threadData_t *threadData);


/*
equation index: 5871
type: SIMPLE_ASSIGN
vz[367] = 0.0
*/
void SpiralGalaxy_eqFunction_5871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5871};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1366]] /* vz[367] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11661(DATA *data, threadData_t *threadData);


/*
equation index: 5873
type: SIMPLE_ASSIGN
z[368] = 0.018880000000000004
*/
void SpiralGalaxy_eqFunction_5873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5873};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2867]] /* z[368] STATE(1,vz[368]) */) = 0.018880000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11674(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11675(DATA *data, threadData_t *threadData);


/*
equation index: 5876
type: SIMPLE_ASSIGN
y[368] = r_init[368] * sin(theta[368] + 0.00472)
*/
void SpiralGalaxy_eqFunction_5876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5876};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2367]] /* y[368] STATE(1,vy[368]) */) = ((data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */)) * (sin((data->simulationInfo->realParameter[1874] /* theta[368] PARAM */) + 0.00472));
  TRACE_POP
}

/*
equation index: 5877
type: SIMPLE_ASSIGN
x[368] = r_init[368] * cos(theta[368] + 0.00472)
*/
void SpiralGalaxy_eqFunction_5877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1867]] /* x[368] STATE(1,vx[368]) */) = ((data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */)) * (cos((data->simulationInfo->realParameter[1874] /* theta[368] PARAM */) + 0.00472));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11676(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11677(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11680(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11679(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11678(DATA *data, threadData_t *threadData);


/*
equation index: 5883
type: SIMPLE_ASSIGN
vx[368] = (-sin(theta[368])) * r_init[368] * omega_c[368]
*/
void SpiralGalaxy_eqFunction_5883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5883};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[367]] /* vx[368] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1874] /* theta[368] PARAM */)))) * (((data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */)) * ((data->simulationInfo->realParameter[872] /* omega_c[368] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11673(DATA *data, threadData_t *threadData);


/*
equation index: 5885
type: SIMPLE_ASSIGN
vy[368] = cos(theta[368]) * r_init[368] * omega_c[368]
*/
void SpiralGalaxy_eqFunction_5885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[867]] /* vy[368] STATE(1) */) = (cos((data->simulationInfo->realParameter[1874] /* theta[368] PARAM */))) * (((data->simulationInfo->realParameter[1373] /* r_init[368] PARAM */)) * ((data->simulationInfo->realParameter[872] /* omega_c[368] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11672(DATA *data, threadData_t *threadData);


/*
equation index: 5887
type: SIMPLE_ASSIGN
vz[368] = 0.0
*/
void SpiralGalaxy_eqFunction_5887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5887};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1367]] /* vz[368] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11671(DATA *data, threadData_t *threadData);


/*
equation index: 5889
type: SIMPLE_ASSIGN
z[369] = 0.01904
*/
void SpiralGalaxy_eqFunction_5889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5889};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2868]] /* z[369] STATE(1,vz[369]) */) = 0.01904;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11684(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11685(DATA *data, threadData_t *threadData);


/*
equation index: 5892
type: SIMPLE_ASSIGN
y[369] = r_init[369] * sin(theta[369] + 0.0047599999999999995)
*/
void SpiralGalaxy_eqFunction_5892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5892};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2368]] /* y[369] STATE(1,vy[369]) */) = ((data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */)) * (sin((data->simulationInfo->realParameter[1875] /* theta[369] PARAM */) + 0.0047599999999999995));
  TRACE_POP
}

/*
equation index: 5893
type: SIMPLE_ASSIGN
x[369] = r_init[369] * cos(theta[369] + 0.0047599999999999995)
*/
void SpiralGalaxy_eqFunction_5893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1868]] /* x[369] STATE(1,vx[369]) */) = ((data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */)) * (cos((data->simulationInfo->realParameter[1875] /* theta[369] PARAM */) + 0.0047599999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11686(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11687(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11690(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11689(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11688(DATA *data, threadData_t *threadData);


/*
equation index: 5899
type: SIMPLE_ASSIGN
vx[369] = (-sin(theta[369])) * r_init[369] * omega_c[369]
*/
void SpiralGalaxy_eqFunction_5899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5899};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[368]] /* vx[369] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1875] /* theta[369] PARAM */)))) * (((data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */)) * ((data->simulationInfo->realParameter[873] /* omega_c[369] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11683(DATA *data, threadData_t *threadData);


/*
equation index: 5901
type: SIMPLE_ASSIGN
vy[369] = cos(theta[369]) * r_init[369] * omega_c[369]
*/
void SpiralGalaxy_eqFunction_5901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[868]] /* vy[369] STATE(1) */) = (cos((data->simulationInfo->realParameter[1875] /* theta[369] PARAM */))) * (((data->simulationInfo->realParameter[1374] /* r_init[369] PARAM */)) * ((data->simulationInfo->realParameter[873] /* omega_c[369] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11682(DATA *data, threadData_t *threadData);


/*
equation index: 5903
type: SIMPLE_ASSIGN
vz[369] = 0.0
*/
void SpiralGalaxy_eqFunction_5903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5903};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1368]] /* vz[369] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11681(DATA *data, threadData_t *threadData);


/*
equation index: 5905
type: SIMPLE_ASSIGN
z[370] = 0.019200000000000002
*/
void SpiralGalaxy_eqFunction_5905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5905};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2869]] /* z[370] STATE(1,vz[370]) */) = 0.019200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11694(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11695(DATA *data, threadData_t *threadData);


/*
equation index: 5908
type: SIMPLE_ASSIGN
y[370] = r_init[370] * sin(theta[370] + 0.0048)
*/
void SpiralGalaxy_eqFunction_5908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5908};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2369]] /* y[370] STATE(1,vy[370]) */) = ((data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */)) * (sin((data->simulationInfo->realParameter[1876] /* theta[370] PARAM */) + 0.0048));
  TRACE_POP
}

/*
equation index: 5909
type: SIMPLE_ASSIGN
x[370] = r_init[370] * cos(theta[370] + 0.0048)
*/
void SpiralGalaxy_eqFunction_5909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1869]] /* x[370] STATE(1,vx[370]) */) = ((data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */)) * (cos((data->simulationInfo->realParameter[1876] /* theta[370] PARAM */) + 0.0048));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11696(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11697(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11700(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11699(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11698(DATA *data, threadData_t *threadData);


/*
equation index: 5915
type: SIMPLE_ASSIGN
vx[370] = (-sin(theta[370])) * r_init[370] * omega_c[370]
*/
void SpiralGalaxy_eqFunction_5915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5915};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[369]] /* vx[370] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1876] /* theta[370] PARAM */)))) * (((data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */)) * ((data->simulationInfo->realParameter[874] /* omega_c[370] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11693(DATA *data, threadData_t *threadData);


/*
equation index: 5917
type: SIMPLE_ASSIGN
vy[370] = cos(theta[370]) * r_init[370] * omega_c[370]
*/
void SpiralGalaxy_eqFunction_5917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[869]] /* vy[370] STATE(1) */) = (cos((data->simulationInfo->realParameter[1876] /* theta[370] PARAM */))) * (((data->simulationInfo->realParameter[1375] /* r_init[370] PARAM */)) * ((data->simulationInfo->realParameter[874] /* omega_c[370] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11692(DATA *data, threadData_t *threadData);


/*
equation index: 5919
type: SIMPLE_ASSIGN
vz[370] = 0.0
*/
void SpiralGalaxy_eqFunction_5919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5919};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1369]] /* vz[370] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11691(DATA *data, threadData_t *threadData);


/*
equation index: 5921
type: SIMPLE_ASSIGN
z[371] = 0.01936
*/
void SpiralGalaxy_eqFunction_5921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5921};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2870]] /* z[371] STATE(1,vz[371]) */) = 0.01936;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11704(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11705(DATA *data, threadData_t *threadData);


/*
equation index: 5924
type: SIMPLE_ASSIGN
y[371] = r_init[371] * sin(theta[371] + 0.00484)
*/
void SpiralGalaxy_eqFunction_5924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5924};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2370]] /* y[371] STATE(1,vy[371]) */) = ((data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */)) * (sin((data->simulationInfo->realParameter[1877] /* theta[371] PARAM */) + 0.00484));
  TRACE_POP
}

/*
equation index: 5925
type: SIMPLE_ASSIGN
x[371] = r_init[371] * cos(theta[371] + 0.00484)
*/
void SpiralGalaxy_eqFunction_5925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1870]] /* x[371] STATE(1,vx[371]) */) = ((data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */)) * (cos((data->simulationInfo->realParameter[1877] /* theta[371] PARAM */) + 0.00484));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11706(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11707(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11710(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11709(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11708(DATA *data, threadData_t *threadData);


/*
equation index: 5931
type: SIMPLE_ASSIGN
vx[371] = (-sin(theta[371])) * r_init[371] * omega_c[371]
*/
void SpiralGalaxy_eqFunction_5931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5931};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[370]] /* vx[371] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1877] /* theta[371] PARAM */)))) * (((data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */)) * ((data->simulationInfo->realParameter[875] /* omega_c[371] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11703(DATA *data, threadData_t *threadData);


/*
equation index: 5933
type: SIMPLE_ASSIGN
vy[371] = cos(theta[371]) * r_init[371] * omega_c[371]
*/
void SpiralGalaxy_eqFunction_5933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[870]] /* vy[371] STATE(1) */) = (cos((data->simulationInfo->realParameter[1877] /* theta[371] PARAM */))) * (((data->simulationInfo->realParameter[1376] /* r_init[371] PARAM */)) * ((data->simulationInfo->realParameter[875] /* omega_c[371] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11702(DATA *data, threadData_t *threadData);


/*
equation index: 5935
type: SIMPLE_ASSIGN
vz[371] = 0.0
*/
void SpiralGalaxy_eqFunction_5935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5935};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1370]] /* vz[371] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11701(DATA *data, threadData_t *threadData);


/*
equation index: 5937
type: SIMPLE_ASSIGN
z[372] = 0.019520000000000003
*/
void SpiralGalaxy_eqFunction_5937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5937};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2871]] /* z[372] STATE(1,vz[372]) */) = 0.019520000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11714(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11715(DATA *data, threadData_t *threadData);


/*
equation index: 5940
type: SIMPLE_ASSIGN
y[372] = r_init[372] * sin(theta[372] + 0.00488)
*/
void SpiralGalaxy_eqFunction_5940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5940};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2371]] /* y[372] STATE(1,vy[372]) */) = ((data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */)) * (sin((data->simulationInfo->realParameter[1878] /* theta[372] PARAM */) + 0.00488));
  TRACE_POP
}

/*
equation index: 5941
type: SIMPLE_ASSIGN
x[372] = r_init[372] * cos(theta[372] + 0.00488)
*/
void SpiralGalaxy_eqFunction_5941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1871]] /* x[372] STATE(1,vx[372]) */) = ((data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */)) * (cos((data->simulationInfo->realParameter[1878] /* theta[372] PARAM */) + 0.00488));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11716(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11717(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11720(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11719(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11718(DATA *data, threadData_t *threadData);


/*
equation index: 5947
type: SIMPLE_ASSIGN
vx[372] = (-sin(theta[372])) * r_init[372] * omega_c[372]
*/
void SpiralGalaxy_eqFunction_5947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5947};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[371]] /* vx[372] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1878] /* theta[372] PARAM */)))) * (((data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */)) * ((data->simulationInfo->realParameter[876] /* omega_c[372] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11713(DATA *data, threadData_t *threadData);


/*
equation index: 5949
type: SIMPLE_ASSIGN
vy[372] = cos(theta[372]) * r_init[372] * omega_c[372]
*/
void SpiralGalaxy_eqFunction_5949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[871]] /* vy[372] STATE(1) */) = (cos((data->simulationInfo->realParameter[1878] /* theta[372] PARAM */))) * (((data->simulationInfo->realParameter[1377] /* r_init[372] PARAM */)) * ((data->simulationInfo->realParameter[876] /* omega_c[372] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11712(DATA *data, threadData_t *threadData);


/*
equation index: 5951
type: SIMPLE_ASSIGN
vz[372] = 0.0
*/
void SpiralGalaxy_eqFunction_5951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5951};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1371]] /* vz[372] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11711(DATA *data, threadData_t *threadData);


/*
equation index: 5953
type: SIMPLE_ASSIGN
z[373] = 0.019680000000000003
*/
void SpiralGalaxy_eqFunction_5953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5953};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2872]] /* z[373] STATE(1,vz[373]) */) = 0.019680000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11724(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11725(DATA *data, threadData_t *threadData);


/*
equation index: 5956
type: SIMPLE_ASSIGN
y[373] = r_init[373] * sin(theta[373] + 0.00492)
*/
void SpiralGalaxy_eqFunction_5956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5956};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2372]] /* y[373] STATE(1,vy[373]) */) = ((data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */)) * (sin((data->simulationInfo->realParameter[1879] /* theta[373] PARAM */) + 0.00492));
  TRACE_POP
}

/*
equation index: 5957
type: SIMPLE_ASSIGN
x[373] = r_init[373] * cos(theta[373] + 0.00492)
*/
void SpiralGalaxy_eqFunction_5957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1872]] /* x[373] STATE(1,vx[373]) */) = ((data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */)) * (cos((data->simulationInfo->realParameter[1879] /* theta[373] PARAM */) + 0.00492));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11726(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11727(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11730(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11729(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11728(DATA *data, threadData_t *threadData);


/*
equation index: 5963
type: SIMPLE_ASSIGN
vx[373] = (-sin(theta[373])) * r_init[373] * omega_c[373]
*/
void SpiralGalaxy_eqFunction_5963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5963};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[372]] /* vx[373] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1879] /* theta[373] PARAM */)))) * (((data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */)) * ((data->simulationInfo->realParameter[877] /* omega_c[373] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11723(DATA *data, threadData_t *threadData);


/*
equation index: 5965
type: SIMPLE_ASSIGN
vy[373] = cos(theta[373]) * r_init[373] * omega_c[373]
*/
void SpiralGalaxy_eqFunction_5965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[872]] /* vy[373] STATE(1) */) = (cos((data->simulationInfo->realParameter[1879] /* theta[373] PARAM */))) * (((data->simulationInfo->realParameter[1378] /* r_init[373] PARAM */)) * ((data->simulationInfo->realParameter[877] /* omega_c[373] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11722(DATA *data, threadData_t *threadData);


/*
equation index: 5967
type: SIMPLE_ASSIGN
vz[373] = 0.0
*/
void SpiralGalaxy_eqFunction_5967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5967};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1372]] /* vz[373] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11721(DATA *data, threadData_t *threadData);


/*
equation index: 5969
type: SIMPLE_ASSIGN
z[374] = 0.019840000000000003
*/
void SpiralGalaxy_eqFunction_5969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5969};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2873]] /* z[374] STATE(1,vz[374]) */) = 0.019840000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11734(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11735(DATA *data, threadData_t *threadData);


/*
equation index: 5972
type: SIMPLE_ASSIGN
y[374] = r_init[374] * sin(theta[374] + 0.00496)
*/
void SpiralGalaxy_eqFunction_5972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5972};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2373]] /* y[374] STATE(1,vy[374]) */) = ((data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */)) * (sin((data->simulationInfo->realParameter[1880] /* theta[374] PARAM */) + 0.00496));
  TRACE_POP
}

/*
equation index: 5973
type: SIMPLE_ASSIGN
x[374] = r_init[374] * cos(theta[374] + 0.00496)
*/
void SpiralGalaxy_eqFunction_5973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1873]] /* x[374] STATE(1,vx[374]) */) = ((data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */)) * (cos((data->simulationInfo->realParameter[1880] /* theta[374] PARAM */) + 0.00496));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11736(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11737(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11740(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11739(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11738(DATA *data, threadData_t *threadData);


/*
equation index: 5979
type: SIMPLE_ASSIGN
vx[374] = (-sin(theta[374])) * r_init[374] * omega_c[374]
*/
void SpiralGalaxy_eqFunction_5979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5979};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[373]] /* vx[374] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1880] /* theta[374] PARAM */)))) * (((data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */)) * ((data->simulationInfo->realParameter[878] /* omega_c[374] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11733(DATA *data, threadData_t *threadData);


/*
equation index: 5981
type: SIMPLE_ASSIGN
vy[374] = cos(theta[374]) * r_init[374] * omega_c[374]
*/
void SpiralGalaxy_eqFunction_5981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[873]] /* vy[374] STATE(1) */) = (cos((data->simulationInfo->realParameter[1880] /* theta[374] PARAM */))) * (((data->simulationInfo->realParameter[1379] /* r_init[374] PARAM */)) * ((data->simulationInfo->realParameter[878] /* omega_c[374] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11732(DATA *data, threadData_t *threadData);


/*
equation index: 5983
type: SIMPLE_ASSIGN
vz[374] = 0.0
*/
void SpiralGalaxy_eqFunction_5983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5983};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1373]] /* vz[374] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11731(DATA *data, threadData_t *threadData);


/*
equation index: 5985
type: SIMPLE_ASSIGN
z[375] = 0.02
*/
void SpiralGalaxy_eqFunction_5985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5985};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2874]] /* z[375] STATE(1,vz[375]) */) = 0.02;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11744(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11745(DATA *data, threadData_t *threadData);


/*
equation index: 5988
type: SIMPLE_ASSIGN
y[375] = r_init[375] * sin(theta[375] + 0.005)
*/
void SpiralGalaxy_eqFunction_5988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5988};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2374]] /* y[375] STATE(1,vy[375]) */) = ((data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */)) * (sin((data->simulationInfo->realParameter[1881] /* theta[375] PARAM */) + 0.005));
  TRACE_POP
}

/*
equation index: 5989
type: SIMPLE_ASSIGN
x[375] = r_init[375] * cos(theta[375] + 0.005)
*/
void SpiralGalaxy_eqFunction_5989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1874]] /* x[375] STATE(1,vx[375]) */) = ((data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */)) * (cos((data->simulationInfo->realParameter[1881] /* theta[375] PARAM */) + 0.005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11746(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11747(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11750(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11749(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11748(DATA *data, threadData_t *threadData);


/*
equation index: 5995
type: SIMPLE_ASSIGN
vx[375] = (-sin(theta[375])) * r_init[375] * omega_c[375]
*/
void SpiralGalaxy_eqFunction_5995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5995};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[374]] /* vx[375] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1881] /* theta[375] PARAM */)))) * (((data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */)) * ((data->simulationInfo->realParameter[879] /* omega_c[375] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11743(DATA *data, threadData_t *threadData);


/*
equation index: 5997
type: SIMPLE_ASSIGN
vy[375] = cos(theta[375]) * r_init[375] * omega_c[375]
*/
void SpiralGalaxy_eqFunction_5997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[874]] /* vy[375] STATE(1) */) = (cos((data->simulationInfo->realParameter[1881] /* theta[375] PARAM */))) * (((data->simulationInfo->realParameter[1380] /* r_init[375] PARAM */)) * ((data->simulationInfo->realParameter[879] /* omega_c[375] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11742(DATA *data, threadData_t *threadData);


/*
equation index: 5999
type: SIMPLE_ASSIGN
vz[375] = 0.0
*/
void SpiralGalaxy_eqFunction_5999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5999};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1374]] /* vz[375] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11741(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_11(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_5501(data, threadData);
  SpiralGalaxy_eqFunction_11432(data, threadData);
  SpiralGalaxy_eqFunction_5503(data, threadData);
  SpiralGalaxy_eqFunction_11431(data, threadData);
  SpiralGalaxy_eqFunction_5505(data, threadData);
  SpiralGalaxy_eqFunction_11444(data, threadData);
  SpiralGalaxy_eqFunction_11445(data, threadData);
  SpiralGalaxy_eqFunction_5508(data, threadData);
  SpiralGalaxy_eqFunction_5509(data, threadData);
  SpiralGalaxy_eqFunction_11446(data, threadData);
  SpiralGalaxy_eqFunction_11447(data, threadData);
  SpiralGalaxy_eqFunction_11450(data, threadData);
  SpiralGalaxy_eqFunction_11449(data, threadData);
  SpiralGalaxy_eqFunction_11448(data, threadData);
  SpiralGalaxy_eqFunction_5515(data, threadData);
  SpiralGalaxy_eqFunction_11443(data, threadData);
  SpiralGalaxy_eqFunction_5517(data, threadData);
  SpiralGalaxy_eqFunction_11442(data, threadData);
  SpiralGalaxy_eqFunction_5519(data, threadData);
  SpiralGalaxy_eqFunction_11441(data, threadData);
  SpiralGalaxy_eqFunction_5521(data, threadData);
  SpiralGalaxy_eqFunction_11454(data, threadData);
  SpiralGalaxy_eqFunction_11455(data, threadData);
  SpiralGalaxy_eqFunction_5524(data, threadData);
  SpiralGalaxy_eqFunction_5525(data, threadData);
  SpiralGalaxy_eqFunction_11456(data, threadData);
  SpiralGalaxy_eqFunction_11457(data, threadData);
  SpiralGalaxy_eqFunction_11460(data, threadData);
  SpiralGalaxy_eqFunction_11459(data, threadData);
  SpiralGalaxy_eqFunction_11458(data, threadData);
  SpiralGalaxy_eqFunction_5531(data, threadData);
  SpiralGalaxy_eqFunction_11453(data, threadData);
  SpiralGalaxy_eqFunction_5533(data, threadData);
  SpiralGalaxy_eqFunction_11452(data, threadData);
  SpiralGalaxy_eqFunction_5535(data, threadData);
  SpiralGalaxy_eqFunction_11451(data, threadData);
  SpiralGalaxy_eqFunction_5537(data, threadData);
  SpiralGalaxy_eqFunction_11464(data, threadData);
  SpiralGalaxy_eqFunction_11465(data, threadData);
  SpiralGalaxy_eqFunction_5540(data, threadData);
  SpiralGalaxy_eqFunction_5541(data, threadData);
  SpiralGalaxy_eqFunction_11466(data, threadData);
  SpiralGalaxy_eqFunction_11467(data, threadData);
  SpiralGalaxy_eqFunction_11470(data, threadData);
  SpiralGalaxy_eqFunction_11469(data, threadData);
  SpiralGalaxy_eqFunction_11468(data, threadData);
  SpiralGalaxy_eqFunction_5547(data, threadData);
  SpiralGalaxy_eqFunction_11463(data, threadData);
  SpiralGalaxy_eqFunction_5549(data, threadData);
  SpiralGalaxy_eqFunction_11462(data, threadData);
  SpiralGalaxy_eqFunction_5551(data, threadData);
  SpiralGalaxy_eqFunction_11461(data, threadData);
  SpiralGalaxy_eqFunction_5553(data, threadData);
  SpiralGalaxy_eqFunction_11474(data, threadData);
  SpiralGalaxy_eqFunction_11475(data, threadData);
  SpiralGalaxy_eqFunction_5556(data, threadData);
  SpiralGalaxy_eqFunction_5557(data, threadData);
  SpiralGalaxy_eqFunction_11476(data, threadData);
  SpiralGalaxy_eqFunction_11477(data, threadData);
  SpiralGalaxy_eqFunction_11480(data, threadData);
  SpiralGalaxy_eqFunction_11479(data, threadData);
  SpiralGalaxy_eqFunction_11478(data, threadData);
  SpiralGalaxy_eqFunction_5563(data, threadData);
  SpiralGalaxy_eqFunction_11473(data, threadData);
  SpiralGalaxy_eqFunction_5565(data, threadData);
  SpiralGalaxy_eqFunction_11472(data, threadData);
  SpiralGalaxy_eqFunction_5567(data, threadData);
  SpiralGalaxy_eqFunction_11471(data, threadData);
  SpiralGalaxy_eqFunction_5569(data, threadData);
  SpiralGalaxy_eqFunction_11484(data, threadData);
  SpiralGalaxy_eqFunction_11485(data, threadData);
  SpiralGalaxy_eqFunction_5572(data, threadData);
  SpiralGalaxy_eqFunction_5573(data, threadData);
  SpiralGalaxy_eqFunction_11486(data, threadData);
  SpiralGalaxy_eqFunction_11487(data, threadData);
  SpiralGalaxy_eqFunction_11490(data, threadData);
  SpiralGalaxy_eqFunction_11489(data, threadData);
  SpiralGalaxy_eqFunction_11488(data, threadData);
  SpiralGalaxy_eqFunction_5579(data, threadData);
  SpiralGalaxy_eqFunction_11483(data, threadData);
  SpiralGalaxy_eqFunction_5581(data, threadData);
  SpiralGalaxy_eqFunction_11482(data, threadData);
  SpiralGalaxy_eqFunction_5583(data, threadData);
  SpiralGalaxy_eqFunction_11481(data, threadData);
  SpiralGalaxy_eqFunction_5585(data, threadData);
  SpiralGalaxy_eqFunction_11494(data, threadData);
  SpiralGalaxy_eqFunction_11495(data, threadData);
  SpiralGalaxy_eqFunction_5588(data, threadData);
  SpiralGalaxy_eqFunction_5589(data, threadData);
  SpiralGalaxy_eqFunction_11496(data, threadData);
  SpiralGalaxy_eqFunction_11497(data, threadData);
  SpiralGalaxy_eqFunction_11500(data, threadData);
  SpiralGalaxy_eqFunction_11499(data, threadData);
  SpiralGalaxy_eqFunction_11498(data, threadData);
  SpiralGalaxy_eqFunction_5595(data, threadData);
  SpiralGalaxy_eqFunction_11493(data, threadData);
  SpiralGalaxy_eqFunction_5597(data, threadData);
  SpiralGalaxy_eqFunction_11492(data, threadData);
  SpiralGalaxy_eqFunction_5599(data, threadData);
  SpiralGalaxy_eqFunction_11491(data, threadData);
  SpiralGalaxy_eqFunction_5601(data, threadData);
  SpiralGalaxy_eqFunction_11504(data, threadData);
  SpiralGalaxy_eqFunction_11505(data, threadData);
  SpiralGalaxy_eqFunction_5604(data, threadData);
  SpiralGalaxy_eqFunction_5605(data, threadData);
  SpiralGalaxy_eqFunction_11506(data, threadData);
  SpiralGalaxy_eqFunction_11507(data, threadData);
  SpiralGalaxy_eqFunction_11510(data, threadData);
  SpiralGalaxy_eqFunction_11509(data, threadData);
  SpiralGalaxy_eqFunction_11508(data, threadData);
  SpiralGalaxy_eqFunction_5611(data, threadData);
  SpiralGalaxy_eqFunction_11503(data, threadData);
  SpiralGalaxy_eqFunction_5613(data, threadData);
  SpiralGalaxy_eqFunction_11502(data, threadData);
  SpiralGalaxy_eqFunction_5615(data, threadData);
  SpiralGalaxy_eqFunction_11501(data, threadData);
  SpiralGalaxy_eqFunction_5617(data, threadData);
  SpiralGalaxy_eqFunction_11514(data, threadData);
  SpiralGalaxy_eqFunction_11515(data, threadData);
  SpiralGalaxy_eqFunction_5620(data, threadData);
  SpiralGalaxy_eqFunction_5621(data, threadData);
  SpiralGalaxy_eqFunction_11516(data, threadData);
  SpiralGalaxy_eqFunction_11517(data, threadData);
  SpiralGalaxy_eqFunction_11520(data, threadData);
  SpiralGalaxy_eqFunction_11519(data, threadData);
  SpiralGalaxy_eqFunction_11518(data, threadData);
  SpiralGalaxy_eqFunction_5627(data, threadData);
  SpiralGalaxy_eqFunction_11513(data, threadData);
  SpiralGalaxy_eqFunction_5629(data, threadData);
  SpiralGalaxy_eqFunction_11512(data, threadData);
  SpiralGalaxy_eqFunction_5631(data, threadData);
  SpiralGalaxy_eqFunction_11511(data, threadData);
  SpiralGalaxy_eqFunction_5633(data, threadData);
  SpiralGalaxy_eqFunction_11524(data, threadData);
  SpiralGalaxy_eqFunction_11525(data, threadData);
  SpiralGalaxy_eqFunction_5636(data, threadData);
  SpiralGalaxy_eqFunction_5637(data, threadData);
  SpiralGalaxy_eqFunction_11526(data, threadData);
  SpiralGalaxy_eqFunction_11527(data, threadData);
  SpiralGalaxy_eqFunction_11530(data, threadData);
  SpiralGalaxy_eqFunction_11529(data, threadData);
  SpiralGalaxy_eqFunction_11528(data, threadData);
  SpiralGalaxy_eqFunction_5643(data, threadData);
  SpiralGalaxy_eqFunction_11523(data, threadData);
  SpiralGalaxy_eqFunction_5645(data, threadData);
  SpiralGalaxy_eqFunction_11522(data, threadData);
  SpiralGalaxy_eqFunction_5647(data, threadData);
  SpiralGalaxy_eqFunction_11521(data, threadData);
  SpiralGalaxy_eqFunction_5649(data, threadData);
  SpiralGalaxy_eqFunction_11534(data, threadData);
  SpiralGalaxy_eqFunction_11535(data, threadData);
  SpiralGalaxy_eqFunction_5652(data, threadData);
  SpiralGalaxy_eqFunction_5653(data, threadData);
  SpiralGalaxy_eqFunction_11536(data, threadData);
  SpiralGalaxy_eqFunction_11537(data, threadData);
  SpiralGalaxy_eqFunction_11540(data, threadData);
  SpiralGalaxy_eqFunction_11539(data, threadData);
  SpiralGalaxy_eqFunction_11538(data, threadData);
  SpiralGalaxy_eqFunction_5659(data, threadData);
  SpiralGalaxy_eqFunction_11533(data, threadData);
  SpiralGalaxy_eqFunction_5661(data, threadData);
  SpiralGalaxy_eqFunction_11532(data, threadData);
  SpiralGalaxy_eqFunction_5663(data, threadData);
  SpiralGalaxy_eqFunction_11531(data, threadData);
  SpiralGalaxy_eqFunction_5665(data, threadData);
  SpiralGalaxy_eqFunction_11544(data, threadData);
  SpiralGalaxy_eqFunction_11545(data, threadData);
  SpiralGalaxy_eqFunction_5668(data, threadData);
  SpiralGalaxy_eqFunction_5669(data, threadData);
  SpiralGalaxy_eqFunction_11546(data, threadData);
  SpiralGalaxy_eqFunction_11547(data, threadData);
  SpiralGalaxy_eqFunction_11550(data, threadData);
  SpiralGalaxy_eqFunction_11549(data, threadData);
  SpiralGalaxy_eqFunction_11548(data, threadData);
  SpiralGalaxy_eqFunction_5675(data, threadData);
  SpiralGalaxy_eqFunction_11543(data, threadData);
  SpiralGalaxy_eqFunction_5677(data, threadData);
  SpiralGalaxy_eqFunction_11542(data, threadData);
  SpiralGalaxy_eqFunction_5679(data, threadData);
  SpiralGalaxy_eqFunction_11541(data, threadData);
  SpiralGalaxy_eqFunction_5681(data, threadData);
  SpiralGalaxy_eqFunction_11554(data, threadData);
  SpiralGalaxy_eqFunction_11555(data, threadData);
  SpiralGalaxy_eqFunction_5684(data, threadData);
  SpiralGalaxy_eqFunction_5685(data, threadData);
  SpiralGalaxy_eqFunction_11556(data, threadData);
  SpiralGalaxy_eqFunction_11557(data, threadData);
  SpiralGalaxy_eqFunction_11560(data, threadData);
  SpiralGalaxy_eqFunction_11559(data, threadData);
  SpiralGalaxy_eqFunction_11558(data, threadData);
  SpiralGalaxy_eqFunction_5691(data, threadData);
  SpiralGalaxy_eqFunction_11553(data, threadData);
  SpiralGalaxy_eqFunction_5693(data, threadData);
  SpiralGalaxy_eqFunction_11552(data, threadData);
  SpiralGalaxy_eqFunction_5695(data, threadData);
  SpiralGalaxy_eqFunction_11551(data, threadData);
  SpiralGalaxy_eqFunction_5697(data, threadData);
  SpiralGalaxy_eqFunction_11564(data, threadData);
  SpiralGalaxy_eqFunction_11565(data, threadData);
  SpiralGalaxy_eqFunction_5700(data, threadData);
  SpiralGalaxy_eqFunction_5701(data, threadData);
  SpiralGalaxy_eqFunction_11566(data, threadData);
  SpiralGalaxy_eqFunction_11567(data, threadData);
  SpiralGalaxy_eqFunction_11570(data, threadData);
  SpiralGalaxy_eqFunction_11569(data, threadData);
  SpiralGalaxy_eqFunction_11568(data, threadData);
  SpiralGalaxy_eqFunction_5707(data, threadData);
  SpiralGalaxy_eqFunction_11563(data, threadData);
  SpiralGalaxy_eqFunction_5709(data, threadData);
  SpiralGalaxy_eqFunction_11562(data, threadData);
  SpiralGalaxy_eqFunction_5711(data, threadData);
  SpiralGalaxy_eqFunction_11561(data, threadData);
  SpiralGalaxy_eqFunction_5713(data, threadData);
  SpiralGalaxy_eqFunction_11574(data, threadData);
  SpiralGalaxy_eqFunction_11575(data, threadData);
  SpiralGalaxy_eqFunction_5716(data, threadData);
  SpiralGalaxy_eqFunction_5717(data, threadData);
  SpiralGalaxy_eqFunction_11576(data, threadData);
  SpiralGalaxy_eqFunction_11577(data, threadData);
  SpiralGalaxy_eqFunction_11580(data, threadData);
  SpiralGalaxy_eqFunction_11579(data, threadData);
  SpiralGalaxy_eqFunction_11578(data, threadData);
  SpiralGalaxy_eqFunction_5723(data, threadData);
  SpiralGalaxy_eqFunction_11573(data, threadData);
  SpiralGalaxy_eqFunction_5725(data, threadData);
  SpiralGalaxy_eqFunction_11572(data, threadData);
  SpiralGalaxy_eqFunction_5727(data, threadData);
  SpiralGalaxy_eqFunction_11571(data, threadData);
  SpiralGalaxy_eqFunction_5729(data, threadData);
  SpiralGalaxy_eqFunction_11584(data, threadData);
  SpiralGalaxy_eqFunction_11585(data, threadData);
  SpiralGalaxy_eqFunction_5732(data, threadData);
  SpiralGalaxy_eqFunction_5733(data, threadData);
  SpiralGalaxy_eqFunction_11586(data, threadData);
  SpiralGalaxy_eqFunction_11587(data, threadData);
  SpiralGalaxy_eqFunction_11590(data, threadData);
  SpiralGalaxy_eqFunction_11589(data, threadData);
  SpiralGalaxy_eqFunction_11588(data, threadData);
  SpiralGalaxy_eqFunction_5739(data, threadData);
  SpiralGalaxy_eqFunction_11583(data, threadData);
  SpiralGalaxy_eqFunction_5741(data, threadData);
  SpiralGalaxy_eqFunction_11582(data, threadData);
  SpiralGalaxy_eqFunction_5743(data, threadData);
  SpiralGalaxy_eqFunction_11581(data, threadData);
  SpiralGalaxy_eqFunction_5745(data, threadData);
  SpiralGalaxy_eqFunction_11594(data, threadData);
  SpiralGalaxy_eqFunction_11595(data, threadData);
  SpiralGalaxy_eqFunction_5748(data, threadData);
  SpiralGalaxy_eqFunction_5749(data, threadData);
  SpiralGalaxy_eqFunction_11596(data, threadData);
  SpiralGalaxy_eqFunction_11597(data, threadData);
  SpiralGalaxy_eqFunction_11600(data, threadData);
  SpiralGalaxy_eqFunction_11599(data, threadData);
  SpiralGalaxy_eqFunction_11598(data, threadData);
  SpiralGalaxy_eqFunction_5755(data, threadData);
  SpiralGalaxy_eqFunction_11593(data, threadData);
  SpiralGalaxy_eqFunction_5757(data, threadData);
  SpiralGalaxy_eqFunction_11592(data, threadData);
  SpiralGalaxy_eqFunction_5759(data, threadData);
  SpiralGalaxy_eqFunction_11591(data, threadData);
  SpiralGalaxy_eqFunction_5761(data, threadData);
  SpiralGalaxy_eqFunction_11604(data, threadData);
  SpiralGalaxy_eqFunction_11605(data, threadData);
  SpiralGalaxy_eqFunction_5764(data, threadData);
  SpiralGalaxy_eqFunction_5765(data, threadData);
  SpiralGalaxy_eqFunction_11606(data, threadData);
  SpiralGalaxy_eqFunction_11607(data, threadData);
  SpiralGalaxy_eqFunction_11610(data, threadData);
  SpiralGalaxy_eqFunction_11609(data, threadData);
  SpiralGalaxy_eqFunction_11608(data, threadData);
  SpiralGalaxy_eqFunction_5771(data, threadData);
  SpiralGalaxy_eqFunction_11603(data, threadData);
  SpiralGalaxy_eqFunction_5773(data, threadData);
  SpiralGalaxy_eqFunction_11602(data, threadData);
  SpiralGalaxy_eqFunction_5775(data, threadData);
  SpiralGalaxy_eqFunction_11601(data, threadData);
  SpiralGalaxy_eqFunction_5777(data, threadData);
  SpiralGalaxy_eqFunction_11614(data, threadData);
  SpiralGalaxy_eqFunction_11615(data, threadData);
  SpiralGalaxy_eqFunction_5780(data, threadData);
  SpiralGalaxy_eqFunction_5781(data, threadData);
  SpiralGalaxy_eqFunction_11616(data, threadData);
  SpiralGalaxy_eqFunction_11617(data, threadData);
  SpiralGalaxy_eqFunction_11620(data, threadData);
  SpiralGalaxy_eqFunction_11619(data, threadData);
  SpiralGalaxy_eqFunction_11618(data, threadData);
  SpiralGalaxy_eqFunction_5787(data, threadData);
  SpiralGalaxy_eqFunction_11613(data, threadData);
  SpiralGalaxy_eqFunction_5789(data, threadData);
  SpiralGalaxy_eqFunction_11612(data, threadData);
  SpiralGalaxy_eqFunction_5791(data, threadData);
  SpiralGalaxy_eqFunction_11611(data, threadData);
  SpiralGalaxy_eqFunction_5793(data, threadData);
  SpiralGalaxy_eqFunction_11624(data, threadData);
  SpiralGalaxy_eqFunction_11625(data, threadData);
  SpiralGalaxy_eqFunction_5796(data, threadData);
  SpiralGalaxy_eqFunction_5797(data, threadData);
  SpiralGalaxy_eqFunction_11626(data, threadData);
  SpiralGalaxy_eqFunction_11627(data, threadData);
  SpiralGalaxy_eqFunction_11630(data, threadData);
  SpiralGalaxy_eqFunction_11629(data, threadData);
  SpiralGalaxy_eqFunction_11628(data, threadData);
  SpiralGalaxy_eqFunction_5803(data, threadData);
  SpiralGalaxy_eqFunction_11623(data, threadData);
  SpiralGalaxy_eqFunction_5805(data, threadData);
  SpiralGalaxy_eqFunction_11622(data, threadData);
  SpiralGalaxy_eqFunction_5807(data, threadData);
  SpiralGalaxy_eqFunction_11621(data, threadData);
  SpiralGalaxy_eqFunction_5809(data, threadData);
  SpiralGalaxy_eqFunction_11634(data, threadData);
  SpiralGalaxy_eqFunction_11635(data, threadData);
  SpiralGalaxy_eqFunction_5812(data, threadData);
  SpiralGalaxy_eqFunction_5813(data, threadData);
  SpiralGalaxy_eqFunction_11636(data, threadData);
  SpiralGalaxy_eqFunction_11637(data, threadData);
  SpiralGalaxy_eqFunction_11640(data, threadData);
  SpiralGalaxy_eqFunction_11639(data, threadData);
  SpiralGalaxy_eqFunction_11638(data, threadData);
  SpiralGalaxy_eqFunction_5819(data, threadData);
  SpiralGalaxy_eqFunction_11633(data, threadData);
  SpiralGalaxy_eqFunction_5821(data, threadData);
  SpiralGalaxy_eqFunction_11632(data, threadData);
  SpiralGalaxy_eqFunction_5823(data, threadData);
  SpiralGalaxy_eqFunction_11631(data, threadData);
  SpiralGalaxy_eqFunction_5825(data, threadData);
  SpiralGalaxy_eqFunction_11644(data, threadData);
  SpiralGalaxy_eqFunction_11645(data, threadData);
  SpiralGalaxy_eqFunction_5828(data, threadData);
  SpiralGalaxy_eqFunction_5829(data, threadData);
  SpiralGalaxy_eqFunction_11646(data, threadData);
  SpiralGalaxy_eqFunction_11647(data, threadData);
  SpiralGalaxy_eqFunction_11650(data, threadData);
  SpiralGalaxy_eqFunction_11649(data, threadData);
  SpiralGalaxy_eqFunction_11648(data, threadData);
  SpiralGalaxy_eqFunction_5835(data, threadData);
  SpiralGalaxy_eqFunction_11643(data, threadData);
  SpiralGalaxy_eqFunction_5837(data, threadData);
  SpiralGalaxy_eqFunction_11642(data, threadData);
  SpiralGalaxy_eqFunction_5839(data, threadData);
  SpiralGalaxy_eqFunction_11641(data, threadData);
  SpiralGalaxy_eqFunction_5841(data, threadData);
  SpiralGalaxy_eqFunction_11654(data, threadData);
  SpiralGalaxy_eqFunction_11655(data, threadData);
  SpiralGalaxy_eqFunction_5844(data, threadData);
  SpiralGalaxy_eqFunction_5845(data, threadData);
  SpiralGalaxy_eqFunction_11656(data, threadData);
  SpiralGalaxy_eqFunction_11657(data, threadData);
  SpiralGalaxy_eqFunction_11660(data, threadData);
  SpiralGalaxy_eqFunction_11659(data, threadData);
  SpiralGalaxy_eqFunction_11658(data, threadData);
  SpiralGalaxy_eqFunction_5851(data, threadData);
  SpiralGalaxy_eqFunction_11653(data, threadData);
  SpiralGalaxy_eqFunction_5853(data, threadData);
  SpiralGalaxy_eqFunction_11652(data, threadData);
  SpiralGalaxy_eqFunction_5855(data, threadData);
  SpiralGalaxy_eqFunction_11651(data, threadData);
  SpiralGalaxy_eqFunction_5857(data, threadData);
  SpiralGalaxy_eqFunction_11664(data, threadData);
  SpiralGalaxy_eqFunction_11665(data, threadData);
  SpiralGalaxy_eqFunction_5860(data, threadData);
  SpiralGalaxy_eqFunction_5861(data, threadData);
  SpiralGalaxy_eqFunction_11666(data, threadData);
  SpiralGalaxy_eqFunction_11667(data, threadData);
  SpiralGalaxy_eqFunction_11670(data, threadData);
  SpiralGalaxy_eqFunction_11669(data, threadData);
  SpiralGalaxy_eqFunction_11668(data, threadData);
  SpiralGalaxy_eqFunction_5867(data, threadData);
  SpiralGalaxy_eqFunction_11663(data, threadData);
  SpiralGalaxy_eqFunction_5869(data, threadData);
  SpiralGalaxy_eqFunction_11662(data, threadData);
  SpiralGalaxy_eqFunction_5871(data, threadData);
  SpiralGalaxy_eqFunction_11661(data, threadData);
  SpiralGalaxy_eqFunction_5873(data, threadData);
  SpiralGalaxy_eqFunction_11674(data, threadData);
  SpiralGalaxy_eqFunction_11675(data, threadData);
  SpiralGalaxy_eqFunction_5876(data, threadData);
  SpiralGalaxy_eqFunction_5877(data, threadData);
  SpiralGalaxy_eqFunction_11676(data, threadData);
  SpiralGalaxy_eqFunction_11677(data, threadData);
  SpiralGalaxy_eqFunction_11680(data, threadData);
  SpiralGalaxy_eqFunction_11679(data, threadData);
  SpiralGalaxy_eqFunction_11678(data, threadData);
  SpiralGalaxy_eqFunction_5883(data, threadData);
  SpiralGalaxy_eqFunction_11673(data, threadData);
  SpiralGalaxy_eqFunction_5885(data, threadData);
  SpiralGalaxy_eqFunction_11672(data, threadData);
  SpiralGalaxy_eqFunction_5887(data, threadData);
  SpiralGalaxy_eqFunction_11671(data, threadData);
  SpiralGalaxy_eqFunction_5889(data, threadData);
  SpiralGalaxy_eqFunction_11684(data, threadData);
  SpiralGalaxy_eqFunction_11685(data, threadData);
  SpiralGalaxy_eqFunction_5892(data, threadData);
  SpiralGalaxy_eqFunction_5893(data, threadData);
  SpiralGalaxy_eqFunction_11686(data, threadData);
  SpiralGalaxy_eqFunction_11687(data, threadData);
  SpiralGalaxy_eqFunction_11690(data, threadData);
  SpiralGalaxy_eqFunction_11689(data, threadData);
  SpiralGalaxy_eqFunction_11688(data, threadData);
  SpiralGalaxy_eqFunction_5899(data, threadData);
  SpiralGalaxy_eqFunction_11683(data, threadData);
  SpiralGalaxy_eqFunction_5901(data, threadData);
  SpiralGalaxy_eqFunction_11682(data, threadData);
  SpiralGalaxy_eqFunction_5903(data, threadData);
  SpiralGalaxy_eqFunction_11681(data, threadData);
  SpiralGalaxy_eqFunction_5905(data, threadData);
  SpiralGalaxy_eqFunction_11694(data, threadData);
  SpiralGalaxy_eqFunction_11695(data, threadData);
  SpiralGalaxy_eqFunction_5908(data, threadData);
  SpiralGalaxy_eqFunction_5909(data, threadData);
  SpiralGalaxy_eqFunction_11696(data, threadData);
  SpiralGalaxy_eqFunction_11697(data, threadData);
  SpiralGalaxy_eqFunction_11700(data, threadData);
  SpiralGalaxy_eqFunction_11699(data, threadData);
  SpiralGalaxy_eqFunction_11698(data, threadData);
  SpiralGalaxy_eqFunction_5915(data, threadData);
  SpiralGalaxy_eqFunction_11693(data, threadData);
  SpiralGalaxy_eqFunction_5917(data, threadData);
  SpiralGalaxy_eqFunction_11692(data, threadData);
  SpiralGalaxy_eqFunction_5919(data, threadData);
  SpiralGalaxy_eqFunction_11691(data, threadData);
  SpiralGalaxy_eqFunction_5921(data, threadData);
  SpiralGalaxy_eqFunction_11704(data, threadData);
  SpiralGalaxy_eqFunction_11705(data, threadData);
  SpiralGalaxy_eqFunction_5924(data, threadData);
  SpiralGalaxy_eqFunction_5925(data, threadData);
  SpiralGalaxy_eqFunction_11706(data, threadData);
  SpiralGalaxy_eqFunction_11707(data, threadData);
  SpiralGalaxy_eqFunction_11710(data, threadData);
  SpiralGalaxy_eqFunction_11709(data, threadData);
  SpiralGalaxy_eqFunction_11708(data, threadData);
  SpiralGalaxy_eqFunction_5931(data, threadData);
  SpiralGalaxy_eqFunction_11703(data, threadData);
  SpiralGalaxy_eqFunction_5933(data, threadData);
  SpiralGalaxy_eqFunction_11702(data, threadData);
  SpiralGalaxy_eqFunction_5935(data, threadData);
  SpiralGalaxy_eqFunction_11701(data, threadData);
  SpiralGalaxy_eqFunction_5937(data, threadData);
  SpiralGalaxy_eqFunction_11714(data, threadData);
  SpiralGalaxy_eqFunction_11715(data, threadData);
  SpiralGalaxy_eqFunction_5940(data, threadData);
  SpiralGalaxy_eqFunction_5941(data, threadData);
  SpiralGalaxy_eqFunction_11716(data, threadData);
  SpiralGalaxy_eqFunction_11717(data, threadData);
  SpiralGalaxy_eqFunction_11720(data, threadData);
  SpiralGalaxy_eqFunction_11719(data, threadData);
  SpiralGalaxy_eqFunction_11718(data, threadData);
  SpiralGalaxy_eqFunction_5947(data, threadData);
  SpiralGalaxy_eqFunction_11713(data, threadData);
  SpiralGalaxy_eqFunction_5949(data, threadData);
  SpiralGalaxy_eqFunction_11712(data, threadData);
  SpiralGalaxy_eqFunction_5951(data, threadData);
  SpiralGalaxy_eqFunction_11711(data, threadData);
  SpiralGalaxy_eqFunction_5953(data, threadData);
  SpiralGalaxy_eqFunction_11724(data, threadData);
  SpiralGalaxy_eqFunction_11725(data, threadData);
  SpiralGalaxy_eqFunction_5956(data, threadData);
  SpiralGalaxy_eqFunction_5957(data, threadData);
  SpiralGalaxy_eqFunction_11726(data, threadData);
  SpiralGalaxy_eqFunction_11727(data, threadData);
  SpiralGalaxy_eqFunction_11730(data, threadData);
  SpiralGalaxy_eqFunction_11729(data, threadData);
  SpiralGalaxy_eqFunction_11728(data, threadData);
  SpiralGalaxy_eqFunction_5963(data, threadData);
  SpiralGalaxy_eqFunction_11723(data, threadData);
  SpiralGalaxy_eqFunction_5965(data, threadData);
  SpiralGalaxy_eqFunction_11722(data, threadData);
  SpiralGalaxy_eqFunction_5967(data, threadData);
  SpiralGalaxy_eqFunction_11721(data, threadData);
  SpiralGalaxy_eqFunction_5969(data, threadData);
  SpiralGalaxy_eqFunction_11734(data, threadData);
  SpiralGalaxy_eqFunction_11735(data, threadData);
  SpiralGalaxy_eqFunction_5972(data, threadData);
  SpiralGalaxy_eqFunction_5973(data, threadData);
  SpiralGalaxy_eqFunction_11736(data, threadData);
  SpiralGalaxy_eqFunction_11737(data, threadData);
  SpiralGalaxy_eqFunction_11740(data, threadData);
  SpiralGalaxy_eqFunction_11739(data, threadData);
  SpiralGalaxy_eqFunction_11738(data, threadData);
  SpiralGalaxy_eqFunction_5979(data, threadData);
  SpiralGalaxy_eqFunction_11733(data, threadData);
  SpiralGalaxy_eqFunction_5981(data, threadData);
  SpiralGalaxy_eqFunction_11732(data, threadData);
  SpiralGalaxy_eqFunction_5983(data, threadData);
  SpiralGalaxy_eqFunction_11731(data, threadData);
  SpiralGalaxy_eqFunction_5985(data, threadData);
  SpiralGalaxy_eqFunction_11744(data, threadData);
  SpiralGalaxy_eqFunction_11745(data, threadData);
  SpiralGalaxy_eqFunction_5988(data, threadData);
  SpiralGalaxy_eqFunction_5989(data, threadData);
  SpiralGalaxy_eqFunction_11746(data, threadData);
  SpiralGalaxy_eqFunction_11747(data, threadData);
  SpiralGalaxy_eqFunction_11750(data, threadData);
  SpiralGalaxy_eqFunction_11749(data, threadData);
  SpiralGalaxy_eqFunction_11748(data, threadData);
  SpiralGalaxy_eqFunction_5995(data, threadData);
  SpiralGalaxy_eqFunction_11743(data, threadData);
  SpiralGalaxy_eqFunction_5997(data, threadData);
  SpiralGalaxy_eqFunction_11742(data, threadData);
  SpiralGalaxy_eqFunction_5999(data, threadData);
  SpiralGalaxy_eqFunction_11741(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif