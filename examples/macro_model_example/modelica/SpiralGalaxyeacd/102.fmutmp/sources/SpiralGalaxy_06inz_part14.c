#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void SpiralGalaxy_eqFunction_12379(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12378(DATA *data, threadData_t *threadData);


/*
equation index: 7003
type: SIMPLE_ASSIGN
vx[438] = (-sin(theta[438])) * r_init[438] * omega_c[438]
*/
void SpiralGalaxy_eqFunction_7003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7003};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[437]] /* vx[438] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1944] /* theta[438] PARAM */)))) * (((data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */)) * ((data->simulationInfo->realParameter[942] /* omega_c[438] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12373(DATA *data, threadData_t *threadData);


/*
equation index: 7005
type: SIMPLE_ASSIGN
vy[438] = cos(theta[438]) * r_init[438] * omega_c[438]
*/
void SpiralGalaxy_eqFunction_7005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7005};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[937]] /* vy[438] STATE(1) */) = (cos((data->simulationInfo->realParameter[1944] /* theta[438] PARAM */))) * (((data->simulationInfo->realParameter[1443] /* r_init[438] PARAM */)) * ((data->simulationInfo->realParameter[942] /* omega_c[438] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12372(DATA *data, threadData_t *threadData);


/*
equation index: 7007
type: SIMPLE_ASSIGN
vz[438] = 0.0
*/
void SpiralGalaxy_eqFunction_7007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7007};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1437]] /* vz[438] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12371(DATA *data, threadData_t *threadData);


/*
equation index: 7009
type: SIMPLE_ASSIGN
z[439] = 0.03024
*/
void SpiralGalaxy_eqFunction_7009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7009};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2938]] /* z[439] STATE(1,vz[439]) */) = 0.03024;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12384(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12385(DATA *data, threadData_t *threadData);


/*
equation index: 7012
type: SIMPLE_ASSIGN
y[439] = r_init[439] * sin(theta[439] + 0.00756)
*/
void SpiralGalaxy_eqFunction_7012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7012};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2438]] /* y[439] STATE(1,vy[439]) */) = ((data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */)) * (sin((data->simulationInfo->realParameter[1945] /* theta[439] PARAM */) + 0.00756));
  TRACE_POP
}

/*
equation index: 7013
type: SIMPLE_ASSIGN
x[439] = r_init[439] * cos(theta[439] + 0.00756)
*/
void SpiralGalaxy_eqFunction_7013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7013};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1938]] /* x[439] STATE(1,vx[439]) */) = ((data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */)) * (cos((data->simulationInfo->realParameter[1945] /* theta[439] PARAM */) + 0.00756));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12386(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12387(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12390(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12389(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12388(DATA *data, threadData_t *threadData);


/*
equation index: 7019
type: SIMPLE_ASSIGN
vx[439] = (-sin(theta[439])) * r_init[439] * omega_c[439]
*/
void SpiralGalaxy_eqFunction_7019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7019};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[438]] /* vx[439] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1945] /* theta[439] PARAM */)))) * (((data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */)) * ((data->simulationInfo->realParameter[943] /* omega_c[439] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12383(DATA *data, threadData_t *threadData);


/*
equation index: 7021
type: SIMPLE_ASSIGN
vy[439] = cos(theta[439]) * r_init[439] * omega_c[439]
*/
void SpiralGalaxy_eqFunction_7021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7021};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[938]] /* vy[439] STATE(1) */) = (cos((data->simulationInfo->realParameter[1945] /* theta[439] PARAM */))) * (((data->simulationInfo->realParameter[1444] /* r_init[439] PARAM */)) * ((data->simulationInfo->realParameter[943] /* omega_c[439] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12382(DATA *data, threadData_t *threadData);


/*
equation index: 7023
type: SIMPLE_ASSIGN
vz[439] = 0.0
*/
void SpiralGalaxy_eqFunction_7023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7023};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1438]] /* vz[439] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12381(DATA *data, threadData_t *threadData);


/*
equation index: 7025
type: SIMPLE_ASSIGN
z[440] = 0.030400000000000003
*/
void SpiralGalaxy_eqFunction_7025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7025};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2939]] /* z[440] STATE(1,vz[440]) */) = 0.030400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12394(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12395(DATA *data, threadData_t *threadData);


/*
equation index: 7028
type: SIMPLE_ASSIGN
y[440] = r_init[440] * sin(theta[440] + 0.0076)
*/
void SpiralGalaxy_eqFunction_7028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7028};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2439]] /* y[440] STATE(1,vy[440]) */) = ((data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */)) * (sin((data->simulationInfo->realParameter[1946] /* theta[440] PARAM */) + 0.0076));
  TRACE_POP
}

/*
equation index: 7029
type: SIMPLE_ASSIGN
x[440] = r_init[440] * cos(theta[440] + 0.0076)
*/
void SpiralGalaxy_eqFunction_7029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7029};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1939]] /* x[440] STATE(1,vx[440]) */) = ((data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */)) * (cos((data->simulationInfo->realParameter[1946] /* theta[440] PARAM */) + 0.0076));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12396(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12397(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12400(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12399(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12398(DATA *data, threadData_t *threadData);


/*
equation index: 7035
type: SIMPLE_ASSIGN
vx[440] = (-sin(theta[440])) * r_init[440] * omega_c[440]
*/
void SpiralGalaxy_eqFunction_7035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7035};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[439]] /* vx[440] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1946] /* theta[440] PARAM */)))) * (((data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */)) * ((data->simulationInfo->realParameter[944] /* omega_c[440] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12393(DATA *data, threadData_t *threadData);


/*
equation index: 7037
type: SIMPLE_ASSIGN
vy[440] = cos(theta[440]) * r_init[440] * omega_c[440]
*/
void SpiralGalaxy_eqFunction_7037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7037};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[939]] /* vy[440] STATE(1) */) = (cos((data->simulationInfo->realParameter[1946] /* theta[440] PARAM */))) * (((data->simulationInfo->realParameter[1445] /* r_init[440] PARAM */)) * ((data->simulationInfo->realParameter[944] /* omega_c[440] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12392(DATA *data, threadData_t *threadData);


/*
equation index: 7039
type: SIMPLE_ASSIGN
vz[440] = 0.0
*/
void SpiralGalaxy_eqFunction_7039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7039};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1439]] /* vz[440] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12391(DATA *data, threadData_t *threadData);


/*
equation index: 7041
type: SIMPLE_ASSIGN
z[441] = 0.030560000000000004
*/
void SpiralGalaxy_eqFunction_7041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7041};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2940]] /* z[441] STATE(1,vz[441]) */) = 0.030560000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12404(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12405(DATA *data, threadData_t *threadData);


/*
equation index: 7044
type: SIMPLE_ASSIGN
y[441] = r_init[441] * sin(theta[441] + 0.00764)
*/
void SpiralGalaxy_eqFunction_7044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7044};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2440]] /* y[441] STATE(1,vy[441]) */) = ((data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */)) * (sin((data->simulationInfo->realParameter[1947] /* theta[441] PARAM */) + 0.00764));
  TRACE_POP
}

/*
equation index: 7045
type: SIMPLE_ASSIGN
x[441] = r_init[441] * cos(theta[441] + 0.00764)
*/
void SpiralGalaxy_eqFunction_7045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7045};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1940]] /* x[441] STATE(1,vx[441]) */) = ((data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */)) * (cos((data->simulationInfo->realParameter[1947] /* theta[441] PARAM */) + 0.00764));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12406(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12407(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12410(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12409(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12408(DATA *data, threadData_t *threadData);


/*
equation index: 7051
type: SIMPLE_ASSIGN
vx[441] = (-sin(theta[441])) * r_init[441] * omega_c[441]
*/
void SpiralGalaxy_eqFunction_7051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7051};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[440]] /* vx[441] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1947] /* theta[441] PARAM */)))) * (((data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */)) * ((data->simulationInfo->realParameter[945] /* omega_c[441] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12403(DATA *data, threadData_t *threadData);


/*
equation index: 7053
type: SIMPLE_ASSIGN
vy[441] = cos(theta[441]) * r_init[441] * omega_c[441]
*/
void SpiralGalaxy_eqFunction_7053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7053};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[940]] /* vy[441] STATE(1) */) = (cos((data->simulationInfo->realParameter[1947] /* theta[441] PARAM */))) * (((data->simulationInfo->realParameter[1446] /* r_init[441] PARAM */)) * ((data->simulationInfo->realParameter[945] /* omega_c[441] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12402(DATA *data, threadData_t *threadData);


/*
equation index: 7055
type: SIMPLE_ASSIGN
vz[441] = 0.0
*/
void SpiralGalaxy_eqFunction_7055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7055};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1440]] /* vz[441] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12401(DATA *data, threadData_t *threadData);


/*
equation index: 7057
type: SIMPLE_ASSIGN
z[442] = 0.03072
*/
void SpiralGalaxy_eqFunction_7057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7057};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2941]] /* z[442] STATE(1,vz[442]) */) = 0.03072;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12414(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12415(DATA *data, threadData_t *threadData);


/*
equation index: 7060
type: SIMPLE_ASSIGN
y[442] = r_init[442] * sin(theta[442] + 0.00768)
*/
void SpiralGalaxy_eqFunction_7060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7060};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2441]] /* y[442] STATE(1,vy[442]) */) = ((data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */)) * (sin((data->simulationInfo->realParameter[1948] /* theta[442] PARAM */) + 0.00768));
  TRACE_POP
}

/*
equation index: 7061
type: SIMPLE_ASSIGN
x[442] = r_init[442] * cos(theta[442] + 0.00768)
*/
void SpiralGalaxy_eqFunction_7061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7061};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1941]] /* x[442] STATE(1,vx[442]) */) = ((data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */)) * (cos((data->simulationInfo->realParameter[1948] /* theta[442] PARAM */) + 0.00768));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12416(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12417(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12420(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12419(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12418(DATA *data, threadData_t *threadData);


/*
equation index: 7067
type: SIMPLE_ASSIGN
vx[442] = (-sin(theta[442])) * r_init[442] * omega_c[442]
*/
void SpiralGalaxy_eqFunction_7067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7067};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[441]] /* vx[442] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1948] /* theta[442] PARAM */)))) * (((data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */)) * ((data->simulationInfo->realParameter[946] /* omega_c[442] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12413(DATA *data, threadData_t *threadData);


/*
equation index: 7069
type: SIMPLE_ASSIGN
vy[442] = cos(theta[442]) * r_init[442] * omega_c[442]
*/
void SpiralGalaxy_eqFunction_7069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7069};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[941]] /* vy[442] STATE(1) */) = (cos((data->simulationInfo->realParameter[1948] /* theta[442] PARAM */))) * (((data->simulationInfo->realParameter[1447] /* r_init[442] PARAM */)) * ((data->simulationInfo->realParameter[946] /* omega_c[442] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12412(DATA *data, threadData_t *threadData);


/*
equation index: 7071
type: SIMPLE_ASSIGN
vz[442] = 0.0
*/
void SpiralGalaxy_eqFunction_7071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7071};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1441]] /* vz[442] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12411(DATA *data, threadData_t *threadData);


/*
equation index: 7073
type: SIMPLE_ASSIGN
z[443] = 0.03088
*/
void SpiralGalaxy_eqFunction_7073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7073};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2942]] /* z[443] STATE(1,vz[443]) */) = 0.03088;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12424(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12425(DATA *data, threadData_t *threadData);


/*
equation index: 7076
type: SIMPLE_ASSIGN
y[443] = r_init[443] * sin(theta[443] + 0.00772)
*/
void SpiralGalaxy_eqFunction_7076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7076};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2442]] /* y[443] STATE(1,vy[443]) */) = ((data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */)) * (sin((data->simulationInfo->realParameter[1949] /* theta[443] PARAM */) + 0.00772));
  TRACE_POP
}

/*
equation index: 7077
type: SIMPLE_ASSIGN
x[443] = r_init[443] * cos(theta[443] + 0.00772)
*/
void SpiralGalaxy_eqFunction_7077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7077};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1942]] /* x[443] STATE(1,vx[443]) */) = ((data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */)) * (cos((data->simulationInfo->realParameter[1949] /* theta[443] PARAM */) + 0.00772));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12426(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12427(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12430(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12429(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12428(DATA *data, threadData_t *threadData);


/*
equation index: 7083
type: SIMPLE_ASSIGN
vx[443] = (-sin(theta[443])) * r_init[443] * omega_c[443]
*/
void SpiralGalaxy_eqFunction_7083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7083};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[442]] /* vx[443] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1949] /* theta[443] PARAM */)))) * (((data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */)) * ((data->simulationInfo->realParameter[947] /* omega_c[443] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12423(DATA *data, threadData_t *threadData);


/*
equation index: 7085
type: SIMPLE_ASSIGN
vy[443] = cos(theta[443]) * r_init[443] * omega_c[443]
*/
void SpiralGalaxy_eqFunction_7085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7085};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[942]] /* vy[443] STATE(1) */) = (cos((data->simulationInfo->realParameter[1949] /* theta[443] PARAM */))) * (((data->simulationInfo->realParameter[1448] /* r_init[443] PARAM */)) * ((data->simulationInfo->realParameter[947] /* omega_c[443] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12422(DATA *data, threadData_t *threadData);


/*
equation index: 7087
type: SIMPLE_ASSIGN
vz[443] = 0.0
*/
void SpiralGalaxy_eqFunction_7087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7087};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1442]] /* vz[443] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12421(DATA *data, threadData_t *threadData);


/*
equation index: 7089
type: SIMPLE_ASSIGN
z[444] = 0.031040000000000005
*/
void SpiralGalaxy_eqFunction_7089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7089};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2943]] /* z[444] STATE(1,vz[444]) */) = 0.031040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12434(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12435(DATA *data, threadData_t *threadData);


/*
equation index: 7092
type: SIMPLE_ASSIGN
y[444] = r_init[444] * sin(theta[444] + 0.00776)
*/
void SpiralGalaxy_eqFunction_7092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7092};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2443]] /* y[444] STATE(1,vy[444]) */) = ((data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */)) * (sin((data->simulationInfo->realParameter[1950] /* theta[444] PARAM */) + 0.00776));
  TRACE_POP
}

/*
equation index: 7093
type: SIMPLE_ASSIGN
x[444] = r_init[444] * cos(theta[444] + 0.00776)
*/
void SpiralGalaxy_eqFunction_7093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7093};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1943]] /* x[444] STATE(1,vx[444]) */) = ((data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */)) * (cos((data->simulationInfo->realParameter[1950] /* theta[444] PARAM */) + 0.00776));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12436(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12437(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12440(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12439(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12438(DATA *data, threadData_t *threadData);


/*
equation index: 7099
type: SIMPLE_ASSIGN
vx[444] = (-sin(theta[444])) * r_init[444] * omega_c[444]
*/
void SpiralGalaxy_eqFunction_7099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7099};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[443]] /* vx[444] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1950] /* theta[444] PARAM */)))) * (((data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */)) * ((data->simulationInfo->realParameter[948] /* omega_c[444] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12433(DATA *data, threadData_t *threadData);


/*
equation index: 7101
type: SIMPLE_ASSIGN
vy[444] = cos(theta[444]) * r_init[444] * omega_c[444]
*/
void SpiralGalaxy_eqFunction_7101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[943]] /* vy[444] STATE(1) */) = (cos((data->simulationInfo->realParameter[1950] /* theta[444] PARAM */))) * (((data->simulationInfo->realParameter[1449] /* r_init[444] PARAM */)) * ((data->simulationInfo->realParameter[948] /* omega_c[444] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12432(DATA *data, threadData_t *threadData);


/*
equation index: 7103
type: SIMPLE_ASSIGN
vz[444] = 0.0
*/
void SpiralGalaxy_eqFunction_7103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7103};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1443]] /* vz[444] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12431(DATA *data, threadData_t *threadData);


/*
equation index: 7105
type: SIMPLE_ASSIGN
z[445] = 0.031200000000000006
*/
void SpiralGalaxy_eqFunction_7105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7105};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2944]] /* z[445] STATE(1,vz[445]) */) = 0.031200000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12444(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12445(DATA *data, threadData_t *threadData);


/*
equation index: 7108
type: SIMPLE_ASSIGN
y[445] = r_init[445] * sin(theta[445] + 0.0078000000000000005)
*/
void SpiralGalaxy_eqFunction_7108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7108};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2444]] /* y[445] STATE(1,vy[445]) */) = ((data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */)) * (sin((data->simulationInfo->realParameter[1951] /* theta[445] PARAM */) + 0.0078000000000000005));
  TRACE_POP
}

/*
equation index: 7109
type: SIMPLE_ASSIGN
x[445] = r_init[445] * cos(theta[445] + 0.0078000000000000005)
*/
void SpiralGalaxy_eqFunction_7109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1944]] /* x[445] STATE(1,vx[445]) */) = ((data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */)) * (cos((data->simulationInfo->realParameter[1951] /* theta[445] PARAM */) + 0.0078000000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12446(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12447(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12450(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12449(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12448(DATA *data, threadData_t *threadData);


/*
equation index: 7115
type: SIMPLE_ASSIGN
vx[445] = (-sin(theta[445])) * r_init[445] * omega_c[445]
*/
void SpiralGalaxy_eqFunction_7115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[444]] /* vx[445] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1951] /* theta[445] PARAM */)))) * (((data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */)) * ((data->simulationInfo->realParameter[949] /* omega_c[445] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12443(DATA *data, threadData_t *threadData);


/*
equation index: 7117
type: SIMPLE_ASSIGN
vy[445] = cos(theta[445]) * r_init[445] * omega_c[445]
*/
void SpiralGalaxy_eqFunction_7117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[944]] /* vy[445] STATE(1) */) = (cos((data->simulationInfo->realParameter[1951] /* theta[445] PARAM */))) * (((data->simulationInfo->realParameter[1450] /* r_init[445] PARAM */)) * ((data->simulationInfo->realParameter[949] /* omega_c[445] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12442(DATA *data, threadData_t *threadData);


/*
equation index: 7119
type: SIMPLE_ASSIGN
vz[445] = 0.0
*/
void SpiralGalaxy_eqFunction_7119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7119};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1444]] /* vz[445] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12441(DATA *data, threadData_t *threadData);


/*
equation index: 7121
type: SIMPLE_ASSIGN
z[446] = 0.03136
*/
void SpiralGalaxy_eqFunction_7121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7121};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2945]] /* z[446] STATE(1,vz[446]) */) = 0.03136;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12454(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12455(DATA *data, threadData_t *threadData);


/*
equation index: 7124
type: SIMPLE_ASSIGN
y[446] = r_init[446] * sin(theta[446] + 0.00784)
*/
void SpiralGalaxy_eqFunction_7124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7124};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2445]] /* y[446] STATE(1,vy[446]) */) = ((data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */)) * (sin((data->simulationInfo->realParameter[1952] /* theta[446] PARAM */) + 0.00784));
  TRACE_POP
}

/*
equation index: 7125
type: SIMPLE_ASSIGN
x[446] = r_init[446] * cos(theta[446] + 0.00784)
*/
void SpiralGalaxy_eqFunction_7125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1945]] /* x[446] STATE(1,vx[446]) */) = ((data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */)) * (cos((data->simulationInfo->realParameter[1952] /* theta[446] PARAM */) + 0.00784));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12456(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12457(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12460(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12459(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12458(DATA *data, threadData_t *threadData);


/*
equation index: 7131
type: SIMPLE_ASSIGN
vx[446] = (-sin(theta[446])) * r_init[446] * omega_c[446]
*/
void SpiralGalaxy_eqFunction_7131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[445]] /* vx[446] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1952] /* theta[446] PARAM */)))) * (((data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */)) * ((data->simulationInfo->realParameter[950] /* omega_c[446] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12453(DATA *data, threadData_t *threadData);


/*
equation index: 7133
type: SIMPLE_ASSIGN
vy[446] = cos(theta[446]) * r_init[446] * omega_c[446]
*/
void SpiralGalaxy_eqFunction_7133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[945]] /* vy[446] STATE(1) */) = (cos((data->simulationInfo->realParameter[1952] /* theta[446] PARAM */))) * (((data->simulationInfo->realParameter[1451] /* r_init[446] PARAM */)) * ((data->simulationInfo->realParameter[950] /* omega_c[446] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12452(DATA *data, threadData_t *threadData);


/*
equation index: 7135
type: SIMPLE_ASSIGN
vz[446] = 0.0
*/
void SpiralGalaxy_eqFunction_7135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7135};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1445]] /* vz[446] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12451(DATA *data, threadData_t *threadData);


/*
equation index: 7137
type: SIMPLE_ASSIGN
z[447] = 0.03152
*/
void SpiralGalaxy_eqFunction_7137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7137};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2946]] /* z[447] STATE(1,vz[447]) */) = 0.03152;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12464(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12465(DATA *data, threadData_t *threadData);


/*
equation index: 7140
type: SIMPLE_ASSIGN
y[447] = r_init[447] * sin(theta[447] + 0.00788)
*/
void SpiralGalaxy_eqFunction_7140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7140};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2446]] /* y[447] STATE(1,vy[447]) */) = ((data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */)) * (sin((data->simulationInfo->realParameter[1953] /* theta[447] PARAM */) + 0.00788));
  TRACE_POP
}

/*
equation index: 7141
type: SIMPLE_ASSIGN
x[447] = r_init[447] * cos(theta[447] + 0.00788)
*/
void SpiralGalaxy_eqFunction_7141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1946]] /* x[447] STATE(1,vx[447]) */) = ((data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */)) * (cos((data->simulationInfo->realParameter[1953] /* theta[447] PARAM */) + 0.00788));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12466(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12467(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12470(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12469(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12468(DATA *data, threadData_t *threadData);


/*
equation index: 7147
type: SIMPLE_ASSIGN
vx[447] = (-sin(theta[447])) * r_init[447] * omega_c[447]
*/
void SpiralGalaxy_eqFunction_7147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[446]] /* vx[447] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1953] /* theta[447] PARAM */)))) * (((data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */)) * ((data->simulationInfo->realParameter[951] /* omega_c[447] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12463(DATA *data, threadData_t *threadData);


/*
equation index: 7149
type: SIMPLE_ASSIGN
vy[447] = cos(theta[447]) * r_init[447] * omega_c[447]
*/
void SpiralGalaxy_eqFunction_7149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[946]] /* vy[447] STATE(1) */) = (cos((data->simulationInfo->realParameter[1953] /* theta[447] PARAM */))) * (((data->simulationInfo->realParameter[1452] /* r_init[447] PARAM */)) * ((data->simulationInfo->realParameter[951] /* omega_c[447] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12462(DATA *data, threadData_t *threadData);


/*
equation index: 7151
type: SIMPLE_ASSIGN
vz[447] = 0.0
*/
void SpiralGalaxy_eqFunction_7151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7151};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1446]] /* vz[447] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12461(DATA *data, threadData_t *threadData);


/*
equation index: 7153
type: SIMPLE_ASSIGN
z[448] = 0.03168
*/
void SpiralGalaxy_eqFunction_7153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7153};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2947]] /* z[448] STATE(1,vz[448]) */) = 0.03168;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12474(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12475(DATA *data, threadData_t *threadData);


/*
equation index: 7156
type: SIMPLE_ASSIGN
y[448] = r_init[448] * sin(theta[448] + 0.00792)
*/
void SpiralGalaxy_eqFunction_7156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7156};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2447]] /* y[448] STATE(1,vy[448]) */) = ((data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */)) * (sin((data->simulationInfo->realParameter[1954] /* theta[448] PARAM */) + 0.00792));
  TRACE_POP
}

/*
equation index: 7157
type: SIMPLE_ASSIGN
x[448] = r_init[448] * cos(theta[448] + 0.00792)
*/
void SpiralGalaxy_eqFunction_7157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1947]] /* x[448] STATE(1,vx[448]) */) = ((data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */)) * (cos((data->simulationInfo->realParameter[1954] /* theta[448] PARAM */) + 0.00792));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12476(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12477(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12480(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12479(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12478(DATA *data, threadData_t *threadData);


/*
equation index: 7163
type: SIMPLE_ASSIGN
vx[448] = (-sin(theta[448])) * r_init[448] * omega_c[448]
*/
void SpiralGalaxy_eqFunction_7163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[447]] /* vx[448] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1954] /* theta[448] PARAM */)))) * (((data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */)) * ((data->simulationInfo->realParameter[952] /* omega_c[448] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12473(DATA *data, threadData_t *threadData);


/*
equation index: 7165
type: SIMPLE_ASSIGN
vy[448] = cos(theta[448]) * r_init[448] * omega_c[448]
*/
void SpiralGalaxy_eqFunction_7165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[947]] /* vy[448] STATE(1) */) = (cos((data->simulationInfo->realParameter[1954] /* theta[448] PARAM */))) * (((data->simulationInfo->realParameter[1453] /* r_init[448] PARAM */)) * ((data->simulationInfo->realParameter[952] /* omega_c[448] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12472(DATA *data, threadData_t *threadData);


/*
equation index: 7167
type: SIMPLE_ASSIGN
vz[448] = 0.0
*/
void SpiralGalaxy_eqFunction_7167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7167};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1447]] /* vz[448] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12471(DATA *data, threadData_t *threadData);


/*
equation index: 7169
type: SIMPLE_ASSIGN
z[449] = 0.03184
*/
void SpiralGalaxy_eqFunction_7169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7169};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2948]] /* z[449] STATE(1,vz[449]) */) = 0.03184;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12484(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12485(DATA *data, threadData_t *threadData);


/*
equation index: 7172
type: SIMPLE_ASSIGN
y[449] = r_init[449] * sin(theta[449] + 0.00796)
*/
void SpiralGalaxy_eqFunction_7172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7172};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2448]] /* y[449] STATE(1,vy[449]) */) = ((data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */)) * (sin((data->simulationInfo->realParameter[1955] /* theta[449] PARAM */) + 0.00796));
  TRACE_POP
}

/*
equation index: 7173
type: SIMPLE_ASSIGN
x[449] = r_init[449] * cos(theta[449] + 0.00796)
*/
void SpiralGalaxy_eqFunction_7173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1948]] /* x[449] STATE(1,vx[449]) */) = ((data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */)) * (cos((data->simulationInfo->realParameter[1955] /* theta[449] PARAM */) + 0.00796));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12486(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12487(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12490(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12489(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12488(DATA *data, threadData_t *threadData);


/*
equation index: 7179
type: SIMPLE_ASSIGN
vx[449] = (-sin(theta[449])) * r_init[449] * omega_c[449]
*/
void SpiralGalaxy_eqFunction_7179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[448]] /* vx[449] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1955] /* theta[449] PARAM */)))) * (((data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */)) * ((data->simulationInfo->realParameter[953] /* omega_c[449] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12483(DATA *data, threadData_t *threadData);


/*
equation index: 7181
type: SIMPLE_ASSIGN
vy[449] = cos(theta[449]) * r_init[449] * omega_c[449]
*/
void SpiralGalaxy_eqFunction_7181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[948]] /* vy[449] STATE(1) */) = (cos((data->simulationInfo->realParameter[1955] /* theta[449] PARAM */))) * (((data->simulationInfo->realParameter[1454] /* r_init[449] PARAM */)) * ((data->simulationInfo->realParameter[953] /* omega_c[449] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12482(DATA *data, threadData_t *threadData);


/*
equation index: 7183
type: SIMPLE_ASSIGN
vz[449] = 0.0
*/
void SpiralGalaxy_eqFunction_7183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7183};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1448]] /* vz[449] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12481(DATA *data, threadData_t *threadData);


/*
equation index: 7185
type: SIMPLE_ASSIGN
z[450] = 0.032
*/
void SpiralGalaxy_eqFunction_7185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7185};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2949]] /* z[450] STATE(1,vz[450]) */) = 0.032;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12494(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12495(DATA *data, threadData_t *threadData);


/*
equation index: 7188
type: SIMPLE_ASSIGN
y[450] = r_init[450] * sin(theta[450] + 0.008)
*/
void SpiralGalaxy_eqFunction_7188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7188};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2449]] /* y[450] STATE(1,vy[450]) */) = ((data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */)) * (sin((data->simulationInfo->realParameter[1956] /* theta[450] PARAM */) + 0.008));
  TRACE_POP
}

/*
equation index: 7189
type: SIMPLE_ASSIGN
x[450] = r_init[450] * cos(theta[450] + 0.008)
*/
void SpiralGalaxy_eqFunction_7189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1949]] /* x[450] STATE(1,vx[450]) */) = ((data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */)) * (cos((data->simulationInfo->realParameter[1956] /* theta[450] PARAM */) + 0.008));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12496(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12497(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12500(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12499(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12498(DATA *data, threadData_t *threadData);


/*
equation index: 7195
type: SIMPLE_ASSIGN
vx[450] = (-sin(theta[450])) * r_init[450] * omega_c[450]
*/
void SpiralGalaxy_eqFunction_7195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[449]] /* vx[450] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1956] /* theta[450] PARAM */)))) * (((data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */)) * ((data->simulationInfo->realParameter[954] /* omega_c[450] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12493(DATA *data, threadData_t *threadData);


/*
equation index: 7197
type: SIMPLE_ASSIGN
vy[450] = cos(theta[450]) * r_init[450] * omega_c[450]
*/
void SpiralGalaxy_eqFunction_7197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[949]] /* vy[450] STATE(1) */) = (cos((data->simulationInfo->realParameter[1956] /* theta[450] PARAM */))) * (((data->simulationInfo->realParameter[1455] /* r_init[450] PARAM */)) * ((data->simulationInfo->realParameter[954] /* omega_c[450] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12492(DATA *data, threadData_t *threadData);


/*
equation index: 7199
type: SIMPLE_ASSIGN
vz[450] = 0.0
*/
void SpiralGalaxy_eqFunction_7199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7199};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1449]] /* vz[450] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12491(DATA *data, threadData_t *threadData);


/*
equation index: 7201
type: SIMPLE_ASSIGN
z[451] = 0.03216000000000001
*/
void SpiralGalaxy_eqFunction_7201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7201};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2950]] /* z[451] STATE(1,vz[451]) */) = 0.03216000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12504(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12505(DATA *data, threadData_t *threadData);


/*
equation index: 7204
type: SIMPLE_ASSIGN
y[451] = r_init[451] * sin(theta[451] + 0.00804)
*/
void SpiralGalaxy_eqFunction_7204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7204};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2450]] /* y[451] STATE(1,vy[451]) */) = ((data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */)) * (sin((data->simulationInfo->realParameter[1957] /* theta[451] PARAM */) + 0.00804));
  TRACE_POP
}

/*
equation index: 7205
type: SIMPLE_ASSIGN
x[451] = r_init[451] * cos(theta[451] + 0.00804)
*/
void SpiralGalaxy_eqFunction_7205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1950]] /* x[451] STATE(1,vx[451]) */) = ((data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */)) * (cos((data->simulationInfo->realParameter[1957] /* theta[451] PARAM */) + 0.00804));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12506(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12507(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12510(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12509(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12508(DATA *data, threadData_t *threadData);


/*
equation index: 7211
type: SIMPLE_ASSIGN
vx[451] = (-sin(theta[451])) * r_init[451] * omega_c[451]
*/
void SpiralGalaxy_eqFunction_7211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[450]] /* vx[451] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1957] /* theta[451] PARAM */)))) * (((data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */)) * ((data->simulationInfo->realParameter[955] /* omega_c[451] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12503(DATA *data, threadData_t *threadData);


/*
equation index: 7213
type: SIMPLE_ASSIGN
vy[451] = cos(theta[451]) * r_init[451] * omega_c[451]
*/
void SpiralGalaxy_eqFunction_7213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[950]] /* vy[451] STATE(1) */) = (cos((data->simulationInfo->realParameter[1957] /* theta[451] PARAM */))) * (((data->simulationInfo->realParameter[1456] /* r_init[451] PARAM */)) * ((data->simulationInfo->realParameter[955] /* omega_c[451] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12502(DATA *data, threadData_t *threadData);


/*
equation index: 7215
type: SIMPLE_ASSIGN
vz[451] = 0.0
*/
void SpiralGalaxy_eqFunction_7215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7215};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1450]] /* vz[451] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12501(DATA *data, threadData_t *threadData);


/*
equation index: 7217
type: SIMPLE_ASSIGN
z[452] = 0.03232
*/
void SpiralGalaxy_eqFunction_7217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7217};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2951]] /* z[452] STATE(1,vz[452]) */) = 0.03232;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12514(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12515(DATA *data, threadData_t *threadData);


/*
equation index: 7220
type: SIMPLE_ASSIGN
y[452] = r_init[452] * sin(theta[452] + 0.00808)
*/
void SpiralGalaxy_eqFunction_7220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7220};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2451]] /* y[452] STATE(1,vy[452]) */) = ((data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */)) * (sin((data->simulationInfo->realParameter[1958] /* theta[452] PARAM */) + 0.00808));
  TRACE_POP
}

/*
equation index: 7221
type: SIMPLE_ASSIGN
x[452] = r_init[452] * cos(theta[452] + 0.00808)
*/
void SpiralGalaxy_eqFunction_7221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1951]] /* x[452] STATE(1,vx[452]) */) = ((data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */)) * (cos((data->simulationInfo->realParameter[1958] /* theta[452] PARAM */) + 0.00808));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12516(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12517(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12520(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12519(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12518(DATA *data, threadData_t *threadData);


/*
equation index: 7227
type: SIMPLE_ASSIGN
vx[452] = (-sin(theta[452])) * r_init[452] * omega_c[452]
*/
void SpiralGalaxy_eqFunction_7227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[451]] /* vx[452] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1958] /* theta[452] PARAM */)))) * (((data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */)) * ((data->simulationInfo->realParameter[956] /* omega_c[452] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12513(DATA *data, threadData_t *threadData);


/*
equation index: 7229
type: SIMPLE_ASSIGN
vy[452] = cos(theta[452]) * r_init[452] * omega_c[452]
*/
void SpiralGalaxy_eqFunction_7229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[951]] /* vy[452] STATE(1) */) = (cos((data->simulationInfo->realParameter[1958] /* theta[452] PARAM */))) * (((data->simulationInfo->realParameter[1457] /* r_init[452] PARAM */)) * ((data->simulationInfo->realParameter[956] /* omega_c[452] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12512(DATA *data, threadData_t *threadData);


/*
equation index: 7231
type: SIMPLE_ASSIGN
vz[452] = 0.0
*/
void SpiralGalaxy_eqFunction_7231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7231};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1451]] /* vz[452] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12511(DATA *data, threadData_t *threadData);


/*
equation index: 7233
type: SIMPLE_ASSIGN
z[453] = 0.03248
*/
void SpiralGalaxy_eqFunction_7233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7233};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2952]] /* z[453] STATE(1,vz[453]) */) = 0.03248;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12524(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12525(DATA *data, threadData_t *threadData);


/*
equation index: 7236
type: SIMPLE_ASSIGN
y[453] = r_init[453] * sin(theta[453] + 0.00812)
*/
void SpiralGalaxy_eqFunction_7236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7236};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2452]] /* y[453] STATE(1,vy[453]) */) = ((data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */)) * (sin((data->simulationInfo->realParameter[1959] /* theta[453] PARAM */) + 0.00812));
  TRACE_POP
}

/*
equation index: 7237
type: SIMPLE_ASSIGN
x[453] = r_init[453] * cos(theta[453] + 0.00812)
*/
void SpiralGalaxy_eqFunction_7237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1952]] /* x[453] STATE(1,vx[453]) */) = ((data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */)) * (cos((data->simulationInfo->realParameter[1959] /* theta[453] PARAM */) + 0.00812));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12526(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12527(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12530(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12529(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12528(DATA *data, threadData_t *threadData);


/*
equation index: 7243
type: SIMPLE_ASSIGN
vx[453] = (-sin(theta[453])) * r_init[453] * omega_c[453]
*/
void SpiralGalaxy_eqFunction_7243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[452]] /* vx[453] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1959] /* theta[453] PARAM */)))) * (((data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */)) * ((data->simulationInfo->realParameter[957] /* omega_c[453] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12523(DATA *data, threadData_t *threadData);


/*
equation index: 7245
type: SIMPLE_ASSIGN
vy[453] = cos(theta[453]) * r_init[453] * omega_c[453]
*/
void SpiralGalaxy_eqFunction_7245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[952]] /* vy[453] STATE(1) */) = (cos((data->simulationInfo->realParameter[1959] /* theta[453] PARAM */))) * (((data->simulationInfo->realParameter[1458] /* r_init[453] PARAM */)) * ((data->simulationInfo->realParameter[957] /* omega_c[453] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12522(DATA *data, threadData_t *threadData);


/*
equation index: 7247
type: SIMPLE_ASSIGN
vz[453] = 0.0
*/
void SpiralGalaxy_eqFunction_7247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7247};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1452]] /* vz[453] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12521(DATA *data, threadData_t *threadData);


/*
equation index: 7249
type: SIMPLE_ASSIGN
z[454] = 0.03264
*/
void SpiralGalaxy_eqFunction_7249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7249};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2953]] /* z[454] STATE(1,vz[454]) */) = 0.03264;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12534(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12535(DATA *data, threadData_t *threadData);


/*
equation index: 7252
type: SIMPLE_ASSIGN
y[454] = r_init[454] * sin(theta[454] + 0.00816)
*/
void SpiralGalaxy_eqFunction_7252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7252};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2453]] /* y[454] STATE(1,vy[454]) */) = ((data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */)) * (sin((data->simulationInfo->realParameter[1960] /* theta[454] PARAM */) + 0.00816));
  TRACE_POP
}

/*
equation index: 7253
type: SIMPLE_ASSIGN
x[454] = r_init[454] * cos(theta[454] + 0.00816)
*/
void SpiralGalaxy_eqFunction_7253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1953]] /* x[454] STATE(1,vx[454]) */) = ((data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */)) * (cos((data->simulationInfo->realParameter[1960] /* theta[454] PARAM */) + 0.00816));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12536(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12537(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12540(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12539(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12538(DATA *data, threadData_t *threadData);


/*
equation index: 7259
type: SIMPLE_ASSIGN
vx[454] = (-sin(theta[454])) * r_init[454] * omega_c[454]
*/
void SpiralGalaxy_eqFunction_7259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[453]] /* vx[454] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1960] /* theta[454] PARAM */)))) * (((data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */)) * ((data->simulationInfo->realParameter[958] /* omega_c[454] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12533(DATA *data, threadData_t *threadData);


/*
equation index: 7261
type: SIMPLE_ASSIGN
vy[454] = cos(theta[454]) * r_init[454] * omega_c[454]
*/
void SpiralGalaxy_eqFunction_7261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[953]] /* vy[454] STATE(1) */) = (cos((data->simulationInfo->realParameter[1960] /* theta[454] PARAM */))) * (((data->simulationInfo->realParameter[1459] /* r_init[454] PARAM */)) * ((data->simulationInfo->realParameter[958] /* omega_c[454] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12532(DATA *data, threadData_t *threadData);


/*
equation index: 7263
type: SIMPLE_ASSIGN
vz[454] = 0.0
*/
void SpiralGalaxy_eqFunction_7263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7263};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1453]] /* vz[454] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12531(DATA *data, threadData_t *threadData);


/*
equation index: 7265
type: SIMPLE_ASSIGN
z[455] = 0.0328
*/
void SpiralGalaxy_eqFunction_7265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7265};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2954]] /* z[455] STATE(1,vz[455]) */) = 0.0328;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12544(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12545(DATA *data, threadData_t *threadData);


/*
equation index: 7268
type: SIMPLE_ASSIGN
y[455] = r_init[455] * sin(theta[455] + 0.0082)
*/
void SpiralGalaxy_eqFunction_7268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7268};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2454]] /* y[455] STATE(1,vy[455]) */) = ((data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */)) * (sin((data->simulationInfo->realParameter[1961] /* theta[455] PARAM */) + 0.0082));
  TRACE_POP
}

/*
equation index: 7269
type: SIMPLE_ASSIGN
x[455] = r_init[455] * cos(theta[455] + 0.0082)
*/
void SpiralGalaxy_eqFunction_7269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1954]] /* x[455] STATE(1,vx[455]) */) = ((data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */)) * (cos((data->simulationInfo->realParameter[1961] /* theta[455] PARAM */) + 0.0082));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12546(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12547(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12550(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12549(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12548(DATA *data, threadData_t *threadData);


/*
equation index: 7275
type: SIMPLE_ASSIGN
vx[455] = (-sin(theta[455])) * r_init[455] * omega_c[455]
*/
void SpiralGalaxy_eqFunction_7275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[454]] /* vx[455] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1961] /* theta[455] PARAM */)))) * (((data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */)) * ((data->simulationInfo->realParameter[959] /* omega_c[455] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12543(DATA *data, threadData_t *threadData);


/*
equation index: 7277
type: SIMPLE_ASSIGN
vy[455] = cos(theta[455]) * r_init[455] * omega_c[455]
*/
void SpiralGalaxy_eqFunction_7277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[954]] /* vy[455] STATE(1) */) = (cos((data->simulationInfo->realParameter[1961] /* theta[455] PARAM */))) * (((data->simulationInfo->realParameter[1460] /* r_init[455] PARAM */)) * ((data->simulationInfo->realParameter[959] /* omega_c[455] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12542(DATA *data, threadData_t *threadData);


/*
equation index: 7279
type: SIMPLE_ASSIGN
vz[455] = 0.0
*/
void SpiralGalaxy_eqFunction_7279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7279};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1454]] /* vz[455] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12541(DATA *data, threadData_t *threadData);


/*
equation index: 7281
type: SIMPLE_ASSIGN
z[456] = 0.03296
*/
void SpiralGalaxy_eqFunction_7281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7281};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2955]] /* z[456] STATE(1,vz[456]) */) = 0.03296;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12554(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12555(DATA *data, threadData_t *threadData);


/*
equation index: 7284
type: SIMPLE_ASSIGN
y[456] = r_init[456] * sin(theta[456] + 0.00824)
*/
void SpiralGalaxy_eqFunction_7284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7284};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2455]] /* y[456] STATE(1,vy[456]) */) = ((data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */)) * (sin((data->simulationInfo->realParameter[1962] /* theta[456] PARAM */) + 0.00824));
  TRACE_POP
}

/*
equation index: 7285
type: SIMPLE_ASSIGN
x[456] = r_init[456] * cos(theta[456] + 0.00824)
*/
void SpiralGalaxy_eqFunction_7285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1955]] /* x[456] STATE(1,vx[456]) */) = ((data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */)) * (cos((data->simulationInfo->realParameter[1962] /* theta[456] PARAM */) + 0.00824));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12556(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12557(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12560(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12559(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12558(DATA *data, threadData_t *threadData);


/*
equation index: 7291
type: SIMPLE_ASSIGN
vx[456] = (-sin(theta[456])) * r_init[456] * omega_c[456]
*/
void SpiralGalaxy_eqFunction_7291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[455]] /* vx[456] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1962] /* theta[456] PARAM */)))) * (((data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */)) * ((data->simulationInfo->realParameter[960] /* omega_c[456] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12553(DATA *data, threadData_t *threadData);


/*
equation index: 7293
type: SIMPLE_ASSIGN
vy[456] = cos(theta[456]) * r_init[456] * omega_c[456]
*/
void SpiralGalaxy_eqFunction_7293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[955]] /* vy[456] STATE(1) */) = (cos((data->simulationInfo->realParameter[1962] /* theta[456] PARAM */))) * (((data->simulationInfo->realParameter[1461] /* r_init[456] PARAM */)) * ((data->simulationInfo->realParameter[960] /* omega_c[456] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12552(DATA *data, threadData_t *threadData);


/*
equation index: 7295
type: SIMPLE_ASSIGN
vz[456] = 0.0
*/
void SpiralGalaxy_eqFunction_7295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1455]] /* vz[456] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12551(DATA *data, threadData_t *threadData);


/*
equation index: 7297
type: SIMPLE_ASSIGN
z[457] = 0.033120000000000004
*/
void SpiralGalaxy_eqFunction_7297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2956]] /* z[457] STATE(1,vz[457]) */) = 0.033120000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12564(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12565(DATA *data, threadData_t *threadData);


/*
equation index: 7300
type: SIMPLE_ASSIGN
y[457] = r_init[457] * sin(theta[457] + 0.008280000000000001)
*/
void SpiralGalaxy_eqFunction_7300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7300};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2456]] /* y[457] STATE(1,vy[457]) */) = ((data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */)) * (sin((data->simulationInfo->realParameter[1963] /* theta[457] PARAM */) + 0.008280000000000001));
  TRACE_POP
}

/*
equation index: 7301
type: SIMPLE_ASSIGN
x[457] = r_init[457] * cos(theta[457] + 0.008280000000000001)
*/
void SpiralGalaxy_eqFunction_7301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1956]] /* x[457] STATE(1,vx[457]) */) = ((data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */)) * (cos((data->simulationInfo->realParameter[1963] /* theta[457] PARAM */) + 0.008280000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12566(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12567(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12570(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12569(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12568(DATA *data, threadData_t *threadData);


/*
equation index: 7307
type: SIMPLE_ASSIGN
vx[457] = (-sin(theta[457])) * r_init[457] * omega_c[457]
*/
void SpiralGalaxy_eqFunction_7307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[456]] /* vx[457] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1963] /* theta[457] PARAM */)))) * (((data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */)) * ((data->simulationInfo->realParameter[961] /* omega_c[457] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12563(DATA *data, threadData_t *threadData);


/*
equation index: 7309
type: SIMPLE_ASSIGN
vy[457] = cos(theta[457]) * r_init[457] * omega_c[457]
*/
void SpiralGalaxy_eqFunction_7309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[956]] /* vy[457] STATE(1) */) = (cos((data->simulationInfo->realParameter[1963] /* theta[457] PARAM */))) * (((data->simulationInfo->realParameter[1462] /* r_init[457] PARAM */)) * ((data->simulationInfo->realParameter[961] /* omega_c[457] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12562(DATA *data, threadData_t *threadData);


/*
equation index: 7311
type: SIMPLE_ASSIGN
vz[457] = 0.0
*/
void SpiralGalaxy_eqFunction_7311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1456]] /* vz[457] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12561(DATA *data, threadData_t *threadData);


/*
equation index: 7313
type: SIMPLE_ASSIGN
z[458] = 0.033280000000000004
*/
void SpiralGalaxy_eqFunction_7313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2957]] /* z[458] STATE(1,vz[458]) */) = 0.033280000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12574(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12575(DATA *data, threadData_t *threadData);


/*
equation index: 7316
type: SIMPLE_ASSIGN
y[458] = r_init[458] * sin(theta[458] + 0.008320000000000001)
*/
void SpiralGalaxy_eqFunction_7316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7316};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2457]] /* y[458] STATE(1,vy[458]) */) = ((data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */)) * (sin((data->simulationInfo->realParameter[1964] /* theta[458] PARAM */) + 0.008320000000000001));
  TRACE_POP
}

/*
equation index: 7317
type: SIMPLE_ASSIGN
x[458] = r_init[458] * cos(theta[458] + 0.008320000000000001)
*/
void SpiralGalaxy_eqFunction_7317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1957]] /* x[458] STATE(1,vx[458]) */) = ((data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */)) * (cos((data->simulationInfo->realParameter[1964] /* theta[458] PARAM */) + 0.008320000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12576(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12577(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12580(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12579(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12578(DATA *data, threadData_t *threadData);


/*
equation index: 7323
type: SIMPLE_ASSIGN
vx[458] = (-sin(theta[458])) * r_init[458] * omega_c[458]
*/
void SpiralGalaxy_eqFunction_7323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[457]] /* vx[458] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1964] /* theta[458] PARAM */)))) * (((data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */)) * ((data->simulationInfo->realParameter[962] /* omega_c[458] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12573(DATA *data, threadData_t *threadData);


/*
equation index: 7325
type: SIMPLE_ASSIGN
vy[458] = cos(theta[458]) * r_init[458] * omega_c[458]
*/
void SpiralGalaxy_eqFunction_7325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[957]] /* vy[458] STATE(1) */) = (cos((data->simulationInfo->realParameter[1964] /* theta[458] PARAM */))) * (((data->simulationInfo->realParameter[1463] /* r_init[458] PARAM */)) * ((data->simulationInfo->realParameter[962] /* omega_c[458] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12572(DATA *data, threadData_t *threadData);


/*
equation index: 7327
type: SIMPLE_ASSIGN
vz[458] = 0.0
*/
void SpiralGalaxy_eqFunction_7327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7327};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1457]] /* vz[458] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12571(DATA *data, threadData_t *threadData);


/*
equation index: 7329
type: SIMPLE_ASSIGN
z[459] = 0.033440000000000004
*/
void SpiralGalaxy_eqFunction_7329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7329};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2958]] /* z[459] STATE(1,vz[459]) */) = 0.033440000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12584(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12585(DATA *data, threadData_t *threadData);


/*
equation index: 7332
type: SIMPLE_ASSIGN
y[459] = r_init[459] * sin(theta[459] + 0.008360000000000001)
*/
void SpiralGalaxy_eqFunction_7332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7332};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2458]] /* y[459] STATE(1,vy[459]) */) = ((data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */)) * (sin((data->simulationInfo->realParameter[1965] /* theta[459] PARAM */) + 0.008360000000000001));
  TRACE_POP
}

/*
equation index: 7333
type: SIMPLE_ASSIGN
x[459] = r_init[459] * cos(theta[459] + 0.008360000000000001)
*/
void SpiralGalaxy_eqFunction_7333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1958]] /* x[459] STATE(1,vx[459]) */) = ((data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */)) * (cos((data->simulationInfo->realParameter[1965] /* theta[459] PARAM */) + 0.008360000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12586(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12587(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12590(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12589(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12588(DATA *data, threadData_t *threadData);


/*
equation index: 7339
type: SIMPLE_ASSIGN
vx[459] = (-sin(theta[459])) * r_init[459] * omega_c[459]
*/
void SpiralGalaxy_eqFunction_7339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[458]] /* vx[459] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1965] /* theta[459] PARAM */)))) * (((data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */)) * ((data->simulationInfo->realParameter[963] /* omega_c[459] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12583(DATA *data, threadData_t *threadData);


/*
equation index: 7341
type: SIMPLE_ASSIGN
vy[459] = cos(theta[459]) * r_init[459] * omega_c[459]
*/
void SpiralGalaxy_eqFunction_7341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[958]] /* vy[459] STATE(1) */) = (cos((data->simulationInfo->realParameter[1965] /* theta[459] PARAM */))) * (((data->simulationInfo->realParameter[1464] /* r_init[459] PARAM */)) * ((data->simulationInfo->realParameter[963] /* omega_c[459] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12582(DATA *data, threadData_t *threadData);


/*
equation index: 7343
type: SIMPLE_ASSIGN
vz[459] = 0.0
*/
void SpiralGalaxy_eqFunction_7343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1458]] /* vz[459] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12581(DATA *data, threadData_t *threadData);


/*
equation index: 7345
type: SIMPLE_ASSIGN
z[460] = 0.033600000000000005
*/
void SpiralGalaxy_eqFunction_7345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2959]] /* z[460] STATE(1,vz[460]) */) = 0.033600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12594(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12595(DATA *data, threadData_t *threadData);


/*
equation index: 7348
type: SIMPLE_ASSIGN
y[460] = r_init[460] * sin(theta[460] + 0.008400000000000001)
*/
void SpiralGalaxy_eqFunction_7348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7348};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2459]] /* y[460] STATE(1,vy[460]) */) = ((data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */)) * (sin((data->simulationInfo->realParameter[1966] /* theta[460] PARAM */) + 0.008400000000000001));
  TRACE_POP
}

/*
equation index: 7349
type: SIMPLE_ASSIGN
x[460] = r_init[460] * cos(theta[460] + 0.008400000000000001)
*/
void SpiralGalaxy_eqFunction_7349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1959]] /* x[460] STATE(1,vx[460]) */) = ((data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */)) * (cos((data->simulationInfo->realParameter[1966] /* theta[460] PARAM */) + 0.008400000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12596(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12597(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12600(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12599(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12598(DATA *data, threadData_t *threadData);


/*
equation index: 7355
type: SIMPLE_ASSIGN
vx[460] = (-sin(theta[460])) * r_init[460] * omega_c[460]
*/
void SpiralGalaxy_eqFunction_7355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[459]] /* vx[460] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1966] /* theta[460] PARAM */)))) * (((data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */)) * ((data->simulationInfo->realParameter[964] /* omega_c[460] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12593(DATA *data, threadData_t *threadData);


/*
equation index: 7357
type: SIMPLE_ASSIGN
vy[460] = cos(theta[460]) * r_init[460] * omega_c[460]
*/
void SpiralGalaxy_eqFunction_7357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[959]] /* vy[460] STATE(1) */) = (cos((data->simulationInfo->realParameter[1966] /* theta[460] PARAM */))) * (((data->simulationInfo->realParameter[1465] /* r_init[460] PARAM */)) * ((data->simulationInfo->realParameter[964] /* omega_c[460] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12592(DATA *data, threadData_t *threadData);


/*
equation index: 7359
type: SIMPLE_ASSIGN
vz[460] = 0.0
*/
void SpiralGalaxy_eqFunction_7359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7359};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1459]] /* vz[460] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12591(DATA *data, threadData_t *threadData);


/*
equation index: 7361
type: SIMPLE_ASSIGN
z[461] = 0.033760000000000005
*/
void SpiralGalaxy_eqFunction_7361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7361};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2960]] /* z[461] STATE(1,vz[461]) */) = 0.033760000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12604(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12605(DATA *data, threadData_t *threadData);


/*
equation index: 7364
type: SIMPLE_ASSIGN
y[461] = r_init[461] * sin(theta[461] + 0.008440000000000001)
*/
void SpiralGalaxy_eqFunction_7364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7364};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2460]] /* y[461] STATE(1,vy[461]) */) = ((data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */)) * (sin((data->simulationInfo->realParameter[1967] /* theta[461] PARAM */) + 0.008440000000000001));
  TRACE_POP
}

/*
equation index: 7365
type: SIMPLE_ASSIGN
x[461] = r_init[461] * cos(theta[461] + 0.008440000000000001)
*/
void SpiralGalaxy_eqFunction_7365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1960]] /* x[461] STATE(1,vx[461]) */) = ((data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */)) * (cos((data->simulationInfo->realParameter[1967] /* theta[461] PARAM */) + 0.008440000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12606(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12607(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12610(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12609(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12608(DATA *data, threadData_t *threadData);


/*
equation index: 7371
type: SIMPLE_ASSIGN
vx[461] = (-sin(theta[461])) * r_init[461] * omega_c[461]
*/
void SpiralGalaxy_eqFunction_7371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7371};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[460]] /* vx[461] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1967] /* theta[461] PARAM */)))) * (((data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */)) * ((data->simulationInfo->realParameter[965] /* omega_c[461] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12603(DATA *data, threadData_t *threadData);


/*
equation index: 7373
type: SIMPLE_ASSIGN
vy[461] = cos(theta[461]) * r_init[461] * omega_c[461]
*/
void SpiralGalaxy_eqFunction_7373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[960]] /* vy[461] STATE(1) */) = (cos((data->simulationInfo->realParameter[1967] /* theta[461] PARAM */))) * (((data->simulationInfo->realParameter[1466] /* r_init[461] PARAM */)) * ((data->simulationInfo->realParameter[965] /* omega_c[461] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12602(DATA *data, threadData_t *threadData);


/*
equation index: 7375
type: SIMPLE_ASSIGN
vz[461] = 0.0
*/
void SpiralGalaxy_eqFunction_7375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7375};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1460]] /* vz[461] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12601(DATA *data, threadData_t *threadData);


/*
equation index: 7377
type: SIMPLE_ASSIGN
z[462] = 0.033920000000000006
*/
void SpiralGalaxy_eqFunction_7377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7377};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2961]] /* z[462] STATE(1,vz[462]) */) = 0.033920000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12614(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12615(DATA *data, threadData_t *threadData);


/*
equation index: 7380
type: SIMPLE_ASSIGN
y[462] = r_init[462] * sin(theta[462] + 0.008480000000000001)
*/
void SpiralGalaxy_eqFunction_7380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7380};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2461]] /* y[462] STATE(1,vy[462]) */) = ((data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */)) * (sin((data->simulationInfo->realParameter[1968] /* theta[462] PARAM */) + 0.008480000000000001));
  TRACE_POP
}

/*
equation index: 7381
type: SIMPLE_ASSIGN
x[462] = r_init[462] * cos(theta[462] + 0.008480000000000001)
*/
void SpiralGalaxy_eqFunction_7381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1961]] /* x[462] STATE(1,vx[462]) */) = ((data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */)) * (cos((data->simulationInfo->realParameter[1968] /* theta[462] PARAM */) + 0.008480000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12616(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12617(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12620(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12619(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12618(DATA *data, threadData_t *threadData);


/*
equation index: 7387
type: SIMPLE_ASSIGN
vx[462] = (-sin(theta[462])) * r_init[462] * omega_c[462]
*/
void SpiralGalaxy_eqFunction_7387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[461]] /* vx[462] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1968] /* theta[462] PARAM */)))) * (((data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */)) * ((data->simulationInfo->realParameter[966] /* omega_c[462] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12613(DATA *data, threadData_t *threadData);


/*
equation index: 7389
type: SIMPLE_ASSIGN
vy[462] = cos(theta[462]) * r_init[462] * omega_c[462]
*/
void SpiralGalaxy_eqFunction_7389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[961]] /* vy[462] STATE(1) */) = (cos((data->simulationInfo->realParameter[1968] /* theta[462] PARAM */))) * (((data->simulationInfo->realParameter[1467] /* r_init[462] PARAM */)) * ((data->simulationInfo->realParameter[966] /* omega_c[462] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12612(DATA *data, threadData_t *threadData);


/*
equation index: 7391
type: SIMPLE_ASSIGN
vz[462] = 0.0
*/
void SpiralGalaxy_eqFunction_7391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7391};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1461]] /* vz[462] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12611(DATA *data, threadData_t *threadData);


/*
equation index: 7393
type: SIMPLE_ASSIGN
z[463] = 0.034080000000000006
*/
void SpiralGalaxy_eqFunction_7393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7393};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2962]] /* z[463] STATE(1,vz[463]) */) = 0.034080000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12624(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12625(DATA *data, threadData_t *threadData);


/*
equation index: 7396
type: SIMPLE_ASSIGN
y[463] = r_init[463] * sin(theta[463] + 0.008520000000000002)
*/
void SpiralGalaxy_eqFunction_7396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7396};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2462]] /* y[463] STATE(1,vy[463]) */) = ((data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */)) * (sin((data->simulationInfo->realParameter[1969] /* theta[463] PARAM */) + 0.008520000000000002));
  TRACE_POP
}

/*
equation index: 7397
type: SIMPLE_ASSIGN
x[463] = r_init[463] * cos(theta[463] + 0.008520000000000002)
*/
void SpiralGalaxy_eqFunction_7397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1962]] /* x[463] STATE(1,vx[463]) */) = ((data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */)) * (cos((data->simulationInfo->realParameter[1969] /* theta[463] PARAM */) + 0.008520000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12626(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12627(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12630(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12629(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12628(DATA *data, threadData_t *threadData);


/*
equation index: 7403
type: SIMPLE_ASSIGN
vx[463] = (-sin(theta[463])) * r_init[463] * omega_c[463]
*/
void SpiralGalaxy_eqFunction_7403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[462]] /* vx[463] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1969] /* theta[463] PARAM */)))) * (((data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */)) * ((data->simulationInfo->realParameter[967] /* omega_c[463] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12623(DATA *data, threadData_t *threadData);


/*
equation index: 7405
type: SIMPLE_ASSIGN
vy[463] = cos(theta[463]) * r_init[463] * omega_c[463]
*/
void SpiralGalaxy_eqFunction_7405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[962]] /* vy[463] STATE(1) */) = (cos((data->simulationInfo->realParameter[1969] /* theta[463] PARAM */))) * (((data->simulationInfo->realParameter[1468] /* r_init[463] PARAM */)) * ((data->simulationInfo->realParameter[967] /* omega_c[463] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12622(DATA *data, threadData_t *threadData);


/*
equation index: 7407
type: SIMPLE_ASSIGN
vz[463] = 0.0
*/
void SpiralGalaxy_eqFunction_7407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7407};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1462]] /* vz[463] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12621(DATA *data, threadData_t *threadData);


/*
equation index: 7409
type: SIMPLE_ASSIGN
z[464] = 0.03424000000000001
*/
void SpiralGalaxy_eqFunction_7409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7409};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2963]] /* z[464] STATE(1,vz[464]) */) = 0.03424000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12634(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12635(DATA *data, threadData_t *threadData);


/*
equation index: 7412
type: SIMPLE_ASSIGN
y[464] = r_init[464] * sin(theta[464] + 0.008560000000000002)
*/
void SpiralGalaxy_eqFunction_7412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7412};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2463]] /* y[464] STATE(1,vy[464]) */) = ((data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */)) * (sin((data->simulationInfo->realParameter[1970] /* theta[464] PARAM */) + 0.008560000000000002));
  TRACE_POP
}

/*
equation index: 7413
type: SIMPLE_ASSIGN
x[464] = r_init[464] * cos(theta[464] + 0.008560000000000002)
*/
void SpiralGalaxy_eqFunction_7413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1963]] /* x[464] STATE(1,vx[464]) */) = ((data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */)) * (cos((data->simulationInfo->realParameter[1970] /* theta[464] PARAM */) + 0.008560000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12636(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12637(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12640(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12639(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12638(DATA *data, threadData_t *threadData);


/*
equation index: 7419
type: SIMPLE_ASSIGN
vx[464] = (-sin(theta[464])) * r_init[464] * omega_c[464]
*/
void SpiralGalaxy_eqFunction_7419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[463]] /* vx[464] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1970] /* theta[464] PARAM */)))) * (((data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */)) * ((data->simulationInfo->realParameter[968] /* omega_c[464] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12633(DATA *data, threadData_t *threadData);


/*
equation index: 7421
type: SIMPLE_ASSIGN
vy[464] = cos(theta[464]) * r_init[464] * omega_c[464]
*/
void SpiralGalaxy_eqFunction_7421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[963]] /* vy[464] STATE(1) */) = (cos((data->simulationInfo->realParameter[1970] /* theta[464] PARAM */))) * (((data->simulationInfo->realParameter[1469] /* r_init[464] PARAM */)) * ((data->simulationInfo->realParameter[968] /* omega_c[464] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12632(DATA *data, threadData_t *threadData);


/*
equation index: 7423
type: SIMPLE_ASSIGN
vz[464] = 0.0
*/
void SpiralGalaxy_eqFunction_7423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7423};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1463]] /* vz[464] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12631(DATA *data, threadData_t *threadData);


/*
equation index: 7425
type: SIMPLE_ASSIGN
z[465] = 0.0344
*/
void SpiralGalaxy_eqFunction_7425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7425};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2964]] /* z[465] STATE(1,vz[465]) */) = 0.0344;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12644(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12645(DATA *data, threadData_t *threadData);


/*
equation index: 7428
type: SIMPLE_ASSIGN
y[465] = r_init[465] * sin(theta[465] + 0.008600000000000002)
*/
void SpiralGalaxy_eqFunction_7428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7428};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2464]] /* y[465] STATE(1,vy[465]) */) = ((data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */)) * (sin((data->simulationInfo->realParameter[1971] /* theta[465] PARAM */) + 0.008600000000000002));
  TRACE_POP
}

/*
equation index: 7429
type: SIMPLE_ASSIGN
x[465] = r_init[465] * cos(theta[465] + 0.008600000000000002)
*/
void SpiralGalaxy_eqFunction_7429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1964]] /* x[465] STATE(1,vx[465]) */) = ((data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */)) * (cos((data->simulationInfo->realParameter[1971] /* theta[465] PARAM */) + 0.008600000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12646(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12647(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12650(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12649(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12648(DATA *data, threadData_t *threadData);


/*
equation index: 7435
type: SIMPLE_ASSIGN
vx[465] = (-sin(theta[465])) * r_init[465] * omega_c[465]
*/
void SpiralGalaxy_eqFunction_7435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[464]] /* vx[465] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1971] /* theta[465] PARAM */)))) * (((data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */)) * ((data->simulationInfo->realParameter[969] /* omega_c[465] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12643(DATA *data, threadData_t *threadData);


/*
equation index: 7437
type: SIMPLE_ASSIGN
vy[465] = cos(theta[465]) * r_init[465] * omega_c[465]
*/
void SpiralGalaxy_eqFunction_7437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[964]] /* vy[465] STATE(1) */) = (cos((data->simulationInfo->realParameter[1971] /* theta[465] PARAM */))) * (((data->simulationInfo->realParameter[1470] /* r_init[465] PARAM */)) * ((data->simulationInfo->realParameter[969] /* omega_c[465] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12642(DATA *data, threadData_t *threadData);


/*
equation index: 7439
type: SIMPLE_ASSIGN
vz[465] = 0.0
*/
void SpiralGalaxy_eqFunction_7439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7439};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1464]] /* vz[465] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12641(DATA *data, threadData_t *threadData);


/*
equation index: 7441
type: SIMPLE_ASSIGN
z[466] = 0.03456000000000001
*/
void SpiralGalaxy_eqFunction_7441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7441};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2965]] /* z[466] STATE(1,vz[466]) */) = 0.03456000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12654(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12655(DATA *data, threadData_t *threadData);


/*
equation index: 7444
type: SIMPLE_ASSIGN
y[466] = r_init[466] * sin(theta[466] + 0.008640000000000002)
*/
void SpiralGalaxy_eqFunction_7444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7444};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2465]] /* y[466] STATE(1,vy[466]) */) = ((data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */)) * (sin((data->simulationInfo->realParameter[1972] /* theta[466] PARAM */) + 0.008640000000000002));
  TRACE_POP
}

/*
equation index: 7445
type: SIMPLE_ASSIGN
x[466] = r_init[466] * cos(theta[466] + 0.008640000000000002)
*/
void SpiralGalaxy_eqFunction_7445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1965]] /* x[466] STATE(1,vx[466]) */) = ((data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */)) * (cos((data->simulationInfo->realParameter[1972] /* theta[466] PARAM */) + 0.008640000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12656(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12657(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12660(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12659(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12658(DATA *data, threadData_t *threadData);


/*
equation index: 7451
type: SIMPLE_ASSIGN
vx[466] = (-sin(theta[466])) * r_init[466] * omega_c[466]
*/
void SpiralGalaxy_eqFunction_7451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[465]] /* vx[466] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1972] /* theta[466] PARAM */)))) * (((data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */)) * ((data->simulationInfo->realParameter[970] /* omega_c[466] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12653(DATA *data, threadData_t *threadData);


/*
equation index: 7453
type: SIMPLE_ASSIGN
vy[466] = cos(theta[466]) * r_init[466] * omega_c[466]
*/
void SpiralGalaxy_eqFunction_7453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[965]] /* vy[466] STATE(1) */) = (cos((data->simulationInfo->realParameter[1972] /* theta[466] PARAM */))) * (((data->simulationInfo->realParameter[1471] /* r_init[466] PARAM */)) * ((data->simulationInfo->realParameter[970] /* omega_c[466] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12652(DATA *data, threadData_t *threadData);


/*
equation index: 7455
type: SIMPLE_ASSIGN
vz[466] = 0.0
*/
void SpiralGalaxy_eqFunction_7455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7455};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1465]] /* vz[466] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12651(DATA *data, threadData_t *threadData);


/*
equation index: 7457
type: SIMPLE_ASSIGN
z[467] = 0.03472000000000001
*/
void SpiralGalaxy_eqFunction_7457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7457};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2966]] /* z[467] STATE(1,vz[467]) */) = 0.03472000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12664(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12665(DATA *data, threadData_t *threadData);


/*
equation index: 7460
type: SIMPLE_ASSIGN
y[467] = r_init[467] * sin(theta[467] + 0.008680000000000002)
*/
void SpiralGalaxy_eqFunction_7460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7460};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2466]] /* y[467] STATE(1,vy[467]) */) = ((data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */)) * (sin((data->simulationInfo->realParameter[1973] /* theta[467] PARAM */) + 0.008680000000000002));
  TRACE_POP
}

/*
equation index: 7461
type: SIMPLE_ASSIGN
x[467] = r_init[467] * cos(theta[467] + 0.008680000000000002)
*/
void SpiralGalaxy_eqFunction_7461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1966]] /* x[467] STATE(1,vx[467]) */) = ((data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */)) * (cos((data->simulationInfo->realParameter[1973] /* theta[467] PARAM */) + 0.008680000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12666(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12667(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12670(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12669(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12668(DATA *data, threadData_t *threadData);


/*
equation index: 7467
type: SIMPLE_ASSIGN
vx[467] = (-sin(theta[467])) * r_init[467] * omega_c[467]
*/
void SpiralGalaxy_eqFunction_7467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[466]] /* vx[467] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1973] /* theta[467] PARAM */)))) * (((data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */)) * ((data->simulationInfo->realParameter[971] /* omega_c[467] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12663(DATA *data, threadData_t *threadData);


/*
equation index: 7469
type: SIMPLE_ASSIGN
vy[467] = cos(theta[467]) * r_init[467] * omega_c[467]
*/
void SpiralGalaxy_eqFunction_7469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[966]] /* vy[467] STATE(1) */) = (cos((data->simulationInfo->realParameter[1973] /* theta[467] PARAM */))) * (((data->simulationInfo->realParameter[1472] /* r_init[467] PARAM */)) * ((data->simulationInfo->realParameter[971] /* omega_c[467] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12662(DATA *data, threadData_t *threadData);


/*
equation index: 7471
type: SIMPLE_ASSIGN
vz[467] = 0.0
*/
void SpiralGalaxy_eqFunction_7471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7471};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1466]] /* vz[467] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12661(DATA *data, threadData_t *threadData);


/*
equation index: 7473
type: SIMPLE_ASSIGN
z[468] = 0.03488
*/
void SpiralGalaxy_eqFunction_7473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7473};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2967]] /* z[468] STATE(1,vz[468]) */) = 0.03488;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12674(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12675(DATA *data, threadData_t *threadData);


/*
equation index: 7476
type: SIMPLE_ASSIGN
y[468] = r_init[468] * sin(theta[468] + 0.008720000000000002)
*/
void SpiralGalaxy_eqFunction_7476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7476};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2467]] /* y[468] STATE(1,vy[468]) */) = ((data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */)) * (sin((data->simulationInfo->realParameter[1974] /* theta[468] PARAM */) + 0.008720000000000002));
  TRACE_POP
}

/*
equation index: 7477
type: SIMPLE_ASSIGN
x[468] = r_init[468] * cos(theta[468] + 0.008720000000000002)
*/
void SpiralGalaxy_eqFunction_7477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1967]] /* x[468] STATE(1,vx[468]) */) = ((data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */)) * (cos((data->simulationInfo->realParameter[1974] /* theta[468] PARAM */) + 0.008720000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12676(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12677(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12680(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12679(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12678(DATA *data, threadData_t *threadData);


/*
equation index: 7483
type: SIMPLE_ASSIGN
vx[468] = (-sin(theta[468])) * r_init[468] * omega_c[468]
*/
void SpiralGalaxy_eqFunction_7483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[467]] /* vx[468] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1974] /* theta[468] PARAM */)))) * (((data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */)) * ((data->simulationInfo->realParameter[972] /* omega_c[468] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12673(DATA *data, threadData_t *threadData);


/*
equation index: 7485
type: SIMPLE_ASSIGN
vy[468] = cos(theta[468]) * r_init[468] * omega_c[468]
*/
void SpiralGalaxy_eqFunction_7485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[967]] /* vy[468] STATE(1) */) = (cos((data->simulationInfo->realParameter[1974] /* theta[468] PARAM */))) * (((data->simulationInfo->realParameter[1473] /* r_init[468] PARAM */)) * ((data->simulationInfo->realParameter[972] /* omega_c[468] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12672(DATA *data, threadData_t *threadData);


/*
equation index: 7487
type: SIMPLE_ASSIGN
vz[468] = 0.0
*/
void SpiralGalaxy_eqFunction_7487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7487};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1467]] /* vz[468] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12671(DATA *data, threadData_t *threadData);


/*
equation index: 7489
type: SIMPLE_ASSIGN
z[469] = 0.03504
*/
void SpiralGalaxy_eqFunction_7489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7489};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2968]] /* z[469] STATE(1,vz[469]) */) = 0.03504;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12684(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12685(DATA *data, threadData_t *threadData);


/*
equation index: 7492
type: SIMPLE_ASSIGN
y[469] = r_init[469] * sin(theta[469] + 0.008759999999999999)
*/
void SpiralGalaxy_eqFunction_7492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7492};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2468]] /* y[469] STATE(1,vy[469]) */) = ((data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */)) * (sin((data->simulationInfo->realParameter[1975] /* theta[469] PARAM */) + 0.008759999999999999));
  TRACE_POP
}

/*
equation index: 7493
type: SIMPLE_ASSIGN
x[469] = r_init[469] * cos(theta[469] + 0.008759999999999999)
*/
void SpiralGalaxy_eqFunction_7493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1968]] /* x[469] STATE(1,vx[469]) */) = ((data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */)) * (cos((data->simulationInfo->realParameter[1975] /* theta[469] PARAM */) + 0.008759999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12686(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12687(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12690(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12689(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12688(DATA *data, threadData_t *threadData);


/*
equation index: 7499
type: SIMPLE_ASSIGN
vx[469] = (-sin(theta[469])) * r_init[469] * omega_c[469]
*/
void SpiralGalaxy_eqFunction_7499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7499};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[468]] /* vx[469] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1975] /* theta[469] PARAM */)))) * (((data->simulationInfo->realParameter[1474] /* r_init[469] PARAM */)) * ((data->simulationInfo->realParameter[973] /* omega_c[469] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12683(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_14(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_12379(data, threadData);
  SpiralGalaxy_eqFunction_12378(data, threadData);
  SpiralGalaxy_eqFunction_7003(data, threadData);
  SpiralGalaxy_eqFunction_12373(data, threadData);
  SpiralGalaxy_eqFunction_7005(data, threadData);
  SpiralGalaxy_eqFunction_12372(data, threadData);
  SpiralGalaxy_eqFunction_7007(data, threadData);
  SpiralGalaxy_eqFunction_12371(data, threadData);
  SpiralGalaxy_eqFunction_7009(data, threadData);
  SpiralGalaxy_eqFunction_12384(data, threadData);
  SpiralGalaxy_eqFunction_12385(data, threadData);
  SpiralGalaxy_eqFunction_7012(data, threadData);
  SpiralGalaxy_eqFunction_7013(data, threadData);
  SpiralGalaxy_eqFunction_12386(data, threadData);
  SpiralGalaxy_eqFunction_12387(data, threadData);
  SpiralGalaxy_eqFunction_12390(data, threadData);
  SpiralGalaxy_eqFunction_12389(data, threadData);
  SpiralGalaxy_eqFunction_12388(data, threadData);
  SpiralGalaxy_eqFunction_7019(data, threadData);
  SpiralGalaxy_eqFunction_12383(data, threadData);
  SpiralGalaxy_eqFunction_7021(data, threadData);
  SpiralGalaxy_eqFunction_12382(data, threadData);
  SpiralGalaxy_eqFunction_7023(data, threadData);
  SpiralGalaxy_eqFunction_12381(data, threadData);
  SpiralGalaxy_eqFunction_7025(data, threadData);
  SpiralGalaxy_eqFunction_12394(data, threadData);
  SpiralGalaxy_eqFunction_12395(data, threadData);
  SpiralGalaxy_eqFunction_7028(data, threadData);
  SpiralGalaxy_eqFunction_7029(data, threadData);
  SpiralGalaxy_eqFunction_12396(data, threadData);
  SpiralGalaxy_eqFunction_12397(data, threadData);
  SpiralGalaxy_eqFunction_12400(data, threadData);
  SpiralGalaxy_eqFunction_12399(data, threadData);
  SpiralGalaxy_eqFunction_12398(data, threadData);
  SpiralGalaxy_eqFunction_7035(data, threadData);
  SpiralGalaxy_eqFunction_12393(data, threadData);
  SpiralGalaxy_eqFunction_7037(data, threadData);
  SpiralGalaxy_eqFunction_12392(data, threadData);
  SpiralGalaxy_eqFunction_7039(data, threadData);
  SpiralGalaxy_eqFunction_12391(data, threadData);
  SpiralGalaxy_eqFunction_7041(data, threadData);
  SpiralGalaxy_eqFunction_12404(data, threadData);
  SpiralGalaxy_eqFunction_12405(data, threadData);
  SpiralGalaxy_eqFunction_7044(data, threadData);
  SpiralGalaxy_eqFunction_7045(data, threadData);
  SpiralGalaxy_eqFunction_12406(data, threadData);
  SpiralGalaxy_eqFunction_12407(data, threadData);
  SpiralGalaxy_eqFunction_12410(data, threadData);
  SpiralGalaxy_eqFunction_12409(data, threadData);
  SpiralGalaxy_eqFunction_12408(data, threadData);
  SpiralGalaxy_eqFunction_7051(data, threadData);
  SpiralGalaxy_eqFunction_12403(data, threadData);
  SpiralGalaxy_eqFunction_7053(data, threadData);
  SpiralGalaxy_eqFunction_12402(data, threadData);
  SpiralGalaxy_eqFunction_7055(data, threadData);
  SpiralGalaxy_eqFunction_12401(data, threadData);
  SpiralGalaxy_eqFunction_7057(data, threadData);
  SpiralGalaxy_eqFunction_12414(data, threadData);
  SpiralGalaxy_eqFunction_12415(data, threadData);
  SpiralGalaxy_eqFunction_7060(data, threadData);
  SpiralGalaxy_eqFunction_7061(data, threadData);
  SpiralGalaxy_eqFunction_12416(data, threadData);
  SpiralGalaxy_eqFunction_12417(data, threadData);
  SpiralGalaxy_eqFunction_12420(data, threadData);
  SpiralGalaxy_eqFunction_12419(data, threadData);
  SpiralGalaxy_eqFunction_12418(data, threadData);
  SpiralGalaxy_eqFunction_7067(data, threadData);
  SpiralGalaxy_eqFunction_12413(data, threadData);
  SpiralGalaxy_eqFunction_7069(data, threadData);
  SpiralGalaxy_eqFunction_12412(data, threadData);
  SpiralGalaxy_eqFunction_7071(data, threadData);
  SpiralGalaxy_eqFunction_12411(data, threadData);
  SpiralGalaxy_eqFunction_7073(data, threadData);
  SpiralGalaxy_eqFunction_12424(data, threadData);
  SpiralGalaxy_eqFunction_12425(data, threadData);
  SpiralGalaxy_eqFunction_7076(data, threadData);
  SpiralGalaxy_eqFunction_7077(data, threadData);
  SpiralGalaxy_eqFunction_12426(data, threadData);
  SpiralGalaxy_eqFunction_12427(data, threadData);
  SpiralGalaxy_eqFunction_12430(data, threadData);
  SpiralGalaxy_eqFunction_12429(data, threadData);
  SpiralGalaxy_eqFunction_12428(data, threadData);
  SpiralGalaxy_eqFunction_7083(data, threadData);
  SpiralGalaxy_eqFunction_12423(data, threadData);
  SpiralGalaxy_eqFunction_7085(data, threadData);
  SpiralGalaxy_eqFunction_12422(data, threadData);
  SpiralGalaxy_eqFunction_7087(data, threadData);
  SpiralGalaxy_eqFunction_12421(data, threadData);
  SpiralGalaxy_eqFunction_7089(data, threadData);
  SpiralGalaxy_eqFunction_12434(data, threadData);
  SpiralGalaxy_eqFunction_12435(data, threadData);
  SpiralGalaxy_eqFunction_7092(data, threadData);
  SpiralGalaxy_eqFunction_7093(data, threadData);
  SpiralGalaxy_eqFunction_12436(data, threadData);
  SpiralGalaxy_eqFunction_12437(data, threadData);
  SpiralGalaxy_eqFunction_12440(data, threadData);
  SpiralGalaxy_eqFunction_12439(data, threadData);
  SpiralGalaxy_eqFunction_12438(data, threadData);
  SpiralGalaxy_eqFunction_7099(data, threadData);
  SpiralGalaxy_eqFunction_12433(data, threadData);
  SpiralGalaxy_eqFunction_7101(data, threadData);
  SpiralGalaxy_eqFunction_12432(data, threadData);
  SpiralGalaxy_eqFunction_7103(data, threadData);
  SpiralGalaxy_eqFunction_12431(data, threadData);
  SpiralGalaxy_eqFunction_7105(data, threadData);
  SpiralGalaxy_eqFunction_12444(data, threadData);
  SpiralGalaxy_eqFunction_12445(data, threadData);
  SpiralGalaxy_eqFunction_7108(data, threadData);
  SpiralGalaxy_eqFunction_7109(data, threadData);
  SpiralGalaxy_eqFunction_12446(data, threadData);
  SpiralGalaxy_eqFunction_12447(data, threadData);
  SpiralGalaxy_eqFunction_12450(data, threadData);
  SpiralGalaxy_eqFunction_12449(data, threadData);
  SpiralGalaxy_eqFunction_12448(data, threadData);
  SpiralGalaxy_eqFunction_7115(data, threadData);
  SpiralGalaxy_eqFunction_12443(data, threadData);
  SpiralGalaxy_eqFunction_7117(data, threadData);
  SpiralGalaxy_eqFunction_12442(data, threadData);
  SpiralGalaxy_eqFunction_7119(data, threadData);
  SpiralGalaxy_eqFunction_12441(data, threadData);
  SpiralGalaxy_eqFunction_7121(data, threadData);
  SpiralGalaxy_eqFunction_12454(data, threadData);
  SpiralGalaxy_eqFunction_12455(data, threadData);
  SpiralGalaxy_eqFunction_7124(data, threadData);
  SpiralGalaxy_eqFunction_7125(data, threadData);
  SpiralGalaxy_eqFunction_12456(data, threadData);
  SpiralGalaxy_eqFunction_12457(data, threadData);
  SpiralGalaxy_eqFunction_12460(data, threadData);
  SpiralGalaxy_eqFunction_12459(data, threadData);
  SpiralGalaxy_eqFunction_12458(data, threadData);
  SpiralGalaxy_eqFunction_7131(data, threadData);
  SpiralGalaxy_eqFunction_12453(data, threadData);
  SpiralGalaxy_eqFunction_7133(data, threadData);
  SpiralGalaxy_eqFunction_12452(data, threadData);
  SpiralGalaxy_eqFunction_7135(data, threadData);
  SpiralGalaxy_eqFunction_12451(data, threadData);
  SpiralGalaxy_eqFunction_7137(data, threadData);
  SpiralGalaxy_eqFunction_12464(data, threadData);
  SpiralGalaxy_eqFunction_12465(data, threadData);
  SpiralGalaxy_eqFunction_7140(data, threadData);
  SpiralGalaxy_eqFunction_7141(data, threadData);
  SpiralGalaxy_eqFunction_12466(data, threadData);
  SpiralGalaxy_eqFunction_12467(data, threadData);
  SpiralGalaxy_eqFunction_12470(data, threadData);
  SpiralGalaxy_eqFunction_12469(data, threadData);
  SpiralGalaxy_eqFunction_12468(data, threadData);
  SpiralGalaxy_eqFunction_7147(data, threadData);
  SpiralGalaxy_eqFunction_12463(data, threadData);
  SpiralGalaxy_eqFunction_7149(data, threadData);
  SpiralGalaxy_eqFunction_12462(data, threadData);
  SpiralGalaxy_eqFunction_7151(data, threadData);
  SpiralGalaxy_eqFunction_12461(data, threadData);
  SpiralGalaxy_eqFunction_7153(data, threadData);
  SpiralGalaxy_eqFunction_12474(data, threadData);
  SpiralGalaxy_eqFunction_12475(data, threadData);
  SpiralGalaxy_eqFunction_7156(data, threadData);
  SpiralGalaxy_eqFunction_7157(data, threadData);
  SpiralGalaxy_eqFunction_12476(data, threadData);
  SpiralGalaxy_eqFunction_12477(data, threadData);
  SpiralGalaxy_eqFunction_12480(data, threadData);
  SpiralGalaxy_eqFunction_12479(data, threadData);
  SpiralGalaxy_eqFunction_12478(data, threadData);
  SpiralGalaxy_eqFunction_7163(data, threadData);
  SpiralGalaxy_eqFunction_12473(data, threadData);
  SpiralGalaxy_eqFunction_7165(data, threadData);
  SpiralGalaxy_eqFunction_12472(data, threadData);
  SpiralGalaxy_eqFunction_7167(data, threadData);
  SpiralGalaxy_eqFunction_12471(data, threadData);
  SpiralGalaxy_eqFunction_7169(data, threadData);
  SpiralGalaxy_eqFunction_12484(data, threadData);
  SpiralGalaxy_eqFunction_12485(data, threadData);
  SpiralGalaxy_eqFunction_7172(data, threadData);
  SpiralGalaxy_eqFunction_7173(data, threadData);
  SpiralGalaxy_eqFunction_12486(data, threadData);
  SpiralGalaxy_eqFunction_12487(data, threadData);
  SpiralGalaxy_eqFunction_12490(data, threadData);
  SpiralGalaxy_eqFunction_12489(data, threadData);
  SpiralGalaxy_eqFunction_12488(data, threadData);
  SpiralGalaxy_eqFunction_7179(data, threadData);
  SpiralGalaxy_eqFunction_12483(data, threadData);
  SpiralGalaxy_eqFunction_7181(data, threadData);
  SpiralGalaxy_eqFunction_12482(data, threadData);
  SpiralGalaxy_eqFunction_7183(data, threadData);
  SpiralGalaxy_eqFunction_12481(data, threadData);
  SpiralGalaxy_eqFunction_7185(data, threadData);
  SpiralGalaxy_eqFunction_12494(data, threadData);
  SpiralGalaxy_eqFunction_12495(data, threadData);
  SpiralGalaxy_eqFunction_7188(data, threadData);
  SpiralGalaxy_eqFunction_7189(data, threadData);
  SpiralGalaxy_eqFunction_12496(data, threadData);
  SpiralGalaxy_eqFunction_12497(data, threadData);
  SpiralGalaxy_eqFunction_12500(data, threadData);
  SpiralGalaxy_eqFunction_12499(data, threadData);
  SpiralGalaxy_eqFunction_12498(data, threadData);
  SpiralGalaxy_eqFunction_7195(data, threadData);
  SpiralGalaxy_eqFunction_12493(data, threadData);
  SpiralGalaxy_eqFunction_7197(data, threadData);
  SpiralGalaxy_eqFunction_12492(data, threadData);
  SpiralGalaxy_eqFunction_7199(data, threadData);
  SpiralGalaxy_eqFunction_12491(data, threadData);
  SpiralGalaxy_eqFunction_7201(data, threadData);
  SpiralGalaxy_eqFunction_12504(data, threadData);
  SpiralGalaxy_eqFunction_12505(data, threadData);
  SpiralGalaxy_eqFunction_7204(data, threadData);
  SpiralGalaxy_eqFunction_7205(data, threadData);
  SpiralGalaxy_eqFunction_12506(data, threadData);
  SpiralGalaxy_eqFunction_12507(data, threadData);
  SpiralGalaxy_eqFunction_12510(data, threadData);
  SpiralGalaxy_eqFunction_12509(data, threadData);
  SpiralGalaxy_eqFunction_12508(data, threadData);
  SpiralGalaxy_eqFunction_7211(data, threadData);
  SpiralGalaxy_eqFunction_12503(data, threadData);
  SpiralGalaxy_eqFunction_7213(data, threadData);
  SpiralGalaxy_eqFunction_12502(data, threadData);
  SpiralGalaxy_eqFunction_7215(data, threadData);
  SpiralGalaxy_eqFunction_12501(data, threadData);
  SpiralGalaxy_eqFunction_7217(data, threadData);
  SpiralGalaxy_eqFunction_12514(data, threadData);
  SpiralGalaxy_eqFunction_12515(data, threadData);
  SpiralGalaxy_eqFunction_7220(data, threadData);
  SpiralGalaxy_eqFunction_7221(data, threadData);
  SpiralGalaxy_eqFunction_12516(data, threadData);
  SpiralGalaxy_eqFunction_12517(data, threadData);
  SpiralGalaxy_eqFunction_12520(data, threadData);
  SpiralGalaxy_eqFunction_12519(data, threadData);
  SpiralGalaxy_eqFunction_12518(data, threadData);
  SpiralGalaxy_eqFunction_7227(data, threadData);
  SpiralGalaxy_eqFunction_12513(data, threadData);
  SpiralGalaxy_eqFunction_7229(data, threadData);
  SpiralGalaxy_eqFunction_12512(data, threadData);
  SpiralGalaxy_eqFunction_7231(data, threadData);
  SpiralGalaxy_eqFunction_12511(data, threadData);
  SpiralGalaxy_eqFunction_7233(data, threadData);
  SpiralGalaxy_eqFunction_12524(data, threadData);
  SpiralGalaxy_eqFunction_12525(data, threadData);
  SpiralGalaxy_eqFunction_7236(data, threadData);
  SpiralGalaxy_eqFunction_7237(data, threadData);
  SpiralGalaxy_eqFunction_12526(data, threadData);
  SpiralGalaxy_eqFunction_12527(data, threadData);
  SpiralGalaxy_eqFunction_12530(data, threadData);
  SpiralGalaxy_eqFunction_12529(data, threadData);
  SpiralGalaxy_eqFunction_12528(data, threadData);
  SpiralGalaxy_eqFunction_7243(data, threadData);
  SpiralGalaxy_eqFunction_12523(data, threadData);
  SpiralGalaxy_eqFunction_7245(data, threadData);
  SpiralGalaxy_eqFunction_12522(data, threadData);
  SpiralGalaxy_eqFunction_7247(data, threadData);
  SpiralGalaxy_eqFunction_12521(data, threadData);
  SpiralGalaxy_eqFunction_7249(data, threadData);
  SpiralGalaxy_eqFunction_12534(data, threadData);
  SpiralGalaxy_eqFunction_12535(data, threadData);
  SpiralGalaxy_eqFunction_7252(data, threadData);
  SpiralGalaxy_eqFunction_7253(data, threadData);
  SpiralGalaxy_eqFunction_12536(data, threadData);
  SpiralGalaxy_eqFunction_12537(data, threadData);
  SpiralGalaxy_eqFunction_12540(data, threadData);
  SpiralGalaxy_eqFunction_12539(data, threadData);
  SpiralGalaxy_eqFunction_12538(data, threadData);
  SpiralGalaxy_eqFunction_7259(data, threadData);
  SpiralGalaxy_eqFunction_12533(data, threadData);
  SpiralGalaxy_eqFunction_7261(data, threadData);
  SpiralGalaxy_eqFunction_12532(data, threadData);
  SpiralGalaxy_eqFunction_7263(data, threadData);
  SpiralGalaxy_eqFunction_12531(data, threadData);
  SpiralGalaxy_eqFunction_7265(data, threadData);
  SpiralGalaxy_eqFunction_12544(data, threadData);
  SpiralGalaxy_eqFunction_12545(data, threadData);
  SpiralGalaxy_eqFunction_7268(data, threadData);
  SpiralGalaxy_eqFunction_7269(data, threadData);
  SpiralGalaxy_eqFunction_12546(data, threadData);
  SpiralGalaxy_eqFunction_12547(data, threadData);
  SpiralGalaxy_eqFunction_12550(data, threadData);
  SpiralGalaxy_eqFunction_12549(data, threadData);
  SpiralGalaxy_eqFunction_12548(data, threadData);
  SpiralGalaxy_eqFunction_7275(data, threadData);
  SpiralGalaxy_eqFunction_12543(data, threadData);
  SpiralGalaxy_eqFunction_7277(data, threadData);
  SpiralGalaxy_eqFunction_12542(data, threadData);
  SpiralGalaxy_eqFunction_7279(data, threadData);
  SpiralGalaxy_eqFunction_12541(data, threadData);
  SpiralGalaxy_eqFunction_7281(data, threadData);
  SpiralGalaxy_eqFunction_12554(data, threadData);
  SpiralGalaxy_eqFunction_12555(data, threadData);
  SpiralGalaxy_eqFunction_7284(data, threadData);
  SpiralGalaxy_eqFunction_7285(data, threadData);
  SpiralGalaxy_eqFunction_12556(data, threadData);
  SpiralGalaxy_eqFunction_12557(data, threadData);
  SpiralGalaxy_eqFunction_12560(data, threadData);
  SpiralGalaxy_eqFunction_12559(data, threadData);
  SpiralGalaxy_eqFunction_12558(data, threadData);
  SpiralGalaxy_eqFunction_7291(data, threadData);
  SpiralGalaxy_eqFunction_12553(data, threadData);
  SpiralGalaxy_eqFunction_7293(data, threadData);
  SpiralGalaxy_eqFunction_12552(data, threadData);
  SpiralGalaxy_eqFunction_7295(data, threadData);
  SpiralGalaxy_eqFunction_12551(data, threadData);
  SpiralGalaxy_eqFunction_7297(data, threadData);
  SpiralGalaxy_eqFunction_12564(data, threadData);
  SpiralGalaxy_eqFunction_12565(data, threadData);
  SpiralGalaxy_eqFunction_7300(data, threadData);
  SpiralGalaxy_eqFunction_7301(data, threadData);
  SpiralGalaxy_eqFunction_12566(data, threadData);
  SpiralGalaxy_eqFunction_12567(data, threadData);
  SpiralGalaxy_eqFunction_12570(data, threadData);
  SpiralGalaxy_eqFunction_12569(data, threadData);
  SpiralGalaxy_eqFunction_12568(data, threadData);
  SpiralGalaxy_eqFunction_7307(data, threadData);
  SpiralGalaxy_eqFunction_12563(data, threadData);
  SpiralGalaxy_eqFunction_7309(data, threadData);
  SpiralGalaxy_eqFunction_12562(data, threadData);
  SpiralGalaxy_eqFunction_7311(data, threadData);
  SpiralGalaxy_eqFunction_12561(data, threadData);
  SpiralGalaxy_eqFunction_7313(data, threadData);
  SpiralGalaxy_eqFunction_12574(data, threadData);
  SpiralGalaxy_eqFunction_12575(data, threadData);
  SpiralGalaxy_eqFunction_7316(data, threadData);
  SpiralGalaxy_eqFunction_7317(data, threadData);
  SpiralGalaxy_eqFunction_12576(data, threadData);
  SpiralGalaxy_eqFunction_12577(data, threadData);
  SpiralGalaxy_eqFunction_12580(data, threadData);
  SpiralGalaxy_eqFunction_12579(data, threadData);
  SpiralGalaxy_eqFunction_12578(data, threadData);
  SpiralGalaxy_eqFunction_7323(data, threadData);
  SpiralGalaxy_eqFunction_12573(data, threadData);
  SpiralGalaxy_eqFunction_7325(data, threadData);
  SpiralGalaxy_eqFunction_12572(data, threadData);
  SpiralGalaxy_eqFunction_7327(data, threadData);
  SpiralGalaxy_eqFunction_12571(data, threadData);
  SpiralGalaxy_eqFunction_7329(data, threadData);
  SpiralGalaxy_eqFunction_12584(data, threadData);
  SpiralGalaxy_eqFunction_12585(data, threadData);
  SpiralGalaxy_eqFunction_7332(data, threadData);
  SpiralGalaxy_eqFunction_7333(data, threadData);
  SpiralGalaxy_eqFunction_12586(data, threadData);
  SpiralGalaxy_eqFunction_12587(data, threadData);
  SpiralGalaxy_eqFunction_12590(data, threadData);
  SpiralGalaxy_eqFunction_12589(data, threadData);
  SpiralGalaxy_eqFunction_12588(data, threadData);
  SpiralGalaxy_eqFunction_7339(data, threadData);
  SpiralGalaxy_eqFunction_12583(data, threadData);
  SpiralGalaxy_eqFunction_7341(data, threadData);
  SpiralGalaxy_eqFunction_12582(data, threadData);
  SpiralGalaxy_eqFunction_7343(data, threadData);
  SpiralGalaxy_eqFunction_12581(data, threadData);
  SpiralGalaxy_eqFunction_7345(data, threadData);
  SpiralGalaxy_eqFunction_12594(data, threadData);
  SpiralGalaxy_eqFunction_12595(data, threadData);
  SpiralGalaxy_eqFunction_7348(data, threadData);
  SpiralGalaxy_eqFunction_7349(data, threadData);
  SpiralGalaxy_eqFunction_12596(data, threadData);
  SpiralGalaxy_eqFunction_12597(data, threadData);
  SpiralGalaxy_eqFunction_12600(data, threadData);
  SpiralGalaxy_eqFunction_12599(data, threadData);
  SpiralGalaxy_eqFunction_12598(data, threadData);
  SpiralGalaxy_eqFunction_7355(data, threadData);
  SpiralGalaxy_eqFunction_12593(data, threadData);
  SpiralGalaxy_eqFunction_7357(data, threadData);
  SpiralGalaxy_eqFunction_12592(data, threadData);
  SpiralGalaxy_eqFunction_7359(data, threadData);
  SpiralGalaxy_eqFunction_12591(data, threadData);
  SpiralGalaxy_eqFunction_7361(data, threadData);
  SpiralGalaxy_eqFunction_12604(data, threadData);
  SpiralGalaxy_eqFunction_12605(data, threadData);
  SpiralGalaxy_eqFunction_7364(data, threadData);
  SpiralGalaxy_eqFunction_7365(data, threadData);
  SpiralGalaxy_eqFunction_12606(data, threadData);
  SpiralGalaxy_eqFunction_12607(data, threadData);
  SpiralGalaxy_eqFunction_12610(data, threadData);
  SpiralGalaxy_eqFunction_12609(data, threadData);
  SpiralGalaxy_eqFunction_12608(data, threadData);
  SpiralGalaxy_eqFunction_7371(data, threadData);
  SpiralGalaxy_eqFunction_12603(data, threadData);
  SpiralGalaxy_eqFunction_7373(data, threadData);
  SpiralGalaxy_eqFunction_12602(data, threadData);
  SpiralGalaxy_eqFunction_7375(data, threadData);
  SpiralGalaxy_eqFunction_12601(data, threadData);
  SpiralGalaxy_eqFunction_7377(data, threadData);
  SpiralGalaxy_eqFunction_12614(data, threadData);
  SpiralGalaxy_eqFunction_12615(data, threadData);
  SpiralGalaxy_eqFunction_7380(data, threadData);
  SpiralGalaxy_eqFunction_7381(data, threadData);
  SpiralGalaxy_eqFunction_12616(data, threadData);
  SpiralGalaxy_eqFunction_12617(data, threadData);
  SpiralGalaxy_eqFunction_12620(data, threadData);
  SpiralGalaxy_eqFunction_12619(data, threadData);
  SpiralGalaxy_eqFunction_12618(data, threadData);
  SpiralGalaxy_eqFunction_7387(data, threadData);
  SpiralGalaxy_eqFunction_12613(data, threadData);
  SpiralGalaxy_eqFunction_7389(data, threadData);
  SpiralGalaxy_eqFunction_12612(data, threadData);
  SpiralGalaxy_eqFunction_7391(data, threadData);
  SpiralGalaxy_eqFunction_12611(data, threadData);
  SpiralGalaxy_eqFunction_7393(data, threadData);
  SpiralGalaxy_eqFunction_12624(data, threadData);
  SpiralGalaxy_eqFunction_12625(data, threadData);
  SpiralGalaxy_eqFunction_7396(data, threadData);
  SpiralGalaxy_eqFunction_7397(data, threadData);
  SpiralGalaxy_eqFunction_12626(data, threadData);
  SpiralGalaxy_eqFunction_12627(data, threadData);
  SpiralGalaxy_eqFunction_12630(data, threadData);
  SpiralGalaxy_eqFunction_12629(data, threadData);
  SpiralGalaxy_eqFunction_12628(data, threadData);
  SpiralGalaxy_eqFunction_7403(data, threadData);
  SpiralGalaxy_eqFunction_12623(data, threadData);
  SpiralGalaxy_eqFunction_7405(data, threadData);
  SpiralGalaxy_eqFunction_12622(data, threadData);
  SpiralGalaxy_eqFunction_7407(data, threadData);
  SpiralGalaxy_eqFunction_12621(data, threadData);
  SpiralGalaxy_eqFunction_7409(data, threadData);
  SpiralGalaxy_eqFunction_12634(data, threadData);
  SpiralGalaxy_eqFunction_12635(data, threadData);
  SpiralGalaxy_eqFunction_7412(data, threadData);
  SpiralGalaxy_eqFunction_7413(data, threadData);
  SpiralGalaxy_eqFunction_12636(data, threadData);
  SpiralGalaxy_eqFunction_12637(data, threadData);
  SpiralGalaxy_eqFunction_12640(data, threadData);
  SpiralGalaxy_eqFunction_12639(data, threadData);
  SpiralGalaxy_eqFunction_12638(data, threadData);
  SpiralGalaxy_eqFunction_7419(data, threadData);
  SpiralGalaxy_eqFunction_12633(data, threadData);
  SpiralGalaxy_eqFunction_7421(data, threadData);
  SpiralGalaxy_eqFunction_12632(data, threadData);
  SpiralGalaxy_eqFunction_7423(data, threadData);
  SpiralGalaxy_eqFunction_12631(data, threadData);
  SpiralGalaxy_eqFunction_7425(data, threadData);
  SpiralGalaxy_eqFunction_12644(data, threadData);
  SpiralGalaxy_eqFunction_12645(data, threadData);
  SpiralGalaxy_eqFunction_7428(data, threadData);
  SpiralGalaxy_eqFunction_7429(data, threadData);
  SpiralGalaxy_eqFunction_12646(data, threadData);
  SpiralGalaxy_eqFunction_12647(data, threadData);
  SpiralGalaxy_eqFunction_12650(data, threadData);
  SpiralGalaxy_eqFunction_12649(data, threadData);
  SpiralGalaxy_eqFunction_12648(data, threadData);
  SpiralGalaxy_eqFunction_7435(data, threadData);
  SpiralGalaxy_eqFunction_12643(data, threadData);
  SpiralGalaxy_eqFunction_7437(data, threadData);
  SpiralGalaxy_eqFunction_12642(data, threadData);
  SpiralGalaxy_eqFunction_7439(data, threadData);
  SpiralGalaxy_eqFunction_12641(data, threadData);
  SpiralGalaxy_eqFunction_7441(data, threadData);
  SpiralGalaxy_eqFunction_12654(data, threadData);
  SpiralGalaxy_eqFunction_12655(data, threadData);
  SpiralGalaxy_eqFunction_7444(data, threadData);
  SpiralGalaxy_eqFunction_7445(data, threadData);
  SpiralGalaxy_eqFunction_12656(data, threadData);
  SpiralGalaxy_eqFunction_12657(data, threadData);
  SpiralGalaxy_eqFunction_12660(data, threadData);
  SpiralGalaxy_eqFunction_12659(data, threadData);
  SpiralGalaxy_eqFunction_12658(data, threadData);
  SpiralGalaxy_eqFunction_7451(data, threadData);
  SpiralGalaxy_eqFunction_12653(data, threadData);
  SpiralGalaxy_eqFunction_7453(data, threadData);
  SpiralGalaxy_eqFunction_12652(data, threadData);
  SpiralGalaxy_eqFunction_7455(data, threadData);
  SpiralGalaxy_eqFunction_12651(data, threadData);
  SpiralGalaxy_eqFunction_7457(data, threadData);
  SpiralGalaxy_eqFunction_12664(data, threadData);
  SpiralGalaxy_eqFunction_12665(data, threadData);
  SpiralGalaxy_eqFunction_7460(data, threadData);
  SpiralGalaxy_eqFunction_7461(data, threadData);
  SpiralGalaxy_eqFunction_12666(data, threadData);
  SpiralGalaxy_eqFunction_12667(data, threadData);
  SpiralGalaxy_eqFunction_12670(data, threadData);
  SpiralGalaxy_eqFunction_12669(data, threadData);
  SpiralGalaxy_eqFunction_12668(data, threadData);
  SpiralGalaxy_eqFunction_7467(data, threadData);
  SpiralGalaxy_eqFunction_12663(data, threadData);
  SpiralGalaxy_eqFunction_7469(data, threadData);
  SpiralGalaxy_eqFunction_12662(data, threadData);
  SpiralGalaxy_eqFunction_7471(data, threadData);
  SpiralGalaxy_eqFunction_12661(data, threadData);
  SpiralGalaxy_eqFunction_7473(data, threadData);
  SpiralGalaxy_eqFunction_12674(data, threadData);
  SpiralGalaxy_eqFunction_12675(data, threadData);
  SpiralGalaxy_eqFunction_7476(data, threadData);
  SpiralGalaxy_eqFunction_7477(data, threadData);
  SpiralGalaxy_eqFunction_12676(data, threadData);
  SpiralGalaxy_eqFunction_12677(data, threadData);
  SpiralGalaxy_eqFunction_12680(data, threadData);
  SpiralGalaxy_eqFunction_12679(data, threadData);
  SpiralGalaxy_eqFunction_12678(data, threadData);
  SpiralGalaxy_eqFunction_7483(data, threadData);
  SpiralGalaxy_eqFunction_12673(data, threadData);
  SpiralGalaxy_eqFunction_7485(data, threadData);
  SpiralGalaxy_eqFunction_12672(data, threadData);
  SpiralGalaxy_eqFunction_7487(data, threadData);
  SpiralGalaxy_eqFunction_12671(data, threadData);
  SpiralGalaxy_eqFunction_7489(data, threadData);
  SpiralGalaxy_eqFunction_12684(data, threadData);
  SpiralGalaxy_eqFunction_12685(data, threadData);
  SpiralGalaxy_eqFunction_7492(data, threadData);
  SpiralGalaxy_eqFunction_7493(data, threadData);
  SpiralGalaxy_eqFunction_12686(data, threadData);
  SpiralGalaxy_eqFunction_12687(data, threadData);
  SpiralGalaxy_eqFunction_12690(data, threadData);
  SpiralGalaxy_eqFunction_12689(data, threadData);
  SpiralGalaxy_eqFunction_12688(data, threadData);
  SpiralGalaxy_eqFunction_7499(data, threadData);
  SpiralGalaxy_eqFunction_12683(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif