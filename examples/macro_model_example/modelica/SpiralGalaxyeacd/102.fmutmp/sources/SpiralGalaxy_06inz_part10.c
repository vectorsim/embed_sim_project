#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void SpiralGalaxy_eqFunction_11129(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11128(DATA *data, threadData_t *threadData);


/*
equation index: 5003
type: SIMPLE_ASSIGN
vx[313] = (-sin(theta[313])) * r_init[313] * omega_c[313]
*/
void SpiralGalaxy_eqFunction_5003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5003};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[312]] /* vx[313] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1819] /* theta[313] PARAM */)))) * (((data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */)) * ((data->simulationInfo->realParameter[817] /* omega_c[313] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11123(DATA *data, threadData_t *threadData);


/*
equation index: 5005
type: SIMPLE_ASSIGN
vy[313] = cos(theta[313]) * r_init[313] * omega_c[313]
*/
void SpiralGalaxy_eqFunction_5005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5005};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[812]] /* vy[313] STATE(1) */) = (cos((data->simulationInfo->realParameter[1819] /* theta[313] PARAM */))) * (((data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */)) * ((data->simulationInfo->realParameter[817] /* omega_c[313] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11122(DATA *data, threadData_t *threadData);


/*
equation index: 5007
type: SIMPLE_ASSIGN
vz[313] = 0.0
*/
void SpiralGalaxy_eqFunction_5007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5007};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1312]] /* vz[313] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11121(DATA *data, threadData_t *threadData);


/*
equation index: 5009
type: SIMPLE_ASSIGN
z[314] = 0.01024
*/
void SpiralGalaxy_eqFunction_5009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5009};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2813]] /* z[314] STATE(1,vz[314]) */) = 0.01024;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11134(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11135(DATA *data, threadData_t *threadData);


/*
equation index: 5012
type: SIMPLE_ASSIGN
y[314] = r_init[314] * sin(theta[314] + 0.00256)
*/
void SpiralGalaxy_eqFunction_5012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5012};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2313]] /* y[314] STATE(1,vy[314]) */) = ((data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */)) * (sin((data->simulationInfo->realParameter[1820] /* theta[314] PARAM */) + 0.00256));
  TRACE_POP
}

/*
equation index: 5013
type: SIMPLE_ASSIGN
x[314] = r_init[314] * cos(theta[314] + 0.00256)
*/
void SpiralGalaxy_eqFunction_5013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5013};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1813]] /* x[314] STATE(1,vx[314]) */) = ((data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */)) * (cos((data->simulationInfo->realParameter[1820] /* theta[314] PARAM */) + 0.00256));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11136(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11137(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11140(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11139(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11138(DATA *data, threadData_t *threadData);


/*
equation index: 5019
type: SIMPLE_ASSIGN
vx[314] = (-sin(theta[314])) * r_init[314] * omega_c[314]
*/
void SpiralGalaxy_eqFunction_5019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5019};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[313]] /* vx[314] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1820] /* theta[314] PARAM */)))) * (((data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */)) * ((data->simulationInfo->realParameter[818] /* omega_c[314] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11133(DATA *data, threadData_t *threadData);


/*
equation index: 5021
type: SIMPLE_ASSIGN
vy[314] = cos(theta[314]) * r_init[314] * omega_c[314]
*/
void SpiralGalaxy_eqFunction_5021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5021};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[813]] /* vy[314] STATE(1) */) = (cos((data->simulationInfo->realParameter[1820] /* theta[314] PARAM */))) * (((data->simulationInfo->realParameter[1319] /* r_init[314] PARAM */)) * ((data->simulationInfo->realParameter[818] /* omega_c[314] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11132(DATA *data, threadData_t *threadData);


/*
equation index: 5023
type: SIMPLE_ASSIGN
vz[314] = 0.0
*/
void SpiralGalaxy_eqFunction_5023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5023};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1313]] /* vz[314] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11131(DATA *data, threadData_t *threadData);


/*
equation index: 5025
type: SIMPLE_ASSIGN
z[315] = 0.010400000000000003
*/
void SpiralGalaxy_eqFunction_5025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5025};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2814]] /* z[315] STATE(1,vz[315]) */) = 0.010400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11144(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11145(DATA *data, threadData_t *threadData);


/*
equation index: 5028
type: SIMPLE_ASSIGN
y[315] = r_init[315] * sin(theta[315] + 0.0026000000000000003)
*/
void SpiralGalaxy_eqFunction_5028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5028};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2314]] /* y[315] STATE(1,vy[315]) */) = ((data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */)) * (sin((data->simulationInfo->realParameter[1821] /* theta[315] PARAM */) + 0.0026000000000000003));
  TRACE_POP
}

/*
equation index: 5029
type: SIMPLE_ASSIGN
x[315] = r_init[315] * cos(theta[315] + 0.0026000000000000003)
*/
void SpiralGalaxy_eqFunction_5029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5029};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1814]] /* x[315] STATE(1,vx[315]) */) = ((data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */)) * (cos((data->simulationInfo->realParameter[1821] /* theta[315] PARAM */) + 0.0026000000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11146(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11147(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11150(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11149(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11148(DATA *data, threadData_t *threadData);


/*
equation index: 5035
type: SIMPLE_ASSIGN
vx[315] = (-sin(theta[315])) * r_init[315] * omega_c[315]
*/
void SpiralGalaxy_eqFunction_5035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5035};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[314]] /* vx[315] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1821] /* theta[315] PARAM */)))) * (((data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */)) * ((data->simulationInfo->realParameter[819] /* omega_c[315] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11143(DATA *data, threadData_t *threadData);


/*
equation index: 5037
type: SIMPLE_ASSIGN
vy[315] = cos(theta[315]) * r_init[315] * omega_c[315]
*/
void SpiralGalaxy_eqFunction_5037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5037};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[814]] /* vy[315] STATE(1) */) = (cos((data->simulationInfo->realParameter[1821] /* theta[315] PARAM */))) * (((data->simulationInfo->realParameter[1320] /* r_init[315] PARAM */)) * ((data->simulationInfo->realParameter[819] /* omega_c[315] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11142(DATA *data, threadData_t *threadData);


/*
equation index: 5039
type: SIMPLE_ASSIGN
vz[315] = 0.0
*/
void SpiralGalaxy_eqFunction_5039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5039};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1314]] /* vz[315] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11141(DATA *data, threadData_t *threadData);


/*
equation index: 5041
type: SIMPLE_ASSIGN
z[316] = 0.01056
*/
void SpiralGalaxy_eqFunction_5041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5041};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2815]] /* z[316] STATE(1,vz[316]) */) = 0.01056;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11154(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11155(DATA *data, threadData_t *threadData);


/*
equation index: 5044
type: SIMPLE_ASSIGN
y[316] = r_init[316] * sin(theta[316] + 0.00264)
*/
void SpiralGalaxy_eqFunction_5044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5044};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2315]] /* y[316] STATE(1,vy[316]) */) = ((data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */)) * (sin((data->simulationInfo->realParameter[1822] /* theta[316] PARAM */) + 0.00264));
  TRACE_POP
}

/*
equation index: 5045
type: SIMPLE_ASSIGN
x[316] = r_init[316] * cos(theta[316] + 0.00264)
*/
void SpiralGalaxy_eqFunction_5045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5045};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1815]] /* x[316] STATE(1,vx[316]) */) = ((data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */)) * (cos((data->simulationInfo->realParameter[1822] /* theta[316] PARAM */) + 0.00264));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11156(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11157(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11160(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11159(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11158(DATA *data, threadData_t *threadData);


/*
equation index: 5051
type: SIMPLE_ASSIGN
vx[316] = (-sin(theta[316])) * r_init[316] * omega_c[316]
*/
void SpiralGalaxy_eqFunction_5051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5051};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[315]] /* vx[316] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1822] /* theta[316] PARAM */)))) * (((data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */)) * ((data->simulationInfo->realParameter[820] /* omega_c[316] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11153(DATA *data, threadData_t *threadData);


/*
equation index: 5053
type: SIMPLE_ASSIGN
vy[316] = cos(theta[316]) * r_init[316] * omega_c[316]
*/
void SpiralGalaxy_eqFunction_5053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5053};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[815]] /* vy[316] STATE(1) */) = (cos((data->simulationInfo->realParameter[1822] /* theta[316] PARAM */))) * (((data->simulationInfo->realParameter[1321] /* r_init[316] PARAM */)) * ((data->simulationInfo->realParameter[820] /* omega_c[316] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11152(DATA *data, threadData_t *threadData);


/*
equation index: 5055
type: SIMPLE_ASSIGN
vz[316] = 0.0
*/
void SpiralGalaxy_eqFunction_5055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5055};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1315]] /* vz[316] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11151(DATA *data, threadData_t *threadData);


/*
equation index: 5057
type: SIMPLE_ASSIGN
z[317] = 0.01072
*/
void SpiralGalaxy_eqFunction_5057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5057};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2816]] /* z[317] STATE(1,vz[317]) */) = 0.01072;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11164(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11165(DATA *data, threadData_t *threadData);


/*
equation index: 5060
type: SIMPLE_ASSIGN
y[317] = r_init[317] * sin(theta[317] + 0.00268)
*/
void SpiralGalaxy_eqFunction_5060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5060};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2316]] /* y[317] STATE(1,vy[317]) */) = ((data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */)) * (sin((data->simulationInfo->realParameter[1823] /* theta[317] PARAM */) + 0.00268));
  TRACE_POP
}

/*
equation index: 5061
type: SIMPLE_ASSIGN
x[317] = r_init[317] * cos(theta[317] + 0.00268)
*/
void SpiralGalaxy_eqFunction_5061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5061};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1816]] /* x[317] STATE(1,vx[317]) */) = ((data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */)) * (cos((data->simulationInfo->realParameter[1823] /* theta[317] PARAM */) + 0.00268));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11166(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11167(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11170(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11169(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11168(DATA *data, threadData_t *threadData);


/*
equation index: 5067
type: SIMPLE_ASSIGN
vx[317] = (-sin(theta[317])) * r_init[317] * omega_c[317]
*/
void SpiralGalaxy_eqFunction_5067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5067};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[316]] /* vx[317] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1823] /* theta[317] PARAM */)))) * (((data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */)) * ((data->simulationInfo->realParameter[821] /* omega_c[317] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11163(DATA *data, threadData_t *threadData);


/*
equation index: 5069
type: SIMPLE_ASSIGN
vy[317] = cos(theta[317]) * r_init[317] * omega_c[317]
*/
void SpiralGalaxy_eqFunction_5069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5069};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[816]] /* vy[317] STATE(1) */) = (cos((data->simulationInfo->realParameter[1823] /* theta[317] PARAM */))) * (((data->simulationInfo->realParameter[1322] /* r_init[317] PARAM */)) * ((data->simulationInfo->realParameter[821] /* omega_c[317] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11162(DATA *data, threadData_t *threadData);


/*
equation index: 5071
type: SIMPLE_ASSIGN
vz[317] = 0.0
*/
void SpiralGalaxy_eqFunction_5071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5071};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1316]] /* vz[317] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11161(DATA *data, threadData_t *threadData);


/*
equation index: 5073
type: SIMPLE_ASSIGN
z[318] = 0.01088
*/
void SpiralGalaxy_eqFunction_5073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5073};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2817]] /* z[318] STATE(1,vz[318]) */) = 0.01088;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11174(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11175(DATA *data, threadData_t *threadData);


/*
equation index: 5076
type: SIMPLE_ASSIGN
y[318] = r_init[318] * sin(theta[318] + 0.00272)
*/
void SpiralGalaxy_eqFunction_5076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5076};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2317]] /* y[318] STATE(1,vy[318]) */) = ((data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */)) * (sin((data->simulationInfo->realParameter[1824] /* theta[318] PARAM */) + 0.00272));
  TRACE_POP
}

/*
equation index: 5077
type: SIMPLE_ASSIGN
x[318] = r_init[318] * cos(theta[318] + 0.00272)
*/
void SpiralGalaxy_eqFunction_5077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5077};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1817]] /* x[318] STATE(1,vx[318]) */) = ((data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */)) * (cos((data->simulationInfo->realParameter[1824] /* theta[318] PARAM */) + 0.00272));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11176(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11177(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11180(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11179(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11178(DATA *data, threadData_t *threadData);


/*
equation index: 5083
type: SIMPLE_ASSIGN
vx[318] = (-sin(theta[318])) * r_init[318] * omega_c[318]
*/
void SpiralGalaxy_eqFunction_5083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5083};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[317]] /* vx[318] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1824] /* theta[318] PARAM */)))) * (((data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */)) * ((data->simulationInfo->realParameter[822] /* omega_c[318] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11173(DATA *data, threadData_t *threadData);


/*
equation index: 5085
type: SIMPLE_ASSIGN
vy[318] = cos(theta[318]) * r_init[318] * omega_c[318]
*/
void SpiralGalaxy_eqFunction_5085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5085};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[817]] /* vy[318] STATE(1) */) = (cos((data->simulationInfo->realParameter[1824] /* theta[318] PARAM */))) * (((data->simulationInfo->realParameter[1323] /* r_init[318] PARAM */)) * ((data->simulationInfo->realParameter[822] /* omega_c[318] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11172(DATA *data, threadData_t *threadData);


/*
equation index: 5087
type: SIMPLE_ASSIGN
vz[318] = 0.0
*/
void SpiralGalaxy_eqFunction_5087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5087};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1317]] /* vz[318] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11171(DATA *data, threadData_t *threadData);


/*
equation index: 5089
type: SIMPLE_ASSIGN
z[319] = 0.011040000000000001
*/
void SpiralGalaxy_eqFunction_5089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5089};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2818]] /* z[319] STATE(1,vz[319]) */) = 0.011040000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11184(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11185(DATA *data, threadData_t *threadData);


/*
equation index: 5092
type: SIMPLE_ASSIGN
y[319] = r_init[319] * sin(theta[319] + 0.0027600000000000003)
*/
void SpiralGalaxy_eqFunction_5092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5092};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2318]] /* y[319] STATE(1,vy[319]) */) = ((data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */)) * (sin((data->simulationInfo->realParameter[1825] /* theta[319] PARAM */) + 0.0027600000000000003));
  TRACE_POP
}

/*
equation index: 5093
type: SIMPLE_ASSIGN
x[319] = r_init[319] * cos(theta[319] + 0.0027600000000000003)
*/
void SpiralGalaxy_eqFunction_5093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5093};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1818]] /* x[319] STATE(1,vx[319]) */) = ((data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */)) * (cos((data->simulationInfo->realParameter[1825] /* theta[319] PARAM */) + 0.0027600000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11186(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11187(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11190(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11189(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11188(DATA *data, threadData_t *threadData);


/*
equation index: 5099
type: SIMPLE_ASSIGN
vx[319] = (-sin(theta[319])) * r_init[319] * omega_c[319]
*/
void SpiralGalaxy_eqFunction_5099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5099};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[318]] /* vx[319] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1825] /* theta[319] PARAM */)))) * (((data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */)) * ((data->simulationInfo->realParameter[823] /* omega_c[319] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11183(DATA *data, threadData_t *threadData);


/*
equation index: 5101
type: SIMPLE_ASSIGN
vy[319] = cos(theta[319]) * r_init[319] * omega_c[319]
*/
void SpiralGalaxy_eqFunction_5101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[818]] /* vy[319] STATE(1) */) = (cos((data->simulationInfo->realParameter[1825] /* theta[319] PARAM */))) * (((data->simulationInfo->realParameter[1324] /* r_init[319] PARAM */)) * ((data->simulationInfo->realParameter[823] /* omega_c[319] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11182(DATA *data, threadData_t *threadData);


/*
equation index: 5103
type: SIMPLE_ASSIGN
vz[319] = 0.0
*/
void SpiralGalaxy_eqFunction_5103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5103};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1318]] /* vz[319] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11181(DATA *data, threadData_t *threadData);


/*
equation index: 5105
type: SIMPLE_ASSIGN
z[320] = 0.011200000000000002
*/
void SpiralGalaxy_eqFunction_5105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5105};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2819]] /* z[320] STATE(1,vz[320]) */) = 0.011200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11194(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11195(DATA *data, threadData_t *threadData);


/*
equation index: 5108
type: SIMPLE_ASSIGN
y[320] = r_init[320] * sin(theta[320] + 0.0028000000000000004)
*/
void SpiralGalaxy_eqFunction_5108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5108};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2319]] /* y[320] STATE(1,vy[320]) */) = ((data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */)) * (sin((data->simulationInfo->realParameter[1826] /* theta[320] PARAM */) + 0.0028000000000000004));
  TRACE_POP
}

/*
equation index: 5109
type: SIMPLE_ASSIGN
x[320] = r_init[320] * cos(theta[320] + 0.0028000000000000004)
*/
void SpiralGalaxy_eqFunction_5109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1819]] /* x[320] STATE(1,vx[320]) */) = ((data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */)) * (cos((data->simulationInfo->realParameter[1826] /* theta[320] PARAM */) + 0.0028000000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11196(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11197(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11200(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11199(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11198(DATA *data, threadData_t *threadData);


/*
equation index: 5115
type: SIMPLE_ASSIGN
vx[320] = (-sin(theta[320])) * r_init[320] * omega_c[320]
*/
void SpiralGalaxy_eqFunction_5115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[319]] /* vx[320] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1826] /* theta[320] PARAM */)))) * (((data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */)) * ((data->simulationInfo->realParameter[824] /* omega_c[320] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11193(DATA *data, threadData_t *threadData);


/*
equation index: 5117
type: SIMPLE_ASSIGN
vy[320] = cos(theta[320]) * r_init[320] * omega_c[320]
*/
void SpiralGalaxy_eqFunction_5117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[819]] /* vy[320] STATE(1) */) = (cos((data->simulationInfo->realParameter[1826] /* theta[320] PARAM */))) * (((data->simulationInfo->realParameter[1325] /* r_init[320] PARAM */)) * ((data->simulationInfo->realParameter[824] /* omega_c[320] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11192(DATA *data, threadData_t *threadData);


/*
equation index: 5119
type: SIMPLE_ASSIGN
vz[320] = 0.0
*/
void SpiralGalaxy_eqFunction_5119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5119};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1319]] /* vz[320] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11191(DATA *data, threadData_t *threadData);


/*
equation index: 5121
type: SIMPLE_ASSIGN
z[321] = 0.011360000000000002
*/
void SpiralGalaxy_eqFunction_5121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5121};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2820]] /* z[321] STATE(1,vz[321]) */) = 0.011360000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11204(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11205(DATA *data, threadData_t *threadData);


/*
equation index: 5124
type: SIMPLE_ASSIGN
y[321] = r_init[321] * sin(theta[321] + 0.0028400000000000005)
*/
void SpiralGalaxy_eqFunction_5124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5124};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2320]] /* y[321] STATE(1,vy[321]) */) = ((data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */)) * (sin((data->simulationInfo->realParameter[1827] /* theta[321] PARAM */) + 0.0028400000000000005));
  TRACE_POP
}

/*
equation index: 5125
type: SIMPLE_ASSIGN
x[321] = r_init[321] * cos(theta[321] + 0.0028400000000000005)
*/
void SpiralGalaxy_eqFunction_5125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1820]] /* x[321] STATE(1,vx[321]) */) = ((data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */)) * (cos((data->simulationInfo->realParameter[1827] /* theta[321] PARAM */) + 0.0028400000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11206(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11207(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11210(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11209(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11208(DATA *data, threadData_t *threadData);


/*
equation index: 5131
type: SIMPLE_ASSIGN
vx[321] = (-sin(theta[321])) * r_init[321] * omega_c[321]
*/
void SpiralGalaxy_eqFunction_5131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[320]] /* vx[321] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1827] /* theta[321] PARAM */)))) * (((data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */)) * ((data->simulationInfo->realParameter[825] /* omega_c[321] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11203(DATA *data, threadData_t *threadData);


/*
equation index: 5133
type: SIMPLE_ASSIGN
vy[321] = cos(theta[321]) * r_init[321] * omega_c[321]
*/
void SpiralGalaxy_eqFunction_5133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[820]] /* vy[321] STATE(1) */) = (cos((data->simulationInfo->realParameter[1827] /* theta[321] PARAM */))) * (((data->simulationInfo->realParameter[1326] /* r_init[321] PARAM */)) * ((data->simulationInfo->realParameter[825] /* omega_c[321] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11202(DATA *data, threadData_t *threadData);


/*
equation index: 5135
type: SIMPLE_ASSIGN
vz[321] = 0.0
*/
void SpiralGalaxy_eqFunction_5135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5135};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1320]] /* vz[321] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11201(DATA *data, threadData_t *threadData);


/*
equation index: 5137
type: SIMPLE_ASSIGN
z[322] = 0.011520000000000002
*/
void SpiralGalaxy_eqFunction_5137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5137};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2821]] /* z[322] STATE(1,vz[322]) */) = 0.011520000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11214(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11215(DATA *data, threadData_t *threadData);


/*
equation index: 5140
type: SIMPLE_ASSIGN
y[322] = r_init[322] * sin(theta[322] + 0.00288)
*/
void SpiralGalaxy_eqFunction_5140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5140};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2321]] /* y[322] STATE(1,vy[322]) */) = ((data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */)) * (sin((data->simulationInfo->realParameter[1828] /* theta[322] PARAM */) + 0.00288));
  TRACE_POP
}

/*
equation index: 5141
type: SIMPLE_ASSIGN
x[322] = r_init[322] * cos(theta[322] + 0.00288)
*/
void SpiralGalaxy_eqFunction_5141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1821]] /* x[322] STATE(1,vx[322]) */) = ((data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */)) * (cos((data->simulationInfo->realParameter[1828] /* theta[322] PARAM */) + 0.00288));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11216(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11217(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11220(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11219(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11218(DATA *data, threadData_t *threadData);


/*
equation index: 5147
type: SIMPLE_ASSIGN
vx[322] = (-sin(theta[322])) * r_init[322] * omega_c[322]
*/
void SpiralGalaxy_eqFunction_5147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[321]] /* vx[322] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1828] /* theta[322] PARAM */)))) * (((data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */)) * ((data->simulationInfo->realParameter[826] /* omega_c[322] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11213(DATA *data, threadData_t *threadData);


/*
equation index: 5149
type: SIMPLE_ASSIGN
vy[322] = cos(theta[322]) * r_init[322] * omega_c[322]
*/
void SpiralGalaxy_eqFunction_5149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[821]] /* vy[322] STATE(1) */) = (cos((data->simulationInfo->realParameter[1828] /* theta[322] PARAM */))) * (((data->simulationInfo->realParameter[1327] /* r_init[322] PARAM */)) * ((data->simulationInfo->realParameter[826] /* omega_c[322] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11212(DATA *data, threadData_t *threadData);


/*
equation index: 5151
type: SIMPLE_ASSIGN
vz[322] = 0.0
*/
void SpiralGalaxy_eqFunction_5151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5151};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1321]] /* vz[322] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11211(DATA *data, threadData_t *threadData);


/*
equation index: 5153
type: SIMPLE_ASSIGN
z[323] = 0.01168
*/
void SpiralGalaxy_eqFunction_5153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5153};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2822]] /* z[323] STATE(1,vz[323]) */) = 0.01168;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11224(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11225(DATA *data, threadData_t *threadData);


/*
equation index: 5156
type: SIMPLE_ASSIGN
y[323] = r_init[323] * sin(theta[323] + 0.0029200000000000003)
*/
void SpiralGalaxy_eqFunction_5156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5156};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2322]] /* y[323] STATE(1,vy[323]) */) = ((data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */)) * (sin((data->simulationInfo->realParameter[1829] /* theta[323] PARAM */) + 0.0029200000000000003));
  TRACE_POP
}

/*
equation index: 5157
type: SIMPLE_ASSIGN
x[323] = r_init[323] * cos(theta[323] + 0.0029200000000000003)
*/
void SpiralGalaxy_eqFunction_5157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1822]] /* x[323] STATE(1,vx[323]) */) = ((data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */)) * (cos((data->simulationInfo->realParameter[1829] /* theta[323] PARAM */) + 0.0029200000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11226(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11227(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11230(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11229(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11228(DATA *data, threadData_t *threadData);


/*
equation index: 5163
type: SIMPLE_ASSIGN
vx[323] = (-sin(theta[323])) * r_init[323] * omega_c[323]
*/
void SpiralGalaxy_eqFunction_5163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[322]] /* vx[323] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1829] /* theta[323] PARAM */)))) * (((data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */)) * ((data->simulationInfo->realParameter[827] /* omega_c[323] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11223(DATA *data, threadData_t *threadData);


/*
equation index: 5165
type: SIMPLE_ASSIGN
vy[323] = cos(theta[323]) * r_init[323] * omega_c[323]
*/
void SpiralGalaxy_eqFunction_5165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[822]] /* vy[323] STATE(1) */) = (cos((data->simulationInfo->realParameter[1829] /* theta[323] PARAM */))) * (((data->simulationInfo->realParameter[1328] /* r_init[323] PARAM */)) * ((data->simulationInfo->realParameter[827] /* omega_c[323] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11222(DATA *data, threadData_t *threadData);


/*
equation index: 5167
type: SIMPLE_ASSIGN
vz[323] = 0.0
*/
void SpiralGalaxy_eqFunction_5167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5167};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1322]] /* vz[323] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11221(DATA *data, threadData_t *threadData);


/*
equation index: 5169
type: SIMPLE_ASSIGN
z[324] = 0.011840000000000002
*/
void SpiralGalaxy_eqFunction_5169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5169};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2823]] /* z[324] STATE(1,vz[324]) */) = 0.011840000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11234(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11235(DATA *data, threadData_t *threadData);


/*
equation index: 5172
type: SIMPLE_ASSIGN
y[324] = r_init[324] * sin(theta[324] + 0.0029600000000000004)
*/
void SpiralGalaxy_eqFunction_5172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5172};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2323]] /* y[324] STATE(1,vy[324]) */) = ((data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */)) * (sin((data->simulationInfo->realParameter[1830] /* theta[324] PARAM */) + 0.0029600000000000004));
  TRACE_POP
}

/*
equation index: 5173
type: SIMPLE_ASSIGN
x[324] = r_init[324] * cos(theta[324] + 0.0029600000000000004)
*/
void SpiralGalaxy_eqFunction_5173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1823]] /* x[324] STATE(1,vx[324]) */) = ((data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */)) * (cos((data->simulationInfo->realParameter[1830] /* theta[324] PARAM */) + 0.0029600000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11236(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11237(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11240(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11239(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11238(DATA *data, threadData_t *threadData);


/*
equation index: 5179
type: SIMPLE_ASSIGN
vx[324] = (-sin(theta[324])) * r_init[324] * omega_c[324]
*/
void SpiralGalaxy_eqFunction_5179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[323]] /* vx[324] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1830] /* theta[324] PARAM */)))) * (((data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */)) * ((data->simulationInfo->realParameter[828] /* omega_c[324] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11233(DATA *data, threadData_t *threadData);


/*
equation index: 5181
type: SIMPLE_ASSIGN
vy[324] = cos(theta[324]) * r_init[324] * omega_c[324]
*/
void SpiralGalaxy_eqFunction_5181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[823]] /* vy[324] STATE(1) */) = (cos((data->simulationInfo->realParameter[1830] /* theta[324] PARAM */))) * (((data->simulationInfo->realParameter[1329] /* r_init[324] PARAM */)) * ((data->simulationInfo->realParameter[828] /* omega_c[324] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11232(DATA *data, threadData_t *threadData);


/*
equation index: 5183
type: SIMPLE_ASSIGN
vz[324] = 0.0
*/
void SpiralGalaxy_eqFunction_5183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5183};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1323]] /* vz[324] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11231(DATA *data, threadData_t *threadData);


/*
equation index: 5185
type: SIMPLE_ASSIGN
z[325] = 0.012000000000000002
*/
void SpiralGalaxy_eqFunction_5185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5185};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2824]] /* z[325] STATE(1,vz[325]) */) = 0.012000000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11244(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11245(DATA *data, threadData_t *threadData);


/*
equation index: 5188
type: SIMPLE_ASSIGN
y[325] = r_init[325] * sin(theta[325] + 0.0030000000000000005)
*/
void SpiralGalaxy_eqFunction_5188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5188};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2324]] /* y[325] STATE(1,vy[325]) */) = ((data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */)) * (sin((data->simulationInfo->realParameter[1831] /* theta[325] PARAM */) + 0.0030000000000000005));
  TRACE_POP
}

/*
equation index: 5189
type: SIMPLE_ASSIGN
x[325] = r_init[325] * cos(theta[325] + 0.0030000000000000005)
*/
void SpiralGalaxy_eqFunction_5189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1824]] /* x[325] STATE(1,vx[325]) */) = ((data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */)) * (cos((data->simulationInfo->realParameter[1831] /* theta[325] PARAM */) + 0.0030000000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11246(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11247(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11250(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11249(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11248(DATA *data, threadData_t *threadData);


/*
equation index: 5195
type: SIMPLE_ASSIGN
vx[325] = (-sin(theta[325])) * r_init[325] * omega_c[325]
*/
void SpiralGalaxy_eqFunction_5195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[324]] /* vx[325] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1831] /* theta[325] PARAM */)))) * (((data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */)) * ((data->simulationInfo->realParameter[829] /* omega_c[325] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11243(DATA *data, threadData_t *threadData);


/*
equation index: 5197
type: SIMPLE_ASSIGN
vy[325] = cos(theta[325]) * r_init[325] * omega_c[325]
*/
void SpiralGalaxy_eqFunction_5197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[824]] /* vy[325] STATE(1) */) = (cos((data->simulationInfo->realParameter[1831] /* theta[325] PARAM */))) * (((data->simulationInfo->realParameter[1330] /* r_init[325] PARAM */)) * ((data->simulationInfo->realParameter[829] /* omega_c[325] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11242(DATA *data, threadData_t *threadData);


/*
equation index: 5199
type: SIMPLE_ASSIGN
vz[325] = 0.0
*/
void SpiralGalaxy_eqFunction_5199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5199};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1324]] /* vz[325] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11241(DATA *data, threadData_t *threadData);


/*
equation index: 5201
type: SIMPLE_ASSIGN
z[326] = 0.01216
*/
void SpiralGalaxy_eqFunction_5201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5201};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2825]] /* z[326] STATE(1,vz[326]) */) = 0.01216;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11254(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11255(DATA *data, threadData_t *threadData);


/*
equation index: 5204
type: SIMPLE_ASSIGN
y[326] = r_init[326] * sin(theta[326] + 0.0030400000000000006)
*/
void SpiralGalaxy_eqFunction_5204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5204};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2325]] /* y[326] STATE(1,vy[326]) */) = ((data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */)) * (sin((data->simulationInfo->realParameter[1832] /* theta[326] PARAM */) + 0.0030400000000000006));
  TRACE_POP
}

/*
equation index: 5205
type: SIMPLE_ASSIGN
x[326] = r_init[326] * cos(theta[326] + 0.0030400000000000006)
*/
void SpiralGalaxy_eqFunction_5205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1825]] /* x[326] STATE(1,vx[326]) */) = ((data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */)) * (cos((data->simulationInfo->realParameter[1832] /* theta[326] PARAM */) + 0.0030400000000000006));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11256(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11257(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11260(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11259(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11258(DATA *data, threadData_t *threadData);


/*
equation index: 5211
type: SIMPLE_ASSIGN
vx[326] = (-sin(theta[326])) * r_init[326] * omega_c[326]
*/
void SpiralGalaxy_eqFunction_5211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[325]] /* vx[326] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1832] /* theta[326] PARAM */)))) * (((data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */)) * ((data->simulationInfo->realParameter[830] /* omega_c[326] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11253(DATA *data, threadData_t *threadData);


/*
equation index: 5213
type: SIMPLE_ASSIGN
vy[326] = cos(theta[326]) * r_init[326] * omega_c[326]
*/
void SpiralGalaxy_eqFunction_5213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[825]] /* vy[326] STATE(1) */) = (cos((data->simulationInfo->realParameter[1832] /* theta[326] PARAM */))) * (((data->simulationInfo->realParameter[1331] /* r_init[326] PARAM */)) * ((data->simulationInfo->realParameter[830] /* omega_c[326] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11252(DATA *data, threadData_t *threadData);


/*
equation index: 5215
type: SIMPLE_ASSIGN
vz[326] = 0.0
*/
void SpiralGalaxy_eqFunction_5215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5215};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1325]] /* vz[326] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11251(DATA *data, threadData_t *threadData);


/*
equation index: 5217
type: SIMPLE_ASSIGN
z[327] = 0.012320000000000001
*/
void SpiralGalaxy_eqFunction_5217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5217};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2826]] /* z[327] STATE(1,vz[327]) */) = 0.012320000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11264(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11265(DATA *data, threadData_t *threadData);


/*
equation index: 5220
type: SIMPLE_ASSIGN
y[327] = r_init[327] * sin(theta[327] + 0.0030800000000000007)
*/
void SpiralGalaxy_eqFunction_5220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5220};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2326]] /* y[327] STATE(1,vy[327]) */) = ((data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */)) * (sin((data->simulationInfo->realParameter[1833] /* theta[327] PARAM */) + 0.0030800000000000007));
  TRACE_POP
}

/*
equation index: 5221
type: SIMPLE_ASSIGN
x[327] = r_init[327] * cos(theta[327] + 0.0030800000000000007)
*/
void SpiralGalaxy_eqFunction_5221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1826]] /* x[327] STATE(1,vx[327]) */) = ((data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */)) * (cos((data->simulationInfo->realParameter[1833] /* theta[327] PARAM */) + 0.0030800000000000007));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11266(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11267(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11270(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11269(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11268(DATA *data, threadData_t *threadData);


/*
equation index: 5227
type: SIMPLE_ASSIGN
vx[327] = (-sin(theta[327])) * r_init[327] * omega_c[327]
*/
void SpiralGalaxy_eqFunction_5227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[326]] /* vx[327] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1833] /* theta[327] PARAM */)))) * (((data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */)) * ((data->simulationInfo->realParameter[831] /* omega_c[327] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11263(DATA *data, threadData_t *threadData);


/*
equation index: 5229
type: SIMPLE_ASSIGN
vy[327] = cos(theta[327]) * r_init[327] * omega_c[327]
*/
void SpiralGalaxy_eqFunction_5229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[826]] /* vy[327] STATE(1) */) = (cos((data->simulationInfo->realParameter[1833] /* theta[327] PARAM */))) * (((data->simulationInfo->realParameter[1332] /* r_init[327] PARAM */)) * ((data->simulationInfo->realParameter[831] /* omega_c[327] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11262(DATA *data, threadData_t *threadData);


/*
equation index: 5231
type: SIMPLE_ASSIGN
vz[327] = 0.0
*/
void SpiralGalaxy_eqFunction_5231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5231};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1326]] /* vz[327] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11261(DATA *data, threadData_t *threadData);


/*
equation index: 5233
type: SIMPLE_ASSIGN
z[328] = 0.012480000000000002
*/
void SpiralGalaxy_eqFunction_5233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5233};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2827]] /* z[328] STATE(1,vz[328]) */) = 0.012480000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11274(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11275(DATA *data, threadData_t *threadData);


/*
equation index: 5236
type: SIMPLE_ASSIGN
y[328] = r_init[328] * sin(theta[328] + 0.003120000000000001)
*/
void SpiralGalaxy_eqFunction_5236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5236};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2327]] /* y[328] STATE(1,vy[328]) */) = ((data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */)) * (sin((data->simulationInfo->realParameter[1834] /* theta[328] PARAM */) + 0.003120000000000001));
  TRACE_POP
}

/*
equation index: 5237
type: SIMPLE_ASSIGN
x[328] = r_init[328] * cos(theta[328] + 0.003120000000000001)
*/
void SpiralGalaxy_eqFunction_5237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1827]] /* x[328] STATE(1,vx[328]) */) = ((data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */)) * (cos((data->simulationInfo->realParameter[1834] /* theta[328] PARAM */) + 0.003120000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11276(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11277(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11280(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11279(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11278(DATA *data, threadData_t *threadData);


/*
equation index: 5243
type: SIMPLE_ASSIGN
vx[328] = (-sin(theta[328])) * r_init[328] * omega_c[328]
*/
void SpiralGalaxy_eqFunction_5243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[327]] /* vx[328] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1834] /* theta[328] PARAM */)))) * (((data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */)) * ((data->simulationInfo->realParameter[832] /* omega_c[328] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11273(DATA *data, threadData_t *threadData);


/*
equation index: 5245
type: SIMPLE_ASSIGN
vy[328] = cos(theta[328]) * r_init[328] * omega_c[328]
*/
void SpiralGalaxy_eqFunction_5245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[827]] /* vy[328] STATE(1) */) = (cos((data->simulationInfo->realParameter[1834] /* theta[328] PARAM */))) * (((data->simulationInfo->realParameter[1333] /* r_init[328] PARAM */)) * ((data->simulationInfo->realParameter[832] /* omega_c[328] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11272(DATA *data, threadData_t *threadData);


/*
equation index: 5247
type: SIMPLE_ASSIGN
vz[328] = 0.0
*/
void SpiralGalaxy_eqFunction_5247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5247};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1327]] /* vz[328] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11271(DATA *data, threadData_t *threadData);


/*
equation index: 5249
type: SIMPLE_ASSIGN
z[329] = 0.01264
*/
void SpiralGalaxy_eqFunction_5249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5249};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2828]] /* z[329] STATE(1,vz[329]) */) = 0.01264;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11284(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11285(DATA *data, threadData_t *threadData);


/*
equation index: 5252
type: SIMPLE_ASSIGN
y[329] = r_init[329] * sin(theta[329] + 0.0031600000000000005)
*/
void SpiralGalaxy_eqFunction_5252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5252};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2328]] /* y[329] STATE(1,vy[329]) */) = ((data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */)) * (sin((data->simulationInfo->realParameter[1835] /* theta[329] PARAM */) + 0.0031600000000000005));
  TRACE_POP
}

/*
equation index: 5253
type: SIMPLE_ASSIGN
x[329] = r_init[329] * cos(theta[329] + 0.0031600000000000005)
*/
void SpiralGalaxy_eqFunction_5253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1828]] /* x[329] STATE(1,vx[329]) */) = ((data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */)) * (cos((data->simulationInfo->realParameter[1835] /* theta[329] PARAM */) + 0.0031600000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11286(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11287(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11290(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11289(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11288(DATA *data, threadData_t *threadData);


/*
equation index: 5259
type: SIMPLE_ASSIGN
vx[329] = (-sin(theta[329])) * r_init[329] * omega_c[329]
*/
void SpiralGalaxy_eqFunction_5259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[328]] /* vx[329] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1835] /* theta[329] PARAM */)))) * (((data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */)) * ((data->simulationInfo->realParameter[833] /* omega_c[329] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11283(DATA *data, threadData_t *threadData);


/*
equation index: 5261
type: SIMPLE_ASSIGN
vy[329] = cos(theta[329]) * r_init[329] * omega_c[329]
*/
void SpiralGalaxy_eqFunction_5261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[828]] /* vy[329] STATE(1) */) = (cos((data->simulationInfo->realParameter[1835] /* theta[329] PARAM */))) * (((data->simulationInfo->realParameter[1334] /* r_init[329] PARAM */)) * ((data->simulationInfo->realParameter[833] /* omega_c[329] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11282(DATA *data, threadData_t *threadData);


/*
equation index: 5263
type: SIMPLE_ASSIGN
vz[329] = 0.0
*/
void SpiralGalaxy_eqFunction_5263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5263};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1328]] /* vz[329] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11281(DATA *data, threadData_t *threadData);


/*
equation index: 5265
type: SIMPLE_ASSIGN
z[330] = 0.012800000000000002
*/
void SpiralGalaxy_eqFunction_5265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5265};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2829]] /* z[330] STATE(1,vz[330]) */) = 0.012800000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11294(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11295(DATA *data, threadData_t *threadData);


/*
equation index: 5268
type: SIMPLE_ASSIGN
y[330] = r_init[330] * sin(theta[330] + 0.0032000000000000006)
*/
void SpiralGalaxy_eqFunction_5268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5268};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2329]] /* y[330] STATE(1,vy[330]) */) = ((data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */)) * (sin((data->simulationInfo->realParameter[1836] /* theta[330] PARAM */) + 0.0032000000000000006));
  TRACE_POP
}

/*
equation index: 5269
type: SIMPLE_ASSIGN
x[330] = r_init[330] * cos(theta[330] + 0.0032000000000000006)
*/
void SpiralGalaxy_eqFunction_5269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1829]] /* x[330] STATE(1,vx[330]) */) = ((data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */)) * (cos((data->simulationInfo->realParameter[1836] /* theta[330] PARAM */) + 0.0032000000000000006));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11296(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11297(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11300(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11299(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11298(DATA *data, threadData_t *threadData);


/*
equation index: 5275
type: SIMPLE_ASSIGN
vx[330] = (-sin(theta[330])) * r_init[330] * omega_c[330]
*/
void SpiralGalaxy_eqFunction_5275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[329]] /* vx[330] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1836] /* theta[330] PARAM */)))) * (((data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */)) * ((data->simulationInfo->realParameter[834] /* omega_c[330] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11293(DATA *data, threadData_t *threadData);


/*
equation index: 5277
type: SIMPLE_ASSIGN
vy[330] = cos(theta[330]) * r_init[330] * omega_c[330]
*/
void SpiralGalaxy_eqFunction_5277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[829]] /* vy[330] STATE(1) */) = (cos((data->simulationInfo->realParameter[1836] /* theta[330] PARAM */))) * (((data->simulationInfo->realParameter[1335] /* r_init[330] PARAM */)) * ((data->simulationInfo->realParameter[834] /* omega_c[330] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11292(DATA *data, threadData_t *threadData);


/*
equation index: 5279
type: SIMPLE_ASSIGN
vz[330] = 0.0
*/
void SpiralGalaxy_eqFunction_5279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5279};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1329]] /* vz[330] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11291(DATA *data, threadData_t *threadData);


/*
equation index: 5281
type: SIMPLE_ASSIGN
z[331] = 0.012960000000000001
*/
void SpiralGalaxy_eqFunction_5281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5281};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2830]] /* z[331] STATE(1,vz[331]) */) = 0.012960000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11304(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11305(DATA *data, threadData_t *threadData);


/*
equation index: 5284
type: SIMPLE_ASSIGN
y[331] = r_init[331] * sin(theta[331] + 0.0032400000000000007)
*/
void SpiralGalaxy_eqFunction_5284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5284};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2330]] /* y[331] STATE(1,vy[331]) */) = ((data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */)) * (sin((data->simulationInfo->realParameter[1837] /* theta[331] PARAM */) + 0.0032400000000000007));
  TRACE_POP
}

/*
equation index: 5285
type: SIMPLE_ASSIGN
x[331] = r_init[331] * cos(theta[331] + 0.0032400000000000007)
*/
void SpiralGalaxy_eqFunction_5285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1830]] /* x[331] STATE(1,vx[331]) */) = ((data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */)) * (cos((data->simulationInfo->realParameter[1837] /* theta[331] PARAM */) + 0.0032400000000000007));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11306(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11307(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11310(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11309(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11308(DATA *data, threadData_t *threadData);


/*
equation index: 5291
type: SIMPLE_ASSIGN
vx[331] = (-sin(theta[331])) * r_init[331] * omega_c[331]
*/
void SpiralGalaxy_eqFunction_5291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[330]] /* vx[331] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1837] /* theta[331] PARAM */)))) * (((data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */)) * ((data->simulationInfo->realParameter[835] /* omega_c[331] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11303(DATA *data, threadData_t *threadData);


/*
equation index: 5293
type: SIMPLE_ASSIGN
vy[331] = cos(theta[331]) * r_init[331] * omega_c[331]
*/
void SpiralGalaxy_eqFunction_5293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[830]] /* vy[331] STATE(1) */) = (cos((data->simulationInfo->realParameter[1837] /* theta[331] PARAM */))) * (((data->simulationInfo->realParameter[1336] /* r_init[331] PARAM */)) * ((data->simulationInfo->realParameter[835] /* omega_c[331] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11302(DATA *data, threadData_t *threadData);


/*
equation index: 5295
type: SIMPLE_ASSIGN
vz[331] = 0.0
*/
void SpiralGalaxy_eqFunction_5295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1330]] /* vz[331] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11301(DATA *data, threadData_t *threadData);


/*
equation index: 5297
type: SIMPLE_ASSIGN
z[332] = 0.013120000000000001
*/
void SpiralGalaxy_eqFunction_5297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2831]] /* z[332] STATE(1,vz[332]) */) = 0.013120000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11314(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11315(DATA *data, threadData_t *threadData);


/*
equation index: 5300
type: SIMPLE_ASSIGN
y[332] = r_init[332] * sin(theta[332] + 0.003280000000000001)
*/
void SpiralGalaxy_eqFunction_5300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5300};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2331]] /* y[332] STATE(1,vy[332]) */) = ((data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */)) * (sin((data->simulationInfo->realParameter[1838] /* theta[332] PARAM */) + 0.003280000000000001));
  TRACE_POP
}

/*
equation index: 5301
type: SIMPLE_ASSIGN
x[332] = r_init[332] * cos(theta[332] + 0.003280000000000001)
*/
void SpiralGalaxy_eqFunction_5301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1831]] /* x[332] STATE(1,vx[332]) */) = ((data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */)) * (cos((data->simulationInfo->realParameter[1838] /* theta[332] PARAM */) + 0.003280000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11316(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11317(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11320(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11319(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11318(DATA *data, threadData_t *threadData);


/*
equation index: 5307
type: SIMPLE_ASSIGN
vx[332] = (-sin(theta[332])) * r_init[332] * omega_c[332]
*/
void SpiralGalaxy_eqFunction_5307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[331]] /* vx[332] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1838] /* theta[332] PARAM */)))) * (((data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */)) * ((data->simulationInfo->realParameter[836] /* omega_c[332] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11313(DATA *data, threadData_t *threadData);


/*
equation index: 5309
type: SIMPLE_ASSIGN
vy[332] = cos(theta[332]) * r_init[332] * omega_c[332]
*/
void SpiralGalaxy_eqFunction_5309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[831]] /* vy[332] STATE(1) */) = (cos((data->simulationInfo->realParameter[1838] /* theta[332] PARAM */))) * (((data->simulationInfo->realParameter[1337] /* r_init[332] PARAM */)) * ((data->simulationInfo->realParameter[836] /* omega_c[332] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11312(DATA *data, threadData_t *threadData);


/*
equation index: 5311
type: SIMPLE_ASSIGN
vz[332] = 0.0
*/
void SpiralGalaxy_eqFunction_5311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1331]] /* vz[332] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11311(DATA *data, threadData_t *threadData);


/*
equation index: 5313
type: SIMPLE_ASSIGN
z[333] = 0.013280000000000002
*/
void SpiralGalaxy_eqFunction_5313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2832]] /* z[333] STATE(1,vz[333]) */) = 0.013280000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11324(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11325(DATA *data, threadData_t *threadData);


/*
equation index: 5316
type: SIMPLE_ASSIGN
y[333] = r_init[333] * sin(theta[333] + 0.003320000000000001)
*/
void SpiralGalaxy_eqFunction_5316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5316};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2332]] /* y[333] STATE(1,vy[333]) */) = ((data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */)) * (sin((data->simulationInfo->realParameter[1839] /* theta[333] PARAM */) + 0.003320000000000001));
  TRACE_POP
}

/*
equation index: 5317
type: SIMPLE_ASSIGN
x[333] = r_init[333] * cos(theta[333] + 0.003320000000000001)
*/
void SpiralGalaxy_eqFunction_5317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1832]] /* x[333] STATE(1,vx[333]) */) = ((data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */)) * (cos((data->simulationInfo->realParameter[1839] /* theta[333] PARAM */) + 0.003320000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11326(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11327(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11330(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11329(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11328(DATA *data, threadData_t *threadData);


/*
equation index: 5323
type: SIMPLE_ASSIGN
vx[333] = (-sin(theta[333])) * r_init[333] * omega_c[333]
*/
void SpiralGalaxy_eqFunction_5323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[332]] /* vx[333] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1839] /* theta[333] PARAM */)))) * (((data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */)) * ((data->simulationInfo->realParameter[837] /* omega_c[333] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11323(DATA *data, threadData_t *threadData);


/*
equation index: 5325
type: SIMPLE_ASSIGN
vy[333] = cos(theta[333]) * r_init[333] * omega_c[333]
*/
void SpiralGalaxy_eqFunction_5325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[832]] /* vy[333] STATE(1) */) = (cos((data->simulationInfo->realParameter[1839] /* theta[333] PARAM */))) * (((data->simulationInfo->realParameter[1338] /* r_init[333] PARAM */)) * ((data->simulationInfo->realParameter[837] /* omega_c[333] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11322(DATA *data, threadData_t *threadData);


/*
equation index: 5327
type: SIMPLE_ASSIGN
vz[333] = 0.0
*/
void SpiralGalaxy_eqFunction_5327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5327};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1332]] /* vz[333] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11321(DATA *data, threadData_t *threadData);


/*
equation index: 5329
type: SIMPLE_ASSIGN
z[334] = 0.013440000000000002
*/
void SpiralGalaxy_eqFunction_5329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5329};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2833]] /* z[334] STATE(1,vz[334]) */) = 0.013440000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11334(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11335(DATA *data, threadData_t *threadData);


/*
equation index: 5332
type: SIMPLE_ASSIGN
y[334] = r_init[334] * sin(theta[334] + 0.003360000000000001)
*/
void SpiralGalaxy_eqFunction_5332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5332};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2333]] /* y[334] STATE(1,vy[334]) */) = ((data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */)) * (sin((data->simulationInfo->realParameter[1840] /* theta[334] PARAM */) + 0.003360000000000001));
  TRACE_POP
}

/*
equation index: 5333
type: SIMPLE_ASSIGN
x[334] = r_init[334] * cos(theta[334] + 0.003360000000000001)
*/
void SpiralGalaxy_eqFunction_5333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1833]] /* x[334] STATE(1,vx[334]) */) = ((data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */)) * (cos((data->simulationInfo->realParameter[1840] /* theta[334] PARAM */) + 0.003360000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11336(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11337(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11340(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11339(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11338(DATA *data, threadData_t *threadData);


/*
equation index: 5339
type: SIMPLE_ASSIGN
vx[334] = (-sin(theta[334])) * r_init[334] * omega_c[334]
*/
void SpiralGalaxy_eqFunction_5339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[333]] /* vx[334] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1840] /* theta[334] PARAM */)))) * (((data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */)) * ((data->simulationInfo->realParameter[838] /* omega_c[334] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11333(DATA *data, threadData_t *threadData);


/*
equation index: 5341
type: SIMPLE_ASSIGN
vy[334] = cos(theta[334]) * r_init[334] * omega_c[334]
*/
void SpiralGalaxy_eqFunction_5341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[833]] /* vy[334] STATE(1) */) = (cos((data->simulationInfo->realParameter[1840] /* theta[334] PARAM */))) * (((data->simulationInfo->realParameter[1339] /* r_init[334] PARAM */)) * ((data->simulationInfo->realParameter[838] /* omega_c[334] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11332(DATA *data, threadData_t *threadData);


/*
equation index: 5343
type: SIMPLE_ASSIGN
vz[334] = 0.0
*/
void SpiralGalaxy_eqFunction_5343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1333]] /* vz[334] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11331(DATA *data, threadData_t *threadData);


/*
equation index: 5345
type: SIMPLE_ASSIGN
z[335] = 0.013600000000000001
*/
void SpiralGalaxy_eqFunction_5345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2834]] /* z[335] STATE(1,vz[335]) */) = 0.013600000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11344(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11345(DATA *data, threadData_t *threadData);


/*
equation index: 5348
type: SIMPLE_ASSIGN
y[335] = r_init[335] * sin(theta[335] + 0.0034000000000000007)
*/
void SpiralGalaxy_eqFunction_5348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5348};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2334]] /* y[335] STATE(1,vy[335]) */) = ((data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */)) * (sin((data->simulationInfo->realParameter[1841] /* theta[335] PARAM */) + 0.0034000000000000007));
  TRACE_POP
}

/*
equation index: 5349
type: SIMPLE_ASSIGN
x[335] = r_init[335] * cos(theta[335] + 0.0034000000000000007)
*/
void SpiralGalaxy_eqFunction_5349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1834]] /* x[335] STATE(1,vx[335]) */) = ((data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */)) * (cos((data->simulationInfo->realParameter[1841] /* theta[335] PARAM */) + 0.0034000000000000007));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11346(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11347(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11350(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11349(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11348(DATA *data, threadData_t *threadData);


/*
equation index: 5355
type: SIMPLE_ASSIGN
vx[335] = (-sin(theta[335])) * r_init[335] * omega_c[335]
*/
void SpiralGalaxy_eqFunction_5355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[334]] /* vx[335] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1841] /* theta[335] PARAM */)))) * (((data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */)) * ((data->simulationInfo->realParameter[839] /* omega_c[335] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11343(DATA *data, threadData_t *threadData);


/*
equation index: 5357
type: SIMPLE_ASSIGN
vy[335] = cos(theta[335]) * r_init[335] * omega_c[335]
*/
void SpiralGalaxy_eqFunction_5357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[834]] /* vy[335] STATE(1) */) = (cos((data->simulationInfo->realParameter[1841] /* theta[335] PARAM */))) * (((data->simulationInfo->realParameter[1340] /* r_init[335] PARAM */)) * ((data->simulationInfo->realParameter[839] /* omega_c[335] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11342(DATA *data, threadData_t *threadData);


/*
equation index: 5359
type: SIMPLE_ASSIGN
vz[335] = 0.0
*/
void SpiralGalaxy_eqFunction_5359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5359};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1334]] /* vz[335] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11341(DATA *data, threadData_t *threadData);


/*
equation index: 5361
type: SIMPLE_ASSIGN
z[336] = 0.013760000000000001
*/
void SpiralGalaxy_eqFunction_5361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5361};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2835]] /* z[336] STATE(1,vz[336]) */) = 0.013760000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11354(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11355(DATA *data, threadData_t *threadData);


/*
equation index: 5364
type: SIMPLE_ASSIGN
y[336] = r_init[336] * sin(theta[336] + 0.0034400000000000008)
*/
void SpiralGalaxy_eqFunction_5364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5364};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2335]] /* y[336] STATE(1,vy[336]) */) = ((data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */)) * (sin((data->simulationInfo->realParameter[1842] /* theta[336] PARAM */) + 0.0034400000000000008));
  TRACE_POP
}

/*
equation index: 5365
type: SIMPLE_ASSIGN
x[336] = r_init[336] * cos(theta[336] + 0.0034400000000000008)
*/
void SpiralGalaxy_eqFunction_5365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1835]] /* x[336] STATE(1,vx[336]) */) = ((data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */)) * (cos((data->simulationInfo->realParameter[1842] /* theta[336] PARAM */) + 0.0034400000000000008));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11356(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11357(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11360(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11359(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11358(DATA *data, threadData_t *threadData);


/*
equation index: 5371
type: SIMPLE_ASSIGN
vx[336] = (-sin(theta[336])) * r_init[336] * omega_c[336]
*/
void SpiralGalaxy_eqFunction_5371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5371};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[335]] /* vx[336] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1842] /* theta[336] PARAM */)))) * (((data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */)) * ((data->simulationInfo->realParameter[840] /* omega_c[336] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11353(DATA *data, threadData_t *threadData);


/*
equation index: 5373
type: SIMPLE_ASSIGN
vy[336] = cos(theta[336]) * r_init[336] * omega_c[336]
*/
void SpiralGalaxy_eqFunction_5373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[835]] /* vy[336] STATE(1) */) = (cos((data->simulationInfo->realParameter[1842] /* theta[336] PARAM */))) * (((data->simulationInfo->realParameter[1341] /* r_init[336] PARAM */)) * ((data->simulationInfo->realParameter[840] /* omega_c[336] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11352(DATA *data, threadData_t *threadData);


/*
equation index: 5375
type: SIMPLE_ASSIGN
vz[336] = 0.0
*/
void SpiralGalaxy_eqFunction_5375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5375};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1335]] /* vz[336] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11351(DATA *data, threadData_t *threadData);


/*
equation index: 5377
type: SIMPLE_ASSIGN
z[337] = 0.013920000000000002
*/
void SpiralGalaxy_eqFunction_5377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5377};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2836]] /* z[337] STATE(1,vz[337]) */) = 0.013920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11364(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11365(DATA *data, threadData_t *threadData);


/*
equation index: 5380
type: SIMPLE_ASSIGN
y[337] = r_init[337] * sin(theta[337] + 0.003480000000000001)
*/
void SpiralGalaxy_eqFunction_5380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5380};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2336]] /* y[337] STATE(1,vy[337]) */) = ((data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */)) * (sin((data->simulationInfo->realParameter[1843] /* theta[337] PARAM */) + 0.003480000000000001));
  TRACE_POP
}

/*
equation index: 5381
type: SIMPLE_ASSIGN
x[337] = r_init[337] * cos(theta[337] + 0.003480000000000001)
*/
void SpiralGalaxy_eqFunction_5381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1836]] /* x[337] STATE(1,vx[337]) */) = ((data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */)) * (cos((data->simulationInfo->realParameter[1843] /* theta[337] PARAM */) + 0.003480000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11366(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11367(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11370(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11369(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11368(DATA *data, threadData_t *threadData);


/*
equation index: 5387
type: SIMPLE_ASSIGN
vx[337] = (-sin(theta[337])) * r_init[337] * omega_c[337]
*/
void SpiralGalaxy_eqFunction_5387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[336]] /* vx[337] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1843] /* theta[337] PARAM */)))) * (((data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */)) * ((data->simulationInfo->realParameter[841] /* omega_c[337] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11363(DATA *data, threadData_t *threadData);


/*
equation index: 5389
type: SIMPLE_ASSIGN
vy[337] = cos(theta[337]) * r_init[337] * omega_c[337]
*/
void SpiralGalaxy_eqFunction_5389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[836]] /* vy[337] STATE(1) */) = (cos((data->simulationInfo->realParameter[1843] /* theta[337] PARAM */))) * (((data->simulationInfo->realParameter[1342] /* r_init[337] PARAM */)) * ((data->simulationInfo->realParameter[841] /* omega_c[337] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11362(DATA *data, threadData_t *threadData);


/*
equation index: 5391
type: SIMPLE_ASSIGN
vz[337] = 0.0
*/
void SpiralGalaxy_eqFunction_5391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5391};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1336]] /* vz[337] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11361(DATA *data, threadData_t *threadData);


/*
equation index: 5393
type: SIMPLE_ASSIGN
z[338] = 0.014080000000000002
*/
void SpiralGalaxy_eqFunction_5393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5393};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2837]] /* z[338] STATE(1,vz[338]) */) = 0.014080000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11374(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11375(DATA *data, threadData_t *threadData);


/*
equation index: 5396
type: SIMPLE_ASSIGN
y[338] = r_init[338] * sin(theta[338] + 0.003520000000000001)
*/
void SpiralGalaxy_eqFunction_5396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5396};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2337]] /* y[338] STATE(1,vy[338]) */) = ((data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */)) * (sin((data->simulationInfo->realParameter[1844] /* theta[338] PARAM */) + 0.003520000000000001));
  TRACE_POP
}

/*
equation index: 5397
type: SIMPLE_ASSIGN
x[338] = r_init[338] * cos(theta[338] + 0.003520000000000001)
*/
void SpiralGalaxy_eqFunction_5397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1837]] /* x[338] STATE(1,vx[338]) */) = ((data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */)) * (cos((data->simulationInfo->realParameter[1844] /* theta[338] PARAM */) + 0.003520000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11376(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11377(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11380(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11379(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11378(DATA *data, threadData_t *threadData);


/*
equation index: 5403
type: SIMPLE_ASSIGN
vx[338] = (-sin(theta[338])) * r_init[338] * omega_c[338]
*/
void SpiralGalaxy_eqFunction_5403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[337]] /* vx[338] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1844] /* theta[338] PARAM */)))) * (((data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */)) * ((data->simulationInfo->realParameter[842] /* omega_c[338] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11373(DATA *data, threadData_t *threadData);


/*
equation index: 5405
type: SIMPLE_ASSIGN
vy[338] = cos(theta[338]) * r_init[338] * omega_c[338]
*/
void SpiralGalaxy_eqFunction_5405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[837]] /* vy[338] STATE(1) */) = (cos((data->simulationInfo->realParameter[1844] /* theta[338] PARAM */))) * (((data->simulationInfo->realParameter[1343] /* r_init[338] PARAM */)) * ((data->simulationInfo->realParameter[842] /* omega_c[338] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11372(DATA *data, threadData_t *threadData);


/*
equation index: 5407
type: SIMPLE_ASSIGN
vz[338] = 0.0
*/
void SpiralGalaxy_eqFunction_5407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5407};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1337]] /* vz[338] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11371(DATA *data, threadData_t *threadData);


/*
equation index: 5409
type: SIMPLE_ASSIGN
z[339] = 0.014240000000000001
*/
void SpiralGalaxy_eqFunction_5409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5409};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2838]] /* z[339] STATE(1,vz[339]) */) = 0.014240000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11384(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11385(DATA *data, threadData_t *threadData);


/*
equation index: 5412
type: SIMPLE_ASSIGN
y[339] = r_init[339] * sin(theta[339] + 0.003560000000000001)
*/
void SpiralGalaxy_eqFunction_5412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5412};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2338]] /* y[339] STATE(1,vy[339]) */) = ((data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */)) * (sin((data->simulationInfo->realParameter[1845] /* theta[339] PARAM */) + 0.003560000000000001));
  TRACE_POP
}

/*
equation index: 5413
type: SIMPLE_ASSIGN
x[339] = r_init[339] * cos(theta[339] + 0.003560000000000001)
*/
void SpiralGalaxy_eqFunction_5413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1838]] /* x[339] STATE(1,vx[339]) */) = ((data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */)) * (cos((data->simulationInfo->realParameter[1845] /* theta[339] PARAM */) + 0.003560000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11386(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11387(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11390(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11389(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11388(DATA *data, threadData_t *threadData);


/*
equation index: 5419
type: SIMPLE_ASSIGN
vx[339] = (-sin(theta[339])) * r_init[339] * omega_c[339]
*/
void SpiralGalaxy_eqFunction_5419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[338]] /* vx[339] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1845] /* theta[339] PARAM */)))) * (((data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */)) * ((data->simulationInfo->realParameter[843] /* omega_c[339] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11383(DATA *data, threadData_t *threadData);


/*
equation index: 5421
type: SIMPLE_ASSIGN
vy[339] = cos(theta[339]) * r_init[339] * omega_c[339]
*/
void SpiralGalaxy_eqFunction_5421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[838]] /* vy[339] STATE(1) */) = (cos((data->simulationInfo->realParameter[1845] /* theta[339] PARAM */))) * (((data->simulationInfo->realParameter[1344] /* r_init[339] PARAM */)) * ((data->simulationInfo->realParameter[843] /* omega_c[339] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11382(DATA *data, threadData_t *threadData);


/*
equation index: 5423
type: SIMPLE_ASSIGN
vz[339] = 0.0
*/
void SpiralGalaxy_eqFunction_5423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5423};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1338]] /* vz[339] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11381(DATA *data, threadData_t *threadData);


/*
equation index: 5425
type: SIMPLE_ASSIGN
z[340] = 0.014400000000000003
*/
void SpiralGalaxy_eqFunction_5425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5425};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2839]] /* z[340] STATE(1,vz[340]) */) = 0.014400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11394(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11395(DATA *data, threadData_t *threadData);


/*
equation index: 5428
type: SIMPLE_ASSIGN
y[340] = r_init[340] * sin(theta[340] + 0.003600000000000001)
*/
void SpiralGalaxy_eqFunction_5428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5428};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2339]] /* y[340] STATE(1,vy[340]) */) = ((data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */)) * (sin((data->simulationInfo->realParameter[1846] /* theta[340] PARAM */) + 0.003600000000000001));
  TRACE_POP
}

/*
equation index: 5429
type: SIMPLE_ASSIGN
x[340] = r_init[340] * cos(theta[340] + 0.003600000000000001)
*/
void SpiralGalaxy_eqFunction_5429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1839]] /* x[340] STATE(1,vx[340]) */) = ((data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */)) * (cos((data->simulationInfo->realParameter[1846] /* theta[340] PARAM */) + 0.003600000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11396(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11397(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11400(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11399(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11398(DATA *data, threadData_t *threadData);


/*
equation index: 5435
type: SIMPLE_ASSIGN
vx[340] = (-sin(theta[340])) * r_init[340] * omega_c[340]
*/
void SpiralGalaxy_eqFunction_5435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[339]] /* vx[340] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1846] /* theta[340] PARAM */)))) * (((data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */)) * ((data->simulationInfo->realParameter[844] /* omega_c[340] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11393(DATA *data, threadData_t *threadData);


/*
equation index: 5437
type: SIMPLE_ASSIGN
vy[340] = cos(theta[340]) * r_init[340] * omega_c[340]
*/
void SpiralGalaxy_eqFunction_5437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[839]] /* vy[340] STATE(1) */) = (cos((data->simulationInfo->realParameter[1846] /* theta[340] PARAM */))) * (((data->simulationInfo->realParameter[1345] /* r_init[340] PARAM */)) * ((data->simulationInfo->realParameter[844] /* omega_c[340] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11392(DATA *data, threadData_t *threadData);


/*
equation index: 5439
type: SIMPLE_ASSIGN
vz[340] = 0.0
*/
void SpiralGalaxy_eqFunction_5439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5439};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1339]] /* vz[340] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11391(DATA *data, threadData_t *threadData);


/*
equation index: 5441
type: SIMPLE_ASSIGN
z[341] = 0.014560000000000002
*/
void SpiralGalaxy_eqFunction_5441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5441};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2840]] /* z[341] STATE(1,vz[341]) */) = 0.014560000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11404(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11405(DATA *data, threadData_t *threadData);


/*
equation index: 5444
type: SIMPLE_ASSIGN
y[341] = r_init[341] * sin(theta[341] + 0.003640000000000001)
*/
void SpiralGalaxy_eqFunction_5444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5444};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2340]] /* y[341] STATE(1,vy[341]) */) = ((data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */)) * (sin((data->simulationInfo->realParameter[1847] /* theta[341] PARAM */) + 0.003640000000000001));
  TRACE_POP
}

/*
equation index: 5445
type: SIMPLE_ASSIGN
x[341] = r_init[341] * cos(theta[341] + 0.003640000000000001)
*/
void SpiralGalaxy_eqFunction_5445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1840]] /* x[341] STATE(1,vx[341]) */) = ((data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */)) * (cos((data->simulationInfo->realParameter[1847] /* theta[341] PARAM */) + 0.003640000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11406(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11407(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11410(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11409(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11408(DATA *data, threadData_t *threadData);


/*
equation index: 5451
type: SIMPLE_ASSIGN
vx[341] = (-sin(theta[341])) * r_init[341] * omega_c[341]
*/
void SpiralGalaxy_eqFunction_5451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[340]] /* vx[341] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1847] /* theta[341] PARAM */)))) * (((data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */)) * ((data->simulationInfo->realParameter[845] /* omega_c[341] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11403(DATA *data, threadData_t *threadData);


/*
equation index: 5453
type: SIMPLE_ASSIGN
vy[341] = cos(theta[341]) * r_init[341] * omega_c[341]
*/
void SpiralGalaxy_eqFunction_5453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[840]] /* vy[341] STATE(1) */) = (cos((data->simulationInfo->realParameter[1847] /* theta[341] PARAM */))) * (((data->simulationInfo->realParameter[1346] /* r_init[341] PARAM */)) * ((data->simulationInfo->realParameter[845] /* omega_c[341] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11402(DATA *data, threadData_t *threadData);


/*
equation index: 5455
type: SIMPLE_ASSIGN
vz[341] = 0.0
*/
void SpiralGalaxy_eqFunction_5455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5455};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1340]] /* vz[341] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11401(DATA *data, threadData_t *threadData);


/*
equation index: 5457
type: SIMPLE_ASSIGN
z[342] = 0.01472
*/
void SpiralGalaxy_eqFunction_5457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5457};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2841]] /* z[342] STATE(1,vz[342]) */) = 0.01472;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11414(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11415(DATA *data, threadData_t *threadData);


/*
equation index: 5460
type: SIMPLE_ASSIGN
y[342] = r_init[342] * sin(theta[342] + 0.003680000000000001)
*/
void SpiralGalaxy_eqFunction_5460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5460};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2341]] /* y[342] STATE(1,vy[342]) */) = ((data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */)) * (sin((data->simulationInfo->realParameter[1848] /* theta[342] PARAM */) + 0.003680000000000001));
  TRACE_POP
}

/*
equation index: 5461
type: SIMPLE_ASSIGN
x[342] = r_init[342] * cos(theta[342] + 0.003680000000000001)
*/
void SpiralGalaxy_eqFunction_5461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1841]] /* x[342] STATE(1,vx[342]) */) = ((data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */)) * (cos((data->simulationInfo->realParameter[1848] /* theta[342] PARAM */) + 0.003680000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11416(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11417(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11420(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11419(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11418(DATA *data, threadData_t *threadData);


/*
equation index: 5467
type: SIMPLE_ASSIGN
vx[342] = (-sin(theta[342])) * r_init[342] * omega_c[342]
*/
void SpiralGalaxy_eqFunction_5467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[341]] /* vx[342] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1848] /* theta[342] PARAM */)))) * (((data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */)) * ((data->simulationInfo->realParameter[846] /* omega_c[342] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11413(DATA *data, threadData_t *threadData);


/*
equation index: 5469
type: SIMPLE_ASSIGN
vy[342] = cos(theta[342]) * r_init[342] * omega_c[342]
*/
void SpiralGalaxy_eqFunction_5469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[841]] /* vy[342] STATE(1) */) = (cos((data->simulationInfo->realParameter[1848] /* theta[342] PARAM */))) * (((data->simulationInfo->realParameter[1347] /* r_init[342] PARAM */)) * ((data->simulationInfo->realParameter[846] /* omega_c[342] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11412(DATA *data, threadData_t *threadData);


/*
equation index: 5471
type: SIMPLE_ASSIGN
vz[342] = 0.0
*/
void SpiralGalaxy_eqFunction_5471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5471};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1341]] /* vz[342] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11411(DATA *data, threadData_t *threadData);


/*
equation index: 5473
type: SIMPLE_ASSIGN
z[343] = 0.014880000000000003
*/
void SpiralGalaxy_eqFunction_5473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5473};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2842]] /* z[343] STATE(1,vz[343]) */) = 0.014880000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11424(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11425(DATA *data, threadData_t *threadData);


/*
equation index: 5476
type: SIMPLE_ASSIGN
y[343] = r_init[343] * sin(theta[343] + 0.003720000000000001)
*/
void SpiralGalaxy_eqFunction_5476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5476};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2342]] /* y[343] STATE(1,vy[343]) */) = ((data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */)) * (sin((data->simulationInfo->realParameter[1849] /* theta[343] PARAM */) + 0.003720000000000001));
  TRACE_POP
}

/*
equation index: 5477
type: SIMPLE_ASSIGN
x[343] = r_init[343] * cos(theta[343] + 0.003720000000000001)
*/
void SpiralGalaxy_eqFunction_5477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1842]] /* x[343] STATE(1,vx[343]) */) = ((data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */)) * (cos((data->simulationInfo->realParameter[1849] /* theta[343] PARAM */) + 0.003720000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11426(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11427(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11430(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11429(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11428(DATA *data, threadData_t *threadData);


/*
equation index: 5483
type: SIMPLE_ASSIGN
vx[343] = (-sin(theta[343])) * r_init[343] * omega_c[343]
*/
void SpiralGalaxy_eqFunction_5483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[342]] /* vx[343] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1849] /* theta[343] PARAM */)))) * (((data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */)) * ((data->simulationInfo->realParameter[847] /* omega_c[343] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11423(DATA *data, threadData_t *threadData);


/*
equation index: 5485
type: SIMPLE_ASSIGN
vy[343] = cos(theta[343]) * r_init[343] * omega_c[343]
*/
void SpiralGalaxy_eqFunction_5485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[842]] /* vy[343] STATE(1) */) = (cos((data->simulationInfo->realParameter[1849] /* theta[343] PARAM */))) * (((data->simulationInfo->realParameter[1348] /* r_init[343] PARAM */)) * ((data->simulationInfo->realParameter[847] /* omega_c[343] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11422(DATA *data, threadData_t *threadData);


/*
equation index: 5487
type: SIMPLE_ASSIGN
vz[343] = 0.0
*/
void SpiralGalaxy_eqFunction_5487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5487};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1342]] /* vz[343] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11421(DATA *data, threadData_t *threadData);


/*
equation index: 5489
type: SIMPLE_ASSIGN
z[344] = 0.015040000000000001
*/
void SpiralGalaxy_eqFunction_5489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5489};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2843]] /* z[344] STATE(1,vz[344]) */) = 0.015040000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11434(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11435(DATA *data, threadData_t *threadData);


/*
equation index: 5492
type: SIMPLE_ASSIGN
y[344] = r_init[344] * sin(theta[344] + 0.003759999999999999)
*/
void SpiralGalaxy_eqFunction_5492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5492};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2343]] /* y[344] STATE(1,vy[344]) */) = ((data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */)) * (sin((data->simulationInfo->realParameter[1850] /* theta[344] PARAM */) + 0.003759999999999999));
  TRACE_POP
}

/*
equation index: 5493
type: SIMPLE_ASSIGN
x[344] = r_init[344] * cos(theta[344] + 0.003759999999999999)
*/
void SpiralGalaxy_eqFunction_5493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1843]] /* x[344] STATE(1,vx[344]) */) = ((data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */)) * (cos((data->simulationInfo->realParameter[1850] /* theta[344] PARAM */) + 0.003759999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11436(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11437(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11440(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11439(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11438(DATA *data, threadData_t *threadData);


/*
equation index: 5499
type: SIMPLE_ASSIGN
vx[344] = (-sin(theta[344])) * r_init[344] * omega_c[344]
*/
void SpiralGalaxy_eqFunction_5499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5499};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[343]] /* vx[344] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1850] /* theta[344] PARAM */)))) * (((data->simulationInfo->realParameter[1349] /* r_init[344] PARAM */)) * ((data->simulationInfo->realParameter[848] /* omega_c[344] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11433(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_10(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_11129(data, threadData);
  SpiralGalaxy_eqFunction_11128(data, threadData);
  SpiralGalaxy_eqFunction_5003(data, threadData);
  SpiralGalaxy_eqFunction_11123(data, threadData);
  SpiralGalaxy_eqFunction_5005(data, threadData);
  SpiralGalaxy_eqFunction_11122(data, threadData);
  SpiralGalaxy_eqFunction_5007(data, threadData);
  SpiralGalaxy_eqFunction_11121(data, threadData);
  SpiralGalaxy_eqFunction_5009(data, threadData);
  SpiralGalaxy_eqFunction_11134(data, threadData);
  SpiralGalaxy_eqFunction_11135(data, threadData);
  SpiralGalaxy_eqFunction_5012(data, threadData);
  SpiralGalaxy_eqFunction_5013(data, threadData);
  SpiralGalaxy_eqFunction_11136(data, threadData);
  SpiralGalaxy_eqFunction_11137(data, threadData);
  SpiralGalaxy_eqFunction_11140(data, threadData);
  SpiralGalaxy_eqFunction_11139(data, threadData);
  SpiralGalaxy_eqFunction_11138(data, threadData);
  SpiralGalaxy_eqFunction_5019(data, threadData);
  SpiralGalaxy_eqFunction_11133(data, threadData);
  SpiralGalaxy_eqFunction_5021(data, threadData);
  SpiralGalaxy_eqFunction_11132(data, threadData);
  SpiralGalaxy_eqFunction_5023(data, threadData);
  SpiralGalaxy_eqFunction_11131(data, threadData);
  SpiralGalaxy_eqFunction_5025(data, threadData);
  SpiralGalaxy_eqFunction_11144(data, threadData);
  SpiralGalaxy_eqFunction_11145(data, threadData);
  SpiralGalaxy_eqFunction_5028(data, threadData);
  SpiralGalaxy_eqFunction_5029(data, threadData);
  SpiralGalaxy_eqFunction_11146(data, threadData);
  SpiralGalaxy_eqFunction_11147(data, threadData);
  SpiralGalaxy_eqFunction_11150(data, threadData);
  SpiralGalaxy_eqFunction_11149(data, threadData);
  SpiralGalaxy_eqFunction_11148(data, threadData);
  SpiralGalaxy_eqFunction_5035(data, threadData);
  SpiralGalaxy_eqFunction_11143(data, threadData);
  SpiralGalaxy_eqFunction_5037(data, threadData);
  SpiralGalaxy_eqFunction_11142(data, threadData);
  SpiralGalaxy_eqFunction_5039(data, threadData);
  SpiralGalaxy_eqFunction_11141(data, threadData);
  SpiralGalaxy_eqFunction_5041(data, threadData);
  SpiralGalaxy_eqFunction_11154(data, threadData);
  SpiralGalaxy_eqFunction_11155(data, threadData);
  SpiralGalaxy_eqFunction_5044(data, threadData);
  SpiralGalaxy_eqFunction_5045(data, threadData);
  SpiralGalaxy_eqFunction_11156(data, threadData);
  SpiralGalaxy_eqFunction_11157(data, threadData);
  SpiralGalaxy_eqFunction_11160(data, threadData);
  SpiralGalaxy_eqFunction_11159(data, threadData);
  SpiralGalaxy_eqFunction_11158(data, threadData);
  SpiralGalaxy_eqFunction_5051(data, threadData);
  SpiralGalaxy_eqFunction_11153(data, threadData);
  SpiralGalaxy_eqFunction_5053(data, threadData);
  SpiralGalaxy_eqFunction_11152(data, threadData);
  SpiralGalaxy_eqFunction_5055(data, threadData);
  SpiralGalaxy_eqFunction_11151(data, threadData);
  SpiralGalaxy_eqFunction_5057(data, threadData);
  SpiralGalaxy_eqFunction_11164(data, threadData);
  SpiralGalaxy_eqFunction_11165(data, threadData);
  SpiralGalaxy_eqFunction_5060(data, threadData);
  SpiralGalaxy_eqFunction_5061(data, threadData);
  SpiralGalaxy_eqFunction_11166(data, threadData);
  SpiralGalaxy_eqFunction_11167(data, threadData);
  SpiralGalaxy_eqFunction_11170(data, threadData);
  SpiralGalaxy_eqFunction_11169(data, threadData);
  SpiralGalaxy_eqFunction_11168(data, threadData);
  SpiralGalaxy_eqFunction_5067(data, threadData);
  SpiralGalaxy_eqFunction_11163(data, threadData);
  SpiralGalaxy_eqFunction_5069(data, threadData);
  SpiralGalaxy_eqFunction_11162(data, threadData);
  SpiralGalaxy_eqFunction_5071(data, threadData);
  SpiralGalaxy_eqFunction_11161(data, threadData);
  SpiralGalaxy_eqFunction_5073(data, threadData);
  SpiralGalaxy_eqFunction_11174(data, threadData);
  SpiralGalaxy_eqFunction_11175(data, threadData);
  SpiralGalaxy_eqFunction_5076(data, threadData);
  SpiralGalaxy_eqFunction_5077(data, threadData);
  SpiralGalaxy_eqFunction_11176(data, threadData);
  SpiralGalaxy_eqFunction_11177(data, threadData);
  SpiralGalaxy_eqFunction_11180(data, threadData);
  SpiralGalaxy_eqFunction_11179(data, threadData);
  SpiralGalaxy_eqFunction_11178(data, threadData);
  SpiralGalaxy_eqFunction_5083(data, threadData);
  SpiralGalaxy_eqFunction_11173(data, threadData);
  SpiralGalaxy_eqFunction_5085(data, threadData);
  SpiralGalaxy_eqFunction_11172(data, threadData);
  SpiralGalaxy_eqFunction_5087(data, threadData);
  SpiralGalaxy_eqFunction_11171(data, threadData);
  SpiralGalaxy_eqFunction_5089(data, threadData);
  SpiralGalaxy_eqFunction_11184(data, threadData);
  SpiralGalaxy_eqFunction_11185(data, threadData);
  SpiralGalaxy_eqFunction_5092(data, threadData);
  SpiralGalaxy_eqFunction_5093(data, threadData);
  SpiralGalaxy_eqFunction_11186(data, threadData);
  SpiralGalaxy_eqFunction_11187(data, threadData);
  SpiralGalaxy_eqFunction_11190(data, threadData);
  SpiralGalaxy_eqFunction_11189(data, threadData);
  SpiralGalaxy_eqFunction_11188(data, threadData);
  SpiralGalaxy_eqFunction_5099(data, threadData);
  SpiralGalaxy_eqFunction_11183(data, threadData);
  SpiralGalaxy_eqFunction_5101(data, threadData);
  SpiralGalaxy_eqFunction_11182(data, threadData);
  SpiralGalaxy_eqFunction_5103(data, threadData);
  SpiralGalaxy_eqFunction_11181(data, threadData);
  SpiralGalaxy_eqFunction_5105(data, threadData);
  SpiralGalaxy_eqFunction_11194(data, threadData);
  SpiralGalaxy_eqFunction_11195(data, threadData);
  SpiralGalaxy_eqFunction_5108(data, threadData);
  SpiralGalaxy_eqFunction_5109(data, threadData);
  SpiralGalaxy_eqFunction_11196(data, threadData);
  SpiralGalaxy_eqFunction_11197(data, threadData);
  SpiralGalaxy_eqFunction_11200(data, threadData);
  SpiralGalaxy_eqFunction_11199(data, threadData);
  SpiralGalaxy_eqFunction_11198(data, threadData);
  SpiralGalaxy_eqFunction_5115(data, threadData);
  SpiralGalaxy_eqFunction_11193(data, threadData);
  SpiralGalaxy_eqFunction_5117(data, threadData);
  SpiralGalaxy_eqFunction_11192(data, threadData);
  SpiralGalaxy_eqFunction_5119(data, threadData);
  SpiralGalaxy_eqFunction_11191(data, threadData);
  SpiralGalaxy_eqFunction_5121(data, threadData);
  SpiralGalaxy_eqFunction_11204(data, threadData);
  SpiralGalaxy_eqFunction_11205(data, threadData);
  SpiralGalaxy_eqFunction_5124(data, threadData);
  SpiralGalaxy_eqFunction_5125(data, threadData);
  SpiralGalaxy_eqFunction_11206(data, threadData);
  SpiralGalaxy_eqFunction_11207(data, threadData);
  SpiralGalaxy_eqFunction_11210(data, threadData);
  SpiralGalaxy_eqFunction_11209(data, threadData);
  SpiralGalaxy_eqFunction_11208(data, threadData);
  SpiralGalaxy_eqFunction_5131(data, threadData);
  SpiralGalaxy_eqFunction_11203(data, threadData);
  SpiralGalaxy_eqFunction_5133(data, threadData);
  SpiralGalaxy_eqFunction_11202(data, threadData);
  SpiralGalaxy_eqFunction_5135(data, threadData);
  SpiralGalaxy_eqFunction_11201(data, threadData);
  SpiralGalaxy_eqFunction_5137(data, threadData);
  SpiralGalaxy_eqFunction_11214(data, threadData);
  SpiralGalaxy_eqFunction_11215(data, threadData);
  SpiralGalaxy_eqFunction_5140(data, threadData);
  SpiralGalaxy_eqFunction_5141(data, threadData);
  SpiralGalaxy_eqFunction_11216(data, threadData);
  SpiralGalaxy_eqFunction_11217(data, threadData);
  SpiralGalaxy_eqFunction_11220(data, threadData);
  SpiralGalaxy_eqFunction_11219(data, threadData);
  SpiralGalaxy_eqFunction_11218(data, threadData);
  SpiralGalaxy_eqFunction_5147(data, threadData);
  SpiralGalaxy_eqFunction_11213(data, threadData);
  SpiralGalaxy_eqFunction_5149(data, threadData);
  SpiralGalaxy_eqFunction_11212(data, threadData);
  SpiralGalaxy_eqFunction_5151(data, threadData);
  SpiralGalaxy_eqFunction_11211(data, threadData);
  SpiralGalaxy_eqFunction_5153(data, threadData);
  SpiralGalaxy_eqFunction_11224(data, threadData);
  SpiralGalaxy_eqFunction_11225(data, threadData);
  SpiralGalaxy_eqFunction_5156(data, threadData);
  SpiralGalaxy_eqFunction_5157(data, threadData);
  SpiralGalaxy_eqFunction_11226(data, threadData);
  SpiralGalaxy_eqFunction_11227(data, threadData);
  SpiralGalaxy_eqFunction_11230(data, threadData);
  SpiralGalaxy_eqFunction_11229(data, threadData);
  SpiralGalaxy_eqFunction_11228(data, threadData);
  SpiralGalaxy_eqFunction_5163(data, threadData);
  SpiralGalaxy_eqFunction_11223(data, threadData);
  SpiralGalaxy_eqFunction_5165(data, threadData);
  SpiralGalaxy_eqFunction_11222(data, threadData);
  SpiralGalaxy_eqFunction_5167(data, threadData);
  SpiralGalaxy_eqFunction_11221(data, threadData);
  SpiralGalaxy_eqFunction_5169(data, threadData);
  SpiralGalaxy_eqFunction_11234(data, threadData);
  SpiralGalaxy_eqFunction_11235(data, threadData);
  SpiralGalaxy_eqFunction_5172(data, threadData);
  SpiralGalaxy_eqFunction_5173(data, threadData);
  SpiralGalaxy_eqFunction_11236(data, threadData);
  SpiralGalaxy_eqFunction_11237(data, threadData);
  SpiralGalaxy_eqFunction_11240(data, threadData);
  SpiralGalaxy_eqFunction_11239(data, threadData);
  SpiralGalaxy_eqFunction_11238(data, threadData);
  SpiralGalaxy_eqFunction_5179(data, threadData);
  SpiralGalaxy_eqFunction_11233(data, threadData);
  SpiralGalaxy_eqFunction_5181(data, threadData);
  SpiralGalaxy_eqFunction_11232(data, threadData);
  SpiralGalaxy_eqFunction_5183(data, threadData);
  SpiralGalaxy_eqFunction_11231(data, threadData);
  SpiralGalaxy_eqFunction_5185(data, threadData);
  SpiralGalaxy_eqFunction_11244(data, threadData);
  SpiralGalaxy_eqFunction_11245(data, threadData);
  SpiralGalaxy_eqFunction_5188(data, threadData);
  SpiralGalaxy_eqFunction_5189(data, threadData);
  SpiralGalaxy_eqFunction_11246(data, threadData);
  SpiralGalaxy_eqFunction_11247(data, threadData);
  SpiralGalaxy_eqFunction_11250(data, threadData);
  SpiralGalaxy_eqFunction_11249(data, threadData);
  SpiralGalaxy_eqFunction_11248(data, threadData);
  SpiralGalaxy_eqFunction_5195(data, threadData);
  SpiralGalaxy_eqFunction_11243(data, threadData);
  SpiralGalaxy_eqFunction_5197(data, threadData);
  SpiralGalaxy_eqFunction_11242(data, threadData);
  SpiralGalaxy_eqFunction_5199(data, threadData);
  SpiralGalaxy_eqFunction_11241(data, threadData);
  SpiralGalaxy_eqFunction_5201(data, threadData);
  SpiralGalaxy_eqFunction_11254(data, threadData);
  SpiralGalaxy_eqFunction_11255(data, threadData);
  SpiralGalaxy_eqFunction_5204(data, threadData);
  SpiralGalaxy_eqFunction_5205(data, threadData);
  SpiralGalaxy_eqFunction_11256(data, threadData);
  SpiralGalaxy_eqFunction_11257(data, threadData);
  SpiralGalaxy_eqFunction_11260(data, threadData);
  SpiralGalaxy_eqFunction_11259(data, threadData);
  SpiralGalaxy_eqFunction_11258(data, threadData);
  SpiralGalaxy_eqFunction_5211(data, threadData);
  SpiralGalaxy_eqFunction_11253(data, threadData);
  SpiralGalaxy_eqFunction_5213(data, threadData);
  SpiralGalaxy_eqFunction_11252(data, threadData);
  SpiralGalaxy_eqFunction_5215(data, threadData);
  SpiralGalaxy_eqFunction_11251(data, threadData);
  SpiralGalaxy_eqFunction_5217(data, threadData);
  SpiralGalaxy_eqFunction_11264(data, threadData);
  SpiralGalaxy_eqFunction_11265(data, threadData);
  SpiralGalaxy_eqFunction_5220(data, threadData);
  SpiralGalaxy_eqFunction_5221(data, threadData);
  SpiralGalaxy_eqFunction_11266(data, threadData);
  SpiralGalaxy_eqFunction_11267(data, threadData);
  SpiralGalaxy_eqFunction_11270(data, threadData);
  SpiralGalaxy_eqFunction_11269(data, threadData);
  SpiralGalaxy_eqFunction_11268(data, threadData);
  SpiralGalaxy_eqFunction_5227(data, threadData);
  SpiralGalaxy_eqFunction_11263(data, threadData);
  SpiralGalaxy_eqFunction_5229(data, threadData);
  SpiralGalaxy_eqFunction_11262(data, threadData);
  SpiralGalaxy_eqFunction_5231(data, threadData);
  SpiralGalaxy_eqFunction_11261(data, threadData);
  SpiralGalaxy_eqFunction_5233(data, threadData);
  SpiralGalaxy_eqFunction_11274(data, threadData);
  SpiralGalaxy_eqFunction_11275(data, threadData);
  SpiralGalaxy_eqFunction_5236(data, threadData);
  SpiralGalaxy_eqFunction_5237(data, threadData);
  SpiralGalaxy_eqFunction_11276(data, threadData);
  SpiralGalaxy_eqFunction_11277(data, threadData);
  SpiralGalaxy_eqFunction_11280(data, threadData);
  SpiralGalaxy_eqFunction_11279(data, threadData);
  SpiralGalaxy_eqFunction_11278(data, threadData);
  SpiralGalaxy_eqFunction_5243(data, threadData);
  SpiralGalaxy_eqFunction_11273(data, threadData);
  SpiralGalaxy_eqFunction_5245(data, threadData);
  SpiralGalaxy_eqFunction_11272(data, threadData);
  SpiralGalaxy_eqFunction_5247(data, threadData);
  SpiralGalaxy_eqFunction_11271(data, threadData);
  SpiralGalaxy_eqFunction_5249(data, threadData);
  SpiralGalaxy_eqFunction_11284(data, threadData);
  SpiralGalaxy_eqFunction_11285(data, threadData);
  SpiralGalaxy_eqFunction_5252(data, threadData);
  SpiralGalaxy_eqFunction_5253(data, threadData);
  SpiralGalaxy_eqFunction_11286(data, threadData);
  SpiralGalaxy_eqFunction_11287(data, threadData);
  SpiralGalaxy_eqFunction_11290(data, threadData);
  SpiralGalaxy_eqFunction_11289(data, threadData);
  SpiralGalaxy_eqFunction_11288(data, threadData);
  SpiralGalaxy_eqFunction_5259(data, threadData);
  SpiralGalaxy_eqFunction_11283(data, threadData);
  SpiralGalaxy_eqFunction_5261(data, threadData);
  SpiralGalaxy_eqFunction_11282(data, threadData);
  SpiralGalaxy_eqFunction_5263(data, threadData);
  SpiralGalaxy_eqFunction_11281(data, threadData);
  SpiralGalaxy_eqFunction_5265(data, threadData);
  SpiralGalaxy_eqFunction_11294(data, threadData);
  SpiralGalaxy_eqFunction_11295(data, threadData);
  SpiralGalaxy_eqFunction_5268(data, threadData);
  SpiralGalaxy_eqFunction_5269(data, threadData);
  SpiralGalaxy_eqFunction_11296(data, threadData);
  SpiralGalaxy_eqFunction_11297(data, threadData);
  SpiralGalaxy_eqFunction_11300(data, threadData);
  SpiralGalaxy_eqFunction_11299(data, threadData);
  SpiralGalaxy_eqFunction_11298(data, threadData);
  SpiralGalaxy_eqFunction_5275(data, threadData);
  SpiralGalaxy_eqFunction_11293(data, threadData);
  SpiralGalaxy_eqFunction_5277(data, threadData);
  SpiralGalaxy_eqFunction_11292(data, threadData);
  SpiralGalaxy_eqFunction_5279(data, threadData);
  SpiralGalaxy_eqFunction_11291(data, threadData);
  SpiralGalaxy_eqFunction_5281(data, threadData);
  SpiralGalaxy_eqFunction_11304(data, threadData);
  SpiralGalaxy_eqFunction_11305(data, threadData);
  SpiralGalaxy_eqFunction_5284(data, threadData);
  SpiralGalaxy_eqFunction_5285(data, threadData);
  SpiralGalaxy_eqFunction_11306(data, threadData);
  SpiralGalaxy_eqFunction_11307(data, threadData);
  SpiralGalaxy_eqFunction_11310(data, threadData);
  SpiralGalaxy_eqFunction_11309(data, threadData);
  SpiralGalaxy_eqFunction_11308(data, threadData);
  SpiralGalaxy_eqFunction_5291(data, threadData);
  SpiralGalaxy_eqFunction_11303(data, threadData);
  SpiralGalaxy_eqFunction_5293(data, threadData);
  SpiralGalaxy_eqFunction_11302(data, threadData);
  SpiralGalaxy_eqFunction_5295(data, threadData);
  SpiralGalaxy_eqFunction_11301(data, threadData);
  SpiralGalaxy_eqFunction_5297(data, threadData);
  SpiralGalaxy_eqFunction_11314(data, threadData);
  SpiralGalaxy_eqFunction_11315(data, threadData);
  SpiralGalaxy_eqFunction_5300(data, threadData);
  SpiralGalaxy_eqFunction_5301(data, threadData);
  SpiralGalaxy_eqFunction_11316(data, threadData);
  SpiralGalaxy_eqFunction_11317(data, threadData);
  SpiralGalaxy_eqFunction_11320(data, threadData);
  SpiralGalaxy_eqFunction_11319(data, threadData);
  SpiralGalaxy_eqFunction_11318(data, threadData);
  SpiralGalaxy_eqFunction_5307(data, threadData);
  SpiralGalaxy_eqFunction_11313(data, threadData);
  SpiralGalaxy_eqFunction_5309(data, threadData);
  SpiralGalaxy_eqFunction_11312(data, threadData);
  SpiralGalaxy_eqFunction_5311(data, threadData);
  SpiralGalaxy_eqFunction_11311(data, threadData);
  SpiralGalaxy_eqFunction_5313(data, threadData);
  SpiralGalaxy_eqFunction_11324(data, threadData);
  SpiralGalaxy_eqFunction_11325(data, threadData);
  SpiralGalaxy_eqFunction_5316(data, threadData);
  SpiralGalaxy_eqFunction_5317(data, threadData);
  SpiralGalaxy_eqFunction_11326(data, threadData);
  SpiralGalaxy_eqFunction_11327(data, threadData);
  SpiralGalaxy_eqFunction_11330(data, threadData);
  SpiralGalaxy_eqFunction_11329(data, threadData);
  SpiralGalaxy_eqFunction_11328(data, threadData);
  SpiralGalaxy_eqFunction_5323(data, threadData);
  SpiralGalaxy_eqFunction_11323(data, threadData);
  SpiralGalaxy_eqFunction_5325(data, threadData);
  SpiralGalaxy_eqFunction_11322(data, threadData);
  SpiralGalaxy_eqFunction_5327(data, threadData);
  SpiralGalaxy_eqFunction_11321(data, threadData);
  SpiralGalaxy_eqFunction_5329(data, threadData);
  SpiralGalaxy_eqFunction_11334(data, threadData);
  SpiralGalaxy_eqFunction_11335(data, threadData);
  SpiralGalaxy_eqFunction_5332(data, threadData);
  SpiralGalaxy_eqFunction_5333(data, threadData);
  SpiralGalaxy_eqFunction_11336(data, threadData);
  SpiralGalaxy_eqFunction_11337(data, threadData);
  SpiralGalaxy_eqFunction_11340(data, threadData);
  SpiralGalaxy_eqFunction_11339(data, threadData);
  SpiralGalaxy_eqFunction_11338(data, threadData);
  SpiralGalaxy_eqFunction_5339(data, threadData);
  SpiralGalaxy_eqFunction_11333(data, threadData);
  SpiralGalaxy_eqFunction_5341(data, threadData);
  SpiralGalaxy_eqFunction_11332(data, threadData);
  SpiralGalaxy_eqFunction_5343(data, threadData);
  SpiralGalaxy_eqFunction_11331(data, threadData);
  SpiralGalaxy_eqFunction_5345(data, threadData);
  SpiralGalaxy_eqFunction_11344(data, threadData);
  SpiralGalaxy_eqFunction_11345(data, threadData);
  SpiralGalaxy_eqFunction_5348(data, threadData);
  SpiralGalaxy_eqFunction_5349(data, threadData);
  SpiralGalaxy_eqFunction_11346(data, threadData);
  SpiralGalaxy_eqFunction_11347(data, threadData);
  SpiralGalaxy_eqFunction_11350(data, threadData);
  SpiralGalaxy_eqFunction_11349(data, threadData);
  SpiralGalaxy_eqFunction_11348(data, threadData);
  SpiralGalaxy_eqFunction_5355(data, threadData);
  SpiralGalaxy_eqFunction_11343(data, threadData);
  SpiralGalaxy_eqFunction_5357(data, threadData);
  SpiralGalaxy_eqFunction_11342(data, threadData);
  SpiralGalaxy_eqFunction_5359(data, threadData);
  SpiralGalaxy_eqFunction_11341(data, threadData);
  SpiralGalaxy_eqFunction_5361(data, threadData);
  SpiralGalaxy_eqFunction_11354(data, threadData);
  SpiralGalaxy_eqFunction_11355(data, threadData);
  SpiralGalaxy_eqFunction_5364(data, threadData);
  SpiralGalaxy_eqFunction_5365(data, threadData);
  SpiralGalaxy_eqFunction_11356(data, threadData);
  SpiralGalaxy_eqFunction_11357(data, threadData);
  SpiralGalaxy_eqFunction_11360(data, threadData);
  SpiralGalaxy_eqFunction_11359(data, threadData);
  SpiralGalaxy_eqFunction_11358(data, threadData);
  SpiralGalaxy_eqFunction_5371(data, threadData);
  SpiralGalaxy_eqFunction_11353(data, threadData);
  SpiralGalaxy_eqFunction_5373(data, threadData);
  SpiralGalaxy_eqFunction_11352(data, threadData);
  SpiralGalaxy_eqFunction_5375(data, threadData);
  SpiralGalaxy_eqFunction_11351(data, threadData);
  SpiralGalaxy_eqFunction_5377(data, threadData);
  SpiralGalaxy_eqFunction_11364(data, threadData);
  SpiralGalaxy_eqFunction_11365(data, threadData);
  SpiralGalaxy_eqFunction_5380(data, threadData);
  SpiralGalaxy_eqFunction_5381(data, threadData);
  SpiralGalaxy_eqFunction_11366(data, threadData);
  SpiralGalaxy_eqFunction_11367(data, threadData);
  SpiralGalaxy_eqFunction_11370(data, threadData);
  SpiralGalaxy_eqFunction_11369(data, threadData);
  SpiralGalaxy_eqFunction_11368(data, threadData);
  SpiralGalaxy_eqFunction_5387(data, threadData);
  SpiralGalaxy_eqFunction_11363(data, threadData);
  SpiralGalaxy_eqFunction_5389(data, threadData);
  SpiralGalaxy_eqFunction_11362(data, threadData);
  SpiralGalaxy_eqFunction_5391(data, threadData);
  SpiralGalaxy_eqFunction_11361(data, threadData);
  SpiralGalaxy_eqFunction_5393(data, threadData);
  SpiralGalaxy_eqFunction_11374(data, threadData);
  SpiralGalaxy_eqFunction_11375(data, threadData);
  SpiralGalaxy_eqFunction_5396(data, threadData);
  SpiralGalaxy_eqFunction_5397(data, threadData);
  SpiralGalaxy_eqFunction_11376(data, threadData);
  SpiralGalaxy_eqFunction_11377(data, threadData);
  SpiralGalaxy_eqFunction_11380(data, threadData);
  SpiralGalaxy_eqFunction_11379(data, threadData);
  SpiralGalaxy_eqFunction_11378(data, threadData);
  SpiralGalaxy_eqFunction_5403(data, threadData);
  SpiralGalaxy_eqFunction_11373(data, threadData);
  SpiralGalaxy_eqFunction_5405(data, threadData);
  SpiralGalaxy_eqFunction_11372(data, threadData);
  SpiralGalaxy_eqFunction_5407(data, threadData);
  SpiralGalaxy_eqFunction_11371(data, threadData);
  SpiralGalaxy_eqFunction_5409(data, threadData);
  SpiralGalaxy_eqFunction_11384(data, threadData);
  SpiralGalaxy_eqFunction_11385(data, threadData);
  SpiralGalaxy_eqFunction_5412(data, threadData);
  SpiralGalaxy_eqFunction_5413(data, threadData);
  SpiralGalaxy_eqFunction_11386(data, threadData);
  SpiralGalaxy_eqFunction_11387(data, threadData);
  SpiralGalaxy_eqFunction_11390(data, threadData);
  SpiralGalaxy_eqFunction_11389(data, threadData);
  SpiralGalaxy_eqFunction_11388(data, threadData);
  SpiralGalaxy_eqFunction_5419(data, threadData);
  SpiralGalaxy_eqFunction_11383(data, threadData);
  SpiralGalaxy_eqFunction_5421(data, threadData);
  SpiralGalaxy_eqFunction_11382(data, threadData);
  SpiralGalaxy_eqFunction_5423(data, threadData);
  SpiralGalaxy_eqFunction_11381(data, threadData);
  SpiralGalaxy_eqFunction_5425(data, threadData);
  SpiralGalaxy_eqFunction_11394(data, threadData);
  SpiralGalaxy_eqFunction_11395(data, threadData);
  SpiralGalaxy_eqFunction_5428(data, threadData);
  SpiralGalaxy_eqFunction_5429(data, threadData);
  SpiralGalaxy_eqFunction_11396(data, threadData);
  SpiralGalaxy_eqFunction_11397(data, threadData);
  SpiralGalaxy_eqFunction_11400(data, threadData);
  SpiralGalaxy_eqFunction_11399(data, threadData);
  SpiralGalaxy_eqFunction_11398(data, threadData);
  SpiralGalaxy_eqFunction_5435(data, threadData);
  SpiralGalaxy_eqFunction_11393(data, threadData);
  SpiralGalaxy_eqFunction_5437(data, threadData);
  SpiralGalaxy_eqFunction_11392(data, threadData);
  SpiralGalaxy_eqFunction_5439(data, threadData);
  SpiralGalaxy_eqFunction_11391(data, threadData);
  SpiralGalaxy_eqFunction_5441(data, threadData);
  SpiralGalaxy_eqFunction_11404(data, threadData);
  SpiralGalaxy_eqFunction_11405(data, threadData);
  SpiralGalaxy_eqFunction_5444(data, threadData);
  SpiralGalaxy_eqFunction_5445(data, threadData);
  SpiralGalaxy_eqFunction_11406(data, threadData);
  SpiralGalaxy_eqFunction_11407(data, threadData);
  SpiralGalaxy_eqFunction_11410(data, threadData);
  SpiralGalaxy_eqFunction_11409(data, threadData);
  SpiralGalaxy_eqFunction_11408(data, threadData);
  SpiralGalaxy_eqFunction_5451(data, threadData);
  SpiralGalaxy_eqFunction_11403(data, threadData);
  SpiralGalaxy_eqFunction_5453(data, threadData);
  SpiralGalaxy_eqFunction_11402(data, threadData);
  SpiralGalaxy_eqFunction_5455(data, threadData);
  SpiralGalaxy_eqFunction_11401(data, threadData);
  SpiralGalaxy_eqFunction_5457(data, threadData);
  SpiralGalaxy_eqFunction_11414(data, threadData);
  SpiralGalaxy_eqFunction_11415(data, threadData);
  SpiralGalaxy_eqFunction_5460(data, threadData);
  SpiralGalaxy_eqFunction_5461(data, threadData);
  SpiralGalaxy_eqFunction_11416(data, threadData);
  SpiralGalaxy_eqFunction_11417(data, threadData);
  SpiralGalaxy_eqFunction_11420(data, threadData);
  SpiralGalaxy_eqFunction_11419(data, threadData);
  SpiralGalaxy_eqFunction_11418(data, threadData);
  SpiralGalaxy_eqFunction_5467(data, threadData);
  SpiralGalaxy_eqFunction_11413(data, threadData);
  SpiralGalaxy_eqFunction_5469(data, threadData);
  SpiralGalaxy_eqFunction_11412(data, threadData);
  SpiralGalaxy_eqFunction_5471(data, threadData);
  SpiralGalaxy_eqFunction_11411(data, threadData);
  SpiralGalaxy_eqFunction_5473(data, threadData);
  SpiralGalaxy_eqFunction_11424(data, threadData);
  SpiralGalaxy_eqFunction_11425(data, threadData);
  SpiralGalaxy_eqFunction_5476(data, threadData);
  SpiralGalaxy_eqFunction_5477(data, threadData);
  SpiralGalaxy_eqFunction_11426(data, threadData);
  SpiralGalaxy_eqFunction_11427(data, threadData);
  SpiralGalaxy_eqFunction_11430(data, threadData);
  SpiralGalaxy_eqFunction_11429(data, threadData);
  SpiralGalaxy_eqFunction_11428(data, threadData);
  SpiralGalaxy_eqFunction_5483(data, threadData);
  SpiralGalaxy_eqFunction_11423(data, threadData);
  SpiralGalaxy_eqFunction_5485(data, threadData);
  SpiralGalaxy_eqFunction_11422(data, threadData);
  SpiralGalaxy_eqFunction_5487(data, threadData);
  SpiralGalaxy_eqFunction_11421(data, threadData);
  SpiralGalaxy_eqFunction_5489(data, threadData);
  SpiralGalaxy_eqFunction_11434(data, threadData);
  SpiralGalaxy_eqFunction_11435(data, threadData);
  SpiralGalaxy_eqFunction_5492(data, threadData);
  SpiralGalaxy_eqFunction_5493(data, threadData);
  SpiralGalaxy_eqFunction_11436(data, threadData);
  SpiralGalaxy_eqFunction_11437(data, threadData);
  SpiralGalaxy_eqFunction_11440(data, threadData);
  SpiralGalaxy_eqFunction_11439(data, threadData);
  SpiralGalaxy_eqFunction_11438(data, threadData);
  SpiralGalaxy_eqFunction_5499(data, threadData);
  SpiralGalaxy_eqFunction_11433(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif