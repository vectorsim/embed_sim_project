#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 6001
type: SIMPLE_ASSIGN
z[376] = 0.02016
*/
void SpiralGalaxy_eqFunction_6001(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6001};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2875]] /* z[376] STATE(1,vz[376]) */) = 0.02016;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11754(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11755(DATA *data, threadData_t *threadData);


/*
equation index: 6004
type: SIMPLE_ASSIGN
y[376] = r_init[376] * sin(theta[376] + 0.00504)
*/
void SpiralGalaxy_eqFunction_6004(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6004};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2375]] /* y[376] STATE(1,vy[376]) */) = ((data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */)) * (sin((data->simulationInfo->realParameter[1882] /* theta[376] PARAM */) + 0.00504));
  TRACE_POP
}

/*
equation index: 6005
type: SIMPLE_ASSIGN
x[376] = r_init[376] * cos(theta[376] + 0.00504)
*/
void SpiralGalaxy_eqFunction_6005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6005};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1875]] /* x[376] STATE(1,vx[376]) */) = ((data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */)) * (cos((data->simulationInfo->realParameter[1882] /* theta[376] PARAM */) + 0.00504));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11756(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11757(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11760(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11759(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11758(DATA *data, threadData_t *threadData);


/*
equation index: 6011
type: SIMPLE_ASSIGN
vx[376] = (-sin(theta[376])) * r_init[376] * omega_c[376]
*/
void SpiralGalaxy_eqFunction_6011(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6011};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[375]] /* vx[376] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1882] /* theta[376] PARAM */)))) * (((data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */)) * ((data->simulationInfo->realParameter[880] /* omega_c[376] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11753(DATA *data, threadData_t *threadData);


/*
equation index: 6013
type: SIMPLE_ASSIGN
vy[376] = cos(theta[376]) * r_init[376] * omega_c[376]
*/
void SpiralGalaxy_eqFunction_6013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6013};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[875]] /* vy[376] STATE(1) */) = (cos((data->simulationInfo->realParameter[1882] /* theta[376] PARAM */))) * (((data->simulationInfo->realParameter[1381] /* r_init[376] PARAM */)) * ((data->simulationInfo->realParameter[880] /* omega_c[376] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11752(DATA *data, threadData_t *threadData);


/*
equation index: 6015
type: SIMPLE_ASSIGN
vz[376] = 0.0
*/
void SpiralGalaxy_eqFunction_6015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6015};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1375]] /* vz[376] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11751(DATA *data, threadData_t *threadData);


/*
equation index: 6017
type: SIMPLE_ASSIGN
z[377] = 0.02032
*/
void SpiralGalaxy_eqFunction_6017(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6017};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2876]] /* z[377] STATE(1,vz[377]) */) = 0.02032;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11764(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11765(DATA *data, threadData_t *threadData);


/*
equation index: 6020
type: SIMPLE_ASSIGN
y[377] = r_init[377] * sin(theta[377] + 0.00508)
*/
void SpiralGalaxy_eqFunction_6020(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6020};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2376]] /* y[377] STATE(1,vy[377]) */) = ((data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */)) * (sin((data->simulationInfo->realParameter[1883] /* theta[377] PARAM */) + 0.00508));
  TRACE_POP
}

/*
equation index: 6021
type: SIMPLE_ASSIGN
x[377] = r_init[377] * cos(theta[377] + 0.00508)
*/
void SpiralGalaxy_eqFunction_6021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6021};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1876]] /* x[377] STATE(1,vx[377]) */) = ((data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */)) * (cos((data->simulationInfo->realParameter[1883] /* theta[377] PARAM */) + 0.00508));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11766(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11767(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11770(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11769(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11768(DATA *data, threadData_t *threadData);


/*
equation index: 6027
type: SIMPLE_ASSIGN
vx[377] = (-sin(theta[377])) * r_init[377] * omega_c[377]
*/
void SpiralGalaxy_eqFunction_6027(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6027};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[376]] /* vx[377] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1883] /* theta[377] PARAM */)))) * (((data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */)) * ((data->simulationInfo->realParameter[881] /* omega_c[377] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11763(DATA *data, threadData_t *threadData);


/*
equation index: 6029
type: SIMPLE_ASSIGN
vy[377] = cos(theta[377]) * r_init[377] * omega_c[377]
*/
void SpiralGalaxy_eqFunction_6029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6029};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[876]] /* vy[377] STATE(1) */) = (cos((data->simulationInfo->realParameter[1883] /* theta[377] PARAM */))) * (((data->simulationInfo->realParameter[1382] /* r_init[377] PARAM */)) * ((data->simulationInfo->realParameter[881] /* omega_c[377] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11762(DATA *data, threadData_t *threadData);


/*
equation index: 6031
type: SIMPLE_ASSIGN
vz[377] = 0.0
*/
void SpiralGalaxy_eqFunction_6031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6031};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1376]] /* vz[377] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11761(DATA *data, threadData_t *threadData);


/*
equation index: 6033
type: SIMPLE_ASSIGN
z[378] = 0.02048
*/
void SpiralGalaxy_eqFunction_6033(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6033};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2877]] /* z[378] STATE(1,vz[378]) */) = 0.02048;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11774(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11775(DATA *data, threadData_t *threadData);


/*
equation index: 6036
type: SIMPLE_ASSIGN
y[378] = r_init[378] * sin(theta[378] + 0.00512)
*/
void SpiralGalaxy_eqFunction_6036(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6036};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2377]] /* y[378] STATE(1,vy[378]) */) = ((data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */)) * (sin((data->simulationInfo->realParameter[1884] /* theta[378] PARAM */) + 0.00512));
  TRACE_POP
}

/*
equation index: 6037
type: SIMPLE_ASSIGN
x[378] = r_init[378] * cos(theta[378] + 0.00512)
*/
void SpiralGalaxy_eqFunction_6037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6037};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1877]] /* x[378] STATE(1,vx[378]) */) = ((data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */)) * (cos((data->simulationInfo->realParameter[1884] /* theta[378] PARAM */) + 0.00512));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11776(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11777(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11780(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11779(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11778(DATA *data, threadData_t *threadData);


/*
equation index: 6043
type: SIMPLE_ASSIGN
vx[378] = (-sin(theta[378])) * r_init[378] * omega_c[378]
*/
void SpiralGalaxy_eqFunction_6043(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6043};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[377]] /* vx[378] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1884] /* theta[378] PARAM */)))) * (((data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */)) * ((data->simulationInfo->realParameter[882] /* omega_c[378] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11773(DATA *data, threadData_t *threadData);


/*
equation index: 6045
type: SIMPLE_ASSIGN
vy[378] = cos(theta[378]) * r_init[378] * omega_c[378]
*/
void SpiralGalaxy_eqFunction_6045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6045};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[877]] /* vy[378] STATE(1) */) = (cos((data->simulationInfo->realParameter[1884] /* theta[378] PARAM */))) * (((data->simulationInfo->realParameter[1383] /* r_init[378] PARAM */)) * ((data->simulationInfo->realParameter[882] /* omega_c[378] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11772(DATA *data, threadData_t *threadData);


/*
equation index: 6047
type: SIMPLE_ASSIGN
vz[378] = 0.0
*/
void SpiralGalaxy_eqFunction_6047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6047};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1377]] /* vz[378] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11771(DATA *data, threadData_t *threadData);


/*
equation index: 6049
type: SIMPLE_ASSIGN
z[379] = 0.020640000000000002
*/
void SpiralGalaxy_eqFunction_6049(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6049};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2878]] /* z[379] STATE(1,vz[379]) */) = 0.020640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11784(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11785(DATA *data, threadData_t *threadData);


/*
equation index: 6052
type: SIMPLE_ASSIGN
y[379] = r_init[379] * sin(theta[379] + 0.0051600000000000005)
*/
void SpiralGalaxy_eqFunction_6052(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6052};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2378]] /* y[379] STATE(1,vy[379]) */) = ((data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */)) * (sin((data->simulationInfo->realParameter[1885] /* theta[379] PARAM */) + 0.0051600000000000005));
  TRACE_POP
}

/*
equation index: 6053
type: SIMPLE_ASSIGN
x[379] = r_init[379] * cos(theta[379] + 0.0051600000000000005)
*/
void SpiralGalaxy_eqFunction_6053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6053};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1878]] /* x[379] STATE(1,vx[379]) */) = ((data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */)) * (cos((data->simulationInfo->realParameter[1885] /* theta[379] PARAM */) + 0.0051600000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11786(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11787(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11790(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11789(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11788(DATA *data, threadData_t *threadData);


/*
equation index: 6059
type: SIMPLE_ASSIGN
vx[379] = (-sin(theta[379])) * r_init[379] * omega_c[379]
*/
void SpiralGalaxy_eqFunction_6059(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6059};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[378]] /* vx[379] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1885] /* theta[379] PARAM */)))) * (((data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */)) * ((data->simulationInfo->realParameter[883] /* omega_c[379] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11783(DATA *data, threadData_t *threadData);


/*
equation index: 6061
type: SIMPLE_ASSIGN
vy[379] = cos(theta[379]) * r_init[379] * omega_c[379]
*/
void SpiralGalaxy_eqFunction_6061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6061};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[878]] /* vy[379] STATE(1) */) = (cos((data->simulationInfo->realParameter[1885] /* theta[379] PARAM */))) * (((data->simulationInfo->realParameter[1384] /* r_init[379] PARAM */)) * ((data->simulationInfo->realParameter[883] /* omega_c[379] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11782(DATA *data, threadData_t *threadData);


/*
equation index: 6063
type: SIMPLE_ASSIGN
vz[379] = 0.0
*/
void SpiralGalaxy_eqFunction_6063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6063};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1378]] /* vz[379] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11781(DATA *data, threadData_t *threadData);


/*
equation index: 6065
type: SIMPLE_ASSIGN
z[380] = 0.020800000000000006
*/
void SpiralGalaxy_eqFunction_6065(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6065};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2879]] /* z[380] STATE(1,vz[380]) */) = 0.020800000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11794(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11795(DATA *data, threadData_t *threadData);


/*
equation index: 6068
type: SIMPLE_ASSIGN
y[380] = r_init[380] * sin(theta[380] + 0.005200000000000001)
*/
void SpiralGalaxy_eqFunction_6068(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6068};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2379]] /* y[380] STATE(1,vy[380]) */) = ((data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */)) * (sin((data->simulationInfo->realParameter[1886] /* theta[380] PARAM */) + 0.005200000000000001));
  TRACE_POP
}

/*
equation index: 6069
type: SIMPLE_ASSIGN
x[380] = r_init[380] * cos(theta[380] + 0.005200000000000001)
*/
void SpiralGalaxy_eqFunction_6069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6069};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1879]] /* x[380] STATE(1,vx[380]) */) = ((data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */)) * (cos((data->simulationInfo->realParameter[1886] /* theta[380] PARAM */) + 0.005200000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11796(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11797(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11800(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11799(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11798(DATA *data, threadData_t *threadData);


/*
equation index: 6075
type: SIMPLE_ASSIGN
vx[380] = (-sin(theta[380])) * r_init[380] * omega_c[380]
*/
void SpiralGalaxy_eqFunction_6075(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6075};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[379]] /* vx[380] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1886] /* theta[380] PARAM */)))) * (((data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */)) * ((data->simulationInfo->realParameter[884] /* omega_c[380] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11793(DATA *data, threadData_t *threadData);


/*
equation index: 6077
type: SIMPLE_ASSIGN
vy[380] = cos(theta[380]) * r_init[380] * omega_c[380]
*/
void SpiralGalaxy_eqFunction_6077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6077};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[879]] /* vy[380] STATE(1) */) = (cos((data->simulationInfo->realParameter[1886] /* theta[380] PARAM */))) * (((data->simulationInfo->realParameter[1385] /* r_init[380] PARAM */)) * ((data->simulationInfo->realParameter[884] /* omega_c[380] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11792(DATA *data, threadData_t *threadData);


/*
equation index: 6079
type: SIMPLE_ASSIGN
vz[380] = 0.0
*/
void SpiralGalaxy_eqFunction_6079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6079};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1379]] /* vz[380] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11791(DATA *data, threadData_t *threadData);


/*
equation index: 6081
type: SIMPLE_ASSIGN
z[381] = 0.020960000000000003
*/
void SpiralGalaxy_eqFunction_6081(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6081};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2880]] /* z[381] STATE(1,vz[381]) */) = 0.020960000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11804(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11805(DATA *data, threadData_t *threadData);


/*
equation index: 6084
type: SIMPLE_ASSIGN
y[381] = r_init[381] * sin(theta[381] + 0.005240000000000001)
*/
void SpiralGalaxy_eqFunction_6084(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6084};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2380]] /* y[381] STATE(1,vy[381]) */) = ((data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */)) * (sin((data->simulationInfo->realParameter[1887] /* theta[381] PARAM */) + 0.005240000000000001));
  TRACE_POP
}

/*
equation index: 6085
type: SIMPLE_ASSIGN
x[381] = r_init[381] * cos(theta[381] + 0.005240000000000001)
*/
void SpiralGalaxy_eqFunction_6085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6085};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1880]] /* x[381] STATE(1,vx[381]) */) = ((data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */)) * (cos((data->simulationInfo->realParameter[1887] /* theta[381] PARAM */) + 0.005240000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11806(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11807(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11810(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11809(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11808(DATA *data, threadData_t *threadData);


/*
equation index: 6091
type: SIMPLE_ASSIGN
vx[381] = (-sin(theta[381])) * r_init[381] * omega_c[381]
*/
void SpiralGalaxy_eqFunction_6091(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6091};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[380]] /* vx[381] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1887] /* theta[381] PARAM */)))) * (((data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */)) * ((data->simulationInfo->realParameter[885] /* omega_c[381] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11803(DATA *data, threadData_t *threadData);


/*
equation index: 6093
type: SIMPLE_ASSIGN
vy[381] = cos(theta[381]) * r_init[381] * omega_c[381]
*/
void SpiralGalaxy_eqFunction_6093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6093};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[880]] /* vy[381] STATE(1) */) = (cos((data->simulationInfo->realParameter[1887] /* theta[381] PARAM */))) * (((data->simulationInfo->realParameter[1386] /* r_init[381] PARAM */)) * ((data->simulationInfo->realParameter[885] /* omega_c[381] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11802(DATA *data, threadData_t *threadData);


/*
equation index: 6095
type: SIMPLE_ASSIGN
vz[381] = 0.0
*/
void SpiralGalaxy_eqFunction_6095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6095};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1380]] /* vz[381] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11801(DATA *data, threadData_t *threadData);


/*
equation index: 6097
type: SIMPLE_ASSIGN
z[382] = 0.02112
*/
void SpiralGalaxy_eqFunction_6097(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6097};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2881]] /* z[382] STATE(1,vz[382]) */) = 0.02112;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11814(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11815(DATA *data, threadData_t *threadData);


/*
equation index: 6100
type: SIMPLE_ASSIGN
y[382] = r_init[382] * sin(theta[382] + 0.00528)
*/
void SpiralGalaxy_eqFunction_6100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6100};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2381]] /* y[382] STATE(1,vy[382]) */) = ((data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */)) * (sin((data->simulationInfo->realParameter[1888] /* theta[382] PARAM */) + 0.00528));
  TRACE_POP
}

/*
equation index: 6101
type: SIMPLE_ASSIGN
x[382] = r_init[382] * cos(theta[382] + 0.00528)
*/
void SpiralGalaxy_eqFunction_6101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1881]] /* x[382] STATE(1,vx[382]) */) = ((data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */)) * (cos((data->simulationInfo->realParameter[1888] /* theta[382] PARAM */) + 0.00528));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11816(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11817(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11820(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11819(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11818(DATA *data, threadData_t *threadData);


/*
equation index: 6107
type: SIMPLE_ASSIGN
vx[382] = (-sin(theta[382])) * r_init[382] * omega_c[382]
*/
void SpiralGalaxy_eqFunction_6107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6107};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[381]] /* vx[382] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1888] /* theta[382] PARAM */)))) * (((data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */)) * ((data->simulationInfo->realParameter[886] /* omega_c[382] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11813(DATA *data, threadData_t *threadData);


/*
equation index: 6109
type: SIMPLE_ASSIGN
vy[382] = cos(theta[382]) * r_init[382] * omega_c[382]
*/
void SpiralGalaxy_eqFunction_6109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[881]] /* vy[382] STATE(1) */) = (cos((data->simulationInfo->realParameter[1888] /* theta[382] PARAM */))) * (((data->simulationInfo->realParameter[1387] /* r_init[382] PARAM */)) * ((data->simulationInfo->realParameter[886] /* omega_c[382] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11812(DATA *data, threadData_t *threadData);


/*
equation index: 6111
type: SIMPLE_ASSIGN
vz[382] = 0.0
*/
void SpiralGalaxy_eqFunction_6111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6111};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1381]] /* vz[382] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11811(DATA *data, threadData_t *threadData);


/*
equation index: 6113
type: SIMPLE_ASSIGN
z[383] = 0.02128
*/
void SpiralGalaxy_eqFunction_6113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6113};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2882]] /* z[383] STATE(1,vz[383]) */) = 0.02128;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11824(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11825(DATA *data, threadData_t *threadData);


/*
equation index: 6116
type: SIMPLE_ASSIGN
y[383] = r_init[383] * sin(theta[383] + 0.00532)
*/
void SpiralGalaxy_eqFunction_6116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6116};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2382]] /* y[383] STATE(1,vy[383]) */) = ((data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */)) * (sin((data->simulationInfo->realParameter[1889] /* theta[383] PARAM */) + 0.00532));
  TRACE_POP
}

/*
equation index: 6117
type: SIMPLE_ASSIGN
x[383] = r_init[383] * cos(theta[383] + 0.00532)
*/
void SpiralGalaxy_eqFunction_6117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1882]] /* x[383] STATE(1,vx[383]) */) = ((data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */)) * (cos((data->simulationInfo->realParameter[1889] /* theta[383] PARAM */) + 0.00532));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11826(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11827(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11830(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11829(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11828(DATA *data, threadData_t *threadData);


/*
equation index: 6123
type: SIMPLE_ASSIGN
vx[383] = (-sin(theta[383])) * r_init[383] * omega_c[383]
*/
void SpiralGalaxy_eqFunction_6123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6123};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[382]] /* vx[383] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1889] /* theta[383] PARAM */)))) * (((data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */)) * ((data->simulationInfo->realParameter[887] /* omega_c[383] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11823(DATA *data, threadData_t *threadData);


/*
equation index: 6125
type: SIMPLE_ASSIGN
vy[383] = cos(theta[383]) * r_init[383] * omega_c[383]
*/
void SpiralGalaxy_eqFunction_6125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[882]] /* vy[383] STATE(1) */) = (cos((data->simulationInfo->realParameter[1889] /* theta[383] PARAM */))) * (((data->simulationInfo->realParameter[1388] /* r_init[383] PARAM */)) * ((data->simulationInfo->realParameter[887] /* omega_c[383] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11822(DATA *data, threadData_t *threadData);


/*
equation index: 6127
type: SIMPLE_ASSIGN
vz[383] = 0.0
*/
void SpiralGalaxy_eqFunction_6127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6127};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1382]] /* vz[383] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11821(DATA *data, threadData_t *threadData);


/*
equation index: 6129
type: SIMPLE_ASSIGN
z[384] = 0.02144
*/
void SpiralGalaxy_eqFunction_6129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6129};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2883]] /* z[384] STATE(1,vz[384]) */) = 0.02144;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11834(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11835(DATA *data, threadData_t *threadData);


/*
equation index: 6132
type: SIMPLE_ASSIGN
y[384] = r_init[384] * sin(theta[384] + 0.00536)
*/
void SpiralGalaxy_eqFunction_6132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6132};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2383]] /* y[384] STATE(1,vy[384]) */) = ((data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */)) * (sin((data->simulationInfo->realParameter[1890] /* theta[384] PARAM */) + 0.00536));
  TRACE_POP
}

/*
equation index: 6133
type: SIMPLE_ASSIGN
x[384] = r_init[384] * cos(theta[384] + 0.00536)
*/
void SpiralGalaxy_eqFunction_6133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1883]] /* x[384] STATE(1,vx[384]) */) = ((data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */)) * (cos((data->simulationInfo->realParameter[1890] /* theta[384] PARAM */) + 0.00536));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11836(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11837(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11840(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11839(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11838(DATA *data, threadData_t *threadData);


/*
equation index: 6139
type: SIMPLE_ASSIGN
vx[384] = (-sin(theta[384])) * r_init[384] * omega_c[384]
*/
void SpiralGalaxy_eqFunction_6139(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6139};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[383]] /* vx[384] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1890] /* theta[384] PARAM */)))) * (((data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */)) * ((data->simulationInfo->realParameter[888] /* omega_c[384] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11833(DATA *data, threadData_t *threadData);


/*
equation index: 6141
type: SIMPLE_ASSIGN
vy[384] = cos(theta[384]) * r_init[384] * omega_c[384]
*/
void SpiralGalaxy_eqFunction_6141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[883]] /* vy[384] STATE(1) */) = (cos((data->simulationInfo->realParameter[1890] /* theta[384] PARAM */))) * (((data->simulationInfo->realParameter[1389] /* r_init[384] PARAM */)) * ((data->simulationInfo->realParameter[888] /* omega_c[384] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11832(DATA *data, threadData_t *threadData);


/*
equation index: 6143
type: SIMPLE_ASSIGN
vz[384] = 0.0
*/
void SpiralGalaxy_eqFunction_6143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6143};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1383]] /* vz[384] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11831(DATA *data, threadData_t *threadData);


/*
equation index: 6145
type: SIMPLE_ASSIGN
z[385] = 0.021600000000000005
*/
void SpiralGalaxy_eqFunction_6145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6145};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2884]] /* z[385] STATE(1,vz[385]) */) = 0.021600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11844(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11845(DATA *data, threadData_t *threadData);


/*
equation index: 6148
type: SIMPLE_ASSIGN
y[385] = r_init[385] * sin(theta[385] + 0.0054)
*/
void SpiralGalaxy_eqFunction_6148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6148};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2384]] /* y[385] STATE(1,vy[385]) */) = ((data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */)) * (sin((data->simulationInfo->realParameter[1891] /* theta[385] PARAM */) + 0.0054));
  TRACE_POP
}

/*
equation index: 6149
type: SIMPLE_ASSIGN
x[385] = r_init[385] * cos(theta[385] + 0.0054)
*/
void SpiralGalaxy_eqFunction_6149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1884]] /* x[385] STATE(1,vx[385]) */) = ((data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */)) * (cos((data->simulationInfo->realParameter[1891] /* theta[385] PARAM */) + 0.0054));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11846(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11847(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11850(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11849(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11848(DATA *data, threadData_t *threadData);


/*
equation index: 6155
type: SIMPLE_ASSIGN
vx[385] = (-sin(theta[385])) * r_init[385] * omega_c[385]
*/
void SpiralGalaxy_eqFunction_6155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6155};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[384]] /* vx[385] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1891] /* theta[385] PARAM */)))) * (((data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */)) * ((data->simulationInfo->realParameter[889] /* omega_c[385] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11843(DATA *data, threadData_t *threadData);


/*
equation index: 6157
type: SIMPLE_ASSIGN
vy[385] = cos(theta[385]) * r_init[385] * omega_c[385]
*/
void SpiralGalaxy_eqFunction_6157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[884]] /* vy[385] STATE(1) */) = (cos((data->simulationInfo->realParameter[1891] /* theta[385] PARAM */))) * (((data->simulationInfo->realParameter[1390] /* r_init[385] PARAM */)) * ((data->simulationInfo->realParameter[889] /* omega_c[385] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11842(DATA *data, threadData_t *threadData);


/*
equation index: 6159
type: SIMPLE_ASSIGN
vz[385] = 0.0
*/
void SpiralGalaxy_eqFunction_6159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6159};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1384]] /* vz[385] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11841(DATA *data, threadData_t *threadData);


/*
equation index: 6161
type: SIMPLE_ASSIGN
z[386] = 0.02176
*/
void SpiralGalaxy_eqFunction_6161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6161};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2885]] /* z[386] STATE(1,vz[386]) */) = 0.02176;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11854(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11855(DATA *data, threadData_t *threadData);


/*
equation index: 6164
type: SIMPLE_ASSIGN
y[386] = r_init[386] * sin(theta[386] + 0.00544)
*/
void SpiralGalaxy_eqFunction_6164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6164};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2385]] /* y[386] STATE(1,vy[386]) */) = ((data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */)) * (sin((data->simulationInfo->realParameter[1892] /* theta[386] PARAM */) + 0.00544));
  TRACE_POP
}

/*
equation index: 6165
type: SIMPLE_ASSIGN
x[386] = r_init[386] * cos(theta[386] + 0.00544)
*/
void SpiralGalaxy_eqFunction_6165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1885]] /* x[386] STATE(1,vx[386]) */) = ((data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */)) * (cos((data->simulationInfo->realParameter[1892] /* theta[386] PARAM */) + 0.00544));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11856(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11857(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11860(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11859(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11858(DATA *data, threadData_t *threadData);


/*
equation index: 6171
type: SIMPLE_ASSIGN
vx[386] = (-sin(theta[386])) * r_init[386] * omega_c[386]
*/
void SpiralGalaxy_eqFunction_6171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6171};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[385]] /* vx[386] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1892] /* theta[386] PARAM */)))) * (((data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */)) * ((data->simulationInfo->realParameter[890] /* omega_c[386] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11853(DATA *data, threadData_t *threadData);


/*
equation index: 6173
type: SIMPLE_ASSIGN
vy[386] = cos(theta[386]) * r_init[386] * omega_c[386]
*/
void SpiralGalaxy_eqFunction_6173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[885]] /* vy[386] STATE(1) */) = (cos((data->simulationInfo->realParameter[1892] /* theta[386] PARAM */))) * (((data->simulationInfo->realParameter[1391] /* r_init[386] PARAM */)) * ((data->simulationInfo->realParameter[890] /* omega_c[386] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11852(DATA *data, threadData_t *threadData);


/*
equation index: 6175
type: SIMPLE_ASSIGN
vz[386] = 0.0
*/
void SpiralGalaxy_eqFunction_6175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6175};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1385]] /* vz[386] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11851(DATA *data, threadData_t *threadData);


/*
equation index: 6177
type: SIMPLE_ASSIGN
z[387] = 0.021920000000000002
*/
void SpiralGalaxy_eqFunction_6177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6177};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2886]] /* z[387] STATE(1,vz[387]) */) = 0.021920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11864(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11865(DATA *data, threadData_t *threadData);


/*
equation index: 6180
type: SIMPLE_ASSIGN
y[387] = r_init[387] * sin(theta[387] + 0.0054800000000000005)
*/
void SpiralGalaxy_eqFunction_6180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6180};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2386]] /* y[387] STATE(1,vy[387]) */) = ((data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */)) * (sin((data->simulationInfo->realParameter[1893] /* theta[387] PARAM */) + 0.0054800000000000005));
  TRACE_POP
}

/*
equation index: 6181
type: SIMPLE_ASSIGN
x[387] = r_init[387] * cos(theta[387] + 0.0054800000000000005)
*/
void SpiralGalaxy_eqFunction_6181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1886]] /* x[387] STATE(1,vx[387]) */) = ((data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */)) * (cos((data->simulationInfo->realParameter[1893] /* theta[387] PARAM */) + 0.0054800000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11866(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11867(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11870(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11869(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11868(DATA *data, threadData_t *threadData);


/*
equation index: 6187
type: SIMPLE_ASSIGN
vx[387] = (-sin(theta[387])) * r_init[387] * omega_c[387]
*/
void SpiralGalaxy_eqFunction_6187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6187};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[386]] /* vx[387] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1893] /* theta[387] PARAM */)))) * (((data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */)) * ((data->simulationInfo->realParameter[891] /* omega_c[387] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11863(DATA *data, threadData_t *threadData);


/*
equation index: 6189
type: SIMPLE_ASSIGN
vy[387] = cos(theta[387]) * r_init[387] * omega_c[387]
*/
void SpiralGalaxy_eqFunction_6189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[886]] /* vy[387] STATE(1) */) = (cos((data->simulationInfo->realParameter[1893] /* theta[387] PARAM */))) * (((data->simulationInfo->realParameter[1392] /* r_init[387] PARAM */)) * ((data->simulationInfo->realParameter[891] /* omega_c[387] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11862(DATA *data, threadData_t *threadData);


/*
equation index: 6191
type: SIMPLE_ASSIGN
vz[387] = 0.0
*/
void SpiralGalaxy_eqFunction_6191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6191};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1386]] /* vz[387] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11861(DATA *data, threadData_t *threadData);


/*
equation index: 6193
type: SIMPLE_ASSIGN
z[388] = 0.022080000000000002
*/
void SpiralGalaxy_eqFunction_6193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6193};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2887]] /* z[388] STATE(1,vz[388]) */) = 0.022080000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11874(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11875(DATA *data, threadData_t *threadData);


/*
equation index: 6196
type: SIMPLE_ASSIGN
y[388] = r_init[388] * sin(theta[388] + 0.005520000000000001)
*/
void SpiralGalaxy_eqFunction_6196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6196};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2387]] /* y[388] STATE(1,vy[388]) */) = ((data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */)) * (sin((data->simulationInfo->realParameter[1894] /* theta[388] PARAM */) + 0.005520000000000001));
  TRACE_POP
}

/*
equation index: 6197
type: SIMPLE_ASSIGN
x[388] = r_init[388] * cos(theta[388] + 0.005520000000000001)
*/
void SpiralGalaxy_eqFunction_6197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1887]] /* x[388] STATE(1,vx[388]) */) = ((data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */)) * (cos((data->simulationInfo->realParameter[1894] /* theta[388] PARAM */) + 0.005520000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11876(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11877(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11880(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11879(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11878(DATA *data, threadData_t *threadData);


/*
equation index: 6203
type: SIMPLE_ASSIGN
vx[388] = (-sin(theta[388])) * r_init[388] * omega_c[388]
*/
void SpiralGalaxy_eqFunction_6203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6203};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[387]] /* vx[388] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1894] /* theta[388] PARAM */)))) * (((data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */)) * ((data->simulationInfo->realParameter[892] /* omega_c[388] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11873(DATA *data, threadData_t *threadData);


/*
equation index: 6205
type: SIMPLE_ASSIGN
vy[388] = cos(theta[388]) * r_init[388] * omega_c[388]
*/
void SpiralGalaxy_eqFunction_6205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[887]] /* vy[388] STATE(1) */) = (cos((data->simulationInfo->realParameter[1894] /* theta[388] PARAM */))) * (((data->simulationInfo->realParameter[1393] /* r_init[388] PARAM */)) * ((data->simulationInfo->realParameter[892] /* omega_c[388] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11872(DATA *data, threadData_t *threadData);


/*
equation index: 6207
type: SIMPLE_ASSIGN
vz[388] = 0.0
*/
void SpiralGalaxy_eqFunction_6207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6207};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1387]] /* vz[388] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11871(DATA *data, threadData_t *threadData);


/*
equation index: 6209
type: SIMPLE_ASSIGN
z[389] = 0.022240000000000003
*/
void SpiralGalaxy_eqFunction_6209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6209};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2888]] /* z[389] STATE(1,vz[389]) */) = 0.022240000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11884(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11885(DATA *data, threadData_t *threadData);


/*
equation index: 6212
type: SIMPLE_ASSIGN
y[389] = r_init[389] * sin(theta[389] + 0.005560000000000001)
*/
void SpiralGalaxy_eqFunction_6212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6212};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2388]] /* y[389] STATE(1,vy[389]) */) = ((data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */)) * (sin((data->simulationInfo->realParameter[1895] /* theta[389] PARAM */) + 0.005560000000000001));
  TRACE_POP
}

/*
equation index: 6213
type: SIMPLE_ASSIGN
x[389] = r_init[389] * cos(theta[389] + 0.005560000000000001)
*/
void SpiralGalaxy_eqFunction_6213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1888]] /* x[389] STATE(1,vx[389]) */) = ((data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */)) * (cos((data->simulationInfo->realParameter[1895] /* theta[389] PARAM */) + 0.005560000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11886(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11887(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11890(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11889(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11888(DATA *data, threadData_t *threadData);


/*
equation index: 6219
type: SIMPLE_ASSIGN
vx[389] = (-sin(theta[389])) * r_init[389] * omega_c[389]
*/
void SpiralGalaxy_eqFunction_6219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6219};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[388]] /* vx[389] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1895] /* theta[389] PARAM */)))) * (((data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */)) * ((data->simulationInfo->realParameter[893] /* omega_c[389] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11883(DATA *data, threadData_t *threadData);


/*
equation index: 6221
type: SIMPLE_ASSIGN
vy[389] = cos(theta[389]) * r_init[389] * omega_c[389]
*/
void SpiralGalaxy_eqFunction_6221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[888]] /* vy[389] STATE(1) */) = (cos((data->simulationInfo->realParameter[1895] /* theta[389] PARAM */))) * (((data->simulationInfo->realParameter[1394] /* r_init[389] PARAM */)) * ((data->simulationInfo->realParameter[893] /* omega_c[389] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11882(DATA *data, threadData_t *threadData);


/*
equation index: 6223
type: SIMPLE_ASSIGN
vz[389] = 0.0
*/
void SpiralGalaxy_eqFunction_6223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6223};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1388]] /* vz[389] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11881(DATA *data, threadData_t *threadData);


/*
equation index: 6225
type: SIMPLE_ASSIGN
z[390] = 0.022400000000000003
*/
void SpiralGalaxy_eqFunction_6225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6225};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2889]] /* z[390] STATE(1,vz[390]) */) = 0.022400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11894(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11895(DATA *data, threadData_t *threadData);


/*
equation index: 6228
type: SIMPLE_ASSIGN
y[390] = r_init[390] * sin(theta[390] + 0.005600000000000001)
*/
void SpiralGalaxy_eqFunction_6228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6228};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2389]] /* y[390] STATE(1,vy[390]) */) = ((data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */)) * (sin((data->simulationInfo->realParameter[1896] /* theta[390] PARAM */) + 0.005600000000000001));
  TRACE_POP
}

/*
equation index: 6229
type: SIMPLE_ASSIGN
x[390] = r_init[390] * cos(theta[390] + 0.005600000000000001)
*/
void SpiralGalaxy_eqFunction_6229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1889]] /* x[390] STATE(1,vx[390]) */) = ((data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */)) * (cos((data->simulationInfo->realParameter[1896] /* theta[390] PARAM */) + 0.005600000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11896(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11897(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11900(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11899(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11898(DATA *data, threadData_t *threadData);


/*
equation index: 6235
type: SIMPLE_ASSIGN
vx[390] = (-sin(theta[390])) * r_init[390] * omega_c[390]
*/
void SpiralGalaxy_eqFunction_6235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6235};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[389]] /* vx[390] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1896] /* theta[390] PARAM */)))) * (((data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */)) * ((data->simulationInfo->realParameter[894] /* omega_c[390] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11893(DATA *data, threadData_t *threadData);


/*
equation index: 6237
type: SIMPLE_ASSIGN
vy[390] = cos(theta[390]) * r_init[390] * omega_c[390]
*/
void SpiralGalaxy_eqFunction_6237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[889]] /* vy[390] STATE(1) */) = (cos((data->simulationInfo->realParameter[1896] /* theta[390] PARAM */))) * (((data->simulationInfo->realParameter[1395] /* r_init[390] PARAM */)) * ((data->simulationInfo->realParameter[894] /* omega_c[390] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11892(DATA *data, threadData_t *threadData);


/*
equation index: 6239
type: SIMPLE_ASSIGN
vz[390] = 0.0
*/
void SpiralGalaxy_eqFunction_6239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6239};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1389]] /* vz[390] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11891(DATA *data, threadData_t *threadData);


/*
equation index: 6241
type: SIMPLE_ASSIGN
z[391] = 0.02256
*/
void SpiralGalaxy_eqFunction_6241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6241};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2890]] /* z[391] STATE(1,vz[391]) */) = 0.02256;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11904(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11905(DATA *data, threadData_t *threadData);


/*
equation index: 6244
type: SIMPLE_ASSIGN
y[391] = r_init[391] * sin(theta[391] + 0.005640000000000001)
*/
void SpiralGalaxy_eqFunction_6244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6244};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2390]] /* y[391] STATE(1,vy[391]) */) = ((data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */)) * (sin((data->simulationInfo->realParameter[1897] /* theta[391] PARAM */) + 0.005640000000000001));
  TRACE_POP
}

/*
equation index: 6245
type: SIMPLE_ASSIGN
x[391] = r_init[391] * cos(theta[391] + 0.005640000000000001)
*/
void SpiralGalaxy_eqFunction_6245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1890]] /* x[391] STATE(1,vx[391]) */) = ((data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */)) * (cos((data->simulationInfo->realParameter[1897] /* theta[391] PARAM */) + 0.005640000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11906(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11907(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11910(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11909(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11908(DATA *data, threadData_t *threadData);


/*
equation index: 6251
type: SIMPLE_ASSIGN
vx[391] = (-sin(theta[391])) * r_init[391] * omega_c[391]
*/
void SpiralGalaxy_eqFunction_6251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6251};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[390]] /* vx[391] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1897] /* theta[391] PARAM */)))) * (((data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */)) * ((data->simulationInfo->realParameter[895] /* omega_c[391] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11903(DATA *data, threadData_t *threadData);


/*
equation index: 6253
type: SIMPLE_ASSIGN
vy[391] = cos(theta[391]) * r_init[391] * omega_c[391]
*/
void SpiralGalaxy_eqFunction_6253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[890]] /* vy[391] STATE(1) */) = (cos((data->simulationInfo->realParameter[1897] /* theta[391] PARAM */))) * (((data->simulationInfo->realParameter[1396] /* r_init[391] PARAM */)) * ((data->simulationInfo->realParameter[895] /* omega_c[391] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11902(DATA *data, threadData_t *threadData);


/*
equation index: 6255
type: SIMPLE_ASSIGN
vz[391] = 0.0
*/
void SpiralGalaxy_eqFunction_6255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6255};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1390]] /* vz[391] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11901(DATA *data, threadData_t *threadData);


/*
equation index: 6257
type: SIMPLE_ASSIGN
z[392] = 0.022720000000000004
*/
void SpiralGalaxy_eqFunction_6257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6257};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2891]] /* z[392] STATE(1,vz[392]) */) = 0.022720000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11914(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11915(DATA *data, threadData_t *threadData);


/*
equation index: 6260
type: SIMPLE_ASSIGN
y[392] = r_init[392] * sin(theta[392] + 0.005680000000000001)
*/
void SpiralGalaxy_eqFunction_6260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6260};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2391]] /* y[392] STATE(1,vy[392]) */) = ((data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */)) * (sin((data->simulationInfo->realParameter[1898] /* theta[392] PARAM */) + 0.005680000000000001));
  TRACE_POP
}

/*
equation index: 6261
type: SIMPLE_ASSIGN
x[392] = r_init[392] * cos(theta[392] + 0.005680000000000001)
*/
void SpiralGalaxy_eqFunction_6261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1891]] /* x[392] STATE(1,vx[392]) */) = ((data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */)) * (cos((data->simulationInfo->realParameter[1898] /* theta[392] PARAM */) + 0.005680000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11916(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11917(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11920(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11919(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11918(DATA *data, threadData_t *threadData);


/*
equation index: 6267
type: SIMPLE_ASSIGN
vx[392] = (-sin(theta[392])) * r_init[392] * omega_c[392]
*/
void SpiralGalaxy_eqFunction_6267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6267};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[391]] /* vx[392] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1898] /* theta[392] PARAM */)))) * (((data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */)) * ((data->simulationInfo->realParameter[896] /* omega_c[392] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11913(DATA *data, threadData_t *threadData);


/*
equation index: 6269
type: SIMPLE_ASSIGN
vy[392] = cos(theta[392]) * r_init[392] * omega_c[392]
*/
void SpiralGalaxy_eqFunction_6269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[891]] /* vy[392] STATE(1) */) = (cos((data->simulationInfo->realParameter[1898] /* theta[392] PARAM */))) * (((data->simulationInfo->realParameter[1397] /* r_init[392] PARAM */)) * ((data->simulationInfo->realParameter[896] /* omega_c[392] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11912(DATA *data, threadData_t *threadData);


/*
equation index: 6271
type: SIMPLE_ASSIGN
vz[392] = 0.0
*/
void SpiralGalaxy_eqFunction_6271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1391]] /* vz[392] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11911(DATA *data, threadData_t *threadData);


/*
equation index: 6273
type: SIMPLE_ASSIGN
z[393] = 0.022880000000000005
*/
void SpiralGalaxy_eqFunction_6273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2892]] /* z[393] STATE(1,vz[393]) */) = 0.022880000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11924(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11925(DATA *data, threadData_t *threadData);


/*
equation index: 6276
type: SIMPLE_ASSIGN
y[393] = r_init[393] * sin(theta[393] + 0.005720000000000001)
*/
void SpiralGalaxy_eqFunction_6276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6276};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2392]] /* y[393] STATE(1,vy[393]) */) = ((data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */)) * (sin((data->simulationInfo->realParameter[1899] /* theta[393] PARAM */) + 0.005720000000000001));
  TRACE_POP
}

/*
equation index: 6277
type: SIMPLE_ASSIGN
x[393] = r_init[393] * cos(theta[393] + 0.005720000000000001)
*/
void SpiralGalaxy_eqFunction_6277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1892]] /* x[393] STATE(1,vx[393]) */) = ((data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */)) * (cos((data->simulationInfo->realParameter[1899] /* theta[393] PARAM */) + 0.005720000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11926(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11927(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11930(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11929(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11928(DATA *data, threadData_t *threadData);


/*
equation index: 6283
type: SIMPLE_ASSIGN
vx[393] = (-sin(theta[393])) * r_init[393] * omega_c[393]
*/
void SpiralGalaxy_eqFunction_6283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6283};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[392]] /* vx[393] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1899] /* theta[393] PARAM */)))) * (((data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */)) * ((data->simulationInfo->realParameter[897] /* omega_c[393] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11923(DATA *data, threadData_t *threadData);


/*
equation index: 6285
type: SIMPLE_ASSIGN
vy[393] = cos(theta[393]) * r_init[393] * omega_c[393]
*/
void SpiralGalaxy_eqFunction_6285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[892]] /* vy[393] STATE(1) */) = (cos((data->simulationInfo->realParameter[1899] /* theta[393] PARAM */))) * (((data->simulationInfo->realParameter[1398] /* r_init[393] PARAM */)) * ((data->simulationInfo->realParameter[897] /* omega_c[393] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11922(DATA *data, threadData_t *threadData);


/*
equation index: 6287
type: SIMPLE_ASSIGN
vz[393] = 0.0
*/
void SpiralGalaxy_eqFunction_6287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6287};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1392]] /* vz[393] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11921(DATA *data, threadData_t *threadData);


/*
equation index: 6289
type: SIMPLE_ASSIGN
z[394] = 0.023040000000000005
*/
void SpiralGalaxy_eqFunction_6289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6289};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2893]] /* z[394] STATE(1,vz[394]) */) = 0.023040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11934(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11935(DATA *data, threadData_t *threadData);


/*
equation index: 6292
type: SIMPLE_ASSIGN
y[394] = r_init[394] * sin(theta[394] + 0.00576)
*/
void SpiralGalaxy_eqFunction_6292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6292};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2393]] /* y[394] STATE(1,vy[394]) */) = ((data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */)) * (sin((data->simulationInfo->realParameter[1900] /* theta[394] PARAM */) + 0.00576));
  TRACE_POP
}

/*
equation index: 6293
type: SIMPLE_ASSIGN
x[394] = r_init[394] * cos(theta[394] + 0.00576)
*/
void SpiralGalaxy_eqFunction_6293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1893]] /* x[394] STATE(1,vx[394]) */) = ((data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */)) * (cos((data->simulationInfo->realParameter[1900] /* theta[394] PARAM */) + 0.00576));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11936(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11937(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11940(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11939(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11938(DATA *data, threadData_t *threadData);


/*
equation index: 6299
type: SIMPLE_ASSIGN
vx[394] = (-sin(theta[394])) * r_init[394] * omega_c[394]
*/
void SpiralGalaxy_eqFunction_6299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6299};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[393]] /* vx[394] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1900] /* theta[394] PARAM */)))) * (((data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */)) * ((data->simulationInfo->realParameter[898] /* omega_c[394] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11933(DATA *data, threadData_t *threadData);


/*
equation index: 6301
type: SIMPLE_ASSIGN
vy[394] = cos(theta[394]) * r_init[394] * omega_c[394]
*/
void SpiralGalaxy_eqFunction_6301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[893]] /* vy[394] STATE(1) */) = (cos((data->simulationInfo->realParameter[1900] /* theta[394] PARAM */))) * (((data->simulationInfo->realParameter[1399] /* r_init[394] PARAM */)) * ((data->simulationInfo->realParameter[898] /* omega_c[394] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11932(DATA *data, threadData_t *threadData);


/*
equation index: 6303
type: SIMPLE_ASSIGN
vz[394] = 0.0
*/
void SpiralGalaxy_eqFunction_6303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6303};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1393]] /* vz[394] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11931(DATA *data, threadData_t *threadData);


/*
equation index: 6305
type: SIMPLE_ASSIGN
z[395] = 0.023200000000000002
*/
void SpiralGalaxy_eqFunction_6305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6305};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2894]] /* z[395] STATE(1,vz[395]) */) = 0.023200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11944(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11945(DATA *data, threadData_t *threadData);


/*
equation index: 6308
type: SIMPLE_ASSIGN
y[395] = r_init[395] * sin(theta[395] + 0.0058000000000000005)
*/
void SpiralGalaxy_eqFunction_6308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6308};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2394]] /* y[395] STATE(1,vy[395]) */) = ((data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */)) * (sin((data->simulationInfo->realParameter[1901] /* theta[395] PARAM */) + 0.0058000000000000005));
  TRACE_POP
}

/*
equation index: 6309
type: SIMPLE_ASSIGN
x[395] = r_init[395] * cos(theta[395] + 0.0058000000000000005)
*/
void SpiralGalaxy_eqFunction_6309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1894]] /* x[395] STATE(1,vx[395]) */) = ((data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */)) * (cos((data->simulationInfo->realParameter[1901] /* theta[395] PARAM */) + 0.0058000000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11946(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11947(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11950(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11949(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11948(DATA *data, threadData_t *threadData);


/*
equation index: 6315
type: SIMPLE_ASSIGN
vx[395] = (-sin(theta[395])) * r_init[395] * omega_c[395]
*/
void SpiralGalaxy_eqFunction_6315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6315};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[394]] /* vx[395] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1901] /* theta[395] PARAM */)))) * (((data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */)) * ((data->simulationInfo->realParameter[899] /* omega_c[395] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11943(DATA *data, threadData_t *threadData);


/*
equation index: 6317
type: SIMPLE_ASSIGN
vy[395] = cos(theta[395]) * r_init[395] * omega_c[395]
*/
void SpiralGalaxy_eqFunction_6317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[894]] /* vy[395] STATE(1) */) = (cos((data->simulationInfo->realParameter[1901] /* theta[395] PARAM */))) * (((data->simulationInfo->realParameter[1400] /* r_init[395] PARAM */)) * ((data->simulationInfo->realParameter[899] /* omega_c[395] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11942(DATA *data, threadData_t *threadData);


/*
equation index: 6319
type: SIMPLE_ASSIGN
vz[395] = 0.0
*/
void SpiralGalaxy_eqFunction_6319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1394]] /* vz[395] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11941(DATA *data, threadData_t *threadData);


/*
equation index: 6321
type: SIMPLE_ASSIGN
z[396] = 0.02336
*/
void SpiralGalaxy_eqFunction_6321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2895]] /* z[396] STATE(1,vz[396]) */) = 0.02336;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11954(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11955(DATA *data, threadData_t *threadData);


/*
equation index: 6324
type: SIMPLE_ASSIGN
y[396] = r_init[396] * sin(theta[396] + 0.005840000000000001)
*/
void SpiralGalaxy_eqFunction_6324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6324};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2395]] /* y[396] STATE(1,vy[396]) */) = ((data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */)) * (sin((data->simulationInfo->realParameter[1902] /* theta[396] PARAM */) + 0.005840000000000001));
  TRACE_POP
}

/*
equation index: 6325
type: SIMPLE_ASSIGN
x[396] = r_init[396] * cos(theta[396] + 0.005840000000000001)
*/
void SpiralGalaxy_eqFunction_6325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1895]] /* x[396] STATE(1,vx[396]) */) = ((data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */)) * (cos((data->simulationInfo->realParameter[1902] /* theta[396] PARAM */) + 0.005840000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11956(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11957(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11960(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11959(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11958(DATA *data, threadData_t *threadData);


/*
equation index: 6331
type: SIMPLE_ASSIGN
vx[396] = (-sin(theta[396])) * r_init[396] * omega_c[396]
*/
void SpiralGalaxy_eqFunction_6331(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6331};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[395]] /* vx[396] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1902] /* theta[396] PARAM */)))) * (((data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */)) * ((data->simulationInfo->realParameter[900] /* omega_c[396] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11953(DATA *data, threadData_t *threadData);


/*
equation index: 6333
type: SIMPLE_ASSIGN
vy[396] = cos(theta[396]) * r_init[396] * omega_c[396]
*/
void SpiralGalaxy_eqFunction_6333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[895]] /* vy[396] STATE(1) */) = (cos((data->simulationInfo->realParameter[1902] /* theta[396] PARAM */))) * (((data->simulationInfo->realParameter[1401] /* r_init[396] PARAM */)) * ((data->simulationInfo->realParameter[900] /* omega_c[396] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11952(DATA *data, threadData_t *threadData);


/*
equation index: 6335
type: SIMPLE_ASSIGN
vz[396] = 0.0
*/
void SpiralGalaxy_eqFunction_6335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1395]] /* vz[396] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11951(DATA *data, threadData_t *threadData);


/*
equation index: 6337
type: SIMPLE_ASSIGN
z[397] = 0.023520000000000003
*/
void SpiralGalaxy_eqFunction_6337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2896]] /* z[397] STATE(1,vz[397]) */) = 0.023520000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11964(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11965(DATA *data, threadData_t *threadData);


/*
equation index: 6340
type: SIMPLE_ASSIGN
y[397] = r_init[397] * sin(theta[397] + 0.005880000000000001)
*/
void SpiralGalaxy_eqFunction_6340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6340};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2396]] /* y[397] STATE(1,vy[397]) */) = ((data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */)) * (sin((data->simulationInfo->realParameter[1903] /* theta[397] PARAM */) + 0.005880000000000001));
  TRACE_POP
}

/*
equation index: 6341
type: SIMPLE_ASSIGN
x[397] = r_init[397] * cos(theta[397] + 0.005880000000000001)
*/
void SpiralGalaxy_eqFunction_6341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1896]] /* x[397] STATE(1,vx[397]) */) = ((data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */)) * (cos((data->simulationInfo->realParameter[1903] /* theta[397] PARAM */) + 0.005880000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11966(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11967(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11970(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11969(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11968(DATA *data, threadData_t *threadData);


/*
equation index: 6347
type: SIMPLE_ASSIGN
vx[397] = (-sin(theta[397])) * r_init[397] * omega_c[397]
*/
void SpiralGalaxy_eqFunction_6347(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6347};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[396]] /* vx[397] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1903] /* theta[397] PARAM */)))) * (((data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */)) * ((data->simulationInfo->realParameter[901] /* omega_c[397] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11963(DATA *data, threadData_t *threadData);


/*
equation index: 6349
type: SIMPLE_ASSIGN
vy[397] = cos(theta[397]) * r_init[397] * omega_c[397]
*/
void SpiralGalaxy_eqFunction_6349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[896]] /* vy[397] STATE(1) */) = (cos((data->simulationInfo->realParameter[1903] /* theta[397] PARAM */))) * (((data->simulationInfo->realParameter[1402] /* r_init[397] PARAM */)) * ((data->simulationInfo->realParameter[901] /* omega_c[397] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11962(DATA *data, threadData_t *threadData);


/*
equation index: 6351
type: SIMPLE_ASSIGN
vz[397] = 0.0
*/
void SpiralGalaxy_eqFunction_6351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6351};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1396]] /* vz[397] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11961(DATA *data, threadData_t *threadData);


/*
equation index: 6353
type: SIMPLE_ASSIGN
z[398] = 0.023680000000000003
*/
void SpiralGalaxy_eqFunction_6353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6353};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2897]] /* z[398] STATE(1,vz[398]) */) = 0.023680000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11974(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11975(DATA *data, threadData_t *threadData);


/*
equation index: 6356
type: SIMPLE_ASSIGN
y[398] = r_init[398] * sin(theta[398] + 0.005920000000000001)
*/
void SpiralGalaxy_eqFunction_6356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6356};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2397]] /* y[398] STATE(1,vy[398]) */) = ((data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */)) * (sin((data->simulationInfo->realParameter[1904] /* theta[398] PARAM */) + 0.005920000000000001));
  TRACE_POP
}

/*
equation index: 6357
type: SIMPLE_ASSIGN
x[398] = r_init[398] * cos(theta[398] + 0.005920000000000001)
*/
void SpiralGalaxy_eqFunction_6357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1897]] /* x[398] STATE(1,vx[398]) */) = ((data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */)) * (cos((data->simulationInfo->realParameter[1904] /* theta[398] PARAM */) + 0.005920000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11976(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11977(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11980(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11979(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11978(DATA *data, threadData_t *threadData);


/*
equation index: 6363
type: SIMPLE_ASSIGN
vx[398] = (-sin(theta[398])) * r_init[398] * omega_c[398]
*/
void SpiralGalaxy_eqFunction_6363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6363};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[397]] /* vx[398] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1904] /* theta[398] PARAM */)))) * (((data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */)) * ((data->simulationInfo->realParameter[902] /* omega_c[398] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11973(DATA *data, threadData_t *threadData);


/*
equation index: 6365
type: SIMPLE_ASSIGN
vy[398] = cos(theta[398]) * r_init[398] * omega_c[398]
*/
void SpiralGalaxy_eqFunction_6365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[897]] /* vy[398] STATE(1) */) = (cos((data->simulationInfo->realParameter[1904] /* theta[398] PARAM */))) * (((data->simulationInfo->realParameter[1403] /* r_init[398] PARAM */)) * ((data->simulationInfo->realParameter[902] /* omega_c[398] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11972(DATA *data, threadData_t *threadData);


/*
equation index: 6367
type: SIMPLE_ASSIGN
vz[398] = 0.0
*/
void SpiralGalaxy_eqFunction_6367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6367};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1397]] /* vz[398] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11971(DATA *data, threadData_t *threadData);


/*
equation index: 6369
type: SIMPLE_ASSIGN
z[399] = 0.023840000000000004
*/
void SpiralGalaxy_eqFunction_6369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6369};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2898]] /* z[399] STATE(1,vz[399]) */) = 0.023840000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11984(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11985(DATA *data, threadData_t *threadData);


/*
equation index: 6372
type: SIMPLE_ASSIGN
y[399] = r_init[399] * sin(theta[399] + 0.005960000000000001)
*/
void SpiralGalaxy_eqFunction_6372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6372};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2398]] /* y[399] STATE(1,vy[399]) */) = ((data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */)) * (sin((data->simulationInfo->realParameter[1905] /* theta[399] PARAM */) + 0.005960000000000001));
  TRACE_POP
}

/*
equation index: 6373
type: SIMPLE_ASSIGN
x[399] = r_init[399] * cos(theta[399] + 0.005960000000000001)
*/
void SpiralGalaxy_eqFunction_6373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1898]] /* x[399] STATE(1,vx[399]) */) = ((data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */)) * (cos((data->simulationInfo->realParameter[1905] /* theta[399] PARAM */) + 0.005960000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11986(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11987(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11990(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11989(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11988(DATA *data, threadData_t *threadData);


/*
equation index: 6379
type: SIMPLE_ASSIGN
vx[399] = (-sin(theta[399])) * r_init[399] * omega_c[399]
*/
void SpiralGalaxy_eqFunction_6379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6379};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[398]] /* vx[399] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1905] /* theta[399] PARAM */)))) * (((data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */)) * ((data->simulationInfo->realParameter[903] /* omega_c[399] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11983(DATA *data, threadData_t *threadData);


/*
equation index: 6381
type: SIMPLE_ASSIGN
vy[399] = cos(theta[399]) * r_init[399] * omega_c[399]
*/
void SpiralGalaxy_eqFunction_6381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[898]] /* vy[399] STATE(1) */) = (cos((data->simulationInfo->realParameter[1905] /* theta[399] PARAM */))) * (((data->simulationInfo->realParameter[1404] /* r_init[399] PARAM */)) * ((data->simulationInfo->realParameter[903] /* omega_c[399] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11982(DATA *data, threadData_t *threadData);


/*
equation index: 6383
type: SIMPLE_ASSIGN
vz[399] = 0.0
*/
void SpiralGalaxy_eqFunction_6383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6383};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1398]] /* vz[399] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11981(DATA *data, threadData_t *threadData);


/*
equation index: 6385
type: SIMPLE_ASSIGN
z[400] = 0.024000000000000004
*/
void SpiralGalaxy_eqFunction_6385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6385};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2899]] /* z[400] STATE(1,vz[400]) */) = 0.024000000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11994(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11995(DATA *data, threadData_t *threadData);


/*
equation index: 6388
type: SIMPLE_ASSIGN
y[400] = r_init[400] * sin(theta[400] + 0.006000000000000001)
*/
void SpiralGalaxy_eqFunction_6388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6388};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2399]] /* y[400] STATE(1,vy[400]) */) = ((data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */)) * (sin((data->simulationInfo->realParameter[1906] /* theta[400] PARAM */) + 0.006000000000000001));
  TRACE_POP
}

/*
equation index: 6389
type: SIMPLE_ASSIGN
x[400] = r_init[400] * cos(theta[400] + 0.006000000000000001)
*/
void SpiralGalaxy_eqFunction_6389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1899]] /* x[400] STATE(1,vx[400]) */) = ((data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */)) * (cos((data->simulationInfo->realParameter[1906] /* theta[400] PARAM */) + 0.006000000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11996(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11997(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12000(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11999(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11998(DATA *data, threadData_t *threadData);


/*
equation index: 6395
type: SIMPLE_ASSIGN
vx[400] = (-sin(theta[400])) * r_init[400] * omega_c[400]
*/
void SpiralGalaxy_eqFunction_6395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6395};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[399]] /* vx[400] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1906] /* theta[400] PARAM */)))) * (((data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */)) * ((data->simulationInfo->realParameter[904] /* omega_c[400] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11993(DATA *data, threadData_t *threadData);


/*
equation index: 6397
type: SIMPLE_ASSIGN
vy[400] = cos(theta[400]) * r_init[400] * omega_c[400]
*/
void SpiralGalaxy_eqFunction_6397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[899]] /* vy[400] STATE(1) */) = (cos((data->simulationInfo->realParameter[1906] /* theta[400] PARAM */))) * (((data->simulationInfo->realParameter[1405] /* r_init[400] PARAM */)) * ((data->simulationInfo->realParameter[904] /* omega_c[400] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11992(DATA *data, threadData_t *threadData);


/*
equation index: 6399
type: SIMPLE_ASSIGN
vz[400] = 0.0
*/
void SpiralGalaxy_eqFunction_6399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6399};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1399]] /* vz[400] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11991(DATA *data, threadData_t *threadData);


/*
equation index: 6401
type: SIMPLE_ASSIGN
z[401] = 0.02416
*/
void SpiralGalaxy_eqFunction_6401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6401};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2900]] /* z[401] STATE(1,vz[401]) */) = 0.02416;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12004(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12005(DATA *data, threadData_t *threadData);


/*
equation index: 6404
type: SIMPLE_ASSIGN
y[401] = r_init[401] * sin(theta[401] + 0.006040000000000001)
*/
void SpiralGalaxy_eqFunction_6404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6404};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2400]] /* y[401] STATE(1,vy[401]) */) = ((data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */)) * (sin((data->simulationInfo->realParameter[1907] /* theta[401] PARAM */) + 0.006040000000000001));
  TRACE_POP
}

/*
equation index: 6405
type: SIMPLE_ASSIGN
x[401] = r_init[401] * cos(theta[401] + 0.006040000000000001)
*/
void SpiralGalaxy_eqFunction_6405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1900]] /* x[401] STATE(1,vx[401]) */) = ((data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */)) * (cos((data->simulationInfo->realParameter[1907] /* theta[401] PARAM */) + 0.006040000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12006(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12007(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12010(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12009(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12008(DATA *data, threadData_t *threadData);


/*
equation index: 6411
type: SIMPLE_ASSIGN
vx[401] = (-sin(theta[401])) * r_init[401] * omega_c[401]
*/
void SpiralGalaxy_eqFunction_6411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6411};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[400]] /* vx[401] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1907] /* theta[401] PARAM */)))) * (((data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */)) * ((data->simulationInfo->realParameter[905] /* omega_c[401] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12003(DATA *data, threadData_t *threadData);


/*
equation index: 6413
type: SIMPLE_ASSIGN
vy[401] = cos(theta[401]) * r_init[401] * omega_c[401]
*/
void SpiralGalaxy_eqFunction_6413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[900]] /* vy[401] STATE(1) */) = (cos((data->simulationInfo->realParameter[1907] /* theta[401] PARAM */))) * (((data->simulationInfo->realParameter[1406] /* r_init[401] PARAM */)) * ((data->simulationInfo->realParameter[905] /* omega_c[401] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12002(DATA *data, threadData_t *threadData);


/*
equation index: 6415
type: SIMPLE_ASSIGN
vz[401] = 0.0
*/
void SpiralGalaxy_eqFunction_6415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6415};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1400]] /* vz[401] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12001(DATA *data, threadData_t *threadData);


/*
equation index: 6417
type: SIMPLE_ASSIGN
z[402] = 0.02432
*/
void SpiralGalaxy_eqFunction_6417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6417};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2901]] /* z[402] STATE(1,vz[402]) */) = 0.02432;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12014(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12015(DATA *data, threadData_t *threadData);


/*
equation index: 6420
type: SIMPLE_ASSIGN
y[402] = r_init[402] * sin(theta[402] + 0.006080000000000001)
*/
void SpiralGalaxy_eqFunction_6420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6420};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2401]] /* y[402] STATE(1,vy[402]) */) = ((data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */)) * (sin((data->simulationInfo->realParameter[1908] /* theta[402] PARAM */) + 0.006080000000000001));
  TRACE_POP
}

/*
equation index: 6421
type: SIMPLE_ASSIGN
x[402] = r_init[402] * cos(theta[402] + 0.006080000000000001)
*/
void SpiralGalaxy_eqFunction_6421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1901]] /* x[402] STATE(1,vx[402]) */) = ((data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */)) * (cos((data->simulationInfo->realParameter[1908] /* theta[402] PARAM */) + 0.006080000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12016(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12017(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12020(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12019(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12018(DATA *data, threadData_t *threadData);


/*
equation index: 6427
type: SIMPLE_ASSIGN
vx[402] = (-sin(theta[402])) * r_init[402] * omega_c[402]
*/
void SpiralGalaxy_eqFunction_6427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6427};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[401]] /* vx[402] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1908] /* theta[402] PARAM */)))) * (((data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */)) * ((data->simulationInfo->realParameter[906] /* omega_c[402] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12013(DATA *data, threadData_t *threadData);


/*
equation index: 6429
type: SIMPLE_ASSIGN
vy[402] = cos(theta[402]) * r_init[402] * omega_c[402]
*/
void SpiralGalaxy_eqFunction_6429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[901]] /* vy[402] STATE(1) */) = (cos((data->simulationInfo->realParameter[1908] /* theta[402] PARAM */))) * (((data->simulationInfo->realParameter[1407] /* r_init[402] PARAM */)) * ((data->simulationInfo->realParameter[906] /* omega_c[402] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12012(DATA *data, threadData_t *threadData);


/*
equation index: 6431
type: SIMPLE_ASSIGN
vz[402] = 0.0
*/
void SpiralGalaxy_eqFunction_6431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6431};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1401]] /* vz[402] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12011(DATA *data, threadData_t *threadData);


/*
equation index: 6433
type: SIMPLE_ASSIGN
z[403] = 0.024480000000000002
*/
void SpiralGalaxy_eqFunction_6433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6433};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2902]] /* z[403] STATE(1,vz[403]) */) = 0.024480000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12024(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12025(DATA *data, threadData_t *threadData);


/*
equation index: 6436
type: SIMPLE_ASSIGN
y[403] = r_init[403] * sin(theta[403] + 0.006120000000000001)
*/
void SpiralGalaxy_eqFunction_6436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6436};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2402]] /* y[403] STATE(1,vy[403]) */) = ((data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */)) * (sin((data->simulationInfo->realParameter[1909] /* theta[403] PARAM */) + 0.006120000000000001));
  TRACE_POP
}

/*
equation index: 6437
type: SIMPLE_ASSIGN
x[403] = r_init[403] * cos(theta[403] + 0.006120000000000001)
*/
void SpiralGalaxy_eqFunction_6437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1902]] /* x[403] STATE(1,vx[403]) */) = ((data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */)) * (cos((data->simulationInfo->realParameter[1909] /* theta[403] PARAM */) + 0.006120000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12026(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12027(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12030(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12029(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12028(DATA *data, threadData_t *threadData);


/*
equation index: 6443
type: SIMPLE_ASSIGN
vx[403] = (-sin(theta[403])) * r_init[403] * omega_c[403]
*/
void SpiralGalaxy_eqFunction_6443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6443};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[402]] /* vx[403] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1909] /* theta[403] PARAM */)))) * (((data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */)) * ((data->simulationInfo->realParameter[907] /* omega_c[403] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12023(DATA *data, threadData_t *threadData);


/*
equation index: 6445
type: SIMPLE_ASSIGN
vy[403] = cos(theta[403]) * r_init[403] * omega_c[403]
*/
void SpiralGalaxy_eqFunction_6445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[902]] /* vy[403] STATE(1) */) = (cos((data->simulationInfo->realParameter[1909] /* theta[403] PARAM */))) * (((data->simulationInfo->realParameter[1408] /* r_init[403] PARAM */)) * ((data->simulationInfo->realParameter[907] /* omega_c[403] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12022(DATA *data, threadData_t *threadData);


/*
equation index: 6447
type: SIMPLE_ASSIGN
vz[403] = 0.0
*/
void SpiralGalaxy_eqFunction_6447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6447};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1402]] /* vz[403] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12021(DATA *data, threadData_t *threadData);


/*
equation index: 6449
type: SIMPLE_ASSIGN
z[404] = 0.024640000000000002
*/
void SpiralGalaxy_eqFunction_6449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6449};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2903]] /* z[404] STATE(1,vz[404]) */) = 0.024640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12034(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12035(DATA *data, threadData_t *threadData);


/*
equation index: 6452
type: SIMPLE_ASSIGN
y[404] = r_init[404] * sin(theta[404] + 0.006160000000000001)
*/
void SpiralGalaxy_eqFunction_6452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6452};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2403]] /* y[404] STATE(1,vy[404]) */) = ((data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */)) * (sin((data->simulationInfo->realParameter[1910] /* theta[404] PARAM */) + 0.006160000000000001));
  TRACE_POP
}

/*
equation index: 6453
type: SIMPLE_ASSIGN
x[404] = r_init[404] * cos(theta[404] + 0.006160000000000001)
*/
void SpiralGalaxy_eqFunction_6453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1903]] /* x[404] STATE(1,vx[404]) */) = ((data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */)) * (cos((data->simulationInfo->realParameter[1910] /* theta[404] PARAM */) + 0.006160000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12036(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12037(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12040(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12039(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12038(DATA *data, threadData_t *threadData);


/*
equation index: 6459
type: SIMPLE_ASSIGN
vx[404] = (-sin(theta[404])) * r_init[404] * omega_c[404]
*/
void SpiralGalaxy_eqFunction_6459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6459};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[403]] /* vx[404] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1910] /* theta[404] PARAM */)))) * (((data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */)) * ((data->simulationInfo->realParameter[908] /* omega_c[404] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12033(DATA *data, threadData_t *threadData);


/*
equation index: 6461
type: SIMPLE_ASSIGN
vy[404] = cos(theta[404]) * r_init[404] * omega_c[404]
*/
void SpiralGalaxy_eqFunction_6461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[903]] /* vy[404] STATE(1) */) = (cos((data->simulationInfo->realParameter[1910] /* theta[404] PARAM */))) * (((data->simulationInfo->realParameter[1409] /* r_init[404] PARAM */)) * ((data->simulationInfo->realParameter[908] /* omega_c[404] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12032(DATA *data, threadData_t *threadData);


/*
equation index: 6463
type: SIMPLE_ASSIGN
vz[404] = 0.0
*/
void SpiralGalaxy_eqFunction_6463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6463};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1403]] /* vz[404] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12031(DATA *data, threadData_t *threadData);


/*
equation index: 6465
type: SIMPLE_ASSIGN
z[405] = 0.024800000000000006
*/
void SpiralGalaxy_eqFunction_6465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6465};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2904]] /* z[405] STATE(1,vz[405]) */) = 0.024800000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12044(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12045(DATA *data, threadData_t *threadData);


/*
equation index: 6468
type: SIMPLE_ASSIGN
y[405] = r_init[405] * sin(theta[405] + 0.0062000000000000015)
*/
void SpiralGalaxy_eqFunction_6468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6468};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2404]] /* y[405] STATE(1,vy[405]) */) = ((data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */)) * (sin((data->simulationInfo->realParameter[1911] /* theta[405] PARAM */) + 0.0062000000000000015));
  TRACE_POP
}

/*
equation index: 6469
type: SIMPLE_ASSIGN
x[405] = r_init[405] * cos(theta[405] + 0.0062000000000000015)
*/
void SpiralGalaxy_eqFunction_6469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1904]] /* x[405] STATE(1,vx[405]) */) = ((data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */)) * (cos((data->simulationInfo->realParameter[1911] /* theta[405] PARAM */) + 0.0062000000000000015));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12046(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12047(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12050(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12049(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12048(DATA *data, threadData_t *threadData);


/*
equation index: 6475
type: SIMPLE_ASSIGN
vx[405] = (-sin(theta[405])) * r_init[405] * omega_c[405]
*/
void SpiralGalaxy_eqFunction_6475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6475};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[404]] /* vx[405] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1911] /* theta[405] PARAM */)))) * (((data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */)) * ((data->simulationInfo->realParameter[909] /* omega_c[405] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12043(DATA *data, threadData_t *threadData);


/*
equation index: 6477
type: SIMPLE_ASSIGN
vy[405] = cos(theta[405]) * r_init[405] * omega_c[405]
*/
void SpiralGalaxy_eqFunction_6477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[904]] /* vy[405] STATE(1) */) = (cos((data->simulationInfo->realParameter[1911] /* theta[405] PARAM */))) * (((data->simulationInfo->realParameter[1410] /* r_init[405] PARAM */)) * ((data->simulationInfo->realParameter[909] /* omega_c[405] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12042(DATA *data, threadData_t *threadData);


/*
equation index: 6479
type: SIMPLE_ASSIGN
vz[405] = 0.0
*/
void SpiralGalaxy_eqFunction_6479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6479};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1404]] /* vz[405] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12041(DATA *data, threadData_t *threadData);


/*
equation index: 6481
type: SIMPLE_ASSIGN
z[406] = 0.024960000000000003
*/
void SpiralGalaxy_eqFunction_6481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6481};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2905]] /* z[406] STATE(1,vz[406]) */) = 0.024960000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12054(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12055(DATA *data, threadData_t *threadData);


/*
equation index: 6484
type: SIMPLE_ASSIGN
y[406] = r_init[406] * sin(theta[406] + 0.006240000000000002)
*/
void SpiralGalaxy_eqFunction_6484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6484};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2405]] /* y[406] STATE(1,vy[406]) */) = ((data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */)) * (sin((data->simulationInfo->realParameter[1912] /* theta[406] PARAM */) + 0.006240000000000002));
  TRACE_POP
}

/*
equation index: 6485
type: SIMPLE_ASSIGN
x[406] = r_init[406] * cos(theta[406] + 0.006240000000000002)
*/
void SpiralGalaxy_eqFunction_6485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1905]] /* x[406] STATE(1,vx[406]) */) = ((data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */)) * (cos((data->simulationInfo->realParameter[1912] /* theta[406] PARAM */) + 0.006240000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12056(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12057(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12060(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12059(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12058(DATA *data, threadData_t *threadData);


/*
equation index: 6491
type: SIMPLE_ASSIGN
vx[406] = (-sin(theta[406])) * r_init[406] * omega_c[406]
*/
void SpiralGalaxy_eqFunction_6491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6491};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[405]] /* vx[406] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1912] /* theta[406] PARAM */)))) * (((data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */)) * ((data->simulationInfo->realParameter[910] /* omega_c[406] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12053(DATA *data, threadData_t *threadData);


/*
equation index: 6493
type: SIMPLE_ASSIGN
vy[406] = cos(theta[406]) * r_init[406] * omega_c[406]
*/
void SpiralGalaxy_eqFunction_6493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[905]] /* vy[406] STATE(1) */) = (cos((data->simulationInfo->realParameter[1912] /* theta[406] PARAM */))) * (((data->simulationInfo->realParameter[1411] /* r_init[406] PARAM */)) * ((data->simulationInfo->realParameter[910] /* omega_c[406] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12052(DATA *data, threadData_t *threadData);


/*
equation index: 6495
type: SIMPLE_ASSIGN
vz[406] = 0.0
*/
void SpiralGalaxy_eqFunction_6495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6495};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1405]] /* vz[406] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12051(DATA *data, threadData_t *threadData);


/*
equation index: 6497
type: SIMPLE_ASSIGN
z[407] = 0.02512
*/
void SpiralGalaxy_eqFunction_6497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6497};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2906]] /* z[407] STATE(1,vz[407]) */) = 0.02512;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_12064(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_12065(DATA *data, threadData_t *threadData);


/*
equation index: 6500
type: SIMPLE_ASSIGN
y[407] = r_init[407] * sin(theta[407] + 0.006279999999999999)
*/
void SpiralGalaxy_eqFunction_6500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6500};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2406]] /* y[407] STATE(1,vy[407]) */) = ((data->simulationInfo->realParameter[1412] /* r_init[407] PARAM */)) * (sin((data->simulationInfo->realParameter[1913] /* theta[407] PARAM */) + 0.006279999999999999));
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_12(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_6001(data, threadData);
  SpiralGalaxy_eqFunction_11754(data, threadData);
  SpiralGalaxy_eqFunction_11755(data, threadData);
  SpiralGalaxy_eqFunction_6004(data, threadData);
  SpiralGalaxy_eqFunction_6005(data, threadData);
  SpiralGalaxy_eqFunction_11756(data, threadData);
  SpiralGalaxy_eqFunction_11757(data, threadData);
  SpiralGalaxy_eqFunction_11760(data, threadData);
  SpiralGalaxy_eqFunction_11759(data, threadData);
  SpiralGalaxy_eqFunction_11758(data, threadData);
  SpiralGalaxy_eqFunction_6011(data, threadData);
  SpiralGalaxy_eqFunction_11753(data, threadData);
  SpiralGalaxy_eqFunction_6013(data, threadData);
  SpiralGalaxy_eqFunction_11752(data, threadData);
  SpiralGalaxy_eqFunction_6015(data, threadData);
  SpiralGalaxy_eqFunction_11751(data, threadData);
  SpiralGalaxy_eqFunction_6017(data, threadData);
  SpiralGalaxy_eqFunction_11764(data, threadData);
  SpiralGalaxy_eqFunction_11765(data, threadData);
  SpiralGalaxy_eqFunction_6020(data, threadData);
  SpiralGalaxy_eqFunction_6021(data, threadData);
  SpiralGalaxy_eqFunction_11766(data, threadData);
  SpiralGalaxy_eqFunction_11767(data, threadData);
  SpiralGalaxy_eqFunction_11770(data, threadData);
  SpiralGalaxy_eqFunction_11769(data, threadData);
  SpiralGalaxy_eqFunction_11768(data, threadData);
  SpiralGalaxy_eqFunction_6027(data, threadData);
  SpiralGalaxy_eqFunction_11763(data, threadData);
  SpiralGalaxy_eqFunction_6029(data, threadData);
  SpiralGalaxy_eqFunction_11762(data, threadData);
  SpiralGalaxy_eqFunction_6031(data, threadData);
  SpiralGalaxy_eqFunction_11761(data, threadData);
  SpiralGalaxy_eqFunction_6033(data, threadData);
  SpiralGalaxy_eqFunction_11774(data, threadData);
  SpiralGalaxy_eqFunction_11775(data, threadData);
  SpiralGalaxy_eqFunction_6036(data, threadData);
  SpiralGalaxy_eqFunction_6037(data, threadData);
  SpiralGalaxy_eqFunction_11776(data, threadData);
  SpiralGalaxy_eqFunction_11777(data, threadData);
  SpiralGalaxy_eqFunction_11780(data, threadData);
  SpiralGalaxy_eqFunction_11779(data, threadData);
  SpiralGalaxy_eqFunction_11778(data, threadData);
  SpiralGalaxy_eqFunction_6043(data, threadData);
  SpiralGalaxy_eqFunction_11773(data, threadData);
  SpiralGalaxy_eqFunction_6045(data, threadData);
  SpiralGalaxy_eqFunction_11772(data, threadData);
  SpiralGalaxy_eqFunction_6047(data, threadData);
  SpiralGalaxy_eqFunction_11771(data, threadData);
  SpiralGalaxy_eqFunction_6049(data, threadData);
  SpiralGalaxy_eqFunction_11784(data, threadData);
  SpiralGalaxy_eqFunction_11785(data, threadData);
  SpiralGalaxy_eqFunction_6052(data, threadData);
  SpiralGalaxy_eqFunction_6053(data, threadData);
  SpiralGalaxy_eqFunction_11786(data, threadData);
  SpiralGalaxy_eqFunction_11787(data, threadData);
  SpiralGalaxy_eqFunction_11790(data, threadData);
  SpiralGalaxy_eqFunction_11789(data, threadData);
  SpiralGalaxy_eqFunction_11788(data, threadData);
  SpiralGalaxy_eqFunction_6059(data, threadData);
  SpiralGalaxy_eqFunction_11783(data, threadData);
  SpiralGalaxy_eqFunction_6061(data, threadData);
  SpiralGalaxy_eqFunction_11782(data, threadData);
  SpiralGalaxy_eqFunction_6063(data, threadData);
  SpiralGalaxy_eqFunction_11781(data, threadData);
  SpiralGalaxy_eqFunction_6065(data, threadData);
  SpiralGalaxy_eqFunction_11794(data, threadData);
  SpiralGalaxy_eqFunction_11795(data, threadData);
  SpiralGalaxy_eqFunction_6068(data, threadData);
  SpiralGalaxy_eqFunction_6069(data, threadData);
  SpiralGalaxy_eqFunction_11796(data, threadData);
  SpiralGalaxy_eqFunction_11797(data, threadData);
  SpiralGalaxy_eqFunction_11800(data, threadData);
  SpiralGalaxy_eqFunction_11799(data, threadData);
  SpiralGalaxy_eqFunction_11798(data, threadData);
  SpiralGalaxy_eqFunction_6075(data, threadData);
  SpiralGalaxy_eqFunction_11793(data, threadData);
  SpiralGalaxy_eqFunction_6077(data, threadData);
  SpiralGalaxy_eqFunction_11792(data, threadData);
  SpiralGalaxy_eqFunction_6079(data, threadData);
  SpiralGalaxy_eqFunction_11791(data, threadData);
  SpiralGalaxy_eqFunction_6081(data, threadData);
  SpiralGalaxy_eqFunction_11804(data, threadData);
  SpiralGalaxy_eqFunction_11805(data, threadData);
  SpiralGalaxy_eqFunction_6084(data, threadData);
  SpiralGalaxy_eqFunction_6085(data, threadData);
  SpiralGalaxy_eqFunction_11806(data, threadData);
  SpiralGalaxy_eqFunction_11807(data, threadData);
  SpiralGalaxy_eqFunction_11810(data, threadData);
  SpiralGalaxy_eqFunction_11809(data, threadData);
  SpiralGalaxy_eqFunction_11808(data, threadData);
  SpiralGalaxy_eqFunction_6091(data, threadData);
  SpiralGalaxy_eqFunction_11803(data, threadData);
  SpiralGalaxy_eqFunction_6093(data, threadData);
  SpiralGalaxy_eqFunction_11802(data, threadData);
  SpiralGalaxy_eqFunction_6095(data, threadData);
  SpiralGalaxy_eqFunction_11801(data, threadData);
  SpiralGalaxy_eqFunction_6097(data, threadData);
  SpiralGalaxy_eqFunction_11814(data, threadData);
  SpiralGalaxy_eqFunction_11815(data, threadData);
  SpiralGalaxy_eqFunction_6100(data, threadData);
  SpiralGalaxy_eqFunction_6101(data, threadData);
  SpiralGalaxy_eqFunction_11816(data, threadData);
  SpiralGalaxy_eqFunction_11817(data, threadData);
  SpiralGalaxy_eqFunction_11820(data, threadData);
  SpiralGalaxy_eqFunction_11819(data, threadData);
  SpiralGalaxy_eqFunction_11818(data, threadData);
  SpiralGalaxy_eqFunction_6107(data, threadData);
  SpiralGalaxy_eqFunction_11813(data, threadData);
  SpiralGalaxy_eqFunction_6109(data, threadData);
  SpiralGalaxy_eqFunction_11812(data, threadData);
  SpiralGalaxy_eqFunction_6111(data, threadData);
  SpiralGalaxy_eqFunction_11811(data, threadData);
  SpiralGalaxy_eqFunction_6113(data, threadData);
  SpiralGalaxy_eqFunction_11824(data, threadData);
  SpiralGalaxy_eqFunction_11825(data, threadData);
  SpiralGalaxy_eqFunction_6116(data, threadData);
  SpiralGalaxy_eqFunction_6117(data, threadData);
  SpiralGalaxy_eqFunction_11826(data, threadData);
  SpiralGalaxy_eqFunction_11827(data, threadData);
  SpiralGalaxy_eqFunction_11830(data, threadData);
  SpiralGalaxy_eqFunction_11829(data, threadData);
  SpiralGalaxy_eqFunction_11828(data, threadData);
  SpiralGalaxy_eqFunction_6123(data, threadData);
  SpiralGalaxy_eqFunction_11823(data, threadData);
  SpiralGalaxy_eqFunction_6125(data, threadData);
  SpiralGalaxy_eqFunction_11822(data, threadData);
  SpiralGalaxy_eqFunction_6127(data, threadData);
  SpiralGalaxy_eqFunction_11821(data, threadData);
  SpiralGalaxy_eqFunction_6129(data, threadData);
  SpiralGalaxy_eqFunction_11834(data, threadData);
  SpiralGalaxy_eqFunction_11835(data, threadData);
  SpiralGalaxy_eqFunction_6132(data, threadData);
  SpiralGalaxy_eqFunction_6133(data, threadData);
  SpiralGalaxy_eqFunction_11836(data, threadData);
  SpiralGalaxy_eqFunction_11837(data, threadData);
  SpiralGalaxy_eqFunction_11840(data, threadData);
  SpiralGalaxy_eqFunction_11839(data, threadData);
  SpiralGalaxy_eqFunction_11838(data, threadData);
  SpiralGalaxy_eqFunction_6139(data, threadData);
  SpiralGalaxy_eqFunction_11833(data, threadData);
  SpiralGalaxy_eqFunction_6141(data, threadData);
  SpiralGalaxy_eqFunction_11832(data, threadData);
  SpiralGalaxy_eqFunction_6143(data, threadData);
  SpiralGalaxy_eqFunction_11831(data, threadData);
  SpiralGalaxy_eqFunction_6145(data, threadData);
  SpiralGalaxy_eqFunction_11844(data, threadData);
  SpiralGalaxy_eqFunction_11845(data, threadData);
  SpiralGalaxy_eqFunction_6148(data, threadData);
  SpiralGalaxy_eqFunction_6149(data, threadData);
  SpiralGalaxy_eqFunction_11846(data, threadData);
  SpiralGalaxy_eqFunction_11847(data, threadData);
  SpiralGalaxy_eqFunction_11850(data, threadData);
  SpiralGalaxy_eqFunction_11849(data, threadData);
  SpiralGalaxy_eqFunction_11848(data, threadData);
  SpiralGalaxy_eqFunction_6155(data, threadData);
  SpiralGalaxy_eqFunction_11843(data, threadData);
  SpiralGalaxy_eqFunction_6157(data, threadData);
  SpiralGalaxy_eqFunction_11842(data, threadData);
  SpiralGalaxy_eqFunction_6159(data, threadData);
  SpiralGalaxy_eqFunction_11841(data, threadData);
  SpiralGalaxy_eqFunction_6161(data, threadData);
  SpiralGalaxy_eqFunction_11854(data, threadData);
  SpiralGalaxy_eqFunction_11855(data, threadData);
  SpiralGalaxy_eqFunction_6164(data, threadData);
  SpiralGalaxy_eqFunction_6165(data, threadData);
  SpiralGalaxy_eqFunction_11856(data, threadData);
  SpiralGalaxy_eqFunction_11857(data, threadData);
  SpiralGalaxy_eqFunction_11860(data, threadData);
  SpiralGalaxy_eqFunction_11859(data, threadData);
  SpiralGalaxy_eqFunction_11858(data, threadData);
  SpiralGalaxy_eqFunction_6171(data, threadData);
  SpiralGalaxy_eqFunction_11853(data, threadData);
  SpiralGalaxy_eqFunction_6173(data, threadData);
  SpiralGalaxy_eqFunction_11852(data, threadData);
  SpiralGalaxy_eqFunction_6175(data, threadData);
  SpiralGalaxy_eqFunction_11851(data, threadData);
  SpiralGalaxy_eqFunction_6177(data, threadData);
  SpiralGalaxy_eqFunction_11864(data, threadData);
  SpiralGalaxy_eqFunction_11865(data, threadData);
  SpiralGalaxy_eqFunction_6180(data, threadData);
  SpiralGalaxy_eqFunction_6181(data, threadData);
  SpiralGalaxy_eqFunction_11866(data, threadData);
  SpiralGalaxy_eqFunction_11867(data, threadData);
  SpiralGalaxy_eqFunction_11870(data, threadData);
  SpiralGalaxy_eqFunction_11869(data, threadData);
  SpiralGalaxy_eqFunction_11868(data, threadData);
  SpiralGalaxy_eqFunction_6187(data, threadData);
  SpiralGalaxy_eqFunction_11863(data, threadData);
  SpiralGalaxy_eqFunction_6189(data, threadData);
  SpiralGalaxy_eqFunction_11862(data, threadData);
  SpiralGalaxy_eqFunction_6191(data, threadData);
  SpiralGalaxy_eqFunction_11861(data, threadData);
  SpiralGalaxy_eqFunction_6193(data, threadData);
  SpiralGalaxy_eqFunction_11874(data, threadData);
  SpiralGalaxy_eqFunction_11875(data, threadData);
  SpiralGalaxy_eqFunction_6196(data, threadData);
  SpiralGalaxy_eqFunction_6197(data, threadData);
  SpiralGalaxy_eqFunction_11876(data, threadData);
  SpiralGalaxy_eqFunction_11877(data, threadData);
  SpiralGalaxy_eqFunction_11880(data, threadData);
  SpiralGalaxy_eqFunction_11879(data, threadData);
  SpiralGalaxy_eqFunction_11878(data, threadData);
  SpiralGalaxy_eqFunction_6203(data, threadData);
  SpiralGalaxy_eqFunction_11873(data, threadData);
  SpiralGalaxy_eqFunction_6205(data, threadData);
  SpiralGalaxy_eqFunction_11872(data, threadData);
  SpiralGalaxy_eqFunction_6207(data, threadData);
  SpiralGalaxy_eqFunction_11871(data, threadData);
  SpiralGalaxy_eqFunction_6209(data, threadData);
  SpiralGalaxy_eqFunction_11884(data, threadData);
  SpiralGalaxy_eqFunction_11885(data, threadData);
  SpiralGalaxy_eqFunction_6212(data, threadData);
  SpiralGalaxy_eqFunction_6213(data, threadData);
  SpiralGalaxy_eqFunction_11886(data, threadData);
  SpiralGalaxy_eqFunction_11887(data, threadData);
  SpiralGalaxy_eqFunction_11890(data, threadData);
  SpiralGalaxy_eqFunction_11889(data, threadData);
  SpiralGalaxy_eqFunction_11888(data, threadData);
  SpiralGalaxy_eqFunction_6219(data, threadData);
  SpiralGalaxy_eqFunction_11883(data, threadData);
  SpiralGalaxy_eqFunction_6221(data, threadData);
  SpiralGalaxy_eqFunction_11882(data, threadData);
  SpiralGalaxy_eqFunction_6223(data, threadData);
  SpiralGalaxy_eqFunction_11881(data, threadData);
  SpiralGalaxy_eqFunction_6225(data, threadData);
  SpiralGalaxy_eqFunction_11894(data, threadData);
  SpiralGalaxy_eqFunction_11895(data, threadData);
  SpiralGalaxy_eqFunction_6228(data, threadData);
  SpiralGalaxy_eqFunction_6229(data, threadData);
  SpiralGalaxy_eqFunction_11896(data, threadData);
  SpiralGalaxy_eqFunction_11897(data, threadData);
  SpiralGalaxy_eqFunction_11900(data, threadData);
  SpiralGalaxy_eqFunction_11899(data, threadData);
  SpiralGalaxy_eqFunction_11898(data, threadData);
  SpiralGalaxy_eqFunction_6235(data, threadData);
  SpiralGalaxy_eqFunction_11893(data, threadData);
  SpiralGalaxy_eqFunction_6237(data, threadData);
  SpiralGalaxy_eqFunction_11892(data, threadData);
  SpiralGalaxy_eqFunction_6239(data, threadData);
  SpiralGalaxy_eqFunction_11891(data, threadData);
  SpiralGalaxy_eqFunction_6241(data, threadData);
  SpiralGalaxy_eqFunction_11904(data, threadData);
  SpiralGalaxy_eqFunction_11905(data, threadData);
  SpiralGalaxy_eqFunction_6244(data, threadData);
  SpiralGalaxy_eqFunction_6245(data, threadData);
  SpiralGalaxy_eqFunction_11906(data, threadData);
  SpiralGalaxy_eqFunction_11907(data, threadData);
  SpiralGalaxy_eqFunction_11910(data, threadData);
  SpiralGalaxy_eqFunction_11909(data, threadData);
  SpiralGalaxy_eqFunction_11908(data, threadData);
  SpiralGalaxy_eqFunction_6251(data, threadData);
  SpiralGalaxy_eqFunction_11903(data, threadData);
  SpiralGalaxy_eqFunction_6253(data, threadData);
  SpiralGalaxy_eqFunction_11902(data, threadData);
  SpiralGalaxy_eqFunction_6255(data, threadData);
  SpiralGalaxy_eqFunction_11901(data, threadData);
  SpiralGalaxy_eqFunction_6257(data, threadData);
  SpiralGalaxy_eqFunction_11914(data, threadData);
  SpiralGalaxy_eqFunction_11915(data, threadData);
  SpiralGalaxy_eqFunction_6260(data, threadData);
  SpiralGalaxy_eqFunction_6261(data, threadData);
  SpiralGalaxy_eqFunction_11916(data, threadData);
  SpiralGalaxy_eqFunction_11917(data, threadData);
  SpiralGalaxy_eqFunction_11920(data, threadData);
  SpiralGalaxy_eqFunction_11919(data, threadData);
  SpiralGalaxy_eqFunction_11918(data, threadData);
  SpiralGalaxy_eqFunction_6267(data, threadData);
  SpiralGalaxy_eqFunction_11913(data, threadData);
  SpiralGalaxy_eqFunction_6269(data, threadData);
  SpiralGalaxy_eqFunction_11912(data, threadData);
  SpiralGalaxy_eqFunction_6271(data, threadData);
  SpiralGalaxy_eqFunction_11911(data, threadData);
  SpiralGalaxy_eqFunction_6273(data, threadData);
  SpiralGalaxy_eqFunction_11924(data, threadData);
  SpiralGalaxy_eqFunction_11925(data, threadData);
  SpiralGalaxy_eqFunction_6276(data, threadData);
  SpiralGalaxy_eqFunction_6277(data, threadData);
  SpiralGalaxy_eqFunction_11926(data, threadData);
  SpiralGalaxy_eqFunction_11927(data, threadData);
  SpiralGalaxy_eqFunction_11930(data, threadData);
  SpiralGalaxy_eqFunction_11929(data, threadData);
  SpiralGalaxy_eqFunction_11928(data, threadData);
  SpiralGalaxy_eqFunction_6283(data, threadData);
  SpiralGalaxy_eqFunction_11923(data, threadData);
  SpiralGalaxy_eqFunction_6285(data, threadData);
  SpiralGalaxy_eqFunction_11922(data, threadData);
  SpiralGalaxy_eqFunction_6287(data, threadData);
  SpiralGalaxy_eqFunction_11921(data, threadData);
  SpiralGalaxy_eqFunction_6289(data, threadData);
  SpiralGalaxy_eqFunction_11934(data, threadData);
  SpiralGalaxy_eqFunction_11935(data, threadData);
  SpiralGalaxy_eqFunction_6292(data, threadData);
  SpiralGalaxy_eqFunction_6293(data, threadData);
  SpiralGalaxy_eqFunction_11936(data, threadData);
  SpiralGalaxy_eqFunction_11937(data, threadData);
  SpiralGalaxy_eqFunction_11940(data, threadData);
  SpiralGalaxy_eqFunction_11939(data, threadData);
  SpiralGalaxy_eqFunction_11938(data, threadData);
  SpiralGalaxy_eqFunction_6299(data, threadData);
  SpiralGalaxy_eqFunction_11933(data, threadData);
  SpiralGalaxy_eqFunction_6301(data, threadData);
  SpiralGalaxy_eqFunction_11932(data, threadData);
  SpiralGalaxy_eqFunction_6303(data, threadData);
  SpiralGalaxy_eqFunction_11931(data, threadData);
  SpiralGalaxy_eqFunction_6305(data, threadData);
  SpiralGalaxy_eqFunction_11944(data, threadData);
  SpiralGalaxy_eqFunction_11945(data, threadData);
  SpiralGalaxy_eqFunction_6308(data, threadData);
  SpiralGalaxy_eqFunction_6309(data, threadData);
  SpiralGalaxy_eqFunction_11946(data, threadData);
  SpiralGalaxy_eqFunction_11947(data, threadData);
  SpiralGalaxy_eqFunction_11950(data, threadData);
  SpiralGalaxy_eqFunction_11949(data, threadData);
  SpiralGalaxy_eqFunction_11948(data, threadData);
  SpiralGalaxy_eqFunction_6315(data, threadData);
  SpiralGalaxy_eqFunction_11943(data, threadData);
  SpiralGalaxy_eqFunction_6317(data, threadData);
  SpiralGalaxy_eqFunction_11942(data, threadData);
  SpiralGalaxy_eqFunction_6319(data, threadData);
  SpiralGalaxy_eqFunction_11941(data, threadData);
  SpiralGalaxy_eqFunction_6321(data, threadData);
  SpiralGalaxy_eqFunction_11954(data, threadData);
  SpiralGalaxy_eqFunction_11955(data, threadData);
  SpiralGalaxy_eqFunction_6324(data, threadData);
  SpiralGalaxy_eqFunction_6325(data, threadData);
  SpiralGalaxy_eqFunction_11956(data, threadData);
  SpiralGalaxy_eqFunction_11957(data, threadData);
  SpiralGalaxy_eqFunction_11960(data, threadData);
  SpiralGalaxy_eqFunction_11959(data, threadData);
  SpiralGalaxy_eqFunction_11958(data, threadData);
  SpiralGalaxy_eqFunction_6331(data, threadData);
  SpiralGalaxy_eqFunction_11953(data, threadData);
  SpiralGalaxy_eqFunction_6333(data, threadData);
  SpiralGalaxy_eqFunction_11952(data, threadData);
  SpiralGalaxy_eqFunction_6335(data, threadData);
  SpiralGalaxy_eqFunction_11951(data, threadData);
  SpiralGalaxy_eqFunction_6337(data, threadData);
  SpiralGalaxy_eqFunction_11964(data, threadData);
  SpiralGalaxy_eqFunction_11965(data, threadData);
  SpiralGalaxy_eqFunction_6340(data, threadData);
  SpiralGalaxy_eqFunction_6341(data, threadData);
  SpiralGalaxy_eqFunction_11966(data, threadData);
  SpiralGalaxy_eqFunction_11967(data, threadData);
  SpiralGalaxy_eqFunction_11970(data, threadData);
  SpiralGalaxy_eqFunction_11969(data, threadData);
  SpiralGalaxy_eqFunction_11968(data, threadData);
  SpiralGalaxy_eqFunction_6347(data, threadData);
  SpiralGalaxy_eqFunction_11963(data, threadData);
  SpiralGalaxy_eqFunction_6349(data, threadData);
  SpiralGalaxy_eqFunction_11962(data, threadData);
  SpiralGalaxy_eqFunction_6351(data, threadData);
  SpiralGalaxy_eqFunction_11961(data, threadData);
  SpiralGalaxy_eqFunction_6353(data, threadData);
  SpiralGalaxy_eqFunction_11974(data, threadData);
  SpiralGalaxy_eqFunction_11975(data, threadData);
  SpiralGalaxy_eqFunction_6356(data, threadData);
  SpiralGalaxy_eqFunction_6357(data, threadData);
  SpiralGalaxy_eqFunction_11976(data, threadData);
  SpiralGalaxy_eqFunction_11977(data, threadData);
  SpiralGalaxy_eqFunction_11980(data, threadData);
  SpiralGalaxy_eqFunction_11979(data, threadData);
  SpiralGalaxy_eqFunction_11978(data, threadData);
  SpiralGalaxy_eqFunction_6363(data, threadData);
  SpiralGalaxy_eqFunction_11973(data, threadData);
  SpiralGalaxy_eqFunction_6365(data, threadData);
  SpiralGalaxy_eqFunction_11972(data, threadData);
  SpiralGalaxy_eqFunction_6367(data, threadData);
  SpiralGalaxy_eqFunction_11971(data, threadData);
  SpiralGalaxy_eqFunction_6369(data, threadData);
  SpiralGalaxy_eqFunction_11984(data, threadData);
  SpiralGalaxy_eqFunction_11985(data, threadData);
  SpiralGalaxy_eqFunction_6372(data, threadData);
  SpiralGalaxy_eqFunction_6373(data, threadData);
  SpiralGalaxy_eqFunction_11986(data, threadData);
  SpiralGalaxy_eqFunction_11987(data, threadData);
  SpiralGalaxy_eqFunction_11990(data, threadData);
  SpiralGalaxy_eqFunction_11989(data, threadData);
  SpiralGalaxy_eqFunction_11988(data, threadData);
  SpiralGalaxy_eqFunction_6379(data, threadData);
  SpiralGalaxy_eqFunction_11983(data, threadData);
  SpiralGalaxy_eqFunction_6381(data, threadData);
  SpiralGalaxy_eqFunction_11982(data, threadData);
  SpiralGalaxy_eqFunction_6383(data, threadData);
  SpiralGalaxy_eqFunction_11981(data, threadData);
  SpiralGalaxy_eqFunction_6385(data, threadData);
  SpiralGalaxy_eqFunction_11994(data, threadData);
  SpiralGalaxy_eqFunction_11995(data, threadData);
  SpiralGalaxy_eqFunction_6388(data, threadData);
  SpiralGalaxy_eqFunction_6389(data, threadData);
  SpiralGalaxy_eqFunction_11996(data, threadData);
  SpiralGalaxy_eqFunction_11997(data, threadData);
  SpiralGalaxy_eqFunction_12000(data, threadData);
  SpiralGalaxy_eqFunction_11999(data, threadData);
  SpiralGalaxy_eqFunction_11998(data, threadData);
  SpiralGalaxy_eqFunction_6395(data, threadData);
  SpiralGalaxy_eqFunction_11993(data, threadData);
  SpiralGalaxy_eqFunction_6397(data, threadData);
  SpiralGalaxy_eqFunction_11992(data, threadData);
  SpiralGalaxy_eqFunction_6399(data, threadData);
  SpiralGalaxy_eqFunction_11991(data, threadData);
  SpiralGalaxy_eqFunction_6401(data, threadData);
  SpiralGalaxy_eqFunction_12004(data, threadData);
  SpiralGalaxy_eqFunction_12005(data, threadData);
  SpiralGalaxy_eqFunction_6404(data, threadData);
  SpiralGalaxy_eqFunction_6405(data, threadData);
  SpiralGalaxy_eqFunction_12006(data, threadData);
  SpiralGalaxy_eqFunction_12007(data, threadData);
  SpiralGalaxy_eqFunction_12010(data, threadData);
  SpiralGalaxy_eqFunction_12009(data, threadData);
  SpiralGalaxy_eqFunction_12008(data, threadData);
  SpiralGalaxy_eqFunction_6411(data, threadData);
  SpiralGalaxy_eqFunction_12003(data, threadData);
  SpiralGalaxy_eqFunction_6413(data, threadData);
  SpiralGalaxy_eqFunction_12002(data, threadData);
  SpiralGalaxy_eqFunction_6415(data, threadData);
  SpiralGalaxy_eqFunction_12001(data, threadData);
  SpiralGalaxy_eqFunction_6417(data, threadData);
  SpiralGalaxy_eqFunction_12014(data, threadData);
  SpiralGalaxy_eqFunction_12015(data, threadData);
  SpiralGalaxy_eqFunction_6420(data, threadData);
  SpiralGalaxy_eqFunction_6421(data, threadData);
  SpiralGalaxy_eqFunction_12016(data, threadData);
  SpiralGalaxy_eqFunction_12017(data, threadData);
  SpiralGalaxy_eqFunction_12020(data, threadData);
  SpiralGalaxy_eqFunction_12019(data, threadData);
  SpiralGalaxy_eqFunction_12018(data, threadData);
  SpiralGalaxy_eqFunction_6427(data, threadData);
  SpiralGalaxy_eqFunction_12013(data, threadData);
  SpiralGalaxy_eqFunction_6429(data, threadData);
  SpiralGalaxy_eqFunction_12012(data, threadData);
  SpiralGalaxy_eqFunction_6431(data, threadData);
  SpiralGalaxy_eqFunction_12011(data, threadData);
  SpiralGalaxy_eqFunction_6433(data, threadData);
  SpiralGalaxy_eqFunction_12024(data, threadData);
  SpiralGalaxy_eqFunction_12025(data, threadData);
  SpiralGalaxy_eqFunction_6436(data, threadData);
  SpiralGalaxy_eqFunction_6437(data, threadData);
  SpiralGalaxy_eqFunction_12026(data, threadData);
  SpiralGalaxy_eqFunction_12027(data, threadData);
  SpiralGalaxy_eqFunction_12030(data, threadData);
  SpiralGalaxy_eqFunction_12029(data, threadData);
  SpiralGalaxy_eqFunction_12028(data, threadData);
  SpiralGalaxy_eqFunction_6443(data, threadData);
  SpiralGalaxy_eqFunction_12023(data, threadData);
  SpiralGalaxy_eqFunction_6445(data, threadData);
  SpiralGalaxy_eqFunction_12022(data, threadData);
  SpiralGalaxy_eqFunction_6447(data, threadData);
  SpiralGalaxy_eqFunction_12021(data, threadData);
  SpiralGalaxy_eqFunction_6449(data, threadData);
  SpiralGalaxy_eqFunction_12034(data, threadData);
  SpiralGalaxy_eqFunction_12035(data, threadData);
  SpiralGalaxy_eqFunction_6452(data, threadData);
  SpiralGalaxy_eqFunction_6453(data, threadData);
  SpiralGalaxy_eqFunction_12036(data, threadData);
  SpiralGalaxy_eqFunction_12037(data, threadData);
  SpiralGalaxy_eqFunction_12040(data, threadData);
  SpiralGalaxy_eqFunction_12039(data, threadData);
  SpiralGalaxy_eqFunction_12038(data, threadData);
  SpiralGalaxy_eqFunction_6459(data, threadData);
  SpiralGalaxy_eqFunction_12033(data, threadData);
  SpiralGalaxy_eqFunction_6461(data, threadData);
  SpiralGalaxy_eqFunction_12032(data, threadData);
  SpiralGalaxy_eqFunction_6463(data, threadData);
  SpiralGalaxy_eqFunction_12031(data, threadData);
  SpiralGalaxy_eqFunction_6465(data, threadData);
  SpiralGalaxy_eqFunction_12044(data, threadData);
  SpiralGalaxy_eqFunction_12045(data, threadData);
  SpiralGalaxy_eqFunction_6468(data, threadData);
  SpiralGalaxy_eqFunction_6469(data, threadData);
  SpiralGalaxy_eqFunction_12046(data, threadData);
  SpiralGalaxy_eqFunction_12047(data, threadData);
  SpiralGalaxy_eqFunction_12050(data, threadData);
  SpiralGalaxy_eqFunction_12049(data, threadData);
  SpiralGalaxy_eqFunction_12048(data, threadData);
  SpiralGalaxy_eqFunction_6475(data, threadData);
  SpiralGalaxy_eqFunction_12043(data, threadData);
  SpiralGalaxy_eqFunction_6477(data, threadData);
  SpiralGalaxy_eqFunction_12042(data, threadData);
  SpiralGalaxy_eqFunction_6479(data, threadData);
  SpiralGalaxy_eqFunction_12041(data, threadData);
  SpiralGalaxy_eqFunction_6481(data, threadData);
  SpiralGalaxy_eqFunction_12054(data, threadData);
  SpiralGalaxy_eqFunction_12055(data, threadData);
  SpiralGalaxy_eqFunction_6484(data, threadData);
  SpiralGalaxy_eqFunction_6485(data, threadData);
  SpiralGalaxy_eqFunction_12056(data, threadData);
  SpiralGalaxy_eqFunction_12057(data, threadData);
  SpiralGalaxy_eqFunction_12060(data, threadData);
  SpiralGalaxy_eqFunction_12059(data, threadData);
  SpiralGalaxy_eqFunction_12058(data, threadData);
  SpiralGalaxy_eqFunction_6491(data, threadData);
  SpiralGalaxy_eqFunction_12053(data, threadData);
  SpiralGalaxy_eqFunction_6493(data, threadData);
  SpiralGalaxy_eqFunction_12052(data, threadData);
  SpiralGalaxy_eqFunction_6495(data, threadData);
  SpiralGalaxy_eqFunction_12051(data, threadData);
  SpiralGalaxy_eqFunction_6497(data, threadData);
  SpiralGalaxy_eqFunction_12064(data, threadData);
  SpiralGalaxy_eqFunction_12065(data, threadData);
  SpiralGalaxy_eqFunction_6500(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif