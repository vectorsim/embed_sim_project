#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 4001
type: SIMPLE_ASSIGN
z[251] = 1.6000000000000004e-4
*/
void SpiralGalaxy_eqFunction_4001(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4001};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2750]] /* z[251] STATE(1,vz[251]) */) = 1.6000000000000004e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10504(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10505(DATA *data, threadData_t *threadData);


/*
equation index: 4004
type: SIMPLE_ASSIGN
y[251] = r_init[251] * sin(theta[251] + 4.000000000000004e-5)
*/
void SpiralGalaxy_eqFunction_4004(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4004};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2250]] /* y[251] STATE(1,vy[251]) */) = ((data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */)) * (sin((data->simulationInfo->realParameter[1757] /* theta[251] PARAM */) + 4.000000000000004e-5));
  TRACE_POP
}

/*
equation index: 4005
type: SIMPLE_ASSIGN
x[251] = r_init[251] * cos(theta[251] + 4.000000000000004e-5)
*/
void SpiralGalaxy_eqFunction_4005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4005};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1750]] /* x[251] STATE(1,vx[251]) */) = ((data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */)) * (cos((data->simulationInfo->realParameter[1757] /* theta[251] PARAM */) + 4.000000000000004e-5));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10506(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10507(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10510(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10509(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10508(DATA *data, threadData_t *threadData);


/*
equation index: 4011
type: SIMPLE_ASSIGN
vx[251] = (-sin(theta[251])) * r_init[251] * omega_c[251]
*/
void SpiralGalaxy_eqFunction_4011(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4011};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[250]] /* vx[251] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1757] /* theta[251] PARAM */)))) * (((data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */)) * ((data->simulationInfo->realParameter[755] /* omega_c[251] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10503(DATA *data, threadData_t *threadData);


/*
equation index: 4013
type: SIMPLE_ASSIGN
vy[251] = cos(theta[251]) * r_init[251] * omega_c[251]
*/
void SpiralGalaxy_eqFunction_4013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4013};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[750]] /* vy[251] STATE(1) */) = (cos((data->simulationInfo->realParameter[1757] /* theta[251] PARAM */))) * (((data->simulationInfo->realParameter[1256] /* r_init[251] PARAM */)) * ((data->simulationInfo->realParameter[755] /* omega_c[251] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10502(DATA *data, threadData_t *threadData);


/*
equation index: 4015
type: SIMPLE_ASSIGN
vz[251] = 0.0
*/
void SpiralGalaxy_eqFunction_4015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4015};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1250]] /* vz[251] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10501(DATA *data, threadData_t *threadData);


/*
equation index: 4017
type: SIMPLE_ASSIGN
z[252] = 3.200000000000001e-4
*/
void SpiralGalaxy_eqFunction_4017(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4017};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2751]] /* z[252] STATE(1,vz[252]) */) = 3.200000000000001e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10514(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10515(DATA *data, threadData_t *threadData);


/*
equation index: 4020
type: SIMPLE_ASSIGN
y[252] = r_init[252] * sin(theta[252] + 8.000000000000007e-5)
*/
void SpiralGalaxy_eqFunction_4020(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4020};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2251]] /* y[252] STATE(1,vy[252]) */) = ((data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */)) * (sin((data->simulationInfo->realParameter[1758] /* theta[252] PARAM */) + 8.000000000000007e-5));
  TRACE_POP
}

/*
equation index: 4021
type: SIMPLE_ASSIGN
x[252] = r_init[252] * cos(theta[252] + 8.000000000000007e-5)
*/
void SpiralGalaxy_eqFunction_4021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4021};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1751]] /* x[252] STATE(1,vx[252]) */) = ((data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */)) * (cos((data->simulationInfo->realParameter[1758] /* theta[252] PARAM */) + 8.000000000000007e-5));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10516(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10517(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10520(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10519(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10518(DATA *data, threadData_t *threadData);


/*
equation index: 4027
type: SIMPLE_ASSIGN
vx[252] = (-sin(theta[252])) * r_init[252] * omega_c[252]
*/
void SpiralGalaxy_eqFunction_4027(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4027};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[251]] /* vx[252] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1758] /* theta[252] PARAM */)))) * (((data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */)) * ((data->simulationInfo->realParameter[756] /* omega_c[252] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10513(DATA *data, threadData_t *threadData);


/*
equation index: 4029
type: SIMPLE_ASSIGN
vy[252] = cos(theta[252]) * r_init[252] * omega_c[252]
*/
void SpiralGalaxy_eqFunction_4029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4029};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[751]] /* vy[252] STATE(1) */) = (cos((data->simulationInfo->realParameter[1758] /* theta[252] PARAM */))) * (((data->simulationInfo->realParameter[1257] /* r_init[252] PARAM */)) * ((data->simulationInfo->realParameter[756] /* omega_c[252] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10512(DATA *data, threadData_t *threadData);


/*
equation index: 4031
type: SIMPLE_ASSIGN
vz[252] = 0.0
*/
void SpiralGalaxy_eqFunction_4031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4031};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1251]] /* vz[252] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10511(DATA *data, threadData_t *threadData);


/*
equation index: 4033
type: SIMPLE_ASSIGN
z[253] = 4.8000000000000007e-4
*/
void SpiralGalaxy_eqFunction_4033(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4033};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2752]] /* z[253] STATE(1,vz[253]) */) = 4.8000000000000007e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10524(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10525(DATA *data, threadData_t *threadData);


/*
equation index: 4036
type: SIMPLE_ASSIGN
y[253] = r_init[253] * sin(theta[253] + 1.2000000000000011e-4)
*/
void SpiralGalaxy_eqFunction_4036(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4036};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2252]] /* y[253] STATE(1,vy[253]) */) = ((data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */)) * (sin((data->simulationInfo->realParameter[1759] /* theta[253] PARAM */) + 1.2000000000000011e-4));
  TRACE_POP
}

/*
equation index: 4037
type: SIMPLE_ASSIGN
x[253] = r_init[253] * cos(theta[253] + 1.2000000000000011e-4)
*/
void SpiralGalaxy_eqFunction_4037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4037};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1752]] /* x[253] STATE(1,vx[253]) */) = ((data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */)) * (cos((data->simulationInfo->realParameter[1759] /* theta[253] PARAM */) + 1.2000000000000011e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10526(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10527(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10530(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10529(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10528(DATA *data, threadData_t *threadData);


/*
equation index: 4043
type: SIMPLE_ASSIGN
vx[253] = (-sin(theta[253])) * r_init[253] * omega_c[253]
*/
void SpiralGalaxy_eqFunction_4043(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4043};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[252]] /* vx[253] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1759] /* theta[253] PARAM */)))) * (((data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */)) * ((data->simulationInfo->realParameter[757] /* omega_c[253] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10523(DATA *data, threadData_t *threadData);


/*
equation index: 4045
type: SIMPLE_ASSIGN
vy[253] = cos(theta[253]) * r_init[253] * omega_c[253]
*/
void SpiralGalaxy_eqFunction_4045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4045};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[752]] /* vy[253] STATE(1) */) = (cos((data->simulationInfo->realParameter[1759] /* theta[253] PARAM */))) * (((data->simulationInfo->realParameter[1258] /* r_init[253] PARAM */)) * ((data->simulationInfo->realParameter[757] /* omega_c[253] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10522(DATA *data, threadData_t *threadData);


/*
equation index: 4047
type: SIMPLE_ASSIGN
vz[253] = 0.0
*/
void SpiralGalaxy_eqFunction_4047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4047};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1252]] /* vz[253] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10521(DATA *data, threadData_t *threadData);


/*
equation index: 4049
type: SIMPLE_ASSIGN
z[254] = 6.400000000000002e-4
*/
void SpiralGalaxy_eqFunction_4049(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4049};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2753]] /* z[254] STATE(1,vz[254]) */) = 6.400000000000002e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10534(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10535(DATA *data, threadData_t *threadData);


/*
equation index: 4052
type: SIMPLE_ASSIGN
y[254] = r_init[254] * sin(theta[254] + 1.6000000000000015e-4)
*/
void SpiralGalaxy_eqFunction_4052(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4052};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2253]] /* y[254] STATE(1,vy[254]) */) = ((data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */)) * (sin((data->simulationInfo->realParameter[1760] /* theta[254] PARAM */) + 1.6000000000000015e-4));
  TRACE_POP
}

/*
equation index: 4053
type: SIMPLE_ASSIGN
x[254] = r_init[254] * cos(theta[254] + 1.6000000000000015e-4)
*/
void SpiralGalaxy_eqFunction_4053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4053};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1753]] /* x[254] STATE(1,vx[254]) */) = ((data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */)) * (cos((data->simulationInfo->realParameter[1760] /* theta[254] PARAM */) + 1.6000000000000015e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10536(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10537(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10540(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10539(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10538(DATA *data, threadData_t *threadData);


/*
equation index: 4059
type: SIMPLE_ASSIGN
vx[254] = (-sin(theta[254])) * r_init[254] * omega_c[254]
*/
void SpiralGalaxy_eqFunction_4059(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4059};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[253]] /* vx[254] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1760] /* theta[254] PARAM */)))) * (((data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */)) * ((data->simulationInfo->realParameter[758] /* omega_c[254] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10533(DATA *data, threadData_t *threadData);


/*
equation index: 4061
type: SIMPLE_ASSIGN
vy[254] = cos(theta[254]) * r_init[254] * omega_c[254]
*/
void SpiralGalaxy_eqFunction_4061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4061};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[753]] /* vy[254] STATE(1) */) = (cos((data->simulationInfo->realParameter[1760] /* theta[254] PARAM */))) * (((data->simulationInfo->realParameter[1259] /* r_init[254] PARAM */)) * ((data->simulationInfo->realParameter[758] /* omega_c[254] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10532(DATA *data, threadData_t *threadData);


/*
equation index: 4063
type: SIMPLE_ASSIGN
vz[254] = 0.0
*/
void SpiralGalaxy_eqFunction_4063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4063};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1253]] /* vz[254] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10531(DATA *data, threadData_t *threadData);


/*
equation index: 4065
type: SIMPLE_ASSIGN
z[255] = 8.000000000000001e-4
*/
void SpiralGalaxy_eqFunction_4065(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4065};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2754]] /* z[255] STATE(1,vz[255]) */) = 8.000000000000001e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10544(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10545(DATA *data, threadData_t *threadData);


/*
equation index: 4068
type: SIMPLE_ASSIGN
y[255] = r_init[255] * sin(theta[255] + 2.0000000000000017e-4)
*/
void SpiralGalaxy_eqFunction_4068(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4068};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2254]] /* y[255] STATE(1,vy[255]) */) = ((data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */)) * (sin((data->simulationInfo->realParameter[1761] /* theta[255] PARAM */) + 2.0000000000000017e-4));
  TRACE_POP
}

/*
equation index: 4069
type: SIMPLE_ASSIGN
x[255] = r_init[255] * cos(theta[255] + 2.0000000000000017e-4)
*/
void SpiralGalaxy_eqFunction_4069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4069};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1754]] /* x[255] STATE(1,vx[255]) */) = ((data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */)) * (cos((data->simulationInfo->realParameter[1761] /* theta[255] PARAM */) + 2.0000000000000017e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10546(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10547(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10550(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10549(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10548(DATA *data, threadData_t *threadData);


/*
equation index: 4075
type: SIMPLE_ASSIGN
vx[255] = (-sin(theta[255])) * r_init[255] * omega_c[255]
*/
void SpiralGalaxy_eqFunction_4075(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4075};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[254]] /* vx[255] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1761] /* theta[255] PARAM */)))) * (((data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */)) * ((data->simulationInfo->realParameter[759] /* omega_c[255] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10543(DATA *data, threadData_t *threadData);


/*
equation index: 4077
type: SIMPLE_ASSIGN
vy[255] = cos(theta[255]) * r_init[255] * omega_c[255]
*/
void SpiralGalaxy_eqFunction_4077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4077};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[754]] /* vy[255] STATE(1) */) = (cos((data->simulationInfo->realParameter[1761] /* theta[255] PARAM */))) * (((data->simulationInfo->realParameter[1260] /* r_init[255] PARAM */)) * ((data->simulationInfo->realParameter[759] /* omega_c[255] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10542(DATA *data, threadData_t *threadData);


/*
equation index: 4079
type: SIMPLE_ASSIGN
vz[255] = 0.0
*/
void SpiralGalaxy_eqFunction_4079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4079};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1254]] /* vz[255] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10541(DATA *data, threadData_t *threadData);


/*
equation index: 4081
type: SIMPLE_ASSIGN
z[256] = 9.600000000000001e-4
*/
void SpiralGalaxy_eqFunction_4081(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4081};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2755]] /* z[256] STATE(1,vz[256]) */) = 9.600000000000001e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10554(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10555(DATA *data, threadData_t *threadData);


/*
equation index: 4084
type: SIMPLE_ASSIGN
y[256] = r_init[256] * sin(theta[256] + 2.4000000000000022e-4)
*/
void SpiralGalaxy_eqFunction_4084(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4084};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2255]] /* y[256] STATE(1,vy[256]) */) = ((data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */)) * (sin((data->simulationInfo->realParameter[1762] /* theta[256] PARAM */) + 2.4000000000000022e-4));
  TRACE_POP
}

/*
equation index: 4085
type: SIMPLE_ASSIGN
x[256] = r_init[256] * cos(theta[256] + 2.4000000000000022e-4)
*/
void SpiralGalaxy_eqFunction_4085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4085};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1755]] /* x[256] STATE(1,vx[256]) */) = ((data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */)) * (cos((data->simulationInfo->realParameter[1762] /* theta[256] PARAM */) + 2.4000000000000022e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10556(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10557(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10560(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10559(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10558(DATA *data, threadData_t *threadData);


/*
equation index: 4091
type: SIMPLE_ASSIGN
vx[256] = (-sin(theta[256])) * r_init[256] * omega_c[256]
*/
void SpiralGalaxy_eqFunction_4091(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4091};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[255]] /* vx[256] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1762] /* theta[256] PARAM */)))) * (((data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */)) * ((data->simulationInfo->realParameter[760] /* omega_c[256] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10553(DATA *data, threadData_t *threadData);


/*
equation index: 4093
type: SIMPLE_ASSIGN
vy[256] = cos(theta[256]) * r_init[256] * omega_c[256]
*/
void SpiralGalaxy_eqFunction_4093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4093};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[755]] /* vy[256] STATE(1) */) = (cos((data->simulationInfo->realParameter[1762] /* theta[256] PARAM */))) * (((data->simulationInfo->realParameter[1261] /* r_init[256] PARAM */)) * ((data->simulationInfo->realParameter[760] /* omega_c[256] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10552(DATA *data, threadData_t *threadData);


/*
equation index: 4095
type: SIMPLE_ASSIGN
vz[256] = 0.0
*/
void SpiralGalaxy_eqFunction_4095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4095};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1255]] /* vz[256] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10551(DATA *data, threadData_t *threadData);


/*
equation index: 4097
type: SIMPLE_ASSIGN
z[257] = 0.0011200000000000003
*/
void SpiralGalaxy_eqFunction_4097(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4097};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2756]] /* z[257] STATE(1,vz[257]) */) = 0.0011200000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10564(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10565(DATA *data, threadData_t *threadData);


/*
equation index: 4100
type: SIMPLE_ASSIGN
y[257] = r_init[257] * sin(theta[257] + 2.8000000000000025e-4)
*/
void SpiralGalaxy_eqFunction_4100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4100};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2256]] /* y[257] STATE(1,vy[257]) */) = ((data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */)) * (sin((data->simulationInfo->realParameter[1763] /* theta[257] PARAM */) + 2.8000000000000025e-4));
  TRACE_POP
}

/*
equation index: 4101
type: SIMPLE_ASSIGN
x[257] = r_init[257] * cos(theta[257] + 2.8000000000000025e-4)
*/
void SpiralGalaxy_eqFunction_4101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1756]] /* x[257] STATE(1,vx[257]) */) = ((data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */)) * (cos((data->simulationInfo->realParameter[1763] /* theta[257] PARAM */) + 2.8000000000000025e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10566(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10567(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10570(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10569(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10568(DATA *data, threadData_t *threadData);


/*
equation index: 4107
type: SIMPLE_ASSIGN
vx[257] = (-sin(theta[257])) * r_init[257] * omega_c[257]
*/
void SpiralGalaxy_eqFunction_4107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4107};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[256]] /* vx[257] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1763] /* theta[257] PARAM */)))) * (((data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */)) * ((data->simulationInfo->realParameter[761] /* omega_c[257] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10563(DATA *data, threadData_t *threadData);


/*
equation index: 4109
type: SIMPLE_ASSIGN
vy[257] = cos(theta[257]) * r_init[257] * omega_c[257]
*/
void SpiralGalaxy_eqFunction_4109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[756]] /* vy[257] STATE(1) */) = (cos((data->simulationInfo->realParameter[1763] /* theta[257] PARAM */))) * (((data->simulationInfo->realParameter[1262] /* r_init[257] PARAM */)) * ((data->simulationInfo->realParameter[761] /* omega_c[257] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10562(DATA *data, threadData_t *threadData);


/*
equation index: 4111
type: SIMPLE_ASSIGN
vz[257] = 0.0
*/
void SpiralGalaxy_eqFunction_4111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4111};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1256]] /* vz[257] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10561(DATA *data, threadData_t *threadData);


/*
equation index: 4113
type: SIMPLE_ASSIGN
z[258] = 0.0012800000000000003
*/
void SpiralGalaxy_eqFunction_4113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4113};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2757]] /* z[258] STATE(1,vz[258]) */) = 0.0012800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10574(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10575(DATA *data, threadData_t *threadData);


/*
equation index: 4116
type: SIMPLE_ASSIGN
y[258] = r_init[258] * sin(theta[258] + 3.200000000000003e-4)
*/
void SpiralGalaxy_eqFunction_4116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4116};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2257]] /* y[258] STATE(1,vy[258]) */) = ((data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */)) * (sin((data->simulationInfo->realParameter[1764] /* theta[258] PARAM */) + 3.200000000000003e-4));
  TRACE_POP
}

/*
equation index: 4117
type: SIMPLE_ASSIGN
x[258] = r_init[258] * cos(theta[258] + 3.200000000000003e-4)
*/
void SpiralGalaxy_eqFunction_4117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1757]] /* x[258] STATE(1,vx[258]) */) = ((data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */)) * (cos((data->simulationInfo->realParameter[1764] /* theta[258] PARAM */) + 3.200000000000003e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10576(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10577(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10580(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10579(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10578(DATA *data, threadData_t *threadData);


/*
equation index: 4123
type: SIMPLE_ASSIGN
vx[258] = (-sin(theta[258])) * r_init[258] * omega_c[258]
*/
void SpiralGalaxy_eqFunction_4123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4123};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[257]] /* vx[258] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1764] /* theta[258] PARAM */)))) * (((data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */)) * ((data->simulationInfo->realParameter[762] /* omega_c[258] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10573(DATA *data, threadData_t *threadData);


/*
equation index: 4125
type: SIMPLE_ASSIGN
vy[258] = cos(theta[258]) * r_init[258] * omega_c[258]
*/
void SpiralGalaxy_eqFunction_4125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[757]] /* vy[258] STATE(1) */) = (cos((data->simulationInfo->realParameter[1764] /* theta[258] PARAM */))) * (((data->simulationInfo->realParameter[1263] /* r_init[258] PARAM */)) * ((data->simulationInfo->realParameter[762] /* omega_c[258] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10572(DATA *data, threadData_t *threadData);


/*
equation index: 4127
type: SIMPLE_ASSIGN
vz[258] = 0.0
*/
void SpiralGalaxy_eqFunction_4127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4127};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1257]] /* vz[258] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10571(DATA *data, threadData_t *threadData);


/*
equation index: 4129
type: SIMPLE_ASSIGN
z[259] = 0.0014400000000000005
*/
void SpiralGalaxy_eqFunction_4129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4129};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2758]] /* z[259] STATE(1,vz[259]) */) = 0.0014400000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10584(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10585(DATA *data, threadData_t *threadData);


/*
equation index: 4132
type: SIMPLE_ASSIGN
y[259] = r_init[259] * sin(theta[259] + 3.6000000000000035e-4)
*/
void SpiralGalaxy_eqFunction_4132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4132};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2258]] /* y[259] STATE(1,vy[259]) */) = ((data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */)) * (sin((data->simulationInfo->realParameter[1765] /* theta[259] PARAM */) + 3.6000000000000035e-4));
  TRACE_POP
}

/*
equation index: 4133
type: SIMPLE_ASSIGN
x[259] = r_init[259] * cos(theta[259] + 3.6000000000000035e-4)
*/
void SpiralGalaxy_eqFunction_4133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1758]] /* x[259] STATE(1,vx[259]) */) = ((data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */)) * (cos((data->simulationInfo->realParameter[1765] /* theta[259] PARAM */) + 3.6000000000000035e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10586(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10587(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10590(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10589(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10588(DATA *data, threadData_t *threadData);


/*
equation index: 4139
type: SIMPLE_ASSIGN
vx[259] = (-sin(theta[259])) * r_init[259] * omega_c[259]
*/
void SpiralGalaxy_eqFunction_4139(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4139};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[258]] /* vx[259] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1765] /* theta[259] PARAM */)))) * (((data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */)) * ((data->simulationInfo->realParameter[763] /* omega_c[259] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10583(DATA *data, threadData_t *threadData);


/*
equation index: 4141
type: SIMPLE_ASSIGN
vy[259] = cos(theta[259]) * r_init[259] * omega_c[259]
*/
void SpiralGalaxy_eqFunction_4141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[758]] /* vy[259] STATE(1) */) = (cos((data->simulationInfo->realParameter[1765] /* theta[259] PARAM */))) * (((data->simulationInfo->realParameter[1264] /* r_init[259] PARAM */)) * ((data->simulationInfo->realParameter[763] /* omega_c[259] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10582(DATA *data, threadData_t *threadData);


/*
equation index: 4143
type: SIMPLE_ASSIGN
vz[259] = 0.0
*/
void SpiralGalaxy_eqFunction_4143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4143};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1258]] /* vz[259] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10581(DATA *data, threadData_t *threadData);


/*
equation index: 4145
type: SIMPLE_ASSIGN
z[260] = 0.0016000000000000003
*/
void SpiralGalaxy_eqFunction_4145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4145};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2759]] /* z[260] STATE(1,vz[260]) */) = 0.0016000000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10594(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10595(DATA *data, threadData_t *threadData);


/*
equation index: 4148
type: SIMPLE_ASSIGN
y[260] = r_init[260] * sin(theta[260] + 4.0000000000000034e-4)
*/
void SpiralGalaxy_eqFunction_4148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4148};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2259]] /* y[260] STATE(1,vy[260]) */) = ((data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */)) * (sin((data->simulationInfo->realParameter[1766] /* theta[260] PARAM */) + 4.0000000000000034e-4));
  TRACE_POP
}

/*
equation index: 4149
type: SIMPLE_ASSIGN
x[260] = r_init[260] * cos(theta[260] + 4.0000000000000034e-4)
*/
void SpiralGalaxy_eqFunction_4149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1759]] /* x[260] STATE(1,vx[260]) */) = ((data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */)) * (cos((data->simulationInfo->realParameter[1766] /* theta[260] PARAM */) + 4.0000000000000034e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10596(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10597(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10600(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10599(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10598(DATA *data, threadData_t *threadData);


/*
equation index: 4155
type: SIMPLE_ASSIGN
vx[260] = (-sin(theta[260])) * r_init[260] * omega_c[260]
*/
void SpiralGalaxy_eqFunction_4155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4155};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[259]] /* vx[260] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1766] /* theta[260] PARAM */)))) * (((data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */)) * ((data->simulationInfo->realParameter[764] /* omega_c[260] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10593(DATA *data, threadData_t *threadData);


/*
equation index: 4157
type: SIMPLE_ASSIGN
vy[260] = cos(theta[260]) * r_init[260] * omega_c[260]
*/
void SpiralGalaxy_eqFunction_4157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[759]] /* vy[260] STATE(1) */) = (cos((data->simulationInfo->realParameter[1766] /* theta[260] PARAM */))) * (((data->simulationInfo->realParameter[1265] /* r_init[260] PARAM */)) * ((data->simulationInfo->realParameter[764] /* omega_c[260] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10592(DATA *data, threadData_t *threadData);


/*
equation index: 4159
type: SIMPLE_ASSIGN
vz[260] = 0.0
*/
void SpiralGalaxy_eqFunction_4159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4159};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1259]] /* vz[260] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10591(DATA *data, threadData_t *threadData);


/*
equation index: 4161
type: SIMPLE_ASSIGN
z[261] = 0.0017600000000000003
*/
void SpiralGalaxy_eqFunction_4161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4161};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2760]] /* z[261] STATE(1,vz[261]) */) = 0.0017600000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10604(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10605(DATA *data, threadData_t *threadData);


/*
equation index: 4164
type: SIMPLE_ASSIGN
y[261] = r_init[261] * sin(theta[261] + 4.400000000000004e-4)
*/
void SpiralGalaxy_eqFunction_4164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4164};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2260]] /* y[261] STATE(1,vy[261]) */) = ((data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */)) * (sin((data->simulationInfo->realParameter[1767] /* theta[261] PARAM */) + 4.400000000000004e-4));
  TRACE_POP
}

/*
equation index: 4165
type: SIMPLE_ASSIGN
x[261] = r_init[261] * cos(theta[261] + 4.400000000000004e-4)
*/
void SpiralGalaxy_eqFunction_4165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1760]] /* x[261] STATE(1,vx[261]) */) = ((data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */)) * (cos((data->simulationInfo->realParameter[1767] /* theta[261] PARAM */) + 4.400000000000004e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10606(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10607(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10610(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10609(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10608(DATA *data, threadData_t *threadData);


/*
equation index: 4171
type: SIMPLE_ASSIGN
vx[261] = (-sin(theta[261])) * r_init[261] * omega_c[261]
*/
void SpiralGalaxy_eqFunction_4171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4171};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[260]] /* vx[261] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1767] /* theta[261] PARAM */)))) * (((data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */)) * ((data->simulationInfo->realParameter[765] /* omega_c[261] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10603(DATA *data, threadData_t *threadData);


/*
equation index: 4173
type: SIMPLE_ASSIGN
vy[261] = cos(theta[261]) * r_init[261] * omega_c[261]
*/
void SpiralGalaxy_eqFunction_4173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[760]] /* vy[261] STATE(1) */) = (cos((data->simulationInfo->realParameter[1767] /* theta[261] PARAM */))) * (((data->simulationInfo->realParameter[1266] /* r_init[261] PARAM */)) * ((data->simulationInfo->realParameter[765] /* omega_c[261] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10602(DATA *data, threadData_t *threadData);


/*
equation index: 4175
type: SIMPLE_ASSIGN
vz[261] = 0.0
*/
void SpiralGalaxy_eqFunction_4175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4175};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1260]] /* vz[261] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10601(DATA *data, threadData_t *threadData);


/*
equation index: 4177
type: SIMPLE_ASSIGN
z[262] = 0.0019200000000000003
*/
void SpiralGalaxy_eqFunction_4177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4177};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2761]] /* z[262] STATE(1,vz[262]) */) = 0.0019200000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10614(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10615(DATA *data, threadData_t *threadData);


/*
equation index: 4180
type: SIMPLE_ASSIGN
y[262] = r_init[262] * sin(theta[262] + 4.8000000000000045e-4)
*/
void SpiralGalaxy_eqFunction_4180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4180};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2261]] /* y[262] STATE(1,vy[262]) */) = ((data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */)) * (sin((data->simulationInfo->realParameter[1768] /* theta[262] PARAM */) + 4.8000000000000045e-4));
  TRACE_POP
}

/*
equation index: 4181
type: SIMPLE_ASSIGN
x[262] = r_init[262] * cos(theta[262] + 4.8000000000000045e-4)
*/
void SpiralGalaxy_eqFunction_4181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1761]] /* x[262] STATE(1,vx[262]) */) = ((data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */)) * (cos((data->simulationInfo->realParameter[1768] /* theta[262] PARAM */) + 4.8000000000000045e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10616(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10617(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10620(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10619(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10618(DATA *data, threadData_t *threadData);


/*
equation index: 4187
type: SIMPLE_ASSIGN
vx[262] = (-sin(theta[262])) * r_init[262] * omega_c[262]
*/
void SpiralGalaxy_eqFunction_4187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4187};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[261]] /* vx[262] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1768] /* theta[262] PARAM */)))) * (((data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */)) * ((data->simulationInfo->realParameter[766] /* omega_c[262] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10613(DATA *data, threadData_t *threadData);


/*
equation index: 4189
type: SIMPLE_ASSIGN
vy[262] = cos(theta[262]) * r_init[262] * omega_c[262]
*/
void SpiralGalaxy_eqFunction_4189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[761]] /* vy[262] STATE(1) */) = (cos((data->simulationInfo->realParameter[1768] /* theta[262] PARAM */))) * (((data->simulationInfo->realParameter[1267] /* r_init[262] PARAM */)) * ((data->simulationInfo->realParameter[766] /* omega_c[262] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10612(DATA *data, threadData_t *threadData);


/*
equation index: 4191
type: SIMPLE_ASSIGN
vz[262] = 0.0
*/
void SpiralGalaxy_eqFunction_4191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4191};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1261]] /* vz[262] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10611(DATA *data, threadData_t *threadData);


/*
equation index: 4193
type: SIMPLE_ASSIGN
z[263] = 0.0020800000000000003
*/
void SpiralGalaxy_eqFunction_4193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4193};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2762]] /* z[263] STATE(1,vz[263]) */) = 0.0020800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10624(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10625(DATA *data, threadData_t *threadData);


/*
equation index: 4196
type: SIMPLE_ASSIGN
y[263] = r_init[263] * sin(theta[263] + 5.200000000000005e-4)
*/
void SpiralGalaxy_eqFunction_4196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4196};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2262]] /* y[263] STATE(1,vy[263]) */) = ((data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */)) * (sin((data->simulationInfo->realParameter[1769] /* theta[263] PARAM */) + 5.200000000000005e-4));
  TRACE_POP
}

/*
equation index: 4197
type: SIMPLE_ASSIGN
x[263] = r_init[263] * cos(theta[263] + 5.200000000000005e-4)
*/
void SpiralGalaxy_eqFunction_4197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1762]] /* x[263] STATE(1,vx[263]) */) = ((data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */)) * (cos((data->simulationInfo->realParameter[1769] /* theta[263] PARAM */) + 5.200000000000005e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10626(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10627(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10630(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10629(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10628(DATA *data, threadData_t *threadData);


/*
equation index: 4203
type: SIMPLE_ASSIGN
vx[263] = (-sin(theta[263])) * r_init[263] * omega_c[263]
*/
void SpiralGalaxy_eqFunction_4203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4203};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[262]] /* vx[263] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1769] /* theta[263] PARAM */)))) * (((data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */)) * ((data->simulationInfo->realParameter[767] /* omega_c[263] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10623(DATA *data, threadData_t *threadData);


/*
equation index: 4205
type: SIMPLE_ASSIGN
vy[263] = cos(theta[263]) * r_init[263] * omega_c[263]
*/
void SpiralGalaxy_eqFunction_4205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[762]] /* vy[263] STATE(1) */) = (cos((data->simulationInfo->realParameter[1769] /* theta[263] PARAM */))) * (((data->simulationInfo->realParameter[1268] /* r_init[263] PARAM */)) * ((data->simulationInfo->realParameter[767] /* omega_c[263] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10622(DATA *data, threadData_t *threadData);


/*
equation index: 4207
type: SIMPLE_ASSIGN
vz[263] = 0.0
*/
void SpiralGalaxy_eqFunction_4207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4207};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1262]] /* vz[263] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10621(DATA *data, threadData_t *threadData);


/*
equation index: 4209
type: SIMPLE_ASSIGN
z[264] = 0.0022400000000000007
*/
void SpiralGalaxy_eqFunction_4209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4209};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2763]] /* z[264] STATE(1,vz[264]) */) = 0.0022400000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10634(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10635(DATA *data, threadData_t *threadData);


/*
equation index: 4212
type: SIMPLE_ASSIGN
y[264] = r_init[264] * sin(theta[264] + 5.600000000000005e-4)
*/
void SpiralGalaxy_eqFunction_4212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4212};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2263]] /* y[264] STATE(1,vy[264]) */) = ((data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */)) * (sin((data->simulationInfo->realParameter[1770] /* theta[264] PARAM */) + 5.600000000000005e-4));
  TRACE_POP
}

/*
equation index: 4213
type: SIMPLE_ASSIGN
x[264] = r_init[264] * cos(theta[264] + 5.600000000000005e-4)
*/
void SpiralGalaxy_eqFunction_4213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1763]] /* x[264] STATE(1,vx[264]) */) = ((data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */)) * (cos((data->simulationInfo->realParameter[1770] /* theta[264] PARAM */) + 5.600000000000005e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10636(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10637(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10640(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10639(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10638(DATA *data, threadData_t *threadData);


/*
equation index: 4219
type: SIMPLE_ASSIGN
vx[264] = (-sin(theta[264])) * r_init[264] * omega_c[264]
*/
void SpiralGalaxy_eqFunction_4219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4219};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[263]] /* vx[264] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1770] /* theta[264] PARAM */)))) * (((data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */)) * ((data->simulationInfo->realParameter[768] /* omega_c[264] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10633(DATA *data, threadData_t *threadData);


/*
equation index: 4221
type: SIMPLE_ASSIGN
vy[264] = cos(theta[264]) * r_init[264] * omega_c[264]
*/
void SpiralGalaxy_eqFunction_4221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[763]] /* vy[264] STATE(1) */) = (cos((data->simulationInfo->realParameter[1770] /* theta[264] PARAM */))) * (((data->simulationInfo->realParameter[1269] /* r_init[264] PARAM */)) * ((data->simulationInfo->realParameter[768] /* omega_c[264] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10632(DATA *data, threadData_t *threadData);


/*
equation index: 4223
type: SIMPLE_ASSIGN
vz[264] = 0.0
*/
void SpiralGalaxy_eqFunction_4223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4223};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1263]] /* vz[264] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10631(DATA *data, threadData_t *threadData);


/*
equation index: 4225
type: SIMPLE_ASSIGN
z[265] = 0.0024000000000000007
*/
void SpiralGalaxy_eqFunction_4225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4225};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2764]] /* z[265] STATE(1,vz[265]) */) = 0.0024000000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10644(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10645(DATA *data, threadData_t *threadData);


/*
equation index: 4228
type: SIMPLE_ASSIGN
y[265] = r_init[265] * sin(theta[265] + 6.000000000000006e-4)
*/
void SpiralGalaxy_eqFunction_4228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4228};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2264]] /* y[265] STATE(1,vy[265]) */) = ((data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */)) * (sin((data->simulationInfo->realParameter[1771] /* theta[265] PARAM */) + 6.000000000000006e-4));
  TRACE_POP
}

/*
equation index: 4229
type: SIMPLE_ASSIGN
x[265] = r_init[265] * cos(theta[265] + 6.000000000000006e-4)
*/
void SpiralGalaxy_eqFunction_4229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1764]] /* x[265] STATE(1,vx[265]) */) = ((data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */)) * (cos((data->simulationInfo->realParameter[1771] /* theta[265] PARAM */) + 6.000000000000006e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10646(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10647(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10650(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10649(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10648(DATA *data, threadData_t *threadData);


/*
equation index: 4235
type: SIMPLE_ASSIGN
vx[265] = (-sin(theta[265])) * r_init[265] * omega_c[265]
*/
void SpiralGalaxy_eqFunction_4235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4235};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[264]] /* vx[265] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1771] /* theta[265] PARAM */)))) * (((data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */)) * ((data->simulationInfo->realParameter[769] /* omega_c[265] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10643(DATA *data, threadData_t *threadData);


/*
equation index: 4237
type: SIMPLE_ASSIGN
vy[265] = cos(theta[265]) * r_init[265] * omega_c[265]
*/
void SpiralGalaxy_eqFunction_4237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[764]] /* vy[265] STATE(1) */) = (cos((data->simulationInfo->realParameter[1771] /* theta[265] PARAM */))) * (((data->simulationInfo->realParameter[1270] /* r_init[265] PARAM */)) * ((data->simulationInfo->realParameter[769] /* omega_c[265] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10642(DATA *data, threadData_t *threadData);


/*
equation index: 4239
type: SIMPLE_ASSIGN
vz[265] = 0.0
*/
void SpiralGalaxy_eqFunction_4239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4239};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1264]] /* vz[265] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10641(DATA *data, threadData_t *threadData);


/*
equation index: 4241
type: SIMPLE_ASSIGN
z[266] = 0.0025600000000000006
*/
void SpiralGalaxy_eqFunction_4241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4241};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2765]] /* z[266] STATE(1,vz[266]) */) = 0.0025600000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10654(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10655(DATA *data, threadData_t *threadData);


/*
equation index: 4244
type: SIMPLE_ASSIGN
y[266] = r_init[266] * sin(theta[266] + 6.400000000000006e-4)
*/
void SpiralGalaxy_eqFunction_4244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4244};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2265]] /* y[266] STATE(1,vy[266]) */) = ((data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */)) * (sin((data->simulationInfo->realParameter[1772] /* theta[266] PARAM */) + 6.400000000000006e-4));
  TRACE_POP
}

/*
equation index: 4245
type: SIMPLE_ASSIGN
x[266] = r_init[266] * cos(theta[266] + 6.400000000000006e-4)
*/
void SpiralGalaxy_eqFunction_4245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1765]] /* x[266] STATE(1,vx[266]) */) = ((data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */)) * (cos((data->simulationInfo->realParameter[1772] /* theta[266] PARAM */) + 6.400000000000006e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10656(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10657(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10660(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10659(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10658(DATA *data, threadData_t *threadData);


/*
equation index: 4251
type: SIMPLE_ASSIGN
vx[266] = (-sin(theta[266])) * r_init[266] * omega_c[266]
*/
void SpiralGalaxy_eqFunction_4251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4251};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[265]] /* vx[266] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1772] /* theta[266] PARAM */)))) * (((data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */)) * ((data->simulationInfo->realParameter[770] /* omega_c[266] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10653(DATA *data, threadData_t *threadData);


/*
equation index: 4253
type: SIMPLE_ASSIGN
vy[266] = cos(theta[266]) * r_init[266] * omega_c[266]
*/
void SpiralGalaxy_eqFunction_4253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[765]] /* vy[266] STATE(1) */) = (cos((data->simulationInfo->realParameter[1772] /* theta[266] PARAM */))) * (((data->simulationInfo->realParameter[1271] /* r_init[266] PARAM */)) * ((data->simulationInfo->realParameter[770] /* omega_c[266] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10652(DATA *data, threadData_t *threadData);


/*
equation index: 4255
type: SIMPLE_ASSIGN
vz[266] = 0.0
*/
void SpiralGalaxy_eqFunction_4255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4255};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1265]] /* vz[266] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10651(DATA *data, threadData_t *threadData);


/*
equation index: 4257
type: SIMPLE_ASSIGN
z[267] = 0.0027200000000000006
*/
void SpiralGalaxy_eqFunction_4257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4257};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2766]] /* z[267] STATE(1,vz[267]) */) = 0.0027200000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10664(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10665(DATA *data, threadData_t *threadData);


/*
equation index: 4260
type: SIMPLE_ASSIGN
y[267] = r_init[267] * sin(theta[267] + 6.800000000000006e-4)
*/
void SpiralGalaxy_eqFunction_4260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4260};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2266]] /* y[267] STATE(1,vy[267]) */) = ((data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */)) * (sin((data->simulationInfo->realParameter[1773] /* theta[267] PARAM */) + 6.800000000000006e-4));
  TRACE_POP
}

/*
equation index: 4261
type: SIMPLE_ASSIGN
x[267] = r_init[267] * cos(theta[267] + 6.800000000000006e-4)
*/
void SpiralGalaxy_eqFunction_4261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1766]] /* x[267] STATE(1,vx[267]) */) = ((data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */)) * (cos((data->simulationInfo->realParameter[1773] /* theta[267] PARAM */) + 6.800000000000006e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10666(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10667(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10670(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10669(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10668(DATA *data, threadData_t *threadData);


/*
equation index: 4267
type: SIMPLE_ASSIGN
vx[267] = (-sin(theta[267])) * r_init[267] * omega_c[267]
*/
void SpiralGalaxy_eqFunction_4267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4267};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[266]] /* vx[267] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1773] /* theta[267] PARAM */)))) * (((data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */)) * ((data->simulationInfo->realParameter[771] /* omega_c[267] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10663(DATA *data, threadData_t *threadData);


/*
equation index: 4269
type: SIMPLE_ASSIGN
vy[267] = cos(theta[267]) * r_init[267] * omega_c[267]
*/
void SpiralGalaxy_eqFunction_4269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[766]] /* vy[267] STATE(1) */) = (cos((data->simulationInfo->realParameter[1773] /* theta[267] PARAM */))) * (((data->simulationInfo->realParameter[1272] /* r_init[267] PARAM */)) * ((data->simulationInfo->realParameter[771] /* omega_c[267] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10662(DATA *data, threadData_t *threadData);


/*
equation index: 4271
type: SIMPLE_ASSIGN
vz[267] = 0.0
*/
void SpiralGalaxy_eqFunction_4271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1266]] /* vz[267] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10661(DATA *data, threadData_t *threadData);


/*
equation index: 4273
type: SIMPLE_ASSIGN
z[268] = 0.002880000000000001
*/
void SpiralGalaxy_eqFunction_4273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2767]] /* z[268] STATE(1,vz[268]) */) = 0.002880000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10674(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10675(DATA *data, threadData_t *threadData);


/*
equation index: 4276
type: SIMPLE_ASSIGN
y[268] = r_init[268] * sin(theta[268] + 7.200000000000007e-4)
*/
void SpiralGalaxy_eqFunction_4276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4276};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2267]] /* y[268] STATE(1,vy[268]) */) = ((data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */)) * (sin((data->simulationInfo->realParameter[1774] /* theta[268] PARAM */) + 7.200000000000007e-4));
  TRACE_POP
}

/*
equation index: 4277
type: SIMPLE_ASSIGN
x[268] = r_init[268] * cos(theta[268] + 7.200000000000007e-4)
*/
void SpiralGalaxy_eqFunction_4277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1767]] /* x[268] STATE(1,vx[268]) */) = ((data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */)) * (cos((data->simulationInfo->realParameter[1774] /* theta[268] PARAM */) + 7.200000000000007e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10676(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10677(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10680(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10679(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10678(DATA *data, threadData_t *threadData);


/*
equation index: 4283
type: SIMPLE_ASSIGN
vx[268] = (-sin(theta[268])) * r_init[268] * omega_c[268]
*/
void SpiralGalaxy_eqFunction_4283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4283};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[267]] /* vx[268] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1774] /* theta[268] PARAM */)))) * (((data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */)) * ((data->simulationInfo->realParameter[772] /* omega_c[268] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10673(DATA *data, threadData_t *threadData);


/*
equation index: 4285
type: SIMPLE_ASSIGN
vy[268] = cos(theta[268]) * r_init[268] * omega_c[268]
*/
void SpiralGalaxy_eqFunction_4285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[767]] /* vy[268] STATE(1) */) = (cos((data->simulationInfo->realParameter[1774] /* theta[268] PARAM */))) * (((data->simulationInfo->realParameter[1273] /* r_init[268] PARAM */)) * ((data->simulationInfo->realParameter[772] /* omega_c[268] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10672(DATA *data, threadData_t *threadData);


/*
equation index: 4287
type: SIMPLE_ASSIGN
vz[268] = 0.0
*/
void SpiralGalaxy_eqFunction_4287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4287};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1267]] /* vz[268] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10671(DATA *data, threadData_t *threadData);


/*
equation index: 4289
type: SIMPLE_ASSIGN
z[269] = 0.00304
*/
void SpiralGalaxy_eqFunction_4289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4289};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2768]] /* z[269] STATE(1,vz[269]) */) = 0.00304;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10684(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10685(DATA *data, threadData_t *threadData);


/*
equation index: 4292
type: SIMPLE_ASSIGN
y[269] = r_init[269] * sin(theta[269] + 7.600000000000007e-4)
*/
void SpiralGalaxy_eqFunction_4292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4292};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2268]] /* y[269] STATE(1,vy[269]) */) = ((data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */)) * (sin((data->simulationInfo->realParameter[1775] /* theta[269] PARAM */) + 7.600000000000007e-4));
  TRACE_POP
}

/*
equation index: 4293
type: SIMPLE_ASSIGN
x[269] = r_init[269] * cos(theta[269] + 7.600000000000007e-4)
*/
void SpiralGalaxy_eqFunction_4293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1768]] /* x[269] STATE(1,vx[269]) */) = ((data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */)) * (cos((data->simulationInfo->realParameter[1775] /* theta[269] PARAM */) + 7.600000000000007e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10686(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10687(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10690(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10689(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10688(DATA *data, threadData_t *threadData);


/*
equation index: 4299
type: SIMPLE_ASSIGN
vx[269] = (-sin(theta[269])) * r_init[269] * omega_c[269]
*/
void SpiralGalaxy_eqFunction_4299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4299};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[268]] /* vx[269] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1775] /* theta[269] PARAM */)))) * (((data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */)) * ((data->simulationInfo->realParameter[773] /* omega_c[269] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10683(DATA *data, threadData_t *threadData);


/*
equation index: 4301
type: SIMPLE_ASSIGN
vy[269] = cos(theta[269]) * r_init[269] * omega_c[269]
*/
void SpiralGalaxy_eqFunction_4301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[768]] /* vy[269] STATE(1) */) = (cos((data->simulationInfo->realParameter[1775] /* theta[269] PARAM */))) * (((data->simulationInfo->realParameter[1274] /* r_init[269] PARAM */)) * ((data->simulationInfo->realParameter[773] /* omega_c[269] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10682(DATA *data, threadData_t *threadData);


/*
equation index: 4303
type: SIMPLE_ASSIGN
vz[269] = 0.0
*/
void SpiralGalaxy_eqFunction_4303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4303};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1268]] /* vz[269] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10681(DATA *data, threadData_t *threadData);


/*
equation index: 4305
type: SIMPLE_ASSIGN
z[270] = 0.0032000000000000006
*/
void SpiralGalaxy_eqFunction_4305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4305};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2769]] /* z[270] STATE(1,vz[270]) */) = 0.0032000000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10694(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10695(DATA *data, threadData_t *threadData);


/*
equation index: 4308
type: SIMPLE_ASSIGN
y[270] = r_init[270] * sin(theta[270] + 8.000000000000007e-4)
*/
void SpiralGalaxy_eqFunction_4308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4308};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2269]] /* y[270] STATE(1,vy[270]) */) = ((data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */)) * (sin((data->simulationInfo->realParameter[1776] /* theta[270] PARAM */) + 8.000000000000007e-4));
  TRACE_POP
}

/*
equation index: 4309
type: SIMPLE_ASSIGN
x[270] = r_init[270] * cos(theta[270] + 8.000000000000007e-4)
*/
void SpiralGalaxy_eqFunction_4309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1769]] /* x[270] STATE(1,vx[270]) */) = ((data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */)) * (cos((data->simulationInfo->realParameter[1776] /* theta[270] PARAM */) + 8.000000000000007e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10696(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10697(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10700(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10699(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10698(DATA *data, threadData_t *threadData);


/*
equation index: 4315
type: SIMPLE_ASSIGN
vx[270] = (-sin(theta[270])) * r_init[270] * omega_c[270]
*/
void SpiralGalaxy_eqFunction_4315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4315};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[269]] /* vx[270] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1776] /* theta[270] PARAM */)))) * (((data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */)) * ((data->simulationInfo->realParameter[774] /* omega_c[270] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10693(DATA *data, threadData_t *threadData);


/*
equation index: 4317
type: SIMPLE_ASSIGN
vy[270] = cos(theta[270]) * r_init[270] * omega_c[270]
*/
void SpiralGalaxy_eqFunction_4317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[769]] /* vy[270] STATE(1) */) = (cos((data->simulationInfo->realParameter[1776] /* theta[270] PARAM */))) * (((data->simulationInfo->realParameter[1275] /* r_init[270] PARAM */)) * ((data->simulationInfo->realParameter[774] /* omega_c[270] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10692(DATA *data, threadData_t *threadData);


/*
equation index: 4319
type: SIMPLE_ASSIGN
vz[270] = 0.0
*/
void SpiralGalaxy_eqFunction_4319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1269]] /* vz[270] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10691(DATA *data, threadData_t *threadData);


/*
equation index: 4321
type: SIMPLE_ASSIGN
z[271] = 0.0033600000000000006
*/
void SpiralGalaxy_eqFunction_4321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2770]] /* z[271] STATE(1,vz[271]) */) = 0.0033600000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10704(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10705(DATA *data, threadData_t *threadData);


/*
equation index: 4324
type: SIMPLE_ASSIGN
y[271] = r_init[271] * sin(theta[271] + 8.400000000000008e-4)
*/
void SpiralGalaxy_eqFunction_4324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4324};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2270]] /* y[271] STATE(1,vy[271]) */) = ((data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */)) * (sin((data->simulationInfo->realParameter[1777] /* theta[271] PARAM */) + 8.400000000000008e-4));
  TRACE_POP
}

/*
equation index: 4325
type: SIMPLE_ASSIGN
x[271] = r_init[271] * cos(theta[271] + 8.400000000000008e-4)
*/
void SpiralGalaxy_eqFunction_4325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1770]] /* x[271] STATE(1,vx[271]) */) = ((data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */)) * (cos((data->simulationInfo->realParameter[1777] /* theta[271] PARAM */) + 8.400000000000008e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10706(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10707(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10710(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10709(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10708(DATA *data, threadData_t *threadData);


/*
equation index: 4331
type: SIMPLE_ASSIGN
vx[271] = (-sin(theta[271])) * r_init[271] * omega_c[271]
*/
void SpiralGalaxy_eqFunction_4331(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4331};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[270]] /* vx[271] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1777] /* theta[271] PARAM */)))) * (((data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */)) * ((data->simulationInfo->realParameter[775] /* omega_c[271] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10703(DATA *data, threadData_t *threadData);


/*
equation index: 4333
type: SIMPLE_ASSIGN
vy[271] = cos(theta[271]) * r_init[271] * omega_c[271]
*/
void SpiralGalaxy_eqFunction_4333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[770]] /* vy[271] STATE(1) */) = (cos((data->simulationInfo->realParameter[1777] /* theta[271] PARAM */))) * (((data->simulationInfo->realParameter[1276] /* r_init[271] PARAM */)) * ((data->simulationInfo->realParameter[775] /* omega_c[271] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10702(DATA *data, threadData_t *threadData);


/*
equation index: 4335
type: SIMPLE_ASSIGN
vz[271] = 0.0
*/
void SpiralGalaxy_eqFunction_4335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1270]] /* vz[271] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10701(DATA *data, threadData_t *threadData);


/*
equation index: 4337
type: SIMPLE_ASSIGN
z[272] = 0.0035200000000000006
*/
void SpiralGalaxy_eqFunction_4337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2771]] /* z[272] STATE(1,vz[272]) */) = 0.0035200000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10714(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10715(DATA *data, threadData_t *threadData);


/*
equation index: 4340
type: SIMPLE_ASSIGN
y[272] = r_init[272] * sin(theta[272] + 8.800000000000008e-4)
*/
void SpiralGalaxy_eqFunction_4340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4340};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2271]] /* y[272] STATE(1,vy[272]) */) = ((data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */)) * (sin((data->simulationInfo->realParameter[1778] /* theta[272] PARAM */) + 8.800000000000008e-4));
  TRACE_POP
}

/*
equation index: 4341
type: SIMPLE_ASSIGN
x[272] = r_init[272] * cos(theta[272] + 8.800000000000008e-4)
*/
void SpiralGalaxy_eqFunction_4341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1771]] /* x[272] STATE(1,vx[272]) */) = ((data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */)) * (cos((data->simulationInfo->realParameter[1778] /* theta[272] PARAM */) + 8.800000000000008e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10716(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10717(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10720(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10719(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10718(DATA *data, threadData_t *threadData);


/*
equation index: 4347
type: SIMPLE_ASSIGN
vx[272] = (-sin(theta[272])) * r_init[272] * omega_c[272]
*/
void SpiralGalaxy_eqFunction_4347(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4347};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[271]] /* vx[272] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1778] /* theta[272] PARAM */)))) * (((data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */)) * ((data->simulationInfo->realParameter[776] /* omega_c[272] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10713(DATA *data, threadData_t *threadData);


/*
equation index: 4349
type: SIMPLE_ASSIGN
vy[272] = cos(theta[272]) * r_init[272] * omega_c[272]
*/
void SpiralGalaxy_eqFunction_4349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[771]] /* vy[272] STATE(1) */) = (cos((data->simulationInfo->realParameter[1778] /* theta[272] PARAM */))) * (((data->simulationInfo->realParameter[1277] /* r_init[272] PARAM */)) * ((data->simulationInfo->realParameter[776] /* omega_c[272] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10712(DATA *data, threadData_t *threadData);


/*
equation index: 4351
type: SIMPLE_ASSIGN
vz[272] = 0.0
*/
void SpiralGalaxy_eqFunction_4351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4351};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1271]] /* vz[272] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10711(DATA *data, threadData_t *threadData);


/*
equation index: 4353
type: SIMPLE_ASSIGN
z[273] = 0.0036800000000000005
*/
void SpiralGalaxy_eqFunction_4353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4353};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2772]] /* z[273] STATE(1,vz[273]) */) = 0.0036800000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10724(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10725(DATA *data, threadData_t *threadData);


/*
equation index: 4356
type: SIMPLE_ASSIGN
y[273] = r_init[273] * sin(theta[273] + 9.200000000000008e-4)
*/
void SpiralGalaxy_eqFunction_4356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4356};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2272]] /* y[273] STATE(1,vy[273]) */) = ((data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */)) * (sin((data->simulationInfo->realParameter[1779] /* theta[273] PARAM */) + 9.200000000000008e-4));
  TRACE_POP
}

/*
equation index: 4357
type: SIMPLE_ASSIGN
x[273] = r_init[273] * cos(theta[273] + 9.200000000000008e-4)
*/
void SpiralGalaxy_eqFunction_4357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1772]] /* x[273] STATE(1,vx[273]) */) = ((data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */)) * (cos((data->simulationInfo->realParameter[1779] /* theta[273] PARAM */) + 9.200000000000008e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10726(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10727(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10730(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10729(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10728(DATA *data, threadData_t *threadData);


/*
equation index: 4363
type: SIMPLE_ASSIGN
vx[273] = (-sin(theta[273])) * r_init[273] * omega_c[273]
*/
void SpiralGalaxy_eqFunction_4363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4363};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[272]] /* vx[273] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1779] /* theta[273] PARAM */)))) * (((data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */)) * ((data->simulationInfo->realParameter[777] /* omega_c[273] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10723(DATA *data, threadData_t *threadData);


/*
equation index: 4365
type: SIMPLE_ASSIGN
vy[273] = cos(theta[273]) * r_init[273] * omega_c[273]
*/
void SpiralGalaxy_eqFunction_4365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[772]] /* vy[273] STATE(1) */) = (cos((data->simulationInfo->realParameter[1779] /* theta[273] PARAM */))) * (((data->simulationInfo->realParameter[1278] /* r_init[273] PARAM */)) * ((data->simulationInfo->realParameter[777] /* omega_c[273] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10722(DATA *data, threadData_t *threadData);


/*
equation index: 4367
type: SIMPLE_ASSIGN
vz[273] = 0.0
*/
void SpiralGalaxy_eqFunction_4367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4367};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1272]] /* vz[273] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10721(DATA *data, threadData_t *threadData);


/*
equation index: 4369
type: SIMPLE_ASSIGN
z[274] = 0.0038400000000000005
*/
void SpiralGalaxy_eqFunction_4369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4369};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2773]] /* z[274] STATE(1,vz[274]) */) = 0.0038400000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10734(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10735(DATA *data, threadData_t *threadData);


/*
equation index: 4372
type: SIMPLE_ASSIGN
y[274] = r_init[274] * sin(theta[274] + 9.600000000000009e-4)
*/
void SpiralGalaxy_eqFunction_4372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4372};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2273]] /* y[274] STATE(1,vy[274]) */) = ((data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */)) * (sin((data->simulationInfo->realParameter[1780] /* theta[274] PARAM */) + 9.600000000000009e-4));
  TRACE_POP
}

/*
equation index: 4373
type: SIMPLE_ASSIGN
x[274] = r_init[274] * cos(theta[274] + 9.600000000000009e-4)
*/
void SpiralGalaxy_eqFunction_4373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1773]] /* x[274] STATE(1,vx[274]) */) = ((data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */)) * (cos((data->simulationInfo->realParameter[1780] /* theta[274] PARAM */) + 9.600000000000009e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10736(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10737(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10740(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10739(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10738(DATA *data, threadData_t *threadData);


/*
equation index: 4379
type: SIMPLE_ASSIGN
vx[274] = (-sin(theta[274])) * r_init[274] * omega_c[274]
*/
void SpiralGalaxy_eqFunction_4379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4379};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[273]] /* vx[274] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1780] /* theta[274] PARAM */)))) * (((data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */)) * ((data->simulationInfo->realParameter[778] /* omega_c[274] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10733(DATA *data, threadData_t *threadData);


/*
equation index: 4381
type: SIMPLE_ASSIGN
vy[274] = cos(theta[274]) * r_init[274] * omega_c[274]
*/
void SpiralGalaxy_eqFunction_4381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[773]] /* vy[274] STATE(1) */) = (cos((data->simulationInfo->realParameter[1780] /* theta[274] PARAM */))) * (((data->simulationInfo->realParameter[1279] /* r_init[274] PARAM */)) * ((data->simulationInfo->realParameter[778] /* omega_c[274] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10732(DATA *data, threadData_t *threadData);


/*
equation index: 4383
type: SIMPLE_ASSIGN
vz[274] = 0.0
*/
void SpiralGalaxy_eqFunction_4383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4383};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1273]] /* vz[274] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10731(DATA *data, threadData_t *threadData);


/*
equation index: 4385
type: SIMPLE_ASSIGN
z[275] = 0.004000000000000001
*/
void SpiralGalaxy_eqFunction_4385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4385};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2774]] /* z[275] STATE(1,vz[275]) */) = 0.004000000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10744(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10745(DATA *data, threadData_t *threadData);


/*
equation index: 4388
type: SIMPLE_ASSIGN
y[275] = r_init[275] * sin(theta[275] + 0.0010000000000000009)
*/
void SpiralGalaxy_eqFunction_4388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4388};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2274]] /* y[275] STATE(1,vy[275]) */) = ((data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */)) * (sin((data->simulationInfo->realParameter[1781] /* theta[275] PARAM */) + 0.0010000000000000009));
  TRACE_POP
}

/*
equation index: 4389
type: SIMPLE_ASSIGN
x[275] = r_init[275] * cos(theta[275] + 0.0010000000000000009)
*/
void SpiralGalaxy_eqFunction_4389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1774]] /* x[275] STATE(1,vx[275]) */) = ((data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */)) * (cos((data->simulationInfo->realParameter[1781] /* theta[275] PARAM */) + 0.0010000000000000009));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10746(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10747(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10750(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10749(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10748(DATA *data, threadData_t *threadData);


/*
equation index: 4395
type: SIMPLE_ASSIGN
vx[275] = (-sin(theta[275])) * r_init[275] * omega_c[275]
*/
void SpiralGalaxy_eqFunction_4395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4395};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[274]] /* vx[275] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1781] /* theta[275] PARAM */)))) * (((data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */)) * ((data->simulationInfo->realParameter[779] /* omega_c[275] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10743(DATA *data, threadData_t *threadData);


/*
equation index: 4397
type: SIMPLE_ASSIGN
vy[275] = cos(theta[275]) * r_init[275] * omega_c[275]
*/
void SpiralGalaxy_eqFunction_4397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[774]] /* vy[275] STATE(1) */) = (cos((data->simulationInfo->realParameter[1781] /* theta[275] PARAM */))) * (((data->simulationInfo->realParameter[1280] /* r_init[275] PARAM */)) * ((data->simulationInfo->realParameter[779] /* omega_c[275] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10742(DATA *data, threadData_t *threadData);


/*
equation index: 4399
type: SIMPLE_ASSIGN
vz[275] = 0.0
*/
void SpiralGalaxy_eqFunction_4399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4399};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1274]] /* vz[275] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10741(DATA *data, threadData_t *threadData);


/*
equation index: 4401
type: SIMPLE_ASSIGN
z[276] = 0.0041600000000000005
*/
void SpiralGalaxy_eqFunction_4401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4401};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2775]] /* z[276] STATE(1,vz[276]) */) = 0.0041600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10754(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10755(DATA *data, threadData_t *threadData);


/*
equation index: 4404
type: SIMPLE_ASSIGN
y[276] = r_init[276] * sin(theta[276] + 0.001040000000000001)
*/
void SpiralGalaxy_eqFunction_4404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4404};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2275]] /* y[276] STATE(1,vy[276]) */) = ((data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */)) * (sin((data->simulationInfo->realParameter[1782] /* theta[276] PARAM */) + 0.001040000000000001));
  TRACE_POP
}

/*
equation index: 4405
type: SIMPLE_ASSIGN
x[276] = r_init[276] * cos(theta[276] + 0.001040000000000001)
*/
void SpiralGalaxy_eqFunction_4405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1775]] /* x[276] STATE(1,vx[276]) */) = ((data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */)) * (cos((data->simulationInfo->realParameter[1782] /* theta[276] PARAM */) + 0.001040000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10756(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10757(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10760(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10759(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10758(DATA *data, threadData_t *threadData);


/*
equation index: 4411
type: SIMPLE_ASSIGN
vx[276] = (-sin(theta[276])) * r_init[276] * omega_c[276]
*/
void SpiralGalaxy_eqFunction_4411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4411};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[275]] /* vx[276] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1782] /* theta[276] PARAM */)))) * (((data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */)) * ((data->simulationInfo->realParameter[780] /* omega_c[276] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10753(DATA *data, threadData_t *threadData);


/*
equation index: 4413
type: SIMPLE_ASSIGN
vy[276] = cos(theta[276]) * r_init[276] * omega_c[276]
*/
void SpiralGalaxy_eqFunction_4413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[775]] /* vy[276] STATE(1) */) = (cos((data->simulationInfo->realParameter[1782] /* theta[276] PARAM */))) * (((data->simulationInfo->realParameter[1281] /* r_init[276] PARAM */)) * ((data->simulationInfo->realParameter[780] /* omega_c[276] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10752(DATA *data, threadData_t *threadData);


/*
equation index: 4415
type: SIMPLE_ASSIGN
vz[276] = 0.0
*/
void SpiralGalaxy_eqFunction_4415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4415};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1275]] /* vz[276] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10751(DATA *data, threadData_t *threadData);


/*
equation index: 4417
type: SIMPLE_ASSIGN
z[277] = 0.004320000000000001
*/
void SpiralGalaxy_eqFunction_4417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4417};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2776]] /* z[277] STATE(1,vz[277]) */) = 0.004320000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10764(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10765(DATA *data, threadData_t *threadData);


/*
equation index: 4420
type: SIMPLE_ASSIGN
y[277] = r_init[277] * sin(theta[277] + 0.0010800000000000009)
*/
void SpiralGalaxy_eqFunction_4420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4420};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2276]] /* y[277] STATE(1,vy[277]) */) = ((data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */)) * (sin((data->simulationInfo->realParameter[1783] /* theta[277] PARAM */) + 0.0010800000000000009));
  TRACE_POP
}

/*
equation index: 4421
type: SIMPLE_ASSIGN
x[277] = r_init[277] * cos(theta[277] + 0.0010800000000000009)
*/
void SpiralGalaxy_eqFunction_4421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1776]] /* x[277] STATE(1,vx[277]) */) = ((data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */)) * (cos((data->simulationInfo->realParameter[1783] /* theta[277] PARAM */) + 0.0010800000000000009));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10766(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10767(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10770(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10769(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10768(DATA *data, threadData_t *threadData);


/*
equation index: 4427
type: SIMPLE_ASSIGN
vx[277] = (-sin(theta[277])) * r_init[277] * omega_c[277]
*/
void SpiralGalaxy_eqFunction_4427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4427};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[276]] /* vx[277] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1783] /* theta[277] PARAM */)))) * (((data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */)) * ((data->simulationInfo->realParameter[781] /* omega_c[277] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10763(DATA *data, threadData_t *threadData);


/*
equation index: 4429
type: SIMPLE_ASSIGN
vy[277] = cos(theta[277]) * r_init[277] * omega_c[277]
*/
void SpiralGalaxy_eqFunction_4429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[776]] /* vy[277] STATE(1) */) = (cos((data->simulationInfo->realParameter[1783] /* theta[277] PARAM */))) * (((data->simulationInfo->realParameter[1282] /* r_init[277] PARAM */)) * ((data->simulationInfo->realParameter[781] /* omega_c[277] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10762(DATA *data, threadData_t *threadData);


/*
equation index: 4431
type: SIMPLE_ASSIGN
vz[277] = 0.0
*/
void SpiralGalaxy_eqFunction_4431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4431};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1276]] /* vz[277] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10761(DATA *data, threadData_t *threadData);


/*
equation index: 4433
type: SIMPLE_ASSIGN
z[278] = 0.004480000000000001
*/
void SpiralGalaxy_eqFunction_4433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4433};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2777]] /* z[278] STATE(1,vz[278]) */) = 0.004480000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10774(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10775(DATA *data, threadData_t *threadData);


/*
equation index: 4436
type: SIMPLE_ASSIGN
y[278] = r_init[278] * sin(theta[278] + 0.001120000000000001)
*/
void SpiralGalaxy_eqFunction_4436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4436};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2277]] /* y[278] STATE(1,vy[278]) */) = ((data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */)) * (sin((data->simulationInfo->realParameter[1784] /* theta[278] PARAM */) + 0.001120000000000001));
  TRACE_POP
}

/*
equation index: 4437
type: SIMPLE_ASSIGN
x[278] = r_init[278] * cos(theta[278] + 0.001120000000000001)
*/
void SpiralGalaxy_eqFunction_4437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1777]] /* x[278] STATE(1,vx[278]) */) = ((data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */)) * (cos((data->simulationInfo->realParameter[1784] /* theta[278] PARAM */) + 0.001120000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10776(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10777(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10780(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10779(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10778(DATA *data, threadData_t *threadData);


/*
equation index: 4443
type: SIMPLE_ASSIGN
vx[278] = (-sin(theta[278])) * r_init[278] * omega_c[278]
*/
void SpiralGalaxy_eqFunction_4443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4443};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[277]] /* vx[278] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1784] /* theta[278] PARAM */)))) * (((data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */)) * ((data->simulationInfo->realParameter[782] /* omega_c[278] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10773(DATA *data, threadData_t *threadData);


/*
equation index: 4445
type: SIMPLE_ASSIGN
vy[278] = cos(theta[278]) * r_init[278] * omega_c[278]
*/
void SpiralGalaxy_eqFunction_4445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[777]] /* vy[278] STATE(1) */) = (cos((data->simulationInfo->realParameter[1784] /* theta[278] PARAM */))) * (((data->simulationInfo->realParameter[1283] /* r_init[278] PARAM */)) * ((data->simulationInfo->realParameter[782] /* omega_c[278] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10772(DATA *data, threadData_t *threadData);


/*
equation index: 4447
type: SIMPLE_ASSIGN
vz[278] = 0.0
*/
void SpiralGalaxy_eqFunction_4447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4447};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1277]] /* vz[278] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10771(DATA *data, threadData_t *threadData);


/*
equation index: 4449
type: SIMPLE_ASSIGN
z[279] = 0.004640000000000001
*/
void SpiralGalaxy_eqFunction_4449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4449};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2778]] /* z[279] STATE(1,vz[279]) */) = 0.004640000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10784(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10785(DATA *data, threadData_t *threadData);


/*
equation index: 4452
type: SIMPLE_ASSIGN
y[279] = r_init[279] * sin(theta[279] + 0.001160000000000001)
*/
void SpiralGalaxy_eqFunction_4452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4452};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2278]] /* y[279] STATE(1,vy[279]) */) = ((data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */)) * (sin((data->simulationInfo->realParameter[1785] /* theta[279] PARAM */) + 0.001160000000000001));
  TRACE_POP
}

/*
equation index: 4453
type: SIMPLE_ASSIGN
x[279] = r_init[279] * cos(theta[279] + 0.001160000000000001)
*/
void SpiralGalaxy_eqFunction_4453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1778]] /* x[279] STATE(1,vx[279]) */) = ((data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */)) * (cos((data->simulationInfo->realParameter[1785] /* theta[279] PARAM */) + 0.001160000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10786(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10787(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10790(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10789(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10788(DATA *data, threadData_t *threadData);


/*
equation index: 4459
type: SIMPLE_ASSIGN
vx[279] = (-sin(theta[279])) * r_init[279] * omega_c[279]
*/
void SpiralGalaxy_eqFunction_4459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4459};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[278]] /* vx[279] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1785] /* theta[279] PARAM */)))) * (((data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */)) * ((data->simulationInfo->realParameter[783] /* omega_c[279] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10783(DATA *data, threadData_t *threadData);


/*
equation index: 4461
type: SIMPLE_ASSIGN
vy[279] = cos(theta[279]) * r_init[279] * omega_c[279]
*/
void SpiralGalaxy_eqFunction_4461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[778]] /* vy[279] STATE(1) */) = (cos((data->simulationInfo->realParameter[1785] /* theta[279] PARAM */))) * (((data->simulationInfo->realParameter[1284] /* r_init[279] PARAM */)) * ((data->simulationInfo->realParameter[783] /* omega_c[279] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10782(DATA *data, threadData_t *threadData);


/*
equation index: 4463
type: SIMPLE_ASSIGN
vz[279] = 0.0
*/
void SpiralGalaxy_eqFunction_4463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4463};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1278]] /* vz[279] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10781(DATA *data, threadData_t *threadData);


/*
equation index: 4465
type: SIMPLE_ASSIGN
z[280] = 0.004800000000000001
*/
void SpiralGalaxy_eqFunction_4465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4465};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2779]] /* z[280] STATE(1,vz[280]) */) = 0.004800000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10794(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10795(DATA *data, threadData_t *threadData);


/*
equation index: 4468
type: SIMPLE_ASSIGN
y[280] = r_init[280] * sin(theta[280] + 0.0012000000000000012)
*/
void SpiralGalaxy_eqFunction_4468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4468};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2279]] /* y[280] STATE(1,vy[280]) */) = ((data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */)) * (sin((data->simulationInfo->realParameter[1786] /* theta[280] PARAM */) + 0.0012000000000000012));
  TRACE_POP
}

/*
equation index: 4469
type: SIMPLE_ASSIGN
x[280] = r_init[280] * cos(theta[280] + 0.0012000000000000012)
*/
void SpiralGalaxy_eqFunction_4469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1779]] /* x[280] STATE(1,vx[280]) */) = ((data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */)) * (cos((data->simulationInfo->realParameter[1786] /* theta[280] PARAM */) + 0.0012000000000000012));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10796(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10797(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10800(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10799(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10798(DATA *data, threadData_t *threadData);


/*
equation index: 4475
type: SIMPLE_ASSIGN
vx[280] = (-sin(theta[280])) * r_init[280] * omega_c[280]
*/
void SpiralGalaxy_eqFunction_4475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4475};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[279]] /* vx[280] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1786] /* theta[280] PARAM */)))) * (((data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */)) * ((data->simulationInfo->realParameter[784] /* omega_c[280] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10793(DATA *data, threadData_t *threadData);


/*
equation index: 4477
type: SIMPLE_ASSIGN
vy[280] = cos(theta[280]) * r_init[280] * omega_c[280]
*/
void SpiralGalaxy_eqFunction_4477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[779]] /* vy[280] STATE(1) */) = (cos((data->simulationInfo->realParameter[1786] /* theta[280] PARAM */))) * (((data->simulationInfo->realParameter[1285] /* r_init[280] PARAM */)) * ((data->simulationInfo->realParameter[784] /* omega_c[280] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10792(DATA *data, threadData_t *threadData);


/*
equation index: 4479
type: SIMPLE_ASSIGN
vz[280] = 0.0
*/
void SpiralGalaxy_eqFunction_4479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4479};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1279]] /* vz[280] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10791(DATA *data, threadData_t *threadData);


/*
equation index: 4481
type: SIMPLE_ASSIGN
z[281] = 0.004960000000000002
*/
void SpiralGalaxy_eqFunction_4481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4481};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2780]] /* z[281] STATE(1,vz[281]) */) = 0.004960000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10804(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10805(DATA *data, threadData_t *threadData);


/*
equation index: 4484
type: SIMPLE_ASSIGN
y[281] = r_init[281] * sin(theta[281] + 0.001240000000000001)
*/
void SpiralGalaxy_eqFunction_4484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4484};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2280]] /* y[281] STATE(1,vy[281]) */) = ((data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */)) * (sin((data->simulationInfo->realParameter[1787] /* theta[281] PARAM */) + 0.001240000000000001));
  TRACE_POP
}

/*
equation index: 4485
type: SIMPLE_ASSIGN
x[281] = r_init[281] * cos(theta[281] + 0.001240000000000001)
*/
void SpiralGalaxy_eqFunction_4485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1780]] /* x[281] STATE(1,vx[281]) */) = ((data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */)) * (cos((data->simulationInfo->realParameter[1787] /* theta[281] PARAM */) + 0.001240000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10806(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10807(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10810(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10809(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10808(DATA *data, threadData_t *threadData);


/*
equation index: 4491
type: SIMPLE_ASSIGN
vx[281] = (-sin(theta[281])) * r_init[281] * omega_c[281]
*/
void SpiralGalaxy_eqFunction_4491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4491};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[280]] /* vx[281] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1787] /* theta[281] PARAM */)))) * (((data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */)) * ((data->simulationInfo->realParameter[785] /* omega_c[281] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10803(DATA *data, threadData_t *threadData);


/*
equation index: 4493
type: SIMPLE_ASSIGN
vy[281] = cos(theta[281]) * r_init[281] * omega_c[281]
*/
void SpiralGalaxy_eqFunction_4493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[780]] /* vy[281] STATE(1) */) = (cos((data->simulationInfo->realParameter[1787] /* theta[281] PARAM */))) * (((data->simulationInfo->realParameter[1286] /* r_init[281] PARAM */)) * ((data->simulationInfo->realParameter[785] /* omega_c[281] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10802(DATA *data, threadData_t *threadData);


/*
equation index: 4495
type: SIMPLE_ASSIGN
vz[281] = 0.0
*/
void SpiralGalaxy_eqFunction_4495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4495};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1280]] /* vz[281] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10801(DATA *data, threadData_t *threadData);


/*
equation index: 4497
type: SIMPLE_ASSIGN
z[282] = 0.0051199999999999996
*/
void SpiralGalaxy_eqFunction_4497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4497};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2781]] /* z[282] STATE(1,vz[282]) */) = 0.0051199999999999996;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10814(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10815(DATA *data, threadData_t *threadData);


/*
equation index: 4500
type: SIMPLE_ASSIGN
y[282] = r_init[282] * sin(theta[282] + 0.001279999999999999)
*/
void SpiralGalaxy_eqFunction_4500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4500};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2281]] /* y[282] STATE(1,vy[282]) */) = ((data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */)) * (sin((data->simulationInfo->realParameter[1788] /* theta[282] PARAM */) + 0.001279999999999999));
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_8(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_4001(data, threadData);
  SpiralGalaxy_eqFunction_10504(data, threadData);
  SpiralGalaxy_eqFunction_10505(data, threadData);
  SpiralGalaxy_eqFunction_4004(data, threadData);
  SpiralGalaxy_eqFunction_4005(data, threadData);
  SpiralGalaxy_eqFunction_10506(data, threadData);
  SpiralGalaxy_eqFunction_10507(data, threadData);
  SpiralGalaxy_eqFunction_10510(data, threadData);
  SpiralGalaxy_eqFunction_10509(data, threadData);
  SpiralGalaxy_eqFunction_10508(data, threadData);
  SpiralGalaxy_eqFunction_4011(data, threadData);
  SpiralGalaxy_eqFunction_10503(data, threadData);
  SpiralGalaxy_eqFunction_4013(data, threadData);
  SpiralGalaxy_eqFunction_10502(data, threadData);
  SpiralGalaxy_eqFunction_4015(data, threadData);
  SpiralGalaxy_eqFunction_10501(data, threadData);
  SpiralGalaxy_eqFunction_4017(data, threadData);
  SpiralGalaxy_eqFunction_10514(data, threadData);
  SpiralGalaxy_eqFunction_10515(data, threadData);
  SpiralGalaxy_eqFunction_4020(data, threadData);
  SpiralGalaxy_eqFunction_4021(data, threadData);
  SpiralGalaxy_eqFunction_10516(data, threadData);
  SpiralGalaxy_eqFunction_10517(data, threadData);
  SpiralGalaxy_eqFunction_10520(data, threadData);
  SpiralGalaxy_eqFunction_10519(data, threadData);
  SpiralGalaxy_eqFunction_10518(data, threadData);
  SpiralGalaxy_eqFunction_4027(data, threadData);
  SpiralGalaxy_eqFunction_10513(data, threadData);
  SpiralGalaxy_eqFunction_4029(data, threadData);
  SpiralGalaxy_eqFunction_10512(data, threadData);
  SpiralGalaxy_eqFunction_4031(data, threadData);
  SpiralGalaxy_eqFunction_10511(data, threadData);
  SpiralGalaxy_eqFunction_4033(data, threadData);
  SpiralGalaxy_eqFunction_10524(data, threadData);
  SpiralGalaxy_eqFunction_10525(data, threadData);
  SpiralGalaxy_eqFunction_4036(data, threadData);
  SpiralGalaxy_eqFunction_4037(data, threadData);
  SpiralGalaxy_eqFunction_10526(data, threadData);
  SpiralGalaxy_eqFunction_10527(data, threadData);
  SpiralGalaxy_eqFunction_10530(data, threadData);
  SpiralGalaxy_eqFunction_10529(data, threadData);
  SpiralGalaxy_eqFunction_10528(data, threadData);
  SpiralGalaxy_eqFunction_4043(data, threadData);
  SpiralGalaxy_eqFunction_10523(data, threadData);
  SpiralGalaxy_eqFunction_4045(data, threadData);
  SpiralGalaxy_eqFunction_10522(data, threadData);
  SpiralGalaxy_eqFunction_4047(data, threadData);
  SpiralGalaxy_eqFunction_10521(data, threadData);
  SpiralGalaxy_eqFunction_4049(data, threadData);
  SpiralGalaxy_eqFunction_10534(data, threadData);
  SpiralGalaxy_eqFunction_10535(data, threadData);
  SpiralGalaxy_eqFunction_4052(data, threadData);
  SpiralGalaxy_eqFunction_4053(data, threadData);
  SpiralGalaxy_eqFunction_10536(data, threadData);
  SpiralGalaxy_eqFunction_10537(data, threadData);
  SpiralGalaxy_eqFunction_10540(data, threadData);
  SpiralGalaxy_eqFunction_10539(data, threadData);
  SpiralGalaxy_eqFunction_10538(data, threadData);
  SpiralGalaxy_eqFunction_4059(data, threadData);
  SpiralGalaxy_eqFunction_10533(data, threadData);
  SpiralGalaxy_eqFunction_4061(data, threadData);
  SpiralGalaxy_eqFunction_10532(data, threadData);
  SpiralGalaxy_eqFunction_4063(data, threadData);
  SpiralGalaxy_eqFunction_10531(data, threadData);
  SpiralGalaxy_eqFunction_4065(data, threadData);
  SpiralGalaxy_eqFunction_10544(data, threadData);
  SpiralGalaxy_eqFunction_10545(data, threadData);
  SpiralGalaxy_eqFunction_4068(data, threadData);
  SpiralGalaxy_eqFunction_4069(data, threadData);
  SpiralGalaxy_eqFunction_10546(data, threadData);
  SpiralGalaxy_eqFunction_10547(data, threadData);
  SpiralGalaxy_eqFunction_10550(data, threadData);
  SpiralGalaxy_eqFunction_10549(data, threadData);
  SpiralGalaxy_eqFunction_10548(data, threadData);
  SpiralGalaxy_eqFunction_4075(data, threadData);
  SpiralGalaxy_eqFunction_10543(data, threadData);
  SpiralGalaxy_eqFunction_4077(data, threadData);
  SpiralGalaxy_eqFunction_10542(data, threadData);
  SpiralGalaxy_eqFunction_4079(data, threadData);
  SpiralGalaxy_eqFunction_10541(data, threadData);
  SpiralGalaxy_eqFunction_4081(data, threadData);
  SpiralGalaxy_eqFunction_10554(data, threadData);
  SpiralGalaxy_eqFunction_10555(data, threadData);
  SpiralGalaxy_eqFunction_4084(data, threadData);
  SpiralGalaxy_eqFunction_4085(data, threadData);
  SpiralGalaxy_eqFunction_10556(data, threadData);
  SpiralGalaxy_eqFunction_10557(data, threadData);
  SpiralGalaxy_eqFunction_10560(data, threadData);
  SpiralGalaxy_eqFunction_10559(data, threadData);
  SpiralGalaxy_eqFunction_10558(data, threadData);
  SpiralGalaxy_eqFunction_4091(data, threadData);
  SpiralGalaxy_eqFunction_10553(data, threadData);
  SpiralGalaxy_eqFunction_4093(data, threadData);
  SpiralGalaxy_eqFunction_10552(data, threadData);
  SpiralGalaxy_eqFunction_4095(data, threadData);
  SpiralGalaxy_eqFunction_10551(data, threadData);
  SpiralGalaxy_eqFunction_4097(data, threadData);
  SpiralGalaxy_eqFunction_10564(data, threadData);
  SpiralGalaxy_eqFunction_10565(data, threadData);
  SpiralGalaxy_eqFunction_4100(data, threadData);
  SpiralGalaxy_eqFunction_4101(data, threadData);
  SpiralGalaxy_eqFunction_10566(data, threadData);
  SpiralGalaxy_eqFunction_10567(data, threadData);
  SpiralGalaxy_eqFunction_10570(data, threadData);
  SpiralGalaxy_eqFunction_10569(data, threadData);
  SpiralGalaxy_eqFunction_10568(data, threadData);
  SpiralGalaxy_eqFunction_4107(data, threadData);
  SpiralGalaxy_eqFunction_10563(data, threadData);
  SpiralGalaxy_eqFunction_4109(data, threadData);
  SpiralGalaxy_eqFunction_10562(data, threadData);
  SpiralGalaxy_eqFunction_4111(data, threadData);
  SpiralGalaxy_eqFunction_10561(data, threadData);
  SpiralGalaxy_eqFunction_4113(data, threadData);
  SpiralGalaxy_eqFunction_10574(data, threadData);
  SpiralGalaxy_eqFunction_10575(data, threadData);
  SpiralGalaxy_eqFunction_4116(data, threadData);
  SpiralGalaxy_eqFunction_4117(data, threadData);
  SpiralGalaxy_eqFunction_10576(data, threadData);
  SpiralGalaxy_eqFunction_10577(data, threadData);
  SpiralGalaxy_eqFunction_10580(data, threadData);
  SpiralGalaxy_eqFunction_10579(data, threadData);
  SpiralGalaxy_eqFunction_10578(data, threadData);
  SpiralGalaxy_eqFunction_4123(data, threadData);
  SpiralGalaxy_eqFunction_10573(data, threadData);
  SpiralGalaxy_eqFunction_4125(data, threadData);
  SpiralGalaxy_eqFunction_10572(data, threadData);
  SpiralGalaxy_eqFunction_4127(data, threadData);
  SpiralGalaxy_eqFunction_10571(data, threadData);
  SpiralGalaxy_eqFunction_4129(data, threadData);
  SpiralGalaxy_eqFunction_10584(data, threadData);
  SpiralGalaxy_eqFunction_10585(data, threadData);
  SpiralGalaxy_eqFunction_4132(data, threadData);
  SpiralGalaxy_eqFunction_4133(data, threadData);
  SpiralGalaxy_eqFunction_10586(data, threadData);
  SpiralGalaxy_eqFunction_10587(data, threadData);
  SpiralGalaxy_eqFunction_10590(data, threadData);
  SpiralGalaxy_eqFunction_10589(data, threadData);
  SpiralGalaxy_eqFunction_10588(data, threadData);
  SpiralGalaxy_eqFunction_4139(data, threadData);
  SpiralGalaxy_eqFunction_10583(data, threadData);
  SpiralGalaxy_eqFunction_4141(data, threadData);
  SpiralGalaxy_eqFunction_10582(data, threadData);
  SpiralGalaxy_eqFunction_4143(data, threadData);
  SpiralGalaxy_eqFunction_10581(data, threadData);
  SpiralGalaxy_eqFunction_4145(data, threadData);
  SpiralGalaxy_eqFunction_10594(data, threadData);
  SpiralGalaxy_eqFunction_10595(data, threadData);
  SpiralGalaxy_eqFunction_4148(data, threadData);
  SpiralGalaxy_eqFunction_4149(data, threadData);
  SpiralGalaxy_eqFunction_10596(data, threadData);
  SpiralGalaxy_eqFunction_10597(data, threadData);
  SpiralGalaxy_eqFunction_10600(data, threadData);
  SpiralGalaxy_eqFunction_10599(data, threadData);
  SpiralGalaxy_eqFunction_10598(data, threadData);
  SpiralGalaxy_eqFunction_4155(data, threadData);
  SpiralGalaxy_eqFunction_10593(data, threadData);
  SpiralGalaxy_eqFunction_4157(data, threadData);
  SpiralGalaxy_eqFunction_10592(data, threadData);
  SpiralGalaxy_eqFunction_4159(data, threadData);
  SpiralGalaxy_eqFunction_10591(data, threadData);
  SpiralGalaxy_eqFunction_4161(data, threadData);
  SpiralGalaxy_eqFunction_10604(data, threadData);
  SpiralGalaxy_eqFunction_10605(data, threadData);
  SpiralGalaxy_eqFunction_4164(data, threadData);
  SpiralGalaxy_eqFunction_4165(data, threadData);
  SpiralGalaxy_eqFunction_10606(data, threadData);
  SpiralGalaxy_eqFunction_10607(data, threadData);
  SpiralGalaxy_eqFunction_10610(data, threadData);
  SpiralGalaxy_eqFunction_10609(data, threadData);
  SpiralGalaxy_eqFunction_10608(data, threadData);
  SpiralGalaxy_eqFunction_4171(data, threadData);
  SpiralGalaxy_eqFunction_10603(data, threadData);
  SpiralGalaxy_eqFunction_4173(data, threadData);
  SpiralGalaxy_eqFunction_10602(data, threadData);
  SpiralGalaxy_eqFunction_4175(data, threadData);
  SpiralGalaxy_eqFunction_10601(data, threadData);
  SpiralGalaxy_eqFunction_4177(data, threadData);
  SpiralGalaxy_eqFunction_10614(data, threadData);
  SpiralGalaxy_eqFunction_10615(data, threadData);
  SpiralGalaxy_eqFunction_4180(data, threadData);
  SpiralGalaxy_eqFunction_4181(data, threadData);
  SpiralGalaxy_eqFunction_10616(data, threadData);
  SpiralGalaxy_eqFunction_10617(data, threadData);
  SpiralGalaxy_eqFunction_10620(data, threadData);
  SpiralGalaxy_eqFunction_10619(data, threadData);
  SpiralGalaxy_eqFunction_10618(data, threadData);
  SpiralGalaxy_eqFunction_4187(data, threadData);
  SpiralGalaxy_eqFunction_10613(data, threadData);
  SpiralGalaxy_eqFunction_4189(data, threadData);
  SpiralGalaxy_eqFunction_10612(data, threadData);
  SpiralGalaxy_eqFunction_4191(data, threadData);
  SpiralGalaxy_eqFunction_10611(data, threadData);
  SpiralGalaxy_eqFunction_4193(data, threadData);
  SpiralGalaxy_eqFunction_10624(data, threadData);
  SpiralGalaxy_eqFunction_10625(data, threadData);
  SpiralGalaxy_eqFunction_4196(data, threadData);
  SpiralGalaxy_eqFunction_4197(data, threadData);
  SpiralGalaxy_eqFunction_10626(data, threadData);
  SpiralGalaxy_eqFunction_10627(data, threadData);
  SpiralGalaxy_eqFunction_10630(data, threadData);
  SpiralGalaxy_eqFunction_10629(data, threadData);
  SpiralGalaxy_eqFunction_10628(data, threadData);
  SpiralGalaxy_eqFunction_4203(data, threadData);
  SpiralGalaxy_eqFunction_10623(data, threadData);
  SpiralGalaxy_eqFunction_4205(data, threadData);
  SpiralGalaxy_eqFunction_10622(data, threadData);
  SpiralGalaxy_eqFunction_4207(data, threadData);
  SpiralGalaxy_eqFunction_10621(data, threadData);
  SpiralGalaxy_eqFunction_4209(data, threadData);
  SpiralGalaxy_eqFunction_10634(data, threadData);
  SpiralGalaxy_eqFunction_10635(data, threadData);
  SpiralGalaxy_eqFunction_4212(data, threadData);
  SpiralGalaxy_eqFunction_4213(data, threadData);
  SpiralGalaxy_eqFunction_10636(data, threadData);
  SpiralGalaxy_eqFunction_10637(data, threadData);
  SpiralGalaxy_eqFunction_10640(data, threadData);
  SpiralGalaxy_eqFunction_10639(data, threadData);
  SpiralGalaxy_eqFunction_10638(data, threadData);
  SpiralGalaxy_eqFunction_4219(data, threadData);
  SpiralGalaxy_eqFunction_10633(data, threadData);
  SpiralGalaxy_eqFunction_4221(data, threadData);
  SpiralGalaxy_eqFunction_10632(data, threadData);
  SpiralGalaxy_eqFunction_4223(data, threadData);
  SpiralGalaxy_eqFunction_10631(data, threadData);
  SpiralGalaxy_eqFunction_4225(data, threadData);
  SpiralGalaxy_eqFunction_10644(data, threadData);
  SpiralGalaxy_eqFunction_10645(data, threadData);
  SpiralGalaxy_eqFunction_4228(data, threadData);
  SpiralGalaxy_eqFunction_4229(data, threadData);
  SpiralGalaxy_eqFunction_10646(data, threadData);
  SpiralGalaxy_eqFunction_10647(data, threadData);
  SpiralGalaxy_eqFunction_10650(data, threadData);
  SpiralGalaxy_eqFunction_10649(data, threadData);
  SpiralGalaxy_eqFunction_10648(data, threadData);
  SpiralGalaxy_eqFunction_4235(data, threadData);
  SpiralGalaxy_eqFunction_10643(data, threadData);
  SpiralGalaxy_eqFunction_4237(data, threadData);
  SpiralGalaxy_eqFunction_10642(data, threadData);
  SpiralGalaxy_eqFunction_4239(data, threadData);
  SpiralGalaxy_eqFunction_10641(data, threadData);
  SpiralGalaxy_eqFunction_4241(data, threadData);
  SpiralGalaxy_eqFunction_10654(data, threadData);
  SpiralGalaxy_eqFunction_10655(data, threadData);
  SpiralGalaxy_eqFunction_4244(data, threadData);
  SpiralGalaxy_eqFunction_4245(data, threadData);
  SpiralGalaxy_eqFunction_10656(data, threadData);
  SpiralGalaxy_eqFunction_10657(data, threadData);
  SpiralGalaxy_eqFunction_10660(data, threadData);
  SpiralGalaxy_eqFunction_10659(data, threadData);
  SpiralGalaxy_eqFunction_10658(data, threadData);
  SpiralGalaxy_eqFunction_4251(data, threadData);
  SpiralGalaxy_eqFunction_10653(data, threadData);
  SpiralGalaxy_eqFunction_4253(data, threadData);
  SpiralGalaxy_eqFunction_10652(data, threadData);
  SpiralGalaxy_eqFunction_4255(data, threadData);
  SpiralGalaxy_eqFunction_10651(data, threadData);
  SpiralGalaxy_eqFunction_4257(data, threadData);
  SpiralGalaxy_eqFunction_10664(data, threadData);
  SpiralGalaxy_eqFunction_10665(data, threadData);
  SpiralGalaxy_eqFunction_4260(data, threadData);
  SpiralGalaxy_eqFunction_4261(data, threadData);
  SpiralGalaxy_eqFunction_10666(data, threadData);
  SpiralGalaxy_eqFunction_10667(data, threadData);
  SpiralGalaxy_eqFunction_10670(data, threadData);
  SpiralGalaxy_eqFunction_10669(data, threadData);
  SpiralGalaxy_eqFunction_10668(data, threadData);
  SpiralGalaxy_eqFunction_4267(data, threadData);
  SpiralGalaxy_eqFunction_10663(data, threadData);
  SpiralGalaxy_eqFunction_4269(data, threadData);
  SpiralGalaxy_eqFunction_10662(data, threadData);
  SpiralGalaxy_eqFunction_4271(data, threadData);
  SpiralGalaxy_eqFunction_10661(data, threadData);
  SpiralGalaxy_eqFunction_4273(data, threadData);
  SpiralGalaxy_eqFunction_10674(data, threadData);
  SpiralGalaxy_eqFunction_10675(data, threadData);
  SpiralGalaxy_eqFunction_4276(data, threadData);
  SpiralGalaxy_eqFunction_4277(data, threadData);
  SpiralGalaxy_eqFunction_10676(data, threadData);
  SpiralGalaxy_eqFunction_10677(data, threadData);
  SpiralGalaxy_eqFunction_10680(data, threadData);
  SpiralGalaxy_eqFunction_10679(data, threadData);
  SpiralGalaxy_eqFunction_10678(data, threadData);
  SpiralGalaxy_eqFunction_4283(data, threadData);
  SpiralGalaxy_eqFunction_10673(data, threadData);
  SpiralGalaxy_eqFunction_4285(data, threadData);
  SpiralGalaxy_eqFunction_10672(data, threadData);
  SpiralGalaxy_eqFunction_4287(data, threadData);
  SpiralGalaxy_eqFunction_10671(data, threadData);
  SpiralGalaxy_eqFunction_4289(data, threadData);
  SpiralGalaxy_eqFunction_10684(data, threadData);
  SpiralGalaxy_eqFunction_10685(data, threadData);
  SpiralGalaxy_eqFunction_4292(data, threadData);
  SpiralGalaxy_eqFunction_4293(data, threadData);
  SpiralGalaxy_eqFunction_10686(data, threadData);
  SpiralGalaxy_eqFunction_10687(data, threadData);
  SpiralGalaxy_eqFunction_10690(data, threadData);
  SpiralGalaxy_eqFunction_10689(data, threadData);
  SpiralGalaxy_eqFunction_10688(data, threadData);
  SpiralGalaxy_eqFunction_4299(data, threadData);
  SpiralGalaxy_eqFunction_10683(data, threadData);
  SpiralGalaxy_eqFunction_4301(data, threadData);
  SpiralGalaxy_eqFunction_10682(data, threadData);
  SpiralGalaxy_eqFunction_4303(data, threadData);
  SpiralGalaxy_eqFunction_10681(data, threadData);
  SpiralGalaxy_eqFunction_4305(data, threadData);
  SpiralGalaxy_eqFunction_10694(data, threadData);
  SpiralGalaxy_eqFunction_10695(data, threadData);
  SpiralGalaxy_eqFunction_4308(data, threadData);
  SpiralGalaxy_eqFunction_4309(data, threadData);
  SpiralGalaxy_eqFunction_10696(data, threadData);
  SpiralGalaxy_eqFunction_10697(data, threadData);
  SpiralGalaxy_eqFunction_10700(data, threadData);
  SpiralGalaxy_eqFunction_10699(data, threadData);
  SpiralGalaxy_eqFunction_10698(data, threadData);
  SpiralGalaxy_eqFunction_4315(data, threadData);
  SpiralGalaxy_eqFunction_10693(data, threadData);
  SpiralGalaxy_eqFunction_4317(data, threadData);
  SpiralGalaxy_eqFunction_10692(data, threadData);
  SpiralGalaxy_eqFunction_4319(data, threadData);
  SpiralGalaxy_eqFunction_10691(data, threadData);
  SpiralGalaxy_eqFunction_4321(data, threadData);
  SpiralGalaxy_eqFunction_10704(data, threadData);
  SpiralGalaxy_eqFunction_10705(data, threadData);
  SpiralGalaxy_eqFunction_4324(data, threadData);
  SpiralGalaxy_eqFunction_4325(data, threadData);
  SpiralGalaxy_eqFunction_10706(data, threadData);
  SpiralGalaxy_eqFunction_10707(data, threadData);
  SpiralGalaxy_eqFunction_10710(data, threadData);
  SpiralGalaxy_eqFunction_10709(data, threadData);
  SpiralGalaxy_eqFunction_10708(data, threadData);
  SpiralGalaxy_eqFunction_4331(data, threadData);
  SpiralGalaxy_eqFunction_10703(data, threadData);
  SpiralGalaxy_eqFunction_4333(data, threadData);
  SpiralGalaxy_eqFunction_10702(data, threadData);
  SpiralGalaxy_eqFunction_4335(data, threadData);
  SpiralGalaxy_eqFunction_10701(data, threadData);
  SpiralGalaxy_eqFunction_4337(data, threadData);
  SpiralGalaxy_eqFunction_10714(data, threadData);
  SpiralGalaxy_eqFunction_10715(data, threadData);
  SpiralGalaxy_eqFunction_4340(data, threadData);
  SpiralGalaxy_eqFunction_4341(data, threadData);
  SpiralGalaxy_eqFunction_10716(data, threadData);
  SpiralGalaxy_eqFunction_10717(data, threadData);
  SpiralGalaxy_eqFunction_10720(data, threadData);
  SpiralGalaxy_eqFunction_10719(data, threadData);
  SpiralGalaxy_eqFunction_10718(data, threadData);
  SpiralGalaxy_eqFunction_4347(data, threadData);
  SpiralGalaxy_eqFunction_10713(data, threadData);
  SpiralGalaxy_eqFunction_4349(data, threadData);
  SpiralGalaxy_eqFunction_10712(data, threadData);
  SpiralGalaxy_eqFunction_4351(data, threadData);
  SpiralGalaxy_eqFunction_10711(data, threadData);
  SpiralGalaxy_eqFunction_4353(data, threadData);
  SpiralGalaxy_eqFunction_10724(data, threadData);
  SpiralGalaxy_eqFunction_10725(data, threadData);
  SpiralGalaxy_eqFunction_4356(data, threadData);
  SpiralGalaxy_eqFunction_4357(data, threadData);
  SpiralGalaxy_eqFunction_10726(data, threadData);
  SpiralGalaxy_eqFunction_10727(data, threadData);
  SpiralGalaxy_eqFunction_10730(data, threadData);
  SpiralGalaxy_eqFunction_10729(data, threadData);
  SpiralGalaxy_eqFunction_10728(data, threadData);
  SpiralGalaxy_eqFunction_4363(data, threadData);
  SpiralGalaxy_eqFunction_10723(data, threadData);
  SpiralGalaxy_eqFunction_4365(data, threadData);
  SpiralGalaxy_eqFunction_10722(data, threadData);
  SpiralGalaxy_eqFunction_4367(data, threadData);
  SpiralGalaxy_eqFunction_10721(data, threadData);
  SpiralGalaxy_eqFunction_4369(data, threadData);
  SpiralGalaxy_eqFunction_10734(data, threadData);
  SpiralGalaxy_eqFunction_10735(data, threadData);
  SpiralGalaxy_eqFunction_4372(data, threadData);
  SpiralGalaxy_eqFunction_4373(data, threadData);
  SpiralGalaxy_eqFunction_10736(data, threadData);
  SpiralGalaxy_eqFunction_10737(data, threadData);
  SpiralGalaxy_eqFunction_10740(data, threadData);
  SpiralGalaxy_eqFunction_10739(data, threadData);
  SpiralGalaxy_eqFunction_10738(data, threadData);
  SpiralGalaxy_eqFunction_4379(data, threadData);
  SpiralGalaxy_eqFunction_10733(data, threadData);
  SpiralGalaxy_eqFunction_4381(data, threadData);
  SpiralGalaxy_eqFunction_10732(data, threadData);
  SpiralGalaxy_eqFunction_4383(data, threadData);
  SpiralGalaxy_eqFunction_10731(data, threadData);
  SpiralGalaxy_eqFunction_4385(data, threadData);
  SpiralGalaxy_eqFunction_10744(data, threadData);
  SpiralGalaxy_eqFunction_10745(data, threadData);
  SpiralGalaxy_eqFunction_4388(data, threadData);
  SpiralGalaxy_eqFunction_4389(data, threadData);
  SpiralGalaxy_eqFunction_10746(data, threadData);
  SpiralGalaxy_eqFunction_10747(data, threadData);
  SpiralGalaxy_eqFunction_10750(data, threadData);
  SpiralGalaxy_eqFunction_10749(data, threadData);
  SpiralGalaxy_eqFunction_10748(data, threadData);
  SpiralGalaxy_eqFunction_4395(data, threadData);
  SpiralGalaxy_eqFunction_10743(data, threadData);
  SpiralGalaxy_eqFunction_4397(data, threadData);
  SpiralGalaxy_eqFunction_10742(data, threadData);
  SpiralGalaxy_eqFunction_4399(data, threadData);
  SpiralGalaxy_eqFunction_10741(data, threadData);
  SpiralGalaxy_eqFunction_4401(data, threadData);
  SpiralGalaxy_eqFunction_10754(data, threadData);
  SpiralGalaxy_eqFunction_10755(data, threadData);
  SpiralGalaxy_eqFunction_4404(data, threadData);
  SpiralGalaxy_eqFunction_4405(data, threadData);
  SpiralGalaxy_eqFunction_10756(data, threadData);
  SpiralGalaxy_eqFunction_10757(data, threadData);
  SpiralGalaxy_eqFunction_10760(data, threadData);
  SpiralGalaxy_eqFunction_10759(data, threadData);
  SpiralGalaxy_eqFunction_10758(data, threadData);
  SpiralGalaxy_eqFunction_4411(data, threadData);
  SpiralGalaxy_eqFunction_10753(data, threadData);
  SpiralGalaxy_eqFunction_4413(data, threadData);
  SpiralGalaxy_eqFunction_10752(data, threadData);
  SpiralGalaxy_eqFunction_4415(data, threadData);
  SpiralGalaxy_eqFunction_10751(data, threadData);
  SpiralGalaxy_eqFunction_4417(data, threadData);
  SpiralGalaxy_eqFunction_10764(data, threadData);
  SpiralGalaxy_eqFunction_10765(data, threadData);
  SpiralGalaxy_eqFunction_4420(data, threadData);
  SpiralGalaxy_eqFunction_4421(data, threadData);
  SpiralGalaxy_eqFunction_10766(data, threadData);
  SpiralGalaxy_eqFunction_10767(data, threadData);
  SpiralGalaxy_eqFunction_10770(data, threadData);
  SpiralGalaxy_eqFunction_10769(data, threadData);
  SpiralGalaxy_eqFunction_10768(data, threadData);
  SpiralGalaxy_eqFunction_4427(data, threadData);
  SpiralGalaxy_eqFunction_10763(data, threadData);
  SpiralGalaxy_eqFunction_4429(data, threadData);
  SpiralGalaxy_eqFunction_10762(data, threadData);
  SpiralGalaxy_eqFunction_4431(data, threadData);
  SpiralGalaxy_eqFunction_10761(data, threadData);
  SpiralGalaxy_eqFunction_4433(data, threadData);
  SpiralGalaxy_eqFunction_10774(data, threadData);
  SpiralGalaxy_eqFunction_10775(data, threadData);
  SpiralGalaxy_eqFunction_4436(data, threadData);
  SpiralGalaxy_eqFunction_4437(data, threadData);
  SpiralGalaxy_eqFunction_10776(data, threadData);
  SpiralGalaxy_eqFunction_10777(data, threadData);
  SpiralGalaxy_eqFunction_10780(data, threadData);
  SpiralGalaxy_eqFunction_10779(data, threadData);
  SpiralGalaxy_eqFunction_10778(data, threadData);
  SpiralGalaxy_eqFunction_4443(data, threadData);
  SpiralGalaxy_eqFunction_10773(data, threadData);
  SpiralGalaxy_eqFunction_4445(data, threadData);
  SpiralGalaxy_eqFunction_10772(data, threadData);
  SpiralGalaxy_eqFunction_4447(data, threadData);
  SpiralGalaxy_eqFunction_10771(data, threadData);
  SpiralGalaxy_eqFunction_4449(data, threadData);
  SpiralGalaxy_eqFunction_10784(data, threadData);
  SpiralGalaxy_eqFunction_10785(data, threadData);
  SpiralGalaxy_eqFunction_4452(data, threadData);
  SpiralGalaxy_eqFunction_4453(data, threadData);
  SpiralGalaxy_eqFunction_10786(data, threadData);
  SpiralGalaxy_eqFunction_10787(data, threadData);
  SpiralGalaxy_eqFunction_10790(data, threadData);
  SpiralGalaxy_eqFunction_10789(data, threadData);
  SpiralGalaxy_eqFunction_10788(data, threadData);
  SpiralGalaxy_eqFunction_4459(data, threadData);
  SpiralGalaxy_eqFunction_10783(data, threadData);
  SpiralGalaxy_eqFunction_4461(data, threadData);
  SpiralGalaxy_eqFunction_10782(data, threadData);
  SpiralGalaxy_eqFunction_4463(data, threadData);
  SpiralGalaxy_eqFunction_10781(data, threadData);
  SpiralGalaxy_eqFunction_4465(data, threadData);
  SpiralGalaxy_eqFunction_10794(data, threadData);
  SpiralGalaxy_eqFunction_10795(data, threadData);
  SpiralGalaxy_eqFunction_4468(data, threadData);
  SpiralGalaxy_eqFunction_4469(data, threadData);
  SpiralGalaxy_eqFunction_10796(data, threadData);
  SpiralGalaxy_eqFunction_10797(data, threadData);
  SpiralGalaxy_eqFunction_10800(data, threadData);
  SpiralGalaxy_eqFunction_10799(data, threadData);
  SpiralGalaxy_eqFunction_10798(data, threadData);
  SpiralGalaxy_eqFunction_4475(data, threadData);
  SpiralGalaxy_eqFunction_10793(data, threadData);
  SpiralGalaxy_eqFunction_4477(data, threadData);
  SpiralGalaxy_eqFunction_10792(data, threadData);
  SpiralGalaxy_eqFunction_4479(data, threadData);
  SpiralGalaxy_eqFunction_10791(data, threadData);
  SpiralGalaxy_eqFunction_4481(data, threadData);
  SpiralGalaxy_eqFunction_10804(data, threadData);
  SpiralGalaxy_eqFunction_10805(data, threadData);
  SpiralGalaxy_eqFunction_4484(data, threadData);
  SpiralGalaxy_eqFunction_4485(data, threadData);
  SpiralGalaxy_eqFunction_10806(data, threadData);
  SpiralGalaxy_eqFunction_10807(data, threadData);
  SpiralGalaxy_eqFunction_10810(data, threadData);
  SpiralGalaxy_eqFunction_10809(data, threadData);
  SpiralGalaxy_eqFunction_10808(data, threadData);
  SpiralGalaxy_eqFunction_4491(data, threadData);
  SpiralGalaxy_eqFunction_10803(data, threadData);
  SpiralGalaxy_eqFunction_4493(data, threadData);
  SpiralGalaxy_eqFunction_10802(data, threadData);
  SpiralGalaxy_eqFunction_4495(data, threadData);
  SpiralGalaxy_eqFunction_10801(data, threadData);
  SpiralGalaxy_eqFunction_4497(data, threadData);
  SpiralGalaxy_eqFunction_10814(data, threadData);
  SpiralGalaxy_eqFunction_10815(data, threadData);
  SpiralGalaxy_eqFunction_4500(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif