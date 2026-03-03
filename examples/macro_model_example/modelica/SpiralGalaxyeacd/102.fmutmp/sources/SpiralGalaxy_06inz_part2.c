#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void SpiralGalaxy_eqFunction_8629(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8628(DATA *data, threadData_t *threadData);


/*
equation index: 1003
type: SIMPLE_ASSIGN
vx[63] = (-sin(theta[63])) * r_init[63] * omega_c[63]
*/
void SpiralGalaxy_eqFunction_1003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1003};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[62]] /* vx[63] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1569] /* theta[63] PARAM */)))) * (((data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */)) * ((data->simulationInfo->realParameter[567] /* omega_c[63] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8623(DATA *data, threadData_t *threadData);


/*
equation index: 1005
type: SIMPLE_ASSIGN
vy[63] = cos(theta[63]) * r_init[63] * omega_c[63]
*/
void SpiralGalaxy_eqFunction_1005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1005};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[562]] /* vy[63] STATE(1) */) = (cos((data->simulationInfo->realParameter[1569] /* theta[63] PARAM */))) * (((data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */)) * ((data->simulationInfo->realParameter[567] /* omega_c[63] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8622(DATA *data, threadData_t *threadData);


/*
equation index: 1007
type: SIMPLE_ASSIGN
vz[63] = 0.0
*/
void SpiralGalaxy_eqFunction_1007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1007};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1062]] /* vz[63] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8621(DATA *data, threadData_t *threadData);


/*
equation index: 1009
type: SIMPLE_ASSIGN
z[64] = -0.029760000000000005
*/
void SpiralGalaxy_eqFunction_1009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1009};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2563]] /* z[64] STATE(1,vz[64]) */) = -0.029760000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8634(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8635(DATA *data, threadData_t *threadData);


/*
equation index: 1012
type: SIMPLE_ASSIGN
y[64] = r_init[64] * sin(theta[64] - 0.00744)
*/
void SpiralGalaxy_eqFunction_1012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1012};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2063]] /* y[64] STATE(1,vy[64]) */) = ((data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */)) * (sin((data->simulationInfo->realParameter[1570] /* theta[64] PARAM */) - 0.00744));
  TRACE_POP
}

/*
equation index: 1013
type: SIMPLE_ASSIGN
x[64] = r_init[64] * cos(theta[64] - 0.00744)
*/
void SpiralGalaxy_eqFunction_1013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1013};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1563]] /* x[64] STATE(1,vx[64]) */) = ((data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */)) * (cos((data->simulationInfo->realParameter[1570] /* theta[64] PARAM */) - 0.00744));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8636(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8637(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8640(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8639(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8638(DATA *data, threadData_t *threadData);


/*
equation index: 1019
type: SIMPLE_ASSIGN
vx[64] = (-sin(theta[64])) * r_init[64] * omega_c[64]
*/
void SpiralGalaxy_eqFunction_1019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1019};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[63]] /* vx[64] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1570] /* theta[64] PARAM */)))) * (((data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */)) * ((data->simulationInfo->realParameter[568] /* omega_c[64] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8633(DATA *data, threadData_t *threadData);


/*
equation index: 1021
type: SIMPLE_ASSIGN
vy[64] = cos(theta[64]) * r_init[64] * omega_c[64]
*/
void SpiralGalaxy_eqFunction_1021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1021};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[563]] /* vy[64] STATE(1) */) = (cos((data->simulationInfo->realParameter[1570] /* theta[64] PARAM */))) * (((data->simulationInfo->realParameter[1069] /* r_init[64] PARAM */)) * ((data->simulationInfo->realParameter[568] /* omega_c[64] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8632(DATA *data, threadData_t *threadData);


/*
equation index: 1023
type: SIMPLE_ASSIGN
vz[64] = 0.0
*/
void SpiralGalaxy_eqFunction_1023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1023};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1063]] /* vz[64] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8631(DATA *data, threadData_t *threadData);


/*
equation index: 1025
type: SIMPLE_ASSIGN
z[65] = -0.0296
*/
void SpiralGalaxy_eqFunction_1025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1025};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2564]] /* z[65] STATE(1,vz[65]) */) = -0.0296;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8644(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8645(DATA *data, threadData_t *threadData);


/*
equation index: 1028
type: SIMPLE_ASSIGN
y[65] = r_init[65] * sin(theta[65] - 0.0074)
*/
void SpiralGalaxy_eqFunction_1028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1028};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2064]] /* y[65] STATE(1,vy[65]) */) = ((data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */)) * (sin((data->simulationInfo->realParameter[1571] /* theta[65] PARAM */) - 0.0074));
  TRACE_POP
}

/*
equation index: 1029
type: SIMPLE_ASSIGN
x[65] = r_init[65] * cos(theta[65] - 0.0074)
*/
void SpiralGalaxy_eqFunction_1029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1029};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1564]] /* x[65] STATE(1,vx[65]) */) = ((data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */)) * (cos((data->simulationInfo->realParameter[1571] /* theta[65] PARAM */) - 0.0074));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8646(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8647(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8650(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8649(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8648(DATA *data, threadData_t *threadData);


/*
equation index: 1035
type: SIMPLE_ASSIGN
vx[65] = (-sin(theta[65])) * r_init[65] * omega_c[65]
*/
void SpiralGalaxy_eqFunction_1035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1035};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[64]] /* vx[65] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1571] /* theta[65] PARAM */)))) * (((data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */)) * ((data->simulationInfo->realParameter[569] /* omega_c[65] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8643(DATA *data, threadData_t *threadData);


/*
equation index: 1037
type: SIMPLE_ASSIGN
vy[65] = cos(theta[65]) * r_init[65] * omega_c[65]
*/
void SpiralGalaxy_eqFunction_1037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1037};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[564]] /* vy[65] STATE(1) */) = (cos((data->simulationInfo->realParameter[1571] /* theta[65] PARAM */))) * (((data->simulationInfo->realParameter[1070] /* r_init[65] PARAM */)) * ((data->simulationInfo->realParameter[569] /* omega_c[65] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8642(DATA *data, threadData_t *threadData);


/*
equation index: 1039
type: SIMPLE_ASSIGN
vz[65] = 0.0
*/
void SpiralGalaxy_eqFunction_1039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1039};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1064]] /* vz[65] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8641(DATA *data, threadData_t *threadData);


/*
equation index: 1041
type: SIMPLE_ASSIGN
z[66] = -0.02944
*/
void SpiralGalaxy_eqFunction_1041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1041};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2565]] /* z[66] STATE(1,vz[66]) */) = -0.02944;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8654(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8655(DATA *data, threadData_t *threadData);


/*
equation index: 1044
type: SIMPLE_ASSIGN
y[66] = r_init[66] * sin(theta[66] - 0.00736)
*/
void SpiralGalaxy_eqFunction_1044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1044};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2065]] /* y[66] STATE(1,vy[66]) */) = ((data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */)) * (sin((data->simulationInfo->realParameter[1572] /* theta[66] PARAM */) - 0.00736));
  TRACE_POP
}

/*
equation index: 1045
type: SIMPLE_ASSIGN
x[66] = r_init[66] * cos(theta[66] - 0.00736)
*/
void SpiralGalaxy_eqFunction_1045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1045};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1565]] /* x[66] STATE(1,vx[66]) */) = ((data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */)) * (cos((data->simulationInfo->realParameter[1572] /* theta[66] PARAM */) - 0.00736));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8656(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8657(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8660(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8659(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8658(DATA *data, threadData_t *threadData);


/*
equation index: 1051
type: SIMPLE_ASSIGN
vx[66] = (-sin(theta[66])) * r_init[66] * omega_c[66]
*/
void SpiralGalaxy_eqFunction_1051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1051};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[65]] /* vx[66] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1572] /* theta[66] PARAM */)))) * (((data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */)) * ((data->simulationInfo->realParameter[570] /* omega_c[66] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8653(DATA *data, threadData_t *threadData);


/*
equation index: 1053
type: SIMPLE_ASSIGN
vy[66] = cos(theta[66]) * r_init[66] * omega_c[66]
*/
void SpiralGalaxy_eqFunction_1053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1053};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[565]] /* vy[66] STATE(1) */) = (cos((data->simulationInfo->realParameter[1572] /* theta[66] PARAM */))) * (((data->simulationInfo->realParameter[1071] /* r_init[66] PARAM */)) * ((data->simulationInfo->realParameter[570] /* omega_c[66] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8652(DATA *data, threadData_t *threadData);


/*
equation index: 1055
type: SIMPLE_ASSIGN
vz[66] = 0.0
*/
void SpiralGalaxy_eqFunction_1055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1055};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1065]] /* vz[66] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8651(DATA *data, threadData_t *threadData);


/*
equation index: 1057
type: SIMPLE_ASSIGN
z[67] = -0.02928
*/
void SpiralGalaxy_eqFunction_1057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1057};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2566]] /* z[67] STATE(1,vz[67]) */) = -0.02928;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8664(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8665(DATA *data, threadData_t *threadData);


/*
equation index: 1060
type: SIMPLE_ASSIGN
y[67] = r_init[67] * sin(theta[67] - 0.00732)
*/
void SpiralGalaxy_eqFunction_1060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1060};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2066]] /* y[67] STATE(1,vy[67]) */) = ((data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */)) * (sin((data->simulationInfo->realParameter[1573] /* theta[67] PARAM */) - 0.00732));
  TRACE_POP
}

/*
equation index: 1061
type: SIMPLE_ASSIGN
x[67] = r_init[67] * cos(theta[67] - 0.00732)
*/
void SpiralGalaxy_eqFunction_1061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1061};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1566]] /* x[67] STATE(1,vx[67]) */) = ((data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */)) * (cos((data->simulationInfo->realParameter[1573] /* theta[67] PARAM */) - 0.00732));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8666(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8667(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8670(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8669(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8668(DATA *data, threadData_t *threadData);


/*
equation index: 1067
type: SIMPLE_ASSIGN
vx[67] = (-sin(theta[67])) * r_init[67] * omega_c[67]
*/
void SpiralGalaxy_eqFunction_1067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1067};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[66]] /* vx[67] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1573] /* theta[67] PARAM */)))) * (((data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */)) * ((data->simulationInfo->realParameter[571] /* omega_c[67] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8663(DATA *data, threadData_t *threadData);


/*
equation index: 1069
type: SIMPLE_ASSIGN
vy[67] = cos(theta[67]) * r_init[67] * omega_c[67]
*/
void SpiralGalaxy_eqFunction_1069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1069};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[566]] /* vy[67] STATE(1) */) = (cos((data->simulationInfo->realParameter[1573] /* theta[67] PARAM */))) * (((data->simulationInfo->realParameter[1072] /* r_init[67] PARAM */)) * ((data->simulationInfo->realParameter[571] /* omega_c[67] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8662(DATA *data, threadData_t *threadData);


/*
equation index: 1071
type: SIMPLE_ASSIGN
vz[67] = 0.0
*/
void SpiralGalaxy_eqFunction_1071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1071};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1066]] /* vz[67] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8661(DATA *data, threadData_t *threadData);


/*
equation index: 1073
type: SIMPLE_ASSIGN
z[68] = -0.029120000000000004
*/
void SpiralGalaxy_eqFunction_1073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1073};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2567]] /* z[68] STATE(1,vz[68]) */) = -0.029120000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8674(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8675(DATA *data, threadData_t *threadData);


/*
equation index: 1076
type: SIMPLE_ASSIGN
y[68] = r_init[68] * sin(theta[68] - 0.00728)
*/
void SpiralGalaxy_eqFunction_1076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1076};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2067]] /* y[68] STATE(1,vy[68]) */) = ((data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */)) * (sin((data->simulationInfo->realParameter[1574] /* theta[68] PARAM */) - 0.00728));
  TRACE_POP
}

/*
equation index: 1077
type: SIMPLE_ASSIGN
x[68] = r_init[68] * cos(theta[68] - 0.00728)
*/
void SpiralGalaxy_eqFunction_1077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1077};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1567]] /* x[68] STATE(1,vx[68]) */) = ((data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */)) * (cos((data->simulationInfo->realParameter[1574] /* theta[68] PARAM */) - 0.00728));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8676(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8677(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8680(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8679(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8678(DATA *data, threadData_t *threadData);


/*
equation index: 1083
type: SIMPLE_ASSIGN
vx[68] = (-sin(theta[68])) * r_init[68] * omega_c[68]
*/
void SpiralGalaxy_eqFunction_1083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1083};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[67]] /* vx[68] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1574] /* theta[68] PARAM */)))) * (((data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */)) * ((data->simulationInfo->realParameter[572] /* omega_c[68] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8673(DATA *data, threadData_t *threadData);


/*
equation index: 1085
type: SIMPLE_ASSIGN
vy[68] = cos(theta[68]) * r_init[68] * omega_c[68]
*/
void SpiralGalaxy_eqFunction_1085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1085};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[567]] /* vy[68] STATE(1) */) = (cos((data->simulationInfo->realParameter[1574] /* theta[68] PARAM */))) * (((data->simulationInfo->realParameter[1073] /* r_init[68] PARAM */)) * ((data->simulationInfo->realParameter[572] /* omega_c[68] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8672(DATA *data, threadData_t *threadData);


/*
equation index: 1087
type: SIMPLE_ASSIGN
vz[68] = 0.0
*/
void SpiralGalaxy_eqFunction_1087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1087};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1067]] /* vz[68] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8671(DATA *data, threadData_t *threadData);


/*
equation index: 1089
type: SIMPLE_ASSIGN
z[69] = -0.028960000000000007
*/
void SpiralGalaxy_eqFunction_1089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1089};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2568]] /* z[69] STATE(1,vz[69]) */) = -0.028960000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8684(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8685(DATA *data, threadData_t *threadData);


/*
equation index: 1092
type: SIMPLE_ASSIGN
y[69] = r_init[69] * sin(theta[69] - 0.00724)
*/
void SpiralGalaxy_eqFunction_1092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1092};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2068]] /* y[69] STATE(1,vy[69]) */) = ((data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */)) * (sin((data->simulationInfo->realParameter[1575] /* theta[69] PARAM */) - 0.00724));
  TRACE_POP
}

/*
equation index: 1093
type: SIMPLE_ASSIGN
x[69] = r_init[69] * cos(theta[69] - 0.00724)
*/
void SpiralGalaxy_eqFunction_1093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1093};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1568]] /* x[69] STATE(1,vx[69]) */) = ((data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */)) * (cos((data->simulationInfo->realParameter[1575] /* theta[69] PARAM */) - 0.00724));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8686(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8687(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8690(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8689(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8688(DATA *data, threadData_t *threadData);


/*
equation index: 1099
type: SIMPLE_ASSIGN
vx[69] = (-sin(theta[69])) * r_init[69] * omega_c[69]
*/
void SpiralGalaxy_eqFunction_1099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1099};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[68]] /* vx[69] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1575] /* theta[69] PARAM */)))) * (((data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */)) * ((data->simulationInfo->realParameter[573] /* omega_c[69] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8683(DATA *data, threadData_t *threadData);


/*
equation index: 1101
type: SIMPLE_ASSIGN
vy[69] = cos(theta[69]) * r_init[69] * omega_c[69]
*/
void SpiralGalaxy_eqFunction_1101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[568]] /* vy[69] STATE(1) */) = (cos((data->simulationInfo->realParameter[1575] /* theta[69] PARAM */))) * (((data->simulationInfo->realParameter[1074] /* r_init[69] PARAM */)) * ((data->simulationInfo->realParameter[573] /* omega_c[69] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8682(DATA *data, threadData_t *threadData);


/*
equation index: 1103
type: SIMPLE_ASSIGN
vz[69] = 0.0
*/
void SpiralGalaxy_eqFunction_1103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1103};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1068]] /* vz[69] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8681(DATA *data, threadData_t *threadData);


/*
equation index: 1105
type: SIMPLE_ASSIGN
z[70] = -0.028800000000000003
*/
void SpiralGalaxy_eqFunction_1105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1105};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2569]] /* z[70] STATE(1,vz[70]) */) = -0.028800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8694(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8695(DATA *data, threadData_t *threadData);


/*
equation index: 1108
type: SIMPLE_ASSIGN
y[70] = r_init[70] * sin(theta[70] - 0.0072)
*/
void SpiralGalaxy_eqFunction_1108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1108};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2069]] /* y[70] STATE(1,vy[70]) */) = ((data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */)) * (sin((data->simulationInfo->realParameter[1576] /* theta[70] PARAM */) - 0.0072));
  TRACE_POP
}

/*
equation index: 1109
type: SIMPLE_ASSIGN
x[70] = r_init[70] * cos(theta[70] - 0.0072)
*/
void SpiralGalaxy_eqFunction_1109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1569]] /* x[70] STATE(1,vx[70]) */) = ((data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */)) * (cos((data->simulationInfo->realParameter[1576] /* theta[70] PARAM */) - 0.0072));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8696(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8697(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8700(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8699(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8698(DATA *data, threadData_t *threadData);


/*
equation index: 1115
type: SIMPLE_ASSIGN
vx[70] = (-sin(theta[70])) * r_init[70] * omega_c[70]
*/
void SpiralGalaxy_eqFunction_1115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[69]] /* vx[70] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1576] /* theta[70] PARAM */)))) * (((data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */)) * ((data->simulationInfo->realParameter[574] /* omega_c[70] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8693(DATA *data, threadData_t *threadData);


/*
equation index: 1117
type: SIMPLE_ASSIGN
vy[70] = cos(theta[70]) * r_init[70] * omega_c[70]
*/
void SpiralGalaxy_eqFunction_1117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[569]] /* vy[70] STATE(1) */) = (cos((data->simulationInfo->realParameter[1576] /* theta[70] PARAM */))) * (((data->simulationInfo->realParameter[1075] /* r_init[70] PARAM */)) * ((data->simulationInfo->realParameter[574] /* omega_c[70] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8692(DATA *data, threadData_t *threadData);


/*
equation index: 1119
type: SIMPLE_ASSIGN
vz[70] = 0.0
*/
void SpiralGalaxy_eqFunction_1119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1119};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1069]] /* vz[70] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8691(DATA *data, threadData_t *threadData);


/*
equation index: 1121
type: SIMPLE_ASSIGN
z[71] = -0.028640000000000002
*/
void SpiralGalaxy_eqFunction_1121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1121};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2570]] /* z[71] STATE(1,vz[71]) */) = -0.028640000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8704(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8705(DATA *data, threadData_t *threadData);


/*
equation index: 1124
type: SIMPLE_ASSIGN
y[71] = r_init[71] * sin(theta[71] - 0.00716)
*/
void SpiralGalaxy_eqFunction_1124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1124};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2070]] /* y[71] STATE(1,vy[71]) */) = ((data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */)) * (sin((data->simulationInfo->realParameter[1577] /* theta[71] PARAM */) - 0.00716));
  TRACE_POP
}

/*
equation index: 1125
type: SIMPLE_ASSIGN
x[71] = r_init[71] * cos(theta[71] - 0.00716)
*/
void SpiralGalaxy_eqFunction_1125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1570]] /* x[71] STATE(1,vx[71]) */) = ((data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */)) * (cos((data->simulationInfo->realParameter[1577] /* theta[71] PARAM */) - 0.00716));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8706(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8707(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8710(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8709(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8708(DATA *data, threadData_t *threadData);


/*
equation index: 1131
type: SIMPLE_ASSIGN
vx[71] = (-sin(theta[71])) * r_init[71] * omega_c[71]
*/
void SpiralGalaxy_eqFunction_1131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[70]] /* vx[71] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1577] /* theta[71] PARAM */)))) * (((data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */)) * ((data->simulationInfo->realParameter[575] /* omega_c[71] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8703(DATA *data, threadData_t *threadData);


/*
equation index: 1133
type: SIMPLE_ASSIGN
vy[71] = cos(theta[71]) * r_init[71] * omega_c[71]
*/
void SpiralGalaxy_eqFunction_1133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[570]] /* vy[71] STATE(1) */) = (cos((data->simulationInfo->realParameter[1577] /* theta[71] PARAM */))) * (((data->simulationInfo->realParameter[1076] /* r_init[71] PARAM */)) * ((data->simulationInfo->realParameter[575] /* omega_c[71] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8702(DATA *data, threadData_t *threadData);


/*
equation index: 1135
type: SIMPLE_ASSIGN
vz[71] = 0.0
*/
void SpiralGalaxy_eqFunction_1135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1135};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1070]] /* vz[71] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8701(DATA *data, threadData_t *threadData);


/*
equation index: 1137
type: SIMPLE_ASSIGN
z[72] = -0.028480000000000002
*/
void SpiralGalaxy_eqFunction_1137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1137};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2571]] /* z[72] STATE(1,vz[72]) */) = -0.028480000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8714(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8715(DATA *data, threadData_t *threadData);


/*
equation index: 1140
type: SIMPLE_ASSIGN
y[72] = r_init[72] * sin(theta[72] - 0.00712)
*/
void SpiralGalaxy_eqFunction_1140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1140};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2071]] /* y[72] STATE(1,vy[72]) */) = ((data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */)) * (sin((data->simulationInfo->realParameter[1578] /* theta[72] PARAM */) - 0.00712));
  TRACE_POP
}

/*
equation index: 1141
type: SIMPLE_ASSIGN
x[72] = r_init[72] * cos(theta[72] - 0.00712)
*/
void SpiralGalaxy_eqFunction_1141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1571]] /* x[72] STATE(1,vx[72]) */) = ((data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */)) * (cos((data->simulationInfo->realParameter[1578] /* theta[72] PARAM */) - 0.00712));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8716(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8717(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8720(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8719(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8718(DATA *data, threadData_t *threadData);


/*
equation index: 1147
type: SIMPLE_ASSIGN
vx[72] = (-sin(theta[72])) * r_init[72] * omega_c[72]
*/
void SpiralGalaxy_eqFunction_1147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[71]] /* vx[72] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1578] /* theta[72] PARAM */)))) * (((data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */)) * ((data->simulationInfo->realParameter[576] /* omega_c[72] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8713(DATA *data, threadData_t *threadData);


/*
equation index: 1149
type: SIMPLE_ASSIGN
vy[72] = cos(theta[72]) * r_init[72] * omega_c[72]
*/
void SpiralGalaxy_eqFunction_1149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[571]] /* vy[72] STATE(1) */) = (cos((data->simulationInfo->realParameter[1578] /* theta[72] PARAM */))) * (((data->simulationInfo->realParameter[1077] /* r_init[72] PARAM */)) * ((data->simulationInfo->realParameter[576] /* omega_c[72] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8712(DATA *data, threadData_t *threadData);


/*
equation index: 1151
type: SIMPLE_ASSIGN
vz[72] = 0.0
*/
void SpiralGalaxy_eqFunction_1151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1151};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1071]] /* vz[72] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8711(DATA *data, threadData_t *threadData);


/*
equation index: 1153
type: SIMPLE_ASSIGN
z[73] = -0.028319999999999998
*/
void SpiralGalaxy_eqFunction_1153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1153};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2572]] /* z[73] STATE(1,vz[73]) */) = -0.028319999999999998;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8724(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8725(DATA *data, threadData_t *threadData);


/*
equation index: 1156
type: SIMPLE_ASSIGN
y[73] = r_init[73] * sin(theta[73] - 0.0070799999999999995)
*/
void SpiralGalaxy_eqFunction_1156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1156};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2072]] /* y[73] STATE(1,vy[73]) */) = ((data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */)) * (sin((data->simulationInfo->realParameter[1579] /* theta[73] PARAM */) - 0.0070799999999999995));
  TRACE_POP
}

/*
equation index: 1157
type: SIMPLE_ASSIGN
x[73] = r_init[73] * cos(theta[73] - 0.0070799999999999995)
*/
void SpiralGalaxy_eqFunction_1157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1572]] /* x[73] STATE(1,vx[73]) */) = ((data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */)) * (cos((data->simulationInfo->realParameter[1579] /* theta[73] PARAM */) - 0.0070799999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8726(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8727(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8730(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8729(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8728(DATA *data, threadData_t *threadData);


/*
equation index: 1163
type: SIMPLE_ASSIGN
vx[73] = (-sin(theta[73])) * r_init[73] * omega_c[73]
*/
void SpiralGalaxy_eqFunction_1163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[72]] /* vx[73] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1579] /* theta[73] PARAM */)))) * (((data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */)) * ((data->simulationInfo->realParameter[577] /* omega_c[73] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8723(DATA *data, threadData_t *threadData);


/*
equation index: 1165
type: SIMPLE_ASSIGN
vy[73] = cos(theta[73]) * r_init[73] * omega_c[73]
*/
void SpiralGalaxy_eqFunction_1165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[572]] /* vy[73] STATE(1) */) = (cos((data->simulationInfo->realParameter[1579] /* theta[73] PARAM */))) * (((data->simulationInfo->realParameter[1078] /* r_init[73] PARAM */)) * ((data->simulationInfo->realParameter[577] /* omega_c[73] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8722(DATA *data, threadData_t *threadData);


/*
equation index: 1167
type: SIMPLE_ASSIGN
vz[73] = 0.0
*/
void SpiralGalaxy_eqFunction_1167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1167};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1072]] /* vz[73] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8721(DATA *data, threadData_t *threadData);


/*
equation index: 1169
type: SIMPLE_ASSIGN
z[74] = -0.02816
*/
void SpiralGalaxy_eqFunction_1169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1169};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2573]] /* z[74] STATE(1,vz[74]) */) = -0.02816;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8734(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8735(DATA *data, threadData_t *threadData);


/*
equation index: 1172
type: SIMPLE_ASSIGN
y[74] = r_init[74] * sin(theta[74] - 0.007039999999999999)
*/
void SpiralGalaxy_eqFunction_1172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1172};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2073]] /* y[74] STATE(1,vy[74]) */) = ((data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */)) * (sin((data->simulationInfo->realParameter[1580] /* theta[74] PARAM */) - 0.007039999999999999));
  TRACE_POP
}

/*
equation index: 1173
type: SIMPLE_ASSIGN
x[74] = r_init[74] * cos(theta[74] - 0.007039999999999999)
*/
void SpiralGalaxy_eqFunction_1173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1573]] /* x[74] STATE(1,vx[74]) */) = ((data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */)) * (cos((data->simulationInfo->realParameter[1580] /* theta[74] PARAM */) - 0.007039999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8736(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8737(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8740(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8739(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8738(DATA *data, threadData_t *threadData);


/*
equation index: 1179
type: SIMPLE_ASSIGN
vx[74] = (-sin(theta[74])) * r_init[74] * omega_c[74]
*/
void SpiralGalaxy_eqFunction_1179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[73]] /* vx[74] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1580] /* theta[74] PARAM */)))) * (((data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */)) * ((data->simulationInfo->realParameter[578] /* omega_c[74] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8733(DATA *data, threadData_t *threadData);


/*
equation index: 1181
type: SIMPLE_ASSIGN
vy[74] = cos(theta[74]) * r_init[74] * omega_c[74]
*/
void SpiralGalaxy_eqFunction_1181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[573]] /* vy[74] STATE(1) */) = (cos((data->simulationInfo->realParameter[1580] /* theta[74] PARAM */))) * (((data->simulationInfo->realParameter[1079] /* r_init[74] PARAM */)) * ((data->simulationInfo->realParameter[578] /* omega_c[74] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8732(DATA *data, threadData_t *threadData);


/*
equation index: 1183
type: SIMPLE_ASSIGN
vz[74] = 0.0
*/
void SpiralGalaxy_eqFunction_1183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1183};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1073]] /* vz[74] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8731(DATA *data, threadData_t *threadData);


/*
equation index: 1185
type: SIMPLE_ASSIGN
z[75] = -0.028000000000000004
*/
void SpiralGalaxy_eqFunction_1185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1185};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2574]] /* z[75] STATE(1,vz[75]) */) = -0.028000000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8744(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8745(DATA *data, threadData_t *threadData);


/*
equation index: 1188
type: SIMPLE_ASSIGN
y[75] = r_init[75] * sin(theta[75] - 0.006999999999999999)
*/
void SpiralGalaxy_eqFunction_1188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1188};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2074]] /* y[75] STATE(1,vy[75]) */) = ((data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */)) * (sin((data->simulationInfo->realParameter[1581] /* theta[75] PARAM */) - 0.006999999999999999));
  TRACE_POP
}

/*
equation index: 1189
type: SIMPLE_ASSIGN
x[75] = r_init[75] * cos(theta[75] - 0.006999999999999999)
*/
void SpiralGalaxy_eqFunction_1189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1574]] /* x[75] STATE(1,vx[75]) */) = ((data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */)) * (cos((data->simulationInfo->realParameter[1581] /* theta[75] PARAM */) - 0.006999999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8746(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8747(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8750(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8749(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8748(DATA *data, threadData_t *threadData);


/*
equation index: 1195
type: SIMPLE_ASSIGN
vx[75] = (-sin(theta[75])) * r_init[75] * omega_c[75]
*/
void SpiralGalaxy_eqFunction_1195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[74]] /* vx[75] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1581] /* theta[75] PARAM */)))) * (((data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */)) * ((data->simulationInfo->realParameter[579] /* omega_c[75] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8743(DATA *data, threadData_t *threadData);


/*
equation index: 1197
type: SIMPLE_ASSIGN
vy[75] = cos(theta[75]) * r_init[75] * omega_c[75]
*/
void SpiralGalaxy_eqFunction_1197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[574]] /* vy[75] STATE(1) */) = (cos((data->simulationInfo->realParameter[1581] /* theta[75] PARAM */))) * (((data->simulationInfo->realParameter[1080] /* r_init[75] PARAM */)) * ((data->simulationInfo->realParameter[579] /* omega_c[75] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8742(DATA *data, threadData_t *threadData);


/*
equation index: 1199
type: SIMPLE_ASSIGN
vz[75] = 0.0
*/
void SpiralGalaxy_eqFunction_1199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1199};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1074]] /* vz[75] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8741(DATA *data, threadData_t *threadData);


/*
equation index: 1201
type: SIMPLE_ASSIGN
z[76] = -0.027840000000000004
*/
void SpiralGalaxy_eqFunction_1201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1201};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2575]] /* z[76] STATE(1,vz[76]) */) = -0.027840000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8754(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8755(DATA *data, threadData_t *threadData);


/*
equation index: 1204
type: SIMPLE_ASSIGN
y[76] = r_init[76] * sin(theta[76] - 0.00696)
*/
void SpiralGalaxy_eqFunction_1204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1204};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2075]] /* y[76] STATE(1,vy[76]) */) = ((data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */)) * (sin((data->simulationInfo->realParameter[1582] /* theta[76] PARAM */) - 0.00696));
  TRACE_POP
}

/*
equation index: 1205
type: SIMPLE_ASSIGN
x[76] = r_init[76] * cos(theta[76] - 0.00696)
*/
void SpiralGalaxy_eqFunction_1205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1575]] /* x[76] STATE(1,vx[76]) */) = ((data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */)) * (cos((data->simulationInfo->realParameter[1582] /* theta[76] PARAM */) - 0.00696));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8756(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8757(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8760(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8759(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8758(DATA *data, threadData_t *threadData);


/*
equation index: 1211
type: SIMPLE_ASSIGN
vx[76] = (-sin(theta[76])) * r_init[76] * omega_c[76]
*/
void SpiralGalaxy_eqFunction_1211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[75]] /* vx[76] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1582] /* theta[76] PARAM */)))) * (((data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */)) * ((data->simulationInfo->realParameter[580] /* omega_c[76] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8753(DATA *data, threadData_t *threadData);


/*
equation index: 1213
type: SIMPLE_ASSIGN
vy[76] = cos(theta[76]) * r_init[76] * omega_c[76]
*/
void SpiralGalaxy_eqFunction_1213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[575]] /* vy[76] STATE(1) */) = (cos((data->simulationInfo->realParameter[1582] /* theta[76] PARAM */))) * (((data->simulationInfo->realParameter[1081] /* r_init[76] PARAM */)) * ((data->simulationInfo->realParameter[580] /* omega_c[76] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8752(DATA *data, threadData_t *threadData);


/*
equation index: 1215
type: SIMPLE_ASSIGN
vz[76] = 0.0
*/
void SpiralGalaxy_eqFunction_1215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1215};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1075]] /* vz[76] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8751(DATA *data, threadData_t *threadData);


/*
equation index: 1217
type: SIMPLE_ASSIGN
z[77] = -0.027680000000000003
*/
void SpiralGalaxy_eqFunction_1217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1217};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2576]] /* z[77] STATE(1,vz[77]) */) = -0.027680000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8764(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8765(DATA *data, threadData_t *threadData);


/*
equation index: 1220
type: SIMPLE_ASSIGN
y[77] = r_init[77] * sin(theta[77] - 0.00692)
*/
void SpiralGalaxy_eqFunction_1220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1220};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2076]] /* y[77] STATE(1,vy[77]) */) = ((data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */)) * (sin((data->simulationInfo->realParameter[1583] /* theta[77] PARAM */) - 0.00692));
  TRACE_POP
}

/*
equation index: 1221
type: SIMPLE_ASSIGN
x[77] = r_init[77] * cos(theta[77] - 0.00692)
*/
void SpiralGalaxy_eqFunction_1221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1576]] /* x[77] STATE(1,vx[77]) */) = ((data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */)) * (cos((data->simulationInfo->realParameter[1583] /* theta[77] PARAM */) - 0.00692));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8766(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8767(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8770(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8769(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8768(DATA *data, threadData_t *threadData);


/*
equation index: 1227
type: SIMPLE_ASSIGN
vx[77] = (-sin(theta[77])) * r_init[77] * omega_c[77]
*/
void SpiralGalaxy_eqFunction_1227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[76]] /* vx[77] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1583] /* theta[77] PARAM */)))) * (((data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */)) * ((data->simulationInfo->realParameter[581] /* omega_c[77] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8763(DATA *data, threadData_t *threadData);


/*
equation index: 1229
type: SIMPLE_ASSIGN
vy[77] = cos(theta[77]) * r_init[77] * omega_c[77]
*/
void SpiralGalaxy_eqFunction_1229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[576]] /* vy[77] STATE(1) */) = (cos((data->simulationInfo->realParameter[1583] /* theta[77] PARAM */))) * (((data->simulationInfo->realParameter[1082] /* r_init[77] PARAM */)) * ((data->simulationInfo->realParameter[581] /* omega_c[77] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8762(DATA *data, threadData_t *threadData);


/*
equation index: 1231
type: SIMPLE_ASSIGN
vz[77] = 0.0
*/
void SpiralGalaxy_eqFunction_1231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1231};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1076]] /* vz[77] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8761(DATA *data, threadData_t *threadData);


/*
equation index: 1233
type: SIMPLE_ASSIGN
z[78] = -0.02752
*/
void SpiralGalaxy_eqFunction_1233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1233};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2577]] /* z[78] STATE(1,vz[78]) */) = -0.02752;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8774(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8775(DATA *data, threadData_t *threadData);


/*
equation index: 1236
type: SIMPLE_ASSIGN
y[78] = r_init[78] * sin(theta[78] - 0.00688)
*/
void SpiralGalaxy_eqFunction_1236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1236};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2077]] /* y[78] STATE(1,vy[78]) */) = ((data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */)) * (sin((data->simulationInfo->realParameter[1584] /* theta[78] PARAM */) - 0.00688));
  TRACE_POP
}

/*
equation index: 1237
type: SIMPLE_ASSIGN
x[78] = r_init[78] * cos(theta[78] - 0.00688)
*/
void SpiralGalaxy_eqFunction_1237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1577]] /* x[78] STATE(1,vx[78]) */) = ((data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */)) * (cos((data->simulationInfo->realParameter[1584] /* theta[78] PARAM */) - 0.00688));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8776(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8777(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8780(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8779(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8778(DATA *data, threadData_t *threadData);


/*
equation index: 1243
type: SIMPLE_ASSIGN
vx[78] = (-sin(theta[78])) * r_init[78] * omega_c[78]
*/
void SpiralGalaxy_eqFunction_1243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[77]] /* vx[78] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1584] /* theta[78] PARAM */)))) * (((data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */)) * ((data->simulationInfo->realParameter[582] /* omega_c[78] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8773(DATA *data, threadData_t *threadData);


/*
equation index: 1245
type: SIMPLE_ASSIGN
vy[78] = cos(theta[78]) * r_init[78] * omega_c[78]
*/
void SpiralGalaxy_eqFunction_1245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[577]] /* vy[78] STATE(1) */) = (cos((data->simulationInfo->realParameter[1584] /* theta[78] PARAM */))) * (((data->simulationInfo->realParameter[1083] /* r_init[78] PARAM */)) * ((data->simulationInfo->realParameter[582] /* omega_c[78] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8772(DATA *data, threadData_t *threadData);


/*
equation index: 1247
type: SIMPLE_ASSIGN
vz[78] = 0.0
*/
void SpiralGalaxy_eqFunction_1247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1247};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1077]] /* vz[78] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8771(DATA *data, threadData_t *threadData);


/*
equation index: 1249
type: SIMPLE_ASSIGN
z[79] = -0.027360000000000002
*/
void SpiralGalaxy_eqFunction_1249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1249};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2578]] /* z[79] STATE(1,vz[79]) */) = -0.027360000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8784(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8785(DATA *data, threadData_t *threadData);


/*
equation index: 1252
type: SIMPLE_ASSIGN
y[79] = r_init[79] * sin(theta[79] - 0.00684)
*/
void SpiralGalaxy_eqFunction_1252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1252};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2078]] /* y[79] STATE(1,vy[79]) */) = ((data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */)) * (sin((data->simulationInfo->realParameter[1585] /* theta[79] PARAM */) - 0.00684));
  TRACE_POP
}

/*
equation index: 1253
type: SIMPLE_ASSIGN
x[79] = r_init[79] * cos(theta[79] - 0.00684)
*/
void SpiralGalaxy_eqFunction_1253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1578]] /* x[79] STATE(1,vx[79]) */) = ((data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */)) * (cos((data->simulationInfo->realParameter[1585] /* theta[79] PARAM */) - 0.00684));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8786(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8787(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8790(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8789(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8788(DATA *data, threadData_t *threadData);


/*
equation index: 1259
type: SIMPLE_ASSIGN
vx[79] = (-sin(theta[79])) * r_init[79] * omega_c[79]
*/
void SpiralGalaxy_eqFunction_1259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[78]] /* vx[79] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1585] /* theta[79] PARAM */)))) * (((data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */)) * ((data->simulationInfo->realParameter[583] /* omega_c[79] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8783(DATA *data, threadData_t *threadData);


/*
equation index: 1261
type: SIMPLE_ASSIGN
vy[79] = cos(theta[79]) * r_init[79] * omega_c[79]
*/
void SpiralGalaxy_eqFunction_1261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[578]] /* vy[79] STATE(1) */) = (cos((data->simulationInfo->realParameter[1585] /* theta[79] PARAM */))) * (((data->simulationInfo->realParameter[1084] /* r_init[79] PARAM */)) * ((data->simulationInfo->realParameter[583] /* omega_c[79] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8782(DATA *data, threadData_t *threadData);


/*
equation index: 1263
type: SIMPLE_ASSIGN
vz[79] = 0.0
*/
void SpiralGalaxy_eqFunction_1263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1263};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1078]] /* vz[79] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8781(DATA *data, threadData_t *threadData);


/*
equation index: 1265
type: SIMPLE_ASSIGN
z[80] = -0.027200000000000002
*/
void SpiralGalaxy_eqFunction_1265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1265};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2579]] /* z[80] STATE(1,vz[80]) */) = -0.027200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8794(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8795(DATA *data, threadData_t *threadData);


/*
equation index: 1268
type: SIMPLE_ASSIGN
y[80] = r_init[80] * sin(theta[80] - 0.0068)
*/
void SpiralGalaxy_eqFunction_1268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1268};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2079]] /* y[80] STATE(1,vy[80]) */) = ((data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */)) * (sin((data->simulationInfo->realParameter[1586] /* theta[80] PARAM */) - 0.0068));
  TRACE_POP
}

/*
equation index: 1269
type: SIMPLE_ASSIGN
x[80] = r_init[80] * cos(theta[80] - 0.0068)
*/
void SpiralGalaxy_eqFunction_1269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1579]] /* x[80] STATE(1,vx[80]) */) = ((data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */)) * (cos((data->simulationInfo->realParameter[1586] /* theta[80] PARAM */) - 0.0068));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8796(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8797(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8800(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8799(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8798(DATA *data, threadData_t *threadData);


/*
equation index: 1275
type: SIMPLE_ASSIGN
vx[80] = (-sin(theta[80])) * r_init[80] * omega_c[80]
*/
void SpiralGalaxy_eqFunction_1275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[79]] /* vx[80] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1586] /* theta[80] PARAM */)))) * (((data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */)) * ((data->simulationInfo->realParameter[584] /* omega_c[80] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8793(DATA *data, threadData_t *threadData);


/*
equation index: 1277
type: SIMPLE_ASSIGN
vy[80] = cos(theta[80]) * r_init[80] * omega_c[80]
*/
void SpiralGalaxy_eqFunction_1277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[579]] /* vy[80] STATE(1) */) = (cos((data->simulationInfo->realParameter[1586] /* theta[80] PARAM */))) * (((data->simulationInfo->realParameter[1085] /* r_init[80] PARAM */)) * ((data->simulationInfo->realParameter[584] /* omega_c[80] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8792(DATA *data, threadData_t *threadData);


/*
equation index: 1279
type: SIMPLE_ASSIGN
vz[80] = 0.0
*/
void SpiralGalaxy_eqFunction_1279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1279};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1079]] /* vz[80] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8791(DATA *data, threadData_t *threadData);


/*
equation index: 1281
type: SIMPLE_ASSIGN
z[81] = -0.027040000000000005
*/
void SpiralGalaxy_eqFunction_1281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1281};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2580]] /* z[81] STATE(1,vz[81]) */) = -0.027040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8804(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8805(DATA *data, threadData_t *threadData);


/*
equation index: 1284
type: SIMPLE_ASSIGN
y[81] = r_init[81] * sin(theta[81] - 0.0067599999999999995)
*/
void SpiralGalaxy_eqFunction_1284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1284};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2080]] /* y[81] STATE(1,vy[81]) */) = ((data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */)) * (sin((data->simulationInfo->realParameter[1587] /* theta[81] PARAM */) - 0.0067599999999999995));
  TRACE_POP
}

/*
equation index: 1285
type: SIMPLE_ASSIGN
x[81] = r_init[81] * cos(theta[81] - 0.0067599999999999995)
*/
void SpiralGalaxy_eqFunction_1285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1580]] /* x[81] STATE(1,vx[81]) */) = ((data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */)) * (cos((data->simulationInfo->realParameter[1587] /* theta[81] PARAM */) - 0.0067599999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8806(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8807(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8810(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8809(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8808(DATA *data, threadData_t *threadData);


/*
equation index: 1291
type: SIMPLE_ASSIGN
vx[81] = (-sin(theta[81])) * r_init[81] * omega_c[81]
*/
void SpiralGalaxy_eqFunction_1291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[80]] /* vx[81] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1587] /* theta[81] PARAM */)))) * (((data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */)) * ((data->simulationInfo->realParameter[585] /* omega_c[81] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8803(DATA *data, threadData_t *threadData);


/*
equation index: 1293
type: SIMPLE_ASSIGN
vy[81] = cos(theta[81]) * r_init[81] * omega_c[81]
*/
void SpiralGalaxy_eqFunction_1293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[580]] /* vy[81] STATE(1) */) = (cos((data->simulationInfo->realParameter[1587] /* theta[81] PARAM */))) * (((data->simulationInfo->realParameter[1086] /* r_init[81] PARAM */)) * ((data->simulationInfo->realParameter[585] /* omega_c[81] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8802(DATA *data, threadData_t *threadData);


/*
equation index: 1295
type: SIMPLE_ASSIGN
vz[81] = 0.0
*/
void SpiralGalaxy_eqFunction_1295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1080]] /* vz[81] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8801(DATA *data, threadData_t *threadData);


/*
equation index: 1297
type: SIMPLE_ASSIGN
z[82] = -0.02688
*/
void SpiralGalaxy_eqFunction_1297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2581]] /* z[82] STATE(1,vz[82]) */) = -0.02688;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8814(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8815(DATA *data, threadData_t *threadData);


/*
equation index: 1300
type: SIMPLE_ASSIGN
y[82] = r_init[82] * sin(theta[82] - 0.006719999999999999)
*/
void SpiralGalaxy_eqFunction_1300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1300};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2081]] /* y[82] STATE(1,vy[82]) */) = ((data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */)) * (sin((data->simulationInfo->realParameter[1588] /* theta[82] PARAM */) - 0.006719999999999999));
  TRACE_POP
}

/*
equation index: 1301
type: SIMPLE_ASSIGN
x[82] = r_init[82] * cos(theta[82] - 0.006719999999999999)
*/
void SpiralGalaxy_eqFunction_1301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1581]] /* x[82] STATE(1,vx[82]) */) = ((data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */)) * (cos((data->simulationInfo->realParameter[1588] /* theta[82] PARAM */) - 0.006719999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8816(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8817(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8820(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8819(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8818(DATA *data, threadData_t *threadData);


/*
equation index: 1307
type: SIMPLE_ASSIGN
vx[82] = (-sin(theta[82])) * r_init[82] * omega_c[82]
*/
void SpiralGalaxy_eqFunction_1307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[81]] /* vx[82] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1588] /* theta[82] PARAM */)))) * (((data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */)) * ((data->simulationInfo->realParameter[586] /* omega_c[82] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8813(DATA *data, threadData_t *threadData);


/*
equation index: 1309
type: SIMPLE_ASSIGN
vy[82] = cos(theta[82]) * r_init[82] * omega_c[82]
*/
void SpiralGalaxy_eqFunction_1309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[581]] /* vy[82] STATE(1) */) = (cos((data->simulationInfo->realParameter[1588] /* theta[82] PARAM */))) * (((data->simulationInfo->realParameter[1087] /* r_init[82] PARAM */)) * ((data->simulationInfo->realParameter[586] /* omega_c[82] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8812(DATA *data, threadData_t *threadData);


/*
equation index: 1311
type: SIMPLE_ASSIGN
vz[82] = 0.0
*/
void SpiralGalaxy_eqFunction_1311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1081]] /* vz[82] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8811(DATA *data, threadData_t *threadData);


/*
equation index: 1313
type: SIMPLE_ASSIGN
z[83] = -0.02672
*/
void SpiralGalaxy_eqFunction_1313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2582]] /* z[83] STATE(1,vz[83]) */) = -0.02672;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8824(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8825(DATA *data, threadData_t *threadData);


/*
equation index: 1316
type: SIMPLE_ASSIGN
y[83] = r_init[83] * sin(theta[83] - 0.006679999999999999)
*/
void SpiralGalaxy_eqFunction_1316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1316};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2082]] /* y[83] STATE(1,vy[83]) */) = ((data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */)) * (sin((data->simulationInfo->realParameter[1589] /* theta[83] PARAM */) - 0.006679999999999999));
  TRACE_POP
}

/*
equation index: 1317
type: SIMPLE_ASSIGN
x[83] = r_init[83] * cos(theta[83] - 0.006679999999999999)
*/
void SpiralGalaxy_eqFunction_1317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1582]] /* x[83] STATE(1,vx[83]) */) = ((data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */)) * (cos((data->simulationInfo->realParameter[1589] /* theta[83] PARAM */) - 0.006679999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8826(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8827(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8830(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8829(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8828(DATA *data, threadData_t *threadData);


/*
equation index: 1323
type: SIMPLE_ASSIGN
vx[83] = (-sin(theta[83])) * r_init[83] * omega_c[83]
*/
void SpiralGalaxy_eqFunction_1323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[82]] /* vx[83] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1589] /* theta[83] PARAM */)))) * (((data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */)) * ((data->simulationInfo->realParameter[587] /* omega_c[83] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8823(DATA *data, threadData_t *threadData);


/*
equation index: 1325
type: SIMPLE_ASSIGN
vy[83] = cos(theta[83]) * r_init[83] * omega_c[83]
*/
void SpiralGalaxy_eqFunction_1325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[582]] /* vy[83] STATE(1) */) = (cos((data->simulationInfo->realParameter[1589] /* theta[83] PARAM */))) * (((data->simulationInfo->realParameter[1088] /* r_init[83] PARAM */)) * ((data->simulationInfo->realParameter[587] /* omega_c[83] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8822(DATA *data, threadData_t *threadData);


/*
equation index: 1327
type: SIMPLE_ASSIGN
vz[83] = 0.0
*/
void SpiralGalaxy_eqFunction_1327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1327};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1082]] /* vz[83] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8821(DATA *data, threadData_t *threadData);


/*
equation index: 1329
type: SIMPLE_ASSIGN
z[84] = -0.026560000000000004
*/
void SpiralGalaxy_eqFunction_1329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1329};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2583]] /* z[84] STATE(1,vz[84]) */) = -0.026560000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8834(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8835(DATA *data, threadData_t *threadData);


/*
equation index: 1332
type: SIMPLE_ASSIGN
y[84] = r_init[84] * sin(theta[84] - 0.006639999999999999)
*/
void SpiralGalaxy_eqFunction_1332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1332};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2083]] /* y[84] STATE(1,vy[84]) */) = ((data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */)) * (sin((data->simulationInfo->realParameter[1590] /* theta[84] PARAM */) - 0.006639999999999999));
  TRACE_POP
}

/*
equation index: 1333
type: SIMPLE_ASSIGN
x[84] = r_init[84] * cos(theta[84] - 0.006639999999999999)
*/
void SpiralGalaxy_eqFunction_1333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1583]] /* x[84] STATE(1,vx[84]) */) = ((data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */)) * (cos((data->simulationInfo->realParameter[1590] /* theta[84] PARAM */) - 0.006639999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8836(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8837(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8840(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8839(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8838(DATA *data, threadData_t *threadData);


/*
equation index: 1339
type: SIMPLE_ASSIGN
vx[84] = (-sin(theta[84])) * r_init[84] * omega_c[84]
*/
void SpiralGalaxy_eqFunction_1339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[83]] /* vx[84] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1590] /* theta[84] PARAM */)))) * (((data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */)) * ((data->simulationInfo->realParameter[588] /* omega_c[84] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8833(DATA *data, threadData_t *threadData);


/*
equation index: 1341
type: SIMPLE_ASSIGN
vy[84] = cos(theta[84]) * r_init[84] * omega_c[84]
*/
void SpiralGalaxy_eqFunction_1341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[583]] /* vy[84] STATE(1) */) = (cos((data->simulationInfo->realParameter[1590] /* theta[84] PARAM */))) * (((data->simulationInfo->realParameter[1089] /* r_init[84] PARAM */)) * ((data->simulationInfo->realParameter[588] /* omega_c[84] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8832(DATA *data, threadData_t *threadData);


/*
equation index: 1343
type: SIMPLE_ASSIGN
vz[84] = 0.0
*/
void SpiralGalaxy_eqFunction_1343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1083]] /* vz[84] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8831(DATA *data, threadData_t *threadData);


/*
equation index: 1345
type: SIMPLE_ASSIGN
z[85] = -0.026400000000000003
*/
void SpiralGalaxy_eqFunction_1345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2584]] /* z[85] STATE(1,vz[85]) */) = -0.026400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8844(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8845(DATA *data, threadData_t *threadData);


/*
equation index: 1348
type: SIMPLE_ASSIGN
y[85] = r_init[85] * sin(theta[85] - 0.006599999999999999)
*/
void SpiralGalaxy_eqFunction_1348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1348};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2084]] /* y[85] STATE(1,vy[85]) */) = ((data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */)) * (sin((data->simulationInfo->realParameter[1591] /* theta[85] PARAM */) - 0.006599999999999999));
  TRACE_POP
}

/*
equation index: 1349
type: SIMPLE_ASSIGN
x[85] = r_init[85] * cos(theta[85] - 0.006599999999999999)
*/
void SpiralGalaxy_eqFunction_1349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1584]] /* x[85] STATE(1,vx[85]) */) = ((data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */)) * (cos((data->simulationInfo->realParameter[1591] /* theta[85] PARAM */) - 0.006599999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8846(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8847(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8850(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8849(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8848(DATA *data, threadData_t *threadData);


/*
equation index: 1355
type: SIMPLE_ASSIGN
vx[85] = (-sin(theta[85])) * r_init[85] * omega_c[85]
*/
void SpiralGalaxy_eqFunction_1355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[84]] /* vx[85] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1591] /* theta[85] PARAM */)))) * (((data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */)) * ((data->simulationInfo->realParameter[589] /* omega_c[85] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8843(DATA *data, threadData_t *threadData);


/*
equation index: 1357
type: SIMPLE_ASSIGN
vy[85] = cos(theta[85]) * r_init[85] * omega_c[85]
*/
void SpiralGalaxy_eqFunction_1357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[584]] /* vy[85] STATE(1) */) = (cos((data->simulationInfo->realParameter[1591] /* theta[85] PARAM */))) * (((data->simulationInfo->realParameter[1090] /* r_init[85] PARAM */)) * ((data->simulationInfo->realParameter[589] /* omega_c[85] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8842(DATA *data, threadData_t *threadData);


/*
equation index: 1359
type: SIMPLE_ASSIGN
vz[85] = 0.0
*/
void SpiralGalaxy_eqFunction_1359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1359};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1084]] /* vz[85] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8841(DATA *data, threadData_t *threadData);


/*
equation index: 1361
type: SIMPLE_ASSIGN
z[86] = -0.026240000000000003
*/
void SpiralGalaxy_eqFunction_1361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1361};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2585]] /* z[86] STATE(1,vz[86]) */) = -0.026240000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8854(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8855(DATA *data, threadData_t *threadData);


/*
equation index: 1364
type: SIMPLE_ASSIGN
y[86] = r_init[86] * sin(theta[86] - 0.006560000000000001)
*/
void SpiralGalaxy_eqFunction_1364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1364};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2085]] /* y[86] STATE(1,vy[86]) */) = ((data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */)) * (sin((data->simulationInfo->realParameter[1592] /* theta[86] PARAM */) - 0.006560000000000001));
  TRACE_POP
}

/*
equation index: 1365
type: SIMPLE_ASSIGN
x[86] = r_init[86] * cos(theta[86] - 0.006560000000000001)
*/
void SpiralGalaxy_eqFunction_1365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1585]] /* x[86] STATE(1,vx[86]) */) = ((data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */)) * (cos((data->simulationInfo->realParameter[1592] /* theta[86] PARAM */) - 0.006560000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8856(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8857(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8860(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8859(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8858(DATA *data, threadData_t *threadData);


/*
equation index: 1371
type: SIMPLE_ASSIGN
vx[86] = (-sin(theta[86])) * r_init[86] * omega_c[86]
*/
void SpiralGalaxy_eqFunction_1371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1371};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[85]] /* vx[86] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1592] /* theta[86] PARAM */)))) * (((data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */)) * ((data->simulationInfo->realParameter[590] /* omega_c[86] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8853(DATA *data, threadData_t *threadData);


/*
equation index: 1373
type: SIMPLE_ASSIGN
vy[86] = cos(theta[86]) * r_init[86] * omega_c[86]
*/
void SpiralGalaxy_eqFunction_1373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[585]] /* vy[86] STATE(1) */) = (cos((data->simulationInfo->realParameter[1592] /* theta[86] PARAM */))) * (((data->simulationInfo->realParameter[1091] /* r_init[86] PARAM */)) * ((data->simulationInfo->realParameter[590] /* omega_c[86] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8852(DATA *data, threadData_t *threadData);


/*
equation index: 1375
type: SIMPLE_ASSIGN
vz[86] = 0.0
*/
void SpiralGalaxy_eqFunction_1375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1375};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1085]] /* vz[86] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8851(DATA *data, threadData_t *threadData);


/*
equation index: 1377
type: SIMPLE_ASSIGN
z[87] = -0.026080000000000002
*/
void SpiralGalaxy_eqFunction_1377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1377};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2586]] /* z[87] STATE(1,vz[87]) */) = -0.026080000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8864(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8865(DATA *data, threadData_t *threadData);


/*
equation index: 1380
type: SIMPLE_ASSIGN
y[87] = r_init[87] * sin(theta[87] - 0.006520000000000001)
*/
void SpiralGalaxy_eqFunction_1380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1380};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2086]] /* y[87] STATE(1,vy[87]) */) = ((data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */)) * (sin((data->simulationInfo->realParameter[1593] /* theta[87] PARAM */) - 0.006520000000000001));
  TRACE_POP
}

/*
equation index: 1381
type: SIMPLE_ASSIGN
x[87] = r_init[87] * cos(theta[87] - 0.006520000000000001)
*/
void SpiralGalaxy_eqFunction_1381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1586]] /* x[87] STATE(1,vx[87]) */) = ((data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */)) * (cos((data->simulationInfo->realParameter[1593] /* theta[87] PARAM */) - 0.006520000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8866(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8867(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8870(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8869(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8868(DATA *data, threadData_t *threadData);


/*
equation index: 1387
type: SIMPLE_ASSIGN
vx[87] = (-sin(theta[87])) * r_init[87] * omega_c[87]
*/
void SpiralGalaxy_eqFunction_1387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[86]] /* vx[87] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1593] /* theta[87] PARAM */)))) * (((data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */)) * ((data->simulationInfo->realParameter[591] /* omega_c[87] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8863(DATA *data, threadData_t *threadData);


/*
equation index: 1389
type: SIMPLE_ASSIGN
vy[87] = cos(theta[87]) * r_init[87] * omega_c[87]
*/
void SpiralGalaxy_eqFunction_1389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[586]] /* vy[87] STATE(1) */) = (cos((data->simulationInfo->realParameter[1593] /* theta[87] PARAM */))) * (((data->simulationInfo->realParameter[1092] /* r_init[87] PARAM */)) * ((data->simulationInfo->realParameter[591] /* omega_c[87] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8862(DATA *data, threadData_t *threadData);


/*
equation index: 1391
type: SIMPLE_ASSIGN
vz[87] = 0.0
*/
void SpiralGalaxy_eqFunction_1391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1391};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1086]] /* vz[87] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8861(DATA *data, threadData_t *threadData);


/*
equation index: 1393
type: SIMPLE_ASSIGN
z[88] = -0.025920000000000002
*/
void SpiralGalaxy_eqFunction_1393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1393};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2587]] /* z[88] STATE(1,vz[88]) */) = -0.025920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8874(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8875(DATA *data, threadData_t *threadData);


/*
equation index: 1396
type: SIMPLE_ASSIGN
y[88] = r_init[88] * sin(theta[88] - 0.0064800000000000005)
*/
void SpiralGalaxy_eqFunction_1396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1396};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2087]] /* y[88] STATE(1,vy[88]) */) = ((data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */)) * (sin((data->simulationInfo->realParameter[1594] /* theta[88] PARAM */) - 0.0064800000000000005));
  TRACE_POP
}

/*
equation index: 1397
type: SIMPLE_ASSIGN
x[88] = r_init[88] * cos(theta[88] - 0.0064800000000000005)
*/
void SpiralGalaxy_eqFunction_1397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1587]] /* x[88] STATE(1,vx[88]) */) = ((data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */)) * (cos((data->simulationInfo->realParameter[1594] /* theta[88] PARAM */) - 0.0064800000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8876(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8877(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8880(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8879(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8878(DATA *data, threadData_t *threadData);


/*
equation index: 1403
type: SIMPLE_ASSIGN
vx[88] = (-sin(theta[88])) * r_init[88] * omega_c[88]
*/
void SpiralGalaxy_eqFunction_1403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[87]] /* vx[88] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1594] /* theta[88] PARAM */)))) * (((data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */)) * ((data->simulationInfo->realParameter[592] /* omega_c[88] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8873(DATA *data, threadData_t *threadData);


/*
equation index: 1405
type: SIMPLE_ASSIGN
vy[88] = cos(theta[88]) * r_init[88] * omega_c[88]
*/
void SpiralGalaxy_eqFunction_1405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[587]] /* vy[88] STATE(1) */) = (cos((data->simulationInfo->realParameter[1594] /* theta[88] PARAM */))) * (((data->simulationInfo->realParameter[1093] /* r_init[88] PARAM */)) * ((data->simulationInfo->realParameter[592] /* omega_c[88] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8872(DATA *data, threadData_t *threadData);


/*
equation index: 1407
type: SIMPLE_ASSIGN
vz[88] = 0.0
*/
void SpiralGalaxy_eqFunction_1407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1407};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1087]] /* vz[88] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8871(DATA *data, threadData_t *threadData);


/*
equation index: 1409
type: SIMPLE_ASSIGN
z[89] = -0.02576
*/
void SpiralGalaxy_eqFunction_1409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1409};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2588]] /* z[89] STATE(1,vz[89]) */) = -0.02576;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8884(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8885(DATA *data, threadData_t *threadData);


/*
equation index: 1412
type: SIMPLE_ASSIGN
y[89] = r_init[89] * sin(theta[89] - 0.00644)
*/
void SpiralGalaxy_eqFunction_1412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1412};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2088]] /* y[89] STATE(1,vy[89]) */) = ((data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */)) * (sin((data->simulationInfo->realParameter[1595] /* theta[89] PARAM */) - 0.00644));
  TRACE_POP
}

/*
equation index: 1413
type: SIMPLE_ASSIGN
x[89] = r_init[89] * cos(theta[89] - 0.00644)
*/
void SpiralGalaxy_eqFunction_1413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1588]] /* x[89] STATE(1,vx[89]) */) = ((data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */)) * (cos((data->simulationInfo->realParameter[1595] /* theta[89] PARAM */) - 0.00644));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8886(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8887(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8890(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8889(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8888(DATA *data, threadData_t *threadData);


/*
equation index: 1419
type: SIMPLE_ASSIGN
vx[89] = (-sin(theta[89])) * r_init[89] * omega_c[89]
*/
void SpiralGalaxy_eqFunction_1419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[88]] /* vx[89] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1595] /* theta[89] PARAM */)))) * (((data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */)) * ((data->simulationInfo->realParameter[593] /* omega_c[89] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8883(DATA *data, threadData_t *threadData);


/*
equation index: 1421
type: SIMPLE_ASSIGN
vy[89] = cos(theta[89]) * r_init[89] * omega_c[89]
*/
void SpiralGalaxy_eqFunction_1421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[588]] /* vy[89] STATE(1) */) = (cos((data->simulationInfo->realParameter[1595] /* theta[89] PARAM */))) * (((data->simulationInfo->realParameter[1094] /* r_init[89] PARAM */)) * ((data->simulationInfo->realParameter[593] /* omega_c[89] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8882(DATA *data, threadData_t *threadData);


/*
equation index: 1423
type: SIMPLE_ASSIGN
vz[89] = 0.0
*/
void SpiralGalaxy_eqFunction_1423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1423};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1088]] /* vz[89] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8881(DATA *data, threadData_t *threadData);


/*
equation index: 1425
type: SIMPLE_ASSIGN
z[90] = -0.025600000000000005
*/
void SpiralGalaxy_eqFunction_1425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1425};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2589]] /* z[90] STATE(1,vz[90]) */) = -0.025600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8894(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8895(DATA *data, threadData_t *threadData);


/*
equation index: 1428
type: SIMPLE_ASSIGN
y[90] = r_init[90] * sin(theta[90] - 0.0064)
*/
void SpiralGalaxy_eqFunction_1428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1428};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2089]] /* y[90] STATE(1,vy[90]) */) = ((data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */)) * (sin((data->simulationInfo->realParameter[1596] /* theta[90] PARAM */) - 0.0064));
  TRACE_POP
}

/*
equation index: 1429
type: SIMPLE_ASSIGN
x[90] = r_init[90] * cos(theta[90] - 0.0064)
*/
void SpiralGalaxy_eqFunction_1429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1589]] /* x[90] STATE(1,vx[90]) */) = ((data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */)) * (cos((data->simulationInfo->realParameter[1596] /* theta[90] PARAM */) - 0.0064));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8896(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8897(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8900(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8899(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8898(DATA *data, threadData_t *threadData);


/*
equation index: 1435
type: SIMPLE_ASSIGN
vx[90] = (-sin(theta[90])) * r_init[90] * omega_c[90]
*/
void SpiralGalaxy_eqFunction_1435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[89]] /* vx[90] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1596] /* theta[90] PARAM */)))) * (((data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */)) * ((data->simulationInfo->realParameter[594] /* omega_c[90] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8893(DATA *data, threadData_t *threadData);


/*
equation index: 1437
type: SIMPLE_ASSIGN
vy[90] = cos(theta[90]) * r_init[90] * omega_c[90]
*/
void SpiralGalaxy_eqFunction_1437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[589]] /* vy[90] STATE(1) */) = (cos((data->simulationInfo->realParameter[1596] /* theta[90] PARAM */))) * (((data->simulationInfo->realParameter[1095] /* r_init[90] PARAM */)) * ((data->simulationInfo->realParameter[594] /* omega_c[90] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8892(DATA *data, threadData_t *threadData);


/*
equation index: 1439
type: SIMPLE_ASSIGN
vz[90] = 0.0
*/
void SpiralGalaxy_eqFunction_1439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1439};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1089]] /* vz[90] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8891(DATA *data, threadData_t *threadData);


/*
equation index: 1441
type: SIMPLE_ASSIGN
z[91] = -0.02544
*/
void SpiralGalaxy_eqFunction_1441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1441};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2590]] /* z[91] STATE(1,vz[91]) */) = -0.02544;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8904(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8905(DATA *data, threadData_t *threadData);


/*
equation index: 1444
type: SIMPLE_ASSIGN
y[91] = r_init[91] * sin(theta[91] - 0.00636)
*/
void SpiralGalaxy_eqFunction_1444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1444};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2090]] /* y[91] STATE(1,vy[91]) */) = ((data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */)) * (sin((data->simulationInfo->realParameter[1597] /* theta[91] PARAM */) - 0.00636));
  TRACE_POP
}

/*
equation index: 1445
type: SIMPLE_ASSIGN
x[91] = r_init[91] * cos(theta[91] - 0.00636)
*/
void SpiralGalaxy_eqFunction_1445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1590]] /* x[91] STATE(1,vx[91]) */) = ((data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */)) * (cos((data->simulationInfo->realParameter[1597] /* theta[91] PARAM */) - 0.00636));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8906(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8907(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8910(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8909(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8908(DATA *data, threadData_t *threadData);


/*
equation index: 1451
type: SIMPLE_ASSIGN
vx[91] = (-sin(theta[91])) * r_init[91] * omega_c[91]
*/
void SpiralGalaxy_eqFunction_1451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[90]] /* vx[91] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1597] /* theta[91] PARAM */)))) * (((data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */)) * ((data->simulationInfo->realParameter[595] /* omega_c[91] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8903(DATA *data, threadData_t *threadData);


/*
equation index: 1453
type: SIMPLE_ASSIGN
vy[91] = cos(theta[91]) * r_init[91] * omega_c[91]
*/
void SpiralGalaxy_eqFunction_1453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[590]] /* vy[91] STATE(1) */) = (cos((data->simulationInfo->realParameter[1597] /* theta[91] PARAM */))) * (((data->simulationInfo->realParameter[1096] /* r_init[91] PARAM */)) * ((data->simulationInfo->realParameter[595] /* omega_c[91] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8902(DATA *data, threadData_t *threadData);


/*
equation index: 1455
type: SIMPLE_ASSIGN
vz[91] = 0.0
*/
void SpiralGalaxy_eqFunction_1455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1455};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1090]] /* vz[91] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8901(DATA *data, threadData_t *threadData);


/*
equation index: 1457
type: SIMPLE_ASSIGN
z[92] = -0.02528
*/
void SpiralGalaxy_eqFunction_1457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1457};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2591]] /* z[92] STATE(1,vz[92]) */) = -0.02528;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8914(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8915(DATA *data, threadData_t *threadData);


/*
equation index: 1460
type: SIMPLE_ASSIGN
y[92] = r_init[92] * sin(theta[92] - 0.00632)
*/
void SpiralGalaxy_eqFunction_1460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1460};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2091]] /* y[92] STATE(1,vy[92]) */) = ((data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */)) * (sin((data->simulationInfo->realParameter[1598] /* theta[92] PARAM */) - 0.00632));
  TRACE_POP
}

/*
equation index: 1461
type: SIMPLE_ASSIGN
x[92] = r_init[92] * cos(theta[92] - 0.00632)
*/
void SpiralGalaxy_eqFunction_1461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1591]] /* x[92] STATE(1,vx[92]) */) = ((data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */)) * (cos((data->simulationInfo->realParameter[1598] /* theta[92] PARAM */) - 0.00632));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8916(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8917(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8920(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8919(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8918(DATA *data, threadData_t *threadData);


/*
equation index: 1467
type: SIMPLE_ASSIGN
vx[92] = (-sin(theta[92])) * r_init[92] * omega_c[92]
*/
void SpiralGalaxy_eqFunction_1467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[91]] /* vx[92] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1598] /* theta[92] PARAM */)))) * (((data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */)) * ((data->simulationInfo->realParameter[596] /* omega_c[92] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8913(DATA *data, threadData_t *threadData);


/*
equation index: 1469
type: SIMPLE_ASSIGN
vy[92] = cos(theta[92]) * r_init[92] * omega_c[92]
*/
void SpiralGalaxy_eqFunction_1469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[591]] /* vy[92] STATE(1) */) = (cos((data->simulationInfo->realParameter[1598] /* theta[92] PARAM */))) * (((data->simulationInfo->realParameter[1097] /* r_init[92] PARAM */)) * ((data->simulationInfo->realParameter[596] /* omega_c[92] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8912(DATA *data, threadData_t *threadData);


/*
equation index: 1471
type: SIMPLE_ASSIGN
vz[92] = 0.0
*/
void SpiralGalaxy_eqFunction_1471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1471};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1091]] /* vz[92] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8911(DATA *data, threadData_t *threadData);


/*
equation index: 1473
type: SIMPLE_ASSIGN
z[93] = -0.02512
*/
void SpiralGalaxy_eqFunction_1473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1473};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2592]] /* z[93] STATE(1,vz[93]) */) = -0.02512;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8924(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8925(DATA *data, threadData_t *threadData);


/*
equation index: 1476
type: SIMPLE_ASSIGN
y[93] = r_init[93] * sin(theta[93] - 0.00628)
*/
void SpiralGalaxy_eqFunction_1476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1476};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2092]] /* y[93] STATE(1,vy[93]) */) = ((data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */)) * (sin((data->simulationInfo->realParameter[1599] /* theta[93] PARAM */) - 0.00628));
  TRACE_POP
}

/*
equation index: 1477
type: SIMPLE_ASSIGN
x[93] = r_init[93] * cos(theta[93] - 0.00628)
*/
void SpiralGalaxy_eqFunction_1477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1592]] /* x[93] STATE(1,vx[93]) */) = ((data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */)) * (cos((data->simulationInfo->realParameter[1599] /* theta[93] PARAM */) - 0.00628));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8926(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8927(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8930(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8929(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8928(DATA *data, threadData_t *threadData);


/*
equation index: 1483
type: SIMPLE_ASSIGN
vx[93] = (-sin(theta[93])) * r_init[93] * omega_c[93]
*/
void SpiralGalaxy_eqFunction_1483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[92]] /* vx[93] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1599] /* theta[93] PARAM */)))) * (((data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */)) * ((data->simulationInfo->realParameter[597] /* omega_c[93] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8923(DATA *data, threadData_t *threadData);


/*
equation index: 1485
type: SIMPLE_ASSIGN
vy[93] = cos(theta[93]) * r_init[93] * omega_c[93]
*/
void SpiralGalaxy_eqFunction_1485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[592]] /* vy[93] STATE(1) */) = (cos((data->simulationInfo->realParameter[1599] /* theta[93] PARAM */))) * (((data->simulationInfo->realParameter[1098] /* r_init[93] PARAM */)) * ((data->simulationInfo->realParameter[597] /* omega_c[93] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8922(DATA *data, threadData_t *threadData);


/*
equation index: 1487
type: SIMPLE_ASSIGN
vz[93] = 0.0
*/
void SpiralGalaxy_eqFunction_1487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1487};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1092]] /* vz[93] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8921(DATA *data, threadData_t *threadData);


/*
equation index: 1489
type: SIMPLE_ASSIGN
z[94] = -0.024960000000000003
*/
void SpiralGalaxy_eqFunction_1489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1489};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2593]] /* z[94] STATE(1,vz[94]) */) = -0.024960000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8934(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8935(DATA *data, threadData_t *threadData);


/*
equation index: 1492
type: SIMPLE_ASSIGN
y[94] = r_init[94] * sin(theta[94] - 0.00624)
*/
void SpiralGalaxy_eqFunction_1492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1492};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2093]] /* y[94] STATE(1,vy[94]) */) = ((data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */)) * (sin((data->simulationInfo->realParameter[1600] /* theta[94] PARAM */) - 0.00624));
  TRACE_POP
}

/*
equation index: 1493
type: SIMPLE_ASSIGN
x[94] = r_init[94] * cos(theta[94] - 0.00624)
*/
void SpiralGalaxy_eqFunction_1493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1593]] /* x[94] STATE(1,vx[94]) */) = ((data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */)) * (cos((data->simulationInfo->realParameter[1600] /* theta[94] PARAM */) - 0.00624));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8936(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8937(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8940(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8939(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8938(DATA *data, threadData_t *threadData);


/*
equation index: 1499
type: SIMPLE_ASSIGN
vx[94] = (-sin(theta[94])) * r_init[94] * omega_c[94]
*/
void SpiralGalaxy_eqFunction_1499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1499};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[93]] /* vx[94] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1600] /* theta[94] PARAM */)))) * (((data->simulationInfo->realParameter[1099] /* r_init[94] PARAM */)) * ((data->simulationInfo->realParameter[598] /* omega_c[94] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8933(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_2(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_8629(data, threadData);
  SpiralGalaxy_eqFunction_8628(data, threadData);
  SpiralGalaxy_eqFunction_1003(data, threadData);
  SpiralGalaxy_eqFunction_8623(data, threadData);
  SpiralGalaxy_eqFunction_1005(data, threadData);
  SpiralGalaxy_eqFunction_8622(data, threadData);
  SpiralGalaxy_eqFunction_1007(data, threadData);
  SpiralGalaxy_eqFunction_8621(data, threadData);
  SpiralGalaxy_eqFunction_1009(data, threadData);
  SpiralGalaxy_eqFunction_8634(data, threadData);
  SpiralGalaxy_eqFunction_8635(data, threadData);
  SpiralGalaxy_eqFunction_1012(data, threadData);
  SpiralGalaxy_eqFunction_1013(data, threadData);
  SpiralGalaxy_eqFunction_8636(data, threadData);
  SpiralGalaxy_eqFunction_8637(data, threadData);
  SpiralGalaxy_eqFunction_8640(data, threadData);
  SpiralGalaxy_eqFunction_8639(data, threadData);
  SpiralGalaxy_eqFunction_8638(data, threadData);
  SpiralGalaxy_eqFunction_1019(data, threadData);
  SpiralGalaxy_eqFunction_8633(data, threadData);
  SpiralGalaxy_eqFunction_1021(data, threadData);
  SpiralGalaxy_eqFunction_8632(data, threadData);
  SpiralGalaxy_eqFunction_1023(data, threadData);
  SpiralGalaxy_eqFunction_8631(data, threadData);
  SpiralGalaxy_eqFunction_1025(data, threadData);
  SpiralGalaxy_eqFunction_8644(data, threadData);
  SpiralGalaxy_eqFunction_8645(data, threadData);
  SpiralGalaxy_eqFunction_1028(data, threadData);
  SpiralGalaxy_eqFunction_1029(data, threadData);
  SpiralGalaxy_eqFunction_8646(data, threadData);
  SpiralGalaxy_eqFunction_8647(data, threadData);
  SpiralGalaxy_eqFunction_8650(data, threadData);
  SpiralGalaxy_eqFunction_8649(data, threadData);
  SpiralGalaxy_eqFunction_8648(data, threadData);
  SpiralGalaxy_eqFunction_1035(data, threadData);
  SpiralGalaxy_eqFunction_8643(data, threadData);
  SpiralGalaxy_eqFunction_1037(data, threadData);
  SpiralGalaxy_eqFunction_8642(data, threadData);
  SpiralGalaxy_eqFunction_1039(data, threadData);
  SpiralGalaxy_eqFunction_8641(data, threadData);
  SpiralGalaxy_eqFunction_1041(data, threadData);
  SpiralGalaxy_eqFunction_8654(data, threadData);
  SpiralGalaxy_eqFunction_8655(data, threadData);
  SpiralGalaxy_eqFunction_1044(data, threadData);
  SpiralGalaxy_eqFunction_1045(data, threadData);
  SpiralGalaxy_eqFunction_8656(data, threadData);
  SpiralGalaxy_eqFunction_8657(data, threadData);
  SpiralGalaxy_eqFunction_8660(data, threadData);
  SpiralGalaxy_eqFunction_8659(data, threadData);
  SpiralGalaxy_eqFunction_8658(data, threadData);
  SpiralGalaxy_eqFunction_1051(data, threadData);
  SpiralGalaxy_eqFunction_8653(data, threadData);
  SpiralGalaxy_eqFunction_1053(data, threadData);
  SpiralGalaxy_eqFunction_8652(data, threadData);
  SpiralGalaxy_eqFunction_1055(data, threadData);
  SpiralGalaxy_eqFunction_8651(data, threadData);
  SpiralGalaxy_eqFunction_1057(data, threadData);
  SpiralGalaxy_eqFunction_8664(data, threadData);
  SpiralGalaxy_eqFunction_8665(data, threadData);
  SpiralGalaxy_eqFunction_1060(data, threadData);
  SpiralGalaxy_eqFunction_1061(data, threadData);
  SpiralGalaxy_eqFunction_8666(data, threadData);
  SpiralGalaxy_eqFunction_8667(data, threadData);
  SpiralGalaxy_eqFunction_8670(data, threadData);
  SpiralGalaxy_eqFunction_8669(data, threadData);
  SpiralGalaxy_eqFunction_8668(data, threadData);
  SpiralGalaxy_eqFunction_1067(data, threadData);
  SpiralGalaxy_eqFunction_8663(data, threadData);
  SpiralGalaxy_eqFunction_1069(data, threadData);
  SpiralGalaxy_eqFunction_8662(data, threadData);
  SpiralGalaxy_eqFunction_1071(data, threadData);
  SpiralGalaxy_eqFunction_8661(data, threadData);
  SpiralGalaxy_eqFunction_1073(data, threadData);
  SpiralGalaxy_eqFunction_8674(data, threadData);
  SpiralGalaxy_eqFunction_8675(data, threadData);
  SpiralGalaxy_eqFunction_1076(data, threadData);
  SpiralGalaxy_eqFunction_1077(data, threadData);
  SpiralGalaxy_eqFunction_8676(data, threadData);
  SpiralGalaxy_eqFunction_8677(data, threadData);
  SpiralGalaxy_eqFunction_8680(data, threadData);
  SpiralGalaxy_eqFunction_8679(data, threadData);
  SpiralGalaxy_eqFunction_8678(data, threadData);
  SpiralGalaxy_eqFunction_1083(data, threadData);
  SpiralGalaxy_eqFunction_8673(data, threadData);
  SpiralGalaxy_eqFunction_1085(data, threadData);
  SpiralGalaxy_eqFunction_8672(data, threadData);
  SpiralGalaxy_eqFunction_1087(data, threadData);
  SpiralGalaxy_eqFunction_8671(data, threadData);
  SpiralGalaxy_eqFunction_1089(data, threadData);
  SpiralGalaxy_eqFunction_8684(data, threadData);
  SpiralGalaxy_eqFunction_8685(data, threadData);
  SpiralGalaxy_eqFunction_1092(data, threadData);
  SpiralGalaxy_eqFunction_1093(data, threadData);
  SpiralGalaxy_eqFunction_8686(data, threadData);
  SpiralGalaxy_eqFunction_8687(data, threadData);
  SpiralGalaxy_eqFunction_8690(data, threadData);
  SpiralGalaxy_eqFunction_8689(data, threadData);
  SpiralGalaxy_eqFunction_8688(data, threadData);
  SpiralGalaxy_eqFunction_1099(data, threadData);
  SpiralGalaxy_eqFunction_8683(data, threadData);
  SpiralGalaxy_eqFunction_1101(data, threadData);
  SpiralGalaxy_eqFunction_8682(data, threadData);
  SpiralGalaxy_eqFunction_1103(data, threadData);
  SpiralGalaxy_eqFunction_8681(data, threadData);
  SpiralGalaxy_eqFunction_1105(data, threadData);
  SpiralGalaxy_eqFunction_8694(data, threadData);
  SpiralGalaxy_eqFunction_8695(data, threadData);
  SpiralGalaxy_eqFunction_1108(data, threadData);
  SpiralGalaxy_eqFunction_1109(data, threadData);
  SpiralGalaxy_eqFunction_8696(data, threadData);
  SpiralGalaxy_eqFunction_8697(data, threadData);
  SpiralGalaxy_eqFunction_8700(data, threadData);
  SpiralGalaxy_eqFunction_8699(data, threadData);
  SpiralGalaxy_eqFunction_8698(data, threadData);
  SpiralGalaxy_eqFunction_1115(data, threadData);
  SpiralGalaxy_eqFunction_8693(data, threadData);
  SpiralGalaxy_eqFunction_1117(data, threadData);
  SpiralGalaxy_eqFunction_8692(data, threadData);
  SpiralGalaxy_eqFunction_1119(data, threadData);
  SpiralGalaxy_eqFunction_8691(data, threadData);
  SpiralGalaxy_eqFunction_1121(data, threadData);
  SpiralGalaxy_eqFunction_8704(data, threadData);
  SpiralGalaxy_eqFunction_8705(data, threadData);
  SpiralGalaxy_eqFunction_1124(data, threadData);
  SpiralGalaxy_eqFunction_1125(data, threadData);
  SpiralGalaxy_eqFunction_8706(data, threadData);
  SpiralGalaxy_eqFunction_8707(data, threadData);
  SpiralGalaxy_eqFunction_8710(data, threadData);
  SpiralGalaxy_eqFunction_8709(data, threadData);
  SpiralGalaxy_eqFunction_8708(data, threadData);
  SpiralGalaxy_eqFunction_1131(data, threadData);
  SpiralGalaxy_eqFunction_8703(data, threadData);
  SpiralGalaxy_eqFunction_1133(data, threadData);
  SpiralGalaxy_eqFunction_8702(data, threadData);
  SpiralGalaxy_eqFunction_1135(data, threadData);
  SpiralGalaxy_eqFunction_8701(data, threadData);
  SpiralGalaxy_eqFunction_1137(data, threadData);
  SpiralGalaxy_eqFunction_8714(data, threadData);
  SpiralGalaxy_eqFunction_8715(data, threadData);
  SpiralGalaxy_eqFunction_1140(data, threadData);
  SpiralGalaxy_eqFunction_1141(data, threadData);
  SpiralGalaxy_eqFunction_8716(data, threadData);
  SpiralGalaxy_eqFunction_8717(data, threadData);
  SpiralGalaxy_eqFunction_8720(data, threadData);
  SpiralGalaxy_eqFunction_8719(data, threadData);
  SpiralGalaxy_eqFunction_8718(data, threadData);
  SpiralGalaxy_eqFunction_1147(data, threadData);
  SpiralGalaxy_eqFunction_8713(data, threadData);
  SpiralGalaxy_eqFunction_1149(data, threadData);
  SpiralGalaxy_eqFunction_8712(data, threadData);
  SpiralGalaxy_eqFunction_1151(data, threadData);
  SpiralGalaxy_eqFunction_8711(data, threadData);
  SpiralGalaxy_eqFunction_1153(data, threadData);
  SpiralGalaxy_eqFunction_8724(data, threadData);
  SpiralGalaxy_eqFunction_8725(data, threadData);
  SpiralGalaxy_eqFunction_1156(data, threadData);
  SpiralGalaxy_eqFunction_1157(data, threadData);
  SpiralGalaxy_eqFunction_8726(data, threadData);
  SpiralGalaxy_eqFunction_8727(data, threadData);
  SpiralGalaxy_eqFunction_8730(data, threadData);
  SpiralGalaxy_eqFunction_8729(data, threadData);
  SpiralGalaxy_eqFunction_8728(data, threadData);
  SpiralGalaxy_eqFunction_1163(data, threadData);
  SpiralGalaxy_eqFunction_8723(data, threadData);
  SpiralGalaxy_eqFunction_1165(data, threadData);
  SpiralGalaxy_eqFunction_8722(data, threadData);
  SpiralGalaxy_eqFunction_1167(data, threadData);
  SpiralGalaxy_eqFunction_8721(data, threadData);
  SpiralGalaxy_eqFunction_1169(data, threadData);
  SpiralGalaxy_eqFunction_8734(data, threadData);
  SpiralGalaxy_eqFunction_8735(data, threadData);
  SpiralGalaxy_eqFunction_1172(data, threadData);
  SpiralGalaxy_eqFunction_1173(data, threadData);
  SpiralGalaxy_eqFunction_8736(data, threadData);
  SpiralGalaxy_eqFunction_8737(data, threadData);
  SpiralGalaxy_eqFunction_8740(data, threadData);
  SpiralGalaxy_eqFunction_8739(data, threadData);
  SpiralGalaxy_eqFunction_8738(data, threadData);
  SpiralGalaxy_eqFunction_1179(data, threadData);
  SpiralGalaxy_eqFunction_8733(data, threadData);
  SpiralGalaxy_eqFunction_1181(data, threadData);
  SpiralGalaxy_eqFunction_8732(data, threadData);
  SpiralGalaxy_eqFunction_1183(data, threadData);
  SpiralGalaxy_eqFunction_8731(data, threadData);
  SpiralGalaxy_eqFunction_1185(data, threadData);
  SpiralGalaxy_eqFunction_8744(data, threadData);
  SpiralGalaxy_eqFunction_8745(data, threadData);
  SpiralGalaxy_eqFunction_1188(data, threadData);
  SpiralGalaxy_eqFunction_1189(data, threadData);
  SpiralGalaxy_eqFunction_8746(data, threadData);
  SpiralGalaxy_eqFunction_8747(data, threadData);
  SpiralGalaxy_eqFunction_8750(data, threadData);
  SpiralGalaxy_eqFunction_8749(data, threadData);
  SpiralGalaxy_eqFunction_8748(data, threadData);
  SpiralGalaxy_eqFunction_1195(data, threadData);
  SpiralGalaxy_eqFunction_8743(data, threadData);
  SpiralGalaxy_eqFunction_1197(data, threadData);
  SpiralGalaxy_eqFunction_8742(data, threadData);
  SpiralGalaxy_eqFunction_1199(data, threadData);
  SpiralGalaxy_eqFunction_8741(data, threadData);
  SpiralGalaxy_eqFunction_1201(data, threadData);
  SpiralGalaxy_eqFunction_8754(data, threadData);
  SpiralGalaxy_eqFunction_8755(data, threadData);
  SpiralGalaxy_eqFunction_1204(data, threadData);
  SpiralGalaxy_eqFunction_1205(data, threadData);
  SpiralGalaxy_eqFunction_8756(data, threadData);
  SpiralGalaxy_eqFunction_8757(data, threadData);
  SpiralGalaxy_eqFunction_8760(data, threadData);
  SpiralGalaxy_eqFunction_8759(data, threadData);
  SpiralGalaxy_eqFunction_8758(data, threadData);
  SpiralGalaxy_eqFunction_1211(data, threadData);
  SpiralGalaxy_eqFunction_8753(data, threadData);
  SpiralGalaxy_eqFunction_1213(data, threadData);
  SpiralGalaxy_eqFunction_8752(data, threadData);
  SpiralGalaxy_eqFunction_1215(data, threadData);
  SpiralGalaxy_eqFunction_8751(data, threadData);
  SpiralGalaxy_eqFunction_1217(data, threadData);
  SpiralGalaxy_eqFunction_8764(data, threadData);
  SpiralGalaxy_eqFunction_8765(data, threadData);
  SpiralGalaxy_eqFunction_1220(data, threadData);
  SpiralGalaxy_eqFunction_1221(data, threadData);
  SpiralGalaxy_eqFunction_8766(data, threadData);
  SpiralGalaxy_eqFunction_8767(data, threadData);
  SpiralGalaxy_eqFunction_8770(data, threadData);
  SpiralGalaxy_eqFunction_8769(data, threadData);
  SpiralGalaxy_eqFunction_8768(data, threadData);
  SpiralGalaxy_eqFunction_1227(data, threadData);
  SpiralGalaxy_eqFunction_8763(data, threadData);
  SpiralGalaxy_eqFunction_1229(data, threadData);
  SpiralGalaxy_eqFunction_8762(data, threadData);
  SpiralGalaxy_eqFunction_1231(data, threadData);
  SpiralGalaxy_eqFunction_8761(data, threadData);
  SpiralGalaxy_eqFunction_1233(data, threadData);
  SpiralGalaxy_eqFunction_8774(data, threadData);
  SpiralGalaxy_eqFunction_8775(data, threadData);
  SpiralGalaxy_eqFunction_1236(data, threadData);
  SpiralGalaxy_eqFunction_1237(data, threadData);
  SpiralGalaxy_eqFunction_8776(data, threadData);
  SpiralGalaxy_eqFunction_8777(data, threadData);
  SpiralGalaxy_eqFunction_8780(data, threadData);
  SpiralGalaxy_eqFunction_8779(data, threadData);
  SpiralGalaxy_eqFunction_8778(data, threadData);
  SpiralGalaxy_eqFunction_1243(data, threadData);
  SpiralGalaxy_eqFunction_8773(data, threadData);
  SpiralGalaxy_eqFunction_1245(data, threadData);
  SpiralGalaxy_eqFunction_8772(data, threadData);
  SpiralGalaxy_eqFunction_1247(data, threadData);
  SpiralGalaxy_eqFunction_8771(data, threadData);
  SpiralGalaxy_eqFunction_1249(data, threadData);
  SpiralGalaxy_eqFunction_8784(data, threadData);
  SpiralGalaxy_eqFunction_8785(data, threadData);
  SpiralGalaxy_eqFunction_1252(data, threadData);
  SpiralGalaxy_eqFunction_1253(data, threadData);
  SpiralGalaxy_eqFunction_8786(data, threadData);
  SpiralGalaxy_eqFunction_8787(data, threadData);
  SpiralGalaxy_eqFunction_8790(data, threadData);
  SpiralGalaxy_eqFunction_8789(data, threadData);
  SpiralGalaxy_eqFunction_8788(data, threadData);
  SpiralGalaxy_eqFunction_1259(data, threadData);
  SpiralGalaxy_eqFunction_8783(data, threadData);
  SpiralGalaxy_eqFunction_1261(data, threadData);
  SpiralGalaxy_eqFunction_8782(data, threadData);
  SpiralGalaxy_eqFunction_1263(data, threadData);
  SpiralGalaxy_eqFunction_8781(data, threadData);
  SpiralGalaxy_eqFunction_1265(data, threadData);
  SpiralGalaxy_eqFunction_8794(data, threadData);
  SpiralGalaxy_eqFunction_8795(data, threadData);
  SpiralGalaxy_eqFunction_1268(data, threadData);
  SpiralGalaxy_eqFunction_1269(data, threadData);
  SpiralGalaxy_eqFunction_8796(data, threadData);
  SpiralGalaxy_eqFunction_8797(data, threadData);
  SpiralGalaxy_eqFunction_8800(data, threadData);
  SpiralGalaxy_eqFunction_8799(data, threadData);
  SpiralGalaxy_eqFunction_8798(data, threadData);
  SpiralGalaxy_eqFunction_1275(data, threadData);
  SpiralGalaxy_eqFunction_8793(data, threadData);
  SpiralGalaxy_eqFunction_1277(data, threadData);
  SpiralGalaxy_eqFunction_8792(data, threadData);
  SpiralGalaxy_eqFunction_1279(data, threadData);
  SpiralGalaxy_eqFunction_8791(data, threadData);
  SpiralGalaxy_eqFunction_1281(data, threadData);
  SpiralGalaxy_eqFunction_8804(data, threadData);
  SpiralGalaxy_eqFunction_8805(data, threadData);
  SpiralGalaxy_eqFunction_1284(data, threadData);
  SpiralGalaxy_eqFunction_1285(data, threadData);
  SpiralGalaxy_eqFunction_8806(data, threadData);
  SpiralGalaxy_eqFunction_8807(data, threadData);
  SpiralGalaxy_eqFunction_8810(data, threadData);
  SpiralGalaxy_eqFunction_8809(data, threadData);
  SpiralGalaxy_eqFunction_8808(data, threadData);
  SpiralGalaxy_eqFunction_1291(data, threadData);
  SpiralGalaxy_eqFunction_8803(data, threadData);
  SpiralGalaxy_eqFunction_1293(data, threadData);
  SpiralGalaxy_eqFunction_8802(data, threadData);
  SpiralGalaxy_eqFunction_1295(data, threadData);
  SpiralGalaxy_eqFunction_8801(data, threadData);
  SpiralGalaxy_eqFunction_1297(data, threadData);
  SpiralGalaxy_eqFunction_8814(data, threadData);
  SpiralGalaxy_eqFunction_8815(data, threadData);
  SpiralGalaxy_eqFunction_1300(data, threadData);
  SpiralGalaxy_eqFunction_1301(data, threadData);
  SpiralGalaxy_eqFunction_8816(data, threadData);
  SpiralGalaxy_eqFunction_8817(data, threadData);
  SpiralGalaxy_eqFunction_8820(data, threadData);
  SpiralGalaxy_eqFunction_8819(data, threadData);
  SpiralGalaxy_eqFunction_8818(data, threadData);
  SpiralGalaxy_eqFunction_1307(data, threadData);
  SpiralGalaxy_eqFunction_8813(data, threadData);
  SpiralGalaxy_eqFunction_1309(data, threadData);
  SpiralGalaxy_eqFunction_8812(data, threadData);
  SpiralGalaxy_eqFunction_1311(data, threadData);
  SpiralGalaxy_eqFunction_8811(data, threadData);
  SpiralGalaxy_eqFunction_1313(data, threadData);
  SpiralGalaxy_eqFunction_8824(data, threadData);
  SpiralGalaxy_eqFunction_8825(data, threadData);
  SpiralGalaxy_eqFunction_1316(data, threadData);
  SpiralGalaxy_eqFunction_1317(data, threadData);
  SpiralGalaxy_eqFunction_8826(data, threadData);
  SpiralGalaxy_eqFunction_8827(data, threadData);
  SpiralGalaxy_eqFunction_8830(data, threadData);
  SpiralGalaxy_eqFunction_8829(data, threadData);
  SpiralGalaxy_eqFunction_8828(data, threadData);
  SpiralGalaxy_eqFunction_1323(data, threadData);
  SpiralGalaxy_eqFunction_8823(data, threadData);
  SpiralGalaxy_eqFunction_1325(data, threadData);
  SpiralGalaxy_eqFunction_8822(data, threadData);
  SpiralGalaxy_eqFunction_1327(data, threadData);
  SpiralGalaxy_eqFunction_8821(data, threadData);
  SpiralGalaxy_eqFunction_1329(data, threadData);
  SpiralGalaxy_eqFunction_8834(data, threadData);
  SpiralGalaxy_eqFunction_8835(data, threadData);
  SpiralGalaxy_eqFunction_1332(data, threadData);
  SpiralGalaxy_eqFunction_1333(data, threadData);
  SpiralGalaxy_eqFunction_8836(data, threadData);
  SpiralGalaxy_eqFunction_8837(data, threadData);
  SpiralGalaxy_eqFunction_8840(data, threadData);
  SpiralGalaxy_eqFunction_8839(data, threadData);
  SpiralGalaxy_eqFunction_8838(data, threadData);
  SpiralGalaxy_eqFunction_1339(data, threadData);
  SpiralGalaxy_eqFunction_8833(data, threadData);
  SpiralGalaxy_eqFunction_1341(data, threadData);
  SpiralGalaxy_eqFunction_8832(data, threadData);
  SpiralGalaxy_eqFunction_1343(data, threadData);
  SpiralGalaxy_eqFunction_8831(data, threadData);
  SpiralGalaxy_eqFunction_1345(data, threadData);
  SpiralGalaxy_eqFunction_8844(data, threadData);
  SpiralGalaxy_eqFunction_8845(data, threadData);
  SpiralGalaxy_eqFunction_1348(data, threadData);
  SpiralGalaxy_eqFunction_1349(data, threadData);
  SpiralGalaxy_eqFunction_8846(data, threadData);
  SpiralGalaxy_eqFunction_8847(data, threadData);
  SpiralGalaxy_eqFunction_8850(data, threadData);
  SpiralGalaxy_eqFunction_8849(data, threadData);
  SpiralGalaxy_eqFunction_8848(data, threadData);
  SpiralGalaxy_eqFunction_1355(data, threadData);
  SpiralGalaxy_eqFunction_8843(data, threadData);
  SpiralGalaxy_eqFunction_1357(data, threadData);
  SpiralGalaxy_eqFunction_8842(data, threadData);
  SpiralGalaxy_eqFunction_1359(data, threadData);
  SpiralGalaxy_eqFunction_8841(data, threadData);
  SpiralGalaxy_eqFunction_1361(data, threadData);
  SpiralGalaxy_eqFunction_8854(data, threadData);
  SpiralGalaxy_eqFunction_8855(data, threadData);
  SpiralGalaxy_eqFunction_1364(data, threadData);
  SpiralGalaxy_eqFunction_1365(data, threadData);
  SpiralGalaxy_eqFunction_8856(data, threadData);
  SpiralGalaxy_eqFunction_8857(data, threadData);
  SpiralGalaxy_eqFunction_8860(data, threadData);
  SpiralGalaxy_eqFunction_8859(data, threadData);
  SpiralGalaxy_eqFunction_8858(data, threadData);
  SpiralGalaxy_eqFunction_1371(data, threadData);
  SpiralGalaxy_eqFunction_8853(data, threadData);
  SpiralGalaxy_eqFunction_1373(data, threadData);
  SpiralGalaxy_eqFunction_8852(data, threadData);
  SpiralGalaxy_eqFunction_1375(data, threadData);
  SpiralGalaxy_eqFunction_8851(data, threadData);
  SpiralGalaxy_eqFunction_1377(data, threadData);
  SpiralGalaxy_eqFunction_8864(data, threadData);
  SpiralGalaxy_eqFunction_8865(data, threadData);
  SpiralGalaxy_eqFunction_1380(data, threadData);
  SpiralGalaxy_eqFunction_1381(data, threadData);
  SpiralGalaxy_eqFunction_8866(data, threadData);
  SpiralGalaxy_eqFunction_8867(data, threadData);
  SpiralGalaxy_eqFunction_8870(data, threadData);
  SpiralGalaxy_eqFunction_8869(data, threadData);
  SpiralGalaxy_eqFunction_8868(data, threadData);
  SpiralGalaxy_eqFunction_1387(data, threadData);
  SpiralGalaxy_eqFunction_8863(data, threadData);
  SpiralGalaxy_eqFunction_1389(data, threadData);
  SpiralGalaxy_eqFunction_8862(data, threadData);
  SpiralGalaxy_eqFunction_1391(data, threadData);
  SpiralGalaxy_eqFunction_8861(data, threadData);
  SpiralGalaxy_eqFunction_1393(data, threadData);
  SpiralGalaxy_eqFunction_8874(data, threadData);
  SpiralGalaxy_eqFunction_8875(data, threadData);
  SpiralGalaxy_eqFunction_1396(data, threadData);
  SpiralGalaxy_eqFunction_1397(data, threadData);
  SpiralGalaxy_eqFunction_8876(data, threadData);
  SpiralGalaxy_eqFunction_8877(data, threadData);
  SpiralGalaxy_eqFunction_8880(data, threadData);
  SpiralGalaxy_eqFunction_8879(data, threadData);
  SpiralGalaxy_eqFunction_8878(data, threadData);
  SpiralGalaxy_eqFunction_1403(data, threadData);
  SpiralGalaxy_eqFunction_8873(data, threadData);
  SpiralGalaxy_eqFunction_1405(data, threadData);
  SpiralGalaxy_eqFunction_8872(data, threadData);
  SpiralGalaxy_eqFunction_1407(data, threadData);
  SpiralGalaxy_eqFunction_8871(data, threadData);
  SpiralGalaxy_eqFunction_1409(data, threadData);
  SpiralGalaxy_eqFunction_8884(data, threadData);
  SpiralGalaxy_eqFunction_8885(data, threadData);
  SpiralGalaxy_eqFunction_1412(data, threadData);
  SpiralGalaxy_eqFunction_1413(data, threadData);
  SpiralGalaxy_eqFunction_8886(data, threadData);
  SpiralGalaxy_eqFunction_8887(data, threadData);
  SpiralGalaxy_eqFunction_8890(data, threadData);
  SpiralGalaxy_eqFunction_8889(data, threadData);
  SpiralGalaxy_eqFunction_8888(data, threadData);
  SpiralGalaxy_eqFunction_1419(data, threadData);
  SpiralGalaxy_eqFunction_8883(data, threadData);
  SpiralGalaxy_eqFunction_1421(data, threadData);
  SpiralGalaxy_eqFunction_8882(data, threadData);
  SpiralGalaxy_eqFunction_1423(data, threadData);
  SpiralGalaxy_eqFunction_8881(data, threadData);
  SpiralGalaxy_eqFunction_1425(data, threadData);
  SpiralGalaxy_eqFunction_8894(data, threadData);
  SpiralGalaxy_eqFunction_8895(data, threadData);
  SpiralGalaxy_eqFunction_1428(data, threadData);
  SpiralGalaxy_eqFunction_1429(data, threadData);
  SpiralGalaxy_eqFunction_8896(data, threadData);
  SpiralGalaxy_eqFunction_8897(data, threadData);
  SpiralGalaxy_eqFunction_8900(data, threadData);
  SpiralGalaxy_eqFunction_8899(data, threadData);
  SpiralGalaxy_eqFunction_8898(data, threadData);
  SpiralGalaxy_eqFunction_1435(data, threadData);
  SpiralGalaxy_eqFunction_8893(data, threadData);
  SpiralGalaxy_eqFunction_1437(data, threadData);
  SpiralGalaxy_eqFunction_8892(data, threadData);
  SpiralGalaxy_eqFunction_1439(data, threadData);
  SpiralGalaxy_eqFunction_8891(data, threadData);
  SpiralGalaxy_eqFunction_1441(data, threadData);
  SpiralGalaxy_eqFunction_8904(data, threadData);
  SpiralGalaxy_eqFunction_8905(data, threadData);
  SpiralGalaxy_eqFunction_1444(data, threadData);
  SpiralGalaxy_eqFunction_1445(data, threadData);
  SpiralGalaxy_eqFunction_8906(data, threadData);
  SpiralGalaxy_eqFunction_8907(data, threadData);
  SpiralGalaxy_eqFunction_8910(data, threadData);
  SpiralGalaxy_eqFunction_8909(data, threadData);
  SpiralGalaxy_eqFunction_8908(data, threadData);
  SpiralGalaxy_eqFunction_1451(data, threadData);
  SpiralGalaxy_eqFunction_8903(data, threadData);
  SpiralGalaxy_eqFunction_1453(data, threadData);
  SpiralGalaxy_eqFunction_8902(data, threadData);
  SpiralGalaxy_eqFunction_1455(data, threadData);
  SpiralGalaxy_eqFunction_8901(data, threadData);
  SpiralGalaxy_eqFunction_1457(data, threadData);
  SpiralGalaxy_eqFunction_8914(data, threadData);
  SpiralGalaxy_eqFunction_8915(data, threadData);
  SpiralGalaxy_eqFunction_1460(data, threadData);
  SpiralGalaxy_eqFunction_1461(data, threadData);
  SpiralGalaxy_eqFunction_8916(data, threadData);
  SpiralGalaxy_eqFunction_8917(data, threadData);
  SpiralGalaxy_eqFunction_8920(data, threadData);
  SpiralGalaxy_eqFunction_8919(data, threadData);
  SpiralGalaxy_eqFunction_8918(data, threadData);
  SpiralGalaxy_eqFunction_1467(data, threadData);
  SpiralGalaxy_eqFunction_8913(data, threadData);
  SpiralGalaxy_eqFunction_1469(data, threadData);
  SpiralGalaxy_eqFunction_8912(data, threadData);
  SpiralGalaxy_eqFunction_1471(data, threadData);
  SpiralGalaxy_eqFunction_8911(data, threadData);
  SpiralGalaxy_eqFunction_1473(data, threadData);
  SpiralGalaxy_eqFunction_8924(data, threadData);
  SpiralGalaxy_eqFunction_8925(data, threadData);
  SpiralGalaxy_eqFunction_1476(data, threadData);
  SpiralGalaxy_eqFunction_1477(data, threadData);
  SpiralGalaxy_eqFunction_8926(data, threadData);
  SpiralGalaxy_eqFunction_8927(data, threadData);
  SpiralGalaxy_eqFunction_8930(data, threadData);
  SpiralGalaxy_eqFunction_8929(data, threadData);
  SpiralGalaxy_eqFunction_8928(data, threadData);
  SpiralGalaxy_eqFunction_1483(data, threadData);
  SpiralGalaxy_eqFunction_8923(data, threadData);
  SpiralGalaxy_eqFunction_1485(data, threadData);
  SpiralGalaxy_eqFunction_8922(data, threadData);
  SpiralGalaxy_eqFunction_1487(data, threadData);
  SpiralGalaxy_eqFunction_8921(data, threadData);
  SpiralGalaxy_eqFunction_1489(data, threadData);
  SpiralGalaxy_eqFunction_8934(data, threadData);
  SpiralGalaxy_eqFunction_8935(data, threadData);
  SpiralGalaxy_eqFunction_1492(data, threadData);
  SpiralGalaxy_eqFunction_1493(data, threadData);
  SpiralGalaxy_eqFunction_8936(data, threadData);
  SpiralGalaxy_eqFunction_8937(data, threadData);
  SpiralGalaxy_eqFunction_8940(data, threadData);
  SpiralGalaxy_eqFunction_8939(data, threadData);
  SpiralGalaxy_eqFunction_8938(data, threadData);
  SpiralGalaxy_eqFunction_1499(data, threadData);
  SpiralGalaxy_eqFunction_8933(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif