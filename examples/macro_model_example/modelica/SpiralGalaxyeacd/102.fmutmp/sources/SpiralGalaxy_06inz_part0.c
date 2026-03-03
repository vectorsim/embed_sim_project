#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 1
type: SIMPLE_ASSIGN
z[1] = -0.03984000000000001
*/
void SpiralGalaxy_eqFunction_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2500]] /* z[1] STATE(1,vz[1]) */) = -0.03984000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8004(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8005(DATA *data, threadData_t *threadData);


/*
equation index: 4
type: SIMPLE_ASSIGN
y[1] = r_init[1] * sin(theta[1] - 0.00996)
*/
void SpiralGalaxy_eqFunction_4(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2000]] /* y[1] STATE(1,vy[1]) */) = ((data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */)) * (sin((data->simulationInfo->realParameter[1507] /* theta[1] PARAM */) - 0.00996));
  TRACE_POP
}

/*
equation index: 5
type: SIMPLE_ASSIGN
x[1] = r_init[1] * cos(theta[1] - 0.00996)
*/
void SpiralGalaxy_eqFunction_5(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,5};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1500]] /* x[1] STATE(1,vx[1]) */) = ((data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */)) * (cos((data->simulationInfo->realParameter[1507] /* theta[1] PARAM */) - 0.00996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8006(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8007(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8010(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8009(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8008(DATA *data, threadData_t *threadData);


/*
equation index: 11
type: SIMPLE_ASSIGN
vx[1] = (-sin(theta[1])) * r_init[1] * omega_c[1]
*/
void SpiralGalaxy_eqFunction_11(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* vx[1] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1507] /* theta[1] PARAM */)))) * (((data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */)) * ((data->simulationInfo->realParameter[505] /* omega_c[1] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8003(DATA *data, threadData_t *threadData);


/*
equation index: 13
type: SIMPLE_ASSIGN
vy[1] = cos(theta[1]) * r_init[1] * omega_c[1]
*/
void SpiralGalaxy_eqFunction_13(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[500]] /* vy[1] STATE(1) */) = (cos((data->simulationInfo->realParameter[1507] /* theta[1] PARAM */))) * (((data->simulationInfo->realParameter[1006] /* r_init[1] PARAM */)) * ((data->simulationInfo->realParameter[505] /* omega_c[1] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8002(DATA *data, threadData_t *threadData);


/*
equation index: 15
type: SIMPLE_ASSIGN
vz[1] = 0.0
*/
void SpiralGalaxy_eqFunction_15(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1000]] /* vz[1] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8001(DATA *data, threadData_t *threadData);


/*
equation index: 17
type: SIMPLE_ASSIGN
z[2] = -0.03968000000000001
*/
void SpiralGalaxy_eqFunction_17(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,17};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2501]] /* z[2] STATE(1,vz[2]) */) = -0.03968000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8014(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8015(DATA *data, threadData_t *threadData);


/*
equation index: 20
type: SIMPLE_ASSIGN
y[2] = r_init[2] * sin(theta[2] - 0.00992)
*/
void SpiralGalaxy_eqFunction_20(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,20};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2001]] /* y[2] STATE(1,vy[2]) */) = ((data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */)) * (sin((data->simulationInfo->realParameter[1508] /* theta[2] PARAM */) - 0.00992));
  TRACE_POP
}

/*
equation index: 21
type: SIMPLE_ASSIGN
x[2] = r_init[2] * cos(theta[2] - 0.00992)
*/
void SpiralGalaxy_eqFunction_21(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,21};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1501]] /* x[2] STATE(1,vx[2]) */) = ((data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */)) * (cos((data->simulationInfo->realParameter[1508] /* theta[2] PARAM */) - 0.00992));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8016(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8017(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8020(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8019(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8018(DATA *data, threadData_t *threadData);


/*
equation index: 27
type: SIMPLE_ASSIGN
vx[2] = (-sin(theta[2])) * r_init[2] * omega_c[2]
*/
void SpiralGalaxy_eqFunction_27(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,27};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* vx[2] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1508] /* theta[2] PARAM */)))) * (((data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */)) * ((data->simulationInfo->realParameter[506] /* omega_c[2] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8013(DATA *data, threadData_t *threadData);


/*
equation index: 29
type: SIMPLE_ASSIGN
vy[2] = cos(theta[2]) * r_init[2] * omega_c[2]
*/
void SpiralGalaxy_eqFunction_29(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,29};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[501]] /* vy[2] STATE(1) */) = (cos((data->simulationInfo->realParameter[1508] /* theta[2] PARAM */))) * (((data->simulationInfo->realParameter[1007] /* r_init[2] PARAM */)) * ((data->simulationInfo->realParameter[506] /* omega_c[2] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8012(DATA *data, threadData_t *threadData);


/*
equation index: 31
type: SIMPLE_ASSIGN
vz[2] = 0.0
*/
void SpiralGalaxy_eqFunction_31(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1001]] /* vz[2] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8011(DATA *data, threadData_t *threadData);


/*
equation index: 33
type: SIMPLE_ASSIGN
z[3] = -0.039520000000000007
*/
void SpiralGalaxy_eqFunction_33(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2502]] /* z[3] STATE(1,vz[3]) */) = -0.039520000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8024(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8025(DATA *data, threadData_t *threadData);


/*
equation index: 36
type: SIMPLE_ASSIGN
y[3] = r_init[3] * sin(theta[3] - 0.00988)
*/
void SpiralGalaxy_eqFunction_36(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2002]] /* y[3] STATE(1,vy[3]) */) = ((data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */)) * (sin((data->simulationInfo->realParameter[1509] /* theta[3] PARAM */) - 0.00988));
  TRACE_POP
}

/*
equation index: 37
type: SIMPLE_ASSIGN
x[3] = r_init[3] * cos(theta[3] - 0.00988)
*/
void SpiralGalaxy_eqFunction_37(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1502]] /* x[3] STATE(1,vx[3]) */) = ((data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */)) * (cos((data->simulationInfo->realParameter[1509] /* theta[3] PARAM */) - 0.00988));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8026(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8027(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8030(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8029(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8028(DATA *data, threadData_t *threadData);


/*
equation index: 43
type: SIMPLE_ASSIGN
vx[3] = (-sin(theta[3])) * r_init[3] * omega_c[3]
*/
void SpiralGalaxy_eqFunction_43(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* vx[3] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1509] /* theta[3] PARAM */)))) * (((data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */)) * ((data->simulationInfo->realParameter[507] /* omega_c[3] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8023(DATA *data, threadData_t *threadData);


/*
equation index: 45
type: SIMPLE_ASSIGN
vy[3] = cos(theta[3]) * r_init[3] * omega_c[3]
*/
void SpiralGalaxy_eqFunction_45(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,45};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[502]] /* vy[3] STATE(1) */) = (cos((data->simulationInfo->realParameter[1509] /* theta[3] PARAM */))) * (((data->simulationInfo->realParameter[1008] /* r_init[3] PARAM */)) * ((data->simulationInfo->realParameter[507] /* omega_c[3] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8022(DATA *data, threadData_t *threadData);


/*
equation index: 47
type: SIMPLE_ASSIGN
vz[3] = 0.0
*/
void SpiralGalaxy_eqFunction_47(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1002]] /* vz[3] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8021(DATA *data, threadData_t *threadData);


/*
equation index: 49
type: SIMPLE_ASSIGN
z[4] = -0.039360000000000006
*/
void SpiralGalaxy_eqFunction_49(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2503]] /* z[4] STATE(1,vz[4]) */) = -0.039360000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8034(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8035(DATA *data, threadData_t *threadData);


/*
equation index: 52
type: SIMPLE_ASSIGN
y[4] = r_init[4] * sin(theta[4] - 0.00984)
*/
void SpiralGalaxy_eqFunction_52(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,52};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2003]] /* y[4] STATE(1,vy[4]) */) = ((data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */)) * (sin((data->simulationInfo->realParameter[1510] /* theta[4] PARAM */) - 0.00984));
  TRACE_POP
}

/*
equation index: 53
type: SIMPLE_ASSIGN
x[4] = r_init[4] * cos(theta[4] - 0.00984)
*/
void SpiralGalaxy_eqFunction_53(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,53};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1503]] /* x[4] STATE(1,vx[4]) */) = ((data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */)) * (cos((data->simulationInfo->realParameter[1510] /* theta[4] PARAM */) - 0.00984));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8036(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8037(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8040(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8039(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8038(DATA *data, threadData_t *threadData);


/*
equation index: 59
type: SIMPLE_ASSIGN
vx[4] = (-sin(theta[4])) * r_init[4] * omega_c[4]
*/
void SpiralGalaxy_eqFunction_59(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,59};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* vx[4] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1510] /* theta[4] PARAM */)))) * (((data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */)) * ((data->simulationInfo->realParameter[508] /* omega_c[4] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8033(DATA *data, threadData_t *threadData);


/*
equation index: 61
type: SIMPLE_ASSIGN
vy[4] = cos(theta[4]) * r_init[4] * omega_c[4]
*/
void SpiralGalaxy_eqFunction_61(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,61};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[503]] /* vy[4] STATE(1) */) = (cos((data->simulationInfo->realParameter[1510] /* theta[4] PARAM */))) * (((data->simulationInfo->realParameter[1009] /* r_init[4] PARAM */)) * ((data->simulationInfo->realParameter[508] /* omega_c[4] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8032(DATA *data, threadData_t *threadData);


/*
equation index: 63
type: SIMPLE_ASSIGN
vz[4] = 0.0
*/
void SpiralGalaxy_eqFunction_63(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,63};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1003]] /* vz[4] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8031(DATA *data, threadData_t *threadData);


/*
equation index: 65
type: SIMPLE_ASSIGN
z[5] = -0.039200000000000006
*/
void SpiralGalaxy_eqFunction_65(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,65};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2504]] /* z[5] STATE(1,vz[5]) */) = -0.039200000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8044(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8045(DATA *data, threadData_t *threadData);


/*
equation index: 68
type: SIMPLE_ASSIGN
y[5] = r_init[5] * sin(theta[5] - 0.0098)
*/
void SpiralGalaxy_eqFunction_68(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,68};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2004]] /* y[5] STATE(1,vy[5]) */) = ((data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */)) * (sin((data->simulationInfo->realParameter[1511] /* theta[5] PARAM */) - 0.0098));
  TRACE_POP
}

/*
equation index: 69
type: SIMPLE_ASSIGN
x[5] = r_init[5] * cos(theta[5] - 0.0098)
*/
void SpiralGalaxy_eqFunction_69(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,69};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1504]] /* x[5] STATE(1,vx[5]) */) = ((data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */)) * (cos((data->simulationInfo->realParameter[1511] /* theta[5] PARAM */) - 0.0098));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8046(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8047(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8050(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8049(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8048(DATA *data, threadData_t *threadData);


/*
equation index: 75
type: SIMPLE_ASSIGN
vx[5] = (-sin(theta[5])) * r_init[5] * omega_c[5]
*/
void SpiralGalaxy_eqFunction_75(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,75};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* vx[5] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1511] /* theta[5] PARAM */)))) * (((data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */)) * ((data->simulationInfo->realParameter[509] /* omega_c[5] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8043(DATA *data, threadData_t *threadData);


/*
equation index: 77
type: SIMPLE_ASSIGN
vy[5] = cos(theta[5]) * r_init[5] * omega_c[5]
*/
void SpiralGalaxy_eqFunction_77(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,77};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[504]] /* vy[5] STATE(1) */) = (cos((data->simulationInfo->realParameter[1511] /* theta[5] PARAM */))) * (((data->simulationInfo->realParameter[1010] /* r_init[5] PARAM */)) * ((data->simulationInfo->realParameter[509] /* omega_c[5] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8042(DATA *data, threadData_t *threadData);


/*
equation index: 79
type: SIMPLE_ASSIGN
vz[5] = 0.0
*/
void SpiralGalaxy_eqFunction_79(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,79};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1004]] /* vz[5] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8041(DATA *data, threadData_t *threadData);


/*
equation index: 81
type: SIMPLE_ASSIGN
z[6] = -0.039040000000000005
*/
void SpiralGalaxy_eqFunction_81(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,81};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2505]] /* z[6] STATE(1,vz[6]) */) = -0.039040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8054(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8055(DATA *data, threadData_t *threadData);


/*
equation index: 84
type: SIMPLE_ASSIGN
y[6] = r_init[6] * sin(theta[6] - 0.00976)
*/
void SpiralGalaxy_eqFunction_84(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,84};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2005]] /* y[6] STATE(1,vy[6]) */) = ((data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */)) * (sin((data->simulationInfo->realParameter[1512] /* theta[6] PARAM */) - 0.00976));
  TRACE_POP
}

/*
equation index: 85
type: SIMPLE_ASSIGN
x[6] = r_init[6] * cos(theta[6] - 0.00976)
*/
void SpiralGalaxy_eqFunction_85(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,85};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1505]] /* x[6] STATE(1,vx[6]) */) = ((data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */)) * (cos((data->simulationInfo->realParameter[1512] /* theta[6] PARAM */) - 0.00976));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8056(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8057(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8060(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8059(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8058(DATA *data, threadData_t *threadData);


/*
equation index: 91
type: SIMPLE_ASSIGN
vx[6] = (-sin(theta[6])) * r_init[6] * omega_c[6]
*/
void SpiralGalaxy_eqFunction_91(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,91};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* vx[6] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1512] /* theta[6] PARAM */)))) * (((data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */)) * ((data->simulationInfo->realParameter[510] /* omega_c[6] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8053(DATA *data, threadData_t *threadData);


/*
equation index: 93
type: SIMPLE_ASSIGN
vy[6] = cos(theta[6]) * r_init[6] * omega_c[6]
*/
void SpiralGalaxy_eqFunction_93(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,93};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[505]] /* vy[6] STATE(1) */) = (cos((data->simulationInfo->realParameter[1512] /* theta[6] PARAM */))) * (((data->simulationInfo->realParameter[1011] /* r_init[6] PARAM */)) * ((data->simulationInfo->realParameter[510] /* omega_c[6] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8052(DATA *data, threadData_t *threadData);


/*
equation index: 95
type: SIMPLE_ASSIGN
vz[6] = 0.0
*/
void SpiralGalaxy_eqFunction_95(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,95};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1005]] /* vz[6] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8051(DATA *data, threadData_t *threadData);


/*
equation index: 97
type: SIMPLE_ASSIGN
z[7] = -0.03888
*/
void SpiralGalaxy_eqFunction_97(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,97};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2506]] /* z[7] STATE(1,vz[7]) */) = -0.03888;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8064(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8065(DATA *data, threadData_t *threadData);


/*
equation index: 100
type: SIMPLE_ASSIGN
y[7] = r_init[7] * sin(theta[7] - 0.00972)
*/
void SpiralGalaxy_eqFunction_100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,100};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2006]] /* y[7] STATE(1,vy[7]) */) = ((data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */)) * (sin((data->simulationInfo->realParameter[1513] /* theta[7] PARAM */) - 0.00972));
  TRACE_POP
}

/*
equation index: 101
type: SIMPLE_ASSIGN
x[7] = r_init[7] * cos(theta[7] - 0.00972)
*/
void SpiralGalaxy_eqFunction_101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1506]] /* x[7] STATE(1,vx[7]) */) = ((data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */)) * (cos((data->simulationInfo->realParameter[1513] /* theta[7] PARAM */) - 0.00972));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8066(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8067(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8070(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8069(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8068(DATA *data, threadData_t *threadData);


/*
equation index: 107
type: SIMPLE_ASSIGN
vx[7] = (-sin(theta[7])) * r_init[7] * omega_c[7]
*/
void SpiralGalaxy_eqFunction_107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,107};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* vx[7] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1513] /* theta[7] PARAM */)))) * (((data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */)) * ((data->simulationInfo->realParameter[511] /* omega_c[7] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8063(DATA *data, threadData_t *threadData);


/*
equation index: 109
type: SIMPLE_ASSIGN
vy[7] = cos(theta[7]) * r_init[7] * omega_c[7]
*/
void SpiralGalaxy_eqFunction_109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[506]] /* vy[7] STATE(1) */) = (cos((data->simulationInfo->realParameter[1513] /* theta[7] PARAM */))) * (((data->simulationInfo->realParameter[1012] /* r_init[7] PARAM */)) * ((data->simulationInfo->realParameter[511] /* omega_c[7] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8062(DATA *data, threadData_t *threadData);


/*
equation index: 111
type: SIMPLE_ASSIGN
vz[7] = 0.0
*/
void SpiralGalaxy_eqFunction_111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,111};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1006]] /* vz[7] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8061(DATA *data, threadData_t *threadData);


/*
equation index: 113
type: SIMPLE_ASSIGN
z[8] = -0.03872
*/
void SpiralGalaxy_eqFunction_113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,113};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2507]] /* z[8] STATE(1,vz[8]) */) = -0.03872;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8074(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8075(DATA *data, threadData_t *threadData);


/*
equation index: 116
type: SIMPLE_ASSIGN
y[8] = r_init[8] * sin(theta[8] - 0.00968)
*/
void SpiralGalaxy_eqFunction_116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,116};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2007]] /* y[8] STATE(1,vy[8]) */) = ((data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */)) * (sin((data->simulationInfo->realParameter[1514] /* theta[8] PARAM */) - 0.00968));
  TRACE_POP
}

/*
equation index: 117
type: SIMPLE_ASSIGN
x[8] = r_init[8] * cos(theta[8] - 0.00968)
*/
void SpiralGalaxy_eqFunction_117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1507]] /* x[8] STATE(1,vx[8]) */) = ((data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */)) * (cos((data->simulationInfo->realParameter[1514] /* theta[8] PARAM */) - 0.00968));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8076(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8077(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8080(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8079(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8078(DATA *data, threadData_t *threadData);


/*
equation index: 123
type: SIMPLE_ASSIGN
vx[8] = (-sin(theta[8])) * r_init[8] * omega_c[8]
*/
void SpiralGalaxy_eqFunction_123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,123};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* vx[8] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1514] /* theta[8] PARAM */)))) * (((data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */)) * ((data->simulationInfo->realParameter[512] /* omega_c[8] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8073(DATA *data, threadData_t *threadData);


/*
equation index: 125
type: SIMPLE_ASSIGN
vy[8] = cos(theta[8]) * r_init[8] * omega_c[8]
*/
void SpiralGalaxy_eqFunction_125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[507]] /* vy[8] STATE(1) */) = (cos((data->simulationInfo->realParameter[1514] /* theta[8] PARAM */))) * (((data->simulationInfo->realParameter[1013] /* r_init[8] PARAM */)) * ((data->simulationInfo->realParameter[512] /* omega_c[8] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8072(DATA *data, threadData_t *threadData);


/*
equation index: 127
type: SIMPLE_ASSIGN
vz[8] = 0.0
*/
void SpiralGalaxy_eqFunction_127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,127};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1007]] /* vz[8] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8071(DATA *data, threadData_t *threadData);


/*
equation index: 129
type: SIMPLE_ASSIGN
z[9] = -0.03856
*/
void SpiralGalaxy_eqFunction_129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,129};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2508]] /* z[9] STATE(1,vz[9]) */) = -0.03856;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8084(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8085(DATA *data, threadData_t *threadData);


/*
equation index: 132
type: SIMPLE_ASSIGN
y[9] = r_init[9] * sin(theta[9] - 0.00964)
*/
void SpiralGalaxy_eqFunction_132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,132};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2008]] /* y[9] STATE(1,vy[9]) */) = ((data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */)) * (sin((data->simulationInfo->realParameter[1515] /* theta[9] PARAM */) - 0.00964));
  TRACE_POP
}

/*
equation index: 133
type: SIMPLE_ASSIGN
x[9] = r_init[9] * cos(theta[9] - 0.00964)
*/
void SpiralGalaxy_eqFunction_133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1508]] /* x[9] STATE(1,vx[9]) */) = ((data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */)) * (cos((data->simulationInfo->realParameter[1515] /* theta[9] PARAM */) - 0.00964));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8086(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8087(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8090(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8089(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8088(DATA *data, threadData_t *threadData);


/*
equation index: 139
type: SIMPLE_ASSIGN
vx[9] = (-sin(theta[9])) * r_init[9] * omega_c[9]
*/
void SpiralGalaxy_eqFunction_139(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,139};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* vx[9] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1515] /* theta[9] PARAM */)))) * (((data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */)) * ((data->simulationInfo->realParameter[513] /* omega_c[9] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8083(DATA *data, threadData_t *threadData);


/*
equation index: 141
type: SIMPLE_ASSIGN
vy[9] = cos(theta[9]) * r_init[9] * omega_c[9]
*/
void SpiralGalaxy_eqFunction_141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[508]] /* vy[9] STATE(1) */) = (cos((data->simulationInfo->realParameter[1515] /* theta[9] PARAM */))) * (((data->simulationInfo->realParameter[1014] /* r_init[9] PARAM */)) * ((data->simulationInfo->realParameter[513] /* omega_c[9] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8082(DATA *data, threadData_t *threadData);


/*
equation index: 143
type: SIMPLE_ASSIGN
vz[9] = 0.0
*/
void SpiralGalaxy_eqFunction_143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,143};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1008]] /* vz[9] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8081(DATA *data, threadData_t *threadData);


/*
equation index: 145
type: SIMPLE_ASSIGN
z[10] = -0.038400000000000004
*/
void SpiralGalaxy_eqFunction_145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,145};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2509]] /* z[10] STATE(1,vz[10]) */) = -0.038400000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8094(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8095(DATA *data, threadData_t *threadData);


/*
equation index: 148
type: SIMPLE_ASSIGN
y[10] = r_init[10] * sin(theta[10] - 0.0096)
*/
void SpiralGalaxy_eqFunction_148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,148};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2009]] /* y[10] STATE(1,vy[10]) */) = ((data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */)) * (sin((data->simulationInfo->realParameter[1516] /* theta[10] PARAM */) - 0.0096));
  TRACE_POP
}

/*
equation index: 149
type: SIMPLE_ASSIGN
x[10] = r_init[10] * cos(theta[10] - 0.0096)
*/
void SpiralGalaxy_eqFunction_149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1509]] /* x[10] STATE(1,vx[10]) */) = ((data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */)) * (cos((data->simulationInfo->realParameter[1516] /* theta[10] PARAM */) - 0.0096));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8096(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8097(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8100(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8099(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8098(DATA *data, threadData_t *threadData);


/*
equation index: 155
type: SIMPLE_ASSIGN
vx[10] = (-sin(theta[10])) * r_init[10] * omega_c[10]
*/
void SpiralGalaxy_eqFunction_155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,155};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* vx[10] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1516] /* theta[10] PARAM */)))) * (((data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */)) * ((data->simulationInfo->realParameter[514] /* omega_c[10] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8093(DATA *data, threadData_t *threadData);


/*
equation index: 157
type: SIMPLE_ASSIGN
vy[10] = cos(theta[10]) * r_init[10] * omega_c[10]
*/
void SpiralGalaxy_eqFunction_157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[509]] /* vy[10] STATE(1) */) = (cos((data->simulationInfo->realParameter[1516] /* theta[10] PARAM */))) * (((data->simulationInfo->realParameter[1015] /* r_init[10] PARAM */)) * ((data->simulationInfo->realParameter[514] /* omega_c[10] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8092(DATA *data, threadData_t *threadData);


/*
equation index: 159
type: SIMPLE_ASSIGN
vz[10] = 0.0
*/
void SpiralGalaxy_eqFunction_159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,159};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1009]] /* vz[10] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8091(DATA *data, threadData_t *threadData);


/*
equation index: 161
type: SIMPLE_ASSIGN
z[11] = -0.03824
*/
void SpiralGalaxy_eqFunction_161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,161};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2510]] /* z[11] STATE(1,vz[11]) */) = -0.03824;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8104(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8105(DATA *data, threadData_t *threadData);


/*
equation index: 164
type: SIMPLE_ASSIGN
y[11] = r_init[11] * sin(theta[11] - 0.009559999999999999)
*/
void SpiralGalaxy_eqFunction_164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,164};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2010]] /* y[11] STATE(1,vy[11]) */) = ((data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */)) * (sin((data->simulationInfo->realParameter[1517] /* theta[11] PARAM */) - 0.009559999999999999));
  TRACE_POP
}

/*
equation index: 165
type: SIMPLE_ASSIGN
x[11] = r_init[11] * cos(theta[11] - 0.009559999999999999)
*/
void SpiralGalaxy_eqFunction_165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1510]] /* x[11] STATE(1,vx[11]) */) = ((data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */)) * (cos((data->simulationInfo->realParameter[1517] /* theta[11] PARAM */) - 0.009559999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8106(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8107(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8110(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8109(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8108(DATA *data, threadData_t *threadData);


/*
equation index: 171
type: SIMPLE_ASSIGN
vx[11] = (-sin(theta[11])) * r_init[11] * omega_c[11]
*/
void SpiralGalaxy_eqFunction_171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,171};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* vx[11] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1517] /* theta[11] PARAM */)))) * (((data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */)) * ((data->simulationInfo->realParameter[515] /* omega_c[11] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8103(DATA *data, threadData_t *threadData);


/*
equation index: 173
type: SIMPLE_ASSIGN
vy[11] = cos(theta[11]) * r_init[11] * omega_c[11]
*/
void SpiralGalaxy_eqFunction_173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[510]] /* vy[11] STATE(1) */) = (cos((data->simulationInfo->realParameter[1517] /* theta[11] PARAM */))) * (((data->simulationInfo->realParameter[1016] /* r_init[11] PARAM */)) * ((data->simulationInfo->realParameter[515] /* omega_c[11] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8102(DATA *data, threadData_t *threadData);


/*
equation index: 175
type: SIMPLE_ASSIGN
vz[11] = 0.0
*/
void SpiralGalaxy_eqFunction_175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,175};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1010]] /* vz[11] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8101(DATA *data, threadData_t *threadData);


/*
equation index: 177
type: SIMPLE_ASSIGN
z[12] = -0.03808
*/
void SpiralGalaxy_eqFunction_177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,177};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2511]] /* z[12] STATE(1,vz[12]) */) = -0.03808;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8114(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8115(DATA *data, threadData_t *threadData);


/*
equation index: 180
type: SIMPLE_ASSIGN
y[12] = r_init[12] * sin(theta[12] - 0.009519999999999999)
*/
void SpiralGalaxy_eqFunction_180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,180};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2011]] /* y[12] STATE(1,vy[12]) */) = ((data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */)) * (sin((data->simulationInfo->realParameter[1518] /* theta[12] PARAM */) - 0.009519999999999999));
  TRACE_POP
}

/*
equation index: 181
type: SIMPLE_ASSIGN
x[12] = r_init[12] * cos(theta[12] - 0.009519999999999999)
*/
void SpiralGalaxy_eqFunction_181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1511]] /* x[12] STATE(1,vx[12]) */) = ((data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */)) * (cos((data->simulationInfo->realParameter[1518] /* theta[12] PARAM */) - 0.009519999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8116(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8117(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8120(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8119(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8118(DATA *data, threadData_t *threadData);


/*
equation index: 187
type: SIMPLE_ASSIGN
vx[12] = (-sin(theta[12])) * r_init[12] * omega_c[12]
*/
void SpiralGalaxy_eqFunction_187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,187};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* vx[12] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1518] /* theta[12] PARAM */)))) * (((data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */)) * ((data->simulationInfo->realParameter[516] /* omega_c[12] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8113(DATA *data, threadData_t *threadData);


/*
equation index: 189
type: SIMPLE_ASSIGN
vy[12] = cos(theta[12]) * r_init[12] * omega_c[12]
*/
void SpiralGalaxy_eqFunction_189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[511]] /* vy[12] STATE(1) */) = (cos((data->simulationInfo->realParameter[1518] /* theta[12] PARAM */))) * (((data->simulationInfo->realParameter[1017] /* r_init[12] PARAM */)) * ((data->simulationInfo->realParameter[516] /* omega_c[12] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8112(DATA *data, threadData_t *threadData);


/*
equation index: 191
type: SIMPLE_ASSIGN
vz[12] = 0.0
*/
void SpiralGalaxy_eqFunction_191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,191};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1011]] /* vz[12] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8111(DATA *data, threadData_t *threadData);


/*
equation index: 193
type: SIMPLE_ASSIGN
z[13] = -0.03792
*/
void SpiralGalaxy_eqFunction_193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,193};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2512]] /* z[13] STATE(1,vz[13]) */) = -0.03792;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8124(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8125(DATA *data, threadData_t *threadData);


/*
equation index: 196
type: SIMPLE_ASSIGN
y[13] = r_init[13] * sin(theta[13] - 0.00948)
*/
void SpiralGalaxy_eqFunction_196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,196};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2012]] /* y[13] STATE(1,vy[13]) */) = ((data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */)) * (sin((data->simulationInfo->realParameter[1519] /* theta[13] PARAM */) - 0.00948));
  TRACE_POP
}

/*
equation index: 197
type: SIMPLE_ASSIGN
x[13] = r_init[13] * cos(theta[13] - 0.00948)
*/
void SpiralGalaxy_eqFunction_197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1512]] /* x[13] STATE(1,vx[13]) */) = ((data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */)) * (cos((data->simulationInfo->realParameter[1519] /* theta[13] PARAM */) - 0.00948));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8126(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8127(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8130(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8129(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8128(DATA *data, threadData_t *threadData);


/*
equation index: 203
type: SIMPLE_ASSIGN
vx[13] = (-sin(theta[13])) * r_init[13] * omega_c[13]
*/
void SpiralGalaxy_eqFunction_203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,203};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* vx[13] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1519] /* theta[13] PARAM */)))) * (((data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */)) * ((data->simulationInfo->realParameter[517] /* omega_c[13] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8123(DATA *data, threadData_t *threadData);


/*
equation index: 205
type: SIMPLE_ASSIGN
vy[13] = cos(theta[13]) * r_init[13] * omega_c[13]
*/
void SpiralGalaxy_eqFunction_205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[512]] /* vy[13] STATE(1) */) = (cos((data->simulationInfo->realParameter[1519] /* theta[13] PARAM */))) * (((data->simulationInfo->realParameter[1018] /* r_init[13] PARAM */)) * ((data->simulationInfo->realParameter[517] /* omega_c[13] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8122(DATA *data, threadData_t *threadData);


/*
equation index: 207
type: SIMPLE_ASSIGN
vz[13] = 0.0
*/
void SpiralGalaxy_eqFunction_207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,207};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1012]] /* vz[13] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8121(DATA *data, threadData_t *threadData);


/*
equation index: 209
type: SIMPLE_ASSIGN
z[14] = -0.03776000000000001
*/
void SpiralGalaxy_eqFunction_209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,209};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2513]] /* z[14] STATE(1,vz[14]) */) = -0.03776000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8134(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8135(DATA *data, threadData_t *threadData);


/*
equation index: 212
type: SIMPLE_ASSIGN
y[14] = r_init[14] * sin(theta[14] - 0.00944)
*/
void SpiralGalaxy_eqFunction_212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,212};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2013]] /* y[14] STATE(1,vy[14]) */) = ((data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */)) * (sin((data->simulationInfo->realParameter[1520] /* theta[14] PARAM */) - 0.00944));
  TRACE_POP
}

/*
equation index: 213
type: SIMPLE_ASSIGN
x[14] = r_init[14] * cos(theta[14] - 0.00944)
*/
void SpiralGalaxy_eqFunction_213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1513]] /* x[14] STATE(1,vx[14]) */) = ((data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */)) * (cos((data->simulationInfo->realParameter[1520] /* theta[14] PARAM */) - 0.00944));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8136(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8137(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8140(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8139(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8138(DATA *data, threadData_t *threadData);


/*
equation index: 219
type: SIMPLE_ASSIGN
vx[14] = (-sin(theta[14])) * r_init[14] * omega_c[14]
*/
void SpiralGalaxy_eqFunction_219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,219};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* vx[14] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1520] /* theta[14] PARAM */)))) * (((data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */)) * ((data->simulationInfo->realParameter[518] /* omega_c[14] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8133(DATA *data, threadData_t *threadData);


/*
equation index: 221
type: SIMPLE_ASSIGN
vy[14] = cos(theta[14]) * r_init[14] * omega_c[14]
*/
void SpiralGalaxy_eqFunction_221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[513]] /* vy[14] STATE(1) */) = (cos((data->simulationInfo->realParameter[1520] /* theta[14] PARAM */))) * (((data->simulationInfo->realParameter[1019] /* r_init[14] PARAM */)) * ((data->simulationInfo->realParameter[518] /* omega_c[14] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8132(DATA *data, threadData_t *threadData);


/*
equation index: 223
type: SIMPLE_ASSIGN
vz[14] = 0.0
*/
void SpiralGalaxy_eqFunction_223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,223};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1013]] /* vz[14] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8131(DATA *data, threadData_t *threadData);


/*
equation index: 225
type: SIMPLE_ASSIGN
z[15] = -0.03760000000000001
*/
void SpiralGalaxy_eqFunction_225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,225};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2514]] /* z[15] STATE(1,vz[15]) */) = -0.03760000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8144(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8145(DATA *data, threadData_t *threadData);


/*
equation index: 228
type: SIMPLE_ASSIGN
y[15] = r_init[15] * sin(theta[15] - 0.0094)
*/
void SpiralGalaxy_eqFunction_228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,228};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2014]] /* y[15] STATE(1,vy[15]) */) = ((data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */)) * (sin((data->simulationInfo->realParameter[1521] /* theta[15] PARAM */) - 0.0094));
  TRACE_POP
}

/*
equation index: 229
type: SIMPLE_ASSIGN
x[15] = r_init[15] * cos(theta[15] - 0.0094)
*/
void SpiralGalaxy_eqFunction_229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1514]] /* x[15] STATE(1,vx[15]) */) = ((data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */)) * (cos((data->simulationInfo->realParameter[1521] /* theta[15] PARAM */) - 0.0094));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8146(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8147(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8150(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8149(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8148(DATA *data, threadData_t *threadData);


/*
equation index: 235
type: SIMPLE_ASSIGN
vx[15] = (-sin(theta[15])) * r_init[15] * omega_c[15]
*/
void SpiralGalaxy_eqFunction_235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,235};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* vx[15] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1521] /* theta[15] PARAM */)))) * (((data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */)) * ((data->simulationInfo->realParameter[519] /* omega_c[15] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8143(DATA *data, threadData_t *threadData);


/*
equation index: 237
type: SIMPLE_ASSIGN
vy[15] = cos(theta[15]) * r_init[15] * omega_c[15]
*/
void SpiralGalaxy_eqFunction_237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[514]] /* vy[15] STATE(1) */) = (cos((data->simulationInfo->realParameter[1521] /* theta[15] PARAM */))) * (((data->simulationInfo->realParameter[1020] /* r_init[15] PARAM */)) * ((data->simulationInfo->realParameter[519] /* omega_c[15] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8142(DATA *data, threadData_t *threadData);


/*
equation index: 239
type: SIMPLE_ASSIGN
vz[15] = 0.0
*/
void SpiralGalaxy_eqFunction_239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,239};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1014]] /* vz[15] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8141(DATA *data, threadData_t *threadData);


/*
equation index: 241
type: SIMPLE_ASSIGN
z[16] = -0.03744
*/
void SpiralGalaxy_eqFunction_241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,241};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2515]] /* z[16] STATE(1,vz[16]) */) = -0.03744;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8154(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8155(DATA *data, threadData_t *threadData);


/*
equation index: 244
type: SIMPLE_ASSIGN
y[16] = r_init[16] * sin(theta[16] - 0.00936)
*/
void SpiralGalaxy_eqFunction_244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,244};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2015]] /* y[16] STATE(1,vy[16]) */) = ((data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */)) * (sin((data->simulationInfo->realParameter[1522] /* theta[16] PARAM */) - 0.00936));
  TRACE_POP
}

/*
equation index: 245
type: SIMPLE_ASSIGN
x[16] = r_init[16] * cos(theta[16] - 0.00936)
*/
void SpiralGalaxy_eqFunction_245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1515]] /* x[16] STATE(1,vx[16]) */) = ((data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */)) * (cos((data->simulationInfo->realParameter[1522] /* theta[16] PARAM */) - 0.00936));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8156(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8157(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8160(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8159(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8158(DATA *data, threadData_t *threadData);


/*
equation index: 251
type: SIMPLE_ASSIGN
vx[16] = (-sin(theta[16])) * r_init[16] * omega_c[16]
*/
void SpiralGalaxy_eqFunction_251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,251};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* vx[16] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1522] /* theta[16] PARAM */)))) * (((data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */)) * ((data->simulationInfo->realParameter[520] /* omega_c[16] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8153(DATA *data, threadData_t *threadData);


/*
equation index: 253
type: SIMPLE_ASSIGN
vy[16] = cos(theta[16]) * r_init[16] * omega_c[16]
*/
void SpiralGalaxy_eqFunction_253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[515]] /* vy[16] STATE(1) */) = (cos((data->simulationInfo->realParameter[1522] /* theta[16] PARAM */))) * (((data->simulationInfo->realParameter[1021] /* r_init[16] PARAM */)) * ((data->simulationInfo->realParameter[520] /* omega_c[16] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8152(DATA *data, threadData_t *threadData);


/*
equation index: 255
type: SIMPLE_ASSIGN
vz[16] = 0.0
*/
void SpiralGalaxy_eqFunction_255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,255};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1015]] /* vz[16] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8151(DATA *data, threadData_t *threadData);


/*
equation index: 257
type: SIMPLE_ASSIGN
z[17] = -0.03728
*/
void SpiralGalaxy_eqFunction_257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,257};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2516]] /* z[17] STATE(1,vz[17]) */) = -0.03728;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8164(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8165(DATA *data, threadData_t *threadData);


/*
equation index: 260
type: SIMPLE_ASSIGN
y[17] = r_init[17] * sin(theta[17] - 0.00932)
*/
void SpiralGalaxy_eqFunction_260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,260};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2016]] /* y[17] STATE(1,vy[17]) */) = ((data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */)) * (sin((data->simulationInfo->realParameter[1523] /* theta[17] PARAM */) - 0.00932));
  TRACE_POP
}

/*
equation index: 261
type: SIMPLE_ASSIGN
x[17] = r_init[17] * cos(theta[17] - 0.00932)
*/
void SpiralGalaxy_eqFunction_261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1516]] /* x[17] STATE(1,vx[17]) */) = ((data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */)) * (cos((data->simulationInfo->realParameter[1523] /* theta[17] PARAM */) - 0.00932));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8166(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8167(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8170(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8169(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8168(DATA *data, threadData_t *threadData);


/*
equation index: 267
type: SIMPLE_ASSIGN
vx[17] = (-sin(theta[17])) * r_init[17] * omega_c[17]
*/
void SpiralGalaxy_eqFunction_267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,267};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* vx[17] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1523] /* theta[17] PARAM */)))) * (((data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */)) * ((data->simulationInfo->realParameter[521] /* omega_c[17] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8163(DATA *data, threadData_t *threadData);


/*
equation index: 269
type: SIMPLE_ASSIGN
vy[17] = cos(theta[17]) * r_init[17] * omega_c[17]
*/
void SpiralGalaxy_eqFunction_269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[516]] /* vy[17] STATE(1) */) = (cos((data->simulationInfo->realParameter[1523] /* theta[17] PARAM */))) * (((data->simulationInfo->realParameter[1022] /* r_init[17] PARAM */)) * ((data->simulationInfo->realParameter[521] /* omega_c[17] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8162(DATA *data, threadData_t *threadData);


/*
equation index: 271
type: SIMPLE_ASSIGN
vz[17] = 0.0
*/
void SpiralGalaxy_eqFunction_271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1016]] /* vz[17] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8161(DATA *data, threadData_t *threadData);


/*
equation index: 273
type: SIMPLE_ASSIGN
z[18] = -0.03712
*/
void SpiralGalaxy_eqFunction_273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2517]] /* z[18] STATE(1,vz[18]) */) = -0.03712;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8174(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8175(DATA *data, threadData_t *threadData);


/*
equation index: 276
type: SIMPLE_ASSIGN
y[18] = r_init[18] * sin(theta[18] - 0.00928)
*/
void SpiralGalaxy_eqFunction_276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,276};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2017]] /* y[18] STATE(1,vy[18]) */) = ((data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */)) * (sin((data->simulationInfo->realParameter[1524] /* theta[18] PARAM */) - 0.00928));
  TRACE_POP
}

/*
equation index: 277
type: SIMPLE_ASSIGN
x[18] = r_init[18] * cos(theta[18] - 0.00928)
*/
void SpiralGalaxy_eqFunction_277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1517]] /* x[18] STATE(1,vx[18]) */) = ((data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */)) * (cos((data->simulationInfo->realParameter[1524] /* theta[18] PARAM */) - 0.00928));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8176(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8177(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8180(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8179(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8178(DATA *data, threadData_t *threadData);


/*
equation index: 283
type: SIMPLE_ASSIGN
vx[18] = (-sin(theta[18])) * r_init[18] * omega_c[18]
*/
void SpiralGalaxy_eqFunction_283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,283};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* vx[18] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1524] /* theta[18] PARAM */)))) * (((data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */)) * ((data->simulationInfo->realParameter[522] /* omega_c[18] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8173(DATA *data, threadData_t *threadData);


/*
equation index: 285
type: SIMPLE_ASSIGN
vy[18] = cos(theta[18]) * r_init[18] * omega_c[18]
*/
void SpiralGalaxy_eqFunction_285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[517]] /* vy[18] STATE(1) */) = (cos((data->simulationInfo->realParameter[1524] /* theta[18] PARAM */))) * (((data->simulationInfo->realParameter[1023] /* r_init[18] PARAM */)) * ((data->simulationInfo->realParameter[522] /* omega_c[18] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8172(DATA *data, threadData_t *threadData);


/*
equation index: 287
type: SIMPLE_ASSIGN
vz[18] = 0.0
*/
void SpiralGalaxy_eqFunction_287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,287};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1017]] /* vz[18] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8171(DATA *data, threadData_t *threadData);


/*
equation index: 289
type: SIMPLE_ASSIGN
z[19] = -0.03696
*/
void SpiralGalaxy_eqFunction_289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,289};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2518]] /* z[19] STATE(1,vz[19]) */) = -0.03696;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8184(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8185(DATA *data, threadData_t *threadData);


/*
equation index: 292
type: SIMPLE_ASSIGN
y[19] = r_init[19] * sin(theta[19] - 0.00924)
*/
void SpiralGalaxy_eqFunction_292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,292};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2018]] /* y[19] STATE(1,vy[19]) */) = ((data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */)) * (sin((data->simulationInfo->realParameter[1525] /* theta[19] PARAM */) - 0.00924));
  TRACE_POP
}

/*
equation index: 293
type: SIMPLE_ASSIGN
x[19] = r_init[19] * cos(theta[19] - 0.00924)
*/
void SpiralGalaxy_eqFunction_293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1518]] /* x[19] STATE(1,vx[19]) */) = ((data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */)) * (cos((data->simulationInfo->realParameter[1525] /* theta[19] PARAM */) - 0.00924));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8186(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8187(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8190(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8189(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8188(DATA *data, threadData_t *threadData);


/*
equation index: 299
type: SIMPLE_ASSIGN
vx[19] = (-sin(theta[19])) * r_init[19] * omega_c[19]
*/
void SpiralGalaxy_eqFunction_299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,299};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* vx[19] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1525] /* theta[19] PARAM */)))) * (((data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */)) * ((data->simulationInfo->realParameter[523] /* omega_c[19] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8183(DATA *data, threadData_t *threadData);


/*
equation index: 301
type: SIMPLE_ASSIGN
vy[19] = cos(theta[19]) * r_init[19] * omega_c[19]
*/
void SpiralGalaxy_eqFunction_301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[518]] /* vy[19] STATE(1) */) = (cos((data->simulationInfo->realParameter[1525] /* theta[19] PARAM */))) * (((data->simulationInfo->realParameter[1024] /* r_init[19] PARAM */)) * ((data->simulationInfo->realParameter[523] /* omega_c[19] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8182(DATA *data, threadData_t *threadData);


/*
equation index: 303
type: SIMPLE_ASSIGN
vz[19] = 0.0
*/
void SpiralGalaxy_eqFunction_303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,303};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1018]] /* vz[19] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8181(DATA *data, threadData_t *threadData);


/*
equation index: 305
type: SIMPLE_ASSIGN
z[20] = -0.0368
*/
void SpiralGalaxy_eqFunction_305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,305};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2519]] /* z[20] STATE(1,vz[20]) */) = -0.0368;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8194(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8195(DATA *data, threadData_t *threadData);


/*
equation index: 308
type: SIMPLE_ASSIGN
y[20] = r_init[20] * sin(theta[20] - 0.0092)
*/
void SpiralGalaxy_eqFunction_308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,308};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2019]] /* y[20] STATE(1,vy[20]) */) = ((data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */)) * (sin((data->simulationInfo->realParameter[1526] /* theta[20] PARAM */) - 0.0092));
  TRACE_POP
}

/*
equation index: 309
type: SIMPLE_ASSIGN
x[20] = r_init[20] * cos(theta[20] - 0.0092)
*/
void SpiralGalaxy_eqFunction_309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1519]] /* x[20] STATE(1,vx[20]) */) = ((data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */)) * (cos((data->simulationInfo->realParameter[1526] /* theta[20] PARAM */) - 0.0092));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8196(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8197(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8200(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8199(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8198(DATA *data, threadData_t *threadData);


/*
equation index: 315
type: SIMPLE_ASSIGN
vx[20] = (-sin(theta[20])) * r_init[20] * omega_c[20]
*/
void SpiralGalaxy_eqFunction_315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,315};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* vx[20] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1526] /* theta[20] PARAM */)))) * (((data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */)) * ((data->simulationInfo->realParameter[524] /* omega_c[20] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8193(DATA *data, threadData_t *threadData);


/*
equation index: 317
type: SIMPLE_ASSIGN
vy[20] = cos(theta[20]) * r_init[20] * omega_c[20]
*/
void SpiralGalaxy_eqFunction_317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[519]] /* vy[20] STATE(1) */) = (cos((data->simulationInfo->realParameter[1526] /* theta[20] PARAM */))) * (((data->simulationInfo->realParameter[1025] /* r_init[20] PARAM */)) * ((data->simulationInfo->realParameter[524] /* omega_c[20] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8192(DATA *data, threadData_t *threadData);


/*
equation index: 319
type: SIMPLE_ASSIGN
vz[20] = 0.0
*/
void SpiralGalaxy_eqFunction_319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1019]] /* vz[20] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8191(DATA *data, threadData_t *threadData);


/*
equation index: 321
type: SIMPLE_ASSIGN
z[21] = -0.036640000000000006
*/
void SpiralGalaxy_eqFunction_321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2520]] /* z[21] STATE(1,vz[21]) */) = -0.036640000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8204(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8205(DATA *data, threadData_t *threadData);


/*
equation index: 324
type: SIMPLE_ASSIGN
y[21] = r_init[21] * sin(theta[21] - 0.00916)
*/
void SpiralGalaxy_eqFunction_324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,324};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2020]] /* y[21] STATE(1,vy[21]) */) = ((data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */)) * (sin((data->simulationInfo->realParameter[1527] /* theta[21] PARAM */) - 0.00916));
  TRACE_POP
}

/*
equation index: 325
type: SIMPLE_ASSIGN
x[21] = r_init[21] * cos(theta[21] - 0.00916)
*/
void SpiralGalaxy_eqFunction_325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1520]] /* x[21] STATE(1,vx[21]) */) = ((data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */)) * (cos((data->simulationInfo->realParameter[1527] /* theta[21] PARAM */) - 0.00916));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8206(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8207(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8210(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8209(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8208(DATA *data, threadData_t *threadData);


/*
equation index: 331
type: SIMPLE_ASSIGN
vx[21] = (-sin(theta[21])) * r_init[21] * omega_c[21]
*/
void SpiralGalaxy_eqFunction_331(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,331};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* vx[21] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1527] /* theta[21] PARAM */)))) * (((data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */)) * ((data->simulationInfo->realParameter[525] /* omega_c[21] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8203(DATA *data, threadData_t *threadData);


/*
equation index: 333
type: SIMPLE_ASSIGN
vy[21] = cos(theta[21]) * r_init[21] * omega_c[21]
*/
void SpiralGalaxy_eqFunction_333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[520]] /* vy[21] STATE(1) */) = (cos((data->simulationInfo->realParameter[1527] /* theta[21] PARAM */))) * (((data->simulationInfo->realParameter[1026] /* r_init[21] PARAM */)) * ((data->simulationInfo->realParameter[525] /* omega_c[21] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8202(DATA *data, threadData_t *threadData);


/*
equation index: 335
type: SIMPLE_ASSIGN
vz[21] = 0.0
*/
void SpiralGalaxy_eqFunction_335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1020]] /* vz[21] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8201(DATA *data, threadData_t *threadData);


/*
equation index: 337
type: SIMPLE_ASSIGN
z[22] = -0.036480000000000005
*/
void SpiralGalaxy_eqFunction_337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2521]] /* z[22] STATE(1,vz[22]) */) = -0.036480000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8214(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8215(DATA *data, threadData_t *threadData);


/*
equation index: 340
type: SIMPLE_ASSIGN
y[22] = r_init[22] * sin(theta[22] - 0.009120000000000001)
*/
void SpiralGalaxy_eqFunction_340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,340};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2021]] /* y[22] STATE(1,vy[22]) */) = ((data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */)) * (sin((data->simulationInfo->realParameter[1528] /* theta[22] PARAM */) - 0.009120000000000001));
  TRACE_POP
}

/*
equation index: 341
type: SIMPLE_ASSIGN
x[22] = r_init[22] * cos(theta[22] - 0.009120000000000001)
*/
void SpiralGalaxy_eqFunction_341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1521]] /* x[22] STATE(1,vx[22]) */) = ((data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */)) * (cos((data->simulationInfo->realParameter[1528] /* theta[22] PARAM */) - 0.009120000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8216(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8217(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8220(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8219(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8218(DATA *data, threadData_t *threadData);


/*
equation index: 347
type: SIMPLE_ASSIGN
vx[22] = (-sin(theta[22])) * r_init[22] * omega_c[22]
*/
void SpiralGalaxy_eqFunction_347(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,347};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* vx[22] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1528] /* theta[22] PARAM */)))) * (((data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */)) * ((data->simulationInfo->realParameter[526] /* omega_c[22] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8213(DATA *data, threadData_t *threadData);


/*
equation index: 349
type: SIMPLE_ASSIGN
vy[22] = cos(theta[22]) * r_init[22] * omega_c[22]
*/
void SpiralGalaxy_eqFunction_349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[521]] /* vy[22] STATE(1) */) = (cos((data->simulationInfo->realParameter[1528] /* theta[22] PARAM */))) * (((data->simulationInfo->realParameter[1027] /* r_init[22] PARAM */)) * ((data->simulationInfo->realParameter[526] /* omega_c[22] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8212(DATA *data, threadData_t *threadData);


/*
equation index: 351
type: SIMPLE_ASSIGN
vz[22] = 0.0
*/
void SpiralGalaxy_eqFunction_351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,351};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1021]] /* vz[22] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8211(DATA *data, threadData_t *threadData);


/*
equation index: 353
type: SIMPLE_ASSIGN
z[23] = -0.036320000000000005
*/
void SpiralGalaxy_eqFunction_353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,353};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2522]] /* z[23] STATE(1,vz[23]) */) = -0.036320000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8224(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8225(DATA *data, threadData_t *threadData);


/*
equation index: 356
type: SIMPLE_ASSIGN
y[23] = r_init[23] * sin(theta[23] - 0.009080000000000001)
*/
void SpiralGalaxy_eqFunction_356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,356};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2022]] /* y[23] STATE(1,vy[23]) */) = ((data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */)) * (sin((data->simulationInfo->realParameter[1529] /* theta[23] PARAM */) - 0.009080000000000001));
  TRACE_POP
}

/*
equation index: 357
type: SIMPLE_ASSIGN
x[23] = r_init[23] * cos(theta[23] - 0.009080000000000001)
*/
void SpiralGalaxy_eqFunction_357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1522]] /* x[23] STATE(1,vx[23]) */) = ((data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */)) * (cos((data->simulationInfo->realParameter[1529] /* theta[23] PARAM */) - 0.009080000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8226(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8227(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8230(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8229(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8228(DATA *data, threadData_t *threadData);


/*
equation index: 363
type: SIMPLE_ASSIGN
vx[23] = (-sin(theta[23])) * r_init[23] * omega_c[23]
*/
void SpiralGalaxy_eqFunction_363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,363};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* vx[23] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1529] /* theta[23] PARAM */)))) * (((data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */)) * ((data->simulationInfo->realParameter[527] /* omega_c[23] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8223(DATA *data, threadData_t *threadData);


/*
equation index: 365
type: SIMPLE_ASSIGN
vy[23] = cos(theta[23]) * r_init[23] * omega_c[23]
*/
void SpiralGalaxy_eqFunction_365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[522]] /* vy[23] STATE(1) */) = (cos((data->simulationInfo->realParameter[1529] /* theta[23] PARAM */))) * (((data->simulationInfo->realParameter[1028] /* r_init[23] PARAM */)) * ((data->simulationInfo->realParameter[527] /* omega_c[23] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8222(DATA *data, threadData_t *threadData);


/*
equation index: 367
type: SIMPLE_ASSIGN
vz[23] = 0.0
*/
void SpiralGalaxy_eqFunction_367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,367};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1022]] /* vz[23] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8221(DATA *data, threadData_t *threadData);


/*
equation index: 369
type: SIMPLE_ASSIGN
z[24] = -0.036160000000000005
*/
void SpiralGalaxy_eqFunction_369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,369};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2523]] /* z[24] STATE(1,vz[24]) */) = -0.036160000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8234(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8235(DATA *data, threadData_t *threadData);


/*
equation index: 372
type: SIMPLE_ASSIGN
y[24] = r_init[24] * sin(theta[24] - 0.009040000000000001)
*/
void SpiralGalaxy_eqFunction_372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,372};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2023]] /* y[24] STATE(1,vy[24]) */) = ((data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */)) * (sin((data->simulationInfo->realParameter[1530] /* theta[24] PARAM */) - 0.009040000000000001));
  TRACE_POP
}

/*
equation index: 373
type: SIMPLE_ASSIGN
x[24] = r_init[24] * cos(theta[24] - 0.009040000000000001)
*/
void SpiralGalaxy_eqFunction_373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1523]] /* x[24] STATE(1,vx[24]) */) = ((data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */)) * (cos((data->simulationInfo->realParameter[1530] /* theta[24] PARAM */) - 0.009040000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8236(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8237(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8240(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8239(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8238(DATA *data, threadData_t *threadData);


/*
equation index: 379
type: SIMPLE_ASSIGN
vx[24] = (-sin(theta[24])) * r_init[24] * omega_c[24]
*/
void SpiralGalaxy_eqFunction_379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,379};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* vx[24] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1530] /* theta[24] PARAM */)))) * (((data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */)) * ((data->simulationInfo->realParameter[528] /* omega_c[24] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8233(DATA *data, threadData_t *threadData);


/*
equation index: 381
type: SIMPLE_ASSIGN
vy[24] = cos(theta[24]) * r_init[24] * omega_c[24]
*/
void SpiralGalaxy_eqFunction_381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[523]] /* vy[24] STATE(1) */) = (cos((data->simulationInfo->realParameter[1530] /* theta[24] PARAM */))) * (((data->simulationInfo->realParameter[1029] /* r_init[24] PARAM */)) * ((data->simulationInfo->realParameter[528] /* omega_c[24] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8232(DATA *data, threadData_t *threadData);


/*
equation index: 383
type: SIMPLE_ASSIGN
vz[24] = 0.0
*/
void SpiralGalaxy_eqFunction_383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,383};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1023]] /* vz[24] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8231(DATA *data, threadData_t *threadData);


/*
equation index: 385
type: SIMPLE_ASSIGN
z[25] = -0.036000000000000004
*/
void SpiralGalaxy_eqFunction_385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,385};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2524]] /* z[25] STATE(1,vz[25]) */) = -0.036000000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8244(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8245(DATA *data, threadData_t *threadData);


/*
equation index: 388
type: SIMPLE_ASSIGN
y[25] = r_init[25] * sin(theta[25] - 0.009000000000000001)
*/
void SpiralGalaxy_eqFunction_388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,388};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2024]] /* y[25] STATE(1,vy[25]) */) = ((data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */)) * (sin((data->simulationInfo->realParameter[1531] /* theta[25] PARAM */) - 0.009000000000000001));
  TRACE_POP
}

/*
equation index: 389
type: SIMPLE_ASSIGN
x[25] = r_init[25] * cos(theta[25] - 0.009000000000000001)
*/
void SpiralGalaxy_eqFunction_389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1524]] /* x[25] STATE(1,vx[25]) */) = ((data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */)) * (cos((data->simulationInfo->realParameter[1531] /* theta[25] PARAM */) - 0.009000000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8246(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8247(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8250(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8249(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8248(DATA *data, threadData_t *threadData);


/*
equation index: 395
type: SIMPLE_ASSIGN
vx[25] = (-sin(theta[25])) * r_init[25] * omega_c[25]
*/
void SpiralGalaxy_eqFunction_395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,395};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* vx[25] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1531] /* theta[25] PARAM */)))) * (((data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */)) * ((data->simulationInfo->realParameter[529] /* omega_c[25] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8243(DATA *data, threadData_t *threadData);


/*
equation index: 397
type: SIMPLE_ASSIGN
vy[25] = cos(theta[25]) * r_init[25] * omega_c[25]
*/
void SpiralGalaxy_eqFunction_397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[524]] /* vy[25] STATE(1) */) = (cos((data->simulationInfo->realParameter[1531] /* theta[25] PARAM */))) * (((data->simulationInfo->realParameter[1030] /* r_init[25] PARAM */)) * ((data->simulationInfo->realParameter[529] /* omega_c[25] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8242(DATA *data, threadData_t *threadData);


/*
equation index: 399
type: SIMPLE_ASSIGN
vz[25] = 0.0
*/
void SpiralGalaxy_eqFunction_399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,399};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1024]] /* vz[25] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8241(DATA *data, threadData_t *threadData);


/*
equation index: 401
type: SIMPLE_ASSIGN
z[26] = -0.035840000000000004
*/
void SpiralGalaxy_eqFunction_401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,401};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2525]] /* z[26] STATE(1,vz[26]) */) = -0.035840000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8254(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8255(DATA *data, threadData_t *threadData);


/*
equation index: 404
type: SIMPLE_ASSIGN
y[26] = r_init[26] * sin(theta[26] - 0.008960000000000001)
*/
void SpiralGalaxy_eqFunction_404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,404};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2025]] /* y[26] STATE(1,vy[26]) */) = ((data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */)) * (sin((data->simulationInfo->realParameter[1532] /* theta[26] PARAM */) - 0.008960000000000001));
  TRACE_POP
}

/*
equation index: 405
type: SIMPLE_ASSIGN
x[26] = r_init[26] * cos(theta[26] - 0.008960000000000001)
*/
void SpiralGalaxy_eqFunction_405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1525]] /* x[26] STATE(1,vx[26]) */) = ((data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */)) * (cos((data->simulationInfo->realParameter[1532] /* theta[26] PARAM */) - 0.008960000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8256(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8257(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8260(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8259(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8258(DATA *data, threadData_t *threadData);


/*
equation index: 411
type: SIMPLE_ASSIGN
vx[26] = (-sin(theta[26])) * r_init[26] * omega_c[26]
*/
void SpiralGalaxy_eqFunction_411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,411};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* vx[26] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1532] /* theta[26] PARAM */)))) * (((data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */)) * ((data->simulationInfo->realParameter[530] /* omega_c[26] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8253(DATA *data, threadData_t *threadData);


/*
equation index: 413
type: SIMPLE_ASSIGN
vy[26] = cos(theta[26]) * r_init[26] * omega_c[26]
*/
void SpiralGalaxy_eqFunction_413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[525]] /* vy[26] STATE(1) */) = (cos((data->simulationInfo->realParameter[1532] /* theta[26] PARAM */))) * (((data->simulationInfo->realParameter[1031] /* r_init[26] PARAM */)) * ((data->simulationInfo->realParameter[530] /* omega_c[26] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8252(DATA *data, threadData_t *threadData);


/*
equation index: 415
type: SIMPLE_ASSIGN
vz[26] = 0.0
*/
void SpiralGalaxy_eqFunction_415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,415};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1025]] /* vz[26] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8251(DATA *data, threadData_t *threadData);


/*
equation index: 417
type: SIMPLE_ASSIGN
z[27] = -0.03568
*/
void SpiralGalaxy_eqFunction_417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,417};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2526]] /* z[27] STATE(1,vz[27]) */) = -0.03568;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8264(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8265(DATA *data, threadData_t *threadData);


/*
equation index: 420
type: SIMPLE_ASSIGN
y[27] = r_init[27] * sin(theta[27] - 0.00892)
*/
void SpiralGalaxy_eqFunction_420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,420};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2026]] /* y[27] STATE(1,vy[27]) */) = ((data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */)) * (sin((data->simulationInfo->realParameter[1533] /* theta[27] PARAM */) - 0.00892));
  TRACE_POP
}

/*
equation index: 421
type: SIMPLE_ASSIGN
x[27] = r_init[27] * cos(theta[27] - 0.00892)
*/
void SpiralGalaxy_eqFunction_421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1526]] /* x[27] STATE(1,vx[27]) */) = ((data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */)) * (cos((data->simulationInfo->realParameter[1533] /* theta[27] PARAM */) - 0.00892));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8266(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8267(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8270(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8269(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8268(DATA *data, threadData_t *threadData);


/*
equation index: 427
type: SIMPLE_ASSIGN
vx[27] = (-sin(theta[27])) * r_init[27] * omega_c[27]
*/
void SpiralGalaxy_eqFunction_427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,427};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* vx[27] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1533] /* theta[27] PARAM */)))) * (((data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */)) * ((data->simulationInfo->realParameter[531] /* omega_c[27] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8263(DATA *data, threadData_t *threadData);


/*
equation index: 429
type: SIMPLE_ASSIGN
vy[27] = cos(theta[27]) * r_init[27] * omega_c[27]
*/
void SpiralGalaxy_eqFunction_429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[526]] /* vy[27] STATE(1) */) = (cos((data->simulationInfo->realParameter[1533] /* theta[27] PARAM */))) * (((data->simulationInfo->realParameter[1032] /* r_init[27] PARAM */)) * ((data->simulationInfo->realParameter[531] /* omega_c[27] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8262(DATA *data, threadData_t *threadData);


/*
equation index: 431
type: SIMPLE_ASSIGN
vz[27] = 0.0
*/
void SpiralGalaxy_eqFunction_431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,431};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1026]] /* vz[27] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8261(DATA *data, threadData_t *threadData);


/*
equation index: 433
type: SIMPLE_ASSIGN
z[28] = -0.03552
*/
void SpiralGalaxy_eqFunction_433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,433};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2527]] /* z[28] STATE(1,vz[28]) */) = -0.03552;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8274(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8275(DATA *data, threadData_t *threadData);


/*
equation index: 436
type: SIMPLE_ASSIGN
y[28] = r_init[28] * sin(theta[28] - 0.00888)
*/
void SpiralGalaxy_eqFunction_436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,436};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2027]] /* y[28] STATE(1,vy[28]) */) = ((data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */)) * (sin((data->simulationInfo->realParameter[1534] /* theta[28] PARAM */) - 0.00888));
  TRACE_POP
}

/*
equation index: 437
type: SIMPLE_ASSIGN
x[28] = r_init[28] * cos(theta[28] - 0.00888)
*/
void SpiralGalaxy_eqFunction_437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1527]] /* x[28] STATE(1,vx[28]) */) = ((data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */)) * (cos((data->simulationInfo->realParameter[1534] /* theta[28] PARAM */) - 0.00888));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8276(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8277(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8280(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8279(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8278(DATA *data, threadData_t *threadData);


/*
equation index: 443
type: SIMPLE_ASSIGN
vx[28] = (-sin(theta[28])) * r_init[28] * omega_c[28]
*/
void SpiralGalaxy_eqFunction_443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,443};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* vx[28] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1534] /* theta[28] PARAM */)))) * (((data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */)) * ((data->simulationInfo->realParameter[532] /* omega_c[28] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8273(DATA *data, threadData_t *threadData);


/*
equation index: 445
type: SIMPLE_ASSIGN
vy[28] = cos(theta[28]) * r_init[28] * omega_c[28]
*/
void SpiralGalaxy_eqFunction_445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[527]] /* vy[28] STATE(1) */) = (cos((data->simulationInfo->realParameter[1534] /* theta[28] PARAM */))) * (((data->simulationInfo->realParameter[1033] /* r_init[28] PARAM */)) * ((data->simulationInfo->realParameter[532] /* omega_c[28] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8272(DATA *data, threadData_t *threadData);


/*
equation index: 447
type: SIMPLE_ASSIGN
vz[28] = 0.0
*/
void SpiralGalaxy_eqFunction_447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,447};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1027]] /* vz[28] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8271(DATA *data, threadData_t *threadData);


/*
equation index: 449
type: SIMPLE_ASSIGN
z[29] = -0.03536
*/
void SpiralGalaxy_eqFunction_449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,449};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2528]] /* z[29] STATE(1,vz[29]) */) = -0.03536;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8284(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8285(DATA *data, threadData_t *threadData);


/*
equation index: 452
type: SIMPLE_ASSIGN
y[29] = r_init[29] * sin(theta[29] - 0.00884)
*/
void SpiralGalaxy_eqFunction_452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,452};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2028]] /* y[29] STATE(1,vy[29]) */) = ((data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */)) * (sin((data->simulationInfo->realParameter[1535] /* theta[29] PARAM */) - 0.00884));
  TRACE_POP
}

/*
equation index: 453
type: SIMPLE_ASSIGN
x[29] = r_init[29] * cos(theta[29] - 0.00884)
*/
void SpiralGalaxy_eqFunction_453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1528]] /* x[29] STATE(1,vx[29]) */) = ((data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */)) * (cos((data->simulationInfo->realParameter[1535] /* theta[29] PARAM */) - 0.00884));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8286(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8287(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8290(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8289(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8288(DATA *data, threadData_t *threadData);


/*
equation index: 459
type: SIMPLE_ASSIGN
vx[29] = (-sin(theta[29])) * r_init[29] * omega_c[29]
*/
void SpiralGalaxy_eqFunction_459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,459};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* vx[29] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1535] /* theta[29] PARAM */)))) * (((data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */)) * ((data->simulationInfo->realParameter[533] /* omega_c[29] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8283(DATA *data, threadData_t *threadData);


/*
equation index: 461
type: SIMPLE_ASSIGN
vy[29] = cos(theta[29]) * r_init[29] * omega_c[29]
*/
void SpiralGalaxy_eqFunction_461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[528]] /* vy[29] STATE(1) */) = (cos((data->simulationInfo->realParameter[1535] /* theta[29] PARAM */))) * (((data->simulationInfo->realParameter[1034] /* r_init[29] PARAM */)) * ((data->simulationInfo->realParameter[533] /* omega_c[29] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8282(DATA *data, threadData_t *threadData);


/*
equation index: 463
type: SIMPLE_ASSIGN
vz[29] = 0.0
*/
void SpiralGalaxy_eqFunction_463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,463};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1028]] /* vz[29] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8281(DATA *data, threadData_t *threadData);


/*
equation index: 465
type: SIMPLE_ASSIGN
z[30] = -0.0352
*/
void SpiralGalaxy_eqFunction_465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,465};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2529]] /* z[30] STATE(1,vz[30]) */) = -0.0352;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8294(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8295(DATA *data, threadData_t *threadData);


/*
equation index: 468
type: SIMPLE_ASSIGN
y[30] = r_init[30] * sin(theta[30] - 0.0088)
*/
void SpiralGalaxy_eqFunction_468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,468};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2029]] /* y[30] STATE(1,vy[30]) */) = ((data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */)) * (sin((data->simulationInfo->realParameter[1536] /* theta[30] PARAM */) - 0.0088));
  TRACE_POP
}

/*
equation index: 469
type: SIMPLE_ASSIGN
x[30] = r_init[30] * cos(theta[30] - 0.0088)
*/
void SpiralGalaxy_eqFunction_469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1529]] /* x[30] STATE(1,vx[30]) */) = ((data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */)) * (cos((data->simulationInfo->realParameter[1536] /* theta[30] PARAM */) - 0.0088));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8296(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8297(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8300(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8299(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8298(DATA *data, threadData_t *threadData);


/*
equation index: 475
type: SIMPLE_ASSIGN
vx[30] = (-sin(theta[30])) * r_init[30] * omega_c[30]
*/
void SpiralGalaxy_eqFunction_475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,475};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* vx[30] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1536] /* theta[30] PARAM */)))) * (((data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */)) * ((data->simulationInfo->realParameter[534] /* omega_c[30] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8293(DATA *data, threadData_t *threadData);


/*
equation index: 477
type: SIMPLE_ASSIGN
vy[30] = cos(theta[30]) * r_init[30] * omega_c[30]
*/
void SpiralGalaxy_eqFunction_477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[529]] /* vy[30] STATE(1) */) = (cos((data->simulationInfo->realParameter[1536] /* theta[30] PARAM */))) * (((data->simulationInfo->realParameter[1035] /* r_init[30] PARAM */)) * ((data->simulationInfo->realParameter[534] /* omega_c[30] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8292(DATA *data, threadData_t *threadData);


/*
equation index: 479
type: SIMPLE_ASSIGN
vz[30] = 0.0
*/
void SpiralGalaxy_eqFunction_479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,479};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1029]] /* vz[30] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8291(DATA *data, threadData_t *threadData);


/*
equation index: 481
type: SIMPLE_ASSIGN
z[31] = -0.03504
*/
void SpiralGalaxy_eqFunction_481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,481};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2530]] /* z[31] STATE(1,vz[31]) */) = -0.03504;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8304(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8305(DATA *data, threadData_t *threadData);


/*
equation index: 484
type: SIMPLE_ASSIGN
y[31] = r_init[31] * sin(theta[31] - 0.00876)
*/
void SpiralGalaxy_eqFunction_484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,484};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2030]] /* y[31] STATE(1,vy[31]) */) = ((data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */)) * (sin((data->simulationInfo->realParameter[1537] /* theta[31] PARAM */) - 0.00876));
  TRACE_POP
}

/*
equation index: 485
type: SIMPLE_ASSIGN
x[31] = r_init[31] * cos(theta[31] - 0.00876)
*/
void SpiralGalaxy_eqFunction_485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1530]] /* x[31] STATE(1,vx[31]) */) = ((data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */)) * (cos((data->simulationInfo->realParameter[1537] /* theta[31] PARAM */) - 0.00876));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8306(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8307(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8310(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8309(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8308(DATA *data, threadData_t *threadData);


/*
equation index: 491
type: SIMPLE_ASSIGN
vx[31] = (-sin(theta[31])) * r_init[31] * omega_c[31]
*/
void SpiralGalaxy_eqFunction_491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,491};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* vx[31] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1537] /* theta[31] PARAM */)))) * (((data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */)) * ((data->simulationInfo->realParameter[535] /* omega_c[31] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8303(DATA *data, threadData_t *threadData);


/*
equation index: 493
type: SIMPLE_ASSIGN
vy[31] = cos(theta[31]) * r_init[31] * omega_c[31]
*/
void SpiralGalaxy_eqFunction_493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[530]] /* vy[31] STATE(1) */) = (cos((data->simulationInfo->realParameter[1537] /* theta[31] PARAM */))) * (((data->simulationInfo->realParameter[1036] /* r_init[31] PARAM */)) * ((data->simulationInfo->realParameter[535] /* omega_c[31] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8302(DATA *data, threadData_t *threadData);


/*
equation index: 495
type: SIMPLE_ASSIGN
vz[31] = 0.0
*/
void SpiralGalaxy_eqFunction_495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,495};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1030]] /* vz[31] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8301(DATA *data, threadData_t *threadData);


/*
equation index: 497
type: SIMPLE_ASSIGN
z[32] = -0.03488
*/
void SpiralGalaxy_eqFunction_497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,497};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2531]] /* z[32] STATE(1,vz[32]) */) = -0.03488;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8314(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8315(DATA *data, threadData_t *threadData);


/*
equation index: 500
type: SIMPLE_ASSIGN
y[32] = r_init[32] * sin(theta[32] - 0.00872)
*/
void SpiralGalaxy_eqFunction_500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,500};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2031]] /* y[32] STATE(1,vy[32]) */) = ((data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */)) * (sin((data->simulationInfo->realParameter[1538] /* theta[32] PARAM */) - 0.00872));
  TRACE_POP
}
OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_1(data, threadData);
  SpiralGalaxy_eqFunction_8004(data, threadData);
  SpiralGalaxy_eqFunction_8005(data, threadData);
  SpiralGalaxy_eqFunction_4(data, threadData);
  SpiralGalaxy_eqFunction_5(data, threadData);
  SpiralGalaxy_eqFunction_8006(data, threadData);
  SpiralGalaxy_eqFunction_8007(data, threadData);
  SpiralGalaxy_eqFunction_8010(data, threadData);
  SpiralGalaxy_eqFunction_8009(data, threadData);
  SpiralGalaxy_eqFunction_8008(data, threadData);
  SpiralGalaxy_eqFunction_11(data, threadData);
  SpiralGalaxy_eqFunction_8003(data, threadData);
  SpiralGalaxy_eqFunction_13(data, threadData);
  SpiralGalaxy_eqFunction_8002(data, threadData);
  SpiralGalaxy_eqFunction_15(data, threadData);
  SpiralGalaxy_eqFunction_8001(data, threadData);
  SpiralGalaxy_eqFunction_17(data, threadData);
  SpiralGalaxy_eqFunction_8014(data, threadData);
  SpiralGalaxy_eqFunction_8015(data, threadData);
  SpiralGalaxy_eqFunction_20(data, threadData);
  SpiralGalaxy_eqFunction_21(data, threadData);
  SpiralGalaxy_eqFunction_8016(data, threadData);
  SpiralGalaxy_eqFunction_8017(data, threadData);
  SpiralGalaxy_eqFunction_8020(data, threadData);
  SpiralGalaxy_eqFunction_8019(data, threadData);
  SpiralGalaxy_eqFunction_8018(data, threadData);
  SpiralGalaxy_eqFunction_27(data, threadData);
  SpiralGalaxy_eqFunction_8013(data, threadData);
  SpiralGalaxy_eqFunction_29(data, threadData);
  SpiralGalaxy_eqFunction_8012(data, threadData);
  SpiralGalaxy_eqFunction_31(data, threadData);
  SpiralGalaxy_eqFunction_8011(data, threadData);
  SpiralGalaxy_eqFunction_33(data, threadData);
  SpiralGalaxy_eqFunction_8024(data, threadData);
  SpiralGalaxy_eqFunction_8025(data, threadData);
  SpiralGalaxy_eqFunction_36(data, threadData);
  SpiralGalaxy_eqFunction_37(data, threadData);
  SpiralGalaxy_eqFunction_8026(data, threadData);
  SpiralGalaxy_eqFunction_8027(data, threadData);
  SpiralGalaxy_eqFunction_8030(data, threadData);
  SpiralGalaxy_eqFunction_8029(data, threadData);
  SpiralGalaxy_eqFunction_8028(data, threadData);
  SpiralGalaxy_eqFunction_43(data, threadData);
  SpiralGalaxy_eqFunction_8023(data, threadData);
  SpiralGalaxy_eqFunction_45(data, threadData);
  SpiralGalaxy_eqFunction_8022(data, threadData);
  SpiralGalaxy_eqFunction_47(data, threadData);
  SpiralGalaxy_eqFunction_8021(data, threadData);
  SpiralGalaxy_eqFunction_49(data, threadData);
  SpiralGalaxy_eqFunction_8034(data, threadData);
  SpiralGalaxy_eqFunction_8035(data, threadData);
  SpiralGalaxy_eqFunction_52(data, threadData);
  SpiralGalaxy_eqFunction_53(data, threadData);
  SpiralGalaxy_eqFunction_8036(data, threadData);
  SpiralGalaxy_eqFunction_8037(data, threadData);
  SpiralGalaxy_eqFunction_8040(data, threadData);
  SpiralGalaxy_eqFunction_8039(data, threadData);
  SpiralGalaxy_eqFunction_8038(data, threadData);
  SpiralGalaxy_eqFunction_59(data, threadData);
  SpiralGalaxy_eqFunction_8033(data, threadData);
  SpiralGalaxy_eqFunction_61(data, threadData);
  SpiralGalaxy_eqFunction_8032(data, threadData);
  SpiralGalaxy_eqFunction_63(data, threadData);
  SpiralGalaxy_eqFunction_8031(data, threadData);
  SpiralGalaxy_eqFunction_65(data, threadData);
  SpiralGalaxy_eqFunction_8044(data, threadData);
  SpiralGalaxy_eqFunction_8045(data, threadData);
  SpiralGalaxy_eqFunction_68(data, threadData);
  SpiralGalaxy_eqFunction_69(data, threadData);
  SpiralGalaxy_eqFunction_8046(data, threadData);
  SpiralGalaxy_eqFunction_8047(data, threadData);
  SpiralGalaxy_eqFunction_8050(data, threadData);
  SpiralGalaxy_eqFunction_8049(data, threadData);
  SpiralGalaxy_eqFunction_8048(data, threadData);
  SpiralGalaxy_eqFunction_75(data, threadData);
  SpiralGalaxy_eqFunction_8043(data, threadData);
  SpiralGalaxy_eqFunction_77(data, threadData);
  SpiralGalaxy_eqFunction_8042(data, threadData);
  SpiralGalaxy_eqFunction_79(data, threadData);
  SpiralGalaxy_eqFunction_8041(data, threadData);
  SpiralGalaxy_eqFunction_81(data, threadData);
  SpiralGalaxy_eqFunction_8054(data, threadData);
  SpiralGalaxy_eqFunction_8055(data, threadData);
  SpiralGalaxy_eqFunction_84(data, threadData);
  SpiralGalaxy_eqFunction_85(data, threadData);
  SpiralGalaxy_eqFunction_8056(data, threadData);
  SpiralGalaxy_eqFunction_8057(data, threadData);
  SpiralGalaxy_eqFunction_8060(data, threadData);
  SpiralGalaxy_eqFunction_8059(data, threadData);
  SpiralGalaxy_eqFunction_8058(data, threadData);
  SpiralGalaxy_eqFunction_91(data, threadData);
  SpiralGalaxy_eqFunction_8053(data, threadData);
  SpiralGalaxy_eqFunction_93(data, threadData);
  SpiralGalaxy_eqFunction_8052(data, threadData);
  SpiralGalaxy_eqFunction_95(data, threadData);
  SpiralGalaxy_eqFunction_8051(data, threadData);
  SpiralGalaxy_eqFunction_97(data, threadData);
  SpiralGalaxy_eqFunction_8064(data, threadData);
  SpiralGalaxy_eqFunction_8065(data, threadData);
  SpiralGalaxy_eqFunction_100(data, threadData);
  SpiralGalaxy_eqFunction_101(data, threadData);
  SpiralGalaxy_eqFunction_8066(data, threadData);
  SpiralGalaxy_eqFunction_8067(data, threadData);
  SpiralGalaxy_eqFunction_8070(data, threadData);
  SpiralGalaxy_eqFunction_8069(data, threadData);
  SpiralGalaxy_eqFunction_8068(data, threadData);
  SpiralGalaxy_eqFunction_107(data, threadData);
  SpiralGalaxy_eqFunction_8063(data, threadData);
  SpiralGalaxy_eqFunction_109(data, threadData);
  SpiralGalaxy_eqFunction_8062(data, threadData);
  SpiralGalaxy_eqFunction_111(data, threadData);
  SpiralGalaxy_eqFunction_8061(data, threadData);
  SpiralGalaxy_eqFunction_113(data, threadData);
  SpiralGalaxy_eqFunction_8074(data, threadData);
  SpiralGalaxy_eqFunction_8075(data, threadData);
  SpiralGalaxy_eqFunction_116(data, threadData);
  SpiralGalaxy_eqFunction_117(data, threadData);
  SpiralGalaxy_eqFunction_8076(data, threadData);
  SpiralGalaxy_eqFunction_8077(data, threadData);
  SpiralGalaxy_eqFunction_8080(data, threadData);
  SpiralGalaxy_eqFunction_8079(data, threadData);
  SpiralGalaxy_eqFunction_8078(data, threadData);
  SpiralGalaxy_eqFunction_123(data, threadData);
  SpiralGalaxy_eqFunction_8073(data, threadData);
  SpiralGalaxy_eqFunction_125(data, threadData);
  SpiralGalaxy_eqFunction_8072(data, threadData);
  SpiralGalaxy_eqFunction_127(data, threadData);
  SpiralGalaxy_eqFunction_8071(data, threadData);
  SpiralGalaxy_eqFunction_129(data, threadData);
  SpiralGalaxy_eqFunction_8084(data, threadData);
  SpiralGalaxy_eqFunction_8085(data, threadData);
  SpiralGalaxy_eqFunction_132(data, threadData);
  SpiralGalaxy_eqFunction_133(data, threadData);
  SpiralGalaxy_eqFunction_8086(data, threadData);
  SpiralGalaxy_eqFunction_8087(data, threadData);
  SpiralGalaxy_eqFunction_8090(data, threadData);
  SpiralGalaxy_eqFunction_8089(data, threadData);
  SpiralGalaxy_eqFunction_8088(data, threadData);
  SpiralGalaxy_eqFunction_139(data, threadData);
  SpiralGalaxy_eqFunction_8083(data, threadData);
  SpiralGalaxy_eqFunction_141(data, threadData);
  SpiralGalaxy_eqFunction_8082(data, threadData);
  SpiralGalaxy_eqFunction_143(data, threadData);
  SpiralGalaxy_eqFunction_8081(data, threadData);
  SpiralGalaxy_eqFunction_145(data, threadData);
  SpiralGalaxy_eqFunction_8094(data, threadData);
  SpiralGalaxy_eqFunction_8095(data, threadData);
  SpiralGalaxy_eqFunction_148(data, threadData);
  SpiralGalaxy_eqFunction_149(data, threadData);
  SpiralGalaxy_eqFunction_8096(data, threadData);
  SpiralGalaxy_eqFunction_8097(data, threadData);
  SpiralGalaxy_eqFunction_8100(data, threadData);
  SpiralGalaxy_eqFunction_8099(data, threadData);
  SpiralGalaxy_eqFunction_8098(data, threadData);
  SpiralGalaxy_eqFunction_155(data, threadData);
  SpiralGalaxy_eqFunction_8093(data, threadData);
  SpiralGalaxy_eqFunction_157(data, threadData);
  SpiralGalaxy_eqFunction_8092(data, threadData);
  SpiralGalaxy_eqFunction_159(data, threadData);
  SpiralGalaxy_eqFunction_8091(data, threadData);
  SpiralGalaxy_eqFunction_161(data, threadData);
  SpiralGalaxy_eqFunction_8104(data, threadData);
  SpiralGalaxy_eqFunction_8105(data, threadData);
  SpiralGalaxy_eqFunction_164(data, threadData);
  SpiralGalaxy_eqFunction_165(data, threadData);
  SpiralGalaxy_eqFunction_8106(data, threadData);
  SpiralGalaxy_eqFunction_8107(data, threadData);
  SpiralGalaxy_eqFunction_8110(data, threadData);
  SpiralGalaxy_eqFunction_8109(data, threadData);
  SpiralGalaxy_eqFunction_8108(data, threadData);
  SpiralGalaxy_eqFunction_171(data, threadData);
  SpiralGalaxy_eqFunction_8103(data, threadData);
  SpiralGalaxy_eqFunction_173(data, threadData);
  SpiralGalaxy_eqFunction_8102(data, threadData);
  SpiralGalaxy_eqFunction_175(data, threadData);
  SpiralGalaxy_eqFunction_8101(data, threadData);
  SpiralGalaxy_eqFunction_177(data, threadData);
  SpiralGalaxy_eqFunction_8114(data, threadData);
  SpiralGalaxy_eqFunction_8115(data, threadData);
  SpiralGalaxy_eqFunction_180(data, threadData);
  SpiralGalaxy_eqFunction_181(data, threadData);
  SpiralGalaxy_eqFunction_8116(data, threadData);
  SpiralGalaxy_eqFunction_8117(data, threadData);
  SpiralGalaxy_eqFunction_8120(data, threadData);
  SpiralGalaxy_eqFunction_8119(data, threadData);
  SpiralGalaxy_eqFunction_8118(data, threadData);
  SpiralGalaxy_eqFunction_187(data, threadData);
  SpiralGalaxy_eqFunction_8113(data, threadData);
  SpiralGalaxy_eqFunction_189(data, threadData);
  SpiralGalaxy_eqFunction_8112(data, threadData);
  SpiralGalaxy_eqFunction_191(data, threadData);
  SpiralGalaxy_eqFunction_8111(data, threadData);
  SpiralGalaxy_eqFunction_193(data, threadData);
  SpiralGalaxy_eqFunction_8124(data, threadData);
  SpiralGalaxy_eqFunction_8125(data, threadData);
  SpiralGalaxy_eqFunction_196(data, threadData);
  SpiralGalaxy_eqFunction_197(data, threadData);
  SpiralGalaxy_eqFunction_8126(data, threadData);
  SpiralGalaxy_eqFunction_8127(data, threadData);
  SpiralGalaxy_eqFunction_8130(data, threadData);
  SpiralGalaxy_eqFunction_8129(data, threadData);
  SpiralGalaxy_eqFunction_8128(data, threadData);
  SpiralGalaxy_eqFunction_203(data, threadData);
  SpiralGalaxy_eqFunction_8123(data, threadData);
  SpiralGalaxy_eqFunction_205(data, threadData);
  SpiralGalaxy_eqFunction_8122(data, threadData);
  SpiralGalaxy_eqFunction_207(data, threadData);
  SpiralGalaxy_eqFunction_8121(data, threadData);
  SpiralGalaxy_eqFunction_209(data, threadData);
  SpiralGalaxy_eqFunction_8134(data, threadData);
  SpiralGalaxy_eqFunction_8135(data, threadData);
  SpiralGalaxy_eqFunction_212(data, threadData);
  SpiralGalaxy_eqFunction_213(data, threadData);
  SpiralGalaxy_eqFunction_8136(data, threadData);
  SpiralGalaxy_eqFunction_8137(data, threadData);
  SpiralGalaxy_eqFunction_8140(data, threadData);
  SpiralGalaxy_eqFunction_8139(data, threadData);
  SpiralGalaxy_eqFunction_8138(data, threadData);
  SpiralGalaxy_eqFunction_219(data, threadData);
  SpiralGalaxy_eqFunction_8133(data, threadData);
  SpiralGalaxy_eqFunction_221(data, threadData);
  SpiralGalaxy_eqFunction_8132(data, threadData);
  SpiralGalaxy_eqFunction_223(data, threadData);
  SpiralGalaxy_eqFunction_8131(data, threadData);
  SpiralGalaxy_eqFunction_225(data, threadData);
  SpiralGalaxy_eqFunction_8144(data, threadData);
  SpiralGalaxy_eqFunction_8145(data, threadData);
  SpiralGalaxy_eqFunction_228(data, threadData);
  SpiralGalaxy_eqFunction_229(data, threadData);
  SpiralGalaxy_eqFunction_8146(data, threadData);
  SpiralGalaxy_eqFunction_8147(data, threadData);
  SpiralGalaxy_eqFunction_8150(data, threadData);
  SpiralGalaxy_eqFunction_8149(data, threadData);
  SpiralGalaxy_eqFunction_8148(data, threadData);
  SpiralGalaxy_eqFunction_235(data, threadData);
  SpiralGalaxy_eqFunction_8143(data, threadData);
  SpiralGalaxy_eqFunction_237(data, threadData);
  SpiralGalaxy_eqFunction_8142(data, threadData);
  SpiralGalaxy_eqFunction_239(data, threadData);
  SpiralGalaxy_eqFunction_8141(data, threadData);
  SpiralGalaxy_eqFunction_241(data, threadData);
  SpiralGalaxy_eqFunction_8154(data, threadData);
  SpiralGalaxy_eqFunction_8155(data, threadData);
  SpiralGalaxy_eqFunction_244(data, threadData);
  SpiralGalaxy_eqFunction_245(data, threadData);
  SpiralGalaxy_eqFunction_8156(data, threadData);
  SpiralGalaxy_eqFunction_8157(data, threadData);
  SpiralGalaxy_eqFunction_8160(data, threadData);
  SpiralGalaxy_eqFunction_8159(data, threadData);
  SpiralGalaxy_eqFunction_8158(data, threadData);
  SpiralGalaxy_eqFunction_251(data, threadData);
  SpiralGalaxy_eqFunction_8153(data, threadData);
  SpiralGalaxy_eqFunction_253(data, threadData);
  SpiralGalaxy_eqFunction_8152(data, threadData);
  SpiralGalaxy_eqFunction_255(data, threadData);
  SpiralGalaxy_eqFunction_8151(data, threadData);
  SpiralGalaxy_eqFunction_257(data, threadData);
  SpiralGalaxy_eqFunction_8164(data, threadData);
  SpiralGalaxy_eqFunction_8165(data, threadData);
  SpiralGalaxy_eqFunction_260(data, threadData);
  SpiralGalaxy_eqFunction_261(data, threadData);
  SpiralGalaxy_eqFunction_8166(data, threadData);
  SpiralGalaxy_eqFunction_8167(data, threadData);
  SpiralGalaxy_eqFunction_8170(data, threadData);
  SpiralGalaxy_eqFunction_8169(data, threadData);
  SpiralGalaxy_eqFunction_8168(data, threadData);
  SpiralGalaxy_eqFunction_267(data, threadData);
  SpiralGalaxy_eqFunction_8163(data, threadData);
  SpiralGalaxy_eqFunction_269(data, threadData);
  SpiralGalaxy_eqFunction_8162(data, threadData);
  SpiralGalaxy_eqFunction_271(data, threadData);
  SpiralGalaxy_eqFunction_8161(data, threadData);
  SpiralGalaxy_eqFunction_273(data, threadData);
  SpiralGalaxy_eqFunction_8174(data, threadData);
  SpiralGalaxy_eqFunction_8175(data, threadData);
  SpiralGalaxy_eqFunction_276(data, threadData);
  SpiralGalaxy_eqFunction_277(data, threadData);
  SpiralGalaxy_eqFunction_8176(data, threadData);
  SpiralGalaxy_eqFunction_8177(data, threadData);
  SpiralGalaxy_eqFunction_8180(data, threadData);
  SpiralGalaxy_eqFunction_8179(data, threadData);
  SpiralGalaxy_eqFunction_8178(data, threadData);
  SpiralGalaxy_eqFunction_283(data, threadData);
  SpiralGalaxy_eqFunction_8173(data, threadData);
  SpiralGalaxy_eqFunction_285(data, threadData);
  SpiralGalaxy_eqFunction_8172(data, threadData);
  SpiralGalaxy_eqFunction_287(data, threadData);
  SpiralGalaxy_eqFunction_8171(data, threadData);
  SpiralGalaxy_eqFunction_289(data, threadData);
  SpiralGalaxy_eqFunction_8184(data, threadData);
  SpiralGalaxy_eqFunction_8185(data, threadData);
  SpiralGalaxy_eqFunction_292(data, threadData);
  SpiralGalaxy_eqFunction_293(data, threadData);
  SpiralGalaxy_eqFunction_8186(data, threadData);
  SpiralGalaxy_eqFunction_8187(data, threadData);
  SpiralGalaxy_eqFunction_8190(data, threadData);
  SpiralGalaxy_eqFunction_8189(data, threadData);
  SpiralGalaxy_eqFunction_8188(data, threadData);
  SpiralGalaxy_eqFunction_299(data, threadData);
  SpiralGalaxy_eqFunction_8183(data, threadData);
  SpiralGalaxy_eqFunction_301(data, threadData);
  SpiralGalaxy_eqFunction_8182(data, threadData);
  SpiralGalaxy_eqFunction_303(data, threadData);
  SpiralGalaxy_eqFunction_8181(data, threadData);
  SpiralGalaxy_eqFunction_305(data, threadData);
  SpiralGalaxy_eqFunction_8194(data, threadData);
  SpiralGalaxy_eqFunction_8195(data, threadData);
  SpiralGalaxy_eqFunction_308(data, threadData);
  SpiralGalaxy_eqFunction_309(data, threadData);
  SpiralGalaxy_eqFunction_8196(data, threadData);
  SpiralGalaxy_eqFunction_8197(data, threadData);
  SpiralGalaxy_eqFunction_8200(data, threadData);
  SpiralGalaxy_eqFunction_8199(data, threadData);
  SpiralGalaxy_eqFunction_8198(data, threadData);
  SpiralGalaxy_eqFunction_315(data, threadData);
  SpiralGalaxy_eqFunction_8193(data, threadData);
  SpiralGalaxy_eqFunction_317(data, threadData);
  SpiralGalaxy_eqFunction_8192(data, threadData);
  SpiralGalaxy_eqFunction_319(data, threadData);
  SpiralGalaxy_eqFunction_8191(data, threadData);
  SpiralGalaxy_eqFunction_321(data, threadData);
  SpiralGalaxy_eqFunction_8204(data, threadData);
  SpiralGalaxy_eqFunction_8205(data, threadData);
  SpiralGalaxy_eqFunction_324(data, threadData);
  SpiralGalaxy_eqFunction_325(data, threadData);
  SpiralGalaxy_eqFunction_8206(data, threadData);
  SpiralGalaxy_eqFunction_8207(data, threadData);
  SpiralGalaxy_eqFunction_8210(data, threadData);
  SpiralGalaxy_eqFunction_8209(data, threadData);
  SpiralGalaxy_eqFunction_8208(data, threadData);
  SpiralGalaxy_eqFunction_331(data, threadData);
  SpiralGalaxy_eqFunction_8203(data, threadData);
  SpiralGalaxy_eqFunction_333(data, threadData);
  SpiralGalaxy_eqFunction_8202(data, threadData);
  SpiralGalaxy_eqFunction_335(data, threadData);
  SpiralGalaxy_eqFunction_8201(data, threadData);
  SpiralGalaxy_eqFunction_337(data, threadData);
  SpiralGalaxy_eqFunction_8214(data, threadData);
  SpiralGalaxy_eqFunction_8215(data, threadData);
  SpiralGalaxy_eqFunction_340(data, threadData);
  SpiralGalaxy_eqFunction_341(data, threadData);
  SpiralGalaxy_eqFunction_8216(data, threadData);
  SpiralGalaxy_eqFunction_8217(data, threadData);
  SpiralGalaxy_eqFunction_8220(data, threadData);
  SpiralGalaxy_eqFunction_8219(data, threadData);
  SpiralGalaxy_eqFunction_8218(data, threadData);
  SpiralGalaxy_eqFunction_347(data, threadData);
  SpiralGalaxy_eqFunction_8213(data, threadData);
  SpiralGalaxy_eqFunction_349(data, threadData);
  SpiralGalaxy_eqFunction_8212(data, threadData);
  SpiralGalaxy_eqFunction_351(data, threadData);
  SpiralGalaxy_eqFunction_8211(data, threadData);
  SpiralGalaxy_eqFunction_353(data, threadData);
  SpiralGalaxy_eqFunction_8224(data, threadData);
  SpiralGalaxy_eqFunction_8225(data, threadData);
  SpiralGalaxy_eqFunction_356(data, threadData);
  SpiralGalaxy_eqFunction_357(data, threadData);
  SpiralGalaxy_eqFunction_8226(data, threadData);
  SpiralGalaxy_eqFunction_8227(data, threadData);
  SpiralGalaxy_eqFunction_8230(data, threadData);
  SpiralGalaxy_eqFunction_8229(data, threadData);
  SpiralGalaxy_eqFunction_8228(data, threadData);
  SpiralGalaxy_eqFunction_363(data, threadData);
  SpiralGalaxy_eqFunction_8223(data, threadData);
  SpiralGalaxy_eqFunction_365(data, threadData);
  SpiralGalaxy_eqFunction_8222(data, threadData);
  SpiralGalaxy_eqFunction_367(data, threadData);
  SpiralGalaxy_eqFunction_8221(data, threadData);
  SpiralGalaxy_eqFunction_369(data, threadData);
  SpiralGalaxy_eqFunction_8234(data, threadData);
  SpiralGalaxy_eqFunction_8235(data, threadData);
  SpiralGalaxy_eqFunction_372(data, threadData);
  SpiralGalaxy_eqFunction_373(data, threadData);
  SpiralGalaxy_eqFunction_8236(data, threadData);
  SpiralGalaxy_eqFunction_8237(data, threadData);
  SpiralGalaxy_eqFunction_8240(data, threadData);
  SpiralGalaxy_eqFunction_8239(data, threadData);
  SpiralGalaxy_eqFunction_8238(data, threadData);
  SpiralGalaxy_eqFunction_379(data, threadData);
  SpiralGalaxy_eqFunction_8233(data, threadData);
  SpiralGalaxy_eqFunction_381(data, threadData);
  SpiralGalaxy_eqFunction_8232(data, threadData);
  SpiralGalaxy_eqFunction_383(data, threadData);
  SpiralGalaxy_eqFunction_8231(data, threadData);
  SpiralGalaxy_eqFunction_385(data, threadData);
  SpiralGalaxy_eqFunction_8244(data, threadData);
  SpiralGalaxy_eqFunction_8245(data, threadData);
  SpiralGalaxy_eqFunction_388(data, threadData);
  SpiralGalaxy_eqFunction_389(data, threadData);
  SpiralGalaxy_eqFunction_8246(data, threadData);
  SpiralGalaxy_eqFunction_8247(data, threadData);
  SpiralGalaxy_eqFunction_8250(data, threadData);
  SpiralGalaxy_eqFunction_8249(data, threadData);
  SpiralGalaxy_eqFunction_8248(data, threadData);
  SpiralGalaxy_eqFunction_395(data, threadData);
  SpiralGalaxy_eqFunction_8243(data, threadData);
  SpiralGalaxy_eqFunction_397(data, threadData);
  SpiralGalaxy_eqFunction_8242(data, threadData);
  SpiralGalaxy_eqFunction_399(data, threadData);
  SpiralGalaxy_eqFunction_8241(data, threadData);
  SpiralGalaxy_eqFunction_401(data, threadData);
  SpiralGalaxy_eqFunction_8254(data, threadData);
  SpiralGalaxy_eqFunction_8255(data, threadData);
  SpiralGalaxy_eqFunction_404(data, threadData);
  SpiralGalaxy_eqFunction_405(data, threadData);
  SpiralGalaxy_eqFunction_8256(data, threadData);
  SpiralGalaxy_eqFunction_8257(data, threadData);
  SpiralGalaxy_eqFunction_8260(data, threadData);
  SpiralGalaxy_eqFunction_8259(data, threadData);
  SpiralGalaxy_eqFunction_8258(data, threadData);
  SpiralGalaxy_eqFunction_411(data, threadData);
  SpiralGalaxy_eqFunction_8253(data, threadData);
  SpiralGalaxy_eqFunction_413(data, threadData);
  SpiralGalaxy_eqFunction_8252(data, threadData);
  SpiralGalaxy_eqFunction_415(data, threadData);
  SpiralGalaxy_eqFunction_8251(data, threadData);
  SpiralGalaxy_eqFunction_417(data, threadData);
  SpiralGalaxy_eqFunction_8264(data, threadData);
  SpiralGalaxy_eqFunction_8265(data, threadData);
  SpiralGalaxy_eqFunction_420(data, threadData);
  SpiralGalaxy_eqFunction_421(data, threadData);
  SpiralGalaxy_eqFunction_8266(data, threadData);
  SpiralGalaxy_eqFunction_8267(data, threadData);
  SpiralGalaxy_eqFunction_8270(data, threadData);
  SpiralGalaxy_eqFunction_8269(data, threadData);
  SpiralGalaxy_eqFunction_8268(data, threadData);
  SpiralGalaxy_eqFunction_427(data, threadData);
  SpiralGalaxy_eqFunction_8263(data, threadData);
  SpiralGalaxy_eqFunction_429(data, threadData);
  SpiralGalaxy_eqFunction_8262(data, threadData);
  SpiralGalaxy_eqFunction_431(data, threadData);
  SpiralGalaxy_eqFunction_8261(data, threadData);
  SpiralGalaxy_eqFunction_433(data, threadData);
  SpiralGalaxy_eqFunction_8274(data, threadData);
  SpiralGalaxy_eqFunction_8275(data, threadData);
  SpiralGalaxy_eqFunction_436(data, threadData);
  SpiralGalaxy_eqFunction_437(data, threadData);
  SpiralGalaxy_eqFunction_8276(data, threadData);
  SpiralGalaxy_eqFunction_8277(data, threadData);
  SpiralGalaxy_eqFunction_8280(data, threadData);
  SpiralGalaxy_eqFunction_8279(data, threadData);
  SpiralGalaxy_eqFunction_8278(data, threadData);
  SpiralGalaxy_eqFunction_443(data, threadData);
  SpiralGalaxy_eqFunction_8273(data, threadData);
  SpiralGalaxy_eqFunction_445(data, threadData);
  SpiralGalaxy_eqFunction_8272(data, threadData);
  SpiralGalaxy_eqFunction_447(data, threadData);
  SpiralGalaxy_eqFunction_8271(data, threadData);
  SpiralGalaxy_eqFunction_449(data, threadData);
  SpiralGalaxy_eqFunction_8284(data, threadData);
  SpiralGalaxy_eqFunction_8285(data, threadData);
  SpiralGalaxy_eqFunction_452(data, threadData);
  SpiralGalaxy_eqFunction_453(data, threadData);
  SpiralGalaxy_eqFunction_8286(data, threadData);
  SpiralGalaxy_eqFunction_8287(data, threadData);
  SpiralGalaxy_eqFunction_8290(data, threadData);
  SpiralGalaxy_eqFunction_8289(data, threadData);
  SpiralGalaxy_eqFunction_8288(data, threadData);
  SpiralGalaxy_eqFunction_459(data, threadData);
  SpiralGalaxy_eqFunction_8283(data, threadData);
  SpiralGalaxy_eqFunction_461(data, threadData);
  SpiralGalaxy_eqFunction_8282(data, threadData);
  SpiralGalaxy_eqFunction_463(data, threadData);
  SpiralGalaxy_eqFunction_8281(data, threadData);
  SpiralGalaxy_eqFunction_465(data, threadData);
  SpiralGalaxy_eqFunction_8294(data, threadData);
  SpiralGalaxy_eqFunction_8295(data, threadData);
  SpiralGalaxy_eqFunction_468(data, threadData);
  SpiralGalaxy_eqFunction_469(data, threadData);
  SpiralGalaxy_eqFunction_8296(data, threadData);
  SpiralGalaxy_eqFunction_8297(data, threadData);
  SpiralGalaxy_eqFunction_8300(data, threadData);
  SpiralGalaxy_eqFunction_8299(data, threadData);
  SpiralGalaxy_eqFunction_8298(data, threadData);
  SpiralGalaxy_eqFunction_475(data, threadData);
  SpiralGalaxy_eqFunction_8293(data, threadData);
  SpiralGalaxy_eqFunction_477(data, threadData);
  SpiralGalaxy_eqFunction_8292(data, threadData);
  SpiralGalaxy_eqFunction_479(data, threadData);
  SpiralGalaxy_eqFunction_8291(data, threadData);
  SpiralGalaxy_eqFunction_481(data, threadData);
  SpiralGalaxy_eqFunction_8304(data, threadData);
  SpiralGalaxy_eqFunction_8305(data, threadData);
  SpiralGalaxy_eqFunction_484(data, threadData);
  SpiralGalaxy_eqFunction_485(data, threadData);
  SpiralGalaxy_eqFunction_8306(data, threadData);
  SpiralGalaxy_eqFunction_8307(data, threadData);
  SpiralGalaxy_eqFunction_8310(data, threadData);
  SpiralGalaxy_eqFunction_8309(data, threadData);
  SpiralGalaxy_eqFunction_8308(data, threadData);
  SpiralGalaxy_eqFunction_491(data, threadData);
  SpiralGalaxy_eqFunction_8303(data, threadData);
  SpiralGalaxy_eqFunction_493(data, threadData);
  SpiralGalaxy_eqFunction_8302(data, threadData);
  SpiralGalaxy_eqFunction_495(data, threadData);
  SpiralGalaxy_eqFunction_8301(data, threadData);
  SpiralGalaxy_eqFunction_497(data, threadData);
  SpiralGalaxy_eqFunction_8314(data, threadData);
  SpiralGalaxy_eqFunction_8315(data, threadData);
  SpiralGalaxy_eqFunction_500(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif