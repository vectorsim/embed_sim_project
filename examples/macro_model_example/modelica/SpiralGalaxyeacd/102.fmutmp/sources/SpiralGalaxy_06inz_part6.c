#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void SpiralGalaxy_eqFunction_9879(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9878(DATA *data, threadData_t *threadData);


/*
equation index: 3003
type: SIMPLE_ASSIGN
vx[188] = (-sin(theta[188])) * r_init[188] * omega_c[188]
*/
void SpiralGalaxy_eqFunction_3003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3003};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[187]] /* vx[188] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1694] /* theta[188] PARAM */)))) * (((data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */)) * ((data->simulationInfo->realParameter[692] /* omega_c[188] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9873(DATA *data, threadData_t *threadData);


/*
equation index: 3005
type: SIMPLE_ASSIGN
vy[188] = cos(theta[188]) * r_init[188] * omega_c[188]
*/
void SpiralGalaxy_eqFunction_3005(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3005};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[687]] /* vy[188] STATE(1) */) = (cos((data->simulationInfo->realParameter[1694] /* theta[188] PARAM */))) * (((data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */)) * ((data->simulationInfo->realParameter[692] /* omega_c[188] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9872(DATA *data, threadData_t *threadData);


/*
equation index: 3007
type: SIMPLE_ASSIGN
vz[188] = 0.0
*/
void SpiralGalaxy_eqFunction_3007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3007};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1187]] /* vz[188] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9871(DATA *data, threadData_t *threadData);


/*
equation index: 3009
type: SIMPLE_ASSIGN
z[189] = -0.009760000000000001
*/
void SpiralGalaxy_eqFunction_3009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3009};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2688]] /* z[189] STATE(1,vz[189]) */) = -0.009760000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9884(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9885(DATA *data, threadData_t *threadData);


/*
equation index: 3012
type: SIMPLE_ASSIGN
y[189] = r_init[189] * sin(theta[189] - 0.00244)
*/
void SpiralGalaxy_eqFunction_3012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3012};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2188]] /* y[189] STATE(1,vy[189]) */) = ((data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */)) * (sin((data->simulationInfo->realParameter[1695] /* theta[189] PARAM */) - 0.00244));
  TRACE_POP
}

/*
equation index: 3013
type: SIMPLE_ASSIGN
x[189] = r_init[189] * cos(theta[189] - 0.00244)
*/
void SpiralGalaxy_eqFunction_3013(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3013};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1688]] /* x[189] STATE(1,vx[189]) */) = ((data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */)) * (cos((data->simulationInfo->realParameter[1695] /* theta[189] PARAM */) - 0.00244));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9886(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9887(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9890(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9889(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9888(DATA *data, threadData_t *threadData);


/*
equation index: 3019
type: SIMPLE_ASSIGN
vx[189] = (-sin(theta[189])) * r_init[189] * omega_c[189]
*/
void SpiralGalaxy_eqFunction_3019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3019};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[188]] /* vx[189] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1695] /* theta[189] PARAM */)))) * (((data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */)) * ((data->simulationInfo->realParameter[693] /* omega_c[189] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9883(DATA *data, threadData_t *threadData);


/*
equation index: 3021
type: SIMPLE_ASSIGN
vy[189] = cos(theta[189]) * r_init[189] * omega_c[189]
*/
void SpiralGalaxy_eqFunction_3021(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3021};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[688]] /* vy[189] STATE(1) */) = (cos((data->simulationInfo->realParameter[1695] /* theta[189] PARAM */))) * (((data->simulationInfo->realParameter[1194] /* r_init[189] PARAM */)) * ((data->simulationInfo->realParameter[693] /* omega_c[189] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9882(DATA *data, threadData_t *threadData);


/*
equation index: 3023
type: SIMPLE_ASSIGN
vz[189] = 0.0
*/
void SpiralGalaxy_eqFunction_3023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3023};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1188]] /* vz[189] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9881(DATA *data, threadData_t *threadData);


/*
equation index: 3025
type: SIMPLE_ASSIGN
z[190] = -0.009600000000000001
*/
void SpiralGalaxy_eqFunction_3025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3025};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2689]] /* z[190] STATE(1,vz[190]) */) = -0.009600000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9894(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9895(DATA *data, threadData_t *threadData);


/*
equation index: 3028
type: SIMPLE_ASSIGN
y[190] = r_init[190] * sin(theta[190] - 0.0024)
*/
void SpiralGalaxy_eqFunction_3028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3028};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2189]] /* y[190] STATE(1,vy[190]) */) = ((data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */)) * (sin((data->simulationInfo->realParameter[1696] /* theta[190] PARAM */) - 0.0024));
  TRACE_POP
}

/*
equation index: 3029
type: SIMPLE_ASSIGN
x[190] = r_init[190] * cos(theta[190] - 0.0024)
*/
void SpiralGalaxy_eqFunction_3029(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3029};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1689]] /* x[190] STATE(1,vx[190]) */) = ((data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */)) * (cos((data->simulationInfo->realParameter[1696] /* theta[190] PARAM */) - 0.0024));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9896(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9897(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9900(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9899(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9898(DATA *data, threadData_t *threadData);


/*
equation index: 3035
type: SIMPLE_ASSIGN
vx[190] = (-sin(theta[190])) * r_init[190] * omega_c[190]
*/
void SpiralGalaxy_eqFunction_3035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3035};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[189]] /* vx[190] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1696] /* theta[190] PARAM */)))) * (((data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */)) * ((data->simulationInfo->realParameter[694] /* omega_c[190] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9893(DATA *data, threadData_t *threadData);


/*
equation index: 3037
type: SIMPLE_ASSIGN
vy[190] = cos(theta[190]) * r_init[190] * omega_c[190]
*/
void SpiralGalaxy_eqFunction_3037(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3037};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[689]] /* vy[190] STATE(1) */) = (cos((data->simulationInfo->realParameter[1696] /* theta[190] PARAM */))) * (((data->simulationInfo->realParameter[1195] /* r_init[190] PARAM */)) * ((data->simulationInfo->realParameter[694] /* omega_c[190] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9892(DATA *data, threadData_t *threadData);


/*
equation index: 3039
type: SIMPLE_ASSIGN
vz[190] = 0.0
*/
void SpiralGalaxy_eqFunction_3039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3039};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1189]] /* vz[190] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9891(DATA *data, threadData_t *threadData);


/*
equation index: 3041
type: SIMPLE_ASSIGN
z[191] = -0.009440000000000002
*/
void SpiralGalaxy_eqFunction_3041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3041};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2690]] /* z[191] STATE(1,vz[191]) */) = -0.009440000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9904(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9905(DATA *data, threadData_t *threadData);


/*
equation index: 3044
type: SIMPLE_ASSIGN
y[191] = r_init[191] * sin(theta[191] - 0.00236)
*/
void SpiralGalaxy_eqFunction_3044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3044};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2190]] /* y[191] STATE(1,vy[191]) */) = ((data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */)) * (sin((data->simulationInfo->realParameter[1697] /* theta[191] PARAM */) - 0.00236));
  TRACE_POP
}

/*
equation index: 3045
type: SIMPLE_ASSIGN
x[191] = r_init[191] * cos(theta[191] - 0.00236)
*/
void SpiralGalaxy_eqFunction_3045(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3045};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1690]] /* x[191] STATE(1,vx[191]) */) = ((data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */)) * (cos((data->simulationInfo->realParameter[1697] /* theta[191] PARAM */) - 0.00236));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9906(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9907(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9910(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9909(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9908(DATA *data, threadData_t *threadData);


/*
equation index: 3051
type: SIMPLE_ASSIGN
vx[191] = (-sin(theta[191])) * r_init[191] * omega_c[191]
*/
void SpiralGalaxy_eqFunction_3051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3051};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[190]] /* vx[191] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1697] /* theta[191] PARAM */)))) * (((data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */)) * ((data->simulationInfo->realParameter[695] /* omega_c[191] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9903(DATA *data, threadData_t *threadData);


/*
equation index: 3053
type: SIMPLE_ASSIGN
vy[191] = cos(theta[191]) * r_init[191] * omega_c[191]
*/
void SpiralGalaxy_eqFunction_3053(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3053};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[690]] /* vy[191] STATE(1) */) = (cos((data->simulationInfo->realParameter[1697] /* theta[191] PARAM */))) * (((data->simulationInfo->realParameter[1196] /* r_init[191] PARAM */)) * ((data->simulationInfo->realParameter[695] /* omega_c[191] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9902(DATA *data, threadData_t *threadData);


/*
equation index: 3055
type: SIMPLE_ASSIGN
vz[191] = 0.0
*/
void SpiralGalaxy_eqFunction_3055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3055};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1190]] /* vz[191] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9901(DATA *data, threadData_t *threadData);


/*
equation index: 3057
type: SIMPLE_ASSIGN
z[192] = -0.00928
*/
void SpiralGalaxy_eqFunction_3057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3057};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2691]] /* z[192] STATE(1,vz[192]) */) = -0.00928;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9914(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9915(DATA *data, threadData_t *threadData);


/*
equation index: 3060
type: SIMPLE_ASSIGN
y[192] = r_init[192] * sin(theta[192] - 0.00232)
*/
void SpiralGalaxy_eqFunction_3060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3060};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2191]] /* y[192] STATE(1,vy[192]) */) = ((data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */)) * (sin((data->simulationInfo->realParameter[1698] /* theta[192] PARAM */) - 0.00232));
  TRACE_POP
}

/*
equation index: 3061
type: SIMPLE_ASSIGN
x[192] = r_init[192] * cos(theta[192] - 0.00232)
*/
void SpiralGalaxy_eqFunction_3061(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3061};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1691]] /* x[192] STATE(1,vx[192]) */) = ((data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */)) * (cos((data->simulationInfo->realParameter[1698] /* theta[192] PARAM */) - 0.00232));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9916(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9917(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9920(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9919(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9918(DATA *data, threadData_t *threadData);


/*
equation index: 3067
type: SIMPLE_ASSIGN
vx[192] = (-sin(theta[192])) * r_init[192] * omega_c[192]
*/
void SpiralGalaxy_eqFunction_3067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3067};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[191]] /* vx[192] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1698] /* theta[192] PARAM */)))) * (((data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */)) * ((data->simulationInfo->realParameter[696] /* omega_c[192] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9913(DATA *data, threadData_t *threadData);


/*
equation index: 3069
type: SIMPLE_ASSIGN
vy[192] = cos(theta[192]) * r_init[192] * omega_c[192]
*/
void SpiralGalaxy_eqFunction_3069(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3069};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[691]] /* vy[192] STATE(1) */) = (cos((data->simulationInfo->realParameter[1698] /* theta[192] PARAM */))) * (((data->simulationInfo->realParameter[1197] /* r_init[192] PARAM */)) * ((data->simulationInfo->realParameter[696] /* omega_c[192] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9912(DATA *data, threadData_t *threadData);


/*
equation index: 3071
type: SIMPLE_ASSIGN
vz[192] = 0.0
*/
void SpiralGalaxy_eqFunction_3071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3071};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1191]] /* vz[192] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9911(DATA *data, threadData_t *threadData);


/*
equation index: 3073
type: SIMPLE_ASSIGN
z[193] = -0.009120000000000001
*/
void SpiralGalaxy_eqFunction_3073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3073};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2692]] /* z[193] STATE(1,vz[193]) */) = -0.009120000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9924(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9925(DATA *data, threadData_t *threadData);


/*
equation index: 3076
type: SIMPLE_ASSIGN
y[193] = r_init[193] * sin(theta[193] - 0.00228)
*/
void SpiralGalaxy_eqFunction_3076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3076};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2192]] /* y[193] STATE(1,vy[193]) */) = ((data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */)) * (sin((data->simulationInfo->realParameter[1699] /* theta[193] PARAM */) - 0.00228));
  TRACE_POP
}

/*
equation index: 3077
type: SIMPLE_ASSIGN
x[193] = r_init[193] * cos(theta[193] - 0.00228)
*/
void SpiralGalaxy_eqFunction_3077(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3077};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1692]] /* x[193] STATE(1,vx[193]) */) = ((data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */)) * (cos((data->simulationInfo->realParameter[1699] /* theta[193] PARAM */) - 0.00228));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9926(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9927(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9930(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9929(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9928(DATA *data, threadData_t *threadData);


/*
equation index: 3083
type: SIMPLE_ASSIGN
vx[193] = (-sin(theta[193])) * r_init[193] * omega_c[193]
*/
void SpiralGalaxy_eqFunction_3083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3083};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[192]] /* vx[193] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1699] /* theta[193] PARAM */)))) * (((data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */)) * ((data->simulationInfo->realParameter[697] /* omega_c[193] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9923(DATA *data, threadData_t *threadData);


/*
equation index: 3085
type: SIMPLE_ASSIGN
vy[193] = cos(theta[193]) * r_init[193] * omega_c[193]
*/
void SpiralGalaxy_eqFunction_3085(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3085};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[692]] /* vy[193] STATE(1) */) = (cos((data->simulationInfo->realParameter[1699] /* theta[193] PARAM */))) * (((data->simulationInfo->realParameter[1198] /* r_init[193] PARAM */)) * ((data->simulationInfo->realParameter[697] /* omega_c[193] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9922(DATA *data, threadData_t *threadData);


/*
equation index: 3087
type: SIMPLE_ASSIGN
vz[193] = 0.0
*/
void SpiralGalaxy_eqFunction_3087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3087};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1192]] /* vz[193] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9921(DATA *data, threadData_t *threadData);


/*
equation index: 3089
type: SIMPLE_ASSIGN
z[194] = -0.008960000000000001
*/
void SpiralGalaxy_eqFunction_3089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3089};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2693]] /* z[194] STATE(1,vz[194]) */) = -0.008960000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9934(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9935(DATA *data, threadData_t *threadData);


/*
equation index: 3092
type: SIMPLE_ASSIGN
y[194] = r_init[194] * sin(theta[194] - 0.00224)
*/
void SpiralGalaxy_eqFunction_3092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3092};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2193]] /* y[194] STATE(1,vy[194]) */) = ((data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */)) * (sin((data->simulationInfo->realParameter[1700] /* theta[194] PARAM */) - 0.00224));
  TRACE_POP
}

/*
equation index: 3093
type: SIMPLE_ASSIGN
x[194] = r_init[194] * cos(theta[194] - 0.00224)
*/
void SpiralGalaxy_eqFunction_3093(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3093};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1693]] /* x[194] STATE(1,vx[194]) */) = ((data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */)) * (cos((data->simulationInfo->realParameter[1700] /* theta[194] PARAM */) - 0.00224));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9936(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9937(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9940(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9939(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9938(DATA *data, threadData_t *threadData);


/*
equation index: 3099
type: SIMPLE_ASSIGN
vx[194] = (-sin(theta[194])) * r_init[194] * omega_c[194]
*/
void SpiralGalaxy_eqFunction_3099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3099};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[193]] /* vx[194] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1700] /* theta[194] PARAM */)))) * (((data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */)) * ((data->simulationInfo->realParameter[698] /* omega_c[194] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9933(DATA *data, threadData_t *threadData);


/*
equation index: 3101
type: SIMPLE_ASSIGN
vy[194] = cos(theta[194]) * r_init[194] * omega_c[194]
*/
void SpiralGalaxy_eqFunction_3101(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3101};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[693]] /* vy[194] STATE(1) */) = (cos((data->simulationInfo->realParameter[1700] /* theta[194] PARAM */))) * (((data->simulationInfo->realParameter[1199] /* r_init[194] PARAM */)) * ((data->simulationInfo->realParameter[698] /* omega_c[194] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9932(DATA *data, threadData_t *threadData);


/*
equation index: 3103
type: SIMPLE_ASSIGN
vz[194] = 0.0
*/
void SpiralGalaxy_eqFunction_3103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3103};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1193]] /* vz[194] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9931(DATA *data, threadData_t *threadData);


/*
equation index: 3105
type: SIMPLE_ASSIGN
z[195] = -0.0088
*/
void SpiralGalaxy_eqFunction_3105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3105};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2694]] /* z[195] STATE(1,vz[195]) */) = -0.0088;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9944(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9945(DATA *data, threadData_t *threadData);


/*
equation index: 3108
type: SIMPLE_ASSIGN
y[195] = r_init[195] * sin(theta[195] - 0.0021999999999999997)
*/
void SpiralGalaxy_eqFunction_3108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3108};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2194]] /* y[195] STATE(1,vy[195]) */) = ((data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */)) * (sin((data->simulationInfo->realParameter[1701] /* theta[195] PARAM */) - 0.0021999999999999997));
  TRACE_POP
}

/*
equation index: 3109
type: SIMPLE_ASSIGN
x[195] = r_init[195] * cos(theta[195] - 0.0021999999999999997)
*/
void SpiralGalaxy_eqFunction_3109(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3109};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1694]] /* x[195] STATE(1,vx[195]) */) = ((data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */)) * (cos((data->simulationInfo->realParameter[1701] /* theta[195] PARAM */) - 0.0021999999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9946(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9947(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9950(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9949(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9948(DATA *data, threadData_t *threadData);


/*
equation index: 3115
type: SIMPLE_ASSIGN
vx[195] = (-sin(theta[195])) * r_init[195] * omega_c[195]
*/
void SpiralGalaxy_eqFunction_3115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[194]] /* vx[195] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1701] /* theta[195] PARAM */)))) * (((data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */)) * ((data->simulationInfo->realParameter[699] /* omega_c[195] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9943(DATA *data, threadData_t *threadData);


/*
equation index: 3117
type: SIMPLE_ASSIGN
vy[195] = cos(theta[195]) * r_init[195] * omega_c[195]
*/
void SpiralGalaxy_eqFunction_3117(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3117};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[694]] /* vy[195] STATE(1) */) = (cos((data->simulationInfo->realParameter[1701] /* theta[195] PARAM */))) * (((data->simulationInfo->realParameter[1200] /* r_init[195] PARAM */)) * ((data->simulationInfo->realParameter[699] /* omega_c[195] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9942(DATA *data, threadData_t *threadData);


/*
equation index: 3119
type: SIMPLE_ASSIGN
vz[195] = 0.0
*/
void SpiralGalaxy_eqFunction_3119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3119};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1194]] /* vz[195] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9941(DATA *data, threadData_t *threadData);


/*
equation index: 3121
type: SIMPLE_ASSIGN
z[196] = -0.00864
*/
void SpiralGalaxy_eqFunction_3121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3121};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2695]] /* z[196] STATE(1,vz[196]) */) = -0.00864;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9954(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9955(DATA *data, threadData_t *threadData);


/*
equation index: 3124
type: SIMPLE_ASSIGN
y[196] = r_init[196] * sin(theta[196] - 0.0021599999999999996)
*/
void SpiralGalaxy_eqFunction_3124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3124};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2195]] /* y[196] STATE(1,vy[196]) */) = ((data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */)) * (sin((data->simulationInfo->realParameter[1702] /* theta[196] PARAM */) - 0.0021599999999999996));
  TRACE_POP
}

/*
equation index: 3125
type: SIMPLE_ASSIGN
x[196] = r_init[196] * cos(theta[196] - 0.0021599999999999996)
*/
void SpiralGalaxy_eqFunction_3125(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3125};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1695]] /* x[196] STATE(1,vx[196]) */) = ((data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */)) * (cos((data->simulationInfo->realParameter[1702] /* theta[196] PARAM */) - 0.0021599999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9956(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9957(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9960(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9959(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9958(DATA *data, threadData_t *threadData);


/*
equation index: 3131
type: SIMPLE_ASSIGN
vx[196] = (-sin(theta[196])) * r_init[196] * omega_c[196]
*/
void SpiralGalaxy_eqFunction_3131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[195]] /* vx[196] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1702] /* theta[196] PARAM */)))) * (((data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */)) * ((data->simulationInfo->realParameter[700] /* omega_c[196] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9953(DATA *data, threadData_t *threadData);


/*
equation index: 3133
type: SIMPLE_ASSIGN
vy[196] = cos(theta[196]) * r_init[196] * omega_c[196]
*/
void SpiralGalaxy_eqFunction_3133(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3133};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[695]] /* vy[196] STATE(1) */) = (cos((data->simulationInfo->realParameter[1702] /* theta[196] PARAM */))) * (((data->simulationInfo->realParameter[1201] /* r_init[196] PARAM */)) * ((data->simulationInfo->realParameter[700] /* omega_c[196] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9952(DATA *data, threadData_t *threadData);


/*
equation index: 3135
type: SIMPLE_ASSIGN
vz[196] = 0.0
*/
void SpiralGalaxy_eqFunction_3135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3135};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1195]] /* vz[196] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9951(DATA *data, threadData_t *threadData);


/*
equation index: 3137
type: SIMPLE_ASSIGN
z[197] = -0.008480000000000001
*/
void SpiralGalaxy_eqFunction_3137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3137};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2696]] /* z[197] STATE(1,vz[197]) */) = -0.008480000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9964(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9965(DATA *data, threadData_t *threadData);


/*
equation index: 3140
type: SIMPLE_ASSIGN
y[197] = r_init[197] * sin(theta[197] - 0.00212)
*/
void SpiralGalaxy_eqFunction_3140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3140};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2196]] /* y[197] STATE(1,vy[197]) */) = ((data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */)) * (sin((data->simulationInfo->realParameter[1703] /* theta[197] PARAM */) - 0.00212));
  TRACE_POP
}

/*
equation index: 3141
type: SIMPLE_ASSIGN
x[197] = r_init[197] * cos(theta[197] - 0.00212)
*/
void SpiralGalaxy_eqFunction_3141(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3141};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1696]] /* x[197] STATE(1,vx[197]) */) = ((data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */)) * (cos((data->simulationInfo->realParameter[1703] /* theta[197] PARAM */) - 0.00212));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9966(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9967(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9970(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9969(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9968(DATA *data, threadData_t *threadData);


/*
equation index: 3147
type: SIMPLE_ASSIGN
vx[197] = (-sin(theta[197])) * r_init[197] * omega_c[197]
*/
void SpiralGalaxy_eqFunction_3147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[196]] /* vx[197] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1703] /* theta[197] PARAM */)))) * (((data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */)) * ((data->simulationInfo->realParameter[701] /* omega_c[197] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9963(DATA *data, threadData_t *threadData);


/*
equation index: 3149
type: SIMPLE_ASSIGN
vy[197] = cos(theta[197]) * r_init[197] * omega_c[197]
*/
void SpiralGalaxy_eqFunction_3149(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3149};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[696]] /* vy[197] STATE(1) */) = (cos((data->simulationInfo->realParameter[1703] /* theta[197] PARAM */))) * (((data->simulationInfo->realParameter[1202] /* r_init[197] PARAM */)) * ((data->simulationInfo->realParameter[701] /* omega_c[197] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9962(DATA *data, threadData_t *threadData);


/*
equation index: 3151
type: SIMPLE_ASSIGN
vz[197] = 0.0
*/
void SpiralGalaxy_eqFunction_3151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3151};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1196]] /* vz[197] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9961(DATA *data, threadData_t *threadData);


/*
equation index: 3153
type: SIMPLE_ASSIGN
z[198] = -0.008320000000000001
*/
void SpiralGalaxy_eqFunction_3153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3153};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2697]] /* z[198] STATE(1,vz[198]) */) = -0.008320000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9974(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9975(DATA *data, threadData_t *threadData);


/*
equation index: 3156
type: SIMPLE_ASSIGN
y[198] = r_init[198] * sin(theta[198] - 0.00208)
*/
void SpiralGalaxy_eqFunction_3156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3156};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2197]] /* y[198] STATE(1,vy[198]) */) = ((data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */)) * (sin((data->simulationInfo->realParameter[1704] /* theta[198] PARAM */) - 0.00208));
  TRACE_POP
}

/*
equation index: 3157
type: SIMPLE_ASSIGN
x[198] = r_init[198] * cos(theta[198] - 0.00208)
*/
void SpiralGalaxy_eqFunction_3157(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3157};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1697]] /* x[198] STATE(1,vx[198]) */) = ((data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */)) * (cos((data->simulationInfo->realParameter[1704] /* theta[198] PARAM */) - 0.00208));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9976(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9977(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9980(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9979(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9978(DATA *data, threadData_t *threadData);


/*
equation index: 3163
type: SIMPLE_ASSIGN
vx[198] = (-sin(theta[198])) * r_init[198] * omega_c[198]
*/
void SpiralGalaxy_eqFunction_3163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[197]] /* vx[198] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1704] /* theta[198] PARAM */)))) * (((data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */)) * ((data->simulationInfo->realParameter[702] /* omega_c[198] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9973(DATA *data, threadData_t *threadData);


/*
equation index: 3165
type: SIMPLE_ASSIGN
vy[198] = cos(theta[198]) * r_init[198] * omega_c[198]
*/
void SpiralGalaxy_eqFunction_3165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3165};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[697]] /* vy[198] STATE(1) */) = (cos((data->simulationInfo->realParameter[1704] /* theta[198] PARAM */))) * (((data->simulationInfo->realParameter[1203] /* r_init[198] PARAM */)) * ((data->simulationInfo->realParameter[702] /* omega_c[198] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9972(DATA *data, threadData_t *threadData);


/*
equation index: 3167
type: SIMPLE_ASSIGN
vz[198] = 0.0
*/
void SpiralGalaxy_eqFunction_3167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3167};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1197]] /* vz[198] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9971(DATA *data, threadData_t *threadData);


/*
equation index: 3169
type: SIMPLE_ASSIGN
z[199] = -0.00816
*/
void SpiralGalaxy_eqFunction_3169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3169};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2698]] /* z[199] STATE(1,vz[199]) */) = -0.00816;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9984(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9985(DATA *data, threadData_t *threadData);


/*
equation index: 3172
type: SIMPLE_ASSIGN
y[199] = r_init[199] * sin(theta[199] - 0.0020399999999999997)
*/
void SpiralGalaxy_eqFunction_3172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3172};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2198]] /* y[199] STATE(1,vy[199]) */) = ((data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */)) * (sin((data->simulationInfo->realParameter[1705] /* theta[199] PARAM */) - 0.0020399999999999997));
  TRACE_POP
}

/*
equation index: 3173
type: SIMPLE_ASSIGN
x[199] = r_init[199] * cos(theta[199] - 0.0020399999999999997)
*/
void SpiralGalaxy_eqFunction_3173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3173};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1698]] /* x[199] STATE(1,vx[199]) */) = ((data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */)) * (cos((data->simulationInfo->realParameter[1705] /* theta[199] PARAM */) - 0.0020399999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9986(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9987(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9990(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9989(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9988(DATA *data, threadData_t *threadData);


/*
equation index: 3179
type: SIMPLE_ASSIGN
vx[199] = (-sin(theta[199])) * r_init[199] * omega_c[199]
*/
void SpiralGalaxy_eqFunction_3179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[198]] /* vx[199] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1705] /* theta[199] PARAM */)))) * (((data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */)) * ((data->simulationInfo->realParameter[703] /* omega_c[199] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9983(DATA *data, threadData_t *threadData);


/*
equation index: 3181
type: SIMPLE_ASSIGN
vy[199] = cos(theta[199]) * r_init[199] * omega_c[199]
*/
void SpiralGalaxy_eqFunction_3181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3181};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[698]] /* vy[199] STATE(1) */) = (cos((data->simulationInfo->realParameter[1705] /* theta[199] PARAM */))) * (((data->simulationInfo->realParameter[1204] /* r_init[199] PARAM */)) * ((data->simulationInfo->realParameter[703] /* omega_c[199] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9982(DATA *data, threadData_t *threadData);


/*
equation index: 3183
type: SIMPLE_ASSIGN
vz[199] = 0.0
*/
void SpiralGalaxy_eqFunction_3183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3183};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1198]] /* vz[199] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9981(DATA *data, threadData_t *threadData);


/*
equation index: 3185
type: SIMPLE_ASSIGN
z[200] = -0.008
*/
void SpiralGalaxy_eqFunction_3185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3185};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2699]] /* z[200] STATE(1,vz[200]) */) = -0.008;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9994(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9995(DATA *data, threadData_t *threadData);


/*
equation index: 3188
type: SIMPLE_ASSIGN
y[200] = r_init[200] * sin(theta[200] - 0.0019999999999999996)
*/
void SpiralGalaxy_eqFunction_3188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3188};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2199]] /* y[200] STATE(1,vy[200]) */) = ((data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */)) * (sin((data->simulationInfo->realParameter[1706] /* theta[200] PARAM */) - 0.0019999999999999996));
  TRACE_POP
}

/*
equation index: 3189
type: SIMPLE_ASSIGN
x[200] = r_init[200] * cos(theta[200] - 0.0019999999999999996)
*/
void SpiralGalaxy_eqFunction_3189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3189};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1699]] /* x[200] STATE(1,vx[200]) */) = ((data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */)) * (cos((data->simulationInfo->realParameter[1706] /* theta[200] PARAM */) - 0.0019999999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9996(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9997(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10000(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9999(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9998(DATA *data, threadData_t *threadData);


/*
equation index: 3195
type: SIMPLE_ASSIGN
vx[200] = (-sin(theta[200])) * r_init[200] * omega_c[200]
*/
void SpiralGalaxy_eqFunction_3195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[199]] /* vx[200] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1706] /* theta[200] PARAM */)))) * (((data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */)) * ((data->simulationInfo->realParameter[704] /* omega_c[200] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9993(DATA *data, threadData_t *threadData);


/*
equation index: 3197
type: SIMPLE_ASSIGN
vy[200] = cos(theta[200]) * r_init[200] * omega_c[200]
*/
void SpiralGalaxy_eqFunction_3197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3197};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[699]] /* vy[200] STATE(1) */) = (cos((data->simulationInfo->realParameter[1706] /* theta[200] PARAM */))) * (((data->simulationInfo->realParameter[1205] /* r_init[200] PARAM */)) * ((data->simulationInfo->realParameter[704] /* omega_c[200] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9992(DATA *data, threadData_t *threadData);


/*
equation index: 3199
type: SIMPLE_ASSIGN
vz[200] = 0.0
*/
void SpiralGalaxy_eqFunction_3199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3199};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1199]] /* vz[200] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9991(DATA *data, threadData_t *threadData);


/*
equation index: 3201
type: SIMPLE_ASSIGN
z[201] = -0.00784
*/
void SpiralGalaxy_eqFunction_3201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3201};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2700]] /* z[201] STATE(1,vz[201]) */) = -0.00784;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10004(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10005(DATA *data, threadData_t *threadData);


/*
equation index: 3204
type: SIMPLE_ASSIGN
y[201] = r_init[201] * sin(theta[201] - 0.0019599999999999995)
*/
void SpiralGalaxy_eqFunction_3204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3204};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2200]] /* y[201] STATE(1,vy[201]) */) = ((data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */)) * (sin((data->simulationInfo->realParameter[1707] /* theta[201] PARAM */) - 0.0019599999999999995));
  TRACE_POP
}

/*
equation index: 3205
type: SIMPLE_ASSIGN
x[201] = r_init[201] * cos(theta[201] - 0.0019599999999999995)
*/
void SpiralGalaxy_eqFunction_3205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3205};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1700]] /* x[201] STATE(1,vx[201]) */) = ((data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */)) * (cos((data->simulationInfo->realParameter[1707] /* theta[201] PARAM */) - 0.0019599999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10006(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10007(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10010(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10009(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10008(DATA *data, threadData_t *threadData);


/*
equation index: 3211
type: SIMPLE_ASSIGN
vx[201] = (-sin(theta[201])) * r_init[201] * omega_c[201]
*/
void SpiralGalaxy_eqFunction_3211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[200]] /* vx[201] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1707] /* theta[201] PARAM */)))) * (((data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */)) * ((data->simulationInfo->realParameter[705] /* omega_c[201] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10003(DATA *data, threadData_t *threadData);


/*
equation index: 3213
type: SIMPLE_ASSIGN
vy[201] = cos(theta[201]) * r_init[201] * omega_c[201]
*/
void SpiralGalaxy_eqFunction_3213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3213};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[700]] /* vy[201] STATE(1) */) = (cos((data->simulationInfo->realParameter[1707] /* theta[201] PARAM */))) * (((data->simulationInfo->realParameter[1206] /* r_init[201] PARAM */)) * ((data->simulationInfo->realParameter[705] /* omega_c[201] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10002(DATA *data, threadData_t *threadData);


/*
equation index: 3215
type: SIMPLE_ASSIGN
vz[201] = 0.0
*/
void SpiralGalaxy_eqFunction_3215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3215};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1200]] /* vz[201] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10001(DATA *data, threadData_t *threadData);


/*
equation index: 3217
type: SIMPLE_ASSIGN
z[202] = -0.00768
*/
void SpiralGalaxy_eqFunction_3217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3217};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2701]] /* z[202] STATE(1,vz[202]) */) = -0.00768;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10014(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10015(DATA *data, threadData_t *threadData);


/*
equation index: 3220
type: SIMPLE_ASSIGN
y[202] = r_init[202] * sin(theta[202] - 0.0019199999999999996)
*/
void SpiralGalaxy_eqFunction_3220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3220};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2201]] /* y[202] STATE(1,vy[202]) */) = ((data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */)) * (sin((data->simulationInfo->realParameter[1708] /* theta[202] PARAM */) - 0.0019199999999999996));
  TRACE_POP
}

/*
equation index: 3221
type: SIMPLE_ASSIGN
x[202] = r_init[202] * cos(theta[202] - 0.0019199999999999996)
*/
void SpiralGalaxy_eqFunction_3221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3221};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1701]] /* x[202] STATE(1,vx[202]) */) = ((data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */)) * (cos((data->simulationInfo->realParameter[1708] /* theta[202] PARAM */) - 0.0019199999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10016(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10017(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10020(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10019(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10018(DATA *data, threadData_t *threadData);


/*
equation index: 3227
type: SIMPLE_ASSIGN
vx[202] = (-sin(theta[202])) * r_init[202] * omega_c[202]
*/
void SpiralGalaxy_eqFunction_3227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[201]] /* vx[202] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1708] /* theta[202] PARAM */)))) * (((data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */)) * ((data->simulationInfo->realParameter[706] /* omega_c[202] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10013(DATA *data, threadData_t *threadData);


/*
equation index: 3229
type: SIMPLE_ASSIGN
vy[202] = cos(theta[202]) * r_init[202] * omega_c[202]
*/
void SpiralGalaxy_eqFunction_3229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3229};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[701]] /* vy[202] STATE(1) */) = (cos((data->simulationInfo->realParameter[1708] /* theta[202] PARAM */))) * (((data->simulationInfo->realParameter[1207] /* r_init[202] PARAM */)) * ((data->simulationInfo->realParameter[706] /* omega_c[202] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10012(DATA *data, threadData_t *threadData);


/*
equation index: 3231
type: SIMPLE_ASSIGN
vz[202] = 0.0
*/
void SpiralGalaxy_eqFunction_3231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3231};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1201]] /* vz[202] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10011(DATA *data, threadData_t *threadData);


/*
equation index: 3233
type: SIMPLE_ASSIGN
z[203] = -0.007520000000000001
*/
void SpiralGalaxy_eqFunction_3233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3233};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2702]] /* z[203] STATE(1,vz[203]) */) = -0.007520000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10024(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10025(DATA *data, threadData_t *threadData);


/*
equation index: 3236
type: SIMPLE_ASSIGN
y[203] = r_init[203] * sin(theta[203] - 0.0018799999999999995)
*/
void SpiralGalaxy_eqFunction_3236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3236};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2202]] /* y[203] STATE(1,vy[203]) */) = ((data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */)) * (sin((data->simulationInfo->realParameter[1709] /* theta[203] PARAM */) - 0.0018799999999999995));
  TRACE_POP
}

/*
equation index: 3237
type: SIMPLE_ASSIGN
x[203] = r_init[203] * cos(theta[203] - 0.0018799999999999995)
*/
void SpiralGalaxy_eqFunction_3237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3237};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1702]] /* x[203] STATE(1,vx[203]) */) = ((data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */)) * (cos((data->simulationInfo->realParameter[1709] /* theta[203] PARAM */) - 0.0018799999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10026(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10027(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10030(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10029(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10028(DATA *data, threadData_t *threadData);


/*
equation index: 3243
type: SIMPLE_ASSIGN
vx[203] = (-sin(theta[203])) * r_init[203] * omega_c[203]
*/
void SpiralGalaxy_eqFunction_3243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[202]] /* vx[203] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1709] /* theta[203] PARAM */)))) * (((data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */)) * ((data->simulationInfo->realParameter[707] /* omega_c[203] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10023(DATA *data, threadData_t *threadData);


/*
equation index: 3245
type: SIMPLE_ASSIGN
vy[203] = cos(theta[203]) * r_init[203] * omega_c[203]
*/
void SpiralGalaxy_eqFunction_3245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3245};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[702]] /* vy[203] STATE(1) */) = (cos((data->simulationInfo->realParameter[1709] /* theta[203] PARAM */))) * (((data->simulationInfo->realParameter[1208] /* r_init[203] PARAM */)) * ((data->simulationInfo->realParameter[707] /* omega_c[203] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10022(DATA *data, threadData_t *threadData);


/*
equation index: 3247
type: SIMPLE_ASSIGN
vz[203] = 0.0
*/
void SpiralGalaxy_eqFunction_3247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3247};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1202]] /* vz[203] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10021(DATA *data, threadData_t *threadData);


/*
equation index: 3249
type: SIMPLE_ASSIGN
z[204] = -0.00736
*/
void SpiralGalaxy_eqFunction_3249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3249};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2703]] /* z[204] STATE(1,vz[204]) */) = -0.00736;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10034(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10035(DATA *data, threadData_t *threadData);


/*
equation index: 3252
type: SIMPLE_ASSIGN
y[204] = r_init[204] * sin(theta[204] - 0.0018400000000000005)
*/
void SpiralGalaxy_eqFunction_3252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3252};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2203]] /* y[204] STATE(1,vy[204]) */) = ((data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */)) * (sin((data->simulationInfo->realParameter[1710] /* theta[204] PARAM */) - 0.0018400000000000005));
  TRACE_POP
}

/*
equation index: 3253
type: SIMPLE_ASSIGN
x[204] = r_init[204] * cos(theta[204] - 0.0018400000000000005)
*/
void SpiralGalaxy_eqFunction_3253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3253};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1703]] /* x[204] STATE(1,vx[204]) */) = ((data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */)) * (cos((data->simulationInfo->realParameter[1710] /* theta[204] PARAM */) - 0.0018400000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10036(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10037(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10040(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10039(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10038(DATA *data, threadData_t *threadData);


/*
equation index: 3259
type: SIMPLE_ASSIGN
vx[204] = (-sin(theta[204])) * r_init[204] * omega_c[204]
*/
void SpiralGalaxy_eqFunction_3259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[203]] /* vx[204] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1710] /* theta[204] PARAM */)))) * (((data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */)) * ((data->simulationInfo->realParameter[708] /* omega_c[204] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10033(DATA *data, threadData_t *threadData);


/*
equation index: 3261
type: SIMPLE_ASSIGN
vy[204] = cos(theta[204]) * r_init[204] * omega_c[204]
*/
void SpiralGalaxy_eqFunction_3261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3261};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[703]] /* vy[204] STATE(1) */) = (cos((data->simulationInfo->realParameter[1710] /* theta[204] PARAM */))) * (((data->simulationInfo->realParameter[1209] /* r_init[204] PARAM */)) * ((data->simulationInfo->realParameter[708] /* omega_c[204] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10032(DATA *data, threadData_t *threadData);


/*
equation index: 3263
type: SIMPLE_ASSIGN
vz[204] = 0.0
*/
void SpiralGalaxy_eqFunction_3263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3263};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1203]] /* vz[204] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10031(DATA *data, threadData_t *threadData);


/*
equation index: 3265
type: SIMPLE_ASSIGN
z[205] = -0.0072000000000000015
*/
void SpiralGalaxy_eqFunction_3265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3265};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2704]] /* z[205] STATE(1,vz[205]) */) = -0.0072000000000000015;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10044(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10045(DATA *data, threadData_t *threadData);


/*
equation index: 3268
type: SIMPLE_ASSIGN
y[205] = r_init[205] * sin(theta[205] - 0.0018000000000000006)
*/
void SpiralGalaxy_eqFunction_3268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3268};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2204]] /* y[205] STATE(1,vy[205]) */) = ((data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */)) * (sin((data->simulationInfo->realParameter[1711] /* theta[205] PARAM */) - 0.0018000000000000006));
  TRACE_POP
}

/*
equation index: 3269
type: SIMPLE_ASSIGN
x[205] = r_init[205] * cos(theta[205] - 0.0018000000000000006)
*/
void SpiralGalaxy_eqFunction_3269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3269};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1704]] /* x[205] STATE(1,vx[205]) */) = ((data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */)) * (cos((data->simulationInfo->realParameter[1711] /* theta[205] PARAM */) - 0.0018000000000000006));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10046(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10047(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10050(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10049(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10048(DATA *data, threadData_t *threadData);


/*
equation index: 3275
type: SIMPLE_ASSIGN
vx[205] = (-sin(theta[205])) * r_init[205] * omega_c[205]
*/
void SpiralGalaxy_eqFunction_3275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[204]] /* vx[205] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1711] /* theta[205] PARAM */)))) * (((data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */)) * ((data->simulationInfo->realParameter[709] /* omega_c[205] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10043(DATA *data, threadData_t *threadData);


/*
equation index: 3277
type: SIMPLE_ASSIGN
vy[205] = cos(theta[205]) * r_init[205] * omega_c[205]
*/
void SpiralGalaxy_eqFunction_3277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[704]] /* vy[205] STATE(1) */) = (cos((data->simulationInfo->realParameter[1711] /* theta[205] PARAM */))) * (((data->simulationInfo->realParameter[1210] /* r_init[205] PARAM */)) * ((data->simulationInfo->realParameter[709] /* omega_c[205] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10042(DATA *data, threadData_t *threadData);


/*
equation index: 3279
type: SIMPLE_ASSIGN
vz[205] = 0.0
*/
void SpiralGalaxy_eqFunction_3279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3279};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1204]] /* vz[205] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10041(DATA *data, threadData_t *threadData);


/*
equation index: 3281
type: SIMPLE_ASSIGN
z[206] = -0.007040000000000001
*/
void SpiralGalaxy_eqFunction_3281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3281};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2705]] /* z[206] STATE(1,vz[206]) */) = -0.007040000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10054(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10055(DATA *data, threadData_t *threadData);


/*
equation index: 3284
type: SIMPLE_ASSIGN
y[206] = r_init[206] * sin(theta[206] - 0.0017600000000000005)
*/
void SpiralGalaxy_eqFunction_3284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3284};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2205]] /* y[206] STATE(1,vy[206]) */) = ((data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */)) * (sin((data->simulationInfo->realParameter[1712] /* theta[206] PARAM */) - 0.0017600000000000005));
  TRACE_POP
}

/*
equation index: 3285
type: SIMPLE_ASSIGN
x[206] = r_init[206] * cos(theta[206] - 0.0017600000000000005)
*/
void SpiralGalaxy_eqFunction_3285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1705]] /* x[206] STATE(1,vx[206]) */) = ((data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */)) * (cos((data->simulationInfo->realParameter[1712] /* theta[206] PARAM */) - 0.0017600000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10056(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10057(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10060(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10059(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10058(DATA *data, threadData_t *threadData);


/*
equation index: 3291
type: SIMPLE_ASSIGN
vx[206] = (-sin(theta[206])) * r_init[206] * omega_c[206]
*/
void SpiralGalaxy_eqFunction_3291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[205]] /* vx[206] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1712] /* theta[206] PARAM */)))) * (((data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */)) * ((data->simulationInfo->realParameter[710] /* omega_c[206] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10053(DATA *data, threadData_t *threadData);


/*
equation index: 3293
type: SIMPLE_ASSIGN
vy[206] = cos(theta[206]) * r_init[206] * omega_c[206]
*/
void SpiralGalaxy_eqFunction_3293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[705]] /* vy[206] STATE(1) */) = (cos((data->simulationInfo->realParameter[1712] /* theta[206] PARAM */))) * (((data->simulationInfo->realParameter[1211] /* r_init[206] PARAM */)) * ((data->simulationInfo->realParameter[710] /* omega_c[206] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10052(DATA *data, threadData_t *threadData);


/*
equation index: 3295
type: SIMPLE_ASSIGN
vz[206] = 0.0
*/
void SpiralGalaxy_eqFunction_3295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1205]] /* vz[206] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10051(DATA *data, threadData_t *threadData);


/*
equation index: 3297
type: SIMPLE_ASSIGN
z[207] = -0.006880000000000001
*/
void SpiralGalaxy_eqFunction_3297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2706]] /* z[207] STATE(1,vz[207]) */) = -0.006880000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10064(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10065(DATA *data, threadData_t *threadData);


/*
equation index: 3300
type: SIMPLE_ASSIGN
y[207] = r_init[207] * sin(theta[207] - 0.0017200000000000004)
*/
void SpiralGalaxy_eqFunction_3300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3300};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2206]] /* y[207] STATE(1,vy[207]) */) = ((data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */)) * (sin((data->simulationInfo->realParameter[1713] /* theta[207] PARAM */) - 0.0017200000000000004));
  TRACE_POP
}

/*
equation index: 3301
type: SIMPLE_ASSIGN
x[207] = r_init[207] * cos(theta[207] - 0.0017200000000000004)
*/
void SpiralGalaxy_eqFunction_3301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3301};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1706]] /* x[207] STATE(1,vx[207]) */) = ((data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */)) * (cos((data->simulationInfo->realParameter[1713] /* theta[207] PARAM */) - 0.0017200000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10066(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10067(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10070(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10069(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10068(DATA *data, threadData_t *threadData);


/*
equation index: 3307
type: SIMPLE_ASSIGN
vx[207] = (-sin(theta[207])) * r_init[207] * omega_c[207]
*/
void SpiralGalaxy_eqFunction_3307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[206]] /* vx[207] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1713] /* theta[207] PARAM */)))) * (((data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */)) * ((data->simulationInfo->realParameter[711] /* omega_c[207] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10063(DATA *data, threadData_t *threadData);


/*
equation index: 3309
type: SIMPLE_ASSIGN
vy[207] = cos(theta[207]) * r_init[207] * omega_c[207]
*/
void SpiralGalaxy_eqFunction_3309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3309};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[706]] /* vy[207] STATE(1) */) = (cos((data->simulationInfo->realParameter[1713] /* theta[207] PARAM */))) * (((data->simulationInfo->realParameter[1212] /* r_init[207] PARAM */)) * ((data->simulationInfo->realParameter[711] /* omega_c[207] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10062(DATA *data, threadData_t *threadData);


/*
equation index: 3311
type: SIMPLE_ASSIGN
vz[207] = 0.0
*/
void SpiralGalaxy_eqFunction_3311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1206]] /* vz[207] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10061(DATA *data, threadData_t *threadData);


/*
equation index: 3313
type: SIMPLE_ASSIGN
z[208] = -0.006720000000000001
*/
void SpiralGalaxy_eqFunction_3313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2707]] /* z[208] STATE(1,vz[208]) */) = -0.006720000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10074(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10075(DATA *data, threadData_t *threadData);


/*
equation index: 3316
type: SIMPLE_ASSIGN
y[208] = r_init[208] * sin(theta[208] - 0.0016800000000000005)
*/
void SpiralGalaxy_eqFunction_3316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3316};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2207]] /* y[208] STATE(1,vy[208]) */) = ((data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */)) * (sin((data->simulationInfo->realParameter[1714] /* theta[208] PARAM */) - 0.0016800000000000005));
  TRACE_POP
}

/*
equation index: 3317
type: SIMPLE_ASSIGN
x[208] = r_init[208] * cos(theta[208] - 0.0016800000000000005)
*/
void SpiralGalaxy_eqFunction_3317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1707]] /* x[208] STATE(1,vx[208]) */) = ((data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */)) * (cos((data->simulationInfo->realParameter[1714] /* theta[208] PARAM */) - 0.0016800000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10076(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10077(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10080(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10079(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10078(DATA *data, threadData_t *threadData);


/*
equation index: 3323
type: SIMPLE_ASSIGN
vx[208] = (-sin(theta[208])) * r_init[208] * omega_c[208]
*/
void SpiralGalaxy_eqFunction_3323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[207]] /* vx[208] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1714] /* theta[208] PARAM */)))) * (((data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */)) * ((data->simulationInfo->realParameter[712] /* omega_c[208] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10073(DATA *data, threadData_t *threadData);


/*
equation index: 3325
type: SIMPLE_ASSIGN
vy[208] = cos(theta[208]) * r_init[208] * omega_c[208]
*/
void SpiralGalaxy_eqFunction_3325(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[707]] /* vy[208] STATE(1) */) = (cos((data->simulationInfo->realParameter[1714] /* theta[208] PARAM */))) * (((data->simulationInfo->realParameter[1213] /* r_init[208] PARAM */)) * ((data->simulationInfo->realParameter[712] /* omega_c[208] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10072(DATA *data, threadData_t *threadData);


/*
equation index: 3327
type: SIMPLE_ASSIGN
vz[208] = 0.0
*/
void SpiralGalaxy_eqFunction_3327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3327};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1207]] /* vz[208] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10071(DATA *data, threadData_t *threadData);


/*
equation index: 3329
type: SIMPLE_ASSIGN
z[209] = -0.006560000000000001
*/
void SpiralGalaxy_eqFunction_3329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3329};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2708]] /* z[209] STATE(1,vz[209]) */) = -0.006560000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10084(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10085(DATA *data, threadData_t *threadData);


/*
equation index: 3332
type: SIMPLE_ASSIGN
y[209] = r_init[209] * sin(theta[209] - 0.0016400000000000004)
*/
void SpiralGalaxy_eqFunction_3332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3332};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2208]] /* y[209] STATE(1,vy[209]) */) = ((data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */)) * (sin((data->simulationInfo->realParameter[1715] /* theta[209] PARAM */) - 0.0016400000000000004));
  TRACE_POP
}

/*
equation index: 3333
type: SIMPLE_ASSIGN
x[209] = r_init[209] * cos(theta[209] - 0.0016400000000000004)
*/
void SpiralGalaxy_eqFunction_3333(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1708]] /* x[209] STATE(1,vx[209]) */) = ((data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */)) * (cos((data->simulationInfo->realParameter[1715] /* theta[209] PARAM */) - 0.0016400000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10086(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10087(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10090(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10089(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10088(DATA *data, threadData_t *threadData);


/*
equation index: 3339
type: SIMPLE_ASSIGN
vx[209] = (-sin(theta[209])) * r_init[209] * omega_c[209]
*/
void SpiralGalaxy_eqFunction_3339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[208]] /* vx[209] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1715] /* theta[209] PARAM */)))) * (((data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */)) * ((data->simulationInfo->realParameter[713] /* omega_c[209] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10083(DATA *data, threadData_t *threadData);


/*
equation index: 3341
type: SIMPLE_ASSIGN
vy[209] = cos(theta[209]) * r_init[209] * omega_c[209]
*/
void SpiralGalaxy_eqFunction_3341(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[708]] /* vy[209] STATE(1) */) = (cos((data->simulationInfo->realParameter[1715] /* theta[209] PARAM */))) * (((data->simulationInfo->realParameter[1214] /* r_init[209] PARAM */)) * ((data->simulationInfo->realParameter[713] /* omega_c[209] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10082(DATA *data, threadData_t *threadData);


/*
equation index: 3343
type: SIMPLE_ASSIGN
vz[209] = 0.0
*/
void SpiralGalaxy_eqFunction_3343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1208]] /* vz[209] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10081(DATA *data, threadData_t *threadData);


/*
equation index: 3345
type: SIMPLE_ASSIGN
z[210] = -0.006400000000000001
*/
void SpiralGalaxy_eqFunction_3345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2709]] /* z[210] STATE(1,vz[210]) */) = -0.006400000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10094(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10095(DATA *data, threadData_t *threadData);


/*
equation index: 3348
type: SIMPLE_ASSIGN
y[210] = r_init[210] * sin(theta[210] - 0.0016000000000000003)
*/
void SpiralGalaxy_eqFunction_3348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3348};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2209]] /* y[210] STATE(1,vy[210]) */) = ((data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */)) * (sin((data->simulationInfo->realParameter[1716] /* theta[210] PARAM */) - 0.0016000000000000003));
  TRACE_POP
}

/*
equation index: 3349
type: SIMPLE_ASSIGN
x[210] = r_init[210] * cos(theta[210] - 0.0016000000000000003)
*/
void SpiralGalaxy_eqFunction_3349(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1709]] /* x[210] STATE(1,vx[210]) */) = ((data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */)) * (cos((data->simulationInfo->realParameter[1716] /* theta[210] PARAM */) - 0.0016000000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10096(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10097(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10100(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10099(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10098(DATA *data, threadData_t *threadData);


/*
equation index: 3355
type: SIMPLE_ASSIGN
vx[210] = (-sin(theta[210])) * r_init[210] * omega_c[210]
*/
void SpiralGalaxy_eqFunction_3355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[209]] /* vx[210] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1716] /* theta[210] PARAM */)))) * (((data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */)) * ((data->simulationInfo->realParameter[714] /* omega_c[210] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10093(DATA *data, threadData_t *threadData);


/*
equation index: 3357
type: SIMPLE_ASSIGN
vy[210] = cos(theta[210]) * r_init[210] * omega_c[210]
*/
void SpiralGalaxy_eqFunction_3357(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[709]] /* vy[210] STATE(1) */) = (cos((data->simulationInfo->realParameter[1716] /* theta[210] PARAM */))) * (((data->simulationInfo->realParameter[1215] /* r_init[210] PARAM */)) * ((data->simulationInfo->realParameter[714] /* omega_c[210] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10092(DATA *data, threadData_t *threadData);


/*
equation index: 3359
type: SIMPLE_ASSIGN
vz[210] = 0.0
*/
void SpiralGalaxy_eqFunction_3359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3359};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1209]] /* vz[210] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10091(DATA *data, threadData_t *threadData);


/*
equation index: 3361
type: SIMPLE_ASSIGN
z[211] = -0.006240000000000001
*/
void SpiralGalaxy_eqFunction_3361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3361};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2710]] /* z[211] STATE(1,vz[211]) */) = -0.006240000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10104(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10105(DATA *data, threadData_t *threadData);


/*
equation index: 3364
type: SIMPLE_ASSIGN
y[211] = r_init[211] * sin(theta[211] - 0.0015600000000000004)
*/
void SpiralGalaxy_eqFunction_3364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3364};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2210]] /* y[211] STATE(1,vy[211]) */) = ((data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */)) * (sin((data->simulationInfo->realParameter[1717] /* theta[211] PARAM */) - 0.0015600000000000004));
  TRACE_POP
}

/*
equation index: 3365
type: SIMPLE_ASSIGN
x[211] = r_init[211] * cos(theta[211] - 0.0015600000000000004)
*/
void SpiralGalaxy_eqFunction_3365(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3365};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1710]] /* x[211] STATE(1,vx[211]) */) = ((data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */)) * (cos((data->simulationInfo->realParameter[1717] /* theta[211] PARAM */) - 0.0015600000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10106(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10107(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10110(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10109(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10108(DATA *data, threadData_t *threadData);


/*
equation index: 3371
type: SIMPLE_ASSIGN
vx[211] = (-sin(theta[211])) * r_init[211] * omega_c[211]
*/
void SpiralGalaxy_eqFunction_3371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3371};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[210]] /* vx[211] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1717] /* theta[211] PARAM */)))) * (((data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */)) * ((data->simulationInfo->realParameter[715] /* omega_c[211] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10103(DATA *data, threadData_t *threadData);


/*
equation index: 3373
type: SIMPLE_ASSIGN
vy[211] = cos(theta[211]) * r_init[211] * omega_c[211]
*/
void SpiralGalaxy_eqFunction_3373(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3373};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[710]] /* vy[211] STATE(1) */) = (cos((data->simulationInfo->realParameter[1717] /* theta[211] PARAM */))) * (((data->simulationInfo->realParameter[1216] /* r_init[211] PARAM */)) * ((data->simulationInfo->realParameter[715] /* omega_c[211] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10102(DATA *data, threadData_t *threadData);


/*
equation index: 3375
type: SIMPLE_ASSIGN
vz[211] = 0.0
*/
void SpiralGalaxy_eqFunction_3375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3375};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1210]] /* vz[211] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10101(DATA *data, threadData_t *threadData);


/*
equation index: 3377
type: SIMPLE_ASSIGN
z[212] = -0.00608
*/
void SpiralGalaxy_eqFunction_3377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3377};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2711]] /* z[212] STATE(1,vz[212]) */) = -0.00608;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10114(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10115(DATA *data, threadData_t *threadData);


/*
equation index: 3380
type: SIMPLE_ASSIGN
y[212] = r_init[212] * sin(theta[212] - 0.0015200000000000003)
*/
void SpiralGalaxy_eqFunction_3380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3380};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2211]] /* y[212] STATE(1,vy[212]) */) = ((data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */)) * (sin((data->simulationInfo->realParameter[1718] /* theta[212] PARAM */) - 0.0015200000000000003));
  TRACE_POP
}

/*
equation index: 3381
type: SIMPLE_ASSIGN
x[212] = r_init[212] * cos(theta[212] - 0.0015200000000000003)
*/
void SpiralGalaxy_eqFunction_3381(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1711]] /* x[212] STATE(1,vx[212]) */) = ((data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */)) * (cos((data->simulationInfo->realParameter[1718] /* theta[212] PARAM */) - 0.0015200000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10116(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10117(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10120(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10119(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10118(DATA *data, threadData_t *threadData);


/*
equation index: 3387
type: SIMPLE_ASSIGN
vx[212] = (-sin(theta[212])) * r_init[212] * omega_c[212]
*/
void SpiralGalaxy_eqFunction_3387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[211]] /* vx[212] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1718] /* theta[212] PARAM */)))) * (((data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */)) * ((data->simulationInfo->realParameter[716] /* omega_c[212] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10113(DATA *data, threadData_t *threadData);


/*
equation index: 3389
type: SIMPLE_ASSIGN
vy[212] = cos(theta[212]) * r_init[212] * omega_c[212]
*/
void SpiralGalaxy_eqFunction_3389(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[711]] /* vy[212] STATE(1) */) = (cos((data->simulationInfo->realParameter[1718] /* theta[212] PARAM */))) * (((data->simulationInfo->realParameter[1217] /* r_init[212] PARAM */)) * ((data->simulationInfo->realParameter[716] /* omega_c[212] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10112(DATA *data, threadData_t *threadData);


/*
equation index: 3391
type: SIMPLE_ASSIGN
vz[212] = 0.0
*/
void SpiralGalaxy_eqFunction_3391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3391};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1211]] /* vz[212] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10111(DATA *data, threadData_t *threadData);


/*
equation index: 3393
type: SIMPLE_ASSIGN
z[213] = -0.005920000000000001
*/
void SpiralGalaxy_eqFunction_3393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3393};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2712]] /* z[213] STATE(1,vz[213]) */) = -0.005920000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10124(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10125(DATA *data, threadData_t *threadData);


/*
equation index: 3396
type: SIMPLE_ASSIGN
y[213] = r_init[213] * sin(theta[213] - 0.0014800000000000002)
*/
void SpiralGalaxy_eqFunction_3396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3396};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2212]] /* y[213] STATE(1,vy[213]) */) = ((data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */)) * (sin((data->simulationInfo->realParameter[1719] /* theta[213] PARAM */) - 0.0014800000000000002));
  TRACE_POP
}

/*
equation index: 3397
type: SIMPLE_ASSIGN
x[213] = r_init[213] * cos(theta[213] - 0.0014800000000000002)
*/
void SpiralGalaxy_eqFunction_3397(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1712]] /* x[213] STATE(1,vx[213]) */) = ((data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */)) * (cos((data->simulationInfo->realParameter[1719] /* theta[213] PARAM */) - 0.0014800000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10126(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10127(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10130(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10129(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10128(DATA *data, threadData_t *threadData);


/*
equation index: 3403
type: SIMPLE_ASSIGN
vx[213] = (-sin(theta[213])) * r_init[213] * omega_c[213]
*/
void SpiralGalaxy_eqFunction_3403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[212]] /* vx[213] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1719] /* theta[213] PARAM */)))) * (((data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */)) * ((data->simulationInfo->realParameter[717] /* omega_c[213] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10123(DATA *data, threadData_t *threadData);


/*
equation index: 3405
type: SIMPLE_ASSIGN
vy[213] = cos(theta[213]) * r_init[213] * omega_c[213]
*/
void SpiralGalaxy_eqFunction_3405(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[712]] /* vy[213] STATE(1) */) = (cos((data->simulationInfo->realParameter[1719] /* theta[213] PARAM */))) * (((data->simulationInfo->realParameter[1218] /* r_init[213] PARAM */)) * ((data->simulationInfo->realParameter[717] /* omega_c[213] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10122(DATA *data, threadData_t *threadData);


/*
equation index: 3407
type: SIMPLE_ASSIGN
vz[213] = 0.0
*/
void SpiralGalaxy_eqFunction_3407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3407};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1212]] /* vz[213] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10121(DATA *data, threadData_t *threadData);


/*
equation index: 3409
type: SIMPLE_ASSIGN
z[214] = -0.005760000000000001
*/
void SpiralGalaxy_eqFunction_3409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3409};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2713]] /* z[214] STATE(1,vz[214]) */) = -0.005760000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10134(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10135(DATA *data, threadData_t *threadData);


/*
equation index: 3412
type: SIMPLE_ASSIGN
y[214] = r_init[214] * sin(theta[214] - 0.00144)
*/
void SpiralGalaxy_eqFunction_3412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3412};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2213]] /* y[214] STATE(1,vy[214]) */) = ((data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */)) * (sin((data->simulationInfo->realParameter[1720] /* theta[214] PARAM */) - 0.00144));
  TRACE_POP
}

/*
equation index: 3413
type: SIMPLE_ASSIGN
x[214] = r_init[214] * cos(theta[214] - 0.00144)
*/
void SpiralGalaxy_eqFunction_3413(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1713]] /* x[214] STATE(1,vx[214]) */) = ((data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */)) * (cos((data->simulationInfo->realParameter[1720] /* theta[214] PARAM */) - 0.00144));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10136(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10137(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10140(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10139(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10138(DATA *data, threadData_t *threadData);


/*
equation index: 3419
type: SIMPLE_ASSIGN
vx[214] = (-sin(theta[214])) * r_init[214] * omega_c[214]
*/
void SpiralGalaxy_eqFunction_3419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[213]] /* vx[214] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1720] /* theta[214] PARAM */)))) * (((data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */)) * ((data->simulationInfo->realParameter[718] /* omega_c[214] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10133(DATA *data, threadData_t *threadData);


/*
equation index: 3421
type: SIMPLE_ASSIGN
vy[214] = cos(theta[214]) * r_init[214] * omega_c[214]
*/
void SpiralGalaxy_eqFunction_3421(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[713]] /* vy[214] STATE(1) */) = (cos((data->simulationInfo->realParameter[1720] /* theta[214] PARAM */))) * (((data->simulationInfo->realParameter[1219] /* r_init[214] PARAM */)) * ((data->simulationInfo->realParameter[718] /* omega_c[214] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10132(DATA *data, threadData_t *threadData);


/*
equation index: 3423
type: SIMPLE_ASSIGN
vz[214] = 0.0
*/
void SpiralGalaxy_eqFunction_3423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3423};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1213]] /* vz[214] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10131(DATA *data, threadData_t *threadData);


/*
equation index: 3425
type: SIMPLE_ASSIGN
z[215] = -0.005600000000000001
*/
void SpiralGalaxy_eqFunction_3425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3425};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2714]] /* z[215] STATE(1,vz[215]) */) = -0.005600000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10144(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10145(DATA *data, threadData_t *threadData);


/*
equation index: 3428
type: SIMPLE_ASSIGN
y[215] = r_init[215] * sin(theta[215] - 0.0014000000000000002)
*/
void SpiralGalaxy_eqFunction_3428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3428};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2214]] /* y[215] STATE(1,vy[215]) */) = ((data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */)) * (sin((data->simulationInfo->realParameter[1721] /* theta[215] PARAM */) - 0.0014000000000000002));
  TRACE_POP
}

/*
equation index: 3429
type: SIMPLE_ASSIGN
x[215] = r_init[215] * cos(theta[215] - 0.0014000000000000002)
*/
void SpiralGalaxy_eqFunction_3429(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1714]] /* x[215] STATE(1,vx[215]) */) = ((data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */)) * (cos((data->simulationInfo->realParameter[1721] /* theta[215] PARAM */) - 0.0014000000000000002));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10146(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10147(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10150(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10149(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10148(DATA *data, threadData_t *threadData);


/*
equation index: 3435
type: SIMPLE_ASSIGN
vx[215] = (-sin(theta[215])) * r_init[215] * omega_c[215]
*/
void SpiralGalaxy_eqFunction_3435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[214]] /* vx[215] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1721] /* theta[215] PARAM */)))) * (((data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */)) * ((data->simulationInfo->realParameter[719] /* omega_c[215] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10143(DATA *data, threadData_t *threadData);


/*
equation index: 3437
type: SIMPLE_ASSIGN
vy[215] = cos(theta[215]) * r_init[215] * omega_c[215]
*/
void SpiralGalaxy_eqFunction_3437(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[714]] /* vy[215] STATE(1) */) = (cos((data->simulationInfo->realParameter[1721] /* theta[215] PARAM */))) * (((data->simulationInfo->realParameter[1220] /* r_init[215] PARAM */)) * ((data->simulationInfo->realParameter[719] /* omega_c[215] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10142(DATA *data, threadData_t *threadData);


/*
equation index: 3439
type: SIMPLE_ASSIGN
vz[215] = 0.0
*/
void SpiralGalaxy_eqFunction_3439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3439};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1214]] /* vz[215] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10141(DATA *data, threadData_t *threadData);


/*
equation index: 3441
type: SIMPLE_ASSIGN
z[216] = -0.00544
*/
void SpiralGalaxy_eqFunction_3441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3441};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2715]] /* z[216] STATE(1,vz[216]) */) = -0.00544;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10154(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10155(DATA *data, threadData_t *threadData);


/*
equation index: 3444
type: SIMPLE_ASSIGN
y[216] = r_init[216] * sin(theta[216] - 0.00136)
*/
void SpiralGalaxy_eqFunction_3444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3444};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2215]] /* y[216] STATE(1,vy[216]) */) = ((data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */)) * (sin((data->simulationInfo->realParameter[1722] /* theta[216] PARAM */) - 0.00136));
  TRACE_POP
}

/*
equation index: 3445
type: SIMPLE_ASSIGN
x[216] = r_init[216] * cos(theta[216] - 0.00136)
*/
void SpiralGalaxy_eqFunction_3445(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1715]] /* x[216] STATE(1,vx[216]) */) = ((data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */)) * (cos((data->simulationInfo->realParameter[1722] /* theta[216] PARAM */) - 0.00136));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10156(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10157(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10160(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10159(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10158(DATA *data, threadData_t *threadData);


/*
equation index: 3451
type: SIMPLE_ASSIGN
vx[216] = (-sin(theta[216])) * r_init[216] * omega_c[216]
*/
void SpiralGalaxy_eqFunction_3451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[215]] /* vx[216] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1722] /* theta[216] PARAM */)))) * (((data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */)) * ((data->simulationInfo->realParameter[720] /* omega_c[216] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10153(DATA *data, threadData_t *threadData);


/*
equation index: 3453
type: SIMPLE_ASSIGN
vy[216] = cos(theta[216]) * r_init[216] * omega_c[216]
*/
void SpiralGalaxy_eqFunction_3453(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[715]] /* vy[216] STATE(1) */) = (cos((data->simulationInfo->realParameter[1722] /* theta[216] PARAM */))) * (((data->simulationInfo->realParameter[1221] /* r_init[216] PARAM */)) * ((data->simulationInfo->realParameter[720] /* omega_c[216] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10152(DATA *data, threadData_t *threadData);


/*
equation index: 3455
type: SIMPLE_ASSIGN
vz[216] = 0.0
*/
void SpiralGalaxy_eqFunction_3455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3455};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1215]] /* vz[216] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10151(DATA *data, threadData_t *threadData);


/*
equation index: 3457
type: SIMPLE_ASSIGN
z[217] = -0.00528
*/
void SpiralGalaxy_eqFunction_3457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3457};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2716]] /* z[217] STATE(1,vz[217]) */) = -0.00528;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10164(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10165(DATA *data, threadData_t *threadData);


/*
equation index: 3460
type: SIMPLE_ASSIGN
y[217] = r_init[217] * sin(theta[217] - 0.00132)
*/
void SpiralGalaxy_eqFunction_3460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3460};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2216]] /* y[217] STATE(1,vy[217]) */) = ((data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */)) * (sin((data->simulationInfo->realParameter[1723] /* theta[217] PARAM */) - 0.00132));
  TRACE_POP
}

/*
equation index: 3461
type: SIMPLE_ASSIGN
x[217] = r_init[217] * cos(theta[217] - 0.00132)
*/
void SpiralGalaxy_eqFunction_3461(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1716]] /* x[217] STATE(1,vx[217]) */) = ((data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */)) * (cos((data->simulationInfo->realParameter[1723] /* theta[217] PARAM */) - 0.00132));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10166(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10167(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10170(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10169(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10168(DATA *data, threadData_t *threadData);


/*
equation index: 3467
type: SIMPLE_ASSIGN
vx[217] = (-sin(theta[217])) * r_init[217] * omega_c[217]
*/
void SpiralGalaxy_eqFunction_3467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[216]] /* vx[217] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1723] /* theta[217] PARAM */)))) * (((data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */)) * ((data->simulationInfo->realParameter[721] /* omega_c[217] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10163(DATA *data, threadData_t *threadData);


/*
equation index: 3469
type: SIMPLE_ASSIGN
vy[217] = cos(theta[217]) * r_init[217] * omega_c[217]
*/
void SpiralGalaxy_eqFunction_3469(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[716]] /* vy[217] STATE(1) */) = (cos((data->simulationInfo->realParameter[1723] /* theta[217] PARAM */))) * (((data->simulationInfo->realParameter[1222] /* r_init[217] PARAM */)) * ((data->simulationInfo->realParameter[721] /* omega_c[217] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10162(DATA *data, threadData_t *threadData);


/*
equation index: 3471
type: SIMPLE_ASSIGN
vz[217] = 0.0
*/
void SpiralGalaxy_eqFunction_3471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3471};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1216]] /* vz[217] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10161(DATA *data, threadData_t *threadData);


/*
equation index: 3473
type: SIMPLE_ASSIGN
z[218] = -0.00512
*/
void SpiralGalaxy_eqFunction_3473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3473};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2717]] /* z[218] STATE(1,vz[218]) */) = -0.00512;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10174(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10175(DATA *data, threadData_t *threadData);


/*
equation index: 3476
type: SIMPLE_ASSIGN
y[218] = r_init[218] * sin(theta[218] - 0.00128)
*/
void SpiralGalaxy_eqFunction_3476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3476};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2217]] /* y[218] STATE(1,vy[218]) */) = ((data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */)) * (sin((data->simulationInfo->realParameter[1724] /* theta[218] PARAM */) - 0.00128));
  TRACE_POP
}

/*
equation index: 3477
type: SIMPLE_ASSIGN
x[218] = r_init[218] * cos(theta[218] - 0.00128)
*/
void SpiralGalaxy_eqFunction_3477(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3477};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1717]] /* x[218] STATE(1,vx[218]) */) = ((data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */)) * (cos((data->simulationInfo->realParameter[1724] /* theta[218] PARAM */) - 0.00128));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10176(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10177(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10180(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10179(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10178(DATA *data, threadData_t *threadData);


/*
equation index: 3483
type: SIMPLE_ASSIGN
vx[218] = (-sin(theta[218])) * r_init[218] * omega_c[218]
*/
void SpiralGalaxy_eqFunction_3483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[217]] /* vx[218] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1724] /* theta[218] PARAM */)))) * (((data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */)) * ((data->simulationInfo->realParameter[722] /* omega_c[218] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10173(DATA *data, threadData_t *threadData);


/*
equation index: 3485
type: SIMPLE_ASSIGN
vy[218] = cos(theta[218]) * r_init[218] * omega_c[218]
*/
void SpiralGalaxy_eqFunction_3485(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[717]] /* vy[218] STATE(1) */) = (cos((data->simulationInfo->realParameter[1724] /* theta[218] PARAM */))) * (((data->simulationInfo->realParameter[1223] /* r_init[218] PARAM */)) * ((data->simulationInfo->realParameter[722] /* omega_c[218] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10172(DATA *data, threadData_t *threadData);


/*
equation index: 3487
type: SIMPLE_ASSIGN
vz[218] = 0.0
*/
void SpiralGalaxy_eqFunction_3487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3487};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1217]] /* vz[218] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10171(DATA *data, threadData_t *threadData);


/*
equation index: 3489
type: SIMPLE_ASSIGN
z[219] = -0.004960000000000001
*/
void SpiralGalaxy_eqFunction_3489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3489};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2718]] /* z[219] STATE(1,vz[219]) */) = -0.004960000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10184(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10185(DATA *data, threadData_t *threadData);


/*
equation index: 3492
type: SIMPLE_ASSIGN
y[219] = r_init[219] * sin(theta[219] - 0.00124)
*/
void SpiralGalaxy_eqFunction_3492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3492};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2218]] /* y[219] STATE(1,vy[219]) */) = ((data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */)) * (sin((data->simulationInfo->realParameter[1725] /* theta[219] PARAM */) - 0.00124));
  TRACE_POP
}

/*
equation index: 3493
type: SIMPLE_ASSIGN
x[219] = r_init[219] * cos(theta[219] - 0.00124)
*/
void SpiralGalaxy_eqFunction_3493(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1718]] /* x[219] STATE(1,vx[219]) */) = ((data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */)) * (cos((data->simulationInfo->realParameter[1725] /* theta[219] PARAM */) - 0.00124));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10186(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10187(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10190(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10189(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10188(DATA *data, threadData_t *threadData);


/*
equation index: 3499
type: SIMPLE_ASSIGN
vx[219] = (-sin(theta[219])) * r_init[219] * omega_c[219]
*/
void SpiralGalaxy_eqFunction_3499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3499};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[218]] /* vx[219] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1725] /* theta[219] PARAM */)))) * (((data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */)) * ((data->simulationInfo->realParameter[723] /* omega_c[219] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10183(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_6(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_9879(data, threadData);
  SpiralGalaxy_eqFunction_9878(data, threadData);
  SpiralGalaxy_eqFunction_3003(data, threadData);
  SpiralGalaxy_eqFunction_9873(data, threadData);
  SpiralGalaxy_eqFunction_3005(data, threadData);
  SpiralGalaxy_eqFunction_9872(data, threadData);
  SpiralGalaxy_eqFunction_3007(data, threadData);
  SpiralGalaxy_eqFunction_9871(data, threadData);
  SpiralGalaxy_eqFunction_3009(data, threadData);
  SpiralGalaxy_eqFunction_9884(data, threadData);
  SpiralGalaxy_eqFunction_9885(data, threadData);
  SpiralGalaxy_eqFunction_3012(data, threadData);
  SpiralGalaxy_eqFunction_3013(data, threadData);
  SpiralGalaxy_eqFunction_9886(data, threadData);
  SpiralGalaxy_eqFunction_9887(data, threadData);
  SpiralGalaxy_eqFunction_9890(data, threadData);
  SpiralGalaxy_eqFunction_9889(data, threadData);
  SpiralGalaxy_eqFunction_9888(data, threadData);
  SpiralGalaxy_eqFunction_3019(data, threadData);
  SpiralGalaxy_eqFunction_9883(data, threadData);
  SpiralGalaxy_eqFunction_3021(data, threadData);
  SpiralGalaxy_eqFunction_9882(data, threadData);
  SpiralGalaxy_eqFunction_3023(data, threadData);
  SpiralGalaxy_eqFunction_9881(data, threadData);
  SpiralGalaxy_eqFunction_3025(data, threadData);
  SpiralGalaxy_eqFunction_9894(data, threadData);
  SpiralGalaxy_eqFunction_9895(data, threadData);
  SpiralGalaxy_eqFunction_3028(data, threadData);
  SpiralGalaxy_eqFunction_3029(data, threadData);
  SpiralGalaxy_eqFunction_9896(data, threadData);
  SpiralGalaxy_eqFunction_9897(data, threadData);
  SpiralGalaxy_eqFunction_9900(data, threadData);
  SpiralGalaxy_eqFunction_9899(data, threadData);
  SpiralGalaxy_eqFunction_9898(data, threadData);
  SpiralGalaxy_eqFunction_3035(data, threadData);
  SpiralGalaxy_eqFunction_9893(data, threadData);
  SpiralGalaxy_eqFunction_3037(data, threadData);
  SpiralGalaxy_eqFunction_9892(data, threadData);
  SpiralGalaxy_eqFunction_3039(data, threadData);
  SpiralGalaxy_eqFunction_9891(data, threadData);
  SpiralGalaxy_eqFunction_3041(data, threadData);
  SpiralGalaxy_eqFunction_9904(data, threadData);
  SpiralGalaxy_eqFunction_9905(data, threadData);
  SpiralGalaxy_eqFunction_3044(data, threadData);
  SpiralGalaxy_eqFunction_3045(data, threadData);
  SpiralGalaxy_eqFunction_9906(data, threadData);
  SpiralGalaxy_eqFunction_9907(data, threadData);
  SpiralGalaxy_eqFunction_9910(data, threadData);
  SpiralGalaxy_eqFunction_9909(data, threadData);
  SpiralGalaxy_eqFunction_9908(data, threadData);
  SpiralGalaxy_eqFunction_3051(data, threadData);
  SpiralGalaxy_eqFunction_9903(data, threadData);
  SpiralGalaxy_eqFunction_3053(data, threadData);
  SpiralGalaxy_eqFunction_9902(data, threadData);
  SpiralGalaxy_eqFunction_3055(data, threadData);
  SpiralGalaxy_eqFunction_9901(data, threadData);
  SpiralGalaxy_eqFunction_3057(data, threadData);
  SpiralGalaxy_eqFunction_9914(data, threadData);
  SpiralGalaxy_eqFunction_9915(data, threadData);
  SpiralGalaxy_eqFunction_3060(data, threadData);
  SpiralGalaxy_eqFunction_3061(data, threadData);
  SpiralGalaxy_eqFunction_9916(data, threadData);
  SpiralGalaxy_eqFunction_9917(data, threadData);
  SpiralGalaxy_eqFunction_9920(data, threadData);
  SpiralGalaxy_eqFunction_9919(data, threadData);
  SpiralGalaxy_eqFunction_9918(data, threadData);
  SpiralGalaxy_eqFunction_3067(data, threadData);
  SpiralGalaxy_eqFunction_9913(data, threadData);
  SpiralGalaxy_eqFunction_3069(data, threadData);
  SpiralGalaxy_eqFunction_9912(data, threadData);
  SpiralGalaxy_eqFunction_3071(data, threadData);
  SpiralGalaxy_eqFunction_9911(data, threadData);
  SpiralGalaxy_eqFunction_3073(data, threadData);
  SpiralGalaxy_eqFunction_9924(data, threadData);
  SpiralGalaxy_eqFunction_9925(data, threadData);
  SpiralGalaxy_eqFunction_3076(data, threadData);
  SpiralGalaxy_eqFunction_3077(data, threadData);
  SpiralGalaxy_eqFunction_9926(data, threadData);
  SpiralGalaxy_eqFunction_9927(data, threadData);
  SpiralGalaxy_eqFunction_9930(data, threadData);
  SpiralGalaxy_eqFunction_9929(data, threadData);
  SpiralGalaxy_eqFunction_9928(data, threadData);
  SpiralGalaxy_eqFunction_3083(data, threadData);
  SpiralGalaxy_eqFunction_9923(data, threadData);
  SpiralGalaxy_eqFunction_3085(data, threadData);
  SpiralGalaxy_eqFunction_9922(data, threadData);
  SpiralGalaxy_eqFunction_3087(data, threadData);
  SpiralGalaxy_eqFunction_9921(data, threadData);
  SpiralGalaxy_eqFunction_3089(data, threadData);
  SpiralGalaxy_eqFunction_9934(data, threadData);
  SpiralGalaxy_eqFunction_9935(data, threadData);
  SpiralGalaxy_eqFunction_3092(data, threadData);
  SpiralGalaxy_eqFunction_3093(data, threadData);
  SpiralGalaxy_eqFunction_9936(data, threadData);
  SpiralGalaxy_eqFunction_9937(data, threadData);
  SpiralGalaxy_eqFunction_9940(data, threadData);
  SpiralGalaxy_eqFunction_9939(data, threadData);
  SpiralGalaxy_eqFunction_9938(data, threadData);
  SpiralGalaxy_eqFunction_3099(data, threadData);
  SpiralGalaxy_eqFunction_9933(data, threadData);
  SpiralGalaxy_eqFunction_3101(data, threadData);
  SpiralGalaxy_eqFunction_9932(data, threadData);
  SpiralGalaxy_eqFunction_3103(data, threadData);
  SpiralGalaxy_eqFunction_9931(data, threadData);
  SpiralGalaxy_eqFunction_3105(data, threadData);
  SpiralGalaxy_eqFunction_9944(data, threadData);
  SpiralGalaxy_eqFunction_9945(data, threadData);
  SpiralGalaxy_eqFunction_3108(data, threadData);
  SpiralGalaxy_eqFunction_3109(data, threadData);
  SpiralGalaxy_eqFunction_9946(data, threadData);
  SpiralGalaxy_eqFunction_9947(data, threadData);
  SpiralGalaxy_eqFunction_9950(data, threadData);
  SpiralGalaxy_eqFunction_9949(data, threadData);
  SpiralGalaxy_eqFunction_9948(data, threadData);
  SpiralGalaxy_eqFunction_3115(data, threadData);
  SpiralGalaxy_eqFunction_9943(data, threadData);
  SpiralGalaxy_eqFunction_3117(data, threadData);
  SpiralGalaxy_eqFunction_9942(data, threadData);
  SpiralGalaxy_eqFunction_3119(data, threadData);
  SpiralGalaxy_eqFunction_9941(data, threadData);
  SpiralGalaxy_eqFunction_3121(data, threadData);
  SpiralGalaxy_eqFunction_9954(data, threadData);
  SpiralGalaxy_eqFunction_9955(data, threadData);
  SpiralGalaxy_eqFunction_3124(data, threadData);
  SpiralGalaxy_eqFunction_3125(data, threadData);
  SpiralGalaxy_eqFunction_9956(data, threadData);
  SpiralGalaxy_eqFunction_9957(data, threadData);
  SpiralGalaxy_eqFunction_9960(data, threadData);
  SpiralGalaxy_eqFunction_9959(data, threadData);
  SpiralGalaxy_eqFunction_9958(data, threadData);
  SpiralGalaxy_eqFunction_3131(data, threadData);
  SpiralGalaxy_eqFunction_9953(data, threadData);
  SpiralGalaxy_eqFunction_3133(data, threadData);
  SpiralGalaxy_eqFunction_9952(data, threadData);
  SpiralGalaxy_eqFunction_3135(data, threadData);
  SpiralGalaxy_eqFunction_9951(data, threadData);
  SpiralGalaxy_eqFunction_3137(data, threadData);
  SpiralGalaxy_eqFunction_9964(data, threadData);
  SpiralGalaxy_eqFunction_9965(data, threadData);
  SpiralGalaxy_eqFunction_3140(data, threadData);
  SpiralGalaxy_eqFunction_3141(data, threadData);
  SpiralGalaxy_eqFunction_9966(data, threadData);
  SpiralGalaxy_eqFunction_9967(data, threadData);
  SpiralGalaxy_eqFunction_9970(data, threadData);
  SpiralGalaxy_eqFunction_9969(data, threadData);
  SpiralGalaxy_eqFunction_9968(data, threadData);
  SpiralGalaxy_eqFunction_3147(data, threadData);
  SpiralGalaxy_eqFunction_9963(data, threadData);
  SpiralGalaxy_eqFunction_3149(data, threadData);
  SpiralGalaxy_eqFunction_9962(data, threadData);
  SpiralGalaxy_eqFunction_3151(data, threadData);
  SpiralGalaxy_eqFunction_9961(data, threadData);
  SpiralGalaxy_eqFunction_3153(data, threadData);
  SpiralGalaxy_eqFunction_9974(data, threadData);
  SpiralGalaxy_eqFunction_9975(data, threadData);
  SpiralGalaxy_eqFunction_3156(data, threadData);
  SpiralGalaxy_eqFunction_3157(data, threadData);
  SpiralGalaxy_eqFunction_9976(data, threadData);
  SpiralGalaxy_eqFunction_9977(data, threadData);
  SpiralGalaxy_eqFunction_9980(data, threadData);
  SpiralGalaxy_eqFunction_9979(data, threadData);
  SpiralGalaxy_eqFunction_9978(data, threadData);
  SpiralGalaxy_eqFunction_3163(data, threadData);
  SpiralGalaxy_eqFunction_9973(data, threadData);
  SpiralGalaxy_eqFunction_3165(data, threadData);
  SpiralGalaxy_eqFunction_9972(data, threadData);
  SpiralGalaxy_eqFunction_3167(data, threadData);
  SpiralGalaxy_eqFunction_9971(data, threadData);
  SpiralGalaxy_eqFunction_3169(data, threadData);
  SpiralGalaxy_eqFunction_9984(data, threadData);
  SpiralGalaxy_eqFunction_9985(data, threadData);
  SpiralGalaxy_eqFunction_3172(data, threadData);
  SpiralGalaxy_eqFunction_3173(data, threadData);
  SpiralGalaxy_eqFunction_9986(data, threadData);
  SpiralGalaxy_eqFunction_9987(data, threadData);
  SpiralGalaxy_eqFunction_9990(data, threadData);
  SpiralGalaxy_eqFunction_9989(data, threadData);
  SpiralGalaxy_eqFunction_9988(data, threadData);
  SpiralGalaxy_eqFunction_3179(data, threadData);
  SpiralGalaxy_eqFunction_9983(data, threadData);
  SpiralGalaxy_eqFunction_3181(data, threadData);
  SpiralGalaxy_eqFunction_9982(data, threadData);
  SpiralGalaxy_eqFunction_3183(data, threadData);
  SpiralGalaxy_eqFunction_9981(data, threadData);
  SpiralGalaxy_eqFunction_3185(data, threadData);
  SpiralGalaxy_eqFunction_9994(data, threadData);
  SpiralGalaxy_eqFunction_9995(data, threadData);
  SpiralGalaxy_eqFunction_3188(data, threadData);
  SpiralGalaxy_eqFunction_3189(data, threadData);
  SpiralGalaxy_eqFunction_9996(data, threadData);
  SpiralGalaxy_eqFunction_9997(data, threadData);
  SpiralGalaxy_eqFunction_10000(data, threadData);
  SpiralGalaxy_eqFunction_9999(data, threadData);
  SpiralGalaxy_eqFunction_9998(data, threadData);
  SpiralGalaxy_eqFunction_3195(data, threadData);
  SpiralGalaxy_eqFunction_9993(data, threadData);
  SpiralGalaxy_eqFunction_3197(data, threadData);
  SpiralGalaxy_eqFunction_9992(data, threadData);
  SpiralGalaxy_eqFunction_3199(data, threadData);
  SpiralGalaxy_eqFunction_9991(data, threadData);
  SpiralGalaxy_eqFunction_3201(data, threadData);
  SpiralGalaxy_eqFunction_10004(data, threadData);
  SpiralGalaxy_eqFunction_10005(data, threadData);
  SpiralGalaxy_eqFunction_3204(data, threadData);
  SpiralGalaxy_eqFunction_3205(data, threadData);
  SpiralGalaxy_eqFunction_10006(data, threadData);
  SpiralGalaxy_eqFunction_10007(data, threadData);
  SpiralGalaxy_eqFunction_10010(data, threadData);
  SpiralGalaxy_eqFunction_10009(data, threadData);
  SpiralGalaxy_eqFunction_10008(data, threadData);
  SpiralGalaxy_eqFunction_3211(data, threadData);
  SpiralGalaxy_eqFunction_10003(data, threadData);
  SpiralGalaxy_eqFunction_3213(data, threadData);
  SpiralGalaxy_eqFunction_10002(data, threadData);
  SpiralGalaxy_eqFunction_3215(data, threadData);
  SpiralGalaxy_eqFunction_10001(data, threadData);
  SpiralGalaxy_eqFunction_3217(data, threadData);
  SpiralGalaxy_eqFunction_10014(data, threadData);
  SpiralGalaxy_eqFunction_10015(data, threadData);
  SpiralGalaxy_eqFunction_3220(data, threadData);
  SpiralGalaxy_eqFunction_3221(data, threadData);
  SpiralGalaxy_eqFunction_10016(data, threadData);
  SpiralGalaxy_eqFunction_10017(data, threadData);
  SpiralGalaxy_eqFunction_10020(data, threadData);
  SpiralGalaxy_eqFunction_10019(data, threadData);
  SpiralGalaxy_eqFunction_10018(data, threadData);
  SpiralGalaxy_eqFunction_3227(data, threadData);
  SpiralGalaxy_eqFunction_10013(data, threadData);
  SpiralGalaxy_eqFunction_3229(data, threadData);
  SpiralGalaxy_eqFunction_10012(data, threadData);
  SpiralGalaxy_eqFunction_3231(data, threadData);
  SpiralGalaxy_eqFunction_10011(data, threadData);
  SpiralGalaxy_eqFunction_3233(data, threadData);
  SpiralGalaxy_eqFunction_10024(data, threadData);
  SpiralGalaxy_eqFunction_10025(data, threadData);
  SpiralGalaxy_eqFunction_3236(data, threadData);
  SpiralGalaxy_eqFunction_3237(data, threadData);
  SpiralGalaxy_eqFunction_10026(data, threadData);
  SpiralGalaxy_eqFunction_10027(data, threadData);
  SpiralGalaxy_eqFunction_10030(data, threadData);
  SpiralGalaxy_eqFunction_10029(data, threadData);
  SpiralGalaxy_eqFunction_10028(data, threadData);
  SpiralGalaxy_eqFunction_3243(data, threadData);
  SpiralGalaxy_eqFunction_10023(data, threadData);
  SpiralGalaxy_eqFunction_3245(data, threadData);
  SpiralGalaxy_eqFunction_10022(data, threadData);
  SpiralGalaxy_eqFunction_3247(data, threadData);
  SpiralGalaxy_eqFunction_10021(data, threadData);
  SpiralGalaxy_eqFunction_3249(data, threadData);
  SpiralGalaxy_eqFunction_10034(data, threadData);
  SpiralGalaxy_eqFunction_10035(data, threadData);
  SpiralGalaxy_eqFunction_3252(data, threadData);
  SpiralGalaxy_eqFunction_3253(data, threadData);
  SpiralGalaxy_eqFunction_10036(data, threadData);
  SpiralGalaxy_eqFunction_10037(data, threadData);
  SpiralGalaxy_eqFunction_10040(data, threadData);
  SpiralGalaxy_eqFunction_10039(data, threadData);
  SpiralGalaxy_eqFunction_10038(data, threadData);
  SpiralGalaxy_eqFunction_3259(data, threadData);
  SpiralGalaxy_eqFunction_10033(data, threadData);
  SpiralGalaxy_eqFunction_3261(data, threadData);
  SpiralGalaxy_eqFunction_10032(data, threadData);
  SpiralGalaxy_eqFunction_3263(data, threadData);
  SpiralGalaxy_eqFunction_10031(data, threadData);
  SpiralGalaxy_eqFunction_3265(data, threadData);
  SpiralGalaxy_eqFunction_10044(data, threadData);
  SpiralGalaxy_eqFunction_10045(data, threadData);
  SpiralGalaxy_eqFunction_3268(data, threadData);
  SpiralGalaxy_eqFunction_3269(data, threadData);
  SpiralGalaxy_eqFunction_10046(data, threadData);
  SpiralGalaxy_eqFunction_10047(data, threadData);
  SpiralGalaxy_eqFunction_10050(data, threadData);
  SpiralGalaxy_eqFunction_10049(data, threadData);
  SpiralGalaxy_eqFunction_10048(data, threadData);
  SpiralGalaxy_eqFunction_3275(data, threadData);
  SpiralGalaxy_eqFunction_10043(data, threadData);
  SpiralGalaxy_eqFunction_3277(data, threadData);
  SpiralGalaxy_eqFunction_10042(data, threadData);
  SpiralGalaxy_eqFunction_3279(data, threadData);
  SpiralGalaxy_eqFunction_10041(data, threadData);
  SpiralGalaxy_eqFunction_3281(data, threadData);
  SpiralGalaxy_eqFunction_10054(data, threadData);
  SpiralGalaxy_eqFunction_10055(data, threadData);
  SpiralGalaxy_eqFunction_3284(data, threadData);
  SpiralGalaxy_eqFunction_3285(data, threadData);
  SpiralGalaxy_eqFunction_10056(data, threadData);
  SpiralGalaxy_eqFunction_10057(data, threadData);
  SpiralGalaxy_eqFunction_10060(data, threadData);
  SpiralGalaxy_eqFunction_10059(data, threadData);
  SpiralGalaxy_eqFunction_10058(data, threadData);
  SpiralGalaxy_eqFunction_3291(data, threadData);
  SpiralGalaxy_eqFunction_10053(data, threadData);
  SpiralGalaxy_eqFunction_3293(data, threadData);
  SpiralGalaxy_eqFunction_10052(data, threadData);
  SpiralGalaxy_eqFunction_3295(data, threadData);
  SpiralGalaxy_eqFunction_10051(data, threadData);
  SpiralGalaxy_eqFunction_3297(data, threadData);
  SpiralGalaxy_eqFunction_10064(data, threadData);
  SpiralGalaxy_eqFunction_10065(data, threadData);
  SpiralGalaxy_eqFunction_3300(data, threadData);
  SpiralGalaxy_eqFunction_3301(data, threadData);
  SpiralGalaxy_eqFunction_10066(data, threadData);
  SpiralGalaxy_eqFunction_10067(data, threadData);
  SpiralGalaxy_eqFunction_10070(data, threadData);
  SpiralGalaxy_eqFunction_10069(data, threadData);
  SpiralGalaxy_eqFunction_10068(data, threadData);
  SpiralGalaxy_eqFunction_3307(data, threadData);
  SpiralGalaxy_eqFunction_10063(data, threadData);
  SpiralGalaxy_eqFunction_3309(data, threadData);
  SpiralGalaxy_eqFunction_10062(data, threadData);
  SpiralGalaxy_eqFunction_3311(data, threadData);
  SpiralGalaxy_eqFunction_10061(data, threadData);
  SpiralGalaxy_eqFunction_3313(data, threadData);
  SpiralGalaxy_eqFunction_10074(data, threadData);
  SpiralGalaxy_eqFunction_10075(data, threadData);
  SpiralGalaxy_eqFunction_3316(data, threadData);
  SpiralGalaxy_eqFunction_3317(data, threadData);
  SpiralGalaxy_eqFunction_10076(data, threadData);
  SpiralGalaxy_eqFunction_10077(data, threadData);
  SpiralGalaxy_eqFunction_10080(data, threadData);
  SpiralGalaxy_eqFunction_10079(data, threadData);
  SpiralGalaxy_eqFunction_10078(data, threadData);
  SpiralGalaxy_eqFunction_3323(data, threadData);
  SpiralGalaxy_eqFunction_10073(data, threadData);
  SpiralGalaxy_eqFunction_3325(data, threadData);
  SpiralGalaxy_eqFunction_10072(data, threadData);
  SpiralGalaxy_eqFunction_3327(data, threadData);
  SpiralGalaxy_eqFunction_10071(data, threadData);
  SpiralGalaxy_eqFunction_3329(data, threadData);
  SpiralGalaxy_eqFunction_10084(data, threadData);
  SpiralGalaxy_eqFunction_10085(data, threadData);
  SpiralGalaxy_eqFunction_3332(data, threadData);
  SpiralGalaxy_eqFunction_3333(data, threadData);
  SpiralGalaxy_eqFunction_10086(data, threadData);
  SpiralGalaxy_eqFunction_10087(data, threadData);
  SpiralGalaxy_eqFunction_10090(data, threadData);
  SpiralGalaxy_eqFunction_10089(data, threadData);
  SpiralGalaxy_eqFunction_10088(data, threadData);
  SpiralGalaxy_eqFunction_3339(data, threadData);
  SpiralGalaxy_eqFunction_10083(data, threadData);
  SpiralGalaxy_eqFunction_3341(data, threadData);
  SpiralGalaxy_eqFunction_10082(data, threadData);
  SpiralGalaxy_eqFunction_3343(data, threadData);
  SpiralGalaxy_eqFunction_10081(data, threadData);
  SpiralGalaxy_eqFunction_3345(data, threadData);
  SpiralGalaxy_eqFunction_10094(data, threadData);
  SpiralGalaxy_eqFunction_10095(data, threadData);
  SpiralGalaxy_eqFunction_3348(data, threadData);
  SpiralGalaxy_eqFunction_3349(data, threadData);
  SpiralGalaxy_eqFunction_10096(data, threadData);
  SpiralGalaxy_eqFunction_10097(data, threadData);
  SpiralGalaxy_eqFunction_10100(data, threadData);
  SpiralGalaxy_eqFunction_10099(data, threadData);
  SpiralGalaxy_eqFunction_10098(data, threadData);
  SpiralGalaxy_eqFunction_3355(data, threadData);
  SpiralGalaxy_eqFunction_10093(data, threadData);
  SpiralGalaxy_eqFunction_3357(data, threadData);
  SpiralGalaxy_eqFunction_10092(data, threadData);
  SpiralGalaxy_eqFunction_3359(data, threadData);
  SpiralGalaxy_eqFunction_10091(data, threadData);
  SpiralGalaxy_eqFunction_3361(data, threadData);
  SpiralGalaxy_eqFunction_10104(data, threadData);
  SpiralGalaxy_eqFunction_10105(data, threadData);
  SpiralGalaxy_eqFunction_3364(data, threadData);
  SpiralGalaxy_eqFunction_3365(data, threadData);
  SpiralGalaxy_eqFunction_10106(data, threadData);
  SpiralGalaxy_eqFunction_10107(data, threadData);
  SpiralGalaxy_eqFunction_10110(data, threadData);
  SpiralGalaxy_eqFunction_10109(data, threadData);
  SpiralGalaxy_eqFunction_10108(data, threadData);
  SpiralGalaxy_eqFunction_3371(data, threadData);
  SpiralGalaxy_eqFunction_10103(data, threadData);
  SpiralGalaxy_eqFunction_3373(data, threadData);
  SpiralGalaxy_eqFunction_10102(data, threadData);
  SpiralGalaxy_eqFunction_3375(data, threadData);
  SpiralGalaxy_eqFunction_10101(data, threadData);
  SpiralGalaxy_eqFunction_3377(data, threadData);
  SpiralGalaxy_eqFunction_10114(data, threadData);
  SpiralGalaxy_eqFunction_10115(data, threadData);
  SpiralGalaxy_eqFunction_3380(data, threadData);
  SpiralGalaxy_eqFunction_3381(data, threadData);
  SpiralGalaxy_eqFunction_10116(data, threadData);
  SpiralGalaxy_eqFunction_10117(data, threadData);
  SpiralGalaxy_eqFunction_10120(data, threadData);
  SpiralGalaxy_eqFunction_10119(data, threadData);
  SpiralGalaxy_eqFunction_10118(data, threadData);
  SpiralGalaxy_eqFunction_3387(data, threadData);
  SpiralGalaxy_eqFunction_10113(data, threadData);
  SpiralGalaxy_eqFunction_3389(data, threadData);
  SpiralGalaxy_eqFunction_10112(data, threadData);
  SpiralGalaxy_eqFunction_3391(data, threadData);
  SpiralGalaxy_eqFunction_10111(data, threadData);
  SpiralGalaxy_eqFunction_3393(data, threadData);
  SpiralGalaxy_eqFunction_10124(data, threadData);
  SpiralGalaxy_eqFunction_10125(data, threadData);
  SpiralGalaxy_eqFunction_3396(data, threadData);
  SpiralGalaxy_eqFunction_3397(data, threadData);
  SpiralGalaxy_eqFunction_10126(data, threadData);
  SpiralGalaxy_eqFunction_10127(data, threadData);
  SpiralGalaxy_eqFunction_10130(data, threadData);
  SpiralGalaxy_eqFunction_10129(data, threadData);
  SpiralGalaxy_eqFunction_10128(data, threadData);
  SpiralGalaxy_eqFunction_3403(data, threadData);
  SpiralGalaxy_eqFunction_10123(data, threadData);
  SpiralGalaxy_eqFunction_3405(data, threadData);
  SpiralGalaxy_eqFunction_10122(data, threadData);
  SpiralGalaxy_eqFunction_3407(data, threadData);
  SpiralGalaxy_eqFunction_10121(data, threadData);
  SpiralGalaxy_eqFunction_3409(data, threadData);
  SpiralGalaxy_eqFunction_10134(data, threadData);
  SpiralGalaxy_eqFunction_10135(data, threadData);
  SpiralGalaxy_eqFunction_3412(data, threadData);
  SpiralGalaxy_eqFunction_3413(data, threadData);
  SpiralGalaxy_eqFunction_10136(data, threadData);
  SpiralGalaxy_eqFunction_10137(data, threadData);
  SpiralGalaxy_eqFunction_10140(data, threadData);
  SpiralGalaxy_eqFunction_10139(data, threadData);
  SpiralGalaxy_eqFunction_10138(data, threadData);
  SpiralGalaxy_eqFunction_3419(data, threadData);
  SpiralGalaxy_eqFunction_10133(data, threadData);
  SpiralGalaxy_eqFunction_3421(data, threadData);
  SpiralGalaxy_eqFunction_10132(data, threadData);
  SpiralGalaxy_eqFunction_3423(data, threadData);
  SpiralGalaxy_eqFunction_10131(data, threadData);
  SpiralGalaxy_eqFunction_3425(data, threadData);
  SpiralGalaxy_eqFunction_10144(data, threadData);
  SpiralGalaxy_eqFunction_10145(data, threadData);
  SpiralGalaxy_eqFunction_3428(data, threadData);
  SpiralGalaxy_eqFunction_3429(data, threadData);
  SpiralGalaxy_eqFunction_10146(data, threadData);
  SpiralGalaxy_eqFunction_10147(data, threadData);
  SpiralGalaxy_eqFunction_10150(data, threadData);
  SpiralGalaxy_eqFunction_10149(data, threadData);
  SpiralGalaxy_eqFunction_10148(data, threadData);
  SpiralGalaxy_eqFunction_3435(data, threadData);
  SpiralGalaxy_eqFunction_10143(data, threadData);
  SpiralGalaxy_eqFunction_3437(data, threadData);
  SpiralGalaxy_eqFunction_10142(data, threadData);
  SpiralGalaxy_eqFunction_3439(data, threadData);
  SpiralGalaxy_eqFunction_10141(data, threadData);
  SpiralGalaxy_eqFunction_3441(data, threadData);
  SpiralGalaxy_eqFunction_10154(data, threadData);
  SpiralGalaxy_eqFunction_10155(data, threadData);
  SpiralGalaxy_eqFunction_3444(data, threadData);
  SpiralGalaxy_eqFunction_3445(data, threadData);
  SpiralGalaxy_eqFunction_10156(data, threadData);
  SpiralGalaxy_eqFunction_10157(data, threadData);
  SpiralGalaxy_eqFunction_10160(data, threadData);
  SpiralGalaxy_eqFunction_10159(data, threadData);
  SpiralGalaxy_eqFunction_10158(data, threadData);
  SpiralGalaxy_eqFunction_3451(data, threadData);
  SpiralGalaxy_eqFunction_10153(data, threadData);
  SpiralGalaxy_eqFunction_3453(data, threadData);
  SpiralGalaxy_eqFunction_10152(data, threadData);
  SpiralGalaxy_eqFunction_3455(data, threadData);
  SpiralGalaxy_eqFunction_10151(data, threadData);
  SpiralGalaxy_eqFunction_3457(data, threadData);
  SpiralGalaxy_eqFunction_10164(data, threadData);
  SpiralGalaxy_eqFunction_10165(data, threadData);
  SpiralGalaxy_eqFunction_3460(data, threadData);
  SpiralGalaxy_eqFunction_3461(data, threadData);
  SpiralGalaxy_eqFunction_10166(data, threadData);
  SpiralGalaxy_eqFunction_10167(data, threadData);
  SpiralGalaxy_eqFunction_10170(data, threadData);
  SpiralGalaxy_eqFunction_10169(data, threadData);
  SpiralGalaxy_eqFunction_10168(data, threadData);
  SpiralGalaxy_eqFunction_3467(data, threadData);
  SpiralGalaxy_eqFunction_10163(data, threadData);
  SpiralGalaxy_eqFunction_3469(data, threadData);
  SpiralGalaxy_eqFunction_10162(data, threadData);
  SpiralGalaxy_eqFunction_3471(data, threadData);
  SpiralGalaxy_eqFunction_10161(data, threadData);
  SpiralGalaxy_eqFunction_3473(data, threadData);
  SpiralGalaxy_eqFunction_10174(data, threadData);
  SpiralGalaxy_eqFunction_10175(data, threadData);
  SpiralGalaxy_eqFunction_3476(data, threadData);
  SpiralGalaxy_eqFunction_3477(data, threadData);
  SpiralGalaxy_eqFunction_10176(data, threadData);
  SpiralGalaxy_eqFunction_10177(data, threadData);
  SpiralGalaxy_eqFunction_10180(data, threadData);
  SpiralGalaxy_eqFunction_10179(data, threadData);
  SpiralGalaxy_eqFunction_10178(data, threadData);
  SpiralGalaxy_eqFunction_3483(data, threadData);
  SpiralGalaxy_eqFunction_10173(data, threadData);
  SpiralGalaxy_eqFunction_3485(data, threadData);
  SpiralGalaxy_eqFunction_10172(data, threadData);
  SpiralGalaxy_eqFunction_3487(data, threadData);
  SpiralGalaxy_eqFunction_10171(data, threadData);
  SpiralGalaxy_eqFunction_3489(data, threadData);
  SpiralGalaxy_eqFunction_10184(data, threadData);
  SpiralGalaxy_eqFunction_10185(data, threadData);
  SpiralGalaxy_eqFunction_3492(data, threadData);
  SpiralGalaxy_eqFunction_3493(data, threadData);
  SpiralGalaxy_eqFunction_10186(data, threadData);
  SpiralGalaxy_eqFunction_10187(data, threadData);
  SpiralGalaxy_eqFunction_10190(data, threadData);
  SpiralGalaxy_eqFunction_10189(data, threadData);
  SpiralGalaxy_eqFunction_10188(data, threadData);
  SpiralGalaxy_eqFunction_3499(data, threadData);
  SpiralGalaxy_eqFunction_10183(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif