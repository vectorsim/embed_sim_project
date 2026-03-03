#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 3501
type: SIMPLE_ASSIGN
vy[219] = cos(theta[219]) * r_init[219] * omega_c[219]
*/
void SpiralGalaxy_eqFunction_3501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[718]] /* vy[219] STATE(1) */) = (cos((data->simulationInfo->realParameter[1725] /* theta[219] PARAM */))) * (((data->simulationInfo->realParameter[1224] /* r_init[219] PARAM */)) * ((data->simulationInfo->realParameter[723] /* omega_c[219] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10182(DATA *data, threadData_t *threadData);


/*
equation index: 3503
type: SIMPLE_ASSIGN
vz[219] = 0.0
*/
void SpiralGalaxy_eqFunction_3503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3503};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1218]] /* vz[219] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10181(DATA *data, threadData_t *threadData);


/*
equation index: 3505
type: SIMPLE_ASSIGN
z[220] = -0.0048000000000000004
*/
void SpiralGalaxy_eqFunction_3505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3505};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2719]] /* z[220] STATE(1,vz[220]) */) = -0.0048000000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10194(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10195(DATA *data, threadData_t *threadData);


/*
equation index: 3508
type: SIMPLE_ASSIGN
y[220] = r_init[220] * sin(theta[220] - 0.0012)
*/
void SpiralGalaxy_eqFunction_3508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3508};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2219]] /* y[220] STATE(1,vy[220]) */) = ((data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */)) * (sin((data->simulationInfo->realParameter[1726] /* theta[220] PARAM */) - 0.0012));
  TRACE_POP
}

/*
equation index: 3509
type: SIMPLE_ASSIGN
x[220] = r_init[220] * cos(theta[220] - 0.0012)
*/
void SpiralGalaxy_eqFunction_3509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1719]] /* x[220] STATE(1,vx[220]) */) = ((data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */)) * (cos((data->simulationInfo->realParameter[1726] /* theta[220] PARAM */) - 0.0012));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10196(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10197(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10200(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10199(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10198(DATA *data, threadData_t *threadData);


/*
equation index: 3515
type: SIMPLE_ASSIGN
vx[220] = (-sin(theta[220])) * r_init[220] * omega_c[220]
*/
void SpiralGalaxy_eqFunction_3515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3515};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[219]] /* vx[220] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1726] /* theta[220] PARAM */)))) * (((data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */)) * ((data->simulationInfo->realParameter[724] /* omega_c[220] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10193(DATA *data, threadData_t *threadData);


/*
equation index: 3517
type: SIMPLE_ASSIGN
vy[220] = cos(theta[220]) * r_init[220] * omega_c[220]
*/
void SpiralGalaxy_eqFunction_3517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[719]] /* vy[220] STATE(1) */) = (cos((data->simulationInfo->realParameter[1726] /* theta[220] PARAM */))) * (((data->simulationInfo->realParameter[1225] /* r_init[220] PARAM */)) * ((data->simulationInfo->realParameter[724] /* omega_c[220] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10192(DATA *data, threadData_t *threadData);


/*
equation index: 3519
type: SIMPLE_ASSIGN
vz[220] = 0.0
*/
void SpiralGalaxy_eqFunction_3519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3519};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1219]] /* vz[220] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10191(DATA *data, threadData_t *threadData);


/*
equation index: 3521
type: SIMPLE_ASSIGN
z[221] = -0.00464
*/
void SpiralGalaxy_eqFunction_3521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3521};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2720]] /* z[221] STATE(1,vz[221]) */) = -0.00464;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10204(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10205(DATA *data, threadData_t *threadData);


/*
equation index: 3524
type: SIMPLE_ASSIGN
y[221] = r_init[221] * sin(theta[221] - 0.00116)
*/
void SpiralGalaxy_eqFunction_3524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3524};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2220]] /* y[221] STATE(1,vy[221]) */) = ((data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */)) * (sin((data->simulationInfo->realParameter[1727] /* theta[221] PARAM */) - 0.00116));
  TRACE_POP
}

/*
equation index: 3525
type: SIMPLE_ASSIGN
x[221] = r_init[221] * cos(theta[221] - 0.00116)
*/
void SpiralGalaxy_eqFunction_3525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1720]] /* x[221] STATE(1,vx[221]) */) = ((data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */)) * (cos((data->simulationInfo->realParameter[1727] /* theta[221] PARAM */) - 0.00116));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10206(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10207(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10210(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10209(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10208(DATA *data, threadData_t *threadData);


/*
equation index: 3531
type: SIMPLE_ASSIGN
vx[221] = (-sin(theta[221])) * r_init[221] * omega_c[221]
*/
void SpiralGalaxy_eqFunction_3531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3531};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[220]] /* vx[221] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1727] /* theta[221] PARAM */)))) * (((data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */)) * ((data->simulationInfo->realParameter[725] /* omega_c[221] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10203(DATA *data, threadData_t *threadData);


/*
equation index: 3533
type: SIMPLE_ASSIGN
vy[221] = cos(theta[221]) * r_init[221] * omega_c[221]
*/
void SpiralGalaxy_eqFunction_3533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[720]] /* vy[221] STATE(1) */) = (cos((data->simulationInfo->realParameter[1727] /* theta[221] PARAM */))) * (((data->simulationInfo->realParameter[1226] /* r_init[221] PARAM */)) * ((data->simulationInfo->realParameter[725] /* omega_c[221] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10202(DATA *data, threadData_t *threadData);


/*
equation index: 3535
type: SIMPLE_ASSIGN
vz[221] = 0.0
*/
void SpiralGalaxy_eqFunction_3535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3535};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1220]] /* vz[221] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10201(DATA *data, threadData_t *threadData);


/*
equation index: 3537
type: SIMPLE_ASSIGN
z[222] = -0.0044800000000000005
*/
void SpiralGalaxy_eqFunction_3537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3537};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2721]] /* z[222] STATE(1,vz[222]) */) = -0.0044800000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10214(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10215(DATA *data, threadData_t *threadData);


/*
equation index: 3540
type: SIMPLE_ASSIGN
y[222] = r_init[222] * sin(theta[222] - 0.00112)
*/
void SpiralGalaxy_eqFunction_3540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3540};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2221]] /* y[222] STATE(1,vy[222]) */) = ((data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */)) * (sin((data->simulationInfo->realParameter[1728] /* theta[222] PARAM */) - 0.00112));
  TRACE_POP
}

/*
equation index: 3541
type: SIMPLE_ASSIGN
x[222] = r_init[222] * cos(theta[222] - 0.00112)
*/
void SpiralGalaxy_eqFunction_3541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1721]] /* x[222] STATE(1,vx[222]) */) = ((data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */)) * (cos((data->simulationInfo->realParameter[1728] /* theta[222] PARAM */) - 0.00112));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10216(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10217(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10220(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10219(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10218(DATA *data, threadData_t *threadData);


/*
equation index: 3547
type: SIMPLE_ASSIGN
vx[222] = (-sin(theta[222])) * r_init[222] * omega_c[222]
*/
void SpiralGalaxy_eqFunction_3547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3547};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[221]] /* vx[222] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1728] /* theta[222] PARAM */)))) * (((data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */)) * ((data->simulationInfo->realParameter[726] /* omega_c[222] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10213(DATA *data, threadData_t *threadData);


/*
equation index: 3549
type: SIMPLE_ASSIGN
vy[222] = cos(theta[222]) * r_init[222] * omega_c[222]
*/
void SpiralGalaxy_eqFunction_3549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[721]] /* vy[222] STATE(1) */) = (cos((data->simulationInfo->realParameter[1728] /* theta[222] PARAM */))) * (((data->simulationInfo->realParameter[1227] /* r_init[222] PARAM */)) * ((data->simulationInfo->realParameter[726] /* omega_c[222] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10212(DATA *data, threadData_t *threadData);


/*
equation index: 3551
type: SIMPLE_ASSIGN
vz[222] = 0.0
*/
void SpiralGalaxy_eqFunction_3551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3551};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1221]] /* vz[222] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10211(DATA *data, threadData_t *threadData);


/*
equation index: 3553
type: SIMPLE_ASSIGN
z[223] = -0.00432
*/
void SpiralGalaxy_eqFunction_3553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3553};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2722]] /* z[223] STATE(1,vz[223]) */) = -0.00432;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10224(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10225(DATA *data, threadData_t *threadData);


/*
equation index: 3556
type: SIMPLE_ASSIGN
y[223] = r_init[223] * sin(theta[223] - 0.0010799999999999998)
*/
void SpiralGalaxy_eqFunction_3556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3556};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2222]] /* y[223] STATE(1,vy[223]) */) = ((data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */)) * (sin((data->simulationInfo->realParameter[1729] /* theta[223] PARAM */) - 0.0010799999999999998));
  TRACE_POP
}

/*
equation index: 3557
type: SIMPLE_ASSIGN
x[223] = r_init[223] * cos(theta[223] - 0.0010799999999999998)
*/
void SpiralGalaxy_eqFunction_3557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1722]] /* x[223] STATE(1,vx[223]) */) = ((data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */)) * (cos((data->simulationInfo->realParameter[1729] /* theta[223] PARAM */) - 0.0010799999999999998));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10226(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10227(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10230(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10229(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10228(DATA *data, threadData_t *threadData);


/*
equation index: 3563
type: SIMPLE_ASSIGN
vx[223] = (-sin(theta[223])) * r_init[223] * omega_c[223]
*/
void SpiralGalaxy_eqFunction_3563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3563};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[222]] /* vx[223] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1729] /* theta[223] PARAM */)))) * (((data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */)) * ((data->simulationInfo->realParameter[727] /* omega_c[223] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10223(DATA *data, threadData_t *threadData);


/*
equation index: 3565
type: SIMPLE_ASSIGN
vy[223] = cos(theta[223]) * r_init[223] * omega_c[223]
*/
void SpiralGalaxy_eqFunction_3565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[722]] /* vy[223] STATE(1) */) = (cos((data->simulationInfo->realParameter[1729] /* theta[223] PARAM */))) * (((data->simulationInfo->realParameter[1228] /* r_init[223] PARAM */)) * ((data->simulationInfo->realParameter[727] /* omega_c[223] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10222(DATA *data, threadData_t *threadData);


/*
equation index: 3567
type: SIMPLE_ASSIGN
vz[223] = 0.0
*/
void SpiralGalaxy_eqFunction_3567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3567};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1222]] /* vz[223] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10221(DATA *data, threadData_t *threadData);


/*
equation index: 3569
type: SIMPLE_ASSIGN
z[224] = -0.0041600000000000005
*/
void SpiralGalaxy_eqFunction_3569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3569};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2723]] /* z[224] STATE(1,vz[224]) */) = -0.0041600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10234(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10235(DATA *data, threadData_t *threadData);


/*
equation index: 3572
type: SIMPLE_ASSIGN
y[224] = r_init[224] * sin(theta[224] - 0.00104)
*/
void SpiralGalaxy_eqFunction_3572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3572};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2223]] /* y[224] STATE(1,vy[224]) */) = ((data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */)) * (sin((data->simulationInfo->realParameter[1730] /* theta[224] PARAM */) - 0.00104));
  TRACE_POP
}

/*
equation index: 3573
type: SIMPLE_ASSIGN
x[224] = r_init[224] * cos(theta[224] - 0.00104)
*/
void SpiralGalaxy_eqFunction_3573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1723]] /* x[224] STATE(1,vx[224]) */) = ((data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */)) * (cos((data->simulationInfo->realParameter[1730] /* theta[224] PARAM */) - 0.00104));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10236(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10237(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10240(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10239(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10238(DATA *data, threadData_t *threadData);


/*
equation index: 3579
type: SIMPLE_ASSIGN
vx[224] = (-sin(theta[224])) * r_init[224] * omega_c[224]
*/
void SpiralGalaxy_eqFunction_3579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3579};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[223]] /* vx[224] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1730] /* theta[224] PARAM */)))) * (((data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */)) * ((data->simulationInfo->realParameter[728] /* omega_c[224] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10233(DATA *data, threadData_t *threadData);


/*
equation index: 3581
type: SIMPLE_ASSIGN
vy[224] = cos(theta[224]) * r_init[224] * omega_c[224]
*/
void SpiralGalaxy_eqFunction_3581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[723]] /* vy[224] STATE(1) */) = (cos((data->simulationInfo->realParameter[1730] /* theta[224] PARAM */))) * (((data->simulationInfo->realParameter[1229] /* r_init[224] PARAM */)) * ((data->simulationInfo->realParameter[728] /* omega_c[224] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10232(DATA *data, threadData_t *threadData);


/*
equation index: 3583
type: SIMPLE_ASSIGN
vz[224] = 0.0
*/
void SpiralGalaxy_eqFunction_3583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3583};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1223]] /* vz[224] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10231(DATA *data, threadData_t *threadData);


/*
equation index: 3585
type: SIMPLE_ASSIGN
z[225] = -0.004
*/
void SpiralGalaxy_eqFunction_3585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3585};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2724]] /* z[225] STATE(1,vz[225]) */) = -0.004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10244(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10245(DATA *data, threadData_t *threadData);


/*
equation index: 3588
type: SIMPLE_ASSIGN
y[225] = r_init[225] * sin(theta[225] - 9.999999999999998e-4)
*/
void SpiralGalaxy_eqFunction_3588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3588};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2224]] /* y[225] STATE(1,vy[225]) */) = ((data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */)) * (sin((data->simulationInfo->realParameter[1731] /* theta[225] PARAM */) - 9.999999999999998e-4));
  TRACE_POP
}

/*
equation index: 3589
type: SIMPLE_ASSIGN
x[225] = r_init[225] * cos(theta[225] - 9.999999999999998e-4)
*/
void SpiralGalaxy_eqFunction_3589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1724]] /* x[225] STATE(1,vx[225]) */) = ((data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */)) * (cos((data->simulationInfo->realParameter[1731] /* theta[225] PARAM */) - 9.999999999999998e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10246(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10247(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10250(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10249(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10248(DATA *data, threadData_t *threadData);


/*
equation index: 3595
type: SIMPLE_ASSIGN
vx[225] = (-sin(theta[225])) * r_init[225] * omega_c[225]
*/
void SpiralGalaxy_eqFunction_3595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3595};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[224]] /* vx[225] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1731] /* theta[225] PARAM */)))) * (((data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */)) * ((data->simulationInfo->realParameter[729] /* omega_c[225] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10243(DATA *data, threadData_t *threadData);


/*
equation index: 3597
type: SIMPLE_ASSIGN
vy[225] = cos(theta[225]) * r_init[225] * omega_c[225]
*/
void SpiralGalaxy_eqFunction_3597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[724]] /* vy[225] STATE(1) */) = (cos((data->simulationInfo->realParameter[1731] /* theta[225] PARAM */))) * (((data->simulationInfo->realParameter[1230] /* r_init[225] PARAM */)) * ((data->simulationInfo->realParameter[729] /* omega_c[225] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10242(DATA *data, threadData_t *threadData);


/*
equation index: 3599
type: SIMPLE_ASSIGN
vz[225] = 0.0
*/
void SpiralGalaxy_eqFunction_3599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3599};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1224]] /* vz[225] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10241(DATA *data, threadData_t *threadData);


/*
equation index: 3601
type: SIMPLE_ASSIGN
z[226] = -0.00384
*/
void SpiralGalaxy_eqFunction_3601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3601};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2725]] /* z[226] STATE(1,vz[226]) */) = -0.00384;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10254(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10255(DATA *data, threadData_t *threadData);


/*
equation index: 3604
type: SIMPLE_ASSIGN
y[226] = r_init[226] * sin(theta[226] - 9.599999999999998e-4)
*/
void SpiralGalaxy_eqFunction_3604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3604};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2225]] /* y[226] STATE(1,vy[226]) */) = ((data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */)) * (sin((data->simulationInfo->realParameter[1732] /* theta[226] PARAM */) - 9.599999999999998e-4));
  TRACE_POP
}

/*
equation index: 3605
type: SIMPLE_ASSIGN
x[226] = r_init[226] * cos(theta[226] - 9.599999999999998e-4)
*/
void SpiralGalaxy_eqFunction_3605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1725]] /* x[226] STATE(1,vx[226]) */) = ((data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */)) * (cos((data->simulationInfo->realParameter[1732] /* theta[226] PARAM */) - 9.599999999999998e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10256(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10257(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10260(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10259(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10258(DATA *data, threadData_t *threadData);


/*
equation index: 3611
type: SIMPLE_ASSIGN
vx[226] = (-sin(theta[226])) * r_init[226] * omega_c[226]
*/
void SpiralGalaxy_eqFunction_3611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3611};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[225]] /* vx[226] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1732] /* theta[226] PARAM */)))) * (((data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */)) * ((data->simulationInfo->realParameter[730] /* omega_c[226] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10253(DATA *data, threadData_t *threadData);


/*
equation index: 3613
type: SIMPLE_ASSIGN
vy[226] = cos(theta[226]) * r_init[226] * omega_c[226]
*/
void SpiralGalaxy_eqFunction_3613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[725]] /* vy[226] STATE(1) */) = (cos((data->simulationInfo->realParameter[1732] /* theta[226] PARAM */))) * (((data->simulationInfo->realParameter[1231] /* r_init[226] PARAM */)) * ((data->simulationInfo->realParameter[730] /* omega_c[226] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10252(DATA *data, threadData_t *threadData);


/*
equation index: 3615
type: SIMPLE_ASSIGN
vz[226] = 0.0
*/
void SpiralGalaxy_eqFunction_3615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3615};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1225]] /* vz[226] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10251(DATA *data, threadData_t *threadData);


/*
equation index: 3617
type: SIMPLE_ASSIGN
z[227] = -0.00368
*/
void SpiralGalaxy_eqFunction_3617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3617};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2726]] /* z[227] STATE(1,vz[227]) */) = -0.00368;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10264(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10265(DATA *data, threadData_t *threadData);


/*
equation index: 3620
type: SIMPLE_ASSIGN
y[227] = r_init[227] * sin(theta[227] - 9.199999999999997e-4)
*/
void SpiralGalaxy_eqFunction_3620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3620};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2226]] /* y[227] STATE(1,vy[227]) */) = ((data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */)) * (sin((data->simulationInfo->realParameter[1733] /* theta[227] PARAM */) - 9.199999999999997e-4));
  TRACE_POP
}

/*
equation index: 3621
type: SIMPLE_ASSIGN
x[227] = r_init[227] * cos(theta[227] - 9.199999999999997e-4)
*/
void SpiralGalaxy_eqFunction_3621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1726]] /* x[227] STATE(1,vx[227]) */) = ((data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */)) * (cos((data->simulationInfo->realParameter[1733] /* theta[227] PARAM */) - 9.199999999999997e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10266(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10267(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10270(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10269(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10268(DATA *data, threadData_t *threadData);


/*
equation index: 3627
type: SIMPLE_ASSIGN
vx[227] = (-sin(theta[227])) * r_init[227] * omega_c[227]
*/
void SpiralGalaxy_eqFunction_3627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3627};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[226]] /* vx[227] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1733] /* theta[227] PARAM */)))) * (((data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */)) * ((data->simulationInfo->realParameter[731] /* omega_c[227] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10263(DATA *data, threadData_t *threadData);


/*
equation index: 3629
type: SIMPLE_ASSIGN
vy[227] = cos(theta[227]) * r_init[227] * omega_c[227]
*/
void SpiralGalaxy_eqFunction_3629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[726]] /* vy[227] STATE(1) */) = (cos((data->simulationInfo->realParameter[1733] /* theta[227] PARAM */))) * (((data->simulationInfo->realParameter[1232] /* r_init[227] PARAM */)) * ((data->simulationInfo->realParameter[731] /* omega_c[227] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10262(DATA *data, threadData_t *threadData);


/*
equation index: 3631
type: SIMPLE_ASSIGN
vz[227] = 0.0
*/
void SpiralGalaxy_eqFunction_3631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3631};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1226]] /* vz[227] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10261(DATA *data, threadData_t *threadData);


/*
equation index: 3633
type: SIMPLE_ASSIGN
z[228] = -0.00352
*/
void SpiralGalaxy_eqFunction_3633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3633};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2727]] /* z[228] STATE(1,vz[228]) */) = -0.00352;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10274(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10275(DATA *data, threadData_t *threadData);


/*
equation index: 3636
type: SIMPLE_ASSIGN
y[228] = r_init[228] * sin(theta[228] - 8.799999999999997e-4)
*/
void SpiralGalaxy_eqFunction_3636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3636};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2227]] /* y[228] STATE(1,vy[228]) */) = ((data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */)) * (sin((data->simulationInfo->realParameter[1734] /* theta[228] PARAM */) - 8.799999999999997e-4));
  TRACE_POP
}

/*
equation index: 3637
type: SIMPLE_ASSIGN
x[228] = r_init[228] * cos(theta[228] - 8.799999999999997e-4)
*/
void SpiralGalaxy_eqFunction_3637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1727]] /* x[228] STATE(1,vx[228]) */) = ((data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */)) * (cos((data->simulationInfo->realParameter[1734] /* theta[228] PARAM */) - 8.799999999999997e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10276(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10277(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10280(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10279(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10278(DATA *data, threadData_t *threadData);


/*
equation index: 3643
type: SIMPLE_ASSIGN
vx[228] = (-sin(theta[228])) * r_init[228] * omega_c[228]
*/
void SpiralGalaxy_eqFunction_3643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3643};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[227]] /* vx[228] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1734] /* theta[228] PARAM */)))) * (((data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */)) * ((data->simulationInfo->realParameter[732] /* omega_c[228] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10273(DATA *data, threadData_t *threadData);


/*
equation index: 3645
type: SIMPLE_ASSIGN
vy[228] = cos(theta[228]) * r_init[228] * omega_c[228]
*/
void SpiralGalaxy_eqFunction_3645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[727]] /* vy[228] STATE(1) */) = (cos((data->simulationInfo->realParameter[1734] /* theta[228] PARAM */))) * (((data->simulationInfo->realParameter[1233] /* r_init[228] PARAM */)) * ((data->simulationInfo->realParameter[732] /* omega_c[228] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10272(DATA *data, threadData_t *threadData);


/*
equation index: 3647
type: SIMPLE_ASSIGN
vz[228] = 0.0
*/
void SpiralGalaxy_eqFunction_3647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3647};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1227]] /* vz[228] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10271(DATA *data, threadData_t *threadData);


/*
equation index: 3649
type: SIMPLE_ASSIGN
z[229] = -0.00336
*/
void SpiralGalaxy_eqFunction_3649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3649};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2728]] /* z[229] STATE(1,vz[229]) */) = -0.00336;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10284(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10285(DATA *data, threadData_t *threadData);


/*
equation index: 3652
type: SIMPLE_ASSIGN
y[229] = r_init[229] * sin(theta[229] - 8.399999999999996e-4)
*/
void SpiralGalaxy_eqFunction_3652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3652};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2228]] /* y[229] STATE(1,vy[229]) */) = ((data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */)) * (sin((data->simulationInfo->realParameter[1735] /* theta[229] PARAM */) - 8.399999999999996e-4));
  TRACE_POP
}

/*
equation index: 3653
type: SIMPLE_ASSIGN
x[229] = r_init[229] * cos(theta[229] - 8.399999999999996e-4)
*/
void SpiralGalaxy_eqFunction_3653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1728]] /* x[229] STATE(1,vx[229]) */) = ((data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */)) * (cos((data->simulationInfo->realParameter[1735] /* theta[229] PARAM */) - 8.399999999999996e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10286(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10287(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10290(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10289(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10288(DATA *data, threadData_t *threadData);


/*
equation index: 3659
type: SIMPLE_ASSIGN
vx[229] = (-sin(theta[229])) * r_init[229] * omega_c[229]
*/
void SpiralGalaxy_eqFunction_3659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3659};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[228]] /* vx[229] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1735] /* theta[229] PARAM */)))) * (((data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */)) * ((data->simulationInfo->realParameter[733] /* omega_c[229] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10283(DATA *data, threadData_t *threadData);


/*
equation index: 3661
type: SIMPLE_ASSIGN
vy[229] = cos(theta[229]) * r_init[229] * omega_c[229]
*/
void SpiralGalaxy_eqFunction_3661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[728]] /* vy[229] STATE(1) */) = (cos((data->simulationInfo->realParameter[1735] /* theta[229] PARAM */))) * (((data->simulationInfo->realParameter[1234] /* r_init[229] PARAM */)) * ((data->simulationInfo->realParameter[733] /* omega_c[229] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10282(DATA *data, threadData_t *threadData);


/*
equation index: 3663
type: SIMPLE_ASSIGN
vz[229] = 0.0
*/
void SpiralGalaxy_eqFunction_3663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3663};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1228]] /* vz[229] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10281(DATA *data, threadData_t *threadData);


/*
equation index: 3665
type: SIMPLE_ASSIGN
z[230] = -0.0032
*/
void SpiralGalaxy_eqFunction_3665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3665};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2729]] /* z[230] STATE(1,vz[230]) */) = -0.0032;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10294(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10295(DATA *data, threadData_t *threadData);


/*
equation index: 3668
type: SIMPLE_ASSIGN
y[230] = r_init[230] * sin(theta[230] - 7.999999999999996e-4)
*/
void SpiralGalaxy_eqFunction_3668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3668};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2229]] /* y[230] STATE(1,vy[230]) */) = ((data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */)) * (sin((data->simulationInfo->realParameter[1736] /* theta[230] PARAM */) - 7.999999999999996e-4));
  TRACE_POP
}

/*
equation index: 3669
type: SIMPLE_ASSIGN
x[230] = r_init[230] * cos(theta[230] - 7.999999999999996e-4)
*/
void SpiralGalaxy_eqFunction_3669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1729]] /* x[230] STATE(1,vx[230]) */) = ((data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */)) * (cos((data->simulationInfo->realParameter[1736] /* theta[230] PARAM */) - 7.999999999999996e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10296(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10297(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10300(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10299(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10298(DATA *data, threadData_t *threadData);


/*
equation index: 3675
type: SIMPLE_ASSIGN
vx[230] = (-sin(theta[230])) * r_init[230] * omega_c[230]
*/
void SpiralGalaxy_eqFunction_3675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3675};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[229]] /* vx[230] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1736] /* theta[230] PARAM */)))) * (((data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */)) * ((data->simulationInfo->realParameter[734] /* omega_c[230] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10293(DATA *data, threadData_t *threadData);


/*
equation index: 3677
type: SIMPLE_ASSIGN
vy[230] = cos(theta[230]) * r_init[230] * omega_c[230]
*/
void SpiralGalaxy_eqFunction_3677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[729]] /* vy[230] STATE(1) */) = (cos((data->simulationInfo->realParameter[1736] /* theta[230] PARAM */))) * (((data->simulationInfo->realParameter[1235] /* r_init[230] PARAM */)) * ((data->simulationInfo->realParameter[734] /* omega_c[230] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10292(DATA *data, threadData_t *threadData);


/*
equation index: 3679
type: SIMPLE_ASSIGN
vz[230] = 0.0
*/
void SpiralGalaxy_eqFunction_3679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3679};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1229]] /* vz[230] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10291(DATA *data, threadData_t *threadData);


/*
equation index: 3681
type: SIMPLE_ASSIGN
z[231] = -0.0030399999999999997
*/
void SpiralGalaxy_eqFunction_3681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3681};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2730]] /* z[231] STATE(1,vz[231]) */) = -0.0030399999999999997;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10304(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10305(DATA *data, threadData_t *threadData);


/*
equation index: 3684
type: SIMPLE_ASSIGN
y[231] = r_init[231] * sin(theta[231] - 7.599999999999996e-4)
*/
void SpiralGalaxy_eqFunction_3684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3684};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2230]] /* y[231] STATE(1,vy[231]) */) = ((data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */)) * (sin((data->simulationInfo->realParameter[1737] /* theta[231] PARAM */) - 7.599999999999996e-4));
  TRACE_POP
}

/*
equation index: 3685
type: SIMPLE_ASSIGN
x[231] = r_init[231] * cos(theta[231] - 7.599999999999996e-4)
*/
void SpiralGalaxy_eqFunction_3685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1730]] /* x[231] STATE(1,vx[231]) */) = ((data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */)) * (cos((data->simulationInfo->realParameter[1737] /* theta[231] PARAM */) - 7.599999999999996e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10306(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10307(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10310(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10309(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10308(DATA *data, threadData_t *threadData);


/*
equation index: 3691
type: SIMPLE_ASSIGN
vx[231] = (-sin(theta[231])) * r_init[231] * omega_c[231]
*/
void SpiralGalaxy_eqFunction_3691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3691};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[230]] /* vx[231] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1737] /* theta[231] PARAM */)))) * (((data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */)) * ((data->simulationInfo->realParameter[735] /* omega_c[231] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10303(DATA *data, threadData_t *threadData);


/*
equation index: 3693
type: SIMPLE_ASSIGN
vy[231] = cos(theta[231]) * r_init[231] * omega_c[231]
*/
void SpiralGalaxy_eqFunction_3693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[730]] /* vy[231] STATE(1) */) = (cos((data->simulationInfo->realParameter[1737] /* theta[231] PARAM */))) * (((data->simulationInfo->realParameter[1236] /* r_init[231] PARAM */)) * ((data->simulationInfo->realParameter[735] /* omega_c[231] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10302(DATA *data, threadData_t *threadData);


/*
equation index: 3695
type: SIMPLE_ASSIGN
vz[231] = 0.0
*/
void SpiralGalaxy_eqFunction_3695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3695};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1230]] /* vz[231] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10301(DATA *data, threadData_t *threadData);


/*
equation index: 3697
type: SIMPLE_ASSIGN
z[232] = -0.00288
*/
void SpiralGalaxy_eqFunction_3697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3697};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2731]] /* z[232] STATE(1,vz[232]) */) = -0.00288;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10314(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10315(DATA *data, threadData_t *threadData);


/*
equation index: 3700
type: SIMPLE_ASSIGN
y[232] = r_init[232] * sin(theta[232] - 7.199999999999995e-4)
*/
void SpiralGalaxy_eqFunction_3700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3700};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2231]] /* y[232] STATE(1,vy[232]) */) = ((data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */)) * (sin((data->simulationInfo->realParameter[1738] /* theta[232] PARAM */) - 7.199999999999995e-4));
  TRACE_POP
}

/*
equation index: 3701
type: SIMPLE_ASSIGN
x[232] = r_init[232] * cos(theta[232] - 7.199999999999995e-4)
*/
void SpiralGalaxy_eqFunction_3701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1731]] /* x[232] STATE(1,vx[232]) */) = ((data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */)) * (cos((data->simulationInfo->realParameter[1738] /* theta[232] PARAM */) - 7.199999999999995e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10316(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10317(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10320(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10319(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10318(DATA *data, threadData_t *threadData);


/*
equation index: 3707
type: SIMPLE_ASSIGN
vx[232] = (-sin(theta[232])) * r_init[232] * omega_c[232]
*/
void SpiralGalaxy_eqFunction_3707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3707};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[231]] /* vx[232] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1738] /* theta[232] PARAM */)))) * (((data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */)) * ((data->simulationInfo->realParameter[736] /* omega_c[232] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10313(DATA *data, threadData_t *threadData);


/*
equation index: 3709
type: SIMPLE_ASSIGN
vy[232] = cos(theta[232]) * r_init[232] * omega_c[232]
*/
void SpiralGalaxy_eqFunction_3709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[731]] /* vy[232] STATE(1) */) = (cos((data->simulationInfo->realParameter[1738] /* theta[232] PARAM */))) * (((data->simulationInfo->realParameter[1237] /* r_init[232] PARAM */)) * ((data->simulationInfo->realParameter[736] /* omega_c[232] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10312(DATA *data, threadData_t *threadData);


/*
equation index: 3711
type: SIMPLE_ASSIGN
vz[232] = 0.0
*/
void SpiralGalaxy_eqFunction_3711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3711};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1231]] /* vz[232] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10311(DATA *data, threadData_t *threadData);


/*
equation index: 3713
type: SIMPLE_ASSIGN
z[233] = -0.0027199999999999998
*/
void SpiralGalaxy_eqFunction_3713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3713};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2732]] /* z[233] STATE(1,vz[233]) */) = -0.0027199999999999998;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10324(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10325(DATA *data, threadData_t *threadData);


/*
equation index: 3716
type: SIMPLE_ASSIGN
y[233] = r_init[233] * sin(theta[233] - 6.799999999999995e-4)
*/
void SpiralGalaxy_eqFunction_3716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3716};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2232]] /* y[233] STATE(1,vy[233]) */) = ((data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */)) * (sin((data->simulationInfo->realParameter[1739] /* theta[233] PARAM */) - 6.799999999999995e-4));
  TRACE_POP
}

/*
equation index: 3717
type: SIMPLE_ASSIGN
x[233] = r_init[233] * cos(theta[233] - 6.799999999999995e-4)
*/
void SpiralGalaxy_eqFunction_3717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1732]] /* x[233] STATE(1,vx[233]) */) = ((data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */)) * (cos((data->simulationInfo->realParameter[1739] /* theta[233] PARAM */) - 6.799999999999995e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10326(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10327(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10330(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10329(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10328(DATA *data, threadData_t *threadData);


/*
equation index: 3723
type: SIMPLE_ASSIGN
vx[233] = (-sin(theta[233])) * r_init[233] * omega_c[233]
*/
void SpiralGalaxy_eqFunction_3723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3723};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[232]] /* vx[233] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1739] /* theta[233] PARAM */)))) * (((data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */)) * ((data->simulationInfo->realParameter[737] /* omega_c[233] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10323(DATA *data, threadData_t *threadData);


/*
equation index: 3725
type: SIMPLE_ASSIGN
vy[233] = cos(theta[233]) * r_init[233] * omega_c[233]
*/
void SpiralGalaxy_eqFunction_3725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[732]] /* vy[233] STATE(1) */) = (cos((data->simulationInfo->realParameter[1739] /* theta[233] PARAM */))) * (((data->simulationInfo->realParameter[1238] /* r_init[233] PARAM */)) * ((data->simulationInfo->realParameter[737] /* omega_c[233] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10322(DATA *data, threadData_t *threadData);


/*
equation index: 3727
type: SIMPLE_ASSIGN
vz[233] = 0.0
*/
void SpiralGalaxy_eqFunction_3727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3727};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1232]] /* vz[233] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10321(DATA *data, threadData_t *threadData);


/*
equation index: 3729
type: SIMPLE_ASSIGN
z[234] = -0.0025599999999999998
*/
void SpiralGalaxy_eqFunction_3729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3729};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2733]] /* z[234] STATE(1,vz[234]) */) = -0.0025599999999999998;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10334(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10335(DATA *data, threadData_t *threadData);


/*
equation index: 3732
type: SIMPLE_ASSIGN
y[234] = r_init[234] * sin(theta[234] - 6.399999999999995e-4)
*/
void SpiralGalaxy_eqFunction_3732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3732};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2233]] /* y[234] STATE(1,vy[234]) */) = ((data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */)) * (sin((data->simulationInfo->realParameter[1740] /* theta[234] PARAM */) - 6.399999999999995e-4));
  TRACE_POP
}

/*
equation index: 3733
type: SIMPLE_ASSIGN
x[234] = r_init[234] * cos(theta[234] - 6.399999999999995e-4)
*/
void SpiralGalaxy_eqFunction_3733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1733]] /* x[234] STATE(1,vx[234]) */) = ((data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */)) * (cos((data->simulationInfo->realParameter[1740] /* theta[234] PARAM */) - 6.399999999999995e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10336(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10337(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10340(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10339(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10338(DATA *data, threadData_t *threadData);


/*
equation index: 3739
type: SIMPLE_ASSIGN
vx[234] = (-sin(theta[234])) * r_init[234] * omega_c[234]
*/
void SpiralGalaxy_eqFunction_3739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3739};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[233]] /* vx[234] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1740] /* theta[234] PARAM */)))) * (((data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */)) * ((data->simulationInfo->realParameter[738] /* omega_c[234] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10333(DATA *data, threadData_t *threadData);


/*
equation index: 3741
type: SIMPLE_ASSIGN
vy[234] = cos(theta[234]) * r_init[234] * omega_c[234]
*/
void SpiralGalaxy_eqFunction_3741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[733]] /* vy[234] STATE(1) */) = (cos((data->simulationInfo->realParameter[1740] /* theta[234] PARAM */))) * (((data->simulationInfo->realParameter[1239] /* r_init[234] PARAM */)) * ((data->simulationInfo->realParameter[738] /* omega_c[234] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10332(DATA *data, threadData_t *threadData);


/*
equation index: 3743
type: SIMPLE_ASSIGN
vz[234] = 0.0
*/
void SpiralGalaxy_eqFunction_3743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3743};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1233]] /* vz[234] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10331(DATA *data, threadData_t *threadData);


/*
equation index: 3745
type: SIMPLE_ASSIGN
z[235] = -0.0024000000000000007
*/
void SpiralGalaxy_eqFunction_3745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3745};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2734]] /* z[235] STATE(1,vz[235]) */) = -0.0024000000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10344(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10345(DATA *data, threadData_t *threadData);


/*
equation index: 3748
type: SIMPLE_ASSIGN
y[235] = r_init[235] * sin(theta[235] - 6.000000000000006e-4)
*/
void SpiralGalaxy_eqFunction_3748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3748};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2234]] /* y[235] STATE(1,vy[235]) */) = ((data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */)) * (sin((data->simulationInfo->realParameter[1741] /* theta[235] PARAM */) - 6.000000000000006e-4));
  TRACE_POP
}

/*
equation index: 3749
type: SIMPLE_ASSIGN
x[235] = r_init[235] * cos(theta[235] - 6.000000000000006e-4)
*/
void SpiralGalaxy_eqFunction_3749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1734]] /* x[235] STATE(1,vx[235]) */) = ((data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */)) * (cos((data->simulationInfo->realParameter[1741] /* theta[235] PARAM */) - 6.000000000000006e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10346(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10347(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10350(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10349(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10348(DATA *data, threadData_t *threadData);


/*
equation index: 3755
type: SIMPLE_ASSIGN
vx[235] = (-sin(theta[235])) * r_init[235] * omega_c[235]
*/
void SpiralGalaxy_eqFunction_3755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3755};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[234]] /* vx[235] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1741] /* theta[235] PARAM */)))) * (((data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */)) * ((data->simulationInfo->realParameter[739] /* omega_c[235] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10343(DATA *data, threadData_t *threadData);


/*
equation index: 3757
type: SIMPLE_ASSIGN
vy[235] = cos(theta[235]) * r_init[235] * omega_c[235]
*/
void SpiralGalaxy_eqFunction_3757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[734]] /* vy[235] STATE(1) */) = (cos((data->simulationInfo->realParameter[1741] /* theta[235] PARAM */))) * (((data->simulationInfo->realParameter[1240] /* r_init[235] PARAM */)) * ((data->simulationInfo->realParameter[739] /* omega_c[235] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10342(DATA *data, threadData_t *threadData);


/*
equation index: 3759
type: SIMPLE_ASSIGN
vz[235] = 0.0
*/
void SpiralGalaxy_eqFunction_3759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3759};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1234]] /* vz[235] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10341(DATA *data, threadData_t *threadData);


/*
equation index: 3761
type: SIMPLE_ASSIGN
z[236] = -0.0022400000000000007
*/
void SpiralGalaxy_eqFunction_3761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3761};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2735]] /* z[236] STATE(1,vz[236]) */) = -0.0022400000000000007;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10354(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10355(DATA *data, threadData_t *threadData);


/*
equation index: 3764
type: SIMPLE_ASSIGN
y[236] = r_init[236] * sin(theta[236] - 5.600000000000005e-4)
*/
void SpiralGalaxy_eqFunction_3764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3764};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2235]] /* y[236] STATE(1,vy[236]) */) = ((data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */)) * (sin((data->simulationInfo->realParameter[1742] /* theta[236] PARAM */) - 5.600000000000005e-4));
  TRACE_POP
}

/*
equation index: 3765
type: SIMPLE_ASSIGN
x[236] = r_init[236] * cos(theta[236] - 5.600000000000005e-4)
*/
void SpiralGalaxy_eqFunction_3765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1735]] /* x[236] STATE(1,vx[236]) */) = ((data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */)) * (cos((data->simulationInfo->realParameter[1742] /* theta[236] PARAM */) - 5.600000000000005e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10356(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10357(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10360(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10359(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10358(DATA *data, threadData_t *threadData);


/*
equation index: 3771
type: SIMPLE_ASSIGN
vx[236] = (-sin(theta[236])) * r_init[236] * omega_c[236]
*/
void SpiralGalaxy_eqFunction_3771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3771};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[235]] /* vx[236] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1742] /* theta[236] PARAM */)))) * (((data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */)) * ((data->simulationInfo->realParameter[740] /* omega_c[236] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10353(DATA *data, threadData_t *threadData);


/*
equation index: 3773
type: SIMPLE_ASSIGN
vy[236] = cos(theta[236]) * r_init[236] * omega_c[236]
*/
void SpiralGalaxy_eqFunction_3773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[735]] /* vy[236] STATE(1) */) = (cos((data->simulationInfo->realParameter[1742] /* theta[236] PARAM */))) * (((data->simulationInfo->realParameter[1241] /* r_init[236] PARAM */)) * ((data->simulationInfo->realParameter[740] /* omega_c[236] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10352(DATA *data, threadData_t *threadData);


/*
equation index: 3775
type: SIMPLE_ASSIGN
vz[236] = 0.0
*/
void SpiralGalaxy_eqFunction_3775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3775};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1235]] /* vz[236] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10351(DATA *data, threadData_t *threadData);


/*
equation index: 3777
type: SIMPLE_ASSIGN
z[237] = -0.0020800000000000003
*/
void SpiralGalaxy_eqFunction_3777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3777};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2736]] /* z[237] STATE(1,vz[237]) */) = -0.0020800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10364(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10365(DATA *data, threadData_t *threadData);


/*
equation index: 3780
type: SIMPLE_ASSIGN
y[237] = r_init[237] * sin(theta[237] - 5.200000000000005e-4)
*/
void SpiralGalaxy_eqFunction_3780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3780};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2236]] /* y[237] STATE(1,vy[237]) */) = ((data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */)) * (sin((data->simulationInfo->realParameter[1743] /* theta[237] PARAM */) - 5.200000000000005e-4));
  TRACE_POP
}

/*
equation index: 3781
type: SIMPLE_ASSIGN
x[237] = r_init[237] * cos(theta[237] - 5.200000000000005e-4)
*/
void SpiralGalaxy_eqFunction_3781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1736]] /* x[237] STATE(1,vx[237]) */) = ((data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */)) * (cos((data->simulationInfo->realParameter[1743] /* theta[237] PARAM */) - 5.200000000000005e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10366(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10367(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10370(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10369(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10368(DATA *data, threadData_t *threadData);


/*
equation index: 3787
type: SIMPLE_ASSIGN
vx[237] = (-sin(theta[237])) * r_init[237] * omega_c[237]
*/
void SpiralGalaxy_eqFunction_3787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3787};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[236]] /* vx[237] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1743] /* theta[237] PARAM */)))) * (((data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */)) * ((data->simulationInfo->realParameter[741] /* omega_c[237] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10363(DATA *data, threadData_t *threadData);


/*
equation index: 3789
type: SIMPLE_ASSIGN
vy[237] = cos(theta[237]) * r_init[237] * omega_c[237]
*/
void SpiralGalaxy_eqFunction_3789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[736]] /* vy[237] STATE(1) */) = (cos((data->simulationInfo->realParameter[1743] /* theta[237] PARAM */))) * (((data->simulationInfo->realParameter[1242] /* r_init[237] PARAM */)) * ((data->simulationInfo->realParameter[741] /* omega_c[237] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10362(DATA *data, threadData_t *threadData);


/*
equation index: 3791
type: SIMPLE_ASSIGN
vz[237] = 0.0
*/
void SpiralGalaxy_eqFunction_3791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3791};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1236]] /* vz[237] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10361(DATA *data, threadData_t *threadData);


/*
equation index: 3793
type: SIMPLE_ASSIGN
z[238] = -0.0019200000000000003
*/
void SpiralGalaxy_eqFunction_3793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3793};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2737]] /* z[238] STATE(1,vz[238]) */) = -0.0019200000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10374(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10375(DATA *data, threadData_t *threadData);


/*
equation index: 3796
type: SIMPLE_ASSIGN
y[238] = r_init[238] * sin(theta[238] - 4.8000000000000045e-4)
*/
void SpiralGalaxy_eqFunction_3796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3796};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2237]] /* y[238] STATE(1,vy[238]) */) = ((data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */)) * (sin((data->simulationInfo->realParameter[1744] /* theta[238] PARAM */) - 4.8000000000000045e-4));
  TRACE_POP
}

/*
equation index: 3797
type: SIMPLE_ASSIGN
x[238] = r_init[238] * cos(theta[238] - 4.8000000000000045e-4)
*/
void SpiralGalaxy_eqFunction_3797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1737]] /* x[238] STATE(1,vx[238]) */) = ((data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */)) * (cos((data->simulationInfo->realParameter[1744] /* theta[238] PARAM */) - 4.8000000000000045e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10376(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10377(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10380(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10379(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10378(DATA *data, threadData_t *threadData);


/*
equation index: 3803
type: SIMPLE_ASSIGN
vx[238] = (-sin(theta[238])) * r_init[238] * omega_c[238]
*/
void SpiralGalaxy_eqFunction_3803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3803};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[237]] /* vx[238] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1744] /* theta[238] PARAM */)))) * (((data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */)) * ((data->simulationInfo->realParameter[742] /* omega_c[238] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10373(DATA *data, threadData_t *threadData);


/*
equation index: 3805
type: SIMPLE_ASSIGN
vy[238] = cos(theta[238]) * r_init[238] * omega_c[238]
*/
void SpiralGalaxy_eqFunction_3805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[737]] /* vy[238] STATE(1) */) = (cos((data->simulationInfo->realParameter[1744] /* theta[238] PARAM */))) * (((data->simulationInfo->realParameter[1243] /* r_init[238] PARAM */)) * ((data->simulationInfo->realParameter[742] /* omega_c[238] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10372(DATA *data, threadData_t *threadData);


/*
equation index: 3807
type: SIMPLE_ASSIGN
vz[238] = 0.0
*/
void SpiralGalaxy_eqFunction_3807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3807};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1237]] /* vz[238] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10371(DATA *data, threadData_t *threadData);


/*
equation index: 3809
type: SIMPLE_ASSIGN
z[239] = -0.0017600000000000003
*/
void SpiralGalaxy_eqFunction_3809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3809};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2738]] /* z[239] STATE(1,vz[239]) */) = -0.0017600000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10384(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10385(DATA *data, threadData_t *threadData);


/*
equation index: 3812
type: SIMPLE_ASSIGN
y[239] = r_init[239] * sin(theta[239] - 4.400000000000004e-4)
*/
void SpiralGalaxy_eqFunction_3812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3812};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2238]] /* y[239] STATE(1,vy[239]) */) = ((data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */)) * (sin((data->simulationInfo->realParameter[1745] /* theta[239] PARAM */) - 4.400000000000004e-4));
  TRACE_POP
}

/*
equation index: 3813
type: SIMPLE_ASSIGN
x[239] = r_init[239] * cos(theta[239] - 4.400000000000004e-4)
*/
void SpiralGalaxy_eqFunction_3813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1738]] /* x[239] STATE(1,vx[239]) */) = ((data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */)) * (cos((data->simulationInfo->realParameter[1745] /* theta[239] PARAM */) - 4.400000000000004e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10386(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10387(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10390(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10389(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10388(DATA *data, threadData_t *threadData);


/*
equation index: 3819
type: SIMPLE_ASSIGN
vx[239] = (-sin(theta[239])) * r_init[239] * omega_c[239]
*/
void SpiralGalaxy_eqFunction_3819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3819};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[238]] /* vx[239] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1745] /* theta[239] PARAM */)))) * (((data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */)) * ((data->simulationInfo->realParameter[743] /* omega_c[239] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10383(DATA *data, threadData_t *threadData);


/*
equation index: 3821
type: SIMPLE_ASSIGN
vy[239] = cos(theta[239]) * r_init[239] * omega_c[239]
*/
void SpiralGalaxy_eqFunction_3821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[738]] /* vy[239] STATE(1) */) = (cos((data->simulationInfo->realParameter[1745] /* theta[239] PARAM */))) * (((data->simulationInfo->realParameter[1244] /* r_init[239] PARAM */)) * ((data->simulationInfo->realParameter[743] /* omega_c[239] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10382(DATA *data, threadData_t *threadData);


/*
equation index: 3823
type: SIMPLE_ASSIGN
vz[239] = 0.0
*/
void SpiralGalaxy_eqFunction_3823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3823};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1238]] /* vz[239] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10381(DATA *data, threadData_t *threadData);


/*
equation index: 3825
type: SIMPLE_ASSIGN
z[240] = -0.0016000000000000003
*/
void SpiralGalaxy_eqFunction_3825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3825};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2739]] /* z[240] STATE(1,vz[240]) */) = -0.0016000000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10394(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10395(DATA *data, threadData_t *threadData);


/*
equation index: 3828
type: SIMPLE_ASSIGN
y[240] = r_init[240] * sin(theta[240] - 4.0000000000000034e-4)
*/
void SpiralGalaxy_eqFunction_3828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3828};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2239]] /* y[240] STATE(1,vy[240]) */) = ((data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */)) * (sin((data->simulationInfo->realParameter[1746] /* theta[240] PARAM */) - 4.0000000000000034e-4));
  TRACE_POP
}

/*
equation index: 3829
type: SIMPLE_ASSIGN
x[240] = r_init[240] * cos(theta[240] - 4.0000000000000034e-4)
*/
void SpiralGalaxy_eqFunction_3829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1739]] /* x[240] STATE(1,vx[240]) */) = ((data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */)) * (cos((data->simulationInfo->realParameter[1746] /* theta[240] PARAM */) - 4.0000000000000034e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10396(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10397(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10400(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10399(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10398(DATA *data, threadData_t *threadData);


/*
equation index: 3835
type: SIMPLE_ASSIGN
vx[240] = (-sin(theta[240])) * r_init[240] * omega_c[240]
*/
void SpiralGalaxy_eqFunction_3835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3835};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[239]] /* vx[240] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1746] /* theta[240] PARAM */)))) * (((data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */)) * ((data->simulationInfo->realParameter[744] /* omega_c[240] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10393(DATA *data, threadData_t *threadData);


/*
equation index: 3837
type: SIMPLE_ASSIGN
vy[240] = cos(theta[240]) * r_init[240] * omega_c[240]
*/
void SpiralGalaxy_eqFunction_3837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[739]] /* vy[240] STATE(1) */) = (cos((data->simulationInfo->realParameter[1746] /* theta[240] PARAM */))) * (((data->simulationInfo->realParameter[1245] /* r_init[240] PARAM */)) * ((data->simulationInfo->realParameter[744] /* omega_c[240] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10392(DATA *data, threadData_t *threadData);


/*
equation index: 3839
type: SIMPLE_ASSIGN
vz[240] = 0.0
*/
void SpiralGalaxy_eqFunction_3839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3839};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1239]] /* vz[240] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10391(DATA *data, threadData_t *threadData);


/*
equation index: 3841
type: SIMPLE_ASSIGN
z[241] = -0.0014400000000000005
*/
void SpiralGalaxy_eqFunction_3841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3841};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2740]] /* z[241] STATE(1,vz[241]) */) = -0.0014400000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10404(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10405(DATA *data, threadData_t *threadData);


/*
equation index: 3844
type: SIMPLE_ASSIGN
y[241] = r_init[241] * sin(theta[241] - 3.6000000000000035e-4)
*/
void SpiralGalaxy_eqFunction_3844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3844};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2240]] /* y[241] STATE(1,vy[241]) */) = ((data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */)) * (sin((data->simulationInfo->realParameter[1747] /* theta[241] PARAM */) - 3.6000000000000035e-4));
  TRACE_POP
}

/*
equation index: 3845
type: SIMPLE_ASSIGN
x[241] = r_init[241] * cos(theta[241] - 3.6000000000000035e-4)
*/
void SpiralGalaxy_eqFunction_3845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1740]] /* x[241] STATE(1,vx[241]) */) = ((data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */)) * (cos((data->simulationInfo->realParameter[1747] /* theta[241] PARAM */) - 3.6000000000000035e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10406(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10407(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10410(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10409(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10408(DATA *data, threadData_t *threadData);


/*
equation index: 3851
type: SIMPLE_ASSIGN
vx[241] = (-sin(theta[241])) * r_init[241] * omega_c[241]
*/
void SpiralGalaxy_eqFunction_3851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3851};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[240]] /* vx[241] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1747] /* theta[241] PARAM */)))) * (((data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */)) * ((data->simulationInfo->realParameter[745] /* omega_c[241] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10403(DATA *data, threadData_t *threadData);


/*
equation index: 3853
type: SIMPLE_ASSIGN
vy[241] = cos(theta[241]) * r_init[241] * omega_c[241]
*/
void SpiralGalaxy_eqFunction_3853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[740]] /* vy[241] STATE(1) */) = (cos((data->simulationInfo->realParameter[1747] /* theta[241] PARAM */))) * (((data->simulationInfo->realParameter[1246] /* r_init[241] PARAM */)) * ((data->simulationInfo->realParameter[745] /* omega_c[241] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10402(DATA *data, threadData_t *threadData);


/*
equation index: 3855
type: SIMPLE_ASSIGN
vz[241] = 0.0
*/
void SpiralGalaxy_eqFunction_3855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3855};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1240]] /* vz[241] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10401(DATA *data, threadData_t *threadData);


/*
equation index: 3857
type: SIMPLE_ASSIGN
z[242] = -0.0012800000000000003
*/
void SpiralGalaxy_eqFunction_3857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3857};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2741]] /* z[242] STATE(1,vz[242]) */) = -0.0012800000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10414(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10415(DATA *data, threadData_t *threadData);


/*
equation index: 3860
type: SIMPLE_ASSIGN
y[242] = r_init[242] * sin(theta[242] - 3.200000000000003e-4)
*/
void SpiralGalaxy_eqFunction_3860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3860};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2241]] /* y[242] STATE(1,vy[242]) */) = ((data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */)) * (sin((data->simulationInfo->realParameter[1748] /* theta[242] PARAM */) - 3.200000000000003e-4));
  TRACE_POP
}

/*
equation index: 3861
type: SIMPLE_ASSIGN
x[242] = r_init[242] * cos(theta[242] - 3.200000000000003e-4)
*/
void SpiralGalaxy_eqFunction_3861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1741]] /* x[242] STATE(1,vx[242]) */) = ((data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */)) * (cos((data->simulationInfo->realParameter[1748] /* theta[242] PARAM */) - 3.200000000000003e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10416(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10417(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10420(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10419(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10418(DATA *data, threadData_t *threadData);


/*
equation index: 3867
type: SIMPLE_ASSIGN
vx[242] = (-sin(theta[242])) * r_init[242] * omega_c[242]
*/
void SpiralGalaxy_eqFunction_3867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3867};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[241]] /* vx[242] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1748] /* theta[242] PARAM */)))) * (((data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */)) * ((data->simulationInfo->realParameter[746] /* omega_c[242] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10413(DATA *data, threadData_t *threadData);


/*
equation index: 3869
type: SIMPLE_ASSIGN
vy[242] = cos(theta[242]) * r_init[242] * omega_c[242]
*/
void SpiralGalaxy_eqFunction_3869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[741]] /* vy[242] STATE(1) */) = (cos((data->simulationInfo->realParameter[1748] /* theta[242] PARAM */))) * (((data->simulationInfo->realParameter[1247] /* r_init[242] PARAM */)) * ((data->simulationInfo->realParameter[746] /* omega_c[242] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10412(DATA *data, threadData_t *threadData);


/*
equation index: 3871
type: SIMPLE_ASSIGN
vz[242] = 0.0
*/
void SpiralGalaxy_eqFunction_3871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3871};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1241]] /* vz[242] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10411(DATA *data, threadData_t *threadData);


/*
equation index: 3873
type: SIMPLE_ASSIGN
z[243] = -0.0011200000000000003
*/
void SpiralGalaxy_eqFunction_3873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3873};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2742]] /* z[243] STATE(1,vz[243]) */) = -0.0011200000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10424(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10425(DATA *data, threadData_t *threadData);


/*
equation index: 3876
type: SIMPLE_ASSIGN
y[243] = r_init[243] * sin(theta[243] - 2.8000000000000025e-4)
*/
void SpiralGalaxy_eqFunction_3876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3876};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2242]] /* y[243] STATE(1,vy[243]) */) = ((data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */)) * (sin((data->simulationInfo->realParameter[1749] /* theta[243] PARAM */) - 2.8000000000000025e-4));
  TRACE_POP
}

/*
equation index: 3877
type: SIMPLE_ASSIGN
x[243] = r_init[243] * cos(theta[243] - 2.8000000000000025e-4)
*/
void SpiralGalaxy_eqFunction_3877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1742]] /* x[243] STATE(1,vx[243]) */) = ((data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */)) * (cos((data->simulationInfo->realParameter[1749] /* theta[243] PARAM */) - 2.8000000000000025e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10426(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10427(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10430(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10429(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10428(DATA *data, threadData_t *threadData);


/*
equation index: 3883
type: SIMPLE_ASSIGN
vx[243] = (-sin(theta[243])) * r_init[243] * omega_c[243]
*/
void SpiralGalaxy_eqFunction_3883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3883};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[242]] /* vx[243] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1749] /* theta[243] PARAM */)))) * (((data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */)) * ((data->simulationInfo->realParameter[747] /* omega_c[243] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10423(DATA *data, threadData_t *threadData);


/*
equation index: 3885
type: SIMPLE_ASSIGN
vy[243] = cos(theta[243]) * r_init[243] * omega_c[243]
*/
void SpiralGalaxy_eqFunction_3885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[742]] /* vy[243] STATE(1) */) = (cos((data->simulationInfo->realParameter[1749] /* theta[243] PARAM */))) * (((data->simulationInfo->realParameter[1248] /* r_init[243] PARAM */)) * ((data->simulationInfo->realParameter[747] /* omega_c[243] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10422(DATA *data, threadData_t *threadData);


/*
equation index: 3887
type: SIMPLE_ASSIGN
vz[243] = 0.0
*/
void SpiralGalaxy_eqFunction_3887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3887};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1242]] /* vz[243] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10421(DATA *data, threadData_t *threadData);


/*
equation index: 3889
type: SIMPLE_ASSIGN
z[244] = -9.600000000000001e-4
*/
void SpiralGalaxy_eqFunction_3889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3889};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2743]] /* z[244] STATE(1,vz[244]) */) = -9.600000000000001e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10434(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10435(DATA *data, threadData_t *threadData);


/*
equation index: 3892
type: SIMPLE_ASSIGN
y[244] = r_init[244] * sin(theta[244] - 2.4000000000000022e-4)
*/
void SpiralGalaxy_eqFunction_3892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3892};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2243]] /* y[244] STATE(1,vy[244]) */) = ((data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */)) * (sin((data->simulationInfo->realParameter[1750] /* theta[244] PARAM */) - 2.4000000000000022e-4));
  TRACE_POP
}

/*
equation index: 3893
type: SIMPLE_ASSIGN
x[244] = r_init[244] * cos(theta[244] - 2.4000000000000022e-4)
*/
void SpiralGalaxy_eqFunction_3893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1743]] /* x[244] STATE(1,vx[244]) */) = ((data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */)) * (cos((data->simulationInfo->realParameter[1750] /* theta[244] PARAM */) - 2.4000000000000022e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10436(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10437(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10440(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10439(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10438(DATA *data, threadData_t *threadData);


/*
equation index: 3899
type: SIMPLE_ASSIGN
vx[244] = (-sin(theta[244])) * r_init[244] * omega_c[244]
*/
void SpiralGalaxy_eqFunction_3899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3899};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[243]] /* vx[244] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1750] /* theta[244] PARAM */)))) * (((data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */)) * ((data->simulationInfo->realParameter[748] /* omega_c[244] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10433(DATA *data, threadData_t *threadData);


/*
equation index: 3901
type: SIMPLE_ASSIGN
vy[244] = cos(theta[244]) * r_init[244] * omega_c[244]
*/
void SpiralGalaxy_eqFunction_3901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[743]] /* vy[244] STATE(1) */) = (cos((data->simulationInfo->realParameter[1750] /* theta[244] PARAM */))) * (((data->simulationInfo->realParameter[1249] /* r_init[244] PARAM */)) * ((data->simulationInfo->realParameter[748] /* omega_c[244] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10432(DATA *data, threadData_t *threadData);


/*
equation index: 3903
type: SIMPLE_ASSIGN
vz[244] = 0.0
*/
void SpiralGalaxy_eqFunction_3903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3903};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1243]] /* vz[244] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10431(DATA *data, threadData_t *threadData);


/*
equation index: 3905
type: SIMPLE_ASSIGN
z[245] = -8.000000000000001e-4
*/
void SpiralGalaxy_eqFunction_3905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3905};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2744]] /* z[245] STATE(1,vz[245]) */) = -8.000000000000001e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10444(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10445(DATA *data, threadData_t *threadData);


/*
equation index: 3908
type: SIMPLE_ASSIGN
y[245] = r_init[245] * sin(theta[245] - 2.0000000000000017e-4)
*/
void SpiralGalaxy_eqFunction_3908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3908};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2244]] /* y[245] STATE(1,vy[245]) */) = ((data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */)) * (sin((data->simulationInfo->realParameter[1751] /* theta[245] PARAM */) - 2.0000000000000017e-4));
  TRACE_POP
}

/*
equation index: 3909
type: SIMPLE_ASSIGN
x[245] = r_init[245] * cos(theta[245] - 2.0000000000000017e-4)
*/
void SpiralGalaxy_eqFunction_3909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1744]] /* x[245] STATE(1,vx[245]) */) = ((data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */)) * (cos((data->simulationInfo->realParameter[1751] /* theta[245] PARAM */) - 2.0000000000000017e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10446(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10447(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10450(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10449(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10448(DATA *data, threadData_t *threadData);


/*
equation index: 3915
type: SIMPLE_ASSIGN
vx[245] = (-sin(theta[245])) * r_init[245] * omega_c[245]
*/
void SpiralGalaxy_eqFunction_3915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3915};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[244]] /* vx[245] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1751] /* theta[245] PARAM */)))) * (((data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */)) * ((data->simulationInfo->realParameter[749] /* omega_c[245] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10443(DATA *data, threadData_t *threadData);


/*
equation index: 3917
type: SIMPLE_ASSIGN
vy[245] = cos(theta[245]) * r_init[245] * omega_c[245]
*/
void SpiralGalaxy_eqFunction_3917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[744]] /* vy[245] STATE(1) */) = (cos((data->simulationInfo->realParameter[1751] /* theta[245] PARAM */))) * (((data->simulationInfo->realParameter[1250] /* r_init[245] PARAM */)) * ((data->simulationInfo->realParameter[749] /* omega_c[245] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10442(DATA *data, threadData_t *threadData);


/*
equation index: 3919
type: SIMPLE_ASSIGN
vz[245] = 0.0
*/
void SpiralGalaxy_eqFunction_3919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3919};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1244]] /* vz[245] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10441(DATA *data, threadData_t *threadData);


/*
equation index: 3921
type: SIMPLE_ASSIGN
z[246] = -6.400000000000002e-4
*/
void SpiralGalaxy_eqFunction_3921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3921};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2745]] /* z[246] STATE(1,vz[246]) */) = -6.400000000000002e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10454(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10455(DATA *data, threadData_t *threadData);


/*
equation index: 3924
type: SIMPLE_ASSIGN
y[246] = r_init[246] * sin(theta[246] - 1.6000000000000015e-4)
*/
void SpiralGalaxy_eqFunction_3924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3924};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2245]] /* y[246] STATE(1,vy[246]) */) = ((data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */)) * (sin((data->simulationInfo->realParameter[1752] /* theta[246] PARAM */) - 1.6000000000000015e-4));
  TRACE_POP
}

/*
equation index: 3925
type: SIMPLE_ASSIGN
x[246] = r_init[246] * cos(theta[246] - 1.6000000000000015e-4)
*/
void SpiralGalaxy_eqFunction_3925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1745]] /* x[246] STATE(1,vx[246]) */) = ((data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */)) * (cos((data->simulationInfo->realParameter[1752] /* theta[246] PARAM */) - 1.6000000000000015e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10456(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10457(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10460(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10459(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10458(DATA *data, threadData_t *threadData);


/*
equation index: 3931
type: SIMPLE_ASSIGN
vx[246] = (-sin(theta[246])) * r_init[246] * omega_c[246]
*/
void SpiralGalaxy_eqFunction_3931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3931};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[245]] /* vx[246] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1752] /* theta[246] PARAM */)))) * (((data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */)) * ((data->simulationInfo->realParameter[750] /* omega_c[246] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10453(DATA *data, threadData_t *threadData);


/*
equation index: 3933
type: SIMPLE_ASSIGN
vy[246] = cos(theta[246]) * r_init[246] * omega_c[246]
*/
void SpiralGalaxy_eqFunction_3933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[745]] /* vy[246] STATE(1) */) = (cos((data->simulationInfo->realParameter[1752] /* theta[246] PARAM */))) * (((data->simulationInfo->realParameter[1251] /* r_init[246] PARAM */)) * ((data->simulationInfo->realParameter[750] /* omega_c[246] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10452(DATA *data, threadData_t *threadData);


/*
equation index: 3935
type: SIMPLE_ASSIGN
vz[246] = 0.0
*/
void SpiralGalaxy_eqFunction_3935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3935};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1245]] /* vz[246] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10451(DATA *data, threadData_t *threadData);


/*
equation index: 3937
type: SIMPLE_ASSIGN
z[247] = -4.8000000000000007e-4
*/
void SpiralGalaxy_eqFunction_3937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3937};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2746]] /* z[247] STATE(1,vz[247]) */) = -4.8000000000000007e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10464(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10465(DATA *data, threadData_t *threadData);


/*
equation index: 3940
type: SIMPLE_ASSIGN
y[247] = r_init[247] * sin(theta[247] - 1.2000000000000011e-4)
*/
void SpiralGalaxy_eqFunction_3940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3940};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2246]] /* y[247] STATE(1,vy[247]) */) = ((data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */)) * (sin((data->simulationInfo->realParameter[1753] /* theta[247] PARAM */) - 1.2000000000000011e-4));
  TRACE_POP
}

/*
equation index: 3941
type: SIMPLE_ASSIGN
x[247] = r_init[247] * cos(theta[247] - 1.2000000000000011e-4)
*/
void SpiralGalaxy_eqFunction_3941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1746]] /* x[247] STATE(1,vx[247]) */) = ((data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */)) * (cos((data->simulationInfo->realParameter[1753] /* theta[247] PARAM */) - 1.2000000000000011e-4));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10466(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10467(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10470(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10469(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10468(DATA *data, threadData_t *threadData);


/*
equation index: 3947
type: SIMPLE_ASSIGN
vx[247] = (-sin(theta[247])) * r_init[247] * omega_c[247]
*/
void SpiralGalaxy_eqFunction_3947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3947};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[246]] /* vx[247] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1753] /* theta[247] PARAM */)))) * (((data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */)) * ((data->simulationInfo->realParameter[751] /* omega_c[247] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10463(DATA *data, threadData_t *threadData);


/*
equation index: 3949
type: SIMPLE_ASSIGN
vy[247] = cos(theta[247]) * r_init[247] * omega_c[247]
*/
void SpiralGalaxy_eqFunction_3949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[746]] /* vy[247] STATE(1) */) = (cos((data->simulationInfo->realParameter[1753] /* theta[247] PARAM */))) * (((data->simulationInfo->realParameter[1252] /* r_init[247] PARAM */)) * ((data->simulationInfo->realParameter[751] /* omega_c[247] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10462(DATA *data, threadData_t *threadData);


/*
equation index: 3951
type: SIMPLE_ASSIGN
vz[247] = 0.0
*/
void SpiralGalaxy_eqFunction_3951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3951};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1246]] /* vz[247] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10461(DATA *data, threadData_t *threadData);


/*
equation index: 3953
type: SIMPLE_ASSIGN
z[248] = -3.200000000000001e-4
*/
void SpiralGalaxy_eqFunction_3953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3953};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2747]] /* z[248] STATE(1,vz[248]) */) = -3.200000000000001e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10474(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10475(DATA *data, threadData_t *threadData);


/*
equation index: 3956
type: SIMPLE_ASSIGN
y[248] = r_init[248] * sin(theta[248] - 8.000000000000007e-5)
*/
void SpiralGalaxy_eqFunction_3956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3956};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2247]] /* y[248] STATE(1,vy[248]) */) = ((data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */)) * (sin((data->simulationInfo->realParameter[1754] /* theta[248] PARAM */) - 8.000000000000007e-5));
  TRACE_POP
}

/*
equation index: 3957
type: SIMPLE_ASSIGN
x[248] = r_init[248] * cos(theta[248] - 8.000000000000007e-5)
*/
void SpiralGalaxy_eqFunction_3957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1747]] /* x[248] STATE(1,vx[248]) */) = ((data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */)) * (cos((data->simulationInfo->realParameter[1754] /* theta[248] PARAM */) - 8.000000000000007e-5));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10476(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10477(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10480(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10479(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10478(DATA *data, threadData_t *threadData);


/*
equation index: 3963
type: SIMPLE_ASSIGN
vx[248] = (-sin(theta[248])) * r_init[248] * omega_c[248]
*/
void SpiralGalaxy_eqFunction_3963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3963};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[247]] /* vx[248] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1754] /* theta[248] PARAM */)))) * (((data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */)) * ((data->simulationInfo->realParameter[752] /* omega_c[248] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10473(DATA *data, threadData_t *threadData);


/*
equation index: 3965
type: SIMPLE_ASSIGN
vy[248] = cos(theta[248]) * r_init[248] * omega_c[248]
*/
void SpiralGalaxy_eqFunction_3965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[747]] /* vy[248] STATE(1) */) = (cos((data->simulationInfo->realParameter[1754] /* theta[248] PARAM */))) * (((data->simulationInfo->realParameter[1253] /* r_init[248] PARAM */)) * ((data->simulationInfo->realParameter[752] /* omega_c[248] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10472(DATA *data, threadData_t *threadData);


/*
equation index: 3967
type: SIMPLE_ASSIGN
vz[248] = 0.0
*/
void SpiralGalaxy_eqFunction_3967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3967};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1247]] /* vz[248] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10471(DATA *data, threadData_t *threadData);


/*
equation index: 3969
type: SIMPLE_ASSIGN
z[249] = -1.6000000000000004e-4
*/
void SpiralGalaxy_eqFunction_3969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3969};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2748]] /* z[249] STATE(1,vz[249]) */) = -1.6000000000000004e-4;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10484(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10485(DATA *data, threadData_t *threadData);


/*
equation index: 3972
type: SIMPLE_ASSIGN
y[249] = r_init[249] * sin(theta[249] - 4.000000000000004e-5)
*/
void SpiralGalaxy_eqFunction_3972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3972};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2248]] /* y[249] STATE(1,vy[249]) */) = ((data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */)) * (sin((data->simulationInfo->realParameter[1755] /* theta[249] PARAM */) - 4.000000000000004e-5));
  TRACE_POP
}

/*
equation index: 3973
type: SIMPLE_ASSIGN
x[249] = r_init[249] * cos(theta[249] - 4.000000000000004e-5)
*/
void SpiralGalaxy_eqFunction_3973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1748]] /* x[249] STATE(1,vx[249]) */) = ((data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */)) * (cos((data->simulationInfo->realParameter[1755] /* theta[249] PARAM */) - 4.000000000000004e-5));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10486(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10487(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10490(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10489(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10488(DATA *data, threadData_t *threadData);


/*
equation index: 3979
type: SIMPLE_ASSIGN
vx[249] = (-sin(theta[249])) * r_init[249] * omega_c[249]
*/
void SpiralGalaxy_eqFunction_3979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3979};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[248]] /* vx[249] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1755] /* theta[249] PARAM */)))) * (((data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */)) * ((data->simulationInfo->realParameter[753] /* omega_c[249] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10483(DATA *data, threadData_t *threadData);


/*
equation index: 3981
type: SIMPLE_ASSIGN
vy[249] = cos(theta[249]) * r_init[249] * omega_c[249]
*/
void SpiralGalaxy_eqFunction_3981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[748]] /* vy[249] STATE(1) */) = (cos((data->simulationInfo->realParameter[1755] /* theta[249] PARAM */))) * (((data->simulationInfo->realParameter[1254] /* r_init[249] PARAM */)) * ((data->simulationInfo->realParameter[753] /* omega_c[249] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10482(DATA *data, threadData_t *threadData);


/*
equation index: 3983
type: SIMPLE_ASSIGN
vz[249] = 0.0
*/
void SpiralGalaxy_eqFunction_3983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3983};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1248]] /* vz[249] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10481(DATA *data, threadData_t *threadData);


/*
equation index: 3985
type: SIMPLE_ASSIGN
z[250] = 0.0
*/
void SpiralGalaxy_eqFunction_3985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3985};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2749]] /* z[250] STATE(1,vz[250]) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10494(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10495(DATA *data, threadData_t *threadData);


/*
equation index: 3988
type: SIMPLE_ASSIGN
y[250] = r_init[250] * sin(theta[250])
*/
void SpiralGalaxy_eqFunction_3988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3988};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2249]] /* y[250] STATE(1,vy[250]) */) = ((data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */)) * (sin((data->simulationInfo->realParameter[1756] /* theta[250] PARAM */)));
  TRACE_POP
}

/*
equation index: 3989
type: SIMPLE_ASSIGN
x[250] = r_init[250] * cos(theta[250])
*/
void SpiralGalaxy_eqFunction_3989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1749]] /* x[250] STATE(1,vx[250]) */) = ((data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */)) * (cos((data->simulationInfo->realParameter[1756] /* theta[250] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10496(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10497(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10500(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10499(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10498(DATA *data, threadData_t *threadData);


/*
equation index: 3995
type: SIMPLE_ASSIGN
vx[250] = (-sin(theta[250])) * r_init[250] * omega_c[250]
*/
void SpiralGalaxy_eqFunction_3995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3995};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[249]] /* vx[250] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1756] /* theta[250] PARAM */)))) * (((data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */)) * ((data->simulationInfo->realParameter[754] /* omega_c[250] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10493(DATA *data, threadData_t *threadData);


/*
equation index: 3997
type: SIMPLE_ASSIGN
vy[250] = cos(theta[250]) * r_init[250] * omega_c[250]
*/
void SpiralGalaxy_eqFunction_3997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[749]] /* vy[250] STATE(1) */) = (cos((data->simulationInfo->realParameter[1756] /* theta[250] PARAM */))) * (((data->simulationInfo->realParameter[1255] /* r_init[250] PARAM */)) * ((data->simulationInfo->realParameter[754] /* omega_c[250] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10492(DATA *data, threadData_t *threadData);


/*
equation index: 3999
type: SIMPLE_ASSIGN
vz[250] = 0.0
*/
void SpiralGalaxy_eqFunction_3999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3999};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1249]] /* vz[250] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10491(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_7(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_3501(data, threadData);
  SpiralGalaxy_eqFunction_10182(data, threadData);
  SpiralGalaxy_eqFunction_3503(data, threadData);
  SpiralGalaxy_eqFunction_10181(data, threadData);
  SpiralGalaxy_eqFunction_3505(data, threadData);
  SpiralGalaxy_eqFunction_10194(data, threadData);
  SpiralGalaxy_eqFunction_10195(data, threadData);
  SpiralGalaxy_eqFunction_3508(data, threadData);
  SpiralGalaxy_eqFunction_3509(data, threadData);
  SpiralGalaxy_eqFunction_10196(data, threadData);
  SpiralGalaxy_eqFunction_10197(data, threadData);
  SpiralGalaxy_eqFunction_10200(data, threadData);
  SpiralGalaxy_eqFunction_10199(data, threadData);
  SpiralGalaxy_eqFunction_10198(data, threadData);
  SpiralGalaxy_eqFunction_3515(data, threadData);
  SpiralGalaxy_eqFunction_10193(data, threadData);
  SpiralGalaxy_eqFunction_3517(data, threadData);
  SpiralGalaxy_eqFunction_10192(data, threadData);
  SpiralGalaxy_eqFunction_3519(data, threadData);
  SpiralGalaxy_eqFunction_10191(data, threadData);
  SpiralGalaxy_eqFunction_3521(data, threadData);
  SpiralGalaxy_eqFunction_10204(data, threadData);
  SpiralGalaxy_eqFunction_10205(data, threadData);
  SpiralGalaxy_eqFunction_3524(data, threadData);
  SpiralGalaxy_eqFunction_3525(data, threadData);
  SpiralGalaxy_eqFunction_10206(data, threadData);
  SpiralGalaxy_eqFunction_10207(data, threadData);
  SpiralGalaxy_eqFunction_10210(data, threadData);
  SpiralGalaxy_eqFunction_10209(data, threadData);
  SpiralGalaxy_eqFunction_10208(data, threadData);
  SpiralGalaxy_eqFunction_3531(data, threadData);
  SpiralGalaxy_eqFunction_10203(data, threadData);
  SpiralGalaxy_eqFunction_3533(data, threadData);
  SpiralGalaxy_eqFunction_10202(data, threadData);
  SpiralGalaxy_eqFunction_3535(data, threadData);
  SpiralGalaxy_eqFunction_10201(data, threadData);
  SpiralGalaxy_eqFunction_3537(data, threadData);
  SpiralGalaxy_eqFunction_10214(data, threadData);
  SpiralGalaxy_eqFunction_10215(data, threadData);
  SpiralGalaxy_eqFunction_3540(data, threadData);
  SpiralGalaxy_eqFunction_3541(data, threadData);
  SpiralGalaxy_eqFunction_10216(data, threadData);
  SpiralGalaxy_eqFunction_10217(data, threadData);
  SpiralGalaxy_eqFunction_10220(data, threadData);
  SpiralGalaxy_eqFunction_10219(data, threadData);
  SpiralGalaxy_eqFunction_10218(data, threadData);
  SpiralGalaxy_eqFunction_3547(data, threadData);
  SpiralGalaxy_eqFunction_10213(data, threadData);
  SpiralGalaxy_eqFunction_3549(data, threadData);
  SpiralGalaxy_eqFunction_10212(data, threadData);
  SpiralGalaxy_eqFunction_3551(data, threadData);
  SpiralGalaxy_eqFunction_10211(data, threadData);
  SpiralGalaxy_eqFunction_3553(data, threadData);
  SpiralGalaxy_eqFunction_10224(data, threadData);
  SpiralGalaxy_eqFunction_10225(data, threadData);
  SpiralGalaxy_eqFunction_3556(data, threadData);
  SpiralGalaxy_eqFunction_3557(data, threadData);
  SpiralGalaxy_eqFunction_10226(data, threadData);
  SpiralGalaxy_eqFunction_10227(data, threadData);
  SpiralGalaxy_eqFunction_10230(data, threadData);
  SpiralGalaxy_eqFunction_10229(data, threadData);
  SpiralGalaxy_eqFunction_10228(data, threadData);
  SpiralGalaxy_eqFunction_3563(data, threadData);
  SpiralGalaxy_eqFunction_10223(data, threadData);
  SpiralGalaxy_eqFunction_3565(data, threadData);
  SpiralGalaxy_eqFunction_10222(data, threadData);
  SpiralGalaxy_eqFunction_3567(data, threadData);
  SpiralGalaxy_eqFunction_10221(data, threadData);
  SpiralGalaxy_eqFunction_3569(data, threadData);
  SpiralGalaxy_eqFunction_10234(data, threadData);
  SpiralGalaxy_eqFunction_10235(data, threadData);
  SpiralGalaxy_eqFunction_3572(data, threadData);
  SpiralGalaxy_eqFunction_3573(data, threadData);
  SpiralGalaxy_eqFunction_10236(data, threadData);
  SpiralGalaxy_eqFunction_10237(data, threadData);
  SpiralGalaxy_eqFunction_10240(data, threadData);
  SpiralGalaxy_eqFunction_10239(data, threadData);
  SpiralGalaxy_eqFunction_10238(data, threadData);
  SpiralGalaxy_eqFunction_3579(data, threadData);
  SpiralGalaxy_eqFunction_10233(data, threadData);
  SpiralGalaxy_eqFunction_3581(data, threadData);
  SpiralGalaxy_eqFunction_10232(data, threadData);
  SpiralGalaxy_eqFunction_3583(data, threadData);
  SpiralGalaxy_eqFunction_10231(data, threadData);
  SpiralGalaxy_eqFunction_3585(data, threadData);
  SpiralGalaxy_eqFunction_10244(data, threadData);
  SpiralGalaxy_eqFunction_10245(data, threadData);
  SpiralGalaxy_eqFunction_3588(data, threadData);
  SpiralGalaxy_eqFunction_3589(data, threadData);
  SpiralGalaxy_eqFunction_10246(data, threadData);
  SpiralGalaxy_eqFunction_10247(data, threadData);
  SpiralGalaxy_eqFunction_10250(data, threadData);
  SpiralGalaxy_eqFunction_10249(data, threadData);
  SpiralGalaxy_eqFunction_10248(data, threadData);
  SpiralGalaxy_eqFunction_3595(data, threadData);
  SpiralGalaxy_eqFunction_10243(data, threadData);
  SpiralGalaxy_eqFunction_3597(data, threadData);
  SpiralGalaxy_eqFunction_10242(data, threadData);
  SpiralGalaxy_eqFunction_3599(data, threadData);
  SpiralGalaxy_eqFunction_10241(data, threadData);
  SpiralGalaxy_eqFunction_3601(data, threadData);
  SpiralGalaxy_eqFunction_10254(data, threadData);
  SpiralGalaxy_eqFunction_10255(data, threadData);
  SpiralGalaxy_eqFunction_3604(data, threadData);
  SpiralGalaxy_eqFunction_3605(data, threadData);
  SpiralGalaxy_eqFunction_10256(data, threadData);
  SpiralGalaxy_eqFunction_10257(data, threadData);
  SpiralGalaxy_eqFunction_10260(data, threadData);
  SpiralGalaxy_eqFunction_10259(data, threadData);
  SpiralGalaxy_eqFunction_10258(data, threadData);
  SpiralGalaxy_eqFunction_3611(data, threadData);
  SpiralGalaxy_eqFunction_10253(data, threadData);
  SpiralGalaxy_eqFunction_3613(data, threadData);
  SpiralGalaxy_eqFunction_10252(data, threadData);
  SpiralGalaxy_eqFunction_3615(data, threadData);
  SpiralGalaxy_eqFunction_10251(data, threadData);
  SpiralGalaxy_eqFunction_3617(data, threadData);
  SpiralGalaxy_eqFunction_10264(data, threadData);
  SpiralGalaxy_eqFunction_10265(data, threadData);
  SpiralGalaxy_eqFunction_3620(data, threadData);
  SpiralGalaxy_eqFunction_3621(data, threadData);
  SpiralGalaxy_eqFunction_10266(data, threadData);
  SpiralGalaxy_eqFunction_10267(data, threadData);
  SpiralGalaxy_eqFunction_10270(data, threadData);
  SpiralGalaxy_eqFunction_10269(data, threadData);
  SpiralGalaxy_eqFunction_10268(data, threadData);
  SpiralGalaxy_eqFunction_3627(data, threadData);
  SpiralGalaxy_eqFunction_10263(data, threadData);
  SpiralGalaxy_eqFunction_3629(data, threadData);
  SpiralGalaxy_eqFunction_10262(data, threadData);
  SpiralGalaxy_eqFunction_3631(data, threadData);
  SpiralGalaxy_eqFunction_10261(data, threadData);
  SpiralGalaxy_eqFunction_3633(data, threadData);
  SpiralGalaxy_eqFunction_10274(data, threadData);
  SpiralGalaxy_eqFunction_10275(data, threadData);
  SpiralGalaxy_eqFunction_3636(data, threadData);
  SpiralGalaxy_eqFunction_3637(data, threadData);
  SpiralGalaxy_eqFunction_10276(data, threadData);
  SpiralGalaxy_eqFunction_10277(data, threadData);
  SpiralGalaxy_eqFunction_10280(data, threadData);
  SpiralGalaxy_eqFunction_10279(data, threadData);
  SpiralGalaxy_eqFunction_10278(data, threadData);
  SpiralGalaxy_eqFunction_3643(data, threadData);
  SpiralGalaxy_eqFunction_10273(data, threadData);
  SpiralGalaxy_eqFunction_3645(data, threadData);
  SpiralGalaxy_eqFunction_10272(data, threadData);
  SpiralGalaxy_eqFunction_3647(data, threadData);
  SpiralGalaxy_eqFunction_10271(data, threadData);
  SpiralGalaxy_eqFunction_3649(data, threadData);
  SpiralGalaxy_eqFunction_10284(data, threadData);
  SpiralGalaxy_eqFunction_10285(data, threadData);
  SpiralGalaxy_eqFunction_3652(data, threadData);
  SpiralGalaxy_eqFunction_3653(data, threadData);
  SpiralGalaxy_eqFunction_10286(data, threadData);
  SpiralGalaxy_eqFunction_10287(data, threadData);
  SpiralGalaxy_eqFunction_10290(data, threadData);
  SpiralGalaxy_eqFunction_10289(data, threadData);
  SpiralGalaxy_eqFunction_10288(data, threadData);
  SpiralGalaxy_eqFunction_3659(data, threadData);
  SpiralGalaxy_eqFunction_10283(data, threadData);
  SpiralGalaxy_eqFunction_3661(data, threadData);
  SpiralGalaxy_eqFunction_10282(data, threadData);
  SpiralGalaxy_eqFunction_3663(data, threadData);
  SpiralGalaxy_eqFunction_10281(data, threadData);
  SpiralGalaxy_eqFunction_3665(data, threadData);
  SpiralGalaxy_eqFunction_10294(data, threadData);
  SpiralGalaxy_eqFunction_10295(data, threadData);
  SpiralGalaxy_eqFunction_3668(data, threadData);
  SpiralGalaxy_eqFunction_3669(data, threadData);
  SpiralGalaxy_eqFunction_10296(data, threadData);
  SpiralGalaxy_eqFunction_10297(data, threadData);
  SpiralGalaxy_eqFunction_10300(data, threadData);
  SpiralGalaxy_eqFunction_10299(data, threadData);
  SpiralGalaxy_eqFunction_10298(data, threadData);
  SpiralGalaxy_eqFunction_3675(data, threadData);
  SpiralGalaxy_eqFunction_10293(data, threadData);
  SpiralGalaxy_eqFunction_3677(data, threadData);
  SpiralGalaxy_eqFunction_10292(data, threadData);
  SpiralGalaxy_eqFunction_3679(data, threadData);
  SpiralGalaxy_eqFunction_10291(data, threadData);
  SpiralGalaxy_eqFunction_3681(data, threadData);
  SpiralGalaxy_eqFunction_10304(data, threadData);
  SpiralGalaxy_eqFunction_10305(data, threadData);
  SpiralGalaxy_eqFunction_3684(data, threadData);
  SpiralGalaxy_eqFunction_3685(data, threadData);
  SpiralGalaxy_eqFunction_10306(data, threadData);
  SpiralGalaxy_eqFunction_10307(data, threadData);
  SpiralGalaxy_eqFunction_10310(data, threadData);
  SpiralGalaxy_eqFunction_10309(data, threadData);
  SpiralGalaxy_eqFunction_10308(data, threadData);
  SpiralGalaxy_eqFunction_3691(data, threadData);
  SpiralGalaxy_eqFunction_10303(data, threadData);
  SpiralGalaxy_eqFunction_3693(data, threadData);
  SpiralGalaxy_eqFunction_10302(data, threadData);
  SpiralGalaxy_eqFunction_3695(data, threadData);
  SpiralGalaxy_eqFunction_10301(data, threadData);
  SpiralGalaxy_eqFunction_3697(data, threadData);
  SpiralGalaxy_eqFunction_10314(data, threadData);
  SpiralGalaxy_eqFunction_10315(data, threadData);
  SpiralGalaxy_eqFunction_3700(data, threadData);
  SpiralGalaxy_eqFunction_3701(data, threadData);
  SpiralGalaxy_eqFunction_10316(data, threadData);
  SpiralGalaxy_eqFunction_10317(data, threadData);
  SpiralGalaxy_eqFunction_10320(data, threadData);
  SpiralGalaxy_eqFunction_10319(data, threadData);
  SpiralGalaxy_eqFunction_10318(data, threadData);
  SpiralGalaxy_eqFunction_3707(data, threadData);
  SpiralGalaxy_eqFunction_10313(data, threadData);
  SpiralGalaxy_eqFunction_3709(data, threadData);
  SpiralGalaxy_eqFunction_10312(data, threadData);
  SpiralGalaxy_eqFunction_3711(data, threadData);
  SpiralGalaxy_eqFunction_10311(data, threadData);
  SpiralGalaxy_eqFunction_3713(data, threadData);
  SpiralGalaxy_eqFunction_10324(data, threadData);
  SpiralGalaxy_eqFunction_10325(data, threadData);
  SpiralGalaxy_eqFunction_3716(data, threadData);
  SpiralGalaxy_eqFunction_3717(data, threadData);
  SpiralGalaxy_eqFunction_10326(data, threadData);
  SpiralGalaxy_eqFunction_10327(data, threadData);
  SpiralGalaxy_eqFunction_10330(data, threadData);
  SpiralGalaxy_eqFunction_10329(data, threadData);
  SpiralGalaxy_eqFunction_10328(data, threadData);
  SpiralGalaxy_eqFunction_3723(data, threadData);
  SpiralGalaxy_eqFunction_10323(data, threadData);
  SpiralGalaxy_eqFunction_3725(data, threadData);
  SpiralGalaxy_eqFunction_10322(data, threadData);
  SpiralGalaxy_eqFunction_3727(data, threadData);
  SpiralGalaxy_eqFunction_10321(data, threadData);
  SpiralGalaxy_eqFunction_3729(data, threadData);
  SpiralGalaxy_eqFunction_10334(data, threadData);
  SpiralGalaxy_eqFunction_10335(data, threadData);
  SpiralGalaxy_eqFunction_3732(data, threadData);
  SpiralGalaxy_eqFunction_3733(data, threadData);
  SpiralGalaxy_eqFunction_10336(data, threadData);
  SpiralGalaxy_eqFunction_10337(data, threadData);
  SpiralGalaxy_eqFunction_10340(data, threadData);
  SpiralGalaxy_eqFunction_10339(data, threadData);
  SpiralGalaxy_eqFunction_10338(data, threadData);
  SpiralGalaxy_eqFunction_3739(data, threadData);
  SpiralGalaxy_eqFunction_10333(data, threadData);
  SpiralGalaxy_eqFunction_3741(data, threadData);
  SpiralGalaxy_eqFunction_10332(data, threadData);
  SpiralGalaxy_eqFunction_3743(data, threadData);
  SpiralGalaxy_eqFunction_10331(data, threadData);
  SpiralGalaxy_eqFunction_3745(data, threadData);
  SpiralGalaxy_eqFunction_10344(data, threadData);
  SpiralGalaxy_eqFunction_10345(data, threadData);
  SpiralGalaxy_eqFunction_3748(data, threadData);
  SpiralGalaxy_eqFunction_3749(data, threadData);
  SpiralGalaxy_eqFunction_10346(data, threadData);
  SpiralGalaxy_eqFunction_10347(data, threadData);
  SpiralGalaxy_eqFunction_10350(data, threadData);
  SpiralGalaxy_eqFunction_10349(data, threadData);
  SpiralGalaxy_eqFunction_10348(data, threadData);
  SpiralGalaxy_eqFunction_3755(data, threadData);
  SpiralGalaxy_eqFunction_10343(data, threadData);
  SpiralGalaxy_eqFunction_3757(data, threadData);
  SpiralGalaxy_eqFunction_10342(data, threadData);
  SpiralGalaxy_eqFunction_3759(data, threadData);
  SpiralGalaxy_eqFunction_10341(data, threadData);
  SpiralGalaxy_eqFunction_3761(data, threadData);
  SpiralGalaxy_eqFunction_10354(data, threadData);
  SpiralGalaxy_eqFunction_10355(data, threadData);
  SpiralGalaxy_eqFunction_3764(data, threadData);
  SpiralGalaxy_eqFunction_3765(data, threadData);
  SpiralGalaxy_eqFunction_10356(data, threadData);
  SpiralGalaxy_eqFunction_10357(data, threadData);
  SpiralGalaxy_eqFunction_10360(data, threadData);
  SpiralGalaxy_eqFunction_10359(data, threadData);
  SpiralGalaxy_eqFunction_10358(data, threadData);
  SpiralGalaxy_eqFunction_3771(data, threadData);
  SpiralGalaxy_eqFunction_10353(data, threadData);
  SpiralGalaxy_eqFunction_3773(data, threadData);
  SpiralGalaxy_eqFunction_10352(data, threadData);
  SpiralGalaxy_eqFunction_3775(data, threadData);
  SpiralGalaxy_eqFunction_10351(data, threadData);
  SpiralGalaxy_eqFunction_3777(data, threadData);
  SpiralGalaxy_eqFunction_10364(data, threadData);
  SpiralGalaxy_eqFunction_10365(data, threadData);
  SpiralGalaxy_eqFunction_3780(data, threadData);
  SpiralGalaxy_eqFunction_3781(data, threadData);
  SpiralGalaxy_eqFunction_10366(data, threadData);
  SpiralGalaxy_eqFunction_10367(data, threadData);
  SpiralGalaxy_eqFunction_10370(data, threadData);
  SpiralGalaxy_eqFunction_10369(data, threadData);
  SpiralGalaxy_eqFunction_10368(data, threadData);
  SpiralGalaxy_eqFunction_3787(data, threadData);
  SpiralGalaxy_eqFunction_10363(data, threadData);
  SpiralGalaxy_eqFunction_3789(data, threadData);
  SpiralGalaxy_eqFunction_10362(data, threadData);
  SpiralGalaxy_eqFunction_3791(data, threadData);
  SpiralGalaxy_eqFunction_10361(data, threadData);
  SpiralGalaxy_eqFunction_3793(data, threadData);
  SpiralGalaxy_eqFunction_10374(data, threadData);
  SpiralGalaxy_eqFunction_10375(data, threadData);
  SpiralGalaxy_eqFunction_3796(data, threadData);
  SpiralGalaxy_eqFunction_3797(data, threadData);
  SpiralGalaxy_eqFunction_10376(data, threadData);
  SpiralGalaxy_eqFunction_10377(data, threadData);
  SpiralGalaxy_eqFunction_10380(data, threadData);
  SpiralGalaxy_eqFunction_10379(data, threadData);
  SpiralGalaxy_eqFunction_10378(data, threadData);
  SpiralGalaxy_eqFunction_3803(data, threadData);
  SpiralGalaxy_eqFunction_10373(data, threadData);
  SpiralGalaxy_eqFunction_3805(data, threadData);
  SpiralGalaxy_eqFunction_10372(data, threadData);
  SpiralGalaxy_eqFunction_3807(data, threadData);
  SpiralGalaxy_eqFunction_10371(data, threadData);
  SpiralGalaxy_eqFunction_3809(data, threadData);
  SpiralGalaxy_eqFunction_10384(data, threadData);
  SpiralGalaxy_eqFunction_10385(data, threadData);
  SpiralGalaxy_eqFunction_3812(data, threadData);
  SpiralGalaxy_eqFunction_3813(data, threadData);
  SpiralGalaxy_eqFunction_10386(data, threadData);
  SpiralGalaxy_eqFunction_10387(data, threadData);
  SpiralGalaxy_eqFunction_10390(data, threadData);
  SpiralGalaxy_eqFunction_10389(data, threadData);
  SpiralGalaxy_eqFunction_10388(data, threadData);
  SpiralGalaxy_eqFunction_3819(data, threadData);
  SpiralGalaxy_eqFunction_10383(data, threadData);
  SpiralGalaxy_eqFunction_3821(data, threadData);
  SpiralGalaxy_eqFunction_10382(data, threadData);
  SpiralGalaxy_eqFunction_3823(data, threadData);
  SpiralGalaxy_eqFunction_10381(data, threadData);
  SpiralGalaxy_eqFunction_3825(data, threadData);
  SpiralGalaxy_eqFunction_10394(data, threadData);
  SpiralGalaxy_eqFunction_10395(data, threadData);
  SpiralGalaxy_eqFunction_3828(data, threadData);
  SpiralGalaxy_eqFunction_3829(data, threadData);
  SpiralGalaxy_eqFunction_10396(data, threadData);
  SpiralGalaxy_eqFunction_10397(data, threadData);
  SpiralGalaxy_eqFunction_10400(data, threadData);
  SpiralGalaxy_eqFunction_10399(data, threadData);
  SpiralGalaxy_eqFunction_10398(data, threadData);
  SpiralGalaxy_eqFunction_3835(data, threadData);
  SpiralGalaxy_eqFunction_10393(data, threadData);
  SpiralGalaxy_eqFunction_3837(data, threadData);
  SpiralGalaxy_eqFunction_10392(data, threadData);
  SpiralGalaxy_eqFunction_3839(data, threadData);
  SpiralGalaxy_eqFunction_10391(data, threadData);
  SpiralGalaxy_eqFunction_3841(data, threadData);
  SpiralGalaxy_eqFunction_10404(data, threadData);
  SpiralGalaxy_eqFunction_10405(data, threadData);
  SpiralGalaxy_eqFunction_3844(data, threadData);
  SpiralGalaxy_eqFunction_3845(data, threadData);
  SpiralGalaxy_eqFunction_10406(data, threadData);
  SpiralGalaxy_eqFunction_10407(data, threadData);
  SpiralGalaxy_eqFunction_10410(data, threadData);
  SpiralGalaxy_eqFunction_10409(data, threadData);
  SpiralGalaxy_eqFunction_10408(data, threadData);
  SpiralGalaxy_eqFunction_3851(data, threadData);
  SpiralGalaxy_eqFunction_10403(data, threadData);
  SpiralGalaxy_eqFunction_3853(data, threadData);
  SpiralGalaxy_eqFunction_10402(data, threadData);
  SpiralGalaxy_eqFunction_3855(data, threadData);
  SpiralGalaxy_eqFunction_10401(data, threadData);
  SpiralGalaxy_eqFunction_3857(data, threadData);
  SpiralGalaxy_eqFunction_10414(data, threadData);
  SpiralGalaxy_eqFunction_10415(data, threadData);
  SpiralGalaxy_eqFunction_3860(data, threadData);
  SpiralGalaxy_eqFunction_3861(data, threadData);
  SpiralGalaxy_eqFunction_10416(data, threadData);
  SpiralGalaxy_eqFunction_10417(data, threadData);
  SpiralGalaxy_eqFunction_10420(data, threadData);
  SpiralGalaxy_eqFunction_10419(data, threadData);
  SpiralGalaxy_eqFunction_10418(data, threadData);
  SpiralGalaxy_eqFunction_3867(data, threadData);
  SpiralGalaxy_eqFunction_10413(data, threadData);
  SpiralGalaxy_eqFunction_3869(data, threadData);
  SpiralGalaxy_eqFunction_10412(data, threadData);
  SpiralGalaxy_eqFunction_3871(data, threadData);
  SpiralGalaxy_eqFunction_10411(data, threadData);
  SpiralGalaxy_eqFunction_3873(data, threadData);
  SpiralGalaxy_eqFunction_10424(data, threadData);
  SpiralGalaxy_eqFunction_10425(data, threadData);
  SpiralGalaxy_eqFunction_3876(data, threadData);
  SpiralGalaxy_eqFunction_3877(data, threadData);
  SpiralGalaxy_eqFunction_10426(data, threadData);
  SpiralGalaxy_eqFunction_10427(data, threadData);
  SpiralGalaxy_eqFunction_10430(data, threadData);
  SpiralGalaxy_eqFunction_10429(data, threadData);
  SpiralGalaxy_eqFunction_10428(data, threadData);
  SpiralGalaxy_eqFunction_3883(data, threadData);
  SpiralGalaxy_eqFunction_10423(data, threadData);
  SpiralGalaxy_eqFunction_3885(data, threadData);
  SpiralGalaxy_eqFunction_10422(data, threadData);
  SpiralGalaxy_eqFunction_3887(data, threadData);
  SpiralGalaxy_eqFunction_10421(data, threadData);
  SpiralGalaxy_eqFunction_3889(data, threadData);
  SpiralGalaxy_eqFunction_10434(data, threadData);
  SpiralGalaxy_eqFunction_10435(data, threadData);
  SpiralGalaxy_eqFunction_3892(data, threadData);
  SpiralGalaxy_eqFunction_3893(data, threadData);
  SpiralGalaxy_eqFunction_10436(data, threadData);
  SpiralGalaxy_eqFunction_10437(data, threadData);
  SpiralGalaxy_eqFunction_10440(data, threadData);
  SpiralGalaxy_eqFunction_10439(data, threadData);
  SpiralGalaxy_eqFunction_10438(data, threadData);
  SpiralGalaxy_eqFunction_3899(data, threadData);
  SpiralGalaxy_eqFunction_10433(data, threadData);
  SpiralGalaxy_eqFunction_3901(data, threadData);
  SpiralGalaxy_eqFunction_10432(data, threadData);
  SpiralGalaxy_eqFunction_3903(data, threadData);
  SpiralGalaxy_eqFunction_10431(data, threadData);
  SpiralGalaxy_eqFunction_3905(data, threadData);
  SpiralGalaxy_eqFunction_10444(data, threadData);
  SpiralGalaxy_eqFunction_10445(data, threadData);
  SpiralGalaxy_eqFunction_3908(data, threadData);
  SpiralGalaxy_eqFunction_3909(data, threadData);
  SpiralGalaxy_eqFunction_10446(data, threadData);
  SpiralGalaxy_eqFunction_10447(data, threadData);
  SpiralGalaxy_eqFunction_10450(data, threadData);
  SpiralGalaxy_eqFunction_10449(data, threadData);
  SpiralGalaxy_eqFunction_10448(data, threadData);
  SpiralGalaxy_eqFunction_3915(data, threadData);
  SpiralGalaxy_eqFunction_10443(data, threadData);
  SpiralGalaxy_eqFunction_3917(data, threadData);
  SpiralGalaxy_eqFunction_10442(data, threadData);
  SpiralGalaxy_eqFunction_3919(data, threadData);
  SpiralGalaxy_eqFunction_10441(data, threadData);
  SpiralGalaxy_eqFunction_3921(data, threadData);
  SpiralGalaxy_eqFunction_10454(data, threadData);
  SpiralGalaxy_eqFunction_10455(data, threadData);
  SpiralGalaxy_eqFunction_3924(data, threadData);
  SpiralGalaxy_eqFunction_3925(data, threadData);
  SpiralGalaxy_eqFunction_10456(data, threadData);
  SpiralGalaxy_eqFunction_10457(data, threadData);
  SpiralGalaxy_eqFunction_10460(data, threadData);
  SpiralGalaxy_eqFunction_10459(data, threadData);
  SpiralGalaxy_eqFunction_10458(data, threadData);
  SpiralGalaxy_eqFunction_3931(data, threadData);
  SpiralGalaxy_eqFunction_10453(data, threadData);
  SpiralGalaxy_eqFunction_3933(data, threadData);
  SpiralGalaxy_eqFunction_10452(data, threadData);
  SpiralGalaxy_eqFunction_3935(data, threadData);
  SpiralGalaxy_eqFunction_10451(data, threadData);
  SpiralGalaxy_eqFunction_3937(data, threadData);
  SpiralGalaxy_eqFunction_10464(data, threadData);
  SpiralGalaxy_eqFunction_10465(data, threadData);
  SpiralGalaxy_eqFunction_3940(data, threadData);
  SpiralGalaxy_eqFunction_3941(data, threadData);
  SpiralGalaxy_eqFunction_10466(data, threadData);
  SpiralGalaxy_eqFunction_10467(data, threadData);
  SpiralGalaxy_eqFunction_10470(data, threadData);
  SpiralGalaxy_eqFunction_10469(data, threadData);
  SpiralGalaxy_eqFunction_10468(data, threadData);
  SpiralGalaxy_eqFunction_3947(data, threadData);
  SpiralGalaxy_eqFunction_10463(data, threadData);
  SpiralGalaxy_eqFunction_3949(data, threadData);
  SpiralGalaxy_eqFunction_10462(data, threadData);
  SpiralGalaxy_eqFunction_3951(data, threadData);
  SpiralGalaxy_eqFunction_10461(data, threadData);
  SpiralGalaxy_eqFunction_3953(data, threadData);
  SpiralGalaxy_eqFunction_10474(data, threadData);
  SpiralGalaxy_eqFunction_10475(data, threadData);
  SpiralGalaxy_eqFunction_3956(data, threadData);
  SpiralGalaxy_eqFunction_3957(data, threadData);
  SpiralGalaxy_eqFunction_10476(data, threadData);
  SpiralGalaxy_eqFunction_10477(data, threadData);
  SpiralGalaxy_eqFunction_10480(data, threadData);
  SpiralGalaxy_eqFunction_10479(data, threadData);
  SpiralGalaxy_eqFunction_10478(data, threadData);
  SpiralGalaxy_eqFunction_3963(data, threadData);
  SpiralGalaxy_eqFunction_10473(data, threadData);
  SpiralGalaxy_eqFunction_3965(data, threadData);
  SpiralGalaxy_eqFunction_10472(data, threadData);
  SpiralGalaxy_eqFunction_3967(data, threadData);
  SpiralGalaxy_eqFunction_10471(data, threadData);
  SpiralGalaxy_eqFunction_3969(data, threadData);
  SpiralGalaxy_eqFunction_10484(data, threadData);
  SpiralGalaxy_eqFunction_10485(data, threadData);
  SpiralGalaxy_eqFunction_3972(data, threadData);
  SpiralGalaxy_eqFunction_3973(data, threadData);
  SpiralGalaxy_eqFunction_10486(data, threadData);
  SpiralGalaxy_eqFunction_10487(data, threadData);
  SpiralGalaxy_eqFunction_10490(data, threadData);
  SpiralGalaxy_eqFunction_10489(data, threadData);
  SpiralGalaxy_eqFunction_10488(data, threadData);
  SpiralGalaxy_eqFunction_3979(data, threadData);
  SpiralGalaxy_eqFunction_10483(data, threadData);
  SpiralGalaxy_eqFunction_3981(data, threadData);
  SpiralGalaxy_eqFunction_10482(data, threadData);
  SpiralGalaxy_eqFunction_3983(data, threadData);
  SpiralGalaxy_eqFunction_10481(data, threadData);
  SpiralGalaxy_eqFunction_3985(data, threadData);
  SpiralGalaxy_eqFunction_10494(data, threadData);
  SpiralGalaxy_eqFunction_10495(data, threadData);
  SpiralGalaxy_eqFunction_3988(data, threadData);
  SpiralGalaxy_eqFunction_3989(data, threadData);
  SpiralGalaxy_eqFunction_10496(data, threadData);
  SpiralGalaxy_eqFunction_10497(data, threadData);
  SpiralGalaxy_eqFunction_10500(data, threadData);
  SpiralGalaxy_eqFunction_10499(data, threadData);
  SpiralGalaxy_eqFunction_10498(data, threadData);
  SpiralGalaxy_eqFunction_3995(data, threadData);
  SpiralGalaxy_eqFunction_10493(data, threadData);
  SpiralGalaxy_eqFunction_3997(data, threadData);
  SpiralGalaxy_eqFunction_10492(data, threadData);
  SpiralGalaxy_eqFunction_3999(data, threadData);
  SpiralGalaxy_eqFunction_10491(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif