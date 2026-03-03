#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 4501
type: SIMPLE_ASSIGN
x[282] = r_init[282] * cos(theta[282] + 0.001279999999999999)
*/
void SpiralGalaxy_eqFunction_4501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1781]] /* x[282] STATE(1,vx[282]) */) = ((data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */)) * (cos((data->simulationInfo->realParameter[1788] /* theta[282] PARAM */) + 0.001279999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10816(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10817(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10820(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10819(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10818(DATA *data, threadData_t *threadData);


/*
equation index: 4507
type: SIMPLE_ASSIGN
vx[282] = (-sin(theta[282])) * r_init[282] * omega_c[282]
*/
void SpiralGalaxy_eqFunction_4507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4507};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[281]] /* vx[282] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1788] /* theta[282] PARAM */)))) * (((data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */)) * ((data->simulationInfo->realParameter[786] /* omega_c[282] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10813(DATA *data, threadData_t *threadData);


/*
equation index: 4509
type: SIMPLE_ASSIGN
vy[282] = cos(theta[282]) * r_init[282] * omega_c[282]
*/
void SpiralGalaxy_eqFunction_4509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[781]] /* vy[282] STATE(1) */) = (cos((data->simulationInfo->realParameter[1788] /* theta[282] PARAM */))) * (((data->simulationInfo->realParameter[1287] /* r_init[282] PARAM */)) * ((data->simulationInfo->realParameter[786] /* omega_c[282] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10812(DATA *data, threadData_t *threadData);


/*
equation index: 4511
type: SIMPLE_ASSIGN
vz[282] = 0.0
*/
void SpiralGalaxy_eqFunction_4511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4511};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1281]] /* vz[282] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10811(DATA *data, threadData_t *threadData);


/*
equation index: 4513
type: SIMPLE_ASSIGN
z[283] = 0.005279999999999999
*/
void SpiralGalaxy_eqFunction_4513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4513};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2782]] /* z[283] STATE(1,vz[283]) */) = 0.005279999999999999;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10824(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10825(DATA *data, threadData_t *threadData);


/*
equation index: 4516
type: SIMPLE_ASSIGN
y[283] = r_init[283] * sin(theta[283] + 0.001319999999999999)
*/
void SpiralGalaxy_eqFunction_4516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4516};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2282]] /* y[283] STATE(1,vy[283]) */) = ((data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */)) * (sin((data->simulationInfo->realParameter[1789] /* theta[283] PARAM */) + 0.001319999999999999));
  TRACE_POP
}

/*
equation index: 4517
type: SIMPLE_ASSIGN
x[283] = r_init[283] * cos(theta[283] + 0.001319999999999999)
*/
void SpiralGalaxy_eqFunction_4517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1782]] /* x[283] STATE(1,vx[283]) */) = ((data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */)) * (cos((data->simulationInfo->realParameter[1789] /* theta[283] PARAM */) + 0.001319999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10826(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10827(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10830(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10829(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10828(DATA *data, threadData_t *threadData);


/*
equation index: 4523
type: SIMPLE_ASSIGN
vx[283] = (-sin(theta[283])) * r_init[283] * omega_c[283]
*/
void SpiralGalaxy_eqFunction_4523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4523};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[282]] /* vx[283] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1789] /* theta[283] PARAM */)))) * (((data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */)) * ((data->simulationInfo->realParameter[787] /* omega_c[283] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10823(DATA *data, threadData_t *threadData);


/*
equation index: 4525
type: SIMPLE_ASSIGN
vy[283] = cos(theta[283]) * r_init[283] * omega_c[283]
*/
void SpiralGalaxy_eqFunction_4525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[782]] /* vy[283] STATE(1) */) = (cos((data->simulationInfo->realParameter[1789] /* theta[283] PARAM */))) * (((data->simulationInfo->realParameter[1288] /* r_init[283] PARAM */)) * ((data->simulationInfo->realParameter[787] /* omega_c[283] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10822(DATA *data, threadData_t *threadData);


/*
equation index: 4527
type: SIMPLE_ASSIGN
vz[283] = 0.0
*/
void SpiralGalaxy_eqFunction_4527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1282]] /* vz[283] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10821(DATA *data, threadData_t *threadData);


/*
equation index: 4529
type: SIMPLE_ASSIGN
z[284] = 0.0054399999999999995
*/
void SpiralGalaxy_eqFunction_4529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4529};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2783]] /* z[284] STATE(1,vz[284]) */) = 0.0054399999999999995;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10834(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10835(DATA *data, threadData_t *threadData);


/*
equation index: 4532
type: SIMPLE_ASSIGN
y[284] = r_init[284] * sin(theta[284] + 0.001359999999999999)
*/
void SpiralGalaxy_eqFunction_4532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4532};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2283]] /* y[284] STATE(1,vy[284]) */) = ((data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */)) * (sin((data->simulationInfo->realParameter[1790] /* theta[284] PARAM */) + 0.001359999999999999));
  TRACE_POP
}

/*
equation index: 4533
type: SIMPLE_ASSIGN
x[284] = r_init[284] * cos(theta[284] + 0.001359999999999999)
*/
void SpiralGalaxy_eqFunction_4533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1783]] /* x[284] STATE(1,vx[284]) */) = ((data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */)) * (cos((data->simulationInfo->realParameter[1790] /* theta[284] PARAM */) + 0.001359999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10836(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10837(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10840(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10839(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10838(DATA *data, threadData_t *threadData);


/*
equation index: 4539
type: SIMPLE_ASSIGN
vx[284] = (-sin(theta[284])) * r_init[284] * omega_c[284]
*/
void SpiralGalaxy_eqFunction_4539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4539};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[283]] /* vx[284] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1790] /* theta[284] PARAM */)))) * (((data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */)) * ((data->simulationInfo->realParameter[788] /* omega_c[284] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10833(DATA *data, threadData_t *threadData);


/*
equation index: 4541
type: SIMPLE_ASSIGN
vy[284] = cos(theta[284]) * r_init[284] * omega_c[284]
*/
void SpiralGalaxy_eqFunction_4541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[783]] /* vy[284] STATE(1) */) = (cos((data->simulationInfo->realParameter[1790] /* theta[284] PARAM */))) * (((data->simulationInfo->realParameter[1289] /* r_init[284] PARAM */)) * ((data->simulationInfo->realParameter[788] /* omega_c[284] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10832(DATA *data, threadData_t *threadData);


/*
equation index: 4543
type: SIMPLE_ASSIGN
vz[284] = 0.0
*/
void SpiralGalaxy_eqFunction_4543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4543};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1283]] /* vz[284] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10831(DATA *data, threadData_t *threadData);


/*
equation index: 4545
type: SIMPLE_ASSIGN
z[285] = 0.0056
*/
void SpiralGalaxy_eqFunction_4545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4545};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2784]] /* z[285] STATE(1,vz[285]) */) = 0.0056;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10844(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10845(DATA *data, threadData_t *threadData);


/*
equation index: 4548
type: SIMPLE_ASSIGN
y[285] = r_init[285] * sin(theta[285] + 0.0013999999999999991)
*/
void SpiralGalaxy_eqFunction_4548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4548};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2284]] /* y[285] STATE(1,vy[285]) */) = ((data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */)) * (sin((data->simulationInfo->realParameter[1791] /* theta[285] PARAM */) + 0.0013999999999999991));
  TRACE_POP
}

/*
equation index: 4549
type: SIMPLE_ASSIGN
x[285] = r_init[285] * cos(theta[285] + 0.0013999999999999991)
*/
void SpiralGalaxy_eqFunction_4549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1784]] /* x[285] STATE(1,vx[285]) */) = ((data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */)) * (cos((data->simulationInfo->realParameter[1791] /* theta[285] PARAM */) + 0.0013999999999999991));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10846(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10847(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10850(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10849(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10848(DATA *data, threadData_t *threadData);


/*
equation index: 4555
type: SIMPLE_ASSIGN
vx[285] = (-sin(theta[285])) * r_init[285] * omega_c[285]
*/
void SpiralGalaxy_eqFunction_4555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4555};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[284]] /* vx[285] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1791] /* theta[285] PARAM */)))) * (((data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */)) * ((data->simulationInfo->realParameter[789] /* omega_c[285] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10843(DATA *data, threadData_t *threadData);


/*
equation index: 4557
type: SIMPLE_ASSIGN
vy[285] = cos(theta[285]) * r_init[285] * omega_c[285]
*/
void SpiralGalaxy_eqFunction_4557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[784]] /* vy[285] STATE(1) */) = (cos((data->simulationInfo->realParameter[1791] /* theta[285] PARAM */))) * (((data->simulationInfo->realParameter[1290] /* r_init[285] PARAM */)) * ((data->simulationInfo->realParameter[789] /* omega_c[285] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10842(DATA *data, threadData_t *threadData);


/*
equation index: 4559
type: SIMPLE_ASSIGN
vz[285] = 0.0
*/
void SpiralGalaxy_eqFunction_4559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4559};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1284]] /* vz[285] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10841(DATA *data, threadData_t *threadData);


/*
equation index: 4561
type: SIMPLE_ASSIGN
z[286] = 0.00576
*/
void SpiralGalaxy_eqFunction_4561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4561};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2785]] /* z[286] STATE(1,vz[286]) */) = 0.00576;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10854(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10855(DATA *data, threadData_t *threadData);


/*
equation index: 4564
type: SIMPLE_ASSIGN
y[286] = r_init[286] * sin(theta[286] + 0.001439999999999999)
*/
void SpiralGalaxy_eqFunction_4564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4564};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2285]] /* y[286] STATE(1,vy[286]) */) = ((data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */)) * (sin((data->simulationInfo->realParameter[1792] /* theta[286] PARAM */) + 0.001439999999999999));
  TRACE_POP
}

/*
equation index: 4565
type: SIMPLE_ASSIGN
x[286] = r_init[286] * cos(theta[286] + 0.001439999999999999)
*/
void SpiralGalaxy_eqFunction_4565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1785]] /* x[286] STATE(1,vx[286]) */) = ((data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */)) * (cos((data->simulationInfo->realParameter[1792] /* theta[286] PARAM */) + 0.001439999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10856(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10857(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10860(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10859(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10858(DATA *data, threadData_t *threadData);


/*
equation index: 4571
type: SIMPLE_ASSIGN
vx[286] = (-sin(theta[286])) * r_init[286] * omega_c[286]
*/
void SpiralGalaxy_eqFunction_4571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4571};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[285]] /* vx[286] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1792] /* theta[286] PARAM */)))) * (((data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */)) * ((data->simulationInfo->realParameter[790] /* omega_c[286] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10853(DATA *data, threadData_t *threadData);


/*
equation index: 4573
type: SIMPLE_ASSIGN
vy[286] = cos(theta[286]) * r_init[286] * omega_c[286]
*/
void SpiralGalaxy_eqFunction_4573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[785]] /* vy[286] STATE(1) */) = (cos((data->simulationInfo->realParameter[1792] /* theta[286] PARAM */))) * (((data->simulationInfo->realParameter[1291] /* r_init[286] PARAM */)) * ((data->simulationInfo->realParameter[790] /* omega_c[286] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10852(DATA *data, threadData_t *threadData);


/*
equation index: 4575
type: SIMPLE_ASSIGN
vz[286] = 0.0
*/
void SpiralGalaxy_eqFunction_4575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4575};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1285]] /* vz[286] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10851(DATA *data, threadData_t *threadData);


/*
equation index: 4577
type: SIMPLE_ASSIGN
z[287] = 0.00592
*/
void SpiralGalaxy_eqFunction_4577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4577};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2786]] /* z[287] STATE(1,vz[287]) */) = 0.00592;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10864(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10865(DATA *data, threadData_t *threadData);


/*
equation index: 4580
type: SIMPLE_ASSIGN
y[287] = r_init[287] * sin(theta[287] + 0.0014799999999999991)
*/
void SpiralGalaxy_eqFunction_4580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4580};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2286]] /* y[287] STATE(1,vy[287]) */) = ((data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */)) * (sin((data->simulationInfo->realParameter[1793] /* theta[287] PARAM */) + 0.0014799999999999991));
  TRACE_POP
}

/*
equation index: 4581
type: SIMPLE_ASSIGN
x[287] = r_init[287] * cos(theta[287] + 0.0014799999999999991)
*/
void SpiralGalaxy_eqFunction_4581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1786]] /* x[287] STATE(1,vx[287]) */) = ((data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */)) * (cos((data->simulationInfo->realParameter[1793] /* theta[287] PARAM */) + 0.0014799999999999991));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10866(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10867(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10870(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10869(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10868(DATA *data, threadData_t *threadData);


/*
equation index: 4587
type: SIMPLE_ASSIGN
vx[287] = (-sin(theta[287])) * r_init[287] * omega_c[287]
*/
void SpiralGalaxy_eqFunction_4587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4587};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[286]] /* vx[287] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1793] /* theta[287] PARAM */)))) * (((data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */)) * ((data->simulationInfo->realParameter[791] /* omega_c[287] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10863(DATA *data, threadData_t *threadData);


/*
equation index: 4589
type: SIMPLE_ASSIGN
vy[287] = cos(theta[287]) * r_init[287] * omega_c[287]
*/
void SpiralGalaxy_eqFunction_4589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[786]] /* vy[287] STATE(1) */) = (cos((data->simulationInfo->realParameter[1793] /* theta[287] PARAM */))) * (((data->simulationInfo->realParameter[1292] /* r_init[287] PARAM */)) * ((data->simulationInfo->realParameter[791] /* omega_c[287] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10862(DATA *data, threadData_t *threadData);


/*
equation index: 4591
type: SIMPLE_ASSIGN
vz[287] = 0.0
*/
void SpiralGalaxy_eqFunction_4591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4591};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1286]] /* vz[287] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10861(DATA *data, threadData_t *threadData);


/*
equation index: 4593
type: SIMPLE_ASSIGN
z[288] = 0.0060799999999999995
*/
void SpiralGalaxy_eqFunction_4593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4593};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2787]] /* z[288] STATE(1,vz[288]) */) = 0.0060799999999999995;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10874(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10875(DATA *data, threadData_t *threadData);


/*
equation index: 4596
type: SIMPLE_ASSIGN
y[288] = r_init[288] * sin(theta[288] + 0.0015199999999999992)
*/
void SpiralGalaxy_eqFunction_4596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4596};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2287]] /* y[288] STATE(1,vy[288]) */) = ((data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */)) * (sin((data->simulationInfo->realParameter[1794] /* theta[288] PARAM */) + 0.0015199999999999992));
  TRACE_POP
}

/*
equation index: 4597
type: SIMPLE_ASSIGN
x[288] = r_init[288] * cos(theta[288] + 0.0015199999999999992)
*/
void SpiralGalaxy_eqFunction_4597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1787]] /* x[288] STATE(1,vx[288]) */) = ((data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */)) * (cos((data->simulationInfo->realParameter[1794] /* theta[288] PARAM */) + 0.0015199999999999992));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10876(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10877(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10880(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10879(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10878(DATA *data, threadData_t *threadData);


/*
equation index: 4603
type: SIMPLE_ASSIGN
vx[288] = (-sin(theta[288])) * r_init[288] * omega_c[288]
*/
void SpiralGalaxy_eqFunction_4603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4603};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[287]] /* vx[288] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1794] /* theta[288] PARAM */)))) * (((data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */)) * ((data->simulationInfo->realParameter[792] /* omega_c[288] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10873(DATA *data, threadData_t *threadData);


/*
equation index: 4605
type: SIMPLE_ASSIGN
vy[288] = cos(theta[288]) * r_init[288] * omega_c[288]
*/
void SpiralGalaxy_eqFunction_4605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[787]] /* vy[288] STATE(1) */) = (cos((data->simulationInfo->realParameter[1794] /* theta[288] PARAM */))) * (((data->simulationInfo->realParameter[1293] /* r_init[288] PARAM */)) * ((data->simulationInfo->realParameter[792] /* omega_c[288] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10872(DATA *data, threadData_t *threadData);


/*
equation index: 4607
type: SIMPLE_ASSIGN
vz[288] = 0.0
*/
void SpiralGalaxy_eqFunction_4607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4607};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1287]] /* vz[288] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10871(DATA *data, threadData_t *threadData);


/*
equation index: 4609
type: SIMPLE_ASSIGN
z[289] = 0.00624
*/
void SpiralGalaxy_eqFunction_4609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4609};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2788]] /* z[289] STATE(1,vz[289]) */) = 0.00624;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10884(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10885(DATA *data, threadData_t *threadData);


/*
equation index: 4612
type: SIMPLE_ASSIGN
y[289] = r_init[289] * sin(theta[289] + 0.001559999999999999)
*/
void SpiralGalaxy_eqFunction_4612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4612};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2288]] /* y[289] STATE(1,vy[289]) */) = ((data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */)) * (sin((data->simulationInfo->realParameter[1795] /* theta[289] PARAM */) + 0.001559999999999999));
  TRACE_POP
}

/*
equation index: 4613
type: SIMPLE_ASSIGN
x[289] = r_init[289] * cos(theta[289] + 0.001559999999999999)
*/
void SpiralGalaxy_eqFunction_4613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1788]] /* x[289] STATE(1,vx[289]) */) = ((data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */)) * (cos((data->simulationInfo->realParameter[1795] /* theta[289] PARAM */) + 0.001559999999999999));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10886(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10887(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10890(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10889(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10888(DATA *data, threadData_t *threadData);


/*
equation index: 4619
type: SIMPLE_ASSIGN
vx[289] = (-sin(theta[289])) * r_init[289] * omega_c[289]
*/
void SpiralGalaxy_eqFunction_4619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4619};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[288]] /* vx[289] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1795] /* theta[289] PARAM */)))) * (((data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */)) * ((data->simulationInfo->realParameter[793] /* omega_c[289] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10883(DATA *data, threadData_t *threadData);


/*
equation index: 4621
type: SIMPLE_ASSIGN
vy[289] = cos(theta[289]) * r_init[289] * omega_c[289]
*/
void SpiralGalaxy_eqFunction_4621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[788]] /* vy[289] STATE(1) */) = (cos((data->simulationInfo->realParameter[1795] /* theta[289] PARAM */))) * (((data->simulationInfo->realParameter[1294] /* r_init[289] PARAM */)) * ((data->simulationInfo->realParameter[793] /* omega_c[289] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10882(DATA *data, threadData_t *threadData);


/*
equation index: 4623
type: SIMPLE_ASSIGN
vz[289] = 0.0
*/
void SpiralGalaxy_eqFunction_4623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4623};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1288]] /* vz[289] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10881(DATA *data, threadData_t *threadData);


/*
equation index: 4625
type: SIMPLE_ASSIGN
z[290] = 0.0064
*/
void SpiralGalaxy_eqFunction_4625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4625};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2789]] /* z[290] STATE(1,vz[290]) */) = 0.0064;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10894(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10895(DATA *data, threadData_t *threadData);


/*
equation index: 4628
type: SIMPLE_ASSIGN
y[290] = r_init[290] * sin(theta[290] + 0.0015999999999999992)
*/
void SpiralGalaxy_eqFunction_4628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4628};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2289]] /* y[290] STATE(1,vy[290]) */) = ((data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */)) * (sin((data->simulationInfo->realParameter[1796] /* theta[290] PARAM */) + 0.0015999999999999992));
  TRACE_POP
}

/*
equation index: 4629
type: SIMPLE_ASSIGN
x[290] = r_init[290] * cos(theta[290] + 0.0015999999999999992)
*/
void SpiralGalaxy_eqFunction_4629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1789]] /* x[290] STATE(1,vx[290]) */) = ((data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */)) * (cos((data->simulationInfo->realParameter[1796] /* theta[290] PARAM */) + 0.0015999999999999992));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10896(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10897(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10900(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10899(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10898(DATA *data, threadData_t *threadData);


/*
equation index: 4635
type: SIMPLE_ASSIGN
vx[290] = (-sin(theta[290])) * r_init[290] * omega_c[290]
*/
void SpiralGalaxy_eqFunction_4635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4635};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[289]] /* vx[290] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1796] /* theta[290] PARAM */)))) * (((data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */)) * ((data->simulationInfo->realParameter[794] /* omega_c[290] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10893(DATA *data, threadData_t *threadData);


/*
equation index: 4637
type: SIMPLE_ASSIGN
vy[290] = cos(theta[290]) * r_init[290] * omega_c[290]
*/
void SpiralGalaxy_eqFunction_4637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[789]] /* vy[290] STATE(1) */) = (cos((data->simulationInfo->realParameter[1796] /* theta[290] PARAM */))) * (((data->simulationInfo->realParameter[1295] /* r_init[290] PARAM */)) * ((data->simulationInfo->realParameter[794] /* omega_c[290] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10892(DATA *data, threadData_t *threadData);


/*
equation index: 4639
type: SIMPLE_ASSIGN
vz[290] = 0.0
*/
void SpiralGalaxy_eqFunction_4639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4639};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1289]] /* vz[290] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10891(DATA *data, threadData_t *threadData);


/*
equation index: 4641
type: SIMPLE_ASSIGN
z[291] = 0.00656
*/
void SpiralGalaxy_eqFunction_4641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4641};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2790]] /* z[291] STATE(1,vz[291]) */) = 0.00656;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10904(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10905(DATA *data, threadData_t *threadData);


/*
equation index: 4644
type: SIMPLE_ASSIGN
y[291] = r_init[291] * sin(theta[291] + 0.0016399999999999993)
*/
void SpiralGalaxy_eqFunction_4644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4644};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2290]] /* y[291] STATE(1,vy[291]) */) = ((data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */)) * (sin((data->simulationInfo->realParameter[1797] /* theta[291] PARAM */) + 0.0016399999999999993));
  TRACE_POP
}

/*
equation index: 4645
type: SIMPLE_ASSIGN
x[291] = r_init[291] * cos(theta[291] + 0.0016399999999999993)
*/
void SpiralGalaxy_eqFunction_4645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1790]] /* x[291] STATE(1,vx[291]) */) = ((data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */)) * (cos((data->simulationInfo->realParameter[1797] /* theta[291] PARAM */) + 0.0016399999999999993));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10906(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10907(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10910(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10909(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10908(DATA *data, threadData_t *threadData);


/*
equation index: 4651
type: SIMPLE_ASSIGN
vx[291] = (-sin(theta[291])) * r_init[291] * omega_c[291]
*/
void SpiralGalaxy_eqFunction_4651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4651};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[290]] /* vx[291] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1797] /* theta[291] PARAM */)))) * (((data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */)) * ((data->simulationInfo->realParameter[795] /* omega_c[291] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10903(DATA *data, threadData_t *threadData);


/*
equation index: 4653
type: SIMPLE_ASSIGN
vy[291] = cos(theta[291]) * r_init[291] * omega_c[291]
*/
void SpiralGalaxy_eqFunction_4653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[790]] /* vy[291] STATE(1) */) = (cos((data->simulationInfo->realParameter[1797] /* theta[291] PARAM */))) * (((data->simulationInfo->realParameter[1296] /* r_init[291] PARAM */)) * ((data->simulationInfo->realParameter[795] /* omega_c[291] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10902(DATA *data, threadData_t *threadData);


/*
equation index: 4655
type: SIMPLE_ASSIGN
vz[291] = 0.0
*/
void SpiralGalaxy_eqFunction_4655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4655};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1290]] /* vz[291] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10901(DATA *data, threadData_t *threadData);


/*
equation index: 4657
type: SIMPLE_ASSIGN
z[292] = 0.00672
*/
void SpiralGalaxy_eqFunction_4657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4657};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2791]] /* z[292] STATE(1,vz[292]) */) = 0.00672;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10914(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10915(DATA *data, threadData_t *threadData);


/*
equation index: 4660
type: SIMPLE_ASSIGN
y[292] = r_init[292] * sin(theta[292] + 0.0016799999999999992)
*/
void SpiralGalaxy_eqFunction_4660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4660};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2291]] /* y[292] STATE(1,vy[292]) */) = ((data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */)) * (sin((data->simulationInfo->realParameter[1798] /* theta[292] PARAM */) + 0.0016799999999999992));
  TRACE_POP
}

/*
equation index: 4661
type: SIMPLE_ASSIGN
x[292] = r_init[292] * cos(theta[292] + 0.0016799999999999992)
*/
void SpiralGalaxy_eqFunction_4661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1791]] /* x[292] STATE(1,vx[292]) */) = ((data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */)) * (cos((data->simulationInfo->realParameter[1798] /* theta[292] PARAM */) + 0.0016799999999999992));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10916(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10917(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10920(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10919(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10918(DATA *data, threadData_t *threadData);


/*
equation index: 4667
type: SIMPLE_ASSIGN
vx[292] = (-sin(theta[292])) * r_init[292] * omega_c[292]
*/
void SpiralGalaxy_eqFunction_4667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4667};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[291]] /* vx[292] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1798] /* theta[292] PARAM */)))) * (((data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */)) * ((data->simulationInfo->realParameter[796] /* omega_c[292] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10913(DATA *data, threadData_t *threadData);


/*
equation index: 4669
type: SIMPLE_ASSIGN
vy[292] = cos(theta[292]) * r_init[292] * omega_c[292]
*/
void SpiralGalaxy_eqFunction_4669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[791]] /* vy[292] STATE(1) */) = (cos((data->simulationInfo->realParameter[1798] /* theta[292] PARAM */))) * (((data->simulationInfo->realParameter[1297] /* r_init[292] PARAM */)) * ((data->simulationInfo->realParameter[796] /* omega_c[292] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10912(DATA *data, threadData_t *threadData);


/*
equation index: 4671
type: SIMPLE_ASSIGN
vz[292] = 0.0
*/
void SpiralGalaxy_eqFunction_4671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4671};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1291]] /* vz[292] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10911(DATA *data, threadData_t *threadData);


/*
equation index: 4673
type: SIMPLE_ASSIGN
z[293] = 0.00688
*/
void SpiralGalaxy_eqFunction_4673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4673};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2792]] /* z[293] STATE(1,vz[293]) */) = 0.00688;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10924(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10925(DATA *data, threadData_t *threadData);


/*
equation index: 4676
type: SIMPLE_ASSIGN
y[293] = r_init[293] * sin(theta[293] + 0.0017199999999999993)
*/
void SpiralGalaxy_eqFunction_4676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4676};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2292]] /* y[293] STATE(1,vy[293]) */) = ((data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */)) * (sin((data->simulationInfo->realParameter[1799] /* theta[293] PARAM */) + 0.0017199999999999993));
  TRACE_POP
}

/*
equation index: 4677
type: SIMPLE_ASSIGN
x[293] = r_init[293] * cos(theta[293] + 0.0017199999999999993)
*/
void SpiralGalaxy_eqFunction_4677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1792]] /* x[293] STATE(1,vx[293]) */) = ((data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */)) * (cos((data->simulationInfo->realParameter[1799] /* theta[293] PARAM */) + 0.0017199999999999993));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10926(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10927(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10930(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10929(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10928(DATA *data, threadData_t *threadData);


/*
equation index: 4683
type: SIMPLE_ASSIGN
vx[293] = (-sin(theta[293])) * r_init[293] * omega_c[293]
*/
void SpiralGalaxy_eqFunction_4683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4683};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[292]] /* vx[293] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1799] /* theta[293] PARAM */)))) * (((data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */)) * ((data->simulationInfo->realParameter[797] /* omega_c[293] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10923(DATA *data, threadData_t *threadData);


/*
equation index: 4685
type: SIMPLE_ASSIGN
vy[293] = cos(theta[293]) * r_init[293] * omega_c[293]
*/
void SpiralGalaxy_eqFunction_4685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[792]] /* vy[293] STATE(1) */) = (cos((data->simulationInfo->realParameter[1799] /* theta[293] PARAM */))) * (((data->simulationInfo->realParameter[1298] /* r_init[293] PARAM */)) * ((data->simulationInfo->realParameter[797] /* omega_c[293] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10922(DATA *data, threadData_t *threadData);


/*
equation index: 4687
type: SIMPLE_ASSIGN
vz[293] = 0.0
*/
void SpiralGalaxy_eqFunction_4687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4687};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1292]] /* vz[293] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10921(DATA *data, threadData_t *threadData);


/*
equation index: 4689
type: SIMPLE_ASSIGN
z[294] = 0.00704
*/
void SpiralGalaxy_eqFunction_4689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4689};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2793]] /* z[294] STATE(1,vz[294]) */) = 0.00704;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10934(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10935(DATA *data, threadData_t *threadData);


/*
equation index: 4692
type: SIMPLE_ASSIGN
y[294] = r_init[294] * sin(theta[294] + 0.0017599999999999994)
*/
void SpiralGalaxy_eqFunction_4692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4692};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2293]] /* y[294] STATE(1,vy[294]) */) = ((data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */)) * (sin((data->simulationInfo->realParameter[1800] /* theta[294] PARAM */) + 0.0017599999999999994));
  TRACE_POP
}

/*
equation index: 4693
type: SIMPLE_ASSIGN
x[294] = r_init[294] * cos(theta[294] + 0.0017599999999999994)
*/
void SpiralGalaxy_eqFunction_4693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1793]] /* x[294] STATE(1,vx[294]) */) = ((data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */)) * (cos((data->simulationInfo->realParameter[1800] /* theta[294] PARAM */) + 0.0017599999999999994));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10936(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10937(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10940(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10939(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10938(DATA *data, threadData_t *threadData);


/*
equation index: 4699
type: SIMPLE_ASSIGN
vx[294] = (-sin(theta[294])) * r_init[294] * omega_c[294]
*/
void SpiralGalaxy_eqFunction_4699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4699};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[293]] /* vx[294] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1800] /* theta[294] PARAM */)))) * (((data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */)) * ((data->simulationInfo->realParameter[798] /* omega_c[294] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10933(DATA *data, threadData_t *threadData);


/*
equation index: 4701
type: SIMPLE_ASSIGN
vy[294] = cos(theta[294]) * r_init[294] * omega_c[294]
*/
void SpiralGalaxy_eqFunction_4701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[793]] /* vy[294] STATE(1) */) = (cos((data->simulationInfo->realParameter[1800] /* theta[294] PARAM */))) * (((data->simulationInfo->realParameter[1299] /* r_init[294] PARAM */)) * ((data->simulationInfo->realParameter[798] /* omega_c[294] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10932(DATA *data, threadData_t *threadData);


/*
equation index: 4703
type: SIMPLE_ASSIGN
vz[294] = 0.0
*/
void SpiralGalaxy_eqFunction_4703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4703};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1293]] /* vz[294] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10931(DATA *data, threadData_t *threadData);


/*
equation index: 4705
type: SIMPLE_ASSIGN
z[295] = 0.007200000000000001
*/
void SpiralGalaxy_eqFunction_4705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4705};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2794]] /* z[295] STATE(1,vz[295]) */) = 0.007200000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10944(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10945(DATA *data, threadData_t *threadData);


/*
equation index: 4708
type: SIMPLE_ASSIGN
y[295] = r_init[295] * sin(theta[295] + 0.0017999999999999995)
*/
void SpiralGalaxy_eqFunction_4708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4708};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2294]] /* y[295] STATE(1,vy[295]) */) = ((data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */)) * (sin((data->simulationInfo->realParameter[1801] /* theta[295] PARAM */) + 0.0017999999999999995));
  TRACE_POP
}

/*
equation index: 4709
type: SIMPLE_ASSIGN
x[295] = r_init[295] * cos(theta[295] + 0.0017999999999999995)
*/
void SpiralGalaxy_eqFunction_4709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1794]] /* x[295] STATE(1,vx[295]) */) = ((data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */)) * (cos((data->simulationInfo->realParameter[1801] /* theta[295] PARAM */) + 0.0017999999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10946(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10947(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10950(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10949(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10948(DATA *data, threadData_t *threadData);


/*
equation index: 4715
type: SIMPLE_ASSIGN
vx[295] = (-sin(theta[295])) * r_init[295] * omega_c[295]
*/
void SpiralGalaxy_eqFunction_4715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4715};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[294]] /* vx[295] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1801] /* theta[295] PARAM */)))) * (((data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */)) * ((data->simulationInfo->realParameter[799] /* omega_c[295] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10943(DATA *data, threadData_t *threadData);


/*
equation index: 4717
type: SIMPLE_ASSIGN
vy[295] = cos(theta[295]) * r_init[295] * omega_c[295]
*/
void SpiralGalaxy_eqFunction_4717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[794]] /* vy[295] STATE(1) */) = (cos((data->simulationInfo->realParameter[1801] /* theta[295] PARAM */))) * (((data->simulationInfo->realParameter[1300] /* r_init[295] PARAM */)) * ((data->simulationInfo->realParameter[799] /* omega_c[295] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10942(DATA *data, threadData_t *threadData);


/*
equation index: 4719
type: SIMPLE_ASSIGN
vz[295] = 0.0
*/
void SpiralGalaxy_eqFunction_4719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4719};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1294]] /* vz[295] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10941(DATA *data, threadData_t *threadData);


/*
equation index: 4721
type: SIMPLE_ASSIGN
z[296] = 0.00736
*/
void SpiralGalaxy_eqFunction_4721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4721};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2795]] /* z[296] STATE(1,vz[296]) */) = 0.00736;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10954(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10955(DATA *data, threadData_t *threadData);


/*
equation index: 4724
type: SIMPLE_ASSIGN
y[296] = r_init[296] * sin(theta[296] + 0.0018399999999999994)
*/
void SpiralGalaxy_eqFunction_4724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4724};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2295]] /* y[296] STATE(1,vy[296]) */) = ((data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */)) * (sin((data->simulationInfo->realParameter[1802] /* theta[296] PARAM */) + 0.0018399999999999994));
  TRACE_POP
}

/*
equation index: 4725
type: SIMPLE_ASSIGN
x[296] = r_init[296] * cos(theta[296] + 0.0018399999999999994)
*/
void SpiralGalaxy_eqFunction_4725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1795]] /* x[296] STATE(1,vx[296]) */) = ((data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */)) * (cos((data->simulationInfo->realParameter[1802] /* theta[296] PARAM */) + 0.0018399999999999994));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10956(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10957(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10960(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10959(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10958(DATA *data, threadData_t *threadData);


/*
equation index: 4731
type: SIMPLE_ASSIGN
vx[296] = (-sin(theta[296])) * r_init[296] * omega_c[296]
*/
void SpiralGalaxy_eqFunction_4731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4731};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[295]] /* vx[296] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1802] /* theta[296] PARAM */)))) * (((data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */)) * ((data->simulationInfo->realParameter[800] /* omega_c[296] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10953(DATA *data, threadData_t *threadData);


/*
equation index: 4733
type: SIMPLE_ASSIGN
vy[296] = cos(theta[296]) * r_init[296] * omega_c[296]
*/
void SpiralGalaxy_eqFunction_4733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[795]] /* vy[296] STATE(1) */) = (cos((data->simulationInfo->realParameter[1802] /* theta[296] PARAM */))) * (((data->simulationInfo->realParameter[1301] /* r_init[296] PARAM */)) * ((data->simulationInfo->realParameter[800] /* omega_c[296] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10952(DATA *data, threadData_t *threadData);


/*
equation index: 4735
type: SIMPLE_ASSIGN
vz[296] = 0.0
*/
void SpiralGalaxy_eqFunction_4735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4735};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1295]] /* vz[296] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10951(DATA *data, threadData_t *threadData);


/*
equation index: 4737
type: SIMPLE_ASSIGN
z[297] = 0.007520000000000001
*/
void SpiralGalaxy_eqFunction_4737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4737};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2796]] /* z[297] STATE(1,vz[297]) */) = 0.007520000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10964(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10965(DATA *data, threadData_t *threadData);


/*
equation index: 4740
type: SIMPLE_ASSIGN
y[297] = r_init[297] * sin(theta[297] + 0.0018799999999999995)
*/
void SpiralGalaxy_eqFunction_4740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4740};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2296]] /* y[297] STATE(1,vy[297]) */) = ((data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */)) * (sin((data->simulationInfo->realParameter[1803] /* theta[297] PARAM */) + 0.0018799999999999995));
  TRACE_POP
}

/*
equation index: 4741
type: SIMPLE_ASSIGN
x[297] = r_init[297] * cos(theta[297] + 0.0018799999999999995)
*/
void SpiralGalaxy_eqFunction_4741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1796]] /* x[297] STATE(1,vx[297]) */) = ((data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */)) * (cos((data->simulationInfo->realParameter[1803] /* theta[297] PARAM */) + 0.0018799999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10966(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10967(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10970(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10969(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10968(DATA *data, threadData_t *threadData);


/*
equation index: 4747
type: SIMPLE_ASSIGN
vx[297] = (-sin(theta[297])) * r_init[297] * omega_c[297]
*/
void SpiralGalaxy_eqFunction_4747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4747};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[296]] /* vx[297] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1803] /* theta[297] PARAM */)))) * (((data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */)) * ((data->simulationInfo->realParameter[801] /* omega_c[297] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10963(DATA *data, threadData_t *threadData);


/*
equation index: 4749
type: SIMPLE_ASSIGN
vy[297] = cos(theta[297]) * r_init[297] * omega_c[297]
*/
void SpiralGalaxy_eqFunction_4749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[796]] /* vy[297] STATE(1) */) = (cos((data->simulationInfo->realParameter[1803] /* theta[297] PARAM */))) * (((data->simulationInfo->realParameter[1302] /* r_init[297] PARAM */)) * ((data->simulationInfo->realParameter[801] /* omega_c[297] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10962(DATA *data, threadData_t *threadData);


/*
equation index: 4751
type: SIMPLE_ASSIGN
vz[297] = 0.0
*/
void SpiralGalaxy_eqFunction_4751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4751};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1296]] /* vz[297] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10961(DATA *data, threadData_t *threadData);


/*
equation index: 4753
type: SIMPLE_ASSIGN
z[298] = 0.00768
*/
void SpiralGalaxy_eqFunction_4753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4753};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2797]] /* z[298] STATE(1,vz[298]) */) = 0.00768;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10974(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10975(DATA *data, threadData_t *threadData);


/*
equation index: 4756
type: SIMPLE_ASSIGN
y[298] = r_init[298] * sin(theta[298] + 0.0019199999999999996)
*/
void SpiralGalaxy_eqFunction_4756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4756};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2297]] /* y[298] STATE(1,vy[298]) */) = ((data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */)) * (sin((data->simulationInfo->realParameter[1804] /* theta[298] PARAM */) + 0.0019199999999999996));
  TRACE_POP
}

/*
equation index: 4757
type: SIMPLE_ASSIGN
x[298] = r_init[298] * cos(theta[298] + 0.0019199999999999996)
*/
void SpiralGalaxy_eqFunction_4757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1797]] /* x[298] STATE(1,vx[298]) */) = ((data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */)) * (cos((data->simulationInfo->realParameter[1804] /* theta[298] PARAM */) + 0.0019199999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10976(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10977(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10980(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10979(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10978(DATA *data, threadData_t *threadData);


/*
equation index: 4763
type: SIMPLE_ASSIGN
vx[298] = (-sin(theta[298])) * r_init[298] * omega_c[298]
*/
void SpiralGalaxy_eqFunction_4763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4763};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[297]] /* vx[298] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1804] /* theta[298] PARAM */)))) * (((data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */)) * ((data->simulationInfo->realParameter[802] /* omega_c[298] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10973(DATA *data, threadData_t *threadData);


/*
equation index: 4765
type: SIMPLE_ASSIGN
vy[298] = cos(theta[298]) * r_init[298] * omega_c[298]
*/
void SpiralGalaxy_eqFunction_4765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[797]] /* vy[298] STATE(1) */) = (cos((data->simulationInfo->realParameter[1804] /* theta[298] PARAM */))) * (((data->simulationInfo->realParameter[1303] /* r_init[298] PARAM */)) * ((data->simulationInfo->realParameter[802] /* omega_c[298] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10972(DATA *data, threadData_t *threadData);


/*
equation index: 4767
type: SIMPLE_ASSIGN
vz[298] = 0.0
*/
void SpiralGalaxy_eqFunction_4767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4767};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1297]] /* vz[298] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10971(DATA *data, threadData_t *threadData);


/*
equation index: 4769
type: SIMPLE_ASSIGN
z[299] = 0.00784
*/
void SpiralGalaxy_eqFunction_4769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4769};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2798]] /* z[299] STATE(1,vz[299]) */) = 0.00784;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10984(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10985(DATA *data, threadData_t *threadData);


/*
equation index: 4772
type: SIMPLE_ASSIGN
y[299] = r_init[299] * sin(theta[299] + 0.0019599999999999995)
*/
void SpiralGalaxy_eqFunction_4772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4772};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2298]] /* y[299] STATE(1,vy[299]) */) = ((data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */)) * (sin((data->simulationInfo->realParameter[1805] /* theta[299] PARAM */) + 0.0019599999999999995));
  TRACE_POP
}

/*
equation index: 4773
type: SIMPLE_ASSIGN
x[299] = r_init[299] * cos(theta[299] + 0.0019599999999999995)
*/
void SpiralGalaxy_eqFunction_4773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1798]] /* x[299] STATE(1,vx[299]) */) = ((data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */)) * (cos((data->simulationInfo->realParameter[1805] /* theta[299] PARAM */) + 0.0019599999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10986(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10987(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10990(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10989(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10988(DATA *data, threadData_t *threadData);


/*
equation index: 4779
type: SIMPLE_ASSIGN
vx[299] = (-sin(theta[299])) * r_init[299] * omega_c[299]
*/
void SpiralGalaxy_eqFunction_4779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4779};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[298]] /* vx[299] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1805] /* theta[299] PARAM */)))) * (((data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */)) * ((data->simulationInfo->realParameter[803] /* omega_c[299] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10983(DATA *data, threadData_t *threadData);


/*
equation index: 4781
type: SIMPLE_ASSIGN
vy[299] = cos(theta[299]) * r_init[299] * omega_c[299]
*/
void SpiralGalaxy_eqFunction_4781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[798]] /* vy[299] STATE(1) */) = (cos((data->simulationInfo->realParameter[1805] /* theta[299] PARAM */))) * (((data->simulationInfo->realParameter[1304] /* r_init[299] PARAM */)) * ((data->simulationInfo->realParameter[803] /* omega_c[299] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10982(DATA *data, threadData_t *threadData);


/*
equation index: 4783
type: SIMPLE_ASSIGN
vz[299] = 0.0
*/
void SpiralGalaxy_eqFunction_4783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4783};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1298]] /* vz[299] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10981(DATA *data, threadData_t *threadData);


/*
equation index: 4785
type: SIMPLE_ASSIGN
z[300] = 0.008
*/
void SpiralGalaxy_eqFunction_4785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4785};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2799]] /* z[300] STATE(1,vz[300]) */) = 0.008;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10994(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10995(DATA *data, threadData_t *threadData);


/*
equation index: 4788
type: SIMPLE_ASSIGN
y[300] = r_init[300] * sin(theta[300] + 0.0019999999999999996)
*/
void SpiralGalaxy_eqFunction_4788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4788};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2299]] /* y[300] STATE(1,vy[300]) */) = ((data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */)) * (sin((data->simulationInfo->realParameter[1806] /* theta[300] PARAM */) + 0.0019999999999999996));
  TRACE_POP
}

/*
equation index: 4789
type: SIMPLE_ASSIGN
x[300] = r_init[300] * cos(theta[300] + 0.0019999999999999996)
*/
void SpiralGalaxy_eqFunction_4789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1799]] /* x[300] STATE(1,vx[300]) */) = ((data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */)) * (cos((data->simulationInfo->realParameter[1806] /* theta[300] PARAM */) + 0.0019999999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10996(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10997(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11000(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10999(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_10998(DATA *data, threadData_t *threadData);


/*
equation index: 4795
type: SIMPLE_ASSIGN
vx[300] = (-sin(theta[300])) * r_init[300] * omega_c[300]
*/
void SpiralGalaxy_eqFunction_4795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4795};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[299]] /* vx[300] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1806] /* theta[300] PARAM */)))) * (((data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */)) * ((data->simulationInfo->realParameter[804] /* omega_c[300] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10993(DATA *data, threadData_t *threadData);


/*
equation index: 4797
type: SIMPLE_ASSIGN
vy[300] = cos(theta[300]) * r_init[300] * omega_c[300]
*/
void SpiralGalaxy_eqFunction_4797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[799]] /* vy[300] STATE(1) */) = (cos((data->simulationInfo->realParameter[1806] /* theta[300] PARAM */))) * (((data->simulationInfo->realParameter[1305] /* r_init[300] PARAM */)) * ((data->simulationInfo->realParameter[804] /* omega_c[300] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10992(DATA *data, threadData_t *threadData);


/*
equation index: 4799
type: SIMPLE_ASSIGN
vz[300] = 0.0
*/
void SpiralGalaxy_eqFunction_4799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4799};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1299]] /* vz[300] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_10991(DATA *data, threadData_t *threadData);


/*
equation index: 4801
type: SIMPLE_ASSIGN
z[301] = 0.00816
*/
void SpiralGalaxy_eqFunction_4801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4801};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2800]] /* z[301] STATE(1,vz[301]) */) = 0.00816;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11004(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11005(DATA *data, threadData_t *threadData);


/*
equation index: 4804
type: SIMPLE_ASSIGN
y[301] = r_init[301] * sin(theta[301] + 0.0020399999999999997)
*/
void SpiralGalaxy_eqFunction_4804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4804};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2300]] /* y[301] STATE(1,vy[301]) */) = ((data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */)) * (sin((data->simulationInfo->realParameter[1807] /* theta[301] PARAM */) + 0.0020399999999999997));
  TRACE_POP
}

/*
equation index: 4805
type: SIMPLE_ASSIGN
x[301] = r_init[301] * cos(theta[301] + 0.0020399999999999997)
*/
void SpiralGalaxy_eqFunction_4805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1800]] /* x[301] STATE(1,vx[301]) */) = ((data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */)) * (cos((data->simulationInfo->realParameter[1807] /* theta[301] PARAM */) + 0.0020399999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11006(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11007(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11010(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11009(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11008(DATA *data, threadData_t *threadData);


/*
equation index: 4811
type: SIMPLE_ASSIGN
vx[301] = (-sin(theta[301])) * r_init[301] * omega_c[301]
*/
void SpiralGalaxy_eqFunction_4811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4811};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[300]] /* vx[301] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1807] /* theta[301] PARAM */)))) * (((data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */)) * ((data->simulationInfo->realParameter[805] /* omega_c[301] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11003(DATA *data, threadData_t *threadData);


/*
equation index: 4813
type: SIMPLE_ASSIGN
vy[301] = cos(theta[301]) * r_init[301] * omega_c[301]
*/
void SpiralGalaxy_eqFunction_4813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[800]] /* vy[301] STATE(1) */) = (cos((data->simulationInfo->realParameter[1807] /* theta[301] PARAM */))) * (((data->simulationInfo->realParameter[1306] /* r_init[301] PARAM */)) * ((data->simulationInfo->realParameter[805] /* omega_c[301] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11002(DATA *data, threadData_t *threadData);


/*
equation index: 4815
type: SIMPLE_ASSIGN
vz[301] = 0.0
*/
void SpiralGalaxy_eqFunction_4815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4815};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1300]] /* vz[301] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11001(DATA *data, threadData_t *threadData);


/*
equation index: 4817
type: SIMPLE_ASSIGN
z[302] = 0.008320000000000001
*/
void SpiralGalaxy_eqFunction_4817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4817};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2801]] /* z[302] STATE(1,vz[302]) */) = 0.008320000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11014(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11015(DATA *data, threadData_t *threadData);


/*
equation index: 4820
type: SIMPLE_ASSIGN
y[302] = r_init[302] * sin(theta[302] + 0.00208)
*/
void SpiralGalaxy_eqFunction_4820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4820};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2301]] /* y[302] STATE(1,vy[302]) */) = ((data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */)) * (sin((data->simulationInfo->realParameter[1808] /* theta[302] PARAM */) + 0.00208));
  TRACE_POP
}

/*
equation index: 4821
type: SIMPLE_ASSIGN
x[302] = r_init[302] * cos(theta[302] + 0.00208)
*/
void SpiralGalaxy_eqFunction_4821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1801]] /* x[302] STATE(1,vx[302]) */) = ((data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */)) * (cos((data->simulationInfo->realParameter[1808] /* theta[302] PARAM */) + 0.00208));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11016(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11017(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11020(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11019(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11018(DATA *data, threadData_t *threadData);


/*
equation index: 4827
type: SIMPLE_ASSIGN
vx[302] = (-sin(theta[302])) * r_init[302] * omega_c[302]
*/
void SpiralGalaxy_eqFunction_4827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4827};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[301]] /* vx[302] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1808] /* theta[302] PARAM */)))) * (((data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */)) * ((data->simulationInfo->realParameter[806] /* omega_c[302] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11013(DATA *data, threadData_t *threadData);


/*
equation index: 4829
type: SIMPLE_ASSIGN
vy[302] = cos(theta[302]) * r_init[302] * omega_c[302]
*/
void SpiralGalaxy_eqFunction_4829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[801]] /* vy[302] STATE(1) */) = (cos((data->simulationInfo->realParameter[1808] /* theta[302] PARAM */))) * (((data->simulationInfo->realParameter[1307] /* r_init[302] PARAM */)) * ((data->simulationInfo->realParameter[806] /* omega_c[302] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11012(DATA *data, threadData_t *threadData);


/*
equation index: 4831
type: SIMPLE_ASSIGN
vz[302] = 0.0
*/
void SpiralGalaxy_eqFunction_4831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4831};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1301]] /* vz[302] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11011(DATA *data, threadData_t *threadData);


/*
equation index: 4833
type: SIMPLE_ASSIGN
z[303] = 0.008480000000000001
*/
void SpiralGalaxy_eqFunction_4833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4833};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2802]] /* z[303] STATE(1,vz[303]) */) = 0.008480000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11024(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11025(DATA *data, threadData_t *threadData);


/*
equation index: 4836
type: SIMPLE_ASSIGN
y[303] = r_init[303] * sin(theta[303] + 0.00212)
*/
void SpiralGalaxy_eqFunction_4836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4836};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2302]] /* y[303] STATE(1,vy[303]) */) = ((data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */)) * (sin((data->simulationInfo->realParameter[1809] /* theta[303] PARAM */) + 0.00212));
  TRACE_POP
}

/*
equation index: 4837
type: SIMPLE_ASSIGN
x[303] = r_init[303] * cos(theta[303] + 0.00212)
*/
void SpiralGalaxy_eqFunction_4837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1802]] /* x[303] STATE(1,vx[303]) */) = ((data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */)) * (cos((data->simulationInfo->realParameter[1809] /* theta[303] PARAM */) + 0.00212));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11026(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11027(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11030(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11029(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11028(DATA *data, threadData_t *threadData);


/*
equation index: 4843
type: SIMPLE_ASSIGN
vx[303] = (-sin(theta[303])) * r_init[303] * omega_c[303]
*/
void SpiralGalaxy_eqFunction_4843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4843};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[302]] /* vx[303] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1809] /* theta[303] PARAM */)))) * (((data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */)) * ((data->simulationInfo->realParameter[807] /* omega_c[303] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11023(DATA *data, threadData_t *threadData);


/*
equation index: 4845
type: SIMPLE_ASSIGN
vy[303] = cos(theta[303]) * r_init[303] * omega_c[303]
*/
void SpiralGalaxy_eqFunction_4845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[802]] /* vy[303] STATE(1) */) = (cos((data->simulationInfo->realParameter[1809] /* theta[303] PARAM */))) * (((data->simulationInfo->realParameter[1308] /* r_init[303] PARAM */)) * ((data->simulationInfo->realParameter[807] /* omega_c[303] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11022(DATA *data, threadData_t *threadData);


/*
equation index: 4847
type: SIMPLE_ASSIGN
vz[303] = 0.0
*/
void SpiralGalaxy_eqFunction_4847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4847};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1302]] /* vz[303] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11021(DATA *data, threadData_t *threadData);


/*
equation index: 4849
type: SIMPLE_ASSIGN
z[304] = 0.00864
*/
void SpiralGalaxy_eqFunction_4849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4849};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2803]] /* z[304] STATE(1,vz[304]) */) = 0.00864;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11034(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11035(DATA *data, threadData_t *threadData);


/*
equation index: 4852
type: SIMPLE_ASSIGN
y[304] = r_init[304] * sin(theta[304] + 0.0021599999999999996)
*/
void SpiralGalaxy_eqFunction_4852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4852};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2303]] /* y[304] STATE(1,vy[304]) */) = ((data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */)) * (sin((data->simulationInfo->realParameter[1810] /* theta[304] PARAM */) + 0.0021599999999999996));
  TRACE_POP
}

/*
equation index: 4853
type: SIMPLE_ASSIGN
x[304] = r_init[304] * cos(theta[304] + 0.0021599999999999996)
*/
void SpiralGalaxy_eqFunction_4853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1803]] /* x[304] STATE(1,vx[304]) */) = ((data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */)) * (cos((data->simulationInfo->realParameter[1810] /* theta[304] PARAM */) + 0.0021599999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11036(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11037(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11040(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11039(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11038(DATA *data, threadData_t *threadData);


/*
equation index: 4859
type: SIMPLE_ASSIGN
vx[304] = (-sin(theta[304])) * r_init[304] * omega_c[304]
*/
void SpiralGalaxy_eqFunction_4859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4859};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[303]] /* vx[304] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1810] /* theta[304] PARAM */)))) * (((data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */)) * ((data->simulationInfo->realParameter[808] /* omega_c[304] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11033(DATA *data, threadData_t *threadData);


/*
equation index: 4861
type: SIMPLE_ASSIGN
vy[304] = cos(theta[304]) * r_init[304] * omega_c[304]
*/
void SpiralGalaxy_eqFunction_4861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[803]] /* vy[304] STATE(1) */) = (cos((data->simulationInfo->realParameter[1810] /* theta[304] PARAM */))) * (((data->simulationInfo->realParameter[1309] /* r_init[304] PARAM */)) * ((data->simulationInfo->realParameter[808] /* omega_c[304] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11032(DATA *data, threadData_t *threadData);


/*
equation index: 4863
type: SIMPLE_ASSIGN
vz[304] = 0.0
*/
void SpiralGalaxy_eqFunction_4863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4863};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1303]] /* vz[304] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11031(DATA *data, threadData_t *threadData);


/*
equation index: 4865
type: SIMPLE_ASSIGN
z[305] = 0.0088
*/
void SpiralGalaxy_eqFunction_4865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4865};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2804]] /* z[305] STATE(1,vz[305]) */) = 0.0088;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11044(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11045(DATA *data, threadData_t *threadData);


/*
equation index: 4868
type: SIMPLE_ASSIGN
y[305] = r_init[305] * sin(theta[305] + 0.0021999999999999997)
*/
void SpiralGalaxy_eqFunction_4868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4868};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2304]] /* y[305] STATE(1,vy[305]) */) = ((data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */)) * (sin((data->simulationInfo->realParameter[1811] /* theta[305] PARAM */) + 0.0021999999999999997));
  TRACE_POP
}

/*
equation index: 4869
type: SIMPLE_ASSIGN
x[305] = r_init[305] * cos(theta[305] + 0.0021999999999999997)
*/
void SpiralGalaxy_eqFunction_4869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1804]] /* x[305] STATE(1,vx[305]) */) = ((data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */)) * (cos((data->simulationInfo->realParameter[1811] /* theta[305] PARAM */) + 0.0021999999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11046(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11047(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11050(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11049(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11048(DATA *data, threadData_t *threadData);


/*
equation index: 4875
type: SIMPLE_ASSIGN
vx[305] = (-sin(theta[305])) * r_init[305] * omega_c[305]
*/
void SpiralGalaxy_eqFunction_4875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4875};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[304]] /* vx[305] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1811] /* theta[305] PARAM */)))) * (((data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */)) * ((data->simulationInfo->realParameter[809] /* omega_c[305] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11043(DATA *data, threadData_t *threadData);


/*
equation index: 4877
type: SIMPLE_ASSIGN
vy[305] = cos(theta[305]) * r_init[305] * omega_c[305]
*/
void SpiralGalaxy_eqFunction_4877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[804]] /* vy[305] STATE(1) */) = (cos((data->simulationInfo->realParameter[1811] /* theta[305] PARAM */))) * (((data->simulationInfo->realParameter[1310] /* r_init[305] PARAM */)) * ((data->simulationInfo->realParameter[809] /* omega_c[305] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11042(DATA *data, threadData_t *threadData);


/*
equation index: 4879
type: SIMPLE_ASSIGN
vz[305] = 0.0
*/
void SpiralGalaxy_eqFunction_4879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4879};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1304]] /* vz[305] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11041(DATA *data, threadData_t *threadData);


/*
equation index: 4881
type: SIMPLE_ASSIGN
z[306] = 0.008960000000000001
*/
void SpiralGalaxy_eqFunction_4881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4881};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2805]] /* z[306] STATE(1,vz[306]) */) = 0.008960000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11054(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11055(DATA *data, threadData_t *threadData);


/*
equation index: 4884
type: SIMPLE_ASSIGN
y[306] = r_init[306] * sin(theta[306] + 0.00224)
*/
void SpiralGalaxy_eqFunction_4884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4884};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2305]] /* y[306] STATE(1,vy[306]) */) = ((data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */)) * (sin((data->simulationInfo->realParameter[1812] /* theta[306] PARAM */) + 0.00224));
  TRACE_POP
}

/*
equation index: 4885
type: SIMPLE_ASSIGN
x[306] = r_init[306] * cos(theta[306] + 0.00224)
*/
void SpiralGalaxy_eqFunction_4885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1805]] /* x[306] STATE(1,vx[306]) */) = ((data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */)) * (cos((data->simulationInfo->realParameter[1812] /* theta[306] PARAM */) + 0.00224));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11056(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11057(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11060(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11059(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11058(DATA *data, threadData_t *threadData);


/*
equation index: 4891
type: SIMPLE_ASSIGN
vx[306] = (-sin(theta[306])) * r_init[306] * omega_c[306]
*/
void SpiralGalaxy_eqFunction_4891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4891};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[305]] /* vx[306] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1812] /* theta[306] PARAM */)))) * (((data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */)) * ((data->simulationInfo->realParameter[810] /* omega_c[306] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11053(DATA *data, threadData_t *threadData);


/*
equation index: 4893
type: SIMPLE_ASSIGN
vy[306] = cos(theta[306]) * r_init[306] * omega_c[306]
*/
void SpiralGalaxy_eqFunction_4893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[805]] /* vy[306] STATE(1) */) = (cos((data->simulationInfo->realParameter[1812] /* theta[306] PARAM */))) * (((data->simulationInfo->realParameter[1311] /* r_init[306] PARAM */)) * ((data->simulationInfo->realParameter[810] /* omega_c[306] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11052(DATA *data, threadData_t *threadData);


/*
equation index: 4895
type: SIMPLE_ASSIGN
vz[306] = 0.0
*/
void SpiralGalaxy_eqFunction_4895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4895};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1305]] /* vz[306] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11051(DATA *data, threadData_t *threadData);


/*
equation index: 4897
type: SIMPLE_ASSIGN
z[307] = 0.009120000000000001
*/
void SpiralGalaxy_eqFunction_4897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4897};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2806]] /* z[307] STATE(1,vz[307]) */) = 0.009120000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11064(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11065(DATA *data, threadData_t *threadData);


/*
equation index: 4900
type: SIMPLE_ASSIGN
y[307] = r_init[307] * sin(theta[307] + 0.00228)
*/
void SpiralGalaxy_eqFunction_4900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4900};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2306]] /* y[307] STATE(1,vy[307]) */) = ((data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */)) * (sin((data->simulationInfo->realParameter[1813] /* theta[307] PARAM */) + 0.00228));
  TRACE_POP
}

/*
equation index: 4901
type: SIMPLE_ASSIGN
x[307] = r_init[307] * cos(theta[307] + 0.00228)
*/
void SpiralGalaxy_eqFunction_4901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1806]] /* x[307] STATE(1,vx[307]) */) = ((data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */)) * (cos((data->simulationInfo->realParameter[1813] /* theta[307] PARAM */) + 0.00228));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11066(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11067(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11070(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11069(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11068(DATA *data, threadData_t *threadData);


/*
equation index: 4907
type: SIMPLE_ASSIGN
vx[307] = (-sin(theta[307])) * r_init[307] * omega_c[307]
*/
void SpiralGalaxy_eqFunction_4907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4907};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[306]] /* vx[307] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1813] /* theta[307] PARAM */)))) * (((data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */)) * ((data->simulationInfo->realParameter[811] /* omega_c[307] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11063(DATA *data, threadData_t *threadData);


/*
equation index: 4909
type: SIMPLE_ASSIGN
vy[307] = cos(theta[307]) * r_init[307] * omega_c[307]
*/
void SpiralGalaxy_eqFunction_4909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[806]] /* vy[307] STATE(1) */) = (cos((data->simulationInfo->realParameter[1813] /* theta[307] PARAM */))) * (((data->simulationInfo->realParameter[1312] /* r_init[307] PARAM */)) * ((data->simulationInfo->realParameter[811] /* omega_c[307] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11062(DATA *data, threadData_t *threadData);


/*
equation index: 4911
type: SIMPLE_ASSIGN
vz[307] = 0.0
*/
void SpiralGalaxy_eqFunction_4911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4911};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1306]] /* vz[307] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11061(DATA *data, threadData_t *threadData);


/*
equation index: 4913
type: SIMPLE_ASSIGN
z[308] = 0.00928
*/
void SpiralGalaxy_eqFunction_4913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4913};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2807]] /* z[308] STATE(1,vz[308]) */) = 0.00928;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11074(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11075(DATA *data, threadData_t *threadData);


/*
equation index: 4916
type: SIMPLE_ASSIGN
y[308] = r_init[308] * sin(theta[308] + 0.00232)
*/
void SpiralGalaxy_eqFunction_4916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4916};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2307]] /* y[308] STATE(1,vy[308]) */) = ((data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */)) * (sin((data->simulationInfo->realParameter[1814] /* theta[308] PARAM */) + 0.00232));
  TRACE_POP
}

/*
equation index: 4917
type: SIMPLE_ASSIGN
x[308] = r_init[308] * cos(theta[308] + 0.00232)
*/
void SpiralGalaxy_eqFunction_4917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1807]] /* x[308] STATE(1,vx[308]) */) = ((data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */)) * (cos((data->simulationInfo->realParameter[1814] /* theta[308] PARAM */) + 0.00232));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11076(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11077(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11080(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11079(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11078(DATA *data, threadData_t *threadData);


/*
equation index: 4923
type: SIMPLE_ASSIGN
vx[308] = (-sin(theta[308])) * r_init[308] * omega_c[308]
*/
void SpiralGalaxy_eqFunction_4923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4923};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[307]] /* vx[308] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1814] /* theta[308] PARAM */)))) * (((data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */)) * ((data->simulationInfo->realParameter[812] /* omega_c[308] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11073(DATA *data, threadData_t *threadData);


/*
equation index: 4925
type: SIMPLE_ASSIGN
vy[308] = cos(theta[308]) * r_init[308] * omega_c[308]
*/
void SpiralGalaxy_eqFunction_4925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[807]] /* vy[308] STATE(1) */) = (cos((data->simulationInfo->realParameter[1814] /* theta[308] PARAM */))) * (((data->simulationInfo->realParameter[1313] /* r_init[308] PARAM */)) * ((data->simulationInfo->realParameter[812] /* omega_c[308] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11072(DATA *data, threadData_t *threadData);


/*
equation index: 4927
type: SIMPLE_ASSIGN
vz[308] = 0.0
*/
void SpiralGalaxy_eqFunction_4927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4927};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1307]] /* vz[308] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11071(DATA *data, threadData_t *threadData);


/*
equation index: 4929
type: SIMPLE_ASSIGN
z[309] = 0.009440000000000002
*/
void SpiralGalaxy_eqFunction_4929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4929};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2808]] /* z[309] STATE(1,vz[309]) */) = 0.009440000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11084(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11085(DATA *data, threadData_t *threadData);


/*
equation index: 4932
type: SIMPLE_ASSIGN
y[309] = r_init[309] * sin(theta[309] + 0.00236)
*/
void SpiralGalaxy_eqFunction_4932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4932};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2308]] /* y[309] STATE(1,vy[309]) */) = ((data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */)) * (sin((data->simulationInfo->realParameter[1815] /* theta[309] PARAM */) + 0.00236));
  TRACE_POP
}

/*
equation index: 4933
type: SIMPLE_ASSIGN
x[309] = r_init[309] * cos(theta[309] + 0.00236)
*/
void SpiralGalaxy_eqFunction_4933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1808]] /* x[309] STATE(1,vx[309]) */) = ((data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */)) * (cos((data->simulationInfo->realParameter[1815] /* theta[309] PARAM */) + 0.00236));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11086(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11087(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11090(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11089(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11088(DATA *data, threadData_t *threadData);


/*
equation index: 4939
type: SIMPLE_ASSIGN
vx[309] = (-sin(theta[309])) * r_init[309] * omega_c[309]
*/
void SpiralGalaxy_eqFunction_4939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4939};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[308]] /* vx[309] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1815] /* theta[309] PARAM */)))) * (((data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */)) * ((data->simulationInfo->realParameter[813] /* omega_c[309] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11083(DATA *data, threadData_t *threadData);


/*
equation index: 4941
type: SIMPLE_ASSIGN
vy[309] = cos(theta[309]) * r_init[309] * omega_c[309]
*/
void SpiralGalaxy_eqFunction_4941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[808]] /* vy[309] STATE(1) */) = (cos((data->simulationInfo->realParameter[1815] /* theta[309] PARAM */))) * (((data->simulationInfo->realParameter[1314] /* r_init[309] PARAM */)) * ((data->simulationInfo->realParameter[813] /* omega_c[309] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11082(DATA *data, threadData_t *threadData);


/*
equation index: 4943
type: SIMPLE_ASSIGN
vz[309] = 0.0
*/
void SpiralGalaxy_eqFunction_4943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4943};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1308]] /* vz[309] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11081(DATA *data, threadData_t *threadData);


/*
equation index: 4945
type: SIMPLE_ASSIGN
z[310] = 0.009600000000000001
*/
void SpiralGalaxy_eqFunction_4945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4945};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2809]] /* z[310] STATE(1,vz[310]) */) = 0.009600000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11094(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11095(DATA *data, threadData_t *threadData);


/*
equation index: 4948
type: SIMPLE_ASSIGN
y[310] = r_init[310] * sin(theta[310] + 0.0024)
*/
void SpiralGalaxy_eqFunction_4948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4948};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2309]] /* y[310] STATE(1,vy[310]) */) = ((data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */)) * (sin((data->simulationInfo->realParameter[1816] /* theta[310] PARAM */) + 0.0024));
  TRACE_POP
}

/*
equation index: 4949
type: SIMPLE_ASSIGN
x[310] = r_init[310] * cos(theta[310] + 0.0024)
*/
void SpiralGalaxy_eqFunction_4949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1809]] /* x[310] STATE(1,vx[310]) */) = ((data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */)) * (cos((data->simulationInfo->realParameter[1816] /* theta[310] PARAM */) + 0.0024));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11096(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11097(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11100(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11099(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11098(DATA *data, threadData_t *threadData);


/*
equation index: 4955
type: SIMPLE_ASSIGN
vx[310] = (-sin(theta[310])) * r_init[310] * omega_c[310]
*/
void SpiralGalaxy_eqFunction_4955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4955};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[309]] /* vx[310] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1816] /* theta[310] PARAM */)))) * (((data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */)) * ((data->simulationInfo->realParameter[814] /* omega_c[310] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11093(DATA *data, threadData_t *threadData);


/*
equation index: 4957
type: SIMPLE_ASSIGN
vy[310] = cos(theta[310]) * r_init[310] * omega_c[310]
*/
void SpiralGalaxy_eqFunction_4957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[809]] /* vy[310] STATE(1) */) = (cos((data->simulationInfo->realParameter[1816] /* theta[310] PARAM */))) * (((data->simulationInfo->realParameter[1315] /* r_init[310] PARAM */)) * ((data->simulationInfo->realParameter[814] /* omega_c[310] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11092(DATA *data, threadData_t *threadData);


/*
equation index: 4959
type: SIMPLE_ASSIGN
vz[310] = 0.0
*/
void SpiralGalaxy_eqFunction_4959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4959};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1309]] /* vz[310] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11091(DATA *data, threadData_t *threadData);


/*
equation index: 4961
type: SIMPLE_ASSIGN
z[311] = 0.009760000000000001
*/
void SpiralGalaxy_eqFunction_4961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4961};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2810]] /* z[311] STATE(1,vz[311]) */) = 0.009760000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11104(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11105(DATA *data, threadData_t *threadData);


/*
equation index: 4964
type: SIMPLE_ASSIGN
y[311] = r_init[311] * sin(theta[311] + 0.00244)
*/
void SpiralGalaxy_eqFunction_4964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4964};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2310]] /* y[311] STATE(1,vy[311]) */) = ((data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */)) * (sin((data->simulationInfo->realParameter[1817] /* theta[311] PARAM */) + 0.00244));
  TRACE_POP
}

/*
equation index: 4965
type: SIMPLE_ASSIGN
x[311] = r_init[311] * cos(theta[311] + 0.00244)
*/
void SpiralGalaxy_eqFunction_4965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1810]] /* x[311] STATE(1,vx[311]) */) = ((data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */)) * (cos((data->simulationInfo->realParameter[1817] /* theta[311] PARAM */) + 0.00244));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11106(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11107(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11110(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11109(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11108(DATA *data, threadData_t *threadData);


/*
equation index: 4971
type: SIMPLE_ASSIGN
vx[311] = (-sin(theta[311])) * r_init[311] * omega_c[311]
*/
void SpiralGalaxy_eqFunction_4971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4971};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[310]] /* vx[311] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1817] /* theta[311] PARAM */)))) * (((data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */)) * ((data->simulationInfo->realParameter[815] /* omega_c[311] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11103(DATA *data, threadData_t *threadData);


/*
equation index: 4973
type: SIMPLE_ASSIGN
vy[311] = cos(theta[311]) * r_init[311] * omega_c[311]
*/
void SpiralGalaxy_eqFunction_4973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[810]] /* vy[311] STATE(1) */) = (cos((data->simulationInfo->realParameter[1817] /* theta[311] PARAM */))) * (((data->simulationInfo->realParameter[1316] /* r_init[311] PARAM */)) * ((data->simulationInfo->realParameter[815] /* omega_c[311] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11102(DATA *data, threadData_t *threadData);


/*
equation index: 4975
type: SIMPLE_ASSIGN
vz[311] = 0.0
*/
void SpiralGalaxy_eqFunction_4975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4975};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1310]] /* vz[311] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11101(DATA *data, threadData_t *threadData);


/*
equation index: 4977
type: SIMPLE_ASSIGN
z[312] = 0.009920000000000002
*/
void SpiralGalaxy_eqFunction_4977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4977};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2811]] /* z[312] STATE(1,vz[312]) */) = 0.009920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11114(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11115(DATA *data, threadData_t *threadData);


/*
equation index: 4980
type: SIMPLE_ASSIGN
y[312] = r_init[312] * sin(theta[312] + 0.00248)
*/
void SpiralGalaxy_eqFunction_4980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4980};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2311]] /* y[312] STATE(1,vy[312]) */) = ((data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */)) * (sin((data->simulationInfo->realParameter[1818] /* theta[312] PARAM */) + 0.00248));
  TRACE_POP
}

/*
equation index: 4981
type: SIMPLE_ASSIGN
x[312] = r_init[312] * cos(theta[312] + 0.00248)
*/
void SpiralGalaxy_eqFunction_4981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1811]] /* x[312] STATE(1,vx[312]) */) = ((data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */)) * (cos((data->simulationInfo->realParameter[1818] /* theta[312] PARAM */) + 0.00248));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11116(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11117(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11120(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11119(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11118(DATA *data, threadData_t *threadData);


/*
equation index: 4987
type: SIMPLE_ASSIGN
vx[312] = (-sin(theta[312])) * r_init[312] * omega_c[312]
*/
void SpiralGalaxy_eqFunction_4987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4987};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[311]] /* vx[312] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1818] /* theta[312] PARAM */)))) * (((data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */)) * ((data->simulationInfo->realParameter[816] /* omega_c[312] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11113(DATA *data, threadData_t *threadData);


/*
equation index: 4989
type: SIMPLE_ASSIGN
vy[312] = cos(theta[312]) * r_init[312] * omega_c[312]
*/
void SpiralGalaxy_eqFunction_4989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[811]] /* vy[312] STATE(1) */) = (cos((data->simulationInfo->realParameter[1818] /* theta[312] PARAM */))) * (((data->simulationInfo->realParameter[1317] /* r_init[312] PARAM */)) * ((data->simulationInfo->realParameter[816] /* omega_c[312] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11112(DATA *data, threadData_t *threadData);


/*
equation index: 4991
type: SIMPLE_ASSIGN
vz[312] = 0.0
*/
void SpiralGalaxy_eqFunction_4991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4991};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1311]] /* vz[312] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11111(DATA *data, threadData_t *threadData);


/*
equation index: 4993
type: SIMPLE_ASSIGN
z[313] = 0.01008
*/
void SpiralGalaxy_eqFunction_4993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4993};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2812]] /* z[313] STATE(1,vz[313]) */) = 0.01008;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11124(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11125(DATA *data, threadData_t *threadData);


/*
equation index: 4996
type: SIMPLE_ASSIGN
y[313] = r_init[313] * sin(theta[313] + 0.00252)
*/
void SpiralGalaxy_eqFunction_4996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4996};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2312]] /* y[313] STATE(1,vy[313]) */) = ((data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */)) * (sin((data->simulationInfo->realParameter[1819] /* theta[313] PARAM */) + 0.00252));
  TRACE_POP
}

/*
equation index: 4997
type: SIMPLE_ASSIGN
x[313] = r_init[313] * cos(theta[313] + 0.00252)
*/
void SpiralGalaxy_eqFunction_4997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1812]] /* x[313] STATE(1,vx[313]) */) = ((data->simulationInfo->realParameter[1318] /* r_init[313] PARAM */)) * (cos((data->simulationInfo->realParameter[1819] /* theta[313] PARAM */) + 0.00252));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_11126(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11127(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_11130(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_9(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_4501(data, threadData);
  SpiralGalaxy_eqFunction_10816(data, threadData);
  SpiralGalaxy_eqFunction_10817(data, threadData);
  SpiralGalaxy_eqFunction_10820(data, threadData);
  SpiralGalaxy_eqFunction_10819(data, threadData);
  SpiralGalaxy_eqFunction_10818(data, threadData);
  SpiralGalaxy_eqFunction_4507(data, threadData);
  SpiralGalaxy_eqFunction_10813(data, threadData);
  SpiralGalaxy_eqFunction_4509(data, threadData);
  SpiralGalaxy_eqFunction_10812(data, threadData);
  SpiralGalaxy_eqFunction_4511(data, threadData);
  SpiralGalaxy_eqFunction_10811(data, threadData);
  SpiralGalaxy_eqFunction_4513(data, threadData);
  SpiralGalaxy_eqFunction_10824(data, threadData);
  SpiralGalaxy_eqFunction_10825(data, threadData);
  SpiralGalaxy_eqFunction_4516(data, threadData);
  SpiralGalaxy_eqFunction_4517(data, threadData);
  SpiralGalaxy_eqFunction_10826(data, threadData);
  SpiralGalaxy_eqFunction_10827(data, threadData);
  SpiralGalaxy_eqFunction_10830(data, threadData);
  SpiralGalaxy_eqFunction_10829(data, threadData);
  SpiralGalaxy_eqFunction_10828(data, threadData);
  SpiralGalaxy_eqFunction_4523(data, threadData);
  SpiralGalaxy_eqFunction_10823(data, threadData);
  SpiralGalaxy_eqFunction_4525(data, threadData);
  SpiralGalaxy_eqFunction_10822(data, threadData);
  SpiralGalaxy_eqFunction_4527(data, threadData);
  SpiralGalaxy_eqFunction_10821(data, threadData);
  SpiralGalaxy_eqFunction_4529(data, threadData);
  SpiralGalaxy_eqFunction_10834(data, threadData);
  SpiralGalaxy_eqFunction_10835(data, threadData);
  SpiralGalaxy_eqFunction_4532(data, threadData);
  SpiralGalaxy_eqFunction_4533(data, threadData);
  SpiralGalaxy_eqFunction_10836(data, threadData);
  SpiralGalaxy_eqFunction_10837(data, threadData);
  SpiralGalaxy_eqFunction_10840(data, threadData);
  SpiralGalaxy_eqFunction_10839(data, threadData);
  SpiralGalaxy_eqFunction_10838(data, threadData);
  SpiralGalaxy_eqFunction_4539(data, threadData);
  SpiralGalaxy_eqFunction_10833(data, threadData);
  SpiralGalaxy_eqFunction_4541(data, threadData);
  SpiralGalaxy_eqFunction_10832(data, threadData);
  SpiralGalaxy_eqFunction_4543(data, threadData);
  SpiralGalaxy_eqFunction_10831(data, threadData);
  SpiralGalaxy_eqFunction_4545(data, threadData);
  SpiralGalaxy_eqFunction_10844(data, threadData);
  SpiralGalaxy_eqFunction_10845(data, threadData);
  SpiralGalaxy_eqFunction_4548(data, threadData);
  SpiralGalaxy_eqFunction_4549(data, threadData);
  SpiralGalaxy_eqFunction_10846(data, threadData);
  SpiralGalaxy_eqFunction_10847(data, threadData);
  SpiralGalaxy_eqFunction_10850(data, threadData);
  SpiralGalaxy_eqFunction_10849(data, threadData);
  SpiralGalaxy_eqFunction_10848(data, threadData);
  SpiralGalaxy_eqFunction_4555(data, threadData);
  SpiralGalaxy_eqFunction_10843(data, threadData);
  SpiralGalaxy_eqFunction_4557(data, threadData);
  SpiralGalaxy_eqFunction_10842(data, threadData);
  SpiralGalaxy_eqFunction_4559(data, threadData);
  SpiralGalaxy_eqFunction_10841(data, threadData);
  SpiralGalaxy_eqFunction_4561(data, threadData);
  SpiralGalaxy_eqFunction_10854(data, threadData);
  SpiralGalaxy_eqFunction_10855(data, threadData);
  SpiralGalaxy_eqFunction_4564(data, threadData);
  SpiralGalaxy_eqFunction_4565(data, threadData);
  SpiralGalaxy_eqFunction_10856(data, threadData);
  SpiralGalaxy_eqFunction_10857(data, threadData);
  SpiralGalaxy_eqFunction_10860(data, threadData);
  SpiralGalaxy_eqFunction_10859(data, threadData);
  SpiralGalaxy_eqFunction_10858(data, threadData);
  SpiralGalaxy_eqFunction_4571(data, threadData);
  SpiralGalaxy_eqFunction_10853(data, threadData);
  SpiralGalaxy_eqFunction_4573(data, threadData);
  SpiralGalaxy_eqFunction_10852(data, threadData);
  SpiralGalaxy_eqFunction_4575(data, threadData);
  SpiralGalaxy_eqFunction_10851(data, threadData);
  SpiralGalaxy_eqFunction_4577(data, threadData);
  SpiralGalaxy_eqFunction_10864(data, threadData);
  SpiralGalaxy_eqFunction_10865(data, threadData);
  SpiralGalaxy_eqFunction_4580(data, threadData);
  SpiralGalaxy_eqFunction_4581(data, threadData);
  SpiralGalaxy_eqFunction_10866(data, threadData);
  SpiralGalaxy_eqFunction_10867(data, threadData);
  SpiralGalaxy_eqFunction_10870(data, threadData);
  SpiralGalaxy_eqFunction_10869(data, threadData);
  SpiralGalaxy_eqFunction_10868(data, threadData);
  SpiralGalaxy_eqFunction_4587(data, threadData);
  SpiralGalaxy_eqFunction_10863(data, threadData);
  SpiralGalaxy_eqFunction_4589(data, threadData);
  SpiralGalaxy_eqFunction_10862(data, threadData);
  SpiralGalaxy_eqFunction_4591(data, threadData);
  SpiralGalaxy_eqFunction_10861(data, threadData);
  SpiralGalaxy_eqFunction_4593(data, threadData);
  SpiralGalaxy_eqFunction_10874(data, threadData);
  SpiralGalaxy_eqFunction_10875(data, threadData);
  SpiralGalaxy_eqFunction_4596(data, threadData);
  SpiralGalaxy_eqFunction_4597(data, threadData);
  SpiralGalaxy_eqFunction_10876(data, threadData);
  SpiralGalaxy_eqFunction_10877(data, threadData);
  SpiralGalaxy_eqFunction_10880(data, threadData);
  SpiralGalaxy_eqFunction_10879(data, threadData);
  SpiralGalaxy_eqFunction_10878(data, threadData);
  SpiralGalaxy_eqFunction_4603(data, threadData);
  SpiralGalaxy_eqFunction_10873(data, threadData);
  SpiralGalaxy_eqFunction_4605(data, threadData);
  SpiralGalaxy_eqFunction_10872(data, threadData);
  SpiralGalaxy_eqFunction_4607(data, threadData);
  SpiralGalaxy_eqFunction_10871(data, threadData);
  SpiralGalaxy_eqFunction_4609(data, threadData);
  SpiralGalaxy_eqFunction_10884(data, threadData);
  SpiralGalaxy_eqFunction_10885(data, threadData);
  SpiralGalaxy_eqFunction_4612(data, threadData);
  SpiralGalaxy_eqFunction_4613(data, threadData);
  SpiralGalaxy_eqFunction_10886(data, threadData);
  SpiralGalaxy_eqFunction_10887(data, threadData);
  SpiralGalaxy_eqFunction_10890(data, threadData);
  SpiralGalaxy_eqFunction_10889(data, threadData);
  SpiralGalaxy_eqFunction_10888(data, threadData);
  SpiralGalaxy_eqFunction_4619(data, threadData);
  SpiralGalaxy_eqFunction_10883(data, threadData);
  SpiralGalaxy_eqFunction_4621(data, threadData);
  SpiralGalaxy_eqFunction_10882(data, threadData);
  SpiralGalaxy_eqFunction_4623(data, threadData);
  SpiralGalaxy_eqFunction_10881(data, threadData);
  SpiralGalaxy_eqFunction_4625(data, threadData);
  SpiralGalaxy_eqFunction_10894(data, threadData);
  SpiralGalaxy_eqFunction_10895(data, threadData);
  SpiralGalaxy_eqFunction_4628(data, threadData);
  SpiralGalaxy_eqFunction_4629(data, threadData);
  SpiralGalaxy_eqFunction_10896(data, threadData);
  SpiralGalaxy_eqFunction_10897(data, threadData);
  SpiralGalaxy_eqFunction_10900(data, threadData);
  SpiralGalaxy_eqFunction_10899(data, threadData);
  SpiralGalaxy_eqFunction_10898(data, threadData);
  SpiralGalaxy_eqFunction_4635(data, threadData);
  SpiralGalaxy_eqFunction_10893(data, threadData);
  SpiralGalaxy_eqFunction_4637(data, threadData);
  SpiralGalaxy_eqFunction_10892(data, threadData);
  SpiralGalaxy_eqFunction_4639(data, threadData);
  SpiralGalaxy_eqFunction_10891(data, threadData);
  SpiralGalaxy_eqFunction_4641(data, threadData);
  SpiralGalaxy_eqFunction_10904(data, threadData);
  SpiralGalaxy_eqFunction_10905(data, threadData);
  SpiralGalaxy_eqFunction_4644(data, threadData);
  SpiralGalaxy_eqFunction_4645(data, threadData);
  SpiralGalaxy_eqFunction_10906(data, threadData);
  SpiralGalaxy_eqFunction_10907(data, threadData);
  SpiralGalaxy_eqFunction_10910(data, threadData);
  SpiralGalaxy_eqFunction_10909(data, threadData);
  SpiralGalaxy_eqFunction_10908(data, threadData);
  SpiralGalaxy_eqFunction_4651(data, threadData);
  SpiralGalaxy_eqFunction_10903(data, threadData);
  SpiralGalaxy_eqFunction_4653(data, threadData);
  SpiralGalaxy_eqFunction_10902(data, threadData);
  SpiralGalaxy_eqFunction_4655(data, threadData);
  SpiralGalaxy_eqFunction_10901(data, threadData);
  SpiralGalaxy_eqFunction_4657(data, threadData);
  SpiralGalaxy_eqFunction_10914(data, threadData);
  SpiralGalaxy_eqFunction_10915(data, threadData);
  SpiralGalaxy_eqFunction_4660(data, threadData);
  SpiralGalaxy_eqFunction_4661(data, threadData);
  SpiralGalaxy_eqFunction_10916(data, threadData);
  SpiralGalaxy_eqFunction_10917(data, threadData);
  SpiralGalaxy_eqFunction_10920(data, threadData);
  SpiralGalaxy_eqFunction_10919(data, threadData);
  SpiralGalaxy_eqFunction_10918(data, threadData);
  SpiralGalaxy_eqFunction_4667(data, threadData);
  SpiralGalaxy_eqFunction_10913(data, threadData);
  SpiralGalaxy_eqFunction_4669(data, threadData);
  SpiralGalaxy_eqFunction_10912(data, threadData);
  SpiralGalaxy_eqFunction_4671(data, threadData);
  SpiralGalaxy_eqFunction_10911(data, threadData);
  SpiralGalaxy_eqFunction_4673(data, threadData);
  SpiralGalaxy_eqFunction_10924(data, threadData);
  SpiralGalaxy_eqFunction_10925(data, threadData);
  SpiralGalaxy_eqFunction_4676(data, threadData);
  SpiralGalaxy_eqFunction_4677(data, threadData);
  SpiralGalaxy_eqFunction_10926(data, threadData);
  SpiralGalaxy_eqFunction_10927(data, threadData);
  SpiralGalaxy_eqFunction_10930(data, threadData);
  SpiralGalaxy_eqFunction_10929(data, threadData);
  SpiralGalaxy_eqFunction_10928(data, threadData);
  SpiralGalaxy_eqFunction_4683(data, threadData);
  SpiralGalaxy_eqFunction_10923(data, threadData);
  SpiralGalaxy_eqFunction_4685(data, threadData);
  SpiralGalaxy_eqFunction_10922(data, threadData);
  SpiralGalaxy_eqFunction_4687(data, threadData);
  SpiralGalaxy_eqFunction_10921(data, threadData);
  SpiralGalaxy_eqFunction_4689(data, threadData);
  SpiralGalaxy_eqFunction_10934(data, threadData);
  SpiralGalaxy_eqFunction_10935(data, threadData);
  SpiralGalaxy_eqFunction_4692(data, threadData);
  SpiralGalaxy_eqFunction_4693(data, threadData);
  SpiralGalaxy_eqFunction_10936(data, threadData);
  SpiralGalaxy_eqFunction_10937(data, threadData);
  SpiralGalaxy_eqFunction_10940(data, threadData);
  SpiralGalaxy_eqFunction_10939(data, threadData);
  SpiralGalaxy_eqFunction_10938(data, threadData);
  SpiralGalaxy_eqFunction_4699(data, threadData);
  SpiralGalaxy_eqFunction_10933(data, threadData);
  SpiralGalaxy_eqFunction_4701(data, threadData);
  SpiralGalaxy_eqFunction_10932(data, threadData);
  SpiralGalaxy_eqFunction_4703(data, threadData);
  SpiralGalaxy_eqFunction_10931(data, threadData);
  SpiralGalaxy_eqFunction_4705(data, threadData);
  SpiralGalaxy_eqFunction_10944(data, threadData);
  SpiralGalaxy_eqFunction_10945(data, threadData);
  SpiralGalaxy_eqFunction_4708(data, threadData);
  SpiralGalaxy_eqFunction_4709(data, threadData);
  SpiralGalaxy_eqFunction_10946(data, threadData);
  SpiralGalaxy_eqFunction_10947(data, threadData);
  SpiralGalaxy_eqFunction_10950(data, threadData);
  SpiralGalaxy_eqFunction_10949(data, threadData);
  SpiralGalaxy_eqFunction_10948(data, threadData);
  SpiralGalaxy_eqFunction_4715(data, threadData);
  SpiralGalaxy_eqFunction_10943(data, threadData);
  SpiralGalaxy_eqFunction_4717(data, threadData);
  SpiralGalaxy_eqFunction_10942(data, threadData);
  SpiralGalaxy_eqFunction_4719(data, threadData);
  SpiralGalaxy_eqFunction_10941(data, threadData);
  SpiralGalaxy_eqFunction_4721(data, threadData);
  SpiralGalaxy_eqFunction_10954(data, threadData);
  SpiralGalaxy_eqFunction_10955(data, threadData);
  SpiralGalaxy_eqFunction_4724(data, threadData);
  SpiralGalaxy_eqFunction_4725(data, threadData);
  SpiralGalaxy_eqFunction_10956(data, threadData);
  SpiralGalaxy_eqFunction_10957(data, threadData);
  SpiralGalaxy_eqFunction_10960(data, threadData);
  SpiralGalaxy_eqFunction_10959(data, threadData);
  SpiralGalaxy_eqFunction_10958(data, threadData);
  SpiralGalaxy_eqFunction_4731(data, threadData);
  SpiralGalaxy_eqFunction_10953(data, threadData);
  SpiralGalaxy_eqFunction_4733(data, threadData);
  SpiralGalaxy_eqFunction_10952(data, threadData);
  SpiralGalaxy_eqFunction_4735(data, threadData);
  SpiralGalaxy_eqFunction_10951(data, threadData);
  SpiralGalaxy_eqFunction_4737(data, threadData);
  SpiralGalaxy_eqFunction_10964(data, threadData);
  SpiralGalaxy_eqFunction_10965(data, threadData);
  SpiralGalaxy_eqFunction_4740(data, threadData);
  SpiralGalaxy_eqFunction_4741(data, threadData);
  SpiralGalaxy_eqFunction_10966(data, threadData);
  SpiralGalaxy_eqFunction_10967(data, threadData);
  SpiralGalaxy_eqFunction_10970(data, threadData);
  SpiralGalaxy_eqFunction_10969(data, threadData);
  SpiralGalaxy_eqFunction_10968(data, threadData);
  SpiralGalaxy_eqFunction_4747(data, threadData);
  SpiralGalaxy_eqFunction_10963(data, threadData);
  SpiralGalaxy_eqFunction_4749(data, threadData);
  SpiralGalaxy_eqFunction_10962(data, threadData);
  SpiralGalaxy_eqFunction_4751(data, threadData);
  SpiralGalaxy_eqFunction_10961(data, threadData);
  SpiralGalaxy_eqFunction_4753(data, threadData);
  SpiralGalaxy_eqFunction_10974(data, threadData);
  SpiralGalaxy_eqFunction_10975(data, threadData);
  SpiralGalaxy_eqFunction_4756(data, threadData);
  SpiralGalaxy_eqFunction_4757(data, threadData);
  SpiralGalaxy_eqFunction_10976(data, threadData);
  SpiralGalaxy_eqFunction_10977(data, threadData);
  SpiralGalaxy_eqFunction_10980(data, threadData);
  SpiralGalaxy_eqFunction_10979(data, threadData);
  SpiralGalaxy_eqFunction_10978(data, threadData);
  SpiralGalaxy_eqFunction_4763(data, threadData);
  SpiralGalaxy_eqFunction_10973(data, threadData);
  SpiralGalaxy_eqFunction_4765(data, threadData);
  SpiralGalaxy_eqFunction_10972(data, threadData);
  SpiralGalaxy_eqFunction_4767(data, threadData);
  SpiralGalaxy_eqFunction_10971(data, threadData);
  SpiralGalaxy_eqFunction_4769(data, threadData);
  SpiralGalaxy_eqFunction_10984(data, threadData);
  SpiralGalaxy_eqFunction_10985(data, threadData);
  SpiralGalaxy_eqFunction_4772(data, threadData);
  SpiralGalaxy_eqFunction_4773(data, threadData);
  SpiralGalaxy_eqFunction_10986(data, threadData);
  SpiralGalaxy_eqFunction_10987(data, threadData);
  SpiralGalaxy_eqFunction_10990(data, threadData);
  SpiralGalaxy_eqFunction_10989(data, threadData);
  SpiralGalaxy_eqFunction_10988(data, threadData);
  SpiralGalaxy_eqFunction_4779(data, threadData);
  SpiralGalaxy_eqFunction_10983(data, threadData);
  SpiralGalaxy_eqFunction_4781(data, threadData);
  SpiralGalaxy_eqFunction_10982(data, threadData);
  SpiralGalaxy_eqFunction_4783(data, threadData);
  SpiralGalaxy_eqFunction_10981(data, threadData);
  SpiralGalaxy_eqFunction_4785(data, threadData);
  SpiralGalaxy_eqFunction_10994(data, threadData);
  SpiralGalaxy_eqFunction_10995(data, threadData);
  SpiralGalaxy_eqFunction_4788(data, threadData);
  SpiralGalaxy_eqFunction_4789(data, threadData);
  SpiralGalaxy_eqFunction_10996(data, threadData);
  SpiralGalaxy_eqFunction_10997(data, threadData);
  SpiralGalaxy_eqFunction_11000(data, threadData);
  SpiralGalaxy_eqFunction_10999(data, threadData);
  SpiralGalaxy_eqFunction_10998(data, threadData);
  SpiralGalaxy_eqFunction_4795(data, threadData);
  SpiralGalaxy_eqFunction_10993(data, threadData);
  SpiralGalaxy_eqFunction_4797(data, threadData);
  SpiralGalaxy_eqFunction_10992(data, threadData);
  SpiralGalaxy_eqFunction_4799(data, threadData);
  SpiralGalaxy_eqFunction_10991(data, threadData);
  SpiralGalaxy_eqFunction_4801(data, threadData);
  SpiralGalaxy_eqFunction_11004(data, threadData);
  SpiralGalaxy_eqFunction_11005(data, threadData);
  SpiralGalaxy_eqFunction_4804(data, threadData);
  SpiralGalaxy_eqFunction_4805(data, threadData);
  SpiralGalaxy_eqFunction_11006(data, threadData);
  SpiralGalaxy_eqFunction_11007(data, threadData);
  SpiralGalaxy_eqFunction_11010(data, threadData);
  SpiralGalaxy_eqFunction_11009(data, threadData);
  SpiralGalaxy_eqFunction_11008(data, threadData);
  SpiralGalaxy_eqFunction_4811(data, threadData);
  SpiralGalaxy_eqFunction_11003(data, threadData);
  SpiralGalaxy_eqFunction_4813(data, threadData);
  SpiralGalaxy_eqFunction_11002(data, threadData);
  SpiralGalaxy_eqFunction_4815(data, threadData);
  SpiralGalaxy_eqFunction_11001(data, threadData);
  SpiralGalaxy_eqFunction_4817(data, threadData);
  SpiralGalaxy_eqFunction_11014(data, threadData);
  SpiralGalaxy_eqFunction_11015(data, threadData);
  SpiralGalaxy_eqFunction_4820(data, threadData);
  SpiralGalaxy_eqFunction_4821(data, threadData);
  SpiralGalaxy_eqFunction_11016(data, threadData);
  SpiralGalaxy_eqFunction_11017(data, threadData);
  SpiralGalaxy_eqFunction_11020(data, threadData);
  SpiralGalaxy_eqFunction_11019(data, threadData);
  SpiralGalaxy_eqFunction_11018(data, threadData);
  SpiralGalaxy_eqFunction_4827(data, threadData);
  SpiralGalaxy_eqFunction_11013(data, threadData);
  SpiralGalaxy_eqFunction_4829(data, threadData);
  SpiralGalaxy_eqFunction_11012(data, threadData);
  SpiralGalaxy_eqFunction_4831(data, threadData);
  SpiralGalaxy_eqFunction_11011(data, threadData);
  SpiralGalaxy_eqFunction_4833(data, threadData);
  SpiralGalaxy_eqFunction_11024(data, threadData);
  SpiralGalaxy_eqFunction_11025(data, threadData);
  SpiralGalaxy_eqFunction_4836(data, threadData);
  SpiralGalaxy_eqFunction_4837(data, threadData);
  SpiralGalaxy_eqFunction_11026(data, threadData);
  SpiralGalaxy_eqFunction_11027(data, threadData);
  SpiralGalaxy_eqFunction_11030(data, threadData);
  SpiralGalaxy_eqFunction_11029(data, threadData);
  SpiralGalaxy_eqFunction_11028(data, threadData);
  SpiralGalaxy_eqFunction_4843(data, threadData);
  SpiralGalaxy_eqFunction_11023(data, threadData);
  SpiralGalaxy_eqFunction_4845(data, threadData);
  SpiralGalaxy_eqFunction_11022(data, threadData);
  SpiralGalaxy_eqFunction_4847(data, threadData);
  SpiralGalaxy_eqFunction_11021(data, threadData);
  SpiralGalaxy_eqFunction_4849(data, threadData);
  SpiralGalaxy_eqFunction_11034(data, threadData);
  SpiralGalaxy_eqFunction_11035(data, threadData);
  SpiralGalaxy_eqFunction_4852(data, threadData);
  SpiralGalaxy_eqFunction_4853(data, threadData);
  SpiralGalaxy_eqFunction_11036(data, threadData);
  SpiralGalaxy_eqFunction_11037(data, threadData);
  SpiralGalaxy_eqFunction_11040(data, threadData);
  SpiralGalaxy_eqFunction_11039(data, threadData);
  SpiralGalaxy_eqFunction_11038(data, threadData);
  SpiralGalaxy_eqFunction_4859(data, threadData);
  SpiralGalaxy_eqFunction_11033(data, threadData);
  SpiralGalaxy_eqFunction_4861(data, threadData);
  SpiralGalaxy_eqFunction_11032(data, threadData);
  SpiralGalaxy_eqFunction_4863(data, threadData);
  SpiralGalaxy_eqFunction_11031(data, threadData);
  SpiralGalaxy_eqFunction_4865(data, threadData);
  SpiralGalaxy_eqFunction_11044(data, threadData);
  SpiralGalaxy_eqFunction_11045(data, threadData);
  SpiralGalaxy_eqFunction_4868(data, threadData);
  SpiralGalaxy_eqFunction_4869(data, threadData);
  SpiralGalaxy_eqFunction_11046(data, threadData);
  SpiralGalaxy_eqFunction_11047(data, threadData);
  SpiralGalaxy_eqFunction_11050(data, threadData);
  SpiralGalaxy_eqFunction_11049(data, threadData);
  SpiralGalaxy_eqFunction_11048(data, threadData);
  SpiralGalaxy_eqFunction_4875(data, threadData);
  SpiralGalaxy_eqFunction_11043(data, threadData);
  SpiralGalaxy_eqFunction_4877(data, threadData);
  SpiralGalaxy_eqFunction_11042(data, threadData);
  SpiralGalaxy_eqFunction_4879(data, threadData);
  SpiralGalaxy_eqFunction_11041(data, threadData);
  SpiralGalaxy_eqFunction_4881(data, threadData);
  SpiralGalaxy_eqFunction_11054(data, threadData);
  SpiralGalaxy_eqFunction_11055(data, threadData);
  SpiralGalaxy_eqFunction_4884(data, threadData);
  SpiralGalaxy_eqFunction_4885(data, threadData);
  SpiralGalaxy_eqFunction_11056(data, threadData);
  SpiralGalaxy_eqFunction_11057(data, threadData);
  SpiralGalaxy_eqFunction_11060(data, threadData);
  SpiralGalaxy_eqFunction_11059(data, threadData);
  SpiralGalaxy_eqFunction_11058(data, threadData);
  SpiralGalaxy_eqFunction_4891(data, threadData);
  SpiralGalaxy_eqFunction_11053(data, threadData);
  SpiralGalaxy_eqFunction_4893(data, threadData);
  SpiralGalaxy_eqFunction_11052(data, threadData);
  SpiralGalaxy_eqFunction_4895(data, threadData);
  SpiralGalaxy_eqFunction_11051(data, threadData);
  SpiralGalaxy_eqFunction_4897(data, threadData);
  SpiralGalaxy_eqFunction_11064(data, threadData);
  SpiralGalaxy_eqFunction_11065(data, threadData);
  SpiralGalaxy_eqFunction_4900(data, threadData);
  SpiralGalaxy_eqFunction_4901(data, threadData);
  SpiralGalaxy_eqFunction_11066(data, threadData);
  SpiralGalaxy_eqFunction_11067(data, threadData);
  SpiralGalaxy_eqFunction_11070(data, threadData);
  SpiralGalaxy_eqFunction_11069(data, threadData);
  SpiralGalaxy_eqFunction_11068(data, threadData);
  SpiralGalaxy_eqFunction_4907(data, threadData);
  SpiralGalaxy_eqFunction_11063(data, threadData);
  SpiralGalaxy_eqFunction_4909(data, threadData);
  SpiralGalaxy_eqFunction_11062(data, threadData);
  SpiralGalaxy_eqFunction_4911(data, threadData);
  SpiralGalaxy_eqFunction_11061(data, threadData);
  SpiralGalaxy_eqFunction_4913(data, threadData);
  SpiralGalaxy_eqFunction_11074(data, threadData);
  SpiralGalaxy_eqFunction_11075(data, threadData);
  SpiralGalaxy_eqFunction_4916(data, threadData);
  SpiralGalaxy_eqFunction_4917(data, threadData);
  SpiralGalaxy_eqFunction_11076(data, threadData);
  SpiralGalaxy_eqFunction_11077(data, threadData);
  SpiralGalaxy_eqFunction_11080(data, threadData);
  SpiralGalaxy_eqFunction_11079(data, threadData);
  SpiralGalaxy_eqFunction_11078(data, threadData);
  SpiralGalaxy_eqFunction_4923(data, threadData);
  SpiralGalaxy_eqFunction_11073(data, threadData);
  SpiralGalaxy_eqFunction_4925(data, threadData);
  SpiralGalaxy_eqFunction_11072(data, threadData);
  SpiralGalaxy_eqFunction_4927(data, threadData);
  SpiralGalaxy_eqFunction_11071(data, threadData);
  SpiralGalaxy_eqFunction_4929(data, threadData);
  SpiralGalaxy_eqFunction_11084(data, threadData);
  SpiralGalaxy_eqFunction_11085(data, threadData);
  SpiralGalaxy_eqFunction_4932(data, threadData);
  SpiralGalaxy_eqFunction_4933(data, threadData);
  SpiralGalaxy_eqFunction_11086(data, threadData);
  SpiralGalaxy_eqFunction_11087(data, threadData);
  SpiralGalaxy_eqFunction_11090(data, threadData);
  SpiralGalaxy_eqFunction_11089(data, threadData);
  SpiralGalaxy_eqFunction_11088(data, threadData);
  SpiralGalaxy_eqFunction_4939(data, threadData);
  SpiralGalaxy_eqFunction_11083(data, threadData);
  SpiralGalaxy_eqFunction_4941(data, threadData);
  SpiralGalaxy_eqFunction_11082(data, threadData);
  SpiralGalaxy_eqFunction_4943(data, threadData);
  SpiralGalaxy_eqFunction_11081(data, threadData);
  SpiralGalaxy_eqFunction_4945(data, threadData);
  SpiralGalaxy_eqFunction_11094(data, threadData);
  SpiralGalaxy_eqFunction_11095(data, threadData);
  SpiralGalaxy_eqFunction_4948(data, threadData);
  SpiralGalaxy_eqFunction_4949(data, threadData);
  SpiralGalaxy_eqFunction_11096(data, threadData);
  SpiralGalaxy_eqFunction_11097(data, threadData);
  SpiralGalaxy_eqFunction_11100(data, threadData);
  SpiralGalaxy_eqFunction_11099(data, threadData);
  SpiralGalaxy_eqFunction_11098(data, threadData);
  SpiralGalaxy_eqFunction_4955(data, threadData);
  SpiralGalaxy_eqFunction_11093(data, threadData);
  SpiralGalaxy_eqFunction_4957(data, threadData);
  SpiralGalaxy_eqFunction_11092(data, threadData);
  SpiralGalaxy_eqFunction_4959(data, threadData);
  SpiralGalaxy_eqFunction_11091(data, threadData);
  SpiralGalaxy_eqFunction_4961(data, threadData);
  SpiralGalaxy_eqFunction_11104(data, threadData);
  SpiralGalaxy_eqFunction_11105(data, threadData);
  SpiralGalaxy_eqFunction_4964(data, threadData);
  SpiralGalaxy_eqFunction_4965(data, threadData);
  SpiralGalaxy_eqFunction_11106(data, threadData);
  SpiralGalaxy_eqFunction_11107(data, threadData);
  SpiralGalaxy_eqFunction_11110(data, threadData);
  SpiralGalaxy_eqFunction_11109(data, threadData);
  SpiralGalaxy_eqFunction_11108(data, threadData);
  SpiralGalaxy_eqFunction_4971(data, threadData);
  SpiralGalaxy_eqFunction_11103(data, threadData);
  SpiralGalaxy_eqFunction_4973(data, threadData);
  SpiralGalaxy_eqFunction_11102(data, threadData);
  SpiralGalaxy_eqFunction_4975(data, threadData);
  SpiralGalaxy_eqFunction_11101(data, threadData);
  SpiralGalaxy_eqFunction_4977(data, threadData);
  SpiralGalaxy_eqFunction_11114(data, threadData);
  SpiralGalaxy_eqFunction_11115(data, threadData);
  SpiralGalaxy_eqFunction_4980(data, threadData);
  SpiralGalaxy_eqFunction_4981(data, threadData);
  SpiralGalaxy_eqFunction_11116(data, threadData);
  SpiralGalaxy_eqFunction_11117(data, threadData);
  SpiralGalaxy_eqFunction_11120(data, threadData);
  SpiralGalaxy_eqFunction_11119(data, threadData);
  SpiralGalaxy_eqFunction_11118(data, threadData);
  SpiralGalaxy_eqFunction_4987(data, threadData);
  SpiralGalaxy_eqFunction_11113(data, threadData);
  SpiralGalaxy_eqFunction_4989(data, threadData);
  SpiralGalaxy_eqFunction_11112(data, threadData);
  SpiralGalaxy_eqFunction_4991(data, threadData);
  SpiralGalaxy_eqFunction_11111(data, threadData);
  SpiralGalaxy_eqFunction_4993(data, threadData);
  SpiralGalaxy_eqFunction_11124(data, threadData);
  SpiralGalaxy_eqFunction_11125(data, threadData);
  SpiralGalaxy_eqFunction_4996(data, threadData);
  SpiralGalaxy_eqFunction_4997(data, threadData);
  SpiralGalaxy_eqFunction_11126(data, threadData);
  SpiralGalaxy_eqFunction_11127(data, threadData);
  SpiralGalaxy_eqFunction_11130(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif