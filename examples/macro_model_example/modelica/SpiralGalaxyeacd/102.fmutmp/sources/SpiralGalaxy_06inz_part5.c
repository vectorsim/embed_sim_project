#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 2501
type: SIMPLE_ASSIGN
x[157] = r_init[157] * cos(theta[157] - 0.00372)
*/
void SpiralGalaxy_eqFunction_2501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1656]] /* x[157] STATE(1,vx[157]) */) = ((data->simulationInfo->realParameter[1162] /* r_init[157] PARAM */)) * (cos((data->simulationInfo->realParameter[1663] /* theta[157] PARAM */) - 0.00372));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9566(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9567(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9570(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9569(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9568(DATA *data, threadData_t *threadData);


/*
equation index: 2507
type: SIMPLE_ASSIGN
vx[157] = (-sin(theta[157])) * r_init[157] * omega_c[157]
*/
void SpiralGalaxy_eqFunction_2507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2507};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[156]] /* vx[157] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1663] /* theta[157] PARAM */)))) * (((data->simulationInfo->realParameter[1162] /* r_init[157] PARAM */)) * ((data->simulationInfo->realParameter[661] /* omega_c[157] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9563(DATA *data, threadData_t *threadData);


/*
equation index: 2509
type: SIMPLE_ASSIGN
vy[157] = cos(theta[157]) * r_init[157] * omega_c[157]
*/
void SpiralGalaxy_eqFunction_2509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[656]] /* vy[157] STATE(1) */) = (cos((data->simulationInfo->realParameter[1663] /* theta[157] PARAM */))) * (((data->simulationInfo->realParameter[1162] /* r_init[157] PARAM */)) * ((data->simulationInfo->realParameter[661] /* omega_c[157] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9562(DATA *data, threadData_t *threadData);


/*
equation index: 2511
type: SIMPLE_ASSIGN
vz[157] = 0.0
*/
void SpiralGalaxy_eqFunction_2511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2511};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1156]] /* vz[157] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9561(DATA *data, threadData_t *threadData);


/*
equation index: 2513
type: SIMPLE_ASSIGN
z[158] = -0.01472
*/
void SpiralGalaxy_eqFunction_2513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2513};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2657]] /* z[158] STATE(1,vz[158]) */) = -0.01472;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9574(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9575(DATA *data, threadData_t *threadData);


/*
equation index: 2516
type: SIMPLE_ASSIGN
y[158] = r_init[158] * sin(theta[158] - 0.00368)
*/
void SpiralGalaxy_eqFunction_2516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2516};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2157]] /* y[158] STATE(1,vy[158]) */) = ((data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */)) * (sin((data->simulationInfo->realParameter[1664] /* theta[158] PARAM */) - 0.00368));
  TRACE_POP
}

/*
equation index: 2517
type: SIMPLE_ASSIGN
x[158] = r_init[158] * cos(theta[158] - 0.00368)
*/
void SpiralGalaxy_eqFunction_2517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1657]] /* x[158] STATE(1,vx[158]) */) = ((data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */)) * (cos((data->simulationInfo->realParameter[1664] /* theta[158] PARAM */) - 0.00368));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9576(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9577(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9580(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9579(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9578(DATA *data, threadData_t *threadData);


/*
equation index: 2523
type: SIMPLE_ASSIGN
vx[158] = (-sin(theta[158])) * r_init[158] * omega_c[158]
*/
void SpiralGalaxy_eqFunction_2523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2523};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[157]] /* vx[158] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1664] /* theta[158] PARAM */)))) * (((data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */)) * ((data->simulationInfo->realParameter[662] /* omega_c[158] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9573(DATA *data, threadData_t *threadData);


/*
equation index: 2525
type: SIMPLE_ASSIGN
vy[158] = cos(theta[158]) * r_init[158] * omega_c[158]
*/
void SpiralGalaxy_eqFunction_2525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[657]] /* vy[158] STATE(1) */) = (cos((data->simulationInfo->realParameter[1664] /* theta[158] PARAM */))) * (((data->simulationInfo->realParameter[1163] /* r_init[158] PARAM */)) * ((data->simulationInfo->realParameter[662] /* omega_c[158] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9572(DATA *data, threadData_t *threadData);


/*
equation index: 2527
type: SIMPLE_ASSIGN
vz[158] = 0.0
*/
void SpiralGalaxy_eqFunction_2527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1157]] /* vz[158] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9571(DATA *data, threadData_t *threadData);


/*
equation index: 2529
type: SIMPLE_ASSIGN
z[159] = -0.014560000000000002
*/
void SpiralGalaxy_eqFunction_2529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2529};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2658]] /* z[159] STATE(1,vz[159]) */) = -0.014560000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9584(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9585(DATA *data, threadData_t *threadData);


/*
equation index: 2532
type: SIMPLE_ASSIGN
y[159] = r_init[159] * sin(theta[159] - 0.00364)
*/
void SpiralGalaxy_eqFunction_2532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2532};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2158]] /* y[159] STATE(1,vy[159]) */) = ((data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */)) * (sin((data->simulationInfo->realParameter[1665] /* theta[159] PARAM */) - 0.00364));
  TRACE_POP
}

/*
equation index: 2533
type: SIMPLE_ASSIGN
x[159] = r_init[159] * cos(theta[159] - 0.00364)
*/
void SpiralGalaxy_eqFunction_2533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1658]] /* x[159] STATE(1,vx[159]) */) = ((data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */)) * (cos((data->simulationInfo->realParameter[1665] /* theta[159] PARAM */) - 0.00364));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9586(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9587(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9590(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9589(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9588(DATA *data, threadData_t *threadData);


/*
equation index: 2539
type: SIMPLE_ASSIGN
vx[159] = (-sin(theta[159])) * r_init[159] * omega_c[159]
*/
void SpiralGalaxy_eqFunction_2539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2539};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[158]] /* vx[159] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1665] /* theta[159] PARAM */)))) * (((data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */)) * ((data->simulationInfo->realParameter[663] /* omega_c[159] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9583(DATA *data, threadData_t *threadData);


/*
equation index: 2541
type: SIMPLE_ASSIGN
vy[159] = cos(theta[159]) * r_init[159] * omega_c[159]
*/
void SpiralGalaxy_eqFunction_2541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[658]] /* vy[159] STATE(1) */) = (cos((data->simulationInfo->realParameter[1665] /* theta[159] PARAM */))) * (((data->simulationInfo->realParameter[1164] /* r_init[159] PARAM */)) * ((data->simulationInfo->realParameter[663] /* omega_c[159] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9582(DATA *data, threadData_t *threadData);


/*
equation index: 2543
type: SIMPLE_ASSIGN
vz[159] = 0.0
*/
void SpiralGalaxy_eqFunction_2543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2543};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1158]] /* vz[159] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9581(DATA *data, threadData_t *threadData);


/*
equation index: 2545
type: SIMPLE_ASSIGN
z[160] = -0.014400000000000001
*/
void SpiralGalaxy_eqFunction_2545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2545};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2659]] /* z[160] STATE(1,vz[160]) */) = -0.014400000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9594(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9595(DATA *data, threadData_t *threadData);


/*
equation index: 2548
type: SIMPLE_ASSIGN
y[160] = r_init[160] * sin(theta[160] - 0.0036)
*/
void SpiralGalaxy_eqFunction_2548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2548};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2159]] /* y[160] STATE(1,vy[160]) */) = ((data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */)) * (sin((data->simulationInfo->realParameter[1666] /* theta[160] PARAM */) - 0.0036));
  TRACE_POP
}

/*
equation index: 2549
type: SIMPLE_ASSIGN
x[160] = r_init[160] * cos(theta[160] - 0.0036)
*/
void SpiralGalaxy_eqFunction_2549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1659]] /* x[160] STATE(1,vx[160]) */) = ((data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */)) * (cos((data->simulationInfo->realParameter[1666] /* theta[160] PARAM */) - 0.0036));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9596(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9597(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9600(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9599(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9598(DATA *data, threadData_t *threadData);


/*
equation index: 2555
type: SIMPLE_ASSIGN
vx[160] = (-sin(theta[160])) * r_init[160] * omega_c[160]
*/
void SpiralGalaxy_eqFunction_2555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2555};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[159]] /* vx[160] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1666] /* theta[160] PARAM */)))) * (((data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */)) * ((data->simulationInfo->realParameter[664] /* omega_c[160] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9593(DATA *data, threadData_t *threadData);


/*
equation index: 2557
type: SIMPLE_ASSIGN
vy[160] = cos(theta[160]) * r_init[160] * omega_c[160]
*/
void SpiralGalaxy_eqFunction_2557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[659]] /* vy[160] STATE(1) */) = (cos((data->simulationInfo->realParameter[1666] /* theta[160] PARAM */))) * (((data->simulationInfo->realParameter[1165] /* r_init[160] PARAM */)) * ((data->simulationInfo->realParameter[664] /* omega_c[160] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9592(DATA *data, threadData_t *threadData);


/*
equation index: 2559
type: SIMPLE_ASSIGN
vz[160] = 0.0
*/
void SpiralGalaxy_eqFunction_2559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2559};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1159]] /* vz[160] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9591(DATA *data, threadData_t *threadData);


/*
equation index: 2561
type: SIMPLE_ASSIGN
z[161] = -0.014240000000000001
*/
void SpiralGalaxy_eqFunction_2561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2561};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2660]] /* z[161] STATE(1,vz[161]) */) = -0.014240000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9604(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9605(DATA *data, threadData_t *threadData);


/*
equation index: 2564
type: SIMPLE_ASSIGN
y[161] = r_init[161] * sin(theta[161] - 0.00356)
*/
void SpiralGalaxy_eqFunction_2564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2564};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2160]] /* y[161] STATE(1,vy[161]) */) = ((data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */)) * (sin((data->simulationInfo->realParameter[1667] /* theta[161] PARAM */) - 0.00356));
  TRACE_POP
}

/*
equation index: 2565
type: SIMPLE_ASSIGN
x[161] = r_init[161] * cos(theta[161] - 0.00356)
*/
void SpiralGalaxy_eqFunction_2565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1660]] /* x[161] STATE(1,vx[161]) */) = ((data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */)) * (cos((data->simulationInfo->realParameter[1667] /* theta[161] PARAM */) - 0.00356));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9606(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9607(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9610(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9609(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9608(DATA *data, threadData_t *threadData);


/*
equation index: 2571
type: SIMPLE_ASSIGN
vx[161] = (-sin(theta[161])) * r_init[161] * omega_c[161]
*/
void SpiralGalaxy_eqFunction_2571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2571};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[160]] /* vx[161] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1667] /* theta[161] PARAM */)))) * (((data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */)) * ((data->simulationInfo->realParameter[665] /* omega_c[161] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9603(DATA *data, threadData_t *threadData);


/*
equation index: 2573
type: SIMPLE_ASSIGN
vy[161] = cos(theta[161]) * r_init[161] * omega_c[161]
*/
void SpiralGalaxy_eqFunction_2573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[660]] /* vy[161] STATE(1) */) = (cos((data->simulationInfo->realParameter[1667] /* theta[161] PARAM */))) * (((data->simulationInfo->realParameter[1166] /* r_init[161] PARAM */)) * ((data->simulationInfo->realParameter[665] /* omega_c[161] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9602(DATA *data, threadData_t *threadData);


/*
equation index: 2575
type: SIMPLE_ASSIGN
vz[161] = 0.0
*/
void SpiralGalaxy_eqFunction_2575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2575};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1160]] /* vz[161] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9601(DATA *data, threadData_t *threadData);


/*
equation index: 2577
type: SIMPLE_ASSIGN
z[162] = -0.01408
*/
void SpiralGalaxy_eqFunction_2577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2577};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2661]] /* z[162] STATE(1,vz[162]) */) = -0.01408;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9614(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9615(DATA *data, threadData_t *threadData);


/*
equation index: 2580
type: SIMPLE_ASSIGN
y[162] = r_init[162] * sin(theta[162] - 0.0035199999999999997)
*/
void SpiralGalaxy_eqFunction_2580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2580};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2161]] /* y[162] STATE(1,vy[162]) */) = ((data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */)) * (sin((data->simulationInfo->realParameter[1668] /* theta[162] PARAM */) - 0.0035199999999999997));
  TRACE_POP
}

/*
equation index: 2581
type: SIMPLE_ASSIGN
x[162] = r_init[162] * cos(theta[162] - 0.0035199999999999997)
*/
void SpiralGalaxy_eqFunction_2581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1661]] /* x[162] STATE(1,vx[162]) */) = ((data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */)) * (cos((data->simulationInfo->realParameter[1668] /* theta[162] PARAM */) - 0.0035199999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9616(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9617(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9620(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9619(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9618(DATA *data, threadData_t *threadData);


/*
equation index: 2587
type: SIMPLE_ASSIGN
vx[162] = (-sin(theta[162])) * r_init[162] * omega_c[162]
*/
void SpiralGalaxy_eqFunction_2587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2587};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[161]] /* vx[162] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1668] /* theta[162] PARAM */)))) * (((data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */)) * ((data->simulationInfo->realParameter[666] /* omega_c[162] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9613(DATA *data, threadData_t *threadData);


/*
equation index: 2589
type: SIMPLE_ASSIGN
vy[162] = cos(theta[162]) * r_init[162] * omega_c[162]
*/
void SpiralGalaxy_eqFunction_2589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[661]] /* vy[162] STATE(1) */) = (cos((data->simulationInfo->realParameter[1668] /* theta[162] PARAM */))) * (((data->simulationInfo->realParameter[1167] /* r_init[162] PARAM */)) * ((data->simulationInfo->realParameter[666] /* omega_c[162] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9612(DATA *data, threadData_t *threadData);


/*
equation index: 2591
type: SIMPLE_ASSIGN
vz[162] = 0.0
*/
void SpiralGalaxy_eqFunction_2591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2591};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1161]] /* vz[162] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9611(DATA *data, threadData_t *threadData);


/*
equation index: 2593
type: SIMPLE_ASSIGN
z[163] = -0.013920000000000002
*/
void SpiralGalaxy_eqFunction_2593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2593};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2662]] /* z[163] STATE(1,vz[163]) */) = -0.013920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9624(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9625(DATA *data, threadData_t *threadData);


/*
equation index: 2596
type: SIMPLE_ASSIGN
y[163] = r_init[163] * sin(theta[163] - 0.00348)
*/
void SpiralGalaxy_eqFunction_2596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2596};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2162]] /* y[163] STATE(1,vy[163]) */) = ((data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */)) * (sin((data->simulationInfo->realParameter[1669] /* theta[163] PARAM */) - 0.00348));
  TRACE_POP
}

/*
equation index: 2597
type: SIMPLE_ASSIGN
x[163] = r_init[163] * cos(theta[163] - 0.00348)
*/
void SpiralGalaxy_eqFunction_2597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1662]] /* x[163] STATE(1,vx[163]) */) = ((data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */)) * (cos((data->simulationInfo->realParameter[1669] /* theta[163] PARAM */) - 0.00348));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9626(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9627(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9630(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9629(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9628(DATA *data, threadData_t *threadData);


/*
equation index: 2603
type: SIMPLE_ASSIGN
vx[163] = (-sin(theta[163])) * r_init[163] * omega_c[163]
*/
void SpiralGalaxy_eqFunction_2603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2603};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[162]] /* vx[163] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1669] /* theta[163] PARAM */)))) * (((data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */)) * ((data->simulationInfo->realParameter[667] /* omega_c[163] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9623(DATA *data, threadData_t *threadData);


/*
equation index: 2605
type: SIMPLE_ASSIGN
vy[163] = cos(theta[163]) * r_init[163] * omega_c[163]
*/
void SpiralGalaxy_eqFunction_2605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[662]] /* vy[163] STATE(1) */) = (cos((data->simulationInfo->realParameter[1669] /* theta[163] PARAM */))) * (((data->simulationInfo->realParameter[1168] /* r_init[163] PARAM */)) * ((data->simulationInfo->realParameter[667] /* omega_c[163] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9622(DATA *data, threadData_t *threadData);


/*
equation index: 2607
type: SIMPLE_ASSIGN
vz[163] = 0.0
*/
void SpiralGalaxy_eqFunction_2607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2607};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1162]] /* vz[163] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9621(DATA *data, threadData_t *threadData);


/*
equation index: 2609
type: SIMPLE_ASSIGN
z[164] = -0.01376
*/
void SpiralGalaxy_eqFunction_2609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2609};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2663]] /* z[164] STATE(1,vz[164]) */) = -0.01376;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9634(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9635(DATA *data, threadData_t *threadData);


/*
equation index: 2612
type: SIMPLE_ASSIGN
y[164] = r_init[164] * sin(theta[164] - 0.00344)
*/
void SpiralGalaxy_eqFunction_2612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2612};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2163]] /* y[164] STATE(1,vy[164]) */) = ((data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */)) * (sin((data->simulationInfo->realParameter[1670] /* theta[164] PARAM */) - 0.00344));
  TRACE_POP
}

/*
equation index: 2613
type: SIMPLE_ASSIGN
x[164] = r_init[164] * cos(theta[164] - 0.00344)
*/
void SpiralGalaxy_eqFunction_2613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1663]] /* x[164] STATE(1,vx[164]) */) = ((data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */)) * (cos((data->simulationInfo->realParameter[1670] /* theta[164] PARAM */) - 0.00344));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9636(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9637(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9640(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9639(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9638(DATA *data, threadData_t *threadData);


/*
equation index: 2619
type: SIMPLE_ASSIGN
vx[164] = (-sin(theta[164])) * r_init[164] * omega_c[164]
*/
void SpiralGalaxy_eqFunction_2619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2619};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[163]] /* vx[164] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1670] /* theta[164] PARAM */)))) * (((data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */)) * ((data->simulationInfo->realParameter[668] /* omega_c[164] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9633(DATA *data, threadData_t *threadData);


/*
equation index: 2621
type: SIMPLE_ASSIGN
vy[164] = cos(theta[164]) * r_init[164] * omega_c[164]
*/
void SpiralGalaxy_eqFunction_2621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[663]] /* vy[164] STATE(1) */) = (cos((data->simulationInfo->realParameter[1670] /* theta[164] PARAM */))) * (((data->simulationInfo->realParameter[1169] /* r_init[164] PARAM */)) * ((data->simulationInfo->realParameter[668] /* omega_c[164] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9632(DATA *data, threadData_t *threadData);


/*
equation index: 2623
type: SIMPLE_ASSIGN
vz[164] = 0.0
*/
void SpiralGalaxy_eqFunction_2623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2623};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1163]] /* vz[164] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9631(DATA *data, threadData_t *threadData);


/*
equation index: 2625
type: SIMPLE_ASSIGN
z[165] = -0.013600000000000001
*/
void SpiralGalaxy_eqFunction_2625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2625};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2664]] /* z[165] STATE(1,vz[165]) */) = -0.013600000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9644(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9645(DATA *data, threadData_t *threadData);


/*
equation index: 2628
type: SIMPLE_ASSIGN
y[165] = r_init[165] * sin(theta[165] - 0.0034)
*/
void SpiralGalaxy_eqFunction_2628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2628};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2164]] /* y[165] STATE(1,vy[165]) */) = ((data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */)) * (sin((data->simulationInfo->realParameter[1671] /* theta[165] PARAM */) - 0.0034));
  TRACE_POP
}

/*
equation index: 2629
type: SIMPLE_ASSIGN
x[165] = r_init[165] * cos(theta[165] - 0.0034)
*/
void SpiralGalaxy_eqFunction_2629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1664]] /* x[165] STATE(1,vx[165]) */) = ((data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */)) * (cos((data->simulationInfo->realParameter[1671] /* theta[165] PARAM */) - 0.0034));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9646(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9647(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9650(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9649(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9648(DATA *data, threadData_t *threadData);


/*
equation index: 2635
type: SIMPLE_ASSIGN
vx[165] = (-sin(theta[165])) * r_init[165] * omega_c[165]
*/
void SpiralGalaxy_eqFunction_2635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2635};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[164]] /* vx[165] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1671] /* theta[165] PARAM */)))) * (((data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */)) * ((data->simulationInfo->realParameter[669] /* omega_c[165] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9643(DATA *data, threadData_t *threadData);


/*
equation index: 2637
type: SIMPLE_ASSIGN
vy[165] = cos(theta[165]) * r_init[165] * omega_c[165]
*/
void SpiralGalaxy_eqFunction_2637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[664]] /* vy[165] STATE(1) */) = (cos((data->simulationInfo->realParameter[1671] /* theta[165] PARAM */))) * (((data->simulationInfo->realParameter[1170] /* r_init[165] PARAM */)) * ((data->simulationInfo->realParameter[669] /* omega_c[165] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9642(DATA *data, threadData_t *threadData);


/*
equation index: 2639
type: SIMPLE_ASSIGN
vz[165] = 0.0
*/
void SpiralGalaxy_eqFunction_2639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2639};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1164]] /* vz[165] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9641(DATA *data, threadData_t *threadData);


/*
equation index: 2641
type: SIMPLE_ASSIGN
z[166] = -0.01344
*/
void SpiralGalaxy_eqFunction_2641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2641};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2665]] /* z[166] STATE(1,vz[166]) */) = -0.01344;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9654(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9655(DATA *data, threadData_t *threadData);


/*
equation index: 2644
type: SIMPLE_ASSIGN
y[166] = r_init[166] * sin(theta[166] - 0.0033599999999999997)
*/
void SpiralGalaxy_eqFunction_2644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2644};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2165]] /* y[166] STATE(1,vy[166]) */) = ((data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */)) * (sin((data->simulationInfo->realParameter[1672] /* theta[166] PARAM */) - 0.0033599999999999997));
  TRACE_POP
}

/*
equation index: 2645
type: SIMPLE_ASSIGN
x[166] = r_init[166] * cos(theta[166] - 0.0033599999999999997)
*/
void SpiralGalaxy_eqFunction_2645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1665]] /* x[166] STATE(1,vx[166]) */) = ((data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */)) * (cos((data->simulationInfo->realParameter[1672] /* theta[166] PARAM */) - 0.0033599999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9656(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9657(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9660(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9659(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9658(DATA *data, threadData_t *threadData);


/*
equation index: 2651
type: SIMPLE_ASSIGN
vx[166] = (-sin(theta[166])) * r_init[166] * omega_c[166]
*/
void SpiralGalaxy_eqFunction_2651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2651};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[165]] /* vx[166] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1672] /* theta[166] PARAM */)))) * (((data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */)) * ((data->simulationInfo->realParameter[670] /* omega_c[166] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9653(DATA *data, threadData_t *threadData);


/*
equation index: 2653
type: SIMPLE_ASSIGN
vy[166] = cos(theta[166]) * r_init[166] * omega_c[166]
*/
void SpiralGalaxy_eqFunction_2653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[665]] /* vy[166] STATE(1) */) = (cos((data->simulationInfo->realParameter[1672] /* theta[166] PARAM */))) * (((data->simulationInfo->realParameter[1171] /* r_init[166] PARAM */)) * ((data->simulationInfo->realParameter[670] /* omega_c[166] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9652(DATA *data, threadData_t *threadData);


/*
equation index: 2655
type: SIMPLE_ASSIGN
vz[166] = 0.0
*/
void SpiralGalaxy_eqFunction_2655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2655};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1165]] /* vz[166] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9651(DATA *data, threadData_t *threadData);


/*
equation index: 2657
type: SIMPLE_ASSIGN
z[167] = -0.013280000000000002
*/
void SpiralGalaxy_eqFunction_2657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2657};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2666]] /* z[167] STATE(1,vz[167]) */) = -0.013280000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9664(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9665(DATA *data, threadData_t *threadData);


/*
equation index: 2660
type: SIMPLE_ASSIGN
y[167] = r_init[167] * sin(theta[167] - 0.0033199999999999996)
*/
void SpiralGalaxy_eqFunction_2660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2660};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2166]] /* y[167] STATE(1,vy[167]) */) = ((data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */)) * (sin((data->simulationInfo->realParameter[1673] /* theta[167] PARAM */) - 0.0033199999999999996));
  TRACE_POP
}

/*
equation index: 2661
type: SIMPLE_ASSIGN
x[167] = r_init[167] * cos(theta[167] - 0.0033199999999999996)
*/
void SpiralGalaxy_eqFunction_2661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1666]] /* x[167] STATE(1,vx[167]) */) = ((data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */)) * (cos((data->simulationInfo->realParameter[1673] /* theta[167] PARAM */) - 0.0033199999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9666(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9667(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9670(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9669(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9668(DATA *data, threadData_t *threadData);


/*
equation index: 2667
type: SIMPLE_ASSIGN
vx[167] = (-sin(theta[167])) * r_init[167] * omega_c[167]
*/
void SpiralGalaxy_eqFunction_2667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2667};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[166]] /* vx[167] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1673] /* theta[167] PARAM */)))) * (((data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */)) * ((data->simulationInfo->realParameter[671] /* omega_c[167] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9663(DATA *data, threadData_t *threadData);


/*
equation index: 2669
type: SIMPLE_ASSIGN
vy[167] = cos(theta[167]) * r_init[167] * omega_c[167]
*/
void SpiralGalaxy_eqFunction_2669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[666]] /* vy[167] STATE(1) */) = (cos((data->simulationInfo->realParameter[1673] /* theta[167] PARAM */))) * (((data->simulationInfo->realParameter[1172] /* r_init[167] PARAM */)) * ((data->simulationInfo->realParameter[671] /* omega_c[167] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9662(DATA *data, threadData_t *threadData);


/*
equation index: 2671
type: SIMPLE_ASSIGN
vz[167] = 0.0
*/
void SpiralGalaxy_eqFunction_2671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2671};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1166]] /* vz[167] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9661(DATA *data, threadData_t *threadData);


/*
equation index: 2673
type: SIMPLE_ASSIGN
z[168] = -0.01312
*/
void SpiralGalaxy_eqFunction_2673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2673};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2667]] /* z[168] STATE(1,vz[168]) */) = -0.01312;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9674(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9675(DATA *data, threadData_t *threadData);


/*
equation index: 2676
type: SIMPLE_ASSIGN
y[168] = r_init[168] * sin(theta[168] - 0.0032799999999999995)
*/
void SpiralGalaxy_eqFunction_2676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2676};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2167]] /* y[168] STATE(1,vy[168]) */) = ((data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */)) * (sin((data->simulationInfo->realParameter[1674] /* theta[168] PARAM */) - 0.0032799999999999995));
  TRACE_POP
}

/*
equation index: 2677
type: SIMPLE_ASSIGN
x[168] = r_init[168] * cos(theta[168] - 0.0032799999999999995)
*/
void SpiralGalaxy_eqFunction_2677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1667]] /* x[168] STATE(1,vx[168]) */) = ((data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */)) * (cos((data->simulationInfo->realParameter[1674] /* theta[168] PARAM */) - 0.0032799999999999995));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9676(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9677(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9680(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9679(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9678(DATA *data, threadData_t *threadData);


/*
equation index: 2683
type: SIMPLE_ASSIGN
vx[168] = (-sin(theta[168])) * r_init[168] * omega_c[168]
*/
void SpiralGalaxy_eqFunction_2683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2683};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[167]] /* vx[168] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1674] /* theta[168] PARAM */)))) * (((data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */)) * ((data->simulationInfo->realParameter[672] /* omega_c[168] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9673(DATA *data, threadData_t *threadData);


/*
equation index: 2685
type: SIMPLE_ASSIGN
vy[168] = cos(theta[168]) * r_init[168] * omega_c[168]
*/
void SpiralGalaxy_eqFunction_2685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[667]] /* vy[168] STATE(1) */) = (cos((data->simulationInfo->realParameter[1674] /* theta[168] PARAM */))) * (((data->simulationInfo->realParameter[1173] /* r_init[168] PARAM */)) * ((data->simulationInfo->realParameter[672] /* omega_c[168] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9672(DATA *data, threadData_t *threadData);


/*
equation index: 2687
type: SIMPLE_ASSIGN
vz[168] = 0.0
*/
void SpiralGalaxy_eqFunction_2687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2687};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1167]] /* vz[168] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9671(DATA *data, threadData_t *threadData);


/*
equation index: 2689
type: SIMPLE_ASSIGN
z[169] = -0.012960000000000001
*/
void SpiralGalaxy_eqFunction_2689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2689};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2668]] /* z[169] STATE(1,vz[169]) */) = -0.012960000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9684(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9685(DATA *data, threadData_t *threadData);


/*
equation index: 2692
type: SIMPLE_ASSIGN
y[169] = r_init[169] * sin(theta[169] - 0.00324)
*/
void SpiralGalaxy_eqFunction_2692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2692};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2168]] /* y[169] STATE(1,vy[169]) */) = ((data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */)) * (sin((data->simulationInfo->realParameter[1675] /* theta[169] PARAM */) - 0.00324));
  TRACE_POP
}

/*
equation index: 2693
type: SIMPLE_ASSIGN
x[169] = r_init[169] * cos(theta[169] - 0.00324)
*/
void SpiralGalaxy_eqFunction_2693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1668]] /* x[169] STATE(1,vx[169]) */) = ((data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */)) * (cos((data->simulationInfo->realParameter[1675] /* theta[169] PARAM */) - 0.00324));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9686(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9687(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9690(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9689(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9688(DATA *data, threadData_t *threadData);


/*
equation index: 2699
type: SIMPLE_ASSIGN
vx[169] = (-sin(theta[169])) * r_init[169] * omega_c[169]
*/
void SpiralGalaxy_eqFunction_2699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2699};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[168]] /* vx[169] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1675] /* theta[169] PARAM */)))) * (((data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */)) * ((data->simulationInfo->realParameter[673] /* omega_c[169] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9683(DATA *data, threadData_t *threadData);


/*
equation index: 2701
type: SIMPLE_ASSIGN
vy[169] = cos(theta[169]) * r_init[169] * omega_c[169]
*/
void SpiralGalaxy_eqFunction_2701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[668]] /* vy[169] STATE(1) */) = (cos((data->simulationInfo->realParameter[1675] /* theta[169] PARAM */))) * (((data->simulationInfo->realParameter[1174] /* r_init[169] PARAM */)) * ((data->simulationInfo->realParameter[673] /* omega_c[169] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9682(DATA *data, threadData_t *threadData);


/*
equation index: 2703
type: SIMPLE_ASSIGN
vz[169] = 0.0
*/
void SpiralGalaxy_eqFunction_2703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2703};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1168]] /* vz[169] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9681(DATA *data, threadData_t *threadData);


/*
equation index: 2705
type: SIMPLE_ASSIGN
z[170] = -0.012800000000000002
*/
void SpiralGalaxy_eqFunction_2705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2705};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2669]] /* z[170] STATE(1,vz[170]) */) = -0.012800000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9694(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9695(DATA *data, threadData_t *threadData);


/*
equation index: 2708
type: SIMPLE_ASSIGN
y[170] = r_init[170] * sin(theta[170] - 0.0031999999999999997)
*/
void SpiralGalaxy_eqFunction_2708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2708};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2169]] /* y[170] STATE(1,vy[170]) */) = ((data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */)) * (sin((data->simulationInfo->realParameter[1676] /* theta[170] PARAM */) - 0.0031999999999999997));
  TRACE_POP
}

/*
equation index: 2709
type: SIMPLE_ASSIGN
x[170] = r_init[170] * cos(theta[170] - 0.0031999999999999997)
*/
void SpiralGalaxy_eqFunction_2709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1669]] /* x[170] STATE(1,vx[170]) */) = ((data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */)) * (cos((data->simulationInfo->realParameter[1676] /* theta[170] PARAM */) - 0.0031999999999999997));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9696(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9697(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9700(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9699(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9698(DATA *data, threadData_t *threadData);


/*
equation index: 2715
type: SIMPLE_ASSIGN
vx[170] = (-sin(theta[170])) * r_init[170] * omega_c[170]
*/
void SpiralGalaxy_eqFunction_2715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2715};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[169]] /* vx[170] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1676] /* theta[170] PARAM */)))) * (((data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */)) * ((data->simulationInfo->realParameter[674] /* omega_c[170] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9693(DATA *data, threadData_t *threadData);


/*
equation index: 2717
type: SIMPLE_ASSIGN
vy[170] = cos(theta[170]) * r_init[170] * omega_c[170]
*/
void SpiralGalaxy_eqFunction_2717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[669]] /* vy[170] STATE(1) */) = (cos((data->simulationInfo->realParameter[1676] /* theta[170] PARAM */))) * (((data->simulationInfo->realParameter[1175] /* r_init[170] PARAM */)) * ((data->simulationInfo->realParameter[674] /* omega_c[170] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9692(DATA *data, threadData_t *threadData);


/*
equation index: 2719
type: SIMPLE_ASSIGN
vz[170] = 0.0
*/
void SpiralGalaxy_eqFunction_2719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2719};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1169]] /* vz[170] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9691(DATA *data, threadData_t *threadData);


/*
equation index: 2721
type: SIMPLE_ASSIGN
z[171] = -0.01264
*/
void SpiralGalaxy_eqFunction_2721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2721};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2670]] /* z[171] STATE(1,vz[171]) */) = -0.01264;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9704(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9705(DATA *data, threadData_t *threadData);


/*
equation index: 2724
type: SIMPLE_ASSIGN
y[171] = r_init[171] * sin(theta[171] - 0.0031599999999999996)
*/
void SpiralGalaxy_eqFunction_2724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2724};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2170]] /* y[171] STATE(1,vy[171]) */) = ((data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */)) * (sin((data->simulationInfo->realParameter[1677] /* theta[171] PARAM */) - 0.0031599999999999996));
  TRACE_POP
}

/*
equation index: 2725
type: SIMPLE_ASSIGN
x[171] = r_init[171] * cos(theta[171] - 0.0031599999999999996)
*/
void SpiralGalaxy_eqFunction_2725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1670]] /* x[171] STATE(1,vx[171]) */) = ((data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */)) * (cos((data->simulationInfo->realParameter[1677] /* theta[171] PARAM */) - 0.0031599999999999996));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9706(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9707(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9710(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9709(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9708(DATA *data, threadData_t *threadData);


/*
equation index: 2731
type: SIMPLE_ASSIGN
vx[171] = (-sin(theta[171])) * r_init[171] * omega_c[171]
*/
void SpiralGalaxy_eqFunction_2731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2731};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[170]] /* vx[171] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1677] /* theta[171] PARAM */)))) * (((data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */)) * ((data->simulationInfo->realParameter[675] /* omega_c[171] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9703(DATA *data, threadData_t *threadData);


/*
equation index: 2733
type: SIMPLE_ASSIGN
vy[171] = cos(theta[171]) * r_init[171] * omega_c[171]
*/
void SpiralGalaxy_eqFunction_2733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[670]] /* vy[171] STATE(1) */) = (cos((data->simulationInfo->realParameter[1677] /* theta[171] PARAM */))) * (((data->simulationInfo->realParameter[1176] /* r_init[171] PARAM */)) * ((data->simulationInfo->realParameter[675] /* omega_c[171] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9702(DATA *data, threadData_t *threadData);


/*
equation index: 2735
type: SIMPLE_ASSIGN
vz[171] = 0.0
*/
void SpiralGalaxy_eqFunction_2735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2735};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1170]] /* vz[171] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9701(DATA *data, threadData_t *threadData);


/*
equation index: 2737
type: SIMPLE_ASSIGN
z[172] = -0.012480000000000002
*/
void SpiralGalaxy_eqFunction_2737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2737};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2671]] /* z[172] STATE(1,vz[172]) */) = -0.012480000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9714(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9715(DATA *data, threadData_t *threadData);


/*
equation index: 2740
type: SIMPLE_ASSIGN
y[172] = r_init[172] * sin(theta[172] - 0.003120000000000001)
*/
void SpiralGalaxy_eqFunction_2740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2740};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2171]] /* y[172] STATE(1,vy[172]) */) = ((data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */)) * (sin((data->simulationInfo->realParameter[1678] /* theta[172] PARAM */) - 0.003120000000000001));
  TRACE_POP
}

/*
equation index: 2741
type: SIMPLE_ASSIGN
x[172] = r_init[172] * cos(theta[172] - 0.003120000000000001)
*/
void SpiralGalaxy_eqFunction_2741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1671]] /* x[172] STATE(1,vx[172]) */) = ((data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */)) * (cos((data->simulationInfo->realParameter[1678] /* theta[172] PARAM */) - 0.003120000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9716(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9717(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9720(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9719(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9718(DATA *data, threadData_t *threadData);


/*
equation index: 2747
type: SIMPLE_ASSIGN
vx[172] = (-sin(theta[172])) * r_init[172] * omega_c[172]
*/
void SpiralGalaxy_eqFunction_2747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2747};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[171]] /* vx[172] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1678] /* theta[172] PARAM */)))) * (((data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */)) * ((data->simulationInfo->realParameter[676] /* omega_c[172] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9713(DATA *data, threadData_t *threadData);


/*
equation index: 2749
type: SIMPLE_ASSIGN
vy[172] = cos(theta[172]) * r_init[172] * omega_c[172]
*/
void SpiralGalaxy_eqFunction_2749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[671]] /* vy[172] STATE(1) */) = (cos((data->simulationInfo->realParameter[1678] /* theta[172] PARAM */))) * (((data->simulationInfo->realParameter[1177] /* r_init[172] PARAM */)) * ((data->simulationInfo->realParameter[676] /* omega_c[172] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9712(DATA *data, threadData_t *threadData);


/*
equation index: 2751
type: SIMPLE_ASSIGN
vz[172] = 0.0
*/
void SpiralGalaxy_eqFunction_2751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2751};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1171]] /* vz[172] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9711(DATA *data, threadData_t *threadData);


/*
equation index: 2753
type: SIMPLE_ASSIGN
z[173] = -0.012320000000000001
*/
void SpiralGalaxy_eqFunction_2753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2753};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2672]] /* z[173] STATE(1,vz[173]) */) = -0.012320000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9724(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9725(DATA *data, threadData_t *threadData);


/*
equation index: 2756
type: SIMPLE_ASSIGN
y[173] = r_init[173] * sin(theta[173] - 0.0030800000000000007)
*/
void SpiralGalaxy_eqFunction_2756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2756};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2172]] /* y[173] STATE(1,vy[173]) */) = ((data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */)) * (sin((data->simulationInfo->realParameter[1679] /* theta[173] PARAM */) - 0.0030800000000000007));
  TRACE_POP
}

/*
equation index: 2757
type: SIMPLE_ASSIGN
x[173] = r_init[173] * cos(theta[173] - 0.0030800000000000007)
*/
void SpiralGalaxy_eqFunction_2757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1672]] /* x[173] STATE(1,vx[173]) */) = ((data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */)) * (cos((data->simulationInfo->realParameter[1679] /* theta[173] PARAM */) - 0.0030800000000000007));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9726(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9727(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9730(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9729(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9728(DATA *data, threadData_t *threadData);


/*
equation index: 2763
type: SIMPLE_ASSIGN
vx[173] = (-sin(theta[173])) * r_init[173] * omega_c[173]
*/
void SpiralGalaxy_eqFunction_2763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2763};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[172]] /* vx[173] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1679] /* theta[173] PARAM */)))) * (((data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */)) * ((data->simulationInfo->realParameter[677] /* omega_c[173] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9723(DATA *data, threadData_t *threadData);


/*
equation index: 2765
type: SIMPLE_ASSIGN
vy[173] = cos(theta[173]) * r_init[173] * omega_c[173]
*/
void SpiralGalaxy_eqFunction_2765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[672]] /* vy[173] STATE(1) */) = (cos((data->simulationInfo->realParameter[1679] /* theta[173] PARAM */))) * (((data->simulationInfo->realParameter[1178] /* r_init[173] PARAM */)) * ((data->simulationInfo->realParameter[677] /* omega_c[173] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9722(DATA *data, threadData_t *threadData);


/*
equation index: 2767
type: SIMPLE_ASSIGN
vz[173] = 0.0
*/
void SpiralGalaxy_eqFunction_2767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2767};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1172]] /* vz[173] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9721(DATA *data, threadData_t *threadData);


/*
equation index: 2769
type: SIMPLE_ASSIGN
z[174] = -0.01216
*/
void SpiralGalaxy_eqFunction_2769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2769};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2673]] /* z[174] STATE(1,vz[174]) */) = -0.01216;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9734(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9735(DATA *data, threadData_t *threadData);


/*
equation index: 2772
type: SIMPLE_ASSIGN
y[174] = r_init[174] * sin(theta[174] - 0.0030400000000000006)
*/
void SpiralGalaxy_eqFunction_2772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2772};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2173]] /* y[174] STATE(1,vy[174]) */) = ((data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */)) * (sin((data->simulationInfo->realParameter[1680] /* theta[174] PARAM */) - 0.0030400000000000006));
  TRACE_POP
}

/*
equation index: 2773
type: SIMPLE_ASSIGN
x[174] = r_init[174] * cos(theta[174] - 0.0030400000000000006)
*/
void SpiralGalaxy_eqFunction_2773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1673]] /* x[174] STATE(1,vx[174]) */) = ((data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */)) * (cos((data->simulationInfo->realParameter[1680] /* theta[174] PARAM */) - 0.0030400000000000006));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9736(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9737(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9740(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9739(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9738(DATA *data, threadData_t *threadData);


/*
equation index: 2779
type: SIMPLE_ASSIGN
vx[174] = (-sin(theta[174])) * r_init[174] * omega_c[174]
*/
void SpiralGalaxy_eqFunction_2779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2779};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[173]] /* vx[174] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1680] /* theta[174] PARAM */)))) * (((data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */)) * ((data->simulationInfo->realParameter[678] /* omega_c[174] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9733(DATA *data, threadData_t *threadData);


/*
equation index: 2781
type: SIMPLE_ASSIGN
vy[174] = cos(theta[174]) * r_init[174] * omega_c[174]
*/
void SpiralGalaxy_eqFunction_2781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[673]] /* vy[174] STATE(1) */) = (cos((data->simulationInfo->realParameter[1680] /* theta[174] PARAM */))) * (((data->simulationInfo->realParameter[1179] /* r_init[174] PARAM */)) * ((data->simulationInfo->realParameter[678] /* omega_c[174] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9732(DATA *data, threadData_t *threadData);


/*
equation index: 2783
type: SIMPLE_ASSIGN
vz[174] = 0.0
*/
void SpiralGalaxy_eqFunction_2783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2783};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1173]] /* vz[174] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9731(DATA *data, threadData_t *threadData);


/*
equation index: 2785
type: SIMPLE_ASSIGN
z[175] = -0.012000000000000002
*/
void SpiralGalaxy_eqFunction_2785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2785};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2674]] /* z[175] STATE(1,vz[175]) */) = -0.012000000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9744(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9745(DATA *data, threadData_t *threadData);


/*
equation index: 2788
type: SIMPLE_ASSIGN
y[175] = r_init[175] * sin(theta[175] - 0.0030000000000000005)
*/
void SpiralGalaxy_eqFunction_2788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2788};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2174]] /* y[175] STATE(1,vy[175]) */) = ((data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */)) * (sin((data->simulationInfo->realParameter[1681] /* theta[175] PARAM */) - 0.0030000000000000005));
  TRACE_POP
}

/*
equation index: 2789
type: SIMPLE_ASSIGN
x[175] = r_init[175] * cos(theta[175] - 0.0030000000000000005)
*/
void SpiralGalaxy_eqFunction_2789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1674]] /* x[175] STATE(1,vx[175]) */) = ((data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */)) * (cos((data->simulationInfo->realParameter[1681] /* theta[175] PARAM */) - 0.0030000000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9746(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9747(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9750(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9749(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9748(DATA *data, threadData_t *threadData);


/*
equation index: 2795
type: SIMPLE_ASSIGN
vx[175] = (-sin(theta[175])) * r_init[175] * omega_c[175]
*/
void SpiralGalaxy_eqFunction_2795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2795};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[174]] /* vx[175] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1681] /* theta[175] PARAM */)))) * (((data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */)) * ((data->simulationInfo->realParameter[679] /* omega_c[175] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9743(DATA *data, threadData_t *threadData);


/*
equation index: 2797
type: SIMPLE_ASSIGN
vy[175] = cos(theta[175]) * r_init[175] * omega_c[175]
*/
void SpiralGalaxy_eqFunction_2797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[674]] /* vy[175] STATE(1) */) = (cos((data->simulationInfo->realParameter[1681] /* theta[175] PARAM */))) * (((data->simulationInfo->realParameter[1180] /* r_init[175] PARAM */)) * ((data->simulationInfo->realParameter[679] /* omega_c[175] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9742(DATA *data, threadData_t *threadData);


/*
equation index: 2799
type: SIMPLE_ASSIGN
vz[175] = 0.0
*/
void SpiralGalaxy_eqFunction_2799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2799};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1174]] /* vz[175] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9741(DATA *data, threadData_t *threadData);


/*
equation index: 2801
type: SIMPLE_ASSIGN
z[176] = -0.011840000000000002
*/
void SpiralGalaxy_eqFunction_2801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2801};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2675]] /* z[176] STATE(1,vz[176]) */) = -0.011840000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9754(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9755(DATA *data, threadData_t *threadData);


/*
equation index: 2804
type: SIMPLE_ASSIGN
y[176] = r_init[176] * sin(theta[176] - 0.0029600000000000004)
*/
void SpiralGalaxy_eqFunction_2804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2804};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2175]] /* y[176] STATE(1,vy[176]) */) = ((data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */)) * (sin((data->simulationInfo->realParameter[1682] /* theta[176] PARAM */) - 0.0029600000000000004));
  TRACE_POP
}

/*
equation index: 2805
type: SIMPLE_ASSIGN
x[176] = r_init[176] * cos(theta[176] - 0.0029600000000000004)
*/
void SpiralGalaxy_eqFunction_2805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1675]] /* x[176] STATE(1,vx[176]) */) = ((data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */)) * (cos((data->simulationInfo->realParameter[1682] /* theta[176] PARAM */) - 0.0029600000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9756(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9757(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9760(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9759(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9758(DATA *data, threadData_t *threadData);


/*
equation index: 2811
type: SIMPLE_ASSIGN
vx[176] = (-sin(theta[176])) * r_init[176] * omega_c[176]
*/
void SpiralGalaxy_eqFunction_2811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2811};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[175]] /* vx[176] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1682] /* theta[176] PARAM */)))) * (((data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */)) * ((data->simulationInfo->realParameter[680] /* omega_c[176] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9753(DATA *data, threadData_t *threadData);


/*
equation index: 2813
type: SIMPLE_ASSIGN
vy[176] = cos(theta[176]) * r_init[176] * omega_c[176]
*/
void SpiralGalaxy_eqFunction_2813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[675]] /* vy[176] STATE(1) */) = (cos((data->simulationInfo->realParameter[1682] /* theta[176] PARAM */))) * (((data->simulationInfo->realParameter[1181] /* r_init[176] PARAM */)) * ((data->simulationInfo->realParameter[680] /* omega_c[176] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9752(DATA *data, threadData_t *threadData);


/*
equation index: 2815
type: SIMPLE_ASSIGN
vz[176] = 0.0
*/
void SpiralGalaxy_eqFunction_2815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2815};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1175]] /* vz[176] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9751(DATA *data, threadData_t *threadData);


/*
equation index: 2817
type: SIMPLE_ASSIGN
z[177] = -0.01168
*/
void SpiralGalaxy_eqFunction_2817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2817};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2676]] /* z[177] STATE(1,vz[177]) */) = -0.01168;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9764(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9765(DATA *data, threadData_t *threadData);


/*
equation index: 2820
type: SIMPLE_ASSIGN
y[177] = r_init[177] * sin(theta[177] - 0.0029200000000000003)
*/
void SpiralGalaxy_eqFunction_2820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2820};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2176]] /* y[177] STATE(1,vy[177]) */) = ((data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */)) * (sin((data->simulationInfo->realParameter[1683] /* theta[177] PARAM */) - 0.0029200000000000003));
  TRACE_POP
}

/*
equation index: 2821
type: SIMPLE_ASSIGN
x[177] = r_init[177] * cos(theta[177] - 0.0029200000000000003)
*/
void SpiralGalaxy_eqFunction_2821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1676]] /* x[177] STATE(1,vx[177]) */) = ((data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */)) * (cos((data->simulationInfo->realParameter[1683] /* theta[177] PARAM */) - 0.0029200000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9766(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9767(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9770(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9769(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9768(DATA *data, threadData_t *threadData);


/*
equation index: 2827
type: SIMPLE_ASSIGN
vx[177] = (-sin(theta[177])) * r_init[177] * omega_c[177]
*/
void SpiralGalaxy_eqFunction_2827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2827};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[176]] /* vx[177] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1683] /* theta[177] PARAM */)))) * (((data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */)) * ((data->simulationInfo->realParameter[681] /* omega_c[177] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9763(DATA *data, threadData_t *threadData);


/*
equation index: 2829
type: SIMPLE_ASSIGN
vy[177] = cos(theta[177]) * r_init[177] * omega_c[177]
*/
void SpiralGalaxy_eqFunction_2829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[676]] /* vy[177] STATE(1) */) = (cos((data->simulationInfo->realParameter[1683] /* theta[177] PARAM */))) * (((data->simulationInfo->realParameter[1182] /* r_init[177] PARAM */)) * ((data->simulationInfo->realParameter[681] /* omega_c[177] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9762(DATA *data, threadData_t *threadData);


/*
equation index: 2831
type: SIMPLE_ASSIGN
vz[177] = 0.0
*/
void SpiralGalaxy_eqFunction_2831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2831};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1176]] /* vz[177] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9761(DATA *data, threadData_t *threadData);


/*
equation index: 2833
type: SIMPLE_ASSIGN
z[178] = -0.011520000000000002
*/
void SpiralGalaxy_eqFunction_2833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2833};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2677]] /* z[178] STATE(1,vz[178]) */) = -0.011520000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9774(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9775(DATA *data, threadData_t *threadData);


/*
equation index: 2836
type: SIMPLE_ASSIGN
y[178] = r_init[178] * sin(theta[178] - 0.00288)
*/
void SpiralGalaxy_eqFunction_2836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2836};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2177]] /* y[178] STATE(1,vy[178]) */) = ((data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */)) * (sin((data->simulationInfo->realParameter[1684] /* theta[178] PARAM */) - 0.00288));
  TRACE_POP
}

/*
equation index: 2837
type: SIMPLE_ASSIGN
x[178] = r_init[178] * cos(theta[178] - 0.00288)
*/
void SpiralGalaxy_eqFunction_2837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1677]] /* x[178] STATE(1,vx[178]) */) = ((data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */)) * (cos((data->simulationInfo->realParameter[1684] /* theta[178] PARAM */) - 0.00288));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9776(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9777(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9780(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9779(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9778(DATA *data, threadData_t *threadData);


/*
equation index: 2843
type: SIMPLE_ASSIGN
vx[178] = (-sin(theta[178])) * r_init[178] * omega_c[178]
*/
void SpiralGalaxy_eqFunction_2843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2843};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[177]] /* vx[178] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1684] /* theta[178] PARAM */)))) * (((data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */)) * ((data->simulationInfo->realParameter[682] /* omega_c[178] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9773(DATA *data, threadData_t *threadData);


/*
equation index: 2845
type: SIMPLE_ASSIGN
vy[178] = cos(theta[178]) * r_init[178] * omega_c[178]
*/
void SpiralGalaxy_eqFunction_2845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[677]] /* vy[178] STATE(1) */) = (cos((data->simulationInfo->realParameter[1684] /* theta[178] PARAM */))) * (((data->simulationInfo->realParameter[1183] /* r_init[178] PARAM */)) * ((data->simulationInfo->realParameter[682] /* omega_c[178] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9772(DATA *data, threadData_t *threadData);


/*
equation index: 2847
type: SIMPLE_ASSIGN
vz[178] = 0.0
*/
void SpiralGalaxy_eqFunction_2847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2847};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1177]] /* vz[178] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9771(DATA *data, threadData_t *threadData);


/*
equation index: 2849
type: SIMPLE_ASSIGN
z[179] = -0.011360000000000002
*/
void SpiralGalaxy_eqFunction_2849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2849};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2678]] /* z[179] STATE(1,vz[179]) */) = -0.011360000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9784(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9785(DATA *data, threadData_t *threadData);


/*
equation index: 2852
type: SIMPLE_ASSIGN
y[179] = r_init[179] * sin(theta[179] - 0.0028400000000000005)
*/
void SpiralGalaxy_eqFunction_2852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2852};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2178]] /* y[179] STATE(1,vy[179]) */) = ((data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */)) * (sin((data->simulationInfo->realParameter[1685] /* theta[179] PARAM */) - 0.0028400000000000005));
  TRACE_POP
}

/*
equation index: 2853
type: SIMPLE_ASSIGN
x[179] = r_init[179] * cos(theta[179] - 0.0028400000000000005)
*/
void SpiralGalaxy_eqFunction_2853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1678]] /* x[179] STATE(1,vx[179]) */) = ((data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */)) * (cos((data->simulationInfo->realParameter[1685] /* theta[179] PARAM */) - 0.0028400000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9786(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9787(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9790(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9789(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9788(DATA *data, threadData_t *threadData);


/*
equation index: 2859
type: SIMPLE_ASSIGN
vx[179] = (-sin(theta[179])) * r_init[179] * omega_c[179]
*/
void SpiralGalaxy_eqFunction_2859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2859};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[178]] /* vx[179] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1685] /* theta[179] PARAM */)))) * (((data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */)) * ((data->simulationInfo->realParameter[683] /* omega_c[179] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9783(DATA *data, threadData_t *threadData);


/*
equation index: 2861
type: SIMPLE_ASSIGN
vy[179] = cos(theta[179]) * r_init[179] * omega_c[179]
*/
void SpiralGalaxy_eqFunction_2861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[678]] /* vy[179] STATE(1) */) = (cos((data->simulationInfo->realParameter[1685] /* theta[179] PARAM */))) * (((data->simulationInfo->realParameter[1184] /* r_init[179] PARAM */)) * ((data->simulationInfo->realParameter[683] /* omega_c[179] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9782(DATA *data, threadData_t *threadData);


/*
equation index: 2863
type: SIMPLE_ASSIGN
vz[179] = 0.0
*/
void SpiralGalaxy_eqFunction_2863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2863};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1178]] /* vz[179] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9781(DATA *data, threadData_t *threadData);


/*
equation index: 2865
type: SIMPLE_ASSIGN
z[180] = -0.011200000000000002
*/
void SpiralGalaxy_eqFunction_2865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2865};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2679]] /* z[180] STATE(1,vz[180]) */) = -0.011200000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9794(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9795(DATA *data, threadData_t *threadData);


/*
equation index: 2868
type: SIMPLE_ASSIGN
y[180] = r_init[180] * sin(theta[180] - 0.0028000000000000004)
*/
void SpiralGalaxy_eqFunction_2868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2868};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2179]] /* y[180] STATE(1,vy[180]) */) = ((data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */)) * (sin((data->simulationInfo->realParameter[1686] /* theta[180] PARAM */) - 0.0028000000000000004));
  TRACE_POP
}

/*
equation index: 2869
type: SIMPLE_ASSIGN
x[180] = r_init[180] * cos(theta[180] - 0.0028000000000000004)
*/
void SpiralGalaxy_eqFunction_2869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1679]] /* x[180] STATE(1,vx[180]) */) = ((data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */)) * (cos((data->simulationInfo->realParameter[1686] /* theta[180] PARAM */) - 0.0028000000000000004));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9796(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9797(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9800(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9799(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9798(DATA *data, threadData_t *threadData);


/*
equation index: 2875
type: SIMPLE_ASSIGN
vx[180] = (-sin(theta[180])) * r_init[180] * omega_c[180]
*/
void SpiralGalaxy_eqFunction_2875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2875};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[179]] /* vx[180] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1686] /* theta[180] PARAM */)))) * (((data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */)) * ((data->simulationInfo->realParameter[684] /* omega_c[180] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9793(DATA *data, threadData_t *threadData);


/*
equation index: 2877
type: SIMPLE_ASSIGN
vy[180] = cos(theta[180]) * r_init[180] * omega_c[180]
*/
void SpiralGalaxy_eqFunction_2877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[679]] /* vy[180] STATE(1) */) = (cos((data->simulationInfo->realParameter[1686] /* theta[180] PARAM */))) * (((data->simulationInfo->realParameter[1185] /* r_init[180] PARAM */)) * ((data->simulationInfo->realParameter[684] /* omega_c[180] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9792(DATA *data, threadData_t *threadData);


/*
equation index: 2879
type: SIMPLE_ASSIGN
vz[180] = 0.0
*/
void SpiralGalaxy_eqFunction_2879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2879};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1179]] /* vz[180] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9791(DATA *data, threadData_t *threadData);


/*
equation index: 2881
type: SIMPLE_ASSIGN
z[181] = -0.011040000000000001
*/
void SpiralGalaxy_eqFunction_2881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2881};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2680]] /* z[181] STATE(1,vz[181]) */) = -0.011040000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9804(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9805(DATA *data, threadData_t *threadData);


/*
equation index: 2884
type: SIMPLE_ASSIGN
y[181] = r_init[181] * sin(theta[181] - 0.0027600000000000003)
*/
void SpiralGalaxy_eqFunction_2884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2884};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2180]] /* y[181] STATE(1,vy[181]) */) = ((data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */)) * (sin((data->simulationInfo->realParameter[1687] /* theta[181] PARAM */) - 0.0027600000000000003));
  TRACE_POP
}

/*
equation index: 2885
type: SIMPLE_ASSIGN
x[181] = r_init[181] * cos(theta[181] - 0.0027600000000000003)
*/
void SpiralGalaxy_eqFunction_2885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1680]] /* x[181] STATE(1,vx[181]) */) = ((data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */)) * (cos((data->simulationInfo->realParameter[1687] /* theta[181] PARAM */) - 0.0027600000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9806(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9807(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9810(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9809(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9808(DATA *data, threadData_t *threadData);


/*
equation index: 2891
type: SIMPLE_ASSIGN
vx[181] = (-sin(theta[181])) * r_init[181] * omega_c[181]
*/
void SpiralGalaxy_eqFunction_2891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2891};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[180]] /* vx[181] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1687] /* theta[181] PARAM */)))) * (((data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */)) * ((data->simulationInfo->realParameter[685] /* omega_c[181] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9803(DATA *data, threadData_t *threadData);


/*
equation index: 2893
type: SIMPLE_ASSIGN
vy[181] = cos(theta[181]) * r_init[181] * omega_c[181]
*/
void SpiralGalaxy_eqFunction_2893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[680]] /* vy[181] STATE(1) */) = (cos((data->simulationInfo->realParameter[1687] /* theta[181] PARAM */))) * (((data->simulationInfo->realParameter[1186] /* r_init[181] PARAM */)) * ((data->simulationInfo->realParameter[685] /* omega_c[181] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9802(DATA *data, threadData_t *threadData);


/*
equation index: 2895
type: SIMPLE_ASSIGN
vz[181] = 0.0
*/
void SpiralGalaxy_eqFunction_2895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2895};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1180]] /* vz[181] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9801(DATA *data, threadData_t *threadData);


/*
equation index: 2897
type: SIMPLE_ASSIGN
z[182] = -0.01088
*/
void SpiralGalaxy_eqFunction_2897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2897};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2681]] /* z[182] STATE(1,vz[182]) */) = -0.01088;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9814(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9815(DATA *data, threadData_t *threadData);


/*
equation index: 2900
type: SIMPLE_ASSIGN
y[182] = r_init[182] * sin(theta[182] - 0.00272)
*/
void SpiralGalaxy_eqFunction_2900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2900};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2181]] /* y[182] STATE(1,vy[182]) */) = ((data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */)) * (sin((data->simulationInfo->realParameter[1688] /* theta[182] PARAM */) - 0.00272));
  TRACE_POP
}

/*
equation index: 2901
type: SIMPLE_ASSIGN
x[182] = r_init[182] * cos(theta[182] - 0.00272)
*/
void SpiralGalaxy_eqFunction_2901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1681]] /* x[182] STATE(1,vx[182]) */) = ((data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */)) * (cos((data->simulationInfo->realParameter[1688] /* theta[182] PARAM */) - 0.00272));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9816(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9817(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9820(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9819(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9818(DATA *data, threadData_t *threadData);


/*
equation index: 2907
type: SIMPLE_ASSIGN
vx[182] = (-sin(theta[182])) * r_init[182] * omega_c[182]
*/
void SpiralGalaxy_eqFunction_2907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2907};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[181]] /* vx[182] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1688] /* theta[182] PARAM */)))) * (((data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */)) * ((data->simulationInfo->realParameter[686] /* omega_c[182] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9813(DATA *data, threadData_t *threadData);


/*
equation index: 2909
type: SIMPLE_ASSIGN
vy[182] = cos(theta[182]) * r_init[182] * omega_c[182]
*/
void SpiralGalaxy_eqFunction_2909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[681]] /* vy[182] STATE(1) */) = (cos((data->simulationInfo->realParameter[1688] /* theta[182] PARAM */))) * (((data->simulationInfo->realParameter[1187] /* r_init[182] PARAM */)) * ((data->simulationInfo->realParameter[686] /* omega_c[182] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9812(DATA *data, threadData_t *threadData);


/*
equation index: 2911
type: SIMPLE_ASSIGN
vz[182] = 0.0
*/
void SpiralGalaxy_eqFunction_2911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2911};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1181]] /* vz[182] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9811(DATA *data, threadData_t *threadData);


/*
equation index: 2913
type: SIMPLE_ASSIGN
z[183] = -0.01072
*/
void SpiralGalaxy_eqFunction_2913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2913};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2682]] /* z[183] STATE(1,vz[183]) */) = -0.01072;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9824(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9825(DATA *data, threadData_t *threadData);


/*
equation index: 2916
type: SIMPLE_ASSIGN
y[183] = r_init[183] * sin(theta[183] - 0.00268)
*/
void SpiralGalaxy_eqFunction_2916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2916};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2182]] /* y[183] STATE(1,vy[183]) */) = ((data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */)) * (sin((data->simulationInfo->realParameter[1689] /* theta[183] PARAM */) - 0.00268));
  TRACE_POP
}

/*
equation index: 2917
type: SIMPLE_ASSIGN
x[183] = r_init[183] * cos(theta[183] - 0.00268)
*/
void SpiralGalaxy_eqFunction_2917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1682]] /* x[183] STATE(1,vx[183]) */) = ((data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */)) * (cos((data->simulationInfo->realParameter[1689] /* theta[183] PARAM */) - 0.00268));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9826(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9827(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9830(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9829(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9828(DATA *data, threadData_t *threadData);


/*
equation index: 2923
type: SIMPLE_ASSIGN
vx[183] = (-sin(theta[183])) * r_init[183] * omega_c[183]
*/
void SpiralGalaxy_eqFunction_2923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2923};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[182]] /* vx[183] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1689] /* theta[183] PARAM */)))) * (((data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */)) * ((data->simulationInfo->realParameter[687] /* omega_c[183] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9823(DATA *data, threadData_t *threadData);


/*
equation index: 2925
type: SIMPLE_ASSIGN
vy[183] = cos(theta[183]) * r_init[183] * omega_c[183]
*/
void SpiralGalaxy_eqFunction_2925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[682]] /* vy[183] STATE(1) */) = (cos((data->simulationInfo->realParameter[1689] /* theta[183] PARAM */))) * (((data->simulationInfo->realParameter[1188] /* r_init[183] PARAM */)) * ((data->simulationInfo->realParameter[687] /* omega_c[183] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9822(DATA *data, threadData_t *threadData);


/*
equation index: 2927
type: SIMPLE_ASSIGN
vz[183] = 0.0
*/
void SpiralGalaxy_eqFunction_2927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2927};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1182]] /* vz[183] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9821(DATA *data, threadData_t *threadData);


/*
equation index: 2929
type: SIMPLE_ASSIGN
z[184] = -0.01056
*/
void SpiralGalaxy_eqFunction_2929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2929};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2683]] /* z[184] STATE(1,vz[184]) */) = -0.01056;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9834(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9835(DATA *data, threadData_t *threadData);


/*
equation index: 2932
type: SIMPLE_ASSIGN
y[184] = r_init[184] * sin(theta[184] - 0.00264)
*/
void SpiralGalaxy_eqFunction_2932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2932};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2183]] /* y[184] STATE(1,vy[184]) */) = ((data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */)) * (sin((data->simulationInfo->realParameter[1690] /* theta[184] PARAM */) - 0.00264));
  TRACE_POP
}

/*
equation index: 2933
type: SIMPLE_ASSIGN
x[184] = r_init[184] * cos(theta[184] - 0.00264)
*/
void SpiralGalaxy_eqFunction_2933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1683]] /* x[184] STATE(1,vx[184]) */) = ((data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */)) * (cos((data->simulationInfo->realParameter[1690] /* theta[184] PARAM */) - 0.00264));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9836(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9837(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9840(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9839(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9838(DATA *data, threadData_t *threadData);


/*
equation index: 2939
type: SIMPLE_ASSIGN
vx[184] = (-sin(theta[184])) * r_init[184] * omega_c[184]
*/
void SpiralGalaxy_eqFunction_2939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2939};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[183]] /* vx[184] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1690] /* theta[184] PARAM */)))) * (((data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */)) * ((data->simulationInfo->realParameter[688] /* omega_c[184] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9833(DATA *data, threadData_t *threadData);


/*
equation index: 2941
type: SIMPLE_ASSIGN
vy[184] = cos(theta[184]) * r_init[184] * omega_c[184]
*/
void SpiralGalaxy_eqFunction_2941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[683]] /* vy[184] STATE(1) */) = (cos((data->simulationInfo->realParameter[1690] /* theta[184] PARAM */))) * (((data->simulationInfo->realParameter[1189] /* r_init[184] PARAM */)) * ((data->simulationInfo->realParameter[688] /* omega_c[184] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9832(DATA *data, threadData_t *threadData);


/*
equation index: 2943
type: SIMPLE_ASSIGN
vz[184] = 0.0
*/
void SpiralGalaxy_eqFunction_2943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2943};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1183]] /* vz[184] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9831(DATA *data, threadData_t *threadData);


/*
equation index: 2945
type: SIMPLE_ASSIGN
z[185] = -0.010400000000000003
*/
void SpiralGalaxy_eqFunction_2945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2945};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2684]] /* z[185] STATE(1,vz[185]) */) = -0.010400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9844(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9845(DATA *data, threadData_t *threadData);


/*
equation index: 2948
type: SIMPLE_ASSIGN
y[185] = r_init[185] * sin(theta[185] - 0.0026000000000000003)
*/
void SpiralGalaxy_eqFunction_2948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2948};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2184]] /* y[185] STATE(1,vy[185]) */) = ((data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */)) * (sin((data->simulationInfo->realParameter[1691] /* theta[185] PARAM */) - 0.0026000000000000003));
  TRACE_POP
}

/*
equation index: 2949
type: SIMPLE_ASSIGN
x[185] = r_init[185] * cos(theta[185] - 0.0026000000000000003)
*/
void SpiralGalaxy_eqFunction_2949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1684]] /* x[185] STATE(1,vx[185]) */) = ((data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */)) * (cos((data->simulationInfo->realParameter[1691] /* theta[185] PARAM */) - 0.0026000000000000003));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9846(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9847(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9850(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9849(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9848(DATA *data, threadData_t *threadData);


/*
equation index: 2955
type: SIMPLE_ASSIGN
vx[185] = (-sin(theta[185])) * r_init[185] * omega_c[185]
*/
void SpiralGalaxy_eqFunction_2955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2955};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[184]] /* vx[185] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1691] /* theta[185] PARAM */)))) * (((data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */)) * ((data->simulationInfo->realParameter[689] /* omega_c[185] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9843(DATA *data, threadData_t *threadData);


/*
equation index: 2957
type: SIMPLE_ASSIGN
vy[185] = cos(theta[185]) * r_init[185] * omega_c[185]
*/
void SpiralGalaxy_eqFunction_2957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[684]] /* vy[185] STATE(1) */) = (cos((data->simulationInfo->realParameter[1691] /* theta[185] PARAM */))) * (((data->simulationInfo->realParameter[1190] /* r_init[185] PARAM */)) * ((data->simulationInfo->realParameter[689] /* omega_c[185] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9842(DATA *data, threadData_t *threadData);


/*
equation index: 2959
type: SIMPLE_ASSIGN
vz[185] = 0.0
*/
void SpiralGalaxy_eqFunction_2959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2959};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1184]] /* vz[185] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9841(DATA *data, threadData_t *threadData);


/*
equation index: 2961
type: SIMPLE_ASSIGN
z[186] = -0.01024
*/
void SpiralGalaxy_eqFunction_2961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2961};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2685]] /* z[186] STATE(1,vz[186]) */) = -0.01024;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9854(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9855(DATA *data, threadData_t *threadData);


/*
equation index: 2964
type: SIMPLE_ASSIGN
y[186] = r_init[186] * sin(theta[186] - 0.00256)
*/
void SpiralGalaxy_eqFunction_2964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2964};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2185]] /* y[186] STATE(1,vy[186]) */) = ((data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */)) * (sin((data->simulationInfo->realParameter[1692] /* theta[186] PARAM */) - 0.00256));
  TRACE_POP
}

/*
equation index: 2965
type: SIMPLE_ASSIGN
x[186] = r_init[186] * cos(theta[186] - 0.00256)
*/
void SpiralGalaxy_eqFunction_2965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1685]] /* x[186] STATE(1,vx[186]) */) = ((data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */)) * (cos((data->simulationInfo->realParameter[1692] /* theta[186] PARAM */) - 0.00256));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9856(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9857(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9860(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9859(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9858(DATA *data, threadData_t *threadData);


/*
equation index: 2971
type: SIMPLE_ASSIGN
vx[186] = (-sin(theta[186])) * r_init[186] * omega_c[186]
*/
void SpiralGalaxy_eqFunction_2971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2971};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[185]] /* vx[186] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1692] /* theta[186] PARAM */)))) * (((data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */)) * ((data->simulationInfo->realParameter[690] /* omega_c[186] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9853(DATA *data, threadData_t *threadData);


/*
equation index: 2973
type: SIMPLE_ASSIGN
vy[186] = cos(theta[186]) * r_init[186] * omega_c[186]
*/
void SpiralGalaxy_eqFunction_2973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[685]] /* vy[186] STATE(1) */) = (cos((data->simulationInfo->realParameter[1692] /* theta[186] PARAM */))) * (((data->simulationInfo->realParameter[1191] /* r_init[186] PARAM */)) * ((data->simulationInfo->realParameter[690] /* omega_c[186] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9852(DATA *data, threadData_t *threadData);


/*
equation index: 2975
type: SIMPLE_ASSIGN
vz[186] = 0.0
*/
void SpiralGalaxy_eqFunction_2975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2975};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1185]] /* vz[186] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9851(DATA *data, threadData_t *threadData);


/*
equation index: 2977
type: SIMPLE_ASSIGN
z[187] = -0.01008
*/
void SpiralGalaxy_eqFunction_2977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2977};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2686]] /* z[187] STATE(1,vz[187]) */) = -0.01008;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9864(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9865(DATA *data, threadData_t *threadData);


/*
equation index: 2980
type: SIMPLE_ASSIGN
y[187] = r_init[187] * sin(theta[187] - 0.00252)
*/
void SpiralGalaxy_eqFunction_2980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2980};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2186]] /* y[187] STATE(1,vy[187]) */) = ((data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */)) * (sin((data->simulationInfo->realParameter[1693] /* theta[187] PARAM */) - 0.00252));
  TRACE_POP
}

/*
equation index: 2981
type: SIMPLE_ASSIGN
x[187] = r_init[187] * cos(theta[187] - 0.00252)
*/
void SpiralGalaxy_eqFunction_2981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1686]] /* x[187] STATE(1,vx[187]) */) = ((data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */)) * (cos((data->simulationInfo->realParameter[1693] /* theta[187] PARAM */) - 0.00252));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9866(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9867(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9870(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9869(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9868(DATA *data, threadData_t *threadData);


/*
equation index: 2987
type: SIMPLE_ASSIGN
vx[187] = (-sin(theta[187])) * r_init[187] * omega_c[187]
*/
void SpiralGalaxy_eqFunction_2987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2987};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[186]] /* vx[187] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1693] /* theta[187] PARAM */)))) * (((data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */)) * ((data->simulationInfo->realParameter[691] /* omega_c[187] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9863(DATA *data, threadData_t *threadData);


/*
equation index: 2989
type: SIMPLE_ASSIGN
vy[187] = cos(theta[187]) * r_init[187] * omega_c[187]
*/
void SpiralGalaxy_eqFunction_2989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[686]] /* vy[187] STATE(1) */) = (cos((data->simulationInfo->realParameter[1693] /* theta[187] PARAM */))) * (((data->simulationInfo->realParameter[1192] /* r_init[187] PARAM */)) * ((data->simulationInfo->realParameter[691] /* omega_c[187] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9862(DATA *data, threadData_t *threadData);


/*
equation index: 2991
type: SIMPLE_ASSIGN
vz[187] = 0.0
*/
void SpiralGalaxy_eqFunction_2991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2991};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1186]] /* vz[187] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9861(DATA *data, threadData_t *threadData);


/*
equation index: 2993
type: SIMPLE_ASSIGN
z[188] = -0.009920000000000002
*/
void SpiralGalaxy_eqFunction_2993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2993};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2687]] /* z[188] STATE(1,vz[188]) */) = -0.009920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9874(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9875(DATA *data, threadData_t *threadData);


/*
equation index: 2996
type: SIMPLE_ASSIGN
y[188] = r_init[188] * sin(theta[188] - 0.00248)
*/
void SpiralGalaxy_eqFunction_2996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2996};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2187]] /* y[188] STATE(1,vy[188]) */) = ((data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */)) * (sin((data->simulationInfo->realParameter[1694] /* theta[188] PARAM */) - 0.00248));
  TRACE_POP
}

/*
equation index: 2997
type: SIMPLE_ASSIGN
x[188] = r_init[188] * cos(theta[188] - 0.00248)
*/
void SpiralGalaxy_eqFunction_2997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1687]] /* x[188] STATE(1,vx[188]) */) = ((data->simulationInfo->realParameter[1193] /* r_init[188] PARAM */)) * (cos((data->simulationInfo->realParameter[1694] /* theta[188] PARAM */) - 0.00248));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_9876(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9877(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_9880(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_5(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_2501(data, threadData);
  SpiralGalaxy_eqFunction_9566(data, threadData);
  SpiralGalaxy_eqFunction_9567(data, threadData);
  SpiralGalaxy_eqFunction_9570(data, threadData);
  SpiralGalaxy_eqFunction_9569(data, threadData);
  SpiralGalaxy_eqFunction_9568(data, threadData);
  SpiralGalaxy_eqFunction_2507(data, threadData);
  SpiralGalaxy_eqFunction_9563(data, threadData);
  SpiralGalaxy_eqFunction_2509(data, threadData);
  SpiralGalaxy_eqFunction_9562(data, threadData);
  SpiralGalaxy_eqFunction_2511(data, threadData);
  SpiralGalaxy_eqFunction_9561(data, threadData);
  SpiralGalaxy_eqFunction_2513(data, threadData);
  SpiralGalaxy_eqFunction_9574(data, threadData);
  SpiralGalaxy_eqFunction_9575(data, threadData);
  SpiralGalaxy_eqFunction_2516(data, threadData);
  SpiralGalaxy_eqFunction_2517(data, threadData);
  SpiralGalaxy_eqFunction_9576(data, threadData);
  SpiralGalaxy_eqFunction_9577(data, threadData);
  SpiralGalaxy_eqFunction_9580(data, threadData);
  SpiralGalaxy_eqFunction_9579(data, threadData);
  SpiralGalaxy_eqFunction_9578(data, threadData);
  SpiralGalaxy_eqFunction_2523(data, threadData);
  SpiralGalaxy_eqFunction_9573(data, threadData);
  SpiralGalaxy_eqFunction_2525(data, threadData);
  SpiralGalaxy_eqFunction_9572(data, threadData);
  SpiralGalaxy_eqFunction_2527(data, threadData);
  SpiralGalaxy_eqFunction_9571(data, threadData);
  SpiralGalaxy_eqFunction_2529(data, threadData);
  SpiralGalaxy_eqFunction_9584(data, threadData);
  SpiralGalaxy_eqFunction_9585(data, threadData);
  SpiralGalaxy_eqFunction_2532(data, threadData);
  SpiralGalaxy_eqFunction_2533(data, threadData);
  SpiralGalaxy_eqFunction_9586(data, threadData);
  SpiralGalaxy_eqFunction_9587(data, threadData);
  SpiralGalaxy_eqFunction_9590(data, threadData);
  SpiralGalaxy_eqFunction_9589(data, threadData);
  SpiralGalaxy_eqFunction_9588(data, threadData);
  SpiralGalaxy_eqFunction_2539(data, threadData);
  SpiralGalaxy_eqFunction_9583(data, threadData);
  SpiralGalaxy_eqFunction_2541(data, threadData);
  SpiralGalaxy_eqFunction_9582(data, threadData);
  SpiralGalaxy_eqFunction_2543(data, threadData);
  SpiralGalaxy_eqFunction_9581(data, threadData);
  SpiralGalaxy_eqFunction_2545(data, threadData);
  SpiralGalaxy_eqFunction_9594(data, threadData);
  SpiralGalaxy_eqFunction_9595(data, threadData);
  SpiralGalaxy_eqFunction_2548(data, threadData);
  SpiralGalaxy_eqFunction_2549(data, threadData);
  SpiralGalaxy_eqFunction_9596(data, threadData);
  SpiralGalaxy_eqFunction_9597(data, threadData);
  SpiralGalaxy_eqFunction_9600(data, threadData);
  SpiralGalaxy_eqFunction_9599(data, threadData);
  SpiralGalaxy_eqFunction_9598(data, threadData);
  SpiralGalaxy_eqFunction_2555(data, threadData);
  SpiralGalaxy_eqFunction_9593(data, threadData);
  SpiralGalaxy_eqFunction_2557(data, threadData);
  SpiralGalaxy_eqFunction_9592(data, threadData);
  SpiralGalaxy_eqFunction_2559(data, threadData);
  SpiralGalaxy_eqFunction_9591(data, threadData);
  SpiralGalaxy_eqFunction_2561(data, threadData);
  SpiralGalaxy_eqFunction_9604(data, threadData);
  SpiralGalaxy_eqFunction_9605(data, threadData);
  SpiralGalaxy_eqFunction_2564(data, threadData);
  SpiralGalaxy_eqFunction_2565(data, threadData);
  SpiralGalaxy_eqFunction_9606(data, threadData);
  SpiralGalaxy_eqFunction_9607(data, threadData);
  SpiralGalaxy_eqFunction_9610(data, threadData);
  SpiralGalaxy_eqFunction_9609(data, threadData);
  SpiralGalaxy_eqFunction_9608(data, threadData);
  SpiralGalaxy_eqFunction_2571(data, threadData);
  SpiralGalaxy_eqFunction_9603(data, threadData);
  SpiralGalaxy_eqFunction_2573(data, threadData);
  SpiralGalaxy_eqFunction_9602(data, threadData);
  SpiralGalaxy_eqFunction_2575(data, threadData);
  SpiralGalaxy_eqFunction_9601(data, threadData);
  SpiralGalaxy_eqFunction_2577(data, threadData);
  SpiralGalaxy_eqFunction_9614(data, threadData);
  SpiralGalaxy_eqFunction_9615(data, threadData);
  SpiralGalaxy_eqFunction_2580(data, threadData);
  SpiralGalaxy_eqFunction_2581(data, threadData);
  SpiralGalaxy_eqFunction_9616(data, threadData);
  SpiralGalaxy_eqFunction_9617(data, threadData);
  SpiralGalaxy_eqFunction_9620(data, threadData);
  SpiralGalaxy_eqFunction_9619(data, threadData);
  SpiralGalaxy_eqFunction_9618(data, threadData);
  SpiralGalaxy_eqFunction_2587(data, threadData);
  SpiralGalaxy_eqFunction_9613(data, threadData);
  SpiralGalaxy_eqFunction_2589(data, threadData);
  SpiralGalaxy_eqFunction_9612(data, threadData);
  SpiralGalaxy_eqFunction_2591(data, threadData);
  SpiralGalaxy_eqFunction_9611(data, threadData);
  SpiralGalaxy_eqFunction_2593(data, threadData);
  SpiralGalaxy_eqFunction_9624(data, threadData);
  SpiralGalaxy_eqFunction_9625(data, threadData);
  SpiralGalaxy_eqFunction_2596(data, threadData);
  SpiralGalaxy_eqFunction_2597(data, threadData);
  SpiralGalaxy_eqFunction_9626(data, threadData);
  SpiralGalaxy_eqFunction_9627(data, threadData);
  SpiralGalaxy_eqFunction_9630(data, threadData);
  SpiralGalaxy_eqFunction_9629(data, threadData);
  SpiralGalaxy_eqFunction_9628(data, threadData);
  SpiralGalaxy_eqFunction_2603(data, threadData);
  SpiralGalaxy_eqFunction_9623(data, threadData);
  SpiralGalaxy_eqFunction_2605(data, threadData);
  SpiralGalaxy_eqFunction_9622(data, threadData);
  SpiralGalaxy_eqFunction_2607(data, threadData);
  SpiralGalaxy_eqFunction_9621(data, threadData);
  SpiralGalaxy_eqFunction_2609(data, threadData);
  SpiralGalaxy_eqFunction_9634(data, threadData);
  SpiralGalaxy_eqFunction_9635(data, threadData);
  SpiralGalaxy_eqFunction_2612(data, threadData);
  SpiralGalaxy_eqFunction_2613(data, threadData);
  SpiralGalaxy_eqFunction_9636(data, threadData);
  SpiralGalaxy_eqFunction_9637(data, threadData);
  SpiralGalaxy_eqFunction_9640(data, threadData);
  SpiralGalaxy_eqFunction_9639(data, threadData);
  SpiralGalaxy_eqFunction_9638(data, threadData);
  SpiralGalaxy_eqFunction_2619(data, threadData);
  SpiralGalaxy_eqFunction_9633(data, threadData);
  SpiralGalaxy_eqFunction_2621(data, threadData);
  SpiralGalaxy_eqFunction_9632(data, threadData);
  SpiralGalaxy_eqFunction_2623(data, threadData);
  SpiralGalaxy_eqFunction_9631(data, threadData);
  SpiralGalaxy_eqFunction_2625(data, threadData);
  SpiralGalaxy_eqFunction_9644(data, threadData);
  SpiralGalaxy_eqFunction_9645(data, threadData);
  SpiralGalaxy_eqFunction_2628(data, threadData);
  SpiralGalaxy_eqFunction_2629(data, threadData);
  SpiralGalaxy_eqFunction_9646(data, threadData);
  SpiralGalaxy_eqFunction_9647(data, threadData);
  SpiralGalaxy_eqFunction_9650(data, threadData);
  SpiralGalaxy_eqFunction_9649(data, threadData);
  SpiralGalaxy_eqFunction_9648(data, threadData);
  SpiralGalaxy_eqFunction_2635(data, threadData);
  SpiralGalaxy_eqFunction_9643(data, threadData);
  SpiralGalaxy_eqFunction_2637(data, threadData);
  SpiralGalaxy_eqFunction_9642(data, threadData);
  SpiralGalaxy_eqFunction_2639(data, threadData);
  SpiralGalaxy_eqFunction_9641(data, threadData);
  SpiralGalaxy_eqFunction_2641(data, threadData);
  SpiralGalaxy_eqFunction_9654(data, threadData);
  SpiralGalaxy_eqFunction_9655(data, threadData);
  SpiralGalaxy_eqFunction_2644(data, threadData);
  SpiralGalaxy_eqFunction_2645(data, threadData);
  SpiralGalaxy_eqFunction_9656(data, threadData);
  SpiralGalaxy_eqFunction_9657(data, threadData);
  SpiralGalaxy_eqFunction_9660(data, threadData);
  SpiralGalaxy_eqFunction_9659(data, threadData);
  SpiralGalaxy_eqFunction_9658(data, threadData);
  SpiralGalaxy_eqFunction_2651(data, threadData);
  SpiralGalaxy_eqFunction_9653(data, threadData);
  SpiralGalaxy_eqFunction_2653(data, threadData);
  SpiralGalaxy_eqFunction_9652(data, threadData);
  SpiralGalaxy_eqFunction_2655(data, threadData);
  SpiralGalaxy_eqFunction_9651(data, threadData);
  SpiralGalaxy_eqFunction_2657(data, threadData);
  SpiralGalaxy_eqFunction_9664(data, threadData);
  SpiralGalaxy_eqFunction_9665(data, threadData);
  SpiralGalaxy_eqFunction_2660(data, threadData);
  SpiralGalaxy_eqFunction_2661(data, threadData);
  SpiralGalaxy_eqFunction_9666(data, threadData);
  SpiralGalaxy_eqFunction_9667(data, threadData);
  SpiralGalaxy_eqFunction_9670(data, threadData);
  SpiralGalaxy_eqFunction_9669(data, threadData);
  SpiralGalaxy_eqFunction_9668(data, threadData);
  SpiralGalaxy_eqFunction_2667(data, threadData);
  SpiralGalaxy_eqFunction_9663(data, threadData);
  SpiralGalaxy_eqFunction_2669(data, threadData);
  SpiralGalaxy_eqFunction_9662(data, threadData);
  SpiralGalaxy_eqFunction_2671(data, threadData);
  SpiralGalaxy_eqFunction_9661(data, threadData);
  SpiralGalaxy_eqFunction_2673(data, threadData);
  SpiralGalaxy_eqFunction_9674(data, threadData);
  SpiralGalaxy_eqFunction_9675(data, threadData);
  SpiralGalaxy_eqFunction_2676(data, threadData);
  SpiralGalaxy_eqFunction_2677(data, threadData);
  SpiralGalaxy_eqFunction_9676(data, threadData);
  SpiralGalaxy_eqFunction_9677(data, threadData);
  SpiralGalaxy_eqFunction_9680(data, threadData);
  SpiralGalaxy_eqFunction_9679(data, threadData);
  SpiralGalaxy_eqFunction_9678(data, threadData);
  SpiralGalaxy_eqFunction_2683(data, threadData);
  SpiralGalaxy_eqFunction_9673(data, threadData);
  SpiralGalaxy_eqFunction_2685(data, threadData);
  SpiralGalaxy_eqFunction_9672(data, threadData);
  SpiralGalaxy_eqFunction_2687(data, threadData);
  SpiralGalaxy_eqFunction_9671(data, threadData);
  SpiralGalaxy_eqFunction_2689(data, threadData);
  SpiralGalaxy_eqFunction_9684(data, threadData);
  SpiralGalaxy_eqFunction_9685(data, threadData);
  SpiralGalaxy_eqFunction_2692(data, threadData);
  SpiralGalaxy_eqFunction_2693(data, threadData);
  SpiralGalaxy_eqFunction_9686(data, threadData);
  SpiralGalaxy_eqFunction_9687(data, threadData);
  SpiralGalaxy_eqFunction_9690(data, threadData);
  SpiralGalaxy_eqFunction_9689(data, threadData);
  SpiralGalaxy_eqFunction_9688(data, threadData);
  SpiralGalaxy_eqFunction_2699(data, threadData);
  SpiralGalaxy_eqFunction_9683(data, threadData);
  SpiralGalaxy_eqFunction_2701(data, threadData);
  SpiralGalaxy_eqFunction_9682(data, threadData);
  SpiralGalaxy_eqFunction_2703(data, threadData);
  SpiralGalaxy_eqFunction_9681(data, threadData);
  SpiralGalaxy_eqFunction_2705(data, threadData);
  SpiralGalaxy_eqFunction_9694(data, threadData);
  SpiralGalaxy_eqFunction_9695(data, threadData);
  SpiralGalaxy_eqFunction_2708(data, threadData);
  SpiralGalaxy_eqFunction_2709(data, threadData);
  SpiralGalaxy_eqFunction_9696(data, threadData);
  SpiralGalaxy_eqFunction_9697(data, threadData);
  SpiralGalaxy_eqFunction_9700(data, threadData);
  SpiralGalaxy_eqFunction_9699(data, threadData);
  SpiralGalaxy_eqFunction_9698(data, threadData);
  SpiralGalaxy_eqFunction_2715(data, threadData);
  SpiralGalaxy_eqFunction_9693(data, threadData);
  SpiralGalaxy_eqFunction_2717(data, threadData);
  SpiralGalaxy_eqFunction_9692(data, threadData);
  SpiralGalaxy_eqFunction_2719(data, threadData);
  SpiralGalaxy_eqFunction_9691(data, threadData);
  SpiralGalaxy_eqFunction_2721(data, threadData);
  SpiralGalaxy_eqFunction_9704(data, threadData);
  SpiralGalaxy_eqFunction_9705(data, threadData);
  SpiralGalaxy_eqFunction_2724(data, threadData);
  SpiralGalaxy_eqFunction_2725(data, threadData);
  SpiralGalaxy_eqFunction_9706(data, threadData);
  SpiralGalaxy_eqFunction_9707(data, threadData);
  SpiralGalaxy_eqFunction_9710(data, threadData);
  SpiralGalaxy_eqFunction_9709(data, threadData);
  SpiralGalaxy_eqFunction_9708(data, threadData);
  SpiralGalaxy_eqFunction_2731(data, threadData);
  SpiralGalaxy_eqFunction_9703(data, threadData);
  SpiralGalaxy_eqFunction_2733(data, threadData);
  SpiralGalaxy_eqFunction_9702(data, threadData);
  SpiralGalaxy_eqFunction_2735(data, threadData);
  SpiralGalaxy_eqFunction_9701(data, threadData);
  SpiralGalaxy_eqFunction_2737(data, threadData);
  SpiralGalaxy_eqFunction_9714(data, threadData);
  SpiralGalaxy_eqFunction_9715(data, threadData);
  SpiralGalaxy_eqFunction_2740(data, threadData);
  SpiralGalaxy_eqFunction_2741(data, threadData);
  SpiralGalaxy_eqFunction_9716(data, threadData);
  SpiralGalaxy_eqFunction_9717(data, threadData);
  SpiralGalaxy_eqFunction_9720(data, threadData);
  SpiralGalaxy_eqFunction_9719(data, threadData);
  SpiralGalaxy_eqFunction_9718(data, threadData);
  SpiralGalaxy_eqFunction_2747(data, threadData);
  SpiralGalaxy_eqFunction_9713(data, threadData);
  SpiralGalaxy_eqFunction_2749(data, threadData);
  SpiralGalaxy_eqFunction_9712(data, threadData);
  SpiralGalaxy_eqFunction_2751(data, threadData);
  SpiralGalaxy_eqFunction_9711(data, threadData);
  SpiralGalaxy_eqFunction_2753(data, threadData);
  SpiralGalaxy_eqFunction_9724(data, threadData);
  SpiralGalaxy_eqFunction_9725(data, threadData);
  SpiralGalaxy_eqFunction_2756(data, threadData);
  SpiralGalaxy_eqFunction_2757(data, threadData);
  SpiralGalaxy_eqFunction_9726(data, threadData);
  SpiralGalaxy_eqFunction_9727(data, threadData);
  SpiralGalaxy_eqFunction_9730(data, threadData);
  SpiralGalaxy_eqFunction_9729(data, threadData);
  SpiralGalaxy_eqFunction_9728(data, threadData);
  SpiralGalaxy_eqFunction_2763(data, threadData);
  SpiralGalaxy_eqFunction_9723(data, threadData);
  SpiralGalaxy_eqFunction_2765(data, threadData);
  SpiralGalaxy_eqFunction_9722(data, threadData);
  SpiralGalaxy_eqFunction_2767(data, threadData);
  SpiralGalaxy_eqFunction_9721(data, threadData);
  SpiralGalaxy_eqFunction_2769(data, threadData);
  SpiralGalaxy_eqFunction_9734(data, threadData);
  SpiralGalaxy_eqFunction_9735(data, threadData);
  SpiralGalaxy_eqFunction_2772(data, threadData);
  SpiralGalaxy_eqFunction_2773(data, threadData);
  SpiralGalaxy_eqFunction_9736(data, threadData);
  SpiralGalaxy_eqFunction_9737(data, threadData);
  SpiralGalaxy_eqFunction_9740(data, threadData);
  SpiralGalaxy_eqFunction_9739(data, threadData);
  SpiralGalaxy_eqFunction_9738(data, threadData);
  SpiralGalaxy_eqFunction_2779(data, threadData);
  SpiralGalaxy_eqFunction_9733(data, threadData);
  SpiralGalaxy_eqFunction_2781(data, threadData);
  SpiralGalaxy_eqFunction_9732(data, threadData);
  SpiralGalaxy_eqFunction_2783(data, threadData);
  SpiralGalaxy_eqFunction_9731(data, threadData);
  SpiralGalaxy_eqFunction_2785(data, threadData);
  SpiralGalaxy_eqFunction_9744(data, threadData);
  SpiralGalaxy_eqFunction_9745(data, threadData);
  SpiralGalaxy_eqFunction_2788(data, threadData);
  SpiralGalaxy_eqFunction_2789(data, threadData);
  SpiralGalaxy_eqFunction_9746(data, threadData);
  SpiralGalaxy_eqFunction_9747(data, threadData);
  SpiralGalaxy_eqFunction_9750(data, threadData);
  SpiralGalaxy_eqFunction_9749(data, threadData);
  SpiralGalaxy_eqFunction_9748(data, threadData);
  SpiralGalaxy_eqFunction_2795(data, threadData);
  SpiralGalaxy_eqFunction_9743(data, threadData);
  SpiralGalaxy_eqFunction_2797(data, threadData);
  SpiralGalaxy_eqFunction_9742(data, threadData);
  SpiralGalaxy_eqFunction_2799(data, threadData);
  SpiralGalaxy_eqFunction_9741(data, threadData);
  SpiralGalaxy_eqFunction_2801(data, threadData);
  SpiralGalaxy_eqFunction_9754(data, threadData);
  SpiralGalaxy_eqFunction_9755(data, threadData);
  SpiralGalaxy_eqFunction_2804(data, threadData);
  SpiralGalaxy_eqFunction_2805(data, threadData);
  SpiralGalaxy_eqFunction_9756(data, threadData);
  SpiralGalaxy_eqFunction_9757(data, threadData);
  SpiralGalaxy_eqFunction_9760(data, threadData);
  SpiralGalaxy_eqFunction_9759(data, threadData);
  SpiralGalaxy_eqFunction_9758(data, threadData);
  SpiralGalaxy_eqFunction_2811(data, threadData);
  SpiralGalaxy_eqFunction_9753(data, threadData);
  SpiralGalaxy_eqFunction_2813(data, threadData);
  SpiralGalaxy_eqFunction_9752(data, threadData);
  SpiralGalaxy_eqFunction_2815(data, threadData);
  SpiralGalaxy_eqFunction_9751(data, threadData);
  SpiralGalaxy_eqFunction_2817(data, threadData);
  SpiralGalaxy_eqFunction_9764(data, threadData);
  SpiralGalaxy_eqFunction_9765(data, threadData);
  SpiralGalaxy_eqFunction_2820(data, threadData);
  SpiralGalaxy_eqFunction_2821(data, threadData);
  SpiralGalaxy_eqFunction_9766(data, threadData);
  SpiralGalaxy_eqFunction_9767(data, threadData);
  SpiralGalaxy_eqFunction_9770(data, threadData);
  SpiralGalaxy_eqFunction_9769(data, threadData);
  SpiralGalaxy_eqFunction_9768(data, threadData);
  SpiralGalaxy_eqFunction_2827(data, threadData);
  SpiralGalaxy_eqFunction_9763(data, threadData);
  SpiralGalaxy_eqFunction_2829(data, threadData);
  SpiralGalaxy_eqFunction_9762(data, threadData);
  SpiralGalaxy_eqFunction_2831(data, threadData);
  SpiralGalaxy_eqFunction_9761(data, threadData);
  SpiralGalaxy_eqFunction_2833(data, threadData);
  SpiralGalaxy_eqFunction_9774(data, threadData);
  SpiralGalaxy_eqFunction_9775(data, threadData);
  SpiralGalaxy_eqFunction_2836(data, threadData);
  SpiralGalaxy_eqFunction_2837(data, threadData);
  SpiralGalaxy_eqFunction_9776(data, threadData);
  SpiralGalaxy_eqFunction_9777(data, threadData);
  SpiralGalaxy_eqFunction_9780(data, threadData);
  SpiralGalaxy_eqFunction_9779(data, threadData);
  SpiralGalaxy_eqFunction_9778(data, threadData);
  SpiralGalaxy_eqFunction_2843(data, threadData);
  SpiralGalaxy_eqFunction_9773(data, threadData);
  SpiralGalaxy_eqFunction_2845(data, threadData);
  SpiralGalaxy_eqFunction_9772(data, threadData);
  SpiralGalaxy_eqFunction_2847(data, threadData);
  SpiralGalaxy_eqFunction_9771(data, threadData);
  SpiralGalaxy_eqFunction_2849(data, threadData);
  SpiralGalaxy_eqFunction_9784(data, threadData);
  SpiralGalaxy_eqFunction_9785(data, threadData);
  SpiralGalaxy_eqFunction_2852(data, threadData);
  SpiralGalaxy_eqFunction_2853(data, threadData);
  SpiralGalaxy_eqFunction_9786(data, threadData);
  SpiralGalaxy_eqFunction_9787(data, threadData);
  SpiralGalaxy_eqFunction_9790(data, threadData);
  SpiralGalaxy_eqFunction_9789(data, threadData);
  SpiralGalaxy_eqFunction_9788(data, threadData);
  SpiralGalaxy_eqFunction_2859(data, threadData);
  SpiralGalaxy_eqFunction_9783(data, threadData);
  SpiralGalaxy_eqFunction_2861(data, threadData);
  SpiralGalaxy_eqFunction_9782(data, threadData);
  SpiralGalaxy_eqFunction_2863(data, threadData);
  SpiralGalaxy_eqFunction_9781(data, threadData);
  SpiralGalaxy_eqFunction_2865(data, threadData);
  SpiralGalaxy_eqFunction_9794(data, threadData);
  SpiralGalaxy_eqFunction_9795(data, threadData);
  SpiralGalaxy_eqFunction_2868(data, threadData);
  SpiralGalaxy_eqFunction_2869(data, threadData);
  SpiralGalaxy_eqFunction_9796(data, threadData);
  SpiralGalaxy_eqFunction_9797(data, threadData);
  SpiralGalaxy_eqFunction_9800(data, threadData);
  SpiralGalaxy_eqFunction_9799(data, threadData);
  SpiralGalaxy_eqFunction_9798(data, threadData);
  SpiralGalaxy_eqFunction_2875(data, threadData);
  SpiralGalaxy_eqFunction_9793(data, threadData);
  SpiralGalaxy_eqFunction_2877(data, threadData);
  SpiralGalaxy_eqFunction_9792(data, threadData);
  SpiralGalaxy_eqFunction_2879(data, threadData);
  SpiralGalaxy_eqFunction_9791(data, threadData);
  SpiralGalaxy_eqFunction_2881(data, threadData);
  SpiralGalaxy_eqFunction_9804(data, threadData);
  SpiralGalaxy_eqFunction_9805(data, threadData);
  SpiralGalaxy_eqFunction_2884(data, threadData);
  SpiralGalaxy_eqFunction_2885(data, threadData);
  SpiralGalaxy_eqFunction_9806(data, threadData);
  SpiralGalaxy_eqFunction_9807(data, threadData);
  SpiralGalaxy_eqFunction_9810(data, threadData);
  SpiralGalaxy_eqFunction_9809(data, threadData);
  SpiralGalaxy_eqFunction_9808(data, threadData);
  SpiralGalaxy_eqFunction_2891(data, threadData);
  SpiralGalaxy_eqFunction_9803(data, threadData);
  SpiralGalaxy_eqFunction_2893(data, threadData);
  SpiralGalaxy_eqFunction_9802(data, threadData);
  SpiralGalaxy_eqFunction_2895(data, threadData);
  SpiralGalaxy_eqFunction_9801(data, threadData);
  SpiralGalaxy_eqFunction_2897(data, threadData);
  SpiralGalaxy_eqFunction_9814(data, threadData);
  SpiralGalaxy_eqFunction_9815(data, threadData);
  SpiralGalaxy_eqFunction_2900(data, threadData);
  SpiralGalaxy_eqFunction_2901(data, threadData);
  SpiralGalaxy_eqFunction_9816(data, threadData);
  SpiralGalaxy_eqFunction_9817(data, threadData);
  SpiralGalaxy_eqFunction_9820(data, threadData);
  SpiralGalaxy_eqFunction_9819(data, threadData);
  SpiralGalaxy_eqFunction_9818(data, threadData);
  SpiralGalaxy_eqFunction_2907(data, threadData);
  SpiralGalaxy_eqFunction_9813(data, threadData);
  SpiralGalaxy_eqFunction_2909(data, threadData);
  SpiralGalaxy_eqFunction_9812(data, threadData);
  SpiralGalaxy_eqFunction_2911(data, threadData);
  SpiralGalaxy_eqFunction_9811(data, threadData);
  SpiralGalaxy_eqFunction_2913(data, threadData);
  SpiralGalaxy_eqFunction_9824(data, threadData);
  SpiralGalaxy_eqFunction_9825(data, threadData);
  SpiralGalaxy_eqFunction_2916(data, threadData);
  SpiralGalaxy_eqFunction_2917(data, threadData);
  SpiralGalaxy_eqFunction_9826(data, threadData);
  SpiralGalaxy_eqFunction_9827(data, threadData);
  SpiralGalaxy_eqFunction_9830(data, threadData);
  SpiralGalaxy_eqFunction_9829(data, threadData);
  SpiralGalaxy_eqFunction_9828(data, threadData);
  SpiralGalaxy_eqFunction_2923(data, threadData);
  SpiralGalaxy_eqFunction_9823(data, threadData);
  SpiralGalaxy_eqFunction_2925(data, threadData);
  SpiralGalaxy_eqFunction_9822(data, threadData);
  SpiralGalaxy_eqFunction_2927(data, threadData);
  SpiralGalaxy_eqFunction_9821(data, threadData);
  SpiralGalaxy_eqFunction_2929(data, threadData);
  SpiralGalaxy_eqFunction_9834(data, threadData);
  SpiralGalaxy_eqFunction_9835(data, threadData);
  SpiralGalaxy_eqFunction_2932(data, threadData);
  SpiralGalaxy_eqFunction_2933(data, threadData);
  SpiralGalaxy_eqFunction_9836(data, threadData);
  SpiralGalaxy_eqFunction_9837(data, threadData);
  SpiralGalaxy_eqFunction_9840(data, threadData);
  SpiralGalaxy_eqFunction_9839(data, threadData);
  SpiralGalaxy_eqFunction_9838(data, threadData);
  SpiralGalaxy_eqFunction_2939(data, threadData);
  SpiralGalaxy_eqFunction_9833(data, threadData);
  SpiralGalaxy_eqFunction_2941(data, threadData);
  SpiralGalaxy_eqFunction_9832(data, threadData);
  SpiralGalaxy_eqFunction_2943(data, threadData);
  SpiralGalaxy_eqFunction_9831(data, threadData);
  SpiralGalaxy_eqFunction_2945(data, threadData);
  SpiralGalaxy_eqFunction_9844(data, threadData);
  SpiralGalaxy_eqFunction_9845(data, threadData);
  SpiralGalaxy_eqFunction_2948(data, threadData);
  SpiralGalaxy_eqFunction_2949(data, threadData);
  SpiralGalaxy_eqFunction_9846(data, threadData);
  SpiralGalaxy_eqFunction_9847(data, threadData);
  SpiralGalaxy_eqFunction_9850(data, threadData);
  SpiralGalaxy_eqFunction_9849(data, threadData);
  SpiralGalaxy_eqFunction_9848(data, threadData);
  SpiralGalaxy_eqFunction_2955(data, threadData);
  SpiralGalaxy_eqFunction_9843(data, threadData);
  SpiralGalaxy_eqFunction_2957(data, threadData);
  SpiralGalaxy_eqFunction_9842(data, threadData);
  SpiralGalaxy_eqFunction_2959(data, threadData);
  SpiralGalaxy_eqFunction_9841(data, threadData);
  SpiralGalaxy_eqFunction_2961(data, threadData);
  SpiralGalaxy_eqFunction_9854(data, threadData);
  SpiralGalaxy_eqFunction_9855(data, threadData);
  SpiralGalaxy_eqFunction_2964(data, threadData);
  SpiralGalaxy_eqFunction_2965(data, threadData);
  SpiralGalaxy_eqFunction_9856(data, threadData);
  SpiralGalaxy_eqFunction_9857(data, threadData);
  SpiralGalaxy_eqFunction_9860(data, threadData);
  SpiralGalaxy_eqFunction_9859(data, threadData);
  SpiralGalaxy_eqFunction_9858(data, threadData);
  SpiralGalaxy_eqFunction_2971(data, threadData);
  SpiralGalaxy_eqFunction_9853(data, threadData);
  SpiralGalaxy_eqFunction_2973(data, threadData);
  SpiralGalaxy_eqFunction_9852(data, threadData);
  SpiralGalaxy_eqFunction_2975(data, threadData);
  SpiralGalaxy_eqFunction_9851(data, threadData);
  SpiralGalaxy_eqFunction_2977(data, threadData);
  SpiralGalaxy_eqFunction_9864(data, threadData);
  SpiralGalaxy_eqFunction_9865(data, threadData);
  SpiralGalaxy_eqFunction_2980(data, threadData);
  SpiralGalaxy_eqFunction_2981(data, threadData);
  SpiralGalaxy_eqFunction_9866(data, threadData);
  SpiralGalaxy_eqFunction_9867(data, threadData);
  SpiralGalaxy_eqFunction_9870(data, threadData);
  SpiralGalaxy_eqFunction_9869(data, threadData);
  SpiralGalaxy_eqFunction_9868(data, threadData);
  SpiralGalaxy_eqFunction_2987(data, threadData);
  SpiralGalaxy_eqFunction_9863(data, threadData);
  SpiralGalaxy_eqFunction_2989(data, threadData);
  SpiralGalaxy_eqFunction_9862(data, threadData);
  SpiralGalaxy_eqFunction_2991(data, threadData);
  SpiralGalaxy_eqFunction_9861(data, threadData);
  SpiralGalaxy_eqFunction_2993(data, threadData);
  SpiralGalaxy_eqFunction_9874(data, threadData);
  SpiralGalaxy_eqFunction_9875(data, threadData);
  SpiralGalaxy_eqFunction_2996(data, threadData);
  SpiralGalaxy_eqFunction_2997(data, threadData);
  SpiralGalaxy_eqFunction_9876(data, threadData);
  SpiralGalaxy_eqFunction_9877(data, threadData);
  SpiralGalaxy_eqFunction_9880(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif