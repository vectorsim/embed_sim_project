#include "SpiralGalaxy_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 501
type: SIMPLE_ASSIGN
x[32] = r_init[32] * cos(theta[32] - 0.00872)
*/
void SpiralGalaxy_eqFunction_501(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,501};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1531]] /* x[32] STATE(1,vx[32]) */) = ((data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */)) * (cos((data->simulationInfo->realParameter[1538] /* theta[32] PARAM */) - 0.00872));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8316(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8317(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8320(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8319(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8318(DATA *data, threadData_t *threadData);


/*
equation index: 507
type: SIMPLE_ASSIGN
vx[32] = (-sin(theta[32])) * r_init[32] * omega_c[32]
*/
void SpiralGalaxy_eqFunction_507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,507};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* vx[32] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1538] /* theta[32] PARAM */)))) * (((data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */)) * ((data->simulationInfo->realParameter[536] /* omega_c[32] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8313(DATA *data, threadData_t *threadData);


/*
equation index: 509
type: SIMPLE_ASSIGN
vy[32] = cos(theta[32]) * r_init[32] * omega_c[32]
*/
void SpiralGalaxy_eqFunction_509(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,509};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[531]] /* vy[32] STATE(1) */) = (cos((data->simulationInfo->realParameter[1538] /* theta[32] PARAM */))) * (((data->simulationInfo->realParameter[1037] /* r_init[32] PARAM */)) * ((data->simulationInfo->realParameter[536] /* omega_c[32] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8312(DATA *data, threadData_t *threadData);


/*
equation index: 511
type: SIMPLE_ASSIGN
vz[32] = 0.0
*/
void SpiralGalaxy_eqFunction_511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,511};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1031]] /* vz[32] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8311(DATA *data, threadData_t *threadData);


/*
equation index: 513
type: SIMPLE_ASSIGN
z[33] = -0.03472
*/
void SpiralGalaxy_eqFunction_513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,513};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2532]] /* z[33] STATE(1,vz[33]) */) = -0.03472;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8324(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8325(DATA *data, threadData_t *threadData);


/*
equation index: 516
type: SIMPLE_ASSIGN
y[33] = r_init[33] * sin(theta[33] - 0.00868)
*/
void SpiralGalaxy_eqFunction_516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,516};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2032]] /* y[33] STATE(1,vy[33]) */) = ((data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */)) * (sin((data->simulationInfo->realParameter[1539] /* theta[33] PARAM */) - 0.00868));
  TRACE_POP
}

/*
equation index: 517
type: SIMPLE_ASSIGN
x[33] = r_init[33] * cos(theta[33] - 0.00868)
*/
void SpiralGalaxy_eqFunction_517(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,517};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1532]] /* x[33] STATE(1,vx[33]) */) = ((data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */)) * (cos((data->simulationInfo->realParameter[1539] /* theta[33] PARAM */) - 0.00868));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8326(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8327(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8330(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8329(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8328(DATA *data, threadData_t *threadData);


/*
equation index: 523
type: SIMPLE_ASSIGN
vx[33] = (-sin(theta[33])) * r_init[33] * omega_c[33]
*/
void SpiralGalaxy_eqFunction_523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,523};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* vx[33] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1539] /* theta[33] PARAM */)))) * (((data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */)) * ((data->simulationInfo->realParameter[537] /* omega_c[33] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8323(DATA *data, threadData_t *threadData);


/*
equation index: 525
type: SIMPLE_ASSIGN
vy[33] = cos(theta[33]) * r_init[33] * omega_c[33]
*/
void SpiralGalaxy_eqFunction_525(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,525};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[532]] /* vy[33] STATE(1) */) = (cos((data->simulationInfo->realParameter[1539] /* theta[33] PARAM */))) * (((data->simulationInfo->realParameter[1038] /* r_init[33] PARAM */)) * ((data->simulationInfo->realParameter[537] /* omega_c[33] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8322(DATA *data, threadData_t *threadData);


/*
equation index: 527
type: SIMPLE_ASSIGN
vz[33] = 0.0
*/
void SpiralGalaxy_eqFunction_527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1032]] /* vz[33] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8321(DATA *data, threadData_t *threadData);


/*
equation index: 529
type: SIMPLE_ASSIGN
z[34] = -0.03456
*/
void SpiralGalaxy_eqFunction_529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,529};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2533]] /* z[34] STATE(1,vz[34]) */) = -0.03456;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8334(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8335(DATA *data, threadData_t *threadData);


/*
equation index: 532
type: SIMPLE_ASSIGN
y[34] = r_init[34] * sin(theta[34] - 0.00864)
*/
void SpiralGalaxy_eqFunction_532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,532};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2033]] /* y[34] STATE(1,vy[34]) */) = ((data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */)) * (sin((data->simulationInfo->realParameter[1540] /* theta[34] PARAM */) - 0.00864));
  TRACE_POP
}

/*
equation index: 533
type: SIMPLE_ASSIGN
x[34] = r_init[34] * cos(theta[34] - 0.00864)
*/
void SpiralGalaxy_eqFunction_533(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,533};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1533]] /* x[34] STATE(1,vx[34]) */) = ((data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */)) * (cos((data->simulationInfo->realParameter[1540] /* theta[34] PARAM */) - 0.00864));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8336(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8337(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8340(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8339(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8338(DATA *data, threadData_t *threadData);


/*
equation index: 539
type: SIMPLE_ASSIGN
vx[34] = (-sin(theta[34])) * r_init[34] * omega_c[34]
*/
void SpiralGalaxy_eqFunction_539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,539};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* vx[34] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1540] /* theta[34] PARAM */)))) * (((data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */)) * ((data->simulationInfo->realParameter[538] /* omega_c[34] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8333(DATA *data, threadData_t *threadData);


/*
equation index: 541
type: SIMPLE_ASSIGN
vy[34] = cos(theta[34]) * r_init[34] * omega_c[34]
*/
void SpiralGalaxy_eqFunction_541(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,541};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[533]] /* vy[34] STATE(1) */) = (cos((data->simulationInfo->realParameter[1540] /* theta[34] PARAM */))) * (((data->simulationInfo->realParameter[1039] /* r_init[34] PARAM */)) * ((data->simulationInfo->realParameter[538] /* omega_c[34] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8332(DATA *data, threadData_t *threadData);


/*
equation index: 543
type: SIMPLE_ASSIGN
vz[34] = 0.0
*/
void SpiralGalaxy_eqFunction_543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,543};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1033]] /* vz[34] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8331(DATA *data, threadData_t *threadData);


/*
equation index: 545
type: SIMPLE_ASSIGN
z[35] = -0.0344
*/
void SpiralGalaxy_eqFunction_545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,545};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2534]] /* z[35] STATE(1,vz[35]) */) = -0.0344;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8344(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8345(DATA *data, threadData_t *threadData);


/*
equation index: 548
type: SIMPLE_ASSIGN
y[35] = r_init[35] * sin(theta[35] - 0.0086)
*/
void SpiralGalaxy_eqFunction_548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,548};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2034]] /* y[35] STATE(1,vy[35]) */) = ((data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */)) * (sin((data->simulationInfo->realParameter[1541] /* theta[35] PARAM */) - 0.0086));
  TRACE_POP
}

/*
equation index: 549
type: SIMPLE_ASSIGN
x[35] = r_init[35] * cos(theta[35] - 0.0086)
*/
void SpiralGalaxy_eqFunction_549(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,549};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1534]] /* x[35] STATE(1,vx[35]) */) = ((data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */)) * (cos((data->simulationInfo->realParameter[1541] /* theta[35] PARAM */) - 0.0086));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8346(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8347(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8350(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8349(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8348(DATA *data, threadData_t *threadData);


/*
equation index: 555
type: SIMPLE_ASSIGN
vx[35] = (-sin(theta[35])) * r_init[35] * omega_c[35]
*/
void SpiralGalaxy_eqFunction_555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,555};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* vx[35] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1541] /* theta[35] PARAM */)))) * (((data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */)) * ((data->simulationInfo->realParameter[539] /* omega_c[35] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8343(DATA *data, threadData_t *threadData);


/*
equation index: 557
type: SIMPLE_ASSIGN
vy[35] = cos(theta[35]) * r_init[35] * omega_c[35]
*/
void SpiralGalaxy_eqFunction_557(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,557};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[534]] /* vy[35] STATE(1) */) = (cos((data->simulationInfo->realParameter[1541] /* theta[35] PARAM */))) * (((data->simulationInfo->realParameter[1040] /* r_init[35] PARAM */)) * ((data->simulationInfo->realParameter[539] /* omega_c[35] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8342(DATA *data, threadData_t *threadData);


/*
equation index: 559
type: SIMPLE_ASSIGN
vz[35] = 0.0
*/
void SpiralGalaxy_eqFunction_559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,559};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1034]] /* vz[35] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8341(DATA *data, threadData_t *threadData);


/*
equation index: 561
type: SIMPLE_ASSIGN
z[36] = -0.03424000000000001
*/
void SpiralGalaxy_eqFunction_561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,561};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2535]] /* z[36] STATE(1,vz[36]) */) = -0.03424000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8354(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8355(DATA *data, threadData_t *threadData);


/*
equation index: 564
type: SIMPLE_ASSIGN
y[36] = r_init[36] * sin(theta[36] - 0.00856)
*/
void SpiralGalaxy_eqFunction_564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,564};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2035]] /* y[36] STATE(1,vy[36]) */) = ((data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */)) * (sin((data->simulationInfo->realParameter[1542] /* theta[36] PARAM */) - 0.00856));
  TRACE_POP
}

/*
equation index: 565
type: SIMPLE_ASSIGN
x[36] = r_init[36] * cos(theta[36] - 0.00856)
*/
void SpiralGalaxy_eqFunction_565(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,565};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1535]] /* x[36] STATE(1,vx[36]) */) = ((data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */)) * (cos((data->simulationInfo->realParameter[1542] /* theta[36] PARAM */) - 0.00856));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8356(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8357(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8360(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8359(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8358(DATA *data, threadData_t *threadData);


/*
equation index: 571
type: SIMPLE_ASSIGN
vx[36] = (-sin(theta[36])) * r_init[36] * omega_c[36]
*/
void SpiralGalaxy_eqFunction_571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,571};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* vx[36] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1542] /* theta[36] PARAM */)))) * (((data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */)) * ((data->simulationInfo->realParameter[540] /* omega_c[36] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8353(DATA *data, threadData_t *threadData);


/*
equation index: 573
type: SIMPLE_ASSIGN
vy[36] = cos(theta[36]) * r_init[36] * omega_c[36]
*/
void SpiralGalaxy_eqFunction_573(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,573};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[535]] /* vy[36] STATE(1) */) = (cos((data->simulationInfo->realParameter[1542] /* theta[36] PARAM */))) * (((data->simulationInfo->realParameter[1041] /* r_init[36] PARAM */)) * ((data->simulationInfo->realParameter[540] /* omega_c[36] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8352(DATA *data, threadData_t *threadData);


/*
equation index: 575
type: SIMPLE_ASSIGN
vz[36] = 0.0
*/
void SpiralGalaxy_eqFunction_575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,575};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1035]] /* vz[36] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8351(DATA *data, threadData_t *threadData);


/*
equation index: 577
type: SIMPLE_ASSIGN
z[37] = -0.03408
*/
void SpiralGalaxy_eqFunction_577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,577};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2536]] /* z[37] STATE(1,vz[37]) */) = -0.03408;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8364(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8365(DATA *data, threadData_t *threadData);


/*
equation index: 580
type: SIMPLE_ASSIGN
y[37] = r_init[37] * sin(theta[37] - 0.00852)
*/
void SpiralGalaxy_eqFunction_580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,580};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2036]] /* y[37] STATE(1,vy[37]) */) = ((data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */)) * (sin((data->simulationInfo->realParameter[1543] /* theta[37] PARAM */) - 0.00852));
  TRACE_POP
}

/*
equation index: 581
type: SIMPLE_ASSIGN
x[37] = r_init[37] * cos(theta[37] - 0.00852)
*/
void SpiralGalaxy_eqFunction_581(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,581};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1536]] /* x[37] STATE(1,vx[37]) */) = ((data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */)) * (cos((data->simulationInfo->realParameter[1543] /* theta[37] PARAM */) - 0.00852));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8366(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8367(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8370(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8369(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8368(DATA *data, threadData_t *threadData);


/*
equation index: 587
type: SIMPLE_ASSIGN
vx[37] = (-sin(theta[37])) * r_init[37] * omega_c[37]
*/
void SpiralGalaxy_eqFunction_587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,587};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* vx[37] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1543] /* theta[37] PARAM */)))) * (((data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */)) * ((data->simulationInfo->realParameter[541] /* omega_c[37] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8363(DATA *data, threadData_t *threadData);


/*
equation index: 589
type: SIMPLE_ASSIGN
vy[37] = cos(theta[37]) * r_init[37] * omega_c[37]
*/
void SpiralGalaxy_eqFunction_589(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,589};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[536]] /* vy[37] STATE(1) */) = (cos((data->simulationInfo->realParameter[1543] /* theta[37] PARAM */))) * (((data->simulationInfo->realParameter[1042] /* r_init[37] PARAM */)) * ((data->simulationInfo->realParameter[541] /* omega_c[37] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8362(DATA *data, threadData_t *threadData);


/*
equation index: 591
type: SIMPLE_ASSIGN
vz[37] = 0.0
*/
void SpiralGalaxy_eqFunction_591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,591};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1036]] /* vz[37] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8361(DATA *data, threadData_t *threadData);


/*
equation index: 593
type: SIMPLE_ASSIGN
z[38] = -0.033920000000000006
*/
void SpiralGalaxy_eqFunction_593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,593};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2537]] /* z[38] STATE(1,vz[38]) */) = -0.033920000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8374(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8375(DATA *data, threadData_t *threadData);


/*
equation index: 596
type: SIMPLE_ASSIGN
y[38] = r_init[38] * sin(theta[38] - 0.00848)
*/
void SpiralGalaxy_eqFunction_596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,596};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2037]] /* y[38] STATE(1,vy[38]) */) = ((data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */)) * (sin((data->simulationInfo->realParameter[1544] /* theta[38] PARAM */) - 0.00848));
  TRACE_POP
}

/*
equation index: 597
type: SIMPLE_ASSIGN
x[38] = r_init[38] * cos(theta[38] - 0.00848)
*/
void SpiralGalaxy_eqFunction_597(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,597};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1537]] /* x[38] STATE(1,vx[38]) */) = ((data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */)) * (cos((data->simulationInfo->realParameter[1544] /* theta[38] PARAM */) - 0.00848));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8376(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8377(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8380(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8379(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8378(DATA *data, threadData_t *threadData);


/*
equation index: 603
type: SIMPLE_ASSIGN
vx[38] = (-sin(theta[38])) * r_init[38] * omega_c[38]
*/
void SpiralGalaxy_eqFunction_603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,603};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* vx[38] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1544] /* theta[38] PARAM */)))) * (((data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */)) * ((data->simulationInfo->realParameter[542] /* omega_c[38] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8373(DATA *data, threadData_t *threadData);


/*
equation index: 605
type: SIMPLE_ASSIGN
vy[38] = cos(theta[38]) * r_init[38] * omega_c[38]
*/
void SpiralGalaxy_eqFunction_605(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,605};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[537]] /* vy[38] STATE(1) */) = (cos((data->simulationInfo->realParameter[1544] /* theta[38] PARAM */))) * (((data->simulationInfo->realParameter[1043] /* r_init[38] PARAM */)) * ((data->simulationInfo->realParameter[542] /* omega_c[38] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8372(DATA *data, threadData_t *threadData);


/*
equation index: 607
type: SIMPLE_ASSIGN
vz[38] = 0.0
*/
void SpiralGalaxy_eqFunction_607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,607};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1037]] /* vz[38] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8371(DATA *data, threadData_t *threadData);


/*
equation index: 609
type: SIMPLE_ASSIGN
z[39] = -0.033760000000000005
*/
void SpiralGalaxy_eqFunction_609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,609};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2538]] /* z[39] STATE(1,vz[39]) */) = -0.033760000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8384(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8385(DATA *data, threadData_t *threadData);


/*
equation index: 612
type: SIMPLE_ASSIGN
y[39] = r_init[39] * sin(theta[39] - 0.00844)
*/
void SpiralGalaxy_eqFunction_612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,612};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2038]] /* y[39] STATE(1,vy[39]) */) = ((data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */)) * (sin((data->simulationInfo->realParameter[1545] /* theta[39] PARAM */) - 0.00844));
  TRACE_POP
}

/*
equation index: 613
type: SIMPLE_ASSIGN
x[39] = r_init[39] * cos(theta[39] - 0.00844)
*/
void SpiralGalaxy_eqFunction_613(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,613};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1538]] /* x[39] STATE(1,vx[39]) */) = ((data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */)) * (cos((data->simulationInfo->realParameter[1545] /* theta[39] PARAM */) - 0.00844));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8386(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8387(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8390(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8389(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8388(DATA *data, threadData_t *threadData);


/*
equation index: 619
type: SIMPLE_ASSIGN
vx[39] = (-sin(theta[39])) * r_init[39] * omega_c[39]
*/
void SpiralGalaxy_eqFunction_619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,619};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* vx[39] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1545] /* theta[39] PARAM */)))) * (((data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */)) * ((data->simulationInfo->realParameter[543] /* omega_c[39] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8383(DATA *data, threadData_t *threadData);


/*
equation index: 621
type: SIMPLE_ASSIGN
vy[39] = cos(theta[39]) * r_init[39] * omega_c[39]
*/
void SpiralGalaxy_eqFunction_621(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,621};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[538]] /* vy[39] STATE(1) */) = (cos((data->simulationInfo->realParameter[1545] /* theta[39] PARAM */))) * (((data->simulationInfo->realParameter[1044] /* r_init[39] PARAM */)) * ((data->simulationInfo->realParameter[543] /* omega_c[39] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8382(DATA *data, threadData_t *threadData);


/*
equation index: 623
type: SIMPLE_ASSIGN
vz[39] = 0.0
*/
void SpiralGalaxy_eqFunction_623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,623};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1038]] /* vz[39] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8381(DATA *data, threadData_t *threadData);


/*
equation index: 625
type: SIMPLE_ASSIGN
z[40] = -0.033600000000000005
*/
void SpiralGalaxy_eqFunction_625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,625};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2539]] /* z[40] STATE(1,vz[40]) */) = -0.033600000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8394(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8395(DATA *data, threadData_t *threadData);


/*
equation index: 628
type: SIMPLE_ASSIGN
y[40] = r_init[40] * sin(theta[40] - 0.0084)
*/
void SpiralGalaxy_eqFunction_628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,628};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2039]] /* y[40] STATE(1,vy[40]) */) = ((data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */)) * (sin((data->simulationInfo->realParameter[1546] /* theta[40] PARAM */) - 0.0084));
  TRACE_POP
}

/*
equation index: 629
type: SIMPLE_ASSIGN
x[40] = r_init[40] * cos(theta[40] - 0.0084)
*/
void SpiralGalaxy_eqFunction_629(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,629};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1539]] /* x[40] STATE(1,vx[40]) */) = ((data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */)) * (cos((data->simulationInfo->realParameter[1546] /* theta[40] PARAM */) - 0.0084));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8396(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8397(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8400(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8399(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8398(DATA *data, threadData_t *threadData);


/*
equation index: 635
type: SIMPLE_ASSIGN
vx[40] = (-sin(theta[40])) * r_init[40] * omega_c[40]
*/
void SpiralGalaxy_eqFunction_635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,635};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* vx[40] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1546] /* theta[40] PARAM */)))) * (((data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */)) * ((data->simulationInfo->realParameter[544] /* omega_c[40] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8393(DATA *data, threadData_t *threadData);


/*
equation index: 637
type: SIMPLE_ASSIGN
vy[40] = cos(theta[40]) * r_init[40] * omega_c[40]
*/
void SpiralGalaxy_eqFunction_637(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,637};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[539]] /* vy[40] STATE(1) */) = (cos((data->simulationInfo->realParameter[1546] /* theta[40] PARAM */))) * (((data->simulationInfo->realParameter[1045] /* r_init[40] PARAM */)) * ((data->simulationInfo->realParameter[544] /* omega_c[40] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8392(DATA *data, threadData_t *threadData);


/*
equation index: 639
type: SIMPLE_ASSIGN
vz[40] = 0.0
*/
void SpiralGalaxy_eqFunction_639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,639};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1039]] /* vz[40] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8391(DATA *data, threadData_t *threadData);


/*
equation index: 641
type: SIMPLE_ASSIGN
z[41] = -0.03344
*/
void SpiralGalaxy_eqFunction_641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,641};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2540]] /* z[41] STATE(1,vz[41]) */) = -0.03344;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8404(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8405(DATA *data, threadData_t *threadData);


/*
equation index: 644
type: SIMPLE_ASSIGN
y[41] = r_init[41] * sin(theta[41] - 0.00836)
*/
void SpiralGalaxy_eqFunction_644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,644};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2040]] /* y[41] STATE(1,vy[41]) */) = ((data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */)) * (sin((data->simulationInfo->realParameter[1547] /* theta[41] PARAM */) - 0.00836));
  TRACE_POP
}

/*
equation index: 645
type: SIMPLE_ASSIGN
x[41] = r_init[41] * cos(theta[41] - 0.00836)
*/
void SpiralGalaxy_eqFunction_645(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,645};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1540]] /* x[41] STATE(1,vx[41]) */) = ((data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */)) * (cos((data->simulationInfo->realParameter[1547] /* theta[41] PARAM */) - 0.00836));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8406(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8407(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8410(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8409(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8408(DATA *data, threadData_t *threadData);


/*
equation index: 651
type: SIMPLE_ASSIGN
vx[41] = (-sin(theta[41])) * r_init[41] * omega_c[41]
*/
void SpiralGalaxy_eqFunction_651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,651};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* vx[41] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1547] /* theta[41] PARAM */)))) * (((data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */)) * ((data->simulationInfo->realParameter[545] /* omega_c[41] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8403(DATA *data, threadData_t *threadData);


/*
equation index: 653
type: SIMPLE_ASSIGN
vy[41] = cos(theta[41]) * r_init[41] * omega_c[41]
*/
void SpiralGalaxy_eqFunction_653(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,653};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[540]] /* vy[41] STATE(1) */) = (cos((data->simulationInfo->realParameter[1547] /* theta[41] PARAM */))) * (((data->simulationInfo->realParameter[1046] /* r_init[41] PARAM */)) * ((data->simulationInfo->realParameter[545] /* omega_c[41] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8402(DATA *data, threadData_t *threadData);


/*
equation index: 655
type: SIMPLE_ASSIGN
vz[41] = 0.0
*/
void SpiralGalaxy_eqFunction_655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,655};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1040]] /* vz[41] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8401(DATA *data, threadData_t *threadData);


/*
equation index: 657
type: SIMPLE_ASSIGN
z[42] = -0.033280000000000004
*/
void SpiralGalaxy_eqFunction_657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,657};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2541]] /* z[42] STATE(1,vz[42]) */) = -0.033280000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8414(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8415(DATA *data, threadData_t *threadData);


/*
equation index: 660
type: SIMPLE_ASSIGN
y[42] = r_init[42] * sin(theta[42] - 0.00832)
*/
void SpiralGalaxy_eqFunction_660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,660};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2041]] /* y[42] STATE(1,vy[42]) */) = ((data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */)) * (sin((data->simulationInfo->realParameter[1548] /* theta[42] PARAM */) - 0.00832));
  TRACE_POP
}

/*
equation index: 661
type: SIMPLE_ASSIGN
x[42] = r_init[42] * cos(theta[42] - 0.00832)
*/
void SpiralGalaxy_eqFunction_661(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,661};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1541]] /* x[42] STATE(1,vx[42]) */) = ((data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */)) * (cos((data->simulationInfo->realParameter[1548] /* theta[42] PARAM */) - 0.00832));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8416(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8417(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8420(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8419(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8418(DATA *data, threadData_t *threadData);


/*
equation index: 667
type: SIMPLE_ASSIGN
vx[42] = (-sin(theta[42])) * r_init[42] * omega_c[42]
*/
void SpiralGalaxy_eqFunction_667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,667};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* vx[42] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1548] /* theta[42] PARAM */)))) * (((data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */)) * ((data->simulationInfo->realParameter[546] /* omega_c[42] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8413(DATA *data, threadData_t *threadData);


/*
equation index: 669
type: SIMPLE_ASSIGN
vy[42] = cos(theta[42]) * r_init[42] * omega_c[42]
*/
void SpiralGalaxy_eqFunction_669(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,669};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[541]] /* vy[42] STATE(1) */) = (cos((data->simulationInfo->realParameter[1548] /* theta[42] PARAM */))) * (((data->simulationInfo->realParameter[1047] /* r_init[42] PARAM */)) * ((data->simulationInfo->realParameter[546] /* omega_c[42] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8412(DATA *data, threadData_t *threadData);


/*
equation index: 671
type: SIMPLE_ASSIGN
vz[42] = 0.0
*/
void SpiralGalaxy_eqFunction_671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,671};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1041]] /* vz[42] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8411(DATA *data, threadData_t *threadData);


/*
equation index: 673
type: SIMPLE_ASSIGN
z[43] = -0.033120000000000004
*/
void SpiralGalaxy_eqFunction_673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,673};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2542]] /* z[43] STATE(1,vz[43]) */) = -0.033120000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8424(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8425(DATA *data, threadData_t *threadData);


/*
equation index: 676
type: SIMPLE_ASSIGN
y[43] = r_init[43] * sin(theta[43] - 0.008280000000000001)
*/
void SpiralGalaxy_eqFunction_676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,676};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2042]] /* y[43] STATE(1,vy[43]) */) = ((data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */)) * (sin((data->simulationInfo->realParameter[1549] /* theta[43] PARAM */) - 0.008280000000000001));
  TRACE_POP
}

/*
equation index: 677
type: SIMPLE_ASSIGN
x[43] = r_init[43] * cos(theta[43] - 0.008280000000000001)
*/
void SpiralGalaxy_eqFunction_677(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,677};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1542]] /* x[43] STATE(1,vx[43]) */) = ((data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */)) * (cos((data->simulationInfo->realParameter[1549] /* theta[43] PARAM */) - 0.008280000000000001));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8426(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8427(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8430(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8429(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8428(DATA *data, threadData_t *threadData);


/*
equation index: 683
type: SIMPLE_ASSIGN
vx[43] = (-sin(theta[43])) * r_init[43] * omega_c[43]
*/
void SpiralGalaxy_eqFunction_683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,683};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* vx[43] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1549] /* theta[43] PARAM */)))) * (((data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */)) * ((data->simulationInfo->realParameter[547] /* omega_c[43] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8423(DATA *data, threadData_t *threadData);


/*
equation index: 685
type: SIMPLE_ASSIGN
vy[43] = cos(theta[43]) * r_init[43] * omega_c[43]
*/
void SpiralGalaxy_eqFunction_685(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,685};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[542]] /* vy[43] STATE(1) */) = (cos((data->simulationInfo->realParameter[1549] /* theta[43] PARAM */))) * (((data->simulationInfo->realParameter[1048] /* r_init[43] PARAM */)) * ((data->simulationInfo->realParameter[547] /* omega_c[43] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8422(DATA *data, threadData_t *threadData);


/*
equation index: 687
type: SIMPLE_ASSIGN
vz[43] = 0.0
*/
void SpiralGalaxy_eqFunction_687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,687};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1042]] /* vz[43] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8421(DATA *data, threadData_t *threadData);


/*
equation index: 689
type: SIMPLE_ASSIGN
z[44] = -0.03296
*/
void SpiralGalaxy_eqFunction_689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,689};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2543]] /* z[44] STATE(1,vz[44]) */) = -0.03296;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8434(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8435(DATA *data, threadData_t *threadData);


/*
equation index: 692
type: SIMPLE_ASSIGN
y[44] = r_init[44] * sin(theta[44] - 0.00824)
*/
void SpiralGalaxy_eqFunction_692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,692};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2043]] /* y[44] STATE(1,vy[44]) */) = ((data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */)) * (sin((data->simulationInfo->realParameter[1550] /* theta[44] PARAM */) - 0.00824));
  TRACE_POP
}

/*
equation index: 693
type: SIMPLE_ASSIGN
x[44] = r_init[44] * cos(theta[44] - 0.00824)
*/
void SpiralGalaxy_eqFunction_693(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,693};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1543]] /* x[44] STATE(1,vx[44]) */) = ((data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */)) * (cos((data->simulationInfo->realParameter[1550] /* theta[44] PARAM */) - 0.00824));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8436(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8437(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8440(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8439(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8438(DATA *data, threadData_t *threadData);


/*
equation index: 699
type: SIMPLE_ASSIGN
vx[44] = (-sin(theta[44])) * r_init[44] * omega_c[44]
*/
void SpiralGalaxy_eqFunction_699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,699};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[43]] /* vx[44] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1550] /* theta[44] PARAM */)))) * (((data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */)) * ((data->simulationInfo->realParameter[548] /* omega_c[44] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8433(DATA *data, threadData_t *threadData);


/*
equation index: 701
type: SIMPLE_ASSIGN
vy[44] = cos(theta[44]) * r_init[44] * omega_c[44]
*/
void SpiralGalaxy_eqFunction_701(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,701};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[543]] /* vy[44] STATE(1) */) = (cos((data->simulationInfo->realParameter[1550] /* theta[44] PARAM */))) * (((data->simulationInfo->realParameter[1049] /* r_init[44] PARAM */)) * ((data->simulationInfo->realParameter[548] /* omega_c[44] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8432(DATA *data, threadData_t *threadData);


/*
equation index: 703
type: SIMPLE_ASSIGN
vz[44] = 0.0
*/
void SpiralGalaxy_eqFunction_703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,703};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1043]] /* vz[44] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8431(DATA *data, threadData_t *threadData);


/*
equation index: 705
type: SIMPLE_ASSIGN
z[45] = -0.0328
*/
void SpiralGalaxy_eqFunction_705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,705};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2544]] /* z[45] STATE(1,vz[45]) */) = -0.0328;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8444(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8445(DATA *data, threadData_t *threadData);


/*
equation index: 708
type: SIMPLE_ASSIGN
y[45] = r_init[45] * sin(theta[45] - 0.0082)
*/
void SpiralGalaxy_eqFunction_708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,708};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2044]] /* y[45] STATE(1,vy[45]) */) = ((data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */)) * (sin((data->simulationInfo->realParameter[1551] /* theta[45] PARAM */) - 0.0082));
  TRACE_POP
}

/*
equation index: 709
type: SIMPLE_ASSIGN
x[45] = r_init[45] * cos(theta[45] - 0.0082)
*/
void SpiralGalaxy_eqFunction_709(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,709};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1544]] /* x[45] STATE(1,vx[45]) */) = ((data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */)) * (cos((data->simulationInfo->realParameter[1551] /* theta[45] PARAM */) - 0.0082));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8446(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8447(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8450(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8449(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8448(DATA *data, threadData_t *threadData);


/*
equation index: 715
type: SIMPLE_ASSIGN
vx[45] = (-sin(theta[45])) * r_init[45] * omega_c[45]
*/
void SpiralGalaxy_eqFunction_715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,715};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[44]] /* vx[45] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1551] /* theta[45] PARAM */)))) * (((data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */)) * ((data->simulationInfo->realParameter[549] /* omega_c[45] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8443(DATA *data, threadData_t *threadData);


/*
equation index: 717
type: SIMPLE_ASSIGN
vy[45] = cos(theta[45]) * r_init[45] * omega_c[45]
*/
void SpiralGalaxy_eqFunction_717(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,717};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[544]] /* vy[45] STATE(1) */) = (cos((data->simulationInfo->realParameter[1551] /* theta[45] PARAM */))) * (((data->simulationInfo->realParameter[1050] /* r_init[45] PARAM */)) * ((data->simulationInfo->realParameter[549] /* omega_c[45] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8442(DATA *data, threadData_t *threadData);


/*
equation index: 719
type: SIMPLE_ASSIGN
vz[45] = 0.0
*/
void SpiralGalaxy_eqFunction_719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,719};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1044]] /* vz[45] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8441(DATA *data, threadData_t *threadData);


/*
equation index: 721
type: SIMPLE_ASSIGN
z[46] = -0.03264
*/
void SpiralGalaxy_eqFunction_721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,721};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2545]] /* z[46] STATE(1,vz[46]) */) = -0.03264;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8454(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8455(DATA *data, threadData_t *threadData);


/*
equation index: 724
type: SIMPLE_ASSIGN
y[46] = r_init[46] * sin(theta[46] - 0.00816)
*/
void SpiralGalaxy_eqFunction_724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,724};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2045]] /* y[46] STATE(1,vy[46]) */) = ((data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */)) * (sin((data->simulationInfo->realParameter[1552] /* theta[46] PARAM */) - 0.00816));
  TRACE_POP
}

/*
equation index: 725
type: SIMPLE_ASSIGN
x[46] = r_init[46] * cos(theta[46] - 0.00816)
*/
void SpiralGalaxy_eqFunction_725(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,725};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1545]] /* x[46] STATE(1,vx[46]) */) = ((data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */)) * (cos((data->simulationInfo->realParameter[1552] /* theta[46] PARAM */) - 0.00816));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8456(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8457(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8460(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8459(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8458(DATA *data, threadData_t *threadData);


/*
equation index: 731
type: SIMPLE_ASSIGN
vx[46] = (-sin(theta[46])) * r_init[46] * omega_c[46]
*/
void SpiralGalaxy_eqFunction_731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,731};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* vx[46] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1552] /* theta[46] PARAM */)))) * (((data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */)) * ((data->simulationInfo->realParameter[550] /* omega_c[46] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8453(DATA *data, threadData_t *threadData);


/*
equation index: 733
type: SIMPLE_ASSIGN
vy[46] = cos(theta[46]) * r_init[46] * omega_c[46]
*/
void SpiralGalaxy_eqFunction_733(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,733};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[545]] /* vy[46] STATE(1) */) = (cos((data->simulationInfo->realParameter[1552] /* theta[46] PARAM */))) * (((data->simulationInfo->realParameter[1051] /* r_init[46] PARAM */)) * ((data->simulationInfo->realParameter[550] /* omega_c[46] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8452(DATA *data, threadData_t *threadData);


/*
equation index: 735
type: SIMPLE_ASSIGN
vz[46] = 0.0
*/
void SpiralGalaxy_eqFunction_735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,735};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1045]] /* vz[46] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8451(DATA *data, threadData_t *threadData);


/*
equation index: 737
type: SIMPLE_ASSIGN
z[47] = -0.03248
*/
void SpiralGalaxy_eqFunction_737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,737};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2546]] /* z[47] STATE(1,vz[47]) */) = -0.03248;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8464(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8465(DATA *data, threadData_t *threadData);


/*
equation index: 740
type: SIMPLE_ASSIGN
y[47] = r_init[47] * sin(theta[47] - 0.00812)
*/
void SpiralGalaxy_eqFunction_740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,740};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2046]] /* y[47] STATE(1,vy[47]) */) = ((data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */)) * (sin((data->simulationInfo->realParameter[1553] /* theta[47] PARAM */) - 0.00812));
  TRACE_POP
}

/*
equation index: 741
type: SIMPLE_ASSIGN
x[47] = r_init[47] * cos(theta[47] - 0.00812)
*/
void SpiralGalaxy_eqFunction_741(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,741};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1546]] /* x[47] STATE(1,vx[47]) */) = ((data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */)) * (cos((data->simulationInfo->realParameter[1553] /* theta[47] PARAM */) - 0.00812));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8466(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8467(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8470(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8469(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8468(DATA *data, threadData_t *threadData);


/*
equation index: 747
type: SIMPLE_ASSIGN
vx[47] = (-sin(theta[47])) * r_init[47] * omega_c[47]
*/
void SpiralGalaxy_eqFunction_747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,747};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* vx[47] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1553] /* theta[47] PARAM */)))) * (((data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */)) * ((data->simulationInfo->realParameter[551] /* omega_c[47] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8463(DATA *data, threadData_t *threadData);


/*
equation index: 749
type: SIMPLE_ASSIGN
vy[47] = cos(theta[47]) * r_init[47] * omega_c[47]
*/
void SpiralGalaxy_eqFunction_749(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,749};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[546]] /* vy[47] STATE(1) */) = (cos((data->simulationInfo->realParameter[1553] /* theta[47] PARAM */))) * (((data->simulationInfo->realParameter[1052] /* r_init[47] PARAM */)) * ((data->simulationInfo->realParameter[551] /* omega_c[47] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8462(DATA *data, threadData_t *threadData);


/*
equation index: 751
type: SIMPLE_ASSIGN
vz[47] = 0.0
*/
void SpiralGalaxy_eqFunction_751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,751};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1046]] /* vz[47] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8461(DATA *data, threadData_t *threadData);


/*
equation index: 753
type: SIMPLE_ASSIGN
z[48] = -0.03232
*/
void SpiralGalaxy_eqFunction_753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,753};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2547]] /* z[48] STATE(1,vz[48]) */) = -0.03232;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8474(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8475(DATA *data, threadData_t *threadData);


/*
equation index: 756
type: SIMPLE_ASSIGN
y[48] = r_init[48] * sin(theta[48] - 0.00808)
*/
void SpiralGalaxy_eqFunction_756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,756};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2047]] /* y[48] STATE(1,vy[48]) */) = ((data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */)) * (sin((data->simulationInfo->realParameter[1554] /* theta[48] PARAM */) - 0.00808));
  TRACE_POP
}

/*
equation index: 757
type: SIMPLE_ASSIGN
x[48] = r_init[48] * cos(theta[48] - 0.00808)
*/
void SpiralGalaxy_eqFunction_757(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,757};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1547]] /* x[48] STATE(1,vx[48]) */) = ((data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */)) * (cos((data->simulationInfo->realParameter[1554] /* theta[48] PARAM */) - 0.00808));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8476(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8477(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8480(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8479(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8478(DATA *data, threadData_t *threadData);


/*
equation index: 763
type: SIMPLE_ASSIGN
vx[48] = (-sin(theta[48])) * r_init[48] * omega_c[48]
*/
void SpiralGalaxy_eqFunction_763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,763};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* vx[48] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1554] /* theta[48] PARAM */)))) * (((data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */)) * ((data->simulationInfo->realParameter[552] /* omega_c[48] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8473(DATA *data, threadData_t *threadData);


/*
equation index: 765
type: SIMPLE_ASSIGN
vy[48] = cos(theta[48]) * r_init[48] * omega_c[48]
*/
void SpiralGalaxy_eqFunction_765(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,765};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[547]] /* vy[48] STATE(1) */) = (cos((data->simulationInfo->realParameter[1554] /* theta[48] PARAM */))) * (((data->simulationInfo->realParameter[1053] /* r_init[48] PARAM */)) * ((data->simulationInfo->realParameter[552] /* omega_c[48] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8472(DATA *data, threadData_t *threadData);


/*
equation index: 767
type: SIMPLE_ASSIGN
vz[48] = 0.0
*/
void SpiralGalaxy_eqFunction_767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,767};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1047]] /* vz[48] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8471(DATA *data, threadData_t *threadData);


/*
equation index: 769
type: SIMPLE_ASSIGN
z[49] = -0.03216000000000001
*/
void SpiralGalaxy_eqFunction_769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,769};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2548]] /* z[49] STATE(1,vz[49]) */) = -0.03216000000000001;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8484(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8485(DATA *data, threadData_t *threadData);


/*
equation index: 772
type: SIMPLE_ASSIGN
y[49] = r_init[49] * sin(theta[49] - 0.00804)
*/
void SpiralGalaxy_eqFunction_772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,772};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2048]] /* y[49] STATE(1,vy[49]) */) = ((data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */)) * (sin((data->simulationInfo->realParameter[1555] /* theta[49] PARAM */) - 0.00804));
  TRACE_POP
}

/*
equation index: 773
type: SIMPLE_ASSIGN
x[49] = r_init[49] * cos(theta[49] - 0.00804)
*/
void SpiralGalaxy_eqFunction_773(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,773};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1548]] /* x[49] STATE(1,vx[49]) */) = ((data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */)) * (cos((data->simulationInfo->realParameter[1555] /* theta[49] PARAM */) - 0.00804));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8486(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8487(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8490(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8489(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8488(DATA *data, threadData_t *threadData);


/*
equation index: 779
type: SIMPLE_ASSIGN
vx[49] = (-sin(theta[49])) * r_init[49] * omega_c[49]
*/
void SpiralGalaxy_eqFunction_779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,779};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* vx[49] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1555] /* theta[49] PARAM */)))) * (((data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */)) * ((data->simulationInfo->realParameter[553] /* omega_c[49] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8483(DATA *data, threadData_t *threadData);


/*
equation index: 781
type: SIMPLE_ASSIGN
vy[49] = cos(theta[49]) * r_init[49] * omega_c[49]
*/
void SpiralGalaxy_eqFunction_781(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,781};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[548]] /* vy[49] STATE(1) */) = (cos((data->simulationInfo->realParameter[1555] /* theta[49] PARAM */))) * (((data->simulationInfo->realParameter[1054] /* r_init[49] PARAM */)) * ((data->simulationInfo->realParameter[553] /* omega_c[49] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8482(DATA *data, threadData_t *threadData);


/*
equation index: 783
type: SIMPLE_ASSIGN
vz[49] = 0.0
*/
void SpiralGalaxy_eqFunction_783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,783};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1048]] /* vz[49] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8481(DATA *data, threadData_t *threadData);


/*
equation index: 785
type: SIMPLE_ASSIGN
z[50] = -0.032
*/
void SpiralGalaxy_eqFunction_785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,785};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2549]] /* z[50] STATE(1,vz[50]) */) = -0.032;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8494(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8495(DATA *data, threadData_t *threadData);


/*
equation index: 788
type: SIMPLE_ASSIGN
y[50] = r_init[50] * sin(theta[50] - 0.008)
*/
void SpiralGalaxy_eqFunction_788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,788};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2049]] /* y[50] STATE(1,vy[50]) */) = ((data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */)) * (sin((data->simulationInfo->realParameter[1556] /* theta[50] PARAM */) - 0.008));
  TRACE_POP
}

/*
equation index: 789
type: SIMPLE_ASSIGN
x[50] = r_init[50] * cos(theta[50] - 0.008)
*/
void SpiralGalaxy_eqFunction_789(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,789};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1549]] /* x[50] STATE(1,vx[50]) */) = ((data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */)) * (cos((data->simulationInfo->realParameter[1556] /* theta[50] PARAM */) - 0.008));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8496(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8497(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8500(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8499(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8498(DATA *data, threadData_t *threadData);


/*
equation index: 795
type: SIMPLE_ASSIGN
vx[50] = (-sin(theta[50])) * r_init[50] * omega_c[50]
*/
void SpiralGalaxy_eqFunction_795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,795};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[49]] /* vx[50] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1556] /* theta[50] PARAM */)))) * (((data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */)) * ((data->simulationInfo->realParameter[554] /* omega_c[50] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8493(DATA *data, threadData_t *threadData);


/*
equation index: 797
type: SIMPLE_ASSIGN
vy[50] = cos(theta[50]) * r_init[50] * omega_c[50]
*/
void SpiralGalaxy_eqFunction_797(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,797};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[549]] /* vy[50] STATE(1) */) = (cos((data->simulationInfo->realParameter[1556] /* theta[50] PARAM */))) * (((data->simulationInfo->realParameter[1055] /* r_init[50] PARAM */)) * ((data->simulationInfo->realParameter[554] /* omega_c[50] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8492(DATA *data, threadData_t *threadData);


/*
equation index: 799
type: SIMPLE_ASSIGN
vz[50] = 0.0
*/
void SpiralGalaxy_eqFunction_799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,799};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1049]] /* vz[50] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8491(DATA *data, threadData_t *threadData);


/*
equation index: 801
type: SIMPLE_ASSIGN
z[51] = -0.03184
*/
void SpiralGalaxy_eqFunction_801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,801};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2550]] /* z[51] STATE(1,vz[51]) */) = -0.03184;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8504(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8505(DATA *data, threadData_t *threadData);


/*
equation index: 804
type: SIMPLE_ASSIGN
y[51] = r_init[51] * sin(theta[51] - 0.00796)
*/
void SpiralGalaxy_eqFunction_804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,804};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2050]] /* y[51] STATE(1,vy[51]) */) = ((data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */)) * (sin((data->simulationInfo->realParameter[1557] /* theta[51] PARAM */) - 0.00796));
  TRACE_POP
}

/*
equation index: 805
type: SIMPLE_ASSIGN
x[51] = r_init[51] * cos(theta[51] - 0.00796)
*/
void SpiralGalaxy_eqFunction_805(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,805};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1550]] /* x[51] STATE(1,vx[51]) */) = ((data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */)) * (cos((data->simulationInfo->realParameter[1557] /* theta[51] PARAM */) - 0.00796));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8506(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8507(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8510(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8509(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8508(DATA *data, threadData_t *threadData);


/*
equation index: 811
type: SIMPLE_ASSIGN
vx[51] = (-sin(theta[51])) * r_init[51] * omega_c[51]
*/
void SpiralGalaxy_eqFunction_811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,811};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[50]] /* vx[51] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1557] /* theta[51] PARAM */)))) * (((data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */)) * ((data->simulationInfo->realParameter[555] /* omega_c[51] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8503(DATA *data, threadData_t *threadData);


/*
equation index: 813
type: SIMPLE_ASSIGN
vy[51] = cos(theta[51]) * r_init[51] * omega_c[51]
*/
void SpiralGalaxy_eqFunction_813(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,813};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[550]] /* vy[51] STATE(1) */) = (cos((data->simulationInfo->realParameter[1557] /* theta[51] PARAM */))) * (((data->simulationInfo->realParameter[1056] /* r_init[51] PARAM */)) * ((data->simulationInfo->realParameter[555] /* omega_c[51] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8502(DATA *data, threadData_t *threadData);


/*
equation index: 815
type: SIMPLE_ASSIGN
vz[51] = 0.0
*/
void SpiralGalaxy_eqFunction_815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,815};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1050]] /* vz[51] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8501(DATA *data, threadData_t *threadData);


/*
equation index: 817
type: SIMPLE_ASSIGN
z[52] = -0.03168
*/
void SpiralGalaxy_eqFunction_817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,817};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2551]] /* z[52] STATE(1,vz[52]) */) = -0.03168;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8514(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8515(DATA *data, threadData_t *threadData);


/*
equation index: 820
type: SIMPLE_ASSIGN
y[52] = r_init[52] * sin(theta[52] - 0.00792)
*/
void SpiralGalaxy_eqFunction_820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,820};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2051]] /* y[52] STATE(1,vy[52]) */) = ((data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */)) * (sin((data->simulationInfo->realParameter[1558] /* theta[52] PARAM */) - 0.00792));
  TRACE_POP
}

/*
equation index: 821
type: SIMPLE_ASSIGN
x[52] = r_init[52] * cos(theta[52] - 0.00792)
*/
void SpiralGalaxy_eqFunction_821(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,821};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1551]] /* x[52] STATE(1,vx[52]) */) = ((data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */)) * (cos((data->simulationInfo->realParameter[1558] /* theta[52] PARAM */) - 0.00792));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8516(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8517(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8520(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8519(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8518(DATA *data, threadData_t *threadData);


/*
equation index: 827
type: SIMPLE_ASSIGN
vx[52] = (-sin(theta[52])) * r_init[52] * omega_c[52]
*/
void SpiralGalaxy_eqFunction_827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,827};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[51]] /* vx[52] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1558] /* theta[52] PARAM */)))) * (((data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */)) * ((data->simulationInfo->realParameter[556] /* omega_c[52] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8513(DATA *data, threadData_t *threadData);


/*
equation index: 829
type: SIMPLE_ASSIGN
vy[52] = cos(theta[52]) * r_init[52] * omega_c[52]
*/
void SpiralGalaxy_eqFunction_829(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,829};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[551]] /* vy[52] STATE(1) */) = (cos((data->simulationInfo->realParameter[1558] /* theta[52] PARAM */))) * (((data->simulationInfo->realParameter[1057] /* r_init[52] PARAM */)) * ((data->simulationInfo->realParameter[556] /* omega_c[52] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8512(DATA *data, threadData_t *threadData);


/*
equation index: 831
type: SIMPLE_ASSIGN
vz[52] = 0.0
*/
void SpiralGalaxy_eqFunction_831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,831};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1051]] /* vz[52] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8511(DATA *data, threadData_t *threadData);


/*
equation index: 833
type: SIMPLE_ASSIGN
z[53] = -0.03152
*/
void SpiralGalaxy_eqFunction_833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,833};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2552]] /* z[53] STATE(1,vz[53]) */) = -0.03152;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8524(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8525(DATA *data, threadData_t *threadData);


/*
equation index: 836
type: SIMPLE_ASSIGN
y[53] = r_init[53] * sin(theta[53] - 0.00788)
*/
void SpiralGalaxy_eqFunction_836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,836};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2052]] /* y[53] STATE(1,vy[53]) */) = ((data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */)) * (sin((data->simulationInfo->realParameter[1559] /* theta[53] PARAM */) - 0.00788));
  TRACE_POP
}

/*
equation index: 837
type: SIMPLE_ASSIGN
x[53] = r_init[53] * cos(theta[53] - 0.00788)
*/
void SpiralGalaxy_eqFunction_837(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,837};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1552]] /* x[53] STATE(1,vx[53]) */) = ((data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */)) * (cos((data->simulationInfo->realParameter[1559] /* theta[53] PARAM */) - 0.00788));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8526(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8527(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8530(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8529(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8528(DATA *data, threadData_t *threadData);


/*
equation index: 843
type: SIMPLE_ASSIGN
vx[53] = (-sin(theta[53])) * r_init[53] * omega_c[53]
*/
void SpiralGalaxy_eqFunction_843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,843};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[52]] /* vx[53] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1559] /* theta[53] PARAM */)))) * (((data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */)) * ((data->simulationInfo->realParameter[557] /* omega_c[53] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8523(DATA *data, threadData_t *threadData);


/*
equation index: 845
type: SIMPLE_ASSIGN
vy[53] = cos(theta[53]) * r_init[53] * omega_c[53]
*/
void SpiralGalaxy_eqFunction_845(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,845};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[552]] /* vy[53] STATE(1) */) = (cos((data->simulationInfo->realParameter[1559] /* theta[53] PARAM */))) * (((data->simulationInfo->realParameter[1058] /* r_init[53] PARAM */)) * ((data->simulationInfo->realParameter[557] /* omega_c[53] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8522(DATA *data, threadData_t *threadData);


/*
equation index: 847
type: SIMPLE_ASSIGN
vz[53] = 0.0
*/
void SpiralGalaxy_eqFunction_847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,847};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1052]] /* vz[53] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8521(DATA *data, threadData_t *threadData);


/*
equation index: 849
type: SIMPLE_ASSIGN
z[54] = -0.03136
*/
void SpiralGalaxy_eqFunction_849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,849};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2553]] /* z[54] STATE(1,vz[54]) */) = -0.03136;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8534(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8535(DATA *data, threadData_t *threadData);


/*
equation index: 852
type: SIMPLE_ASSIGN
y[54] = r_init[54] * sin(theta[54] - 0.00784)
*/
void SpiralGalaxy_eqFunction_852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,852};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2053]] /* y[54] STATE(1,vy[54]) */) = ((data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */)) * (sin((data->simulationInfo->realParameter[1560] /* theta[54] PARAM */) - 0.00784));
  TRACE_POP
}

/*
equation index: 853
type: SIMPLE_ASSIGN
x[54] = r_init[54] * cos(theta[54] - 0.00784)
*/
void SpiralGalaxy_eqFunction_853(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,853};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1553]] /* x[54] STATE(1,vx[54]) */) = ((data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */)) * (cos((data->simulationInfo->realParameter[1560] /* theta[54] PARAM */) - 0.00784));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8536(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8537(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8540(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8539(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8538(DATA *data, threadData_t *threadData);


/*
equation index: 859
type: SIMPLE_ASSIGN
vx[54] = (-sin(theta[54])) * r_init[54] * omega_c[54]
*/
void SpiralGalaxy_eqFunction_859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,859};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[53]] /* vx[54] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1560] /* theta[54] PARAM */)))) * (((data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */)) * ((data->simulationInfo->realParameter[558] /* omega_c[54] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8533(DATA *data, threadData_t *threadData);


/*
equation index: 861
type: SIMPLE_ASSIGN
vy[54] = cos(theta[54]) * r_init[54] * omega_c[54]
*/
void SpiralGalaxy_eqFunction_861(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,861};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[553]] /* vy[54] STATE(1) */) = (cos((data->simulationInfo->realParameter[1560] /* theta[54] PARAM */))) * (((data->simulationInfo->realParameter[1059] /* r_init[54] PARAM */)) * ((data->simulationInfo->realParameter[558] /* omega_c[54] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8532(DATA *data, threadData_t *threadData);


/*
equation index: 863
type: SIMPLE_ASSIGN
vz[54] = 0.0
*/
void SpiralGalaxy_eqFunction_863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,863};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1053]] /* vz[54] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8531(DATA *data, threadData_t *threadData);


/*
equation index: 865
type: SIMPLE_ASSIGN
z[55] = -0.031200000000000006
*/
void SpiralGalaxy_eqFunction_865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,865};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2554]] /* z[55] STATE(1,vz[55]) */) = -0.031200000000000006;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8544(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8545(DATA *data, threadData_t *threadData);


/*
equation index: 868
type: SIMPLE_ASSIGN
y[55] = r_init[55] * sin(theta[55] - 0.0078000000000000005)
*/
void SpiralGalaxy_eqFunction_868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,868};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2054]] /* y[55] STATE(1,vy[55]) */) = ((data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */)) * (sin((data->simulationInfo->realParameter[1561] /* theta[55] PARAM */) - 0.0078000000000000005));
  TRACE_POP
}

/*
equation index: 869
type: SIMPLE_ASSIGN
x[55] = r_init[55] * cos(theta[55] - 0.0078000000000000005)
*/
void SpiralGalaxy_eqFunction_869(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,869};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1554]] /* x[55] STATE(1,vx[55]) */) = ((data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */)) * (cos((data->simulationInfo->realParameter[1561] /* theta[55] PARAM */) - 0.0078000000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8546(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8547(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8550(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8549(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8548(DATA *data, threadData_t *threadData);


/*
equation index: 875
type: SIMPLE_ASSIGN
vx[55] = (-sin(theta[55])) * r_init[55] * omega_c[55]
*/
void SpiralGalaxy_eqFunction_875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,875};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[54]] /* vx[55] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1561] /* theta[55] PARAM */)))) * (((data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */)) * ((data->simulationInfo->realParameter[559] /* omega_c[55] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8543(DATA *data, threadData_t *threadData);


/*
equation index: 877
type: SIMPLE_ASSIGN
vy[55] = cos(theta[55]) * r_init[55] * omega_c[55]
*/
void SpiralGalaxy_eqFunction_877(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,877};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[554]] /* vy[55] STATE(1) */) = (cos((data->simulationInfo->realParameter[1561] /* theta[55] PARAM */))) * (((data->simulationInfo->realParameter[1060] /* r_init[55] PARAM */)) * ((data->simulationInfo->realParameter[559] /* omega_c[55] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8542(DATA *data, threadData_t *threadData);


/*
equation index: 879
type: SIMPLE_ASSIGN
vz[55] = 0.0
*/
void SpiralGalaxy_eqFunction_879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,879};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1054]] /* vz[55] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8541(DATA *data, threadData_t *threadData);


/*
equation index: 881
type: SIMPLE_ASSIGN
z[56] = -0.031040000000000005
*/
void SpiralGalaxy_eqFunction_881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,881};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2555]] /* z[56] STATE(1,vz[56]) */) = -0.031040000000000005;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8554(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8555(DATA *data, threadData_t *threadData);


/*
equation index: 884
type: SIMPLE_ASSIGN
y[56] = r_init[56] * sin(theta[56] - 0.00776)
*/
void SpiralGalaxy_eqFunction_884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,884};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2055]] /* y[56] STATE(1,vy[56]) */) = ((data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */)) * (sin((data->simulationInfo->realParameter[1562] /* theta[56] PARAM */) - 0.00776));
  TRACE_POP
}

/*
equation index: 885
type: SIMPLE_ASSIGN
x[56] = r_init[56] * cos(theta[56] - 0.00776)
*/
void SpiralGalaxy_eqFunction_885(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,885};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1555]] /* x[56] STATE(1,vx[56]) */) = ((data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */)) * (cos((data->simulationInfo->realParameter[1562] /* theta[56] PARAM */) - 0.00776));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8556(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8557(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8560(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8559(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8558(DATA *data, threadData_t *threadData);


/*
equation index: 891
type: SIMPLE_ASSIGN
vx[56] = (-sin(theta[56])) * r_init[56] * omega_c[56]
*/
void SpiralGalaxy_eqFunction_891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,891};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[55]] /* vx[56] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1562] /* theta[56] PARAM */)))) * (((data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */)) * ((data->simulationInfo->realParameter[560] /* omega_c[56] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8553(DATA *data, threadData_t *threadData);


/*
equation index: 893
type: SIMPLE_ASSIGN
vy[56] = cos(theta[56]) * r_init[56] * omega_c[56]
*/
void SpiralGalaxy_eqFunction_893(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,893};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[555]] /* vy[56] STATE(1) */) = (cos((data->simulationInfo->realParameter[1562] /* theta[56] PARAM */))) * (((data->simulationInfo->realParameter[1061] /* r_init[56] PARAM */)) * ((data->simulationInfo->realParameter[560] /* omega_c[56] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8552(DATA *data, threadData_t *threadData);


/*
equation index: 895
type: SIMPLE_ASSIGN
vz[56] = 0.0
*/
void SpiralGalaxy_eqFunction_895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,895};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1055]] /* vz[56] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8551(DATA *data, threadData_t *threadData);


/*
equation index: 897
type: SIMPLE_ASSIGN
z[57] = -0.03088
*/
void SpiralGalaxy_eqFunction_897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,897};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2556]] /* z[57] STATE(1,vz[57]) */) = -0.03088;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8564(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8565(DATA *data, threadData_t *threadData);


/*
equation index: 900
type: SIMPLE_ASSIGN
y[57] = r_init[57] * sin(theta[57] - 0.00772)
*/
void SpiralGalaxy_eqFunction_900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,900};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2056]] /* y[57] STATE(1,vy[57]) */) = ((data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */)) * (sin((data->simulationInfo->realParameter[1563] /* theta[57] PARAM */) - 0.00772));
  TRACE_POP
}

/*
equation index: 901
type: SIMPLE_ASSIGN
x[57] = r_init[57] * cos(theta[57] - 0.00772)
*/
void SpiralGalaxy_eqFunction_901(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,901};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1556]] /* x[57] STATE(1,vx[57]) */) = ((data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */)) * (cos((data->simulationInfo->realParameter[1563] /* theta[57] PARAM */) - 0.00772));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8566(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8567(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8570(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8569(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8568(DATA *data, threadData_t *threadData);


/*
equation index: 907
type: SIMPLE_ASSIGN
vx[57] = (-sin(theta[57])) * r_init[57] * omega_c[57]
*/
void SpiralGalaxy_eqFunction_907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,907};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[56]] /* vx[57] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1563] /* theta[57] PARAM */)))) * (((data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */)) * ((data->simulationInfo->realParameter[561] /* omega_c[57] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8563(DATA *data, threadData_t *threadData);


/*
equation index: 909
type: SIMPLE_ASSIGN
vy[57] = cos(theta[57]) * r_init[57] * omega_c[57]
*/
void SpiralGalaxy_eqFunction_909(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,909};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[556]] /* vy[57] STATE(1) */) = (cos((data->simulationInfo->realParameter[1563] /* theta[57] PARAM */))) * (((data->simulationInfo->realParameter[1062] /* r_init[57] PARAM */)) * ((data->simulationInfo->realParameter[561] /* omega_c[57] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8562(DATA *data, threadData_t *threadData);


/*
equation index: 911
type: SIMPLE_ASSIGN
vz[57] = 0.0
*/
void SpiralGalaxy_eqFunction_911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,911};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1056]] /* vz[57] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8561(DATA *data, threadData_t *threadData);


/*
equation index: 913
type: SIMPLE_ASSIGN
z[58] = -0.03072
*/
void SpiralGalaxy_eqFunction_913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,913};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2557]] /* z[58] STATE(1,vz[58]) */) = -0.03072;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8574(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8575(DATA *data, threadData_t *threadData);


/*
equation index: 916
type: SIMPLE_ASSIGN
y[58] = r_init[58] * sin(theta[58] - 0.00768)
*/
void SpiralGalaxy_eqFunction_916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,916};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2057]] /* y[58] STATE(1,vy[58]) */) = ((data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */)) * (sin((data->simulationInfo->realParameter[1564] /* theta[58] PARAM */) - 0.00768));
  TRACE_POP
}

/*
equation index: 917
type: SIMPLE_ASSIGN
x[58] = r_init[58] * cos(theta[58] - 0.00768)
*/
void SpiralGalaxy_eqFunction_917(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,917};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1557]] /* x[58] STATE(1,vx[58]) */) = ((data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */)) * (cos((data->simulationInfo->realParameter[1564] /* theta[58] PARAM */) - 0.00768));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8576(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8577(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8580(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8579(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8578(DATA *data, threadData_t *threadData);


/*
equation index: 923
type: SIMPLE_ASSIGN
vx[58] = (-sin(theta[58])) * r_init[58] * omega_c[58]
*/
void SpiralGalaxy_eqFunction_923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,923};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[57]] /* vx[58] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1564] /* theta[58] PARAM */)))) * (((data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */)) * ((data->simulationInfo->realParameter[562] /* omega_c[58] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8573(DATA *data, threadData_t *threadData);


/*
equation index: 925
type: SIMPLE_ASSIGN
vy[58] = cos(theta[58]) * r_init[58] * omega_c[58]
*/
void SpiralGalaxy_eqFunction_925(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,925};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[557]] /* vy[58] STATE(1) */) = (cos((data->simulationInfo->realParameter[1564] /* theta[58] PARAM */))) * (((data->simulationInfo->realParameter[1063] /* r_init[58] PARAM */)) * ((data->simulationInfo->realParameter[562] /* omega_c[58] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8572(DATA *data, threadData_t *threadData);


/*
equation index: 927
type: SIMPLE_ASSIGN
vz[58] = 0.0
*/
void SpiralGalaxy_eqFunction_927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,927};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1057]] /* vz[58] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8571(DATA *data, threadData_t *threadData);


/*
equation index: 929
type: SIMPLE_ASSIGN
z[59] = -0.030560000000000004
*/
void SpiralGalaxy_eqFunction_929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,929};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2558]] /* z[59] STATE(1,vz[59]) */) = -0.030560000000000004;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8584(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8585(DATA *data, threadData_t *threadData);


/*
equation index: 932
type: SIMPLE_ASSIGN
y[59] = r_init[59] * sin(theta[59] - 0.00764)
*/
void SpiralGalaxy_eqFunction_932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,932};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2058]] /* y[59] STATE(1,vy[59]) */) = ((data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */)) * (sin((data->simulationInfo->realParameter[1565] /* theta[59] PARAM */) - 0.00764));
  TRACE_POP
}

/*
equation index: 933
type: SIMPLE_ASSIGN
x[59] = r_init[59] * cos(theta[59] - 0.00764)
*/
void SpiralGalaxy_eqFunction_933(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,933};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1558]] /* x[59] STATE(1,vx[59]) */) = ((data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */)) * (cos((data->simulationInfo->realParameter[1565] /* theta[59] PARAM */) - 0.00764));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8586(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8587(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8590(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8589(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8588(DATA *data, threadData_t *threadData);


/*
equation index: 939
type: SIMPLE_ASSIGN
vx[59] = (-sin(theta[59])) * r_init[59] * omega_c[59]
*/
void SpiralGalaxy_eqFunction_939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,939};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[58]] /* vx[59] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1565] /* theta[59] PARAM */)))) * (((data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */)) * ((data->simulationInfo->realParameter[563] /* omega_c[59] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8583(DATA *data, threadData_t *threadData);


/*
equation index: 941
type: SIMPLE_ASSIGN
vy[59] = cos(theta[59]) * r_init[59] * omega_c[59]
*/
void SpiralGalaxy_eqFunction_941(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,941};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[558]] /* vy[59] STATE(1) */) = (cos((data->simulationInfo->realParameter[1565] /* theta[59] PARAM */))) * (((data->simulationInfo->realParameter[1064] /* r_init[59] PARAM */)) * ((data->simulationInfo->realParameter[563] /* omega_c[59] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8582(DATA *data, threadData_t *threadData);


/*
equation index: 943
type: SIMPLE_ASSIGN
vz[59] = 0.0
*/
void SpiralGalaxy_eqFunction_943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,943};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1058]] /* vz[59] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8581(DATA *data, threadData_t *threadData);


/*
equation index: 945
type: SIMPLE_ASSIGN
z[60] = -0.030400000000000003
*/
void SpiralGalaxy_eqFunction_945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,945};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2559]] /* z[60] STATE(1,vz[60]) */) = -0.030400000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8594(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8595(DATA *data, threadData_t *threadData);


/*
equation index: 948
type: SIMPLE_ASSIGN
y[60] = r_init[60] * sin(theta[60] - 0.0076)
*/
void SpiralGalaxy_eqFunction_948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,948};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2059]] /* y[60] STATE(1,vy[60]) */) = ((data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */)) * (sin((data->simulationInfo->realParameter[1566] /* theta[60] PARAM */) - 0.0076));
  TRACE_POP
}

/*
equation index: 949
type: SIMPLE_ASSIGN
x[60] = r_init[60] * cos(theta[60] - 0.0076)
*/
void SpiralGalaxy_eqFunction_949(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,949};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1559]] /* x[60] STATE(1,vx[60]) */) = ((data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */)) * (cos((data->simulationInfo->realParameter[1566] /* theta[60] PARAM */) - 0.0076));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8596(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8597(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8600(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8599(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8598(DATA *data, threadData_t *threadData);


/*
equation index: 955
type: SIMPLE_ASSIGN
vx[60] = (-sin(theta[60])) * r_init[60] * omega_c[60]
*/
void SpiralGalaxy_eqFunction_955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,955};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[59]] /* vx[60] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1566] /* theta[60] PARAM */)))) * (((data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */)) * ((data->simulationInfo->realParameter[564] /* omega_c[60] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8593(DATA *data, threadData_t *threadData);


/*
equation index: 957
type: SIMPLE_ASSIGN
vy[60] = cos(theta[60]) * r_init[60] * omega_c[60]
*/
void SpiralGalaxy_eqFunction_957(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,957};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[559]] /* vy[60] STATE(1) */) = (cos((data->simulationInfo->realParameter[1566] /* theta[60] PARAM */))) * (((data->simulationInfo->realParameter[1065] /* r_init[60] PARAM */)) * ((data->simulationInfo->realParameter[564] /* omega_c[60] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8592(DATA *data, threadData_t *threadData);


/*
equation index: 959
type: SIMPLE_ASSIGN
vz[60] = 0.0
*/
void SpiralGalaxy_eqFunction_959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,959};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1059]] /* vz[60] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8591(DATA *data, threadData_t *threadData);


/*
equation index: 961
type: SIMPLE_ASSIGN
z[61] = -0.03024
*/
void SpiralGalaxy_eqFunction_961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,961};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2560]] /* z[61] STATE(1,vz[61]) */) = -0.03024;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8604(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8605(DATA *data, threadData_t *threadData);


/*
equation index: 964
type: SIMPLE_ASSIGN
y[61] = r_init[61] * sin(theta[61] - 0.00756)
*/
void SpiralGalaxy_eqFunction_964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,964};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2060]] /* y[61] STATE(1,vy[61]) */) = ((data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */)) * (sin((data->simulationInfo->realParameter[1567] /* theta[61] PARAM */) - 0.00756));
  TRACE_POP
}

/*
equation index: 965
type: SIMPLE_ASSIGN
x[61] = r_init[61] * cos(theta[61] - 0.00756)
*/
void SpiralGalaxy_eqFunction_965(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,965};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1560]] /* x[61] STATE(1,vx[61]) */) = ((data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */)) * (cos((data->simulationInfo->realParameter[1567] /* theta[61] PARAM */) - 0.00756));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8606(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8607(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8610(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8609(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8608(DATA *data, threadData_t *threadData);


/*
equation index: 971
type: SIMPLE_ASSIGN
vx[61] = (-sin(theta[61])) * r_init[61] * omega_c[61]
*/
void SpiralGalaxy_eqFunction_971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,971};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[60]] /* vx[61] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1567] /* theta[61] PARAM */)))) * (((data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */)) * ((data->simulationInfo->realParameter[565] /* omega_c[61] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8603(DATA *data, threadData_t *threadData);


/*
equation index: 973
type: SIMPLE_ASSIGN
vy[61] = cos(theta[61]) * r_init[61] * omega_c[61]
*/
void SpiralGalaxy_eqFunction_973(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,973};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[560]] /* vy[61] STATE(1) */) = (cos((data->simulationInfo->realParameter[1567] /* theta[61] PARAM */))) * (((data->simulationInfo->realParameter[1066] /* r_init[61] PARAM */)) * ((data->simulationInfo->realParameter[565] /* omega_c[61] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8602(DATA *data, threadData_t *threadData);


/*
equation index: 975
type: SIMPLE_ASSIGN
vz[61] = 0.0
*/
void SpiralGalaxy_eqFunction_975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,975};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1060]] /* vz[61] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8601(DATA *data, threadData_t *threadData);


/*
equation index: 977
type: SIMPLE_ASSIGN
z[62] = -0.030080000000000003
*/
void SpiralGalaxy_eqFunction_977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,977};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2561]] /* z[62] STATE(1,vz[62]) */) = -0.030080000000000003;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8614(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8615(DATA *data, threadData_t *threadData);


/*
equation index: 980
type: SIMPLE_ASSIGN
y[62] = r_init[62] * sin(theta[62] - 0.00752)
*/
void SpiralGalaxy_eqFunction_980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,980};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2061]] /* y[62] STATE(1,vy[62]) */) = ((data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */)) * (sin((data->simulationInfo->realParameter[1568] /* theta[62] PARAM */) - 0.00752));
  TRACE_POP
}

/*
equation index: 981
type: SIMPLE_ASSIGN
x[62] = r_init[62] * cos(theta[62] - 0.00752)
*/
void SpiralGalaxy_eqFunction_981(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,981};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1561]] /* x[62] STATE(1,vx[62]) */) = ((data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */)) * (cos((data->simulationInfo->realParameter[1568] /* theta[62] PARAM */) - 0.00752));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8616(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8617(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8620(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8619(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8618(DATA *data, threadData_t *threadData);


/*
equation index: 987
type: SIMPLE_ASSIGN
vx[62] = (-sin(theta[62])) * r_init[62] * omega_c[62]
*/
void SpiralGalaxy_eqFunction_987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,987};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[61]] /* vx[62] STATE(1) */) = ((-sin((data->simulationInfo->realParameter[1568] /* theta[62] PARAM */)))) * (((data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */)) * ((data->simulationInfo->realParameter[566] /* omega_c[62] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8613(DATA *data, threadData_t *threadData);


/*
equation index: 989
type: SIMPLE_ASSIGN
vy[62] = cos(theta[62]) * r_init[62] * omega_c[62]
*/
void SpiralGalaxy_eqFunction_989(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,989};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[561]] /* vy[62] STATE(1) */) = (cos((data->simulationInfo->realParameter[1568] /* theta[62] PARAM */))) * (((data->simulationInfo->realParameter[1067] /* r_init[62] PARAM */)) * ((data->simulationInfo->realParameter[566] /* omega_c[62] PARAM */)));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8612(DATA *data, threadData_t *threadData);


/*
equation index: 991
type: SIMPLE_ASSIGN
vz[62] = 0.0
*/
void SpiralGalaxy_eqFunction_991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,991};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1061]] /* vz[62] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8611(DATA *data, threadData_t *threadData);


/*
equation index: 993
type: SIMPLE_ASSIGN
z[63] = -0.029920000000000002
*/
void SpiralGalaxy_eqFunction_993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,993};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2562]] /* z[63] STATE(1,vz[63]) */) = -0.029920000000000002;
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8624(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8625(DATA *data, threadData_t *threadData);


/*
equation index: 996
type: SIMPLE_ASSIGN
y[63] = r_init[63] * sin(theta[63] - 0.0074800000000000005)
*/
void SpiralGalaxy_eqFunction_996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,996};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2062]] /* y[63] STATE(1,vy[63]) */) = ((data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */)) * (sin((data->simulationInfo->realParameter[1569] /* theta[63] PARAM */) - 0.0074800000000000005));
  TRACE_POP
}

/*
equation index: 997
type: SIMPLE_ASSIGN
x[63] = r_init[63] * cos(theta[63] - 0.0074800000000000005)
*/
void SpiralGalaxy_eqFunction_997(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,997};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1562]] /* x[63] STATE(1,vx[63]) */) = ((data->simulationInfo->realParameter[1068] /* r_init[63] PARAM */)) * (cos((data->simulationInfo->realParameter[1569] /* theta[63] PARAM */) - 0.0074800000000000005));
  TRACE_POP
}
extern void SpiralGalaxy_eqFunction_8626(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8627(DATA *data, threadData_t *threadData);

extern void SpiralGalaxy_eqFunction_8630(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void SpiralGalaxy_functionInitialEquations_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  SpiralGalaxy_eqFunction_501(data, threadData);
  SpiralGalaxy_eqFunction_8316(data, threadData);
  SpiralGalaxy_eqFunction_8317(data, threadData);
  SpiralGalaxy_eqFunction_8320(data, threadData);
  SpiralGalaxy_eqFunction_8319(data, threadData);
  SpiralGalaxy_eqFunction_8318(data, threadData);
  SpiralGalaxy_eqFunction_507(data, threadData);
  SpiralGalaxy_eqFunction_8313(data, threadData);
  SpiralGalaxy_eqFunction_509(data, threadData);
  SpiralGalaxy_eqFunction_8312(data, threadData);
  SpiralGalaxy_eqFunction_511(data, threadData);
  SpiralGalaxy_eqFunction_8311(data, threadData);
  SpiralGalaxy_eqFunction_513(data, threadData);
  SpiralGalaxy_eqFunction_8324(data, threadData);
  SpiralGalaxy_eqFunction_8325(data, threadData);
  SpiralGalaxy_eqFunction_516(data, threadData);
  SpiralGalaxy_eqFunction_517(data, threadData);
  SpiralGalaxy_eqFunction_8326(data, threadData);
  SpiralGalaxy_eqFunction_8327(data, threadData);
  SpiralGalaxy_eqFunction_8330(data, threadData);
  SpiralGalaxy_eqFunction_8329(data, threadData);
  SpiralGalaxy_eqFunction_8328(data, threadData);
  SpiralGalaxy_eqFunction_523(data, threadData);
  SpiralGalaxy_eqFunction_8323(data, threadData);
  SpiralGalaxy_eqFunction_525(data, threadData);
  SpiralGalaxy_eqFunction_8322(data, threadData);
  SpiralGalaxy_eqFunction_527(data, threadData);
  SpiralGalaxy_eqFunction_8321(data, threadData);
  SpiralGalaxy_eqFunction_529(data, threadData);
  SpiralGalaxy_eqFunction_8334(data, threadData);
  SpiralGalaxy_eqFunction_8335(data, threadData);
  SpiralGalaxy_eqFunction_532(data, threadData);
  SpiralGalaxy_eqFunction_533(data, threadData);
  SpiralGalaxy_eqFunction_8336(data, threadData);
  SpiralGalaxy_eqFunction_8337(data, threadData);
  SpiralGalaxy_eqFunction_8340(data, threadData);
  SpiralGalaxy_eqFunction_8339(data, threadData);
  SpiralGalaxy_eqFunction_8338(data, threadData);
  SpiralGalaxy_eqFunction_539(data, threadData);
  SpiralGalaxy_eqFunction_8333(data, threadData);
  SpiralGalaxy_eqFunction_541(data, threadData);
  SpiralGalaxy_eqFunction_8332(data, threadData);
  SpiralGalaxy_eqFunction_543(data, threadData);
  SpiralGalaxy_eqFunction_8331(data, threadData);
  SpiralGalaxy_eqFunction_545(data, threadData);
  SpiralGalaxy_eqFunction_8344(data, threadData);
  SpiralGalaxy_eqFunction_8345(data, threadData);
  SpiralGalaxy_eqFunction_548(data, threadData);
  SpiralGalaxy_eqFunction_549(data, threadData);
  SpiralGalaxy_eqFunction_8346(data, threadData);
  SpiralGalaxy_eqFunction_8347(data, threadData);
  SpiralGalaxy_eqFunction_8350(data, threadData);
  SpiralGalaxy_eqFunction_8349(data, threadData);
  SpiralGalaxy_eqFunction_8348(data, threadData);
  SpiralGalaxy_eqFunction_555(data, threadData);
  SpiralGalaxy_eqFunction_8343(data, threadData);
  SpiralGalaxy_eqFunction_557(data, threadData);
  SpiralGalaxy_eqFunction_8342(data, threadData);
  SpiralGalaxy_eqFunction_559(data, threadData);
  SpiralGalaxy_eqFunction_8341(data, threadData);
  SpiralGalaxy_eqFunction_561(data, threadData);
  SpiralGalaxy_eqFunction_8354(data, threadData);
  SpiralGalaxy_eqFunction_8355(data, threadData);
  SpiralGalaxy_eqFunction_564(data, threadData);
  SpiralGalaxy_eqFunction_565(data, threadData);
  SpiralGalaxy_eqFunction_8356(data, threadData);
  SpiralGalaxy_eqFunction_8357(data, threadData);
  SpiralGalaxy_eqFunction_8360(data, threadData);
  SpiralGalaxy_eqFunction_8359(data, threadData);
  SpiralGalaxy_eqFunction_8358(data, threadData);
  SpiralGalaxy_eqFunction_571(data, threadData);
  SpiralGalaxy_eqFunction_8353(data, threadData);
  SpiralGalaxy_eqFunction_573(data, threadData);
  SpiralGalaxy_eqFunction_8352(data, threadData);
  SpiralGalaxy_eqFunction_575(data, threadData);
  SpiralGalaxy_eqFunction_8351(data, threadData);
  SpiralGalaxy_eqFunction_577(data, threadData);
  SpiralGalaxy_eqFunction_8364(data, threadData);
  SpiralGalaxy_eqFunction_8365(data, threadData);
  SpiralGalaxy_eqFunction_580(data, threadData);
  SpiralGalaxy_eqFunction_581(data, threadData);
  SpiralGalaxy_eqFunction_8366(data, threadData);
  SpiralGalaxy_eqFunction_8367(data, threadData);
  SpiralGalaxy_eqFunction_8370(data, threadData);
  SpiralGalaxy_eqFunction_8369(data, threadData);
  SpiralGalaxy_eqFunction_8368(data, threadData);
  SpiralGalaxy_eqFunction_587(data, threadData);
  SpiralGalaxy_eqFunction_8363(data, threadData);
  SpiralGalaxy_eqFunction_589(data, threadData);
  SpiralGalaxy_eqFunction_8362(data, threadData);
  SpiralGalaxy_eqFunction_591(data, threadData);
  SpiralGalaxy_eqFunction_8361(data, threadData);
  SpiralGalaxy_eqFunction_593(data, threadData);
  SpiralGalaxy_eqFunction_8374(data, threadData);
  SpiralGalaxy_eqFunction_8375(data, threadData);
  SpiralGalaxy_eqFunction_596(data, threadData);
  SpiralGalaxy_eqFunction_597(data, threadData);
  SpiralGalaxy_eqFunction_8376(data, threadData);
  SpiralGalaxy_eqFunction_8377(data, threadData);
  SpiralGalaxy_eqFunction_8380(data, threadData);
  SpiralGalaxy_eqFunction_8379(data, threadData);
  SpiralGalaxy_eqFunction_8378(data, threadData);
  SpiralGalaxy_eqFunction_603(data, threadData);
  SpiralGalaxy_eqFunction_8373(data, threadData);
  SpiralGalaxy_eqFunction_605(data, threadData);
  SpiralGalaxy_eqFunction_8372(data, threadData);
  SpiralGalaxy_eqFunction_607(data, threadData);
  SpiralGalaxy_eqFunction_8371(data, threadData);
  SpiralGalaxy_eqFunction_609(data, threadData);
  SpiralGalaxy_eqFunction_8384(data, threadData);
  SpiralGalaxy_eqFunction_8385(data, threadData);
  SpiralGalaxy_eqFunction_612(data, threadData);
  SpiralGalaxy_eqFunction_613(data, threadData);
  SpiralGalaxy_eqFunction_8386(data, threadData);
  SpiralGalaxy_eqFunction_8387(data, threadData);
  SpiralGalaxy_eqFunction_8390(data, threadData);
  SpiralGalaxy_eqFunction_8389(data, threadData);
  SpiralGalaxy_eqFunction_8388(data, threadData);
  SpiralGalaxy_eqFunction_619(data, threadData);
  SpiralGalaxy_eqFunction_8383(data, threadData);
  SpiralGalaxy_eqFunction_621(data, threadData);
  SpiralGalaxy_eqFunction_8382(data, threadData);
  SpiralGalaxy_eqFunction_623(data, threadData);
  SpiralGalaxy_eqFunction_8381(data, threadData);
  SpiralGalaxy_eqFunction_625(data, threadData);
  SpiralGalaxy_eqFunction_8394(data, threadData);
  SpiralGalaxy_eqFunction_8395(data, threadData);
  SpiralGalaxy_eqFunction_628(data, threadData);
  SpiralGalaxy_eqFunction_629(data, threadData);
  SpiralGalaxy_eqFunction_8396(data, threadData);
  SpiralGalaxy_eqFunction_8397(data, threadData);
  SpiralGalaxy_eqFunction_8400(data, threadData);
  SpiralGalaxy_eqFunction_8399(data, threadData);
  SpiralGalaxy_eqFunction_8398(data, threadData);
  SpiralGalaxy_eqFunction_635(data, threadData);
  SpiralGalaxy_eqFunction_8393(data, threadData);
  SpiralGalaxy_eqFunction_637(data, threadData);
  SpiralGalaxy_eqFunction_8392(data, threadData);
  SpiralGalaxy_eqFunction_639(data, threadData);
  SpiralGalaxy_eqFunction_8391(data, threadData);
  SpiralGalaxy_eqFunction_641(data, threadData);
  SpiralGalaxy_eqFunction_8404(data, threadData);
  SpiralGalaxy_eqFunction_8405(data, threadData);
  SpiralGalaxy_eqFunction_644(data, threadData);
  SpiralGalaxy_eqFunction_645(data, threadData);
  SpiralGalaxy_eqFunction_8406(data, threadData);
  SpiralGalaxy_eqFunction_8407(data, threadData);
  SpiralGalaxy_eqFunction_8410(data, threadData);
  SpiralGalaxy_eqFunction_8409(data, threadData);
  SpiralGalaxy_eqFunction_8408(data, threadData);
  SpiralGalaxy_eqFunction_651(data, threadData);
  SpiralGalaxy_eqFunction_8403(data, threadData);
  SpiralGalaxy_eqFunction_653(data, threadData);
  SpiralGalaxy_eqFunction_8402(data, threadData);
  SpiralGalaxy_eqFunction_655(data, threadData);
  SpiralGalaxy_eqFunction_8401(data, threadData);
  SpiralGalaxy_eqFunction_657(data, threadData);
  SpiralGalaxy_eqFunction_8414(data, threadData);
  SpiralGalaxy_eqFunction_8415(data, threadData);
  SpiralGalaxy_eqFunction_660(data, threadData);
  SpiralGalaxy_eqFunction_661(data, threadData);
  SpiralGalaxy_eqFunction_8416(data, threadData);
  SpiralGalaxy_eqFunction_8417(data, threadData);
  SpiralGalaxy_eqFunction_8420(data, threadData);
  SpiralGalaxy_eqFunction_8419(data, threadData);
  SpiralGalaxy_eqFunction_8418(data, threadData);
  SpiralGalaxy_eqFunction_667(data, threadData);
  SpiralGalaxy_eqFunction_8413(data, threadData);
  SpiralGalaxy_eqFunction_669(data, threadData);
  SpiralGalaxy_eqFunction_8412(data, threadData);
  SpiralGalaxy_eqFunction_671(data, threadData);
  SpiralGalaxy_eqFunction_8411(data, threadData);
  SpiralGalaxy_eqFunction_673(data, threadData);
  SpiralGalaxy_eqFunction_8424(data, threadData);
  SpiralGalaxy_eqFunction_8425(data, threadData);
  SpiralGalaxy_eqFunction_676(data, threadData);
  SpiralGalaxy_eqFunction_677(data, threadData);
  SpiralGalaxy_eqFunction_8426(data, threadData);
  SpiralGalaxy_eqFunction_8427(data, threadData);
  SpiralGalaxy_eqFunction_8430(data, threadData);
  SpiralGalaxy_eqFunction_8429(data, threadData);
  SpiralGalaxy_eqFunction_8428(data, threadData);
  SpiralGalaxy_eqFunction_683(data, threadData);
  SpiralGalaxy_eqFunction_8423(data, threadData);
  SpiralGalaxy_eqFunction_685(data, threadData);
  SpiralGalaxy_eqFunction_8422(data, threadData);
  SpiralGalaxy_eqFunction_687(data, threadData);
  SpiralGalaxy_eqFunction_8421(data, threadData);
  SpiralGalaxy_eqFunction_689(data, threadData);
  SpiralGalaxy_eqFunction_8434(data, threadData);
  SpiralGalaxy_eqFunction_8435(data, threadData);
  SpiralGalaxy_eqFunction_692(data, threadData);
  SpiralGalaxy_eqFunction_693(data, threadData);
  SpiralGalaxy_eqFunction_8436(data, threadData);
  SpiralGalaxy_eqFunction_8437(data, threadData);
  SpiralGalaxy_eqFunction_8440(data, threadData);
  SpiralGalaxy_eqFunction_8439(data, threadData);
  SpiralGalaxy_eqFunction_8438(data, threadData);
  SpiralGalaxy_eqFunction_699(data, threadData);
  SpiralGalaxy_eqFunction_8433(data, threadData);
  SpiralGalaxy_eqFunction_701(data, threadData);
  SpiralGalaxy_eqFunction_8432(data, threadData);
  SpiralGalaxy_eqFunction_703(data, threadData);
  SpiralGalaxy_eqFunction_8431(data, threadData);
  SpiralGalaxy_eqFunction_705(data, threadData);
  SpiralGalaxy_eqFunction_8444(data, threadData);
  SpiralGalaxy_eqFunction_8445(data, threadData);
  SpiralGalaxy_eqFunction_708(data, threadData);
  SpiralGalaxy_eqFunction_709(data, threadData);
  SpiralGalaxy_eqFunction_8446(data, threadData);
  SpiralGalaxy_eqFunction_8447(data, threadData);
  SpiralGalaxy_eqFunction_8450(data, threadData);
  SpiralGalaxy_eqFunction_8449(data, threadData);
  SpiralGalaxy_eqFunction_8448(data, threadData);
  SpiralGalaxy_eqFunction_715(data, threadData);
  SpiralGalaxy_eqFunction_8443(data, threadData);
  SpiralGalaxy_eqFunction_717(data, threadData);
  SpiralGalaxy_eqFunction_8442(data, threadData);
  SpiralGalaxy_eqFunction_719(data, threadData);
  SpiralGalaxy_eqFunction_8441(data, threadData);
  SpiralGalaxy_eqFunction_721(data, threadData);
  SpiralGalaxy_eqFunction_8454(data, threadData);
  SpiralGalaxy_eqFunction_8455(data, threadData);
  SpiralGalaxy_eqFunction_724(data, threadData);
  SpiralGalaxy_eqFunction_725(data, threadData);
  SpiralGalaxy_eqFunction_8456(data, threadData);
  SpiralGalaxy_eqFunction_8457(data, threadData);
  SpiralGalaxy_eqFunction_8460(data, threadData);
  SpiralGalaxy_eqFunction_8459(data, threadData);
  SpiralGalaxy_eqFunction_8458(data, threadData);
  SpiralGalaxy_eqFunction_731(data, threadData);
  SpiralGalaxy_eqFunction_8453(data, threadData);
  SpiralGalaxy_eqFunction_733(data, threadData);
  SpiralGalaxy_eqFunction_8452(data, threadData);
  SpiralGalaxy_eqFunction_735(data, threadData);
  SpiralGalaxy_eqFunction_8451(data, threadData);
  SpiralGalaxy_eqFunction_737(data, threadData);
  SpiralGalaxy_eqFunction_8464(data, threadData);
  SpiralGalaxy_eqFunction_8465(data, threadData);
  SpiralGalaxy_eqFunction_740(data, threadData);
  SpiralGalaxy_eqFunction_741(data, threadData);
  SpiralGalaxy_eqFunction_8466(data, threadData);
  SpiralGalaxy_eqFunction_8467(data, threadData);
  SpiralGalaxy_eqFunction_8470(data, threadData);
  SpiralGalaxy_eqFunction_8469(data, threadData);
  SpiralGalaxy_eqFunction_8468(data, threadData);
  SpiralGalaxy_eqFunction_747(data, threadData);
  SpiralGalaxy_eqFunction_8463(data, threadData);
  SpiralGalaxy_eqFunction_749(data, threadData);
  SpiralGalaxy_eqFunction_8462(data, threadData);
  SpiralGalaxy_eqFunction_751(data, threadData);
  SpiralGalaxy_eqFunction_8461(data, threadData);
  SpiralGalaxy_eqFunction_753(data, threadData);
  SpiralGalaxy_eqFunction_8474(data, threadData);
  SpiralGalaxy_eqFunction_8475(data, threadData);
  SpiralGalaxy_eqFunction_756(data, threadData);
  SpiralGalaxy_eqFunction_757(data, threadData);
  SpiralGalaxy_eqFunction_8476(data, threadData);
  SpiralGalaxy_eqFunction_8477(data, threadData);
  SpiralGalaxy_eqFunction_8480(data, threadData);
  SpiralGalaxy_eqFunction_8479(data, threadData);
  SpiralGalaxy_eqFunction_8478(data, threadData);
  SpiralGalaxy_eqFunction_763(data, threadData);
  SpiralGalaxy_eqFunction_8473(data, threadData);
  SpiralGalaxy_eqFunction_765(data, threadData);
  SpiralGalaxy_eqFunction_8472(data, threadData);
  SpiralGalaxy_eqFunction_767(data, threadData);
  SpiralGalaxy_eqFunction_8471(data, threadData);
  SpiralGalaxy_eqFunction_769(data, threadData);
  SpiralGalaxy_eqFunction_8484(data, threadData);
  SpiralGalaxy_eqFunction_8485(data, threadData);
  SpiralGalaxy_eqFunction_772(data, threadData);
  SpiralGalaxy_eqFunction_773(data, threadData);
  SpiralGalaxy_eqFunction_8486(data, threadData);
  SpiralGalaxy_eqFunction_8487(data, threadData);
  SpiralGalaxy_eqFunction_8490(data, threadData);
  SpiralGalaxy_eqFunction_8489(data, threadData);
  SpiralGalaxy_eqFunction_8488(data, threadData);
  SpiralGalaxy_eqFunction_779(data, threadData);
  SpiralGalaxy_eqFunction_8483(data, threadData);
  SpiralGalaxy_eqFunction_781(data, threadData);
  SpiralGalaxy_eqFunction_8482(data, threadData);
  SpiralGalaxy_eqFunction_783(data, threadData);
  SpiralGalaxy_eqFunction_8481(data, threadData);
  SpiralGalaxy_eqFunction_785(data, threadData);
  SpiralGalaxy_eqFunction_8494(data, threadData);
  SpiralGalaxy_eqFunction_8495(data, threadData);
  SpiralGalaxy_eqFunction_788(data, threadData);
  SpiralGalaxy_eqFunction_789(data, threadData);
  SpiralGalaxy_eqFunction_8496(data, threadData);
  SpiralGalaxy_eqFunction_8497(data, threadData);
  SpiralGalaxy_eqFunction_8500(data, threadData);
  SpiralGalaxy_eqFunction_8499(data, threadData);
  SpiralGalaxy_eqFunction_8498(data, threadData);
  SpiralGalaxy_eqFunction_795(data, threadData);
  SpiralGalaxy_eqFunction_8493(data, threadData);
  SpiralGalaxy_eqFunction_797(data, threadData);
  SpiralGalaxy_eqFunction_8492(data, threadData);
  SpiralGalaxy_eqFunction_799(data, threadData);
  SpiralGalaxy_eqFunction_8491(data, threadData);
  SpiralGalaxy_eqFunction_801(data, threadData);
  SpiralGalaxy_eqFunction_8504(data, threadData);
  SpiralGalaxy_eqFunction_8505(data, threadData);
  SpiralGalaxy_eqFunction_804(data, threadData);
  SpiralGalaxy_eqFunction_805(data, threadData);
  SpiralGalaxy_eqFunction_8506(data, threadData);
  SpiralGalaxy_eqFunction_8507(data, threadData);
  SpiralGalaxy_eqFunction_8510(data, threadData);
  SpiralGalaxy_eqFunction_8509(data, threadData);
  SpiralGalaxy_eqFunction_8508(data, threadData);
  SpiralGalaxy_eqFunction_811(data, threadData);
  SpiralGalaxy_eqFunction_8503(data, threadData);
  SpiralGalaxy_eqFunction_813(data, threadData);
  SpiralGalaxy_eqFunction_8502(data, threadData);
  SpiralGalaxy_eqFunction_815(data, threadData);
  SpiralGalaxy_eqFunction_8501(data, threadData);
  SpiralGalaxy_eqFunction_817(data, threadData);
  SpiralGalaxy_eqFunction_8514(data, threadData);
  SpiralGalaxy_eqFunction_8515(data, threadData);
  SpiralGalaxy_eqFunction_820(data, threadData);
  SpiralGalaxy_eqFunction_821(data, threadData);
  SpiralGalaxy_eqFunction_8516(data, threadData);
  SpiralGalaxy_eqFunction_8517(data, threadData);
  SpiralGalaxy_eqFunction_8520(data, threadData);
  SpiralGalaxy_eqFunction_8519(data, threadData);
  SpiralGalaxy_eqFunction_8518(data, threadData);
  SpiralGalaxy_eqFunction_827(data, threadData);
  SpiralGalaxy_eqFunction_8513(data, threadData);
  SpiralGalaxy_eqFunction_829(data, threadData);
  SpiralGalaxy_eqFunction_8512(data, threadData);
  SpiralGalaxy_eqFunction_831(data, threadData);
  SpiralGalaxy_eqFunction_8511(data, threadData);
  SpiralGalaxy_eqFunction_833(data, threadData);
  SpiralGalaxy_eqFunction_8524(data, threadData);
  SpiralGalaxy_eqFunction_8525(data, threadData);
  SpiralGalaxy_eqFunction_836(data, threadData);
  SpiralGalaxy_eqFunction_837(data, threadData);
  SpiralGalaxy_eqFunction_8526(data, threadData);
  SpiralGalaxy_eqFunction_8527(data, threadData);
  SpiralGalaxy_eqFunction_8530(data, threadData);
  SpiralGalaxy_eqFunction_8529(data, threadData);
  SpiralGalaxy_eqFunction_8528(data, threadData);
  SpiralGalaxy_eqFunction_843(data, threadData);
  SpiralGalaxy_eqFunction_8523(data, threadData);
  SpiralGalaxy_eqFunction_845(data, threadData);
  SpiralGalaxy_eqFunction_8522(data, threadData);
  SpiralGalaxy_eqFunction_847(data, threadData);
  SpiralGalaxy_eqFunction_8521(data, threadData);
  SpiralGalaxy_eqFunction_849(data, threadData);
  SpiralGalaxy_eqFunction_8534(data, threadData);
  SpiralGalaxy_eqFunction_8535(data, threadData);
  SpiralGalaxy_eqFunction_852(data, threadData);
  SpiralGalaxy_eqFunction_853(data, threadData);
  SpiralGalaxy_eqFunction_8536(data, threadData);
  SpiralGalaxy_eqFunction_8537(data, threadData);
  SpiralGalaxy_eqFunction_8540(data, threadData);
  SpiralGalaxy_eqFunction_8539(data, threadData);
  SpiralGalaxy_eqFunction_8538(data, threadData);
  SpiralGalaxy_eqFunction_859(data, threadData);
  SpiralGalaxy_eqFunction_8533(data, threadData);
  SpiralGalaxy_eqFunction_861(data, threadData);
  SpiralGalaxy_eqFunction_8532(data, threadData);
  SpiralGalaxy_eqFunction_863(data, threadData);
  SpiralGalaxy_eqFunction_8531(data, threadData);
  SpiralGalaxy_eqFunction_865(data, threadData);
  SpiralGalaxy_eqFunction_8544(data, threadData);
  SpiralGalaxy_eqFunction_8545(data, threadData);
  SpiralGalaxy_eqFunction_868(data, threadData);
  SpiralGalaxy_eqFunction_869(data, threadData);
  SpiralGalaxy_eqFunction_8546(data, threadData);
  SpiralGalaxy_eqFunction_8547(data, threadData);
  SpiralGalaxy_eqFunction_8550(data, threadData);
  SpiralGalaxy_eqFunction_8549(data, threadData);
  SpiralGalaxy_eqFunction_8548(data, threadData);
  SpiralGalaxy_eqFunction_875(data, threadData);
  SpiralGalaxy_eqFunction_8543(data, threadData);
  SpiralGalaxy_eqFunction_877(data, threadData);
  SpiralGalaxy_eqFunction_8542(data, threadData);
  SpiralGalaxy_eqFunction_879(data, threadData);
  SpiralGalaxy_eqFunction_8541(data, threadData);
  SpiralGalaxy_eqFunction_881(data, threadData);
  SpiralGalaxy_eqFunction_8554(data, threadData);
  SpiralGalaxy_eqFunction_8555(data, threadData);
  SpiralGalaxy_eqFunction_884(data, threadData);
  SpiralGalaxy_eqFunction_885(data, threadData);
  SpiralGalaxy_eqFunction_8556(data, threadData);
  SpiralGalaxy_eqFunction_8557(data, threadData);
  SpiralGalaxy_eqFunction_8560(data, threadData);
  SpiralGalaxy_eqFunction_8559(data, threadData);
  SpiralGalaxy_eqFunction_8558(data, threadData);
  SpiralGalaxy_eqFunction_891(data, threadData);
  SpiralGalaxy_eqFunction_8553(data, threadData);
  SpiralGalaxy_eqFunction_893(data, threadData);
  SpiralGalaxy_eqFunction_8552(data, threadData);
  SpiralGalaxy_eqFunction_895(data, threadData);
  SpiralGalaxy_eqFunction_8551(data, threadData);
  SpiralGalaxy_eqFunction_897(data, threadData);
  SpiralGalaxy_eqFunction_8564(data, threadData);
  SpiralGalaxy_eqFunction_8565(data, threadData);
  SpiralGalaxy_eqFunction_900(data, threadData);
  SpiralGalaxy_eqFunction_901(data, threadData);
  SpiralGalaxy_eqFunction_8566(data, threadData);
  SpiralGalaxy_eqFunction_8567(data, threadData);
  SpiralGalaxy_eqFunction_8570(data, threadData);
  SpiralGalaxy_eqFunction_8569(data, threadData);
  SpiralGalaxy_eqFunction_8568(data, threadData);
  SpiralGalaxy_eqFunction_907(data, threadData);
  SpiralGalaxy_eqFunction_8563(data, threadData);
  SpiralGalaxy_eqFunction_909(data, threadData);
  SpiralGalaxy_eqFunction_8562(data, threadData);
  SpiralGalaxy_eqFunction_911(data, threadData);
  SpiralGalaxy_eqFunction_8561(data, threadData);
  SpiralGalaxy_eqFunction_913(data, threadData);
  SpiralGalaxy_eqFunction_8574(data, threadData);
  SpiralGalaxy_eqFunction_8575(data, threadData);
  SpiralGalaxy_eqFunction_916(data, threadData);
  SpiralGalaxy_eqFunction_917(data, threadData);
  SpiralGalaxy_eqFunction_8576(data, threadData);
  SpiralGalaxy_eqFunction_8577(data, threadData);
  SpiralGalaxy_eqFunction_8580(data, threadData);
  SpiralGalaxy_eqFunction_8579(data, threadData);
  SpiralGalaxy_eqFunction_8578(data, threadData);
  SpiralGalaxy_eqFunction_923(data, threadData);
  SpiralGalaxy_eqFunction_8573(data, threadData);
  SpiralGalaxy_eqFunction_925(data, threadData);
  SpiralGalaxy_eqFunction_8572(data, threadData);
  SpiralGalaxy_eqFunction_927(data, threadData);
  SpiralGalaxy_eqFunction_8571(data, threadData);
  SpiralGalaxy_eqFunction_929(data, threadData);
  SpiralGalaxy_eqFunction_8584(data, threadData);
  SpiralGalaxy_eqFunction_8585(data, threadData);
  SpiralGalaxy_eqFunction_932(data, threadData);
  SpiralGalaxy_eqFunction_933(data, threadData);
  SpiralGalaxy_eqFunction_8586(data, threadData);
  SpiralGalaxy_eqFunction_8587(data, threadData);
  SpiralGalaxy_eqFunction_8590(data, threadData);
  SpiralGalaxy_eqFunction_8589(data, threadData);
  SpiralGalaxy_eqFunction_8588(data, threadData);
  SpiralGalaxy_eqFunction_939(data, threadData);
  SpiralGalaxy_eqFunction_8583(data, threadData);
  SpiralGalaxy_eqFunction_941(data, threadData);
  SpiralGalaxy_eqFunction_8582(data, threadData);
  SpiralGalaxy_eqFunction_943(data, threadData);
  SpiralGalaxy_eqFunction_8581(data, threadData);
  SpiralGalaxy_eqFunction_945(data, threadData);
  SpiralGalaxy_eqFunction_8594(data, threadData);
  SpiralGalaxy_eqFunction_8595(data, threadData);
  SpiralGalaxy_eqFunction_948(data, threadData);
  SpiralGalaxy_eqFunction_949(data, threadData);
  SpiralGalaxy_eqFunction_8596(data, threadData);
  SpiralGalaxy_eqFunction_8597(data, threadData);
  SpiralGalaxy_eqFunction_8600(data, threadData);
  SpiralGalaxy_eqFunction_8599(data, threadData);
  SpiralGalaxy_eqFunction_8598(data, threadData);
  SpiralGalaxy_eqFunction_955(data, threadData);
  SpiralGalaxy_eqFunction_8593(data, threadData);
  SpiralGalaxy_eqFunction_957(data, threadData);
  SpiralGalaxy_eqFunction_8592(data, threadData);
  SpiralGalaxy_eqFunction_959(data, threadData);
  SpiralGalaxy_eqFunction_8591(data, threadData);
  SpiralGalaxy_eqFunction_961(data, threadData);
  SpiralGalaxy_eqFunction_8604(data, threadData);
  SpiralGalaxy_eqFunction_8605(data, threadData);
  SpiralGalaxy_eqFunction_964(data, threadData);
  SpiralGalaxy_eqFunction_965(data, threadData);
  SpiralGalaxy_eqFunction_8606(data, threadData);
  SpiralGalaxy_eqFunction_8607(data, threadData);
  SpiralGalaxy_eqFunction_8610(data, threadData);
  SpiralGalaxy_eqFunction_8609(data, threadData);
  SpiralGalaxy_eqFunction_8608(data, threadData);
  SpiralGalaxy_eqFunction_971(data, threadData);
  SpiralGalaxy_eqFunction_8603(data, threadData);
  SpiralGalaxy_eqFunction_973(data, threadData);
  SpiralGalaxy_eqFunction_8602(data, threadData);
  SpiralGalaxy_eqFunction_975(data, threadData);
  SpiralGalaxy_eqFunction_8601(data, threadData);
  SpiralGalaxy_eqFunction_977(data, threadData);
  SpiralGalaxy_eqFunction_8614(data, threadData);
  SpiralGalaxy_eqFunction_8615(data, threadData);
  SpiralGalaxy_eqFunction_980(data, threadData);
  SpiralGalaxy_eqFunction_981(data, threadData);
  SpiralGalaxy_eqFunction_8616(data, threadData);
  SpiralGalaxy_eqFunction_8617(data, threadData);
  SpiralGalaxy_eqFunction_8620(data, threadData);
  SpiralGalaxy_eqFunction_8619(data, threadData);
  SpiralGalaxy_eqFunction_8618(data, threadData);
  SpiralGalaxy_eqFunction_987(data, threadData);
  SpiralGalaxy_eqFunction_8613(data, threadData);
  SpiralGalaxy_eqFunction_989(data, threadData);
  SpiralGalaxy_eqFunction_8612(data, threadData);
  SpiralGalaxy_eqFunction_991(data, threadData);
  SpiralGalaxy_eqFunction_8611(data, threadData);
  SpiralGalaxy_eqFunction_993(data, threadData);
  SpiralGalaxy_eqFunction_8624(data, threadData);
  SpiralGalaxy_eqFunction_8625(data, threadData);
  SpiralGalaxy_eqFunction_996(data, threadData);
  SpiralGalaxy_eqFunction_997(data, threadData);
  SpiralGalaxy_eqFunction_8626(data, threadData);
  SpiralGalaxy_eqFunction_8627(data, threadData);
  SpiralGalaxy_eqFunction_8630(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif