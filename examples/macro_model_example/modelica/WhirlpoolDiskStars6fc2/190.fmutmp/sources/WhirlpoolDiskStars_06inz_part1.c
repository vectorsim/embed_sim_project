#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void WhirlpoolDiskStars_eqFunction_2829(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2828(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2826(DATA *data, threadData_t *threadData);


/*
equation index: 431
type: SIMPLE_ASSIGN
vz[27] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,431};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[346]] /* vz[27] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2821(DATA *data, threadData_t *threadData);


/*
equation index: 433
type: SIMPLE_ASSIGN
z[28] = -2.6
*/
void WhirlpoolDiskStars_eqFunction_433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,433};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[827]] /* z[28] STATE(1,vz[28]) */) = -2.6;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2834(DATA *data, threadData_t *threadData);


/*
equation index: 435
type: SIMPLE_ASSIGN
y[28] = r_init[28] * sin(theta[28] + armOffset[28])
*/
void WhirlpoolDiskStars_eqFunction_435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[667]] /* y[28] STATE(1,vy[28]) */) = ((data->simulationInfo->realParameter[192] /* r_init[28] PARAM */)) * (sin((data->simulationInfo->realParameter[352] /* theta[28] PARAM */) + (data->simulationInfo->realParameter[30] /* armOffset[28] PARAM */)));
  TRACE_POP
}

/*
equation index: 436
type: SIMPLE_ASSIGN
vx[28] = (-y[28]) * sqrt(G * Md / r_init[28] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,436};
  modelica_real tmp108;
  modelica_real tmp109;
  tmp108 = (data->simulationInfo->realParameter[192] /* r_init[28] PARAM */);
  tmp109 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp108 * tmp108 * tmp108),"r_init[28] ^ 3.0",equationIndexes);
  if(!(tmp109 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[28] ^ 3.0) was %g should be >= 0", tmp109);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* vx[28] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[667]] /* y[28] STATE(1,vy[28]) */))) * (sqrt(tmp109));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2833(DATA *data, threadData_t *threadData);


/*
equation index: 438
type: SIMPLE_ASSIGN
x[28] = r_init[28] * cos(theta[28] + armOffset[28])
*/
void WhirlpoolDiskStars_eqFunction_438(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,438};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[507]] /* x[28] STATE(1,vx[28]) */) = ((data->simulationInfo->realParameter[192] /* r_init[28] PARAM */)) * (cos((data->simulationInfo->realParameter[352] /* theta[28] PARAM */) + (data->simulationInfo->realParameter[30] /* armOffset[28] PARAM */)));
  TRACE_POP
}

/*
equation index: 439
type: SIMPLE_ASSIGN
vy[28] = x[28] * sqrt(G * Md / r_init[28] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,439};
  modelica_real tmp110;
  modelica_real tmp111;
  tmp110 = (data->simulationInfo->realParameter[192] /* r_init[28] PARAM */);
  tmp111 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp110 * tmp110 * tmp110),"r_init[28] ^ 3.0",equationIndexes);
  if(!(tmp111 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[28] ^ 3.0) was %g should be >= 0", tmp111);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[187]] /* vy[28] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[507]] /* x[28] STATE(1,vx[28]) */)) * (sqrt(tmp111));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2832(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2835(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2837(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2840(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2839(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2838(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2836(DATA *data, threadData_t *threadData);


/*
equation index: 447
type: SIMPLE_ASSIGN
vz[28] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,447};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[347]] /* vz[28] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2831(DATA *data, threadData_t *threadData);


/*
equation index: 449
type: SIMPLE_ASSIGN
z[29] = -2.5500000000000003
*/
void WhirlpoolDiskStars_eqFunction_449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,449};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[828]] /* z[29] STATE(1,vz[29]) */) = -2.5500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2844(DATA *data, threadData_t *threadData);


/*
equation index: 451
type: SIMPLE_ASSIGN
y[29] = r_init[29] * sin(theta[29] + armOffset[29])
*/
void WhirlpoolDiskStars_eqFunction_451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[668]] /* y[29] STATE(1,vy[29]) */) = ((data->simulationInfo->realParameter[193] /* r_init[29] PARAM */)) * (sin((data->simulationInfo->realParameter[353] /* theta[29] PARAM */) + (data->simulationInfo->realParameter[31] /* armOffset[29] PARAM */)));
  TRACE_POP
}

/*
equation index: 452
type: SIMPLE_ASSIGN
vx[29] = (-y[29]) * sqrt(G * Md / r_init[29] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,452};
  modelica_real tmp112;
  modelica_real tmp113;
  tmp112 = (data->simulationInfo->realParameter[193] /* r_init[29] PARAM */);
  tmp113 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp112 * tmp112 * tmp112),"r_init[29] ^ 3.0",equationIndexes);
  if(!(tmp113 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[29] ^ 3.0) was %g should be >= 0", tmp113);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* vx[29] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[668]] /* y[29] STATE(1,vy[29]) */))) * (sqrt(tmp113));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2843(DATA *data, threadData_t *threadData);


/*
equation index: 454
type: SIMPLE_ASSIGN
x[29] = r_init[29] * cos(theta[29] + armOffset[29])
*/
void WhirlpoolDiskStars_eqFunction_454(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,454};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[508]] /* x[29] STATE(1,vx[29]) */) = ((data->simulationInfo->realParameter[193] /* r_init[29] PARAM */)) * (cos((data->simulationInfo->realParameter[353] /* theta[29] PARAM */) + (data->simulationInfo->realParameter[31] /* armOffset[29] PARAM */)));
  TRACE_POP
}

/*
equation index: 455
type: SIMPLE_ASSIGN
vy[29] = x[29] * sqrt(G * Md / r_init[29] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,455};
  modelica_real tmp114;
  modelica_real tmp115;
  tmp114 = (data->simulationInfo->realParameter[193] /* r_init[29] PARAM */);
  tmp115 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp114 * tmp114 * tmp114),"r_init[29] ^ 3.0",equationIndexes);
  if(!(tmp115 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[29] ^ 3.0) was %g should be >= 0", tmp115);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[188]] /* vy[29] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[508]] /* x[29] STATE(1,vx[29]) */)) * (sqrt(tmp115));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2842(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2845(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2847(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2850(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2849(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2848(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2846(DATA *data, threadData_t *threadData);


/*
equation index: 463
type: SIMPLE_ASSIGN
vz[29] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,463};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[348]] /* vz[29] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2841(DATA *data, threadData_t *threadData);


/*
equation index: 465
type: SIMPLE_ASSIGN
z[30] = -2.5
*/
void WhirlpoolDiskStars_eqFunction_465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,465};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[829]] /* z[30] STATE(1,vz[30]) */) = -2.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2854(DATA *data, threadData_t *threadData);


/*
equation index: 467
type: SIMPLE_ASSIGN
y[30] = r_init[30] * sin(theta[30] + armOffset[30])
*/
void WhirlpoolDiskStars_eqFunction_467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[669]] /* y[30] STATE(1,vy[30]) */) = ((data->simulationInfo->realParameter[194] /* r_init[30] PARAM */)) * (sin((data->simulationInfo->realParameter[354] /* theta[30] PARAM */) + (data->simulationInfo->realParameter[32] /* armOffset[30] PARAM */)));
  TRACE_POP
}

/*
equation index: 468
type: SIMPLE_ASSIGN
vx[30] = (-y[30]) * sqrt(G * Md / r_init[30] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,468};
  modelica_real tmp116;
  modelica_real tmp117;
  tmp116 = (data->simulationInfo->realParameter[194] /* r_init[30] PARAM */);
  tmp117 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp116 * tmp116 * tmp116),"r_init[30] ^ 3.0",equationIndexes);
  if(!(tmp117 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[30] ^ 3.0) was %g should be >= 0", tmp117);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* vx[30] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[669]] /* y[30] STATE(1,vy[30]) */))) * (sqrt(tmp117));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2853(DATA *data, threadData_t *threadData);


/*
equation index: 470
type: SIMPLE_ASSIGN
x[30] = r_init[30] * cos(theta[30] + armOffset[30])
*/
void WhirlpoolDiskStars_eqFunction_470(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,470};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[509]] /* x[30] STATE(1,vx[30]) */) = ((data->simulationInfo->realParameter[194] /* r_init[30] PARAM */)) * (cos((data->simulationInfo->realParameter[354] /* theta[30] PARAM */) + (data->simulationInfo->realParameter[32] /* armOffset[30] PARAM */)));
  TRACE_POP
}

/*
equation index: 471
type: SIMPLE_ASSIGN
vy[30] = x[30] * sqrt(G * Md / r_init[30] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,471};
  modelica_real tmp118;
  modelica_real tmp119;
  tmp118 = (data->simulationInfo->realParameter[194] /* r_init[30] PARAM */);
  tmp119 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp118 * tmp118 * tmp118),"r_init[30] ^ 3.0",equationIndexes);
  if(!(tmp119 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[30] ^ 3.0) was %g should be >= 0", tmp119);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[189]] /* vy[30] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[509]] /* x[30] STATE(1,vx[30]) */)) * (sqrt(tmp119));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2852(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2855(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2857(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2860(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2859(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2858(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2856(DATA *data, threadData_t *threadData);


/*
equation index: 479
type: SIMPLE_ASSIGN
vz[30] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,479};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[349]] /* vz[30] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2851(DATA *data, threadData_t *threadData);


/*
equation index: 481
type: SIMPLE_ASSIGN
z[31] = -2.45
*/
void WhirlpoolDiskStars_eqFunction_481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,481};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[830]] /* z[31] STATE(1,vz[31]) */) = -2.45;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2864(DATA *data, threadData_t *threadData);


/*
equation index: 483
type: SIMPLE_ASSIGN
y[31] = r_init[31] * sin(theta[31] + armOffset[31])
*/
void WhirlpoolDiskStars_eqFunction_483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[670]] /* y[31] STATE(1,vy[31]) */) = ((data->simulationInfo->realParameter[195] /* r_init[31] PARAM */)) * (sin((data->simulationInfo->realParameter[355] /* theta[31] PARAM */) + (data->simulationInfo->realParameter[33] /* armOffset[31] PARAM */)));
  TRACE_POP
}

/*
equation index: 484
type: SIMPLE_ASSIGN
vx[31] = (-y[31]) * sqrt(G * Md / r_init[31] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,484};
  modelica_real tmp120;
  modelica_real tmp121;
  tmp120 = (data->simulationInfo->realParameter[195] /* r_init[31] PARAM */);
  tmp121 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp120 * tmp120 * tmp120),"r_init[31] ^ 3.0",equationIndexes);
  if(!(tmp121 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[31] ^ 3.0) was %g should be >= 0", tmp121);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* vx[31] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[670]] /* y[31] STATE(1,vy[31]) */))) * (sqrt(tmp121));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2863(DATA *data, threadData_t *threadData);


/*
equation index: 486
type: SIMPLE_ASSIGN
x[31] = r_init[31] * cos(theta[31] + armOffset[31])
*/
void WhirlpoolDiskStars_eqFunction_486(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,486};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[510]] /* x[31] STATE(1,vx[31]) */) = ((data->simulationInfo->realParameter[195] /* r_init[31] PARAM */)) * (cos((data->simulationInfo->realParameter[355] /* theta[31] PARAM */) + (data->simulationInfo->realParameter[33] /* armOffset[31] PARAM */)));
  TRACE_POP
}

/*
equation index: 487
type: SIMPLE_ASSIGN
vy[31] = x[31] * sqrt(G * Md / r_init[31] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,487};
  modelica_real tmp122;
  modelica_real tmp123;
  tmp122 = (data->simulationInfo->realParameter[195] /* r_init[31] PARAM */);
  tmp123 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp122 * tmp122 * tmp122),"r_init[31] ^ 3.0",equationIndexes);
  if(!(tmp123 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[31] ^ 3.0) was %g should be >= 0", tmp123);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[190]] /* vy[31] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[510]] /* x[31] STATE(1,vx[31]) */)) * (sqrt(tmp123));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2862(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2865(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2867(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2870(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2869(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2868(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2866(DATA *data, threadData_t *threadData);


/*
equation index: 495
type: SIMPLE_ASSIGN
vz[31] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,495};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[350]] /* vz[31] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2861(DATA *data, threadData_t *threadData);


/*
equation index: 497
type: SIMPLE_ASSIGN
z[32] = -2.4000000000000004
*/
void WhirlpoolDiskStars_eqFunction_497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,497};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[831]] /* z[32] STATE(1,vz[32]) */) = -2.4000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2874(DATA *data, threadData_t *threadData);


/*
equation index: 499
type: SIMPLE_ASSIGN
y[32] = r_init[32] * sin(theta[32] + armOffset[32])
*/
void WhirlpoolDiskStars_eqFunction_499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,499};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[671]] /* y[32] STATE(1,vy[32]) */) = ((data->simulationInfo->realParameter[196] /* r_init[32] PARAM */)) * (sin((data->simulationInfo->realParameter[356] /* theta[32] PARAM */) + (data->simulationInfo->realParameter[34] /* armOffset[32] PARAM */)));
  TRACE_POP
}

/*
equation index: 500
type: SIMPLE_ASSIGN
vx[32] = (-y[32]) * sqrt(G * Md / r_init[32] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,500};
  modelica_real tmp124;
  modelica_real tmp125;
  tmp124 = (data->simulationInfo->realParameter[196] /* r_init[32] PARAM */);
  tmp125 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp124 * tmp124 * tmp124),"r_init[32] ^ 3.0",equationIndexes);
  if(!(tmp125 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[32] ^ 3.0) was %g should be >= 0", tmp125);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* vx[32] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[671]] /* y[32] STATE(1,vy[32]) */))) * (sqrt(tmp125));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2873(DATA *data, threadData_t *threadData);


/*
equation index: 502
type: SIMPLE_ASSIGN
x[32] = r_init[32] * cos(theta[32] + armOffset[32])
*/
void WhirlpoolDiskStars_eqFunction_502(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,502};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[511]] /* x[32] STATE(1,vx[32]) */) = ((data->simulationInfo->realParameter[196] /* r_init[32] PARAM */)) * (cos((data->simulationInfo->realParameter[356] /* theta[32] PARAM */) + (data->simulationInfo->realParameter[34] /* armOffset[32] PARAM */)));
  TRACE_POP
}

/*
equation index: 503
type: SIMPLE_ASSIGN
vy[32] = x[32] * sqrt(G * Md / r_init[32] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,503};
  modelica_real tmp126;
  modelica_real tmp127;
  tmp126 = (data->simulationInfo->realParameter[196] /* r_init[32] PARAM */);
  tmp127 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp126 * tmp126 * tmp126),"r_init[32] ^ 3.0",equationIndexes);
  if(!(tmp127 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[32] ^ 3.0) was %g should be >= 0", tmp127);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[191]] /* vy[32] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[511]] /* x[32] STATE(1,vx[32]) */)) * (sqrt(tmp127));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2872(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2875(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2877(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2880(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2879(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2878(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2876(DATA *data, threadData_t *threadData);


/*
equation index: 511
type: SIMPLE_ASSIGN
vz[32] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,511};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[351]] /* vz[32] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2871(DATA *data, threadData_t *threadData);


/*
equation index: 513
type: SIMPLE_ASSIGN
z[33] = -2.35
*/
void WhirlpoolDiskStars_eqFunction_513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,513};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[832]] /* z[33] STATE(1,vz[33]) */) = -2.35;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2884(DATA *data, threadData_t *threadData);


/*
equation index: 515
type: SIMPLE_ASSIGN
y[33] = r_init[33] * sin(theta[33] + armOffset[33])
*/
void WhirlpoolDiskStars_eqFunction_515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,515};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[672]] /* y[33] STATE(1,vy[33]) */) = ((data->simulationInfo->realParameter[197] /* r_init[33] PARAM */)) * (sin((data->simulationInfo->realParameter[357] /* theta[33] PARAM */) + (data->simulationInfo->realParameter[35] /* armOffset[33] PARAM */)));
  TRACE_POP
}

/*
equation index: 516
type: SIMPLE_ASSIGN
vx[33] = (-y[33]) * sqrt(G * Md / r_init[33] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,516};
  modelica_real tmp128;
  modelica_real tmp129;
  tmp128 = (data->simulationInfo->realParameter[197] /* r_init[33] PARAM */);
  tmp129 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp128 * tmp128 * tmp128),"r_init[33] ^ 3.0",equationIndexes);
  if(!(tmp129 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[33] ^ 3.0) was %g should be >= 0", tmp129);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* vx[33] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[672]] /* y[33] STATE(1,vy[33]) */))) * (sqrt(tmp129));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2883(DATA *data, threadData_t *threadData);


/*
equation index: 518
type: SIMPLE_ASSIGN
x[33] = r_init[33] * cos(theta[33] + armOffset[33])
*/
void WhirlpoolDiskStars_eqFunction_518(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,518};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[512]] /* x[33] STATE(1,vx[33]) */) = ((data->simulationInfo->realParameter[197] /* r_init[33] PARAM */)) * (cos((data->simulationInfo->realParameter[357] /* theta[33] PARAM */) + (data->simulationInfo->realParameter[35] /* armOffset[33] PARAM */)));
  TRACE_POP
}

/*
equation index: 519
type: SIMPLE_ASSIGN
vy[33] = x[33] * sqrt(G * Md / r_init[33] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,519};
  modelica_real tmp130;
  modelica_real tmp131;
  tmp130 = (data->simulationInfo->realParameter[197] /* r_init[33] PARAM */);
  tmp131 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp130 * tmp130 * tmp130),"r_init[33] ^ 3.0",equationIndexes);
  if(!(tmp131 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[33] ^ 3.0) was %g should be >= 0", tmp131);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[192]] /* vy[33] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[512]] /* x[33] STATE(1,vx[33]) */)) * (sqrt(tmp131));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2882(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2885(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2887(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2890(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2889(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2888(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2886(DATA *data, threadData_t *threadData);


/*
equation index: 527
type: SIMPLE_ASSIGN
vz[33] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[352]] /* vz[33] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2881(DATA *data, threadData_t *threadData);


/*
equation index: 529
type: SIMPLE_ASSIGN
z[34] = -2.3000000000000003
*/
void WhirlpoolDiskStars_eqFunction_529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,529};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[833]] /* z[34] STATE(1,vz[34]) */) = -2.3000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2894(DATA *data, threadData_t *threadData);


/*
equation index: 531
type: SIMPLE_ASSIGN
y[34] = r_init[34] * sin(theta[34] + armOffset[34])
*/
void WhirlpoolDiskStars_eqFunction_531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,531};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[673]] /* y[34] STATE(1,vy[34]) */) = ((data->simulationInfo->realParameter[198] /* r_init[34] PARAM */)) * (sin((data->simulationInfo->realParameter[358] /* theta[34] PARAM */) + (data->simulationInfo->realParameter[36] /* armOffset[34] PARAM */)));
  TRACE_POP
}

/*
equation index: 532
type: SIMPLE_ASSIGN
vx[34] = (-y[34]) * sqrt(G * Md / r_init[34] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,532};
  modelica_real tmp132;
  modelica_real tmp133;
  tmp132 = (data->simulationInfo->realParameter[198] /* r_init[34] PARAM */);
  tmp133 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp132 * tmp132 * tmp132),"r_init[34] ^ 3.0",equationIndexes);
  if(!(tmp133 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[34] ^ 3.0) was %g should be >= 0", tmp133);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* vx[34] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[673]] /* y[34] STATE(1,vy[34]) */))) * (sqrt(tmp133));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2893(DATA *data, threadData_t *threadData);


/*
equation index: 534
type: SIMPLE_ASSIGN
x[34] = r_init[34] * cos(theta[34] + armOffset[34])
*/
void WhirlpoolDiskStars_eqFunction_534(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,534};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[513]] /* x[34] STATE(1,vx[34]) */) = ((data->simulationInfo->realParameter[198] /* r_init[34] PARAM */)) * (cos((data->simulationInfo->realParameter[358] /* theta[34] PARAM */) + (data->simulationInfo->realParameter[36] /* armOffset[34] PARAM */)));
  TRACE_POP
}

/*
equation index: 535
type: SIMPLE_ASSIGN
vy[34] = x[34] * sqrt(G * Md / r_init[34] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,535};
  modelica_real tmp134;
  modelica_real tmp135;
  tmp134 = (data->simulationInfo->realParameter[198] /* r_init[34] PARAM */);
  tmp135 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp134 * tmp134 * tmp134),"r_init[34] ^ 3.0",equationIndexes);
  if(!(tmp135 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[34] ^ 3.0) was %g should be >= 0", tmp135);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[193]] /* vy[34] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[513]] /* x[34] STATE(1,vx[34]) */)) * (sqrt(tmp135));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2892(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2895(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2897(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2900(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2899(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2898(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2896(DATA *data, threadData_t *threadData);


/*
equation index: 543
type: SIMPLE_ASSIGN
vz[34] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,543};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[353]] /* vz[34] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2891(DATA *data, threadData_t *threadData);


/*
equation index: 545
type: SIMPLE_ASSIGN
z[35] = -2.25
*/
void WhirlpoolDiskStars_eqFunction_545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,545};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[834]] /* z[35] STATE(1,vz[35]) */) = -2.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2904(DATA *data, threadData_t *threadData);


/*
equation index: 547
type: SIMPLE_ASSIGN
y[35] = r_init[35] * sin(theta[35] + armOffset[35])
*/
void WhirlpoolDiskStars_eqFunction_547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,547};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[674]] /* y[35] STATE(1,vy[35]) */) = ((data->simulationInfo->realParameter[199] /* r_init[35] PARAM */)) * (sin((data->simulationInfo->realParameter[359] /* theta[35] PARAM */) + (data->simulationInfo->realParameter[37] /* armOffset[35] PARAM */)));
  TRACE_POP
}

/*
equation index: 548
type: SIMPLE_ASSIGN
vx[35] = (-y[35]) * sqrt(G * Md / r_init[35] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,548};
  modelica_real tmp136;
  modelica_real tmp137;
  tmp136 = (data->simulationInfo->realParameter[199] /* r_init[35] PARAM */);
  tmp137 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp136 * tmp136 * tmp136),"r_init[35] ^ 3.0",equationIndexes);
  if(!(tmp137 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[35] ^ 3.0) was %g should be >= 0", tmp137);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* vx[35] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[674]] /* y[35] STATE(1,vy[35]) */))) * (sqrt(tmp137));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2903(DATA *data, threadData_t *threadData);


/*
equation index: 550
type: SIMPLE_ASSIGN
x[35] = r_init[35] * cos(theta[35] + armOffset[35])
*/
void WhirlpoolDiskStars_eqFunction_550(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,550};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[514]] /* x[35] STATE(1,vx[35]) */) = ((data->simulationInfo->realParameter[199] /* r_init[35] PARAM */)) * (cos((data->simulationInfo->realParameter[359] /* theta[35] PARAM */) + (data->simulationInfo->realParameter[37] /* armOffset[35] PARAM */)));
  TRACE_POP
}

/*
equation index: 551
type: SIMPLE_ASSIGN
vy[35] = x[35] * sqrt(G * Md / r_init[35] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,551};
  modelica_real tmp138;
  modelica_real tmp139;
  tmp138 = (data->simulationInfo->realParameter[199] /* r_init[35] PARAM */);
  tmp139 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp138 * tmp138 * tmp138),"r_init[35] ^ 3.0",equationIndexes);
  if(!(tmp139 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[35] ^ 3.0) was %g should be >= 0", tmp139);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[194]] /* vy[35] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[514]] /* x[35] STATE(1,vx[35]) */)) * (sqrt(tmp139));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2902(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2905(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2907(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2910(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2909(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2908(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2906(DATA *data, threadData_t *threadData);


/*
equation index: 559
type: SIMPLE_ASSIGN
vz[35] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,559};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[354]] /* vz[35] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2901(DATA *data, threadData_t *threadData);


/*
equation index: 561
type: SIMPLE_ASSIGN
z[36] = -2.2
*/
void WhirlpoolDiskStars_eqFunction_561(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,561};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[835]] /* z[36] STATE(1,vz[36]) */) = -2.2;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2914(DATA *data, threadData_t *threadData);


/*
equation index: 563
type: SIMPLE_ASSIGN
y[36] = r_init[36] * sin(theta[36] + armOffset[36])
*/
void WhirlpoolDiskStars_eqFunction_563(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,563};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[675]] /* y[36] STATE(1,vy[36]) */) = ((data->simulationInfo->realParameter[200] /* r_init[36] PARAM */)) * (sin((data->simulationInfo->realParameter[360] /* theta[36] PARAM */) + (data->simulationInfo->realParameter[38] /* armOffset[36] PARAM */)));
  TRACE_POP
}

/*
equation index: 564
type: SIMPLE_ASSIGN
vx[36] = (-y[36]) * sqrt(G * Md / r_init[36] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_564(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,564};
  modelica_real tmp140;
  modelica_real tmp141;
  tmp140 = (data->simulationInfo->realParameter[200] /* r_init[36] PARAM */);
  tmp141 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp140 * tmp140 * tmp140),"r_init[36] ^ 3.0",equationIndexes);
  if(!(tmp141 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[36] ^ 3.0) was %g should be >= 0", tmp141);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* vx[36] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[675]] /* y[36] STATE(1,vy[36]) */))) * (sqrt(tmp141));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2913(DATA *data, threadData_t *threadData);


/*
equation index: 566
type: SIMPLE_ASSIGN
x[36] = r_init[36] * cos(theta[36] + armOffset[36])
*/
void WhirlpoolDiskStars_eqFunction_566(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,566};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[515]] /* x[36] STATE(1,vx[36]) */) = ((data->simulationInfo->realParameter[200] /* r_init[36] PARAM */)) * (cos((data->simulationInfo->realParameter[360] /* theta[36] PARAM */) + (data->simulationInfo->realParameter[38] /* armOffset[36] PARAM */)));
  TRACE_POP
}

/*
equation index: 567
type: SIMPLE_ASSIGN
vy[36] = x[36] * sqrt(G * Md / r_init[36] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,567};
  modelica_real tmp142;
  modelica_real tmp143;
  tmp142 = (data->simulationInfo->realParameter[200] /* r_init[36] PARAM */);
  tmp143 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp142 * tmp142 * tmp142),"r_init[36] ^ 3.0",equationIndexes);
  if(!(tmp143 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[36] ^ 3.0) was %g should be >= 0", tmp143);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[195]] /* vy[36] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[515]] /* x[36] STATE(1,vx[36]) */)) * (sqrt(tmp143));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2912(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2915(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2917(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2920(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2919(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2918(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2916(DATA *data, threadData_t *threadData);


/*
equation index: 575
type: SIMPLE_ASSIGN
vz[36] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,575};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[355]] /* vz[36] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2911(DATA *data, threadData_t *threadData);


/*
equation index: 577
type: SIMPLE_ASSIGN
z[37] = -2.15
*/
void WhirlpoolDiskStars_eqFunction_577(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,577};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[836]] /* z[37] STATE(1,vz[37]) */) = -2.15;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2924(DATA *data, threadData_t *threadData);


/*
equation index: 579
type: SIMPLE_ASSIGN
y[37] = r_init[37] * sin(theta[37] + armOffset[37])
*/
void WhirlpoolDiskStars_eqFunction_579(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,579};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[676]] /* y[37] STATE(1,vy[37]) */) = ((data->simulationInfo->realParameter[201] /* r_init[37] PARAM */)) * (sin((data->simulationInfo->realParameter[361] /* theta[37] PARAM */) + (data->simulationInfo->realParameter[39] /* armOffset[37] PARAM */)));
  TRACE_POP
}

/*
equation index: 580
type: SIMPLE_ASSIGN
vx[37] = (-y[37]) * sqrt(G * Md / r_init[37] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_580(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,580};
  modelica_real tmp144;
  modelica_real tmp145;
  tmp144 = (data->simulationInfo->realParameter[201] /* r_init[37] PARAM */);
  tmp145 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp144 * tmp144 * tmp144),"r_init[37] ^ 3.0",equationIndexes);
  if(!(tmp145 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[37] ^ 3.0) was %g should be >= 0", tmp145);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* vx[37] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[676]] /* y[37] STATE(1,vy[37]) */))) * (sqrt(tmp145));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2923(DATA *data, threadData_t *threadData);


/*
equation index: 582
type: SIMPLE_ASSIGN
x[37] = r_init[37] * cos(theta[37] + armOffset[37])
*/
void WhirlpoolDiskStars_eqFunction_582(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,582};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[516]] /* x[37] STATE(1,vx[37]) */) = ((data->simulationInfo->realParameter[201] /* r_init[37] PARAM */)) * (cos((data->simulationInfo->realParameter[361] /* theta[37] PARAM */) + (data->simulationInfo->realParameter[39] /* armOffset[37] PARAM */)));
  TRACE_POP
}

/*
equation index: 583
type: SIMPLE_ASSIGN
vy[37] = x[37] * sqrt(G * Md / r_init[37] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,583};
  modelica_real tmp146;
  modelica_real tmp147;
  tmp146 = (data->simulationInfo->realParameter[201] /* r_init[37] PARAM */);
  tmp147 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp146 * tmp146 * tmp146),"r_init[37] ^ 3.0",equationIndexes);
  if(!(tmp147 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[37] ^ 3.0) was %g should be >= 0", tmp147);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[196]] /* vy[37] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[516]] /* x[37] STATE(1,vx[37]) */)) * (sqrt(tmp147));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2922(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2925(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2927(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2930(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2929(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2928(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2926(DATA *data, threadData_t *threadData);


/*
equation index: 591
type: SIMPLE_ASSIGN
vz[37] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,591};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[356]] /* vz[37] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2921(DATA *data, threadData_t *threadData);


/*
equation index: 593
type: SIMPLE_ASSIGN
z[38] = -2.1
*/
void WhirlpoolDiskStars_eqFunction_593(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,593};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[837]] /* z[38] STATE(1,vz[38]) */) = -2.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2934(DATA *data, threadData_t *threadData);


/*
equation index: 595
type: SIMPLE_ASSIGN
y[38] = r_init[38] * sin(theta[38] + armOffset[38])
*/
void WhirlpoolDiskStars_eqFunction_595(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,595};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[677]] /* y[38] STATE(1,vy[38]) */) = ((data->simulationInfo->realParameter[202] /* r_init[38] PARAM */)) * (sin((data->simulationInfo->realParameter[362] /* theta[38] PARAM */) + (data->simulationInfo->realParameter[40] /* armOffset[38] PARAM */)));
  TRACE_POP
}

/*
equation index: 596
type: SIMPLE_ASSIGN
vx[38] = (-y[38]) * sqrt(G * Md / r_init[38] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_596(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,596};
  modelica_real tmp148;
  modelica_real tmp149;
  tmp148 = (data->simulationInfo->realParameter[202] /* r_init[38] PARAM */);
  tmp149 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp148 * tmp148 * tmp148),"r_init[38] ^ 3.0",equationIndexes);
  if(!(tmp149 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[38] ^ 3.0) was %g should be >= 0", tmp149);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* vx[38] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[677]] /* y[38] STATE(1,vy[38]) */))) * (sqrt(tmp149));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2933(DATA *data, threadData_t *threadData);


/*
equation index: 598
type: SIMPLE_ASSIGN
x[38] = r_init[38] * cos(theta[38] + armOffset[38])
*/
void WhirlpoolDiskStars_eqFunction_598(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,598};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[517]] /* x[38] STATE(1,vx[38]) */) = ((data->simulationInfo->realParameter[202] /* r_init[38] PARAM */)) * (cos((data->simulationInfo->realParameter[362] /* theta[38] PARAM */) + (data->simulationInfo->realParameter[40] /* armOffset[38] PARAM */)));
  TRACE_POP
}

/*
equation index: 599
type: SIMPLE_ASSIGN
vy[38] = x[38] * sqrt(G * Md / r_init[38] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,599};
  modelica_real tmp150;
  modelica_real tmp151;
  tmp150 = (data->simulationInfo->realParameter[202] /* r_init[38] PARAM */);
  tmp151 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp150 * tmp150 * tmp150),"r_init[38] ^ 3.0",equationIndexes);
  if(!(tmp151 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[38] ^ 3.0) was %g should be >= 0", tmp151);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[197]] /* vy[38] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[517]] /* x[38] STATE(1,vx[38]) */)) * (sqrt(tmp151));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2932(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2935(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2937(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2940(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2939(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2938(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2936(DATA *data, threadData_t *threadData);


/*
equation index: 607
type: SIMPLE_ASSIGN
vz[38] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,607};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[357]] /* vz[38] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2931(DATA *data, threadData_t *threadData);


/*
equation index: 609
type: SIMPLE_ASSIGN
z[39] = -2.0500000000000003
*/
void WhirlpoolDiskStars_eqFunction_609(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,609};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[838]] /* z[39] STATE(1,vz[39]) */) = -2.0500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2944(DATA *data, threadData_t *threadData);


/*
equation index: 611
type: SIMPLE_ASSIGN
y[39] = r_init[39] * sin(theta[39] + armOffset[39])
*/
void WhirlpoolDiskStars_eqFunction_611(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,611};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[678]] /* y[39] STATE(1,vy[39]) */) = ((data->simulationInfo->realParameter[203] /* r_init[39] PARAM */)) * (sin((data->simulationInfo->realParameter[363] /* theta[39] PARAM */) + (data->simulationInfo->realParameter[41] /* armOffset[39] PARAM */)));
  TRACE_POP
}

/*
equation index: 612
type: SIMPLE_ASSIGN
vx[39] = (-y[39]) * sqrt(G * Md / r_init[39] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_612(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,612};
  modelica_real tmp152;
  modelica_real tmp153;
  tmp152 = (data->simulationInfo->realParameter[203] /* r_init[39] PARAM */);
  tmp153 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp152 * tmp152 * tmp152),"r_init[39] ^ 3.0",equationIndexes);
  if(!(tmp153 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[39] ^ 3.0) was %g should be >= 0", tmp153);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* vx[39] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[678]] /* y[39] STATE(1,vy[39]) */))) * (sqrt(tmp153));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2943(DATA *data, threadData_t *threadData);


/*
equation index: 614
type: SIMPLE_ASSIGN
x[39] = r_init[39] * cos(theta[39] + armOffset[39])
*/
void WhirlpoolDiskStars_eqFunction_614(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,614};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[518]] /* x[39] STATE(1,vx[39]) */) = ((data->simulationInfo->realParameter[203] /* r_init[39] PARAM */)) * (cos((data->simulationInfo->realParameter[363] /* theta[39] PARAM */) + (data->simulationInfo->realParameter[41] /* armOffset[39] PARAM */)));
  TRACE_POP
}

/*
equation index: 615
type: SIMPLE_ASSIGN
vy[39] = x[39] * sqrt(G * Md / r_init[39] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,615};
  modelica_real tmp154;
  modelica_real tmp155;
  tmp154 = (data->simulationInfo->realParameter[203] /* r_init[39] PARAM */);
  tmp155 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp154 * tmp154 * tmp154),"r_init[39] ^ 3.0",equationIndexes);
  if(!(tmp155 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[39] ^ 3.0) was %g should be >= 0", tmp155);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[198]] /* vy[39] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[518]] /* x[39] STATE(1,vx[39]) */)) * (sqrt(tmp155));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2942(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2945(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2947(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2950(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2949(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2948(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2946(DATA *data, threadData_t *threadData);


/*
equation index: 623
type: SIMPLE_ASSIGN
vz[39] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,623};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[358]] /* vz[39] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2941(DATA *data, threadData_t *threadData);


/*
equation index: 625
type: SIMPLE_ASSIGN
z[40] = -2.0
*/
void WhirlpoolDiskStars_eqFunction_625(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,625};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[839]] /* z[40] STATE(1,vz[40]) */) = -2.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2954(DATA *data, threadData_t *threadData);


/*
equation index: 627
type: SIMPLE_ASSIGN
y[40] = r_init[40] * sin(theta[40] + armOffset[40])
*/
void WhirlpoolDiskStars_eqFunction_627(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,627};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[679]] /* y[40] STATE(1,vy[40]) */) = ((data->simulationInfo->realParameter[204] /* r_init[40] PARAM */)) * (sin((data->simulationInfo->realParameter[364] /* theta[40] PARAM */) + (data->simulationInfo->realParameter[42] /* armOffset[40] PARAM */)));
  TRACE_POP
}

/*
equation index: 628
type: SIMPLE_ASSIGN
vx[40] = (-y[40]) * sqrt(G * Md / r_init[40] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_628(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,628};
  modelica_real tmp156;
  modelica_real tmp157;
  tmp156 = (data->simulationInfo->realParameter[204] /* r_init[40] PARAM */);
  tmp157 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp156 * tmp156 * tmp156),"r_init[40] ^ 3.0",equationIndexes);
  if(!(tmp157 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[40] ^ 3.0) was %g should be >= 0", tmp157);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* vx[40] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[679]] /* y[40] STATE(1,vy[40]) */))) * (sqrt(tmp157));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2953(DATA *data, threadData_t *threadData);


/*
equation index: 630
type: SIMPLE_ASSIGN
x[40] = r_init[40] * cos(theta[40] + armOffset[40])
*/
void WhirlpoolDiskStars_eqFunction_630(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,630};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[519]] /* x[40] STATE(1,vx[40]) */) = ((data->simulationInfo->realParameter[204] /* r_init[40] PARAM */)) * (cos((data->simulationInfo->realParameter[364] /* theta[40] PARAM */) + (data->simulationInfo->realParameter[42] /* armOffset[40] PARAM */)));
  TRACE_POP
}

/*
equation index: 631
type: SIMPLE_ASSIGN
vy[40] = x[40] * sqrt(G * Md / r_init[40] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,631};
  modelica_real tmp158;
  modelica_real tmp159;
  tmp158 = (data->simulationInfo->realParameter[204] /* r_init[40] PARAM */);
  tmp159 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp158 * tmp158 * tmp158),"r_init[40] ^ 3.0",equationIndexes);
  if(!(tmp159 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[40] ^ 3.0) was %g should be >= 0", tmp159);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[199]] /* vy[40] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[519]] /* x[40] STATE(1,vx[40]) */)) * (sqrt(tmp159));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2952(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2955(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2957(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2960(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2959(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2958(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2956(DATA *data, threadData_t *threadData);


/*
equation index: 639
type: SIMPLE_ASSIGN
vz[40] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,639};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[359]] /* vz[40] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2951(DATA *data, threadData_t *threadData);


/*
equation index: 641
type: SIMPLE_ASSIGN
z[41] = -1.9500000000000002
*/
void WhirlpoolDiskStars_eqFunction_641(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,641};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[840]] /* z[41] STATE(1,vz[41]) */) = -1.9500000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2964(DATA *data, threadData_t *threadData);


/*
equation index: 643
type: SIMPLE_ASSIGN
y[41] = r_init[41] * sin(theta[41] + armOffset[41])
*/
void WhirlpoolDiskStars_eqFunction_643(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,643};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[680]] /* y[41] STATE(1,vy[41]) */) = ((data->simulationInfo->realParameter[205] /* r_init[41] PARAM */)) * (sin((data->simulationInfo->realParameter[365] /* theta[41] PARAM */) + (data->simulationInfo->realParameter[43] /* armOffset[41] PARAM */)));
  TRACE_POP
}

/*
equation index: 644
type: SIMPLE_ASSIGN
vx[41] = (-y[41]) * sqrt(G * Md / r_init[41] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_644(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,644};
  modelica_real tmp160;
  modelica_real tmp161;
  tmp160 = (data->simulationInfo->realParameter[205] /* r_init[41] PARAM */);
  tmp161 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp160 * tmp160 * tmp160),"r_init[41] ^ 3.0",equationIndexes);
  if(!(tmp161 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[41] ^ 3.0) was %g should be >= 0", tmp161);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* vx[41] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[680]] /* y[41] STATE(1,vy[41]) */))) * (sqrt(tmp161));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2963(DATA *data, threadData_t *threadData);


/*
equation index: 646
type: SIMPLE_ASSIGN
x[41] = r_init[41] * cos(theta[41] + armOffset[41])
*/
void WhirlpoolDiskStars_eqFunction_646(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,646};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[520]] /* x[41] STATE(1,vx[41]) */) = ((data->simulationInfo->realParameter[205] /* r_init[41] PARAM */)) * (cos((data->simulationInfo->realParameter[365] /* theta[41] PARAM */) + (data->simulationInfo->realParameter[43] /* armOffset[41] PARAM */)));
  TRACE_POP
}

/*
equation index: 647
type: SIMPLE_ASSIGN
vy[41] = x[41] * sqrt(G * Md / r_init[41] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,647};
  modelica_real tmp162;
  modelica_real tmp163;
  tmp162 = (data->simulationInfo->realParameter[205] /* r_init[41] PARAM */);
  tmp163 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp162 * tmp162 * tmp162),"r_init[41] ^ 3.0",equationIndexes);
  if(!(tmp163 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[41] ^ 3.0) was %g should be >= 0", tmp163);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[200]] /* vy[41] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[520]] /* x[41] STATE(1,vx[41]) */)) * (sqrt(tmp163));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2962(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2965(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2967(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2970(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2969(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2968(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2966(DATA *data, threadData_t *threadData);


/*
equation index: 655
type: SIMPLE_ASSIGN
vz[41] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,655};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[360]] /* vz[41] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2961(DATA *data, threadData_t *threadData);


/*
equation index: 657
type: SIMPLE_ASSIGN
z[42] = -1.9000000000000001
*/
void WhirlpoolDiskStars_eqFunction_657(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,657};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[841]] /* z[42] STATE(1,vz[42]) */) = -1.9000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2974(DATA *data, threadData_t *threadData);


/*
equation index: 659
type: SIMPLE_ASSIGN
y[42] = r_init[42] * sin(theta[42] + armOffset[42])
*/
void WhirlpoolDiskStars_eqFunction_659(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,659};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[681]] /* y[42] STATE(1,vy[42]) */) = ((data->simulationInfo->realParameter[206] /* r_init[42] PARAM */)) * (sin((data->simulationInfo->realParameter[366] /* theta[42] PARAM */) + (data->simulationInfo->realParameter[44] /* armOffset[42] PARAM */)));
  TRACE_POP
}

/*
equation index: 660
type: SIMPLE_ASSIGN
vx[42] = (-y[42]) * sqrt(G * Md / r_init[42] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_660(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,660};
  modelica_real tmp164;
  modelica_real tmp165;
  tmp164 = (data->simulationInfo->realParameter[206] /* r_init[42] PARAM */);
  tmp165 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp164 * tmp164 * tmp164),"r_init[42] ^ 3.0",equationIndexes);
  if(!(tmp165 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[42] ^ 3.0) was %g should be >= 0", tmp165);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[41]] /* vx[42] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[681]] /* y[42] STATE(1,vy[42]) */))) * (sqrt(tmp165));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2973(DATA *data, threadData_t *threadData);


/*
equation index: 662
type: SIMPLE_ASSIGN
x[42] = r_init[42] * cos(theta[42] + armOffset[42])
*/
void WhirlpoolDiskStars_eqFunction_662(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,662};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[521]] /* x[42] STATE(1,vx[42]) */) = ((data->simulationInfo->realParameter[206] /* r_init[42] PARAM */)) * (cos((data->simulationInfo->realParameter[366] /* theta[42] PARAM */) + (data->simulationInfo->realParameter[44] /* armOffset[42] PARAM */)));
  TRACE_POP
}

/*
equation index: 663
type: SIMPLE_ASSIGN
vy[42] = x[42] * sqrt(G * Md / r_init[42] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,663};
  modelica_real tmp166;
  modelica_real tmp167;
  tmp166 = (data->simulationInfo->realParameter[206] /* r_init[42] PARAM */);
  tmp167 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp166 * tmp166 * tmp166),"r_init[42] ^ 3.0",equationIndexes);
  if(!(tmp167 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[42] ^ 3.0) was %g should be >= 0", tmp167);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[201]] /* vy[42] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[521]] /* x[42] STATE(1,vx[42]) */)) * (sqrt(tmp167));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2972(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2975(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2977(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2980(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2979(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2978(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2976(DATA *data, threadData_t *threadData);


/*
equation index: 671
type: SIMPLE_ASSIGN
vz[42] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,671};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[361]] /* vz[42] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2971(DATA *data, threadData_t *threadData);


/*
equation index: 673
type: SIMPLE_ASSIGN
z[43] = -1.85
*/
void WhirlpoolDiskStars_eqFunction_673(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,673};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[842]] /* z[43] STATE(1,vz[43]) */) = -1.85;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2984(DATA *data, threadData_t *threadData);


/*
equation index: 675
type: SIMPLE_ASSIGN
y[43] = r_init[43] * sin(theta[43] + armOffset[43])
*/
void WhirlpoolDiskStars_eqFunction_675(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,675};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[682]] /* y[43] STATE(1,vy[43]) */) = ((data->simulationInfo->realParameter[207] /* r_init[43] PARAM */)) * (sin((data->simulationInfo->realParameter[367] /* theta[43] PARAM */) + (data->simulationInfo->realParameter[45] /* armOffset[43] PARAM */)));
  TRACE_POP
}

/*
equation index: 676
type: SIMPLE_ASSIGN
vx[43] = (-y[43]) * sqrt(G * Md / r_init[43] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_676(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,676};
  modelica_real tmp168;
  modelica_real tmp169;
  tmp168 = (data->simulationInfo->realParameter[207] /* r_init[43] PARAM */);
  tmp169 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp168 * tmp168 * tmp168),"r_init[43] ^ 3.0",equationIndexes);
  if(!(tmp169 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[43] ^ 3.0) was %g should be >= 0", tmp169);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[42]] /* vx[43] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[682]] /* y[43] STATE(1,vy[43]) */))) * (sqrt(tmp169));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2983(DATA *data, threadData_t *threadData);


/*
equation index: 678
type: SIMPLE_ASSIGN
x[43] = r_init[43] * cos(theta[43] + armOffset[43])
*/
void WhirlpoolDiskStars_eqFunction_678(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,678};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[522]] /* x[43] STATE(1,vx[43]) */) = ((data->simulationInfo->realParameter[207] /* r_init[43] PARAM */)) * (cos((data->simulationInfo->realParameter[367] /* theta[43] PARAM */) + (data->simulationInfo->realParameter[45] /* armOffset[43] PARAM */)));
  TRACE_POP
}

/*
equation index: 679
type: SIMPLE_ASSIGN
vy[43] = x[43] * sqrt(G * Md / r_init[43] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,679};
  modelica_real tmp170;
  modelica_real tmp171;
  tmp170 = (data->simulationInfo->realParameter[207] /* r_init[43] PARAM */);
  tmp171 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp170 * tmp170 * tmp170),"r_init[43] ^ 3.0",equationIndexes);
  if(!(tmp171 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[43] ^ 3.0) was %g should be >= 0", tmp171);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[202]] /* vy[43] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[522]] /* x[43] STATE(1,vx[43]) */)) * (sqrt(tmp171));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2982(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2985(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2987(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2990(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2989(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2988(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2986(DATA *data, threadData_t *threadData);


/*
equation index: 687
type: SIMPLE_ASSIGN
vz[43] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,687};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[362]] /* vz[43] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2981(DATA *data, threadData_t *threadData);


/*
equation index: 689
type: SIMPLE_ASSIGN
z[44] = -1.8
*/
void WhirlpoolDiskStars_eqFunction_689(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,689};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[843]] /* z[44] STATE(1,vz[44]) */) = -1.8;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2994(DATA *data, threadData_t *threadData);


/*
equation index: 691
type: SIMPLE_ASSIGN
y[44] = r_init[44] * sin(theta[44] + armOffset[44])
*/
void WhirlpoolDiskStars_eqFunction_691(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,691};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[683]] /* y[44] STATE(1,vy[44]) */) = ((data->simulationInfo->realParameter[208] /* r_init[44] PARAM */)) * (sin((data->simulationInfo->realParameter[368] /* theta[44] PARAM */) + (data->simulationInfo->realParameter[46] /* armOffset[44] PARAM */)));
  TRACE_POP
}

/*
equation index: 692
type: SIMPLE_ASSIGN
vx[44] = (-y[44]) * sqrt(G * Md / r_init[44] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_692(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,692};
  modelica_real tmp172;
  modelica_real tmp173;
  tmp172 = (data->simulationInfo->realParameter[208] /* r_init[44] PARAM */);
  tmp173 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp172 * tmp172 * tmp172),"r_init[44] ^ 3.0",equationIndexes);
  if(!(tmp173 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[44] ^ 3.0) was %g should be >= 0", tmp173);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[43]] /* vx[44] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[683]] /* y[44] STATE(1,vy[44]) */))) * (sqrt(tmp173));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2993(DATA *data, threadData_t *threadData);


/*
equation index: 694
type: SIMPLE_ASSIGN
x[44] = r_init[44] * cos(theta[44] + armOffset[44])
*/
void WhirlpoolDiskStars_eqFunction_694(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,694};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[523]] /* x[44] STATE(1,vx[44]) */) = ((data->simulationInfo->realParameter[208] /* r_init[44] PARAM */)) * (cos((data->simulationInfo->realParameter[368] /* theta[44] PARAM */) + (data->simulationInfo->realParameter[46] /* armOffset[44] PARAM */)));
  TRACE_POP
}

/*
equation index: 695
type: SIMPLE_ASSIGN
vy[44] = x[44] * sqrt(G * Md / r_init[44] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,695};
  modelica_real tmp174;
  modelica_real tmp175;
  tmp174 = (data->simulationInfo->realParameter[208] /* r_init[44] PARAM */);
  tmp175 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp174 * tmp174 * tmp174),"r_init[44] ^ 3.0",equationIndexes);
  if(!(tmp175 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[44] ^ 3.0) was %g should be >= 0", tmp175);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[203]] /* vy[44] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[523]] /* x[44] STATE(1,vx[44]) */)) * (sqrt(tmp175));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2992(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2995(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2997(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3000(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2999(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2998(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2996(DATA *data, threadData_t *threadData);


/*
equation index: 703
type: SIMPLE_ASSIGN
vz[44] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,703};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[363]] /* vz[44] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2991(DATA *data, threadData_t *threadData);


/*
equation index: 705
type: SIMPLE_ASSIGN
z[45] = -1.75
*/
void WhirlpoolDiskStars_eqFunction_705(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,705};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[844]] /* z[45] STATE(1,vz[45]) */) = -1.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3004(DATA *data, threadData_t *threadData);


/*
equation index: 707
type: SIMPLE_ASSIGN
y[45] = r_init[45] * sin(theta[45] + armOffset[45])
*/
void WhirlpoolDiskStars_eqFunction_707(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,707};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[684]] /* y[45] STATE(1,vy[45]) */) = ((data->simulationInfo->realParameter[209] /* r_init[45] PARAM */)) * (sin((data->simulationInfo->realParameter[369] /* theta[45] PARAM */) + (data->simulationInfo->realParameter[47] /* armOffset[45] PARAM */)));
  TRACE_POP
}

/*
equation index: 708
type: SIMPLE_ASSIGN
vx[45] = (-y[45]) * sqrt(G * Md / r_init[45] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_708(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,708};
  modelica_real tmp176;
  modelica_real tmp177;
  tmp176 = (data->simulationInfo->realParameter[209] /* r_init[45] PARAM */);
  tmp177 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp176 * tmp176 * tmp176),"r_init[45] ^ 3.0",equationIndexes);
  if(!(tmp177 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[45] ^ 3.0) was %g should be >= 0", tmp177);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[44]] /* vx[45] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[684]] /* y[45] STATE(1,vy[45]) */))) * (sqrt(tmp177));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3003(DATA *data, threadData_t *threadData);


/*
equation index: 710
type: SIMPLE_ASSIGN
x[45] = r_init[45] * cos(theta[45] + armOffset[45])
*/
void WhirlpoolDiskStars_eqFunction_710(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,710};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[524]] /* x[45] STATE(1,vx[45]) */) = ((data->simulationInfo->realParameter[209] /* r_init[45] PARAM */)) * (cos((data->simulationInfo->realParameter[369] /* theta[45] PARAM */) + (data->simulationInfo->realParameter[47] /* armOffset[45] PARAM */)));
  TRACE_POP
}

/*
equation index: 711
type: SIMPLE_ASSIGN
vy[45] = x[45] * sqrt(G * Md / r_init[45] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,711};
  modelica_real tmp178;
  modelica_real tmp179;
  tmp178 = (data->simulationInfo->realParameter[209] /* r_init[45] PARAM */);
  tmp179 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp178 * tmp178 * tmp178),"r_init[45] ^ 3.0",equationIndexes);
  if(!(tmp179 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[45] ^ 3.0) was %g should be >= 0", tmp179);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[204]] /* vy[45] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[524]] /* x[45] STATE(1,vx[45]) */)) * (sqrt(tmp179));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3002(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3005(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3007(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3010(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3009(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3008(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3006(DATA *data, threadData_t *threadData);


/*
equation index: 719
type: SIMPLE_ASSIGN
vz[45] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,719};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[364]] /* vz[45] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3001(DATA *data, threadData_t *threadData);


/*
equation index: 721
type: SIMPLE_ASSIGN
z[46] = -1.7000000000000002
*/
void WhirlpoolDiskStars_eqFunction_721(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,721};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[845]] /* z[46] STATE(1,vz[46]) */) = -1.7000000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3014(DATA *data, threadData_t *threadData);


/*
equation index: 723
type: SIMPLE_ASSIGN
y[46] = r_init[46] * sin(theta[46] + armOffset[46])
*/
void WhirlpoolDiskStars_eqFunction_723(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,723};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[685]] /* y[46] STATE(1,vy[46]) */) = ((data->simulationInfo->realParameter[210] /* r_init[46] PARAM */)) * (sin((data->simulationInfo->realParameter[370] /* theta[46] PARAM */) + (data->simulationInfo->realParameter[48] /* armOffset[46] PARAM */)));
  TRACE_POP
}

/*
equation index: 724
type: SIMPLE_ASSIGN
vx[46] = (-y[46]) * sqrt(G * Md / r_init[46] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_724(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,724};
  modelica_real tmp180;
  modelica_real tmp181;
  tmp180 = (data->simulationInfo->realParameter[210] /* r_init[46] PARAM */);
  tmp181 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp180 * tmp180 * tmp180),"r_init[46] ^ 3.0",equationIndexes);
  if(!(tmp181 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[46] ^ 3.0) was %g should be >= 0", tmp181);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[45]] /* vx[46] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[685]] /* y[46] STATE(1,vy[46]) */))) * (sqrt(tmp181));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3013(DATA *data, threadData_t *threadData);


/*
equation index: 726
type: SIMPLE_ASSIGN
x[46] = r_init[46] * cos(theta[46] + armOffset[46])
*/
void WhirlpoolDiskStars_eqFunction_726(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,726};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[525]] /* x[46] STATE(1,vx[46]) */) = ((data->simulationInfo->realParameter[210] /* r_init[46] PARAM */)) * (cos((data->simulationInfo->realParameter[370] /* theta[46] PARAM */) + (data->simulationInfo->realParameter[48] /* armOffset[46] PARAM */)));
  TRACE_POP
}

/*
equation index: 727
type: SIMPLE_ASSIGN
vy[46] = x[46] * sqrt(G * Md / r_init[46] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,727};
  modelica_real tmp182;
  modelica_real tmp183;
  tmp182 = (data->simulationInfo->realParameter[210] /* r_init[46] PARAM */);
  tmp183 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp182 * tmp182 * tmp182),"r_init[46] ^ 3.0",equationIndexes);
  if(!(tmp183 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[46] ^ 3.0) was %g should be >= 0", tmp183);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[205]] /* vy[46] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[525]] /* x[46] STATE(1,vx[46]) */)) * (sqrt(tmp183));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3012(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3015(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3017(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3020(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3019(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3018(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3016(DATA *data, threadData_t *threadData);


/*
equation index: 735
type: SIMPLE_ASSIGN
vz[46] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,735};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[365]] /* vz[46] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3011(DATA *data, threadData_t *threadData);


/*
equation index: 737
type: SIMPLE_ASSIGN
z[47] = -1.6500000000000001
*/
void WhirlpoolDiskStars_eqFunction_737(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,737};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[846]] /* z[47] STATE(1,vz[47]) */) = -1.6500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3024(DATA *data, threadData_t *threadData);


/*
equation index: 739
type: SIMPLE_ASSIGN
y[47] = r_init[47] * sin(theta[47] + armOffset[47])
*/
void WhirlpoolDiskStars_eqFunction_739(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,739};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[686]] /* y[47] STATE(1,vy[47]) */) = ((data->simulationInfo->realParameter[211] /* r_init[47] PARAM */)) * (sin((data->simulationInfo->realParameter[371] /* theta[47] PARAM */) + (data->simulationInfo->realParameter[49] /* armOffset[47] PARAM */)));
  TRACE_POP
}

/*
equation index: 740
type: SIMPLE_ASSIGN
vx[47] = (-y[47]) * sqrt(G * Md / r_init[47] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_740(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,740};
  modelica_real tmp184;
  modelica_real tmp185;
  tmp184 = (data->simulationInfo->realParameter[211] /* r_init[47] PARAM */);
  tmp185 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp184 * tmp184 * tmp184),"r_init[47] ^ 3.0",equationIndexes);
  if(!(tmp185 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[47] ^ 3.0) was %g should be >= 0", tmp185);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[46]] /* vx[47] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[686]] /* y[47] STATE(1,vy[47]) */))) * (sqrt(tmp185));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3023(DATA *data, threadData_t *threadData);


/*
equation index: 742
type: SIMPLE_ASSIGN
x[47] = r_init[47] * cos(theta[47] + armOffset[47])
*/
void WhirlpoolDiskStars_eqFunction_742(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,742};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[526]] /* x[47] STATE(1,vx[47]) */) = ((data->simulationInfo->realParameter[211] /* r_init[47] PARAM */)) * (cos((data->simulationInfo->realParameter[371] /* theta[47] PARAM */) + (data->simulationInfo->realParameter[49] /* armOffset[47] PARAM */)));
  TRACE_POP
}

/*
equation index: 743
type: SIMPLE_ASSIGN
vy[47] = x[47] * sqrt(G * Md / r_init[47] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,743};
  modelica_real tmp186;
  modelica_real tmp187;
  tmp186 = (data->simulationInfo->realParameter[211] /* r_init[47] PARAM */);
  tmp187 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp186 * tmp186 * tmp186),"r_init[47] ^ 3.0",equationIndexes);
  if(!(tmp187 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[47] ^ 3.0) was %g should be >= 0", tmp187);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[206]] /* vy[47] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[526]] /* x[47] STATE(1,vx[47]) */)) * (sqrt(tmp187));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3022(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3025(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3027(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3030(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3029(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3028(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3026(DATA *data, threadData_t *threadData);


/*
equation index: 751
type: SIMPLE_ASSIGN
vz[47] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,751};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[366]] /* vz[47] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3021(DATA *data, threadData_t *threadData);


/*
equation index: 753
type: SIMPLE_ASSIGN
z[48] = -1.6
*/
void WhirlpoolDiskStars_eqFunction_753(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,753};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[847]] /* z[48] STATE(1,vz[48]) */) = -1.6;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3034(DATA *data, threadData_t *threadData);


/*
equation index: 755
type: SIMPLE_ASSIGN
y[48] = r_init[48] * sin(theta[48] + armOffset[48])
*/
void WhirlpoolDiskStars_eqFunction_755(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,755};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[687]] /* y[48] STATE(1,vy[48]) */) = ((data->simulationInfo->realParameter[212] /* r_init[48] PARAM */)) * (sin((data->simulationInfo->realParameter[372] /* theta[48] PARAM */) + (data->simulationInfo->realParameter[50] /* armOffset[48] PARAM */)));
  TRACE_POP
}

/*
equation index: 756
type: SIMPLE_ASSIGN
vx[48] = (-y[48]) * sqrt(G * Md / r_init[48] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_756(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,756};
  modelica_real tmp188;
  modelica_real tmp189;
  tmp188 = (data->simulationInfo->realParameter[212] /* r_init[48] PARAM */);
  tmp189 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp188 * tmp188 * tmp188),"r_init[48] ^ 3.0",equationIndexes);
  if(!(tmp189 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[48] ^ 3.0) was %g should be >= 0", tmp189);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* vx[48] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[687]] /* y[48] STATE(1,vy[48]) */))) * (sqrt(tmp189));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3033(DATA *data, threadData_t *threadData);


/*
equation index: 758
type: SIMPLE_ASSIGN
x[48] = r_init[48] * cos(theta[48] + armOffset[48])
*/
void WhirlpoolDiskStars_eqFunction_758(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,758};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[527]] /* x[48] STATE(1,vx[48]) */) = ((data->simulationInfo->realParameter[212] /* r_init[48] PARAM */)) * (cos((data->simulationInfo->realParameter[372] /* theta[48] PARAM */) + (data->simulationInfo->realParameter[50] /* armOffset[48] PARAM */)));
  TRACE_POP
}

/*
equation index: 759
type: SIMPLE_ASSIGN
vy[48] = x[48] * sqrt(G * Md / r_init[48] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,759};
  modelica_real tmp190;
  modelica_real tmp191;
  tmp190 = (data->simulationInfo->realParameter[212] /* r_init[48] PARAM */);
  tmp191 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp190 * tmp190 * tmp190),"r_init[48] ^ 3.0",equationIndexes);
  if(!(tmp191 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[48] ^ 3.0) was %g should be >= 0", tmp191);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[207]] /* vy[48] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[527]] /* x[48] STATE(1,vx[48]) */)) * (sqrt(tmp191));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3032(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3035(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3037(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3040(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3039(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3038(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3036(DATA *data, threadData_t *threadData);


/*
equation index: 767
type: SIMPLE_ASSIGN
vz[48] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,767};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[367]] /* vz[48] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3031(DATA *data, threadData_t *threadData);


/*
equation index: 769
type: SIMPLE_ASSIGN
z[49] = -1.55
*/
void WhirlpoolDiskStars_eqFunction_769(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,769};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[848]] /* z[49] STATE(1,vz[49]) */) = -1.55;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3044(DATA *data, threadData_t *threadData);


/*
equation index: 771
type: SIMPLE_ASSIGN
y[49] = r_init[49] * sin(theta[49] + armOffset[49])
*/
void WhirlpoolDiskStars_eqFunction_771(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,771};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[688]] /* y[49] STATE(1,vy[49]) */) = ((data->simulationInfo->realParameter[213] /* r_init[49] PARAM */)) * (sin((data->simulationInfo->realParameter[373] /* theta[49] PARAM */) + (data->simulationInfo->realParameter[51] /* armOffset[49] PARAM */)));
  TRACE_POP
}

/*
equation index: 772
type: SIMPLE_ASSIGN
vx[49] = (-y[49]) * sqrt(G * Md / r_init[49] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_772(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,772};
  modelica_real tmp192;
  modelica_real tmp193;
  tmp192 = (data->simulationInfo->realParameter[213] /* r_init[49] PARAM */);
  tmp193 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp192 * tmp192 * tmp192),"r_init[49] ^ 3.0",equationIndexes);
  if(!(tmp193 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[49] ^ 3.0) was %g should be >= 0", tmp193);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* vx[49] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[688]] /* y[49] STATE(1,vy[49]) */))) * (sqrt(tmp193));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3043(DATA *data, threadData_t *threadData);


/*
equation index: 774
type: SIMPLE_ASSIGN
x[49] = r_init[49] * cos(theta[49] + armOffset[49])
*/
void WhirlpoolDiskStars_eqFunction_774(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,774};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[528]] /* x[49] STATE(1,vx[49]) */) = ((data->simulationInfo->realParameter[213] /* r_init[49] PARAM */)) * (cos((data->simulationInfo->realParameter[373] /* theta[49] PARAM */) + (data->simulationInfo->realParameter[51] /* armOffset[49] PARAM */)));
  TRACE_POP
}

/*
equation index: 775
type: SIMPLE_ASSIGN
vy[49] = x[49] * sqrt(G * Md / r_init[49] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,775};
  modelica_real tmp194;
  modelica_real tmp195;
  tmp194 = (data->simulationInfo->realParameter[213] /* r_init[49] PARAM */);
  tmp195 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp194 * tmp194 * tmp194),"r_init[49] ^ 3.0",equationIndexes);
  if(!(tmp195 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[49] ^ 3.0) was %g should be >= 0", tmp195);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[208]] /* vy[49] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[528]] /* x[49] STATE(1,vx[49]) */)) * (sqrt(tmp195));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3042(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3045(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3047(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3050(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3049(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3048(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3046(DATA *data, threadData_t *threadData);


/*
equation index: 783
type: SIMPLE_ASSIGN
vz[49] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,783};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[368]] /* vz[49] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3041(DATA *data, threadData_t *threadData);


/*
equation index: 785
type: SIMPLE_ASSIGN
z[50] = -1.5
*/
void WhirlpoolDiskStars_eqFunction_785(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,785};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[849]] /* z[50] STATE(1,vz[50]) */) = -1.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3054(DATA *data, threadData_t *threadData);


/*
equation index: 787
type: SIMPLE_ASSIGN
y[50] = r_init[50] * sin(theta[50] + armOffset[50])
*/
void WhirlpoolDiskStars_eqFunction_787(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,787};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[689]] /* y[50] STATE(1,vy[50]) */) = ((data->simulationInfo->realParameter[214] /* r_init[50] PARAM */)) * (sin((data->simulationInfo->realParameter[374] /* theta[50] PARAM */) + (data->simulationInfo->realParameter[52] /* armOffset[50] PARAM */)));
  TRACE_POP
}

/*
equation index: 788
type: SIMPLE_ASSIGN
vx[50] = (-y[50]) * sqrt(G * Md / r_init[50] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_788(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,788};
  modelica_real tmp196;
  modelica_real tmp197;
  tmp196 = (data->simulationInfo->realParameter[214] /* r_init[50] PARAM */);
  tmp197 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp196 * tmp196 * tmp196),"r_init[50] ^ 3.0",equationIndexes);
  if(!(tmp197 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[50] ^ 3.0) was %g should be >= 0", tmp197);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[49]] /* vx[50] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[689]] /* y[50] STATE(1,vy[50]) */))) * (sqrt(tmp197));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3053(DATA *data, threadData_t *threadData);


/*
equation index: 790
type: SIMPLE_ASSIGN
x[50] = r_init[50] * cos(theta[50] + armOffset[50])
*/
void WhirlpoolDiskStars_eqFunction_790(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,790};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[529]] /* x[50] STATE(1,vx[50]) */) = ((data->simulationInfo->realParameter[214] /* r_init[50] PARAM */)) * (cos((data->simulationInfo->realParameter[374] /* theta[50] PARAM */) + (data->simulationInfo->realParameter[52] /* armOffset[50] PARAM */)));
  TRACE_POP
}

/*
equation index: 791
type: SIMPLE_ASSIGN
vy[50] = x[50] * sqrt(G * Md / r_init[50] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,791};
  modelica_real tmp198;
  modelica_real tmp199;
  tmp198 = (data->simulationInfo->realParameter[214] /* r_init[50] PARAM */);
  tmp199 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp198 * tmp198 * tmp198),"r_init[50] ^ 3.0",equationIndexes);
  if(!(tmp199 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[50] ^ 3.0) was %g should be >= 0", tmp199);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[209]] /* vy[50] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[529]] /* x[50] STATE(1,vx[50]) */)) * (sqrt(tmp199));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3052(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3055(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3057(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3060(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3059(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3058(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3056(DATA *data, threadData_t *threadData);


/*
equation index: 799
type: SIMPLE_ASSIGN
vz[50] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,799};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[369]] /* vz[50] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3051(DATA *data, threadData_t *threadData);


/*
equation index: 801
type: SIMPLE_ASSIGN
z[51] = -1.4500000000000002
*/
void WhirlpoolDiskStars_eqFunction_801(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,801};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[850]] /* z[51] STATE(1,vz[51]) */) = -1.4500000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3064(DATA *data, threadData_t *threadData);


/*
equation index: 803
type: SIMPLE_ASSIGN
y[51] = r_init[51] * sin(theta[51] + armOffset[51])
*/
void WhirlpoolDiskStars_eqFunction_803(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,803};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[690]] /* y[51] STATE(1,vy[51]) */) = ((data->simulationInfo->realParameter[215] /* r_init[51] PARAM */)) * (sin((data->simulationInfo->realParameter[375] /* theta[51] PARAM */) + (data->simulationInfo->realParameter[53] /* armOffset[51] PARAM */)));
  TRACE_POP
}

/*
equation index: 804
type: SIMPLE_ASSIGN
vx[51] = (-y[51]) * sqrt(G * Md / r_init[51] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_804(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,804};
  modelica_real tmp200;
  modelica_real tmp201;
  tmp200 = (data->simulationInfo->realParameter[215] /* r_init[51] PARAM */);
  tmp201 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp200 * tmp200 * tmp200),"r_init[51] ^ 3.0",equationIndexes);
  if(!(tmp201 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[51] ^ 3.0) was %g should be >= 0", tmp201);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[50]] /* vx[51] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[690]] /* y[51] STATE(1,vy[51]) */))) * (sqrt(tmp201));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3063(DATA *data, threadData_t *threadData);


/*
equation index: 806
type: SIMPLE_ASSIGN
x[51] = r_init[51] * cos(theta[51] + armOffset[51])
*/
void WhirlpoolDiskStars_eqFunction_806(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,806};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[530]] /* x[51] STATE(1,vx[51]) */) = ((data->simulationInfo->realParameter[215] /* r_init[51] PARAM */)) * (cos((data->simulationInfo->realParameter[375] /* theta[51] PARAM */) + (data->simulationInfo->realParameter[53] /* armOffset[51] PARAM */)));
  TRACE_POP
}

/*
equation index: 807
type: SIMPLE_ASSIGN
vy[51] = x[51] * sqrt(G * Md / r_init[51] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,807};
  modelica_real tmp202;
  modelica_real tmp203;
  tmp202 = (data->simulationInfo->realParameter[215] /* r_init[51] PARAM */);
  tmp203 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp202 * tmp202 * tmp202),"r_init[51] ^ 3.0",equationIndexes);
  if(!(tmp203 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[51] ^ 3.0) was %g should be >= 0", tmp203);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[210]] /* vy[51] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[530]] /* x[51] STATE(1,vx[51]) */)) * (sqrt(tmp203));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3062(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3065(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3067(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3070(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3069(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3068(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3066(DATA *data, threadData_t *threadData);


/*
equation index: 815
type: SIMPLE_ASSIGN
vz[51] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,815};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[370]] /* vz[51] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3061(DATA *data, threadData_t *threadData);


/*
equation index: 817
type: SIMPLE_ASSIGN
z[52] = -1.4000000000000001
*/
void WhirlpoolDiskStars_eqFunction_817(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,817};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[851]] /* z[52] STATE(1,vz[52]) */) = -1.4000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3074(DATA *data, threadData_t *threadData);


/*
equation index: 819
type: SIMPLE_ASSIGN
y[52] = r_init[52] * sin(theta[52] + armOffset[52])
*/
void WhirlpoolDiskStars_eqFunction_819(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,819};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[691]] /* y[52] STATE(1,vy[52]) */) = ((data->simulationInfo->realParameter[216] /* r_init[52] PARAM */)) * (sin((data->simulationInfo->realParameter[376] /* theta[52] PARAM */) + (data->simulationInfo->realParameter[54] /* armOffset[52] PARAM */)));
  TRACE_POP
}

/*
equation index: 820
type: SIMPLE_ASSIGN
vx[52] = (-y[52]) * sqrt(G * Md / r_init[52] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_820(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,820};
  modelica_real tmp204;
  modelica_real tmp205;
  tmp204 = (data->simulationInfo->realParameter[216] /* r_init[52] PARAM */);
  tmp205 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp204 * tmp204 * tmp204),"r_init[52] ^ 3.0",equationIndexes);
  if(!(tmp205 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[52] ^ 3.0) was %g should be >= 0", tmp205);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[51]] /* vx[52] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[691]] /* y[52] STATE(1,vy[52]) */))) * (sqrt(tmp205));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3073(DATA *data, threadData_t *threadData);


/*
equation index: 822
type: SIMPLE_ASSIGN
x[52] = r_init[52] * cos(theta[52] + armOffset[52])
*/
void WhirlpoolDiskStars_eqFunction_822(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,822};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[531]] /* x[52] STATE(1,vx[52]) */) = ((data->simulationInfo->realParameter[216] /* r_init[52] PARAM */)) * (cos((data->simulationInfo->realParameter[376] /* theta[52] PARAM */) + (data->simulationInfo->realParameter[54] /* armOffset[52] PARAM */)));
  TRACE_POP
}

/*
equation index: 823
type: SIMPLE_ASSIGN
vy[52] = x[52] * sqrt(G * Md / r_init[52] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,823};
  modelica_real tmp206;
  modelica_real tmp207;
  tmp206 = (data->simulationInfo->realParameter[216] /* r_init[52] PARAM */);
  tmp207 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp206 * tmp206 * tmp206),"r_init[52] ^ 3.0",equationIndexes);
  if(!(tmp207 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[52] ^ 3.0) was %g should be >= 0", tmp207);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[211]] /* vy[52] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[531]] /* x[52] STATE(1,vx[52]) */)) * (sqrt(tmp207));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3072(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3075(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3077(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3080(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3079(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3078(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3076(DATA *data, threadData_t *threadData);


/*
equation index: 831
type: SIMPLE_ASSIGN
vz[52] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,831};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[371]] /* vz[52] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3071(DATA *data, threadData_t *threadData);


/*
equation index: 833
type: SIMPLE_ASSIGN
z[53] = -1.35
*/
void WhirlpoolDiskStars_eqFunction_833(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,833};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[852]] /* z[53] STATE(1,vz[53]) */) = -1.35;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3084(DATA *data, threadData_t *threadData);


/*
equation index: 835
type: SIMPLE_ASSIGN
y[53] = r_init[53] * sin(theta[53] + armOffset[53])
*/
void WhirlpoolDiskStars_eqFunction_835(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,835};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[692]] /* y[53] STATE(1,vy[53]) */) = ((data->simulationInfo->realParameter[217] /* r_init[53] PARAM */)) * (sin((data->simulationInfo->realParameter[377] /* theta[53] PARAM */) + (data->simulationInfo->realParameter[55] /* armOffset[53] PARAM */)));
  TRACE_POP
}

/*
equation index: 836
type: SIMPLE_ASSIGN
vx[53] = (-y[53]) * sqrt(G * Md / r_init[53] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_836(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,836};
  modelica_real tmp208;
  modelica_real tmp209;
  tmp208 = (data->simulationInfo->realParameter[217] /* r_init[53] PARAM */);
  tmp209 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp208 * tmp208 * tmp208),"r_init[53] ^ 3.0",equationIndexes);
  if(!(tmp209 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[53] ^ 3.0) was %g should be >= 0", tmp209);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[52]] /* vx[53] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[692]] /* y[53] STATE(1,vy[53]) */))) * (sqrt(tmp209));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3083(DATA *data, threadData_t *threadData);


/*
equation index: 838
type: SIMPLE_ASSIGN
x[53] = r_init[53] * cos(theta[53] + armOffset[53])
*/
void WhirlpoolDiskStars_eqFunction_838(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,838};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[532]] /* x[53] STATE(1,vx[53]) */) = ((data->simulationInfo->realParameter[217] /* r_init[53] PARAM */)) * (cos((data->simulationInfo->realParameter[377] /* theta[53] PARAM */) + (data->simulationInfo->realParameter[55] /* armOffset[53] PARAM */)));
  TRACE_POP
}

/*
equation index: 839
type: SIMPLE_ASSIGN
vy[53] = x[53] * sqrt(G * Md / r_init[53] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,839};
  modelica_real tmp210;
  modelica_real tmp211;
  tmp210 = (data->simulationInfo->realParameter[217] /* r_init[53] PARAM */);
  tmp211 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp210 * tmp210 * tmp210),"r_init[53] ^ 3.0",equationIndexes);
  if(!(tmp211 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[53] ^ 3.0) was %g should be >= 0", tmp211);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[212]] /* vy[53] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[532]] /* x[53] STATE(1,vx[53]) */)) * (sqrt(tmp211));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3082(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3085(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3087(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3090(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3089(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3088(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3086(DATA *data, threadData_t *threadData);


/*
equation index: 847
type: SIMPLE_ASSIGN
vz[53] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,847};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[372]] /* vz[53] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3081(DATA *data, threadData_t *threadData);


/*
equation index: 849
type: SIMPLE_ASSIGN
z[54] = -1.3
*/
void WhirlpoolDiskStars_eqFunction_849(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,849};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[853]] /* z[54] STATE(1,vz[54]) */) = -1.3;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3094(DATA *data, threadData_t *threadData);


/*
equation index: 851
type: SIMPLE_ASSIGN
y[54] = r_init[54] * sin(theta[54] + armOffset[54])
*/
void WhirlpoolDiskStars_eqFunction_851(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,851};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[693]] /* y[54] STATE(1,vy[54]) */) = ((data->simulationInfo->realParameter[218] /* r_init[54] PARAM */)) * (sin((data->simulationInfo->realParameter[378] /* theta[54] PARAM */) + (data->simulationInfo->realParameter[56] /* armOffset[54] PARAM */)));
  TRACE_POP
}

/*
equation index: 852
type: SIMPLE_ASSIGN
vx[54] = (-y[54]) * sqrt(G * Md / r_init[54] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_852(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,852};
  modelica_real tmp212;
  modelica_real tmp213;
  tmp212 = (data->simulationInfo->realParameter[218] /* r_init[54] PARAM */);
  tmp213 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp212 * tmp212 * tmp212),"r_init[54] ^ 3.0",equationIndexes);
  if(!(tmp213 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[54] ^ 3.0) was %g should be >= 0", tmp213);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[53]] /* vx[54] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[693]] /* y[54] STATE(1,vy[54]) */))) * (sqrt(tmp213));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3093(DATA *data, threadData_t *threadData);


/*
equation index: 854
type: SIMPLE_ASSIGN
x[54] = r_init[54] * cos(theta[54] + armOffset[54])
*/
void WhirlpoolDiskStars_eqFunction_854(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,854};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[533]] /* x[54] STATE(1,vx[54]) */) = ((data->simulationInfo->realParameter[218] /* r_init[54] PARAM */)) * (cos((data->simulationInfo->realParameter[378] /* theta[54] PARAM */) + (data->simulationInfo->realParameter[56] /* armOffset[54] PARAM */)));
  TRACE_POP
}
OMC_DISABLE_OPT
void WhirlpoolDiskStars_functionInitialEquations_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_2829(data, threadData);
  WhirlpoolDiskStars_eqFunction_2828(data, threadData);
  WhirlpoolDiskStars_eqFunction_2826(data, threadData);
  WhirlpoolDiskStars_eqFunction_431(data, threadData);
  WhirlpoolDiskStars_eqFunction_2821(data, threadData);
  WhirlpoolDiskStars_eqFunction_433(data, threadData);
  WhirlpoolDiskStars_eqFunction_2834(data, threadData);
  WhirlpoolDiskStars_eqFunction_435(data, threadData);
  WhirlpoolDiskStars_eqFunction_436(data, threadData);
  WhirlpoolDiskStars_eqFunction_2833(data, threadData);
  WhirlpoolDiskStars_eqFunction_438(data, threadData);
  WhirlpoolDiskStars_eqFunction_439(data, threadData);
  WhirlpoolDiskStars_eqFunction_2832(data, threadData);
  WhirlpoolDiskStars_eqFunction_2835(data, threadData);
  WhirlpoolDiskStars_eqFunction_2837(data, threadData);
  WhirlpoolDiskStars_eqFunction_2840(data, threadData);
  WhirlpoolDiskStars_eqFunction_2839(data, threadData);
  WhirlpoolDiskStars_eqFunction_2838(data, threadData);
  WhirlpoolDiskStars_eqFunction_2836(data, threadData);
  WhirlpoolDiskStars_eqFunction_447(data, threadData);
  WhirlpoolDiskStars_eqFunction_2831(data, threadData);
  WhirlpoolDiskStars_eqFunction_449(data, threadData);
  WhirlpoolDiskStars_eqFunction_2844(data, threadData);
  WhirlpoolDiskStars_eqFunction_451(data, threadData);
  WhirlpoolDiskStars_eqFunction_452(data, threadData);
  WhirlpoolDiskStars_eqFunction_2843(data, threadData);
  WhirlpoolDiskStars_eqFunction_454(data, threadData);
  WhirlpoolDiskStars_eqFunction_455(data, threadData);
  WhirlpoolDiskStars_eqFunction_2842(data, threadData);
  WhirlpoolDiskStars_eqFunction_2845(data, threadData);
  WhirlpoolDiskStars_eqFunction_2847(data, threadData);
  WhirlpoolDiskStars_eqFunction_2850(data, threadData);
  WhirlpoolDiskStars_eqFunction_2849(data, threadData);
  WhirlpoolDiskStars_eqFunction_2848(data, threadData);
  WhirlpoolDiskStars_eqFunction_2846(data, threadData);
  WhirlpoolDiskStars_eqFunction_463(data, threadData);
  WhirlpoolDiskStars_eqFunction_2841(data, threadData);
  WhirlpoolDiskStars_eqFunction_465(data, threadData);
  WhirlpoolDiskStars_eqFunction_2854(data, threadData);
  WhirlpoolDiskStars_eqFunction_467(data, threadData);
  WhirlpoolDiskStars_eqFunction_468(data, threadData);
  WhirlpoolDiskStars_eqFunction_2853(data, threadData);
  WhirlpoolDiskStars_eqFunction_470(data, threadData);
  WhirlpoolDiskStars_eqFunction_471(data, threadData);
  WhirlpoolDiskStars_eqFunction_2852(data, threadData);
  WhirlpoolDiskStars_eqFunction_2855(data, threadData);
  WhirlpoolDiskStars_eqFunction_2857(data, threadData);
  WhirlpoolDiskStars_eqFunction_2860(data, threadData);
  WhirlpoolDiskStars_eqFunction_2859(data, threadData);
  WhirlpoolDiskStars_eqFunction_2858(data, threadData);
  WhirlpoolDiskStars_eqFunction_2856(data, threadData);
  WhirlpoolDiskStars_eqFunction_479(data, threadData);
  WhirlpoolDiskStars_eqFunction_2851(data, threadData);
  WhirlpoolDiskStars_eqFunction_481(data, threadData);
  WhirlpoolDiskStars_eqFunction_2864(data, threadData);
  WhirlpoolDiskStars_eqFunction_483(data, threadData);
  WhirlpoolDiskStars_eqFunction_484(data, threadData);
  WhirlpoolDiskStars_eqFunction_2863(data, threadData);
  WhirlpoolDiskStars_eqFunction_486(data, threadData);
  WhirlpoolDiskStars_eqFunction_487(data, threadData);
  WhirlpoolDiskStars_eqFunction_2862(data, threadData);
  WhirlpoolDiskStars_eqFunction_2865(data, threadData);
  WhirlpoolDiskStars_eqFunction_2867(data, threadData);
  WhirlpoolDiskStars_eqFunction_2870(data, threadData);
  WhirlpoolDiskStars_eqFunction_2869(data, threadData);
  WhirlpoolDiskStars_eqFunction_2868(data, threadData);
  WhirlpoolDiskStars_eqFunction_2866(data, threadData);
  WhirlpoolDiskStars_eqFunction_495(data, threadData);
  WhirlpoolDiskStars_eqFunction_2861(data, threadData);
  WhirlpoolDiskStars_eqFunction_497(data, threadData);
  WhirlpoolDiskStars_eqFunction_2874(data, threadData);
  WhirlpoolDiskStars_eqFunction_499(data, threadData);
  WhirlpoolDiskStars_eqFunction_500(data, threadData);
  WhirlpoolDiskStars_eqFunction_2873(data, threadData);
  WhirlpoolDiskStars_eqFunction_502(data, threadData);
  WhirlpoolDiskStars_eqFunction_503(data, threadData);
  WhirlpoolDiskStars_eqFunction_2872(data, threadData);
  WhirlpoolDiskStars_eqFunction_2875(data, threadData);
  WhirlpoolDiskStars_eqFunction_2877(data, threadData);
  WhirlpoolDiskStars_eqFunction_2880(data, threadData);
  WhirlpoolDiskStars_eqFunction_2879(data, threadData);
  WhirlpoolDiskStars_eqFunction_2878(data, threadData);
  WhirlpoolDiskStars_eqFunction_2876(data, threadData);
  WhirlpoolDiskStars_eqFunction_511(data, threadData);
  WhirlpoolDiskStars_eqFunction_2871(data, threadData);
  WhirlpoolDiskStars_eqFunction_513(data, threadData);
  WhirlpoolDiskStars_eqFunction_2884(data, threadData);
  WhirlpoolDiskStars_eqFunction_515(data, threadData);
  WhirlpoolDiskStars_eqFunction_516(data, threadData);
  WhirlpoolDiskStars_eqFunction_2883(data, threadData);
  WhirlpoolDiskStars_eqFunction_518(data, threadData);
  WhirlpoolDiskStars_eqFunction_519(data, threadData);
  WhirlpoolDiskStars_eqFunction_2882(data, threadData);
  WhirlpoolDiskStars_eqFunction_2885(data, threadData);
  WhirlpoolDiskStars_eqFunction_2887(data, threadData);
  WhirlpoolDiskStars_eqFunction_2890(data, threadData);
  WhirlpoolDiskStars_eqFunction_2889(data, threadData);
  WhirlpoolDiskStars_eqFunction_2888(data, threadData);
  WhirlpoolDiskStars_eqFunction_2886(data, threadData);
  WhirlpoolDiskStars_eqFunction_527(data, threadData);
  WhirlpoolDiskStars_eqFunction_2881(data, threadData);
  WhirlpoolDiskStars_eqFunction_529(data, threadData);
  WhirlpoolDiskStars_eqFunction_2894(data, threadData);
  WhirlpoolDiskStars_eqFunction_531(data, threadData);
  WhirlpoolDiskStars_eqFunction_532(data, threadData);
  WhirlpoolDiskStars_eqFunction_2893(data, threadData);
  WhirlpoolDiskStars_eqFunction_534(data, threadData);
  WhirlpoolDiskStars_eqFunction_535(data, threadData);
  WhirlpoolDiskStars_eqFunction_2892(data, threadData);
  WhirlpoolDiskStars_eqFunction_2895(data, threadData);
  WhirlpoolDiskStars_eqFunction_2897(data, threadData);
  WhirlpoolDiskStars_eqFunction_2900(data, threadData);
  WhirlpoolDiskStars_eqFunction_2899(data, threadData);
  WhirlpoolDiskStars_eqFunction_2898(data, threadData);
  WhirlpoolDiskStars_eqFunction_2896(data, threadData);
  WhirlpoolDiskStars_eqFunction_543(data, threadData);
  WhirlpoolDiskStars_eqFunction_2891(data, threadData);
  WhirlpoolDiskStars_eqFunction_545(data, threadData);
  WhirlpoolDiskStars_eqFunction_2904(data, threadData);
  WhirlpoolDiskStars_eqFunction_547(data, threadData);
  WhirlpoolDiskStars_eqFunction_548(data, threadData);
  WhirlpoolDiskStars_eqFunction_2903(data, threadData);
  WhirlpoolDiskStars_eqFunction_550(data, threadData);
  WhirlpoolDiskStars_eqFunction_551(data, threadData);
  WhirlpoolDiskStars_eqFunction_2902(data, threadData);
  WhirlpoolDiskStars_eqFunction_2905(data, threadData);
  WhirlpoolDiskStars_eqFunction_2907(data, threadData);
  WhirlpoolDiskStars_eqFunction_2910(data, threadData);
  WhirlpoolDiskStars_eqFunction_2909(data, threadData);
  WhirlpoolDiskStars_eqFunction_2908(data, threadData);
  WhirlpoolDiskStars_eqFunction_2906(data, threadData);
  WhirlpoolDiskStars_eqFunction_559(data, threadData);
  WhirlpoolDiskStars_eqFunction_2901(data, threadData);
  WhirlpoolDiskStars_eqFunction_561(data, threadData);
  WhirlpoolDiskStars_eqFunction_2914(data, threadData);
  WhirlpoolDiskStars_eqFunction_563(data, threadData);
  WhirlpoolDiskStars_eqFunction_564(data, threadData);
  WhirlpoolDiskStars_eqFunction_2913(data, threadData);
  WhirlpoolDiskStars_eqFunction_566(data, threadData);
  WhirlpoolDiskStars_eqFunction_567(data, threadData);
  WhirlpoolDiskStars_eqFunction_2912(data, threadData);
  WhirlpoolDiskStars_eqFunction_2915(data, threadData);
  WhirlpoolDiskStars_eqFunction_2917(data, threadData);
  WhirlpoolDiskStars_eqFunction_2920(data, threadData);
  WhirlpoolDiskStars_eqFunction_2919(data, threadData);
  WhirlpoolDiskStars_eqFunction_2918(data, threadData);
  WhirlpoolDiskStars_eqFunction_2916(data, threadData);
  WhirlpoolDiskStars_eqFunction_575(data, threadData);
  WhirlpoolDiskStars_eqFunction_2911(data, threadData);
  WhirlpoolDiskStars_eqFunction_577(data, threadData);
  WhirlpoolDiskStars_eqFunction_2924(data, threadData);
  WhirlpoolDiskStars_eqFunction_579(data, threadData);
  WhirlpoolDiskStars_eqFunction_580(data, threadData);
  WhirlpoolDiskStars_eqFunction_2923(data, threadData);
  WhirlpoolDiskStars_eqFunction_582(data, threadData);
  WhirlpoolDiskStars_eqFunction_583(data, threadData);
  WhirlpoolDiskStars_eqFunction_2922(data, threadData);
  WhirlpoolDiskStars_eqFunction_2925(data, threadData);
  WhirlpoolDiskStars_eqFunction_2927(data, threadData);
  WhirlpoolDiskStars_eqFunction_2930(data, threadData);
  WhirlpoolDiskStars_eqFunction_2929(data, threadData);
  WhirlpoolDiskStars_eqFunction_2928(data, threadData);
  WhirlpoolDiskStars_eqFunction_2926(data, threadData);
  WhirlpoolDiskStars_eqFunction_591(data, threadData);
  WhirlpoolDiskStars_eqFunction_2921(data, threadData);
  WhirlpoolDiskStars_eqFunction_593(data, threadData);
  WhirlpoolDiskStars_eqFunction_2934(data, threadData);
  WhirlpoolDiskStars_eqFunction_595(data, threadData);
  WhirlpoolDiskStars_eqFunction_596(data, threadData);
  WhirlpoolDiskStars_eqFunction_2933(data, threadData);
  WhirlpoolDiskStars_eqFunction_598(data, threadData);
  WhirlpoolDiskStars_eqFunction_599(data, threadData);
  WhirlpoolDiskStars_eqFunction_2932(data, threadData);
  WhirlpoolDiskStars_eqFunction_2935(data, threadData);
  WhirlpoolDiskStars_eqFunction_2937(data, threadData);
  WhirlpoolDiskStars_eqFunction_2940(data, threadData);
  WhirlpoolDiskStars_eqFunction_2939(data, threadData);
  WhirlpoolDiskStars_eqFunction_2938(data, threadData);
  WhirlpoolDiskStars_eqFunction_2936(data, threadData);
  WhirlpoolDiskStars_eqFunction_607(data, threadData);
  WhirlpoolDiskStars_eqFunction_2931(data, threadData);
  WhirlpoolDiskStars_eqFunction_609(data, threadData);
  WhirlpoolDiskStars_eqFunction_2944(data, threadData);
  WhirlpoolDiskStars_eqFunction_611(data, threadData);
  WhirlpoolDiskStars_eqFunction_612(data, threadData);
  WhirlpoolDiskStars_eqFunction_2943(data, threadData);
  WhirlpoolDiskStars_eqFunction_614(data, threadData);
  WhirlpoolDiskStars_eqFunction_615(data, threadData);
  WhirlpoolDiskStars_eqFunction_2942(data, threadData);
  WhirlpoolDiskStars_eqFunction_2945(data, threadData);
  WhirlpoolDiskStars_eqFunction_2947(data, threadData);
  WhirlpoolDiskStars_eqFunction_2950(data, threadData);
  WhirlpoolDiskStars_eqFunction_2949(data, threadData);
  WhirlpoolDiskStars_eqFunction_2948(data, threadData);
  WhirlpoolDiskStars_eqFunction_2946(data, threadData);
  WhirlpoolDiskStars_eqFunction_623(data, threadData);
  WhirlpoolDiskStars_eqFunction_2941(data, threadData);
  WhirlpoolDiskStars_eqFunction_625(data, threadData);
  WhirlpoolDiskStars_eqFunction_2954(data, threadData);
  WhirlpoolDiskStars_eqFunction_627(data, threadData);
  WhirlpoolDiskStars_eqFunction_628(data, threadData);
  WhirlpoolDiskStars_eqFunction_2953(data, threadData);
  WhirlpoolDiskStars_eqFunction_630(data, threadData);
  WhirlpoolDiskStars_eqFunction_631(data, threadData);
  WhirlpoolDiskStars_eqFunction_2952(data, threadData);
  WhirlpoolDiskStars_eqFunction_2955(data, threadData);
  WhirlpoolDiskStars_eqFunction_2957(data, threadData);
  WhirlpoolDiskStars_eqFunction_2960(data, threadData);
  WhirlpoolDiskStars_eqFunction_2959(data, threadData);
  WhirlpoolDiskStars_eqFunction_2958(data, threadData);
  WhirlpoolDiskStars_eqFunction_2956(data, threadData);
  WhirlpoolDiskStars_eqFunction_639(data, threadData);
  WhirlpoolDiskStars_eqFunction_2951(data, threadData);
  WhirlpoolDiskStars_eqFunction_641(data, threadData);
  WhirlpoolDiskStars_eqFunction_2964(data, threadData);
  WhirlpoolDiskStars_eqFunction_643(data, threadData);
  WhirlpoolDiskStars_eqFunction_644(data, threadData);
  WhirlpoolDiskStars_eqFunction_2963(data, threadData);
  WhirlpoolDiskStars_eqFunction_646(data, threadData);
  WhirlpoolDiskStars_eqFunction_647(data, threadData);
  WhirlpoolDiskStars_eqFunction_2962(data, threadData);
  WhirlpoolDiskStars_eqFunction_2965(data, threadData);
  WhirlpoolDiskStars_eqFunction_2967(data, threadData);
  WhirlpoolDiskStars_eqFunction_2970(data, threadData);
  WhirlpoolDiskStars_eqFunction_2969(data, threadData);
  WhirlpoolDiskStars_eqFunction_2968(data, threadData);
  WhirlpoolDiskStars_eqFunction_2966(data, threadData);
  WhirlpoolDiskStars_eqFunction_655(data, threadData);
  WhirlpoolDiskStars_eqFunction_2961(data, threadData);
  WhirlpoolDiskStars_eqFunction_657(data, threadData);
  WhirlpoolDiskStars_eqFunction_2974(data, threadData);
  WhirlpoolDiskStars_eqFunction_659(data, threadData);
  WhirlpoolDiskStars_eqFunction_660(data, threadData);
  WhirlpoolDiskStars_eqFunction_2973(data, threadData);
  WhirlpoolDiskStars_eqFunction_662(data, threadData);
  WhirlpoolDiskStars_eqFunction_663(data, threadData);
  WhirlpoolDiskStars_eqFunction_2972(data, threadData);
  WhirlpoolDiskStars_eqFunction_2975(data, threadData);
  WhirlpoolDiskStars_eqFunction_2977(data, threadData);
  WhirlpoolDiskStars_eqFunction_2980(data, threadData);
  WhirlpoolDiskStars_eqFunction_2979(data, threadData);
  WhirlpoolDiskStars_eqFunction_2978(data, threadData);
  WhirlpoolDiskStars_eqFunction_2976(data, threadData);
  WhirlpoolDiskStars_eqFunction_671(data, threadData);
  WhirlpoolDiskStars_eqFunction_2971(data, threadData);
  WhirlpoolDiskStars_eqFunction_673(data, threadData);
  WhirlpoolDiskStars_eqFunction_2984(data, threadData);
  WhirlpoolDiskStars_eqFunction_675(data, threadData);
  WhirlpoolDiskStars_eqFunction_676(data, threadData);
  WhirlpoolDiskStars_eqFunction_2983(data, threadData);
  WhirlpoolDiskStars_eqFunction_678(data, threadData);
  WhirlpoolDiskStars_eqFunction_679(data, threadData);
  WhirlpoolDiskStars_eqFunction_2982(data, threadData);
  WhirlpoolDiskStars_eqFunction_2985(data, threadData);
  WhirlpoolDiskStars_eqFunction_2987(data, threadData);
  WhirlpoolDiskStars_eqFunction_2990(data, threadData);
  WhirlpoolDiskStars_eqFunction_2989(data, threadData);
  WhirlpoolDiskStars_eqFunction_2988(data, threadData);
  WhirlpoolDiskStars_eqFunction_2986(data, threadData);
  WhirlpoolDiskStars_eqFunction_687(data, threadData);
  WhirlpoolDiskStars_eqFunction_2981(data, threadData);
  WhirlpoolDiskStars_eqFunction_689(data, threadData);
  WhirlpoolDiskStars_eqFunction_2994(data, threadData);
  WhirlpoolDiskStars_eqFunction_691(data, threadData);
  WhirlpoolDiskStars_eqFunction_692(data, threadData);
  WhirlpoolDiskStars_eqFunction_2993(data, threadData);
  WhirlpoolDiskStars_eqFunction_694(data, threadData);
  WhirlpoolDiskStars_eqFunction_695(data, threadData);
  WhirlpoolDiskStars_eqFunction_2992(data, threadData);
  WhirlpoolDiskStars_eqFunction_2995(data, threadData);
  WhirlpoolDiskStars_eqFunction_2997(data, threadData);
  WhirlpoolDiskStars_eqFunction_3000(data, threadData);
  WhirlpoolDiskStars_eqFunction_2999(data, threadData);
  WhirlpoolDiskStars_eqFunction_2998(data, threadData);
  WhirlpoolDiskStars_eqFunction_2996(data, threadData);
  WhirlpoolDiskStars_eqFunction_703(data, threadData);
  WhirlpoolDiskStars_eqFunction_2991(data, threadData);
  WhirlpoolDiskStars_eqFunction_705(data, threadData);
  WhirlpoolDiskStars_eqFunction_3004(data, threadData);
  WhirlpoolDiskStars_eqFunction_707(data, threadData);
  WhirlpoolDiskStars_eqFunction_708(data, threadData);
  WhirlpoolDiskStars_eqFunction_3003(data, threadData);
  WhirlpoolDiskStars_eqFunction_710(data, threadData);
  WhirlpoolDiskStars_eqFunction_711(data, threadData);
  WhirlpoolDiskStars_eqFunction_3002(data, threadData);
  WhirlpoolDiskStars_eqFunction_3005(data, threadData);
  WhirlpoolDiskStars_eqFunction_3007(data, threadData);
  WhirlpoolDiskStars_eqFunction_3010(data, threadData);
  WhirlpoolDiskStars_eqFunction_3009(data, threadData);
  WhirlpoolDiskStars_eqFunction_3008(data, threadData);
  WhirlpoolDiskStars_eqFunction_3006(data, threadData);
  WhirlpoolDiskStars_eqFunction_719(data, threadData);
  WhirlpoolDiskStars_eqFunction_3001(data, threadData);
  WhirlpoolDiskStars_eqFunction_721(data, threadData);
  WhirlpoolDiskStars_eqFunction_3014(data, threadData);
  WhirlpoolDiskStars_eqFunction_723(data, threadData);
  WhirlpoolDiskStars_eqFunction_724(data, threadData);
  WhirlpoolDiskStars_eqFunction_3013(data, threadData);
  WhirlpoolDiskStars_eqFunction_726(data, threadData);
  WhirlpoolDiskStars_eqFunction_727(data, threadData);
  WhirlpoolDiskStars_eqFunction_3012(data, threadData);
  WhirlpoolDiskStars_eqFunction_3015(data, threadData);
  WhirlpoolDiskStars_eqFunction_3017(data, threadData);
  WhirlpoolDiskStars_eqFunction_3020(data, threadData);
  WhirlpoolDiskStars_eqFunction_3019(data, threadData);
  WhirlpoolDiskStars_eqFunction_3018(data, threadData);
  WhirlpoolDiskStars_eqFunction_3016(data, threadData);
  WhirlpoolDiskStars_eqFunction_735(data, threadData);
  WhirlpoolDiskStars_eqFunction_3011(data, threadData);
  WhirlpoolDiskStars_eqFunction_737(data, threadData);
  WhirlpoolDiskStars_eqFunction_3024(data, threadData);
  WhirlpoolDiskStars_eqFunction_739(data, threadData);
  WhirlpoolDiskStars_eqFunction_740(data, threadData);
  WhirlpoolDiskStars_eqFunction_3023(data, threadData);
  WhirlpoolDiskStars_eqFunction_742(data, threadData);
  WhirlpoolDiskStars_eqFunction_743(data, threadData);
  WhirlpoolDiskStars_eqFunction_3022(data, threadData);
  WhirlpoolDiskStars_eqFunction_3025(data, threadData);
  WhirlpoolDiskStars_eqFunction_3027(data, threadData);
  WhirlpoolDiskStars_eqFunction_3030(data, threadData);
  WhirlpoolDiskStars_eqFunction_3029(data, threadData);
  WhirlpoolDiskStars_eqFunction_3028(data, threadData);
  WhirlpoolDiskStars_eqFunction_3026(data, threadData);
  WhirlpoolDiskStars_eqFunction_751(data, threadData);
  WhirlpoolDiskStars_eqFunction_3021(data, threadData);
  WhirlpoolDiskStars_eqFunction_753(data, threadData);
  WhirlpoolDiskStars_eqFunction_3034(data, threadData);
  WhirlpoolDiskStars_eqFunction_755(data, threadData);
  WhirlpoolDiskStars_eqFunction_756(data, threadData);
  WhirlpoolDiskStars_eqFunction_3033(data, threadData);
  WhirlpoolDiskStars_eqFunction_758(data, threadData);
  WhirlpoolDiskStars_eqFunction_759(data, threadData);
  WhirlpoolDiskStars_eqFunction_3032(data, threadData);
  WhirlpoolDiskStars_eqFunction_3035(data, threadData);
  WhirlpoolDiskStars_eqFunction_3037(data, threadData);
  WhirlpoolDiskStars_eqFunction_3040(data, threadData);
  WhirlpoolDiskStars_eqFunction_3039(data, threadData);
  WhirlpoolDiskStars_eqFunction_3038(data, threadData);
  WhirlpoolDiskStars_eqFunction_3036(data, threadData);
  WhirlpoolDiskStars_eqFunction_767(data, threadData);
  WhirlpoolDiskStars_eqFunction_3031(data, threadData);
  WhirlpoolDiskStars_eqFunction_769(data, threadData);
  WhirlpoolDiskStars_eqFunction_3044(data, threadData);
  WhirlpoolDiskStars_eqFunction_771(data, threadData);
  WhirlpoolDiskStars_eqFunction_772(data, threadData);
  WhirlpoolDiskStars_eqFunction_3043(data, threadData);
  WhirlpoolDiskStars_eqFunction_774(data, threadData);
  WhirlpoolDiskStars_eqFunction_775(data, threadData);
  WhirlpoolDiskStars_eqFunction_3042(data, threadData);
  WhirlpoolDiskStars_eqFunction_3045(data, threadData);
  WhirlpoolDiskStars_eqFunction_3047(data, threadData);
  WhirlpoolDiskStars_eqFunction_3050(data, threadData);
  WhirlpoolDiskStars_eqFunction_3049(data, threadData);
  WhirlpoolDiskStars_eqFunction_3048(data, threadData);
  WhirlpoolDiskStars_eqFunction_3046(data, threadData);
  WhirlpoolDiskStars_eqFunction_783(data, threadData);
  WhirlpoolDiskStars_eqFunction_3041(data, threadData);
  WhirlpoolDiskStars_eqFunction_785(data, threadData);
  WhirlpoolDiskStars_eqFunction_3054(data, threadData);
  WhirlpoolDiskStars_eqFunction_787(data, threadData);
  WhirlpoolDiskStars_eqFunction_788(data, threadData);
  WhirlpoolDiskStars_eqFunction_3053(data, threadData);
  WhirlpoolDiskStars_eqFunction_790(data, threadData);
  WhirlpoolDiskStars_eqFunction_791(data, threadData);
  WhirlpoolDiskStars_eqFunction_3052(data, threadData);
  WhirlpoolDiskStars_eqFunction_3055(data, threadData);
  WhirlpoolDiskStars_eqFunction_3057(data, threadData);
  WhirlpoolDiskStars_eqFunction_3060(data, threadData);
  WhirlpoolDiskStars_eqFunction_3059(data, threadData);
  WhirlpoolDiskStars_eqFunction_3058(data, threadData);
  WhirlpoolDiskStars_eqFunction_3056(data, threadData);
  WhirlpoolDiskStars_eqFunction_799(data, threadData);
  WhirlpoolDiskStars_eqFunction_3051(data, threadData);
  WhirlpoolDiskStars_eqFunction_801(data, threadData);
  WhirlpoolDiskStars_eqFunction_3064(data, threadData);
  WhirlpoolDiskStars_eqFunction_803(data, threadData);
  WhirlpoolDiskStars_eqFunction_804(data, threadData);
  WhirlpoolDiskStars_eqFunction_3063(data, threadData);
  WhirlpoolDiskStars_eqFunction_806(data, threadData);
  WhirlpoolDiskStars_eqFunction_807(data, threadData);
  WhirlpoolDiskStars_eqFunction_3062(data, threadData);
  WhirlpoolDiskStars_eqFunction_3065(data, threadData);
  WhirlpoolDiskStars_eqFunction_3067(data, threadData);
  WhirlpoolDiskStars_eqFunction_3070(data, threadData);
  WhirlpoolDiskStars_eqFunction_3069(data, threadData);
  WhirlpoolDiskStars_eqFunction_3068(data, threadData);
  WhirlpoolDiskStars_eqFunction_3066(data, threadData);
  WhirlpoolDiskStars_eqFunction_815(data, threadData);
  WhirlpoolDiskStars_eqFunction_3061(data, threadData);
  WhirlpoolDiskStars_eqFunction_817(data, threadData);
  WhirlpoolDiskStars_eqFunction_3074(data, threadData);
  WhirlpoolDiskStars_eqFunction_819(data, threadData);
  WhirlpoolDiskStars_eqFunction_820(data, threadData);
  WhirlpoolDiskStars_eqFunction_3073(data, threadData);
  WhirlpoolDiskStars_eqFunction_822(data, threadData);
  WhirlpoolDiskStars_eqFunction_823(data, threadData);
  WhirlpoolDiskStars_eqFunction_3072(data, threadData);
  WhirlpoolDiskStars_eqFunction_3075(data, threadData);
  WhirlpoolDiskStars_eqFunction_3077(data, threadData);
  WhirlpoolDiskStars_eqFunction_3080(data, threadData);
  WhirlpoolDiskStars_eqFunction_3079(data, threadData);
  WhirlpoolDiskStars_eqFunction_3078(data, threadData);
  WhirlpoolDiskStars_eqFunction_3076(data, threadData);
  WhirlpoolDiskStars_eqFunction_831(data, threadData);
  WhirlpoolDiskStars_eqFunction_3071(data, threadData);
  WhirlpoolDiskStars_eqFunction_833(data, threadData);
  WhirlpoolDiskStars_eqFunction_3084(data, threadData);
  WhirlpoolDiskStars_eqFunction_835(data, threadData);
  WhirlpoolDiskStars_eqFunction_836(data, threadData);
  WhirlpoolDiskStars_eqFunction_3083(data, threadData);
  WhirlpoolDiskStars_eqFunction_838(data, threadData);
  WhirlpoolDiskStars_eqFunction_839(data, threadData);
  WhirlpoolDiskStars_eqFunction_3082(data, threadData);
  WhirlpoolDiskStars_eqFunction_3085(data, threadData);
  WhirlpoolDiskStars_eqFunction_3087(data, threadData);
  WhirlpoolDiskStars_eqFunction_3090(data, threadData);
  WhirlpoolDiskStars_eqFunction_3089(data, threadData);
  WhirlpoolDiskStars_eqFunction_3088(data, threadData);
  WhirlpoolDiskStars_eqFunction_3086(data, threadData);
  WhirlpoolDiskStars_eqFunction_847(data, threadData);
  WhirlpoolDiskStars_eqFunction_3081(data, threadData);
  WhirlpoolDiskStars_eqFunction_849(data, threadData);
  WhirlpoolDiskStars_eqFunction_3094(data, threadData);
  WhirlpoolDiskStars_eqFunction_851(data, threadData);
  WhirlpoolDiskStars_eqFunction_852(data, threadData);
  WhirlpoolDiskStars_eqFunction_3093(data, threadData);
  WhirlpoolDiskStars_eqFunction_854(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif