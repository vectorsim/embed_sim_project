#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void WhirlpoolDiskStars_eqFunction_3892(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3895(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3897(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3900(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3899(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3898(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3896(DATA *data, threadData_t *threadData);


/*
equation index: 2143
type: SIMPLE_ASSIGN
vz[134] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2143};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[453]] /* vz[134] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3891(DATA *data, threadData_t *threadData);


/*
equation index: 2145
type: SIMPLE_ASSIGN
z[135] = 2.75
*/
void WhirlpoolDiskStars_eqFunction_2145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2145};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[934]] /* z[135] STATE(1,vz[135]) */) = 2.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3904(DATA *data, threadData_t *threadData);


/*
equation index: 2147
type: SIMPLE_ASSIGN
y[135] = r_init[135] * sin(theta[135] + armOffset[135])
*/
void WhirlpoolDiskStars_eqFunction_2147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[774]] /* y[135] STATE(1,vy[135]) */) = ((data->simulationInfo->realParameter[299] /* r_init[135] PARAM */)) * (sin((data->simulationInfo->realParameter[459] /* theta[135] PARAM */) + (data->simulationInfo->realParameter[137] /* armOffset[135] PARAM */)));
  TRACE_POP
}

/*
equation index: 2148
type: SIMPLE_ASSIGN
vx[135] = (-y[135]) * sqrt(G * Md / r_init[135] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2148};
  modelica_real tmp536;
  modelica_real tmp537;
  tmp536 = (data->simulationInfo->realParameter[299] /* r_init[135] PARAM */);
  tmp537 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp536 * tmp536 * tmp536),"r_init[135] ^ 3.0",equationIndexes);
  if(!(tmp537 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[135] ^ 3.0) was %g should be >= 0", tmp537);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[134]] /* vx[135] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[774]] /* y[135] STATE(1,vy[135]) */))) * (sqrt(tmp537));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3903(DATA *data, threadData_t *threadData);


/*
equation index: 2150
type: SIMPLE_ASSIGN
x[135] = r_init[135] * cos(theta[135] + armOffset[135])
*/
void WhirlpoolDiskStars_eqFunction_2150(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2150};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[614]] /* x[135] STATE(1,vx[135]) */) = ((data->simulationInfo->realParameter[299] /* r_init[135] PARAM */)) * (cos((data->simulationInfo->realParameter[459] /* theta[135] PARAM */) + (data->simulationInfo->realParameter[137] /* armOffset[135] PARAM */)));
  TRACE_POP
}

/*
equation index: 2151
type: SIMPLE_ASSIGN
vy[135] = x[135] * sqrt(G * Md / r_init[135] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2151};
  modelica_real tmp538;
  modelica_real tmp539;
  tmp538 = (data->simulationInfo->realParameter[299] /* r_init[135] PARAM */);
  tmp539 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp538 * tmp538 * tmp538),"r_init[135] ^ 3.0",equationIndexes);
  if(!(tmp539 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[135] ^ 3.0) was %g should be >= 0", tmp539);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[294]] /* vy[135] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[614]] /* x[135] STATE(1,vx[135]) */)) * (sqrt(tmp539));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3902(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3905(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3907(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3910(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3909(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3908(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3906(DATA *data, threadData_t *threadData);


/*
equation index: 2159
type: SIMPLE_ASSIGN
vz[135] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2159};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[454]] /* vz[135] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3901(DATA *data, threadData_t *threadData);


/*
equation index: 2161
type: SIMPLE_ASSIGN
z[136] = 2.8000000000000003
*/
void WhirlpoolDiskStars_eqFunction_2161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2161};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[935]] /* z[136] STATE(1,vz[136]) */) = 2.8000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3914(DATA *data, threadData_t *threadData);


/*
equation index: 2163
type: SIMPLE_ASSIGN
y[136] = r_init[136] * sin(theta[136] + armOffset[136])
*/
void WhirlpoolDiskStars_eqFunction_2163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[775]] /* y[136] STATE(1,vy[136]) */) = ((data->simulationInfo->realParameter[300] /* r_init[136] PARAM */)) * (sin((data->simulationInfo->realParameter[460] /* theta[136] PARAM */) + (data->simulationInfo->realParameter[138] /* armOffset[136] PARAM */)));
  TRACE_POP
}

/*
equation index: 2164
type: SIMPLE_ASSIGN
vx[136] = (-y[136]) * sqrt(G * Md / r_init[136] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2164};
  modelica_real tmp540;
  modelica_real tmp541;
  tmp540 = (data->simulationInfo->realParameter[300] /* r_init[136] PARAM */);
  tmp541 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp540 * tmp540 * tmp540),"r_init[136] ^ 3.0",equationIndexes);
  if(!(tmp541 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[136] ^ 3.0) was %g should be >= 0", tmp541);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[135]] /* vx[136] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[775]] /* y[136] STATE(1,vy[136]) */))) * (sqrt(tmp541));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3913(DATA *data, threadData_t *threadData);


/*
equation index: 2166
type: SIMPLE_ASSIGN
x[136] = r_init[136] * cos(theta[136] + armOffset[136])
*/
void WhirlpoolDiskStars_eqFunction_2166(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2166};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[615]] /* x[136] STATE(1,vx[136]) */) = ((data->simulationInfo->realParameter[300] /* r_init[136] PARAM */)) * (cos((data->simulationInfo->realParameter[460] /* theta[136] PARAM */) + (data->simulationInfo->realParameter[138] /* armOffset[136] PARAM */)));
  TRACE_POP
}

/*
equation index: 2167
type: SIMPLE_ASSIGN
vy[136] = x[136] * sqrt(G * Md / r_init[136] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2167};
  modelica_real tmp542;
  modelica_real tmp543;
  tmp542 = (data->simulationInfo->realParameter[300] /* r_init[136] PARAM */);
  tmp543 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp542 * tmp542 * tmp542),"r_init[136] ^ 3.0",equationIndexes);
  if(!(tmp543 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[136] ^ 3.0) was %g should be >= 0", tmp543);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[295]] /* vy[136] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[615]] /* x[136] STATE(1,vx[136]) */)) * (sqrt(tmp543));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3912(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3915(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3917(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3920(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3919(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3918(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3916(DATA *data, threadData_t *threadData);


/*
equation index: 2175
type: SIMPLE_ASSIGN
vz[136] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2175};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[455]] /* vz[136] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3911(DATA *data, threadData_t *threadData);


/*
equation index: 2177
type: SIMPLE_ASSIGN
z[137] = 2.85
*/
void WhirlpoolDiskStars_eqFunction_2177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2177};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[936]] /* z[137] STATE(1,vz[137]) */) = 2.85;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3924(DATA *data, threadData_t *threadData);


/*
equation index: 2179
type: SIMPLE_ASSIGN
y[137] = r_init[137] * sin(theta[137] + armOffset[137])
*/
void WhirlpoolDiskStars_eqFunction_2179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[776]] /* y[137] STATE(1,vy[137]) */) = ((data->simulationInfo->realParameter[301] /* r_init[137] PARAM */)) * (sin((data->simulationInfo->realParameter[461] /* theta[137] PARAM */) + (data->simulationInfo->realParameter[139] /* armOffset[137] PARAM */)));
  TRACE_POP
}

/*
equation index: 2180
type: SIMPLE_ASSIGN
vx[137] = (-y[137]) * sqrt(G * Md / r_init[137] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2180};
  modelica_real tmp544;
  modelica_real tmp545;
  tmp544 = (data->simulationInfo->realParameter[301] /* r_init[137] PARAM */);
  tmp545 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp544 * tmp544 * tmp544),"r_init[137] ^ 3.0",equationIndexes);
  if(!(tmp545 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[137] ^ 3.0) was %g should be >= 0", tmp545);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[136]] /* vx[137] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[776]] /* y[137] STATE(1,vy[137]) */))) * (sqrt(tmp545));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3923(DATA *data, threadData_t *threadData);


/*
equation index: 2182
type: SIMPLE_ASSIGN
x[137] = r_init[137] * cos(theta[137] + armOffset[137])
*/
void WhirlpoolDiskStars_eqFunction_2182(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2182};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[616]] /* x[137] STATE(1,vx[137]) */) = ((data->simulationInfo->realParameter[301] /* r_init[137] PARAM */)) * (cos((data->simulationInfo->realParameter[461] /* theta[137] PARAM */) + (data->simulationInfo->realParameter[139] /* armOffset[137] PARAM */)));
  TRACE_POP
}

/*
equation index: 2183
type: SIMPLE_ASSIGN
vy[137] = x[137] * sqrt(G * Md / r_init[137] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2183};
  modelica_real tmp546;
  modelica_real tmp547;
  tmp546 = (data->simulationInfo->realParameter[301] /* r_init[137] PARAM */);
  tmp547 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp546 * tmp546 * tmp546),"r_init[137] ^ 3.0",equationIndexes);
  if(!(tmp547 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[137] ^ 3.0) was %g should be >= 0", tmp547);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[296]] /* vy[137] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[616]] /* x[137] STATE(1,vx[137]) */)) * (sqrt(tmp547));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3922(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3925(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3927(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3930(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3929(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3928(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3926(DATA *data, threadData_t *threadData);


/*
equation index: 2191
type: SIMPLE_ASSIGN
vz[137] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2191};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[456]] /* vz[137] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3921(DATA *data, threadData_t *threadData);


/*
equation index: 2193
type: SIMPLE_ASSIGN
z[138] = 2.9000000000000004
*/
void WhirlpoolDiskStars_eqFunction_2193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2193};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[937]] /* z[138] STATE(1,vz[138]) */) = 2.9000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3934(DATA *data, threadData_t *threadData);


/*
equation index: 2195
type: SIMPLE_ASSIGN
y[138] = r_init[138] * sin(theta[138] + armOffset[138])
*/
void WhirlpoolDiskStars_eqFunction_2195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[777]] /* y[138] STATE(1,vy[138]) */) = ((data->simulationInfo->realParameter[302] /* r_init[138] PARAM */)) * (sin((data->simulationInfo->realParameter[462] /* theta[138] PARAM */) + (data->simulationInfo->realParameter[140] /* armOffset[138] PARAM */)));
  TRACE_POP
}

/*
equation index: 2196
type: SIMPLE_ASSIGN
vx[138] = (-y[138]) * sqrt(G * Md / r_init[138] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2196};
  modelica_real tmp548;
  modelica_real tmp549;
  tmp548 = (data->simulationInfo->realParameter[302] /* r_init[138] PARAM */);
  tmp549 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp548 * tmp548 * tmp548),"r_init[138] ^ 3.0",equationIndexes);
  if(!(tmp549 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[138] ^ 3.0) was %g should be >= 0", tmp549);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[137]] /* vx[138] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[777]] /* y[138] STATE(1,vy[138]) */))) * (sqrt(tmp549));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3933(DATA *data, threadData_t *threadData);


/*
equation index: 2198
type: SIMPLE_ASSIGN
x[138] = r_init[138] * cos(theta[138] + armOffset[138])
*/
void WhirlpoolDiskStars_eqFunction_2198(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2198};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[617]] /* x[138] STATE(1,vx[138]) */) = ((data->simulationInfo->realParameter[302] /* r_init[138] PARAM */)) * (cos((data->simulationInfo->realParameter[462] /* theta[138] PARAM */) + (data->simulationInfo->realParameter[140] /* armOffset[138] PARAM */)));
  TRACE_POP
}

/*
equation index: 2199
type: SIMPLE_ASSIGN
vy[138] = x[138] * sqrt(G * Md / r_init[138] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2199};
  modelica_real tmp550;
  modelica_real tmp551;
  tmp550 = (data->simulationInfo->realParameter[302] /* r_init[138] PARAM */);
  tmp551 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp550 * tmp550 * tmp550),"r_init[138] ^ 3.0",equationIndexes);
  if(!(tmp551 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[138] ^ 3.0) was %g should be >= 0", tmp551);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[297]] /* vy[138] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[617]] /* x[138] STATE(1,vx[138]) */)) * (sqrt(tmp551));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3932(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3935(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3937(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3940(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3939(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3938(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3936(DATA *data, threadData_t *threadData);


/*
equation index: 2207
type: SIMPLE_ASSIGN
vz[138] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2207};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[457]] /* vz[138] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3931(DATA *data, threadData_t *threadData);


/*
equation index: 2209
type: SIMPLE_ASSIGN
z[139] = 2.95
*/
void WhirlpoolDiskStars_eqFunction_2209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2209};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[938]] /* z[139] STATE(1,vz[139]) */) = 2.95;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3944(DATA *data, threadData_t *threadData);


/*
equation index: 2211
type: SIMPLE_ASSIGN
y[139] = r_init[139] * sin(theta[139] + armOffset[139])
*/
void WhirlpoolDiskStars_eqFunction_2211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[778]] /* y[139] STATE(1,vy[139]) */) = ((data->simulationInfo->realParameter[303] /* r_init[139] PARAM */)) * (sin((data->simulationInfo->realParameter[463] /* theta[139] PARAM */) + (data->simulationInfo->realParameter[141] /* armOffset[139] PARAM */)));
  TRACE_POP
}

/*
equation index: 2212
type: SIMPLE_ASSIGN
vx[139] = (-y[139]) * sqrt(G * Md / r_init[139] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2212};
  modelica_real tmp552;
  modelica_real tmp553;
  tmp552 = (data->simulationInfo->realParameter[303] /* r_init[139] PARAM */);
  tmp553 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp552 * tmp552 * tmp552),"r_init[139] ^ 3.0",equationIndexes);
  if(!(tmp553 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[139] ^ 3.0) was %g should be >= 0", tmp553);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[138]] /* vx[139] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[778]] /* y[139] STATE(1,vy[139]) */))) * (sqrt(tmp553));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3943(DATA *data, threadData_t *threadData);


/*
equation index: 2214
type: SIMPLE_ASSIGN
x[139] = r_init[139] * cos(theta[139] + armOffset[139])
*/
void WhirlpoolDiskStars_eqFunction_2214(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2214};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[618]] /* x[139] STATE(1,vx[139]) */) = ((data->simulationInfo->realParameter[303] /* r_init[139] PARAM */)) * (cos((data->simulationInfo->realParameter[463] /* theta[139] PARAM */) + (data->simulationInfo->realParameter[141] /* armOffset[139] PARAM */)));
  TRACE_POP
}

/*
equation index: 2215
type: SIMPLE_ASSIGN
vy[139] = x[139] * sqrt(G * Md / r_init[139] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2215};
  modelica_real tmp554;
  modelica_real tmp555;
  tmp554 = (data->simulationInfo->realParameter[303] /* r_init[139] PARAM */);
  tmp555 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp554 * tmp554 * tmp554),"r_init[139] ^ 3.0",equationIndexes);
  if(!(tmp555 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[139] ^ 3.0) was %g should be >= 0", tmp555);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[298]] /* vy[139] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[618]] /* x[139] STATE(1,vx[139]) */)) * (sqrt(tmp555));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3942(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3945(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3947(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3950(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3949(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3948(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3946(DATA *data, threadData_t *threadData);


/*
equation index: 2223
type: SIMPLE_ASSIGN
vz[139] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2223};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[458]] /* vz[139] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3941(DATA *data, threadData_t *threadData);


/*
equation index: 2225
type: SIMPLE_ASSIGN
z[140] = 3.0
*/
void WhirlpoolDiskStars_eqFunction_2225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2225};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[939]] /* z[140] STATE(1,vz[140]) */) = 3.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3954(DATA *data, threadData_t *threadData);


/*
equation index: 2227
type: SIMPLE_ASSIGN
y[140] = r_init[140] * sin(theta[140] + armOffset[140])
*/
void WhirlpoolDiskStars_eqFunction_2227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[779]] /* y[140] STATE(1,vy[140]) */) = ((data->simulationInfo->realParameter[304] /* r_init[140] PARAM */)) * (sin((data->simulationInfo->realParameter[464] /* theta[140] PARAM */) + (data->simulationInfo->realParameter[142] /* armOffset[140] PARAM */)));
  TRACE_POP
}

/*
equation index: 2228
type: SIMPLE_ASSIGN
vx[140] = (-y[140]) * sqrt(G * Md / r_init[140] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2228};
  modelica_real tmp556;
  modelica_real tmp557;
  tmp556 = (data->simulationInfo->realParameter[304] /* r_init[140] PARAM */);
  tmp557 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp556 * tmp556 * tmp556),"r_init[140] ^ 3.0",equationIndexes);
  if(!(tmp557 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[140] ^ 3.0) was %g should be >= 0", tmp557);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[139]] /* vx[140] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[779]] /* y[140] STATE(1,vy[140]) */))) * (sqrt(tmp557));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3953(DATA *data, threadData_t *threadData);


/*
equation index: 2230
type: SIMPLE_ASSIGN
x[140] = r_init[140] * cos(theta[140] + armOffset[140])
*/
void WhirlpoolDiskStars_eqFunction_2230(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2230};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[619]] /* x[140] STATE(1,vx[140]) */) = ((data->simulationInfo->realParameter[304] /* r_init[140] PARAM */)) * (cos((data->simulationInfo->realParameter[464] /* theta[140] PARAM */) + (data->simulationInfo->realParameter[142] /* armOffset[140] PARAM */)));
  TRACE_POP
}

/*
equation index: 2231
type: SIMPLE_ASSIGN
vy[140] = x[140] * sqrt(G * Md / r_init[140] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2231};
  modelica_real tmp558;
  modelica_real tmp559;
  tmp558 = (data->simulationInfo->realParameter[304] /* r_init[140] PARAM */);
  tmp559 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp558 * tmp558 * tmp558),"r_init[140] ^ 3.0",equationIndexes);
  if(!(tmp559 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[140] ^ 3.0) was %g should be >= 0", tmp559);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[299]] /* vy[140] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[619]] /* x[140] STATE(1,vx[140]) */)) * (sqrt(tmp559));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3952(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3955(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3957(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3960(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3959(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3958(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3956(DATA *data, threadData_t *threadData);


/*
equation index: 2239
type: SIMPLE_ASSIGN
vz[140] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2239};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[459]] /* vz[140] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3951(DATA *data, threadData_t *threadData);


/*
equation index: 2241
type: SIMPLE_ASSIGN
z[141] = 3.0500000000000003
*/
void WhirlpoolDiskStars_eqFunction_2241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2241};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[940]] /* z[141] STATE(1,vz[141]) */) = 3.0500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3964(DATA *data, threadData_t *threadData);


/*
equation index: 2243
type: SIMPLE_ASSIGN
y[141] = r_init[141] * sin(theta[141] + armOffset[141])
*/
void WhirlpoolDiskStars_eqFunction_2243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[780]] /* y[141] STATE(1,vy[141]) */) = ((data->simulationInfo->realParameter[305] /* r_init[141] PARAM */)) * (sin((data->simulationInfo->realParameter[465] /* theta[141] PARAM */) + (data->simulationInfo->realParameter[143] /* armOffset[141] PARAM */)));
  TRACE_POP
}

/*
equation index: 2244
type: SIMPLE_ASSIGN
vx[141] = (-y[141]) * sqrt(G * Md / r_init[141] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2244};
  modelica_real tmp560;
  modelica_real tmp561;
  tmp560 = (data->simulationInfo->realParameter[305] /* r_init[141] PARAM */);
  tmp561 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp560 * tmp560 * tmp560),"r_init[141] ^ 3.0",equationIndexes);
  if(!(tmp561 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[141] ^ 3.0) was %g should be >= 0", tmp561);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[140]] /* vx[141] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[780]] /* y[141] STATE(1,vy[141]) */))) * (sqrt(tmp561));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3963(DATA *data, threadData_t *threadData);


/*
equation index: 2246
type: SIMPLE_ASSIGN
x[141] = r_init[141] * cos(theta[141] + armOffset[141])
*/
void WhirlpoolDiskStars_eqFunction_2246(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2246};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[620]] /* x[141] STATE(1,vx[141]) */) = ((data->simulationInfo->realParameter[305] /* r_init[141] PARAM */)) * (cos((data->simulationInfo->realParameter[465] /* theta[141] PARAM */) + (data->simulationInfo->realParameter[143] /* armOffset[141] PARAM */)));
  TRACE_POP
}

/*
equation index: 2247
type: SIMPLE_ASSIGN
vy[141] = x[141] * sqrt(G * Md / r_init[141] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2247};
  modelica_real tmp562;
  modelica_real tmp563;
  tmp562 = (data->simulationInfo->realParameter[305] /* r_init[141] PARAM */);
  tmp563 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp562 * tmp562 * tmp562),"r_init[141] ^ 3.0",equationIndexes);
  if(!(tmp563 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[141] ^ 3.0) was %g should be >= 0", tmp563);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[300]] /* vy[141] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[620]] /* x[141] STATE(1,vx[141]) */)) * (sqrt(tmp563));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3962(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3965(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3967(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3970(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3969(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3968(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3966(DATA *data, threadData_t *threadData);


/*
equation index: 2255
type: SIMPLE_ASSIGN
vz[141] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2255};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[460]] /* vz[141] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3961(DATA *data, threadData_t *threadData);


/*
equation index: 2257
type: SIMPLE_ASSIGN
z[142] = 3.1
*/
void WhirlpoolDiskStars_eqFunction_2257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2257};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[941]] /* z[142] STATE(1,vz[142]) */) = 3.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3974(DATA *data, threadData_t *threadData);


/*
equation index: 2259
type: SIMPLE_ASSIGN
y[142] = r_init[142] * sin(theta[142] + armOffset[142])
*/
void WhirlpoolDiskStars_eqFunction_2259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[781]] /* y[142] STATE(1,vy[142]) */) = ((data->simulationInfo->realParameter[306] /* r_init[142] PARAM */)) * (sin((data->simulationInfo->realParameter[466] /* theta[142] PARAM */) + (data->simulationInfo->realParameter[144] /* armOffset[142] PARAM */)));
  TRACE_POP
}

/*
equation index: 2260
type: SIMPLE_ASSIGN
vx[142] = (-y[142]) * sqrt(G * Md / r_init[142] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2260};
  modelica_real tmp564;
  modelica_real tmp565;
  tmp564 = (data->simulationInfo->realParameter[306] /* r_init[142] PARAM */);
  tmp565 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp564 * tmp564 * tmp564),"r_init[142] ^ 3.0",equationIndexes);
  if(!(tmp565 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[142] ^ 3.0) was %g should be >= 0", tmp565);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[141]] /* vx[142] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[781]] /* y[142] STATE(1,vy[142]) */))) * (sqrt(tmp565));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3973(DATA *data, threadData_t *threadData);


/*
equation index: 2262
type: SIMPLE_ASSIGN
x[142] = r_init[142] * cos(theta[142] + armOffset[142])
*/
void WhirlpoolDiskStars_eqFunction_2262(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2262};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[621]] /* x[142] STATE(1,vx[142]) */) = ((data->simulationInfo->realParameter[306] /* r_init[142] PARAM */)) * (cos((data->simulationInfo->realParameter[466] /* theta[142] PARAM */) + (data->simulationInfo->realParameter[144] /* armOffset[142] PARAM */)));
  TRACE_POP
}

/*
equation index: 2263
type: SIMPLE_ASSIGN
vy[142] = x[142] * sqrt(G * Md / r_init[142] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2263};
  modelica_real tmp566;
  modelica_real tmp567;
  tmp566 = (data->simulationInfo->realParameter[306] /* r_init[142] PARAM */);
  tmp567 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp566 * tmp566 * tmp566),"r_init[142] ^ 3.0",equationIndexes);
  if(!(tmp567 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[142] ^ 3.0) was %g should be >= 0", tmp567);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[301]] /* vy[142] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[621]] /* x[142] STATE(1,vx[142]) */)) * (sqrt(tmp567));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3972(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3975(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3977(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3980(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3979(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3978(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3976(DATA *data, threadData_t *threadData);


/*
equation index: 2271
type: SIMPLE_ASSIGN
vz[142] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[461]] /* vz[142] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3971(DATA *data, threadData_t *threadData);


/*
equation index: 2273
type: SIMPLE_ASSIGN
z[143] = 3.1500000000000004
*/
void WhirlpoolDiskStars_eqFunction_2273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[942]] /* z[143] STATE(1,vz[143]) */) = 3.1500000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3984(DATA *data, threadData_t *threadData);


/*
equation index: 2275
type: SIMPLE_ASSIGN
y[143] = r_init[143] * sin(theta[143] + armOffset[143])
*/
void WhirlpoolDiskStars_eqFunction_2275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[782]] /* y[143] STATE(1,vy[143]) */) = ((data->simulationInfo->realParameter[307] /* r_init[143] PARAM */)) * (sin((data->simulationInfo->realParameter[467] /* theta[143] PARAM */) + (data->simulationInfo->realParameter[145] /* armOffset[143] PARAM */)));
  TRACE_POP
}

/*
equation index: 2276
type: SIMPLE_ASSIGN
vx[143] = (-y[143]) * sqrt(G * Md / r_init[143] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2276};
  modelica_real tmp568;
  modelica_real tmp569;
  tmp568 = (data->simulationInfo->realParameter[307] /* r_init[143] PARAM */);
  tmp569 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp568 * tmp568 * tmp568),"r_init[143] ^ 3.0",equationIndexes);
  if(!(tmp569 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[143] ^ 3.0) was %g should be >= 0", tmp569);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[142]] /* vx[143] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[782]] /* y[143] STATE(1,vy[143]) */))) * (sqrt(tmp569));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3983(DATA *data, threadData_t *threadData);


/*
equation index: 2278
type: SIMPLE_ASSIGN
x[143] = r_init[143] * cos(theta[143] + armOffset[143])
*/
void WhirlpoolDiskStars_eqFunction_2278(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2278};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[622]] /* x[143] STATE(1,vx[143]) */) = ((data->simulationInfo->realParameter[307] /* r_init[143] PARAM */)) * (cos((data->simulationInfo->realParameter[467] /* theta[143] PARAM */) + (data->simulationInfo->realParameter[145] /* armOffset[143] PARAM */)));
  TRACE_POP
}

/*
equation index: 2279
type: SIMPLE_ASSIGN
vy[143] = x[143] * sqrt(G * Md / r_init[143] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2279};
  modelica_real tmp570;
  modelica_real tmp571;
  tmp570 = (data->simulationInfo->realParameter[307] /* r_init[143] PARAM */);
  tmp571 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp570 * tmp570 * tmp570),"r_init[143] ^ 3.0",equationIndexes);
  if(!(tmp571 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[143] ^ 3.0) was %g should be >= 0", tmp571);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[302]] /* vy[143] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[622]] /* x[143] STATE(1,vx[143]) */)) * (sqrt(tmp571));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3982(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3985(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3987(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3990(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3989(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3988(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3986(DATA *data, threadData_t *threadData);


/*
equation index: 2287
type: SIMPLE_ASSIGN
vz[143] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2287};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[462]] /* vz[143] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3981(DATA *data, threadData_t *threadData);


/*
equation index: 2289
type: SIMPLE_ASSIGN
z[144] = 3.2
*/
void WhirlpoolDiskStars_eqFunction_2289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2289};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[943]] /* z[144] STATE(1,vz[144]) */) = 3.2;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3994(DATA *data, threadData_t *threadData);


/*
equation index: 2291
type: SIMPLE_ASSIGN
y[144] = r_init[144] * sin(theta[144] + armOffset[144])
*/
void WhirlpoolDiskStars_eqFunction_2291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[783]] /* y[144] STATE(1,vy[144]) */) = ((data->simulationInfo->realParameter[308] /* r_init[144] PARAM */)) * (sin((data->simulationInfo->realParameter[468] /* theta[144] PARAM */) + (data->simulationInfo->realParameter[146] /* armOffset[144] PARAM */)));
  TRACE_POP
}

/*
equation index: 2292
type: SIMPLE_ASSIGN
vx[144] = (-y[144]) * sqrt(G * Md / r_init[144] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2292};
  modelica_real tmp572;
  modelica_real tmp573;
  tmp572 = (data->simulationInfo->realParameter[308] /* r_init[144] PARAM */);
  tmp573 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp572 * tmp572 * tmp572),"r_init[144] ^ 3.0",equationIndexes);
  if(!(tmp573 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[144] ^ 3.0) was %g should be >= 0", tmp573);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[143]] /* vx[144] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[783]] /* y[144] STATE(1,vy[144]) */))) * (sqrt(tmp573));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3993(DATA *data, threadData_t *threadData);


/*
equation index: 2294
type: SIMPLE_ASSIGN
x[144] = r_init[144] * cos(theta[144] + armOffset[144])
*/
void WhirlpoolDiskStars_eqFunction_2294(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2294};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[623]] /* x[144] STATE(1,vx[144]) */) = ((data->simulationInfo->realParameter[308] /* r_init[144] PARAM */)) * (cos((data->simulationInfo->realParameter[468] /* theta[144] PARAM */) + (data->simulationInfo->realParameter[146] /* armOffset[144] PARAM */)));
  TRACE_POP
}

/*
equation index: 2295
type: SIMPLE_ASSIGN
vy[144] = x[144] * sqrt(G * Md / r_init[144] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2295};
  modelica_real tmp574;
  modelica_real tmp575;
  tmp574 = (data->simulationInfo->realParameter[308] /* r_init[144] PARAM */);
  tmp575 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp574 * tmp574 * tmp574),"r_init[144] ^ 3.0",equationIndexes);
  if(!(tmp575 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[144] ^ 3.0) was %g should be >= 0", tmp575);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[303]] /* vy[144] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[623]] /* x[144] STATE(1,vx[144]) */)) * (sqrt(tmp575));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3992(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3995(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3997(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4000(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3999(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3998(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3996(DATA *data, threadData_t *threadData);


/*
equation index: 2303
type: SIMPLE_ASSIGN
vz[144] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2303};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[463]] /* vz[144] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3991(DATA *data, threadData_t *threadData);


/*
equation index: 2305
type: SIMPLE_ASSIGN
z[145] = 3.25
*/
void WhirlpoolDiskStars_eqFunction_2305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2305};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[944]] /* z[145] STATE(1,vz[145]) */) = 3.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4004(DATA *data, threadData_t *threadData);


/*
equation index: 2307
type: SIMPLE_ASSIGN
y[145] = r_init[145] * sin(theta[145] + armOffset[145])
*/
void WhirlpoolDiskStars_eqFunction_2307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[784]] /* y[145] STATE(1,vy[145]) */) = ((data->simulationInfo->realParameter[309] /* r_init[145] PARAM */)) * (sin((data->simulationInfo->realParameter[469] /* theta[145] PARAM */) + (data->simulationInfo->realParameter[147] /* armOffset[145] PARAM */)));
  TRACE_POP
}

/*
equation index: 2308
type: SIMPLE_ASSIGN
vx[145] = (-y[145]) * sqrt(G * Md / r_init[145] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2308};
  modelica_real tmp576;
  modelica_real tmp577;
  tmp576 = (data->simulationInfo->realParameter[309] /* r_init[145] PARAM */);
  tmp577 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp576 * tmp576 * tmp576),"r_init[145] ^ 3.0",equationIndexes);
  if(!(tmp577 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[145] ^ 3.0) was %g should be >= 0", tmp577);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[144]] /* vx[145] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[784]] /* y[145] STATE(1,vy[145]) */))) * (sqrt(tmp577));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4003(DATA *data, threadData_t *threadData);


/*
equation index: 2310
type: SIMPLE_ASSIGN
x[145] = r_init[145] * cos(theta[145] + armOffset[145])
*/
void WhirlpoolDiskStars_eqFunction_2310(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2310};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[624]] /* x[145] STATE(1,vx[145]) */) = ((data->simulationInfo->realParameter[309] /* r_init[145] PARAM */)) * (cos((data->simulationInfo->realParameter[469] /* theta[145] PARAM */) + (data->simulationInfo->realParameter[147] /* armOffset[145] PARAM */)));
  TRACE_POP
}

/*
equation index: 2311
type: SIMPLE_ASSIGN
vy[145] = x[145] * sqrt(G * Md / r_init[145] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2311};
  modelica_real tmp578;
  modelica_real tmp579;
  tmp578 = (data->simulationInfo->realParameter[309] /* r_init[145] PARAM */);
  tmp579 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp578 * tmp578 * tmp578),"r_init[145] ^ 3.0",equationIndexes);
  if(!(tmp579 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[145] ^ 3.0) was %g should be >= 0", tmp579);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[304]] /* vy[145] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[624]] /* x[145] STATE(1,vx[145]) */)) * (sqrt(tmp579));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4002(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4005(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4007(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4010(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4009(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4008(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4006(DATA *data, threadData_t *threadData);


/*
equation index: 2319
type: SIMPLE_ASSIGN
vz[145] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[464]] /* vz[145] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4001(DATA *data, threadData_t *threadData);


/*
equation index: 2321
type: SIMPLE_ASSIGN
z[146] = 3.3000000000000003
*/
void WhirlpoolDiskStars_eqFunction_2321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[945]] /* z[146] STATE(1,vz[146]) */) = 3.3000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4014(DATA *data, threadData_t *threadData);


/*
equation index: 2323
type: SIMPLE_ASSIGN
y[146] = r_init[146] * sin(theta[146] + armOffset[146])
*/
void WhirlpoolDiskStars_eqFunction_2323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[785]] /* y[146] STATE(1,vy[146]) */) = ((data->simulationInfo->realParameter[310] /* r_init[146] PARAM */)) * (sin((data->simulationInfo->realParameter[470] /* theta[146] PARAM */) + (data->simulationInfo->realParameter[148] /* armOffset[146] PARAM */)));
  TRACE_POP
}

/*
equation index: 2324
type: SIMPLE_ASSIGN
vx[146] = (-y[146]) * sqrt(G * Md / r_init[146] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2324};
  modelica_real tmp580;
  modelica_real tmp581;
  tmp580 = (data->simulationInfo->realParameter[310] /* r_init[146] PARAM */);
  tmp581 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp580 * tmp580 * tmp580),"r_init[146] ^ 3.0",equationIndexes);
  if(!(tmp581 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[146] ^ 3.0) was %g should be >= 0", tmp581);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[145]] /* vx[146] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[785]] /* y[146] STATE(1,vy[146]) */))) * (sqrt(tmp581));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4013(DATA *data, threadData_t *threadData);


/*
equation index: 2326
type: SIMPLE_ASSIGN
x[146] = r_init[146] * cos(theta[146] + armOffset[146])
*/
void WhirlpoolDiskStars_eqFunction_2326(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2326};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[625]] /* x[146] STATE(1,vx[146]) */) = ((data->simulationInfo->realParameter[310] /* r_init[146] PARAM */)) * (cos((data->simulationInfo->realParameter[470] /* theta[146] PARAM */) + (data->simulationInfo->realParameter[148] /* armOffset[146] PARAM */)));
  TRACE_POP
}

/*
equation index: 2327
type: SIMPLE_ASSIGN
vy[146] = x[146] * sqrt(G * Md / r_init[146] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2327};
  modelica_real tmp582;
  modelica_real tmp583;
  tmp582 = (data->simulationInfo->realParameter[310] /* r_init[146] PARAM */);
  tmp583 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp582 * tmp582 * tmp582),"r_init[146] ^ 3.0",equationIndexes);
  if(!(tmp583 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[146] ^ 3.0) was %g should be >= 0", tmp583);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[305]] /* vy[146] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[625]] /* x[146] STATE(1,vx[146]) */)) * (sqrt(tmp583));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4012(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4015(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4017(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4020(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4019(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4018(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4016(DATA *data, threadData_t *threadData);


/*
equation index: 2335
type: SIMPLE_ASSIGN
vz[146] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[465]] /* vz[146] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4011(DATA *data, threadData_t *threadData);


/*
equation index: 2337
type: SIMPLE_ASSIGN
z[147] = 3.35
*/
void WhirlpoolDiskStars_eqFunction_2337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[946]] /* z[147] STATE(1,vz[147]) */) = 3.35;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4024(DATA *data, threadData_t *threadData);


/*
equation index: 2339
type: SIMPLE_ASSIGN
y[147] = r_init[147] * sin(theta[147] + armOffset[147])
*/
void WhirlpoolDiskStars_eqFunction_2339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[786]] /* y[147] STATE(1,vy[147]) */) = ((data->simulationInfo->realParameter[311] /* r_init[147] PARAM */)) * (sin((data->simulationInfo->realParameter[471] /* theta[147] PARAM */) + (data->simulationInfo->realParameter[149] /* armOffset[147] PARAM */)));
  TRACE_POP
}

/*
equation index: 2340
type: SIMPLE_ASSIGN
vx[147] = (-y[147]) * sqrt(G * Md / r_init[147] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2340};
  modelica_real tmp584;
  modelica_real tmp585;
  tmp584 = (data->simulationInfo->realParameter[311] /* r_init[147] PARAM */);
  tmp585 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp584 * tmp584 * tmp584),"r_init[147] ^ 3.0",equationIndexes);
  if(!(tmp585 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[147] ^ 3.0) was %g should be >= 0", tmp585);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[146]] /* vx[147] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[786]] /* y[147] STATE(1,vy[147]) */))) * (sqrt(tmp585));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4023(DATA *data, threadData_t *threadData);


/*
equation index: 2342
type: SIMPLE_ASSIGN
x[147] = r_init[147] * cos(theta[147] + armOffset[147])
*/
void WhirlpoolDiskStars_eqFunction_2342(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2342};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[626]] /* x[147] STATE(1,vx[147]) */) = ((data->simulationInfo->realParameter[311] /* r_init[147] PARAM */)) * (cos((data->simulationInfo->realParameter[471] /* theta[147] PARAM */) + (data->simulationInfo->realParameter[149] /* armOffset[147] PARAM */)));
  TRACE_POP
}

/*
equation index: 2343
type: SIMPLE_ASSIGN
vy[147] = x[147] * sqrt(G * Md / r_init[147] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2343};
  modelica_real tmp586;
  modelica_real tmp587;
  tmp586 = (data->simulationInfo->realParameter[311] /* r_init[147] PARAM */);
  tmp587 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp586 * tmp586 * tmp586),"r_init[147] ^ 3.0",equationIndexes);
  if(!(tmp587 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[147] ^ 3.0) was %g should be >= 0", tmp587);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[306]] /* vy[147] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[626]] /* x[147] STATE(1,vx[147]) */)) * (sqrt(tmp587));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4022(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4025(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4027(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4030(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4029(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4028(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4026(DATA *data, threadData_t *threadData);


/*
equation index: 2351
type: SIMPLE_ASSIGN
vz[147] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2351};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[466]] /* vz[147] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4021(DATA *data, threadData_t *threadData);


/*
equation index: 2353
type: SIMPLE_ASSIGN
z[148] = 3.4000000000000004
*/
void WhirlpoolDiskStars_eqFunction_2353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2353};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[947]] /* z[148] STATE(1,vz[148]) */) = 3.4000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4034(DATA *data, threadData_t *threadData);


/*
equation index: 2355
type: SIMPLE_ASSIGN
y[148] = r_init[148] * sin(theta[148] + armOffset[148])
*/
void WhirlpoolDiskStars_eqFunction_2355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[787]] /* y[148] STATE(1,vy[148]) */) = ((data->simulationInfo->realParameter[312] /* r_init[148] PARAM */)) * (sin((data->simulationInfo->realParameter[472] /* theta[148] PARAM */) + (data->simulationInfo->realParameter[150] /* armOffset[148] PARAM */)));
  TRACE_POP
}

/*
equation index: 2356
type: SIMPLE_ASSIGN
vx[148] = (-y[148]) * sqrt(G * Md / r_init[148] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2356};
  modelica_real tmp588;
  modelica_real tmp589;
  tmp588 = (data->simulationInfo->realParameter[312] /* r_init[148] PARAM */);
  tmp589 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp588 * tmp588 * tmp588),"r_init[148] ^ 3.0",equationIndexes);
  if(!(tmp589 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[148] ^ 3.0) was %g should be >= 0", tmp589);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[147]] /* vx[148] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[787]] /* y[148] STATE(1,vy[148]) */))) * (sqrt(tmp589));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4033(DATA *data, threadData_t *threadData);


/*
equation index: 2358
type: SIMPLE_ASSIGN
x[148] = r_init[148] * cos(theta[148] + armOffset[148])
*/
void WhirlpoolDiskStars_eqFunction_2358(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2358};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[627]] /* x[148] STATE(1,vx[148]) */) = ((data->simulationInfo->realParameter[312] /* r_init[148] PARAM */)) * (cos((data->simulationInfo->realParameter[472] /* theta[148] PARAM */) + (data->simulationInfo->realParameter[150] /* armOffset[148] PARAM */)));
  TRACE_POP
}

/*
equation index: 2359
type: SIMPLE_ASSIGN
vy[148] = x[148] * sqrt(G * Md / r_init[148] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2359};
  modelica_real tmp590;
  modelica_real tmp591;
  tmp590 = (data->simulationInfo->realParameter[312] /* r_init[148] PARAM */);
  tmp591 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp590 * tmp590 * tmp590),"r_init[148] ^ 3.0",equationIndexes);
  if(!(tmp591 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[148] ^ 3.0) was %g should be >= 0", tmp591);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[307]] /* vy[148] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[627]] /* x[148] STATE(1,vx[148]) */)) * (sqrt(tmp591));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4032(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4035(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4037(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4040(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4039(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4038(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4036(DATA *data, threadData_t *threadData);


/*
equation index: 2367
type: SIMPLE_ASSIGN
vz[148] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2367};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[467]] /* vz[148] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4031(DATA *data, threadData_t *threadData);


/*
equation index: 2369
type: SIMPLE_ASSIGN
z[149] = 3.45
*/
void WhirlpoolDiskStars_eqFunction_2369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2369};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[948]] /* z[149] STATE(1,vz[149]) */) = 3.45;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4044(DATA *data, threadData_t *threadData);


/*
equation index: 2371
type: SIMPLE_ASSIGN
y[149] = r_init[149] * sin(theta[149] + armOffset[149])
*/
void WhirlpoolDiskStars_eqFunction_2371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2371};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[788]] /* y[149] STATE(1,vy[149]) */) = ((data->simulationInfo->realParameter[313] /* r_init[149] PARAM */)) * (sin((data->simulationInfo->realParameter[473] /* theta[149] PARAM */) + (data->simulationInfo->realParameter[151] /* armOffset[149] PARAM */)));
  TRACE_POP
}

/*
equation index: 2372
type: SIMPLE_ASSIGN
vx[149] = (-y[149]) * sqrt(G * Md / r_init[149] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2372};
  modelica_real tmp592;
  modelica_real tmp593;
  tmp592 = (data->simulationInfo->realParameter[313] /* r_init[149] PARAM */);
  tmp593 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp592 * tmp592 * tmp592),"r_init[149] ^ 3.0",equationIndexes);
  if(!(tmp593 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[149] ^ 3.0) was %g should be >= 0", tmp593);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[148]] /* vx[149] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[788]] /* y[149] STATE(1,vy[149]) */))) * (sqrt(tmp593));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4043(DATA *data, threadData_t *threadData);


/*
equation index: 2374
type: SIMPLE_ASSIGN
x[149] = r_init[149] * cos(theta[149] + armOffset[149])
*/
void WhirlpoolDiskStars_eqFunction_2374(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2374};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[628]] /* x[149] STATE(1,vx[149]) */) = ((data->simulationInfo->realParameter[313] /* r_init[149] PARAM */)) * (cos((data->simulationInfo->realParameter[473] /* theta[149] PARAM */) + (data->simulationInfo->realParameter[151] /* armOffset[149] PARAM */)));
  TRACE_POP
}

/*
equation index: 2375
type: SIMPLE_ASSIGN
vy[149] = x[149] * sqrt(G * Md / r_init[149] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2375};
  modelica_real tmp594;
  modelica_real tmp595;
  tmp594 = (data->simulationInfo->realParameter[313] /* r_init[149] PARAM */);
  tmp595 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp594 * tmp594 * tmp594),"r_init[149] ^ 3.0",equationIndexes);
  if(!(tmp595 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[149] ^ 3.0) was %g should be >= 0", tmp595);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[308]] /* vy[149] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[628]] /* x[149] STATE(1,vx[149]) */)) * (sqrt(tmp595));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4042(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4045(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4047(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4050(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4049(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4048(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4046(DATA *data, threadData_t *threadData);


/*
equation index: 2383
type: SIMPLE_ASSIGN
vz[149] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2383};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[468]] /* vz[149] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4041(DATA *data, threadData_t *threadData);


/*
equation index: 2385
type: SIMPLE_ASSIGN
z[150] = 3.5
*/
void WhirlpoolDiskStars_eqFunction_2385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2385};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[949]] /* z[150] STATE(1,vz[150]) */) = 3.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4054(DATA *data, threadData_t *threadData);


/*
equation index: 2387
type: SIMPLE_ASSIGN
y[150] = r_init[150] * sin(theta[150] + armOffset[150])
*/
void WhirlpoolDiskStars_eqFunction_2387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[789]] /* y[150] STATE(1,vy[150]) */) = ((data->simulationInfo->realParameter[314] /* r_init[150] PARAM */)) * (sin((data->simulationInfo->realParameter[474] /* theta[150] PARAM */) + (data->simulationInfo->realParameter[152] /* armOffset[150] PARAM */)));
  TRACE_POP
}

/*
equation index: 2388
type: SIMPLE_ASSIGN
vx[150] = (-y[150]) * sqrt(G * Md / r_init[150] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2388};
  modelica_real tmp596;
  modelica_real tmp597;
  tmp596 = (data->simulationInfo->realParameter[314] /* r_init[150] PARAM */);
  tmp597 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp596 * tmp596 * tmp596),"r_init[150] ^ 3.0",equationIndexes);
  if(!(tmp597 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[150] ^ 3.0) was %g should be >= 0", tmp597);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[149]] /* vx[150] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[789]] /* y[150] STATE(1,vy[150]) */))) * (sqrt(tmp597));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4053(DATA *data, threadData_t *threadData);


/*
equation index: 2390
type: SIMPLE_ASSIGN
x[150] = r_init[150] * cos(theta[150] + armOffset[150])
*/
void WhirlpoolDiskStars_eqFunction_2390(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2390};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[629]] /* x[150] STATE(1,vx[150]) */) = ((data->simulationInfo->realParameter[314] /* r_init[150] PARAM */)) * (cos((data->simulationInfo->realParameter[474] /* theta[150] PARAM */) + (data->simulationInfo->realParameter[152] /* armOffset[150] PARAM */)));
  TRACE_POP
}

/*
equation index: 2391
type: SIMPLE_ASSIGN
vy[150] = x[150] * sqrt(G * Md / r_init[150] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2391};
  modelica_real tmp598;
  modelica_real tmp599;
  tmp598 = (data->simulationInfo->realParameter[314] /* r_init[150] PARAM */);
  tmp599 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp598 * tmp598 * tmp598),"r_init[150] ^ 3.0",equationIndexes);
  if(!(tmp599 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[150] ^ 3.0) was %g should be >= 0", tmp599);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[309]] /* vy[150] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[629]] /* x[150] STATE(1,vx[150]) */)) * (sqrt(tmp599));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4052(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4055(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4057(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4060(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4059(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4058(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4056(DATA *data, threadData_t *threadData);


/*
equation index: 2399
type: SIMPLE_ASSIGN
vz[150] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2399};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[469]] /* vz[150] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4051(DATA *data, threadData_t *threadData);


/*
equation index: 2401
type: SIMPLE_ASSIGN
z[151] = 3.5500000000000003
*/
void WhirlpoolDiskStars_eqFunction_2401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2401};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[950]] /* z[151] STATE(1,vz[151]) */) = 3.5500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4064(DATA *data, threadData_t *threadData);


/*
equation index: 2403
type: SIMPLE_ASSIGN
y[151] = r_init[151] * sin(theta[151] + armOffset[151])
*/
void WhirlpoolDiskStars_eqFunction_2403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[790]] /* y[151] STATE(1,vy[151]) */) = ((data->simulationInfo->realParameter[315] /* r_init[151] PARAM */)) * (sin((data->simulationInfo->realParameter[475] /* theta[151] PARAM */) + (data->simulationInfo->realParameter[153] /* armOffset[151] PARAM */)));
  TRACE_POP
}

/*
equation index: 2404
type: SIMPLE_ASSIGN
vx[151] = (-y[151]) * sqrt(G * Md / r_init[151] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2404};
  modelica_real tmp600;
  modelica_real tmp601;
  tmp600 = (data->simulationInfo->realParameter[315] /* r_init[151] PARAM */);
  tmp601 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp600 * tmp600 * tmp600),"r_init[151] ^ 3.0",equationIndexes);
  if(!(tmp601 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[151] ^ 3.0) was %g should be >= 0", tmp601);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[150]] /* vx[151] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[790]] /* y[151] STATE(1,vy[151]) */))) * (sqrt(tmp601));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4063(DATA *data, threadData_t *threadData);


/*
equation index: 2406
type: SIMPLE_ASSIGN
x[151] = r_init[151] * cos(theta[151] + armOffset[151])
*/
void WhirlpoolDiskStars_eqFunction_2406(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2406};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[630]] /* x[151] STATE(1,vx[151]) */) = ((data->simulationInfo->realParameter[315] /* r_init[151] PARAM */)) * (cos((data->simulationInfo->realParameter[475] /* theta[151] PARAM */) + (data->simulationInfo->realParameter[153] /* armOffset[151] PARAM */)));
  TRACE_POP
}

/*
equation index: 2407
type: SIMPLE_ASSIGN
vy[151] = x[151] * sqrt(G * Md / r_init[151] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2407};
  modelica_real tmp602;
  modelica_real tmp603;
  tmp602 = (data->simulationInfo->realParameter[315] /* r_init[151] PARAM */);
  tmp603 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp602 * tmp602 * tmp602),"r_init[151] ^ 3.0",equationIndexes);
  if(!(tmp603 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[151] ^ 3.0) was %g should be >= 0", tmp603);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[310]] /* vy[151] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[630]] /* x[151] STATE(1,vx[151]) */)) * (sqrt(tmp603));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4062(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4065(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4067(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4070(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4069(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4068(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4066(DATA *data, threadData_t *threadData);


/*
equation index: 2415
type: SIMPLE_ASSIGN
vz[151] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2415};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[470]] /* vz[151] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4061(DATA *data, threadData_t *threadData);


/*
equation index: 2417
type: SIMPLE_ASSIGN
z[152] = 3.6
*/
void WhirlpoolDiskStars_eqFunction_2417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2417};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[951]] /* z[152] STATE(1,vz[152]) */) = 3.6;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4074(DATA *data, threadData_t *threadData);


/*
equation index: 2419
type: SIMPLE_ASSIGN
y[152] = r_init[152] * sin(theta[152] + armOffset[152])
*/
void WhirlpoolDiskStars_eqFunction_2419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[791]] /* y[152] STATE(1,vy[152]) */) = ((data->simulationInfo->realParameter[316] /* r_init[152] PARAM */)) * (sin((data->simulationInfo->realParameter[476] /* theta[152] PARAM */) + (data->simulationInfo->realParameter[154] /* armOffset[152] PARAM */)));
  TRACE_POP
}

/*
equation index: 2420
type: SIMPLE_ASSIGN
vx[152] = (-y[152]) * sqrt(G * Md / r_init[152] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2420};
  modelica_real tmp604;
  modelica_real tmp605;
  tmp604 = (data->simulationInfo->realParameter[316] /* r_init[152] PARAM */);
  tmp605 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp604 * tmp604 * tmp604),"r_init[152] ^ 3.0",equationIndexes);
  if(!(tmp605 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[152] ^ 3.0) was %g should be >= 0", tmp605);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[151]] /* vx[152] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[791]] /* y[152] STATE(1,vy[152]) */))) * (sqrt(tmp605));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4073(DATA *data, threadData_t *threadData);


/*
equation index: 2422
type: SIMPLE_ASSIGN
x[152] = r_init[152] * cos(theta[152] + armOffset[152])
*/
void WhirlpoolDiskStars_eqFunction_2422(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2422};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[631]] /* x[152] STATE(1,vx[152]) */) = ((data->simulationInfo->realParameter[316] /* r_init[152] PARAM */)) * (cos((data->simulationInfo->realParameter[476] /* theta[152] PARAM */) + (data->simulationInfo->realParameter[154] /* armOffset[152] PARAM */)));
  TRACE_POP
}

/*
equation index: 2423
type: SIMPLE_ASSIGN
vy[152] = x[152] * sqrt(G * Md / r_init[152] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2423};
  modelica_real tmp606;
  modelica_real tmp607;
  tmp606 = (data->simulationInfo->realParameter[316] /* r_init[152] PARAM */);
  tmp607 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp606 * tmp606 * tmp606),"r_init[152] ^ 3.0",equationIndexes);
  if(!(tmp607 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[152] ^ 3.0) was %g should be >= 0", tmp607);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[311]] /* vy[152] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[631]] /* x[152] STATE(1,vx[152]) */)) * (sqrt(tmp607));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4072(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4075(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4077(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4080(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4079(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4078(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4076(DATA *data, threadData_t *threadData);


/*
equation index: 2431
type: SIMPLE_ASSIGN
vz[152] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2431};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[471]] /* vz[152] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4071(DATA *data, threadData_t *threadData);


/*
equation index: 2433
type: SIMPLE_ASSIGN
z[153] = 3.6500000000000004
*/
void WhirlpoolDiskStars_eqFunction_2433(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2433};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[952]] /* z[153] STATE(1,vz[153]) */) = 3.6500000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4084(DATA *data, threadData_t *threadData);


/*
equation index: 2435
type: SIMPLE_ASSIGN
y[153] = r_init[153] * sin(theta[153] + armOffset[153])
*/
void WhirlpoolDiskStars_eqFunction_2435(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[792]] /* y[153] STATE(1,vy[153]) */) = ((data->simulationInfo->realParameter[317] /* r_init[153] PARAM */)) * (sin((data->simulationInfo->realParameter[477] /* theta[153] PARAM */) + (data->simulationInfo->realParameter[155] /* armOffset[153] PARAM */)));
  TRACE_POP
}

/*
equation index: 2436
type: SIMPLE_ASSIGN
vx[153] = (-y[153]) * sqrt(G * Md / r_init[153] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2436(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2436};
  modelica_real tmp608;
  modelica_real tmp609;
  tmp608 = (data->simulationInfo->realParameter[317] /* r_init[153] PARAM */);
  tmp609 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp608 * tmp608 * tmp608),"r_init[153] ^ 3.0",equationIndexes);
  if(!(tmp609 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[153] ^ 3.0) was %g should be >= 0", tmp609);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[152]] /* vx[153] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[792]] /* y[153] STATE(1,vy[153]) */))) * (sqrt(tmp609));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4083(DATA *data, threadData_t *threadData);


/*
equation index: 2438
type: SIMPLE_ASSIGN
x[153] = r_init[153] * cos(theta[153] + armOffset[153])
*/
void WhirlpoolDiskStars_eqFunction_2438(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2438};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[632]] /* x[153] STATE(1,vx[153]) */) = ((data->simulationInfo->realParameter[317] /* r_init[153] PARAM */)) * (cos((data->simulationInfo->realParameter[477] /* theta[153] PARAM */) + (data->simulationInfo->realParameter[155] /* armOffset[153] PARAM */)));
  TRACE_POP
}

/*
equation index: 2439
type: SIMPLE_ASSIGN
vy[153] = x[153] * sqrt(G * Md / r_init[153] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2439};
  modelica_real tmp610;
  modelica_real tmp611;
  tmp610 = (data->simulationInfo->realParameter[317] /* r_init[153] PARAM */);
  tmp611 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp610 * tmp610 * tmp610),"r_init[153] ^ 3.0",equationIndexes);
  if(!(tmp611 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[153] ^ 3.0) was %g should be >= 0", tmp611);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[312]] /* vy[153] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[632]] /* x[153] STATE(1,vx[153]) */)) * (sqrt(tmp611));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4082(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4085(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4087(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4090(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4089(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4088(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4086(DATA *data, threadData_t *threadData);


/*
equation index: 2447
type: SIMPLE_ASSIGN
vz[153] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2447};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[472]] /* vz[153] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4081(DATA *data, threadData_t *threadData);


/*
equation index: 2449
type: SIMPLE_ASSIGN
z[154] = 3.7
*/
void WhirlpoolDiskStars_eqFunction_2449(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2449};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[953]] /* z[154] STATE(1,vz[154]) */) = 3.7;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4094(DATA *data, threadData_t *threadData);


/*
equation index: 2451
type: SIMPLE_ASSIGN
y[154] = r_init[154] * sin(theta[154] + armOffset[154])
*/
void WhirlpoolDiskStars_eqFunction_2451(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[793]] /* y[154] STATE(1,vy[154]) */) = ((data->simulationInfo->realParameter[318] /* r_init[154] PARAM */)) * (sin((data->simulationInfo->realParameter[478] /* theta[154] PARAM */) + (data->simulationInfo->realParameter[156] /* armOffset[154] PARAM */)));
  TRACE_POP
}

/*
equation index: 2452
type: SIMPLE_ASSIGN
vx[154] = (-y[154]) * sqrt(G * Md / r_init[154] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2452(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2452};
  modelica_real tmp612;
  modelica_real tmp613;
  tmp612 = (data->simulationInfo->realParameter[318] /* r_init[154] PARAM */);
  tmp613 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp612 * tmp612 * tmp612),"r_init[154] ^ 3.0",equationIndexes);
  if(!(tmp613 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[154] ^ 3.0) was %g should be >= 0", tmp613);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[153]] /* vx[154] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[793]] /* y[154] STATE(1,vy[154]) */))) * (sqrt(tmp613));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4093(DATA *data, threadData_t *threadData);


/*
equation index: 2454
type: SIMPLE_ASSIGN
x[154] = r_init[154] * cos(theta[154] + armOffset[154])
*/
void WhirlpoolDiskStars_eqFunction_2454(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2454};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[633]] /* x[154] STATE(1,vx[154]) */) = ((data->simulationInfo->realParameter[318] /* r_init[154] PARAM */)) * (cos((data->simulationInfo->realParameter[478] /* theta[154] PARAM */) + (data->simulationInfo->realParameter[156] /* armOffset[154] PARAM */)));
  TRACE_POP
}

/*
equation index: 2455
type: SIMPLE_ASSIGN
vy[154] = x[154] * sqrt(G * Md / r_init[154] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2455};
  modelica_real tmp614;
  modelica_real tmp615;
  tmp614 = (data->simulationInfo->realParameter[318] /* r_init[154] PARAM */);
  tmp615 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp614 * tmp614 * tmp614),"r_init[154] ^ 3.0",equationIndexes);
  if(!(tmp615 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[154] ^ 3.0) was %g should be >= 0", tmp615);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[313]] /* vy[154] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[633]] /* x[154] STATE(1,vx[154]) */)) * (sqrt(tmp615));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4092(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4095(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4097(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4100(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4099(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4098(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4096(DATA *data, threadData_t *threadData);


/*
equation index: 2463
type: SIMPLE_ASSIGN
vz[154] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2463};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[473]] /* vz[154] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4091(DATA *data, threadData_t *threadData);


/*
equation index: 2465
type: SIMPLE_ASSIGN
z[155] = 3.75
*/
void WhirlpoolDiskStars_eqFunction_2465(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2465};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[954]] /* z[155] STATE(1,vz[155]) */) = 3.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4104(DATA *data, threadData_t *threadData);


/*
equation index: 2467
type: SIMPLE_ASSIGN
y[155] = r_init[155] * sin(theta[155] + armOffset[155])
*/
void WhirlpoolDiskStars_eqFunction_2467(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[794]] /* y[155] STATE(1,vy[155]) */) = ((data->simulationInfo->realParameter[319] /* r_init[155] PARAM */)) * (sin((data->simulationInfo->realParameter[479] /* theta[155] PARAM */) + (data->simulationInfo->realParameter[157] /* armOffset[155] PARAM */)));
  TRACE_POP
}

/*
equation index: 2468
type: SIMPLE_ASSIGN
vx[155] = (-y[155]) * sqrt(G * Md / r_init[155] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2468(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2468};
  modelica_real tmp616;
  modelica_real tmp617;
  tmp616 = (data->simulationInfo->realParameter[319] /* r_init[155] PARAM */);
  tmp617 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp616 * tmp616 * tmp616),"r_init[155] ^ 3.0",equationIndexes);
  if(!(tmp617 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[155] ^ 3.0) was %g should be >= 0", tmp617);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[154]] /* vx[155] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[794]] /* y[155] STATE(1,vy[155]) */))) * (sqrt(tmp617));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4103(DATA *data, threadData_t *threadData);


/*
equation index: 2470
type: SIMPLE_ASSIGN
x[155] = r_init[155] * cos(theta[155] + armOffset[155])
*/
void WhirlpoolDiskStars_eqFunction_2470(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2470};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[634]] /* x[155] STATE(1,vx[155]) */) = ((data->simulationInfo->realParameter[319] /* r_init[155] PARAM */)) * (cos((data->simulationInfo->realParameter[479] /* theta[155] PARAM */) + (data->simulationInfo->realParameter[157] /* armOffset[155] PARAM */)));
  TRACE_POP
}

/*
equation index: 2471
type: SIMPLE_ASSIGN
vy[155] = x[155] * sqrt(G * Md / r_init[155] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2471};
  modelica_real tmp618;
  modelica_real tmp619;
  tmp618 = (data->simulationInfo->realParameter[319] /* r_init[155] PARAM */);
  tmp619 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp618 * tmp618 * tmp618),"r_init[155] ^ 3.0",equationIndexes);
  if(!(tmp619 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[155] ^ 3.0) was %g should be >= 0", tmp619);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[314]] /* vy[155] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[634]] /* x[155] STATE(1,vx[155]) */)) * (sqrt(tmp619));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4102(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4105(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4107(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4110(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4109(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4108(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4106(DATA *data, threadData_t *threadData);


/*
equation index: 2479
type: SIMPLE_ASSIGN
vz[155] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2479};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[474]] /* vz[155] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4101(DATA *data, threadData_t *threadData);


/*
equation index: 2481
type: SIMPLE_ASSIGN
z[156] = 3.8000000000000003
*/
void WhirlpoolDiskStars_eqFunction_2481(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2481};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[955]] /* z[156] STATE(1,vz[156]) */) = 3.8000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4114(DATA *data, threadData_t *threadData);


/*
equation index: 2483
type: SIMPLE_ASSIGN
y[156] = r_init[156] * sin(theta[156] + armOffset[156])
*/
void WhirlpoolDiskStars_eqFunction_2483(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[795]] /* y[156] STATE(1,vy[156]) */) = ((data->simulationInfo->realParameter[320] /* r_init[156] PARAM */)) * (sin((data->simulationInfo->realParameter[480] /* theta[156] PARAM */) + (data->simulationInfo->realParameter[158] /* armOffset[156] PARAM */)));
  TRACE_POP
}

/*
equation index: 2484
type: SIMPLE_ASSIGN
vx[156] = (-y[156]) * sqrt(G * Md / r_init[156] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2484(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2484};
  modelica_real tmp620;
  modelica_real tmp621;
  tmp620 = (data->simulationInfo->realParameter[320] /* r_init[156] PARAM */);
  tmp621 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp620 * tmp620 * tmp620),"r_init[156] ^ 3.0",equationIndexes);
  if(!(tmp621 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[156] ^ 3.0) was %g should be >= 0", tmp621);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[155]] /* vx[156] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[795]] /* y[156] STATE(1,vy[156]) */))) * (sqrt(tmp621));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4113(DATA *data, threadData_t *threadData);


/*
equation index: 2486
type: SIMPLE_ASSIGN
x[156] = r_init[156] * cos(theta[156] + armOffset[156])
*/
void WhirlpoolDiskStars_eqFunction_2486(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2486};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[635]] /* x[156] STATE(1,vx[156]) */) = ((data->simulationInfo->realParameter[320] /* r_init[156] PARAM */)) * (cos((data->simulationInfo->realParameter[480] /* theta[156] PARAM */) + (data->simulationInfo->realParameter[158] /* armOffset[156] PARAM */)));
  TRACE_POP
}

/*
equation index: 2487
type: SIMPLE_ASSIGN
vy[156] = x[156] * sqrt(G * Md / r_init[156] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2487};
  modelica_real tmp622;
  modelica_real tmp623;
  tmp622 = (data->simulationInfo->realParameter[320] /* r_init[156] PARAM */);
  tmp623 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp622 * tmp622 * tmp622),"r_init[156] ^ 3.0",equationIndexes);
  if(!(tmp623 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[156] ^ 3.0) was %g should be >= 0", tmp623);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[315]] /* vy[156] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[635]] /* x[156] STATE(1,vx[156]) */)) * (sqrt(tmp623));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4112(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4115(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4117(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4120(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4119(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4118(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4116(DATA *data, threadData_t *threadData);


/*
equation index: 2495
type: SIMPLE_ASSIGN
vz[156] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2495};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[475]] /* vz[156] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4111(DATA *data, threadData_t *threadData);


/*
equation index: 2497
type: SIMPLE_ASSIGN
z[157] = 3.85
*/
void WhirlpoolDiskStars_eqFunction_2497(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2497};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[956]] /* z[157] STATE(1,vz[157]) */) = 3.85;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4124(DATA *data, threadData_t *threadData);


/*
equation index: 2499
type: SIMPLE_ASSIGN
y[157] = r_init[157] * sin(theta[157] + armOffset[157])
*/
void WhirlpoolDiskStars_eqFunction_2499(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2499};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[796]] /* y[157] STATE(1,vy[157]) */) = ((data->simulationInfo->realParameter[321] /* r_init[157] PARAM */)) * (sin((data->simulationInfo->realParameter[481] /* theta[157] PARAM */) + (data->simulationInfo->realParameter[159] /* armOffset[157] PARAM */)));
  TRACE_POP
}

/*
equation index: 2500
type: SIMPLE_ASSIGN
vx[157] = (-y[157]) * sqrt(G * Md / r_init[157] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2500(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2500};
  modelica_real tmp624;
  modelica_real tmp625;
  tmp624 = (data->simulationInfo->realParameter[321] /* r_init[157] PARAM */);
  tmp625 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp624 * tmp624 * tmp624),"r_init[157] ^ 3.0",equationIndexes);
  if(!(tmp625 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[157] ^ 3.0) was %g should be >= 0", tmp625);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[156]] /* vx[157] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[796]] /* y[157] STATE(1,vy[157]) */))) * (sqrt(tmp625));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4123(DATA *data, threadData_t *threadData);


/*
equation index: 2502
type: SIMPLE_ASSIGN
x[157] = r_init[157] * cos(theta[157] + armOffset[157])
*/
void WhirlpoolDiskStars_eqFunction_2502(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2502};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[636]] /* x[157] STATE(1,vx[157]) */) = ((data->simulationInfo->realParameter[321] /* r_init[157] PARAM */)) * (cos((data->simulationInfo->realParameter[481] /* theta[157] PARAM */) + (data->simulationInfo->realParameter[159] /* armOffset[157] PARAM */)));
  TRACE_POP
}

/*
equation index: 2503
type: SIMPLE_ASSIGN
vy[157] = x[157] * sqrt(G * Md / r_init[157] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2503};
  modelica_real tmp626;
  modelica_real tmp627;
  tmp626 = (data->simulationInfo->realParameter[321] /* r_init[157] PARAM */);
  tmp627 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp626 * tmp626 * tmp626),"r_init[157] ^ 3.0",equationIndexes);
  if(!(tmp627 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[157] ^ 3.0) was %g should be >= 0", tmp627);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[316]] /* vy[157] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[636]] /* x[157] STATE(1,vx[157]) */)) * (sqrt(tmp627));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4122(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4125(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4127(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4130(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4129(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4128(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4126(DATA *data, threadData_t *threadData);


/*
equation index: 2511
type: SIMPLE_ASSIGN
vz[157] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2511};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[476]] /* vz[157] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4121(DATA *data, threadData_t *threadData);


/*
equation index: 2513
type: SIMPLE_ASSIGN
z[158] = 3.9000000000000004
*/
void WhirlpoolDiskStars_eqFunction_2513(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2513};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[957]] /* z[158] STATE(1,vz[158]) */) = 3.9000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4134(DATA *data, threadData_t *threadData);


/*
equation index: 2515
type: SIMPLE_ASSIGN
y[158] = r_init[158] * sin(theta[158] + armOffset[158])
*/
void WhirlpoolDiskStars_eqFunction_2515(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2515};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[797]] /* y[158] STATE(1,vy[158]) */) = ((data->simulationInfo->realParameter[322] /* r_init[158] PARAM */)) * (sin((data->simulationInfo->realParameter[482] /* theta[158] PARAM */) + (data->simulationInfo->realParameter[160] /* armOffset[158] PARAM */)));
  TRACE_POP
}

/*
equation index: 2516
type: SIMPLE_ASSIGN
vx[158] = (-y[158]) * sqrt(G * Md / r_init[158] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2516(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2516};
  modelica_real tmp628;
  modelica_real tmp629;
  tmp628 = (data->simulationInfo->realParameter[322] /* r_init[158] PARAM */);
  tmp629 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp628 * tmp628 * tmp628),"r_init[158] ^ 3.0",equationIndexes);
  if(!(tmp629 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[158] ^ 3.0) was %g should be >= 0", tmp629);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[157]] /* vx[158] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[797]] /* y[158] STATE(1,vy[158]) */))) * (sqrt(tmp629));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4133(DATA *data, threadData_t *threadData);


/*
equation index: 2518
type: SIMPLE_ASSIGN
x[158] = r_init[158] * cos(theta[158] + armOffset[158])
*/
void WhirlpoolDiskStars_eqFunction_2518(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2518};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[637]] /* x[158] STATE(1,vx[158]) */) = ((data->simulationInfo->realParameter[322] /* r_init[158] PARAM */)) * (cos((data->simulationInfo->realParameter[482] /* theta[158] PARAM */) + (data->simulationInfo->realParameter[160] /* armOffset[158] PARAM */)));
  TRACE_POP
}

/*
equation index: 2519
type: SIMPLE_ASSIGN
vy[158] = x[158] * sqrt(G * Md / r_init[158] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2519};
  modelica_real tmp630;
  modelica_real tmp631;
  tmp630 = (data->simulationInfo->realParameter[322] /* r_init[158] PARAM */);
  tmp631 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp630 * tmp630 * tmp630),"r_init[158] ^ 3.0",equationIndexes);
  if(!(tmp631 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[158] ^ 3.0) was %g should be >= 0", tmp631);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[317]] /* vy[158] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[637]] /* x[158] STATE(1,vx[158]) */)) * (sqrt(tmp631));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4132(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4135(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4137(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4140(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4139(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4138(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4136(DATA *data, threadData_t *threadData);


/*
equation index: 2527
type: SIMPLE_ASSIGN
vz[158] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[477]] /* vz[158] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4131(DATA *data, threadData_t *threadData);


/*
equation index: 2529
type: SIMPLE_ASSIGN
z[159] = 3.95
*/
void WhirlpoolDiskStars_eqFunction_2529(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2529};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[958]] /* z[159] STATE(1,vz[159]) */) = 3.95;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4144(DATA *data, threadData_t *threadData);


/*
equation index: 2531
type: SIMPLE_ASSIGN
y[159] = r_init[159] * sin(theta[159] + armOffset[159])
*/
void WhirlpoolDiskStars_eqFunction_2531(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2531};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[798]] /* y[159] STATE(1,vy[159]) */) = ((data->simulationInfo->realParameter[323] /* r_init[159] PARAM */)) * (sin((data->simulationInfo->realParameter[483] /* theta[159] PARAM */) + (data->simulationInfo->realParameter[161] /* armOffset[159] PARAM */)));
  TRACE_POP
}

/*
equation index: 2532
type: SIMPLE_ASSIGN
vx[159] = (-y[159]) * sqrt(G * Md / r_init[159] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2532(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2532};
  modelica_real tmp632;
  modelica_real tmp633;
  tmp632 = (data->simulationInfo->realParameter[323] /* r_init[159] PARAM */);
  tmp633 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp632 * tmp632 * tmp632),"r_init[159] ^ 3.0",equationIndexes);
  if(!(tmp633 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[159] ^ 3.0) was %g should be >= 0", tmp633);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[158]] /* vx[159] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[798]] /* y[159] STATE(1,vy[159]) */))) * (sqrt(tmp633));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4143(DATA *data, threadData_t *threadData);


/*
equation index: 2534
type: SIMPLE_ASSIGN
x[159] = r_init[159] * cos(theta[159] + armOffset[159])
*/
void WhirlpoolDiskStars_eqFunction_2534(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2534};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[638]] /* x[159] STATE(1,vx[159]) */) = ((data->simulationInfo->realParameter[323] /* r_init[159] PARAM */)) * (cos((data->simulationInfo->realParameter[483] /* theta[159] PARAM */) + (data->simulationInfo->realParameter[161] /* armOffset[159] PARAM */)));
  TRACE_POP
}

/*
equation index: 2535
type: SIMPLE_ASSIGN
vy[159] = x[159] * sqrt(G * Md / r_init[159] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2535};
  modelica_real tmp634;
  modelica_real tmp635;
  tmp634 = (data->simulationInfo->realParameter[323] /* r_init[159] PARAM */);
  tmp635 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp634 * tmp634 * tmp634),"r_init[159] ^ 3.0",equationIndexes);
  if(!(tmp635 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[159] ^ 3.0) was %g should be >= 0", tmp635);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[318]] /* vy[159] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[638]] /* x[159] STATE(1,vx[159]) */)) * (sqrt(tmp635));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4142(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4145(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4147(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4150(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4149(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4148(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4146(DATA *data, threadData_t *threadData);


/*
equation index: 2543
type: SIMPLE_ASSIGN
vz[159] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2543};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[478]] /* vz[159] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4141(DATA *data, threadData_t *threadData);


/*
equation index: 2545
type: SIMPLE_ASSIGN
z[160] = 4.0
*/
void WhirlpoolDiskStars_eqFunction_2545(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2545};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[959]] /* z[160] STATE(1,vz[160]) */) = 4.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4154(DATA *data, threadData_t *threadData);


/*
equation index: 2547
type: SIMPLE_ASSIGN
y[160] = r_init[160] * sin(theta[160] + armOffset[160])
*/
void WhirlpoolDiskStars_eqFunction_2547(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2547};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[799]] /* y[160] STATE(1,vy[160]) */) = ((data->simulationInfo->realParameter[324] /* r_init[160] PARAM */)) * (sin((data->simulationInfo->realParameter[484] /* theta[160] PARAM */) + (data->simulationInfo->realParameter[162] /* armOffset[160] PARAM */)));
  TRACE_POP
}

/*
equation index: 2548
type: SIMPLE_ASSIGN
vx[160] = (-y[160]) * sqrt(G * Md / r_init[160] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2548(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2548};
  modelica_real tmp636;
  modelica_real tmp637;
  tmp636 = (data->simulationInfo->realParameter[324] /* r_init[160] PARAM */);
  tmp637 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp636 * tmp636 * tmp636),"r_init[160] ^ 3.0",equationIndexes);
  if(!(tmp637 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[160] ^ 3.0) was %g should be >= 0", tmp637);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[159]] /* vx[160] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[799]] /* y[160] STATE(1,vy[160]) */))) * (sqrt(tmp637));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4153(DATA *data, threadData_t *threadData);


/*
equation index: 2550
type: SIMPLE_ASSIGN
x[160] = r_init[160] * cos(theta[160] + armOffset[160])
*/
void WhirlpoolDiskStars_eqFunction_2550(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2550};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[639]] /* x[160] STATE(1,vx[160]) */) = ((data->simulationInfo->realParameter[324] /* r_init[160] PARAM */)) * (cos((data->simulationInfo->realParameter[484] /* theta[160] PARAM */) + (data->simulationInfo->realParameter[162] /* armOffset[160] PARAM */)));
  TRACE_POP
}

/*
equation index: 2551
type: SIMPLE_ASSIGN
vy[160] = x[160] * sqrt(G * Md / r_init[160] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2551};
  modelica_real tmp638;
  modelica_real tmp639;
  tmp638 = (data->simulationInfo->realParameter[324] /* r_init[160] PARAM */);
  tmp639 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp638 * tmp638 * tmp638),"r_init[160] ^ 3.0",equationIndexes);
  if(!(tmp639 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[160] ^ 3.0) was %g should be >= 0", tmp639);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[319]] /* vy[160] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[639]] /* x[160] STATE(1,vx[160]) */)) * (sqrt(tmp639));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4152(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4155(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4157(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4160(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4159(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4158(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_4156(DATA *data, threadData_t *threadData);


/*
equation index: 2559
type: SIMPLE_ASSIGN
vz[160] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2559};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[479]] /* vz[160] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_4151(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void WhirlpoolDiskStars_functionInitialEquations_5(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_3892(data, threadData);
  WhirlpoolDiskStars_eqFunction_3895(data, threadData);
  WhirlpoolDiskStars_eqFunction_3897(data, threadData);
  WhirlpoolDiskStars_eqFunction_3900(data, threadData);
  WhirlpoolDiskStars_eqFunction_3899(data, threadData);
  WhirlpoolDiskStars_eqFunction_3898(data, threadData);
  WhirlpoolDiskStars_eqFunction_3896(data, threadData);
  WhirlpoolDiskStars_eqFunction_2143(data, threadData);
  WhirlpoolDiskStars_eqFunction_3891(data, threadData);
  WhirlpoolDiskStars_eqFunction_2145(data, threadData);
  WhirlpoolDiskStars_eqFunction_3904(data, threadData);
  WhirlpoolDiskStars_eqFunction_2147(data, threadData);
  WhirlpoolDiskStars_eqFunction_2148(data, threadData);
  WhirlpoolDiskStars_eqFunction_3903(data, threadData);
  WhirlpoolDiskStars_eqFunction_2150(data, threadData);
  WhirlpoolDiskStars_eqFunction_2151(data, threadData);
  WhirlpoolDiskStars_eqFunction_3902(data, threadData);
  WhirlpoolDiskStars_eqFunction_3905(data, threadData);
  WhirlpoolDiskStars_eqFunction_3907(data, threadData);
  WhirlpoolDiskStars_eqFunction_3910(data, threadData);
  WhirlpoolDiskStars_eqFunction_3909(data, threadData);
  WhirlpoolDiskStars_eqFunction_3908(data, threadData);
  WhirlpoolDiskStars_eqFunction_3906(data, threadData);
  WhirlpoolDiskStars_eqFunction_2159(data, threadData);
  WhirlpoolDiskStars_eqFunction_3901(data, threadData);
  WhirlpoolDiskStars_eqFunction_2161(data, threadData);
  WhirlpoolDiskStars_eqFunction_3914(data, threadData);
  WhirlpoolDiskStars_eqFunction_2163(data, threadData);
  WhirlpoolDiskStars_eqFunction_2164(data, threadData);
  WhirlpoolDiskStars_eqFunction_3913(data, threadData);
  WhirlpoolDiskStars_eqFunction_2166(data, threadData);
  WhirlpoolDiskStars_eqFunction_2167(data, threadData);
  WhirlpoolDiskStars_eqFunction_3912(data, threadData);
  WhirlpoolDiskStars_eqFunction_3915(data, threadData);
  WhirlpoolDiskStars_eqFunction_3917(data, threadData);
  WhirlpoolDiskStars_eqFunction_3920(data, threadData);
  WhirlpoolDiskStars_eqFunction_3919(data, threadData);
  WhirlpoolDiskStars_eqFunction_3918(data, threadData);
  WhirlpoolDiskStars_eqFunction_3916(data, threadData);
  WhirlpoolDiskStars_eqFunction_2175(data, threadData);
  WhirlpoolDiskStars_eqFunction_3911(data, threadData);
  WhirlpoolDiskStars_eqFunction_2177(data, threadData);
  WhirlpoolDiskStars_eqFunction_3924(data, threadData);
  WhirlpoolDiskStars_eqFunction_2179(data, threadData);
  WhirlpoolDiskStars_eqFunction_2180(data, threadData);
  WhirlpoolDiskStars_eqFunction_3923(data, threadData);
  WhirlpoolDiskStars_eqFunction_2182(data, threadData);
  WhirlpoolDiskStars_eqFunction_2183(data, threadData);
  WhirlpoolDiskStars_eqFunction_3922(data, threadData);
  WhirlpoolDiskStars_eqFunction_3925(data, threadData);
  WhirlpoolDiskStars_eqFunction_3927(data, threadData);
  WhirlpoolDiskStars_eqFunction_3930(data, threadData);
  WhirlpoolDiskStars_eqFunction_3929(data, threadData);
  WhirlpoolDiskStars_eqFunction_3928(data, threadData);
  WhirlpoolDiskStars_eqFunction_3926(data, threadData);
  WhirlpoolDiskStars_eqFunction_2191(data, threadData);
  WhirlpoolDiskStars_eqFunction_3921(data, threadData);
  WhirlpoolDiskStars_eqFunction_2193(data, threadData);
  WhirlpoolDiskStars_eqFunction_3934(data, threadData);
  WhirlpoolDiskStars_eqFunction_2195(data, threadData);
  WhirlpoolDiskStars_eqFunction_2196(data, threadData);
  WhirlpoolDiskStars_eqFunction_3933(data, threadData);
  WhirlpoolDiskStars_eqFunction_2198(data, threadData);
  WhirlpoolDiskStars_eqFunction_2199(data, threadData);
  WhirlpoolDiskStars_eqFunction_3932(data, threadData);
  WhirlpoolDiskStars_eqFunction_3935(data, threadData);
  WhirlpoolDiskStars_eqFunction_3937(data, threadData);
  WhirlpoolDiskStars_eqFunction_3940(data, threadData);
  WhirlpoolDiskStars_eqFunction_3939(data, threadData);
  WhirlpoolDiskStars_eqFunction_3938(data, threadData);
  WhirlpoolDiskStars_eqFunction_3936(data, threadData);
  WhirlpoolDiskStars_eqFunction_2207(data, threadData);
  WhirlpoolDiskStars_eqFunction_3931(data, threadData);
  WhirlpoolDiskStars_eqFunction_2209(data, threadData);
  WhirlpoolDiskStars_eqFunction_3944(data, threadData);
  WhirlpoolDiskStars_eqFunction_2211(data, threadData);
  WhirlpoolDiskStars_eqFunction_2212(data, threadData);
  WhirlpoolDiskStars_eqFunction_3943(data, threadData);
  WhirlpoolDiskStars_eqFunction_2214(data, threadData);
  WhirlpoolDiskStars_eqFunction_2215(data, threadData);
  WhirlpoolDiskStars_eqFunction_3942(data, threadData);
  WhirlpoolDiskStars_eqFunction_3945(data, threadData);
  WhirlpoolDiskStars_eqFunction_3947(data, threadData);
  WhirlpoolDiskStars_eqFunction_3950(data, threadData);
  WhirlpoolDiskStars_eqFunction_3949(data, threadData);
  WhirlpoolDiskStars_eqFunction_3948(data, threadData);
  WhirlpoolDiskStars_eqFunction_3946(data, threadData);
  WhirlpoolDiskStars_eqFunction_2223(data, threadData);
  WhirlpoolDiskStars_eqFunction_3941(data, threadData);
  WhirlpoolDiskStars_eqFunction_2225(data, threadData);
  WhirlpoolDiskStars_eqFunction_3954(data, threadData);
  WhirlpoolDiskStars_eqFunction_2227(data, threadData);
  WhirlpoolDiskStars_eqFunction_2228(data, threadData);
  WhirlpoolDiskStars_eqFunction_3953(data, threadData);
  WhirlpoolDiskStars_eqFunction_2230(data, threadData);
  WhirlpoolDiskStars_eqFunction_2231(data, threadData);
  WhirlpoolDiskStars_eqFunction_3952(data, threadData);
  WhirlpoolDiskStars_eqFunction_3955(data, threadData);
  WhirlpoolDiskStars_eqFunction_3957(data, threadData);
  WhirlpoolDiskStars_eqFunction_3960(data, threadData);
  WhirlpoolDiskStars_eqFunction_3959(data, threadData);
  WhirlpoolDiskStars_eqFunction_3958(data, threadData);
  WhirlpoolDiskStars_eqFunction_3956(data, threadData);
  WhirlpoolDiskStars_eqFunction_2239(data, threadData);
  WhirlpoolDiskStars_eqFunction_3951(data, threadData);
  WhirlpoolDiskStars_eqFunction_2241(data, threadData);
  WhirlpoolDiskStars_eqFunction_3964(data, threadData);
  WhirlpoolDiskStars_eqFunction_2243(data, threadData);
  WhirlpoolDiskStars_eqFunction_2244(data, threadData);
  WhirlpoolDiskStars_eqFunction_3963(data, threadData);
  WhirlpoolDiskStars_eqFunction_2246(data, threadData);
  WhirlpoolDiskStars_eqFunction_2247(data, threadData);
  WhirlpoolDiskStars_eqFunction_3962(data, threadData);
  WhirlpoolDiskStars_eqFunction_3965(data, threadData);
  WhirlpoolDiskStars_eqFunction_3967(data, threadData);
  WhirlpoolDiskStars_eqFunction_3970(data, threadData);
  WhirlpoolDiskStars_eqFunction_3969(data, threadData);
  WhirlpoolDiskStars_eqFunction_3968(data, threadData);
  WhirlpoolDiskStars_eqFunction_3966(data, threadData);
  WhirlpoolDiskStars_eqFunction_2255(data, threadData);
  WhirlpoolDiskStars_eqFunction_3961(data, threadData);
  WhirlpoolDiskStars_eqFunction_2257(data, threadData);
  WhirlpoolDiskStars_eqFunction_3974(data, threadData);
  WhirlpoolDiskStars_eqFunction_2259(data, threadData);
  WhirlpoolDiskStars_eqFunction_2260(data, threadData);
  WhirlpoolDiskStars_eqFunction_3973(data, threadData);
  WhirlpoolDiskStars_eqFunction_2262(data, threadData);
  WhirlpoolDiskStars_eqFunction_2263(data, threadData);
  WhirlpoolDiskStars_eqFunction_3972(data, threadData);
  WhirlpoolDiskStars_eqFunction_3975(data, threadData);
  WhirlpoolDiskStars_eqFunction_3977(data, threadData);
  WhirlpoolDiskStars_eqFunction_3980(data, threadData);
  WhirlpoolDiskStars_eqFunction_3979(data, threadData);
  WhirlpoolDiskStars_eqFunction_3978(data, threadData);
  WhirlpoolDiskStars_eqFunction_3976(data, threadData);
  WhirlpoolDiskStars_eqFunction_2271(data, threadData);
  WhirlpoolDiskStars_eqFunction_3971(data, threadData);
  WhirlpoolDiskStars_eqFunction_2273(data, threadData);
  WhirlpoolDiskStars_eqFunction_3984(data, threadData);
  WhirlpoolDiskStars_eqFunction_2275(data, threadData);
  WhirlpoolDiskStars_eqFunction_2276(data, threadData);
  WhirlpoolDiskStars_eqFunction_3983(data, threadData);
  WhirlpoolDiskStars_eqFunction_2278(data, threadData);
  WhirlpoolDiskStars_eqFunction_2279(data, threadData);
  WhirlpoolDiskStars_eqFunction_3982(data, threadData);
  WhirlpoolDiskStars_eqFunction_3985(data, threadData);
  WhirlpoolDiskStars_eqFunction_3987(data, threadData);
  WhirlpoolDiskStars_eqFunction_3990(data, threadData);
  WhirlpoolDiskStars_eqFunction_3989(data, threadData);
  WhirlpoolDiskStars_eqFunction_3988(data, threadData);
  WhirlpoolDiskStars_eqFunction_3986(data, threadData);
  WhirlpoolDiskStars_eqFunction_2287(data, threadData);
  WhirlpoolDiskStars_eqFunction_3981(data, threadData);
  WhirlpoolDiskStars_eqFunction_2289(data, threadData);
  WhirlpoolDiskStars_eqFunction_3994(data, threadData);
  WhirlpoolDiskStars_eqFunction_2291(data, threadData);
  WhirlpoolDiskStars_eqFunction_2292(data, threadData);
  WhirlpoolDiskStars_eqFunction_3993(data, threadData);
  WhirlpoolDiskStars_eqFunction_2294(data, threadData);
  WhirlpoolDiskStars_eqFunction_2295(data, threadData);
  WhirlpoolDiskStars_eqFunction_3992(data, threadData);
  WhirlpoolDiskStars_eqFunction_3995(data, threadData);
  WhirlpoolDiskStars_eqFunction_3997(data, threadData);
  WhirlpoolDiskStars_eqFunction_4000(data, threadData);
  WhirlpoolDiskStars_eqFunction_3999(data, threadData);
  WhirlpoolDiskStars_eqFunction_3998(data, threadData);
  WhirlpoolDiskStars_eqFunction_3996(data, threadData);
  WhirlpoolDiskStars_eqFunction_2303(data, threadData);
  WhirlpoolDiskStars_eqFunction_3991(data, threadData);
  WhirlpoolDiskStars_eqFunction_2305(data, threadData);
  WhirlpoolDiskStars_eqFunction_4004(data, threadData);
  WhirlpoolDiskStars_eqFunction_2307(data, threadData);
  WhirlpoolDiskStars_eqFunction_2308(data, threadData);
  WhirlpoolDiskStars_eqFunction_4003(data, threadData);
  WhirlpoolDiskStars_eqFunction_2310(data, threadData);
  WhirlpoolDiskStars_eqFunction_2311(data, threadData);
  WhirlpoolDiskStars_eqFunction_4002(data, threadData);
  WhirlpoolDiskStars_eqFunction_4005(data, threadData);
  WhirlpoolDiskStars_eqFunction_4007(data, threadData);
  WhirlpoolDiskStars_eqFunction_4010(data, threadData);
  WhirlpoolDiskStars_eqFunction_4009(data, threadData);
  WhirlpoolDiskStars_eqFunction_4008(data, threadData);
  WhirlpoolDiskStars_eqFunction_4006(data, threadData);
  WhirlpoolDiskStars_eqFunction_2319(data, threadData);
  WhirlpoolDiskStars_eqFunction_4001(data, threadData);
  WhirlpoolDiskStars_eqFunction_2321(data, threadData);
  WhirlpoolDiskStars_eqFunction_4014(data, threadData);
  WhirlpoolDiskStars_eqFunction_2323(data, threadData);
  WhirlpoolDiskStars_eqFunction_2324(data, threadData);
  WhirlpoolDiskStars_eqFunction_4013(data, threadData);
  WhirlpoolDiskStars_eqFunction_2326(data, threadData);
  WhirlpoolDiskStars_eqFunction_2327(data, threadData);
  WhirlpoolDiskStars_eqFunction_4012(data, threadData);
  WhirlpoolDiskStars_eqFunction_4015(data, threadData);
  WhirlpoolDiskStars_eqFunction_4017(data, threadData);
  WhirlpoolDiskStars_eqFunction_4020(data, threadData);
  WhirlpoolDiskStars_eqFunction_4019(data, threadData);
  WhirlpoolDiskStars_eqFunction_4018(data, threadData);
  WhirlpoolDiskStars_eqFunction_4016(data, threadData);
  WhirlpoolDiskStars_eqFunction_2335(data, threadData);
  WhirlpoolDiskStars_eqFunction_4011(data, threadData);
  WhirlpoolDiskStars_eqFunction_2337(data, threadData);
  WhirlpoolDiskStars_eqFunction_4024(data, threadData);
  WhirlpoolDiskStars_eqFunction_2339(data, threadData);
  WhirlpoolDiskStars_eqFunction_2340(data, threadData);
  WhirlpoolDiskStars_eqFunction_4023(data, threadData);
  WhirlpoolDiskStars_eqFunction_2342(data, threadData);
  WhirlpoolDiskStars_eqFunction_2343(data, threadData);
  WhirlpoolDiskStars_eqFunction_4022(data, threadData);
  WhirlpoolDiskStars_eqFunction_4025(data, threadData);
  WhirlpoolDiskStars_eqFunction_4027(data, threadData);
  WhirlpoolDiskStars_eqFunction_4030(data, threadData);
  WhirlpoolDiskStars_eqFunction_4029(data, threadData);
  WhirlpoolDiskStars_eqFunction_4028(data, threadData);
  WhirlpoolDiskStars_eqFunction_4026(data, threadData);
  WhirlpoolDiskStars_eqFunction_2351(data, threadData);
  WhirlpoolDiskStars_eqFunction_4021(data, threadData);
  WhirlpoolDiskStars_eqFunction_2353(data, threadData);
  WhirlpoolDiskStars_eqFunction_4034(data, threadData);
  WhirlpoolDiskStars_eqFunction_2355(data, threadData);
  WhirlpoolDiskStars_eqFunction_2356(data, threadData);
  WhirlpoolDiskStars_eqFunction_4033(data, threadData);
  WhirlpoolDiskStars_eqFunction_2358(data, threadData);
  WhirlpoolDiskStars_eqFunction_2359(data, threadData);
  WhirlpoolDiskStars_eqFunction_4032(data, threadData);
  WhirlpoolDiskStars_eqFunction_4035(data, threadData);
  WhirlpoolDiskStars_eqFunction_4037(data, threadData);
  WhirlpoolDiskStars_eqFunction_4040(data, threadData);
  WhirlpoolDiskStars_eqFunction_4039(data, threadData);
  WhirlpoolDiskStars_eqFunction_4038(data, threadData);
  WhirlpoolDiskStars_eqFunction_4036(data, threadData);
  WhirlpoolDiskStars_eqFunction_2367(data, threadData);
  WhirlpoolDiskStars_eqFunction_4031(data, threadData);
  WhirlpoolDiskStars_eqFunction_2369(data, threadData);
  WhirlpoolDiskStars_eqFunction_4044(data, threadData);
  WhirlpoolDiskStars_eqFunction_2371(data, threadData);
  WhirlpoolDiskStars_eqFunction_2372(data, threadData);
  WhirlpoolDiskStars_eqFunction_4043(data, threadData);
  WhirlpoolDiskStars_eqFunction_2374(data, threadData);
  WhirlpoolDiskStars_eqFunction_2375(data, threadData);
  WhirlpoolDiskStars_eqFunction_4042(data, threadData);
  WhirlpoolDiskStars_eqFunction_4045(data, threadData);
  WhirlpoolDiskStars_eqFunction_4047(data, threadData);
  WhirlpoolDiskStars_eqFunction_4050(data, threadData);
  WhirlpoolDiskStars_eqFunction_4049(data, threadData);
  WhirlpoolDiskStars_eqFunction_4048(data, threadData);
  WhirlpoolDiskStars_eqFunction_4046(data, threadData);
  WhirlpoolDiskStars_eqFunction_2383(data, threadData);
  WhirlpoolDiskStars_eqFunction_4041(data, threadData);
  WhirlpoolDiskStars_eqFunction_2385(data, threadData);
  WhirlpoolDiskStars_eqFunction_4054(data, threadData);
  WhirlpoolDiskStars_eqFunction_2387(data, threadData);
  WhirlpoolDiskStars_eqFunction_2388(data, threadData);
  WhirlpoolDiskStars_eqFunction_4053(data, threadData);
  WhirlpoolDiskStars_eqFunction_2390(data, threadData);
  WhirlpoolDiskStars_eqFunction_2391(data, threadData);
  WhirlpoolDiskStars_eqFunction_4052(data, threadData);
  WhirlpoolDiskStars_eqFunction_4055(data, threadData);
  WhirlpoolDiskStars_eqFunction_4057(data, threadData);
  WhirlpoolDiskStars_eqFunction_4060(data, threadData);
  WhirlpoolDiskStars_eqFunction_4059(data, threadData);
  WhirlpoolDiskStars_eqFunction_4058(data, threadData);
  WhirlpoolDiskStars_eqFunction_4056(data, threadData);
  WhirlpoolDiskStars_eqFunction_2399(data, threadData);
  WhirlpoolDiskStars_eqFunction_4051(data, threadData);
  WhirlpoolDiskStars_eqFunction_2401(data, threadData);
  WhirlpoolDiskStars_eqFunction_4064(data, threadData);
  WhirlpoolDiskStars_eqFunction_2403(data, threadData);
  WhirlpoolDiskStars_eqFunction_2404(data, threadData);
  WhirlpoolDiskStars_eqFunction_4063(data, threadData);
  WhirlpoolDiskStars_eqFunction_2406(data, threadData);
  WhirlpoolDiskStars_eqFunction_2407(data, threadData);
  WhirlpoolDiskStars_eqFunction_4062(data, threadData);
  WhirlpoolDiskStars_eqFunction_4065(data, threadData);
  WhirlpoolDiskStars_eqFunction_4067(data, threadData);
  WhirlpoolDiskStars_eqFunction_4070(data, threadData);
  WhirlpoolDiskStars_eqFunction_4069(data, threadData);
  WhirlpoolDiskStars_eqFunction_4068(data, threadData);
  WhirlpoolDiskStars_eqFunction_4066(data, threadData);
  WhirlpoolDiskStars_eqFunction_2415(data, threadData);
  WhirlpoolDiskStars_eqFunction_4061(data, threadData);
  WhirlpoolDiskStars_eqFunction_2417(data, threadData);
  WhirlpoolDiskStars_eqFunction_4074(data, threadData);
  WhirlpoolDiskStars_eqFunction_2419(data, threadData);
  WhirlpoolDiskStars_eqFunction_2420(data, threadData);
  WhirlpoolDiskStars_eqFunction_4073(data, threadData);
  WhirlpoolDiskStars_eqFunction_2422(data, threadData);
  WhirlpoolDiskStars_eqFunction_2423(data, threadData);
  WhirlpoolDiskStars_eqFunction_4072(data, threadData);
  WhirlpoolDiskStars_eqFunction_4075(data, threadData);
  WhirlpoolDiskStars_eqFunction_4077(data, threadData);
  WhirlpoolDiskStars_eqFunction_4080(data, threadData);
  WhirlpoolDiskStars_eqFunction_4079(data, threadData);
  WhirlpoolDiskStars_eqFunction_4078(data, threadData);
  WhirlpoolDiskStars_eqFunction_4076(data, threadData);
  WhirlpoolDiskStars_eqFunction_2431(data, threadData);
  WhirlpoolDiskStars_eqFunction_4071(data, threadData);
  WhirlpoolDiskStars_eqFunction_2433(data, threadData);
  WhirlpoolDiskStars_eqFunction_4084(data, threadData);
  WhirlpoolDiskStars_eqFunction_2435(data, threadData);
  WhirlpoolDiskStars_eqFunction_2436(data, threadData);
  WhirlpoolDiskStars_eqFunction_4083(data, threadData);
  WhirlpoolDiskStars_eqFunction_2438(data, threadData);
  WhirlpoolDiskStars_eqFunction_2439(data, threadData);
  WhirlpoolDiskStars_eqFunction_4082(data, threadData);
  WhirlpoolDiskStars_eqFunction_4085(data, threadData);
  WhirlpoolDiskStars_eqFunction_4087(data, threadData);
  WhirlpoolDiskStars_eqFunction_4090(data, threadData);
  WhirlpoolDiskStars_eqFunction_4089(data, threadData);
  WhirlpoolDiskStars_eqFunction_4088(data, threadData);
  WhirlpoolDiskStars_eqFunction_4086(data, threadData);
  WhirlpoolDiskStars_eqFunction_2447(data, threadData);
  WhirlpoolDiskStars_eqFunction_4081(data, threadData);
  WhirlpoolDiskStars_eqFunction_2449(data, threadData);
  WhirlpoolDiskStars_eqFunction_4094(data, threadData);
  WhirlpoolDiskStars_eqFunction_2451(data, threadData);
  WhirlpoolDiskStars_eqFunction_2452(data, threadData);
  WhirlpoolDiskStars_eqFunction_4093(data, threadData);
  WhirlpoolDiskStars_eqFunction_2454(data, threadData);
  WhirlpoolDiskStars_eqFunction_2455(data, threadData);
  WhirlpoolDiskStars_eqFunction_4092(data, threadData);
  WhirlpoolDiskStars_eqFunction_4095(data, threadData);
  WhirlpoolDiskStars_eqFunction_4097(data, threadData);
  WhirlpoolDiskStars_eqFunction_4100(data, threadData);
  WhirlpoolDiskStars_eqFunction_4099(data, threadData);
  WhirlpoolDiskStars_eqFunction_4098(data, threadData);
  WhirlpoolDiskStars_eqFunction_4096(data, threadData);
  WhirlpoolDiskStars_eqFunction_2463(data, threadData);
  WhirlpoolDiskStars_eqFunction_4091(data, threadData);
  WhirlpoolDiskStars_eqFunction_2465(data, threadData);
  WhirlpoolDiskStars_eqFunction_4104(data, threadData);
  WhirlpoolDiskStars_eqFunction_2467(data, threadData);
  WhirlpoolDiskStars_eqFunction_2468(data, threadData);
  WhirlpoolDiskStars_eqFunction_4103(data, threadData);
  WhirlpoolDiskStars_eqFunction_2470(data, threadData);
  WhirlpoolDiskStars_eqFunction_2471(data, threadData);
  WhirlpoolDiskStars_eqFunction_4102(data, threadData);
  WhirlpoolDiskStars_eqFunction_4105(data, threadData);
  WhirlpoolDiskStars_eqFunction_4107(data, threadData);
  WhirlpoolDiskStars_eqFunction_4110(data, threadData);
  WhirlpoolDiskStars_eqFunction_4109(data, threadData);
  WhirlpoolDiskStars_eqFunction_4108(data, threadData);
  WhirlpoolDiskStars_eqFunction_4106(data, threadData);
  WhirlpoolDiskStars_eqFunction_2479(data, threadData);
  WhirlpoolDiskStars_eqFunction_4101(data, threadData);
  WhirlpoolDiskStars_eqFunction_2481(data, threadData);
  WhirlpoolDiskStars_eqFunction_4114(data, threadData);
  WhirlpoolDiskStars_eqFunction_2483(data, threadData);
  WhirlpoolDiskStars_eqFunction_2484(data, threadData);
  WhirlpoolDiskStars_eqFunction_4113(data, threadData);
  WhirlpoolDiskStars_eqFunction_2486(data, threadData);
  WhirlpoolDiskStars_eqFunction_2487(data, threadData);
  WhirlpoolDiskStars_eqFunction_4112(data, threadData);
  WhirlpoolDiskStars_eqFunction_4115(data, threadData);
  WhirlpoolDiskStars_eqFunction_4117(data, threadData);
  WhirlpoolDiskStars_eqFunction_4120(data, threadData);
  WhirlpoolDiskStars_eqFunction_4119(data, threadData);
  WhirlpoolDiskStars_eqFunction_4118(data, threadData);
  WhirlpoolDiskStars_eqFunction_4116(data, threadData);
  WhirlpoolDiskStars_eqFunction_2495(data, threadData);
  WhirlpoolDiskStars_eqFunction_4111(data, threadData);
  WhirlpoolDiskStars_eqFunction_2497(data, threadData);
  WhirlpoolDiskStars_eqFunction_4124(data, threadData);
  WhirlpoolDiskStars_eqFunction_2499(data, threadData);
  WhirlpoolDiskStars_eqFunction_2500(data, threadData);
  WhirlpoolDiskStars_eqFunction_4123(data, threadData);
  WhirlpoolDiskStars_eqFunction_2502(data, threadData);
  WhirlpoolDiskStars_eqFunction_2503(data, threadData);
  WhirlpoolDiskStars_eqFunction_4122(data, threadData);
  WhirlpoolDiskStars_eqFunction_4125(data, threadData);
  WhirlpoolDiskStars_eqFunction_4127(data, threadData);
  WhirlpoolDiskStars_eqFunction_4130(data, threadData);
  WhirlpoolDiskStars_eqFunction_4129(data, threadData);
  WhirlpoolDiskStars_eqFunction_4128(data, threadData);
  WhirlpoolDiskStars_eqFunction_4126(data, threadData);
  WhirlpoolDiskStars_eqFunction_2511(data, threadData);
  WhirlpoolDiskStars_eqFunction_4121(data, threadData);
  WhirlpoolDiskStars_eqFunction_2513(data, threadData);
  WhirlpoolDiskStars_eqFunction_4134(data, threadData);
  WhirlpoolDiskStars_eqFunction_2515(data, threadData);
  WhirlpoolDiskStars_eqFunction_2516(data, threadData);
  WhirlpoolDiskStars_eqFunction_4133(data, threadData);
  WhirlpoolDiskStars_eqFunction_2518(data, threadData);
  WhirlpoolDiskStars_eqFunction_2519(data, threadData);
  WhirlpoolDiskStars_eqFunction_4132(data, threadData);
  WhirlpoolDiskStars_eqFunction_4135(data, threadData);
  WhirlpoolDiskStars_eqFunction_4137(data, threadData);
  WhirlpoolDiskStars_eqFunction_4140(data, threadData);
  WhirlpoolDiskStars_eqFunction_4139(data, threadData);
  WhirlpoolDiskStars_eqFunction_4138(data, threadData);
  WhirlpoolDiskStars_eqFunction_4136(data, threadData);
  WhirlpoolDiskStars_eqFunction_2527(data, threadData);
  WhirlpoolDiskStars_eqFunction_4131(data, threadData);
  WhirlpoolDiskStars_eqFunction_2529(data, threadData);
  WhirlpoolDiskStars_eqFunction_4144(data, threadData);
  WhirlpoolDiskStars_eqFunction_2531(data, threadData);
  WhirlpoolDiskStars_eqFunction_2532(data, threadData);
  WhirlpoolDiskStars_eqFunction_4143(data, threadData);
  WhirlpoolDiskStars_eqFunction_2534(data, threadData);
  WhirlpoolDiskStars_eqFunction_2535(data, threadData);
  WhirlpoolDiskStars_eqFunction_4142(data, threadData);
  WhirlpoolDiskStars_eqFunction_4145(data, threadData);
  WhirlpoolDiskStars_eqFunction_4147(data, threadData);
  WhirlpoolDiskStars_eqFunction_4150(data, threadData);
  WhirlpoolDiskStars_eqFunction_4149(data, threadData);
  WhirlpoolDiskStars_eqFunction_4148(data, threadData);
  WhirlpoolDiskStars_eqFunction_4146(data, threadData);
  WhirlpoolDiskStars_eqFunction_2543(data, threadData);
  WhirlpoolDiskStars_eqFunction_4141(data, threadData);
  WhirlpoolDiskStars_eqFunction_2545(data, threadData);
  WhirlpoolDiskStars_eqFunction_4154(data, threadData);
  WhirlpoolDiskStars_eqFunction_2547(data, threadData);
  WhirlpoolDiskStars_eqFunction_2548(data, threadData);
  WhirlpoolDiskStars_eqFunction_4153(data, threadData);
  WhirlpoolDiskStars_eqFunction_2550(data, threadData);
  WhirlpoolDiskStars_eqFunction_2551(data, threadData);
  WhirlpoolDiskStars_eqFunction_4152(data, threadData);
  WhirlpoolDiskStars_eqFunction_4155(data, threadData);
  WhirlpoolDiskStars_eqFunction_4157(data, threadData);
  WhirlpoolDiskStars_eqFunction_4160(data, threadData);
  WhirlpoolDiskStars_eqFunction_4159(data, threadData);
  WhirlpoolDiskStars_eqFunction_4158(data, threadData);
  WhirlpoolDiskStars_eqFunction_4156(data, threadData);
  WhirlpoolDiskStars_eqFunction_2559(data, threadData);
  WhirlpoolDiskStars_eqFunction_4151(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif