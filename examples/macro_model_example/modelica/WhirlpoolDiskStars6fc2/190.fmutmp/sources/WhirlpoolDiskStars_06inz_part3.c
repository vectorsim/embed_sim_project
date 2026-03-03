#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void WhirlpoolDiskStars_eqFunction_3364(DATA *data, threadData_t *threadData);


/*
equation index: 1283
type: SIMPLE_ASSIGN
y[81] = r_init[81] * sin(theta[81] + armOffset[81])
*/
void WhirlpoolDiskStars_eqFunction_1283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1283};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[720]] /* y[81] STATE(1,vy[81]) */) = ((data->simulationInfo->realParameter[245] /* r_init[81] PARAM */)) * (sin((data->simulationInfo->realParameter[405] /* theta[81] PARAM */) + (data->simulationInfo->realParameter[83] /* armOffset[81] PARAM */)));
  TRACE_POP
}

/*
equation index: 1284
type: SIMPLE_ASSIGN
vx[81] = (-y[81]) * sqrt(G * Md / r_init[81] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1284};
  modelica_real tmp320;
  modelica_real tmp321;
  tmp320 = (data->simulationInfo->realParameter[245] /* r_init[81] PARAM */);
  tmp321 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp320 * tmp320 * tmp320),"r_init[81] ^ 3.0",equationIndexes);
  if(!(tmp321 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[81] ^ 3.0) was %g should be >= 0", tmp321);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[80]] /* vx[81] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[720]] /* y[81] STATE(1,vy[81]) */))) * (sqrt(tmp321));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3363(DATA *data, threadData_t *threadData);


/*
equation index: 1286
type: SIMPLE_ASSIGN
x[81] = r_init[81] * cos(theta[81] + armOffset[81])
*/
void WhirlpoolDiskStars_eqFunction_1286(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1286};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[560]] /* x[81] STATE(1,vx[81]) */) = ((data->simulationInfo->realParameter[245] /* r_init[81] PARAM */)) * (cos((data->simulationInfo->realParameter[405] /* theta[81] PARAM */) + (data->simulationInfo->realParameter[83] /* armOffset[81] PARAM */)));
  TRACE_POP
}

/*
equation index: 1287
type: SIMPLE_ASSIGN
vy[81] = x[81] * sqrt(G * Md / r_init[81] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1287};
  modelica_real tmp322;
  modelica_real tmp323;
  tmp322 = (data->simulationInfo->realParameter[245] /* r_init[81] PARAM */);
  tmp323 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp322 * tmp322 * tmp322),"r_init[81] ^ 3.0",equationIndexes);
  if(!(tmp323 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[81] ^ 3.0) was %g should be >= 0", tmp323);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[240]] /* vy[81] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[560]] /* x[81] STATE(1,vx[81]) */)) * (sqrt(tmp323));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3362(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3365(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3367(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3370(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3369(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3368(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3366(DATA *data, threadData_t *threadData);


/*
equation index: 1295
type: SIMPLE_ASSIGN
vz[81] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[400]] /* vz[81] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3361(DATA *data, threadData_t *threadData);


/*
equation index: 1297
type: SIMPLE_ASSIGN
z[82] = 0.1
*/
void WhirlpoolDiskStars_eqFunction_1297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[881]] /* z[82] STATE(1,vz[82]) */) = 0.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3374(DATA *data, threadData_t *threadData);


/*
equation index: 1299
type: SIMPLE_ASSIGN
y[82] = r_init[82] * sin(theta[82] + armOffset[82])
*/
void WhirlpoolDiskStars_eqFunction_1299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1299};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[721]] /* y[82] STATE(1,vy[82]) */) = ((data->simulationInfo->realParameter[246] /* r_init[82] PARAM */)) * (sin((data->simulationInfo->realParameter[406] /* theta[82] PARAM */) + (data->simulationInfo->realParameter[84] /* armOffset[82] PARAM */)));
  TRACE_POP
}

/*
equation index: 1300
type: SIMPLE_ASSIGN
vx[82] = (-y[82]) * sqrt(G * Md / r_init[82] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1300};
  modelica_real tmp324;
  modelica_real tmp325;
  tmp324 = (data->simulationInfo->realParameter[246] /* r_init[82] PARAM */);
  tmp325 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp324 * tmp324 * tmp324),"r_init[82] ^ 3.0",equationIndexes);
  if(!(tmp325 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[82] ^ 3.0) was %g should be >= 0", tmp325);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[81]] /* vx[82] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[721]] /* y[82] STATE(1,vy[82]) */))) * (sqrt(tmp325));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3373(DATA *data, threadData_t *threadData);


/*
equation index: 1302
type: SIMPLE_ASSIGN
x[82] = r_init[82] * cos(theta[82] + armOffset[82])
*/
void WhirlpoolDiskStars_eqFunction_1302(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1302};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[561]] /* x[82] STATE(1,vx[82]) */) = ((data->simulationInfo->realParameter[246] /* r_init[82] PARAM */)) * (cos((data->simulationInfo->realParameter[406] /* theta[82] PARAM */) + (data->simulationInfo->realParameter[84] /* armOffset[82] PARAM */)));
  TRACE_POP
}

/*
equation index: 1303
type: SIMPLE_ASSIGN
vy[82] = x[82] * sqrt(G * Md / r_init[82] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1303};
  modelica_real tmp326;
  modelica_real tmp327;
  tmp326 = (data->simulationInfo->realParameter[246] /* r_init[82] PARAM */);
  tmp327 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp326 * tmp326 * tmp326),"r_init[82] ^ 3.0",equationIndexes);
  if(!(tmp327 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[82] ^ 3.0) was %g should be >= 0", tmp327);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[241]] /* vy[82] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[561]] /* x[82] STATE(1,vx[82]) */)) * (sqrt(tmp327));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3372(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3375(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3377(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3380(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3379(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3378(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3376(DATA *data, threadData_t *threadData);


/*
equation index: 1311
type: SIMPLE_ASSIGN
vz[82] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[401]] /* vz[82] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3371(DATA *data, threadData_t *threadData);


/*
equation index: 1313
type: SIMPLE_ASSIGN
z[83] = 0.15000000000000002
*/
void WhirlpoolDiskStars_eqFunction_1313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[882]] /* z[83] STATE(1,vz[83]) */) = 0.15000000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3384(DATA *data, threadData_t *threadData);


/*
equation index: 1315
type: SIMPLE_ASSIGN
y[83] = r_init[83] * sin(theta[83] + armOffset[83])
*/
void WhirlpoolDiskStars_eqFunction_1315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1315};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[722]] /* y[83] STATE(1,vy[83]) */) = ((data->simulationInfo->realParameter[247] /* r_init[83] PARAM */)) * (sin((data->simulationInfo->realParameter[407] /* theta[83] PARAM */) + (data->simulationInfo->realParameter[85] /* armOffset[83] PARAM */)));
  TRACE_POP
}

/*
equation index: 1316
type: SIMPLE_ASSIGN
vx[83] = (-y[83]) * sqrt(G * Md / r_init[83] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1316};
  modelica_real tmp328;
  modelica_real tmp329;
  tmp328 = (data->simulationInfo->realParameter[247] /* r_init[83] PARAM */);
  tmp329 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp328 * tmp328 * tmp328),"r_init[83] ^ 3.0",equationIndexes);
  if(!(tmp329 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[83] ^ 3.0) was %g should be >= 0", tmp329);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[82]] /* vx[83] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[722]] /* y[83] STATE(1,vy[83]) */))) * (sqrt(tmp329));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3383(DATA *data, threadData_t *threadData);


/*
equation index: 1318
type: SIMPLE_ASSIGN
x[83] = r_init[83] * cos(theta[83] + armOffset[83])
*/
void WhirlpoolDiskStars_eqFunction_1318(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1318};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[562]] /* x[83] STATE(1,vx[83]) */) = ((data->simulationInfo->realParameter[247] /* r_init[83] PARAM */)) * (cos((data->simulationInfo->realParameter[407] /* theta[83] PARAM */) + (data->simulationInfo->realParameter[85] /* armOffset[83] PARAM */)));
  TRACE_POP
}

/*
equation index: 1319
type: SIMPLE_ASSIGN
vy[83] = x[83] * sqrt(G * Md / r_init[83] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1319};
  modelica_real tmp330;
  modelica_real tmp331;
  tmp330 = (data->simulationInfo->realParameter[247] /* r_init[83] PARAM */);
  tmp331 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp330 * tmp330 * tmp330),"r_init[83] ^ 3.0",equationIndexes);
  if(!(tmp331 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[83] ^ 3.0) was %g should be >= 0", tmp331);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[242]] /* vy[83] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[562]] /* x[83] STATE(1,vx[83]) */)) * (sqrt(tmp331));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3382(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3385(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3387(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3390(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3389(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3388(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3386(DATA *data, threadData_t *threadData);


/*
equation index: 1327
type: SIMPLE_ASSIGN
vz[83] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1327};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[402]] /* vz[83] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3381(DATA *data, threadData_t *threadData);


/*
equation index: 1329
type: SIMPLE_ASSIGN
z[84] = 0.2
*/
void WhirlpoolDiskStars_eqFunction_1329(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1329};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[883]] /* z[84] STATE(1,vz[84]) */) = 0.2;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3394(DATA *data, threadData_t *threadData);


/*
equation index: 1331
type: SIMPLE_ASSIGN
y[84] = r_init[84] * sin(theta[84] + armOffset[84])
*/
void WhirlpoolDiskStars_eqFunction_1331(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1331};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[723]] /* y[84] STATE(1,vy[84]) */) = ((data->simulationInfo->realParameter[248] /* r_init[84] PARAM */)) * (sin((data->simulationInfo->realParameter[408] /* theta[84] PARAM */) + (data->simulationInfo->realParameter[86] /* armOffset[84] PARAM */)));
  TRACE_POP
}

/*
equation index: 1332
type: SIMPLE_ASSIGN
vx[84] = (-y[84]) * sqrt(G * Md / r_init[84] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1332(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1332};
  modelica_real tmp332;
  modelica_real tmp333;
  tmp332 = (data->simulationInfo->realParameter[248] /* r_init[84] PARAM */);
  tmp333 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp332 * tmp332 * tmp332),"r_init[84] ^ 3.0",equationIndexes);
  if(!(tmp333 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[84] ^ 3.0) was %g should be >= 0", tmp333);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[83]] /* vx[84] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[723]] /* y[84] STATE(1,vy[84]) */))) * (sqrt(tmp333));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3393(DATA *data, threadData_t *threadData);


/*
equation index: 1334
type: SIMPLE_ASSIGN
x[84] = r_init[84] * cos(theta[84] + armOffset[84])
*/
void WhirlpoolDiskStars_eqFunction_1334(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1334};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[563]] /* x[84] STATE(1,vx[84]) */) = ((data->simulationInfo->realParameter[248] /* r_init[84] PARAM */)) * (cos((data->simulationInfo->realParameter[408] /* theta[84] PARAM */) + (data->simulationInfo->realParameter[86] /* armOffset[84] PARAM */)));
  TRACE_POP
}

/*
equation index: 1335
type: SIMPLE_ASSIGN
vy[84] = x[84] * sqrt(G * Md / r_init[84] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1335};
  modelica_real tmp334;
  modelica_real tmp335;
  tmp334 = (data->simulationInfo->realParameter[248] /* r_init[84] PARAM */);
  tmp335 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp334 * tmp334 * tmp334),"r_init[84] ^ 3.0",equationIndexes);
  if(!(tmp335 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[84] ^ 3.0) was %g should be >= 0", tmp335);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[243]] /* vy[84] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[563]] /* x[84] STATE(1,vx[84]) */)) * (sqrt(tmp335));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3392(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3395(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3397(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3400(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3399(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3398(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3396(DATA *data, threadData_t *threadData);


/*
equation index: 1343
type: SIMPLE_ASSIGN
vz[84] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[403]] /* vz[84] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3391(DATA *data, threadData_t *threadData);


/*
equation index: 1345
type: SIMPLE_ASSIGN
z[85] = 0.25
*/
void WhirlpoolDiskStars_eqFunction_1345(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[884]] /* z[85] STATE(1,vz[85]) */) = 0.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3404(DATA *data, threadData_t *threadData);


/*
equation index: 1347
type: SIMPLE_ASSIGN
y[85] = r_init[85] * sin(theta[85] + armOffset[85])
*/
void WhirlpoolDiskStars_eqFunction_1347(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1347};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[724]] /* y[85] STATE(1,vy[85]) */) = ((data->simulationInfo->realParameter[249] /* r_init[85] PARAM */)) * (sin((data->simulationInfo->realParameter[409] /* theta[85] PARAM */) + (data->simulationInfo->realParameter[87] /* armOffset[85] PARAM */)));
  TRACE_POP
}

/*
equation index: 1348
type: SIMPLE_ASSIGN
vx[85] = (-y[85]) * sqrt(G * Md / r_init[85] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1348(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1348};
  modelica_real tmp336;
  modelica_real tmp337;
  tmp336 = (data->simulationInfo->realParameter[249] /* r_init[85] PARAM */);
  tmp337 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp336 * tmp336 * tmp336),"r_init[85] ^ 3.0",equationIndexes);
  if(!(tmp337 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[85] ^ 3.0) was %g should be >= 0", tmp337);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[84]] /* vx[85] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[724]] /* y[85] STATE(1,vy[85]) */))) * (sqrt(tmp337));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3403(DATA *data, threadData_t *threadData);


/*
equation index: 1350
type: SIMPLE_ASSIGN
x[85] = r_init[85] * cos(theta[85] + armOffset[85])
*/
void WhirlpoolDiskStars_eqFunction_1350(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1350};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[564]] /* x[85] STATE(1,vx[85]) */) = ((data->simulationInfo->realParameter[249] /* r_init[85] PARAM */)) * (cos((data->simulationInfo->realParameter[409] /* theta[85] PARAM */) + (data->simulationInfo->realParameter[87] /* armOffset[85] PARAM */)));
  TRACE_POP
}

/*
equation index: 1351
type: SIMPLE_ASSIGN
vy[85] = x[85] * sqrt(G * Md / r_init[85] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1351};
  modelica_real tmp338;
  modelica_real tmp339;
  tmp338 = (data->simulationInfo->realParameter[249] /* r_init[85] PARAM */);
  tmp339 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp338 * tmp338 * tmp338),"r_init[85] ^ 3.0",equationIndexes);
  if(!(tmp339 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[85] ^ 3.0) was %g should be >= 0", tmp339);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[244]] /* vy[85] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[564]] /* x[85] STATE(1,vx[85]) */)) * (sqrt(tmp339));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3402(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3405(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3407(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3410(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3409(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3408(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3406(DATA *data, threadData_t *threadData);


/*
equation index: 1359
type: SIMPLE_ASSIGN
vz[85] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1359};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[404]] /* vz[85] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3401(DATA *data, threadData_t *threadData);


/*
equation index: 1361
type: SIMPLE_ASSIGN
z[86] = 0.30000000000000004
*/
void WhirlpoolDiskStars_eqFunction_1361(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1361};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[885]] /* z[86] STATE(1,vz[86]) */) = 0.30000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3414(DATA *data, threadData_t *threadData);


/*
equation index: 1363
type: SIMPLE_ASSIGN
y[86] = r_init[86] * sin(theta[86] + armOffset[86])
*/
void WhirlpoolDiskStars_eqFunction_1363(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1363};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[725]] /* y[86] STATE(1,vy[86]) */) = ((data->simulationInfo->realParameter[250] /* r_init[86] PARAM */)) * (sin((data->simulationInfo->realParameter[410] /* theta[86] PARAM */) + (data->simulationInfo->realParameter[88] /* armOffset[86] PARAM */)));
  TRACE_POP
}

/*
equation index: 1364
type: SIMPLE_ASSIGN
vx[86] = (-y[86]) * sqrt(G * Md / r_init[86] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1364(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1364};
  modelica_real tmp340;
  modelica_real tmp341;
  tmp340 = (data->simulationInfo->realParameter[250] /* r_init[86] PARAM */);
  tmp341 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp340 * tmp340 * tmp340),"r_init[86] ^ 3.0",equationIndexes);
  if(!(tmp341 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[86] ^ 3.0) was %g should be >= 0", tmp341);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[85]] /* vx[86] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[725]] /* y[86] STATE(1,vy[86]) */))) * (sqrt(tmp341));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3413(DATA *data, threadData_t *threadData);


/*
equation index: 1366
type: SIMPLE_ASSIGN
x[86] = r_init[86] * cos(theta[86] + armOffset[86])
*/
void WhirlpoolDiskStars_eqFunction_1366(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1366};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[565]] /* x[86] STATE(1,vx[86]) */) = ((data->simulationInfo->realParameter[250] /* r_init[86] PARAM */)) * (cos((data->simulationInfo->realParameter[410] /* theta[86] PARAM */) + (data->simulationInfo->realParameter[88] /* armOffset[86] PARAM */)));
  TRACE_POP
}

/*
equation index: 1367
type: SIMPLE_ASSIGN
vy[86] = x[86] * sqrt(G * Md / r_init[86] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1367};
  modelica_real tmp342;
  modelica_real tmp343;
  tmp342 = (data->simulationInfo->realParameter[250] /* r_init[86] PARAM */);
  tmp343 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp342 * tmp342 * tmp342),"r_init[86] ^ 3.0",equationIndexes);
  if(!(tmp343 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[86] ^ 3.0) was %g should be >= 0", tmp343);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[245]] /* vy[86] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[565]] /* x[86] STATE(1,vx[86]) */)) * (sqrt(tmp343));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3412(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3415(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3417(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3420(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3419(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3418(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3416(DATA *data, threadData_t *threadData);


/*
equation index: 1375
type: SIMPLE_ASSIGN
vz[86] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1375};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[405]] /* vz[86] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3411(DATA *data, threadData_t *threadData);


/*
equation index: 1377
type: SIMPLE_ASSIGN
z[87] = 0.35000000000000003
*/
void WhirlpoolDiskStars_eqFunction_1377(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1377};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[886]] /* z[87] STATE(1,vz[87]) */) = 0.35000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3424(DATA *data, threadData_t *threadData);


/*
equation index: 1379
type: SIMPLE_ASSIGN
y[87] = r_init[87] * sin(theta[87] + armOffset[87])
*/
void WhirlpoolDiskStars_eqFunction_1379(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1379};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[726]] /* y[87] STATE(1,vy[87]) */) = ((data->simulationInfo->realParameter[251] /* r_init[87] PARAM */)) * (sin((data->simulationInfo->realParameter[411] /* theta[87] PARAM */) + (data->simulationInfo->realParameter[89] /* armOffset[87] PARAM */)));
  TRACE_POP
}

/*
equation index: 1380
type: SIMPLE_ASSIGN
vx[87] = (-y[87]) * sqrt(G * Md / r_init[87] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1380(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1380};
  modelica_real tmp344;
  modelica_real tmp345;
  tmp344 = (data->simulationInfo->realParameter[251] /* r_init[87] PARAM */);
  tmp345 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp344 * tmp344 * tmp344),"r_init[87] ^ 3.0",equationIndexes);
  if(!(tmp345 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[87] ^ 3.0) was %g should be >= 0", tmp345);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[86]] /* vx[87] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[726]] /* y[87] STATE(1,vy[87]) */))) * (sqrt(tmp345));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3423(DATA *data, threadData_t *threadData);


/*
equation index: 1382
type: SIMPLE_ASSIGN
x[87] = r_init[87] * cos(theta[87] + armOffset[87])
*/
void WhirlpoolDiskStars_eqFunction_1382(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1382};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[566]] /* x[87] STATE(1,vx[87]) */) = ((data->simulationInfo->realParameter[251] /* r_init[87] PARAM */)) * (cos((data->simulationInfo->realParameter[411] /* theta[87] PARAM */) + (data->simulationInfo->realParameter[89] /* armOffset[87] PARAM */)));
  TRACE_POP
}

/*
equation index: 1383
type: SIMPLE_ASSIGN
vy[87] = x[87] * sqrt(G * Md / r_init[87] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1383};
  modelica_real tmp346;
  modelica_real tmp347;
  tmp346 = (data->simulationInfo->realParameter[251] /* r_init[87] PARAM */);
  tmp347 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp346 * tmp346 * tmp346),"r_init[87] ^ 3.0",equationIndexes);
  if(!(tmp347 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[87] ^ 3.0) was %g should be >= 0", tmp347);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[246]] /* vy[87] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[566]] /* x[87] STATE(1,vx[87]) */)) * (sqrt(tmp347));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3422(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3425(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3427(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3430(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3429(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3428(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3426(DATA *data, threadData_t *threadData);


/*
equation index: 1391
type: SIMPLE_ASSIGN
vz[87] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1391};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[406]] /* vz[87] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3421(DATA *data, threadData_t *threadData);


/*
equation index: 1393
type: SIMPLE_ASSIGN
z[88] = 0.4
*/
void WhirlpoolDiskStars_eqFunction_1393(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1393};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[887]] /* z[88] STATE(1,vz[88]) */) = 0.4;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3434(DATA *data, threadData_t *threadData);


/*
equation index: 1395
type: SIMPLE_ASSIGN
y[88] = r_init[88] * sin(theta[88] + armOffset[88])
*/
void WhirlpoolDiskStars_eqFunction_1395(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1395};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[727]] /* y[88] STATE(1,vy[88]) */) = ((data->simulationInfo->realParameter[252] /* r_init[88] PARAM */)) * (sin((data->simulationInfo->realParameter[412] /* theta[88] PARAM */) + (data->simulationInfo->realParameter[90] /* armOffset[88] PARAM */)));
  TRACE_POP
}

/*
equation index: 1396
type: SIMPLE_ASSIGN
vx[88] = (-y[88]) * sqrt(G * Md / r_init[88] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1396(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1396};
  modelica_real tmp348;
  modelica_real tmp349;
  tmp348 = (data->simulationInfo->realParameter[252] /* r_init[88] PARAM */);
  tmp349 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp348 * tmp348 * tmp348),"r_init[88] ^ 3.0",equationIndexes);
  if(!(tmp349 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[88] ^ 3.0) was %g should be >= 0", tmp349);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[87]] /* vx[88] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[727]] /* y[88] STATE(1,vy[88]) */))) * (sqrt(tmp349));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3433(DATA *data, threadData_t *threadData);


/*
equation index: 1398
type: SIMPLE_ASSIGN
x[88] = r_init[88] * cos(theta[88] + armOffset[88])
*/
void WhirlpoolDiskStars_eqFunction_1398(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1398};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[567]] /* x[88] STATE(1,vx[88]) */) = ((data->simulationInfo->realParameter[252] /* r_init[88] PARAM */)) * (cos((data->simulationInfo->realParameter[412] /* theta[88] PARAM */) + (data->simulationInfo->realParameter[90] /* armOffset[88] PARAM */)));
  TRACE_POP
}

/*
equation index: 1399
type: SIMPLE_ASSIGN
vy[88] = x[88] * sqrt(G * Md / r_init[88] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1399};
  modelica_real tmp350;
  modelica_real tmp351;
  tmp350 = (data->simulationInfo->realParameter[252] /* r_init[88] PARAM */);
  tmp351 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp350 * tmp350 * tmp350),"r_init[88] ^ 3.0",equationIndexes);
  if(!(tmp351 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[88] ^ 3.0) was %g should be >= 0", tmp351);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[247]] /* vy[88] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[567]] /* x[88] STATE(1,vx[88]) */)) * (sqrt(tmp351));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3432(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3435(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3437(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3440(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3439(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3438(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3436(DATA *data, threadData_t *threadData);


/*
equation index: 1407
type: SIMPLE_ASSIGN
vz[88] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1407};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[407]] /* vz[88] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3431(DATA *data, threadData_t *threadData);


/*
equation index: 1409
type: SIMPLE_ASSIGN
z[89] = 0.45
*/
void WhirlpoolDiskStars_eqFunction_1409(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1409};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[888]] /* z[89] STATE(1,vz[89]) */) = 0.45;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3444(DATA *data, threadData_t *threadData);


/*
equation index: 1411
type: SIMPLE_ASSIGN
y[89] = r_init[89] * sin(theta[89] + armOffset[89])
*/
void WhirlpoolDiskStars_eqFunction_1411(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1411};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[728]] /* y[89] STATE(1,vy[89]) */) = ((data->simulationInfo->realParameter[253] /* r_init[89] PARAM */)) * (sin((data->simulationInfo->realParameter[413] /* theta[89] PARAM */) + (data->simulationInfo->realParameter[91] /* armOffset[89] PARAM */)));
  TRACE_POP
}

/*
equation index: 1412
type: SIMPLE_ASSIGN
vx[89] = (-y[89]) * sqrt(G * Md / r_init[89] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1412(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1412};
  modelica_real tmp352;
  modelica_real tmp353;
  tmp352 = (data->simulationInfo->realParameter[253] /* r_init[89] PARAM */);
  tmp353 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp352 * tmp352 * tmp352),"r_init[89] ^ 3.0",equationIndexes);
  if(!(tmp353 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[89] ^ 3.0) was %g should be >= 0", tmp353);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[88]] /* vx[89] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[728]] /* y[89] STATE(1,vy[89]) */))) * (sqrt(tmp353));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3443(DATA *data, threadData_t *threadData);


/*
equation index: 1414
type: SIMPLE_ASSIGN
x[89] = r_init[89] * cos(theta[89] + armOffset[89])
*/
void WhirlpoolDiskStars_eqFunction_1414(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1414};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[568]] /* x[89] STATE(1,vx[89]) */) = ((data->simulationInfo->realParameter[253] /* r_init[89] PARAM */)) * (cos((data->simulationInfo->realParameter[413] /* theta[89] PARAM */) + (data->simulationInfo->realParameter[91] /* armOffset[89] PARAM */)));
  TRACE_POP
}

/*
equation index: 1415
type: SIMPLE_ASSIGN
vy[89] = x[89] * sqrt(G * Md / r_init[89] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1415};
  modelica_real tmp354;
  modelica_real tmp355;
  tmp354 = (data->simulationInfo->realParameter[253] /* r_init[89] PARAM */);
  tmp355 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp354 * tmp354 * tmp354),"r_init[89] ^ 3.0",equationIndexes);
  if(!(tmp355 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[89] ^ 3.0) was %g should be >= 0", tmp355);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[248]] /* vy[89] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[568]] /* x[89] STATE(1,vx[89]) */)) * (sqrt(tmp355));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3442(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3445(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3447(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3450(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3449(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3448(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3446(DATA *data, threadData_t *threadData);


/*
equation index: 1423
type: SIMPLE_ASSIGN
vz[89] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1423};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[408]] /* vz[89] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3441(DATA *data, threadData_t *threadData);


/*
equation index: 1425
type: SIMPLE_ASSIGN
z[90] = 0.5
*/
void WhirlpoolDiskStars_eqFunction_1425(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1425};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[889]] /* z[90] STATE(1,vz[90]) */) = 0.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3454(DATA *data, threadData_t *threadData);


/*
equation index: 1427
type: SIMPLE_ASSIGN
y[90] = r_init[90] * sin(theta[90] + armOffset[90])
*/
void WhirlpoolDiskStars_eqFunction_1427(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1427};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[729]] /* y[90] STATE(1,vy[90]) */) = ((data->simulationInfo->realParameter[254] /* r_init[90] PARAM */)) * (sin((data->simulationInfo->realParameter[414] /* theta[90] PARAM */) + (data->simulationInfo->realParameter[92] /* armOffset[90] PARAM */)));
  TRACE_POP
}

/*
equation index: 1428
type: SIMPLE_ASSIGN
vx[90] = (-y[90]) * sqrt(G * Md / r_init[90] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1428(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1428};
  modelica_real tmp356;
  modelica_real tmp357;
  tmp356 = (data->simulationInfo->realParameter[254] /* r_init[90] PARAM */);
  tmp357 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp356 * tmp356 * tmp356),"r_init[90] ^ 3.0",equationIndexes);
  if(!(tmp357 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[90] ^ 3.0) was %g should be >= 0", tmp357);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[89]] /* vx[90] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[729]] /* y[90] STATE(1,vy[90]) */))) * (sqrt(tmp357));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3453(DATA *data, threadData_t *threadData);


/*
equation index: 1430
type: SIMPLE_ASSIGN
x[90] = r_init[90] * cos(theta[90] + armOffset[90])
*/
void WhirlpoolDiskStars_eqFunction_1430(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1430};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[569]] /* x[90] STATE(1,vx[90]) */) = ((data->simulationInfo->realParameter[254] /* r_init[90] PARAM */)) * (cos((data->simulationInfo->realParameter[414] /* theta[90] PARAM */) + (data->simulationInfo->realParameter[92] /* armOffset[90] PARAM */)));
  TRACE_POP
}

/*
equation index: 1431
type: SIMPLE_ASSIGN
vy[90] = x[90] * sqrt(G * Md / r_init[90] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1431(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1431};
  modelica_real tmp358;
  modelica_real tmp359;
  tmp358 = (data->simulationInfo->realParameter[254] /* r_init[90] PARAM */);
  tmp359 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp358 * tmp358 * tmp358),"r_init[90] ^ 3.0",equationIndexes);
  if(!(tmp359 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[90] ^ 3.0) was %g should be >= 0", tmp359);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[249]] /* vy[90] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[569]] /* x[90] STATE(1,vx[90]) */)) * (sqrt(tmp359));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3452(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3455(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3457(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3460(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3459(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3458(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3456(DATA *data, threadData_t *threadData);


/*
equation index: 1439
type: SIMPLE_ASSIGN
vz[90] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1439(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1439};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[409]] /* vz[90] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3451(DATA *data, threadData_t *threadData);


/*
equation index: 1441
type: SIMPLE_ASSIGN
z[91] = 0.55
*/
void WhirlpoolDiskStars_eqFunction_1441(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1441};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[890]] /* z[91] STATE(1,vz[91]) */) = 0.55;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3464(DATA *data, threadData_t *threadData);


/*
equation index: 1443
type: SIMPLE_ASSIGN
y[91] = r_init[91] * sin(theta[91] + armOffset[91])
*/
void WhirlpoolDiskStars_eqFunction_1443(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1443};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[730]] /* y[91] STATE(1,vy[91]) */) = ((data->simulationInfo->realParameter[255] /* r_init[91] PARAM */)) * (sin((data->simulationInfo->realParameter[415] /* theta[91] PARAM */) + (data->simulationInfo->realParameter[93] /* armOffset[91] PARAM */)));
  TRACE_POP
}

/*
equation index: 1444
type: SIMPLE_ASSIGN
vx[91] = (-y[91]) * sqrt(G * Md / r_init[91] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1444(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1444};
  modelica_real tmp360;
  modelica_real tmp361;
  tmp360 = (data->simulationInfo->realParameter[255] /* r_init[91] PARAM */);
  tmp361 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp360 * tmp360 * tmp360),"r_init[91] ^ 3.0",equationIndexes);
  if(!(tmp361 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[91] ^ 3.0) was %g should be >= 0", tmp361);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[90]] /* vx[91] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[730]] /* y[91] STATE(1,vy[91]) */))) * (sqrt(tmp361));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3463(DATA *data, threadData_t *threadData);


/*
equation index: 1446
type: SIMPLE_ASSIGN
x[91] = r_init[91] * cos(theta[91] + armOffset[91])
*/
void WhirlpoolDiskStars_eqFunction_1446(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1446};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[570]] /* x[91] STATE(1,vx[91]) */) = ((data->simulationInfo->realParameter[255] /* r_init[91] PARAM */)) * (cos((data->simulationInfo->realParameter[415] /* theta[91] PARAM */) + (data->simulationInfo->realParameter[93] /* armOffset[91] PARAM */)));
  TRACE_POP
}

/*
equation index: 1447
type: SIMPLE_ASSIGN
vy[91] = x[91] * sqrt(G * Md / r_init[91] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1447(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1447};
  modelica_real tmp362;
  modelica_real tmp363;
  tmp362 = (data->simulationInfo->realParameter[255] /* r_init[91] PARAM */);
  tmp363 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp362 * tmp362 * tmp362),"r_init[91] ^ 3.0",equationIndexes);
  if(!(tmp363 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[91] ^ 3.0) was %g should be >= 0", tmp363);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[250]] /* vy[91] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[570]] /* x[91] STATE(1,vx[91]) */)) * (sqrt(tmp363));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3462(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3465(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3467(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3470(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3469(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3468(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3466(DATA *data, threadData_t *threadData);


/*
equation index: 1455
type: SIMPLE_ASSIGN
vz[91] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1455(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1455};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[410]] /* vz[91] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3461(DATA *data, threadData_t *threadData);


/*
equation index: 1457
type: SIMPLE_ASSIGN
z[92] = 0.6000000000000001
*/
void WhirlpoolDiskStars_eqFunction_1457(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1457};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[891]] /* z[92] STATE(1,vz[92]) */) = 0.6000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3474(DATA *data, threadData_t *threadData);


/*
equation index: 1459
type: SIMPLE_ASSIGN
y[92] = r_init[92] * sin(theta[92] + armOffset[92])
*/
void WhirlpoolDiskStars_eqFunction_1459(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1459};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[731]] /* y[92] STATE(1,vy[92]) */) = ((data->simulationInfo->realParameter[256] /* r_init[92] PARAM */)) * (sin((data->simulationInfo->realParameter[416] /* theta[92] PARAM */) + (data->simulationInfo->realParameter[94] /* armOffset[92] PARAM */)));
  TRACE_POP
}

/*
equation index: 1460
type: SIMPLE_ASSIGN
vx[92] = (-y[92]) * sqrt(G * Md / r_init[92] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1460(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1460};
  modelica_real tmp364;
  modelica_real tmp365;
  tmp364 = (data->simulationInfo->realParameter[256] /* r_init[92] PARAM */);
  tmp365 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp364 * tmp364 * tmp364),"r_init[92] ^ 3.0",equationIndexes);
  if(!(tmp365 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[92] ^ 3.0) was %g should be >= 0", tmp365);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[91]] /* vx[92] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[731]] /* y[92] STATE(1,vy[92]) */))) * (sqrt(tmp365));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3473(DATA *data, threadData_t *threadData);


/*
equation index: 1462
type: SIMPLE_ASSIGN
x[92] = r_init[92] * cos(theta[92] + armOffset[92])
*/
void WhirlpoolDiskStars_eqFunction_1462(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1462};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[571]] /* x[92] STATE(1,vx[92]) */) = ((data->simulationInfo->realParameter[256] /* r_init[92] PARAM */)) * (cos((data->simulationInfo->realParameter[416] /* theta[92] PARAM */) + (data->simulationInfo->realParameter[94] /* armOffset[92] PARAM */)));
  TRACE_POP
}

/*
equation index: 1463
type: SIMPLE_ASSIGN
vy[92] = x[92] * sqrt(G * Md / r_init[92] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1463(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1463};
  modelica_real tmp366;
  modelica_real tmp367;
  tmp366 = (data->simulationInfo->realParameter[256] /* r_init[92] PARAM */);
  tmp367 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp366 * tmp366 * tmp366),"r_init[92] ^ 3.0",equationIndexes);
  if(!(tmp367 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[92] ^ 3.0) was %g should be >= 0", tmp367);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[251]] /* vy[92] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[571]] /* x[92] STATE(1,vx[92]) */)) * (sqrt(tmp367));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3472(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3475(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3477(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3480(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3479(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3478(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3476(DATA *data, threadData_t *threadData);


/*
equation index: 1471
type: SIMPLE_ASSIGN
vz[92] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1471(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1471};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[411]] /* vz[92] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3471(DATA *data, threadData_t *threadData);


/*
equation index: 1473
type: SIMPLE_ASSIGN
z[93] = 0.65
*/
void WhirlpoolDiskStars_eqFunction_1473(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1473};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[892]] /* z[93] STATE(1,vz[93]) */) = 0.65;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3484(DATA *data, threadData_t *threadData);


/*
equation index: 1475
type: SIMPLE_ASSIGN
y[93] = r_init[93] * sin(theta[93] + armOffset[93])
*/
void WhirlpoolDiskStars_eqFunction_1475(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1475};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[732]] /* y[93] STATE(1,vy[93]) */) = ((data->simulationInfo->realParameter[257] /* r_init[93] PARAM */)) * (sin((data->simulationInfo->realParameter[417] /* theta[93] PARAM */) + (data->simulationInfo->realParameter[95] /* armOffset[93] PARAM */)));
  TRACE_POP
}

/*
equation index: 1476
type: SIMPLE_ASSIGN
vx[93] = (-y[93]) * sqrt(G * Md / r_init[93] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1476(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1476};
  modelica_real tmp368;
  modelica_real tmp369;
  tmp368 = (data->simulationInfo->realParameter[257] /* r_init[93] PARAM */);
  tmp369 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp368 * tmp368 * tmp368),"r_init[93] ^ 3.0",equationIndexes);
  if(!(tmp369 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[93] ^ 3.0) was %g should be >= 0", tmp369);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[92]] /* vx[93] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[732]] /* y[93] STATE(1,vy[93]) */))) * (sqrt(tmp369));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3483(DATA *data, threadData_t *threadData);


/*
equation index: 1478
type: SIMPLE_ASSIGN
x[93] = r_init[93] * cos(theta[93] + armOffset[93])
*/
void WhirlpoolDiskStars_eqFunction_1478(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1478};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[572]] /* x[93] STATE(1,vx[93]) */) = ((data->simulationInfo->realParameter[257] /* r_init[93] PARAM */)) * (cos((data->simulationInfo->realParameter[417] /* theta[93] PARAM */) + (data->simulationInfo->realParameter[95] /* armOffset[93] PARAM */)));
  TRACE_POP
}

/*
equation index: 1479
type: SIMPLE_ASSIGN
vy[93] = x[93] * sqrt(G * Md / r_init[93] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1479(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1479};
  modelica_real tmp370;
  modelica_real tmp371;
  tmp370 = (data->simulationInfo->realParameter[257] /* r_init[93] PARAM */);
  tmp371 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp370 * tmp370 * tmp370),"r_init[93] ^ 3.0",equationIndexes);
  if(!(tmp371 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[93] ^ 3.0) was %g should be >= 0", tmp371);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[252]] /* vy[93] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[572]] /* x[93] STATE(1,vx[93]) */)) * (sqrt(tmp371));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3482(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3485(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3487(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3490(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3489(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3488(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3486(DATA *data, threadData_t *threadData);


/*
equation index: 1487
type: SIMPLE_ASSIGN
vz[93] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1487(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1487};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[412]] /* vz[93] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3481(DATA *data, threadData_t *threadData);


/*
equation index: 1489
type: SIMPLE_ASSIGN
z[94] = 0.7000000000000001
*/
void WhirlpoolDiskStars_eqFunction_1489(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1489};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[893]] /* z[94] STATE(1,vz[94]) */) = 0.7000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3494(DATA *data, threadData_t *threadData);


/*
equation index: 1491
type: SIMPLE_ASSIGN
y[94] = r_init[94] * sin(theta[94] + armOffset[94])
*/
void WhirlpoolDiskStars_eqFunction_1491(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1491};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[733]] /* y[94] STATE(1,vy[94]) */) = ((data->simulationInfo->realParameter[258] /* r_init[94] PARAM */)) * (sin((data->simulationInfo->realParameter[418] /* theta[94] PARAM */) + (data->simulationInfo->realParameter[96] /* armOffset[94] PARAM */)));
  TRACE_POP
}

/*
equation index: 1492
type: SIMPLE_ASSIGN
vx[94] = (-y[94]) * sqrt(G * Md / r_init[94] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1492(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1492};
  modelica_real tmp372;
  modelica_real tmp373;
  tmp372 = (data->simulationInfo->realParameter[258] /* r_init[94] PARAM */);
  tmp373 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp372 * tmp372 * tmp372),"r_init[94] ^ 3.0",equationIndexes);
  if(!(tmp373 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[94] ^ 3.0) was %g should be >= 0", tmp373);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[93]] /* vx[94] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[733]] /* y[94] STATE(1,vy[94]) */))) * (sqrt(tmp373));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3493(DATA *data, threadData_t *threadData);


/*
equation index: 1494
type: SIMPLE_ASSIGN
x[94] = r_init[94] * cos(theta[94] + armOffset[94])
*/
void WhirlpoolDiskStars_eqFunction_1494(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1494};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[573]] /* x[94] STATE(1,vx[94]) */) = ((data->simulationInfo->realParameter[258] /* r_init[94] PARAM */)) * (cos((data->simulationInfo->realParameter[418] /* theta[94] PARAM */) + (data->simulationInfo->realParameter[96] /* armOffset[94] PARAM */)));
  TRACE_POP
}

/*
equation index: 1495
type: SIMPLE_ASSIGN
vy[94] = x[94] * sqrt(G * Md / r_init[94] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1495(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1495};
  modelica_real tmp374;
  modelica_real tmp375;
  tmp374 = (data->simulationInfo->realParameter[258] /* r_init[94] PARAM */);
  tmp375 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp374 * tmp374 * tmp374),"r_init[94] ^ 3.0",equationIndexes);
  if(!(tmp375 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[94] ^ 3.0) was %g should be >= 0", tmp375);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[253]] /* vy[94] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[573]] /* x[94] STATE(1,vx[94]) */)) * (sqrt(tmp375));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3492(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3495(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3497(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3500(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3499(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3498(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3496(DATA *data, threadData_t *threadData);


/*
equation index: 1503
type: SIMPLE_ASSIGN
vz[94] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1503(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1503};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[413]] /* vz[94] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3491(DATA *data, threadData_t *threadData);


/*
equation index: 1505
type: SIMPLE_ASSIGN
z[95] = 0.75
*/
void WhirlpoolDiskStars_eqFunction_1505(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1505};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[894]] /* z[95] STATE(1,vz[95]) */) = 0.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3504(DATA *data, threadData_t *threadData);


/*
equation index: 1507
type: SIMPLE_ASSIGN
y[95] = r_init[95] * sin(theta[95] + armOffset[95])
*/
void WhirlpoolDiskStars_eqFunction_1507(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1507};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[734]] /* y[95] STATE(1,vy[95]) */) = ((data->simulationInfo->realParameter[259] /* r_init[95] PARAM */)) * (sin((data->simulationInfo->realParameter[419] /* theta[95] PARAM */) + (data->simulationInfo->realParameter[97] /* armOffset[95] PARAM */)));
  TRACE_POP
}

/*
equation index: 1508
type: SIMPLE_ASSIGN
vx[95] = (-y[95]) * sqrt(G * Md / r_init[95] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1508(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1508};
  modelica_real tmp376;
  modelica_real tmp377;
  tmp376 = (data->simulationInfo->realParameter[259] /* r_init[95] PARAM */);
  tmp377 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp376 * tmp376 * tmp376),"r_init[95] ^ 3.0",equationIndexes);
  if(!(tmp377 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[95] ^ 3.0) was %g should be >= 0", tmp377);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[94]] /* vx[95] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[734]] /* y[95] STATE(1,vy[95]) */))) * (sqrt(tmp377));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3503(DATA *data, threadData_t *threadData);


/*
equation index: 1510
type: SIMPLE_ASSIGN
x[95] = r_init[95] * cos(theta[95] + armOffset[95])
*/
void WhirlpoolDiskStars_eqFunction_1510(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1510};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[574]] /* x[95] STATE(1,vx[95]) */) = ((data->simulationInfo->realParameter[259] /* r_init[95] PARAM */)) * (cos((data->simulationInfo->realParameter[419] /* theta[95] PARAM */) + (data->simulationInfo->realParameter[97] /* armOffset[95] PARAM */)));
  TRACE_POP
}

/*
equation index: 1511
type: SIMPLE_ASSIGN
vy[95] = x[95] * sqrt(G * Md / r_init[95] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1511(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1511};
  modelica_real tmp378;
  modelica_real tmp379;
  tmp378 = (data->simulationInfo->realParameter[259] /* r_init[95] PARAM */);
  tmp379 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp378 * tmp378 * tmp378),"r_init[95] ^ 3.0",equationIndexes);
  if(!(tmp379 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[95] ^ 3.0) was %g should be >= 0", tmp379);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[254]] /* vy[95] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[574]] /* x[95] STATE(1,vx[95]) */)) * (sqrt(tmp379));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3502(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3505(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3507(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3510(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3509(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3508(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3506(DATA *data, threadData_t *threadData);


/*
equation index: 1519
type: SIMPLE_ASSIGN
vz[95] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1519(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1519};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[414]] /* vz[95] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3501(DATA *data, threadData_t *threadData);


/*
equation index: 1521
type: SIMPLE_ASSIGN
z[96] = 0.8
*/
void WhirlpoolDiskStars_eqFunction_1521(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1521};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[895]] /* z[96] STATE(1,vz[96]) */) = 0.8;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3514(DATA *data, threadData_t *threadData);


/*
equation index: 1523
type: SIMPLE_ASSIGN
y[96] = r_init[96] * sin(theta[96] + armOffset[96])
*/
void WhirlpoolDiskStars_eqFunction_1523(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1523};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[735]] /* y[96] STATE(1,vy[96]) */) = ((data->simulationInfo->realParameter[260] /* r_init[96] PARAM */)) * (sin((data->simulationInfo->realParameter[420] /* theta[96] PARAM */) + (data->simulationInfo->realParameter[98] /* armOffset[96] PARAM */)));
  TRACE_POP
}

/*
equation index: 1524
type: SIMPLE_ASSIGN
vx[96] = (-y[96]) * sqrt(G * Md / r_init[96] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1524(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1524};
  modelica_real tmp380;
  modelica_real tmp381;
  tmp380 = (data->simulationInfo->realParameter[260] /* r_init[96] PARAM */);
  tmp381 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp380 * tmp380 * tmp380),"r_init[96] ^ 3.0",equationIndexes);
  if(!(tmp381 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[96] ^ 3.0) was %g should be >= 0", tmp381);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[95]] /* vx[96] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[735]] /* y[96] STATE(1,vy[96]) */))) * (sqrt(tmp381));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3513(DATA *data, threadData_t *threadData);


/*
equation index: 1526
type: SIMPLE_ASSIGN
x[96] = r_init[96] * cos(theta[96] + armOffset[96])
*/
void WhirlpoolDiskStars_eqFunction_1526(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1526};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[575]] /* x[96] STATE(1,vx[96]) */) = ((data->simulationInfo->realParameter[260] /* r_init[96] PARAM */)) * (cos((data->simulationInfo->realParameter[420] /* theta[96] PARAM */) + (data->simulationInfo->realParameter[98] /* armOffset[96] PARAM */)));
  TRACE_POP
}

/*
equation index: 1527
type: SIMPLE_ASSIGN
vy[96] = x[96] * sqrt(G * Md / r_init[96] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1527(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1527};
  modelica_real tmp382;
  modelica_real tmp383;
  tmp382 = (data->simulationInfo->realParameter[260] /* r_init[96] PARAM */);
  tmp383 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp382 * tmp382 * tmp382),"r_init[96] ^ 3.0",equationIndexes);
  if(!(tmp383 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[96] ^ 3.0) was %g should be >= 0", tmp383);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[255]] /* vy[96] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[575]] /* x[96] STATE(1,vx[96]) */)) * (sqrt(tmp383));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3512(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3515(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3517(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3520(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3519(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3518(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3516(DATA *data, threadData_t *threadData);


/*
equation index: 1535
type: SIMPLE_ASSIGN
vz[96] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1535(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1535};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[415]] /* vz[96] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3511(DATA *data, threadData_t *threadData);


/*
equation index: 1537
type: SIMPLE_ASSIGN
z[97] = 0.8500000000000001
*/
void WhirlpoolDiskStars_eqFunction_1537(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1537};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[896]] /* z[97] STATE(1,vz[97]) */) = 0.8500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3524(DATA *data, threadData_t *threadData);


/*
equation index: 1539
type: SIMPLE_ASSIGN
y[97] = r_init[97] * sin(theta[97] + armOffset[97])
*/
void WhirlpoolDiskStars_eqFunction_1539(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1539};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[736]] /* y[97] STATE(1,vy[97]) */) = ((data->simulationInfo->realParameter[261] /* r_init[97] PARAM */)) * (sin((data->simulationInfo->realParameter[421] /* theta[97] PARAM */) + (data->simulationInfo->realParameter[99] /* armOffset[97] PARAM */)));
  TRACE_POP
}

/*
equation index: 1540
type: SIMPLE_ASSIGN
vx[97] = (-y[97]) * sqrt(G * Md / r_init[97] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1540(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1540};
  modelica_real tmp384;
  modelica_real tmp385;
  tmp384 = (data->simulationInfo->realParameter[261] /* r_init[97] PARAM */);
  tmp385 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp384 * tmp384 * tmp384),"r_init[97] ^ 3.0",equationIndexes);
  if(!(tmp385 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[97] ^ 3.0) was %g should be >= 0", tmp385);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[96]] /* vx[97] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[736]] /* y[97] STATE(1,vy[97]) */))) * (sqrt(tmp385));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3523(DATA *data, threadData_t *threadData);


/*
equation index: 1542
type: SIMPLE_ASSIGN
x[97] = r_init[97] * cos(theta[97] + armOffset[97])
*/
void WhirlpoolDiskStars_eqFunction_1542(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1542};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[576]] /* x[97] STATE(1,vx[97]) */) = ((data->simulationInfo->realParameter[261] /* r_init[97] PARAM */)) * (cos((data->simulationInfo->realParameter[421] /* theta[97] PARAM */) + (data->simulationInfo->realParameter[99] /* armOffset[97] PARAM */)));
  TRACE_POP
}

/*
equation index: 1543
type: SIMPLE_ASSIGN
vy[97] = x[97] * sqrt(G * Md / r_init[97] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1543(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1543};
  modelica_real tmp386;
  modelica_real tmp387;
  tmp386 = (data->simulationInfo->realParameter[261] /* r_init[97] PARAM */);
  tmp387 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp386 * tmp386 * tmp386),"r_init[97] ^ 3.0",equationIndexes);
  if(!(tmp387 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[97] ^ 3.0) was %g should be >= 0", tmp387);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[256]] /* vy[97] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[576]] /* x[97] STATE(1,vx[97]) */)) * (sqrt(tmp387));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3522(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3525(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3527(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3530(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3529(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3528(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3526(DATA *data, threadData_t *threadData);


/*
equation index: 1551
type: SIMPLE_ASSIGN
vz[97] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1551(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1551};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[416]] /* vz[97] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3521(DATA *data, threadData_t *threadData);


/*
equation index: 1553
type: SIMPLE_ASSIGN
z[98] = 0.9
*/
void WhirlpoolDiskStars_eqFunction_1553(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1553};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[897]] /* z[98] STATE(1,vz[98]) */) = 0.9;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3534(DATA *data, threadData_t *threadData);


/*
equation index: 1555
type: SIMPLE_ASSIGN
y[98] = r_init[98] * sin(theta[98] + armOffset[98])
*/
void WhirlpoolDiskStars_eqFunction_1555(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1555};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[737]] /* y[98] STATE(1,vy[98]) */) = ((data->simulationInfo->realParameter[262] /* r_init[98] PARAM */)) * (sin((data->simulationInfo->realParameter[422] /* theta[98] PARAM */) + (data->simulationInfo->realParameter[100] /* armOffset[98] PARAM */)));
  TRACE_POP
}

/*
equation index: 1556
type: SIMPLE_ASSIGN
vx[98] = (-y[98]) * sqrt(G * Md / r_init[98] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1556(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1556};
  modelica_real tmp388;
  modelica_real tmp389;
  tmp388 = (data->simulationInfo->realParameter[262] /* r_init[98] PARAM */);
  tmp389 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp388 * tmp388 * tmp388),"r_init[98] ^ 3.0",equationIndexes);
  if(!(tmp389 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[98] ^ 3.0) was %g should be >= 0", tmp389);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[97]] /* vx[98] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[737]] /* y[98] STATE(1,vy[98]) */))) * (sqrt(tmp389));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3533(DATA *data, threadData_t *threadData);


/*
equation index: 1558
type: SIMPLE_ASSIGN
x[98] = r_init[98] * cos(theta[98] + armOffset[98])
*/
void WhirlpoolDiskStars_eqFunction_1558(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1558};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[577]] /* x[98] STATE(1,vx[98]) */) = ((data->simulationInfo->realParameter[262] /* r_init[98] PARAM */)) * (cos((data->simulationInfo->realParameter[422] /* theta[98] PARAM */) + (data->simulationInfo->realParameter[100] /* armOffset[98] PARAM */)));
  TRACE_POP
}

/*
equation index: 1559
type: SIMPLE_ASSIGN
vy[98] = x[98] * sqrt(G * Md / r_init[98] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1559(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1559};
  modelica_real tmp390;
  modelica_real tmp391;
  tmp390 = (data->simulationInfo->realParameter[262] /* r_init[98] PARAM */);
  tmp391 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp390 * tmp390 * tmp390),"r_init[98] ^ 3.0",equationIndexes);
  if(!(tmp391 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[98] ^ 3.0) was %g should be >= 0", tmp391);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[257]] /* vy[98] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[577]] /* x[98] STATE(1,vx[98]) */)) * (sqrt(tmp391));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3532(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3535(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3537(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3540(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3539(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3538(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3536(DATA *data, threadData_t *threadData);


/*
equation index: 1567
type: SIMPLE_ASSIGN
vz[98] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1567(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1567};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[417]] /* vz[98] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3531(DATA *data, threadData_t *threadData);


/*
equation index: 1569
type: SIMPLE_ASSIGN
z[99] = 0.9500000000000001
*/
void WhirlpoolDiskStars_eqFunction_1569(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1569};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[898]] /* z[99] STATE(1,vz[99]) */) = 0.9500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3544(DATA *data, threadData_t *threadData);


/*
equation index: 1571
type: SIMPLE_ASSIGN
y[99] = r_init[99] * sin(theta[99] + armOffset[99])
*/
void WhirlpoolDiskStars_eqFunction_1571(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1571};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[738]] /* y[99] STATE(1,vy[99]) */) = ((data->simulationInfo->realParameter[263] /* r_init[99] PARAM */)) * (sin((data->simulationInfo->realParameter[423] /* theta[99] PARAM */) + (data->simulationInfo->realParameter[101] /* armOffset[99] PARAM */)));
  TRACE_POP
}

/*
equation index: 1572
type: SIMPLE_ASSIGN
vx[99] = (-y[99]) * sqrt(G * Md / r_init[99] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1572(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1572};
  modelica_real tmp392;
  modelica_real tmp393;
  tmp392 = (data->simulationInfo->realParameter[263] /* r_init[99] PARAM */);
  tmp393 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp392 * tmp392 * tmp392),"r_init[99] ^ 3.0",equationIndexes);
  if(!(tmp393 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[99] ^ 3.0) was %g should be >= 0", tmp393);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[98]] /* vx[99] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[738]] /* y[99] STATE(1,vy[99]) */))) * (sqrt(tmp393));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3543(DATA *data, threadData_t *threadData);


/*
equation index: 1574
type: SIMPLE_ASSIGN
x[99] = r_init[99] * cos(theta[99] + armOffset[99])
*/
void WhirlpoolDiskStars_eqFunction_1574(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1574};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[578]] /* x[99] STATE(1,vx[99]) */) = ((data->simulationInfo->realParameter[263] /* r_init[99] PARAM */)) * (cos((data->simulationInfo->realParameter[423] /* theta[99] PARAM */) + (data->simulationInfo->realParameter[101] /* armOffset[99] PARAM */)));
  TRACE_POP
}

/*
equation index: 1575
type: SIMPLE_ASSIGN
vy[99] = x[99] * sqrt(G * Md / r_init[99] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1575(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1575};
  modelica_real tmp394;
  modelica_real tmp395;
  tmp394 = (data->simulationInfo->realParameter[263] /* r_init[99] PARAM */);
  tmp395 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp394 * tmp394 * tmp394),"r_init[99] ^ 3.0",equationIndexes);
  if(!(tmp395 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[99] ^ 3.0) was %g should be >= 0", tmp395);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[258]] /* vy[99] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[578]] /* x[99] STATE(1,vx[99]) */)) * (sqrt(tmp395));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3542(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3545(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3547(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3550(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3549(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3548(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3546(DATA *data, threadData_t *threadData);


/*
equation index: 1583
type: SIMPLE_ASSIGN
vz[99] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1583(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1583};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[418]] /* vz[99] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3541(DATA *data, threadData_t *threadData);


/*
equation index: 1585
type: SIMPLE_ASSIGN
z[100] = 1.0
*/
void WhirlpoolDiskStars_eqFunction_1585(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1585};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[899]] /* z[100] STATE(1,vz[100]) */) = 1.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3554(DATA *data, threadData_t *threadData);


/*
equation index: 1587
type: SIMPLE_ASSIGN
y[100] = r_init[100] * sin(theta[100] + armOffset[100])
*/
void WhirlpoolDiskStars_eqFunction_1587(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1587};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[739]] /* y[100] STATE(1,vy[100]) */) = ((data->simulationInfo->realParameter[264] /* r_init[100] PARAM */)) * (sin((data->simulationInfo->realParameter[424] /* theta[100] PARAM */) + (data->simulationInfo->realParameter[102] /* armOffset[100] PARAM */)));
  TRACE_POP
}

/*
equation index: 1588
type: SIMPLE_ASSIGN
vx[100] = (-y[100]) * sqrt(G * Md / r_init[100] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1588(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1588};
  modelica_real tmp396;
  modelica_real tmp397;
  tmp396 = (data->simulationInfo->realParameter[264] /* r_init[100] PARAM */);
  tmp397 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp396 * tmp396 * tmp396),"r_init[100] ^ 3.0",equationIndexes);
  if(!(tmp397 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[100] ^ 3.0) was %g should be >= 0", tmp397);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[99]] /* vx[100] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[739]] /* y[100] STATE(1,vy[100]) */))) * (sqrt(tmp397));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3553(DATA *data, threadData_t *threadData);


/*
equation index: 1590
type: SIMPLE_ASSIGN
x[100] = r_init[100] * cos(theta[100] + armOffset[100])
*/
void WhirlpoolDiskStars_eqFunction_1590(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1590};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[579]] /* x[100] STATE(1,vx[100]) */) = ((data->simulationInfo->realParameter[264] /* r_init[100] PARAM */)) * (cos((data->simulationInfo->realParameter[424] /* theta[100] PARAM */) + (data->simulationInfo->realParameter[102] /* armOffset[100] PARAM */)));
  TRACE_POP
}

/*
equation index: 1591
type: SIMPLE_ASSIGN
vy[100] = x[100] * sqrt(G * Md / r_init[100] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1591(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1591};
  modelica_real tmp398;
  modelica_real tmp399;
  tmp398 = (data->simulationInfo->realParameter[264] /* r_init[100] PARAM */);
  tmp399 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp398 * tmp398 * tmp398),"r_init[100] ^ 3.0",equationIndexes);
  if(!(tmp399 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[100] ^ 3.0) was %g should be >= 0", tmp399);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[259]] /* vy[100] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[579]] /* x[100] STATE(1,vx[100]) */)) * (sqrt(tmp399));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3552(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3555(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3557(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3560(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3559(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3558(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3556(DATA *data, threadData_t *threadData);


/*
equation index: 1599
type: SIMPLE_ASSIGN
vz[100] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1599(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1599};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[419]] /* vz[100] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3551(DATA *data, threadData_t *threadData);


/*
equation index: 1601
type: SIMPLE_ASSIGN
z[101] = 1.05
*/
void WhirlpoolDiskStars_eqFunction_1601(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1601};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[900]] /* z[101] STATE(1,vz[101]) */) = 1.05;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3564(DATA *data, threadData_t *threadData);


/*
equation index: 1603
type: SIMPLE_ASSIGN
y[101] = r_init[101] * sin(theta[101] + armOffset[101])
*/
void WhirlpoolDiskStars_eqFunction_1603(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1603};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[740]] /* y[101] STATE(1,vy[101]) */) = ((data->simulationInfo->realParameter[265] /* r_init[101] PARAM */)) * (sin((data->simulationInfo->realParameter[425] /* theta[101] PARAM */) + (data->simulationInfo->realParameter[103] /* armOffset[101] PARAM */)));
  TRACE_POP
}

/*
equation index: 1604
type: SIMPLE_ASSIGN
vx[101] = (-y[101]) * sqrt(G * Md / r_init[101] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1604(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1604};
  modelica_real tmp400;
  modelica_real tmp401;
  tmp400 = (data->simulationInfo->realParameter[265] /* r_init[101] PARAM */);
  tmp401 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp400 * tmp400 * tmp400),"r_init[101] ^ 3.0",equationIndexes);
  if(!(tmp401 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[101] ^ 3.0) was %g should be >= 0", tmp401);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[100]] /* vx[101] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[740]] /* y[101] STATE(1,vy[101]) */))) * (sqrt(tmp401));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3563(DATA *data, threadData_t *threadData);


/*
equation index: 1606
type: SIMPLE_ASSIGN
x[101] = r_init[101] * cos(theta[101] + armOffset[101])
*/
void WhirlpoolDiskStars_eqFunction_1606(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1606};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[580]] /* x[101] STATE(1,vx[101]) */) = ((data->simulationInfo->realParameter[265] /* r_init[101] PARAM */)) * (cos((data->simulationInfo->realParameter[425] /* theta[101] PARAM */) + (data->simulationInfo->realParameter[103] /* armOffset[101] PARAM */)));
  TRACE_POP
}

/*
equation index: 1607
type: SIMPLE_ASSIGN
vy[101] = x[101] * sqrt(G * Md / r_init[101] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1607(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1607};
  modelica_real tmp402;
  modelica_real tmp403;
  tmp402 = (data->simulationInfo->realParameter[265] /* r_init[101] PARAM */);
  tmp403 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp402 * tmp402 * tmp402),"r_init[101] ^ 3.0",equationIndexes);
  if(!(tmp403 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[101] ^ 3.0) was %g should be >= 0", tmp403);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[260]] /* vy[101] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[580]] /* x[101] STATE(1,vx[101]) */)) * (sqrt(tmp403));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3562(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3565(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3567(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3570(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3569(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3568(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3566(DATA *data, threadData_t *threadData);


/*
equation index: 1615
type: SIMPLE_ASSIGN
vz[101] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1615(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1615};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[420]] /* vz[101] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3561(DATA *data, threadData_t *threadData);


/*
equation index: 1617
type: SIMPLE_ASSIGN
z[102] = 1.1
*/
void WhirlpoolDiskStars_eqFunction_1617(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1617};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[901]] /* z[102] STATE(1,vz[102]) */) = 1.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3574(DATA *data, threadData_t *threadData);


/*
equation index: 1619
type: SIMPLE_ASSIGN
y[102] = r_init[102] * sin(theta[102] + armOffset[102])
*/
void WhirlpoolDiskStars_eqFunction_1619(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1619};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[741]] /* y[102] STATE(1,vy[102]) */) = ((data->simulationInfo->realParameter[266] /* r_init[102] PARAM */)) * (sin((data->simulationInfo->realParameter[426] /* theta[102] PARAM */) + (data->simulationInfo->realParameter[104] /* armOffset[102] PARAM */)));
  TRACE_POP
}

/*
equation index: 1620
type: SIMPLE_ASSIGN
vx[102] = (-y[102]) * sqrt(G * Md / r_init[102] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1620(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1620};
  modelica_real tmp404;
  modelica_real tmp405;
  tmp404 = (data->simulationInfo->realParameter[266] /* r_init[102] PARAM */);
  tmp405 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp404 * tmp404 * tmp404),"r_init[102] ^ 3.0",equationIndexes);
  if(!(tmp405 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[102] ^ 3.0) was %g should be >= 0", tmp405);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[101]] /* vx[102] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[741]] /* y[102] STATE(1,vy[102]) */))) * (sqrt(tmp405));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3573(DATA *data, threadData_t *threadData);


/*
equation index: 1622
type: SIMPLE_ASSIGN
x[102] = r_init[102] * cos(theta[102] + armOffset[102])
*/
void WhirlpoolDiskStars_eqFunction_1622(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1622};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[581]] /* x[102] STATE(1,vx[102]) */) = ((data->simulationInfo->realParameter[266] /* r_init[102] PARAM */)) * (cos((data->simulationInfo->realParameter[426] /* theta[102] PARAM */) + (data->simulationInfo->realParameter[104] /* armOffset[102] PARAM */)));
  TRACE_POP
}

/*
equation index: 1623
type: SIMPLE_ASSIGN
vy[102] = x[102] * sqrt(G * Md / r_init[102] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1623(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1623};
  modelica_real tmp406;
  modelica_real tmp407;
  tmp406 = (data->simulationInfo->realParameter[266] /* r_init[102] PARAM */);
  tmp407 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp406 * tmp406 * tmp406),"r_init[102] ^ 3.0",equationIndexes);
  if(!(tmp407 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[102] ^ 3.0) was %g should be >= 0", tmp407);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[261]] /* vy[102] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[581]] /* x[102] STATE(1,vx[102]) */)) * (sqrt(tmp407));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3572(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3575(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3577(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3580(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3579(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3578(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3576(DATA *data, threadData_t *threadData);


/*
equation index: 1631
type: SIMPLE_ASSIGN
vz[102] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1631(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1631};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[421]] /* vz[102] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3571(DATA *data, threadData_t *threadData);


/*
equation index: 1633
type: SIMPLE_ASSIGN
z[103] = 1.1500000000000001
*/
void WhirlpoolDiskStars_eqFunction_1633(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1633};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[902]] /* z[103] STATE(1,vz[103]) */) = 1.1500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3584(DATA *data, threadData_t *threadData);


/*
equation index: 1635
type: SIMPLE_ASSIGN
y[103] = r_init[103] * sin(theta[103] + armOffset[103])
*/
void WhirlpoolDiskStars_eqFunction_1635(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1635};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[742]] /* y[103] STATE(1,vy[103]) */) = ((data->simulationInfo->realParameter[267] /* r_init[103] PARAM */)) * (sin((data->simulationInfo->realParameter[427] /* theta[103] PARAM */) + (data->simulationInfo->realParameter[105] /* armOffset[103] PARAM */)));
  TRACE_POP
}

/*
equation index: 1636
type: SIMPLE_ASSIGN
vx[103] = (-y[103]) * sqrt(G * Md / r_init[103] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1636(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1636};
  modelica_real tmp408;
  modelica_real tmp409;
  tmp408 = (data->simulationInfo->realParameter[267] /* r_init[103] PARAM */);
  tmp409 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp408 * tmp408 * tmp408),"r_init[103] ^ 3.0",equationIndexes);
  if(!(tmp409 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[103] ^ 3.0) was %g should be >= 0", tmp409);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[102]] /* vx[103] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[742]] /* y[103] STATE(1,vy[103]) */))) * (sqrt(tmp409));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3583(DATA *data, threadData_t *threadData);


/*
equation index: 1638
type: SIMPLE_ASSIGN
x[103] = r_init[103] * cos(theta[103] + armOffset[103])
*/
void WhirlpoolDiskStars_eqFunction_1638(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1638};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[582]] /* x[103] STATE(1,vx[103]) */) = ((data->simulationInfo->realParameter[267] /* r_init[103] PARAM */)) * (cos((data->simulationInfo->realParameter[427] /* theta[103] PARAM */) + (data->simulationInfo->realParameter[105] /* armOffset[103] PARAM */)));
  TRACE_POP
}

/*
equation index: 1639
type: SIMPLE_ASSIGN
vy[103] = x[103] * sqrt(G * Md / r_init[103] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1639(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1639};
  modelica_real tmp410;
  modelica_real tmp411;
  tmp410 = (data->simulationInfo->realParameter[267] /* r_init[103] PARAM */);
  tmp411 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp410 * tmp410 * tmp410),"r_init[103] ^ 3.0",equationIndexes);
  if(!(tmp411 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[103] ^ 3.0) was %g should be >= 0", tmp411);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[262]] /* vy[103] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[582]] /* x[103] STATE(1,vx[103]) */)) * (sqrt(tmp411));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3582(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3585(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3587(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3590(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3589(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3588(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3586(DATA *data, threadData_t *threadData);


/*
equation index: 1647
type: SIMPLE_ASSIGN
vz[103] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1647(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1647};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[422]] /* vz[103] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3581(DATA *data, threadData_t *threadData);


/*
equation index: 1649
type: SIMPLE_ASSIGN
z[104] = 1.2000000000000002
*/
void WhirlpoolDiskStars_eqFunction_1649(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1649};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[903]] /* z[104] STATE(1,vz[104]) */) = 1.2000000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3594(DATA *data, threadData_t *threadData);


/*
equation index: 1651
type: SIMPLE_ASSIGN
y[104] = r_init[104] * sin(theta[104] + armOffset[104])
*/
void WhirlpoolDiskStars_eqFunction_1651(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1651};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[743]] /* y[104] STATE(1,vy[104]) */) = ((data->simulationInfo->realParameter[268] /* r_init[104] PARAM */)) * (sin((data->simulationInfo->realParameter[428] /* theta[104] PARAM */) + (data->simulationInfo->realParameter[106] /* armOffset[104] PARAM */)));
  TRACE_POP
}

/*
equation index: 1652
type: SIMPLE_ASSIGN
vx[104] = (-y[104]) * sqrt(G * Md / r_init[104] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1652(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1652};
  modelica_real tmp412;
  modelica_real tmp413;
  tmp412 = (data->simulationInfo->realParameter[268] /* r_init[104] PARAM */);
  tmp413 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp412 * tmp412 * tmp412),"r_init[104] ^ 3.0",equationIndexes);
  if(!(tmp413 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[104] ^ 3.0) was %g should be >= 0", tmp413);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[103]] /* vx[104] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[743]] /* y[104] STATE(1,vy[104]) */))) * (sqrt(tmp413));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3593(DATA *data, threadData_t *threadData);


/*
equation index: 1654
type: SIMPLE_ASSIGN
x[104] = r_init[104] * cos(theta[104] + armOffset[104])
*/
void WhirlpoolDiskStars_eqFunction_1654(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1654};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[583]] /* x[104] STATE(1,vx[104]) */) = ((data->simulationInfo->realParameter[268] /* r_init[104] PARAM */)) * (cos((data->simulationInfo->realParameter[428] /* theta[104] PARAM */) + (data->simulationInfo->realParameter[106] /* armOffset[104] PARAM */)));
  TRACE_POP
}

/*
equation index: 1655
type: SIMPLE_ASSIGN
vy[104] = x[104] * sqrt(G * Md / r_init[104] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1655(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1655};
  modelica_real tmp414;
  modelica_real tmp415;
  tmp414 = (data->simulationInfo->realParameter[268] /* r_init[104] PARAM */);
  tmp415 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp414 * tmp414 * tmp414),"r_init[104] ^ 3.0",equationIndexes);
  if(!(tmp415 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[104] ^ 3.0) was %g should be >= 0", tmp415);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[263]] /* vy[104] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[583]] /* x[104] STATE(1,vx[104]) */)) * (sqrt(tmp415));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3592(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3595(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3597(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3600(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3599(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3598(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3596(DATA *data, threadData_t *threadData);


/*
equation index: 1663
type: SIMPLE_ASSIGN
vz[104] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1663(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1663};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[423]] /* vz[104] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3591(DATA *data, threadData_t *threadData);


/*
equation index: 1665
type: SIMPLE_ASSIGN
z[105] = 1.25
*/
void WhirlpoolDiskStars_eqFunction_1665(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1665};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[904]] /* z[105] STATE(1,vz[105]) */) = 1.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3604(DATA *data, threadData_t *threadData);


/*
equation index: 1667
type: SIMPLE_ASSIGN
y[105] = r_init[105] * sin(theta[105] + armOffset[105])
*/
void WhirlpoolDiskStars_eqFunction_1667(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1667};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[744]] /* y[105] STATE(1,vy[105]) */) = ((data->simulationInfo->realParameter[269] /* r_init[105] PARAM */)) * (sin((data->simulationInfo->realParameter[429] /* theta[105] PARAM */) + (data->simulationInfo->realParameter[107] /* armOffset[105] PARAM */)));
  TRACE_POP
}

/*
equation index: 1668
type: SIMPLE_ASSIGN
vx[105] = (-y[105]) * sqrt(G * Md / r_init[105] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1668(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1668};
  modelica_real tmp416;
  modelica_real tmp417;
  tmp416 = (data->simulationInfo->realParameter[269] /* r_init[105] PARAM */);
  tmp417 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp416 * tmp416 * tmp416),"r_init[105] ^ 3.0",equationIndexes);
  if(!(tmp417 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[105] ^ 3.0) was %g should be >= 0", tmp417);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[104]] /* vx[105] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[744]] /* y[105] STATE(1,vy[105]) */))) * (sqrt(tmp417));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3603(DATA *data, threadData_t *threadData);


/*
equation index: 1670
type: SIMPLE_ASSIGN
x[105] = r_init[105] * cos(theta[105] + armOffset[105])
*/
void WhirlpoolDiskStars_eqFunction_1670(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1670};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[584]] /* x[105] STATE(1,vx[105]) */) = ((data->simulationInfo->realParameter[269] /* r_init[105] PARAM */)) * (cos((data->simulationInfo->realParameter[429] /* theta[105] PARAM */) + (data->simulationInfo->realParameter[107] /* armOffset[105] PARAM */)));
  TRACE_POP
}

/*
equation index: 1671
type: SIMPLE_ASSIGN
vy[105] = x[105] * sqrt(G * Md / r_init[105] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1671(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1671};
  modelica_real tmp418;
  modelica_real tmp419;
  tmp418 = (data->simulationInfo->realParameter[269] /* r_init[105] PARAM */);
  tmp419 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp418 * tmp418 * tmp418),"r_init[105] ^ 3.0",equationIndexes);
  if(!(tmp419 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[105] ^ 3.0) was %g should be >= 0", tmp419);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[264]] /* vy[105] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[584]] /* x[105] STATE(1,vx[105]) */)) * (sqrt(tmp419));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3602(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3605(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3607(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3610(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3609(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3608(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3606(DATA *data, threadData_t *threadData);


/*
equation index: 1679
type: SIMPLE_ASSIGN
vz[105] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1679(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1679};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[424]] /* vz[105] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3601(DATA *data, threadData_t *threadData);


/*
equation index: 1681
type: SIMPLE_ASSIGN
z[106] = 1.3
*/
void WhirlpoolDiskStars_eqFunction_1681(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1681};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[905]] /* z[106] STATE(1,vz[106]) */) = 1.3;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3614(DATA *data, threadData_t *threadData);


/*
equation index: 1683
type: SIMPLE_ASSIGN
y[106] = r_init[106] * sin(theta[106] + armOffset[106])
*/
void WhirlpoolDiskStars_eqFunction_1683(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1683};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[745]] /* y[106] STATE(1,vy[106]) */) = ((data->simulationInfo->realParameter[270] /* r_init[106] PARAM */)) * (sin((data->simulationInfo->realParameter[430] /* theta[106] PARAM */) + (data->simulationInfo->realParameter[108] /* armOffset[106] PARAM */)));
  TRACE_POP
}

/*
equation index: 1684
type: SIMPLE_ASSIGN
vx[106] = (-y[106]) * sqrt(G * Md / r_init[106] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1684(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1684};
  modelica_real tmp420;
  modelica_real tmp421;
  tmp420 = (data->simulationInfo->realParameter[270] /* r_init[106] PARAM */);
  tmp421 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp420 * tmp420 * tmp420),"r_init[106] ^ 3.0",equationIndexes);
  if(!(tmp421 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[106] ^ 3.0) was %g should be >= 0", tmp421);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[105]] /* vx[106] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[745]] /* y[106] STATE(1,vy[106]) */))) * (sqrt(tmp421));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3613(DATA *data, threadData_t *threadData);


/*
equation index: 1686
type: SIMPLE_ASSIGN
x[106] = r_init[106] * cos(theta[106] + armOffset[106])
*/
void WhirlpoolDiskStars_eqFunction_1686(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1686};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[585]] /* x[106] STATE(1,vx[106]) */) = ((data->simulationInfo->realParameter[270] /* r_init[106] PARAM */)) * (cos((data->simulationInfo->realParameter[430] /* theta[106] PARAM */) + (data->simulationInfo->realParameter[108] /* armOffset[106] PARAM */)));
  TRACE_POP
}

/*
equation index: 1687
type: SIMPLE_ASSIGN
vy[106] = x[106] * sqrt(G * Md / r_init[106] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1687(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1687};
  modelica_real tmp422;
  modelica_real tmp423;
  tmp422 = (data->simulationInfo->realParameter[270] /* r_init[106] PARAM */);
  tmp423 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp422 * tmp422 * tmp422),"r_init[106] ^ 3.0",equationIndexes);
  if(!(tmp423 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[106] ^ 3.0) was %g should be >= 0", tmp423);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[265]] /* vy[106] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[585]] /* x[106] STATE(1,vx[106]) */)) * (sqrt(tmp423));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3612(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3615(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3617(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3620(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3619(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3618(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3616(DATA *data, threadData_t *threadData);


/*
equation index: 1695
type: SIMPLE_ASSIGN
vz[106] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1695(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1695};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[425]] /* vz[106] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3611(DATA *data, threadData_t *threadData);


/*
equation index: 1697
type: SIMPLE_ASSIGN
z[107] = 1.35
*/
void WhirlpoolDiskStars_eqFunction_1697(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1697};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[906]] /* z[107] STATE(1,vz[107]) */) = 1.35;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3624(DATA *data, threadData_t *threadData);


/*
equation index: 1699
type: SIMPLE_ASSIGN
y[107] = r_init[107] * sin(theta[107] + armOffset[107])
*/
void WhirlpoolDiskStars_eqFunction_1699(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1699};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[746]] /* y[107] STATE(1,vy[107]) */) = ((data->simulationInfo->realParameter[271] /* r_init[107] PARAM */)) * (sin((data->simulationInfo->realParameter[431] /* theta[107] PARAM */) + (data->simulationInfo->realParameter[109] /* armOffset[107] PARAM */)));
  TRACE_POP
}

/*
equation index: 1700
type: SIMPLE_ASSIGN
vx[107] = (-y[107]) * sqrt(G * Md / r_init[107] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1700(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1700};
  modelica_real tmp424;
  modelica_real tmp425;
  tmp424 = (data->simulationInfo->realParameter[271] /* r_init[107] PARAM */);
  tmp425 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp424 * tmp424 * tmp424),"r_init[107] ^ 3.0",equationIndexes);
  if(!(tmp425 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[107] ^ 3.0) was %g should be >= 0", tmp425);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[106]] /* vx[107] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[746]] /* y[107] STATE(1,vy[107]) */))) * (sqrt(tmp425));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3623(DATA *data, threadData_t *threadData);


/*
equation index: 1702
type: SIMPLE_ASSIGN
x[107] = r_init[107] * cos(theta[107] + armOffset[107])
*/
void WhirlpoolDiskStars_eqFunction_1702(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1702};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[586]] /* x[107] STATE(1,vx[107]) */) = ((data->simulationInfo->realParameter[271] /* r_init[107] PARAM */)) * (cos((data->simulationInfo->realParameter[431] /* theta[107] PARAM */) + (data->simulationInfo->realParameter[109] /* armOffset[107] PARAM */)));
  TRACE_POP
}

/*
equation index: 1703
type: SIMPLE_ASSIGN
vy[107] = x[107] * sqrt(G * Md / r_init[107] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1703(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1703};
  modelica_real tmp426;
  modelica_real tmp427;
  tmp426 = (data->simulationInfo->realParameter[271] /* r_init[107] PARAM */);
  tmp427 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp426 * tmp426 * tmp426),"r_init[107] ^ 3.0",equationIndexes);
  if(!(tmp427 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[107] ^ 3.0) was %g should be >= 0", tmp427);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[266]] /* vy[107] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[586]] /* x[107] STATE(1,vx[107]) */)) * (sqrt(tmp427));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3622(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3625(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3627(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3630(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3629(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void WhirlpoolDiskStars_functionInitialEquations_3(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_3364(data, threadData);
  WhirlpoolDiskStars_eqFunction_1283(data, threadData);
  WhirlpoolDiskStars_eqFunction_1284(data, threadData);
  WhirlpoolDiskStars_eqFunction_3363(data, threadData);
  WhirlpoolDiskStars_eqFunction_1286(data, threadData);
  WhirlpoolDiskStars_eqFunction_1287(data, threadData);
  WhirlpoolDiskStars_eqFunction_3362(data, threadData);
  WhirlpoolDiskStars_eqFunction_3365(data, threadData);
  WhirlpoolDiskStars_eqFunction_3367(data, threadData);
  WhirlpoolDiskStars_eqFunction_3370(data, threadData);
  WhirlpoolDiskStars_eqFunction_3369(data, threadData);
  WhirlpoolDiskStars_eqFunction_3368(data, threadData);
  WhirlpoolDiskStars_eqFunction_3366(data, threadData);
  WhirlpoolDiskStars_eqFunction_1295(data, threadData);
  WhirlpoolDiskStars_eqFunction_3361(data, threadData);
  WhirlpoolDiskStars_eqFunction_1297(data, threadData);
  WhirlpoolDiskStars_eqFunction_3374(data, threadData);
  WhirlpoolDiskStars_eqFunction_1299(data, threadData);
  WhirlpoolDiskStars_eqFunction_1300(data, threadData);
  WhirlpoolDiskStars_eqFunction_3373(data, threadData);
  WhirlpoolDiskStars_eqFunction_1302(data, threadData);
  WhirlpoolDiskStars_eqFunction_1303(data, threadData);
  WhirlpoolDiskStars_eqFunction_3372(data, threadData);
  WhirlpoolDiskStars_eqFunction_3375(data, threadData);
  WhirlpoolDiskStars_eqFunction_3377(data, threadData);
  WhirlpoolDiskStars_eqFunction_3380(data, threadData);
  WhirlpoolDiskStars_eqFunction_3379(data, threadData);
  WhirlpoolDiskStars_eqFunction_3378(data, threadData);
  WhirlpoolDiskStars_eqFunction_3376(data, threadData);
  WhirlpoolDiskStars_eqFunction_1311(data, threadData);
  WhirlpoolDiskStars_eqFunction_3371(data, threadData);
  WhirlpoolDiskStars_eqFunction_1313(data, threadData);
  WhirlpoolDiskStars_eqFunction_3384(data, threadData);
  WhirlpoolDiskStars_eqFunction_1315(data, threadData);
  WhirlpoolDiskStars_eqFunction_1316(data, threadData);
  WhirlpoolDiskStars_eqFunction_3383(data, threadData);
  WhirlpoolDiskStars_eqFunction_1318(data, threadData);
  WhirlpoolDiskStars_eqFunction_1319(data, threadData);
  WhirlpoolDiskStars_eqFunction_3382(data, threadData);
  WhirlpoolDiskStars_eqFunction_3385(data, threadData);
  WhirlpoolDiskStars_eqFunction_3387(data, threadData);
  WhirlpoolDiskStars_eqFunction_3390(data, threadData);
  WhirlpoolDiskStars_eqFunction_3389(data, threadData);
  WhirlpoolDiskStars_eqFunction_3388(data, threadData);
  WhirlpoolDiskStars_eqFunction_3386(data, threadData);
  WhirlpoolDiskStars_eqFunction_1327(data, threadData);
  WhirlpoolDiskStars_eqFunction_3381(data, threadData);
  WhirlpoolDiskStars_eqFunction_1329(data, threadData);
  WhirlpoolDiskStars_eqFunction_3394(data, threadData);
  WhirlpoolDiskStars_eqFunction_1331(data, threadData);
  WhirlpoolDiskStars_eqFunction_1332(data, threadData);
  WhirlpoolDiskStars_eqFunction_3393(data, threadData);
  WhirlpoolDiskStars_eqFunction_1334(data, threadData);
  WhirlpoolDiskStars_eqFunction_1335(data, threadData);
  WhirlpoolDiskStars_eqFunction_3392(data, threadData);
  WhirlpoolDiskStars_eqFunction_3395(data, threadData);
  WhirlpoolDiskStars_eqFunction_3397(data, threadData);
  WhirlpoolDiskStars_eqFunction_3400(data, threadData);
  WhirlpoolDiskStars_eqFunction_3399(data, threadData);
  WhirlpoolDiskStars_eqFunction_3398(data, threadData);
  WhirlpoolDiskStars_eqFunction_3396(data, threadData);
  WhirlpoolDiskStars_eqFunction_1343(data, threadData);
  WhirlpoolDiskStars_eqFunction_3391(data, threadData);
  WhirlpoolDiskStars_eqFunction_1345(data, threadData);
  WhirlpoolDiskStars_eqFunction_3404(data, threadData);
  WhirlpoolDiskStars_eqFunction_1347(data, threadData);
  WhirlpoolDiskStars_eqFunction_1348(data, threadData);
  WhirlpoolDiskStars_eqFunction_3403(data, threadData);
  WhirlpoolDiskStars_eqFunction_1350(data, threadData);
  WhirlpoolDiskStars_eqFunction_1351(data, threadData);
  WhirlpoolDiskStars_eqFunction_3402(data, threadData);
  WhirlpoolDiskStars_eqFunction_3405(data, threadData);
  WhirlpoolDiskStars_eqFunction_3407(data, threadData);
  WhirlpoolDiskStars_eqFunction_3410(data, threadData);
  WhirlpoolDiskStars_eqFunction_3409(data, threadData);
  WhirlpoolDiskStars_eqFunction_3408(data, threadData);
  WhirlpoolDiskStars_eqFunction_3406(data, threadData);
  WhirlpoolDiskStars_eqFunction_1359(data, threadData);
  WhirlpoolDiskStars_eqFunction_3401(data, threadData);
  WhirlpoolDiskStars_eqFunction_1361(data, threadData);
  WhirlpoolDiskStars_eqFunction_3414(data, threadData);
  WhirlpoolDiskStars_eqFunction_1363(data, threadData);
  WhirlpoolDiskStars_eqFunction_1364(data, threadData);
  WhirlpoolDiskStars_eqFunction_3413(data, threadData);
  WhirlpoolDiskStars_eqFunction_1366(data, threadData);
  WhirlpoolDiskStars_eqFunction_1367(data, threadData);
  WhirlpoolDiskStars_eqFunction_3412(data, threadData);
  WhirlpoolDiskStars_eqFunction_3415(data, threadData);
  WhirlpoolDiskStars_eqFunction_3417(data, threadData);
  WhirlpoolDiskStars_eqFunction_3420(data, threadData);
  WhirlpoolDiskStars_eqFunction_3419(data, threadData);
  WhirlpoolDiskStars_eqFunction_3418(data, threadData);
  WhirlpoolDiskStars_eqFunction_3416(data, threadData);
  WhirlpoolDiskStars_eqFunction_1375(data, threadData);
  WhirlpoolDiskStars_eqFunction_3411(data, threadData);
  WhirlpoolDiskStars_eqFunction_1377(data, threadData);
  WhirlpoolDiskStars_eqFunction_3424(data, threadData);
  WhirlpoolDiskStars_eqFunction_1379(data, threadData);
  WhirlpoolDiskStars_eqFunction_1380(data, threadData);
  WhirlpoolDiskStars_eqFunction_3423(data, threadData);
  WhirlpoolDiskStars_eqFunction_1382(data, threadData);
  WhirlpoolDiskStars_eqFunction_1383(data, threadData);
  WhirlpoolDiskStars_eqFunction_3422(data, threadData);
  WhirlpoolDiskStars_eqFunction_3425(data, threadData);
  WhirlpoolDiskStars_eqFunction_3427(data, threadData);
  WhirlpoolDiskStars_eqFunction_3430(data, threadData);
  WhirlpoolDiskStars_eqFunction_3429(data, threadData);
  WhirlpoolDiskStars_eqFunction_3428(data, threadData);
  WhirlpoolDiskStars_eqFunction_3426(data, threadData);
  WhirlpoolDiskStars_eqFunction_1391(data, threadData);
  WhirlpoolDiskStars_eqFunction_3421(data, threadData);
  WhirlpoolDiskStars_eqFunction_1393(data, threadData);
  WhirlpoolDiskStars_eqFunction_3434(data, threadData);
  WhirlpoolDiskStars_eqFunction_1395(data, threadData);
  WhirlpoolDiskStars_eqFunction_1396(data, threadData);
  WhirlpoolDiskStars_eqFunction_3433(data, threadData);
  WhirlpoolDiskStars_eqFunction_1398(data, threadData);
  WhirlpoolDiskStars_eqFunction_1399(data, threadData);
  WhirlpoolDiskStars_eqFunction_3432(data, threadData);
  WhirlpoolDiskStars_eqFunction_3435(data, threadData);
  WhirlpoolDiskStars_eqFunction_3437(data, threadData);
  WhirlpoolDiskStars_eqFunction_3440(data, threadData);
  WhirlpoolDiskStars_eqFunction_3439(data, threadData);
  WhirlpoolDiskStars_eqFunction_3438(data, threadData);
  WhirlpoolDiskStars_eqFunction_3436(data, threadData);
  WhirlpoolDiskStars_eqFunction_1407(data, threadData);
  WhirlpoolDiskStars_eqFunction_3431(data, threadData);
  WhirlpoolDiskStars_eqFunction_1409(data, threadData);
  WhirlpoolDiskStars_eqFunction_3444(data, threadData);
  WhirlpoolDiskStars_eqFunction_1411(data, threadData);
  WhirlpoolDiskStars_eqFunction_1412(data, threadData);
  WhirlpoolDiskStars_eqFunction_3443(data, threadData);
  WhirlpoolDiskStars_eqFunction_1414(data, threadData);
  WhirlpoolDiskStars_eqFunction_1415(data, threadData);
  WhirlpoolDiskStars_eqFunction_3442(data, threadData);
  WhirlpoolDiskStars_eqFunction_3445(data, threadData);
  WhirlpoolDiskStars_eqFunction_3447(data, threadData);
  WhirlpoolDiskStars_eqFunction_3450(data, threadData);
  WhirlpoolDiskStars_eqFunction_3449(data, threadData);
  WhirlpoolDiskStars_eqFunction_3448(data, threadData);
  WhirlpoolDiskStars_eqFunction_3446(data, threadData);
  WhirlpoolDiskStars_eqFunction_1423(data, threadData);
  WhirlpoolDiskStars_eqFunction_3441(data, threadData);
  WhirlpoolDiskStars_eqFunction_1425(data, threadData);
  WhirlpoolDiskStars_eqFunction_3454(data, threadData);
  WhirlpoolDiskStars_eqFunction_1427(data, threadData);
  WhirlpoolDiskStars_eqFunction_1428(data, threadData);
  WhirlpoolDiskStars_eqFunction_3453(data, threadData);
  WhirlpoolDiskStars_eqFunction_1430(data, threadData);
  WhirlpoolDiskStars_eqFunction_1431(data, threadData);
  WhirlpoolDiskStars_eqFunction_3452(data, threadData);
  WhirlpoolDiskStars_eqFunction_3455(data, threadData);
  WhirlpoolDiskStars_eqFunction_3457(data, threadData);
  WhirlpoolDiskStars_eqFunction_3460(data, threadData);
  WhirlpoolDiskStars_eqFunction_3459(data, threadData);
  WhirlpoolDiskStars_eqFunction_3458(data, threadData);
  WhirlpoolDiskStars_eqFunction_3456(data, threadData);
  WhirlpoolDiskStars_eqFunction_1439(data, threadData);
  WhirlpoolDiskStars_eqFunction_3451(data, threadData);
  WhirlpoolDiskStars_eqFunction_1441(data, threadData);
  WhirlpoolDiskStars_eqFunction_3464(data, threadData);
  WhirlpoolDiskStars_eqFunction_1443(data, threadData);
  WhirlpoolDiskStars_eqFunction_1444(data, threadData);
  WhirlpoolDiskStars_eqFunction_3463(data, threadData);
  WhirlpoolDiskStars_eqFunction_1446(data, threadData);
  WhirlpoolDiskStars_eqFunction_1447(data, threadData);
  WhirlpoolDiskStars_eqFunction_3462(data, threadData);
  WhirlpoolDiskStars_eqFunction_3465(data, threadData);
  WhirlpoolDiskStars_eqFunction_3467(data, threadData);
  WhirlpoolDiskStars_eqFunction_3470(data, threadData);
  WhirlpoolDiskStars_eqFunction_3469(data, threadData);
  WhirlpoolDiskStars_eqFunction_3468(data, threadData);
  WhirlpoolDiskStars_eqFunction_3466(data, threadData);
  WhirlpoolDiskStars_eqFunction_1455(data, threadData);
  WhirlpoolDiskStars_eqFunction_3461(data, threadData);
  WhirlpoolDiskStars_eqFunction_1457(data, threadData);
  WhirlpoolDiskStars_eqFunction_3474(data, threadData);
  WhirlpoolDiskStars_eqFunction_1459(data, threadData);
  WhirlpoolDiskStars_eqFunction_1460(data, threadData);
  WhirlpoolDiskStars_eqFunction_3473(data, threadData);
  WhirlpoolDiskStars_eqFunction_1462(data, threadData);
  WhirlpoolDiskStars_eqFunction_1463(data, threadData);
  WhirlpoolDiskStars_eqFunction_3472(data, threadData);
  WhirlpoolDiskStars_eqFunction_3475(data, threadData);
  WhirlpoolDiskStars_eqFunction_3477(data, threadData);
  WhirlpoolDiskStars_eqFunction_3480(data, threadData);
  WhirlpoolDiskStars_eqFunction_3479(data, threadData);
  WhirlpoolDiskStars_eqFunction_3478(data, threadData);
  WhirlpoolDiskStars_eqFunction_3476(data, threadData);
  WhirlpoolDiskStars_eqFunction_1471(data, threadData);
  WhirlpoolDiskStars_eqFunction_3471(data, threadData);
  WhirlpoolDiskStars_eqFunction_1473(data, threadData);
  WhirlpoolDiskStars_eqFunction_3484(data, threadData);
  WhirlpoolDiskStars_eqFunction_1475(data, threadData);
  WhirlpoolDiskStars_eqFunction_1476(data, threadData);
  WhirlpoolDiskStars_eqFunction_3483(data, threadData);
  WhirlpoolDiskStars_eqFunction_1478(data, threadData);
  WhirlpoolDiskStars_eqFunction_1479(data, threadData);
  WhirlpoolDiskStars_eqFunction_3482(data, threadData);
  WhirlpoolDiskStars_eqFunction_3485(data, threadData);
  WhirlpoolDiskStars_eqFunction_3487(data, threadData);
  WhirlpoolDiskStars_eqFunction_3490(data, threadData);
  WhirlpoolDiskStars_eqFunction_3489(data, threadData);
  WhirlpoolDiskStars_eqFunction_3488(data, threadData);
  WhirlpoolDiskStars_eqFunction_3486(data, threadData);
  WhirlpoolDiskStars_eqFunction_1487(data, threadData);
  WhirlpoolDiskStars_eqFunction_3481(data, threadData);
  WhirlpoolDiskStars_eqFunction_1489(data, threadData);
  WhirlpoolDiskStars_eqFunction_3494(data, threadData);
  WhirlpoolDiskStars_eqFunction_1491(data, threadData);
  WhirlpoolDiskStars_eqFunction_1492(data, threadData);
  WhirlpoolDiskStars_eqFunction_3493(data, threadData);
  WhirlpoolDiskStars_eqFunction_1494(data, threadData);
  WhirlpoolDiskStars_eqFunction_1495(data, threadData);
  WhirlpoolDiskStars_eqFunction_3492(data, threadData);
  WhirlpoolDiskStars_eqFunction_3495(data, threadData);
  WhirlpoolDiskStars_eqFunction_3497(data, threadData);
  WhirlpoolDiskStars_eqFunction_3500(data, threadData);
  WhirlpoolDiskStars_eqFunction_3499(data, threadData);
  WhirlpoolDiskStars_eqFunction_3498(data, threadData);
  WhirlpoolDiskStars_eqFunction_3496(data, threadData);
  WhirlpoolDiskStars_eqFunction_1503(data, threadData);
  WhirlpoolDiskStars_eqFunction_3491(data, threadData);
  WhirlpoolDiskStars_eqFunction_1505(data, threadData);
  WhirlpoolDiskStars_eqFunction_3504(data, threadData);
  WhirlpoolDiskStars_eqFunction_1507(data, threadData);
  WhirlpoolDiskStars_eqFunction_1508(data, threadData);
  WhirlpoolDiskStars_eqFunction_3503(data, threadData);
  WhirlpoolDiskStars_eqFunction_1510(data, threadData);
  WhirlpoolDiskStars_eqFunction_1511(data, threadData);
  WhirlpoolDiskStars_eqFunction_3502(data, threadData);
  WhirlpoolDiskStars_eqFunction_3505(data, threadData);
  WhirlpoolDiskStars_eqFunction_3507(data, threadData);
  WhirlpoolDiskStars_eqFunction_3510(data, threadData);
  WhirlpoolDiskStars_eqFunction_3509(data, threadData);
  WhirlpoolDiskStars_eqFunction_3508(data, threadData);
  WhirlpoolDiskStars_eqFunction_3506(data, threadData);
  WhirlpoolDiskStars_eqFunction_1519(data, threadData);
  WhirlpoolDiskStars_eqFunction_3501(data, threadData);
  WhirlpoolDiskStars_eqFunction_1521(data, threadData);
  WhirlpoolDiskStars_eqFunction_3514(data, threadData);
  WhirlpoolDiskStars_eqFunction_1523(data, threadData);
  WhirlpoolDiskStars_eqFunction_1524(data, threadData);
  WhirlpoolDiskStars_eqFunction_3513(data, threadData);
  WhirlpoolDiskStars_eqFunction_1526(data, threadData);
  WhirlpoolDiskStars_eqFunction_1527(data, threadData);
  WhirlpoolDiskStars_eqFunction_3512(data, threadData);
  WhirlpoolDiskStars_eqFunction_3515(data, threadData);
  WhirlpoolDiskStars_eqFunction_3517(data, threadData);
  WhirlpoolDiskStars_eqFunction_3520(data, threadData);
  WhirlpoolDiskStars_eqFunction_3519(data, threadData);
  WhirlpoolDiskStars_eqFunction_3518(data, threadData);
  WhirlpoolDiskStars_eqFunction_3516(data, threadData);
  WhirlpoolDiskStars_eqFunction_1535(data, threadData);
  WhirlpoolDiskStars_eqFunction_3511(data, threadData);
  WhirlpoolDiskStars_eqFunction_1537(data, threadData);
  WhirlpoolDiskStars_eqFunction_3524(data, threadData);
  WhirlpoolDiskStars_eqFunction_1539(data, threadData);
  WhirlpoolDiskStars_eqFunction_1540(data, threadData);
  WhirlpoolDiskStars_eqFunction_3523(data, threadData);
  WhirlpoolDiskStars_eqFunction_1542(data, threadData);
  WhirlpoolDiskStars_eqFunction_1543(data, threadData);
  WhirlpoolDiskStars_eqFunction_3522(data, threadData);
  WhirlpoolDiskStars_eqFunction_3525(data, threadData);
  WhirlpoolDiskStars_eqFunction_3527(data, threadData);
  WhirlpoolDiskStars_eqFunction_3530(data, threadData);
  WhirlpoolDiskStars_eqFunction_3529(data, threadData);
  WhirlpoolDiskStars_eqFunction_3528(data, threadData);
  WhirlpoolDiskStars_eqFunction_3526(data, threadData);
  WhirlpoolDiskStars_eqFunction_1551(data, threadData);
  WhirlpoolDiskStars_eqFunction_3521(data, threadData);
  WhirlpoolDiskStars_eqFunction_1553(data, threadData);
  WhirlpoolDiskStars_eqFunction_3534(data, threadData);
  WhirlpoolDiskStars_eqFunction_1555(data, threadData);
  WhirlpoolDiskStars_eqFunction_1556(data, threadData);
  WhirlpoolDiskStars_eqFunction_3533(data, threadData);
  WhirlpoolDiskStars_eqFunction_1558(data, threadData);
  WhirlpoolDiskStars_eqFunction_1559(data, threadData);
  WhirlpoolDiskStars_eqFunction_3532(data, threadData);
  WhirlpoolDiskStars_eqFunction_3535(data, threadData);
  WhirlpoolDiskStars_eqFunction_3537(data, threadData);
  WhirlpoolDiskStars_eqFunction_3540(data, threadData);
  WhirlpoolDiskStars_eqFunction_3539(data, threadData);
  WhirlpoolDiskStars_eqFunction_3538(data, threadData);
  WhirlpoolDiskStars_eqFunction_3536(data, threadData);
  WhirlpoolDiskStars_eqFunction_1567(data, threadData);
  WhirlpoolDiskStars_eqFunction_3531(data, threadData);
  WhirlpoolDiskStars_eqFunction_1569(data, threadData);
  WhirlpoolDiskStars_eqFunction_3544(data, threadData);
  WhirlpoolDiskStars_eqFunction_1571(data, threadData);
  WhirlpoolDiskStars_eqFunction_1572(data, threadData);
  WhirlpoolDiskStars_eqFunction_3543(data, threadData);
  WhirlpoolDiskStars_eqFunction_1574(data, threadData);
  WhirlpoolDiskStars_eqFunction_1575(data, threadData);
  WhirlpoolDiskStars_eqFunction_3542(data, threadData);
  WhirlpoolDiskStars_eqFunction_3545(data, threadData);
  WhirlpoolDiskStars_eqFunction_3547(data, threadData);
  WhirlpoolDiskStars_eqFunction_3550(data, threadData);
  WhirlpoolDiskStars_eqFunction_3549(data, threadData);
  WhirlpoolDiskStars_eqFunction_3548(data, threadData);
  WhirlpoolDiskStars_eqFunction_3546(data, threadData);
  WhirlpoolDiskStars_eqFunction_1583(data, threadData);
  WhirlpoolDiskStars_eqFunction_3541(data, threadData);
  WhirlpoolDiskStars_eqFunction_1585(data, threadData);
  WhirlpoolDiskStars_eqFunction_3554(data, threadData);
  WhirlpoolDiskStars_eqFunction_1587(data, threadData);
  WhirlpoolDiskStars_eqFunction_1588(data, threadData);
  WhirlpoolDiskStars_eqFunction_3553(data, threadData);
  WhirlpoolDiskStars_eqFunction_1590(data, threadData);
  WhirlpoolDiskStars_eqFunction_1591(data, threadData);
  WhirlpoolDiskStars_eqFunction_3552(data, threadData);
  WhirlpoolDiskStars_eqFunction_3555(data, threadData);
  WhirlpoolDiskStars_eqFunction_3557(data, threadData);
  WhirlpoolDiskStars_eqFunction_3560(data, threadData);
  WhirlpoolDiskStars_eqFunction_3559(data, threadData);
  WhirlpoolDiskStars_eqFunction_3558(data, threadData);
  WhirlpoolDiskStars_eqFunction_3556(data, threadData);
  WhirlpoolDiskStars_eqFunction_1599(data, threadData);
  WhirlpoolDiskStars_eqFunction_3551(data, threadData);
  WhirlpoolDiskStars_eqFunction_1601(data, threadData);
  WhirlpoolDiskStars_eqFunction_3564(data, threadData);
  WhirlpoolDiskStars_eqFunction_1603(data, threadData);
  WhirlpoolDiskStars_eqFunction_1604(data, threadData);
  WhirlpoolDiskStars_eqFunction_3563(data, threadData);
  WhirlpoolDiskStars_eqFunction_1606(data, threadData);
  WhirlpoolDiskStars_eqFunction_1607(data, threadData);
  WhirlpoolDiskStars_eqFunction_3562(data, threadData);
  WhirlpoolDiskStars_eqFunction_3565(data, threadData);
  WhirlpoolDiskStars_eqFunction_3567(data, threadData);
  WhirlpoolDiskStars_eqFunction_3570(data, threadData);
  WhirlpoolDiskStars_eqFunction_3569(data, threadData);
  WhirlpoolDiskStars_eqFunction_3568(data, threadData);
  WhirlpoolDiskStars_eqFunction_3566(data, threadData);
  WhirlpoolDiskStars_eqFunction_1615(data, threadData);
  WhirlpoolDiskStars_eqFunction_3561(data, threadData);
  WhirlpoolDiskStars_eqFunction_1617(data, threadData);
  WhirlpoolDiskStars_eqFunction_3574(data, threadData);
  WhirlpoolDiskStars_eqFunction_1619(data, threadData);
  WhirlpoolDiskStars_eqFunction_1620(data, threadData);
  WhirlpoolDiskStars_eqFunction_3573(data, threadData);
  WhirlpoolDiskStars_eqFunction_1622(data, threadData);
  WhirlpoolDiskStars_eqFunction_1623(data, threadData);
  WhirlpoolDiskStars_eqFunction_3572(data, threadData);
  WhirlpoolDiskStars_eqFunction_3575(data, threadData);
  WhirlpoolDiskStars_eqFunction_3577(data, threadData);
  WhirlpoolDiskStars_eqFunction_3580(data, threadData);
  WhirlpoolDiskStars_eqFunction_3579(data, threadData);
  WhirlpoolDiskStars_eqFunction_3578(data, threadData);
  WhirlpoolDiskStars_eqFunction_3576(data, threadData);
  WhirlpoolDiskStars_eqFunction_1631(data, threadData);
  WhirlpoolDiskStars_eqFunction_3571(data, threadData);
  WhirlpoolDiskStars_eqFunction_1633(data, threadData);
  WhirlpoolDiskStars_eqFunction_3584(data, threadData);
  WhirlpoolDiskStars_eqFunction_1635(data, threadData);
  WhirlpoolDiskStars_eqFunction_1636(data, threadData);
  WhirlpoolDiskStars_eqFunction_3583(data, threadData);
  WhirlpoolDiskStars_eqFunction_1638(data, threadData);
  WhirlpoolDiskStars_eqFunction_1639(data, threadData);
  WhirlpoolDiskStars_eqFunction_3582(data, threadData);
  WhirlpoolDiskStars_eqFunction_3585(data, threadData);
  WhirlpoolDiskStars_eqFunction_3587(data, threadData);
  WhirlpoolDiskStars_eqFunction_3590(data, threadData);
  WhirlpoolDiskStars_eqFunction_3589(data, threadData);
  WhirlpoolDiskStars_eqFunction_3588(data, threadData);
  WhirlpoolDiskStars_eqFunction_3586(data, threadData);
  WhirlpoolDiskStars_eqFunction_1647(data, threadData);
  WhirlpoolDiskStars_eqFunction_3581(data, threadData);
  WhirlpoolDiskStars_eqFunction_1649(data, threadData);
  WhirlpoolDiskStars_eqFunction_3594(data, threadData);
  WhirlpoolDiskStars_eqFunction_1651(data, threadData);
  WhirlpoolDiskStars_eqFunction_1652(data, threadData);
  WhirlpoolDiskStars_eqFunction_3593(data, threadData);
  WhirlpoolDiskStars_eqFunction_1654(data, threadData);
  WhirlpoolDiskStars_eqFunction_1655(data, threadData);
  WhirlpoolDiskStars_eqFunction_3592(data, threadData);
  WhirlpoolDiskStars_eqFunction_3595(data, threadData);
  WhirlpoolDiskStars_eqFunction_3597(data, threadData);
  WhirlpoolDiskStars_eqFunction_3600(data, threadData);
  WhirlpoolDiskStars_eqFunction_3599(data, threadData);
  WhirlpoolDiskStars_eqFunction_3598(data, threadData);
  WhirlpoolDiskStars_eqFunction_3596(data, threadData);
  WhirlpoolDiskStars_eqFunction_1663(data, threadData);
  WhirlpoolDiskStars_eqFunction_3591(data, threadData);
  WhirlpoolDiskStars_eqFunction_1665(data, threadData);
  WhirlpoolDiskStars_eqFunction_3604(data, threadData);
  WhirlpoolDiskStars_eqFunction_1667(data, threadData);
  WhirlpoolDiskStars_eqFunction_1668(data, threadData);
  WhirlpoolDiskStars_eqFunction_3603(data, threadData);
  WhirlpoolDiskStars_eqFunction_1670(data, threadData);
  WhirlpoolDiskStars_eqFunction_1671(data, threadData);
  WhirlpoolDiskStars_eqFunction_3602(data, threadData);
  WhirlpoolDiskStars_eqFunction_3605(data, threadData);
  WhirlpoolDiskStars_eqFunction_3607(data, threadData);
  WhirlpoolDiskStars_eqFunction_3610(data, threadData);
  WhirlpoolDiskStars_eqFunction_3609(data, threadData);
  WhirlpoolDiskStars_eqFunction_3608(data, threadData);
  WhirlpoolDiskStars_eqFunction_3606(data, threadData);
  WhirlpoolDiskStars_eqFunction_1679(data, threadData);
  WhirlpoolDiskStars_eqFunction_3601(data, threadData);
  WhirlpoolDiskStars_eqFunction_1681(data, threadData);
  WhirlpoolDiskStars_eqFunction_3614(data, threadData);
  WhirlpoolDiskStars_eqFunction_1683(data, threadData);
  WhirlpoolDiskStars_eqFunction_1684(data, threadData);
  WhirlpoolDiskStars_eqFunction_3613(data, threadData);
  WhirlpoolDiskStars_eqFunction_1686(data, threadData);
  WhirlpoolDiskStars_eqFunction_1687(data, threadData);
  WhirlpoolDiskStars_eqFunction_3612(data, threadData);
  WhirlpoolDiskStars_eqFunction_3615(data, threadData);
  WhirlpoolDiskStars_eqFunction_3617(data, threadData);
  WhirlpoolDiskStars_eqFunction_3620(data, threadData);
  WhirlpoolDiskStars_eqFunction_3619(data, threadData);
  WhirlpoolDiskStars_eqFunction_3618(data, threadData);
  WhirlpoolDiskStars_eqFunction_3616(data, threadData);
  WhirlpoolDiskStars_eqFunction_1695(data, threadData);
  WhirlpoolDiskStars_eqFunction_3611(data, threadData);
  WhirlpoolDiskStars_eqFunction_1697(data, threadData);
  WhirlpoolDiskStars_eqFunction_3624(data, threadData);
  WhirlpoolDiskStars_eqFunction_1699(data, threadData);
  WhirlpoolDiskStars_eqFunction_1700(data, threadData);
  WhirlpoolDiskStars_eqFunction_3623(data, threadData);
  WhirlpoolDiskStars_eqFunction_1702(data, threadData);
  WhirlpoolDiskStars_eqFunction_1703(data, threadData);
  WhirlpoolDiskStars_eqFunction_3622(data, threadData);
  WhirlpoolDiskStars_eqFunction_3625(data, threadData);
  WhirlpoolDiskStars_eqFunction_3627(data, threadData);
  WhirlpoolDiskStars_eqFunction_3630(data, threadData);
  WhirlpoolDiskStars_eqFunction_3629(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif