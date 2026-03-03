#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 1
type: SIMPLE_ASSIGN
z[1] = -3.95
*/
void WhirlpoolDiskStars_eqFunction_1(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[800]] /* z[1] STATE(1,vz[1]) */) = -3.95;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2564(DATA *data, threadData_t *threadData);


/*
equation index: 3
type: SIMPLE_ASSIGN
y[1] = r_init[1] * sin(theta[1] + armOffset[1])
*/
void WhirlpoolDiskStars_eqFunction_3(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,3};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[640]] /* y[1] STATE(1,vy[1]) */) = ((data->simulationInfo->realParameter[165] /* r_init[1] PARAM */)) * (sin((data->simulationInfo->realParameter[325] /* theta[1] PARAM */) + (data->simulationInfo->realParameter[3] /* armOffset[1] PARAM */)));
  TRACE_POP
}

/*
equation index: 4
type: SIMPLE_ASSIGN
vx[1] = (-y[1]) * sqrt(G * Md / r_init[1] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_4(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4};
  modelica_real tmp0;
  modelica_real tmp1;
  tmp0 = (data->simulationInfo->realParameter[165] /* r_init[1] PARAM */);
  tmp1 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp0 * tmp0 * tmp0),"r_init[1] ^ 3.0",equationIndexes);
  if(!(tmp1 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[1] ^ 3.0) was %g should be >= 0", tmp1);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* vx[1] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[640]] /* y[1] STATE(1,vy[1]) */))) * (sqrt(tmp1));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2563(DATA *data, threadData_t *threadData);


/*
equation index: 6
type: SIMPLE_ASSIGN
x[1] = r_init[1] * cos(theta[1] + armOffset[1])
*/
void WhirlpoolDiskStars_eqFunction_6(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,6};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[480]] /* x[1] STATE(1,vx[1]) */) = ((data->simulationInfo->realParameter[165] /* r_init[1] PARAM */)) * (cos((data->simulationInfo->realParameter[325] /* theta[1] PARAM */) + (data->simulationInfo->realParameter[3] /* armOffset[1] PARAM */)));
  TRACE_POP
}

/*
equation index: 7
type: SIMPLE_ASSIGN
vy[1] = x[1] * sqrt(G * Md / r_init[1] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_7(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,7};
  modelica_real tmp2;
  modelica_real tmp3;
  tmp2 = (data->simulationInfo->realParameter[165] /* r_init[1] PARAM */);
  tmp3 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp2 * tmp2 * tmp2),"r_init[1] ^ 3.0",equationIndexes);
  if(!(tmp3 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[1] ^ 3.0) was %g should be >= 0", tmp3);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[160]] /* vy[1] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[480]] /* x[1] STATE(1,vx[1]) */)) * (sqrt(tmp3));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2562(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2565(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2567(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2570(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2569(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2568(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2566(DATA *data, threadData_t *threadData);


/*
equation index: 15
type: SIMPLE_ASSIGN
vz[1] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_15(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[320]] /* vz[1] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2561(DATA *data, threadData_t *threadData);


/*
equation index: 17
type: SIMPLE_ASSIGN
z[2] = -3.9000000000000004
*/
void WhirlpoolDiskStars_eqFunction_17(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,17};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[801]] /* z[2] STATE(1,vz[2]) */) = -3.9000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2574(DATA *data, threadData_t *threadData);


/*
equation index: 19
type: SIMPLE_ASSIGN
y[2] = r_init[2] * sin(theta[2] + armOffset[2])
*/
void WhirlpoolDiskStars_eqFunction_19(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,19};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[641]] /* y[2] STATE(1,vy[2]) */) = ((data->simulationInfo->realParameter[166] /* r_init[2] PARAM */)) * (sin((data->simulationInfo->realParameter[326] /* theta[2] PARAM */) + (data->simulationInfo->realParameter[4] /* armOffset[2] PARAM */)));
  TRACE_POP
}

/*
equation index: 20
type: SIMPLE_ASSIGN
vx[2] = (-y[2]) * sqrt(G * Md / r_init[2] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_20(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,20};
  modelica_real tmp4;
  modelica_real tmp5;
  tmp4 = (data->simulationInfo->realParameter[166] /* r_init[2] PARAM */);
  tmp5 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp4 * tmp4 * tmp4),"r_init[2] ^ 3.0",equationIndexes);
  if(!(tmp5 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[2] ^ 3.0) was %g should be >= 0", tmp5);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* vx[2] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[641]] /* y[2] STATE(1,vy[2]) */))) * (sqrt(tmp5));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2573(DATA *data, threadData_t *threadData);


/*
equation index: 22
type: SIMPLE_ASSIGN
x[2] = r_init[2] * cos(theta[2] + armOffset[2])
*/
void WhirlpoolDiskStars_eqFunction_22(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,22};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[481]] /* x[2] STATE(1,vx[2]) */) = ((data->simulationInfo->realParameter[166] /* r_init[2] PARAM */)) * (cos((data->simulationInfo->realParameter[326] /* theta[2] PARAM */) + (data->simulationInfo->realParameter[4] /* armOffset[2] PARAM */)));
  TRACE_POP
}

/*
equation index: 23
type: SIMPLE_ASSIGN
vy[2] = x[2] * sqrt(G * Md / r_init[2] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_23(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,23};
  modelica_real tmp6;
  modelica_real tmp7;
  tmp6 = (data->simulationInfo->realParameter[166] /* r_init[2] PARAM */);
  tmp7 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp6 * tmp6 * tmp6),"r_init[2] ^ 3.0",equationIndexes);
  if(!(tmp7 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[2] ^ 3.0) was %g should be >= 0", tmp7);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[161]] /* vy[2] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[481]] /* x[2] STATE(1,vx[2]) */)) * (sqrt(tmp7));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2572(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2575(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2577(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2580(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2579(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2578(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2576(DATA *data, threadData_t *threadData);


/*
equation index: 31
type: SIMPLE_ASSIGN
vz[2] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_31(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[321]] /* vz[2] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2571(DATA *data, threadData_t *threadData);


/*
equation index: 33
type: SIMPLE_ASSIGN
z[3] = -3.85
*/
void WhirlpoolDiskStars_eqFunction_33(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[802]] /* z[3] STATE(1,vz[3]) */) = -3.85;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2584(DATA *data, threadData_t *threadData);


/*
equation index: 35
type: SIMPLE_ASSIGN
y[3] = r_init[3] * sin(theta[3] + armOffset[3])
*/
void WhirlpoolDiskStars_eqFunction_35(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[642]] /* y[3] STATE(1,vy[3]) */) = ((data->simulationInfo->realParameter[167] /* r_init[3] PARAM */)) * (sin((data->simulationInfo->realParameter[327] /* theta[3] PARAM */) + (data->simulationInfo->realParameter[5] /* armOffset[3] PARAM */)));
  TRACE_POP
}

/*
equation index: 36
type: SIMPLE_ASSIGN
vx[3] = (-y[3]) * sqrt(G * Md / r_init[3] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_36(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,36};
  modelica_real tmp8;
  modelica_real tmp9;
  tmp8 = (data->simulationInfo->realParameter[167] /* r_init[3] PARAM */);
  tmp9 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp8 * tmp8 * tmp8),"r_init[3] ^ 3.0",equationIndexes);
  if(!(tmp9 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[3] ^ 3.0) was %g should be >= 0", tmp9);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* vx[3] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[642]] /* y[3] STATE(1,vy[3]) */))) * (sqrt(tmp9));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2583(DATA *data, threadData_t *threadData);


/*
equation index: 38
type: SIMPLE_ASSIGN
x[3] = r_init[3] * cos(theta[3] + armOffset[3])
*/
void WhirlpoolDiskStars_eqFunction_38(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[482]] /* x[3] STATE(1,vx[3]) */) = ((data->simulationInfo->realParameter[167] /* r_init[3] PARAM */)) * (cos((data->simulationInfo->realParameter[327] /* theta[3] PARAM */) + (data->simulationInfo->realParameter[5] /* armOffset[3] PARAM */)));
  TRACE_POP
}

/*
equation index: 39
type: SIMPLE_ASSIGN
vy[3] = x[3] * sqrt(G * Md / r_init[3] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_39(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,39};
  modelica_real tmp10;
  modelica_real tmp11;
  tmp10 = (data->simulationInfo->realParameter[167] /* r_init[3] PARAM */);
  tmp11 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp10 * tmp10 * tmp10),"r_init[3] ^ 3.0",equationIndexes);
  if(!(tmp11 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[3] ^ 3.0) was %g should be >= 0", tmp11);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[162]] /* vy[3] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[482]] /* x[3] STATE(1,vx[3]) */)) * (sqrt(tmp11));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2582(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2585(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2587(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2590(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2589(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2588(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2586(DATA *data, threadData_t *threadData);


/*
equation index: 47
type: SIMPLE_ASSIGN
vz[3] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_47(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,47};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[322]] /* vz[3] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2581(DATA *data, threadData_t *threadData);


/*
equation index: 49
type: SIMPLE_ASSIGN
z[4] = -3.8000000000000003
*/
void WhirlpoolDiskStars_eqFunction_49(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,49};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[803]] /* z[4] STATE(1,vz[4]) */) = -3.8000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2594(DATA *data, threadData_t *threadData);


/*
equation index: 51
type: SIMPLE_ASSIGN
y[4] = r_init[4] * sin(theta[4] + armOffset[4])
*/
void WhirlpoolDiskStars_eqFunction_51(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,51};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[643]] /* y[4] STATE(1,vy[4]) */) = ((data->simulationInfo->realParameter[168] /* r_init[4] PARAM */)) * (sin((data->simulationInfo->realParameter[328] /* theta[4] PARAM */) + (data->simulationInfo->realParameter[6] /* armOffset[4] PARAM */)));
  TRACE_POP
}

/*
equation index: 52
type: SIMPLE_ASSIGN
vx[4] = (-y[4]) * sqrt(G * Md / r_init[4] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_52(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,52};
  modelica_real tmp12;
  modelica_real tmp13;
  tmp12 = (data->simulationInfo->realParameter[168] /* r_init[4] PARAM */);
  tmp13 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp12 * tmp12 * tmp12),"r_init[4] ^ 3.0",equationIndexes);
  if(!(tmp13 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[4] ^ 3.0) was %g should be >= 0", tmp13);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* vx[4] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[643]] /* y[4] STATE(1,vy[4]) */))) * (sqrt(tmp13));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2593(DATA *data, threadData_t *threadData);


/*
equation index: 54
type: SIMPLE_ASSIGN
x[4] = r_init[4] * cos(theta[4] + armOffset[4])
*/
void WhirlpoolDiskStars_eqFunction_54(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,54};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[483]] /* x[4] STATE(1,vx[4]) */) = ((data->simulationInfo->realParameter[168] /* r_init[4] PARAM */)) * (cos((data->simulationInfo->realParameter[328] /* theta[4] PARAM */) + (data->simulationInfo->realParameter[6] /* armOffset[4] PARAM */)));
  TRACE_POP
}

/*
equation index: 55
type: SIMPLE_ASSIGN
vy[4] = x[4] * sqrt(G * Md / r_init[4] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_55(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,55};
  modelica_real tmp14;
  modelica_real tmp15;
  tmp14 = (data->simulationInfo->realParameter[168] /* r_init[4] PARAM */);
  tmp15 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp14 * tmp14 * tmp14),"r_init[4] ^ 3.0",equationIndexes);
  if(!(tmp15 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[4] ^ 3.0) was %g should be >= 0", tmp15);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[163]] /* vy[4] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[483]] /* x[4] STATE(1,vx[4]) */)) * (sqrt(tmp15));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2592(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2595(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2597(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2600(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2599(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2598(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2596(DATA *data, threadData_t *threadData);


/*
equation index: 63
type: SIMPLE_ASSIGN
vz[4] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_63(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,63};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[323]] /* vz[4] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2591(DATA *data, threadData_t *threadData);


/*
equation index: 65
type: SIMPLE_ASSIGN
z[5] = -3.75
*/
void WhirlpoolDiskStars_eqFunction_65(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,65};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[804]] /* z[5] STATE(1,vz[5]) */) = -3.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2604(DATA *data, threadData_t *threadData);


/*
equation index: 67
type: SIMPLE_ASSIGN
y[5] = r_init[5] * sin(theta[5] + armOffset[5])
*/
void WhirlpoolDiskStars_eqFunction_67(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,67};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[644]] /* y[5] STATE(1,vy[5]) */) = ((data->simulationInfo->realParameter[169] /* r_init[5] PARAM */)) * (sin((data->simulationInfo->realParameter[329] /* theta[5] PARAM */) + (data->simulationInfo->realParameter[7] /* armOffset[5] PARAM */)));
  TRACE_POP
}

/*
equation index: 68
type: SIMPLE_ASSIGN
vx[5] = (-y[5]) * sqrt(G * Md / r_init[5] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_68(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,68};
  modelica_real tmp16;
  modelica_real tmp17;
  tmp16 = (data->simulationInfo->realParameter[169] /* r_init[5] PARAM */);
  tmp17 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp16 * tmp16 * tmp16),"r_init[5] ^ 3.0",equationIndexes);
  if(!(tmp17 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[5] ^ 3.0) was %g should be >= 0", tmp17);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* vx[5] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[644]] /* y[5] STATE(1,vy[5]) */))) * (sqrt(tmp17));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2603(DATA *data, threadData_t *threadData);


/*
equation index: 70
type: SIMPLE_ASSIGN
x[5] = r_init[5] * cos(theta[5] + armOffset[5])
*/
void WhirlpoolDiskStars_eqFunction_70(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,70};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[484]] /* x[5] STATE(1,vx[5]) */) = ((data->simulationInfo->realParameter[169] /* r_init[5] PARAM */)) * (cos((data->simulationInfo->realParameter[329] /* theta[5] PARAM */) + (data->simulationInfo->realParameter[7] /* armOffset[5] PARAM */)));
  TRACE_POP
}

/*
equation index: 71
type: SIMPLE_ASSIGN
vy[5] = x[5] * sqrt(G * Md / r_init[5] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_71(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,71};
  modelica_real tmp18;
  modelica_real tmp19;
  tmp18 = (data->simulationInfo->realParameter[169] /* r_init[5] PARAM */);
  tmp19 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp18 * tmp18 * tmp18),"r_init[5] ^ 3.0",equationIndexes);
  if(!(tmp19 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[5] ^ 3.0) was %g should be >= 0", tmp19);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[164]] /* vy[5] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[484]] /* x[5] STATE(1,vx[5]) */)) * (sqrt(tmp19));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2602(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2605(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2607(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2610(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2609(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2608(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2606(DATA *data, threadData_t *threadData);


/*
equation index: 79
type: SIMPLE_ASSIGN
vz[5] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_79(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,79};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[324]] /* vz[5] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2601(DATA *data, threadData_t *threadData);


/*
equation index: 81
type: SIMPLE_ASSIGN
z[6] = -3.7
*/
void WhirlpoolDiskStars_eqFunction_81(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,81};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[805]] /* z[6] STATE(1,vz[6]) */) = -3.7;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2614(DATA *data, threadData_t *threadData);


/*
equation index: 83
type: SIMPLE_ASSIGN
y[6] = r_init[6] * sin(theta[6] + armOffset[6])
*/
void WhirlpoolDiskStars_eqFunction_83(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,83};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[645]] /* y[6] STATE(1,vy[6]) */) = ((data->simulationInfo->realParameter[170] /* r_init[6] PARAM */)) * (sin((data->simulationInfo->realParameter[330] /* theta[6] PARAM */) + (data->simulationInfo->realParameter[8] /* armOffset[6] PARAM */)));
  TRACE_POP
}

/*
equation index: 84
type: SIMPLE_ASSIGN
vx[6] = (-y[6]) * sqrt(G * Md / r_init[6] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_84(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,84};
  modelica_real tmp20;
  modelica_real tmp21;
  tmp20 = (data->simulationInfo->realParameter[170] /* r_init[6] PARAM */);
  tmp21 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp20 * tmp20 * tmp20),"r_init[6] ^ 3.0",equationIndexes);
  if(!(tmp21 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[6] ^ 3.0) was %g should be >= 0", tmp21);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* vx[6] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[645]] /* y[6] STATE(1,vy[6]) */))) * (sqrt(tmp21));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2613(DATA *data, threadData_t *threadData);


/*
equation index: 86
type: SIMPLE_ASSIGN
x[6] = r_init[6] * cos(theta[6] + armOffset[6])
*/
void WhirlpoolDiskStars_eqFunction_86(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,86};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[485]] /* x[6] STATE(1,vx[6]) */) = ((data->simulationInfo->realParameter[170] /* r_init[6] PARAM */)) * (cos((data->simulationInfo->realParameter[330] /* theta[6] PARAM */) + (data->simulationInfo->realParameter[8] /* armOffset[6] PARAM */)));
  TRACE_POP
}

/*
equation index: 87
type: SIMPLE_ASSIGN
vy[6] = x[6] * sqrt(G * Md / r_init[6] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_87(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,87};
  modelica_real tmp22;
  modelica_real tmp23;
  tmp22 = (data->simulationInfo->realParameter[170] /* r_init[6] PARAM */);
  tmp23 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp22 * tmp22 * tmp22),"r_init[6] ^ 3.0",equationIndexes);
  if(!(tmp23 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[6] ^ 3.0) was %g should be >= 0", tmp23);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[165]] /* vy[6] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[485]] /* x[6] STATE(1,vx[6]) */)) * (sqrt(tmp23));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2612(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2615(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2617(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2620(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2619(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2618(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2616(DATA *data, threadData_t *threadData);


/*
equation index: 95
type: SIMPLE_ASSIGN
vz[6] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_95(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,95};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[325]] /* vz[6] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2611(DATA *data, threadData_t *threadData);


/*
equation index: 97
type: SIMPLE_ASSIGN
z[7] = -3.6500000000000004
*/
void WhirlpoolDiskStars_eqFunction_97(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,97};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[806]] /* z[7] STATE(1,vz[7]) */) = -3.6500000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2624(DATA *data, threadData_t *threadData);


/*
equation index: 99
type: SIMPLE_ASSIGN
y[7] = r_init[7] * sin(theta[7] + armOffset[7])
*/
void WhirlpoolDiskStars_eqFunction_99(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,99};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[646]] /* y[7] STATE(1,vy[7]) */) = ((data->simulationInfo->realParameter[171] /* r_init[7] PARAM */)) * (sin((data->simulationInfo->realParameter[331] /* theta[7] PARAM */) + (data->simulationInfo->realParameter[9] /* armOffset[7] PARAM */)));
  TRACE_POP
}

/*
equation index: 100
type: SIMPLE_ASSIGN
vx[7] = (-y[7]) * sqrt(G * Md / r_init[7] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,100};
  modelica_real tmp24;
  modelica_real tmp25;
  tmp24 = (data->simulationInfo->realParameter[171] /* r_init[7] PARAM */);
  tmp25 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp24 * tmp24 * tmp24),"r_init[7] ^ 3.0",equationIndexes);
  if(!(tmp25 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[7] ^ 3.0) was %g should be >= 0", tmp25);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* vx[7] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[646]] /* y[7] STATE(1,vy[7]) */))) * (sqrt(tmp25));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2623(DATA *data, threadData_t *threadData);


/*
equation index: 102
type: SIMPLE_ASSIGN
x[7] = r_init[7] * cos(theta[7] + armOffset[7])
*/
void WhirlpoolDiskStars_eqFunction_102(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,102};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[486]] /* x[7] STATE(1,vx[7]) */) = ((data->simulationInfo->realParameter[171] /* r_init[7] PARAM */)) * (cos((data->simulationInfo->realParameter[331] /* theta[7] PARAM */) + (data->simulationInfo->realParameter[9] /* armOffset[7] PARAM */)));
  TRACE_POP
}

/*
equation index: 103
type: SIMPLE_ASSIGN
vy[7] = x[7] * sqrt(G * Md / r_init[7] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,103};
  modelica_real tmp26;
  modelica_real tmp27;
  tmp26 = (data->simulationInfo->realParameter[171] /* r_init[7] PARAM */);
  tmp27 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp26 * tmp26 * tmp26),"r_init[7] ^ 3.0",equationIndexes);
  if(!(tmp27 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[7] ^ 3.0) was %g should be >= 0", tmp27);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[166]] /* vy[7] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[486]] /* x[7] STATE(1,vx[7]) */)) * (sqrt(tmp27));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2622(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2625(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2627(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2630(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2629(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2628(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2626(DATA *data, threadData_t *threadData);


/*
equation index: 111
type: SIMPLE_ASSIGN
vz[7] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,111};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[326]] /* vz[7] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2621(DATA *data, threadData_t *threadData);


/*
equation index: 113
type: SIMPLE_ASSIGN
z[8] = -3.6
*/
void WhirlpoolDiskStars_eqFunction_113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,113};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[807]] /* z[8] STATE(1,vz[8]) */) = -3.6;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2634(DATA *data, threadData_t *threadData);


/*
equation index: 115
type: SIMPLE_ASSIGN
y[8] = r_init[8] * sin(theta[8] + armOffset[8])
*/
void WhirlpoolDiskStars_eqFunction_115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[647]] /* y[8] STATE(1,vy[8]) */) = ((data->simulationInfo->realParameter[172] /* r_init[8] PARAM */)) * (sin((data->simulationInfo->realParameter[332] /* theta[8] PARAM */) + (data->simulationInfo->realParameter[10] /* armOffset[8] PARAM */)));
  TRACE_POP
}

/*
equation index: 116
type: SIMPLE_ASSIGN
vx[8] = (-y[8]) * sqrt(G * Md / r_init[8] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,116};
  modelica_real tmp28;
  modelica_real tmp29;
  tmp28 = (data->simulationInfo->realParameter[172] /* r_init[8] PARAM */);
  tmp29 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp28 * tmp28 * tmp28),"r_init[8] ^ 3.0",equationIndexes);
  if(!(tmp29 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[8] ^ 3.0) was %g should be >= 0", tmp29);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* vx[8] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[647]] /* y[8] STATE(1,vy[8]) */))) * (sqrt(tmp29));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2633(DATA *data, threadData_t *threadData);


/*
equation index: 118
type: SIMPLE_ASSIGN
x[8] = r_init[8] * cos(theta[8] + armOffset[8])
*/
void WhirlpoolDiskStars_eqFunction_118(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,118};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[487]] /* x[8] STATE(1,vx[8]) */) = ((data->simulationInfo->realParameter[172] /* r_init[8] PARAM */)) * (cos((data->simulationInfo->realParameter[332] /* theta[8] PARAM */) + (data->simulationInfo->realParameter[10] /* armOffset[8] PARAM */)));
  TRACE_POP
}

/*
equation index: 119
type: SIMPLE_ASSIGN
vy[8] = x[8] * sqrt(G * Md / r_init[8] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,119};
  modelica_real tmp30;
  modelica_real tmp31;
  tmp30 = (data->simulationInfo->realParameter[172] /* r_init[8] PARAM */);
  tmp31 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp30 * tmp30 * tmp30),"r_init[8] ^ 3.0",equationIndexes);
  if(!(tmp31 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[8] ^ 3.0) was %g should be >= 0", tmp31);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[167]] /* vy[8] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[487]] /* x[8] STATE(1,vx[8]) */)) * (sqrt(tmp31));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2632(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2635(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2637(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2640(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2639(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2638(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2636(DATA *data, threadData_t *threadData);


/*
equation index: 127
type: SIMPLE_ASSIGN
vz[8] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,127};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[327]] /* vz[8] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2631(DATA *data, threadData_t *threadData);


/*
equation index: 129
type: SIMPLE_ASSIGN
z[9] = -3.5500000000000003
*/
void WhirlpoolDiskStars_eqFunction_129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,129};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[808]] /* z[9] STATE(1,vz[9]) */) = -3.5500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2644(DATA *data, threadData_t *threadData);


/*
equation index: 131
type: SIMPLE_ASSIGN
y[9] = r_init[9] * sin(theta[9] + armOffset[9])
*/
void WhirlpoolDiskStars_eqFunction_131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[648]] /* y[9] STATE(1,vy[9]) */) = ((data->simulationInfo->realParameter[173] /* r_init[9] PARAM */)) * (sin((data->simulationInfo->realParameter[333] /* theta[9] PARAM */) + (data->simulationInfo->realParameter[11] /* armOffset[9] PARAM */)));
  TRACE_POP
}

/*
equation index: 132
type: SIMPLE_ASSIGN
vx[9] = (-y[9]) * sqrt(G * Md / r_init[9] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,132};
  modelica_real tmp32;
  modelica_real tmp33;
  tmp32 = (data->simulationInfo->realParameter[173] /* r_init[9] PARAM */);
  tmp33 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp32 * tmp32 * tmp32),"r_init[9] ^ 3.0",equationIndexes);
  if(!(tmp33 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[9] ^ 3.0) was %g should be >= 0", tmp33);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* vx[9] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[648]] /* y[9] STATE(1,vy[9]) */))) * (sqrt(tmp33));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2643(DATA *data, threadData_t *threadData);


/*
equation index: 134
type: SIMPLE_ASSIGN
x[9] = r_init[9] * cos(theta[9] + armOffset[9])
*/
void WhirlpoolDiskStars_eqFunction_134(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,134};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[488]] /* x[9] STATE(1,vx[9]) */) = ((data->simulationInfo->realParameter[173] /* r_init[9] PARAM */)) * (cos((data->simulationInfo->realParameter[333] /* theta[9] PARAM */) + (data->simulationInfo->realParameter[11] /* armOffset[9] PARAM */)));
  TRACE_POP
}

/*
equation index: 135
type: SIMPLE_ASSIGN
vy[9] = x[9] * sqrt(G * Md / r_init[9] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,135};
  modelica_real tmp34;
  modelica_real tmp35;
  tmp34 = (data->simulationInfo->realParameter[173] /* r_init[9] PARAM */);
  tmp35 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp34 * tmp34 * tmp34),"r_init[9] ^ 3.0",equationIndexes);
  if(!(tmp35 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[9] ^ 3.0) was %g should be >= 0", tmp35);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[168]] /* vy[9] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[488]] /* x[9] STATE(1,vx[9]) */)) * (sqrt(tmp35));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2642(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2645(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2647(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2650(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2649(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2648(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2646(DATA *data, threadData_t *threadData);


/*
equation index: 143
type: SIMPLE_ASSIGN
vz[9] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,143};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[328]] /* vz[9] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2641(DATA *data, threadData_t *threadData);


/*
equation index: 145
type: SIMPLE_ASSIGN
z[10] = -3.5
*/
void WhirlpoolDiskStars_eqFunction_145(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,145};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[809]] /* z[10] STATE(1,vz[10]) */) = -3.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2654(DATA *data, threadData_t *threadData);


/*
equation index: 147
type: SIMPLE_ASSIGN
y[10] = r_init[10] * sin(theta[10] + armOffset[10])
*/
void WhirlpoolDiskStars_eqFunction_147(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,147};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[649]] /* y[10] STATE(1,vy[10]) */) = ((data->simulationInfo->realParameter[174] /* r_init[10] PARAM */)) * (sin((data->simulationInfo->realParameter[334] /* theta[10] PARAM */) + (data->simulationInfo->realParameter[12] /* armOffset[10] PARAM */)));
  TRACE_POP
}

/*
equation index: 148
type: SIMPLE_ASSIGN
vx[10] = (-y[10]) * sqrt(G * Md / r_init[10] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_148(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,148};
  modelica_real tmp36;
  modelica_real tmp37;
  tmp36 = (data->simulationInfo->realParameter[174] /* r_init[10] PARAM */);
  tmp37 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp36 * tmp36 * tmp36),"r_init[10] ^ 3.0",equationIndexes);
  if(!(tmp37 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[10] ^ 3.0) was %g should be >= 0", tmp37);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* vx[10] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[649]] /* y[10] STATE(1,vy[10]) */))) * (sqrt(tmp37));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2653(DATA *data, threadData_t *threadData);


/*
equation index: 150
type: SIMPLE_ASSIGN
x[10] = r_init[10] * cos(theta[10] + armOffset[10])
*/
void WhirlpoolDiskStars_eqFunction_150(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,150};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[489]] /* x[10] STATE(1,vx[10]) */) = ((data->simulationInfo->realParameter[174] /* r_init[10] PARAM */)) * (cos((data->simulationInfo->realParameter[334] /* theta[10] PARAM */) + (data->simulationInfo->realParameter[12] /* armOffset[10] PARAM */)));
  TRACE_POP
}

/*
equation index: 151
type: SIMPLE_ASSIGN
vy[10] = x[10] * sqrt(G * Md / r_init[10] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,151};
  modelica_real tmp38;
  modelica_real tmp39;
  tmp38 = (data->simulationInfo->realParameter[174] /* r_init[10] PARAM */);
  tmp39 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp38 * tmp38 * tmp38),"r_init[10] ^ 3.0",equationIndexes);
  if(!(tmp39 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[10] ^ 3.0) was %g should be >= 0", tmp39);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[169]] /* vy[10] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[489]] /* x[10] STATE(1,vx[10]) */)) * (sqrt(tmp39));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2652(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2655(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2657(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2660(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2659(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2658(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2656(DATA *data, threadData_t *threadData);


/*
equation index: 159
type: SIMPLE_ASSIGN
vz[10] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,159};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[329]] /* vz[10] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2651(DATA *data, threadData_t *threadData);


/*
equation index: 161
type: SIMPLE_ASSIGN
z[11] = -3.45
*/
void WhirlpoolDiskStars_eqFunction_161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,161};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[810]] /* z[11] STATE(1,vz[11]) */) = -3.45;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2664(DATA *data, threadData_t *threadData);


/*
equation index: 163
type: SIMPLE_ASSIGN
y[11] = r_init[11] * sin(theta[11] + armOffset[11])
*/
void WhirlpoolDiskStars_eqFunction_163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,163};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[650]] /* y[11] STATE(1,vy[11]) */) = ((data->simulationInfo->realParameter[175] /* r_init[11] PARAM */)) * (sin((data->simulationInfo->realParameter[335] /* theta[11] PARAM */) + (data->simulationInfo->realParameter[13] /* armOffset[11] PARAM */)));
  TRACE_POP
}

/*
equation index: 164
type: SIMPLE_ASSIGN
vx[11] = (-y[11]) * sqrt(G * Md / r_init[11] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,164};
  modelica_real tmp40;
  modelica_real tmp41;
  tmp40 = (data->simulationInfo->realParameter[175] /* r_init[11] PARAM */);
  tmp41 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp40 * tmp40 * tmp40),"r_init[11] ^ 3.0",equationIndexes);
  if(!(tmp41 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[11] ^ 3.0) was %g should be >= 0", tmp41);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* vx[11] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[650]] /* y[11] STATE(1,vy[11]) */))) * (sqrt(tmp41));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2663(DATA *data, threadData_t *threadData);


/*
equation index: 166
type: SIMPLE_ASSIGN
x[11] = r_init[11] * cos(theta[11] + armOffset[11])
*/
void WhirlpoolDiskStars_eqFunction_166(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,166};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[490]] /* x[11] STATE(1,vx[11]) */) = ((data->simulationInfo->realParameter[175] /* r_init[11] PARAM */)) * (cos((data->simulationInfo->realParameter[335] /* theta[11] PARAM */) + (data->simulationInfo->realParameter[13] /* armOffset[11] PARAM */)));
  TRACE_POP
}

/*
equation index: 167
type: SIMPLE_ASSIGN
vy[11] = x[11] * sqrt(G * Md / r_init[11] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,167};
  modelica_real tmp42;
  modelica_real tmp43;
  tmp42 = (data->simulationInfo->realParameter[175] /* r_init[11] PARAM */);
  tmp43 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp42 * tmp42 * tmp42),"r_init[11] ^ 3.0",equationIndexes);
  if(!(tmp43 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[11] ^ 3.0) was %g should be >= 0", tmp43);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[170]] /* vy[11] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[490]] /* x[11] STATE(1,vx[11]) */)) * (sqrt(tmp43));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2662(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2665(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2667(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2670(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2669(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2668(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2666(DATA *data, threadData_t *threadData);


/*
equation index: 175
type: SIMPLE_ASSIGN
vz[11] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,175};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[330]] /* vz[11] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2661(DATA *data, threadData_t *threadData);


/*
equation index: 177
type: SIMPLE_ASSIGN
z[12] = -3.4000000000000004
*/
void WhirlpoolDiskStars_eqFunction_177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,177};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[811]] /* z[12] STATE(1,vz[12]) */) = -3.4000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2674(DATA *data, threadData_t *threadData);


/*
equation index: 179
type: SIMPLE_ASSIGN
y[12] = r_init[12] * sin(theta[12] + armOffset[12])
*/
void WhirlpoolDiskStars_eqFunction_179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,179};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[651]] /* y[12] STATE(1,vy[12]) */) = ((data->simulationInfo->realParameter[176] /* r_init[12] PARAM */)) * (sin((data->simulationInfo->realParameter[336] /* theta[12] PARAM */) + (data->simulationInfo->realParameter[14] /* armOffset[12] PARAM */)));
  TRACE_POP
}

/*
equation index: 180
type: SIMPLE_ASSIGN
vx[12] = (-y[12]) * sqrt(G * Md / r_init[12] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,180};
  modelica_real tmp44;
  modelica_real tmp45;
  tmp44 = (data->simulationInfo->realParameter[176] /* r_init[12] PARAM */);
  tmp45 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp44 * tmp44 * tmp44),"r_init[12] ^ 3.0",equationIndexes);
  if(!(tmp45 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[12] ^ 3.0) was %g should be >= 0", tmp45);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* vx[12] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[651]] /* y[12] STATE(1,vy[12]) */))) * (sqrt(tmp45));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2673(DATA *data, threadData_t *threadData);


/*
equation index: 182
type: SIMPLE_ASSIGN
x[12] = r_init[12] * cos(theta[12] + armOffset[12])
*/
void WhirlpoolDiskStars_eqFunction_182(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,182};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[491]] /* x[12] STATE(1,vx[12]) */) = ((data->simulationInfo->realParameter[176] /* r_init[12] PARAM */)) * (cos((data->simulationInfo->realParameter[336] /* theta[12] PARAM */) + (data->simulationInfo->realParameter[14] /* armOffset[12] PARAM */)));
  TRACE_POP
}

/*
equation index: 183
type: SIMPLE_ASSIGN
vy[12] = x[12] * sqrt(G * Md / r_init[12] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,183};
  modelica_real tmp46;
  modelica_real tmp47;
  tmp46 = (data->simulationInfo->realParameter[176] /* r_init[12] PARAM */);
  tmp47 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp46 * tmp46 * tmp46),"r_init[12] ^ 3.0",equationIndexes);
  if(!(tmp47 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[12] ^ 3.0) was %g should be >= 0", tmp47);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[171]] /* vy[12] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[491]] /* x[12] STATE(1,vx[12]) */)) * (sqrt(tmp47));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2672(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2675(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2677(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2680(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2679(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2678(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2676(DATA *data, threadData_t *threadData);


/*
equation index: 191
type: SIMPLE_ASSIGN
vz[12] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,191};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[331]] /* vz[12] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2671(DATA *data, threadData_t *threadData);


/*
equation index: 193
type: SIMPLE_ASSIGN
z[13] = -3.35
*/
void WhirlpoolDiskStars_eqFunction_193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,193};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[812]] /* z[13] STATE(1,vz[13]) */) = -3.35;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2684(DATA *data, threadData_t *threadData);


/*
equation index: 195
type: SIMPLE_ASSIGN
y[13] = r_init[13] * sin(theta[13] + armOffset[13])
*/
void WhirlpoolDiskStars_eqFunction_195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,195};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[652]] /* y[13] STATE(1,vy[13]) */) = ((data->simulationInfo->realParameter[177] /* r_init[13] PARAM */)) * (sin((data->simulationInfo->realParameter[337] /* theta[13] PARAM */) + (data->simulationInfo->realParameter[15] /* armOffset[13] PARAM */)));
  TRACE_POP
}

/*
equation index: 196
type: SIMPLE_ASSIGN
vx[13] = (-y[13]) * sqrt(G * Md / r_init[13] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,196};
  modelica_real tmp48;
  modelica_real tmp49;
  tmp48 = (data->simulationInfo->realParameter[177] /* r_init[13] PARAM */);
  tmp49 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp48 * tmp48 * tmp48),"r_init[13] ^ 3.0",equationIndexes);
  if(!(tmp49 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[13] ^ 3.0) was %g should be >= 0", tmp49);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* vx[13] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[652]] /* y[13] STATE(1,vy[13]) */))) * (sqrt(tmp49));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2683(DATA *data, threadData_t *threadData);


/*
equation index: 198
type: SIMPLE_ASSIGN
x[13] = r_init[13] * cos(theta[13] + armOffset[13])
*/
void WhirlpoolDiskStars_eqFunction_198(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,198};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[492]] /* x[13] STATE(1,vx[13]) */) = ((data->simulationInfo->realParameter[177] /* r_init[13] PARAM */)) * (cos((data->simulationInfo->realParameter[337] /* theta[13] PARAM */) + (data->simulationInfo->realParameter[15] /* armOffset[13] PARAM */)));
  TRACE_POP
}

/*
equation index: 199
type: SIMPLE_ASSIGN
vy[13] = x[13] * sqrt(G * Md / r_init[13] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,199};
  modelica_real tmp50;
  modelica_real tmp51;
  tmp50 = (data->simulationInfo->realParameter[177] /* r_init[13] PARAM */);
  tmp51 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp50 * tmp50 * tmp50),"r_init[13] ^ 3.0",equationIndexes);
  if(!(tmp51 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[13] ^ 3.0) was %g should be >= 0", tmp51);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[172]] /* vy[13] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[492]] /* x[13] STATE(1,vx[13]) */)) * (sqrt(tmp51));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2682(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2685(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2687(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2690(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2689(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2688(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2686(DATA *data, threadData_t *threadData);


/*
equation index: 207
type: SIMPLE_ASSIGN
vz[13] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,207};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[332]] /* vz[13] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2681(DATA *data, threadData_t *threadData);


/*
equation index: 209
type: SIMPLE_ASSIGN
z[14] = -3.3000000000000003
*/
void WhirlpoolDiskStars_eqFunction_209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,209};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[813]] /* z[14] STATE(1,vz[14]) */) = -3.3000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2694(DATA *data, threadData_t *threadData);


/*
equation index: 211
type: SIMPLE_ASSIGN
y[14] = r_init[14] * sin(theta[14] + armOffset[14])
*/
void WhirlpoolDiskStars_eqFunction_211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,211};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[653]] /* y[14] STATE(1,vy[14]) */) = ((data->simulationInfo->realParameter[178] /* r_init[14] PARAM */)) * (sin((data->simulationInfo->realParameter[338] /* theta[14] PARAM */) + (data->simulationInfo->realParameter[16] /* armOffset[14] PARAM */)));
  TRACE_POP
}

/*
equation index: 212
type: SIMPLE_ASSIGN
vx[14] = (-y[14]) * sqrt(G * Md / r_init[14] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,212};
  modelica_real tmp52;
  modelica_real tmp53;
  tmp52 = (data->simulationInfo->realParameter[178] /* r_init[14] PARAM */);
  tmp53 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp52 * tmp52 * tmp52),"r_init[14] ^ 3.0",equationIndexes);
  if(!(tmp53 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[14] ^ 3.0) was %g should be >= 0", tmp53);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* vx[14] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[653]] /* y[14] STATE(1,vy[14]) */))) * (sqrt(tmp53));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2693(DATA *data, threadData_t *threadData);


/*
equation index: 214
type: SIMPLE_ASSIGN
x[14] = r_init[14] * cos(theta[14] + armOffset[14])
*/
void WhirlpoolDiskStars_eqFunction_214(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,214};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[493]] /* x[14] STATE(1,vx[14]) */) = ((data->simulationInfo->realParameter[178] /* r_init[14] PARAM */)) * (cos((data->simulationInfo->realParameter[338] /* theta[14] PARAM */) + (data->simulationInfo->realParameter[16] /* armOffset[14] PARAM */)));
  TRACE_POP
}

/*
equation index: 215
type: SIMPLE_ASSIGN
vy[14] = x[14] * sqrt(G * Md / r_init[14] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,215};
  modelica_real tmp54;
  modelica_real tmp55;
  tmp54 = (data->simulationInfo->realParameter[178] /* r_init[14] PARAM */);
  tmp55 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp54 * tmp54 * tmp54),"r_init[14] ^ 3.0",equationIndexes);
  if(!(tmp55 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[14] ^ 3.0) was %g should be >= 0", tmp55);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[173]] /* vy[14] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[493]] /* x[14] STATE(1,vx[14]) */)) * (sqrt(tmp55));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2692(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2695(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2697(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2700(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2699(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2698(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2696(DATA *data, threadData_t *threadData);


/*
equation index: 223
type: SIMPLE_ASSIGN
vz[14] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,223};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[333]] /* vz[14] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2691(DATA *data, threadData_t *threadData);


/*
equation index: 225
type: SIMPLE_ASSIGN
z[15] = -3.25
*/
void WhirlpoolDiskStars_eqFunction_225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,225};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[814]] /* z[15] STATE(1,vz[15]) */) = -3.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2704(DATA *data, threadData_t *threadData);


/*
equation index: 227
type: SIMPLE_ASSIGN
y[15] = r_init[15] * sin(theta[15] + armOffset[15])
*/
void WhirlpoolDiskStars_eqFunction_227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,227};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[654]] /* y[15] STATE(1,vy[15]) */) = ((data->simulationInfo->realParameter[179] /* r_init[15] PARAM */)) * (sin((data->simulationInfo->realParameter[339] /* theta[15] PARAM */) + (data->simulationInfo->realParameter[17] /* armOffset[15] PARAM */)));
  TRACE_POP
}

/*
equation index: 228
type: SIMPLE_ASSIGN
vx[15] = (-y[15]) * sqrt(G * Md / r_init[15] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,228};
  modelica_real tmp56;
  modelica_real tmp57;
  tmp56 = (data->simulationInfo->realParameter[179] /* r_init[15] PARAM */);
  tmp57 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp56 * tmp56 * tmp56),"r_init[15] ^ 3.0",equationIndexes);
  if(!(tmp57 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[15] ^ 3.0) was %g should be >= 0", tmp57);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* vx[15] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[654]] /* y[15] STATE(1,vy[15]) */))) * (sqrt(tmp57));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2703(DATA *data, threadData_t *threadData);


/*
equation index: 230
type: SIMPLE_ASSIGN
x[15] = r_init[15] * cos(theta[15] + armOffset[15])
*/
void WhirlpoolDiskStars_eqFunction_230(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,230};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[494]] /* x[15] STATE(1,vx[15]) */) = ((data->simulationInfo->realParameter[179] /* r_init[15] PARAM */)) * (cos((data->simulationInfo->realParameter[339] /* theta[15] PARAM */) + (data->simulationInfo->realParameter[17] /* armOffset[15] PARAM */)));
  TRACE_POP
}

/*
equation index: 231
type: SIMPLE_ASSIGN
vy[15] = x[15] * sqrt(G * Md / r_init[15] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,231};
  modelica_real tmp58;
  modelica_real tmp59;
  tmp58 = (data->simulationInfo->realParameter[179] /* r_init[15] PARAM */);
  tmp59 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp58 * tmp58 * tmp58),"r_init[15] ^ 3.0",equationIndexes);
  if(!(tmp59 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[15] ^ 3.0) was %g should be >= 0", tmp59);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[174]] /* vy[15] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[494]] /* x[15] STATE(1,vx[15]) */)) * (sqrt(tmp59));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2702(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2705(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2707(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2710(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2709(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2708(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2706(DATA *data, threadData_t *threadData);


/*
equation index: 239
type: SIMPLE_ASSIGN
vz[15] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,239};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[334]] /* vz[15] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2701(DATA *data, threadData_t *threadData);


/*
equation index: 241
type: SIMPLE_ASSIGN
z[16] = -3.2
*/
void WhirlpoolDiskStars_eqFunction_241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,241};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[815]] /* z[16] STATE(1,vz[16]) */) = -3.2;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2714(DATA *data, threadData_t *threadData);


/*
equation index: 243
type: SIMPLE_ASSIGN
y[16] = r_init[16] * sin(theta[16] + armOffset[16])
*/
void WhirlpoolDiskStars_eqFunction_243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,243};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[655]] /* y[16] STATE(1,vy[16]) */) = ((data->simulationInfo->realParameter[180] /* r_init[16] PARAM */)) * (sin((data->simulationInfo->realParameter[340] /* theta[16] PARAM */) + (data->simulationInfo->realParameter[18] /* armOffset[16] PARAM */)));
  TRACE_POP
}

/*
equation index: 244
type: SIMPLE_ASSIGN
vx[16] = (-y[16]) * sqrt(G * Md / r_init[16] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,244};
  modelica_real tmp60;
  modelica_real tmp61;
  tmp60 = (data->simulationInfo->realParameter[180] /* r_init[16] PARAM */);
  tmp61 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp60 * tmp60 * tmp60),"r_init[16] ^ 3.0",equationIndexes);
  if(!(tmp61 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[16] ^ 3.0) was %g should be >= 0", tmp61);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* vx[16] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[655]] /* y[16] STATE(1,vy[16]) */))) * (sqrt(tmp61));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2713(DATA *data, threadData_t *threadData);


/*
equation index: 246
type: SIMPLE_ASSIGN
x[16] = r_init[16] * cos(theta[16] + armOffset[16])
*/
void WhirlpoolDiskStars_eqFunction_246(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,246};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[495]] /* x[16] STATE(1,vx[16]) */) = ((data->simulationInfo->realParameter[180] /* r_init[16] PARAM */)) * (cos((data->simulationInfo->realParameter[340] /* theta[16] PARAM */) + (data->simulationInfo->realParameter[18] /* armOffset[16] PARAM */)));
  TRACE_POP
}

/*
equation index: 247
type: SIMPLE_ASSIGN
vy[16] = x[16] * sqrt(G * Md / r_init[16] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,247};
  modelica_real tmp62;
  modelica_real tmp63;
  tmp62 = (data->simulationInfo->realParameter[180] /* r_init[16] PARAM */);
  tmp63 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp62 * tmp62 * tmp62),"r_init[16] ^ 3.0",equationIndexes);
  if(!(tmp63 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[16] ^ 3.0) was %g should be >= 0", tmp63);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[175]] /* vy[16] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[495]] /* x[16] STATE(1,vx[16]) */)) * (sqrt(tmp63));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2712(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2715(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2717(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2720(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2719(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2718(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2716(DATA *data, threadData_t *threadData);


/*
equation index: 255
type: SIMPLE_ASSIGN
vz[16] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,255};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[335]] /* vz[16] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2711(DATA *data, threadData_t *threadData);


/*
equation index: 257
type: SIMPLE_ASSIGN
z[17] = -3.1500000000000004
*/
void WhirlpoolDiskStars_eqFunction_257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,257};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[816]] /* z[17] STATE(1,vz[17]) */) = -3.1500000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2724(DATA *data, threadData_t *threadData);


/*
equation index: 259
type: SIMPLE_ASSIGN
y[17] = r_init[17] * sin(theta[17] + armOffset[17])
*/
void WhirlpoolDiskStars_eqFunction_259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,259};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[656]] /* y[17] STATE(1,vy[17]) */) = ((data->simulationInfo->realParameter[181] /* r_init[17] PARAM */)) * (sin((data->simulationInfo->realParameter[341] /* theta[17] PARAM */) + (data->simulationInfo->realParameter[19] /* armOffset[17] PARAM */)));
  TRACE_POP
}

/*
equation index: 260
type: SIMPLE_ASSIGN
vx[17] = (-y[17]) * sqrt(G * Md / r_init[17] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,260};
  modelica_real tmp64;
  modelica_real tmp65;
  tmp64 = (data->simulationInfo->realParameter[181] /* r_init[17] PARAM */);
  tmp65 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp64 * tmp64 * tmp64),"r_init[17] ^ 3.0",equationIndexes);
  if(!(tmp65 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[17] ^ 3.0) was %g should be >= 0", tmp65);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* vx[17] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[656]] /* y[17] STATE(1,vy[17]) */))) * (sqrt(tmp65));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2723(DATA *data, threadData_t *threadData);


/*
equation index: 262
type: SIMPLE_ASSIGN
x[17] = r_init[17] * cos(theta[17] + armOffset[17])
*/
void WhirlpoolDiskStars_eqFunction_262(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,262};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[496]] /* x[17] STATE(1,vx[17]) */) = ((data->simulationInfo->realParameter[181] /* r_init[17] PARAM */)) * (cos((data->simulationInfo->realParameter[341] /* theta[17] PARAM */) + (data->simulationInfo->realParameter[19] /* armOffset[17] PARAM */)));
  TRACE_POP
}

/*
equation index: 263
type: SIMPLE_ASSIGN
vy[17] = x[17] * sqrt(G * Md / r_init[17] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,263};
  modelica_real tmp66;
  modelica_real tmp67;
  tmp66 = (data->simulationInfo->realParameter[181] /* r_init[17] PARAM */);
  tmp67 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp66 * tmp66 * tmp66),"r_init[17] ^ 3.0",equationIndexes);
  if(!(tmp67 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[17] ^ 3.0) was %g should be >= 0", tmp67);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[176]] /* vy[17] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[496]] /* x[17] STATE(1,vx[17]) */)) * (sqrt(tmp67));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2722(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2725(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2727(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2730(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2729(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2728(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2726(DATA *data, threadData_t *threadData);


/*
equation index: 271
type: SIMPLE_ASSIGN
vz[17] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[336]] /* vz[17] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2721(DATA *data, threadData_t *threadData);


/*
equation index: 273
type: SIMPLE_ASSIGN
z[18] = -3.1
*/
void WhirlpoolDiskStars_eqFunction_273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[817]] /* z[18] STATE(1,vz[18]) */) = -3.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2734(DATA *data, threadData_t *threadData);


/*
equation index: 275
type: SIMPLE_ASSIGN
y[18] = r_init[18] * sin(theta[18] + armOffset[18])
*/
void WhirlpoolDiskStars_eqFunction_275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[657]] /* y[18] STATE(1,vy[18]) */) = ((data->simulationInfo->realParameter[182] /* r_init[18] PARAM */)) * (sin((data->simulationInfo->realParameter[342] /* theta[18] PARAM */) + (data->simulationInfo->realParameter[20] /* armOffset[18] PARAM */)));
  TRACE_POP
}

/*
equation index: 276
type: SIMPLE_ASSIGN
vx[18] = (-y[18]) * sqrt(G * Md / r_init[18] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,276};
  modelica_real tmp68;
  modelica_real tmp69;
  tmp68 = (data->simulationInfo->realParameter[182] /* r_init[18] PARAM */);
  tmp69 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp68 * tmp68 * tmp68),"r_init[18] ^ 3.0",equationIndexes);
  if(!(tmp69 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[18] ^ 3.0) was %g should be >= 0", tmp69);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* vx[18] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[657]] /* y[18] STATE(1,vy[18]) */))) * (sqrt(tmp69));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2733(DATA *data, threadData_t *threadData);


/*
equation index: 278
type: SIMPLE_ASSIGN
x[18] = r_init[18] * cos(theta[18] + armOffset[18])
*/
void WhirlpoolDiskStars_eqFunction_278(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,278};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[497]] /* x[18] STATE(1,vx[18]) */) = ((data->simulationInfo->realParameter[182] /* r_init[18] PARAM */)) * (cos((data->simulationInfo->realParameter[342] /* theta[18] PARAM */) + (data->simulationInfo->realParameter[20] /* armOffset[18] PARAM */)));
  TRACE_POP
}

/*
equation index: 279
type: SIMPLE_ASSIGN
vy[18] = x[18] * sqrt(G * Md / r_init[18] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,279};
  modelica_real tmp70;
  modelica_real tmp71;
  tmp70 = (data->simulationInfo->realParameter[182] /* r_init[18] PARAM */);
  tmp71 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp70 * tmp70 * tmp70),"r_init[18] ^ 3.0",equationIndexes);
  if(!(tmp71 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[18] ^ 3.0) was %g should be >= 0", tmp71);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[177]] /* vy[18] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[497]] /* x[18] STATE(1,vx[18]) */)) * (sqrt(tmp71));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2732(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2735(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2737(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2740(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2739(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2738(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2736(DATA *data, threadData_t *threadData);


/*
equation index: 287
type: SIMPLE_ASSIGN
vz[18] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,287};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[337]] /* vz[18] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2731(DATA *data, threadData_t *threadData);


/*
equation index: 289
type: SIMPLE_ASSIGN
z[19] = -3.0500000000000003
*/
void WhirlpoolDiskStars_eqFunction_289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,289};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[818]] /* z[19] STATE(1,vz[19]) */) = -3.0500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2744(DATA *data, threadData_t *threadData);


/*
equation index: 291
type: SIMPLE_ASSIGN
y[19] = r_init[19] * sin(theta[19] + armOffset[19])
*/
void WhirlpoolDiskStars_eqFunction_291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[658]] /* y[19] STATE(1,vy[19]) */) = ((data->simulationInfo->realParameter[183] /* r_init[19] PARAM */)) * (sin((data->simulationInfo->realParameter[343] /* theta[19] PARAM */) + (data->simulationInfo->realParameter[21] /* armOffset[19] PARAM */)));
  TRACE_POP
}

/*
equation index: 292
type: SIMPLE_ASSIGN
vx[19] = (-y[19]) * sqrt(G * Md / r_init[19] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,292};
  modelica_real tmp72;
  modelica_real tmp73;
  tmp72 = (data->simulationInfo->realParameter[183] /* r_init[19] PARAM */);
  tmp73 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp72 * tmp72 * tmp72),"r_init[19] ^ 3.0",equationIndexes);
  if(!(tmp73 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[19] ^ 3.0) was %g should be >= 0", tmp73);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* vx[19] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[658]] /* y[19] STATE(1,vy[19]) */))) * (sqrt(tmp73));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2743(DATA *data, threadData_t *threadData);


/*
equation index: 294
type: SIMPLE_ASSIGN
x[19] = r_init[19] * cos(theta[19] + armOffset[19])
*/
void WhirlpoolDiskStars_eqFunction_294(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,294};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[498]] /* x[19] STATE(1,vx[19]) */) = ((data->simulationInfo->realParameter[183] /* r_init[19] PARAM */)) * (cos((data->simulationInfo->realParameter[343] /* theta[19] PARAM */) + (data->simulationInfo->realParameter[21] /* armOffset[19] PARAM */)));
  TRACE_POP
}

/*
equation index: 295
type: SIMPLE_ASSIGN
vy[19] = x[19] * sqrt(G * Md / r_init[19] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,295};
  modelica_real tmp74;
  modelica_real tmp75;
  tmp74 = (data->simulationInfo->realParameter[183] /* r_init[19] PARAM */);
  tmp75 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp74 * tmp74 * tmp74),"r_init[19] ^ 3.0",equationIndexes);
  if(!(tmp75 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[19] ^ 3.0) was %g should be >= 0", tmp75);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[178]] /* vy[19] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[498]] /* x[19] STATE(1,vx[19]) */)) * (sqrt(tmp75));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2742(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2745(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2747(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2750(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2749(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2748(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2746(DATA *data, threadData_t *threadData);


/*
equation index: 303
type: SIMPLE_ASSIGN
vz[19] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,303};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[338]] /* vz[19] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2741(DATA *data, threadData_t *threadData);


/*
equation index: 305
type: SIMPLE_ASSIGN
z[20] = -3.0
*/
void WhirlpoolDiskStars_eqFunction_305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,305};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[819]] /* z[20] STATE(1,vz[20]) */) = -3.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2754(DATA *data, threadData_t *threadData);


/*
equation index: 307
type: SIMPLE_ASSIGN
y[20] = r_init[20] * sin(theta[20] + armOffset[20])
*/
void WhirlpoolDiskStars_eqFunction_307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,307};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[659]] /* y[20] STATE(1,vy[20]) */) = ((data->simulationInfo->realParameter[184] /* r_init[20] PARAM */)) * (sin((data->simulationInfo->realParameter[344] /* theta[20] PARAM */) + (data->simulationInfo->realParameter[22] /* armOffset[20] PARAM */)));
  TRACE_POP
}

/*
equation index: 308
type: SIMPLE_ASSIGN
vx[20] = (-y[20]) * sqrt(G * Md / r_init[20] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,308};
  modelica_real tmp76;
  modelica_real tmp77;
  tmp76 = (data->simulationInfo->realParameter[184] /* r_init[20] PARAM */);
  tmp77 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp76 * tmp76 * tmp76),"r_init[20] ^ 3.0",equationIndexes);
  if(!(tmp77 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[20] ^ 3.0) was %g should be >= 0", tmp77);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* vx[20] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[659]] /* y[20] STATE(1,vy[20]) */))) * (sqrt(tmp77));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2753(DATA *data, threadData_t *threadData);


/*
equation index: 310
type: SIMPLE_ASSIGN
x[20] = r_init[20] * cos(theta[20] + armOffset[20])
*/
void WhirlpoolDiskStars_eqFunction_310(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,310};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[499]] /* x[20] STATE(1,vx[20]) */) = ((data->simulationInfo->realParameter[184] /* r_init[20] PARAM */)) * (cos((data->simulationInfo->realParameter[344] /* theta[20] PARAM */) + (data->simulationInfo->realParameter[22] /* armOffset[20] PARAM */)));
  TRACE_POP
}

/*
equation index: 311
type: SIMPLE_ASSIGN
vy[20] = x[20] * sqrt(G * Md / r_init[20] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,311};
  modelica_real tmp78;
  modelica_real tmp79;
  tmp78 = (data->simulationInfo->realParameter[184] /* r_init[20] PARAM */);
  tmp79 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp78 * tmp78 * tmp78),"r_init[20] ^ 3.0",equationIndexes);
  if(!(tmp79 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[20] ^ 3.0) was %g should be >= 0", tmp79);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[179]] /* vy[20] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[499]] /* x[20] STATE(1,vx[20]) */)) * (sqrt(tmp79));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2752(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2755(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2757(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2760(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2759(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2758(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2756(DATA *data, threadData_t *threadData);


/*
equation index: 319
type: SIMPLE_ASSIGN
vz[20] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[339]] /* vz[20] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2751(DATA *data, threadData_t *threadData);


/*
equation index: 321
type: SIMPLE_ASSIGN
z[21] = -2.95
*/
void WhirlpoolDiskStars_eqFunction_321(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[820]] /* z[21] STATE(1,vz[21]) */) = -2.95;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2764(DATA *data, threadData_t *threadData);


/*
equation index: 323
type: SIMPLE_ASSIGN
y[21] = r_init[21] * sin(theta[21] + armOffset[21])
*/
void WhirlpoolDiskStars_eqFunction_323(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[660]] /* y[21] STATE(1,vy[21]) */) = ((data->simulationInfo->realParameter[185] /* r_init[21] PARAM */)) * (sin((data->simulationInfo->realParameter[345] /* theta[21] PARAM */) + (data->simulationInfo->realParameter[23] /* armOffset[21] PARAM */)));
  TRACE_POP
}

/*
equation index: 324
type: SIMPLE_ASSIGN
vx[21] = (-y[21]) * sqrt(G * Md / r_init[21] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_324(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,324};
  modelica_real tmp80;
  modelica_real tmp81;
  tmp80 = (data->simulationInfo->realParameter[185] /* r_init[21] PARAM */);
  tmp81 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp80 * tmp80 * tmp80),"r_init[21] ^ 3.0",equationIndexes);
  if(!(tmp81 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[21] ^ 3.0) was %g should be >= 0", tmp81);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* vx[21] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[660]] /* y[21] STATE(1,vy[21]) */))) * (sqrt(tmp81));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2763(DATA *data, threadData_t *threadData);


/*
equation index: 326
type: SIMPLE_ASSIGN
x[21] = r_init[21] * cos(theta[21] + armOffset[21])
*/
void WhirlpoolDiskStars_eqFunction_326(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,326};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[500]] /* x[21] STATE(1,vx[21]) */) = ((data->simulationInfo->realParameter[185] /* r_init[21] PARAM */)) * (cos((data->simulationInfo->realParameter[345] /* theta[21] PARAM */) + (data->simulationInfo->realParameter[23] /* armOffset[21] PARAM */)));
  TRACE_POP
}

/*
equation index: 327
type: SIMPLE_ASSIGN
vy[21] = x[21] * sqrt(G * Md / r_init[21] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_327(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,327};
  modelica_real tmp82;
  modelica_real tmp83;
  tmp82 = (data->simulationInfo->realParameter[185] /* r_init[21] PARAM */);
  tmp83 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp82 * tmp82 * tmp82),"r_init[21] ^ 3.0",equationIndexes);
  if(!(tmp83 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[21] ^ 3.0) was %g should be >= 0", tmp83);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[180]] /* vy[21] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[500]] /* x[21] STATE(1,vx[21]) */)) * (sqrt(tmp83));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2762(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2765(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2767(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2770(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2769(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2768(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2766(DATA *data, threadData_t *threadData);


/*
equation index: 335
type: SIMPLE_ASSIGN
vz[21] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_335(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[340]] /* vz[21] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2761(DATA *data, threadData_t *threadData);


/*
equation index: 337
type: SIMPLE_ASSIGN
z[22] = -2.9000000000000004
*/
void WhirlpoolDiskStars_eqFunction_337(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[821]] /* z[22] STATE(1,vz[22]) */) = -2.9000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2774(DATA *data, threadData_t *threadData);


/*
equation index: 339
type: SIMPLE_ASSIGN
y[22] = r_init[22] * sin(theta[22] + armOffset[22])
*/
void WhirlpoolDiskStars_eqFunction_339(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[661]] /* y[22] STATE(1,vy[22]) */) = ((data->simulationInfo->realParameter[186] /* r_init[22] PARAM */)) * (sin((data->simulationInfo->realParameter[346] /* theta[22] PARAM */) + (data->simulationInfo->realParameter[24] /* armOffset[22] PARAM */)));
  TRACE_POP
}

/*
equation index: 340
type: SIMPLE_ASSIGN
vx[22] = (-y[22]) * sqrt(G * Md / r_init[22] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_340(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,340};
  modelica_real tmp84;
  modelica_real tmp85;
  tmp84 = (data->simulationInfo->realParameter[186] /* r_init[22] PARAM */);
  tmp85 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp84 * tmp84 * tmp84),"r_init[22] ^ 3.0",equationIndexes);
  if(!(tmp85 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[22] ^ 3.0) was %g should be >= 0", tmp85);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* vx[22] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[661]] /* y[22] STATE(1,vy[22]) */))) * (sqrt(tmp85));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2773(DATA *data, threadData_t *threadData);


/*
equation index: 342
type: SIMPLE_ASSIGN
x[22] = r_init[22] * cos(theta[22] + armOffset[22])
*/
void WhirlpoolDiskStars_eqFunction_342(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,342};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[501]] /* x[22] STATE(1,vx[22]) */) = ((data->simulationInfo->realParameter[186] /* r_init[22] PARAM */)) * (cos((data->simulationInfo->realParameter[346] /* theta[22] PARAM */) + (data->simulationInfo->realParameter[24] /* armOffset[22] PARAM */)));
  TRACE_POP
}

/*
equation index: 343
type: SIMPLE_ASSIGN
vy[22] = x[22] * sqrt(G * Md / r_init[22] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_343(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,343};
  modelica_real tmp86;
  modelica_real tmp87;
  tmp86 = (data->simulationInfo->realParameter[186] /* r_init[22] PARAM */);
  tmp87 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp86 * tmp86 * tmp86),"r_init[22] ^ 3.0",equationIndexes);
  if(!(tmp87 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[22] ^ 3.0) was %g should be >= 0", tmp87);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[181]] /* vy[22] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[501]] /* x[22] STATE(1,vx[22]) */)) * (sqrt(tmp87));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2772(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2775(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2777(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2780(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2779(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2778(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2776(DATA *data, threadData_t *threadData);


/*
equation index: 351
type: SIMPLE_ASSIGN
vz[22] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_351(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,351};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[341]] /* vz[22] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2771(DATA *data, threadData_t *threadData);


/*
equation index: 353
type: SIMPLE_ASSIGN
z[23] = -2.85
*/
void WhirlpoolDiskStars_eqFunction_353(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,353};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[822]] /* z[23] STATE(1,vz[23]) */) = -2.85;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2784(DATA *data, threadData_t *threadData);


/*
equation index: 355
type: SIMPLE_ASSIGN
y[23] = r_init[23] * sin(theta[23] + armOffset[23])
*/
void WhirlpoolDiskStars_eqFunction_355(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[662]] /* y[23] STATE(1,vy[23]) */) = ((data->simulationInfo->realParameter[187] /* r_init[23] PARAM */)) * (sin((data->simulationInfo->realParameter[347] /* theta[23] PARAM */) + (data->simulationInfo->realParameter[25] /* armOffset[23] PARAM */)));
  TRACE_POP
}

/*
equation index: 356
type: SIMPLE_ASSIGN
vx[23] = (-y[23]) * sqrt(G * Md / r_init[23] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_356(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,356};
  modelica_real tmp88;
  modelica_real tmp89;
  tmp88 = (data->simulationInfo->realParameter[187] /* r_init[23] PARAM */);
  tmp89 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp88 * tmp88 * tmp88),"r_init[23] ^ 3.0",equationIndexes);
  if(!(tmp89 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[23] ^ 3.0) was %g should be >= 0", tmp89);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* vx[23] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[662]] /* y[23] STATE(1,vy[23]) */))) * (sqrt(tmp89));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2783(DATA *data, threadData_t *threadData);


/*
equation index: 358
type: SIMPLE_ASSIGN
x[23] = r_init[23] * cos(theta[23] + armOffset[23])
*/
void WhirlpoolDiskStars_eqFunction_358(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,358};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[502]] /* x[23] STATE(1,vx[23]) */) = ((data->simulationInfo->realParameter[187] /* r_init[23] PARAM */)) * (cos((data->simulationInfo->realParameter[347] /* theta[23] PARAM */) + (data->simulationInfo->realParameter[25] /* armOffset[23] PARAM */)));
  TRACE_POP
}

/*
equation index: 359
type: SIMPLE_ASSIGN
vy[23] = x[23] * sqrt(G * Md / r_init[23] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_359(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,359};
  modelica_real tmp90;
  modelica_real tmp91;
  tmp90 = (data->simulationInfo->realParameter[187] /* r_init[23] PARAM */);
  tmp91 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp90 * tmp90 * tmp90),"r_init[23] ^ 3.0",equationIndexes);
  if(!(tmp91 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[23] ^ 3.0) was %g should be >= 0", tmp91);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[182]] /* vy[23] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[502]] /* x[23] STATE(1,vx[23]) */)) * (sqrt(tmp91));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2782(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2785(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2787(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2790(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2789(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2788(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2786(DATA *data, threadData_t *threadData);


/*
equation index: 367
type: SIMPLE_ASSIGN
vz[23] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_367(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,367};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[342]] /* vz[23] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2781(DATA *data, threadData_t *threadData);


/*
equation index: 369
type: SIMPLE_ASSIGN
z[24] = -2.8000000000000003
*/
void WhirlpoolDiskStars_eqFunction_369(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,369};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[823]] /* z[24] STATE(1,vz[24]) */) = -2.8000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2794(DATA *data, threadData_t *threadData);


/*
equation index: 371
type: SIMPLE_ASSIGN
y[24] = r_init[24] * sin(theta[24] + armOffset[24])
*/
void WhirlpoolDiskStars_eqFunction_371(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,371};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[663]] /* y[24] STATE(1,vy[24]) */) = ((data->simulationInfo->realParameter[188] /* r_init[24] PARAM */)) * (sin((data->simulationInfo->realParameter[348] /* theta[24] PARAM */) + (data->simulationInfo->realParameter[26] /* armOffset[24] PARAM */)));
  TRACE_POP
}

/*
equation index: 372
type: SIMPLE_ASSIGN
vx[24] = (-y[24]) * sqrt(G * Md / r_init[24] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_372(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,372};
  modelica_real tmp92;
  modelica_real tmp93;
  tmp92 = (data->simulationInfo->realParameter[188] /* r_init[24] PARAM */);
  tmp93 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp92 * tmp92 * tmp92),"r_init[24] ^ 3.0",equationIndexes);
  if(!(tmp93 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[24] ^ 3.0) was %g should be >= 0", tmp93);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* vx[24] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[663]] /* y[24] STATE(1,vy[24]) */))) * (sqrt(tmp93));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2793(DATA *data, threadData_t *threadData);


/*
equation index: 374
type: SIMPLE_ASSIGN
x[24] = r_init[24] * cos(theta[24] + armOffset[24])
*/
void WhirlpoolDiskStars_eqFunction_374(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,374};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[503]] /* x[24] STATE(1,vx[24]) */) = ((data->simulationInfo->realParameter[188] /* r_init[24] PARAM */)) * (cos((data->simulationInfo->realParameter[348] /* theta[24] PARAM */) + (data->simulationInfo->realParameter[26] /* armOffset[24] PARAM */)));
  TRACE_POP
}

/*
equation index: 375
type: SIMPLE_ASSIGN
vy[24] = x[24] * sqrt(G * Md / r_init[24] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_375(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,375};
  modelica_real tmp94;
  modelica_real tmp95;
  tmp94 = (data->simulationInfo->realParameter[188] /* r_init[24] PARAM */);
  tmp95 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp94 * tmp94 * tmp94),"r_init[24] ^ 3.0",equationIndexes);
  if(!(tmp95 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[24] ^ 3.0) was %g should be >= 0", tmp95);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[183]] /* vy[24] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[503]] /* x[24] STATE(1,vx[24]) */)) * (sqrt(tmp95));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2792(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2795(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2797(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2800(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2799(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2798(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2796(DATA *data, threadData_t *threadData);


/*
equation index: 383
type: SIMPLE_ASSIGN
vz[24] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_383(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,383};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[343]] /* vz[24] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2791(DATA *data, threadData_t *threadData);


/*
equation index: 385
type: SIMPLE_ASSIGN
z[25] = -2.75
*/
void WhirlpoolDiskStars_eqFunction_385(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,385};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[824]] /* z[25] STATE(1,vz[25]) */) = -2.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2804(DATA *data, threadData_t *threadData);


/*
equation index: 387
type: SIMPLE_ASSIGN
y[25] = r_init[25] * sin(theta[25] + armOffset[25])
*/
void WhirlpoolDiskStars_eqFunction_387(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[664]] /* y[25] STATE(1,vy[25]) */) = ((data->simulationInfo->realParameter[189] /* r_init[25] PARAM */)) * (sin((data->simulationInfo->realParameter[349] /* theta[25] PARAM */) + (data->simulationInfo->realParameter[27] /* armOffset[25] PARAM */)));
  TRACE_POP
}

/*
equation index: 388
type: SIMPLE_ASSIGN
vx[25] = (-y[25]) * sqrt(G * Md / r_init[25] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_388(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,388};
  modelica_real tmp96;
  modelica_real tmp97;
  tmp96 = (data->simulationInfo->realParameter[189] /* r_init[25] PARAM */);
  tmp97 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp96 * tmp96 * tmp96),"r_init[25] ^ 3.0",equationIndexes);
  if(!(tmp97 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[25] ^ 3.0) was %g should be >= 0", tmp97);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* vx[25] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[664]] /* y[25] STATE(1,vy[25]) */))) * (sqrt(tmp97));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2803(DATA *data, threadData_t *threadData);


/*
equation index: 390
type: SIMPLE_ASSIGN
x[25] = r_init[25] * cos(theta[25] + armOffset[25])
*/
void WhirlpoolDiskStars_eqFunction_390(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,390};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[504]] /* x[25] STATE(1,vx[25]) */) = ((data->simulationInfo->realParameter[189] /* r_init[25] PARAM */)) * (cos((data->simulationInfo->realParameter[349] /* theta[25] PARAM */) + (data->simulationInfo->realParameter[27] /* armOffset[25] PARAM */)));
  TRACE_POP
}

/*
equation index: 391
type: SIMPLE_ASSIGN
vy[25] = x[25] * sqrt(G * Md / r_init[25] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_391(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,391};
  modelica_real tmp98;
  modelica_real tmp99;
  tmp98 = (data->simulationInfo->realParameter[189] /* r_init[25] PARAM */);
  tmp99 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp98 * tmp98 * tmp98),"r_init[25] ^ 3.0",equationIndexes);
  if(!(tmp99 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[25] ^ 3.0) was %g should be >= 0", tmp99);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[184]] /* vy[25] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[504]] /* x[25] STATE(1,vx[25]) */)) * (sqrt(tmp99));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2802(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2805(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2807(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2810(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2809(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2808(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2806(DATA *data, threadData_t *threadData);


/*
equation index: 399
type: SIMPLE_ASSIGN
vz[25] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_399(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,399};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[344]] /* vz[25] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2801(DATA *data, threadData_t *threadData);


/*
equation index: 401
type: SIMPLE_ASSIGN
z[26] = -2.7
*/
void WhirlpoolDiskStars_eqFunction_401(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,401};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[825]] /* z[26] STATE(1,vz[26]) */) = -2.7;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2814(DATA *data, threadData_t *threadData);


/*
equation index: 403
type: SIMPLE_ASSIGN
y[26] = r_init[26] * sin(theta[26] + armOffset[26])
*/
void WhirlpoolDiskStars_eqFunction_403(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[665]] /* y[26] STATE(1,vy[26]) */) = ((data->simulationInfo->realParameter[190] /* r_init[26] PARAM */)) * (sin((data->simulationInfo->realParameter[350] /* theta[26] PARAM */) + (data->simulationInfo->realParameter[28] /* armOffset[26] PARAM */)));
  TRACE_POP
}

/*
equation index: 404
type: SIMPLE_ASSIGN
vx[26] = (-y[26]) * sqrt(G * Md / r_init[26] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_404(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,404};
  modelica_real tmp100;
  modelica_real tmp101;
  tmp100 = (data->simulationInfo->realParameter[190] /* r_init[26] PARAM */);
  tmp101 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp100 * tmp100 * tmp100),"r_init[26] ^ 3.0",equationIndexes);
  if(!(tmp101 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[26] ^ 3.0) was %g should be >= 0", tmp101);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* vx[26] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[665]] /* y[26] STATE(1,vy[26]) */))) * (sqrt(tmp101));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2813(DATA *data, threadData_t *threadData);


/*
equation index: 406
type: SIMPLE_ASSIGN
x[26] = r_init[26] * cos(theta[26] + armOffset[26])
*/
void WhirlpoolDiskStars_eqFunction_406(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,406};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[505]] /* x[26] STATE(1,vx[26]) */) = ((data->simulationInfo->realParameter[190] /* r_init[26] PARAM */)) * (cos((data->simulationInfo->realParameter[350] /* theta[26] PARAM */) + (data->simulationInfo->realParameter[28] /* armOffset[26] PARAM */)));
  TRACE_POP
}

/*
equation index: 407
type: SIMPLE_ASSIGN
vy[26] = x[26] * sqrt(G * Md / r_init[26] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_407(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,407};
  modelica_real tmp102;
  modelica_real tmp103;
  tmp102 = (data->simulationInfo->realParameter[190] /* r_init[26] PARAM */);
  tmp103 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp102 * tmp102 * tmp102),"r_init[26] ^ 3.0",equationIndexes);
  if(!(tmp103 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[26] ^ 3.0) was %g should be >= 0", tmp103);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[185]] /* vy[26] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[505]] /* x[26] STATE(1,vx[26]) */)) * (sqrt(tmp103));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2812(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2815(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2817(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2820(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2819(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2818(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2816(DATA *data, threadData_t *threadData);


/*
equation index: 415
type: SIMPLE_ASSIGN
vz[26] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_415(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,415};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[345]] /* vz[26] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2811(DATA *data, threadData_t *threadData);


/*
equation index: 417
type: SIMPLE_ASSIGN
z[27] = -2.6500000000000004
*/
void WhirlpoolDiskStars_eqFunction_417(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,417};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[826]] /* z[27] STATE(1,vz[27]) */) = -2.6500000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2824(DATA *data, threadData_t *threadData);


/*
equation index: 419
type: SIMPLE_ASSIGN
y[27] = r_init[27] * sin(theta[27] + armOffset[27])
*/
void WhirlpoolDiskStars_eqFunction_419(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[666]] /* y[27] STATE(1,vy[27]) */) = ((data->simulationInfo->realParameter[191] /* r_init[27] PARAM */)) * (sin((data->simulationInfo->realParameter[351] /* theta[27] PARAM */) + (data->simulationInfo->realParameter[29] /* armOffset[27] PARAM */)));
  TRACE_POP
}

/*
equation index: 420
type: SIMPLE_ASSIGN
vx[27] = (-y[27]) * sqrt(G * Md / r_init[27] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_420(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,420};
  modelica_real tmp104;
  modelica_real tmp105;
  tmp104 = (data->simulationInfo->realParameter[191] /* r_init[27] PARAM */);
  tmp105 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp104 * tmp104 * tmp104),"r_init[27] ^ 3.0",equationIndexes);
  if(!(tmp105 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[27] ^ 3.0) was %g should be >= 0", tmp105);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* vx[27] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[666]] /* y[27] STATE(1,vy[27]) */))) * (sqrt(tmp105));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2823(DATA *data, threadData_t *threadData);


/*
equation index: 422
type: SIMPLE_ASSIGN
x[27] = r_init[27] * cos(theta[27] + armOffset[27])
*/
void WhirlpoolDiskStars_eqFunction_422(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,422};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[506]] /* x[27] STATE(1,vx[27]) */) = ((data->simulationInfo->realParameter[191] /* r_init[27] PARAM */)) * (cos((data->simulationInfo->realParameter[351] /* theta[27] PARAM */) + (data->simulationInfo->realParameter[29] /* armOffset[27] PARAM */)));
  TRACE_POP
}

/*
equation index: 423
type: SIMPLE_ASSIGN
vy[27] = x[27] * sqrt(G * Md / r_init[27] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_423(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,423};
  modelica_real tmp106;
  modelica_real tmp107;
  tmp106 = (data->simulationInfo->realParameter[191] /* r_init[27] PARAM */);
  tmp107 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp106 * tmp106 * tmp106),"r_init[27] ^ 3.0",equationIndexes);
  if(!(tmp107 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[27] ^ 3.0) was %g should be >= 0", tmp107);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[186]] /* vy[27] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[506]] /* x[27] STATE(1,vx[27]) */)) * (sqrt(tmp107));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_2822(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2825(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2827(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_2830(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void WhirlpoolDiskStars_functionInitialEquations_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_1(data, threadData);
  WhirlpoolDiskStars_eqFunction_2564(data, threadData);
  WhirlpoolDiskStars_eqFunction_3(data, threadData);
  WhirlpoolDiskStars_eqFunction_4(data, threadData);
  WhirlpoolDiskStars_eqFunction_2563(data, threadData);
  WhirlpoolDiskStars_eqFunction_6(data, threadData);
  WhirlpoolDiskStars_eqFunction_7(data, threadData);
  WhirlpoolDiskStars_eqFunction_2562(data, threadData);
  WhirlpoolDiskStars_eqFunction_2565(data, threadData);
  WhirlpoolDiskStars_eqFunction_2567(data, threadData);
  WhirlpoolDiskStars_eqFunction_2570(data, threadData);
  WhirlpoolDiskStars_eqFunction_2569(data, threadData);
  WhirlpoolDiskStars_eqFunction_2568(data, threadData);
  WhirlpoolDiskStars_eqFunction_2566(data, threadData);
  WhirlpoolDiskStars_eqFunction_15(data, threadData);
  WhirlpoolDiskStars_eqFunction_2561(data, threadData);
  WhirlpoolDiskStars_eqFunction_17(data, threadData);
  WhirlpoolDiskStars_eqFunction_2574(data, threadData);
  WhirlpoolDiskStars_eqFunction_19(data, threadData);
  WhirlpoolDiskStars_eqFunction_20(data, threadData);
  WhirlpoolDiskStars_eqFunction_2573(data, threadData);
  WhirlpoolDiskStars_eqFunction_22(data, threadData);
  WhirlpoolDiskStars_eqFunction_23(data, threadData);
  WhirlpoolDiskStars_eqFunction_2572(data, threadData);
  WhirlpoolDiskStars_eqFunction_2575(data, threadData);
  WhirlpoolDiskStars_eqFunction_2577(data, threadData);
  WhirlpoolDiskStars_eqFunction_2580(data, threadData);
  WhirlpoolDiskStars_eqFunction_2579(data, threadData);
  WhirlpoolDiskStars_eqFunction_2578(data, threadData);
  WhirlpoolDiskStars_eqFunction_2576(data, threadData);
  WhirlpoolDiskStars_eqFunction_31(data, threadData);
  WhirlpoolDiskStars_eqFunction_2571(data, threadData);
  WhirlpoolDiskStars_eqFunction_33(data, threadData);
  WhirlpoolDiskStars_eqFunction_2584(data, threadData);
  WhirlpoolDiskStars_eqFunction_35(data, threadData);
  WhirlpoolDiskStars_eqFunction_36(data, threadData);
  WhirlpoolDiskStars_eqFunction_2583(data, threadData);
  WhirlpoolDiskStars_eqFunction_38(data, threadData);
  WhirlpoolDiskStars_eqFunction_39(data, threadData);
  WhirlpoolDiskStars_eqFunction_2582(data, threadData);
  WhirlpoolDiskStars_eqFunction_2585(data, threadData);
  WhirlpoolDiskStars_eqFunction_2587(data, threadData);
  WhirlpoolDiskStars_eqFunction_2590(data, threadData);
  WhirlpoolDiskStars_eqFunction_2589(data, threadData);
  WhirlpoolDiskStars_eqFunction_2588(data, threadData);
  WhirlpoolDiskStars_eqFunction_2586(data, threadData);
  WhirlpoolDiskStars_eqFunction_47(data, threadData);
  WhirlpoolDiskStars_eqFunction_2581(data, threadData);
  WhirlpoolDiskStars_eqFunction_49(data, threadData);
  WhirlpoolDiskStars_eqFunction_2594(data, threadData);
  WhirlpoolDiskStars_eqFunction_51(data, threadData);
  WhirlpoolDiskStars_eqFunction_52(data, threadData);
  WhirlpoolDiskStars_eqFunction_2593(data, threadData);
  WhirlpoolDiskStars_eqFunction_54(data, threadData);
  WhirlpoolDiskStars_eqFunction_55(data, threadData);
  WhirlpoolDiskStars_eqFunction_2592(data, threadData);
  WhirlpoolDiskStars_eqFunction_2595(data, threadData);
  WhirlpoolDiskStars_eqFunction_2597(data, threadData);
  WhirlpoolDiskStars_eqFunction_2600(data, threadData);
  WhirlpoolDiskStars_eqFunction_2599(data, threadData);
  WhirlpoolDiskStars_eqFunction_2598(data, threadData);
  WhirlpoolDiskStars_eqFunction_2596(data, threadData);
  WhirlpoolDiskStars_eqFunction_63(data, threadData);
  WhirlpoolDiskStars_eqFunction_2591(data, threadData);
  WhirlpoolDiskStars_eqFunction_65(data, threadData);
  WhirlpoolDiskStars_eqFunction_2604(data, threadData);
  WhirlpoolDiskStars_eqFunction_67(data, threadData);
  WhirlpoolDiskStars_eqFunction_68(data, threadData);
  WhirlpoolDiskStars_eqFunction_2603(data, threadData);
  WhirlpoolDiskStars_eqFunction_70(data, threadData);
  WhirlpoolDiskStars_eqFunction_71(data, threadData);
  WhirlpoolDiskStars_eqFunction_2602(data, threadData);
  WhirlpoolDiskStars_eqFunction_2605(data, threadData);
  WhirlpoolDiskStars_eqFunction_2607(data, threadData);
  WhirlpoolDiskStars_eqFunction_2610(data, threadData);
  WhirlpoolDiskStars_eqFunction_2609(data, threadData);
  WhirlpoolDiskStars_eqFunction_2608(data, threadData);
  WhirlpoolDiskStars_eqFunction_2606(data, threadData);
  WhirlpoolDiskStars_eqFunction_79(data, threadData);
  WhirlpoolDiskStars_eqFunction_2601(data, threadData);
  WhirlpoolDiskStars_eqFunction_81(data, threadData);
  WhirlpoolDiskStars_eqFunction_2614(data, threadData);
  WhirlpoolDiskStars_eqFunction_83(data, threadData);
  WhirlpoolDiskStars_eqFunction_84(data, threadData);
  WhirlpoolDiskStars_eqFunction_2613(data, threadData);
  WhirlpoolDiskStars_eqFunction_86(data, threadData);
  WhirlpoolDiskStars_eqFunction_87(data, threadData);
  WhirlpoolDiskStars_eqFunction_2612(data, threadData);
  WhirlpoolDiskStars_eqFunction_2615(data, threadData);
  WhirlpoolDiskStars_eqFunction_2617(data, threadData);
  WhirlpoolDiskStars_eqFunction_2620(data, threadData);
  WhirlpoolDiskStars_eqFunction_2619(data, threadData);
  WhirlpoolDiskStars_eqFunction_2618(data, threadData);
  WhirlpoolDiskStars_eqFunction_2616(data, threadData);
  WhirlpoolDiskStars_eqFunction_95(data, threadData);
  WhirlpoolDiskStars_eqFunction_2611(data, threadData);
  WhirlpoolDiskStars_eqFunction_97(data, threadData);
  WhirlpoolDiskStars_eqFunction_2624(data, threadData);
  WhirlpoolDiskStars_eqFunction_99(data, threadData);
  WhirlpoolDiskStars_eqFunction_100(data, threadData);
  WhirlpoolDiskStars_eqFunction_2623(data, threadData);
  WhirlpoolDiskStars_eqFunction_102(data, threadData);
  WhirlpoolDiskStars_eqFunction_103(data, threadData);
  WhirlpoolDiskStars_eqFunction_2622(data, threadData);
  WhirlpoolDiskStars_eqFunction_2625(data, threadData);
  WhirlpoolDiskStars_eqFunction_2627(data, threadData);
  WhirlpoolDiskStars_eqFunction_2630(data, threadData);
  WhirlpoolDiskStars_eqFunction_2629(data, threadData);
  WhirlpoolDiskStars_eqFunction_2628(data, threadData);
  WhirlpoolDiskStars_eqFunction_2626(data, threadData);
  WhirlpoolDiskStars_eqFunction_111(data, threadData);
  WhirlpoolDiskStars_eqFunction_2621(data, threadData);
  WhirlpoolDiskStars_eqFunction_113(data, threadData);
  WhirlpoolDiskStars_eqFunction_2634(data, threadData);
  WhirlpoolDiskStars_eqFunction_115(data, threadData);
  WhirlpoolDiskStars_eqFunction_116(data, threadData);
  WhirlpoolDiskStars_eqFunction_2633(data, threadData);
  WhirlpoolDiskStars_eqFunction_118(data, threadData);
  WhirlpoolDiskStars_eqFunction_119(data, threadData);
  WhirlpoolDiskStars_eqFunction_2632(data, threadData);
  WhirlpoolDiskStars_eqFunction_2635(data, threadData);
  WhirlpoolDiskStars_eqFunction_2637(data, threadData);
  WhirlpoolDiskStars_eqFunction_2640(data, threadData);
  WhirlpoolDiskStars_eqFunction_2639(data, threadData);
  WhirlpoolDiskStars_eqFunction_2638(data, threadData);
  WhirlpoolDiskStars_eqFunction_2636(data, threadData);
  WhirlpoolDiskStars_eqFunction_127(data, threadData);
  WhirlpoolDiskStars_eqFunction_2631(data, threadData);
  WhirlpoolDiskStars_eqFunction_129(data, threadData);
  WhirlpoolDiskStars_eqFunction_2644(data, threadData);
  WhirlpoolDiskStars_eqFunction_131(data, threadData);
  WhirlpoolDiskStars_eqFunction_132(data, threadData);
  WhirlpoolDiskStars_eqFunction_2643(data, threadData);
  WhirlpoolDiskStars_eqFunction_134(data, threadData);
  WhirlpoolDiskStars_eqFunction_135(data, threadData);
  WhirlpoolDiskStars_eqFunction_2642(data, threadData);
  WhirlpoolDiskStars_eqFunction_2645(data, threadData);
  WhirlpoolDiskStars_eqFunction_2647(data, threadData);
  WhirlpoolDiskStars_eqFunction_2650(data, threadData);
  WhirlpoolDiskStars_eqFunction_2649(data, threadData);
  WhirlpoolDiskStars_eqFunction_2648(data, threadData);
  WhirlpoolDiskStars_eqFunction_2646(data, threadData);
  WhirlpoolDiskStars_eqFunction_143(data, threadData);
  WhirlpoolDiskStars_eqFunction_2641(data, threadData);
  WhirlpoolDiskStars_eqFunction_145(data, threadData);
  WhirlpoolDiskStars_eqFunction_2654(data, threadData);
  WhirlpoolDiskStars_eqFunction_147(data, threadData);
  WhirlpoolDiskStars_eqFunction_148(data, threadData);
  WhirlpoolDiskStars_eqFunction_2653(data, threadData);
  WhirlpoolDiskStars_eqFunction_150(data, threadData);
  WhirlpoolDiskStars_eqFunction_151(data, threadData);
  WhirlpoolDiskStars_eqFunction_2652(data, threadData);
  WhirlpoolDiskStars_eqFunction_2655(data, threadData);
  WhirlpoolDiskStars_eqFunction_2657(data, threadData);
  WhirlpoolDiskStars_eqFunction_2660(data, threadData);
  WhirlpoolDiskStars_eqFunction_2659(data, threadData);
  WhirlpoolDiskStars_eqFunction_2658(data, threadData);
  WhirlpoolDiskStars_eqFunction_2656(data, threadData);
  WhirlpoolDiskStars_eqFunction_159(data, threadData);
  WhirlpoolDiskStars_eqFunction_2651(data, threadData);
  WhirlpoolDiskStars_eqFunction_161(data, threadData);
  WhirlpoolDiskStars_eqFunction_2664(data, threadData);
  WhirlpoolDiskStars_eqFunction_163(data, threadData);
  WhirlpoolDiskStars_eqFunction_164(data, threadData);
  WhirlpoolDiskStars_eqFunction_2663(data, threadData);
  WhirlpoolDiskStars_eqFunction_166(data, threadData);
  WhirlpoolDiskStars_eqFunction_167(data, threadData);
  WhirlpoolDiskStars_eqFunction_2662(data, threadData);
  WhirlpoolDiskStars_eqFunction_2665(data, threadData);
  WhirlpoolDiskStars_eqFunction_2667(data, threadData);
  WhirlpoolDiskStars_eqFunction_2670(data, threadData);
  WhirlpoolDiskStars_eqFunction_2669(data, threadData);
  WhirlpoolDiskStars_eqFunction_2668(data, threadData);
  WhirlpoolDiskStars_eqFunction_2666(data, threadData);
  WhirlpoolDiskStars_eqFunction_175(data, threadData);
  WhirlpoolDiskStars_eqFunction_2661(data, threadData);
  WhirlpoolDiskStars_eqFunction_177(data, threadData);
  WhirlpoolDiskStars_eqFunction_2674(data, threadData);
  WhirlpoolDiskStars_eqFunction_179(data, threadData);
  WhirlpoolDiskStars_eqFunction_180(data, threadData);
  WhirlpoolDiskStars_eqFunction_2673(data, threadData);
  WhirlpoolDiskStars_eqFunction_182(data, threadData);
  WhirlpoolDiskStars_eqFunction_183(data, threadData);
  WhirlpoolDiskStars_eqFunction_2672(data, threadData);
  WhirlpoolDiskStars_eqFunction_2675(data, threadData);
  WhirlpoolDiskStars_eqFunction_2677(data, threadData);
  WhirlpoolDiskStars_eqFunction_2680(data, threadData);
  WhirlpoolDiskStars_eqFunction_2679(data, threadData);
  WhirlpoolDiskStars_eqFunction_2678(data, threadData);
  WhirlpoolDiskStars_eqFunction_2676(data, threadData);
  WhirlpoolDiskStars_eqFunction_191(data, threadData);
  WhirlpoolDiskStars_eqFunction_2671(data, threadData);
  WhirlpoolDiskStars_eqFunction_193(data, threadData);
  WhirlpoolDiskStars_eqFunction_2684(data, threadData);
  WhirlpoolDiskStars_eqFunction_195(data, threadData);
  WhirlpoolDiskStars_eqFunction_196(data, threadData);
  WhirlpoolDiskStars_eqFunction_2683(data, threadData);
  WhirlpoolDiskStars_eqFunction_198(data, threadData);
  WhirlpoolDiskStars_eqFunction_199(data, threadData);
  WhirlpoolDiskStars_eqFunction_2682(data, threadData);
  WhirlpoolDiskStars_eqFunction_2685(data, threadData);
  WhirlpoolDiskStars_eqFunction_2687(data, threadData);
  WhirlpoolDiskStars_eqFunction_2690(data, threadData);
  WhirlpoolDiskStars_eqFunction_2689(data, threadData);
  WhirlpoolDiskStars_eqFunction_2688(data, threadData);
  WhirlpoolDiskStars_eqFunction_2686(data, threadData);
  WhirlpoolDiskStars_eqFunction_207(data, threadData);
  WhirlpoolDiskStars_eqFunction_2681(data, threadData);
  WhirlpoolDiskStars_eqFunction_209(data, threadData);
  WhirlpoolDiskStars_eqFunction_2694(data, threadData);
  WhirlpoolDiskStars_eqFunction_211(data, threadData);
  WhirlpoolDiskStars_eqFunction_212(data, threadData);
  WhirlpoolDiskStars_eqFunction_2693(data, threadData);
  WhirlpoolDiskStars_eqFunction_214(data, threadData);
  WhirlpoolDiskStars_eqFunction_215(data, threadData);
  WhirlpoolDiskStars_eqFunction_2692(data, threadData);
  WhirlpoolDiskStars_eqFunction_2695(data, threadData);
  WhirlpoolDiskStars_eqFunction_2697(data, threadData);
  WhirlpoolDiskStars_eqFunction_2700(data, threadData);
  WhirlpoolDiskStars_eqFunction_2699(data, threadData);
  WhirlpoolDiskStars_eqFunction_2698(data, threadData);
  WhirlpoolDiskStars_eqFunction_2696(data, threadData);
  WhirlpoolDiskStars_eqFunction_223(data, threadData);
  WhirlpoolDiskStars_eqFunction_2691(data, threadData);
  WhirlpoolDiskStars_eqFunction_225(data, threadData);
  WhirlpoolDiskStars_eqFunction_2704(data, threadData);
  WhirlpoolDiskStars_eqFunction_227(data, threadData);
  WhirlpoolDiskStars_eqFunction_228(data, threadData);
  WhirlpoolDiskStars_eqFunction_2703(data, threadData);
  WhirlpoolDiskStars_eqFunction_230(data, threadData);
  WhirlpoolDiskStars_eqFunction_231(data, threadData);
  WhirlpoolDiskStars_eqFunction_2702(data, threadData);
  WhirlpoolDiskStars_eqFunction_2705(data, threadData);
  WhirlpoolDiskStars_eqFunction_2707(data, threadData);
  WhirlpoolDiskStars_eqFunction_2710(data, threadData);
  WhirlpoolDiskStars_eqFunction_2709(data, threadData);
  WhirlpoolDiskStars_eqFunction_2708(data, threadData);
  WhirlpoolDiskStars_eqFunction_2706(data, threadData);
  WhirlpoolDiskStars_eqFunction_239(data, threadData);
  WhirlpoolDiskStars_eqFunction_2701(data, threadData);
  WhirlpoolDiskStars_eqFunction_241(data, threadData);
  WhirlpoolDiskStars_eqFunction_2714(data, threadData);
  WhirlpoolDiskStars_eqFunction_243(data, threadData);
  WhirlpoolDiskStars_eqFunction_244(data, threadData);
  WhirlpoolDiskStars_eqFunction_2713(data, threadData);
  WhirlpoolDiskStars_eqFunction_246(data, threadData);
  WhirlpoolDiskStars_eqFunction_247(data, threadData);
  WhirlpoolDiskStars_eqFunction_2712(data, threadData);
  WhirlpoolDiskStars_eqFunction_2715(data, threadData);
  WhirlpoolDiskStars_eqFunction_2717(data, threadData);
  WhirlpoolDiskStars_eqFunction_2720(data, threadData);
  WhirlpoolDiskStars_eqFunction_2719(data, threadData);
  WhirlpoolDiskStars_eqFunction_2718(data, threadData);
  WhirlpoolDiskStars_eqFunction_2716(data, threadData);
  WhirlpoolDiskStars_eqFunction_255(data, threadData);
  WhirlpoolDiskStars_eqFunction_2711(data, threadData);
  WhirlpoolDiskStars_eqFunction_257(data, threadData);
  WhirlpoolDiskStars_eqFunction_2724(data, threadData);
  WhirlpoolDiskStars_eqFunction_259(data, threadData);
  WhirlpoolDiskStars_eqFunction_260(data, threadData);
  WhirlpoolDiskStars_eqFunction_2723(data, threadData);
  WhirlpoolDiskStars_eqFunction_262(data, threadData);
  WhirlpoolDiskStars_eqFunction_263(data, threadData);
  WhirlpoolDiskStars_eqFunction_2722(data, threadData);
  WhirlpoolDiskStars_eqFunction_2725(data, threadData);
  WhirlpoolDiskStars_eqFunction_2727(data, threadData);
  WhirlpoolDiskStars_eqFunction_2730(data, threadData);
  WhirlpoolDiskStars_eqFunction_2729(data, threadData);
  WhirlpoolDiskStars_eqFunction_2728(data, threadData);
  WhirlpoolDiskStars_eqFunction_2726(data, threadData);
  WhirlpoolDiskStars_eqFunction_271(data, threadData);
  WhirlpoolDiskStars_eqFunction_2721(data, threadData);
  WhirlpoolDiskStars_eqFunction_273(data, threadData);
  WhirlpoolDiskStars_eqFunction_2734(data, threadData);
  WhirlpoolDiskStars_eqFunction_275(data, threadData);
  WhirlpoolDiskStars_eqFunction_276(data, threadData);
  WhirlpoolDiskStars_eqFunction_2733(data, threadData);
  WhirlpoolDiskStars_eqFunction_278(data, threadData);
  WhirlpoolDiskStars_eqFunction_279(data, threadData);
  WhirlpoolDiskStars_eqFunction_2732(data, threadData);
  WhirlpoolDiskStars_eqFunction_2735(data, threadData);
  WhirlpoolDiskStars_eqFunction_2737(data, threadData);
  WhirlpoolDiskStars_eqFunction_2740(data, threadData);
  WhirlpoolDiskStars_eqFunction_2739(data, threadData);
  WhirlpoolDiskStars_eqFunction_2738(data, threadData);
  WhirlpoolDiskStars_eqFunction_2736(data, threadData);
  WhirlpoolDiskStars_eqFunction_287(data, threadData);
  WhirlpoolDiskStars_eqFunction_2731(data, threadData);
  WhirlpoolDiskStars_eqFunction_289(data, threadData);
  WhirlpoolDiskStars_eqFunction_2744(data, threadData);
  WhirlpoolDiskStars_eqFunction_291(data, threadData);
  WhirlpoolDiskStars_eqFunction_292(data, threadData);
  WhirlpoolDiskStars_eqFunction_2743(data, threadData);
  WhirlpoolDiskStars_eqFunction_294(data, threadData);
  WhirlpoolDiskStars_eqFunction_295(data, threadData);
  WhirlpoolDiskStars_eqFunction_2742(data, threadData);
  WhirlpoolDiskStars_eqFunction_2745(data, threadData);
  WhirlpoolDiskStars_eqFunction_2747(data, threadData);
  WhirlpoolDiskStars_eqFunction_2750(data, threadData);
  WhirlpoolDiskStars_eqFunction_2749(data, threadData);
  WhirlpoolDiskStars_eqFunction_2748(data, threadData);
  WhirlpoolDiskStars_eqFunction_2746(data, threadData);
  WhirlpoolDiskStars_eqFunction_303(data, threadData);
  WhirlpoolDiskStars_eqFunction_2741(data, threadData);
  WhirlpoolDiskStars_eqFunction_305(data, threadData);
  WhirlpoolDiskStars_eqFunction_2754(data, threadData);
  WhirlpoolDiskStars_eqFunction_307(data, threadData);
  WhirlpoolDiskStars_eqFunction_308(data, threadData);
  WhirlpoolDiskStars_eqFunction_2753(data, threadData);
  WhirlpoolDiskStars_eqFunction_310(data, threadData);
  WhirlpoolDiskStars_eqFunction_311(data, threadData);
  WhirlpoolDiskStars_eqFunction_2752(data, threadData);
  WhirlpoolDiskStars_eqFunction_2755(data, threadData);
  WhirlpoolDiskStars_eqFunction_2757(data, threadData);
  WhirlpoolDiskStars_eqFunction_2760(data, threadData);
  WhirlpoolDiskStars_eqFunction_2759(data, threadData);
  WhirlpoolDiskStars_eqFunction_2758(data, threadData);
  WhirlpoolDiskStars_eqFunction_2756(data, threadData);
  WhirlpoolDiskStars_eqFunction_319(data, threadData);
  WhirlpoolDiskStars_eqFunction_2751(data, threadData);
  WhirlpoolDiskStars_eqFunction_321(data, threadData);
  WhirlpoolDiskStars_eqFunction_2764(data, threadData);
  WhirlpoolDiskStars_eqFunction_323(data, threadData);
  WhirlpoolDiskStars_eqFunction_324(data, threadData);
  WhirlpoolDiskStars_eqFunction_2763(data, threadData);
  WhirlpoolDiskStars_eqFunction_326(data, threadData);
  WhirlpoolDiskStars_eqFunction_327(data, threadData);
  WhirlpoolDiskStars_eqFunction_2762(data, threadData);
  WhirlpoolDiskStars_eqFunction_2765(data, threadData);
  WhirlpoolDiskStars_eqFunction_2767(data, threadData);
  WhirlpoolDiskStars_eqFunction_2770(data, threadData);
  WhirlpoolDiskStars_eqFunction_2769(data, threadData);
  WhirlpoolDiskStars_eqFunction_2768(data, threadData);
  WhirlpoolDiskStars_eqFunction_2766(data, threadData);
  WhirlpoolDiskStars_eqFunction_335(data, threadData);
  WhirlpoolDiskStars_eqFunction_2761(data, threadData);
  WhirlpoolDiskStars_eqFunction_337(data, threadData);
  WhirlpoolDiskStars_eqFunction_2774(data, threadData);
  WhirlpoolDiskStars_eqFunction_339(data, threadData);
  WhirlpoolDiskStars_eqFunction_340(data, threadData);
  WhirlpoolDiskStars_eqFunction_2773(data, threadData);
  WhirlpoolDiskStars_eqFunction_342(data, threadData);
  WhirlpoolDiskStars_eqFunction_343(data, threadData);
  WhirlpoolDiskStars_eqFunction_2772(data, threadData);
  WhirlpoolDiskStars_eqFunction_2775(data, threadData);
  WhirlpoolDiskStars_eqFunction_2777(data, threadData);
  WhirlpoolDiskStars_eqFunction_2780(data, threadData);
  WhirlpoolDiskStars_eqFunction_2779(data, threadData);
  WhirlpoolDiskStars_eqFunction_2778(data, threadData);
  WhirlpoolDiskStars_eqFunction_2776(data, threadData);
  WhirlpoolDiskStars_eqFunction_351(data, threadData);
  WhirlpoolDiskStars_eqFunction_2771(data, threadData);
  WhirlpoolDiskStars_eqFunction_353(data, threadData);
  WhirlpoolDiskStars_eqFunction_2784(data, threadData);
  WhirlpoolDiskStars_eqFunction_355(data, threadData);
  WhirlpoolDiskStars_eqFunction_356(data, threadData);
  WhirlpoolDiskStars_eqFunction_2783(data, threadData);
  WhirlpoolDiskStars_eqFunction_358(data, threadData);
  WhirlpoolDiskStars_eqFunction_359(data, threadData);
  WhirlpoolDiskStars_eqFunction_2782(data, threadData);
  WhirlpoolDiskStars_eqFunction_2785(data, threadData);
  WhirlpoolDiskStars_eqFunction_2787(data, threadData);
  WhirlpoolDiskStars_eqFunction_2790(data, threadData);
  WhirlpoolDiskStars_eqFunction_2789(data, threadData);
  WhirlpoolDiskStars_eqFunction_2788(data, threadData);
  WhirlpoolDiskStars_eqFunction_2786(data, threadData);
  WhirlpoolDiskStars_eqFunction_367(data, threadData);
  WhirlpoolDiskStars_eqFunction_2781(data, threadData);
  WhirlpoolDiskStars_eqFunction_369(data, threadData);
  WhirlpoolDiskStars_eqFunction_2794(data, threadData);
  WhirlpoolDiskStars_eqFunction_371(data, threadData);
  WhirlpoolDiskStars_eqFunction_372(data, threadData);
  WhirlpoolDiskStars_eqFunction_2793(data, threadData);
  WhirlpoolDiskStars_eqFunction_374(data, threadData);
  WhirlpoolDiskStars_eqFunction_375(data, threadData);
  WhirlpoolDiskStars_eqFunction_2792(data, threadData);
  WhirlpoolDiskStars_eqFunction_2795(data, threadData);
  WhirlpoolDiskStars_eqFunction_2797(data, threadData);
  WhirlpoolDiskStars_eqFunction_2800(data, threadData);
  WhirlpoolDiskStars_eqFunction_2799(data, threadData);
  WhirlpoolDiskStars_eqFunction_2798(data, threadData);
  WhirlpoolDiskStars_eqFunction_2796(data, threadData);
  WhirlpoolDiskStars_eqFunction_383(data, threadData);
  WhirlpoolDiskStars_eqFunction_2791(data, threadData);
  WhirlpoolDiskStars_eqFunction_385(data, threadData);
  WhirlpoolDiskStars_eqFunction_2804(data, threadData);
  WhirlpoolDiskStars_eqFunction_387(data, threadData);
  WhirlpoolDiskStars_eqFunction_388(data, threadData);
  WhirlpoolDiskStars_eqFunction_2803(data, threadData);
  WhirlpoolDiskStars_eqFunction_390(data, threadData);
  WhirlpoolDiskStars_eqFunction_391(data, threadData);
  WhirlpoolDiskStars_eqFunction_2802(data, threadData);
  WhirlpoolDiskStars_eqFunction_2805(data, threadData);
  WhirlpoolDiskStars_eqFunction_2807(data, threadData);
  WhirlpoolDiskStars_eqFunction_2810(data, threadData);
  WhirlpoolDiskStars_eqFunction_2809(data, threadData);
  WhirlpoolDiskStars_eqFunction_2808(data, threadData);
  WhirlpoolDiskStars_eqFunction_2806(data, threadData);
  WhirlpoolDiskStars_eqFunction_399(data, threadData);
  WhirlpoolDiskStars_eqFunction_2801(data, threadData);
  WhirlpoolDiskStars_eqFunction_401(data, threadData);
  WhirlpoolDiskStars_eqFunction_2814(data, threadData);
  WhirlpoolDiskStars_eqFunction_403(data, threadData);
  WhirlpoolDiskStars_eqFunction_404(data, threadData);
  WhirlpoolDiskStars_eqFunction_2813(data, threadData);
  WhirlpoolDiskStars_eqFunction_406(data, threadData);
  WhirlpoolDiskStars_eqFunction_407(data, threadData);
  WhirlpoolDiskStars_eqFunction_2812(data, threadData);
  WhirlpoolDiskStars_eqFunction_2815(data, threadData);
  WhirlpoolDiskStars_eqFunction_2817(data, threadData);
  WhirlpoolDiskStars_eqFunction_2820(data, threadData);
  WhirlpoolDiskStars_eqFunction_2819(data, threadData);
  WhirlpoolDiskStars_eqFunction_2818(data, threadData);
  WhirlpoolDiskStars_eqFunction_2816(data, threadData);
  WhirlpoolDiskStars_eqFunction_415(data, threadData);
  WhirlpoolDiskStars_eqFunction_2811(data, threadData);
  WhirlpoolDiskStars_eqFunction_417(data, threadData);
  WhirlpoolDiskStars_eqFunction_2824(data, threadData);
  WhirlpoolDiskStars_eqFunction_419(data, threadData);
  WhirlpoolDiskStars_eqFunction_420(data, threadData);
  WhirlpoolDiskStars_eqFunction_2823(data, threadData);
  WhirlpoolDiskStars_eqFunction_422(data, threadData);
  WhirlpoolDiskStars_eqFunction_423(data, threadData);
  WhirlpoolDiskStars_eqFunction_2822(data, threadData);
  WhirlpoolDiskStars_eqFunction_2825(data, threadData);
  WhirlpoolDiskStars_eqFunction_2827(data, threadData);
  WhirlpoolDiskStars_eqFunction_2830(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif