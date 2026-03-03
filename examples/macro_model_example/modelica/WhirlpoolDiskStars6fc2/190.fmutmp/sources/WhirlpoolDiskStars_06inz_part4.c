#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif
extern void WhirlpoolDiskStars_eqFunction_3628(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3626(DATA *data, threadData_t *threadData);


/*
equation index: 1711
type: SIMPLE_ASSIGN
vz[107] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1711(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1711};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[426]] /* vz[107] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3621(DATA *data, threadData_t *threadData);


/*
equation index: 1713
type: SIMPLE_ASSIGN
z[108] = 1.4000000000000001
*/
void WhirlpoolDiskStars_eqFunction_1713(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1713};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[907]] /* z[108] STATE(1,vz[108]) */) = 1.4000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3634(DATA *data, threadData_t *threadData);


/*
equation index: 1715
type: SIMPLE_ASSIGN
y[108] = r_init[108] * sin(theta[108] + armOffset[108])
*/
void WhirlpoolDiskStars_eqFunction_1715(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1715};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[747]] /* y[108] STATE(1,vy[108]) */) = ((data->simulationInfo->realParameter[272] /* r_init[108] PARAM */)) * (sin((data->simulationInfo->realParameter[432] /* theta[108] PARAM */) + (data->simulationInfo->realParameter[110] /* armOffset[108] PARAM */)));
  TRACE_POP
}

/*
equation index: 1716
type: SIMPLE_ASSIGN
vx[108] = (-y[108]) * sqrt(G * Md / r_init[108] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1716(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1716};
  modelica_real tmp428;
  modelica_real tmp429;
  tmp428 = (data->simulationInfo->realParameter[272] /* r_init[108] PARAM */);
  tmp429 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp428 * tmp428 * tmp428),"r_init[108] ^ 3.0",equationIndexes);
  if(!(tmp429 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[108] ^ 3.0) was %g should be >= 0", tmp429);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[107]] /* vx[108] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[747]] /* y[108] STATE(1,vy[108]) */))) * (sqrt(tmp429));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3633(DATA *data, threadData_t *threadData);


/*
equation index: 1718
type: SIMPLE_ASSIGN
x[108] = r_init[108] * cos(theta[108] + armOffset[108])
*/
void WhirlpoolDiskStars_eqFunction_1718(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1718};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[587]] /* x[108] STATE(1,vx[108]) */) = ((data->simulationInfo->realParameter[272] /* r_init[108] PARAM */)) * (cos((data->simulationInfo->realParameter[432] /* theta[108] PARAM */) + (data->simulationInfo->realParameter[110] /* armOffset[108] PARAM */)));
  TRACE_POP
}

/*
equation index: 1719
type: SIMPLE_ASSIGN
vy[108] = x[108] * sqrt(G * Md / r_init[108] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1719(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1719};
  modelica_real tmp430;
  modelica_real tmp431;
  tmp430 = (data->simulationInfo->realParameter[272] /* r_init[108] PARAM */);
  tmp431 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp430 * tmp430 * tmp430),"r_init[108] ^ 3.0",equationIndexes);
  if(!(tmp431 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[108] ^ 3.0) was %g should be >= 0", tmp431);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[267]] /* vy[108] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[587]] /* x[108] STATE(1,vx[108]) */)) * (sqrt(tmp431));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3632(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3635(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3637(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3640(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3639(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3638(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3636(DATA *data, threadData_t *threadData);


/*
equation index: 1727
type: SIMPLE_ASSIGN
vz[108] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1727(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1727};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[427]] /* vz[108] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3631(DATA *data, threadData_t *threadData);


/*
equation index: 1729
type: SIMPLE_ASSIGN
z[109] = 1.4500000000000002
*/
void WhirlpoolDiskStars_eqFunction_1729(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1729};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[908]] /* z[109] STATE(1,vz[109]) */) = 1.4500000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3644(DATA *data, threadData_t *threadData);


/*
equation index: 1731
type: SIMPLE_ASSIGN
y[109] = r_init[109] * sin(theta[109] + armOffset[109])
*/
void WhirlpoolDiskStars_eqFunction_1731(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1731};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[748]] /* y[109] STATE(1,vy[109]) */) = ((data->simulationInfo->realParameter[273] /* r_init[109] PARAM */)) * (sin((data->simulationInfo->realParameter[433] /* theta[109] PARAM */) + (data->simulationInfo->realParameter[111] /* armOffset[109] PARAM */)));
  TRACE_POP
}

/*
equation index: 1732
type: SIMPLE_ASSIGN
vx[109] = (-y[109]) * sqrt(G * Md / r_init[109] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1732(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1732};
  modelica_real tmp432;
  modelica_real tmp433;
  tmp432 = (data->simulationInfo->realParameter[273] /* r_init[109] PARAM */);
  tmp433 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp432 * tmp432 * tmp432),"r_init[109] ^ 3.0",equationIndexes);
  if(!(tmp433 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[109] ^ 3.0) was %g should be >= 0", tmp433);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[108]] /* vx[109] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[748]] /* y[109] STATE(1,vy[109]) */))) * (sqrt(tmp433));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3643(DATA *data, threadData_t *threadData);


/*
equation index: 1734
type: SIMPLE_ASSIGN
x[109] = r_init[109] * cos(theta[109] + armOffset[109])
*/
void WhirlpoolDiskStars_eqFunction_1734(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1734};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[588]] /* x[109] STATE(1,vx[109]) */) = ((data->simulationInfo->realParameter[273] /* r_init[109] PARAM */)) * (cos((data->simulationInfo->realParameter[433] /* theta[109] PARAM */) + (data->simulationInfo->realParameter[111] /* armOffset[109] PARAM */)));
  TRACE_POP
}

/*
equation index: 1735
type: SIMPLE_ASSIGN
vy[109] = x[109] * sqrt(G * Md / r_init[109] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1735(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1735};
  modelica_real tmp434;
  modelica_real tmp435;
  tmp434 = (data->simulationInfo->realParameter[273] /* r_init[109] PARAM */);
  tmp435 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp434 * tmp434 * tmp434),"r_init[109] ^ 3.0",equationIndexes);
  if(!(tmp435 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[109] ^ 3.0) was %g should be >= 0", tmp435);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[268]] /* vy[109] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[588]] /* x[109] STATE(1,vx[109]) */)) * (sqrt(tmp435));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3642(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3645(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3647(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3650(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3649(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3648(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3646(DATA *data, threadData_t *threadData);


/*
equation index: 1743
type: SIMPLE_ASSIGN
vz[109] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1743(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1743};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[428]] /* vz[109] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3641(DATA *data, threadData_t *threadData);


/*
equation index: 1745
type: SIMPLE_ASSIGN
z[110] = 1.5
*/
void WhirlpoolDiskStars_eqFunction_1745(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1745};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[909]] /* z[110] STATE(1,vz[110]) */) = 1.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3654(DATA *data, threadData_t *threadData);


/*
equation index: 1747
type: SIMPLE_ASSIGN
y[110] = r_init[110] * sin(theta[110] + armOffset[110])
*/
void WhirlpoolDiskStars_eqFunction_1747(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1747};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[749]] /* y[110] STATE(1,vy[110]) */) = ((data->simulationInfo->realParameter[274] /* r_init[110] PARAM */)) * (sin((data->simulationInfo->realParameter[434] /* theta[110] PARAM */) + (data->simulationInfo->realParameter[112] /* armOffset[110] PARAM */)));
  TRACE_POP
}

/*
equation index: 1748
type: SIMPLE_ASSIGN
vx[110] = (-y[110]) * sqrt(G * Md / r_init[110] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1748(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1748};
  modelica_real tmp436;
  modelica_real tmp437;
  tmp436 = (data->simulationInfo->realParameter[274] /* r_init[110] PARAM */);
  tmp437 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp436 * tmp436 * tmp436),"r_init[110] ^ 3.0",equationIndexes);
  if(!(tmp437 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[110] ^ 3.0) was %g should be >= 0", tmp437);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[109]] /* vx[110] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[749]] /* y[110] STATE(1,vy[110]) */))) * (sqrt(tmp437));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3653(DATA *data, threadData_t *threadData);


/*
equation index: 1750
type: SIMPLE_ASSIGN
x[110] = r_init[110] * cos(theta[110] + armOffset[110])
*/
void WhirlpoolDiskStars_eqFunction_1750(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1750};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[589]] /* x[110] STATE(1,vx[110]) */) = ((data->simulationInfo->realParameter[274] /* r_init[110] PARAM */)) * (cos((data->simulationInfo->realParameter[434] /* theta[110] PARAM */) + (data->simulationInfo->realParameter[112] /* armOffset[110] PARAM */)));
  TRACE_POP
}

/*
equation index: 1751
type: SIMPLE_ASSIGN
vy[110] = x[110] * sqrt(G * Md / r_init[110] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1751(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1751};
  modelica_real tmp438;
  modelica_real tmp439;
  tmp438 = (data->simulationInfo->realParameter[274] /* r_init[110] PARAM */);
  tmp439 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp438 * tmp438 * tmp438),"r_init[110] ^ 3.0",equationIndexes);
  if(!(tmp439 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[110] ^ 3.0) was %g should be >= 0", tmp439);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[269]] /* vy[110] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[589]] /* x[110] STATE(1,vx[110]) */)) * (sqrt(tmp439));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3652(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3655(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3657(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3660(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3659(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3658(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3656(DATA *data, threadData_t *threadData);


/*
equation index: 1759
type: SIMPLE_ASSIGN
vz[110] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1759(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1759};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[429]] /* vz[110] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3651(DATA *data, threadData_t *threadData);


/*
equation index: 1761
type: SIMPLE_ASSIGN
z[111] = 1.55
*/
void WhirlpoolDiskStars_eqFunction_1761(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1761};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[910]] /* z[111] STATE(1,vz[111]) */) = 1.55;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3664(DATA *data, threadData_t *threadData);


/*
equation index: 1763
type: SIMPLE_ASSIGN
y[111] = r_init[111] * sin(theta[111] + armOffset[111])
*/
void WhirlpoolDiskStars_eqFunction_1763(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1763};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[750]] /* y[111] STATE(1,vy[111]) */) = ((data->simulationInfo->realParameter[275] /* r_init[111] PARAM */)) * (sin((data->simulationInfo->realParameter[435] /* theta[111] PARAM */) + (data->simulationInfo->realParameter[113] /* armOffset[111] PARAM */)));
  TRACE_POP
}

/*
equation index: 1764
type: SIMPLE_ASSIGN
vx[111] = (-y[111]) * sqrt(G * Md / r_init[111] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1764(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1764};
  modelica_real tmp440;
  modelica_real tmp441;
  tmp440 = (data->simulationInfo->realParameter[275] /* r_init[111] PARAM */);
  tmp441 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp440 * tmp440 * tmp440),"r_init[111] ^ 3.0",equationIndexes);
  if(!(tmp441 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[111] ^ 3.0) was %g should be >= 0", tmp441);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[110]] /* vx[111] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[750]] /* y[111] STATE(1,vy[111]) */))) * (sqrt(tmp441));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3663(DATA *data, threadData_t *threadData);


/*
equation index: 1766
type: SIMPLE_ASSIGN
x[111] = r_init[111] * cos(theta[111] + armOffset[111])
*/
void WhirlpoolDiskStars_eqFunction_1766(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1766};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[590]] /* x[111] STATE(1,vx[111]) */) = ((data->simulationInfo->realParameter[275] /* r_init[111] PARAM */)) * (cos((data->simulationInfo->realParameter[435] /* theta[111] PARAM */) + (data->simulationInfo->realParameter[113] /* armOffset[111] PARAM */)));
  TRACE_POP
}

/*
equation index: 1767
type: SIMPLE_ASSIGN
vy[111] = x[111] * sqrt(G * Md / r_init[111] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1767(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1767};
  modelica_real tmp442;
  modelica_real tmp443;
  tmp442 = (data->simulationInfo->realParameter[275] /* r_init[111] PARAM */);
  tmp443 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp442 * tmp442 * tmp442),"r_init[111] ^ 3.0",equationIndexes);
  if(!(tmp443 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[111] ^ 3.0) was %g should be >= 0", tmp443);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[270]] /* vy[111] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[590]] /* x[111] STATE(1,vx[111]) */)) * (sqrt(tmp443));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3662(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3665(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3667(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3670(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3669(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3668(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3666(DATA *data, threadData_t *threadData);


/*
equation index: 1775
type: SIMPLE_ASSIGN
vz[111] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1775(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1775};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[430]] /* vz[111] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3661(DATA *data, threadData_t *threadData);


/*
equation index: 1777
type: SIMPLE_ASSIGN
z[112] = 1.6
*/
void WhirlpoolDiskStars_eqFunction_1777(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1777};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[911]] /* z[112] STATE(1,vz[112]) */) = 1.6;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3674(DATA *data, threadData_t *threadData);


/*
equation index: 1779
type: SIMPLE_ASSIGN
y[112] = r_init[112] * sin(theta[112] + armOffset[112])
*/
void WhirlpoolDiskStars_eqFunction_1779(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1779};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[751]] /* y[112] STATE(1,vy[112]) */) = ((data->simulationInfo->realParameter[276] /* r_init[112] PARAM */)) * (sin((data->simulationInfo->realParameter[436] /* theta[112] PARAM */) + (data->simulationInfo->realParameter[114] /* armOffset[112] PARAM */)));
  TRACE_POP
}

/*
equation index: 1780
type: SIMPLE_ASSIGN
vx[112] = (-y[112]) * sqrt(G * Md / r_init[112] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1780(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1780};
  modelica_real tmp444;
  modelica_real tmp445;
  tmp444 = (data->simulationInfo->realParameter[276] /* r_init[112] PARAM */);
  tmp445 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp444 * tmp444 * tmp444),"r_init[112] ^ 3.0",equationIndexes);
  if(!(tmp445 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[112] ^ 3.0) was %g should be >= 0", tmp445);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[111]] /* vx[112] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[751]] /* y[112] STATE(1,vy[112]) */))) * (sqrt(tmp445));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3673(DATA *data, threadData_t *threadData);


/*
equation index: 1782
type: SIMPLE_ASSIGN
x[112] = r_init[112] * cos(theta[112] + armOffset[112])
*/
void WhirlpoolDiskStars_eqFunction_1782(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1782};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[591]] /* x[112] STATE(1,vx[112]) */) = ((data->simulationInfo->realParameter[276] /* r_init[112] PARAM */)) * (cos((data->simulationInfo->realParameter[436] /* theta[112] PARAM */) + (data->simulationInfo->realParameter[114] /* armOffset[112] PARAM */)));
  TRACE_POP
}

/*
equation index: 1783
type: SIMPLE_ASSIGN
vy[112] = x[112] * sqrt(G * Md / r_init[112] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1783(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1783};
  modelica_real tmp446;
  modelica_real tmp447;
  tmp446 = (data->simulationInfo->realParameter[276] /* r_init[112] PARAM */);
  tmp447 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp446 * tmp446 * tmp446),"r_init[112] ^ 3.0",equationIndexes);
  if(!(tmp447 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[112] ^ 3.0) was %g should be >= 0", tmp447);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[271]] /* vy[112] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[591]] /* x[112] STATE(1,vx[112]) */)) * (sqrt(tmp447));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3672(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3675(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3677(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3680(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3679(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3678(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3676(DATA *data, threadData_t *threadData);


/*
equation index: 1791
type: SIMPLE_ASSIGN
vz[112] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1791(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1791};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[431]] /* vz[112] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3671(DATA *data, threadData_t *threadData);


/*
equation index: 1793
type: SIMPLE_ASSIGN
z[113] = 1.6500000000000001
*/
void WhirlpoolDiskStars_eqFunction_1793(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1793};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[912]] /* z[113] STATE(1,vz[113]) */) = 1.6500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3684(DATA *data, threadData_t *threadData);


/*
equation index: 1795
type: SIMPLE_ASSIGN
y[113] = r_init[113] * sin(theta[113] + armOffset[113])
*/
void WhirlpoolDiskStars_eqFunction_1795(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1795};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[752]] /* y[113] STATE(1,vy[113]) */) = ((data->simulationInfo->realParameter[277] /* r_init[113] PARAM */)) * (sin((data->simulationInfo->realParameter[437] /* theta[113] PARAM */) + (data->simulationInfo->realParameter[115] /* armOffset[113] PARAM */)));
  TRACE_POP
}

/*
equation index: 1796
type: SIMPLE_ASSIGN
vx[113] = (-y[113]) * sqrt(G * Md / r_init[113] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1796(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1796};
  modelica_real tmp448;
  modelica_real tmp449;
  tmp448 = (data->simulationInfo->realParameter[277] /* r_init[113] PARAM */);
  tmp449 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp448 * tmp448 * tmp448),"r_init[113] ^ 3.0",equationIndexes);
  if(!(tmp449 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[113] ^ 3.0) was %g should be >= 0", tmp449);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[112]] /* vx[113] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[752]] /* y[113] STATE(1,vy[113]) */))) * (sqrt(tmp449));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3683(DATA *data, threadData_t *threadData);


/*
equation index: 1798
type: SIMPLE_ASSIGN
x[113] = r_init[113] * cos(theta[113] + armOffset[113])
*/
void WhirlpoolDiskStars_eqFunction_1798(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1798};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[592]] /* x[113] STATE(1,vx[113]) */) = ((data->simulationInfo->realParameter[277] /* r_init[113] PARAM */)) * (cos((data->simulationInfo->realParameter[437] /* theta[113] PARAM */) + (data->simulationInfo->realParameter[115] /* armOffset[113] PARAM */)));
  TRACE_POP
}

/*
equation index: 1799
type: SIMPLE_ASSIGN
vy[113] = x[113] * sqrt(G * Md / r_init[113] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1799(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1799};
  modelica_real tmp450;
  modelica_real tmp451;
  tmp450 = (data->simulationInfo->realParameter[277] /* r_init[113] PARAM */);
  tmp451 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp450 * tmp450 * tmp450),"r_init[113] ^ 3.0",equationIndexes);
  if(!(tmp451 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[113] ^ 3.0) was %g should be >= 0", tmp451);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[272]] /* vy[113] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[592]] /* x[113] STATE(1,vx[113]) */)) * (sqrt(tmp451));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3682(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3685(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3687(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3690(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3689(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3688(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3686(DATA *data, threadData_t *threadData);


/*
equation index: 1807
type: SIMPLE_ASSIGN
vz[113] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1807(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1807};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[432]] /* vz[113] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3681(DATA *data, threadData_t *threadData);


/*
equation index: 1809
type: SIMPLE_ASSIGN
z[114] = 1.7000000000000002
*/
void WhirlpoolDiskStars_eqFunction_1809(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1809};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[913]] /* z[114] STATE(1,vz[114]) */) = 1.7000000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3694(DATA *data, threadData_t *threadData);


/*
equation index: 1811
type: SIMPLE_ASSIGN
y[114] = r_init[114] * sin(theta[114] + armOffset[114])
*/
void WhirlpoolDiskStars_eqFunction_1811(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1811};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[753]] /* y[114] STATE(1,vy[114]) */) = ((data->simulationInfo->realParameter[278] /* r_init[114] PARAM */)) * (sin((data->simulationInfo->realParameter[438] /* theta[114] PARAM */) + (data->simulationInfo->realParameter[116] /* armOffset[114] PARAM */)));
  TRACE_POP
}

/*
equation index: 1812
type: SIMPLE_ASSIGN
vx[114] = (-y[114]) * sqrt(G * Md / r_init[114] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1812(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1812};
  modelica_real tmp452;
  modelica_real tmp453;
  tmp452 = (data->simulationInfo->realParameter[278] /* r_init[114] PARAM */);
  tmp453 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp452 * tmp452 * tmp452),"r_init[114] ^ 3.0",equationIndexes);
  if(!(tmp453 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[114] ^ 3.0) was %g should be >= 0", tmp453);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[113]] /* vx[114] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[753]] /* y[114] STATE(1,vy[114]) */))) * (sqrt(tmp453));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3693(DATA *data, threadData_t *threadData);


/*
equation index: 1814
type: SIMPLE_ASSIGN
x[114] = r_init[114] * cos(theta[114] + armOffset[114])
*/
void WhirlpoolDiskStars_eqFunction_1814(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1814};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[593]] /* x[114] STATE(1,vx[114]) */) = ((data->simulationInfo->realParameter[278] /* r_init[114] PARAM */)) * (cos((data->simulationInfo->realParameter[438] /* theta[114] PARAM */) + (data->simulationInfo->realParameter[116] /* armOffset[114] PARAM */)));
  TRACE_POP
}

/*
equation index: 1815
type: SIMPLE_ASSIGN
vy[114] = x[114] * sqrt(G * Md / r_init[114] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1815(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1815};
  modelica_real tmp454;
  modelica_real tmp455;
  tmp454 = (data->simulationInfo->realParameter[278] /* r_init[114] PARAM */);
  tmp455 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp454 * tmp454 * tmp454),"r_init[114] ^ 3.0",equationIndexes);
  if(!(tmp455 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[114] ^ 3.0) was %g should be >= 0", tmp455);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[273]] /* vy[114] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[593]] /* x[114] STATE(1,vx[114]) */)) * (sqrt(tmp455));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3692(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3695(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3697(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3700(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3699(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3698(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3696(DATA *data, threadData_t *threadData);


/*
equation index: 1823
type: SIMPLE_ASSIGN
vz[114] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1823(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1823};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[433]] /* vz[114] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3691(DATA *data, threadData_t *threadData);


/*
equation index: 1825
type: SIMPLE_ASSIGN
z[115] = 1.75
*/
void WhirlpoolDiskStars_eqFunction_1825(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1825};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[914]] /* z[115] STATE(1,vz[115]) */) = 1.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3704(DATA *data, threadData_t *threadData);


/*
equation index: 1827
type: SIMPLE_ASSIGN
y[115] = r_init[115] * sin(theta[115] + armOffset[115])
*/
void WhirlpoolDiskStars_eqFunction_1827(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1827};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[754]] /* y[115] STATE(1,vy[115]) */) = ((data->simulationInfo->realParameter[279] /* r_init[115] PARAM */)) * (sin((data->simulationInfo->realParameter[439] /* theta[115] PARAM */) + (data->simulationInfo->realParameter[117] /* armOffset[115] PARAM */)));
  TRACE_POP
}

/*
equation index: 1828
type: SIMPLE_ASSIGN
vx[115] = (-y[115]) * sqrt(G * Md / r_init[115] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1828(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1828};
  modelica_real tmp456;
  modelica_real tmp457;
  tmp456 = (data->simulationInfo->realParameter[279] /* r_init[115] PARAM */);
  tmp457 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp456 * tmp456 * tmp456),"r_init[115] ^ 3.0",equationIndexes);
  if(!(tmp457 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[115] ^ 3.0) was %g should be >= 0", tmp457);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[114]] /* vx[115] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[754]] /* y[115] STATE(1,vy[115]) */))) * (sqrt(tmp457));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3703(DATA *data, threadData_t *threadData);


/*
equation index: 1830
type: SIMPLE_ASSIGN
x[115] = r_init[115] * cos(theta[115] + armOffset[115])
*/
void WhirlpoolDiskStars_eqFunction_1830(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1830};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[594]] /* x[115] STATE(1,vx[115]) */) = ((data->simulationInfo->realParameter[279] /* r_init[115] PARAM */)) * (cos((data->simulationInfo->realParameter[439] /* theta[115] PARAM */) + (data->simulationInfo->realParameter[117] /* armOffset[115] PARAM */)));
  TRACE_POP
}

/*
equation index: 1831
type: SIMPLE_ASSIGN
vy[115] = x[115] * sqrt(G * Md / r_init[115] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1831(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1831};
  modelica_real tmp458;
  modelica_real tmp459;
  tmp458 = (data->simulationInfo->realParameter[279] /* r_init[115] PARAM */);
  tmp459 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp458 * tmp458 * tmp458),"r_init[115] ^ 3.0",equationIndexes);
  if(!(tmp459 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[115] ^ 3.0) was %g should be >= 0", tmp459);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[274]] /* vy[115] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[594]] /* x[115] STATE(1,vx[115]) */)) * (sqrt(tmp459));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3702(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3705(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3707(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3710(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3709(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3708(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3706(DATA *data, threadData_t *threadData);


/*
equation index: 1839
type: SIMPLE_ASSIGN
vz[115] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1839(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1839};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[434]] /* vz[115] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3701(DATA *data, threadData_t *threadData);


/*
equation index: 1841
type: SIMPLE_ASSIGN
z[116] = 1.8
*/
void WhirlpoolDiskStars_eqFunction_1841(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1841};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[915]] /* z[116] STATE(1,vz[116]) */) = 1.8;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3714(DATA *data, threadData_t *threadData);


/*
equation index: 1843
type: SIMPLE_ASSIGN
y[116] = r_init[116] * sin(theta[116] + armOffset[116])
*/
void WhirlpoolDiskStars_eqFunction_1843(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1843};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[755]] /* y[116] STATE(1,vy[116]) */) = ((data->simulationInfo->realParameter[280] /* r_init[116] PARAM */)) * (sin((data->simulationInfo->realParameter[440] /* theta[116] PARAM */) + (data->simulationInfo->realParameter[118] /* armOffset[116] PARAM */)));
  TRACE_POP
}

/*
equation index: 1844
type: SIMPLE_ASSIGN
vx[116] = (-y[116]) * sqrt(G * Md / r_init[116] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1844(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1844};
  modelica_real tmp460;
  modelica_real tmp461;
  tmp460 = (data->simulationInfo->realParameter[280] /* r_init[116] PARAM */);
  tmp461 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp460 * tmp460 * tmp460),"r_init[116] ^ 3.0",equationIndexes);
  if(!(tmp461 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[116] ^ 3.0) was %g should be >= 0", tmp461);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[115]] /* vx[116] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[755]] /* y[116] STATE(1,vy[116]) */))) * (sqrt(tmp461));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3713(DATA *data, threadData_t *threadData);


/*
equation index: 1846
type: SIMPLE_ASSIGN
x[116] = r_init[116] * cos(theta[116] + armOffset[116])
*/
void WhirlpoolDiskStars_eqFunction_1846(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1846};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[595]] /* x[116] STATE(1,vx[116]) */) = ((data->simulationInfo->realParameter[280] /* r_init[116] PARAM */)) * (cos((data->simulationInfo->realParameter[440] /* theta[116] PARAM */) + (data->simulationInfo->realParameter[118] /* armOffset[116] PARAM */)));
  TRACE_POP
}

/*
equation index: 1847
type: SIMPLE_ASSIGN
vy[116] = x[116] * sqrt(G * Md / r_init[116] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1847(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1847};
  modelica_real tmp462;
  modelica_real tmp463;
  tmp462 = (data->simulationInfo->realParameter[280] /* r_init[116] PARAM */);
  tmp463 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp462 * tmp462 * tmp462),"r_init[116] ^ 3.0",equationIndexes);
  if(!(tmp463 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[116] ^ 3.0) was %g should be >= 0", tmp463);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[275]] /* vy[116] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[595]] /* x[116] STATE(1,vx[116]) */)) * (sqrt(tmp463));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3712(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3715(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3717(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3720(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3719(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3718(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3716(DATA *data, threadData_t *threadData);


/*
equation index: 1855
type: SIMPLE_ASSIGN
vz[116] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1855};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[435]] /* vz[116] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3711(DATA *data, threadData_t *threadData);


/*
equation index: 1857
type: SIMPLE_ASSIGN
z[117] = 1.85
*/
void WhirlpoolDiskStars_eqFunction_1857(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1857};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[916]] /* z[117] STATE(1,vz[117]) */) = 1.85;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3724(DATA *data, threadData_t *threadData);


/*
equation index: 1859
type: SIMPLE_ASSIGN
y[117] = r_init[117] * sin(theta[117] + armOffset[117])
*/
void WhirlpoolDiskStars_eqFunction_1859(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1859};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[756]] /* y[117] STATE(1,vy[117]) */) = ((data->simulationInfo->realParameter[281] /* r_init[117] PARAM */)) * (sin((data->simulationInfo->realParameter[441] /* theta[117] PARAM */) + (data->simulationInfo->realParameter[119] /* armOffset[117] PARAM */)));
  TRACE_POP
}

/*
equation index: 1860
type: SIMPLE_ASSIGN
vx[117] = (-y[117]) * sqrt(G * Md / r_init[117] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1860(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1860};
  modelica_real tmp464;
  modelica_real tmp465;
  tmp464 = (data->simulationInfo->realParameter[281] /* r_init[117] PARAM */);
  tmp465 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp464 * tmp464 * tmp464),"r_init[117] ^ 3.0",equationIndexes);
  if(!(tmp465 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[117] ^ 3.0) was %g should be >= 0", tmp465);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[116]] /* vx[117] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[756]] /* y[117] STATE(1,vy[117]) */))) * (sqrt(tmp465));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3723(DATA *data, threadData_t *threadData);


/*
equation index: 1862
type: SIMPLE_ASSIGN
x[117] = r_init[117] * cos(theta[117] + armOffset[117])
*/
void WhirlpoolDiskStars_eqFunction_1862(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1862};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[596]] /* x[117] STATE(1,vx[117]) */) = ((data->simulationInfo->realParameter[281] /* r_init[117] PARAM */)) * (cos((data->simulationInfo->realParameter[441] /* theta[117] PARAM */) + (data->simulationInfo->realParameter[119] /* armOffset[117] PARAM */)));
  TRACE_POP
}

/*
equation index: 1863
type: SIMPLE_ASSIGN
vy[117] = x[117] * sqrt(G * Md / r_init[117] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1863};
  modelica_real tmp466;
  modelica_real tmp467;
  tmp466 = (data->simulationInfo->realParameter[281] /* r_init[117] PARAM */);
  tmp467 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp466 * tmp466 * tmp466),"r_init[117] ^ 3.0",equationIndexes);
  if(!(tmp467 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[117] ^ 3.0) was %g should be >= 0", tmp467);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[276]] /* vy[117] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[596]] /* x[117] STATE(1,vx[117]) */)) * (sqrt(tmp467));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3722(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3725(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3727(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3730(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3729(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3728(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3726(DATA *data, threadData_t *threadData);


/*
equation index: 1871
type: SIMPLE_ASSIGN
vz[117] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1871};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[436]] /* vz[117] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3721(DATA *data, threadData_t *threadData);


/*
equation index: 1873
type: SIMPLE_ASSIGN
z[118] = 1.9000000000000001
*/
void WhirlpoolDiskStars_eqFunction_1873(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1873};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[917]] /* z[118] STATE(1,vz[118]) */) = 1.9000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3734(DATA *data, threadData_t *threadData);


/*
equation index: 1875
type: SIMPLE_ASSIGN
y[118] = r_init[118] * sin(theta[118] + armOffset[118])
*/
void WhirlpoolDiskStars_eqFunction_1875(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1875};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[757]] /* y[118] STATE(1,vy[118]) */) = ((data->simulationInfo->realParameter[282] /* r_init[118] PARAM */)) * (sin((data->simulationInfo->realParameter[442] /* theta[118] PARAM */) + (data->simulationInfo->realParameter[120] /* armOffset[118] PARAM */)));
  TRACE_POP
}

/*
equation index: 1876
type: SIMPLE_ASSIGN
vx[118] = (-y[118]) * sqrt(G * Md / r_init[118] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1876(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1876};
  modelica_real tmp468;
  modelica_real tmp469;
  tmp468 = (data->simulationInfo->realParameter[282] /* r_init[118] PARAM */);
  tmp469 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp468 * tmp468 * tmp468),"r_init[118] ^ 3.0",equationIndexes);
  if(!(tmp469 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[118] ^ 3.0) was %g should be >= 0", tmp469);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[117]] /* vx[118] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[757]] /* y[118] STATE(1,vy[118]) */))) * (sqrt(tmp469));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3733(DATA *data, threadData_t *threadData);


/*
equation index: 1878
type: SIMPLE_ASSIGN
x[118] = r_init[118] * cos(theta[118] + armOffset[118])
*/
void WhirlpoolDiskStars_eqFunction_1878(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1878};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[597]] /* x[118] STATE(1,vx[118]) */) = ((data->simulationInfo->realParameter[282] /* r_init[118] PARAM */)) * (cos((data->simulationInfo->realParameter[442] /* theta[118] PARAM */) + (data->simulationInfo->realParameter[120] /* armOffset[118] PARAM */)));
  TRACE_POP
}

/*
equation index: 1879
type: SIMPLE_ASSIGN
vy[118] = x[118] * sqrt(G * Md / r_init[118] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1879};
  modelica_real tmp470;
  modelica_real tmp471;
  tmp470 = (data->simulationInfo->realParameter[282] /* r_init[118] PARAM */);
  tmp471 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp470 * tmp470 * tmp470),"r_init[118] ^ 3.0",equationIndexes);
  if(!(tmp471 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[118] ^ 3.0) was %g should be >= 0", tmp471);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[277]] /* vy[118] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[597]] /* x[118] STATE(1,vx[118]) */)) * (sqrt(tmp471));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3732(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3735(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3737(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3740(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3739(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3738(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3736(DATA *data, threadData_t *threadData);


/*
equation index: 1887
type: SIMPLE_ASSIGN
vz[118] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1887};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[437]] /* vz[118] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3731(DATA *data, threadData_t *threadData);


/*
equation index: 1889
type: SIMPLE_ASSIGN
z[119] = 1.9500000000000002
*/
void WhirlpoolDiskStars_eqFunction_1889(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1889};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[918]] /* z[119] STATE(1,vz[119]) */) = 1.9500000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3744(DATA *data, threadData_t *threadData);


/*
equation index: 1891
type: SIMPLE_ASSIGN
y[119] = r_init[119] * sin(theta[119] + armOffset[119])
*/
void WhirlpoolDiskStars_eqFunction_1891(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1891};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[758]] /* y[119] STATE(1,vy[119]) */) = ((data->simulationInfo->realParameter[283] /* r_init[119] PARAM */)) * (sin((data->simulationInfo->realParameter[443] /* theta[119] PARAM */) + (data->simulationInfo->realParameter[121] /* armOffset[119] PARAM */)));
  TRACE_POP
}

/*
equation index: 1892
type: SIMPLE_ASSIGN
vx[119] = (-y[119]) * sqrt(G * Md / r_init[119] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1892(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1892};
  modelica_real tmp472;
  modelica_real tmp473;
  tmp472 = (data->simulationInfo->realParameter[283] /* r_init[119] PARAM */);
  tmp473 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp472 * tmp472 * tmp472),"r_init[119] ^ 3.0",equationIndexes);
  if(!(tmp473 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[119] ^ 3.0) was %g should be >= 0", tmp473);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[118]] /* vx[119] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[758]] /* y[119] STATE(1,vy[119]) */))) * (sqrt(tmp473));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3743(DATA *data, threadData_t *threadData);


/*
equation index: 1894
type: SIMPLE_ASSIGN
x[119] = r_init[119] * cos(theta[119] + armOffset[119])
*/
void WhirlpoolDiskStars_eqFunction_1894(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1894};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[598]] /* x[119] STATE(1,vx[119]) */) = ((data->simulationInfo->realParameter[283] /* r_init[119] PARAM */)) * (cos((data->simulationInfo->realParameter[443] /* theta[119] PARAM */) + (data->simulationInfo->realParameter[121] /* armOffset[119] PARAM */)));
  TRACE_POP
}

/*
equation index: 1895
type: SIMPLE_ASSIGN
vy[119] = x[119] * sqrt(G * Md / r_init[119] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1895};
  modelica_real tmp474;
  modelica_real tmp475;
  tmp474 = (data->simulationInfo->realParameter[283] /* r_init[119] PARAM */);
  tmp475 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp474 * tmp474 * tmp474),"r_init[119] ^ 3.0",equationIndexes);
  if(!(tmp475 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[119] ^ 3.0) was %g should be >= 0", tmp475);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[278]] /* vy[119] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[598]] /* x[119] STATE(1,vx[119]) */)) * (sqrt(tmp475));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3742(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3745(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3747(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3750(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3749(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3748(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3746(DATA *data, threadData_t *threadData);


/*
equation index: 1903
type: SIMPLE_ASSIGN
vz[119] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1903};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[438]] /* vz[119] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3741(DATA *data, threadData_t *threadData);


/*
equation index: 1905
type: SIMPLE_ASSIGN
z[120] = 2.0
*/
void WhirlpoolDiskStars_eqFunction_1905(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1905};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[919]] /* z[120] STATE(1,vz[120]) */) = 2.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3754(DATA *data, threadData_t *threadData);


/*
equation index: 1907
type: SIMPLE_ASSIGN
y[120] = r_init[120] * sin(theta[120] + armOffset[120])
*/
void WhirlpoolDiskStars_eqFunction_1907(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1907};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[759]] /* y[120] STATE(1,vy[120]) */) = ((data->simulationInfo->realParameter[284] /* r_init[120] PARAM */)) * (sin((data->simulationInfo->realParameter[444] /* theta[120] PARAM */) + (data->simulationInfo->realParameter[122] /* armOffset[120] PARAM */)));
  TRACE_POP
}

/*
equation index: 1908
type: SIMPLE_ASSIGN
vx[120] = (-y[120]) * sqrt(G * Md / r_init[120] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1908(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1908};
  modelica_real tmp476;
  modelica_real tmp477;
  tmp476 = (data->simulationInfo->realParameter[284] /* r_init[120] PARAM */);
  tmp477 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp476 * tmp476 * tmp476),"r_init[120] ^ 3.0",equationIndexes);
  if(!(tmp477 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[120] ^ 3.0) was %g should be >= 0", tmp477);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[119]] /* vx[120] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[759]] /* y[120] STATE(1,vy[120]) */))) * (sqrt(tmp477));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3753(DATA *data, threadData_t *threadData);


/*
equation index: 1910
type: SIMPLE_ASSIGN
x[120] = r_init[120] * cos(theta[120] + armOffset[120])
*/
void WhirlpoolDiskStars_eqFunction_1910(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1910};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[599]] /* x[120] STATE(1,vx[120]) */) = ((data->simulationInfo->realParameter[284] /* r_init[120] PARAM */)) * (cos((data->simulationInfo->realParameter[444] /* theta[120] PARAM */) + (data->simulationInfo->realParameter[122] /* armOffset[120] PARAM */)));
  TRACE_POP
}

/*
equation index: 1911
type: SIMPLE_ASSIGN
vy[120] = x[120] * sqrt(G * Md / r_init[120] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1911};
  modelica_real tmp478;
  modelica_real tmp479;
  tmp478 = (data->simulationInfo->realParameter[284] /* r_init[120] PARAM */);
  tmp479 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp478 * tmp478 * tmp478),"r_init[120] ^ 3.0",equationIndexes);
  if(!(tmp479 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[120] ^ 3.0) was %g should be >= 0", tmp479);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[279]] /* vy[120] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[599]] /* x[120] STATE(1,vx[120]) */)) * (sqrt(tmp479));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3752(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3755(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3757(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3760(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3759(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3758(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3756(DATA *data, threadData_t *threadData);


/*
equation index: 1919
type: SIMPLE_ASSIGN
vz[120] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1919};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[439]] /* vz[120] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3751(DATA *data, threadData_t *threadData);


/*
equation index: 1921
type: SIMPLE_ASSIGN
z[121] = 2.0500000000000003
*/
void WhirlpoolDiskStars_eqFunction_1921(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1921};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[920]] /* z[121] STATE(1,vz[121]) */) = 2.0500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3764(DATA *data, threadData_t *threadData);


/*
equation index: 1923
type: SIMPLE_ASSIGN
y[121] = r_init[121] * sin(theta[121] + armOffset[121])
*/
void WhirlpoolDiskStars_eqFunction_1923(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1923};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[760]] /* y[121] STATE(1,vy[121]) */) = ((data->simulationInfo->realParameter[285] /* r_init[121] PARAM */)) * (sin((data->simulationInfo->realParameter[445] /* theta[121] PARAM */) + (data->simulationInfo->realParameter[123] /* armOffset[121] PARAM */)));
  TRACE_POP
}

/*
equation index: 1924
type: SIMPLE_ASSIGN
vx[121] = (-y[121]) * sqrt(G * Md / r_init[121] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1924(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1924};
  modelica_real tmp480;
  modelica_real tmp481;
  tmp480 = (data->simulationInfo->realParameter[285] /* r_init[121] PARAM */);
  tmp481 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp480 * tmp480 * tmp480),"r_init[121] ^ 3.0",equationIndexes);
  if(!(tmp481 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[121] ^ 3.0) was %g should be >= 0", tmp481);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[120]] /* vx[121] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[760]] /* y[121] STATE(1,vy[121]) */))) * (sqrt(tmp481));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3763(DATA *data, threadData_t *threadData);


/*
equation index: 1926
type: SIMPLE_ASSIGN
x[121] = r_init[121] * cos(theta[121] + armOffset[121])
*/
void WhirlpoolDiskStars_eqFunction_1926(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1926};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[600]] /* x[121] STATE(1,vx[121]) */) = ((data->simulationInfo->realParameter[285] /* r_init[121] PARAM */)) * (cos((data->simulationInfo->realParameter[445] /* theta[121] PARAM */) + (data->simulationInfo->realParameter[123] /* armOffset[121] PARAM */)));
  TRACE_POP
}

/*
equation index: 1927
type: SIMPLE_ASSIGN
vy[121] = x[121] * sqrt(G * Md / r_init[121] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1927};
  modelica_real tmp482;
  modelica_real tmp483;
  tmp482 = (data->simulationInfo->realParameter[285] /* r_init[121] PARAM */);
  tmp483 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp482 * tmp482 * tmp482),"r_init[121] ^ 3.0",equationIndexes);
  if(!(tmp483 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[121] ^ 3.0) was %g should be >= 0", tmp483);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[280]] /* vy[121] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[600]] /* x[121] STATE(1,vx[121]) */)) * (sqrt(tmp483));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3762(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3765(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3767(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3770(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3769(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3768(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3766(DATA *data, threadData_t *threadData);


/*
equation index: 1935
type: SIMPLE_ASSIGN
vz[121] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1935};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[440]] /* vz[121] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3761(DATA *data, threadData_t *threadData);


/*
equation index: 1937
type: SIMPLE_ASSIGN
z[122] = 2.1
*/
void WhirlpoolDiskStars_eqFunction_1937(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1937};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[921]] /* z[122] STATE(1,vz[122]) */) = 2.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3774(DATA *data, threadData_t *threadData);


/*
equation index: 1939
type: SIMPLE_ASSIGN
y[122] = r_init[122] * sin(theta[122] + armOffset[122])
*/
void WhirlpoolDiskStars_eqFunction_1939(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1939};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[761]] /* y[122] STATE(1,vy[122]) */) = ((data->simulationInfo->realParameter[286] /* r_init[122] PARAM */)) * (sin((data->simulationInfo->realParameter[446] /* theta[122] PARAM */) + (data->simulationInfo->realParameter[124] /* armOffset[122] PARAM */)));
  TRACE_POP
}

/*
equation index: 1940
type: SIMPLE_ASSIGN
vx[122] = (-y[122]) * sqrt(G * Md / r_init[122] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1940(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1940};
  modelica_real tmp484;
  modelica_real tmp485;
  tmp484 = (data->simulationInfo->realParameter[286] /* r_init[122] PARAM */);
  tmp485 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp484 * tmp484 * tmp484),"r_init[122] ^ 3.0",equationIndexes);
  if(!(tmp485 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[122] ^ 3.0) was %g should be >= 0", tmp485);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[121]] /* vx[122] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[761]] /* y[122] STATE(1,vy[122]) */))) * (sqrt(tmp485));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3773(DATA *data, threadData_t *threadData);


/*
equation index: 1942
type: SIMPLE_ASSIGN
x[122] = r_init[122] * cos(theta[122] + armOffset[122])
*/
void WhirlpoolDiskStars_eqFunction_1942(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1942};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[601]] /* x[122] STATE(1,vx[122]) */) = ((data->simulationInfo->realParameter[286] /* r_init[122] PARAM */)) * (cos((data->simulationInfo->realParameter[446] /* theta[122] PARAM */) + (data->simulationInfo->realParameter[124] /* armOffset[122] PARAM */)));
  TRACE_POP
}

/*
equation index: 1943
type: SIMPLE_ASSIGN
vy[122] = x[122] * sqrt(G * Md / r_init[122] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1943};
  modelica_real tmp486;
  modelica_real tmp487;
  tmp486 = (data->simulationInfo->realParameter[286] /* r_init[122] PARAM */);
  tmp487 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp486 * tmp486 * tmp486),"r_init[122] ^ 3.0",equationIndexes);
  if(!(tmp487 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[122] ^ 3.0) was %g should be >= 0", tmp487);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[281]] /* vy[122] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[601]] /* x[122] STATE(1,vx[122]) */)) * (sqrt(tmp487));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3772(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3775(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3777(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3780(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3779(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3778(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3776(DATA *data, threadData_t *threadData);


/*
equation index: 1951
type: SIMPLE_ASSIGN
vz[122] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1951};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[441]] /* vz[122] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3771(DATA *data, threadData_t *threadData);


/*
equation index: 1953
type: SIMPLE_ASSIGN
z[123] = 2.15
*/
void WhirlpoolDiskStars_eqFunction_1953(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1953};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[922]] /* z[123] STATE(1,vz[123]) */) = 2.15;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3784(DATA *data, threadData_t *threadData);


/*
equation index: 1955
type: SIMPLE_ASSIGN
y[123] = r_init[123] * sin(theta[123] + armOffset[123])
*/
void WhirlpoolDiskStars_eqFunction_1955(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1955};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[762]] /* y[123] STATE(1,vy[123]) */) = ((data->simulationInfo->realParameter[287] /* r_init[123] PARAM */)) * (sin((data->simulationInfo->realParameter[447] /* theta[123] PARAM */) + (data->simulationInfo->realParameter[125] /* armOffset[123] PARAM */)));
  TRACE_POP
}

/*
equation index: 1956
type: SIMPLE_ASSIGN
vx[123] = (-y[123]) * sqrt(G * Md / r_init[123] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1956(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1956};
  modelica_real tmp488;
  modelica_real tmp489;
  tmp488 = (data->simulationInfo->realParameter[287] /* r_init[123] PARAM */);
  tmp489 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp488 * tmp488 * tmp488),"r_init[123] ^ 3.0",equationIndexes);
  if(!(tmp489 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[123] ^ 3.0) was %g should be >= 0", tmp489);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[122]] /* vx[123] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[762]] /* y[123] STATE(1,vy[123]) */))) * (sqrt(tmp489));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3783(DATA *data, threadData_t *threadData);


/*
equation index: 1958
type: SIMPLE_ASSIGN
x[123] = r_init[123] * cos(theta[123] + armOffset[123])
*/
void WhirlpoolDiskStars_eqFunction_1958(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1958};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[602]] /* x[123] STATE(1,vx[123]) */) = ((data->simulationInfo->realParameter[287] /* r_init[123] PARAM */)) * (cos((data->simulationInfo->realParameter[447] /* theta[123] PARAM */) + (data->simulationInfo->realParameter[125] /* armOffset[123] PARAM */)));
  TRACE_POP
}

/*
equation index: 1959
type: SIMPLE_ASSIGN
vy[123] = x[123] * sqrt(G * Md / r_init[123] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1959};
  modelica_real tmp490;
  modelica_real tmp491;
  tmp490 = (data->simulationInfo->realParameter[287] /* r_init[123] PARAM */);
  tmp491 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp490 * tmp490 * tmp490),"r_init[123] ^ 3.0",equationIndexes);
  if(!(tmp491 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[123] ^ 3.0) was %g should be >= 0", tmp491);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[282]] /* vy[123] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[602]] /* x[123] STATE(1,vx[123]) */)) * (sqrt(tmp491));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3782(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3785(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3787(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3790(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3789(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3788(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3786(DATA *data, threadData_t *threadData);


/*
equation index: 1967
type: SIMPLE_ASSIGN
vz[123] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1967};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[442]] /* vz[123] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3781(DATA *data, threadData_t *threadData);


/*
equation index: 1969
type: SIMPLE_ASSIGN
z[124] = 2.2
*/
void WhirlpoolDiskStars_eqFunction_1969(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1969};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[923]] /* z[124] STATE(1,vz[124]) */) = 2.2;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3794(DATA *data, threadData_t *threadData);


/*
equation index: 1971
type: SIMPLE_ASSIGN
y[124] = r_init[124] * sin(theta[124] + armOffset[124])
*/
void WhirlpoolDiskStars_eqFunction_1971(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1971};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[763]] /* y[124] STATE(1,vy[124]) */) = ((data->simulationInfo->realParameter[288] /* r_init[124] PARAM */)) * (sin((data->simulationInfo->realParameter[448] /* theta[124] PARAM */) + (data->simulationInfo->realParameter[126] /* armOffset[124] PARAM */)));
  TRACE_POP
}

/*
equation index: 1972
type: SIMPLE_ASSIGN
vx[124] = (-y[124]) * sqrt(G * Md / r_init[124] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1972(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1972};
  modelica_real tmp492;
  modelica_real tmp493;
  tmp492 = (data->simulationInfo->realParameter[288] /* r_init[124] PARAM */);
  tmp493 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp492 * tmp492 * tmp492),"r_init[124] ^ 3.0",equationIndexes);
  if(!(tmp493 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[124] ^ 3.0) was %g should be >= 0", tmp493);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[123]] /* vx[124] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[763]] /* y[124] STATE(1,vy[124]) */))) * (sqrt(tmp493));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3793(DATA *data, threadData_t *threadData);


/*
equation index: 1974
type: SIMPLE_ASSIGN
x[124] = r_init[124] * cos(theta[124] + armOffset[124])
*/
void WhirlpoolDiskStars_eqFunction_1974(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1974};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[603]] /* x[124] STATE(1,vx[124]) */) = ((data->simulationInfo->realParameter[288] /* r_init[124] PARAM */)) * (cos((data->simulationInfo->realParameter[448] /* theta[124] PARAM */) + (data->simulationInfo->realParameter[126] /* armOffset[124] PARAM */)));
  TRACE_POP
}

/*
equation index: 1975
type: SIMPLE_ASSIGN
vy[124] = x[124] * sqrt(G * Md / r_init[124] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1975};
  modelica_real tmp494;
  modelica_real tmp495;
  tmp494 = (data->simulationInfo->realParameter[288] /* r_init[124] PARAM */);
  tmp495 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp494 * tmp494 * tmp494),"r_init[124] ^ 3.0",equationIndexes);
  if(!(tmp495 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[124] ^ 3.0) was %g should be >= 0", tmp495);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[283]] /* vy[124] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[603]] /* x[124] STATE(1,vx[124]) */)) * (sqrt(tmp495));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3792(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3795(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3797(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3800(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3799(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3798(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3796(DATA *data, threadData_t *threadData);


/*
equation index: 1983
type: SIMPLE_ASSIGN
vz[124] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1983};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[443]] /* vz[124] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3791(DATA *data, threadData_t *threadData);


/*
equation index: 1985
type: SIMPLE_ASSIGN
z[125] = 2.25
*/
void WhirlpoolDiskStars_eqFunction_1985(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1985};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[924]] /* z[125] STATE(1,vz[125]) */) = 2.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3804(DATA *data, threadData_t *threadData);


/*
equation index: 1987
type: SIMPLE_ASSIGN
y[125] = r_init[125] * sin(theta[125] + armOffset[125])
*/
void WhirlpoolDiskStars_eqFunction_1987(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1987};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[764]] /* y[125] STATE(1,vy[125]) */) = ((data->simulationInfo->realParameter[289] /* r_init[125] PARAM */)) * (sin((data->simulationInfo->realParameter[449] /* theta[125] PARAM */) + (data->simulationInfo->realParameter[127] /* armOffset[125] PARAM */)));
  TRACE_POP
}

/*
equation index: 1988
type: SIMPLE_ASSIGN
vx[125] = (-y[125]) * sqrt(G * Md / r_init[125] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1988(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1988};
  modelica_real tmp496;
  modelica_real tmp497;
  tmp496 = (data->simulationInfo->realParameter[289] /* r_init[125] PARAM */);
  tmp497 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp496 * tmp496 * tmp496),"r_init[125] ^ 3.0",equationIndexes);
  if(!(tmp497 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[125] ^ 3.0) was %g should be >= 0", tmp497);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[124]] /* vx[125] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[764]] /* y[125] STATE(1,vy[125]) */))) * (sqrt(tmp497));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3803(DATA *data, threadData_t *threadData);


/*
equation index: 1990
type: SIMPLE_ASSIGN
x[125] = r_init[125] * cos(theta[125] + armOffset[125])
*/
void WhirlpoolDiskStars_eqFunction_1990(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1990};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[604]] /* x[125] STATE(1,vx[125]) */) = ((data->simulationInfo->realParameter[289] /* r_init[125] PARAM */)) * (cos((data->simulationInfo->realParameter[449] /* theta[125] PARAM */) + (data->simulationInfo->realParameter[127] /* armOffset[125] PARAM */)));
  TRACE_POP
}

/*
equation index: 1991
type: SIMPLE_ASSIGN
vy[125] = x[125] * sqrt(G * Md / r_init[125] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1991};
  modelica_real tmp498;
  modelica_real tmp499;
  tmp498 = (data->simulationInfo->realParameter[289] /* r_init[125] PARAM */);
  tmp499 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp498 * tmp498 * tmp498),"r_init[125] ^ 3.0",equationIndexes);
  if(!(tmp499 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[125] ^ 3.0) was %g should be >= 0", tmp499);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[284]] /* vy[125] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[604]] /* x[125] STATE(1,vx[125]) */)) * (sqrt(tmp499));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3802(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3805(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3807(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3810(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3809(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3808(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3806(DATA *data, threadData_t *threadData);


/*
equation index: 1999
type: SIMPLE_ASSIGN
vz[125] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1999};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[444]] /* vz[125] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3801(DATA *data, threadData_t *threadData);


/*
equation index: 2001
type: SIMPLE_ASSIGN
z[126] = 2.3000000000000003
*/
void WhirlpoolDiskStars_eqFunction_2001(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2001};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[925]] /* z[126] STATE(1,vz[126]) */) = 2.3000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3814(DATA *data, threadData_t *threadData);


/*
equation index: 2003
type: SIMPLE_ASSIGN
y[126] = r_init[126] * sin(theta[126] + armOffset[126])
*/
void WhirlpoolDiskStars_eqFunction_2003(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2003};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[765]] /* y[126] STATE(1,vy[126]) */) = ((data->simulationInfo->realParameter[290] /* r_init[126] PARAM */)) * (sin((data->simulationInfo->realParameter[450] /* theta[126] PARAM */) + (data->simulationInfo->realParameter[128] /* armOffset[126] PARAM */)));
  TRACE_POP
}

/*
equation index: 2004
type: SIMPLE_ASSIGN
vx[126] = (-y[126]) * sqrt(G * Md / r_init[126] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2004(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2004};
  modelica_real tmp500;
  modelica_real tmp501;
  tmp500 = (data->simulationInfo->realParameter[290] /* r_init[126] PARAM */);
  tmp501 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp500 * tmp500 * tmp500),"r_init[126] ^ 3.0",equationIndexes);
  if(!(tmp501 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[126] ^ 3.0) was %g should be >= 0", tmp501);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[125]] /* vx[126] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[765]] /* y[126] STATE(1,vy[126]) */))) * (sqrt(tmp501));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3813(DATA *data, threadData_t *threadData);


/*
equation index: 2006
type: SIMPLE_ASSIGN
x[126] = r_init[126] * cos(theta[126] + armOffset[126])
*/
void WhirlpoolDiskStars_eqFunction_2006(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2006};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[605]] /* x[126] STATE(1,vx[126]) */) = ((data->simulationInfo->realParameter[290] /* r_init[126] PARAM */)) * (cos((data->simulationInfo->realParameter[450] /* theta[126] PARAM */) + (data->simulationInfo->realParameter[128] /* armOffset[126] PARAM */)));
  TRACE_POP
}

/*
equation index: 2007
type: SIMPLE_ASSIGN
vy[126] = x[126] * sqrt(G * Md / r_init[126] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2007};
  modelica_real tmp502;
  modelica_real tmp503;
  tmp502 = (data->simulationInfo->realParameter[290] /* r_init[126] PARAM */);
  tmp503 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp502 * tmp502 * tmp502),"r_init[126] ^ 3.0",equationIndexes);
  if(!(tmp503 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[126] ^ 3.0) was %g should be >= 0", tmp503);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[285]] /* vy[126] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[605]] /* x[126] STATE(1,vx[126]) */)) * (sqrt(tmp503));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3812(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3815(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3817(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3820(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3819(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3818(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3816(DATA *data, threadData_t *threadData);


/*
equation index: 2015
type: SIMPLE_ASSIGN
vz[126] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2015};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[445]] /* vz[126] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3811(DATA *data, threadData_t *threadData);


/*
equation index: 2017
type: SIMPLE_ASSIGN
z[127] = 2.35
*/
void WhirlpoolDiskStars_eqFunction_2017(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2017};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[926]] /* z[127] STATE(1,vz[127]) */) = 2.35;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3824(DATA *data, threadData_t *threadData);


/*
equation index: 2019
type: SIMPLE_ASSIGN
y[127] = r_init[127] * sin(theta[127] + armOffset[127])
*/
void WhirlpoolDiskStars_eqFunction_2019(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2019};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[766]] /* y[127] STATE(1,vy[127]) */) = ((data->simulationInfo->realParameter[291] /* r_init[127] PARAM */)) * (sin((data->simulationInfo->realParameter[451] /* theta[127] PARAM */) + (data->simulationInfo->realParameter[129] /* armOffset[127] PARAM */)));
  TRACE_POP
}

/*
equation index: 2020
type: SIMPLE_ASSIGN
vx[127] = (-y[127]) * sqrt(G * Md / r_init[127] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2020(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2020};
  modelica_real tmp504;
  modelica_real tmp505;
  tmp504 = (data->simulationInfo->realParameter[291] /* r_init[127] PARAM */);
  tmp505 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp504 * tmp504 * tmp504),"r_init[127] ^ 3.0",equationIndexes);
  if(!(tmp505 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[127] ^ 3.0) was %g should be >= 0", tmp505);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[126]] /* vx[127] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[766]] /* y[127] STATE(1,vy[127]) */))) * (sqrt(tmp505));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3823(DATA *data, threadData_t *threadData);


/*
equation index: 2022
type: SIMPLE_ASSIGN
x[127] = r_init[127] * cos(theta[127] + armOffset[127])
*/
void WhirlpoolDiskStars_eqFunction_2022(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2022};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[606]] /* x[127] STATE(1,vx[127]) */) = ((data->simulationInfo->realParameter[291] /* r_init[127] PARAM */)) * (cos((data->simulationInfo->realParameter[451] /* theta[127] PARAM */) + (data->simulationInfo->realParameter[129] /* armOffset[127] PARAM */)));
  TRACE_POP
}

/*
equation index: 2023
type: SIMPLE_ASSIGN
vy[127] = x[127] * sqrt(G * Md / r_init[127] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2023};
  modelica_real tmp506;
  modelica_real tmp507;
  tmp506 = (data->simulationInfo->realParameter[291] /* r_init[127] PARAM */);
  tmp507 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp506 * tmp506 * tmp506),"r_init[127] ^ 3.0",equationIndexes);
  if(!(tmp507 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[127] ^ 3.0) was %g should be >= 0", tmp507);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[286]] /* vy[127] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[606]] /* x[127] STATE(1,vx[127]) */)) * (sqrt(tmp507));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3822(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3825(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3827(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3830(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3829(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3828(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3826(DATA *data, threadData_t *threadData);


/*
equation index: 2031
type: SIMPLE_ASSIGN
vz[127] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2031};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[446]] /* vz[127] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3821(DATA *data, threadData_t *threadData);


/*
equation index: 2033
type: SIMPLE_ASSIGN
z[128] = 2.4000000000000004
*/
void WhirlpoolDiskStars_eqFunction_2033(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2033};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[927]] /* z[128] STATE(1,vz[128]) */) = 2.4000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3834(DATA *data, threadData_t *threadData);


/*
equation index: 2035
type: SIMPLE_ASSIGN
y[128] = r_init[128] * sin(theta[128] + armOffset[128])
*/
void WhirlpoolDiskStars_eqFunction_2035(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2035};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[767]] /* y[128] STATE(1,vy[128]) */) = ((data->simulationInfo->realParameter[292] /* r_init[128] PARAM */)) * (sin((data->simulationInfo->realParameter[452] /* theta[128] PARAM */) + (data->simulationInfo->realParameter[130] /* armOffset[128] PARAM */)));
  TRACE_POP
}

/*
equation index: 2036
type: SIMPLE_ASSIGN
vx[128] = (-y[128]) * sqrt(G * Md / r_init[128] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2036(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2036};
  modelica_real tmp508;
  modelica_real tmp509;
  tmp508 = (data->simulationInfo->realParameter[292] /* r_init[128] PARAM */);
  tmp509 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp508 * tmp508 * tmp508),"r_init[128] ^ 3.0",equationIndexes);
  if(!(tmp509 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[128] ^ 3.0) was %g should be >= 0", tmp509);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[127]] /* vx[128] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[767]] /* y[128] STATE(1,vy[128]) */))) * (sqrt(tmp509));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3833(DATA *data, threadData_t *threadData);


/*
equation index: 2038
type: SIMPLE_ASSIGN
x[128] = r_init[128] * cos(theta[128] + armOffset[128])
*/
void WhirlpoolDiskStars_eqFunction_2038(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2038};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[607]] /* x[128] STATE(1,vx[128]) */) = ((data->simulationInfo->realParameter[292] /* r_init[128] PARAM */)) * (cos((data->simulationInfo->realParameter[452] /* theta[128] PARAM */) + (data->simulationInfo->realParameter[130] /* armOffset[128] PARAM */)));
  TRACE_POP
}

/*
equation index: 2039
type: SIMPLE_ASSIGN
vy[128] = x[128] * sqrt(G * Md / r_init[128] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2039};
  modelica_real tmp510;
  modelica_real tmp511;
  tmp510 = (data->simulationInfo->realParameter[292] /* r_init[128] PARAM */);
  tmp511 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp510 * tmp510 * tmp510),"r_init[128] ^ 3.0",equationIndexes);
  if(!(tmp511 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[128] ^ 3.0) was %g should be >= 0", tmp511);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[287]] /* vy[128] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[607]] /* x[128] STATE(1,vx[128]) */)) * (sqrt(tmp511));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3832(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3835(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3837(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3840(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3839(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3838(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3836(DATA *data, threadData_t *threadData);


/*
equation index: 2047
type: SIMPLE_ASSIGN
vz[128] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2047};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[447]] /* vz[128] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3831(DATA *data, threadData_t *threadData);


/*
equation index: 2049
type: SIMPLE_ASSIGN
z[129] = 2.45
*/
void WhirlpoolDiskStars_eqFunction_2049(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2049};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[928]] /* z[129] STATE(1,vz[129]) */) = 2.45;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3844(DATA *data, threadData_t *threadData);


/*
equation index: 2051
type: SIMPLE_ASSIGN
y[129] = r_init[129] * sin(theta[129] + armOffset[129])
*/
void WhirlpoolDiskStars_eqFunction_2051(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2051};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[768]] /* y[129] STATE(1,vy[129]) */) = ((data->simulationInfo->realParameter[293] /* r_init[129] PARAM */)) * (sin((data->simulationInfo->realParameter[453] /* theta[129] PARAM */) + (data->simulationInfo->realParameter[131] /* armOffset[129] PARAM */)));
  TRACE_POP
}

/*
equation index: 2052
type: SIMPLE_ASSIGN
vx[129] = (-y[129]) * sqrt(G * Md / r_init[129] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2052(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2052};
  modelica_real tmp512;
  modelica_real tmp513;
  tmp512 = (data->simulationInfo->realParameter[293] /* r_init[129] PARAM */);
  tmp513 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp512 * tmp512 * tmp512),"r_init[129] ^ 3.0",equationIndexes);
  if(!(tmp513 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[129] ^ 3.0) was %g should be >= 0", tmp513);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[128]] /* vx[129] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[768]] /* y[129] STATE(1,vy[129]) */))) * (sqrt(tmp513));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3843(DATA *data, threadData_t *threadData);


/*
equation index: 2054
type: SIMPLE_ASSIGN
x[129] = r_init[129] * cos(theta[129] + armOffset[129])
*/
void WhirlpoolDiskStars_eqFunction_2054(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2054};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[608]] /* x[129] STATE(1,vx[129]) */) = ((data->simulationInfo->realParameter[293] /* r_init[129] PARAM */)) * (cos((data->simulationInfo->realParameter[453] /* theta[129] PARAM */) + (data->simulationInfo->realParameter[131] /* armOffset[129] PARAM */)));
  TRACE_POP
}

/*
equation index: 2055
type: SIMPLE_ASSIGN
vy[129] = x[129] * sqrt(G * Md / r_init[129] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2055};
  modelica_real tmp514;
  modelica_real tmp515;
  tmp514 = (data->simulationInfo->realParameter[293] /* r_init[129] PARAM */);
  tmp515 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp514 * tmp514 * tmp514),"r_init[129] ^ 3.0",equationIndexes);
  if(!(tmp515 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[129] ^ 3.0) was %g should be >= 0", tmp515);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[288]] /* vy[129] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[608]] /* x[129] STATE(1,vx[129]) */)) * (sqrt(tmp515));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3842(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3845(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3847(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3850(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3849(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3848(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3846(DATA *data, threadData_t *threadData);


/*
equation index: 2063
type: SIMPLE_ASSIGN
vz[129] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2063};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[448]] /* vz[129] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3841(DATA *data, threadData_t *threadData);


/*
equation index: 2065
type: SIMPLE_ASSIGN
z[130] = 2.5
*/
void WhirlpoolDiskStars_eqFunction_2065(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2065};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[929]] /* z[130] STATE(1,vz[130]) */) = 2.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3854(DATA *data, threadData_t *threadData);


/*
equation index: 2067
type: SIMPLE_ASSIGN
y[130] = r_init[130] * sin(theta[130] + armOffset[130])
*/
void WhirlpoolDiskStars_eqFunction_2067(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2067};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[769]] /* y[130] STATE(1,vy[130]) */) = ((data->simulationInfo->realParameter[294] /* r_init[130] PARAM */)) * (sin((data->simulationInfo->realParameter[454] /* theta[130] PARAM */) + (data->simulationInfo->realParameter[132] /* armOffset[130] PARAM */)));
  TRACE_POP
}

/*
equation index: 2068
type: SIMPLE_ASSIGN
vx[130] = (-y[130]) * sqrt(G * Md / r_init[130] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2068(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2068};
  modelica_real tmp516;
  modelica_real tmp517;
  tmp516 = (data->simulationInfo->realParameter[294] /* r_init[130] PARAM */);
  tmp517 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp516 * tmp516 * tmp516),"r_init[130] ^ 3.0",equationIndexes);
  if(!(tmp517 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[130] ^ 3.0) was %g should be >= 0", tmp517);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[129]] /* vx[130] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[769]] /* y[130] STATE(1,vy[130]) */))) * (sqrt(tmp517));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3853(DATA *data, threadData_t *threadData);


/*
equation index: 2070
type: SIMPLE_ASSIGN
x[130] = r_init[130] * cos(theta[130] + armOffset[130])
*/
void WhirlpoolDiskStars_eqFunction_2070(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2070};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[609]] /* x[130] STATE(1,vx[130]) */) = ((data->simulationInfo->realParameter[294] /* r_init[130] PARAM */)) * (cos((data->simulationInfo->realParameter[454] /* theta[130] PARAM */) + (data->simulationInfo->realParameter[132] /* armOffset[130] PARAM */)));
  TRACE_POP
}

/*
equation index: 2071
type: SIMPLE_ASSIGN
vy[130] = x[130] * sqrt(G * Md / r_init[130] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2071};
  modelica_real tmp518;
  modelica_real tmp519;
  tmp518 = (data->simulationInfo->realParameter[294] /* r_init[130] PARAM */);
  tmp519 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp518 * tmp518 * tmp518),"r_init[130] ^ 3.0",equationIndexes);
  if(!(tmp519 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[130] ^ 3.0) was %g should be >= 0", tmp519);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[289]] /* vy[130] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[609]] /* x[130] STATE(1,vx[130]) */)) * (sqrt(tmp519));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3852(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3855(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3857(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3860(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3859(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3858(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3856(DATA *data, threadData_t *threadData);


/*
equation index: 2079
type: SIMPLE_ASSIGN
vz[130] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2079};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[449]] /* vz[130] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3851(DATA *data, threadData_t *threadData);


/*
equation index: 2081
type: SIMPLE_ASSIGN
z[131] = 2.5500000000000003
*/
void WhirlpoolDiskStars_eqFunction_2081(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2081};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[930]] /* z[131] STATE(1,vz[131]) */) = 2.5500000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3864(DATA *data, threadData_t *threadData);


/*
equation index: 2083
type: SIMPLE_ASSIGN
y[131] = r_init[131] * sin(theta[131] + armOffset[131])
*/
void WhirlpoolDiskStars_eqFunction_2083(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2083};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[770]] /* y[131] STATE(1,vy[131]) */) = ((data->simulationInfo->realParameter[295] /* r_init[131] PARAM */)) * (sin((data->simulationInfo->realParameter[455] /* theta[131] PARAM */) + (data->simulationInfo->realParameter[133] /* armOffset[131] PARAM */)));
  TRACE_POP
}

/*
equation index: 2084
type: SIMPLE_ASSIGN
vx[131] = (-y[131]) * sqrt(G * Md / r_init[131] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2084(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2084};
  modelica_real tmp520;
  modelica_real tmp521;
  tmp520 = (data->simulationInfo->realParameter[295] /* r_init[131] PARAM */);
  tmp521 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp520 * tmp520 * tmp520),"r_init[131] ^ 3.0",equationIndexes);
  if(!(tmp521 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[131] ^ 3.0) was %g should be >= 0", tmp521);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[130]] /* vx[131] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[770]] /* y[131] STATE(1,vy[131]) */))) * (sqrt(tmp521));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3863(DATA *data, threadData_t *threadData);


/*
equation index: 2086
type: SIMPLE_ASSIGN
x[131] = r_init[131] * cos(theta[131] + armOffset[131])
*/
void WhirlpoolDiskStars_eqFunction_2086(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2086};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[610]] /* x[131] STATE(1,vx[131]) */) = ((data->simulationInfo->realParameter[295] /* r_init[131] PARAM */)) * (cos((data->simulationInfo->realParameter[455] /* theta[131] PARAM */) + (data->simulationInfo->realParameter[133] /* armOffset[131] PARAM */)));
  TRACE_POP
}

/*
equation index: 2087
type: SIMPLE_ASSIGN
vy[131] = x[131] * sqrt(G * Md / r_init[131] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2087};
  modelica_real tmp522;
  modelica_real tmp523;
  tmp522 = (data->simulationInfo->realParameter[295] /* r_init[131] PARAM */);
  tmp523 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp522 * tmp522 * tmp522),"r_init[131] ^ 3.0",equationIndexes);
  if(!(tmp523 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[131] ^ 3.0) was %g should be >= 0", tmp523);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[290]] /* vy[131] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[610]] /* x[131] STATE(1,vx[131]) */)) * (sqrt(tmp523));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3862(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3865(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3867(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3870(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3869(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3868(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3866(DATA *data, threadData_t *threadData);


/*
equation index: 2095
type: SIMPLE_ASSIGN
vz[131] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2095};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[450]] /* vz[131] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3861(DATA *data, threadData_t *threadData);


/*
equation index: 2097
type: SIMPLE_ASSIGN
z[132] = 2.6
*/
void WhirlpoolDiskStars_eqFunction_2097(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2097};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[931]] /* z[132] STATE(1,vz[132]) */) = 2.6;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3874(DATA *data, threadData_t *threadData);


/*
equation index: 2099
type: SIMPLE_ASSIGN
y[132] = r_init[132] * sin(theta[132] + armOffset[132])
*/
void WhirlpoolDiskStars_eqFunction_2099(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2099};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[771]] /* y[132] STATE(1,vy[132]) */) = ((data->simulationInfo->realParameter[296] /* r_init[132] PARAM */)) * (sin((data->simulationInfo->realParameter[456] /* theta[132] PARAM */) + (data->simulationInfo->realParameter[134] /* armOffset[132] PARAM */)));
  TRACE_POP
}

/*
equation index: 2100
type: SIMPLE_ASSIGN
vx[132] = (-y[132]) * sqrt(G * Md / r_init[132] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2100(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2100};
  modelica_real tmp524;
  modelica_real tmp525;
  tmp524 = (data->simulationInfo->realParameter[296] /* r_init[132] PARAM */);
  tmp525 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp524 * tmp524 * tmp524),"r_init[132] ^ 3.0",equationIndexes);
  if(!(tmp525 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[132] ^ 3.0) was %g should be >= 0", tmp525);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[131]] /* vx[132] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[771]] /* y[132] STATE(1,vy[132]) */))) * (sqrt(tmp525));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3873(DATA *data, threadData_t *threadData);


/*
equation index: 2102
type: SIMPLE_ASSIGN
x[132] = r_init[132] * cos(theta[132] + armOffset[132])
*/
void WhirlpoolDiskStars_eqFunction_2102(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2102};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[611]] /* x[132] STATE(1,vx[132]) */) = ((data->simulationInfo->realParameter[296] /* r_init[132] PARAM */)) * (cos((data->simulationInfo->realParameter[456] /* theta[132] PARAM */) + (data->simulationInfo->realParameter[134] /* armOffset[132] PARAM */)));
  TRACE_POP
}

/*
equation index: 2103
type: SIMPLE_ASSIGN
vy[132] = x[132] * sqrt(G * Md / r_init[132] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2103};
  modelica_real tmp526;
  modelica_real tmp527;
  tmp526 = (data->simulationInfo->realParameter[296] /* r_init[132] PARAM */);
  tmp527 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp526 * tmp526 * tmp526),"r_init[132] ^ 3.0",equationIndexes);
  if(!(tmp527 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[132] ^ 3.0) was %g should be >= 0", tmp527);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[291]] /* vy[132] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[611]] /* x[132] STATE(1,vx[132]) */)) * (sqrt(tmp527));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3872(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3875(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3877(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3880(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3879(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3878(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3876(DATA *data, threadData_t *threadData);


/*
equation index: 2111
type: SIMPLE_ASSIGN
vz[132] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2111};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[451]] /* vz[132] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3871(DATA *data, threadData_t *threadData);


/*
equation index: 2113
type: SIMPLE_ASSIGN
z[133] = 2.6500000000000004
*/
void WhirlpoolDiskStars_eqFunction_2113(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2113};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[932]] /* z[133] STATE(1,vz[133]) */) = 2.6500000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3884(DATA *data, threadData_t *threadData);


/*
equation index: 2115
type: SIMPLE_ASSIGN
y[133] = r_init[133] * sin(theta[133] + armOffset[133])
*/
void WhirlpoolDiskStars_eqFunction_2115(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2115};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[772]] /* y[133] STATE(1,vy[133]) */) = ((data->simulationInfo->realParameter[297] /* r_init[133] PARAM */)) * (sin((data->simulationInfo->realParameter[457] /* theta[133] PARAM */) + (data->simulationInfo->realParameter[135] /* armOffset[133] PARAM */)));
  TRACE_POP
}

/*
equation index: 2116
type: SIMPLE_ASSIGN
vx[133] = (-y[133]) * sqrt(G * Md / r_init[133] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2116(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2116};
  modelica_real tmp528;
  modelica_real tmp529;
  tmp528 = (data->simulationInfo->realParameter[297] /* r_init[133] PARAM */);
  tmp529 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp528 * tmp528 * tmp528),"r_init[133] ^ 3.0",equationIndexes);
  if(!(tmp529 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[133] ^ 3.0) was %g should be >= 0", tmp529);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[132]] /* vx[133] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[772]] /* y[133] STATE(1,vy[133]) */))) * (sqrt(tmp529));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3883(DATA *data, threadData_t *threadData);


/*
equation index: 2118
type: SIMPLE_ASSIGN
x[133] = r_init[133] * cos(theta[133] + armOffset[133])
*/
void WhirlpoolDiskStars_eqFunction_2118(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2118};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[612]] /* x[133] STATE(1,vx[133]) */) = ((data->simulationInfo->realParameter[297] /* r_init[133] PARAM */)) * (cos((data->simulationInfo->realParameter[457] /* theta[133] PARAM */) + (data->simulationInfo->realParameter[135] /* armOffset[133] PARAM */)));
  TRACE_POP
}

/*
equation index: 2119
type: SIMPLE_ASSIGN
vy[133] = x[133] * sqrt(G * Md / r_init[133] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2119};
  modelica_real tmp530;
  modelica_real tmp531;
  tmp530 = (data->simulationInfo->realParameter[297] /* r_init[133] PARAM */);
  tmp531 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp530 * tmp530 * tmp530),"r_init[133] ^ 3.0",equationIndexes);
  if(!(tmp531 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[133] ^ 3.0) was %g should be >= 0", tmp531);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[292]] /* vy[133] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[612]] /* x[133] STATE(1,vx[133]) */)) * (sqrt(tmp531));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3882(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3885(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3887(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3890(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3889(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3888(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3886(DATA *data, threadData_t *threadData);


/*
equation index: 2127
type: SIMPLE_ASSIGN
vz[133] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_2127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2127};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[452]] /* vz[133] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3881(DATA *data, threadData_t *threadData);


/*
equation index: 2129
type: SIMPLE_ASSIGN
z[134] = 2.7
*/
void WhirlpoolDiskStars_eqFunction_2129(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2129};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[933]] /* z[134] STATE(1,vz[134]) */) = 2.7;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3894(DATA *data, threadData_t *threadData);


/*
equation index: 2131
type: SIMPLE_ASSIGN
y[134] = r_init[134] * sin(theta[134] + armOffset[134])
*/
void WhirlpoolDiskStars_eqFunction_2131(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2131};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[773]] /* y[134] STATE(1,vy[134]) */) = ((data->simulationInfo->realParameter[298] /* r_init[134] PARAM */)) * (sin((data->simulationInfo->realParameter[458] /* theta[134] PARAM */) + (data->simulationInfo->realParameter[136] /* armOffset[134] PARAM */)));
  TRACE_POP
}

/*
equation index: 2132
type: SIMPLE_ASSIGN
vx[134] = (-y[134]) * sqrt(G * Md / r_init[134] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2132(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2132};
  modelica_real tmp532;
  modelica_real tmp533;
  tmp532 = (data->simulationInfo->realParameter[298] /* r_init[134] PARAM */);
  tmp533 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp532 * tmp532 * tmp532),"r_init[134] ^ 3.0",equationIndexes);
  if(!(tmp533 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[134] ^ 3.0) was %g should be >= 0", tmp533);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[133]] /* vx[134] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[773]] /* y[134] STATE(1,vy[134]) */))) * (sqrt(tmp533));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3893(DATA *data, threadData_t *threadData);


/*
equation index: 2134
type: SIMPLE_ASSIGN
x[134] = r_init[134] * cos(theta[134] + armOffset[134])
*/
void WhirlpoolDiskStars_eqFunction_2134(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2134};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[613]] /* x[134] STATE(1,vx[134]) */) = ((data->simulationInfo->realParameter[298] /* r_init[134] PARAM */)) * (cos((data->simulationInfo->realParameter[458] /* theta[134] PARAM */) + (data->simulationInfo->realParameter[136] /* armOffset[134] PARAM */)));
  TRACE_POP
}

/*
equation index: 2135
type: SIMPLE_ASSIGN
vy[134] = x[134] * sqrt(G * Md / r_init[134] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_2135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,2135};
  modelica_real tmp534;
  modelica_real tmp535;
  tmp534 = (data->simulationInfo->realParameter[298] /* r_init[134] PARAM */);
  tmp535 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp534 * tmp534 * tmp534),"r_init[134] ^ 3.0",equationIndexes);
  if(!(tmp535 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[134] ^ 3.0) was %g should be >= 0", tmp535);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[293]] /* vy[134] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[613]] /* x[134] STATE(1,vx[134]) */)) * (sqrt(tmp535));
  TRACE_POP
}
OMC_DISABLE_OPT
void WhirlpoolDiskStars_functionInitialEquations_4(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_3628(data, threadData);
  WhirlpoolDiskStars_eqFunction_3626(data, threadData);
  WhirlpoolDiskStars_eqFunction_1711(data, threadData);
  WhirlpoolDiskStars_eqFunction_3621(data, threadData);
  WhirlpoolDiskStars_eqFunction_1713(data, threadData);
  WhirlpoolDiskStars_eqFunction_3634(data, threadData);
  WhirlpoolDiskStars_eqFunction_1715(data, threadData);
  WhirlpoolDiskStars_eqFunction_1716(data, threadData);
  WhirlpoolDiskStars_eqFunction_3633(data, threadData);
  WhirlpoolDiskStars_eqFunction_1718(data, threadData);
  WhirlpoolDiskStars_eqFunction_1719(data, threadData);
  WhirlpoolDiskStars_eqFunction_3632(data, threadData);
  WhirlpoolDiskStars_eqFunction_3635(data, threadData);
  WhirlpoolDiskStars_eqFunction_3637(data, threadData);
  WhirlpoolDiskStars_eqFunction_3640(data, threadData);
  WhirlpoolDiskStars_eqFunction_3639(data, threadData);
  WhirlpoolDiskStars_eqFunction_3638(data, threadData);
  WhirlpoolDiskStars_eqFunction_3636(data, threadData);
  WhirlpoolDiskStars_eqFunction_1727(data, threadData);
  WhirlpoolDiskStars_eqFunction_3631(data, threadData);
  WhirlpoolDiskStars_eqFunction_1729(data, threadData);
  WhirlpoolDiskStars_eqFunction_3644(data, threadData);
  WhirlpoolDiskStars_eqFunction_1731(data, threadData);
  WhirlpoolDiskStars_eqFunction_1732(data, threadData);
  WhirlpoolDiskStars_eqFunction_3643(data, threadData);
  WhirlpoolDiskStars_eqFunction_1734(data, threadData);
  WhirlpoolDiskStars_eqFunction_1735(data, threadData);
  WhirlpoolDiskStars_eqFunction_3642(data, threadData);
  WhirlpoolDiskStars_eqFunction_3645(data, threadData);
  WhirlpoolDiskStars_eqFunction_3647(data, threadData);
  WhirlpoolDiskStars_eqFunction_3650(data, threadData);
  WhirlpoolDiskStars_eqFunction_3649(data, threadData);
  WhirlpoolDiskStars_eqFunction_3648(data, threadData);
  WhirlpoolDiskStars_eqFunction_3646(data, threadData);
  WhirlpoolDiskStars_eqFunction_1743(data, threadData);
  WhirlpoolDiskStars_eqFunction_3641(data, threadData);
  WhirlpoolDiskStars_eqFunction_1745(data, threadData);
  WhirlpoolDiskStars_eqFunction_3654(data, threadData);
  WhirlpoolDiskStars_eqFunction_1747(data, threadData);
  WhirlpoolDiskStars_eqFunction_1748(data, threadData);
  WhirlpoolDiskStars_eqFunction_3653(data, threadData);
  WhirlpoolDiskStars_eqFunction_1750(data, threadData);
  WhirlpoolDiskStars_eqFunction_1751(data, threadData);
  WhirlpoolDiskStars_eqFunction_3652(data, threadData);
  WhirlpoolDiskStars_eqFunction_3655(data, threadData);
  WhirlpoolDiskStars_eqFunction_3657(data, threadData);
  WhirlpoolDiskStars_eqFunction_3660(data, threadData);
  WhirlpoolDiskStars_eqFunction_3659(data, threadData);
  WhirlpoolDiskStars_eqFunction_3658(data, threadData);
  WhirlpoolDiskStars_eqFunction_3656(data, threadData);
  WhirlpoolDiskStars_eqFunction_1759(data, threadData);
  WhirlpoolDiskStars_eqFunction_3651(data, threadData);
  WhirlpoolDiskStars_eqFunction_1761(data, threadData);
  WhirlpoolDiskStars_eqFunction_3664(data, threadData);
  WhirlpoolDiskStars_eqFunction_1763(data, threadData);
  WhirlpoolDiskStars_eqFunction_1764(data, threadData);
  WhirlpoolDiskStars_eqFunction_3663(data, threadData);
  WhirlpoolDiskStars_eqFunction_1766(data, threadData);
  WhirlpoolDiskStars_eqFunction_1767(data, threadData);
  WhirlpoolDiskStars_eqFunction_3662(data, threadData);
  WhirlpoolDiskStars_eqFunction_3665(data, threadData);
  WhirlpoolDiskStars_eqFunction_3667(data, threadData);
  WhirlpoolDiskStars_eqFunction_3670(data, threadData);
  WhirlpoolDiskStars_eqFunction_3669(data, threadData);
  WhirlpoolDiskStars_eqFunction_3668(data, threadData);
  WhirlpoolDiskStars_eqFunction_3666(data, threadData);
  WhirlpoolDiskStars_eqFunction_1775(data, threadData);
  WhirlpoolDiskStars_eqFunction_3661(data, threadData);
  WhirlpoolDiskStars_eqFunction_1777(data, threadData);
  WhirlpoolDiskStars_eqFunction_3674(data, threadData);
  WhirlpoolDiskStars_eqFunction_1779(data, threadData);
  WhirlpoolDiskStars_eqFunction_1780(data, threadData);
  WhirlpoolDiskStars_eqFunction_3673(data, threadData);
  WhirlpoolDiskStars_eqFunction_1782(data, threadData);
  WhirlpoolDiskStars_eqFunction_1783(data, threadData);
  WhirlpoolDiskStars_eqFunction_3672(data, threadData);
  WhirlpoolDiskStars_eqFunction_3675(data, threadData);
  WhirlpoolDiskStars_eqFunction_3677(data, threadData);
  WhirlpoolDiskStars_eqFunction_3680(data, threadData);
  WhirlpoolDiskStars_eqFunction_3679(data, threadData);
  WhirlpoolDiskStars_eqFunction_3678(data, threadData);
  WhirlpoolDiskStars_eqFunction_3676(data, threadData);
  WhirlpoolDiskStars_eqFunction_1791(data, threadData);
  WhirlpoolDiskStars_eqFunction_3671(data, threadData);
  WhirlpoolDiskStars_eqFunction_1793(data, threadData);
  WhirlpoolDiskStars_eqFunction_3684(data, threadData);
  WhirlpoolDiskStars_eqFunction_1795(data, threadData);
  WhirlpoolDiskStars_eqFunction_1796(data, threadData);
  WhirlpoolDiskStars_eqFunction_3683(data, threadData);
  WhirlpoolDiskStars_eqFunction_1798(data, threadData);
  WhirlpoolDiskStars_eqFunction_1799(data, threadData);
  WhirlpoolDiskStars_eqFunction_3682(data, threadData);
  WhirlpoolDiskStars_eqFunction_3685(data, threadData);
  WhirlpoolDiskStars_eqFunction_3687(data, threadData);
  WhirlpoolDiskStars_eqFunction_3690(data, threadData);
  WhirlpoolDiskStars_eqFunction_3689(data, threadData);
  WhirlpoolDiskStars_eqFunction_3688(data, threadData);
  WhirlpoolDiskStars_eqFunction_3686(data, threadData);
  WhirlpoolDiskStars_eqFunction_1807(data, threadData);
  WhirlpoolDiskStars_eqFunction_3681(data, threadData);
  WhirlpoolDiskStars_eqFunction_1809(data, threadData);
  WhirlpoolDiskStars_eqFunction_3694(data, threadData);
  WhirlpoolDiskStars_eqFunction_1811(data, threadData);
  WhirlpoolDiskStars_eqFunction_1812(data, threadData);
  WhirlpoolDiskStars_eqFunction_3693(data, threadData);
  WhirlpoolDiskStars_eqFunction_1814(data, threadData);
  WhirlpoolDiskStars_eqFunction_1815(data, threadData);
  WhirlpoolDiskStars_eqFunction_3692(data, threadData);
  WhirlpoolDiskStars_eqFunction_3695(data, threadData);
  WhirlpoolDiskStars_eqFunction_3697(data, threadData);
  WhirlpoolDiskStars_eqFunction_3700(data, threadData);
  WhirlpoolDiskStars_eqFunction_3699(data, threadData);
  WhirlpoolDiskStars_eqFunction_3698(data, threadData);
  WhirlpoolDiskStars_eqFunction_3696(data, threadData);
  WhirlpoolDiskStars_eqFunction_1823(data, threadData);
  WhirlpoolDiskStars_eqFunction_3691(data, threadData);
  WhirlpoolDiskStars_eqFunction_1825(data, threadData);
  WhirlpoolDiskStars_eqFunction_3704(data, threadData);
  WhirlpoolDiskStars_eqFunction_1827(data, threadData);
  WhirlpoolDiskStars_eqFunction_1828(data, threadData);
  WhirlpoolDiskStars_eqFunction_3703(data, threadData);
  WhirlpoolDiskStars_eqFunction_1830(data, threadData);
  WhirlpoolDiskStars_eqFunction_1831(data, threadData);
  WhirlpoolDiskStars_eqFunction_3702(data, threadData);
  WhirlpoolDiskStars_eqFunction_3705(data, threadData);
  WhirlpoolDiskStars_eqFunction_3707(data, threadData);
  WhirlpoolDiskStars_eqFunction_3710(data, threadData);
  WhirlpoolDiskStars_eqFunction_3709(data, threadData);
  WhirlpoolDiskStars_eqFunction_3708(data, threadData);
  WhirlpoolDiskStars_eqFunction_3706(data, threadData);
  WhirlpoolDiskStars_eqFunction_1839(data, threadData);
  WhirlpoolDiskStars_eqFunction_3701(data, threadData);
  WhirlpoolDiskStars_eqFunction_1841(data, threadData);
  WhirlpoolDiskStars_eqFunction_3714(data, threadData);
  WhirlpoolDiskStars_eqFunction_1843(data, threadData);
  WhirlpoolDiskStars_eqFunction_1844(data, threadData);
  WhirlpoolDiskStars_eqFunction_3713(data, threadData);
  WhirlpoolDiskStars_eqFunction_1846(data, threadData);
  WhirlpoolDiskStars_eqFunction_1847(data, threadData);
  WhirlpoolDiskStars_eqFunction_3712(data, threadData);
  WhirlpoolDiskStars_eqFunction_3715(data, threadData);
  WhirlpoolDiskStars_eqFunction_3717(data, threadData);
  WhirlpoolDiskStars_eqFunction_3720(data, threadData);
  WhirlpoolDiskStars_eqFunction_3719(data, threadData);
  WhirlpoolDiskStars_eqFunction_3718(data, threadData);
  WhirlpoolDiskStars_eqFunction_3716(data, threadData);
  WhirlpoolDiskStars_eqFunction_1855(data, threadData);
  WhirlpoolDiskStars_eqFunction_3711(data, threadData);
  WhirlpoolDiskStars_eqFunction_1857(data, threadData);
  WhirlpoolDiskStars_eqFunction_3724(data, threadData);
  WhirlpoolDiskStars_eqFunction_1859(data, threadData);
  WhirlpoolDiskStars_eqFunction_1860(data, threadData);
  WhirlpoolDiskStars_eqFunction_3723(data, threadData);
  WhirlpoolDiskStars_eqFunction_1862(data, threadData);
  WhirlpoolDiskStars_eqFunction_1863(data, threadData);
  WhirlpoolDiskStars_eqFunction_3722(data, threadData);
  WhirlpoolDiskStars_eqFunction_3725(data, threadData);
  WhirlpoolDiskStars_eqFunction_3727(data, threadData);
  WhirlpoolDiskStars_eqFunction_3730(data, threadData);
  WhirlpoolDiskStars_eqFunction_3729(data, threadData);
  WhirlpoolDiskStars_eqFunction_3728(data, threadData);
  WhirlpoolDiskStars_eqFunction_3726(data, threadData);
  WhirlpoolDiskStars_eqFunction_1871(data, threadData);
  WhirlpoolDiskStars_eqFunction_3721(data, threadData);
  WhirlpoolDiskStars_eqFunction_1873(data, threadData);
  WhirlpoolDiskStars_eqFunction_3734(data, threadData);
  WhirlpoolDiskStars_eqFunction_1875(data, threadData);
  WhirlpoolDiskStars_eqFunction_1876(data, threadData);
  WhirlpoolDiskStars_eqFunction_3733(data, threadData);
  WhirlpoolDiskStars_eqFunction_1878(data, threadData);
  WhirlpoolDiskStars_eqFunction_1879(data, threadData);
  WhirlpoolDiskStars_eqFunction_3732(data, threadData);
  WhirlpoolDiskStars_eqFunction_3735(data, threadData);
  WhirlpoolDiskStars_eqFunction_3737(data, threadData);
  WhirlpoolDiskStars_eqFunction_3740(data, threadData);
  WhirlpoolDiskStars_eqFunction_3739(data, threadData);
  WhirlpoolDiskStars_eqFunction_3738(data, threadData);
  WhirlpoolDiskStars_eqFunction_3736(data, threadData);
  WhirlpoolDiskStars_eqFunction_1887(data, threadData);
  WhirlpoolDiskStars_eqFunction_3731(data, threadData);
  WhirlpoolDiskStars_eqFunction_1889(data, threadData);
  WhirlpoolDiskStars_eqFunction_3744(data, threadData);
  WhirlpoolDiskStars_eqFunction_1891(data, threadData);
  WhirlpoolDiskStars_eqFunction_1892(data, threadData);
  WhirlpoolDiskStars_eqFunction_3743(data, threadData);
  WhirlpoolDiskStars_eqFunction_1894(data, threadData);
  WhirlpoolDiskStars_eqFunction_1895(data, threadData);
  WhirlpoolDiskStars_eqFunction_3742(data, threadData);
  WhirlpoolDiskStars_eqFunction_3745(data, threadData);
  WhirlpoolDiskStars_eqFunction_3747(data, threadData);
  WhirlpoolDiskStars_eqFunction_3750(data, threadData);
  WhirlpoolDiskStars_eqFunction_3749(data, threadData);
  WhirlpoolDiskStars_eqFunction_3748(data, threadData);
  WhirlpoolDiskStars_eqFunction_3746(data, threadData);
  WhirlpoolDiskStars_eqFunction_1903(data, threadData);
  WhirlpoolDiskStars_eqFunction_3741(data, threadData);
  WhirlpoolDiskStars_eqFunction_1905(data, threadData);
  WhirlpoolDiskStars_eqFunction_3754(data, threadData);
  WhirlpoolDiskStars_eqFunction_1907(data, threadData);
  WhirlpoolDiskStars_eqFunction_1908(data, threadData);
  WhirlpoolDiskStars_eqFunction_3753(data, threadData);
  WhirlpoolDiskStars_eqFunction_1910(data, threadData);
  WhirlpoolDiskStars_eqFunction_1911(data, threadData);
  WhirlpoolDiskStars_eqFunction_3752(data, threadData);
  WhirlpoolDiskStars_eqFunction_3755(data, threadData);
  WhirlpoolDiskStars_eqFunction_3757(data, threadData);
  WhirlpoolDiskStars_eqFunction_3760(data, threadData);
  WhirlpoolDiskStars_eqFunction_3759(data, threadData);
  WhirlpoolDiskStars_eqFunction_3758(data, threadData);
  WhirlpoolDiskStars_eqFunction_3756(data, threadData);
  WhirlpoolDiskStars_eqFunction_1919(data, threadData);
  WhirlpoolDiskStars_eqFunction_3751(data, threadData);
  WhirlpoolDiskStars_eqFunction_1921(data, threadData);
  WhirlpoolDiskStars_eqFunction_3764(data, threadData);
  WhirlpoolDiskStars_eqFunction_1923(data, threadData);
  WhirlpoolDiskStars_eqFunction_1924(data, threadData);
  WhirlpoolDiskStars_eqFunction_3763(data, threadData);
  WhirlpoolDiskStars_eqFunction_1926(data, threadData);
  WhirlpoolDiskStars_eqFunction_1927(data, threadData);
  WhirlpoolDiskStars_eqFunction_3762(data, threadData);
  WhirlpoolDiskStars_eqFunction_3765(data, threadData);
  WhirlpoolDiskStars_eqFunction_3767(data, threadData);
  WhirlpoolDiskStars_eqFunction_3770(data, threadData);
  WhirlpoolDiskStars_eqFunction_3769(data, threadData);
  WhirlpoolDiskStars_eqFunction_3768(data, threadData);
  WhirlpoolDiskStars_eqFunction_3766(data, threadData);
  WhirlpoolDiskStars_eqFunction_1935(data, threadData);
  WhirlpoolDiskStars_eqFunction_3761(data, threadData);
  WhirlpoolDiskStars_eqFunction_1937(data, threadData);
  WhirlpoolDiskStars_eqFunction_3774(data, threadData);
  WhirlpoolDiskStars_eqFunction_1939(data, threadData);
  WhirlpoolDiskStars_eqFunction_1940(data, threadData);
  WhirlpoolDiskStars_eqFunction_3773(data, threadData);
  WhirlpoolDiskStars_eqFunction_1942(data, threadData);
  WhirlpoolDiskStars_eqFunction_1943(data, threadData);
  WhirlpoolDiskStars_eqFunction_3772(data, threadData);
  WhirlpoolDiskStars_eqFunction_3775(data, threadData);
  WhirlpoolDiskStars_eqFunction_3777(data, threadData);
  WhirlpoolDiskStars_eqFunction_3780(data, threadData);
  WhirlpoolDiskStars_eqFunction_3779(data, threadData);
  WhirlpoolDiskStars_eqFunction_3778(data, threadData);
  WhirlpoolDiskStars_eqFunction_3776(data, threadData);
  WhirlpoolDiskStars_eqFunction_1951(data, threadData);
  WhirlpoolDiskStars_eqFunction_3771(data, threadData);
  WhirlpoolDiskStars_eqFunction_1953(data, threadData);
  WhirlpoolDiskStars_eqFunction_3784(data, threadData);
  WhirlpoolDiskStars_eqFunction_1955(data, threadData);
  WhirlpoolDiskStars_eqFunction_1956(data, threadData);
  WhirlpoolDiskStars_eqFunction_3783(data, threadData);
  WhirlpoolDiskStars_eqFunction_1958(data, threadData);
  WhirlpoolDiskStars_eqFunction_1959(data, threadData);
  WhirlpoolDiskStars_eqFunction_3782(data, threadData);
  WhirlpoolDiskStars_eqFunction_3785(data, threadData);
  WhirlpoolDiskStars_eqFunction_3787(data, threadData);
  WhirlpoolDiskStars_eqFunction_3790(data, threadData);
  WhirlpoolDiskStars_eqFunction_3789(data, threadData);
  WhirlpoolDiskStars_eqFunction_3788(data, threadData);
  WhirlpoolDiskStars_eqFunction_3786(data, threadData);
  WhirlpoolDiskStars_eqFunction_1967(data, threadData);
  WhirlpoolDiskStars_eqFunction_3781(data, threadData);
  WhirlpoolDiskStars_eqFunction_1969(data, threadData);
  WhirlpoolDiskStars_eqFunction_3794(data, threadData);
  WhirlpoolDiskStars_eqFunction_1971(data, threadData);
  WhirlpoolDiskStars_eqFunction_1972(data, threadData);
  WhirlpoolDiskStars_eqFunction_3793(data, threadData);
  WhirlpoolDiskStars_eqFunction_1974(data, threadData);
  WhirlpoolDiskStars_eqFunction_1975(data, threadData);
  WhirlpoolDiskStars_eqFunction_3792(data, threadData);
  WhirlpoolDiskStars_eqFunction_3795(data, threadData);
  WhirlpoolDiskStars_eqFunction_3797(data, threadData);
  WhirlpoolDiskStars_eqFunction_3800(data, threadData);
  WhirlpoolDiskStars_eqFunction_3799(data, threadData);
  WhirlpoolDiskStars_eqFunction_3798(data, threadData);
  WhirlpoolDiskStars_eqFunction_3796(data, threadData);
  WhirlpoolDiskStars_eqFunction_1983(data, threadData);
  WhirlpoolDiskStars_eqFunction_3791(data, threadData);
  WhirlpoolDiskStars_eqFunction_1985(data, threadData);
  WhirlpoolDiskStars_eqFunction_3804(data, threadData);
  WhirlpoolDiskStars_eqFunction_1987(data, threadData);
  WhirlpoolDiskStars_eqFunction_1988(data, threadData);
  WhirlpoolDiskStars_eqFunction_3803(data, threadData);
  WhirlpoolDiskStars_eqFunction_1990(data, threadData);
  WhirlpoolDiskStars_eqFunction_1991(data, threadData);
  WhirlpoolDiskStars_eqFunction_3802(data, threadData);
  WhirlpoolDiskStars_eqFunction_3805(data, threadData);
  WhirlpoolDiskStars_eqFunction_3807(data, threadData);
  WhirlpoolDiskStars_eqFunction_3810(data, threadData);
  WhirlpoolDiskStars_eqFunction_3809(data, threadData);
  WhirlpoolDiskStars_eqFunction_3808(data, threadData);
  WhirlpoolDiskStars_eqFunction_3806(data, threadData);
  WhirlpoolDiskStars_eqFunction_1999(data, threadData);
  WhirlpoolDiskStars_eqFunction_3801(data, threadData);
  WhirlpoolDiskStars_eqFunction_2001(data, threadData);
  WhirlpoolDiskStars_eqFunction_3814(data, threadData);
  WhirlpoolDiskStars_eqFunction_2003(data, threadData);
  WhirlpoolDiskStars_eqFunction_2004(data, threadData);
  WhirlpoolDiskStars_eqFunction_3813(data, threadData);
  WhirlpoolDiskStars_eqFunction_2006(data, threadData);
  WhirlpoolDiskStars_eqFunction_2007(data, threadData);
  WhirlpoolDiskStars_eqFunction_3812(data, threadData);
  WhirlpoolDiskStars_eqFunction_3815(data, threadData);
  WhirlpoolDiskStars_eqFunction_3817(data, threadData);
  WhirlpoolDiskStars_eqFunction_3820(data, threadData);
  WhirlpoolDiskStars_eqFunction_3819(data, threadData);
  WhirlpoolDiskStars_eqFunction_3818(data, threadData);
  WhirlpoolDiskStars_eqFunction_3816(data, threadData);
  WhirlpoolDiskStars_eqFunction_2015(data, threadData);
  WhirlpoolDiskStars_eqFunction_3811(data, threadData);
  WhirlpoolDiskStars_eqFunction_2017(data, threadData);
  WhirlpoolDiskStars_eqFunction_3824(data, threadData);
  WhirlpoolDiskStars_eqFunction_2019(data, threadData);
  WhirlpoolDiskStars_eqFunction_2020(data, threadData);
  WhirlpoolDiskStars_eqFunction_3823(data, threadData);
  WhirlpoolDiskStars_eqFunction_2022(data, threadData);
  WhirlpoolDiskStars_eqFunction_2023(data, threadData);
  WhirlpoolDiskStars_eqFunction_3822(data, threadData);
  WhirlpoolDiskStars_eqFunction_3825(data, threadData);
  WhirlpoolDiskStars_eqFunction_3827(data, threadData);
  WhirlpoolDiskStars_eqFunction_3830(data, threadData);
  WhirlpoolDiskStars_eqFunction_3829(data, threadData);
  WhirlpoolDiskStars_eqFunction_3828(data, threadData);
  WhirlpoolDiskStars_eqFunction_3826(data, threadData);
  WhirlpoolDiskStars_eqFunction_2031(data, threadData);
  WhirlpoolDiskStars_eqFunction_3821(data, threadData);
  WhirlpoolDiskStars_eqFunction_2033(data, threadData);
  WhirlpoolDiskStars_eqFunction_3834(data, threadData);
  WhirlpoolDiskStars_eqFunction_2035(data, threadData);
  WhirlpoolDiskStars_eqFunction_2036(data, threadData);
  WhirlpoolDiskStars_eqFunction_3833(data, threadData);
  WhirlpoolDiskStars_eqFunction_2038(data, threadData);
  WhirlpoolDiskStars_eqFunction_2039(data, threadData);
  WhirlpoolDiskStars_eqFunction_3832(data, threadData);
  WhirlpoolDiskStars_eqFunction_3835(data, threadData);
  WhirlpoolDiskStars_eqFunction_3837(data, threadData);
  WhirlpoolDiskStars_eqFunction_3840(data, threadData);
  WhirlpoolDiskStars_eqFunction_3839(data, threadData);
  WhirlpoolDiskStars_eqFunction_3838(data, threadData);
  WhirlpoolDiskStars_eqFunction_3836(data, threadData);
  WhirlpoolDiskStars_eqFunction_2047(data, threadData);
  WhirlpoolDiskStars_eqFunction_3831(data, threadData);
  WhirlpoolDiskStars_eqFunction_2049(data, threadData);
  WhirlpoolDiskStars_eqFunction_3844(data, threadData);
  WhirlpoolDiskStars_eqFunction_2051(data, threadData);
  WhirlpoolDiskStars_eqFunction_2052(data, threadData);
  WhirlpoolDiskStars_eqFunction_3843(data, threadData);
  WhirlpoolDiskStars_eqFunction_2054(data, threadData);
  WhirlpoolDiskStars_eqFunction_2055(data, threadData);
  WhirlpoolDiskStars_eqFunction_3842(data, threadData);
  WhirlpoolDiskStars_eqFunction_3845(data, threadData);
  WhirlpoolDiskStars_eqFunction_3847(data, threadData);
  WhirlpoolDiskStars_eqFunction_3850(data, threadData);
  WhirlpoolDiskStars_eqFunction_3849(data, threadData);
  WhirlpoolDiskStars_eqFunction_3848(data, threadData);
  WhirlpoolDiskStars_eqFunction_3846(data, threadData);
  WhirlpoolDiskStars_eqFunction_2063(data, threadData);
  WhirlpoolDiskStars_eqFunction_3841(data, threadData);
  WhirlpoolDiskStars_eqFunction_2065(data, threadData);
  WhirlpoolDiskStars_eqFunction_3854(data, threadData);
  WhirlpoolDiskStars_eqFunction_2067(data, threadData);
  WhirlpoolDiskStars_eqFunction_2068(data, threadData);
  WhirlpoolDiskStars_eqFunction_3853(data, threadData);
  WhirlpoolDiskStars_eqFunction_2070(data, threadData);
  WhirlpoolDiskStars_eqFunction_2071(data, threadData);
  WhirlpoolDiskStars_eqFunction_3852(data, threadData);
  WhirlpoolDiskStars_eqFunction_3855(data, threadData);
  WhirlpoolDiskStars_eqFunction_3857(data, threadData);
  WhirlpoolDiskStars_eqFunction_3860(data, threadData);
  WhirlpoolDiskStars_eqFunction_3859(data, threadData);
  WhirlpoolDiskStars_eqFunction_3858(data, threadData);
  WhirlpoolDiskStars_eqFunction_3856(data, threadData);
  WhirlpoolDiskStars_eqFunction_2079(data, threadData);
  WhirlpoolDiskStars_eqFunction_3851(data, threadData);
  WhirlpoolDiskStars_eqFunction_2081(data, threadData);
  WhirlpoolDiskStars_eqFunction_3864(data, threadData);
  WhirlpoolDiskStars_eqFunction_2083(data, threadData);
  WhirlpoolDiskStars_eqFunction_2084(data, threadData);
  WhirlpoolDiskStars_eqFunction_3863(data, threadData);
  WhirlpoolDiskStars_eqFunction_2086(data, threadData);
  WhirlpoolDiskStars_eqFunction_2087(data, threadData);
  WhirlpoolDiskStars_eqFunction_3862(data, threadData);
  WhirlpoolDiskStars_eqFunction_3865(data, threadData);
  WhirlpoolDiskStars_eqFunction_3867(data, threadData);
  WhirlpoolDiskStars_eqFunction_3870(data, threadData);
  WhirlpoolDiskStars_eqFunction_3869(data, threadData);
  WhirlpoolDiskStars_eqFunction_3868(data, threadData);
  WhirlpoolDiskStars_eqFunction_3866(data, threadData);
  WhirlpoolDiskStars_eqFunction_2095(data, threadData);
  WhirlpoolDiskStars_eqFunction_3861(data, threadData);
  WhirlpoolDiskStars_eqFunction_2097(data, threadData);
  WhirlpoolDiskStars_eqFunction_3874(data, threadData);
  WhirlpoolDiskStars_eqFunction_2099(data, threadData);
  WhirlpoolDiskStars_eqFunction_2100(data, threadData);
  WhirlpoolDiskStars_eqFunction_3873(data, threadData);
  WhirlpoolDiskStars_eqFunction_2102(data, threadData);
  WhirlpoolDiskStars_eqFunction_2103(data, threadData);
  WhirlpoolDiskStars_eqFunction_3872(data, threadData);
  WhirlpoolDiskStars_eqFunction_3875(data, threadData);
  WhirlpoolDiskStars_eqFunction_3877(data, threadData);
  WhirlpoolDiskStars_eqFunction_3880(data, threadData);
  WhirlpoolDiskStars_eqFunction_3879(data, threadData);
  WhirlpoolDiskStars_eqFunction_3878(data, threadData);
  WhirlpoolDiskStars_eqFunction_3876(data, threadData);
  WhirlpoolDiskStars_eqFunction_2111(data, threadData);
  WhirlpoolDiskStars_eqFunction_3871(data, threadData);
  WhirlpoolDiskStars_eqFunction_2113(data, threadData);
  WhirlpoolDiskStars_eqFunction_3884(data, threadData);
  WhirlpoolDiskStars_eqFunction_2115(data, threadData);
  WhirlpoolDiskStars_eqFunction_2116(data, threadData);
  WhirlpoolDiskStars_eqFunction_3883(data, threadData);
  WhirlpoolDiskStars_eqFunction_2118(data, threadData);
  WhirlpoolDiskStars_eqFunction_2119(data, threadData);
  WhirlpoolDiskStars_eqFunction_3882(data, threadData);
  WhirlpoolDiskStars_eqFunction_3885(data, threadData);
  WhirlpoolDiskStars_eqFunction_3887(data, threadData);
  WhirlpoolDiskStars_eqFunction_3890(data, threadData);
  WhirlpoolDiskStars_eqFunction_3889(data, threadData);
  WhirlpoolDiskStars_eqFunction_3888(data, threadData);
  WhirlpoolDiskStars_eqFunction_3886(data, threadData);
  WhirlpoolDiskStars_eqFunction_2127(data, threadData);
  WhirlpoolDiskStars_eqFunction_3881(data, threadData);
  WhirlpoolDiskStars_eqFunction_2129(data, threadData);
  WhirlpoolDiskStars_eqFunction_3894(data, threadData);
  WhirlpoolDiskStars_eqFunction_2131(data, threadData);
  WhirlpoolDiskStars_eqFunction_2132(data, threadData);
  WhirlpoolDiskStars_eqFunction_3893(data, threadData);
  WhirlpoolDiskStars_eqFunction_2134(data, threadData);
  WhirlpoolDiskStars_eqFunction_2135(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif