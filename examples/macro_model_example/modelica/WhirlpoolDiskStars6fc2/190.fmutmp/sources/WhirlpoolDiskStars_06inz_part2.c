#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 855
type: SIMPLE_ASSIGN
vy[54] = x[54] * sqrt(G * Md / r_init[54] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_855(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,855};
  modelica_real tmp214;
  modelica_real tmp215;
  tmp214 = (data->simulationInfo->realParameter[218] /* r_init[54] PARAM */);
  tmp215 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp214 * tmp214 * tmp214),"r_init[54] ^ 3.0",equationIndexes);
  if(!(tmp215 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[54] ^ 3.0) was %g should be >= 0", tmp215);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[213]] /* vy[54] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[533]] /* x[54] STATE(1,vx[54]) */)) * (sqrt(tmp215));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3092(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3095(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3097(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3100(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3099(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3098(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3096(DATA *data, threadData_t *threadData);


/*
equation index: 863
type: SIMPLE_ASSIGN
vz[54] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_863(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,863};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[373]] /* vz[54] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3091(DATA *data, threadData_t *threadData);


/*
equation index: 865
type: SIMPLE_ASSIGN
z[55] = -1.25
*/
void WhirlpoolDiskStars_eqFunction_865(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,865};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[854]] /* z[55] STATE(1,vz[55]) */) = -1.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3104(DATA *data, threadData_t *threadData);


/*
equation index: 867
type: SIMPLE_ASSIGN
y[55] = r_init[55] * sin(theta[55] + armOffset[55])
*/
void WhirlpoolDiskStars_eqFunction_867(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,867};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[694]] /* y[55] STATE(1,vy[55]) */) = ((data->simulationInfo->realParameter[219] /* r_init[55] PARAM */)) * (sin((data->simulationInfo->realParameter[379] /* theta[55] PARAM */) + (data->simulationInfo->realParameter[57] /* armOffset[55] PARAM */)));
  TRACE_POP
}

/*
equation index: 868
type: SIMPLE_ASSIGN
vx[55] = (-y[55]) * sqrt(G * Md / r_init[55] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_868(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,868};
  modelica_real tmp216;
  modelica_real tmp217;
  tmp216 = (data->simulationInfo->realParameter[219] /* r_init[55] PARAM */);
  tmp217 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp216 * tmp216 * tmp216),"r_init[55] ^ 3.0",equationIndexes);
  if(!(tmp217 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[55] ^ 3.0) was %g should be >= 0", tmp217);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[54]] /* vx[55] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[694]] /* y[55] STATE(1,vy[55]) */))) * (sqrt(tmp217));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3103(DATA *data, threadData_t *threadData);


/*
equation index: 870
type: SIMPLE_ASSIGN
x[55] = r_init[55] * cos(theta[55] + armOffset[55])
*/
void WhirlpoolDiskStars_eqFunction_870(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,870};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[534]] /* x[55] STATE(1,vx[55]) */) = ((data->simulationInfo->realParameter[219] /* r_init[55] PARAM */)) * (cos((data->simulationInfo->realParameter[379] /* theta[55] PARAM */) + (data->simulationInfo->realParameter[57] /* armOffset[55] PARAM */)));
  TRACE_POP
}

/*
equation index: 871
type: SIMPLE_ASSIGN
vy[55] = x[55] * sqrt(G * Md / r_init[55] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_871(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,871};
  modelica_real tmp218;
  modelica_real tmp219;
  tmp218 = (data->simulationInfo->realParameter[219] /* r_init[55] PARAM */);
  tmp219 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp218 * tmp218 * tmp218),"r_init[55] ^ 3.0",equationIndexes);
  if(!(tmp219 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[55] ^ 3.0) was %g should be >= 0", tmp219);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[214]] /* vy[55] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[534]] /* x[55] STATE(1,vx[55]) */)) * (sqrt(tmp219));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3102(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3105(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3107(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3110(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3109(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3108(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3106(DATA *data, threadData_t *threadData);


/*
equation index: 879
type: SIMPLE_ASSIGN
vz[55] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_879(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,879};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[374]] /* vz[55] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3101(DATA *data, threadData_t *threadData);


/*
equation index: 881
type: SIMPLE_ASSIGN
z[56] = -1.2000000000000002
*/
void WhirlpoolDiskStars_eqFunction_881(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,881};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[855]] /* z[56] STATE(1,vz[56]) */) = -1.2000000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3114(DATA *data, threadData_t *threadData);


/*
equation index: 883
type: SIMPLE_ASSIGN
y[56] = r_init[56] * sin(theta[56] + armOffset[56])
*/
void WhirlpoolDiskStars_eqFunction_883(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,883};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[695]] /* y[56] STATE(1,vy[56]) */) = ((data->simulationInfo->realParameter[220] /* r_init[56] PARAM */)) * (sin((data->simulationInfo->realParameter[380] /* theta[56] PARAM */) + (data->simulationInfo->realParameter[58] /* armOffset[56] PARAM */)));
  TRACE_POP
}

/*
equation index: 884
type: SIMPLE_ASSIGN
vx[56] = (-y[56]) * sqrt(G * Md / r_init[56] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_884(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,884};
  modelica_real tmp220;
  modelica_real tmp221;
  tmp220 = (data->simulationInfo->realParameter[220] /* r_init[56] PARAM */);
  tmp221 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp220 * tmp220 * tmp220),"r_init[56] ^ 3.0",equationIndexes);
  if(!(tmp221 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[56] ^ 3.0) was %g should be >= 0", tmp221);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[55]] /* vx[56] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[695]] /* y[56] STATE(1,vy[56]) */))) * (sqrt(tmp221));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3113(DATA *data, threadData_t *threadData);


/*
equation index: 886
type: SIMPLE_ASSIGN
x[56] = r_init[56] * cos(theta[56] + armOffset[56])
*/
void WhirlpoolDiskStars_eqFunction_886(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,886};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[535]] /* x[56] STATE(1,vx[56]) */) = ((data->simulationInfo->realParameter[220] /* r_init[56] PARAM */)) * (cos((data->simulationInfo->realParameter[380] /* theta[56] PARAM */) + (data->simulationInfo->realParameter[58] /* armOffset[56] PARAM */)));
  TRACE_POP
}

/*
equation index: 887
type: SIMPLE_ASSIGN
vy[56] = x[56] * sqrt(G * Md / r_init[56] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_887(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,887};
  modelica_real tmp222;
  modelica_real tmp223;
  tmp222 = (data->simulationInfo->realParameter[220] /* r_init[56] PARAM */);
  tmp223 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp222 * tmp222 * tmp222),"r_init[56] ^ 3.0",equationIndexes);
  if(!(tmp223 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[56] ^ 3.0) was %g should be >= 0", tmp223);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[215]] /* vy[56] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[535]] /* x[56] STATE(1,vx[56]) */)) * (sqrt(tmp223));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3112(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3115(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3117(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3120(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3119(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3118(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3116(DATA *data, threadData_t *threadData);


/*
equation index: 895
type: SIMPLE_ASSIGN
vz[56] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_895(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,895};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[375]] /* vz[56] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3111(DATA *data, threadData_t *threadData);


/*
equation index: 897
type: SIMPLE_ASSIGN
z[57] = -1.1500000000000001
*/
void WhirlpoolDiskStars_eqFunction_897(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,897};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[856]] /* z[57] STATE(1,vz[57]) */) = -1.1500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3124(DATA *data, threadData_t *threadData);


/*
equation index: 899
type: SIMPLE_ASSIGN
y[57] = r_init[57] * sin(theta[57] + armOffset[57])
*/
void WhirlpoolDiskStars_eqFunction_899(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,899};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[696]] /* y[57] STATE(1,vy[57]) */) = ((data->simulationInfo->realParameter[221] /* r_init[57] PARAM */)) * (sin((data->simulationInfo->realParameter[381] /* theta[57] PARAM */) + (data->simulationInfo->realParameter[59] /* armOffset[57] PARAM */)));
  TRACE_POP
}

/*
equation index: 900
type: SIMPLE_ASSIGN
vx[57] = (-y[57]) * sqrt(G * Md / r_init[57] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_900(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,900};
  modelica_real tmp224;
  modelica_real tmp225;
  tmp224 = (data->simulationInfo->realParameter[221] /* r_init[57] PARAM */);
  tmp225 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp224 * tmp224 * tmp224),"r_init[57] ^ 3.0",equationIndexes);
  if(!(tmp225 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[57] ^ 3.0) was %g should be >= 0", tmp225);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[56]] /* vx[57] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[696]] /* y[57] STATE(1,vy[57]) */))) * (sqrt(tmp225));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3123(DATA *data, threadData_t *threadData);


/*
equation index: 902
type: SIMPLE_ASSIGN
x[57] = r_init[57] * cos(theta[57] + armOffset[57])
*/
void WhirlpoolDiskStars_eqFunction_902(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,902};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[536]] /* x[57] STATE(1,vx[57]) */) = ((data->simulationInfo->realParameter[221] /* r_init[57] PARAM */)) * (cos((data->simulationInfo->realParameter[381] /* theta[57] PARAM */) + (data->simulationInfo->realParameter[59] /* armOffset[57] PARAM */)));
  TRACE_POP
}

/*
equation index: 903
type: SIMPLE_ASSIGN
vy[57] = x[57] * sqrt(G * Md / r_init[57] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_903(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,903};
  modelica_real tmp226;
  modelica_real tmp227;
  tmp226 = (data->simulationInfo->realParameter[221] /* r_init[57] PARAM */);
  tmp227 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp226 * tmp226 * tmp226),"r_init[57] ^ 3.0",equationIndexes);
  if(!(tmp227 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[57] ^ 3.0) was %g should be >= 0", tmp227);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[216]] /* vy[57] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[536]] /* x[57] STATE(1,vx[57]) */)) * (sqrt(tmp227));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3122(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3125(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3127(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3130(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3129(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3128(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3126(DATA *data, threadData_t *threadData);


/*
equation index: 911
type: SIMPLE_ASSIGN
vz[57] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_911(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,911};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[376]] /* vz[57] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3121(DATA *data, threadData_t *threadData);


/*
equation index: 913
type: SIMPLE_ASSIGN
z[58] = -1.1
*/
void WhirlpoolDiskStars_eqFunction_913(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,913};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[857]] /* z[58] STATE(1,vz[58]) */) = -1.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3134(DATA *data, threadData_t *threadData);


/*
equation index: 915
type: SIMPLE_ASSIGN
y[58] = r_init[58] * sin(theta[58] + armOffset[58])
*/
void WhirlpoolDiskStars_eqFunction_915(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,915};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[697]] /* y[58] STATE(1,vy[58]) */) = ((data->simulationInfo->realParameter[222] /* r_init[58] PARAM */)) * (sin((data->simulationInfo->realParameter[382] /* theta[58] PARAM */) + (data->simulationInfo->realParameter[60] /* armOffset[58] PARAM */)));
  TRACE_POP
}

/*
equation index: 916
type: SIMPLE_ASSIGN
vx[58] = (-y[58]) * sqrt(G * Md / r_init[58] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_916(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,916};
  modelica_real tmp228;
  modelica_real tmp229;
  tmp228 = (data->simulationInfo->realParameter[222] /* r_init[58] PARAM */);
  tmp229 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp228 * tmp228 * tmp228),"r_init[58] ^ 3.0",equationIndexes);
  if(!(tmp229 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[58] ^ 3.0) was %g should be >= 0", tmp229);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[57]] /* vx[58] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[697]] /* y[58] STATE(1,vy[58]) */))) * (sqrt(tmp229));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3133(DATA *data, threadData_t *threadData);


/*
equation index: 918
type: SIMPLE_ASSIGN
x[58] = r_init[58] * cos(theta[58] + armOffset[58])
*/
void WhirlpoolDiskStars_eqFunction_918(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,918};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[537]] /* x[58] STATE(1,vx[58]) */) = ((data->simulationInfo->realParameter[222] /* r_init[58] PARAM */)) * (cos((data->simulationInfo->realParameter[382] /* theta[58] PARAM */) + (data->simulationInfo->realParameter[60] /* armOffset[58] PARAM */)));
  TRACE_POP
}

/*
equation index: 919
type: SIMPLE_ASSIGN
vy[58] = x[58] * sqrt(G * Md / r_init[58] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_919(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,919};
  modelica_real tmp230;
  modelica_real tmp231;
  tmp230 = (data->simulationInfo->realParameter[222] /* r_init[58] PARAM */);
  tmp231 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp230 * tmp230 * tmp230),"r_init[58] ^ 3.0",equationIndexes);
  if(!(tmp231 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[58] ^ 3.0) was %g should be >= 0", tmp231);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[217]] /* vy[58] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[537]] /* x[58] STATE(1,vx[58]) */)) * (sqrt(tmp231));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3132(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3135(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3137(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3140(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3139(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3138(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3136(DATA *data, threadData_t *threadData);


/*
equation index: 927
type: SIMPLE_ASSIGN
vz[58] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_927(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,927};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[377]] /* vz[58] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3131(DATA *data, threadData_t *threadData);


/*
equation index: 929
type: SIMPLE_ASSIGN
z[59] = -1.05
*/
void WhirlpoolDiskStars_eqFunction_929(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,929};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[858]] /* z[59] STATE(1,vz[59]) */) = -1.05;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3144(DATA *data, threadData_t *threadData);


/*
equation index: 931
type: SIMPLE_ASSIGN
y[59] = r_init[59] * sin(theta[59] + armOffset[59])
*/
void WhirlpoolDiskStars_eqFunction_931(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,931};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[698]] /* y[59] STATE(1,vy[59]) */) = ((data->simulationInfo->realParameter[223] /* r_init[59] PARAM */)) * (sin((data->simulationInfo->realParameter[383] /* theta[59] PARAM */) + (data->simulationInfo->realParameter[61] /* armOffset[59] PARAM */)));
  TRACE_POP
}

/*
equation index: 932
type: SIMPLE_ASSIGN
vx[59] = (-y[59]) * sqrt(G * Md / r_init[59] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_932(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,932};
  modelica_real tmp232;
  modelica_real tmp233;
  tmp232 = (data->simulationInfo->realParameter[223] /* r_init[59] PARAM */);
  tmp233 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp232 * tmp232 * tmp232),"r_init[59] ^ 3.0",equationIndexes);
  if(!(tmp233 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[59] ^ 3.0) was %g should be >= 0", tmp233);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[58]] /* vx[59] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[698]] /* y[59] STATE(1,vy[59]) */))) * (sqrt(tmp233));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3143(DATA *data, threadData_t *threadData);


/*
equation index: 934
type: SIMPLE_ASSIGN
x[59] = r_init[59] * cos(theta[59] + armOffset[59])
*/
void WhirlpoolDiskStars_eqFunction_934(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,934};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[538]] /* x[59] STATE(1,vx[59]) */) = ((data->simulationInfo->realParameter[223] /* r_init[59] PARAM */)) * (cos((data->simulationInfo->realParameter[383] /* theta[59] PARAM */) + (data->simulationInfo->realParameter[61] /* armOffset[59] PARAM */)));
  TRACE_POP
}

/*
equation index: 935
type: SIMPLE_ASSIGN
vy[59] = x[59] * sqrt(G * Md / r_init[59] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_935(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,935};
  modelica_real tmp234;
  modelica_real tmp235;
  tmp234 = (data->simulationInfo->realParameter[223] /* r_init[59] PARAM */);
  tmp235 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp234 * tmp234 * tmp234),"r_init[59] ^ 3.0",equationIndexes);
  if(!(tmp235 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[59] ^ 3.0) was %g should be >= 0", tmp235);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[218]] /* vy[59] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[538]] /* x[59] STATE(1,vx[59]) */)) * (sqrt(tmp235));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3142(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3145(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3147(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3150(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3149(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3148(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3146(DATA *data, threadData_t *threadData);


/*
equation index: 943
type: SIMPLE_ASSIGN
vz[59] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_943(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,943};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[378]] /* vz[59] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3141(DATA *data, threadData_t *threadData);


/*
equation index: 945
type: SIMPLE_ASSIGN
z[60] = -1.0
*/
void WhirlpoolDiskStars_eqFunction_945(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,945};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[859]] /* z[60] STATE(1,vz[60]) */) = -1.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3154(DATA *data, threadData_t *threadData);


/*
equation index: 947
type: SIMPLE_ASSIGN
y[60] = r_init[60] * sin(theta[60] + armOffset[60])
*/
void WhirlpoolDiskStars_eqFunction_947(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,947};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[699]] /* y[60] STATE(1,vy[60]) */) = ((data->simulationInfo->realParameter[224] /* r_init[60] PARAM */)) * (sin((data->simulationInfo->realParameter[384] /* theta[60] PARAM */) + (data->simulationInfo->realParameter[62] /* armOffset[60] PARAM */)));
  TRACE_POP
}

/*
equation index: 948
type: SIMPLE_ASSIGN
vx[60] = (-y[60]) * sqrt(G * Md / r_init[60] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_948(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,948};
  modelica_real tmp236;
  modelica_real tmp237;
  tmp236 = (data->simulationInfo->realParameter[224] /* r_init[60] PARAM */);
  tmp237 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp236 * tmp236 * tmp236),"r_init[60] ^ 3.0",equationIndexes);
  if(!(tmp237 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[60] ^ 3.0) was %g should be >= 0", tmp237);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[59]] /* vx[60] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[699]] /* y[60] STATE(1,vy[60]) */))) * (sqrt(tmp237));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3153(DATA *data, threadData_t *threadData);


/*
equation index: 950
type: SIMPLE_ASSIGN
x[60] = r_init[60] * cos(theta[60] + armOffset[60])
*/
void WhirlpoolDiskStars_eqFunction_950(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,950};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[539]] /* x[60] STATE(1,vx[60]) */) = ((data->simulationInfo->realParameter[224] /* r_init[60] PARAM */)) * (cos((data->simulationInfo->realParameter[384] /* theta[60] PARAM */) + (data->simulationInfo->realParameter[62] /* armOffset[60] PARAM */)));
  TRACE_POP
}

/*
equation index: 951
type: SIMPLE_ASSIGN
vy[60] = x[60] * sqrt(G * Md / r_init[60] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_951(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,951};
  modelica_real tmp238;
  modelica_real tmp239;
  tmp238 = (data->simulationInfo->realParameter[224] /* r_init[60] PARAM */);
  tmp239 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp238 * tmp238 * tmp238),"r_init[60] ^ 3.0",equationIndexes);
  if(!(tmp239 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[60] ^ 3.0) was %g should be >= 0", tmp239);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[219]] /* vy[60] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[539]] /* x[60] STATE(1,vx[60]) */)) * (sqrt(tmp239));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3152(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3155(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3157(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3160(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3159(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3158(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3156(DATA *data, threadData_t *threadData);


/*
equation index: 959
type: SIMPLE_ASSIGN
vz[60] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_959(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,959};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[379]] /* vz[60] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3151(DATA *data, threadData_t *threadData);


/*
equation index: 961
type: SIMPLE_ASSIGN
z[61] = -0.9500000000000001
*/
void WhirlpoolDiskStars_eqFunction_961(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,961};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[860]] /* z[61] STATE(1,vz[61]) */) = -0.9500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3164(DATA *data, threadData_t *threadData);


/*
equation index: 963
type: SIMPLE_ASSIGN
y[61] = r_init[61] * sin(theta[61] + armOffset[61])
*/
void WhirlpoolDiskStars_eqFunction_963(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,963};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[700]] /* y[61] STATE(1,vy[61]) */) = ((data->simulationInfo->realParameter[225] /* r_init[61] PARAM */)) * (sin((data->simulationInfo->realParameter[385] /* theta[61] PARAM */) + (data->simulationInfo->realParameter[63] /* armOffset[61] PARAM */)));
  TRACE_POP
}

/*
equation index: 964
type: SIMPLE_ASSIGN
vx[61] = (-y[61]) * sqrt(G * Md / r_init[61] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_964(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,964};
  modelica_real tmp240;
  modelica_real tmp241;
  tmp240 = (data->simulationInfo->realParameter[225] /* r_init[61] PARAM */);
  tmp241 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp240 * tmp240 * tmp240),"r_init[61] ^ 3.0",equationIndexes);
  if(!(tmp241 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[61] ^ 3.0) was %g should be >= 0", tmp241);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[60]] /* vx[61] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[700]] /* y[61] STATE(1,vy[61]) */))) * (sqrt(tmp241));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3163(DATA *data, threadData_t *threadData);


/*
equation index: 966
type: SIMPLE_ASSIGN
x[61] = r_init[61] * cos(theta[61] + armOffset[61])
*/
void WhirlpoolDiskStars_eqFunction_966(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,966};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[540]] /* x[61] STATE(1,vx[61]) */) = ((data->simulationInfo->realParameter[225] /* r_init[61] PARAM */)) * (cos((data->simulationInfo->realParameter[385] /* theta[61] PARAM */) + (data->simulationInfo->realParameter[63] /* armOffset[61] PARAM */)));
  TRACE_POP
}

/*
equation index: 967
type: SIMPLE_ASSIGN
vy[61] = x[61] * sqrt(G * Md / r_init[61] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_967(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,967};
  modelica_real tmp242;
  modelica_real tmp243;
  tmp242 = (data->simulationInfo->realParameter[225] /* r_init[61] PARAM */);
  tmp243 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp242 * tmp242 * tmp242),"r_init[61] ^ 3.0",equationIndexes);
  if(!(tmp243 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[61] ^ 3.0) was %g should be >= 0", tmp243);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[220]] /* vy[61] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[540]] /* x[61] STATE(1,vx[61]) */)) * (sqrt(tmp243));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3162(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3165(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3167(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3170(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3169(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3168(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3166(DATA *data, threadData_t *threadData);


/*
equation index: 975
type: SIMPLE_ASSIGN
vz[61] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_975(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,975};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[380]] /* vz[61] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3161(DATA *data, threadData_t *threadData);


/*
equation index: 977
type: SIMPLE_ASSIGN
z[62] = -0.9
*/
void WhirlpoolDiskStars_eqFunction_977(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,977};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[861]] /* z[62] STATE(1,vz[62]) */) = -0.9;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3174(DATA *data, threadData_t *threadData);


/*
equation index: 979
type: SIMPLE_ASSIGN
y[62] = r_init[62] * sin(theta[62] + armOffset[62])
*/
void WhirlpoolDiskStars_eqFunction_979(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,979};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[701]] /* y[62] STATE(1,vy[62]) */) = ((data->simulationInfo->realParameter[226] /* r_init[62] PARAM */)) * (sin((data->simulationInfo->realParameter[386] /* theta[62] PARAM */) + (data->simulationInfo->realParameter[64] /* armOffset[62] PARAM */)));
  TRACE_POP
}

/*
equation index: 980
type: SIMPLE_ASSIGN
vx[62] = (-y[62]) * sqrt(G * Md / r_init[62] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_980(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,980};
  modelica_real tmp244;
  modelica_real tmp245;
  tmp244 = (data->simulationInfo->realParameter[226] /* r_init[62] PARAM */);
  tmp245 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp244 * tmp244 * tmp244),"r_init[62] ^ 3.0",equationIndexes);
  if(!(tmp245 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[62] ^ 3.0) was %g should be >= 0", tmp245);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[61]] /* vx[62] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[701]] /* y[62] STATE(1,vy[62]) */))) * (sqrt(tmp245));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3173(DATA *data, threadData_t *threadData);


/*
equation index: 982
type: SIMPLE_ASSIGN
x[62] = r_init[62] * cos(theta[62] + armOffset[62])
*/
void WhirlpoolDiskStars_eqFunction_982(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,982};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[541]] /* x[62] STATE(1,vx[62]) */) = ((data->simulationInfo->realParameter[226] /* r_init[62] PARAM */)) * (cos((data->simulationInfo->realParameter[386] /* theta[62] PARAM */) + (data->simulationInfo->realParameter[64] /* armOffset[62] PARAM */)));
  TRACE_POP
}

/*
equation index: 983
type: SIMPLE_ASSIGN
vy[62] = x[62] * sqrt(G * Md / r_init[62] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_983(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,983};
  modelica_real tmp246;
  modelica_real tmp247;
  tmp246 = (data->simulationInfo->realParameter[226] /* r_init[62] PARAM */);
  tmp247 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp246 * tmp246 * tmp246),"r_init[62] ^ 3.0",equationIndexes);
  if(!(tmp247 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[62] ^ 3.0) was %g should be >= 0", tmp247);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[221]] /* vy[62] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[541]] /* x[62] STATE(1,vx[62]) */)) * (sqrt(tmp247));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3172(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3175(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3177(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3180(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3179(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3178(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3176(DATA *data, threadData_t *threadData);


/*
equation index: 991
type: SIMPLE_ASSIGN
vz[62] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_991(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,991};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[381]] /* vz[62] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3171(DATA *data, threadData_t *threadData);


/*
equation index: 993
type: SIMPLE_ASSIGN
z[63] = -0.8500000000000001
*/
void WhirlpoolDiskStars_eqFunction_993(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,993};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[862]] /* z[63] STATE(1,vz[63]) */) = -0.8500000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3184(DATA *data, threadData_t *threadData);


/*
equation index: 995
type: SIMPLE_ASSIGN
y[63] = r_init[63] * sin(theta[63] + armOffset[63])
*/
void WhirlpoolDiskStars_eqFunction_995(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,995};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[702]] /* y[63] STATE(1,vy[63]) */) = ((data->simulationInfo->realParameter[227] /* r_init[63] PARAM */)) * (sin((data->simulationInfo->realParameter[387] /* theta[63] PARAM */) + (data->simulationInfo->realParameter[65] /* armOffset[63] PARAM */)));
  TRACE_POP
}

/*
equation index: 996
type: SIMPLE_ASSIGN
vx[63] = (-y[63]) * sqrt(G * Md / r_init[63] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_996(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,996};
  modelica_real tmp248;
  modelica_real tmp249;
  tmp248 = (data->simulationInfo->realParameter[227] /* r_init[63] PARAM */);
  tmp249 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp248 * tmp248 * tmp248),"r_init[63] ^ 3.0",equationIndexes);
  if(!(tmp249 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[63] ^ 3.0) was %g should be >= 0", tmp249);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[62]] /* vx[63] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[702]] /* y[63] STATE(1,vy[63]) */))) * (sqrt(tmp249));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3183(DATA *data, threadData_t *threadData);


/*
equation index: 998
type: SIMPLE_ASSIGN
x[63] = r_init[63] * cos(theta[63] + armOffset[63])
*/
void WhirlpoolDiskStars_eqFunction_998(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,998};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[542]] /* x[63] STATE(1,vx[63]) */) = ((data->simulationInfo->realParameter[227] /* r_init[63] PARAM */)) * (cos((data->simulationInfo->realParameter[387] /* theta[63] PARAM */) + (data->simulationInfo->realParameter[65] /* armOffset[63] PARAM */)));
  TRACE_POP
}

/*
equation index: 999
type: SIMPLE_ASSIGN
vy[63] = x[63] * sqrt(G * Md / r_init[63] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_999(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,999};
  modelica_real tmp250;
  modelica_real tmp251;
  tmp250 = (data->simulationInfo->realParameter[227] /* r_init[63] PARAM */);
  tmp251 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp250 * tmp250 * tmp250),"r_init[63] ^ 3.0",equationIndexes);
  if(!(tmp251 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[63] ^ 3.0) was %g should be >= 0", tmp251);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[222]] /* vy[63] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[542]] /* x[63] STATE(1,vx[63]) */)) * (sqrt(tmp251));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3182(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3185(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3187(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3190(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3189(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3188(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3186(DATA *data, threadData_t *threadData);


/*
equation index: 1007
type: SIMPLE_ASSIGN
vz[63] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1007(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1007};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[382]] /* vz[63] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3181(DATA *data, threadData_t *threadData);


/*
equation index: 1009
type: SIMPLE_ASSIGN
z[64] = -0.8
*/
void WhirlpoolDiskStars_eqFunction_1009(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1009};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[863]] /* z[64] STATE(1,vz[64]) */) = -0.8;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3194(DATA *data, threadData_t *threadData);


/*
equation index: 1011
type: SIMPLE_ASSIGN
y[64] = r_init[64] * sin(theta[64] + armOffset[64])
*/
void WhirlpoolDiskStars_eqFunction_1011(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1011};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[703]] /* y[64] STATE(1,vy[64]) */) = ((data->simulationInfo->realParameter[228] /* r_init[64] PARAM */)) * (sin((data->simulationInfo->realParameter[388] /* theta[64] PARAM */) + (data->simulationInfo->realParameter[66] /* armOffset[64] PARAM */)));
  TRACE_POP
}

/*
equation index: 1012
type: SIMPLE_ASSIGN
vx[64] = (-y[64]) * sqrt(G * Md / r_init[64] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1012(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1012};
  modelica_real tmp252;
  modelica_real tmp253;
  tmp252 = (data->simulationInfo->realParameter[228] /* r_init[64] PARAM */);
  tmp253 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp252 * tmp252 * tmp252),"r_init[64] ^ 3.0",equationIndexes);
  if(!(tmp253 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[64] ^ 3.0) was %g should be >= 0", tmp253);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[63]] /* vx[64] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[703]] /* y[64] STATE(1,vy[64]) */))) * (sqrt(tmp253));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3193(DATA *data, threadData_t *threadData);


/*
equation index: 1014
type: SIMPLE_ASSIGN
x[64] = r_init[64] * cos(theta[64] + armOffset[64])
*/
void WhirlpoolDiskStars_eqFunction_1014(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1014};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[543]] /* x[64] STATE(1,vx[64]) */) = ((data->simulationInfo->realParameter[228] /* r_init[64] PARAM */)) * (cos((data->simulationInfo->realParameter[388] /* theta[64] PARAM */) + (data->simulationInfo->realParameter[66] /* armOffset[64] PARAM */)));
  TRACE_POP
}

/*
equation index: 1015
type: SIMPLE_ASSIGN
vy[64] = x[64] * sqrt(G * Md / r_init[64] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1015(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1015};
  modelica_real tmp254;
  modelica_real tmp255;
  tmp254 = (data->simulationInfo->realParameter[228] /* r_init[64] PARAM */);
  tmp255 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp254 * tmp254 * tmp254),"r_init[64] ^ 3.0",equationIndexes);
  if(!(tmp255 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[64] ^ 3.0) was %g should be >= 0", tmp255);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[223]] /* vy[64] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[543]] /* x[64] STATE(1,vx[64]) */)) * (sqrt(tmp255));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3192(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3195(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3197(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3200(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3199(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3198(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3196(DATA *data, threadData_t *threadData);


/*
equation index: 1023
type: SIMPLE_ASSIGN
vz[64] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1023(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1023};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[383]] /* vz[64] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3191(DATA *data, threadData_t *threadData);


/*
equation index: 1025
type: SIMPLE_ASSIGN
z[65] = -0.75
*/
void WhirlpoolDiskStars_eqFunction_1025(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1025};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[864]] /* z[65] STATE(1,vz[65]) */) = -0.75;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3204(DATA *data, threadData_t *threadData);


/*
equation index: 1027
type: SIMPLE_ASSIGN
y[65] = r_init[65] * sin(theta[65] + armOffset[65])
*/
void WhirlpoolDiskStars_eqFunction_1027(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1027};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[704]] /* y[65] STATE(1,vy[65]) */) = ((data->simulationInfo->realParameter[229] /* r_init[65] PARAM */)) * (sin((data->simulationInfo->realParameter[389] /* theta[65] PARAM */) + (data->simulationInfo->realParameter[67] /* armOffset[65] PARAM */)));
  TRACE_POP
}

/*
equation index: 1028
type: SIMPLE_ASSIGN
vx[65] = (-y[65]) * sqrt(G * Md / r_init[65] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1028(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1028};
  modelica_real tmp256;
  modelica_real tmp257;
  tmp256 = (data->simulationInfo->realParameter[229] /* r_init[65] PARAM */);
  tmp257 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp256 * tmp256 * tmp256),"r_init[65] ^ 3.0",equationIndexes);
  if(!(tmp257 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[65] ^ 3.0) was %g should be >= 0", tmp257);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[64]] /* vx[65] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[704]] /* y[65] STATE(1,vy[65]) */))) * (sqrt(tmp257));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3203(DATA *data, threadData_t *threadData);


/*
equation index: 1030
type: SIMPLE_ASSIGN
x[65] = r_init[65] * cos(theta[65] + armOffset[65])
*/
void WhirlpoolDiskStars_eqFunction_1030(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1030};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[544]] /* x[65] STATE(1,vx[65]) */) = ((data->simulationInfo->realParameter[229] /* r_init[65] PARAM */)) * (cos((data->simulationInfo->realParameter[389] /* theta[65] PARAM */) + (data->simulationInfo->realParameter[67] /* armOffset[65] PARAM */)));
  TRACE_POP
}

/*
equation index: 1031
type: SIMPLE_ASSIGN
vy[65] = x[65] * sqrt(G * Md / r_init[65] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1031(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1031};
  modelica_real tmp258;
  modelica_real tmp259;
  tmp258 = (data->simulationInfo->realParameter[229] /* r_init[65] PARAM */);
  tmp259 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp258 * tmp258 * tmp258),"r_init[65] ^ 3.0",equationIndexes);
  if(!(tmp259 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[65] ^ 3.0) was %g should be >= 0", tmp259);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[224]] /* vy[65] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[544]] /* x[65] STATE(1,vx[65]) */)) * (sqrt(tmp259));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3202(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3205(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3207(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3210(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3209(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3208(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3206(DATA *data, threadData_t *threadData);


/*
equation index: 1039
type: SIMPLE_ASSIGN
vz[65] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1039(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1039};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[384]] /* vz[65] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3201(DATA *data, threadData_t *threadData);


/*
equation index: 1041
type: SIMPLE_ASSIGN
z[66] = -0.7000000000000001
*/
void WhirlpoolDiskStars_eqFunction_1041(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1041};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[865]] /* z[66] STATE(1,vz[66]) */) = -0.7000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3214(DATA *data, threadData_t *threadData);


/*
equation index: 1043
type: SIMPLE_ASSIGN
y[66] = r_init[66] * sin(theta[66] + armOffset[66])
*/
void WhirlpoolDiskStars_eqFunction_1043(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1043};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[705]] /* y[66] STATE(1,vy[66]) */) = ((data->simulationInfo->realParameter[230] /* r_init[66] PARAM */)) * (sin((data->simulationInfo->realParameter[390] /* theta[66] PARAM */) + (data->simulationInfo->realParameter[68] /* armOffset[66] PARAM */)));
  TRACE_POP
}

/*
equation index: 1044
type: SIMPLE_ASSIGN
vx[66] = (-y[66]) * sqrt(G * Md / r_init[66] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1044(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1044};
  modelica_real tmp260;
  modelica_real tmp261;
  tmp260 = (data->simulationInfo->realParameter[230] /* r_init[66] PARAM */);
  tmp261 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp260 * tmp260 * tmp260),"r_init[66] ^ 3.0",equationIndexes);
  if(!(tmp261 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[66] ^ 3.0) was %g should be >= 0", tmp261);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[65]] /* vx[66] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[705]] /* y[66] STATE(1,vy[66]) */))) * (sqrt(tmp261));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3213(DATA *data, threadData_t *threadData);


/*
equation index: 1046
type: SIMPLE_ASSIGN
x[66] = r_init[66] * cos(theta[66] + armOffset[66])
*/
void WhirlpoolDiskStars_eqFunction_1046(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1046};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[545]] /* x[66] STATE(1,vx[66]) */) = ((data->simulationInfo->realParameter[230] /* r_init[66] PARAM */)) * (cos((data->simulationInfo->realParameter[390] /* theta[66] PARAM */) + (data->simulationInfo->realParameter[68] /* armOffset[66] PARAM */)));
  TRACE_POP
}

/*
equation index: 1047
type: SIMPLE_ASSIGN
vy[66] = x[66] * sqrt(G * Md / r_init[66] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1047(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1047};
  modelica_real tmp262;
  modelica_real tmp263;
  tmp262 = (data->simulationInfo->realParameter[230] /* r_init[66] PARAM */);
  tmp263 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp262 * tmp262 * tmp262),"r_init[66] ^ 3.0",equationIndexes);
  if(!(tmp263 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[66] ^ 3.0) was %g should be >= 0", tmp263);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[225]] /* vy[66] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[545]] /* x[66] STATE(1,vx[66]) */)) * (sqrt(tmp263));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3212(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3215(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3217(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3220(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3219(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3218(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3216(DATA *data, threadData_t *threadData);


/*
equation index: 1055
type: SIMPLE_ASSIGN
vz[66] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1055(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1055};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[385]] /* vz[66] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3211(DATA *data, threadData_t *threadData);


/*
equation index: 1057
type: SIMPLE_ASSIGN
z[67] = -0.65
*/
void WhirlpoolDiskStars_eqFunction_1057(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1057};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[866]] /* z[67] STATE(1,vz[67]) */) = -0.65;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3224(DATA *data, threadData_t *threadData);


/*
equation index: 1059
type: SIMPLE_ASSIGN
y[67] = r_init[67] * sin(theta[67] + armOffset[67])
*/
void WhirlpoolDiskStars_eqFunction_1059(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1059};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[706]] /* y[67] STATE(1,vy[67]) */) = ((data->simulationInfo->realParameter[231] /* r_init[67] PARAM */)) * (sin((data->simulationInfo->realParameter[391] /* theta[67] PARAM */) + (data->simulationInfo->realParameter[69] /* armOffset[67] PARAM */)));
  TRACE_POP
}

/*
equation index: 1060
type: SIMPLE_ASSIGN
vx[67] = (-y[67]) * sqrt(G * Md / r_init[67] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1060(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1060};
  modelica_real tmp264;
  modelica_real tmp265;
  tmp264 = (data->simulationInfo->realParameter[231] /* r_init[67] PARAM */);
  tmp265 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp264 * tmp264 * tmp264),"r_init[67] ^ 3.0",equationIndexes);
  if(!(tmp265 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[67] ^ 3.0) was %g should be >= 0", tmp265);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[66]] /* vx[67] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[706]] /* y[67] STATE(1,vy[67]) */))) * (sqrt(tmp265));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3223(DATA *data, threadData_t *threadData);


/*
equation index: 1062
type: SIMPLE_ASSIGN
x[67] = r_init[67] * cos(theta[67] + armOffset[67])
*/
void WhirlpoolDiskStars_eqFunction_1062(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1062};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[546]] /* x[67] STATE(1,vx[67]) */) = ((data->simulationInfo->realParameter[231] /* r_init[67] PARAM */)) * (cos((data->simulationInfo->realParameter[391] /* theta[67] PARAM */) + (data->simulationInfo->realParameter[69] /* armOffset[67] PARAM */)));
  TRACE_POP
}

/*
equation index: 1063
type: SIMPLE_ASSIGN
vy[67] = x[67] * sqrt(G * Md / r_init[67] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1063(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1063};
  modelica_real tmp266;
  modelica_real tmp267;
  tmp266 = (data->simulationInfo->realParameter[231] /* r_init[67] PARAM */);
  tmp267 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp266 * tmp266 * tmp266),"r_init[67] ^ 3.0",equationIndexes);
  if(!(tmp267 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[67] ^ 3.0) was %g should be >= 0", tmp267);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[226]] /* vy[67] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[546]] /* x[67] STATE(1,vx[67]) */)) * (sqrt(tmp267));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3222(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3225(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3227(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3230(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3229(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3228(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3226(DATA *data, threadData_t *threadData);


/*
equation index: 1071
type: SIMPLE_ASSIGN
vz[67] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1071(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1071};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[386]] /* vz[67] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3221(DATA *data, threadData_t *threadData);


/*
equation index: 1073
type: SIMPLE_ASSIGN
z[68] = -0.6000000000000001
*/
void WhirlpoolDiskStars_eqFunction_1073(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1073};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[867]] /* z[68] STATE(1,vz[68]) */) = -0.6000000000000001;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3234(DATA *data, threadData_t *threadData);


/*
equation index: 1075
type: SIMPLE_ASSIGN
y[68] = r_init[68] * sin(theta[68] + armOffset[68])
*/
void WhirlpoolDiskStars_eqFunction_1075(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1075};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[707]] /* y[68] STATE(1,vy[68]) */) = ((data->simulationInfo->realParameter[232] /* r_init[68] PARAM */)) * (sin((data->simulationInfo->realParameter[392] /* theta[68] PARAM */) + (data->simulationInfo->realParameter[70] /* armOffset[68] PARAM */)));
  TRACE_POP
}

/*
equation index: 1076
type: SIMPLE_ASSIGN
vx[68] = (-y[68]) * sqrt(G * Md / r_init[68] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1076(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1076};
  modelica_real tmp268;
  modelica_real tmp269;
  tmp268 = (data->simulationInfo->realParameter[232] /* r_init[68] PARAM */);
  tmp269 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp268 * tmp268 * tmp268),"r_init[68] ^ 3.0",equationIndexes);
  if(!(tmp269 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[68] ^ 3.0) was %g should be >= 0", tmp269);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[67]] /* vx[68] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[707]] /* y[68] STATE(1,vy[68]) */))) * (sqrt(tmp269));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3233(DATA *data, threadData_t *threadData);


/*
equation index: 1078
type: SIMPLE_ASSIGN
x[68] = r_init[68] * cos(theta[68] + armOffset[68])
*/
void WhirlpoolDiskStars_eqFunction_1078(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1078};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[547]] /* x[68] STATE(1,vx[68]) */) = ((data->simulationInfo->realParameter[232] /* r_init[68] PARAM */)) * (cos((data->simulationInfo->realParameter[392] /* theta[68] PARAM */) + (data->simulationInfo->realParameter[70] /* armOffset[68] PARAM */)));
  TRACE_POP
}

/*
equation index: 1079
type: SIMPLE_ASSIGN
vy[68] = x[68] * sqrt(G * Md / r_init[68] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1079(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1079};
  modelica_real tmp270;
  modelica_real tmp271;
  tmp270 = (data->simulationInfo->realParameter[232] /* r_init[68] PARAM */);
  tmp271 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp270 * tmp270 * tmp270),"r_init[68] ^ 3.0",equationIndexes);
  if(!(tmp271 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[68] ^ 3.0) was %g should be >= 0", tmp271);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[227]] /* vy[68] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[547]] /* x[68] STATE(1,vx[68]) */)) * (sqrt(tmp271));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3232(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3235(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3237(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3240(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3239(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3238(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3236(DATA *data, threadData_t *threadData);


/*
equation index: 1087
type: SIMPLE_ASSIGN
vz[68] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1087(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1087};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[387]] /* vz[68] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3231(DATA *data, threadData_t *threadData);


/*
equation index: 1089
type: SIMPLE_ASSIGN
z[69] = -0.55
*/
void WhirlpoolDiskStars_eqFunction_1089(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1089};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[868]] /* z[69] STATE(1,vz[69]) */) = -0.55;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3244(DATA *data, threadData_t *threadData);


/*
equation index: 1091
type: SIMPLE_ASSIGN
y[69] = r_init[69] * sin(theta[69] + armOffset[69])
*/
void WhirlpoolDiskStars_eqFunction_1091(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1091};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[708]] /* y[69] STATE(1,vy[69]) */) = ((data->simulationInfo->realParameter[233] /* r_init[69] PARAM */)) * (sin((data->simulationInfo->realParameter[393] /* theta[69] PARAM */) + (data->simulationInfo->realParameter[71] /* armOffset[69] PARAM */)));
  TRACE_POP
}

/*
equation index: 1092
type: SIMPLE_ASSIGN
vx[69] = (-y[69]) * sqrt(G * Md / r_init[69] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1092(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1092};
  modelica_real tmp272;
  modelica_real tmp273;
  tmp272 = (data->simulationInfo->realParameter[233] /* r_init[69] PARAM */);
  tmp273 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp272 * tmp272 * tmp272),"r_init[69] ^ 3.0",equationIndexes);
  if(!(tmp273 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[69] ^ 3.0) was %g should be >= 0", tmp273);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[68]] /* vx[69] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[708]] /* y[69] STATE(1,vy[69]) */))) * (sqrt(tmp273));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3243(DATA *data, threadData_t *threadData);


/*
equation index: 1094
type: SIMPLE_ASSIGN
x[69] = r_init[69] * cos(theta[69] + armOffset[69])
*/
void WhirlpoolDiskStars_eqFunction_1094(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1094};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[548]] /* x[69] STATE(1,vx[69]) */) = ((data->simulationInfo->realParameter[233] /* r_init[69] PARAM */)) * (cos((data->simulationInfo->realParameter[393] /* theta[69] PARAM */) + (data->simulationInfo->realParameter[71] /* armOffset[69] PARAM */)));
  TRACE_POP
}

/*
equation index: 1095
type: SIMPLE_ASSIGN
vy[69] = x[69] * sqrt(G * Md / r_init[69] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1095(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1095};
  modelica_real tmp274;
  modelica_real tmp275;
  tmp274 = (data->simulationInfo->realParameter[233] /* r_init[69] PARAM */);
  tmp275 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp274 * tmp274 * tmp274),"r_init[69] ^ 3.0",equationIndexes);
  if(!(tmp275 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[69] ^ 3.0) was %g should be >= 0", tmp275);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[228]] /* vy[69] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[548]] /* x[69] STATE(1,vx[69]) */)) * (sqrt(tmp275));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3242(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3245(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3247(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3250(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3249(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3248(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3246(DATA *data, threadData_t *threadData);


/*
equation index: 1103
type: SIMPLE_ASSIGN
vz[69] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1103(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1103};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[388]] /* vz[69] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3241(DATA *data, threadData_t *threadData);


/*
equation index: 1105
type: SIMPLE_ASSIGN
z[70] = -0.5
*/
void WhirlpoolDiskStars_eqFunction_1105(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1105};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[869]] /* z[70] STATE(1,vz[70]) */) = -0.5;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3254(DATA *data, threadData_t *threadData);


/*
equation index: 1107
type: SIMPLE_ASSIGN
y[70] = r_init[70] * sin(theta[70] + armOffset[70])
*/
void WhirlpoolDiskStars_eqFunction_1107(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1107};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[709]] /* y[70] STATE(1,vy[70]) */) = ((data->simulationInfo->realParameter[234] /* r_init[70] PARAM */)) * (sin((data->simulationInfo->realParameter[394] /* theta[70] PARAM */) + (data->simulationInfo->realParameter[72] /* armOffset[70] PARAM */)));
  TRACE_POP
}

/*
equation index: 1108
type: SIMPLE_ASSIGN
vx[70] = (-y[70]) * sqrt(G * Md / r_init[70] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1108(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1108};
  modelica_real tmp276;
  modelica_real tmp277;
  tmp276 = (data->simulationInfo->realParameter[234] /* r_init[70] PARAM */);
  tmp277 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp276 * tmp276 * tmp276),"r_init[70] ^ 3.0",equationIndexes);
  if(!(tmp277 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[70] ^ 3.0) was %g should be >= 0", tmp277);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[69]] /* vx[70] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[709]] /* y[70] STATE(1,vy[70]) */))) * (sqrt(tmp277));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3253(DATA *data, threadData_t *threadData);


/*
equation index: 1110
type: SIMPLE_ASSIGN
x[70] = r_init[70] * cos(theta[70] + armOffset[70])
*/
void WhirlpoolDiskStars_eqFunction_1110(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1110};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[549]] /* x[70] STATE(1,vx[70]) */) = ((data->simulationInfo->realParameter[234] /* r_init[70] PARAM */)) * (cos((data->simulationInfo->realParameter[394] /* theta[70] PARAM */) + (data->simulationInfo->realParameter[72] /* armOffset[70] PARAM */)));
  TRACE_POP
}

/*
equation index: 1111
type: SIMPLE_ASSIGN
vy[70] = x[70] * sqrt(G * Md / r_init[70] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1111(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1111};
  modelica_real tmp278;
  modelica_real tmp279;
  tmp278 = (data->simulationInfo->realParameter[234] /* r_init[70] PARAM */);
  tmp279 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp278 * tmp278 * tmp278),"r_init[70] ^ 3.0",equationIndexes);
  if(!(tmp279 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[70] ^ 3.0) was %g should be >= 0", tmp279);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[229]] /* vy[70] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[549]] /* x[70] STATE(1,vx[70]) */)) * (sqrt(tmp279));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3252(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3255(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3257(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3260(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3259(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3258(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3256(DATA *data, threadData_t *threadData);


/*
equation index: 1119
type: SIMPLE_ASSIGN
vz[70] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1119(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1119};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[389]] /* vz[70] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3251(DATA *data, threadData_t *threadData);


/*
equation index: 1121
type: SIMPLE_ASSIGN
z[71] = -0.45
*/
void WhirlpoolDiskStars_eqFunction_1121(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1121};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[870]] /* z[71] STATE(1,vz[71]) */) = -0.45;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3264(DATA *data, threadData_t *threadData);


/*
equation index: 1123
type: SIMPLE_ASSIGN
y[71] = r_init[71] * sin(theta[71] + armOffset[71])
*/
void WhirlpoolDiskStars_eqFunction_1123(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1123};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[710]] /* y[71] STATE(1,vy[71]) */) = ((data->simulationInfo->realParameter[235] /* r_init[71] PARAM */)) * (sin((data->simulationInfo->realParameter[395] /* theta[71] PARAM */) + (data->simulationInfo->realParameter[73] /* armOffset[71] PARAM */)));
  TRACE_POP
}

/*
equation index: 1124
type: SIMPLE_ASSIGN
vx[71] = (-y[71]) * sqrt(G * Md / r_init[71] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1124(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1124};
  modelica_real tmp280;
  modelica_real tmp281;
  tmp280 = (data->simulationInfo->realParameter[235] /* r_init[71] PARAM */);
  tmp281 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp280 * tmp280 * tmp280),"r_init[71] ^ 3.0",equationIndexes);
  if(!(tmp281 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[71] ^ 3.0) was %g should be >= 0", tmp281);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[70]] /* vx[71] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[710]] /* y[71] STATE(1,vy[71]) */))) * (sqrt(tmp281));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3263(DATA *data, threadData_t *threadData);


/*
equation index: 1126
type: SIMPLE_ASSIGN
x[71] = r_init[71] * cos(theta[71] + armOffset[71])
*/
void WhirlpoolDiskStars_eqFunction_1126(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1126};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[550]] /* x[71] STATE(1,vx[71]) */) = ((data->simulationInfo->realParameter[235] /* r_init[71] PARAM */)) * (cos((data->simulationInfo->realParameter[395] /* theta[71] PARAM */) + (data->simulationInfo->realParameter[73] /* armOffset[71] PARAM */)));
  TRACE_POP
}

/*
equation index: 1127
type: SIMPLE_ASSIGN
vy[71] = x[71] * sqrt(G * Md / r_init[71] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1127(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1127};
  modelica_real tmp282;
  modelica_real tmp283;
  tmp282 = (data->simulationInfo->realParameter[235] /* r_init[71] PARAM */);
  tmp283 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp282 * tmp282 * tmp282),"r_init[71] ^ 3.0",equationIndexes);
  if(!(tmp283 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[71] ^ 3.0) was %g should be >= 0", tmp283);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[230]] /* vy[71] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[550]] /* x[71] STATE(1,vx[71]) */)) * (sqrt(tmp283));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3262(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3265(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3267(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3270(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3269(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3268(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3266(DATA *data, threadData_t *threadData);


/*
equation index: 1135
type: SIMPLE_ASSIGN
vz[71] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1135(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1135};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[390]] /* vz[71] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3261(DATA *data, threadData_t *threadData);


/*
equation index: 1137
type: SIMPLE_ASSIGN
z[72] = -0.4
*/
void WhirlpoolDiskStars_eqFunction_1137(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1137};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[871]] /* z[72] STATE(1,vz[72]) */) = -0.4;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3274(DATA *data, threadData_t *threadData);


/*
equation index: 1139
type: SIMPLE_ASSIGN
y[72] = r_init[72] * sin(theta[72] + armOffset[72])
*/
void WhirlpoolDiskStars_eqFunction_1139(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1139};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[711]] /* y[72] STATE(1,vy[72]) */) = ((data->simulationInfo->realParameter[236] /* r_init[72] PARAM */)) * (sin((data->simulationInfo->realParameter[396] /* theta[72] PARAM */) + (data->simulationInfo->realParameter[74] /* armOffset[72] PARAM */)));
  TRACE_POP
}

/*
equation index: 1140
type: SIMPLE_ASSIGN
vx[72] = (-y[72]) * sqrt(G * Md / r_init[72] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1140(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1140};
  modelica_real tmp284;
  modelica_real tmp285;
  tmp284 = (data->simulationInfo->realParameter[236] /* r_init[72] PARAM */);
  tmp285 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp284 * tmp284 * tmp284),"r_init[72] ^ 3.0",equationIndexes);
  if(!(tmp285 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[72] ^ 3.0) was %g should be >= 0", tmp285);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[71]] /* vx[72] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[711]] /* y[72] STATE(1,vy[72]) */))) * (sqrt(tmp285));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3273(DATA *data, threadData_t *threadData);


/*
equation index: 1142
type: SIMPLE_ASSIGN
x[72] = r_init[72] * cos(theta[72] + armOffset[72])
*/
void WhirlpoolDiskStars_eqFunction_1142(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1142};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[551]] /* x[72] STATE(1,vx[72]) */) = ((data->simulationInfo->realParameter[236] /* r_init[72] PARAM */)) * (cos((data->simulationInfo->realParameter[396] /* theta[72] PARAM */) + (data->simulationInfo->realParameter[74] /* armOffset[72] PARAM */)));
  TRACE_POP
}

/*
equation index: 1143
type: SIMPLE_ASSIGN
vy[72] = x[72] * sqrt(G * Md / r_init[72] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1143(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1143};
  modelica_real tmp286;
  modelica_real tmp287;
  tmp286 = (data->simulationInfo->realParameter[236] /* r_init[72] PARAM */);
  tmp287 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp286 * tmp286 * tmp286),"r_init[72] ^ 3.0",equationIndexes);
  if(!(tmp287 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[72] ^ 3.0) was %g should be >= 0", tmp287);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[231]] /* vy[72] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[551]] /* x[72] STATE(1,vx[72]) */)) * (sqrt(tmp287));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3272(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3275(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3277(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3280(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3279(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3278(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3276(DATA *data, threadData_t *threadData);


/*
equation index: 1151
type: SIMPLE_ASSIGN
vz[72] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1151(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1151};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[391]] /* vz[72] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3271(DATA *data, threadData_t *threadData);


/*
equation index: 1153
type: SIMPLE_ASSIGN
z[73] = -0.35000000000000003
*/
void WhirlpoolDiskStars_eqFunction_1153(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1153};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[872]] /* z[73] STATE(1,vz[73]) */) = -0.35000000000000003;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3284(DATA *data, threadData_t *threadData);


/*
equation index: 1155
type: SIMPLE_ASSIGN
y[73] = r_init[73] * sin(theta[73] + armOffset[73])
*/
void WhirlpoolDiskStars_eqFunction_1155(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1155};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[712]] /* y[73] STATE(1,vy[73]) */) = ((data->simulationInfo->realParameter[237] /* r_init[73] PARAM */)) * (sin((data->simulationInfo->realParameter[397] /* theta[73] PARAM */) + (data->simulationInfo->realParameter[75] /* armOffset[73] PARAM */)));
  TRACE_POP
}

/*
equation index: 1156
type: SIMPLE_ASSIGN
vx[73] = (-y[73]) * sqrt(G * Md / r_init[73] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1156(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1156};
  modelica_real tmp288;
  modelica_real tmp289;
  tmp288 = (data->simulationInfo->realParameter[237] /* r_init[73] PARAM */);
  tmp289 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp288 * tmp288 * tmp288),"r_init[73] ^ 3.0",equationIndexes);
  if(!(tmp289 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[73] ^ 3.0) was %g should be >= 0", tmp289);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[72]] /* vx[73] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[712]] /* y[73] STATE(1,vy[73]) */))) * (sqrt(tmp289));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3283(DATA *data, threadData_t *threadData);


/*
equation index: 1158
type: SIMPLE_ASSIGN
x[73] = r_init[73] * cos(theta[73] + armOffset[73])
*/
void WhirlpoolDiskStars_eqFunction_1158(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1158};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[552]] /* x[73] STATE(1,vx[73]) */) = ((data->simulationInfo->realParameter[237] /* r_init[73] PARAM */)) * (cos((data->simulationInfo->realParameter[397] /* theta[73] PARAM */) + (data->simulationInfo->realParameter[75] /* armOffset[73] PARAM */)));
  TRACE_POP
}

/*
equation index: 1159
type: SIMPLE_ASSIGN
vy[73] = x[73] * sqrt(G * Md / r_init[73] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1159(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1159};
  modelica_real tmp290;
  modelica_real tmp291;
  tmp290 = (data->simulationInfo->realParameter[237] /* r_init[73] PARAM */);
  tmp291 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp290 * tmp290 * tmp290),"r_init[73] ^ 3.0",equationIndexes);
  if(!(tmp291 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[73] ^ 3.0) was %g should be >= 0", tmp291);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[232]] /* vy[73] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[552]] /* x[73] STATE(1,vx[73]) */)) * (sqrt(tmp291));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3282(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3285(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3287(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3290(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3289(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3288(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3286(DATA *data, threadData_t *threadData);


/*
equation index: 1167
type: SIMPLE_ASSIGN
vz[73] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1167};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[392]] /* vz[73] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3281(DATA *data, threadData_t *threadData);


/*
equation index: 1169
type: SIMPLE_ASSIGN
z[74] = -0.30000000000000004
*/
void WhirlpoolDiskStars_eqFunction_1169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1169};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[873]] /* z[74] STATE(1,vz[74]) */) = -0.30000000000000004;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3294(DATA *data, threadData_t *threadData);


/*
equation index: 1171
type: SIMPLE_ASSIGN
y[74] = r_init[74] * sin(theta[74] + armOffset[74])
*/
void WhirlpoolDiskStars_eqFunction_1171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1171};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[713]] /* y[74] STATE(1,vy[74]) */) = ((data->simulationInfo->realParameter[238] /* r_init[74] PARAM */)) * (sin((data->simulationInfo->realParameter[398] /* theta[74] PARAM */) + (data->simulationInfo->realParameter[76] /* armOffset[74] PARAM */)));
  TRACE_POP
}

/*
equation index: 1172
type: SIMPLE_ASSIGN
vx[74] = (-y[74]) * sqrt(G * Md / r_init[74] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1172};
  modelica_real tmp292;
  modelica_real tmp293;
  tmp292 = (data->simulationInfo->realParameter[238] /* r_init[74] PARAM */);
  tmp293 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp292 * tmp292 * tmp292),"r_init[74] ^ 3.0",equationIndexes);
  if(!(tmp293 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[74] ^ 3.0) was %g should be >= 0", tmp293);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[73]] /* vx[74] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[713]] /* y[74] STATE(1,vy[74]) */))) * (sqrt(tmp293));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3293(DATA *data, threadData_t *threadData);


/*
equation index: 1174
type: SIMPLE_ASSIGN
x[74] = r_init[74] * cos(theta[74] + armOffset[74])
*/
void WhirlpoolDiskStars_eqFunction_1174(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1174};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[553]] /* x[74] STATE(1,vx[74]) */) = ((data->simulationInfo->realParameter[238] /* r_init[74] PARAM */)) * (cos((data->simulationInfo->realParameter[398] /* theta[74] PARAM */) + (data->simulationInfo->realParameter[76] /* armOffset[74] PARAM */)));
  TRACE_POP
}

/*
equation index: 1175
type: SIMPLE_ASSIGN
vy[74] = x[74] * sqrt(G * Md / r_init[74] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1175};
  modelica_real tmp294;
  modelica_real tmp295;
  tmp294 = (data->simulationInfo->realParameter[238] /* r_init[74] PARAM */);
  tmp295 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp294 * tmp294 * tmp294),"r_init[74] ^ 3.0",equationIndexes);
  if(!(tmp295 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[74] ^ 3.0) was %g should be >= 0", tmp295);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[233]] /* vy[74] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[553]] /* x[74] STATE(1,vx[74]) */)) * (sqrt(tmp295));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3292(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3295(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3297(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3300(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3299(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3298(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3296(DATA *data, threadData_t *threadData);


/*
equation index: 1183
type: SIMPLE_ASSIGN
vz[74] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1183};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[393]] /* vz[74] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3291(DATA *data, threadData_t *threadData);


/*
equation index: 1185
type: SIMPLE_ASSIGN
z[75] = -0.25
*/
void WhirlpoolDiskStars_eqFunction_1185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1185};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[874]] /* z[75] STATE(1,vz[75]) */) = -0.25;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3304(DATA *data, threadData_t *threadData);


/*
equation index: 1187
type: SIMPLE_ASSIGN
y[75] = r_init[75] * sin(theta[75] + armOffset[75])
*/
void WhirlpoolDiskStars_eqFunction_1187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1187};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[714]] /* y[75] STATE(1,vy[75]) */) = ((data->simulationInfo->realParameter[239] /* r_init[75] PARAM */)) * (sin((data->simulationInfo->realParameter[399] /* theta[75] PARAM */) + (data->simulationInfo->realParameter[77] /* armOffset[75] PARAM */)));
  TRACE_POP
}

/*
equation index: 1188
type: SIMPLE_ASSIGN
vx[75] = (-y[75]) * sqrt(G * Md / r_init[75] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1188};
  modelica_real tmp296;
  modelica_real tmp297;
  tmp296 = (data->simulationInfo->realParameter[239] /* r_init[75] PARAM */);
  tmp297 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp296 * tmp296 * tmp296),"r_init[75] ^ 3.0",equationIndexes);
  if(!(tmp297 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[75] ^ 3.0) was %g should be >= 0", tmp297);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[74]] /* vx[75] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[714]] /* y[75] STATE(1,vy[75]) */))) * (sqrt(tmp297));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3303(DATA *data, threadData_t *threadData);


/*
equation index: 1190
type: SIMPLE_ASSIGN
x[75] = r_init[75] * cos(theta[75] + armOffset[75])
*/
void WhirlpoolDiskStars_eqFunction_1190(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1190};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[554]] /* x[75] STATE(1,vx[75]) */) = ((data->simulationInfo->realParameter[239] /* r_init[75] PARAM */)) * (cos((data->simulationInfo->realParameter[399] /* theta[75] PARAM */) + (data->simulationInfo->realParameter[77] /* armOffset[75] PARAM */)));
  TRACE_POP
}

/*
equation index: 1191
type: SIMPLE_ASSIGN
vy[75] = x[75] * sqrt(G * Md / r_init[75] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1191};
  modelica_real tmp298;
  modelica_real tmp299;
  tmp298 = (data->simulationInfo->realParameter[239] /* r_init[75] PARAM */);
  tmp299 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp298 * tmp298 * tmp298),"r_init[75] ^ 3.0",equationIndexes);
  if(!(tmp299 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[75] ^ 3.0) was %g should be >= 0", tmp299);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[234]] /* vy[75] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[554]] /* x[75] STATE(1,vx[75]) */)) * (sqrt(tmp299));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3302(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3305(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3307(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3310(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3309(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3308(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3306(DATA *data, threadData_t *threadData);


/*
equation index: 1199
type: SIMPLE_ASSIGN
vz[75] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1199};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[394]] /* vz[75] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3301(DATA *data, threadData_t *threadData);


/*
equation index: 1201
type: SIMPLE_ASSIGN
z[76] = -0.2
*/
void WhirlpoolDiskStars_eqFunction_1201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1201};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[875]] /* z[76] STATE(1,vz[76]) */) = -0.2;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3314(DATA *data, threadData_t *threadData);


/*
equation index: 1203
type: SIMPLE_ASSIGN
y[76] = r_init[76] * sin(theta[76] + armOffset[76])
*/
void WhirlpoolDiskStars_eqFunction_1203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1203};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[715]] /* y[76] STATE(1,vy[76]) */) = ((data->simulationInfo->realParameter[240] /* r_init[76] PARAM */)) * (sin((data->simulationInfo->realParameter[400] /* theta[76] PARAM */) + (data->simulationInfo->realParameter[78] /* armOffset[76] PARAM */)));
  TRACE_POP
}

/*
equation index: 1204
type: SIMPLE_ASSIGN
vx[76] = (-y[76]) * sqrt(G * Md / r_init[76] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1204};
  modelica_real tmp300;
  modelica_real tmp301;
  tmp300 = (data->simulationInfo->realParameter[240] /* r_init[76] PARAM */);
  tmp301 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp300 * tmp300 * tmp300),"r_init[76] ^ 3.0",equationIndexes);
  if(!(tmp301 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[76] ^ 3.0) was %g should be >= 0", tmp301);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[75]] /* vx[76] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[715]] /* y[76] STATE(1,vy[76]) */))) * (sqrt(tmp301));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3313(DATA *data, threadData_t *threadData);


/*
equation index: 1206
type: SIMPLE_ASSIGN
x[76] = r_init[76] * cos(theta[76] + armOffset[76])
*/
void WhirlpoolDiskStars_eqFunction_1206(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1206};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[555]] /* x[76] STATE(1,vx[76]) */) = ((data->simulationInfo->realParameter[240] /* r_init[76] PARAM */)) * (cos((data->simulationInfo->realParameter[400] /* theta[76] PARAM */) + (data->simulationInfo->realParameter[78] /* armOffset[76] PARAM */)));
  TRACE_POP
}

/*
equation index: 1207
type: SIMPLE_ASSIGN
vy[76] = x[76] * sqrt(G * Md / r_init[76] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1207};
  modelica_real tmp302;
  modelica_real tmp303;
  tmp302 = (data->simulationInfo->realParameter[240] /* r_init[76] PARAM */);
  tmp303 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp302 * tmp302 * tmp302),"r_init[76] ^ 3.0",equationIndexes);
  if(!(tmp303 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[76] ^ 3.0) was %g should be >= 0", tmp303);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[235]] /* vy[76] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[555]] /* x[76] STATE(1,vx[76]) */)) * (sqrt(tmp303));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3312(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3315(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3317(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3320(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3319(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3318(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3316(DATA *data, threadData_t *threadData);


/*
equation index: 1215
type: SIMPLE_ASSIGN
vz[76] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1215};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[395]] /* vz[76] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3311(DATA *data, threadData_t *threadData);


/*
equation index: 1217
type: SIMPLE_ASSIGN
z[77] = -0.15000000000000002
*/
void WhirlpoolDiskStars_eqFunction_1217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1217};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[876]] /* z[77] STATE(1,vz[77]) */) = -0.15000000000000002;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3324(DATA *data, threadData_t *threadData);


/*
equation index: 1219
type: SIMPLE_ASSIGN
y[77] = r_init[77] * sin(theta[77] + armOffset[77])
*/
void WhirlpoolDiskStars_eqFunction_1219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1219};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[716]] /* y[77] STATE(1,vy[77]) */) = ((data->simulationInfo->realParameter[241] /* r_init[77] PARAM */)) * (sin((data->simulationInfo->realParameter[401] /* theta[77] PARAM */) + (data->simulationInfo->realParameter[79] /* armOffset[77] PARAM */)));
  TRACE_POP
}

/*
equation index: 1220
type: SIMPLE_ASSIGN
vx[77] = (-y[77]) * sqrt(G * Md / r_init[77] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1220};
  modelica_real tmp304;
  modelica_real tmp305;
  tmp304 = (data->simulationInfo->realParameter[241] /* r_init[77] PARAM */);
  tmp305 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp304 * tmp304 * tmp304),"r_init[77] ^ 3.0",equationIndexes);
  if(!(tmp305 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[77] ^ 3.0) was %g should be >= 0", tmp305);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[76]] /* vx[77] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[716]] /* y[77] STATE(1,vy[77]) */))) * (sqrt(tmp305));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3323(DATA *data, threadData_t *threadData);


/*
equation index: 1222
type: SIMPLE_ASSIGN
x[77] = r_init[77] * cos(theta[77] + armOffset[77])
*/
void WhirlpoolDiskStars_eqFunction_1222(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1222};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[556]] /* x[77] STATE(1,vx[77]) */) = ((data->simulationInfo->realParameter[241] /* r_init[77] PARAM */)) * (cos((data->simulationInfo->realParameter[401] /* theta[77] PARAM */) + (data->simulationInfo->realParameter[79] /* armOffset[77] PARAM */)));
  TRACE_POP
}

/*
equation index: 1223
type: SIMPLE_ASSIGN
vy[77] = x[77] * sqrt(G * Md / r_init[77] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1223};
  modelica_real tmp306;
  modelica_real tmp307;
  tmp306 = (data->simulationInfo->realParameter[241] /* r_init[77] PARAM */);
  tmp307 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp306 * tmp306 * tmp306),"r_init[77] ^ 3.0",equationIndexes);
  if(!(tmp307 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[77] ^ 3.0) was %g should be >= 0", tmp307);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[236]] /* vy[77] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[556]] /* x[77] STATE(1,vx[77]) */)) * (sqrt(tmp307));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3322(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3325(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3327(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3330(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3329(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3328(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3326(DATA *data, threadData_t *threadData);


/*
equation index: 1231
type: SIMPLE_ASSIGN
vz[77] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1231};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[396]] /* vz[77] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3321(DATA *data, threadData_t *threadData);


/*
equation index: 1233
type: SIMPLE_ASSIGN
z[78] = -0.1
*/
void WhirlpoolDiskStars_eqFunction_1233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1233};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[877]] /* z[78] STATE(1,vz[78]) */) = -0.1;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3334(DATA *data, threadData_t *threadData);


/*
equation index: 1235
type: SIMPLE_ASSIGN
y[78] = r_init[78] * sin(theta[78] + armOffset[78])
*/
void WhirlpoolDiskStars_eqFunction_1235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1235};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[717]] /* y[78] STATE(1,vy[78]) */) = ((data->simulationInfo->realParameter[242] /* r_init[78] PARAM */)) * (sin((data->simulationInfo->realParameter[402] /* theta[78] PARAM */) + (data->simulationInfo->realParameter[80] /* armOffset[78] PARAM */)));
  TRACE_POP
}

/*
equation index: 1236
type: SIMPLE_ASSIGN
vx[78] = (-y[78]) * sqrt(G * Md / r_init[78] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1236};
  modelica_real tmp308;
  modelica_real tmp309;
  tmp308 = (data->simulationInfo->realParameter[242] /* r_init[78] PARAM */);
  tmp309 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp308 * tmp308 * tmp308),"r_init[78] ^ 3.0",equationIndexes);
  if(!(tmp309 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[78] ^ 3.0) was %g should be >= 0", tmp309);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[77]] /* vx[78] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[717]] /* y[78] STATE(1,vy[78]) */))) * (sqrt(tmp309));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3333(DATA *data, threadData_t *threadData);


/*
equation index: 1238
type: SIMPLE_ASSIGN
x[78] = r_init[78] * cos(theta[78] + armOffset[78])
*/
void WhirlpoolDiskStars_eqFunction_1238(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1238};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[557]] /* x[78] STATE(1,vx[78]) */) = ((data->simulationInfo->realParameter[242] /* r_init[78] PARAM */)) * (cos((data->simulationInfo->realParameter[402] /* theta[78] PARAM */) + (data->simulationInfo->realParameter[80] /* armOffset[78] PARAM */)));
  TRACE_POP
}

/*
equation index: 1239
type: SIMPLE_ASSIGN
vy[78] = x[78] * sqrt(G * Md / r_init[78] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1239};
  modelica_real tmp310;
  modelica_real tmp311;
  tmp310 = (data->simulationInfo->realParameter[242] /* r_init[78] PARAM */);
  tmp311 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp310 * tmp310 * tmp310),"r_init[78] ^ 3.0",equationIndexes);
  if(!(tmp311 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[78] ^ 3.0) was %g should be >= 0", tmp311);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[237]] /* vy[78] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[557]] /* x[78] STATE(1,vx[78]) */)) * (sqrt(tmp311));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3332(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3335(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3337(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3340(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3339(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3338(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3336(DATA *data, threadData_t *threadData);


/*
equation index: 1247
type: SIMPLE_ASSIGN
vz[78] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1247};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[397]] /* vz[78] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3331(DATA *data, threadData_t *threadData);


/*
equation index: 1249
type: SIMPLE_ASSIGN
z[79] = -0.05
*/
void WhirlpoolDiskStars_eqFunction_1249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1249};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[878]] /* z[79] STATE(1,vz[79]) */) = -0.05;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3344(DATA *data, threadData_t *threadData);


/*
equation index: 1251
type: SIMPLE_ASSIGN
y[79] = r_init[79] * sin(theta[79] + armOffset[79])
*/
void WhirlpoolDiskStars_eqFunction_1251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1251};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[718]] /* y[79] STATE(1,vy[79]) */) = ((data->simulationInfo->realParameter[243] /* r_init[79] PARAM */)) * (sin((data->simulationInfo->realParameter[403] /* theta[79] PARAM */) + (data->simulationInfo->realParameter[81] /* armOffset[79] PARAM */)));
  TRACE_POP
}

/*
equation index: 1252
type: SIMPLE_ASSIGN
vx[79] = (-y[79]) * sqrt(G * Md / r_init[79] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1252};
  modelica_real tmp312;
  modelica_real tmp313;
  tmp312 = (data->simulationInfo->realParameter[243] /* r_init[79] PARAM */);
  tmp313 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp312 * tmp312 * tmp312),"r_init[79] ^ 3.0",equationIndexes);
  if(!(tmp313 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[79] ^ 3.0) was %g should be >= 0", tmp313);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[78]] /* vx[79] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[718]] /* y[79] STATE(1,vy[79]) */))) * (sqrt(tmp313));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3343(DATA *data, threadData_t *threadData);


/*
equation index: 1254
type: SIMPLE_ASSIGN
x[79] = r_init[79] * cos(theta[79] + armOffset[79])
*/
void WhirlpoolDiskStars_eqFunction_1254(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1254};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[558]] /* x[79] STATE(1,vx[79]) */) = ((data->simulationInfo->realParameter[243] /* r_init[79] PARAM */)) * (cos((data->simulationInfo->realParameter[403] /* theta[79] PARAM */) + (data->simulationInfo->realParameter[81] /* armOffset[79] PARAM */)));
  TRACE_POP
}

/*
equation index: 1255
type: SIMPLE_ASSIGN
vy[79] = x[79] * sqrt(G * Md / r_init[79] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1255};
  modelica_real tmp314;
  modelica_real tmp315;
  tmp314 = (data->simulationInfo->realParameter[243] /* r_init[79] PARAM */);
  tmp315 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp314 * tmp314 * tmp314),"r_init[79] ^ 3.0",equationIndexes);
  if(!(tmp315 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[79] ^ 3.0) was %g should be >= 0", tmp315);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[238]] /* vy[79] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[558]] /* x[79] STATE(1,vx[79]) */)) * (sqrt(tmp315));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3342(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3345(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3347(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3350(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3349(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3348(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3346(DATA *data, threadData_t *threadData);


/*
equation index: 1263
type: SIMPLE_ASSIGN
vz[79] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1263};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[398]] /* vz[79] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3341(DATA *data, threadData_t *threadData);


/*
equation index: 1265
type: SIMPLE_ASSIGN
z[80] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1265};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[879]] /* z[80] STATE(1,vz[80]) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3354(DATA *data, threadData_t *threadData);


/*
equation index: 1267
type: SIMPLE_ASSIGN
y[80] = r_init[80] * sin(theta[80] + armOffset[80])
*/
void WhirlpoolDiskStars_eqFunction_1267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1267};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[719]] /* y[80] STATE(1,vy[80]) */) = ((data->simulationInfo->realParameter[244] /* r_init[80] PARAM */)) * (sin((data->simulationInfo->realParameter[404] /* theta[80] PARAM */) + (data->simulationInfo->realParameter[82] /* armOffset[80] PARAM */)));
  TRACE_POP
}

/*
equation index: 1268
type: SIMPLE_ASSIGN
vx[80] = (-y[80]) * sqrt(G * Md / r_init[80] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1268};
  modelica_real tmp316;
  modelica_real tmp317;
  tmp316 = (data->simulationInfo->realParameter[244] /* r_init[80] PARAM */);
  tmp317 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp316 * tmp316 * tmp316),"r_init[80] ^ 3.0",equationIndexes);
  if(!(tmp317 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[80] ^ 3.0) was %g should be >= 0", tmp317);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[79]] /* vx[80] STATE(1) */) = ((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[719]] /* y[80] STATE(1,vy[80]) */))) * (sqrt(tmp317));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3353(DATA *data, threadData_t *threadData);


/*
equation index: 1270
type: SIMPLE_ASSIGN
x[80] = r_init[80] * cos(theta[80] + armOffset[80])
*/
void WhirlpoolDiskStars_eqFunction_1270(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1270};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[559]] /* x[80] STATE(1,vx[80]) */) = ((data->simulationInfo->realParameter[244] /* r_init[80] PARAM */)) * (cos((data->simulationInfo->realParameter[404] /* theta[80] PARAM */) + (data->simulationInfo->realParameter[82] /* armOffset[80] PARAM */)));
  TRACE_POP
}

/*
equation index: 1271
type: SIMPLE_ASSIGN
vy[80] = x[80] * sqrt(G * Md / r_init[80] ^ 3.0)
*/
void WhirlpoolDiskStars_eqFunction_1271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1271};
  modelica_real tmp318;
  modelica_real tmp319;
  tmp318 = (data->simulationInfo->realParameter[244] /* r_init[80] PARAM */);
  tmp319 = DIVISION_SIM(((data->simulationInfo->realParameter[0] /* G PARAM */)) * ((data->simulationInfo->realParameter[1] /* Md PARAM */)),(tmp318 * tmp318 * tmp318),"r_init[80] ^ 3.0",equationIndexes);
  if(!(tmp319 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(G * Md / r_init[80] ^ 3.0) was %g should be >= 0", tmp319);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[239]] /* vy[80] STATE(1) */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[559]] /* x[80] STATE(1,vx[80]) */)) * (sqrt(tmp319));
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3352(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3355(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3357(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3360(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3359(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3358(DATA *data, threadData_t *threadData);

extern void WhirlpoolDiskStars_eqFunction_3356(DATA *data, threadData_t *threadData);


/*
equation index: 1279
type: SIMPLE_ASSIGN
vz[80] = 0.0
*/
void WhirlpoolDiskStars_eqFunction_1279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1279};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[399]] /* vz[80] STATE(1) */) = 0.0;
  TRACE_POP
}
extern void WhirlpoolDiskStars_eqFunction_3351(DATA *data, threadData_t *threadData);


/*
equation index: 1281
type: SIMPLE_ASSIGN
z[81] = 0.05
*/
void WhirlpoolDiskStars_eqFunction_1281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,1281};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[880]] /* z[81] STATE(1,vz[81]) */) = 0.05;
  TRACE_POP
}
OMC_DISABLE_OPT
void WhirlpoolDiskStars_functionInitialEquations_2(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_855(data, threadData);
  WhirlpoolDiskStars_eqFunction_3092(data, threadData);
  WhirlpoolDiskStars_eqFunction_3095(data, threadData);
  WhirlpoolDiskStars_eqFunction_3097(data, threadData);
  WhirlpoolDiskStars_eqFunction_3100(data, threadData);
  WhirlpoolDiskStars_eqFunction_3099(data, threadData);
  WhirlpoolDiskStars_eqFunction_3098(data, threadData);
  WhirlpoolDiskStars_eqFunction_3096(data, threadData);
  WhirlpoolDiskStars_eqFunction_863(data, threadData);
  WhirlpoolDiskStars_eqFunction_3091(data, threadData);
  WhirlpoolDiskStars_eqFunction_865(data, threadData);
  WhirlpoolDiskStars_eqFunction_3104(data, threadData);
  WhirlpoolDiskStars_eqFunction_867(data, threadData);
  WhirlpoolDiskStars_eqFunction_868(data, threadData);
  WhirlpoolDiskStars_eqFunction_3103(data, threadData);
  WhirlpoolDiskStars_eqFunction_870(data, threadData);
  WhirlpoolDiskStars_eqFunction_871(data, threadData);
  WhirlpoolDiskStars_eqFunction_3102(data, threadData);
  WhirlpoolDiskStars_eqFunction_3105(data, threadData);
  WhirlpoolDiskStars_eqFunction_3107(data, threadData);
  WhirlpoolDiskStars_eqFunction_3110(data, threadData);
  WhirlpoolDiskStars_eqFunction_3109(data, threadData);
  WhirlpoolDiskStars_eqFunction_3108(data, threadData);
  WhirlpoolDiskStars_eqFunction_3106(data, threadData);
  WhirlpoolDiskStars_eqFunction_879(data, threadData);
  WhirlpoolDiskStars_eqFunction_3101(data, threadData);
  WhirlpoolDiskStars_eqFunction_881(data, threadData);
  WhirlpoolDiskStars_eqFunction_3114(data, threadData);
  WhirlpoolDiskStars_eqFunction_883(data, threadData);
  WhirlpoolDiskStars_eqFunction_884(data, threadData);
  WhirlpoolDiskStars_eqFunction_3113(data, threadData);
  WhirlpoolDiskStars_eqFunction_886(data, threadData);
  WhirlpoolDiskStars_eqFunction_887(data, threadData);
  WhirlpoolDiskStars_eqFunction_3112(data, threadData);
  WhirlpoolDiskStars_eqFunction_3115(data, threadData);
  WhirlpoolDiskStars_eqFunction_3117(data, threadData);
  WhirlpoolDiskStars_eqFunction_3120(data, threadData);
  WhirlpoolDiskStars_eqFunction_3119(data, threadData);
  WhirlpoolDiskStars_eqFunction_3118(data, threadData);
  WhirlpoolDiskStars_eqFunction_3116(data, threadData);
  WhirlpoolDiskStars_eqFunction_895(data, threadData);
  WhirlpoolDiskStars_eqFunction_3111(data, threadData);
  WhirlpoolDiskStars_eqFunction_897(data, threadData);
  WhirlpoolDiskStars_eqFunction_3124(data, threadData);
  WhirlpoolDiskStars_eqFunction_899(data, threadData);
  WhirlpoolDiskStars_eqFunction_900(data, threadData);
  WhirlpoolDiskStars_eqFunction_3123(data, threadData);
  WhirlpoolDiskStars_eqFunction_902(data, threadData);
  WhirlpoolDiskStars_eqFunction_903(data, threadData);
  WhirlpoolDiskStars_eqFunction_3122(data, threadData);
  WhirlpoolDiskStars_eqFunction_3125(data, threadData);
  WhirlpoolDiskStars_eqFunction_3127(data, threadData);
  WhirlpoolDiskStars_eqFunction_3130(data, threadData);
  WhirlpoolDiskStars_eqFunction_3129(data, threadData);
  WhirlpoolDiskStars_eqFunction_3128(data, threadData);
  WhirlpoolDiskStars_eqFunction_3126(data, threadData);
  WhirlpoolDiskStars_eqFunction_911(data, threadData);
  WhirlpoolDiskStars_eqFunction_3121(data, threadData);
  WhirlpoolDiskStars_eqFunction_913(data, threadData);
  WhirlpoolDiskStars_eqFunction_3134(data, threadData);
  WhirlpoolDiskStars_eqFunction_915(data, threadData);
  WhirlpoolDiskStars_eqFunction_916(data, threadData);
  WhirlpoolDiskStars_eqFunction_3133(data, threadData);
  WhirlpoolDiskStars_eqFunction_918(data, threadData);
  WhirlpoolDiskStars_eqFunction_919(data, threadData);
  WhirlpoolDiskStars_eqFunction_3132(data, threadData);
  WhirlpoolDiskStars_eqFunction_3135(data, threadData);
  WhirlpoolDiskStars_eqFunction_3137(data, threadData);
  WhirlpoolDiskStars_eqFunction_3140(data, threadData);
  WhirlpoolDiskStars_eqFunction_3139(data, threadData);
  WhirlpoolDiskStars_eqFunction_3138(data, threadData);
  WhirlpoolDiskStars_eqFunction_3136(data, threadData);
  WhirlpoolDiskStars_eqFunction_927(data, threadData);
  WhirlpoolDiskStars_eqFunction_3131(data, threadData);
  WhirlpoolDiskStars_eqFunction_929(data, threadData);
  WhirlpoolDiskStars_eqFunction_3144(data, threadData);
  WhirlpoolDiskStars_eqFunction_931(data, threadData);
  WhirlpoolDiskStars_eqFunction_932(data, threadData);
  WhirlpoolDiskStars_eqFunction_3143(data, threadData);
  WhirlpoolDiskStars_eqFunction_934(data, threadData);
  WhirlpoolDiskStars_eqFunction_935(data, threadData);
  WhirlpoolDiskStars_eqFunction_3142(data, threadData);
  WhirlpoolDiskStars_eqFunction_3145(data, threadData);
  WhirlpoolDiskStars_eqFunction_3147(data, threadData);
  WhirlpoolDiskStars_eqFunction_3150(data, threadData);
  WhirlpoolDiskStars_eqFunction_3149(data, threadData);
  WhirlpoolDiskStars_eqFunction_3148(data, threadData);
  WhirlpoolDiskStars_eqFunction_3146(data, threadData);
  WhirlpoolDiskStars_eqFunction_943(data, threadData);
  WhirlpoolDiskStars_eqFunction_3141(data, threadData);
  WhirlpoolDiskStars_eqFunction_945(data, threadData);
  WhirlpoolDiskStars_eqFunction_3154(data, threadData);
  WhirlpoolDiskStars_eqFunction_947(data, threadData);
  WhirlpoolDiskStars_eqFunction_948(data, threadData);
  WhirlpoolDiskStars_eqFunction_3153(data, threadData);
  WhirlpoolDiskStars_eqFunction_950(data, threadData);
  WhirlpoolDiskStars_eqFunction_951(data, threadData);
  WhirlpoolDiskStars_eqFunction_3152(data, threadData);
  WhirlpoolDiskStars_eqFunction_3155(data, threadData);
  WhirlpoolDiskStars_eqFunction_3157(data, threadData);
  WhirlpoolDiskStars_eqFunction_3160(data, threadData);
  WhirlpoolDiskStars_eqFunction_3159(data, threadData);
  WhirlpoolDiskStars_eqFunction_3158(data, threadData);
  WhirlpoolDiskStars_eqFunction_3156(data, threadData);
  WhirlpoolDiskStars_eqFunction_959(data, threadData);
  WhirlpoolDiskStars_eqFunction_3151(data, threadData);
  WhirlpoolDiskStars_eqFunction_961(data, threadData);
  WhirlpoolDiskStars_eqFunction_3164(data, threadData);
  WhirlpoolDiskStars_eqFunction_963(data, threadData);
  WhirlpoolDiskStars_eqFunction_964(data, threadData);
  WhirlpoolDiskStars_eqFunction_3163(data, threadData);
  WhirlpoolDiskStars_eqFunction_966(data, threadData);
  WhirlpoolDiskStars_eqFunction_967(data, threadData);
  WhirlpoolDiskStars_eqFunction_3162(data, threadData);
  WhirlpoolDiskStars_eqFunction_3165(data, threadData);
  WhirlpoolDiskStars_eqFunction_3167(data, threadData);
  WhirlpoolDiskStars_eqFunction_3170(data, threadData);
  WhirlpoolDiskStars_eqFunction_3169(data, threadData);
  WhirlpoolDiskStars_eqFunction_3168(data, threadData);
  WhirlpoolDiskStars_eqFunction_3166(data, threadData);
  WhirlpoolDiskStars_eqFunction_975(data, threadData);
  WhirlpoolDiskStars_eqFunction_3161(data, threadData);
  WhirlpoolDiskStars_eqFunction_977(data, threadData);
  WhirlpoolDiskStars_eqFunction_3174(data, threadData);
  WhirlpoolDiskStars_eqFunction_979(data, threadData);
  WhirlpoolDiskStars_eqFunction_980(data, threadData);
  WhirlpoolDiskStars_eqFunction_3173(data, threadData);
  WhirlpoolDiskStars_eqFunction_982(data, threadData);
  WhirlpoolDiskStars_eqFunction_983(data, threadData);
  WhirlpoolDiskStars_eqFunction_3172(data, threadData);
  WhirlpoolDiskStars_eqFunction_3175(data, threadData);
  WhirlpoolDiskStars_eqFunction_3177(data, threadData);
  WhirlpoolDiskStars_eqFunction_3180(data, threadData);
  WhirlpoolDiskStars_eqFunction_3179(data, threadData);
  WhirlpoolDiskStars_eqFunction_3178(data, threadData);
  WhirlpoolDiskStars_eqFunction_3176(data, threadData);
  WhirlpoolDiskStars_eqFunction_991(data, threadData);
  WhirlpoolDiskStars_eqFunction_3171(data, threadData);
  WhirlpoolDiskStars_eqFunction_993(data, threadData);
  WhirlpoolDiskStars_eqFunction_3184(data, threadData);
  WhirlpoolDiskStars_eqFunction_995(data, threadData);
  WhirlpoolDiskStars_eqFunction_996(data, threadData);
  WhirlpoolDiskStars_eqFunction_3183(data, threadData);
  WhirlpoolDiskStars_eqFunction_998(data, threadData);
  WhirlpoolDiskStars_eqFunction_999(data, threadData);
  WhirlpoolDiskStars_eqFunction_3182(data, threadData);
  WhirlpoolDiskStars_eqFunction_3185(data, threadData);
  WhirlpoolDiskStars_eqFunction_3187(data, threadData);
  WhirlpoolDiskStars_eqFunction_3190(data, threadData);
  WhirlpoolDiskStars_eqFunction_3189(data, threadData);
  WhirlpoolDiskStars_eqFunction_3188(data, threadData);
  WhirlpoolDiskStars_eqFunction_3186(data, threadData);
  WhirlpoolDiskStars_eqFunction_1007(data, threadData);
  WhirlpoolDiskStars_eqFunction_3181(data, threadData);
  WhirlpoolDiskStars_eqFunction_1009(data, threadData);
  WhirlpoolDiskStars_eqFunction_3194(data, threadData);
  WhirlpoolDiskStars_eqFunction_1011(data, threadData);
  WhirlpoolDiskStars_eqFunction_1012(data, threadData);
  WhirlpoolDiskStars_eqFunction_3193(data, threadData);
  WhirlpoolDiskStars_eqFunction_1014(data, threadData);
  WhirlpoolDiskStars_eqFunction_1015(data, threadData);
  WhirlpoolDiskStars_eqFunction_3192(data, threadData);
  WhirlpoolDiskStars_eqFunction_3195(data, threadData);
  WhirlpoolDiskStars_eqFunction_3197(data, threadData);
  WhirlpoolDiskStars_eqFunction_3200(data, threadData);
  WhirlpoolDiskStars_eqFunction_3199(data, threadData);
  WhirlpoolDiskStars_eqFunction_3198(data, threadData);
  WhirlpoolDiskStars_eqFunction_3196(data, threadData);
  WhirlpoolDiskStars_eqFunction_1023(data, threadData);
  WhirlpoolDiskStars_eqFunction_3191(data, threadData);
  WhirlpoolDiskStars_eqFunction_1025(data, threadData);
  WhirlpoolDiskStars_eqFunction_3204(data, threadData);
  WhirlpoolDiskStars_eqFunction_1027(data, threadData);
  WhirlpoolDiskStars_eqFunction_1028(data, threadData);
  WhirlpoolDiskStars_eqFunction_3203(data, threadData);
  WhirlpoolDiskStars_eqFunction_1030(data, threadData);
  WhirlpoolDiskStars_eqFunction_1031(data, threadData);
  WhirlpoolDiskStars_eqFunction_3202(data, threadData);
  WhirlpoolDiskStars_eqFunction_3205(data, threadData);
  WhirlpoolDiskStars_eqFunction_3207(data, threadData);
  WhirlpoolDiskStars_eqFunction_3210(data, threadData);
  WhirlpoolDiskStars_eqFunction_3209(data, threadData);
  WhirlpoolDiskStars_eqFunction_3208(data, threadData);
  WhirlpoolDiskStars_eqFunction_3206(data, threadData);
  WhirlpoolDiskStars_eqFunction_1039(data, threadData);
  WhirlpoolDiskStars_eqFunction_3201(data, threadData);
  WhirlpoolDiskStars_eqFunction_1041(data, threadData);
  WhirlpoolDiskStars_eqFunction_3214(data, threadData);
  WhirlpoolDiskStars_eqFunction_1043(data, threadData);
  WhirlpoolDiskStars_eqFunction_1044(data, threadData);
  WhirlpoolDiskStars_eqFunction_3213(data, threadData);
  WhirlpoolDiskStars_eqFunction_1046(data, threadData);
  WhirlpoolDiskStars_eqFunction_1047(data, threadData);
  WhirlpoolDiskStars_eqFunction_3212(data, threadData);
  WhirlpoolDiskStars_eqFunction_3215(data, threadData);
  WhirlpoolDiskStars_eqFunction_3217(data, threadData);
  WhirlpoolDiskStars_eqFunction_3220(data, threadData);
  WhirlpoolDiskStars_eqFunction_3219(data, threadData);
  WhirlpoolDiskStars_eqFunction_3218(data, threadData);
  WhirlpoolDiskStars_eqFunction_3216(data, threadData);
  WhirlpoolDiskStars_eqFunction_1055(data, threadData);
  WhirlpoolDiskStars_eqFunction_3211(data, threadData);
  WhirlpoolDiskStars_eqFunction_1057(data, threadData);
  WhirlpoolDiskStars_eqFunction_3224(data, threadData);
  WhirlpoolDiskStars_eqFunction_1059(data, threadData);
  WhirlpoolDiskStars_eqFunction_1060(data, threadData);
  WhirlpoolDiskStars_eqFunction_3223(data, threadData);
  WhirlpoolDiskStars_eqFunction_1062(data, threadData);
  WhirlpoolDiskStars_eqFunction_1063(data, threadData);
  WhirlpoolDiskStars_eqFunction_3222(data, threadData);
  WhirlpoolDiskStars_eqFunction_3225(data, threadData);
  WhirlpoolDiskStars_eqFunction_3227(data, threadData);
  WhirlpoolDiskStars_eqFunction_3230(data, threadData);
  WhirlpoolDiskStars_eqFunction_3229(data, threadData);
  WhirlpoolDiskStars_eqFunction_3228(data, threadData);
  WhirlpoolDiskStars_eqFunction_3226(data, threadData);
  WhirlpoolDiskStars_eqFunction_1071(data, threadData);
  WhirlpoolDiskStars_eqFunction_3221(data, threadData);
  WhirlpoolDiskStars_eqFunction_1073(data, threadData);
  WhirlpoolDiskStars_eqFunction_3234(data, threadData);
  WhirlpoolDiskStars_eqFunction_1075(data, threadData);
  WhirlpoolDiskStars_eqFunction_1076(data, threadData);
  WhirlpoolDiskStars_eqFunction_3233(data, threadData);
  WhirlpoolDiskStars_eqFunction_1078(data, threadData);
  WhirlpoolDiskStars_eqFunction_1079(data, threadData);
  WhirlpoolDiskStars_eqFunction_3232(data, threadData);
  WhirlpoolDiskStars_eqFunction_3235(data, threadData);
  WhirlpoolDiskStars_eqFunction_3237(data, threadData);
  WhirlpoolDiskStars_eqFunction_3240(data, threadData);
  WhirlpoolDiskStars_eqFunction_3239(data, threadData);
  WhirlpoolDiskStars_eqFunction_3238(data, threadData);
  WhirlpoolDiskStars_eqFunction_3236(data, threadData);
  WhirlpoolDiskStars_eqFunction_1087(data, threadData);
  WhirlpoolDiskStars_eqFunction_3231(data, threadData);
  WhirlpoolDiskStars_eqFunction_1089(data, threadData);
  WhirlpoolDiskStars_eqFunction_3244(data, threadData);
  WhirlpoolDiskStars_eqFunction_1091(data, threadData);
  WhirlpoolDiskStars_eqFunction_1092(data, threadData);
  WhirlpoolDiskStars_eqFunction_3243(data, threadData);
  WhirlpoolDiskStars_eqFunction_1094(data, threadData);
  WhirlpoolDiskStars_eqFunction_1095(data, threadData);
  WhirlpoolDiskStars_eqFunction_3242(data, threadData);
  WhirlpoolDiskStars_eqFunction_3245(data, threadData);
  WhirlpoolDiskStars_eqFunction_3247(data, threadData);
  WhirlpoolDiskStars_eqFunction_3250(data, threadData);
  WhirlpoolDiskStars_eqFunction_3249(data, threadData);
  WhirlpoolDiskStars_eqFunction_3248(data, threadData);
  WhirlpoolDiskStars_eqFunction_3246(data, threadData);
  WhirlpoolDiskStars_eqFunction_1103(data, threadData);
  WhirlpoolDiskStars_eqFunction_3241(data, threadData);
  WhirlpoolDiskStars_eqFunction_1105(data, threadData);
  WhirlpoolDiskStars_eqFunction_3254(data, threadData);
  WhirlpoolDiskStars_eqFunction_1107(data, threadData);
  WhirlpoolDiskStars_eqFunction_1108(data, threadData);
  WhirlpoolDiskStars_eqFunction_3253(data, threadData);
  WhirlpoolDiskStars_eqFunction_1110(data, threadData);
  WhirlpoolDiskStars_eqFunction_1111(data, threadData);
  WhirlpoolDiskStars_eqFunction_3252(data, threadData);
  WhirlpoolDiskStars_eqFunction_3255(data, threadData);
  WhirlpoolDiskStars_eqFunction_3257(data, threadData);
  WhirlpoolDiskStars_eqFunction_3260(data, threadData);
  WhirlpoolDiskStars_eqFunction_3259(data, threadData);
  WhirlpoolDiskStars_eqFunction_3258(data, threadData);
  WhirlpoolDiskStars_eqFunction_3256(data, threadData);
  WhirlpoolDiskStars_eqFunction_1119(data, threadData);
  WhirlpoolDiskStars_eqFunction_3251(data, threadData);
  WhirlpoolDiskStars_eqFunction_1121(data, threadData);
  WhirlpoolDiskStars_eqFunction_3264(data, threadData);
  WhirlpoolDiskStars_eqFunction_1123(data, threadData);
  WhirlpoolDiskStars_eqFunction_1124(data, threadData);
  WhirlpoolDiskStars_eqFunction_3263(data, threadData);
  WhirlpoolDiskStars_eqFunction_1126(data, threadData);
  WhirlpoolDiskStars_eqFunction_1127(data, threadData);
  WhirlpoolDiskStars_eqFunction_3262(data, threadData);
  WhirlpoolDiskStars_eqFunction_3265(data, threadData);
  WhirlpoolDiskStars_eqFunction_3267(data, threadData);
  WhirlpoolDiskStars_eqFunction_3270(data, threadData);
  WhirlpoolDiskStars_eqFunction_3269(data, threadData);
  WhirlpoolDiskStars_eqFunction_3268(data, threadData);
  WhirlpoolDiskStars_eqFunction_3266(data, threadData);
  WhirlpoolDiskStars_eqFunction_1135(data, threadData);
  WhirlpoolDiskStars_eqFunction_3261(data, threadData);
  WhirlpoolDiskStars_eqFunction_1137(data, threadData);
  WhirlpoolDiskStars_eqFunction_3274(data, threadData);
  WhirlpoolDiskStars_eqFunction_1139(data, threadData);
  WhirlpoolDiskStars_eqFunction_1140(data, threadData);
  WhirlpoolDiskStars_eqFunction_3273(data, threadData);
  WhirlpoolDiskStars_eqFunction_1142(data, threadData);
  WhirlpoolDiskStars_eqFunction_1143(data, threadData);
  WhirlpoolDiskStars_eqFunction_3272(data, threadData);
  WhirlpoolDiskStars_eqFunction_3275(data, threadData);
  WhirlpoolDiskStars_eqFunction_3277(data, threadData);
  WhirlpoolDiskStars_eqFunction_3280(data, threadData);
  WhirlpoolDiskStars_eqFunction_3279(data, threadData);
  WhirlpoolDiskStars_eqFunction_3278(data, threadData);
  WhirlpoolDiskStars_eqFunction_3276(data, threadData);
  WhirlpoolDiskStars_eqFunction_1151(data, threadData);
  WhirlpoolDiskStars_eqFunction_3271(data, threadData);
  WhirlpoolDiskStars_eqFunction_1153(data, threadData);
  WhirlpoolDiskStars_eqFunction_3284(data, threadData);
  WhirlpoolDiskStars_eqFunction_1155(data, threadData);
  WhirlpoolDiskStars_eqFunction_1156(data, threadData);
  WhirlpoolDiskStars_eqFunction_3283(data, threadData);
  WhirlpoolDiskStars_eqFunction_1158(data, threadData);
  WhirlpoolDiskStars_eqFunction_1159(data, threadData);
  WhirlpoolDiskStars_eqFunction_3282(data, threadData);
  WhirlpoolDiskStars_eqFunction_3285(data, threadData);
  WhirlpoolDiskStars_eqFunction_3287(data, threadData);
  WhirlpoolDiskStars_eqFunction_3290(data, threadData);
  WhirlpoolDiskStars_eqFunction_3289(data, threadData);
  WhirlpoolDiskStars_eqFunction_3288(data, threadData);
  WhirlpoolDiskStars_eqFunction_3286(data, threadData);
  WhirlpoolDiskStars_eqFunction_1167(data, threadData);
  WhirlpoolDiskStars_eqFunction_3281(data, threadData);
  WhirlpoolDiskStars_eqFunction_1169(data, threadData);
  WhirlpoolDiskStars_eqFunction_3294(data, threadData);
  WhirlpoolDiskStars_eqFunction_1171(data, threadData);
  WhirlpoolDiskStars_eqFunction_1172(data, threadData);
  WhirlpoolDiskStars_eqFunction_3293(data, threadData);
  WhirlpoolDiskStars_eqFunction_1174(data, threadData);
  WhirlpoolDiskStars_eqFunction_1175(data, threadData);
  WhirlpoolDiskStars_eqFunction_3292(data, threadData);
  WhirlpoolDiskStars_eqFunction_3295(data, threadData);
  WhirlpoolDiskStars_eqFunction_3297(data, threadData);
  WhirlpoolDiskStars_eqFunction_3300(data, threadData);
  WhirlpoolDiskStars_eqFunction_3299(data, threadData);
  WhirlpoolDiskStars_eqFunction_3298(data, threadData);
  WhirlpoolDiskStars_eqFunction_3296(data, threadData);
  WhirlpoolDiskStars_eqFunction_1183(data, threadData);
  WhirlpoolDiskStars_eqFunction_3291(data, threadData);
  WhirlpoolDiskStars_eqFunction_1185(data, threadData);
  WhirlpoolDiskStars_eqFunction_3304(data, threadData);
  WhirlpoolDiskStars_eqFunction_1187(data, threadData);
  WhirlpoolDiskStars_eqFunction_1188(data, threadData);
  WhirlpoolDiskStars_eqFunction_3303(data, threadData);
  WhirlpoolDiskStars_eqFunction_1190(data, threadData);
  WhirlpoolDiskStars_eqFunction_1191(data, threadData);
  WhirlpoolDiskStars_eqFunction_3302(data, threadData);
  WhirlpoolDiskStars_eqFunction_3305(data, threadData);
  WhirlpoolDiskStars_eqFunction_3307(data, threadData);
  WhirlpoolDiskStars_eqFunction_3310(data, threadData);
  WhirlpoolDiskStars_eqFunction_3309(data, threadData);
  WhirlpoolDiskStars_eqFunction_3308(data, threadData);
  WhirlpoolDiskStars_eqFunction_3306(data, threadData);
  WhirlpoolDiskStars_eqFunction_1199(data, threadData);
  WhirlpoolDiskStars_eqFunction_3301(data, threadData);
  WhirlpoolDiskStars_eqFunction_1201(data, threadData);
  WhirlpoolDiskStars_eqFunction_3314(data, threadData);
  WhirlpoolDiskStars_eqFunction_1203(data, threadData);
  WhirlpoolDiskStars_eqFunction_1204(data, threadData);
  WhirlpoolDiskStars_eqFunction_3313(data, threadData);
  WhirlpoolDiskStars_eqFunction_1206(data, threadData);
  WhirlpoolDiskStars_eqFunction_1207(data, threadData);
  WhirlpoolDiskStars_eqFunction_3312(data, threadData);
  WhirlpoolDiskStars_eqFunction_3315(data, threadData);
  WhirlpoolDiskStars_eqFunction_3317(data, threadData);
  WhirlpoolDiskStars_eqFunction_3320(data, threadData);
  WhirlpoolDiskStars_eqFunction_3319(data, threadData);
  WhirlpoolDiskStars_eqFunction_3318(data, threadData);
  WhirlpoolDiskStars_eqFunction_3316(data, threadData);
  WhirlpoolDiskStars_eqFunction_1215(data, threadData);
  WhirlpoolDiskStars_eqFunction_3311(data, threadData);
  WhirlpoolDiskStars_eqFunction_1217(data, threadData);
  WhirlpoolDiskStars_eqFunction_3324(data, threadData);
  WhirlpoolDiskStars_eqFunction_1219(data, threadData);
  WhirlpoolDiskStars_eqFunction_1220(data, threadData);
  WhirlpoolDiskStars_eqFunction_3323(data, threadData);
  WhirlpoolDiskStars_eqFunction_1222(data, threadData);
  WhirlpoolDiskStars_eqFunction_1223(data, threadData);
  WhirlpoolDiskStars_eqFunction_3322(data, threadData);
  WhirlpoolDiskStars_eqFunction_3325(data, threadData);
  WhirlpoolDiskStars_eqFunction_3327(data, threadData);
  WhirlpoolDiskStars_eqFunction_3330(data, threadData);
  WhirlpoolDiskStars_eqFunction_3329(data, threadData);
  WhirlpoolDiskStars_eqFunction_3328(data, threadData);
  WhirlpoolDiskStars_eqFunction_3326(data, threadData);
  WhirlpoolDiskStars_eqFunction_1231(data, threadData);
  WhirlpoolDiskStars_eqFunction_3321(data, threadData);
  WhirlpoolDiskStars_eqFunction_1233(data, threadData);
  WhirlpoolDiskStars_eqFunction_3334(data, threadData);
  WhirlpoolDiskStars_eqFunction_1235(data, threadData);
  WhirlpoolDiskStars_eqFunction_1236(data, threadData);
  WhirlpoolDiskStars_eqFunction_3333(data, threadData);
  WhirlpoolDiskStars_eqFunction_1238(data, threadData);
  WhirlpoolDiskStars_eqFunction_1239(data, threadData);
  WhirlpoolDiskStars_eqFunction_3332(data, threadData);
  WhirlpoolDiskStars_eqFunction_3335(data, threadData);
  WhirlpoolDiskStars_eqFunction_3337(data, threadData);
  WhirlpoolDiskStars_eqFunction_3340(data, threadData);
  WhirlpoolDiskStars_eqFunction_3339(data, threadData);
  WhirlpoolDiskStars_eqFunction_3338(data, threadData);
  WhirlpoolDiskStars_eqFunction_3336(data, threadData);
  WhirlpoolDiskStars_eqFunction_1247(data, threadData);
  WhirlpoolDiskStars_eqFunction_3331(data, threadData);
  WhirlpoolDiskStars_eqFunction_1249(data, threadData);
  WhirlpoolDiskStars_eqFunction_3344(data, threadData);
  WhirlpoolDiskStars_eqFunction_1251(data, threadData);
  WhirlpoolDiskStars_eqFunction_1252(data, threadData);
  WhirlpoolDiskStars_eqFunction_3343(data, threadData);
  WhirlpoolDiskStars_eqFunction_1254(data, threadData);
  WhirlpoolDiskStars_eqFunction_1255(data, threadData);
  WhirlpoolDiskStars_eqFunction_3342(data, threadData);
  WhirlpoolDiskStars_eqFunction_3345(data, threadData);
  WhirlpoolDiskStars_eqFunction_3347(data, threadData);
  WhirlpoolDiskStars_eqFunction_3350(data, threadData);
  WhirlpoolDiskStars_eqFunction_3349(data, threadData);
  WhirlpoolDiskStars_eqFunction_3348(data, threadData);
  WhirlpoolDiskStars_eqFunction_3346(data, threadData);
  WhirlpoolDiskStars_eqFunction_1263(data, threadData);
  WhirlpoolDiskStars_eqFunction_3341(data, threadData);
  WhirlpoolDiskStars_eqFunction_1265(data, threadData);
  WhirlpoolDiskStars_eqFunction_3354(data, threadData);
  WhirlpoolDiskStars_eqFunction_1267(data, threadData);
  WhirlpoolDiskStars_eqFunction_1268(data, threadData);
  WhirlpoolDiskStars_eqFunction_3353(data, threadData);
  WhirlpoolDiskStars_eqFunction_1270(data, threadData);
  WhirlpoolDiskStars_eqFunction_1271(data, threadData);
  WhirlpoolDiskStars_eqFunction_3352(data, threadData);
  WhirlpoolDiskStars_eqFunction_3355(data, threadData);
  WhirlpoolDiskStars_eqFunction_3357(data, threadData);
  WhirlpoolDiskStars_eqFunction_3360(data, threadData);
  WhirlpoolDiskStars_eqFunction_3359(data, threadData);
  WhirlpoolDiskStars_eqFunction_3358(data, threadData);
  WhirlpoolDiskStars_eqFunction_3356(data, threadData);
  WhirlpoolDiskStars_eqFunction_1279(data, threadData);
  WhirlpoolDiskStars_eqFunction_3351(data, threadData);
  WhirlpoolDiskStars_eqFunction_1281(data, threadData);
  TRACE_POP
}
#if defined(__cplusplus)
}
#endif