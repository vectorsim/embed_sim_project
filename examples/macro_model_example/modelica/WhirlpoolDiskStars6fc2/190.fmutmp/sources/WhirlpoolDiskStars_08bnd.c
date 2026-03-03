/* update bound parameters and variable attributes (start, nominal, min, max) */
#include "WhirlpoolDiskStars_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

OMC_DISABLE_OPT
int WhirlpoolDiskStars_updateBoundVariableAttributes(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  /* min ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating min-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  /* max ******************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating max-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  /* nominal **************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating nominal-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  /* start ****************************************************** */
  infoStreamPrint(OMC_LOG_INIT, 1, "updating primary start-values");
  if (OMC_ACTIVE_STREAM(OMC_LOG_INIT)) messageClose(OMC_LOG_INIT);
  
  TRACE_POP
  return 0;
}

void WhirlpoolDiskStars_updateBoundParameters_0(DATA *data, threadData_t *threadData);

/*
equation index: 4161
type: SIMPLE_ASSIGN
r_init[160] = 3.0 * exp(k * (theta[160] + armOffset[160]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4161(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4161};
  (data->simulationInfo->realParameter[324] /* r_init[160] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[484] /* theta[160] PARAM */) + (data->simulationInfo->realParameter[162] /* armOffset[160] PARAM */))));
  TRACE_POP
}

/*
equation index: 4162
type: SIMPLE_ASSIGN
r_init[159] = 3.0 * exp(k * (theta[159] + armOffset[159]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4162(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4162};
  (data->simulationInfo->realParameter[323] /* r_init[159] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[483] /* theta[159] PARAM */) + (data->simulationInfo->realParameter[161] /* armOffset[159] PARAM */))));
  TRACE_POP
}

/*
equation index: 4163
type: SIMPLE_ASSIGN
r_init[158] = 3.0 * exp(k * (theta[158] + armOffset[158]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4163(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4163};
  (data->simulationInfo->realParameter[322] /* r_init[158] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[482] /* theta[158] PARAM */) + (data->simulationInfo->realParameter[160] /* armOffset[158] PARAM */))));
  TRACE_POP
}

/*
equation index: 4164
type: SIMPLE_ASSIGN
r_init[157] = 3.0 * exp(k * (theta[157] + armOffset[157]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4164(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4164};
  (data->simulationInfo->realParameter[321] /* r_init[157] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[481] /* theta[157] PARAM */) + (data->simulationInfo->realParameter[159] /* armOffset[157] PARAM */))));
  TRACE_POP
}

/*
equation index: 4165
type: SIMPLE_ASSIGN
r_init[156] = 3.0 * exp(k * (theta[156] + armOffset[156]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4165(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4165};
  (data->simulationInfo->realParameter[320] /* r_init[156] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[480] /* theta[156] PARAM */) + (data->simulationInfo->realParameter[158] /* armOffset[156] PARAM */))));
  TRACE_POP
}

/*
equation index: 4166
type: SIMPLE_ASSIGN
r_init[155] = 3.0 * exp(k * (theta[155] + armOffset[155]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4166(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4166};
  (data->simulationInfo->realParameter[319] /* r_init[155] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[479] /* theta[155] PARAM */) + (data->simulationInfo->realParameter[157] /* armOffset[155] PARAM */))));
  TRACE_POP
}

/*
equation index: 4167
type: SIMPLE_ASSIGN
r_init[154] = 3.0 * exp(k * (theta[154] + armOffset[154]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4167(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4167};
  (data->simulationInfo->realParameter[318] /* r_init[154] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[478] /* theta[154] PARAM */) + (data->simulationInfo->realParameter[156] /* armOffset[154] PARAM */))));
  TRACE_POP
}

/*
equation index: 4168
type: SIMPLE_ASSIGN
r_init[153] = 3.0 * exp(k * (theta[153] + armOffset[153]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4168(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4168};
  (data->simulationInfo->realParameter[317] /* r_init[153] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[477] /* theta[153] PARAM */) + (data->simulationInfo->realParameter[155] /* armOffset[153] PARAM */))));
  TRACE_POP
}

/*
equation index: 4169
type: SIMPLE_ASSIGN
r_init[152] = 3.0 * exp(k * (theta[152] + armOffset[152]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4169(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4169};
  (data->simulationInfo->realParameter[316] /* r_init[152] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[476] /* theta[152] PARAM */) + (data->simulationInfo->realParameter[154] /* armOffset[152] PARAM */))));
  TRACE_POP
}

/*
equation index: 4170
type: SIMPLE_ASSIGN
r_init[151] = 3.0 * exp(k * (theta[151] + armOffset[151]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4170(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4170};
  (data->simulationInfo->realParameter[315] /* r_init[151] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[475] /* theta[151] PARAM */) + (data->simulationInfo->realParameter[153] /* armOffset[151] PARAM */))));
  TRACE_POP
}

/*
equation index: 4171
type: SIMPLE_ASSIGN
r_init[150] = 3.0 * exp(k * (theta[150] + armOffset[150]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4171(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4171};
  (data->simulationInfo->realParameter[314] /* r_init[150] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[474] /* theta[150] PARAM */) + (data->simulationInfo->realParameter[152] /* armOffset[150] PARAM */))));
  TRACE_POP
}

/*
equation index: 4172
type: SIMPLE_ASSIGN
r_init[149] = 3.0 * exp(k * (theta[149] + armOffset[149]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4172(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4172};
  (data->simulationInfo->realParameter[313] /* r_init[149] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[473] /* theta[149] PARAM */) + (data->simulationInfo->realParameter[151] /* armOffset[149] PARAM */))));
  TRACE_POP
}

/*
equation index: 4173
type: SIMPLE_ASSIGN
r_init[148] = 3.0 * exp(k * (theta[148] + armOffset[148]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4173(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4173};
  (data->simulationInfo->realParameter[312] /* r_init[148] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[472] /* theta[148] PARAM */) + (data->simulationInfo->realParameter[150] /* armOffset[148] PARAM */))));
  TRACE_POP
}

/*
equation index: 4174
type: SIMPLE_ASSIGN
r_init[147] = 3.0 * exp(k * (theta[147] + armOffset[147]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4174(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4174};
  (data->simulationInfo->realParameter[311] /* r_init[147] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[471] /* theta[147] PARAM */) + (data->simulationInfo->realParameter[149] /* armOffset[147] PARAM */))));
  TRACE_POP
}

/*
equation index: 4175
type: SIMPLE_ASSIGN
r_init[146] = 3.0 * exp(k * (theta[146] + armOffset[146]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4175(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4175};
  (data->simulationInfo->realParameter[310] /* r_init[146] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[470] /* theta[146] PARAM */) + (data->simulationInfo->realParameter[148] /* armOffset[146] PARAM */))));
  TRACE_POP
}

/*
equation index: 4176
type: SIMPLE_ASSIGN
r_init[145] = 3.0 * exp(k * (theta[145] + armOffset[145]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4176(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4176};
  (data->simulationInfo->realParameter[309] /* r_init[145] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[469] /* theta[145] PARAM */) + (data->simulationInfo->realParameter[147] /* armOffset[145] PARAM */))));
  TRACE_POP
}

/*
equation index: 4177
type: SIMPLE_ASSIGN
r_init[144] = 3.0 * exp(k * (theta[144] + armOffset[144]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4177(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4177};
  (data->simulationInfo->realParameter[308] /* r_init[144] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[468] /* theta[144] PARAM */) + (data->simulationInfo->realParameter[146] /* armOffset[144] PARAM */))));
  TRACE_POP
}

/*
equation index: 4178
type: SIMPLE_ASSIGN
r_init[143] = 3.0 * exp(k * (theta[143] + armOffset[143]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4178(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4178};
  (data->simulationInfo->realParameter[307] /* r_init[143] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[467] /* theta[143] PARAM */) + (data->simulationInfo->realParameter[145] /* armOffset[143] PARAM */))));
  TRACE_POP
}

/*
equation index: 4179
type: SIMPLE_ASSIGN
r_init[142] = 3.0 * exp(k * (theta[142] + armOffset[142]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4179(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4179};
  (data->simulationInfo->realParameter[306] /* r_init[142] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[466] /* theta[142] PARAM */) + (data->simulationInfo->realParameter[144] /* armOffset[142] PARAM */))));
  TRACE_POP
}

/*
equation index: 4180
type: SIMPLE_ASSIGN
r_init[141] = 3.0 * exp(k * (theta[141] + armOffset[141]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4180(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4180};
  (data->simulationInfo->realParameter[305] /* r_init[141] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[465] /* theta[141] PARAM */) + (data->simulationInfo->realParameter[143] /* armOffset[141] PARAM */))));
  TRACE_POP
}

/*
equation index: 4181
type: SIMPLE_ASSIGN
r_init[140] = 3.0 * exp(k * (theta[140] + armOffset[140]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4181(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4181};
  (data->simulationInfo->realParameter[304] /* r_init[140] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[464] /* theta[140] PARAM */) + (data->simulationInfo->realParameter[142] /* armOffset[140] PARAM */))));
  TRACE_POP
}

/*
equation index: 4182
type: SIMPLE_ASSIGN
r_init[139] = 3.0 * exp(k * (theta[139] + armOffset[139]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4182(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4182};
  (data->simulationInfo->realParameter[303] /* r_init[139] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[463] /* theta[139] PARAM */) + (data->simulationInfo->realParameter[141] /* armOffset[139] PARAM */))));
  TRACE_POP
}

/*
equation index: 4183
type: SIMPLE_ASSIGN
r_init[138] = 3.0 * exp(k * (theta[138] + armOffset[138]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4183(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4183};
  (data->simulationInfo->realParameter[302] /* r_init[138] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[462] /* theta[138] PARAM */) + (data->simulationInfo->realParameter[140] /* armOffset[138] PARAM */))));
  TRACE_POP
}

/*
equation index: 4184
type: SIMPLE_ASSIGN
r_init[137] = 3.0 * exp(k * (theta[137] + armOffset[137]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4184(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4184};
  (data->simulationInfo->realParameter[301] /* r_init[137] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[461] /* theta[137] PARAM */) + (data->simulationInfo->realParameter[139] /* armOffset[137] PARAM */))));
  TRACE_POP
}

/*
equation index: 4185
type: SIMPLE_ASSIGN
r_init[136] = 3.0 * exp(k * (theta[136] + armOffset[136]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4185(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4185};
  (data->simulationInfo->realParameter[300] /* r_init[136] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[460] /* theta[136] PARAM */) + (data->simulationInfo->realParameter[138] /* armOffset[136] PARAM */))));
  TRACE_POP
}

/*
equation index: 4186
type: SIMPLE_ASSIGN
r_init[135] = 3.0 * exp(k * (theta[135] + armOffset[135]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4186(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4186};
  (data->simulationInfo->realParameter[299] /* r_init[135] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[459] /* theta[135] PARAM */) + (data->simulationInfo->realParameter[137] /* armOffset[135] PARAM */))));
  TRACE_POP
}

/*
equation index: 4187
type: SIMPLE_ASSIGN
r_init[134] = 3.0 * exp(k * (theta[134] + armOffset[134]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4187(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4187};
  (data->simulationInfo->realParameter[298] /* r_init[134] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[458] /* theta[134] PARAM */) + (data->simulationInfo->realParameter[136] /* armOffset[134] PARAM */))));
  TRACE_POP
}

/*
equation index: 4188
type: SIMPLE_ASSIGN
r_init[133] = 3.0 * exp(k * (theta[133] + armOffset[133]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4188(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4188};
  (data->simulationInfo->realParameter[297] /* r_init[133] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[457] /* theta[133] PARAM */) + (data->simulationInfo->realParameter[135] /* armOffset[133] PARAM */))));
  TRACE_POP
}

/*
equation index: 4189
type: SIMPLE_ASSIGN
r_init[132] = 3.0 * exp(k * (theta[132] + armOffset[132]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4189(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4189};
  (data->simulationInfo->realParameter[296] /* r_init[132] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[456] /* theta[132] PARAM */) + (data->simulationInfo->realParameter[134] /* armOffset[132] PARAM */))));
  TRACE_POP
}

/*
equation index: 4190
type: SIMPLE_ASSIGN
r_init[131] = 3.0 * exp(k * (theta[131] + armOffset[131]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4190(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4190};
  (data->simulationInfo->realParameter[295] /* r_init[131] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[455] /* theta[131] PARAM */) + (data->simulationInfo->realParameter[133] /* armOffset[131] PARAM */))));
  TRACE_POP
}

/*
equation index: 4191
type: SIMPLE_ASSIGN
r_init[130] = 3.0 * exp(k * (theta[130] + armOffset[130]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4191(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4191};
  (data->simulationInfo->realParameter[294] /* r_init[130] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[454] /* theta[130] PARAM */) + (data->simulationInfo->realParameter[132] /* armOffset[130] PARAM */))));
  TRACE_POP
}

/*
equation index: 4192
type: SIMPLE_ASSIGN
r_init[129] = 3.0 * exp(k * (theta[129] + armOffset[129]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4192(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4192};
  (data->simulationInfo->realParameter[293] /* r_init[129] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[453] /* theta[129] PARAM */) + (data->simulationInfo->realParameter[131] /* armOffset[129] PARAM */))));
  TRACE_POP
}

/*
equation index: 4193
type: SIMPLE_ASSIGN
r_init[128] = 3.0 * exp(k * (theta[128] + armOffset[128]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4193(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4193};
  (data->simulationInfo->realParameter[292] /* r_init[128] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[452] /* theta[128] PARAM */) + (data->simulationInfo->realParameter[130] /* armOffset[128] PARAM */))));
  TRACE_POP
}

/*
equation index: 4194
type: SIMPLE_ASSIGN
r_init[127] = 3.0 * exp(k * (theta[127] + armOffset[127]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4194(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4194};
  (data->simulationInfo->realParameter[291] /* r_init[127] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[451] /* theta[127] PARAM */) + (data->simulationInfo->realParameter[129] /* armOffset[127] PARAM */))));
  TRACE_POP
}

/*
equation index: 4195
type: SIMPLE_ASSIGN
r_init[126] = 3.0 * exp(k * (theta[126] + armOffset[126]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4195(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4195};
  (data->simulationInfo->realParameter[290] /* r_init[126] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[450] /* theta[126] PARAM */) + (data->simulationInfo->realParameter[128] /* armOffset[126] PARAM */))));
  TRACE_POP
}

/*
equation index: 4196
type: SIMPLE_ASSIGN
r_init[125] = 3.0 * exp(k * (theta[125] + armOffset[125]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4196(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4196};
  (data->simulationInfo->realParameter[289] /* r_init[125] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[449] /* theta[125] PARAM */) + (data->simulationInfo->realParameter[127] /* armOffset[125] PARAM */))));
  TRACE_POP
}

/*
equation index: 4197
type: SIMPLE_ASSIGN
r_init[124] = 3.0 * exp(k * (theta[124] + armOffset[124]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4197(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4197};
  (data->simulationInfo->realParameter[288] /* r_init[124] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[448] /* theta[124] PARAM */) + (data->simulationInfo->realParameter[126] /* armOffset[124] PARAM */))));
  TRACE_POP
}

/*
equation index: 4198
type: SIMPLE_ASSIGN
r_init[123] = 3.0 * exp(k * (theta[123] + armOffset[123]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4198(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4198};
  (data->simulationInfo->realParameter[287] /* r_init[123] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[447] /* theta[123] PARAM */) + (data->simulationInfo->realParameter[125] /* armOffset[123] PARAM */))));
  TRACE_POP
}

/*
equation index: 4199
type: SIMPLE_ASSIGN
r_init[122] = 3.0 * exp(k * (theta[122] + armOffset[122]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4199(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4199};
  (data->simulationInfo->realParameter[286] /* r_init[122] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[446] /* theta[122] PARAM */) + (data->simulationInfo->realParameter[124] /* armOffset[122] PARAM */))));
  TRACE_POP
}

/*
equation index: 4200
type: SIMPLE_ASSIGN
r_init[121] = 3.0 * exp(k * (theta[121] + armOffset[121]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4200(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4200};
  (data->simulationInfo->realParameter[285] /* r_init[121] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[445] /* theta[121] PARAM */) + (data->simulationInfo->realParameter[123] /* armOffset[121] PARAM */))));
  TRACE_POP
}

/*
equation index: 4201
type: SIMPLE_ASSIGN
r_init[120] = 3.0 * exp(k * (theta[120] + armOffset[120]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4201(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4201};
  (data->simulationInfo->realParameter[284] /* r_init[120] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[444] /* theta[120] PARAM */) + (data->simulationInfo->realParameter[122] /* armOffset[120] PARAM */))));
  TRACE_POP
}

/*
equation index: 4202
type: SIMPLE_ASSIGN
r_init[119] = 3.0 * exp(k * (theta[119] + armOffset[119]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4202(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4202};
  (data->simulationInfo->realParameter[283] /* r_init[119] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[443] /* theta[119] PARAM */) + (data->simulationInfo->realParameter[121] /* armOffset[119] PARAM */))));
  TRACE_POP
}

/*
equation index: 4203
type: SIMPLE_ASSIGN
r_init[118] = 3.0 * exp(k * (theta[118] + armOffset[118]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4203(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4203};
  (data->simulationInfo->realParameter[282] /* r_init[118] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[442] /* theta[118] PARAM */) + (data->simulationInfo->realParameter[120] /* armOffset[118] PARAM */))));
  TRACE_POP
}

/*
equation index: 4204
type: SIMPLE_ASSIGN
r_init[117] = 3.0 * exp(k * (theta[117] + armOffset[117]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4204(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4204};
  (data->simulationInfo->realParameter[281] /* r_init[117] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[441] /* theta[117] PARAM */) + (data->simulationInfo->realParameter[119] /* armOffset[117] PARAM */))));
  TRACE_POP
}

/*
equation index: 4205
type: SIMPLE_ASSIGN
r_init[116] = 3.0 * exp(k * (theta[116] + armOffset[116]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4205(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4205};
  (data->simulationInfo->realParameter[280] /* r_init[116] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[440] /* theta[116] PARAM */) + (data->simulationInfo->realParameter[118] /* armOffset[116] PARAM */))));
  TRACE_POP
}

/*
equation index: 4206
type: SIMPLE_ASSIGN
r_init[115] = 3.0 * exp(k * (theta[115] + armOffset[115]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4206(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4206};
  (data->simulationInfo->realParameter[279] /* r_init[115] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[439] /* theta[115] PARAM */) + (data->simulationInfo->realParameter[117] /* armOffset[115] PARAM */))));
  TRACE_POP
}

/*
equation index: 4207
type: SIMPLE_ASSIGN
r_init[114] = 3.0 * exp(k * (theta[114] + armOffset[114]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4207(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4207};
  (data->simulationInfo->realParameter[278] /* r_init[114] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[438] /* theta[114] PARAM */) + (data->simulationInfo->realParameter[116] /* armOffset[114] PARAM */))));
  TRACE_POP
}

/*
equation index: 4208
type: SIMPLE_ASSIGN
r_init[113] = 3.0 * exp(k * (theta[113] + armOffset[113]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4208(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4208};
  (data->simulationInfo->realParameter[277] /* r_init[113] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[437] /* theta[113] PARAM */) + (data->simulationInfo->realParameter[115] /* armOffset[113] PARAM */))));
  TRACE_POP
}

/*
equation index: 4209
type: SIMPLE_ASSIGN
r_init[112] = 3.0 * exp(k * (theta[112] + armOffset[112]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4209(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4209};
  (data->simulationInfo->realParameter[276] /* r_init[112] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[436] /* theta[112] PARAM */) + (data->simulationInfo->realParameter[114] /* armOffset[112] PARAM */))));
  TRACE_POP
}

/*
equation index: 4210
type: SIMPLE_ASSIGN
r_init[111] = 3.0 * exp(k * (theta[111] + armOffset[111]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4210(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4210};
  (data->simulationInfo->realParameter[275] /* r_init[111] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[435] /* theta[111] PARAM */) + (data->simulationInfo->realParameter[113] /* armOffset[111] PARAM */))));
  TRACE_POP
}

/*
equation index: 4211
type: SIMPLE_ASSIGN
r_init[110] = 3.0 * exp(k * (theta[110] + armOffset[110]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4211(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4211};
  (data->simulationInfo->realParameter[274] /* r_init[110] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[434] /* theta[110] PARAM */) + (data->simulationInfo->realParameter[112] /* armOffset[110] PARAM */))));
  TRACE_POP
}

/*
equation index: 4212
type: SIMPLE_ASSIGN
r_init[109] = 3.0 * exp(k * (theta[109] + armOffset[109]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4212(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4212};
  (data->simulationInfo->realParameter[273] /* r_init[109] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[433] /* theta[109] PARAM */) + (data->simulationInfo->realParameter[111] /* armOffset[109] PARAM */))));
  TRACE_POP
}

/*
equation index: 4213
type: SIMPLE_ASSIGN
r_init[108] = 3.0 * exp(k * (theta[108] + armOffset[108]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4213(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4213};
  (data->simulationInfo->realParameter[272] /* r_init[108] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[432] /* theta[108] PARAM */) + (data->simulationInfo->realParameter[110] /* armOffset[108] PARAM */))));
  TRACE_POP
}

/*
equation index: 4214
type: SIMPLE_ASSIGN
r_init[107] = 3.0 * exp(k * (theta[107] + armOffset[107]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4214(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4214};
  (data->simulationInfo->realParameter[271] /* r_init[107] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[431] /* theta[107] PARAM */) + (data->simulationInfo->realParameter[109] /* armOffset[107] PARAM */))));
  TRACE_POP
}

/*
equation index: 4215
type: SIMPLE_ASSIGN
r_init[106] = 3.0 * exp(k * (theta[106] + armOffset[106]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4215(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4215};
  (data->simulationInfo->realParameter[270] /* r_init[106] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[430] /* theta[106] PARAM */) + (data->simulationInfo->realParameter[108] /* armOffset[106] PARAM */))));
  TRACE_POP
}

/*
equation index: 4216
type: SIMPLE_ASSIGN
r_init[105] = 3.0 * exp(k * (theta[105] + armOffset[105]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4216(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4216};
  (data->simulationInfo->realParameter[269] /* r_init[105] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[429] /* theta[105] PARAM */) + (data->simulationInfo->realParameter[107] /* armOffset[105] PARAM */))));
  TRACE_POP
}

/*
equation index: 4217
type: SIMPLE_ASSIGN
r_init[104] = 3.0 * exp(k * (theta[104] + armOffset[104]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4217(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4217};
  (data->simulationInfo->realParameter[268] /* r_init[104] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[428] /* theta[104] PARAM */) + (data->simulationInfo->realParameter[106] /* armOffset[104] PARAM */))));
  TRACE_POP
}

/*
equation index: 4218
type: SIMPLE_ASSIGN
r_init[103] = 3.0 * exp(k * (theta[103] + armOffset[103]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4218(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4218};
  (data->simulationInfo->realParameter[267] /* r_init[103] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[427] /* theta[103] PARAM */) + (data->simulationInfo->realParameter[105] /* armOffset[103] PARAM */))));
  TRACE_POP
}

/*
equation index: 4219
type: SIMPLE_ASSIGN
r_init[102] = 3.0 * exp(k * (theta[102] + armOffset[102]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4219(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4219};
  (data->simulationInfo->realParameter[266] /* r_init[102] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[426] /* theta[102] PARAM */) + (data->simulationInfo->realParameter[104] /* armOffset[102] PARAM */))));
  TRACE_POP
}

/*
equation index: 4220
type: SIMPLE_ASSIGN
r_init[101] = 3.0 * exp(k * (theta[101] + armOffset[101]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4220(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4220};
  (data->simulationInfo->realParameter[265] /* r_init[101] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[425] /* theta[101] PARAM */) + (data->simulationInfo->realParameter[103] /* armOffset[101] PARAM */))));
  TRACE_POP
}

/*
equation index: 4221
type: SIMPLE_ASSIGN
r_init[100] = 3.0 * exp(k * (theta[100] + armOffset[100]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4221(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4221};
  (data->simulationInfo->realParameter[264] /* r_init[100] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[424] /* theta[100] PARAM */) + (data->simulationInfo->realParameter[102] /* armOffset[100] PARAM */))));
  TRACE_POP
}

/*
equation index: 4222
type: SIMPLE_ASSIGN
r_init[99] = 3.0 * exp(k * (theta[99] + armOffset[99]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4222(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4222};
  (data->simulationInfo->realParameter[263] /* r_init[99] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[423] /* theta[99] PARAM */) + (data->simulationInfo->realParameter[101] /* armOffset[99] PARAM */))));
  TRACE_POP
}

/*
equation index: 4223
type: SIMPLE_ASSIGN
r_init[98] = 3.0 * exp(k * (theta[98] + armOffset[98]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4223(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4223};
  (data->simulationInfo->realParameter[262] /* r_init[98] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[422] /* theta[98] PARAM */) + (data->simulationInfo->realParameter[100] /* armOffset[98] PARAM */))));
  TRACE_POP
}

/*
equation index: 4224
type: SIMPLE_ASSIGN
r_init[97] = 3.0 * exp(k * (theta[97] + armOffset[97]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4224(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4224};
  (data->simulationInfo->realParameter[261] /* r_init[97] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[421] /* theta[97] PARAM */) + (data->simulationInfo->realParameter[99] /* armOffset[97] PARAM */))));
  TRACE_POP
}

/*
equation index: 4225
type: SIMPLE_ASSIGN
r_init[96] = 3.0 * exp(k * (theta[96] + armOffset[96]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4225(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4225};
  (data->simulationInfo->realParameter[260] /* r_init[96] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[420] /* theta[96] PARAM */) + (data->simulationInfo->realParameter[98] /* armOffset[96] PARAM */))));
  TRACE_POP
}

/*
equation index: 4226
type: SIMPLE_ASSIGN
r_init[95] = 3.0 * exp(k * (theta[95] + armOffset[95]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4226(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4226};
  (data->simulationInfo->realParameter[259] /* r_init[95] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[419] /* theta[95] PARAM */) + (data->simulationInfo->realParameter[97] /* armOffset[95] PARAM */))));
  TRACE_POP
}

/*
equation index: 4227
type: SIMPLE_ASSIGN
r_init[94] = 3.0 * exp(k * (theta[94] + armOffset[94]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4227(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4227};
  (data->simulationInfo->realParameter[258] /* r_init[94] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[418] /* theta[94] PARAM */) + (data->simulationInfo->realParameter[96] /* armOffset[94] PARAM */))));
  TRACE_POP
}

/*
equation index: 4228
type: SIMPLE_ASSIGN
r_init[93] = 3.0 * exp(k * (theta[93] + armOffset[93]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4228(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4228};
  (data->simulationInfo->realParameter[257] /* r_init[93] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[417] /* theta[93] PARAM */) + (data->simulationInfo->realParameter[95] /* armOffset[93] PARAM */))));
  TRACE_POP
}

/*
equation index: 4229
type: SIMPLE_ASSIGN
r_init[92] = 3.0 * exp(k * (theta[92] + armOffset[92]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4229(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4229};
  (data->simulationInfo->realParameter[256] /* r_init[92] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[416] /* theta[92] PARAM */) + (data->simulationInfo->realParameter[94] /* armOffset[92] PARAM */))));
  TRACE_POP
}

/*
equation index: 4230
type: SIMPLE_ASSIGN
r_init[91] = 3.0 * exp(k * (theta[91] + armOffset[91]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4230(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4230};
  (data->simulationInfo->realParameter[255] /* r_init[91] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[415] /* theta[91] PARAM */) + (data->simulationInfo->realParameter[93] /* armOffset[91] PARAM */))));
  TRACE_POP
}

/*
equation index: 4231
type: SIMPLE_ASSIGN
r_init[90] = 3.0 * exp(k * (theta[90] + armOffset[90]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4231(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4231};
  (data->simulationInfo->realParameter[254] /* r_init[90] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[414] /* theta[90] PARAM */) + (data->simulationInfo->realParameter[92] /* armOffset[90] PARAM */))));
  TRACE_POP
}

/*
equation index: 4232
type: SIMPLE_ASSIGN
r_init[89] = 3.0 * exp(k * (theta[89] + armOffset[89]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4232(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4232};
  (data->simulationInfo->realParameter[253] /* r_init[89] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[413] /* theta[89] PARAM */) + (data->simulationInfo->realParameter[91] /* armOffset[89] PARAM */))));
  TRACE_POP
}

/*
equation index: 4233
type: SIMPLE_ASSIGN
r_init[88] = 3.0 * exp(k * (theta[88] + armOffset[88]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4233(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4233};
  (data->simulationInfo->realParameter[252] /* r_init[88] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[412] /* theta[88] PARAM */) + (data->simulationInfo->realParameter[90] /* armOffset[88] PARAM */))));
  TRACE_POP
}

/*
equation index: 4234
type: SIMPLE_ASSIGN
r_init[87] = 3.0 * exp(k * (theta[87] + armOffset[87]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4234(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4234};
  (data->simulationInfo->realParameter[251] /* r_init[87] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[411] /* theta[87] PARAM */) + (data->simulationInfo->realParameter[89] /* armOffset[87] PARAM */))));
  TRACE_POP
}

/*
equation index: 4235
type: SIMPLE_ASSIGN
r_init[86] = 3.0 * exp(k * (theta[86] + armOffset[86]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4235(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4235};
  (data->simulationInfo->realParameter[250] /* r_init[86] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[410] /* theta[86] PARAM */) + (data->simulationInfo->realParameter[88] /* armOffset[86] PARAM */))));
  TRACE_POP
}

/*
equation index: 4236
type: SIMPLE_ASSIGN
r_init[85] = 3.0 * exp(k * (theta[85] + armOffset[85]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4236(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4236};
  (data->simulationInfo->realParameter[249] /* r_init[85] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[409] /* theta[85] PARAM */) + (data->simulationInfo->realParameter[87] /* armOffset[85] PARAM */))));
  TRACE_POP
}

/*
equation index: 4237
type: SIMPLE_ASSIGN
r_init[84] = 3.0 * exp(k * (theta[84] + armOffset[84]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4237(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4237};
  (data->simulationInfo->realParameter[248] /* r_init[84] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[408] /* theta[84] PARAM */) + (data->simulationInfo->realParameter[86] /* armOffset[84] PARAM */))));
  TRACE_POP
}

/*
equation index: 4238
type: SIMPLE_ASSIGN
r_init[83] = 3.0 * exp(k * (theta[83] + armOffset[83]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4238(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4238};
  (data->simulationInfo->realParameter[247] /* r_init[83] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[407] /* theta[83] PARAM */) + (data->simulationInfo->realParameter[85] /* armOffset[83] PARAM */))));
  TRACE_POP
}

/*
equation index: 4239
type: SIMPLE_ASSIGN
r_init[82] = 3.0 * exp(k * (theta[82] + armOffset[82]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4239(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4239};
  (data->simulationInfo->realParameter[246] /* r_init[82] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[406] /* theta[82] PARAM */) + (data->simulationInfo->realParameter[84] /* armOffset[82] PARAM */))));
  TRACE_POP
}

/*
equation index: 4240
type: SIMPLE_ASSIGN
r_init[81] = 3.0 * exp(k * (theta[81] + armOffset[81]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4240(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4240};
  (data->simulationInfo->realParameter[245] /* r_init[81] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[405] /* theta[81] PARAM */) + (data->simulationInfo->realParameter[83] /* armOffset[81] PARAM */))));
  TRACE_POP
}

/*
equation index: 4241
type: SIMPLE_ASSIGN
r_init[80] = 3.0 * exp(k * (theta[80] + armOffset[80]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4241(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4241};
  (data->simulationInfo->realParameter[244] /* r_init[80] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[404] /* theta[80] PARAM */) + (data->simulationInfo->realParameter[82] /* armOffset[80] PARAM */))));
  TRACE_POP
}

/*
equation index: 4242
type: SIMPLE_ASSIGN
r_init[79] = 3.0 * exp(k * (theta[79] + armOffset[79]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4242(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4242};
  (data->simulationInfo->realParameter[243] /* r_init[79] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[403] /* theta[79] PARAM */) + (data->simulationInfo->realParameter[81] /* armOffset[79] PARAM */))));
  TRACE_POP
}

/*
equation index: 4243
type: SIMPLE_ASSIGN
r_init[78] = 3.0 * exp(k * (theta[78] + armOffset[78]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4243(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4243};
  (data->simulationInfo->realParameter[242] /* r_init[78] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[402] /* theta[78] PARAM */) + (data->simulationInfo->realParameter[80] /* armOffset[78] PARAM */))));
  TRACE_POP
}

/*
equation index: 4244
type: SIMPLE_ASSIGN
r_init[77] = 3.0 * exp(k * (theta[77] + armOffset[77]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4244(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4244};
  (data->simulationInfo->realParameter[241] /* r_init[77] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[401] /* theta[77] PARAM */) + (data->simulationInfo->realParameter[79] /* armOffset[77] PARAM */))));
  TRACE_POP
}

/*
equation index: 4245
type: SIMPLE_ASSIGN
r_init[76] = 3.0 * exp(k * (theta[76] + armOffset[76]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4245(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4245};
  (data->simulationInfo->realParameter[240] /* r_init[76] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[400] /* theta[76] PARAM */) + (data->simulationInfo->realParameter[78] /* armOffset[76] PARAM */))));
  TRACE_POP
}

/*
equation index: 4246
type: SIMPLE_ASSIGN
r_init[75] = 3.0 * exp(k * (theta[75] + armOffset[75]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4246(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4246};
  (data->simulationInfo->realParameter[239] /* r_init[75] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[399] /* theta[75] PARAM */) + (data->simulationInfo->realParameter[77] /* armOffset[75] PARAM */))));
  TRACE_POP
}

/*
equation index: 4247
type: SIMPLE_ASSIGN
r_init[74] = 3.0 * exp(k * (theta[74] + armOffset[74]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4247(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4247};
  (data->simulationInfo->realParameter[238] /* r_init[74] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[398] /* theta[74] PARAM */) + (data->simulationInfo->realParameter[76] /* armOffset[74] PARAM */))));
  TRACE_POP
}

/*
equation index: 4248
type: SIMPLE_ASSIGN
r_init[73] = 3.0 * exp(k * (theta[73] + armOffset[73]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4248(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4248};
  (data->simulationInfo->realParameter[237] /* r_init[73] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[397] /* theta[73] PARAM */) + (data->simulationInfo->realParameter[75] /* armOffset[73] PARAM */))));
  TRACE_POP
}

/*
equation index: 4249
type: SIMPLE_ASSIGN
r_init[72] = 3.0 * exp(k * (theta[72] + armOffset[72]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4249(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4249};
  (data->simulationInfo->realParameter[236] /* r_init[72] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[396] /* theta[72] PARAM */) + (data->simulationInfo->realParameter[74] /* armOffset[72] PARAM */))));
  TRACE_POP
}

/*
equation index: 4250
type: SIMPLE_ASSIGN
r_init[71] = 3.0 * exp(k * (theta[71] + armOffset[71]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4250(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4250};
  (data->simulationInfo->realParameter[235] /* r_init[71] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[395] /* theta[71] PARAM */) + (data->simulationInfo->realParameter[73] /* armOffset[71] PARAM */))));
  TRACE_POP
}

/*
equation index: 4251
type: SIMPLE_ASSIGN
r_init[70] = 3.0 * exp(k * (theta[70] + armOffset[70]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4251(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4251};
  (data->simulationInfo->realParameter[234] /* r_init[70] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[394] /* theta[70] PARAM */) + (data->simulationInfo->realParameter[72] /* armOffset[70] PARAM */))));
  TRACE_POP
}

/*
equation index: 4252
type: SIMPLE_ASSIGN
r_init[69] = 3.0 * exp(k * (theta[69] + armOffset[69]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4252(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4252};
  (data->simulationInfo->realParameter[233] /* r_init[69] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[393] /* theta[69] PARAM */) + (data->simulationInfo->realParameter[71] /* armOffset[69] PARAM */))));
  TRACE_POP
}

/*
equation index: 4253
type: SIMPLE_ASSIGN
r_init[68] = 3.0 * exp(k * (theta[68] + armOffset[68]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4253(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4253};
  (data->simulationInfo->realParameter[232] /* r_init[68] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[392] /* theta[68] PARAM */) + (data->simulationInfo->realParameter[70] /* armOffset[68] PARAM */))));
  TRACE_POP
}

/*
equation index: 4254
type: SIMPLE_ASSIGN
r_init[67] = 3.0 * exp(k * (theta[67] + armOffset[67]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4254(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4254};
  (data->simulationInfo->realParameter[231] /* r_init[67] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[391] /* theta[67] PARAM */) + (data->simulationInfo->realParameter[69] /* armOffset[67] PARAM */))));
  TRACE_POP
}

/*
equation index: 4255
type: SIMPLE_ASSIGN
r_init[66] = 3.0 * exp(k * (theta[66] + armOffset[66]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4255(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4255};
  (data->simulationInfo->realParameter[230] /* r_init[66] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[390] /* theta[66] PARAM */) + (data->simulationInfo->realParameter[68] /* armOffset[66] PARAM */))));
  TRACE_POP
}

/*
equation index: 4256
type: SIMPLE_ASSIGN
r_init[65] = 3.0 * exp(k * (theta[65] + armOffset[65]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4256(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4256};
  (data->simulationInfo->realParameter[229] /* r_init[65] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[389] /* theta[65] PARAM */) + (data->simulationInfo->realParameter[67] /* armOffset[65] PARAM */))));
  TRACE_POP
}

/*
equation index: 4257
type: SIMPLE_ASSIGN
r_init[64] = 3.0 * exp(k * (theta[64] + armOffset[64]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4257(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4257};
  (data->simulationInfo->realParameter[228] /* r_init[64] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[388] /* theta[64] PARAM */) + (data->simulationInfo->realParameter[66] /* armOffset[64] PARAM */))));
  TRACE_POP
}

/*
equation index: 4258
type: SIMPLE_ASSIGN
r_init[63] = 3.0 * exp(k * (theta[63] + armOffset[63]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4258(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4258};
  (data->simulationInfo->realParameter[227] /* r_init[63] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[387] /* theta[63] PARAM */) + (data->simulationInfo->realParameter[65] /* armOffset[63] PARAM */))));
  TRACE_POP
}

/*
equation index: 4259
type: SIMPLE_ASSIGN
r_init[62] = 3.0 * exp(k * (theta[62] + armOffset[62]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4259(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4259};
  (data->simulationInfo->realParameter[226] /* r_init[62] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[386] /* theta[62] PARAM */) + (data->simulationInfo->realParameter[64] /* armOffset[62] PARAM */))));
  TRACE_POP
}

/*
equation index: 4260
type: SIMPLE_ASSIGN
r_init[61] = 3.0 * exp(k * (theta[61] + armOffset[61]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4260(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4260};
  (data->simulationInfo->realParameter[225] /* r_init[61] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[385] /* theta[61] PARAM */) + (data->simulationInfo->realParameter[63] /* armOffset[61] PARAM */))));
  TRACE_POP
}

/*
equation index: 4261
type: SIMPLE_ASSIGN
r_init[60] = 3.0 * exp(k * (theta[60] + armOffset[60]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4261(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4261};
  (data->simulationInfo->realParameter[224] /* r_init[60] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[384] /* theta[60] PARAM */) + (data->simulationInfo->realParameter[62] /* armOffset[60] PARAM */))));
  TRACE_POP
}

/*
equation index: 4262
type: SIMPLE_ASSIGN
r_init[59] = 3.0 * exp(k * (theta[59] + armOffset[59]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4262(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4262};
  (data->simulationInfo->realParameter[223] /* r_init[59] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[383] /* theta[59] PARAM */) + (data->simulationInfo->realParameter[61] /* armOffset[59] PARAM */))));
  TRACE_POP
}

/*
equation index: 4263
type: SIMPLE_ASSIGN
r_init[58] = 3.0 * exp(k * (theta[58] + armOffset[58]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4263(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4263};
  (data->simulationInfo->realParameter[222] /* r_init[58] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[382] /* theta[58] PARAM */) + (data->simulationInfo->realParameter[60] /* armOffset[58] PARAM */))));
  TRACE_POP
}

/*
equation index: 4264
type: SIMPLE_ASSIGN
r_init[57] = 3.0 * exp(k * (theta[57] + armOffset[57]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4264(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4264};
  (data->simulationInfo->realParameter[221] /* r_init[57] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[381] /* theta[57] PARAM */) + (data->simulationInfo->realParameter[59] /* armOffset[57] PARAM */))));
  TRACE_POP
}

/*
equation index: 4265
type: SIMPLE_ASSIGN
r_init[56] = 3.0 * exp(k * (theta[56] + armOffset[56]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4265(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4265};
  (data->simulationInfo->realParameter[220] /* r_init[56] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[380] /* theta[56] PARAM */) + (data->simulationInfo->realParameter[58] /* armOffset[56] PARAM */))));
  TRACE_POP
}

/*
equation index: 4266
type: SIMPLE_ASSIGN
r_init[55] = 3.0 * exp(k * (theta[55] + armOffset[55]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4266(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4266};
  (data->simulationInfo->realParameter[219] /* r_init[55] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[379] /* theta[55] PARAM */) + (data->simulationInfo->realParameter[57] /* armOffset[55] PARAM */))));
  TRACE_POP
}

/*
equation index: 4267
type: SIMPLE_ASSIGN
r_init[54] = 3.0 * exp(k * (theta[54] + armOffset[54]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4267(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4267};
  (data->simulationInfo->realParameter[218] /* r_init[54] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[378] /* theta[54] PARAM */) + (data->simulationInfo->realParameter[56] /* armOffset[54] PARAM */))));
  TRACE_POP
}

/*
equation index: 4268
type: SIMPLE_ASSIGN
r_init[53] = 3.0 * exp(k * (theta[53] + armOffset[53]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4268(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4268};
  (data->simulationInfo->realParameter[217] /* r_init[53] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[377] /* theta[53] PARAM */) + (data->simulationInfo->realParameter[55] /* armOffset[53] PARAM */))));
  TRACE_POP
}

/*
equation index: 4269
type: SIMPLE_ASSIGN
r_init[52] = 3.0 * exp(k * (theta[52] + armOffset[52]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4269(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4269};
  (data->simulationInfo->realParameter[216] /* r_init[52] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[376] /* theta[52] PARAM */) + (data->simulationInfo->realParameter[54] /* armOffset[52] PARAM */))));
  TRACE_POP
}

/*
equation index: 4270
type: SIMPLE_ASSIGN
r_init[51] = 3.0 * exp(k * (theta[51] + armOffset[51]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4270(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4270};
  (data->simulationInfo->realParameter[215] /* r_init[51] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[375] /* theta[51] PARAM */) + (data->simulationInfo->realParameter[53] /* armOffset[51] PARAM */))));
  TRACE_POP
}

/*
equation index: 4271
type: SIMPLE_ASSIGN
r_init[50] = 3.0 * exp(k * (theta[50] + armOffset[50]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4271(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4271};
  (data->simulationInfo->realParameter[214] /* r_init[50] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[374] /* theta[50] PARAM */) + (data->simulationInfo->realParameter[52] /* armOffset[50] PARAM */))));
  TRACE_POP
}

/*
equation index: 4272
type: SIMPLE_ASSIGN
r_init[49] = 3.0 * exp(k * (theta[49] + armOffset[49]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4272(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4272};
  (data->simulationInfo->realParameter[213] /* r_init[49] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[373] /* theta[49] PARAM */) + (data->simulationInfo->realParameter[51] /* armOffset[49] PARAM */))));
  TRACE_POP
}

/*
equation index: 4273
type: SIMPLE_ASSIGN
r_init[48] = 3.0 * exp(k * (theta[48] + armOffset[48]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4273(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4273};
  (data->simulationInfo->realParameter[212] /* r_init[48] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[372] /* theta[48] PARAM */) + (data->simulationInfo->realParameter[50] /* armOffset[48] PARAM */))));
  TRACE_POP
}

/*
equation index: 4274
type: SIMPLE_ASSIGN
r_init[47] = 3.0 * exp(k * (theta[47] + armOffset[47]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4274(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4274};
  (data->simulationInfo->realParameter[211] /* r_init[47] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[371] /* theta[47] PARAM */) + (data->simulationInfo->realParameter[49] /* armOffset[47] PARAM */))));
  TRACE_POP
}

/*
equation index: 4275
type: SIMPLE_ASSIGN
r_init[46] = 3.0 * exp(k * (theta[46] + armOffset[46]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4275(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4275};
  (data->simulationInfo->realParameter[210] /* r_init[46] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[370] /* theta[46] PARAM */) + (data->simulationInfo->realParameter[48] /* armOffset[46] PARAM */))));
  TRACE_POP
}

/*
equation index: 4276
type: SIMPLE_ASSIGN
r_init[45] = 3.0 * exp(k * (theta[45] + armOffset[45]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4276(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4276};
  (data->simulationInfo->realParameter[209] /* r_init[45] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[369] /* theta[45] PARAM */) + (data->simulationInfo->realParameter[47] /* armOffset[45] PARAM */))));
  TRACE_POP
}

/*
equation index: 4277
type: SIMPLE_ASSIGN
r_init[44] = 3.0 * exp(k * (theta[44] + armOffset[44]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4277(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4277};
  (data->simulationInfo->realParameter[208] /* r_init[44] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[368] /* theta[44] PARAM */) + (data->simulationInfo->realParameter[46] /* armOffset[44] PARAM */))));
  TRACE_POP
}

/*
equation index: 4278
type: SIMPLE_ASSIGN
r_init[43] = 3.0 * exp(k * (theta[43] + armOffset[43]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4278(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4278};
  (data->simulationInfo->realParameter[207] /* r_init[43] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[367] /* theta[43] PARAM */) + (data->simulationInfo->realParameter[45] /* armOffset[43] PARAM */))));
  TRACE_POP
}

/*
equation index: 4279
type: SIMPLE_ASSIGN
r_init[42] = 3.0 * exp(k * (theta[42] + armOffset[42]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4279(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4279};
  (data->simulationInfo->realParameter[206] /* r_init[42] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[366] /* theta[42] PARAM */) + (data->simulationInfo->realParameter[44] /* armOffset[42] PARAM */))));
  TRACE_POP
}

/*
equation index: 4280
type: SIMPLE_ASSIGN
r_init[41] = 3.0 * exp(k * (theta[41] + armOffset[41]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4280(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4280};
  (data->simulationInfo->realParameter[205] /* r_init[41] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[365] /* theta[41] PARAM */) + (data->simulationInfo->realParameter[43] /* armOffset[41] PARAM */))));
  TRACE_POP
}

/*
equation index: 4281
type: SIMPLE_ASSIGN
r_init[40] = 3.0 * exp(k * (theta[40] + armOffset[40]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4281(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4281};
  (data->simulationInfo->realParameter[204] /* r_init[40] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[364] /* theta[40] PARAM */) + (data->simulationInfo->realParameter[42] /* armOffset[40] PARAM */))));
  TRACE_POP
}

/*
equation index: 4282
type: SIMPLE_ASSIGN
r_init[39] = 3.0 * exp(k * (theta[39] + armOffset[39]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4282(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4282};
  (data->simulationInfo->realParameter[203] /* r_init[39] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[363] /* theta[39] PARAM */) + (data->simulationInfo->realParameter[41] /* armOffset[39] PARAM */))));
  TRACE_POP
}

/*
equation index: 4283
type: SIMPLE_ASSIGN
r_init[38] = 3.0 * exp(k * (theta[38] + armOffset[38]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4283(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4283};
  (data->simulationInfo->realParameter[202] /* r_init[38] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[362] /* theta[38] PARAM */) + (data->simulationInfo->realParameter[40] /* armOffset[38] PARAM */))));
  TRACE_POP
}

/*
equation index: 4284
type: SIMPLE_ASSIGN
r_init[37] = 3.0 * exp(k * (theta[37] + armOffset[37]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4284(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4284};
  (data->simulationInfo->realParameter[201] /* r_init[37] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[361] /* theta[37] PARAM */) + (data->simulationInfo->realParameter[39] /* armOffset[37] PARAM */))));
  TRACE_POP
}

/*
equation index: 4285
type: SIMPLE_ASSIGN
r_init[36] = 3.0 * exp(k * (theta[36] + armOffset[36]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4285(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4285};
  (data->simulationInfo->realParameter[200] /* r_init[36] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[360] /* theta[36] PARAM */) + (data->simulationInfo->realParameter[38] /* armOffset[36] PARAM */))));
  TRACE_POP
}

/*
equation index: 4286
type: SIMPLE_ASSIGN
r_init[35] = 3.0 * exp(k * (theta[35] + armOffset[35]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4286(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4286};
  (data->simulationInfo->realParameter[199] /* r_init[35] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[359] /* theta[35] PARAM */) + (data->simulationInfo->realParameter[37] /* armOffset[35] PARAM */))));
  TRACE_POP
}

/*
equation index: 4287
type: SIMPLE_ASSIGN
r_init[34] = 3.0 * exp(k * (theta[34] + armOffset[34]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4287(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4287};
  (data->simulationInfo->realParameter[198] /* r_init[34] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[358] /* theta[34] PARAM */) + (data->simulationInfo->realParameter[36] /* armOffset[34] PARAM */))));
  TRACE_POP
}

/*
equation index: 4288
type: SIMPLE_ASSIGN
r_init[33] = 3.0 * exp(k * (theta[33] + armOffset[33]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4288(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4288};
  (data->simulationInfo->realParameter[197] /* r_init[33] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[357] /* theta[33] PARAM */) + (data->simulationInfo->realParameter[35] /* armOffset[33] PARAM */))));
  TRACE_POP
}

/*
equation index: 4289
type: SIMPLE_ASSIGN
r_init[32] = 3.0 * exp(k * (theta[32] + armOffset[32]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4289(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4289};
  (data->simulationInfo->realParameter[196] /* r_init[32] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[356] /* theta[32] PARAM */) + (data->simulationInfo->realParameter[34] /* armOffset[32] PARAM */))));
  TRACE_POP
}

/*
equation index: 4290
type: SIMPLE_ASSIGN
r_init[31] = 3.0 * exp(k * (theta[31] + armOffset[31]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4290(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4290};
  (data->simulationInfo->realParameter[195] /* r_init[31] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[355] /* theta[31] PARAM */) + (data->simulationInfo->realParameter[33] /* armOffset[31] PARAM */))));
  TRACE_POP
}

/*
equation index: 4291
type: SIMPLE_ASSIGN
r_init[30] = 3.0 * exp(k * (theta[30] + armOffset[30]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4291(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4291};
  (data->simulationInfo->realParameter[194] /* r_init[30] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[354] /* theta[30] PARAM */) + (data->simulationInfo->realParameter[32] /* armOffset[30] PARAM */))));
  TRACE_POP
}

/*
equation index: 4292
type: SIMPLE_ASSIGN
r_init[29] = 3.0 * exp(k * (theta[29] + armOffset[29]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4292(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4292};
  (data->simulationInfo->realParameter[193] /* r_init[29] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[353] /* theta[29] PARAM */) + (data->simulationInfo->realParameter[31] /* armOffset[29] PARAM */))));
  TRACE_POP
}

/*
equation index: 4293
type: SIMPLE_ASSIGN
r_init[28] = 3.0 * exp(k * (theta[28] + armOffset[28]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4293(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4293};
  (data->simulationInfo->realParameter[192] /* r_init[28] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[352] /* theta[28] PARAM */) + (data->simulationInfo->realParameter[30] /* armOffset[28] PARAM */))));
  TRACE_POP
}

/*
equation index: 4294
type: SIMPLE_ASSIGN
r_init[27] = 3.0 * exp(k * (theta[27] + armOffset[27]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4294(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4294};
  (data->simulationInfo->realParameter[191] /* r_init[27] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[351] /* theta[27] PARAM */) + (data->simulationInfo->realParameter[29] /* armOffset[27] PARAM */))));
  TRACE_POP
}

/*
equation index: 4295
type: SIMPLE_ASSIGN
r_init[26] = 3.0 * exp(k * (theta[26] + armOffset[26]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4295(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4295};
  (data->simulationInfo->realParameter[190] /* r_init[26] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[350] /* theta[26] PARAM */) + (data->simulationInfo->realParameter[28] /* armOffset[26] PARAM */))));
  TRACE_POP
}

/*
equation index: 4296
type: SIMPLE_ASSIGN
r_init[25] = 3.0 * exp(k * (theta[25] + armOffset[25]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4296(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4296};
  (data->simulationInfo->realParameter[189] /* r_init[25] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[349] /* theta[25] PARAM */) + (data->simulationInfo->realParameter[27] /* armOffset[25] PARAM */))));
  TRACE_POP
}

/*
equation index: 4297
type: SIMPLE_ASSIGN
r_init[24] = 3.0 * exp(k * (theta[24] + armOffset[24]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4297(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4297};
  (data->simulationInfo->realParameter[188] /* r_init[24] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[348] /* theta[24] PARAM */) + (data->simulationInfo->realParameter[26] /* armOffset[24] PARAM */))));
  TRACE_POP
}

/*
equation index: 4298
type: SIMPLE_ASSIGN
r_init[23] = 3.0 * exp(k * (theta[23] + armOffset[23]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4298(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4298};
  (data->simulationInfo->realParameter[187] /* r_init[23] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[347] /* theta[23] PARAM */) + (data->simulationInfo->realParameter[25] /* armOffset[23] PARAM */))));
  TRACE_POP
}

/*
equation index: 4299
type: SIMPLE_ASSIGN
r_init[22] = 3.0 * exp(k * (theta[22] + armOffset[22]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4299(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4299};
  (data->simulationInfo->realParameter[186] /* r_init[22] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[346] /* theta[22] PARAM */) + (data->simulationInfo->realParameter[24] /* armOffset[22] PARAM */))));
  TRACE_POP
}

/*
equation index: 4300
type: SIMPLE_ASSIGN
r_init[21] = 3.0 * exp(k * (theta[21] + armOffset[21]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4300(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4300};
  (data->simulationInfo->realParameter[185] /* r_init[21] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[345] /* theta[21] PARAM */) + (data->simulationInfo->realParameter[23] /* armOffset[21] PARAM */))));
  TRACE_POP
}

/*
equation index: 4301
type: SIMPLE_ASSIGN
r_init[20] = 3.0 * exp(k * (theta[20] + armOffset[20]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4301(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4301};
  (data->simulationInfo->realParameter[184] /* r_init[20] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[344] /* theta[20] PARAM */) + (data->simulationInfo->realParameter[22] /* armOffset[20] PARAM */))));
  TRACE_POP
}

/*
equation index: 4302
type: SIMPLE_ASSIGN
r_init[19] = 3.0 * exp(k * (theta[19] + armOffset[19]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4302(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4302};
  (data->simulationInfo->realParameter[183] /* r_init[19] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[343] /* theta[19] PARAM */) + (data->simulationInfo->realParameter[21] /* armOffset[19] PARAM */))));
  TRACE_POP
}

/*
equation index: 4303
type: SIMPLE_ASSIGN
r_init[18] = 3.0 * exp(k * (theta[18] + armOffset[18]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4303(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4303};
  (data->simulationInfo->realParameter[182] /* r_init[18] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[342] /* theta[18] PARAM */) + (data->simulationInfo->realParameter[20] /* armOffset[18] PARAM */))));
  TRACE_POP
}

/*
equation index: 4304
type: SIMPLE_ASSIGN
r_init[17] = 3.0 * exp(k * (theta[17] + armOffset[17]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4304(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4304};
  (data->simulationInfo->realParameter[181] /* r_init[17] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[341] /* theta[17] PARAM */) + (data->simulationInfo->realParameter[19] /* armOffset[17] PARAM */))));
  TRACE_POP
}

/*
equation index: 4305
type: SIMPLE_ASSIGN
r_init[16] = 3.0 * exp(k * (theta[16] + armOffset[16]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4305(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4305};
  (data->simulationInfo->realParameter[180] /* r_init[16] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[340] /* theta[16] PARAM */) + (data->simulationInfo->realParameter[18] /* armOffset[16] PARAM */))));
  TRACE_POP
}

/*
equation index: 4306
type: SIMPLE_ASSIGN
r_init[15] = 3.0 * exp(k * (theta[15] + armOffset[15]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4306(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4306};
  (data->simulationInfo->realParameter[179] /* r_init[15] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[339] /* theta[15] PARAM */) + (data->simulationInfo->realParameter[17] /* armOffset[15] PARAM */))));
  TRACE_POP
}

/*
equation index: 4307
type: SIMPLE_ASSIGN
r_init[14] = 3.0 * exp(k * (theta[14] + armOffset[14]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4307(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4307};
  (data->simulationInfo->realParameter[178] /* r_init[14] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[338] /* theta[14] PARAM */) + (data->simulationInfo->realParameter[16] /* armOffset[14] PARAM */))));
  TRACE_POP
}

/*
equation index: 4308
type: SIMPLE_ASSIGN
r_init[13] = 3.0 * exp(k * (theta[13] + armOffset[13]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4308(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4308};
  (data->simulationInfo->realParameter[177] /* r_init[13] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[337] /* theta[13] PARAM */) + (data->simulationInfo->realParameter[15] /* armOffset[13] PARAM */))));
  TRACE_POP
}

/*
equation index: 4309
type: SIMPLE_ASSIGN
r_init[12] = 3.0 * exp(k * (theta[12] + armOffset[12]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4309(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4309};
  (data->simulationInfo->realParameter[176] /* r_init[12] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[336] /* theta[12] PARAM */) + (data->simulationInfo->realParameter[14] /* armOffset[12] PARAM */))));
  TRACE_POP
}

/*
equation index: 4310
type: SIMPLE_ASSIGN
r_init[11] = 3.0 * exp(k * (theta[11] + armOffset[11]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4310(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4310};
  (data->simulationInfo->realParameter[175] /* r_init[11] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[335] /* theta[11] PARAM */) + (data->simulationInfo->realParameter[13] /* armOffset[11] PARAM */))));
  TRACE_POP
}

/*
equation index: 4311
type: SIMPLE_ASSIGN
r_init[10] = 3.0 * exp(k * (theta[10] + armOffset[10]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4311(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4311};
  (data->simulationInfo->realParameter[174] /* r_init[10] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[334] /* theta[10] PARAM */) + (data->simulationInfo->realParameter[12] /* armOffset[10] PARAM */))));
  TRACE_POP
}

/*
equation index: 4312
type: SIMPLE_ASSIGN
r_init[9] = 3.0 * exp(k * (theta[9] + armOffset[9]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4312(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4312};
  (data->simulationInfo->realParameter[173] /* r_init[9] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[333] /* theta[9] PARAM */) + (data->simulationInfo->realParameter[11] /* armOffset[9] PARAM */))));
  TRACE_POP
}

/*
equation index: 4313
type: SIMPLE_ASSIGN
r_init[8] = 3.0 * exp(k * (theta[8] + armOffset[8]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4313(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4313};
  (data->simulationInfo->realParameter[172] /* r_init[8] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[332] /* theta[8] PARAM */) + (data->simulationInfo->realParameter[10] /* armOffset[8] PARAM */))));
  TRACE_POP
}

/*
equation index: 4314
type: SIMPLE_ASSIGN
r_init[7] = 3.0 * exp(k * (theta[7] + armOffset[7]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4314(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4314};
  (data->simulationInfo->realParameter[171] /* r_init[7] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[331] /* theta[7] PARAM */) + (data->simulationInfo->realParameter[9] /* armOffset[7] PARAM */))));
  TRACE_POP
}

/*
equation index: 4315
type: SIMPLE_ASSIGN
r_init[6] = 3.0 * exp(k * (theta[6] + armOffset[6]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4315(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4315};
  (data->simulationInfo->realParameter[170] /* r_init[6] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[330] /* theta[6] PARAM */) + (data->simulationInfo->realParameter[8] /* armOffset[6] PARAM */))));
  TRACE_POP
}

/*
equation index: 4316
type: SIMPLE_ASSIGN
r_init[5] = 3.0 * exp(k * (theta[5] + armOffset[5]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4316(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4316};
  (data->simulationInfo->realParameter[169] /* r_init[5] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[329] /* theta[5] PARAM */) + (data->simulationInfo->realParameter[7] /* armOffset[5] PARAM */))));
  TRACE_POP
}

/*
equation index: 4317
type: SIMPLE_ASSIGN
r_init[4] = 3.0 * exp(k * (theta[4] + armOffset[4]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4317(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4317};
  (data->simulationInfo->realParameter[168] /* r_init[4] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[328] /* theta[4] PARAM */) + (data->simulationInfo->realParameter[6] /* armOffset[4] PARAM */))));
  TRACE_POP
}

/*
equation index: 4318
type: SIMPLE_ASSIGN
r_init[3] = 3.0 * exp(k * (theta[3] + armOffset[3]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4318(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4318};
  (data->simulationInfo->realParameter[167] /* r_init[3] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[327] /* theta[3] PARAM */) + (data->simulationInfo->realParameter[5] /* armOffset[3] PARAM */))));
  TRACE_POP
}

/*
equation index: 4319
type: SIMPLE_ASSIGN
r_init[2] = 3.0 * exp(k * (theta[2] + armOffset[2]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4319(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4319};
  (data->simulationInfo->realParameter[166] /* r_init[2] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[326] /* theta[2] PARAM */) + (data->simulationInfo->realParameter[4] /* armOffset[2] PARAM */))));
  TRACE_POP
}

/*
equation index: 4320
type: SIMPLE_ASSIGN
r_init[1] = 3.0 * exp(k * (theta[1] + armOffset[1]))
*/
OMC_DISABLE_OPT
static void WhirlpoolDiskStars_eqFunction_4320(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,4320};
  (data->simulationInfo->realParameter[165] /* r_init[1] PARAM */) = (3.0) * (exp(((data->simulationInfo->realParameter[164] /* k PARAM */)) * ((data->simulationInfo->realParameter[325] /* theta[1] PARAM */) + (data->simulationInfo->realParameter[3] /* armOffset[1] PARAM */))));
  TRACE_POP
}
OMC_DISABLE_OPT
void WhirlpoolDiskStars_updateBoundParameters_0(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  WhirlpoolDiskStars_eqFunction_4161(data, threadData);
  WhirlpoolDiskStars_eqFunction_4162(data, threadData);
  WhirlpoolDiskStars_eqFunction_4163(data, threadData);
  WhirlpoolDiskStars_eqFunction_4164(data, threadData);
  WhirlpoolDiskStars_eqFunction_4165(data, threadData);
  WhirlpoolDiskStars_eqFunction_4166(data, threadData);
  WhirlpoolDiskStars_eqFunction_4167(data, threadData);
  WhirlpoolDiskStars_eqFunction_4168(data, threadData);
  WhirlpoolDiskStars_eqFunction_4169(data, threadData);
  WhirlpoolDiskStars_eqFunction_4170(data, threadData);
  WhirlpoolDiskStars_eqFunction_4171(data, threadData);
  WhirlpoolDiskStars_eqFunction_4172(data, threadData);
  WhirlpoolDiskStars_eqFunction_4173(data, threadData);
  WhirlpoolDiskStars_eqFunction_4174(data, threadData);
  WhirlpoolDiskStars_eqFunction_4175(data, threadData);
  WhirlpoolDiskStars_eqFunction_4176(data, threadData);
  WhirlpoolDiskStars_eqFunction_4177(data, threadData);
  WhirlpoolDiskStars_eqFunction_4178(data, threadData);
  WhirlpoolDiskStars_eqFunction_4179(data, threadData);
  WhirlpoolDiskStars_eqFunction_4180(data, threadData);
  WhirlpoolDiskStars_eqFunction_4181(data, threadData);
  WhirlpoolDiskStars_eqFunction_4182(data, threadData);
  WhirlpoolDiskStars_eqFunction_4183(data, threadData);
  WhirlpoolDiskStars_eqFunction_4184(data, threadData);
  WhirlpoolDiskStars_eqFunction_4185(data, threadData);
  WhirlpoolDiskStars_eqFunction_4186(data, threadData);
  WhirlpoolDiskStars_eqFunction_4187(data, threadData);
  WhirlpoolDiskStars_eqFunction_4188(data, threadData);
  WhirlpoolDiskStars_eqFunction_4189(data, threadData);
  WhirlpoolDiskStars_eqFunction_4190(data, threadData);
  WhirlpoolDiskStars_eqFunction_4191(data, threadData);
  WhirlpoolDiskStars_eqFunction_4192(data, threadData);
  WhirlpoolDiskStars_eqFunction_4193(data, threadData);
  WhirlpoolDiskStars_eqFunction_4194(data, threadData);
  WhirlpoolDiskStars_eqFunction_4195(data, threadData);
  WhirlpoolDiskStars_eqFunction_4196(data, threadData);
  WhirlpoolDiskStars_eqFunction_4197(data, threadData);
  WhirlpoolDiskStars_eqFunction_4198(data, threadData);
  WhirlpoolDiskStars_eqFunction_4199(data, threadData);
  WhirlpoolDiskStars_eqFunction_4200(data, threadData);
  WhirlpoolDiskStars_eqFunction_4201(data, threadData);
  WhirlpoolDiskStars_eqFunction_4202(data, threadData);
  WhirlpoolDiskStars_eqFunction_4203(data, threadData);
  WhirlpoolDiskStars_eqFunction_4204(data, threadData);
  WhirlpoolDiskStars_eqFunction_4205(data, threadData);
  WhirlpoolDiskStars_eqFunction_4206(data, threadData);
  WhirlpoolDiskStars_eqFunction_4207(data, threadData);
  WhirlpoolDiskStars_eqFunction_4208(data, threadData);
  WhirlpoolDiskStars_eqFunction_4209(data, threadData);
  WhirlpoolDiskStars_eqFunction_4210(data, threadData);
  WhirlpoolDiskStars_eqFunction_4211(data, threadData);
  WhirlpoolDiskStars_eqFunction_4212(data, threadData);
  WhirlpoolDiskStars_eqFunction_4213(data, threadData);
  WhirlpoolDiskStars_eqFunction_4214(data, threadData);
  WhirlpoolDiskStars_eqFunction_4215(data, threadData);
  WhirlpoolDiskStars_eqFunction_4216(data, threadData);
  WhirlpoolDiskStars_eqFunction_4217(data, threadData);
  WhirlpoolDiskStars_eqFunction_4218(data, threadData);
  WhirlpoolDiskStars_eqFunction_4219(data, threadData);
  WhirlpoolDiskStars_eqFunction_4220(data, threadData);
  WhirlpoolDiskStars_eqFunction_4221(data, threadData);
  WhirlpoolDiskStars_eqFunction_4222(data, threadData);
  WhirlpoolDiskStars_eqFunction_4223(data, threadData);
  WhirlpoolDiskStars_eqFunction_4224(data, threadData);
  WhirlpoolDiskStars_eqFunction_4225(data, threadData);
  WhirlpoolDiskStars_eqFunction_4226(data, threadData);
  WhirlpoolDiskStars_eqFunction_4227(data, threadData);
  WhirlpoolDiskStars_eqFunction_4228(data, threadData);
  WhirlpoolDiskStars_eqFunction_4229(data, threadData);
  WhirlpoolDiskStars_eqFunction_4230(data, threadData);
  WhirlpoolDiskStars_eqFunction_4231(data, threadData);
  WhirlpoolDiskStars_eqFunction_4232(data, threadData);
  WhirlpoolDiskStars_eqFunction_4233(data, threadData);
  WhirlpoolDiskStars_eqFunction_4234(data, threadData);
  WhirlpoolDiskStars_eqFunction_4235(data, threadData);
  WhirlpoolDiskStars_eqFunction_4236(data, threadData);
  WhirlpoolDiskStars_eqFunction_4237(data, threadData);
  WhirlpoolDiskStars_eqFunction_4238(data, threadData);
  WhirlpoolDiskStars_eqFunction_4239(data, threadData);
  WhirlpoolDiskStars_eqFunction_4240(data, threadData);
  WhirlpoolDiskStars_eqFunction_4241(data, threadData);
  WhirlpoolDiskStars_eqFunction_4242(data, threadData);
  WhirlpoolDiskStars_eqFunction_4243(data, threadData);
  WhirlpoolDiskStars_eqFunction_4244(data, threadData);
  WhirlpoolDiskStars_eqFunction_4245(data, threadData);
  WhirlpoolDiskStars_eqFunction_4246(data, threadData);
  WhirlpoolDiskStars_eqFunction_4247(data, threadData);
  WhirlpoolDiskStars_eqFunction_4248(data, threadData);
  WhirlpoolDiskStars_eqFunction_4249(data, threadData);
  WhirlpoolDiskStars_eqFunction_4250(data, threadData);
  WhirlpoolDiskStars_eqFunction_4251(data, threadData);
  WhirlpoolDiskStars_eqFunction_4252(data, threadData);
  WhirlpoolDiskStars_eqFunction_4253(data, threadData);
  WhirlpoolDiskStars_eqFunction_4254(data, threadData);
  WhirlpoolDiskStars_eqFunction_4255(data, threadData);
  WhirlpoolDiskStars_eqFunction_4256(data, threadData);
  WhirlpoolDiskStars_eqFunction_4257(data, threadData);
  WhirlpoolDiskStars_eqFunction_4258(data, threadData);
  WhirlpoolDiskStars_eqFunction_4259(data, threadData);
  WhirlpoolDiskStars_eqFunction_4260(data, threadData);
  WhirlpoolDiskStars_eqFunction_4261(data, threadData);
  WhirlpoolDiskStars_eqFunction_4262(data, threadData);
  WhirlpoolDiskStars_eqFunction_4263(data, threadData);
  WhirlpoolDiskStars_eqFunction_4264(data, threadData);
  WhirlpoolDiskStars_eqFunction_4265(data, threadData);
  WhirlpoolDiskStars_eqFunction_4266(data, threadData);
  WhirlpoolDiskStars_eqFunction_4267(data, threadData);
  WhirlpoolDiskStars_eqFunction_4268(data, threadData);
  WhirlpoolDiskStars_eqFunction_4269(data, threadData);
  WhirlpoolDiskStars_eqFunction_4270(data, threadData);
  WhirlpoolDiskStars_eqFunction_4271(data, threadData);
  WhirlpoolDiskStars_eqFunction_4272(data, threadData);
  WhirlpoolDiskStars_eqFunction_4273(data, threadData);
  WhirlpoolDiskStars_eqFunction_4274(data, threadData);
  WhirlpoolDiskStars_eqFunction_4275(data, threadData);
  WhirlpoolDiskStars_eqFunction_4276(data, threadData);
  WhirlpoolDiskStars_eqFunction_4277(data, threadData);
  WhirlpoolDiskStars_eqFunction_4278(data, threadData);
  WhirlpoolDiskStars_eqFunction_4279(data, threadData);
  WhirlpoolDiskStars_eqFunction_4280(data, threadData);
  WhirlpoolDiskStars_eqFunction_4281(data, threadData);
  WhirlpoolDiskStars_eqFunction_4282(data, threadData);
  WhirlpoolDiskStars_eqFunction_4283(data, threadData);
  WhirlpoolDiskStars_eqFunction_4284(data, threadData);
  WhirlpoolDiskStars_eqFunction_4285(data, threadData);
  WhirlpoolDiskStars_eqFunction_4286(data, threadData);
  WhirlpoolDiskStars_eqFunction_4287(data, threadData);
  WhirlpoolDiskStars_eqFunction_4288(data, threadData);
  WhirlpoolDiskStars_eqFunction_4289(data, threadData);
  WhirlpoolDiskStars_eqFunction_4290(data, threadData);
  WhirlpoolDiskStars_eqFunction_4291(data, threadData);
  WhirlpoolDiskStars_eqFunction_4292(data, threadData);
  WhirlpoolDiskStars_eqFunction_4293(data, threadData);
  WhirlpoolDiskStars_eqFunction_4294(data, threadData);
  WhirlpoolDiskStars_eqFunction_4295(data, threadData);
  WhirlpoolDiskStars_eqFunction_4296(data, threadData);
  WhirlpoolDiskStars_eqFunction_4297(data, threadData);
  WhirlpoolDiskStars_eqFunction_4298(data, threadData);
  WhirlpoolDiskStars_eqFunction_4299(data, threadData);
  WhirlpoolDiskStars_eqFunction_4300(data, threadData);
  WhirlpoolDiskStars_eqFunction_4301(data, threadData);
  WhirlpoolDiskStars_eqFunction_4302(data, threadData);
  WhirlpoolDiskStars_eqFunction_4303(data, threadData);
  WhirlpoolDiskStars_eqFunction_4304(data, threadData);
  WhirlpoolDiskStars_eqFunction_4305(data, threadData);
  WhirlpoolDiskStars_eqFunction_4306(data, threadData);
  WhirlpoolDiskStars_eqFunction_4307(data, threadData);
  WhirlpoolDiskStars_eqFunction_4308(data, threadData);
  WhirlpoolDiskStars_eqFunction_4309(data, threadData);
  WhirlpoolDiskStars_eqFunction_4310(data, threadData);
  WhirlpoolDiskStars_eqFunction_4311(data, threadData);
  WhirlpoolDiskStars_eqFunction_4312(data, threadData);
  WhirlpoolDiskStars_eqFunction_4313(data, threadData);
  WhirlpoolDiskStars_eqFunction_4314(data, threadData);
  WhirlpoolDiskStars_eqFunction_4315(data, threadData);
  WhirlpoolDiskStars_eqFunction_4316(data, threadData);
  WhirlpoolDiskStars_eqFunction_4317(data, threadData);
  WhirlpoolDiskStars_eqFunction_4318(data, threadData);
  WhirlpoolDiskStars_eqFunction_4319(data, threadData);
  WhirlpoolDiskStars_eqFunction_4320(data, threadData);
  TRACE_POP
}
OMC_DISABLE_OPT
int WhirlpoolDiskStars_updateBoundParameters(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  (data->simulationInfo->integerParameter[0] /* N PARAM */) = ((modelica_integer) 160);
  data->modelData->integerParameterData[0].time_unvarying = 1;
  WhirlpoolDiskStars_updateBoundParameters_0(data, threadData);
  TRACE_POP
  return 0;
}

#if defined(__cplusplus)
}
#endif

