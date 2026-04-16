#include "Pendulum_model.h"
#if defined(__cplusplus)
extern "C" {
#endif

/*
equation index: 270
type: SIMPLE_ASSIGN
rev.cylinder.r_shape[2] = (-rev.e[2]) * 0.5 * rev.cylinderLength
*/
void Pendulum_eqFunction_270(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,270};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[95]] /* rev.cylinder.r_shape[2] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[52]] /* rev.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 270;
}

/*
equation index: 271
type: SIMPLE_ASSIGN
rev.cylinder.r_shape[3] = (-rev.e[3]) * 0.5 * rev.cylinderLength
*/
void Pendulum_eqFunction_271(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,271};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[96]] /* rev.cylinder.r_shape[3] variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */))) * ((0.5) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[52]] /* rev.cylinderLength PARAM */)));
  threadData->lastEquationSolved = 271;
}

/*
equation index: 272
type: SIMPLE_ASSIGN
rev.cylinder.widthDirection[1] = 0.0
*/
void Pendulum_eqFunction_272(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,272};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[97]] /* rev.cylinder.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 272;
}

/*
equation index: 273
type: SIMPLE_ASSIGN
rev.cylinder.widthDirection[2] = 1.0
*/
void Pendulum_eqFunction_273(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,273};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[98]] /* rev.cylinder.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 273;
}

/*
equation index: 274
type: SIMPLE_ASSIGN
rev.cylinder.widthDirection[3] = 0.0
*/
void Pendulum_eqFunction_274(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,274};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[99]] /* rev.cylinder.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 274;
}

/*
equation index: 275
type: SIMPLE_ASSIGN
rev.cylinder.color[3] = 0.0
*/
void Pendulum_eqFunction_275(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,275};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[92]] /* rev.cylinder.color[3] variable */) = 0.0;
  threadData->lastEquationSolved = 275;
}

/*
equation index: 276
type: SIMPLE_ASSIGN
rev.cylinder.color[2] = 0.0
*/
void Pendulum_eqFunction_276(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,276};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[91]] /* rev.cylinder.color[2] variable */) = 0.0;
  threadData->lastEquationSolved = 276;
}

/*
equation index: 277
type: SIMPLE_ASSIGN
rev.cylinder.color[1] = 255.0
*/
void Pendulum_eqFunction_277(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,277};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[90]] /* rev.cylinder.color[1] variable */) = 255.0;
  threadData->lastEquationSolved = 277;
}

/*
equation index: 278
type: SIMPLE_ASSIGN
rev.cylinderColor[1] = 255
*/
void Pendulum_eqFunction_278(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,278};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[3]] /* rev.cylinderColor[1] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 278;
}

/*
equation index: 279
type: SIMPLE_ASSIGN
rev.cylinderColor[2] = 0.0
*/
void Pendulum_eqFunction_279(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,279};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[4]] /* rev.cylinderColor[2] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 279;
}

/*
equation index: 280
type: SIMPLE_ASSIGN
rev.cylinderColor[3] = 0.0
*/
void Pendulum_eqFunction_280(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,280};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[5]] /* rev.cylinderColor[3] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 280;
}

/*
equation index: 281
type: SIMPLE_ASSIGN
body.cylinder.r_shape[1] = 0.0
*/
void Pendulum_eqFunction_281(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,281};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* body.cylinder.r_shape[1] variable */) = 0.0;
  threadData->lastEquationSolved = 281;
}

/*
equation index: 282
type: SIMPLE_ASSIGN
body.cylinder.r_shape[2] = 0.0
*/
void Pendulum_eqFunction_282(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,282};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* body.cylinder.r_shape[2] variable */) = 0.0;
  threadData->lastEquationSolved = 282;
}

/*
equation index: 283
type: SIMPLE_ASSIGN
body.cylinder.r_shape[3] = 0.0
*/
void Pendulum_eqFunction_283(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,283};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* body.cylinder.r_shape[3] variable */) = 0.0;
  threadData->lastEquationSolved = 283;
}

/*
equation index: 284
type: SIMPLE_ASSIGN
body.cylinder.widthDirection[1] = 0.0
*/
void Pendulum_eqFunction_284(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,284};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[29]] /* body.cylinder.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 284;
}

/*
equation index: 285
type: SIMPLE_ASSIGN
body.cylinder.widthDirection[2] = 1.0
*/
void Pendulum_eqFunction_285(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,285};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[30]] /* body.cylinder.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 285;
}

/*
equation index: 286
type: SIMPLE_ASSIGN
body.cylinder.widthDirection[3] = 0.0
*/
void Pendulum_eqFunction_286(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,286};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[31]] /* body.cylinder.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 286;
}

/*
equation index: 287
type: SIMPLE_ASSIGN
body.sphere.r_shape[1] = body.r_CM[1] - 0.05555555555555555
*/
void Pendulum_eqFunction_287(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,287};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[67]] /* body.sphere.r_shape[1] variable */) = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* body.r_CM[1] PARAM */) - 0.05555555555555555;
  threadData->lastEquationSolved = 287;
}

/*
equation index: 288
type: SIMPLE_ASSIGN
body.sphere.lengthDirection[1] = 1.0
*/
void Pendulum_eqFunction_288(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,288};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[64]] /* body.sphere.lengthDirection[1] variable */) = 1.0;
  threadData->lastEquationSolved = 288;
}

/*
equation index: 289
type: SIMPLE_ASSIGN
body.sphere.lengthDirection[2] = 0.0
*/
void Pendulum_eqFunction_289(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,289};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[65]] /* body.sphere.lengthDirection[2] variable */) = 0.0;
  threadData->lastEquationSolved = 289;
}

/*
equation index: 290
type: SIMPLE_ASSIGN
body.sphere.lengthDirection[3] = 0.0
*/
void Pendulum_eqFunction_290(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,290};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[66]] /* body.sphere.lengthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 290;
}

/*
equation index: 291
type: SIMPLE_ASSIGN
body.sphere.widthDirection[1] = 0.0
*/
void Pendulum_eqFunction_291(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,291};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[69]] /* body.sphere.widthDirection[1] variable */) = 0.0;
  threadData->lastEquationSolved = 291;
}

/*
equation index: 292
type: SIMPLE_ASSIGN
body.sphere.widthDirection[2] = 1.0
*/
void Pendulum_eqFunction_292(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,292};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[70]] /* body.sphere.widthDirection[2] variable */) = 1.0;
  threadData->lastEquationSolved = 292;
}

/*
equation index: 293
type: SIMPLE_ASSIGN
body.sphere.widthDirection[3] = 0.0
*/
void Pendulum_eqFunction_293(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,293};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[71]] /* body.sphere.widthDirection[3] variable */) = 0.0;
  threadData->lastEquationSolved = 293;
}

/*
equation index: 294
type: SIMPLE_ASSIGN
body.sphere.color[3] = 255.0
*/
void Pendulum_eqFunction_294(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,294};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[60]] /* body.sphere.color[3] variable */) = 255.0;
  threadData->lastEquationSolved = 294;
}

/*
equation index: 295
type: SIMPLE_ASSIGN
body.sphere.color[2] = 128.0
*/
void Pendulum_eqFunction_295(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,295};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[59]] /* body.sphere.color[2] variable */) = 128.0;
  threadData->lastEquationSolved = 295;
}

/*
equation index: 296
type: SIMPLE_ASSIGN
body.sphere.color[1] = 0.0
*/
void Pendulum_eqFunction_296(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,296};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[58]] /* body.sphere.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 296;
}

/*
equation index: 297
type: SIMPLE_ASSIGN
body.cylinder.color[3] = 255.0
*/
void Pendulum_eqFunction_297(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,297};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* body.cylinder.color[3] variable */) = 255.0;
  threadData->lastEquationSolved = 297;
}

/*
equation index: 298
type: SIMPLE_ASSIGN
body.cylinder.color[2] = 128.0
*/
void Pendulum_eqFunction_298(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,298};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* body.cylinder.color[2] variable */) = 128.0;
  threadData->lastEquationSolved = 298;
}

/*
equation index: 299
type: SIMPLE_ASSIGN
body.cylinder.color[1] = 0.0
*/
void Pendulum_eqFunction_299(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,299};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* body.cylinder.color[1] variable */) = 0.0;
  threadData->lastEquationSolved = 299;
}

/*
equation index: 300
type: SIMPLE_ASSIGN
body.sphereColor[1] = 0.0
*/
void Pendulum_eqFunction_300(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,300};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[0]] /* body.sphereColor[1] DISCRETE */) = 0.0;
  threadData->lastEquationSolved = 300;
}

/*
equation index: 301
type: SIMPLE_ASSIGN
body.sphereColor[2] = 128
*/
void Pendulum_eqFunction_301(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,301};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[1]] /* body.sphereColor[2] DISCRETE */) = ((modelica_integer) 128);
  threadData->lastEquationSolved = 301;
}

/*
equation index: 302
type: SIMPLE_ASSIGN
body.sphereColor[3] = 255
*/
void Pendulum_eqFunction_302(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,302};
  (data->localData[0]->integerVars[data->simulationInfo->integerVarsIndex[2]] /* body.sphereColor[3] DISCRETE */) = ((modelica_integer) 255);
  threadData->lastEquationSolved = 302;
}

/*
equation index: 303
type: SIMPLE_ASSIGN
$START.body.Q[1] = body.Q_start[1]
*/
void Pendulum_eqFunction_303(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,303};
  ((modelica_real *)((data->modelData->realVarsData[14] /* body.Q[1] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[15]] /* body.Q_start[1] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* body.Q[1] variable */) = ((modelica_real *)((data->modelData->realVarsData[14] /* body.Q[1] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[14].info /* body.Q[1] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* body.Q[1] variable */));
  threadData->lastEquationSolved = 303;
}

/*
equation index: 304
type: SIMPLE_ASSIGN
body.Q[1] = 0.0
*/
void Pendulum_eqFunction_304(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,304};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[14]] /* body.Q[1] variable */) = 0.0;
  threadData->lastEquationSolved = 304;
}

/*
equation index: 305
type: SIMPLE_ASSIGN
$START.body.Q[2] = body.Q_start[2]
*/
void Pendulum_eqFunction_305(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,305};
  ((modelica_real *)((data->modelData->realVarsData[15] /* body.Q[2] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[16]] /* body.Q_start[2] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* body.Q[2] variable */) = ((modelica_real *)((data->modelData->realVarsData[15] /* body.Q[2] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[15].info /* body.Q[2] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* body.Q[2] variable */));
  threadData->lastEquationSolved = 305;
}

/*
equation index: 306
type: SIMPLE_ASSIGN
body.Q[2] = 0.0
*/
void Pendulum_eqFunction_306(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,306};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* body.Q[2] variable */) = 0.0;
  threadData->lastEquationSolved = 306;
}

/*
equation index: 307
type: SIMPLE_ASSIGN
$START.body.Q[3] = body.Q_start[3]
*/
void Pendulum_eqFunction_307(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,307};
  ((modelica_real *)((data->modelData->realVarsData[16] /* body.Q[3] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[17]] /* body.Q_start[3] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* body.Q[3] variable */) = ((modelica_real *)((data->modelData->realVarsData[16] /* body.Q[3] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[16].info /* body.Q[3] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* body.Q[3] variable */));
  threadData->lastEquationSolved = 307;
}

/*
equation index: 308
type: SIMPLE_ASSIGN
body.Q[3] = 0.0
*/
void Pendulum_eqFunction_308(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,308};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* body.Q[3] variable */) = 0.0;
  threadData->lastEquationSolved = 308;
}

/*
equation index: 309
type: SIMPLE_ASSIGN
$START.body.Q[4] = body.Q_start[4]
*/
void Pendulum_eqFunction_309(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,309};
  ((modelica_real *)((data->modelData->realVarsData[17] /* body.Q[4] variable */).attribute .start.data))[0] = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[18]] /* body.Q_start[4] PARAM */);
    (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* body.Q[4] variable */) = ((modelica_real *)((data->modelData->realVarsData[17] /* body.Q[4] variable */).attribute .start.data))[0];
    infoStreamPrint(OMC_LOG_INIT_V, 0, "updated start value: %s(start=%g)", data->modelData->realVarsData[17].info /* body.Q[4] */.name, (modelica_real) (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* body.Q[4] variable */));
  threadData->lastEquationSolved = 309;
}

/*
equation index: 310
type: SIMPLE_ASSIGN
body.Q[4] = 1.0
*/
void Pendulum_eqFunction_310(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,310};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* body.Q[4] variable */) = 1.0;
  threadData->lastEquationSolved = 310;
}

/*
equation index: 311
type: SIMPLE_ASSIGN
body.phi[1] = 0.0
*/
void Pendulum_eqFunction_311(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,311};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[49]] /* body.phi[1] variable */) = 0.0;
  threadData->lastEquationSolved = 311;
}

/*
equation index: 312
type: SIMPLE_ASSIGN
body.phi[2] = 0.0
*/
void Pendulum_eqFunction_312(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,312};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[50]] /* body.phi[2] variable */) = 0.0;
  threadData->lastEquationSolved = 312;
}

/*
equation index: 313
type: SIMPLE_ASSIGN
body.phi[3] = 0.0
*/
void Pendulum_eqFunction_313(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,313};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[51]] /* body.phi[3] variable */) = 0.0;
  threadData->lastEquationSolved = 313;
}

/*
equation index: 314
type: SIMPLE_ASSIGN
body.phi_d[1] = 0.0
*/
void Pendulum_eqFunction_314(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,314};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[52]] /* body.phi_d[1] variable */) = 0.0;
  threadData->lastEquationSolved = 314;
}

/*
equation index: 315
type: SIMPLE_ASSIGN
body.phi_d[2] = 0.0
*/
void Pendulum_eqFunction_315(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,315};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[53]] /* body.phi_d[2] variable */) = 0.0;
  threadData->lastEquationSolved = 315;
}

/*
equation index: 316
type: SIMPLE_ASSIGN
body.phi_d[3] = 0.0
*/
void Pendulum_eqFunction_316(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,316};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[54]] /* body.phi_d[3] variable */) = 0.0;
  threadData->lastEquationSolved = 316;
}

/*
equation index: 317
type: SIMPLE_ASSIGN
body.phi_dd[1] = 0.0
*/
void Pendulum_eqFunction_317(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,317};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[55]] /* body.phi_dd[1] variable */) = 0.0;
  threadData->lastEquationSolved = 317;
}

/*
equation index: 318
type: SIMPLE_ASSIGN
body.phi_dd[2] = 0.0
*/
void Pendulum_eqFunction_318(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,318};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[56]] /* body.phi_dd[2] variable */) = 0.0;
  threadData->lastEquationSolved = 318;
}

/*
equation index: 319
type: SIMPLE_ASSIGN
body.phi_dd[3] = 0.0
*/
void Pendulum_eqFunction_319(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,319};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[57]] /* body.phi_dd[3] variable */) = 0.0;
  threadData->lastEquationSolved = 319;
}

/*
equation index: 320
type: SIMPLE_ASSIGN
world.x_arrowLine.extra = 0.0
*/
void Pendulum_eqFunction_320(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,320};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[211]] /* world.x_arrowLine.extra variable */) = 0.0;
  threadData->lastEquationSolved = 320;
}

/*
equation index: 321
type: SIMPLE_ASSIGN
world.x_arrowLine.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_321(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,321};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[221]] /* world.x_arrowLine.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 321;
}

/*
equation index: 322
type: SIMPLE_ASSIGN
world.x_arrowHead.extra = 0.0
*/
void Pendulum_eqFunction_322(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,322};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[186]] /* world.x_arrowHead.extra variable */) = 0.0;
  threadData->lastEquationSolved = 322;
}

/*
equation index: 323
type: SIMPLE_ASSIGN
world.x_arrowHead.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_323(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,323};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[195]] /* world.x_arrowHead.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 323;
}

/*
equation index: 324
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].extra = 0.0
*/
void Pendulum_eqFunction_324(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,324};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[249]] /* world.x_label.cylinders[1].extra variable */) = 0.0;
  threadData->lastEquationSolved = 324;
}

/*
equation index: 325
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].length = 1.4142135623730951 * abs(world.scaledLabel)
*/
void Pendulum_eqFunction_325(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,325};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[252]] /* world.x_label.cylinders[2].length variable */) = (1.4142135623730951) * (fabs((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)));
  threadData->lastEquationSolved = 325;
}

/*
equation index: 326
type: SIMPLE_ASSIGN
world.x_label.cylinders[1].length = 1.4142135623730951 * abs(world.scaledLabel)
*/
void Pendulum_eqFunction_326(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,326};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[251]] /* world.x_label.cylinders[1].length variable */) = (1.4142135623730951) * (fabs((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)));
  threadData->lastEquationSolved = 326;
}

/*
equation index: 327
type: SIMPLE_ASSIGN
world.x_label.r_abs[3] = 0.0
*/
void Pendulum_eqFunction_327(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,327};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[288]] /* world.x_label.r_abs[3] variable */) = 0.0;
  threadData->lastEquationSolved = 327;
}

/*
equation index: 328
type: SIMPLE_ASSIGN
world.x_label.r_abs[2] = 0.0
*/
void Pendulum_eqFunction_328(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,328};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[287]] /* world.x_label.r_abs[2] variable */) = 0.0;
  threadData->lastEquationSolved = 328;
}

/*
equation index: 329
type: SIMPLE_ASSIGN
world.x_label.r[1] = 0.0
*/
void Pendulum_eqFunction_329(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,329};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[284]] /* world.x_label.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 329;
}

/*
equation index: 330
type: SIMPLE_ASSIGN
world.x_label.r[2] = 0.0
*/
void Pendulum_eqFunction_330(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,330};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[285]] /* world.x_label.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 330;
}

/*
equation index: 331
type: SIMPLE_ASSIGN
world.x_label.r[3] = 0.0
*/
void Pendulum_eqFunction_331(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,331};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[286]] /* world.x_label.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 331;
}

/*
equation index: 332
type: SIMPLE_ASSIGN
world.x_label.r_lines[2] = 0.0
*/
void Pendulum_eqFunction_332(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,332};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[289]] /* world.x_label.r_lines[2] variable */) = 0.0;
  threadData->lastEquationSolved = 332;
}

/*
equation index: 333
type: SIMPLE_ASSIGN
world.x_label.r_lines[3] = 0.0
*/
void Pendulum_eqFunction_333(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,333};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[290]] /* world.x_label.r_lines[3] variable */) = 0.0;
  threadData->lastEquationSolved = 333;
}

/*
equation index: 334
type: SIMPLE_ASSIGN
world.x_label.n_x[1] = 1.0
*/
void Pendulum_eqFunction_334(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,334};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[278]] /* world.x_label.n_x[1] variable */) = 1.0;
  threadData->lastEquationSolved = 334;
}

/*
equation index: 335
type: SIMPLE_ASSIGN
world.x_label.n_x[2] = 0.0
*/
void Pendulum_eqFunction_335(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,335};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[279]] /* world.x_label.n_x[2] variable */) = 0.0;
  threadData->lastEquationSolved = 335;
}

/*
equation index: 336
type: SIMPLE_ASSIGN
world.x_label.n_x[3] = 0.0
*/
void Pendulum_eqFunction_336(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,336};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[280]] /* world.x_label.n_x[3] variable */) = 0.0;
  threadData->lastEquationSolved = 336;
}

/*
equation index: 337
type: SIMPLE_ASSIGN
world.x_label.n_y[1] = 0.0
*/
void Pendulum_eqFunction_337(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,337};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[281]] /* world.x_label.n_y[1] variable */) = 0.0;
  threadData->lastEquationSolved = 337;
}

/*
equation index: 338
type: SIMPLE_ASSIGN
world.x_label.n_y[2] = 1.0
*/
void Pendulum_eqFunction_338(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,338};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[282]] /* world.x_label.n_y[2] variable */) = 1.0;
  threadData->lastEquationSolved = 338;
}

/*
equation index: 339
type: SIMPLE_ASSIGN
world.x_label.n_y[3] = 0.0
*/
void Pendulum_eqFunction_339(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,339};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[283]] /* world.x_label.n_y[3] variable */) = 0.0;
  threadData->lastEquationSolved = 339;
}

/*
equation index: 340
type: SIMPLE_ASSIGN
world.x_label.lines[1,1,1] = 0.0
*/
void Pendulum_eqFunction_340(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,340};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[274]] /* world.x_label.lines[1,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 340;
}

/*
equation index: 341
type: SIMPLE_ASSIGN
world.x_label.lines[1,1,2] = 0.0
*/
void Pendulum_eqFunction_341(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,341};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[275]] /* world.x_label.lines[1,1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 341;
}

/*
equation index: 342
type: SIMPLE_ASSIGN
world.x_label.lines[2,1,1] = 0.0
*/
void Pendulum_eqFunction_342(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,342};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[276]] /* world.x_label.lines[2,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 342;
}

/*
equation index: 343
type: SIMPLE_ASSIGN
world.x_label.lines[2,2,2] = 0.0
*/
void Pendulum_eqFunction_343(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,343};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[277]] /* world.x_label.lines[2,2,2] variable */) = 0.0;
  threadData->lastEquationSolved = 343;
}

/*
equation index: 344
type: SIMPLE_ASSIGN
world.x_label.R.T[1,1] = 1.0
*/
void Pendulum_eqFunction_344(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,344};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[225]] /* world.x_label.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 344;
}

/*
equation index: 345
type: SIMPLE_ASSIGN
world.x_label.R.T[1,2] = 0.0
*/
void Pendulum_eqFunction_345(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,345};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[226]] /* world.x_label.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 345;
}

/*
equation index: 346
type: SIMPLE_ASSIGN
world.x_label.R.T[1,3] = 0.0
*/
void Pendulum_eqFunction_346(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,346};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[227]] /* world.x_label.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 346;
}

/*
equation index: 347
type: SIMPLE_ASSIGN
world.x_label.R.T[2,1] = 0.0
*/
void Pendulum_eqFunction_347(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,347};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[228]] /* world.x_label.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 347;
}

/*
equation index: 348
type: SIMPLE_ASSIGN
world.x_label.R.T[2,2] = 1.0
*/
void Pendulum_eqFunction_348(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,348};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[229]] /* world.x_label.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 348;
}

/*
equation index: 349
type: SIMPLE_ASSIGN
world.x_label.R.T[2,3] = 0.0
*/
void Pendulum_eqFunction_349(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,349};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[230]] /* world.x_label.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 349;
}

/*
equation index: 350
type: SIMPLE_ASSIGN
world.x_label.R.T[3,1] = 0.0
*/
void Pendulum_eqFunction_350(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,350};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[231]] /* world.x_label.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 350;
}

/*
equation index: 351
type: SIMPLE_ASSIGN
world.x_label.R.T[3,2] = 0.0
*/
void Pendulum_eqFunction_351(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,351};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[232]] /* world.x_label.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 351;
}

/*
equation index: 352
type: SIMPLE_ASSIGN
world.x_label.R.T[3,3] = 1.0
*/
void Pendulum_eqFunction_352(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,352};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[233]] /* world.x_label.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 352;
}

/*
equation index: 353
type: SIMPLE_ASSIGN
world.x_label.cylinders[2].extra = 0.0
*/
void Pendulum_eqFunction_353(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,353};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[250]] /* world.x_label.cylinders[2].extra variable */) = 0.0;
  threadData->lastEquationSolved = 353;
}

/*
equation index: 354
type: SIMPLE_ASSIGN
world.x_label.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_354(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,354};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[291]] /* world.x_label.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 354;
}

/*
equation index: 355
type: SIMPLE_ASSIGN
world.y_arrowLine.extra = 0.0
*/
void Pendulum_eqFunction_355(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,355};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[332]] /* world.y_arrowLine.extra variable */) = 0.0;
  threadData->lastEquationSolved = 355;
}

/*
equation index: 356
type: SIMPLE_ASSIGN
world.y_arrowLine.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_356(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,356};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[342]] /* world.y_arrowLine.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 356;
}

/*
equation index: 357
type: SIMPLE_ASSIGN
world.y_arrowHead.extra = 0.0
*/
void Pendulum_eqFunction_357(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,357};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[307]] /* world.y_arrowHead.extra variable */) = 0.0;
  threadData->lastEquationSolved = 357;
}

/*
equation index: 358
type: SIMPLE_ASSIGN
world.y_arrowHead.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_358(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,358};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[316]] /* world.y_arrowHead.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 358;
}

/*
equation index: 359
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].extra = 0.0
*/
void Pendulum_eqFunction_359(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,359};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[370]] /* world.y_label.cylinders[1].extra variable */) = 0.0;
  threadData->lastEquationSolved = 359;
}

/*
equation index: 360
type: SIMPLE_ASSIGN
world.y_label.lines[1,2,2] = world.scaledLabel * 1.5
*/
void Pendulum_eqFunction_360(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,360};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[397]] /* world.y_label.lines[1,2,2] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)) * (1.5);
  threadData->lastEquationSolved = 360;
}

/*
equation index: 361
type: SIMPLE_ASSIGN
world.y_label.cylinders[1].length = sqrt(world.scaledLabel ^ 2.0 + world.y_label.lines[1,2,2] ^ 2.0)
*/
void Pendulum_eqFunction_361(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,361};
  modelica_real tmp0;
  modelica_real tmp1;
  modelica_real tmp2;
  tmp0 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */);
  tmp1 = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[397]] /* world.y_label.lines[1,2,2] variable */);
  tmp2 = (tmp0 * tmp0) + (tmp1 * tmp1);
  if(!(tmp2 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(world.scaledLabel ^ 2.0 + world.y_label.lines[1,2,2] ^ 2.0) was %g should be >= 0", tmp2);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[372]] /* world.y_label.cylinders[1].length variable */) = sqrt(tmp2);
  threadData->lastEquationSolved = 361;
}

/*
equation index: 362
type: SIMPLE_ASSIGN
world.y_label.lines[2,1,2] = world.scaledLabel * 1.5
*/
void Pendulum_eqFunction_362(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,362};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[399]] /* world.y_label.lines[2,1,2] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)) * (1.5);
  threadData->lastEquationSolved = 362;
}

/*
equation index: 363
type: SIMPLE_ASSIGN
world.y_label.lines[2,2,1] = world.scaledLabel * 0.5
*/
void Pendulum_eqFunction_363(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,363};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[400]] /* world.y_label.lines[2,2,1] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)) * (0.5);
  threadData->lastEquationSolved = 363;
}

/*
equation index: 364
type: SIMPLE_ASSIGN
world.y_label.lines[2,2,2] = world.scaledLabel * 0.75
*/
void Pendulum_eqFunction_364(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,364};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[401]] /* world.y_label.lines[2,2,2] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)) * (0.75);
  threadData->lastEquationSolved = 364;
}

/*
equation index: 365
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].length = sqrt(world.y_label.lines[2,2,1] ^ 2.0 + (world.y_label.lines[2,2,2] - world.y_label.lines[2,1,2]) ^ 2.0)
*/
void Pendulum_eqFunction_365(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,365};
  modelica_real tmp3;
  modelica_real tmp4;
  modelica_real tmp5;
  tmp3 = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[400]] /* world.y_label.lines[2,2,1] variable */);
  tmp4 = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[401]] /* world.y_label.lines[2,2,2] variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[399]] /* world.y_label.lines[2,1,2] variable */);
  tmp5 = (tmp3 * tmp3) + (tmp4 * tmp4);
  if(!(tmp5 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(world.y_label.lines[2,2,1] ^ 2.0 + (world.y_label.lines[2,2,2] - world.y_label.lines[2,1,2]) ^ 2.0) was %g should be >= 0", tmp5);
    }
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[373]] /* world.y_label.cylinders[2].length variable */) = sqrt(tmp5);
  threadData->lastEquationSolved = 365;
}
extern void Pendulum_eqFunction_573(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_575(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_578(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_581(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_576(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_574(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_580(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_577(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_579(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_582(DATA *data, threadData_t *threadData);


/*
equation index: 376
type: SIMPLE_ASSIGN
world.y_label.r_abs[3] = 0.0
*/
void Pendulum_eqFunction_376(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,376};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[412]] /* world.y_label.r_abs[3] variable */) = 0.0;
  threadData->lastEquationSolved = 376;
}

/*
equation index: 377
type: SIMPLE_ASSIGN
world.y_label.r_abs[1] = 0.0
*/
void Pendulum_eqFunction_377(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,377};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[411]] /* world.y_label.r_abs[1] variable */) = 0.0;
  threadData->lastEquationSolved = 377;
}

/*
equation index: 378
type: SIMPLE_ASSIGN
world.y_label.r[1] = 0.0
*/
void Pendulum_eqFunction_378(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,378};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[408]] /* world.y_label.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 378;
}

/*
equation index: 379
type: SIMPLE_ASSIGN
world.y_label.r[2] = 0.0
*/
void Pendulum_eqFunction_379(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,379};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[409]] /* world.y_label.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 379;
}

/*
equation index: 380
type: SIMPLE_ASSIGN
world.y_label.r[3] = 0.0
*/
void Pendulum_eqFunction_380(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,380};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[410]] /* world.y_label.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 380;
}

/*
equation index: 381
type: SIMPLE_ASSIGN
world.y_label.r_lines[1] = 0.0
*/
void Pendulum_eqFunction_381(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,381};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[413]] /* world.y_label.r_lines[1] variable */) = 0.0;
  threadData->lastEquationSolved = 381;
}

/*
equation index: 382
type: SIMPLE_ASSIGN
world.y_label.r_lines[3] = 0.0
*/
void Pendulum_eqFunction_382(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,382};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[414]] /* world.y_label.r_lines[3] variable */) = 0.0;
  threadData->lastEquationSolved = 382;
}

/*
equation index: 383
type: SIMPLE_ASSIGN
world.y_label.n_x[1] = 0.0
*/
void Pendulum_eqFunction_383(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,383};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[402]] /* world.y_label.n_x[1] variable */) = 0.0;
  threadData->lastEquationSolved = 383;
}

/*
equation index: 384
type: SIMPLE_ASSIGN
world.y_label.n_x[2] = 1.0
*/
void Pendulum_eqFunction_384(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,384};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[403]] /* world.y_label.n_x[2] variable */) = 1.0;
  threadData->lastEquationSolved = 384;
}

/*
equation index: 385
type: SIMPLE_ASSIGN
world.y_label.n_x[3] = 0.0
*/
void Pendulum_eqFunction_385(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,385};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[404]] /* world.y_label.n_x[3] variable */) = 0.0;
  threadData->lastEquationSolved = 385;
}

/*
equation index: 386
type: SIMPLE_ASSIGN
world.y_label.n_y[1] = -1.0
*/
void Pendulum_eqFunction_386(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,386};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[405]] /* world.y_label.n_y[1] variable */) = -1.0;
  threadData->lastEquationSolved = 386;
}

/*
equation index: 387
type: SIMPLE_ASSIGN
world.y_label.n_y[2] = 0.0
*/
void Pendulum_eqFunction_387(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,387};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[406]] /* world.y_label.n_y[2] variable */) = 0.0;
  threadData->lastEquationSolved = 387;
}

/*
equation index: 388
type: SIMPLE_ASSIGN
world.y_label.n_y[3] = 0.0
*/
void Pendulum_eqFunction_388(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,388};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[407]] /* world.y_label.n_y[3] variable */) = 0.0;
  threadData->lastEquationSolved = 388;
}

/*
equation index: 389
type: SIMPLE_ASSIGN
world.y_label.lines[1,1,1] = 0.0
*/
void Pendulum_eqFunction_389(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,389};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[395]] /* world.y_label.lines[1,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 389;
}

/*
equation index: 390
type: SIMPLE_ASSIGN
world.y_label.lines[1,1,2] = 0.0
*/
void Pendulum_eqFunction_390(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,390};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[396]] /* world.y_label.lines[1,1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 390;
}

/*
equation index: 391
type: SIMPLE_ASSIGN
world.y_label.lines[2,1,1] = 0.0
*/
void Pendulum_eqFunction_391(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,391};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[398]] /* world.y_label.lines[2,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 391;
}

/*
equation index: 392
type: SIMPLE_ASSIGN
world.y_label.R.T[1,1] = 1.0
*/
void Pendulum_eqFunction_392(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,392};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[346]] /* world.y_label.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 392;
}

/*
equation index: 393
type: SIMPLE_ASSIGN
world.y_label.R.T[1,2] = 0.0
*/
void Pendulum_eqFunction_393(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,393};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[347]] /* world.y_label.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 393;
}

/*
equation index: 394
type: SIMPLE_ASSIGN
world.y_label.R.T[1,3] = 0.0
*/
void Pendulum_eqFunction_394(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,394};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[348]] /* world.y_label.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 394;
}

/*
equation index: 395
type: SIMPLE_ASSIGN
world.y_label.R.T[2,1] = 0.0
*/
void Pendulum_eqFunction_395(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,395};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[349]] /* world.y_label.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 395;
}

/*
equation index: 396
type: SIMPLE_ASSIGN
world.y_label.R.T[2,2] = 1.0
*/
void Pendulum_eqFunction_396(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,396};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[350]] /* world.y_label.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 396;
}

/*
equation index: 397
type: SIMPLE_ASSIGN
world.y_label.R.T[2,3] = 0.0
*/
void Pendulum_eqFunction_397(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,397};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[351]] /* world.y_label.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 397;
}

/*
equation index: 398
type: SIMPLE_ASSIGN
world.y_label.R.T[3,1] = 0.0
*/
void Pendulum_eqFunction_398(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,398};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[352]] /* world.y_label.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 398;
}

/*
equation index: 399
type: SIMPLE_ASSIGN
world.y_label.R.T[3,2] = 0.0
*/
void Pendulum_eqFunction_399(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,399};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[353]] /* world.y_label.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 399;
}

/*
equation index: 400
type: SIMPLE_ASSIGN
world.y_label.R.T[3,3] = 1.0
*/
void Pendulum_eqFunction_400(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,400};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[354]] /* world.y_label.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 400;
}

/*
equation index: 401
type: SIMPLE_ASSIGN
world.y_label.cylinders[2].extra = 0.0
*/
void Pendulum_eqFunction_401(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,401};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[371]] /* world.y_label.cylinders[2].extra variable */) = 0.0;
  threadData->lastEquationSolved = 401;
}

/*
equation index: 402
type: SIMPLE_ASSIGN
world.y_label.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_402(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,402};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[415]] /* world.y_label.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 402;
}

/*
equation index: 403
type: SIMPLE_ASSIGN
world.z_arrowLine.extra = 0.0
*/
void Pendulum_eqFunction_403(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,403};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[456]] /* world.z_arrowLine.extra variable */) = 0.0;
  threadData->lastEquationSolved = 403;
}

/*
equation index: 404
type: SIMPLE_ASSIGN
world.z_arrowLine.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_404(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,404};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[466]] /* world.z_arrowLine.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 404;
}

/*
equation index: 405
type: SIMPLE_ASSIGN
world.z_arrowHead.extra = 0.0
*/
void Pendulum_eqFunction_405(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,405};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[431]] /* world.z_arrowHead.extra variable */) = 0.0;
  threadData->lastEquationSolved = 405;
}

/*
equation index: 406
type: SIMPLE_ASSIGN
world.z_arrowHead.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_406(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,406};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[440]] /* world.z_arrowHead.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 406;
}

/*
equation index: 407
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].extra = 0.0
*/
void Pendulum_eqFunction_407(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,407};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[494]] /* world.z_label.cylinders[1].extra variable */) = 0.0;
  threadData->lastEquationSolved = 407;
}

/*
equation index: 408
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].extra = 0.0
*/
void Pendulum_eqFunction_408(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,408};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[495]] /* world.z_label.cylinders[2].extra variable */) = 0.0;
  threadData->lastEquationSolved = 408;
}

/*
equation index: 409
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].length = 1.4142135623730951 * abs(world.scaledLabel)
*/
void Pendulum_eqFunction_409(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,409};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[499]] /* world.z_label.cylinders[3].length variable */) = (1.4142135623730951) * (fabs((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */)));
  threadData->lastEquationSolved = 409;
}

/*
equation index: 410
type: SIMPLE_ASSIGN
world.z_label.cylinders[2].length = abs(world.scaledLabel)
*/
void Pendulum_eqFunction_410(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,410};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[498]] /* world.z_label.cylinders[2].length variable */) = fabs((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */));
  threadData->lastEquationSolved = 410;
}

/*
equation index: 411
type: SIMPLE_ASSIGN
world.z_label.cylinders[1].length = abs(world.scaledLabel)
*/
void Pendulum_eqFunction_411(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,411};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[497]] /* world.z_label.cylinders[1].length variable */) = fabs((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[99]] /* world.scaledLabel PARAM */));
  threadData->lastEquationSolved = 411;
}

/*
equation index: 412
type: SIMPLE_ASSIGN
world.z_label.r_abs[2] = 0.0
*/
void Pendulum_eqFunction_412(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,412};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[549]] /* world.z_label.r_abs[2] variable */) = 0.0;
  threadData->lastEquationSolved = 412;
}

/*
equation index: 413
type: SIMPLE_ASSIGN
world.z_label.r_abs[1] = 0.0
*/
void Pendulum_eqFunction_413(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,413};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[548]] /* world.z_label.r_abs[1] variable */) = 0.0;
  threadData->lastEquationSolved = 413;
}

/*
equation index: 414
type: SIMPLE_ASSIGN
world.z_label.r[1] = 0.0
*/
void Pendulum_eqFunction_414(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,414};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[545]] /* world.z_label.r[1] variable */) = 0.0;
  threadData->lastEquationSolved = 414;
}

/*
equation index: 415
type: SIMPLE_ASSIGN
world.z_label.r[2] = 0.0
*/
void Pendulum_eqFunction_415(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,415};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[546]] /* world.z_label.r[2] variable */) = 0.0;
  threadData->lastEquationSolved = 415;
}

/*
equation index: 416
type: SIMPLE_ASSIGN
world.z_label.r[3] = 0.0
*/
void Pendulum_eqFunction_416(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,416};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[547]] /* world.z_label.r[3] variable */) = 0.0;
  threadData->lastEquationSolved = 416;
}

/*
equation index: 417
type: SIMPLE_ASSIGN
world.z_label.r_lines[1] = 0.0
*/
void Pendulum_eqFunction_417(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,417};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[550]] /* world.z_label.r_lines[1] variable */) = 0.0;
  threadData->lastEquationSolved = 417;
}

/*
equation index: 418
type: SIMPLE_ASSIGN
world.z_label.r_lines[2] = 0.0
*/
void Pendulum_eqFunction_418(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,418};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[551]] /* world.z_label.r_lines[2] variable */) = 0.0;
  threadData->lastEquationSolved = 418;
}

/*
equation index: 419
type: SIMPLE_ASSIGN
world.z_label.n_x[1] = 0.0
*/
void Pendulum_eqFunction_419(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,419};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[539]] /* world.z_label.n_x[1] variable */) = 0.0;
  threadData->lastEquationSolved = 419;
}

/*
equation index: 420
type: SIMPLE_ASSIGN
world.z_label.n_x[2] = 0.0
*/
void Pendulum_eqFunction_420(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,420};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[540]] /* world.z_label.n_x[2] variable */) = 0.0;
  threadData->lastEquationSolved = 420;
}

/*
equation index: 421
type: SIMPLE_ASSIGN
world.z_label.n_x[3] = 1.0
*/
void Pendulum_eqFunction_421(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,421};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[541]] /* world.z_label.n_x[3] variable */) = 1.0;
  threadData->lastEquationSolved = 421;
}

/*
equation index: 422
type: SIMPLE_ASSIGN
world.z_label.n_y[1] = 0.0
*/
void Pendulum_eqFunction_422(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,422};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[542]] /* world.z_label.n_y[1] variable */) = 0.0;
  threadData->lastEquationSolved = 422;
}

/*
equation index: 423
type: SIMPLE_ASSIGN
world.z_label.n_y[2] = 1.0
*/
void Pendulum_eqFunction_423(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,423};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[543]] /* world.z_label.n_y[2] variable */) = 1.0;
  threadData->lastEquationSolved = 423;
}

/*
equation index: 424
type: SIMPLE_ASSIGN
world.z_label.n_y[3] = 0.0
*/
void Pendulum_eqFunction_424(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,424};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[544]] /* world.z_label.n_y[3] variable */) = 0.0;
  threadData->lastEquationSolved = 424;
}

/*
equation index: 425
type: SIMPLE_ASSIGN
world.z_label.lines[1,1,1] = 0.0
*/
void Pendulum_eqFunction_425(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,425};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[533]] /* world.z_label.lines[1,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 425;
}

/*
equation index: 426
type: SIMPLE_ASSIGN
world.z_label.lines[1,1,2] = 0.0
*/
void Pendulum_eqFunction_426(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,426};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[534]] /* world.z_label.lines[1,1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 426;
}

/*
equation index: 427
type: SIMPLE_ASSIGN
world.z_label.lines[1,2,2] = 0.0
*/
void Pendulum_eqFunction_427(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,427};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[535]] /* world.z_label.lines[1,2,2] variable */) = 0.0;
  threadData->lastEquationSolved = 427;
}

/*
equation index: 428
type: SIMPLE_ASSIGN
world.z_label.lines[2,1,1] = 0.0
*/
void Pendulum_eqFunction_428(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,428};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[536]] /* world.z_label.lines[2,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 428;
}

/*
equation index: 429
type: SIMPLE_ASSIGN
world.z_label.lines[3,1,1] = 0.0
*/
void Pendulum_eqFunction_429(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,429};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[537]] /* world.z_label.lines[3,1,1] variable */) = 0.0;
  threadData->lastEquationSolved = 429;
}

/*
equation index: 430
type: SIMPLE_ASSIGN
world.z_label.lines[3,2,2] = 0.0
*/
void Pendulum_eqFunction_430(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,430};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[538]] /* world.z_label.lines[3,2,2] variable */) = 0.0;
  threadData->lastEquationSolved = 430;
}

/*
equation index: 431
type: SIMPLE_ASSIGN
world.z_label.R.T[1,1] = 1.0
*/
void Pendulum_eqFunction_431(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,431};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[470]] /* world.z_label.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 431;
}

/*
equation index: 432
type: SIMPLE_ASSIGN
world.z_label.R.T[1,2] = 0.0
*/
void Pendulum_eqFunction_432(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,432};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[471]] /* world.z_label.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 432;
}

/*
equation index: 433
type: SIMPLE_ASSIGN
world.z_label.R.T[1,3] = 0.0
*/
void Pendulum_eqFunction_433(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,433};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[472]] /* world.z_label.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 433;
}

/*
equation index: 434
type: SIMPLE_ASSIGN
world.z_label.R.T[2,1] = 0.0
*/
void Pendulum_eqFunction_434(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,434};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[473]] /* world.z_label.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 434;
}

/*
equation index: 435
type: SIMPLE_ASSIGN
world.z_label.R.T[2,2] = 1.0
*/
void Pendulum_eqFunction_435(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,435};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[474]] /* world.z_label.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 435;
}

/*
equation index: 436
type: SIMPLE_ASSIGN
world.z_label.R.T[2,3] = 0.0
*/
void Pendulum_eqFunction_436(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,436};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[475]] /* world.z_label.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 436;
}

/*
equation index: 437
type: SIMPLE_ASSIGN
world.z_label.R.T[3,1] = 0.0
*/
void Pendulum_eqFunction_437(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,437};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[476]] /* world.z_label.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 437;
}

/*
equation index: 438
type: SIMPLE_ASSIGN
world.z_label.R.T[3,2] = 0.0
*/
void Pendulum_eqFunction_438(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,438};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[477]] /* world.z_label.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 438;
}

/*
equation index: 439
type: SIMPLE_ASSIGN
world.z_label.R.T[3,3] = 1.0
*/
void Pendulum_eqFunction_439(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,439};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[478]] /* world.z_label.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 439;
}

/*
equation index: 440
type: SIMPLE_ASSIGN
world.z_label.cylinders[3].extra = 0.0
*/
void Pendulum_eqFunction_440(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,440};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[496]] /* world.z_label.cylinders[3].extra variable */) = 0.0;
  threadData->lastEquationSolved = 440;
}

/*
equation index: 441
type: SIMPLE_ASSIGN
world.z_label.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_441(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,441};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[552]] /* world.z_label.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 441;
}

/*
equation index: 442
type: SIMPLE_ASSIGN
world.gravityArrowLine.extra = 0.0
*/
void Pendulum_eqFunction_442(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,442};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[160]] /* world.gravityArrowLine.extra variable */) = 0.0;
  threadData->lastEquationSolved = 442;
}

/*
equation index: 443
type: SIMPLE_ASSIGN
world.gravityArrowLine.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_443(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,443};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[167]] /* world.gravityArrowLine.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 443;
}

/*
equation index: 444
type: SIMPLE_ASSIGN
world.gravityArrowHead.extra = 0.0
*/
void Pendulum_eqFunction_444(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,444};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[136]] /* world.gravityArrowHead.extra variable */) = 0.0;
  threadData->lastEquationSolved = 444;
}

/*
equation index: 445
type: SIMPLE_ASSIGN
world.gravityArrowHead.specularCoefficient = 0.0
*/
void Pendulum_eqFunction_445(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,445};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[144]] /* world.gravityArrowHead.specularCoefficient variable */) = 0.0;
  threadData->lastEquationSolved = 445;
}

/*
equation index: 446
type: SIMPLE_ASSIGN
rev.cylinder.extra = 0.0
*/
void Pendulum_eqFunction_446(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,446};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[93]] /* rev.cylinder.extra variable */) = 0.0;
  threadData->lastEquationSolved = 446;
}

/*
equation index: 447
type: SIMPLE_ASSIGN
body.a_0[1] = 0.0
*/
void Pendulum_eqFunction_447(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,447};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* body.a_0[1] variable */) = 0.0;
  threadData->lastEquationSolved = 447;
}

/*
equation index: 448
type: SIMPLE_ASSIGN
body.a_0[2] = 0.0
*/
void Pendulum_eqFunction_448(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,448};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[19]] /* body.a_0[2] variable */) = 0.0;
  threadData->lastEquationSolved = 448;
}

/*
equation index: 449
type: SIMPLE_ASSIGN
body.a_0[3] = 0.0
*/
void Pendulum_eqFunction_449(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,449};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[20]] /* body.a_0[3] variable */) = 0.0;
  threadData->lastEquationSolved = 449;
}

/*
equation index: 450
type: SIMPLE_ASSIGN
body.v_0[3] = 0.0
*/
void Pendulum_eqFunction_450(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,450};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[74]] /* body.v_0[3] DUMMY_STATE */) = 0.0;
  threadData->lastEquationSolved = 450;
}

/*
equation index: 451
type: SIMPLE_ASSIGN
body.v_0[2] = 0.0
*/
void Pendulum_eqFunction_451(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,451};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[73]] /* body.v_0[2] DUMMY_STATE */) = 0.0;
  threadData->lastEquationSolved = 451;
}

/*
equation index: 452
type: SIMPLE_ASSIGN
body.v_0[1] = 0.0
*/
void Pendulum_eqFunction_452(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,452};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[72]] /* body.v_0[1] DUMMY_STATE */) = 0.0;
  threadData->lastEquationSolved = 452;
}

/*
equation index: 453
type: SIMPLE_ASSIGN
world.frame_b.R.T[1,1] = 1.0
*/
void Pendulum_eqFunction_453(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,453};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[106]] /* world.frame_b.R.T[1,1] variable */) = 1.0;
  threadData->lastEquationSolved = 453;
}

/*
equation index: 454
type: SIMPLE_ASSIGN
world.frame_b.R.T[1,2] = 0.0
*/
void Pendulum_eqFunction_454(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,454};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[107]] /* world.frame_b.R.T[1,2] variable */) = 0.0;
  threadData->lastEquationSolved = 454;
}

/*
equation index: 455
type: SIMPLE_ASSIGN
world.frame_b.R.T[1,3] = 0.0
*/
void Pendulum_eqFunction_455(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,455};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[108]] /* world.frame_b.R.T[1,3] variable */) = 0.0;
  threadData->lastEquationSolved = 455;
}

/*
equation index: 456
type: SIMPLE_ASSIGN
world.frame_b.R.T[2,1] = 0.0
*/
void Pendulum_eqFunction_456(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,456};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[109]] /* world.frame_b.R.T[2,1] variable */) = 0.0;
  threadData->lastEquationSolved = 456;
}

/*
equation index: 457
type: SIMPLE_ASSIGN
world.frame_b.R.T[2,2] = 1.0
*/
void Pendulum_eqFunction_457(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,457};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[110]] /* world.frame_b.R.T[2,2] variable */) = 1.0;
  threadData->lastEquationSolved = 457;
}

/*
equation index: 458
type: SIMPLE_ASSIGN
world.frame_b.R.T[2,3] = 0.0
*/
void Pendulum_eqFunction_458(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,458};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[111]] /* world.frame_b.R.T[2,3] variable */) = 0.0;
  threadData->lastEquationSolved = 458;
}

/*
equation index: 459
type: SIMPLE_ASSIGN
world.frame_b.R.T[3,1] = 0.0
*/
void Pendulum_eqFunction_459(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,459};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[112]] /* world.frame_b.R.T[3,1] variable */) = 0.0;
  threadData->lastEquationSolved = 459;
}

/*
equation index: 460
type: SIMPLE_ASSIGN
world.frame_b.R.T[3,2] = 0.0
*/
void Pendulum_eqFunction_460(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,460};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[113]] /* world.frame_b.R.T[3,2] variable */) = 0.0;
  threadData->lastEquationSolved = 460;
}

/*
equation index: 461
type: SIMPLE_ASSIGN
world.frame_b.R.T[3,3] = 1.0
*/
void Pendulum_eqFunction_461(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,461};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[114]] /* world.frame_b.R.T[3,3] variable */) = 1.0;
  threadData->lastEquationSolved = 461;
}

/*
equation index: 462
type: SIMPLE_ASSIGN
world.frame_b.R.w[1] = 0.0
*/
void Pendulum_eqFunction_462(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,462};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[115]] /* world.frame_b.R.w[1] variable */) = 0.0;
  threadData->lastEquationSolved = 462;
}

/*
equation index: 463
type: SIMPLE_ASSIGN
world.frame_b.R.w[2] = 0.0
*/
void Pendulum_eqFunction_463(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,463};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[116]] /* world.frame_b.R.w[2] variable */) = 0.0;
  threadData->lastEquationSolved = 463;
}

/*
equation index: 464
type: SIMPLE_ASSIGN
world.frame_b.R.w[3] = 0.0
*/
void Pendulum_eqFunction_464(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,464};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[117]] /* world.frame_b.R.w[3] variable */) = 0.0;
  threadData->lastEquationSolved = 464;
}

/*
equation index: 465
type: SIMPLE_ASSIGN
world.frame_b.r_0[1] = 0.0
*/
void Pendulum_eqFunction_465(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,465};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[118]] /* world.frame_b.r_0[1] variable */) = 0.0;
  threadData->lastEquationSolved = 465;
}

/*
equation index: 466
type: SIMPLE_ASSIGN
world.frame_b.r_0[2] = 0.0
*/
void Pendulum_eqFunction_466(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,466};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[119]] /* world.frame_b.r_0[2] variable */) = 0.0;
  threadData->lastEquationSolved = 466;
}

/*
equation index: 467
type: SIMPLE_ASSIGN
world.frame_b.r_0[3] = 0.0
*/
void Pendulum_eqFunction_467(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,467};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[120]] /* world.frame_b.r_0[3] variable */) = 0.0;
  threadData->lastEquationSolved = 467;
}

/*
equation index: 468
type: SIMPLE_ASSIGN
body.g_0[1] = 0.0
*/
void Pendulum_eqFunction_468(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,468};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[47]] /* body.g_0[1] variable */) = 0.0;
  threadData->lastEquationSolved = 468;
}

/*
equation index: 469
type: SIMPLE_ASSIGN
body.g_0[3] = 0.0
*/
void Pendulum_eqFunction_469(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,469};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[48]] /* body.g_0[3] variable */) = 0.0;
  threadData->lastEquationSolved = 469;
}

/*
equation index: 470
type: SIMPLE_ASSIGN
body.cylinder.length = if sqrt(body.r_CM * body.r_CM) > 0.05555555555555555 then sqrt(body.r_CM * body.r_CM) - (if body.cylinderDiameter > 0.12222222222222223 then 0.05555555555555555 else 0.0) else 0.0
*/
void Pendulum_eqFunction_470(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,470};
  real_array tmp6;
  real_array tmp7;
  modelica_real tmp8;
  modelica_boolean tmp9;
  real_array tmp10;
  real_array tmp11;
  modelica_real tmp12;
  modelica_boolean tmp13;
  modelica_boolean tmp14;
  modelica_real tmp15;
  real_array_create(&tmp6, ((modelica_real*)&((&data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* body.r_CM[1] PARAM */)[((modelica_integer) 1) - 1])), 1, (_index_t)3);
  real_array_create(&tmp7, ((modelica_real*)&((&data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* body.r_CM[1] PARAM */)[((modelica_integer) 1) - 1])), 1, (_index_t)3);
  tmp8 = mul_real_scalar_product(tmp6, tmp7);
  if(!(tmp8 >= 0.0))
  {
    if (data->simulationInfo->noThrowAsserts) {
      FILE_INFO info = {"",0,0,0,0,0};
      infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      data->simulationInfo->needToReThrow = 1;
    } else {
      FILE_INFO info = {"",0,0,0,0,0};
      omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
      throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(body.r_CM * body.r_CM) was %g should be >= 0", tmp8);
    }
  }tmp9 = Greater(sqrt(tmp8),0.05555555555555555);
  tmp14 = (modelica_boolean)tmp9;
  if(tmp14)
  {
    real_array_create(&tmp10, ((modelica_real*)&((&data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* body.r_CM[1] PARAM */)[((modelica_integer) 1) - 1])), 1, (_index_t)3);
    real_array_create(&tmp11, ((modelica_real*)&((&data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[39]] /* body.r_CM[1] PARAM */)[((modelica_integer) 1) - 1])), 1, (_index_t)3);
    tmp12 = mul_real_scalar_product(tmp10, tmp11);
    if(!(tmp12 >= 0.0))
    {
      if (data->simulationInfo->noThrowAsserts) {
        FILE_INFO info = {"",0,0,0,0,0};
        infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        data->simulationInfo->needToReThrow = 1;
      } else {
        FILE_INFO info = {"",0,0,0,0,0};
        omc_assert_warning(info, "The following assertion has been violated %sat time %f", initial() ? "during initialization " : "", data->localData[0]->timeValue);
        throwStreamPrintWithEquationIndexes(threadData, info, equationIndexes, "Model error: Argument of sqrt(body.r_CM * body.r_CM) was %g should be >= 0", tmp12);
      }
    }tmp13 = Greater((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[34]] /* body.cylinderDiameter PARAM */),0.12222222222222223);
    tmp15 = sqrt(tmp12) - ((tmp13?0.05555555555555555:0.0));
  }
  else
  {
    tmp15 = 0.0;
  }
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* body.cylinder.length variable */) = tmp15;
  threadData->lastEquationSolved = 470;
}

/*
equation index: 471
type: SIMPLE_ASSIGN
body.cylinder.extra = 0.0
*/
void Pendulum_eqFunction_471(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,471};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* body.cylinder.extra variable */) = 0.0;
  threadData->lastEquationSolved = 471;
}

/*
equation index: 472
type: SIMPLE_ASSIGN
body.sphere.length = 0.1111111111111111
*/
void Pendulum_eqFunction_472(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,472};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[63]] /* body.sphere.length variable */) = 0.1111111111111111;
  threadData->lastEquationSolved = 472;
}

/*
equation index: 473
type: SIMPLE_ASSIGN
body.sphere.width = 0.1111111111111111
*/
void Pendulum_eqFunction_473(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,473};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[68]] /* body.sphere.width variable */) = 0.1111111111111111;
  threadData->lastEquationSolved = 473;
}

/*
equation index: 474
type: SIMPLE_ASSIGN
body.sphere.height = 0.1111111111111111
*/
void Pendulum_eqFunction_474(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,474};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[62]] /* body.sphere.height variable */) = 0.1111111111111111;
  threadData->lastEquationSolved = 474;
}

/*
equation index: 475
type: SIMPLE_ASSIGN
body.sphere.extra = 0.0
*/
void Pendulum_eqFunction_475(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,475};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[61]] /* body.sphere.extra variable */) = 0.0;
  threadData->lastEquationSolved = 475;
}

/*
equation index: 476
type: SIMPLE_ASSIGN
rev.phi = $START.rev.phi
*/
void Pendulum_eqFunction_476(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,476};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */) = ((modelica_real *)((data->modelData->realVarsData[0] /* rev.phi STATE(1,rev.w) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 476;
}
extern void Pendulum_eqFunction_608(DATA *data, threadData_t *threadData);


/*
equation index: 478
type: SIMPLE_ASSIGN
body.frame_a.R.T[3,3] = rev.e[3] ^ 2.0 + (1.0 - rev.e[3] ^ 2.0) * cos(rev.phi)
*/
void Pendulum_eqFunction_478(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,478};
  modelica_real tmp16;
  modelica_real tmp17;
  tmp16 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */);
  tmp17 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */);
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[40]] /* body.frame_a.R.T[3,3] variable */) = (tmp16 * tmp16) + (1.0 - ((tmp17 * tmp17))) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */)));
  threadData->lastEquationSolved = 478;
}

/*
equation index: 479
type: SIMPLE_ASSIGN
body.frame_a.R.T[3,2] = (rev.e[3] - rev.e[3] * cos(rev.phi)) * rev.e[2] - rev.e[1] * sin(rev.phi)
*/
void Pendulum_eqFunction_479(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,479};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[39]] /* body.frame_a.R.T[3,2] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))))) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))));
  threadData->lastEquationSolved = 479;
}

/*
equation index: 480
type: SIMPLE_ASSIGN
body.frame_a.R.T[3,1] = (rev.e[3] - rev.e[3] * cos(rev.phi)) * rev.e[1] + rev.e[2] * sin(rev.phi)
*/
void Pendulum_eqFunction_480(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,480};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[38]] /* body.frame_a.R.T[3,1] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))))) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */)));
  threadData->lastEquationSolved = 480;
}

/*
equation index: 481
type: SIMPLE_ASSIGN
body.frame_a.R.T[2,3] = (rev.e[2] - rev.e[2] * cos(rev.phi)) * rev.e[3] + rev.e[1] * sin(rev.phi)
*/
void Pendulum_eqFunction_481(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,481};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[37]] /* body.frame_a.R.T[2,3] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))))) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */)));
  threadData->lastEquationSolved = 481;
}

/*
equation index: 482
type: SIMPLE_ASSIGN
body.frame_a.R.T[2,2] = rev.e[2] ^ 2.0 + (1.0 - rev.e[2] ^ 2.0) * cos(rev.phi)
*/
void Pendulum_eqFunction_482(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,482};
  modelica_real tmp18;
  modelica_real tmp19;
  tmp18 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */);
  tmp19 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */);
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[36]] /* body.frame_a.R.T[2,2] variable */) = (tmp18 * tmp18) + (1.0 - ((tmp19 * tmp19))) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */)));
  threadData->lastEquationSolved = 482;
}

/*
equation index: 483
type: SIMPLE_ASSIGN
body.frame_a.R.T[2,1] = (rev.e[2] - rev.e[2] * cos(rev.phi)) * rev.e[1] - rev.e[3] * sin(rev.phi)
*/
void Pendulum_eqFunction_483(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,483};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[35]] /* body.frame_a.R.T[2,1] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))))) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))));
  threadData->lastEquationSolved = 483;
}

/*
equation index: 484
type: SIMPLE_ASSIGN
body.frame_a.R.T[1,3] = (rev.e[1] - rev.e[1] * cos(rev.phi)) * rev.e[3] - rev.e[2] * sin(rev.phi)
*/
void Pendulum_eqFunction_484(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,484};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[34]] /* body.frame_a.R.T[1,3] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))))) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))));
  threadData->lastEquationSolved = 484;
}

/*
equation index: 485
type: SIMPLE_ASSIGN
body.frame_a.R.T[1,2] = (rev.e[1] - rev.e[1] * cos(rev.phi)) * rev.e[2] + rev.e[3] * sin(rev.phi)
*/
void Pendulum_eqFunction_485(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,485};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[33]] /* body.frame_a.R.T[1,2] variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */) - (((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */))))) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) * (sin((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */)));
  threadData->lastEquationSolved = 485;
}

/*
equation index: 486
type: SIMPLE_ASSIGN
body.frame_a.R.T[1,1] = rev.e[1] ^ 2.0 + (1.0 - rev.e[1] ^ 2.0) * cos(rev.phi)
*/
void Pendulum_eqFunction_486(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,486};
  modelica_real tmp20;
  modelica_real tmp21;
  tmp20 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */);
  tmp21 = (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */);
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[32]] /* body.frame_a.R.T[1,1] variable */) = (tmp20 * tmp20) + (1.0 - ((tmp21 * tmp21))) * (cos((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* rev.phi STATE(1,rev.w) */)));
  threadData->lastEquationSolved = 486;
}

/*
equation index: 487
type: SIMPLE_ASSIGN
rev.w = $START.rev.w
*/
void Pendulum_eqFunction_487(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,487};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* rev.w STATE(1,damper.a_rel) */) = ((modelica_real *)((data->modelData->realVarsData[1] /* rev.w STATE(1,damper.a_rel) */).attribute .start.data))[0];
  threadData->lastEquationSolved = 487;
}

/*
equation index: 488
type: SIMPLE_ASSIGN
rev.R_rel.w[1] = rev.e[1] * rev.w
*/
void Pendulum_eqFunction_488(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,488};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[86]] /* rev.R_rel.w[1] DUMMY_STATE */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[53]] /* rev.e[1] PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* rev.w STATE(1,damper.a_rel) */));
  threadData->lastEquationSolved = 488;
}

/*
equation index: 489
type: SIMPLE_ASSIGN
body.w_a[1] = rev.R_rel.w[1]
*/
void Pendulum_eqFunction_489(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,489};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[75]] /* body.w_a[1] DUMMY_STATE */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[86]] /* rev.R_rel.w[1] DUMMY_STATE */);
  threadData->lastEquationSolved = 489;
}

/*
equation index: 490
type: SIMPLE_ASSIGN
rev.R_rel.w[2] = rev.e[2] * rev.w
*/
void Pendulum_eqFunction_490(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,490};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[87]] /* rev.R_rel.w[2] DUMMY_STATE */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[54]] /* rev.e[2] PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* rev.w STATE(1,damper.a_rel) */));
  threadData->lastEquationSolved = 490;
}

/*
equation index: 491
type: SIMPLE_ASSIGN
body.w_a[2] = rev.R_rel.w[2]
*/
void Pendulum_eqFunction_491(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,491};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[76]] /* body.w_a[2] DUMMY_STATE */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[87]] /* rev.R_rel.w[2] DUMMY_STATE */);
  threadData->lastEquationSolved = 491;
}

/*
equation index: 492
type: SIMPLE_ASSIGN
rev.R_rel.w[3] = rev.e[3] * rev.w
*/
void Pendulum_eqFunction_492(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,492};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[88]] /* rev.R_rel.w[3] DUMMY_STATE */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[55]] /* rev.e[3] PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* rev.w STATE(1,damper.a_rel) */));
  threadData->lastEquationSolved = 492;
}

/*
equation index: 493
type: SIMPLE_ASSIGN
body.w_a[3] = rev.R_rel.w[3]
*/
void Pendulum_eqFunction_493(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,493};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[77]] /* body.w_a[3] DUMMY_STATE */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[88]] /* rev.R_rel.w[3] DUMMY_STATE */);
  threadData->lastEquationSolved = 493;
}
extern void Pendulum_eqFunction_562(DATA *data, threadData_t *threadData);


/*
equation index: 495
type: SIMPLE_ASSIGN
$DER.damper.phi_rel = damper.w_rel
*/
void Pendulum_eqFunction_495(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,495};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(damper.phi_rel) DUMMY_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[85]] /* damper.w_rel DUMMY_STATE */);
  threadData->lastEquationSolved = 495;
}

/*
equation index: 496
type: SIMPLE_ASSIGN
damper.tau = damper.d * damper.w_rel
*/
void Pendulum_eqFunction_496(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,496};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[84]] /* damper.tau variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[49]] /* damper.d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[85]] /* damper.w_rel DUMMY_STATE */));
  threadData->lastEquationSolved = 496;
}

/*
equation index: 497
type: SIMPLE_ASSIGN
damper.lossPower = damper.tau * damper.w_rel
*/
void Pendulum_eqFunction_497(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,497};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[82]] /* damper.lossPower variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[84]] /* damper.tau variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[85]] /* damper.w_rel DUMMY_STATE */));
  threadData->lastEquationSolved = 497;
}

/*
equation index: 521
type: LINEAR

<var>body.z_a[3]</var>
<var>body.z_a[1]</var>
<row>

</row>
<matrix>
</matrix>
*/
OMC_DISABLE_OPT
void Pendulum_eqFunction_521(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,521};
  /* Linear equation system */
  int retValue;
  double aux_x[2] = { (data->localData[1]->realVars[data->simulationInfo->realVarsIndex[80]] /* body.z_a[3] variable */),(data->localData[1]->realVars[data->simulationInfo->realVarsIndex[78]] /* body.z_a[1] variable */) };
  infoStreamPrint(OMC_LOG_DT, 0, "Solving linear system 521 (STRICT TEARING SET if tearing enabled) at time = %18.10e", data->localData[0]->timeValue);

  retValue = solve_linear_system(data, threadData, 0, &aux_x[0]);

  /* check if solution process was successful */
  if (retValue > 0){
    const int indexes[2] = {1,521};
    throwStreamPrintWithEquationIndexes(threadData, omc_dummyFileInfo, indexes, "Solving linear system 521 failed at time=%.15g.\nFor more information please use -lv LOG_LS.", data->localData[0]->timeValue);
  }
  /* write solution */
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[80]] /* body.z_a[3] variable */) = aux_x[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[78]] /* body.z_a[1] variable */) = aux_x[1];

  threadData->lastEquationSolved = 521;
}
extern void Pendulum_eqFunction_643(DATA *data, threadData_t *threadData);


/*
equation index: 523
type: SIMPLE_ASSIGN
damper.a_rel = rev.a
*/
void Pendulum_eqFunction_523(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,523};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[81]] /* damper.a_rel variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[89]] /* rev.a variable */);
  threadData->lastEquationSolved = 523;
}
extern void Pendulum_eqFunction_646(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_641(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_648(DATA *data, threadData_t *threadData);


/*
equation index: 527
type: SIMPLE_ASSIGN
$DER.rev.w = rev.a
*/
void Pendulum_eqFunction_527(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,527};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* der(rev.w) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[89]] /* rev.a variable */);
  threadData->lastEquationSolved = 527;
}
extern void Pendulum_eqFunction_650(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_651(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_654(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_649(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_652(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_655(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_599(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_583(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_586(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_587(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_584(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_585(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_594(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_595(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_591(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_592(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_588(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_589(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_596(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_597(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_593(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_590(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_598(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_563(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_565(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_568(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_571(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_566(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_564(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_570(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_567(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_569(DATA *data, threadData_t *threadData);

extern void Pendulum_eqFunction_572(DATA *data, threadData_t *threadData);

OMC_DISABLE_OPT
void Pendulum_functionInitialEquations_1(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[268])(DATA*, threadData_t*) = {
    Pendulum_eqFunction_270,
    Pendulum_eqFunction_271,
    Pendulum_eqFunction_272,
    Pendulum_eqFunction_273,
    Pendulum_eqFunction_274,
    Pendulum_eqFunction_275,
    Pendulum_eqFunction_276,
    Pendulum_eqFunction_277,
    Pendulum_eqFunction_278,
    Pendulum_eqFunction_279,
    Pendulum_eqFunction_280,
    Pendulum_eqFunction_281,
    Pendulum_eqFunction_282,
    Pendulum_eqFunction_283,
    Pendulum_eqFunction_284,
    Pendulum_eqFunction_285,
    Pendulum_eqFunction_286,
    Pendulum_eqFunction_287,
    Pendulum_eqFunction_288,
    Pendulum_eqFunction_289,
    Pendulum_eqFunction_290,
    Pendulum_eqFunction_291,
    Pendulum_eqFunction_292,
    Pendulum_eqFunction_293,
    Pendulum_eqFunction_294,
    Pendulum_eqFunction_295,
    Pendulum_eqFunction_296,
    Pendulum_eqFunction_297,
    Pendulum_eqFunction_298,
    Pendulum_eqFunction_299,
    Pendulum_eqFunction_300,
    Pendulum_eqFunction_301,
    Pendulum_eqFunction_302,
    Pendulum_eqFunction_303,
    Pendulum_eqFunction_304,
    Pendulum_eqFunction_305,
    Pendulum_eqFunction_306,
    Pendulum_eqFunction_307,
    Pendulum_eqFunction_308,
    Pendulum_eqFunction_309,
    Pendulum_eqFunction_310,
    Pendulum_eqFunction_311,
    Pendulum_eqFunction_312,
    Pendulum_eqFunction_313,
    Pendulum_eqFunction_314,
    Pendulum_eqFunction_315,
    Pendulum_eqFunction_316,
    Pendulum_eqFunction_317,
    Pendulum_eqFunction_318,
    Pendulum_eqFunction_319,
    Pendulum_eqFunction_320,
    Pendulum_eqFunction_321,
    Pendulum_eqFunction_322,
    Pendulum_eqFunction_323,
    Pendulum_eqFunction_324,
    Pendulum_eqFunction_325,
    Pendulum_eqFunction_326,
    Pendulum_eqFunction_327,
    Pendulum_eqFunction_328,
    Pendulum_eqFunction_329,
    Pendulum_eqFunction_330,
    Pendulum_eqFunction_331,
    Pendulum_eqFunction_332,
    Pendulum_eqFunction_333,
    Pendulum_eqFunction_334,
    Pendulum_eqFunction_335,
    Pendulum_eqFunction_336,
    Pendulum_eqFunction_337,
    Pendulum_eqFunction_338,
    Pendulum_eqFunction_339,
    Pendulum_eqFunction_340,
    Pendulum_eqFunction_341,
    Pendulum_eqFunction_342,
    Pendulum_eqFunction_343,
    Pendulum_eqFunction_344,
    Pendulum_eqFunction_345,
    Pendulum_eqFunction_346,
    Pendulum_eqFunction_347,
    Pendulum_eqFunction_348,
    Pendulum_eqFunction_349,
    Pendulum_eqFunction_350,
    Pendulum_eqFunction_351,
    Pendulum_eqFunction_352,
    Pendulum_eqFunction_353,
    Pendulum_eqFunction_354,
    Pendulum_eqFunction_355,
    Pendulum_eqFunction_356,
    Pendulum_eqFunction_357,
    Pendulum_eqFunction_358,
    Pendulum_eqFunction_359,
    Pendulum_eqFunction_360,
    Pendulum_eqFunction_361,
    Pendulum_eqFunction_362,
    Pendulum_eqFunction_363,
    Pendulum_eqFunction_364,
    Pendulum_eqFunction_365,
    Pendulum_eqFunction_573,
    Pendulum_eqFunction_575,
    Pendulum_eqFunction_578,
    Pendulum_eqFunction_581,
    Pendulum_eqFunction_576,
    Pendulum_eqFunction_574,
    Pendulum_eqFunction_580,
    Pendulum_eqFunction_577,
    Pendulum_eqFunction_579,
    Pendulum_eqFunction_582,
    Pendulum_eqFunction_376,
    Pendulum_eqFunction_377,
    Pendulum_eqFunction_378,
    Pendulum_eqFunction_379,
    Pendulum_eqFunction_380,
    Pendulum_eqFunction_381,
    Pendulum_eqFunction_382,
    Pendulum_eqFunction_383,
    Pendulum_eqFunction_384,
    Pendulum_eqFunction_385,
    Pendulum_eqFunction_386,
    Pendulum_eqFunction_387,
    Pendulum_eqFunction_388,
    Pendulum_eqFunction_389,
    Pendulum_eqFunction_390,
    Pendulum_eqFunction_391,
    Pendulum_eqFunction_392,
    Pendulum_eqFunction_393,
    Pendulum_eqFunction_394,
    Pendulum_eqFunction_395,
    Pendulum_eqFunction_396,
    Pendulum_eqFunction_397,
    Pendulum_eqFunction_398,
    Pendulum_eqFunction_399,
    Pendulum_eqFunction_400,
    Pendulum_eqFunction_401,
    Pendulum_eqFunction_402,
    Pendulum_eqFunction_403,
    Pendulum_eqFunction_404,
    Pendulum_eqFunction_405,
    Pendulum_eqFunction_406,
    Pendulum_eqFunction_407,
    Pendulum_eqFunction_408,
    Pendulum_eqFunction_409,
    Pendulum_eqFunction_410,
    Pendulum_eqFunction_411,
    Pendulum_eqFunction_412,
    Pendulum_eqFunction_413,
    Pendulum_eqFunction_414,
    Pendulum_eqFunction_415,
    Pendulum_eqFunction_416,
    Pendulum_eqFunction_417,
    Pendulum_eqFunction_418,
    Pendulum_eqFunction_419,
    Pendulum_eqFunction_420,
    Pendulum_eqFunction_421,
    Pendulum_eqFunction_422,
    Pendulum_eqFunction_423,
    Pendulum_eqFunction_424,
    Pendulum_eqFunction_425,
    Pendulum_eqFunction_426,
    Pendulum_eqFunction_427,
    Pendulum_eqFunction_428,
    Pendulum_eqFunction_429,
    Pendulum_eqFunction_430,
    Pendulum_eqFunction_431,
    Pendulum_eqFunction_432,
    Pendulum_eqFunction_433,
    Pendulum_eqFunction_434,
    Pendulum_eqFunction_435,
    Pendulum_eqFunction_436,
    Pendulum_eqFunction_437,
    Pendulum_eqFunction_438,
    Pendulum_eqFunction_439,
    Pendulum_eqFunction_440,
    Pendulum_eqFunction_441,
    Pendulum_eqFunction_442,
    Pendulum_eqFunction_443,
    Pendulum_eqFunction_444,
    Pendulum_eqFunction_445,
    Pendulum_eqFunction_446,
    Pendulum_eqFunction_447,
    Pendulum_eqFunction_448,
    Pendulum_eqFunction_449,
    Pendulum_eqFunction_450,
    Pendulum_eqFunction_451,
    Pendulum_eqFunction_452,
    Pendulum_eqFunction_453,
    Pendulum_eqFunction_454,
    Pendulum_eqFunction_455,
    Pendulum_eqFunction_456,
    Pendulum_eqFunction_457,
    Pendulum_eqFunction_458,
    Pendulum_eqFunction_459,
    Pendulum_eqFunction_460,
    Pendulum_eqFunction_461,
    Pendulum_eqFunction_462,
    Pendulum_eqFunction_463,
    Pendulum_eqFunction_464,
    Pendulum_eqFunction_465,
    Pendulum_eqFunction_466,
    Pendulum_eqFunction_467,
    Pendulum_eqFunction_468,
    Pendulum_eqFunction_469,
    Pendulum_eqFunction_470,
    Pendulum_eqFunction_471,
    Pendulum_eqFunction_472,
    Pendulum_eqFunction_473,
    Pendulum_eqFunction_474,
    Pendulum_eqFunction_475,
    Pendulum_eqFunction_476,
    Pendulum_eqFunction_608,
    Pendulum_eqFunction_478,
    Pendulum_eqFunction_479,
    Pendulum_eqFunction_480,
    Pendulum_eqFunction_481,
    Pendulum_eqFunction_482,
    Pendulum_eqFunction_483,
    Pendulum_eqFunction_484,
    Pendulum_eqFunction_485,
    Pendulum_eqFunction_486,
    Pendulum_eqFunction_487,
    Pendulum_eqFunction_488,
    Pendulum_eqFunction_489,
    Pendulum_eqFunction_490,
    Pendulum_eqFunction_491,
    Pendulum_eqFunction_492,
    Pendulum_eqFunction_493,
    Pendulum_eqFunction_562,
    Pendulum_eqFunction_495,
    Pendulum_eqFunction_496,
    Pendulum_eqFunction_497,
    Pendulum_eqFunction_521,
    Pendulum_eqFunction_643,
    Pendulum_eqFunction_523,
    Pendulum_eqFunction_646,
    Pendulum_eqFunction_641,
    Pendulum_eqFunction_648,
    Pendulum_eqFunction_527,
    Pendulum_eqFunction_650,
    Pendulum_eqFunction_651,
    Pendulum_eqFunction_654,
    Pendulum_eqFunction_649,
    Pendulum_eqFunction_652,
    Pendulum_eqFunction_655,
    Pendulum_eqFunction_599,
    Pendulum_eqFunction_583,
    Pendulum_eqFunction_586,
    Pendulum_eqFunction_587,
    Pendulum_eqFunction_584,
    Pendulum_eqFunction_585,
    Pendulum_eqFunction_594,
    Pendulum_eqFunction_595,
    Pendulum_eqFunction_591,
    Pendulum_eqFunction_592,
    Pendulum_eqFunction_588,
    Pendulum_eqFunction_589,
    Pendulum_eqFunction_596,
    Pendulum_eqFunction_597,
    Pendulum_eqFunction_593,
    Pendulum_eqFunction_590,
    Pendulum_eqFunction_598,
    Pendulum_eqFunction_563,
    Pendulum_eqFunction_565,
    Pendulum_eqFunction_568,
    Pendulum_eqFunction_571,
    Pendulum_eqFunction_566,
    Pendulum_eqFunction_564,
    Pendulum_eqFunction_570,
    Pendulum_eqFunction_567,
    Pendulum_eqFunction_569,
    Pendulum_eqFunction_572
  };
  
  for (int id = 0; id < 268; id++) {
    eqFunctions[id](data, threadData);
  }
}
#if defined(__cplusplus)
}
#endif