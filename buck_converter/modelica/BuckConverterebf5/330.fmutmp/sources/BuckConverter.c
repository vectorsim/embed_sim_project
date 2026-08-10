/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "BuckConverter_model.h"
#include "simulation/solver/events.h"
#include "util/real_array.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int BuckConverter_input_function(DATA *data, threadData_t *threadData)
{
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* duty variable */) = data->simulationInfo->inputVars[0];
  
  return 0;
}

int BuckConverter_input_function_init(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[7].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[0] = real_get(data->modelData->realVarsData[7].attribute.start, 0);
  
  return 0;
}

int BuckConverter_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[7].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[0], 0, &data->modelData->realVarsData[7].attribute.start);
  
  return 0;
}

int BuckConverter_inputNames(DATA *data, char ** names){
  names[0] = (char *) data->modelData->realVarsData[7].info.name;
  
  return 0;
}

int BuckConverter_data_function(DATA *data, threadData_t *threadData)
{
  return 0;
}

int BuckConverter_dataReconciliationInputNames(DATA *data, char ** names){
  
  return 0;
}

int BuckConverter_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  
  return 0;
}

int BuckConverter_output_function(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* I_L variable */);
  data->simulationInfo->outputVars[1] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* I_load variable */);
  data->simulationInfo->outputVars[2] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* V_out variable */);
  
  return 0;
}

int BuckConverter_setc_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

int BuckConverter_setb_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}


/*
equation index: 8
type: SIMPLE_ASSIGN
I_load = $outputAlias_V_out / R_load
*/
void BuckConverter_eqFunction_8(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,8};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* I_load variable */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* $outputAlias_V_out STATE(1) */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[2]] /* R_load PARAM */),"R_load",equationIndexes);
  threadData->lastEquationSolved = 8;
}

/*
equation index: 9
type: SIMPLE_ASSIGN
$DER.$outputAlias_V_out = ($outputAlias_I_L - I_load) / C
*/
void BuckConverter_eqFunction_9(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,9};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* der($outputAlias_V_out) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* $outputAlias_I_L STATE(1) */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* I_load variable */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[0]] /* C PARAM */),"C",equationIndexes);
  threadData->lastEquationSolved = 9;
}

/*
equation index: 10
type: SIMPLE_ASSIGN
$DER.$outputAlias_I_L = (duty * V_in - $outputAlias_V_out) / L
*/
void BuckConverter_eqFunction_10(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,10};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* der($outputAlias_I_L) STATE_DER */) = DIVISION_SIM(((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* duty variable */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* V_in PARAM */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* $outputAlias_V_out STATE(1) */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[1]] /* L PARAM */),"L",equationIndexes);
  threadData->lastEquationSolved = 10;
}

/*
equation index: 11
type: SIMPLE_ASSIGN
I_L = $outputAlias_I_L
*/
void BuckConverter_eqFunction_11(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,11};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* I_L variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* $outputAlias_I_L STATE(1) */);
  threadData->lastEquationSolved = 11;
}

/*
equation index: 12
type: SIMPLE_ASSIGN
V_out = $outputAlias_V_out
*/
void BuckConverter_eqFunction_12(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* V_out variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* $outputAlias_V_out STATE(1) */);
  threadData->lastEquationSolved = 12;
}

OMC_DISABLE_OPT
int BuckConverter_functionDAE(DATA *data, threadData_t *threadData)
{
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  BuckConverter_functionLocalKnownVars(data, threadData);
  static void (*const eqFunctions[5])(DATA*, threadData_t*) = {
    BuckConverter_eqFunction_8,
    BuckConverter_eqFunction_9,
    BuckConverter_eqFunction_10,
    BuckConverter_eqFunction_11,
    BuckConverter_eqFunction_12
  };
  
  for (int id = 0; id < 5; id++) {
    eqFunctions[id](data, threadData);
  }
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  return 0;
}


int BuckConverter_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

/* forwarded equations */
extern void BuckConverter_eqFunction_8(DATA* data, threadData_t *threadData);
extern void BuckConverter_eqFunction_9(DATA* data, threadData_t *threadData);
extern void BuckConverter_eqFunction_10(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[3])(DATA*, threadData_t*) = {
    BuckConverter_eqFunction_8,
    BuckConverter_eqFunction_9,
    BuckConverter_eqFunction_10
  };
  
  for (int id = 0; id < 3; id++) {
    eqFunctions[id](data, threadData);
  }
}

int BuckConverter_functionODE(DATA *data, threadData_t *threadData)
{
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  BuckConverter_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  return 0;
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);
extern int _main_OptimizationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);

#include "BuckConverter_12jac.h"
#include "BuckConverter_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks BuckConverter_callback = {
  NULL,    /* performSimulation */
  NULL,    /* performQSSSimulation */
  NULL,    /* updateContinuousSystem */
  BuckConverter_callExternalObjectDestructors,    /* callExternalObjectDestructors */
  NULL,    /* initialNonLinearSystem */
  NULL,    /* initialLinearSystem */
  NULL,    /* initialMixedSystem */
  #if !defined(OMC_NO_STATESELECTION)
  BuckConverter_initializeStateSets,
  #else
  NULL,
  #endif    /* initializeStateSets */
  BuckConverter_initializeDAEmodeData,
  BuckConverter_functionODE,
  BuckConverter_functionAlgebraics,
  BuckConverter_functionDAE,
  BuckConverter_functionLocalKnownVars,
  BuckConverter_input_function,
  BuckConverter_input_function_init,
  BuckConverter_input_function_updateStartValues,
  BuckConverter_data_function,
  BuckConverter_output_function,
  BuckConverter_setc_function,
  BuckConverter_setb_function,
  BuckConverter_function_storeDelayed,
  BuckConverter_function_storeSpatialDistribution,
  BuckConverter_function_initSpatialDistribution,
  BuckConverter_updateBoundVariableAttributes,
  BuckConverter_functionInitialEquations,
  GLOBAL_EQUIDISTANT_HOMOTOPY,
  NULL,
  BuckConverter_functionRemovedInitialEquations,
  BuckConverter_updateBoundParameters,
  BuckConverter_checkForAsserts,
  BuckConverter_function_ZeroCrossingsEquations,
  BuckConverter_function_ZeroCrossings,
  BuckConverter_function_updateRelations,
  BuckConverter_zeroCrossingDescription,
  BuckConverter_relationDescription,
  BuckConverter_function_initSample,
  BuckConverter_INDEX_JAC_A,
  BuckConverter_INDEX_JAC_B,
  BuckConverter_INDEX_JAC_C,
  BuckConverter_INDEX_JAC_D,
  BuckConverter_INDEX_JAC_F,
  BuckConverter_INDEX_JAC_H,
  BuckConverter_initialAnalyticJacobianA,
  BuckConverter_initialAnalyticJacobianB,
  BuckConverter_initialAnalyticJacobianC,
  BuckConverter_initialAnalyticJacobianD,
  BuckConverter_initialAnalyticJacobianF,
  BuckConverter_initialAnalyticJacobianH,
  BuckConverter_functionJacA_column,
  BuckConverter_functionJacB_column,
  BuckConverter_functionJacC_column,
  BuckConverter_functionJacD_column,
  BuckConverter_functionJacF_column,
  BuckConverter_functionJacH_column,
  BuckConverter_linear_model_frame,
  BuckConverter_linear_model_datarecovery_frame,
  BuckConverter_mayer,
  BuckConverter_lagrange,
  BuckConverter_getInputVarIndicesInOptimization,
  BuckConverter_pickUpBoundsForInputsInOptimization,
  BuckConverter_setInputData,
  BuckConverter_getTimeGrid,
  BuckConverter_symbolicInlineSystem,
  BuckConverter_function_initSynchronous,
  BuckConverter_function_updateSynchronous,
  BuckConverter_function_equationsSynchronous,
  BuckConverter_inputNames,
  BuckConverter_dataReconciliationInputNames,
  BuckConverter_dataReconciliationUnmeasuredVariables,
  BuckConverter_read_simulation_info,
  BuckConverter_read_input_fmu,
  NULL,
  NULL,
  -1,
  NULL,
  NULL,
  -1

};

#define _OMC_LIT_RESOURCE_0_name_data "BuckConverter"
#define _OMC_LIT_RESOURCE_0_dir_data "/home/epl05/EMProject/buck_converter/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,13,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,45,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "Complex"
#define _OMC_LIT_RESOURCE_1_dir_data "/home/epl05/.openmodelica/libraries/Complex 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,7,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,58,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "Modelica"
#define _OMC_LIT_RESOURCE_2_dir_data "/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,8,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,59,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_3_dir_data "/home/epl05/.openmodelica/libraries/ModelicaServices 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,16,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,67,_OMC_LIT_RESOURCE_3_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,8,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir)}};
void BuckConverter_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &BuckConverter_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "BuckConverter";
  data->modelData->modelFilePrefix = "BuckConverter";
  data->modelData->modelFileName = "BuckConverter.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "/home/epl05/EMProject/buck_converter/modelica";
  data->modelData->modelGUID = "{cd558b59-bce1-45ee-b561-3fa86b462bce}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/BuckConverter_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStatesArray = 2;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesRealArray = 8;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersRealArray = 5;
  data->modelData->nParametersIntegerArray = 0;
  data->modelData->nParametersBooleanArray = 0;
  data->modelData->nParametersStringArray = 0;
  data->modelData->nParametersReal = 5;
  data->modelData->nParametersInteger = 0;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nAliasRealArray = 2;
  data->modelData->nAliasIntegerArray = 0;
  data->modelData->nAliasBooleanArray = 0;
  data->modelData->nAliasStringArray = 0;
  data->modelData->nInputVars = 1;
  data->modelData->nOutputVars = 3;
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 13;
  data->modelData->nMixedSystems = 0;
  data->modelData->nLinearSystems = 0;
  data->modelData->nNonLinearSystems = 0;
  data->modelData->nStateSets = 0;
  data->modelData->nJacobians = 6;
  data->modelData->nOptimizeConstraints = 0;
  data->modelData->nOptimizeFinalConstraints = 0;
  data->modelData->nDelayExpressions = 0;
  data->modelData->nBaseClocks = 0;
  data->modelData->nSpatialDistributions = 0;
  data->modelData->nSensitivityVars = 0;
  data->modelData->nSensitivityParamVars = 0;
  data->modelData->nSetcVars = 0;
  data->modelData->ndataReconVars = 0;
  data->modelData->nSetbVars = 0;
  data->modelData->nRelatedBoundaryConditions = 0;
  data->modelData->linearizationDumpLanguage = OMC_LINEARIZE_DUMP_LANGUAGE_MODELICA;
}

static int rml_execution_failed()
{
  fflush(NULL);
  fprintf(stderr, "Execution failed!\n");
  fflush(NULL);
  return 1;
}

