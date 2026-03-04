/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "ThreePhaseMotor_model.h"
#include "simulation/solver/events.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int ThreePhaseMotor_input_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* T_load variable */) = data->simulationInfo->inputVars[0];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* v_d variable */) = data->simulationInfo->inputVars[1];
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* v_q variable */) = data->simulationInfo->inputVars[2];
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_input_function_init(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->simulationInfo->inputVars[0] = data->modelData->realVarsData[9].attribute.start;
  data->simulationInfo->inputVars[1] = data->modelData->realVarsData[12].attribute.start;
  data->simulationInfo->inputVars[2] = data->modelData->realVarsData[13].attribute.start;
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  data->modelData->realVarsData[9].attribute.start = data->simulationInfo->inputVars[0];
  data->modelData->realVarsData[12].attribute.start = data->simulationInfo->inputVars[1];
  data->modelData->realVarsData[13].attribute.start = data->simulationInfo->inputVars[2];
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_inputNames(DATA *data, char ** names){
  TRACE_PUSH

  names[0] = (char *) data->modelData->realVarsData[9].info.name;
  names[1] = (char *) data->modelData->realVarsData[12].info.name;
  names[2] = (char *) data->modelData->realVarsData[13].info.name;
  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_data_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  TRACE_POP
  return 0;
}

int ThreePhaseMotor_dataReconciliationInputNames(DATA *data, char ** names){
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_output_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_setc_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

int ThreePhaseMotor_setb_function(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}


/*
equation index: 12
type: SIMPLE_ASSIGN
speed_rpm = 9.549296585513721 * omega_m
*/
void ThreePhaseMotor_eqFunction_12(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,12};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* speed_rpm variable */) = (9.549296585513721) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  TRACE_POP
}
/*
equation index: 13
type: SIMPLE_ASSIGN
T_em = 1.5 * p * i_q * (lambda_pm + (L_d - L_q) * i_d)
*/
void ThreePhaseMotor_eqFunction_13(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,13};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* T_em variable */) = (1.5) * (((data->simulationInfo->realParameter[6] /* p PARAM */)) * (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) * ((data->simulationInfo->realParameter[5] /* lambda_pm PARAM */) + ((data->simulationInfo->realParameter[2] /* L_d PARAM */) - (data->simulationInfo->realParameter[3] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)))));
  TRACE_POP
}
/*
equation index: 14
type: SIMPLE_ASSIGN
$DER.omega_m = (T_em + (-B) * omega_m - T_load) / J
*/
void ThreePhaseMotor_eqFunction_14(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,14};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(omega_m) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* T_em variable */) + ((-(data->simulationInfo->realParameter[0] /* B PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */)) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[9]] /* T_load variable */),(data->simulationInfo->realParameter[1] /* J PARAM */),"J",equationIndexes);
  TRACE_POP
}
/*
equation index: 15
type: SIMPLE_ASSIGN
omega_e = p * omega_m
*/
void ThreePhaseMotor_eqFunction_15(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,15};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* omega_e variable */) = ((data->simulationInfo->realParameter[6] /* p PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* omega_m STATE(1) */));
  TRACE_POP
}
/*
equation index: 16
type: SIMPLE_ASSIGN
$DER.i_d = (v_d + omega_e * L_q * i_q - R * i_d) / L_d
*/
void ThreePhaseMotor_eqFunction_16(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,16};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(i_d) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[12]] /* v_d variable */) + ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[3] /* L_q PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */))) - (((data->simulationInfo->realParameter[4] /* R PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */))),(data->simulationInfo->realParameter[2] /* L_d PARAM */),"L_d",equationIndexes);
  TRACE_POP
}
/*
equation index: 17
type: SIMPLE_ASSIGN
$DER.i_q = (v_q + (-R) * i_q - omega_e * (L_d * i_d + lambda_pm)) / L_q
*/
void ThreePhaseMotor_eqFunction_17(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,17};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(i_q) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[13]] /* v_q variable */) + ((-(data->simulationInfo->realParameter[4] /* R PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* i_q STATE(1) */)) - (((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* omega_e variable */)) * (((data->simulationInfo->realParameter[2] /* L_d PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* i_d STATE(1) */)) + (data->simulationInfo->realParameter[5] /* lambda_pm PARAM */))),(data->simulationInfo->realParameter[3] /* L_q PARAM */),"L_q",equationIndexes);
  TRACE_POP
}
/*
equation index: 18
type: SIMPLE_ASSIGN
$DER.theta_e = omega_e
*/
void ThreePhaseMotor_eqFunction_18(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  const int equationIndexes[2] = {1,18};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(theta_e) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[10]] /* omega_e variable */);
  TRACE_POP
}

OMC_DISABLE_OPT
int ThreePhaseMotor_functionDAE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  ThreePhaseMotor_functionLocalKnownVars(data, threadData);
  ThreePhaseMotor_eqFunction_12(data, threadData);

  ThreePhaseMotor_eqFunction_13(data, threadData);

  ThreePhaseMotor_eqFunction_14(data, threadData);

  ThreePhaseMotor_eqFunction_15(data, threadData);

  ThreePhaseMotor_eqFunction_16(data, threadData);

  ThreePhaseMotor_eqFunction_17(data, threadData);

  ThreePhaseMotor_eqFunction_18(data, threadData);
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  TRACE_POP
  return 0;
}


int ThreePhaseMotor_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH

  
  TRACE_POP
  return 0;
}

/* forwarded equations */
extern void ThreePhaseMotor_eqFunction_13(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_eqFunction_14(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_eqFunction_15(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_eqFunction_16(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_eqFunction_17(DATA* data, threadData_t *threadData);
extern void ThreePhaseMotor_eqFunction_18(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  int id;

  static void (*const eqFunctions[6])(DATA*, threadData_t*) = {
    ThreePhaseMotor_eqFunction_13,
    ThreePhaseMotor_eqFunction_14,
    ThreePhaseMotor_eqFunction_15,
    ThreePhaseMotor_eqFunction_16,
    ThreePhaseMotor_eqFunction_17,
    ThreePhaseMotor_eqFunction_18
  };
  
  static const int eqIndices[6] = {
    13,
    14,
    15,
    16,
    17,
    18
  };
  
  for (id = 0; id < 6; id++) {
    eqFunctions[id](data, threadData);
    threadData->lastEquationSolved = eqIndices[id];
  }
}

int ThreePhaseMotor_functionODE(DATA *data, threadData_t *threadData)
{
  TRACE_PUSH
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  ThreePhaseMotor_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  TRACE_POP
  return 0;
}

void ThreePhaseMotor_computeVarIndices(size_t* realIndex, size_t* integerIndex, size_t* booleanIndex, size_t* stringIndex)
{
  TRACE_PUSH

  size_t i_real = 0;
  size_t i_integer = 0;
  size_t i_boolean = 0;
  size_t i_string = 0;

  realIndex[0] = 0;
  integerIndex[0] = 0;
  booleanIndex[0] = 0;
  stringIndex[0] = 0;

  /* stateVars */
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_d STATE(1) */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* i_q STATE(1) */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* omega_m STATE(1) */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* theta_e STATE(1,omega_e) */
  
  /* derivativeVars */
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(i_d) STATE_DER */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(i_q) STATE_DER */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(omega_m) STATE_DER */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* der(theta_e) STATE_DER */
  
  /* algVars */
  realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* T_em variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* T_load variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* omega_e variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* speed_rpm variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_d variable */realIndex[i_real+1] = realIndex[i_real] + ((modelica_integer) 1); i_real++;  /* v_q variable */
  
  /* discreteAlgVars */
  
  /* realOptimizeConstraintsVars */
  
  /* realOptimizeFinalConstraintsVars */
  
  
  /* intAlgVars */
  
  /* boolAlgVars */
  
  /* stringAlgVars */
  
  TRACE_POP
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char**argv, DATA *data, threadData_t *threadData);

#include "ThreePhaseMotor_12jac.h"
#include "ThreePhaseMotor_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks ThreePhaseMotor_callback = {
   NULL,    /* performSimulation */
   NULL,    /* performQSSSimulation */
   NULL,    /* updateContinuousSystem */
   ThreePhaseMotor_callExternalObjectDestructors,    /* callExternalObjectDestructors */
   NULL,    /* initialNonLinearSystem */
   NULL,    /* initialLinearSystem */
   NULL,    /* initialMixedSystem */
   #if !defined(OMC_NO_STATESELECTION)
   ThreePhaseMotor_initializeStateSets,
   #else
   NULL,
   #endif    /* initializeStateSets */
   ThreePhaseMotor_initializeDAEmodeData,
   ThreePhaseMotor_computeVarIndices,
   ThreePhaseMotor_functionODE,
   ThreePhaseMotor_functionAlgebraics,
   ThreePhaseMotor_functionDAE,
   ThreePhaseMotor_functionLocalKnownVars,
   ThreePhaseMotor_input_function,
   ThreePhaseMotor_input_function_init,
   ThreePhaseMotor_input_function_updateStartValues,
   ThreePhaseMotor_data_function,
   ThreePhaseMotor_output_function,
   ThreePhaseMotor_setc_function,
   ThreePhaseMotor_setb_function,
   ThreePhaseMotor_function_storeDelayed,
   ThreePhaseMotor_function_storeSpatialDistribution,
   ThreePhaseMotor_function_initSpatialDistribution,
   ThreePhaseMotor_updateBoundVariableAttributes,
   ThreePhaseMotor_functionInitialEquations,
   1, /* useHomotopy - 0: local homotopy (equidistant lambda), 1: global homotopy (equidistant lambda), 2: new global homotopy approach (adaptive lambda), 3: new local homotopy approach (adaptive lambda)*/
   NULL,
   ThreePhaseMotor_functionRemovedInitialEquations,
   ThreePhaseMotor_updateBoundParameters,
   ThreePhaseMotor_checkForAsserts,
   ThreePhaseMotor_function_ZeroCrossingsEquations,
   ThreePhaseMotor_function_ZeroCrossings,
   ThreePhaseMotor_function_updateRelations,
   ThreePhaseMotor_zeroCrossingDescription,
   ThreePhaseMotor_relationDescription,
   ThreePhaseMotor_function_initSample,
   ThreePhaseMotor_INDEX_JAC_A,
   ThreePhaseMotor_INDEX_JAC_B,
   ThreePhaseMotor_INDEX_JAC_C,
   ThreePhaseMotor_INDEX_JAC_D,
   ThreePhaseMotor_INDEX_JAC_F,
   ThreePhaseMotor_INDEX_JAC_H,
   ThreePhaseMotor_initialAnalyticJacobianA,
   ThreePhaseMotor_initialAnalyticJacobianB,
   ThreePhaseMotor_initialAnalyticJacobianC,
   ThreePhaseMotor_initialAnalyticJacobianD,
   ThreePhaseMotor_initialAnalyticJacobianF,
   ThreePhaseMotor_initialAnalyticJacobianH,
   ThreePhaseMotor_functionJacA_column,
   ThreePhaseMotor_functionJacB_column,
   ThreePhaseMotor_functionJacC_column,
   ThreePhaseMotor_functionJacD_column,
   ThreePhaseMotor_functionJacF_column,
   ThreePhaseMotor_functionJacH_column,
   ThreePhaseMotor_linear_model_frame,
   ThreePhaseMotor_linear_model_datarecovery_frame,
   ThreePhaseMotor_mayer,
   ThreePhaseMotor_lagrange,
   ThreePhaseMotor_pickUpBoundsForInputsInOptimization,
   ThreePhaseMotor_setInputData,
   ThreePhaseMotor_getTimeGrid,
   ThreePhaseMotor_symbolicInlineSystem,
   ThreePhaseMotor_function_initSynchronous,
   ThreePhaseMotor_function_updateSynchronous,
   ThreePhaseMotor_function_equationsSynchronous,
   ThreePhaseMotor_inputNames,
   ThreePhaseMotor_dataReconciliationInputNames,
   ThreePhaseMotor_dataReconciliationUnmeasuredVariables,
   ThreePhaseMotor_read_simulation_info,
   ThreePhaseMotor_read_input_fmu,
   NULL,
   NULL,
   -1,
   NULL,
   NULL,
   -1

};

#define _OMC_LIT_RESOURCE_0_name_data "Complex"
#define _OMC_LIT_RESOURCE_0_dir_data "C:/Users/paul/AppData/Roaming/.openmodelica/libraries/Complex 4.0.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,7,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,76,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "Modelica"
#define _OMC_LIT_RESOURCE_1_dir_data "C:/Users/paul/AppData/Roaming/.openmodelica/libraries/Modelica 4.0.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,8,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,77,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_2_dir_data "C:/Users/paul/AppData/Roaming/.openmodelica/libraries/ModelicaServices 4.0.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,16,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,85,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "ThreePhaseMotor"
#define _OMC_LIT_RESOURCE_3_dir_data "C:/EmbedSimProject/electrical_blocks/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,15,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,45,_OMC_LIT_RESOURCE_3_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,8,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir)}};
void ThreePhaseMotor_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &ThreePhaseMotor_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "ThreePhaseMotor";
  data->modelData->modelFilePrefix = "ThreePhaseMotor";
  data->modelData->modelFileName = "ThreePhaseMotor.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "C:/EmbedSimProject/electrical_blocks/modelica";
  data->modelData->modelGUID = "{40f3f8f5-30e5-403b-8cbc-8b66771bae7e}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/ThreePhaseMotor_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStates = 4;
  data->modelData->nVariablesRealArray = 14;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersReal = 7;
  data->modelData->nParametersInteger = 0;
  data->modelData->nParametersBoolean = 0;
  data->modelData->nParametersString = 0;
  data->modelData->nInputVars = 3;
  data->modelData->nOutputVars = 0;
  data->modelData->nAliasReal = 0;
  data->modelData->nAliasInteger = 0;
  data->modelData->nAliasBoolean = 0;
  data->modelData->nAliasString = 0;
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 19;
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

