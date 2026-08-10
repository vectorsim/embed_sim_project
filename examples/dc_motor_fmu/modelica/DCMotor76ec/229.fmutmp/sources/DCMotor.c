/* Main Simulation File */

#if defined(__cplusplus)
extern "C" {
#endif

#include "DCMotor_model.h"
#include "simulation/solver/events.h"
#include "util/real_array.h"



/* dummy VARINFO and FILEINFO */
const VAR_INFO dummyVAR_INFO = omc_dummyVarInfo;

int DCMotor_input_function(DATA *data, threadData_t *threadData)
{
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* u variable */) = data->simulationInfo->inputVars[0];
  
  return 0;
}

int DCMotor_input_function_init(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[27].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  data->simulationInfo->inputVars[0] = real_get(data->modelData->realVarsData[27].attribute.start, 0);
  
  return 0;
}

int DCMotor_input_function_updateStartValues(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData, data->modelData->realVarsData[27].dimension.numberOfDimensions == 0, "Handling of array variables not yet implemetned.");
  put_real_element(data->simulationInfo->inputVars[0], 0, &data->modelData->realVarsData[27].attribute.start);
  
  return 0;
}

int DCMotor_inputNames(DATA *data, char ** names){
  names[0] = (char *) data->modelData->realVarsData[27].info.name;
  
  return 0;
}

int DCMotor_data_function(DATA *data, threadData_t *threadData)
{
  return 0;
}

int DCMotor_dataReconciliationInputNames(DATA *data, char ** names){
  
  return 0;
}

int DCMotor_dataReconciliationUnmeasuredVariables(DATA *data, char ** names)
{
  
  return 0;
}

int DCMotor_output_function(DATA *data, threadData_t *threadData)
{
  data->simulationInfo->outputVars[0] = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* w variable */);
  
  return 0;
}

int DCMotor_setc_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

int DCMotor_setb_function(DATA *data, threadData_t *threadData)
{
  
  return 0;
}


/*
equation index: 29
type: SIMPLE_ASSIGN
$DER.emf.phi = inertia.w
*/
void DCMotor_eqFunction_29(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,29};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[8]] /* der(emf.phi) DUMMY_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* inertia.w STATE(1,inertia.a) */);
  threadData->lastEquationSolved = 29;
}

/*
equation index: 30
type: SIMPLE_ASSIGN
emf.w = inertia.w
*/
void DCMotor_eqFunction_30(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,30};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[18]] /* emf.w variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* inertia.w STATE(1,inertia.a) */);
  threadData->lastEquationSolved = 30;
}

/*
equation index: 31
type: SIMPLE_ASSIGN
resistor.v = resistor.R_actual * inductor.i
*/
void DCMotor_eqFunction_31(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,31};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* resistor.v variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[24]] /* resistor.R_actual variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* inductor.i STATE(1) */));
  threadData->lastEquationSolved = 31;
}

/*
equation index: 32
type: SIMPLE_ASSIGN
resistor.LossPower = resistor.v * inductor.i
*/
void DCMotor_eqFunction_32(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,32};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[23]] /* resistor.LossPower variable */) = ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* resistor.v variable */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* inductor.i STATE(1) */));
  threadData->lastEquationSolved = 32;
}

/*
equation index: 33
type: SIMPLE_ASSIGN
resistor.n.v = u - resistor.v
*/
void DCMotor_eqFunction_33(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,33};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* resistor.n.v variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[27]] /* u variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[26]] /* resistor.v variable */);
  threadData->lastEquationSolved = 33;
}

/*
equation index: 34
type: SIMPLE_ASSIGN
emf.v = emf.k * inertia.w
*/
void DCMotor_eqFunction_34(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,34};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf.v variable */) = ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* emf.k PARAM */)) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* inertia.w STATE(1,inertia.a) */));
  threadData->lastEquationSolved = 34;
}

/*
equation index: 35
type: SIMPLE_ASSIGN
inductor.v = resistor.n.v - emf.v
*/
void DCMotor_eqFunction_35(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,35};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* inductor.v variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[25]] /* resistor.n.v variable */) - (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[17]] /* emf.v variable */);
  threadData->lastEquationSolved = 35;
}

/*
equation index: 36
type: SIMPLE_ASSIGN
$DER.inductor.i = inductor.v / inductor.L
*/
void DCMotor_eqFunction_36(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,36};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[5]] /* der(inductor.i) STATE_DER */) = DIVISION_SIM((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[21]] /* inductor.v variable */),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[7]] /* inductor.L PARAM */),"inductor.L",equationIndexes);
  threadData->lastEquationSolved = 36;
}

/*
equation index: 37
type: SIMPLE_ASSIGN
emf.tau = (-emf.k) * inductor.i
*/
void DCMotor_eqFunction_37(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,37};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* emf.tau variable */) = ((-(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[6]] /* emf.k PARAM */))) * ((data->localData[0]->realVars[data->simulationInfo->realVarsIndex[1]] /* inductor.i STATE(1) */));
  threadData->lastEquationSolved = 37;
}

/*
equation index: 38
type: SIMPLE_ASSIGN
inertia.a = (-emf.tau) / inertia.J
*/
void DCMotor_eqFunction_38(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,38};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* inertia.a variable */) = DIVISION_SIM((-(data->localData[0]->realVars[data->simulationInfo->realVarsIndex[16]] /* emf.tau variable */)),(data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[8]] /* inertia.J PARAM */),"inertia.J",equationIndexes);
  threadData->lastEquationSolved = 38;
}

/*
equation index: 39
type: SIMPLE_ASSIGN
$DER.inertia.w = inertia.a
*/
void DCMotor_eqFunction_39(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,39};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[7]] /* der(inertia.w) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[22]] /* inertia.a variable */);
  threadData->lastEquationSolved = 39;
}

/*
equation index: 40
type: SIMPLE_ASSIGN
$DER.inertia.phi = inertia.w
*/
void DCMotor_eqFunction_40(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,40};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[6]] /* der(inertia.phi) STATE_DER */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* inertia.w STATE(1,inertia.a) */);
  threadData->lastEquationSolved = 40;
}

/*
equation index: 41
type: SIMPLE_ASSIGN
damper.flange_b.phi = inertia.phi + damper.phi_rel
*/
void DCMotor_eqFunction_41(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,41};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[11]] /* damper.flange_b.phi variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* inertia.phi STATE(1,inertia.w) */) + (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[0]] /* damper.phi_rel STATE(1) */);
  threadData->lastEquationSolved = 41;
}

/*
equation index: 42
type: SIMPLE_ASSIGN
$DER.damper.phi_rel = 0.0
*/
void DCMotor_eqFunction_42(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,42};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[4]] /* der(damper.phi_rel) STATE_DER */) = 0.0;
  threadData->lastEquationSolved = 42;
}

/*
equation index: 43
type: SIMPLE_ASSIGN
w = inertia.w
*/
void DCMotor_eqFunction_43(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,43};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[28]] /* w variable */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[3]] /* inertia.w STATE(1,inertia.a) */);
  threadData->lastEquationSolved = 43;
}

/*
equation index: 44
type: SIMPLE_ASSIGN
emf.phi = inertia.phi - emf.fixed.phi0
*/
void DCMotor_eqFunction_44(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,44};
  (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[15]] /* emf.phi DUMMY_STATE */) = (data->localData[0]->realVars[data->simulationInfo->realVarsIndex[2]] /* inertia.phi STATE(1,inertia.w) */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[3]] /* emf.fixed.phi0 PARAM */);
  threadData->lastEquationSolved = 44;
}

/*
equation index: 45
type: ALGORITHM

  assert(1.0 + resistor.alpha * (resistor.T - resistor.T_ref) >= 2.220446049250313e-16, "Temperature outside scope of model!");
*/
void DCMotor_eqFunction_45(DATA *data, threadData_t *threadData)
{
  const int equationIndexes[2] = {1,45};
  modelica_boolean tmp0;
  static const MMC_DEFSTRINGLIT(tmp1,35,"Temperature outside scope of model!");
  static int tmp2 = 0;
  {
    tmp0 = GreaterEq(1.0 + ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[13]] /* resistor.alpha PARAM */)) * ((data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[10]] /* resistor.T PARAM */) - (data->simulationInfo->realParameter[data->simulationInfo->realParamsIndex[12]] /* resistor.T_ref PARAM */)),2.220446049250313e-16);
    if(!tmp0)
    {
      {
        const char* assert_cond = "(1.0 + resistor.alpha * (resistor.T - resistor.T_ref) >= 2.220446049250313e-16)";
        if (data->simulationInfo->noThrowAsserts) {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Basic/Resistor.mo",15,3,16,43,0};
          infoStreamPrintWithEquationIndexes(OMC_LOG_ASSERT, info, 0, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(MMC_REFSTRINGLIT(tmp1)));
          data->simulationInfo->needToReThrow = 1;
        } else {
          FILE_INFO info = {"/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om/Electrical/Analog/Basic/Resistor.mo",15,3,16,43,0};
          omc_assert_withEquationIndexes(threadData, info, equationIndexes, "The following assertion has been violated %sat time %f\n(%s) --> \"%s\"", initial() ? "during initialization " : "", data->localData[0]->timeValue, assert_cond, MMC_STRINGDATA(MMC_REFSTRINGLIT(tmp1)));
        }
      }
    }
  }
  threadData->lastEquationSolved = 45;
}

OMC_DISABLE_OPT
int DCMotor_functionDAE(DATA *data, threadData_t *threadData)
{
  int equationIndexes[1] = {0};
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_DAE);
#endif

  data->simulationInfo->needToIterate = 0;
  data->simulationInfo->discreteCall = 1;
  DCMotor_functionLocalKnownVars(data, threadData);
  static void (*const eqFunctions[17])(DATA*, threadData_t*) = {
    DCMotor_eqFunction_29,
    DCMotor_eqFunction_30,
    DCMotor_eqFunction_31,
    DCMotor_eqFunction_32,
    DCMotor_eqFunction_33,
    DCMotor_eqFunction_34,
    DCMotor_eqFunction_35,
    DCMotor_eqFunction_36,
    DCMotor_eqFunction_37,
    DCMotor_eqFunction_38,
    DCMotor_eqFunction_39,
    DCMotor_eqFunction_40,
    DCMotor_eqFunction_41,
    DCMotor_eqFunction_42,
    DCMotor_eqFunction_43,
    DCMotor_eqFunction_44,
    DCMotor_eqFunction_45
  };
  
  for (int id = 0; id < 17; id++) {
    eqFunctions[id](data, threadData);
  }
  data->simulationInfo->discreteCall = 0;
  
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_DAE);
#endif
  return 0;
}


int DCMotor_functionLocalKnownVars(DATA *data, threadData_t *threadData)
{
  
  return 0;
}

/* forwarded equations */
extern void DCMotor_eqFunction_31(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_33(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_34(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_35(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_36(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_37(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_38(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_39(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_40(DATA* data, threadData_t *threadData);
extern void DCMotor_eqFunction_42(DATA* data, threadData_t *threadData);

static void functionODE_system0(DATA *data, threadData_t *threadData)
{
  static void (*const eqFunctions[10])(DATA*, threadData_t*) = {
    DCMotor_eqFunction_31,
    DCMotor_eqFunction_33,
    DCMotor_eqFunction_34,
    DCMotor_eqFunction_35,
    DCMotor_eqFunction_36,
    DCMotor_eqFunction_37,
    DCMotor_eqFunction_38,
    DCMotor_eqFunction_39,
    DCMotor_eqFunction_40,
    DCMotor_eqFunction_42
  };
  
  for (int id = 0; id < 10; id++) {
    eqFunctions[id](data, threadData);
  }
}

int DCMotor_functionODE(DATA *data, threadData_t *threadData)
{
#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_tick(SIM_TIMER_FUNCTION_ODE);
#endif

  
  data->simulationInfo->callStatistics.functionODE++;
  
  DCMotor_functionLocalKnownVars(data, threadData);
  functionODE_system0(data, threadData);

#if !defined(OMC_MINIMAL_RUNTIME)
  if (measure_time_flag) rt_accumulate(SIM_TIMER_FUNCTION_ODE);
#endif

  return 0;
}

/* forward the main in the simulation runtime */
extern int _main_SimulationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);
extern int _main_OptimizationRuntime(int argc, char **argv, DATA *data, threadData_t *threadData);

#include "DCMotor_12jac.h"
#include "DCMotor_13opt.h"

struct OpenModelicaGeneratedFunctionCallbacks DCMotor_callback = {
  NULL,    /* performSimulation */
  NULL,    /* performQSSSimulation */
  NULL,    /* updateContinuousSystem */
  DCMotor_callExternalObjectDestructors,    /* callExternalObjectDestructors */
  NULL,    /* initialNonLinearSystem */
  NULL,    /* initialLinearSystem */
  NULL,    /* initialMixedSystem */
  #if !defined(OMC_NO_STATESELECTION)
  DCMotor_initializeStateSets,
  #else
  NULL,
  #endif    /* initializeStateSets */
  DCMotor_initializeDAEmodeData,
  DCMotor_functionODE,
  DCMotor_functionAlgebraics,
  DCMotor_functionDAE,
  DCMotor_functionLocalKnownVars,
  DCMotor_input_function,
  DCMotor_input_function_init,
  DCMotor_input_function_updateStartValues,
  DCMotor_data_function,
  DCMotor_output_function,
  DCMotor_setc_function,
  DCMotor_setb_function,
  DCMotor_function_storeDelayed,
  DCMotor_function_storeSpatialDistribution,
  DCMotor_function_initSpatialDistribution,
  DCMotor_updateBoundVariableAttributes,
  DCMotor_functionInitialEquations,
  GLOBAL_EQUIDISTANT_HOMOTOPY,
  NULL,
  DCMotor_functionRemovedInitialEquations,
  DCMotor_updateBoundParameters,
  DCMotor_checkForAsserts,
  DCMotor_function_ZeroCrossingsEquations,
  DCMotor_function_ZeroCrossings,
  DCMotor_function_updateRelations,
  DCMotor_zeroCrossingDescription,
  DCMotor_relationDescription,
  DCMotor_function_initSample,
  DCMotor_INDEX_JAC_A,
  DCMotor_INDEX_JAC_B,
  DCMotor_INDEX_JAC_C,
  DCMotor_INDEX_JAC_D,
  DCMotor_INDEX_JAC_F,
  DCMotor_INDEX_JAC_H,
  DCMotor_initialAnalyticJacobianA,
  DCMotor_initialAnalyticJacobianB,
  DCMotor_initialAnalyticJacobianC,
  DCMotor_initialAnalyticJacobianD,
  DCMotor_initialAnalyticJacobianF,
  DCMotor_initialAnalyticJacobianH,
  DCMotor_functionJacA_column,
  DCMotor_functionJacB_column,
  DCMotor_functionJacC_column,
  DCMotor_functionJacD_column,
  DCMotor_functionJacF_column,
  DCMotor_functionJacH_column,
  DCMotor_linear_model_frame,
  DCMotor_linear_model_datarecovery_frame,
  DCMotor_mayer,
  DCMotor_lagrange,
  DCMotor_getInputVarIndicesInOptimization,
  DCMotor_pickUpBoundsForInputsInOptimization,
  DCMotor_setInputData,
  DCMotor_getTimeGrid,
  DCMotor_symbolicInlineSystem,
  DCMotor_function_initSynchronous,
  DCMotor_function_updateSynchronous,
  DCMotor_function_equationsSynchronous,
  DCMotor_inputNames,
  DCMotor_dataReconciliationInputNames,
  DCMotor_dataReconciliationUnmeasuredVariables,
  DCMotor_read_simulation_info,
  DCMotor_read_input_fmu,
  NULL,
  NULL,
  -1,
  NULL,
  NULL,
  -1

};

#define _OMC_LIT_RESOURCE_0_name_data "Complex"
#define _OMC_LIT_RESOURCE_0_dir_data "/home/epl05/.openmodelica/libraries/Complex 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_name,7,_OMC_LIT_RESOURCE_0_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir,58,_OMC_LIT_RESOURCE_0_dir_data);

#define _OMC_LIT_RESOURCE_1_name_data "DCMotor"
#define _OMC_LIT_RESOURCE_1_dir_data "/home/epl05/EMProject/examples/dc_motor_fmu/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_name,7,_OMC_LIT_RESOURCE_1_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir,52,_OMC_LIT_RESOURCE_1_dir_data);

#define _OMC_LIT_RESOURCE_2_name_data "Modelica"
#define _OMC_LIT_RESOURCE_2_dir_data "/home/epl05/.openmodelica/libraries/Modelica 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_name,8,_OMC_LIT_RESOURCE_2_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir,59,_OMC_LIT_RESOURCE_2_dir_data);

#define _OMC_LIT_RESOURCE_3_name_data "ModelicaServices"
#define _OMC_LIT_RESOURCE_3_dir_data "/home/epl05/.openmodelica/libraries/ModelicaServices 4.1.0+maint.om"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_name,16,_OMC_LIT_RESOURCE_3_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir,67,_OMC_LIT_RESOURCE_3_dir_data);

#define _OMC_LIT_RESOURCE_4_name_data "PMSM_Plant_FMU"
#define _OMC_LIT_RESOURCE_4_dir_data "/home/epl05/EMProject/fs_electrical_machines/modelica"
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_4_name,14,_OMC_LIT_RESOURCE_4_name_data);
static const MMC_DEFSTRINGLIT(_OMC_LIT_RESOURCE_4_dir,53,_OMC_LIT_RESOURCE_4_dir_data);

static const MMC_DEFSTRUCTLIT(_OMC_LIT_RESOURCES,10,MMC_ARRAY_TAG) {MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_0_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_1_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_2_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_3_dir), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_4_name), MMC_REFSTRINGLIT(_OMC_LIT_RESOURCE_4_dir)}};
void DCMotor_setupDataStruc(DATA *data, threadData_t *threadData)
{
  assertStreamPrint(threadData,0!=data, "Error while initialize Data");
  threadData->localRoots[LOCAL_ROOT_SIMULATION_DATA] = data;
  data->callback = &DCMotor_callback;
  OpenModelica_updateUriMapping(threadData, MMC_REFSTRUCTLIT(_OMC_LIT_RESOURCES));
  data->modelData->modelName = "DCMotor";
  data->modelData->modelFilePrefix = "DCMotor";
  data->modelData->modelFileName = "DCMotor.mo";
  data->modelData->resultFileName = NULL;
  data->modelData->modelDir = "/home/epl05/EMProject/examples/dc_motor_fmu/modelica";
  data->modelData->modelGUID = "{a0a2d8a1-e76e-40b1-9282-53b3d9396dc1}";
  data->modelData->initXMLData = NULL;
  data->modelData->modelDataXml.infoXMLData = NULL;
  GC_asprintf(&data->modelData->modelDataXml.fileName, "%s/DCMotor_info.json", data->modelData->resourcesDir);
  data->modelData->runTestsuite = 0;
  data->modelData->nStatesArray = 4;
  data->modelData->nDiscreteReal = 0;
  data->modelData->nVariablesRealArray = 29;
  data->modelData->nVariablesIntegerArray = 0;
  data->modelData->nVariablesBooleanArray = 0;
  data->modelData->nVariablesStringArray = 0;
  data->modelData->nParametersRealArray = 14;
  data->modelData->nParametersIntegerArray = 2;
  data->modelData->nParametersBooleanArray = 3;
  data->modelData->nParametersStringArray = 0;
  data->modelData->nParametersReal = 14;
  data->modelData->nParametersInteger = 2;
  data->modelData->nParametersBoolean = 3;
  data->modelData->nParametersString = 0;
  data->modelData->nAliasRealArray = 32;
  data->modelData->nAliasIntegerArray = 0;
  data->modelData->nAliasBooleanArray = 0;
  data->modelData->nAliasStringArray = 0;
  data->modelData->nInputVars = 1;
  data->modelData->nOutputVars = 1;
  data->modelData->nZeroCrossings = 0;
  data->modelData->nSamples = 0;
  data->modelData->nRelations = 0;
  data->modelData->nMathEvents = 0;
  data->modelData->nExtObjs = 0;
  data->modelData->modelDataXml.modelInfoXmlLength = 0;
  data->modelData->modelDataXml.nFunctions = 0;
  data->modelData->modelDataXml.nProfileBlocks = 0;
  data->modelData->modelDataXml.nEquations = 74;
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

