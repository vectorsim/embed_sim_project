// define class name and unique id
#define MODEL_IDENTIFIER PMSM_Motor_WithSensors
#define MODEL_GUID "{33f30a05-f53c-4141-abb4-ba15d014c8ce}"

// include fmu header files, typedefs and macros
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "openmodelica.h"
#include "openmodelica_func.h"
#include "simulation_data.h"
#include "util/omc_error.h"
#include "PMSM_Motor_WithSensors_functions.h"
#include "simulation/solver/initialization/initialization.h"
#include "simulation/solver/events.h"
#define FMI2_FUNCTION_PREFIX PMSM_Motor_WithSensors_
#include "fmi2Functions.h"
#include "fmi-export/fmu2_model_interface.h"
#include "fmi-export/fmu_read_flags.h"

#ifdef __cplusplus
extern "C" {
#endif

void setStartValues(ModelInstance *comp);
void setDefaultStartValues(ModelInstance *comp);
void eventUpdate(ModelInstance* comp, fmi2EventInfo* eventInfo);
fmi2Real getReal(ModelInstance* comp, const fmi2ValueReference vr);
fmi2Status setReal(ModelInstance* comp, const fmi2ValueReference vr, const fmi2Real value);
fmi2Integer getInteger(ModelInstance* comp, const fmi2ValueReference vr);
fmi2Status setInteger(ModelInstance* comp, const fmi2ValueReference vr, const fmi2Integer value);
fmi2Boolean getBoolean(ModelInstance* comp, const fmi2ValueReference vr);
fmi2Status setBoolean(ModelInstance* comp, const fmi2ValueReference vr, const fmi2Boolean value);
fmi2String getString(ModelInstance* comp, const fmi2ValueReference vr);
fmi2Status setString(ModelInstance* comp, const fmi2ValueReference vr, fmi2String value);
fmi2Status setExternalFunction(ModelInstance* c, const fmi2ValueReference vr, const void* value);
fmi2ValueReference mapInputReference2InputNumber(const fmi2ValueReference vr);
fmi2ValueReference mapOutputReference2OutputNumber(const fmi2ValueReference vr);
fmi2ValueReference mapOutputReference2RealOutputDerivatives(const fmi2ValueReference vr);
fmi2ValueReference mapInitialUnknownsdependentIndex(const fmi2ValueReference vr);
fmi2ValueReference mapInitialUnknownsIndependentIndex(const fmi2ValueReference vr);

// define model size
#define NUMBER_OF_STATES 4
#define NUMBER_OF_EVENT_INDICATORS 0
#define NUMBER_OF_REALS 59
#define NUMBER_OF_REAL_INPUTS 5
#define NUMBER_OF_INTEGERS 0
#define NUMBER_OF_STRINGS 0
#define NUMBER_OF_BOOLEANS 0
#define NUMBER_OF_EXTERNALFUNCTIONS 0

// define initial state vector as vector of value references
#define STATES { 0, 1, 2, 3 }
#define STATESDERIVATIVES { 4, 5, 6, 7 }


// implementation of the Model Exchange functions
extern void PMSM_Motor_WithSensors_setupDataStruc(DATA *data, threadData_t *threadData);
#define fmu2_model_interface_setupDataStruc PMSM_Motor_WithSensors_setupDataStruc
#include "fmi-export/fmu2_model_interface.c.inc"
#include "fmi-export/fmu_read_flags.c.inc"

// Set values for all variables that define a start value
OMC_DISABLE_OPT
void setDefaultStartValues(ModelInstance *comp) {
  comp->fmuData->modelData->realVarsData[0].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[1].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[2].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[3].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[4].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[5].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[6].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[7].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[8].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[9].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[10].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[11].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[12].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[13].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[14].attribute.start = 0.5;
  comp->fmuData->modelData->realVarsData[15].attribute.start = 0.5;
  comp->fmuData->modelData->realVarsData[16].attribute.start = 0.5;
  comp->fmuData->modelData->realVarsData[17].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[18].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[19].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[20].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[21].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[22].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[23].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[24].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[25].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[26].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[27].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[28].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[29].attribute.start = 24.0;
  comp->fmuData->modelData->realVarsData[30].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[31].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[32].attribute.start = 24.0;
  comp->fmuData->modelData->realVarsData[33].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[34].attribute.start = 0.0;
  comp->fmuData->modelData->realVarsData[35].attribute.start = 24.0;
  comp->fmuData->modelData->realVarsData[36].attribute.start = 0;
  comp->fmuData->modelData->realVarsData[37].attribute.start = 48.0;
  comp->fmuData->modelData->realVarsData[38].attribute.start = 24.0;
  comp->fmuData->modelData->realVarsData[39].attribute.start = 0;
  comp->fmuData->modelData->realParameterData[0].attribute.start = 0.001;
  comp->fmuData->modelData->realParameterData[1].attribute.start = 0.002;
  comp->fmuData->modelData->realParameterData[2].attribute.start = 0.005;
  comp->fmuData->modelData->realParameterData[3].attribute.start = 0.006;
  comp->fmuData->modelData->realParameterData[4].attribute.start = 0.5;
  comp->fmuData->modelData->realParameterData[5].attribute.start = 0.175;
  comp->fmuData->modelData->realParameterData[6].attribute.start = 2.0;
}
// Set values for all variables that define a start value
OMC_DISABLE_OPT
void setStartValues(ModelInstance *comp) {
  comp->fmuData->modelData->realVarsData[0].attribute.start =  comp->fmuData->localData[0]->realVars[0];
  comp->fmuData->modelData->realVarsData[1].attribute.start =  comp->fmuData->localData[0]->realVars[1];
  comp->fmuData->modelData->realVarsData[2].attribute.start =  comp->fmuData->localData[0]->realVars[2];
  comp->fmuData->modelData->realVarsData[3].attribute.start =  comp->fmuData->localData[0]->realVars[3];
  comp->fmuData->modelData->realVarsData[4].attribute.start =  comp->fmuData->localData[0]->realVars[4];
  comp->fmuData->modelData->realVarsData[5].attribute.start =  comp->fmuData->localData[0]->realVars[5];
  comp->fmuData->modelData->realVarsData[6].attribute.start =  comp->fmuData->localData[0]->realVars[6];
  comp->fmuData->modelData->realVarsData[7].attribute.start =  comp->fmuData->localData[0]->realVars[7];
  comp->fmuData->modelData->realVarsData[8].attribute.start =  comp->fmuData->localData[0]->realVars[8];
  comp->fmuData->modelData->realVarsData[9].attribute.start =  comp->fmuData->localData[0]->realVars[9];
  comp->fmuData->modelData->realVarsData[10].attribute.start =  comp->fmuData->localData[0]->realVars[10];
  comp->fmuData->modelData->realVarsData[11].attribute.start =  comp->fmuData->localData[0]->realVars[11];
  comp->fmuData->modelData->realVarsData[12].attribute.start =  comp->fmuData->localData[0]->realVars[12];
  comp->fmuData->modelData->realVarsData[13].attribute.start =  comp->fmuData->localData[0]->realVars[13];
  comp->fmuData->modelData->realVarsData[14].attribute.start =  comp->fmuData->localData[0]->realVars[14];
  comp->fmuData->modelData->realVarsData[15].attribute.start =  comp->fmuData->localData[0]->realVars[15];
  comp->fmuData->modelData->realVarsData[16].attribute.start =  comp->fmuData->localData[0]->realVars[16];
  comp->fmuData->modelData->realVarsData[17].attribute.start =  comp->fmuData->localData[0]->realVars[17];
  comp->fmuData->modelData->realVarsData[18].attribute.start =  comp->fmuData->localData[0]->realVars[18];
  comp->fmuData->modelData->realVarsData[19].attribute.start =  comp->fmuData->localData[0]->realVars[19];
  comp->fmuData->modelData->realVarsData[20].attribute.start =  comp->fmuData->localData[0]->realVars[20];
  comp->fmuData->modelData->realVarsData[21].attribute.start =  comp->fmuData->localData[0]->realVars[21];
  comp->fmuData->modelData->realVarsData[22].attribute.start =  comp->fmuData->localData[0]->realVars[22];
  comp->fmuData->modelData->realVarsData[23].attribute.start =  comp->fmuData->localData[0]->realVars[23];
  comp->fmuData->modelData->realVarsData[24].attribute.start =  comp->fmuData->localData[0]->realVars[24];
  comp->fmuData->modelData->realVarsData[25].attribute.start =  comp->fmuData->localData[0]->realVars[25];
  comp->fmuData->modelData->realVarsData[26].attribute.start =  comp->fmuData->localData[0]->realVars[26];
  comp->fmuData->modelData->realVarsData[27].attribute.start =  comp->fmuData->localData[0]->realVars[27];
  comp->fmuData->modelData->realVarsData[28].attribute.start =  comp->fmuData->localData[0]->realVars[28];
  comp->fmuData->modelData->realVarsData[29].attribute.start =  comp->fmuData->localData[0]->realVars[29];
  comp->fmuData->modelData->realVarsData[30].attribute.start =  comp->fmuData->localData[0]->realVars[30];
  comp->fmuData->modelData->realVarsData[31].attribute.start =  comp->fmuData->localData[0]->realVars[31];
  comp->fmuData->modelData->realVarsData[32].attribute.start =  comp->fmuData->localData[0]->realVars[32];
  comp->fmuData->modelData->realVarsData[33].attribute.start =  comp->fmuData->localData[0]->realVars[33];
  comp->fmuData->modelData->realVarsData[34].attribute.start =  comp->fmuData->localData[0]->realVars[34];
  comp->fmuData->modelData->realVarsData[35].attribute.start =  comp->fmuData->localData[0]->realVars[35];
  comp->fmuData->modelData->realVarsData[36].attribute.start =  comp->fmuData->localData[0]->realVars[36];
  comp->fmuData->modelData->realVarsData[37].attribute.start =  comp->fmuData->localData[0]->realVars[37];
  comp->fmuData->modelData->realVarsData[38].attribute.start =  comp->fmuData->localData[0]->realVars[38];
  comp->fmuData->modelData->realVarsData[39].attribute.start =  comp->fmuData->localData[0]->realVars[39];
  comp->fmuData->modelData->realParameterData[0].attribute.start = comp->fmuData->simulationInfo->realParameter[0];
  comp->fmuData->modelData->realParameterData[1].attribute.start = comp->fmuData->simulationInfo->realParameter[1];
  comp->fmuData->modelData->realParameterData[2].attribute.start = comp->fmuData->simulationInfo->realParameter[2];
  comp->fmuData->modelData->realParameterData[3].attribute.start = comp->fmuData->simulationInfo->realParameter[3];
  comp->fmuData->modelData->realParameterData[4].attribute.start = comp->fmuData->simulationInfo->realParameter[4];
  comp->fmuData->modelData->realParameterData[5].attribute.start = comp->fmuData->simulationInfo->realParameter[5];
  comp->fmuData->modelData->realParameterData[6].attribute.start = comp->fmuData->simulationInfo->realParameter[6];
}

// Used to set the next time event, if any.
void eventUpdate(ModelInstance* comp, fmi2EventInfo* eventInfo) {
}

static const int realAliasIndexes[12] = {
  12, 17, 18, 19, 20, 21, 23, 2, 26, 27, 12, 20
};

fmi2Real getReal(ModelInstance* comp, const fmi2ValueReference vr) {
  if (vr < 40) {
    return comp->fmuData->localData[0]->realVars[vr];
  }
  if (vr < 47) {
    return comp->fmuData->simulationInfo->realParameter[vr-40];
  }
  if (vr < 59) {
    int ix = realAliasIndexes[vr-47];
    return ix>=0 ? getReal(comp, ix) : -getReal(comp, -(ix+1));
  }
  return NAN;
}

fmi2Status setReal(ModelInstance* comp, const fmi2ValueReference vr, const fmi2Real value) {
  // set start value attribute for all variable that has start value, till initialization mode
  if (vr < 40 && (comp->state == model_state_instantiated || comp->state == model_state_initialization_mode)) {
    comp->fmuData->modelData->realVarsData[vr].attribute.start = value;
  }
  if (vr < 40) {
    comp->fmuData->localData[0]->realVars[vr] = value;
    return fmi2OK;
  }
  if (vr < 47) {
    comp->fmuData->simulationInfo->realParameter[vr-40] = value;
    return fmi2OK;
  }
  if (vr < 59) {
    int ix = realAliasIndexes[vr-47];
    return ix >= 0 ? setReal(comp, ix, value) : setReal(comp, -(ix+1), -value);
  }
  return fmi2Error;
}

fmi2Integer getInteger(ModelInstance* comp, const fmi2ValueReference vr) {
  if (vr < 0) {
    return comp->fmuData->localData[0]->integerVars[vr];
  }
  if (vr < 0) {
    return comp->fmuData->simulationInfo->integerParameter[vr-0];
  }
  return 0;
}

fmi2Status setInteger(ModelInstance* comp, const fmi2ValueReference vr, const fmi2Integer value) {
  // set start value attribute for all variable that has start value, till initialization mode
  if (vr < 0 && (comp->state == model_state_instantiated || comp->state == model_state_initialization_mode)) {
    comp->fmuData->modelData->integerVarsData[vr].attribute.start = value;
  }
  if (vr < 0) {
    comp->fmuData->localData[0]->integerVars[vr] = value;
    return fmi2OK;
  }
  if (vr < 0) {
    comp->fmuData->simulationInfo->integerParameter[vr-0] = value;
    return fmi2OK;
  }
  return fmi2Error;
}
fmi2Boolean getBoolean(ModelInstance* comp, const fmi2ValueReference vr) {
  switch (vr) {
    default:
      return fmi2False;
  }
}

fmi2Status setBoolean(ModelInstance* comp, const fmi2ValueReference vr, const fmi2Boolean value) {
  switch (vr) {
    default:
      return fmi2Error;
  }
  return fmi2OK;
}

fmi2String getString(ModelInstance* comp, const fmi2ValueReference vr) {
  switch (vr) {
    default:
      return "";
  }
}

fmi2Status setString(ModelInstance* comp, const fmi2ValueReference vr, fmi2String value) {
  switch (vr) {
    default:
      return fmi2Error;
  }
  return fmi2OK;
}

fmi2Status setExternalFunction(ModelInstance* c, const fmi2ValueReference vr, const void* value){
  switch (vr) {
    default:
      return fmi2Error;
  }
  return fmi2OK;
}

/* function maps input references to a input index used in partialDerivatives */
fmi2ValueReference mapInputReference2InputNumber(const fmi2ValueReference vr) {
    switch (vr) {
      case 13: return 0; break;
      case 14: return 1; break;
      case 15: return 2; break;
      case 16: return 3; break;
      case 37: return 4; break;
      default:
        return -1;
    }
}
/* function maps output references to a input index used in partialDerivatives */
fmi2ValueReference mapOutputReference2OutputNumber(const fmi2ValueReference vr) {
    switch (vr) {
      case 12: return 0; break;
      case 17: return 1; break;
      case 18: return 2; break;
      case 19: return 3; break;
      case 20: return 4; break;
      case 21: return 5; break;
      case 23: return 6; break;
      case 25: return 7; break;
      case 26: return 8; break;
      case 27: return 9; break;
      default:
        return -1;
    }
}
/* function maps output references to an internal output Real derivatives */
fmi2ValueReference mapOutputReference2RealOutputDerivatives(const fmi2ValueReference vr) {
    switch (vr) {
      case 12: return -1; break;
      case 17: return -1; break;
      case 18: return -1; break;
      case 19: return -1; break;
      case 20: return -1; break;
      case 21: return -1; break;
      case 23: return -1; break;
      case 25: return -1; break;
      case 26: return -1; break;
      case 27: return -1; break;
      default:
        return -1;
    }
}
/* function maps initialUnknowns UnknownVars ValueReferences to an internal partial derivatives index */
fmi2ValueReference mapInitialUnknownsdependentIndex(const fmi2ValueReference vr) {
    switch (vr) {
      case 4: return 0; break;
      case 5: return 1; break;
      case 6: return 2; break;
      case 7: return 3; break;
      case 12: return 4; break;
      case 17: return 5; break;
      case 18: return 6; break;
      case 19: return 7; break;
      case 20: return 8; break;
      case 21: return 9; break;
      case 23: return 10; break;
      case 25: return 11; break;
      case 26: return 12; break;
      case 27: return 13; break;
      default:
        return -1;
    }
}
/* function maps initialUnknowns knownVars ValueReferences to an internal partial derivatives index */
fmi2ValueReference mapInitialUnknownsIndependentIndex(const fmi2ValueReference vr) {
    switch (vr) {
      case 0: return 0; break;
      case 1: return 1; break;
      case 2: return 2; break;
      case 3: return 3; break;
      case 13: return 4; break;
      case 14: return 5; break;
      case 15: return 6; break;
      case 16: return 7; break;
      case 37: return 8; break;
      case 40: return 9; break;
      case 41: return 10; break;
      case 42: return 11; break;
      case 43: return 12; break;
      case 44: return 13; break;
      case 45: return 14; break;
      case 46: return 15; break;
      default:
        return -1;
    }
}

#ifdef __cplusplus
}
#endif

