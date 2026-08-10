#ifndef PMSM_Plant_FMU_FMU_H
#define PMSM_Plant_FMU_FMU_H

#include "simulation_data.h"

// define class name and unique id
#define MODEL_IDENTIFIER PMSM_Plant_FMU
#define MODEL_GUID "{2e628212-0a6d-4e07-91c8-893359e9ea45}"

// define model size
#define NUMBER_OF_STATES 4
#define NUMBER_OF_EVENT_INDICATORS 0
#define NUMBER_OF_REALS 52
#define NUMBER_OF_REAL_INPUTS 5
#define NUMBER_OF_INTEGERS 1
#define NUMBER_OF_STRINGS 0
#define NUMBER_OF_BOOLEANS 0
#define NUMBER_OF_EXTERNALFUNCTIONS 0

// define initial state vector as vector of value references
#define STATES { 0, 1, 2, 3 }
#define STATESDERIVATIVES { 4, 5, 6, 7 }


#ifdef __cplusplus
extern "C" {
#endif

extern void PMSM_Plant_FMU_setupDataStruc(DATA *data, threadData_t *threadData);
#define fmu2_model_interface_setupDataStruc PMSM_Plant_FMU_setupDataStruc
#include "fmi-export/fmu2_model_interface.h"
#include "fmi-export/fmu_read_flags.h"

#define FMI2_FUNCTION_PREFIX PMSM_Plant_FMU_
#include "fmi2Functions.h"
#include "fmi-export/fmu2_model_interface.h"
#include "fmi-export/fmu_read_flags.h"

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


#ifdef __cplusplus
}
#endif

#endif /* PMSM_Plant_FMU_FMU_H */