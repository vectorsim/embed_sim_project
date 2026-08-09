#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

#include "embed_sim_foc_types.h"
#include "embed_sim_control.h"
#include "embed_sim_motor_parameter.h"
#include <stdint.h>


extern void DFC_Init(void);
extern void DFC_Step(EmbedSimCtrlInput_T* const InputPtr,
                         const EmbedSimMachineParam_T* const MaschinePtr,
                         EmbedSimCtrlOutput_T* const OutputPtr);

#endif
