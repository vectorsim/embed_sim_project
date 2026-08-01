/**********************************************************************************************************************
 * \file        embed_sim_compiler.h
 * \brief       AUTOSAR compiler abstraction macros for EmbedSim CDD layer.
 *
 * \details
 *
 * \version     1.0.0
 * \date        2025-05-24
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBEDSIM_EMBED_SIM_CONTROL_H_
#define EMBEDSIM_EMBED_SIM_CONTROL_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Input Structure for Control Loop
 */
typedef struct
{
    real32_T    SpeedRefRpm;    /**< Mechanical speed reference [RPM] */
    real32_T    DutyU;          /**< Phase U PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyV;          /**< Phase V PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyW;          /**< Phase W PWM duty cycle [0.0 .. 1.0] */
    real32_T    Iu;             /**< Current Phase U [A] */
    real32_T    Iv;             /**< Current Phase V [A] */
    real32_T    Iw;             /**< Current Phase W [A] */
    real32_T    RotorSpeed;     /**< Rotor Velocity in RPM [RPM] */
    real32_T    SampleTime;     /**< Control loop sample time [s] */
    real32_T    RotorPosition;  /**< Rotor Position [RAD] */
    uint32_T    Valid;          /**< Flag of Validation */
} EmbedSimCtrlInput_T;

/**
 * \brief  Output Structure for Control Loop
 */
typedef struct
{
    real32_T    DutyU;          /**< Phase U PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyV;          /**< Phase V PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyW;          /**< Phase W PWM duty cycle [0.0 .. 1.0] */
    uint32_T    SvmSector;      /**< SVM Sector */
    real32_T    RotorSpeed;     /**< Rotor Velocity in RPM [RPM] */
    real32_T    RotorPosition;  /**< Rotor Position [RAD] */
    uint32_T    Valid;          /**< Flag of Validation */
} EmbedSimCtrlOutput_T;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation.
 */
extern void EmbedSim_ControlInit(void);

/**
 * \brief   Top-level PMSM control step.
 */
extern void EmbedSim_ControlStep(EmbedSimCtrlInput_T* InputPtr, EmbedSimCtrlOutput_T* OutputPtr);

/**
 * \brief   Estimate RPM from phase currents.
 */
extern real32_T EmbedSim_EstimateRpm(EmbedSimCtrlInput_T* InputPtr);

#endif /* EMBEDSIM_EMBED_SIM_CONTROL_H_ */
