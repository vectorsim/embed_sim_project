/**********************************************************************************************************************
 * \file      embed_sim_cython_interface.c
 * \brief     Top-level PMSM control module with DFC controller.
 *
 * \details   Defines the main control structures and functions for permanent magnet
 *            synchronous motors (PMSM). Supports open-loop and DFC control modes
 *            with smooth reference trajectory generation.
 *            Targets 32-bit MCUs (Infineon AURIX TriCore, ARM Cortex-M4).
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier
 *              - Rule  8.6 : No definitions in header files
 *              - Rule 17.2 : No recursion
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCase_P
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.0.0
 * \date      2026-08-12
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#include "embed_sim_cython_interface.h"
#include "embed_sim_control.h"
#include <stddef.h>

/******************************************************************************
 *--------------------------------------Public Functions-----------------------
 ******************************************************************************/

/**
 * \brief   Cython interface initialization
 *
 * \details Wrapper function for Python/Cython interface to initialize
 *          the control module.
 *
 * \return  void
 */
void EmbedSim_CythonControlInit(void)
{
    EmbedSim_ControlInit();
}


/**
 * \brief   Cython interface control step
 *
 * \details Wrapper function for Python/Cython interface to execute one
 *          control step with direct parameter passing.
 *
 * \param[in]  Iu                      Phase U current [A]
 * \param[in]  Iv                      Phase V current [A]
 * \param[in]  Iw                      Phase W current [A]
 * \param[in]  RotorPositionSensor     Rotor position from sensor [rad]
 * \param[in]  RotorVelocitySensor     Rotor speed from sensor [RPM]
 * \param[in]  AngularVelocityRefRpm   Speed reference [RPM]
 * \param[in]  Vdc                     DC bus voltage [V]
 * \param[in]  SampleTime              Sample time [s]
 * \param[in]  CtrlAlg                 Control algorithm selection
 * \param[in]  ValidIn                 Input validity flag
 * \param[out] PwmU                    Phase U PWM duty cycle [0-1]
 * \param[out] PwmV                    Phase V PWM duty cycle [0-1]
 * \param[out] PwmW                    Phase W PWM duty cycle [0-1]
 * \param[out] ValidOut                Output validity flag
 *
 * \return  void
 */
void EmbedSim_CythonControlStep(
    float Iu_P,
    float Iv_P,
    float Iw_P,
    float RotorPositionSensor,
    float RotorVelocitySensor,
    float AngularVelocityRefRpm,
    float Vdc,
    float SampleTime,
    unsigned int CtrlAlg,
    unsigned int ValidIn,
    float* const PwmU_P,
    float* const PwmV_P,
    float* const PwmW_P,
    unsigned int * const ValidOut_P)
{
    EmbedSimCtrlInput_T *inputPtr;
    EmbedSimCtrlOutput_T *outputPtr;

    inputPtr = TractionMotor_G.InputPtr;
    outputPtr = TractionMotor_G.OutputPtr;

    /* Populate Input Structure */
    inputPtr->Iu = Iu_P;
    inputPtr->Iv = Iv_P;
    inputPtr->Iw = Iw_P;
    inputPtr->RotorPositionSensorM   = RotorPositionSensor;
    inputPtr->RotorSpeedSensorM      = RotorVelocitySensor;
    inputPtr->AngularVelocityRefRpmM = AngularVelocityRefRpm;
    inputPtr->Vdc                    = Vdc;
    inputPtr->SampleTime             = SampleTime;
    inputPtr->CtrlAlg                = CtrlAlg;
    inputPtr->Valid                  = ValidIn;

    /*  Execute main control */
    EmbedSim_ControlStep(&TractionMotor_G);

    /* Populate Output Structure */
    *PwmU_P     = outputPtr->DutyU;
    *PwmV_P     = outputPtr->DutyV;
    *PwmW_P     = outputPtr->DutyW;
    *ValidOut_P =outputPtr->Valid;
}

/**
 * \brief   Get motor state for unified reporting via Cython
 *
 * \details Returns the current motor state structure filled with values
 *          from the control system. This provides a unified view of motor
 *          operation for display and logging in Python.
 *
 * \param[out] statePtr  Pointer to state structure to fill
 *
 * \return  void
 */
void EmbedSim_CythonGetMotorState(EmbedSimMotorState_T * const StatePtr)
{
    EmbedSim_GetMotorState(&TractionMotor_G, StatePtr);

}
