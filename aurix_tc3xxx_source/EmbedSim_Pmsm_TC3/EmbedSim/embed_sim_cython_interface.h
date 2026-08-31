/**********************************************************************************************************************
 * \file      embed_sim_cython_interface.h
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

#ifndef EMBED_SIM_CYTHON_INTERFACE_H
#define EMBED_SIM_CYTHON_INTERFACE_H

#include "embed_sim_control.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the EmbedSim controller for Cython/Python.
 */
extern void EmbedSim_CythonControlInit(void);

/**
 * @brief Execute one control cycle from Cython/Python.
 *
 * @param[in]  Iu_P                  Phase-U current.
 * @param[in]  Iv_P                  Phase-V current.
 * @param[in]  Iw_P                  Phase-W current.
 * @param[in]  RotorPositionSensor   Rotor mechanical position [rad].
 * @param[in]  RotorVelocitySensor   Rotor speed [RPM].
 * @param[in]  AngularVelocityRefRpm  Speed reference [RPM].
 * @param[in]  Vdc                   DC bus voltage [V].
 * @param[in]  SampleTime            Control sample time [s].
 * @param[in]  CtrlAlg               Controller algorithm selector.
 * @param[in]  ValidIn               Input validity flag.
 * @param[out] PwmU_P                PWM duty U.
 * @param[out] PwmV_P                PWM duty V.
 * @param[out] PwmW_P                PWM duty W.
 * @param[out] ValidOut_P             Output validity flag.
 */
extern void EmbedSim_CythonControlStep(
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
    float * const PwmU_P,
    float * const PwmV_P,
    float * const PwmW_P,
    unsigned int * const ValidOut_P
);

/**
 * @brief Obtain the complete motor state.
 *
 * @param[out] StatePtr Pointer to motor state structure.
 */
extern void EmbedSim_CythonGetMotorState(EmbedSimMotorState_T * const StatePtr);

#ifdef __cplusplus
}
#endif

#endif /* EMBED_SIM_CYTHON_INTERFACE_H */
