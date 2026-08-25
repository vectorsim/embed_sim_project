/**********************************************************************************************************************
 * \file      embed_sim_cython_interface.h
 * \brief     Cython interface for EmbedSim motor control library.
 *
 * \details   Provides C-callable wrapper functions for Python/Cython integration.
 *            This allows the motor control library to be called from Python
 *            applications for simulation, testing, and rapid prototyping.
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
 * \date      2026-08-22
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBEDSIM_EMBED_SIM_CYTHON_INTERFACE_H_
#define EMBEDSIM_EMBED_SIM_CYTHON_INTERFACE_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"
#include "embed_sim_control.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/* No public macros required for this interface */

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/* No public data structures required for this interface */

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize Cython interface
 *
 * \details Initializes the motor control module for use with Python/Cython.
 *          This must be called once before any control step is executed.
 *          Wraps the EmbedSim_ControlInit() function for external use.
 *
 * \return  void
 */
extern void EmbedSim_CythonControlInit(void);

/**
 * \brief   Execute one control step via Cython interface
 *
 * \details Performs one step of motor control with parameters passed
 *          directly as arguments. This function is designed to be called
 *          from Python/Cython with minimal overhead.
 *
 *          Control flow:
 *            1. Copy input parameters to global control structure
 *            2. Execute one control step
 *            3. Copy output parameters back to caller
 *
 * \param[in]  Iu                      Phase U current [A]
 * \param[in]  Iv                      Phase V current [A]
 * \param[in]  Iw                      Phase W current [A]
 * \param[in]  RotorPositionSensor     Rotor electrical position from sensor [rad]
 * \param[in]  RotorVelocitySensor     Rotor mechanical speed from sensor [RPM]
 * \param[in]  AngularVelocityRefRpm   Desired mechanical speed reference [RPM]
 * \param[in]  Vdc                     DC bus voltage [V]
 * \param[in]  SampleTime              Control loop sample time [s]
 * \param[in]  CtrlAlg                 Control algorithm selection (0=Open-loop, 1=DFC)
 * \param[in]  ValidIn                 Input validity flag (0x1 = valid)
 * \param[out] PwmU                    Phase U PWM duty cycle [0.0 .. 1.0]
 * \param[out] PwmV                    Phase V PWM duty cycle [0.0 .. 1.0]
 * \param[out] PwmW                    Phase W PWM duty cycle [0.0 .. 1.0]
 * \param[out] ValidOut                Output validity flag (0x1 = valid)
 *
 * \return  void
 */
extern void EmbedSim_CythonControlStep(
    /* Input parameters */
    real32_T  Iu,                      /**< Phase U current [A] */
    real32_T  Iv,                      /**< Phase V current [A] */
    real32_T  Iw,                      /**< Phase W current [A] */
    real32_T  RotorPositionSensor,     /**< Rotor position [rad] */
    real32_T  RotorVelocitySensor,     /**< Rotor speed [RPM] */
    real32_T  AngularVelocityRefRpm,   /**< Speed reference [RPM] */
    real32_T  Vdc,                     /**< DC bus voltage [V] */
    real32_T  SampleTime,              /**< Sample time [s] */
    uint32_T  CtrlAlg,                 /**< Control algorithm selection */
    uint32_T  ValidIn,                 /**< Input validity flag */
    /* Output parameters */
    real32_T* PwmU,                    /**< Phase U PWM duty cycle [0-1] */
    real32_T* PwmV,                    /**< Phase V PWM duty cycle [0-1] */
    real32_T* PwmW,                    /**< Phase W PWM duty cycle [0-1] */
    uint32_T* ValidOut                 /**< Output validity flag */
);

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
extern void EmbedSim_CythonGetMotorState(EmbedSimMotorState_T* const statePtr);

#endif /* EMBEDSIM_EMBED_SIM_CYTHON_INTERFACE_H_ */