/**********************************************************************************************************************
 * \file        embed_sim_control.h
 * \brief       Top-level PMSM control module
 *
 * \details
 *
 * \version     1.1.0
 * \date        2026-08-07
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBEDSIM_EMBED_SIM_CONTROL_H_
#define EMBEDSIM_EMBED_SIM_CONTROL_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"
#include "embed_sim_matrix.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief  Conversion: RPM to rad/s */
#define CON_RPM_TO_RAD(RPM)             ((RPM * ES_MATH_2PI_F) / 60.0F)

/** \brief  Conversion: rad/s to RPM */
#define CON_RAD_TO_RPM(RAD)             ((RAD * 60.0F) / ES_MATH_2PI_F)

/** \brief  Maximum speed in RPM */
#define MAX_SPEED_RPM                   (3000.0F)


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Control algorithm selection
 */
typedef enum
{
    SIM_CTRL_OPEN_LOOP  = 0x0U,
    SIM_CTRL_DFC        = 0x1U
} EmbedSimCtrl_T;

/**
 * \brief  Input Structure for Control Loop
 */
typedef struct
{
    real32_T    AngularVelocityRefRpm;       /**< Mechanical speed reference [RPM]      */
    real32_T    RotorPositionRef;            /**< Rotor position reference [rad]        */
    real32_T    AngularVelocityRef;          /**< Angular velocity [rad/s]              */
    real32_T    AngularAccerlerationRef;     /**< Angular acceleration [rad/s²]         */
    real32_T    AngularJerkRef;              /**< Angular jerk [rad/s³]                 */

    real32_T    DutyU;                      /**< Phase U PWM duty cycle [0.0 .. 1.0]    */
    real32_T    DutyV;                      /**< Phase V PWM duty cycle [0.0 .. 1.0]    */
    real32_T    DutyW;                      /**< Phase W PWM duty cycle [0.0 .. 1.0]    */
    real32_T    Iu;                         /**< Current Phase U [A]                    */
    real32_T    Iv;                         /**< Current Phase V [A]                    */
    real32_T    Iw;                         /**< Current Phase W [A]                    */
    real32_T    RotorSpeedSensor;           /**< Rotor Velocity in RPM [RPM] (Sensor)   */
    real32_T    RotorSpeedEst;              /**< Rotor Velocity in RPM [RPM] (Estimated)*/
    real32_T    SampleTime;                 /**< Control loop sample time [s]           */
    real32_T    RotorPositionSensor;        /**< Rotor Position [RAD] (Sensor)          */
    real32_T    RotorPositionEst;           /**< Rotor Position [RAD] (Estimated)       */
    uint32_T    SwitchToClosedLoop;                     /**< Flag to indicate RampUp    */
    uint32_T    Valid;                      /**< Flag of Validation                     */
    uint32_T    CtrlAlg;                    /**< Control algorithm selector             */
    real32_T    Vdc;                        /**< DC Voltage                             */
} EmbedSimCtrlInput_T;

/**
 * \brief  Output Structure for Control Loop
 */
typedef struct
{
    real32_T    DutyU;                      /**< Phase U PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyV;                      /**< Phase V PWM duty cycle [0.0 .. 1.0] */
    real32_T    DutyW;                      /**< Phase W PWM duty cycle [0.0 .. 1.0] */
    uint32_T    SvmSector;                  /**< SVM Sector                           */
    real32_T    RotorSpeedEst;              /**< Rotor Velocity in RPM [RPM] */
    real32_T    RotorPositionEst;           /**< Rotor Position [RAD] */
    uint32_T    Valid;                      /**< Flag of Validation */
} EmbedSimCtrlOutput_T;

/**
 * \brief   Permanent Magnet Synchronous Motor (PMSM) parameters
 */
typedef struct
{
    real32_T    PolePairs;              /**< Number of pole pairs                         [-]      */
    real32_T    Rs;                     /**< Stator resistance                            [Ohm]    */
    real32_T    Ld;                     /**< Direct-axis inductance                       [H]      */
    real32_T    Lq;                     /**< Quadrature-axis inductance                   [H]      */
    real32_T    FluxPm;                 /**< Permanent magnet flux linkage                [Wb]     */
    real32_T    J;                      /**< Rotor inertia                                [kg·m²]  */
    real32_T    B;                      /**< Viscous damping coefficient                  [N·m·s]  */
    real32_T    RatedCurrent;           /**< Rated current                                [A]      */
    real32_T    RatedSpeed;             /**< Rated speed                                  [RPM]    */
    real32_T    MaxSpeed;               /**< Maximum allowable speed                      [RPM]    */
    real32_T    MaxCurrent;             /**< Maximum allowable current                    [A]      */
    real32_T    TorqueConstant;         /**< Torque constant (Kt)                         [N·m/A]  */
    real32_T    BackEmfConstant;        /**< Back EMF constant (Ke)                       [V·s]    */
    real32_T    Vdc;                    /**< DC bus voltage                               [V]      */
    real32_T    TorqueLoad;             /**< Torque Load                                  [N·m]    */
} EmbedSimMachineParam_T;

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
 * \brief   Estimate RPM from phase currents (back-EMF observer).
 */
extern void EmbedSim_EstimateRpm(EmbedSimMachineParam_T* MachineParamPtr, EmbedSimCtrlInput_T* InputPtr);

/**
 * \brief   Generate smooth reference trajectory with jerk limiting.
 */
extern void EmbedSim_CalculateRef(EmbedSimCtrlInput_T* InputPtr);

/**
 * \brief   Wrap angle to [0, 2π) range.
 */
extern void EmbedSim_WrapAngle(real32_T* anglePtr);

extern real32_T  EmbedSim_Clamp(real32_T value, real32_T min, real32_T max);

#endif /* EMBEDSIM_EMBED_SIM_CONTROL_H_ */
