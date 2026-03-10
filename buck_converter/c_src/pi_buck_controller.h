/*
 * pi_buck_controller.h
 * =====================
 *
 * Proportional-Integral voltage controller for Buck Converter.
 *
 * POSITION IN CONTROL CHAIN
 * -------------------------
 *   V_ref ──► [PI Buck] ──► duty ──► [BuckConverter Plant]
 *   V_meas ──►
 *
 * ALGORITHM
 * ---------
 *   e(k)      = V_ref(k) − V_meas(k)
 *   integ(k)  = clamp(integ(k-1) + e(k)·dt,  −duty_max/Ki,  +duty_max/Ki)
 *   duty(k)   = clamp(Kp·e(k) + Ki·integ(k), 0.0,          duty_max)
 *
 * Anti-windup: integrator clamped to prevent duty cycle saturation.
 *
 * TARGET
 * ------
 *   Primary  : Infineon Aurix TriCore  (TASKING ctc)
 *   Secondary: ARM Cortex-M4           (GCC / LLVM)
 *   Simulation: Windows / Linux        (via Cython wrapper)
 *
 * @author EmbedSim Framework
 * @version 1.0.0
 * @date 2025
 */

#ifndef PI_BUCK_CONTROLLER_H
#define PI_BUCK_CONTROLLER_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Sys_Types.h"    /**< real32_T, uint8_T */

/******************************************************************************/
/*----------------------------- Data Structures ------------------------------*/
/******************************************************************************/

/**
 * @brief PI Buck controller parameters
 */
typedef struct
{
    real32_T Kp;           /**< Proportional gain                      [1/V]    */
    real32_T Ki;           /**< Integral gain                          [1/(V·s)] */
    real32_T duty_max;     /**< Maximum duty cycle (typically 0.95)    [0-1]    */
    real32_T duty_min;     /**< Minimum duty cycle (typically 0.05)    [0-1]    */
    real32_T Ts;           /**< Sample time                            [s]      */
} PI_Buck_Params_T;

/**
 * @brief PI Buck controller run-time state
 */
typedef struct
{
    real32_T integrator;   /**< Error integral accumulator                       */
    real32_T prev_error;   /**< Previous error (for debugging)                   */
    real32_T last_output;  /**< Last duty cycle output                           */
} PI_Buck_State_T;

/**
 * @brief Full PI Buck block
 */
typedef struct
{
    PI_Buck_Params_T params;
    PI_Buck_State_T  state;
} PI_Buck_Block_T;

/**
 * @brief PI Buck input bundle
 */
typedef struct
{
    real32_T V_ref;        /**< Voltage reference  [V] */
    real32_T V_meas;       /**< Measured voltage   [V] */
} PI_Buck_Input_T;

/**
 * @brief PI Buck output bundle
 */
typedef struct
{
    real32_T duty;         /**< PWM duty cycle [0-1] */
} PI_Buck_Output_T;

/******************************************************************************/
/*------------------------ Function Prototypes --------------------------------*/
/******************************************************************************/

/**
 * @brief Initialize PI Buck block with default parameters.
 * @param[out] pPI  Pointer to PI Buck block
 */
extern void PI_Buck_Init(PI_Buck_Block_T* pPI);

/**
 * @brief Set PI Buck parameters.
 *
 * @param[out] pPI       Pointer to PI Buck block
 * @param[in]  Kp        Proportional gain
 * @param[in]  Ki        Integral gain
 * @param[in]  duty_max  Maximum duty cycle [0-1]
 * @param[in]  duty_min  Minimum duty cycle [0-1]
 * @param[in]  Ts        Sample time [s]
 */
extern void PI_Buck_SetParams(PI_Buck_Block_T* pPI,
                              real32_T         Kp,
                              real32_T         Ki,
                              real32_T         duty_max,
                              real32_T         duty_min,
                              real32_T         Ts);

/**
 * @brief Reset integrator state.
 * @param[out] pPI  Pointer to PI Buck block
 */
extern void PI_Buck_ResetState(PI_Buck_Block_T* pPI);

/**
 * @brief Execute one PI Buck compute step.
 *
 * @param[in,out] pPI   Pointer to PI Buck block
 * @param[in]     pIn   Voltage reference and measured voltage
 * @param[in]     dt    Time step (optional, overrides Ts if > 0) [s]
 * @param[out]    pOut  PWM duty cycle
 */
extern void PI_Buck_Compute(PI_Buck_Block_T*       pPI,
                            const PI_Buck_Input_T* pIn,
                            real32_T               dt,
                            PI_Buck_Output_T*      pOut);

#endif /* PI_BUCK_CONTROLLER_H */