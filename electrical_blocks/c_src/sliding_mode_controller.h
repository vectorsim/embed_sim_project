/*
 * sliding_mode_controller.h
 * =========================
 *
 * Sliding Mode Controller (SMC) for PMSM FOC inner loop.
 * Operates on the d-q rotating reference frame.
 *
 * THEORY
 * ------
 *   SMC drives a system state to a sliding surface s(e) = 0 and then
 *   constrains it to slide along that surface to the equilibrium.
 *
 *   Sliding surface (first-order):
 *       s(e)  = e  +  lambda * integral(e)          (PI-like surface)
 *       where e = x_ref - x_meas
 *
 *   Control law:
 *       u = u_eq  +  u_sw
 *       u_eq = feedforward (equivalent control)
 *       u_sw = -K_sw * sat(s / phi)                 (saturation — avoids chattering)
 *
 *   Saturation function (boundary layer, MISRA safe):
 *       sat(v) = v             if |v| <= 1
 *       sat(v) = sign(v)       if |v| >  1
 *
 * CHANNELS
 * --------
 *   Two independent SMC channels — one per d-q axis:
 *       Channel 0 : d-axis (flux control)
 *       Channel 1 : q-axis (torque control)
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

#ifndef SLIDING_MODE_CONTROLLER_H
#define SLIDING_MODE_CONTROLLER_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Sys_Types.h"    /**< real32_T, uint8_T etc. */

/******************************************************************************/
/*--------------------------------- Macros -----------------------------------*/
/******************************************************************************/

#define SMC_NUM_CHANNELS  (2U)    /**< d-axis + q-axis */

/******************************************************************************/
/*----------------------------- Data Structures ------------------------------*/
/******************************************************************************/

/**
 * @brief Per-channel SMC parameters
 */
typedef struct
{
    real32_T lambda;     /**< Sliding surface slope  (integral gain)    [rad/s] */
    real32_T K_sw;       /**< Switching gain                             [V/1]  */
    real32_T phi;        /**< Boundary layer thickness (chattering avoidance)   */
    real32_T out_min;    /**< Output saturation — lower limit            [V]    */
    real32_T out_max;    /**< Output saturation — upper limit            [V]    */
} SMC_Params_T;

/**
 * @brief Per-channel SMC run-time state
 */
typedef struct
{
    real32_T integral;   /**< Integral of error — used for sliding surface      */
    real32_T prev_error; /**< Previous error — reserved for future extensions   */
    real32_T surface;    /**< Last computed sliding surface value  s(e)         */
    real32_T output;     /**< Last computed control output                      */
} SMC_State_T;

/**
 * @brief Full SMC block (both d and q channels)
 */
typedef struct
{
    SMC_Params_T params[SMC_NUM_CHANNELS];  /**< [0]=d-axis  [1]=q-axis        */
    SMC_State_T  state[SMC_NUM_CHANNELS];   /**< Run-time state per channel     */
} SMC_Block_T;

/**
 * @brief SMC input/output bundle for one compute step
 */
typedef struct
{
    real32_T ref_d;      /**< d-axis reference (current set-point)  [A]        */
    real32_T ref_q;      /**< q-axis reference (current set-point)  [A]        */
    real32_T meas_d;     /**< d-axis measured current               [A]        */
    real32_T meas_q;     /**< q-axis measured current               [A]        */
} SMC_Input_T;

/**
 * @brief SMC output voltages
 */
typedef struct
{
    real32_T v_d;        /**< d-axis voltage command                [V]        */
    real32_T v_q;        /**< q-axis voltage command                [V]        */
} SMC_Output_T;

/******************************************************************************/
/*------------------------ Function Prototypes --------------------------------*/
/******************************************************************************/

/**
 * @brief Initialize SMC block with default PMSM parameters.
 *
 * Sets sensible defaults suitable for a small PMSM (R≈0.5 Ω, L≈5 mH).
 * Call before the first SMC_Compute().
 *
 * @param[out] pSMC   Pointer to SMC block
 */
extern void SMC_Init(SMC_Block_T* pSMC);

/**
 * @brief Configure SMC parameters for a specific channel.
 *
 * @param[out] pSMC      Pointer to SMC block
 * @param[in]  channel   0 = d-axis, 1 = q-axis
 * @param[in]  lambda    Sliding surface slope   (integral weight)
 * @param[in]  K_sw      Switching gain
 * @param[in]  phi       Boundary layer thickness (chattering avoidance)
 * @param[in]  out_min   Minimum output clamp    [V]
 * @param[in]  out_max   Maximum output clamp    [V]
 */
extern void SMC_SetParams(SMC_Block_T* pSMC,
                          uint8_T      channel,
                          real32_T     lambda,
                          real32_T     K_sw,
                          real32_T     phi,
                          real32_T     out_min,
                          real32_T     out_max);

/**
 * @brief Reset per-channel state (integrator, surface).
 *
 * Call on fault recovery or mode transitions.
 *
 * @param[out] pSMC      Pointer to SMC block
 * @param[in]  channel   0 = d-axis, 1 = q-axis  (255 = all channels)
 */
extern void SMC_ResetState(SMC_Block_T* pSMC, uint8_T channel);

/**
 * @brief Execute one SMC compute step (both d and q axes).
 *
 * @param[in,out] pSMC   Pointer to SMC block
 * @param[in]     pIn    Pointer to reference and measured currents
 * @param[in]     dt     Time step                                  [s]
 * @param[out]    pOut   Pointer to d-q voltage outputs
 */
extern void SMC_Compute(SMC_Block_T*       pSMC,
                        const SMC_Input_T* pIn,
                        real32_T           dt,
                        SMC_Output_T*      pOut);

#endif /* SLIDING_MODE_CONTROLLER_H */
