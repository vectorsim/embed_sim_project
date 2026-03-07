/*
 * speed_pi_controller.h
 * =====================
 *
 * Proportional-Integral speed controller for PMSM FOC outer loop.
 *
 * POSITION IN FOC CHAIN
 * ---------------------
 *   omega_ref ──► [SpeedPI] ──► iq_ref ──► [SMC inner loop]
 *   omega_meas ──►              id_ref = 0 (MTPA)
 *
 * ALGORITHM
 * ---------
 *   e(k)      = omega_ref(k) − omega_meas(k)
 *   integ(k)  = clamp(integ(k-1) + e(k)·dt,  −i_max/Ki,  +i_max/Ki)
 *   iq_ref(k) = clamp(Kp·e(k) + Ki·integ(k), −i_max,     +i_max)
 *   id_ref    = 0  (surface-mounted PMSM / MTPA at id=0)
 *
 * Anti-windup: integrator clamped to prevent iq_ref from saturating.
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

#ifndef SPEED_PI_CONTROLLER_H
#define SPEED_PI_CONTROLLER_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Sys_Types.h"    /**< real32_T, uint8_T */

/******************************************************************************/
/*----------------------------- Data Structures ------------------------------*/
/******************************************************************************/

/**
 * @brief Speed PI controller parameters
 */
typedef struct
{
    real32_T Kp;       /**< Proportional gain                      [A·s/rad] */
    real32_T Ki;       /**< Integral gain                          [A/rad]   */
    real32_T i_max;    /**< Output current clamp                   [A]       */
} SpeedPI_Params_T;

/**
 * @brief Speed PI controller run-time state
 */
typedef struct
{
    real32_T integrator;   /**< Error integral accumulator                   */
    real32_T prev_error;   /**< Previous error (reserved for D term)         */
} SpeedPI_State_T;

/**
 * @brief Full Speed PI block
 */
typedef struct
{
    SpeedPI_Params_T params;
    SpeedPI_State_T  state;
} SpeedPI_Block_T;

/**
 * @brief Speed PI input bundle
 */
typedef struct
{
    real32_T omega_ref;    /**< Speed reference  [rad/s] */
    real32_T omega_meas;   /**< Measured speed   [rad/s] */
} SpeedPI_Input_T;

/**
 * @brief Speed PI output bundle
 */
typedef struct
{
    real32_T id_ref;   /**< d-axis current reference — always 0  [A] */
    real32_T iq_ref;   /**< q-axis current reference (torque)    [A] */
} SpeedPI_Output_T;

/******************************************************************************/
/*------------------------ Function Prototypes --------------------------------*/
/******************************************************************************/

/**
 * @brief Initialize Speed PI block with default parameters.
 * @param[out] pPI  Pointer to SpeedPI block
 */
extern void SpeedPI_Init(SpeedPI_Block_T* pPI);

/**
 * @brief Set Speed PI parameters.
 *
 * @param[out] pPI    Pointer to SpeedPI block
 * @param[in]  Kp     Proportional gain
 * @param[in]  Ki     Integral gain
 * @param[in]  i_max  Output clamp  [A]
 */
extern void SpeedPI_SetParams(SpeedPI_Block_T* pPI,
                               real32_T         Kp,
                               real32_T         Ki,
                               real32_T         i_max);

/**
 * @brief Reset integrator state.
 * @param[out] pPI  Pointer to SpeedPI block
 */
extern void SpeedPI_ResetState(SpeedPI_Block_T* pPI);

/**
 * @brief Execute one Speed PI compute step.
 *
 * @param[in,out] pPI   Pointer to SpeedPI block
 * @param[in]     pIn   Speed reference and measured speed
 * @param[in]     dt    Time step  [s]
 * @param[out]    pOut  id_ref (=0) and iq_ref
 */
extern void SpeedPI_Compute(SpeedPI_Block_T*       pPI,
                             const SpeedPI_Input_T* pIn,
                             real32_T               dt,
                             SpeedPI_Output_T*      pOut);

#endif /* SPEED_PI_CONTROLLER_H */
