/**
 * @file    svpwm.h
 * @brief   Space Vector PWM modulator — EmbedSim CodeGen block
 *
 * Converts three-phase voltages [v_a, v_b, v_c] to duty cycles [0..1].
 *
 *   duty_x = v_x / v_dc + 0.5f       (centred sinusoidal PWM)
 *   duty_x = clamp(duty_x, 0.0f, 1.0f)
 *
 * No overmodulation. Suitable for two-level IGBT/MOSFET inverter.
 *
 * MISRA C:2012 compliant.
 * Target : Infineon AURIX TC3xx  (TASKING vx compiler)
 * Safety : ASIL-D compatible (no dynamic allocation, no recursion)
 *
 * @author  EmbedSim Framework
 */

#ifndef SVPWM_H
#define SVPWM_H

#include "Sys_Types.h"   /* real32_T, uint8_T, etc. */

/* --------------------------------------------------------------------------
 * State structure
 * --------------------------------------------------------------------------
 * Stateless block — struct kept for uniform CodeGen interface.
 * All fields initialised to zero by BSS / explicit init.
 */
typedef struct
{
    real32_T duty_a;   /**< Output duty cycle phase A [0..1] */
    real32_T duty_b;   /**< Output duty cycle phase B [0..1] */
    real32_T duty_c;   /**< Output duty cycle phase C [0..1] */
    real32_T v_dc;     /**< DC-bus voltage used for normalisation [V] */
} SVPWM_Block_T;

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

/**
 * @brief  Initialise SVPWM block state.
 * @param  blk    Pointer to block state structure.
 * @param  v_dc   DC-bus voltage [V] (must be > 0).
 */
extern void SVPWM_Init(SVPWM_Block_T * const blk, real32_T v_dc);

/**
 * @brief  Compute duty cycles from phase voltages.
 *
 * @param  blk    Pointer to block state.
 * @param  v_a    Phase-A voltage w.r.t. virtual neutral [V].
 * @param  v_b    Phase-B voltage [V].
 * @param  v_c    Phase-C voltage [V].
 * @param  duty_a Output: duty cycle A [0..1].
 * @param  duty_b Output: duty cycle B [0..1].
 * @param  duty_c Output: duty cycle C [0..1].
 */
extern void SVPWM_Compute(
    SVPWM_Block_T * const blk,
    real32_T  v_a,
    real32_T  v_b,
    real32_T  v_c,
    real32_T * const duty_a,
    real32_T * const duty_b,
    real32_T * const duty_c
);

#endif /* SVPWM_H */
