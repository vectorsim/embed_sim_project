/**
 * @file    svpwm.c
 * @brief   Space Vector PWM modulator — EmbedSim CodeGen block
 *
 * MISRA C:2012 compliant.
 * Target : Infineon AURIX TC3xx  (TASKING vx compiler)
 * Safety : ASIL-D
 *
 * @author  EmbedSim Framework
 */

/* TASKING compiler: place in cached data flash if needed */
/* #pragma section ".rodata" */

#include "svpwm.h"

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

/** Clamp x to [lo, hi]. */
static real32_T svpwm_clamp(real32_T x, real32_T lo, real32_T hi)
{
    real32_T result = x;
    if (result < lo) { result = lo; }
    if (result > hi) { result = hi; }
    return result;
}

/* --------------------------------------------------------------------------
 * SVPWM_Init
 * -------------------------------------------------------------------------- */
void SVPWM_Init(SVPWM_Block_T * const blk, real32_T v_dc)
{
    blk->duty_a = 0.5f;
    blk->duty_b = 0.5f;
    blk->duty_c = 0.5f;
    blk->v_dc   = v_dc;
}

/* --------------------------------------------------------------------------
 * SVPWM_Compute
 * --------------------------------------------------------------------------
 *
 * Algorithm (centred sinusoidal PWM):
 *
 *   duty_x = v_x / v_dc + 0.5      (maps ±v_dc/2 → [0, 1])
 *   duty_x = clamp(duty_x, 0, 1)
 *
 * This is equivalent to the natural-sampled sinusoidal PWM technique and
 * produces the same fundamental as space-vector modulation up to the
 * linear modulation limit (v_peak ≤ v_dc / 2).
 *
 * For third-harmonic injection (extends linear range to v_dc/√3) the
 * caller should pre-add the zero-sequence term before calling this function.
 */
void SVPWM_Compute(
    SVPWM_Block_T * const blk,
    real32_T  v_a,
    real32_T  v_b,
    real32_T  v_c,
    real32_T * const duty_a,
    real32_T * const duty_b,
    real32_T * const duty_c
)
{
    real32_T inv_vdc;
    real32_T da;
    real32_T db;
    real32_T dc;

    /* Guard against division by zero */
    if (blk->v_dc > 0.0f)
    {
        inv_vdc = 1.0f / blk->v_dc;
    }
    else
    {
        inv_vdc = 0.0f;
    }

    da = svpwm_clamp((v_a * inv_vdc) + 0.5f, 0.0f, 1.0f);
    db = svpwm_clamp((v_b * inv_vdc) + 0.5f, 0.0f, 1.0f);
    dc = svpwm_clamp((v_c * inv_vdc) + 0.5f, 0.0f, 1.0f);

    blk->duty_a = da;
    blk->duty_b = db;
    blk->duty_c = dc;

    *duty_a = da;
    *duty_b = db;
    *duty_c = dc;
}
