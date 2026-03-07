/*
 * speed_pi_controller.c
 * =====================
 *
 * Proportional-Integral speed controller for PMSM FOC outer loop.
 *
 * ALGORITHM
 * ---------
 *   e(k)      = omega_ref(k) − omega_meas(k)
 *   integ(k)  = clamp(integ(k-1) + e(k)·dt,  −i_max/Ki,  +i_max/Ki)
 *   iq_ref(k) = clamp(Kp·e(k) + Ki·integ(k), −i_max,     +i_max)
 *   id_ref    = 0
 *
 * Anti-windup: integrator is clamped so the output cannot wind beyond i_max.
 *
 * @author EmbedSim Framework
 * @version 1.0.0
 * @date 2025
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "speed_pi_controller.h"

/******************************************************************************/
/*-----------------------------Static Helpers---------------------------------*/
/******************************************************************************/

static inline real32_T pi_clamp(real32_T val, real32_T lo, real32_T hi)
{
    if (val > hi) return hi;
    if (val < lo) return lo;
    return val;
}

/******************************************************************************/
/*-------------------------Function Implementations---------------------------*/
/******************************************************************************/

void SpeedPI_Init(SpeedPI_Block_T* pPI)
{
    /* Default tuning — suitable for a small PMSM (J≈2e-3, p=2) */
    pPI->params.Kp    = 0.5f;
    pPI->params.Ki    = 5.0f;
    pPI->params.i_max = 10.0f;

    pPI->state.integrator  = 0.0f;
    pPI->state.prev_error  = 0.0f;
}

void SpeedPI_SetParams(SpeedPI_Block_T* pPI,
                        real32_T         Kp,
                        real32_T         Ki,
                        real32_T         i_max)
{
    pPI->params.Kp    = Kp;
    pPI->params.Ki    = (Ki > 0.0f) ? Ki : 1e-6f;  /* guard div/0 */
    pPI->params.i_max = i_max;
}

void SpeedPI_ResetState(SpeedPI_Block_T* pPI)
{
    pPI->state.integrator = 0.0f;
    pPI->state.prev_error = 0.0f;
}

void SpeedPI_Compute(SpeedPI_Block_T*       pPI,
                      const SpeedPI_Input_T* pIn,
                      real32_T               dt,
                      SpeedPI_Output_T*      pOut)
{
    real32_T error;
    real32_T integ_limit;
    real32_T raw_iq;

    /* ── 1. Error ──────────────────────────────────────────────────────────── */
    error = pIn->omega_ref - pIn->omega_meas;

    /* ── 2. Integrator with anti-windup clamp ─────────────────────────────── */
    integ_limit = pPI->params.i_max / pPI->params.Ki;
    pPI->state.integrator = pi_clamp(
        pPI->state.integrator + error * dt,
        -integ_limit,
         integ_limit
    );

    /* ── 3. PI output ─────────────────────────────────────────────────────── */
    raw_iq = pPI->params.Kp * error
           + pPI->params.Ki * pPI->state.integrator;

    pOut->id_ref = 0.0f;
    pOut->iq_ref = pi_clamp(raw_iq, -pPI->params.i_max, pPI->params.i_max);

    pPI->state.prev_error = error;
}
