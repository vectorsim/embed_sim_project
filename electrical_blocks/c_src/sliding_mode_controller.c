/*
 * sliding_mode_controller.c
 * =========================
 *
 * Sliding Mode Controller (SMC) for PMSM FOC — d-q axis inner loop.
 *
 * ALGORITHM SUMMARY
 * -----------------
 *   For each axis (d and q) independently:
 *
 *   1.  Error:
 *           e(k)  = ref(k) - meas(k)
 *
 *   2.  Sliding surface (PI-type, trapezoidal integration):
 *           s(k)  = e(k)  +  lambda * integral_e(k)
 *           integral_e(k) = integral_e(k-1)  +  e(k) * dt
 *
 *   3.  Saturation function (boundary layer — avoids chattering):
 *           sat(v) = v              |v| <= 1   (inside boundary layer)
 *           sat(v) = +1.0f          v  >  1
 *           sat(v) = -1.0f          v  < -1
 *
 *   4.  Control output:
 *           u(k)  = +K_sw * sat(s(k) / phi)
 *
 *   5.  Anti-windup / output clamp:
 *           u(k)  = clamp(u(k), out_min, out_max)
 *           If output is clamped, the integral is NOT updated (back-calculation).
 *
 * EMBEDDED NOTES
 * --------------
 *   - No heap, no dynamic allocation.
 *   - All arithmetic in float (real32_T) — native to TriCore FPU.
 *   - No global variables — caller owns SMC_Block_T.
 *   - Pure and re-entrant per channel.
 *
 * @author EmbedSim Framework
 * @version 1.0.0
 * @date 2025
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "sliding_mode_controller.h"
#include <math.h>     /* fabsf */

/******************************************************************************/
/*-----------------------------Static Helpers---------------------------------*/
/******************************************************************************/

/**
 * @brief Clamp a float value to [lo, hi].
 */
static inline real32_T smc_clamp(real32_T val, real32_T lo, real32_T hi)
{
    if (val > hi) return hi;
    if (val < lo) return lo;
    return val;
}

/**
 * @brief Saturation function  sat(v) ∈ [-1, +1].
 *
 * Inside the boundary layer (|v| <= 1) the control law is linear,
 * which eliminates chattering.  Outside it switches at unit gain.
 */
static inline real32_T smc_sat(real32_T v)
{
    if (v >  1.0f) return  1.0f;
    if (v < -1.0f) return -1.0f;
    return v;
}

/**
 * @brief Compute one SMC channel output.
 *
 * @param[in,out] pP     Params for this channel
 * @param[in,out] pS     State for this channel
 * @param[in]     ref    Reference value  [A]
 * @param[in]     meas   Measured value   [A]
 * @param[in]     dt     Time step        [s]
 * @return               Voltage command  [V]
 */
static real32_T smc_channel(const SMC_Params_T* pP,
                             SMC_State_T*        pS,
                             real32_T            ref,
                             real32_T            meas,
                             real32_T            dt)
{
    real32_T error;
    real32_T surface;
    real32_T raw_out;
    real32_T clamped_out;

    /* ── 1. Error ─────────────────────────────────────────────────────────── */
    error = ref - meas;

    /* ── 2. Sliding surface ────────────────────────────────────────────────── */
    pS->integral += error * dt;
    surface       = error + pP->lambda * pS->integral;
    pS->surface   = surface;

    /* ── 3+4. Control law ──────────────────────────────────────────────────── */
    raw_out = pP->K_sw * smc_sat(surface / pP->phi);

    /* ── 5. Output clamp + anti-windup ─────────────────────────────────────── */
    clamped_out = smc_clamp(raw_out, pP->out_min, pP->out_max);

    /* Back-calculation: if output saturated, un-wind the integral */
    if (clamped_out != raw_out)
    {
        pS->integral -= error * dt;   /* undo this step's integration */
    }

    pS->prev_error = error;
    pS->output     = clamped_out;

    return clamped_out;
}

/******************************************************************************/
/*-------------------------Function Implementations---------------------------*/
/******************************************************************************/

void SMC_Init(SMC_Block_T* pSMC)
{
    uint8_T ch;

    /* Default parameters — tuned for a small PMSM (R≈0.5Ω, L≈5mH) */
    for (ch = 0U; ch < SMC_NUM_CHANNELS; ch++)
    {
        /* d-axis (ch=0) and q-axis (ch=1) share the same initial defaults */
        pSMC->params[ch].lambda   = 500.0f;    /* surface slope            */
        pSMC->params[ch].K_sw     = 24.0f;     /* switching gain  [V]      */
        pSMC->params[ch].phi      = 5.0f;      /* boundary layer  [A]      */
        pSMC->params[ch].out_min  = -24.0f;    /* ±24 V DC bus             */
        pSMC->params[ch].out_max  =  24.0f;

        /* Zero state */
        pSMC->state[ch].integral   = 0.0f;
        pSMC->state[ch].prev_error = 0.0f;
        pSMC->state[ch].surface    = 0.0f;
        pSMC->state[ch].output     = 0.0f;
    }
}

void SMC_SetParams(SMC_Block_T* pSMC,
                   uint8_T      channel,
                   real32_T     lambda,
                   real32_T     K_sw,
                   real32_T     phi,
                   real32_T     out_min,
                   real32_T     out_max)
{
    if (channel >= SMC_NUM_CHANNELS) return;

    pSMC->params[channel].lambda  = lambda;
    pSMC->params[channel].K_sw    = K_sw;
    pSMC->params[channel].phi     = (phi > 0.0f) ? phi : 1e-6f;  /* guard /0 */
    pSMC->params[channel].out_min = out_min;
    pSMC->params[channel].out_max = out_max;
}

void SMC_ResetState(SMC_Block_T* pSMC, uint8_T channel)
{
    uint8_T ch;

    if (channel == 255U)
    {
        /* Reset all channels */
        for (ch = 0U; ch < SMC_NUM_CHANNELS; ch++)
        {
            pSMC->state[ch].integral   = 0.0f;
            pSMC->state[ch].prev_error = 0.0f;
            pSMC->state[ch].surface    = 0.0f;
            pSMC->state[ch].output     = 0.0f;
        }
    }
    else if (channel < SMC_NUM_CHANNELS)
    {
        pSMC->state[channel].integral   = 0.0f;
        pSMC->state[channel].prev_error = 0.0f;
        pSMC->state[channel].surface    = 0.0f;
        pSMC->state[channel].output     = 0.0f;
    }
}

void SMC_Compute(SMC_Block_T*       pSMC,
                 const SMC_Input_T* pIn,
                 real32_T           dt,
                 SMC_Output_T*      pOut)
{
    /* d-axis channel (0) */
    pOut->v_d = smc_channel(
        &pSMC->params[0],
        &pSMC->state[0],
        pIn->ref_d,
        pIn->meas_d,
        dt
    );

    /* q-axis channel (1) */
    pOut->v_q = smc_channel(
        &pSMC->params[1],
        &pSMC->state[1],
        pIn->ref_q,
        pIn->meas_q,
        dt
    );
}
