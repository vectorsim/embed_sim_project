/*
 * pi_buck_controller.c
 * =====================
 *
 * Proportional-Integral voltage controller for Buck Converter.
 *
 * ALGORITHM
 * ---------
 *   e(k)      = V_ref(k) − V_meas(k)
 *   integ(k)  = clamp(integ(k-1) + e(k)·dt,  −duty_max/Ki,  +duty_max/Ki)
 *   duty(k)   = clamp(Kp·e(k) + Ki·integ(k), duty_min,      duty_max)
 *
 * Anti-windup: integrator clamped so the duty cycle cannot wind beyond limits.
 *
 * @author EmbedSim Framework
 * @version 1.0.0
 * @date 2025
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "pi_buck_controller.h"

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

void PI_Buck_Init(PI_Buck_Block_T* pPI)
{
    /* Default tuning — suitable for a typical buck converter
     * (L=100µH, C=100µF, Vin=24V, Vout=12V, fsw=100kHz) */
    pPI->params.Kp        = 0.1f;     /* 0.1 duty cycle per volt of error */
    pPI->params.Ki        = 5.0f;     /* 5.0 duty cycle per volt-second */
    pPI->params.duty_max  = 0.95f;    /* 95% max duty cycle */
    pPI->params.duty_min  = 0.05f;    /* 5% min duty cycle */
    pPI->params.Ts        = 0.0001f;  /* 10 kHz control loop (100µs) */

    pPI->state.integrator  = 0.0f;
    pPI->state.prev_error  = 0.0f;
    pPI->state.last_output = 0.0f;
}

void PI_Buck_SetParams(PI_Buck_Block_T* pPI,
                       real32_T         Kp,
                       real32_T         Ki,
                       real32_T         duty_max,
                       real32_T         duty_min,
                       real32_T         Ts)
{
    pPI->params.Kp        = Kp;
    pPI->params.Ki        = (Ki > 0.0f) ? Ki : 1e-6f;  /* guard div/0 */
    pPI->params.duty_max  = pi_clamp(duty_max, 0.0f, 1.0f);
    pPI->params.duty_min  = pi_clamp(duty_min, 0.0f, pPI->params.duty_max);
    pPI->params.Ts        = (Ts > 0.0f) ? Ts : 1e-6f;
}

void PI_Buck_ResetState(PI_Buck_Block_T* pPI)
{
    pPI->state.integrator  = 0.0f;
    pPI->state.prev_error  = 0.0f;
    pPI->state.last_output = 0.0f;
}

void PI_Buck_Compute(PI_Buck_Block_T*       pPI,
                     const PI_Buck_Input_T* pIn,
                     real32_T               dt,
                     PI_Buck_Output_T*      pOut)
{
    real32_T error;
    real32_T integ_limit;
    real32_T raw_duty;
    real32_T sample_time;

    /* Use provided dt if > 0, otherwise use configured Ts */
    sample_time = (dt > 0.0f) ? dt : pPI->params.Ts;

    /* ── 1. Error ──────────────────────────────────────────────────────────── */
    error = pIn->V_ref - pIn->V_meas;

    /* ── 2. Integrator with anti-windup clamp ─────────────────────────────── */
    /* Limit integrator so that Ki * integ cannot exceed duty_max */
    integ_limit = pPI->params.duty_max / pPI->params.Ki;
    pPI->state.integrator = pi_clamp(
        pPI->state.integrator + error * sample_time,
        -integ_limit,
         integ_limit
    );

    /* ── 3. PI output ─────────────────────────────────────────────────────── */
    raw_duty = pPI->params.Kp * error
             + pPI->params.Ki * pPI->state.integrator;

    pOut->duty = pi_clamp(raw_duty,
                          pPI->params.duty_min,
                          pPI->params.duty_max);

    pPI->state.last_output = pOut->duty;
    pPI->state.prev_error = error;
}