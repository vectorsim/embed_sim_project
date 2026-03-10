/**
 * @file    smo.c
 * @brief   Sliding Mode Observer (SMO) for PMSM sensorless FOC
 *
 * MISRA C:2012 compliant.
 * Target : Infineon AURIX TC3xx  (TASKING vx compiler)
 * Safety : ASIL-D
 *
 * @author  EmbedSim Framework
 */

#include "smo.h"
#include <math.h>   /* atan2f, cosf, sinf — from TASKING runtime or libm */

/* --------------------------------------------------------------------------
 * Constants
 * -------------------------------------------------------------------------- */
#define SMO_PI          (3.14159265358979323846f)
#define SMO_TWO_PI      (6.28318530717958647692f)
#define SMO_INV_TWO_PI  (0.15915494309189534561f)  /* 1 / (2π) */

/* --------------------------------------------------------------------------
 * Internal helpers
 * -------------------------------------------------------------------------- */

/**
 * Saturation function with boundary layer — replaces hard sign() to reduce
 * chattering on the current observer injection term.
 *
 *   sat(x, φ) = x/φ   if |x| < φ
 *             = sign(x)  otherwise
 */
static real32_T smo_sat(real32_T x, real32_T phi)
{
    real32_T result;
    if (phi <= 0.0f)
    {
        /* Degenerate case: return sign(x) */
        result = (x >= 0.0f) ? 1.0f : -1.0f;
    }
    else if (x > phi)
    {
        result = 1.0f;
    }
    else if (x < -phi)
    {
        result = -1.0f;
    }
    else
    {
        result = x / phi;
    }
    return result;
}

/**
 * Unwrap angle difference to (-π, π].
 * Prevents speed spikes at the θ = ±π discontinuity.
 */
static real32_T smo_unwrap_delta(real32_T delta)
{
    real32_T d = delta;
    /* Bring d into (-2π, 2π) first */
    while (d >  SMO_PI) { d -= SMO_TWO_PI; }
    while (d < -SMO_PI) { d += SMO_TWO_PI; }
    return d;
}

/* --------------------------------------------------------------------------
 * SMO_Init
 * -------------------------------------------------------------------------- */
void SMO_Init(SMO_Block_T * const blk, const SMO_Params_T * const prm)
{
    /* Copy parameters */
    blk->prm = *prm;

    /* Clear observer states */
    blk->i_alpha_hat  = 0.0f;
    blk->i_beta_hat   = 0.0f;
    blk->e_alpha_hat  = 0.0f;
    blk->e_beta_hat   = 0.0f;
    blk->theta_e_hat  = 0.0f;
    blk->theta_e_prev = 0.0f;
    blk->omega_e_filt = 0.0f;

    /* Clear outputs */
    blk->y[0] = 0.0f;
    blk->y[1] = 0.0f;
    blk->y[2] = 0.0f;
    blk->y[3] = 0.0f;
}

/* --------------------------------------------------------------------------
 * SMO_Compute
 * -------------------------------------------------------------------------- */
void SMO_Compute(
    SMO_Block_T * const blk,
    real32_T dt,
    real32_T i_alpha,
    real32_T i_beta,
    real32_T v_alpha,
    real32_T v_beta,
    real32_T * const y
)
{
    real32_T i_alpha_hat_new;
    real32_T i_beta_hat_new;
    real32_T err_alpha;
    real32_T err_beta;
    real32_T z_alpha;
    real32_T z_beta;
    real32_T e_alpha_new;
    real32_T e_beta_new;
    real32_T theta_new;
    real32_T delta_theta;
    real32_T omega_e_raw;
    real32_T omega_e_new;
    real32_T omega_m;
    real32_T cos_th;
    real32_T sin_th;
    real32_T i_d_hat;
    real32_T i_q_hat;
    real32_T inv_L;

    /* Guard against L = 0 */
    if (blk->prm.L > 0.0f)
    {
        inv_L = 1.0f / blk->prm.L;
    }
    else
    {
        inv_L = 0.0f;
    }

    /* ------------------------------------------------------------------
     * Step 1: Current estimation errors
     * ------------------------------------------------------------------ */
    err_alpha = i_alpha - blk->i_alpha_hat;
    err_beta  = i_beta  - blk->i_beta_hat;

    /* ------------------------------------------------------------------
     * Step 2: Sliding injection z = K_smo * sat(err / phi)
     * ------------------------------------------------------------------ */
    z_alpha = blk->prm.K_smo * smo_sat(err_alpha, blk->prm.phi);
    z_beta  = blk->prm.K_smo * smo_sat(err_beta,  blk->prm.phi);

    /* ------------------------------------------------------------------
     * Step 3: Euler integration of current observer
     *   dî/dt = (v - R·î + z) / L
     * ------------------------------------------------------------------ */
    i_alpha_hat_new = blk->i_alpha_hat
        + dt * inv_L * (v_alpha - blk->prm.R * blk->i_alpha_hat + z_alpha);
    i_beta_hat_new  = blk->i_beta_hat
        + dt * inv_L * (v_beta  - blk->prm.R * blk->i_beta_hat  + z_beta);

    blk->i_alpha_hat = i_alpha_hat_new;
    blk->i_beta_hat  = i_beta_hat_new;

    /* ------------------------------------------------------------------
     * Step 4: Back-EMF LPF
     *   ê += wc_emf * (z - ê) * dt
     * ------------------------------------------------------------------ */
    e_alpha_new = blk->e_alpha_hat
        + blk->prm.wc_emf * (z_alpha - blk->e_alpha_hat) * dt;
    e_beta_new  = blk->e_beta_hat
        + blk->prm.wc_emf * (z_beta  - blk->e_beta_hat)  * dt;

    blk->e_alpha_hat = e_alpha_new;
    blk->e_beta_hat  = e_beta_new;

    /* ------------------------------------------------------------------
     * Step 5: Electrical angle from back-EMF
     *   θ̂_e = atan2(-ê_α, ê_β)
     *
     * This follows from the IPMSM back-EMF model:
     *   e_α = -ω_e · λ_pm · sin(θ_e)   →  -e_α ∝ sin(θ_e)
     *   e_β =  ω_e · λ_pm · cos(θ_e)   →   e_β ∝ cos(θ_e)
     * ------------------------------------------------------------------ */
    theta_new = atan2f(-e_alpha_new, e_beta_new);
    blk->theta_e_hat = theta_new;

    /* ------------------------------------------------------------------
     * Step 6: Electrical speed — differentiate θ̂_e + LPF
     * ------------------------------------------------------------------ */
    delta_theta  = smo_unwrap_delta(theta_new - blk->theta_e_prev);
    blk->theta_e_prev = theta_new;

    if (dt > 0.0f)
    {
        omega_e_raw = delta_theta / dt;
    }
    else
    {
        omega_e_raw = blk->omega_e_filt;
    }

    omega_e_new = blk->omega_e_filt
        + blk->prm.wc_spd * (omega_e_raw - blk->omega_e_filt) * dt;
    blk->omega_e_filt = omega_e_new;

    /* Mechanical speed: ω_m = ω_e / p */
    if (blk->prm.p > 0.0f)
    {
        omega_m = omega_e_new / blk->prm.p;
    }
    else
    {
        omega_m = 0.0f;
    }

    /* ------------------------------------------------------------------
     * Step 7: dq current estimate via Park transform
     *   î_d =  î_α · cos(θ̂_e) + î_β · sin(θ̂_e)
     *   î_q = -î_α · sin(θ̂_e) + î_β · cos(θ̂_e)
     * ------------------------------------------------------------------ */
    cos_th  = cosf(theta_new);
    sin_th  = sinf(theta_new);
    i_d_hat =  i_alpha_hat_new * cos_th + i_beta_hat_new * sin_th;
    i_q_hat = -i_alpha_hat_new * sin_th + i_beta_hat_new * cos_th;

    /* ------------------------------------------------------------------
     * Step 8: Pack outputs
     *   y[0] = θ̂_e   y[1] = ω̂_m   y[2] = î_d   y[3] = î_q
     * ------------------------------------------------------------------ */
    blk->y[0] = theta_new;
    blk->y[1] = omega_m;
    blk->y[2] = i_d_hat;
    blk->y[3] = i_q_hat;

    y[0] = blk->y[0];
    y[1] = blk->y[1];
    y[2] = blk->y[2];
    y[3] = blk->y[3];
}
