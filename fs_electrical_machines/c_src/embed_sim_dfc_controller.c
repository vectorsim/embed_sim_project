/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     Differential Flatness FOC Controller
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#include "embed_sim_dfc_controller.h"
#include <math.h>
#include <string.h>

#define DFC_ZERO_F   ((MatrixFloat)0.0f)
#define DFC_ONE_F    ((MatrixFloat)1.0f)
#define DFC_TWO_PI_F ((MatrixFloat)6.28318530717959f)
#define DFC_PI_F     ((MatrixFloat)3.14159265358979f)


/*********************************************************************************************************************/
/*--------------------------------------------Private Function Prototypes--------------------------------------------*/
/*********************************************************************************************************************/

static MatrixFloat DFC_Clamp(MatrixFloat value, MatrixFloat limit);
static MatrixFloat DFC_FusionAlpha(MatrixFloat omega_abs);
static MatrixFloat DFC_SMOSwitch(MatrixFloat error);

static void DFC_SpeedFusion_Update(
    DFC_SpeedFusion_T * const fusion,
    MatrixFloat               theta_m,
    MatrixFloat               omega_smo_e,
    MatrixFloat               dt,
    MatrixFloat             * const theta_e,
    MatrixFloat             * const omega_e,
    MatrixFloat             * const omega_meas_mech);

static void DFC_SMO_Step(
    DFC_SMO_T     * const smo,
    MatrixFloat           v_alpha,
    MatrixFloat           v_beta,
    MatrixFloat           i_alpha,
    MatrixFloat           i_beta,
    MatrixFloat           dt,
    uint32_T              warmup_cnt,
    MatrixFloat         * const omega_e_smo);

static void DFC_VoltageLaw(
    MatrixFloat                 iq_ref,
    MatrixFloat                 diq_dt,
    MatrixFloat                 id_meas,
    MatrixFloat                 iq_meas,
    MatrixFloat                 omega_e,
    MatrixFloat               * const vd,
    MatrixFloat               * const vq);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

static MatrixFloat DFC_Clamp(const MatrixFloat value, const MatrixFloat limit)
{
    MatrixFloat result = value;

    if (result > limit)
    {
        result = limit;
    }
    else if (result < -limit)
    {
        result = -limit;
    }
    else
    {
        /* Within range - no action required */
    }

    return result;
}


static MatrixFloat DFC_FusionAlpha(const MatrixFloat omega_abs)
{
    MatrixFloat result;

    if (omega_abs <= DFC_FUSION_OMEGA_LO)
    {
        result = DFC_ZERO_F;
    }
    else if (omega_abs >= DFC_FUSION_OMEGA_HI)
    {
        result = DFC_ONE_F;
    }
    else
    {
        /* Linear interpolation between thresholds */
        result = (omega_abs - DFC_FUSION_OMEGA_LO) / (DFC_FUSION_OMEGA_HI - DFC_FUSION_OMEGA_LO);
    }

    return result;
}


static MatrixFloat DFC_SMOSwitch(const MatrixFloat error)
{
    MatrixFloat result;
    const MatrixFloat width = (MatrixFloat)0.01f;
    const MatrixFloat arg = error / width;

    if (arg > (MatrixFloat)5.0f)
    {
        result = DFC_ONE_F;
    }
    else if (arg < (MatrixFloat)-5.0f)
    {
        result = -DFC_ONE_F;
    }
    else
    {
        /* Linear region for smooth switching */
        result = arg * (MatrixFloat)0.2f;
    }

    return result;
}


static void DFC_SpeedFusion_Update(
    DFC_SpeedFusion_T * const fusion,
    const MatrixFloat         theta_m,
    const MatrixFloat         omega_smo_e,
    const MatrixFloat         dt,
    MatrixFloat             * const theta_e,
    MatrixFloat             * const omega_e,
    MatrixFloat             * const omega_meas_mech)
{
    MatrixFloat delta;
    MatrixFloat omega_raw;
    MatrixFloat alpha;
    MatrixFloat iir_coeff;
    MatrixFloat omega_enc_e;

    if ((fusion == NULL) || (theta_e == NULL) || (omega_e == NULL) || (omega_meas_mech == NULL))
    {
        return;
    }

    /* Electrical angle from encoder */
    *theta_e = (MatrixFloat)DFC_P_POLES * theta_m;

    /* Raw speed from encoder differentiation */
    delta = theta_m - fusion->theta_m_prev;
    while (delta > DFC_PI_F)
    {
        delta -= DFC_TWO_PI_F;
    }
    while (delta < -DFC_PI_F)
    {
        delta += DFC_TWO_PI_F;
    }

    omega_raw = (dt > DFC_ZERO_F) ? (delta / dt) : DFC_ZERO_F;

    /* Adaptive IIR filter for encoder speed */
    alpha = DFC_FusionAlpha(fabsf(fusion->omega_e_prev));
    iir_coeff = DFC_FUSION_IIR_LO + alpha * (DFC_FUSION_IIR_HI - DFC_FUSION_IIR_LO);
    fusion->omega_enc_filt = ((DFC_ONE_F - iir_coeff) * fusion->omega_enc_filt) + (iir_coeff * omega_raw);

    /* Use filtered encoder speed for control */
    *omega_meas_mech = fusion->omega_enc_filt;

    /* Fused electrical speed */
    omega_enc_e = (MatrixFloat)DFC_P_POLES * fusion->omega_enc_filt;
    *omega_e = ((DFC_ONE_F - alpha) * omega_enc_e) + (alpha * omega_smo_e);

    /* Update state */
    fusion->theta_m_prev = theta_m;
    fusion->omega_e_prev = *omega_e;
    fusion->alpha = alpha;
    fusion->omega_enc_mech = fusion->omega_enc_filt;
}


static void DFC_SMO_Step(
    DFC_SMO_T     * const smo,
    const MatrixFloat     v_alpha,
    const MatrixFloat     v_beta,
    const MatrixFloat     i_alpha,
    const MatrixFloat     i_beta,
    const MatrixFloat     dt,
    const uint32_T        warmup_cnt,
    MatrixFloat         * const omega_e_smo)
{
    MatrixFloat err_alpha, err_beta;
    MatrixFloat sw_alpha, sw_beta;
    MatrixFloat inv_L;
    MatrixFloat lpf_alpha;
    MatrixFloat theta_e_new, delta;
    const MatrixFloat L_avg = (DFC_L_D + DFC_L_Q) * (MatrixFloat)0.5f;

    if ((smo == NULL) || (omega_e_smo == NULL))
    {
        return;
    }

    inv_L = DFC_ONE_F / L_avg;
    lpf_alpha = dt / (DFC_SMO_TAU_E + dt);

    /* Current estimation errors */
    err_alpha = i_alpha - smo->i_hat_alpha;
    err_beta  = i_beta  - smo->i_hat_beta;

    /* Switching injection */
    sw_alpha = DFC_SMO_K * DFC_SMOSwitch(err_alpha);
    sw_beta  = DFC_SMO_K * DFC_SMOSwitch(err_beta);

    /* Observer current update (forward Euler) */
    smo->i_hat_alpha += dt * inv_L * (v_alpha - DFC_R_S * smo->i_hat_alpha - sw_alpha);
    smo->i_hat_beta  += dt * inv_L * (v_beta  - DFC_R_S * smo->i_hat_beta  - sw_beta);

    /* Back-EMF LPF */
    smo->e_hat_alpha += lpf_alpha * (sw_alpha - smo->e_hat_alpha);
    smo->e_hat_beta  += lpf_alpha * (sw_beta  - smo->e_hat_beta);

    /* Electrical angle from back-EMF */
    theta_e_new = atan2f(-smo->e_hat_alpha, smo->e_hat_beta);

    /* Unwrap for speed extraction */
    delta = theta_e_new - smo->theta_e_prev;
    while (delta > DFC_PI_F)
    {
        delta -= DFC_TWO_PI_F;
    }
    while (delta < -DFC_PI_F)
    {
        delta += DFC_TWO_PI_F;
    }

    /* Electrical speed estimate - gate during warmup */
    if ((dt > DFC_ZERO_F) && (warmup_cnt > DFC_SMO_WARMUP_STEPS))
    {
        smo->omega_e_hat = delta / dt;
    }
    else
    {
        smo->omega_e_hat = DFC_ZERO_F;
    }

    smo->theta_e_prev = theta_e_new;
    smo->theta_e_hat = theta_e_new;
    *omega_e_smo = smo->omega_e_hat;
}


static void DFC_VoltageLaw(
    const MatrixFloat     iq_ref,
    const MatrixFloat     diq_dt,
    const MatrixFloat     id_meas,
    const MatrixFloat     iq_meas,
    const MatrixFloat     omega_e,
    MatrixFloat         * const vd,
    MatrixFloat         * const vq)
{
    MatrixFloat vd_out, vq_out, magnitude, scale;

    if ((vd == NULL) || (vq == NULL))
    {
        return;
    }

    /* Differential flatness voltage equations (id_ref = 0 for MTPA) */
    vd_out = (DFC_R_S * DFC_ZERO_F)
           + (DFC_L_D * DFC_ZERO_F)
           - (omega_e * DFC_L_Q * iq_ref)
           + (DFC_KP_ID * (DFC_ZERO_F - id_meas));

    vq_out = (DFC_R_S * iq_ref)
           + (DFC_L_Q * diq_dt)
           + (omega_e * DFC_L_D * DFC_ZERO_F)
           + (omega_e * DFC_LAMBDA_PM)
           + (DFC_KP_IQ * (iq_ref - iq_meas));

    /* Hexagon voltage saturation */
    magnitude = sqrtf(vd_out * vd_out + vq_out * vq_out);
    if (magnitude > DFC_V_MAX)
    {
        scale = DFC_V_MAX / magnitude;
        vd_out *= scale;
        vq_out *= scale;
    }
    else
    {
        /* Within hexagon - no saturation required */
    }

    *vd = vd_out;
    *vq = vq_out;
}


/*********************************************************************************************************************/
/*------------------------------------------------- Public API -------------------------------------------------------*/
/*********************************************************************************************************************/

void DFC_Controller_Init(DFC_State_T * const s, const MatrixFloat dt)
{
    if (s != NULL)
    {
        /* Zero-initialise the entire struct */
        (void)memset(s, 0, sizeof(DFC_State_T));

        /* Explicitly initialise transform states */
        Clarke_Init(&s->clarke_state);
        Park_Init(&s->park_state);
        InvPark_Init(&s->inv_park_state);

        /* SpeedFusion initialisation */
        s->fusion.theta_m_prev = DFC_ZERO_F;
        s->fusion.omega_enc_filt = DFC_ZERO_F;
        s->fusion.omega_e_prev = DFC_ZERO_F;
        s->fusion.alpha = DFC_ZERO_F;
        s->fusion.omega_enc_mech = DFC_ZERO_F;

        /* SMO initialisation */
        s->smo.i_hat_alpha = DFC_ZERO_F;
        s->smo.i_hat_beta = DFC_ZERO_F;
        s->smo.e_hat_alpha = DFC_ZERO_F;
        s->smo.e_hat_beta = DFC_ZERO_F;
        s->smo.theta_e_hat = DFC_ZERO_F;
        s->smo.omega_e_hat = DFC_ZERO_F;
        s->smo.theta_e_prev = DFC_ZERO_F;

        /* Delayed voltages */
        s->v_alpha_prev = DFC_ZERO_F;
        s->v_beta_prev = DFC_ZERO_F;

        /* Reference trajectory */
        s->iq_ref_prev = DFC_ZERO_F;
        s->diq_filt = DFC_ZERO_F;

        /* Warmup counter */
        s->smo_warmup_cnt = 0U;

        /* Diagnostic logging */
        s->log_speed_ref   = DFC_ZERO_F;
        s->log_iq_ref      = DFC_ZERO_F;
        s->log_id          = DFC_ZERO_F;
        s->log_iq          = DFC_ZERO_F;
        s->log_alpha       = DFC_ZERO_F;
        s->log_omega_e     = DFC_ZERO_F;
        s->log_counter     = 0U;
        s->log_next_time   = DFC_LOG_INTERVAL;
    }
    else
    {
        /* NULL guard - MISRA C:2012 Rule 15.7 */
    }

    (void)dt;   /* Unused - retained for API consistency */
}


void DFC_Controller_Step(
    DFC_State_T        * const s,
    const DFC_Input_T  * const u,
    const MatrixFloat           dt,
    DFC_Output_T       * const y)
{
    MatrixFloat i_alpha, i_beta;
    MatrixFloat id_meas, iq_meas;
    MatrixFloat theta_e, omega_e, omega_meas_mech, omega_smo_e;
    MatrixFloat speed_err, iq_ref, diq_dt, vd, vq;
    MatrixFloat lpf_alpha;
    const MatrixFloat diq_tau = DFC_DIQ_TAU;

    if ((s == NULL) || (u == NULL) || (y == NULL))
    {
        return;
    }

    s->smo_warmup_cnt++;

    /* Clarke: abc -> alphabeta */
    Clarke_Step(&s->clarke_state,
                u->ia, u->ib, u->ic,
                &i_alpha, &i_beta);

    /* SMO step */
    DFC_SMO_Step(&s->smo,
                 s->v_alpha_prev, s->v_beta_prev,
                 i_alpha, i_beta,
                 dt, s->smo_warmup_cnt,
                 &omega_smo_e);

    /* SpeedFusion: theta_e (encoder), omega_e (complementary) */
    DFC_SpeedFusion_Update(&s->fusion,
                           u->theta_m,
                           omega_smo_e,
                           dt,
                           &theta_e,
                           &omega_e,
                           &omega_meas_mech);

    /* Speed P-loop -> iq_ref */
    speed_err = u->omega_ref_mech - omega_meas_mech;
    iq_ref = DFC_KP_SPEED * speed_err;
    iq_ref = DFC_Clamp(iq_ref, DFC_I_MAX);

    /* Current derivative (LPF-filtered finite difference) */
    lpf_alpha = dt / (diq_tau + dt);
    if (dt > DFC_ZERO_F)
    {
        diq_dt = (iq_ref - s->iq_ref_prev) / dt;
    }
    else
    {
        diq_dt = DFC_ZERO_F;
    }
    s->diq_filt = ((DFC_ONE_F - lpf_alpha) * s->diq_filt) + (lpf_alpha * diq_dt);

    /* Update previous reference */
    s->iq_ref_prev = iq_ref;

    /* Park transform: alphabeta -> dq */
    Park_Step(&s->park_state,
              i_alpha, i_beta, theta_e,
              &id_meas, &iq_meas);

    /* Flatness voltage law */
    DFC_VoltageLaw(iq_ref,
                   s->diq_filt,
                   id_meas,
                   iq_meas,
                   omega_e,
                   &vd, &vq);

    /* Inverse Park transform: dq -> alphabeta */
    InvPark_Step(&s->inv_park_state,
                 vd, vq, theta_e,
                 &y->v_alpha, &y->v_beta);

    /* Store voltages for SMO next step (z-1) */
    s->v_alpha_prev = y->v_alpha;
    s->v_beta_prev  = y->v_beta;

    /* Diagnostic logging at 1 kHz */
    if (dt > DFC_ZERO_F)
    {
        s->log_counter++;

        if (((MatrixFloat)s->log_counter * dt) >= s->log_next_time)
        {
            s->log_speed_ref   = u->omega_ref_mech * (MatrixFloat)60.0f / DFC_TWO_PI_F;
            s->log_iq_ref      = iq_ref;
            s->log_id          = id_meas;
            s->log_iq          = iq_meas;
            s->log_alpha       = s->fusion.alpha;
            s->log_omega_e     = omega_e;
            s->log_next_time  += DFC_LOG_INTERVAL;
        }
        else
        {
            /* Not yet time */
        }
    }
    else
    {
        /* dt = 0 - logging disabled */
    }
}


void DFC_Controller_Reset(DFC_State_T * const s)
{
    if (s != NULL)
    {
        DFC_Controller_Init(s, DFC_ZERO_F);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


void DFC_Controller_GetDiagnostics(
    const DFC_State_T * const s,
    MatrixFloat       * const speed_ref_rpm,
    MatrixFloat       * const iq_ref,
    MatrixFloat       * const id,
    MatrixFloat       * const iq,
    MatrixFloat       * const alpha,
    MatrixFloat       * const omega_e)
{
    if ((s != NULL) &&
        (speed_ref_rpm != NULL) &&
        (iq_ref != NULL) &&
        (id != NULL) &&
        (iq != NULL) &&
        (alpha != NULL) &&
        (omega_e != NULL))
    {
        *speed_ref_rpm = s->log_speed_ref;
        *iq_ref        = s->log_iq_ref;
        *id            = s->log_id;
        *iq            = s->log_iq;
        *alpha         = s->log_alpha;
        *omega_e       = s->log_omega_e;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}
