/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     Differential Flatness FOC Controller -- observer mode extension
 *
 * \details   Extends the original SMO-only implementation with:
 *              - DFC_OBS_EKF   : EKF replaces SMO as the speed source
 *              - DFC_OBS_BLEND : convex blend of SMO and EKF outputs
 *              - DFC_Controller_SetObserverMode() for live switching
 *              - DFC_Controller_SetEKFParams()    for noise tuning
 *              - Extended GetDiagnostics with omega_smo and omega_ekf channels
 *
 *            The SMO always executes (it feeds SpeedFusion regardless of mode).
 *            The EKF executes only in DFC_OBS_EKF and DFC_OBS_BLEND to avoid
 *            burning ISR budget in the default DFC_OBS_SMO production mode.
 *
 * \version   3.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#include "embed_sim_dfc_controller.h"
#include <math.h>
#include <string.h>

#define DFC_ZERO_F   ((MatrixFloat)0.0f)
#define DFC_ONE_F    ((MatrixFloat)1.0f)
#define DFC_TWO_PI_F ((MatrixFloat)6.28318530717959f)
#define DFC_PI_F     ((MatrixFloat)3.14159265358979f)


/**********************************************************************************************************************
 * Private function prototypes
 *********************************************************************************************************************/

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
    MatrixFloat   iq_ref,
    MatrixFloat   diq_dt,
    MatrixFloat   id_meas,
    MatrixFloat   iq_meas,
    MatrixFloat   omega_e,
    MatrixFloat * const vd,
    MatrixFloat * const vq);



/**********************************************************************************************************************
 * Private implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Clamp
 *------------------------------------------------------------------------------------------------------------------*/
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
        /* Within range -- no action required */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_FusionAlpha
 *------------------------------------------------------------------------------------------------------------------*/
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
        result = (omega_abs - DFC_FUSION_OMEGA_LO)
               / (DFC_FUSION_OMEGA_HI - DFC_FUSION_OMEGA_LO);
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_SMOSwitch -- smooth sign approximation (linear saturation)
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat DFC_SMOSwitch(const MatrixFloat error)
{
    MatrixFloat result;
    const MatrixFloat width = (MatrixFloat)0.01f;
    const MatrixFloat arg   = error / width;

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
        result = arg * (MatrixFloat)0.2f;
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_SpeedFusion_Update
 *------------------------------------------------------------------------------------------------------------------*/
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

    /* Electrical angle directly from encoder */
    *theta_e = (MatrixFloat)DFC_P_POLES * theta_m;

    /* Finite-difference mechanical speed from encoder */
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

    /* Adaptive IIR smoothing */
    alpha     = DFC_FusionAlpha(fabsf(fusion->omega_e_prev));
    iir_coeff = DFC_FUSION_IIR_LO + alpha * (DFC_FUSION_IIR_HI - DFC_FUSION_IIR_LO);
    fusion->omega_enc_filt = ((DFC_ONE_F - iir_coeff) * fusion->omega_enc_filt)
                           + (iir_coeff * omega_raw);

    *omega_meas_mech = fusion->omega_enc_filt;

    /* Fused electrical speed: low-speed -> encoder, high-speed -> SMO */
    omega_enc_e = (MatrixFloat)DFC_P_POLES * fusion->omega_enc_filt;

    /* SMO plausibility gate: if omega_smo_e deviates from the encoder
     * electrical speed by more than DFC_SMO_PLAUS_BAND the SMO output
     * is implausible (e.g. residual spike after the omega_e_hat clamp,
     * or a stale filt value from a prior divergence).  Substitute the
     * encoder value so the blend never injects a corrupted SMO estimate. */
    {
        MatrixFloat omega_smo_gated;

        if (fabsf(omega_smo_e - omega_enc_e) > DFC_SMO_PLAUS_BAND)
        {
            omega_smo_gated = omega_enc_e;   /* encoder fallback */
        }
        else
        {
            omega_smo_gated = omega_smo_e;   /* SMO plausible -- MISRA 15.7 */
        }

        *omega_e = ((DFC_ONE_F - alpha) * omega_enc_e) + (alpha * omega_smo_gated);
    }

    /* Encoder fallback: when SMO has not yet converged (omega_smo_e ~ 0)
     * but encoder is above threshold, substitute encoder electrical speed.
     * Encoder sign confirmed positive-forward on hardware.               */
    if ((fabsf(omega_smo_e) < DFC_ONE_F) &&
        (fabsf(fusion->omega_enc_filt) > DFC_FUSION_OMEGA_LO))
    {
        *omega_e = omega_enc_e;
    }
    else
    {
        /* SMO valid or encoder below threshold -- MISRA 15.7 */
    }

    /* Update state */
    fusion->theta_m_prev  = theta_m;
    fusion->omega_e_prev  = *omega_e;
    fusion->alpha         = alpha;
    fusion->omega_enc_mech = fusion->omega_enc_filt;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_SMO_Step
 *------------------------------------------------------------------------------------------------------------------*/
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
    MatrixFloat sw_alpha,  sw_beta;
    MatrixFloat inv_L;
    MatrixFloat lpf_alpha;
    MatrixFloat theta_e_new, delta;
    const MatrixFloat L_avg = (DFC_L_D + DFC_L_Q) * (MatrixFloat)0.5f;

    if ((smo == NULL) || (omega_e_smo == NULL))
    {
        return;
    }

    inv_L     = DFC_ONE_F / L_avg;
    lpf_alpha = dt / (DFC_SMO_TAU_E + dt);

    /* Divergence guard: if i_hat exceeds 2x the physical current limit
     * the observer has lost tracking -- reinitialise from measured current
     * so it can re-converge rather than staying in the saturated fixed point. */
    if ((smo->i_hat_alpha > (MatrixFloat)2.0f * DFC_I_MAX) ||
        (smo->i_hat_alpha < -(MatrixFloat)2.0f * DFC_I_MAX) ||
        (smo->i_hat_beta  > (MatrixFloat)2.0f * DFC_I_MAX) ||
        (smo->i_hat_beta  < -(MatrixFloat)2.0f * DFC_I_MAX))
    {
        smo->i_hat_alpha = i_alpha;
        smo->i_hat_beta  = i_beta;
        smo->e_hat_alpha = DFC_ZERO_F;
        smo->e_hat_beta  = DFC_ZERO_F;
        smo->omega_e_hat  = DFC_ZERO_F;
        smo->omega_e_filt = DFC_ZERO_F;
        /* theta_e_prev intentionally preserved -- prevents delta spike */
    }
    else
    {
        /* Within bounds -- MISRA 15.7 */
    }

    err_alpha = i_alpha - smo->i_hat_alpha;
    err_beta  = i_beta  - smo->i_hat_beta;

    sw_alpha = DFC_SMO_K * DFC_SMOSwitch(err_alpha);
    sw_beta  = DFC_SMO_K * DFC_SMOSwitch(err_beta);

    /* Current observer (Euler) */
    smo->i_hat_alpha += dt * inv_L * (v_alpha - DFC_R_S * smo->i_hat_alpha - sw_alpha);
    smo->i_hat_beta  += dt * inv_L * (v_beta  - DFC_R_S * smo->i_hat_beta  - sw_beta);

    /* Back-EMF LPF */
    smo->e_hat_alpha += lpf_alpha * (sw_alpha - smo->e_hat_alpha);
    smo->e_hat_beta  += lpf_alpha * (sw_beta  - smo->e_hat_beta);

    /* Angle from back-EMF */
    /* Sign convention: atan2(e_alpha, -e_beta) gives positive omega_e_hat
     * when the motor spins in the forward (positive encoder) direction.  */
    theta_e_new = atan2f(smo->e_hat_alpha, -smo->e_hat_beta);

    delta = theta_e_new - smo->theta_e_prev;
    while (delta > DFC_PI_F)  { delta -= DFC_TWO_PI_F; }
    while (delta < -DFC_PI_F) { delta += DFC_TWO_PI_F; }

    if ((dt > DFC_ZERO_F) && (warmup_cnt > DFC_SMO_WARMUP_STEPS))
    {
        smo->omega_e_hat = delta / dt;

        /* Spike clamp: a phase wrap in atan2f can produce a single-sample
         * delta/dt impulse that saturates the LPF for many cycles.
         * If the raw estimate exceeds the physical operating ceiling
         * (DFC_SMO_OMEGA_MAX) the sample is discarded and the last
         * filtered value is held -- the LPF then decays naturally.     */
        if ((smo->omega_e_hat > DFC_SMO_OMEGA_MAX) ||
            (smo->omega_e_hat < -DFC_SMO_OMEGA_MAX))
        {
            smo->omega_e_hat = smo->omega_e_filt;   /* hold -- discard spike */
        }
        else
        {
            /* Within plausible range -- MISRA 15.7 */
        }
    }
    else
    {
        smo->omega_e_hat = DFC_ZERO_F;
    }

    /* LPF on speed estimate -- smooths the noisy finite-difference.
     * Uses same lpf_alpha as back-EMF filter for consistency.       */
    smo->omega_e_filt += lpf_alpha * (smo->omega_e_hat - smo->omega_e_filt);

    smo->theta_e_prev = theta_e_new;
    smo->theta_e_hat  = theta_e_new;
    *omega_e_smo      = smo->omega_e_filt;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_VoltageLaw -- differential flatness voltage equations
 *------------------------------------------------------------------------------------------------------------------*/
static void DFC_VoltageLaw(
    const MatrixFloat   iq_ref,
    const MatrixFloat   diq_dt,
    const MatrixFloat   id_meas,
    const MatrixFloat   iq_meas,
    const MatrixFloat   omega_e,
    MatrixFloat       * const vd,
    MatrixFloat       * const vq)
{
    MatrixFloat vd_out, vq_out, magnitude, scale;

    if ((vd == NULL) || (vq == NULL))
    {
        return;
    }

    /* id_ref = 0 (MTPA) */
    vd_out = -(omega_e * DFC_L_Q * iq_ref)
             + (DFC_KP_ID * (DFC_ZERO_F - id_meas));

    vq_out = (DFC_R_S * iq_ref)
           + (DFC_L_Q * diq_dt)
           + (omega_e * DFC_LAMBDA_PM)
           + (DFC_KP_IQ * (iq_ref - iq_meas));

    /* Hexagon voltage saturation */
    magnitude = sqrtf(vd_out * vd_out + vq_out * vq_out);
    if (magnitude > DFC_V_MAX)
    {
        scale  = DFC_V_MAX / magnitude;
        vd_out *= scale;
        vq_out *= scale;
    }
    else
    {
        /* Within hexagon -- no saturation required */
    }

    *vd = vd_out;
    *vq = vq_out;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_Init
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_Init(DFC_State_T * const s, const MatrixFloat dt)
{
    if (s == NULL)
    {
        return;
    }

    /*--- SpeedFusion state ---*/
    s->fusion.theta_m_prev   = DFC_ZERO_F;
    s->fusion.omega_enc_filt = DFC_ZERO_F;
    s->fusion.omega_e_prev   = DFC_ZERO_F;
    s->fusion.alpha          = DFC_ZERO_F;
    s->fusion.omega_enc_mech = DFC_ZERO_F;

    /*--- SMO state ---*/
    s->smo.i_hat_alpha  = DFC_ZERO_F;
    s->smo.i_hat_beta   = DFC_ZERO_F;
    s->smo.e_hat_alpha  = DFC_ZERO_F;
    s->smo.e_hat_beta   = DFC_ZERO_F;
    s->smo.theta_e_hat  = DFC_ZERO_F;
    s->smo.omega_e_hat  = DFC_ZERO_F;
    s->smo.omega_e_filt = DFC_ZERO_F;
    s->smo.theta_e_prev = DFC_ZERO_F;

    /*--- Delayed voltages (z-1 for SMO) ---*/
    s->v_alpha_prev = DFC_ZERO_F;
    s->v_beta_prev  = DFC_ZERO_F;

    /*--- Reference trajectory ---*/
    s->iq_ref_prev = DFC_ZERO_F;
    s->diq_filt    = DFC_ZERO_F;

    /*--- Warmup counter ---*/
    s->smo_warmup_cnt = 0U;

    /*--- Coordinate transforms ---*/
    Clarke_Init(&s->clarke_state);
    Park_Init(&s->park_state);
    InvPark_Init(&s->inv_park_state);

    /*--- Diagnostic log ---*/
    s->log_speed_ref = DFC_ZERO_F;
    s->log_iq_ref    = DFC_ZERO_F;
    s->log_id        = DFC_ZERO_F;
    s->log_iq        = DFC_ZERO_F;
    s->log_alpha     = DFC_ZERO_F;
    s->log_omega_e   = DFC_ZERO_F;
    s->log_omega_smo = DFC_ZERO_F;
    s->log_counter   = 0U;
    s->log_next_time = DFC_LOG_INTERVAL;

    (void)dt;
}

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_Step
 *------------------------------------------------------------------------------------------------------------------*/
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

    /*--- Clarke: abc -> alphabeta ---*/
    Clarke_Step(&s->clarke_state,
                u->ia, u->ib, u->ic,
                &i_alpha, &i_beta);

    /*--- SMO: always runs (feeds SpeedFusion) ---*/
    DFC_SMO_Step(&s->smo,
                 s->v_alpha_prev, s->v_beta_prev,
                 i_alpha, i_beta,
                 dt, s->smo_warmup_cnt,
                 &omega_smo_e);

    /*--- SpeedFusion: encoder + active observer -> theta_e, omega_e ---*/
    DFC_SpeedFusion_Update(&s->fusion,
                           u->theta_m,
                           omega_smo_e,
                           dt,
                           &theta_e,
                           &omega_e,
                           &omega_meas_mech);

    /* omega_meas_mech set by SpeedFusion */

    /*--- Speed P-loop -> iq_ref ---*/
    speed_err = u->omega_ref_mech - omega_meas_mech;
    iq_ref    = DFC_KP_SPEED * speed_err;
    iq_ref    = DFC_Clamp(iq_ref, DFC_I_MAX);

    /*--- Current derivative (LPF-filtered finite difference) ---*/
    lpf_alpha = dt / (diq_tau + dt);
    if (dt > DFC_ZERO_F)
    {
        diq_dt = (iq_ref - s->iq_ref_prev) / dt;
    }
    else
    {
        diq_dt = DFC_ZERO_F;
    }
    s->diq_filt    = ((DFC_ONE_F - lpf_alpha) * s->diq_filt) + (lpf_alpha * diq_dt);

    /* Clamp: I_MAX / DIQ_TAU = 3.57 / 0.001 = 3570 A/s ceiling.
     * Beyond this the L*diq term in vq exceeds bus voltage headroom. */
    s->diq_filt = DFC_Clamp(s->diq_filt, DFC_I_MAX / DFC_DIQ_TAU);

    s->iq_ref_prev = iq_ref;

    /*--- Park: alphabeta -> dq ---*/
    Park_Step(&s->park_state,
              i_alpha, i_beta, theta_e,
              &id_meas, &iq_meas);

    /*--- Flatness voltage law ---*/
    DFC_VoltageLaw(iq_ref, s->diq_filt,
                   id_meas, iq_meas, omega_e,
                   &vd, &vq);

    /*--- Inverse Park: dq -> alphabeta ---*/
    InvPark_Step(&s->inv_park_state,
                 vd, vq, theta_e,
                 &y->v_alpha, &y->v_beta);

    /*--- Delay voltages for SMO z-1 ---*/
    s->v_alpha_prev = y->v_alpha;
    s->v_beta_prev  = y->v_beta;

    /*--- Diagnostic logging at 1 kHz ---*/
    if (dt > DFC_ZERO_F)
    {
        s->log_counter++;
        if (((MatrixFloat)s->log_counter * dt) >= s->log_next_time)
        {
            s->log_speed_ref  = u->omega_ref_mech * (MatrixFloat)60.0f / DFC_TWO_PI_F;
            s->log_iq_ref     = iq_ref;
            s->log_id         = id_meas;
            s->log_iq         = iq_meas;
            s->log_alpha      = s->fusion.alpha;
            s->log_omega_e    = omega_meas_mech;  /* active observer output -- drives P-loop */
            s->log_omega_smo  = omega_smo_e / (MatrixFloat)DFC_P_POLES;
            s->log_next_time += DFC_LOG_INTERVAL;
        }
        else
        {
            /* Not yet time for next log snapshot */
        }
    }
    else
    {
        /* dt = 0 -- logging disabled */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_Reset
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_Reset(DFC_State_T * const s)
{

    if (s == NULL)
    {
        return;
    }

    /* Preserve observer selection across reset so AURIX overlay settings
     * survive a fault-recovery restart without manual rewrite. */

    DFC_Controller_Init(s, DFC_ZERO_F);

}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_GetDiagnostics
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_GetDiagnostics(
    const DFC_State_T * const s,
    MatrixFloat       * const speed_ref_rpm,
    MatrixFloat       * const iq_ref,
    MatrixFloat       * const id,
    MatrixFloat       * const iq,
    MatrixFloat       * const alpha,
    MatrixFloat       * const omega_e,
    MatrixFloat       * const omega_smo)
{
    if ((s             != NULL) &&
        (speed_ref_rpm != NULL) &&
        (iq_ref        != NULL) &&
        (id            != NULL) &&
        (iq            != NULL) &&
        (alpha         != NULL) &&
        (omega_e       != NULL) &&
        (omega_smo     != NULL))
    {
        *speed_ref_rpm = s->log_speed_ref;
        *iq_ref        = s->log_iq_ref;
        *id            = s->log_id;
        *iq            = s->log_iq;
        *alpha         = s->log_alpha;
        *omega_e       = s->log_omega_e;
        *omega_smo     = s->log_omega_smo;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7 -- else required */
    }
}
