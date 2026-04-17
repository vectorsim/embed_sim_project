/**
 * @file      embed_sim_mpc_controller.c
 * @brief     Model Predictive Control FOC Controller -- NANOTEC DB42S02
 * @details   Implements 3-state receding-horizon MPC with analytical solution.
 *            State vector: x = [id, iq, omega_m]
 *            Input vector: u = [vd, vq]
 * @version   2.0.0
 * @copyright Copyright (C) EmbedSim 2025
 *
 * @par MISRA C:2012 Compliance:
 *      - Rule 7.2: All float literals have 'f' suffix
 *      - Rule 8.1: All types explicit via MatrixFloat/uint32_T
 *      - Rule 10.4: No mixed-mode arithmetic
 *      - Rule 15.5: Single return per function
 *      - Rule 15.7: Every if-else chain has final else
 *      - Rule 17.5: All pointer parameters checked for NULL
 */

#include "embed_sim_mpc_controller.h"
#include <stddef.h>   /* For NULL definition (MISRA Rule 20.9) */
#include <math.h>     /* For fabsf, fminf, fmaxf, tanhf */

/* Local constants (MISRA Rule 7.2) */
#define MPC_ZERO_F   ((MatrixFloat)0.0f)
#define MPC_ONE_F    ((MatrixFloat)1.0f)
#define MPC_TWO_PI_F ((MatrixFloat)6.28318530717959f)
#define MPC_PI_F     ((MatrixFloat)3.14159265358979f)
#define MPC_BOUNDARY_WIDTH ((MatrixFloat)0.01f)  /**< SMO boundary layer [A] */


/*********************************************************************************************************************/
/*                                              Static Helper Functions                                              */
/*********************************************************************************************************************/

/**
 * @brief   Symmetric magnitude clamp (single limit)
 * @param[in] value Value to clamp
 * @param[in] limit Positive magnitude limit
 * @return   value clamped to [-limit, +limit]
 *
 * @par MISRA Compliance:
 *      - Rule 15.7: If-else chain has final else
 */
static MatrixFloat MPC_Clamp(const MatrixFloat value, const MatrixFloat limit)
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
        /* Within range - no action required (MISRA Rule 15.7) */
    }

    return result;
}


/**
 * @brief   Symmetric clamp with independent lower and upper bounds
 * @param[in] value Value to clamp
 * @param[in] min_val Minimum allowed value
 * @param[in] max_val Maximum allowed value
 * @return   value clamped to [min_val, max_val]
 *
 * @par MISRA Compliance:
 *      - Rule 15.7: If-else chain has final else
 */
static MatrixFloat MPC_ClampMinMax(const MatrixFloat value,
                                    const MatrixFloat min_val,
                                    const MatrixFloat max_val)
{
    MatrixFloat result = value;

    if (result > max_val)
    {
        result = max_val;
    }
    else if (result < min_val)
    {
        result = min_val;
    }
    else
    {
        /* Within range - no action required (MISRA Rule 15.7) */
    }

    return result;
}


/**
 * @brief   Encoder speed estimator update
 * @param[in,out] enc     Encoder state structure
 * @param[in]     theta_m Mechanical angle from encoder [rad]
 * @param[in]     dt      Sampling period [s]
 * @return        IIR-filtered mechanical speed [rad/s]
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: Checks enc for NULL
 *      - Rule 15.5: Single return point
 */
static MatrixFloat MPC_EncSpeed_Update(MPC_EncSpeed_T* const enc,
                                        const MatrixFloat theta_m,
                                        const MatrixFloat dt)
{
    MatrixFloat delta;
    MatrixFloat omega_raw;
    MatrixFloat result;

    /* NULL check (MISRA Rule 17.5) */
    if (enc == NULL)
    {
        result = MPC_ZERO_F;
    }
    else if (dt <= MPC_ZERO_F)
    {
        result = enc->omega_filt;
    }
    else
    {
        /* Unwrap angle delta to (-pi, +pi] */
        delta = theta_m - enc->theta_m_prev;

        while (delta > MPC_PI_F)
        {
            delta -= MPC_TWO_PI_F;
        }
        while (delta < -MPC_PI_F)
        {
            delta += MPC_TWO_PI_F;
        }

        enc->theta_m_unwrapped += delta;

        /* Finite-difference speed */
        omega_raw = delta / dt;

        /* IIR smoothing */
        enc->omega_filt = ((MPC_ONE_F - MPC_ENC_IIR) * enc->omega_filt)
                        + (MPC_ENC_IIR * omega_raw);

        /* Persist state */
        enc->theta_m_prev = theta_m;

        result = enc->omega_filt;
    }

    return result;
}


/**
 * @brief   SMO switching function (tanh with boundary layer)
 * @param[in] error Current estimation error [A]
 * @return   Smooth switching value in [-1.0f, +1.0f]
 */
static MatrixFloat MPC_SMOSwitch(const MatrixFloat error)
{
    return tanhf(error / MPC_BOUNDARY_WIDTH);
}


/**
 * @brief   Sliding Mode Observer step
 * @param[in,out] smo      SMO state structure
 * @param[in]     v_alpha  Alpha voltage from previous step [V]
 * @param[in]     v_beta   Beta voltage from previous step [V]
 * @param[in]     i_alpha  Measured alpha current [A]
 * @param[in]     i_beta   Measured beta current [A]
 * @param[in]     dt       Sampling period [s]
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: Checks smo for NULL
 *      - Rule 15.5: Single return point
 */
static void MPC_SMO_Step(MPC_SMO_T* const smo,
                          const MatrixFloat v_alpha,
                          const MatrixFloat v_beta,
                          const MatrixFloat i_alpha,
                          const MatrixFloat i_beta,
                          const MatrixFloat dt)
{
    MatrixFloat err_alpha;
    MatrixFloat err_beta;
    MatrixFloat sw_alpha;
    MatrixFloat sw_beta;
    MatrixFloat inv_L;
    MatrixFloat alpha;

    /* NULL check (MISRA Rule 17.5) */
    if (smo == NULL)
    {
        return;
    }

    if (dt <= MPC_ZERO_F)
    {
        return;
    }

    inv_L = MPC_ONE_F / MPC_L;
    alpha = smo->alpha_lpf;

    /* Current estimation errors */
    err_alpha = i_alpha - smo->i_alpha_hat;
    err_beta  = i_beta  - smo->i_beta_hat;

    /* Smooth switching signals */
    sw_alpha = MPC_SMO_K * MPC_SMOSwitch(err_alpha);
    sw_beta  = MPC_SMO_K * MPC_SMOSwitch(err_beta);

    /* Current observer (Forward Euler) */
    smo->i_alpha_hat += dt * inv_L * (v_alpha - MPC_R_S * smo->i_alpha_hat - sw_alpha);
    smo->i_beta_hat  += dt * inv_L * (v_beta  - MPC_R_S * smo->i_beta_hat  - sw_beta);

    /* Back-EMF LPF */
    smo->e_alpha_filt += alpha * (sw_alpha - smo->e_alpha_filt);
    smo->e_beta_filt  += alpha * (sw_beta  - smo->e_beta_filt);
}


/**
 * @brief   Analytical MPC solver (closed-form, O(N))
 * @param[in]  x0        Initial state [id, iq, omega]
 * @param[in]  omega_ref Speed reference [rad/s]
 * @param[in]  ed_hat    D-axis BEMF feedforward [V]
 * @param[in]  eq_hat    Q-axis BEMF feedforward [V]
 * @param[in]  dt        Sampling period [s]
 * @param[out] vd        D-axis voltage command [V]
 * @param[out] vq        Q-axis voltage command [V]
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: Checks pointer parameters for NULL
 *      - Rule 15.5: Single return point
 */
static void MPC_SolveMPC(const MPC_State_T* const x0,
                          const MatrixFloat omega_ref,
                          const MatrixFloat ed_hat,
                          const MatrixFloat eq_hat,
                          const MatrixFloat dt,
                          MatrixFloat* const vd,
                          MatrixFloat* const vq)
{
    MatrixFloat omega_e;
    MatrixFloat inv_L;
    MatrixFloat a;
    MatrixFloat b;
    MatrixFloat dt_J;
    MatrixFloat id_free;
    MatrixFloat iq_free;
    MatrixFloat omega_free;
    MatrixFloat bk;
    MatrixFloat ek;
    MatrixFloat sum_bk_err_d;
    MatrixFloat sum_bk_err_q;
    MatrixFloat sum_ek_err;
    MatrixFloat sum_bk2;
    MatrixFloat sum_ek2;
    MatrixFloat denom_d;
    MatrixFloat denom_q;
    MatrixFloat vd_mpc;
    MatrixFloat vq_mpc;
    int32_T k;

    /* NULL checks (MISRA Rule 17.5) */
    if ((x0 == NULL) || (vd == NULL) || (vq == NULL))
    {
        return;
    }

    /* Pre-compute constants */
    omega_e = (MatrixFloat)MPC_P_POLES * x0->omega;
    inv_L   = MPC_ONE_F / MPC_L;
    a       = MPC_ONE_F - dt * MPC_R_S * inv_L;
    b       = dt * inv_L;
    dt_J    = dt / MPC_J_ROTOR;

    /* Free-run trajectory (u = 0) */
    id_free    = MPC_Clamp(x0->id, MPC_I_MAX);
    iq_free    = MPC_Clamp(x0->iq, MPC_I_MAX);
    omega_free = x0->omega;

    /* Accumulators initialisation */
    bk = MPC_ZERO_F;
    ek = MPC_ZERO_F;
    sum_bk_err_d = MPC_ZERO_F;
    sum_bk_err_q = MPC_ZERO_F;
    sum_ek_err   = MPC_ZERO_F;
    sum_bk2      = MPC_ZERO_F;
    sum_ek2      = MPC_ZERO_F;

    for (k = 0; k < (int32_T)MPC_N; k++)
    {
        MatrixFloat f_d;
        MatrixFloat f_q;
        MatrixFloat f_omega;

        /* Cross-coupling disturbances.
         * FIX: f_q must include the permanent-magnet flux linkage term
         *      (-omega_e * MPC_LAMBDA_PM) which is the dominant BEMF component.
         *      Without it the MPC free-run trajectory underestimates the q-axis
         *      back-EMF by  omega_e * lambda_pm  (= 1.17 V at 2000 RPM), causing
         *      the solver to underestimate the required vq and resulting in
         *      iq_SS ≈ 0 A under load.
         * PYTHON ALIGNMENT: MPCControllerBlock.compute_py() line
         *      f_q = dt/L * (-we*L*id_free - we*lambda_pm)  — identical. */
        f_d     = dt * inv_L * ( omega_e * MPC_L * iq_free);
        f_q     = dt * inv_L * ((-omega_e * MPC_L * id_free) - (omega_e * MPC_LAMBDA_PM));
        f_omega = dt_J * (MPC_KT * iq_free - MPC_B_FRICTION * omega_free);

        /* Propagate free-run states */
        id_free    = a * id_free    + f_d;
        iq_free    = a * iq_free    + f_q;
        omega_free = omega_free     + f_omega;
        id_free    = MPC_Clamp(id_free, MPC_I_MAX);
        iq_free    = MPC_Clamp(iq_free, MPC_I_MAX);

        /* Step-response update */
        bk = bk * a + b;
        ek += dt_J * MPC_KT * bk;

        /* Gradient accumulation */
        sum_bk_err_d += bk * (MPC_ZERO_F - id_free);
        sum_bk_err_q += bk * (MPC_ZERO_F - iq_free);
        sum_ek_err   += ek * (omega_ref - omega_free);
        sum_bk2      += bk * bk;
        sum_ek2      += ek * ek;
    }

    /* Analytical optimal inputs */
    denom_d = MPC_Q_ID * sum_bk2 + MPC_R_VD;
    denom_q = MPC_Q_OMEGA * sum_ek2 + MPC_Q_IQ * sum_bk2 + MPC_R_VQ;

    if (denom_d > (MatrixFloat)1e-30f)
    {
        vd_mpc = (MPC_Q_ID * sum_bk_err_d) / denom_d;
    }
    else
    {
        vd_mpc = MPC_ZERO_F;
    }

    if (denom_q > (MatrixFloat)1e-30f)
    {
        vq_mpc = (MPC_Q_OMEGA * sum_ek_err + MPC_Q_IQ * sum_bk_err_q) / denom_q;
    }
    else
    {
        vq_mpc = MPC_ZERO_F;
    }

    /* BEMF feedforward + hexagon clamp */
    *vd = MPC_Clamp(vd_mpc + ed_hat, MPC_V_MAX);
    *vq = MPC_Clamp(vq_mpc + eq_hat, MPC_V_MAX);
}


/*********************************************************************************************************************/
/*                                              Public API Functions                                                 */
/*********************************************************************************************************************/

/**
 * @brief   Initialise all controller state to zero
 * @param[out] s   Controller state (must not be NULL)
 * @param[in]  dt  Nominal sampling period [s]
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: Checks s for NULL
 *      - Rule 9.1: All members explicitly initialised
 */
void MPC_Controller_Init(MPC_Controller_T* const s, const MatrixFloat dt)
{
    if (s == NULL)
    {
        return;
    }

    /* Encoder state */
    s->enc.theta_m_prev      = MPC_ZERO_F;
    s->enc.theta_m_unwrapped = MPC_ZERO_F;
    s->enc.omega_filt        = MPC_ZERO_F;

    /* SMO state */
    s->smo.i_alpha_hat  = MPC_ZERO_F;
    s->smo.i_beta_hat   = MPC_ZERO_F;
    s->smo.e_alpha_filt = MPC_ZERO_F;
    s->smo.e_beta_filt  = MPC_ZERO_F;

    /* Pre-compute SMO LPF coefficient */
    {
        MatrixFloat wc = (MatrixFloat)2.0f * MPC_PI_F * MPC_SMO_FC;
        s->smo.alpha_lpf = wc * dt / (MPC_ONE_F + wc * dt);
    }

    /* Internal state */
    s->v_alpha_prev       = MPC_ZERO_F;
    s->v_beta_prev        = MPC_ZERO_F;
    s->iq_limit           = MPC_ZERO_F;
    s->speed_err_integral = MPC_ZERO_F;

    /* Diagnostic log */
    s->log_speed_ref  = MPC_ZERO_F;
    s->log_speed      = MPC_ZERO_F;
    s->log_id         = MPC_ZERO_F;
    s->log_iq         = MPC_ZERO_F;
    s->log_vd         = MPC_ZERO_F;
    s->log_vq         = MPC_ZERO_F;
    s->log_counter    = 0U;
    s->log_next_time  = MPC_ZERO_F;

    /* Coordinate transforms */
    Clarke_Init(&s->clarke_state);
    Park_Init(&s->park_state);
    Park_Init(&s->park_emf_state);
    InvPark_Init(&s->inv_park_state);
}


/**
 * @brief   Execute one complete MPC step
 * @param[in,out] s   Controller state (must not be NULL)
 * @param[in]     u   Input structure (must not be NULL)
 * @param[in]     dt  Sampling period [s]
 * @param[out]    y   Output structure (must not be NULL)
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: All pointer parameters checked for NULL
 *      - Rule 15.5: Single return point
 */
void MPC_Controller_Step(MPC_Controller_T* const s,
                          const MPC_Input_T* const u,
                          const MatrixFloat dt,
                          MPC_Output_T* const y)
{
    MatrixFloat i_alpha;
    MatrixFloat i_beta;
    MatrixFloat id_meas;
    MatrixFloat iq_meas;
    MatrixFloat theta_e;
    MatrixFloat omega_m;
    MatrixFloat e_alpha_filt;
    MatrixFloat e_beta_filt;
    MatrixFloat ed_hat_raw;
    MatrixFloat eq_hat_raw;
    MatrixFloat ed_hat;
    MatrixFloat eq_hat;
    MatrixFloat omega_e;
    MatrixFloat bemf_max;
    MatrixFloat vd;
    MatrixFloat vq;
    MatrixFloat vq_lim;
    MatrixFloat speed_err;
    MatrixFloat v_alpha;
    MatrixFloat v_beta;
    MPC_State_T x0;

    /* NULL checks (MISRA Rule 17.5) */
    if ((s == NULL) || (u == NULL) || (y == NULL))
    {
        return;
    }

    /* Electrical angle from encoder */
    theta_e = (MatrixFloat)MPC_P_POLES * u->theta_m;

    /* Encoder speed estimate */
    omega_m = MPC_EncSpeed_Update(&s->enc, u->theta_m, dt);

    /* Clarke transform: abc -> alphabeta */
    Clarke_Step(&s->clarke_state, u->ia, u->ib, u->ic, &i_alpha, &i_beta);

    /* SMO step */
    MPC_SMO_Step(&s->smo, s->v_alpha_prev, s->v_beta_prev,
                 i_alpha, i_beta, dt);

    e_alpha_filt = s->smo.e_alpha_filt;
    e_beta_filt  = s->smo.e_beta_filt;

    /* Park transform: alphabeta -> dq (currents) */
    Park_Step(&s->park_state, i_alpha, i_beta, theta_e, &id_meas, &iq_meas);

    /* Park transform on back-EMF for feedforward */
    Park_Step(&s->park_emf_state, e_alpha_filt, e_beta_filt, theta_e,
              &ed_hat_raw, &eq_hat_raw);

    /* Physical BEMF clamp */
    omega_e = (MatrixFloat)MPC_P_POLES * omega_m;
    bemf_max = (MatrixFloat)fabs(omega_e) * MPC_LAMBDA_PM;
    ed_hat = MPC_Clamp(ed_hat_raw, bemf_max);
    eq_hat = MPC_Clamp(eq_hat_raw, bemf_max);

    /* Soft-start ramp */
    {
        MatrixFloat iq_limit_new = s->iq_limit + MPC_I_MAX * dt / MPC_SOFTSTART_T;
        s->iq_limit = fminf(MPC_I_MAX, iq_limit_new);
    }

    /* MPC solver */
    x0.id    = MPC_Clamp(id_meas, MPC_I_MAX);
    x0.iq    = MPC_Clamp(iq_meas, MPC_I_MAX);
    x0.omega = omega_m;
    MPC_SolveMPC(&x0, u->omega_ref_mech, ed_hat, eq_hat, dt, &vd, &vq);

    /* Soft-start vq limit */
    vq_lim = (s->iq_limit / MPC_I_MAX) * MPC_V_MAX;
    vq = MPC_Clamp(vq, vq_lim);

    /* Speed-error integral correction (anti-windup) */
    speed_err = u->omega_ref_mech - omega_m;
    s->speed_err_integral += speed_err * dt;

    {
        MatrixFloat head = MPC_ClampMinMax(MPC_V_MAX - (MatrixFloat)fabs(vq),
                                           MPC_ZERO_F, MPC_V_MAX);
        MatrixFloat int_max = head / (MPC_KI_V + (MatrixFloat)1e-30f);
        s->speed_err_integral = MPC_Clamp(s->speed_err_integral, int_max);
    }

    vq = MPC_Clamp(vq + MPC_KI_V * s->speed_err_integral, MPC_V_MAX);

    /* Inverse Park: dq -> alphabeta */
    InvPark_Step(&s->inv_park_state, vd, vq, theta_e, &v_alpha, &v_beta);

    /* Latch voltages for next step's SMO (z-1 delay) */
    s->v_alpha_prev = v_alpha;
    s->v_beta_prev  = v_beta;

    /* Normalise for SVPWM */
    y->v_alpha = MPC_Clamp(v_alpha / MPC_SVPWM_GAIN, MPC_ONE_F);
    y->v_beta  = MPC_Clamp(v_beta  / MPC_SVPWM_GAIN, MPC_ONE_F);

    /*
     * Diagnostic log — written every MPC_DIAG_STEPS ISR ticks.
     *
     * ALIGNMENT NOTE: MPC_Controller_GetDiagnostics() reads these fields.
     * The Python MPCControllerBlock._log_step() writes the identical set
     * at the same rate (DIAG_STEPS = 20 → 1 kHz at 20 kHz ISR).
     * All values stored in their natural units:
     *   log_speed_ref / log_speed  : [RPM]   (×60/(2π) from rad/s)
     *   log_id / log_iq            : [A]
     *   log_vd / log_vq            : [V]     (physical, before SVPWM norm)
     *
     * C: MPC_DIAG_STEPS = 20 mirrors Python: DIAG_STEPS = 20
     */
    s->log_counter++;
    if (s->log_counter >= (unsigned int)MPC_DIAG_STEPS)
    {
        /* Convert rad/s → RPM for log fields (mirrors Python _log_step) */
        MatrixFloat rpm_scale = (MatrixFloat)60.0f / MPC_TWO_PI_F;

        s->log_speed_ref = u->omega_ref_mech * rpm_scale;
        s->log_speed     = omega_m            * rpm_scale;
        s->log_id        = id_meas;
        s->log_iq        = iq_meas;
        s->log_vd        = vd;
        s->log_vq        = vq;
        s->log_counter   = 0U;
    }
    else
    {
        /* MISRA Rule 15.7: final else required */
    }
}


/**
 * @brief   Reset all integrators and dynamic state
 * @param[in,out] s  Controller state (must not be NULL)
 *
 * @details Zeroes every integrator and observer state WITHOUT destroying the
 *          pre-computed SMO LPF coefficient alpha_lpf.  Calling
 *          MPC_Controller_Init(s, 0) would set alpha_lpf = 0 (because
 *          wc*0/(1+wc*0) = 0), making the SMO back-EMF filter permanently
 *          blind — this is the root cause of SMO divergence on hardware when
 *          the controller is reset mid-run.
 *
 *          PYTHON ALIGNMENT: MPCControllerBlock.reset() in mpc_controller_block.py
 *          explicitly calls self._smo.reset() which only zeroes the four current
 *          and EMF state variables — it preserves _alpha_lpf (set by set_dt() once
 *          at construction).  This C Reset() mirrors that exact behaviour.
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: Checks s for NULL
 *      - Rule 15.5: Single return point
 */
void MPC_Controller_Reset(MPC_Controller_T* const s)
{
    MatrixFloat saved_alpha_lpf;   /* Preserve pre-computed SMO LPF coefficient */

    if (s == NULL)
    {
        return;
    }

    /* Save the pre-computed LPF coefficient before zeroing */
    saved_alpha_lpf = s->smo.alpha_lpf;

    /* Zero all dynamic state (mirrors MPC_Controller_Init) */
    MPC_Controller_Init(s, MPC_ZERO_F);

    /* Restore alpha_lpf — Init(dt=0) sets it to 0 which blinds the SMO */
    s->smo.alpha_lpf = saved_alpha_lpf;
}


/**
 * @brief   Read the latest diagnostic snapshot
 * @param[in]  s              Controller state (must not be NULL)
 * @param[out] speed_ref_rpm  Speed reference [RPM]
 * @param[out] speed_rpm      Actual speed [RPM]
 * @param[out] id_meas        D-axis current [A]
 * @param[out] iq_meas        Q-axis current [A]
 * @param[out] vd             D-axis voltage [V]
 * @param[out] vq             Q-axis voltage [V]
 * @param[out] iq_limit       Soft-start iq ceiling; ramps 0→I_MAX over
 *                            MPC_SOFTSTART_T [s] then holds at I_MAX [A]
 *
 * @par MISRA Compliance:
 *      - Rule 17.5: All pointer parameters checked for NULL
 *      - Rule 15.7: All-or-nothing NULL check with final else
 */
void MPC_Controller_GetDiagnostics(const MPC_Controller_T* const s,
                                    MatrixFloat* const speed_ref_rpm,
                                    MatrixFloat* const speed_rpm,
                                    MatrixFloat* const id_meas,
                                    MatrixFloat* const iq_meas,
                                    MatrixFloat* const vd,
                                    MatrixFloat* const vq,
                                    MatrixFloat* const iq_limit)
{
    if ((s != NULL) &&
        (speed_ref_rpm != NULL) &&
        (speed_rpm != NULL) &&
        (id_meas != NULL) &&
        (iq_meas != NULL) &&
        (vd != NULL) &&
        (vq != NULL) &&
        (iq_limit != NULL))
    {
        *speed_ref_rpm = s->log_speed_ref;
        *speed_rpm     = s->log_speed;
        *id_meas       = s->log_id;
        *iq_meas       = s->log_iq;
        *vd            = s->log_vd;
        *vq            = s->log_vq;
        *iq_limit      = s->iq_limit;
    }
    else
    {
        /* MISRA Rule 15.7 - final else required */
    }
}