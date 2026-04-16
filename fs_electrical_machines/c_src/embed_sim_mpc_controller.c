/**
 **********************************************************************************************************************
 * \file      embed_sim_mpc_controller.c
 * \brief     3-state analytical MPC for PMSM — NANOTEC DB42S02 / AURIX TC3xx.
 *
 * Direct C port of mpc_controller_block.py (_solve_mpc + compute_py).
 * Every numerical operation matches the Python reference one-to-one.
 *
 * \version   2.1.0
 * \copyright Copyright (C) EmbedSim 2025
 **********************************************************************************************************************/

/**********************************************************************************************************************/
/*-----------------------------------------------------Includes-------------------------------------------------------*/
/**********************************************************************************************************************/
#include "embed_sim_mpc_controller.h"
#include "embed_sim_coordinate_transform.h"   /* EmbedSim_Clarke / Park / InvPark */

#include <string.h>   /* memset  */
#include <math.h>     /* fabsf, tanhf, floorf */


/**********************************************************************************************************************/
/*--------------------------------------------Private Function Prototypes---------------------------------------------*/
/**********************************************************************************************************************/

/**
 * \brief  Clamp x to [lo, hi].
 * \return Clamped value as MatrixFloat.
 */
static MatrixFloat mpc_clamp(MatrixFloat x, MatrixFloat lo, MatrixFloat hi);

/**
 * \brief  tanh(x) using libm tanhf — TriCore has HW support.
 * \return MatrixFloat result.
 */
static MatrixFloat mpc_tanh(MatrixFloat x);

/**
 * \brief  Estimate mechanical speed from encoder angle delta with IIR filter.
 *
 * Mirrors _get_speed() in mpc_controller_block.py.
 * IIR: omega_filt = 0.8·prev + 0.2·raw   (τ ≈ 4 steps at 50 µs dt)
 * 2π unwrap applied before finite-difference.
 *
 * \param[in,out] st       Controller state.
 * \param[in]     theta_m  Mechanical angle, current step [rad].
 * \param[in]     dt       Sample period [s].
 * \return                 Filtered mechanical speed [rad/s].
 */
static MatrixFloat mpc_get_speed(MPC_Controller_T *st,
                                 MatrixFloat        theta_m,
                                 MatrixFloat        dt);

/**
 * \brief  One SMO step in the αβ frame.
 *
 * Mirrors _smo_step() in mpc_controller_block.py.
 *
 * \param[in,out] st       Controller state.
 * \param[in]     i_alpha  Measured α-axis current [A].
 * \param[in]     i_beta   Measured β-axis current [A].
 * \param[in]     v_alpha  Applied α-axis voltage, previous step [V].
 * \param[in]     v_beta   Applied β-axis voltage, previous step [V].
 * \param[in]     dt       Sample period [s].
 */
static void mpc_smo_step(MPC_Controller_T *st,
                         MatrixFloat        i_alpha,
                         MatrixFloat        i_beta,
                         MatrixFloat        v_alpha,
                         MatrixFloat        v_beta,
                         MatrixFloat        dt);

/**
 * \brief  Analytical 3-state MPC solver — returns (vd_total, vq_total).
 *
 * Mirrors _solve_mpc() in mpc_controller_block.py.
 * Free-run trajectory propagated without BEMF (handled by feedforward).
 * Step-response coefficients bk (current) and ek (speed) accumulated.
 * Optimal vd_mpc, vq_mpc solved in closed form.
 * BEMF feedforward added: vd = vd_mpc + ed_hat, vq = vq_mpc + eq_hat.
 *
 * \param[in]  id0        d-axis current at t=0 [A]
 * \param[in]  iq0        q-axis current at t=0 [A]
 * \param[in]  omega_m    Mechanical speed at t=0 [rad/s]
 * \param[in]  omega_ref  Speed reference [rad/s mechanical]
 * \param[in]  ed_hat     d-axis back-EMF estimate [V]  (BEMF-clamped)
 * \param[in]  eq_hat     q-axis back-EMF estimate [V]  (BEMF-clamped)
 * \param[in]  dt         Sample period [s]
 * \param[in]  gains      Runtime gain set (weights)
 * \param[out] vd_out     d-axis voltage command [V]
 * \param[out] vq_out     q-axis voltage command [V]
 */
static void mpc_solve(MatrixFloat  id0,
                      MatrixFloat  iq0,
                      MatrixFloat  omega_m,
                      MatrixFloat  omega_ref,
                      MatrixFloat  ed_hat,
                      MatrixFloat  eq_hat,
                      MatrixFloat  dt,
                      const MPC_GainSet_T *gains,
                      MatrixFloat *vd_out,
                      MatrixFloat *vq_out);


/**********************************************************************************************************************/
/*------------------------------------------------Private Functions---------------------------------------------------*/
/**********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * mpc_clamp
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat mpc_clamp(MatrixFloat x, MatrixFloat lo, MatrixFloat hi)
{
    MatrixFloat result;

    if (x < lo)
    {
        result = lo;
    }
    else if (x > hi)
    {
        result = hi;
    }
    else
    {
        result = x;
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * mpc_tanh
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat mpc_tanh(MatrixFloat x)
{
    return (MatrixFloat)tanhf((float)x);
}


/*--------------------------------------------------------------------------------------------------------------------
 * mpc_get_speed
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat mpc_get_speed(MPC_Controller_T *st,
                                 MatrixFloat        theta_m,
                                 MatrixFloat        dt)
{
    MatrixFloat delta;
    MatrixFloat omega_raw;

    if (dt <= MPC_ZERO_F)
    {
        return st->omega_filt;
    }
    else
    {
        /* MISRA 15.7: else required */
    }

    delta = theta_m - st->last_theta_m;

    /* Unwrap to (−π, +π] */
    delta -= (MPC_TWO_F * MPC_PI_F) *
             (MatrixFloat)floorf((float)((delta + MPC_PI_F) /
                                        (MPC_TWO_F * MPC_PI_F)));

    omega_raw        = delta / dt;
    st->omega_filt   = (MatrixFloat)0.8F * st->omega_filt
                     + (MatrixFloat)0.2F * omega_raw;
    st->last_theta_m = theta_m;

    return st->omega_filt;
}


/*--------------------------------------------------------------------------------------------------------------------
 * mpc_smo_step
 *------------------------------------------------------------------------------------------------------------------*/
static void mpc_smo_step(MPC_Controller_T *st,
                         MatrixFloat        i_alpha,
                         MatrixFloat        i_beta,
                         MatrixFloat        v_alpha,
                         MatrixFloat        v_beta,
                         MatrixFloat        dt)
{
    MatrixFloat inv_L;
    MatrixFloat err_alpha;
    MatrixFloat err_beta;
    MatrixFloat sw_alpha;
    MatrixFloat sw_beta;
    MatrixFloat smo_alpha;
    MatrixFloat wc;

    if (dt <= MPC_ZERO_F)
    {
        return;   /* MISRA 15.5: exceptional early return on invalid dt */
    }
    else
    {
        /* MISRA 15.7: else required */
    }

    wc        = MPC_TWO_F * MPC_PI_F * MPC_SMO_FC;
    smo_alpha = (wc * dt) / (MPC_ONE_F + wc * dt);

    inv_L     = MPC_ONE_F / MPC_L;
    err_alpha = i_alpha - st->i_alpha_hat;
    err_beta  = i_beta  - st->i_beta_hat;

    sw_alpha = MPC_SMO_K * mpc_tanh(err_alpha / (MatrixFloat)0.01F);
    sw_beta  = MPC_SMO_K * mpc_tanh(err_beta  / (MatrixFloat)0.01F);

    st->i_alpha_hat += dt * inv_L *
                       (v_alpha - MPC_R_S * st->i_alpha_hat - sw_alpha);
    st->i_beta_hat  += dt * inv_L *
                       (v_beta  - MPC_R_S * st->i_beta_hat  - sw_beta);

    st->e_alpha_filt += smo_alpha * (sw_alpha - st->e_alpha_filt);
    st->e_beta_filt  += smo_alpha * (sw_beta  - st->e_beta_filt);
}


/*--------------------------------------------------------------------------------------------------------------------
 * mpc_solve
 *------------------------------------------------------------------------------------------------------------------*/
static void mpc_solve(MatrixFloat  id0,
                      MatrixFloat  iq0,
                      MatrixFloat  omega_m,
                      MatrixFloat  omega_ref,
                      MatrixFloat  ed_hat,
                      MatrixFloat  eq_hat,
                      MatrixFloat  dt,
                      const MPC_GainSet_T *gains,
                      MatrixFloat *vd_out,
                      MatrixFloat *vq_out)
{
    /* Derived constants */
    const MatrixFloat omega_e = (MatrixFloat)MPC_P_POLES * omega_m;
    const MatrixFloat inv_L   = MPC_ONE_F / MPC_L;
    const MatrixFloat a       = MPC_ONE_F - dt * MPC_R_S * inv_L;   /* current decay  */
    const MatrixFloat b       = dt * inv_L;                           /* input gain     */
    const MatrixFloat dt_J    = dt / MPC_J;                           /* speed integral */

    /* Free-run trajectory state */
    MatrixFloat id_free    = id0;
    MatrixFloat iq_free    = iq0;
    MatrixFloat omega_free = omega_m;

    /* Step-response accumulators */
    MatrixFloat bk = MPC_ZERO_F;
    MatrixFloat ek = MPC_ZERO_F;

    /* Cost gradient accumulators */
    MatrixFloat sum_bk_err_d = MPC_ZERO_F;
    MatrixFloat sum_bk_err_q = MPC_ZERO_F;  /* NEW: iq penalty term */
    MatrixFloat sum_ek_err   = MPC_ZERO_F;
    MatrixFloat sum_bk2      = MPC_ZERO_F;
    MatrixFloat sum_ek2      = MPC_ZERO_F;

    MatrixFloat f_d;
    MatrixFloat f_q;
    MatrixFloat f_omega;
    MatrixFloat denom_d;
    MatrixFloat denom_q;
    MatrixFloat vd_mpc;
    MatrixFloat vq_mpc;
    uint32_T    k;

    /* Clamp initial currents for prediction */
    id_free = mpc_clamp(id_free, -MPC_I_MAX, MPC_I_MAX);
    iq_free = mpc_clamp(iq_free, -MPC_I_MAX, MPC_I_MAX);

    for (k = 0U; k < MPC_N; k++)
    {
        /* ── Free-run step (u=0, BEMF cancelled by feedforward) ──────────── */
        f_d     = dt * inv_L * ( omega_e * MPC_L * iq_free);
        f_q     = dt * inv_L * (-omega_e * MPC_L * id_free);
        f_omega = dt_J * (MPC_KT * iq_free - MPC_B * omega_free);

        id_free    = a * id_free    + f_d;
        iq_free    = a * iq_free    + f_q;
        omega_free = omega_free     + f_omega;

        /* Apply current limits during prediction */
        id_free = mpc_clamp(id_free, -MPC_I_MAX, MPC_I_MAX);
        iq_free = mpc_clamp(iq_free, -MPC_I_MAX, MPC_I_MAX);

        /* ── Step-response coefficients ──────────────────────────────────── */
        bk = bk * a + b;              /* iq response to unit vq                */
        ek += dt_J * MPC_KT * bk;    /* omega response — uses current bk      */

        /* ── Gradient accumulation ───────────────────────────────────────── */
        /* d-axis: target id_ref = 0 */
        sum_bk_err_d += bk * (MPC_ZERO_F - id_free);
        /* iq penalty term: target iq_ref = 0 (we penalise iq directly) */
        sum_bk_err_q += bk * (MPC_ZERO_F - iq_free);
        /* speed term: target omega_ref */
        sum_ek_err   += ek * (omega_ref - omega_free);
        sum_bk2      += bk * bk;
        sum_ek2      += ek * ek;
    }

    /* ── Analytical optimal vd_mpc ──────────────────────────────────────── */
    denom_d = gains->q_id * sum_bk2 + gains->r_vd;

    if (denom_d > MPC_DENOM_MIN)
    {
        vd_mpc = gains->q_id * sum_bk_err_d / denom_d;
    }
    else
    {
        vd_mpc = MPC_ZERO_F;
    }

    /* ── Analytical optimal vq_mpc ──────────────────────────────────────── */
    /* NEW: Include Q_iq in denominator and numerator (matches Python) */
    denom_q = gains->q_omega * sum_ek2 + gains->q_iq * sum_bk2 + gains->r_vq;

    if (denom_q > MPC_DENOM_MIN)
    {
        vq_mpc = (gains->q_omega * sum_ek_err + gains->q_iq * sum_bk_err_q) / denom_q;
    }
    else
    {
        vq_mpc = MPC_ZERO_F;
    }

    /* ── BEMF feedforward + voltage saturation ───────────────────────────── */
    *vd_out = mpc_clamp(vd_mpc + ed_hat, -MPC_V_MAX, MPC_V_MAX);
    *vq_out = mpc_clamp(vq_mpc + eq_hat, -MPC_V_MAX, MPC_V_MAX);
}


/**********************************************************************************************************************/
/*-------------------------------------------------Public Functions---------------------------------------------------*/
/**********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * MPC_Controller_Init
 *------------------------------------------------------------------------------------------------------------------*/
void MPC_Controller_Init(MPC_Controller_T *st)
{
    if (st != NULL)
    {
        /* MISRA Rule 21.6: memset is permitted for plain-old-data struct zero-init */
        (void)memset(st, 0, sizeof(MPC_Controller_T));
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * MPC_GainSet_GetDefault
 *------------------------------------------------------------------------------------------------------------------*/
void MPC_GainSet_GetDefault(MPC_GainSet_T *gains)
{
    if (gains != NULL)
    {
        gains->q_id    = MPC_Q_ID;
        gains->q_iq    = MPC_Q_IQ;
        gains->q_omega = MPC_Q_OMEGA;
        gains->r_vd    = MPC_R_VD;
        gains->r_vq    = MPC_R_VQ;
        gains->ki_v    = MPC_KI_V;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * MPC_Controller_StepWithGains
 *------------------------------------------------------------------------------------------------------------------*/
void MPC_Controller_StepWithGains(MPC_Controller_T       *st,
                                  const MPC_Input_T      *in,
                                  MPC_Output_T           *out,
                                  const MPC_GainSet_T    *gains)
{
    MatrixFloat theta_e;
    MatrixFloat omega_m;
    MatrixFloat i_alpha;
    MatrixFloat i_beta;
    MatrixFloat id_meas;
    MatrixFloat iq_meas;
    MatrixFloat ed_hat_raw;
    MatrixFloat eq_hat_raw;
    MatrixFloat bemf_max;
    MatrixFloat omega_e;
    MatrixFloat ed_hat;
    MatrixFloat eq_hat;
    MatrixFloat id0;
    MatrixFloat iq0;
    MatrixFloat vd;
    MatrixFloat vq;
    MatrixFloat vq_lim;
    MatrixFloat sp_err;
    MatrixFloat head;
    MatrixFloat intmax;
    MatrixFloat v_alpha;
    MatrixFloat v_beta;
    MPC_GainSet_T default_gains;

    /* Guard against NULL pointers — MISRA Dir 4.1 */
    if ((st == NULL) || (in == NULL) || (out == NULL))
    {
        return;   /* MISRA 15.5: exceptional early return */
    }
    else
    {
        /* MISRA 15.7: else required */
    }

    /* Use provided gains or fall back to defaults */
    if (gains == NULL)
    {
        MPC_GainSet_GetDefault(&default_gains);
        gains = &default_gains;
    }
    else
    {
        /* gains already valid */
    }

    /* ── Electrical angle ─────────────────────────────────────────────────── */
    theta_e = (MatrixFloat)MPC_P_POLES * in->theta_m;
    omega_m = mpc_get_speed(st, in->theta_m, MPC_DT);

    /* ── Clarke: [ia, ib, ic] → [i_alpha, i_beta] ───────────────────────── */
    EmbedSim_Clarke(in->ia, in->ib, in->ic, &i_alpha, &i_beta);

    /* ── SMO back-EMF estimation ─────────────────────────────────────────── */
    mpc_smo_step(st,
                 i_alpha, i_beta,
                 st->v_alpha_prev, st->v_beta_prev,
                 MPC_DT);

    /* ── Park: [i_alpha, i_beta] → [id, iq] ─────────────────────────────── */
    EmbedSim_Park(i_alpha, i_beta, theta_e, &id_meas, &iq_meas);

    /* ── Park: SMO back-EMF αβ → [ed_hat_raw, eq_hat_raw] ──────────────── */
    EmbedSim_Park(st->e_alpha_filt, st->e_beta_filt,
                  theta_e, &ed_hat_raw, &eq_hat_raw);

    /* ── Physical BEMF clamp ─────────────────────────────────────────────── */
    omega_e  = (MatrixFloat)MPC_P_POLES * omega_m;
    bemf_max = (MatrixFloat)fabsf((float)omega_e) * MPC_LAMBDA_PM;
    ed_hat   = mpc_clamp(ed_hat_raw, -bemf_max, bemf_max);
    eq_hat   = mpc_clamp(eq_hat_raw, -bemf_max, bemf_max);

    /* ── Soft-start: ramp iq_limit 0 → I_MAX over MPC_SOFTSTART_T ───────── */
    st->iq_limit = mpc_clamp(
        st->iq_limit + MPC_I_MAX * MPC_DT / MPC_SOFTSTART_T,
        MPC_ZERO_F,
        MPC_I_MAX);

    /* ── Clamp measured currents ─────────────────────────────────────────── */
    id0 = mpc_clamp(id_meas, -MPC_I_MAX, MPC_I_MAX);
    iq0 = mpc_clamp(iq_meas, -MPC_I_MAX, MPC_I_MAX);

    /* ── 3-state MPC solver (with runtime gains) ─────────────────────────── */
    mpc_solve(id0, iq0, omega_m, in->omega_ref_mech,
              ed_hat, eq_hat, MPC_DT, gains, &vd, &vq);

    /* ── Soft-start vq limit ─────────────────────────────────────────────── */
    vq_lim = (st->iq_limit / MPC_I_MAX) * MPC_V_MAX;
    vq     = mpc_clamp(vq, -vq_lim, vq_lim);

    /* ── Speed error integral correction (eliminates MPC steady-state offset) */
    sp_err                  = in->omega_ref_mech - omega_m;
    st->speed_err_integral += sp_err * MPC_DT;

    /* Anti-windup: keep integral contribution within remaining vq headroom */
    head   = MPC_V_MAX - (MatrixFloat)fabsf((float)vq);
    head   = mpc_clamp(head, MPC_ZERO_F, MPC_V_MAX);
    intmax = head / (gains->ki_v + MPC_DENOM_MIN);

    st->speed_err_integral = mpc_clamp(st->speed_err_integral,
                                       -intmax, intmax);

    vq = mpc_clamp(vq + gains->ki_v * st->speed_err_integral,
                   -MPC_V_MAX, MPC_V_MAX);

    /* ── InvPark: [vd, vq] → [v_alpha, v_beta] ──────────────────────────── */
    EmbedSim_InvPark(vd, vq, theta_e, &v_alpha, &v_beta);

    /* ── Store previous voltages for SMO (physical [V], not normalised) ──── */
    st->v_alpha_prev = v_alpha;
    st->v_beta_prev  = v_beta;

    /* ── Write outputs ───────────────────────────────────────────────────── */
    out->v_alpha = v_alpha;
    out->v_beta  = v_beta;
}


/*--------------------------------------------------------------------------------------------------------------------
 * MPC_Controller_Step (legacy with compile-time gains)
 *------------------------------------------------------------------------------------------------------------------*/
void MPC_Controller_Step(MPC_Controller_T       *st,
                         const MPC_Input_T      *in,
                         MPC_Output_T           *out)
{
    /* Call the gains version with NULL (uses defaults) */
    MPC_Controller_StepWithGains(st, in, out, NULL);
}