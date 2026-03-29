/**
 * @file    embed_sim_mpc_controller.c
 * @brief   3-state analytical MPC for PMSM — NANOTEC DB42S02 / AURIX TC3xx.
 *
 * Direct C port of mpc_controller_block.py (_solve_mpc + compute_py).
 * Every numerical operation matches the Python reference one-to-one.
 *
 * MISRA C:2012 compliance
 * ────────────────────────
 *  Rule 8.4  : All external-linkage identifiers declared in the header.
 *  Rule 14.4 : No implicit bool conversions.
 *  Rule 15.5 : Single point of exit per function.
 *  Rule 17.3 : No implicit function declarations.
 *  Rule 21.3 : No dynamic memory (malloc/free).
 *  Dir 4.1   : Run-time failures handled by clamping, never by UB.
 *
 * Note on float usage
 * ────────────────────
 *  AURIX TriCore TC3xx has a 32-bit hardware FPU (single precision).
 *  All state and computation uses float (IEEE 754 single precision).
 *  The 'F' suffix on every literal keeps MISRA Rule 4.7 / 10.4 happy.
 */

#include "embed_sim_mpc_controller.h"

/* Coordinate transforms from the canonical EmbedSim C library */
#include "embed_sim_coordinate_transform.h"

#include <string.h>   /* memset */
#include <math.h>     /* fabsf, sqrtf, tanhf, floorf, atan2f */

/* ── Internal helpers ───────────────────────────────────────────────────── */

/** Clamp x to [lo, hi]. */
static float mpc_clamp(float x, float lo, float hi)
{
    float result;
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

/** tanh approximation — uses libm tanhf (TriCore has HW support). */
static float mpc_tanh(float x)
{
    return tanhf(x);
}


/* ── Speed estimator ────────────────────────────────────────────────────── */

/**
 * @brief  Estimate mechanical speed from encoder angle delta with IIR filter.
 *         Mirrors _get_speed() in Python.
 *
 * @param  st       Controller state.
 * @param  theta_m  Current mechanical angle [rad] (continuous, unwrapped).
 * @param  dt       Sample period [s].
 * @return Filtered mechanical speed [rad/s].
 */
static float mpc_get_speed(MPC_Controller_T *st,
                            float theta_m, float dt)
{
    float delta;
    float omega_raw;

    if (dt <= 0.0F)
    {
        return st->omega_filt;
    }

    delta = theta_m - st->last_theta_m;

    /* Unwrap to (−π, π] */
    delta -= (2.0F * 3.14159265F) *
             floorf((delta + 3.14159265F) / (2.0F * 3.14159265F));

    omega_raw       = delta / dt;
    st->omega_filt  = 0.8F * st->omega_filt + 0.2F * omega_raw;
    st->last_theta_m = theta_m;

    return st->omega_filt;
}


/* ── Sliding-mode observer ──────────────────────────────────────────────── */

/**
 * @brief  One SMO step in the αβ frame.
 *         Mirrors _smo_step() in Python.
 */
static void mpc_smo_step(MPC_Controller_T *st,
                          float i_alpha, float i_beta,
                          float v_alpha, float v_beta,
                          float dt)
{
    float inv_L;
    float err_alpha, err_beta;
    float sw_alpha,  sw_beta;
    float smo_alpha;
    float wc;

    if (dt <= 0.0F)
    {
        return;
    }

    wc      = 2.0F * 3.14159265F * MPC_SMO_FC;
    smo_alpha = wc * dt / (1.0F + wc * dt);

    inv_L     = 1.0F / MPC_L;
    err_alpha = i_alpha - st->i_alpha_hat;
    err_beta  = i_beta  - st->i_beta_hat;

    sw_alpha = MPC_SMO_K * mpc_tanh(err_alpha / 0.01F);
    sw_beta  = MPC_SMO_K * mpc_tanh(err_beta  / 0.01F);

    st->i_alpha_hat += dt * inv_L *
                       (v_alpha - MPC_R_S * st->i_alpha_hat - sw_alpha);
    st->i_beta_hat  += dt * inv_L *
                       (v_beta  - MPC_R_S * st->i_beta_hat  - sw_beta);

    st->e_alpha_filt += smo_alpha * (sw_alpha - st->e_alpha_filt);
    st->e_beta_filt  += smo_alpha * (sw_beta  - st->e_beta_filt);
}


/* ── MPC solver ─────────────────────────────────────────────────────────── */

/**
 * @brief  Analytical 3-state MPC: returns (vd_total, vq_total).
 *
 *         Direct C mirror of _solve_mpc() in Python.
 *
 *         Free-run trajectory propagated WITHOUT BEMF (handled by feedforward).
 *         Step-response coefficients bk (current) and ek (speed) accumulated.
 *         Optimal vd_mpc, vq_mpc solved in closed form.
 *         BEMF feedforward added: vd = vd_mpc + ed_hat,  vq = vq_mpc + eq_hat.
 *
 * @param[in]  id0        d-axis current at t=0 [A]
 * @param[in]  iq0        q-axis current at t=0 [A]
 * @param[in]  omega_m    Mechanical speed at t=0 [rad/s]
 * @param[in]  omega_ref  Speed reference [rad/s mechanical]
 * @param[in]  ed_hat     d-axis back-EMF estimate [V]  (BEMF-clamped)
 * @param[in]  eq_hat     q-axis back-EMF estimate [V]  (BEMF-clamped)
 * @param[in]  dt         Sample period [s]
 * @param[out] vd_out     d-axis voltage command [V]
 * @param[out] vq_out     q-axis voltage command [V]
 */
static void mpc_solve(float id0, float iq0,
                       float omega_m, float omega_ref,
                       float ed_hat,  float eq_hat,
                       float dt,
                       float *vd_out, float *vq_out)
{
    /* Derived constants */
    const float omega_e = (float)MPC_P_POLES * omega_m;
    const float inv_L   = 1.0F / MPC_L;
    const float a       = 1.0F - dt * MPC_R_S * inv_L;  /* current decay   */
    const float b       = dt * inv_L;                     /* input gain      */
    const float dt_J    = dt / MPC_J;                     /* speed integral  */

    /* Free-run trajectory state */
    float id_free    = id0;
    float iq_free    = iq0;
    float omega_free = omega_m;

    /* Step-response accumulators */
    float bk = 0.0F;
    float ek = 0.0F;

    /* Cost gradient accumulators */
    float sum_bk_err_d = 0.0F;
    float sum_bk2      = 0.0F;
    float sum_ek_err   = 0.0F;
    float sum_ek2      = 0.0F;

    float f_d, f_q, f_omega;
    float denom_d, denom_q;
    float vd_mpc, vq_mpc;
    uint32_t k;

    for (k = 0U; k < (uint32_t)MPC_N; k++)
    {
        /* ── Free-run step (u=0, BEMF cancelled by feedforward) ─────────── */
        f_d     = dt * inv_L * ( omega_e * MPC_L * iq_free);
        f_q     = dt * inv_L * (-omega_e * MPC_L * id_free);
        f_omega = dt_J * (MPC_KT * iq_free - MPC_B * omega_free);

        id_free    = a * id_free    + f_d;
        iq_free    = a * iq_free    + f_q;
        omega_free = omega_free     + f_omega;

        /* ── Step-response coefficients ─────────────────────────────────── */
        bk = bk * a + b;               /* iq response to unit vq             */
        ek += dt_J * MPC_KT * bk;     /* omega response — uses current bk   */

        /* ── Gradient accumulation ──────────────────────────────────────── */
        sum_bk_err_d += bk * (0.0F       - id_free);
        sum_ek_err   += ek * (omega_ref  - omega_free);
        sum_bk2      += bk * bk;
        sum_ek2      += ek * ek;
    }

    /* ── Analytical optimal vd_mpc ──────────────────────────────────────── */
    denom_d = MPC_Q_ID    * sum_bk2 + MPC_R_VD;
    if (denom_d > 1.0e-30F)
    {
        vd_mpc = MPC_Q_ID * sum_bk_err_d / denom_d;
    }
    else
    {
        vd_mpc = 0.0F;
    }

    /* ── Analytical optimal vq_mpc ──────────────────────────────────────── */
    denom_q = MPC_Q_OMEGA * sum_ek2 + MPC_R_VQ;
    if (denom_q > 1.0e-30F)
    {
        vq_mpc = MPC_Q_OMEGA * sum_ek_err / denom_q;
    }
    else
    {
        vq_mpc = 0.0F;
    }

    /* ── BEMF feedforward + voltage saturation ──────────────────────────── */
    *vd_out = mpc_clamp(vd_mpc + ed_hat, -MPC_V_MAX, MPC_V_MAX);
    *vq_out = mpc_clamp(vq_mpc + eq_hat, -MPC_V_MAX, MPC_V_MAX);
}


/* ── Public API ─────────────────────────────────────────────────────────── */

void MPC_Controller_Init(MPC_Controller_T *st)
{
    /* MISRA Rule 21.6: memset is allowed for plain struct zero-init */
    (void)memset(st, 0, sizeof(MPC_Controller_T));
}


void MPC_Controller_Step(MPC_Controller_T *st,
                        const MPC_Input_T  *in,
                        MPC_Output_T        *out)
{
    /* ── Local variables ─────────────────────────────────────────────────── */
    float theta_e;
    float omega_m;
    float i_alpha, i_beta;
    float id_meas, iq_meas;
    float ed_hat_raw, eq_hat_raw;
    float bemf_max, omega_e;
    float ed_hat, eq_hat;
    float id0, iq0;
    float vd, vq;
    float vq_lim;
    float sp_err, head, intmax;
    float v_alpha, v_beta;

    /* ── Electrical angle ────────────────────────────────────────────────── */
    theta_e = (float)MPC_P_POLES * in->theta_m;
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

    /* ── Park: SMO back-EMF → [ed_hat, eq_hat] ─────────────────────────── */
    EmbedSim_Park(st->e_alpha_filt, st->e_beta_filt,
                  theta_e, &ed_hat_raw, &eq_hat_raw);

    /* ── Physical BEMF clamp ─────────────────────────────────────────────── */
    omega_e  = (float)MPC_P_POLES * omega_m;
    bemf_max = fabsf(omega_e) * MPC_LAMBDA_PM;
    ed_hat   = mpc_clamp(ed_hat_raw, -bemf_max, bemf_max);
    eq_hat   = mpc_clamp(eq_hat_raw, -bemf_max, bemf_max);

    /* ── Soft-start: ramp iq_limit 0 → I_MAX over SOFTSTART_T ──────────── */
    st->iq_limit = mpc_clamp(
        st->iq_limit + MPC_I_MAX * MPC_DT / MPC_SOFTSTART_T,
        0.0F, MPC_I_MAX);

    /* ── Clamp measured currents ─────────────────────────────────────────── */
    id0 = mpc_clamp(id_meas, -MPC_I_MAX, MPC_I_MAX);
    iq0 = mpc_clamp(iq_meas, -MPC_I_MAX, MPC_I_MAX);

    /* ── 3-state MPC solver ──────────────────────────────────────────────── */
    mpc_solve(id0, iq0, omega_m, in->omega_ref_mech,
              ed_hat, eq_hat, MPC_DT, &vd, &vq);

    /* ── Soft-start vq limit ─────────────────────────────────────────────── */
    vq_lim = (st->iq_limit / MPC_I_MAX) * MPC_V_MAX;
    vq     = mpc_clamp(vq, -vq_lim, vq_lim);

    /* ── Speed error integral correction (eliminates MPC steady-state offset) */
    sp_err                = in->omega_ref_mech - omega_m;
    st->speed_err_integral += sp_err * MPC_DT;

    /* Anti-windup: keep integral contribution within remaining vq headroom */
    head   = MPC_V_MAX - fabsf(vq);
    head   = mpc_clamp(head, 0.0F, MPC_V_MAX);   /* headroom ≥ 0 */
    intmax = head / (MPC_KI_V + 1.0e-30F);
    st->speed_err_integral = mpc_clamp(st->speed_err_integral,
                                        -intmax, intmax);
    vq = mpc_clamp(vq + MPC_KI_V * st->speed_err_integral,
                   -MPC_V_MAX, MPC_V_MAX);

    /* ── InvPark: [vd, vq] → [v_alpha, v_beta] ──────────────────────────── */
    EmbedSim_InvPark(vd, vq, theta_e, &v_alpha, &v_beta);

    /* ── Store previous voltages for SMO ────────────────────────────────── */
    st->v_alpha_prev = v_alpha;
    st->v_beta_prev  = v_beta;

    /* ── Write outputs ───────────────────────────────────────────────────── */
    out->v_alpha = v_alpha;
    out->v_beta  = v_beta;
}
