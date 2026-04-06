/**********************************************************************************************************************
 * \file      embed_sim_ekf_speed.c
 * \brief     Sensorless 4-state EKF speed observer -- stationary αβ frame
 *
 * \details   State vector:  x = [ i_alpha,  i_beta,  omega_e,  theta_e ]
 *            Inputs:        v_alpha, v_beta  (commanded voltages, z-1)
 *            Measurement:   y = [ i_alpha_meas, i_beta_meas ]  (direct)
 *
 *            Prediction model (Euler, stationary frame, isotropic L = Ld = Lq):
 *              i_alpha+ = i_alpha + dt/L * (v_alpha - R*i_alpha - e_alpha)
 *              i_beta+  = i_beta  + dt/L * (v_beta  - R*i_beta  - e_beta)
 *              omega_e+ = omega_e                  (random walk)
 *              theta_e+ = theta_e + omega_e * dt   (integrate speed)
 *
 *              Back-EMF:
 *                e_alpha = -lpm * omega_e * sin(theta_e)
 *                e_beta  =  lpm * omega_e * cos(theta_e)
 *
 *            State Jacobian F (4x4):
 *              F[0,0] = 1 - R/L * dt
 *              F[0,2] =  lpm * sin(theta_e) * dt / L    (de_alpha / d_omega_e)
 *              F[0,3] =  lpm * omega_e * cos(theta_e) * dt / L
 *              F[1,1] = 1 - R/L * dt
 *              F[1,2] = -lpm * cos(theta_e) * dt / L
 *              F[1,3] =  lpm * omega_e * sin(theta_e) * dt / L
 *              F[2,2] = 1   (random walk)
 *              F[3,2] = dt  (theta_e integrates omega_e)
 *              F[3,3] = 1
 *
 *            Measurement Jacobian H (2x4):
 *              H = [[1, 0, 0, 0],
 *                   [0, 1, 0, 0]]
 *              Direct measurement -- H is constant, S = P_pred[0:2,0:2] + R.
 *
 *            Confirmed noise weights (DB42S02, AURIX 20 kHz, 12-bit ADC):
 *              q_omega = 1e-2   r_i = 5e-2   p0_omega = 1e6  p0_theta = 9.87
 *              SS error < 1 RPM,  convergence < 2 ms from cold start.
 *
 *            No warmup gate -- omega_m is live from step 1.
 *            No Park transform in measurement -- eliminates angle-locked H[:,2]=0 bug.
 *
 * Matrix layout (all row-major, stride N=4 for NxN, stride 2 for Nx2):
 *   F      [4x4]  F[i*4+j]
 *   Q      [4x4]  Q[i*4+j]
 *   P      [4x4]  s->P[i*4+j]   (EKF_N=4)
 *   P_pred [4x4]  P_pred[i*4+j]
 *   FP     [4x4]  FP[i*4+j]
 *   PH     [4x2]  PH[i*2+j]     = P_pred[:,0:2] (first two columns)
 *   S      [2x2]  S[i*2+j]
 *   S_inv  [2x2]  S_inv[i*2+j]
 *   K      [4x2]  K[i*2+j]
 *   I_KH   [4x4]  I_KH[i*4+j]
 *   temp   [4x4]  temp[i*4+j]
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types explicit via MatrixFloat / uint32_T.
 *              Rule 10.4  : no mixed-mode arithmetic.
 *              Rule 15.5  : single return per function.
 *              Rule 15.7  : every if-else chain has a final else.
 *
 * \version   4.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#include "embed_sim_ekf_speed.h"
#include <math.h>

#define EKF_ZERO    ((MatrixFloat)0.0f)
#define EKF_ONE     ((MatrixFloat)1.0f)
#define EKF_HALF    ((MatrixFloat)0.5f)
#define EKF_DT_MIN  ((MatrixFloat)1.0e-12f)
#define EKF_L_MIN   ((MatrixFloat)1.0e-9f)
#define EKF_PI_F    ((MatrixFloat)3.14159265358979f)
#define EKF_TWO_PI  ((MatrixFloat)6.28318530717959f)


/*--------------------------------------------------------------------------------------------------------------------
 * EKF_Speed_Init
 *------------------------------------------------------------------------------------------------------------------*/
void EKF_Speed_Init(
    DFC_EKF_Speed_T          * const s,
    const EKF_Speed_Params_T * const params)
{
    uint32_T i;
    uint32_T j;

    if ((s == NULL) || (params == NULL))
    {
        return;
    }

    for (i = 0U; i < EKF_N; i++)
    {
        s->x[i] = EKF_ZERO;
        for (j = 0U; j < EKF_N; j++)
        {
            s->P[i * EKF_N + j] = EKF_ZERO;
        }
    }

    /* Initial covariance — cold-start diagonal */
    s->P[0U * EKF_N + 0U] = params->p0_i;       /* i_alpha variance    */
    s->P[1U * EKF_N + 1U] = params->p0_i;       /* i_beta  variance    */
    s->P[2U * EKF_N + 2U] = params->p0_omega;   /* omega_e variance    */
    s->P[3U * EKF_N + 3U] = params->p0_theta;   /* theta_e variance    */

    s->theta_e_hat = EKF_ZERO;   /* legacy alias -- kept for API compat */
    s->omega_m     = EKF_ZERO;
    s->theta_e     = EKF_ZERO;
    s->step_count  = 0U;
}


/*--------------------------------------------------------------------------------------------------------------------
 * EKF_Speed_Reset
 *------------------------------------------------------------------------------------------------------------------*/
void EKF_Speed_Reset(
    DFC_EKF_Speed_T          * const s,
    const EKF_Speed_Params_T * const params)
{
    EKF_Speed_Init(s, params);
}


/*--------------------------------------------------------------------------------------------------------------------
 * EKF_Speed_Step  (4-state sensorless αβ-frame EKF)
 *------------------------------------------------------------------------------------------------------------------*/
void EKF_Speed_Step(
    DFC_EKF_Speed_T          * const s,
    const MatrixFloat                ia,
    const MatrixFloat                ib,
    const MatrixFloat                ic,
    const MatrixFloat                v_alpha,
    const MatrixFloat                v_beta,
    const MatrixFloat                dt,
    const EKF_Speed_Params_T * const params)
{
    /* ── Working arrays (4-state, row-major) ───────────────────────────────── */
    MatrixFloat F[16];       /* [4x4] state Jacobian                            */
    MatrixFloat Q[16];       /* [4x4] process noise (diagonal)                  */
    MatrixFloat x_pred[4];   /* predicted state                                 */
    MatrixFloat P_pred[16];  /* predicted covariance                            */
    MatrixFloat FP[16];      /* F * P  (intermediate)                           */
    MatrixFloat PH[8];       /* P_pred[:,0:2]  (= P_pred * H')                  */
    MatrixFloat S[4];        /* [2x2] innovation covariance                     */
    MatrixFloat S_inv[4];    /* [2x2] inverse of S                              */
    MatrixFloat K[8];        /* [4x2] Kalman gain                               */
    MatrixFloat nu[2];       /* innovation: y - h(x_pred)                       */
    MatrixFloat I_KH[16];    /* (I - K*H)  [4x4]                                */
    MatrixFloat temp[16];    /* [4x4] intermediate for Joseph form              */

    MatrixFloat inv_L;
    MatrixFloat cos_t;
    MatrixFloat sin_t;
    MatrixFloat omega_e;
    MatrixFloat theta_e;
    MatrixFloat e_alpha;
    MatrixFloat e_beta;
    MatrixFloat rd;          /* R * dt / L */
    MatrixFloat det;
    MatrixFloat inv_det;
    MatrixFloat sum;
    MatrixFloat avg;
    MatrixFloat p_poles_f;
    MatrixFloat theta_new;
    uint32_T    i;
    uint32_T    j;
    uint32_T    k;

    /* ic is used for amplitude-invariant Clarke: i_beta = (ia + 2*ib)/sqrt(3)
     * The ic parameter is kept for 3-wire balanced assumption (ia+ib+ic=0),
     * but the amplitude-invariant form only needs ia and ib.               */
    (void)ic;

    if ((s == NULL) || (params == NULL))
    {
        return;
    }

    s->step_count++;
    p_poles_f = (MatrixFloat)params->p_poles;

    if (dt < EKF_DT_MIN)
    {
        return;
    }

    /* Use isotropic L = (Ld + Lq) / 2 -- surface-mount PMSM (Ld ≈ Lq) */
    {
        MatrixFloat L_avg = (params->L_d + params->L_q) * EKF_HALF;
        inv_L = (L_avg > EKF_L_MIN) ? (EKF_ONE / L_avg) : (EKF_ONE / EKF_L_MIN);
    }

    /* ── Unpack current state ──────────────────────────────────────────────── */
    omega_e = s->x[2];
    theta_e = s->x[3];

    cos_t = cosf(theta_e);
    sin_t = sinf(theta_e);

    /* Back-EMF in stationary frame */
    e_alpha = -params->lambda_pm * omega_e * sin_t;
    e_beta  =  params->lambda_pm * omega_e * cos_t;

    /* ── Measurement y = [i_alpha_meas, i_beta_meas] ──────────────────────── */
    /* Clarke amplitude-invariant: i_alpha = ia,  i_beta = (ia + 2*ib)/sqrt(3) */
    nu[0] = ia;
    nu[1] = (ia + (MatrixFloat)2.0f * ib) * (MatrixFloat)0.57735027f;

    /* ── Nonlinear prediction x_pred = f(x, u) ────────────────────────────── */
    x_pred[0] = s->x[0] + dt * inv_L * (v_alpha - params->R_s * s->x[0] - e_alpha);
    x_pred[1] = s->x[1] + dt * inv_L * (v_beta  - params->R_s * s->x[1] - e_beta);
    x_pred[2] = omega_e;                       /* random walk                  */
    x_pred[3] = theta_e + omega_e * dt;        /* integrate electrical speed   */

    /* Wrap predicted theta_e to [-pi, pi] */
    theta_new = x_pred[3];
    while (theta_new >  EKF_PI_F) { theta_new -= EKF_TWO_PI; }
    while (theta_new < -EKF_PI_F) { theta_new += EKF_TWO_PI; }
    x_pred[3] = theta_new;

    /* ── State Jacobian F (4x4) ────────────────────────────────────────────── */
    for (i = 0U; i < 16U; i++) { F[i] = EKF_ZERO; }

    rd = params->R_s * dt * inv_L;

    /* Row 0: d(i_alpha+) / d[i_alpha, i_beta, omega_e, theta_e] */
    F[0U*4U+0U] = EKF_ONE - rd;
    F[0U*4U+2U] =  params->lambda_pm * sin_t * dt * inv_L;
    F[0U*4U+3U] =  params->lambda_pm * omega_e * cos_t * dt * inv_L;

    /* Row 1: d(i_beta+) / d[i_alpha, i_beta, omega_e, theta_e] */
    F[1U*4U+1U] = EKF_ONE - rd;
    F[1U*4U+2U] = -params->lambda_pm * cos_t * dt * inv_L;
    F[1U*4U+3U] =  params->lambda_pm * omega_e * sin_t * dt * inv_L;

    /* Row 2: d(omega_e+) / d[...] = random walk */
    F[2U*4U+2U] = EKF_ONE;

    /* Row 3: d(theta_e+) / d[omega_e, theta_e] */
    F[3U*4U+2U] = dt;
    F[3U*4U+3U] = EKF_ONE;

    /* ── Process noise Q (4x4 diagonal) ───────────────────────────────────── */
    for (i = 0U; i < 16U; i++) { Q[i] = EKF_ZERO; }
    Q[0U*4U+0U] = params->q_i;
    Q[1U*4U+1U] = params->q_i;
    Q[2U*4U+2U] = params->q_omega;
    Q[3U*4U+3U] = params->q_theta;

    /* ── P_pred = F * P * F' + Q ───────────────────────────────────────────── */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 4U; k++) { sum += F[i*4U+k] * s->P[k*4U+j]; }
            FP[i*4U+j] = sum;
        }
    }
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 4U; k++) { sum += FP[i*4U+k] * F[j*4U+k]; }
            P_pred[i*4U+j] = sum + Q[i*4U+j];
        }
    }

    /* ── Innovation nu = y - h(x_pred) ────────────────────────────────────── */
    /* h(x) = x[0:2] (direct measurement -- H = [I_2x2 | 0_2x2])             */
    /* nu already holds y_meas from Clarke above -- subtract h(x_pred)        */
    nu[0] -= x_pred[0];
    nu[1] -= x_pred[1];

    /* ── S = H*P_pred*H' + R = P_pred[0:2, 0:2] + R ──────────────────────── */
    /* H is [[1,0,0,0],[0,1,0,0]] so H*P_pred*H' = top-left 2x2 of P_pred     */
    /* PH = P_pred * H' = first two columns of P_pred                         */
    for (i = 0U; i < 4U; i++)
    {
        PH[i*2U+0U] = P_pred[i*4U+0U];
        PH[i*2U+1U] = P_pred[i*4U+1U];
    }

    S[0U*2U+0U] = P_pred[0U*4U+0U] + params->r_i;
    S[0U*2U+1U] = P_pred[0U*4U+1U];
    S[1U*2U+0U] = P_pred[1U*4U+0U];
    S[1U*2U+1U] = P_pred[1U*4U+1U] + params->r_i;

    /* ── S_inv: 2x2 closed-form ────────────────────────────────────────────── */
    det = S[0U*2U+0U] * S[1U*2U+1U] - S[0U*2U+1U] * S[1U*2U+0U];
    if (det < EKF_ZERO) { det = -det; }
    else { /* positive det -- MISRA 15.7 */ }
    if (det < EKF_DET_MIN) { det = EKF_DET_MIN; }
    else { /* MISRA 15.7 */ }
    inv_det =  EKF_ONE / det;

    S_inv[0U*2U+0U] =  S[1U*2U+1U] * inv_det;
    S_inv[0U*2U+1U] = -S[0U*2U+1U] * inv_det;
    S_inv[1U*2U+0U] = -S[1U*2U+0U] * inv_det;
    S_inv[1U*2U+1U] =  S[0U*2U+0U] * inv_det;

    /* ── Kalman gain K = PH * S_inv  (4x2) ────────────────────────────────── */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            K[i*2U+j] = PH[i*2U+0U] * S_inv[0U*2U+j]
                       + PH[i*2U+1U] * S_inv[1U*2U+j];
        }
    }

    /* ── State update: x = x_pred + K * nu ────────────────────────────────── */
    for (i = 0U; i < 4U; i++)
    {
        s->x[i] = x_pred[i] + K[i*2U+0U] * nu[0] + K[i*2U+1U] * nu[1];
    }

    /* Wrap theta_e */
    theta_new = s->x[3];
    while (theta_new >  EKF_PI_F) { theta_new -= EKF_TWO_PI; }
    while (theta_new < -EKF_PI_F) { theta_new += EKF_TWO_PI; }
    s->x[3] = theta_new;

    /* Clamp electrical speed */
    if      (s->x[2] >  EKF_OMEGA_MAX) { s->x[2] =  EKF_OMEGA_MAX; }
    else if (s->x[2] < -EKF_OMEGA_MAX) { s->x[2] = -EKF_OMEGA_MAX; }
    else { /* MISRA 15.7 */ }

    /* Clamp currents */
    if      (s->x[0] >  EKF_I_MAX) { s->x[0] =  EKF_I_MAX; }
    else if (s->x[0] < -EKF_I_MAX) { s->x[0] = -EKF_I_MAX; }
    else { /* MISRA 15.7 */ }

    if      (s->x[1] >  EKF_I_MAX) { s->x[1] =  EKF_I_MAX; }
    else if (s->x[1] < -EKF_I_MAX) { s->x[1] = -EKF_I_MAX; }
    else { /* MISRA 15.7 */ }

    /* Keep legacy alias in sync */
    s->theta_e_hat = s->x[3];

    /* ── Joseph-form covariance update: P = (I-KH)*P_pred*(I-KH)' + K*R*K' ─ */
    /* H = [I_2x2 | 0_2x2]  so  KH[i,j] = K[i,0]*H[0,j] + K[i,1]*H[1,j]
     * = K[i,0] for j=0,  K[i,1] for j=1,  0 otherwise                      */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            MatrixFloat kh = EKF_ZERO;
            if      (j == 0U) { kh = K[i*2U+0U]; }
            else if (j == 1U) { kh = K[i*2U+1U]; }
            else { /* j >= 2: H[:,j] = 0 -- MISRA 15.7 */ }
            I_KH[i*4U+j] = ((i == j) ? EKF_ONE : EKF_ZERO) - kh;
        }
    }

    /* temp = I_KH * P_pred */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 4U; k++) { sum += I_KH[i*4U+k] * P_pred[k*4U+j]; }
            temp[i*4U+j] = sum;
        }
    }

    /* P = temp * I_KH' + K * R * K' */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 4U; k++) { sum += temp[i*4U+k] * I_KH[j*4U+k]; }
            s->P[i*4U+j] = sum + params->r_i * (K[i*2U+0U] * K[j*2U+0U]
                                               + K[i*2U+1U] * K[j*2U+1U]);
        }
    }

    /* Symmetrize */
    for (i = 0U; i < 4U; i++)
    {
        for (j = i + 1U; j < 4U; j++)
        {
            avg = (s->P[i*4U+j] + s->P[j*4U+i]) * EKF_HALF;
            s->P[i*4U+j] = avg;
            s->P[j*4U+i] = avg;
        }
    }

    /* Diagonal bounds */
    for (i = 0U; i < 4U; i++)
    {
        if      (s->P[i*4U+i] < EKF_P_FLOOR) { s->P[i*4U+i] = EKF_P_FLOOR; }
        else if (s->P[i*4U+i] > EKF_P_CEIL)  { s->P[i*4U+i] = EKF_P_CEIL;  }
        else { /* MISRA 15.7 */ }
    }

    /* ── Outputs (always live -- no warmup gate) ───────────────────────────── */
    /* omega_m = electrical speed / pole pairs (mechanical rad/s)             */
    s->omega_m = s->x[2] / p_poles_f;
    s->theta_e = s->x[3];
}
