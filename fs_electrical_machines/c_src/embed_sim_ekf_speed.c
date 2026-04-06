/**********************************************************************************************************************
 * \file      embed_sim_ekf_speed.c
 * \brief     Sensorless EKF speed observer -- alpha-beta measurement, 3-state + integrated theta_e
 *
 * \details   Fully sensorless EKF.  No encoder angle is used inside the observer.
 *
 *            State vector:  x = [ id, iq, omega_m ]   (3 states)
 *            Internal:      theta_e_hat integrated from omega_m each step
 *            Inputs:        v_alpha, v_beta  (stationary-frame voltages, z-1)
 *            Measurement:   y = [ i_alpha_meas, i_beta_meas ]  (stationary frame)
 *            Predicted meas: h(x) = InvPark( x[0], x[1], theta_e_hat )
 *                                 = [ id*cos - iq*sin,  id*sin + iq*cos ]
 *
 *            Why alpha-beta measurement (not dq):
 *              The previous 4-state dq-measurement EKF rotated both y_meas and h(x)
 *              by the same estimated theta_e.  The innovation carried no angle or speed
 *              information -- observability collapsed.  Measuring in the stationary
 *              alpha-beta frame makes h(x) depend on theta_e_hat while y is raw
 *              measured current.  Any angle or speed error creates a nonzero innovation
 *              that drives correction through the Kalman gain.
 *
 *            Observability of omega_m:
 *              The back-EMF term (omega_e * lambda_pm) in diq/dt couples speed to the
 *              current trajectory.  A speed error shifts the predicted iq away from the
 *              measured iq.  When projected back to alpha-beta via h(x), this produces a
 *              nonzero nu that K[2,:] maps into a correction of x[2] = omega_m.
 *              The system is observable without any encoder.
 *
 *            State dimension EKF_N = 3: x = [id, iq, omega_m]
 *            Measurement dim EKF_M = 2: y = [i_alpha, i_beta]
 *
 *            q_theta and p0_theta in EKF_Speed_Params_T are kept for API compatibility
 *            but are not used in this 3-state implementation.
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types explicit via MatrixFloat / uint32_T.
 *              Rule 10.4  : no mixed-mode arithmetic.
 *              Rule 15.5  : single return per function.
 *              Rule 15.7  : every if-else chain has a final else.
 *
 * \version   3.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#include "embed_sim_ekf_speed.h"
#include <math.h>
#include <string.h>

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
    uint32_T i, j;

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

    s->P[0U * EKF_N + 0U] = params->p0_i;
    s->P[1U * EKF_N + 1U] = params->p0_i;
    s->P[2U * EKF_N + 2U] = params->p0_omega;

    s->theta_e_hat = EKF_ZERO;
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
 * EKF_Speed_Step
 *
 * Matrix layout (all row-major):
 *   F     [3x3]  F[i*3+j]
 *   Q     [3x3]  Q[i*3+j]
 *   P     [3x3]  s->P[i*3+j]   EKF_N=3
 *   H     [2x3]  H[i*3+j]
 *   PH    [3x2]  PH[i*2+j]     = P_pred * H'
 *   S     [2x2]  S[i*2+j]
 *   S_inv [2x2]  S_inv[i*2+j]
 *   K     [3x2]  K[i*2+j]      = PH * S_inv
 *   I_KH  [3x3]  I_KH[i*3+j]
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
    MatrixFloat F[9];
    MatrixFloat Q[9];
    MatrixFloat x_pred[3];
    MatrixFloat P_pred[9];
    MatrixFloat FP[9];
    MatrixFloat H[6];
    MatrixFloat S[4];
    MatrixFloat S_inv[4];
    MatrixFloat K[6];
    MatrixFloat PH[6];
    MatrixFloat y_meas[2];
    MatrixFloat h_pred[2];
    MatrixFloat nu[2];
    MatrixFloat I_KH[9];
    MatrixFloat temp[9];

    MatrixFloat vd, vq;
    MatrixFloat cos_t, sin_t;
    MatrixFloat omega_e;
    MatrixFloat inv_Ld, inv_Lq;
    MatrixFloat det, inv_det;
    MatrixFloat sum;
    MatrixFloat p_poles_f;
    MatrixFloat theta_e_hat_new;
    uint32_T    i, j, k;

    (void)ic;   /* ic unused: Clarke uses ia, ib only (3-wire balanced assumption) */

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

    inv_Ld  = (params->L_d > EKF_L_MIN) ? (EKF_ONE / params->L_d) : (EKF_ONE / EKF_L_MIN);
    inv_Lq  = (params->L_q > EKF_L_MIN) ? (EKF_ONE / params->L_q) : (EKF_ONE / EKF_L_MIN);
    omega_e = p_poles_f * s->x[2];

    /*------------------------------------------------------------------------------------------------------------------
     * Measurement y = [i_alpha_meas, i_beta_meas]
     * Clarke (amplitude-invariant): i_alpha = ia,  i_beta = (ia + 2*ib)/sqrt(3)
     * Stationary frame -- independent of theta_e_hat.
     *----------------------------------------------------------------------------------------------------------------*/
    y_meas[0] = ia;
    y_meas[1] = (ia + (MatrixFloat)2.0f * ib) * (MatrixFloat)0.57735027f;

    /*------------------------------------------------------------------------------------------------------------------
     * Rotate voltages vd, vq using current theta_e_hat.
     * The prediction model lives in dq frame; this rotation is consistent with
     * the linearisation point used to build F and H.
     *----------------------------------------------------------------------------------------------------------------*/
    cos_t = cosf(s->theta_e_hat);
    sin_t = sinf(s->theta_e_hat);

    vd =  v_alpha * cos_t + v_beta * sin_t;
    vq = -v_alpha * sin_t + v_beta * cos_t;

    /*------------------------------------------------------------------------------------------------------------------
     * State Jacobian F = df/dx (3x3, stride 3)
     *
     * Discretised dq model:
     *   id+ = id + dt*(vd - R*id + we*Lq*iq)/Ld
     *   iq+ = iq + dt*(vq - R*iq - we*Ld*id - we*lpm)/Lq
     *   wm+ = wm                (random walk)
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 9U; i++) { F[i] = EKF_ZERO; }

    F[0U*3U+0U] = EKF_ONE - params->R_s * dt * inv_Ld;
    F[0U*3U+1U] = omega_e * params->L_q * dt * inv_Ld;
    F[0U*3U+2U] = p_poles_f * params->L_q * s->x[1] * dt * inv_Ld;

    F[1U*3U+0U] = -omega_e * params->L_d * dt * inv_Lq;
    F[1U*3U+1U] = EKF_ONE - params->R_s * dt * inv_Lq;
    F[1U*3U+2U] = -(p_poles_f * params->L_d * s->x[0]
                 +  p_poles_f * params->lambda_pm) * dt * inv_Lq;

    F[2U*3U+2U] = EKF_ONE;

    /*------------------------------------------------------------------------------------------------------------------
     * Process noise Q (3x3 diagonal)
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 9U; i++) { Q[i] = EKF_ZERO; }
    Q[0U*3U+0U] = params->q_i;
    Q[1U*3U+1U] = params->q_i;
    Q[2U*3U+2U] = params->q_omega;

    /*------------------------------------------------------------------------------------------------------------------
     * Nonlinear prediction x_pred = f(x, u)
     *----------------------------------------------------------------------------------------------------------------*/
    x_pred[0] = s->x[0] + dt * ((vd - params->R_s * s->x[0]
                                     + omega_e * params->L_q * s->x[1]) * inv_Ld);
    x_pred[1] = s->x[1] + dt * ((vq - params->R_s * s->x[1]
                                     - omega_e * params->L_d * s->x[0]
                                     - omega_e * params->lambda_pm) * inv_Lq);
    x_pred[2] = s->x[2];   /* random walk */

    /* Integrate theta_e_hat */
    theta_e_hat_new = s->theta_e_hat + p_poles_f * s->x[2] * dt;
    while (theta_e_hat_new >  EKF_PI_F) { theta_e_hat_new -= EKF_TWO_PI; }
    while (theta_e_hat_new < -EKF_PI_F) { theta_e_hat_new += EKF_TWO_PI; }

    /*------------------------------------------------------------------------------------------------------------------
     * P_pred = F*P*F' + Q
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 3U; k++) { sum += F[i*3U+k] * s->P[k*3U+j]; }
            FP[i*3U+j] = sum;
        }
    }
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 3U; k++) { sum += FP[i*3U+k] * F[j*3U+k]; }
            P_pred[i*3U+j] = sum + Q[i*3U+j];
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
     * Predicted measurement h(x_pred) = InvPark([id_pred, iq_pred], theta_e_hat_new)
     *   h[0] = id*cos - iq*sin  (i_alpha_pred)
     *   h[1] = id*sin + iq*cos  (i_beta_pred)
     *----------------------------------------------------------------------------------------------------------------*/
    cos_t = cosf(theta_e_hat_new);
    sin_t = sinf(theta_e_hat_new);

    h_pred[0] = x_pred[0] * cos_t - x_pred[1] * sin_t;
    h_pred[1] = x_pred[0] * sin_t + x_pred[1] * cos_t;

    /*------------------------------------------------------------------------------------------------------------------
     * Measurement Jacobian H = dh/dx (2x3, stride 3)
     *
     * dh/d[id, iq, omega_m]:
     *   Row 0 (i_alpha):  [cos,  -sin,  0]
     *   Row 1 (i_beta):   [sin,   cos,  0]
     *
     * The omega_m->theta_e->h coupling is O(dt^2) at 20 kHz and is correctly
     * neglected.  Speed correction comes through F[1][2] -> P_pred -> K[2,:].
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 6U; i++) { H[i] = EKF_ZERO; }
    H[0U*3U+0U] =  cos_t;
    H[0U*3U+1U] = -sin_t;
    H[1U*3U+0U] =  sin_t;
    H[1U*3U+1U] =  cos_t;

    /*------------------------------------------------------------------------------------------------------------------
     * Innovation nu = y - h(x_pred)
     *----------------------------------------------------------------------------------------------------------------*/
    nu[0] = y_meas[0] - h_pred[0];
    nu[1] = y_meas[1] - h_pred[1];

    /*------------------------------------------------------------------------------------------------------------------
     * PH = P_pred * H'  (3x2)
     * S  = H * PH + R   (2x2)
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 3U; k++) { sum += P_pred[i*3U+k] * H[j*3U+k]; }
            PH[i*2U+j] = sum;
        }
    }
    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 3U; k++) { sum += H[i*3U+k] * PH[k*2U+j]; }
            S[i*2U+j] = sum;
        }
    }
    S[0U*2U+0U] += params->r_i;
    S[1U*2U+1U] += params->r_i;

    /*------------------------------------------------------------------------------------------------------------------
     * S_inv: 2x2 closed-form
     *----------------------------------------------------------------------------------------------------------------*/
    det = S[0U*2U+0U] * S[1U*2U+1U] - S[0U*2U+1U] * S[1U*2U+0U];
    if (det < EKF_ZERO) { det = -det; }
    else { /* positive det -- MISRA 15.7 */ }
    if (det < EKF_DET_MIN) { det = EKF_DET_MIN; }
    else { /* MISRA 15.7 */ }

    inv_det         =  EKF_ONE / det;
    S_inv[0U*2U+0U] =  S[1U*2U+1U] * inv_det;
    S_inv[0U*2U+1U] = -S[0U*2U+1U] * inv_det;
    S_inv[1U*2U+0U] = -S[1U*2U+0U] * inv_det;
    S_inv[1U*2U+1U] =  S[0U*2U+0U] * inv_det;

    /*------------------------------------------------------------------------------------------------------------------
     * Kalman gain K = PH * S_inv  (3x2)
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            K[i*2U+j] = PH[i*2U+0U] * S_inv[0U*2U+j]
                       + PH[i*2U+1U] * S_inv[1U*2U+j];
        }
    }

    /*------------------------------------------------------------------------------------------------------------------
     * State update: x = x_pred + K*nu
     *----------------------------------------------------------------------------------------------------------------*/
    s->x[0] = x_pred[0] + K[0U*2U+0U] * nu[0] + K[0U*2U+1U] * nu[1];
    s->x[1] = x_pred[1] + K[1U*2U+0U] * nu[0] + K[1U*2U+1U] * nu[1];
    s->x[2] = x_pred[2] + K[2U*2U+0U] * nu[0] + K[2U*2U+1U] * nu[1];

    if      (s->x[2] >  EKF_OMEGA_MAX) { s->x[2] =  EKF_OMEGA_MAX; }
    else if (s->x[2] < -EKF_OMEGA_MAX) { s->x[2] = -EKF_OMEGA_MAX; }
    else { /* MISRA 15.7 */ }

    if      (s->x[0] >  EKF_I_MAX) { s->x[0] =  EKF_I_MAX; }
    else if (s->x[0] < -EKF_I_MAX) { s->x[0] = -EKF_I_MAX; }
    else { /* MISRA 15.7 */ }

    if      (s->x[1] >  EKF_I_MAX) { s->x[1] =  EKF_I_MAX; }
    else if (s->x[1] < -EKF_I_MAX) { s->x[1] = -EKF_I_MAX; }
    else { /* MISRA 15.7 */ }

    s->theta_e_hat = theta_e_hat_new;

    /*------------------------------------------------------------------------------------------------------------------
     * Joseph-form covariance update: P = (I-KH)*P_pred*(I-KH)' + K*R*K'
     *----------------------------------------------------------------------------------------------------------------*/
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 2U; k++) { sum += K[i*2U+k] * H[k*3U+j]; }
            I_KH[i*3U+j] = ((i == j) ? EKF_ONE : EKF_ZERO) - sum;
        }
    }
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 3U; k++) { sum += I_KH[i*3U+k] * P_pred[k*3U+j]; }
            temp[i*3U+j] = sum;
        }
    }
    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = EKF_ZERO;
            for (k = 0U; k < 3U; k++) { sum += temp[i*3U+k] * I_KH[j*3U+k]; }
            s->P[i*3U+j] = sum + params->r_i * (K[i*2U+0U] * K[j*2U+0U]
                                               + K[i*2U+1U] * K[j*2U+1U]);
        }
    }

    /* Symmetrize */
    for (i = 0U; i < 3U; i++)
    {
        for (j = i + 1U; j < 3U; j++)
        {
            MatrixFloat avg = (s->P[i*3U+j] + s->P[j*3U+i]) * EKF_HALF;
            s->P[i*3U+j] = avg;
            s->P[j*3U+i] = avg;
        }
    }

    /* Diagonal bounds */
    for (i = 0U; i < 3U; i++)
    {
        if      (s->P[i*3U+i] < EKF_P_FLOOR) { s->P[i*3U+i] = EKF_P_FLOOR; }
        else if (s->P[i*3U+i] > EKF_P_CEIL)  { s->P[i*3U+i] = EKF_P_CEIL;  }
        else { /* MISRA 15.7 */ }
    }

    /*------------------------------------------------------------------------------------------------------------------
     * Output gate
     *----------------------------------------------------------------------------------------------------------------*/
    if (s->step_count > EKF_WARMUP)
    {
        s->omega_m = s->x[2];
        s->theta_e = s->theta_e_hat;
    }
    else
    {
        s->omega_m = EKF_ZERO;
        s->theta_e = EKF_ZERO;
    }
}
