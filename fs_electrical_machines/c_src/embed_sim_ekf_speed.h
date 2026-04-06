/**********************************************************************************************************************
 * \file      embed_sim_ekf_speed.h
 * \brief     Extended Kalman Filter speed observer for PMSM -- NANOTEC DB42S02
 *
 * \details   Four-state EKF for sensorless / encoder-aided speed estimation.
 *
 *            State vector:  x = [ id,      iq,      omega_m,  theta_e ]
 *            Inputs:        ia, ib, ic  (phase currents [A])
 *                           v_alpha, v_beta  (stationary-frame voltages [V])
 *            Outputs:       omega_m  (mechanical speed [rad/s])
 *                           theta_e  (electrical angle [rad])
 *
 *            Prediction uses nonlinear Euler integration of the dq voltage
 *            model.  The measurement Jacobian H linearises the Park transform
 *            around the current angle estimate.  Covariance update uses the
 *            numerically stable Joseph form.
 *
 *            Outputs are gated to zero for EKF_WARMUP steps (~20 ms at 20 kHz)
 *            while the covariance matrix converges from its initial diagonal P0.
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types explicit via MatrixFloat / uint32_T.
 *              Rule 10.4  : no mixed-mode arithmetic.
 *              Rule 15.5  : single return per function.
 *              Rule 15.7  : every if-else chain has a final else.
 *
 * \version   2.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#ifndef EMBED_SIM_EKF_SPEED_H_
#define EMBED_SIM_EKF_SPEED_H_

#include "embed_sim_matrix.h"


/**********************************************************************************************************************
 * \defgroup EKF_Constants  EKF fixed dimensions and guard values
 * @{
 *********************************************************************************************************************/

/** \brief State dimension: [id, iq, omega_m]  (theta_e integrated separately) */
#define EKF_N           (3U)

/** \brief Measurement dimension: [i_alpha_meas, i_beta_meas]             */
#define EKF_M           (2U)

/** \brief Warmup step count before outputs are valid (~20 ms at 20 kHz)  */
#define EKF_WARMUP      (400U)

/** \brief Maximum mechanical speed estimate [rad/s] -- hard clamp        */
#define EKF_OMEGA_MAX   ((MatrixFloat)700.0f)

/** \brief Covariance diagonal floor -- prevents P from collapsing        */
#define EKF_P_FLOOR     ((MatrixFloat)1.0e-8f)

/** \brief Covariance diagonal ceiling -- prevents P from diverging       */
#define EKF_P_CEIL      ((MatrixFloat)1.0e6f)

/** \brief Innovation covariance determinant guard [prevents div-by-zero] */
#define EKF_DET_MIN     ((MatrixFloat)1.0e-12f)

/** \brief Current state clamp [A] -- 10x motor I_MAX, prevents integrator runaway */
#define EKF_I_MAX       ((MatrixFloat)35.7f)

/** @} */


/**********************************************************************************************************************
 * \struct  DFC_EKF_Speed_T
 * \brief   EKF observer state.  Owned by value inside DFC_State_T.
 *
 * \details Initialise with EKF_Speed_Init() before first EKF_Speed_Step().
 *          Do not read omega_m or theta_e until step_count > EKF_WARMUP.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat x[EKF_N];          /**< State vector: [id, iq, omega_m]              */
    MatrixFloat P[EKF_N * EKF_N];  /**< Covariance matrix 3x3, row-major             */
    MatrixFloat theta_e_hat;       /**< Internal integrated electrical angle [rad]   */
    MatrixFloat omega_m;           /**< Gated mechanical speed output [rad/s]        */
    MatrixFloat theta_e;           /**< Gated electrical angle output [rad]          */
    uint32_T    step_count;        /**< Steps since last Init -- used for warmup gate */
} DFC_EKF_Speed_T;


/**********************************************************************************************************************
 * \struct  EKF_Speed_Params_T
 * \brief   EKF tuning and motor parameters.
 *
 * \details Populated by DFC_Controller_Init() from the motor #defines in
 *          embed_sim_dfc_controller.h.  Override via
 *          DFC_Controller_SetEKFParams() to tune noise without recompiling.
 *********************************************************************************************************************/
typedef struct
{
    /*--- Motor parameters ---*/
    MatrixFloat R_s;        /**< Stator resistance [Ohm]                            */
    MatrixFloat L_d;        /**< d-axis inductance [H]                              */
    MatrixFloat L_q;        /**< q-axis inductance [H]                              */
    MatrixFloat lambda_pm;  /**< Permanent magnet flux linkage [Wb]                 */
    uint32_T    p_poles;    /**< Pole pairs                                         */

    /*--- Process noise covariances (diagonal Q entries) ---*/
    MatrixFloat q_i;        /**< Current process noise [A^2]       typical 1e-4     */
    MatrixFloat q_omega;    /**< Speed process noise [(rad/s)^2]   typical 1e-2     */
    MatrixFloat q_theta;    /**< Angle process noise [rad^2]        typical 1e-4     */

    /*--- Measurement noise covariance ---*/
    MatrixFloat r_i;        /**< Current measurement noise [A^2]   typical 1e-4     */

    /*--- Initial covariance diagonal (P0) ---*/
    MatrixFloat p0_i;       /**< Initial current covariance [A^2]  typical 1.0      */
    MatrixFloat p0_omega;   /**< Initial speed covariance [(rad/s)^2] typical 100.0 */
    MatrixFloat p0_theta;   /**< Initial angle covariance [rad^2]  typical 1.0      */
} EKF_Speed_Params_T;


/**********************************************************************************************************************
 * \defgroup EKF_API  Public API
 * @{
 *********************************************************************************************************************/

/**
 * \brief   Initialise EKF state and covariance.
 *
 * \details Zeroes x[], sets P to diagonal(p0_i, p0_i, p0_omega, p0_theta),
 *          resets step_count, and clears omega_m / theta_e outputs.
 *          Call once before the first EKF_Speed_Step(), or after a fault reset.
 *
 * \param[out] s       EKF state.  Must not be NULL.
 * \param[in]  params  Noise and motor parameters.  Must not be NULL.
 */
extern void EKF_Speed_Init(
    DFC_EKF_Speed_T          * const s,
    const EKF_Speed_Params_T * const params);

/**
 * \brief   Execute one EKF predict-update cycle.  Call from the 20 kHz ISR.
 *
 * \details Sensorless 3-state EKF: x = [id, iq, omega_m].
 *          theta_e is integrated internally from omega_m -- no encoder needed.
 *          Measurement is [i_alpha, i_beta] in the stationary frame.
 *          Predicted measurement is h(x) = InvPark([id,iq], theta_e_hat).
 *          On return, s->omega_m and s->theta_e hold the gated estimates.
 *          Both are zero while s->step_count <= EKF_WARMUP (~20 ms at 20 kHz).
 *
 * \param[in,out] s        EKF state.
 * \param[in]     ia       Phase A current [A].
 * \param[in]     ib       Phase B current [A].
 * \param[in]     ic       Phase C current [A]  (unused, retained for API consistency).
 * \param[in]     v_alpha  Alpha-axis voltage from previous step [V].
 * \param[in]     v_beta   Beta-axis voltage from previous step [V].
 * \param[in]     dt       Step period [s].
 * \param[in]     params   Noise and motor parameters.
 */
extern void EKF_Speed_Step(
    DFC_EKF_Speed_T          * const s,
    const MatrixFloat                 ia,
    const MatrixFloat                 ib,
    const MatrixFloat                 ic,
    const MatrixFloat                 v_alpha,
    const MatrixFloat                 v_beta,
    const MatrixFloat                 dt,
    const EKF_Speed_Params_T * const params);

/**
 * \brief   Reset EKF state.  Equivalent to EKF_Speed_Init().
 *
 * \param[in,out] s       EKF state.
 * \param[in]     params  Noise and motor parameters.
 */
extern void EKF_Speed_Reset(
    DFC_EKF_Speed_T          * const s,
    const EKF_Speed_Params_T * const params);

/** @} */

#endif /* EMBED_SIM_EKF_SPEED_H_ */
