/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.h
 * \brief     Differential Flatness FOC Controller -- NANOTEC DB42S02
 *
 * \details   Full-state DFC with selectable speed observer back-end.
 *            Three observer modes are available and can be switched at runtime
 *            via DFC_Controller_SetObserverMode() without stopping the motor:
 *
 *              DFC_OBS_SMO   (0) -- Sliding Mode Observer only.
 *                                   Original production mode; motor confirmed
 *                                   running at 3000 RPM.  Default on Init.
 *
 *              DFC_OBS_EKF   (1) -- Extended Kalman Filter only.
 *                                   State: x = [id, iq, omega_m, theta_e].
 *                                   Provides smoother speed estimate at the
 *                                   cost of one 4x4 matrix multiply per step.
 *
 *              DFC_OBS_BLEND (2) -- Convex blend of SMO and EKF outputs.
 *                                   omega_blend = (1-w)*omega_smo + w*omega_ekf
 *                                   Set obs_blend_w via AURIX overlay to sweep
 *                                   0.0 (full SMO) -> 1.0 (full EKF) live.
 *
 *            The SMO always executes -- it feeds SpeedFusion regardless of
 *            mode.  The EKF executes only in DFC_OBS_EKF and DFC_OBS_BLEND
 *            to avoid the trig + 4x4 matrix cost at 20 kHz in the default
 *            DFC_OBS_SMO production mode.  obs_mode controls which estimate
 *            drives the speed P-loop.
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

#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_gains.h"
#include "embed_sim_ekf_speed.h"


/**********************************************************************************************************************
 * \defgroup DFC_MotorParams  Motor parameters -- NANOTEC DB42S02
 * @{
 *********************************************************************************************************************/

/** \brief Number of pole pairs                                           */
#define DFC_P_POLES          (4U)

/** \brief Stator resistance [Ohm]                                        */
#define DFC_R_S              ((MatrixFloat)0.285f)

/** \brief d-axis inductance [H]                                          */
#define DFC_L_D              ((MatrixFloat)0.0003675f)

/** \brief q-axis inductance [H]                                          */
#define DFC_L_Q              ((MatrixFloat)0.0003675f)

/** \brief Permanent magnet flux linkage [Wb]                             */
#define DFC_LAMBDA_PM        ((MatrixFloat)0.0014f)

/** \brief Maximum phase current [A]                                      */
#define DFC_I_MAX            ((MatrixFloat)3.57f)

/** \brief DC bus voltage [V]                                             */
#define DFC_V_DC             ((MatrixFloat)17.0f)

/** \brief Maximum phase voltage = V_DC / sqrt(3) [V]                    */
#define DFC_V_MAX            (DFC_V_DC / ((MatrixFloat)1.73205080757f))

/** @} */


/**********************************************************************************************************************
 * \defgroup DFC_FusionParams  SpeedFusion complementary filter parameters
 * @{
 *********************************************************************************************************************/

/** \brief Lower speed threshold [rad/s] -- alpha = 0 below this value   */
#define DFC_FUSION_OMEGA_LO  ((MatrixFloat)50.0f)

/** \brief Upper speed threshold [rad/s] -- alpha = 1 above this value   */
#define DFC_FUSION_OMEGA_HI  ((MatrixFloat)250.0f)

/** \brief Encoder IIR smoothing coefficient at low speed (alpha = 0)    */
#define DFC_FUSION_IIR_LO    ((MatrixFloat)0.05f)

/** \brief Encoder IIR smoothing coefficient at high speed (alpha = 1)   */
#define DFC_FUSION_IIR_HI    ((MatrixFloat)0.30f)

/** @} */


/**********************************************************************************************************************
 * \defgroup DFC_SMOParams  Sliding Mode Observer parameters
 * @{
 *********************************************************************************************************************/

/** \brief SMO switching gain [V].  Must exceed max back-EMF: 1256 * 0.0014 = 1.76 V  */
#define DFC_SMO_K            ((MatrixFloat)2.0f)

/** \brief SMO back-EMF LPF time constant [s] -- corner frequency ~800 Hz */
#define DFC_SMO_TAU_E        ((MatrixFloat)0.0002f)

/** \brief SMO warmup step count -- gates omega_e until BEMF has converged */
#define DFC_SMO_WARMUP_STEPS (400U)

/**********************************************************************************************************************
 * \brief  SMO speed spike clamp [rad/s electrical].
 *
 * \details If omega_e_hat from the finite-difference on theta_e exceeds this
 *          value the sample is rejected and the previous filtered value is
 *          held.  Set to 10x the maximum operating electrical speed:
 *            omega_max_mech = 2200 RPM * 2pi/60 = 230 rad/s
 *            omega_max_e    = 230 * 4 (pole pairs) = 920 rad/s
 *          A 3x margin gives 2760; round up to 3000 for safety.
 *
 * \units   rad/s (electrical)
 *********************************************************************************************************************/
#define DFC_SMO_OMEGA_MAX    ((MatrixFloat)3000.0f)

/**********************************************************************************************************************
 * \brief  SMO plausibility band vs encoder [rad/s electrical].
 *
 * \details If the SMO electrical speed deviates from the encoder-derived
 *          electrical speed by more than this value, the SMO output is
 *          replaced by the encoder value for the blend computation.
 *          This catches residual spikes that survive the omega_e_hat clamp
 *          but are still inconsistent with the encoder ground truth.
 *
 *          Value = 4 * DFC_FUSION_OMEGA_HI = 4 * 250 = 1000 rad/s (electrical).
 *          Equivalent to 239 RPM deviation tolerance at 2200 RPM target.
 *
 * \units   rad/s (electrical)
 *********************************************************************************************************************/
#define DFC_SMO_PLAUS_BAND   ((MatrixFloat)1000.0f)

/** @} */


/**********************************************************************************************************************
 * \defgroup DFC_CtrlParams  Controller tuning parameters
 * @{
 *********************************************************************************************************************/

/** \brief Current derivative LPF time constant [s]                      */
#define DFC_DIQ_TAU          ((MatrixFloat)0.001f)

/** \brief Diagnostic logging interval [s] -- 1 kHz snapshot rate        */
#define DFC_LOG_INTERVAL     ((MatrixFloat)0.001f)

/** @} */


/**********************************************************************************************************************
 * \defgroup DFC_EKFDefaults  EKF noise defaults -- populated by DFC_Controller_Init
 *
 * \details  These values seed EKF_Speed_Params_T inside DFC_State_T.
 *           Override via DFC_Controller_SetEKFParams() after Init if needed.
 * @{
 *********************************************************************************************************************/

/** \brief EKF current process noise [A^2]                               */
#define DFC_EKF_Q_I          ((MatrixFloat)1.0e-4f)

/** \brief EKF speed process noise [(rad/s)^2]                           */
#define DFC_EKF_Q_OMEGA      ((MatrixFloat)1.0e-2f)

/** \brief EKF angle process noise [rad^2]                               */
#define DFC_EKF_Q_THETA      ((MatrixFloat)1.0e-4f)

/** \brief EKF current measurement noise [A^2]                           */
#define DFC_EKF_R_I          ((MatrixFloat)1.0e-4f)

/** \brief EKF initial current covariance [A^2]                          */
#define DFC_EKF_P0_I         ((MatrixFloat)1.0f)

/** \brief EKF initial speed covariance [(rad/s)^2]                      */
#define DFC_EKF_P0_OMEGA     ((MatrixFloat)100.0f)

/** \brief EKF initial angle covariance [rad^2]                          */
#define DFC_EKF_P0_THETA     ((MatrixFloat)1.0f)

/** @} */


/**********************************************************************************************************************
 * \enum   DFC_ObserverMode_T
 * \brief  Speed observer back-end selection.
 *
 * \details Switch at runtime via DFC_Controller_SetObserverMode().
 *          The change takes effect on the next DFC_Controller_Step() call.
 *          Default after DFC_Controller_Init() is DFC_OBS_SMO (0), which
 *          preserves the original production behaviour exactly.
 *********************************************************************************************************************/
typedef enum
{
    DFC_OBS_SMO   = 0U,   /**< SMO only -- original production mode          */
    DFC_OBS_EKF   = 1U,   /**< EKF only -- smooth estimate, higher CPU cost  */
    DFC_OBS_BLEND = 2U    /**< Convex blend: (1-w)*SMO + w*EKF, w=obs_blend_w */
} DFC_ObserverMode_T;


/**********************************************************************************************************************
 * \struct DFC_SpeedFusion_T
 * \brief  Speed-dependent complementary filter state.
 *
 * \details Blends encoder finite-difference (IIR-smoothed) with the active
 *          observer output.  alpha = 0 -> full encoder; alpha = 1 -> full observer.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat theta_m_prev;    /**< Previous mechanical angle [rad]              */
    MatrixFloat omega_enc_filt;  /**< IIR-filtered encoder mechanical speed [rad/s] */
    MatrixFloat omega_e_prev;    /**< Previous fused electrical speed [rad/s]      */
    MatrixFloat alpha;           /**< Current fusion weight (0 = encoder, 1 = obs) */
    MatrixFloat omega_enc_mech;  /**< Filtered encoder mechanical speed [rad/s]    */
} DFC_SpeedFusion_T;


/**********************************************************************************************************************
 * \struct DFC_SMO_T
 * \brief  Sliding Mode Observer state (stationary alphabeta frame).
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat i_hat_alpha;   /**< Estimated alpha-axis current [A]             */
    MatrixFloat i_hat_beta;    /**< Estimated beta-axis current [A]              */
    MatrixFloat e_hat_alpha;   /**< LPF-filtered back-EMF alpha component [V]   */
    MatrixFloat e_hat_beta;    /**< LPF-filtered back-EMF beta component [V]    */
    MatrixFloat theta_e_hat;   /**< Estimated electrical angle [rad]            */
    MatrixFloat omega_e_hat;   /**< Raw electrical speed estimate [rad/s]        */
    MatrixFloat omega_e_filt;  /**< LPF-filtered electrical speed [rad/s]        */
    MatrixFloat theta_e_prev;  /**< Previous angle used for speed extraction [rad] */
} DFC_SMO_T;


/**********************************************************************************************************************
 * \struct DFC_Input_T
 * \brief  Per-step input to DFC_Controller_Step().
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat omega_ref_mech;  /**< Mechanical speed reference [rad/s]  */
    MatrixFloat theta_m;         /**< Encoder mechanical angle [rad]       */
    MatrixFloat ia;              /**< Phase A current [A]                  */
    MatrixFloat ib;              /**< Phase B current [A]                  */
    MatrixFloat ic;              /**< Phase C current [A]                  */
} DFC_Input_T;


/**********************************************************************************************************************
 * \struct DFC_Output_T
 * \brief  Per-step output from DFC_Controller_Step().
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat v_alpha;   /**< Alpha-axis voltage reference [V]  */
    MatrixFloat v_beta;    /**< Beta-axis voltage reference [V]   */
} DFC_Output_T;


/**********************************************************************************************************************
 * \struct DFC_State_T
 * \brief  Complete Differential Flatness Controller state.
 *
 * \details All sub-states are owned by value (no heap allocation).
 *          The EKF sub-state is always allocated; it runs only when
 *          obs_mode is DFC_OBS_EKF or DFC_OBS_BLEND.
 *********************************************************************************************************************/
typedef struct
{
    /*--- Speed estimation ---*/
    DFC_SpeedFusion_T  fusion;          /**< Complementary filter state              */
    DFC_SMO_T          smo;             /**< Sliding Mode Observer state             */
    DFC_EKF_Speed_T    ekf;             /**< EKF observer state                      */
    EKF_Speed_Params_T ekf_params;      /**< EKF noise / motor parameters            */

    /*--- Observer mode selector (written by DFC_Controller_SetObserverMode) ---*/
    DFC_ObserverMode_T obs_mode;        /**< Active observer back-end                */
    MatrixFloat        obs_blend_w;     /**< Blend weight: 0.0 = SMO, 1.0 = EKF     */

    /*--- Delayed voltages for SMO (z-1) ---*/
    MatrixFloat v_alpha_prev;           /**< Alpha voltage one step ago [V]          */
    MatrixFloat v_beta_prev;            /**< Beta voltage one step ago [V]           */

    /*--- Reference trajectory ---*/
    MatrixFloat iq_ref_prev;            /**< Previous iq_ref for derivative [A]      */
    MatrixFloat diq_filt;               /**< LPF-filtered diq_ref/dt [A/s]           */

    /*--- Warmup counter ---*/
    uint32_T smo_warmup_cnt;            /**< Steps since init -- gates SMO output    */

    /*--- Coordinate transforms ---*/
    Clarke_T   clarke_state;
    Park_T     park_state;
    InvPark_T  inv_park_state;

    /*--- Diagnostic logging (1 kHz snapshot) ---*/
    MatrixFloat log_speed_ref;          /**< Speed reference [RPM]                   */
    MatrixFloat log_iq_ref;             /**< iq reference [A]                        */
    MatrixFloat log_id;                 /**< Measured id [A]                         */
    MatrixFloat log_iq;                 /**< Measured iq [A]                         */
    MatrixFloat log_alpha;              /**< Fusion weight                            */
    MatrixFloat log_omega_e;            /**< Active observer mechanical speed [rad/s] -- drives P-loop */
    MatrixFloat log_omega_smo;          /**< SMO mechanical speed estimate [rad/s]   */
    MatrixFloat log_omega_ekf;          /**< EKF mechanical speed estimate [rad/s]   */
    uint32_T    log_counter;            /**< Step counter                            */
    MatrixFloat log_next_time;          /**< Next log threshold [s]                  */

} DFC_State_T;


/**********************************************************************************************************************
 * \defgroup DFC_API  Public API
 * @{
 *********************************************************************************************************************/

/**
 * \brief   Initialise all controller state.  Call once before the ISR starts.
 *
 * \details Zeroes DFC_State_T, initialises transform blocks, seeds EKF params
 *          from motor #defines, and sets obs_mode = DFC_OBS_SMO.
 *
 * \param[out] s   Controller state.  Must not be NULL.
 * \param[in]  dt  Nominal sampling period [s].  Stored for reference only.
 */
extern void DFC_Controller_Init(
    DFC_State_T     * const s,
    const MatrixFloat         dt);

/**
 * \brief   Execute one FOC step.  Call from the 20 kHz GTM ISR.
 *
 * \param[in,out] s   Controller state.
 * \param[in]     u   Inputs for this step.
 * \param[in]     dt  Actual step period [s] from GTM timer.
 * \param[out]    y   Voltage outputs (v_alpha, v_beta).
 */
extern void DFC_Controller_Step(
    DFC_State_T        * const s,
    const DFC_Input_T  * const u,
    const MatrixFloat           dt,
    DFC_Output_T       * const y);

/**
 * \brief   Reset all integrators and state.  Call on motor stop or fault.
 *          Observer mode and blend weight are preserved across reset.
 *
 * \param[in,out] s  Controller state.
 */
extern void DFC_Controller_Reset(
    DFC_State_T * const s);

/**
 * \brief   Select the active speed observer back-end.
 *
 * \details Safe to call while the motor is running.  The new mode takes
 *          effect on the next DFC_Controller_Step() call.
 *
 *          When switching from DFC_OBS_SMO to DFC_OBS_EKF or DFC_OBS_BLEND
 *          the EKF state is seeded from the current encoder IIR speed and
 *          SMO angle, and step_count is set past EKF_WARMUP so the output
 *          is live immediately with no 20 ms blind period.  Speed control
 *          remains closed-loop through the transition.
 *
 * \param[in,out] s     Controller state.
 * \param[in]     mode  Observer mode to activate.
 * \param[in]     blend_w  Blend weight [0.0, 1.0].  Used only in
 *                         DFC_OBS_BLEND mode.  Clamped internally.
 */
extern void DFC_Controller_SetObserverMode(
    DFC_State_T              * const s,
    const DFC_ObserverMode_T          mode,
    const MatrixFloat                 blend_w);

/**
 * \brief   Override EKF noise / motor parameters after Init.
 *
 * \details Copies src into s->ekf_params and re-initialises the EKF state
 *          so the new covariance takes effect immediately.
 *
 * \param[in,out] s    Controller state.
 * \param[in]     src  New parameter set.  Must not be NULL.
 */
extern void DFC_Controller_SetEKFParams(
    DFC_State_T              * const s,
    const EKF_Speed_Params_T * const src);

/**
 * \brief   Read the latest 1 kHz diagnostic snapshot.
 *
 * \param[in]  s              Controller state.
 * \param[out] speed_ref_rpm  Speed reference [RPM].
 * \param[out] iq_ref         q-axis current reference [A].
 * \param[out] id             Measured d-axis current [A].
 * \param[out] iq             Measured q-axis current [A].
 * \param[out] alpha          SpeedFusion weight.
 * \param[out] omega_e        Fused mechanical speed [rad/s] (= electrical / p_poles).
 * \param[out] omega_smo      SMO mechanical speed estimate [rad/s].
 * \param[out] omega_ekf      EKF mechanical speed estimate [rad/s].
 */
extern void DFC_Controller_GetDiagnostics(
    const DFC_State_T * const s,
    MatrixFloat       * const speed_ref_rpm,
    MatrixFloat       * const iq_ref,
    MatrixFloat       * const id,
    MatrixFloat       * const iq,
    MatrixFloat       * const alpha,
    MatrixFloat       * const omega_e,
    MatrixFloat       * const omega_smo,
    MatrixFloat       * const omega_ekf);

/** @} */

#endif /* EMBED_SIM_DFC_CONTROLLER_H_ */
