/**********************************************************************************************************************
 * \file      PI_FOC.h
 * \brief     PI Field Oriented Control for NANOTEC DB42S02.
 *
 * Implements closed-loop PI FOC with:
 *   - Current loop: pole-zero cancellation (ωc_i = 2π×500 Hz)
 *   - Speed loop: PI with anti-windup (Ti = 0.1 s, ωc_ω = 2π×15 Hz)
 *   - MTPA operation (id_ref = 0)
 *
 * The block integrates Clarke, Park, PI controllers, and Inverse Park
 * transforms into a single atomic control step.
 *
 * Target: Infineon AURIX TriCore TC3xx, ARM Cortex-M4
 *
 * \version   1.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *
 *********************************************************************************************************************/

#ifndef PI_FOC_H_
#define PI_FOC_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_matrix.h"   /* MatrixFloat (= real32_T) — also pulls in embed_sim_sys_types.h */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup pi_foc_motor_params  Motor parameters (NANOTEC DB42S02)
 * \{
 */
/** \brief Number of pole pairs */
#define PI_FOC_P_POLES          (4U)

/** \brief Stator resistance [Ω] */
#define PI_FOC_R_S              ((MatrixFloat)0.19f)

/** \brief d-axis inductance [H] */
#define PI_FOC_L_D              ((MatrixFloat)0.000125f)

/** \brief q-axis inductance [H] */
#define PI_FOC_L_Q              ((MatrixFloat)0.000125f)

/** \brief Permanent magnet flux linkage [Wb] */
#define PI_FOC_LAMBDA_PM        ((MatrixFloat)0.0014f)

/** \brief Rotor inertia [kg·m²] */
#define PI_FOC_J_ROTOR          ((MatrixFloat)2.4e-6f)

/** \brief Friction coefficient [N·m·s/rad] */
#define PI_FOC_B_FRICTION       ((MatrixFloat)1e-6f)

/** \brief Maximum phase current [A] */
#define PI_FOC_I_MAX            ((MatrixFloat)3.57f)

/** \brief DC bus voltage [V] */
#define PI_FOC_V_DC             ((MatrixFloat)17.0f)

/** \brief Maximum phase voltage (V_DC / √3) [V] */
#define PI_FOC_V_MAX            (PI_FOC_V_DC / ((MatrixFloat)1.73205080757f))
/** \} */

/** \addtogroup pi_foc_control_gains  PI controller gains
 * \{
 */
/** \brief Current loop bandwidth ωc_i = 2π×500 Hz [rad/s] */
#define PI_FOC_WC_I             ((MatrixFloat)3141.592653589793f)

/** \brief Current loop proportional gain Kp_i = L·ωc_i [V/A] */
#define PI_FOC_KP_I             (PI_FOC_L_D * PI_FOC_WC_I)

/** \brief Current loop integral gain Ki_i = R·ωc_i [V/(A·s)] */
#define PI_FOC_KI_I             (PI_FOC_R_S * PI_FOC_WC_I)

/** \brief Speed loop bandwidth ωc_ω = 2π×15 Hz [rad/s] */
#define PI_FOC_WC_SPD           ((MatrixFloat)94.24777960769379f)

/** \brief Torque constant KT = 1.5·p·λ_pm [N·m/A] */
#define PI_FOC_KT               (((MatrixFloat)1.5f) * (MatrixFloat)PI_FOC_P_POLES * PI_FOC_LAMBDA_PM)

/** \brief Speed loop proportional gain Kp_ω = J·ωc_ω / KT [A·s/rad] */
#define PI_FOC_KP_SPD           (PI_FOC_J_ROTOR * PI_FOC_WC_SPD / PI_FOC_KT)

/** \brief Speed loop integral gain Ki_ω = Kp_ω / Ti (Ti = 0.1 s) [A/rad] */
#define PI_FOC_KI_SPD           (PI_FOC_KP_SPD / ((MatrixFloat)0.1f))
/** \} */

/** \addtogroup pi_foc_limits  Anti-windup limits
 * \{
 */
/** \brief Voltage integrator clamp limit (V_max / Ki_i) */
#define PI_FOC_V_LIM            (PI_FOC_V_MAX / PI_FOC_KI_I)

/** \brief Current integrator clamp limit (I_max / Ki_ω) */
#define PI_FOC_IQ_LIM           (PI_FOC_I_MAX / PI_FOC_KI_SPD)

/** \brief 1 kHz diagnostic logging interval [s] */
#define PI_FOC_LOG_INTERVAL     ((MatrixFloat)0.001f)
/** \} */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup pi_foc_types  PI FOC state structure
 * \{
 */

/**
 * \struct PI_FOC_T
 * \brief State structure for the PI FOC controller block.
 *
 * Stores integrator states and diagnostic information for the
 * combined FOC + PI controller.
 */
typedef struct
{
    /* Integrator states */
    MatrixFloat int_id;      /**< d-axis current integrator state */
    MatrixFloat int_iq;      /**< q-axis current integrator state */
    MatrixFloat int_spd;     /**< Speed integrator state */

    /* Reference values */
    MatrixFloat iq_ref;      /**< q-axis current reference [A] (diagnostic) */
    MatrixFloat id_ref;      /**< d-axis current reference [A] (MTPA = 0) */

    /* Voltage outputs */
    MatrixFloat vd;          /**< d-axis voltage reference [V] */
    MatrixFloat vq;          /**< q-axis voltage reference [V] */

    /* Diagnostic logging (1 kHz) */
    MatrixFloat log_speed;      /**< Last logged speed [rad/s] */
    MatrixFloat log_speed_ref;  /**< Last logged speed reference [rad/s] */
    MatrixFloat log_iq_meas;    /**< Last logged measured iq [A] */
    MatrixFloat log_id_meas;    /**< Last logged measured id [A] */
    uint32_T    log_counter;    /**< Logging counter (not used, retained) */
    MatrixFloat log_next_time;  /**< Next logging time [s] */
} PI_FOC_T;

/** \} */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup pi_foc_api  Public API
 * \{
 */

/**
 * \brief  Initialise the PI FOC controller state.
 *
 * Resets all integrators and diagnostic counters to zero.
 *
 * \param[out] s  State structure to initialise (must not be NULL).
 */
extern void PI_FOC_Init(PI_FOC_T * const s);

/**
 * \brief  Apply the complete PI FOC control step.
 *
 * Performs:
 *   1. Clarke transform (abc → αβ)
 *   2. Park transform (αβ → dq)
 *   3. Speed PI controller (output iq_ref)
 *   4. Current PI controllers (d and q) with feed-forward decoupling
 *   5. Voltage saturation (hexagon limiting)
 *   6. Inverse Park transform (dq → αβ)
 *
 * \param[in,out] s         State structure (integrators, diagnostics).
 * \param[in]     omega_ref Speed reference [rad/s].
 * \param[in]     omega_m   Measured mechanical speed [rad/s].
 * \param[in]     theta_e   Electrical angle [rad].
 * \param[in]     ia        Phase A current [A].
 * \param[in]     ib        Phase B current [A].
 * \param[in]     ic        Phase C current [A].
 * \param[in]     dt        Time step [s].
 * \param[out]    v_alpha   α-axis voltage reference [V] (must not be NULL).
 * \param[out]    v_beta    β-axis voltage reference [V] (must not be NULL).
 * \param[out]    vdc       DC bus voltage [V] (pass-through, must not be NULL).
 */
extern void PI_FOC_Step(
    PI_FOC_T       * const s,
    MatrixFloat            omega_ref,
    MatrixFloat            omega_m,
    MatrixFloat            theta_e,
    MatrixFloat            ia,
    MatrixFloat            ib,
    MatrixFloat            ic,
    MatrixFloat            dt,
    MatrixFloat    * const v_alpha,
    MatrixFloat    * const v_beta,
    MatrixFloat    * const vdc);

/**
 * \brief  Reset the PI FOC controller state.
 *
 * Re-initialises all integrators and diagnostic counters.
 *
 * \param[out] s  State structure to reset (must not be NULL).
 */
extern void PI_FOC_Reset(PI_FOC_T * const s);

/**
 * \brief  Retrieve diagnostic log data.
 *
 * Provides the most recently logged values for monitoring.
 *
 * \param[in]  s         State structure.
 * \param[out] speed     Last logged speed [rad/s] (must not be NULL).
 * \param[out] speed_ref Last logged speed reference [rad/s] (must not be NULL).
 * \param[out] iq        Last logged measured iq [A] (must not be NULL).
 * \param[out] id        Last logged measured id [A] (must not be NULL).
 */
extern void PI_FOC_GetDiagnostics(
    const PI_FOC_T * const s,
    MatrixFloat    * const speed,
    MatrixFloat    * const speed_ref,
    MatrixFloat    * const iq,
    MatrixFloat    * const id);

/** \} */

#endif /* PI_FOC_H_ */