/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.h
 * \brief     Differential Flatness FOC Controller — NANOTEC DB42S02 / AURIX TC3xx
 *
 * Implements complete FOC control chain using differential flatness:
 *   [ia, ib, ic] → Clarke → [iα, iβ]
 *   → Park(θ_e) → [id, iq]
 *   → Speed P-loop → iq_ref
 *   → DFC voltage law → [vd, vq]
 *   → InvPark(θ_e) → [vα, vβ] → SVPWM
 *
 * SpeedFusion complementary filter:
 *   - Encoder provides exact θ_e = p·θ_m
 *   - ω_e = (1-α)·ω_enc + α·ω_smo
 *   - α transitions from 0→1 across [50, 250] rad/s via sigmoid
 *   - Adaptive encoder IIR: heavier smoothing at low speed
 *
 * Sliding Mode Observer (αβ frame):
 *   - Estimates back-EMF and ω̂_e for SpeedFusion
 *   - Classical SMO with sigmoid switching (smooth, no chattering)
 *   - 2nd-order LPF on back-EMF (τ_e = 2 ms)
 *
 * Flatness voltage law:
 *   vd = R·id_ref + Ld·d(id_ref)/dt − ω_e·Lq·iq_ref + Kp_id·(id_ref−id)
 *   vq = R·iq_ref + Lq·d(iq_ref)/dt + ω_e·Ld·id_ref + ω_e·λ_pm + Kp_iq·(iq_ref−iq)
 *
 * MISRA C:2012 compliance
 * -----------------------
 *   Rule 8.7  : all state in caller-supplied structs (no static locals)
 *   Rule 15.5 : single exit per function (except early NULL guard returns)
 *   Rule 15.7 : mandatory else clauses on all if-else chains
 *   Rule 21.8 : no memset on float-bearing structs (explicit Init calls)
 *   All float literals carry the f suffix — no implicit double promotion.
 *
 * \version   1.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_gains.h"   /* Contains DFC_GainSet_T definition */


/*********************************************************************************************************************/
/*-------------------------------------- Motor parameters (NANOTEC DB42S02) ------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Number of pole pairs */
#define DFC_P_POLES          (4U)

/** \brief Stator resistance [Ω] */
#define DFC_R_S              ((MatrixFloat)0.285f)

/** \brief d-axis inductance [H] */
#define DFC_L_D              ((MatrixFloat)0.0003675f)

/** \brief q-axis inductance [H] */
#define DFC_L_Q              ((MatrixFloat)0.0003675f)

/** \brief Permanent magnet flux linkage [Wb] */
#define DFC_LAMBDA_PM        ((MatrixFloat)0.0014f)

/** \brief Maximum phase current [A] */
#define DFC_I_MAX            ((MatrixFloat)3.57f)

/** \brief DC bus voltage [V] */
#define DFC_V_DC             ((MatrixFloat)17.0f)

/** \brief Maximum phase voltage = V_DC / √3  [V] */
#define DFC_V_MAX            (DFC_V_DC / ((MatrixFloat)1.73205080757f))


/*********************************************************************************************************************/
/*-------------------------------------- SpeedFusion parameters ------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief SpeedFusion lower threshold [rad/s] — α=0 below this speed */
#define DFC_FUSION_OMEGA_LO  ((MatrixFloat)50.0f)

/** \brief SpeedFusion upper threshold [rad/s] — α=1 above this speed */
#define DFC_FUSION_OMEGA_HI  ((MatrixFloat)250.0f)

/** \brief Sigmoid steepness factor — dimensionless */
#define DFC_FUSION_GAMMA     ((MatrixFloat)2.0f)

/** \brief Encoder IIR coefficient at low speed (α=0) — dimensionless */
#define DFC_FUSION_IIR_LO    ((MatrixFloat)0.05f)

/** \brief Encoder IIR coefficient at high speed (α=1) — dimensionless */
#define DFC_FUSION_IIR_HI    ((MatrixFloat)0.30f)


/*********************************************************************************************************************/
/*-------------------------------------- SMO parameters --------------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief SMO switching gain [V] — must exceed max back-EMF = ωe_max·λ_pm = 1256·0.0014 = 1.76 V */
#define DFC_SMO_K            ((MatrixFloat)6.0f)

/** \brief SMO back-EMF LPF time constant [s] — fc = 1/(2π·τ) ≈ 80 Hz */
#define DFC_SMO_TAU_E        ((MatrixFloat)0.002f)

/** \brief Sigmoid width for smooth switching [1/A] — 5.0 gives soft transition */
#define DFC_SMO_SIGMOID_W    ((MatrixFloat)5.0f)

/** \brief SMO warmup steps — ignore ω̂_e for first N steps until BEMF converges */
#define DFC_SMO_WARMUP_STEPS (400U)

/** \brief Current derivative LPF time constant [s] — τ = 1 ms */
#define DFC_DIQ_TAU          ((MatrixFloat)0.001f)

/** \brief Logging interval [s] — 1 kHz */
#define DFC_LOG_INTERVAL     ((MatrixFloat)0.001f)


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \struct DFC_SpeedFusion_T
 * \brief  Speed-dependent complementary filter state.
 */
typedef struct
{
    MatrixFloat theta_m_prev;      /**< Previous mechanical angle for differentiation [rad] */
    MatrixFloat omega_enc_filt;    /**< Filtered encoder mechanical speed [rad/s] */
    MatrixFloat omega_e_prev;      /**< Previous fused electrical speed [rad/s] */
    MatrixFloat alpha;             /**< Current fusion weight (0=encoder, 1=SMO) — diagnostic */
    MatrixFloat omega_enc_mech;    /**< Raw encoder mechanical speed — diagnostic */
} DFC_SpeedFusion_T;


/**
 * \struct DFC_SMO_T
 * \brief  Sliding Mode Observer state (αβ frame).
 */
typedef struct
{
    MatrixFloat i_hat_alpha;       /**< Estimated α-axis current [A] */
    MatrixFloat i_hat_beta;        /**< Estimated β-axis current [A] */
    MatrixFloat e_hat_alpha;       /**< Filtered back-EMF α [V] */
    MatrixFloat e_hat_beta;        /**< Filtered back-EMF β [V] */
    MatrixFloat theta_e_hat;       /**< Estimated electrical angle [rad] — diagnostic only */
    MatrixFloat omega_e_hat;       /**< Estimated electrical speed [rad/s] */
    MatrixFloat theta_e_prev;      /**< Previous angle for speed extraction [rad] */
} DFC_SMO_T;


/**
 * \struct DFC_Input_T
 * \brief  Input to DFC_Controller_Step (identical to SMC_Input_T).
 */
typedef struct
{
    MatrixFloat omega_ref_mech;  /**< Mechanical speed reference [rad/s] */
    MatrixFloat theta_m;         /**< Mechanical angle from encoder [rad] — accumulating, NOT wrapped */
    MatrixFloat ia;              /**< Phase A current [A] */
    MatrixFloat ib;              /**< Phase B current [A] */
    MatrixFloat ic;              /**< Phase C current [A] */
} DFC_Input_T;


/**
 * \struct DFC_Output_T
 * \brief  Output from DFC_Controller_Step.
 *
 * These fields carry physical voltages [V].
 * The SVPWM block expects normalised [-1,+1] values.
 * Caller must divide by DFC_SVPWM_GAIN = V_DC/2 before SVPWM.
 */
typedef struct
{
    MatrixFloat v_alpha;   /**< α-axis voltage [V] */
    MatrixFloat v_beta;    /**< β-axis voltage [V] */
} DFC_Output_T;


/**
 * \struct DFC_State_T
 * \brief  Full Differential Flatness Controller state.
 *
 * Allocate statically:
 *   static DFC_State_T dfc_state;
 * Initialise once:
 *   DFC_Controller_Init(&dfc_state, EMBEDSIM_DT);
 */
typedef struct
{
    /* SpeedFusion state */
    DFC_SpeedFusion_T fusion;

    /* SMO state */
    DFC_SMO_T smo;

    /* Delayed voltages for SMO (z⁻¹) */
    MatrixFloat v_alpha_prev;      /**< α voltage from previous step [V] */
    MatrixFloat v_beta_prev;       /**< β voltage from previous step [V] */

    /* Reference trajectory */
    MatrixFloat theta_ref;         /**< Integrated reference angle [rad] */
    MatrixFloat iq_ref_prev;       /**< Previous iq_ref for derivative [A] */
    MatrixFloat diq_filt;          /**< LPF-filtered diq_ref/dt [A/s] */

    /* SMO warmup counter */
    uint32_T smo_warmup_cnt;       /**< Steps since start — gate ω̂_e until threshold */

    /* Coordinate transforms — MISRA Rule 8.7 compliance (no static locals) */
    Clarke_T   clarke_state;
    Park_T     park_state;
    InvPark_T  inv_park_state;

    /* Diagnostic logging (updated at 1 kHz) */
    MatrixFloat log_speed_ref;     /**< Speed reference at log time [RPM] */
    MatrixFloat log_iq_ref;        /**< iq reference at log time [A] */
    MatrixFloat log_id;            /**< Measured id at log time [A] */
    MatrixFloat log_iq;            /**< Measured iq at log time [A] */
    MatrixFloat log_alpha;         /**< Fusion weight at log time */
    MatrixFloat log_omega_e;       /**< Fused electrical speed at log time [rad/s] */
    uint32_T    log_counter;       /**< Step counter for logging */
    MatrixFloat log_next_time;     /**< Next log time [s] */

} DFC_State_T;


/*********************************************************************************************************************/
/*-------------------------------------- Global runtime gain set -----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Active DFC gains.
 *
 * Initialised from DFC_KP_SPEED / DFC_KP_ID / DFC_KP_IQ defaults
 * via DFC_Controller_Init().  Override at runtime without recompiling.
 */
extern DFC_GainSet_T g_dfc_gains;


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Initialise controller state.  Call once before the ISR starts.
 */
extern void DFC_Controller_Init(
    DFC_State_T * const s,
    const MatrixFloat   dt);

/**
 * \brief  Execute one FOC step.  Call from the 20 kHz GTM ISR.
 */
extern void DFC_Controller_Step(
    DFC_State_T        * const s,
    const DFC_Input_T  * const u,
    const MatrixFloat           dt,
    DFC_Output_T       * const y);

/**
 * \brief  Reset all integrators and state.  Call on motor stop or fault.
 */
extern void DFC_Controller_Reset(DFC_State_T * const s);

/**
 * \brief  Read diagnostic snapshot.
 */
extern void DFC_Controller_GetDiagnostics(
    const DFC_State_T * const s,
    MatrixFloat       * const speed_ref_rpm,
    MatrixFloat       * const iq_ref,
    MatrixFloat       * const id,
    MatrixFloat       * const iq,
    MatrixFloat       * const alpha,
    MatrixFloat       * const omega_e);

/**
 * \brief  Copy a gain set into g_dfc_gains (ISR-safe).
 */
extern void DFC_GainSet_SetFromSchedule(const DFC_GainSet_T * const src);

#endif /* EMBED_SIM_DFC_CONTROLLER_H_ */