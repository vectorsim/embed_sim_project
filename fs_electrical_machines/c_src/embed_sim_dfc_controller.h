/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.h
 * \brief     Differential Flatness FOC Controller
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_gains.h"

/*********************************************************************************************************************/
/*-------------------------------------- Motor parameters (NANOTEC DB42S02) -----------------------------------------*/
/*********************************************************************************************************************/

/** \brief Number of pole pairs */
#define DFC_P_POLES          (4U)

/** \brief Stator resistance [Ohm] */
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

/** \brief Maximum phase voltage = V_DC / sqrt(3) [V] */
#define DFC_V_MAX            (DFC_V_DC / ((MatrixFloat)1.73205080757f))


/*********************************************************************************************************************/
/*-------------------------------------- SpeedFusion parameters ------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief SpeedFusion lower threshold [rad/s] - alpha=0 below this speed */
#define DFC_FUSION_OMEGA_LO  ((MatrixFloat)50.0f)

/** \brief SpeedFusion upper threshold [rad/s] - alpha=1 above this speed */
#define DFC_FUSION_OMEGA_HI  ((MatrixFloat)250.0f)

/** \brief Encoder IIR coefficient at low speed (alpha=0) */
#define DFC_FUSION_IIR_LO    ((MatrixFloat)0.05f)

/** \brief Encoder IIR coefficient at high speed (alpha=1) */
#define DFC_FUSION_IIR_HI    ((MatrixFloat)0.30f)


/*********************************************************************************************************************/
/*-------------------------------------- SMO parameters --------------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief SMO switching gain [V] - must exceed max back-EMF (1256 * 0.0014 = 1.76V) */
#define DFC_SMO_K            ((MatrixFloat)6.0f)

/** \brief SMO back-EMF LPF time constant [s] - fc ~80 Hz */
#define DFC_SMO_TAU_E        ((MatrixFloat)0.002f)

/** \brief SMO warmup steps - ignore omega_e until BEMF converges */
#define DFC_SMO_WARMUP_STEPS (400U)

/** \brief Current derivative LPF time constant [s] */
#define DFC_DIQ_TAU          ((MatrixFloat)0.001f)

/** \brief Logging interval [s] - 1 kHz */
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
    MatrixFloat theta_m_prev;      /**< Previous mechanical angle [rad] */
    MatrixFloat omega_enc_filt;    /**< Filtered encoder mechanical speed [rad/s] */
    MatrixFloat omega_e_prev;      /**< Previous fused electrical speed [rad/s] */
    MatrixFloat alpha;             /**< Current fusion weight (0=encoder, 1=SMO) */
    MatrixFloat omega_enc_mech;    /**< Raw encoder mechanical speed [rad/s] */
} DFC_SpeedFusion_T;

/**
 * \struct DFC_SMO_T
 * \brief  Sliding Mode Observer state (alphabeta frame).
 */
typedef struct
{
    MatrixFloat i_hat_alpha;       /**< Estimated alpha-axis current [A] */
    MatrixFloat i_hat_beta;        /**< Estimated beta-axis current [A] */
    MatrixFloat e_hat_alpha;       /**< Filtered back-EMF alpha [V] */
    MatrixFloat e_hat_beta;        /**< Filtered back-EMF beta [V] */
    MatrixFloat theta_e_hat;       /**< Estimated electrical angle [rad] */
    MatrixFloat omega_e_hat;       /**< Estimated electrical speed [rad/s] */
    MatrixFloat theta_e_prev;      /**< Previous angle for speed extraction [rad] */
} DFC_SMO_T;

/**
 * \struct DFC_Input_T
 * \brief  Input to DFC_Controller_Step.
 */
typedef struct
{
    MatrixFloat omega_ref_mech;  /**< Mechanical speed reference [rad/s] */
    MatrixFloat theta_m;         /**< Mechanical angle from encoder [rad] */
    MatrixFloat ia;              /**< Phase A current [A] */
    MatrixFloat ib;              /**< Phase B current [A] */
    MatrixFloat ic;              /**< Phase C current [A] */
} DFC_Input_T;

/**
 * \struct DFC_Output_T
 * \brief  Output from DFC_Controller_Step.
 */
typedef struct
{
    MatrixFloat v_alpha;   /**< Alpha-axis voltage [V] */
    MatrixFloat v_beta;    /**< Beta-axis voltage [V] */
} DFC_Output_T;

/**
 * \struct DFC_State_T
 * \brief  Full Differential Flatness Controller state.
 */
typedef struct
{
    /* Speed fusion state */
    DFC_SpeedFusion_T fusion;

    /* SMO state */
    DFC_SMO_T smo;

    /* Delayed voltages for SMO (z-1) */
    MatrixFloat v_alpha_prev;
    MatrixFloat v_beta_prev;

    /* Reference trajectory */
    MatrixFloat iq_ref_prev;       /**< Previous iq_ref for derivative [A] */
    MatrixFloat diq_filt;          /**< LPF-filtered diq_ref/dt [A/s] */

    /* SMO warmup counter */
    uint32_T smo_warmup_cnt;

    /* Coordinate transforms */
    Clarke_T   clarke_state;
    Park_T     park_state;
    InvPark_T  inv_park_state;

    /* Diagnostic logging */
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
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Initialise controller state. Call once before the ISR starts.
 */
extern void DFC_Controller_Init(DFC_State_T * const s, const MatrixFloat dt);

/**
 * \brief  Execute one FOC step. Call from the 20 kHz GTM ISR.
 */
extern void DFC_Controller_Step(
    DFC_State_T        * const s,
    const DFC_Input_T  * const u,
    const MatrixFloat           dt,
    DFC_Output_T       * const y);

/**
 * \brief  Reset all integrators and state. Call on motor stop or fault.
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

#endif /* EMBED_SIM_DFC_CONTROLLER_H_ */
