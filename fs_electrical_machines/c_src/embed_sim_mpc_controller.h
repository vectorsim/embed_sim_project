/**
 * @file      embed_sim_mpc_controller.h
 * @brief     Model Predictive Control FOC Controller -- NANOTEC DB42S02
 * @details   3-state receding-horizon MPC with SMO and encoder speed estimation
 * @version   2.0.1
 * @copyright Copyright (C) EmbedSim 2025
 */

#ifndef EMBED_SIM_MPC_CONTROLLER_H_
#define EMBED_SIM_MPC_CONTROLLER_H_

#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_mpc_gains.h"

/* Math constants (MISRA Rule 7.2) */
#define MPC_PI_F     ((MatrixFloat)3.14159265358979f)
#define MPC_TWO_PI_F ((MatrixFloat)6.28318530717959f)

/* Motor parameters (aligned with DB42S02) */
#define MPC_P_POLES     (4U)
#define MPC_R_S         ((MatrixFloat)0.285f)
#define MPC_L           ((MatrixFloat)0.0003675f)
#define MPC_LAMBDA_PM   ((MatrixFloat)0.0014f)
#define MPC_J_ROTOR     ((MatrixFloat)2.4e-6f)
#define MPC_B_FRICTION  ((MatrixFloat)1e-6f)
#define MPC_I_MAX       ((MatrixFloat)3.57f)
#define MPC_V_DC        ((MatrixFloat)17.0f)
#define MPC_V_MAX       (MPC_V_DC / ((MatrixFloat)1.73205080757f))
#define MPC_SVPWM_GAIN  (MPC_V_DC / ((MatrixFloat)2.0f))
#define MPC_KT          ((MatrixFloat)0.0084f)  /* 1.5 * p * λpm */

/* SMO parameters */
#define MPC_SMO_K       ((MatrixFloat)4.68f)
#define MPC_SMO_FC      ((MatrixFloat)1000.0f)

/*
 * Diagnostic log decimation factor.
 * MPC_Controller_Step() writes log fields every MPC_DIAG_STEPS ISR ticks.
 * At 20 kHz: 20 steps → 1 kHz log rate.
 * PYTHON ALIGNMENT: MPCControllerBlock.DIAG_STEPS = 20 in mpc_controller_block.py
 */
#define MPC_DIAG_STEPS  (20U)

/*
 * Fixed architectural parameters -- not tuned by CMA-ES.
 *
 * MPC_N          : Prediction horizon (loop count in MPC free-run solver).
 *                  PYTHON ALIGNMENT: MPC_N = 10 in db42s02_closed_loop_mpc_foc_20k.py
 *
 * MPC_ENC_IIR    : First-order IIR coefficient for encoder finite-difference speed.
 *                  omega_filt = (1 - MPC_ENC_IIR)*omega_filt + MPC_ENC_IIR*omega_raw
 *                  At 20 kHz: alpha=0.05 → ~160 Hz bandwidth.
 *
 * MPC_SOFTSTART_T: Soft-start ramp duration [s].  iq_limit ramps from 0 to
 *                  MPC_I_MAX over this interval at each ISR tick:
 *                  iq_limit += MPC_I_MAX * dt / MPC_SOFTSTART_T
 *                  PYTHON ALIGNMENT: MPCControllerBlock(SOFTSTART_T=0.1)
 */
#define MPC_N           (10U)
#define MPC_ENC_IIR     ((MatrixFloat)0.05f)
#define MPC_SOFTSTART_T ((MatrixFloat)0.1f)


/* MPC solver state structure */
typedef struct
{
    MatrixFloat id;      /**< D-axis current [A] */
    MatrixFloat iq;      /**< Q-axis current [A] */
    MatrixFloat omega;   /**< Mechanical speed [rad/s] */
} MPC_State_T;

/* Encoder speed estimator state */
typedef struct
{
    MatrixFloat theta_m_prev;      /**< Previous encoder angle [rad] */
    MatrixFloat theta_m_unwrapped; /**< Continuously unwrapped angle [rad] */
    MatrixFloat omega_filt;        /**< IIR-filtered speed [rad/s] */
} MPC_EncSpeed_T;

/* SMO state */
typedef struct
{
    MatrixFloat i_alpha_hat;       /**< Estimated alpha current [A] */
    MatrixFloat i_beta_hat;        /**< Estimated beta current [A] */
    MatrixFloat e_alpha_filt;      /**< LPF back-EMF alpha [V] */
    MatrixFloat e_beta_filt;       /**< LPF back-EMF beta [V] */
    MatrixFloat alpha_lpf;         /**< LPF coefficient [-] */
} MPC_SMO_T;

/* Input structure */
typedef struct
{
    MatrixFloat omega_ref_mech;    /**< Speed reference [rad/s] */
    MatrixFloat theta_m;           /**< Encoder angle [rad] */
    MatrixFloat ia;                /**< Phase A current [A] */
    MatrixFloat ib;                /**< Phase B current [A] */
    MatrixFloat ic;                /**< Phase C current [A] */
} MPC_Input_T;

/* Output structure */
typedef struct
{
    MatrixFloat v_alpha;           /**< Alpha voltage (normalised) */
    MatrixFloat v_beta;            /**< Beta voltage (normalised) */
} MPC_Output_T;

/* Complete controller state */
typedef struct
{
    MPC_EncSpeed_T enc;            /**< Encoder speed estimator */
    MPC_SMO_T      smo;            /**< Sliding Mode Observer */
    MatrixFloat    v_alpha_prev;   /**< Previous alpha voltage [V] */
    MatrixFloat    v_beta_prev;    /**< Previous beta voltage [V] */
    MatrixFloat    iq_limit;       /**< Soft-start current limit [A] */
    MatrixFloat    speed_err_integral; /**< Speed error integral [rad] */
    MatrixFloat    log_speed_ref;  /**< Speed reference log [RPM] */
    MatrixFloat    log_speed;      /**< Speed log [RPM] */
    MatrixFloat    log_id;         /**< Id current log [A] */
    MatrixFloat    log_iq;         /**< Iq current log [A] */
    MatrixFloat    log_vd;         /**< Vd voltage log [V] */
    MatrixFloat    log_vq;         /**< Vq voltage log [V] */
    unsigned int   log_counter;    /**< Log counter */
    MatrixFloat    log_next_time;  /**< Next log time [s] */
    Clarke_T       clarke_state;   /**< Clarke transform */
    Park_T         park_state;     /**< Park transform */
    Park_T         park_emf_state; /**< Park for EMF */
    InvPark_T      inv_park_state; /**< Inverse Park */
} MPC_Controller_T;

/* Function prototypes */
void MPC_Controller_Init(MPC_Controller_T* s, const MatrixFloat dt);
void MPC_Controller_Step(MPC_Controller_T* s, const MPC_Input_T* u,
                         const MatrixFloat dt, MPC_Output_T* y);
void MPC_Controller_Reset(MPC_Controller_T* s);
void MPC_Controller_GetDiagnostics(const MPC_Controller_T* s,
                                   MatrixFloat* speed_ref_rpm,
                                   MatrixFloat* speed_rpm,
                                   MatrixFloat* id_meas,
                                   MatrixFloat* iq_meas,
                                   MatrixFloat* vd,
                                   MatrixFloat* vq,
                                   MatrixFloat* iq_limit);

#endif /* EMBED_SIM_MPC_CONTROLLER_H_ */
