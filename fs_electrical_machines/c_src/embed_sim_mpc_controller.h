/**
 **********************************************************************************************************************
 * \file      embed_sim_mpc_controller.h
 * \brief     3-state analytical MPC for PMSM speed and current control.
 *            NANOTEC DB42S02  |  AURIX TC3xx  |  20 kHz
 *
 * Architecture
 * ────────────
 * State  : x = [id, iq, omega_m]
 * Input  : u = [vd, vq]
 * Output : [v_alpha, v_beta]
 *
 * Cost function (minimised analytically at each 50 µs step):
 *   J = Σ_{k=1}^{N} [ Q_id · id_k²
 *                    + Q_omega · (omega_k − omega_ref)²
 *                    + R_vd · vd²  +  R_vq · vq² ]
 *
 * Closed-form solution (O(N), no iteration):
 *   vd_mpc = Q_id   · Σ bk·(0      − id_free_k) / (Q_id   · Σ bk² + R_vd)
 *   vq_mpc = Q_omega· Σ ek·(ω_ref  − ω_free_k ) / (Q_omega· Σ ek² + R_vq)
 *
 *   bk  = accumulated iq step-response to unit vq (constant over horizon)
 *   ek  = Σ_{j≤k} (dt/J)·KT·bj  (omega response)
 *
 * BEMF feedforward (exact cancellation):
 *   vd = vd_mpc + ed_hat
 *   vq = vq_mpc + eq_hat
 *
 * Scalar type convention
 * ──────────────────────
 * All floating-point scalars use MatrixFloat (≡ real32_T) from
 * embed_sim_matrix.h.  No bare C99 'float' appears in any public API.
 * Integer loop counters and status fields use uint32_T from
 * embed_sim_sys_types.h.  Boolean flags use boolean_T.
 *
 * MISRA C:2012 compliance notes
 * ──────────────────────────────
 *  - All functions have external linkage declared here.
 *  - No dynamic memory allocation.
 *  - No recursion.
 *  - State held in a single MPC_Controller_T struct (caller owns).
 *  - MatrixFloat used throughout (AURIX TriCore has hardware FPU).
 *
 * \version   2.0.0
 * \copyright Copyright (C) EmbedSim 2025
 **********************************************************************************************************************/

#ifndef EMBED_SIM_MPC_CONTROLLER_H
#define EMBED_SIM_MPC_CONTROLLER_H

#include "embed_sim_sys_types.h"   /* uint32_T, boolean_T          */
#include "embed_sim_matrix.h"      /* MatrixFloat (≡ real32_T)     */


/**********************************************************************************************************************/
/*------------------------------------------------------Macros--------------------------------------------------------*/
/**********************************************************************************************************************/

/** \addtogroup mpc_motor_params  Motor parameters (DB42S02)
 * \{
 */
#define MPC_P_POLES    (4U)
#define MPC_R_S        ((MatrixFloat)0.19F)
#define MPC_L          ((MatrixFloat)0.000125F)
#define MPC_LAMBDA_PM  ((MatrixFloat)0.0014F)
#define MPC_J          ((MatrixFloat)2.4e-6F)
#define MPC_B          ((MatrixFloat)1.0e-6F)
#define MPC_I_MAX      ((MatrixFloat)3.57F)
#define MPC_V_DC       ((MatrixFloat)17.0F)
#define MPC_V_MAX      ((MatrixFloat)9.8150F)   /**< Vdc / sqrt(3) [V] */
#define MPC_KT         ((MatrixFloat)0.0084F)   /**< 1.5 · P · lambda_pm [N·m/A] */
/** \} */

/** \addtogroup mpc_solver_params  Solver parameters
 * \{
 */
#define MPC_N          (10U)                      /**< Prediction horizon (steps)  */
#define MPC_DT         ((MatrixFloat)50.0e-6F)   /**< Sample period [s]           */
#define MPC_Q_ID       ((MatrixFloat)10.0F)      /**< id cost weight              */
#define MPC_Q_OMEGA    ((MatrixFloat)500.0F)     /**< speed cost weight           */
#define MPC_R_VD       ((MatrixFloat)0.01F)      /**< vd effort weight            */
#define MPC_R_VQ       ((MatrixFloat)0.01F)      /**< vq effort weight            */
/** \} */

/** \addtogroup mpc_smo_params  SMO parameters
 * \{
 */
#define MPC_SMO_K      ((MatrixFloat)4.68F)      /**< Switching gain [V]          */
#define MPC_SMO_FC     ((MatrixFloat)1000.0F)    /**< LPF cut-off [Hz]            */
/** \} */

/** \addtogroup mpc_integral_params  Speed integral correction
 * \{
 */
#define MPC_KI_V       ((MatrixFloat)0.03F)      /**< Integral gain [V·s/rad]     */
/** \} */

/** \addtogroup mpc_softstart_params  Soft-start
 * \{
 */
#define MPC_SOFTSTART_T ((MatrixFloat)0.1F)      /**< Ramp time [s]               */
/** \} */

/** \addtogroup mpc_internal_constants  Internal numeric constants
 * \{
 */
#define MPC_ZERO_F     ((MatrixFloat)0.0F)
#define MPC_ONE_F      ((MatrixFloat)1.0F)
#define MPC_TWO_F      ((MatrixFloat)2.0F)
#define MPC_PI_F       ((MatrixFloat)3.14159265F)
#define MPC_DENOM_MIN  ((MatrixFloat)1.0e-30F)   /**< Guard against divide-by-zero */
/** \} */


/**********************************************************************************************************************/
/*------------------------------------------------Data Structures-----------------------------------------------------*/
/**********************************************************************************************************************/

/**
 * \struct MPC_Controller_T
 * \brief  Persistent MPC controller state.
 *         Zero-initialise before first call via MPC_Controller_Init().
 *
 * All scalar fields are MatrixFloat.  Boolean flags are boolean_T.
 * No bare 'float' appears in this struct.
 */
typedef struct
{
    /* SMO observer state (αβ frame) */
    MatrixFloat  i_alpha_hat;       /**< Estimated α-axis stator current [A]        */
    MatrixFloat  i_beta_hat;        /**< Estimated β-axis stator current [A]        */
    MatrixFloat  e_alpha_filt;      /**< LPF-filtered α-axis back-EMF estimate [V]  */
    MatrixFloat  e_beta_filt;       /**< LPF-filtered β-axis back-EMF estimate [V]  */
    MatrixFloat  v_alpha_prev;      /**< Applied α-axis voltage, previous step [V]  */
    MatrixFloat  v_beta_prev;       /**< Applied β-axis voltage, previous step [V]  */

    /* Speed estimator */
    MatrixFloat  omega_filt;        /**< IIR-filtered mechanical speed [rad/s]      */
    MatrixFloat  last_theta_m;      /**< Mechanical angle at previous step [rad]    */

    /* Soft-start */
    MatrixFloat  iq_limit;          /**< Rising iq current limit [A]                */

    /* Speed error integral (steady-state offset correction) */
    MatrixFloat  speed_err_integral; /**< Accumulated speed error integral [rad]    */

} MPC_Controller_T;


/**
 * \struct MPC_Input_T
 * \brief  Input bus: one call per 50 µs ISR tick.
 */
typedef struct
{
    MatrixFloat  omega_ref_mech; /**< Speed reference [rad/s mechanical]  */
    MatrixFloat  theta_m;        /**< Rotor mechanical angle [rad]        */
    MatrixFloat  ia;             /**< Phase current A [A]                 */
    MatrixFloat  ib;             /**< Phase current B [A]                 */
    MatrixFloat  ic;             /**< Phase current C [A]                 */
} MPC_Input_T;


/**
 * \struct MPC_Output_T
 * \brief  Output bus.
 */
typedef struct
{
    MatrixFloat  v_alpha;  /**< α-axis stator voltage command [V] */
    MatrixFloat  v_beta;   /**< β-axis stator voltage command [V] */
} MPC_Output_T;


/**********************************************************************************************************************/
/*------------------------------------------------Function Prototypes-------------------------------------------------*/
/**********************************************************************************************************************/

/**
 * \brief  Initialise (zero-clear) controller state.
 * \param  st  Pointer to state struct allocated by caller (must not be NULL).
 */
extern void MPC_Controller_Init(MPC_Controller_T *st);

/**
 * \brief  Run one 50 µs MPC step.
 * \param  st   Persistent state (updated in-place, must not be NULL).
 * \param  in   Sensor inputs for this step (must not be NULL).
 * \param  out  Voltage commands written here (must not be NULL).
 */
extern void MPC_Controller_Step(MPC_Controller_T       *st,
                                const MPC_Input_T      *in,
                                MPC_Output_T           *out);


#endif /* EMBED_SIM_MPC_CONTROLLER_H */
