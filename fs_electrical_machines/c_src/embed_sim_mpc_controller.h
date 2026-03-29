/**
 * @file    embed_sim_mpc_controller.h
 * @brief   3-state analytical MPC for PMSM speed and current control.
 *          NANOTEC DB42S02  |  AURIX TC3xx  |  20 kHz
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
 *   vd_mpc = Q_id   · Σ bk·(0       − id_free_k) / (Q_id   · Σ bk² + R_vd)
 *   vq_mpc = Q_omega· Σ ek·(ω_ref   − ω_free_k ) / (Q_omega· Σ ek² + R_vq)
 *
 *   bk  = accumulated iq step-response to unit vq (constant over horizon)
 *   ek  = Σ_{j≤k} (dt/J)·KT·bj  (omega response)
 *
 * BEMF feedforward (exact cancellation):
 *   vd = vd_mpc + ed_hat
 *   vq = vq_mpc + eq_hat
 *
 * MISRA C:2012 compliance notes
 * ──────────────────────────────
 *  - All functions have external linkage declared here.
 *  - No dynamic memory allocation.
 *  - No recursion.
 *  - State held in a single MPC_Controller_T struct (caller owns).
 *  - float used throughout (AURIX TriCore has hardware FPU).
 */

#ifndef EMBED_SIM_MPC_CONTROLLER_H
#define EMBED_SIM_MPC_CONTROLLER_H

#include <stdint.h>   /* uint32_t */

/* ── Motor parameters (DB42S02) ─────────────────────────────────────────── */
#define MPC_P_POLES    (4)
#define MPC_R_S        (0.19F)
#define MPC_L          (0.000125F)
#define MPC_LAMBDA_PM  (0.0014F)
#define MPC_J          (2.4e-6F)
#define MPC_B          (1.0e-6F)
#define MPC_I_MAX      (3.57F)
#define MPC_V_DC       (17.0F)
#define MPC_V_MAX      (9.8150F)   /* Vdc / sqrt(3) */
#define MPC_KT         (0.0084F)   /* 1.5 · P · lambda_pm */

/* ── Solver parameters ──────────────────────────────────────────────────── */
#define MPC_N          (10)        /* Prediction horizon (steps) */
#define MPC_DT         (50.0e-6F) /* Sample period [s] */
#define MPC_Q_ID       (10.0F)
#define MPC_Q_OMEGA    (500.0F)
#define MPC_R_VD       (0.01F)
#define MPC_R_VQ       (0.01F)

/* ── SMO parameters ──────────────────────────────────────────────────────── */
#define MPC_SMO_K      (4.68F)     /* Switching gain [V] */
#define MPC_SMO_FC     (1000.0F)   /* LPF cut-off [Hz] */

/* ── Speed integral correction ───────────────────────────────────────────── */
#define MPC_KI_V       (0.03F)     /* Integral gain [V/(rad/s)] */

/* ── Softstart ───────────────────────────────────────────────────────────── */
#define MPC_SOFTSTART_T (0.1F)     /* Ramp time [s] */


/**
 * @brief  Persistent MPC controller state.
 *         Zero-initialise before first call.
 */
typedef struct
{
    /* SMO observer state (αβ frame) */
    float  i_alpha_hat;
    float  i_beta_hat;
    float  e_alpha_filt;
    float  e_beta_filt;
    float  v_alpha_prev;
    float  v_beta_prev;

    /* Speed estimator */
    float  omega_filt;
    float  last_theta_m;

    /* Soft-start */
    float  iq_limit;

    /* Speed error integral */
    float  speed_err_integral;

} MPC_Controller_T;


/**
 * @brief  Input bus: one call per 50 µs ISR tick.
 */
typedef struct
{
    float  omega_ref_mech; /**< Speed reference [rad/s mechanical] */
    float  theta_m;        /**< Rotor mechanical angle [rad]        */
    float  ia;             /**< Phase current A [A]                 */
    float  ib;             /**< Phase current B [A]                 */
    float  ic;             /**< Phase current C [A]                 */
} MPC_Input_T;


/**
 * @brief  Output bus.
 */
typedef struct
{
    float  v_alpha;  /**< α-axis stator voltage command [V] */
    float  v_beta;   /**< β-axis stator voltage command [V] */
} MPC_Output_T;


/* ── Public API ─────────────────────────────────────────────────────────── */

/**
 * @brief  Initialise (zero-clear) controller state.
 * @param  st  Pointer to state struct allocated by caller.
 */
extern void MPC_Controller_Init(MPC_Controller_T *st);

/**
 * @brief  Run one 50 µs MPC step.
 * @param  st   Persistent state (updated in-place).
 * @param  in   Sensor inputs for this step.
 * @param  out  Voltage commands written here.
 */
extern void MPC_Controller_Step(MPC_Controller_T *st,
                               const MPC_Input_T  *in,
                               MPC_Output_T        *out);


#endif /* EMBED_SIM_MPC_CONTROLLER_H */
