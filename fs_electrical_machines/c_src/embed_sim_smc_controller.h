/**********************************************************************************************************************
 * \file      embed_sim_smc_controller.h
 * \brief     Sliding Mode Control FOC for NANOTEC DB42S02.
 *
 * Implements pure Sliding Mode Control (SMC) with integral sliding surface:
 *   - Speed SMC:   s = e + λ·∫e + γ·∫∫e   (zero steady-state error)
 *   - Current SMC: equivalent control + switching with boundary layer
 *
 * Signal flow (complete FOC chain):
 *   [ia, ib, ic] → Clarke → [iα, iβ] → Park → [id, iq]
 *   [ω_ref_mech, ω_m, id, iq] → SMC → [vd, vq] → InvPark → [vα, vβ]
 *
 * Gain update workflow
 * --------------------
 * Primary (compile-time):
 *   smc_fmu_tuner.py patches smc_gains_config.h, then recompile and flash.
 *   SMC_Controller_Init() loads the #define values into g_smc_gains at startup.
 *   This file (embed_sim_smc_controller.h) is NEVER modified by the tuner.
 *
 * Optional (runtime, no recompile):
 *   a) UDE debugger — write g_smc_gains.ks_w etc. live while running.
 *   b) UART loader  — python smc_uart_loader.py --port COM4 --schedule gains.json
 *   c) Gain schedule — SMC_GainSchedule_Interpolate() in the speed-control task.
 *
 * Gain design
 * -----------
 *   Current loop: ωc_i = 2π×800 Hz,  Ks_i = L·ωc_i
 *   Speed loop:   λ = 2π×20 Hz,  γ = 2π×5 Hz,  Ks_w = J·λ² + T_max/φ
 *   Auto-tuned with smc_fmu_tuner.py (DE + GP Bayesian).
 *
 * Target: Infineon AURIX TriCore TC3xx, ARM Cortex-M4
 *
 * \version   2.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *
 *********************************************************************************************************************/

#ifndef SMC_CONTROLLER_H_
#define SMC_CONTROLLER_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_matrix.h"               /* MatrixFloat (= real32_T) */
#include "embed_sim_coordinate_transform.h"  /* Clarke_T, Park_T, InvPark_T */


/*********************************************************************************************************************/
/*-------------------------------------- Motor parameters (NANOTEC DB42S02) ------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Number of pole pairs */
#define SMC_P_POLES          (4U)

/** \brief Stator resistance [Ω] */
#define SMC_R_S              ((MatrixFloat)0.19f)

/** \brief d-axis inductance [H] */
#define SMC_L_D              ((MatrixFloat)0.000125f)

/** \brief q-axis inductance [H] */
#define SMC_L_Q              ((MatrixFloat)0.000125f)

/** \brief Permanent magnet flux linkage [Wb] */
#define SMC_LAMBDA_PM        ((MatrixFloat)0.0014f)

/** \brief Rotor inertia [kg·m²] */
#define SMC_J_ROTOR          ((MatrixFloat)2.4e-6f)

/** \brief Friction coefficient [N·m·s/rad] */
#define SMC_B_FRICTION       ((MatrixFloat)1e-6f)

/** \brief Maximum phase current [A] */
#define SMC_I_MAX            ((MatrixFloat)3.57f)

/** \brief DC bus voltage [V] */
#define SMC_V_DC             ((MatrixFloat)17.0f)

/** \brief Maximum phase voltage = V_DC / √3  [V] */
#define SMC_V_MAX            (SMC_V_DC / ((MatrixFloat)1.73205080757f))

/** \brief Torque constant KT = 1.5·p·λ_pm  [N·m/A] */
#define SMC_KT               ((MatrixFloat)0.0084f)


/*********************************************************************************************************************/
/*-------------------------------- Fixed sliding surface coefficients (NOT tuned) ------------------------------------*/
/*********************************************************************************************************************/
/*
 * These are derived from bandwidth targets — they are fixed design choices,
 * not free parameters.  Do NOT change without re-deriving the stability
 * conditions for the SMC surface.
 */

/** \brief Current loop bandwidth ωc_i = 2π×800 Hz  [rad/s] */
#define SMC_WC_I             ((MatrixFloat)5026.548245743669f)

/** \brief Sliding surface slope λ = 2π×20 Hz  [rad/s]
 *  Controls how fast the error trajectory is pulled onto the surface. */
#define SMC_LAMBDA_W         ((MatrixFloat)125.66370614359172f)

/** \brief Double-integral coefficient γ = 2π×5 Hz  [rad/s]
 *  Adds integral action to eliminate constant disturbances. */
#define SMC_GAMMA_W          ((MatrixFloat)31.41592653589793f)


/*********************************************************************************************************************/
/*--------------- Tunable gain defaults — see smc_gains_config.h (patched by smc_fmu_tuner.py) ----------------------*/
/*********************************************************************************************************************/
/*
 * SMC_KS_I, SMC_PHI_I, SMC_T_MAX, SMC_PHI_W, SMC_KS_W, SMC_ETA_W
 * are defined in smc_gains_config.h.
 *
 * smc_fmu_tuner.py patches ONLY that file — this header is never modified
 * by the tuner.  Recompile after patching to load new startup defaults into
 * g_smc_gains via SMC_Controller_Init().
 */
#include "smc_gains_config.h"


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \struct SMC_GainSet_T
 * \brief  Runtime-configurable SMC gains.
 *
 * This struct lives in RAM and can be written at any time without
 * recompiling.  SMC_Controller_Step() reads gains from g_smc_gains.
 *
 * Member  │ Units  │ Tunes      │ Description
 * ────────┼────────┼────────────┼──────────────────────────────────────
 * ks_w    │ N·m    │ smc_fmu_   │ Speed SMC switching gain
 * eta_w   │ —      │ tuner.py   │ Speed SMC linear damping
 * phi_w   │ rad/s  │            │ Speed boundary layer thickness
 * ks_i    │ V      │            │ Current SMC switching gain
 * phi_i   │ A      │            │ Current boundary layer thickness
 */
typedef struct
{
    MatrixFloat ks_w;   /**< Speed switching gain      [N·m]   */
    MatrixFloat eta_w;  /**< Speed linear damping      [—]     */
    MatrixFloat phi_w;  /**< Speed boundary layer      [rad/s] */
    MatrixFloat ks_i;   /**< Current switching gain    [V]     */
    MatrixFloat phi_i;  /**< Current boundary layer    [A]     */
} SMC_GainSet_T;


/**
 * \struct SMC_GainTableEntry_T
 * \brief  One row in a gain schedule table.
 *
 * Build the table with smc_fmu_tuner.py --schedule, then declare it as:
 *
 *   #include "smc_gain_schedule.h"
 *   extern const SMC_GainTableEntry_T g_smc_schedule[];
 *   extern const uint32_T             g_smc_schedule_n;
 *
 * In the speed-control task (e.g. 1 ms background):
 *   SMC_GainSchedule_Interpolate(speed_rpm,
 *                                g_smc_schedule, g_smc_schedule_n,
 *                                &g_smc_gains);
 */
typedef struct
{
    MatrixFloat    rpm;    /**< Speed operating point [RPM]  */
    SMC_GainSet_T  gains;  /**< Optimal gains at this speed  */
} SMC_GainTableEntry_T;


/**
 * \struct SMC_Input_T
 * \brief  Input to SMC_Controller_Step.
 */
typedef struct
{
    MatrixFloat omega_ref_mech;  /**< Mechanical speed reference [rad/s] */
    MatrixFloat theta_m;         /**< Mechanical angle from encoder [rad] — must be accumulating, NOT wrapped */
    MatrixFloat ia;              /**< Phase A current [A] */
    MatrixFloat ib;              /**< Phase B current [A] */
    MatrixFloat ic;              /**< Phase C current [A] */
} SMC_Input_T;


/**
 * \struct SMC_Output_T
 * \brief  Output from SMC_Controller_Step.
 */
typedef struct
{
    MatrixFloat v_alpha;   /**< α-axis voltage for SVPWM [V] */
    MatrixFloat v_beta;    /**< β-axis voltage for SVPWM [V] */
} SMC_Output_T;


/**
 * \struct SMC_Controller_T
 * \brief  Full controller state — integrators, transforms, diagnostics.
 *
 * Allocate statically:
 *   static SMC_Controller_T smc_state;
 * Initialise once:
 *   SMC_Controller_Init(&smc_state, EMBEDSIM_DT);
 */
typedef struct
{
    /* Speed SMC integrator states */
    MatrixFloat int_spd;       /**< ∫(ω_ref - ω_m) dt   [rad]   */
    MatrixFloat int2_spd;      /**< ∫∫(ω_ref - ω_m) dt  [rad·s] */

#if !defined(SMC_INTEGRATOR_EULER)
    MatrixFloat e_prev;        /**< Speed error at previous step        [rad/s] */
    MatrixFloat int_spd_prev;  /**< ∫e at previous step (Heun & Tustin) [rad]   */
#endif

    /* Speed estimation */
    MatrixFloat omega_m;       /**< Estimated mechanical speed     [rad/s] */
    MatrixFloat omega_filt;    /**< LPF-filtered speed estimate    [rad/s] */
    MatrixFloat theta_m_prev;  /**< Previous mechanical angle      [rad]   */

    /* Diagnostic references */
    MatrixFloat iq_ref;        /**< q-axis current reference [A] */
    MatrixFloat id_ref;        /**< d-axis current reference [A] — MTPA = 0 */

    /* Diagnostic voltage outputs */
    MatrixFloat vd;            /**< d-axis voltage [V] */
    MatrixFloat vq;            /**< q-axis voltage [V] */

    /* Embedded transform states (MISRA Rule 8.7 — no static locals) */
    Clarke_T   clarke_state;
    Park_T     park_state;
    InvPark_T  inv_park_state;

    /* Diagnostic log (updated at 1 kHz) */
    MatrixFloat log_speed;
    MatrixFloat log_speed_ref;
    MatrixFloat log_iq_meas;
    MatrixFloat log_id_meas;
    uint32_T    log_counter;
    MatrixFloat log_next_time;

} SMC_Controller_T;


/*********************************************************************************************************************/
/*------------------------------ Global runtime gain set (RAM — writable at runtime) ---------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Active SMC gains.
 *
 * Initialised from the SMC_KS_W / SMC_ETA_W / ... defaults in smc_gains_config.h
 * via SMC_Controller_Init().  Override at runtime via UDE, UART loader, or
 * SMC_GainSchedule_Interpolate() without recompiling.
 *
 * \note   Writes must be atomic at the application level (disable ISR or
 *         use a double-buffer scheme) if the control loop runs at high
 *         priority.  For AURIX: disable GTM ISR before writing, re-enable
 *         after.  Struct is 5 × 4 = 20 bytes — fits in a single cache line.
 */
extern SMC_GainSet_T g_smc_gains;


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Initialise controller state.  Call once before the ISR starts.
 */
extern void SMC_Controller_Init(
    SMC_Controller_T * const s,
    const MatrixFloat         dt);

/**
 * \brief  Execute one FOC step.  Call from the 20 kHz GTM ISR.
 *
 * Reads gains from g_smc_gains.  No mutex needed if gains are updated
 * only between ISR activations (e.g. in a 1 ms background task).
 */
extern void SMC_Controller_Step(
    SMC_Controller_T  * const s,
    const SMC_Input_T * const u,
    const MatrixFloat         dt,
    SMC_Output_T      * const y);

/**
 * \brief  Reset all integrators.  Call on motor stop or fault.
 */
extern void SMC_Controller_Reset(SMC_Controller_T * const s);

/**
 * \brief  Read last 1 kHz diagnostic snapshot.
 */
extern void SMC_Controller_GetDiagnostics(
    const SMC_Controller_T * const s,
    MatrixFloat            * const speed,
    MatrixFloat            * const speed_ref,
    MatrixFloat            * const iq,
    MatrixFloat            * const id);

/**
 * \brief  Copy a gain set into g_smc_gains (ISR-safe field-by-field copy).
 *
 * Preferred over direct struct assignment to avoid partial writes if the
 * compiler splits the assignment across multiple store instructions.
 * Disable the GTM ISR around this call on AURIX.
 */
extern void SMC_GainSet_SetFromSchedule(const SMC_GainSet_T * const src);

/**
 * \brief  Interpolate gain schedule and write result to *out.
 *
 * Linear interpolation between the two bracketing table entries.
 * Clamps to the first/last entry outside the table range.
 *
 * Typical call site (1 ms background task):
 * \code
 *   SMC_GainSchedule_Interpolate(speed_rpm,
 *                                g_smc_schedule,
 *                                SMC_SCHEDULE_N,
 *                                &g_smc_gains);
 * \endcode
 *
 * \param[in]  omega_rpm  Current mechanical speed [RPM].
 * \param[in]  table      Gain table (ascending RPM order).
 * \param[in]  n          Number of table entries.
 * \param[out] out        Destination gain set — pass &g_smc_gains directly.
 */
extern void SMC_GainSchedule_Interpolate(
    MatrixFloat                  omega_rpm,
    const SMC_GainTableEntry_T * const table,
    uint32_T                     n,
    SMC_GainSet_T              * const out);

#endif /* SMC_CONTROLLER_H_ */
