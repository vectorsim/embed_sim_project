/**********************************************************************************************************************
 * \file      embed_sim_smc_controller.h
 * \brief     Sliding Mode Control FOC for NANOTEC DB42S02.
 *
 * Implements pure Sliding Mode Control (SMC) with integral sliding surface:
 *   - Speed SMC:    s = e + λ·∫e   (first-order, γ=0 — double integral disabled)
 *   - Current SMC:  classical FOC equivalent control + switching
 *
 * Encoder fault detection:
 *   - Validates encoder angle steps against physical limits
 *   - Detects glitches and excessive jumps
 *   - Automatically switches to sensorless mode if encoder fails
 *   - Blends encoder and SMO estimates based on confidence
 *
 * Signal flow (complete FOC chain):
 *   [ia, ib, ic] → Clarke → [iα, iβ] → Park(θ_e) → [id, iq]
 *   θ_e = p·θ_m  (exact from encoder or SMO blended)
 *   ω_m = Δθ_m/dt + IIR  (encoder finite-difference or SMO)
 *   [ω_ref, ω_m, id, iq] → SMC → [vd, vq] → InvPark(θ_e) → [vα, vβ]
 *
 * Equivalent control (no SMO in current loop path):
 *   ed_hat =  R·id_meas - ωe·Lq·iq_meas
 *   eq_hat =  R·iq_meas + ωe·(Ld·id_meas + λpm)
 *   SMO still runs for diagnostics / sensorless backup, NOT used in vd/vq.
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
 * \version   2.2.0
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
#define SMC_R_S              ((MatrixFloat)0.285f)   /* DELTA: R_phase = R_line*3/2 = 0.19*1.5 */

/** \brief d-axis inductance [H] */
#define SMC_L_D              ((MatrixFloat)0.0003675f) /* DELTA: L_phase = L_line*3/2 = 0.245e-3*1.5 */

/** \brief q-axis inductance [H] */
#define SMC_L_Q              ((MatrixFloat)0.0003675f) /* DELTA: L_phase = L_line*3/2 */

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

/** \brief SVPWM chain gain = V_DC / 2  [—]
 *
 *  The SVPWM block (SVPWMPackBlock → SVPWMBlock) amplifies a normalised
 *  reference in [-1, +1] by V_DC/2 before applying it to the inverter.
 *  All controller voltages (vd_eq, vq_eq, switching terms) are computed in
 *  physical units [V] inside SMC_Controller_Step().  The final v_alpha /
 *  v_beta outputs are divided by this gain so the SVPWM block on AURIX
 *  sees the correct normalised reference.
 *
 *  Matched to Python:
 *    SMC_SVPWM_GAIN = V_DC / 2.0   (smc_controller_block.py, line 229)
 *    vd_eq = vd_eq_physical / SMC_SVPWM_GAIN   (line 599)
 *  At 17 V bus:  gain = 8.5, so 0.625 V physical → 0.0735 normalised. */
#define SMC_SVPWM_GAIN       (SMC_V_DC / (MatrixFloat)2.0f)


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

/** \brief Sliding surface slope λ = 2π×10 Hz  [rad/s]
 *  First-order integral surface: s = e + λ·∫e.
 *  Double-integral disabled (GAMMA_W=0) — was causing phase lag and
 *  instability on this low-inertia motor. */
#define SMC_LAMBDA_W         ((MatrixFloat)62.83185307179586f)

/** \brief Double-integral coefficient γ = 0  [—]
 *  Disabled.  Set to non-zero only if steady-state load rejection via
 *  the integral term alone is insufficient. */
#define SMC_GAMMA_W          ((MatrixFloat)0.0f)


/*********************************************************************************************************************/
/*------------------------------------ Sliding Mode Observer parameters ----------------------------------------------*/
/*********************************************************************************************************************/

/** \brief SMO switching gain k [V]
 *  k > |e_BEMF_max| = ωe_max·λpm = 1256·0.0014 = 1.76 V → 2.5 V gives 1.4× margin.
 *  Must NOT be set to V_MAX (14.7 V) — that injects ±14.7 V per step into the observer,
 *  ΔI = 14.7·50e-6/125e-6 = 5.9 A/step → observer diverges → id runaway.
 *  Increased from 0.75V to 2.5V for reliable tracking at 3000 RPM. */
#define SMC_SMO_K            ((MatrixFloat)2.5f)  /* k > BEMF_max = we_max*lpm = 1256*0.0014 = 1.76V, 1.4x margin */

/** \brief SMO back-EMF LPF cutoff  ωc = 2π×500 Hz  [rad/s] */
#define SMC_SMO_WC           ((MatrixFloat)3141.592653589793f)

/** \brief SMO back-EMF LPF coefficient α = ωc·dt / (1 + ωc·dt)
 *  Pre-computed at dt = 50 µs (20 kHz sampling) — retained for reference only.
 *  SMC_SMO_Step() now computes α dynamically each call from SMC_SMO_WC and the
 *  supplied dt, so this constant is not used in the control loop.
 *  If dt changes (e.g. variable-rate debug mode), the dynamic computation is
 *  automatically correct without touching this header. */
#define SMC_SMO_LPF_ALPHA    ((MatrixFloat)0.13588f)


/*********************************************************************************************************************/
/*------------------------------------ Encoder speed estimator parameters --------------------------------------------*/
/*********************************************************************************************************************/

/** \brief IIR cutoff frequency for encoder speed estimator -- fast branch [Hz].
 *  Applied when |omega_m_filt| > 52.4 rad/s (500 RPM).
 *  At dt=50 µs: alpha_fast = 0.300, tau ~ 3 steps (150 µs).
 *  Below 500 RPM: alpha_slow = 0.01 (100-step boxcar) suppresses GPT12 rollover spikes.
 *  Matched to Python _get_speed_from_encoder() alpha_win logic. */
#define SMC_SPEED_IIR_FC     ((MatrixFloat)1364.2f)


/*********************************************************************************************************************/
/*-------------------------------------- Soft-start current ramp parameter -------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Soft-start ramp duration [s].
 *  iq_limit ramps from 0 → I_MAX over this interval.
 *  Absorbs the motor_delay zero-fallback spike at t=0 without dead-time.
 *  50 ms = 1000 steps at 20 kHz. */
#define SMC_SOFTSTART_T        ((MatrixFloat)0.05f)


/*********************************************************************************************************************/
/*--------------- Tunable gain defaults — see embed_sim_smc_gains.h (patched by smc_fmu_tuner.py) -------------------*/
/*********************************************************************************************************************/
/*
 * SMC_KS_I, SMC_PHI_I, SMC_T_MAX, SMC_PHI_W, SMC_KS_W, SMC_ETA_W
 * are defined in smc_gains_config.h.
 *
 * smc_fmu_tuner.py patches ONLY that file — this header is never modified
 * by the tuner.  Recompile after patching to load new startup defaults into
 * g_smc_gains via SMC_Controller_Init().
 */
#include "embed_sim_smc_gains.h"


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \enum SMC_ControlMode_T
 * \brief  Control mode selection (encoder vs sensorless)
 */
typedef enum
{
    SMC_MODE_ENCODER = 0U,      /**< Use encoder for position/speed feedback */
    SMC_MODE_SENSORLESS = 1U    /**< Use SMO for position/speed feedback */
} SMC_ControlMode_T;

/**
 * \struct SMC_GainSet_T
 * \brief  Runtime-configurable SMC gains.
 *
 * This struct lives in RAM and can be written at any time without
 * recompiling.  SMC_Controller_Step() reads gains from g_smc_gains.
 *
 * Member  │ Units  │ Tunes      │ Description
 * ────────┼────────┼────────────┼──────────────────────────────────────
 * ks_w    │ A      │ smc_fmu_   │ Speed SMC switching gain (q-axis current amplitude)
 * eta_w   │ —      │ tuner.py   │ Speed SMC linear damping
 * phi_w   │ rad/s  │            │ Speed boundary layer thickness
 * ks_i    │ V      │            │ Current SMC switching gain (physical, pre-SVPWM)
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
 *
 * \warning  These fields carry a \b normalised reference, \b not physical volts.
 *           SMC_Controller_Step() computes v_alpha / v_beta in physical [V] via
 *           the Inverse-Park transform, then divides by
 *           SMC_SVPWM_GAIN = V_DC / 2 before writing these fields.
 *           The SVPWM block (SVPWMPack → SVM_CalculateDutyCycle) therefore
 *           receives a modulation index in [−1, +1] as required.
 *           Do \b not use these values as physical voltages; read
 *           SMC_Controller_T::v_alpha_prev / v_beta_prev for the physical [V]
 *           values stored before normalisation (used by the SMO observer).
 */
typedef struct
{
    MatrixFloat v_alpha;   /**< α-axis normalised reference for SVPWM [−1, +1] (= v_α_physical / SMC_SVPWM_GAIN) */
    MatrixFloat v_beta;    /**< β-axis normalised reference for SVPWM [−1, +1] (= v_β_physical / SMC_SVPWM_GAIN) */
} SMC_Output_T;


/**
 * \struct SMC_Controller_T
 * \brief  Full controller state — integrators, SMO, transforms, diagnostics.
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

    /* ── Sliding Mode Observer (SMO) state ──────────────────────────────── */
    MatrixFloat i_alpha_hat;    /**< Estimated α-axis current [A]          */
    MatrixFloat i_beta_hat;     /**< Estimated β-axis current [A]          */
    MatrixFloat e_alpha_filt;   /**< LPF-filtered back-EMF ê_α [V]        */
    MatrixFloat e_beta_filt;    /**< LPF-filtered back-EMF ê_β [V]        */
    MatrixFloat theta_e_hat;    /**< Estimated electrical angle [rad]      */
    MatrixFloat theta_e_hat_prev; /**< Previous θ̂_e for speed extraction  */
    MatrixFloat omega_m_hat;    /**< Estimated mechanical speed [rad/s]    */
    MatrixFloat v_alpha_prev;   /**< Applied v_α at previous step [V]      */
    MatrixFloat v_beta_prev;    /**< Applied v_β at previous step [V]      */

    /* Speed and angle — populated from encoder each step */
    MatrixFloat omega_m;       /**< Filtered mechanical speed [rad/s] — encoder diff + IIR */
    MatrixFloat theta_m_prev;  /**< theta_m at previous step for finite-difference [rad]   */
    MatrixFloat omega_m_filt;  /**< IIR-filtered raw speed estimate [rad/s]                */

    /* Soft-start current limit: ramps 0 → I_MAX over SMC_SOFTSTART_T seconds */
    MatrixFloat iq_limit;      /**< Current soft limit [A] — rises each step               */

    /* Diagnostic references */
    MatrixFloat iq_ref;        /**< q-axis current reference [A] */
    MatrixFloat id_ref;        /**< d-axis current reference [A] — MTPA = 0 */

    /* Diagnostic voltage outputs — normalised (÷ V_DC/2), not physical [V].
     * SMC_CurrentSMC() applies the SVPWM normalisation before returning,
     * so these hold the values passed to InvPark (same units as v_alpha/v_beta). */
    MatrixFloat vd;            /**< d-axis voltage normalised (= vd_physical / SMC_SVPWM_GAIN) */
    MatrixFloat vq;            /**< q-axis voltage normalised (= vq_physical / SMC_SVPWM_GAIN) */

    /* Embedded transform states (MISRA Rule 8.7 — no static locals) */
    Clarke_T   clarke_state;
    Park_T     park_state;
    InvPark_T  inv_park_state;

    /* Diagnostic log (updated at 1 kHz) */
    MatrixFloat log_speed;
    MatrixFloat log_speed_ref;
    MatrixFloat log_iq_meas;
    MatrixFloat log_id_meas;
    MatrixFloat log_control_mode;      /**< Control mode at log time */
    MatrixFloat log_encoder_glitches;  /**< Glitch count at log time */
    uint32_T    log_counter;
    MatrixFloat log_next_time;

    /* Encoder fault detection */
    uint32_T    encoder_glitch_count;  /**< Number of consecutive glitches detected */
    uint32_T    encoder_good_count;    /**< Number of consecutive good samples */
    uint32_T    encoder_fault_detected; /**< 1 if encoder fault has been detected */
    uint32_T    control_mode;          /**< Current control mode (0=encoder, 1=sensorless) */

    /* Back-EMF PLL speed estimator -- added AFTER all existing fields.
     * No existing field offsets are disturbed.
     * Second-order PLL: wn = 2*pi*80 Hz, zeta = 1/sqrt(2)
     *   Kp = 2*zeta*wn = 711.0   Ki = wn^2 = 252657
     * Tracks theta_e_hat from SMO. Pure algorithmic -- no registers.
     * Matches MATLAB Embedded Coder bemf_pll block output. */
    MatrixFloat _reserved_pll_1;  /**< Reserved -- was theta_e_pll */
    MatrixFloat _reserved_pll_2;  /**< Reserved -- was omega_e_pll */

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
 * \brief  Get encoder fault status.
 */
extern void SMC_Controller_GetFaultStatus(
    const SMC_Controller_T * const s,
    uint32_T               * const control_mode,
    uint32_T               * const glitch_count,
    uint32_T               * const fault_detected);

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
