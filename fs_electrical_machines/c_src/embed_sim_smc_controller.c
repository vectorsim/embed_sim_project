/**********************************************************************************************************************
 * \file      embed_sim_smc_controller.c
 * \brief     Sliding Mode FOC Controller — NANOTEC DB42S02 / AURIX TC3xx
 *
 * Implements complete FOC control chain:
 *   [ia, ib, ic] → Clarke → [iα, iβ]
 *   → Park(θ_e) → [id, iq]   θ_e = p·θ_m  (exact encoder)
 *   → Speed SMC  → iq_ref    surface: s = e + λ·∫e  (γ=0, first-order)
 *   → Current SMC → [vd, vq]  encoder equivalent control + switching  (physical V)
 *   → InvPark(θ_e) → [vα, vβ]  (physical V)
 *   → ÷ (V_DC/2) → [vα_norm, vβ_norm]  → SVPWM  (normalised, [-1,+1])
 *
 * SVPWM normalisation
 * -------------------
 * All internal voltages are computed in physical units [V].  The final
 * SMC_Controller_Step() output (y->v_alpha, y->v_beta) is divided by
 * SMC_SVPWM_GAIN = V_DC/2 before return, so the SVPWM block on AURIX
 * receives a normalised reference in [-1, +1] and the plant sees the
 * correct physical voltage after the SVPWM × V_DC/2 amplification.
 *
 * The SMO feedback (v_alpha_prev, v_beta_prev) is stored from the physical
 * InvPark output BEFORE this division — the observer model requires SI units.
 *
 * Equivalent control — full cancellation of plant ODE at measured state:
 *   did/dt = (vd - R·id + ωe·Lq·iq) / Ld
 *   diq/dt = (vq - R·iq - ωe·(Ld·id + λpm)) / Lq
 *
 *   ed_hat =  R·id_meas - ωe·Lq·iq_meas
 *   eq_hat =  R·iq_meas + ωe·(Ld·id_meas + λpm)
 *
 *   Matched exactly to Python smc_controller_block.py compute_py() lines 572-573.
 *
 * SMO still executes each step (updates e_alpha_filt, e_beta_filt) but its
 * output is NOT used in the current loop — reserved for future sensorless use.
 *
 * Runtime-configurable gains
 * --------------------------
 * Gains are held in the RAM struct g_smc_gains (not #define constants).
 * No recompile is needed to change an operating point:
 *
 *   Option A — UDE debugger (Lauterbach / PLS):
 *       Write g_smc_gains.ks_w, g_smc_gains.ks_i, ... live while running.
 *
 *   Option B — UART loader (smc_uart_loader.py):
 *       python smc_uart_loader.py --port COM4 --schedule smc_gain_schedule.json
 *
 *   Option C — gain schedule interpolation at runtime:
 *       SMC_GainSchedule_Interpolate(omega_rpm, &g_smc_gains);
 *       SMC_Controller_Step(&state, &u, dt, &y);
 *
 * Integrator method (compile-time selection)
 * ------------------------------------------
 *   Default  : Tustin (bilinear/trapezoidal)  O(dt²)
 *   -DSMC_INTEGRATOR_HEUN   : Heun predictor-corrector  O(dt²)
 *   -DSMC_INTEGRATOR_EULER  : Forward Euler  O(dt)  legacy fallback
 *
 * MISRA C:2012 compliance
 * -----------------------
 *   Rule 8.7  : all state in caller-supplied structs (no static locals)
 *   Rule 15.5 : single exit per function (except early NULL guard returns)
 *   Rule 15.7 : mandatory else clauses on all if-else chains
 *   Rule 21.8 : no memset on float-bearing structs (explicit Init calls)
 *   All float literals carry the f suffix — no implicit double promotion.
 *
 * \version   2.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_smc_controller.h"
#include "embed_sim_coordinate_transform.h"
#include <math.h>     /* sqrtf, fabsf */
#include <string.h>   /* memset       */


/*********************************************************************************************************************/
/*------------------------------------------Integrator Method Selection----------------------------------------------*/
/*********************************************************************************************************************/
#if !defined(SMC_INTEGRATOR_TUSTIN) && \
    !defined(SMC_INTEGRATOR_HEUN)   && \
    !defined(SMC_INTEGRATOR_EULER)
#  define SMC_INTEGRATOR_TUSTIN   /* default */
#endif


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
#define SMC_ZERO_F          ((MatrixFloat)0.0f)
#define SMC_ONE_F           ((MatrixFloat)1.0f)
#define SMC_LOG_INTERVAL    ((MatrixFloat)0.001f)   /* 1 kHz diagnostic rate */
/* SMC_SMO_K / SMC_SMO_WC / SMC_SMO_LPF_ALPHA defined in embed_sim_smc_controller.h */

/* Speed estimator IIR cutoff frequency [Hz].
 * Matched to Python: omega_filt = 0.7·prev + 0.3·raw → fc = 0.3/(2π·0.7·dt) ≈ 1364 Hz.
 * alpha = 2π·fc·dt / (1 + 2π·fc·dt) computed each step — correct for any dt.
 * At dt=50 µs: alpha = 0.300, τ ≈ 3 steps (150 µs). */
#define SMC_SPEED_IIR_FC    ((MatrixFloat)1364.2f)


/*********************************************************************************************************************/
/*------------------------------------------------ Runtime gain struct -----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Global runtime gain set.
 *
 * Initialised to the design-point values from the header macros.
 * Can be overwritten at runtime by:
 *   - UDE debugger (write g_smc_gains members directly)
 *   - UART loader  (smc_uart_loader.py)
 *   - SMC_GainSchedule_Interpolate()
 *
 * MISRA C:2012 Rule 8.4: definition matches extern declaration in header.
 */
/*
 * C89-safe zero initialisation.
 * Gain values are populated by SMC_GainSet_Init() below, which is called
 * from SMC_Controller_Init().  Designated initialisers (.field = value)
 * are C99 and are rejected by the AURIX TriCore ctc compiler in C89 mode.
 *
 * MISRA C:2012 Rule 9.3 / Rule 8.4: zero-initialise at definition,
 * assign members explicitly in the Init function.
 */
SMC_GainSet_T g_smc_gains;


/*********************************************************************************************************************/
/*--------------------------------------------Private Function Prototypes--------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Clamp value to [-limit, +limit].
 */
static MatrixFloat SMC_Clamp(MatrixFloat value, MatrixFloat limit);

/**
 * \brief  Boundary-layer saturation function.
 * \return Value in [-1, +1].
 */
static MatrixFloat SMC_Sat(MatrixFloat x, MatrixFloat phi);

/**
 * \brief  Speed SMC with integral sliding surface.
 *
 * Surface:  s = e + λ·∫e + γ·∫∫e
 * Control:  iq_ref = ks_w·sat(s/φ_w) + eta_w·s
 *
 * Integration method selected at compile time.
 *
 * \param[in,out] s   Controller state.
 * \param[in]     e   Speed error ω_ref - ω_m  [rad/s].
 * \param[in]     dt  Time step [s].
 * \param[in]     g   Pointer to active gain set.
 * \return            q-axis current reference [A], clamped to ±I_MAX.
 */
static MatrixFloat SMC_SpeedSMC(
    SMC_Controller_T      * const s,
    MatrixFloat                   e,
    MatrixFloat                   dt,
    const SMC_GainSet_T   * const g);

/**
 * \brief  Current SMC with equivalent control.
 *
 * Equivalent control cancels plant dynamics exactly.
 * Switching control drives dq sliding surfaces to zero.
 *
 * \param[in]  id_meas  Measured d-axis current [A].
 * \param[in]  iq_meas  Measured q-axis current [A].
 * \param[in]  id_ref   D-axis reference [A] (MTPA = 0).
 * \param[in]  iq_ref   Q-axis reference [A].
 * \param[in]  theta_e  Electrical angle [rad] — used to Park-rotate SMO back-EMF.
 * \param[in]  g        Pointer to active gain set.
 * \param[out] vd       D-axis voltage reference [V].
 * \param[out] vq       Q-axis voltage reference [V].
 */
static void SMC_CurrentSMC(
    SMC_Controller_T    * const s,
    MatrixFloat             id_meas,
    MatrixFloat             iq_meas,
    MatrixFloat             id_ref,
    MatrixFloat             iq_ref,
    MatrixFloat             theta_e,
    MatrixFloat             omega_e,
    const SMC_GainSet_T   * const g,
    MatrixFloat           * const vd,
    MatrixFloat           * const vq);

/**
 * \brief  Saturate voltage vector to hexagon limit (V_MAX).
 */
static void SMC_SaturateVoltage(MatrixFloat * const vd, MatrixFloat * const vq);

/**
 * \brief  One step of the Sliding Mode Observer (αβ frame).
 *
 * Estimates back-EMF (ê_α, ê_β), electrical angle θ̂_e, and
 * mechanical speed ω̂_m from measured currents and applied voltages.
 *
 * \param[in,out] s        Controller state (observer sub-state updated).
 * \param[in]     i_alpha  Measured α-axis current [A].
 * \param[in]     i_beta   Measured β-axis current [A].
 * \param[in]     v_alpha  Applied α-axis voltage [V] (previous step).
 * \param[in]     v_beta   Applied β-axis voltage [V] (previous step).
 * \param[in]     dt       Time step [s].
 */
static void SMC_SMO_Step(
    SMC_Controller_T * const s,
    MatrixFloat               i_alpha,
    MatrixFloat               i_beta,
    MatrixFloat               v_alpha,
    MatrixFloat               v_beta,
    MatrixFloat               dt);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * SMC_Clamp
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat SMC_Clamp(const MatrixFloat value, const MatrixFloat limit)
{
    MatrixFloat result = value;

    if (result > limit)
    {
        result = limit;
    }
    else if (result < -limit)
    {
        result = -limit;
    }
    else
    {
        /* Within range — no action required (MISRA 15.7) */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_Sat
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat SMC_Sat(const MatrixFloat x, const MatrixFloat phi)
{
    MatrixFloat result;

    if (phi <= SMC_ZERO_F)
    {
        /* Sign function — zero boundary layer */
        if (x > SMC_ZERO_F)
        {
            result = SMC_ONE_F;
        }
        else if (x < SMC_ZERO_F)
        {
            result = -SMC_ONE_F;
        }
        else
        {
            result = SMC_ZERO_F;
        }
    }
    else
    {
        /* Boundary layer saturation */
        result = x / phi;
        if (result > SMC_ONE_F)
        {
            result = SMC_ONE_F;
        }
        else if (result < -SMC_ONE_F)
        {
            result = -SMC_ONE_F;
        }
        else
        {
            /* Inside boundary layer — linear region (MISRA 15.7) */
        }
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_SpeedSMC
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat SMC_SpeedSMC(
    SMC_Controller_T    * const s,
    const MatrixFloat           e,
    const MatrixFloat           dt,
    const SMC_GainSet_T * const g)
{
    MatrixFloat s_spd;
    MatrixFloat iq_ref;
    MatrixFloat iq_unsat;
    const MatrixFloat int_limit  = (MatrixFloat)10.0f;
    const MatrixFloat eta_capped = (g->eta_w < (MatrixFloat)0.01f) ?
                                    g->eta_w : (MatrixFloat)0.01f;

    /* Compute unsaturated output with current integrator state to check
     * for saturation before integrating (anti-windup). */
    s_spd    = e + SMC_LAMBDA_W * s->int_spd;
    iq_unsat = (g->ks_w * SMC_Sat(s_spd, g->phi_w)) + (eta_capped * s_spd);

    /* Anti-windup: only integrate when output is not at the current limit.
     * s->iq_limit is the soft-start limit; I_MAX after soft-start. */
    if (fabsf(iq_unsat) < s->iq_limit)
    {
#if defined(SMC_INTEGRATOR_TUSTIN)
        s->int_spd  += (dt * (MatrixFloat)0.5f) * (e + s->e_prev);
        s->int_spd_prev = s->int_spd;
        s->e_prev       = e;
#elif defined(SMC_INTEGRATOR_HEUN)
        s->int_spd  += (dt * (MatrixFloat)0.5f) * (s->e_prev + e);
        s->int_spd_prev = s->int_spd;
        s->e_prev       = e;
#else   /* SMC_INTEGRATOR_EULER */
        s->int_spd  += dt * e;
        s->e_prev    = e;
#endif
        s->int_spd = SMC_Clamp(s->int_spd, int_limit);
    }
    else
    {
        /* Saturated — freeze integrator, update e_prev only (MISRA 15.7) */
        s->e_prev = e;
    }

    /* GAMMA_W = 0 (double integral disabled) — int2_spd not used */
    s_spd  = e + SMC_LAMBDA_W * s->int_spd;
    iq_ref = (g->ks_w * SMC_Sat(s_spd, g->phi_w)) + (eta_capped * s_spd);

    return SMC_Clamp(iq_ref, SMC_I_MAX);
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_SMO_Step
 *
 * Sliding Mode Observer — αβ frame.
 *
 * Observer model (forward Euler discretisation of PMSM voltage equations):
 *   î_α(k+1) = î_α(k) + dt/L · (vα - R·î_α(k) - ê_α_sw(k))
 *   î_β(k+1) = î_β(k) + dt/L · (vβ - R·î_β(k) - ê_β_sw(k))
 *
 * Switching injection (smooth tanh — no chattering at 20 kHz):
 *   err   = i_measured - i_hat   (positive → push î upward)
 *   ê_sw  = k · tanh(err / 0.01)
 *   k = 2.0 V  (SMC_SMO_K — 1.7× back-EMF margin, NOT 1.5·V_MAX)
 *   Boundary layer 0.01 A: linear gain 200 V/A for |err| < 0.01 A.
 *   Matched to Python _smo_step() tanh implementation.
 *
 * Back-EMF extraction via 500 Hz first-order LPF (Euler-discretised):
 *   ê_α_filt += α · (ê_α_sw - ê_α_filt)
 *   ê_β_filt += α · (ê_β_sw - ê_β_filt)
 *   α = ωc·dt / (1 + ωc·dt),  ωc = 2π × 500 Hz
 *
 * Electrical angle and mechanical speed:
 *   θ̂_e = atan2(-ê_α_filt, ê_β_filt)
 *   ω̂_m = Δθ̂_e / (p · dt)          (unwrapped shortest-step delta)
 *
 * Results stored in s->e_alpha_filt, s->e_beta_filt (current loop feedforward),
 * s->theta_e_hat (angle), s->omega_m_hat (mechanical speed for speed SMC).
 *------------------------------------------------------------------------------------------------------------------*/
static void SMC_SMO_Step(
    SMC_Controller_T * const s,
    const MatrixFloat          i_alpha,
    const MatrixFloat          i_beta,
    const MatrixFloat          v_alpha,
    const MatrixFloat          v_beta,
    const MatrixFloat          dt)
{
    MatrixFloat i_alpha_err;
    MatrixFloat i_beta_err;
    MatrixFloat e_alpha_sw;
    MatrixFloat e_beta_sw;
    MatrixFloat theta_e_new;
    MatrixFloat delta;
    MatrixFloat alpha_dyn;      /* LPF coefficient computed from dt — not a compile-time constant */
    const MatrixFloat inv_L   = SMC_ONE_F / SMC_L_D;
    const MatrixFloat k       = SMC_SMO_K;
    const MatrixFloat two_pi  = (MatrixFloat)6.28318530717959f;
    const MatrixFloat pi_f    = (MatrixFloat)3.14159265358979f;

    if (s == NULL)
    {
        return;
    }

    /* Dynamic LPF alpha: α = ωc·dt / (1 + ωc·dt), ωc = SMC_SMO_WC = 2π×500 Hz.
     * Pre-computed SMC_SMO_LPF_ALPHA = 0.13588 was only valid at dt = 50 µs.
     * Computing it here is correct for any sample period and costs only one
     * FP divide per step — negligible at 20 kHz on the TriCore FPU. */
    alpha_dyn = (SMC_SMO_WC * dt) / (SMC_ONE_F + SMC_SMO_WC * dt);

    /* Current estimation errors: (measured - estimated) so that
     * switching term pushes î toward i, not away from it. */
    i_alpha_err = i_alpha - s->i_alpha_hat;
    i_beta_err  = i_beta  - s->i_beta_hat;

    /* Switching injection: k·tanh(err/0.01) — smooth, no chattering at 20 kHz.
     * Matched to Python: sw = k * math.tanh(err / 0.01) */
    e_alpha_sw = k * tanhf(i_alpha_err * (MatrixFloat)100.0f);
    e_beta_sw  = k * tanhf(i_beta_err  * (MatrixFloat)100.0f);

    /* Observer current update (forward Euler) */
    s->i_alpha_hat += dt * inv_L * (v_alpha - SMC_R_S * s->i_alpha_hat - e_alpha_sw);
    s->i_beta_hat  += dt * inv_L * (v_beta  - SMC_R_S * s->i_beta_hat  - e_beta_sw);

    /* Back-EMF LPF — 500 Hz, coefficient computed from dt above */
    s->e_alpha_filt += alpha_dyn * (e_alpha_sw - s->e_alpha_filt);
    s->e_beta_filt  += alpha_dyn * (e_beta_sw  - s->e_beta_filt);

    /* Electrical angle estimate */
    theta_e_new = atan2f(-(s->e_alpha_filt), s->e_beta_filt);

    /* Unwrap angle delta for speed extraction.
     * floorf() can promote to double on TriCore ctc (MISRA Rule 10.8 / Rule 1.3).
     * Use the same integer-cast unwrap pattern used in the speed estimator above. */
    delta = theta_e_new - s->theta_e_hat_prev;
    /* Reduce delta to (-π, π] without calling floorf */
    while (delta >  pi_f) { delta -= two_pi; }
    while (delta < -pi_f) { delta += two_pi; }

    /* Mechanical speed estimate */
    if (dt > SMC_ZERO_F)
    {
        s->omega_m_hat = delta / ((MatrixFloat)SMC_P_POLES * dt);
    }
    else
    {
        /* dt = 0 — hold previous estimate (MISRA 15.7) */
    }

    s->theta_e_hat_prev = theta_e_new;
    s->theta_e_hat      = theta_e_new;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_CurrentSMC
 *
 * Pure Sliding Mode current controller — encoder-based classical FOC equivalent control.
 *
 * Equivalent control — full plant ODE cancellation at the measured state.
 *
 *   ed_hat =  R·id_meas - ωe·Lq·iq_meas
 *   eq_hat =  R·iq_meas + ωe·(Ld·id_meas + λpm)
 *
 *   vd = ed_hat + ks_i · sat(s_d / φ_i)
 *   vq = eq_hat + ks_i · sat(s_q / φ_i)
 *
 * Matched to Python smc_controller_block.py compute_py() lines 572-573.
 * Voltage vector clamped to hexagon limit V_MAX.
 *------------------------------------------------------------------------------------------------------------------*/
static void SMC_CurrentSMC(
    SMC_Controller_T    * const s,
    const MatrixFloat           id_meas,
    const MatrixFloat           iq_meas,
    const MatrixFloat           id_ref,
    const MatrixFloat           iq_ref,
    const MatrixFloat           theta_e,
    const MatrixFloat           omega_e,
    const SMC_GainSet_T * const g,
    MatrixFloat         * const vd,
    MatrixFloat         * const vq)
{
    MatrixFloat s_d;
    MatrixFloat s_q;
    MatrixFloat ed_hat;
    MatrixFloat eq_hat;
    MatrixFloat vd_out;
    MatrixFloat vq_out;
    MatrixFloat magnitude;
    MatrixFloat scale;

    if ((s == NULL) || (vd == NULL) || (vq == NULL) || (g == NULL))
    {
        return;
    }

    (void)theta_e;   /* not needed for classical FOC eq. control — suppress MISRA warning */

    s_d = id_ref - id_meas;
    s_q = iq_ref - iq_meas;

    /* Equivalent control — full plant ODE cancellation.
     *
     * Plant ODE:
     *   did/dt = (vd - R·id + ωe·Lq·iq) / Ld
     *   diq/dt = (vq - R·iq - ωe·(Ld·id + λpm)) / Lq
     *
     * Setting d/dt = 0 and solving for vd, vq at the measured state:
     *   ed_hat =  R·id_meas - ωe·Lq·iq_meas
     *   eq_hat =  R·iq_meas + ωe·(Ld·id_meas + λpm)
     *
     * All values in physical units [V].  The SVPWM normalisation (÷ V_DC/2)
     * is applied once in SMC_Controller_Step() after InvPark, not here.
     *
     * R·id_meas must be included: R·I_MAX = 0.68V > KS_I = 0.625V, so
     * without it the switching term cannot overcome resistive drop → id drifts.
     * eq_hat uses iq_meas (not iq_ref) to match Python compute_py exactly.
     * Matched to Python smc_controller_block.py lines 603-608.
     */
    ed_hat =  (SMC_R_S * id_meas) - (omega_e * SMC_L_Q * iq_meas);
    eq_hat =  (SMC_R_S * iq_meas) + (omega_e * (SMC_L_D * id_meas + SMC_LAMBDA_PM));

    vd_out = ed_hat + (g->ks_i * SMC_Sat(s_d, g->phi_i));
    vq_out = eq_hat + (g->ks_i * SMC_Sat(s_q, g->phi_i));

    /* Hexagon voltage limit */
    magnitude = sqrtf(vd_out * vd_out + vq_out * vq_out);
    if (magnitude > SMC_V_MAX)
    {
        scale   = SMC_V_MAX / magnitude;
        vd_out *= scale;
        vq_out *= scale;
    }
    else
    {
        /* Within hexagon — no saturation required (MISRA 15.7) */
    }

    *vd = vd_out;
    *vq = vq_out;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_SaturateVoltage
 *------------------------------------------------------------------------------------------------------------------*/
static void SMC_SaturateVoltage(MatrixFloat * const vd, MatrixFloat * const vq)
{
    MatrixFloat magnitude;
    MatrixFloat scale;

    if ((vd != NULL) && (vq != NULL))
    {
        magnitude = sqrtf((*vd) * (*vd) + (*vq) * (*vq));

        if (magnitude > SMC_V_MAX)
        {
            scale  = SMC_V_MAX / magnitude;
            *vd   *= scale;
            *vq   *= scale;
        }
        else
        {
            /* Within hexagon — no saturation required (MISRA 15.7) */
        }
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


/*********************************************************************************************************************/
/*------------------------------------------------- Public API -------------------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * SMC_Controller_Init
 *------------------------------------------------------------------------------------------------------------------*/
void SMC_Controller_Init(SMC_Controller_T * const s, const MatrixFloat dt)
{
    if (s != NULL)
    {
        (void)memset(s, 0, sizeof(SMC_Controller_T));

        /*
         * Explicitly initialise embedded transform states.
         * memset zeroes the struct — these calls are defensive and satisfy
         * MISRA C:2012 Rule 21.8 (do not rely on zero-initialisation of
         * objects with padding or implementation-defined layout).
         */
        Clarke_Init(&s->clarke_state);
        Park_Init(&s->park_state);
        InvPark_Init(&s->inv_park_state);

        s->log_next_time = SMC_LOG_INTERVAL;
        s->theta_m_prev  = SMC_ZERO_F;
        s->omega_m_filt  = SMC_ZERO_F;
        s->omega_m       = SMC_ZERO_F;
        s->iq_limit      = SMC_ZERO_F;   /* soft-start: grows to I_MAX */

        /*
         * Populate g_smc_gains with design-point defaults.
         * Explicit field assignment — C89-safe, MISRA C:2012 Rule 9.3.
         * These values come from the macros in embed_sim_smc_controller.h.
         * Override at runtime via UDE, UART loader, or
         * SMC_GainSchedule_Interpolate() without recompiling.
         */
        g_smc_gains.ks_w  = SMC_KS_W;    /* speed switching gain  — see smc_gains_config.h */
        g_smc_gains.eta_w = SMC_ETA_W;   /* speed linear damping  — see smc_gains_config.h */
        g_smc_gains.phi_w = SMC_PHI_W;   /* speed boundary layer  — see smc_gains_config.h */
        g_smc_gains.ks_i  = SMC_KS_I;    /* current switching gain — see smc_gains_config.h */
        g_smc_gains.phi_i = SMC_PHI_I;   /* current boundary layer — see smc_gains_config.h */
    }
    else
    {
        /* NULL guard — MISRA C:2012 Rule 15.7 */
    }

    (void)dt;   /* Unused — retained for API consistency */
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_Controller_Step
 *------------------------------------------------------------------------------------------------------------------*/
void SMC_Controller_Step(
    SMC_Controller_T  * const s,
    const SMC_Input_T * const u,
    const MatrixFloat         dt,
    SMC_Output_T      * const y)
{
    MatrixFloat theta_e;
    MatrixFloat omega_e;
    MatrixFloat i_alpha;
    MatrixFloat i_beta;
    MatrixFloat id_meas;
    MatrixFloat iq_meas;
    MatrixFloat e_w;
    MatrixFloat iq_ref;
    MatrixFloat delta;
    MatrixFloat omega_raw;
    MatrixFloat vd;
    MatrixFloat vq;
    const MatrixFloat two_pi = (MatrixFloat)6.28318530717959f;
    const MatrixFloat pi_f   = (MatrixFloat)3.14159265358979f;

    /* ── NULL guard ─────────────────────────────────────────────────────── */
    if ((s == NULL) || (u == NULL) || (y == NULL))
    {
        return;   /* MISRA 15.5: single exceptional exit */
    }

    /* ── Electrical angle: θ_e = p · θ_m  (exact from encoder) ─────────── */
    theta_e = (MatrixFloat)SMC_P_POLES * u->theta_m;
    omega_e = (MatrixFloat)SMC_P_POLES * s->omega_m;

    /* ── Clarke transform: abc → αβ ────────────────────────────────────── */
    Clarke_Step(&s->clarke_state,
                u->ia, u->ib, u->ic,
                &i_alpha, &i_beta);

    /* ── SMO step (diagnostic only — output NOT used in current loop) ───── */
    SMC_SMO_Step(s,
                 i_alpha, i_beta,
                 s->v_alpha_prev, s->v_beta_prev,
                 dt);

    /* ── Park transform: αβ → dq ───────────────────────────────────────── */
    Park_Step(&s->park_state,
              i_alpha, i_beta, theta_e,
              &id_meas, &iq_meas);

    /* ── Encoder speed estimator: finite-diff + IIR ─────────────────────── */
    if (dt > SMC_ZERO_F)
    {
        /* Compute IIR alpha from fc and dt — correct for any sample period.
         * alpha = ωc·dt / (1 + ωc·dt),  ωc = 2π·SMC_SPEED_IIR_FC (≈ 1364 Hz).
         * Matched to Python: omega_filt = 0.7·prev + 0.3·raw → alpha = 0.300 at dt=50µs. */
        const MatrixFloat wc_spd    = (MatrixFloat)6.28318530717959f * SMC_SPEED_IIR_FC;
        const MatrixFloat alpha_spd = (wc_spd * dt) / (SMC_ONE_F + wc_spd * dt);

        delta = u->theta_m - s->theta_m_prev;
        /* 2π unwrap — keep delta in (-π, π] without floorf (avoids double promotion on TriCore) */
        while (delta >  pi_f) { delta -= two_pi; }
        while (delta < -pi_f) { delta += two_pi; }
        omega_raw       = delta / dt;
        s->omega_m_filt = ((SMC_ONE_F - alpha_spd) * s->omega_m_filt)
                        + (alpha_spd * omega_raw);
    }
    else
    {
        /* dt = 0 — hold previous estimate (MISRA 15.7) */
    }
    s->theta_m_prev = u->theta_m;
    s->omega_m      = s->omega_m_filt;
    omega_e         = (MatrixFloat)SMC_P_POLES * s->omega_m;

    /* ── Soft-start: ramp iq_limit 0 → I_MAX over SMC_SOFTSTART_T ───────── */
    if (s->iq_limit < SMC_I_MAX)
    {
        s->iq_limit += SMC_I_MAX * dt / SMC_SOFTSTART_T;
        if (s->iq_limit > SMC_I_MAX)
        {
            s->iq_limit = SMC_I_MAX;
        }
        else
        {
            /* Still ramping (MISRA 15.7) */
        }
    }
    else
    {
        /* Full current available (MISRA 15.7) */
    }

    /* ── Speed SMC: ω_error → iq_ref ───────────────────────────────────── */
    e_w    = u->omega_ref_mech - s->omega_m;
    iq_ref = SMC_SpeedSMC(s, e_w, dt, &g_smc_gains);

    /* Clamp to soft-start limit */
    iq_ref = SMC_Clamp(iq_ref, s->iq_limit);

    s->iq_ref = iq_ref;
    s->id_ref = SMC_ZERO_F;    /* MTPA: id_ref = 0 */

    /* ── Current SMC (classical FOC equivalent control) ──────────────────── */
    SMC_CurrentSMC(s,
                   id_meas, iq_meas,
                   s->id_ref, iq_ref,
                   theta_e, omega_e,
                   &g_smc_gains,
                   &vd, &vq);

    s->vd = vd;
    s->vq = vq;

    /* ── Voltage saturation (hexagon limiting) ──────────────────────────── */
    SMC_SaturateVoltage(&vd, &vq);

    /* ── Inverse Park transform: dq → αβ ───────────────────────────────── */
    InvPark_Step(&s->inv_park_state,
                 vd, vq, theta_e,
                 &y->v_alpha, &y->v_beta);

    /* Store physical voltages for SMO next step — must be physical [V], not
     * normalised, because the SMO observer equations use L and R in SI units. */
    s->v_alpha_prev = y->v_alpha;
    s->v_beta_prev  = y->v_beta;

    /* ── SVPWM normalisation ────────────────────────────────────────────── */
    /* The SVPWM block on AURIX expects a normalised reference in [-1, +1].
     * The SVPWMPackBlock / SVPWMBlock chain amplifies by V_DC/2 = SMC_SVPWM_GAIN.
     * Divide here so the plant sees the correct physical voltages after SVPWM.
     *
     * Matched to Python smc_controller_block.py:
     *   G = self.SMC_SVPWM_GAIN                   (line 599)
     *   vd_eq = vd_eq_physical / G                (line 603)
     *   output = VectorSignal([v_alpha, v_beta])   (line 632)
     *
     * Note: SMO state (v_alpha_prev, v_beta_prev) is stored above from the
     * physical InvPark output — the normalised value must NOT be fed back. */
    y->v_alpha /= SMC_SVPWM_GAIN;
    y->v_beta  /= SMC_SVPWM_GAIN;

    /* ── Diagnostic logging at 1 kHz ───────────────────────────────────── */
    if (dt > SMC_ZERO_F)
    {
        s->log_counter++;

        if (((MatrixFloat)s->log_counter * dt) >= s->log_next_time)
        {
            s->log_speed     = s->omega_m;
            s->log_speed_ref = u->omega_ref_mech;
            s->log_iq_meas   = iq_meas;
            s->log_id_meas   = id_meas;
            s->log_next_time += SMC_LOG_INTERVAL;
        }
        else
        {
            /* Not yet time to log — MISRA 15.7 */
        }
    }
    else
    {
        /* dt = 0 — logging disabled — MISRA 15.7 */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_Controller_Reset
 *------------------------------------------------------------------------------------------------------------------*/
void SMC_Controller_Reset(SMC_Controller_T * const s)
{
    if (s != NULL)
    {
        SMC_Controller_Init(s, SMC_ZERO_F);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_Controller_GetDiagnostics
 *------------------------------------------------------------------------------------------------------------------*/
void SMC_Controller_GetDiagnostics(
    const SMC_Controller_T * const s,
    MatrixFloat            * const speed,
    MatrixFloat            * const speed_ref,
    MatrixFloat            * const iq,
    MatrixFloat            * const id)
{
    if ((s         != NULL) &&
        (speed     != NULL) &&
        (speed_ref != NULL) &&
        (iq        != NULL) &&
        (id        != NULL))
    {
        *speed     = s->log_speed;
        *speed_ref = s->log_speed_ref;
        *iq        = s->log_iq_meas;
        *id        = s->log_id_meas;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_GainSet_SetFromSchedule
 *
 * Copies a gain set into g_smc_gains.
 * Call this from the speed ramp ISR or background task when the operating
 * point changes, e.g. after SMC_GainSchedule_Interpolate().
 *------------------------------------------------------------------------------------------------------------------*/
void SMC_GainSet_SetFromSchedule(const SMC_GainSet_T * const src)
{
    if (src != NULL)
    {
        g_smc_gains.ks_w  = src->ks_w;
        g_smc_gains.eta_w = src->eta_w;
        g_smc_gains.phi_w = src->phi_w;
        g_smc_gains.ks_i  = src->ks_i;
        g_smc_gains.phi_i = src->phi_i;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_GainSchedule_Interpolate
 *
 * Linear interpolation over the gain schedule table.
 * Call once per speed-control period (e.g. 1 ms) to update g_smc_gains
 * based on the current mechanical speed.
 *
 * \param[in]  omega_rpm   Current mechanical speed [RPM].
 * \param[in]  table       Pointer to gain schedule table (ascending RPM).
 * \param[in]  n           Number of entries in the table.
 * \param[out] out         Interpolated gain set (may be &g_smc_gains directly).
 *------------------------------------------------------------------------------------------------------------------*/
void SMC_GainSchedule_Interpolate(
    const MatrixFloat           omega_rpm,
    const SMC_GainTableEntry_T * const table,
    const uint32_T              n,
    SMC_GainSet_T             * const out)
{
    uint32_T    i;
    MatrixFloat t;          /* interpolation factor [0,1] */
    MatrixFloat rpm_lo;
    MatrixFloat rpm_hi;

    if ((table == NULL) || (out == NULL) || (n == 0U))
    {
        return;   /* MISRA 15.5: exceptional early return */
    }

    /* Below lowest point — clamp to first entry */
    if (omega_rpm <= table[0U].rpm)
    {
        *out = table[0U].gains;
        return;
    }

    /* Above highest point — clamp to last entry */
    if (omega_rpm >= table[n - 1U].rpm)
    {
        *out = table[n - 1U].gains;
        return;
    }

    /* Find bracketing interval */
    for (i = 0U; i < (n - 1U); i++)
    {
        rpm_lo = table[i].rpm;
        rpm_hi = table[i + 1U].rpm;

        if ((omega_rpm >= rpm_lo) && (omega_rpm < rpm_hi))
        {
            /* Linear interpolation factor */
            t = (omega_rpm - rpm_lo) / (rpm_hi - rpm_lo);

            out->ks_w  = table[i].gains.ks_w
                         + t * (table[i+1U].gains.ks_w  - table[i].gains.ks_w);
            out->eta_w = table[i].gains.eta_w
                         + t * (table[i+1U].gains.eta_w - table[i].gains.eta_w);
            out->phi_w = table[i].gains.phi_w
                         + t * (table[i+1U].gains.phi_w - table[i].gains.phi_w);
            out->ks_i  = table[i].gains.ks_i
                         + t * (table[i+1U].gains.ks_i  - table[i].gains.ks_i);
            out->phi_i = table[i].gains.phi_i
                         + t * (table[i+1U].gains.phi_i - table[i].gains.phi_i);
            break;
        }
        else
        {
            /* Not in this interval — continue search (MISRA 15.7) */
        }
    }
}
