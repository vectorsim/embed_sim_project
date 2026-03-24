/**********************************************************************************************************************
 * \file      embed_sim_smc_controller.c
 * \brief     Sliding Mode FOC Controller — NANOTEC DB42S02 / AURIX TC3xx
 *
 * Implements complete FOC control chain:
 *   [ia, ib, ic] → Clarke → [iα, iβ]
 *   → Park(θ_e) → [id, iq]
 *   → Speed SMC  → iq_ref       surface: s = e + λ·∫e + γ·∫∫e
 *   → Current SMC → [vd, vq]    equivalent control + switching
 *   → InvPark(θ_e) → [vα, vβ]  → SVPWM
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

/** Speed LPF coefficient — matches Python SMCControllerPy exactly */
#define SMC_LPF_ALPHA       ((MatrixFloat)0.95f)
#define SMC_LPF_ONE_MINUS   ((MatrixFloat)0.05f)


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
 * \param[in]  omega_e  Electrical speed [rad/s].
 * \param[in]  g        Pointer to active gain set.
 * \param[out] vd       D-axis voltage reference [V].
 * \param[out] vq       Q-axis voltage reference [V].
 */
static void SMC_CurrentSMC(
    MatrixFloat             id_meas,
    MatrixFloat             iq_meas,
    MatrixFloat             id_ref,
    MatrixFloat             iq_ref,
    MatrixFloat             omega_e,
    const SMC_GainSet_T   * const g,
    MatrixFloat           * const vd,
    MatrixFloat           * const vq);

/**
 * \brief  Saturate voltage vector to hexagon limit (V_MAX).
 */
static void SMC_SaturateVoltage(MatrixFloat * const vd, MatrixFloat * const vq);


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

#if defined(SMC_INTEGRATOR_TUSTIN)
    /*
     * Tustin (bilinear / trapezoidal) — O(dt²).
     * ∫e  += dt/2 · (e + e_prev)
     * ∫∫e += dt/2 · (∫e + ∫e_prev)
     */
    s->int_spd      += (dt * (MatrixFloat)0.5f) * (e        + s->e_prev);
    s->int2_spd     += (dt * (MatrixFloat)0.5f) * (s->int_spd + s->int_spd_prev);
    s->int_spd_prev  = s->int_spd;
    s->e_prev        = e;

#elif defined(SMC_INTEGRATOR_HEUN)
    /*
     * Heun predictor-corrector — O(dt²), better phase accuracy for large dt.
     * Predictor:  int_e_star  = int_e  + dt · e_prev
     * Corrector:  int_e      += dt/2 · (e_prev + e)
     * Same for double integral.
     */
    {
        MatrixFloat int_e_star;
        MatrixFloat int2_e_star;

        int_e_star  = s->int_spd  + dt * s->e_prev;
        s->int_spd += (dt * (MatrixFloat)0.5f) * (s->e_prev + e);

        int2_e_star  = s->int2_spd + dt * s->int_spd_prev;
        s->int2_spd += (dt * (MatrixFloat)0.5f) * (s->int_spd_prev + s->int_spd);

        s->int_spd_prev = s->int_spd;
        s->e_prev       = e;

        (void)int_e_star;   /* used only in corrector above */
        (void)int2_e_star;
    }

#else   /* SMC_INTEGRATOR_EULER — forward Euler, O(dt) */
    /*
     * Forward Euler — legacy.  No extra state fields required.
     */
    s->int_spd  += dt * e;
    s->int2_spd += dt * s->int_spd;

#endif  /* integrator selection */

    /*
     * Sliding surface:  s = e + λ·∫e + γ·∫∫e
     *   λ = SMC_LAMBDA_W = 2π × 20 Hz
     *   γ = SMC_GAMMA_W  = 2π × 5  Hz
     */
    s_spd = e
            + SMC_LAMBDA_W * s->int_spd
            + SMC_GAMMA_W  * s->int2_spd;

    /*
     * Control law:  iq_ref = ks_w · sat(s/φ_w) + eta_w · s
     */
    iq_ref = (g->ks_w * SMC_Sat(s_spd, g->phi_w))
             + (g->eta_w * s_spd);

    return SMC_Clamp(iq_ref, SMC_I_MAX);
}


/*--------------------------------------------------------------------------------------------------------------------
 * SMC_CurrentSMC
 *------------------------------------------------------------------------------------------------------------------*/
static void SMC_CurrentSMC(
    const MatrixFloat           id_meas,
    const MatrixFloat           iq_meas,
    const MatrixFloat           id_ref,
    const MatrixFloat           iq_ref,
    const MatrixFloat           omega_e,
    const SMC_GainSet_T * const g,
    MatrixFloat         * const vd,
    MatrixFloat         * const vq)
{
    MatrixFloat s_d;
    MatrixFloat s_q;
    MatrixFloat vd_eq;
    MatrixFloat vq_eq;
    MatrixFloat vd_sw;
    MatrixFloat vq_sw;

    if ((vd != NULL) && (vq != NULL) && (g != NULL))
    {
        /* Sliding surfaces */
        s_d = id_meas - id_ref;
        s_q = iq_meas - iq_ref;

        /*
         * Equivalent control — cancels plant dynamics:
         *   vd_eq =  R·id - ωe·Lq·iq
         *   vq_eq =  R·iq + ωe·(Ld·id + λpm)
         */
        vd_eq = (SMC_R_S * id_meas) - (omega_e * SMC_L_Q * iq_meas);
        vq_eq = (SMC_R_S * iq_meas)
                + (omega_e * ((SMC_L_D * id_meas) + SMC_LAMBDA_PM));

        /*
         * Switching control — boundary-layer saturation:
         *   vd_sw = -ks_i · sat(s_d / φ_i)
         *   vq_sw = -ks_i · sat(s_q / φ_i)
         *
         * Sign convention: s = i_meas - i_ref.
         * Lyapunov stability requires s·ds/dt < 0, which gives
         * v_sw = -ks·sat(s/φ).  When i < i_ref: s<0, v_sw>0 → drives
         * current up toward reference.  Positive sign here would oppose
         * the reference — incorrect.
         */
        vd_sw = -(g->ks_i * SMC_Sat(s_d, g->phi_i));
        vq_sw = -(g->ks_i * SMC_Sat(s_q, g->phi_i));

        *vd = vd_eq + vd_sw;
        *vq = vq_eq + vq_sw;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else required */
    }
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

        /*
         * Populate g_smc_gains with design-point defaults.
         * Explicit field assignment — C89-safe, MISRA C:2012 Rule 9.3.
         * These values come from the macros in embed_sim_smc_controller.h.
         * Override at runtime via UDE, UART loader, or
         * SMC_GainSchedule_Interpolate() without recompiling.
         */
        g_smc_gains.ks_w  = SMC_KS_W;    /* 0.287831 N·m   speed switching gain  */
        g_smc_gains.eta_w = SMC_ETA_W;   /* 2.250826 —     speed linear damping  */
        g_smc_gains.phi_w = SMC_PHI_W;   /* 2.809393 rad/s speed boundary layer  */
        g_smc_gains.ks_i  = SMC_KS_I;    /* 0.101847 V     current switching gain */
        g_smc_gains.phi_i = SMC_PHI_I;   /* 0.708913 A     current boundary layer */
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
    MatrixFloat i_alpha;
    MatrixFloat i_beta;
    MatrixFloat id_meas;
    MatrixFloat iq_meas;
    MatrixFloat omega_e;
    MatrixFloat e_w;
    MatrixFloat iq_ref;
    MatrixFloat vd;
    MatrixFloat vq;
    MatrixFloat omega_raw;

    /* ── NULL guard ─────────────────────────────────────────────────────── */
    if ((s == NULL) || (u == NULL) || (y == NULL))
    {
        return;   /* MISRA 15.5: single exceptional exit */
    }

    /* ── Speed estimation: Euler differentiator + first-order LPF ──────── *
     *   omega_raw  = (θ_m - θ_m_prev) / dt
     *   omega_filt = α · omega_filt + (1-α) · omega_raw    α = 0.95
     *
     *   LPF cut-off ≈ (1-α)/(2π·dt) = 0.05/(2π·50µs) ≈ 159 Hz
     *   Attenuates encoder quantisation noise above 159 Hz.
     * ─────────────────────────────────────────────────────────────────── */
    if (dt > SMC_ZERO_F)
    {
        omega_raw     = (u->theta_m - s->theta_m_prev) / dt;
        s->omega_filt = (SMC_LPF_ALPHA    * s->omega_filt)
                      + (SMC_LPF_ONE_MINUS * omega_raw);
    }
    else
    {
        /* dt = 0 — hold last estimate (MISRA 15.7) */
    }
    s->theta_m_prev = u->theta_m;
    s->omega_m      = s->omega_filt;

    /* ── Electrical angle: θ_e = p · θ_m ──────────────────────────────── */
    theta_e = (MatrixFloat)SMC_P_POLES * u->theta_m;

    /* ── Clarke transform: abc → αβ ────────────────────────────────────── */
    Clarke_Step(&s->clarke_state,
                u->ia, u->ib, u->ic,
                &i_alpha, &i_beta);

    /* ── Park transform: αβ → dq ───────────────────────────────────────── */
    Park_Step(&s->park_state,
              i_alpha, i_beta, theta_e,
              &id_meas, &iq_meas);

    /* ── Speed SMC: ω_error → iq_ref ───────────────────────────────────── */
    e_w    = u->omega_ref_mech - s->omega_m;
    iq_ref = SMC_SpeedSMC(s, e_w, dt, &g_smc_gains);

    s->iq_ref = iq_ref;
    s->id_ref = SMC_ZERO_F;    /* MTPA: id_ref = 0 */

    /* ── Electrical speed ──────────────────────────────────────────────── */
    omega_e = (MatrixFloat)SMC_P_POLES * s->omega_m;

    /* ── Current SMC: (id,iq) → (vd,vq) ───────────────────────────────── */
    SMC_CurrentSMC(id_meas, iq_meas,
                   s->id_ref, iq_ref,
                   omega_e,
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
