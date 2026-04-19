/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     Differential Flatness FOC Controller -- NANOTEC DB42S02
 *
 * \details   Implements full-state Field-Oriented Control (FOC) for a surface-mounted
 *            PMSM using the Differential Flatness method.
 *
 *            WHAT IS DIFFERENTIAL FLATNESS?
 *            ================================
 *            A classical PI-FOC controller treats the d- and q-axis voltage equations
 *            as two independent SISO loops and tunes each PI independently.  That works,
 *            but the two loops are physically coupled through back-EMF cross-coupling
 *            terms.  Differential Flatness exploits the fact that the PMSM voltage
 *            equations are already linear and decoupled when written in the dq-frame:
 *
 *              vd = -omega_e * Lq * iq  +  e_d          [d-axis: dominated by cross-coupling]
 *              vq =  R*iq + Lq*diq/dt + omega_e*lambda   [q-axis: back-EMF feedforward]
 *
 *            By computing these "flatness feedforward" terms analytically and injecting
 *            them as a model-based precompensation, the residual error seen by the
 *            feedback gains (Kp_id, Kp_iq) is small.  The result is faster transients
 *            and better high-speed tracking than a pure PI approach, using smaller
 *            feedback gains (lower risk of instability).
 *
 *            SIGNAL FLOW
 *            ============
 *
 *              [omega_ref]
 *                   |
 *              [Speed P-loop]  -->  iq_ref  (clamped to I_MAX)
 *                   |
 *              [LPF: diq/dt]   -->  diq_filt
 *                   |
 *              [ia, ib, ic]  -->  [Clarke]  -->  [i_alpha, i_beta]
 *                                                      |
 *                                               [SMO_Step]  -->  omega_smo_e
 *                                                      |
 *              [theta_m]  -->  [SpeedFusion]  -->  theta_e, omega_e, omega_meas_mech
 *                                                      |
 *                              [i_alpha, i_beta]  -->  [Park]  -->  id_meas, iq_meas
 *                                                      |
 *                                              [DFC_VoltageLaw]  -->  vd, vq
 *                                                      |
 *                                              [InvPark]  -->  v_alpha, v_beta
 *
 *            SPEED ESTIMATION -- SpeedFusion
 *            ================================
 *            Two independent speed sources are available:
 *
 *              1. Encoder finite-difference: low noise, immune to startup problems,
 *                 but quantisation-limited at very low speed.
 *
 *              2. SMO (Sliding Mode Observer): model-based, accurate at medium-to-high
 *                 speed once the back-EMF has converged, but noisy at low speed and
 *                 zero-speed and requires a warmup period.
 *
 *            SpeedFusion blends them with a speed-dependent weight alpha:
 *
 *              alpha = 0   below DFC_FUSION_OMEGA_LO  ->  pure encoder
 *              alpha = 1   above DFC_FUSION_OMEGA_HI  ->  pure SMO
 *              alpha = linear ramp between the two thresholds
 *
 *            A plausibility gate additionally substitutes the encoder value if the
 *            SMO speed deviates by more than DFC_SMO_PLAUS_BAND -- this catches
 *            residual spikes that survive the omega_e_hat clamp inside DFC_SMO_Step.
 *
 * \version   3.2.0
 * \copyright Copyright (C) EmbedSim 2025
 *
 * \par Change history
 *   v3.2.0  Fix 3: DFC_VoltageLaw() -- d-axis PI integral action.
 *           New DFC_State_T.id_integral [V] zeroed in Init/Reset.
 *           DFC_KI_ID = DFC_KP_ID*0.10, DFC_ID_INT_LIMIT = 2.0 V in gains.h.
 *           Conditional anti-windup: integrator frozen when |vd_out| = DFC_V_MAX.
 *           DFC_VoltageLaw() signature extended with DFC_State_T* and dt args.
 *
 *   v3.1.0  Fix 1: DFC_VoltageLaw() -- d-axis-priority voltage saturation (kept).
 *
 *   v3.1.0  Fix 2 REVERTED: iq_ref->iq_meas caused speed-loop collapse.
 *           iq_ref retained; DC id bias now eliminated by Fix 3 integrator.
 *********************************************************************************************************************/


/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "embed_sim_dfc_controller.h"
#include <math.h>       /* fabsf, sqrtf, atan2f                               */
#include <string.h>     /* memset (used indirectly via Clarke/Park init)       */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief Typed zero literal -- avoids bare 0.0f and satisfies MISRA Rule 7.2. */
#define DFC_ZERO_F   ((MatrixFloat)0.0f)

/** \brief Typed unity literal.                                                  */
#define DFC_ONE_F    ((MatrixFloat)1.0f)

/** \brief 2*pi -- full electrical revolution in radians.                        */
#define DFC_TWO_PI_F ((MatrixFloat)6.28318530717959f)

/** \brief pi -- half revolution; used for angle wrap-around arithmetic.         */
#define DFC_PI_F     ((MatrixFloat)3.14159265358979f)


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*
 * All helpers are file-scope (static).  Declaring them before their first use
 * satisfies MISRA C:2012 Rule 8.4 (visible prototype before definition).
 */

static MatrixFloat DFC_Clamp(MatrixFloat value, MatrixFloat limit);
static MatrixFloat DFC_FusionAlpha(MatrixFloat omega_abs);
static MatrixFloat DFC_SMOSwitch(MatrixFloat error);

static void DFC_SpeedFusion_Update(
    DFC_SpeedFusion_T * const fusion,
    MatrixFloat               theta_m,
    MatrixFloat               omega_smo_e,
    MatrixFloat               dt,
    MatrixFloat             * const theta_e,
    MatrixFloat             * const omega_e,
    MatrixFloat             * const omega_meas_mech);

static void DFC_SMO_Step(
    DFC_SMO_T     * const smo,
    MatrixFloat           v_alpha,
    MatrixFloat           v_beta,
    MatrixFloat           i_alpha,
    MatrixFloat           i_beta,
    MatrixFloat           dt,
    uint32_T              warmup_cnt,
    MatrixFloat         * const omega_e_smo);

static void DFC_VoltageLaw(
    DFC_State_T * const s,
    MatrixFloat         iq_ref,
    MatrixFloat         diq_dt,
    MatrixFloat         id_meas,
    MatrixFloat         iq_meas,
    MatrixFloat         omega_e,
    MatrixFloat         dt,
    MatrixFloat * const vd,
    MatrixFloat * const vq);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Clamp
 *
 * \brief  Symmetric magnitude clamp: forces |value| <= limit.
 *
 * \details Used in two places:
 *            - iq_ref   clamped to DFC_I_MAX (3.57 A) to protect the motor.
 *            - diq_filt clamped to I_MAX/DIQ_TAU (3570 A/s) so the L*diq/dt
 *              feedforward term in vq cannot exceed the bus voltage headroom.
 *
 *          Implemented with an explicit three-branch if-else-if-else to satisfy
 *          MISRA C:2012 Rule 15.7 (every if-else chain must have a final else).
 *
 * \param[in] value  Value to clamp.
 * \param[in] limit  Positive magnitude limit.
 * \return           value clamped to [-limit, +limit].
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat DFC_Clamp(const MatrixFloat value, const MatrixFloat limit)
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
        /* Within range -- no action required */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_FusionAlpha
 *
 * \brief  Compute the SpeedFusion blend weight alpha from mechanical speed magnitude.
 *
 * \details Implements a piecewise-linear speed gate:
 *
 *            |omega| <= OMEGA_LO  ->  alpha = 0.0  (full encoder, SMO not trusted)
 *            |omega| >= OMEGA_HI  ->  alpha = 1.0  (full SMO, encoder only for plausibility)
 *            between               ->  alpha = linear ramp
 *
 *          Physical motivation: at low speed the back-EMF (= omega_e * lambda_pm)
 *          is too small for the SMO to distinguish from noise.  The encoder provides
 *          ground truth in that regime.  Above OMEGA_HI the SMO has fully converged
 *          and its noise is lower than encoder quantisation noise at 20 kHz.
 *
 *          Note: alpha is computed from the *previous fused* speed (fusion->omega_e_prev)
 *          inside DFC_SpeedFusion_Update -- one-step delayed but avoids algebraic loop.
 *
 * \param[in] omega_abs  |omega| in rad/s (mechanical or electrical, same threshold units).
 * \return               Blend weight in [0.0, 1.0].
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat DFC_FusionAlpha(const MatrixFloat omega_abs)
{
    MatrixFloat result;

    if (omega_abs <= DFC_FUSION_OMEGA_LO)
    {
        result = DFC_ZERO_F;
    }
    else if (omega_abs >= DFC_FUSION_OMEGA_HI)
    {
        result = DFC_ONE_F;
    }
    else
    {
        /* Linear interpolation between the two thresholds.
         * Numerator = how far above OMEGA_LO we are.
         * Denominator = width of the transition band.             */
        result = (omega_abs - DFC_FUSION_OMEGA_LO)
               / (DFC_FUSION_OMEGA_HI - DFC_FUSION_OMEGA_LO);
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_SMOSwitch
 *
 * \brief  Smooth sign approximation (linear saturation function) for the SMO.
 *
 * \details A pure sign() switching function injects high-frequency chattering
 *          into the current observer, which then propagates into the back-EMF
 *          estimate and the speed signal.  Replacing sign() with a linear
 *          saturation that transitions through zero over a small boundary layer
 *          (width = 0.01 A) eliminates chattering while preserving the sliding
 *          mode convergence property for errors larger than the layer.
 *
 *          The saturation function is:
 *
 *            sat(e) = e / width         if |e/width| <= 5
 *                   = +1                if  e/width  >  5
 *                   = -1                if  e/width  < -5
 *
 *          The ±5 clamp means the function is exactly ±1 for |e| > 0.05 A
 *          (approx 1.4% of I_MAX), so the observer still switches hard outside
 *          the thin boundary layer.
 *
 * \param[in] error  Current estimation error i_meas - i_hat [A].
 * \return           Smooth sign approximation in [-1.0, +1.0].
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat DFC_SMOSwitch(const MatrixFloat error)
{
    MatrixFloat result;
    const MatrixFloat width = (MatrixFloat)0.01f;   /* Boundary layer width [A] */
    const MatrixFloat arg   = error / width;        /* Normalised error         */

    if (arg > (MatrixFloat)5.0f)
    {
        result = DFC_ONE_F;
    }
    else if (arg < (MatrixFloat)-5.0f)
    {
        result = -DFC_ONE_F;
    }
    else
    {
        /* Linear region: slope = 1/width, but normalised arg already has
         * width divided out, so the effective slope here is 0.2 = 1/5.   */
        result = arg * (MatrixFloat)0.2f;
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_SpeedFusion_Update
 *
 * \brief  Blend encoder and SMO speed estimates into a single fused output.
 *
 * \details Executes the following pipeline each ISR step:
 *
 *          1. ELECTRICAL ANGLE FROM ENCODER
 *             theta_e = p * theta_m   (exact, no filtering)
 *             This is used for Park/InvPark transforms.  The encoder angle is
 *             ground truth -- it does not need filtering.
 *
 *          2. ENCODER SPEED (finite-difference with wrap correction)
 *             delta = theta_m[k] - theta_m[k-1]  (unwrapped to [-pi, pi])
 *             omega_raw = delta / dt
 *             The while loops perform manual modular reduction.  A standard
 *             fmodf call would be cleaner but is not available on all AURIX
 *             toolchains without the floating-point library.
 *
 *          3. ADAPTIVE IIR SMOOTHING ON ENCODER SPEED
 *             iir_coeff blends between IIR_LO (heavy smoothing at low speed,
 *             where quantisation noise is high) and IIR_HI (light smoothing at
 *             high speed, where quantisation noise is low relative to signal).
 *             Larger iir_coeff = faster tracking = less smoothing.
 *
 *          4. SMO PLAUSIBILITY GATE
 *             If |omega_smo_e - omega_enc_e| > DFC_SMO_PLAUS_BAND the SMO
 *             output is implausible (spike, stale value, or divergence) and
 *             is replaced by the encoder-derived electrical speed before the
 *             blend.  This prevents a corrupted SMO estimate from entering
 *             the speed loop even briefly.
 *
 *          5. CONVEX BLEND
 *             omega_e = (1 - alpha) * omega_enc_e + alpha * omega_smo_gated
 *             alpha is computed from the *previous* fused speed (one-step lag)
 *             to avoid an algebraic dependency loop.
 *
 *          6. ENCODER FALLBACK (startup guard)
 *             If the SMO has not yet converged (omega_smo_e ~ 0) but the
 *             encoder shows we are already moving fast, substitute the encoder
 *             electrical speed directly.  This prevents the blend from pulling
 *             omega_e toward zero during the SMO warmup period.
 *
 * \param[in,out] fusion           SpeedFusion state struct.
 * \param[in]     theta_m          Encoder mechanical angle [rad].
 * \param[in]     omega_smo_e      SMO electrical speed [rad/s] (from DFC_SMO_Step).
 * \param[in]     dt               Step period [s].
 * \param[out]    theta_e          Electrical angle for Park transform [rad].
 * \param[out]    omega_e          Fused electrical speed [rad/s].
 * \param[out]    omega_meas_mech  Filtered encoder mechanical speed [rad/s]
 *                                 (used as feedback for the speed P-loop).
 *------------------------------------------------------------------------------------------------------------------*/
static void DFC_SpeedFusion_Update(
    DFC_SpeedFusion_T * const fusion,
    const MatrixFloat         theta_m,
    const MatrixFloat         omega_smo_e,
    const MatrixFloat         dt,
    MatrixFloat             * const theta_e,
    MatrixFloat             * const omega_e,
    MatrixFloat             * const omega_meas_mech)
{
    MatrixFloat delta;          /* Unwrapped angle increment [rad]            */
    MatrixFloat omega_raw;      /* Raw finite-difference speed [rad/s mech]   */
    MatrixFloat alpha;          /* SpeedFusion blend weight [0, 1]            */
    MatrixFloat iir_coeff;      /* Adaptive IIR coefficient for this step     */
    MatrixFloat omega_enc_e;    /* Encoder-derived electrical speed [rad/s]   */

    /* Guard against NULL pointers -- MISRA C:2012 Rule 17.5 */
    if ((fusion == NULL) || (theta_e == NULL) || (omega_e == NULL) || (omega_meas_mech == NULL))
    {
        return;
    }

    /* ---- 1. Electrical angle -- exact, no filtering ---- */
    *theta_e = (MatrixFloat)DFC_P_POLES * theta_m;

    /* ---- 2. Encoder finite-difference speed ---- */
    delta = theta_m - fusion->theta_m_prev;

    /* Unwrap to (-pi, +pi] to handle the 2pi discontinuity when the
     * encoder angle rolls over.  Without this, a single-step wrap produces
     * a spurious omega_raw spike of ±2pi/dt = ±125 664 rad/s at 20 kHz. */
    while (delta > DFC_PI_F)
    {
        delta -= DFC_TWO_PI_F;
    }
    while (delta < -DFC_PI_F)
    {
        delta += DFC_TWO_PI_F;
    }

    /* Guard against dt = 0 (first call or timer fault) to avoid division by zero. */
    omega_raw = (dt > DFC_ZERO_F) ? (delta / dt) : DFC_ZERO_F;

    /* ---- 3. Adaptive IIR on encoder speed ---- */
    /* alpha from *previous* fused speed -- one-step lag avoids algebraic loop.
     * At low speed: alpha ~ 0, iir_coeff = IIR_LO (heavy smoothing, 5% weight on new sample).
     * At high speed: alpha ~ 1, iir_coeff = IIR_HI (lighter smoothing, 30% weight).         */
    alpha     = DFC_FusionAlpha(fabsf(fusion->omega_e_prev));
    iir_coeff = DFC_FUSION_IIR_LO + alpha * (DFC_FUSION_IIR_HI - DFC_FUSION_IIR_LO);

    /* Standard exponential IIR: y[k] = (1-a)*y[k-1] + a*x[k] */
    fusion->omega_enc_filt = ((DFC_ONE_F - iir_coeff) * fusion->omega_enc_filt)
                           + (iir_coeff * omega_raw);

    /* Expose the filtered encoder speed for the speed P-loop feedback. */
    *omega_meas_mech = fusion->omega_enc_filt;

    /* Convert encoder mechanical speed to electrical speed for the blend. */
    omega_enc_e = (MatrixFloat)DFC_P_POLES * fusion->omega_enc_filt;

    /* ---- 4. SMO plausibility gate + 5. Convex blend ---- */
    {
        MatrixFloat omega_smo_gated;

        /* Replace the SMO output with the encoder if the two estimates differ
         * by more than DFC_SMO_PLAUS_BAND.  This catches:
         *   - Single-sample atan2f phase-wrap spikes that survived the omega_e_hat
         *     magnitude clamp inside DFC_SMO_Step (those are rejected at source,
         *     but this is a second line of defence).
         *   - Stale omega_e_filt values from a prior SMO divergence/reinit.    */
        if (fabsf(omega_smo_e - omega_enc_e) > DFC_SMO_PLAUS_BAND)
        {
            omega_smo_gated = omega_enc_e;   /* encoder fallback */
        }
        else
        {
            omega_smo_gated = omega_smo_e;   /* SMO plausible -- MISRA 15.7 */
        }

        /* Convex blend: alpha=0 -> pure encoder; alpha=1 -> pure SMO. */
        *omega_e = ((DFC_ONE_F - alpha) * omega_enc_e) + (alpha * omega_smo_gated);
    }

    /* ---- 6. Encoder fallback during SMO warmup ---- */
    /* During the DFC_SMO_WARMUP_STEPS period omega_smo_e is forced to zero
     * (see DFC_SMO_Step).  If the motor is already spinning (encoder > OMEGA_LO)
     * the blend above would incorrectly pull omega_e toward zero.  Override it
     * with the encoder electrical speed so the speed P-loop sees reality.     */
    if ((fabsf(omega_smo_e) < DFC_ONE_F) &&
        (fabsf(fusion->omega_enc_filt) > DFC_FUSION_OMEGA_LO))
    {
        *omega_e = omega_enc_e;
    }
    else
    {
        /* SMO valid or encoder below threshold -- MISRA 15.7 */
    }

    /* ---- Update persistent state ---- */
    fusion->theta_m_prev   = theta_m;
    fusion->omega_e_prev   = *omega_e;
    fusion->alpha          = alpha;
    fusion->omega_enc_mech = fusion->omega_enc_filt;   /* diagnostic copy */
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_SMO_Step
 *
 * \brief  Execute one step of the Sliding Mode Observer (stationary alphabeta frame).
 *
 * \details The SMO is a reduced-order current model that uses the mismatch between
 *          measured and estimated currents to extract the back-EMF, from which the
 *          electrical angle and speed are derived.
 *
 *          PMSM CURRENT MODEL (alphabeta frame):
 *          =======================================
 *          The full PMSM voltage equation in the alphabeta frame is:
 *
 *            L * di_hat/dt = v - R * i_hat - e_hat
 *
 *          where e_hat = [e_alpha, e_beta]^T is the back-EMF vector.
 *          In an ideal SMO we inject a switching term instead of e_hat:
 *
 *            L * di_hat/dt = v - R * i_hat - K * sat(i - i_hat)
 *
 *          The switching gain K must exceed the maximum back-EMF magnitude:
 *            |e_max| = omega_e_max * lambda_pm = 920 * 0.0014 = 1.29 V
 *          DFC_SMO_K = 2.0 V provides a comfortable margin (see header).
 *
 *          BACK-EMF EXTRACTION:
 *          =====================
 *          In sliding mode (error small) the switching term equals the back-EMF.
 *          Filtering sat(error) through a first-order LPF gives:
 *
 *            e_hat_alpha += alpha_lpf * (sw_alpha - e_hat_alpha)
 *
 *          The LPF corner frequency is 1 / (2*pi * TAU_E) ~ 800 Hz, chosen to
 *          pass the fundamental back-EMF (up to ~1.5 kHz at 3000 RPM) while
 *          suppressing the high-frequency switching residual.
 *
 *          ANGLE AND SPEED EXTRACTION:
 *          ============================
 *          For a surface-mounted PMSM (SPMSM, Ld = Lq):
 *
 *            e_alpha = +omega_e * lambda_pm * sin(theta_e)
 *            e_beta  = -omega_e * lambda_pm * cos(theta_e)
 *
 *          Therefore:
 *
 *            theta_e_hat = atan2(e_alpha, -e_beta)      [sign gives positive omega for CW]
 *            omega_e_hat = delta_theta / dt              [finite-difference on angle]
 *
 *          SPIKE PROTECTION:
 *          ==================
 *          atan2f has a 2*pi discontinuity.  A single wrap causes delta_theta ~ ±2*pi,
 *          so omega_e_hat spikes to ±2*pi/(50 us) = ±125 664 rad/s.  The magnitude
 *          clamp at DFC_SMO_OMEGA_MAX rejects these samples and holds the last filtered
 *          value, letting the LPF decay naturally on the next valid sample.
 *
 *          WARMUP GATE:
 *          =============
 *          For the first DFC_SMO_WARMUP_STEPS = 400 steps (20 ms at 20 kHz) the
 *          back-EMF LPF has not yet converged, so omega_e_smo is forced to zero.
 *          SpeedFusion's encoder fallback (step 6 above) covers this period.
 *
 *          DIVERGENCE GUARD:
 *          ==================
 *          If i_hat drifts beyond 2 * I_MAX the observer has left the sliding
 *          surface (e.g. due to a parameter mismatch or a large transient).
 *          Reinitialising i_hat from i_meas puts the observer back on surface
 *          quickly.  theta_e_prev is intentionally preserved so that the next
 *          finite-difference delta is small rather than a full-revolution jump.
 *
 * \param[in,out] smo          SMO state struct.
 * \param[in]     v_alpha      Alpha-axis voltage one step ago [V] (z-1 delay).
 * \param[in]     v_beta       Beta-axis voltage one step ago [V]  (z-1 delay).
 * \param[in]     i_alpha      Measured alpha current [A].
 * \param[in]     i_beta       Measured beta current [A].
 * \param[in]     dt           Step period [s].
 * \param[in]     warmup_cnt   Steps since controller init -- gates speed output.
 * \param[out]    omega_e_smo  LPF-filtered SMO electrical speed [rad/s].
 *------------------------------------------------------------------------------------------------------------------*/
static void DFC_SMO_Step(
    DFC_SMO_T     * const smo,
    const MatrixFloat     v_alpha,
    const MatrixFloat     v_beta,
    const MatrixFloat     i_alpha,
    const MatrixFloat     i_beta,
    const MatrixFloat     dt,
    const uint32_T        warmup_cnt,
    MatrixFloat         * const omega_e_smo)
{
    MatrixFloat err_alpha, err_beta;   /* Current estimation errors [A]             */
    MatrixFloat sw_alpha,  sw_beta;    /* Scaled switching signals [V]              */
    MatrixFloat inv_L;                 /* 1 / L_avg [1/H]                           */
    MatrixFloat lpf_alpha;             /* First-order LPF coefficient               */
    MatrixFloat theta_e_new, delta;    /* New angle estimate and unwrapped increment */
    const MatrixFloat L_avg = (DFC_L_D + DFC_L_Q) * (MatrixFloat)0.5f;
                               /* Average inductance [H].  Equals L_D for SPMSM.   */

    if ((smo == NULL) || (omega_e_smo == NULL))
    {
        return;
    }

    inv_L = DFC_ONE_F / L_avg;

    /* Tustin-compatible LPF coefficient: alpha = dt / (tau + dt).
     * For TAU_E = 0.2 ms and dt = 50 us: alpha ~ 0.2 (moderate smoothing). */
    lpf_alpha = dt / (DFC_SMO_TAU_E + dt);

    /* ---- Divergence guard ---- */
    /* If estimated current has left the physically possible range the observer
     * is tracking a fictitious operating point.  Reinitialise from measurement
     * so it can re-converge.  Use 2 * I_MAX as the threshold so normal transients
     * (including the demagnetisation peak) do not trigger a spurious reset.     */
    if ((smo->i_hat_alpha > (MatrixFloat)2.0f * DFC_I_MAX) ||
        (smo->i_hat_alpha < -(MatrixFloat)2.0f * DFC_I_MAX) ||
        (smo->i_hat_beta  > (MatrixFloat)2.0f * DFC_I_MAX) ||
        (smo->i_hat_beta  < -(MatrixFloat)2.0f * DFC_I_MAX))
    {
        /* Snap i_hat to measured current -- immediately puts error near zero. */
        smo->i_hat_alpha = i_alpha;
        smo->i_hat_beta  = i_beta;
        /* Clear back-EMF and speed -- they were based on a corrupted trajectory. */
        smo->e_hat_alpha = DFC_ZERO_F;
        smo->e_hat_beta  = DFC_ZERO_F;
        smo->omega_e_hat  = DFC_ZERO_F;
        smo->omega_e_filt = DFC_ZERO_F;
        /* theta_e_prev intentionally preserved -- prevents delta spike */
    }
    else
    {
        /* Within bounds -- MISRA 15.7 */
    }

    /* ---- Current estimation errors ---- */
    err_alpha = i_alpha - smo->i_hat_alpha;
    err_beta  = i_beta  - smo->i_hat_beta;

    /* ---- Switching signals (smooth sign * gain) ---- */
    /* K * sat(err) replaces K * sign(err).  Magnitude = DFC_SMO_K [V].
     * Units check: DFC_SMO_K [V] * dimensionless sat() = [V], which is
     * the correct unit for the back-EMF term in the current ODE.           */
    sw_alpha = DFC_SMO_K * DFC_SMOSwitch(err_alpha);
    sw_beta  = DFC_SMO_K * DFC_SMOSwitch(err_beta);

    /* ---- Current observer -- Forward Euler integration ----
     *
     *   di_hat/dt = (1/L) * (v - R*i_hat - sw)
     *   i_hat[k+1] = i_hat[k] + dt * di_hat/dt
     *
     * v_alpha/v_beta are the *previous step's* voltages (z-1 delay) because
     * the ISR reads ADC currents at the same instant it writes the new PWM duty
     * cycle; the voltage that drove those currents was commanded one step earlier. */
    smo->i_hat_alpha += dt * inv_L * (v_alpha - DFC_R_S * smo->i_hat_alpha - sw_alpha);
    smo->i_hat_beta  += dt * inv_L * (v_beta  - DFC_R_S * smo->i_hat_beta  - sw_beta);

    /* ---- Back-EMF LPF ----
     *
     * In sliding mode the switching signal equals the back-EMF on average.
     * Low-pass filtering it extracts the fundamental component:
     *   e_hat[k+1] = e_hat[k] + alpha * (sw[k] - e_hat[k])
     *
     * Corner frequency: 1 / (2*pi * TAU_E) = 1 / (2*pi * 0.0002) ~ 800 Hz.
     * At 3000 RPM: fundamental back-EMF = 4 poles * 50 rev/s = 200 Hz -- well inside. */
    smo->e_hat_alpha += lpf_alpha * (sw_alpha - smo->e_hat_alpha);
    smo->e_hat_beta  += lpf_alpha * (sw_beta  - smo->e_hat_beta);

    /* ---- Electrical angle from back-EMF vector ----
     *
     * For a SPMSM the back-EMF in alphabeta is:
     *   e_alpha = +omega_e * lambda_pm * sin(theta_e)
     *   e_beta  = -omega_e * lambda_pm * cos(theta_e)
     *
     * atan2(e_alpha, -e_beta) = atan2(sin(theta_e), cos(theta_e)) = theta_e.
     * The negation of e_beta is the key sign: it aligns the angle convention
     * with the encoder (positive theta_m -> positive omega_e).                */
    theta_e_new = atan2f(smo->e_hat_alpha, -smo->e_hat_beta);

    /* ---- Unwrap angle increment ---- */
    /* Same wrap correction as in SpeedFusion -- prevents ±2pi spikes when
     * theta_e_new crosses the atan2f branch cut at ±pi.                     */
    delta = theta_e_new - smo->theta_e_prev;
    while (delta > DFC_PI_F)  { delta -= DFC_TWO_PI_F; }
    while (delta < -DFC_PI_F) { delta += DFC_TWO_PI_F; }

    /* ---- Speed from finite-difference (gated by warmup counter) ---- */
    if ((dt > DFC_ZERO_F) && (warmup_cnt > DFC_SMO_WARMUP_STEPS))
    {
        smo->omega_e_hat = delta / dt;

        /* Spike clamp: reject samples where |omega_e_hat| > DFC_SMO_OMEGA_MAX.
         * These arise from atan2f wrap events where the unwrap loop partially
         * compensates but floating-point rounding leaves a residual impulse.
         * Holding the last filtered value is safe because the LPF will decay
         * it toward the true speed within a few milliseconds.                */
        if ((smo->omega_e_hat > DFC_SMO_OMEGA_MAX) ||
            (smo->omega_e_hat < -DFC_SMO_OMEGA_MAX))
        {
            smo->omega_e_hat = smo->omega_e_filt;   /* hold -- discard spike */
        }
        else
        {
            /* Within plausible range -- MISRA 15.7 */
        }
    }
    else
    {
        /* Warmup active or dt = 0 -- suppress speed output to avoid injecting
         * a transient into SpeedFusion before the back-EMF has settled.      */
        smo->omega_e_hat = DFC_ZERO_F;
    }

    /* ---- LPF on speed ----
     * Uses the same lpf_alpha as the back-EMF filter for consistency.
     * Smooths the noisy finite-difference without introducing a separate
     * time constant that could desynchronise angle and speed.               */
    smo->omega_e_filt += lpf_alpha * (smo->omega_e_hat - smo->omega_e_filt);

    /* ---- Persist angle state ---- */
    smo->theta_e_prev = theta_e_new;   /* for next step's finite-difference */
    smo->theta_e_hat  = theta_e_new;   /* diagnostic access                 */
    *omega_e_smo      = smo->omega_e_filt;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_VoltageLaw
 *
 * \brief  Compute dq-frame voltage references using the differential flatness equations.
 *
 * \details This is the mathematical heart of the DFC controller.
 *
 *          CLASSICAL PI-FOC COMPARISON:
 *          =============================
 *          A standard PI-FOC computes:
 *            vd = Kp_d * (0 - id) + Ki_d * integral(0 - id)
 *            vq = Kp_q * (iq_ref - iq) + Ki_q * integral(iq_ref - iq)
 *
 *          The cross-coupling (omega_e * L * iq in vd; omega_e * lambda in vq)
 *          is left as a disturbance for the integrators to reject.
 *
 *          FLATNESS FEEDFORWARD:
 *          ======================
 *          The DFC pre-computes these cross-coupling terms analytically and
 *          adds them to the output.  The feedback gains only need to correct
 *          model mismatch and disturbances, not fight the steady-state coupling.
 *
 *          D-AXIS VOLTAGE EQUATION (Fix 1 + Fix 3):
 *
 *            vd = -omega_e * Lq * iq_ref          [flatness: cancels q->d coupling]
 *               + Kp_id * (0 - id_meas)           [proportional: id = 0 MTPA]
 *               + id_integral                     [integral Fix 3: eliminates DC bias]
 *
 *          iq_ref (not iq_meas) is used in the decoupling term.  Using iq_meas
 *          was tested and caused speed-loop collapse: ADC noise on iq_meas
 *          propagated through vd -> v_alpha_prev -> SMO -> SpeedFusion.
 *          The residual omega_e*Lq*(iq_meas - iq_ref) is a DC disturbance at
 *          steady state; the id_integral integrator is what eliminates it.
 *
 *          INTEGRAL ACTION (Fix 3):
 *          ==========================
 *          id_integral += DFC_KI_ID * dt * (0 - id_meas)
 *          id_integral  = clamp(id_integral, +/-DFC_ID_INT_LIMIT)
 *          Anti-windup: integrator frozen when vd_out hits the DFC_V_MAX clamp.
 *
 *          Q-AXIS VOLTAGE EQUATION:
 *
 *            vq = R * iq_ref                       [resistive drop at reference current]
 *               + Lq * diq/dt                      [inductive drop for current ramp]
 *               + omega_e * lambda_pm              [back-EMF cancellation]
 *               + Kp_iq * (iq_ref - iq_meas)       [residual error correction]
 *
 *          The first three terms are the "flatness" part: they constitute the exact
 *          vq needed to track iq_ref with zero steady-state error if the model were
 *          perfect.  The Kp_iq term corrects for R error, lambda_pm error, and
 *          disturbances.
 *
 *          VOLTAGE SATURATION -- PRIORITY-BASED (v3.1):
 *          ==============================================
 *          Fix 1 (v3.1): replaced proportional circle clipping with d-axis-priority
 *          saturation.  The old proportional approach attenuated the small vd
 *          correction by the same scale factor as the large vq, collapsing Kp_id
 *          authority precisely when heavy load demands most of the voltage budget.
 *
 *          New approach:
 *            Step 1 -- vd clamped to [-V_MAX, +V_MAX] (full id-correction authority).
 *            Step 2 -- vq_max = sqrt(V_MAX^2 - vd^2)  (remaining headroom).
 *            Step 3 -- vq clamped to [-vq_max, +vq_max].
 *
 *          When not saturated this is identical to the old path (no clamp fires).
 *
 * \param[in]  iq_ref    Q-axis current reference [A].
 * \param[in]  diq_dt    LPF-filtered derivative of iq_ref [A/s].
 * \param[in]  id_meas   Measured d-axis current [A].
 * \param[in]  iq_meas   Measured q-axis current [A].
 * \param[in]  omega_e   Fused electrical speed [rad/s].
 * \param[out] vd        D-axis voltage reference [V].
 * \param[out] vq        Q-axis voltage reference [V].
 *------------------------------------------------------------------------------------------------------------------*/
static void DFC_VoltageLaw(
    DFC_State_T       * const s,
    const MatrixFloat         iq_ref,
    const MatrixFloat         diq_dt,
    const MatrixFloat         id_meas,
    const MatrixFloat         iq_meas,
    const MatrixFloat         omega_e,
    const MatrixFloat         dt,
    MatrixFloat       * const vd,
    MatrixFloat       * const vq)
{
    MatrixFloat vd_out, vq_out, vq_max, vd_sq;
    MatrixFloat id_error;              /* [A]  id tracking error = id_ref - id_meas = 0 - id_meas */
    MatrixFloat vd_unsaturated;        /* [V]  vd before priority clamp (needed for anti-windup)  */

    if ((s == NULL) || (vd == NULL) || (vq == NULL))
    {
        return;
    }

    /* ---- D-axis: decoupling + MTPA enforcement + integral action (Fix 3) ----
     *
     * id_ref = 0 A (MTPA for SPMSM: no reluctance saliency, Ld = Lq).
     *
     * Voltage law:
     *   vd = -omega_e * Lq * iq_ref        [flatness: cancels q->d cross-coupling]
     *      + Kp_id * (0 - id_meas)         [proportional: corrects transient id error]
     *      + id_integral                   [integral Fix 3: eliminates DC id offset]
     *
     * NOTE: iq_ref (not iq_meas) in the decoupling term.  Using iq_meas was tested
     * and caused speed-loop collapse because Park-transform ADC noise propagated
     * through vd -> v_alpha_prev -> SMO -> SpeedFusion.  iq_ref is smooth (speed
     * P-loop output) and does not inject noise.  The residual
     * omega_e*Lq*(iq_meas - iq_ref) is a DC component at steady state; the
     * integrator id_integral is exactly what handles this DC term.              */
    id_error = DFC_ZERO_F - id_meas;   /* [A]  id_ref = 0 (MTPA) */

    vd_out = -(omega_e * DFC_L_Q * iq_ref)     /* [V] cross-coupling cancel  */
             + (DFC_KP_ID * id_error)           /* [V] proportional action    */
             + s->id_integral;                  /* [V] integral action Fix 3  */

    vd_unsaturated = vd_out;           /* save before clamping for anti-windup */

    /* ---- Q-axis: flatness feedforward + residual feedback ---- */
    /* Physical interpretation of each term:
     *   R * iq_ref             : voltage to overcome winding resistance at iq_ref
     *   Lq * diq_dt            : voltage to ramp iq (di/dt across inductance)
     *   omega_e * lambda_pm    : back-EMF cancellation at current speed
     *   Kp_iq * (iq_ref - iq)  : proportional correction for model mismatch + ADC noise */
    vq_out = (DFC_R_S   * iq_ref)
           + (DFC_L_Q   * diq_dt)
           + (omega_e   * DFC_LAMBDA_PM)
           + (DFC_KP_IQ * (iq_ref - iq_meas));

    /* ---- Voltage saturation: d-axis priority, q-axis gets remainder (Fix 1) ----
     *
     * Step 1 -- vd clamped to [-V_MAX, +V_MAX] independently.
     *           Gives the id-correction path full voltage budget regardless of vq.
     *
     * Step 2 -- remaining headroom: vq_max = sqrt(V_MAX^2 - vd^2).
     *           Follows from the circle constraint ||(vd,vq)|| <= V_MAX once vd fixed.
     *
     * Step 3 -- vq clamped to [-vq_max, +vq_max].
     *
     * When not saturated (typical) neither clamp fires; output identical to pre-Fix-1.
     * MISRA C:2012 Rule 15.7: all if-else chains have a final else.           */

    /* Step 1: clamp vd */
    if (vd_out > DFC_V_MAX)
    {
        vd_out = DFC_V_MAX;
    }
    else if (vd_out < -DFC_V_MAX)
    {
        vd_out = -DFC_V_MAX;
    }
    else
    {
        /* vd within range -- no action */
    }

    /* Step 2: remaining headroom for vq */
    vd_sq  = vd_out * vd_out;                          /* [V^2] always >= 0 after clamp */
    vq_max = sqrtf(DFC_V_MAX * DFC_V_MAX - vd_sq);     /* [V]   always >= 0            */

    /* Step 3: clamp vq */
    if (vq_out > vq_max)
    {
        vq_out = vq_max;
    }
    else if (vq_out < -vq_max)
    {
        vq_out = -vq_max;
    }
    else
    {
        /* vq within headroom -- no action */
    }

    /* ---- d-axis integrator update with conditional anti-windup (Fix 3) ----
     *
     * Conditional integration: update id_integral only when vd is NOT saturated.
     * When vd_out = vd_unsaturated the clamp did not fire; the integrator may
     * accumulate.  When they differ the clamp fired; freezing the integrator
     * prevents it from winding up further in the wrong direction.
     *
     * This is equivalent to the "back-calculation" anti-windup structure but
     * simpler: instead of subtracting the saturation error, we just don't add.
     *
     * Additional hard clamp: id_integral bounded to ±DFC_ID_INT_LIMIT = ±2.0 V
     * regardless of saturation, preventing pathological accumulation at startup. */
    if (vd_out == vd_unsaturated)
    {
        /* Not saturated: update integrator */
        s->id_integral += DFC_KI_ID * dt * id_error;

        /* Hard clamp */
        if (s->id_integral > DFC_ID_INT_LIMIT)
        {
            s->id_integral = DFC_ID_INT_LIMIT;
        }
        else if (s->id_integral < -DFC_ID_INT_LIMIT)
        {
            s->id_integral = -DFC_ID_INT_LIMIT;
        }
        else
        {
            /* Within clamp -- no action */
        }
    }
    else
    {
        /* Saturated: freeze integrator (anti-windup) */
    }

    *vd = vd_out;
    *vq = vq_out;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_Init
 *
 * \brief  Initialise all controller state to zero.  Call once before the ISR starts.
 *
 * \details Explicit field-by-field zeroing is used rather than memset so that:
 *            - Each initialisation line is traceable to its struct member.
 *            - MISRA C:2012 Rule 9.1 (variables set before use) is demonstrably
 *              satisfied at code review without relying on linker BSS guarantees.
 *            - Future non-zero initial values (e.g. a pre-loaded theta_m) can
 *              be inserted without restructuring the function.
 *
 *          The coordinate transform blocks (Clarke, Park, InvPark) are initialised
 *          via their own Init functions so that any internal state they carry
 *          (lookup tables, flags) is correctly set up.
 *
 *          (void)dt is used to suppress the "unused parameter" warning from
 *          static analysis tools.  dt is accepted by the API for forward
 *          compatibility (future versions may use it for pre-computation) but
 *          is not needed during zeroing.
 *
 * \param[out] s   Controller state struct (must not be NULL).
 * \param[in]  dt  Nominal sampling period [s] -- reserved, not used.
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_Init(DFC_State_T * const s, const MatrixFloat dt)
{
    if (s == NULL)
    {
        return;
    }

    /*--- SpeedFusion state ---*/
    s->fusion.theta_m_prev   = DFC_ZERO_F;   /* No previous angle at startup          */
    s->fusion.omega_enc_filt = DFC_ZERO_F;   /* IIR filter output starts at rest      */
    s->fusion.omega_e_prev   = DFC_ZERO_F;   /* Previous fused speed starts at rest   */
    s->fusion.alpha          = DFC_ZERO_F;   /* Full encoder weight until speed builds */
    s->fusion.omega_enc_mech = DFC_ZERO_F;   /* Diagnostic copy                       */

    /*--- SMO state ---*/
    s->smo.i_hat_alpha  = DFC_ZERO_F;   /* Current observer initialises at zero     */
    s->smo.i_hat_beta   = DFC_ZERO_F;
    s->smo.e_hat_alpha  = DFC_ZERO_F;   /* Back-EMF LPF initialises at zero         */
    s->smo.e_hat_beta   = DFC_ZERO_F;
    s->smo.theta_e_hat  = DFC_ZERO_F;   /* Angle estimate (overwritten on first step)*/
    s->smo.omega_e_hat  = DFC_ZERO_F;   /* Raw speed (gated by warmup counter)       */
    s->smo.omega_e_filt = DFC_ZERO_F;   /* LPF-smoothed speed                        */
    s->smo.theta_e_prev = DFC_ZERO_F;   /* Previous angle for finite-difference      */

    /*--- d-axis PI integrator (Fix 3) ---*/
    s->id_integral = DFC_ZERO_F;   /* Zero on cold start; Reset() calls Init() */

    /*--- Delayed voltages (z-1 for SMO) ---*/
    /* The SMO uses the voltage commanded in the *previous* step because the
     * ADC samples and the PWM update occur at the same ISR edge.            */
    s->v_alpha_prev = DFC_ZERO_F;
    s->v_beta_prev  = DFC_ZERO_F;

    /*--- Reference trajectory ---*/
    s->iq_ref_prev = DFC_ZERO_F;   /* For iq derivative finite-difference    */
    s->diq_filt    = DFC_ZERO_F;   /* LPF state for diq/dt feedforward term  */

    /*--- Warmup counter ---*/
    s->smo_warmup_cnt = 0U;   /* Counts ISR steps; gates SMO speed output */

    /*--- Coordinate transforms ---*/
    Clarke_Init(&s->clarke_state);
    Park_Init(&s->park_state);
    InvPark_Init(&s->inv_park_state);

    /*--- Diagnostic log ---*/
    s->log_speed_ref = DFC_ZERO_F;
    s->log_iq_ref    = DFC_ZERO_F;
    s->log_id        = DFC_ZERO_F;
    s->log_iq        = DFC_ZERO_F;
    s->log_alpha     = DFC_ZERO_F;
    s->log_omega_e   = DFC_ZERO_F;
    s->log_omega_smo = DFC_ZERO_F;
    s->log_counter   = 0U;
    s->log_next_time = DFC_LOG_INTERVAL;   /* First snapshot at t = 1 ms */

    (void)dt;   /* Reserved for future use -- suppress unused-parameter warning */
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_Step
 *
 * \brief  Execute one complete FOC step.  Called from the 20 kHz GTM ISR.
 *
 * \details The function executes the full signal chain (see file-level diagram)
 *          in a fixed sequence.  All sub-operations are O(1) with no dynamic
 *          allocation and no unbounded loops -- safe for hard real-time use.
 *
 *          EXECUTION SEQUENCE:
 *          ====================
 *          1.  Increment warmup counter (gates SMO speed output).
 *          2.  Clarke transform: abc -> alphabeta currents.
 *          3.  SMO step: alphabeta currents + previous voltages -> omega_smo_e.
 *          4.  SpeedFusion: encoder theta_m + omega_smo_e -> theta_e, omega_e.
 *          5.  Speed P-loop: omega_ref - omega_meas -> iq_ref (clamped).
 *          6.  Current derivative LPF: finite-difference(iq_ref) -> diq_filt.
 *          7.  Park transform: alphabeta currents + theta_e -> id_meas, iq_meas.
 *          8.  Flatness voltage law: iq_ref, diq_filt, id, iq, omega_e -> vd, vq.
 *          9.  Inverse Park: vd, vq + theta_e -> v_alpha, v_beta.
 *          10. Store v_alpha_prev, v_beta_prev for next step's SMO.
 *          11. Diagnostic log snapshot at 1 kHz.
 *
 *          NOTE ON THE Z-1 VOLTAGE DELAY (step 10):
 *          The SMO in step 3 uses s->v_alpha_prev (the voltage from the *previous*
 *          ISR call) rather than the newly computed v_alpha.  This is physically
 *          correct: the ADC captures currents at the start of the ISR while the
 *          previous PWM duty cycle is still active; the new duty cycle only takes
 *          effect after the ISR writes to the GTM compare registers.
 *
 *          CURRENT DERIVATIVE LPF (step 6):
 *          diq/dt is computed as a finite-difference and then smoothed with an
 *          exponential LPF (Tustin coefficient = dt / (tau + dt)).
 *          The magnitude is clamped to I_MAX / DIQ_TAU = 3570 A/s so the
 *          L*diq/dt term in vq cannot request more voltage than the bus can deliver.
 *
 * \param[in,out] s   Controller state (must not be NULL).
 * \param[in]     u   Per-step inputs: speed reference, encoder angle, abc currents.
 * \param[in]     dt  Actual step period [s] from GTM hardware timer.
 * \param[out]    y   Alpha/beta voltage references for SVPWM.
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_Step(
    DFC_State_T        * const s,
    const DFC_Input_T  * const u,
    const MatrixFloat           dt,
    DFC_Output_T       * const y)
{
    MatrixFloat i_alpha, i_beta;                        /* Clarke output [A]                    */
    MatrixFloat id_meas, iq_meas;                       /* Park output [A]                      */
    MatrixFloat theta_e, omega_e, omega_meas_mech;      /* SpeedFusion outputs                  */
    MatrixFloat omega_smo_e;                            /* SMO electrical speed [rad/s]         */
    MatrixFloat speed_err, iq_ref, diq_dt, vd, vq;     /* Intermediate control signals         */
    MatrixFloat lpf_alpha;                              /* Tustin LPF coefficient for diq/dt    */
    const MatrixFloat diq_tau = DFC_DIQ_TAU;            /* LPF time constant [s] -- local alias */

    if ((s == NULL) || (u == NULL) || (y == NULL))
    {
        return;
    }

    /* ---- 1. Warmup counter ---- */
    /* Saturates at UINT32_MAX after ~2.4 days at 20 kHz -- not a concern. */
    s->smo_warmup_cnt++;

    /* ---- 2. Clarke transform: abc -> alphabeta ---- */
    /* Converts three-phase currents (sum-to-zero constraint) into a two-component
     * stationary orthogonal frame.  Clarke is exact for balanced three-phase loads.
     * ic is included (rather than derived from -ia-ib) to catch ADC faults.    */
    Clarke_Step(&s->clarke_state,
                u->ia, u->ib, u->ic,
                &i_alpha, &i_beta);

    /* ---- 3. SMO: always runs, feeds SpeedFusion ---- */
    /* Uses voltages from the *previous* ISR step (z-1 delay -- see header note).
     * Outputs omega_smo_e in rad/s electrical.                                  */
    DFC_SMO_Step(&s->smo,
                 s->v_alpha_prev, s->v_beta_prev,
                 i_alpha, i_beta,
                 dt, s->smo_warmup_cnt,
                 &omega_smo_e);

    /* ---- 4. SpeedFusion: encoder + SMO -> fused angle and speed ---- */
    /* theta_e  : used for Park and InvPark (must be low-noise, encoder-derived).
     * omega_e  : fused electrical speed (used inside VoltageLaw feedforward).
     * omega_meas_mech: IIR-filtered encoder mechanical speed -- speed P-loop feedback. */
    DFC_SpeedFusion_Update(&s->fusion,
                           u->theta_m,
                           omega_smo_e,
                           dt,
                           &theta_e,
                           &omega_e,
                           &omega_meas_mech);

    /* ---- 5. Speed P-loop: error -> iq_ref ---- */
    /* Proportional-only speed controller.  An integrator is deliberately omitted:
     *   - The flatness feedforward (R*iq + omega*lambda) already handles steady-state.
     *   - Adding an integrator risks windup during the SMO warmup transient.
     * Gain DFC_KP_SPEED = 0.4 A/(rad/s): saturates at 30 rad/s (286 RPM) error. */
    speed_err = u->omega_ref_mech - omega_meas_mech;
    iq_ref    = DFC_KP_SPEED * speed_err;
    iq_ref    = DFC_Clamp(iq_ref, DFC_I_MAX);   /* Hard limit to rated current */

    /* ---- 6. Current derivative LPF: diq_ref/dt for flatness feedforward ---- */
    /* Tustin coefficient: alpha = dt / (tau + dt).
     * At dt = 50 us and tau = 1 ms: alpha = 0.048 (heavy smoothing -- appropriate
     * because the finite-difference amplifies any quantisation in iq_ref by 1/dt). */
    lpf_alpha = dt / (diq_tau + dt);

    if (dt > DFC_ZERO_F)
    {
        diq_dt = (iq_ref - s->iq_ref_prev) / dt;   /* Finite-difference [A/s] */
    }
    else
    {
        diq_dt = DFC_ZERO_F;   /* dt = 0 guard -- avoids divide-by-zero */
    }

    /* Exponential IIR on diq_dt */
    s->diq_filt = ((DFC_ONE_F - lpf_alpha) * s->diq_filt) + (lpf_alpha * diq_dt);

    /* Clamp diq_filt so the L*diq term in vq cannot exceed bus voltage headroom.
     * Ceiling: I_MAX / DIQ_TAU = 3.57 / 0.001 = 3570 A/s.
     * At this rate L*diq = 0.000368 * 3570 = 1.31 V, well within the 9.8 V bus margin. */
    s->diq_filt = DFC_Clamp(s->diq_filt, DFC_I_MAX / DFC_DIQ_TAU);

    s->iq_ref_prev = iq_ref;   /* Store for next step's finite-difference */

    /* ---- 7. Park transform: alphabeta -> dq ---- */
    /* Rotates the stationary alphabeta frame into the rotor-synchronous dq frame
     * using theta_e from SpeedFusion.  In the dq frame id is the flux-producing
     * component (driven to zero for MTPA) and iq is the torque-producing component. */
    Park_Step(&s->park_state,
              i_alpha, i_beta, theta_e,
              &id_meas, &iq_meas);

    /* ---- 8. Flatness voltage law: dq voltages ---- */
    /* Computes the DFC voltage commands with feedforward + feedback.
     * See DFC_VoltageLaw for full derivation.                        */
    DFC_VoltageLaw(s, iq_ref, s->diq_filt,
                   id_meas, iq_meas, omega_e, dt,
                   &vd, &vq);

    /* ---- 9. Inverse Park: dq -> alphabeta voltage ---- */
    /* Rotates the dq voltage commands back into the stationary frame for SVPWM.
     * Uses the same theta_e as step 7 -- consistent within one ISR step.       */
    InvPark_Step(&s->inv_park_state,
                 vd, vq, theta_e,
                 &y->v_alpha, &y->v_beta);

    /* ---- 10. Latch voltages for next step's SMO (z-1 delay) ---- */
    /* The SMO in the *next* ISR call will use these values to predict i_hat.   */
    s->v_alpha_prev = y->v_alpha;
    s->v_beta_prev  = y->v_beta;

    /* ---- 11. Diagnostic logging at 1 kHz ---- */
    /* The log is a simple snapshot: no ring buffer, no DMA -- just a set of
     * shadowed variables updated once per millisecond.  The 1 kHz rate is
     * sufficient for speed and current trends while keeping the ISR budget
     * negligible (one floating-point comparison and a few assignments).        */
    if (dt > DFC_ZERO_F)
    {
        s->log_counter++;
        if (((MatrixFloat)s->log_counter * dt) >= s->log_next_time)
        {
            /* Convert omega_ref from rad/s to RPM for operator-friendly display */
            s->log_speed_ref  = u->omega_ref_mech * (MatrixFloat)60.0f / DFC_TWO_PI_F;
            s->log_iq_ref     = iq_ref;
            s->log_id         = id_meas;
            s->log_iq         = iq_meas;
            s->log_alpha      = s->fusion.alpha;
            /* log_omega_e stores the *mechanical* speed driving the P-loop (rad/s). */
            s->log_omega_e    = omega_meas_mech;
            /* log_omega_smo: convert SMO electrical speed to mechanical for comparison. */
            s->log_omega_smo  = omega_smo_e / (MatrixFloat)DFC_P_POLES;
            s->log_next_time += DFC_LOG_INTERVAL;   /* Advance threshold by 1 ms */
        }
        else
        {
            /* Not yet time for next log snapshot */
        }
    }
    else
    {
        /* dt = 0 -- logging disabled */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_Reset
 *
 * \brief  Reset all integrators and dynamic state.  Call on motor stop or fault.
 *
 * \details Delegates to DFC_Controller_Init() to guarantee that the reset
 *          produces exactly the same initial state as a cold start.
 *          Having a single code path for "zero everything" prevents subtle
 *          state residuals that can arise when Reset and Init are maintained
 *          separately and diverge over time.
 *
 *          The dt argument to Init is passed as zero -- it is not used during
 *          initialisation (see DFC_Controller_Init).
 *
 *          Observer mode and blend weight (if any runtime-configurable state is
 *          added in future) should be preserved across a reset so that AURIX
 *          overlay settings survive a fault-recovery restart without the host
 *          needing to retransmit them.
 *
 * \param[in,out] s  Controller state (must not be NULL).
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_Reset(DFC_State_T * const s)
{
    if (s == NULL)
    {
        return;
    }

    /* Preserve any runtime observer configuration here if added in future
     * (e.g. save mode/blend before Init, restore after).                  */

    DFC_Controller_Init(s, DFC_ZERO_F);
}


/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Controller_GetDiagnostics
 *
 * \brief  Read the latest 1 kHz diagnostic snapshot into caller-supplied pointers.
 *
 * \details All seven output pointers are checked simultaneously before any write.
 *          This prevents a partial update (some pointers valid, some NULL) from
 *          leaving the caller's variables in an inconsistent state.
 *
 *          The guard structure (all-or-nothing NULL check) uses a single
 *          compound if-else rather than early returns to satisfy
 *          MISRA C:2012 Rule 15.5 (single point of exit per function).
 *
 * \param[in]  s              Controller state (must not be NULL).
 * \param[out] speed_ref_rpm  Speed reference [RPM].
 * \param[out] iq_ref         Q-axis current reference [A].
 * \param[out] id             Measured d-axis current [A].
 * \param[out] iq             Measured q-axis current [A].
 * \param[out] alpha          SpeedFusion blend weight [0.0, 1.0].
 * \param[out] omega_e        Filtered encoder mechanical speed [rad/s] (P-loop feedback).
 * \param[out] omega_smo      SMO mechanical speed estimate [rad/s].
 *------------------------------------------------------------------------------------------------------------------*/
void DFC_Controller_GetDiagnostics(
    const DFC_State_T * const s,
    MatrixFloat       * const speed_ref_rpm,
    MatrixFloat       * const iq_ref,
    MatrixFloat       * const id,
    MatrixFloat       * const iq,
    MatrixFloat       * const alpha,
    MatrixFloat       * const omega_e,
    MatrixFloat       * const omega_smo)
{
    /* All-or-nothing NULL check: every output pointer must be valid.
     * If any one is NULL the entire read is skipped -- no partial update. */
    if ((s             != NULL) &&
        (speed_ref_rpm != NULL) &&
        (iq_ref        != NULL) &&
        (id            != NULL) &&
        (iq            != NULL) &&
        (alpha         != NULL) &&
        (omega_e       != NULL) &&
        (omega_smo     != NULL))
    {
        *speed_ref_rpm = s->log_speed_ref;
        *iq_ref        = s->log_iq_ref;
        *id            = s->log_id;
        *iq            = s->log_iq;
        *alpha         = s->log_alpha;
        *omega_e       = s->log_omega_e;
        *omega_smo     = s->log_omega_smo;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7 -- else required */
    }
}
