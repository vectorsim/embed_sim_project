/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.h
 * \brief     Differential Flatness FOC Controller -- NANOTEC DB42S02
 *
 * \details   Full-state DFC with SMO + encoder SpeedFusion speed estimation.
 *
 *            ARCHITECTURE SUMMARY
 *            =====================
 *            The controller implements Field-Oriented Control (FOC) using the
 *            Differential Flatness method.  Three cascaded loops execute in
 *            sequence every ISR step (dt = 50 us at 20 kHz):
 *
 *              OUTER LOOP -- Speed [rad/s] -> iq_ref [A]
 *                iq_ref = KP_SPEED * (omega_ref - omega_meas)
 *                Feedback source: SpeedFusion (encoder IIR + SMO blend)
 *
 *              INNER LOOP D -- id [A] -> vd [V]
 *                vd = -omega_e * Lq * iq_ref          [flatness decoupling]
 *                   + KP_ID * (0 - id_meas)           [MTPA enforcement]
 *
 *              INNER LOOP Q -- iq [A] -> vq [V]
 *                vq = R*iq_ref + Lq*diq/dt + omega_e*lambda_pm  [flatness feedforward]
 *                   + KP_IQ * (iq_ref - iq_meas)                [residual correction]
 *
 *            SPEED ESTIMATION -- SpeedFusion
 *            ================================
 *            Two complementary speed sources are blended:
 *
 *              Encoder finite-difference [rad/s mech]:
 *                Accurate at low speed; quantisation-limited at high speed (20 kHz).
 *                Smoothed by an adaptive IIR with coefficient blending between
 *                IIR_LO (heavy) at low speed and IIR_HI (light) at high speed.
 *
 *              SMO electrical speed [rad/s elec]:
 *                Model-based; accurate at high speed once the back-EMF LPF has
 *                converged (~20 ms warmup).  Inherently noisy at low speed where
 *                back-EMF magnitude is small relative to switching ripple.
 *
 *            Blend weight alpha (0 = encoder, 1 = SMO) varies linearly with speed
 *            between DFC_FUSION_OMEGA_LO and DFC_FUSION_OMEGA_HI.
 *            A plausibility gate substitutes the encoder value if the SMO deviates
 *            by more than DFC_SMO_PLAUS_BAND [rad/s elec], providing a second
 *            line of defence against un-clamped SMO spikes.
 *
 *            COORDINATE FRAME CONVENTIONS
 *            ==============================
 *              alpha/beta : stationary two-phase frame (Clarke output)
 *              d/q        : rotor-synchronous frame (Park output)
 *              theta_m    : mechanical angle [rad],  0 to 2*pi
 *              theta_e    : electrical angle [rad] = p * theta_m
 *              omega_m    : mechanical speed [rad/s]
 *              omega_e    : electrical speed [rad/s] = p * omega_m
 *              p          : number of pole pairs = DFC_P_POLES = 4
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types explicit via MatrixFloat / uint32_T.
 *              Rule 10.4  : no mixed-mode arithmetic.
 *              Rule 15.5  : single return per function.
 *              Rule 15.7  : every if-else chain has a final else.
 *
 * \version   3.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_gains.h"


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \defgroup DFC_MotorParams  Motor parameters -- NANOTEC DB42S02
 * \{
 *
 * All motor parameters are compile-time constants because they are physical
 * properties of a fixed hardware motor.  Changing them requires a firmware
 * rebuild and re-validation under ISO 26262.
 *
 * Parameter source: NANOTEC DB42S02 datasheet + hardware commissioning at 17 V.
 */

/**********************************************************************************************************************
 * \brief  Number of pole pairs.
 *
 * \details Converts between mechanical and electrical quantities:
 *            theta_e [rad]   = DFC_P_POLES * theta_m [rad]
 *            omega_e [rad/s] = DFC_P_POLES * omega_m [rad/s]
 *
 *          The DB42S02 has 8 poles (4 pole pairs).  Verify from the motor nameplate
 *          or by counting back-EMF cycles per mechanical revolution with an
 *          oscilloscope: one full electrical cycle per pole pair.
 *
 * \units   dimensionless  [pole pairs]
 *********************************************************************************************************************/
#define DFC_P_POLES          (4U)

/**********************************************************************************************************************
 * \brief  Stator resistance (per phase, line-to-neutral).
 *
 * \details Appears in the flatness q-axis feedforward:
 *            vq_ff [V] += R_S [Ohm] * iq_ref [A]
 *
 *          Measured at 25 degC.  R_S increases approximately 0.4 %/degC with
 *          copper winding temperature; at 100 degC R_S ~ 0.314 Ohm (+10 %).
 *          The KP_IQ feedback term corrects for this thermal drift automatically.
 *
 *          Value: 0.285 Ohm (commissioning measurement; datasheet quotes 0.19 Ohm
 *          line-to-line = 0.095 Ohm per phase for a DELTA winding -- but the
 *          DB42S02 in DELTA configuration presents R_line / 3 to the inverter
 *          in dq frame terms, giving the effective value used here).
 *
 * \units   Ohm  [V/A]
 *********************************************************************************************************************/
#define DFC_R_S              ((MatrixFloat)0.285f)

/**********************************************************************************************************************
 * \brief  D-axis inductance (per phase, line-to-neutral).
 *
 * \details Appears in the SMO current observer denominator:
 *            di_hat/dt [A/s] = (1 / L_avg [H]) * (v [V] - R*i_hat [V] - sw [V])
 *
 *          For a surface-mounted PMSM (SPMSM) Ld = Lq -- there is no reluctance
 *          saliency because the rotor magnet and the air gap have similar permeability.
 *          See also DFC_L_Q.
 *
 *          Value: 3.675e-4 H = 0.3675 mH.
 *
 * \units   H  [V*s/A]
 *********************************************************************************************************************/
#define DFC_L_D              ((MatrixFloat)0.0003675f)

/**********************************************************************************************************************
 * \brief  Q-axis inductance (per phase, line-to-neutral).
 *
 * \details Appears in two flatness feedforward terms:
 *            vd_ff [V]  = -omega_e [rad/s] * L_Q [H] * iq_ref [A]   [cross-coupling cancellation]
 *            vq_ff [V] +=  L_Q [H] * diq_ref/dt [A/s]               [inductive drop]
 *
 *          Equals DFC_L_D for this SPMSM (Ld = Lq = 0.3675 mH).
 *          For an interior PMSM (IPMSM) Lq > Ld and the two constants would differ.
 *
 * \units   H  [V*s/A]
 *********************************************************************************************************************/
#define DFC_L_Q              ((MatrixFloat)0.0003675f)

/**********************************************************************************************************************
 * \brief  Permanent magnet flux linkage (peak, referred to stator).
 *
 * \details Appears in the flatness back-EMF cancellation term:
 *            vq_ff [V] += omega_e [rad/s] * LAMBDA_PM [Wb]
 *
 *          Also determines the SMO switching gain requirement:
 *            |e_max| [V] = omega_e_max [rad/s] * LAMBDA_PM [Wb]
 *                        = 920 * 0.0014 = 1.29 V
 *          DFC_SMO_K = 2.0 V exceeds this, ensuring sliding mode is reached.
 *
 *          Physical interpretation: LAMBDA_PM is the flux that the permanent
 *          magnet links through the stator winding at the d-axis alignment.
 *          The back-EMF amplitude at speed omega_e is E = omega_e * LAMBDA_PM.
 *
 * \units   Wb  [V*s]  (Weber = Volt * second)
 *********************************************************************************************************************/
#define DFC_LAMBDA_PM        ((MatrixFloat)0.0014f)

/**********************************************************************************************************************
 * \brief  Maximum continuous phase current (peak, sinusoidal).
 *
 * \details Used in two places:
 *            1. iq_ref clamp in the speed P-loop:
 *                 iq_ref [A] = clamp(KP_SPEED * speed_err, I_MAX)
 *            2. SMO divergence guard threshold:
 *                 if |i_hat| > 2 * I_MAX [A] --> reinitialise observer
 *
 *          The 2 * I_MAX threshold for the divergence guard allows for transient
 *          overcurrent during acceleration ramps without falsely triggering a
 *          reinitialisation.
 *
 * \units   A  (ampere, peak sinusoidal)
 *********************************************************************************************************************/
#define DFC_I_MAX            ((MatrixFloat)3.57f)

/**********************************************************************************************************************
 * \brief  DC bus voltage.
 *
 * \details Used only to define DFC_V_MAX (the maximum deliverable phase voltage).
 *          The hardware target is 48 V; 17 V is the bench supply used during
 *          initial commissioning of the DB42S02 motor.
 *
 * \units   V  (volt, DC)
 *********************************************************************************************************************/
#define DFC_V_DC             ((MatrixFloat)17.0f)

/**********************************************************************************************************************
 * \brief  Maximum deliverable phase voltage (peak, line-to-neutral).
 *
 * \details For Space Vector PWM (SVPWM) the inscribed circle of the voltage
 *          hexagon has radius:
 *            V_MAX [V] = V_DC [V] / sqrt(3)
 *                      = 17.0 / 1.73205 = 9.81 V
 *
 *          DFC_VoltageLaw scales (vd, vq) proportionally if their vector magnitude
 *          exceeds this limit, keeping the voltage vector inside the hexagon while
 *          preserving the current angle (id/iq ratio).
 *
 *          Note: sqrt(3) is inlined as a literal to avoid a runtime call and to
 *          comply with MISRA C:2012 Rule 21.5 (no <math.h> at file scope).
 *
 * \units   V  (volt, peak line-to-neutral)
 *********************************************************************************************************************/
#define DFC_V_MAX            (DFC_V_DC / ((MatrixFloat)1.73205080757f))

/** \} */  /* end defgroup DFC_MotorParams */


/** \defgroup DFC_FusionParams  SpeedFusion complementary filter parameters
 * \{
 */

/**********************************************************************************************************************
 * \brief  Lower speed threshold for SpeedFusion blend.
 *
 * \details Below this speed, alpha = 0 and the fused output is purely the
 *          encoder-derived speed.  The SMO output is gated out because at low
 *          speed the back-EMF amplitude (omega_e * lambda_pm) is too small for
 *          the observer to distinguish from switching ripple.
 *
 *          At omega_m = 50 rad/s (478 RPM):
 *            back-EMF = 50 * 4 * 0.0014 = 0.28 V  (vs. DFC_SMO_K = 2.0 V switching gain)
 *          The signal-to-noise ratio at this speed is approximately 0.28/2.0 = 14 %,
 *          which is too low for reliable angle extraction.
 *
 * \units   rad/s  (mechanical)
 *********************************************************************************************************************/
#define DFC_FUSION_OMEGA_LO  ((MatrixFloat)50.0f)

/**********************************************************************************************************************
 * \brief  Upper speed threshold for SpeedFusion blend.
 *
 * \details Above this speed, alpha = 1 and the fused output is purely the
 *          (plausibility-gated) SMO speed.  The encoder is still used for
 *          theta_e (electrical angle for Park/InvPark) and for the plausibility
 *          check, but it no longer contributes to the speed value.
 *
 *          At omega_m = 250 rad/s (2387 RPM):
 *            back-EMF = 250 * 4 * 0.0014 = 1.40 V  (70 % of DFC_SMO_K)
 *          Signal-to-noise ratio is high enough for the SMO to track reliably.
 *
 *          Transition band: 50 to 250 rad/s (478 to 2387 RPM).
 *
 * \units   rad/s  (mechanical)
 *********************************************************************************************************************/
#define DFC_FUSION_OMEGA_HI  ((MatrixFloat)250.0f)

/**********************************************************************************************************************
 * \brief  Encoder IIR smoothing coefficient at low speed (alpha = 0).
 *
 * \details Exponential IIR: y[k] = (1 - coeff) * y[k-1] + coeff * x[k].
 *          A coefficient of 0.05 means each new encoder sample contributes
 *          5 % to the filtered output -- heavy smoothing to suppress the
 *          quantisation noise that dominates at low speed.
 *
 *          Effective IIR time constant at low speed:
 *            tau_eff [s] = dt [s] * (1 - IIR_LO) / IIR_LO
 *                        = 50e-6 * 0.95 / 0.05 = 950 us
 *
 * \units   dimensionless  (IIR coefficient, range (0, 1))
 *********************************************************************************************************************/
#define DFC_FUSION_IIR_LO    ((MatrixFloat)0.05f)

/**********************************************************************************************************************
 * \brief  Encoder IIR smoothing coefficient at high speed (alpha = 1).
 *
 * \details Lighter smoothing at high speed because encoder quantisation noise
 *          is smaller relative to the signal.  A coefficient of 0.30 means
 *          30 % weight on the new sample -- faster tracking, lower latency.
 *
 *          Effective IIR time constant at high speed:
 *            tau_eff [s] = dt [s] * (1 - IIR_HI) / IIR_HI
 *                        = 50e-6 * 0.70 / 0.30 = 117 us
 *
 * \units   dimensionless  (IIR coefficient, range (0, 1))
 *********************************************************************************************************************/
#define DFC_FUSION_IIR_HI    ((MatrixFloat)0.30f)

/** \} */  /* end defgroup DFC_FusionParams */


/** \defgroup DFC_SMOParams  Sliding Mode Observer parameters
 * \{
 */

/**********************************************************************************************************************
 * \brief  SMO switching gain.
 *
 * \details The switching gain K [V] must exceed the maximum back-EMF magnitude
 *          to guarantee that the observer reaches and maintains the sliding surface:
 *
 *            |e_max| [V] = omega_e_max [rad/s] * lambda_pm [Wb]
 *                        = (2200 RPM * 2*pi/60 * 4 poles) * 0.0014
 *                        = 920 [rad/s] * 0.0014 [Wb]
 *                        = 1.29 V
 *
 *          DFC_SMO_K = 2.0 V provides a 55 % margin above the worst-case back-EMF.
 *          A larger K reduces convergence time but increases chattering amplitude;
 *          2.0 V is a compromise tuned for the DB42S02 at the 17 V bus.
 *
 * \units   V  (volt)
 *********************************************************************************************************************/
#define DFC_SMO_K            ((MatrixFloat)2.0f)

/**********************************************************************************************************************
 * \brief  SMO back-EMF low-pass filter time constant.
 *
 * \details The back-EMF LPF extracts the fundamental sinusoidal component from
 *          the high-frequency switching signal sw = K * sat(i - i_hat):
 *
 *            e_hat[k+1] = e_hat[k] + alpha_lpf * (sw[k] - e_hat[k])
 *
 *          where alpha_lpf = dt / (TAU_E + dt).
 *
 *          Corner frequency: f_c [Hz] = 1 / (2*pi * TAU_E)
 *                                     = 1 / (2*pi * 0.0002) = 796 Hz
 *
 *          At 3000 RPM the electrical fundamental is:
 *            f_e [Hz] = 3000 / 60 * 4 = 200 Hz
 *          The LPF passes this (796 Hz >> 200 Hz) with a phase lag of:
 *            phi [deg] = -atan(200 / 796) = -14.1 deg
 *          KP_IQ in the voltage law corrects the resulting iq tracking error.
 *
 * \units   s  (second)
 *********************************************************************************************************************/
#define DFC_SMO_TAU_E        ((MatrixFloat)0.0002f)

/**********************************************************************************************************************
 * \brief  SMO warmup step count before speed output is enabled.
 *
 * \details During the first DFC_SMO_WARMUP_STEPS ISR calls the back-EMF LPF has
 *          not yet converged and omega_e_smo is forced to zero.  SpeedFusion's
 *          encoder fallback covers this period.
 *
 *          Warmup duration:
 *            t_warmup [s] = DFC_SMO_WARMUP_STEPS * dt [s]
 *                         = 400 * 50e-6 = 20 ms
 *
 *          At 20 ms the back-EMF LPF (TAU_E = 0.2 ms) has completed
 *            20 ms / 0.2 ms = 100 time constants -- fully converged.
 *
 * \units   dimensionless  (ISR step count)
 *********************************************************************************************************************/
#define DFC_SMO_WARMUP_STEPS (400U)

/**********************************************************************************************************************
 * \brief  SMO electrical speed spike clamp magnitude.
 *
 * \details The finite-difference on theta_e_hat can produce large single-sample
 *          spikes when atan2f crosses its branch cut at ±pi.  Samples whose
 *          magnitude exceeds this limit are discarded and the previous filtered
 *          value is held.
 *
 *          Derivation:
 *            omega_mech_max [rad/s] = 2200 RPM * 2*pi / 60 = 230 rad/s
 *            omega_e_max [rad/s]    = 230 * 4 (pole pairs) = 920 rad/s
 *            Clamp = 3x safety margin: 920 * 3 = 2760 -> rounded up to 3000 rad/s
 *
 *          Any |omega_e_hat| > 3000 rad/s cannot be a genuine motor speed and
 *          must be a numerical artefact from the atan2f discontinuity.
 *
 * \units   rad/s  (electrical)
 *********************************************************************************************************************/
#define DFC_SMO_OMEGA_MAX    ((MatrixFloat)3000.0f)

/**********************************************************************************************************************
 * \brief  SMO plausibility band relative to encoder electrical speed.
 *
 * \details A second line of defence after DFC_SMO_OMEGA_MAX.  If the SMO
 *          electrical speed deviates from the encoder-derived electrical speed
 *          by more than this band, the SMO value is replaced by the encoder
 *          value before the SpeedFusion blend computation.
 *
 *          Derivation:
 *            DFC_SMO_PLAUS_BAND = 4 * DFC_FUSION_OMEGA_HI * DFC_P_POLES
 *                               = 4 * 250 [rad/s mech] * 4 [pole pairs]
 *                               = 4000 rad/s  electrical
 *          ... but the define is set to 1000 rad/s (electrical), equivalent to:
 *            1000 / 4 = 250 rad/s mechanical = 2387 RPM deviation tolerance.
 *          This is approximately 10 % of the omega_e_hat clamp, catching
 *          residual spikes that survive the magnitude clamp but remain
 *          inconsistent with the encoder ground truth.
 *
 * \units   rad/s  (electrical)
 *********************************************************************************************************************/
#define DFC_SMO_PLAUS_BAND   ((MatrixFloat)1000.0f)

/** \} */  /* end defgroup DFC_SMOParams */


/** \defgroup DFC_CtrlParams  Controller timing parameters
 * \{
 */

/**********************************************************************************************************************
 * \brief  Current derivative LPF time constant.
 *
 * \details The finite-difference diq/dt is smoothed by a first-order IIR:
 *            alpha_lpf = dt / (DIQ_TAU + dt)
 *
 *          At dt = 50 us and DIQ_TAU = 1 ms:
 *            alpha_lpf = 50e-6 / (1e-3 + 50e-6) = 0.048  (heavy smoothing)
 *
 *          This is intentionally heavy because finite-differencing amplifies
 *          any quantisation in iq_ref by a factor of 1/dt = 20 000 s^-1.
 *          The LPF corner frequency is 1/(2*pi*DIQ_TAU) = 159 Hz, well below
 *          the electrical fundamental at 3000 RPM (200 Hz) to avoid attenuating
 *          the legitimate diq signal during rapid acceleration.
 *
 *          The filtered derivative is clamped to I_MAX / DIQ_TAU = 3570 A/s
 *          so that the Lq * diq/dt voltage term cannot exceed:
 *            Lq [H] * 3570 [A/s] = 368e-6 * 3570 = 1.31 V
 *          which is within the 9.81 V bus headroom at 17 V.
 *
 * \units   s  (second)
 *********************************************************************************************************************/
#define DFC_DIQ_TAU          ((MatrixFloat)0.001f)

/**********************************************************************************************************************
 * \brief  Diagnostic logging interval.
 *
 * \details The diagnostic snapshot (speed_ref, iq_ref, id, iq, alpha, omega_e,
 *          omega_smo) is updated once per DFC_LOG_INTERVAL seconds.  At 1 ms
 *          the effective log rate is 1 kHz, which is sufficient to capture speed
 *          and current trends while consuming negligible ISR budget.
 *
 *          The log is a simple set of shadowed MatrixFloat variables -- no ring
 *          buffer, no DMA.  The host reads them via DFC_Controller_GetDiagnostics()
 *          at its own rate (typically 10-100 Hz over UART/CAN).
 *
 * \units   s  (second)
 *********************************************************************************************************************/
#define DFC_LOG_INTERVAL     ((MatrixFloat)0.001f)

/** \} */  /* end defgroup DFC_CtrlParams */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \defgroup DFC_Structs  State and I/O structures
 * \{
 */

/**********************************************************************************************************************
 * \struct DFC_SpeedFusion_T
 * \brief  Speed-dependent complementary filter state.
 *
 * \details Blends encoder finite-difference (IIR-smoothed) with the SMO output.
 *          alpha = 0 -> full encoder; alpha = 1 -> full SMO (subject to
 *          plausibility gate).  Operates in the mechanical speed domain;
 *          conversion to electrical speed is performed inside the function.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat theta_m_prev;    /**< Previous encoder mechanical angle [rad].
                                  *   Used to compute finite-difference speed:
                                  *   omega_raw = (theta_m - theta_m_prev) / dt. */

    MatrixFloat omega_enc_filt;  /**< IIR-filtered encoder mechanical speed [rad/s].
                                  *   This is the primary feedback for the speed P-loop
                                  *   and the encoder contribution to the fused speed. */

    MatrixFloat omega_e_prev;    /**< Previous fused electrical speed [rad/s].
                                  *   Used to compute alpha on the *next* step,
                                  *   breaking the algebraic dependency loop.    */

    MatrixFloat alpha;           /**< Current SpeedFusion blend weight [dimensionless, 0..1].
                                  *   0 = pure encoder; 1 = pure SMO.
                                  *   Logged via DFC_Controller_GetDiagnostics(). */

    MatrixFloat omega_enc_mech;  /**< Diagnostic copy of omega_enc_filt [rad/s mech].
                                  *   Exposed for external monitoring; equals
                                  *   omega_enc_filt at all times.               */
} DFC_SpeedFusion_T;


/**********************************************************************************************************************
 * \struct DFC_SMO_T
 * \brief  Sliding Mode Observer state (stationary alphabeta frame).
 *
 * \details All state variables are in SI units and in the alphabeta (stationary)
 *          frame.  The SMO runs every ISR step regardless of the SpeedFusion
 *          blend weight -- its output is always available to SpeedFusion.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat i_hat_alpha;   /**< Estimated alpha-axis current [A].
                                *   Updated by the Euler-discretised current observer:
                                *   i_hat_alpha += dt/L * (v_alpha - R*i_hat_alpha - sw_alpha). */

    MatrixFloat i_hat_beta;    /**< Estimated beta-axis current [A].
                                *   Symmetric to i_hat_alpha on the beta axis.    */

    MatrixFloat e_hat_alpha;   /**< LPF-filtered back-EMF alpha component [V].
                                *   Extracted from the switching signal sw_alpha
                                *   through a first-order IIR (corner ~800 Hz).  */

    MatrixFloat e_hat_beta;    /**< LPF-filtered back-EMF beta component [V].
                                *   Symmetric to e_hat_alpha on the beta axis.    */

    MatrixFloat theta_e_hat;   /**< Estimated electrical angle [rad].
                                *   Computed as atan2(e_hat_alpha, -e_hat_beta).
                                *   Diagnostic; not fed back into the control law
                                *   (SpeedFusion uses the encoder theta_e for Park). */

    MatrixFloat omega_e_hat;   /**< Raw electrical speed estimate [rad/s elec].
                                *   Finite-difference: (theta_e_hat[k] - theta_e_hat[k-1]) / dt.
                                *   Clamped to ±DFC_SMO_OMEGA_MAX before filtering. */

    MatrixFloat omega_e_filt;  /**< LPF-filtered electrical speed [rad/s elec].
                                *   Smoothed with the same IIR alpha as e_hat.
                                *   This is the value passed to SpeedFusion.       */

    MatrixFloat theta_e_prev;  /**< Previous angle for finite-difference [rad].
                                *   Preserved across divergence-guard reinitialisation
                                *   to prevent a large delta on the next step.      */
} DFC_SMO_T;


/**********************************************************************************************************************
 * \struct DFC_Input_T
 * \brief  Per-step input bundle passed to DFC_Controller_Step().
 *
 * \details All fields are in SI units.  The caller (GTM ISR) populates this
 *          struct from hardware sources before each call.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat omega_ref_mech;  /**< Mechanical speed reference [rad/s].
                                  *   Source: host command via AURIX overlay or
                                  *   CAN frame.  Positive = forward direction.   */

    MatrixFloat theta_m;         /**< Encoder mechanical angle [rad], range [0, 2*pi).
                                  *   Source: AURIX GTM TIM capture of quadrature
                                  *   encoder pulses, converted to radians.
                                  *   Used for: electrical angle (Park/InvPark) and
                                  *   encoder finite-difference speed.             */

    MatrixFloat ia;              /**< Phase A current [A], instantaneous.
                                  *   Source: ADC measurement of shunt voltage,
                                  *   scaled by shunt resistance and gain.
                                  *   Sign convention: positive = current flowing
                                  *   into the motor terminal.                     */

    MatrixFloat ib;              /**< Phase B current [A].  Same convention as ia. */

    MatrixFloat ic;              /**< Phase C current [A].  Same convention as ia.
                                  *   Note: for a star-connected motor ia+ib+ic = 0;
                                  *   measuring all three allows ADC fault detection. */
} DFC_Input_T;


/**********************************************************************************************************************
 * \struct DFC_Output_T
 * \brief  Per-step output bundle written by DFC_Controller_Step().
 *
 * \details The caller passes these voltages to the SVPWM modulator, which
 *          converts them to three-phase duty cycles for the GTM PWM channels.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat v_alpha;   /**< Alpha-axis voltage reference [V].
                            *   Output of the Inverse Park transform.
                            *   Range: [-DFC_V_MAX, +DFC_V_MAX].          */

    MatrixFloat v_beta;    /**< Beta-axis voltage reference [V].
                            *   Output of the Inverse Park transform.
                            *   Range: [-DFC_V_MAX, +DFC_V_MAX].          */
} DFC_Output_T;


/**********************************************************************************************************************
 * \struct DFC_State_T
 * \brief  Complete Differential Flatness Controller state.
 *
 * \details Owns all sub-states by value (no heap allocation).  The struct is
 *          intended to be declared as a single static object in the application
 *          layer and passed by pointer to all DFC API functions.
 *
 *          Total size at MatrixFloat = float (4 bytes):
 *            DFC_SpeedFusion_T :  5 fields *  4 = 20 bytes
 *            DFC_SMO_T         :  8 fields *  4 = 32 bytes
 *            Delayed voltages  :  2 fields *  4 =  8 bytes
 *            Reference state   :  2 fields *  4 =  8 bytes
 *            Warmup counter    :  1 field  *  4 =  4 bytes
 *            Transform states  :  ~12 bytes (Clarke + Park + InvPark)
 *            Diagnostic log    :  7 floats + 2 uint32 = 36 bytes
 *            Total             :  ~ 120 bytes
 *********************************************************************************************************************/
typedef struct
{
    /*--- Speed estimation ---*/
    DFC_SpeedFusion_T  fusion;          /**< Complementary filter state [see DFC_SpeedFusion_T]. */
    DFC_SMO_T          smo;             /**< Sliding Mode Observer state [see DFC_SMO_T].        */

    /*--- Delayed voltages for SMO (z-1) ---*/
    MatrixFloat v_alpha_prev;           /**< Alpha voltage commanded in previous ISR step [V].
                                         *   Fed into DFC_SMO_Step as the "applied" voltage
                                         *   because the ADC captures currents while the
                                         *   previous duty cycle is still active.               */

    MatrixFloat v_beta_prev;            /**< Beta voltage commanded in previous ISR step [V].
                                         *   Same z-1 delay rationale as v_alpha_prev.         */

    /*--- Reference trajectory ---*/
    MatrixFloat iq_ref_prev;            /**< Q-axis current reference from previous step [A].
                                         *   Used to compute the finite-difference diq_ref/dt. */

    MatrixFloat diq_filt;               /**< LPF-filtered diq_ref/dt [A/s].
                                         *   Feeds the Lq * diq/dt flatness term in vq.
                                         *   Clamped to I_MAX / DIQ_TAU = 3570 A/s.           */

    /*--- Warmup counter ---*/
    uint32_T smo_warmup_cnt;            /**< ISR steps since DFC_Controller_Init() [dimensionless].
                                         *   Gates SMO speed output: output is zero until
                                         *   smo_warmup_cnt > DFC_SMO_WARMUP_STEPS (400 steps = 20 ms). */

    /*--- Coordinate transforms ---*/
    Clarke_T   clarke_state;            /**< Clarke transform internal state (abc -> alphabeta). */
    Park_T     park_state;              /**< Park transform internal state (alphabeta -> dq).    */
    InvPark_T  inv_park_state;          /**< Inverse Park internal state (dq -> alphabeta).      */

    /*--- Diagnostic logging (1 kHz snapshot) ---*/
    MatrixFloat log_speed_ref;          /**< Speed reference snapshot [RPM].
                                         *   Converted from rad/s: RPM = omega * 60 / (2*pi). */

    MatrixFloat log_iq_ref;             /**< Q-axis current reference snapshot [A].             */

    MatrixFloat log_id;                 /**< Measured d-axis current snapshot [A].              */

    MatrixFloat log_iq;                 /**< Measured q-axis current snapshot [A].              */

    MatrixFloat log_alpha;              /**< SpeedFusion blend weight snapshot [dimensionless].  */

    MatrixFloat log_omega_e;            /**< Filtered encoder mechanical speed [rad/s].
                                         *   This is the omega_meas_mech value that drives
                                         *   the speed P-loop -- not the fused electrical speed. */

    MatrixFloat log_omega_smo;          /**< SMO mechanical speed estimate [rad/s].
                                         *   Derived from omega_e_filt / DFC_P_POLES.
                                         *   Compare with log_omega_e to assess SMO health.    */

    uint32_T    log_counter;            /**< Running ISR step count [dimensionless].
                                         *   Used to determine when the next log snapshot is due. */

    MatrixFloat log_next_time;          /**< Time of next log snapshot [s].
                                         *   Initialised to DFC_LOG_INTERVAL; advanced by
                                         *   DFC_LOG_INTERVAL on each snapshot.               */

} DFC_State_T;

/** \} */  /* end defgroup DFC_Structs */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/** \defgroup DFC_API  Public API
 * \{
 */

/**
 * \brief   Initialise all controller state to zero.  Call once before the ISR starts.
 *
 * \details Performs explicit field-by-field zeroing of DFC_State_T (rather than
 *          memset) so that each member initialisation is individually traceable
 *          at code review and MISRA C:2012 Rule 9.1 compliance is demonstrable
 *          without relying on linker BSS guarantees.
 *
 *          Also initialises the coordinate transform blocks via their own Init
 *          functions (Clarke_Init, Park_Init, InvPark_Init).
 *
 *          The dt parameter is accepted for API forward-compatibility but is
 *          not used during zeroing; (void)dt suppresses the unused-parameter
 *          warning from static analysis tools.
 *
 * \param[out] s   Controller state.  Must not be NULL.
 * \param[in]  dt  Nominal sampling period [s].  Reserved; not used internally.
 */
extern void DFC_Controller_Init(
    DFC_State_T     * const s,
    const MatrixFloat         dt);

/**
 * \brief   Execute one complete FOC step.  Call from the 20 kHz GTM ISR.
 *
 * \details Executes the full signal chain in a fixed, O(1) sequence with no
 *          dynamic memory allocation and no unbounded loops:
 *            1. Increment warmup counter.
 *            2. Clarke: ia [A], ib [A], ic [A]  ->  i_alpha [A], i_beta [A].
 *            3. SMO: v_prev [V], i_meas [A]     ->  omega_smo_e [rad/s elec].
 *            4. SpeedFusion: theta_m [rad]       ->  theta_e [rad], omega_e [rad/s elec].
 *            5. Speed P-loop: omega_ref [rad/s]  ->  iq_ref [A].
 *            6. Current derivative LPF           ->  diq_filt [A/s].
 *            7. Park: i_alpha [A], i_beta [A]    ->  id_meas [A], iq_meas [A].
 *            8. Flatness voltage law             ->  vd [V], vq [V].
 *            9. Inverse Park: vd [V], vq [V]     ->  v_alpha [V], v_beta [V].
 *           10. Latch v_alpha_prev, v_beta_prev for next step's SMO.
 *           11. Diagnostic log snapshot at 1 kHz.
 *
 * \param[in,out] s   Controller state.  Must not be NULL.
 * \param[in]     u   Per-step inputs: omega_ref_mech [rad/s], theta_m [rad],
 *                    ia [A], ib [A], ic [A].  Must not be NULL.
 * \param[in]     dt  Actual step period [s] measured by GTM hardware timer.
 *                    Must be > 0; typically 50 us at 20 kHz.
 * \param[out]    y   Voltage outputs: v_alpha [V], v_beta [V].  Must not be NULL.
 */
extern void DFC_Controller_Step(
    DFC_State_T        * const s,
    const DFC_Input_T  * const u,
    const MatrixFloat           dt,
    DFC_Output_T       * const y);

/**
 * \brief   Reset all integrators and dynamic state.  Call on motor stop or fault.
 *
 * \details Delegates to DFC_Controller_Init() with dt = 0 to guarantee a single
 *          canonical "zero everything" code path.  This prevents subtle state
 *          residuals that arise when Reset and Init diverge over time.
 *
 *          Any runtime observer configuration (gain sets, overlay values) should
 *          be saved before and restored after the call if persistence across a
 *          fault-recovery restart is required.
 *
 * \param[in,out] s  Controller state.  Must not be NULL.
 */
extern void DFC_Controller_Reset(
    DFC_State_T * const s);

/**
 * \brief   Read the latest 1 kHz diagnostic snapshot.
 *
 * \details All seven output pointers are checked simultaneously before any
 *          write.  If any pointer is NULL the entire read is skipped, preventing
 *          a partial update from leaving caller variables in an inconsistent state.
 *
 *          The snapshot is updated by DFC_Controller_Step() once per
 *          DFC_LOG_INTERVAL = 1 ms; the caller may read it at any slower rate.
 *
 * \param[in]  s              Controller state.  Must not be NULL.
 * \param[out] speed_ref_rpm  Speed reference [RPM].
 *                            Conversion: RPM = omega_ref_mech [rad/s] * 60 / (2*pi).
 * \param[out] iq_ref         Q-axis current reference [A].
 * \param[out] id             Measured d-axis current [A].
 * \param[out] iq             Measured q-axis current [A].
 * \param[out] alpha          SpeedFusion blend weight [dimensionless, 0.0 .. 1.0].
 *                            0 = pure encoder, 1 = pure SMO.
 * \param[out] omega_e        Filtered encoder mechanical speed [rad/s].
 *                            This is the omega_meas_mech value driving the speed P-loop.
 * \param[out] omega_smo      SMO mechanical speed estimate [rad/s].
 *                            Derived from SMO electrical speed / DFC_P_POLES.
 *                            Compare with omega_e to assess SMO convergence quality.
 */
extern void DFC_Controller_GetDiagnostics(
    const DFC_State_T * const s,
    MatrixFloat       * const speed_ref_rpm,
    MatrixFloat       * const iq_ref,
    MatrixFloat       * const id,
    MatrixFloat       * const iq,
    MatrixFloat       * const alpha,
    MatrixFloat       * const omega_e,
    MatrixFloat       * const omega_smo);

/** \} */  /* end defgroup DFC_API */

#endif /* EMBED_SIM_DFC_CONTROLLER_H_ */
