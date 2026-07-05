/**********************************************************************************************************************
 * \file      embed_sim_dfc_gains.h
 * \brief     Differential Flatness Controller — tunable gain defaults for NANOTEC DB42S02 (sensorless).
 *
 * \details   Defines the compile-time gain constants and the runtime-configurable
 *            DFC_GainSet_T structure consumed by DFC_Step().
 *
 *            GAIN ARCHITECTURE (v4 — sensorless, full flatness feedforward)
 *            ================================================================
 *            The speed loop is no longer P-only on the raw reference.  A 2nd-order
 *            reference shaper produces a smooth speed trajectory OmegaRefF [rad/s]
 *            and its analytic derivative AlphaRefF [rad/s^2].  The mechanical
 *            flatness inversion supplies the exact torque current:
 *
 *              IqFf  [A] = (J * AlphaRefF + B * OmegaRefF) / KT
 *              IqRef [A] = IqFf + KP_SPEED * (OmegaRefF - OmegaMeas)
 *
 *            The current loops keep the v3 structure:
 *
 *              Vd [V]  = R*IdRef - OmegaE*Lq*IqRef
 *                      + KP_ID * (IdRef - IdMeas) + IdIntegral      [P + I]
 *              Vq [V]  = R*IqRef + Lq*dIqRef/dt + OmegaE*(Ld*IdRef + LambdaPm)
 *                      + KP_IQ * (IqRef - IqMeas)                   [P only]
 *
 *            GAIN TUNING GUIDE (R = 0.19 Ohm, Ld = Lq = 0.125 mH)
 *            =====================================================
 *            1. KP_SPEED [A/(rad/s)]: theoretical I_MAX / omega_err_max
 *                 = 3.57 A / 30 (rad/s) = 0.119 A/(rad/s).
 *               With the flatness feedforward carrying the trajectory, the
 *               feedback only corrects model mismatch; 0.10 is sufficient.
 *
 *            2. KP_ID [V/A]: closed-loop d-axis bandwidth = KP_ID / Ld.
 *               At 0.15 V/A: 0.15 / 125e-6 = 1200 rad/s (191 Hz),
 *               6x the 200 Hz electrical fundamental at 3000 RPM.
 *
 *            3. KP_IQ [V/A]: DISCRETE STABILITY governs, not continuous
 *               bandwidth.  The sampled q-loop with the one-step voltage
 *               latch (VPrev / ADC timing) has per-step gain KP_IQ * Dt / Lq;
 *               at 2.5 V/A this equals 2.5 * 50e-6 / 125e-6 = 1.0 — ON the
 *               period-2 stability boundary (SiL-measured: sustained
 *               alternating SMO angle error, duty chatter).  0.8 V/A gives
 *               per-step gain 0.32 and a 6400 rad/s loop, still 5x the
 *               1257 rad/s electrical fundamental at 3000 RPM; the flatness
 *               feedforward carries the trajectory, so feedback stays small.
 *
 *            4. REF_WN [rad/s]: reference shaper natural frequency.  Sets the
 *               closed-loop speed trajectory bandwidth.  40 rad/s gives a
 *               ~100 ms settle without saturating IqRef on a 1500 RPM step:
 *                 Iq_peak ~ (J * WN^2 * omega_step) / (2 * KT)  (zeta = 1)
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types are explicit via the MatrixFloat typedef.
 *              Rule 20.10 : no token-pasting operators used.
 *
 * \version   4.1.0
 * \date      2026-07-04
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_GAINS_H_
#define EMBED_SIM_DFC_GAINS_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_matrix.h"    /* MatrixFloat = real32_T */

/**********************************************************************************************************************
 * Macros — Compile-Time Default Gains
 *********************************************************************************************************************/

/**
 * \brief   Speed proportional gain.
 * \details Law: IqFb [A] = DFC_KP_SPEED * (OmegaRefF - OmegaMeas) [rad/s mech].
 *          Corrects residual error left by the mechanical flatness feedforward
 *          (J/B/KT mismatch, unmodelled load torque).
 * \units   A / (rad/s)
 */
#define DFC_KP_SPEED         ((MatrixFloat)0.10f)

/**
 * \brief   D-axis current proportional gain.
 * \details Law: Vd [V] += DFC_KP_ID * (IdRef - IdMeas) [A].
 *          Closed-loop d-axis bandwidth = KP_ID / Ld = 0.15 / 125e-6 = 1200 rad/s.
 * \units   V / A
 */
#define DFC_KP_ID            ((MatrixFloat)0.15f)

/**
 * \brief   Q-axis current proportional gain.
 * \details Law: Vq [V] += DFC_KP_IQ * (IqRef - IqMeas) [A].
 *          v4.1.0: 2.5 -> 0.8.  Per-step discrete loop gain KP_IQ * Dt / Lq
 *          must stay well below 1.0 with the one-step voltage latch; 2.5 sat
 *          exactly on the boundary (measured period-2 SMO/current chatter).
 *          0.8 -> per-step gain 0.32, loop corner 6400 rad/s (5x the
 *          electrical fundamental at 3000 RPM).
 * \units   V / A
 */
#define DFC_KP_IQ            ((MatrixFloat)0.8f)

/**
 * \brief   D-axis current integral gain.
 * \details Law: IdIntegral [V] += DFC_KI_ID * (IdRef - IdMeas) * dt;  Vd += IdIntegral.
 *          Removes the DC d-axis residual left by the decoupling term under load.
 *          Ti = KP_ID / KI_ID = 1 / 0.30 * (1 / KP_ID); ~1 decade below the P crossover.
 *          Set the runtime mirror KiId to 0.0f to fall back to P-only.
 * \units   V / (A*s)
 */
#define DFC_KI_ID            (DFC_KP_ID * (MatrixFloat)0.30f)

/**
 * \brief   Magnitude clamp for the d-axis integrator accumulator.
 * \details Prevents wind-up during SMO warmup and sustained voltage saturation.
 *          Worst-case decoupling residual at 3000 RPM:
 *            OmegaEMax * Lq * dIq = 1257 * 125e-6 * 0.5 = 0.079 V  -> 2.0 V is >25x margin.
 * \units   V
 */
#define DFC_ID_INT_LIMIT     ((MatrixFloat)2.0f)

/**
 * \brief   Reference shaper natural frequency.
 * \details 2nd-order critically damped trajectory filter on the speed command:
 *            AlphaRefF' = WN^2 * (OmegaCmd - OmegaRefF) - 2*ZETA*WN*AlphaRefF
 *            OmegaRefF' = AlphaRefF
 *          Provides the smooth OmegaRefF and analytic AlphaRefF consumed by the
 *          mechanical flatness inversion (no noisy numerical differentiation).
 * \units   rad/s
 */
#define DFC_REF_WN           ((MatrixFloat)40.0f)

/**
 * \brief   Reference shaper damping ratio.
 * \details 1.0 = critically damped: no overshoot on the speed trajectory.
 * \units   dimensionless
 */
#define DFC_REF_ZETA         ((MatrixFloat)1.0f)

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \struct  DFC_GainSet_T
 * \brief   Runtime-configurable mirror of the compile-time gain constants.
 *
 * \details Populate and pass to DFC_GainSet_Apply() to retune without a rebuild
 *          — e.g. from a gain-scheduling table or an AURIX overlay write during
 *          hardware-in-the-loop commissioning.  Units match the corresponding
 *          compile-time constants.
 */
typedef struct
{
    MatrixFloat  KpSpeed;   /**< Speed P-gain              [A/(rad/s)]  (DFC_KP_SPEED) */
    MatrixFloat  KpId;      /**< d-axis current P-gain     [V/A]        (DFC_KP_ID)    */
    MatrixFloat  KpIq;      /**< q-axis current P-gain     [V/A]        (DFC_KP_IQ)    */
    MatrixFloat  KiId;      /**< d-axis current I-gain     [V/(A*s)]    (DFC_KI_ID)    */
    MatrixFloat  RefWn;     /**< Reference shaper wn       [rad/s]      (DFC_REF_WN)   */
    MatrixFloat  RefZeta;   /**< Reference shaper zeta     [-]          (DFC_REF_ZETA) */
} DFC_GainSet_T;

#endif /* EMBED_SIM_DFC_GAINS_H_ */
