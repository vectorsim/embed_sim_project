/**********************************************************************************************************************
 * \file      embed_sim_dfc_gains.h
 * \brief     Differential Flatness Controller -- tunable gain defaults for NANOTEC DB42S02
 *
 * \details   Defines the three compile-time gain constants and the runtime-configurable
 *            DFC_GainSet_T structure used by DFC_Controller_Step().
 *
 *            GAIN ARCHITECTURE
 *            ==================
 *            The DFC uses three independent gains, each with its own physical unit
 *            determined by the dimensional analysis of the control law it appears in:
 *
 *              iq_ref [A]   = KP_SPEED [A/(rad/s)]  *  speed_err [rad/s]
 *              vd     [V]  += KP_ID    [V/A]         *  id_err    [A]
 *              vq     [V]  += KP_IQ    [V/A]         *  iq_err    [A]
 *
 *            WHY P-ONLY FOR SPEED?
 *            ======================
 *            A classical cascade FOC uses a PI speed controller whose integrator
 *            removes the steady-state speed error.  In the DFC architecture the
 *            flatness feedforward (R*iq_ref + Lq*diq/dt + omega_e*lambda_pm) already
 *            supplies the exact vq required to maintain constant speed against
 *            winding resistance and back-EMF -- there is no persistent speed error
 *            for an integrator to correct.  Adding an integrator risks wind-up
 *            during the 20 ms SMO warmup transient and complicates fault recovery.
 *            The P-only design is deliberate and sufficient for this application.
 *
 *            WHY P+I FOR D-AXIS (Fix 3)?
 *            ================================
 *            KP_ID corrects model mismatch and ADC noise on the d-axis (id).
 *            However the flatness decoupling term -omega_e*Lq*iq_ref leaves a
 *            residual DC disturbance omega_e*Lq*(iq_meas - iq_ref) that a
 *            proportional gain alone cannot eliminate at steady state.
 *            DFC_KI_ID adds a slow integrator (Ti = 3.3 s) that drives id to
 *            zero against this DC load-dependent offset.
 *            KP_IQ remains P-only: the q-axis flatness feedforward is accurate
 *            enough that no DC error persists, and the SMO phase lag is already
 *            corrected by the high KP_IQ bandwidth.
 *
 *            GAIN TUNING GUIDE
 *            ==================
 *            1. KP_SPEED [A/(rad/s)]: start at I_MAX / omega_err_max
 *                 = 3.57 A / 30 (rad/s) = 0.119 A/(rad/s).
 *               Increase until speed step response is acceptably fast.
 *               Hardware commissioning at 17 V bus settled at 0.4 A/(rad/s).
 *               The saturation point then shifts to 3.57 / 0.4 = 8.9 rad/s (85 RPM).
 *
 *            2. KP_ID [V/A]: closed-loop d-axis bandwidth ≈ KP_ID / Ld.
 *               At KP_ID = 0.4 V/A: bandwidth = 0.4 / 368e-6 = 1087 rad/s (173 Hz),
 *               which is 5.4x the maximum electrical frequency at 3000 RPM.
 *               Reduce if ringing on id appears after speed steps.
 *
 *            3. KP_IQ [V/A]: must be large enough to correct the SMO back-EMF
 *               phase lag at maximum speed.  At KP_IQ = 8.0 V/A: bandwidth =
 *               8.0 / 368e-6 = 21 739 rad/s (3460 Hz), 17x the 200 Hz electrical
 *               fundamental at 3000 RPM.
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types are explicit via the MatrixFloat typedef.
 *              Rule 20.10 : no token-pasting operators used.
 *
 * \version   2.1.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_GAINS_H_
#define EMBED_SIM_DFC_GAINS_H_

#include "embed_sim_matrix.h"    /* MatrixFloat = real32_T */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \defgroup DFC_Gains  Compile-time default gains
 * \{
 *
 * These constants are the gain values compiled into the firmware image.
 * They can be overridden at runtime without recompilation via DFC_GainSet_T
 * and an AURIX overlay write -- useful for hardware-in-the-loop gain sweeps.
 */

/**********************************************************************************************************************
 * \brief  Speed proportional gain.
 *
 * \details Control law:
 *            iq_ref [A] = DFC_KP_SPEED [A/(rad/s)] * (omega_ref - omega_meas) [rad/s]
 *
 *          Design derivation:
 *            Choose KP_SPEED so that the rated current I_MAX is demanded at a
 *            speed error of omega_err_max = 30 rad/s (286 RPM):
 *              KP_SPEED = I_MAX / omega_err_max = 3.57 A / 30 (rad/s) = 0.119 A/(rad/s)
 *
 *          Hardware commissioning result:
 *            The theoretical value (0.119 A/(rad/s)) gave sluggish acceleration
 *            on the 17 V bus because the flatness feedforward is not perfectly
 *            matched to the hardware motor parameters.  Increasing to 0.4 A/(rad/s)
 *            gave acceptable step response while remaining stable at 3000 RPM.
 *            The saturation breakpoint shifts to 3.57 A / 0.4 (A/(rad/s)) = 8.9 rad/s
 *            (85 RPM) speed error.
 *
 * \units   A / (rad/s)
 *********************************************************************************************************************/
#define DFC_KP_SPEED  ((MatrixFloat)0.4f)

/**********************************************************************************************************************
 * \brief  D-axis current proportional gain.
 *
 * \details Control law (inside DFC_VoltageLaw):
 *            vd [V] += DFC_KP_ID [V/A] * (0 [A] - id_meas [A])
 *
 *          Physical role:
 *            The d-axis reference is id_ref = 0 A (Maximum Torque Per Amp for a
 *            surface-mounted PMSM with Ld = Lq, zero reluctance saliency).
 *            KP_ID drives id_meas toward zero against cross-coupling from the
 *            q-axis (the flatness decoupling term -omega_e * Lq * iq_ref handles
 *            the dominant coupling; KP_ID corrects only the residual due to
 *            parameter mismatch) and against ADC measurement noise.
 *
 *          Closed-loop d-axis bandwidth:
 *            omega_cl_d [rad/s] = KP_ID [V/A] / Ld [H]
 *                               = 0.4  / 368e-6
 *                               = 1087 rad/s  (173 Hz)
 *            This is 5.4x the maximum electrical frequency (200 Hz at 3000 RPM),
 *            so id settles within one electrical period after any disturbance.
 *
 *          Why not higher?
 *            The flatness decoupling already cancels the dominant d-axis disturbance.
 *            A moderate KP_ID avoids amplifying ADC noise [A_ADC] * KP_ID [V/A]
 *            into vd [V], which would appear as high-frequency ripple on id.
 *
 * \units   V / A
 *********************************************************************************************************************/
#define DFC_KP_ID     ((MatrixFloat)0.4f)

/**********************************************************************************************************************
 * \brief  Q-axis current proportional gain.
 *
 * \details Control law (inside DFC_VoltageLaw):
 *            vq [V] += DFC_KP_IQ [V/A] * (iq_ref [A] - iq_meas [A])
 *
 *          Physical role:
 *            The flatness feedforward supplies:
 *              R [Ohm] * iq_ref [A]              = resistive drop at reference current [V]
 *              Lq [H]  * diq_ref/dt [A/s]        = inductive drop for current ramp [V]
 *              omega_e [rad/s] * lambda_pm [Wb]  = back-EMF cancellation [V]
 *            KP_IQ corrects the residual iq tracking error from:
 *              a) Stator resistance mismatch  (R nominal 0.285 Ohm; tolerance ~20 %)
 *              b) SMO back-EMF LPF phase lag  (corner ~800 Hz; ~14 deg lag at 200 Hz)
 *              c) ADC current measurement noise and gain error
 *
 *          Closed-loop q-axis bandwidth:
 *            omega_cl_q [rad/s] = KP_IQ [V/A] / Lq [H]
 *                               = 8.0 / 368e-6
 *                               = 21 739 rad/s  (3460 Hz)
 *            This is 17x the electrical fundamental at 3000 RPM (200 Hz), so the
 *            SMO back-EMF lag is fully compensated within a small fraction of
 *            one electrical period.
 *
 *          Why KP_IQ >> KP_ID?
 *            The d-axis error is small because the flatness decoupling is accurate
 *            at any speed.  The q-axis error grows with speed because the SMO
 *            back-EMF estimate carries a speed-dependent phase lag.  A higher
 *            KP_IQ is needed to suppress iq error at 3000 RPM without sacrificing
 *            low-speed stability.  The closed-loop bandwidth ratio is:
 *              omega_cl_q / omega_cl_d = KP_IQ / KP_ID = 8.0 / 0.4 = 20x
 *
 * \units   V / A
 *********************************************************************************************************************/
#define DFC_KP_IQ     ((MatrixFloat)8.0f)

/**********************************************************************************************************************
 * \brief  D-axis current integral gain (Fix 3).
 *
 * \details Eliminates the steady-state id offset that the proportional term
 *          DFC_KP_ID alone cannot reject.
 *
 *          Under heavy load the flatness decoupling leaves a residual DC
 *          disturbance on the d-axis.  An integrator is required to drive
 *          id to zero against a DC input.
 *
 *          Value: DFC_KI_ID = DFC_KP_ID * 0.30
 *            = 0.4 * 0.30 = 0.12  V/(A*s)
 *
 *          This gives ~1 decade separation between the P and I crossover
 *          frequencies while converging within the 5-second simulation window.
 *
 *          Integrator time constant:
 *            Ti_id [s] = DFC_KP_ID / DFC_KI_ID = 0.4 / 0.12 = 3.3 s
 *          At 20 kHz the integrator acts on a timescale of ~3.3 s, fast enough
 *          to visibly reduce the load-dependent id offset within 5 seconds while
 *          remaining well separated from the ms-timescale P-loop transients.
 *
 * \units   V / (A * s)  =  V/(A*s)
 *********************************************************************************************************************/
#define DFC_KI_ID     (DFC_KP_ID * (MatrixFloat)0.30f)   /* Ti = 1/0.30 * (1/KP_ID) ~ 3.3 s */

/**********************************************************************************************************************
 * \brief  Magnitude clamp for the d-axis integrator state.
 *
 * \details The id_integral accumulator is clamped to ±DFC_ID_INT_LIMIT [V]
 *          to prevent excessive wind-up during the SMO warmup transient or
 *          during sustained voltage saturation.
 *
 *          Derivation:
 *            At maximum speed and rated iq, the d-axis decoupling residual is:
 *              delta_vd = omega_e_max * Lq * (iq_meas - iq_ref)
 *                       = 838 * 368e-6 * 0.5  [A, worst-case iq error]
 *                       = 0.15 V
 *            A 2.0 V clamp provides 13x margin, ensuring the integrator can
 *            reject the DC offset without saturating the d-axis voltage budget.
 *
 * \units   V  (volt)
 *********************************************************************************************************************/
#define DFC_ID_INT_LIMIT  ((MatrixFloat)2.0f)

/** \} */  /* end defgroup DFC_Gains */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \defgroup DFC_GainSet  Runtime gain structure
 * \{
 */

/**********************************************************************************************************************
 * \struct  DFC_GainSet_T
 * \brief   Runtime-configurable mirror of the three compile-time gain constants.
 *
 * \details Mirrors DFC_KP_SPEED, DFC_KP_ID, and DFC_KP_IQ exactly.
 *          Populate this struct and pass it to DFC_GainSet_Apply() to update
 *          gains without recompilation -- for example from a gain-scheduling
 *          table indexed by operating speed, or via an AURIX overlay write
 *          from a host calibration tool during hardware-in-the-loop testing.
 *
 *          The struct members carry the same physical units as the corresponding
 *          compile-time constants; see each field comment for the unit and
 *          the control law it participates in.
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat kp_speed;    /**< Speed P-gain [A/(rad/s)].
                              *   Law: iq_ref = kp_speed * (omega_ref - omega_meas).
                              *   Nominal: DFC_KP_SPEED = 0.4 A/(rad/s).
                              *   Saturates at I_MAX when speed error = I_MAX / kp_speed
                              *   = 3.57 / 0.4 = 8.9 rad/s (85 RPM).              */

    MatrixFloat kp_id;       /**< D-axis current P-gain [V/A].
                              *   Law: vd += kp_id * (0 - id_meas).
                              *   Nominal: DFC_KP_ID = 0.4 V/A.
                              *   Closed-loop d-axis BW: kp_id / Ld = 1087 rad/s. */

    MatrixFloat kp_iq;       /**< Q-axis current P-gain [V/A].
                              *   Law: vq += kp_iq * (iq_ref - iq_meas).
                              *   Nominal: DFC_KP_IQ = 8.0 V/A.
                              *   Closed-loop q-axis BW: kp_iq / Lq = 21739 rad/s.*/

    MatrixFloat ki_id;       /**< D-axis current I-gain [V/(A*s)] (Fix 3).
                              *   Law: id_integral += ki_id * dt * id_error.
                              *        vd += id_integral.
                              *   Nominal: DFC_KI_ID = DFC_KP_ID * 0.30 = 0.12 V/(A*s).
                              *   Set to 0.0 to disable integral action (P-only). */
} DFC_GainSet_T;

/** \} */  /* end defgroup DFC_GainSet */


#endif /* EMBED_SIM_DFC_GAINS_H_ */
