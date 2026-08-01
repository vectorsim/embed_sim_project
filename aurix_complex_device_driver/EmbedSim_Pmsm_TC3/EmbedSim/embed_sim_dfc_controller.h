/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.h
 * \brief     Sensorless Differential Flatness FOC Controller — NANOTEC DB42S02.
 *
 * \details   Command input : mechanical speed reference        [RPM]
 *            Sensors       : three phase currents (U, V, W)    [A]
 *            Output        : SVPWM phase duty cycles Ta/Tb/Tc [0-1], mechanical
 *                            speed + rotor position estimates, echoed currents
 *
 *            No position sensor is used.  Rotor angle and speed are estimated by a
 *            Sliding Mode Observer (SMO) on the αβ current error; startup is handled
 *            by an align / open-loop I-f ramp state machine.
 *
 *            OPERATING MODES
 *            ================
 *              DFC_MODE_ALIGN      Rotor pre-positioning: constant current vector at
 *                                  ThetaE = 0 for DFC_ALIGN_TIME_S.  Locks the rotor
 *                                  d-axis to a known angle before the ramp.
 *
 *              DFC_MODE_OPENLOOP   I-f startup: the current vector rotates at a
 *                                  linearly ramped frequency (DFC_OL_ACCEL_E) up to
 *                                  DFC_OL_OMEGA_HANDOVER_E.  The rotor follows
 *                                  synchronously; the SMO converges in parallel.
 *
 *              DFC_MODE_CLOSEDLOOP Full flatness control on the SMO angle:
 *
 *                Reference shaper (2nd order, critically damped):
 *                  AlphaRefF' = Wn^2 * (OmegaCmd - OmegaRefF) - 2*Zeta*Wn*AlphaRefF
 *                  OmegaRefF' = AlphaRefF
 *
 *                Mechanical flatness inversion (flat output y = ThetaM):
 *                  IqFf  = (J * AlphaRefF + B * OmegaRefF) / KT     [A]
 *                  IqRef = IqFf + KpSpeed * (OmegaRefF - OmegaMeas) [A, clamped to I_MAX]
 *
 *                Electrical flatness inversion (voltage law):
 *                  Vd = R*IdRef - OmegaE*Lq*IqRef
 *                     + KpId*(IdRef - IdMeas) + IdIntegral                 [V]
 *                  Vq = R*IqRef + Lq*dIqRef/dt + OmegaE*(Ld*IdRef + LambdaPm)
 *                     + KpIq*(IqRef - IqMeas)                              [V]
 *
 *                d-axis-priority saturation: Vd is clamped to ±V_MAX first;
 *                Vq receives the remaining budget sqrt(V_MAX^2 - Vd^2).
 *                The d-axis integrator freezes while Vd is saturated (anti-windup).
 *
 *            SLIDING MODE OBSERVER
 *            ======================
 *              Current observer (Euler, αβ frame, per axis):
 *                IHat'  = (VPrev - R*IHat - Z) / L            [A/s]
 *                Z      = SMO_K * sat((IHat - IMeas) / SMO_E0)  [V]
 *              Back-EMF extraction:  EHat = LPF(Z),  corner DFC_SMO_LPF_W.
 *              Angle:                ThetaE = atan2(-EAlpha, EBeta).
 *              Speed:                wrapped finite difference of ThetaE,
 *                                    clamped to ±DFC_SMO_OMEGA_MAX_E, then LPF'd.
 *              The observer uses the previous step's commanded voltage (z^-1)
 *              because the ADC samples currents while that duty is still active.
 *
 *            COORDINATE FRAME CONVENTIONS
 *            ==============================
 *              ThetaM [rad] mechanical angle;  ThetaE = P_POLES * ThetaM
 *              OmegaM [rad/s] mechanical;      OmegaE = P_POLES * OmegaM
 *              Clarke/Park via embed_sim_coordinate_transform.h (amplitude-invariant).
 *
 * \note      MISRA C:2012 compliance
 *              Dir   4.11 : library argument validity checked — atan2f is never
 *                           called with both arguments zero (undefined, C99
 *                           7.12.4.4); a zero voltage vector maps to angle 0.
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types explicit via MatrixFloat / uint32_T.
 *              Rule  9.1  : every DFC_Output_T field is written on every path,
 *                           including the safe-duty (0.5) mid-step error path.
 *              Rule 10.4  : no mixed-mode arithmetic.
 *              Rule 15.5  : single return per function.
 *              Rule 15.7  : every if-else-if chain has a final else.
 *
 * \version   4.3.2
 * \date      2026-07-04
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *
 * \par Change history
 *   v4.3.2  Critical bug fixes:
 *           - Corrected SMO current observer equation (R/L² bug, was scaling
 *             resistive term by extra 1/L)
 *           - Removed unused macros DFC_R_LQ, DFC_LD_LAMBDA
 *           - Improved Dfc_WrapTwoPi with while loops (safe for large deltas)
 *           - Added epsilon tolerance for zero voltage vector check
 *           - Added isfinite() checks for NaN protection
 *   v4.3.1  Optimizations: precomputed constants, reduced math calls
 *   v4.3.0  Loop option A/B for GTM integration
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_CONTROLLER_H_
#define EMBED_SIM_DFC_CONTROLLER_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_foc_types.h"
#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_dfc_gains.h"

/**********************************************************************************************************************
 * Macros — Motor Parameters (NANOTEC DB42S02, datasheet values)
 *********************************************************************************************************************/

/** \brief  Number of pole pairs.  ThetaE = DFC_P_POLES * ThetaM.        [pole pairs]   */
#define DFC_P_POLES              (4U)

/** \brief  DFC_P_POLES as MatrixFloat for mixed rad/s conversions.      [pole pairs]   */
#define DFC_P_POLES_F            ((MatrixFloat)4.0f)

/** \brief  Stator resistance, per phase.  Vq feedforward: R * IqRef.    [Ohm]          */
#define DFC_R_S                  ((MatrixFloat)0.19f)

/** \brief  d-axis inductance.  SPMSM: Ld = Lq (no reluctance saliency). [H]            */
#define DFC_L_D                  ((MatrixFloat)0.000125f)

/** \brief  q-axis inductance.  Decoupling: -OmegaE*Lq*IqRef; Lq*dIq/dt. [H]            */
#define DFC_L_Q                  ((MatrixFloat)0.000125f)

/** \brief  Permanent-magnet flux linkage.  Back-EMF: OmegaE * LambdaPm. [Wb]           */
#define DFC_LAMBDA_PM            ((MatrixFloat)0.0014f)

/** \brief  Rotor inertia.  Flatness feedforward: J * AlphaRefF.         [kg*m^2]       */
#define DFC_J_ROTOR              ((MatrixFloat)2.4e-6f)

/** \brief  Viscous friction coefficient.  Feedforward: B * OmegaRefF.   [N*m*s/rad]    */
#define DFC_B_FRIC               ((MatrixFloat)1.0e-6f)

/** \brief  Torque constant KT = 1.5 * P * LambdaPm.                     [N*m/A]        */
#define DFC_KT                   ((MatrixFloat)(1.5f * 4.0f * 0.0014f))

/** \brief  Maximum continuous phase current (peak).  IqRef clamp.       [A]            */
#define DFC_I_MAX                ((MatrixFloat)3.57f)

/** \brief  DC bus voltage (17 V bench supply; 48 V final target).       [V]            */
#define DFC_V_DC                 ((MatrixFloat)17.0f)

/** \brief  Max phase voltage: SVPWM inscribed circle V_DC / sqrt(3).    [V]            */
#define DFC_V_MAX                (DFC_V_DC / ES_MATH_SQRT3_F)

/**
 * \brief   Linear synthesis ceiling of the internal SVPWM stage.       [V]            */
#define DFC_V_LIN                (DFC_V_DC * ES_MATH_HALF_F)

/**
 * \brief   Volts-to-per-unit scaling for the internal SVPWM stage.
 * \units   1 / V
 */
#define DFC_V_TO_PU              (ES_MATH_SQRT3_F / DFC_V_DC)

/**********************************************************************************************************************
 * Macros — Reference Input Conversion
 *********************************************************************************************************************/

/** \brief  RPM -> rad/s (mechanical):  2*pi / 60.                       [rad/(s*RPM)]  */
#define DFC_RPM_TO_RADPS         (ES_MATH_2PI_F / ((MatrixFloat)60.0f))

/** \brief  Mechanical speed command clamp (3000 RPM = 314.16 rad/s).    [rad/s mech]   */
#define DFC_OMEGA_CMD_MAX        ((MatrixFloat)314.159265f)

/**********************************************************************************************************************
 * Macros — Sliding Mode Observer Parameters
 *********************************************************************************************************************/

/** \brief  SMO switching gain.  Must exceed max back-EMF.              [V]            */
#define DFC_SMO_K                ((MatrixFloat)2.5f)

/** \brief  Sigmoid boundary layer of the sat() switching function.      [A]            */
#define DFC_SMO_E0               ((MatrixFloat)1.25f)

/** \brief  Back-EMF extraction LPF corner (~800 Hz).                     [rad/s]        */
#define DFC_SMO_LPF_W            ((MatrixFloat)5000.0f)

/** \brief  Speed estimate LPF corner.                                    [rad/s]        */
#define DFC_SMO_SPEED_LPF_W      ((MatrixFloat)300.0f)

/** \brief  Raw SMO speed plausibility clamp (backstop only).            [rad/s elec]   */
#define DFC_SMO_OMEGA_MAX_E      ((MatrixFloat)6000.0f)

/** \brief  SMO speed innovation clamp.                                  [rad/s elec]   */
#define DFC_SMO_INNOV_MAX_E      ((MatrixFloat)1000.0f)

/** \brief  Observer current divergence guard threshold: 2 * I_MAX.     [A]            */
#define DFC_SMO_I_GUARD          (ES_MATH_TWO_F * DFC_I_MAX)

/** \brief  SMO warmup duration: speed output gated to zero.             [s]            */
#define DFC_SMO_WARMUP_TIME_S    ((MatrixFloat)0.020f)

/** \brief  Epsilon for zero detection and NaN protection.               [dimensionless] */
#define DFC_EPSILON              ((MatrixFloat)1e-6f)

/**********************************************************************************************************************
 * Macros — Load-Torque Observer Parameters
 *********************************************************************************************************************/

/** \brief  Observer bandwidth.  150 rad/s.                              [rad/s]        */
#define DFC_OBS_OMEGA_O          ((MatrixFloat)150.0f)

/** \brief  Speed-error gain L1 = 2 * OMEGA_O (critically damped).       [1/s]          */
#define DFC_OBS_L1               (ES_MATH_TWO_F * DFC_OBS_OMEGA_O)

/** \brief  Torque-error gain L2 = J * OMEGA_O^2.                       [N*m/(rad)]    */
#define DFC_OBS_L2               (DFC_J_ROTOR * DFC_OBS_OMEGA_O * DFC_OBS_OMEGA_O)

/** \brief  Load-torque estimate clamp: the stall torque KT * I_MAX.     [N*m]          */
#define DFC_TL_MAX               (DFC_KT * DFC_I_MAX)

/** \brief  Load-torque estimate slew limit.                             [N*m/s]        */
#define DFC_TL_SLEW_MAX          ((MatrixFloat)0.1f)

/** \brief  Hold-off after the open-loop -> closed-loop handover.        [s]            */
#define DFC_OBS_HOLDOFF_S        ((MatrixFloat)0.50f)

/**********************************************************************************************************************
 * Macros — Startup State Machine Parameters
 *********************************************************************************************************************/

/** \brief  Rotor alignment duration at ThetaE = 0.                     [s]            */
#define DFC_ALIGN_TIME_S         ((MatrixFloat)0.30f)

/** \brief  Alignment / open-loop boost current.                        [A]            */
#define DFC_OL_I_BOOST           ((MatrixFloat)1.5f)

/** \brief  Open-loop electrical acceleration of the I-f ramp.          [rad/s^2 elec] */
#define DFC_OL_ACCEL_E           ((MatrixFloat)400.0f)

/** \brief  Electrical speed at which control hands over to the SMO.    [rad/s elec]   */
#define DFC_OL_OMEGA_HANDOVER_E  ((MatrixFloat)200.0f)

/** \brief  SMO plausibility band for handover.                         [rad/s elec]   */
#define DFC_OL_HANDOVER_BAND_E   ((MatrixFloat)50.0f)

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \enum   DFC_Mode_T
 * \brief  Startup / run state of the sensorless controller.
 */
typedef enum
{
    DFC_MODE_ALIGN      = 0U,   /**< Rotor pre-positioning at ThetaE = 0.       */
    DFC_MODE_OPENLOOP   = 1U,   /**< I-f ramp: rotating current vector.         */
    DFC_MODE_CLOSEDLOOP = 2U    /**< Flatness control on the SMO angle.         */
} DFC_Mode_T;

/**
 * \struct DFC_Smo_T
 * \brief  Sliding Mode Observer state (αβ frame).
 */
typedef struct
{
    FocAlphaBeta_T  IHat;         /**< Estimated stator current           [A]           */
    FocAlphaBeta_T  EHat;         /**< LPF-filtered back-EMF estimate     [V]           */
    MatrixFloat     ThetaE;       /**< Estimated electrical angle         [rad, 0-2pi)  */
    MatrixFloat     ThetaEPrev;   /**< Previous angle (finite difference) [rad]         */
    MatrixFloat     OmegaEFilt;   /**< LPF-filtered electrical speed      [rad/s elec]  */
    MatrixFloat     WarmupTime;   /**< Accumulated time since Init        [s]           */
} DFC_Smo_T;

/**
 * \enum   DFC_LoopOption_T
 * \brief  Caller-selected flatness loop option.
 */
typedef enum
{
    DFC_LOOP_OPENLOOP   = 0U,   /**< Option A: I-f open loop, no SMO handover.  */
    DFC_LOOP_CLOSEDLOOP = 1U    /**< Option B: full flatness closed loop.       */
} DFC_LoopOption_T;

/**
 * \struct DFC_Input_T
 * \brief  Per-step input bundle for DFC_Step().
 */
typedef struct
{
    MatrixFloat       SpeedRefRpm;   /**< Mechanical speed reference      [RPM]         */
    FocUvw_T          PhaseCurrents; /**< Measured phase currents U, V, W [A]           */
    DFC_LoopOption_T  LoopOption;    /**< Loop option A (I-f hold) / B (closed loop).   */
} DFC_Input_T;

/**
 * \struct DFC_Output_T
 * \brief  Per-step output bundle written by DFC_Step().
 */
typedef struct
{
    MatrixFloat    Ta;              /**< Phase-A (U) duty cycle           [0.0 ... 1.0]  */
    MatrixFloat    Tb;              /**< Phase-B (V) duty cycle           [0.0 ... 1.0]  */
    MatrixFloat    Tc;              /**< Phase-C (W) duty cycle           [0.0 ... 1.0]  */
    MatrixFloat    AngularVelocity; /**< SMO mechanical speed estimate    [rad/s mech]   */
    MatrixFloat    RotorPosition;   /**< Integrated mechanical angle      [rad, 0-2pi)   */
    FocUvw_T       PhaseCurrents;   /**< Measured currents; W = -U - V    [A]            */
    DFC_Mode_T     Mode;            /**< Active controller mode.                         */
} DFC_Output_T;

/**
 * \struct DFC_Diag_T
 * \brief  Diagnostic snapshot, refreshed every DFC_Step().
 */
typedef struct
{
    MatrixFloat     OmegaRefF;    /**< Shaped speed reference             [rad/s mech]  */
    MatrixFloat     OmegaMeas;    /**< SMO mechanical speed estimate      [rad/s mech]  */
    MatrixFloat     IqRef;        /**< q-axis current reference           [A]           */
    FocDq_T         IdqMeas;      /**< Measured dq currents               [A]           */
    MatrixFloat     IdIntegral;   /**< d-axis integrator accumulator      [V]           */
    FocDq_T         VDq;          /**< dq voltage reference               [V]           */
    FocAlphaBeta_T  VAlphaBeta;   /**< Alpha-beta voltage ref (InvPark)   [V]           */
    FocAngle_T      Angle;        /**< Electrical angle used this step    [rad, 0-2pi)  */
    SVM_Sector_T    Sector;       /**< Active SVPWM sector                [SVM_Sector_T] */
    MatrixFloat     TLoadHat;     /**< Load-torque observer estimate      [N*m]         */
} DFC_Diag_T;

/**
 * \struct DFC_State_T
 * \brief  Complete controller state.  Owns all sub-states by value; no heap.
 */
typedef struct
{
    /*--- Angle / speed estimation ---*/
    DFC_Smo_T       Smo;          /**< Sliding Mode Observer state.                     */

    /*--- Startup state machine ---*/
    DFC_Mode_T      Mode;         /**< Active mode (ALIGN -> OPENLOOP -> CLOSEDLOOP).   */
    MatrixFloat     TimeInMode;   /**< Time accumulated in the active mode  [s]         */
    MatrixFloat     ThetaOl;      /**< Open-loop commanded angle            [rad, 0-2pi)*/
    MatrixFloat     OmegaOlE;     /**< Open-loop commanded speed            [rad/s elec]*/

    /*--- Reference shaper (closed loop) ---*/
    MatrixFloat     OmegaRefF;    /**< Shaped speed reference               [rad/s mech]*/
    MatrixFloat     AlphaRefF;    /**< Shaped acceleration reference        [rad/s^2]   */

    /*--- Current reference trajectory ---*/
    MatrixFloat     IqRefPrev;    /**< IqRef of previous step               [A]         */
    MatrixFloat     DIqFilt;      /**< LPF-filtered dIqRef/dt               [A/s]       */

    /*--- d-axis integrator ---*/
    MatrixFloat     IdIntegral;   /**< d-axis PI accumulator, clamped to
                                   *   ±DFC_ID_INT_LIMIT; frozen while Vd
                                   *   is saturated (anti-windup)           [V]         */

    /*--- SMO z^-1 voltage latch ---*/
    FocAlphaBeta_T  VPrev;        /**< αβ voltage commanded previous step   [V]         */

    /*--- Mechanical position reconstruction ---*/
    MatrixFloat     ThetaMech;    /**< Integrated mechanical angle from the
                                   *   SMO speed estimate; absolute offset
                                   *   arbitrary, rate exact  [rad, 0-2pi)   */

    /*--- Load-torque observer ---*/
    MatrixFloat     ObsOmega;     /**< Observer mechanical speed state      [rad/s mech]*/
    MatrixFloat     ObsOmegaF;    /**< ObsOmega refiltered at the SMO speed
                                   *   LPF corner.                         [rad/s mech]*/
    MatrixFloat     ObsTLoad;     /**< Load-torque estimate, clamped to
                                   *   ±DFC_TL_MAX, slew-limited            [N*m]       */

    /*--- Runtime gains ---*/
    DFC_GainSet_T   Gains;        /**< Active gain set (defaults on Init).              */

    /*--- Diagnostics ---*/
    DFC_Diag_T      Diag;         /**< Latest snapshot.                                 */
} DFC_State_T;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialize all controller state.  Call once before the ISR starts.
 *
 * \param[out] State_P  Controller state (must not be NULL)
 * \return  MATRIX_SUCCESS or MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_T DFC_Init(
    DFC_State_T          * const State_P);

/**
 * \brief   Execute one complete sensorless FOC + SVPWM step.  Call from the
 *          20 kHz GTM ISR.
 *
 * \param[in,out] State_P  Controller state (must not be NULL)
 * \param[in]     In_P     Speed reference [RPM] + phase currents [A] (must not be NULL)
 * \param[in]     Dt       Step period [s], measured by STM/GTM; must be > 0
 * \param[out]    Out_P    Duties + mechanical estimates + currents (must not be NULL)
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if any pointer is NULL.
 *          MATRIX_ERROR_OUT_OF_BOUNDS if Dt <= 0.
 */
extern MatrixStatus_T DFC_Step(
    DFC_State_T          * const State_P,
    const DFC_Input_T    * const In_P,
    const MatrixFloat            Dt,
    DFC_Output_T         * const Out_P);

/**
 * \brief   Reset all integrators and dynamic state.  Call on motor stop or fault.
 *
 * \param[in,out] State_P  Controller state (must not be NULL)
 * \return  MATRIX_SUCCESS or MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_T DFC_Reset(
    DFC_State_T          * const State_P);

/**
 * \brief   Apply a runtime gain set (HIL retune / gain scheduling).
 *
 * \param[in,out] State_P  Controller state (must not be NULL)
 * \param[in]     Gains_P  New gain set (must not be NULL)
 * \return  MATRIX_SUCCESS or MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_T DFC_GainSet_Apply(
    DFC_State_T          * const State_P,
    const DFC_GainSet_T  * const Gains_P);

/**
 * \brief   Read the latest diagnostic snapshot.
 *
 * \param[in]  State_P  Controller state (must not be NULL)
 * \param[out] Diag_P   Destination snapshot (must not be NULL)
 * \return  MATRIX_SUCCESS or MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_T DFC_GetDiagnostics(
    const DFC_State_T    * const State_P,
    DFC_Diag_T           * const Diag_P);

#endif /* EMBED_SIM_DFC_CONTROLLER_H_ */
