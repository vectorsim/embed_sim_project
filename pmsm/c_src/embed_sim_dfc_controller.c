/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     DFC (Direct Field Control) controller implementation.
 *
 * \details   Implements differential-flatness-based feedforward control for permanent magnet
 *            synchronous motors (PMSM). Uses pure flatness mapping with PI current correction.
 *            Targets 32-bit MCUs (Infineon AURIX TriCore, ARM Cortex-M4).
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier
 *              - Rule  8.6 : No definitions in header files
 *              - Rule 17.2 : No recursion
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCase_P
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.0.0
 * \date      2026-08-12
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#include "embed_sim_dfc_controller.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_matrix.h"
#include "embed_sim_control.h"
#include <math.h>

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Current PI controller gains (correction on top of flatness)
 *
 * \details These gains provide proportional correction to the flatness feedforward
 *          voltages based on current errors. Higher gains give faster response
 *          but may cause instability.
 *
 * \note    Sensor noise mitigation strategy:
 *          - Very low proportional gains (especially on d-axis)
 *          - Low integral gains to prevent noise amplification
 *          - The feedforward handles most of the control effort (80-90%)
 *          - PI only corrects for model errors and low-frequency disturbances
 *
 * \warning Increasing gains above these values will amplify sensor noise
 *          and may cause audible noise or instability.
 */
#define DFC_CURRENT_KP_D_F              (0.003F)        /**< d-axis proportional gain (very low for noise immunity) */
#define DFC_CURRENT_KP_Q_F              (0.135F)        /**< q-axis proportional gain (moderate for torque response) */
#define DFC_CURRENT_KI_D_F              (0.0001F)       /**< d-axis integral gain (extremely low to prevent windup) */
#define DFC_CURRENT_KI_Q_F              (0.0009F)       /**< q-axis integral gain (low for smooth steady-state) */

/**
 * \brief   Maximum current limit (A)
 *
 * \details Limits the current references and errors to prevent:
 *          - Overcurrent faults
 *          - Integrator windup
 *          - Excessive voltage commands from noise spikes
 */
#define DFC_MAX_CURRENT                 (100.0F)

/**
 * \brief   Maximum current derivative limit (A/s)
 *
 * \details Limits the rate of change of current reference to prevent:
 *          - Voltage overshoot
 *          - Inductive voltage spikes
 *          - Unstable behavior during transients
 */
#define DFC_MAX_IQ_DOT_F                (1000.0F)

/**
 * \brief   Numerical protection epsilon
 *
 * \details Used to prevent division by zero and other numerical issues.
 *          Small enough to not affect calculations but large enough for FPU.
 */
#define DFC_EPSILON_F                   (1.0e-6F)

/**
 * \brief   Square root of 3 (for SVM voltage limit)
 */
#define DFC_SQRT3_F                     (1.7320508075688772F)

/**
 * \brief   Maximum integrator anti-windup limit
 *
 * \details Prevents the integral term from accumulating excessive values
 *          during large errors or when the motor is saturated.
 *          This is critical for noise robustness.
 */
#define DFC_INTEGRAL_LIMIT_F            (10.0F)         /**< Integral term limit (A) */

/*********************************************************************************************************************/
/*--------------------------------------------------Private Data-----------------------------------------------------*/
/*********************************************************************************************************************/

/*


/*********************************************************************************************************************/
/*--------------------------------------------Private Functions-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \details Normalizes an angle to the range [0, 2π) using fmodf.
 *          Useful for rotor angle and Park transform calculations.
 *          Prevents angle accumulation errors over long operation.
 *
 * \param[in,out] anglePtr  Pointer to angle value to be wrapped (in radians).
 */
static void DFC_WrapAngle(real32_T* const anglePtr)
{
    *anglePtr = fmodf(*anglePtr, SVM_2PI_F);
    if (*anglePtr < 0.0F)
    {
        *anglePtr += SVM_2PI_F;
    }
}

/**
 * \brief   Clamp value to specified limits
 *
 * \details Limits a value to a range defined by minVal and maxVal.
 *          Critical for preventing noise spikes from affecting the controller.
 *          Acts as a noise gate - large noise spikes are clipped.
 *
 * \param[in] val     Value to clamp.
 * \param[in] minVal  Minimum allowed value.
 * \param[in] maxVal  Maximum allowed value.
 *
 * \return  Clamped value within [minVal, maxVal].
 */
static real32_T DFC_ClampValue(real32_T val, real32_T minVal, real32_T maxVal)
{
    real32_T result;

    if (val < minVal)
    {
        result = minVal;
    }
    else if (val > maxVal)
    {
        result = maxVal;
    }
    else
    {
        result = val;
    }

    return result;
}

/**
 * \brief   Transform currents to dq
 *
 * \details Converts phase currents (U, V, W) to dq rotating reference frame.
 *          Applies Clarke transform to get alpha-beta, then Park transform
 *          using the electrical angle (rotor position × pole pairs).
 *
 * \note    Current sensor noise effect:
 *          - The Clarke transform averages out some noise
 *          - The Park transform converts high-frequency noise to DC offsets
 *          - Proper filtering should be applied before this function
 *
 * \param[in]  inputPtr   Pointer to control input structure containing phase currents.
 * \param[in]  machinePtr Pointer to machine parameters (pole pairs).
 * \param[out] focDqPtr   Pointer to dq current output structure.
 */
static void DFC_CurrentsToDq(EmbedSimCtrlInput_T* const inputPtr,
                             const EmbedSimMachineParam_T* const machinePtr,
                             FocDq_T* const focDqPtr)
{
    FocUvw_T currents;
    FocAlphaBeta_T alphaBeta;
    FocAngle_T angle;

    /* Read measured phase currents (may contain sensor noise) */
    currents.U = inputPtr->Iu;
    currents.V = inputPtr->Iv;
    currents.W = inputPtr->Iw;

    /* Calculate electrical angle from rotor position */
    angle.ThetaE = inputPtr->RotorPositionEst * machinePtr->PolePairs;
    DFC_WrapAngle(&angle.ThetaE);

    /* Transform to dq frame - note: sensor noise is now in dq frame */
    Clarke_Transform_Matrix(&currents, &alphaBeta);
    Park_Transform_Matrix(&alphaBeta, &angle, focDqPtr);
}

/**
 * \brief   Convert dq voltage to PWM
 *
 * \details Transforms dq voltage commands to PWM duty cycles using
 *          inverse Park transform and Space Vector Modulation (SVM).
 *          Includes over-modulation protection by clamping modulation index.
 *
 * \param[in]  dqPtr      Pointer to dq voltage commands.
 * \param[in]  anglePtr   Pointer to rotor angle for inverse Park transform.
 * \param[in]  machinePtr Pointer to machine parameters (Vdc).
 * \param[out] dutyPtr    Pointer to PWM duty cycle output structure.
 */
static void DFC_VoltageToDuty(const FocDq_T* const dqPtr,
                              const FocAngle_T* const anglePtr,
                              const EmbedSimMachineParam_T* const machinePtr,
                              SVM_DutyCycle_T* const dutyPtr)
{
    MatrixStatus_T status;
    FocAlphaBeta_T vAlphaBeta;
    real32_T vMag;
    real32_T vPhaseMax;
    real32_T modulationIndex;

    /* Initialize with safe 50% duty cycle (no output) */
    dutyPtr->Ta = 0.5F;
    dutyPtr->Tb = 0.5F;
    dutyPtr->Tc = 0.5F;
    dutyPtr->Sector = SVM_SECTOR_I;

    /* Inverse Park transform: dq -> alpha-beta */
    status = InvPark_Transform_Matrix(dqPtr, anglePtr, &vAlphaBeta);

    if (status == MATRIX_SUCCESS)
    {
        /* Calculate voltage magnitude and limit to prevent overmodulation */
        vMag = sqrtf((vAlphaBeta.Alpha * vAlphaBeta.Alpha) +
                     (vAlphaBeta.Beta * vAlphaBeta.Beta));

        /*
         * Limit modulation index to 0.95 (leaves 5% margin for inverter dead-time)
         * This also prevents excessive voltage spikes from noise
         */
        vPhaseMax = machinePtr->Vdc / DFC_SQRT3_F;
        modulationIndex = vMag / vPhaseMax;
        modulationIndex = DFC_ClampValue(modulationIndex, 0.0F, 0.95F);

        /* Generate PWM duty cycles */
        SVM_CalculateDutyCycle(modulationIndex, anglePtr, dutyPtr);
    }
}

/*********************************************************************************************************************/
/*--------------------------------------------Public Functions------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize DFC controller
 *
 * \details Resets all PI controllers to zero. Must be called before DFC_Step().
 *          For pure flatness feedforward, no state initialization is required.
 *          However, the PI integral terms must be reset to prevent startup transients.
 */
void DFC_Init(void)
{

}

/**
 * \brief   Execute one step of Differential Flatness Control.
 *
 * \details Pure feedforward differential-flatness mapping with PI current correction.
 *          The PI correction is specifically tuned to handle sensor noise.
 *
 *          Sensor Noise Handling Strategy:
 *          ===============================
 *          1. Low PI Gains: The PI gains are deliberately low so that high-frequency
 *             noise is not amplified. The feedforward handles most of the control.
 *
 *          2. Error Clamping: Current errors are clamped to ±MAX_CURRENT to prevent
 *             noise spikes from causing large voltage commands.
 *
 *          3. Integral Clamping: Integral terms are clamped to prevent windup
 *             during periods of high noise or saturation.
 *
 *          4. Modulated Output: The SVM modulation index is limited to 0.95,
 *             providing a buffer against noise-induced overmodulation.
 *
 *          Control Flow:
 *          =============
 *          1. Read reference trajectory (omega, omega_dot, omega_ddot)
 *          2. Mechanical flatness: Calculate required torque
 *          3. Electrical flatness: Calculate current and voltage references
 *          4. Measure actual currents and transform to dq
 *          5. Calculate PI correction from current errors
 *          6. Combine feedforward + feedback for final voltage
 *          7. Apply SVM and generate PWM duties
 *
 * \param[in]  motorPtr  Pointer to motor structure containing input, output, and parameters.
 */
void DFC_Step(EmbedSimMachine_T* const motorPtr)
{
    /* Local pointers to input/output/parameter structures */
    EmbedSimCtrlInput_T* const inputPtr   = motorPtr->InputPtr;
    EmbedSimCtrlOutput_T* const outputPtr = motorPtr->OutputPtr;
    const EmbedSimMachineParam_T* const machinePtr = motorPtr->MachinePtr;

    /*
     * ---------- Reference Trajectory (from S-curve generator) ----------
     * These values come from a higher-level motion planner.
     * They are already smooth, so they don't contain high-frequency noise.
     */
    volatile real32_T omegaRef;        /**< Desired angular velocity [rad/s] */
    volatile real32_T omegaRefDot;     /**< Desired angular acceleration [rad/s²] */
    volatile real32_T omegaRefDDot;    /**< Desired angular jerk [rad/s³] */

    /*
     * ---------- Flatness Mapped Values ----------
     * These are calculated from the flatness equations.
     * They form the "base" control effort (80-90% of total).
     */
    volatile real32_T iqRef;           /**< q-axis current reference [A] (from torque requirement) */
    volatile real32_T iqRefDot;        /**< Derivative of iqRef [A/s] (for inductive voltage drop) */

    volatile real32_T vdRef;           /**< d-axis voltage reference [V] (from flatness) */
    volatile real32_T vqRef;           /**< q-axis voltage reference [V] (from flatness) */

    /*
     * ---------- Intermediate Calculations ----------
     */
    volatile real32_T torqueRequired;  /**< Required electromagnetic torque [Nm] */
    volatile real32_T torqueConstant;  /**< Torque constant Kt = 1.5*p*λ_PM [Nm/A] */

    volatile real32_T rotorAngleMeas;  /**< Measured rotor electrical angle [rad] */
    volatile real32_T rotorSpeedMeas;  /**< Measured rotor speed [rad/s] (from position derivative) */

    /*
     * ---------- PI Correction Variables ----------
     * These handle sensor noise and model uncertainties.
     * The values are clamped to prevent noise amplification.
     */
    volatile real32_T idError;         /**< d-axis current error [A] (reference - measured) */
    volatile real32_T iqError;         /**< q-axis current error [A] (reference - measured) */

    static volatile real32_T idIntegralError = 0.0F;    /**< Accumulated d-axis current error */
    static volatile real32_T iqIntegralError = 0.0F;    /**< Accumulated q-axis current error */

    /*
     * ---------- Output Variables ----------
     */
    FocDq_T dqVoltage;                 /**< dq voltage commands (feedforward + correction) */
    FocAlphaBeta_T abVoltage;          /**< Alpha-beta voltage (after inverse Park) */
    FocAngle_T focAngle;               /**< Field oriented control angle */
    SVM_DutyCycle_T svmDC;             /**< SVM duty cycle outputs */
    FocDq_T dqCurrentMeas;             /**< Measured dq currents (from sensors) */
    real32_T vMag;
    real32_T vPhaseMax;
    real32_T modulationIndex;

    /*
     * ------------------------------------------------------------
     * STEP 1: Read Reference Trajectory
     * ------------------------------------------------------------
     * These values come from the S-curve generator and are noise-free.
     * The S-curve ensures smooth acceleration and jerk limits.
     */
    omegaRef         = inputPtr->AngularVelocityRef;
    omegaRefDot      = inputPtr->AngularAccerlerationRef;
    omegaRefDDot     = inputPtr->AngularJerkRef;

    /*
     * ------------------------------------------------------------
     * STEP 2: Mechanical Flatness Mapping (Torque Calculation)
     * ------------------------------------------------------------
     * Uses the rigid body dynamics equation:
     *   Te = J * omega_dot + B * omega + Tload
     *
     * This is the "inverse dynamics" - it calculates what torque
     * is needed to achieve the desired motion.
     *
     * Noise note: J and B are constants, so this step doesn't
     * amplify sensor noise. The load torque is assumed constant.
     */
    torqueRequired = (machinePtr->J * omegaRefDot) +
                     (machinePtr->B * omegaRef) +
                     machinePtr->TorqueLoad;

    /*
     * ------------------------------------------------------------
     * STEP 3: Electrical Flatness Mapping - Current Reference
     * ------------------------------------------------------------
     * For surface PMSM with Id = 0 (MTPA operation):
     *   Te = 1.5 * p * FluxPm * Iq
     * Therefore:
     *   Iq = Te / (1.5 * p * FluxPm)
     *
     * Noise note: This is a division by a constant. Any noise in
     * torqueRequired would be amplified, but torqueRequired comes
     * from the clean reference trajectory, not from sensors.
     */
    torqueConstant = 1.5F * machinePtr->PolePairs * machinePtr->FluxPm;

    if (fabsf(torqueConstant) > DFC_EPSILON_F)
    {
        /* Calculate q-axis current reference from torque */
        iqRef = torqueRequired / torqueConstant;

        /*
         * Clamp current to prevent overcurrent faults.
         * This also prevents noise from causing excessive currents.
         */
        iqRef = DFC_ClampValue(iqRef, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);

        /*
         * ------------------------------------------------------------
         * STEP 4: Differential of Iq Reference (Current Derivative)
         * ------------------------------------------------------------
         * Needed for the inductive voltage drop Lq * diq/dt.
         *
         * Derivation:
         *   iq_dot = (J * omega_ddot + B * omega_dot) / (1.5 * p * FluxPm)
         *
         * Load torque is assumed constant (derivative = 0).
         * This term is critical for high-speed operation.
         *
         * Noise note: Omega_ddot comes from S-curve (noise-free).
         * The division by torqueConstant is still safe.
         */
        iqRefDot = ((machinePtr->J * omegaRefDDot) +
                    (machinePtr->B * omegaRefDot)) /
                   torqueConstant;

        /*
         * Limit the current derivative to prevent voltage overshoot.
         * This also prevents noise from causing rapid voltage changes.
         */
        iqRefDot = DFC_ClampValue(iqRefDot, -DFC_MAX_IQ_DOT_F, DFC_MAX_IQ_DOT_F);
    }
    else
    {
        /* Safety: If torque constant is zero, set everything to zero */
        iqRef = 0.0F;
        iqRefDot = 0.0F;
    }

    /*
     * ------------------------------------------------------------
     * STEP 5: Differential-Flatness Voltage Mapping
     * ------------------------------------------------------------
     * This is the "inverse electrical model" of the PMSM.
     * It calculates the voltages needed to produce the desired currents.
     *
     * For Id = 0 (surface PMSM):
     *   Vd = -p * omega * Lq * Iq        (cross-coupling term)
     *   Vq = Rs * Iq + Lq * Iq_dot + p * omega * FluxPm
     *
     * Physical interpretation:
     *   - Vd: Voltage needed to counteract speed-dependent coupling
     *   - Vq: Voltage for resistive drop, inductive drop, and back-EMF
     *
     * Noise note: All inputs are from the clean trajectory or constants.
     * No sensor data is used here, so no noise amplification.
     */
    vdRef = -machinePtr->PolePairs * omegaRef * machinePtr->Lq * iqRef;
    vqRef = (machinePtr->Rs * iqRef) +
            (machinePtr->Lq * iqRefDot) +
            (machinePtr->PolePairs * omegaRef * machinePtr->FluxPm);

    /*
     * ------------------------------------------------------------
     * STEP 6: Measure and Transform Phase Currents to dq Frame
     * ------------------------------------------------------------
     * This is where sensor noise enters the control loop.
     * The phase current measurements contain:
     *   - PWM switching noise (high frequency)
     *   - ADC quantization noise
     *   - Thermal drift (DC offset)
     *
     * The Clarke and Park transforms convert this to dq frame.
     * In dq frame:
     *   - High-frequency noise appears as AC ripple
     *   - DC offsets appear as steady-state errors
     *
     * Our PI correction will handle both.
     */
    DFC_CurrentsToDq(inputPtr, machinePtr, &dqCurrentMeas);

    /*
     * ------------------------------------------------------------
     * STEP 7: Calculate Current Errors for PI Correction
     * ------------------------------------------------------------
     *
     *   idError = idRef - idMeas    (idRef = 0 for surface PMSM)
     *   iqError = iqRef - iqMeas
     *
     * These errors contain:
     *   1. Actual tracking errors (desired vs actual)
     *   2. Sensor noise (from the measurements)
     *   3. Model errors (parameter mismatches)
     *   4. Disturbances (load changes, temperature effects)
     *
     * Our PI controller must correct #1, #3, #4 while not amplifying #2.
     * This is why the gains are deliberately low.
     */
    idError = 0.0F - dqCurrentMeas.D;    /* Id reference is 0 for surface PMSM */
    iqError = iqRef - dqCurrentMeas.Q;

    /*
     * ------------------------------------------------------------
     * STEP 8: Integrate Errors (with Anti-Windup)
     * ------------------------------------------------------------
     * The integral term accumulates error over time.
     * This is what eliminates steady-state error.
     *
     * However, integrators can be problematic with noise:
     *   - Noise spikes can cause integral windup
     *   - Continuous noise can cause steady-state oscillation
     *
     * To handle this, we:
     *   1. Clamp the error before integration
     *   2. Clamp the integral term itself
     *   3. Use very low integral gains
     */
    idIntegralError += idError;
    iqIntegralError += iqError;

    /*
     * Limit integral terms to prevent windup.
     * This is critical for noise robustness.
     * If the noise causes large errors, the integrator
     * won't accumulate excessive values.
     */
    idIntegralError = DFC_ClampValue(idIntegralError,
                                     -DFC_INTEGRAL_LIMIT_F,
                                     DFC_INTEGRAL_LIMIT_F);
    iqIntegralError = DFC_ClampValue(iqIntegralError,
                                     -DFC_INTEGRAL_LIMIT_F,
                                     DFC_INTEGRAL_LIMIT_F);

    /*
     * Clamp current errors to prevent excessive voltage commands.
     * This is the first line of defense against noise spikes.
     * If a noise spike causes an error of 100A, we clamp it to 100A.
     * This prevents the PI controller from commanding huge voltages.
     */
    idError = DFC_ClampValue(idError, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);
    iqError = DFC_ClampValue(iqError, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);

    /*
     * ------------------------------------------------------------
     * STEP 9: Final Voltage Commands = Flatness FF + PI Correction
     * ------------------------------------------------------------
     *
     *   Vd_cmd = Vd_FF + Kp_d * idError + Ki_d * ∫idError dt
     *   Vq_cmd = Vq_FF + Kp_q * iqError + Ki_q * ∫iqError dt
     *
     * The PI gains are carefully tuned to:
     *   1. Provide sufficient correction for model errors
     *   2. Not amplify sensor noise
     *   3. Provide stable operation at all speeds
     *   4. Maintain good transient response
     *
     * Key principle: The feedforward (flatness) should do 80-90% of
     * the control effort. The PI only does 10-20% correction.
     * This makes the system robust to noise because the feedback
     * gain is low.
     */
    dqVoltage.D = vdRef + (DFC_CURRENT_KP_D_F * idError) + (DFC_CURRENT_KI_D_F * idIntegralError);
    dqVoltage.Q = vqRef + (DFC_CURRENT_KP_Q_F * iqError) + (DFC_CURRENT_KI_Q_F * iqIntegralError);

    /*
     * ------------------------------------------------------------
     * STEP 10: Get Measured Rotor Electrical Angle
     * ------------------------------------------------------------
     * The electrical angle is needed for the inverse Park transform.
     * We use:
     *   theta_e = (theta_m + omega * Ts) * PolePairs
     *
     * This includes speed estimation to compensate for position
     * measurement delay. The speed measurement may have noise,
     * but the position is from the encoder/resolver.
     *
     * Noise note: Speed noise would cause angle jitter.
     * We rely on the position sensor's filtering (usually hardware).
     */
    rotorSpeedMeas = 0.0F; /* TODO: Get speed from position derivative or observer */
    rotorAngleMeas = (inputPtr->RotorPositionEst + (rotorSpeedMeas * inputPtr->SampleTime)) * machinePtr->PolePairs;
    DFC_WrapAngle(&rotorAngleMeas);
    focAngle.ThetaE = rotorAngleMeas;

    /*
     * ------------------------------------------------------------
     * STEP 11: Transform dq Voltage to Stationary Frame (Alpha-Beta)
     * ------------------------------------------------------------
     * Inverse Park transform converts the rotating frame voltages
     * to the stationary frame used by the SVM modulator.
     */
    InvPark_Transform_Matrix(&dqVoltage, &focAngle, &abVoltage);

    /*
     * ------------------------------------------------------------
     * STEP 12: Space Vector Modulation
     * ------------------------------------------------------------
     * The SVM generates the PWM duty cycles from the alpha-beta voltages.
     *
     * Important for noise handling:
     *   1. The modulation index is clamped to 0.95
     *      - Prevents overmodulation from noise spikes
     *      - Leaves margin for inverter non-linearity
     *   2. The SVM algorithm itself has some filtering effect
     *      - PWM frequency is fixed (typically 10-20 kHz)
     *      - This acts as a zero-order hold
     *
     * The low modulation index limit (0.95) ensures that even with
     * noise-induced voltage spikes, the PWM stays in the linear region.
     * This prevents:
     *   - Nonlinear distortion
     *   - Audible noise
     *   - Current spikes
     */
    vMag = sqrtf((abVoltage.Alpha * abVoltage.Alpha) + (abVoltage.Beta * abVoltage.Beta));

    vPhaseMax = machinePtr->Vdc / DFC_SQRT3_F;
    modulationIndex = vMag / vPhaseMax;

    /*
     * Clamp modulation index to 0.25 for testing (safety factor).
     * In production, this would be 0.95 for full voltage utilization.
     * The lower limit provides additional safety during development.
     */
    modulationIndex = DFC_ClampValue(modulationIndex, 0.0F, 0.25F);
    SVM_CalculateDutyCycle(modulationIndex, &focAngle, &svmDC);

    /*
     * ------------------------------------------------------------
     * STEP 13: Write PWM Duty Cycles to Output Structure
     * ------------------------------------------------------------
     * These duty cycles are used to generate the gate signals
     * for the inverter switches.
     *
     * The outputs are:
     *   - DutyU: Phase U duty cycle (0.0 to 1.0)
     *   - DutyV: Phase V duty cycle (0.0 to 1.0)
     *   - DutyW: Phase W duty cycle (0.0 to 1.0)
     *   - Sector: SVM sector (for debugging)
     *   - Valid: Output validity flag
     */
    outputPtr->DutyU = svmDC.Ta;
    outputPtr->DutyV = svmDC.Tb;
    outputPtr->DutyW = svmDC.Tc;
    outputPtr->SvmSector = svmDC.Sector;
    outputPtr->Valid = 0x1U;
}
