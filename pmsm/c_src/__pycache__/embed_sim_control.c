/**********************************************************************************************************************
 * \file      embed_sim_control.c
 * \brief     Top-level PMSM control module with DFC controller.
 *
 * \details   Implements the main control loop for permanent magnet synchronous motors
 *            (PMSM). Supports open-loop and DFC control modes with smooth reference
 *            trajectory generation.
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

#include "embed_sim_control.h"
#include "embed_sim_motor_parameter.h"
#include "embed_sim_matrix.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_controller.h"
#include "embed_sim_cython_interface.h"
#include <stddef.h>
#include <math.h>
#include <string.h>

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/* Macros are defined in embed_sim_control.h */

/*********************************************************************************************************************/
/*--------------------------------------------------Private Data-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Traction motor parameters
 *
 * \details Static instance of PMSM parameters for the traction motor.
 *          Values are taken from MP_* macros defined in motor_parameter.h.
 */
static EmbedSimMachineParam_T TractionMotorParams_G =
{
    .PolePairs              = MP_POLES,
    .Rs                     = MP_R_S,
    .Ld                     = MP_L_D,
    .Lq                     = MP_L_Q,
    .FluxPm                 = MP_LAMBDA_PM,
    .J                      = MP_J_ROTOR,
    .B                      = MP_B_FRIC,
    .Vdc                    = MP_V_DC,
    .ParamPidCurrentQProp   = DFC_CURRENT_KP_Q_F,
    .ParamPidCurrentQInteg  = DFC_CURRENT_KI_Q_F,
    .ParamPidCurrentDProp   = DFC_CURRENT_KP_D_F,
    .ParamPidCurrentDInteg  = DFC_CURRENT_KI_D_F,
    .ParamPidSpeedQProp     = DFC_SPEED_KP_Q_F ,
    .ParamPidSpeedQInteg    = DFC_SPEED_KI_Q_F,
    .ParamPidIntegralLimit  = DFC_INTEGRAL_LIMIT_F
};

/**
 * \brief  Global Control Data Structures for Traction Motor
 */
EmbedSimCtrlInput_T    TractionMotorInput_G;     /**< Input data (references & feedback) */
EmbedSimCtrlOutput_T   TractionMotorOutput_G;    /**< Output data (PWM duty cycles) */
EmbedSimMachine_T      TractionMotor_G;          /**< Combined motor structure */

/*********************************************************************************************************************/
/*-----------------------------------------Private Function Prototypes-----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Execute one step of open-loop motor control
 *
 * \param[in]  inputPtr  Pointer to input structure.
 * \param[in]  paraPtr   Pointer to motor parameters.
 * \param[out] outputPtr Pointer to output structure.
 */
static void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T* const inputPtr,
                                  EmbedSimMachineParam_T* const paraPtr,
                                  EmbedSimCtrlOutput_T* const outputPtr);

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \param[in,out] anglePtr  Pointer to angle value to be wrapped (in radians).
 */
static void EmbedSim_WrapAngle(real32_T* anglePtr);

/**
 * \brief   Clamp value to specified limits
 *
 * \param[in] val     Value to clamp.
 * \param[in] minVal  Minimum allowed value.
 * \param[in] maxVal  Maximum allowed value.
 *
 * \return  Clamped value within [minVal, maxVal].
 */
static real32_T EmbedSim_ClampValue(real32_T val, real32_T minVal, real32_T maxVal);

/**
 * \brief   Update observer estimates and prepare inputs for the controller.
 *
 * \details Copies sensor readings to the estimated fields, clamps the speed
 *          reference, and increments the loop counter. This function is called
 *          at the beginning of each control step to provide the latest feedback.
 *
 * \param[in,out] inputPtr  Pointer to control input structure containing
 *                          sensor data and references.
 */
static void EmbedSim_ExecuteObserver(EmbedSimCtrlInput_T* const inputPtr);

/*********************************************************************************************************************/
/*--------------------------------------Private Function Implementations---------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \details Normalizes an angle to the range [0, 2π) using fmodf.
 *          Useful for rotor angle and Park transform calculations.
 *
 * \param[in,out] anglePtr  Pointer to angle value to be wrapped (in radians).
 */
static void EmbedSim_WrapAngle(real32_T* anglePtr)
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
 *          If value is below minVal, returns minVal.
 *          If value is above maxVal, returns maxVal.
 *          Otherwise returns the original value.
 *
 * \param[in] val     Value to clamp.
 * \param[in] minVal  Minimum allowed value.
 * \param[in] maxVal  Maximum allowed value.
 *
 * \return  Clamped value within [minVal, maxVal].
 */
static real32_T EmbedSim_ClampValue(real32_T val, real32_T minVal, real32_T maxVal)
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
 * \brief   Execute one step of open-loop motor control
 *
 * \details Performs open-loop control for a surface PMSM motor using Id=0 control
 *          strategy. Calculates the rotor angle based on the reference angular
 *          velocity, generates DQ voltage commands, and converts them to duty
 *          cycles using Space Vector PWM (SVPWM).
 *
 * \param[in]  inputPtr   Pointer to input structure containing:
 *                        - AngularVelocityRef: Target angular velocity (rad/s)
 *                        - SampleTime: Time step for integration (s)
 *                        - Valid: Flag indicating if input data is valid
 * \param[in]  paraPtr    Pointer to motor parameters (Vdc, pole pairs).
 * \param[out] outputPtr  Pointer to output structure where results are stored:
 *                        - DutyU, DutyV, DutyW: PWM duty cycles [0, 1]
 *                        - SvmSector: Active SVPWM sector (if valid)
 *                        - Valid: Status flag (0x1 if valid, 0x0 if invalid)
 *
 * \note The modulation index is fixed at 0.2 in this implementation.
 * \note The rotor angle is wrapped to the range [0, 2π) after each step.
 */
static void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T*    const inputPtr,
                                  EmbedSimMachineParam_T* const paraPtr,
                                  EmbedSimCtrlOutput_T*   const outputPtr)
{
    static real32_T rotorAngleE = 0.0F;    /**< Electrical rotor angle [rad]        */
    FocAngle_T      focAngle;              /**< Field-oriented control angle        */
    FocDq_T         dqVoltage;             /**< dq voltage commands                 */
    SVM_DutyCycle_T svmDC;                 /**< SVM duty cycle outputs              */
    real32_T        angularVelocityE;      /**< Electrical angular velocity [rad/s] */
    real32_T        modulation;            /**< Modulation index (fixed at 0.2)     */

    /* Initialise outputs to safe default values */
    outputPtr->DutyU = 0.5F;
    outputPtr->DutyV = 0.5F;
    outputPtr->DutyW = 0.5F;
    outputPtr->Valid = 0x0U;

    modulation = 0.2F;  /* Fixed modulation index for open-loop */

    /* Only execute if input data is valid */
    if (inputPtr->Valid == 0x1U)
    {
        /* Calculate electrical angular velocity (mechanical × pole pairs) */
        angularVelocityE = inputPtr->RotorVelocityRefM * paraPtr->PolePairs;

        /* Update rotor angle by integration */
        rotorAngleE += (angularVelocityE * inputPtr->SampleTime);
        EmbedSim_WrapAngle(&rotorAngleE);

        focAngle.ThetaE = rotorAngleE;

        /* Id = 0 control for surface PMSM */
        dqVoltage.D = 0.0F;

        /* Vq magnitude = modulation × (Vdc/√3) */
        dqVoltage.Q = (paraPtr->Vdc / SVM_SQRT3_F) * modulation;

        /* Convert dq voltage to PWM using SVPWM */
        if (SVM_CalculateDutyCycleFromDq(&dqVoltage, &focAngle, paraPtr->Vdc, &svmDC) == MATRIX_SUCCESS)
        {
            outputPtr->DutyU = svmDC.Ta;
            outputPtr->DutyV = svmDC.Tb;
            outputPtr->DutyW = svmDC.Tc;
            outputPtr->SvmSector = svmDC.Sector;
            outputPtr->Valid = 0x1U;
        }
    }
}

/*********************************************************************************************************************/
/*--------------------------------------Public Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize control module
 *
 * \details Initializes the SVPWM and DFC controller modules.
 *          Must be called once before any control step.
 *
 * \return  void
 */
void EmbedSim_ControlInit(void)
{
    /* Initialize motor structure with pointers to data */
    TractionMotor_G.InputPtr   = &TractionMotorInput_G;
    TractionMotor_G.OutputPtr  = &TractionMotorOutput_G;
    TractionMotor_G.MachinePtr = &TractionMotorParams_G;
    /* Initialize SVM module */
    SVM_Init();
    /* Initialize DFC controller */
    DFC_Init(&TractionMotor_G);
}

/**
 * \brief   Top-level PMSM control step
 *
 * \details Executes one control cycle. Based on the input validity and
 *          control mode, either open-loop or DFC control is performed.
 *          For DFC mode, estimates rotor position and speed from sensors.
 *
 * \param[in]  motorPtr  Pointer to motor structure containing:
 *                       - InputPtr   : Control input (references & feedback)
 *                       - OutputPtr  : Control output (voltage commands)
 *                       - MachinePtr : Motor parameters
 *
 * \return  void
 */
void EmbedSim_ControlStep(EmbedSimMachine_T* const MotorPtr)
{
    EmbedSimCtrlInput_T*    inputPtr  = MotorPtr->InputPtr;
    EmbedSimCtrlOutput_T*   outputPtr = MotorPtr->OutputPtr;
    EmbedSimMachineParam_T* paraPtr   = MotorPtr->MachinePtr;

    /* Update observer estimates from sensors */
    EmbedSim_ExecuteObserver(inputPtr);

    /* Update DC bus voltage from input (allows run‑time variation) */
    paraPtr->Vdc = inputPtr->Vdc;

    /* Only execute if input data is valid */
    if (inputPtr->Valid == 0x1U)
    {
        switch (inputPtr->CtrlAlg)
        {
            case SIM_CTRL_OPEN_LOOP:
                EmbedSim_OpenLoopStep(inputPtr, paraPtr, outputPtr);
                break;
            case SIM_CTRL_DFC:
                /* Generate smooth trajectory and then run DFC */
                DFC_Step(MotorPtr);
                break;
            default:
                /* No action for unknown algorithm */
                break;
        }
    }
}

/**
 * \brief   Check if the motor is spinning and has reached ≥95% of the reference speed.
 *
 * \details Uses a debounce timer (in seconds) to avoid noise. Returns 1 only after
 *          the measured speed has stayed above 95% of the given reference for the
 *          specified duration.
 *
 * \param[in] InputPtr     Pointer to control input structure.
 * \param[in] SpeedRefRPM  Reference speed in RPM.
 * \param[in] Duration     Required debounce time in seconds.
 *
 * \return  1 if motor is spinning fast enough, 0 otherwise.
 */
uint32_T EmbedSim_IsMotorSpinning(const EmbedSimCtrlInput_T* const InputPtr, const real32_T const SpeedRefRPM, const real32_T const Duration)
{
    static uint32_T successCounter = 0U;
    uint32_T result;
    real32_T speed;
    real32_T threshold;
    uint32_T requiredSamples;
    real32_T samplesFloat;

    samplesFloat = Duration / InputPtr->SampleTime;       /* always > 0 if Duration > 0 */
    result = 0x0U;

    /* Round up to next whole sample count */
    samplesFloat = ceilf(samplesFloat);
    requiredSamples = (uint32_T)samplesFloat;   /* safe cast, no overflow expected */

    /* Ensure at least one sample – in case Duration is 0 */
    if (requiredSamples == 0U)
    {
        requiredSamples = 1U;
    }

    speed = InputPtr->RotorSpeedSensorM;

    /* Validate reference speed and sensor reading */
    if((isnan(speed) == 0) && (isinf(speed) == 0))
    {
        threshold =  SpeedRefRPM;

        if (fabsf(speed) > threshold)
        {
            if (successCounter < MAX_uint32_T)
            {
                successCounter++;
            }
        }
        else
        {
            successCounter = 0U;
        }

        if (successCounter >= requiredSamples)
        {
            result = 1U;
            successCounter = 0U;
        }
    }
    else
    {
        successCounter = 0U;
    }

    return result;
}

/**
 * \brief   Check if the motor has stopped (speed below a low threshold).
 *
 * \details Uses a debounce counter to avoid noise. Returns 1 only after the
 *          estimated speed has stayed below 0.2 RPM for a given number of
 *          consecutive samples.
 *
 * \param[in] InputPtr    Pointer to control input structure.
 * \param[in] PastIndex   Number of consecutive valid samples required.
 *
 * \return  1 if motor is stopped, 0 otherwise.
 */
uint32_T EmbedSim_IsNotSpinning(const EmbedSimCtrlInput_T* const InputPtr,
                                uint32_T PastIndex)
{
    static uint32_T successCounter = 0U;
    real32_T  speed;
    uint32_T  validSpeed;
    uint32_T  result;

    speed = InputPtr->RotorSpeedObsEstM;
    validSpeed = 0x1U;
    result = 0U;

    /* Check for invalid floating-point values */
    if ((isnan(speed) != 0x0U) || (isinf(speed) != 0x0U))
    {
        successCounter++;
        validSpeed = 0U;
    }

    if (validSpeed == 0x1U)
    {
        /* If speed magnitude is below 0.2 RPM, consider stopped */
        if (fabsf(speed) < 5.0f)
        {
            if (successCounter < MAX_int32_T)
            {
                successCounter = successCounter + 1U;
            }
        }
        else
        {
            successCounter = 0U;
        }
    }

    if (successCounter > PastIndex)
    {
        result = 1U;
        successCounter = 0U;
    }

    return result;
}

/**
 * \brief   Get current motor state for unified reporting
 *
 * \details Fills the motor state structure with current values from
 *          the control system. This provides a unified view of motor
 *          operation for display and logging.
 *
 * \param[in]  motorPtr   Pointer to motor structure
 * \param[out] statePtr   Pointer to state structure to fill
 *
 * \return  void
 */
void EmbedSim_GetMotorState(EmbedSimMachine_T* const motorPtr,
                            EmbedSimMotorState_T* const statePtr)
{
    EmbedSimCtrlInput_T*    inputPtr  = motorPtr->InputPtr;
    EmbedSimCtrlOutput_T*   outputPtr = motorPtr->OutputPtr;
    EmbedSimMachineParam_T* paraPtr   = motorPtr->MachinePtr;
    FocUvw_T uvw;
    FocAlphaBeta_T ab;
    FocAngle_T angle;
    FocDq_T dq;


    /* ===== Mechanical ===== */
    statePtr->SpeedRpm = inputPtr->RotorSpeedSensorM;
    statePtr->SpeedRadS = CON_RPM_TO_RAD(inputPtr->RotorSpeedSensorM);
    statePtr->PositionRad = inputPtr->RotorPositionSensorM;
    statePtr->AccelerationRpmS = CON_RAD_TO_RPM(inputPtr->RotorAccelerationRefM);
    statePtr->JerkRpmS3 = CON_RAD_TO_RPM(inputPtr->RotorJerkRefM);

    /* ===== Currents ===== */
    statePtr->Ia = inputPtr->Iu;
    statePtr->Ib = inputPtr->Iv;
    statePtr->Ic = inputPtr->Iw;

    /* DQ currents - use coordinate transforms */
    uvw.U = inputPtr->Iu;
    uvw.V = inputPtr->Iv;
    uvw.W = inputPtr->Iw;

    Clarke_Transform_Matrix(&uvw, &ab);
    statePtr->Ialpha = ab.Alpha;
    statePtr->Ibeta = ab.Beta;

    angle.ThetaE = inputPtr->RotorPositionSensorM * paraPtr->PolePairs;
    EmbedSim_WrapAngle(&angle.ThetaE);
    Park_Transform_Matrix(&ab, &angle, &dq);
    statePtr->Id = dq.D;
    statePtr->Iq = dq.Q;

    /* ===== PWM ===== */
    statePtr->DutyU = outputPtr->DutyU;
    statePtr->DutyV = outputPtr->DutyV;
    statePtr->DutyW = outputPtr->DutyW;
    statePtr->SvmSector = outputPtr->SvmSector;

    /* ===== References ===== */
    statePtr->SpeedRefRpm = inputPtr->AngularVelocityRefRpmM;
    statePtr->SpeedRefRadS = CON_RPM_TO_RAD(inputPtr->AngularVelocityRefRpmM);
    statePtr->IqRef = 0.0F;  /* Calculated in DFC */
    statePtr->IdRef = 0.0F;

    /* ===== Control Mode ===== */
    statePtr->SwitchToClosedLoop = inputPtr->SwitchToClosedLoop;
    statePtr->ControlReInit = inputPtr->ControlReInit;
    statePtr->ControllerMode = inputPtr->CtrlAlg;

    /* ===== PI States ===== */
    statePtr->SpeedIntegral = paraPtr->SpeedIntegralError;
    statePtr->IdIntegral = paraPtr->IdIntegralError;
    statePtr->IqIntegral = paraPtr->IqIntegralError;

    /* ===== Startup ===== */
    statePtr->StartupModulation = paraPtr->SvmModulationIndex;
    statePtr->StartupTheta = paraPtr->StartupThetaE;
    statePtr->StartupTime = 0.0F;  /* Tracked in DFC */

    /* ===== Spinning ===== */
    statePtr->SpinningPastIndex = DFC_SPINNING_PAST_INDEX;
    statePtr->StoppedPastIndex = DFC_STOPPED_PAST_INDEX;
    statePtr->IsStopped = EmbedSim_IsNotSpinning(inputPtr, DFC_STOPPED_PAST_INDEX);

    /* ===== Trajectory ===== */
    statePtr->TrajSpeedRpm = CON_RAD_TO_RPM(inputPtr->RotorVelocityRefM);
    statePtr->TrajAccelRpmS = CON_RAD_TO_RPM(inputPtr->RotorAccelerationRefM);
    statePtr->TrajJerkRpmS3 = CON_RAD_TO_RPM(inputPtr->RotorJerkRefM);

    /* ===== Speed Error ===== */
    statePtr->SpeedErrorRpm = inputPtr->AngularVelocityRefRpmM - inputPtr->RotorSpeedSensorM;
    statePtr->SpeedErrorRadS = CON_RPM_TO_RAD(statePtr->SpeedErrorRpm);
    if (fabsf(inputPtr->AngularVelocityRefRpmM) > 0.1F)
    {
        statePtr->SpeedErrorPercent = (statePtr->SpeedErrorRpm / inputPtr->AngularVelocityRefRpmM) * 100.0F;
    }
    else
    {
        statePtr->SpeedErrorPercent = 0.0F;
    }

    /* ===== Status ===== */
    statePtr->Valid = 0x1U;
    statePtr->LoopCounter = inputPtr->LoopCounter;
    statePtr->Dt = inputPtr->SampleTime;

    /* ===== Torque ===== */
    if (statePtr->SwitchToClosedLoop == 0x1U)
    {
        real32_T omega_rad = statePtr->SpeedRadS;
        real32_T omega_dot_rad = statePtr->AccelerationRpmS * ES_MATH_2PI_F / 60.0F;

        /* Mechanical flatness */
        statePtr->TorqueFF = paraPtr->J * omega_dot_rad + paraPtr->B * omega_rad;
        statePtr->TorqueConstant = 1.5F * paraPtr->PolePairs * paraPtr->FluxPm;
        if (fabsf(statePtr->TorqueConstant) > 1e-6F)
        {
            statePtr->IqRef = statePtr->TorqueFF / statePtr->TorqueConstant;
        }
        statePtr->TorqueTotal = statePtr->TorqueFF;
    }
}

/**
 * \brief   Cython interface initialization
 *
 * \details Wrapper function for Python/Cython interface to initialize
 *          the control module.
 *
 * \return  void
 */
void EmbedSim_CythonControlInit(void)
{
    EmbedSim_ControlInit();
}



/**
 * \brief   Calculates online time-optimal jerk-limited trajectory with reality awareness
 *
 * \details Generates smooth S-curve trajectory with reality correction:
 *          - Uses sensor feedback to detect if motor is stuck
 *          - STABILIZES trajectory when motor falls behind
 *          - Prevents trajectory from running away
 *          - Single entry, single exit (MISRA Rule 14.7)
 *
 * \param[in,out] InputPtr  Pointer to control input structure
 * \param[in]     ParaPtr   Pointer to motor parameters (unused)
 */
void EmbedSim_CalculateOptimalPath(EmbedSimCtrlInput_T* const InputPtr,
                                   EmbedSimMachineParam_T* const ParaPtr)
{
    real32_T targetOmega;
    real32_T sampleTime;
    real32_T sampleTimeSq;
    real32_T sampleTimeCu;
    real32_T plannedVelocity;
    real32_T plannedAccel;
    real32_T actualVelocity;
    real32_T velocityError;
    real32_T velocityErrorMagnitude;
    real32_T direction;
    real32_T speedMax;
    real32_T accelMax;
    real32_T jerkMax;
    real32_T stoppingAccel;
    real32_T desiredAccel;
    real32_T jerkRequest;
    real32_T oldVelocity;
    real32_T oldAccel;
    real32_T positionIncrement;

    /* Reality check variables */
    real32_T speedGap;
    uint32_T isMotorStopped;
    real32_T correction;

    const real32_T half = 0.5F;
    const real32_T sixth = 0.1666666667F;
    const real32_T settleTol = 0.01F;
    const real32_T fallBehindThreshold = 2.0F;      /* 2 rad/s = ~19 RPM */
    const real32_T catchUpFactor = 0.1F;             /* 10% reduction per step */
    const real32_T maxSpeedReduction = 2.0F;         /* Max 2 rad/s per step */
    const real32_T gentleCorrectionFactor = 0.02F;   /* 2% gentle correction */
    const real32_T gentleMaxCorrection = 0.5F;       /* Max 0.5 rad/s */

    (void)ParaPtr;

    /* Initialize all variables */
    targetOmega = 0.0F;
    sampleTime = 0.0F;
    sampleTimeSq = 0.0F;
    sampleTimeCu = 0.0F;
    plannedVelocity = 0.0F;
    plannedAccel = 0.0F;
    actualVelocity = 0.0F;
    velocityError = 0.0F;
    velocityErrorMagnitude = 0.0F;
    direction = 0.0F;
    speedMax = 0.0F;
    accelMax = 0.0F;
    jerkMax = 0.0F;
    stoppingAccel = 0.0F;
    desiredAccel = 0.0F;
    jerkRequest = 0.0F;
    oldVelocity = 0.0F;
    oldAccel = 0.0F;
    positionIncrement = 0.0F;
    speedGap = 0.0F;
    isMotorStopped = 0U;
    correction = 0.0F;

    sampleTime = InputPtr->SampleTime;
    sampleTimeSq = sampleTime * sampleTime;
    sampleTimeCu = sampleTimeSq * sampleTime;

    targetOmega = CON_RPM_TO_RAD(InputPtr->AngularVelocityRefRpmM);
    speedMax = CON_RPM_TO_RAD(MAX_SPEED_RPM);
    accelMax = CON_RPM_TO_RAD(MAX_ACCEL_RPM);
    jerkMax = CON_RPM_TO_RAD(MAX_JERK_RPM);

    targetOmega = EmbedSim_ClampValue(targetOmega, -speedMax, speedMax);

    /* ================================================================
     * STEP 1: REINITIALIZATION - Reset trajectory to match reality
     * ================================================================ */
    if (InputPtr->ControlReInit == 1U)
    {
        InputPtr->RotorVelocityRefM = CON_RPM_TO_RAD(InputPtr->RotorSpeedObsEstM);
        InputPtr->RotorAccelerationRefM = 0.0F;
        InputPtr->RotorJerkRefM = 0.0F;
        InputPtr->RotorPositionRefM = InputPtr->RotorPositionObsEstM;
        InputPtr->ControlReInit = 0x0U;
    }

    /* ================================================================
     * STEP 2: READ CURRENT STATE
     * ================================================================ */
    plannedVelocity = InputPtr->RotorVelocityRefM;
    plannedAccel = InputPtr->RotorAccelerationRefM;
    actualVelocity = CON_RPM_TO_RAD(InputPtr->RotorSpeedObsEstM);

    /* ================================================================
     * STEP 3: REALITY CHECK - Is motor falling behind?
     * ================================================================ */
    speedGap = plannedVelocity - actualVelocity;
    isMotorStopped = EmbedSim_IsNotSpinning(InputPtr, DFC_STOPPED_PAST_INDEX);

    /* ================================================================
     * STEP 4: REALITY CORRECTION - Motor is stuck
     * ================================================================ */
    if ((speedGap > fallBehindThreshold) && (isMotorStopped == 1U))
    {
        /* Motor is STUCK - stabilize trajectory */
        correction = speedGap * catchUpFactor;
        if (correction > maxSpeedReduction)
        {
            correction = maxSpeedReduction;
        }

        /* Reduce planned velocity to let motor catch up */
        plannedVelocity = plannedVelocity - correction;

        /* Don't go below target */
        if (plannedVelocity < targetOmega)
        {
            plannedVelocity = targetOmega;
        }

        /* Don't drop too far below actual speed */
        if (plannedVelocity < (actualVelocity + 0.5F))
        {
            plannedVelocity = actualVelocity + 0.5F;
        }

        /* Don't go negative when target is positive */
        if ((targetOmega > 0.0F) && (plannedVelocity < 0.0F))
        {
            plannedVelocity = 0.0F;
        }

        /* Apply stabilized trajectory */
        InputPtr->RotorVelocityRefM = plannedVelocity;
        InputPtr->RotorAccelerationRefM = 0.0F;
        InputPtr->RotorJerkRefM = 0.0F;
        InputPtr->RotorPositionRefM += (plannedVelocity * sampleTime);

        InputPtr->RotorVelocityRefM = EmbedSim_ClampValue(
            InputPtr->RotorVelocityRefM, -speedMax, speedMax);
    }

    /* ================================================================
     * STEP 5: REALITY CORRECTION - Motor lagging (gentle)
     * ================================================================ */
    else if (speedGap > fallBehindThreshold)
    {
        /* Motor is lagging but not stuck - gentle correction */
        correction = speedGap * gentleCorrectionFactor;
        if (correction > gentleMaxCorrection)
        {
            correction = gentleMaxCorrection;
        }

        plannedVelocity = plannedVelocity - correction;

        if (plannedVelocity < targetOmega)
        {
            plannedVelocity = targetOmega;
        }

        InputPtr->RotorVelocityRefM = plannedVelocity;
        InputPtr->RotorAccelerationRefM = 0.0F;
        InputPtr->RotorJerkRefM = 0.0F;
        InputPtr->RotorPositionRefM += (plannedVelocity * sampleTime);

        InputPtr->RotorVelocityRefM = EmbedSim_ClampValue(
            InputPtr->RotorVelocityRefM, -speedMax, speedMax);
    }

    /* ================================================================
     * STEP 6: NORMAL TRAJECTORY GENERATION
     * ================================================================ */
    else
    {
        velocityError = targetOmega - plannedVelocity;
        velocityErrorMagnitude = fabsf(velocityError);

        /* Deadband settle */
        if ((velocityErrorMagnitude < SPEED_SETTLE_TOL) &&
            (fabsf(plannedAccel) < settleTol))
        {
            InputPtr->RotorVelocityRefM = targetOmega;
            InputPtr->RotorAccelerationRefM = 0.0F;
            InputPtr->RotorJerkRefM = 0.0F;
            InputPtr->RotorPositionRefM += (targetOmega * sampleTime);
        }
        else
        {
            direction = (velocityError >= 0.0F) ? 1.0F : -1.0F;

            stoppingAccel = sqrtf(2.0F * jerkMax * velocityErrorMagnitude);
            desiredAccel = direction * ((accelMax < stoppingAccel) ? accelMax : stoppingAccel);

            /* Reality-aware acceleration limit */
            if ((actualVelocity < plannedVelocity) && (desiredAccel > 0.0F))
            {
                desiredAccel = desiredAccel * 0.5F;
            }

            desiredAccel = EmbedSim_ClampValue(desiredAccel, -accelMax, accelMax);

            jerkRequest = (desiredAccel - plannedAccel) / sampleTime;
            InputPtr->RotorJerkRefM = EmbedSim_ClampValue(jerkRequest, -jerkMax, jerkMax);

            oldVelocity = plannedVelocity;
            oldAccel = plannedAccel;

            /* Integrate velocity */
            InputPtr->RotorVelocityRefM = oldVelocity
                                          + (oldAccel * sampleTime)
                                          + (half * InputPtr->RotorJerkRefM * sampleTimeSq);

            /* Integrate acceleration */
            InputPtr->RotorAccelerationRefM = oldAccel
                                              + (InputPtr->RotorJerkRefM * sampleTime);

            InputPtr->RotorAccelerationRefM = EmbedSim_ClampValue(
                InputPtr->RotorAccelerationRefM, -accelMax, accelMax);
            InputPtr->RotorVelocityRefM = EmbedSim_ClampValue(
                InputPtr->RotorVelocityRefM, -speedMax, speedMax);

            /* Prevent overshoot */
            if (((direction > 0.0F) && (InputPtr->RotorVelocityRefM > targetOmega)) ||
                ((direction < 0.0F) && (InputPtr->RotorVelocityRefM < targetOmega)))
            {
                InputPtr->RotorVelocityRefM = targetOmega;
                InputPtr->RotorAccelerationRefM = 0.0F;
                InputPtr->RotorJerkRefM = 0.0F;
            }

            /* Integrate position */
            positionIncrement = (oldVelocity * sampleTime)
                                + (half * oldAccel * sampleTimeSq)
                                + (sixth * InputPtr->RotorJerkRefM * sampleTimeCu);

            InputPtr->RotorPositionRefM += positionIncrement;
        }
    }

    /* Single exit point - MISRA Rule 14.7 */
    return;
}


/**
 * \brief   Update observer estimates from sensor readings.
 *
 * \details This function copies the raw sensor values (position and speed)
 *          to the estimated fields used by the controller.
 *
 * \param[in,out] InputPtr  Pointer to control input structure.
 */
void EmbedSim_ExecuteObserver(EmbedSimCtrlInput_T* const InputPtr)
{
    /* Increment loop counter for diagnostic purposes */
    InputPtr->LoopCounter++;

    /* Clamp RPM reference to maximum speed */
    InputPtr->AngularVelocityRefRpmM = EmbedSim_ClampValue(InputPtr->AngularVelocityRefRpmM,
                                                           -MAX_SPEED_RPM,
                                                           MAX_SPEED_RPM);

    /* DO NOT overwrite RotorVelocityRefM here!
     * The trajectory generator (EmbedSim_CalculateOptimalPath) owns this variable.
     * Only set it during reinitialization via ControlReInit flag.
     * This allows the trajectory to stabilize when motor is stuck.
     */

    /* Use sensor readings directly as estimates (no filtering) */
    InputPtr->RotorPositionObsEstM = InputPtr->RotorPositionSensorM;
    InputPtr->RotorSpeedObsEstM    = InputPtr->RotorSpeedSensorM;

    /* Switch to closed-loop decision is handled in DFC_Step */
}


/**
 * \brief   Cython interface control step
 *
 * \details Wrapper function for Python/Cython interface to execute one
 *          control step with direct parameter passing.
 *
 * \param[in]  Iu                      Phase U current [A]
 * \param[in]  Iv                      Phase V current [A]
 * \param[in]  Iw                      Phase W current [A]
 * \param[in]  RotorPositionSensor     Rotor position from sensor [rad]
 * \param[in]  RotorVelocitySensor     Rotor speed from sensor [RPM]
 * \param[in]  AngularVelocityRefRpm   Speed reference [RPM]
 * \param[in]  Vdc                     DC bus voltage [V]
 * \param[in]  SampleTime              Sample time [s]
 * \param[in]  CtrlAlg                 Control algorithm selection
 * \param[in]  ValidIn                 Input validity flag
 * \param[out] PwmU                    Phase U PWM duty cycle [0-1]
 * \param[out] PwmV                    Phase V PWM duty cycle [0-1]
 * \param[out] PwmW                    Phase W PWM duty cycle [0-1]
 * \param[out] ValidOut                Output validity flag
 *
 * \return  void
 */
void EmbedSim_CythonControlStep(
    /* Inputs */
    real32_T  Iu,                      /* [A] */
    real32_T  Iv,                      /* [A] */
    real32_T  Iw,                      /* [A] */
    real32_T  RotorPositionSensor,     /* [rad] */
    real32_T  RotorVelocitySensor,     /* [RPM] */
    real32_T  AngularVelocityRefRpm,   /* [RPM] */
    real32_T  Vdc,                     /* [V] */
    real32_T  SampleTime,              /* [s] */
    uint32_T  CtrlAlg,
    uint32_T  ValidIn,
    /* Outputs */
    real32_T* PwmU,
    real32_T* PwmV,
    real32_T* PwmW,
    uint32_T* ValidOut)
{
    /* Copy input data to global structure */
    TractionMotorInput_G.Iu = Iu;
    TractionMotorInput_G.Iv = Iv;
    TractionMotorInput_G.Iw = Iw;
    TractionMotorInput_G.RotorPositionSensorM = RotorPositionSensor;
    TractionMotorInput_G.RotorSpeedSensorM = RotorVelocitySensor;
    TractionMotorInput_G.AngularVelocityRefRpmM = AngularVelocityRefRpm;
    TractionMotorInput_G.SampleTime = SampleTime;
    TractionMotorInput_G.Vdc = Vdc;
    TractionMotorInput_G.CtrlAlg = CtrlAlg;
    TractionMotorInput_G.Valid = ValidIn;

    /* Execute one control step */
    EmbedSim_ControlStep(&TractionMotor_G);

    /* Copy output data */
    *PwmU  = TractionMotorOutput_G.DutyU;
    *PwmV  = TractionMotorOutput_G.DutyV;
    *PwmW  = TractionMotorOutput_G.DutyW;
    *ValidOut = TractionMotorOutput_G.Valid;
}


/**
 * \brief   Get motor state for unified reporting via Cython
 *
 * \details Returns the current motor state structure filled with values
 *          from the control system. This provides a unified view of motor
 *          operation for display and logging in Python.
 *
 * \param[out] statePtr  Pointer to state structure to fill
 *
 * \return  void
 */
void EmbedSim_CythonGetMotorState(EmbedSimMotorState_T* const StatePtr)
{
    /* Check if pointer is valid using NULL (not comparing to int) */
    if (StatePtr != NULL)
    {
        EmbedSim_GetMotorState(&TractionMotor_G, StatePtr);
    }
}
