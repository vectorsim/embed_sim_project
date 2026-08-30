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
#include <stdio.h>
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
 * \brief   Update observer estimates and prepare inputs for the controller.
 *
 * \details Copies sensor readings to the estimated fields, clamps the speed
 *          reference, and increments the loop counter. This function is called
 *          at the beginning of each control step to provide the latest feedback.
 *
 * \param[in,out] inputPtr  Pointer to control input structure containing
 *                          sensor data and references.
 */
static void EmbedSim_ExecuteObserver(EmbedSimMachine_T* const MotorPtr);


/**
 * \brief   Prints Debug Information
 *
 * \details Used in Python Simulation
 *
 * \param[in,out] inputPtr  Pointer to State Structue(Cythen)
 *
 */
static void EmbedSim_ControlStatePrint(const EmbedSimMotorState_T* const StatePtr);

/*********************************************************************************************************************/
/*--------------------------------------Private Function Implementations---------------------------------------------*/
/*********************************************************************************************************************/

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
        EmbedSim_WrapAngleTwoPi(&rotorAngleE);

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


/**
 * \brief   Update observer estimates from sensor readings.
 *
 * \details Copies raw sensor values to estimated fields, increments loop counter,
 *          clamps speed reference, and implements a smooth convergence algorithm
 *          for the electrical rotor angle model when in closed‑loop control.
 *
 * \note    Assumes the observer (e.g., PLL or state estimator) has already
 *          validated and filtered the sensor readings. This function only
 *          aligns the internal model angle with the estimated angle.
 *
 * \param[in,out] MotorPtr  Pointer to the motor structure.
 */
void EmbedSim_ExecuteObserver(EmbedSimMachine_T* const MotorPtr)
{
    EmbedSimCtrlInput_T*    iPtr = MotorPtr->InputPtr;
    EmbedSimMachineParam_T* mPtr = MotorPtr->MachinePtr;
    real32_T rotorSensorPosE;
    real32_T angleDiff;
    real32_T absErr;
    real32_T gain;
    real32_T omegaE;
    real32_T feedforward;

    /* Increment loop counter for diagnostic purposes */
    iPtr->LoopCounter++;

    /* Clamp RPM reference to maximum speed and convert to rad/s */
    iPtr->AngularVelocityRefRpmM = EmbedSim_ClampValue(iPtr->AngularVelocityRefRpmM,
                                                       -MAX_SPEED_RPM, MAX_SPEED_RPM);
    iPtr->RotorVelocityRefM = CON_RPM_TO_RAD(iPtr->AngularVelocityRefRpmM);

    /* Copy observer estimates (already validated by the observer module) */
    iPtr->RotorPositionObsEstM = iPtr->RotorPositionSensorM;
    iPtr->RotorSpeedObsEstM    = iPtr->RotorSpeedSensorM;

    /* Only update the model angle when in closed‑loop control */
    if (iPtr->SwitchToClosedLoop == 0x1U)
    {
        /* Compute electrical angle from mechanical position */
        rotorSensorPosE = iPtr->RotorPositionObsEstM * mPtr->PolePairs;
        EmbedSim_WrapAngleTwoPi(&rotorSensorPosE);

        /* Shortest signed angular distance [-π, π) */
        angleDiff = EmbedSim_AngleDistance(rotorSensorPosE, mPtr->SvmRotorThetaE);
        absErr = fabsf(angleDiff);

        /* ---------- Smooth gain scheduling ---------- */
        if (absErr < ES_ANGLE_CORR_THRESHOLD_RAD)
        {
            gain = 1.0F;   /* Full correction */
        }
        else
        {
            /* Gain smoothly transitions from ES_ANGLE_CORR_SLOW_GAIN to 1.0 */
            gain = ES_ANGLE_CORR_SLOW_GAIN +
                   (1.0F - ES_ANGLE_CORR_SLOW_GAIN) * (ES_ANGLE_CORR_THRESHOLD_RAD / absErr);
            gain = EmbedSim_ClampValue(gain, ES_ANGLE_CORR_SLOW_GAIN, 1.0F);
        }
        mPtr->SvmRotorThetaE += gain * angleDiff;

        /* ---------- Half‑sample delay compensation (only when locked) ---------- */
        if (absErr < ES_ANGLE_CORR_THRESHOLD_RAD)
        {
            omegaE = CON_RPM_TO_RAD(iPtr->RotorSpeedObsEstM) * mPtr->PolePairs;
            feedforward = omegaE * ES_MEASUREMENT_DELAY_FACTOR * iPtr->SampleTime;
            feedforward = EmbedSim_ClampValue(feedforward,
                                              -ES_MAX_ANGLE_STEP_RAD,
                                               ES_MAX_ANGLE_STEP_RAD);
            mPtr->SvmRotorThetaE += feedforward;
        }

        /* Keep angle within [0, 2π) */
        EmbedSim_WrapAngleTwoPi(&mPtr->SvmRotorThetaE);
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
    EmbedSim_ExecuteObserver(MotorPtr);

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
uint32_T EmbedSim_IsMotorSpinning(const EmbedSimCtrlInput_T* const InputPtr, real32_T SpeedRefRPM, real32_T Duration)
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
uint32_T EmbedSim_IsNotSpinning(const EmbedSimCtrlInput_T* const InputPtr, uint32_T PastIndex)
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
        if(fabsf(speed) < 0.2F)
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


void EmbedSim_GetMotorState(EmbedSimMachine_T* const motorPtr, EmbedSimMotorState_T* const statePtr)
{
    EmbedSimCtrlInput_T*    inputPtr  = motorPtr->InputPtr;
    EmbedSimCtrlOutput_T*   outputPtr = motorPtr->OutputPtr;
    EmbedSimMachineParam_T* paraPtr   = motorPtr->MachinePtr;
    FocUvw_T uvw;
    FocAlphaBeta_T ab;
    FocAngle_T angle;
    FocDq_T dq;

    /* ===== Mechanical ===== */
    statePtr->SpeedRpm = inputPtr->RotorSpeedObsEstM;;
    statePtr->PositionRad = inputPtr->RotorPositionObsEstM;


    /* ===== PWM ===== */
    statePtr->DutyU = outputPtr->DutyU;
    statePtr->DutyV = outputPtr->DutyV;
    statePtr->DutyW = outputPtr->DutyW;
    statePtr->SvmSector = outputPtr->SvmSector;


    /* ===== Control Mode ===== */
    statePtr->SwitchToClosedLoop = inputPtr->SwitchToClosedLoop;
    statePtr->LoopCounter        = inputPtr->LoopCounter;
    statePtr->Valid              = inputPtr->Valid;
}


void EmbedSim_CalculateJerkLimitedTrajectory(EmbedSimCtrlInput_T* const InputPtr, const EmbedSimMachineParam_T* const ParaPtr)
{
    real32_T targetOmega;
    real32_T errorOmega;
    real32_T stoppingAccel;
    real32_T desiredAccel;
    real32_T jerkRequest;
    real32_T previousVelocity;
    real32_T previousAcceleration;

    real32_T sampleTime;
    real32_T jerkMax;
    real32_T accelMax;
    real32_T speedMax;

    (void)ParaPtr;

    sampleTime = InputPtr->SampleTime;

    /*
     * Convert the requested target speed from RPM to rad/s and
     * limit it to the configured maximum speed.
     */
    targetOmega = CON_RPM_TO_RAD(InputPtr->AngularVelocityRefRpmM);
    speedMax    = CON_RPM_TO_RAD(MAX_SPEED_RPM);
    accelMax    = CON_RPM_TO_RAD(MAX_ACCEL_RPM);
    jerkMax     = CON_RPM_TO_RAD(MAX_JERK_RPM);

    /*
     * Reset the dynamic trajectory states when control
     * re-initialization is requested. The current velocity is
     * retained as the starting point of the new trajectory.
     */
    if (InputPtr->ControlReInit == 1U)
    {
        InputPtr->RotorAccelerationRefM = 0.0F;
        InputPtr->RotorJerkRefM = 0.0F;
        InputPtr->ControlReInit = 0U;
    }

    /*
     * Calculate the remaining velocity error between the current
     * trajectory velocity and the requested target velocity.
     */
    errorOmega = targetOmega - InputPtr->RotorVelocityRefM;

    /*
     * If the velocity error and acceleration are sufficiently small,
     * consider the trajectory settled and remove residual dynamic states.
     */
    if((fabsf(errorOmega) < SPEED_SETTLE_TOL) && (fabsf(InputPtr->RotorAccelerationRefM) < 0.01F))
    {
        InputPtr->RotorVelocityRefM = targetOmega;
        InputPtr->RotorAccelerationRefM = 0.0F;
        InputPtr->RotorJerkRefM = 0.0F;
    }
    else
    {
        /*
         * Calculate the acceleration required to remove the remaining
         * velocity error using the configured maximum jerk.
         *
         * a_stop = sqrt(2 * Jmax * |velocity_error|)
         */
        stoppingAccel = sqrtf(2.0F * jerkMax * fabsf(errorOmega));

        /*
         * Determine the acceleration direction from the velocity error
         * and limit the acceleration magnitude to the configured maximum.
         */
        desiredAccel =  ((errorOmega >= 0.0F) ? 1.0F : -1.0F) *  ((accelMax < stoppingAccel) ? accelMax : stoppingAccel);

        /*
         * Store the trajectory states from the previous sample.
         * These values are used for consistent integration.
         */
        previousVelocity = InputPtr->RotorVelocityRefM;
        previousAcceleration = InputPtr->RotorAccelerationRefM;

        /*
         * Calculate the jerk required to move the previous acceleration
         * toward the desired acceleration within one sample period.
         */
        jerkRequest =  (desiredAccel - previousAcceleration) / sampleTime;

        /*
         * Limit the requested jerk to the configured positive and
         * negative jerk limits.
         */
        InputPtr->RotorJerkRefM = EmbedSim_ClampValue(jerkRequest, -jerkMax, jerkMax);

        /*
         * Integrate velocity assuming constant jerk during
         * the current sample period.
         */
        InputPtr->RotorVelocityRefM =  previousVelocity +
                                       (previousAcceleration * sampleTime) +
                                       (0.5F * InputPtr->RotorJerkRefM * sampleTime * sampleTime);

        /*
         * Integrate acceleration using the applied jerk.
         */
        InputPtr->RotorAccelerationRefM = previousAcceleration + InputPtr->RotorJerkRefM * sampleTime;

        /*
         * Enforce the configured acceleration and velocity limits.
         */
        InputPtr->RotorAccelerationRefM = EmbedSim_ClampValue(InputPtr->RotorAccelerationRefM, -accelMax,  accelMax);

        InputPtr->RotorVelocityRefM = EmbedSim_ClampValue( InputPtr->RotorVelocityRefM, -speedMax,  speedMax);

        /*
         * Prevent the trajectory from crossing the target velocity.
         * If an overshoot is detected, settle directly at the target
         * and remove the remaining acceleration and jerk.
         */
        if(((errorOmega > 0.0F) && (InputPtr->RotorVelocityRefM > targetOmega))  ||
            ((errorOmega < 0.0F) && (InputPtr->RotorVelocityRefM < targetOmega)))
        {
            InputPtr->RotorVelocityRefM = targetOmega;
            InputPtr->RotorAccelerationRefM = 0.0F;
            InputPtr->RotorJerkRefM = 0.0F;
        }

        /*
         * Integrate position using the previous velocity and acceleration
         * and the constant jerk applied during the current sample.
         */
        InputPtr->RotorPositionRefM =  InputPtr->RotorPositionRefM +
                                       previousVelocity * sampleTime +
                                       0.5F * previousAcceleration *
                                       sampleTime * sampleTime +
                                      (1.0F / 6.0F) *  InputPtr->RotorJerkRefM *  sampleTime * sampleTime * sampleTime;
    }
}


void EmbedSim_WrapAngleTwoPi(real32_T* AnglePtr)
{
    *AnglePtr = fmodf(*AnglePtr, ES_MATH_2PI_F);
    if(*AnglePtr < 0.0F)
    {
        *AnglePtr += ES_MATH_2PI_F;
    }
}

 real32_T EmbedSim_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal)
 {
     real32_T result;

     if(Val < MinVal)
     {
         result = MinVal;
     }
     else if (Val > MaxVal)
     {
         result = MaxVal;
     }
     else
     {
         result = Val;
     }

     return result;
 }

real32_T EmbedSim_AngleDistance(real32_T Angle1, real32_T Angle2)
 {
     real32_T AngleDistance;

     AngleDistance = Angle1 - Angle2;

     if(AngleDistance >= ES_MATH_PI_F)
     {
         AngleDistance -= ES_MATH_2PI_F;
     }
     else if (AngleDistance < -ES_MATH_PI_F)
     {
         AngleDistance += ES_MATH_2PI_F;
     }

     return AngleDistance;
 }



void EmbedSim_ControlDebug(const EmbedSimMachine_T * const MotorPtr)
{
   const EmbedSimCtrlInput_T * const inputPtr = MotorPtr->InputPtr;
   const EmbedSimCtrlOutput_T * const outputPtr = MotorPtr->OutputPtr;
   const EmbedSimMachineParam_T * const paraPtr = MotorPtr->MachinePtr;

   printf("\n");
   printf("============================================================\n");
   printf("              EmbedSim_ControlStep DEBUG\n");
   printf("============================================================\n");

   /* ------------------------------------------------------------
    * Controller inputs
    * ------------------------------------------------------------ */
   printf("INPUTS\n");
   printf("------------------------------------------------------------\n");

   printf("  Iu                  = %10.5f A\n", inputPtr->Iu);
   printf("  Iv                  = %10.5f A\n", inputPtr->Iv);
   printf("  Iw                  = %10.5f A\n", inputPtr->Iw);

   printf("  RotorPosition       = %10.6f rad\n",
          inputPtr->RotorPositionSensorM);

   printf("  RotorSpeed          = %10.3f RPM\n",
          inputPtr->RotorSpeedSensorM);

   printf("  SpeedReference      = %10.3f RPM\n",
          inputPtr->AngularVelocityRefRpmM);

   printf("  Vdc                 = %10.4f V\n",
          inputPtr->Vdc);

   printf("  SampleTime          = %10.8f s\n",
          inputPtr->SampleTime);

   printf("  CtrlAlg             = %u\n",
          inputPtr->CtrlAlg);

   printf("  Valid               = %u\n",
          inputPtr->Valid);


   /* ------------------------------------------------------------
    * Machine parameters
    * ------------------------------------------------------------ */
   printf("\n");
   printf("MACHINE PARAMETERS\n");
   printf("------------------------------------------------------------\n");

   printf("  Vdc                 = %10.4f V\n",
          paraPtr->Vdc);


   /* ------------------------------------------------------------
    * Controller outputs
    * ------------------------------------------------------------ */
   printf("\n");
   printf("OUTPUTS\n");
   printf("------------------------------------------------------------\n");

   printf("  DutyU               = %10.6f\n",
          outputPtr->DutyU);

   printf("  DutyV               = %10.6f\n",
          outputPtr->DutyV);

   printf("  DutyW               = %10.6f\n",
          outputPtr->DutyW);

   printf("  Valid               = %u\n",
          outputPtr->Valid);

   printf("============================================================\n");

   fflush(stdout);
}



void EmbedSim_ControlStatePrint(const EmbedSimMotorState_T* const StatePtr)
{

   printf("\n");
   printf("============================================================\n");
   printf("              EmbedSim MOTOR STATE DEBUG\n");
   printf("============================================================\n");

   /* ===== Mechanical ===== */
   printf("MECHANICAL\n");
   printf("------------------------------------------------------------\n");
   printf("  Speed              = %10.3f RPM\n", StatePtr->SpeedRpm);



   printf("============================================================\n");
}













