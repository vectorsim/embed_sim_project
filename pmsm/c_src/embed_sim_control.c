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
 *              - Local variables: Lower pascalCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.0.0
 * \date      2026-08-09
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

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/


/*********************************************************************************************************************/
/*--------------------------------------------------Private Data-----------------------------------------------------*/
/*********************************************************************************************************************/

static EmbedSimMachineParam_T TractionMotorParams_G =
{
    .PolePairs        = MP_POLES,
    .Rs               = MP_R_S,
    .Ld               = MP_L_D,
    .Lq               = MP_L_Q,
    .FluxPm           = MP_LAMBDA_PM,
    .J                = MP_J_ROTOR,
    .B                = MP_B_FRIC,
    .Vdc              = MP_V_DC
};

EmbedSimCtrlInput_T    TractionMotorInput_G;
EmbedSimCtrlOutput_T   TractionMotorOutput_G;
EmbedSimMachine_T      TractionMotor_G;

/*********************************************************************************************************************/
/*-----------------------------------------Private Function Prototypes-----------------------------------------------*/
/*********************************************************************************************************************/

static void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T* const InputPtr, EmbedSimMachineParam_T*  const ParaPtr,EmbedSimCtrlOutput_T* const OutputPtr);

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \param[in,out] AnglePtr  Pointer to angle value to be wrapped (in radians).
 */
static void EmbedSim_WrapAngle(real32_T* AnglePtr);

/**
 * \brief   Clamp value to specified limits
 *
 * \param[in] Val     Value to clamp.
 * \param[in] MinVal  Minimum allowed value.
 * \param[in] MaxVal  Maximum allowed value.
 *
 * \return  Clamped value within [MinVal, MaxVal].
 */
static real32_T EmbedSim_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal);

/**
 * \brief   Apply smoothing to jerk signal
 *
 * \param[in] RawJerk         Raw jerk input value.
 * \param[in] PreviousJerk    Previous filtered jerk value.
 * \param[in] SmoothingFactor Smoothing factor (0.0 to 1.0).
 *
 * \return  Filtered jerk value.
 */
static real32_T EmbedSim_SmoothJerk(real32_T RawJerk, real32_T PreviousJerk, real32_T SmoothingFactor);

/**
 * \brief   Calculate reference trajectory
 *
 * \param[in,out] InputPtr  Pointer to control input structure containing
 *                          references and feedback signals.
 */
static void EmbedSim_CalculateRef(EmbedSimCtrlInput_T* const InputPtr);


/*********************************************************************************************************************/
/*--------------------------------------Private Function Implementations---------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \details Normalizes an angle to the range [0, 2π) using fmodf.
 *          Useful for rotor angle and Park transform calculations.
 *
 * \param[in,out] AnglePtr  Pointer to angle value to be wrapped (in radians).
 */
static void EmbedSim_WrapAngle(real32_T* AnglePtr)
{
    *AnglePtr = fmodf(*AnglePtr, SVM_2PI_F);
    if (*AnglePtr < 0.0F)
    {
        *AnglePtr += SVM_2PI_F;
    }
}

/**
 * \brief   Clamp value to specified limits
 *
 * \details Limits a value to a range defined by MinVal and MaxVal.
 *          If value is below MinVal, returns MinVal.
 *          If value is above MaxVal, returns MaxVal.
 *          Otherwise returns the original value.
 *
 * \param[in] Val     Value to clamp.
 * \param[in] MinVal  Minimum allowed value.
 * \param[in] MaxVal  Maximum allowed value.
 *
 * \return  Clamped value within [MinVal, MaxVal].
 */
static real32_T EmbedSim_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal)
{
    real32_T result;

    if (Val < MinVal)
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

/**
 * \brief   Apply smoothing to jerk signal
 *
 * \details Performs a first-order low-pass filter (exponential smoothing) on
 *          the jerk signal. The smoothing factor determines the filter response:
 *          - 1.0 = no smoothing (raw jerk passed through)
 *          - 0.0 = maximum smoothing (previous jerk retained)
 *
 * \param[in] RawJerk         Raw jerk input value.
 * \param[in] PreviousJerk    Previous filtered jerk value.
 * \param[in] SmoothingFactor Smoothing factor (0.0 to 1.0).
 *
 * \return  Filtered jerk value.
 */
static real32_T EmbedSim_SmoothJerk(real32_T RawJerk, real32_T PreviousJerk, real32_T SmoothingFactor)
{
    real32_T smoothingFactor;
    real32_T rawJerk;
    real32_T previousJerk;
    real32_T result;

    smoothingFactor = SmoothingFactor;
    rawJerk = RawJerk;
    previousJerk = PreviousJerk;

    if (smoothingFactor > 1.0F)
    {
        smoothingFactor = 1.0F;
    }
    else if (smoothingFactor < 0.0F)
    {
        smoothingFactor = 0.0F;
    }
    else
    {
        /* No action - smoothingFactor is already within valid range */
    }

    result = (smoothingFactor * rawJerk) + ((1.0F - smoothingFactor) * previousJerk);

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
 * \param[in]  InputPtr   Pointer to input structure containing:
 *                        - AngularVelocityRef: Target angular velocity (rad/s)
 *                        - SampleTime: Time step for integration (s)
 *                        - Valid: Flag indicating if input data is valid
 * \param[out] OutputPtr  Pointer to output structure where results are stored:
 *                        - DutyU, DutyV, DutyW: PWM duty cycles [0, 1]
 *                        - SvmSector: Active SVPWM sector (if valid)
 *                        - Valid: Status flag (0x1 if valid, 0x0 if invalid)
 *
 * \note The modulation index is fixed at 0.2 in this implementation.
 * \note The rotor angle is wrapped to the range [0, 2π) after each step.
 * \note This function uses the global MotorParams_G structure for motor parameters.
 */
void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T* const InputPtr, EmbedSimMachineParam_T*  const ParaPtr, EmbedSimCtrlOutput_T* const OutputPtr)
{
    static real32_T rotorAngleE = 0.0F;
    FocAngle_T      focAngle;
    FocDq_T         dqVoltage;
    SVM_DutyCycle_T svmDC;
    real32_T        angularVelocityE;
    real32_T        modulation;

    OutputPtr->DutyU = 0.5F;
    OutputPtr->DutyV = 0.5F;
    OutputPtr->DutyW = 0.5F;
    OutputPtr->Valid = 0x0U;

    modulation = 0.2F; /* fixed */

    if (InputPtr->Valid == 0x1U)
    {
        angularVelocityE = InputPtr->AngularVelocityRef * ParaPtr->PolePairs;
        rotorAngleE += (angularVelocityE * InputPtr->SampleTime);
        EmbedSim_WrapAngle(&rotorAngleE);

        focAngle.ThetaE = rotorAngleE;

        /* Id = 0 control for surface PMSM */
        dqVoltage.D = 0.0F;
        /* Vq magnitude = modulation * (Vdc/√3) */
        dqVoltage.Q = (ParaPtr->Vdc / SVM_SQRT3_F) * modulation;

        /* Use fixed SVPWM with Vdc parameter */
        if (SVM_CalculateDutyCycleFromDq(&dqVoltage, &focAngle, ParaPtr->Vdc, &svmDC) == MATRIX_SUCCESS)
        {
            OutputPtr->DutyU = svmDC.Ta;
            OutputPtr->DutyV = svmDC.Tb;
            OutputPtr->DutyW = svmDC.Tc;
            OutputPtr->SvmSector = svmDC.Sector;
            OutputPtr->Valid = 0x1U;
        }
    }
}


/**
 * \brief   Calculate reference trajectory
 *
 * \details Generates smooth speed and position reference trajectories using
 *          jerk-limited acceleration profiling. Implements a closed-loop
 *          position observer that transitions from open-loop to closed-loop
 *          control based on rotor speed.
 *
 * \note    Acceleration is not explicitly clamped; it is limited by the
 *          jerk limits (MAX_JERK_RPM). This means the acceleration will
 *          naturally stay within bounds if the jerk limits are tuned
 *          appropriately.
 *
 * \param[in,out] InputPtr  Pointer to control input structure containing
 *                          references and feedback signals.
 */
static void EmbedSim_CalculateRef(EmbedSimCtrlInput_T* const InputPtr)
{
    static real32_T currentSpeedRpm = 0.0F;
    static real32_T currentAccelRpm = 0.0F;
    static real32_T currentJerkRpm = 0.0F;
    static real32_T currentPositionRad = 0.0F;
    static uint32_T isRolling = 0U;

    uint32_T switchToCloseLoop;

    real32_T dt;
    real32_T targetSpeed;
    real32_T speedError;
    real32_T absSpeedError;
    real32_T accelTarget;
    real32_T rawJerk;
    real32_T currentSpeedRad;
    real32_T positionSensor;

    /* Initialise switch flag to open-loop by default */
    switchToCloseLoop = 0U;
    dt = InputPtr->SampleTime;
    positionSensor = InputPtr->RotorPositionSensor;

    /* Limit target speed */
    targetSpeed = EmbedSim_ClampValue(InputPtr->AngularVelocityRefRpm, -MAX_SPEED_RPM, MAX_SPEED_RPM);

    /* Signed speed error (needed for direction) */
    speedError = targetSpeed - currentSpeedRpm;
    absSpeedError = fabsf(speedError);

    /* Check if speed has settled */
    if (absSpeedError < SPEED_SETTLE_TOL)
    {
        currentSpeedRpm = targetSpeed;
        currentAccelRpm = 0.0F;
        currentJerkRpm = 0.0F;
    }
    else
    {
        /* Calculate desired acceleration (no hard limit – jerk will shape it) */
        accelTarget = speedError * 0.25F;

        /* Calculate raw jerk required to reach accelTarget */
        rawJerk = (accelTarget - currentAccelRpm) / dt;
        rawJerk = EmbedSim_ClampValue(rawJerk, -MAX_JERK_RPM, MAX_JERK_RPM);

        /* Apply smoothing and update acceleration */
        currentJerkRpm = EmbedSim_SmoothJerk(rawJerk, currentJerkRpm, JERK_SMOOTHING_FACTOR);
        currentAccelRpm += currentJerkRpm * dt;

        /* Update speed */
        currentSpeedRpm += currentAccelRpm * dt;

        /* Prevent overshoot (still needed to avoid passing the target) */
        if ( ((speedError > 0.0F) && (currentSpeedRpm > targetSpeed)) ||
             ((speedError < 0.0F) && (currentSpeedRpm < targetSpeed)) )
        {
            currentSpeedRpm = targetSpeed;
            currentAccelRpm = 0.0F;
            currentJerkRpm = 0.0F;
        }
    }

    /* Position update */
    currentSpeedRad = CON_RPM_TO_RAD(currentSpeedRpm);

    if (isRolling != 0U)
    {
        /* Already in rolling state */
        if ((InputPtr->RotorSpeedSensor > CLOSED_LOOP_MIN_SPEED) ||
            (InputPtr->RotorSpeedSensor < -CLOSED_LOOP_MIN_SPEED))
        {
            /* Sensor speed valid - use sensor position and allow closed-loop */
            currentPositionRad = positionSensor;
            switchToCloseLoop = 0x1U;
        }
        else
        {
            /* Sensor speed invalid - use estimated position, stay open-loop */
            switchToCloseLoop = 0x0U;
            currentPositionRad += currentSpeedRad * dt;
            EmbedSim_WrapAngle(&currentPositionRad);
        }
    }
    else
    {
        /* Not yet rolling */
        if ((currentSpeedRpm > CLOSED_LOOP_MIN_SPEED) ||
            (currentSpeedRpm < -CLOSED_LOOP_MIN_SPEED))
        {
            /* Speed threshold reached - transition to rolling and allow closed-loop */
            isRolling = 1U;
            switchToCloseLoop = 0x1U;
            currentPositionRad = positionSensor;
        }
        else
        {
            /* Speed below threshold - use estimated position, stay open-loop */
            switchToCloseLoop = 0x0U;
            currentPositionRad += currentSpeedRad * dt;
            EmbedSim_WrapAngle(&currentPositionRad);
        }
    }

    /* Write outputs */
    InputPtr->RotorPositionRef = currentPositionRad;
    InputPtr->AngularVelocityRef = currentSpeedRad;
    InputPtr->AngularAccerlerationRef = CON_RPM_TO_RAD(currentAccelRpm);
    InputPtr->AngularJerkRef = CON_RPM_TO_RAD(currentJerkRpm);
    InputPtr->SwitchToClosedLoop = switchToCloseLoop;
}


/*********************************************************************************************************************/
/*--------------------------------------Public Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize control module
 *
 * \details Initializes the SVPWM and DFC controller modules.
 *          Must be called once before any control step.
 */
void EmbedSim_ControlInit(void)
{
    SVM_Init();
    DFC_Init();

    TractionMotorInput_G.CtrlAlg = 0;
   /* Initialise Traction Motor */
    TractionMotor_G.InputPtr   = &TractionMotorInput_G;
    TractionMotor_G.OutputPtr  = &TractionMotorOutput_G;
    TractionMotor_G.MaschinePtr = &TractionMotorParams_G;
}

/**
 * \brief   Top-level PMSM control step
 *
 * \details Executes one control cycle. Based on the input validity and
 *          control mode, either open-loop or DFC control is performed.
 *          For DFC mode, estimates rotor position and speed from sensors.
 *
 * \param[in]     InputPtr  Pointer to control input structure.
 * \param[in,out] OutputPtr Pointer to control output structure.
 */
void EmbedSim_ControlStep(EmbedSimMachine_T*  const MotorPtr)
{
    EmbedSimCtrlInput_T*    inputPtr  = MotorPtr->InputPtr;
    EmbedSimCtrlOutput_T*   outputPtr = MotorPtr->OutputPtr;
    EmbedSimMachineParam_T* pPtr      = MotorPtr->MaschinePtr;
    if (inputPtr->Valid == 0x1U)
    {
        pPtr->Vdc = inputPtr->Vdc;
        EmbedSim_CalculateRef(inputPtr);

        if (inputPtr->SwitchToClosedLoop != 0x1U)
        {
            EmbedSim_OpenLoopStep(inputPtr, pPtr, outputPtr);
        }
        else
        {
            /* Substitute by Observer */
            inputPtr->RotorPositionEst = inputPtr->RotorPositionSensor;
            inputPtr->RotorSpeedEst    = inputPtr->RotorSpeedSensor;

            /* Select the Control */
            switch (inputPtr->CtrlAlg)
            {
                case SIM_CTRL_DFC:
                    DFC_Step(MotorPtr);
                    break;

                default:
                    EmbedSim_OpenLoopStep(inputPtr, pPtr, outputPtr);
                    break;
            }
        }
    }
}


void EmbedSim_CythonControlInit(void)
{
    EmbedSim_ControlInit();
}



extern void EmbedSim_CythonControlStep(
                                     /* input */
                                      real32_T  Iu,                  /*  [A] */
                                      real32_T  Iv,                    /*  [A] */
                                      real32_T  Iw,                      /*  [A] */
                                      real32_T  RotorPositionSensor,     /* RAD */
                                      real32_T RotorVelocitySensor,      /* RPM Mechanichal */
                                      real32_T  AngularVelocityRefRpm,   /* RPM Mechanichal */
                                      real32_T  Vdc,                     /*  [V]*/
                                      real32_T  SampleTime,
                                      uint32_T  CtrlAlg,
                                      uint32_T   ValidIn,
                                      /* output*/
                                      real32_T* PwmU,
                                      real32_T* PwmV,
                                      real32_T* PwmW,
                                      uint32_T*   ValidOut)
 {
    TractionMotorInput_G.Iu = Iu;
    TractionMotorInput_G.Iv = Iv;
    TractionMotorInput_G.Iw = Iw;


    TractionMotorInput_G.RotorPositionSensor = RotorPositionSensor;
    TractionMotorInput_G.RotorSpeedSensor    = RotorVelocitySensor;
    TractionMotorInput_G.AngularVelocityRefRpm =  AngularVelocityRefRpm;

     TractionMotorInput_G.SampleTime = SampleTime;  // 50us = 20 kHz
     TractionMotorInput_G.Vdc = Vdc;

    TractionMotorInput_G.Valid = ValidIn;

    EmbedSim_ControlStep(&TractionMotor_G);

   *PwmU  = TractionMotorOutput_G.DutyU;
   *PwmV  = TractionMotorOutput_G.DutyV;
   *PwmW  = TractionMotorOutput_G.DutyW;
   *ValidOut = TractionMotorOutput_G.Valid;

}
