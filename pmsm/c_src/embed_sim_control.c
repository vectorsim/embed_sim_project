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
    .PolePairs        = MP_POLES,
    .Rs               = MP_R_S,
    .Ld               = MP_L_D,
    .Lq               = MP_L_Q,
    .FluxPm           = MP_LAMBDA_PM,
    .J                = MP_J_ROTOR,
    .B                = MP_B_FRIC,
    .Vdc              = MP_V_DC
};

/**
 * \brief  Global control data
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
 * \brief   Apply smoothing to jerk signal
 *
 * \param[in] rawJerk         Raw jerk input value.
 * \param[in] previousJerk    Previous filtered jerk value.
 * \param[in] smoothingFactor Smoothing factor (0.0 to 1.0).
 *
 * \return  Filtered jerk value.
 */
static real32_T EmbedSim_SmoothJerk(real32_T rawJerk,
                                    real32_T previousJerk,
                                    real32_T smoothingFactor);

/**
 * \brief   Calculate reference trajectory
 *
 * \param[in,out] inputPtr  Pointer to control input structure containing
 *                          references and feedback signals.
 */
static void EmbedSim_CalculateRef(EmbedSimCtrlInput_T* const inputPtr);

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
 * \brief   Apply smoothing to jerk signal
 *
 * \details Performs a first-order low-pass filter (exponential smoothing) on
 *          the jerk signal. The smoothing factor determines the filter response:
 *          - 1.0 = no smoothing (raw jerk passed through)
 *          - 0.0 = maximum smoothing (previous jerk retained)
 *
 * \param[in] rawJerk         Raw jerk input value.
 * \param[in] previousJerk    Previous filtered jerk value.
 * \param[in] smoothingFactor Smoothing factor (0.0 to 1.0).
 *
 * \return  Filtered jerk value.
 */
static real32_T EmbedSim_SmoothJerk(real32_T rawJerk,
                                    real32_T previousJerk,
                                    real32_T smoothingFactor)
{
    real32_T smoothFactor;
    real32_T result;

    /* Clamp smoothing factor to valid range [0, 1] */
    smoothFactor = smoothingFactor;
    if (smoothFactor > 1.0F)
    {
        smoothFactor = 1.0F;
    }
    else if (smoothFactor < 0.0F)
    {
        smoothFactor = 0.0F;
    }
    else
    {
        /* No action - smoothingFactor is already within valid range */
    }

    /* Apply first-order low-pass filter */
    result = (smoothFactor * rawJerk) + ((1.0F - smoothFactor) * previousJerk);

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
static void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T* const inputPtr,
                                  EmbedSimMachineParam_T* const paraPtr,
                                  EmbedSimCtrlOutput_T* const outputPtr)
{
    static real32_T rotorAngleE = 0.0F;     /**< Electrical rotor angle [rad] */
    FocAngle_T      focAngle;              /**< Field-oriented control angle */
    FocDq_T         dqVoltage;             /**< dq voltage commands */
    SVM_DutyCycle_T svmDC;                 /**< SVM duty cycle outputs */
    real32_T        angularVelocityE;      /**< Electrical angular velocity [rad/s] */
    real32_T        modulation;            /**< Modulation index (fixed at 0.2) */

    /* Initialise outputs to safe default values */
    outputPtr->DutyU = 0.5F;
    outputPtr->DutyV = 0.5F;
    outputPtr->DutyW = 0.5F;
    outputPtr->Valid = 0x0U;

    modulation = 0.1F;  /* Fixed modulation index for open-loop */

    /* Only execute if input data is valid */
    if (inputPtr->Valid == 0x1U)
    {
        /* Calculate electrical angular velocity (mechanical × pole pairs) */
        angularVelocityE = inputPtr->AngularVelocityRef * paraPtr->PolePairs;

        /* Update rotor angle by integration */
        rotorAngleE += (angularVelocityE * inputPtr->SampleTime);
        EmbedSim_WrapAngle(&rotorAngleE);

        focAngle.ThetaE = rotorAngleE;

        /* Id = 0 control for surface PMSM */
        dqVoltage.D = 0.0F;

        /* Vq magnitude = modulation × (Vdc/√3) */
        dqVoltage.Q = (paraPtr->Vdc / SVM_SQRT3_F) * modulation;

        /* Convert dq voltage to PWM using SVPWM */
        if (SVM_CalculateDutyCycleFromDq(&dqVoltage, &focAngle,
                                         paraPtr->Vdc, &svmDC) == MATRIX_SUCCESS)
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
 * \brief   Calculate reference trajectory
 *
 * \details Generates smooth speed and position reference trajectories using
 *          jerk-limited acceleration profiling. Implements a closed-loop
 *          position observer that transitions from open-loop to closed-loop
 *          control based on rotor speed.
 *
 *          The trajectory generation uses:
 *            - Jerk-limited acceleration for smooth speed transitions
 *            - Speed settle tolerance to reduce overshoot
 *            - Sensor feedback for position correction when speed is valid
 *
 * \note    Acceleration is not explicitly clamped; it is limited by the
 *          jerk limits (MAX_JERK_RPM). This means the acceleration will
 *          naturally stay within bounds if the jerk limits are tuned
 *          appropriately.
 *
 * \param[in,out] inputPtr  Pointer to control input structure containing
 *                          references and feedback signals.
 */
static void EmbedSim_CalculateRef(EmbedSimCtrlInput_T* const inputPtr)
{
    static real32_T currentSpeedRpm = 0.0F;      /**< Current speed [RPM] */
    static real32_T currentAccelRpm = 0.0F;      /**< Current acceleration [RPM/s] */
    static real32_T currentJerkRpm = 0.0F;       /**< Current jerk [RPM/s²] */
    static real32_T currentPositionRad = 0.0F;   /**< Current position [rad] */
    static uint32_T isRolling = 0U;              /**< Flag indicating motor is rolling */

    real32_T dt;                 /**< Sample time [s] */
    real32_T targetSpeed;        /**< Target speed [RPM] */
    real32_T speedError;         /**< Speed error [RPM] */
    real32_T absSpeedError;      /**< Absolute speed error [RPM] */
    real32_T accelTarget;        /**< Target acceleration [RPM/s] */
    real32_T rawJerk;            /**< Raw jerk before smoothing [RPM/s²] */
    real32_T currentSpeedRad;    /**< Current speed [rad/s] */
    real32_T positionSensor;     /**< Sensor position [rad] */

    /* Initialise switch flag to open-loop by default */
    dt = inputPtr->SampleTime;
    positionSensor = inputPtr->RotorPositionSensor;

    /* Limit target speed to safe range */
    targetSpeed = EmbedSim_ClampValue(inputPtr->AngularVelocityRefRpm,
                                      -MAX_SPEED_RPM, MAX_SPEED_RPM);

    /* Signed speed error (needed for direction) */
    speedError = targetSpeed - currentSpeedRpm;
    absSpeedError = fabsf(speedError);

    /*
     * ------------------------------------------------------------
     * Speed trajectory generation with jerk limiting
     * ------------------------------------------------------------
     */

    /* Check if speed has settled within tolerance */
    if (absSpeedError < SPEED_SETTLE_TOL)
    {
        /* Speed settled - hold steady */
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

        /* Apply smoothing to jerk for smoother acceleration transitions */
        currentJerkRpm = EmbedSim_SmoothJerk(rawJerk, currentJerkRpm,
                                             JERK_SMOOTHING_FACTOR);

        /* Update acceleration and speed */
        currentAccelRpm += currentJerkRpm * dt;
        currentSpeedRpm += currentAccelRpm * dt;

        /* Prevent overshoot (still needed to avoid passing the target) */
        if (((speedError > 0.0F) && (currentSpeedRpm > targetSpeed)) ||
            ((speedError < 0.0F) && (currentSpeedRpm < targetSpeed)))
        {
            currentSpeedRpm = targetSpeed;
            currentAccelRpm = 0.0F;
            currentJerkRpm = 0.0F;
        }
    }

    /*
     * ------------------------------------------------------------
     * Position estimation and closed-loop transition logic
     * ------------------------------------------------------------
     */

    /* Convert speed to rad/s for position integration */
    currentSpeedRad = CON_RPM_TO_RAD(currentSpeedRpm);

    if (isRolling != 0U)
    {
        /* Already in rolling state */
        if ((inputPtr->RotorSpeedSensor > CLOSED_LOOP_MIN_SPEED) ||
            (inputPtr->RotorSpeedSensor < -CLOSED_LOOP_MIN_SPEED))
        {
            /* Sensor speed valid - use sensor position and allow closed-loop */
            currentPositionRad = positionSensor;
            inputPtr->SwitchToClosedLoop = 0x1U;
        }
        else
        {
            /* Sensor speed invalid - use estimated position, stay open-loop */
            inputPtr->SwitchToClosedLoop = 0x0U;
            currentPositionRad += currentSpeedRad * dt;
            EmbedSim_WrapAngle(&currentPositionRad);
        }
    }
    else
    {
        /* Not yet rolling - check if speed threshold is reached */
        if ((currentSpeedRpm > CLOSED_LOOP_MIN_SPEED) ||
            (currentSpeedRpm < -CLOSED_LOOP_MIN_SPEED))
        {
            /* Speed threshold reached - transition to rolling and allow closed-loop */
            isRolling = 1U;
            inputPtr->SwitchToClosedLoop = 0x1U;
            currentPositionRad = positionSensor;
        }
        else
        {
            /* Speed below threshold - use estimated position, stay open-loop */
            inputPtr->SwitchToClosedLoop = 0x0U;
            currentPositionRad += currentSpeedRad * dt;
            EmbedSim_WrapAngle(&currentPositionRad);
        }
    }

    /*
     * ------------------------------------------------------------
     * Write outputs to input structure
     * ------------------------------------------------------------
     */

    inputPtr->RotorPositionRef = currentPositionRad;
    inputPtr->AngularVelocityRef = currentSpeedRad;
    inputPtr->AngularAccerlerationRef = CON_RPM_TO_RAD(currentAccelRpm);
    inputPtr->AngularJerkRef = CON_RPM_TO_RAD(currentJerkRpm);
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
    /* Initialize SVM module */
    SVM_Init();
    /* Initialize DFC controller */
    DFC_Init();
    /* Initialize motor structure with pointers to data */
    TractionMotor_G.InputPtr   = &TractionMotorInput_G;
    TractionMotor_G.OutputPtr  = &TractionMotorOutput_G;
    TractionMotor_G.MachinePtr = &TractionMotorParams_G;

    TractionMotor_G.InputPtr->CtrlAlg =SIM_CTRL_DFC;  // SIM_CTRL_OPEN_LOOP;SIM_CTRL_DFC

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
void EmbedSim_ControlStep(EmbedSimMachine_T* const motorPtr)
{
    EmbedSimCtrlInput_T*    inputPtr  = motorPtr->InputPtr;
    EmbedSimCtrlOutput_T*   outputPtr = motorPtr->OutputPtr;
    EmbedSimMachineParam_T* paraPtr   = motorPtr->MachinePtr;

    /* Only execute if input data is valid */
    if (inputPtr->Valid == 0x1U)
    {
        /* Update DC bus voltage from input */
        paraPtr->Vdc = inputPtr->Vdc;

        /* Generate smooth reference trajectory */
        EmbedSim_CalculateRef(inputPtr);

        /* Select control mode based on closed-loop flag */
        if (inputPtr->SwitchToClosedLoop != 0x1U)
        {
            /* Open-loop control (startup / low speed) */
            EmbedSim_OpenLoopStep(inputPtr, paraPtr, outputPtr);
        }
        else
        {
            /* Closed-loop control - use sensor feedback */
            inputPtr->RotorPositionEst = inputPtr->RotorPositionSensor;
            inputPtr->RotorSpeedEst    = inputPtr->RotorSpeedSensor;

            /* Execute selected control algorithm */
            switch (inputPtr->CtrlAlg)
            {
                case SIM_CTRL_DFC:
                    DFC_Step(motorPtr);
                    break;

                default:
                    /* Fallback to open-loop if unknown algorithm */
                    EmbedSim_OpenLoopStep(inputPtr, paraPtr, outputPtr);
                    break;
            }
        }
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
    TractionMotorInput_G.RotorPositionSensor = RotorPositionSensor;
    TractionMotorInput_G.RotorSpeedSensor = RotorVelocitySensor;
    TractionMotorInput_G.AngularVelocityRefRpm = AngularVelocityRefRpm;
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
