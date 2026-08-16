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
    .ParamPidIntegralLimit  =  DFC_INTEGRAL_LIMIT_F
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
 * \brief   Calculate reference trajectory
 *
 * \param[in,out] inputPtr  Pointer to control input structure containing
 *                          references and feedback signals.
 */
static void EmbedSim_ExceuteObserver(EmbedSimCtrlInput_T* const inputPtr);

static void EmbedSim_CalculateTimeOptimalSCurve(EmbedSimCtrlInput_T*   const InputPtr,
                                                EmbedSimMachineParam_T* const ParaPtr);

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
    /* Initialize SVM module */
    SVM_Init();
    /* Initialize DFC controller */
    DFC_Init();
    /* Initialize motor structure with pointers to data */
    TractionMotor_G.InputPtr   = &TractionMotorInput_G;
    TractionMotor_G.OutputPtr  = &TractionMotorOutput_G;
    TractionMotor_G.MachinePtr = &TractionMotorParams_G;
    TractionMotor_G.InputPtr->SwitchToClosedLoop = 0x0U;
    TractionMotor_G.InputPtr->LoopCounter        = 0x0U;
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

        EmbedSim_ExceuteObserver(inputPtr);
        /* Update DC bus voltage from input */
        paraPtr->Vdc = inputPtr->Vdc;


        /* Select control mode based on closed-loop flag */
        if((inputPtr->CtrlAlg == 0U) || (inputPtr->SwitchToClosedLoop == 0x0U))
        {
            /* Open-loop control (startup / low speed) - use smooth ref */
            EmbedSim_OpenLoopStep(inputPtr, paraPtr, outputPtr);
        }
        else
        {
            /* Closed-loop control */
            /* Execute selected control algorithm */
            switch (inputPtr->CtrlAlg)
            {
                case SIM_CTRL_DFC:
                    /* Generate smooth reference trajectory using Time-Optimal S-Curve */
                    EmbedSim_CalculateTimeOptimalSCurve(inputPtr, paraPtr);
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
 * \brief   Calculates Time Optimal Spline for Differential Flatness
 *
 * \details Implements a time-optimal S-curve trajectory generator with
 *          jerk-limited acceleration profile. The trajectory consists of
 *          three phases:
 *          1. Acceleration with increasing jerk (+J_max)
 *          2. Acceleration with decreasing jerk (-J_max)
 *          3. Constant speed (zero jerk)
 *
 *          The time-optimal profile minimizes the time to reach the target
 *          speed while respecting the jerk limit.
 *
 * \param[in,out] InputPtr  Pointer to control input structure containing
 *                          references and feedback signals.
 * \param[in]     ParaPtr   Pointer to motor parameters (for J, B, etc.)
 */
/**
 * \brief   Calculates online time-optimal jerk-limited trajectory
 *
 * \details Generates omega_ref and its derivatives online.
 *
 *          State:
 *              omega_ref_dot  = acceleration
 *
 *          Control:
 *              jerk
 *
 *          At every sample:
 *
 *              1. Calculate speed error
 *              2. Determine direction to target
 *              3. Calculate braking distance
 *              4. Select jerk:
 *
 *                 +Jmax : accelerate toward target
 *                  0    : hold maximum acceleration
 *                 -Jmax : remove acceleration before target
 *
 *              5. Integrate acceleration
 *              6. Integrate velocity
 *
 *          No predefined T1 or TTotal is required.
 *
 * \param[in,out] InputPtr  Pointer to control input structure.
 * \param[in]     ParaPtr   Pointer to motor parameters.
 */
void EmbedSim_CalculateTimeOptimalSCurve( EmbedSimCtrlInput_T* const InputPtr,
                                          EmbedSimMachineParam_T* const ParaPtr)
{
    /* Persistent trajectory states (in rad/s, rad/s²) */
    static real32_T sCurveOmegaRef = 0.0F;
    static real32_T sCurveAccelRef = 0.0F;
    static real32_T sCurvePositionRef = 0.0F;
    static real32_T sCurveLastTargetRpm = 0.0F;

    real32_T targetRpm;
    real32_T targetOmega;
    real32_T sampleTime;

    real32_T errorOmega;
    real32_T distanceToTarget;
    real32_T direction;

    real32_T jerkMax;          /* rad/s³ */
    real32_T accelMax;         /* rad/s² */
    real32_T speedMax;         /* rad/s   */

    real32_T stoppingAccel;    /* rad/s² */
    real32_T desiredAccel;     /* rad/s² */
    real32_T jerkRequest;      /* rad/s³ */
    real32_T jerk;             /* rad/s³ */
    real32_T newAccel;
    real32_T newOmega;
    real32_T newPosition;

    (void)ParaPtr;

    sampleTime = InputPtr->SampleTime;

    /* ------------------------------------------------------------
     * 1. Read target and limits (convert from RPM to rad/s)
     * ------------------------------------------------------------ */
    targetRpm   = InputPtr->AngularVelocityRefRpmM;
    targetOmega = InputPtr->RotorVelocityRefM;   /* already in rad/s */
    speedMax    = CON_RPM_TO_RAD(MAX_SPEED_RPM);
    accelMax    = CON_RPM_TO_RAD(MAX_ACCEL_RPM);
    jerkMax     = CON_RPM_TO_RAD(MAX_JERK_RPM);

    /* ------------------------------------------------------------
     * 2. Clamp target to max speed
     * ------------------------------------------------------------ */
    if (targetOmega > speedMax)  targetOmega = speedMax;
    if (targetOmega < -speedMax) targetOmega = -speedMax;

    /* ------------------------------------------------------------
     * 3. Initialise trajectory from measured speed on re‑init
     * ------------------------------------------------------------ */
    if (InputPtr->ControlReInit == 1U)
    {
        real32_T omegaSensor = CON_RPM_TO_RAD(InputPtr->RotorSpeedObsEstM);
        sCurveOmegaRef      = omegaSensor;
        sCurveAccelRef      = 0.0F;
        sCurvePositionRef   = InputPtr->RotorPositionObsEstM;
        sCurveLastTargetRpm = targetRpm;
        InputPtr->ControlReInit = 0x0U;
    }

    /* ------------------------------------------------------------
     * 4. If target changes significantly, just update last target
     *    (no replanning needed, algorithm is adaptive)
     * ------------------------------------------------------------ */
    if (fabsf(targetRpm - sCurveLastTargetRpm) > 10.0F)
    {
        sCurveLastTargetRpm = targetRpm;
    }

    /* ------------------------------------------------------------
     * 5. Speed error (use trajectory state, not measured)
     * ------------------------------------------------------------ */
    errorOmega = targetOmega - sCurveOmegaRef;
    distanceToTarget = fabsf(errorOmega);

    /* ------------------------------------------------------------
     * 6. Deadband: if very close to target and accel small, settle
     * ------------------------------------------------------------ */
    if (distanceToTarget < SPEED_SETTLE_TOL && fabsf(sCurveAccelRef) < 0.01F)
    {
        sCurveOmegaRef = targetOmega;
        sCurveAccelRef = 0.0F;
        jerk = 0.0F;
    }
    else
    {
        /* --------------------------------------------------------
         * 7. Direction toward target
         * -------------------------------------------------------- */
        direction = (errorOmega >= 0.0F) ? 1.0F : -1.0F;

        /* --------------------------------------------------------
         * 8. Calculate stopping acceleration:
         *    a_stop = sqrt( 2 * Jmax * |error| )
         *    This is the maximum acceleration that can be reduced
         *    to zero exactly at the target using max jerk.
         * -------------------------------------------------------- */
        stoppingAccel = sqrtf( 2.0F * jerkMax * distanceToTarget );

        /* --------------------------------------------------------
         * 9. Desired acceleration = direction * min(accelMax, a_stop)
         * -------------------------------------------------------- */
        desiredAccel = direction * ( (accelMax < stoppingAccel) ? accelMax : stoppingAccel );

        /* --------------------------------------------------------
         * 10. Compute jerk needed to reach desired acceleration in one step
         * -------------------------------------------------------- */
        jerkRequest = (desiredAccel - sCurveAccelRef) / sampleTime;

        /* --------------------------------------------------------
         * 11. Clamp jerk to ±jerkMax
         * -------------------------------------------------------- */
        if (jerkRequest > jerkMax)      jerk = jerkMax;
        else if (jerkRequest < -jerkMax) jerk = -jerkMax;
        else                             jerk = jerkRequest;

        /* --------------------------------------------------------
         * 12. Integrate acceleration: a += j * dt
         * -------------------------------------------------------- */
        newAccel = sCurveAccelRef + jerk * sampleTime;

        /* --------------------------------------------------------
         * 13. Clamp acceleration to ±accelMax
         * -------------------------------------------------------- */
        if (newAccel > accelMax)      newAccel = accelMax;
        else if (newAccel < -accelMax) newAccel = -accelMax;

        /* --------------------------------------------------------
         * 14. Integrate speed using second-order formula
         *     (matches Python's Euler but more accurate)
         * -------------------------------------------------------- */
        newOmega = sCurveOmegaRef + sCurveAccelRef * sampleTime + 0.5F * jerk * sampleTime * sampleTime;

        /* --------------------------------------------------------
         * 15. Clamp speed to ±speedMax
         * -------------------------------------------------------- */
        if (newOmega > speedMax)      newOmega = speedMax;
        else if (newOmega < -speedMax) newOmega = -speedMax;

        /* --------------------------------------------------------
         * 16. Prevent overshoot
         * -------------------------------------------------------- */
        if ( (direction > 0.0F && newOmega > targetOmega) ||
             (direction < 0.0F && newOmega < targetOmega) )
        {
            newOmega = targetOmega;
            newAccel = 0.0F;
            jerk = 0.0F;
        }

        /* --------------------------------------------------------
         * 17. Integrate position (second-order with jerk)
         *     (used by observer/controller; keep for consistency)
         * -------------------------------------------------------- */
        newPosition = sCurvePositionRef
                    + sCurveOmegaRef * sampleTime
                    + 0.5F * sCurveAccelRef * sampleTime * sampleTime
                    + (1.0F / 6.0F) * jerk * sampleTime * sampleTime * sampleTime;

        /* --------------------------------------------------------
         * 18. Update states
         * -------------------------------------------------------- */
        sCurveOmegaRef   = newOmega;
        sCurveAccelRef   = newAccel;
        sCurvePositionRef = newPosition;
    }

    /* ------------------------------------------------------------
     * 19. Write outputs
     * ------------------------------------------------------------ */
    InputPtr->RotorPositionRefM      = sCurvePositionRef;
    InputPtr->RotorVelocityRefM      = sCurveOmegaRef;
    InputPtr->RotorAccerlerationRefM = sCurveAccelRef;
    InputPtr->RotorJerkRefM          = jerk;   /* dω/dt */
}
void EmbedSim_ExceuteObserver(EmbedSimCtrlInput_T* const InputPtr)
{


    InputPtr->LoopCounter++;

    /* Limit target speed to safe range */
    InputPtr->AngularVelocityRefRpmM = EmbedSim_ClampValue(InputPtr->AngularVelocityRefRpmM,-MAX_SPEED_RPM, MAX_SPEED_RPM);
    InputPtr->RotorVelocityRefM    = CON_RPM_TO_RAD(InputPtr->AngularVelocityRefRpmM);
    InputPtr->RotorPositionObsEstM = InputPtr->RotorPositionSensorM;  /* later from Observer */
    InputPtr->RotorSpeedObsEstM    = InputPtr->RotorSpeedSensorM;

     if(( InputPtr->LoopCounter >19500) && (fabs(InputPtr->RotorSpeedObsEstM)> CLOSED_LOOP_MIN_SPEED) && (InputPtr->SwitchToClosedLoop==0))
     {
         InputPtr->SwitchToClosedLoop = 0x1U;
         InputPtr->ControlReInit      = 0x1U;
     }
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
