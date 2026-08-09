/**********************************************************************************************************************
 * \file embed_sim_control.c
 * \brief Top-level PMSM control module with ENHANCED DFC
 *********************************************************************************************************************/

#include "embed_sim_control.h"
#include "embed_sim_motor_parameter.h"
#include "embed_sim_matrix.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_controller.h"
#include <stddef.h>
#include <math.h>

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

#define MIN_CURRENT_MAG         (0.0001F)
#define SPEED_FILTER_COEFF      (0.1F)
#define MAX_CURRENT_LIMIT       (100.0F)
#define TARGET_CURRENT          (10.0F)
#define VALID_FLAG              (0x1U)
#define INVALID_FLAG            (0x0U)

/* Open-loop test parameters */
#define OPEN_LOOP_VOLTAGE       (0.2F)      /**< Test voltage [modulation] */
#define OPEN_LOOP_SPEED_RPM     (500.0F)    /**< Test speed [RPM] */
/**********************************************************************************************************************
 * \file embed_sim_control.c
 * \brief Top-level PMSM control module with ENHANCED DFC
 *********************************************************************************************************************/

#include "embed_sim_control.h"
#include "embed_sim_motor_parameter.h"
#include "embed_sim_matrix.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_dfc_controller.h"
#include <stddef.h>
#include <math.h>

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

#define MIN_CURRENT_MAG         (0.0001F)
#define SPEED_FILTER_COEFF      (0.1F)
#define MAX_CURRENT_LIMIT       (100.0F)
#define TARGET_CURRENT          (10.0F)
#define VALID_FLAG              (0x1U)
#define INVALID_FLAG            (0x0U)

/* Open-loop test parameters */
#define OPEN_LOOP_VOLTAGE       (0.2F)      /**< Test voltage [modulation] */
#define OPEN_LOOP_SPEED_RPM     (500.0F)    /**< Test speed [RPM] */
#define OPEN_LOOP_RAMP_TIME     (2.0F)      /**< Ramp-up time [s] */


static  EmbedSimMachineParam_T MotorParams_G =
{
    .PolePairs        = 4.0F,
    .Rs               = 0.19F,
    .Ld               = 0.000125F,
    .Lq               = 0.000125F,
    .FluxPm           = 0.0014F,
    .J                = 2.4e-6F,
    .B                = 1.0e-6F,
    .RatedCurrent     = 5.0F,
    .RatedSpeed       = 2000.0F,
    .MaxSpeed         = 3000.0F,
    .MaxCurrent       = 10.0F,
    .TorqueConstant   = 0.008F,
    .BackEmfConstant  = 0.008F,
    .Vdc              = 12.0F
};

/*********************************************************************************************************************/
/*-----------------------------------------Private Function Prototypes-----------------------------------------------*/
/*********************************************************************************************************************/

static void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T* InputPtr, EmbedSimCtrlOutput_T* OutputPtr);
static uint32_T EmbedSim_CheckClosedLoopTransition(EmbedSimCtrlInput_T* InputPtr,
                                                    EmbedSimMachineParam_T* MachineParamPtr,
                                                    EmbedSimCtrlOutput_T* OutputPtr);

static real32_T EmbedSim_ClampValue(real32_T value, real32_T minVal, real32_T maxVal);
static uint32_T EmbedSim_IsValidFloat(real32_T value);
static real32_T EmbedSim_SmoothJerk(real32_T rawJerk, real32_T previousJerk, real32_T smoothingFactor);

/*********************************************************************************************************************/
/*--------------------------------------Private Function Implementations---------------------------------------------*/
/*********************************************************************************************************************/

static uint32_T EmbedSim_IsValidFloat(real32_T value)
{
    uint32_T valid = 0U;
    if (!isnan(value) && !isinf(value)) valid = 1U;
    return valid;
}

static real32_T EmbedSim_ClampValue(real32_T value, real32_T minVal, real32_T maxVal)
{
    real32_T result = value;
    if (result < minVal) result = minVal;
    else if (result > maxVal) result = maxVal;
    else { /* no action */ }
    return result;
}

void EmbedSim_WrapAngle(real32_T* anglePtr)
{
    if (anglePtr != NULL)
    {
        while (*anglePtr < 0.0F) *anglePtr += ES_MATH_2PI_F;
        while (*anglePtr >= ES_MATH_2PI_F) *anglePtr -= ES_MATH_2PI_F;
    }
}

static real32_T EmbedSim_SmoothJerk(real32_T rawJerk, real32_T previousJerk, real32_T smoothingFactor)
{
    if (smoothingFactor > 1.0F) smoothingFactor = 1.0F;
    else if (smoothingFactor < 0.0F) smoothingFactor = 0.0F;
    else { /* no action */ }
    return (smoothingFactor * rawJerk) + ((1.0F - smoothingFactor) * previousJerk);
}



/**
 * \brief  Execute one step of open-loop motor control.
 *
 * This function performs open-loop control for a surface PMSM motor using
 * Id=0 control strategy. It calculates the rotor angle based on the reference
 * angular velocity, generates DQ voltage commands, and converts them to
 * duty cycles using Space Vector PWM (SVPWM).
 *
 * The function maintains internal state for the rotor angle between calls.
 * Outputs are only valid when the input is marked as valid.
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
 * \note The rotor angle is wrapped to the range [-PI, PI] after each step.
 * \note This function uses the global MotorParams_G structure for motor parameters.
 */
static void EmbedSim_OpenLoopStep(EmbedSimCtrlInput_T* InputPtr, EmbedSimCtrlOutput_T* OutputPtr)
{
    static real32_T rotorAngleE = 0.0F;
    FocAngle_T      focAngle;
    FocDq_T         dqVoltage;
    SVM_DutyCycle_T svmDC;
    real32_T        dcV;
    real32_T        angularVelocityE;
    real32_T        modulation;

    OutputPtr->DutyU = 0.5F;
    OutputPtr->DutyV = 0.5F;
    OutputPtr->DutyW = 0.5F;
    OutputPtr->Valid = 0x0U;

    modulation = 0.2F; /* fixed */

    if (InputPtr->Valid == VALID_FLAG)
    {
        angularVelocityE = InputPtr->AngularVelocityRef * MotorParams_G.PolePairs;
        rotorAngleE += (angularVelocityE * InputPtr->SampleTime);
        EmbedSim_WrapAngle(&rotorAngleE);

        focAngle.ThetaE = rotorAngleE;

        /* Id = 0 control for surface PMSM */
        dqVoltage.D = 0.0F;
        /* Vq magnitude = modulation * (Vdc/√3) */
        dqVoltage.Q = (MotorParams_G.Vdc/SVM_SQRT3_F) * modulation;

        /* Use fixed SVPWM with Vdc parameter */
        if (SVM_CalculateDutyCycleFromDq(&dqVoltage, &focAngle, MotorParams_G.Vdc, &svmDC) == MATRIX_SUCCESS)
        {
            OutputPtr->DutyU = svmDC.Ta;
            OutputPtr->DutyV = svmDC.Tb;
            OutputPtr->DutyW = svmDC.Tc;
            OutputPtr->SvmSector = svmDC.Sector;
            OutputPtr->Valid = 0x1U;
        }
    }

}
/**********************************************************************************************************************
 * ENHANCED OPEN-LOOP STEP WITH DIAGNOSTICS
 * For testing: Back-EMF, Current sensors, Encoder
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * EmbedSim_GetOpenLoopDiag - Get open-loop diagnostics
 *------------------------------------------------------------------------------------------------------------------*/
/************************************************************************************************************/
/*--------------------------------------Public Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

void EmbedSim_ControlInit(void)
{
    SVM_Init();
    DFC_Init();
}



void EmbedSim_ControlStep(EmbedSimCtrlInput_T* InputPtr, EmbedSimCtrlOutput_T* OutputPtr)
{

    if (InputPtr->Valid == VALID_FLAG)
    {
        MotorParams_G.Vdc = InputPtr->Vdc;
        EmbedSim_CalculateRef(InputPtr);
        if(InputPtr->SwitchToClosedLoop != 01U)
        {
            EmbedSim_OpenLoopStep(InputPtr, OutputPtr);
            if(EmbedSim_CheckClosedLoopTransition(InputPtr,&MotorParams_G, OutputPtr)==0x1U)
            {
                InputPtr->SwitchToClosedLoop = 0x1U;
            }
        }
        else
        {
            //EmbedSim_OpenLoopStep(InputPtr, OutputPtr);
            DFC_Step(InputPtr, &MotorParams_G, OutputPtr); // shadow until implemented

        }
    }

}



void EmbedSim_EstimateRpm(EmbedSimMachineParam_T* MachineParamPtr, EmbedSimCtrlInput_T* InputPtr)
{
    /* Keep existing implementation */
}

void EmbedSim_CalculateRef(EmbedSimCtrlInput_T* InputPtr)
{
    static real32_T currentSpeedRpm = 0.0F;
    static real32_T currentAccelRpm = 0.0F;
    static real32_T currentJerkRpm = 0.0F;
    static real32_T currentPositionRad = 0.0F;
    static uint32_T isRolling = 0U;



    real32_T dt = InputPtr->SampleTime;
    if (dt <= 0.0F) dt = 0.00005F;

    real32_T targetSpeed = InputPtr->AngularVelocityRefRpm;
    if (targetSpeed > MAX_SPEED_RPM) targetSpeed = MAX_SPEED_RPM;
    else if (targetSpeed < -MAX_SPEED_RPM) targetSpeed = -MAX_SPEED_RPM;

    real32_T speedError = targetSpeed - currentSpeedRpm;
    real32_T absSpeedError = (speedError > 0.0F) ? speedError : -speedError;

    if (absSpeedError < SPEED_SETTLE_TOL)
    {
        currentSpeedRpm = targetSpeed;
        currentAccelRpm = 0.0F;
        currentJerkRpm = 0.0F;
    }
    else
    {
        real32_T accelTarget = speedError * 0.5F;
        if (accelTarget > MAX_ACCEL_RPM) accelTarget = MAX_ACCEL_RPM;
        else if (accelTarget < -MAX_ACCEL_RPM) accelTarget = -MAX_ACCEL_RPM;

        real32_T rawJerk = (accelTarget - currentAccelRpm) / dt;
        if (rawJerk > MAX_JERK_RPM) rawJerk = MAX_JERK_RPM;
        else if (rawJerk < -MAX_JERK_RPM) rawJerk = -MAX_JERK_RPM;

        currentJerkRpm = EmbedSim_SmoothJerk(rawJerk, currentJerkRpm, JERK_SMOOTHING_FACTOR);
        currentAccelRpm += currentJerkRpm * dt;

        if ((accelTarget > 0.0F) && (currentAccelRpm > accelTarget)) currentAccelRpm = accelTarget;
        else if ((accelTarget < 0.0F) && (currentAccelRpm < accelTarget)) currentAccelRpm = accelTarget;

        currentSpeedRpm += currentAccelRpm * dt;

        if ((speedError > 0.0F) && (currentSpeedRpm > targetSpeed))
        {
            currentSpeedRpm = targetSpeed;
            currentAccelRpm = 0.0F;
            currentJerkRpm = 0.0F;
        }
        else if ((speedError < 0.0F) && (currentSpeedRpm < targetSpeed))
        {
            currentSpeedRpm = targetSpeed;
            currentAccelRpm = 0.0F;
            currentJerkRpm = 0.0F;
        }
    }

    /* Position update */
    if (isRolling != 0U)
    {
        if ((InputPtr->RotorSpeedSensor > CLOSED_LOOP_MIN_SPEED) ||
            (InputPtr->RotorSpeedSensor < -CLOSED_LOOP_MIN_SPEED))
        {
            currentPositionRad = InputPtr->RotorPositionSensor;
        }
        else
        {
            currentPositionRad += CON_RPM_TO_RAD(currentSpeedRpm) * dt;
            while (currentPositionRad >= ES_MATH_2PI_F) currentPositionRad -= ES_MATH_2PI_F;
            while (currentPositionRad < 0.0F) currentPositionRad += ES_MATH_2PI_F;
        }
    }
    else
    {


        if ((currentSpeedRpm > CLOSED_LOOP_MIN_SPEED) ||
            (currentSpeedRpm < -CLOSED_LOOP_MIN_SPEED))
        {
            isRolling = 1U;
            currentPositionRad = InputPtr->RotorPositionSensor;
        }
        else
        {
            currentPositionRad += CON_RPM_TO_RAD(currentSpeedRpm) * dt;
            while (currentPositionRad >= ES_MATH_2PI_F) currentPositionRad -= ES_MATH_2PI_F;
            while (currentPositionRad < 0.0F) currentPositionRad += ES_MATH_2PI_F;
        }
    }

    InputPtr->AngularVelocityRefRpm = currentSpeedRpm;
    InputPtr->RotorPositionRef = currentPositionRad;
    InputPtr->AngularVelocityRef = CON_RPM_TO_RAD(currentSpeedRpm);
    InputPtr->RotorSpeedSensor = currentSpeedRpm;
}


/*********************************************************************************************************************/
/*--------------------------------------Private Function Implementations---------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*--------------------------------------Private Function Implementations---------------------------------------------*/
/*********************************************************************************************************************/

static uint32_T EmbedSim_CheckClosedLoopTransition(EmbedSimCtrlInput_T* InputPtr,
                                                    EmbedSimMachineParam_T* MachineParamPtr,
                                                    EmbedSimCtrlOutput_T* OutputPtr)
{
    static uint32_T good_counter = 0U;
    static uint32_T init = 0U;
    static real32_T Id_filt = 0.0F;
    static real32_T Iq_filt = 0.0F;

    real32_T dt;
    real32_T omega_e;
    real32_T alpha;
    real32_T Vdc, Va, Vb, Vc;
    FocUvw_T uvw_voltage, uvw_current;
    FocAlphaBeta_T alpha_beta_voltage, alpha_beta_current;
    FocAngle_T foc_angle;
    FocDq_T dq_voltage, dq_current;
    MatrixStatus_T status;
    uint32_T closed_loop_ready = 0U;

    /* Validate inputs - single exit point */
    if ((InputPtr != NULL) && (MachineParamPtr != NULL) && (OutputPtr != NULL))
    {
        /* If already in closed loop, reset and return */
        if (InputPtr->SwitchToClosedLoop != 0x0U)
        {
            good_counter = 0U;
            init = 0U;
            Id_filt = 0.0F;
            Iq_filt = 0.0F;
        }
        /* Check if we have valid measurements and speed */
        else if ((InputPtr->Valid == VALID_FLAG) &&
                 (InputPtr->AngularVelocityRef > 0.001F))
        {
            omega_e = InputPtr->AngularVelocityRef * MachineParamPtr->PolePairs;

            /* Sample time with safe default */
            dt = InputPtr->SampleTime;
            if (dt <= 0.0F) dt = 0.00005F;
            alpha = dt / (0.001F + dt);

            /* --- Transform currents to DQ --- */
            uvw_current.U = InputPtr->Iu;
            uvw_current.V = InputPtr->Iv;
            uvw_current.W = InputPtr->Iw;

            status = Clarke_Transform_Matrix(&uvw_current, &alpha_beta_current);
            if (status == MATRIX_SUCCESS)
            {
                foc_angle.ThetaE = InputPtr->RotorPositionRef * MachineParamPtr->PolePairs;
                status = Park_Transform_Matrix(&alpha_beta_current, &foc_angle, &dq_current);
                if (status == MATRIX_SUCCESS)
                {
                    /* Filter Id and Iq */
                    if (!init)
                    {
                        Id_filt = dq_current.D;
                        Iq_filt = dq_current.Q;
                        init = 1U;
                    }
                    else
                    {
                        Id_filt += alpha * (dq_current.D - Id_filt);
                        Iq_filt += alpha * (dq_current.Q - Iq_filt);
                    }

                    /* --- Reconstruct Vq from duty cycles --- */
                    Vdc = MachineParamPtr->Vdc;
                    Va = (OutputPtr->DutyU - 0.5F) * Vdc;
                    Vb = (OutputPtr->DutyV - 0.5F) * Vdc;
                    Vc = (OutputPtr->DutyW - 0.5F) * Vdc;

                    uvw_voltage.U = Va;
                    uvw_voltage.V = Vb;
                    uvw_voltage.W = Vc;

                    status = Clarke_Transform_Matrix(&uvw_voltage, &alpha_beta_voltage);
                    if (status == MATRIX_SUCCESS)
                    {
                        status = Park_Transform_Matrix(&alpha_beta_voltage, &foc_angle, &dq_voltage);
                        if (status == MATRIX_SUCCESS)
                        {
                            /* Calculate expected Iq from back-EMF */
                            real32_T Iq_expected = (dq_voltage.Q - omega_e * MachineParamPtr->FluxPm)
                                                   / MachineParamPtr->Rs;

                            /* Check if measured Iq matches expected */
                            if ((fabsf(Iq_filt - Iq_expected) < 0.5F) && (Iq_filt > 0.5F))
                            {
                                good_counter++;
                                if (good_counter > 100U)
                                {
                                    closed_loop_ready = 1U;
                                    good_counter = 0U;
                                    init = 0U;
                                    Id_filt = 0.0F;
                                    Iq_filt = 0.0F;
                                }
                            }
                            else
                            {
                                good_counter = 0U;
                            }
                        }
                    }
                }
            }
        }
    }

    return closed_loop_ready;
}

real32_T  EmbedSim_Clamp(real32_T value, real32_T min, real32_T max)
{
   real32_T result;

   if (value < min)
   {
       result = min;
   } else if (value > max)
   {
       result = max;
   } else
   {
       result = value;
   }

   return result;
}

