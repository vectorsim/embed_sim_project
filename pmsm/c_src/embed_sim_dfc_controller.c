/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     DFC (Differential Flatness Control) controller implementation.
 *
 * \details   Implements differential-flatness-based feedforward and speed-feedback
 *            control for permanent magnet synchronous motors (PMSM).
 *
 *            Controller structure (v2.2):
 *
 *              Mechanical layer:
 *                e_ω = ω* - ω
 *                ω̇_des = ω̇* + k_ω * e_ω
 *                T* = J * ω̇_des + B*ω + T_L
 *                i_q* = T* / K_T
 *
 *              Electrical layer (feedback-linearized):
 *                e_d = 0 - i_d
 *                e_q = i_q* - i_q
 *                v_d = R_s*i_d - p*ω*L_q*i_q + L_d*k_d*e_d
 *                v_q = R_s*i_q + L_q*(i̇_q_FF + k_q*e_q) + p*ω*(L_d*i_d + λ_PM)
 *
 * \version   2.2.0
 * \date      2026-08-12
 * \author    EmbedSim / EV Light Vehicle Foundation
 *********************************************************************************************************************/

#include "embed_sim_dfc_controller.h"
#include "embed_sim_motor_parameter.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_matrix.h"
#include "embed_sim_control.h"
#include <math.h>

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Speed feedback gain (Paper Eq. 55)
 *          k_ω > 0 determines convergence rate. τ = 1/k_ω
 */
#define DFC_SPEED_GAIN_F                (20.0F)

/**
 * \brief   Current feedback gains
 *          k_d, k_q > 0 determine current error convergence rate
 *          τ_d = 1/k_d, τ_q = 1/k_q
 *          Recommended: 500 s^-1 → τ = 2ms
 */
#define DFC_CURRENT_GAIN_D_F            (500.0F)
#define DFC_CURRENT_GAIN_Q_F            (500.0F)

/**
 * \brief   Maximum current derivative (A/s)
 *          Limits the rate of change of q-axis current
 */
#define DFC_MAX_IQ_DOT_F                (1000.0F)

/**
 * \brief   Numerical protection epsilon
 */
#define DFC_EPSILON_F                   (1.0e-6F)

/**
 * \brief   Maximum modulation index for SVM
 */
#define DFC_MAX_MODULATION_F            (0.95F)

/**
 * \brief   Square root of 3 (for SVM voltage limit)
 */
#define DFC_SQRT3_F                     (1.7320508075688772F)

/*********************************************************************************************************************/
/*--------------------------------------------------Private Data-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Controller gains (tunable at runtime via DFC_Init())
 */
static real32_T SpeedGain;
static real32_T CurrentGainD;
static real32_T CurrentGainQ;

/*********************************************************************************************************************/
/*--------------------------------------------Private Function Prototypes--------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Clamp value to specified limits
 */
static real32_T DFC_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal);

/**
 * \brief   Wrap angle to [0, 2pi)
 */
static void DFC_WrapAngle(real32_T * const AnglePtr);

/**
 * \brief   Transform phase currents to dq frame
 */
static void DFC_CurrentsToDq(EmbedSimCtrlInput_T * const InputPtr,
                             const EmbedSimMachineParam_T * const MachinePtr,
                             FocDq_T * const FocDqPtr);

/**
 * \brief   Limit voltage vector magnitude to MaxVoltage
 */
static void DFC_LimitVoltageVector(FocDq_T * const DqPtr, real32_T MaxVoltage);

/**
 * \brief   Convert dq voltage to SVM duty cycles
 */
static void DFC_VoltageToDuty(const FocDq_T * const DqPtr,
                              const FocAngle_T * const AnglePtr,
                              real32_T Vdc,
                              SVM_DutyCycle_T * const DutyPtr);

/*********************************************************************************************************************/
/*--------------------------------------------Private Functions-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Clamp value to specified limits
 */
static real32_T DFC_ClampValue(real32_T Val, real32_T MinVal, real32_T MaxVal)
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
 * \brief   Wrap angle to [0, 2pi)
 */
static void DFC_WrapAngle(real32_T * const AnglePtr)
{
    *AnglePtr = fmodf(*AnglePtr, SVM_2PI_F);

    if (*AnglePtr < 0.0F)
    {
        *AnglePtr += SVM_2PI_F;
    }
}

/**
 * \brief   Transform phase currents to dq frame
 */
static void DFC_CurrentsToDq(EmbedSimCtrlInput_T * const InputPtr,
                             const EmbedSimMachineParam_T * const MachinePtr,
                             FocDq_T * const FocDqPtr)
{
    FocUvw_T currents;
    FocAlphaBeta_T alphaBeta;
    FocAngle_T angle;

    currents.U = InputPtr->Iu;
    currents.V = InputPtr->Iv;
    currents.W = InputPtr->Iw;

    angle.ThetaE = InputPtr->RotorPositionEst * MachinePtr->PolePairs;
    DFC_WrapAngle(&angle.ThetaE);

    Clarke_Transform_Matrix(&currents, &alphaBeta);
    Park_Transform_Matrix(&alphaBeta, &angle, FocDqPtr);
}

/**
 * \brief   Limit voltage vector magnitude to MaxVoltage
 *
 * \param[in,out] DqPtr      Pointer to dq voltage command (modified)
 * \param[in]     MaxVoltage Maximum voltage magnitude (Vdc/√3)
 */
static void DFC_LimitVoltageVector(FocDq_T * const DqPtr, real32_T MaxVoltage)
{
    real32_T voltageMagnitude;
    real32_T scale;

    voltageMagnitude = sqrtf((DqPtr->D * DqPtr->D) +
                             (DqPtr->Q * DqPtr->Q));

    if (MaxVoltage <= DFC_EPSILON_F)
    {
        DqPtr->D = 0.0F;
        DqPtr->Q = 0.0F;
    }
    else if (voltageMagnitude > MaxVoltage)
    {
        scale = MaxVoltage / voltageMagnitude;
        DqPtr->D *= scale;
        DqPtr->Q *= scale;
    }
}

/**
 * \brief   Convert dq voltage to SVM duty cycles
 */
static void DFC_VoltageToDuty(const FocDq_T * const DqPtr,
                              const FocAngle_T * const AnglePtr,
                              real32_T Vdc,
                              SVM_DutyCycle_T * const DutyPtr)
{
    MatrixStatus_T status;
    FocAlphaBeta_T voltageAlphaBeta;
    real32_T voltageMagnitude;
    real32_T voltageMaximum;
    real32_T modulationIndex;

    DutyPtr->Ta = 0.5F;
    DutyPtr->Tb = 0.5F;
    DutyPtr->Tc = 0.5F;
    DutyPtr->Sector = SVM_SECTOR_I;

    status = InvPark_Transform_Matrix(DqPtr, AnglePtr, &voltageAlphaBeta);

    if (status == MATRIX_SUCCESS)
    {
        voltageMagnitude = sqrtf((voltageAlphaBeta.Alpha * voltageAlphaBeta.Alpha) +
                                 (voltageAlphaBeta.Beta * voltageAlphaBeta.Beta));

        voltageMaximum = Vdc / DFC_SQRT3_F;

        if (voltageMaximum > DFC_EPSILON_F)
        {
            modulationIndex = voltageMagnitude / voltageMaximum;
            modulationIndex = DFC_ClampValue(modulationIndex, 0.0F, DFC_MAX_MODULATION_F);

            SVM_CalculateDutyCycle(modulationIndex, AnglePtr, DutyPtr);
        }
    }
}

/*********************************************************************************************************************/
/*--------------------------------------------Public Functions------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize DFC controller
 *
 * \details Sets controller gains to configured values.
 *          Must be called before DFC_Step().
 */
void DFC_Init(void)
{
    SpeedGain = DFC_SPEED_GAIN_F;
    CurrentGainD = DFC_CURRENT_GAIN_D_F;
    CurrentGainQ = DFC_CURRENT_GAIN_Q_F;
}

/**
 * \brief   Execute DFC control step
 *
 * \details Implements the differential-flatness controller with
 *          feedback-linearization:
 *
 *            e_ω = ω* - ω
 *            ω̇_des = ω̇* + k_ω * e_ω
 *            T_cmd = J * ω̇_des + B*ω + T_L
 *            i_q_cmd = T_cmd / K_T
 *            i̇_q_FF = (J*ω̈* + B*ω̇*) / K_T
 *            e_d = 0 - i_d
 *            e_q = i_q_cmd - i_q
 *            v_d = R_s*i_d - p*ω*L_q*i_q + L_d*k_d*e_d
 *            v_q = R_s*i_q + L_q*(i̇_q_FF + k_q*e_q) + p*ω*(L_d*i_d + λ_PM)
 *
 * \param[in]  MotorPtr  Pointer to motor structure
 */
void DFC_Step(EmbedSimMachine_T * const MotorPtr)
{
    /* Pointers to input/output structures */
    EmbedSimCtrlInput_T *inputPtr;
    const EmbedSimMachineParam_T *machinePtr;
    EmbedSimCtrlOutput_T *outputPtr;

    /* Reference trajectory (already in rad/s, rad/s², rad/s³) */
    real32_T omegaRef;       /* ω* - desired angular velocity [rad/s]     */
    real32_T omegaRefDot;    /* ω̇* - desired angular acceleration [rad/s²] */
    real32_T omegaRefDDot;   /* ω̈* - desired angular jerk [rad/s³]        */

    /* Measurements - ACTUAL values */
    real32_T omegaMeas;      /* ω - actual measured speed [rad/s]        */
    FocDq_T dqCurrentMeas;   /* i_d, i_q - actual measured currents      */

    /* Speed loop variables */
    real32_T speedError;     /* e_ω = ω* - ω [rad/s]                     */
    real32_T desiredAccel;   /* ω̇_des = ω̇* + k_ω * e_ω [rad/s²]          */
    real32_T torqueCmd;      /* T_cmd [Nm]                               */

    /* Current variables */
    real32_T torqueConstant; /* K_T = 1.5 * p * λ_PM [Nm/A]              */
    real32_T iqRef;          /* i_q_cmd [A]                              */
    real32_T iqRefDotFF;     /* i̇_q_FF [A/s] - feedforward derivative    */

    /* Current errors */
    real32_T idError;        /* e_d = 0 - i_d [A]                        */
    real32_T iqError;        /* e_q = i_q_cmd - i_q [A]                  */

    /* Voltage commands */
    real32_T vdRef;          /* v_d_cmd [V]                              */
    real32_T vqRef;          /* v_q_cmd [V]                              */

    /* PWM */
    FocDq_T dqVoltage;
    FocAngle_T focAngle;
    SVM_DutyCycle_T svmDutyCycle;
    real32_T rotorAngleElectrical;

    /* Limits */
    real32_T maxTorque;      /* T_max [Nm]                               */
    real32_T maxVoltage;     /* V_max = Vdc/√3 [V]                       */

    /*--------------------------------------------------------------------
     * 1. Get pointers from motor structure
     *--------------------------------------------------------------------*/
    inputPtr = MotorPtr->InputPtr;
    machinePtr = MotorPtr->MaschinePtr;
    outputPtr = MotorPtr->OutputPtr;

    /*--------------------------------------------------------------------
     * 2. Calculate limits
     *--------------------------------------------------------------------*/
    maxVoltage = machinePtr->Vdc / DFC_SQRT3_F;

    /*--------------------------------------------------------------------
     * 3. Read reference trajectory (already clamped in CalculateRef)
     *--------------------------------------------------------------------*/
    omegaRef = inputPtr->AngularVelocityRef;
    omegaRefDot = inputPtr->AngularAccerlerationRef;
    omegaRefDDot = inputPtr->AngularJerkRef;

    /*--------------------------------------------------------------------
     * 4. Read measurements - ACTUAL speed and currents!
     *--------------------------------------------------------------------*/
    omegaMeas = inputPtr->RotorSpeedEst;  /* Already in rad/s */
    DFC_CurrentsToDq(inputPtr, machinePtr, &dqCurrentMeas);

    /*--------------------------------------------------------------------
     * 5. Speed error (Paper Eq. 51)
     *    e_ω = ω* - ω
     *--------------------------------------------------------------------*/
    speedError = omegaRef - omegaMeas;

    /*--------------------------------------------------------------------
     * 6. Desired acceleration with feedback (Paper Eq. 57)
     *    ω̇_des = ω̇* + k_ω * e_ω
     *--------------------------------------------------------------------*/
    desiredAccel = omegaRefDot + (SpeedGain * speedError);

    /*--------------------------------------------------------------------
     * 7. Torque command (Paper Eq. 59)
     *    T_cmd = J * ω̇_des + B*ω + T_L
     *--------------------------------------------------------------------*/
    torqueCmd = (machinePtr->J * desiredAccel) +
                (machinePtr->B * omegaMeas) +
                machinePtr->TorqueLoad;

    /* Torque constraint (Paper Eq. 111-113) */
    torqueConstant = 1.5F * machinePtr->PolePairs * machinePtr->FluxPm;
    maxTorque = fabsf(torqueConstant) * DFC_MAX_CURRENT;
    torqueCmd = DFC_ClampValue(torqueCmd, -maxTorque, maxTorque);

    /*--------------------------------------------------------------------
     * 8. Current command (Paper Eq. 68-69)
     *    i_q_cmd = T_cmd / K_T
     *--------------------------------------------------------------------*/
    if (fabsf(torqueConstant) > DFC_EPSILON_F)
    {
        iqRef = torqueCmd / torqueConstant;
        iqRef = DFC_ClampValue(iqRef, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);

        /*----------------------------------------------------------------
         * 9. Feedforward current derivative (Paper Eq. 50)
         *    i̇_q_FF = (J*ω̈* + B*ω̇*) / K_T
         *
         *    NOTE: This is the FEEDFORWARD derivative only.
         *    The feedback term (k_q * e_q) handles the difference
         *    between the actual and commanded current derivative.
         *----------------------------------------------------------------*/
        iqRefDotFF = ((machinePtr->J * omegaRefDDot) +
                      (machinePtr->B * omegaRefDot)) / torqueConstant;

        /* Clamp current derivative to prevent excessive rates */
        iqRefDotFF = DFC_ClampValue(iqRefDotFF, -DFC_MAX_IQ_DOT_F, DFC_MAX_IQ_DOT_F);
    }
    else
    {
        iqRef = 0.0F;
        iqRefDotFF = 0.0F;
    }

    /*--------------------------------------------------------------------
     * 10. Current errors
     *     e_d = i_d* - i_d = 0 - i_d
     *     e_q = i_q* - i_q
     *--------------------------------------------------------------------*/
    idError = 0.0F - dqCurrentMeas.D;
    iqError = iqRef - dqCurrentMeas.Q;

    /* Limit current error used by feedback-linearization (saturation) */
    idError = DFC_ClampValue(idError, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);
    iqError = DFC_ClampValue(iqError, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);

    /*--------------------------------------------------------------------
     * 11. d-axis voltage WITH CORRECT FEEDBACK-LINEARIZATION
     *
     *     v_d = R_s*i_d - p*ω*L_q*i_q + L_d*(i̇_d* + k_d*e_d)
     *
     *     with i_d* = 0, i̇_d* = 0:
     *
     *     v_d = R_s*i_d - p*ω*L_q*i_q + L_d*k_d*e_d
     *
     *     CRITICAL: Uses ACTUAL ω and ACTUAL i_q for decoupling!
     *--------------------------------------------------------------------*/
    vdRef = (machinePtr->Rs * dqCurrentMeas.D) -
            (machinePtr->PolePairs * omegaMeas * machinePtr->Lq * dqCurrentMeas.Q) +
            (machinePtr->Ld * (CurrentGainD * idError));

    /*--------------------------------------------------------------------
     * 12. q-axis voltage WITH CORRECT FEEDBACK-LINEARIZATION
     *
     *     v_q = R_s*i_q + L_q*(i̇_q_FF + k_q*e_q) + p*ω*(L_d*i_d + λ_PM)
     *
     *     CRITICAL: Uses ACTUAL ω and ACTUAL i_d for decoupling!
     *--------------------------------------------------------------------*/
    vqRef = (machinePtr->Rs * dqCurrentMeas.Q) +
            (machinePtr->Lq * (iqRefDotFF + (CurrentGainQ * iqError))) +
            (machinePtr->PolePairs * omegaMeas *
             ((machinePtr->Ld * dqCurrentMeas.D) + machinePtr->FluxPm));

    /*--------------------------------------------------------------------
     * 13. Voltage vector limiting (Paper Eq. 124)
     *     sqrt(v_d² + v_q²) ≤ V_max
     *--------------------------------------------------------------------*/
    dqVoltage.D = vdRef;
    dqVoltage.Q = vqRef;
    DFC_LimitVoltageVector(&dqVoltage, maxVoltage);

    /*--------------------------------------------------------------------
     * 14. Transform dq voltage to stationary frame and SVM
     *--------------------------------------------------------------------*/
    rotorAngleElectrical = inputPtr->RotorPositionEst * machinePtr->PolePairs;
    DFC_WrapAngle(&rotorAngleElectrical);

    focAngle.ThetaE = rotorAngleElectrical;
    DFC_VoltageToDuty(&dqVoltage, &focAngle, machinePtr->Vdc, &svmDutyCycle);

    /*--------------------------------------------------------------------
     * 15. PWM outputs
     *--------------------------------------------------------------------*/
    outputPtr->DutyU = svmDutyCycle.Ta;
    outputPtr->DutyV = svmDutyCycle.Tb;
    outputPtr->DutyW = svmDutyCycle.Tc;
    outputPtr->SvmSector = svmDutyCycle.Sector;
    outputPtr->Valid = 0x1U;

    /* Single exit point at end of function */
}