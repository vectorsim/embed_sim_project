/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     DFC (Direct Field Control) controller implementation.
 *
 * \details   Implements differential‑flatness feedforward with speed and current PI loops.
 *            Startup is a 0.3 s open‑loop voltage ramp (modulation 0.05 → 0.20, fixed 300 RPM),
 *            exactly matching the Python version. After startup, the controller runs
 *            closed‑loop DFC indefinitely.
 *
 *            The state is controlled solely by `SwitchToClosedLoop`:
 *            - 0 : startup phase (open‑loop ramp)
 *            - 1 : closed‑loop DFC
 *
 *            Resetting is done via `ControlReInit = 1` (or calling `DFC_Reset()`),
 *            which sets `SwitchToClosedLoop = 0` and clears all integrators.
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule 17.2 : No recursion.
 *              - Rule 14.7 : Single return point.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase
 *              - Output pointers: PascalCase_P
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.0.0
 * \date      2026-08-17
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
 * \def DFC_MAX_CURRENT
 * \brief   Maximum current limit (A).
 */
#define DFC_MAX_CURRENT                 (100.0F)

/**
 * \def DFC_MAX_IQ_DOT_F
 * \brief   Maximum current derivative limit (A/s).
 */
#define DFC_MAX_IQ_DOT_F                (1000.0F)

/**
 * \def DFC_EPSILON_F
 * \brief   Numerical protection epsilon.
 */
#define DFC_EPSILON_F                   (1.0e-6F)

/**
 * \def DFC_SQRT3_F
 * \brief   Square root of 3 (for SVM voltage limit).
 */
#define DFC_SQRT3_F                     (1.7320508075688772F)

/*********************************************************************************************************************/
/*--------------------------------------------------Private Data-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \var speedIntegralError
 * \brief   Accumulated speed error for the outer speed PI controller.
 */
static real32_T speedIntegralError = 0.0F;

/**
 * \var idIntegralError
 * \brief   Accumulated d‑axis current error for the inner current PI.
 */
static real32_T idIntegralError    = 0.0F;

/**
 * \var iqIntegralError
 * \brief   Accumulated q‑axis current error for the inner current PI.
 */
static real32_T iqIntegralError    = 0.0F;

/**
 * \var startupThetaE
 * \brief   Electrical angle used during the open‑loop startup phase (rad).
 */
static real32_T startupThetaE = 0.0F;

/*********************************************************************************************************************/
/*--------------------------------------------Private Functions-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Wrap an angle to the range [0, 2π).
 *
 * \param[in,out] anglePtr  Pointer to the angle value to be wrapped (in radians).
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
 * \brief   Clamp a value to specified limits.
 *
 * \param[in] val     Value to clamp.
 * \param[in] minVal  Minimum allowed value.
 * \param[in] maxVal  Maximum allowed value.
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
 * \brief   Transform phase currents to the dq rotating reference frame.
 *
 * \param[in]  inputPtr   Pointer to control input structure (phase currents).
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

    currents.U = inputPtr->Iu;
    currents.V = inputPtr->Iv;
    currents.W = inputPtr->Iw;

    angle.ThetaE = inputPtr->RotorPositionObsEstM * machinePtr->PolePairs;
    DFC_WrapAngle(&angle.ThetaE);

    Clarke_Transform_Matrix(&currents, &alphaBeta);
    Park_Transform_Matrix(&alphaBeta, &angle, focDqPtr);
}

/*********************************************************************************************************************/
/*--------------------------------------------Public Functions------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize the DFC controller.
 *
 * \details Resets all integrators and the startup angle.
 *          The controller will start in open‑loop (SwitchToClosedLoop = 0)
 *          and run the startup ramp on the next step.
 *
 * \return  void
 */
void DFC_Init(void)
{
    speedIntegralError = 0.0F;
    idIntegralError    = 0.0F;
    iqIntegralError    = 0.0F;
    startupThetaE      = 0.0F;
}

/**
 * \brief   Reset the DFC controller state.
 *
 * \details Clears all integrators and resets the startup angle.
 *          The caller must also set `ControlReInit = 1` (or the reset
 *          will take effect on the next step when `ControlReInit` is
 *          handled in DFC_Step). This function is typically called
 *          from `EmbedSim_ResetController()`.
 *
 * \return  void
 */
void DFC_Reset(void)
{
    speedIntegralError = 0.0F;
    idIntegralError    = 0.0F;
    iqIntegralError    = 0.0F;
    startupThetaE      = 0.0F;
}

/**
 * \brief   Execute one step of Differential Flatness Control.
 *
 * \details The controller state is determined by `SwitchToClosedLoop`:
 *          - **0** : Startup phase – runs open‑loop voltage ramp for
 *                    `DFC_STARTUP_TIME_S` seconds (0.3 s by default).
 *          - **1** : Normal closed‑loop DFC.
 *
 *          Resetting is done by setting `ControlReInit = 1`, which
 *          clears integrators and forces `SwitchToClosedLoop = 0`
 *          to re‑enter the startup ramp.
 *
 *          This function has a single return point for MISRA compliance.
 *
 * \param[in]  motorPtr  Pointer to the motor structure.
 *
 * \return  void
 */
void DFC_Step(EmbedSimMachine_T* const motorPtr)
{
    EmbedSimCtrlInput_T* const inputPtr   = motorPtr->InputPtr;
    EmbedSimCtrlOutput_T* const outputPtr = motorPtr->OutputPtr;
    const EmbedSimMachineParam_T* const machinePtr = motorPtr->MachinePtr;

    /*
     * ---------- Local variables (one per line, MISRA Rule 8.5) ----------
     */
    real32_T omegaRef;
    real32_T omegaRefDot;
    real32_T omegaRefDDot;
    real32_T omegaMeas;

    real32_T speedError;
    real32_T torqueCorrection;
    real32_T torqueFeedforward;
    real32_T torqueRequired;

    real32_T torqueConstant;
    real32_T iqRef;
    real32_T iqRefDot;
    real32_T vdRef;
    real32_T vqRef;

    volatile real32_T idError;
    volatile real32_T iqError;
    volatile real32_T vdCorr;
    volatile real32_T vqCorr;

    FocDq_T dqCurrentMeas;
    FocDq_T dqVoltage;
    FocAngle_T focAngle;
    FocAlphaBeta_T abVoltage;
    SVM_DutyCycle_T svmDC;

    real32_T vMag;
    real32_T vPhaseMax;
    real32_T modulationIndex;

    static real32_T elapsed;
    real32_T ramp;
    static real32_T modulation = 0.0F;
    real32_T omegaStartupE;
    FocDq_T startupDqVoltage;
    FocAngle_T startupAngle;
    FocAlphaBeta_T startupAbVoltage;
    SVM_DutyCycle_T startupSvmDC;
    real32_T startupVMag;
    real32_T startupVPhaseMax;
    real32_T startupModIdx;

    /* ================================================================
     * 1. Reset on request (ControlReInit)
     * ================================================================ */
    if((EmbedSim_IsMotorSpinning(inputPtr, 80000U)==0x1U)  &&  (inputPtr->SwitchToClosedLoop != 0x1U) )
    {
        /* Startup time expired: switch to closed‑loop */
         inputPtr->SwitchToClosedLoop = 0x1U;
         DFC_Reset();
    }
    if (inputPtr->ControlReInit == 1U)
    {
        DFC_Reset();
        modulation = 0;
        inputPtr->SwitchToClosedLoop = 0x0U;
        inputPtr->ControlReInit = 0;

    }

    /* ================================================================
     * 2. Startup phase (open‑loop voltage ramp)
     *    Runs only when SwitchToClosedLoop == 0
     * ================================================================ */
    if (inputPtr->SwitchToClosedLoop == 0x0U)
    {


            modulation += DFC_STARTUP_MOD_MIN;
            modulation = DFC_ClampValue(modulation,
                                        DFC_STARTUP_MOD_MIN,
                                        DFC_STARTUP_MOD_MAX);

            omegaStartupE = machinePtr->PolePairs * CON_RPM_TO_RAD(inputPtr->AngularVelocityRefRpmM);
            startupThetaE += omegaStartupE * inputPtr->SampleTime;
            DFC_WrapAngle(&startupThetaE);

            startupDqVoltage.D = 0.0F;
            startupDqVoltage.Q = (machinePtr->Vdc / DFC_SQRT3_F) * modulation;

            startupAngle.ThetaE = startupThetaE;
            InvPark_Transform_Matrix(&startupDqVoltage,
                                     &startupAngle,
                                     &startupAbVoltage);

            startupVMag = sqrtf(startupAbVoltage.Alpha * startupAbVoltage.Alpha +
                                startupAbVoltage.Beta  * startupAbVoltage.Beta);
            startupVPhaseMax = machinePtr->Vdc / DFC_SQRT3_F;
            startupModIdx = startupVMag / startupVPhaseMax;
            startupModIdx = DFC_ClampValue(startupModIdx, 0.0F, 0.90F);
            SVM_CalculateDutyCycle(startupModIdx, &startupAngle, &startupSvmDC);

            /* Write startup output */
            outputPtr->DutyU = startupSvmDC.Ta;
            outputPtr->DutyV = startupSvmDC.Tb;
            outputPtr->DutyW = startupSvmDC.Tc;
            outputPtr->SvmSector = startupSvmDC.Sector;
            outputPtr->Valid = 0x1U;


    }

    /* ================================================================
     * 3. Normal DFC (closed‑loop) – only if SwitchToClosedLoop == 1
     * ================================================================ */
    if(inputPtr->SwitchToClosedLoop == 1U)
    {
        omegaRef     = inputPtr->RotorVelocityRefM;
        omegaRefDot  = inputPtr->RotorAccerlerationRefM;
        omegaRefDDot = inputPtr->RotorJerkRefM;
        omegaMeas    = CON_RPM_TO_RAD(inputPtr->RotorSpeedObsEstM);

        /* Speed PI (torque correction) */
        speedError = omegaRef - omegaMeas;
        speedIntegralError += speedError;
        speedIntegralError = DFC_ClampValue(speedIntegralError,
                                            -machinePtr->ParamPidIntegralLimit,
                                             machinePtr->ParamPidIntegralLimit);
        torqueCorrection = (machinePtr->ParamPidSpeedQProp * speedError) +
                           (machinePtr->ParamPidSpeedQInteg * speedIntegralError);

        /* Mechanical flatness */
        torqueFeedforward = (machinePtr->J * omegaRefDot) +
                            (machinePtr->B * omegaRef) +
                            machinePtr->TorqueLoad;
        torqueRequired = torqueFeedforward + torqueCorrection;

        /* Electrical flatness */
        torqueConstant = 1.5F * machinePtr->PolePairs * machinePtr->FluxPm;
        if (fabsf(torqueConstant) > DFC_EPSILON_F)
        {
            iqRef = torqueRequired / torqueConstant;
            iqRef = DFC_ClampValue(iqRef, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);
            iqRefDot = ((machinePtr->J * omegaRefDDot) +
                        (machinePtr->B * omegaRefDot)) / torqueConstant;
            iqRefDot = DFC_ClampValue(iqRefDot,
                                      -DFC_MAX_IQ_DOT_F,
                                      DFC_MAX_IQ_DOT_F);
        }
        else
        {
            iqRef    = 0.0F;
            iqRefDot = 0.0F;
        }

        /* Voltage feedforward */
        vdRef = -machinePtr->PolePairs * omegaRef * machinePtr->Lq * iqRef;
        vqRef = (machinePtr->Rs * iqRef) +
                (machinePtr->Lq * iqRefDot) +
                (machinePtr->PolePairs * omegaRef * machinePtr->FluxPm);

        /* Measure currents */
        DFC_CurrentsToDq(inputPtr, machinePtr, &dqCurrentMeas);

        /* Current PI */
        idError = 0.0F - dqCurrentMeas.D;
        iqError = iqRef - dqCurrentMeas.Q;

        idIntegralError += idError;
        iqIntegralError += iqError;
        idIntegralError = DFC_ClampValue(idIntegralError,
                                         -machinePtr->ParamPidIntegralLimit,
                                          machinePtr->ParamPidIntegralLimit);
        iqIntegralError = DFC_ClampValue(iqIntegralError,
                                         -machinePtr->ParamPidIntegralLimit,
                                          machinePtr->ParamPidIntegralLimit);

        idError = DFC_ClampValue(idError, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);
        iqError = DFC_ClampValue(iqError, -DFC_MAX_CURRENT, DFC_MAX_CURRENT);

        vdCorr = (machinePtr->ParamPidCurrentDProp * idError) +
                 (machinePtr->ParamPidCurrentDInteg * idIntegralError);
        vqCorr = (machinePtr->ParamPidCurrentQProp * iqError) +
                 (machinePtr->ParamPidCurrentQInteg * iqIntegralError);

        /* Final voltage */
        dqVoltage.D = vdRef + vdCorr;
        dqVoltage.Q = vqRef + vqCorr;

        /* Inverse Park */
        focAngle.ThetaE = inputPtr->RotorPositionObsEstM * machinePtr->PolePairs;
        DFC_WrapAngle(&focAngle.ThetaE);
        InvPark_Transform_Matrix(&dqVoltage, &focAngle, &abVoltage);

        /* SVM */
        vMag = sqrtf(abVoltage.Alpha * abVoltage.Alpha +
                     abVoltage.Beta  * abVoltage.Beta);
        vPhaseMax = machinePtr->Vdc / DFC_SQRT3_F;
        modulationIndex = vMag / vPhaseMax;
        modulationIndex = DFC_ClampValue(modulationIndex, 0.0F, 0.90F);
        SVM_CalculateDutyCycle(modulationIndex, &focAngle, &svmDC);

        /* Write normal DFC output */
        outputPtr->DutyU = svmDC.Ta;
        outputPtr->DutyV = svmDC.Tb;
        outputPtr->DutyW = svmDC.Tc;
        outputPtr->SvmSector = svmDC.Sector;
        outputPtr->Valid = 0x1U;

        startupThetaE = focAngle.ThetaE;
        if(EmbedSim_IsNotSpinning(inputPtr,100U)==0x1U)
        {
            inputPtr->ControlReInit = 0x1U;
        }
    }


    /* ================================================================
     * Single return point – MISRA Rule 14.7
     * ================================================================ */
    return;
}
