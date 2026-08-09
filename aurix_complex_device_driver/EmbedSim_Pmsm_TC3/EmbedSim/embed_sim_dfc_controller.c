/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     DFC (Direct Field Control) controller implementation.
 *
 * \details   Implements PI-based speed and current control loops for permanent magnet
 *            synchronous motors (PMSM). Includes anti-windup and output limiting.
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

#include "embed_sim_dfc_controller.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_matrix.h"
#include "embed_sim_control.h"
#include <math.h>



#define DFC_CURRENT_KP_D_F    (0.15F)
#define DFC_CURRENT_KP_Q_F    (0.1F)

/*********************************************************************************************************************/
/*--------------------------------------------------Private Data-----------------------------------------------------*/
/*********************************************************************************************************************/



/*********************************************************************************************************************/
/*--------------------------------------------Private Functions-----------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Wrap angle to [0, 2pi)
 *
 * \details Normalizes an angle to the range [0, 2π) using fmodf.
 *          Useful for rotor angle and Park transform calculations.
 *
 * \param[in,out] AnglePtr  Pointer to angle value to be wrapped (in radians).
 */
static void DFC_WrapAngle(real32_T* AnglePtr)
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
 * \brief   Transform currents to dq
 *
 * \details Converts phase currents (U, V, W) to dq rotating reference frame.
 *          Applies Clarke transform to get alpha-beta, then Park transform
 *          using the electrical angle (rotor position × pole pairs).
 *
 * \param[in]  InputPtr  Pointer to control input structure containing phase currents.
 * \param[in]  MPtr      Pointer to machine parameters (pole pairs).
 * \param[out] FocDqPtr  Pointer to dq current output structure.
 */
static void DFC_CurrentsToDq(EmbedSimCtrlInput_T* const InputPtr,
                             const EmbedSimMachineParam_T* const MPtr,
                             FocDq_T* const FocDqPtr)
{
    FocUvw_T currents;
    FocAlphaBeta_T alphaBeta;
    FocAngle_T angle;

    currents.U = InputPtr->Iu;
    currents.V = InputPtr->Iv;
    currents.W = InputPtr->Iw;

    angle.ThetaE = InputPtr->RotorPositionEst * MPtr->PolePairs;
    DFC_WrapAngle(&angle.ThetaE);

    Clarke_Transform_Matrix(&currents, &alphaBeta);
    Park_Transform_Matrix(&alphaBeta, &angle, FocDqPtr);
}

/**
 * \brief   Convert dq voltage to PWM
 *
 * \details Transforms dq voltage commands to PWM duty cycles using
 *          inverse Park transform and Space Vector Modulation (SVM).
 *          Includes over-modulation protection by clamping modulation index.
 *
 * \param[in]  DqPtr      Pointer to dq voltage commands.
 * \param[in]  AnglePtr   Pointer to rotor angle for inverse Park transform.
 * \param[in]  MachinePtr Pointer to machine parameters (Vdc).
 * \param[out] DutyPtr    Pointer to PWM duty cycle output structure.
 */
static void DFC_VoltageToDuty(const FocDq_T* const DqPtr,
                              FocAngle_T const AnglePtr,
                              const EmbedSimMachineParam_T* const MachinePtr,
                              SVM_DutyCycle_T* const DutyPtr)
{
    MatrixStatus_T status;
    FocAlphaBeta_T vAlphaBeta;
    real32_T vMag;
    real32_T vPhaseMax;
    real32_T modulationIndex;

    DutyPtr->Ta = 0.5F;
    DutyPtr->Tb = 0.5F;
    DutyPtr->Tc = 0.5F;
    DutyPtr->Sector = SVM_SECTOR_I;

    status = InvPark_Transform_Matrix(DqPtr, &AnglePtr, &vAlphaBeta);

    if (status == MATRIX_SUCCESS)
    {
        vMag = sqrtf((vAlphaBeta.Alpha * vAlphaBeta.Alpha) +
                     (vAlphaBeta.Beta * vAlphaBeta.Beta));

        vPhaseMax = MachinePtr->Vdc / 1.73205080757F;
        modulationIndex = vMag / vPhaseMax;
        modulationIndex = DFC_ClampValue(modulationIndex, 0.0F, 0.95F);

        SVM_CalculateDutyCycle(modulationIndex, &AnglePtr, DutyPtr);
    }
}




/*********************************************************************************************************************/
/*--------------------------------------------Public Functions------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialize DFC controller
 *
 * \details Sets default PI gains, limits, and anti-windup values for speed,
 *          Iq, and Id controllers. Sets Initialized flag to 1.
 *          Default values:
 *          - SpeedPI: Kp=0.4, Ki=0.0, UpperLimit=50.0, LowerLimit=-50.0, AntiWindup=0.1
 *          - IqPI:    Kp=0.2, Ki=0.0, UpperLimit=100.0, LowerLimit=-100.0, AntiWindup=0.1
 *          - IdPI:    Kp=0.1, Ki=0.0, UpperLimit=50.0, LowerLimit=-50.0, AntiWindup=0.1
 */
void DFC_Init(void)
{


}





/**
 * \brief  Execute one step of Differential Flatness Control.
 *
 * \details
 * Pure feedforward differential-flatness mapping for a PMSM.
 *
 * Assumptions for the first implementation:
 *   - Id_ref = 0
 *   - Ld = Lq
 *   - Load torque is constant
 *   - Rotor speed reference is generated by an S-curve
 *
 * The S-curve provides:
 *
 *   omega_ref
 *   omega_ref_dot
 *   omega_ref_ddot
 *
 * The differential-flatness mapping calculates:
 *
 *   Iq_ref
 *   Iq_ref_dot
 *   Vd_ref
 *   Vq_ref
 *
 * No PI controller is used in this version.
 *
 * \param[in]  InputPtr   Control input and reference values.
 * \param[in]  MPtr       PMSM machine parameters.
 * \param[out] OutputPtr  PWM output.
 */
void DFC_Step(EmbedSimCtrlInput_T * const InputPtr,
              const EmbedSimMachineParam_T * const MPtr,
              EmbedSimCtrlOutput_T * const OutputPtr)
{
    volatile real32_T omegaRef;
    volatile real32_T omegaRefDot;
    volatile real32_T omegaRefDDot;

    volatile real32_T iqRef;
    volatile real32_T iqRefDot;

    volatile real32_T vdRef;
    volatile real32_T vqRef;

    volatile real32_T torqueRequired;
    volatile real32_T torqueConstant;

    volatile real32_T rotorAngleMeas;
    volatile real32_T rotorSpeedMeas;

    volatile real32_T idError;
    volatile real32_T iqError ;

    FocDq_T dqVoltage;
    FocAngle_T focAngle;
    SVM_DutyCycle_T svmDC;
    FocDq_T dqCurrentMeas;

    /*
     * ------------------------------------------------------------
     * 1. Read reference trajectory
     * ------------------------------------------------------------
     */

    omegaRef     = InputPtr->AngularVelocityRef;
    omegaRefDot  = InputPtr->AngularAccerlerationRef;
    omegaRefDDot = InputPtr->AngularJerkRef;

    /*
     * ------------------------------------------------------------
     * 2. Mechanical flatness mapping
     *
     *     Te = J * omega_dot + B * omega + Tload
     * ------------------------------------------------------------
     */

    torqueRequired = (MPtr->J * omegaRefDot) + (MPtr->B * omegaRef) + MPtr->TorqueLoad;

    /*
     * For Id = 0:
     *
     *     Te = 1.5 * p * FluxPm * Iq
     *
     * Therefore:
     *
     *     Iq = Te / (1.5 * p * FluxPm)
     */

    torqueConstant =  1.5F * MPtr->PolePairs *  MPtr->FluxPm;

    if (fabsf(torqueConstant) > 1.0e-6F)
    {
        iqRef = torqueRequired / torqueConstant;

        /*
         * --------------------------------------------------------
         * 3. Differential of Iq reference
         *
         *     Iq_dot =
         *       (J * omega_ddot + B * omega_dot)
         *       / (1.5 * p * FluxPm)
         *
         * Load torque is assumed constant.
         * --------------------------------------------------------
         */

        iqRefDot =
            ((MPtr->J * omegaRefDDot) +
             (MPtr->B * omegaRefDot)) /
            torqueConstant;
    }
    else
    {
        iqRef = 0.0F;
        iqRefDot = 0.0F;
    }

    /*
     * ------------------------------------------------------------
     * 4. Differential-flatness voltage mapping and add feedback correction
     *
     * Id = 0
     *
     *     Vd = -p * omega * Lq * Iq
     *
     *     Vq = Rs * Iq
     *          + Lq * Iq_dot
     *          + p * omega * FluxPm
     * ------------------------------------------------------------
     */

    vdRef =  -MPtr->PolePairs * omegaRef *MPtr->Lq * iqRef;
    vqRef = (MPtr->Rs * iqRef) + (MPtr->Lq * iqRefDot) + (MPtr->PolePairs * omegaRef *  MPtr->FluxPm);

    DFC_CurrentsToDq(InputPtr, MPtr, &dqCurrentMeas);
    idError = 0.0F - dqCurrentMeas.D;
    iqError = vqRef - dqCurrentMeas.Q;




    /*
     * ------------------------------------------------------------
     * 5. Create dq voltage command
     * ------------------------------------------------------------
     */

    idError = 0.0F - dqCurrentMeas.D;
    iqError = iqRef - dqCurrentMeas.Q;

    dqVoltage.D = vdRef + (DFC_CURRENT_KP_D_F * idError);
    dqVoltage.Q = vqRef + (DFC_CURRENT_KP_Q_F * iqError);

    /*
     * ------------------------------------------------------------
     * 6. Get measured rotor angle
     * ------------------------------------------------------------
     */

    rotorSpeedMeas = CON_RPM_TO_RAD(InputPtr->RotorSpeedEst);

    (void)rotorSpeedMeas;

    rotorAngleMeas = InputPtr->RotorPositionEst *  MPtr->PolePairs;

    DFC_WrapAngle(&rotorAngleMeas);

    focAngle.ThetaE = rotorAngleMeas;

    /*
     * ------------------------------------------------------------
     * 7. dq voltage -> inverse Park -> SVM
     * ------------------------------------------------------------
     */

    DFC_VoltageToDuty( &dqVoltage,focAngle,MPtr, &svmDC);

    /*
     * ------------------------------------------------------------
     * 8. PWM outputs
     * ------------------------------------------------------------
     */

    OutputPtr->DutyU = svmDC.Ta;
    OutputPtr->DutyV = svmDC.Tb;
    OutputPtr->DutyW = svmDC.Tc;
    OutputPtr->SvmSector = svmDC.Sector;
    OutputPtr->Valid = 0x1U;
}


