/**********************************************************************************************************************
 * \file        embed_sim_dfc_controller.c
 * \brief       TRUE DIFFERENTIAL FLATNESS CONTROL for PMSM
 * \version     3.0.0
 *
 * Flat outputs: y1 = θ, y2 = id
 * Pure feedforward + small correction. NO startup logic.
 *********************************************************************************************************************/

#include "embed_sim_dfc_controller.h"
#include "embed_sim_sv_pwm.h"
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_matrix.h"
#include "embed_sim_control.h"
#include <math.h>
#include <stddef.h>

/**********************************************************************************************************************
 * Private Definitions
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/


/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/


/**********************************************************************************************************************
 * Private Functions
 *********************************************************************************************************************/


/**
 * \brief  Read and transform sensor currents to dq reference frame.
 *
 * This function reads the three-phase current sensors (Iu, Iv, Iw) and
 * transforms them to the synchronous dq reference frame using the rotor
 * electrical angle derived from the position sensor.
 *
 * \param[in]  InputPtr      Pointer to input structure containing:
 *                           - Iu, Iv, Iw: Phase current sensor readings (A)
 *                           - RotorPositionSensor: Measured rotor position (rad)
 * \param[in]  MPtr          Pointer to motor parameters containing:
 *                           - PolePairs: Number of pole pairs
 * \param[out] SensorDqPtr   Pointer to output structure for dq currents:
 *                           - D: d-axis current from sensor (A)
 *                           - Q: q-axis current from sensor (A)
 */
static void DFC_ReadSensorDQ(EmbedSimCtrlInput_T* const InputPtr,
                              const EmbedSimMachineParam_T* const MPtr,
                              FocDq_T* const SensorDqPtr)
{
    FocAngle_T focAngle;
    FocUvw_T uvwCurrent;
    FocAlphaBeta_T alphaBetaCurrent;
    MatrixStatus_T status;
    real32_T id = 0.0F;
    real32_T iq = 0.0F;

    /* Calculate electrical angle from rotor position sensor */
    focAngle.ThetaE = InputPtr->RotorPositionSensor * MPtr->PolePairs;
    EmbedSim_WrapAngle(&focAngle.ThetaE);

    /* Read phase currents from sensors */
    uvwCurrent.U = InputPtr->Iu;
    uvwCurrent.V = InputPtr->Iv;
    uvwCurrent.W = InputPtr->Iw;

    /* Transform phase currents to alpha-beta (Clarke transform) */
    status = Clarke_Transform_Matrix(&uvwCurrent, &alphaBetaCurrent);
    if (status == MATRIX_SUCCESS)
    {
        /* Transform alpha-beta to dq (Park transform) */
        status = Park_Transform_Matrix(&alphaBetaCurrent, &focAngle, SensorDqPtr);
        if (status == MATRIX_SUCCESS)
        {
            id = SensorDqPtr->D;
            iq = SensorDqPtr->Q;
        }
    }

    /* Single exit point with results (or zeros on error) */
    SensorDqPtr->D = id;
    SensorDqPtr->Q = iq;
}

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

    angle.ThetaE = InputPtr->RotorPositionSensor * MPtr->PolePairs;
    EmbedSim_WrapAngle(&angle.ThetaE);

    Clarke_Transform_Matrix(&currents, &alphaBeta);
    Park_Transform_Matrix(&alphaBeta, &angle, FocDqPtr);
}

static void DFC_VoltageToDuty(const FocDq_T* const DqPtr,
                                FocAngle_T const AnglePtr,
                                const EmbedSimMachineParam_T* const MachinePtr,
                                SVM_DutyCycle_T* const DutyPtr)
{
    MatrixStatus_T status;
    FocAlphaBeta_T vAlphaBeta;
    real32_T Vmag = 0.0F;
    real32_T Vphase_max = 0.0F;
    real32_T modulationIndex = 0.0F;

    /* Default to 50% duty cycle */
    DutyPtr->Ta = 0.5F;
    DutyPtr->Tb = 0.5F;
    DutyPtr->Tc = 0.5F;
    DutyPtr->Sector = SVM_SECTOR_I;

    /* Inverse Park transform: DQ → AlphaBeta */
    status = InvPark_Transform_Matrix(DqPtr, &AnglePtr, &vAlphaBeta);

    if (status == MATRIX_SUCCESS) {
        /* Calculate magnitude of αβ voltage vector */
        Vmag = sqrtf((vAlphaBeta.Alpha * vAlphaBeta.Alpha) +
                     (vAlphaBeta.Beta * vAlphaBeta.Beta));

        /* Calculate modulation index from DC bus voltage */
        Vphase_max = MachinePtr->Vdc / SVM_SQRT3_F;
        modulationIndex = Vmag / Vphase_max;

        /* Clamp modulation index to safe range */
        modulationIndex = EmbedSim_Clamp(modulationIndex, 0.0F, 0.95F);

        /* Calculate SVPWM duty cycles */
        SVM_CalculateDutyCycle(modulationIndex, &AnglePtr, DutyPtr);
    }

}


/**********************************************************************************************************************
 * TRUE DIFFERENTIAL FLATNESS - PURE FLATNESS, NO STARTUP
 *********************************************************************************************************************/


/**********************************************************************************************************************
 * PUBLIC FUNCTIONS
 *********************************************************************************************************************/

void DFC_Init(void)
{

}

/**
 * \brief  Execute one step of Differential Flatness Control (DFC).
 *
 * This function implements the complete differential flatness mapping for PMSM control
 * as derived in the mathematical model. It reconstructs the full state from the flat
 * outputs (rotor angle and d-axis current) and computes the required feedforward dq voltages
 * with additional feedback corrections.
 *
 * \param[in]  InputPtr    Pointer to input structure containing:
 *                         - RotorPositionRef: Mechanical angle reference (rad)
 *                         - AngularVelocityRef: Mechanical speed reference (rad/s)
 *                         - AngularAccerlerationRef: Mechanical acceleration reference (rad/s²)
 *                         - AngularJerkRef: Mechanical jerk reference (rad/s³)
 *                         - Iu, Iv, Iw: Phase current sensor readings (A)
 *                         - RotorSpeedSensor: Measured rotor speed (RPM)
 *                         - RotorPositionSensor: Measured rotor position (rad)
 *                         - SampleTime: Control loop sample time (s)
 * \param[in]  MPtr        Pointer to motor parameters
 * \param[out] OutputPtr   Pointer to output structure where results are stored
 */
void DFC_Step(EmbedSimCtrlInput_T* const InputPtr,
              const EmbedSimMachineParam_T* const MPtr,
              EmbedSimCtrlOutput_T* const OutputPtr)
{
    /* Flat outputs (y1 and y2 from differential flatness theory) - References */
    real32_T rotorAngleRef;          /* y1 = θm_ref - mechanical angle reference (rad) */
    real32_T rotorSpeedRef;          /* y1_dot = ωm_ref - mechanical speed reference (rad/s) */
    real32_T rotorAccelRef;          /* y1_ddot = αm_ref - mechanical acceleration reference (rad/s²) */
    real32_T rotorJerkRef;           /* y1_dddot = jerk_ref - mechanical jerk reference (rad/s³) */
    real32_T dAxisCurrentRef;        /* y2 = id_ref - d-axis current reference (A) - ALWAYS 0 for SPMSM */
    real32_T dAxisCurrentRefDot;     /* y2_dot = d(id_ref)/dt - d-axis current derivative (A/s) - ALWAYS 0 */

    /* Feedforward states reconstructed from flat outputs */
    real32_T rotorAngleFF;           /* θm_FF - feedforward mechanical angle (rad) */
    real32_T rotorSpeedFF;           /* ωm_FF - feedforward mechanical speed (rad/s) */
    real32_T dAxisCurrentFF;         /* id_FF - feedforward d-axis current (A) - ALWAYS 0 */
    real32_T qAxisCurrentFF;         /* iq_FF - feedforward q-axis current (A) */
    real32_T qAxisCurrentFFDot;      /* d(iq_FF)/dt - feedforward q-axis current derivative (A/s) */

    /* Feedback variables */
    real32_T currentErrorId;         /* id_ref - id_meas (A) - id_ref = 0 */
    real32_T currentErrorIq;         /* iq_FF - iq_meas (A) */
    real32_T feedbackVd;             /* Feedback correction for d-axis voltage: kpId * currentErrorId (V) */
    real32_T feedbackVq;             /* Feedback correction for q-axis voltage: kpIq * currentErrorIq (V) */

    /* Feedforward + Feedback control inputs (dq voltages) */
    FocDq_T dqVoltageFF;             /* Feedforward dq voltages from differential flatness */
    FocDq_T dqVoltageTotal;          /* Total dq voltages (FF + FB) */

    /* Intermediate variables for flatness equations */
    real32_T torqueConstant;         /* Kt(id) = (3/2)*p*(ψf + (Ld-Lq)*id) */
    real32_T numerator;              /* N = J*αm + B*ωm + TL */
    real32_T denominator;            /* D = (3/2)*p*(ψf + (Ld-Lq)*id) */
    real32_T numeratorDot;           /* N_dot = J*jerk + B*αm */
    real32_T denominatorDot;         /* D_dot = (3/2)*p*(Ld-Lq)*id_dot */

    /* SVM variables */
    FocAngle_T focAngle;
    SVM_DutyCycle_T svmDC;

    /* Sensor feedback variables */
    real32_T rotorSpeedMeas;         /* Measured mechanical speed (rad/s) */
    real32_T rotorAngleMeas;         /* Measured electrical angle (rad) */
    FocDq_T dqCurrentMeas;           /* Measured currents in dq frame */

    /* PI controller gains (tunable) */
    const real32_T kpSpeed = 0.5F;   /* Proportional gain for speed control */
    const real32_T kpIq = 0.1F;      /* Proportional gain for Iq current control */
    const real32_T kpId = 0.1F;      /* Proportional gain for Id current control */

    /* --- Step 0: Read Sensor Feedback --- */
    DFC_CurrentsToDq(InputPtr, MPtr, &dqCurrentMeas);
    rotorSpeedMeas = CON_RPM_TO_RAD(InputPtr->RotorSpeedSensor);
    rotorAngleMeas = InputPtr->RotorPositionSensor * MPtr->PolePairs;
    EmbedSim_WrapAngle(&rotorAngleMeas);

    /* --- Step 1: Get reference flat outputs and their derivatives --- */
    rotorAngleRef = InputPtr->RotorPositionRef;
    rotorSpeedRef = InputPtr->AngularVelocityRef;
    rotorAccelRef = InputPtr->AngularAccerlerationRef;
    rotorJerkRef = InputPtr->AngularJerkRef;

    /* --- IMPORTANT: Id = 0 control for surface PMSM (SPMSM) ---
     * For SPMSM, Ld = Lq, so maximum torque per ampere (MTPA) is achieved
     * with id = 0. This simplifies the control significantly.
     */
    dAxisCurrentRef = 0.0F;           /* y2 = id_ref = 0 for SPMSM (Id=0 control) */
    dAxisCurrentRefDot = 0.0F;        /* d(id_ref)/dt = 0 (constant reference) */

    /* --- Step 2: State reconstruction from flat outputs (Feedforward) --- */
    /* Equation (32): θm_FF = y1 */
    rotorAngleFF = rotorAngleRef;

    /* Equation (33): ωm_FF = y˙1 */
    rotorSpeedFF = rotorSpeedRef;

    /* Equation (34): id_FF = y2 = 0 */
    dAxisCurrentFF = dAxisCurrentRef;

    /* Calculate torque constant Kt(id) from equation (8)
     * For SPMSM with id = 0: Kt = (3/2)*p*ψf (constant!)
     */
    torqueConstant = 1.5F * MPtr->PolePairs *
                     (MPtr->FluxPm + (MPtr->Ld - MPtr->Lq) * dAxisCurrentFF);
    /* Note: For SPMSM, (Ld - Lq) = 0, so Kt = (3/2)*p*ψf */

    /* Equation (35): iq_FF = (J*y¨1 + B*y˙1 + TL) / ((3/2)*p*(ψf + (Ld-Lq)*y2))
     * With y2 = id = 0: iq_FF = (J*αm + B*ωm + TL) / ((3/2)*p*ψf)
     */
    numerator = MPtr->J * rotorAccelRef + MPtr->B * rotorSpeedRef + MPtr->TorqueLoad;
    denominator = 1.5F * MPtr->PolePairs *
                  (MPtr->FluxPm + (MPtr->Ld - MPtr->Lq) * dAxisCurrentRef);

    /* Avoid division by zero */
    if (fabsf(denominator) > 1e-6F)
    {
        qAxisCurrentFF = numerator / denominator;
    }
    else
    {
        qAxisCurrentFF = 0.0F;
    }

    /* --- Step 3: Calculate derivative of feedforward q-axis current --- */
    /* Equation (25): N_dot = J*y¨¨1 + B*y¨1 + T˙L */
    numeratorDot = MPtr->J * rotorJerkRef + MPtr->B * rotorAccelRef;

    /* Equation (26): D_dot = (3/2)*p*(Ld-Lq)*y˙2
     * With y˙2 = d(id)/dt = 0: D_dot = 0
     */
    denominatorDot = 1.5F * MPtr->PolePairs * (MPtr->Ld - MPtr->Lq) * dAxisCurrentRefDot;

    /* Equation (27): i˙q_FF = (N_dot*D - N*D_dot) / D²
     * With D_dot = 0: i˙q_FF = N_dot / D
     */
    if (fabsf(denominator) > 1e-6F)
    {
        qAxisCurrentFFDot = (numeratorDot * denominator - numerator * denominatorDot) /
                            (denominator * denominator);
    }
    else
    {
        qAxisCurrentFFDot = 0.0F;
    }

    /* --- Step 4: Calculate feedforward control inputs using flatness --- */
    /* Equation (36): vd_FF = Rs*y2 + Ld*y˙2 - p*y˙1*Lq*iq_FF
     * With y2 = 0 and y˙2 = 0: vd_FF = -p*ωm*Lq*iq_FF
     */
    dqVoltageFF.D = MPtr->Rs * dAxisCurrentRef +
                    MPtr->Ld * dAxisCurrentRefDot -
                    MPtr->PolePairs * rotorSpeedFF * MPtr->Lq * qAxisCurrentFF;

    /* Equation (37): vq_FF = Rs*iq_FF + Lq*i˙q_FF + p*y˙1*(Ld*y2 + ψf)
     * With y2 = 0: vq_FF = Rs*iq_FF + Lq*i˙q_FF + p*ωm*ψf
     */
    dqVoltageFF.Q = MPtr->Rs * qAxisCurrentFF +
                    MPtr->Lq * qAxisCurrentFFDot +
                    MPtr->PolePairs * rotorSpeedFF *
                    (MPtr->Ld * dAxisCurrentRef + MPtr->FluxPm);

    /* --- Step 5: Calculate Feedback Corrections --- */
    /* Current feedback around DFC feedforward */
    /* vd = vd_FF + kpId * (id_ref - id_meas), with id_ref = 0 */
    currentErrorId = dAxisCurrentRef - dqCurrentMeas.D;
    feedbackVd = kpId * currentErrorId;

    /* vq = vq_FF + kpIq * (iq_FF - iq_meas) */
    currentErrorIq = qAxisCurrentFF - dqCurrentMeas.Q;
    feedbackVq = kpIq * currentErrorIq;

    /* --- Step 6: Combine Feedforward + Feedback --- */
    dqVoltageTotal.D = dqVoltageFF.D + feedbackVd;
    dqVoltageTotal.Q = dqVoltageFF.Q + feedbackVq;

    /* --- Step 7: Convert total dq voltages to PWM duty cycles using DFC_VoltageToDuty --- */
    focAngle.ThetaE = rotorAngleMeas;

    /* FIX: Pass focAngle by value (not pointer) */
    DFC_VoltageToDuty(&dqVoltageTotal, focAngle, MPtr, &svmDC);

    /* --- Step 8: Write Outputs --- */
    OutputPtr->DutyU = svmDC.Ta;
    OutputPtr->DutyV = svmDC.Tb;
    OutputPtr->DutyW = svmDC.Tc;
    OutputPtr->SvmSector = svmDC.Sector;
    OutputPtr->Valid = 0x1U;
}
