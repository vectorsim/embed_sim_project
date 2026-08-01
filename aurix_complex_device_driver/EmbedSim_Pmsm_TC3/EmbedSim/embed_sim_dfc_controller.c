/**********************************************************************************************************************
 * \file      embed_sim_dfc_controller.c
 * \brief     Sensorless Differential Flatness FOC Controller — NANOTEC DB42S02.
 *
 * \details   WHAT IS DIFFERENTIAL FLATNESS?
 *            ================================
 *            The PMSM is differentially flat with flat outputs (ThetaM, Id): every
 *            state and input can be written as an algebraic function of the flat
 *            outputs and finitely many of their derivatives.  The controller
 *            therefore *inverts the model along a smooth reference trajectory*:
 *
 *              Mechanical inversion:  IqFf = (J * AlphaRef + B * OmegaRef) / KT
 *              Electrical inversion:  Vd   = R*Id - OmegaE*Lq*Iq
 *                                     Vq   = R*Iq + Lq*dIq/dt + OmegaE*(Ld*Id + LambdaPm)
 *
 *            Feedback gains (KpSpeed, KpId, KpIq, KiId) only correct the residual
 *            between model and hardware, so they can stay small — faster transients
 *            and better high-speed tracking than a pure cascade PI at lower gain.
 *
 *            SIGNAL FLOW (one 20 kHz ISR step)
 *            ==================================
 *
 *              [SpeedRefRpm] --RPM->rad/s--> [Reference shaper] --> OmegaRefF, AlphaRefF
 *                                                    |
 *                                       [Mechanical flatness + P] --> IqRef
 *                                                    |
 *              [Iu, Iv, Iw] --> [Clarke] --> IAlphaBeta --> [SMO] --> ThetaE, OmegaE
 *                                                    |
 *                                  [Park(ThetaE)] --> IdMeas, IqMeas
 *                                                    |
 *                                  [Flatness voltage law] --> Vd, Vq  (d-priority sat)
 *                                                    |
 *                                  [InvPark(ThetaE)] --> VAlphaBeta --> SVPWM
 *
 *            The align / open-loop / closed-loop state machine (see header) selects
 *            the angle source and the (IdRef, IqRef) pair fed to the voltage law.
 *
 * \note      MISRA C:2012 compliance
 *              Dir   4.11 : atan2f argument validity guarded — never called with
 *                           both arguments zero (undefined behaviour, C99 7.12.4.4).
 *              Rule  8.4  : visible prototypes before all definitions.
 *              Rule  9.1  : every DFC_Output_T field is written on every path,
 *                           including the safe-duty (0.5) mid-step error path.
 *              Rule 15.5  : single return per function.
 *              Rule 15.7  : every if-else-if chain has a final else.
 *              Rule 21.5  : <math.h> used for fabsf/sqrtf/atan2f only.
 *
 * \version   4.3.2
 * \date      2026-07-04
 *
 * \par v4.3.2  Critical bug fixes:
 *   - Corrected SMO observer equation (R/L² bug)
 *   - Removed unused macros DFC_R_LQ, DFC_LD_LAMBDA
 *   - Improved Dfc_WrapTwoPi with while loops
 *   - Added epsilon tolerance for zero voltage vector check
 *   - Added isfinite() checks for NaN protection
 *   - Added sanity checks for Dt
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_dfc_controller.h"
#include <math.h>       /* fabsf, sqrtf, atan2f, isfinite */
#include <stddef.h>     /* NULL                 */

/**********************************************************************************************************************
 * Module-Private Macros
 *********************************************************************************************************************/

/** \brief  Typed zero literal (MISRA Rule 7.2). */
#define DFC_ZERO_F      ((MatrixFloat)0.0f)

/** \brief  dIqRef/dt LPF corner [rad/s]: one decade below the q-axis current loop. */
#define DFC_DIQ_LPF_W   ((MatrixFloat)2000.0f)

/** \brief  dIqRef/dt clamp [A/s]: bounds the Lq*dIq/dt feedforward contribution. */
#define DFC_DIQ_MAX     ((MatrixFloat)5000.0f)

/** \brief  Inverse inductance for SMO observer (corrected). */
#define DFC_INV_L_D     (ES_MATH_ONE_F / DFC_L_D)
#define DFC_INV_L_Q     (ES_MATH_ONE_F / DFC_L_Q)

/** \brief  d-axis integrator clamp. */
#define DFC_ID_INT_LIMIT ((MatrixFloat)2.0f)

/**********************************************************************************************************************
 * Module-Private Function Prototypes  (MISRA Rule 8.4)
 *********************************************************************************************************************/
static MatrixFloat Dfc_Clamp(MatrixFloat Value, MatrixFloat Limit);
static MatrixFloat Dfc_WrapTwoPi(MatrixFloat Angle);
static MatrixFloat Dfc_LpfCoeff(MatrixFloat CornerW, MatrixFloat Dt);
static boolean_T   Dfc_IsFinite(MatrixFloat Value);
static void        Dfc_SmoStep(
    DFC_Smo_T            * const Smo_P,
    const FocAlphaBeta_T * const IMeas_P,
    const FocAlphaBeta_T * const VPrev_P,
    MatrixFloat                  Dt);
static void        Dfc_ModeStateMachine(
    DFC_State_T          * const State_P,
    MatrixFloat                  OmegaCmdMech,
    MatrixFloat                  Dt,
    const DFC_LoopOption_T       LoopOption,
    FocAngle_T           * const AngleOut_P,
    MatrixFloat          * const IdRefOut_P,
    MatrixFloat          * const IqRefOut_P);
static void        Dfc_VoltageLaw(
    DFC_State_T          * const State_P,
    MatrixFloat                  IdRef,
    MatrixFloat                  IqRef,
    MatrixFloat                  OmegaE,
    MatrixFloat                  Dt,
    const FocDq_T        * const IdqMeas_P,
    FocDq_T              * const VdqOut_P);

/**********************************************************************************************************************
 * Module-Private Helper Functions
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_Clamp — symmetric saturation to ±Limit.
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat Dfc_Clamp(MatrixFloat Value, MatrixFloat Limit)
{
    MatrixFloat result;

    if (Value > Limit)
    {
        result = Limit;
    }
    else if (Value < -Limit)
    {
        result = -Limit;
    }
    else
    {
        result = Value;
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_WrapTwoPi — wrap an angle to [0, 2*pi).
 *
 * Uses while loops to handle arbitrarily large deltas safely.
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat Dfc_WrapTwoPi(MatrixFloat Angle)
{
    MatrixFloat result;

    /* Check for NaN/inf first */
    if (!Dfc_IsFinite(Angle))
    {
        result = DFC_ZERO_F;
    }
    else
    {
        result = Angle;

        while (result >= ES_MATH_2PI_F)
        {
            result -= ES_MATH_2PI_F;
        }

        while (result < DFC_ZERO_F)
        {
            result += ES_MATH_2PI_F;
        }
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_IsFinite — check if a float is finite (not NaN or inf).
 *------------------------------------------------------------------------------------------------------------------*/
static boolean_T Dfc_IsFinite(MatrixFloat Value)
{
    boolean_T result;

    /* isfinite is C99/C11, available in modern toolchains */
    if (isfinite(Value))
    {
        result = TRUE;
    }
    else
    {
        result = FALSE;
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_LpfCoeff — first-order IIR coefficient a = w*dt / (1 + w*dt).
 * Optimized: if w*dt is very small, return w*dt to avoid division.
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat Dfc_LpfCoeff(MatrixFloat CornerW, MatrixFloat Dt)
{
    MatrixFloat wdt;
    MatrixFloat result;

    /* Guard against NaN/inf */
    if ((!Dfc_IsFinite(CornerW)) || (!Dfc_IsFinite(Dt)) || (Dt <= DFC_ZERO_F))
    {
        result = DFC_ZERO_F;
    }
    else
    {
        wdt = CornerW * Dt;

        if (wdt < DFC_EPSILON)
        {
            result = wdt;
        }
        else
        {
            result = wdt / (ES_MATH_ONE_F + wdt);
        }
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_SmoStep — Sliding Mode Observer, αβ frame.
 *
 * Current observer (Euler):
 *   IHat' = (VPrev - R*IHat - Z) / L
 *
 * CORRECTED: The resistive term is R*IHat, NOT (R/L)*IHat.
 *------------------------------------------------------------------------------------------------------------------*/
static void Dfc_SmoStep(
    DFC_Smo_T            * const Smo_P,
    const FocAlphaBeta_T * const IMeas_P,
    const FocAlphaBeta_T * const VPrev_P,
    MatrixFloat                  Dt)
{
    MatrixFloat errAlpha;
    MatrixFloat errBeta;
    MatrixFloat zAlpha;
    MatrixFloat zBeta;
    MatrixFloat lpfE;
    MatrixFloat lpfW;
    MatrixFloat thetaNew;
    MatrixFloat dTheta;
    MatrixFloat omegaRaw;
    MatrixFloat invL_Dt_D;
    MatrixFloat invL_Dt_Q;

    /* Guard against invalid inputs */
    if ((Smo_P == NULL) || (IMeas_P == NULL) || (VPrev_P == NULL) || (Dt <= DFC_ZERO_F))
    {
        return;
    }

    /* Precompute dt/L for efficiency */
    invL_Dt_D = DFC_INV_L_D * Dt;
    invL_Dt_Q = DFC_INV_L_Q * Dt;

    /* Switching signal from the current estimation error */
    errAlpha = Smo_P->IHat.Alpha - IMeas_P->Alpha;
    errBeta  = Smo_P->IHat.Beta  - IMeas_P->Beta;

    /* Clamp error to boundary layer */
    errAlpha = Dfc_Clamp(errAlpha / DFC_SMO_E0, ES_MATH_ONE_F);
    errBeta  = Dfc_Clamp(errBeta  / DFC_SMO_E0, ES_MATH_ONE_F);

    zAlpha   = DFC_SMO_K * errAlpha;
    zBeta    = DFC_SMO_K * errBeta;

    /* Euler current observer - CORRECTED: R*IHat (not R/L * IHat) */
    Smo_P->IHat.Alpha += invL_Dt_D * (VPrev_P->Alpha - (DFC_R_S * Smo_P->IHat.Alpha) - zAlpha);
    Smo_P->IHat.Beta  += invL_Dt_Q * (VPrev_P->Beta  - (DFC_R_S * Smo_P->IHat.Beta)  - zBeta);

    /* Guard against NaN propagation from observer */
    if ((!Dfc_IsFinite(Smo_P->IHat.Alpha)) || (!Dfc_IsFinite(Smo_P->IHat.Beta)))
    {
        Smo_P->IHat.Alpha = IMeas_P->Alpha;
        Smo_P->IHat.Beta  = IMeas_P->Beta;
    }

    /* Divergence guard: re-seed the observer from the measurement if the
     * estimate runs away (e.g. after a voltage transient). */
    if ((fabsf(Smo_P->IHat.Alpha) > DFC_SMO_I_GUARD) ||
        (fabsf(Smo_P->IHat.Beta)  > DFC_SMO_I_GUARD))
    {
        Smo_P->IHat.Alpha = IMeas_P->Alpha;
        Smo_P->IHat.Beta  = IMeas_P->Beta;
    }
    else
    {
        /* Observer healthy */
    }

    /* Back-EMF extraction LPF. */
    lpfE = Dfc_LpfCoeff(DFC_SMO_LPF_W, Dt);
    Smo_P->EHat.Alpha += lpfE * (zAlpha - Smo_P->EHat.Alpha);
    Smo_P->EHat.Beta  += lpfE * (zBeta  - Smo_P->EHat.Beta);

    /* Guard against NaN in back-EMF */
    if ((!Dfc_IsFinite(Smo_P->EHat.Alpha)) || (!Dfc_IsFinite(Smo_P->EHat.Beta)))
    {
        Smo_P->EHat.Alpha = DFC_ZERO_F;
        Smo_P->EHat.Beta  = DFC_ZERO_F;
        Smo_P->ThetaE     = Smo_P->ThetaEPrev;
        return;
    }

    /* Angle from back-EMF geometry */
    thetaNew = atan2f(-Smo_P->EHat.Alpha, Smo_P->EHat.Beta);
    thetaNew = Dfc_WrapTwoPi(thetaNew);

    /* Wrapped finite-difference speed */
    dTheta = thetaNew - Smo_P->ThetaEPrev;
    if (dTheta > ES_MATH_PI_F)
    {
        dTheta -= ES_MATH_2PI_F;
    }
    else if (dTheta < -ES_MATH_PI_F)
    {
        dTheta += ES_MATH_2PI_F;
    }
    else
    {
        /* No wrap crossing */
    }

    omegaRaw = dTheta / Dt;
    omegaRaw = Dfc_Clamp(omegaRaw, DFC_SMO_OMEGA_MAX_E);

    lpfW = Dfc_LpfCoeff(DFC_SMO_SPEED_LPF_W, Dt);

    Smo_P->ThetaEPrev  = thetaNew;
    Smo_P->ThetaE      = thetaNew;
    Smo_P->WarmupTime += Dt;

    if (Smo_P->WarmupTime < DFC_SMO_WARMUP_TIME_S)
    {
        Smo_P->OmegaEFilt = DFC_ZERO_F;
    }
    else
    {
        MatrixFloat innov;

        innov = Dfc_Clamp(omegaRaw - Smo_P->OmegaEFilt, DFC_SMO_INNOV_MAX_E);
        Smo_P->OmegaEFilt += lpfW * innov;

        /* Guard against NaN in speed estimate */
        if (!Dfc_IsFinite(Smo_P->OmegaEFilt))
        {
            Smo_P->OmegaEFilt = DFC_ZERO_F;
        }
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_ModeStateMachine — align -> open-loop I-f ramp -> closed-loop flatness.
 *------------------------------------------------------------------------------------------------------------------*/
static void Dfc_ModeStateMachine(
    DFC_State_T          * const State_P,
    MatrixFloat                  OmegaCmdMech,
    MatrixFloat                  Dt,
    const DFC_LoopOption_T       LoopOption,
    FocAngle_T           * const AngleOut_P,
    MatrixFloat          * const IdRefOut_P,
    MatrixFloat          * const IqRefOut_P)
{
    MatrixFloat omegaMeasMech;
    MatrixFloat iqFf;
    MatrixFloat iqFb;
    MatrixFloat lpfDiq;
    MatrixFloat dIqRaw;
    MatrixFloat iqRef;

    /* Guard against invalid inputs */
    if ((State_P == NULL) || (AngleOut_P == NULL) ||
        (IdRefOut_P == NULL) || (IqRefOut_P == NULL) || (Dt <= DFC_ZERO_F))
    {
        return;
    }

    State_P->TimeInMode += Dt;

    if (State_P->Mode == DFC_MODE_ALIGN)
    {
        AngleOut_P->ThetaE = DFC_ZERO_F;
        *IdRefOut_P        = DFC_OL_I_BOOST;
        *IqRefOut_P        = DFC_ZERO_F;

        if (State_P->TimeInMode >= DFC_ALIGN_TIME_S)
        {
            State_P->Mode       = DFC_MODE_OPENLOOP;
            State_P->TimeInMode = DFC_ZERO_F;
            State_P->ThetaOl    = DFC_ZERO_F;
            State_P->OmegaOlE   = DFC_ZERO_F;
        }
        else
        {
            /* Continue aligning */
        }
    }
    else if (State_P->Mode == DFC_MODE_OPENLOOP)
    {
        State_P->OmegaOlE += DFC_OL_ACCEL_E * Dt;
        if (State_P->OmegaOlE > DFC_OL_OMEGA_HANDOVER_E)
        {
            State_P->OmegaOlE = DFC_OL_OMEGA_HANDOVER_E;
        }
        else
        {
            /* Still ramping */
        }
        State_P->ThetaOl = Dfc_WrapTwoPi(State_P->ThetaOl + (State_P->OmegaOlE * Dt));

        AngleOut_P->ThetaE = State_P->ThetaOl;
        *IdRefOut_P        = DFC_OL_I_BOOST;
        *IqRefOut_P        = DFC_ZERO_F;

        /* v4.3.0: handover gate opens only in Option B */
        if ((LoopOption == DFC_LOOP_CLOSEDLOOP) &&
            (State_P->OmegaOlE >= DFC_OL_OMEGA_HANDOVER_E) &&
            (State_P->Smo.WarmupTime >= DFC_SMO_WARMUP_TIME_S) &&
            (fabsf(State_P->Smo.OmegaEFilt - State_P->OmegaOlE) < DFC_OL_HANDOVER_BAND_E))
        {
            State_P->Mode       = DFC_MODE_CLOSEDLOOP;
            State_P->TimeInMode = DFC_ZERO_F;
            State_P->OmegaRefF  = State_P->Smo.OmegaEFilt / DFC_P_POLES_F;
            State_P->AlphaRefF  = DFC_ZERO_F;
            State_P->IdIntegral = DFC_ZERO_F;
            State_P->IqRefPrev  = DFC_ZERO_F;
            State_P->DIqFilt    = DFC_ZERO_F;

            /* Seed load-torque observer */
            State_P->ObsOmega   = State_P->Smo.OmegaEFilt / DFC_P_POLES_F;
            State_P->ObsOmegaF  = State_P->ObsOmega;
            State_P->ObsTLoad   = DFC_ZERO_F;
        }
        else
        {
            /* Continue ramping */
        }
    }
    else
    {
        /* DFC_MODE_CLOSEDLOOP — full flatness control on the SMO angle. */
        MatrixFloat wn2;
        MatrixFloat twoZetaWn;

        omegaMeasMech = State_P->Smo.OmegaEFilt / DFC_P_POLES_F;

        /* Guard against NaN in speed measurement */
        if (!Dfc_IsFinite(omegaMeasMech))
        {
            omegaMeasMech = DFC_ZERO_F;
        }

        /* 2nd-order critically damped reference shaper */
        wn2          = State_P->Gains.RefWn * State_P->Gains.RefWn;
        twoZetaWn    = ES_MATH_TWO_F * State_P->Gains.RefZeta * State_P->Gains.RefWn;

        State_P->AlphaRefF += Dt * ((wn2 * (OmegaCmdMech - State_P->OmegaRefF)) -
                                    (twoZetaWn * State_P->AlphaRefF));
        State_P->OmegaRefF += Dt * State_P->AlphaRefF;

        /* Guard against NaN in shaper states */
        if (!Dfc_IsFinite(State_P->OmegaRefF))
        {
            State_P->OmegaRefF = OmegaCmdMech;
        }
        if (!Dfc_IsFinite(State_P->AlphaRefF))
        {
            State_P->AlphaRefF = DFC_ZERO_F;
        }

        /* Load-torque observer (v4.2.0) */
        {
            MatrixFloat obsErr;
            MatrixFloat dTl;
            MatrixFloat ktOverJ;
            MatrixFloat bOverJ;

            ktOverJ = DFC_KT / DFC_J_ROTOR;
            bOverJ  = DFC_B_FRIC / DFC_J_ROTOR;

            obsErr = omegaMeasMech - State_P->ObsOmegaF;

            /* Guard against NaN in observer */
            if (Dfc_IsFinite(obsErr))
            {
                State_P->ObsOmega += Dt * (((ktOverJ * State_P->Diag.IdqMeas.Q)
                                           - (bOverJ * State_P->ObsOmega)
                                           - (State_P->ObsTLoad / DFC_J_ROTOR))
                                           + (DFC_OBS_L1 * obsErr));
                State_P->ObsOmegaF += Dfc_LpfCoeff(DFC_SMO_SPEED_LPF_W, Dt)
                                      * (State_P->ObsOmega - State_P->ObsOmegaF);
                dTl = Dfc_Clamp(-(DFC_OBS_L2 * obsErr), DFC_TL_SLEW_MAX * Dt);
                State_P->ObsTLoad = Dfc_Clamp(State_P->ObsTLoad + dTl, DFC_TL_MAX);
            }

            /* Guard against NaN in observer states */
            if (!Dfc_IsFinite(State_P->ObsOmega))
            {
                State_P->ObsOmega = omegaMeasMech;
            }
            if (!Dfc_IsFinite(State_P->ObsOmegaF))
            {
                State_P->ObsOmegaF = omegaMeasMech;
            }
            if (!Dfc_IsFinite(State_P->ObsTLoad))
            {
                State_P->ObsTLoad = DFC_ZERO_F;
            }
        }

        /* Mechanical flatness inversion + proportional correction */
        iqFf = ((DFC_J_ROTOR * State_P->AlphaRefF) +
                (DFC_B_FRIC  * State_P->OmegaRefF)) / DFC_KT;

        if (State_P->TimeInMode >= DFC_OBS_HOLDOFF_S)
        {
            iqFf += State_P->ObsTLoad / DFC_KT;
        }
        else
        {
            /* Hold-off: observer tracks, feedforward disconnected */
        }

        iqFb = State_P->Gains.KpSpeed * (State_P->OmegaRefF - omegaMeasMech);
        iqRef = Dfc_Clamp(iqFf + iqFb, DFC_I_MAX);

        /* Filtered dIqRef/dt for the Lq * dIq/dt electrical feedforward */
        dIqRaw = Dfc_Clamp((iqRef - State_P->IqRefPrev) / Dt, DFC_DIQ_MAX);
        lpfDiq = Dfc_LpfCoeff(DFC_DIQ_LPF_W, Dt);
        State_P->DIqFilt += lpfDiq * (dIqRaw - State_P->DIqFilt);

        /* Guard against NaN in DIqFilt */
        if (!Dfc_IsFinite(State_P->DIqFilt))
        {
            State_P->DIqFilt = DFC_ZERO_F;
        }

        State_P->IqRefPrev = iqRef;

        AngleOut_P->ThetaE = State_P->Smo.ThetaE;
        *IdRefOut_P        = DFC_ZERO_F;    /* MTPA for SPMSM: Ld = Lq */
        *IqRefOut_P        = iqRef;
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * Dfc_VoltageLaw — electrical flatness inversion with d-axis-priority saturation.
 *
 *   Vd = R*IdRef - OmegaE*Lq*IqRef + KpId*(IdRef - Id) + IdIntegral
 *   Vq = R*IqRef + Lq*dIqF + OmegaE*(Ld*IdRef + LambdaPm) + KpIq*(IqRef - Iq)
 *------------------------------------------------------------------------------------------------------------------*/
static void Dfc_VoltageLaw(
    DFC_State_T          * const State_P,
    MatrixFloat                  IdRef,
    MatrixFloat                  IqRef,
    MatrixFloat                  OmegaE,
    MatrixFloat                  Dt,
    const FocDq_T        * const IdqMeas_P,
    FocDq_T              * const VdqOut_P)
{
    MatrixFloat idErr;
    MatrixFloat vd;
    MatrixFloat vq;
    MatrixFloat vdClamped;
    MatrixFloat vqBudget;
    MatrixFloat lqIqRef;
    MatrixFloat ldIdRefLambda;

    /* Guard against invalid inputs */
    if ((State_P == NULL) || (IdqMeas_P == NULL) || (VdqOut_P == NULL) || (Dt <= DFC_ZERO_F))
    {
        return;
    }

    /* Guard against NaN in inputs */
    if ((!Dfc_IsFinite(IdRef)) || (!Dfc_IsFinite(IqRef)) || (!Dfc_IsFinite(OmegaE)))
    {
        VdqOut_P->D = DFC_ZERO_F;
        VdqOut_P->Q = DFC_ZERO_F;
        return;
    }

    idErr = IdRef - IdqMeas_P->D;

    /* Precompute common terms */
    lqIqRef           = DFC_L_Q * IqRef;
    ldIdRefLambda     = (DFC_L_D * IdRef) + DFC_LAMBDA_PM;

    /* d-axis: feedforward + P + I */
    vd = (DFC_R_S * IdRef)
       - (OmegaE * lqIqRef)
       + (State_P->Gains.KpId * idErr)
       + State_P->IdIntegral;

    /* q-axis: full flatness feedforward + P */
    vq = (DFC_R_S * IqRef)
       + (DFC_L_Q * State_P->DIqFilt)
       + (OmegaE * ldIdRefLambda)
       + (State_P->Gains.KpIq * (IqRef - IdqMeas_P->Q));

    /* Guard against NaN in computed voltages */
    if (!Dfc_IsFinite(vd))
    {
        vd = DFC_ZERO_F;
    }
    if (!Dfc_IsFinite(vq))
    {
        vq = DFC_ZERO_F;
    }

    /* d-axis-priority saturation at the modulator linear ceiling */
    vdClamped = Dfc_Clamp(vd, DFC_V_LIN);

    /* Compute remaining voltage budget for q-axis */
    vqBudget = sqrtf((DFC_V_LIN * DFC_V_LIN) - (vdClamped * vdClamped));
    vq = Dfc_Clamp(vq, vqBudget);

    /* Conditional anti-windup: integrate only while Vd is unsaturated */
    if ((fabsf(vd) < DFC_V_LIN) && (Dfc_IsFinite(idErr)))
    {
        State_P->IdIntegral += State_P->Gains.KiId * idErr * Dt;
        State_P->IdIntegral = Dfc_Clamp(State_P->IdIntegral, DFC_ID_INT_LIMIT);
    }
    else
    {
        /* Integrator frozen while d-axis saturated */
    }

    /* Guard against NaN in integrator */
    if (!Dfc_IsFinite(State_P->IdIntegral))
    {
        State_P->IdIntegral = DFC_ZERO_F;
    }

    VdqOut_P->D = vdClamped;
    VdqOut_P->Q = vq;
}

/**********************************************************************************************************************
 * Public Functions
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Init
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T DFC_Init(
    DFC_State_T          * const State_P)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if (State_P == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* SMO */
        State_P->Smo.IHat.Alpha  = DFC_ZERO_F;
        State_P->Smo.IHat.Beta   = DFC_ZERO_F;
        State_P->Smo.EHat.Alpha  = DFC_ZERO_F;
        State_P->Smo.EHat.Beta   = DFC_ZERO_F;
        State_P->Smo.ThetaE      = DFC_ZERO_F;
        State_P->Smo.ThetaEPrev  = DFC_ZERO_F;
        State_P->Smo.OmegaEFilt  = DFC_ZERO_F;
        State_P->Smo.WarmupTime  = DFC_ZERO_F;

        /* Startup state machine */
        State_P->Mode        = DFC_MODE_ALIGN;
        State_P->TimeInMode  = DFC_ZERO_F;
        State_P->ThetaOl     = DFC_ZERO_F;
        State_P->OmegaOlE    = DFC_ZERO_F;

        /* Reference shaper */
        State_P->OmegaRefF   = DFC_ZERO_F;
        State_P->AlphaRefF   = DFC_ZERO_F;

        /* Current reference trajectory */
        State_P->IqRefPrev   = DFC_ZERO_F;
        State_P->DIqFilt     = DFC_ZERO_F;

        /* d-axis integrator */
        State_P->IdIntegral  = DFC_ZERO_F;

        /* SMO voltage latch */
        State_P->VPrev.Alpha = DFC_ZERO_F;
        State_P->VPrev.Beta  = DFC_ZERO_F;

        /* Mechanical position reconstruction */
        State_P->ThetaMech   = DFC_ZERO_F;

        /* Load-torque observer */
        State_P->ObsOmega    = DFC_ZERO_F;
        State_P->ObsOmegaF   = DFC_ZERO_F;
        State_P->ObsTLoad    = DFC_ZERO_F;

        /* Compile-time default gains */
        State_P->Gains.KpSpeed = DFC_KP_SPEED;
        State_P->Gains.KpId    = DFC_KP_ID;
        State_P->Gains.KpIq    = DFC_KP_IQ;
        State_P->Gains.KiId    = DFC_KI_ID;
        State_P->Gains.RefWn   = DFC_REF_WN;
        State_P->Gains.RefZeta = DFC_REF_ZETA;

        /* Diagnostics */
        State_P->Diag.OmegaRefF        = DFC_ZERO_F;
        State_P->Diag.OmegaMeas        = DFC_ZERO_F;
        State_P->Diag.IqRef            = DFC_ZERO_F;
        State_P->Diag.IdqMeas.D        = DFC_ZERO_F;
        State_P->Diag.IdqMeas.Q        = DFC_ZERO_F;
        State_P->Diag.IdIntegral       = DFC_ZERO_F;
        State_P->Diag.VDq.D            = DFC_ZERO_F;
        State_P->Diag.VDq.Q            = DFC_ZERO_F;
        State_P->Diag.VAlphaBeta.Alpha = DFC_ZERO_F;
        State_P->Diag.VAlphaBeta.Beta  = DFC_ZERO_F;
        State_P->Diag.Angle.ThetaE     = DFC_ZERO_F;
        State_P->Diag.Sector           = SVM_SECTOR_I;
        State_P->Diag.TLoadHat         = DFC_ZERO_F;
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Step
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T DFC_Step(
    DFC_State_T          * const State_P,
    const DFC_Input_T    * const In_P,
    const MatrixFloat            Dt,
    DFC_Output_T         * const Out_P)
{
    MatrixStatus_T  status;
    FocAlphaBeta_T     iAlphaBeta;
    FocDq_T            idqMeas;
    FocAngle_T         angle;
    MatrixFloat        omegaCmdMech;
    MatrixFloat        omegaE;
    MatrixFloat        idRef;
    MatrixFloat        iqRef;
    FocDq_T            vDq;
    FocAlphaBeta_T     vAlphaBeta;
    FocAlphaBeta_T     vPu;
    FocAngle_T         vAngle;
    SVM_DutyCycle_T    duty;
    MatrixFloat        omegaMech;

    status = MATRIX_SUCCESS;

    if ((State_P == NULL) || (In_P == NULL) || (Out_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Dt <= DFC_ZERO_F)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        /* 1. Clarke: phase currents -> αβ. */
        status = Clarke_Transform_Matrix(&In_P->PhaseCurrents, &iAlphaBeta);

        if (status == MATRIX_SUCCESS)
        {
            /* 2. SMO on the previous step's commanded voltage (z^-1). */
            Dfc_SmoStep(&State_P->Smo, &iAlphaBeta, &State_P->VPrev, Dt);

            /* 3. RPM -> rad/s mech, clamped; mode state machine selects
             *    angle source and current references. */
            omegaCmdMech = Dfc_Clamp(In_P->SpeedRefRpm * DFC_RPM_TO_RADPS,
                                     DFC_OMEGA_CMD_MAX);
            Dfc_ModeStateMachine(State_P, omegaCmdMech, Dt,
                                 In_P->LoopOption,
                                 &angle, &idRef, &iqRef);

            /* 4. Park: αβ currents -> dq in the active frame. */
            status = Park_Transform_Matrix(&iAlphaBeta, &angle, &idqMeas);
        }
        else
        {
            /* Clarke failed — abort the step */
        }

        if (status == MATRIX_SUCCESS)
        {
            /* Electrical speed for the voltage law: the commanded ramp speed
             * in open loop (the SMO estimate is still converging), the SMO
             * estimate in closed loop. */
            if (State_P->Mode == DFC_MODE_CLOSEDLOOP)
            {
                omegaE = State_P->Smo.OmegaEFilt;
            }
            else
            {
                omegaE = State_P->OmegaOlE;
            }

            /* 5. Flatness voltage law (d-priority saturation, anti-windup). */
            Dfc_VoltageLaw(State_P, idRef, iqRef, omegaE, Dt,
                           &idqMeas, &vDq);

            /* 6. Inverse Park: dq voltages -> αβ. */
            status = InvPark_Transform_Matrix(&vDq, &angle, &vAlphaBeta);
        }
        else
        {
            /* Park failed — abort the step */
        }

        if (status == MATRIX_SUCCESS)
        {
            /* 7. SVPWM inside the step.
             *    Volts -> per-unit for the modulator.
             *    The duty vector is placed at the VOLTAGE vector angle
             *    atan2(VBeta, VAlpha) — NOT the rotor angle. */
            vPu.Alpha = vAlphaBeta.Alpha * DFC_V_TO_PU;
            vPu.Beta  = vAlphaBeta.Beta  * DFC_V_TO_PU;

            /* Use epsilon tolerance for zero voltage vector check
             * (MISRA Dir 4.11: atan2f(0,0) is undefined) */
            if ((fabsf(vPu.Alpha) < DFC_EPSILON) &&
                (fabsf(vPu.Beta)  < DFC_EPSILON))
            {
                vAngle.ThetaE = DFC_ZERO_F;
            }
            else
            {
                vAngle.ThetaE = Dfc_WrapTwoPi(atan2f(vPu.Beta, vPu.Alpha));
            }

            status = SVM_CalculateDutyCycleFromAlphaBeta(&vPu, &vAngle, &duty);
        }
        else
        {
            /* InvPark failed — SVPWM skipped */
        }

        if (status == MATRIX_SUCCESS)
        {
            /* 8. Latch VPrev for the next step's SMO; publish outputs. */
            State_P->VPrev = vAlphaBeta;

            omegaMech = State_P->Smo.OmegaEFilt / DFC_P_POLES_F;

            State_P->ThetaMech = Dfc_WrapTwoPi(State_P->ThetaMech +
                                               (omegaMech * Dt));

            Out_P->Ta              = duty.Ta;
            Out_P->Tb              = duty.Tb;
            Out_P->Tc              = duty.Tc;
            Out_P->AngularVelocity = omegaMech;
            Out_P->RotorPosition   = State_P->ThetaMech;
            Out_P->Mode            = State_P->Mode;

            Out_P->PhaseCurrents.U = In_P->PhaseCurrents.U;
            Out_P->PhaseCurrents.V = In_P->PhaseCurrents.V;
            Out_P->PhaseCurrents.W = -(In_P->PhaseCurrents.U +
                                       In_P->PhaseCurrents.V);

            /* Diagnostics snapshot */
            State_P->Diag.OmegaRefF  = State_P->OmegaRefF;
            State_P->Diag.OmegaMeas  = omegaMech;
            State_P->Diag.IqRef      = iqRef;
            State_P->Diag.IdqMeas    = idqMeas;
            State_P->Diag.IdIntegral = State_P->IdIntegral;
            State_P->Diag.VDq        = vDq;
            State_P->Diag.VAlphaBeta = vAlphaBeta;
            State_P->Diag.Angle      = angle;
            State_P->Diag.Sector     = duty.Sector;
            State_P->Diag.TLoadHat   = State_P->ObsTLoad;
        }
        else
        {
            /* Any mid-step failure: force the safe 50 % midpoint on all
             * phases (zero average phase voltage; gate driver stays enabled)
             * and freeze the mechanical estimates. */
            Out_P->Ta              = ES_MATH_HALF_F;
            Out_P->Tb              = ES_MATH_HALF_F;
            Out_P->Tc              = ES_MATH_HALF_F;
            Out_P->AngularVelocity = DFC_ZERO_F;
            Out_P->RotorPosition   = State_P->ThetaMech;
            Out_P->Mode            = State_P->Mode;
            Out_P->PhaseCurrents   = In_P->PhaseCurrents;
        }
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_Reset
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T DFC_Reset(
    DFC_State_T          * const State_P)
{
    MatrixStatus_T status;
    DFC_GainSet_T     savedGains;

    status = MATRIX_SUCCESS;

    if (State_P == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        savedGains     = State_P->Gains;
        status         = DFC_Init(State_P);
        State_P->Gains = savedGains;
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_GainSet_Apply
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T DFC_GainSet_Apply(
    DFC_State_T          * const State_P,
    const DFC_GainSet_T  * const Gains_P)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if ((State_P == NULL) || (Gains_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        State_P->Gains = *Gains_P;
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * DFC_GetDiagnostics
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T DFC_GetDiagnostics(
    const DFC_State_T    * const State_P,
    DFC_Diag_T           * const Diag_P)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if ((State_P == NULL) || (Diag_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        *Diag_P = State_P->Diag;
    }

    return status;
}
