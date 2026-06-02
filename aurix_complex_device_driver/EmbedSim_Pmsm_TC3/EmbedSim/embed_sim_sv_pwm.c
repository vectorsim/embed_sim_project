/**********************************************************************************************************************
 * \file      embed_sim_sv_pwm.c
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation implementation.
 *            Matrix-based implementation using coordinate transforms.
 *
 * \details   Implements symmetric centre-aligned SVPWM (T0 = T7).
 *            Uses matrix transforms from embed_sim_coordinate_transform.h
 *            for coordinate conversions.
 *
 * \version   2.1.0
 * \date      2025-05-24
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_sv_pwm.h"
#include <math.h>
#include <stddef.h>

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  0.0   [dimensionless] */
#define SVM_ZERO_F   ((MatrixFloat)0.0f)

/** \brief  Maximum modulation index for linear range = √3/2   [dimensionless] */
#define SVM_MAX_LINEAR_MOD   ES_MATH_HALF_SQRT3_F

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Determine the SVPWM sector from a wrapped electrical angle.
 *
 * \param[in]  AngleRad  Electrical angle   [rad, any value]
 * \return               Resolved SVM_Sector_T (I – VI)
 */
static SVM_Sector_T SVM_GetSectorFromAngle(MatrixFloat AngleRad);

/**
 * \brief   Calculate normalised active-vector times T1 and T2.
 *
 * \param[in]  ActiveSector  Active sector (I – VI)
 * \param[in]  AngleRad      Electrical angle        [rad]
 * \param[in]  ModIndex      Modulation index        [0.0 – 1.0]
 * \param[out] T1Out_P       Normalised time T1      (must not be NULL)
 * \param[out] T2Out_P       Normalised time T2      (must not be NULL)
 * \return  void
 */
static void SVM_CalculateTimes(
    SVM_Sector_T     ActiveSector,
    MatrixFloat      AngleRad,
    MatrixFloat      ModIndex,
    MatrixFloat    * const T1Out_P,
    MatrixFloat    * const T2Out_P);

/**
 * \brief   Clamp a float value to [0.0, 1.0].
 *
 * \param[in]  Value  Input value
 * \return            Clamped value   [0.0 – 1.0]
 */
static MatrixFloat SVM_ClampFloat(MatrixFloat Value);

/**
 * \brief   Calculate duty cycles from active vector times and sector.
 *
 * \param[in]  T1          Active vector time T1
 * \param[in]  T2          Active vector time T2
 * \param[in]  Sector      Active sector
 * \param[out] DutyOut_P   Output duty cycles
 * \return  void
 */
static void SVM_CalculateDutyFromTimes(
    MatrixFloat           T1,
    MatrixFloat           T2,
    SVM_Sector_T          Sector,
    SVM_DutyCycle_T * const DutyOut_P);

/**
 * \brief   Convert αβ voltage to modulation index.
 *
 * \param[in]  V_AlphaBeta  αβ voltage vector
 * \return                  Modulation index [0.0 – 1.0]
 */
static MatrixFloat SVM_AlphaBetaToModIndex(const FocAlphaBeta_T * const V_AlphaBeta_P);

/**********************************************************************************************************************
 * Private Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_ClampFloat
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat SVM_ClampFloat(MatrixFloat Value)
{
    MatrixFloat result;

    result = Value;

    if (result < SVM_ZERO_F)
    {
        result = SVM_ZERO_F;
    }
    else if (result > ES_MATH_ONE_F)
    {
        result = ES_MATH_ONE_F;
    }
    else
    {
        /* Value already within [0, 1] — no action required. */
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetSectorFromAngle
 *------------------------------------------------------------------------------------------------------------------*/
static SVM_Sector_T SVM_GetSectorFromAngle(MatrixFloat AngleRad)
{
    MatrixFloat  angle_norm;
    SVM_Sector_T sector;

    /* Wrap angle into [0, 2π). */
    angle_norm = AngleRad;
    while (angle_norm < SVM_ZERO_F)
    {
        angle_norm += SVM_2PI_F;
    }
    while (angle_norm >= SVM_2PI_F)
    {
        angle_norm -= SVM_2PI_F;
    }

    /* Map 60°-wide bands to sectors. */
    if (angle_norm < SVM_PI_OVER_3_F)
    {
        sector = SVM_SECTOR_I;
    }
    else if (angle_norm < SVM_2PI_OVER_3_F)
    {
        sector = SVM_SECTOR_II;
    }
    else if (angle_norm < SVM_PI_F)
    {
        sector = SVM_SECTOR_III;
    }
    else if (angle_norm < SVM_4PI_OVER_3_F)
    {
        sector = SVM_SECTOR_IV;
    }
    else if (angle_norm < SVM_5PI_OVER_3_F)
    {
        sector = SVM_SECTOR_V;
    }
    else
    {
        sector = SVM_SECTOR_VI;
    }

    return sector;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_CalculateTimes
 *------------------------------------------------------------------------------------------------------------------*/
static void SVM_CalculateTimes(
    SVM_Sector_T     ActiveSector,
    MatrixFloat      AngleRad,
    MatrixFloat      ModIndex,
    MatrixFloat    * const T1Out_P,
    MatrixFloat    * const T2Out_P)
{
    MatrixFloat cos_t1;
    MatrixFloat cos_t2;
    MatrixFloat scale;
    MatrixFloat sum;

    scale  = SVM_SQRT3_OVER_2_F * ModIndex;
    cos_t1 = SVM_ZERO_F;
    cos_t2 = SVM_ZERO_F;

    switch (ActiveSector)
    {
        case SVM_SECTOR_I:
            cos_t1 = cosf(AngleRad + SVM_PI_OVER_6_F);
            cos_t2 = cosf(AngleRad - SVM_PI_OVER_2_F);
            break;

        case SVM_SECTOR_II:
            cos_t1 = cosf(AngleRad - SVM_PI_OVER_6_F);
            cos_t2 = cosf(AngleRad - (5.0f * SVM_PI_OVER_6_F));
            break;

        case SVM_SECTOR_III:
            cos_t1 = cosf(AngleRad - SVM_PI_OVER_2_F);
            cos_t2 = cosf(AngleRad - (7.0f * SVM_PI_OVER_6_F));
            break;

        case SVM_SECTOR_IV:
            cos_t1 = cosf(AngleRad - (5.0f * SVM_PI_OVER_6_F));
            cos_t2 = cosf(AngleRad - (3.0f * SVM_PI_OVER_2_F));
            break;

        case SVM_SECTOR_V:
            cos_t1 = cosf(AngleRad - (7.0f  * SVM_PI_OVER_6_F));
            cos_t2 = cosf(AngleRad - (11.0f * SVM_PI_OVER_6_F));
            break;

        case SVM_SECTOR_VI:
            cos_t1 = cosf(AngleRad - (3.0f * SVM_PI_OVER_2_F));
            cos_t2 = cosf(AngleRad - SVM_PI_OVER_6_F);
            break;

        default:
            cos_t1 = SVM_ZERO_F;
            cos_t2 = SVM_ZERO_F;
            break;
    }

    *T1Out_P = scale * cos_t1;
    *T2Out_P = scale * cos_t2;

    /* Clamp to non-negative. */
    if (*T1Out_P < SVM_ZERO_F) { *T1Out_P = SVM_ZERO_F; }
    else { /* no action */ }

    if (*T2Out_P < SVM_ZERO_F) { *T2Out_P = SVM_ZERO_F; }
    else { /* no action */ }

    /* Overmodulation guard: rescale if T1 + T2 > 1. */
    sum = *T1Out_P + *T2Out_P;
    if (sum > ES_MATH_ONE_F)
    {
        *T1Out_P = *T1Out_P / sum;
        *T2Out_P = *T2Out_P / sum;
    }
    else
    {
        /* No action — within linear modulation range. */
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_CalculateDutyFromTimes
 *------------------------------------------------------------------------------------------------------------------*/
static void SVM_CalculateDutyFromTimes(
    MatrixFloat           T1,
    MatrixFloat           T2,
    SVM_Sector_T          Sector,
    SVM_DutyCycle_T * const DutyOut_P)
{
    MatrixFloat t0;
    MatrixFloat ta;
    MatrixFloat tb;
    MatrixFloat tc;

    t0 = (ES_MATH_ONE_F - T1 - T2) * ES_MATH_HALF_F;
    if (t0 < SVM_ZERO_F)
    {
        t0 = SVM_ZERO_F;
    }
    else
    {
        /* No action — t0 already valid. */
    }

    switch (Sector)
    {
        case SVM_SECTOR_I:
            ta = T1 + T2 + t0;
            tb = T2 + t0;
            tc = t0;
            break;

        case SVM_SECTOR_II:
            ta = T1 + t0;
            tb = T1 + T2 + t0;
            tc = t0;
            break;

        case SVM_SECTOR_III:
            ta = t0;
            tb = T1 + T2 + t0;
            tc = T2 + t0;
            break;

        case SVM_SECTOR_IV:
            ta = t0;
            tb = T1 + t0;
            tc = T1 + T2 + t0;
            break;

        case SVM_SECTOR_V:
            ta = T2 + t0;
            tb = t0;
            tc = T1 + T2 + t0;
            break;

        case SVM_SECTOR_VI:
            ta = T1 + T2 + t0;
            tb = t0;
            tc = T1 + t0;
            break;

        default:
            ta = SVM_ZERO_F;
            tb = SVM_ZERO_F;
            tc = SVM_ZERO_F;
            break;
    }

    DutyOut_P->Ta = Matrix_FloatToQ31(SVM_ClampFloat(ta));
    DutyOut_P->Tb = Matrix_FloatToQ31(SVM_ClampFloat(tb));
    DutyOut_P->Tc = Matrix_FloatToQ31(SVM_ClampFloat(tc));
    DutyOut_P->Sector = Sector;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_AlphaBetaToModIndex
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat SVM_AlphaBetaToModIndex(const FocAlphaBeta_T * const V_AlphaBeta_P)
{
    MatrixFloat mod_index;
    MatrixFloat magnitude_sq;

    if (V_AlphaBeta_P != NULL)
    {
        /* |V|² = Vα² + Vβ² */
        magnitude_sq = (V_AlphaBeta_P->Alpha * V_AlphaBeta_P->Alpha) +
                       (V_AlphaBeta_P->Beta  * V_AlphaBeta_P->Beta);

        /* |V| = sqrt(|V|²) */
        mod_index = sqrtf(magnitude_sq);

        /* Clamp to maximum linear range (√3/2) and normalise */
        if (mod_index > SVM_MAX_LINEAR_MOD)
        {
            mod_index = SVM_MAX_LINEAR_MOD;
        }
        else
        {
            /* No action */
        }

        /* Normalise to [0, 1] range */
        mod_index = mod_index / SVM_MAX_LINEAR_MOD;
    }
    else
    {
        mod_index = SVM_ZERO_F;
    }

    return mod_index;
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_Init
 *------------------------------------------------------------------------------------------------------------------*/
void SVM_Init(void)
{
    /* Initialize the coordinate transform matrices */
    Transform_Init();
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_CalculateDutyCycleFromAlphaBeta
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_CalculateDutyCycleFromAlphaBeta(
    const FocAlphaBeta_T * const V_AlphaBeta_P,
    const FocAngle_T     * const Angle_P,
    SVM_DutyCycle_T      * const DutyOut_P)
{
    MatrixFloat       t1;
    MatrixFloat       t2;
    MatrixFloat       angle_rad;
    MatrixFloat       mod_index;
    SVM_Sector_T      sector;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((V_AlphaBeta_P == NULL) || (Angle_P == NULL) || (DutyOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        angle_rad = Angle_P->ThetaE;
        sector    = SVM_GetSectorFromAngle(angle_rad);
        mod_index = SVM_AlphaBetaToModIndex(V_AlphaBeta_P);

        SVM_CalculateTimes(sector, angle_rad, mod_index, &t1, &t2);
        SVM_CalculateDutyFromTimes(t1, t2, sector, DutyOut_P);
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_CalculateDutyCycle
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_CalculateDutyCycle(
    MatrixFloat                    ModIndex,
    const FocAngle_T     * const   Angle_P,
    SVM_DutyCycle_T      * const   DutyOut_P)
{
    MatrixFloat       t1;
    MatrixFloat       t2;
    MatrixFloat       angle_rad;
    SVM_Sector_T      sector;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((Angle_P == NULL) || (DutyOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((ModIndex < SVM_ZERO_F) || (ModIndex > ES_MATH_ONE_F))
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        angle_rad = Angle_P->ThetaE;
        sector    = SVM_GetSectorFromAngle(angle_rad);

        SVM_CalculateTimes(sector, angle_rad, ModIndex, &t1, &t2);
        SVM_CalculateDutyFromTimes(t1, t2, sector, DutyOut_P);
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_CalculateDutyCycleFromDq
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_CalculateDutyCycleFromDq(
    const FocDq_T        * const V_Dq_P,
    const FocAngle_T     * const Angle_P,
    SVM_DutyCycle_T      * const DutyOut_P)
{
    MatrixStatus_Type   status;
    FocAlphaBeta_T      v_alpha_beta;

    status = MATRIX_SUCCESS;

    if ((V_Dq_P == NULL) || (Angle_P == NULL) || (DutyOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* Convert dq to αβ using Inverse-Park transform (matrix-based) */
        status = InvPark_Transform_Matrix(V_Dq_P, Angle_P, &v_alpha_beta);

        if (status == MATRIX_SUCCESS)
        {
            /* Calculate duty cycles from αβ voltages */
            status = SVM_CalculateDutyCycleFromAlphaBeta(&v_alpha_beta, Angle_P, DutyOut_P);
        }
        else
        {
            /* Inverse-Park transform failed */
        }
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetSectorFromAlphaBeta
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_GetSectorFromAlphaBeta(
    const FocAlphaBeta_T * const V_AlphaBeta_P,
    SVM_Sector_T         * const SectorOut_P)
{
    MatrixFloat       angle_rad;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((V_AlphaBeta_P == NULL) || (SectorOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* Calculate angle from αβ components using atan2 */
        angle_rad = atan2f(V_AlphaBeta_P->Beta, V_AlphaBeta_P->Alpha);
        *SectorOut_P = SVM_GetSectorFromAngle(angle_rad);
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetSectorFromDQ
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_GetSectorFromDQ(
    MatrixElement              Vd,
    MatrixElement              Vq,
    SVM_Sector_T     * const   SectorOut_P)
{
    MatrixFloat       vd_f;
    MatrixFloat       vq_f;
    MatrixFloat       sqrt3_vd;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if (SectorOut_P == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        vd_f     = Matrix_Q31ToFloat(Vd);
        vq_f     = Matrix_Q31ToFloat(Vq);
        sqrt3_vd = SVM_SQRT3_F * vd_f;

        if (vq_f >= SVM_ZERO_F)
        {
            if (vq_f > sqrt3_vd)
            {
                *SectorOut_P = SVM_SECTOR_II;
            }
            else if (vq_f > -sqrt3_vd)
            {
                *SectorOut_P = SVM_SECTOR_I;
            }
            else
            {
                *SectorOut_P = SVM_SECTOR_VI;
            }
        }
        else
        {
            if (vq_f < -sqrt3_vd)
            {
                *SectorOut_P = SVM_SECTOR_V;
            }
            else if (vq_f < sqrt3_vd)
            {
                *SectorOut_P = SVM_SECTOR_IV;
            }
            else
            {
                *SectorOut_P = SVM_SECTOR_III;
            }
        }
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetCompareValues
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_GetCompareValues(
    const SVM_DutyCycle_T  * const DutyIn_P,
    uint32_T                       TimerPeriod,
    uint32_T               * const CompAOut_P,
    uint32_T               * const CompBOut_P,
    uint32_T               * const CompCOut_P)
{
    uint32_T          ta_ticks;
    uint32_T          tb_ticks;
    uint32_T          tc_ticks;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((DutyIn_P   == NULL) ||
        (CompAOut_P == NULL) ||
        (CompBOut_P == NULL) ||
        (CompCOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (TimerPeriod == 0U)
    {
        status = MATRIX_ERROR_DIV_BY_ZERO;
    }
    else
    {
        /* Scale Q31 → ticks via 64-bit intermediate */
        ta_ticks = (uint32_T)(((uint64_T)(uint32_T)DutyIn_P->Ta * (uint64_T)TimerPeriod)
                              / (uint64_T)SVM_Q31_ONE);
        tb_ticks = (uint32_T)(((uint64_T)(uint32_T)DutyIn_P->Tb * (uint64_T)TimerPeriod)
                              / (uint64_T)SVM_Q31_ONE);
        tc_ticks = (uint32_T)(((uint64_T)(uint32_T)DutyIn_P->Tc * (uint64_T)TimerPeriod)
                              / (uint64_T)SVM_Q31_ONE);

        /* Centre-aligned compare = (Period − OnTime) / 2 */
        *CompAOut_P = (TimerPeriod - ta_ticks) / 2U;
        *CompBOut_P = (TimerPeriod - tb_ticks) / 2U;
        *CompCOut_P = (TimerPeriod - tc_ticks) / 2U;
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetDutyCyclesFloat
 *------------------------------------------------------------------------------------------------------------------*/
void SVM_GetDutyCyclesFloat(
    const SVM_DutyCycle_T  * const DutyIn_P,
    MatrixFloat            * const TaOut_P,
    MatrixFloat            * const TbOut_P,
    MatrixFloat            * const TcOut_P)
{
    if ((DutyIn_P != NULL) &&
        (TaOut_P  != NULL) &&
        (TbOut_P  != NULL) &&
        (TcOut_P  != NULL))
    {
        *TaOut_P = Matrix_Q31ToFloat(DutyIn_P->Ta);
        *TbOut_P = Matrix_Q31ToFloat(DutyIn_P->Tb);
        *TcOut_P = Matrix_Q31ToFloat(DutyIn_P->Tc);
    }
    else
    {
        /* NULL pointer detected — function does nothing. */
    }
}
