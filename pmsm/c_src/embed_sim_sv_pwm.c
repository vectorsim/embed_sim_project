/**********************************************************************************************************************
 * \file      embed_sim_sv_pwm.c
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation implementation.
 *********************************************************************************************************************/

#include "embed_sim_sv_pwm.h"
#include <math.h>
#include <stddef.h>

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/
#define SVM_ZERO_F   ((MatrixFloat)0.0f)

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
static SVM_Sector_T SVM_GetSectorFromAngle(MatrixFloat AngleRad);

static void SVM_CalculateTimes(
    SVM_Sector_T     ActiveSector,
    MatrixFloat      AngleRad,
    MatrixFloat      ModIndex,
    MatrixFloat    * const T1Out_P,
    MatrixFloat    * const T2Out_P);

static MatrixFloat SVM_ClampFloat(MatrixFloat Value);

static void SVM_CalculateDutyFromTimes(
    MatrixFloat           T1,
    MatrixFloat           T2,
    SVM_Sector_T          Sector,
    SVM_DutyCycle_T * const DutyOut_P);

/**********************************************************************************************************************
 * Private Function Implementations
 *********************************************************************************************************************/

static MatrixFloat SVM_ClampFloat(MatrixFloat Value)
{
    MatrixFloat result = Value;
    if (result < SVM_ZERO_F) result = SVM_ZERO_F;
    else if (result > ES_MATH_ONE_F) result = ES_MATH_ONE_F;
    return result;
}

static SVM_Sector_T SVM_GetSectorFromAngle(MatrixFloat AngleRad)
{
    MatrixFloat angle_norm = AngleRad;
    SVM_Sector_T sector;

    while (angle_norm < SVM_ZERO_F) angle_norm += SVM_2PI_F;
    while (angle_norm >= SVM_2PI_F) angle_norm -= SVM_2PI_F;

    if (angle_norm < SVM_PI_OVER_3_F)
        sector = SVM_SECTOR_I;
    else if (angle_norm < SVM_2PI_OVER_3_F)
        sector = SVM_SECTOR_II;
    else if (angle_norm < SVM_PI_F)
        sector = SVM_SECTOR_III;
    else if (angle_norm < SVM_4PI_OVER_3_F)
        sector = SVM_SECTOR_IV;
    else if (angle_norm < SVM_5PI_OVER_3_F)
        sector = SVM_SECTOR_V;
    else
        sector = SVM_SECTOR_VI;

    return sector;
}

static void SVM_CalculateTimes(
    SVM_Sector_T     ActiveSector,
    MatrixFloat      AngleRad,
    MatrixFloat      ModIndex,
    MatrixFloat    * const T1Out_P,
    MatrixFloat    * const T2Out_P)
{
    MatrixFloat cos_t1 = SVM_ZERO_F;
    MatrixFloat cos_t2 = SVM_ZERO_F;
    MatrixFloat scale = SVM_SQRT3_OVER_2_F * ModIndex;
    MatrixFloat sum;

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
            cos_t1 = cosf(AngleRad - (7.0f * SVM_PI_OVER_6_F));
            cos_t2 = cosf(AngleRad - (11.0f * SVM_PI_OVER_6_F));
            break;
        case SVM_SECTOR_VI:
            cos_t1 = cosf(AngleRad - (3.0f * SVM_PI_OVER_2_F));
            cos_t2 = cosf(AngleRad - SVM_PI_OVER_6_F);
            break;
        default:
            break;
    }

    *T1Out_P = scale * cos_t1;
    *T2Out_P = scale * cos_t2;

    if (*T1Out_P < SVM_ZERO_F) *T1Out_P = SVM_ZERO_F;
    if (*T2Out_P < SVM_ZERO_F) *T2Out_P = SVM_ZERO_F;

    sum = *T1Out_P + *T2Out_P;
    if (sum > ES_MATH_ONE_F)
    {
        *T1Out_P = *T1Out_P / sum;
        *T2Out_P = *T2Out_P / sum;
    }
}

static void SVM_CalculateDutyFromTimes(
    MatrixFloat           T1,
    MatrixFloat           T2,
    SVM_Sector_T          Sector,
    SVM_DutyCycle_T * const DutyOut_P)
{
    MatrixFloat t0 = (ES_MATH_ONE_F - T1 - T2) * ES_MATH_HALF_F;
    MatrixFloat ta, tb, tc;

    if (t0 < SVM_ZERO_F) t0 = SVM_ZERO_F;

    switch (Sector)
    {
        case SVM_SECTOR_I:
            ta = T1 + T2 + t0; tb = T2 + t0; tc = t0;
            break;
        case SVM_SECTOR_II:
            ta = T1 + t0; tb = T1 + T2 + t0; tc = t0;
            break;
        case SVM_SECTOR_III:
            ta = t0; tb = T1 + T2 + t0; tc = T2 + t0;
            break;
        case SVM_SECTOR_IV:
            ta = t0; tb = T1 + t0; tc = T1 + T2 + t0;
            break;
        case SVM_SECTOR_V:
            ta = T2 + t0; tb = t0; tc = T1 + T2 + t0;
            break;
        case SVM_SECTOR_VI:
            ta = T1 + T2 + t0; tb = t0; tc = T1 + t0;
            break;
        default:
            ta = SVM_ZERO_F; tb = SVM_ZERO_F; tc = SVM_ZERO_F;
            break;
    }

    DutyOut_P->Ta = SVM_ClampFloat(ta);
    DutyOut_P->Tb = SVM_ClampFloat(tb);
    DutyOut_P->Tc = SVM_ClampFloat(tc);
    DutyOut_P->Sector = Sector;
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

void SVM_Init(void)
{
    Transform_Init();
}

MatrixStatus_T SVM_CalculateDutyCycle(
    MatrixFloat                    ModIndex,
    const FocAngle_T     * const   Angle_P,
    SVM_DutyCycle_T      * const   DutyOut_P)
{
    MatrixFloat       t1, t2, angle_rad;
    SVM_Sector_T      sector;
    MatrixStatus_T status = MATRIX_SUCCESS;

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
        sector = SVM_GetSectorFromAngle(angle_rad);
        SVM_CalculateTimes(sector, angle_rad, ModIndex, &t1, &t2);
        SVM_CalculateDutyFromTimes(t1, t2, sector, DutyOut_P);
    }

    return status;
}

MatrixStatus_T SVM_CalculateDutyCycleFromAlphaBeta(
    const FocAlphaBeta_T * const V_AlphaBeta_P,
    const FocAngle_T     * const Angle_P,
    MatrixFloat                  Vdc,
    SVM_DutyCycle_T      * const DutyOut_P)
{
    MatrixFloat       t1, t2, angle_rad, mod_index, magnitude;
    SVM_Sector_T      sector;
    MatrixStatus_T status = MATRIX_SUCCESS;

    if ((V_AlphaBeta_P == NULL) || (Angle_P == NULL) || (DutyOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Vdc <= 0.0F)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        /* Calculate magnitude of αβ voltage */
        magnitude = sqrtf((V_AlphaBeta_P->Alpha * V_AlphaBeta_P->Alpha) +
                          (V_AlphaBeta_P->Beta  * V_AlphaBeta_P->Beta));

        /* Normalize by Vdc/√3 (SVPWM linear range) */
        MatrixFloat Vphase_max = Vdc / SVM_SQRT3_F;
        mod_index = magnitude / Vphase_max;

        /* Clamp to [0, 1] */
        if (mod_index > ES_MATH_ONE_F) mod_index = ES_MATH_ONE_F;
        else if (mod_index < SVM_ZERO_F) mod_index = SVM_ZERO_F;

        angle_rad = Angle_P->ThetaE;
        sector = SVM_GetSectorFromAngle(angle_rad);
        SVM_CalculateTimes(sector, angle_rad, mod_index, &t1, &t2);
        SVM_CalculateDutyFromTimes(t1, t2, sector, DutyOut_P);
    }

    return status;
}

MatrixStatus_T SVM_CalculateDutyCycleFromDq(
    const FocDq_T        * const V_Dq_P,
    const FocAngle_T     * const Angle_P,
    MatrixFloat                  Vdc,
    SVM_DutyCycle_T      * const DutyOut_P)
{
    MatrixStatus_T status = MATRIX_SUCCESS;
    FocAlphaBeta_T v_alpha_beta;

    if ((V_Dq_P == NULL) || (Angle_P == NULL) || (DutyOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Vdc <= 0.0F)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        /* dq → αβ using Inverse-Park */
        status = InvPark_Transform_Matrix(V_Dq_P, Angle_P, &v_alpha_beta);

        if (status == MATRIX_SUCCESS)
        {
            /* αβ → SVPWM duty cycles with Vdc */
            status = SVM_CalculateDutyCycleFromAlphaBeta(&v_alpha_beta, Angle_P, Vdc, DutyOut_P);
        }
    }

    return status;
}

MatrixStatus_T SVM_GetCompareValues(
    const SVM_DutyCycle_T  * const DutyIn_P,
    uint32_T                       TimerPeriod,
    uint32_T               * const CompAOut_P,
    uint32_T               * const CompBOut_P,
    uint32_T               * const CompCOut_P)
{
    uint32_T ta_ticks, tb_ticks, tc_ticks;
    MatrixStatus_T status = MATRIX_SUCCESS;

    if ((DutyIn_P == NULL) || (CompAOut_P == NULL) || (CompBOut_P == NULL) || (CompCOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (TimerPeriod == 0U)
    {
        status = MATRIX_ERROR_DIV_BY_ZERO;
    }
    else
    {
        ta_ticks = (uint32_T)(DutyIn_P->Ta * (MatrixFloat)TimerPeriod);
        tb_ticks = (uint32_T)(DutyIn_P->Tb * (MatrixFloat)TimerPeriod);
        tc_ticks = (uint32_T)(DutyIn_P->Tc * (MatrixFloat)TimerPeriod);

        *CompAOut_P = (TimerPeriod - ta_ticks) / 2U;
        *CompBOut_P = (TimerPeriod - tb_ticks) / 2U;
        *CompCOut_P = (TimerPeriod - tc_ticks) / 2U;
    }

    return status;
}
