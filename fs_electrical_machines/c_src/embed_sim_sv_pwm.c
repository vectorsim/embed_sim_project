/**********************************************************************************************************************
 * \file      SV_PWM.c
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation implementation.
 *
 * Implements symmetric centre-aligned SVPWM (T0 = T7).  All sector
 * arithmetic is performed in float; results are converted to Q31 before
 * being stored in the output structure.
 *
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sv_pwm.h"
#include <math.h>
#include <stddef.h>


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
#define SVM_HALF_F   (0.5f)  /**< 1/2  */
#define SVM_ONE_F    (1.0f)  /**< 1.0  */
#define SVM_ZERO_F   (0.0f)  /**< 0.0  */


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Determine the SVPWM sector from an electrical angle.
 *
 * The angle is first wrapped into [0, 2π) before the 60°-wide sector
 * boundary comparisons are applied.
 *
 * \param[in] angle_rad  Electrical angle in radians (float, any value).
 * \return               Resolved #SVM_Sector_Type (I – VI).
 */
static SVM_Sector_Type SVM_GetSectorFromAngle(const MatrixFloat angle_rad);

/**
 * \brief  Calculate normalised active-vector times T1 and T2.
 *
 * Computes T1 and T2 for the given sector using the cosine projection
 * formulae from Table 9.3.  Both outputs are clamped to [0, 1] and
 * rescaled if overmodulation is detected (T1 + T2 > 1).
 *
 * \param[in]  sector     Active sector (I – VI).
 * \param[in]  angle_rad  Electrical angle in radians (float).
 * \param[in]  m          Modulation index (float, 0.0 – 1.0).
 * \param[out] t1         Normalised active-vector time T1 (float).
 * \param[out] t2         Normalised active-vector time T2 (float).
 */
static void SVM_CalculateTimes(
    const SVM_Sector_Type  sector,
    const MatrixFloat      angle_rad,
    const MatrixFloat      m,
    MatrixFloat    * const t1,
    MatrixFloat    * const t2);

/**
 * \brief  Clamp a float value to [0.0, 1.0].
 *
 * \param[in] value  Input value.
 * \return           Value clamped to [0.0, 1.0].
 */
static MatrixFloat SVM_ClampFloat(const MatrixFloat value);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * SVM_ClampFloat
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat SVM_ClampFloat(const MatrixFloat value)
{
    MatrixFloat result;

    result = value;

    if (result < SVM_ZERO_F)
    {
        result = SVM_ZERO_F;
    }
    else if (result > SVM_ONE_F)
    {
        result = SVM_ONE_F;
    }
    else
    {
        /* Value already within [0, 1] – no action required. */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetSectorFromAngle
 *------------------------------------------------------------------------------------------------------------------*/
static SVM_Sector_Type SVM_GetSectorFromAngle(const MatrixFloat angle_rad)
{
    MatrixFloat     angle_norm;
    SVM_Sector_Type sector;

    /* Wrap angle into [0, 2π). */
    angle_norm = angle_rad;

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
    const SVM_Sector_Type  sector,
    const MatrixFloat      angle_rad,
    const MatrixFloat      m,
    MatrixFloat    * const t1,
    MatrixFloat    * const t2)
{
    MatrixFloat cos_t1;
    MatrixFloat cos_t2;
    MatrixFloat scale;

    scale  = SVM_SQRT3_OVER_2_F * m;
    *t1    = SVM_ZERO_F;
    *t2    = SVM_ZERO_F;

    /* Cosine projections per sector (Table 9.3). */
    switch (sector)
    {
        case SVM_SECTOR_I:
        {
            cos_t1 = cosf(angle_rad + SVM_PI_OVER_6_F);
            cos_t2 = cosf(angle_rad - SVM_PI_OVER_2_F);
            break;
        }

        case SVM_SECTOR_II:
        {
            cos_t1 = cosf(angle_rad - SVM_PI_OVER_6_F);
            cos_t2 = cosf(angle_rad - (5.0f * SVM_PI_OVER_6_F));
            break;
        }

        case SVM_SECTOR_III:
        {
            cos_t1 = cosf(angle_rad - SVM_PI_OVER_2_F);
            cos_t2 = cosf(angle_rad - (7.0f * SVM_PI_OVER_6_F));
            break;
        }

        case SVM_SECTOR_IV:
        {
            cos_t1 = cosf(angle_rad - (5.0f * SVM_PI_OVER_6_F));
            cos_t2 = cosf(angle_rad - (3.0f * SVM_PI_OVER_2_F));
            break;
        }

        case SVM_SECTOR_V:
        {
            cos_t1 = cosf(angle_rad - (7.0f  * SVM_PI_OVER_6_F));
            cos_t2 = cosf(angle_rad - (11.0f * SVM_PI_OVER_6_F));
            break;
        }

        case SVM_SECTOR_VI:
        {
            cos_t1 = cosf(angle_rad - (3.0f * SVM_PI_OVER_2_F));
            cos_t2 = cosf(angle_rad - SVM_PI_OVER_6_F);
            break;
        }

        default:
        {
            /* MISRA C 2012 Rule 16.4: default clause required. */
            cos_t1 = SVM_ZERO_F;
            cos_t2 = SVM_ZERO_F;
            break;
        }
    }

    *t1 = scale * cos_t1;
    *t2 = scale * cos_t2;

    /* Clamp individual times to non-negative. */
    if (*t1 < SVM_ZERO_F)
    {
        *t1 = SVM_ZERO_F;
    }
    else
    {
        /* No action – already non-negative. */
    }

    if (*t2 < SVM_ZERO_F)
    {
        *t2 = SVM_ZERO_F;
    }
    else
    {
        /* No action – already non-negative. */
    }

    /* Overmodulation protection: rescale so that T1 + T2 ≤ 1. */
    if ((*t1 + *t2) > SVM_ONE_F)
    {
        MatrixFloat inv_sum;

        inv_sum = SVM_ONE_F / (*t1 + *t2);
        *t1    *= inv_sum;
        *t2    *= inv_sum;
    }
    else
    {
        /* No action – within linear modulation range. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVM_CalculateDutyCycle
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_CalculateDutyCycle(
    const MatrixFloat          modulation_index,
    const MatrixFloat          angle_rad,
    SVM_DutyCycle_Type * const duty)
{
    MatrixFloat     t1;
    MatrixFloat     t2;
    MatrixFloat     t0;
    MatrixFloat     ta;
    MatrixFloat     tb;
    MatrixFloat     tc;
    SVM_Sector_Type sector;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if (duty == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((modulation_index < SVM_ZERO_F) || (modulation_index > SVM_ONE_F))
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        /* Resolve sector and compute active-vector times. */
        sector       = SVM_GetSectorFromAngle(angle_rad);
        duty->sector = sector;

        SVM_CalculateTimes(sector, angle_rad, modulation_index, &t1, &t2);

        /* Zero-vector time; split symmetrically (T0 = T7). */
        t0 = (SVM_ONE_F - t1 - t2) * SVM_HALF_F;

        if (t0 < SVM_ZERO_F)
        {
            t0 = SVM_ZERO_F;
        }
        else
        {
            /* No action – t0 already valid. */
        }

        /* Assemble per-sector duty cycles. */
        switch (sector)
        {
            case SVM_SECTOR_I:
            {
                ta = t1 + t2 + t0;
                tb = t2 + t0;
                tc = t0;
                break;
            }

            case SVM_SECTOR_II:
            {
                ta = t1 + t0;
                tb = t1 + t2 + t0;
                tc = t0;
                break;
            }

            case SVM_SECTOR_III:
            {
                ta = t0;
                tb = t1 + t2 + t0;
                tc = t2 + t0;
                break;
            }

            case SVM_SECTOR_IV:
            {
                ta = t0;
                tb = t1 + t0;
                tc = t1 + t2 + t0;
                break;
            }

            case SVM_SECTOR_V:
            {
                ta = t2 + t0;
                tb = t0;
                tc = t1 + t2 + t0;
                break;
            }

            case SVM_SECTOR_VI:
            {
                ta = t1 + t2 + t0;
                tb = t0;
                tc = t1 + t0;
                break;
            }

            default:
            {
                /* MISRA C 2012 Rule 16.4: default clause required. */
                status = MATRIX_ERROR_OUT_OF_BOUNDS;
                ta     = SVM_ZERO_F;
                tb     = SVM_ZERO_F;
                tc     = SVM_ZERO_F;
                break;
            }
        }

        if (status == MATRIX_SUCCESS)
        {
            /* Clamp to [0, 1] then store as Q31. */
            duty->ta = Matrix_FloatToQ31(SVM_ClampFloat(ta));
            duty->tb = Matrix_FloatToQ31(SVM_ClampFloat(tb));
            duty->tc = Matrix_FloatToQ31(SVM_ClampFloat(tc));
        }
        else
        {
            /* No action – error already captured. */
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetSectorFromDQ
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_GetSectorFromDQ(
    const MatrixElement        vd,
    const MatrixElement        vq,
    SVM_Sector_Type    * const sector)
{
    MatrixFloat       vd_f;
    MatrixFloat       vq_f;
    MatrixFloat       sqrt3_vd;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if (sector == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        vd_f      = Matrix_Q31ToFloat(vd);
        vq_f      = Matrix_Q31ToFloat(vq);
        sqrt3_vd  = SVM_SQRT3_F * vd_f;

        /* Upper half-plane (vq ≥ 0): sectors I, II, VI.
         * Lower half-plane (vq < 0): sectors III, IV, V. */
        if (vq_f >= SVM_ZERO_F)
        {
            if (vq_f > sqrt3_vd)
            {
                *sector = SVM_SECTOR_II;
            }
            else if (vq_f > -sqrt3_vd)
            {
                *sector = SVM_SECTOR_I;
            }
            else
            {
                *sector = SVM_SECTOR_VI;
            }
        }
        else
        {
            if (vq_f < -sqrt3_vd)
            {
                *sector = SVM_SECTOR_V;
            }
            else if (vq_f < sqrt3_vd)
            {
                *sector = SVM_SECTOR_IV;
            }
            else
            {
                *sector = SVM_SECTOR_III;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetCompareValues
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type SVM_GetCompareValues(
    const SVM_DutyCycle_Type * const duty,
    const uint32_T                   timer_period,
    uint32_T               * const   compare_a,
    uint32_T               * const   compare_b,
    uint32_T               * const   compare_c)
{
    uint32_T          ta_ticks;
    uint32_T          tb_ticks;
    uint32_T          tc_ticks;
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((duty == NULL) || (compare_a == NULL) ||
        (compare_b == NULL) || (compare_c == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (timer_period == 0U)
    {
        status = MATRIX_ERROR_DIV_BY_ZERO;
    }
    else
    {
        /* Scale Q31 duty cycles to timer ticks via 64-bit intermediate. */
        ta_ticks = (uint32_T)(((uint64_T)duty->ta * (uint64_T)timer_period) / (uint64_T)Q31_ONE);
        tb_ticks = (uint32_T)(((uint64_T)duty->tb * (uint64_T)timer_period) / (uint64_T)Q31_ONE);
        tc_ticks = (uint32_T)(((uint64_T)duty->tc * (uint64_T)timer_period) / (uint64_T)Q31_ONE);

        /* Centre-aligned (up-down) PWM: compare = (period − on_time) / 2.
         * The high-side switch is on for on_time ticks, centred in the period. */
        *compare_a = (timer_period - ta_ticks) / 2U;
        *compare_b = (timer_period - tb_ticks) / 2U;
        *compare_c = (timer_period - tc_ticks) / 2U;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVM_GetDutyCyclesFloat
 *------------------------------------------------------------------------------------------------------------------*/
void SVM_GetDutyCyclesFloat(
    const SVM_DutyCycle_Type * const duty,
    MatrixFloat              * const ta,
    MatrixFloat              * const tb,
    MatrixFloat              * const tc)
{
    if ((duty != NULL) && (ta != NULL) && (tb != NULL) && (tc != NULL))
    {
        *ta = Matrix_Q31ToFloat(duty->ta);
        *tb = Matrix_Q31ToFloat(duty->tb);
        *tc = Matrix_Q31ToFloat(duty->tc);
    }
    else
    {
        /* MISRA C 2012 Rule 15.7: else clause required.
         * NULL pointer(s) detected – function does nothing. */
    }
}
