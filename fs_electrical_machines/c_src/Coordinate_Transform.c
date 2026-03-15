/*
 * Coordinate_Transform.c
 * ======================
 * Clarke / Park / InvPark / InvClarke transforms for FOC motor control.
 *
 * Working type: MatrixFloat (= real32_T from Matrix.h / Sys_Types.h).
 * Uses cosf / sinf from <math.h> — single-precision, no double promotion.
 *
 * Matrix.h is included for:
 *   - MatrixFloat type (= real32_T)
 *   - Matrix_FloatToQ31 / Matrix_Q31ToFloat at application boundary
 *   - Consistent type infrastructure with all other fs_electrical_machines blocks
 *
 * MISRA C:2012:
 *   - No dynamic memory
 *   - No recursion
 *   - Single exit per function (MISRA 15.5)
 *   - All literals typed with 'f' suffix — no implicit double promotion
 *   - NULL guards on all pointer arguments
 *
 * Version: 1.1.0
 */

#include "Coordinate_Transform.h"
#include <math.h>    /* cosf, sinf */
#include <string.h>  /* memset     */

/* ============================================================================
 * Private constants  (MatrixFloat = real32_T)
 * ==========================================================================*/

/** Power-invariant Clarke: 2/3 */
#define CT_TWO_THIRDS    ((MatrixFloat)0.66666667f)

/** Power-invariant Clarke: 1/3 */
#define CT_ONE_THIRD     ((MatrixFloat)0.33333333f)

/** 1 / sqrt(3) */
#define CT_INV_SQRT3     ((MatrixFloat)0.57735027f)

/** sqrt(3) / 2 */
#define CT_HALF_SQRT3    ((MatrixFloat)0.86602540f)

/** 0.5 */
#define CT_HALF          ((MatrixFloat)0.50000000f)

/* ============================================================================
 * Clarke transform
 * ==========================================================================*/

void Clarke_Init(Clarke_T * const pS)
{
    if (pS != NULL)
    {
        (void)memset(pS, 0, sizeof(Clarke_T));
    }
}

void Clarke_Step(Clarke_T * const pS,
                 MatrixFloat ia, MatrixFloat ib, MatrixFloat ic,
                 MatrixFloat * const alpha_out,
                 MatrixFloat * const beta_out)
{
    (void)pS;   /* combinatorial — state reserved for future pre-filter */

    if ((alpha_out != NULL) && (beta_out != NULL))
    {
        /*
         * Power-invariant Clarke:
         *   i_alpha = (2/3)*ia - (1/3)*ib - (1/3)*ic
         *   i_beta  = (1/sqrt(3))*(ib - ic)
         */
        *alpha_out = (CT_TWO_THIRDS * ia) - (CT_ONE_THIRD * ib) - (CT_ONE_THIRD * ic);
        *beta_out  = CT_INV_SQRT3 * (ib - ic);
    }
}

/* ============================================================================
 * Park transform
 * ==========================================================================*/

void Park_Init(Park_T * const pS)
{
    if (pS != NULL)
    {
        (void)memset(pS, 0, sizeof(Park_T));
    }
}

void Park_Step(Park_T * const pS,
               MatrixFloat alpha, MatrixFloat beta, MatrixFloat theta,
               MatrixFloat * const d_out,
               MatrixFloat * const q_out)
{
    MatrixFloat cos_t;
    MatrixFloat sin_t;

    (void)pS;

    if ((d_out != NULL) && (q_out != NULL))
    {
        cos_t = cosf(theta);
        sin_t = sinf(theta);

        /*
         * Park:
         *   i_d =  i_alpha * cos(θ) + i_beta * sin(θ)
         *   i_q = -i_alpha * sin(θ) + i_beta * cos(θ)
         */
        *d_out = ( alpha * cos_t) + (beta * sin_t);
        *q_out = (-alpha * sin_t) + (beta * cos_t);
    }
}

/* ============================================================================
 * Inverse Park transform
 * ==========================================================================*/

void InvPark_Init(InvPark_T * const pS)
{
    if (pS != NULL)
    {
        (void)memset(pS, 0, sizeof(InvPark_T));
    }
}

void InvPark_Step(InvPark_T * const pS,
                  MatrixFloat d, MatrixFloat q, MatrixFloat theta,
                  MatrixFloat * const alpha_out,
                  MatrixFloat * const beta_out)
{
    MatrixFloat cos_t;
    MatrixFloat sin_t;

    (void)pS;

    if ((alpha_out != NULL) && (beta_out != NULL))
    {
        cos_t = cosf(theta);
        sin_t = sinf(theta);

        /*
         * Inverse Park:
         *   v_alpha = v_d * cos(θ) - v_q * sin(θ)
         *   v_beta  = v_d * sin(θ) + v_q * cos(θ)
         */
        *alpha_out = (d * cos_t) - (q * sin_t);
        *beta_out  = (d * sin_t) + (q * cos_t);
    }
}

/* ============================================================================
 * Inverse Clarke transform
 * ==========================================================================*/

void InvClarke_Init(InvClarke_T * const pS)
{
    if (pS != NULL)
    {
        (void)memset(pS, 0, sizeof(InvClarke_T));
    }
}

void InvClarke_Step(InvClarke_T * const pS,
                    MatrixFloat alpha, MatrixFloat beta,
                    MatrixFloat * const va_out,
                    MatrixFloat * const vb_out,
                    MatrixFloat * const vc_out)
{
    (void)pS;

    if ((va_out != NULL) && (vb_out != NULL) && (vc_out != NULL))
    {
        /*
         * Inverse Clarke:
         *   v_a =  v_alpha
         *   v_b = -(1/2)*v_alpha + (sqrt(3)/2)*v_beta
         *   v_c = -(1/2)*v_alpha - (sqrt(3)/2)*v_beta
         */
        *va_out =  alpha;
        *vb_out = (-CT_HALF * alpha) + (CT_HALF_SQRT3 * beta);
        *vc_out = (-CT_HALF * alpha) - (CT_HALF_SQRT3 * beta);
    }
}
