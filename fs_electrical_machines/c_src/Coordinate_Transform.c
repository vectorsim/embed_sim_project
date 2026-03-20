/**********************************************************************************************************************
 * \file      Coordinate_Transform.c
 * \brief     Clarke, Park, Inverse-Park and Inverse-Clarke transform implementations for FOC motor control.
 *
 * Working type: \c MatrixFloat (= \c real32_T from \c Matrix.h / \c Sys_Types.h).
 * Uses \c cosf and \c sinf from \c <math.h> — single-precision, no double promotion.
 *
 * \c Matrix.h is included for:
 *   - \c MatrixFloat type (= \c real32_T)
 *   - \c Matrix_FloatToQ31 / \c Matrix_Q31ToFloat at the application boundary
 *   - Consistent type infrastructure with all other \c fs_electrical_machines blocks
 *
 * MISRA C:2012 compliance notes:
 *   - No dynamic memory allocation
 *   - No recursion
 *   - Single exit per function (Rule 15.5)
 *   - All literals carry the \c f suffix — no implicit double promotion
 *   - NULL guards on all pointer arguments
 *
 * \version   1.1.0
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Coordinate_Transform.h"
#include <math.h>    /* cosf, sinf        */
#include <string.h>  /* memset            */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup ct_private_constants  Private numeric constants (MatrixFloat = real32_T)
 * \{
 */
/** \brief Power-invariant Clarke scale factor: 2/3. */
#define CT_TWO_THIRDS   ((MatrixFloat)0.66666667f)

/** \brief Power-invariant Clarke scale factor: 1/3. */
#define CT_ONE_THIRD    ((MatrixFloat)0.33333333f)

/** \brief 1 / √3 = 0.57735027… */
#define CT_INV_SQRT3    ((MatrixFloat)0.57735027f)

/** \brief √3 / 2 = 0.86602540… */
#define CT_HALF_SQRT3   ((MatrixFloat)0.86602540f)

/** \brief 0.5 */
#define CT_HALF         ((MatrixFloat)0.50000000f)
/** \} */


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
/* None — all functions are public; prototypes are in Coordinate_Transform.h */


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Clarke_Init
 *------------------------------------------------------------------------------------------------------------------*/
void Clarke_Init(Clarke_T * const s)
{
    if (s != NULL)
    {
        (void)memset(s, 0, sizeof(Clarke_T));
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Clarke_Step
 *
 * Power-invariant Clarke transform:
 *   i_α = (2/3)·i_a − (1/3)·i_b − (1/3)·i_c
 *   i_β = (1/√3)·(i_b − i_c)
 *------------------------------------------------------------------------------------------------------------------*/
void Clarke_Step(
    Clarke_T       * const s,
    MatrixFloat            ia,
    MatrixFloat            ib,
    MatrixFloat            ic,
    MatrixFloat    * const alpha_out,
    MatrixFloat    * const beta_out)
{
    (void)s;  /* Combinatorial block — state reserved for future pre-filter. */

    if ((alpha_out != NULL) && (beta_out != NULL))
    {
        *alpha_out = (CT_TWO_THIRDS * ia) - (CT_ONE_THIRD * ib) - (CT_ONE_THIRD * ic);
        *beta_out  =  CT_INV_SQRT3 * (ib - ic);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Park_Init
 *------------------------------------------------------------------------------------------------------------------*/
void Park_Init(Park_T * const s)
{
    if (s != NULL)
    {
        (void)memset(s, 0, sizeof(Park_T));
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Park_Step
 *
 * Park (forward) transform:
 *   i_d =  i_α·cos(θ) + i_β·sin(θ)
 *   i_q = −i_α·sin(θ) + i_β·cos(θ)
 *------------------------------------------------------------------------------------------------------------------*/
void Park_Step(
    Park_T         * const s,
    MatrixFloat            alpha,
    MatrixFloat            beta,
    MatrixFloat            theta,
    MatrixFloat    * const d_out,
    MatrixFloat    * const q_out)
{
    MatrixFloat cos_t;
    MatrixFloat sin_t;

    (void)s;  /* Combinatorial block — state unused. */

    if ((d_out != NULL) && (q_out != NULL))
    {
        cos_t = cosf(theta);
        sin_t = sinf(theta);

        *d_out = ( alpha * cos_t) + (beta * sin_t);
        *q_out = (-alpha * sin_t) + (beta * cos_t);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * InvPark_Init
 *------------------------------------------------------------------------------------------------------------------*/
void InvPark_Init(InvPark_T * const s)
{
    if (s != NULL)
    {
        (void)memset(s, 0, sizeof(InvPark_T));
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * InvPark_Step
 *
 * Inverse-Park transform:
 *   v_α = v_d·cos(θ) − v_q·sin(θ)
 *   v_β = v_d·sin(θ) + v_q·cos(θ)
 *------------------------------------------------------------------------------------------------------------------*/
void InvPark_Step(
    InvPark_T      * const s,
    MatrixFloat            d,
    MatrixFloat            q,
    MatrixFloat            theta,
    MatrixFloat    * const alpha_out,
    MatrixFloat    * const beta_out)
{
    MatrixFloat cos_t;
    MatrixFloat sin_t;

    (void)s;  /* Combinatorial block — state unused. */

    if ((alpha_out != NULL) && (beta_out != NULL))
    {
        cos_t = cosf(theta);
        sin_t = sinf(theta);

        *alpha_out = (d * cos_t) - (q * sin_t);
        *beta_out  = (d * sin_t) + (q * cos_t);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * InvClarke_Init
 *------------------------------------------------------------------------------------------------------------------*/
void InvClarke_Init(InvClarke_T * const s)
{
    if (s != NULL)
    {
        (void)memset(s, 0, sizeof(InvClarke_T));
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * InvClarke_Step
 *
 * Inverse-Clarke transform:
 *   v_a =  v_α
 *   v_b = −(1/2)·v_α + (√3/2)·v_β
 *   v_c = −(1/2)·v_α − (√3/2)·v_β
 *------------------------------------------------------------------------------------------------------------------*/
void InvClarke_Step(
    InvClarke_T    * const s,
    MatrixFloat            alpha,
    MatrixFloat            beta,
    MatrixFloat    * const va_out,
    MatrixFloat    * const vb_out,
    MatrixFloat    * const vc_out)
{
    (void)s;  /* Combinatorial block — state unused. */

    if ((va_out != NULL) && (vb_out != NULL) && (vc_out != NULL))
    {
        *va_out =  alpha;
        *vb_out = (-CT_HALF * alpha) + (CT_HALF_SQRT3 * beta);
        *vc_out = (-CT_HALF * alpha) - (CT_HALF_SQRT3 * beta);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required. */
    }
}
