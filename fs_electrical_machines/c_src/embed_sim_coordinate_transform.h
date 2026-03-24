/**********************************************************************************************************************
 * \file      embed_sim_coordinate_transform.h
 * \brief     Clarke, Park, Inverse-Park and Inverse-Clarke transforms for FOC motor control.
 *
 * Uses \c MatrixFloat (= \c real32_T) from \c Matrix.h as the working type.
 * \c Matrix.h already includes \c Sys_Types.h — no duplicate include required.
 *
 * Signal conventions:
 * \code
 *   Clarke    : [i_a,  i_b,  i_c]  →  [i_α, i_β]         (power-invariant)
 *   Park      : [i_α,  i_β,  θ_e]  →  [i_d, i_q]
 *   InvPark   : [v_d,  v_q,  θ_e]  →  [v_α, v_β]
 *   InvClarke : [v_α,  v_β       ]  →  [v_a, v_b, v_c]
 * \endcode
 *
 * Constraints:
 *   - MISRA C:2012 compliant
 *   - No dynamic memory allocation
 *   - No recursion
 *   - All outputs via pointer arguments
 *
 * Target: Infineon AURIX TriCore TC3xx, ARM Cortex-M4
 *
 * \version   1.1.0
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

#ifndef COORDINATE_TRANSFORM_H_
#define COORDINATE_TRANSFORM_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_matrix.h"   /* MatrixFloat (= real32_T), Matrix_FloatToQ31,
                         Matrix_Q31ToFloat — also pulls in Sys_Types.h  */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup ct_types  Transform state structures
 * \{
 */

/**
 * \struct Clarke_T
 * \brief  State structure for the Clarke transform block.
 *
 * Purely combinatorial — the struct is retained for API symmetry with
 * stateful blocks and to allow future extension (e.g. a pre-filter).
 * The \c dummy field prevents a zero-size struct (MISRA C:2012 Rule 6.7.2).
 */
typedef struct
{
    MatrixFloat  dummy;  /**< Reserved — struct non-empty placeholder. */
} Clarke_T;

/**
 * \struct Park_T
 * \brief  State structure for the Park transform block.
 *
 * Purely combinatorial — same rationale as #Clarke_T.
 */
typedef struct
{
    MatrixFloat  dummy;  /**< Reserved — struct non-empty placeholder. */
} Park_T;

/**
 * \struct InvPark_T
 * \brief  State structure for the Inverse-Park transform block.
 *
 * Purely combinatorial — same rationale as #Clarke_T.
 */
typedef struct
{
    MatrixFloat  dummy;  /**< Reserved — struct non-empty placeholder. */
} InvPark_T;

/**
 * \struct InvClarke_T
 * \brief  State structure for the Inverse-Clarke transform block.
 *
 * Purely combinatorial — same rationale as #Clarke_T.
 */
typedef struct
{
    MatrixFloat  dummy;  /**< Reserved — struct non-empty placeholder. */
} InvClarke_T;

/** \} */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup ct_clarke  Clarke transform
 * \{
 */

/**
 * \brief  Initialise the Clarke transform block.
 *
 * \param[out] s  State struct to initialise (must not be NULL).
 */
extern void Clarke_Init(Clarke_T * const s);

/**
 * \brief  Apply the power-invariant Clarke transform.
 *
 * Formula:
 * \code
 *   i_α = (2/3)·i_a − (1/3)·i_b − (1/3)·i_c
 *   i_β = (1/√3)·(i_b − i_c)
 * \endcode
 *
 * \param[in,out] s         State struct (combinatorial — unused internally).
 * \param[in]     ia        Phase-A current [A].
 * \param[in]     ib        Phase-B current [A].
 * \param[in]     ic        Phase-C current [A].
 * \param[out]    alpha_out α-axis current i_α [A] (must not be NULL).
 * \param[out]    beta_out  β-axis current i_β [A] (must not be NULL).
 */
extern void Clarke_Step(
    Clarke_T       * const s,
    MatrixFloat            ia,
    MatrixFloat            ib,
    MatrixFloat            ic,
    MatrixFloat    * const alpha_out,
    MatrixFloat    * const beta_out);

/** \} */

/** \addtogroup ct_park  Park transform
 * \{
 */

/**
 * \brief  Initialise the Park transform block.
 *
 * \param[out] s  State struct to initialise (must not be NULL).
 */
extern void Park_Init(Park_T * const s);

/**
 * \brief  Apply the Park (forward) transform.
 *
 * Formula:
 * \code
 *   i_d =  i_α·cos(θ) + i_β·sin(θ)
 *   i_q = −i_α·sin(θ) + i_β·cos(θ)
 * \endcode
 *
 * \param[in,out] s       State struct (combinatorial — unused internally).
 * \param[in]     alpha   α-axis quantity i_α [A].
 * \param[in]     beta    β-axis quantity i_β [A].
 * \param[in]     theta   Electrical rotor angle θ_e [rad].
 * \param[out]    d_out   d-axis output i_d [A] (must not be NULL).
 * \param[out]    q_out   q-axis output i_q [A] (must not be NULL).
 */
extern void Park_Step(
    Park_T         * const s,
    MatrixFloat            alpha,
    MatrixFloat            beta,
    MatrixFloat            theta,
    MatrixFloat    * const d_out,
    MatrixFloat    * const q_out);

/** \} */

/** \addtogroup ct_invpark  Inverse-Park transform
 * \{
 */

/**
 * \brief  Initialise the Inverse-Park transform block.
 *
 * \param[out] s  State struct to initialise (must not be NULL).
 */
extern void InvPark_Init(InvPark_T * const s);

/**
 * \brief  Apply the Inverse-Park transform.
 *
 * Formula:
 * \code
 *   v_α = v_d·cos(θ) − v_q·sin(θ)
 *   v_β = v_d·sin(θ) + v_q·cos(θ)
 * \endcode
 *
 * \param[in,out] s         State struct (combinatorial — unused internally).
 * \param[in]     d         d-axis voltage v_d [V].
 * \param[in]     q         q-axis voltage v_q [V].
 * \param[in]     theta     Electrical rotor angle θ_e [rad].
 * \param[out]    alpha_out α-axis voltage v_α [V] (must not be NULL).
 * \param[out]    beta_out  β-axis voltage v_β [V] (must not be NULL).
 */
extern void InvPark_Step(
    InvPark_T      * const s,
    MatrixFloat            d,
    MatrixFloat            q,
    MatrixFloat            theta,
    MatrixFloat    * const alpha_out,
    MatrixFloat    * const beta_out);

/** \} */

/** \addtogroup ct_invclarke  Inverse-Clarke transform
 * \{
 */

/**
 * \brief  Initialise the Inverse-Clarke transform block.
 *
 * \param[out] s  State struct to initialise (must not be NULL).
 */
extern void InvClarke_Init(InvClarke_T * const s);

/**
 * \brief  Apply the Inverse-Clarke transform.
 *
 * Formula:
 * \code
 *   v_a =  v_α
 *   v_b = −(1/2)·v_α + (√3/2)·v_β
 *   v_c = −(1/2)·v_α − (√3/2)·v_β
 * \endcode
 *
 * \param[in,out] s       State struct (combinatorial — unused internally).
 * \param[in]     alpha   α-axis voltage v_α [V].
 * \param[in]     beta    β-axis voltage v_β [V].
 * \param[out]    va_out  Phase-A voltage v_a [V] (must not be NULL).
 * \param[out]    vb_out  Phase-B voltage v_b [V] (must not be NULL).
 * \param[out]    vc_out  Phase-C voltage v_c [V] (must not be NULL).
 */
extern void InvClarke_Step(
    InvClarke_T    * const s,
    MatrixFloat            alpha,
    MatrixFloat            beta,
    MatrixFloat    * const va_out,
    MatrixFloat    * const vb_out,
    MatrixFloat    * const vc_out);

/** \} */

#endif /* COORDINATE_TRANSFORM_H_ */
