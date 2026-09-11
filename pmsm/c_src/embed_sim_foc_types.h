/**********************************************************************************************************************
 * \file      embed_sim_foc_types.h
 * \brief     Shared domain-signal structures for the EmbedSim FOC pipeline.
 *
 * \details   Defines the four frame-signal structures used across all FOC blocks:
 *
 *              FocUvw_T        — three-phase natural frame   (U, V, W)
 *              FocAlphaBeta_T  — stationary αβ frame         (Alpha, Beta)
 *              FocDq_T         — rotating dq frame           (D, Q)
 *              FocAngle_T      — electrical rotor angle      (ThetaE)
 *
 *            Signal flow through the FOC pipeline:
 * \code
 *   FocUvw_T (currents)
 *       │
 *       ▼  Clarke_Step
 *   FocAlphaBeta_T
 *       │
 *       ▼  Park_Step  ◄── FocAngle_T
 *   FocDq_T
 *       │  (PI controllers)
 *       ▼  InvPark_Step  ◄── FocAngle_T
 *   FocAlphaBeta_T
 *       │
 *       ▼  InvClarke_Step
 *   FocUvw_T (duties)
 *       │
 *       ▼  SVM_CalculateDutyCycle
 *   SVM_DutyCycle_T
 * \endcode
 *
 *            All structures carry MatrixFloat (= real32_T) members.
 *            Physical units depend on context:
 *              - Current path  : [A]
 *              - Voltage path  : [V]
 *              - Duty-cycle path: [0.0 – 1.0, dimensionless]
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule 8.5  : One declaration per identifier
 *              - Rule 8.6  : No definitions in header files
 *              - No dynamic memory allocation
 *
 * \note      EmbedSim naming convention:
 *              - Typedefs      : Pascal_Snake_Case_T
 *              - Struct members: PascalCase  (Alpha, Beta, ThetaE)
 *              - Macros        : UPPER_SNAKE_CASE
 *
 * \version   1.0.0
 * \date      2025-05-24
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_FOC_TYPES_H_
#define EMBED_SIM_FOC_TYPES_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_matrix.h"   /* MatrixFloat (= real32_T) — also pulls embed_sim_sys_types.h */

/**********************************************************************************************************************
 * Domain-Signal Structures
 *********************************************************************************************************************/

/**
 * \struct FocUvw_T
 * \brief  Three-phase natural-frame signal.
 *
 * \details Used for phase currents at the Clarke input and phase duty cycles
 *          at the Inverse-Clarke output.  Balanced load assumed: W = −U − V.
 *
 *          Units depend on context:
 *            - Phase currents : [A]
 *            - Phase voltages : [V]
 *            - Duty cycles    : [0.0 – 1.0, dimensionless]
 */
typedef struct
{
    MatrixFloat  U;   /**< Phase U signal   [context-dependent] */
    MatrixFloat  V;   /**< Phase V signal   [context-dependent] */
    MatrixFloat  W;   /**< Phase W signal   [context-dependent] */
} FocUvw_T;

/**
 * \struct FocAlphaBeta_T
 * \brief  Stationary αβ-frame signal (Clarke / Inverse-Clarke boundary).
 *
 * \details Output of Clarke_Step; input to Park_Step.
 *          Output of InvPark_Step; input to InvClarke_Step.
 *
 *          Units depend on context:
 *            - αβ currents : [A]
 *            - αβ voltages : [V]
 */
typedef struct
{
    MatrixFloat  Alpha;   /**< α-axis signal   [context-dependent] */
    MatrixFloat  Beta;    /**< β-axis signal   [context-dependent] */
} FocAlphaBeta_T;

/**
 * \struct FocDq_T
 * \brief  Rotating dq-frame signal (Park / Inverse-Park boundary).
 *
 * \details Output of Park_Step; input to dq PI controllers and InvPark_Step.
 *
 *          Units depend on context:
 *            - dq currents : [A]
 *            - dq voltages : [V]
 */
typedef struct
{
    MatrixFloat  D;   /**< d-axis signal   [context-dependent] */
    MatrixFloat  Q;   /**< q-axis signal   [context-dependent] */
} FocDq_T;

/**
 * \struct FocAngle_T
 * \brief  Electrical rotor angle passed to Park and Inverse-Park transforms.
 *
 * \details Supplied by the position estimator (SMO, encoder, resolver).
 *          Wrapped to [0, 2π) by the caller before each step.
 */
typedef struct
{
    MatrixFloat  ThetaE;   /**< Electrical rotor angle   [rad, 0 – 2π) */
} FocAngle_T;

#endif /* EMBED_SIM_FOC_TYPES_H_ */
