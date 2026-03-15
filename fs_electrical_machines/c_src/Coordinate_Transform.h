/*
 * Coordinate_Transform.h
 * ======================
 * Clarke / Park / InvPark / InvClarke transforms for FOC motor control.
 *
 * Uses MatrixFloat (= real32_T) from Matrix.h as the working type.
 * Matrix.h already includes Sys_Types.h — no duplicate include needed.
 *
 * MISRA C:2012 compliant:
 *   - No dynamic memory allocation
 *   - No recursion
 *   - All outputs via pointer arguments
 *   - Single include guard
 *
 * Signal conventions
 * ------------------
 *   Clarke    : [i_a, i_b, i_c]        -> [i_alpha, i_beta]   (power-invariant)
 *   Park      : [i_alpha, i_beta, θ_e] -> [i_d, i_q]
 *   InvPark   : [v_d, v_q, θ_e]        -> [v_alpha, v_beta]
 *   InvClarke : [v_alpha, v_beta]       -> [v_a, v_b, v_c]
 *
 * Target : Infineon AURIX TriCore, ARM Cortex-M4
 * Author : EmbedSim / fs_electrical_machines
 * Version: 1.1.0
 */

#ifndef COORDINATE_TRANSFORM_H
#define COORDINATE_TRANSFORM_H

#include "Matrix.h"   /* MatrixFloat (=real32_T), Matrix_FloatToQ31,
                         Matrix_Q31ToFloat — also pulls in Sys_Types.h */

/* ============================================================================
 * Clarke transform state
 * ==========================================================================*/

/**
 * @struct Clarke_T
 * @brief  State structure for the Clarke transform block.
 *
 * Purely combinatorial — struct kept for API symmetry and future extension.
 */
typedef struct {
    MatrixFloat dummy; /**< Reserved — keeps struct non-empty (MISRA 6.7.2) */
} Clarke_T;

extern void Clarke_Init(Clarke_T * const pS);

/**
 * Power-invariant Clarke:
 *   i_alpha = (2/3)*ia - (1/3)*ib - (1/3)*ic
 *   i_beta  = (1/sqrt(3))*(ib - ic)
 */
extern void Clarke_Step(Clarke_T * const pS,
                        MatrixFloat ia, MatrixFloat ib, MatrixFloat ic,
                        MatrixFloat * const alpha_out,
                        MatrixFloat * const beta_out);

/* ============================================================================
 * Park transform state
 * ==========================================================================*/

typedef struct {
    MatrixFloat dummy;
} Park_T;

extern void Park_Init(Park_T * const pS);

/**
 * Park:
 *   i_d =  i_alpha * cos(theta) + i_beta * sin(theta)
 *   i_q = -i_alpha * sin(theta) + i_beta * cos(theta)
 */
extern void Park_Step(Park_T * const pS,
                      MatrixFloat alpha, MatrixFloat beta, MatrixFloat theta,
                      MatrixFloat * const d_out,
                      MatrixFloat * const q_out);

/* ============================================================================
 * Inverse Park transform state
 * ==========================================================================*/

typedef struct {
    MatrixFloat dummy;
} InvPark_T;

extern void InvPark_Init(InvPark_T * const pS);

/**
 * Inverse Park:
 *   v_alpha = v_d * cos(theta) - v_q * sin(theta)
 *   v_beta  = v_d * sin(theta) + v_q * cos(theta)
 */
extern void InvPark_Step(InvPark_T * const pS,
                         MatrixFloat d, MatrixFloat q, MatrixFloat theta,
                         MatrixFloat * const alpha_out,
                         MatrixFloat * const beta_out);

/* ============================================================================
 * Inverse Clarke transform state
 * ==========================================================================*/

typedef struct {
    MatrixFloat dummy;
} InvClarke_T;

extern void InvClarke_Init(InvClarke_T * const pS);

/**
 * Inverse Clarke:
 *   v_a =  v_alpha
 *   v_b = -(1/2)*v_alpha + (sqrt(3)/2)*v_beta
 *   v_c = -(1/2)*v_alpha - (sqrt(3)/2)*v_beta
 */
extern void InvClarke_Step(InvClarke_T * const pS,
                            MatrixFloat alpha, MatrixFloat beta,
                            MatrixFloat * const va_out,
                            MatrixFloat * const vb_out,
                            MatrixFloat * const vc_out);

#endif /* COORDINATE_TRANSFORM_H */
