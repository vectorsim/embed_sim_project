/*
 * coordinate_transform.h
 * ======================
 *
 * Clarke and Park coordinate transforms for PMSM FOC.
 * MATRIX OPTIMIZED VERSION - All transforms use matrix operations.
 *
 * TARGET
 * ------
 *   Primary  : Infineon Aurix TriCore  (TASKING ctc compiler)
 *   Secondary: ARM Cortex-M4           (GCC / LLVM)
 *   Simulation: Windows / Linux        (via Cython wrapper)
 *
 * TWO WORLDS
 * ----------
 *   [WORLD 1 — Python Simulation]
 *       Compiled into coordinate_transform_wrapper.pyd via Cython.
 *       real32_T = float — no difference at runtime.
 *
 *   [WORLD 2 — Aurix Embedded Target]
 *       Compiled directly into firmware by TASKING ctc.
 *       real32_T resolved from sys_types.h (Simulink ERT / AUTOSAR).
 *       cosf / sinf from TASKING libm — hardware FPU accelerated.
 *       No heap, no OS, no dynamic allocation.
 *
 * SIGNAL FLOW (FOC coordinate chain)
 * -----------
 *
 *   Phase3Signal        AlphaBetaSignal      DQSignal
 *   [a, b, c]     ──►  [alpha, beta]   ──►  [d, q]
 *   stationary          stationary           rotating
 *                 ◄──                  ◄──
 *             inv_clarke           inv_park + theta
 *
 * MATRIX REPRESENTATION
 * ---------------------
 *   All transforms are implemented as matrix multiplications:
 *
 *   Clarke:     [α] = [2/3  -1/3  -1/3] [A]
 *               [β]   [0    1/√3  -1/√3] [B]
 *                                         [C]
 *
 *   Park:       [d] = [ cosθ  sinθ] [α]
 *               [q]   [-sinθ  cosθ] [β]
 *
 * AURIX / TASKING NOTES
 * ---------------------
 *   - real32_T = float (32-bit IEEE 754) from sys_types.h.
 *     Matches TriCore FPU natively. Never use double in FOC inner loop.
 *   - All literals carry 'f' suffix to prevent implicit double promotion.
 *   - const pointer on inputs: MISRA-C:2012 Rule 8.13.
 *   - No global state — all functions pure and re-entrant.
 *   - Matrix multiplication functions are in Matrix_Operations.h
 *
 * @author EmbedSim Framework
 * @version 2.0.0
 * @date 2024
 */

#ifndef COORDINATE_TRANSFORM_H
#define COORDINATE_TRANSFORM_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Sys_Types.h"           /**< Simulink ERT / AUTOSAR standard types */
#include "Matrix_Operations.h"    /**< Matrix operations for all transforms */

/******************************************************************************/
/*-----------------------------Data Structures--------------------------------*/
/******************************************************************************/

/**
 * @brief Three-phase stationary reference frame signals
 *
 * Represents the three-phase currents or voltages in the stationary
 * reference frame (a, b, c coordinates).
 */
typedef struct
{
    real32_T A;  /**< Phase A component */
    real32_T B;  /**< Phase B component */
    real32_T C;  /**< Phase C component */
} Phase3Signal_T;

/**
 * @brief Two-phase stationary reference frame signals
 *
 * Represents the equivalent two-phase system in the stationary
 * α-β reference frame (Clarke transform output).
 */
typedef struct
{
    real32_T Alpha;  /**< Alpha axis component (real axis) */
    real32_T Beta;   /**< Beta axis component (imaginary axis) */
} AlphaBetaSignal_T;

/**
 * @brief Rotating reference frame signals
 *
 * Represents the direct and quadrature axis components in the
 * rotating d-q reference frame (Park transform output).
 */
typedef struct
{
    real32_T D;  /**< Direct axis component (flux-producing) */
    real32_T Q;  /**< Quadrature axis component (torque-producing) */
} DQSignal_T;

/**
 * @brief Clarke transform matrix (3x2)
 *
 * Pre-computed constant matrix for Clarke transform.
 * Stored as a complete type for optimization.
 */
typedef Matrix3x2_T ClarkeMatrix_T;

/**
 * @brief Park transform matrix (2x2 rotation)
 *
 * Updated each cycle with current rotor angle.
 */
typedef Matrix2x2_T ParkMatrix_T;

/******************************************************************************/
/*-------------------------Global Function Prototypes-------------------------*/
/******************************************************************************/

/**
 * @brief Initialize Clarke transform matrix (power-invariant form)
 *
 * Creates the matrix: [α; β] = [C] * [A; B; C]
 * where C = [2/3  -1/3  -1/3; 0  1/√3  -1/√3]
 *
 * @param[out] pMatrix  Clarke transform matrix (3x2)
 *
 * @note This matrix is constant and only needs initialization once.
 * @see Clarke_Transform
 */
extern void Clarke_InitMatrix(ClarkeMatrix_T* pMatrix);

/**
 * @brief Initialize inverse Clarke transform matrix
 *
 * Creates the matrix: [A; B; C] = [C]⁻¹ * [α; β]
 * Output is a 3x2 matrix (3 rows for A,B,C, 2 columns for α,β)
 *
 * @param[out] pMatrix  Inverse Clarke matrix (3x2)
 */
extern void InvClarke_InitMatrix(Matrix3x2_T* pMatrix);

/**
 * @brief Initialize Park rotation matrix for given angle
 *
 * Creates the matrix: [d; q] = [R(θ)] * [α; β]
 * where R(θ) = [cosθ  sinθ; -sinθ  cosθ]
 *
 * @param[out] pMatrix    Rotation matrix (2x2)
 * @param[in]  theta      Electrical angle in radians
 *
 * @note This should be called each control cycle with updated angle.
 * @see Park_Transform
 */
extern void Park_InitMatrix(ParkMatrix_T* pMatrix, real32_T theta);

/**
 * @brief Initialize inverse Park rotation matrix
 *
 * Creates the matrix: [α; β] = [R(θ)]⁻¹ * [d; q]
 * where R⁻¹(θ) = [cosθ  -sinθ; sinθ  cosθ]
 *
 * @param[out] pMatrix    Inverse rotation matrix (2x2)
 * @param[in]  theta      Electrical angle in radians
 */
extern void InvPark_InitMatrix(ParkMatrix_T* pMatrix, real32_T theta);

/**
 * @brief Perform Clarke transformation (3-phase to 2-phase stationary)
 *
 * Converts three-phase stationary reference frame signals (a, b, c)
 * to two-phase stationary reference frame signals (α, β) using matrix
 * multiplication: [α; β] = [C] * [A; B; C]
 *
 * @param[in]  pMatrix           Pointer to Clarke matrix (pre-initialized)
 * @param[in]  pPhase3SignalIn   Pointer to input three-phase signals
 * @param[out] pAlphaBetaSignalOut Pointer to output α-β signals
 *
 * @note This function is reentrant and uses no global state
 * @note All arithmetic is performed in single-precision float
 *
 * @code{.c}
 * ClarkeMatrix_T clarke_mat;
 * Clarke_InitMatrix(&clarke_mat);
 *
 * Phase3Signal_T input = {1.0f, 0.5f, -0.5f};
 * AlphaBetaSignal_T output;
 * Clarke_Transform(&clarke_mat, &input, &output);
 * @endcode
 */
extern void Clarke_Transform(const ClarkeMatrix_T* pMatrix,
                             const Phase3Signal_T* pPhase3SignalIn,
                             AlphaBetaSignal_T* pAlphaBetaSignalOut);

/**
 * @brief Perform inverse Clarke transform (2-phase to 3-phase)
 *
 * Converts two-phase stationary signals (α, β) back to three-phase
 * signals (a, b, c): [A; B; C] = [C]⁻¹ * [α; β]
 *
 * @param[in]  pMatrix           Pointer to inverse Clarke matrix (3x2)
 * @param[in]  pAlphaBetaSignalIn Pointer to input α-β signals
 * @param[out] pPhase3SignalOut  Pointer to output three-phase signals
 */
extern void InvClarke_Transform(const Matrix3x2_T* pMatrix,
                                const AlphaBetaSignal_T* pAlphaBetaSignalIn,
                                Phase3Signal_T* pPhase3SignalOut);

/**
 * @brief Perform Park transform (stationary to rotating frame)
 *
 * Converts stationary frame signals (α, β) to rotating frame
 * signals (d, q): [d; q] = [R(θ)] * [α; β]
 *
 * @param[in]  pMatrix           Pointer to Park rotation matrix
 * @param[in]  pAlphaBetaSignalIn Pointer to input α-β signals
 * @param[out] pDQSignalOut      Pointer to output d-q signals
 */
extern void Park_Transform(const ParkMatrix_T* pMatrix,
                           const AlphaBetaSignal_T* pAlphaBetaSignalIn,
                           DQSignal_T* pDQSignalOut);

/**
 * @brief Perform inverse Park transform (rotating to stationary frame)
 *
 * Converts rotating frame signals (d, q) back to stationary frame
 * signals (α, β): [α; β] = [R(θ)]⁻¹ * [d; q]
 *
 * @param[in]  pMatrix           Pointer to inverse Park matrix
 * @param[in]  pDQSignalIn       Pointer to input d-q signals
 * @param[out] pAlphaBetaSignalOut Pointer to output α-β signals
 */
extern void InvPark_Transform(const ParkMatrix_T* pMatrix,
                              const DQSignal_T* pDQSignalIn,
                              AlphaBetaSignal_T* pAlphaBetaSignalOut);

/**
 * @brief Combined Clarke+Park transform in one step
 *
 * Optimized transform from three-phase directly to d-q:
 * [d; q] = [R(θ)] * [C] * [A; B; C]
 *
 * @param[in]  pParkMatrix      Pointer to Park rotation matrix
 * @param[in]  pClarkeMatrix    Pointer to Clarke matrix
 * @param[in]  pPhase3SignalIn  Pointer to input three-phase signals
 * @param[out] pDQSignalOut     Pointer to output d-q signals
 *
 * @note Useful for FOC implementations to reduce operations
 */
extern void ClarkePark_Transform(const ParkMatrix_T* pParkMatrix,
                                 const ClarkeMatrix_T* pClarkeMatrix,
                                 const Phase3Signal_T* pPhase3SignalIn,
                                 DQSignal_T* pDQSignalOut);

#endif /* COORDINATE_TRANSFORM_H */