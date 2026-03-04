/*
 * Matrix_Operations.h
 * ===================
 *
 * Pure matrix operations for PMSM control systems.
 * MISRA C:2012 compliant - No memcpy, explicit element-wise operations.
 * ASIL-D ready.
 *
 * TARGET
 * ------
 *   Primary  : Infineon Aurix TriCore (TASKING ctc)
 *   Secondary: ARM Cortex-M4 (GCC/LLVM)
 *   Simulation: Windows/Linux (via Cython)
 *
 * @author EmbedSim Framework
 * @version 2.0.0
 * @date 2024
 */

#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Sys_Types.h"

/******************************************************************************/
/*------------------------------Matrix Types----------------------------------*/
/******************************************************************************/

/**
 * @brief Fixed-size matrix types for PMSM control
 * All matrices are row-major: M[row][col]
 * MISRA C:2012 Rule 18.4 - No unions or variant records
 */

/* Square matrices */
typedef struct {
    real32_T M[2][2];
} Matrix2x2_T;

typedef struct {
    real32_T M[3][3];
} Matrix3x3_T;

typedef struct {
    real32_T M[4][4];
} Matrix4x4_T;

/* Rectangular matrices for transforms */
typedef struct {
    real32_T M[3][2];  /**< 3 rows, 2 columns (Clarke transform) */
} Matrix3x2_T;

typedef struct {
    real32_T M[2][3];  /**< 2 rows, 3 columns (Inverse Clarke) */
} Matrix2x3_T;

typedef struct {
    real32_T M[4][3];  /**< 4 rows, 3 columns (State space B matrix) */
} Matrix4x3_T;

typedef struct {
    real32_T M[3][4];  /**< 3 rows, 4 columns (Output C matrix) */
} Matrix3x4_T;

/* Vector types */
typedef struct {
    real32_T V[2];
} Vector2_T;

typedef struct {
    real32_T V[3];
} Vector3_T;

typedef struct {
    real32_T V[4];
} Vector4_T;

/******************************************************************************/
/*-------------------------Matrix Access Macros-------------------------------*/
/******************************************************************************/

/* MISRA C:2012 Rule 20.10 - Parentheses around macro parameters */
/** @brief Get matrix element at (row, col) */
#define MAT_ELEM(pMat, row, col) ((pMat)->M[(row)][(col)])

/** @brief Get vector element at index */
#define VEC_ELEM(pVec, idx) ((pVec)->V[(idx)])

/** @brief Set matrix element */
#define MAT_SET(pMat, row, col, val) ((pMat)->M[(row)][(col)] = (val))

/** @brief Set vector element */
#define VEC_SET(pVec, idx, val) ((pVec)->V[(idx)] = (val))

/******************************************************************************/
/*-------------------------Initialization Functions---------------------------*/
/******************************************************************************/

/**
 * @brief Set all matrix elements to zero
 * @param pC Pointer to destination matrix
 * MISRA C:2012 Rule 17.7 - Function parameters used
 */
extern void Matrix_Zero_2x2(Matrix2x2_T* const pC);
extern void Matrix_Zero_3x3(Matrix3x3_T* const pC);
extern void Matrix_Zero_4x4(Matrix4x4_T* const pC);
extern void Matrix_Zero_3x2(Matrix3x2_T* const pC);
extern void Matrix_Zero_2x3(Matrix2x3_T* const pC);
extern void Matrix_Zero_4x3(Matrix4x3_T* const pC);
extern void Matrix_Zero_3x4(Matrix3x4_T* const pC);
extern void Matrix_Zero_Vector2(Vector2_T* const pC);
extern void Matrix_Zero_Vector3(Vector3_T* const pC);
extern void Matrix_Zero_Vector4(Vector4_T* const pC);

/**
 * @brief Set matrix to identity (square matrices only)
 * @param pC Pointer to destination matrix
 */
extern void Matrix_Identity_2x2(Matrix2x2_T* const pC);
extern void Matrix_Identity_3x3(Matrix3x3_T* const pC);
extern void Matrix_Identity_4x4(Matrix4x4_T* const pC);

/**
 * @brief Copy matrix from source to destination
 * @param pA Pointer to source matrix
 * @param pC Pointer to destination matrix
 * MISRA C:2012 Rule 21.15 - No memcpy, explicit loops
 */
extern void Matrix_Copy_2x2(const Matrix2x2_T* const pA, Matrix2x2_T* const pC);
extern void Matrix_Copy_3x3(const Matrix3x3_T* const pA, Matrix3x3_T* const pC);
extern void Matrix_Copy_4x4(const Matrix4x4_T* const pA, Matrix4x4_T* const pC);
extern void Matrix_Copy_3x2(const Matrix3x2_T* const pA, Matrix3x2_T* const pC);
extern void Matrix_Copy_2x3(const Matrix2x3_T* const pA, Matrix2x3_T* const pC);
extern void Matrix_Copy_4x3(const Matrix4x3_T* const pA, Matrix4x3_T* const pC);
extern void Matrix_Copy_3x4(const Matrix3x4_T* const pA, Matrix3x4_T* const pC);
extern void Matrix_Copy_Vector2(const Vector2_T* const pA, Vector2_T* const pC);
extern void Matrix_Copy_Vector3(const Vector3_T* const pA, Vector3_T* const pC);
extern void Matrix_Copy_Vector4(const Vector4_T* const pA, Vector4_T* const pC);

/******************************************************************************/
/*-------------------------Basic Matrix Operations----------------------------*/
/******************************************************************************/

/**
 * @brief Add two matrices: pC = pA + pB
 * @param pA First input matrix
 * @param pB Second input matrix
 * @param pC Destination matrix
 */
extern void Matrix_Add_2x2(const Matrix2x2_T* const pA,
                           const Matrix2x2_T* const pB,
                           Matrix2x2_T* const pC);
extern void Matrix_Add_3x3(const Matrix3x3_T* const pA,
                           const Matrix3x3_T* const pB,
                           Matrix3x3_T* const pC);
extern void Matrix_Add_4x4(const Matrix4x4_T* const pA,
                           const Matrix4x4_T* const pB,
                           Matrix4x4_T* const pC);
extern void Matrix_Add_Vector2(const Vector2_T* const pA,
                               const Vector2_T* const pB,
                               Vector2_T* const pC);
extern void Matrix_Add_Vector3(const Vector3_T* const pA,
                               const Vector3_T* const pB,
                               Vector3_T* const pC);
extern void Matrix_Add_Vector4(const Vector4_T* const pA,
                               const Vector4_T* const pB,
                               Vector4_T* const pC);

/**
 * @brief Subtract two matrices: pC = pA - pB
 * @param pA First input matrix
 * @param pB Second input matrix
 * @param pC Destination matrix
 */
extern void Matrix_Sub_2x2(const Matrix2x2_T* const pA,
                           const Matrix2x2_T* const pB,
                           Matrix2x2_T* const pC);
extern void Matrix_Sub_3x3(const Matrix3x3_T* const pA,
                           const Matrix3x3_T* const pB,
                           Matrix3x3_T* const pC);
extern void Matrix_Sub_4x4(const Matrix4x4_T* const pA,
                           const Matrix4x4_T* const pB,
                           Matrix4x4_T* const pC);
extern void Matrix_Sub_Vector2(const Vector2_T* const pA,
                               const Vector2_T* const pB,
                               Vector2_T* const pC);
extern void Matrix_Sub_Vector3(const Vector3_T* const pA,
                               const Vector3_T* const pB,
                               Vector3_T* const pC);
extern void Matrix_Sub_Vector4(const Vector4_T* const pA,
                               const Vector4_T* const pB,
                               Vector4_T* const pC);

/**
 * @brief Scale matrix by scalar: pC = Delta * pA
 * @param pA Input matrix
 * @param Delta Scaling factor
 * @param pC Destination matrix
 */
extern void Matrix_Scale_2x2(const Matrix2x2_T* const pA,
                             const real32_T Delta,
                             Matrix2x2_T* const pC);
extern void Matrix_Scale_3x3(const Matrix3x3_T* const pA,
                             const real32_T Delta,
                             Matrix3x3_T* const pC);
extern void Matrix_Scale_4x4(const Matrix4x4_T* const pA,
                             const real32_T Delta,
                             Matrix4x4_T* const pC);
extern void Matrix_Scale_3x2(const Matrix3x2_T* const pA,
                             const real32_T Delta,
                             Matrix3x2_T* const pC);
extern void Matrix_Scale_2x3(const Matrix2x3_T* const pA,
                             const real32_T Delta,
                             Matrix2x3_T* const pC);
extern void Matrix_Scale_4x3(const Matrix4x3_T* const pA,
                             const real32_T Delta,
                             Matrix4x3_T* const pC);
extern void Matrix_Scale_3x4(const Matrix3x4_T* const pA,
                             const real32_T Delta,
                             Matrix3x4_T* const pC);
extern void Matrix_Scale_Vector2(const Vector2_T* const pA,
                                 const real32_T Delta,
                                 Vector2_T* const pC);
extern void Matrix_Scale_Vector3(const Vector3_T* const pA,
                                 const real32_T Delta,
                                 Vector3_T* const pC);
extern void Matrix_Scale_Vector4(const Vector4_T* const pA,
                                 const real32_T Delta,
                                 Vector4_T* const pC);

/******************************************************************************/
/*-------------------------Matrix Transposition-------------------------------*/
/******************************************************************************/

/**
 * @brief Transpose square matrix: pC = pA^T
 * @param pA Input matrix
 * @param pC Destination matrix (can be same as pA for in-place)
 */
extern void Matrix_Transpose_2x2(const Matrix2x2_T* const pA, Matrix2x2_T* const pC);
extern void Matrix_Transpose_3x3(const Matrix3x3_T* const pA, Matrix3x3_T* const pC);
extern void Matrix_Transpose_4x4(const Matrix4x4_T* const pA, Matrix4x4_T* const pC);

/**
 * @brief Transpose rectangular matrices
 * @param pA Input matrix
 * @param pC Destination matrix (must be different type)
 */
extern void Matrix_Transpose_3x2_to_2x3(const Matrix3x2_T* const pA, Matrix2x3_T* const pC);
extern void Matrix_Transpose_2x3_to_3x2(const Matrix2x3_T* const pA, Matrix3x2_T* const pC);
extern void Matrix_Transpose_4x3_to_3x4(const Matrix4x3_T* const pA, Matrix3x4_T* const pC);
extern void Matrix_Transpose_3x4_to_4x3(const Matrix3x4_T* const pA, Matrix4x3_T* const pC);

/******************************************************************************/
/*-------------------------Matrix Multiplication------------------------------*/
/******************************************************************************/

/**
 * @brief Matrix multiplication: pC = pA * pB
 * @param pA First input matrix (left side)
 * @param pB Second input matrix (right side)
 * @param pC Destination matrix
 * MISRA C:2012 Rule 13.6 - No side effects in conditionals
 */

/* 2x2 operations */
extern void Matrix_Mul_2x2_2x2(const Matrix2x2_T* const pA,
                               const Matrix2x2_T* const pB,
                               Matrix2x2_T* const pC);
extern void Matrix_Mul_2x2_Vec2(const Matrix2x2_T* const pA,
                                const Vector2_T* const pB,
                                Vector2_T* const pC);

/* 3x3 operations */
extern void Matrix_Mul_3x3_3x3(const Matrix3x3_T* const pA,
                               const Matrix3x3_T* const pB,
                               Matrix3x3_T* const pC);
extern void Matrix_Mul_3x3_Vec3(const Matrix3x3_T* const pA,
                                const Vector3_T* const pB,
                                Vector3_T* const pC);

/* 4x4 operations */
extern void Matrix_Mul_4x4_4x4(const Matrix4x4_T* const pA,
                               const Matrix4x4_T* const pB,
                               Matrix4x4_T* const pC);
extern void Matrix_Mul_4x4_Vec4(const Matrix4x4_T* const pA,
                                const Vector4_T* const pB,
                                Vector4_T* const pC);

/* Mixed dimension multiplications */
extern void Matrix_Mul_4x4_4x3(const Matrix4x4_T* const pA,
                               const Matrix4x3_T* const pB,
                               Matrix4x3_T* const pC);
extern void Matrix_Mul_3x4_4x3(const Matrix3x4_T* const pA,
                               const Matrix4x3_T* const pB,
                               Matrix3x3_T* const pC);
extern void Matrix_Mul_4x3_3x4(const Matrix4x3_T* const pA,
                               const Matrix3x4_T* const pB,
                               Matrix4x4_T* const pC);
extern void Matrix_Mul_4x3_3x3(const Matrix4x3_T* const pA,
                               const Matrix3x3_T* const pB,
                               Matrix4x3_T* const pC);
extern void Matrix_Mul_3x4_4x4(const Matrix3x4_T* const pA,
                               const Matrix4x4_T* const pB,
                               Matrix3x4_T* const pC);

/* Transform-specific multiplications */
extern void Matrix_Mul_3x2_Vec2(const Matrix3x2_T* const pA,
                                const Vector2_T* const pB,
                                Vector3_T* const pC);
extern void Matrix_Mul_2x3_Vec3(const Matrix2x3_T* const pA,
                                const Vector3_T* const pB,
                                Vector2_T* const pC);

/******************************************************************************/
/*-------------------------Matrix Inversion-----------------------------------*/
/******************************************************************************/

/**
 * @brief Invert 2x2 matrix: pC = pA^(-1)
 * @param pA Input matrix
 * @param pC Destination matrix
 * @return 1U if successful, 0U if singular
 * MISRA C:2012 Rule 16.7 - Single point of exit
 */
extern uint8_T Matrix_Invert_2x2(const Matrix2x2_T* const pA, Matrix2x2_T* const pC);

/**
 * @brief Invert 3x3 matrix: pC = pA^(-1)
 * @param pA Input matrix
 * @param pC Destination matrix
 * @return 1U if successful, 0U if singular
 */
extern uint8_T Matrix_Invert_3x3(const Matrix3x3_T* const pA, Matrix3x3_T* const pC);

/**
 * @brief Invert 4x4 matrix: pC = pA^(-1)
 * @param pA Input matrix
 * @param pC Destination matrix
 * @return 1U if successful, 0U if singular
 */
extern uint8_T Matrix_Invert_4x4(const Matrix4x4_T* const pA, Matrix4x4_T* const pC);

/******************************************************************************/
/*-------------------------Determinant Calculations---------------------------*/
/******************************************************************************/

/**
 * @brief Calculate determinant of matrix
 * @param pA Input matrix
 * @return Determinant value
 */
extern real32_T Matrix_Det_2x2(const Matrix2x2_T* const pA);
extern real32_T Matrix_Det_3x3(const Matrix3x3_T* const pA);
extern real32_T Matrix_Det_4x4(const Matrix4x4_T* const pA);

/******************************************************************************/
/*-------------------------PMSM Specific Matrices-----------------------------*/
/******************************************************************************/

/**
 * @brief Initialize Clarke transform matrix (3-phase to 2-phase)
 * @param pC Destination matrix
 * MISRA C:2012 Rule 10.8 - Casts not performed
 */
extern void Matrix_Clarke_Init(Matrix3x2_T* const pC);

/**
 * @brief Initialize inverse Clarke matrix
 * @param pC Destination matrix
 */
extern void Matrix_InvClarke_Init(Matrix3x2_T* const pC);

/**
 * @brief Initialize Park transform matrix for given angle
 * @param pC Destination matrix
 * @param Theta Electrical angle in radians
 */
extern void Matrix_Park_Init(Matrix2x2_T* const pC, const real32_T Theta);

/**
 * @brief Initialize inverse Park matrix
 * @param pC Destination matrix
 * @param Theta Electrical angle in radians
 */
extern void Matrix_InvPark_Init(Matrix2x2_T* const pC, const real32_T Theta);

#endif /* MATRIX_OPERATIONS_H */