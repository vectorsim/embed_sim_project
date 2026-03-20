/**********************************************************************************************************************
 * \file      Matrix.h
 * \brief     32-bit fixed-point (Q31) linear algebra library for embedded systems.
 *
 * Provides static-allocation matrix operations targeting 32-bit MCUs
 * (Infineon AURIX TriCore, ARM Cortex-M4).  All algorithms are iterative —
 * no recursion — and the implementation is MISRA C:2012 / AUTOSAR C compliant.
 *
 * Key properties:
 *   - Q31 fixed-point arithmetic (1 sign bit + 31 fractional bits)
 *   - Maximum matrix size: 8 × 8
 *   - No dynamic memory allocation
 *   - Supports eigenvalue decomposition (iterative Jacobi method)
 *
 * \version   5.0.0
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

#ifndef MATRIX_H_
#define MATRIX_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Sys_Types.h"


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup matrix_limits  Dimension limits
 * \{
 */
/** \brief Maximum number of rows (and eigenvalues) supported. */
#define MATRIX_MAX_ROWS   (8U)

/** \brief Maximum number of columns supported. */
#define MATRIX_MAX_COLS   (8U)

/** \brief Maximum number of eigenvalues that can be stored. */
#define MATRIX_MAX_EIGEN  (8U)
/** \} */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup matrix_types  Types, enumerations and structures
 * \{
 */

/**
 * \typedef MatrixElement
 * \brief   32-bit Q31 fixed-point scalar type.
 *
 * Encoding: 1 sign bit + 31 fractional bits.
 * Range    : [−1.0, +0.9999999995]
 * Resolution: ~4.66 × 10⁻¹⁰
 */
typedef int32_T MatrixElement;

/**
 * \typedef MatrixFloat
 * \brief   32-bit floating-point type used for intermediate calculations.
 *
 * Maps to \c real32_T from \c Sys_Types.h.
 */
typedef real32_T MatrixFloat;

/**
 * \enum  MatrixStatus_Type
 * \brief Status codes returned by all matrix operations.
 */
typedef enum
{
    MATRIX_SUCCESS                   =  0,  /**< Operation completed successfully.            */
    MATRIX_ERROR_NULL_PTR            =  1,  /**< NULL pointer supplied.                       */
    MATRIX_ERROR_DIMENSION_MISMATCH  =  2,  /**< Incompatible matrix dimensions.              */
    MATRIX_ERROR_SINGULAR            =  3,  /**< Matrix is singular (determinant ≈ 0).        */
    MATRIX_ERROR_SIZE_EXCEEDED       =  4,  /**< Matrix exceeds maximum supported dimensions. */
    MATRIX_ERROR_DIV_BY_ZERO         =  5,  /**< Division by zero attempted.                  */
    MATRIX_ERROR_NOT_SQUARE          =  6,  /**< Operation requires a square matrix.          */
    MATRIX_ERROR_BUFFER_OVERFLOW     =  7,  /**< Destination buffer is insufficient.          */
    MATRIX_ERROR_OUT_OF_BOUNDS       =  8,  /**< Index out of valid range.                    */
    MATRIX_ERROR_NON_POSITIVE_DEFINITE =  9, /**< Matrix is not positive-definite.            */
    MATRIX_ERROR_NOT_INVERTIBLE      = 10,  /**< Matrix cannot be inverted.                   */
    MATRIX_ERROR_NOT_CONVERGENT      = 11,  /**< Iterative algorithm did not converge.        */
    MATRIX_ERROR_MAX_ITERATIONS      = 12   /**< Maximum iteration count reached.             */
} MatrixStatus_Type;

/**
 * \struct Matrix_Type
 * \brief  Matrix handle for static-allocation use.
 *
 * All storage is provided by the caller via a fixed-size buffer; no heap
 * allocation occurs.  The \c stride field enables sub-matrix views without
 * copying.
 */
typedef struct
{
    MatrixElement  * data;      /**< Pointer to the Q31 data buffer (row-major).       */
    uint32_T         rows;      /**< Active row count.                                 */
    uint32_T         cols;      /**< Active column count.                              */
    uint32_T         max_rows;  /**< Allocated row capacity.                           */
    uint32_T         max_cols;  /**< Allocated column capacity.                        */
    uint32_T         is_view;   /**< TRUE if this handle is a view of another matrix.  */
    uint32_T         stride;    /**< Row stride in elements (equals max_cols normally).*/
} Matrix_Type;

/**
 * \struct MatrixEigen_Type
 * \brief  Result container for eigenvalue / eigenvector decomposition.
 */
typedef struct
{
    MatrixFloat  eigenvalues[MATRIX_MAX_EIGEN];                   /**< Real eigenvalues (diagonal of diagonalised matrix). */
    MatrixFloat  eigenvectors[MATRIX_MAX_ROWS * MATRIX_MAX_COLS]; /**< Eigenvectors stored column-wise.                    */
    uint32_T     num_eigenvalues;                                  /**< Number of eigenvalues computed.                     */
    uint32_T     iterations;                                       /**< Jacobi sweeps consumed.                             */
} MatrixEigen_Type;

/** \} */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Initialisation
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_init  Initialisation
 * \{
 */

/**
 * \brief  Initialise a matrix handle with a caller-supplied static buffer.
 *
 * Sets \c rows and \c cols to \p max_rows and \p max_cols respectively and
 * zero-fills the entire buffer via \c memset.
 *
 * \param[out] matrix    Matrix handle to initialise (must not be NULL).
 * \param[in]  buffer    Caller-allocated buffer of size
 *                       \p max_rows × \p max_cols × sizeof(#MatrixElement).
 * \param[in]  max_rows  Row capacity (1 … #MATRIX_MAX_ROWS).
 * \param[in]  max_cols  Column capacity (1 … #MATRIX_MAX_COLS).
 */
extern void Matrix_Init(
    Matrix_Type    * const matrix,
    MatrixElement  * const buffer,
    const uint32_T         max_rows,
    const uint32_T         max_cols);

/**
 * \brief  Set the active dimensions of a matrix without touching data.
 *
 * \param[in,out] matrix  Matrix handle (must not be NULL).
 * \param[in]     rows    Active rows    (1 … \c matrix->max_rows).
 * \param[in]     cols    Active columns (1 … \c matrix->max_cols).
 */
extern void Matrix_SetDimensions(
    Matrix_Type  * const matrix,
    const uint32_T       rows,
    const uint32_T       cols);

/**
 * \brief  Zero all elements within the active dimensions.
 *
 * \param[in,out] matrix  Matrix handle (must not be NULL).
 */
extern void Matrix_Zero(Matrix_Type * const matrix);

/**
 * \brief  Set a square matrix to the identity (I).
 *
 * \param[in,out] matrix  Square matrix handle (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE.
 */
extern MatrixStatus_Type Matrix_Identity(Matrix_Type * const matrix);

/**
 * \brief  Copy \p src into \p dest.
 *
 * \p dest must have sufficient capacity to hold all elements of \p src.
 *
 * \param[out] dest  Destination matrix (must not be NULL).
 * \param[in]  src   Source matrix (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_Copy(
    Matrix_Type        * const dest,
    const Matrix_Type  * const src);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Element access
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_element_access  Element access
 * \{
 */

/**
 * \brief  Write a Q31 value to a single element.
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     row     0-based row index (< \c matrix->rows).
 * \param[in]     col     0-based column index (< \c matrix->cols).
 * \param[in]     value   Q31 value to write.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_Type Matrix_SetElement(
    Matrix_Type    * const matrix,
    const uint32_T         row,
    const uint32_T         col,
    const MatrixElement    value);

/**
 * \brief  Write a float value (auto-converted to Q31) to a single element.
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     row     0-based row index.
 * \param[in]     col     0-based column index.
 * \param[in]     value   Float in [−1.0, 1.0] (clamped).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_Type Matrix_SetElementFloat(
    Matrix_Type    * const matrix,
    const uint32_T         row,
    const uint32_T         col,
    const MatrixFloat      value);

/**
 * \brief  Read a Q31 value from a single element.
 *
 * \param[in]  matrix  Source matrix (must not be NULL).
 * \param[in]  row     0-based row index.
 * \param[in]  col     0-based column index.
 * \param[out] value   Receives the Q31 value (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_Type Matrix_GetElement(
    const Matrix_Type  * const matrix,
    const uint32_T             row,
    const uint32_T             col,
    MatrixElement      * const value);

/**
 * \brief  Read a single element and return it as a float.
 *
 * \param[in]  matrix  Source matrix (must not be NULL).
 * \param[in]  row     0-based row index.
 * \param[in]  col     0-based column index.
 * \param[out] value   Receives the float value (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_Type Matrix_GetElementFloat(
    const Matrix_Type  * const matrix,
    const uint32_T             row,
    const uint32_T             col,
    MatrixFloat        * const value);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Basic arithmetic operations
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_basic_ops  Basic arithmetic
 * \{
 */

/**
 * \brief  Element-wise addition: \p result = \p a + \p b (saturating Q31).
 *
 * \param[in]  a       Left operand.
 * \param[in]  b       Right operand (same dimensions as \p a).
 * \param[out] result  Output matrix (buffer must be ≥ \p a dimensions).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_Add(
    const Matrix_Type  * const a,
    const Matrix_Type  * const b,
    Matrix_Type        * const result);

/**
 * \brief  Element-wise subtraction: \p result = \p a − \p b (saturating Q31).
 *
 * \param[in]  a       Minuend.
 * \param[in]  b       Subtrahend (same dimensions as \p a).
 * \param[out] result  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_Subtract(
    const Matrix_Type  * const a,
    const Matrix_Type  * const b,
    Matrix_Type        * const result);

/**
 * \brief  Matrix multiplication: \p result = \p a × \p b (Q31, saturating).
 *
 * \p a is (m × n), \p b is (n × p), \p result is (m × p).
 *
 * \param[in]  a       Left factor  (m × n).
 * \param[in]  b       Right factor (n × p).
 * \param[out] result  Product matrix (buffer must be ≥ m × p).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_Multiply(
    const Matrix_Type  * const a,
    const Matrix_Type  * const b,
    Matrix_Type        * const result);

/**
 * \brief  Scalar multiplication: \p result = \p scalar × \p matrix (Q31).
 *
 * \param[in]  matrix  Input matrix.
 * \param[in]  scalar  Q31 scalar.
 * \param[out] result  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_ScalarMultiply(
    const Matrix_Type  * const matrix,
    const MatrixElement        scalar,
    Matrix_Type        * const result);

/**
 * \brief  Scalar multiplication with float input (auto-converted to Q31).
 *
 * \param[in]  matrix  Input matrix.
 * \param[in]  scalar  Float scalar (clamped to [−1.0, 1.0]).
 * \param[out] result  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_ScalarMultiplyFloat(
    const Matrix_Type  * const matrix,
    const MatrixFloat          scalar,
    Matrix_Type        * const result);

/**
 * \brief  Transpose: \p result = \p matrix ᵀ.
 *
 * \param[in]  matrix  Input matrix (m × n).
 * \param[out] result  Output matrix (buffer must be ≥ n × m).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_Type Matrix_Transpose(
    const Matrix_Type  * const matrix,
    Matrix_Type        * const result);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Advanced operations (iterative, no recursion)
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_advanced_ops  Advanced operations
 * \{
 */

/**
 * \brief  General determinant dispatcher (1 × 1 … 8 × 8).
 *
 * Delegates to the appropriately-sized specialised function below.
 *
 * \param[in]  matrix  Square input matrix (must not be NULL).
 * \param[out] det     Computed determinant (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SIZE_EXCEEDED; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Determinant(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of a 2 × 2 matrix.
 *         Formula: det = a₁₁·a₂₂ − a₁₂·a₂₁ (double-precision intermediate).
 *
 * \param[in]  matrix  2 × 2 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_Type Matrix_Determinant2x2(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of a 3 × 3 matrix (Sarrus' rule, double precision).
 *
 * \param[in]  matrix  3 × 3 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_Type Matrix_Determinant3x3(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of a 4 × 4 matrix (direct formula, double precision).
 *
 * \param[in]  matrix  4 × 4 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_Type Matrix_Determinant4x4(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of a 5 × 5 matrix via LU decomposition.
 *
 * \param[in]  matrix  5 × 5 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Determinant5x5(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of a 6 × 6 matrix via LU decomposition.
 *
 * \param[in]  matrix  6 × 6 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Determinant6x6(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of a 7 × 7 matrix via LU decomposition.
 *
 * \param[in]  matrix  7 × 7 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Determinant7x7(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  Determinant of an 8 × 8 matrix via LU decomposition.
 *
 * \param[in]  matrix  8 × 8 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Determinant8x8(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const det);

/**
 * \brief  General matrix inverse dispatcher (2 × 2 … 8 × 8, Gauss-Jordan).
 *
 * Uses optimised closed-form implementations for 2 × 2, 3 × 3, and 4 × 4;
 * falls back to augmented Gauss-Jordan for larger sizes.
 *
 * \param[in]  matrix  Square, invertible input matrix.
 * \param[out] result  Inverse matrix (buffer must be ≥ input dimensions).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SIZE_EXCEEDED; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Inverse(
    const Matrix_Type  * const matrix,
    Matrix_Type        * const result);

/**
 * \brief  Inverse of a 2 × 2 matrix.
 *         Formula: inv(A) = (1/det) · [ a₂₂  −a₁₂; −a₂₁  a₁₁ ]
 *
 * \param[in]  matrix  2 × 2 input matrix.
 * \param[out] result  2 × 2 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Inverse2x2(
    const Matrix_Type  * const matrix,
    Matrix_Type        * const result);

/**
 * \brief  Inverse of a 3 × 3 matrix (cofactor / adjugate method).
 *
 * \param[in]  matrix  3 × 3 input matrix.
 * \param[out] result  3 × 3 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Inverse3x3(
    const Matrix_Type  * const matrix,
    Matrix_Type        * const result);

/**
 * \brief  Inverse of a 4 × 4 matrix (augmented Gauss-Jordan, partial pivot).
 *
 * \param[in]  matrix  4 × 4 input matrix.
 * \param[out] result  4 × 4 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Inverse4x4(
    const Matrix_Type  * const matrix,
    Matrix_Type        * const result);

/**
 * \brief  Eigenvalue decomposition via iterative Jacobi method (symmetric matrices).
 *
 * Modifies \p matrix in-place during the sweep.  On success, the diagonal
 * of \p matrix holds the eigenvalues and \p eigen carries copies plus the
 * eigenvector columns.
 *
 * \param[in,out] matrix          Symmetric input matrix (modified in-place).
 * \param[out]    eigen           Eigenvalue / eigenvector result structure.
 * \param[in]     max_iterations  Maximum Jacobi sweeps (0 uses #JACOBI_MAX_ITER).
 * \param[in]     tolerance       Off-diagonal convergence threshold (0 uses default).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_MAX_ITERATIONS.
 */
extern MatrixStatus_Type Matrix_Eigenvalues(
    Matrix_Type      * const matrix,
    MatrixEigen_Type * const eigen,
    const uint32_T           max_iterations,
    const MatrixFloat        tolerance);

/**
 * \brief  Eigenvalues only (no eigenvectors; faster than #Matrix_Eigenvalues).
 *
 * \param[in,out] matrix          Symmetric input matrix (modified in-place).
 * \param[out]    eigenvalues     Array of at least \c matrix->rows floats.
 * \param[in]     max_iterations  Maximum Jacobi sweeps.
 * \param[in]     tolerance       Off-diagonal convergence threshold.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_MAX_ITERATIONS.
 */
extern MatrixStatus_Type Matrix_EigenvaluesOnly(
    Matrix_Type    * const matrix,
    MatrixFloat    * const eigenvalues,
    const uint32_T         max_iterations,
    const MatrixFloat      tolerance);

/**
 * \brief  LU decomposition with partial pivoting (in-place, iterative).
 *
 * On return \p matrix contains L (strict lower triangular, unit diagonal
 * not stored) and U (upper triangular) interleaved in the standard packed
 * form, and \p pivot records the row permutation.
 *
 * \param[in,out] matrix  Square input matrix; overwritten with L+U.
 * \param[out]    pivot   Permutation array (at least \c matrix->rows entries).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_LU(
    Matrix_Type  * const matrix,
    uint32_T     * const pivot);

/**
 * \brief  Solve A·x = b via LU decomposition (delegates to Gauss-Jordan).
 *
 * \param[in]  a  Square coefficient matrix.
 * \param[in]  b  Right-hand side matrix or vector.
 * \param[out] x  Solution (buffer must be ≥ a->rows × b->cols).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_Solve(
    const Matrix_Type  * const a,
    const Matrix_Type  * const b,
    Matrix_Type        * const x);

/**
 * \brief  Solve A·x = b via augmented Gauss-Jordan elimination (iterative).
 *
 * Well-suited for small systems (≤ 8 × 8).
 *
 * \param[in]  a  Square coefficient matrix.
 * \param[in]  b  Right-hand side matrix or vector.
 * \param[out] x  Solution.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_Type Matrix_SolveGaussJordan(
    const Matrix_Type  * const a,
    const Matrix_Type  * const b,
    Matrix_Type        * const x);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Matrix properties
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_properties  Properties
 * \{
 */

/**
 * \brief  Test whether a matrix is square.
 *
 * \param[in] matrix  Matrix to test (NULL → FALSE).
 * \return   TRUE if \c rows == \c cols, FALSE otherwise.
 */
extern boolean_T Matrix_IsSquare(const Matrix_Type * const matrix);

/**
 * \brief  Test whether a matrix is symmetric within a tolerance.
 *
 * \param[in] matrix     Matrix to test.
 * \param[in] tolerance  Maximum permitted element-wise asymmetry (float).
 *                       Pass 0.0f to use the internal default threshold.
 * \return   TRUE if |a[i][j] − a[j][i]| ≤ \p tolerance for all i, j.
 */
extern boolean_T Matrix_IsSymmetric(
    const Matrix_Type  * const matrix,
    const MatrixFloat          tolerance);

/**
 * \brief  Compute the trace (sum of diagonal elements).
 *
 * \param[in]  matrix  Square input matrix.
 * \param[out] trace   Receives the trace value.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE.
 */
extern MatrixStatus_Type Matrix_Trace(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const trace);

/**
 * \brief  Compute the Frobenius norm: ‖A‖_F = √(Σᵢⱼ aᵢⱼ²).
 *
 * \param[in]  matrix  Input matrix.
 * \param[out] norm    Receives the Frobenius norm.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_Type Matrix_NormFrobenius(
    const Matrix_Type  * const matrix,
    MatrixFloat        * const norm);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Utility functions
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_utility  Utility
 * \{
 */

/**
 * \brief  Test element-wise equality within a tolerance.
 *
 * \param[in] a          First matrix.
 * \param[in] b          Second matrix (must have same dimensions as \p a).
 * \param[in] tolerance  Maximum permitted element-wise difference (float).
 * \return   TRUE if all |a[i][j] − b[i][j]| ≤ \p tolerance.
 */
extern boolean_T Matrix_IsEqual(
    const Matrix_Type  * const a,
    const Matrix_Type  * const b,
    const MatrixFloat          tolerance);

/**
 * \brief  Retrieve the active dimensions of a matrix.
 *
 * Silently does nothing if any pointer is NULL.
 *
 * \param[in]  matrix  Source matrix.
 * \param[out] rows    Receives active row count.
 * \param[out] cols    Receives active column count.
 */
extern void Matrix_GetDimensions(
    const Matrix_Type  * const matrix,
    uint32_T           * const rows,
    uint32_T           * const cols);

/**
 * \brief  Fill all active elements with a constant Q31 value.
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     value   Q31 fill value.
 */
extern void Matrix_Fill(
    Matrix_Type        * const matrix,
    const MatrixElement        value);

/**
 * \brief  Fill all active elements with a constant float value (auto-converted).
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     value   Float fill value (clamped to [−1.0, 1.0]).
 */
extern void Matrix_FillFloat(
    Matrix_Type    * const matrix,
    const MatrixFloat      value);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Q31 conversion helpers
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_conversion  Q31 conversion helpers
 * \{
 */

/**
 * \brief  Convert a float to Q31 fixed-point.
 *
 * \p value is clamped to [−1.0, +0.9999999995] before conversion.
 * Formula: result = value × 2³¹
 *
 * \param[in] value  Float input.
 * \return           Q31 representation.
 */
extern MatrixElement Matrix_FloatToQ31(const MatrixFloat value);

/**
 * \brief  Convert a Q31 fixed-point value to float.
 *
 * Formula: result = value / 2³¹
 *
 * \param[in] value  Q31 input.
 * \return           Float in [−1.0, +1.0].
 */
extern MatrixFloat Matrix_Q31ToFloat(const MatrixElement value);

/** \} */

#endif /* MATRIX_H_ */
