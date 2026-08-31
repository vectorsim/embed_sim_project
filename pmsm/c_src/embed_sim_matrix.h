/**********************************************************************************************************************
 * \file      embed_sim_matrix.h
 * \brief     Single-precision floating-point linear algebra library for embedded systems.
 *
 * \details   Provides static-allocation matrix operations targeting 32-bit MCUs
 *            (Infineon AURIX TriCore, ARM Cortex-M4).  All algorithms are iterative —
 *            no recursion — and the implementation is MISRA C:2012 compliant.
 *
 *            Key properties:
 *              - IEEE 754 single-precision (real32_T) arithmetic
 *              - Maximum matrix size: 8 × 8
 *              - No dynamic memory allocation
 *              - Eigenvalue decomposition via iterative Jacobi method
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per identifier
 *              - Rule  8.6 : No definitions in header files
 *              - Rule 17.2 : No recursion
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCasePtr  (e.g., MatrixPtr, DetOutPtr)
 *              - Output pointers: PascalCasePtr
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   6.1.0
 * \date      2026-08-30
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_MATRIX_H_
#define EMBED_SIM_MATRIX_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"


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
 * \brief   Single-precision floating-point scalar type.
 *
 * Maps to \c real32_T (IEEE 754, 32-bit).
 * Range    : approximately ±3.4 × 10³⁸
 * Precision: ~7 significant decimal digits
 */
typedef real32_T MatrixElement;

/**
 * \typedef MatrixFloat
 * \brief   32-bit floating-point type used for intermediate calculations.
 *
 * Maps to \c real32_T from \c Sys_Types.h.
 */
typedef real32_T MatrixFloat;

/**
 * \enum  MatrixStatus_T
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
} MatrixStatus_T;

/**
 * \struct Matrix_T
 * \brief  Matrix handle for static-allocation use.
 *
 * All storage is provided by the caller via a fixed-size buffer; no heap
 * allocation occurs.  The \c stride field enables sub-matrix views without
 * copying.
 */
typedef struct
{
    MatrixElement  * Data;      /**< Pointer to the real32_T data buffer (row-major).    */
    uint32_T         Rows;      /**< Active row count.                                 */
    uint32_T         Cols;      /**< Active column count.                              */
    uint32_T         MaxRows;  /**< Allocated row capacity.                           */
    uint32_T         MaxCols;  /**< Allocated column capacity.                        */
    uint32_T         IsView;   /**< TRUE if this handle is a view of another matrix.  */
    uint32_T         Stride;    /**< Row stride in elements (equals max_cols normally).*/
} Matrix_T;

/**
 * \struct MatrixEigen_T
 * \brief  Result container for eigenvalue / eigenvector decomposition.
 */
typedef struct
{
    MatrixFloat  Eigenvalues[MATRIX_MAX_EIGEN];                   /**< Real eigenvalues (diagonal of diagonalised matrix). */
    MatrixFloat  Eigenvectors[MATRIX_MAX_ROWS * MATRIX_MAX_COLS]; /**< Eigenvectors stored column-wise.                    */
    uint32_T     NumEigenvalues;                                  /**< Number of eigenvalues computed.                     */
    uint32_T     Iterations;                                       /**< Jacobi sweeps consumed.                             */
} MatrixEigen_T;

/** \} */


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
 * Sets \c Rows and \c Cols to \p MaxRows and \p MaxCols respectively and
 * zero-fills the entire buffer via \c memset.
 *
 * \param[out] MatrixPtr   Matrix handle to initialise (must not be NULL).
 * \param[in]  buffer      Caller-allocated buffer of size
 *                         \p MaxRows × \p MaxCols × sizeof(#MatrixElement).
 * \param[in] MaxRows      Row capacity (1 … #MATRIX_MAX_ROWS).
 * \param[in] MaxCols      Column capacity (1 … #MATRIX_MAX_COLS).
 */
extern void Matrix_Init(
    Matrix_T    * const MatrixPtr,
    MatrixElement  * const buffer,
    const uint32_T MaxRows,
    const uint32_T MaxCols);

/**
 * \brief  Set the active dimensions of a matrix without touching data.
 *
 * \param[in,out] MatrixPtr  Matrix handle (must not be NULL).
 * \param[in]     Rows       Active rows    (1 … \c MatrixPtr->MaxRows).
 * \param[in]     Cols       Active columns (1 … \c MatrixPtr->MaxCols).
 */
extern void Matrix_SetDimensions(
    Matrix_T  * const MatrixPtr,
    const uint32_T Rows,
    const uint32_T Cols);

/**
 * \brief  Zero all elements within the active dimensions.
 *
 * \param[in,out] MatrixPtr  Matrix handle (must not be NULL).
 */
extern void Matrix_Zero(Matrix_T * const MatrixPtr);

/**
 * \brief  Set a square matrix to the identity (I).
 *
 * \param[in,out] MatrixPtr  Square matrix handle (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE.
 */
extern MatrixStatus_T Matrix_Identity(Matrix_T * const MatrixPtr);

/**
 * \brief  Copy \p SrcPtr into \p DestPtr.
 *
 * \p DestPtr must have sufficient capacity to hold all elements of \p SrcPtr.
 *
 * \param[out] DestPtr  Destination matrix (must not be NULL).
 * \param[in]  SrcPtr   Source matrix (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Copy(
    Matrix_T        * const DestPtr,
    const Matrix_T  * const SrcPtr);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Element access
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_element_access  Element access
 * \{
 */

/**
 * \brief  Write a real32_T value to a single element.
 *
 * \param[in,out] MatrixPtr  Target matrix (must not be NULL).
 * \param[in]     Row        0-based row index (< \c MatrixPtr->Rows).
 * \param[in]     Col        0-based column index (< \c MatrixPtr->Cols).
 * \param[in]     Value      real32_T value to write.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_SetElement(
    Matrix_T    * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixElement    Value);

/**
 * \brief  Write a float value to a single element.
 *
 * \param[in,out] MatrixPtr  Target matrix (must not be NULL).
 * \param[in]     Row        0-based row index.
 * \param[in]     Col        0-based column index.
 * \param[in]     Value      Float in [−1.0, 1.0] (clamped).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_SetElementFloat(
    Matrix_T    * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixFloat      Value);

/**
 * \brief  Read a real32_T value from a single element.
 *
 * \param[in]  MatrixPtr  Source matrix (must not be NULL).
 * \param[in]  Row        0-based row index.
 * \param[in]  Col        0-based column index.
 * \param[out] ValueOutPtr Receives the real32_T value (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_GetElement(
    const Matrix_T  * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    MatrixElement      * const ValueOutPtr);

/**
 * \brief  Read a single element and return it as a float.
 *
 * \param[in]  MatrixPtr   Source matrix (must not be NULL).
 * \param[in]  Row         0-based row index.
 * \param[in]  Col         0-based column index.
 * \param[out] ValueOutPtr Receives the float value (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_GetElementFloat(
    const Matrix_T  * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    MatrixFloat        * const ValueOutPtr);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Basic arithmetic operations
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_basic_ops  Basic arithmetic
 * \{
 */

/**
 * \brief  Element-wise addition: \p ResultPtr = \p APtr + \p BPtr .
 *
 * \param[in]  APtr       Left operand.
 * \param[in]  BPtr       Right operand (same dimensions as \p APtr).
 * \param[out] ResultPtr  Output matrix (buffer must be ≥ \p APtr dimensions).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Add(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Element-wise subtraction: \p ResultPtr = \p APtr − \p BPtr .
 *
 * \param[in]  APtr       Minuend.
 * \param[in]  BPtr       Subtrahend (same dimensions as \p APtr).
 * \param[out] ResultPtr  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Subtract(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Matrix multiplication: \p ResultPtr = \p APtr × \p BPtr .
 *
 * \p APtr is (m × n), \p BPtr is (n × p), \p ResultPtr is (m × p).
 *
 * \param[in]  APtr       Left factor  (m × n).
 * \param[in]  BPtr       Right factor (n × p).
 * \param[out] ResultPtr  Product matrix (buffer must be ≥ m × p).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Multiply(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Scalar multiplication: \p ResultPtr = \p Scalar × \p MatrixPtr .
 *
 * \param[in]  MatrixPtr  Input matrix.
 * \param[in]  Scalar     real32_T scalar.
 * \param[out] ResultPtr  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_ScalarMultiply(
    const Matrix_T  * const MatrixPtr,
    const MatrixElement        Scalar,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Scalar multiplication with float input.
 *
 * \param[in]  MatrixPtr  Input matrix.
 * \param[in]  Scalar     Float scalar (clamped to [−1.0, 1.0]).
 * \param[out] ResultPtr  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_ScalarMultiplyFloat(
    const Matrix_T  * const MatrixPtr,
    const MatrixFloat          Scalar,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Transpose: \p ResultPtr = \p MatrixPtr ᵀ.
 *
 * \param[in]  MatrixPtr  Input matrix (m × n).
 * \param[out] ResultPtr  Output matrix (buffer must be ≥ n × m).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Transpose(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const ResultPtr);

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
 * \param[in]  MatrixPtr   Square input matrix (must not be NULL).
 * \param[out] DetOutPtr   Computed determinant (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SIZE_EXCEEDED; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of a 2 × 2 matrix.
 *         Formula: det = a₁₁·a₂₂ − a₁₂·a₂₁ (double-precision intermediate).
 *
 * \param[in]  MatrixPtr  2 × 2 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_T Matrix_Determinant2x2(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of a 3 × 3 matrix (Sarrus' rule, double precision).
 *
 * \param[in]  MatrixPtr  3 × 3 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_T Matrix_Determinant3x3(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of a 4 × 4 matrix (direct formula, double precision).
 *
 * \param[in]  MatrixPtr  4 × 4 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_T Matrix_Determinant4x4(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of a 5 × 5 matrix via LU decomposition.
 *
 * \param[in]  MatrixPtr  5 × 5 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant5x5(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of a 6 × 6 matrix via LU decomposition.
 *
 * \param[in]  MatrixPtr  6 × 6 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant6x6(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of a 7 × 7 matrix via LU decomposition.
 *
 * \param[in]  MatrixPtr  7 × 7 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant7x7(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  Determinant of an 8 × 8 matrix via LU decomposition.
 *
 * \param[in]  MatrixPtr  8 × 8 input matrix.
 * \param[out] DetOutPtr  Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant8x8(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr);

/**
 * \brief  General matrix inverse dispatcher (2 × 2 … 8 × 8, Gauss-Jordan).
 *
 * Uses optimised closed-form implementations for 2 × 2, 3 × 3, and 4 × 4;
 * falls back to augmented Gauss-Jordan for larger sizes.
 *
 * \param[in]  MatrixPtr   Square, invertible input matrix.
 * \param[out] ResultPtr   Inverse matrix (buffer must be ≥ input dimensions).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SIZE_EXCEEDED; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Inverse of a 2 × 2 matrix.
 *         Formula: inv(A) = (1/det) · [ a₂₂  −a₁₂; −a₂₁  a₁₁ ]
 *
 * \param[in]  MatrixPtr  2 × 2 input matrix.
 * \param[out] ResultPtr  2 × 2 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse2x2(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Inverse of a 3 × 3 matrix (cofactor / adjugate method).
 *
 * \param[in]  MatrixPtr  3 × 3 input matrix.
 * \param[out] ResultPtr  3 × 3 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse3x3(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Inverse of a 4 × 4 matrix (augmented Gauss-Jordan, partial pivot).
 *
 * \param[in]  MatrixPtr  4 × 4 input matrix.
 * \param[out] ResultPtr  4 × 4 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse4x4(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const ResultPtr);

/**
 * \brief  Eigenvalue decomposition via iterative Jacobi method (symmetric matrices).
 *
 * Modifies \p MatrixPtr in-place during the sweep.  On success, the diagonal
 * of \p MatrixPtr holds the eigenvalues and \p EigenOutPtr carries copies plus the
 * eigenvector columns.
 *
 * \param[in,out] MatrixPtr        Symmetric input matrix (modified in-place).
 * \param[out]    EigenOutPtr      Eigenvalue / eigenvector result structure.
 * \param[in]     MaxIterations    Maximum Jacobi sweeps (0 uses #JACOBI_MAX_ITER).
 * \param[in]     Tolerance        Off-diagonal convergence threshold (0 uses default).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_MAX_ITERATIONS.
 */
extern MatrixStatus_T Matrix_Eigenvalues(
    Matrix_T      * const MatrixPtr,
    MatrixEigen_T * const EigenOutPtr,
    const uint32_T MaxIterations,
    const MatrixFloat        Tolerance);

/**
 * \brief  Eigenvalues only (no eigenvectors; faster than #Matrix_Eigenvalues).
 *
 * \param[in,out] MatrixPtr            Symmetric input matrix (modified in-place).
 * \param[out]    EigenvaluesOutPtr    Array of at least \c MatrixPtr->Rows floats.
 * \param[in]     MaxIterations        Maximum Jacobi sweeps.
 * \param[in]     Tolerance            Off-diagonal convergence threshold.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_MAX_ITERATIONS.
 */
extern MatrixStatus_T Matrix_EigenvaluesOnly(
    Matrix_T    * const MatrixPtr,
    MatrixFloat    * const EigenvaluesOutPtr,
    const uint32_T MaxIterations,
    const MatrixFloat      Tolerance);

/**
 * \brief  LU decomposition with partial pivoting (in-place, iterative).
 *
 * On return \p MatrixPtr contains L (strict lower triangular, unit diagonal
 * not stored) and U (upper triangular) interleaved in the standard packed
 * form, and \p pivot records the row permutation.
 *
 * \param[in,out] MatrixPtr  Square input matrix; overwritten with L+U.
 * \param[out]    pivot      Permutation array (at least \c MatrixPtr->Rows entries).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_LU(
    Matrix_T  * const MatrixPtr,
    uint32_T     * const pivot);

/**
 * \brief  Solve A·x = b via LU decomposition (delegates to Gauss-Jordan).
 *
 * \param[in]  APtr  Square coefficient matrix.
 * \param[in]  BPtr  Right-hand side matrix or vector.
 * \param[out] XPtr  Solution (buffer must be ≥ APtr->Rows × BPtr->Cols).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Solve(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr);

/**
 * \brief  Solve A·x = b via augmented Gauss-Jordan elimination (iterative).
 *
 * Well-suited for small systems (≤ 8 × 8).
 *
 * \param[in]  APtr  Square coefficient matrix.
 * \param[in]  BPtr  Right-hand side matrix or vector.
 * \param[out] XPtr  Solution.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_SolveGaussJordan(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr);

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
 * \param[in] MatrixPtr  Matrix to test (NULL → FALSE).
 * \return   TRUE if \c Rows == \c Cols, FALSE otherwise.
 */
extern boolean_T Matrix_IsSquare(const Matrix_T * const MatrixPtr);

/**
 * \brief  Test whether a matrix is symmetric within a tolerance.
 *
 * \param[in] MatrixPtr    Matrix to test.
 * \param[in] Tolerance    Maximum permitted element-wise asymmetry (float).
 *                         Pass 0.0f to use the internal default threshold.
 * \return   TRUE if |a[i][j] − a[j][i]| ≤ \p Tolerance for all i, j.
 */
extern boolean_T Matrix_IsSymmetric(
    const Matrix_T  * const MatrixPtr,
    const MatrixFloat          Tolerance);

/**
 * \brief  Compute the trace (sum of diagonal elements).
 *
 * \param[in]  MatrixPtr    Square input matrix.
 * \param[out] TraceOutPtr  Receives the trace value.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE.
 */
extern MatrixStatus_T Matrix_Trace(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const TraceOutPtr);

/**
 * \brief  Compute the Frobenius norm: ‖A‖_F = √(Σᵢⱼ aᵢⱼ²).
 *
 * \param[in]  MatrixPtr   Input matrix.
 * \param[out] NormOutPtr  Receives the Frobenius norm.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_T Matrix_NormFrobenius(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const NormOutPtr);

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
 * \param[in] APtr       First matrix.
 * \param[in] BPtr       Second matrix (must have same dimensions as \p APtr).
 * \param[in] Tolerance  Maximum permitted element-wise difference (float).
 * \return   TRUE if all |a[i][j] − b[i][j]| ≤ \p Tolerance.
 */
extern boolean_T Matrix_IsEqual(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    const MatrixFloat          Tolerance);

/**
 * \brief  Retrieve the active dimensions of a matrix.
 *
 * Silently does nothing if any pointer is NULL.
 *
 * \param[in]  MatrixPtr   Source matrix.
 * \param[out] RowsOutPtr  Receives active row count.
 * \param[out] ColsOutPtr  Receives active column count.
 */
extern void Matrix_GetDimensions(
    const Matrix_T  * const MatrixPtr,
    uint32_T           * const RowsOutPtr,
    uint32_T           * const ColsOutPtr);

/**
 * \brief  Fill all active elements with a constant real32_T value.
 *
 * \param[in,out] MatrixPtr  Target matrix (must not be NULL).
 * \param[in]     Value      real32_T fill value.
 */
extern void Matrix_Fill(
    Matrix_T        * const MatrixPtr,
    const MatrixElement        Value);

/**
 * \brief  Fill all active elements with a constant float value (auto-converted).
 *
 * \param[in,out] MatrixPtr  Target matrix (must not be NULL).
 * \param[in]     Value      Float fill value (clamped to [−1.0, 1.0]).
 */
extern void Matrix_FillFloat(
    Matrix_T    * const MatrixPtr,
    const MatrixFloat      Value);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Conversion helpers  (formerly Q31; now float identity — retained for API compatibility)
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_conversion  Conversion helpers
 * \{
 */

/**
 * \brief  Convert a float to MatrixElement (identity — MatrixElement is now real32_T).
 *
 * Retained for source compatibility with callers that used the Q31 API.
 * Value is clamped to [−1.0, +1.0] to preserve prior semantics.
 *
 * \param[in] Value  Float input.
 * \return           Same value clamped to [−1.0, +1.0].
 */
extern MatrixElement Matrix_FloatToQ31(const MatrixFloat Value);

/**
 * \brief  Convert MatrixElement to float (identity — MatrixElement is now real32_T).
 *
 * Retained for source compatibility with callers that used the Q31 API.
 *
 * \param[in] Value  MatrixElement (real32_T) input.
 * \return           Same value unchanged.
 */
extern MatrixFloat Matrix_Q31ToFloat(const MatrixElement Value);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Kalman Filter Specific Operations
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_kalman  Kalman Filter Operations
 * \{
 */

/**
 * \brief  Cholesky decomposition: A = L * L^T (positive definite symmetric)
 *
 * \param[in]  MatrixPtr  Symmetric positive definite matrix
 * \param[out] LPtr       Lower triangular matrix (same dimensions)
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NOT_SQUARE;
 *          MATRIX_ERROR_NON_POSITIVE_DEFINITE; MATRIX_ERROR_SIZE_EXCEEDED
 */
extern MatrixStatus_T Matrix_Cholesky(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const LPtr);

/**
 * \brief  Forward substitution: L * x = b (L lower triangular)
 *
 * \param[in]  LPtr  Lower triangular matrix
 * \param[in]  BPtr  Right-hand side
 * \param[out] XPtr  Solution vector
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_ForwardSubstitution(
    const Matrix_T  * const LPtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr);

/**
 * \brief  Backward substitution: U * x = b (U upper triangular)
 *
 * \param[in]  UPtr  Upper triangular matrix
 * \param[in]  BPtr  Right-hand side
 * \param[out] XPtr  Solution vector
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_BackwardSubstitution(
    const Matrix_T  * const UPtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr);

/**
 * \brief  Symmetric rank-1 update: A = A + alpha * v * v^T
 *
 * Used in Kalman filter covariance updates.
 *
 * \param[in,out] APtr   Symmetric matrix to update
 * \param[in]     VPtr   Vector (n×1 matrix)
 * \param[in]     Alpha  Scalar (real32_T)
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_SymmetricRank1Update(
    Matrix_T        * const APtr,
    const Matrix_T  * const VPtr,
    const MatrixElement        Alpha);

/**
 * \brief  Symmetric rank-1 update with float alpha
 *
 * \param[in,out] APtr   Symmetric matrix to update
 * \param[in]     VPtr   Vector (n×1 matrix)
 * \param[in]     Alpha  Float scalar
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_SymmetricRank1UpdateFloat(
    Matrix_T        * const APtr,
    const Matrix_T  * const VPtr,
    const MatrixFloat          Alpha);

/**
 * \brief  Matrix square root for positive semidefinite matrices (Denman-Beavers)
 *
 * Computes S such that S * S^T = A. Useful for square-root Kalman filters.
 *
 * \param[in]  MatrixPtr  Symmetric positive semidefinite matrix
 * \param[out] ResultPtr  Square root matrix (lower triangular)
 * \param[in]  MaxIter    Maximum iterations (0 = use default 10)
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NON_POSITIVE_DEFINITE;
 *          MATRIX_ERROR_MAX_ITERATIONS
 */
extern MatrixStatus_T Matrix_MatrixSquareRoot(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const ResultPtr,
    const uint32_T MaxIter);

/**
 * \brief  Condition number estimation (1-norm)
 *
 * \param[in]  MatrixPtr   Input matrix
 * \param[out] CondOutPtr  Condition number estimate
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_NOT_SQUARE
 */
extern MatrixStatus_T Matrix_ConditionNumber(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const CondOutPtr);

/**
 * \brief  Check if matrix is positive definite via Cholesky attempt
 *
 * \param[in]  MatrixPtr  Matrix to test
 * \return  TRUE if positive definite, FALSE otherwise
 */
extern boolean_T Matrix_IsPositiveDefinite(const Matrix_T * const MatrixPtr);

/** \} */


#endif /* EMBED_SIM_MATRIX_H_ */