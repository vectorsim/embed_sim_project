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
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCase_P
 *              - Local variables: lower_snake_case
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   5.1.0
 * \date      2025-05-24
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
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
 * \struct Matrix_Type
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
} Matrix_Type;

/**
 * \struct MatrixEigen_Type
 * \brief  Result container for eigenvalue / eigenvector decomposition.
 */
typedef struct
{
    MatrixFloat  Eigenvalues[MATRIX_MAX_EIGEN];                   /**< Real eigenvalues (diagonal of diagonalised matrix). */
    MatrixFloat  Eigenvectors[MATRIX_MAX_ROWS * MATRIX_MAX_COLS]; /**< Eigenvectors stored column-wise.                    */
    uint32_T     NumEigenvalues;                                  /**< Number of eigenvalues computed.                     */
    uint32_T     Iterations;                                       /**< Jacobi sweeps consumed.                             */
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
 * Sets \c Rows and \c Cols to \p MaxRows and \p MaxCols respectively and
 * zero-fills the entire buffer via \c memset.
 *
 * \param[out] matrix    Matrix handle to initialise (must not be NULL).
 * \param[in]  buffer    Caller-allocated buffer of size
 *                       \p MaxRows × \p MaxCols × sizeof(#MatrixElement).
 * \param[in] MaxRows  Row capacity (1 … #MATRIX_MAX_ROWS).
 * \param[in] MaxCols  Column capacity (1 … #MATRIX_MAX_COLS).
 */
extern void Matrix_Init(
    Matrix_Type    * const Matrix_P,
    MatrixElement  * const Buffer,
    const uint32_T MaxRows,
    const uint32_T MaxCols);

/**
 * \brief  Set the active dimensions of a matrix without touching data.
 *
 * \param[in,out] matrix  Matrix handle (must not be NULL).
 * \param[in]     rows    Active rows    (1 … \c matrix->MaxRows).
 * \param[in]     cols    Active columns (1 … \c matrix->MaxCols).
 */
extern void Matrix_SetDimensions(
    Matrix_Type  * const Matrix_P,
    const uint32_T Rows,
    const uint32_T Cols);

/**
 * \brief  Zero all elements within the active dimensions.
 *
 * \param[in,out] matrix  Matrix handle (must not be NULL).
 */
extern void Matrix_Zero(Matrix_Type * const Matrix_P);

/**
 * \brief  Set a square matrix to the identity (I).
 *
 * \param[in,out] matrix  Square matrix handle (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE.
 */
extern MatrixStatus_T Matrix_Identity(Matrix_Type * const Matrix_P);

/**
 * \brief  Copy \p src into \p dest.
 *
 * \p dest must have sufficient capacity to hold all elements of \p src.
 *
 * \param[out] dest  Destination matrix (must not be NULL).
 * \param[in]  src   Source matrix (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Copy(
    Matrix_Type        * const Dest_P,
    const Matrix_Type  * const Src_P);

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
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     row     0-based row index (< \c Matrix_P->Rows).
 * \param[in]     col     0-based column index (< \c Matrix_P->Cols).
 * \param[in]     value   real32_T value to write.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_SetElement(
    Matrix_Type    * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixElement    Value);

/**
 * \brief  Write a float value to a single element.
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     row     0-based row index.
 * \param[in]     col     0-based column index.
 * \param[in]     value   Float in [−1.0, 1.0] (clamped).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_SetElementFloat(
    Matrix_Type    * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixFloat      Value);

/**
 * \brief  Read a real32_T value from a single element.
 *
 * \param[in]  matrix  Source matrix (must not be NULL).
 * \param[in]  row     0-based row index.
 * \param[in]  col     0-based column index.
 * \param[out] value   Receives the real32_T value (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_GetElement(
    const Matrix_Type  * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    MatrixElement      * const ValueOut_P);

/**
 * \brief  Read a single element and return it as a float.
 *
 * \param[in]  matrix  Source matrix (must not be NULL).
 * \param[in]  row     0-based row index.
 * \param[in]  col     0-based column index.
 * \param[out] value   Receives the float value (must not be NULL).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_OUT_OF_BOUNDS.
 */
extern MatrixStatus_T Matrix_GetElementFloat(
    const Matrix_Type  * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    MatrixFloat        * const ValueOut_P);

/** \} */

/*--------------------------------------------------------------------------------------------------------------------
 * Basic arithmetic operations
 *------------------------------------------------------------------------------------------------------------------*/

/** \addtogroup matrix_basic_ops  Basic arithmetic
 * \{
 */

/**
 * \brief  Element-wise addition: \p result = \p A + \p B .
 *
 * \param[in]  a       Left operand.
 * \param[in]  b       Right operand (same dimensions as \p A).
 * \param[out] result  Output matrix (buffer must be ≥ \p A dimensions).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Add(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Element-wise subtraction: \p result = \p A − \p B .
 *
 * \param[in]  a       Minuend.
 * \param[in]  b       Subtrahend (same dimensions as \p A).
 * \param[out] result  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Subtract(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Matrix multiplication: \p result = \p A × \p B .
 *
 * \p A is (m × n), \p B is (n × p), \p result is (m × p).
 *
 * \param[in]  a       Left factor  (m × n).
 * \param[in]  b       Right factor (n × p).
 * \param[out] result  Product matrix (buffer must be ≥ m × p).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Multiply(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Scalar multiplication: \p result = \p scalar × \p matrix .
 *
 * \param[in]  matrix  Input matrix.
 * \param[in]  scalar  real32_T scalar.
 * \param[out] result  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_ScalarMultiply(
    const Matrix_Type  * const Matrix_P,
    const MatrixElement        Scalar,
    Matrix_Type        * const Result_P);

/**
 * \brief  Scalar multiplication with float input.
 *
 * \param[in]  matrix  Input matrix.
 * \param[in]  scalar  Float scalar (clamped to [−1.0, 1.0]).
 * \param[out] result  Output matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_ScalarMultiplyFloat(
    const Matrix_Type  * const Matrix_P,
    const MatrixFloat          Scalar,
    Matrix_Type        * const Result_P);

/**
 * \brief  Transpose: \p result = \p matrix ᵀ.
 *
 * \param[in]  matrix  Input matrix (m × n).
 * \param[out] result  Output matrix (buffer must be ≥ n × m).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_SIZE_EXCEEDED.
 */
extern MatrixStatus_T Matrix_Transpose(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const Result_P);

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
extern MatrixStatus_T Matrix_Determinant(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of a 2 × 2 matrix.
 *         Formula: det = a₁₁·a₂₂ − a₁₂·a₂₁ (double-precision intermediate).
 *
 * \param[in]  matrix  2 × 2 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_T Matrix_Determinant2x2(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of a 3 × 3 matrix (Sarrus' rule, double precision).
 *
 * \param[in]  matrix  3 × 3 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_T Matrix_Determinant3x3(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of a 4 × 4 matrix (direct formula, double precision).
 *
 * \param[in]  matrix  4 × 4 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_DIMENSION_MISMATCH.
 */
extern MatrixStatus_T Matrix_Determinant4x4(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of a 5 × 5 matrix via LU decomposition.
 *
 * \param[in]  matrix  5 × 5 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant5x5(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of a 6 × 6 matrix via LU decomposition.
 *
 * \param[in]  matrix  6 × 6 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant6x6(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of a 7 × 7 matrix via LU decomposition.
 *
 * \param[in]  matrix  7 × 7 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant7x7(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

/**
 * \brief  Determinant of an 8 × 8 matrix via LU decomposition.
 *
 * \param[in]  matrix  8 × 8 input matrix.
 * \param[out] det     Computed determinant.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Determinant8x8(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P);

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
extern MatrixStatus_T Matrix_Inverse(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Inverse of a 2 × 2 matrix.
 *         Formula: inv(A) = (1/det) · [ a₂₂  −a₁₂; −a₂₁  a₁₁ ]
 *
 * \param[in]  matrix  2 × 2 input matrix.
 * \param[out] result  2 × 2 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse2x2(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Inverse of a 3 × 3 matrix (cofactor / adjugate method).
 *
 * \param[in]  matrix  3 × 3 input matrix.
 * \param[out] result  3 × 3 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse3x3(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Inverse of a 4 × 4 matrix (augmented Gauss-Jordan, partial pivot).
 *
 * \param[in]  matrix  4 × 4 input matrix.
 * \param[out] result  4 × 4 inverse matrix.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Inverse4x4(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const Result_P);

/**
 * \brief  Eigenvalue decomposition via iterative Jacobi method (symmetric matrices).
 *
 * Modifies \p matrix in-place during the sweep.  On success, the diagonal
 * of \p matrix holds the eigenvalues and \p eigen carries copies plus the
 * eigenvector columns.
 *
 * \param[in,out] matrix          Symmetric input matrix (modified in-place).
 * \param[out]    eigen           Eigenvalue / eigenvector result structure.
 * \param[in] MaxIterations  Maximum Jacobi sweeps (0 uses #JACOBI_MAX_ITER).
 * \param[in]     tolerance       Off-diagonal convergence threshold (0 uses default).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_MAX_ITERATIONS.
 */
extern MatrixStatus_T Matrix_Eigenvalues(
    Matrix_Type      * const Matrix_P,
    MatrixEigen_Type * const EigenOut_P,
    const uint32_T MaxIterations,
    const MatrixFloat        Tolerance);

/**
 * \brief  Eigenvalues only (no eigenvectors; faster than #Matrix_Eigenvalues).
 *
 * \param[in,out] matrix          Symmetric input matrix (modified in-place).
 * \param[out]    eigenvalues     Array of at least \c Matrix_P->Rows floats.
 * \param[in] MaxIterations  Maximum Jacobi sweeps.
 * \param[in]     tolerance       Off-diagonal convergence threshold.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_MAX_ITERATIONS.
 */
extern MatrixStatus_T Matrix_EigenvaluesOnly(
    Matrix_Type    * const Matrix_P,
    MatrixFloat    * const EigenvaluesOut_P,
    const uint32_T MaxIterations,
    const MatrixFloat      Tolerance);

/**
 * \brief  LU decomposition with partial pivoting (in-place, iterative).
 *
 * On return \p matrix contains L (strict lower triangular, unit diagonal
 * not stored) and U (upper triangular) interleaved in the standard packed
 * form, and \p pivot records the row permutation.
 *
 * \param[in,out] matrix  Square input matrix; overwritten with L+U.
 * \param[out]    pivot   Permutation array (at least \c Matrix_P->Rows entries).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_LU(
    Matrix_Type  * const Matrix_P,
    uint32_T     * const pivot);

/**
 * \brief  Solve A·x = b via LU decomposition (delegates to Gauss-Jordan).
 *
 * \param[in]  a  Square coefficient matrix.
 * \param[in]  b  Right-hand side matrix or vector.
 * \param[out] x  Solution (buffer must be ≥ a->Rows × b->Cols).
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE;
 *          #MATRIX_ERROR_DIMENSION_MISMATCH; #MATRIX_ERROR_SIZE_EXCEEDED;
 *          #MATRIX_ERROR_SINGULAR.
 */
extern MatrixStatus_T Matrix_Solve(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P);

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
extern MatrixStatus_T Matrix_SolveGaussJordan(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P);

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
 * \return   TRUE if \c Rows == \c Cols, FALSE otherwise.
 */
extern boolean_T Matrix_IsSquare(const Matrix_Type * const Matrix_P);

/**
 * \brief  Test whether a matrix is symmetric within a tolerance.
 *
 * \param[in] matrix     Matrix to test.
 * \param[in] tolerance  Maximum permitted element-wise asymmetry (float).
 *                       Pass 0.0f to use the internal default threshold.
 * \return   TRUE if |a[i][j] − a[j][i]| ≤ \p tolerance for all i, j.
 */
extern boolean_T Matrix_IsSymmetric(
    const Matrix_Type  * const Matrix_P,
    const MatrixFloat          Tolerance);

/**
 * \brief  Compute the trace (sum of diagonal elements).
 *
 * \param[in]  matrix  Square input matrix.
 * \param[out] trace   Receives the trace value.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR; #MATRIX_ERROR_NOT_SQUARE.
 */
extern MatrixStatus_T Matrix_Trace(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const TraceOut_P);

/**
 * \brief  Compute the Frobenius norm: ‖A‖_F = √(Σᵢⱼ aᵢⱼ²).
 *
 * \param[in]  matrix  Input matrix.
 * \param[out] norm    Receives the Frobenius norm.
 * \return  #MATRIX_SUCCESS; #MATRIX_ERROR_NULL_PTR.
 */
extern MatrixStatus_T Matrix_NormFrobenius(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const NormOut_P);

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
 * \param[in] A          First matrix.
 * \param[in] B          Second matrix (must have same dimensions as \p A).
 * \param[in] tolerance  Maximum permitted element-wise difference (float).
 * \return   TRUE if all |a[i][j] − b[i][j]| ≤ \p tolerance.
 */
extern boolean_T Matrix_IsEqual(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    const MatrixFloat          Tolerance);

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
    const Matrix_Type  * const Matrix_P,
    uint32_T           * const RowsOut_P,
    uint32_T           * const ColsOut_P);

/**
 * \brief  Fill all active elements with a constant real32_T value.
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     value   real32_T fill value.
 */
extern void Matrix_Fill(
    Matrix_Type        * const Matrix_P,
    const MatrixElement        Value);

/**
 * \brief  Fill all active elements with a constant float value (auto-converted).
 *
 * \param[in,out] matrix  Target matrix (must not be NULL).
 * \param[in]     value   Float fill value (clamped to [−1.0, 1.0]).
 */
extern void Matrix_FillFloat(
    Matrix_Type    * const Matrix_P,
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
 * \param[in]  matrix  Symmetric positive definite matrix
 * \param[out] L       Lower triangular matrix (same dimensions)
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NOT_SQUARE;
 *          MATRIX_ERROR_NON_POSITIVE_DEFINITE; MATRIX_ERROR_SIZE_EXCEEDED
 */
extern MatrixStatus_T Matrix_Cholesky(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const L_P);

/**
 * \brief  Forward substitution: L * x = b (L lower triangular)
 *
 * \param[in]  L   Lower triangular matrix
 * \param[in]  b   Right-hand side
 * \param[out] x   Solution vector
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_ForwardSubstitution(
    const Matrix_Type  * const L_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P);

/**
 * \brief  Backward substitution: U * x = b (U upper triangular)
 *
 * \param[in]  U   Upper triangular matrix
 * \param[in]  b   Right-hand side
 * \param[out] x   Solution vector
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_BackwardSubstitution(
    const Matrix_Type  * const U_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P);

/**
 * \brief  Symmetric rank-1 update: A = A + alpha * v * v^T
 *
 * Used in Kalman filter covariance updates.
 *
 * \param[in,out] A     Symmetric matrix to update
 * \param[in]     v     Vector (n×1 matrix)
 * \param[in]     alpha Scalar (real32_T)
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_SymmetricRank1Update(
    Matrix_Type        * const A_P,
    const Matrix_Type  * const V_P,
    const MatrixElement        Alpha);

/**
 * \brief  Symmetric rank-1 update with float alpha
 *
 * \param[in,out] A     Symmetric matrix to update
 * \param[in]     v     Vector (n×1 matrix)
 * \param[in]     alpha Float scalar
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_DIMENSION_MISMATCH
 */
extern MatrixStatus_T Matrix_SymmetricRank1UpdateFloat(
    Matrix_Type        * const A_P,
    const Matrix_Type  * const V_P,
    const MatrixFloat          Alpha);

/**
 * \brief  Matrix square root for positive semidefinite matrices (Denman-Beavers)
 *
 * Computes S such that S * S^T = A. Useful for square-root Kalman filters.
 *
 * \param[in]  matrix     Symmetric positive semidefinite matrix
 * \param[out] result     Square root matrix (lower triangular)
 * \param[in]  max_iter   Maximum iterations (0 = use default 10)
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NON_POSITIVE_DEFINITE;
 *          MATRIX_ERROR_MAX_ITERATIONS
 */
extern MatrixStatus_T Matrix_MatrixSquareRoot(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const Result_P,
    const uint32_T MaxIter);

/**
 * \brief  Condition number estimation (1-norm)
 *
 * \param[in]  matrix   Input matrix
 * \param[out] cond     Condition number estimate
 * \return  MATRIX_SUCCESS; MATRIX_ERROR_NULL_PTR; MATRIX_ERROR_NOT_SQUARE
 */
extern MatrixStatus_T Matrix_ConditionNumber(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const CondOut_P);

/**
 * \brief  Check if matrix is positive definite via Cholesky attempt
 *
 * \param[in]  matrix   Matrix to test
 * \return  TRUE if positive definite, FALSE otherwise
 */
extern boolean_T Matrix_IsPositiveDefinite(const Matrix_Type * const Matrix_P);

/** \} */


#endif /* EMBED_SIM_MATRIX_H_ */
