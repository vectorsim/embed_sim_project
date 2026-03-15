/*
 * Matrix.h
 * ========
 * 32-bit Linear Algebra Library for Embedded Systems
 *
 * MISRA C:2012 / AUTOSAR C compliant
 * - NO RECURSION (all algorithms iterative)
 * - 32-bit fixed-point arithmetic (Q31 format)
 * - Supports matrices up to 8x8 with eigenvalue calculation
 * - All indices use uint32_T for sizes, int32_T for values
 *
 * Target: 32-bit MCUs (Infineon Aurix TriCore, ARM Cortex-M4)
 * Version: 5.0.0
 *
 * @file Matrix.h
 * @brief Linear algebra library for embedded systems
 * @author EmbedSim Project
 * @date 2024
 * @copyright MIT License
 */

#ifndef MATRIX_H
#define MATRIX_H

/* AUTOSAR standard types from your system */
#include "Sys_Types.h"

/**
 * @def MATRIX_MAX_ROWS
 * @brief Maximum number of rows supported (8 for eigenvalue calculation)
 */
#define MATRIX_MAX_ROWS     (8U)

/**
 * @def MATRIX_MAX_COLS
 * @brief Maximum number of columns supported
 */
#define MATRIX_MAX_COLS     (8U)

/**
 * @def MATRIX_MAX_EIGEN
 * @brief Maximum number of eigenvalues that can be stored
 */
#define MATRIX_MAX_EIGEN     (8U)

/**
 * @typedef MatrixElement
 * @brief 32-bit fixed-point element type in Q31 format
 *
 * Q31 format: 1 sign bit + 31 fractional bits
 * Range: [-1.0, 0.9999999995]
 * Resolution: 2.33e-10
 */
typedef int32_T MatrixElement;

/**
 * @typedef MatrixFloat
 * @brief 32-bit floating-point type for intermediate calculations
 * Uses real32_T from Sys_Types.h
 */
typedef real32_T MatrixFloat;

/**
 * @enum MatrixStatus_Type
 * @brief Status codes returned by matrix operations
 */
typedef enum {
    MATRIX_SUCCESS = 0,                    /**< Operation completed successfully */
    MATRIX_ERROR_NULL_PTR = 1,              /**< NULL pointer provided */
    MATRIX_ERROR_DIMENSION_MISMATCH = 2,    /**< Matrix dimensions incompatible */
    MATRIX_ERROR_SINGULAR = 3,               /**< Matrix is singular (det = 0) */
    MATRIX_ERROR_SIZE_EXCEEDED = 4,          /**< Matrix exceeds maximum dimensions */
    MATRIX_ERROR_DIV_BY_ZERO = 5,            /**< Division by zero attempted */
    MATRIX_ERROR_NOT_SQUARE = 6,             /**< Operation requires square matrix */
    MATRIX_ERROR_BUFFER_OVERFLOW = 7,        /**< Buffer size insufficient */
    MATRIX_ERROR_OUT_OF_BOUNDS = 8,          /**< Index out of bounds */
    MATRIX_ERROR_NON_POSITIVE_DEFINITE = 9,  /**< Matrix not positive definite */
    MATRIX_ERROR_NOT_INVERTIBLE = 10,        /**< Matrix cannot be inverted */
    MATRIX_ERROR_NOT_CONVERGENT = 11,        /**< Iterative algorithm did not converge */
    MATRIX_ERROR_MAX_ITERATIONS = 12         /**< Maximum iterations reached */
} MatrixStatus_Type;

/**
 * @struct Matrix_Type
 * @brief Matrix handle structure for static allocation
 *
 * All matrices use static buffers - no dynamic memory allocation.
 * The stride allows for submatrix views without copying data.
 */
typedef struct {
    MatrixElement*  data;           /**< Pointer to 32-bit data buffer */
    uint32_T        rows;           /**< Current number of rows */
    uint32_T        cols;           /**< Current number of columns */
    uint32_T        max_rows;       /**< Maximum allocated rows */
    uint32_T        max_cols;       /**< Maximum allocated columns */
    uint32_T        is_view;        /**< Boolean flag: TRUE if view of another matrix */
    uint32_T        stride;         /**< Row stride in elements (for views) */
} Matrix_Type;

/**
 * @struct MatrixEigen_Type
 * @brief Structure to hold eigenvalue decomposition results
 */
typedef struct {
    MatrixFloat     eigenvalues[MATRIX_MAX_EIGEN];     /**< Real eigenvalues found */
    MatrixFloat     eigenvectors[MATRIX_MAX_ROWS * MATRIX_MAX_COLS]; /**< Eigenvectors (column-wise) */
    uint32_T        num_eigenvalues;                    /**< Number of eigenvalues found */
    uint32_T        iterations;                          /**< Number of iterations used */
} MatrixEigen_Type;

/*==============================================================================
 * INITIALIZATION FUNCTIONS
 *============================================================================*/

/**
 * @brief Initialize a matrix with a static buffer
 *
 * @param[out] matrix  Pointer to matrix structure to initialize
 * @param[in]  buffer  Pointer to data buffer (size: max_rows * max_cols)
 * @param[in]  max_rows Maximum number of rows that can be stored
 * @param[in]  max_cols Maximum number of columns that can be stored
 *
 * @pre matrix != NULL
 * @pre buffer != NULL
 * @pre max_rows > 0U && max_rows <= MATRIX_MAX_ROWS
 * @pre max_cols > 0U && max_cols <= MATRIX_MAX_COLS
 *
 * @post matrix->data == buffer
 * @post matrix->rows == max_rows
 * @post matrix->cols == max_cols
 * @post matrix->max_rows == max_rows
 * @post matrix->max_cols == max_cols
 * @post matrix->is_view == FALSE
 * @post matrix->stride == max_cols
 * @post All elements initialized to zero
 */
extern void Matrix_Init(Matrix_Type* const matrix,
                        MatrixElement* const buffer,
                        const uint32_T max_rows,
                        const uint32_T max_cols);

/**
 * @brief Set current dimensions of a matrix
 *
 * @param[in,out] matrix Pointer to matrix structure
 * @param[in]     rows   Number of rows to set (must be <= max_rows)
 * @param[in]     cols   Number of columns to set (must be <= max_cols)
 *
 * @pre matrix != NULL
 * @pre rows > 0U && rows <= matrix->max_rows
 * @pre cols > 0U && cols <= matrix->max_cols
 *
 * @post matrix->rows == rows
 * @post matrix->cols == cols
 */
extern void Matrix_SetDimensions(Matrix_Type* const matrix,
                                 const uint32_T rows,
                                 const uint32_T cols);

/**
 * @brief Set all matrix elements to zero
 *
 * @param[in,out] matrix Pointer to matrix structure
 *
 * @pre matrix != NULL
 *
 * @post All elements in the current dimension range are zero
 */
extern void Matrix_Zero(Matrix_Type* const matrix);

/**
 * @brief Set matrix to identity matrix (ones on diagonal, zeros elsewhere)
 *
 * @param[in,out] matrix Pointer to matrix structure
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS           Operation successful
 * @retval MATRIX_ERROR_NULL_PTR    matrix is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE  matrix is not square
 *
 * @pre matrix != NULL
 * @pre matrix->rows == matrix->cols (square matrix)
 *
 * @post Diagonal elements set to Q31_ONE
 * @post Off-diagonal elements set to Q31_ZERO
 */
extern MatrixStatus_Type Matrix_Identity(Matrix_Type* const matrix);

/**
 * @brief Copy one matrix to another
 *
 * @param[out] dest Destination matrix
 * @param[in]  src  Source matrix
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS           Operation successful
 * @retval MATRIX_ERROR_NULL_PTR    dest or src is NULL
 * @retval MATRIX_ERROR_SIZE_EXCEEDED dest buffer too small
 *
 * @pre dest != NULL && src != NULL
 * @pre dest->max_rows >= src->rows
 * @pre dest->max_cols >= src->cols
 *
 * @post dest->rows == src->rows
 * @post dest->cols == src->cols
 * @post dest elements equal src elements
 */
extern MatrixStatus_Type Matrix_Copy(Matrix_Type* const dest,
                                      const Matrix_Type* const src);

/*==============================================================================
 * ELEMENT ACCESS FUNCTIONS
 *============================================================================*/

/**
 * @brief Set a single matrix element (Q31 format)
 *
 * @param[in,out] matrix Pointer to matrix structure
 * @param[in]     row    Row index (0-based)
 * @param[in]     col    Column index (0-based)
 * @param[in]     value  Q31 value to set
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS           Operation successful
 * @retval MATRIX_ERROR_NULL_PTR    matrix is NULL
 * @retval MATRIX_ERROR_OUT_OF_BOUNDS row or col out of range
 *
 * @pre matrix != NULL
 * @pre row < matrix->rows
 * @pre col < matrix->cols
 */
extern MatrixStatus_Type Matrix_SetElement(Matrix_Type* const matrix,
                                           const uint32_T row,
                                           const uint32_T col,
                                           const MatrixElement value);

/**
 * @brief Set a single matrix element (float format, auto-converted to Q31)
 *
 * @param[in,out] matrix Pointer to matrix structure
 * @param[in]     row    Row index (0-based)
 * @param[in]     col    Column index (0-based)
 * @param[in]     value  Float value to set (clamped to [-1.0, 1.0])
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS           Operation successful
 * @retval MATRIX_ERROR_NULL_PTR    matrix is NULL
 * @retval MATRIX_ERROR_OUT_OF_BOUNDS row or col out of range
 *
 * @pre matrix != NULL
 * @pre row < matrix->rows
 * @pre col < matrix->cols
 */
extern MatrixStatus_Type Matrix_SetElementFloat(Matrix_Type* const matrix,
                                                const uint32_T row,
                                                const uint32_T col,
                                                const MatrixFloat value);

/**
 * @brief Get a single matrix element (Q31 format)
 *
 * @param[in]  matrix Pointer to matrix structure
 * @param[in]  row    Row index (0-based)
 * @param[in]  col    Column index (0-based)
 * @param[out] value  Pointer to store the Q31 value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS           Operation successful
 * @retval MATRIX_ERROR_NULL_PTR    matrix or value is NULL
 * @retval MATRIX_ERROR_OUT_OF_BOUNDS row or col out of range
 *
 * @pre matrix != NULL && value != NULL
 * @pre row < matrix->rows
 * @pre col < matrix->cols
 */
extern MatrixStatus_Type Matrix_GetElement(const Matrix_Type* const matrix,
                                           const uint32_T row,
                                           const uint32_T col,
                                           MatrixElement* const value);

/**
 * @brief Get a single matrix element as float
 *
 * @param[in]  matrix Pointer to matrix structure
 * @param[in]  row    Row index (0-based)
 * @param[in]  col    Column index (0-based)
 * @param[out] value  Pointer to store the float value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS           Operation successful
 * @retval MATRIX_ERROR_NULL_PTR    matrix or value is NULL
 * @retval MATRIX_ERROR_OUT_OF_BOUNDS row or col out of range
 *
 * @pre matrix != NULL && value != NULL
 * @pre row < matrix->rows
 * @pre col < matrix->cols
 */
extern MatrixStatus_Type Matrix_GetElementFloat(const Matrix_Type* const matrix,
                                                const uint32_T row,
                                                const uint32_T col,
                                                MatrixFloat* const value);

/*==============================================================================
 * BASIC OPERATIONS
 *============================================================================*/

/**
 * @brief Add two matrices: result = a + b
 *
 * @param[in]  a      First matrix
 * @param[in]  b      Second matrix (same dimensions as a)
 * @param[out] result Result matrix
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS                Operation successful
 * @retval MATRIX_ERROR_NULL_PTR         a, b, or result is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH a and b dimensions differ
 * @retval MATRIX_ERROR_SIZE_EXCEEDED    result buffer too small
 *
 * @pre a != NULL && b != NULL && result != NULL
 * @pre a->rows == b->rows && a->cols == b->cols
 * @pre result->max_rows >= a->rows
 * @pre result->max_cols >= a->cols
 *
 * @post result->rows == a->rows
 * @post result->cols == a->cols
 */
extern MatrixStatus_Type Matrix_Add(const Matrix_Type* const a,
                                     const Matrix_Type* const b,
                                     Matrix_Type* const result);

/**
 * @brief Subtract two matrices: result = a - b
 *
 * @param[in]  a      First matrix
 * @param[in]  b      Second matrix (same dimensions as a)
 * @param[out] result Result matrix
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS                Operation successful
 * @retval MATRIX_ERROR_NULL_PTR         a, b, or result is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH a and b dimensions differ
 * @retval MATRIX_ERROR_SIZE_EXCEEDED    result buffer too small
 *
 * @pre a != NULL && b != NULL && result != NULL
 * @pre a->rows == b->rows && a->cols == b->cols
 * @pre result->max_rows >= a->rows
 * @pre result->max_cols >= a->cols
 */
extern MatrixStatus_Type Matrix_Subtract(const Matrix_Type* const a,
                                          const Matrix_Type* const b,
                                          Matrix_Type* const result);

/**
 * @brief Multiply two matrices: result = a * b
 *
 * @param[in]  a      First matrix (m x n)
 * @param[in]  b      Second matrix (n x p)
 * @param[out] result Result matrix (m x p)
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS                Operation successful
 * @retval MATRIX_ERROR_NULL_PTR         a, b, or result is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH a->cols != b->rows
 * @retval MATRIX_ERROR_SIZE_EXCEEDED    result buffer too small
 *
 * @pre a != NULL && b != NULL && result != NULL
 * @pre a->cols == b->rows
 * @pre result->max_rows >= a->rows
 * @pre result->max_cols >= b->cols
 *
 * @post result->rows == a->rows
 * @post result->cols == b->cols
 */
extern MatrixStatus_Type Matrix_Multiply(const Matrix_Type* const a,
                                          const Matrix_Type* const b,
                                          Matrix_Type* const result);

/**
 * @brief Multiply matrix by scalar: result = scalar * matrix
 *
 * @param[in]  matrix Input matrix
 * @param[in]  scalar Q31 scalar value
 * @param[out] result Result matrix
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_SIZE_EXCEEDED result buffer too small
 *
 * @pre matrix != NULL && result != NULL
 * @pre result->max_rows >= matrix->rows
 * @pre result->max_cols >= matrix->cols
 */
extern MatrixStatus_Type Matrix_ScalarMultiply(const Matrix_Type* const matrix,
                                                const MatrixElement scalar,
                                                Matrix_Type* const result);

/**
 * @brief Multiply matrix by float scalar (auto-converted to Q31)
 *
 * @param[in]  matrix Input matrix
 * @param[in]  scalar Float scalar value (clamped to [-1.0, 1.0])
 * @param[out] result Result matrix
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_SIZE_EXCEEDED result buffer too small
 */
extern MatrixStatus_Type Matrix_ScalarMultiplyFloat(const Matrix_Type* const matrix,
                                                    const MatrixFloat scalar,
                                                    Matrix_Type* const result);

/**
 * @brief Transpose matrix: result = matrix^T
 *
 * @param[in]  matrix Input matrix
 * @param[out] result Result matrix (must have swapped dimensions capacity)
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_SIZE_EXCEEDED result buffer too small
 *
 * @pre matrix != NULL && result != NULL
 * @pre result->max_rows >= matrix->cols
 * @pre result->max_cols >= matrix->rows
 *
 * @post result->rows == matrix->cols
 * @post result->cols == matrix->rows
 * @post result[i][j] == matrix[j][i] for all i,j
 */
extern MatrixStatus_Type Matrix_Transpose(const Matrix_Type* const matrix,
                                           Matrix_Type* const result);

/*==============================================================================
 * ADVANCED OPERATIONS - ITERATIVE, NO RECURSION
 *============================================================================*/

/**
 * @brief Calculate determinant using LU decomposition (iterative, no recursion)
 *
 * @param[in]  matrix Input matrix (must be square)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    matrix is not square
 * @retval MATRIX_ERROR_SIZE_EXCEEDED matrix size > 8x8
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 *
 * @pre matrix != NULL && det != NULL
 * @pre matrix->rows == matrix->cols (square matrix)
 * @pre matrix->rows <= MATRIX_MAX_ROWS
 *
 * Supports matrices up to 8x8
 */
extern MatrixStatus_Type Matrix_Determinant(const Matrix_Type* const matrix,
                                             MatrixFloat* const det);

/**
 * @brief Calculate determinant of 2x2 matrix (optimized)
 *
 * @param[in]  matrix Input matrix (must be 2x2)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 2x2
 *
 * Formula: det = a11*a22 - a12*a21
 */
extern MatrixStatus_Type Matrix_Determinant2x2(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate determinant of 3x3 matrix (optimized)
 *
 * @param[in]  matrix Input matrix (must be 3x3)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 3x3
 */
extern MatrixStatus_Type Matrix_Determinant3x3(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate determinant of 4x4 matrix (optimized)
 *
 * @param[in]  matrix Input matrix (must be 4x4)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 4x4
 */
extern MatrixStatus_Type Matrix_Determinant4x4(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate determinant of 5x5 matrix using LU decomposition
 *
 * @param[in]  matrix Input matrix (must be 5x5)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 5x5
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 */
extern MatrixStatus_Type Matrix_Determinant5x5(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate determinant of 6x6 matrix using LU decomposition
 *
 * @param[in]  matrix Input matrix (must be 6x6)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 6x6
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 */
extern MatrixStatus_Type Matrix_Determinant6x6(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate determinant of 7x7 matrix using LU decomposition
 *
 * @param[in]  matrix Input matrix (must be 7x7)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 7x7
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 */
extern MatrixStatus_Type Matrix_Determinant7x7(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate determinant of 8x8 matrix using LU decomposition
 *
 * @param[in]  matrix Input matrix (must be 8x8)
 * @param[out] det    Pointer to store determinant value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or det is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 8x8
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 */
extern MatrixStatus_Type Matrix_Determinant8x8(const Matrix_Type* const matrix,
                                                MatrixFloat* const det);

/**
 * @brief Calculate inverse using Gauss-Jordan elimination (iterative, no recursion)
 *
 * @param[in]  matrix Input matrix (must be square and invertible)
 * @param[out] result Result matrix to store inverse
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    matrix is not square
 * @retval MATRIX_ERROR_SIZE_EXCEEDED result buffer too small
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular (not invertible)
 *
 * @pre matrix != NULL && result != NULL
 * @pre matrix->rows == matrix->cols (square matrix)
 * @pre result->max_rows >= matrix->rows
 * @pre result->max_cols >= matrix->cols
 *
 * @post result * matrix = I (within numerical tolerance)
 */
extern MatrixStatus_Type Matrix_Inverse(const Matrix_Type* const matrix,
                                         Matrix_Type* const result);

/**
 * @brief Calculate inverse of 2x2 matrix (optimized)
 *
 * @param[in]  matrix Input matrix (must be 2x2 and invertible)
 * @param[out] result Result matrix to store inverse
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 2x2
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 *
 * Formula: inv(A) = (1/det) * [ a22 -a12; -a21 a11 ]
 */
extern MatrixStatus_Type Matrix_Inverse2x2(const Matrix_Type* const matrix,
                                            Matrix_Type* const result);

/**
 * @brief Calculate inverse of 3x3 matrix (optimized)
 *
 * @param[in]  matrix Input matrix (must be 3x3 and invertible)
 * @param[out] result Result matrix to store inverse
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 3x3
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 */
extern MatrixStatus_Type Matrix_Inverse3x3(const Matrix_Type* const matrix,
                                            Matrix_Type* const result);

/**
 * @brief Calculate inverse of 4x4 matrix (optimized)
 *
 * @param[in]  matrix Input matrix (must be 4x4 and invertible)
 * @param[out] result Result matrix to store inverse
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or result is NULL
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH matrix is not 4x4
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 */
extern MatrixStatus_Type Matrix_Inverse4x4(const Matrix_Type* const matrix,
                                            Matrix_Type* const result);

/**
 * @brief Calculate eigenvalues using iterative Jacobi method (no recursion)
 *
 * @param[in,out] matrix Input symmetric matrix (modified during computation)
 * @param[out]    eigen  Structure to store eigenvalues and eigenvectors
 * @param[in]     max_iterations Maximum number of Jacobi rotations
 * @param[in]     tolerance Convergence tolerance
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or eigen is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    matrix is not square
 * @retval MATRIX_ERROR_MAX_ITERATIONS Maximum iterations reached without convergence
 *
 * @pre matrix != NULL && eigen != NULL
 * @pre matrix->rows == matrix->cols (square matrix)
 * @pre matrix->rows <= MATRIX_MAX_ROWS
 * @pre max_iterations > 0U
 *
 * @note Input matrix should be symmetric for real eigenvalues
 */
extern MatrixStatus_Type Matrix_Eigenvalues(Matrix_Type* const matrix,
                                            MatrixEigen_Type* const eigen,
                                            const uint32_T max_iterations,
                                            const MatrixFloat tolerance);

/**
 * @brief Calculate eigenvalues only (faster, no eigenvectors)
 *
 * @param[in,out] matrix Input symmetric matrix (modified during computation)
 * @param[out]    eigenvalues Array to store eigenvalues (size must be at least matrix->rows)
 * @param[in]     max_iterations Maximum number of Jacobi rotations
 * @param[in]     tolerance Convergence tolerance
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or eigenvalues is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    matrix is not square
 * @retval MATRIX_ERROR_MAX_ITERATIONS Maximum iterations reached
 */
extern MatrixStatus_Type Matrix_EigenvaluesOnly(Matrix_Type* const matrix,
                                                MatrixFloat* const eigenvalues,
                                                const uint32_T max_iterations,
                                                const MatrixFloat tolerance);

/**
 * @brief LU decomposition with partial pivoting (iterative)
 *
 * @param[in,out] matrix Input matrix, overwritten with L+U
 * @param[out]    pivot  Pivot indices array (size must be at least matrix->rows)
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or pivot is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    matrix is not square
 * @retval MATRIX_ERROR_SINGULAR      matrix is singular
 *
 * @post matrix contains L (strict lower triangular) and U (upper triangular)
 * @post L has ones on diagonal (not stored)
 */
extern MatrixStatus_Type Matrix_LU(Matrix_Type* const matrix,
                                    uint32_T* const pivot);

/**
 * @brief Solve linear system using LU decomposition
 *
 * @param[in]  a Coefficient matrix A (must be square)
 * @param[in]  b Right-hand side matrix/vector
 * @param[out] x Solution matrix/vector
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      a, b, or x is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    a is not square
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH a->rows != b->rows
 * @retval MATRIX_ERROR_SIZE_EXCEEDED x buffer too small
 * @retval MATRIX_ERROR_SINGULAR      a is singular
 */
extern MatrixStatus_Type Matrix_Solve(const Matrix_Type* const a,
                                       const Matrix_Type* const b,
                                       Matrix_Type* const x);

/**
 * @brief Solve using Gauss-Jordan elimination (iterative)
 *
 * @param[in]  a Coefficient matrix A (must be square)
 * @param[in]  b Right-hand side matrix/vector
 * @param[out] x Solution matrix/vector
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      a, b, or x is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    a is not square
 * @retval MATRIX_ERROR_DIMENSION_MISMATCH a->rows != b->rows
 * @retval MATRIX_ERROR_SIZE_EXCEEDED x buffer too small
 * @retval MATRIX_ERROR_SINGULAR      a is singular
 *
 * @note This is a direct solver, good for small matrices (<= 8x8)
 */
extern MatrixStatus_Type Matrix_SolveGaussJordan(const Matrix_Type* const a,
                                                  const Matrix_Type* const b,
                                                  Matrix_Type* const x);

/*==============================================================================
 * MATRIX PROPERTIES
 *============================================================================*/

/**
 * @brief Check if matrix is square
 *
 * @param[in] matrix Pointer to matrix structure
 * @return TRUE if square, FALSE otherwise
 *
 * @retval TRUE  matrix->rows == matrix->cols
 * @retval FALSE matrix is NULL or not square
 */
extern boolean_T Matrix_IsSquare(const Matrix_Type* const matrix);

/**
 * @brief Check if matrix is symmetric within tolerance
 *
 * @param[in] matrix    Pointer to matrix structure
 * @param[in] tolerance Maximum allowed element-wise difference (float)
 * @return TRUE if symmetric, FALSE otherwise
 *
 * @retval TRUE  |a[i][j] - a[j][i]| <= tolerance for all i,j
 * @retval FALSE matrix is NULL, not square, or asymmetric
 */
extern boolean_T Matrix_IsSymmetric(const Matrix_Type* const matrix,
                                     const MatrixFloat tolerance);

/**
 * @brief Calculate trace of matrix (sum of diagonal elements)
 *
 * @param[in]  matrix Input matrix (must be square)
 * @param[out] trace  Pointer to store trace value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or trace is NULL
 * @retval MATRIX_ERROR_NOT_SQUARE    matrix is not square
 */
extern MatrixStatus_Type Matrix_Trace(const Matrix_Type* const matrix,
                                       MatrixFloat* const trace);

/**
 * @brief Calculate Frobenius norm of matrix
 *
 * @param[in]  matrix Input matrix
 * @param[out] norm   Pointer to store norm value
 * @return MATRIX_SUCCESS on success, error code otherwise
 *
 * @retval MATRIX_SUCCESS             Operation successful
 * @retval MATRIX_ERROR_NULL_PTR      matrix or norm is NULL
 *
 * Formula: norm = sqrt(sum(i,j) a[i][j]^2)
 */
extern MatrixStatus_Type Matrix_NormFrobenius(const Matrix_Type* const matrix,
                                               MatrixFloat* const norm);

/*==============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * @brief Check if two matrices are equal within tolerance
 *
 * @param[in] a         First matrix
 * @param[in] b         Second matrix
 * @param[in] tolerance Maximum allowed element-wise difference (float)
 * @return TRUE if equal, FALSE otherwise
 *
 * @retval TRUE  All |a[i][j] - b[i][j]| <= tolerance
 * @retval FALSE Matrices differ or any parameter is NULL
 */
extern boolean_T Matrix_IsEqual(const Matrix_Type* const a,
                                 const Matrix_Type* const b,
                                 const MatrixFloat tolerance);

/**
 * @brief Get matrix dimensions
 *
 * @param[in]  matrix Pointer to matrix structure
 * @param[out] rows   Pointer to store number of rows
 * @param[out] cols   Pointer to store number of columns
 */
extern void Matrix_GetDimensions(const Matrix_Type* const matrix,
                                  uint32_T* const rows,
                                  uint32_T* const cols);

/**
 * @brief Fill matrix with constant Q31 value
 *
 * @param[in,out] matrix Pointer to matrix structure
 * @param[in]     value  Q31 value to fill
 */
extern void Matrix_Fill(Matrix_Type* const matrix,
                        const MatrixElement value);

/**
 * @brief Fill matrix with constant float value (auto-converted to Q31)
 *
 * @param[in,out] matrix Pointer to matrix structure
 * @param[in]     value  Float value to fill (clamped to [-1.0, 1.0])
 */
extern void Matrix_FillFloat(Matrix_Type* const matrix,
                             const MatrixFloat value);

/*==============================================================================
 * CONVERSION HELPERS
 *============================================================================*/

/**
 * @brief Convert float to Q31 fixed-point
 *
 * @param[in] value Float value (clamped to range [-1.0, 1.0])
 * @return Q31 fixed-point value
 *
 * Formula: result = value * 2^31
 */
extern MatrixElement Matrix_FloatToQ31(const MatrixFloat value);

/**
 * @brief Convert Q31 fixed-point to float
 *
 * @param[in] value Q31 fixed-point value
 * @return Float value in range [-1.0, 1.0]
 *
 * Formula: result = value / 2^31
 */
extern MatrixFloat Matrix_Q31ToFloat(const MatrixElement value);

#endif /* MATRIX_H */