/*
 * Matrix.c
 * ========
 * 32-bit Linear Algebra Library Implementation
 *
 * Version: 5.0.0
 */

#include "Matrix.h"
#include <string.h>
#include <math.h>

/*==============================================================================
 * PRIVATE CONSTANTS
 *============================================================================*/

#define Q31_ONE          ((int32_T)0x7FFFFFFF)
#define Q31_HALF         ((int32_T)0x40000000)
#define Q31_QUARTER      ((int32_T)0x20000000)
#define Q31_EPSILON      ((int32_T)0x00000001)
#define Q31_ZERO         ((int32_T)0x00000000)
#define Q31_MINUS_ONE    ((int32_T)0x80000000)

/* Q31 scale: 2^31 — used for float<->Q31 conversions */
#define Q31_SCALE_F      (2147483648.0f)
#define Q31_SCALE_D      (2147483648.0)
/* Q31 scale as a signed 64-bit integer — used in MulQ31 / DivQ31.
 * The LL suffix ensures the constant is typed as long long (int64_T)
 * on all targets, avoiding implementation-defined behaviour when
 * 2147483648 exceeds the range of 32-bit int (MISRA Rule 7.2).     */
#define Q31_SCALE_I      (2147483648LL)

#define ZERO_THRESHOLD_Q31   ((int32_T)0x00000100)
#define ZERO_THRESHOLD_FLOAT (1.0e-6f)
#define JACOBI_TOLERANCE     (1.0e-6f)
#define JACOBI_MAX_ITER      (50U)
#define JACOBI_PI_OVER_4     (0.78539816339f)

/*==============================================================================
 * PRIVATE MACROS
 *============================================================================*/

#define MATRIX_INDEX(m, r, c)   (((r) * (m)->stride) + (c))
#define IS_ZERO_Q31(x)          (((x) < ZERO_THRESHOLD_Q31) && ((x) > -ZERO_THRESHOLD_Q31))
#define IS_ZERO_FLOAT(x)        (((x) < ZERO_THRESHOLD_FLOAT) && ((x) > -ZERO_THRESHOLD_FLOAT))
#define ABS_FLOAT(x)            (((x) >= 0.0f) ? (x) : -(x))

/*==============================================================================
 * PRIVATE FUNCTIONS - FIXED-POINT ARITHMETIC
 *============================================================================*/

static MatrixElement Matrix_MulQ31(const MatrixElement a, const MatrixElement b) {
    int64_T temp;
    MatrixElement result;

    /*
     * Multiply two Q31 values: result = (a * b) / 2^31
     * Division by Q31_SCALE_I (2147483648LL) is used instead of >> 31 because
     * right-shift of a signed integer is implementation-defined
     * (MISRA C:2012 Rule 12.2).  Integer division of a signed type
     * is well-defined for all values on two's-complement targets.
     */
    temp = ((int64_T)a * (int64_T)b) / (int64_T)Q31_SCALE_I;

    if (temp > (int64_T)Q31_ONE) {
        result = Q31_ONE;
    } else if (temp < (int64_T)Q31_MINUS_ONE) {
        result = Q31_MINUS_ONE;
    } else {
        result = (MatrixElement)temp;
    }

    return result;
}

static MatrixElement Matrix_DivQ31(const MatrixElement a, const MatrixElement b) {
    int64_T temp;
    MatrixElement result;

    if (b == Q31_ZERO) {
        result = Q31_ZERO;
    } else {
        /*
         * Divide two Q31 values: result = (a * 2^31) / b
         * Multiplication by Q31_SCALE_I (2147483648LL) is used instead of << 31
         * because left-shift of a signed integer is implementation-defined
         * (MISRA C:2012 Rule 12.2).
         */
        temp = ((int64_T)a * (int64_T)Q31_SCALE_I) / (int64_T)b;

        if (temp > (int64_T)Q31_ONE) {
            result = Q31_ONE;
        } else if (temp < (int64_T)Q31_MINUS_ONE) {
            result = Q31_MINUS_ONE;
        } else {
            result = (MatrixElement)temp;
        }
    }

    return result;
}

static MatrixElement Matrix_AbsQ31(const MatrixElement x) {
    MatrixElement result;

    if (x < (MatrixElement)0) {
        if (x == Q31_MINUS_ONE) {
            result = Q31_ONE;
        } else {
            result = -x;
        }
    } else {
        result = x;
    }

    return result;
}

/*==============================================================================
 * PRIVATE FUNCTIONS - VALIDATION
 *============================================================================*/

static boolean_T Matrix_IsValidIndex(const Matrix_Type* const matrix,
                             const uint32_T row,
                             const uint32_T col) {
    boolean_T result;

    result = TRUE;

    if (matrix == NULL) {
        result = FALSE;
    } else if (row >= matrix->rows) {
        result = FALSE;
    } else if (col >= matrix->cols) {
        result = FALSE;
    } else {
        /* No action */
    }

    return result;
}

static MatrixStatus_Type Matrix_CheckAddSub(const Matrix_Type* const a,
                                      const Matrix_Type* const b,
                                      const Matrix_Type* const result) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((a == NULL) || (b == NULL) || (result == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((a->rows != b->rows) || (a->cols != b->cols)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else if ((result->max_rows < a->rows) || (result->max_cols < a->cols)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        /* No action */
    }

    return status;
}

static MatrixStatus_Type Matrix_CheckMultiply(const Matrix_Type* const a,
                                        const Matrix_Type* const b,
                                        const Matrix_Type* const result) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((a == NULL) || (b == NULL) || (result == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (a->cols != b->rows) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else if ((result->max_rows < a->rows) || (result->max_cols < b->cols)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        /* No action */
    }

    return status;
}

/*==============================================================================
 * INITIALIZATION FUNCTIONS
 *============================================================================*/

void Matrix_Init(Matrix_Type* const matrix,
                MatrixElement* const buffer,
                const uint32_T max_rows,
                const uint32_T max_cols) {
    if ((matrix != NULL) && (buffer != NULL)) {
        matrix->data = buffer;
        matrix->max_rows = max_rows;
        matrix->max_cols = max_cols;
        matrix->rows = max_rows;
        matrix->cols = max_cols;
        matrix->is_view = FALSE;
        matrix->stride = max_cols;

        (void)memset(buffer, 0,
                    (size_t)max_rows * (size_t)max_cols * sizeof(MatrixElement));
    } else {
        /* No action */
    }
}

void Matrix_SetDimensions(Matrix_Type* const matrix,
                         const uint32_T rows,
                         const uint32_T cols) {
    if (matrix != NULL) {
        if ((rows <= matrix->max_rows) && (cols <= matrix->max_cols) &&
            (rows > 0U) && (cols > 0U)) {
            matrix->rows = rows;
            matrix->cols = cols;
        } else {
            /* No action */
        }
    } else {
        /* No action */
    }
}

void Matrix_Zero(Matrix_Type* const matrix) {
    uint32_T row;
    uint32_T col;

    if (matrix != NULL) {
        for (row = 0U; row < matrix->rows; row++) {
            for (col = 0U; col < matrix->cols; col++) {
                matrix->data[MATRIX_INDEX(matrix, row, col)] = Q31_ZERO;
            }
        }
    } else {
        /* No action */
    }
}

MatrixStatus_Type Matrix_Identity(Matrix_Type* const matrix) {
    MatrixStatus_Type status;

    uint32_T row;
    uint32_T col;

    status = MATRIX_SUCCESS;

    if (matrix == NULL) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (matrix->rows != matrix->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else {
        for (row = 0U; row < matrix->rows; row++) {
            for (col = 0U; col < matrix->cols; col++) {
                if (row == col) {
                    matrix->data[MATRIX_INDEX(matrix, row, col)] = Q31_ONE;
                } else {
                    matrix->data[MATRIX_INDEX(matrix, row, col)] = Q31_ZERO;
                }
            }
        }
    }

    return status;
}

MatrixStatus_Type Matrix_Copy(Matrix_Type* const dest,
                              const Matrix_Type* const src) {
    MatrixStatus_Type status;

    uint32_T row;
    uint32_T col;

    status = MATRIX_SUCCESS;

    if ((dest == NULL) || (src == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((dest->max_rows < src->rows) || (dest->max_cols < src->cols)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        for (row = 0U; row < src->rows; row++) {
            for (col = 0U; col < src->cols; col++) {
                dest->data[MATRIX_INDEX(dest, row, col)] =
                    src->data[MATRIX_INDEX(src, row, col)];
            }
        }

        dest->rows = src->rows;
        dest->cols = src->cols;
    }

    return status;
}

/*==============================================================================
 * ELEMENT ACCESS FUNCTIONS
 *============================================================================*/

MatrixStatus_Type Matrix_SetElement(Matrix_Type* const matrix,
                                   const uint32_T row,
                                   const uint32_T col,
                                   const MatrixElement value) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if (matrix == NULL) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (Matrix_IsValidIndex(matrix, row, col) == FALSE) {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    } else {
        matrix->data[MATRIX_INDEX(matrix, row, col)] = value;
    }

    return status;
}

MatrixStatus_Type Matrix_SetElementFloat(Matrix_Type* const matrix,
                                        const uint32_T row,
                                        const uint32_T col,
                                        const MatrixFloat value) {
    MatrixStatus_Type status;

    MatrixElement q31_value;

    q31_value = Matrix_FloatToQ31(value);
    status = Matrix_SetElement(matrix, row, col, q31_value);

    return status;
}

MatrixStatus_Type Matrix_GetElement(const Matrix_Type* const matrix,
                                   const uint32_T row,
                                   const uint32_T col,
                                   MatrixElement* const value) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (value == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (Matrix_IsValidIndex(matrix, row, col) == FALSE) {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    } else {
        *value = matrix->data[MATRIX_INDEX(matrix, row, col)];
    }

    return status;
}

MatrixStatus_Type Matrix_GetElementFloat(const Matrix_Type* const matrix,
                                        const uint32_T row,
                                        const uint32_T col,
                                        MatrixFloat* const value) {
    MatrixStatus_Type status;

    MatrixElement q31_value;

    status = Matrix_GetElement(matrix, row, col, &q31_value);

    if (status == MATRIX_SUCCESS) {
        *value = Matrix_Q31ToFloat(q31_value);
    } else {
        /* No action */
    }

    return status;
}

/*==============================================================================
 * BASIC OPERATIONS
 *============================================================================*/

MatrixStatus_Type Matrix_Add(const Matrix_Type* const a,
                             const Matrix_Type* const b,
                             Matrix_Type* const result) {
    MatrixStatus_Type status;
    uint32_T row;
    uint32_T col;
    MatrixElement sum;

    status = Matrix_CheckAddSub(a, b, result);

    if (status == MATRIX_SUCCESS) {
        for (row = 0U; row < a->rows; row++) {
            for (col = 0U; col < a->cols; col++) {
                sum = a->data[MATRIX_INDEX(a, row, col)] +
                      b->data[MATRIX_INDEX(b, row, col)];

                if (((a->data[MATRIX_INDEX(a, row, col)] > 0) &&
                     (b->data[MATRIX_INDEX(b, row, col)] > 0) &&
                     (sum < 0)) ||
                    ((a->data[MATRIX_INDEX(a, row, col)] < 0) &&
                     (b->data[MATRIX_INDEX(b, row, col)] < 0) &&
                     (sum > 0))) {

                    if (a->data[MATRIX_INDEX(a, row, col)] > 0) {
                        sum = Q31_ONE;
                    } else {
                        sum = Q31_MINUS_ONE;
                    }
                } else {
                    /* No action */
                }

                result->data[MATRIX_INDEX(result, row, col)] = sum;
            }
        }

        result->rows = a->rows;
        result->cols = a->cols;
    } else {
        /* No action */
    }

    return status;
}

MatrixStatus_Type Matrix_Subtract(const Matrix_Type* const a,
                                  const Matrix_Type* const b,
                                  Matrix_Type* const result) {
    MatrixStatus_Type status;
    uint32_T row;
    uint32_T col;
    MatrixElement diff;

    status = Matrix_CheckAddSub(a, b, result);

    if (status == MATRIX_SUCCESS) {
        for (row = 0U; row < a->rows; row++) {
            for (col = 0U; col < a->cols; col++) {
                diff = a->data[MATRIX_INDEX(a, row, col)] -
                       b->data[MATRIX_INDEX(b, row, col)];

                if (((a->data[MATRIX_INDEX(a, row, col)] > 0) &&
                     (b->data[MATRIX_INDEX(b, row, col)] < 0) &&
                     (diff < 0)) ||
                    ((a->data[MATRIX_INDEX(a, row, col)] < 0) &&
                     (b->data[MATRIX_INDEX(b, row, col)] > 0) &&
                     (diff > 0))) {

                    if (a->data[MATRIX_INDEX(a, row, col)] > 0) {
                        diff = Q31_ONE;
                    } else {
                        diff = Q31_MINUS_ONE;
                    }
                } else {
                    /* No action */
                }

                result->data[MATRIX_INDEX(result, row, col)] = diff;
            }
        }

        result->rows = a->rows;
        result->cols = a->cols;
    } else {
        /* No action */
    }

    return status;
}

MatrixStatus_Type Matrix_Multiply(const Matrix_Type* const a,
                                  const Matrix_Type* const b,
                                  Matrix_Type* const result) {
    MatrixStatus_Type status;
    uint32_T i;
    uint32_T j;
    uint32_T k;
    MatrixElement sum;
    boolean_T saturated;

    status = Matrix_CheckMultiply(a, b, result);

    if (status == MATRIX_SUCCESS) {
        for (i = 0U; i < a->rows; i++) {
            for (j = 0U; j < b->cols; j++) {
                saturated = FALSE;
                sum = Q31_ZERO;

                for (k = 0U; (k < a->cols) && (saturated == FALSE); k++) {
                    sum += Matrix_MulQ31(a->data[MATRIX_INDEX(a, i, k)],
                                   b->data[MATRIX_INDEX(b, k, j)]);

                    if (sum > Q31_ONE) {
                        sum = Q31_ONE;
                        saturated = TRUE;
                    } else if (sum < Q31_MINUS_ONE) {
                        sum = Q31_MINUS_ONE;
                        saturated = TRUE;
                    } else {
                        /* No action */
                    }
                }

                result->data[MATRIX_INDEX(result, i, j)] = sum;
            }
        }

        result->rows = a->rows;
        result->cols = b->cols;
    } else {
        /* No action */
    }

    return status;
}

MatrixStatus_Type Matrix_ScalarMultiply(const Matrix_Type* const matrix,
                                        const MatrixElement scalar,
                                        Matrix_Type* const result) {
    MatrixStatus_Type status;

    uint32_T row;
    uint32_T col;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (result == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((result->max_rows < matrix->rows) ||
               (result->max_cols < matrix->cols)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        for (row = 0U; row < matrix->rows; row++) {
            for (col = 0U; col < matrix->cols; col++) {
                result->data[MATRIX_INDEX(result, row, col)] =
                    Matrix_MulQ31(matrix->data[MATRIX_INDEX(matrix, row, col)], scalar);
            }
        }

        result->rows = matrix->rows;
        result->cols = matrix->cols;
    }

    return status;
}

MatrixStatus_Type Matrix_ScalarMultiplyFloat(const Matrix_Type* const matrix,
                                            const MatrixFloat scalar,
                                            Matrix_Type* const result) {
    return Matrix_ScalarMultiply(matrix, Matrix_FloatToQ31(scalar), result);
}

MatrixStatus_Type Matrix_Transpose(const Matrix_Type* const matrix,
                                   Matrix_Type* const result) {
    MatrixStatus_Type status;

    uint32_T row;
    uint32_T col;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (result == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((result->max_rows < matrix->cols) ||
               (result->max_cols < matrix->rows)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        for (row = 0U; row < matrix->rows; row++) {
            for (col = 0U; col < matrix->cols; col++) {
                result->data[MATRIX_INDEX(result, col, row)] =
                    matrix->data[MATRIX_INDEX(matrix, row, col)];
            }
        }

        result->rows = matrix->cols;
        result->cols = matrix->rows;
    }

    return status;
}

/*==============================================================================
 * IMPROVED DETERMINANT FUNCTIONS - Using double precision for accuracy
 *============================================================================*/

MatrixStatus_Type Matrix_Determinant2x2(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 2U) || (matrix->cols != 2U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        real64_T a11;
        real64_T a12;
        real64_T a21;
        real64_T a22;

        /* Use double precision for better accuracy */
        a11 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 0U, 0U)] / Q31_SCALE_D;
        a12 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 0U, 1U)] / Q31_SCALE_D;
        a21 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 1U, 0U)] / Q31_SCALE_D;
        a22 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 1U, 1U)] / Q31_SCALE_D;

        *det = (MatrixFloat)((a11 * a22) - (a12 * a21));
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant3x3(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 3U) || (matrix->cols != 3U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        uint32_T i;
        uint32_T j;
        real64_T m[3][3];
        real64_T term1;
        real64_T term2;
        real64_T term3;

        /* Convert to double precision */
        for (i = 0U; i < 3U; i++) {
            for (j = 0U; j < 3U; j++) {
                m[i][j] = (real64_T)matrix->data[MATRIX_INDEX(matrix, i, j)] / Q31_SCALE_D;
            }
        }

        /* Calculate determinant with better numerical stability */
        term1 = m[0][0] * ((m[1][1] * m[2][2]) - (m[1][2] * m[2][1]));
        term2 = m[0][1] * ((m[1][0] * m[2][2]) - (m[1][2] * m[2][0]));
        term3 = m[0][2] * ((m[1][0] * m[2][1]) - (m[1][1] * m[2][0]));

        *det = (MatrixFloat)(term1 - term2 + term3);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant4x4(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 4U) || (matrix->cols != 4U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        uint32_T i;
        uint32_T j;
        real64_T m[4][4];
        real64_T term1;
        real64_T term2;
        real64_T term3;
        real64_T term4;

        /* Convert to double precision */
        for (i = 0U; i < 4U; i++) {
            for (j = 0U; j < 4U; j++) {
                m[i][j] = (real64_T)matrix->data[MATRIX_INDEX(matrix, i, j)] / Q31_SCALE_D;
            }
        }

        /* Direct formula for 4x4 determinant */
        term1 = m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0] -
                m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] * m[2][2] * m[3][0] +
                m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] * m[1][2] * m[2][3] * m[3][0];

        term2 = -m[0][3] * m[1][2] * m[2][0] * m[3][1] + m[0][2] * m[1][3] * m[2][0] * m[3][1] +
                 m[0][3] * m[1][0] * m[2][2] * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1] -
                 m[0][2] * m[1][0] * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1];

        term3 = m[0][3] * m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2] -
                m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] * m[3][2] +
                m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] * m[2][3] * m[3][2];

        term4 = -m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] * m[1][2] * m[2][0] * m[3][3] +
                 m[0][2] * m[1][0] * m[2][1] * m[3][3] - m[0][0] * m[1][2] * m[2][1] * m[3][3] -
                 m[0][1] * m[1][0] * m[2][2] * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3];

        *det = (MatrixFloat)(term1 + term2 + term3 + term4);
    }

    return status;
}

/**
 * @brief Helper function for determinant using LU decomposition
 */
static MatrixStatus_Type Matrix_DeterminantLU(Matrix_Type* const matrix,
                                       MatrixFloat* const det) {
    MatrixStatus_Type status;
    uint32_T i;
    uint32_T n;
    uint32_T pivot[MATRIX_MAX_ROWS];
    int32_T sign;
    real64_T diag_product;

    sign = 1;
    diag_product = 1.0;

    n = matrix->rows;

    status = Matrix_LU(matrix, pivot);

    if (status == MATRIX_SUCCESS) {
        for (i = 0U; i < n; i++) {
            if (pivot[i] != i) {
                sign = -sign;
            } else {
                /* No action */
            }
        }

        for (i = 0U; i < n; i++) {
            diag_product *= (real64_T)matrix->data[MATRIX_INDEX(matrix, i, i)] / Q31_SCALE_D;
        }

        *det = (MatrixFloat)((real64_T)sign * diag_product);
    } else {
        *det = 0.0f;
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant5x5(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;
    MatrixElement work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type work;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 5U) || (matrix->cols != 5U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        Matrix_Init(&work, work_buffer, 5U, 5U);
        (void)Matrix_Copy(&work, matrix);
        status = Matrix_DeterminantLU(&work, det);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant6x6(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;
    MatrixElement work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type work;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 6U) || (matrix->cols != 6U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        Matrix_Init(&work, work_buffer, 6U, 6U);
        (void)Matrix_Copy(&work, matrix);
        status = Matrix_DeterminantLU(&work, det);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant7x7(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;
    MatrixElement work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type work;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 7U) || (matrix->cols != 7U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        Matrix_Init(&work, work_buffer, 7U, 7U);
        (void)Matrix_Copy(&work, matrix);
        status = Matrix_DeterminantLU(&work, det);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant8x8(const Matrix_Type* const matrix,
                                        MatrixFloat* const det) {
    MatrixStatus_Type status;
    MatrixElement work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type work;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 8U) || (matrix->cols != 8U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else {
        Matrix_Init(&work, work_buffer, 8U, 8U);
        (void)Matrix_Copy(&work, matrix);
        status = Matrix_DeterminantLU(&work, det);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant(const Matrix_Type* const matrix,
                                     MatrixFloat* const det) {
    MatrixStatus_Type status;

    uint32_T n;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (det == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (matrix->rows != matrix->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else {
        n = matrix->rows;

        if (n == 1U) {
            *det = Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, 0U, 0U)]);
        } else if (n == 2U) {
            status = Matrix_Determinant2x2(matrix, det);
        } else if (n == 3U) {
            status = Matrix_Determinant3x3(matrix, det);
        } else if (n == 4U) {
            status = Matrix_Determinant4x4(matrix, det);
        } else if (n == 5U) {
            status = Matrix_Determinant5x5(matrix, det);
        } else if (n == 6U) {
            status = Matrix_Determinant6x6(matrix, det);
        } else if (n == 7U) {
            status = Matrix_Determinant7x7(matrix, det);
        } else if (n == 8U) {
            status = Matrix_Determinant8x8(matrix, det);
        } else {
            status = MATRIX_ERROR_SIZE_EXCEEDED;
        }
    }

    return status;
}

/*==============================================================================
 * IMPROVED INVERSE FUNCTIONS
 *============================================================================*/

MatrixStatus_Type Matrix_Inverse2x2(const Matrix_Type* const matrix,
                                    Matrix_Type* const result) {
    MatrixStatus_Type status;
    MatrixFloat det;
    real64_T a11;
    real64_T a12;
    real64_T a21;
    real64_T a22;
    real64_T inv_det;

    status = Matrix_Determinant2x2(matrix, &det);

    if (status == MATRIX_SUCCESS) {
        if (ABS_FLOAT(det) < ZERO_THRESHOLD_FLOAT) {
            status = MATRIX_ERROR_SINGULAR;
        } else {
            inv_det = 1.0 / (real64_T)det;

            /* Use double precision for inverse calculation */
            a11 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 0U, 0U)] / Q31_SCALE_D;
            a12 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 0U, 1U)] / Q31_SCALE_D;
            a21 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 1U, 0U)] / Q31_SCALE_D;
            a22 = (real64_T)matrix->data[MATRIX_INDEX(matrix, 1U, 1U)] / Q31_SCALE_D;

            result->data[MATRIX_INDEX(result, 0U, 0U)] = Matrix_FloatToQ31((MatrixFloat)( a22 * inv_det));
            result->data[MATRIX_INDEX(result, 0U, 1U)] = Matrix_FloatToQ31((MatrixFloat)(-a12 * inv_det));
            result->data[MATRIX_INDEX(result, 1U, 0U)] = Matrix_FloatToQ31((MatrixFloat)(-a21 * inv_det));
            result->data[MATRIX_INDEX(result, 1U, 1U)] = Matrix_FloatToQ31((MatrixFloat)( a11 * inv_det));

            result->rows = 2U;
            result->cols = 2U;
        }
    } else {
        /* No action */
    }

    return status;
}

MatrixStatus_Type Matrix_Inverse3x3(const Matrix_Type* const matrix,
                                    Matrix_Type* const result) {
    MatrixStatus_Type status;
    MatrixFloat det;
    real64_T m[3][3];
    real64_T c[3][3];
    real64_T inv_det;
    uint32_T i;
    uint32_T j;

    status = Matrix_Determinant3x3(matrix, &det);

    if (status == MATRIX_SUCCESS) {
        if (ABS_FLOAT(det) < ZERO_THRESHOLD_FLOAT) {
            status = MATRIX_ERROR_SINGULAR;
        } else {
            inv_det = 1.0 / (real64_T)det;

            /* Convert to double precision */
            for (i = 0U; i < 3U; i++) {
                for (j = 0U; j < 3U; j++) {
                    m[i][j] = (real64_T)matrix->data[MATRIX_INDEX(matrix, i, j)] / Q31_SCALE_D;
                }
            }

            /* Calculate cofactors with double precision */
            c[0][0] =  (m[1][1] * m[2][2] - m[1][2] * m[2][1]);
            c[0][1] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]);
            c[0][2] =  (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

            c[1][0] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]);
            c[1][1] =  (m[0][0] * m[2][2] - m[0][2] * m[2][0]);
            c[1][2] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]);

            c[2][0] =  (m[0][1] * m[1][2] - m[0][2] * m[1][1]);
            c[2][1] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]);
            c[2][2] =  (m[0][0] * m[1][1] - m[0][1] * m[1][0]);

            /* Inverse = (1/det) * C^T */
            for (i = 0U; i < 3U; i++) {
                for (j = 0U; j < 3U; j++) {
                    result->data[MATRIX_INDEX(result, i, j)] =
                        Matrix_FloatToQ31((MatrixFloat)(c[j][i] * inv_det));
                }
            }

            result->rows = 3U;
            result->cols = 3U;
        }
    } else {
        /* No action */
    }

    return status;
}

MatrixStatus_Type Matrix_Inverse4x4(const Matrix_Type* const matrix,
                                    Matrix_Type* const result) {
    MatrixStatus_Type status;

    MatrixElement aug_buffer[4U * 8U];
    Matrix_Type aug;
    uint32_T i;
    uint32_T j;
    uint32_T k;
    uint32_T n;
    uint32_T max_row;
    MatrixElement max_val;
    MatrixElement val;
    MatrixElement temp;
    MatrixElement pivot;
    MatrixElement factor;
    boolean_T singular;

    n = 4U;
    singular = FALSE;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (result == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if ((matrix->rows != 4U) || (matrix->cols != 4U)) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else if ((result->max_rows < 4U) || (result->max_cols < 4U)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        Matrix_Init(&aug, aug_buffer, 4U, 8U);

        /* Copy A */
        for (i = 0U; i < n; i++) {
            for (j = 0U; j < n; j++) {
                aug.data[MATRIX_INDEX(&aug, i, j)] =
                    matrix->data[MATRIX_INDEX(matrix, i, j)];
            }
        }

        /* Set I */
        for (i = 0U; i < n; i++) {
            for (j = n; j < 2U * n; j++) {
                if ((j - n) == i) {
                    aug.data[MATRIX_INDEX(&aug, i, j)] = Q31_ONE;
                } else {
                    aug.data[MATRIX_INDEX(&aug, i, j)] = Q31_ZERO;
                }
            }
        }

        /* Gauss-Jordan elimination with partial pivoting */
        for (i = 0U; i < n; i++) {
            /* Find pivot */
            max_row = i;
            max_val = Matrix_AbsQ31(aug.data[MATRIX_INDEX(&aug, i, i)]);

            for (k = i + 1U; k < n; k++) {
                val = Matrix_AbsQ31(aug.data[MATRIX_INDEX(&aug, k, i)]);
                if (val > max_val) {
                    max_val = val;
                    max_row = k;
                }
            }

            if (max_val < ZERO_THRESHOLD_Q31) {
                singular = TRUE;
            } else {
                /* Swap rows if necessary */
                if (max_row != i) {
                    for (j = 0U; j < 2U * n; j++) {
                        temp = aug.data[MATRIX_INDEX(&aug, i, j)];
                        aug.data[MATRIX_INDEX(&aug, i, j)] = aug.data[MATRIX_INDEX(&aug, max_row, j)];
                        aug.data[MATRIX_INDEX(&aug, max_row, j)] = temp;
                    }
                } else {
                    /* No action */
                }

                /* Normalize pivot row */
                pivot = aug.data[MATRIX_INDEX(&aug, i, i)];
                for (j = i; j < 2U * n; j++) {
                    aug.data[MATRIX_INDEX(&aug, i, j)] =
                        Matrix_DivQ31(aug.data[MATRIX_INDEX(&aug, i, j)], pivot);
                }

                /* Eliminate other rows */
                for (k = 0U; k < n; k++) {
                    if (k != i) {
                        factor = aug.data[MATRIX_INDEX(&aug, k, i)];
                        for (j = i; j < 2U * n; j++) {
                            aug.data[MATRIX_INDEX(&aug, k, j)] -=
                                Matrix_MulQ31(factor, aug.data[MATRIX_INDEX(&aug, i, j)]);
                        }
                    } else {
                        /* No action */
                    }
                }
            }
        }

        if (singular != FALSE) {
            status = MATRIX_ERROR_SINGULAR;
        } else {
            /* Extract inverse */
            for (i = 0U; i < n; i++) {
                for (j = 0U; j < n; j++) {
                    result->data[MATRIX_INDEX(result, i, j)] =
                        aug.data[MATRIX_INDEX(&aug, i, n + j)];
                }
            }

            result->rows = n;
            result->cols = n;
        }
    }

    return status;
}

MatrixStatus_Type Matrix_Inverse(const Matrix_Type* const matrix,
                                 Matrix_Type* const result) {
    MatrixStatus_Type status;

    uint32_T n;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (result == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (matrix->rows != matrix->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else if ((result->max_rows < matrix->rows) ||
               (result->max_cols < matrix->cols)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        n = matrix->rows;

        if (n == 2U) {
            status = Matrix_Inverse2x2(matrix, result);
        } else if (n == 3U) {
            status = Matrix_Inverse3x3(matrix, result);
        } else if (n == 4U) {
            status = Matrix_Inverse4x4(matrix, result);
        } else {
            /* Generic Gauss-Jordan for 5x5 to 8x8 */
            MatrixElement aug_buffer[MATRIX_MAX_ROWS * (2U * MATRIX_MAX_COLS)];
            Matrix_Type aug;
            uint32_T i;
            uint32_T j;
            uint32_T k;
            uint32_T max_row;
            MatrixElement max_val;
            MatrixElement val;
            MatrixElement temp;
            MatrixElement pivot;
            MatrixElement factor;
            boolean_T singular;

            singular = FALSE;
            Matrix_Init(&aug, aug_buffer, n, 2U * n);

            /* Copy A */
            for (i = 0U; i < n; i++) {
                for (j = 0U; j < n; j++) {
                    aug.data[MATRIX_INDEX(&aug, i, j)] =
                        matrix->data[MATRIX_INDEX(matrix, i, j)];
                }
            }

            /* Set I */
            for (i = 0U; i < n; i++) {
                for (j = n; j < 2U * n; j++) {
                    if ((j - n) == i) {
                        aug.data[MATRIX_INDEX(&aug, i, j)] = Q31_ONE;
                    } else {
                        aug.data[MATRIX_INDEX(&aug, i, j)] = Q31_ZERO;
                    }
                }
            }

            /* Gauss-Jordan elimination with partial pivoting */
            for (i = 0U; i < n; i++) {
                /* Find pivot */
                max_row = i;
                max_val = Matrix_AbsQ31(aug.data[MATRIX_INDEX(&aug, i, i)]);

                for (k = i + 1U; k < n; k++) {
                    val = Matrix_AbsQ31(aug.data[MATRIX_INDEX(&aug, k, i)]);
                    if (val > max_val) {
                        max_val = val;
                        max_row = k;
                    }
                }

                if (max_val < ZERO_THRESHOLD_Q31) {
                    singular = TRUE;
                } else {
                    /* Swap rows if necessary */
                    if (max_row != i) {
                        for (j = 0U; j < 2U * n; j++) {
                            temp = aug.data[MATRIX_INDEX(&aug, i, j)];
                            aug.data[MATRIX_INDEX(&aug, i, j)] = aug.data[MATRIX_INDEX(&aug, max_row, j)];
                            aug.data[MATRIX_INDEX(&aug, max_row, j)] = temp;
                        }
                    } else {
                        /* No action */
                    }

                    /* Normalize pivot row */
                    pivot = aug.data[MATRIX_INDEX(&aug, i, i)];
                    for (j = i; j < 2U * n; j++) {
                        aug.data[MATRIX_INDEX(&aug, i, j)] =
                            Matrix_DivQ31(aug.data[MATRIX_INDEX(&aug, i, j)], pivot);
                    }

                    /* Eliminate other rows */
                    for (k = 0U; k < n; k++) {
                        if (k != i) {
                            factor = aug.data[MATRIX_INDEX(&aug, k, i)];
                            for (j = i; j < 2U * n; j++) {
                                aug.data[MATRIX_INDEX(&aug, k, j)] -=
                                    Matrix_MulQ31(factor, aug.data[MATRIX_INDEX(&aug, i, j)]);
                            }
                        } else {
                            /* No action */
                        }
                    }
                }
            }

            if (singular != FALSE) {
                status = MATRIX_ERROR_SINGULAR;
            } else {
                for (i = 0U; i < n; i++) {
                    for (j = 0U; j < n; j++) {
                        result->data[MATRIX_INDEX(result, i, j)] =
                            aug.data[MATRIX_INDEX(&aug, i, n + j)];
                    }
                }

                result->rows = n;
                result->cols = n;
            }
        }
    }

    return status;
}

/*==============================================================================
 * EIGENVALUE FUNCTIONS
 *============================================================================*/

MatrixStatus_Type Matrix_Eigenvalues(Matrix_Type* const matrix,
                                     MatrixEigen_Type* const eigen,
                                     const uint32_T max_iterations,
                                     const MatrixFloat tolerance) {
    MatrixStatus_Type status;

    uint32_T n;
    uint32_T iter;
    uint32_T p;
    uint32_T q;
    uint32_T i;
    MatrixFloat max_off_diag;
    MatrixFloat app;
    MatrixFloat aqq;
    MatrixFloat apq;
    MatrixFloat theta;
    MatrixFloat c;
    MatrixFloat s;
    MatrixFloat temp;
    Matrix_Type v_matrix;
    MatrixElement v_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    boolean_T use_tolerance;
    uint32_T max_iter;
    boolean_T converged_early;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (eigen == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (matrix->rows != matrix->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else {
        n = matrix->rows;
        max_iter = (max_iterations > 0U) ? max_iterations : JACOBI_MAX_ITER;
        use_tolerance = (tolerance > 0.0f) ? TRUE : FALSE;
        converged_early = FALSE;

        eigen->num_eigenvalues = n;
        eigen->iterations = 0U;

        Matrix_Init(&v_matrix, v_buffer, n, n);
        (void)Matrix_Identity(&v_matrix);

        for (iter = 0U; iter < max_iter; iter++) {
            uint32_T j;
            MatrixFloat val;
            MatrixFloat a_ip;
            MatrixFloat a_iq;
            MatrixFloat v_ip;
            MatrixFloat v_iq;
            boolean_T converged;

            eigen->iterations = iter + 1U;

            /* Find largest off-diagonal element */
            max_off_diag = 0.0f;
            p = 0U;
            q = 1U;

            for (i = 0U; i < n; i++) {
                for (j = i + 1U; j < n; j++) {
                    val = ABS_FLOAT(
                        Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, i, j)]));
                    if (val > max_off_diag) {
                        max_off_diag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if ((use_tolerance != FALSE) && (max_off_diag < tolerance)) {
                converged = TRUE;
            } else if (max_off_diag < JACOBI_TOLERANCE) {
                converged = TRUE;
            } else {
                converged = FALSE;
            }

            if (converged != FALSE) {
                converged_early = TRUE;
                iter = max_iter; /* terminate for-loop */
            } else {
                app = Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, p, p)]);
                aqq = Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, q, q)]);
                apq = Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, p, q)]);

                if (ABS_FLOAT(aqq - app) < ZERO_THRESHOLD_FLOAT) {
                    theta = JACOBI_PI_OVER_4;
                } else {
                    theta = 0.5f * atan2f(2.0f * apq, aqq - app);
                }

                c = cosf(theta);
                s = sinf(theta);

                temp = app;
                app = (c * c * temp) - (2.0f * c * s * apq) + (s * s * aqq);
                aqq = (s * s * temp) + (2.0f * c * s * apq) + (c * c * aqq);

                matrix->data[MATRIX_INDEX(matrix, p, p)] = Matrix_FloatToQ31(app);
                matrix->data[MATRIX_INDEX(matrix, q, q)] = Matrix_FloatToQ31(aqq);
                matrix->data[MATRIX_INDEX(matrix, p, q)] = Q31_ZERO;
                matrix->data[MATRIX_INDEX(matrix, q, p)] = Q31_ZERO;

                for (i = 0U; i < n; i++) {
                    if ((i != p) && (i != q)) {
                        a_ip = Matrix_Q31ToFloat(
                            matrix->data[MATRIX_INDEX(matrix, i, p)]);
                        a_iq = Matrix_Q31ToFloat(
                            matrix->data[MATRIX_INDEX(matrix, i, q)]);

                        matrix->data[MATRIX_INDEX(matrix, i, p)] =
                            Matrix_FloatToQ31((c * a_ip) - (s * a_iq));
                        matrix->data[MATRIX_INDEX(matrix, p, i)] =
                            matrix->data[MATRIX_INDEX(matrix, i, p)];
                        matrix->data[MATRIX_INDEX(matrix, i, q)] =
                            Matrix_FloatToQ31((s * a_ip) + (c * a_iq));
                        matrix->data[MATRIX_INDEX(matrix, q, i)] =
                            matrix->data[MATRIX_INDEX(matrix, i, q)];
                    } else {
                        /* No action */
                    }
                }

                for (i = 0U; i < n; i++) {
                    v_ip = Matrix_Q31ToFloat(
                        v_matrix.data[MATRIX_INDEX(&v_matrix, i, p)]);
                    v_iq = Matrix_Q31ToFloat(
                        v_matrix.data[MATRIX_INDEX(&v_matrix, i, q)]);

                    v_matrix.data[MATRIX_INDEX(&v_matrix, i, p)] =
                        Matrix_FloatToQ31((c * v_ip) - (s * v_iq));
                    v_matrix.data[MATRIX_INDEX(&v_matrix, i, q)] =
                        Matrix_FloatToQ31((s * v_ip) + (c * v_iq));
                }
            }
        }

        for (i = 0U; i < n; i++) {
            eigen->eigenvalues[i] = Matrix_Q31ToFloat(
                matrix->data[MATRIX_INDEX(matrix, i, i)]);
        }

        {
            uint32_T i2;
            uint32_T j2;
            for (i2 = 0U; i2 < n; i2++) {
                for (j2 = 0U; j2 < n; j2++) {
                    eigen->eigenvectors[(i2 * n) + j2] = Matrix_Q31ToFloat(
                        v_matrix.data[MATRIX_INDEX(&v_matrix, i2, j2)]);
                }
            }
        }

        if ((iter >= max_iter) && (converged_early == FALSE)) {
            status = MATRIX_ERROR_MAX_ITERATIONS;
        } else {
            /* No action — converged successfully */
        }
    }

    return status;
}

MatrixStatus_Type Matrix_EigenvaluesOnly(Matrix_Type* const matrix,
                                        MatrixFloat* const eigenvalues,
                                        const uint32_T max_iterations,
                                        const MatrixFloat tolerance) {
    MatrixStatus_Type status;
    MatrixEigen_Type eigen;
    uint32_T i;

    status = Matrix_Eigenvalues(matrix, &eigen, max_iterations, tolerance);

    if (status == MATRIX_SUCCESS) {
        for (i = 0U; i < eigen.num_eigenvalues; i++) {
            eigenvalues[i] = eigen.eigenvalues[i];
        }
    }

    return status;
}

/*==============================================================================
 * LU DECOMPOSITION
 *============================================================================*/

MatrixStatus_Type Matrix_LU(Matrix_Type* const matrix,
                            uint32_T* const pivot) {
    MatrixStatus_Type status;

    uint32_T i;
    uint32_T j;
    uint32_T k;
    uint32_T pivot_row;
    MatrixElement factor;
    MatrixElement temp;
    uint32_T n;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (pivot == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (matrix->rows != matrix->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else {
        n = matrix->rows;

        for (i = 0U; i < n; i++) {
            pivot[i] = i;
        }

        for (k = 0U; (k < (n - 1U)) && (status == MATRIX_SUCCESS); k++) {
            pivot_row = k;
            for (i = k + 1U; i < n; i++) {
                if (Matrix_AbsQ31(matrix->data[MATRIX_INDEX(matrix, i, k)]) >
                    Matrix_AbsQ31(matrix->data[MATRIX_INDEX(matrix, pivot_row, k)])) {
                    pivot_row = i;
                }
            }

            if (pivot_row != k) {
                for (j = 0U; j < n; j++) {
                    temp = matrix->data[MATRIX_INDEX(matrix, k, j)];
                    matrix->data[MATRIX_INDEX(matrix, k, j)] =
                        matrix->data[MATRIX_INDEX(matrix, pivot_row, j)];
                    matrix->data[MATRIX_INDEX(matrix, pivot_row, j)] = temp;
                }

                i = pivot[k];
                pivot[k] = pivot[pivot_row];
                pivot[pivot_row] = i;
            } else {
                /* No action */
            }

            if (Matrix_AbsQ31(matrix->data[MATRIX_INDEX(matrix, k, k)]) < ZERO_THRESHOLD_Q31) {
                status = MATRIX_ERROR_SINGULAR;
            } else {
                for (i = k + 1U; i < n; i++) {
                    factor = Matrix_DivQ31(matrix->data[MATRIX_INDEX(matrix, i, k)],
                                    matrix->data[MATRIX_INDEX(matrix, k, k)]);
                    matrix->data[MATRIX_INDEX(matrix, i, k)] = factor;

                    for (j = k + 1U; j < n; j++) {
                        matrix->data[MATRIX_INDEX(matrix, i, j)] -=
                            Matrix_MulQ31(factor, matrix->data[MATRIX_INDEX(matrix, k, j)]);
                    }
                }
            }
        }

        /* Check the last diagonal element — the k-loop runs k=0..n-2 and
         * only checks matrix[k][k] before factoring rows k+1..n-1.
         * matrix[n-1][n-1] is the final U diagonal and is never tested
         * inside the loop.  A singular matrix produces a near-zero value
         * here that must be caught explicitly.                           */
        if ((status == MATRIX_SUCCESS) && (n > 0U)) {
            if (Matrix_AbsQ31(matrix->data[MATRIX_INDEX(matrix, n - 1U, n - 1U)])
                    < ZERO_THRESHOLD_Q31) {
                status = MATRIX_ERROR_SINGULAR;
            } else {
                /* No action */
            }
        } else {
            /* No action */
        }
    }

    return status;
}

/*==============================================================================
 * SOLVE LINEAR SYSTEMS
 *============================================================================*/

MatrixStatus_Type Matrix_SolveGaussJordan(const Matrix_Type* const a,
                                          const Matrix_Type* const b,
                                          Matrix_Type* const x) {
    MatrixStatus_Type status;

    MatrixElement aug_buffer[MATRIX_MAX_ROWS * (MATRIX_MAX_COLS + MATRIX_MAX_COLS)];
    Matrix_Type aug;
    uint32_T i;
    uint32_T j;
    uint32_T k;
    uint32_T n;
    uint32_T m;
    MatrixElement pivot;
    MatrixElement factor;
    boolean_T singular;

    singular = FALSE;

    status = MATRIX_SUCCESS;

    if ((a == NULL) || (b == NULL) || (x == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (a->rows != a->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else if (a->rows != b->rows) {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    } else if ((x->max_rows < a->rows) || (x->max_cols < b->cols)) {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    } else {
        n = a->rows;
        m = b->cols;

        Matrix_Init(&aug, aug_buffer, n, n + m);

        for (i = 0U; i < n; i++) {
            for (j = 0U; j < n; j++) {
                aug.data[MATRIX_INDEX(&aug, i, j)] =
                    a->data[MATRIX_INDEX(a, i, j)];
            }
        }

        for (i = 0U; i < n; i++) {
            for (j = 0U; j < m; j++) {
                aug.data[MATRIX_INDEX(&aug, i, n + j)] =
                    b->data[MATRIX_INDEX(b, i, j)];
            }
        }

        for (i = 0U; i < n; i++) {
            pivot = aug.data[MATRIX_INDEX(&aug, i, i)];
            if (Matrix_AbsQ31(pivot) < ZERO_THRESHOLD_Q31) {
                singular = TRUE;
            } else {
                for (j = i; j < n + m; j++) {
                    aug.data[MATRIX_INDEX(&aug, i, j)] =
                        Matrix_DivQ31(aug.data[MATRIX_INDEX(&aug, i, j)], pivot);
                }

                for (k = 0U; k < n; k++) {
                    if (k != i) {
                        factor = aug.data[MATRIX_INDEX(&aug, k, i)];
                        for (j = i; j < n + m; j++) {
                            aug.data[MATRIX_INDEX(&aug, k, j)] -=
                                Matrix_MulQ31(factor, aug.data[MATRIX_INDEX(&aug, i, j)]);
                        }
                    } else {
                        /* No action */
                    }
                }
            }
        }

        if (singular != FALSE) {
            status = MATRIX_ERROR_SINGULAR;
        } else {
            for (i = 0U; i < n; i++) {
                for (j = 0U; j < m; j++) {
                    x->data[MATRIX_INDEX(x, i, j)] =
                        aug.data[MATRIX_INDEX(&aug, i, n + j)];
                }
            }

            x->rows = n;
            x->cols = m;
        }
    }

    return status;
}

MatrixStatus_Type Matrix_Solve(const Matrix_Type* const a,
                               const Matrix_Type* const b,
                               Matrix_Type* const x) {
    return Matrix_SolveGaussJordan(a, b, x);
}

/*==============================================================================
 * MATRIX PROPERTIES
 *============================================================================*/

boolean_T Matrix_IsSquare(const Matrix_Type* const matrix) {
    boolean_T result;

    result = FALSE;

    if (matrix != NULL) {
        if (matrix->rows == matrix->cols) {
            result = TRUE;
        }
    }

    return result;
}

boolean_T Matrix_IsSymmetric(const Matrix_Type* const matrix,
                             const MatrixFloat tolerance) {
    boolean_T result;
    uint32_T i;
    uint32_T j;
    MatrixFloat a_ij;
    MatrixFloat a_ji;
    MatrixFloat diff;
    MatrixFloat tol;

    result = TRUE;
    tol = (tolerance > 0.0f) ? tolerance : ZERO_THRESHOLD_FLOAT;

    if (matrix == NULL) {
        result = FALSE;
    } else if (matrix->rows != matrix->cols) {
        result = FALSE;
    } else {
        for (i = 0U; (i < matrix->rows) && (result != FALSE); i++) {
            for (j = i + 1U; (j < matrix->cols) && (result != FALSE); j++) {
                a_ij = Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, i, j)]);
                a_ji = Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, j, i)]);
                diff = ABS_FLOAT(a_ij - a_ji);

                if (diff > tol) {
                    result = FALSE;
                } else {
                    /* No action */
                }
            }
        }
    }

    return result;
}

MatrixStatus_Type Matrix_Trace(const Matrix_Type* const matrix,
                               MatrixFloat* const trace) {
    MatrixStatus_Type status;

    uint32_T i;
    MatrixFloat sum;

    sum = 0.0f;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (trace == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else if (matrix->rows != matrix->cols) {
        status = MATRIX_ERROR_NOT_SQUARE;
    } else {
        for (i = 0U; i < matrix->rows; i++) {
            sum += Matrix_Q31ToFloat(matrix->data[MATRIX_INDEX(matrix, i, i)]);
        }

        *trace = sum;
    }

    return status;
}

MatrixStatus_Type Matrix_NormFrobenius(const Matrix_Type* const matrix,
                                       MatrixFloat* const norm) {
    MatrixStatus_Type status;

    uint32_T i;
    uint32_T j;
    real64_T sum;
    real64_T val;

    sum = 0.0;

    status = MATRIX_SUCCESS;

    if ((matrix == NULL) || (norm == NULL)) {
        status = MATRIX_ERROR_NULL_PTR;
    } else {
        for (i = 0U; i < matrix->rows; i++) {
            for (j = 0U; j < matrix->cols; j++) {
                val = (real64_T)matrix->data[MATRIX_INDEX(matrix, i, j)] / Q31_SCALE_D;
                sum += val * val;
            }
        }

        *norm = (MatrixFloat)sqrt(sum);
    }

    return status;
}

/*==============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

boolean_T Matrix_IsEqual(const Matrix_Type* const a,
                         const Matrix_Type* const b,
                         const MatrixFloat tolerance) {
    boolean_T result;
    uint32_T i;
    uint32_T j;
    real64_T diff;
    real64_T tol;
    real64_T a_val;
    real64_T b_val;

    result = TRUE;
    tol = (tolerance > 0.0f) ? (real64_T)tolerance : (real64_T)ZERO_THRESHOLD_FLOAT;

    if ((a == NULL) || (b == NULL)) {
        result = FALSE;
    } else if ((a->rows != b->rows) || (a->cols != b->cols)) {
        result = FALSE;
    } else {
        for (i = 0U; (i < a->rows) && (result != FALSE); i++) {
            for (j = 0U; (j < a->cols) && (result != FALSE); j++) {
                a_val = (real64_T)a->data[MATRIX_INDEX(a, i, j)] / Q31_SCALE_D;
                b_val = (real64_T)b->data[MATRIX_INDEX(b, i, j)] / Q31_SCALE_D;
                diff = (a_val > b_val) ? (a_val - b_val) : (b_val - a_val);

                if (diff > tol) {
                    result = FALSE;
                } else {
                    /* No action */
                }
            }
        }
    }

    return result;
}

void Matrix_GetDimensions(const Matrix_Type* const matrix,
                          uint32_T* const rows,
                          uint32_T* const cols) {
    if ((matrix != NULL) && (rows != NULL) && (cols != NULL)) {
        *rows = matrix->rows;
        *cols = matrix->cols;
    }
}

void Matrix_Fill(Matrix_Type* const matrix,
                 const MatrixElement value) {
    uint32_T i;
    uint32_T j;

    if (matrix != NULL) {
        for (i = 0U; i < matrix->rows; i++) {
            for (j = 0U; j < matrix->cols; j++) {
                matrix->data[MATRIX_INDEX(matrix, i, j)] = value;
            }
        }
    }
}

void Matrix_FillFloat(Matrix_Type* const matrix,
                      const MatrixFloat value) {
    Matrix_Fill(matrix, Matrix_FloatToQ31(value));
}

/*==============================================================================
 * CONVERSION FUNCTIONS
 *============================================================================*/

MatrixElement Matrix_FloatToQ31(const MatrixFloat value) {
    MatrixElement result;
    MatrixFloat clamped;

    if (value > 0.9999999995f) {
        clamped = 0.9999999995f;
    } else if (value < -1.0f) {
        clamped = -1.0f;
    } else {
        clamped = value;
    }

    result = (MatrixElement)(clamped * Q31_SCALE_F);

    return result;
}

MatrixFloat Matrix_Q31ToFloat(const MatrixElement value) {
    MatrixFloat result;

    result = (MatrixFloat)value / Q31_SCALE_F;

    return result;
}