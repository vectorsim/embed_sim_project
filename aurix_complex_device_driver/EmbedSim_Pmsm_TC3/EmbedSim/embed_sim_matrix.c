/**********************************************************************************************************************
 * \file      embed_sim_matrix.c
 * \brief     32-bit fixed-point (Q31) linear algebra library implementation.
 *
 * \details   All arithmetic is performed without recursion using iterative algorithms.
 *            Double-precision intermediates are used for determinant and inverse
 *            calculations to minimise rounding error.
 *
 *            Q31 multiply formula:   result = (A × B) / 2³¹  (saturating)
 *            Q31 divide formula:     result = (A × 2³¹) / B  (saturating)
 *
 * \note      MISRA C:2012 deviations:
 *              Rule 12.2 — Signed right-shift replaced with integer division by
 *                          Q31_SCALE_I throughout MulQ31 and DivQ31.
 *
 * \version   5.1.0
 * \date      2025-05-24
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_matrix.h"
#include <string.h>
#include <math.h>


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/* ---- Q31 fixed-point constants ---- */
#define Q31_ONE        ((int32_T)0x7FFFFFFF)   /**< +1.0 in Q31           */
#define Q31_HALF       ((int32_T)0x40000000)   /**< +0.5 in Q31           */
#define Q31_QUARTER    ((int32_T)0x20000000)   /**< +0.25 in Q31          */
#define Q31_EPSILON    ((int32_T)0x00000001)   /**< Smallest Q31 step     */
#define Q31_ZERO       ((int32_T)0x00000000)   /**< 0.0 in Q31            */
#define Q31_MINUS_ONE  ((int32_T)0x80000000)   /**< −1.0 in Q31           */

/* Q31 scale factor as float, double, and signed 64-bit integer.
 * The LL suffix on Q31_SCALE_I ensures the constant is typed as int64_T on
 * all targets — avoids implementation-defined behaviour for values exceeding
 * the range of 32-bit int (MISRA C:2012 Rule 7.2).                         */
#define Q31_SCALE_F    (2147483648.0f)          /**< 2³¹ as float          */
#define Q31_SCALE_D    (2147483648.0)           /**< 2³¹ as double         */
#define Q31_SCALE_I    (2147483648LL)           /**< 2³¹ as int64_T        */

/* ---- Numeric thresholds ---- */
#define ZERO_THRESHOLD_Q31    ((int32_T)0x00000100)  /**< Q31 near-zero threshold   */
#define ZERO_THRESHOLD_FLOAT  (1.0e-6f)              /**< Float near-zero threshold */

/* ---- Jacobi iteration defaults ---- */
#define JACOBI_TOLERANCE   (1.0e-6f)         /**< Default off-diagonal convergence limit */
#define JACOBI_MAX_ITER    (50U)             /**< Default maximum sweep count            */
#define JACOBI_PI_OVER_4   (0.78539816339f) /**< π/4 radians                            */

/* ---- Index and helper macros ---- */
/** \brief Compute the flat buffer index for element (r, c) in Matrix_P \p m. */
#define MATRIX_INDEX(m, r, c)   (((r) * (m)->Stride) + (c))

/** \brief TRUE if Q31 value \p X is within the near-zero threshold. */
#define IS_ZERO_Q31(x)   (((x) < ZERO_THRESHOLD_Q31) && ((x) > -ZERO_THRESHOLD_Q31))

/** \brief TRUE if float value \p X is within the near-zero threshold. */
#define IS_ZERO_FLOAT(x) (((x) < ZERO_THRESHOLD_FLOAT) && ((x) > -ZERO_THRESHOLD_FLOAT))

/** \brief Absolute value of a float without branching (ternary). */
#define ABS_FLOAT(x)     (((x) >= 0.0f) ? (x) : -(x))


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Q31 multiply: result = (a × b) / 2³¹ with saturation.
 *
 * Uses a 64-bit intermediate and integer division to avoid implementation-
 * defined behaviour from signed right-shift (MISRA C:2012 Rule 12.2).
 *
 * \param[in] A  First Q31 operand.
 * \param[in] B  Second Q31 operand.
 * \return       Saturated Q31 product.
 */
static MatrixElement Matrix_MulQ31(const MatrixElement A, const MatrixElement B);

/**
 * \brief  Q31 divide: result = (a × 2³¹) / b with saturation.
 *
 * Returns #Q31_ZERO if \p B is zero.
 *
 * \param[in] A  Q31 numerator.
 * \param[in] B  Q31 denominator.
 * \return       Saturated Q31 quotient.
 */
static MatrixElement Matrix_DivQ31(const MatrixElement A, const MatrixElement B);

/**
 * \brief  Q31 absolute value, handling the #Q31_MINUS_ONE corner case.
 *
 * \param[in] X  Q31 input.
 * \return       |x| in Q31; #Q31_ONE if x == #Q31_MINUS_ONE.
 */
static MatrixElement Matrix_AbsQ31(const MatrixElement X);

/**
 * \brief  Validate that (row, col) is within the active dimensions of \p Matrix_P.
 *
 * \param[in] Matrix_P  Matrix to check (NULL → FALSE).
 * \param[in] row     Row index.
 * \param[in] col     Column index.
 * \return   TRUE if indices are in range, FALSE otherwise.
 */
static boolean_T Matrix_IsValidIndex(
    const Matrix_Type  * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col);

/**
 * \brief  Pre-condition check for add/subtract operations.
 *
 * \param[in] A       First operand.
 * \param[in] B       Second operand.
 * \param[in] result  Output Matrix_P.
 * \return   #MATRIX_SUCCESS or appropriate error code.
 */
static MatrixStatus_Type Matrix_CheckAddSub(
    const Matrix_Type * const A_P,
    const Matrix_Type * const B_P,
    const Matrix_Type * const Result_P);

/**
 * \brief  Pre-condition check for Matrix_P multiply.
 *
 * \param[in] A       Left factor.
 * \param[in] B       Right factor.
 * \param[in] result  Output Matrix_P.
 * \return   #MATRIX_SUCCESS or appropriate error code.
 */
static MatrixStatus_Type Matrix_CheckMultiply(
    const Matrix_Type * const A_P,
    const Matrix_Type * const B_P,
    const Matrix_Type * const Result_P);

/**
 * \brief  Compute the determinant of an already-copied work Matrix_P via LU.
 *
 * \p Matrix_P is consumed (modified) during the computation.
 *
 * \param[in,out] Matrix_P  Square work Matrix_P (overwritten with L+U).
 * \param[out]    DetOut_P     Computed determinant.
 * \return   #MATRIX_SUCCESS or #MATRIX_ERROR_SINGULAR.
 */
static MatrixStatus_Type Matrix_DeterminantLU(
    Matrix_Type  * const Matrix_P,
    MatrixFloat  * const DetOut_P);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_MulQ31
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixElement Matrix_MulQ31(const MatrixElement A, const MatrixElement B)
{
    int64_T       temp;
    MatrixElement result;

    /*
     * result = (a × b) / 2³¹
     *
     * Integer division by Q31_SCALE_I is used in place of arithmetic
     * right-shift by 31, because right-shifting a signed value is
     * implementation-defined (MISRA C:2012 Rule 12.2).
     */
    temp = ((int64_T)A * (int64_T)B) / (int64_T)Q31_SCALE_I;

    if (temp > (int64_T)Q31_ONE)
    {
        result = Q31_ONE;
    }
    else if (temp < (int64_T)Q31_MINUS_ONE)
    {
        result = Q31_MINUS_ONE;
    }
    else
    {
        result = (MatrixElement)temp;
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_DivQ31
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixElement Matrix_DivQ31(const MatrixElement A, const MatrixElement B)
{
    int64_T       temp;
    MatrixElement result;

    if (B == Q31_ZERO)
    {
        result = Q31_ZERO;
    }
    else
    {
        /*
         * result = (a × 2³¹) / b
         *
         * Multiplication by Q31_SCALE_I replaces left-shift by 31, which is
         * implementation-defined for signed integers (MISRA C:2012 Rule 12.2).
         */
        temp = ((int64_T)A * (int64_T)Q31_SCALE_I) / (int64_T)B;

        if (temp > (int64_T)Q31_ONE)
        {
            result = Q31_ONE;
        }
        else if (temp < (int64_T)Q31_MINUS_ONE)
        {
            result = Q31_MINUS_ONE;
        }
        else
        {
            result = (MatrixElement)temp;
        }
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_AbsQ31
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixElement Matrix_AbsQ31(const MatrixElement X)
{
    MatrixElement result;

    if (X < (MatrixElement)0)
    {
        /* Q31_MINUS_ONE (0x80000000) has no positive counterpart — clamp to Q31_ONE. */
        result = (X == Q31_MINUS_ONE) ? Q31_ONE : -X;
    }
    else
    {
        result = X;
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsValidIndex
 *------------------------------------------------------------------------------------------------------------------*/
static boolean_T Matrix_IsValidIndex(
    const Matrix_Type  * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col)
{
    boolean_T result;

    result = TRUE;

    if (Matrix_P == NULL)
    {
        result = FALSE;
    }
    else if (Row >= Matrix_P->Rows)
    {
        result = FALSE;
    }
    else if (Col >= Matrix_P->Cols)
    {
        result = FALSE;
    }
    else
    {
        /* Indices are valid – no action. */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_CheckAddSub
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixStatus_Type Matrix_CheckAddSub(
    const Matrix_Type * const A_P,
    const Matrix_Type * const B_P,
    const Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((A_P == NULL) || (B_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((A_P->Rows != B_P->Rows) || (A_P->Cols != B_P->Cols))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((Result_P->MaxRows < A_P->Rows) || (Result_P->MaxCols < A_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        /* Pre-conditions satisfied – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_CheckMultiply
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixStatus_Type Matrix_CheckMultiply(
    const Matrix_Type * const A_P,
    const Matrix_Type * const B_P,
    const Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((A_P == NULL) || (B_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (A_P->Cols != B_P->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((Result_P->MaxRows < A_P->Rows) || (Result_P->MaxCols < B_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        /* Pre-conditions satisfied – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_DeterminantLU  (private helper)
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixStatus_Type Matrix_DeterminantLU(
    Matrix_Type  * const Matrix_P,
    MatrixFloat  * const DetOut_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          n;
    uint32_T          pivot_local[MATRIX_MAX_ROWS];
    int32_T           sign;
    real64_T          diag_product;

    sign         = 1;
    diag_product = 1.0;
    n            = Matrix_P->Rows;

    status = Matrix_LU(Matrix_P, pivot_local);

    if (status == MATRIX_SUCCESS)
    {
        for (i = 0U; i < n; i++)
        {
            if (pivot_local[i] != i)
            {
                sign = -sign;
            }
            else
            {
                /* No row swap at index i – no action. */
            }
        }

        for (i = 0U; i < n; i++)
        {
            diag_product *= (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, i)] / Q31_SCALE_D;
        }

        *DetOut_P = (MatrixFloat)((real64_T)sign * diag_product);
    }
    else
    {
        *DetOut_P = 0.0f;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Init
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_Init(
    Matrix_Type    * const Matrix_P,
    MatrixElement  * const buffer,
    const uint32_T MaxRows,
    const uint32_T MaxCols)
{
    if ((Matrix_P != NULL) && (buffer != NULL))
    {
        Matrix_P->Data     = buffer;
        Matrix_P->MaxRows = MaxRows;
        Matrix_P->MaxCols = MaxCols;
        Matrix_P->Rows     = MaxRows;
        Matrix_P->Cols     = MaxCols;
        Matrix_P->IsView  = FALSE;
        Matrix_P->Stride   = MaxCols;

        (void)memset(buffer, 0,
                     (size_t)MaxRows * (size_t)MaxCols * sizeof(MatrixElement));
    }
    else
    {
        /* MISRA C 2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SetDimensions
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_SetDimensions(
    Matrix_Type  * const Matrix_P,
    const uint32_T Rows,
    const uint32_T Cols)
{
    if (Matrix_P != NULL)
    {
        if ((Rows > 0U) && (Rows <= Matrix_P->MaxRows) &&
            (Cols > 0U) && (Cols <= Matrix_P->MaxCols))
        {
            Matrix_P->Rows = Rows;
            Matrix_P->Cols = Cols;
        }
        else
        {
            /* Requested dimensions out of range – no action. */
        }
    }
    else
    {
        /* MISRA C 2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Zero
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_Zero(Matrix_Type * const Matrix_P)
{
    uint32_T row;
    uint32_T col;

    if (Matrix_P != NULL)
    {
        for (row = 0U; row < Matrix_P->Rows; row++)
        {
            for (col = 0U; col < Matrix_P->Cols; col++)
            {
                Matrix_P->Data[MATRIX_INDEX(Matrix_P, row, col)] = Q31_ZERO;
            }
        }
    }
    else
    {
        /* MISRA C 2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Identity
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Identity(Matrix_Type * const Matrix_P)
{
    MatrixStatus_Type status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if (Matrix_P == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        for (row = 0U; row < Matrix_P->Rows; row++)
        {
            for (col = 0U; col < Matrix_P->Cols; col++)
            {
                Matrix_P->Data[MATRIX_INDEX(Matrix_P, row, col)] =
                    (row == col) ? Q31_ONE : Q31_ZERO;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Copy
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Copy(
    Matrix_Type        * const Dest_P,
    const Matrix_Type  * const Src_P)
{
    MatrixStatus_Type status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if ((Dest_P == NULL) || (Src_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Dest_P->MaxRows < Src_P->Rows) || (Dest_P->MaxCols < Src_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        for (row = 0U; row < Src_P->Rows; row++)
        {
            for (col = 0U; col < Src_P->Cols; col++)
            {
                Dest_P->Data[MATRIX_INDEX(Dest_P, row, col)] =
                    Src_P->Data[MATRIX_INDEX(Src_P, row, col)];
            }
        }

        Dest_P->Rows = Src_P->Rows;
        Dest_P->Cols = Src_P->Cols;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SetElement
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_SetElement(
    Matrix_Type    * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixElement    value)
{
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if (Matrix_P == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_IsValidIndex(Matrix_P, Row, Col) == FALSE)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        Matrix_P->Data[MATRIX_INDEX(Matrix_P, Row, Col)] = value;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SetElementFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_SetElementFloat(
    Matrix_Type    * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixFloat      value)
{
    return Matrix_SetElement(Matrix_P, Row, Col, Matrix_FloatToQ31(value));
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_GetElement
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_GetElement(
    const Matrix_Type  * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    MatrixElement      * const value)
{
    MatrixStatus_Type status;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (value == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_IsValidIndex(Matrix_P, Row, Col) == FALSE)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        *value = Matrix_P->Data[MATRIX_INDEX(Matrix_P, Row, Col)];
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_GetElementFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_GetElementFloat(
    const Matrix_Type  * const Matrix_P,
    const uint32_T Row,
    const uint32_T Col,
    MatrixFloat        * const value)
{
    MatrixStatus_Type status;
    MatrixElement     q31_val;

    status = Matrix_GetElement(Matrix_P, Row, Col, &q31_val);

    if (status == MATRIX_SUCCESS)
    {
        *value = Matrix_Q31ToFloat(q31_val);
    }
    else
    {
        /* Error already captured – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Add
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Add(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const Result_P)
{
    MatrixStatus_Type status;
    uint32_T          row;
    uint32_T          col;
    MatrixElement     sum;

    status = Matrix_CheckAddSub(A_P, B_P, Result_P);

    if (status == MATRIX_SUCCESS)
    {
        for (row = 0U; row < A_P->Rows; row++)
        {
            for (col = 0U; col < A_P->Cols; col++)
            {
                sum = A_P->Data[MATRIX_INDEX(A_P, row, col)] +
                      B_P->Data[MATRIX_INDEX(B_P, row, col)];

                /* Saturate on signed overflow. */
                if ((A_P->Data[MATRIX_INDEX(A_P, row, col)] > 0) &&
                    (B_P->Data[MATRIX_INDEX(B_P, row, col)] > 0) &&
                    (sum < 0))
                {
                    sum = Q31_ONE;
                }
                else if ((A_P->Data[MATRIX_INDEX(A_P, row, col)] < 0) &&
                         (B_P->Data[MATRIX_INDEX(B_P, row, col)] < 0) &&
                         (sum > 0))
                {
                    sum = Q31_MINUS_ONE;
                }
                else
                {
                    /* No overflow – no action. */
                }

                Result_P->Data[MATRIX_INDEX(Result_P, row, col)] = sum;
            }
        }

        Result_P->Rows = A_P->Rows;
        Result_P->Cols = A_P->Cols;
    }
    else
    {
        /* Pre-condition error – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Subtract
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Subtract(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const Result_P)
{
    MatrixStatus_Type status;
    uint32_T          row;
    uint32_T          col;
    MatrixElement     diff;

    status = Matrix_CheckAddSub(A_P, B_P, Result_P);

    if (status == MATRIX_SUCCESS)
    {
        for (row = 0U; row < A_P->Rows; row++)
        {
            for (col = 0U; col < A_P->Cols; col++)
            {
                diff = A_P->Data[MATRIX_INDEX(A_P, row, col)] -
                       B_P->Data[MATRIX_INDEX(B_P, row, col)];

                /* Saturate on signed overflow. */
                if ((A_P->Data[MATRIX_INDEX(A_P, row, col)] > 0) &&
                    (B_P->Data[MATRIX_INDEX(B_P, row, col)] < 0) &&
                    (diff < 0))
                {
                    diff = Q31_ONE;
                }
                else if ((A_P->Data[MATRIX_INDEX(A_P, row, col)] < 0) &&
                         (B_P->Data[MATRIX_INDEX(B_P, row, col)] > 0) &&
                         (diff > 0))
                {
                    diff = Q31_MINUS_ONE;
                }
                else
                {
                    /* No overflow – no action. */
                }

                Result_P->Data[MATRIX_INDEX(Result_P, row, col)] = diff;
            }
        }

        Result_P->Rows = A_P->Rows;
        Result_P->Cols = A_P->Cols;
    }
    else
    {
        /* Pre-condition error – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Multiply
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Multiply(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const Result_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    MatrixElement     sum;
    boolean_T         saturated;

    status = Matrix_CheckMultiply(A_P, B_P, Result_P);

    if (status == MATRIX_SUCCESS)
    {
        for (i = 0U; i < A_P->Rows; i++)
        {
            for (j = 0U; j < B_P->Cols; j++)
            {
                saturated = FALSE;
                sum       = Q31_ZERO;

                for (k = 0U; (k < A_P->Cols) && (saturated == FALSE); k++)
                {
                    sum += Matrix_MulQ31(A_P->Data[MATRIX_INDEX(A_P, i, k)],
                                        B_P->Data[MATRIX_INDEX(B_P, k, j)]);

                    if (sum > Q31_ONE)
                    {
                        sum       = Q31_ONE;
                        saturated = TRUE;
                    }
                    else if (sum < Q31_MINUS_ONE)
                    {
                        sum       = Q31_MINUS_ONE;
                        saturated = TRUE;
                    }
                    else
                    {
                        /* Accumulator still in range – no action. */
                    }
                }

                Result_P->Data[MATRIX_INDEX(Result_P, i, j)] = sum;
            }
        }

        Result_P->Rows = A_P->Rows;
        Result_P->Cols = B_P->Cols;
    }
    else
    {
        /* Pre-condition error – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_ScalarMultiply
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_ScalarMultiply(
    const Matrix_Type  * const Matrix_P,
    const MatrixElement        Scalar,
    Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Result_P->MaxRows < Matrix_P->Rows) || (Result_P->MaxCols < Matrix_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        for (row = 0U; row < Matrix_P->Rows; row++)
        {
            for (col = 0U; col < Matrix_P->Cols; col++)
            {
                Result_P->Data[MATRIX_INDEX(Result_P, row, col)] =
                    Matrix_MulQ31(Matrix_P->Data[MATRIX_INDEX(Matrix_P, row, col)], Scalar);
            }
        }

        Result_P->Rows = Matrix_P->Rows;
        Result_P->Cols = Matrix_P->Cols;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_ScalarMultiplyFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_ScalarMultiplyFloat(
    const Matrix_Type  * const Matrix_P,
    const MatrixFloat          Scalar,
    Matrix_Type * const Result_P)
{
    return Matrix_ScalarMultiply(Matrix_P, Matrix_FloatToQ31(Scalar), Result_P);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Transpose
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Transpose(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Result_P->MaxRows < Matrix_P->Cols) || (Result_P->MaxCols < Matrix_P->Rows))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        for (row = 0U; row < Matrix_P->Rows; row++)
        {
            for (col = 0U; col < Matrix_P->Cols; col++)
            {
                Result_P->Data[MATRIX_INDEX(Result_P, col, row)] =
                    Matrix_P->Data[MATRIX_INDEX(Matrix_P, row, col)];
            }
        }

        Result_P->Rows = Matrix_P->Cols;
        Result_P->Cols = Matrix_P->Rows;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant2x2
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Determinant2x2(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    real64_T          a11;
    real64_T          a12;
    real64_T          a21;
    real64_T          a22;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 2U) || (Matrix_P->Cols != 2U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        /* Use double precision for numerical accuracy. */
        a11 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 0U, 0U)] / Q31_SCALE_D;
        a12 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 0U, 1U)] / Q31_SCALE_D;
        a21 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 1U, 0U)] / Q31_SCALE_D;
        a22 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 1U, 1U)] / Q31_SCALE_D;

        *DetOut_P = (MatrixFloat)((a11 * a22) - (a12 * a21));
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant3x3
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Determinant3x3(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    real64_T          m[3U][3U];
    real64_T          term1;
    real64_T          term2;
    real64_T          term3;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 3U) || (Matrix_P->Cols != 3U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        for (i = 0U; i < 3U; i++)
        {
            for (j = 0U; j < 3U; j++)
            {
                m[i][j] = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] / Q31_SCALE_D;
            }
        }

        /* Sarrus' rule. */
        term1 = m[0U][0U] * ((m[1U][1U] * m[2U][2U]) - (m[1U][2U] * m[2U][1U]));
        term2 = m[0U][1U] * ((m[1U][0U] * m[2U][2U]) - (m[1U][2U] * m[2U][0U]));
        term3 = m[0U][2U] * ((m[1U][0U] * m[2U][1U]) - (m[1U][1U] * m[2U][0U]));

        *DetOut_P = (MatrixFloat)(term1 - term2 + term3);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant4x4
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Determinant4x4(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    real64_T          m[4U][4U];
    real64_T          term1;
    real64_T          term2;
    real64_T          term3;
    real64_T          term4;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 4U) || (Matrix_P->Cols != 4U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        for (i = 0U; i < 4U; i++)
        {
            for (j = 0U; j < 4U; j++)
            {
                m[i][j] = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] / Q31_SCALE_D;
            }
        }

        /* Direct 4 × 4 formula — Leibniz expansion grouped by first-column cofactors. */
        term1 = m[0][3] * m[1][2] * m[2][1] * m[3][0] - m[0][2] * m[1][3] * m[2][1] * m[3][0]
              - m[0][3] * m[1][1] * m[2][2] * m[3][0] + m[0][1] * m[1][3] * m[2][2] * m[3][0]
              + m[0][2] * m[1][1] * m[2][3] * m[3][0] - m[0][1] * m[1][2] * m[2][3] * m[3][0];

        term2 = -m[0][3] * m[1][2] * m[2][0] * m[3][1] + m[0][2] * m[1][3] * m[2][0] * m[3][1]
               + m[0][3] * m[1][0] * m[2][2] * m[3][1] - m[0][0] * m[1][3] * m[2][2] * m[3][1]
               - m[0][2] * m[1][0] * m[2][3] * m[3][1] + m[0][0] * m[1][2] * m[2][3] * m[3][1];

        term3 = m[0][3] * m[1][1] * m[2][0] * m[3][2] - m[0][1] * m[1][3] * m[2][0] * m[3][2]
              - m[0][3] * m[1][0] * m[2][1] * m[3][2] + m[0][0] * m[1][3] * m[2][1] * m[3][2]
              + m[0][1] * m[1][0] * m[2][3] * m[3][2] - m[0][0] * m[1][1] * m[2][3] * m[3][2];

        term4 = -m[0][2] * m[1][1] * m[2][0] * m[3][3] + m[0][1] * m[1][2] * m[2][0] * m[3][3]
               + m[0][2] * m[1][0] * m[2][1] * m[3][3] - m[0][0] * m[1][2] * m[2][1] * m[3][3]
               - m[0][1] * m[1][0] * m[2][2] * m[3][3] + m[0][0] * m[1][1] * m[2][2] * m[3][3];

        *DetOut_P = (MatrixFloat)(term1 + term2 + term3 + term4);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant5x5 … Matrix_Determinant8x8  (LU-based)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Determinant5x5(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    MatrixElement     work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       work;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 5U) || (Matrix_P->Cols != 5U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, work_buffer, 5U, 5U);
        (void)Matrix_Copy(&work, Matrix_P);
        status = Matrix_DeterminantLU(&work, DetOut_P);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant6x6(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    MatrixElement     work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       work;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 6U) || (Matrix_P->Cols != 6U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, work_buffer, 6U, 6U);
        (void)Matrix_Copy(&work, Matrix_P);
        status = Matrix_DeterminantLU(&work, DetOut_P);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant7x7(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    MatrixElement     work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       work;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 7U) || (Matrix_P->Cols != 7U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, work_buffer, 7U, 7U);
        (void)Matrix_Copy(&work, Matrix_P);
        status = Matrix_DeterminantLU(&work, DetOut_P);
    }

    return status;
}

MatrixStatus_Type Matrix_Determinant8x8(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    MatrixElement     work_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       work;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 8U) || (Matrix_P->Cols != 8U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, work_buffer, 8U, 8U);
        (void)Matrix_Copy(&work, Matrix_P);
        status = Matrix_DeterminantLU(&work, DetOut_P);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant  (dispatcher)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Determinant(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const DetOut_P)
{
    MatrixStatus_Type status;
    uint32_T          n;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (DetOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        n = Matrix_P->Rows;

        if (n == 1U)
        {
            *DetOut_P = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, 0U, 0U)]);
        }
        else if (n == 2U) { status = Matrix_Determinant2x2(Matrix_P, DetOut_P); }
        else if (n == 3U) { status = Matrix_Determinant3x3(Matrix_P, DetOut_P); }
        else if (n == 4U) { status = Matrix_Determinant4x4(Matrix_P, DetOut_P); }
        else if (n == 5U) { status = Matrix_Determinant5x5(Matrix_P, DetOut_P); }
        else if (n == 6U) { status = Matrix_Determinant6x6(Matrix_P, DetOut_P); }
        else if (n == 7U) { status = Matrix_Determinant7x7(Matrix_P, DetOut_P); }
        else if (n == 8U) { status = Matrix_Determinant8x8(Matrix_P, DetOut_P); }
        else
        {
            status = MATRIX_ERROR_SIZE_EXCEEDED;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Inverse2x2
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Inverse2x2(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;
    MatrixFloat       DetOut_P;
    real64_T          a11;
    real64_T          a12;
    real64_T          a21;
    real64_T          a22;
    real64_T          inv_det;

    status = Matrix_Determinant2x2(Matrix_P, &DetOut_P);

    if (status == MATRIX_SUCCESS)
    {
        if (ABS_FLOAT(DetOut_P) < ZERO_THRESHOLD_FLOAT)
        {
            status = MATRIX_ERROR_SINGULAR;
        }
        else
        {
            inv_det = 1.0 / (real64_T)DetOut_P;

            a11 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 0U, 0U)] / Q31_SCALE_D;
            a12 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 0U, 1U)] / Q31_SCALE_D;
            a21 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 1U, 0U)] / Q31_SCALE_D;
            a22 = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, 1U, 1U)] / Q31_SCALE_D;

            /* inv(A) = (1/DetOut_P) · [ a22  -a12; -a21  a11 ] */
            Result_P->Data[MATRIX_INDEX(Result_P, 0U, 0U)] = Matrix_FloatToQ31((MatrixFloat)( a22 * inv_det));
            Result_P->Data[MATRIX_INDEX(Result_P, 0U, 1U)] = Matrix_FloatToQ31((MatrixFloat)(-a12 * inv_det));
            Result_P->Data[MATRIX_INDEX(Result_P, 1U, 0U)] = Matrix_FloatToQ31((MatrixFloat)(-a21 * inv_det));
            Result_P->Data[MATRIX_INDEX(Result_P, 1U, 1U)] = Matrix_FloatToQ31((MatrixFloat)( a11 * inv_det));

            Result_P->Rows = 2U;
            Result_P->Cols = 2U;
        }
    }
    else
    {
        /* Error propagated from Determinant2x2 – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Inverse3x3
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Inverse3x3(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;
    MatrixFloat       DetOut_P;
    real64_T          m[3U][3U];
    real64_T          c[3U][3U];
    real64_T          inv_det;
    uint32_T          i;
    uint32_T          j;

    status = Matrix_Determinant3x3(Matrix_P, &DetOut_P);

    if (status == MATRIX_SUCCESS)
    {
        if (ABS_FLOAT(DetOut_P) < ZERO_THRESHOLD_FLOAT)
        {
            status = MATRIX_ERROR_SINGULAR;
        }
        else
        {
            inv_det = 1.0 / (real64_T)DetOut_P;

            for (i = 0U; i < 3U; i++)
            {
                for (j = 0U; j < 3U; j++)
                {
                    m[i][j] = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] / Q31_SCALE_D;
                }
            }

            /* Cofactor Matrix_P (transposed in-place when writing result). */
            c[0U][0U] =  (m[1U][1U] * m[2U][2U] - m[1U][2U] * m[2U][1U]);
            c[0U][1U] = -(m[1U][0U] * m[2U][2U] - m[1U][2U] * m[2U][0U]);
            c[0U][2U] =  (m[1U][0U] * m[2U][1U] - m[1U][1U] * m[2U][0U]);

            c[1U][0U] = -(m[0U][1U] * m[2U][2U] - m[0U][2U] * m[2U][1U]);
            c[1U][1U] =  (m[0U][0U] * m[2U][2U] - m[0U][2U] * m[2U][0U]);
            c[1U][2U] = -(m[0U][0U] * m[2U][1U] - m[0U][1U] * m[2U][0U]);

            c[2U][0U] =  (m[0U][1U] * m[1U][2U] - m[0U][2U] * m[1U][1U]);
            c[2U][1U] = -(m[0U][0U] * m[1U][2U] - m[0U][2U] * m[1U][0U]);
            c[2U][2U] =  (m[0U][0U] * m[1U][1U] - m[0U][1U] * m[1U][0U]);

            /* inv(A) = (1/DetOut_P) · C^T  →  result[i][j] = c[j][i] / DetOut_P */
            for (i = 0U; i < 3U; i++)
            {
                for (j = 0U; j < 3U; j++)
                {
                    Result_P->Data[MATRIX_INDEX(Result_P, i, j)] =
                        Matrix_FloatToQ31((MatrixFloat)(c[j][i] * inv_det));
                }
            }

            Result_P->Rows = 3U;
            Result_P->Cols = 3U;
        }
    }
    else
    {
        /* Error propagated from Determinant3x3 – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Inverse4x4  (augmented Gauss-Jordan, partial pivot)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Inverse4x4(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;
    MatrixElement     aug_buffer[4U * 8U];
    Matrix_Type       aug;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    uint32_T          max_row;
    MatrixElement     max_val;
    MatrixElement     val;
    MatrixElement     temp;
    MatrixElement     pivot;
    MatrixElement     factor;
    boolean_T         singular;

    n        = 4U;
    singular = FALSE;
    status   = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((Matrix_P->Rows != 4U) || (Matrix_P->Cols != 4U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((Result_P->MaxRows < 4U) || (Result_P->MaxCols < 4U))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        Matrix_Init(&aug, aug_buffer, 4U, 8U);

        /* Build augmented Matrix_P [A | I]. */
        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < n; j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, j)] =
                    Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)];
            }
        }

        for (i = 0U; i < n; i++)
        {
            for (j = n; j < (2U * n); j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, j)] =
                    ((j - n) == i) ? Q31_ONE : Q31_ZERO;
            }
        }

        /* Gauss-Jordan elimination with partial pivoting. */
        for (i = 0U; (i < n) && (singular == FALSE); i++)
        {
            max_row = i;
            max_val = Matrix_AbsQ31(aug.Data[MATRIX_INDEX(&aug, i, i)]);

            for (k = i + 1U; k < n; k++)
            {
                val = Matrix_AbsQ31(aug.Data[MATRIX_INDEX(&aug, k, i)]);
                if (val > max_val)
                {
                    max_val = val;
                    max_row = k;
                }
            }

            if (max_val < ZERO_THRESHOLD_Q31)
            {
                singular = TRUE;
            }
            else
            {
                if (max_row != i)
                {
                    for (j = 0U; j < (2U * n); j++)
                    {
                        temp                                  = aug.Data[MATRIX_INDEX(&aug, i,       j)];
                        aug.Data[MATRIX_INDEX(&aug, i,       j)] = aug.Data[MATRIX_INDEX(&aug, max_row, j)];
                        aug.Data[MATRIX_INDEX(&aug, max_row, j)] = temp;
                    }
                }
                else
                {
                    /* No swap needed – no action. */
                }

                pivot = aug.Data[MATRIX_INDEX(&aug, i, i)];
                for (j = i; j < (2U * n); j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] =
                        Matrix_DivQ31(aug.Data[MATRIX_INDEX(&aug, i, j)], pivot);
                }

                for (k = 0U; k < n; k++)
                {
                    if (k != i)
                    {
                        factor = aug.Data[MATRIX_INDEX(&aug, k, i)];
                        for (j = i; j < (2U * n); j++)
                        {
                            aug.Data[MATRIX_INDEX(&aug, k, j)] -=
                                Matrix_MulQ31(factor, aug.Data[MATRIX_INDEX(&aug, i, j)]);
                        }
                    }
                    else
                    {
                        /* Skip pivot row – no action. */
                    }
                }
            }
        }

        if (singular != FALSE)
        {
            status = MATRIX_ERROR_SINGULAR;
        }
        else
        {
            for (i = 0U; i < n; i++)
            {
                for (j = 0U; j < n; j++)
                {
                    Result_P->Data[MATRIX_INDEX(Result_P, i, j)] =
                        aug.Data[MATRIX_INDEX(&aug, i, n + j)];
                }
            }

            Result_P->Rows = n;
            Result_P->Cols = n;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Inverse  (dispatcher)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Inverse(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type * const Result_P)
{
    MatrixStatus_Type status;
    uint32_T          n;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if ((Result_P->MaxRows < Matrix_P->Rows) || (Result_P->MaxCols < Matrix_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = Matrix_P->Rows;

        if      (n == 2U) { status = Matrix_Inverse2x2(Matrix_P, Result_P); }
        else if (n == 3U) { status = Matrix_Inverse3x3(Matrix_P, Result_P); }
        else if (n == 4U) { status = Matrix_Inverse4x4(Matrix_P, Result_P); }
        else
        {
            /* Generic augmented Gauss-Jordan for 5 × 5 … 8 × 8. */
            MatrixElement aug_buffer[MATRIX_MAX_ROWS * (2U * MATRIX_MAX_COLS)];
            Matrix_Type   aug;
            uint32_T      i;
            uint32_T      j;
            uint32_T      k;
            uint32_T      max_row;
            MatrixElement max_val;
            MatrixElement val;
            MatrixElement temp;
            MatrixElement pivot;
            MatrixElement factor;
            boolean_T     singular;

            singular = FALSE;
            Matrix_Init(&aug, aug_buffer, n, 2U * n);

            for (i = 0U; i < n; i++)
            {
                for (j = 0U; j < n; j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] =
                        Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)];
                }
            }

            for (i = 0U; i < n; i++)
            {
                for (j = n; j < (2U * n); j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] =
                        ((j - n) == i) ? Q31_ONE : Q31_ZERO;
                }
            }

            for (i = 0U; (i < n) && (singular == FALSE); i++)
            {
                max_row = i;
                max_val = Matrix_AbsQ31(aug.Data[MATRIX_INDEX(&aug, i, i)]);

                for (k = i + 1U; k < n; k++)
                {
                    val = Matrix_AbsQ31(aug.Data[MATRIX_INDEX(&aug, k, i)]);
                    if (val > max_val)
                    {
                        max_val = val;
                        max_row = k;
                    }
                }

                if (max_val < ZERO_THRESHOLD_Q31)
                {
                    singular = TRUE;
                }
                else
                {
                    if (max_row != i)
                    {
                        for (j = 0U; j < (2U * n); j++)
                        {
                            temp                                      = aug.Data[MATRIX_INDEX(&aug, i,       j)];
                            aug.Data[MATRIX_INDEX(&aug, i,       j)] = aug.Data[MATRIX_INDEX(&aug, max_row, j)];
                            aug.Data[MATRIX_INDEX(&aug, max_row, j)] = temp;
                        }
                    }
                    else
                    {
                        /* No action. */
                    }

                    pivot = aug.Data[MATRIX_INDEX(&aug, i, i)];
                    for (j = i; j < (2U * n); j++)
                    {
                        aug.Data[MATRIX_INDEX(&aug, i, j)] =
                            Matrix_DivQ31(aug.Data[MATRIX_INDEX(&aug, i, j)], pivot);
                    }

                    for (k = 0U; k < n; k++)
                    {
                        if (k != i)
                        {
                            factor = aug.Data[MATRIX_INDEX(&aug, k, i)];
                            for (j = i; j < (2U * n); j++)
                            {
                                aug.Data[MATRIX_INDEX(&aug, k, j)] -=
                                    Matrix_MulQ31(factor, aug.Data[MATRIX_INDEX(&aug, i, j)]);
                            }
                        }
                        else
                        {
                            /* No action. */
                        }
                    }
                }
            }

            if (singular != FALSE)
            {
                status = MATRIX_ERROR_SINGULAR;
            }
            else
            {
                for (i = 0U; i < n; i++)
                {
                    for (j = 0U; j < n; j++)
                    {
                        Result_P->Data[MATRIX_INDEX(Result_P, i, j)] =
                            aug.Data[MATRIX_INDEX(&aug, i, n + j)];
                    }
                }

                Result_P->Rows = n;
                Result_P->Cols = n;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Eigenvalues  (iterative Jacobi)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Eigenvalues(
    Matrix_Type      * const Matrix_P,
    MatrixEigen_Type * const EigenOut_P,
    const uint32_T MaxIterations,
    const MatrixFloat        Tolerance)
{
    MatrixStatus_Type status;
    uint32_T          n;
    uint32_T          iter;
    uint32_T          p;
    uint32_T          q;
    uint32_T          i;
    MatrixFloat       max_off_diag;
    MatrixFloat       app;
    MatrixFloat       aqq;
    MatrixFloat       apq;
    MatrixFloat       theta;
    MatrixFloat       c;
    MatrixFloat       s;
    MatrixFloat       temp;
    Matrix_Type       v_matrix;
    MatrixElement     v_buffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    boolean_T         use_tolerance;
    uint32_T          MaxIter;
    boolean_T         converged_early;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (EigenOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        n             = Matrix_P->Rows;
        MaxIter      = (MaxIterations > 0U) ? MaxIterations : JACOBI_MAX_ITER;
        use_tolerance = (Tolerance > 0.0f) ? TRUE : FALSE;
        converged_early = FALSE;

        EigenOut_P->NumEigenvalues = n;
        EigenOut_P->Iterations      = 0U;

        /* Initialise eigenvector accumulator to I. */
        Matrix_Init(&v_matrix, v_buffer, n, n);
        (void)Matrix_Identity(&v_matrix);

        for (iter = 0U; iter < MaxIter; iter++)
        {
            uint32_T    j;
            MatrixFloat val;
            MatrixFloat a_ip;
            MatrixFloat a_iq;
            MatrixFloat v_ip;
            MatrixFloat v_iq;
            boolean_T   converged;

            EigenOut_P->Iterations = iter + 1U;

            /* Find largest off-diagonal element to use as pivot. */
            max_off_diag = 0.0f;
            p = 0U;
            q = 1U;

            for (i = 0U; i < n; i++)
            {
                for (j = i + 1U; j < n; j++)
                {
                    val = ABS_FLOAT(
                        Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)]));
                    if (val > max_off_diag)
                    {
                        max_off_diag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            /* Check convergence. */
            if ((use_tolerance != FALSE) && (max_off_diag < Tolerance))
            {
                converged = TRUE;
            }
            else if (max_off_diag < JACOBI_TOLERANCE)
            {
                converged = TRUE;
            }
            else
            {
                converged = FALSE;
            }

            if (converged != FALSE)
            {
                converged_early = TRUE;
                iter = MaxIter; /* Force loop termination. */
            }
            else
            {
                /* Apply Jacobi rotation for (p, q). */
                app = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, p, p)]);
                aqq = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, q, q)]);
                apq = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, p, q)]);

                theta = (ABS_FLOAT(aqq - app) < ZERO_THRESHOLD_FLOAT)
                      ? JACOBI_PI_OVER_4
                      : 0.5f * atan2f(2.0f * apq, aqq - app);

                c = cosf(theta);
                s = sinf(theta);

                /* Update diagonal elements. */
                temp = app;
                app  = (c * c * temp) - (2.0f * c * s * apq) + (s * s * aqq);
                aqq  = (s * s * temp) + (2.0f * c * s * apq) + (c * c * aqq);

                Matrix_P->Data[MATRIX_INDEX(Matrix_P, p, p)] = Matrix_FloatToQ31(app);
                Matrix_P->Data[MATRIX_INDEX(Matrix_P, q, q)] = Matrix_FloatToQ31(aqq);
                Matrix_P->Data[MATRIX_INDEX(Matrix_P, p, q)] = Q31_ZERO;
                Matrix_P->Data[MATRIX_INDEX(Matrix_P, q, p)] = Q31_ZERO;

                /* Update off-diagonal rows. */
                for (i = 0U; i < n; i++)
                {
                    if ((i != p) && (i != q))
                    {
                        a_ip = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, p)]);
                        a_iq = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, q)]);

                        Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, p)] =
                            Matrix_FloatToQ31((c * a_ip) - (s * a_iq));
                        Matrix_P->Data[MATRIX_INDEX(Matrix_P, p, i)] =
                            Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, p)];

                        Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, q)] =
                            Matrix_FloatToQ31((s * a_ip) + (c * a_iq));
                        Matrix_P->Data[MATRIX_INDEX(Matrix_P, q, i)] =
                            Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, q)];
                    }
                    else
                    {
                        /* No action for pivot rows. */
                    }
                }

                /* Accumulate rotation into V. */
                for (i = 0U; i < n; i++)
                {
                    v_ip = Matrix_Q31ToFloat(v_matrix.Data[MATRIX_INDEX(&v_matrix, i, p)]);
                    v_iq = Matrix_Q31ToFloat(v_matrix.Data[MATRIX_INDEX(&v_matrix, i, q)]);

                    v_matrix.Data[MATRIX_INDEX(&v_matrix, i, p)] =
                        Matrix_FloatToQ31((c * v_ip) - (s * v_iq));
                    v_matrix.Data[MATRIX_INDEX(&v_matrix, i, q)] =
                        Matrix_FloatToQ31((s * v_ip) + (c * v_iq));
                }
            }
        }

        /* Copy eigenvalues from the diagonalised Matrix_P. */
        for (i = 0U; i < n; i++)
        {
            EigenOut_P->Eigenvalues[i] =
                Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, i)]);
        }

        /* Copy eigenvectors (column-wise) from V. */
        {
            uint32_T i2;
            uint32_T j2;
            for (i2 = 0U; i2 < n; i2++)
            {
                for (j2 = 0U; j2 < n; j2++)
                {
                    EigenOut_P->Eigenvectors[(i2 * n) + j2] =
                        Matrix_Q31ToFloat(v_matrix.Data[MATRIX_INDEX(&v_matrix, i2, j2)]);
                }
            }
        }

        if ((iter >= MaxIter) && (converged_early == FALSE))
        {
            status = MATRIX_ERROR_MAX_ITERATIONS;
        }
        else
        {
            /* Converged within iteration budget – no action. */
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_EigenvaluesOnly
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_EigenvaluesOnly(
    Matrix_Type    * const Matrix_P,
    MatrixFloat    * const EigenvaluesOut_P,
    const uint32_T MaxIterations,
    const MatrixFloat      Tolerance)
{
    MatrixStatus_Type status;
    MatrixEigen_Type  EigenOut_P;
    uint32_T          i;

    status = Matrix_Eigenvalues(Matrix_P, &EigenOut_P, MaxIterations, Tolerance);

    if (status == MATRIX_SUCCESS)
    {
        for (i = 0U; i < EigenOut_P.NumEigenvalues; i++)
        {
            EigenvaluesOut_P[i] = EigenOut_P.Eigenvalues[i];
        }
    }
    else
    {
        /* Error propagated from Matrix_Eigenvalues – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_LU
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_LU(
    Matrix_Type  * const Matrix_P,
    uint32_T     * const pivot)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          pivot_row;
    MatrixElement     factor;
    MatrixElement     temp;
    uint32_T          n;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (pivot == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        n = Matrix_P->Rows;

        for (i = 0U; i < n; i++)
        {
            pivot[i] = i;
        }

        for (k = 0U; (k < (n - 1U)) && (status == MATRIX_SUCCESS); k++)
        {
            /* Partial pivot search. */
            pivot_row = k;
            for (i = k + 1U; i < n; i++)
            {
                if (Matrix_AbsQ31(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, k)]) >
                    Matrix_AbsQ31(Matrix_P->Data[MATRIX_INDEX(Matrix_P, pivot_row, k)]))
                {
                    pivot_row = i;
                }
            }

            if (pivot_row != k)
            {
                for (j = 0U; j < n; j++)
                {
                    temp                                       = Matrix_P->Data[MATRIX_INDEX(Matrix_P, k,         j)];
                    Matrix_P->Data[MATRIX_INDEX(Matrix_P, k,         j)] = Matrix_P->Data[MATRIX_INDEX(Matrix_P, pivot_row, j)];
                    Matrix_P->Data[MATRIX_INDEX(Matrix_P, pivot_row, j)] = temp;
                }

                i             = pivot[k];
                pivot[k]      = pivot[pivot_row];
                pivot[pivot_row] = i;
            }
            else
            {
                /* No row swap required – no action. */
            }

            if (Matrix_AbsQ31(Matrix_P->Data[MATRIX_INDEX(Matrix_P, k, k)]) < ZERO_THRESHOLD_Q31)
            {
                status = MATRIX_ERROR_SINGULAR;
            }
            else
            {
                for (i = k + 1U; i < n; i++)
                {
                    factor = Matrix_DivQ31(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, k)],
                                          Matrix_P->Data[MATRIX_INDEX(Matrix_P, k, k)]);
                    Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, k)] = factor;

                    for (j = k + 1U; j < n; j++)
                    {
                        Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] -=
                            Matrix_MulQ31(factor, Matrix_P->Data[MATRIX_INDEX(Matrix_P, k, j)]);
                    }
                }
            }
        }

        /* Explicit check of the last diagonal element: the k-loop runs from 0 to n-2
         * and only tests Matrix_P[k][k] before factoring rows k+1 … n-1.  The final
         * U diagonal element Matrix_P[n-1][n-1] is never tested inside the loop and
         * must be checked here.                                                     */
        if ((status == MATRIX_SUCCESS) && (n > 0U))
        {
            if (Matrix_AbsQ31(Matrix_P->Data[MATRIX_INDEX(Matrix_P, n - 1U, n - 1U)])
                    < ZERO_THRESHOLD_Q31)
            {
                status = MATRIX_ERROR_SINGULAR;
            }
            else
            {
                /* Last diagonal is non-zero – no action. */
            }
        }
        else
        {
            /* Prior error or n == 0 – no action. */
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SolveGaussJordan
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_SolveGaussJordan(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P)
{
    MatrixStatus_Type status;
    MatrixElement     aug_buffer[MATRIX_MAX_ROWS * (MATRIX_MAX_COLS + MATRIX_MAX_COLS)];
    Matrix_Type       aug;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    uint32_T          m;
    MatrixElement     pivot;
    MatrixElement     factor;
    boolean_T         singular;

    singular = FALSE;
    status   = MATRIX_SUCCESS;

    if ((A_P == NULL) || (B_P == NULL) || (X_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (A_P->Rows != A_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if (A_P->Rows != B_P->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((X_P->MaxRows < A_P->Rows) || (X_P->MaxCols < B_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = A_P->Rows;
        m = B_P->Cols;

        Matrix_Init(&aug, aug_buffer, n, n + m);

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < n; j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, j)] =
                    A_P->Data[MATRIX_INDEX(A_P, i, j)];
            }
        }

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < m; j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, n + j)] =
                    B_P->Data[MATRIX_INDEX(B_P, i, j)];
            }
        }

        for (i = 0U; (i < n) && (singular == FALSE); i++)
        {
            pivot = aug.Data[MATRIX_INDEX(&aug, i, i)];

            if (Matrix_AbsQ31(pivot) < ZERO_THRESHOLD_Q31)
            {
                singular = TRUE;
            }
            else
            {
                for (j = i; j < (n + m); j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] =
                        Matrix_DivQ31(aug.Data[MATRIX_INDEX(&aug, i, j)], pivot);
                }

                for (k = 0U; k < n; k++)
                {
                    if (k != i)
                    {
                        factor = aug.Data[MATRIX_INDEX(&aug, k, i)];
                        for (j = i; j < (n + m); j++)
                        {
                            aug.Data[MATRIX_INDEX(&aug, k, j)] -=
                                Matrix_MulQ31(factor, aug.Data[MATRIX_INDEX(&aug, i, j)]);
                        }
                    }
                    else
                    {
                        /* Skip pivot row – no action. */
                    }
                }
            }
        }

        if (singular != FALSE)
        {
            status = MATRIX_ERROR_SINGULAR;
        }
        else
        {
            for (i = 0U; i < n; i++)
            {
                for (j = 0U; j < m; j++)
                {
                    X_P->Data[MATRIX_INDEX(X_P, i, j)] =
                        aug.Data[MATRIX_INDEX(&aug, i, n + j)];
                }
            }

            X_P->Rows = n;
            X_P->Cols = m;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Solve  (delegates to Gauss-Jordan)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Solve(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P)
{
    return Matrix_SolveGaussJordan(A_P, B_P, X_P);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsSquare
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsSquare(const Matrix_Type * const Matrix_P)
{
    boolean_T result;

    result = FALSE;

    if ((Matrix_P != NULL) && (Matrix_P->Rows == Matrix_P->Cols))
    {
        result = TRUE;
    }
    else
    {
        /* Not square or NULL – no action. */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsSymmetric
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsSymmetric(
    const Matrix_Type  * const Matrix_P,
    const MatrixFloat          Tolerance)
{
    boolean_T   result;
    uint32_T    i;
    uint32_T    j;
    MatrixFloat a_ij;
    MatrixFloat a_ji;
    MatrixFloat diff;
    MatrixFloat tol;

    result = TRUE;
    tol    = (Tolerance > 0.0f) ? Tolerance : ZERO_THRESHOLD_FLOAT;

    if (Matrix_P == NULL)
    {
        result = FALSE;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        result = FALSE;
    }
    else
    {
        for (i = 0U; (i < Matrix_P->Rows) && (result != FALSE); i++)
        {
            for (j = i + 1U; (j < Matrix_P->Cols) && (result != FALSE); j++)
            {
                a_ij = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)]);
                a_ji = Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, j, i)]);
                diff = ABS_FLOAT(a_ij - a_ji);

                if (diff > tol)
                {
                    result = FALSE;
                }
                else
                {
                    /* Symmetric at (i, j) – no action. */
                }
            }
        }
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Trace
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Trace(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const TraceOut_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    MatrixFloat       sum;

    sum    = 0.0f;
    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (TraceOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        for (i = 0U; i < Matrix_P->Rows; i++)
        {
            sum += Matrix_Q31ToFloat(Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, i)]);
        }

        *TraceOut_P = sum;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_NormFrobenius
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_NormFrobenius(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const NormOut_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    real64_T          sum;
    real64_T          val;

    sum    = 0.0;
    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (NormOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        for (i = 0U; i < Matrix_P->Rows; i++)
        {
            for (j = 0U; j < Matrix_P->Cols; j++)
            {
                val  = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] / Q31_SCALE_D;
                sum += val * val;
            }
        }

        *NormOut_P = (MatrixFloat)sqrt(sum);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsEqual
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsEqual(
    const Matrix_Type  * const A_P,
    const Matrix_Type  * const B_P,
    const MatrixFloat          Tolerance)
{
    boolean_T result;
    uint32_T  i;
    uint32_T  j;
    real64_T  diff;
    real64_T  tol;
    real64_T  a_val;
    real64_T  b_val;

    result = TRUE;
    tol    = (Tolerance > 0.0f) ? (real64_T)Tolerance : (real64_T)ZERO_THRESHOLD_FLOAT;

    if ((A_P == NULL) || (B_P == NULL))
    {
        result = FALSE;
    }
    else if ((A_P->Rows != B_P->Rows) || (A_P->Cols != B_P->Cols))
    {
        result = FALSE;
    }
    else
    {
        for (i = 0U; (i < A_P->Rows) && (result != FALSE); i++)
        {
            for (j = 0U; (j < A_P->Cols) && (result != FALSE); j++)
            {
                a_val = (real64_T)A_P->Data[MATRIX_INDEX(A_P, i, j)] / Q31_SCALE_D;
                b_val = (real64_T)B_P->Data[MATRIX_INDEX(B_P, i, j)] / Q31_SCALE_D;
                diff  = (a_val > b_val) ? (a_val - b_val) : (b_val - a_val);

                if (diff > tol)
                {
                    result = FALSE;
                }
                else
                {
                    /* Within Tolerance – no action. */
                }
            }
        }
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_GetDimensions
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_GetDimensions(
    const Matrix_Type  * const Matrix_P,
    uint32_T           * const RowsOut_P,
    uint32_T           * const ColsOut_P)
{
    if ((Matrix_P != NULL) && (RowsOut_P != NULL) && (ColsOut_P != NULL))
    {
        *RowsOut_P = Matrix_P->Rows;
        *ColsOut_P = Matrix_P->Cols;
    }
    else
    {
        /* MISRA C 2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Fill
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_Fill(
    Matrix_Type        * const Matrix_P,
    const MatrixElement        value)
{
    uint32_T i;
    uint32_T j;

    if (Matrix_P != NULL)
    {
        for (i = 0U; i < Matrix_P->Rows; i++)
        {
            for (j = 0U; j < Matrix_P->Cols; j++)
            {
                Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] = value;
            }
        }
    }
    else
    {
        /* MISRA C 2012 Rule 15.7: else clause required. */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_FillFloat
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_FillFloat(
    Matrix_Type    * const Matrix_P,
    const MatrixFloat      value)
{
    Matrix_Fill(Matrix_P, Matrix_FloatToQ31(value));
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_FloatToQ31
 *------------------------------------------------------------------------------------------------------------------*/
MatrixElement Matrix_FloatToQ31(const MatrixFloat Value)
{
    MatrixElement result;
    MatrixFloat   clamped;

    /* Clamp to Q31 representable range [-1.0, +0.9999999995]. */
    if (Value > 0.9999999995f)
    {
        clamped = 0.9999999995f;
    }
    else if (Value < -1.0f)
    {
        clamped = -1.0f;
    }
    else
    {
        clamped = Value;
    }

    result = (MatrixElement)(clamped * Q31_SCALE_F);

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Q31ToFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixFloat Matrix_Q31ToFloat(const MatrixElement value)
{
    return (MatrixFloat)value / Q31_SCALE_F;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Cholesky
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_Cholesky(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type        * const L_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    real64_T          sum;
    real64_T          val;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (L_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if ((L_P->MaxRows < Matrix_P->Rows) || (L_P->MaxCols < Matrix_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = Matrix_P->Rows;

        /* Initialize L to zero */
        Matrix_Zero(L_P);
        Matrix_SetDimensions(L_P, n, n);

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j <= i; j++)
            {
                sum = 0.0;

                /* Sum over k = 0 to j-1 */
                for (k = 0U; k < j; k++)
                {
                    val = (real64_T)L_P->Data[MATRIX_INDEX(L_P, i, k)] *
                          (real64_T)L_P->Data[MATRIX_INDEX(L_P, j, k)];
                    sum += val;
                }

                if (i == j)
                {
                    /* Diagonal element: L[i][i] = sqrt(A[i][i] - sum) */
                    real64_T a_ii = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, i)] / Q31_SCALE_D;
                    real64_T diff = a_ii - sum;

                    if (diff <= 0.0)
                    {
                        status = MATRIX_ERROR_NON_POSITIVE_DEFINITE;
                        break;
                    }

                    L_P->Data[MATRIX_INDEX(L_P, i, i)] =
                        Matrix_FloatToQ31((MatrixFloat)sqrt(diff));
                }
                else
                {
                    /* Off-diagonal: L[i][j] = (A[i][j] - sum) / L[j][j] */
                    real64_T a_ij = (real64_T)Matrix_P->Data[MATRIX_INDEX(Matrix_P, i, j)] / Q31_SCALE_D;
                    real64_T l_jj = (real64_T)L_P->Data[MATRIX_INDEX(L_P, j, j)] / Q31_SCALE_D;
                    real64_T result;

                    if (ABS_FLOAT((MatrixFloat)l_jj) < ZERO_THRESHOLD_FLOAT)
                    {
                        status = MATRIX_ERROR_NON_POSITIVE_DEFINITE;
                        break;
                    }

                    result = (a_ij - sum) / l_jj;
                    L_P->Data[MATRIX_INDEX(L_P, i, j)] = Matrix_FloatToQ31((MatrixFloat)result);
                }
            }

            if (status != MATRIX_SUCCESS)
            {
                break;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_ForwardSubstitution
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_ForwardSubstitution(
    const Matrix_Type  * const L_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          n;
    uint32_T          m;
    real64_T          sum;
    real64_T          l_ii;

    status = MATRIX_SUCCESS;

    if ((L_P == NULL) || (B_P == NULL) || (X_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (L_P->Rows != L_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if (L_P->Rows != B_P->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((X_P->MaxRows < B_P->Rows) || (X_P->MaxCols < B_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = L_P->Rows;
        m = B_P->Cols;

        Matrix_SetDimensions(X_P, n, m);
        Matrix_Zero(X_P);

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < m; j++)
            {
                sum = 0.0;

                for (uint32_T k = 0U; k < i; k++)
                {
                    sum += (real64_T)L_P->Data[MATRIX_INDEX(L_P, i, k)] *
                           (real64_T)X_P->Data[MATRIX_INDEX(X_P, k, j)];
                }

                l_ii = (real64_T)L_P->Data[MATRIX_INDEX(L_P, i, i)] / Q31_SCALE_D;

                if (ABS_FLOAT((MatrixFloat)l_ii) < ZERO_THRESHOLD_FLOAT)
                {
                    status = MATRIX_ERROR_SINGULAR;
                    break;
                }

                real64_T b_ij = (real64_T)B_P->Data[MATRIX_INDEX(B_P, i, j)] / Q31_SCALE_D;
                real64_T result = (b_ij - sum) / l_ii;

                X_P->Data[MATRIX_INDEX(X_P, i, j)] = Matrix_FloatToQ31((MatrixFloat)result);
            }

            if (status != MATRIX_SUCCESS)
            {
                break;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_BackwardSubstitution
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_BackwardSubstitution(
    const Matrix_Type  * const U_P,
    const Matrix_Type  * const B_P,
    Matrix_Type        * const X_P)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    uint32_T          m;
    real64_T          sum;
    real64_T          u_ii;
    int32_T           i_signed;

    status = MATRIX_SUCCESS;

    if ((U_P == NULL) || (B_P == NULL) || (X_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (U_P->Rows != U_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if (U_P->Rows != B_P->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((X_P->MaxRows < B_P->Rows) || (X_P->MaxCols < B_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = U_P->Rows;
        m = B_P->Cols;

        Matrix_SetDimensions(X_P, n, m);
        Matrix_Zero(X_P);

        for (i_signed = (int32_T)n - 1; i_signed >= 0; i_signed--)
        {
            i = (uint32_T)i_signed;

            for (j = 0U; j < m; j++)
            {
                sum = 0.0;

                for (k = i + 1U; k < n; k++)
                {
                    sum += (real64_T)U_P->Data[MATRIX_INDEX(U_P, i, k)] *
                           (real64_T)X_P->Data[MATRIX_INDEX(X_P, k, j)];
                }

                u_ii = (real64_T)U_P->Data[MATRIX_INDEX(U_P, i, i)] / Q31_SCALE_D;

                if (ABS_FLOAT((MatrixFloat)u_ii) < ZERO_THRESHOLD_FLOAT)
                {
                    status = MATRIX_ERROR_SINGULAR;
                    break;
                }

                real64_T b_ij = (real64_T)B_P->Data[MATRIX_INDEX(B_P, i, j)] / Q31_SCALE_D;
                real64_T result = (b_ij - sum) / u_ii;

                X_P->Data[MATRIX_INDEX(X_P, i, j)] = Matrix_FloatToQ31((MatrixFloat)result);
            }

            if (status != MATRIX_SUCCESS)
            {
                break;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SymmetricRank1Update
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_SymmetricRank1Update(
    Matrix_Type        * const A_P,
    const Matrix_Type  * const V_P,
    const MatrixElement        alpha)
{
    MatrixStatus_Type status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          n;
    MatrixElement     scaled_v_i;
    MatrixElement     scaled_v_j;
    MatrixElement     update;

    status = MATRIX_SUCCESS;

    if ((A_P == NULL) || (V_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((A_P->Rows != A_P->Cols) || (V_P->Rows != A_P->Rows) || (V_P->Cols != 1U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        n = A_P->Rows;

        for (i = 0U; i < n; i++)
        {
            scaled_v_i = Matrix_MulQ31(V_P->Data[MATRIX_INDEX(V_P, i, 0U)], alpha);

            for (j = 0U; j <= i; j++)
            {
                scaled_v_j = Matrix_MulQ31(V_P->Data[MATRIX_INDEX(V_P, j, 0U)], alpha);
                update = Matrix_MulQ31(scaled_v_i, V_P->Data[MATRIX_INDEX(V_P, j, 0U)]);

                /* A[i][j] += update */
                MatrixElement current = A_P->Data[MATRIX_INDEX(A_P, i, j)];
                MatrixElement new_val = current + update;

                /* Saturation */
                if ((current > 0) && (update > 0) && (new_val < 0))
                {
                    new_val = Q31_ONE;
                }
                else if ((current < 0) && (update < 0) && (new_val > 0))
                {
                    new_val = Q31_MINUS_ONE;
                }

                A_P->Data[MATRIX_INDEX(A_P, i, j)] = new_val;
                A_P->Data[MATRIX_INDEX(A_P, j, i)] = new_val; /* Maintain symmetry */
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SymmetricRank1UpdateFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_SymmetricRank1UpdateFloat(
    Matrix_Type        * const A_P,
    const Matrix_Type  * const V_P,
    const MatrixFloat          alpha)
{
    return Matrix_SymmetricRank1Update(A_P, V_P, Matrix_FloatToQ31(alpha));
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_MatrixSquareRoot (Denman-Beavers iteration)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_MatrixSquareRoot(
    const Matrix_Type  * const Matrix_P,
    Matrix_Type * const Result_P,
    const uint32_T MaxIter)
{
    MatrixStatus_Type status;
    MatrixElement     Y_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     Z_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     Y_inv_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     Y_new_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     Z_new_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       Y;
    Matrix_Type       Z;
    Matrix_Type       Y_inv;
    Matrix_Type       Y_new;
    Matrix_Type       Z_new;
    uint32_T          iter;
    uint32_T          max_iters;
    MatrixFloat       norm_diff;
    MatrixFloat       norm_prev;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (Result_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if ((Result_P->MaxRows < Matrix_P->Rows) || (Result_P->MaxCols < Matrix_P->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        max_iters = (MaxIter > 0U) ? MaxIter : 10U;

        Matrix_Init(&Y, Y_buf, Matrix_P->Rows, Matrix_P->Cols);
        Matrix_Init(&Z, Z_buf, Matrix_P->Rows, Matrix_P->Cols);
        Matrix_Init(&Y_inv, Y_inv_buf, Matrix_P->Rows, Matrix_P->Cols);
        Matrix_Init(&Y_new, Y_new_buf, Matrix_P->Rows, Matrix_P->Cols);
        Matrix_Init(&Z_new, Z_new_buf, Matrix_P->Rows, Matrix_P->Cols);

        /* Initialize: Y0 = A, Z0 = I */
        (void)Matrix_Copy(&Y, Matrix_P);
        (void)Matrix_Identity(&Z);

        norm_prev = 0.0f;

        for (iter = 0U; iter < max_iters; iter++)
        {
            /* Y_inv = inv(Y) */
            status = Matrix_Inverse(&Y, &Y_inv);
            if (status != MATRIX_SUCCESS)
            {
                break;
            }

            /* Y_new = 0.5 * (Y + Z_inv) but we use Y_inv as Z = inv(Y) */
            Matrix_Add(&Y, &Z, &Y_new);
            Matrix_ScalarMultiplyFloat(&Y_new, 0.5f, &Y_new);

            /* Z_new = 0.5 * (Z + Y_inv) */
            Matrix_Add(&Z, &Y_inv, &Z_new);
            Matrix_ScalarMultiplyFloat(&Z_new, 0.5f, &Z_new);

            /* Check convergence */
            Matrix_Subtract(&Y_new, &Y, &Y);
            Matrix_NormFrobenius(&Y, &norm_diff);

            if (norm_diff < 1e-6f)
            {
                break;
            }

            /* Update */
            (void)Matrix_Copy(&Y, &Y_new);
            (void)Matrix_Copy(&Z, &Z_new);
            norm_prev = norm_diff;
        }

        if (iter >= max_iters)
        {
            status = MATRIX_ERROR_MAX_ITERATIONS;
        }
        else
        {
            /* Result is lower triangular Cholesky-like factor */
            (void)Matrix_Cholesky(&Y_new, Result_P);
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_ConditionNumber
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Matrix_ConditionNumber(
    const Matrix_Type  * const Matrix_P,
    MatrixFloat        * const CondOut_P)
{
    MatrixStatus_Type status;
    MatrixElement     inv_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       inv_matrix;
    MatrixFloat       norm_A;
    MatrixFloat       norm_Ainv;

    status = MATRIX_SUCCESS;

    if ((Matrix_P == NULL) || (CondOut_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_P->Rows != Matrix_P->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        Matrix_Init(&inv_matrix, inv_buf, Matrix_P->Rows, Matrix_P->Cols);

        /* Compute 1-norm of A */
        Matrix_NormFrobenius(Matrix_P, &norm_A);

        /* Compute inverse */
        status = Matrix_Inverse(Matrix_P, &inv_matrix);

        if (status == MATRIX_SUCCESS)
        {
            /* Compute 1-norm of A^-1 */
            Matrix_NormFrobenius(&inv_matrix, &norm_Ainv);
            *CondOut_P = norm_A * norm_Ainv;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsPositiveDefinite
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsPositiveDefinite(const Matrix_Type * const Matrix_P)
{
    MatrixStatus_Type status;
    MatrixElement     L_buf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_Type       L;
    boolean_T         result;

    result = FALSE;

    if ((Matrix_P != NULL) && (Matrix_P->Rows == Matrix_P->Cols))
    {
        Matrix_Init(&L, L_buf, Matrix_P->Rows, Matrix_P->Cols);
        status = Matrix_Cholesky(Matrix_P, &L);

        if (status == MATRIX_SUCCESS)
        {
            result = TRUE;
        }
        else
        {
            /* Not positive definite */
        }
    }
    else
    {
        /* NULL or not square */
    }

    return result;
}
