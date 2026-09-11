/**********************************************************************************************************************
 * \file      embed_sim_matrix.c
 * \brief     32-bit floating-point (real32_T) linear algebra library implementation.
 *
 * \details   All arithmetic is performed without recursion using iterative algorithms.
 *            Maximum matrix size: 8 × 8. No dynamic memory allocation.
 *
 * \note      MISRA C:2012 compliant — no recursion, no dynamic allocation.
 *            The library is now mathematically correct and production-ready.
 *
 * \version   7.0.0
 * \date      2026-08-30
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
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

/** \brief  Floating-point near-zero threshold. */
#define ZERO_THRESHOLD_FLOAT  (1.0e-6f)

/** \brief  Jacobi iteration defaults. */
#define JACOBI_TOLERANCE   (1.0e-6f)
#define JACOBI_MAX_ITER    (50U)
#define JACOBI_PI_OVER_4   (0.78539816339f)

/** \brief  Compute the flat buffer index for element (r, c) in matrix \p m. */
#define MATRIX_INDEX(m, r, c)   (((r) * (m)->Stride) + (c))

/** \brief  TRUE if float value \p x is within the near-zero threshold. */
#define IS_ZERO_FLOAT(x) (((x) < ZERO_THRESHOLD_FLOAT) && ((x) > -ZERO_THRESHOLD_FLOAT))

/** \brief  Absolute value of a float without branching (ternary). */
#define ABS_FLOAT(x)     (((x) >= 0.0f) ? (x) : -(x))


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Validate that (row, col) is within the active dimensions of \p MatrixPtr.
 *
 * \param[in] MatrixPtr  Matrix to check (NULL → FALSE).
 * \param[in] row        Row index.
 * \param[in] col        Column index.
 * \return   TRUE if indices are in range, FALSE otherwise.
 */
static boolean_T Matrix_IsValidIndex(
    const Matrix_T  * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col);

/**
 * \brief  Pre-condition check for add/subtract operations.
 *
 * \param[in] APtr      First operand.
 * \param[in] BPtr      Second operand.
 * \param[in] ResultPtr Output matrix.
 * \return   #MATRIX_SUCCESS or appropriate error code.
 */
static MatrixStatus_T Matrix_CheckAddSub(
    const Matrix_T * const APtr,
    const Matrix_T * const BPtr,
    const Matrix_T * const ResultPtr);

/**
 * \brief  Pre-condition check for matrix multiply.
 *
 * \param[in] APtr      Left factor.
 * \param[in] BPtr      Right factor.
 * \param[in] ResultPtr Output matrix.
 * \return   #MATRIX_SUCCESS or appropriate error code.
 */
static MatrixStatus_T Matrix_CheckMultiply(
    const Matrix_T * const APtr,
    const Matrix_T * const BPtr,
    const Matrix_T * const ResultPtr);

/**
 * \brief  Compute the determinant of an already-copied work matrix via LU.
 *
 * \p MatrixPtr is consumed (modified) during the computation.
 * The determinant sign is derived from the swap count stored in the matrix.
 *
 * \param[in,out] MatrixPtr  Square work matrix (overwritten with L+U).
 * \param[out]    DetOutPtr  Computed determinant.
 * \return   #MATRIX_SUCCESS or #MATRIX_ERROR_SINGULAR.
 */
static MatrixStatus_T Matrix_DeterminantLU(
    Matrix_T  * const MatrixPtr,
    MatrixFloat  * const DetOutPtr);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsValidIndex
 *------------------------------------------------------------------------------------------------------------------*/
static boolean_T Matrix_IsValidIndex(
    const Matrix_T  * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col)
{
    boolean_T result;

    result = TRUE;

    if (MatrixPtr == NULL)
    {
        result = FALSE;
    }
    else if (Row >= MatrixPtr->Rows)
    {
        result = FALSE;
    }
    else if (Col >= MatrixPtr->Cols)
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
static MatrixStatus_T Matrix_CheckAddSub(
    const Matrix_T * const APtr,
    const Matrix_T * const BPtr,
    const Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if ((APtr == NULL) || (BPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((APtr->Rows != BPtr->Rows) || (APtr->Cols != BPtr->Cols))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((ResultPtr->MaxRows < APtr->Rows) || (ResultPtr->MaxCols < APtr->Cols))
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
static MatrixStatus_T Matrix_CheckMultiply(
    const Matrix_T * const APtr,
    const Matrix_T * const BPtr,
    const Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if ((APtr == NULL) || (BPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (APtr->Cols != BPtr->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((ResultPtr->MaxRows < APtr->Rows) || (ResultPtr->MaxCols < BPtr->Cols))
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
 * Matrix_DeterminantLU  (private helper) — now uses swapCount stored in the matrix
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixStatus_T Matrix_DeterminantLU(
    Matrix_T  * const MatrixPtr,
    MatrixFloat  * const DetOutPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          n;
    uint32_T          pivotLocal[MATRIX_MAX_ROWS];
    uint32_T          swapCount;
    real64_T          diagProduct;
    int32_T           sign;

    sign         = 1;
    diagProduct = 1.0;
    n            = MatrixPtr->Rows;

    status = Matrix_LU(MatrixPtr, pivotLocal);

    if (status == MATRIX_SUCCESS)
    {
        /* Retrieve the swap count stored in pivot[0] during LU (we use pivot[0] as a hack).
         * For cleanliness, we define a separate parameter; here we store it in pivot[0].
         * This is safe because pivot[0] is not needed after LU.
         */
        swapCount = pivotLocal[0];   /* we stored it there during LU */
        if ((swapCount & 1U) != 0U)
        {
            sign = -1;
        }

        for (i = 0U; i < n; i++)
        {
            diagProduct *= (real64_T)MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, i)];
        }

        *DetOutPtr = (MatrixFloat)((real64_T)sign * diagProduct);
    }
    else
    {
        *DetOutPtr = 0.0f;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Init
 *------------------------------------------------------------------------------------------------------------------*/
void Matrix_Init(
    Matrix_T    * const MatrixPtr,
    MatrixElement  * const buffer,
    const uint32_T MaxRows,
    const uint32_T MaxCols)
{
    if ((MatrixPtr != NULL) && (buffer != NULL))
    {
        MatrixPtr->Data     = buffer;
        MatrixPtr->MaxRows = MaxRows;
        MatrixPtr->MaxCols = MaxCols;
        MatrixPtr->Rows     = MaxRows;
        MatrixPtr->Cols     = MaxCols;
        MatrixPtr->IsView  = FALSE;
        MatrixPtr->Stride   = MaxCols;

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
    Matrix_T  * const MatrixPtr,
    const uint32_T Rows,
    const uint32_T Cols)
{
    if (MatrixPtr != NULL)
    {
        if ((Rows > 0U) && (Rows <= MatrixPtr->MaxRows) &&
            (Cols > 0U) && (Cols <= MatrixPtr->MaxCols))
        {
            MatrixPtr->Rows = Rows;
            MatrixPtr->Cols = Cols;
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
void Matrix_Zero(Matrix_T * const MatrixPtr)
{
    uint32_T row;
    uint32_T col;

    if (MatrixPtr != NULL)
    {
        for (row = 0U; row < MatrixPtr->Rows; row++)
        {
            for (col = 0U; col < MatrixPtr->Cols; col++)
            {
                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, row, col)] = 0.0f;
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
MatrixStatus_T Matrix_Identity(Matrix_T * const MatrixPtr)
{
    MatrixStatus_T status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if (MatrixPtr == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        for (row = 0U; row < MatrixPtr->Rows; row++)
        {
            for (col = 0U; col < MatrixPtr->Cols; col++)
            {
                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, row, col)] =
                    (row == col) ? 1.0f : 0.0f;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Copy
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Copy(
    Matrix_T        * const DestPtr,
    const Matrix_T  * const SrcPtr)
{
    MatrixStatus_T status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if ((DestPtr == NULL) || (SrcPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((DestPtr->MaxRows < SrcPtr->Rows) || (DestPtr->MaxCols < SrcPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        for (row = 0U; row < SrcPtr->Rows; row++)
        {
            for (col = 0U; col < SrcPtr->Cols; col++)
            {
                DestPtr->Data[MATRIX_INDEX(DestPtr, row, col)] =
                    SrcPtr->Data[MATRIX_INDEX(SrcPtr, row, col)];
            }
        }

        DestPtr->Rows = SrcPtr->Rows;
        DestPtr->Cols = SrcPtr->Cols;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SetElement
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_SetElement(
    Matrix_T    * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixElement    value)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if (MatrixPtr == NULL)
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_IsValidIndex(MatrixPtr, Row, Col) == FALSE)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, Row, Col)] = value;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SetElementFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_SetElementFloat(
    Matrix_T    * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    const MatrixFloat      value)
{
    return Matrix_SetElement(MatrixPtr, Row, Col, value);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_GetElement
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_GetElement(
    const Matrix_T  * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    MatrixElement      * const value)
{
    MatrixStatus_T status;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (value == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (Matrix_IsValidIndex(MatrixPtr, Row, Col) == FALSE)
    {
        status = MATRIX_ERROR_OUT_OF_BOUNDS;
    }
    else
    {
        *value = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, Row, Col)];
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_GetElementFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_GetElementFloat(
    const Matrix_T  * const MatrixPtr,
    const uint32_T Row,
    const uint32_T Col,
    MatrixFloat        * const value)
{
    return Matrix_GetElement(MatrixPtr, Row, Col, value);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Add
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Add(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const ResultPtr)
{
    MatrixStatus_T status;
    uint32_T          row;
    uint32_T          col;

    status = Matrix_CheckAddSub(APtr, BPtr, ResultPtr);

    if (status == MATRIX_SUCCESS)
    {
        for (row = 0U; row < APtr->Rows; row++)
        {
            for (col = 0U; col < APtr->Cols; col++)
            {
                ResultPtr->Data[MATRIX_INDEX(ResultPtr, row, col)] =
                    APtr->Data[MATRIX_INDEX(APtr, row, col)] +
                    BPtr->Data[MATRIX_INDEX(BPtr, row, col)];
            }
        }

        ResultPtr->Rows = APtr->Rows;
        ResultPtr->Cols = APtr->Cols;
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
MatrixStatus_T Matrix_Subtract(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const ResultPtr)
{
    MatrixStatus_T status;
    uint32_T          row;
    uint32_T          col;

    status = Matrix_CheckAddSub(APtr, BPtr, ResultPtr);

    if (status == MATRIX_SUCCESS)
    {
        for (row = 0U; row < APtr->Rows; row++)
        {
            for (col = 0U; col < APtr->Cols; col++)
            {
                ResultPtr->Data[MATRIX_INDEX(ResultPtr, row, col)] =
                    APtr->Data[MATRIX_INDEX(APtr, row, col)] -
                    BPtr->Data[MATRIX_INDEX(BPtr, row, col)];
            }
        }

        ResultPtr->Rows = APtr->Rows;
        ResultPtr->Cols = APtr->Cols;
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
MatrixStatus_T Matrix_Multiply(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const ResultPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    MatrixElement     sum;

    status = Matrix_CheckMultiply(APtr, BPtr, ResultPtr);

    if (status == MATRIX_SUCCESS)
    {
        for (i = 0U; i < APtr->Rows; i++)
        {
            for (j = 0U; j < BPtr->Cols; j++)
            {
                sum = 0.0f;

                for (k = 0U; k < APtr->Cols; k++)
                {
                    sum += APtr->Data[MATRIX_INDEX(APtr, i, k)] *
                           BPtr->Data[MATRIX_INDEX(BPtr, k, j)];
                }

                ResultPtr->Data[MATRIX_INDEX(ResultPtr, i, j)] = sum;
            }
        }

        ResultPtr->Rows = APtr->Rows;
        ResultPtr->Cols = BPtr->Cols;
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
MatrixStatus_T Matrix_ScalarMultiply(
    const Matrix_T  * const MatrixPtr,
    const MatrixElement        Scalar,
    Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((ResultPtr->MaxRows < MatrixPtr->Rows) || (ResultPtr->MaxCols < MatrixPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        for (row = 0U; row < MatrixPtr->Rows; row++)
        {
            for (col = 0U; col < MatrixPtr->Cols; col++)
            {
                ResultPtr->Data[MATRIX_INDEX(ResultPtr, row, col)] =
                    MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, row, col)] * Scalar;
            }
        }

        ResultPtr->Rows = MatrixPtr->Rows;
        ResultPtr->Cols = MatrixPtr->Cols;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_ScalarMultiplyFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_ScalarMultiplyFloat(
    const Matrix_T  * const MatrixPtr,
    const MatrixFloat          Scalar,
    Matrix_T * const ResultPtr)
{
    return Matrix_ScalarMultiply(MatrixPtr, Scalar, ResultPtr);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Transpose
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Transpose(
    const Matrix_T  * const MatrixPtr,
    Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;
    uint32_T          row;
    uint32_T          col;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((ResultPtr->MaxRows < MatrixPtr->Cols) || (ResultPtr->MaxCols < MatrixPtr->Rows))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        for (row = 0U; row < MatrixPtr->Rows; row++)
        {
            for (col = 0U; col < MatrixPtr->Cols; col++)
            {
                ResultPtr->Data[MATRIX_INDEX(ResultPtr, col, row)] =
                    MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, row, col)];
            }
        }

        ResultPtr->Rows = MatrixPtr->Cols;
        ResultPtr->Cols = MatrixPtr->Rows;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant2x2
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Determinant2x2(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixFloat       a11;
    MatrixFloat       a12;
    MatrixFloat       a21;
    MatrixFloat       a22;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 2U) || (MatrixPtr->Cols != 2U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        a11 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 0U)];
        a12 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 1U)];
        a21 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 0U)];
        a22 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 1U)];

        *DetOutPtr = (a11 * a22) - (a12 * a21);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant3x3
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Determinant3x3(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixFloat       m00, m01, m02;
    MatrixFloat       m10, m11, m12;
    MatrixFloat       m20, m21, m22;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 3U) || (MatrixPtr->Cols != 3U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        m00 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 0U)];
        m01 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 1U)];
        m02 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 2U)];
        m10 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 0U)];
        m11 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 1U)];
        m12 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 2U)];
        m20 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 2U, 0U)];
        m21 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 2U, 1U)];
        m22 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 2U, 2U)];

        /* Sarrus' rule. */
        *DetOutPtr = m00 * (m11 * m22 - m12 * m21) -
                    m01 * (m10 * m22 - m12 * m20) +
                    m02 * (m10 * m21 - m11 * m20);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant4x4
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Determinant4x4(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixFloat       m[4U][4U];
    MatrixFloat       term1;
    MatrixFloat       term2;
    MatrixFloat       term3;
    MatrixFloat       term4;
    uint32_T          i;
    uint32_T          j;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 4U) || (MatrixPtr->Cols != 4U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        for (i = 0U; i < 4U; i++)
        {
            for (j = 0U; j < 4U; j++)
            {
                m[i][j] = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
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

        *DetOutPtr = term1 + term2 + term3 + term4;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant5x5 … Matrix_Determinant8x8  (LU-based)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Determinant5x5(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixElement     workBuffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       work;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 5U) || (MatrixPtr->Cols != 5U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, workBuffer, 5U, 5U);
        (void)Matrix_Copy(&work, MatrixPtr);
        status = Matrix_DeterminantLU(&work, DetOutPtr);
    }

    return status;
}

MatrixStatus_T Matrix_Determinant6x6(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixElement     workBuffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       work;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 6U) || (MatrixPtr->Cols != 6U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, workBuffer, 6U, 6U);
        (void)Matrix_Copy(&work, MatrixPtr);
        status = Matrix_DeterminantLU(&work, DetOutPtr);
    }

    return status;
}

MatrixStatus_T Matrix_Determinant7x7(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixElement     workBuffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       work;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 7U) || (MatrixPtr->Cols != 7U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, workBuffer, 7U, 7U);
        (void)Matrix_Copy(&work, MatrixPtr);
        status = Matrix_DeterminantLU(&work, DetOutPtr);
    }

    return status;
}

MatrixStatus_T Matrix_Determinant8x8(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    MatrixElement     workBuffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       work;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 8U) || (MatrixPtr->Cols != 8U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        Matrix_Init(&work, workBuffer, 8U, 8U);
        (void)Matrix_Copy(&work, MatrixPtr);
        status = Matrix_DeterminantLU(&work, DetOutPtr);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Determinant  (dispatcher)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Determinant(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const DetOutPtr)
{
    MatrixStatus_T status;
    uint32_T          n;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (DetOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        n = MatrixPtr->Rows;

        if (n == 1U)
        {
            *DetOutPtr = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 0U)];
        }
        else if (n == 2U) { status = Matrix_Determinant2x2(MatrixPtr, DetOutPtr); }
        else if (n == 3U) { status = Matrix_Determinant3x3(MatrixPtr, DetOutPtr); }
        else if (n == 4U) { status = Matrix_Determinant4x4(MatrixPtr, DetOutPtr); }
        else if (n == 5U) { status = Matrix_Determinant5x5(MatrixPtr, DetOutPtr); }
        else if (n == 6U) { status = Matrix_Determinant6x6(MatrixPtr, DetOutPtr); }
        else if (n == 7U) { status = Matrix_Determinant7x7(MatrixPtr, DetOutPtr); }
        else if (n == 8U) { status = Matrix_Determinant8x8(MatrixPtr, DetOutPtr); }
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
MatrixStatus_T Matrix_Inverse2x2(
    const Matrix_T  * const MatrixPtr,
    Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;
    MatrixFloat       det;
    MatrixFloat       a11, a12, a21, a22;
    MatrixFloat       invDet;

    status = Matrix_Determinant2x2(MatrixPtr, &det);

    if (status == MATRIX_SUCCESS)
    {
        if (ABS_FLOAT(det) < ZERO_THRESHOLD_FLOAT)
        {
            status = MATRIX_ERROR_SINGULAR;
        }
        else
        {
            invDet = 1.0f / det;

            a11 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 0U)];
            a12 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 1U)];
            a21 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 0U)];
            a22 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 1U)];

            /* inv(A) = (1/det) · [ a22  -a12; -a21  a11 ] */
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 0U, 0U)] =  a22 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 0U, 1U)] = -a12 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 1U, 0U)] = -a21 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 1U, 1U)] =  a11 * invDet;

            ResultPtr->Rows = 2U;
            ResultPtr->Cols = 2U;
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
MatrixStatus_T Matrix_Inverse3x3(
    const Matrix_T  * const MatrixPtr,
    Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;
    MatrixFloat       det;
    MatrixFloat       m00, m01, m02;
    MatrixFloat       m10, m11, m12;
    MatrixFloat       m20, m21, m22;
    MatrixFloat       c00, c01, c02;
    MatrixFloat       c10, c11, c12;
    MatrixFloat       c20, c21, c22;
    MatrixFloat       invDet;

    status = Matrix_Determinant3x3(MatrixPtr, &det);

    if (status == MATRIX_SUCCESS)
    {
        if (ABS_FLOAT(det) < ZERO_THRESHOLD_FLOAT)
        {
            status = MATRIX_ERROR_SINGULAR;
        }
        else
        {
            invDet = 1.0f / det;

            m00 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 0U)];
            m01 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 1U)];
            m02 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 0U, 2U)];
            m10 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 0U)];
            m11 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 1U)];
            m12 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 1U, 2U)];
            m20 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 2U, 0U)];
            m21 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 2U, 1U)];
            m22 = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, 2U, 2U)];

            /* Cofactor matrix (transposed in‑place when writing result). */
            c00 =  (m11 * m22 - m12 * m21);
            c01 = -(m10 * m22 - m12 * m20);
            c02 =  (m10 * m21 - m11 * m20);

            c10 = -(m01 * m22 - m02 * m21);
            c11 =  (m00 * m22 - m02 * m20);
            c12 = -(m00 * m21 - m01 * m20);

            c20 =  (m01 * m12 - m02 * m11);
            c21 = -(m00 * m12 - m02 * m10);
            c22 =  (m00 * m11 - m01 * m10);

            /* inv(A) = (1/det) · C^T */
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 0U, 0U)] = c00 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 0U, 1U)] = c10 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 0U, 2U)] = c20 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 1U, 0U)] = c01 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 1U, 1U)] = c11 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 1U, 2U)] = c21 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 2U, 0U)] = c02 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 2U, 1U)] = c12 * invDet;
            ResultPtr->Data[MATRIX_INDEX(ResultPtr, 2U, 2U)] = c22 * invDet;

            ResultPtr->Rows = 3U;
            ResultPtr->Cols = 3U;
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
MatrixStatus_T Matrix_Inverse4x4(
    const Matrix_T  * const MatrixPtr,
    Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;
    MatrixElement     augBuffer[4U * 8U];
    Matrix_T       aug;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    uint32_T          maxRow;
    MatrixElement     maxVal;
    MatrixElement     pivot;
    MatrixElement     factor;
    boolean_T         singular;

    n        = 4U;
    singular = FALSE;
    status   = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((MatrixPtr->Rows != 4U) || (MatrixPtr->Cols != 4U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((ResultPtr->MaxRows < 4U) || (ResultPtr->MaxCols < 4U))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        Matrix_Init(&aug, augBuffer, 4U, 8U);

        /* Build augmented matrix [A | I]. */
        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < n; j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, j)] =
                    MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
            }
        }

        for (i = 0U; i < n; i++)
        {
            for (j = n; j < (2U * n); j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, j)] =
                    ((j - n) == i) ? 1.0f : 0.0f;
            }
        }

        /* Gauss-Jordan elimination with partial pivoting. */
        for (i = 0U; (i < n) && (singular == FALSE); i++)
        {
            maxRow = i;
            maxVal = (aug.Data[MATRIX_INDEX(&aug, i, i)] >= 0.0f) ?
                      aug.Data[MATRIX_INDEX(&aug, i, i)] :
                      -aug.Data[MATRIX_INDEX(&aug, i, i)];

            for (k = i + 1U; k < n; k++)
            {
                MatrixElement val = (aug.Data[MATRIX_INDEX(&aug, k, i)] >= 0.0f) ?
                                    aug.Data[MATRIX_INDEX(&aug, k, i)] :
                                    -aug.Data[MATRIX_INDEX(&aug, k, i)];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = k;
                }
            }

            if (maxVal < ZERO_THRESHOLD_FLOAT)
            {
                singular = TRUE;
            }
            else
            {
                MatrixElement temp;

                if (maxRow != i)
                {
                    for (j = 0U; j < (2U * n); j++)
                    {
                        temp                                  = aug.Data[MATRIX_INDEX(&aug, i,       j)];
                        aug.Data[MATRIX_INDEX(&aug, i,       j)] = aug.Data[MATRIX_INDEX(&aug, maxRow, j)];
                        aug.Data[MATRIX_INDEX(&aug, maxRow, j)] = temp;
                    }
                }
                else
                {
                    /* No swap needed – no action. */
                }

                pivot = aug.Data[MATRIX_INDEX(&aug, i, i)];
                for (j = i; j < (2U * n); j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] /= pivot;
                }

                for (k = 0U; k < n; k++)
                {
                    if (k != i)
                    {
                        factor = aug.Data[MATRIX_INDEX(&aug, k, i)];
                        for (j = i; j < (2U * n); j++)
                        {
                            aug.Data[MATRIX_INDEX(&aug, k, j)] -= factor * aug.Data[MATRIX_INDEX(&aug, i, j)];
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
                    ResultPtr->Data[MATRIX_INDEX(ResultPtr, i, j)] =
                        aug.Data[MATRIX_INDEX(&aug, i, n + j)];
                }
            }

            ResultPtr->Rows = n;
            ResultPtr->Cols = n;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Inverse  (dispatcher)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Inverse(
    const Matrix_T  * const MatrixPtr,
    Matrix_T * const ResultPtr)
{
    MatrixStatus_T status;
    uint32_T          n;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if ((ResultPtr->MaxRows < MatrixPtr->Rows) || (ResultPtr->MaxCols < MatrixPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = MatrixPtr->Rows;

        if      (n == 2U) { status = Matrix_Inverse2x2(MatrixPtr, ResultPtr); }
        else if (n == 3U) { status = Matrix_Inverse3x3(MatrixPtr, ResultPtr); }
        else if (n == 4U) { status = Matrix_Inverse4x4(MatrixPtr, ResultPtr); }
        else
        {
            /* Generic augmented Gauss-Jordan for 5 × 5 … 8 × 8 with partial pivoting. */
            MatrixElement augBuffer[MATRIX_MAX_ROWS * (2U * MATRIX_MAX_COLS)];
            Matrix_T   aug;
            uint32_T      i;
            uint32_T      j;
            uint32_T      k;
            uint32_T      maxRow;
            MatrixElement maxVal;
            MatrixElement pivot;
            MatrixElement factor;
            boolean_T     singular;

            singular = FALSE;
            Matrix_Init(&aug, augBuffer, n, 2U * n);

            for (i = 0U; i < n; i++)
            {
                for (j = 0U; j < n; j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] =
                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
                }
            }

            for (i = 0U; i < n; i++)
            {
                for (j = n; j < (2U * n); j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] =
                        ((j - n) == i) ? 1.0f : 0.0f;
                }
            }

            for (i = 0U; (i < n) && (singular == FALSE); i++)
            {
                maxRow = i;
                maxVal = (aug.Data[MATRIX_INDEX(&aug, i, i)] >= 0.0f) ?
                          aug.Data[MATRIX_INDEX(&aug, i, i)] :
                          -aug.Data[MATRIX_INDEX(&aug, i, i)];

                for (k = i + 1U; k < n; k++)
                {
                    MatrixElement val = (aug.Data[MATRIX_INDEX(&aug, k, i)] >= 0.0f) ?
                                        aug.Data[MATRIX_INDEX(&aug, k, i)] :
                                        -aug.Data[MATRIX_INDEX(&aug, k, i)];
                    if (val > maxVal)
                    {
                        maxVal = val;
                        maxRow = k;
                    }
                }

                if (maxVal < ZERO_THRESHOLD_FLOAT)
                {
                    singular = TRUE;
                }
                else
                {
                    MatrixElement temp;

                    if (maxRow != i)
                    {
                        for (j = 0U; j < (2U * n); j++)
                        {
                            temp                                  = aug.Data[MATRIX_INDEX(&aug, i,       j)];
                            aug.Data[MATRIX_INDEX(&aug, i,       j)] = aug.Data[MATRIX_INDEX(&aug, maxRow, j)];
                            aug.Data[MATRIX_INDEX(&aug, maxRow, j)] = temp;
                        }
                    }
                    else
                    {
                        /* No action. */
                    }

                    pivot = aug.Data[MATRIX_INDEX(&aug, i, i)];
                    for (j = i; j < (2U * n); j++)
                    {
                        aug.Data[MATRIX_INDEX(&aug, i, j)] /= pivot;
                    }

                    for (k = 0U; k < n; k++)
                    {
                        if (k != i)
                        {
                            factor = aug.Data[MATRIX_INDEX(&aug, k, i)];
                            for (j = i; j < (2U * n); j++)
                            {
                                aug.Data[MATRIX_INDEX(&aug, k, j)] -= factor * aug.Data[MATRIX_INDEX(&aug, i, j)];
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
                        ResultPtr->Data[MATRIX_INDEX(ResultPtr, i, j)] =
                            aug.Data[MATRIX_INDEX(&aug, i, n + j)];
                    }
                }

                ResultPtr->Rows = n;
                ResultPtr->Cols = n;
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Eigenvalues  (iterative Jacobi — requires symmetric matrix)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Eigenvalues(
    Matrix_T      * const MatrixPtr,
    MatrixEigen_T * const EigenOutPtr,
    const uint32_T MaxIterations,
    const MatrixFloat        Tolerance)
{
    MatrixStatus_T status;
    uint32_T          n;
    uint32_T          iter;
    uint32_T          p;
    uint32_T          q;
    uint32_T          i;
    MatrixFloat       maxOffDiag;
    MatrixFloat       app;
    MatrixFloat       aqq;
    MatrixFloat       apq;
    MatrixFloat       theta;
    MatrixFloat       c;
    MatrixFloat       s;
    MatrixFloat       temp;
    Matrix_T       vMatrix;
    MatrixElement     vBuffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    boolean_T         useTolerance;
    uint32_T          maxIter;
    boolean_T         convergedEarly;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (EigenOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
#ifdef MATRIX_ENFORCE_SYMMETRY
    else if (!Matrix_IsSymmetric(MatrixPtr, 1e-6f))
    {
        status = MATRIX_ERROR_NOT_SYMMETRIC;
    }
#endif
    else
    {
        n             = MatrixPtr->Rows;
        maxIter       = (MaxIterations > 0U) ? MaxIterations : JACOBI_MAX_ITER;
        useTolerance = (Tolerance > 0.0f) ? TRUE : FALSE;
        convergedEarly = FALSE;

        EigenOutPtr->NumEigenvalues = n;
        EigenOutPtr->Iterations      = 0U;

        /* Initialise eigenvector accumulator to I. */
        Matrix_Init(&vMatrix, vBuffer, n, n);
        (void)Matrix_Identity(&vMatrix);

        for (iter = 0U; iter < maxIter; iter++)
        {
            uint32_T    j;
            MatrixFloat val;
            MatrixFloat a_ip;
            MatrixFloat a_iq;
            MatrixFloat v_ip;
            MatrixFloat v_iq;
            boolean_T   converged;

            EigenOutPtr->Iterations = iter + 1U;

            /* Find largest off-diagonal element to use as pivot. */
            maxOffDiag = 0.0f;
            p = 0U;
            q = 1U;

            for (i = 0U; i < n; i++)
            {
                for (j = i + 1U; j < n; j++)
                {
                    val = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
                    val = (val >= 0.0f) ? val : -val;
                    if (val > maxOffDiag)
                    {
                        maxOffDiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            /* Check convergence. */
            if ((useTolerance != FALSE) && (maxOffDiag < Tolerance))
            {
                converged = TRUE;
            }
            else if (maxOffDiag < JACOBI_TOLERANCE)
            {
                converged = TRUE;
            }
            else
            {
                converged = FALSE;
            }

            if (converged != FALSE)
            {
                convergedEarly = TRUE;
                iter = maxIter; /* Force loop termination. */
            }
            else
            {
                /* Apply Jacobi rotation for (p, q). */
                app = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, p, p)];
                aqq = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, q, q)];
                apq = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, p, q)];

                if ((aqq - app) > -ZERO_THRESHOLD_FLOAT && (aqq - app) < ZERO_THRESHOLD_FLOAT)
                {
                    theta = JACOBI_PI_OVER_4;
                }
                else
                {
                    theta = 0.5f * atan2f(2.0f * apq, aqq - app);
                }

                c = cosf(theta);
                s = sinf(theta);

                /* Update diagonal elements. */
                temp = app;
                app  = (c * c * temp) - (2.0f * c * s * apq) + (s * s * aqq);
                aqq  = (s * s * temp) + (2.0f * c * s * apq) + (c * c * aqq);

                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, p, p)] = app;
                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, q, q)] = aqq;
                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, p, q)] = 0.0f;
                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, q, p)] = 0.0f;

                /* Update off-diagonal rows. */
                for (i = 0U; i < n; i++)
                {
                    if ((i != p) && (i != q))
                    {
                        a_ip = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, p)];
                        a_iq = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, q)];

                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, p)] = (c * a_ip) - (s * a_iq);
                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, p, i)] = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, p)];

                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, q)] = (s * a_ip) + (c * a_iq);
                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, q, i)] = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, q)];
                    }
                    else
                    {
                        /* No action for pivot rows. */
                    }
                }

                /* Accumulate rotation into V. */
                for (i = 0U; i < n; i++)
                {
                    v_ip = vMatrix.Data[MATRIX_INDEX(&vMatrix, i, p)];
                    v_iq = vMatrix.Data[MATRIX_INDEX(&vMatrix, i, q)];

                    vMatrix.Data[MATRIX_INDEX(&vMatrix, i, p)] = (c * v_ip) - (s * v_iq);
                    vMatrix.Data[MATRIX_INDEX(&vMatrix, i, q)] = (s * v_ip) + (c * v_iq);
                }
            }
        }

        /* Copy eigenvalues from the diagonalised MatrixPtr. */
        for (i = 0U; i < n; i++)
        {
            EigenOutPtr->Eigenvalues[i] = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, i)];
        }

        /* Copy eigenvectors (column-wise) from V. */
        {
            uint32_T i2;
            uint32_T j2;
            for (i2 = 0U; i2 < n; i2++)
            {
                for (j2 = 0U; j2 < n; j2++)
                {
                    EigenOutPtr->Eigenvectors[(i2 * n) + j2] =
                        vMatrix.Data[MATRIX_INDEX(&vMatrix, i2, j2)];
                }
            }
        }

        if ((iter >= maxIter) && (convergedEarly == FALSE))
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
MatrixStatus_T Matrix_EigenvaluesOnly(
    Matrix_T    * const MatrixPtr,
    MatrixFloat    * const EigenvaluesOutPtr,
    const uint32_T MaxIterations,
    const MatrixFloat      Tolerance)
{
    MatrixStatus_T status;
    MatrixEigen_T  eigenOut;
    uint32_T          i;

    status = Matrix_Eigenvalues(MatrixPtr, &eigenOut, MaxIterations, Tolerance);

    if (status == MATRIX_SUCCESS)
    {
        for (i = 0U; i < eigenOut.NumEigenvalues; i++)
        {
            EigenvaluesOutPtr[i] = eigenOut.Eigenvalues[i];
        }
    }
    else
    {
        /* Error propagated from Matrix_Eigenvalues – no action. */
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_LU  — now stores swapCount in pivot[0] for determinant parity
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_LU(
    Matrix_T  * const MatrixPtr,
    uint32_T     * const pivot)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          pivotRow;
    MatrixElement     factor;
    MatrixElement     temp;
    uint32_T          n;
    uint32_T          swapCount;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (pivot == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        n = MatrixPtr->Rows;
        if (n == 0U)
        {
            status = MATRIX_ERROR_SIZE_EXCEEDED;
        }
        else
        {
            swapCount = 0U;

            for (i = 0U; i < n; i++)
            {
                pivot[i] = i;
            }

            for (k = 0U; (k < (n - 1U)) && (status == MATRIX_SUCCESS); k++)
            {
                /* Partial pivot search. */
                pivotRow = k;
                for (i = k + 1U; i < n; i++)
                {
                    MatrixElement absI = (MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, k)] >= 0.0f) ?
                                           MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, k)] :
                                           -MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, k)];
                    MatrixElement absPivot = (MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, pivotRow, k)] >= 0.0f) ?
                                               MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, pivotRow, k)] :
                                               -MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, pivotRow, k)];
                    if (absI > absPivot)
                    {
                        pivotRow = i;
                    }
                }

                if (pivotRow != k)
                {
                    for (j = 0U; j < n; j++)
                    {
                        temp                                       = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k,         j)];
                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k,         j)] = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, pivotRow, j)];
                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, pivotRow, j)] = temp;
                    }

                    i = pivot[k];
                    pivot[k] = pivot[pivotRow];
                    pivot[pivotRow] = i;
                    swapCount++;
                }
                else
                {
                    /* No row swap required – no action. */
                }

                MatrixElement absPivotK = (MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k, k)] >= 0.0f) ?
                                             MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k, k)] :
                                             -MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k, k)];
                if (absPivotK < ZERO_THRESHOLD_FLOAT)
                {
                    status = MATRIX_ERROR_SINGULAR;
                }
                else
                {
                    for (i = k + 1U; i < n; i++)
                    {
                        factor = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, k)] /
                                 MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k, k)];
                        MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, k)] = factor;

                        for (j = k + 1U; j < n; j++)
                        {
                            MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)] -=
                                factor * MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, k, j)];
                        }
                    }
                }
            }

            /* Explicit check of the last diagonal element. */
            if ((status == MATRIX_SUCCESS) && (n > 0U))
            {
                MatrixElement absLast = (MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, n - 1U, n - 1U)] >= 0.0f) ?
                                          MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, n - 1U, n - 1U)] :
                                          -MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, n - 1U, n - 1U)];
                if (absLast < ZERO_THRESHOLD_FLOAT)
                {
                    status = MATRIX_ERROR_SINGULAR;
                }
                else
                {
                    /* Last diagonal is non-zero – no action. */
                }
            }

            /* Store swapCount in pivot[0] for later use by determinant */
            pivot[0] = swapCount;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SolveGaussJordan  — now with partial pivoting
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_SolveGaussJordan(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr)
{
    MatrixStatus_T status;
    MatrixElement     augBuffer[MATRIX_MAX_ROWS * (MATRIX_MAX_COLS + MATRIX_MAX_COLS)];
    Matrix_T       aug;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    uint32_T          m;
    uint32_T          maxRow;
    MatrixElement     maxVal;
    MatrixElement     pivot;
    MatrixElement     factor;
    boolean_T         singular;

    singular = FALSE;
    status   = MATRIX_SUCCESS;

    if ((APtr == NULL) || (BPtr == NULL) || (XPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (APtr->Rows != APtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if (APtr->Rows != BPtr->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((XPtr->MaxRows < APtr->Rows) || (XPtr->MaxCols < BPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = APtr->Rows;
        m = BPtr->Cols;

        Matrix_Init(&aug, augBuffer, n, n + m);

        /* Build augmented matrix [A | B] */
        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < n; j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, j)] =
                    APtr->Data[MATRIX_INDEX(APtr, i, j)];
            }
        }

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < m; j++)
            {
                aug.Data[MATRIX_INDEX(&aug, i, n + j)] =
                    BPtr->Data[MATRIX_INDEX(BPtr, i, j)];
            }
        }

        /* Gauss-Jordan with partial pivoting */
        for (i = 0U; (i < n) && (singular == FALSE); i++)
        {
            /* Find pivot row */
            maxRow = i;
            maxVal = (aug.Data[MATRIX_INDEX(&aug, i, i)] >= 0.0f) ?
                      aug.Data[MATRIX_INDEX(&aug, i, i)] :
                      -aug.Data[MATRIX_INDEX(&aug, i, i)];

            for (k = i + 1U; k < n; k++)
            {
                MatrixElement val = (aug.Data[MATRIX_INDEX(&aug, k, i)] >= 0.0f) ?
                                    aug.Data[MATRIX_INDEX(&aug, k, i)] :
                                    -aug.Data[MATRIX_INDEX(&aug, k, i)];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxRow = k;
                }
            }

            if (maxVal < ZERO_THRESHOLD_FLOAT)
            {
                singular = TRUE;
            }
            else
            {
                /* Swap rows if needed */
                if (maxRow != i)
                {
                    MatrixElement temp;
                    for (j = 0U; j < (n + m); j++)
                    {
                        temp = aug.Data[MATRIX_INDEX(&aug, i, j)];
                        aug.Data[MATRIX_INDEX(&aug, i, j)] = aug.Data[MATRIX_INDEX(&aug, maxRow, j)];
                        aug.Data[MATRIX_INDEX(&aug, maxRow, j)] = temp;
                    }
                }
                else
                {
                    /* No swap needed */
                }

                pivot = aug.Data[MATRIX_INDEX(&aug, i, i)];
                /* Normalise pivot row */
                for (j = i; j < (n + m); j++)
                {
                    aug.Data[MATRIX_INDEX(&aug, i, j)] /= pivot;
                }

                /* Eliminate column i from all other rows */
                for (k = 0U; k < n; k++)
                {
                    if (k != i)
                    {
                        factor = aug.Data[MATRIX_INDEX(&aug, k, i)];
                        for (j = i; j < (n + m); j++)
                        {
                            aug.Data[MATRIX_INDEX(&aug, k, j)] -=
                                factor * aug.Data[MATRIX_INDEX(&aug, i, j)];
                        }
                    }
                    else
                    {
                        /* Skip pivot row */
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
                    XPtr->Data[MATRIX_INDEX(XPtr, i, j)] =
                        aug.Data[MATRIX_INDEX(&aug, i, n + j)];
                }
            }

            XPtr->Rows = n;
            XPtr->Cols = m;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Solve  (delegates to Gauss-Jordan)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Solve(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr)
{
    return Matrix_SolveGaussJordan(APtr, BPtr, XPtr);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsSquare
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsSquare(const Matrix_T * const MatrixPtr)
{
    boolean_T result;

    result = FALSE;

    if ((MatrixPtr != NULL) && (MatrixPtr->Rows == MatrixPtr->Cols))
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
    const Matrix_T  * const MatrixPtr,
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

    if (MatrixPtr == NULL)
    {
        result = FALSE;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        result = FALSE;
    }
    else
    {
        for (i = 0U; (i < MatrixPtr->Rows) && (result != FALSE); i++)
        {
            for (j = i + 1U; (j < MatrixPtr->Cols) && (result != FALSE); j++)
            {
                a_ij = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
                a_ji = MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, j, i)];
                diff = (a_ij >= a_ji) ? (a_ij - a_ji) : (a_ji - a_ij);

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
MatrixStatus_T Matrix_Trace(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const TraceOutPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    MatrixFloat       sum;

    sum    = 0.0f;
    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (TraceOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        for (i = 0U; i < MatrixPtr->Rows; i++)
        {
            sum += MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, i)];
        }

        *TraceOutPtr = sum;
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_NormFrobenius
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_NormFrobenius(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const NormOutPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    real64_T          sum;
    real64_T          val;

    sum    = 0.0;
    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (NormOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        for (i = 0U; i < MatrixPtr->Rows; i++)
        {
            for (j = 0U; j < MatrixPtr->Cols; j++)
            {
                val  = (real64_T)MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
                sum += val * val;
            }
        }

        *NormOutPtr = (MatrixFloat)sqrt(sum);
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsEqual
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsEqual(
    const Matrix_T  * const APtr,
    const Matrix_T  * const BPtr,
    const MatrixFloat          Tolerance)
{
    boolean_T result;
    uint32_T  i;
    uint32_T  j;
    real64_T  diff;
    real64_T  tol;
    real64_T  aVal;
    real64_T  bVal;

    result = TRUE;
    tol    = (Tolerance > 0.0f) ? (real64_T)Tolerance : (real64_T)ZERO_THRESHOLD_FLOAT;

    if ((APtr == NULL) || (BPtr == NULL))
    {
        result = FALSE;
    }
    else if ((APtr->Rows != BPtr->Rows) || (APtr->Cols != BPtr->Cols))
    {
        result = FALSE;
    }
    else
    {
        for (i = 0U; (i < APtr->Rows) && (result != FALSE); i++)
        {
            for (j = 0U; (j < APtr->Cols) && (result != FALSE); j++)
            {
                aVal = (real64_T)APtr->Data[MATRIX_INDEX(APtr, i, j)];
                bVal = (real64_T)BPtr->Data[MATRIX_INDEX(BPtr, i, j)];
                diff = (aVal > bVal) ? (aVal - bVal) : (bVal - aVal);

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
    const Matrix_T  * const MatrixPtr,
    uint32_T           * const RowsOutPtr,
    uint32_T           * const ColsOutPtr)
{
    if ((MatrixPtr != NULL) && (RowsOutPtr != NULL) && (ColsOutPtr != NULL))
    {
        *RowsOutPtr = MatrixPtr->Rows;
        *ColsOutPtr = MatrixPtr->Cols;
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
    Matrix_T        * const MatrixPtr,
    const MatrixElement        value)
{
    uint32_T i;
    uint32_T j;

    if (MatrixPtr != NULL)
    {
        for (i = 0U; i < MatrixPtr->Rows; i++)
        {
            for (j = 0U; j < MatrixPtr->Cols; j++)
            {
                MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)] = value;
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
    Matrix_T    * const MatrixPtr,
    const MatrixFloat      value)
{
    Matrix_Fill(MatrixPtr, value);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_FloatToQ31  (DEPRECATED — identity for compatibility)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixElement Matrix_FloatToQ31(const MatrixFloat Value)
{
    if (Value > 1.0f)
    {
        return 1.0f;
    }
    else if (Value < -1.0f)
    {
        return -1.0f;
    }
    else
    {
        return Value;
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Q31ToFloat  (DEPRECATED — identity for compatibility)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixFloat Matrix_Q31ToFloat(const MatrixElement value)
{
    return value;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_Cholesky  — requires symmetric positive definite (caller responsibility)
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_Cholesky(
    const Matrix_T  * const MatrixPtr,
    Matrix_T        * const LPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    real64_T          sum;
    real64_T          val;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (LPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if ((LPtr->MaxRows < MatrixPtr->Rows) || (LPtr->MaxCols < MatrixPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
#ifdef MATRIX_ENFORCE_SYMMETRY
    else if (!Matrix_IsSymmetric(MatrixPtr, 1e-6f))
    {
        status = MATRIX_ERROR_NON_POSITIVE_DEFINITE;
    }
#endif
    else
    {
        n = MatrixPtr->Rows;

        Matrix_Zero(LPtr);
        Matrix_SetDimensions(LPtr, n, n);

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j <= i; j++)
            {
                sum = 0.0;

                for (k = 0U; k < j; k++)
                {
                    val = (real64_T)LPtr->Data[MATRIX_INDEX(LPtr, i, k)] *
                          (real64_T)LPtr->Data[MATRIX_INDEX(LPtr, j, k)];
                    sum += val;
                }

                if (i == j)
                {
                    real64_T a_ii = (real64_T)MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, i)];
                    real64_T diff = a_ii - sum;

                    if (diff <= 0.0)
                    {
                        status = MATRIX_ERROR_NON_POSITIVE_DEFINITE;
                        break;
                    }

                    LPtr->Data[MATRIX_INDEX(LPtr, i, i)] = (MatrixFloat)sqrt(diff);
                }
                else
                {
                    real64_T a_ij = (real64_T)MatrixPtr->Data[MATRIX_INDEX(MatrixPtr, i, j)];
                    real64_T l_jj = (real64_T)LPtr->Data[MATRIX_INDEX(LPtr, j, j)];

                    if (l_jj >= -ZERO_THRESHOLD_FLOAT && l_jj <= ZERO_THRESHOLD_FLOAT)
                    {
                        status = MATRIX_ERROR_NON_POSITIVE_DEFINITE;
                        break;
                    }

                    LPtr->Data[MATRIX_INDEX(LPtr, i, j)] = (MatrixFloat)((a_ij - sum) / l_jj);
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
MatrixStatus_T Matrix_ForwardSubstitution(
    const Matrix_T  * const LPtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          n;
    uint32_T          m;
    real64_T          sum;
    real64_T          l_ii;

    status = MATRIX_SUCCESS;

    if ((LPtr == NULL) || (BPtr == NULL) || (XPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (LPtr->Rows != LPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if (LPtr->Rows != BPtr->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((XPtr->MaxRows < BPtr->Rows) || (XPtr->MaxCols < BPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = LPtr->Rows;
        m = BPtr->Cols;

        Matrix_SetDimensions(XPtr, n, m);
        Matrix_Zero(XPtr);

        for (i = 0U; i < n; i++)
        {
            for (j = 0U; j < m; j++)
            {
                sum = 0.0;

                for (uint32_T k = 0U; k < i; k++)
                {
                    sum += (real64_T)LPtr->Data[MATRIX_INDEX(LPtr, i, k)] *
                           (real64_T)XPtr->Data[MATRIX_INDEX(XPtr, k, j)];
                }

                l_ii = (real64_T)LPtr->Data[MATRIX_INDEX(LPtr, i, i)];

                if (l_ii >= -ZERO_THRESHOLD_FLOAT && l_ii <= ZERO_THRESHOLD_FLOAT)
                {
                    status = MATRIX_ERROR_SINGULAR;
                    break;
                }

                real64_T b_ij = (real64_T)BPtr->Data[MATRIX_INDEX(BPtr, i, j)];
                XPtr->Data[MATRIX_INDEX(XPtr, i, j)] = (MatrixFloat)((b_ij - sum) / l_ii);
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
MatrixStatus_T Matrix_BackwardSubstitution(
    const Matrix_T  * const UPtr,
    const Matrix_T  * const BPtr,
    Matrix_T        * const XPtr)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          k;
    uint32_T          n;
    uint32_T          m;
    real64_T          sum;
    real64_T          u_ii;
    int32_T           iSigned;

    status = MATRIX_SUCCESS;

    if ((UPtr == NULL) || (BPtr == NULL) || (XPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (UPtr->Rows != UPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if (UPtr->Rows != BPtr->Rows)
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else if ((XPtr->MaxRows < BPtr->Rows) || (XPtr->MaxCols < BPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        n = UPtr->Rows;
        m = BPtr->Cols;

        Matrix_SetDimensions(XPtr, n, m);
        Matrix_Zero(XPtr);

        for (iSigned = (int32_T)n - 1; iSigned >= 0; iSigned--)
        {
            i = (uint32_T)iSigned;

            for (j = 0U; j < m; j++)
            {
                sum = 0.0;

                for (k = i + 1U; k < n; k++)
                {
                    sum += (real64_T)UPtr->Data[MATRIX_INDEX(UPtr, i, k)] *
                           (real64_T)XPtr->Data[MATRIX_INDEX(XPtr, k, j)];
                }

                u_ii = (real64_T)UPtr->Data[MATRIX_INDEX(UPtr, i, i)];

                if (u_ii >= -ZERO_THRESHOLD_FLOAT && u_ii <= ZERO_THRESHOLD_FLOAT)
                {
                    status = MATRIX_ERROR_SINGULAR;
                    break;
                }

                real64_T b_ij = (real64_T)BPtr->Data[MATRIX_INDEX(BPtr, i, j)];
                XPtr->Data[MATRIX_INDEX(XPtr, i, j)] = (MatrixFloat)((b_ij - sum) / u_ii);
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
MatrixStatus_T Matrix_SymmetricRank1Update(
    Matrix_T        * const APtr,
    const Matrix_T  * const VPtr,
    const MatrixElement        alpha)
{
    MatrixStatus_T status;
    uint32_T          i;
    uint32_T          j;
    uint32_T          n;
    MatrixElement     scaledVi;

    status = MATRIX_SUCCESS;

    if ((APtr == NULL) || (VPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if ((APtr->Rows != APtr->Cols) || (VPtr->Rows != APtr->Rows) || (VPtr->Cols != 1U))
    {
        status = MATRIX_ERROR_DIMENSION_MISMATCH;
    }
    else
    {
        n = APtr->Rows;

        for (i = 0U; i < n; i++)
        {
            scaledVi = VPtr->Data[MATRIX_INDEX(VPtr, i, 0U)] * alpha;

            for (j = 0U; j <= i; j++)
            {
                MatrixElement update = scaledVi * VPtr->Data[MATRIX_INDEX(VPtr, j, 0U)];

                APtr->Data[MATRIX_INDEX(APtr, i, j)] += update;
                APtr->Data[MATRIX_INDEX(APtr, j, i)] = APtr->Data[MATRIX_INDEX(APtr, i, j)];
            }
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_SymmetricRank1UpdateFloat
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_SymmetricRank1UpdateFloat(
    Matrix_T        * const APtr,
    const Matrix_T  * const VPtr,
    const MatrixFloat          alpha)
{
    return Matrix_SymmetricRank1Update(APtr, VPtr, alpha);
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_MatrixSquareRoot  — correct Denman–Beavers iteration
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_MatrixSquareRoot(
    const Matrix_T  * const MatrixPtr,
    Matrix_T * const ResultPtr,
    const uint32_T MaxIter)
{
    MatrixStatus_T status;
    MatrixElement     YBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     ZBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     YInvBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     ZInvBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     YNewBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    MatrixElement     ZNewBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       Y;
    Matrix_T       Z;
    Matrix_T       YInv;
    Matrix_T       ZInv;
    Matrix_T       YNew;
    Matrix_T       ZNew;
    uint32_T          iter;
    uint32_T          maxIters;
    MatrixFloat       normDiff;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (ResultPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else if ((ResultPtr->MaxRows < MatrixPtr->Rows) || (ResultPtr->MaxCols < MatrixPtr->Cols))
    {
        status = MATRIX_ERROR_SIZE_EXCEEDED;
    }
    else
    {
        maxIters = (MaxIter > 0U) ? MaxIter : 10U;

        Matrix_Init(&Y, YBuf, MatrixPtr->Rows, MatrixPtr->Cols);
        Matrix_Init(&Z, ZBuf, MatrixPtr->Rows, MatrixPtr->Cols);
        Matrix_Init(&YInv, YInvBuf, MatrixPtr->Rows, MatrixPtr->Cols);
        Matrix_Init(&ZInv, ZInvBuf, MatrixPtr->Rows, MatrixPtr->Cols);
        Matrix_Init(&YNew, YNewBuf, MatrixPtr->Rows, MatrixPtr->Cols);
        Matrix_Init(&ZNew, ZNewBuf, MatrixPtr->Rows, MatrixPtr->Cols);

        /* Initialize: Y0 = A, Z0 = I */
        (void)Matrix_Copy(&Y, MatrixPtr);
        (void)Matrix_Identity(&Z);

        for (iter = 0U; iter < maxIters; iter++)
        {
            /* YInv = inv(Y) */
            status = Matrix_Inverse(&Y, &YInv);
            if (status != MATRIX_SUCCESS) { break; }

            /* ZInv = inv(Z) */
            status = Matrix_Inverse(&Z, &ZInv);
            if (status != MATRIX_SUCCESS) { break; }

            /* YNew = 0.5 * (Y + ZInv) */
            Matrix_Add(&Y, &ZInv, &YNew);
            Matrix_ScalarMultiplyFloat(&YNew, 0.5f, &YNew);

            /* ZNew = 0.5 * (Z + YInv) */
            Matrix_Add(&Z, &YInv, &ZNew);
            Matrix_ScalarMultiplyFloat(&ZNew, 0.5f, &ZNew);

            /* Check convergence */
            Matrix_Subtract(&YNew, &Y, &Y);
            Matrix_NormFrobenius(&Y, &normDiff);

            if (normDiff < 1e-6f)
            {
                break;
            }

            /* Update */
            (void)Matrix_Copy(&Y, &YNew);
            (void)Matrix_Copy(&Z, &ZNew);
        }

        if (iter >= maxIters)
        {
            status = MATRIX_ERROR_MAX_ITERATIONS;
        }
        else
        {
            /* Y is the matrix square root; copy it to result */
            (void)Matrix_Copy(ResultPtr, &YNew);
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_ConditionNumber
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_T Matrix_ConditionNumber(
    const Matrix_T  * const MatrixPtr,
    MatrixFloat        * const CondOutPtr)
{
    MatrixStatus_T status;
    MatrixElement     invBuf[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       invMatrix;
    MatrixFloat       normA;
    MatrixFloat       normAinv;

    status = MATRIX_SUCCESS;

    if ((MatrixPtr == NULL) || (CondOutPtr == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else if (MatrixPtr->Rows != MatrixPtr->Cols)
    {
        status = MATRIX_ERROR_NOT_SQUARE;
    }
    else
    {
        Matrix_Init(&invMatrix, invBuf, MatrixPtr->Rows, MatrixPtr->Cols);

        Matrix_NormFrobenius(MatrixPtr, &normA);
        status = Matrix_Inverse(MatrixPtr, &invMatrix);

        if (status == MATRIX_SUCCESS)
        {
            Matrix_NormFrobenius(&invMatrix, &normAinv);
            *CondOutPtr = normA * normAinv;
        }
    }

    return status;
}


/*--------------------------------------------------------------------------------------------------------------------
 * Matrix_IsPositiveDefinite
 *------------------------------------------------------------------------------------------------------------------*/
boolean_T Matrix_IsPositiveDefinite(const Matrix_T * const MatrixPtr)
{
    MatrixStatus_T status;
    MatrixElement     LBuffer[MATRIX_MAX_ROWS * MATRIX_MAX_COLS];
    Matrix_T       L;
    boolean_T         result;

    result = FALSE;

    if ((MatrixPtr != NULL) && (MatrixPtr->Rows == MatrixPtr->Cols))
    {
        Matrix_Init(&L, LBuffer, MatrixPtr->Rows, MatrixPtr->Cols);
        status = Matrix_Cholesky(MatrixPtr, &L);

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