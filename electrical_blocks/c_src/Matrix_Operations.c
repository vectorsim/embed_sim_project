/*
 * Matrix_Operations.c
 * ===================
 *
 * Implementation of pure matrix operations for PMSM control.
 * MISRA C:2012 compliant - No memcpy, explicit element-wise operations.
 * ASIL-D ready.
 *
 * @author EmbedSim Framework
 * @version 2.0.0
 * @date 2024
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Matrix_Operations.h"
#include <math.h>           /* For fabsf in determinant calculations */
/* Platform_Types.h not present in this repo; no boolean types used in this TU */

/* MISRA C:2012 Rule 20.1 - No #include in function-like macro */
/* MISRA C:2012 Rule 20.9 - No undef or redefine reserved symbols */

/******************************************************************************/
/*-----------------------------------Macros-----------------------------------*/
/******************************************************************************/

/* MISRA C:2012 Rule 5.4 - No reuse of identifiers */
#define MATRIX_EPSILON (1.0e-6f)

/******************************************************************************/
/*-------------------------TASKING Pragmas (if needed)------------------------*/
/******************************************************************************/

#if defined(CT_USE_FAST_MEMORY) && defined(__TASKING__)
#  define MAT_FAST_START   _Pragma("section code \"cpu0_pspr\"")
#  define MAT_FAST_END     _Pragma("section code restore")
#else
#  define MAT_FAST_START
#  define MAT_FAST_END
#endif

/******************************************************************************/
/*-------------------------Initialization Functions---------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Zero_2x2(Matrix2x2_T* const pC)
{
    /* MISRA C:2012 Rule 14.3 - Controlling expressions not invariant */
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            /* MISRA C:2012 Rule 10.3 - Assignment to narrower type */
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_3x3(Matrix3x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_4x4(Matrix4x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_3x2(Matrix3x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_2x3(Matrix2x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_4x3(Matrix4x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_3x4(Matrix3x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = 0.0f;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_Vector2(Vector2_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 2U; i++)
    {
        pC->V[i] = 0.0f;
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_Vector3(Vector3_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 3U; i++)
    {
        pC->V[i] = 0.0f;
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Zero_Vector4(Vector4_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 4U; i++)
    {
        pC->V[i] = 0.0f;
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Identity Functions---------------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Identity_2x2(Matrix2x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            if (i == j)
            {
                pC->M[i][j] = 1.0f;
            }
            else
            {
                pC->M[i][j] = 0.0f;
            }
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Identity_3x3(Matrix3x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            if (i == j)
            {
                pC->M[i][j] = 1.0f;
            }
            else
            {
                pC->M[i][j] = 0.0f;
            }
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Identity_4x4(Matrix4x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            if (i == j)
            {
                pC->M[i][j] = 1.0f;
            }
            else
            {
                pC->M[i][j] = 0.0f;
            }
        }
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Copy Functions (No memcpy)-------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Copy_2x2(const Matrix2x2_T* const pA, Matrix2x2_T* const pC)
{
    uint8_T i, j;

    /* MISRA C:2012 Rule 11.3 - No pointer conversion */
    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_3x3(const Matrix3x3_T* const pA, Matrix3x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_4x4(const Matrix4x4_T* const pA, Matrix4x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_3x2(const Matrix3x2_T* const pA, Matrix3x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_2x3(const Matrix2x3_T* const pA, Matrix2x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_4x3(const Matrix4x3_T* const pA, Matrix4x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_3x4(const Matrix3x4_T* const pA, Matrix3x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_Vector2(const Vector2_T* const pA, Vector2_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 2U; i++)
    {
        pC->V[i] = pA->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_Vector3(const Vector3_T* const pA, Vector3_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 3U; i++)
    {
        pC->V[i] = pA->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Copy_Vector4(const Vector4_T* const pA, Vector4_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 4U; i++)
    {
        pC->V[i] = pA->V[i];
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Matrix Addition------------------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Add_2x2(const Matrix2x2_T* const pA,
                    const Matrix2x2_T* const pB,
                    Matrix2x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = pA->M[i][j] + pB->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Add_3x3(const Matrix3x3_T* const pA,
                    const Matrix3x3_T* const pB,
                    Matrix3x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[i][j] + pB->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Add_4x4(const Matrix4x4_T* const pA,
                    const Matrix4x4_T* const pB,
                    Matrix4x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = pA->M[i][j] + pB->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Add_Vector2(const Vector2_T* const pA,
                        const Vector2_T* const pB,
                        Vector2_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 2U; i++)
    {
        pC->V[i] = pA->V[i] + pB->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Add_Vector3(const Vector3_T* const pA,
                        const Vector3_T* const pB,
                        Vector3_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 3U; i++)
    {
        pC->V[i] = pA->V[i] + pB->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Add_Vector4(const Vector4_T* const pA,
                        const Vector4_T* const pB,
                        Vector4_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 4U; i++)
    {
        pC->V[i] = pA->V[i] + pB->V[i];
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Matrix Subtraction---------------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Sub_2x2(const Matrix2x2_T* const pA,
                    const Matrix2x2_T* const pB,
                    Matrix2x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = pA->M[i][j] - pB->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Sub_3x3(const Matrix3x3_T* const pA,
                    const Matrix3x3_T* const pB,
                    Matrix3x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[i][j] - pB->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Sub_4x4(const Matrix4x4_T* const pA,
                    const Matrix4x4_T* const pB,
                    Matrix4x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = pA->M[i][j] - pB->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Sub_Vector2(const Vector2_T* const pA,
                        const Vector2_T* const pB,
                        Vector2_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 2U; i++)
    {
        pC->V[i] = pA->V[i] - pB->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Sub_Vector3(const Vector3_T* const pA,
                        const Vector3_T* const pB,
                        Vector3_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 3U; i++)
    {
        pC->V[i] = pA->V[i] - pB->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Sub_Vector4(const Vector4_T* const pA,
                        const Vector4_T* const pB,
                        Vector4_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 4U; i++)
    {
        pC->V[i] = pA->V[i] - pB->V[i];
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Matrix Scaling-------------------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Scale_2x2(const Matrix2x2_T* const pA,
                      const real32_T Delta,
                      Matrix2x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_3x3(const Matrix3x3_T* const pA,
                      const real32_T Delta,
                      Matrix3x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_4x4(const Matrix4x4_T* const pA,
                      const real32_T Delta,
                      Matrix4x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_3x2(const Matrix3x2_T* const pA,
                      const real32_T Delta,
                      Matrix3x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_2x3(const Matrix2x3_T* const pA,
                      const real32_T Delta,
                      Matrix2x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_4x3(const Matrix4x3_T* const pA,
                      const real32_T Delta,
                      Matrix4x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_3x4(const Matrix3x4_T* const pA,
                      const real32_T Delta,
                      Matrix3x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = Delta * pA->M[i][j];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_Vector2(const Vector2_T* const pA,
                          const real32_T Delta,
                          Vector2_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 2U; i++)
    {
        pC->V[i] = Delta * pA->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_Vector3(const Vector3_T* const pA,
                          const real32_T Delta,
                          Vector3_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 3U; i++)
    {
        pC->V[i] = Delta * pA->V[i];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Scale_Vector4(const Vector4_T* const pA,
                          const real32_T Delta,
                          Vector4_T* const pC)
{
    uint8_T i;

    for (i = 0U; i < 4U; i++)
    {
        pC->V[i] = Delta * pA->V[i];
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Matrix Transposition-------------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Transpose_2x2(const Matrix2x2_T* const pA, Matrix2x2_T* const pC)
{
    /* MISRA C:2012 Rule 13.4 - No assignment in condition */
    real32_T temp;

    /* Handle in-place transpose */
    if (pA == pC)
    {
        temp = pC->M[0][1];
        pC->M[0][1] = pC->M[1][0];
        pC->M[1][0] = temp;
    }
    else
    {
        pC->M[0][0] = pA->M[0][0];
        pC->M[0][1] = pA->M[1][0];
        pC->M[1][0] = pA->M[0][1];
        pC->M[1][1] = pA->M[1][1];
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Transpose_3x3(const Matrix3x3_T* const pA, Matrix3x3_T* const pC)
{
    uint8_T i, j;

    if (pA == pC)
    {
        Matrix3x3_T Temp;

        for (i = 0U; i < 3U; i++)
        {
            for (j = 0U; j < 3U; j++)
            {
                Temp.M[i][j] = pA->M[j][i];
            }
        }

        for (i = 0U; i < 3U; i++)
        {
            for (j = 0U; j < 3U; j++)
            {
                pC->M[i][j] = Temp.M[i][j];
            }
        }
    }
    else
    {
        for (i = 0U; i < 3U; i++)
        {
            for (j = 0U; j < 3U; j++)
            {
                pC->M[i][j] = pA->M[j][i];
            }
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Transpose_4x4(const Matrix4x4_T* const pA, Matrix4x4_T* const pC)
{
    uint8_T i, j;

    if (pA == pC)
    {
        Matrix4x4_T Temp;

        for (i = 0U; i < 4U; i++)
        {
            for (j = 0U; j < 4U; j++)
            {
                Temp.M[i][j] = pA->M[j][i];
            }
        }

        for (i = 0U; i < 4U; i++)
        {
            for (j = 0U; j < 4U; j++)
            {
                pC->M[i][j] = Temp.M[i][j];
            }
        }
    }
    else
    {
        for (i = 0U; i < 4U; i++)
        {
            for (j = 0U; j < 4U; j++)
            {
                pC->M[i][j] = pA->M[j][i];
            }
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Transpose_3x2_to_2x3(const Matrix3x2_T* const pA, Matrix2x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[j][i];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Transpose_2x3_to_3x2(const Matrix2x3_T* const pA, Matrix3x2_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            pC->M[i][j] = pA->M[j][i];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Transpose_4x3_to_3x4(const Matrix4x3_T* const pA, Matrix3x4_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = pA->M[j][i];
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Transpose_3x4_to_4x3(const Matrix3x4_T* const pA, Matrix4x3_T* const pC)
{
    uint8_T i, j;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            pC->M[i][j] = pA->M[j][i];
        }
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Matrix Multiplication------------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Mul_2x2_2x2(const Matrix2x2_T* const pA,
                        const Matrix2x2_T* const pB,
                        Matrix2x2_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 2U; i++)
    {
        for (j = 0U; j < 2U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 2U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_2x2_Vec2(const Matrix2x2_T* const pA,
                         const Vector2_T* const pB,
                         Vector2_T* const pC)
{
    uint8_T i, j;
    real32_T sum;

    for (i = 0U; i < 2U; i++)
    {
        sum = 0.0f;
        for (j = 0U; j < 2U; j++)
        {
            sum += pA->M[i][j] * pB->V[j];
        }
        pC->V[i] = sum;
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_3x3_3x3(const Matrix3x3_T* const pA,
                        const Matrix3x3_T* const pB,
                        Matrix3x3_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 3U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_3x3_Vec3(const Matrix3x3_T* const pA,
                         const Vector3_T* const pB,
                         Vector3_T* const pC)
{
    uint8_T i, j;
    real32_T sum;

    for (i = 0U; i < 3U; i++)
    {
        sum = 0.0f;
        for (j = 0U; j < 3U; j++)
        {
            sum += pA->M[i][j] * pB->V[j];
        }
        pC->V[i] = sum;
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_4x4_4x4(const Matrix4x4_T* const pA,
                        const Matrix4x4_T* const pB,
                        Matrix4x4_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 4U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_4x4_Vec4(const Matrix4x4_T* const pA,
                         const Vector4_T* const pB,
                         Vector4_T* const pC)
{
    uint8_T i, j;
    real32_T sum;

    for (i = 0U; i < 4U; i++)
    {
        sum = 0.0f;
        for (j = 0U; j < 4U; j++)
        {
            sum += pA->M[i][j] * pB->V[j];
        }
        pC->V[i] = sum;
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_4x4_4x3(const Matrix4x4_T* const pA,
                        const Matrix4x3_T* const pB,
                        Matrix4x3_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 4U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_3x4_4x3(const Matrix3x4_T* const pA,
                        const Matrix4x3_T* const pB,
                        Matrix3x3_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 4U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_4x3_3x4(const Matrix4x3_T* const pA,
                        const Matrix3x4_T* const pB,
                        Matrix4x4_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 3U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_4x3_3x3(const Matrix4x3_T* const pA,
                        const Matrix3x3_T* const pB,
                        Matrix4x3_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 3U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 3U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_3x4_4x4(const Matrix3x4_T* const pA,
                        const Matrix4x4_T* const pB,
                        Matrix3x4_T* const pC)
{
    uint8_T i, j, k;
    real32_T sum;

    for (i = 0U; i < 3U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            sum = 0.0f;
            for (k = 0U; k < 4U; k++)
            {
                sum += pA->M[i][k] * pB->M[k][j];
            }
            pC->M[i][j] = sum;
        }
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_3x2_Vec2(const Matrix3x2_T* const pA,
                         const Vector2_T* const pB,
                         Vector3_T* const pC)
{
    uint8_T i, j;
    real32_T sum;

    for (i = 0U; i < 3U; i++)
    {
        sum = 0.0f;
        for (j = 0U; j < 2U; j++)
        {
            sum += pA->M[i][j] * pB->V[j];
        }
        pC->V[i] = sum;
    }
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Mul_2x3_Vec3(const Matrix2x3_T* const pA,
                         const Vector3_T* const pB,
                         Vector2_T* const pC)
{
    uint8_T i, j;
    real32_T sum;

    for (i = 0U; i < 2U; i++)
    {
        sum = 0.0f;
        for (j = 0U; j < 3U; j++)
        {
            sum += pA->M[i][j] * pB->V[j];
        }
        pC->V[i] = sum;
    }
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Determinant Calculations---------------------------*/
/******************************************************************************/

MAT_FAST_START
real32_T Matrix_Det_2x2(const Matrix2x2_T* const pA)
{
    return (pA->M[0][0] * pA->M[1][1]) - (pA->M[0][1] * pA->M[1][0]);
}
MAT_FAST_END

MAT_FAST_START
real32_T Matrix_Det_3x3(const Matrix3x3_T* const pA)
{
    real32_T det;

    det = pA->M[0][0] * (pA->M[1][1] * pA->M[2][2] - pA->M[1][2] * pA->M[2][1])
        - pA->M[0][1] * (pA->M[1][0] * pA->M[2][2] - pA->M[1][2] * pA->M[2][0])
        + pA->M[0][2] * (pA->M[1][0] * pA->M[2][1] - pA->M[1][1] * pA->M[2][0]);

    return det;
}
MAT_FAST_END

MAT_FAST_START
real32_T Matrix_Det_4x4(const Matrix4x4_T* const pA)
{
    real32_T det;
    real32_T a11, a12, a13, a14, a21, a22, a23, a24;
    real32_T a31, a32, a33, a34, a41, a42, a43, a44;

    a11 = pA->M[0][0]; a12 = pA->M[0][1]; a13 = pA->M[0][2]; a14 = pA->M[0][3];
    a21 = pA->M[1][0]; a22 = pA->M[1][1]; a23 = pA->M[1][2]; a24 = pA->M[1][3];
    a31 = pA->M[2][0]; a32 = pA->M[2][1]; a33 = pA->M[2][2]; a34 = pA->M[2][3];
    a41 = pA->M[3][0]; a42 = pA->M[3][1]; a43 = pA->M[3][2]; a44 = pA->M[3][3];

    det = a11 * (a22 * (a33 * a44 - a34 * a43) -
                 a23 * (a32 * a44 - a34 * a42) +
                 a24 * (a32 * a43 - a33 * a42))
        - a12 * (a21 * (a33 * a44 - a34 * a43) -
                 a23 * (a31 * a44 - a34 * a41) +
                 a24 * (a31 * a43 - a33 * a41))
        + a13 * (a21 * (a32 * a44 - a34 * a42) -
                 a22 * (a31 * a44 - a34 * a41) +
                 a24 * (a31 * a42 - a32 * a41))
        - a14 * (a21 * (a32 * a43 - a33 * a42) -
                 a22 * (a31 * a43 - a33 * a41) +
                 a23 * (a31 * a42 - a32 * a41));

    return det;
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------Matrix Inversion-----------------------------------*/
/******************************************************************************/

MAT_FAST_START
uint8_T Matrix_Invert_2x2(const Matrix2x2_T* const pA, Matrix2x2_T* const pC)
{
    real32_T det;
    real32_T inv_det;
    uint8_T status = 0U;

    det = Matrix_Det_2x2(pA);

    /* MISRA C:2012 Rule 14.4 - Non-zero test */
    if ((det > MATRIX_EPSILON) || (det < -MATRIX_EPSILON))
    {
        inv_det = 1.0f / det;

        pC->M[0][0] =  pA->M[1][1] * inv_det;
        pC->M[0][1] = -pA->M[0][1] * inv_det;
        pC->M[1][0] = -pA->M[1][0] * inv_det;
        pC->M[1][1] =  pA->M[0][0] * inv_det;

        status = 1U;
    }

    return status;
}
MAT_FAST_END

MAT_FAST_START
uint8_T Matrix_Invert_3x3(const Matrix3x3_T* const pA, Matrix3x3_T* const pC)
{
    real32_T det;
    real32_T inv_det;
    uint8_T status = 0U;

    det = Matrix_Det_3x3(pA);

    if ((det > MATRIX_EPSILON) || (det < -MATRIX_EPSILON))
    {
        inv_det = 1.0f / det;

        /* Compute adjugate matrix and multiply by 1/det */
        pC->M[0][0] =  (pA->M[1][1] * pA->M[2][2] - pA->M[1][2] * pA->M[2][1]) * inv_det;
        pC->M[0][1] = -(pA->M[0][1] * pA->M[2][2] - pA->M[0][2] * pA->M[2][1]) * inv_det;
        pC->M[0][2] =  (pA->M[0][1] * pA->M[1][2] - pA->M[0][2] * pA->M[1][1]) * inv_det;

        pC->M[1][0] = -(pA->M[1][0] * pA->M[2][2] - pA->M[1][2] * pA->M[2][0]) * inv_det;
        pC->M[1][1] =  (pA->M[0][0] * pA->M[2][2] - pA->M[0][2] * pA->M[2][0]) * inv_det;
        pC->M[1][2] = -(pA->M[0][0] * pA->M[1][2] - pA->M[0][2] * pA->M[1][0]) * inv_det;

        pC->M[2][0] =  (pA->M[1][0] * pA->M[2][1] - pA->M[1][1] * pA->M[2][0]) * inv_det;
        pC->M[2][1] = -(pA->M[0][0] * pA->M[2][1] - pA->M[0][1] * pA->M[2][0]) * inv_det;
        pC->M[2][2] =  (pA->M[0][0] * pA->M[1][1] - pA->M[0][1] * pA->M[1][0]) * inv_det;

        status = 1U;
    }

    return status;
}
MAT_FAST_END

MAT_FAST_START
uint8_T Matrix_Invert_4x4(const Matrix4x4_T* const pA, Matrix4x4_T* const pC)
{
    /* Gauss-Jordan elimination with partial pivoting */
    real32_T aug[4][8];
    real32_T temp, pivot;
    uint8_T i, j, k, pivot_row;
    uint8_T status = 0U;

    /* Create augmented matrix [A | I] */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            aug[i][j] = pA->M[i][j];
            aug[i][j + 4U] = (i == j) ? 1.0f : 0.0f;
        }
    }

    /* Gaussian elimination */
    for (i = 0U; i < 4U; i++)
    {
        /* Find pivot */
        pivot_row = i;
        pivot = aug[i][i];

        for (j = i + 1U; j < 4U; j++)
        {
            /* BUG FIX: use fabsf for correct absolute-value pivot comparison */
            if (fabsf(aug[j][i]) > fabsf(pivot))
            {
                pivot = aug[j][i];
                pivot_row = j;
            }
        }

        /* Check for singular matrix using updated pivot after row search */
        if (fabsf(pivot) < MATRIX_EPSILON)
        {
            /* MISRA C:2012 Rule 15.5: single exit point - use flag instead of early return */
            status = 0U;
            break; /* exit Gaussian elimination loop */
        }

        /* Swap rows if necessary */
        if (pivot_row != i)
        {
            for (k = 0U; k < 8U; k++)
            {
                temp = aug[i][k];
                aug[i][k] = aug[pivot_row][k];
                aug[pivot_row][k] = temp;
            }
        }

        /* Normalize pivot row */
        pivot = aug[i][i];
        for (k = 0U; k < 8U; k++)
        {
            aug[i][k] /= pivot;
        }

        /* Eliminate column */
        for (j = 0U; j < 4U; j++)
        {
            if (j != i)
            {
                temp = aug[j][i];
                for (k = 0U; k < 8U; k++)
                {
                    aug[j][k] -= temp * aug[i][k];
                }
            }
        }
    }

    /* Extract inverse from augmented matrix */
    for (i = 0U; i < 4U; i++)
    {
        for (j = 0U; j < 4U; j++)
        {
            pC->M[i][j] = aug[i][j + 4U];
        }
    }

    status = 1U;
    return status;
}
MAT_FAST_END

/******************************************************************************/
/*-------------------------PMSM Specific Matrices-----------------------------*/
/******************************************************************************/

MAT_FAST_START
void Matrix_Clarke_Init(Matrix3x2_T* const pC)
{
    /* Power-invariant Clarke matrix */
    pC->M[0][0] = 0.66666667f;   /* 2/3 */
    pC->M[1][0] = -0.33333333f;  /* -1/3 */
    pC->M[2][0] = -0.33333333f;  /* -1/3 */

    pC->M[0][1] = 0.0f;
    pC->M[1][1] = 0.57735027f;   /* 1/√3 */
    pC->M[2][1] = -0.57735027f;  /* -1/√3 */
}
MAT_FAST_END

MAT_FAST_START
void Matrix_InvClarke_Init(Matrix3x2_T* const pC)
{
    /* Inverse Clarke matrix */
    pC->M[0][0] = 1.0f;
    pC->M[0][1] = 0.0f;

    pC->M[1][0] = -0.5f;
    pC->M[1][1] = 0.86602540f;   /* √3/2 */

    pC->M[2][0] = -0.5f;
    pC->M[2][1] = -0.86602540f;  /* -√3/2 */
}
MAT_FAST_END

MAT_FAST_START
void Matrix_Park_Init(Matrix2x2_T* const pC, const real32_T Theta)
{
    real32_T cos_theta;
    real32_T sin_theta;

    cos_theta = cosf(Theta);
    sin_theta = sinf(Theta);

    /* Park rotation matrix */
    pC->M[0][0] = cos_theta;
    pC->M[0][1] = sin_theta;
    pC->M[1][0] = -sin_theta;
    pC->M[1][1] = cos_theta;
}
MAT_FAST_END

MAT_FAST_START
void Matrix_InvPark_Init(Matrix2x2_T* const pC, const real32_T Theta)
{
    real32_T cos_theta;
    real32_T sin_theta;

    cos_theta = cosf(Theta);
    sin_theta = sinf(Theta);

    /* Inverse Park (transpose) */
    pC->M[0][0] = cos_theta;
    pC->M[0][1] = -sin_theta;
    pC->M[1][0] = sin_theta;
    pC->M[1][1] = cos_theta;
}
MAT_FAST_END