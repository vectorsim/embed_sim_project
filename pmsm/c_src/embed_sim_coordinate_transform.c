/**********************************************************************************************************************
 * \file      embed_sim_coordinate_transform.c
 * \brief     Matrix-based Clarke, Park, Inverse-Park and Inverse-Clarke transforms
 *            for FOC motor control using EmbedSim matrix library.
 *
 * \details   All transforms use matrix multiplication from embed_sim_matrix.h
 *            for clarity and code reuse.
 *
 * \version   2.1.0
 * \date      2025-05-24
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_coordinate_transform.h"
#include "embed_sim_compiler.h"
#include <math.h>
#include <stddef.h>

/**********************************************************************************************************************
 * Module-Private Matrix Buffers (static allocation)
 *
 * static gives these file scope only — not visible outside this TU.
 * Naming: UPPER_SNAKE_CASE prefix + _G suffix per EmbedSim convention.
 *********************************************************************************************************************/
static MatrixElement Clarke_Matrix_Data_G[CLARKE_ROWS * CLARKE_COLS];
static MatrixElement Inv_Clarke_Matrix_Data_G[INV_CLARKE_ROWS * INV_CLARKE_COLS];
static MatrixElement Park_Cos_Matrix_Data_G[PARK_ROWS * PARK_COLS];
static MatrixElement Park_Sin_Matrix_Data_G[PARK_ROWS * PARK_COLS];
static MatrixElement Inv_Park_Cos_Matrix_Data_G[INV_PARK_ROWS * INV_PARK_COLS];
static MatrixElement Inv_Park_Sin_Matrix_Data_G[INV_PARK_ROWS * INV_PARK_COLS];

/**********************************************************************************************************************
 * Module-Private Matrix Handles
 *********************************************************************************************************************/
static Matrix_Type Clarke_Matrix_G;
static Matrix_Type Inv_Clarke_Matrix_G;
static Matrix_Type Park_Cos_Matrix_G;
static Matrix_Type Park_Sin_Matrix_G;
static Matrix_Type Inv_Park_Cos_Matrix_G;
static Matrix_Type Inv_Park_Sin_Matrix_G;

/**********************************************************************************************************************
 * Private Helper Functions
 *********************************************************************************************************************/
/* All coordinate conversions are performed inline within each public function.
 * No module-private helpers are currently required.                           */

/**********************************************************************************************************************
 * Public Functions
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Transform_Init
 *------------------------------------------------------------------------------------------------------------------*/
void Transform_Init(void)
{
    /* Initialize Clarke transform matrix: [1, 0; 1/√3, 2/√3] */
    Matrix_Init(&Clarke_Matrix_G, Clarke_Matrix_Data_G, CLARKE_ROWS, CLARKE_COLS);
    Matrix_SetElementFloat(&Clarke_Matrix_G, 0U, 0U, ES_MATH_ONE_F);
    Matrix_SetElementFloat(&Clarke_Matrix_G, 0U, 1U, 0.0f);
    Matrix_SetElementFloat(&Clarke_Matrix_G, 1U, 0U, ES_MATH_INV_SQRT3_F);
    Matrix_SetElementFloat(&Clarke_Matrix_G, 1U, 1U, ES_MATH_TWO_INV_SQRT3_F);

    /* Initialize Inverse-Clarke transform matrix */
    Matrix_Init(&Inv_Clarke_Matrix_G, Inv_Clarke_Matrix_Data_G, INV_CLARKE_ROWS, INV_CLARKE_COLS);
    Matrix_SetElementFloat(&Inv_Clarke_Matrix_G, 0U, 0U,  ES_MATH_ONE_F);
    Matrix_SetElementFloat(&Inv_Clarke_Matrix_G, 0U, 1U,  0.0f);
    Matrix_SetElementFloat(&Inv_Clarke_Matrix_G, 1U, 0U, -ES_MATH_HALF_F);
    Matrix_SetElementFloat(&Inv_Clarke_Matrix_G, 1U, 1U,  ES_MATH_HALF_SQRT3_F);
    Matrix_SetElementFloat(&Inv_Clarke_Matrix_G, 2U, 0U, -ES_MATH_HALF_F);
    Matrix_SetElementFloat(&Inv_Clarke_Matrix_G, 2U, 1U, -ES_MATH_HALF_SQRT3_F);

    /* Initialize Park matrices (will be updated per transform call) */
    Matrix_Init(&Park_Cos_Matrix_G, Park_Cos_Matrix_Data_G, PARK_ROWS, PARK_COLS);
    Matrix_Init(&Park_Sin_Matrix_G, Park_Sin_Matrix_Data_G, PARK_ROWS, PARK_COLS);
    Matrix_Init(&Inv_Park_Cos_Matrix_G, Inv_Park_Cos_Matrix_Data_G, INV_PARK_ROWS, INV_PARK_COLS);
    Matrix_Init(&Inv_Park_Sin_Matrix_G, Inv_Park_Sin_Matrix_Data_G, INV_PARK_ROWS, INV_PARK_COLS);
}

/*--------------------------------------------------------------------------------------------------------------------
 * Clarke_Transform_Matrix
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Clarke_Transform_Matrix(
    const FocUvw_T       * const In_P,
    FocAlphaBeta_T       * const Out_P)
{
    MatrixStatus_Type status;
    MatrixElement     input_buffer[CLARKE_COLS];
    MatrixElement     output_buffer[CLARKE_ROWS];
    Matrix_Type       input_vec;
    Matrix_Type       output_vec;

    status = MATRIX_SUCCESS;

    if ((In_P == NULL) || (Out_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* Create input vector [U; V] (2×1) */
        Matrix_Init(&input_vec, input_buffer, CLARKE_COLS, 1U);
        Matrix_SetElementFloat(&input_vec, 0U, 0U, In_P->U);
        Matrix_SetElementFloat(&input_vec, 1U, 0U, In_P->V);

        /* Create output vector (2×1) */
        Matrix_Init(&output_vec, output_buffer, CLARKE_ROWS, 1U);

        /* Multiply: output = Clarke_matrix × input */
        status = Matrix_Multiply(&Clarke_Matrix_G, &input_vec, &output_vec);

        if (status == MATRIX_SUCCESS)
        {
            Matrix_GetElementFloat(&output_vec, 0U, 0U, &Out_P->Alpha);
            Matrix_GetElementFloat(&output_vec, 1U, 0U, &Out_P->Beta);
        }
        else
        {
            /* Multiplication failed */
        }
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Park_Transform_Matrix
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type Park_Transform_Matrix(
    const FocAlphaBeta_T * const In_P,
    const FocAngle_T     * const Angle_P,
    FocDq_T              * const Out_P)
{
    MatrixStatus_Type status;
    MatrixElement     input_buffer[PARK_COLS];
    MatrixElement     output_buffer[PARK_ROWS];
    Matrix_Type       input_vec;
    Matrix_Type       output_vec;
    MatrixFloat       cos_theta;
    MatrixFloat       sin_theta;

    status = MATRIX_SUCCESS;

    if ((In_P == NULL) || (Angle_P == NULL) || (Out_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* Compute sin and cos of electrical angle */
        cos_theta = cosf(Angle_P->ThetaE);
        sin_theta = sinf(Angle_P->ThetaE);

        /* Build Park transform matrix: [cosθ, sinθ; -sinθ, cosθ] */
        Matrix_SetElementFloat(&Park_Cos_Matrix_G, 0U, 0U,  cos_theta);
        Matrix_SetElementFloat(&Park_Cos_Matrix_G, 0U, 1U,  sin_theta);
        Matrix_SetElementFloat(&Park_Cos_Matrix_G, 1U, 0U, -sin_theta);
        Matrix_SetElementFloat(&Park_Cos_Matrix_G, 1U, 1U,  cos_theta);

        /* Create input vector [Alpha; Beta] (2×1) */
        Matrix_Init(&input_vec, input_buffer, PARK_COLS, 1U);
        Matrix_SetElementFloat(&input_vec, 0U, 0U, In_P->Alpha);
        Matrix_SetElementFloat(&input_vec, 1U, 0U, In_P->Beta);

        /* Create output vector (2×1) */
        Matrix_Init(&output_vec, output_buffer, PARK_ROWS, 1U);

        /* Multiply: output = Park_matrix × input */
        status = Matrix_Multiply(&Park_Cos_Matrix_G, &input_vec, &output_vec);

        if (status == MATRIX_SUCCESS)
        {
            Matrix_GetElementFloat(&output_vec, 0U, 0U, &Out_P->D);
            Matrix_GetElementFloat(&output_vec, 1U, 0U, &Out_P->Q);
        }
        else
        {
            /* Multiplication failed */
        }
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * InvPark_Transform_Matrix
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type InvPark_Transform_Matrix(
    const FocDq_T        * const In_P,
    const FocAngle_T     * const Angle_P,
    FocAlphaBeta_T       * const Out_P)
{
    MatrixStatus_Type status;
    MatrixElement     input_buffer[INV_PARK_COLS];
    MatrixElement     output_buffer[INV_PARK_ROWS];
    Matrix_Type       input_vec;
    Matrix_Type       output_vec;
    MatrixFloat       cos_theta;
    MatrixFloat       sin_theta;

    status = MATRIX_SUCCESS;

    if ((In_P == NULL) || (Angle_P == NULL) || (Out_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* Compute sin and cos of electrical angle */
        cos_theta = cosf(Angle_P->ThetaE);
        sin_theta = sinf(Angle_P->ThetaE);

        /* Build Inverse-Park transform matrix: [cosθ, -sinθ; sinθ, cosθ] */
        Matrix_SetElementFloat(&Inv_Park_Cos_Matrix_G, 0U, 0U,  cos_theta);
        Matrix_SetElementFloat(&Inv_Park_Cos_Matrix_G, 0U, 1U, -sin_theta);
        Matrix_SetElementFloat(&Inv_Park_Cos_Matrix_G, 1U, 0U,  sin_theta);
        Matrix_SetElementFloat(&Inv_Park_Cos_Matrix_G, 1U, 1U,  cos_theta);

        /* Create input vector [D; Q] (2×1) */
        Matrix_Init(&input_vec, input_buffer, INV_PARK_COLS, 1U);
        Matrix_SetElementFloat(&input_vec, 0U, 0U, In_P->D);
        Matrix_SetElementFloat(&input_vec, 1U, 0U, In_P->Q);

        /* Create output vector (2×1) */
        Matrix_Init(&output_vec, output_buffer, INV_PARK_ROWS, 1U);

        /* Multiply: output = InvPark_matrix × input */
        status = Matrix_Multiply(&Inv_Park_Cos_Matrix_G, &input_vec, &output_vec);

        if (status == MATRIX_SUCCESS)
        {
            Matrix_GetElementFloat(&output_vec, 0U, 0U, &Out_P->Alpha);
            Matrix_GetElementFloat(&output_vec, 1U, 0U, &Out_P->Beta);
        }
        else
        {
            /* Multiplication failed */
        }
    }

    return status;
}

/*--------------------------------------------------------------------------------------------------------------------
 * InvClarke_Transform_Matrix
 *------------------------------------------------------------------------------------------------------------------*/
MatrixStatus_Type InvClarke_Transform_Matrix(
    const FocAlphaBeta_T * const In_P,
    FocUvw_T             * const Out_P)
{
    MatrixStatus_Type status;
    MatrixElement     input_buffer[INV_CLARKE_COLS];
    MatrixElement     output_buffer[INV_CLARKE_ROWS];
    Matrix_Type       input_vec;
    Matrix_Type       output_vec;

    status = MATRIX_SUCCESS;

    if ((In_P == NULL) || (Out_P == NULL))
    {
        status = MATRIX_ERROR_NULL_PTR;
    }
    else
    {
        /* Create input vector [Alpha; Beta] (2×1) */
        Matrix_Init(&input_vec, input_buffer, INV_CLARKE_COLS, 1U);
        Matrix_SetElementFloat(&input_vec, 0U, 0U, In_P->Alpha);
        Matrix_SetElementFloat(&input_vec, 1U, 0U, In_P->Beta);

        /* Create output vector (3×1) */
        Matrix_Init(&output_vec, output_buffer, INV_CLARKE_ROWS, 1U);

        /* Multiply: output = InvClarke_matrix × input */
        status = Matrix_Multiply(&Inv_Clarke_Matrix_G, &input_vec, &output_vec);

        if (status == MATRIX_SUCCESS)
        {
            Matrix_GetElementFloat(&output_vec, 0U, 0U, &Out_P->U);
            Matrix_GetElementFloat(&output_vec, 1U, 0U, &Out_P->V);
            Matrix_GetElementFloat(&output_vec, 2U, 0U, &Out_P->W);
        }
        else
        {
            /* Multiplication failed */
        }
    }

    return status;
}
