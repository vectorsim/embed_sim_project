/*
 * coordinate_transform.c
 * ======================
 *
 * Matrix-based Clarke and Park transforms for PMSM FOC.
 * All transforms implemented as matrix multiplications.
 *
 * @author EmbedSim Framework
 * @version 2.0.0
 * @date 2024
 */

/******************************************************************************/
/*----------------------------------Includes----------------------------------*/
/******************************************************************************/

#include "Coordinate_Transform.h"
#include "Matrix_Operations.h"
#include <math.h>    /**< For cosf, sinf */

/* ─────────────────────────────────────────────────────────────────────────────
 * TASKING fast-memory pragma helpers
 * ───────────────────────────────────────────────────────────────────────────*/
#if defined(CT_USE_FAST_MEMORY) && defined(__TASKING__)
#  define CT_FAST_START   _Pragma("section code \"cpu0_pspr\"")
#  define CT_FAST_END     _Pragma("section code restore")
#else
#  define CT_FAST_START
#  define CT_FAST_END
#endif

/******************************************************************************/
/*-----------------------------------Macros-----------------------------------*/
/******************************************************************************/

#define CT_TWO_THIRDS   (0.66666667f)   /**< 2/3 */
#define CT_ONE_THIRD    (0.33333333f)   /**< 1/3 */
#define CT_INV_SQRT3    (0.57735027f)   /**< 1/√3 */
#define CT_HALF_SQRT3   (0.86602540f)   /**< √3/2 */

/******************************************************************************/
/*-------------------------Static Conversion Helpers--------------------------*/
/******************************************************************************/

/**
 * @brief Convert Phase3Signal_T to Vector3_T for matrix operations
 */
static inline void Phase3_To_Vector(const Phase3Signal_T* pIn, Vector3_T* pOut)
{
    pOut->V[0] = pIn->A;
    pOut->V[1] = pIn->B;
    pOut->V[2] = pIn->C;
}

/**
 * @brief Convert AlphaBetaSignal_T to Vector2_T for matrix operations
 */
static inline void AlphaBeta_To_Vector(const AlphaBetaSignal_T* pIn, Vector2_T* pOut)
{
    pOut->V[0] = pIn->Alpha;
    pOut->V[1] = pIn->Beta;
}

/**
 * @brief Convert DQSignal_T to Vector2_T for matrix operations
 */
static inline void DQ_To_Vector(const DQSignal_T* pIn, Vector2_T* pOut)
{
    pOut->V[0] = pIn->D;
    pOut->V[1] = pIn->Q;
}

/**
 * @brief Convert Vector3_T back to Phase3Signal_T
 */
static inline void Vector_To_Phase3(const Vector3_T* pIn, Phase3Signal_T* pOut)
{
    pOut->A = pIn->V[0];
    pOut->B = pIn->V[1];
    pOut->C = pIn->V[2];
}

/**
 * @brief Convert Vector2_T back to AlphaBetaSignal_T
 */
static inline void Vector_To_AlphaBeta(const Vector2_T* pIn, AlphaBetaSignal_T* pOut)
{
    pOut->Alpha = pIn->V[0];
    pOut->Beta = pIn->V[1];
}

/**
 * @brief Convert Vector2_T back to DQSignal_T
 */
static inline void Vector_To_DQ(const Vector2_T* pIn, DQSignal_T* pOut)
{
    pOut->D = pIn->V[0];
    pOut->Q = pIn->V[1];
}

/******************************************************************************/
/*-------------------------Function Implementations---------------------------*/
/******************************************************************************/

CT_FAST_START
void Clarke_InitMatrix(ClarkeMatrix_T* pMatrix)
{
    /* Power-invariant Clarke matrix:
     * [α]   = [ 2/3   -1/3   -1/3 ] [A]
     * [β]     [ 0     1/√3   -1/√3] [B]
     *                                 [C]
     */
    MAT_SET(pMatrix, 0, 0, CT_TWO_THIRDS);    /* α = f(A) */
    MAT_SET(pMatrix, 1, 0, -CT_ONE_THIRD);    /* α = f(B) */
    MAT_SET(pMatrix, 2, 0, -CT_ONE_THIRD);    /* α = f(C) */

    MAT_SET(pMatrix, 0, 1, 0.0f);              /* β independent of A */
    MAT_SET(pMatrix, 1, 1, CT_INV_SQRT3);      /* β = f(B) */
    MAT_SET(pMatrix, 2, 1, -CT_INV_SQRT3);     /* β = f(C) */
}
CT_FAST_END

CT_FAST_START
void InvClarke_InitMatrix(Matrix3x2_T* pMatrix)
{
    /* Inverse Clarke matrix (power-invariant):
     * [A]   = [ 1    0    ] [α]
     * [B]     [ -1/2  √3/2] [β]
     * [C]     [ -1/2 -√3/2]
     */
    MAT_SET(pMatrix, 0, 0, 1.0f);               /* A = f(α) */
    MAT_SET(pMatrix, 0, 1, 0.0f);               /* A = f(β) */

    MAT_SET(pMatrix, 1, 0, -0.5f);              /* B = f(α) */
    MAT_SET(pMatrix, 1, 1, CT_HALF_SQRT3);      /* B = f(β) */

    MAT_SET(pMatrix, 2, 0, -0.5f);              /* C = f(α) */
    MAT_SET(pMatrix, 2, 1, -CT_HALF_SQRT3);     /* C = f(β) */
}
CT_FAST_END

CT_FAST_START
void Park_InitMatrix(ParkMatrix_T* pMatrix, real32_T theta)
{
    real32_T cos_theta = cosf(theta);
    real32_T sin_theta = sinf(theta);

    /* Park rotation matrix (d-q aligned with rotor flux):
     * [d]   = [ cosθ  sinθ ] [α]
     * [q]     [-sinθ  cosθ ] [β]
     */
    MAT_SET(pMatrix, 0, 0, cos_theta);    /* d = f(α) */
    MAT_SET(pMatrix, 0, 1, sin_theta);    /* d = f(β) */
    MAT_SET(pMatrix, 1, 0, -sin_theta);   /* q = f(α) */
    MAT_SET(pMatrix, 1, 1, cos_theta);    /* q = f(β) */
}
CT_FAST_END

CT_FAST_START
void InvPark_InitMatrix(ParkMatrix_T* pMatrix, real32_T theta)
{
    real32_T cos_theta = cosf(theta);
    real32_T sin_theta = sinf(theta);

    /* Inverse Park (just transpose for rotation matrix):
     * [α]   = [ cosθ  -sinθ ] [d]
     * [β]     [ sinθ   cosθ ] [q]
     */
    MAT_SET(pMatrix, 0, 0, cos_theta);    /* α = f(d) */
    MAT_SET(pMatrix, 0, 1, -sin_theta);   /* α = f(q) */
    MAT_SET(pMatrix, 1, 0, sin_theta);    /* β = f(d) */
    MAT_SET(pMatrix, 1, 1, cos_theta);    /* β = f(q) */
}
CT_FAST_END

CT_FAST_START
void Clarke_Transform(const ClarkeMatrix_T* pMatrix,
                      const Phase3Signal_T* pPhase3SignalIn,
                      AlphaBetaSignal_T* pAlphaBetaSignalOut)
{
    Vector3_T in_vec;
    Vector2_T out_vec;

    /* Convert input struct to vector */
    Phase3_To_Vector(pPhase3SignalIn, &in_vec);

    /* Matrix multiplication: out = A^T * in  (since matrix is 3x2) */
    out_vec.V[0] = MAT_ELEM(pMatrix, 0, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 1, 0) * in_vec.V[1] +
                   MAT_ELEM(pMatrix, 2, 0) * in_vec.V[2];

    out_vec.V[1] = MAT_ELEM(pMatrix, 0, 1) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 1, 1) * in_vec.V[1] +
                   MAT_ELEM(pMatrix, 2, 1) * in_vec.V[2];

    /* Convert result vector back to output struct */
    Vector_To_AlphaBeta(&out_vec, pAlphaBetaSignalOut);
}
CT_FAST_END

CT_FAST_START
void InvClarke_Transform(const Matrix3x2_T* pMatrix,
                         const AlphaBetaSignal_T* pAlphaBetaSignalIn,
                         Phase3Signal_T* pPhase3SignalOut)
{
    Vector2_T in_vec;
    Vector3_T out_vec;

    /* Convert input struct to vector */
    AlphaBeta_To_Vector(pAlphaBetaSignalIn, &in_vec);

    /* Matrix multiplication: out = A * in */
    out_vec.V[0] = MAT_ELEM(pMatrix, 0, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 0, 1) * in_vec.V[1];

    out_vec.V[1] = MAT_ELEM(pMatrix, 1, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 1, 1) * in_vec.V[1];

    out_vec.V[2] = MAT_ELEM(pMatrix, 2, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 2, 1) * in_vec.V[1];

    /* Convert result vector back to output struct */
    Vector_To_Phase3(&out_vec, pPhase3SignalOut);
}
CT_FAST_END

CT_FAST_START
void Park_Transform(const ParkMatrix_T* pMatrix,
                    const AlphaBetaSignal_T* pAlphaBetaSignalIn,
                    DQSignal_T* pDQSignalOut)
{
    Vector2_T in_vec;
    Vector2_T out_vec;

    /* Convert input struct to vector */
    AlphaBeta_To_Vector(pAlphaBetaSignalIn, &in_vec);

    /* Matrix multiplication: out = A * in */
    out_vec.V[0] = MAT_ELEM(pMatrix, 0, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 0, 1) * in_vec.V[1];

    out_vec.V[1] = MAT_ELEM(pMatrix, 1, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 1, 1) * in_vec.V[1];

    /* Convert result vector back to output struct */
    Vector_To_DQ(&out_vec, pDQSignalOut);
}
CT_FAST_END

CT_FAST_START
void InvPark_Transform(const ParkMatrix_T* pMatrix,
                       const DQSignal_T* pDQSignalIn,
                       AlphaBetaSignal_T* pAlphaBetaSignalOut)
{
    Vector2_T in_vec;
    Vector2_T out_vec;

    /* Convert input struct to vector */
    DQ_To_Vector(pDQSignalIn, &in_vec);

    /* Matrix multiplication: out = A * in */
    out_vec.V[0] = MAT_ELEM(pMatrix, 0, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 0, 1) * in_vec.V[1];

    out_vec.V[1] = MAT_ELEM(pMatrix, 1, 0) * in_vec.V[0] +
                   MAT_ELEM(pMatrix, 1, 1) * in_vec.V[1];

    /* Convert result vector back to output struct */
    Vector_To_AlphaBeta(&out_vec, pAlphaBetaSignalOut);
}
CT_FAST_END

CT_FAST_START
void ClarkePark_Transform(const ParkMatrix_T* pParkMatrix,
                          const ClarkeMatrix_T* pClarkeMatrix,
                          const Phase3Signal_T* pPhase3SignalIn,
                          DQSignal_T* pDQSignalOut)
{
    /* First do Clarke to get alpha-beta */
    AlphaBetaSignal_T alpha_beta;
    Clarke_Transform(pClarkeMatrix, pPhase3SignalIn, &alpha_beta);

    /* Then do Park to get d-q */
    Park_Transform(pParkMatrix, &alpha_beta, pDQSignalOut);
}
CT_FAST_END