/* ============================================================================
 * coordinate_transform.c
 * ============================================================================
 * Clarke / Park transformation implementations for PMSM FOC
 *
 * All functions are stateless (pure).  Safe to call from ISRs.
 * ============================================================================
 */

#include "coordinate_transform.h"
#include <math.h>

/* ── Constants ────────────────────────────────────────────────────────────── */
#define SQRT3        1.73205081f   /* √3             */
#define INV_SQRT3    0.57735027f   /* 1/√3           */
#define TWO_THIRDS   0.66666667f   /* 2/3            */
#define ONE_THIRD    0.33333333f   /* 1/3            */
#define HALF_SQRT3   0.86602540f   /* √3/2           */


/* ============================================================================
 * Clarke transform   abc → αβ   (power-invariant)
 * ============================================================================ */

void clarke_transform_compute(const ClarkeInputs* in, ClarkeOutputs* out)
{
    out->alpha = TWO_THIRDS * in->ia - ONE_THIRD * in->ib - ONE_THIRD * in->ic;
    out->beta  = (in->ib - in->ic) * INV_SQRT3;
}

/* FIXED: Added missing function name */
void clarke_transform_compute_flat(const float* in_buf, float* out_buf)
{
    ClarkeInputs  in  = { in_buf[0], in_buf[1], in_buf[2] };
    ClarkeOutputs out_s;
    clarke_transform_compute(&in, &out_s);
    out_buf[0] = out_s.alpha;
    out_buf[1] = out_s.beta;
}

/* ============================================================================
 * Inverse Clarke transform   αβ → abc
 * ============================================================================ */
void inv_clarke_transform_compute(const AlphaBetaInputs* in, ABCOutputs* out)
{
    out->a = in->alpha;
    out->b = -0.5f * in->alpha + HALF_SQRT3 * in->beta;
    out->c = -0.5f * in->alpha - HALF_SQRT3 * in->beta;
}

void inv_clarke_transform_compute_flat(const float* in_buf, float* out_buf)
{
    AlphaBetaInputs in = { in_buf[0], in_buf[1] };
    ABCOutputs out_s;
    inv_clarke_transform_compute(&in, &out_s);
    out_buf[0] = out_s.a;
    out_buf[1] = out_s.b;
    out_buf[2] = out_s.c;
}

/* ============================================================================
 * Park transform   αβ → dq
 * ============================================================================ */
void park_transform_compute(const AlphaBetaInputs* in, float theta, DQOutputs* out)
{
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    out->d =  in->alpha * cos_theta + in->beta * sin_theta;
    out->q = -in->alpha * sin_theta + in->beta * cos_theta;
}

void park_transform_compute_flat(const float* in_buf, float theta, float* out_buf)
{
    AlphaBetaInputs in = { in_buf[0], in_buf[1] };
    DQOutputs out_s;
    park_transform_compute(&in, theta, &out_s);
    out_buf[0] = out_s.d;
    out_buf[1] = out_s.q;
}

/* ============================================================================
 * Inverse Park transform   dq → αβ
 * ============================================================================ */
void inv_park_transform_compute(const DQInputs* in, float theta, AlphaBetaOutputs* out)
{
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    out->alpha = in->d * cos_theta - in->q * sin_theta;
    out->beta  = in->d * sin_theta + in->q * cos_theta;
}

void inv_park_transform_compute_flat(const float* in_buf, float theta, float* out_buf)
{
    DQInputs in = { in_buf[0], in_buf[1] };
    AlphaBetaOutputs out_s;
    inv_park_transform_compute(&in, theta, &out_s);
    out_buf[0] = out_s.alpha;
    out_buf[1] = out_s.beta;
}