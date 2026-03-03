/* ============================================================================
 * transforms.c
 * ============================================================================
 * Clarke / Park transformation implementations for PMSM FOC
 *
 * All functions are stateless (pure).  Safe to call from ISRs.
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o libtransforms.so transforms.c -lm
 * ============================================================================
 */

#include "transforms.h"
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

void inv_clarke_transform_compute(const InvClarkeInputs* in, InvClarkeOutputs* out)
{
    const float alpha = in->alpha;
    const float beta  = in->beta;

    out->ia =  alpha;
    out->ib = -0.5f * alpha + HALF_SQRT3 * beta;
    out->ic = -0.5f * alpha - HALF_SQRT3 * beta;
}

void inv_clarke_transform_compute_flat(const float* in_buf, float* out_buf)
{
    InvClarkeInputs  in  = { in_buf[0], in_buf[1] };
    InvClarkeOutputs out_s;
    inv_clarke_transform_compute(&in, &out_s);
    out_buf[0] = out_s.ia;
    out_buf[1] = out_s.ib;
    out_buf[2] = out_s.ic;
}


/* ============================================================================
 * Park transform   αβ → dq
 * ============================================================================ */

void park_transform_compute(const ParkInputs* in, ParkOutputs* out)
{
    const float cos_th = cosf(in->theta);
    const float sin_th = sinf(in->theta);

    out->d =  in->alpha * cos_th + in->beta * sin_th;
    out->q = -in->alpha * sin_th + in->beta * cos_th;
}

void park_transform_compute_flat(const float* in_buf, float* out_buf)
{
    ParkInputs  in  = { in_buf[0], in_buf[1], in_buf[2] };
    ParkOutputs out_s;
    park_transform_compute(&in, &out_s);
    out_buf[0] = out_s.d;
    out_buf[1] = out_s.q;
}


/* ============================================================================
 * Inverse Park transform   dq → αβ
 * ============================================================================ */

void inv_park_transform_compute(const InvParkInputs* in, InvParkOutputs* out)
{
    const float cos_th = cosf(in->theta);
    const float sin_th = sinf(in->theta);

    out->alpha = in->d * cos_th - in->q * sin_th;
    out->beta  = in->d * sin_th + in->q * cos_th;
}

void inv_park_transform_compute_flat(const float* in_buf, float* out_buf)
{
    InvParkInputs  in  = { in_buf[0], in_buf[1], in_buf[2] };
    InvParkOutputs out_s;
    inv_park_transform_compute(&in, &out_s);
    out_buf[0] = out_s.alpha;
    out_buf[1] = out_s.beta;
}
