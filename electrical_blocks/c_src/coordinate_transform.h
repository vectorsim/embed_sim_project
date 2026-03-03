/* ============================================================================
 * coordinate_transform.h
 * ============================================================================
 * Clarke / Park transformation functions for PMSM FOC
 *
 * All functions are pure (no internal state) and reentrant.
 * ============================================================================
 */

#ifndef COORDINATE_TRANSFORM_H
#define COORDINATE_TRANSFORM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <math.h>

/* --------------------------------------------------------------------------
 * Data structures
 * -------------------------------------------------------------------------- */
typedef struct ClarkeInputs {
    float ia;
    float ib;
    float ic;
} ClarkeInputs;

typedef struct ClarkeOutputs {
    float alpha;
    float beta;
} ClarkeOutputs;

typedef struct AlphaBetaInputs {
    float alpha;
    float beta;
} AlphaBetaInputs;

typedef struct ABCOutputs {
    float a;
    float b;
    float c;
} ABCOutputs;

typedef struct DQInputs {
    float d;
    float q;
} DQInputs;

typedef struct DQOutputs {
    float d;
    float q;
} DQOutputs;

typedef struct AlphaBetaOutputs {
    float alpha;
    float beta;
} AlphaBetaOutputs;

/* --------------------------------------------------------------------------
 * Clarke transform   abc → αβ   (power-invariant)
 * -------------------------------------------------------------------------- */
void clarke_transform_compute(const ClarkeInputs* in, ClarkeOutputs* out);
void clarke_transform_compute_flat(const float* in_buf, float* out_buf);

/* --------------------------------------------------------------------------
 * Inverse Clarke transform   αβ → abc
 * -------------------------------------------------------------------------- */
void inv_clarke_transform_compute(const AlphaBetaInputs* in, ABCOutputs* out);
void inv_clarke_transform_compute_flat(const float* in_buf, float* out_buf);

/* --------------------------------------------------------------------------
 * Park transform   αβ → dq
 * -------------------------------------------------------------------------- */
void park_transform_compute(const AlphaBetaInputs* in, float theta, DQOutputs* out);
void park_transform_compute_flat(const float* in_buf, float theta, float* out_buf);

/* --------------------------------------------------------------------------
 * Inverse Park transform   dq → αβ
 * -------------------------------------------------------------------------- */
void inv_park_transform_compute(const DQInputs* in, float theta, AlphaBetaOutputs* out);
void inv_park_transform_compute_flat(const float* in_buf, float theta, float* out_buf);

#ifdef __cplusplus
}
#endif

#endif /* COORDINATE_TRANSFORM_H */