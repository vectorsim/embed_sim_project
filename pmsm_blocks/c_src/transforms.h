/* ============================================================================
 * transforms.h
 * ============================================================================
 * Clarke / Park transformation functions for PMSM FOC
 *
 * Functions
 * ---------
 *   clarke_transform_compute        abc  →  αβ  (power-invariant)
 *   inv_clarke_transform_compute    αβ   →  abc
 *   park_transform_compute          αβ   →  dq
 *   inv_park_transform_compute      dq   →  αβ
 *
 * All functions are pure (no internal state) and reentrant.
 *
 * Compile:
 *   gcc -O2 -shared -fPIC -o libtransforms.so transforms.c -lm
 * ============================================================================
 */

#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#ifdef __cplusplus
extern "C" {
#endif

/* --------------------------------------------------------------------------
 * Clarke transform   abc → αβ   (power-invariant)
 *
 *   α = (2·ia − ib − ic) / 3
 *   β = (ib − ic) / √3
 * -------------------------------------------------------------------------- */
typedef struct ClarkeInputs  { float ia; float ib; float ic; } ClarkeInputs;
typedef struct ClarkeOutputs { float alpha; float beta;       } ClarkeOutputs;

void clarke_transform_compute(const ClarkeInputs*  in,
                               ClarkeOutputs*       out);

/* Flat-array wrappers for Cython bridge */
void clarke_transform_compute_flat(const float* in_buf,   /* [ia, ib, ic]   */
                                    float*       out_buf); /* [alpha, beta]  */

/* --------------------------------------------------------------------------
 * Inverse Clarke transform   αβ → abc
 *
 *   ia =  α
 *   ib = −α/2 + β·√3/2
 *   ic = −α/2 − β·√3/2
 * -------------------------------------------------------------------------- */
typedef struct InvClarkeInputs  { float alpha; float beta;       } InvClarkeInputs;
typedef struct InvClarkeOutputs { float ia; float ib; float ic; } InvClarkeOutputs;

void inv_clarke_transform_compute(const InvClarkeInputs*  in,
                                   InvClarkeOutputs*       out);

void inv_clarke_transform_compute_flat(const float* in_buf,   /* [alpha, beta]  */
                                        float*       out_buf); /* [ia, ib, ic]   */

/* --------------------------------------------------------------------------
 * Park transform   αβ → dq
 *
 *   d =  α·cosf(θ) + β·sinf(θ)
 *   q = −α·sinf(θ) + β·cosf(θ)
 * -------------------------------------------------------------------------- */
typedef struct ParkInputs  { float alpha; float beta; float theta; } ParkInputs;
typedef struct ParkOutputs { float d; float q;                      } ParkOutputs;

void park_transform_compute(const ParkInputs*  in,
                             ParkOutputs*       out);

void park_transform_compute_flat(const float* in_buf,   /* [alpha, beta, theta] */
                                  float*       out_buf); /* [d, q]               */

/* --------------------------------------------------------------------------
 * Inverse Park transform   dq → αβ
 *
 *   α = d·cosf(θ) − q·sinf(θ)
 *   β = d·sinf(θ) + q·cosf(θ)
 * -------------------------------------------------------------------------- */
typedef struct InvParkInputs  { float d; float q; float theta; } InvParkInputs;
typedef struct InvParkOutputs { float alpha; float beta;        } InvParkOutputs;

void inv_park_transform_compute(const InvParkInputs*  in,
                                 InvParkOutputs*       out);

void inv_park_transform_compute_flat(const float* in_buf,   /* [d, q, theta]    */
                                      float*       out_buf); /* [alpha, beta]    */

#ifdef __cplusplus
}
#endif

#endif /* TRANSFORMS_H */
