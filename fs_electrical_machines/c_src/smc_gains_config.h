/**********************************************************************************************************************
 * \file      smc_gains_config.h
 * \brief     SMC tunable gain defaults — NANOTEC DB42S02
 *
 * Physics-sized for PMSM_Python_Plant + 20 mN·m load.
 *
 * Architecture: pure SMC with Sliding Mode Observer (SMO).
 *   - Speed loop : integral sliding surface  s = e + λ·∫e + γ·∫∫e
 *                  iq_ref = KS_W·sat(s/PHI_W) + ETA_W·s
 *   - Current loop: SMO back-EMF as equivalent control (no PI, no cross-coupling)
 *                  vd = ed_hat + KS_I·sat(s_d/PHI_I)
 *                  vq = eq_hat + KS_I·sat(s_q/PHI_I)
 *                  (ed_hat, eq_hat) = Park(ê_α_filt, ê_β_filt) from SMO
 *
 * Design rules:
 *   KS_W  ≥ T_load_max / KT  =  0.025 / 0.0084  =  2.976 A  → set 3.0 A
 *   ETA_W ≤ 0.01  (small-signal damping only — controller enforces this cap)
 *   PHI_W = 545.0 rad/s  (boundary layer — matched to Python backend)
 *   KS_I  = L·ωc_i  =  125e-6 × 2π×800  =  0.628 V
 *             Pure switching amplitude on top of SMO equivalent control.
 *             Smaller than a PI-based loop needs — SMO handles steady-state.
 *   PHI_I = 0.5 A   (boundary layer — smooths switching at low current error)
 *
 * SMO parameters (fixed — not tunable, defined in embed_sim_smc_controller.h):
 *   k     = 1.5 · V_MAX ≈ 14.72 V   (switching gain)
 *   fc    = 500 Hz                    (back-EMF LPF cutoff)
 *   alpha = 0.13588                   (LPF coefficient at 20 kHz)
 *********************************************************************************************************************/

#ifndef SMC_GAINS_CONFIG_H_
#define SMC_GAINS_CONFIG_H_

#include "embed_sim_matrix.h"

/** \brief Speed SMC switching gain [A].
 *  Must be ≥ T_load_max/KT = 0.025/0.0084 ≈ 2.976 A to produce rated torque
 *  at steady state (sat = 1 at the surface boundary).
 *  Tuner search range: [2.5, 5.0] A. */
#define SMC_KS_W     ((MatrixFloat)3.0f)

/** \brief Speed SMC linear damping term [—].
 *  Provides small-signal damping inside the boundary layer.
 *  Controller enforces a hard cap of 0.01 — values above this cause
 *  integrator wind-up in the double-integral sliding surface.
 *  Tuner search range: [0.001, 0.01]. */
#define SMC_ETA_W    ((MatrixFloat)0.01f)

/** \brief Speed SMC boundary layer thickness [rad/s].
 *  Wider → less chattering, slower transient.
 *  Narrower → more chattering, faster response.
 *  Value matched to Python backend: PHI_W = 545.0 rad/s.
 *  (Previous value of 8.0 rad/s caused full bang-bang chattering and
 *  ~10 000 RPM oscillation when use_c_backend=True.) */
#define SMC_PHI_W    ((MatrixFloat)545.0f)

/** \brief Current SMC switching gain [V].
 *  Nominal value = L·ωc_i = 125e-6 × 2π×800 ≈ 0.628 V.
 *  Acts as a pure switching correction on top of the SMO equivalent control
 *  (ed_hat, eq_hat).  No PI integrator — the observer handles steady-state.
 *  Tuner search range: [0.3, 1.2] V. */
#define SMC_KS_I     ((MatrixFloat)0.6283f)

/** \brief Current SMC boundary layer thickness [A].
 *  Smooths the switching term inside the current error band.
 *  Tuner search range: [0.1, 1.0] A. */
#define SMC_PHI_I    ((MatrixFloat)0.5f)

#endif /* SMC_GAINS_CONFIG_H_ */
