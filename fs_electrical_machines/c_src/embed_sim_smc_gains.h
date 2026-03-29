/**********************************************************************************************************************
 * \file      embed_sim_smc_gains.h
 * \brief     SMC tunable gain defaults — NANOTEC DB42S02
 *
 * Physics-derived for AURIX TC3xx @ 20 kHz, 17V bus, 20 mN·m max load.
 *
 * Architecture: encoder-based FOC + classical equivalent control (no SMO in loop).
 *   Speed loop : integral sliding surface  s = e + λ·∫e
 *                iq_ref = KS_W·sat(s/PHI_W) + ETA_W·s
 *   Current loop: encoder cross-coupling equivalent control
 *                vd_eq = R·id_meas - ωe·Lq·iq_meas
 *                vq_eq = R·iq_meas + ωe·(Ld·id_meas + λpm)
 *                vd = vd_eq + KS_I·sat(s_d/PHI_I)
 *                vq = vq_eq + KS_I·sat(s_q/PHI_I)
 *
 * Gain derivation:
 *   KS_W  ≥ T_load_max/KT = 0.020/0.0084 = 2.381 A  → 3.095 A (+30% margin)
 *   PHI_W = KS_W·e_max / (I_MAX/3) = 3.095·209.4/1.19 = 544.8 rad/s
 *   ETA_W ≤ 0.01  (small-signal damping only)
 *   KS_I  = PHI_I·L/(2·dt) = 0.5·125e-6/(2·50e-6) = 0.625 V  (pole at z=0.5)
 *   PHI_I = 0.5 A   slew = KS_I·dt/L = 0.25 A/step < PHI_I ✓
 *
 * Patched by smc_fmu_tuner.py — do not modify other headers.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_SMC_GAINS_H_
#define EMBED_SIM_SMC_GAINS_H_

#include "embed_sim_matrix.h"

/** \brief Speed SMC switching gain [A].
 *  KS_W ≥ T_load_max/KT = 0.020/0.0084 = 2.381 A → 3.095 A (+30% margin). */
#define SMC_KS_W     ((MatrixFloat)3.095f)

/** \brief Speed SMC linear damping term [—].
 *  Small-signal damping inside boundary layer.  Hard cap of 0.01 enforced
 *  in SMC_SpeedSMC().  Tuner search range: [0.001, 0.01]. */
#define SMC_ETA_W    ((MatrixFloat)0.001f)

/** \brief Speed SMC boundary layer thickness [rad/s].
 *  PHI_W = KS_W·e_max/(I_MAX/3) = 3.095·209.4/1.19 = 545 rad/s.
 *  Sized so iq_ref = I_MAX/3 at maximum ramp error (not saturated).
 *  Tuner search range: [200.0, 800.0] rad/s. */
#define SMC_PHI_W    ((MatrixFloat)545.0f)

/** \brief Current SMC switching gain [V] — physical, before SVPWM normalisation.
 *  Discrete pole placement at z = 0.5:  KS_I = PHI_I·L/(2·dt).
 *  = 0.5 × 125e-6 / (2 × 50e-6) = 0.625 V.  BW = 1592 Hz.
 *  Must exceed R·I_MAX = 0.678 V to reject resistive drop without feedforward.
 *  SMC_Controller_Step() divides all output voltages by SMC_SVPWM_GAIN (= V_DC/2 = 8.5)
 *  before writing y->v_alpha / y->v_beta, so the SVPWM block receives the correct
 *  normalised reference (0.0735 normalised → 0.625 V physical at the plant).
 *  Tuner search range: [0.5, 1.2] V. */
#define SMC_KS_I     ((MatrixFloat)0.625f)

/** \brief Current SMC boundary layer thickness [A].
 *  Slew = KS_I·dt/L = 0.25 A/step < PHI_I = 0.5 A → no overshoot. ✓
 *  Tuner search range: [0.2, 1.0] A. */
#define SMC_PHI_I    ((MatrixFloat)0.5f)

#endif /* EMBED_SIM_SMC_GAINS_H_ */
