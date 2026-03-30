/*******************************************************************************************************************
 * \file      embed_sim_smc_gains.h
 * \brief     SMC tunable gain defaults -- NANOTEC DB42S02
 *
 * Physics-derived / auto-tuned for AURIX TC3xx @ 20 kHz, 17.0 V bus,
 * 20 mN.m max load (T_LOAD_HEAVY).
 *
 * Architecture: encoder-based FOC + classical equivalent control (no SMO in loop).
 *   Speed loop : integral sliding surface  s = e + lambda*integral(e)
 *                iq_ref = KS_W*sat(s/PHI_W) + ETA_W*s
 *   Current loop: equivalent control (full plant ODE cancellation)
 *                ed_hat = R*id_meas - we*Lq*iq_meas
 *                eq_hat = R*iq_meas + we*(Ld*id_meas + lambda_pm)
 *                vd = ed_hat + KS_I*sat(s_d/PHI_I)
 *                vq = eq_hat + KS_I*sat(s_q/PHI_I)
 *
 * Gain derivation (tuned values):
 *   KS_W  >= T_load_max/KT = 0.020/0.0084 = 2.381 A
 *           --> 5.554994 A  (x2.33 margin)
 *   PHI_W   tuned by Latin-Hypercube search
 *           --> 279.005762 rad/s
 *   KS_I    discrete z-pole at z = 0.9765
 *           Current slew = KS_I*dt/L = 0.0235 A/step  < PHI_I --> no overshoot
 *           --> 0.058730 V  (physical; divided by V_DC/2 = 8.5 inside SMC_Controller_Step)
 *   PHI_I   current boundary layer
 *           --> 0.277341 A
 *
 * Written by db42s02_closed_loop_smc_foc_20k.py -- do not edit manually.
 * Recompile embed_sim_smc_controller.c after patching this file.
 *
 * MISRA C:2012 compliance:
 *   Rule 7.2  : all float literals carry the f suffix (no implicit double promotion).
 *   Rule 20.10: no token-pasting operators used.
 *   Rule 8.1  : all types explicit via MatrixFloat typedef (= real32_T).
 *******************************************************************************************************************/

#ifndef EMBED_SIM_SMC_GAINS_H_
#define EMBED_SIM_SMC_GAINS_H_

#include "embed_sim_matrix.h"

/** \brief Speed SMC switching amplitude [A].
 *
 *  Minimum condition for load rejection (Utkin 1992 s5.3):
 *    KS_W >= T_load_max / KT = 0.020 N.m / 0.0084 N.m/A = 2.381 A
 *  Tuned value: 5.554994 A  (x2.33 margin).
 *
 *  Units  : A  (q-axis current amplitude)
 *  Range  : [1.667, 8.333] A
 *  Tuned  : Latin-Hypercube search (cost = W_SS*ss_err + W_BUMP*bump + W_ID*id_rms + W_CHAT*chat) */
#define SMC_KS_W     ((MatrixFloat)5.554994f)

/** \brief Speed SMC linear damping inside the boundary layer [dimensionless].
 *
 *  Provides smooth proportional action for |s| < PHI_W.
 *  Hard-capped at 0.01 inside SMC_SpeedSMC() -- not a tuning target.
 *  Setting this above 0.01 has no effect.
 *
 *  Units  : dimensionless
 *  Range  : [0.001, 0.010] */
#define SMC_ETA_W    ((MatrixFloat)0.001000f)

/** \brief Speed SMC boundary layer thickness [rad/s].
 *
 *  Transition region between switching (bang-bang) and proportional control:
 *    |s| > PHI_W  -->  bang-bang: iq_ref = +/- KS_W
 *    |s| < PHI_W  -->  linear:    iq_ref = KS_W*(s/PHI_W) + ETA_W*s
 *  Larger PHI_W reduces chattering but widens the speed dead-band.
 *  Tuned value: 279.005762 rad/s.
 *
 *  Units  : rad/s  (mechanical speed error)
 *  Range  : [150.0, 1000.0] rad/s
 *  Tuned  : Latin-Hypercube search */
#define SMC_PHI_W    ((MatrixFloat)279.005762f)

/** \brief Current SMC switching gain [V] -- physical, before SVPWM normalisation.
 *
 *  SMC_Controller_Step() divides all output voltages by SMC_SVPWM_GAIN = V_DC/2 = 8.5
 *  before writing y->v_alpha / y->v_beta, so the SVPWM block receives a normalised
 *  reference in [-1, +1] and the plant sees the correct physical voltages.
 *
 *  Discrete pole placement (Krishnan PMSM Drives, Ch.4):
 *    z-pole = 1 - KS_I*dt/L = 1 - 0.058730*5.00e-05/1.250e-04 = 0.9765
 *  Current slew per sample:
 *    slew = KS_I*dt/L = 0.0235 A/step   (< PHI_I --> no overshoot)
 *
 *  Units  : V  (physical phase voltage, pre-SVPWM)
 *  Range  : [0.3392, 4.9075] V
 *  Tuned  : Latin-Hypercube search */
#define SMC_KS_I     ((MatrixFloat)0.058730f)

/** \brief Current SMC boundary layer thickness [A].
 *
 *  Controls smooth vs. switching behaviour of the d- and q-axis current loops.
 *  Stability condition: slew = KS_I*dt/L < PHI_I (no inter-sample overshoot).
 *    slew = 0.0235 A/step   PHI_I = 0.2773 A   --> < PHI_I --> no overshoot
 *
 *  Units  : A  (dq current error)
 *  Range  : [0.100, 1.500] A
 *  Tuned  : Latin-Hypercube search */
#define SMC_PHI_I    ((MatrixFloat)0.277341f)

#endif /* EMBED_SIM_SMC_GAINS_H_ */
