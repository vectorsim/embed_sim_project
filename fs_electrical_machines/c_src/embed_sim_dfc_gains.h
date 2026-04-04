/*******************************************************************************************************************
 * \file      embed_sim_dfc_gains.h
 * \brief     DFC tunable gain defaults -- NANOTEC DB42S02
 *
 * Differential Flatness Controller gains:
 *   Speed P-loop:  iq_ref = Kp_speed · (ω_ref - ω_m)
 *   Current loop:  vd = ... + Kp_id·(id_ref−id)
 *                 vq = ... + Kp_iq·(iq_ref−iq)
 *
 * Tuned for AURIX TC3xx @ 20 kHz, 17.0 V bus.
 *
 * MISRA C:2012 compliance:
 *   Rule 7.2  : all float literals carry the f suffix.
 *   Rule 20.10: no token-pasting operators used.
 *   Rule 8.1  : all types explicit via MatrixFloat typedef.
 *******************************************************************************************************************/

#ifndef EMBED_SIM_DFC_GAINS_H_
#define EMBED_SIM_DFC_GAINS_H_

#include "embed_sim_matrix.h"

/**
 * \struct DFC_GainSet_T
 * \brief  Runtime-configurable DFC gains.
 */
typedef struct
{
    MatrixFloat kp_speed;   /**< Speed P-gain [A/(rad/s)] — maps speed error to iq_ref */
    MatrixFloat kp_id;      /**< D-axis current P-gain [V/A] */
    MatrixFloat kp_iq;      /**< Q-axis current P-gain [V/A] */
} DFC_GainSet_T;

/**
 * \brief  Speed P-gain [A/(rad/s)].
 *
 * Design: I_MAX / ω_error_max = 3.57 / 30 = 0.119
 * Gives full current at 30 rad/s (286 RPM) error.
 */
#define DFC_KP_SPEED     ((MatrixFloat)0.4f)

/**
 * \brief  D-axis current P-gain [V/A].
 *
 * Design: 2.0 V/A — provides ~0.37 A/step correction at 20 kHz
 * (L_d = 367.5 µH, Δi = Kp·dt/L = 2.0·50e-6/367.5e-6 = 0.272 A/step)
 */
#define DFC_KP_ID        ((MatrixFloat)0.4f)

/**
 * \brief  Q-axis current P-gain [V/A].
 *
 * Same as DFC_KP_ID for symmetry (L_d = L_q for surface-mount PMSM).
 */
#define DFC_KP_IQ        ((MatrixFloat)8.0f)

#endif /* EMBED_SIM_DFC_GAINS_H_ */