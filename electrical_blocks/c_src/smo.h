/**
 * @file    smo.h
 * @brief   Sliding Mode Observer (SMO) for PMSM sensorless FOC — EmbedSim block
 *
 * Utkin-type SMO operating in the αβ stationary frame.
 *
 * States (observer)
 * -----------------
 *   î_α, î_β      : estimated stator currents [A]
 *
 * Internal low-pass filter states
 * --------------------------------
 *   ê_α, ê_β      : filtered back-EMF estimates [V]
 *   ω̂_e_filt      : filtered electrical angular velocity [rad/s]
 *
 * Outputs  (SMO_Block_T.y[])
 * --------------------------
 *   y[0] = θ̂_e    : estimated electrical angle [rad]  (-π .. π)
 *   y[1] = ω̂_m    : estimated mechanical speed [rad/s]
 *   y[2] = î_d    : estimated d-axis current [A]
 *   y[3] = î_q    : estimated q-axis current [A]
 *
 * Algorithm
 * ---------
 *   Current observer (Euler, step dt):
 *     dî_α/dt = (v_α - R·î_α + z_α) / L
 *     dî_β/dt = (v_β - R·î_β + z_β) / L
 *
 *   Sliding injection (sat replaces sign → boundary-layer chattering reduction):
 *     z_α = K_smo · sat(ĩ_α / φ)
 *     z_β = K_smo · sat(ĩ_β / φ)
 *
 *   Back-EMF LPF (first-order, wc_emf rad/s):
 *     ê_α += wc_emf · (z_α - ê_α) · dt
 *     ê_β += wc_emf · (z_β - ê_β) · dt
 *
 *   Angle:
 *     θ̂_e = atan2f(-ê_α, ê_β)
 *
 *   Speed (differentiate θ̂_e + LPF at wc_spd rad/s, then scale):
 *     Δθ = θ̂_e - θ̂_e_prev  (π-unwrapped)
 *     ω̂_e_raw = Δθ / dt
 *     ω̂_e_filt += wc_spd · (ω̂_e_raw - ω̂_e_filt) · dt
 *     ω̂_m = ω̂_e_filt / p
 *
 *   dq estimate (Park from αβ):
 *     î_d =  î_α · cos(θ̂_e) + î_β · sin(θ̂_e)
 *     î_q = -î_α · sin(θ̂_e) + î_β · cos(θ̂_e)
 *
 * MISRA C:2012 compliant.
 * Target : Infineon AURIX TC3xx  (TASKING vx compiler)
 * Safety : ASIL-D compatible
 *
 * @author  EmbedSim Framework
 */

#ifndef SMO_H
#define SMO_H

#include "Sys_Types.h"

/* --------------------------------------------------------------------------
 * Parameters structure  (initialised once, read-only during run)
 * -------------------------------------------------------------------------- */
typedef struct
{
    real32_T R;        /**< Stator resistance [Ω]                          */
    real32_T L;        /**< Average inductance (Ld+Lq)/2 [H]              */
    real32_T K_smo;    /**< Sliding gain [V]                               */
    real32_T wc_emf;   /**< Back-EMF LPF corner frequency [rad/s]         */
    real32_T wc_spd;   /**< Speed LPF corner frequency [rad/s]            */
    real32_T phi;      /**< Boundary-layer width for sat() [A]            */
    real32_T p;        /**< Number of pole pairs                          */
} SMO_Params_T;

/* --------------------------------------------------------------------------
 * State structure
 * -------------------------------------------------------------------------- */
typedef struct
{
    /* Parameters (copied at init, fixed during run) */
    SMO_Params_T prm;

    /* Observer states */
    real32_T i_alpha_hat;   /**< Estimated i_α [A]                        */
    real32_T i_beta_hat;    /**< Estimated i_β [A]                        */

    /* Back-EMF LPF states */
    real32_T e_alpha_hat;   /**< Filtered back-EMF α [V]                  */
    real32_T e_beta_hat;    /**< Filtered back-EMF β [V]                  */

    /* Angle + speed tracking */
    real32_T theta_e_hat;   /**< Estimated electrical angle [rad]         */
    real32_T theta_e_prev;  /**< Previous step angle (for diff) [rad]     */
    real32_T omega_e_filt;  /**< Filtered electrical speed [rad/s]        */

    /* Outputs */
    real32_T y[4];          /**< [θ̂_e, ω̂_m, î_d, î_q]                   */
} SMO_Block_T;

/* --------------------------------------------------------------------------
 * Public API
 * -------------------------------------------------------------------------- */

/**
 * @brief  Initialise SMO block with motor parameters.
 * @param  blk  Pointer to SMO state structure.
 * @param  prm  Pointer to parameter structure (copied into blk).
 */
extern void SMO_Init(SMO_Block_T * const blk, const SMO_Params_T * const prm);

/**
 * @brief  Execute one SMO step.
 *
 * @param  blk       Pointer to SMO state.
 * @param  dt        Timestep [s].
 * @param  i_alpha   Measured stator current α [A].
 * @param  i_beta    Measured stator current β [A].
 * @param  v_alpha   Applied stator voltage α [V]  (from previous step — z⁻¹).
 * @param  v_beta    Applied stator voltage β [V].
 * @param  y         Output array[4]: [θ̂_e, ω̂_m, î_d, î_q].
 */
extern void SMO_Compute(
    SMO_Block_T * const blk,
    real32_T dt,
    real32_T i_alpha,
    real32_T i_beta,
    real32_T v_alpha,
    real32_T v_beta,
    real32_T * const y
);

#endif /* SMO_H */
