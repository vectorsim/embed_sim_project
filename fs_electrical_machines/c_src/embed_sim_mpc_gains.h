/**
 * @file      embed_sim_mpc_gains.h
 * @brief     MPC Controller -- tunable gain defaults for NANOTEC DB42S02
 * @details   Defines MPC weight constants used by MPC_Controller_Step()
 * @version   2.0.0
 * @copyright Copyright (C) EmbedSim 2025
 *
 * @par MISRA C:2012 Compliance:
 *      - Rule 7.2: All float literals have 'f' suffix
 *      - Rule 8.1: All types explicit via MatrixFloat typedef
 *      - Rule 20.10: No token-pasting operators used
 */

#ifndef EMBED_SIM_MPC_GAINS_H_
#define EMBED_SIM_MPC_GAINS_H_

#include "embed_sim_matrix.h"    /* MatrixFloat = real32_T */


/*********************************************************************************************************************/
/*                                 Compile-time MPC weights (MISRA Rule 7.2)                                         */
/*********************************************************************************************************************/

/**
 * @defgroup MPC_Gains Compile-time MPC weight defaults
 * @{
 */

/**
 * @brief   Prediction horizon
 * @details Number of steps into the future the MPC predicts.
 *          At 20 kHz, N=10 gives a 500 µs prediction horizon.
 * @units   dimensionless
 */
#define MPC_N          (10)

/**
 * @brief   d-axis state cost
 * @details Penalises id deviation from 0 (MTPA for SPMSM)
 *          Higher values → faster id convergence but may cause overshoot
 * @units   dimensionless
 */
#define MPC_Q_ID       ((MatrixFloat)10.82f)

/**
 * @brief   q-axis state cost (regulariser)
 * @details Light regularisation on iq. Must be << MPC_Q_OMEGA
 * @units   dimensionless
 */
#define MPC_Q_IQ       ((MatrixFloat)0.01f)

/**
 * @brief   Speed tracking cost
 * @details Dominant weight for speed tracking performance.
 *          FIXED during tuning (not modified by NN tuner)
 * @units   dimensionless
 */
#define MPC_Q_OMEGA    ((MatrixFloat)500.0f)

/**
 * @brief   vd control effort weight
 * @details Penalises large d-axis voltage commands.
 *          Higher values → more conservative vd
 * @units   dimensionless
 */
#define MPC_R_VD       ((MatrixFloat)0.001f)

/**
 * @brief   vq control effort weight
 * @details Penalises large q-axis voltage commands.
 *          Higher values → damps cross-coupling overshoot
 * @units   dimensionless
 */
#define MPC_R_VQ       ((MatrixFloat)0.005f)

/**
 * @brief   Speed error integral gain
 * @details Eliminates steady-state speed offset.
 *          Range: [0.005, 0.1] typical for DB42S02
 * @units   V/(rad/s·s)
 */
#define MPC_KI_V       ((MatrixFloat)0.01f)

/**
 * @brief   Soft-start ramp time
 * @details Time over which iq_limit ramps from 0 to I_MAX [s]
 * @units   s
 */
#define MPC_SOFTSTART_T ((MatrixFloat)0.1f)

/**
 * @brief   Encoder IIR filter coefficient
 * @details Smoothing for encoder speed estimate.
 *          Effective time constant ≈ dt·(1-iir)/iir = 200 µs at 20 kHz
 * @units   dimensionless
 */
#define MPC_ENC_IIR    ((MatrixFloat)0.20f)

/**
 * @brief   SMO switching gain
 * @details Must exceed max back-EMF = ωe_max·λpm = 838·0.0014 = 1.17 V.
 *          4× margin gives robust convergence.
 * @units   V
 */
#define MPC_SMO_K      ((MatrixFloat)4.68f)

/**
 * @brief   SMO back-EMF LPF corner frequency
 * @details Filters the switching residual from the back-EMF estimate.
 * @units   Hz
 */
#define MPC_SMO_FC     ((MatrixFloat)1000.0f)

/** @} */


/*********************************************************************************************************************/
/*                                              Runtime Gain Structure                                               */
/*********************************************************************************************************************/

/**
 * @defgroup MPC_GainSet Runtime MPC weight structure
 * @{
 */

/**
 * @struct  MPC_GainSet_T
 * @brief   Runtime-configurable mirror of MPC weight constants
 * @details Populate this struct to document or inspect which MPC weights were
 *          active during a tuning run.
 *
 *          IMPORTANT — compile-time vs runtime:
 *          MPC_Controller_Step() reads MPC_Q_ID, MPC_Q_IQ, MPC_R_VD, MPC_R_VQ,
 *          MPC_KI_V directly as compile-time #defines (above).  This struct does
 *          NOT override those defines at runtime.
 *
 *          To apply CMA-ES-tuned weights to the C backend:
 *            1. Run the Python tuner  (--tune flag).
 *            2. Tuner writes c_src/embed_sim_mpc_gains.h with new #define values.
 *            3. Recompile:  python setup_mpc_controller.py build_ext --inplace
 *
 *          The Python backend (MPCControllerBlock.compute_py) reads weights from
 *          self.Q_id etc. at every step — no recompile needed there.
 *
 *          PYTHON ALIGNMENT: MPCControllerWrapper.set_gains() / get_gains() in
 *          mpc_controller_wrapper.pyx stores these values for inspection only.
 */
typedef struct
{
    MatrixFloat Q_id;      /**< d-axis state cost - default: 10.82f */
    MatrixFloat Q_iq;      /**< q-axis regulariser - default: 0.01f */
    MatrixFloat R_vd;      /**< vd effort weight - default: 0.001f */
    MatrixFloat R_vq;      /**< vq effort weight - default: 0.005f */
    MatrixFloat KI_v;      /**< Speed integral gain - default: 0.01f */
    MatrixFloat Q_omega;   /**< Speed cost (FIXED) - default: 500.0f */
} MPC_GainSet_T;

/** @} */

#endif /* EMBED_SIM_MPC_GAINS_H_ */