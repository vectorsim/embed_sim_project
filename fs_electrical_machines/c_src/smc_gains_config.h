/**********************************************************************************************************************
 * \file      smc_gains_config.h
 * \brief     SMC tunable gain defaults — NANOTEC DB42S02
 *
 * Physics-sized for PMSM_Python_Plant + 20 mN·m load.
 *
 * Design rules:
 *   KS_W ≥ T_load_max / KT  =  0.025 / 0.0084  =  2.976 A  → set 3.0 A
 *   ETA_W ≤ 0.01  (small-signal damping only — large value causes wind-up)
 *   PHI_W = 8.0 rad/s  (boundary layer width)
 *   KS_I  = L·ωc_i  =  125e-6 × 2π×800  =  0.628 V
 *   PHI_I = 0.5 A
 *********************************************************************************************************************/

#ifndef SMC_GAINS_CONFIG_H_
#define SMC_GAINS_CONFIG_H_

#include "embed_sim_matrix.h"

/** Speed SMC switching gain [A] — must be ≥ T_load_max/KT */
#define SMC_KS_W     ((MatrixFloat)3.0f)

/** Speed SMC linear damping [—] — small-signal damping only */
#define SMC_ETA_W    ((MatrixFloat)0.01f)

/** Speed boundary layer [rad/s] */
#define SMC_PHI_W    ((MatrixFloat)8.0f)

/** Current SMC switching gain [V]  =  L·ωc_i */
#define SMC_KS_I     ((MatrixFloat)0.6283f)

/** Current boundary layer [A] */
#define SMC_PHI_I    ((MatrixFloat)0.5f)

#endif /* SMC_GAINS_CONFIG_H_ */
