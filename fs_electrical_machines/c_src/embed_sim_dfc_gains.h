/**********************************************************************************************************************
 * \file      embed_sim_dfc_gains.h
 * \brief     Differential Flatness Controller -- tunable gain defaults for NANOTEC DB42S02
 *
 * \details   Defines the three compile-time gain constants and the runtime-configurable
 *            DFC_GainSet_T structure used by DFC_Controller_Step().
 *
 *            Speed P-loop:  iq_ref = KP_SPEED * (omega_ref - omega_m)
 *            Current loops: vd    += KP_ID    * (0        - id_meas)
 *                           vq    += KP_IQ    * (iq_ref   - iq_meas)
 *
 *            Tuned for AURIX TC3xx @ 20 kHz, Vdc = 17.0 V bus.
 *
 * \note      MISRA C:2012 compliance
 *              Rule  7.2  : all float literals carry the 'f' suffix.
 *              Rule  8.1  : all types are explicit via the MatrixFloat typedef.
 *              Rule 20.10 : no token-pasting operators used.
 *
 * \version   2.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *********************************************************************************************************************/

#ifndef EMBED_SIM_DFC_GAINS_H_
#define EMBED_SIM_DFC_GAINS_H_

#include "embed_sim_matrix.h"


/**********************************************************************************************************************
 * \brief  Speed proportional gain.
 *
 * \details Design: I_MAX / omega_error_max = 3.57 A / 30 rad/s = 0.119 A/(rad/s).
 *          Saturates current at 30 rad/s (286 RPM) speed error.
 *          Increased to 0.4 after hardware commissioning at 17 V.
 *
 * \units   A / (rad/s)
 *********************************************************************************************************************/
#define DFC_KP_SPEED  ((MatrixFloat)0.4f)

/**********************************************************************************************************************
 * \brief  D-axis current proportional gain.
 *
 * \details Stabilises the id = 0 (MTPA) constraint against cross-coupling.
 *
 * \units   V / A
 *********************************************************************************************************************/
#define DFC_KP_ID     ((MatrixFloat)0.4f)

/**********************************************************************************************************************
 * \brief  Q-axis current proportional gain.
 *
 * \details Corrects residual iq tracking error after the flatness feedforward.
 *          Set high enough to suppress back-EMF estimation lag at 3000 RPM.
 *
 * \units   V / A
 *********************************************************************************************************************/
#define DFC_KP_IQ     ((MatrixFloat)8.0f)


/**********************************************************************************************************************
 * \struct  DFC_GainSet_T
 * \brief   Runtime-configurable gain set for the Differential Flatness Controller.
 *
 * \details Mirrors the three compile-time #define constants above.
 *          Populate and pass to DFC_GainSet_Apply() to update gains without
 *          recompilation (e.g. from a gain-scheduling table or AURIX overlay).
 *********************************************************************************************************************/
typedef struct
{
    MatrixFloat kp_speed;    /**< Speed P-gain [A/(rad/s)]      \see DFC_KP_SPEED */
    MatrixFloat kp_id;       /**< D-axis current P-gain [V/A]   \see DFC_KP_ID    */
    MatrixFloat kp_iq;       /**< Q-axis current P-gain [V/A]   \see DFC_KP_IQ    */
} DFC_GainSet_T;


#endif /* EMBED_SIM_DFC_GAINS_H_ */
