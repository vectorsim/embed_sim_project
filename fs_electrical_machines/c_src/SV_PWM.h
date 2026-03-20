/**********************************************************************************************************************
 * \file      SV_PWM.h
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation interface.
 *
 * Provides sector detection, active-vector time calculation, and duty-cycle
 * generation for symmetric (T0 = T7) centre-aligned SVPWM on a three-phase
 * inverter.  All duty-cycle values are carried internally as Q31 fixed-point
 * quantities; helper functions expose float conversions for monitoring.
 *
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

#ifndef EMBEDSIM_MIDSYS_SVPWM_H_
#define EMBEDSIM_MIDSYS_SVPWM_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Sys_Types.h"
#include "Matrix.h"
#include <math.h>

/* MISRA C 2012 Rule 21.1: math.h is permitted for floating-point utilities */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup svpwm_angle_constants  Angle constants (radians)
 * \{
 */
#ifndef SVM_PI_OVER_6_F
#define SVM_PI_OVER_6_F          (0.5235987756f)   /**< π/6  =  30° */
#endif

#ifndef SVM_PI_OVER_3_F
#define SVM_PI_OVER_3_F          (1.0471975512f)   /**< π/3  =  60° */
#endif

#ifndef SVM_PI_OVER_2_F
#define SVM_PI_OVER_2_F          (1.57079632679f)  /**< π/2  =  90° */
#endif

#ifndef SVM_2PI_OVER_3_F
#define SVM_2PI_OVER_3_F         (2.0943951024f)   /**< 2π/3 = 120° */
#endif

#ifndef SVM_PI_F
#define SVM_PI_F                 (3.14159265359f)  /**< π    = 180° */
#endif

#ifndef SVM_4PI_OVER_3_F
#define SVM_4PI_OVER_3_F         (4.18879020479f)  /**< 4π/3 = 240° */
#endif

#ifndef SVM_5PI_OVER_3_F
#define SVM_5PI_OVER_3_F         (5.23598775598f)  /**< 5π/3 = 300° */
#endif

#ifndef SVM_2PI_F
#define SVM_2PI_F                (6.28318530718f)  /**< 2π   = 360° */
#endif
/** \} */

/** \addtogroup svpwm_math_constants  Mathematical constants
 * \{
 */
#ifndef SVM_SQRT3_F
#define SVM_SQRT3_F              (1.73205080757f)  /**< √3                */
#endif

/** \brief √3 / 2 */
#define SVM_SQRT3_OVER_2_F       (0.86602540378f)
/** \} */

/** \addtogroup svpwm_q31_constants  Q31 fixed-point scaling
 * \{
 */
/** \brief Scaling factor for Q31 ↔ float conversion (2^31). */
#define Q31_SCALE_F              (2147483648.0f)

/** \brief Q31 representation of 1.0 (full-scale positive). */
#define Q31_ONE                  ((int32_T)0x7FFFFFFF)

/** \brief Q31 representation of 0.0. */
#define Q31_ZERO                 ((int32_T)0x00000000)
/** \} */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup svpwm_types  Types and enumerations
 * \{
 */

/**
 * \enum  SVM_Sector_Type
 * \brief Six 60°-wide sectors of the space-vector hexagon.
 *
 * Sector boundaries follow the standard ±30° convention centred on the
 * +α axis:
 *   - Sector I   :   0° – 60°
 *   - Sector II  :  60° – 120°
 *   - Sector III : 120° – 180°
 *   - Sector IV  : 180° – 240°
 *   - Sector V   : 240° – 300°
 *   - Sector VI  : 300° – 360°
 */
typedef enum
{
    SVM_SECTOR_I   = 0U,  /**< Sector I   :   0° –  60° */
    SVM_SECTOR_II  = 1U,  /**< Sector II  :  60° – 120° */
    SVM_SECTOR_III = 2U,  /**< Sector III : 120° – 180° */
    SVM_SECTOR_IV  = 3U,  /**< Sector IV  : 180° – 240° */
    SVM_SECTOR_V   = 4U,  /**< Sector V   : 240° – 300° */
    SVM_SECTOR_VI  = 5U   /**< Sector VI  : 300° – 360° */
} SVM_Sector_Type;

/**
 * \struct SVM_DutyCycle_Type
 * \brief  Per-phase duty cycles and the associated sector, in Q31 format.
 *
 * Q31 encoding:
 *   - #Q31_ONE  (0x7FFFFFFF) → 100 % duty cycle
 *   - #Q31_ZERO (0x00000000) →   0 % duty cycle
 *
 * The duty cycles represent the normalised on-time of the high-side switch
 * in each phase, relative to the full switching period.
 */
typedef struct
{
    MatrixElement    ta;      /**< Phase-A duty cycle (Q31, 0 … Q31_ONE) */
    MatrixElement    tb;      /**< Phase-B duty cycle (Q31, 0 … Q31_ONE) */
    MatrixElement    tc;      /**< Phase-C duty cycle (Q31, 0 … Q31_ONE) */
    SVM_Sector_Type  sector;  /**< Active sector (informational)          */
} SVM_DutyCycle_Type;

/** \} */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup svpwm_api  Public API
 * \{
 */

/**
 * \brief  Calculate SVPWM duty cycles from modulation index and rotor angle.
 *
 * Implements symmetric space-vector modulation (T0 = T7) according to the
 * standard sector switching table.  The modulation index \p modulation_index
 * is the normalised amplitude m' ∈ [0, 1]; \p angle_rad is the electrical
 * angle ωt ∈ [0, 2π).
 *
 * \param[in]  modulation_index  Modulation index m' (float, 0.0 – 1.0).
 * \param[in]  angle_rad         Electrical angle ωt in radians (float, 0 – 2π).
 * \param[out] duty              Output duty-cycle structure (must not be NULL).
 * \return     #MATRIX_SUCCESS on success; #MATRIX_ERROR_NULL_PTR if \p duty is
 *             NULL; #MATRIX_ERROR_OUT_OF_BOUNDS if \p modulation_index is
 *             outside [0, 1].
 */
extern MatrixStatus_Type SVM_CalculateDutyCycle(
    const MatrixFloat          modulation_index,
    const MatrixFloat          angle_rad,
    SVM_DutyCycle_Type * const duty);

/**
 * \brief  Determine the SVPWM sector from dq-frame voltage components.
 *
 * Resolves the active sector directly from the αβ-frame projections of the
 * voltage vector (passed as Q31 values).
 *
 * \param[in]  vd      d-axis voltage (Q31, range [−Q31_ONE, Q31_ONE]).
 * \param[in]  vq      q-axis voltage (Q31, range [−Q31_ONE, Q31_ONE]).
 * \param[out] sector  Resolved sector (must not be NULL).
 * \return     #MATRIX_SUCCESS on success; #MATRIX_ERROR_NULL_PTR if
 *             \p sector is NULL.
 */
extern MatrixStatus_Type SVM_GetSectorFromDQ(
    const MatrixElement        vd,
    const MatrixElement        vq,
    SVM_Sector_Type    * const sector);

/**
 * \brief  Convert Q31 duty cycles to centre-aligned PWM compare values.
 *
 * Maps the normalised duty cycles in \p duty to timer-tick compare values
 * suitable for an up-down (centre-aligned) counter with period
 * \p timer_period ticks.
 *
 * \param[in]  duty         Source duty-cycle structure (must not be NULL).
 * \param[in]  timer_period Timer period in ticks (must be > 0).
 * \param[out] compare_a    Compare value for phase A (0 … timer_period).
 * \param[out] compare_b    Compare value for phase B (0 … timer_period).
 * \param[out] compare_c    Compare value for phase C (0 … timer_period).
 * \return     #MATRIX_SUCCESS on success; #MATRIX_ERROR_NULL_PTR if any
 *             pointer is NULL; #MATRIX_ERROR_DIV_BY_ZERO if
 *             \p timer_period is zero.
 */
extern MatrixStatus_Type SVM_GetCompareValues(
    const SVM_DutyCycle_Type * const duty,
    const uint32_T                   timer_period,
    uint32_T               * const   compare_a,
    uint32_T               * const   compare_b,
    uint32_T               * const   compare_c);

/**
 * \brief  Read back phase duty cycles as floating-point values.
 *
 * Utility function for monitoring and debugging; converts the Q31 fields of
 * \p duty back to normalised floats in [0.0, 1.0].  Silently does nothing if
 * any pointer is NULL.
 *
 * \param[in]  duty  Source duty-cycle structure.
 * \param[out] ta    Phase-A duty cycle (float, 0.0 – 1.0).
 * \param[out] tb    Phase-B duty cycle (float, 0.0 – 1.0).
 * \param[out] tc    Phase-C duty cycle (float, 0.0 – 1.0).
 */
extern void SVM_GetDutyCyclesFloat(
    const SVM_DutyCycle_Type * const duty,
    MatrixFloat              * const ta,
    MatrixFloat              * const tb,
    MatrixFloat              * const tc);

/** \} */

#endif /* EMBEDSIM_MIDSYS_SVPWM_H_ */
