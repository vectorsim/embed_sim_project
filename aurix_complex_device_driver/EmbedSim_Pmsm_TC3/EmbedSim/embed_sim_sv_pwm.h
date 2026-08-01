/**********************************************************************************************************************
 * \file      embed_sim_sv_pwm.h
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation interface.
 *            Matrix-based implementation using coordinate transforms.
 *
 * \details   Consumes FocAlphaBeta_T (Valpha, Vbeta) and FocAngle_T (ThetaE)
 *            and produces SVM_DutyCycle_T (Ta, Tb, Tc).
 *
 *            Uses the matrix-based transforms from embed_sim_coordinate_transform.h
 *            for coordinate conversions.
 *
 * \version   2.1.0
 * \date      2025-05-24
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_SV_PWM_H_
#define EMBED_SIM_SV_PWM_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "embed_sim_foc_types.h"
#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"

/**********************************************************************************************************************
 * Macros — Angle Constants  [rad]
 *
 * Thin aliases over ES_MATH_ central definitions in embed_sim_sys_types.h.
 * Existing callers using SVM_PI_F etc. continue to compile without change.
 *********************************************************************************************************************/

/** \brief  π/6  =  30°   [rad] */
#define SVM_PI_OVER_6_F          ES_MATH_PI_OVER_6_F

/** \brief  π/3  =  60°   [rad] */
#define SVM_PI_OVER_3_F          ES_MATH_PI_OVER_3_F

/** \brief  π/2  =  90°   [rad] */
#define SVM_PI_OVER_2_F          ES_MATH_PI_OVER_2_F

/** \brief  2π/3 = 120°   [rad] */
#define SVM_2PI_OVER_3_F         ES_MATH_2PI_OVER_3_F

/** \brief  π    = 180°   [rad] */
#define SVM_PI_F                 ES_MATH_PI_F

/** \brief  4π/3 = 240°   [rad] */
#define SVM_4PI_OVER_3_F         ES_MATH_4PI_OVER_3_F

/** \brief  5π/3 = 300°   [rad] */
#define SVM_5PI_OVER_3_F         ES_MATH_5PI_OVER_3_F

/** \brief  2π   = 360°   [rad] */
#define SVM_2PI_F                ES_MATH_2PI_F

/**********************************************************************************************************************
 * Macros — Mathematical Constants  [dimensionless]
 *
 * Thin aliases over ES_MATH_ central definitions in embed_sim_sys_types.h.
 *********************************************************************************************************************/

/** \brief  √3              [dimensionless] */
#define SVM_SQRT3_F              ES_MATH_SQRT3_F

/** \brief  √3 / 2          [dimensionless] */
#define SVM_SQRT3_OVER_2_F       ES_MATH_HALF_SQRT3_F

/** \brief  2/√3            [dimensionless] */
#define SVM_TWO_OVER_SQRT3_F     ES_MATH_TWO_INV_SQRT3_F

/** \brief  2/3             [dimensionless] */
#define SVM_TWO_THIRDS_F         ES_MATH_TWO_THIRDS_F

/** \brief  1/3             [dimensionless] */
#define SVM_ONE_THIRD_F          ES_MATH_ONE_THIRD_F

/**********************************************************************************************************************
 * Macros — Q31 Fixed-Point Scaling  (DEPRECATED — retained as zero/one aliases only)
 *
 * MatrixElement is now real32_T.  These macros remain so that any existing
 * call sites that reference SVM_Q31_ONE / SVM_Q31_ZERO still compile, but
 * they now resolve to float 1.0f / 0.0f rather than integer bit patterns.
 *********************************************************************************************************************/

/** \brief  Duty-cycle value representing 100 % (was Q31 0x7FFFFFFF, now 1.0f) */
#define SVM_Q31_ONE              (ES_MATH_ONE_F)

/** \brief  Duty-cycle value representing   0 % (was Q31 0x00000000, now 0.0f) */
#define SVM_Q31_ZERO             (0.0f)

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \enum   SVM_Sector_T
 * \brief  Six 60°-wide sectors of the space-vector hexagon.
 */
typedef enum
{
    SVM_SECTOR_I   = 0U,   /**< Sector I   :   0° –  60°   [dimensionless] */
    SVM_SECTOR_II  = 1U,   /**< Sector II  :  60° – 120°   [dimensionless] */
    SVM_SECTOR_III = 2U,   /**< Sector III : 120° – 180°   [dimensionless] */
    SVM_SECTOR_IV  = 3U,   /**< Sector IV  : 180° – 240°   [dimensionless] */
    SVM_SECTOR_V   = 4U,   /**< Sector V   : 240° – 300°   [dimensionless] */
    SVM_SECTOR_VI  = 5U    /**< Sector VI  : 300° – 360°   [dimensionless] */
} SVM_Sector_T;

/**
 * \struct SVM_DutyCycle_T
 * \brief  Per-phase duty cycles and the active sector.
 */
typedef struct
{
    MatrixFloat    Ta;      /**< Phase-A (U) duty cycle   [0.0 … 1.0] */
    MatrixFloat    Tb;      /**< Phase-B (V) duty cycle   [0.0 … 1.0] */
    MatrixFloat    Tc;      /**< Phase-C (W) duty cycle   [0.0 … 1.0] */
    SVM_Sector_T   Sector;  /**< Active sector            [SVM_Sector_T]   */
} SVM_DutyCycle_T;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialize SVPWM module (call once at startup)
 *
 * \return  void
 */
extern void SVM_Init(void);

/**
 * \brief   Calculate SVPWM duty cycles from αβ voltage vector and rotor angle.
 *
 * \details Uses Clarke transform to convert duty cycles back to UVW.
 *          This is an alternative approach using the matrix transforms.
 *
 * \param[in]  V_AlphaBeta  αβ voltage vector (must not be NULL)
 * \param[in]  Angle_P      Electrical rotor angle struct (must not be NULL)
 * \param[out] DutyOut_P    Output duty-cycle structure (must not be NULL)
 *
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if any pointer is NULL.
 */
extern MatrixStatus_T SVM_CalculateDutyCycleFromAlphaBeta(
    const FocAlphaBeta_T * const V_AlphaBeta_P,
    const FocAngle_T     * const Angle_P,
    SVM_DutyCycle_T      * const DutyOut_P);

/**
 * \brief   Calculate SVPWM duty cycles from modulation index and angle.
 *
 * \details Original method using modulation index directly.
 *
 * \param[in]  ModIndex   Normalised modulation index    [0.0 – 1.0]
 * \param[in]  Angle_P    Electrical rotor angle struct  (must not be NULL)
 * \param[out] DutyOut_P  Output duty-cycle structure    (must not be NULL)
 *
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if any pointer is NULL.
 *          MATRIX_ERROR_OUT_OF_BOUNDS if ModIndex is outside [0.0, 1.0].
 */
extern MatrixStatus_T SVM_CalculateDutyCycle(
    MatrixFloat                    ModIndex,
    const FocAngle_T     * const   Angle_P,
    SVM_DutyCycle_T      * const   DutyOut_P);

/**
 * \brief   Calculate SVPWM duty cycles from dq voltage vector.
 *
 * \details Transforms dq → αβ using Inverse-Park, then calculates duty cycles.
 *
 * \param[in]  V_Dq         dq voltage vector (must not be NULL)
 * \param[in]  Angle_P      Electrical rotor angle struct (must not be NULL)
 * \param[out] DutyOut_P    Output duty-cycle structure (must not be NULL)
 *
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if any pointer is NULL.
 */
extern MatrixStatus_T SVM_CalculateDutyCycleFromDq(
    const FocDq_T        * const V_Dq_P,
    const FocAngle_T     * const Angle_P,
    SVM_DutyCycle_T      * const DutyOut_P);

/**
 * \brief   Determine the SVPWM sector from αβ voltage components.
 *
 * \param[in]  V_AlphaBeta  αβ voltage vector (must not be NULL)
 * \param[out] SectorOut_P  Resolved sector (must not be NULL)
 *
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if any pointer is NULL.
 */
extern MatrixStatus_T SVM_GetSectorFromAlphaBeta(
    const FocAlphaBeta_T * const V_AlphaBeta_P,
    SVM_Sector_T         * const SectorOut_P);

/**
 * \brief   Determine the SVPWM sector from dq-frame voltage components.
 *
 * \param[in]  Vd          d-axis voltage   [real32_T]
 * \param[in]  Vq          q-axis voltage   [real32_T]
 * \param[out] SectorOut_P Resolved sector  (must not be NULL)
 *
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if SectorOut_P is NULL.
 */
extern MatrixStatus_T SVM_GetSectorFromDQ(
    MatrixElement              Vd,
    MatrixElement              Vq,
    SVM_Sector_T     * const   SectorOut_P);

/**
 * \brief   Convert floating-point duty cycles to centre-aligned PWM compare values.
 *
 * \param[in]  DutyIn_P    Source duty-cycle structure   (must not be NULL)
 * \param[in]  TimerPeriod Timer period in CLK ticks     (must be > 0)
 * \param[out] CompAOut_P  Compare value for phase U     [0 … TimerPeriod]
 * \param[out] CompBOut_P  Compare value for phase V     [0 … TimerPeriod]
 * \param[out] CompCOut_P  Compare value for phase W     [0 … TimerPeriod]
 *
 * \return  MATRIX_SUCCESS on success.
 *          MATRIX_ERROR_NULL_PTR if any pointer is NULL.
 *          MATRIX_ERROR_DIV_BY_ZERO if TimerPeriod is zero.
 */
extern MatrixStatus_T SVM_GetCompareValues(
    const SVM_DutyCycle_T  * const DutyIn_P,
    uint32_T                       TimerPeriod,
    uint32_T               * const CompAOut_P,
    uint32_T               * const CompBOut_P,
    uint32_T               * const CompCOut_P);

/**
 * \brief   Read back per-phase duty cycles as floating-point values.
 *
 * \param[in]  DutyIn_P  Source duty-cycle structure
 * \param[out] TaOut_P   Phase-U duty cycle   [0.0 – 1.0]
 * \param[out] TbOut_P   Phase-V duty cycle   [0.0 – 1.0]
 * \param[out] TcOut_P   Phase-W duty cycle   [0.0 – 1.0]
 *
 * \return  void
 */
extern void SVM_GetDutyCyclesFloat(
    const SVM_DutyCycle_T  * const DutyIn_P,
    MatrixFloat            * const TaOut_P,
    MatrixFloat            * const TbOut_P,
    MatrixFloat            * const TcOut_P);

#endif /* EMBED_SIM_SV_PWM_H_ */
