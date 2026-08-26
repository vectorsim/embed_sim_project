/**********************************************************************************************************************
 * \file      embed_sim_sv_pwm.h
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation interface.
 *            Matrix-based implementation using coordinate transforms.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCase_P
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.0.0
 * \date      2026-08-27
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_SV_PWM_H_
#define EMBED_SIM_SV_PWM_H_

#include "embed_sim_foc_types.h"
#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"

/**********************************************************************************************************************
 * Macros — Angle Constants
 *********************************************************************************************************************/
#define SVM_PI_OVER_6_F          ES_MATH_PI_OVER_6_F          /**< π/6 rad */
#define SVM_PI_OVER_3_F          ES_MATH_PI_OVER_3_F          /**< π/3 rad */
#define SVM_PI_OVER_2_F          ES_MATH_PI_OVER_2_F          /**< π/2 rad */
#define SVM_2PI_OVER_3_F         ES_MATH_2PI_OVER_3_F         /**< 2π/3 rad */
#define SVM_PI_F                 ES_MATH_PI_F                 /**< π rad */
#define SVM_4PI_OVER_3_F         ES_MATH_4PI_OVER_3_F         /**< 4π/3 rad */
#define SVM_5PI_OVER_3_F         ES_MATH_5PI_OVER_3_F         /**< 5π/3 rad */
#define SVM_2PI_F                ES_MATH_2PI_F                /**< 2π rad */

/**********************************************************************************************************************
 * Macros — Mathematical Constants
 *********************************************************************************************************************/
#define SVM_SQRT3_F              ES_MATH_SQRT3_F              /**< √3 */
#define SVM_SQRT3_OVER_2_F       ES_MATH_HALF_SQRT3_F         /**< √3/2 */
#define SVM_TWO_OVER_SQRT3_F     ES_MATH_TWO_INV_SQRT3_F      /**< 2/√3 */
#define SVM_TWO_THIRDS_F         ES_MATH_TWO_THIRDS_F         /**< 2/3 */
#define SVM_ONE_THIRD_F          ES_MATH_ONE_THIRD_F          /**< 1/3 */

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \brief   Space Vector PWM sector enumeration.
 */
typedef enum
{
    SVM_SECTOR_I   = 0U,    /**< Sector I   (0°  to 60°)  */
    SVM_SECTOR_II  = 1U,    /**< Sector II  (60° to 120°) */
    SVM_SECTOR_III = 2U,    /**< Sector III (120° to 180°) */
    SVM_SECTOR_IV  = 3U,    /**< Sector IV  (180° to 240°) */
    SVM_SECTOR_V   = 4U,    /**< Sector V   (240° to 300°) */
    SVM_SECTOR_VI  = 5U     /**< Sector VI  (300° to 360°) */
} SVM_Sector_T;

/**
 * \brief   SVPWM duty cycle output structure.
 */
typedef struct
{
    MatrixFloat    Ta;      /**< Phase-A (U) duty cycle   [0.0 … 1.0] */
    MatrixFloat    Tb;      /**< Phase-B (V) duty cycle   [0.0 … 1.0] */
    MatrixFloat    Tc;      /**< Phase-C (W) duty cycle   [0.0 … 1.0] */
    SVM_Sector_T   Sector;  /**< Active sector            [SVM_Sector_T] */
} SVM_DutyCycle_T;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialize SVPWM module.
 *
 * \details Must be called once at system startup before any other SVPWM functions.
 *
 * \return  void
 */
extern void SVM_Init(void);

/**
 * \brief   Calculate SVPWM duty cycles from modulation index and electrical angle.
 *
 * \details This is the direct method where modulation index is in range [0.0, 1.0],
 *          where 1.0 corresponds to maximum linear SVPWM output (Vdc/√3 phase voltage).
 *
 * \param[in]  ModIndex     Modulation index [0.0 … 1.0]
 * \param[in]  AnglePtr     Pointer to electrical angle structure
 * \param[out] DutyOutPtr   Pointer to duty cycle output structure
 *
 * \return  MATRIX_SUCCESS on success, error code otherwise
 */
extern MatrixStatus_T SVM_CalculateDutyCycle(
    MatrixFloat                    ModIndex,
    const FocAngle_T     * const   AnglePtr,
    SVM_DutyCycle_T      * const   DutyOutPtr);

/**
 * \brief   Calculate SVPWM duty cycles from αβ voltage vector.
 *
 * \details This function computes the modulation index from the αβ voltage magnitude
 *          and DC bus voltage, then generates SVPWM duty cycles using the provided
 *          electrical angle.
 *
 * \param[in]  VAlphaBetaPtr  Pointer to αβ voltage vector
 * \param[in]  AnglePtr       Pointer to electrical angle structure
 * \param[in]  Vdc            DC bus voltage [V]
 * \param[out] DutyOutPtr     Pointer to duty cycle output structure
 *
 * \return  MATRIX_SUCCESS on success, error code otherwise
 */
extern MatrixStatus_T SVM_CalculateDutyCycleFromAlphaBeta(
    const FocAlphaBeta_T * const VAlphaBetaPtr,
    const FocAngle_T     * const AnglePtr,
    MatrixFloat                  Vdc,
    SVM_DutyCycle_T      * const DutyOutPtr);

/**
 * \brief   Calculate SVPWM duty cycles from dq voltage vector.
 *
 * \details This is the primary function for FOC applications with Id=0 control.
 *          It transforms dq voltages to αβ using inverse Park transform,
 *          then generates SVPWM duty cycles.
 *
 * \param[in]  VDqPtr       Pointer to dq voltage vector [V]
 * \param[in]  AnglePtr     Pointer to electrical angle structure
 * \param[in]  Vdc          DC bus voltage [V]
 * \param[out] DutyOutPtr   Pointer to duty cycle output structure
 *
 * \return  MATRIX_SUCCESS on success, error code otherwise
 */
extern MatrixStatus_T SVM_CalculateDutyCycleFromDq(
    const FocDq_T        * const VDqPtr,
    const FocAngle_T     * const AnglePtr,
    MatrixFloat                  Vdc,
    SVM_DutyCycle_T      * const DutyOutPtr);

/**
 * \brief   Convert floating-point duty cycles to centre-aligned PWM compare values.
 *
 * \param[in]  DutyInPtr      Pointer to duty cycle input structure [0.0 … 1.0]
 * \param[in]  TimerPeriod    Timer period in ticks
 * \param[out] CompAOutPtr    Pointer to compare value for phase A
 * \param[out] CompBOutPtr    Pointer to compare value for phase B
 * \param[out] CompCOutPtr    Pointer to compare value for phase C
 *
 * \return  MATRIX_SUCCESS on success, error code otherwise
 */
extern MatrixStatus_T SVM_GetCompareValues(
    const SVM_DutyCycle_T  * const DutyInPtr,
    uint32_T                       TimerPeriod,
    uint32_T               * const CompAOutPtr,
    uint32_T               * const CompBOutPtr,
    uint32_T               * const CompCOutPtr);

#endif /* EMBED_SIM_SV_PWM_H_ */
