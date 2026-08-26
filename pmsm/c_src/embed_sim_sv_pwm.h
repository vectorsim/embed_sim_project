/**********************************************************************************************************************
 * \file      embed_sim_sv_pwm.h
 * \brief     Space Vector PWM (SVPWM) duty-cycle calculation interface.
 *            Matrix-based implementation using coordinate transforms.
 *********************************************************************************************************************/

#ifndef EMBED_SIM_SV_PWM_H_
#define EMBED_SIM_SV_PWM_H_

#include "embed_sim_foc_types.h"
#include "embed_sim_matrix.h"
#include "embed_sim_coordinate_transform.h"

/**********************************************************************************************************************
 * Macros — Angle Constants
 *********************************************************************************************************************/
#define SVM_PI_OVER_6_F          ES_MATH_PI_OVER_6_F
#define SVM_PI_OVER_3_F          ES_MATH_PI_OVER_3_F
#define SVM_PI_OVER_2_F          ES_MATH_PI_OVER_2_F
#define SVM_2PI_OVER_3_F         ES_MATH_2PI_OVER_3_F
#define SVM_PI_F                 ES_MATH_PI_F
#define SVM_4PI_OVER_3_F         ES_MATH_4PI_OVER_3_F
#define SVM_5PI_OVER_3_F         ES_MATH_5PI_OVER_3_F
#define SVM_2PI_F                ES_MATH_2PI_F

/**********************************************************************************************************************
 * Macros — Mathematical Constants
 *********************************************************************************************************************/
#define SVM_SQRT3_F              ES_MATH_SQRT3_F
#define SVM_SQRT3_OVER_2_F       ES_MATH_HALF_SQRT3_F
#define SVM_TWO_OVER_SQRT3_F     ES_MATH_TWO_INV_SQRT3_F
#define SVM_TWO_THIRDS_F         ES_MATH_TWO_THIRDS_F
#define SVM_ONE_THIRD_F          ES_MATH_ONE_THIRD_F

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/
typedef enum
{
    SVM_SECTOR_I   = 0U,
    SVM_SECTOR_II  = 1U,
    SVM_SECTOR_III = 2U,
    SVM_SECTOR_IV  = 3U,
    SVM_SECTOR_V   = 4U,
    SVM_SECTOR_VI  = 5U
} SVM_Sector_T;

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
 */
extern void SVM_Init(void);

/**
 * \brief   Calculate SVPWM duty cycles from modulation index and angle.
 *          This is the direct method - modulation index is 0.0 to 1.0
 *          where 1.0 = Vdc/√3 phase voltage (maximum linear SVPWM)
 */
extern MatrixStatus_T SVM_CalculateDutyCycle(
    MatrixFloat                    ModIndex,
    const FocAngle_T     * const   Angle_P,
    SVM_DutyCycle_T      * const   DutyOut_P);

/**
 * \brief   Calculate SVPWM duty cycles from αβ voltage vector.
 *          V_AlphaBeta should be the actual phase voltage values.
 *          Vdc is used to normalize the voltage to modulation index.
 */
extern MatrixStatus_T SVM_CalculateDutyCycleFromAlphaBeta(
    const FocAlphaBeta_T * const V_AlphaBeta_P,
    MatrixFloat                  Vdc,
    MatrixFloat                  maxModulationIndex,   /* New parameter (e.g., 0.80F) */
    SVM_DutyCycle_T      * const DutyOut_P);

/**
 * \brief   Calculate SVPWM duty cycles from dq voltage vector.
 *          This is the primary function for FOC with Id=0 control.
 *          Vd and Vq are the actual dq voltages.
 *          Vdc is used to normalize to modulation index.
 */
extern MatrixStatus_T SVM_CalculateDutyCycleFromDq(
    const FocDq_T        * const V_Dq_P,
    const FocAngle_T     * const Angle_P,
    MatrixFloat                  Vdc,
    SVM_DutyCycle_T      * const DutyOut_P);

/**
 * \brief   Convert floating-point duty cycles to centre-aligned PWM compare values.
 */
extern MatrixStatus_T SVM_GetCompareValues(
    const SVM_DutyCycle_T  * const DutyIn_P,
    uint32_T                       TimerPeriod,
    uint32_T               * const CompAOut_P,
    uint32_T               * const CompBOut_P,
    uint32_T               * const CompCOut_P);

#endif /* EMBED_SIM_SV_PWM_H_ */
