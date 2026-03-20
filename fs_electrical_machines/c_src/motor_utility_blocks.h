/**********************************************************************************************************************
 * \file      Motor_Utility_Blocks.h
 * \brief     EmbedSim stateful block API for the NANOTEC DB42S02 open-loop V/f controller.
 *
 * Five blocks used in the open-loop V/f controller chain:
 *
 *   SpeedRamp  — linear ramp 0 → ω_target [rad/s], then hold
 *   VfAngle    — open-loop angle integrator + V/f voltage law
 *   VfDQ       — extract [v_d, v_q]  from VfAngle output
 *   VfTheta    — extract [θ_e]       from VfAngle output
 *   DutyPack   — Inverse-Clarke + centred PWM → three-phase duty cycles
 *   SVPWMPack  — polar conversion [v_α, v_β] → [V_ref, α_angle, V_dc]
 *
 * Each block follows the EmbedSim C convention:
 *   \code
 *   <Block>_Init(&state, ...)       // call once at startup
 *   <Block>_Step(&state, u, dt, y)  // call every sample period
 *   \endcode
 *
 * Inputs and outputs are flat \c real32_T arrays, compatible with the
 * Cython wrapper pattern used throughout \c fs_electrical_machines.
 *
 * Constraints:
 *   - MISRA C:2012 compliant
 *   - No dynamic memory allocation
 *   - No static locals — all state lives in caller-supplied structs
 *
 * Target: Infineon AURIX TriCore TC3xx  (\c real32_T = \c float, \c Sys_Types.h)
 *
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

#ifndef MOTOR_UTILITY_BLOCKS_H_
#define MOTOR_UTILITY_BLOCKS_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Sys_Types.h"

#ifdef __cplusplus
extern "C" {
#endif


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup mub_constants  Mathematical constants
 * \{
 */
/** \brief 2π = 6.28318530… */
#define MUB_TWO_PI   (6.28318530f)

/** \brief √3 / 2 = 0.86602540… */
#define MUB_SQRT3_2  (0.86602540f)
/** \} */


/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup mub_types  Block state structures
 * \{
 */

/**
 * \struct SpeedRamp_T
 * \brief  State for the linear speed-ramp source block.
 *
 * Output y[0]: ω_m_ref [rad/s]
 */
typedef struct
{
    real32_T  ramp_value;  /**< Current ramp output [rad/s].          */
    real32_T  rate;        /**< Ramp slew rate [rad/s²].              */
    real32_T  target;      /**< Hold value once ramp completes [rad/s].*/
} SpeedRamp_T;

/**
 * \struct VfAngle_T
 * \brief  State for the open-loop angle integrator and V/f voltage law.
 *
 * Input  u[0]: ω_m_ref [rad/s] \n
 * Output y[0]: v_d [V] (always 0 for open-loop V/f) \n
 *        y[1]: v_q [V] \n
 *        y[2]: θ_e [rad]
 */
typedef struct
{
    real32_T  theta_e;       /**< Electrical angle accumulator [rad].   */
    real32_T  vf_ratio;      /**< V/f voltage gain [V·s/rad].           */
    real32_T  v_phase_peak;  /**< Peak phase voltage limit [V].         */
    uint8_T   p_poles;       /**< Number of pole pairs.                 */
} VfAngle_T;

/**
 * \struct VfDQ_T
 * \brief  Stateless pass-through extractor — [v_d, v_q] from VfAngle output.
 *
 * The placeholder field exists so PYXInspector sees the uniform Init/Step
 * pattern without a zero-size struct.
 *
 * Input  u[0..2]: full VfAngle output [v_d, v_q, θ_e] \n
 * Output y[0]: v_d [V] \n
 *        y[1]: v_q [V]
 */
typedef struct
{
    uint8_T  _reserved;  /**< No state — placeholder for uniform API.  */
} VfDQ_T;

/**
 * \struct VfTheta_T
 * \brief  Stateless pass-through extractor — [θ_e] from VfAngle output.
 *
 * Input  u[0..2]: full VfAngle output [v_d, v_q, θ_e] \n
 * Output y[0]: θ_e [rad]
 */
typedef struct
{
    uint8_T  _reserved;  /**< No state — placeholder for uniform API.  */
} VfTheta_T;

/**
 * \struct DutyPack_T
 * \brief  State for the Inverse-Clarke + centred PWM duty-cycle block.
 *
 * Input  u[0]: v_α [V] \n
 *        u[1]: v_β [V] \n
 * Output y[0]: duty_a [0, 1] \n
 *        y[1]: duty_b [0, 1] \n
 *        y[2]: duty_c [0, 1] \n
 *        y[3]: V_dc   [V]  (pass-through constant) \n
 *        y[4]: T_load [Nm] (zero — placeholder for FMU interface)
 */
typedef struct
{
    real32_T  v_dc;  /**< DC bus voltage [V].                          */
} DutyPack_T;

/**
 * \struct SVPWMPack_T
 * \brief  State for the αβ polar-conversion block feeding the SVPWM stage.
 *
 * Input  u[0]: v_α [V] \n
 *        u[1]: v_β [V] \n
 * Output y[0]: V_ref       [V]   = √(v_α² + v_β²) \n
 *        y[1]: alpha_angle [rad] = atan2(v_β, v_α) \n
 *        y[2]: V_dc        [V]   (pass-through constant)
 */
typedef struct
{
    real32_T  v_dc;  /**< DC bus voltage [V] — set once at Init.       */
} SVPWMPack_T;

/** \} */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup mub_speedramp  SpeedRamp block
 * \{
 */

/**
 * \brief  Initialise the SpeedRamp block.
 *
 * \param[out] s             State struct to initialise (must not be NULL).
 * \param[in]  omega_target  Final hold value [rad/s].
 * \param[in]  ramp_time     Time to reach \p omega_target from zero [s].
 *                           Pass 0 or negative for an instantaneous step.
 */
extern void SpeedRamp_Init(
    SpeedRamp_T * s,
    real32_T      omega_target,
    real32_T      ramp_time);

/**
 * \brief  Advance the SpeedRamp block by one sample period.
 *
 * Source block — no input array.
 *
 * \param[in,out] s   State struct.
 * \param[in]     dt  Sample period [s].
 * \param[out]    y   Output array: y[0] = ω_m_ref [rad/s].
 *
 * NUM_INPUTS  = 0 \n
 * OUTPUT_SIZE = 1
 */
extern void SpeedRamp_Step(
    SpeedRamp_T * s,
    real32_T      dt,
    real32_T    * y);

/** \} */

/** \addtogroup mub_vfangle  VfAngle block
 * \{
 */

/**
 * \brief  Initialise the VfAngle block.
 *
 * \param[out] s             State struct to initialise (must not be NULL).
 * \param[in]  vf_ratio      V/f voltage gain [V·s/rad].
 * \param[in]  v_phase_peak  Peak phase voltage limit [V].
 * \param[in]  p_poles       Number of pole pairs.
 */
extern void VfAngle_Init(
    VfAngle_T * s,
    real32_T    vf_ratio,
    real32_T    v_phase_peak,
    uint8_T     p_poles);

/**
 * \brief  Advance the VfAngle block by one sample period.
 *
 * \param[in,out] s   State struct.
 * \param[in]     u   Input array: u[0] = ω_m_ref [rad/s].
 * \param[in]     dt  Sample period [s].
 * \param[out]    y   Output array: y[0] = v_d [V],
 *                                  y[1] = v_q [V],
 *                                  y[2] = θ_e [rad].
 *
 * NUM_INPUTS  = 1 \n
 * OUTPUT_SIZE = 3
 */
extern void VfAngle_Step(
    VfAngle_T      * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y);

/** \} */

/** \addtogroup mub_vfdq  VfDQ block
 * \{
 */

/**
 * \brief  Initialise the VfDQ pass-through block.
 *
 * \param[out] s  State struct to initialise (must not be NULL).
 */
extern void VfDQ_Init(VfDQ_T * s);

/**
 * \brief  Advance the VfDQ block by one sample period.
 *
 * \param[in,out] s   State struct (stateless — unused internally).
 * \param[in]     u   Input array: u[0..2] = full VfAngle output.
 * \param[in]     dt  Sample period [s] (unused — combinatorial block).
 * \param[out]    y   Output array: y[0] = v_d [V], y[1] = v_q [V].
 *
 * NUM_INPUTS  = 3 \n
 * OUTPUT_SIZE = 2
 */
extern void VfDQ_Step(
    VfDQ_T         * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y);

/** \} */

/** \addtogroup mub_vftheta  VfTheta block
 * \{
 */

/**
 * \brief  Initialise the VfTheta pass-through block.
 *
 * \param[out] s  State struct to initialise (must not be NULL).
 */
extern void VfTheta_Init(VfTheta_T * s);

/**
 * \brief  Advance the VfTheta block by one sample period.
 *
 * \param[in,out] s   State struct (stateless — unused internally).
 * \param[in]     u   Input array: u[0..2] = full VfAngle output.
 * \param[in]     dt  Sample period [s] (unused — combinatorial block).
 * \param[out]    y   Output array: y[0] = θ_e [rad].
 *
 * NUM_INPUTS  = 3 \n
 * OUTPUT_SIZE = 1
 */
extern void VfTheta_Step(
    VfTheta_T      * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y);

/** \} */

/** \addtogroup mub_dutypack  DutyPack block
 * \{
 */

/**
 * \brief  Initialise the DutyPack block.
 *
 * \param[out] s    State struct to initialise (must not be NULL).
 * \param[in]  v_dc DC bus voltage [V].
 */
extern void DutyPack_Init(
    DutyPack_T * s,
    real32_T     v_dc);

/**
 * \brief  Advance the DutyPack block by one sample period.
 *
 * Computes Inverse-Clarke voltages then maps to centred duty cycles
 * clamped to [0.02, 0.98] (2 % dead-time guard).
 *
 * \param[in,out] s   State struct.
 * \param[in]     u   Input array: u[0] = v_α [V], u[1] = v_β [V].
 * \param[in]     dt  Sample period [s] (unused — combinatorial block).
 * \param[out]    y   Output array: y[0] = duty_a, y[1] = duty_b,
 *                                  y[2] = duty_c, y[3] = V_dc [V],
 *                                  y[4] = T_load [Nm] (zero placeholder).
 *
 * NUM_INPUTS  = 2 \n
 * OUTPUT_SIZE = 5
 */
extern void DutyPack_Step(
    DutyPack_T     * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y);

/** \} */

/** \addtogroup mub_svpwmpack  SVPWMPack block
 * \{
 */

/**
 * \brief  Initialise the SVPWMPack block.
 *
 * \param[out] s    State struct to initialise (must not be NULL).
 * \param[in]  v_dc DC bus voltage [V].
 */
extern void SVPWMPack_Init(
    SVPWMPack_T * s,
    real32_T      v_dc);

/**
 * \brief  Advance the SVPWMPack block by one sample period.
 *
 * Converts αβ voltage vector to polar form.
 *
 * \param[in,out] s   State struct.
 * \param[in]     u   Input array: u[0] = v_α [V], u[1] = v_β [V].
 * \param[in]     dt  Sample period [s] (unused — combinatorial block).
 * \param[out]    y   Output array: y[0] = V_ref [V],
 *                                  y[1] = alpha_angle [rad],
 *                                  y[2] = V_dc [V].
 *
 * NUM_INPUTS  = 2 \n
 * OUTPUT_SIZE = 3
 */
extern void SVPWMPack_Step(
    SVPWMPack_T    * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y);

/** \} */

#ifdef __cplusplus
}
#endif

#endif /* MOTOR_UTILITY_BLOCKS_H_ */
