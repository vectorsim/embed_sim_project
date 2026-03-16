/**
 * motor_utility_blocks.h
 * ======================
 * EmbedSim — NANOTEC DB42S02  Open-loop V/f controller blocks
 *
 * Five stateful blocks used in the open-loop V/f controller chain:
 *
 *   SpeedRamp   — linear ramp 0 → omega_target [rad/s], then hold
 *   VfAngle     — open-loop angle integrator + V/f voltage law
 *   VfDQ        — extract [v_d, v_q] from VfAngle output
 *   VfTheta     — extract [theta_e]  from VfAngle output
 *   DutyPack    — InvClarke + centred PWM → three phase duty cycles
 *
 * Each block follows the EmbedSim C convention:
 *   <Block>_Init(&state, ...)          — call once at startup
 *   <Block>_Step(&state, u, dt, y)    — call every sample period
 *
 * Inputs / outputs are flat real32_T arrays — compatible with the
 * Cython wrapper pattern used throughout fs_electrical_machines.
 *
 * MISRA C:2012 compliant.
 * No dynamic memory.  No static locals — all state in caller structs.
 * Target: AURIX TriCore TC3xx  (real32_T = float, Sys_Types.h)
 *
 * Part of fs_electrical_machines/c_src/
 */

#ifndef MOTOR_UTILITY_BLOCKS_H
#define MOTOR_UTILITY_BLOCKS_H

#include "Sys_Types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Mathematical constants
 * ========================================================================== */
#define MUB_TWO_PI   (6.28318530f)
#define MUB_SQRT3_2  (0.86602540f)   /*!< sqrt(3) / 2  */

/* ============================================================================
 * SpeedRamp
 *   Output y[0]: omega_m_ref [rad/s]
 * ========================================================================== */
typedef struct
{
    real32_T ramp_value;   /*!< Current ramp state [rad/s]          */
    real32_T rate;         /*!< Ramp rate [rad/s per second]        */
    real32_T target;       /*!< Hold value [rad/s]                  */
} SpeedRamp_T;

/**
 * SpeedRamp_Init
 *   s            : pointer to state struct
 *   omega_target : final hold value [rad/s]
 *   ramp_time    : time to reach omega_target from zero [s]
 */
void SpeedRamp_Init(SpeedRamp_T *s,
                    real32_T     omega_target,
                    real32_T     ramp_time);

/**
 * SpeedRamp_Step
 *   s  : pointer to state struct
 *   dt : sample period [s]
 *   y  : output — y[0] = omega_m_ref [rad/s]
 *
 * NUM_INPUTS  = 0  (source block — no u[] argument)
 * OUTPUT_SIZE = 1
 */
void SpeedRamp_Step(SpeedRamp_T *s,
                    real32_T     dt,
                    real32_T    *y);

/* ============================================================================
 * VfAngle
 *   Input  u[0]: omega_m_ref [rad/s]
 *   Output y[0]: v_d   [V]   (always 0 for open-loop V/f)
 *          y[1]: v_q   [V]
 *          y[2]: theta_e [rad]
 * ========================================================================== */
typedef struct
{
    real32_T theta_e;      /*!< Electrical angle accumulator [rad]  */
    real32_T vf_ratio;     /*!< V/f gain [V·s/rad]                  */
    real32_T v_phase_peak; /*!< Peak phase voltage limit [V]        */
    uint8_T  p_poles;      /*!< Number of pole pairs                */
} VfAngle_T;

/**
 * VfAngle_Init
 *   s            : pointer to state struct
 *   vf_ratio     : V/f voltage gain [V·s/rad]
 *   v_phase_peak : peak phase voltage limit [V]
 *   p_poles      : pole pairs
 */
void VfAngle_Init(VfAngle_T *s,
                  real32_T   vf_ratio,
                  real32_T   v_phase_peak,
                  uint8_T    p_poles);

/**
 * VfAngle_Step
 *   s  : pointer to state struct
 *   u  : input  — u[0] = omega_m_ref [rad/s]
 *   dt : sample period [s]
 *   y  : output — y[0]=v_d, y[1]=v_q, y[2]=theta_e
 *
 * NUM_INPUTS  = 1
 * OUTPUT_SIZE = 3
 */
void VfAngle_Step(VfAngle_T  *s,
                  const real32_T *u,
                  real32_T        dt,
                  real32_T       *y);

/* ============================================================================
 * VfDQ
 *   Input  u[0..2]: full VfAngle output [v_d, v_q, theta_e]
 *   Output y[0]: v_d [V]
 *          y[1]: v_q [V]
 *
 *   Stateless pass-through — state struct is a zero-byte placeholder
 *   kept only so PYXInspector sees the same Init/Step pattern.
 * ========================================================================== */
typedef struct
{
    uint8_T _reserved;   /*!< No state — placeholder for uniform API */
} VfDQ_T;

void VfDQ_Init(VfDQ_T *s);

/**
 * VfDQ_Step
 *   NUM_INPUTS  = 3  (full VfAngle output)
 *   OUTPUT_SIZE = 2
 */
void VfDQ_Step(VfDQ_T         *s,
               const real32_T *u,
               real32_T        dt,
               real32_T       *y);

/* ============================================================================
 * VfTheta
 *   Input  u[0..2]: full VfAngle output [v_d, v_q, theta_e]
 *   Output y[0]: theta_e [rad]
 *
 *   Stateless pass-through — same placeholder pattern as VfDQ.
 * ========================================================================== */
typedef struct
{
    uint8_T _reserved;
} VfTheta_T;

void VfTheta_Init(VfTheta_T *s);

/**
 * VfTheta_Step
 *   NUM_INPUTS  = 3
 *   OUTPUT_SIZE = 1
 */
void VfTheta_Step(VfTheta_T      *s,
                  const real32_T *u,
                  real32_T        dt,
                  real32_T       *y);

/* ============================================================================
 * DutyPack
 *   Input  u[0]: v_alpha [V]
 *          u[1]: v_beta  [V]
 *   Output y[0]: duty_a  [0..1]
 *          y[1]: duty_b  [0..1]
 *          y[2]: duty_c  [0..1]
 *          y[3]: V_dc    [V]   (pass-through constant)
 *          y[4]: T_load  [Nm]  (zero — placeholder for FMU interface)
 *
 *   Stateless combinatorial block — placeholder struct for uniform API.
 * ========================================================================== */
typedef struct
{
    real32_T v_dc;         /*!< DC bus voltage [V]                  */
} DutyPack_T;

/**
 * DutyPack_Init
 *   s    : pointer to state struct
 *   v_dc : DC bus voltage [V]
 */
void DutyPack_Init(DutyPack_T *s,
                   real32_T    v_dc);

/**
 * DutyPack_Step
 *   u  : input  — u[0]=v_alpha, u[1]=v_beta
 *   dt : sample period [s]  (unused — combinatorial block)
 *   y  : output — y[0..4] as above
 *
 *   NUM_INPUTS  = 2
 *   OUTPUT_SIZE = 5
 */
void DutyPack_Step(DutyPack_T     *s,
                   const real32_T *u,
                   real32_T        dt,
                   real32_T       *y);

/* ============================================================================
 * SVPWMPack
 *   Input  u[0]: v_alpha [V]
 *          u[1]: v_beta  [V]
 *   Output y[0]: Vref        [V]    = sqrt(v_alpha^2 + v_beta^2)
 *          y[1]: alpha_angle [rad]  = atan2(v_beta, v_alpha)
 *          y[2]: V_dc        [V]    = compile-time constant (set by Init)
 *
 *   Stateless combinatorial block — placeholder struct for uniform API.
 *   Requires <math.h> sqrtf() and atan2f().
 * ========================================================================== */
typedef struct
{
    real32_T v_dc;         /*!< DC bus voltage [V] — set once at Init      */
} SVPWMPack_T;

/**
 * SVPWMPack_Init
 *   s    : pointer to state struct
 *   v_dc : DC bus voltage [V]
 */
void SVPWMPack_Init(SVPWMPack_T *s,
                    real32_T     v_dc);

/**
 * SVPWMPack_Step
 *   u  : input  — u[0]=v_alpha, u[1]=v_beta
 *   dt : sample period [s]  (unused — combinatorial block)
 *   y  : output — y[0]=Vref, y[1]=alpha_angle, y[2]=V_dc
 *
 *   NUM_INPUTS  = 2
 *   OUTPUT_SIZE = 3
 */
void SVPWMPack_Step(SVPWMPack_T    *s,
                    const real32_T *u,
                    real32_T        dt,
                    real32_T       *y);

#ifdef __cplusplus
}
#endif

#endif /* MOTOR_UTILITY_BLOCKS_H */
