/**
 * motor_utility_blocks.c
 * ======================
 * EmbedSim — NANOTEC DB42S02  Open-loop V/f controller blocks
 *
 * Implementation of:
 *   SpeedRamp   VfAngle   VfDQ   VfTheta   DutyPack
 *
 * MISRA C:2012 compliant.
 *   Rule 14.4  — all conditionals on boolean / relational expressions
 *   Rule 15.5  — single return per function
 *   Rule 21.8  — no static locals; all state in caller structs
 *   Rule 10.x  — explicit casts for mixed-type arithmetic
 *
 * No dynamic memory.  No standard library beyond <math.h> fabsf().
 * Target: AURIX TriCore TC3xx
 */

#include "motor_utility_blocks.h"
#include <math.h>   /* fabsf */

/* ============================================================================
 * Internal helpers
 * ========================================================================== */

/** Clamp x to [lo, hi].  Branchless-friendly on TriCore. */
static real32_T mub_clamp(real32_T x, real32_T lo, real32_T hi)
{
    real32_T result = x;
    if (result < lo) { result = lo; }
    if (result > hi) { result = hi; }
    return result;
}

/** Wrap angle to [0, 2π). */
static real32_T mub_wrap_angle(real32_T theta)
{
    real32_T result = theta;
    if (result >= MUB_TWO_PI) { result -= MUB_TWO_PI; }
    if (result <  0.0f)       { result += MUB_TWO_PI; }
    return result;
}

/* ============================================================================
 * SpeedRamp
 * ========================================================================== */

void SpeedRamp_Init(SpeedRamp_T *s,
                    real32_T     omega_target,
                    real32_T     ramp_time)
{
    s->ramp_value = 0.0f;
    s->target     = omega_target;

    /* Guard: ramp_time <= 0 → instantaneous step */
    if (ramp_time > 0.0f)
    {
        s->rate = omega_target / ramp_time;
    }
    else
    {
        s->rate = omega_target;   /* reach target in one step */
    }
}

void SpeedRamp_Step(SpeedRamp_T *s,
                    real32_T     dt,
                    real32_T    *y)
{
    if (s->ramp_value < s->target)
    {
        s->ramp_value += s->rate * dt;
        if (s->ramp_value > s->target)
        {
            s->ramp_value = s->target;   /* clamp at hold value */
        }
    }
    y[0] = s->ramp_value;
}

/* ============================================================================
 * VfAngle
 * ========================================================================== */

void VfAngle_Init(VfAngle_T *s,
                  real32_T   vf_ratio,
                  real32_T   v_phase_peak,
                  uint8_T    p_poles)
{
    s->theta_e      = 0.0f;
    s->vf_ratio     = vf_ratio;
    s->v_phase_peak = v_phase_peak;
    s->p_poles      = p_poles;
}

void VfAngle_Step(VfAngle_T      *s,
                  const real32_T *u,
                  real32_T        dt,
                  real32_T       *y)
{
    const real32_T omega_m = u[0];
    const real32_T omega_e = (real32_T)s->p_poles * omega_m;
    real32_T       vq      = s->vf_ratio * fabsf(omega_e);

    /* Voltage limit */
    if (vq > s->v_phase_peak) { vq = s->v_phase_peak; }

    /* Angle integrator — wrap to [0, 2π) */
    s->theta_e = mub_wrap_angle(s->theta_e + omega_e * dt);

    y[0] = 0.0f;        /* v_d  — always zero for open-loop V/f */
    y[1] = vq;          /* v_q                                   */
    y[2] = s->theta_e;  /* theta_e                               */
}

/* ============================================================================
 * VfDQ  — pass-through extractor  [v_d, v_q] from VfAngle output
 * ========================================================================== */

void VfDQ_Init(VfDQ_T *s)
{
    s->_reserved = 0U;
}

void VfDQ_Step(VfDQ_T         *s,
               const real32_T *u,
               real32_T        dt,
               real32_T       *y)
{
    (void)s;    /* stateless */
    (void)dt;   /* combinatorial */
    y[0] = u[0];   /* v_d     */
    y[1] = u[1];   /* v_q     */
}

/* ============================================================================
 * VfTheta  — pass-through extractor  [theta_e] from VfAngle output
 * ========================================================================== */

void VfTheta_Init(VfTheta_T *s)
{
    s->_reserved = 0U;
}

void VfTheta_Step(VfTheta_T      *s,
                  const real32_T *u,
                  real32_T        dt,
                  real32_T       *y)
{
    (void)s;    /* stateless */
    (void)dt;   /* combinatorial */
    y[0] = u[2];   /* theta_e  — element [2] of VfAngle output */
}

/* ============================================================================
 * SVPWMPack  — polar conversion [v_alpha, v_beta] → [Vref, alpha_angle, Vdc]
 *
 *   Vref        = sqrtf(v_alpha^2 + v_beta^2)
 *   alpha_angle = atan2f(v_beta, v_alpha)
 *   y[2]        = v_dc  (compile-time constant, passed through)
 * ========================================================================== */

void SVPWMPack_Init(SVPWMPack_T *s,
                    real32_T     v_dc)
{
    s->v_dc = v_dc;
}

void SVPWMPack_Step(SVPWMPack_T    *s,
                    const real32_T *u,
                    real32_T        dt,
                    real32_T       *y)
{
    (void)dt;   /* combinatorial block */

    const real32_T v_alpha = u[0];
    const real32_T v_beta  = u[1];

    y[0] = sqrtf((v_alpha * v_alpha) + (v_beta * v_beta));   /* Vref        */
    y[1] = atan2f(v_beta, v_alpha);                           /* alpha_angle */
    y[2] = s->v_dc;                                           /* V_dc        */
}

/* ============================================================================
 * DutyPack  — InvClarke + centred PWM
 *
 *   v_a =  v_alpha
 *   v_b = -0.5 * v_alpha + (sqrt(3)/2) * v_beta
 *   v_c = -0.5 * v_alpha - (sqrt(3)/2) * v_beta
 *
 *   duty_x = 0.5 + v_x / V_dc          (centred modulation)
 *   duty_x = clamp(duty_x, 0.02, 0.98) (2 % dead-time guard)
 * ========================================================================== */

void DutyPack_Init(DutyPack_T *s,
                   real32_T    v_dc)
{
    s->v_dc = v_dc;
}

void DutyPack_Step(DutyPack_T     *s,
                   const real32_T *u,
                   real32_T        dt,
                   real32_T       *y)
{
    const real32_T v_alpha = u[0];
    const real32_T v_beta  = u[1];
    const real32_T inv_vdc = 1.0f / s->v_dc;

    (void)dt;   /* combinatorial block */

    const real32_T va = v_alpha;
    const real32_T vb = (-0.5f * v_alpha) + (MUB_SQRT3_2 * v_beta);
    const real32_T vc = (-0.5f * v_alpha) - (MUB_SQRT3_2 * v_beta);

    y[0] = mub_clamp(0.5f + (va * inv_vdc), 0.02f, 0.98f);   /* duty_a */
    y[1] = mub_clamp(0.5f + (vb * inv_vdc), 0.02f, 0.98f);   /* duty_b */
    y[2] = mub_clamp(0.5f + (vc * inv_vdc), 0.02f, 0.98f);   /* duty_c */
    y[3] = s->v_dc;   /* V_dc pass-through — read by FMU interface     */
    y[4] = 0.0f;      /* T_load placeholder                             */
}
