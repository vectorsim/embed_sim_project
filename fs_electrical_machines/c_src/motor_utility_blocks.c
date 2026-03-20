/**********************************************************************************************************************
 * \file      Motor_Utility_Blocks.c
 * \brief     EmbedSim open-loop V/f controller block implementations.
 *
 * Implements: SpeedRamp, VfAngle, VfDQ, VfTheta, DutyPack, SVPWMPack.
 *
 * MISRA C:2012 compliance notes:
 *   Rule 14.4  — all conditionals use boolean / relational expressions
 *   Rule 15.5  — single return per function
 *   Rule 21.8  — no static locals; all state in caller-supplied structs
 *   Rule 10.x  — explicit casts for mixed-type arithmetic
 *
 * No dynamic memory allocation.
 * Standard library use limited to \c <math.h>: \c fabsf, \c sqrtf, \c atan2f.
 *
 * Target: Infineon AURIX TriCore TC3xx
 *
 * \copyright Copyright (C) EmbedSim 2024
 *
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Motor_Utility_Blocks.h"
#include <math.h>   /* fabsf, sqrtf, atan2f */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Clamp \p x to the closed interval [\p lo, \p hi].
 *
 * \param[in] x   Value to clamp.
 * \param[in] lo  Lower bound.
 * \param[in] hi  Upper bound.
 * \return        Clamped value.
 */
static real32_T MUB_Clamp(real32_T x, real32_T lo, real32_T hi);

/**
 * \brief  Wrap angle \p theta into [0, 2π).
 *
 * Handles the common ±1-period overshoot arising from a single Euler
 * integration step; does not loop for multiple-period jumps.
 *
 * \param[in] theta  Angle in radians.
 * \return           Wrapped angle in [0, 2π).
 */
static real32_T MUB_WrapAngle(real32_T theta);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * MUB_Clamp  (private)
 *------------------------------------------------------------------------------------------------------------------*/
static real32_T MUB_Clamp(real32_T x, real32_T lo, real32_T hi)
{
    real32_T result;

    result = x;

    if (result < lo)
    {
        result = lo;
    }
    else
    {
        /* No action — already above lower bound. */
    }

    if (result > hi)
    {
        result = hi;
    }
    else
    {
        /* No action — already below upper bound. */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * MUB_WrapAngle  (private)
 *------------------------------------------------------------------------------------------------------------------*/
static real32_T MUB_WrapAngle(real32_T theta)
{
    real32_T result;

    result = theta;

    if (result >= MUB_TWO_PI)
    {
        result -= MUB_TWO_PI;
    }
    else
    {
        /* No action — below upper wrap boundary. */
    }

    if (result < 0.0f)
    {
        result += MUB_TWO_PI;
    }
    else
    {
        /* No action — above lower wrap boundary. */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SpeedRamp_Init
 *------------------------------------------------------------------------------------------------------------------*/
void SpeedRamp_Init(
    SpeedRamp_T * s,
    real32_T      omega_target,
    real32_T      ramp_time)
{
    s->ramp_value = 0.0f;
    s->target     = omega_target;

    /* Guard: ramp_time ≤ 0 → instantaneous step to target in one sample. */
    if (ramp_time > 0.0f)
    {
        s->rate = omega_target / ramp_time;
    }
    else
    {
        s->rate = omega_target;
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * SpeedRamp_Step
 *------------------------------------------------------------------------------------------------------------------*/
void SpeedRamp_Step(
    SpeedRamp_T * s,
    real32_T      dt,
    real32_T    * y)
{
    if (s->ramp_value < s->target)
    {
        s->ramp_value += s->rate * dt;

        if (s->ramp_value > s->target)
        {
            s->ramp_value = s->target;  /* Clamp at hold value. */
        }
        else
        {
            /* No action — still ramping. */
        }
    }
    else
    {
        /* No action — ramp already at or above target. */
    }

    y[0] = s->ramp_value;
}


/*--------------------------------------------------------------------------------------------------------------------
 * VfAngle_Init
 *------------------------------------------------------------------------------------------------------------------*/
void VfAngle_Init(
    VfAngle_T * s,
    real32_T    vf_ratio,
    real32_T    v_phase_peak,
    uint8_T     p_poles)
{
    s->theta_e      = 0.0f;
    s->vf_ratio     = vf_ratio;
    s->v_phase_peak = v_phase_peak;
    s->p_poles      = p_poles;
}


/*--------------------------------------------------------------------------------------------------------------------
 * VfAngle_Step
 *------------------------------------------------------------------------------------------------------------------*/
void VfAngle_Step(
    VfAngle_T      * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y)
{
    real32_T omega_m;
    real32_T omega_e;
    real32_T vq;

    omega_m = u[0];
    omega_e = (real32_T)s->p_poles * omega_m;
    vq      = s->vf_ratio * fabsf(omega_e);

    /* Voltage limit — clamp v_q to peak phase voltage. */
    if (vq > s->v_phase_peak)
    {
        vq = s->v_phase_peak;
    }
    else
    {
        /* No action — within voltage limit. */
    }

    /* Euler angle integrator — wrap result to [0, 2π). */
    s->theta_e = MUB_WrapAngle(s->theta_e + (omega_e * dt));

    y[0] = 0.0f;       /* v_d  — always zero for open-loop V/f. */
    y[1] = vq;         /* v_q.                                   */
    y[2] = s->theta_e; /* θ_e.                                   */
}


/*--------------------------------------------------------------------------------------------------------------------
 * VfDQ_Init
 *------------------------------------------------------------------------------------------------------------------*/
void VfDQ_Init(VfDQ_T * s)
{
    s->_reserved = 0U;
}


/*--------------------------------------------------------------------------------------------------------------------
 * VfDQ_Step
 *------------------------------------------------------------------------------------------------------------------*/
void VfDQ_Step(
    VfDQ_T         * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y)
{
    (void)s;   /* Stateless block — state unused. */
    (void)dt;  /* Combinatorial block — dt unused. */

    y[0] = u[0];  /* v_d. */
    y[1] = u[1];  /* v_q. */
}


/*--------------------------------------------------------------------------------------------------------------------
 * VfTheta_Init
 *------------------------------------------------------------------------------------------------------------------*/
void VfTheta_Init(VfTheta_T * s)
{
    s->_reserved = 0U;
}


/*--------------------------------------------------------------------------------------------------------------------
 * VfTheta_Step
 *------------------------------------------------------------------------------------------------------------------*/
void VfTheta_Step(
    VfTheta_T      * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y)
{
    (void)s;   /* Stateless block — state unused. */
    (void)dt;  /* Combinatorial block — dt unused. */

    y[0] = u[2];  /* θ_e — element [2] of VfAngle output. */
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVPWMPack_Init
 *------------------------------------------------------------------------------------------------------------------*/
void SVPWMPack_Init(
    SVPWMPack_T * s,
    real32_T      v_dc)
{
    s->v_dc = v_dc;
}


/*--------------------------------------------------------------------------------------------------------------------
 * SVPWMPack_Step
 *------------------------------------------------------------------------------------------------------------------*/
void SVPWMPack_Step(
    SVPWMPack_T    * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y)
{
    real32_T v_alpha;
    real32_T v_beta;

    (void)dt;  /* Combinatorial block — dt unused. */

    v_alpha = u[0];
    v_beta  = u[1];

    y[0] = sqrtf((v_alpha * v_alpha) + (v_beta * v_beta));  /* V_ref.       */
    y[1] = atan2f(v_beta, v_alpha);                          /* alpha_angle. */
    y[2] = s->v_dc;                                          /* V_dc.        */
}


/*--------------------------------------------------------------------------------------------------------------------
 * DutyPack_Init
 *------------------------------------------------------------------------------------------------------------------*/
void DutyPack_Init(
    DutyPack_T * s,
    real32_T     v_dc)
{
    s->v_dc = v_dc;
}


/*--------------------------------------------------------------------------------------------------------------------
 * DutyPack_Step
 *
 * Inverse-Clarke transform:
 *   v_a =  v_α
 *   v_b = −0.5·v_α + (√3/2)·v_β
 *   v_c = −0.5·v_α − (√3/2)·v_β
 *
 * Centred modulation:
 *   duty_x = 0.5 + v_x / V_dc
 *   duty_x = clamp(duty_x, 0.02, 0.98)   [2 % dead-time guard]
 *------------------------------------------------------------------------------------------------------------------*/
void DutyPack_Step(
    DutyPack_T     * s,
    const real32_T * u,
    real32_T         dt,
    real32_T       * y)
{
    real32_T v_alpha;
    real32_T v_beta;
    real32_T inv_vdc;
    real32_T va;
    real32_T vb;
    real32_T vc;

    (void)dt;  /* Combinatorial block — dt unused. */

    v_alpha = u[0];
    v_beta  = u[1];
    inv_vdc = 1.0f / s->v_dc;

    /* Inverse-Clarke. */
    va =  v_alpha;
    vb = (-0.5f * v_alpha) + (MUB_SQRT3_2 * v_beta);
    vc = (-0.5f * v_alpha) - (MUB_SQRT3_2 * v_beta);

    /* Centred duty cycles, clamped to [0.02, 0.98]. */
    y[0] = MUB_Clamp(0.5f + (va * inv_vdc), 0.02f, 0.98f);  /* duty_a. */
    y[1] = MUB_Clamp(0.5f + (vb * inv_vdc), 0.02f, 0.98f);  /* duty_b. */
    y[2] = MUB_Clamp(0.5f + (vc * inv_vdc), 0.02f, 0.98f);  /* duty_c. */
    y[3] = s->v_dc;  /* V_dc — pass-through for FMU interface.          */
    y[4] = 0.0f;     /* T_load — zero placeholder.                       */
}
