/**********************************************************************************************************************
 * \file      PI_FOC.c
 * \brief     PI Field Oriented Control implementation.
 *
 * Implements complete FOC control loop with:
 *   - Clarke, Park, Inverse Park transforms (reuses Coordinate_Transform.h)
 *   - Speed PI controller with anti-windup
 *   - Current PI controllers with feed-forward decoupling
 *   - Voltage saturation (hexagon limiting)
 *
 * Working type: \c MatrixFloat (= \c real32_T from \c Matrix.h / \c Sys_Types.h).
 * Uses \c cosf and \c sinf from \c <math.h> — single-precision, no double promotion.
 *
 * MISRA C:2012 compliance notes:
 *   - No dynamic memory allocation
 *   - No recursion
 *   - Single exit per function (Rule 15.5)
 *   - All literals carry the \c f suffix — no implicit double promotion
 *   - NULL guards on all pointer arguments
 *
 * \version   1.0.0
 * \copyright Copyright (C) EmbedSim 2025
 *
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_pi_foc.h"
#include "embed_sim_coordinate_transform.h"
#include <math.h>    /* cosf, sinf, sqrtf */
#include <string.h>  /* memset */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \addtogroup pi_foc_private_constants  Private numeric constants
 * \{
 */
/** \brief Zero (float) */
#define PI_FOC_ZERO_F   ((MatrixFloat)0.0f)

/** \brief One (float) */
#define PI_FOC_ONE_F    ((MatrixFloat)1.0f)
/** \} */


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None — state is held in caller-provided PI_FOC_T structure */


/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None */


/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Clamp a float value to [-limit, +limit].
 *
 * \param[in] value Input value.
 * \param[in] limit Upper/lower bound.
 * \return          Clamped value.
 */
static MatrixFloat PI_FOC_Clamp(const MatrixFloat value, const MatrixFloat limit);

/**
 * \brief  Speed PI controller with anti-windup.
 *
 * \param[in,out] int_spd Speed integrator state.
 * \param[in]     error   Speed error (ω_ref - ω_m) [rad/s].
 * \param[in]     dt      Time step [s].
 * \return                q-axis current reference [A] (clamped to ±I_max).
 */
static MatrixFloat PI_FOC_SpeedPI(
    MatrixFloat * const int_spd,
    const MatrixFloat    error,
    const MatrixFloat    dt);

/**
 * \brief  Current PI controller for d or q axis with anti-windup.
 *
 * \param[in,out] integrator Integrator state.
 * \param[in]     error      Current error (ref - meas) [A].
 * \param[in]     dt         Time step [s].
 * \param[in]     kp         Proportional gain [V/A].
 * \param[in]     ki         Integral gain [V/(A·s)].
 * \param[in]     clamp      Integrator clamp limit.
 * \return                   Voltage reference [V].
 */
static MatrixFloat PI_FOC_CurrentPI(
    MatrixFloat * const integrator,
    const MatrixFloat    error,
    const MatrixFloat    dt,
    const MatrixFloat    kp,
    const MatrixFloat    ki,
    const MatrixFloat    clamp);

/**
 * \brief  Saturate a voltage vector to the maximum allowable magnitude.
 *
 * \param[in,out] vd d-axis voltage [V].
 * \param[in,out] vq q-axis voltage [V].
 */
static void PI_FOC_SaturateVoltage(MatrixFloat * const vd, MatrixFloat * const vq);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_Clamp
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat PI_FOC_Clamp(const MatrixFloat value, const MatrixFloat limit)
{
    MatrixFloat result;

    result = value;

    if (result > limit)
    {
        result = limit;
    }
    else if (result < -limit)
    {
        result = -limit;
    }
    else
    {
        /* Already within bounds – no action required */
    }

    return result;
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_SpeedPI
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat PI_FOC_SpeedPI(
    MatrixFloat * const int_spd,
    const MatrixFloat    error,
    const MatrixFloat    dt)
{
    MatrixFloat iq_ref;
    MatrixFloat int_new;

    /* Update integrator with anti-windup clamp */
    int_new = *int_spd + (error * dt);
    *int_spd = PI_FOC_Clamp(int_new, PI_FOC_IQ_LIM);

    /* PI output: iq_ref = Kp·error + Ki·∫error·dt */
    iq_ref = (PI_FOC_KP_SPD * error) + (PI_FOC_KI_SPD * (*int_spd));

    /* Clamp output to motor current limit */
    return PI_FOC_Clamp(iq_ref, PI_FOC_I_MAX);
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_CurrentPI
 *------------------------------------------------------------------------------------------------------------------*/
static MatrixFloat PI_FOC_CurrentPI(
    MatrixFloat * const integrator,
    const MatrixFloat    error,
    const MatrixFloat    dt,
    const MatrixFloat    kp,
    const MatrixFloat    ki,
    const MatrixFloat    clamp)
{
    MatrixFloat output;
    MatrixFloat int_new;

    /* Update integrator with anti-windup clamp */
    int_new = *integrator + (error * dt);
    *integrator = PI_FOC_Clamp(int_new, clamp);

    /* PI output: v = Kp·error + Ki·∫error·dt */
    output = (kp * error) + (ki * (*integrator));

    return output;
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_SaturateVoltage
 *------------------------------------------------------------------------------------------------------------------*/
static void PI_FOC_SaturateVoltage(MatrixFloat * const vd, MatrixFloat * const vq)
{
    MatrixFloat magnitude;
    MatrixFloat scale;

    if ((vd != NULL) && (vq != NULL))
    {
        magnitude = sqrtf((*vd) * (*vd) + (*vq) * (*vq));

        if (magnitude > PI_FOC_V_MAX)
        {
            scale = PI_FOC_V_MAX / magnitude;
            *vd *= scale;
            *vq *= scale;
        }
        else
        {
            /* Within linear modulation range – no action required */
        }
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_Init
 *------------------------------------------------------------------------------------------------------------------*/
void PI_FOC_Init(PI_FOC_T * const s)
{
    if (s != NULL)
    {
        (void)memset(s, 0, sizeof(PI_FOC_T));
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_Step
 *------------------------------------------------------------------------------------------------------------------*/
void PI_FOC_Step(
    PI_FOC_T       * const s,
    MatrixFloat            omega_ref,
    MatrixFloat            omega_m,
    MatrixFloat            theta_e,
    MatrixFloat            ia,
    MatrixFloat            ib,
    MatrixFloat            ic,
    MatrixFloat            dt,
    MatrixFloat    * const v_alpha,
    MatrixFloat    * const v_beta,
    MatrixFloat    * const vdc)
{
    MatrixFloat i_alpha;
    MatrixFloat i_beta;
    MatrixFloat id_meas;
    MatrixFloat iq_meas;
    MatrixFloat error_spd;
    MatrixFloat error_id;
    MatrixFloat error_iq;
    MatrixFloat we;              /* Electrical speed [rad/s] */
    MatrixFloat vd;
    MatrixFloat vq;
    MatrixFloat iq_ref;

    /* Static transform states (combinatorial — initialised once) */
    static Clarke_T    clarke_state;
    static Park_T      park_state;
    static InvPark_T   inv_park_state;

    /* Input validation */
    if ((s == NULL) || (v_alpha == NULL) || (v_beta == NULL) || (vdc == NULL))
    {
        /* MISRA C:2012 Rule 15.7: else clause required — function does nothing */
        return;
    }

    /* Initialise transform states once (static ensures only first call) */
    {
        static uint32_T init_done = 0U;
        if (init_done == 0U)
        {
            Clarke_Init(&clarke_state);
            Park_Init(&park_state);
            InvPark_Init(&inv_park_state);
            init_done = 1U;
        }
        else
        {
            /* Already initialised – no action required */
        }
    }

    /* 1. Clarke transform: abc → αβ */
    Clarke_Step(&clarke_state, ia, ib, ic, &i_alpha, &i_beta);

    /* 2. Park transform: αβ → dq (currents) */
    Park_Step(&park_state, i_alpha, i_beta, theta_e, &id_meas, &iq_meas);

    /* 3. Speed PI controller (output iq_ref) */
    error_spd = omega_ref - omega_m;
    iq_ref = PI_FOC_SpeedPI(&s->int_spd, error_spd, dt);

    /* Store for diagnostics */
    s->iq_ref = iq_ref;
    s->id_ref = PI_FOC_ZERO_F;  /* MTPA operation */

    /* 4. Electrical speed for feed-forward decoupling */
    we = (MatrixFloat)PI_FOC_P_POLES * omega_m;

    /* 5. Current PI controllers with feed-forward decoupling */
    error_id = s->id_ref - id_meas;
    error_iq = iq_ref - iq_meas;

    vd = PI_FOC_CurrentPI(&s->int_id, error_id, dt,
                          PI_FOC_KP_I, PI_FOC_KI_I, PI_FOC_V_LIM);
    vq = PI_FOC_CurrentPI(&s->int_iq, error_iq, dt,
                          PI_FOC_KP_I, PI_FOC_KI_I, PI_FOC_V_LIM);

    /* Add feed-forward decoupling terms:
     *   vd += -ωe·Lq·iq
     *   vq +=  ωe·(Ld·id + λ_pm)
     */
    vd += -we * PI_FOC_L_Q * iq_meas;
    vq +=  we * (PI_FOC_L_D * id_meas + PI_FOC_LAMBDA_PM);

    /* Store for diagnostics */
    s->vd = vd;
    s->vq = vq;

    /* 6. Voltage saturation (hexagon limiting) */
    PI_FOC_SaturateVoltage(&vd, &vq);

    /* 7. Inverse Park transform: dq → αβ */
    InvPark_Step(&inv_park_state, vd, vq, theta_e, v_alpha, v_beta);

    /* 8. Pass through DC bus voltage */
    *vdc = PI_FOC_V_DC;

    /* 9. Diagnostic logging at 1 kHz */
    if (dt > PI_FOC_ZERO_F)
    {
        s->log_counter++;

        if (((MatrixFloat)s->log_counter * dt) >= s->log_next_time)
        {
            s->log_speed     = omega_m;
            s->log_speed_ref = omega_ref;
            s->log_iq_meas   = iq_meas;
            s->log_id_meas   = id_meas;
            s->log_next_time += PI_FOC_LOG_INTERVAL;
        }
        else
        {
            /* No logging at this step – no action required */
        }
    }
    else
    {
        /* dt is zero – logging disabled – no action required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_Reset
 *------------------------------------------------------------------------------------------------------------------*/
void PI_FOC_Reset(PI_FOC_T * const s)
{
    if (s != NULL)
    {
        PI_FOC_Init(s);
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required */
    }
}


/*--------------------------------------------------------------------------------------------------------------------
 * PI_FOC_GetDiagnostics
 *------------------------------------------------------------------------------------------------------------------*/
void PI_FOC_GetDiagnostics(
    const PI_FOC_T * const s,
    MatrixFloat    * const speed,
    MatrixFloat    * const speed_ref,
    MatrixFloat    * const iq,
    MatrixFloat    * const id)
{
    if ((s != NULL) && (speed != NULL) && (speed_ref != NULL) &&
        (iq != NULL) && (id != NULL))
    {
        *speed     = s->log_speed;
        *speed_ref = s->log_speed_ref;
        *iq        = s->log_iq_meas;
        *id        = s->log_id_meas;
    }
    else
    {
        /* MISRA C:2012 Rule 15.7: else clause required – function does nothing */
    }
}