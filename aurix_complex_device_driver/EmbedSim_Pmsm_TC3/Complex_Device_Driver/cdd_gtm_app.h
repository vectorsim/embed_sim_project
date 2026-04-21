/**********************************************************************************************************************
 * \file        cdd_gtm_app.h
 * \brief       GTM ATOM0 + DTM driver interface for 3-phase FOC PWM generation
 *              on the AURIX TC3xx Motor Control Power Board (AP32541).
 *
 * \details     Architecture:
 *
 *              ATOM0_CH0  — Master PWM, generates control loop ISR at CCU1 event
 *                           Output: P00.0  (scope probe / master trigger)
 *
 *              ATOM0_CH3  — ADC trigger for Resolver SIN/COS + DC-link
 *                           Internal only — GTM_ADCTRIG3OUT0/1 routed to EVADC G3/G11/G8
 *                           No physical pin output
 *
 *              ATOM0_CH4  — ADC trigger for Phase current (U/V/W)
 *                           Internal only — GTM_ADCTRIG0OUT0 routed to EVADC G0/G1/G2
 *                           No physical pin output
 *
 *              ATOM0_CH5 → DTM_CH0 — Phase U
 *                           DTM output L: P00.2  IL1  low-side  switch 1
 *                           DTM output H: P00.3  /IH1 high-side switch 1
 *
 *              ATOM0_CH6 → DTM_CH1 — Phase V
 *                           DTM output L: P00.4  IL2  low-side  switch 2
 *                           DTM output H: P00.5  /IH2 high-side switch 2
 *
 *              ATOM0_CH7 → DTM_CH2 — Phase W
 *                           DTM output L: P00.6  IL3  low-side  switch 3
 *                           DTM output H: P00.7  /IH3 high-side switch 3
 *
 *              DTM dead-time:
 *                  Both falling-edge and rising-edge dead-time are programmed
 *                  via DTM_CHx_DTV.  Default = CDD_GTM_DTM_DEAD_TIME_TICKS
 *                  (see cdd_config.h).
 *
 *              PWM update:
 *                  Duty cycle is updated by writing SR0/SR1 of each ATOM channel.
 *                  The AGC HOST_TRIG mechanism ensures shadow-register transfer
 *                  at the next PWM period start.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gtm_app.c
 *              - Rule 17.2 : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_GTM_APP_H_
#define CDD_GTM_APP_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"

/**********************************************************************************************************************
 * Configuration
 *********************************************************************************************************************/

/**
 * \brief   Dead-time in GTM CMU CLK0 ticks applied to both edges by the DTM.
 *
 * \details At 200 MHz CMU CLK0:  1 tick = 5 ns
 *          Default 20 ticks = 100 ns dead-time — suitable for TLE9180 + IPG20N04S4.
 *          Adjust to match gate driver turn-on/turn-off delay from TLE9180 datasheet.
 */
#ifndef CDD_GTM_DTM_DEAD_TIME_TICKS
#define CDD_GTM_DTM_DEAD_TIME_TICKS     (20U)
#endif

/**********************************************************************************************************************
 * Data Types
 *********************************************************************************************************************/

/**
 * \brief   Three-phase PWM duty cycle structure.
 *
 * \details Each field is a normalised duty cycle in range [0.0 .. 1.0].
 *          Values outside this range are clamped inside GTM_Set_PWM_Duty().
 */
typedef struct
{
    real32_T    DutyU;      /**< \brief Phase U duty cycle  [0..1] */
    real32_T    DutyV;      /**< \brief Phase V duty cycle  [0..1] */
    real32_T    DutyW;      /**< \brief Phase W duty cycle  [0..1] */
} GTM_PWM_Duty_T;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises GTM CMU, ATOM0 channels, DTM channels, and pin mux.
 *
 * \details Call sequence:
 *              1. Set GTM CMU CLK0 frequency (via cdd_sys_utility)
 *              2. Configure ATOM0_CH0 master in SOMP mode
 *              3. Configure ATOM0_CH3/CH4 ADC trigger channels (internal)
 *              4. Configure ATOM0_CH5/CH6/CH7 phase PWM channels in SOMP mode
 *              5. Configure DTM_CH0/CH1/CH2 with dead-time
 *              6. Set TOUTSEL + call GPIO_Configure_GTM_<phase>_<Pxx_y>() per pin
 *              7. Arm AGC with HOST_TRIG
 *
 *          Pad configuration (IOCR + PDR) for P00.0 / P00.2–P00.7 is owned
 *          by cdd_gpio_app — call Initialize_GPIO_Module() before this function.
 *
 *          Must be called after cdd_sys_utility GTM CMU CLK0 setup and
 *          before global interrupt enable.
 *
 * \return  None
 */
extern void Initialize_GTM_Module(void);

/**
 * \brief   Updates the three-phase PWM duty cycles.
 *
 * \details Writes SR0/SR1 registers of ATOM0_CH5/CH6/CH7 and the ADC
 *          trigger channels SR0/SR1.  The shadow registers are transferred
 *          to the active compare registers at the next AGC update event
 *          (HOST_TRIG or automatic period update).
 *
 *          This function is called from the control ISR (GTM_ATOM_00_CH_00_ISR)
 *          every PWM period.
 *
 * \param   Duty_Ptr   Pointer to the duty cycle structure  [0..1 per phase]
 * \return  None
 */
extern void GTM_Set_PWM_Duty(const GTM_PWM_Duty_T * const Duty_Ptr);

/**
 * \brief   Returns the current control loop period in GTM CMU CLK0 ticks.
 * \return  Period ticks  [CLK0 ticks]
 */
extern uint32_T GTM_Get_Period_Ticks(void);

/**
 * \brief   Returns the control loop sample time in seconds.
 * \return  Sample time  [s]
 */
extern real32_T GTM_Get_Sample_Time(void);

/**
 * \brief   Enables all ATOM0 PWM outputs via AGC ENDIS and OUTEN.
 * \details Call after TLE9180 is configured and in normal state.
 * \return  None
 */
extern void GTM_Enable_PWM_Outputs(void);

/**
 * \brief   Disables all ATOM0 PWM outputs immediately via AGC ENDIS.
 * \details Safe to call from any context including ISR.
 * \return  None
 */
extern void GTM_Disable_PWM_Outputs(void);

#endif /* CDD_GTM_APP_H_ */
