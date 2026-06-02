/**********************************************************************************************************************
 * \file        cdd_gtm_app.h
 * \brief       GTM ATOM0 direct 6-channel driver interface for 3-phase FOC PWM
 *              generation on the AP32541 motor control board (TC38x).
 *
 * \details     Channel assignment (TOUTSEL values from TC38x UM appx1):
 *
 *              ATOM0_CH0  — Master PWM carrier,P00.0 CCU1 ISR at half-period → CPU
 *              ATOM0_CH1  — Phase U LS   IL1  P00.2  active HIGH  SL=0  SOMP slave
 *              ATOM0_CH2  — Phase U HS  /IH1  P00.3  active LOW   SL=0  SOMP slave
 *              ATOM0_CH3  — Phase V LS   IL2  P00.4  active HIGH  SL=0  SOMP slave
 *              ATOM0_CH4  — Phase V HS  /IH2  P00.5  active LOW   SL=0  SOMP slave
 *              ATOM0_CH5  — Phase W LS   IL3  P00.6  active HIGH  SL=0  SOMP slave
 *              ATOM0_CH6  — Phase W HS  /IH3  P00.7  active LOW   SL=0  SOMP slave
 *              ATOM0_CH7  — ADC trigger (valley-aligned, internal only)
 *
 *              CddGtm_SetPwmDuty() is STATIC (internal to cdd_gtm_app.c).
 *              Open-loop V/f:
 *                  CddGtm_OpenLoopSetRpm(rpm, mi) — arms open-loop at target speed
 *                  CddGtm_OpenLoopStop()           — disarms, returns to 50% duty
 *                  CddGtm_OpenLoopRun()            — STATIC, called from ISR only
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gtm_app.c
 *              - Rule 17.2 : No recursion
 *
 * \version     1.2.0
 * \date        2025-05-24
 * \author      EmbedSim / EV Light Vehicle Foundation
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
#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Function Prototypes — Initialisation
 *********************************************************************************************************************/

/**
 * \brief   Initialises GTM CMU, ATOM0 channels CH0–CH7, CDTM0 DTM4/DTM5, and pin mux.
 *
 * \details Sequence:
 *              1. Compute PeriodTicks / HalfPeriodTicks / SampleTime in CddApp_G
 *              2. ATOM0_CH0  master SOMP carrier, CCU1 ISR armed (SRE=0)
 *              3. ATOM0_CH1–CH6  Phase U/V/W LS+HS (SL safe-off initial state)
 *              4. ATOM0_CH7  ADC valley trigger → ADCTRIG0 → EVADC G0/G1/G2
 *              5. CDTM0_DTM4/5  CLK_SEL = CMU CLK0, passthrough
 *              6. TOUTSEL + GPIO per pin
 *              7. Write 50% duty to CddApp_G.DutyU/V/W → shadow registers
 *              8. Initialise OL_State_G (open-loop state, active = 0)
 *
 *          HOST_TRIG is NOT issued here. Call CddGtm_Start() after CddApp_InitInverter().
 *
 * \return  void
 */
extern void CddGtm_Init(void);

/**
 * \brief   Issues HOST_TRIG — transfers shadow registers to active compare registers
 *          and starts the ATOM0 carrier.  PWM is live after this call.
 *
 * \details Must be called after CddGtm_Init() and CddApp_InitInverter().
 *          Arm the ISR (SRC SRE=1) after this.
 *
 * \return  void
 */
extern void CddGtm_Start(void);

/**********************************************************************************************************************
 * Function Prototypes — Accessors
 *********************************************************************************************************************/

/**
 * \brief   Returns the control loop carrier period in CMU CLK0 ticks.
 * \return  Period ticks  [CLK0 ticks]
 */
extern uint32_T CddGtm_GetPeriodTicks(void);

/**
 * \brief   Returns the control loop sample time in seconds.
 * \return  Sample time  [s]
 */
extern real32_T CddGtm_GetSampleTime(void);

#endif /* CDD_GTM_APP_H_ */
