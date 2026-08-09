/**********************************************************************************************************************
 * \file      cdd_gtm_app.h
 * \brief     GTM ATOM0 direct 6-channel driver interface for 3-phase FOC PWM
 *            generation on the AP32541 motor control board (TC38x).
 *
 * \details   Provides static-allocation GTM configuration targeting Infineon
 *            AURIX TC38x.  All algorithms are iterative — no recursion — and
 *            the implementation is MISRA C:2012 compliant.
 *
 *            Channel assignment (TOUTSEL values from TC38x UM appx1):
 *
 *            ATOM0_CH0  — Master PWM carrier,P00.0 CCU1 ISR at half-period → CPU
 *            ATOM0_CH1  — Phase U LS   IL1  P00.2  active HIGH  SL=0  SOMP slave
 *            ATOM0_CH2  — Phase U HS  /IH1  P00.3  active LOW   SL=0  SOMP slave
 *            ATOM0_CH3  — Phase V LS   IL2  P00.4  active HIGH  SL=0  SOMP slave
 *            ATOM0_CH4  — Phase V HS  /IH2  P00.5  active LOW   SL=0  SOMP slave
 *            ATOM0_CH5  — Phase W LS   IL3  P00.6  active HIGH  SL=0  SOMP slave
 *            ATOM0_CH6  — Phase W HS  /IH3  P00.7  active LOW   SL=0  SOMP slave
 *            ATOM0_CH7  — ADC trigger  P00.8 (TOUT17)  duty 0.9, EVADC G0/G1/G2
 *
 *            CddGtm_SetPwmDuty(), CddGtm_RunOpenLoop(), CddGtm_RunDfc() are
 *            STATIC (internal to cdd_gtm_app.c) — dispatched by the 20 kHz
 *            ISR on the mode latched from CddApp_G.CtrlMode.
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : All functions have explicit return type
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gtm_app.c
 *              - Rule 17.2 : No recursion
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCase_P
 *              - Local variables: lower_snake_case
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.6.0
 * \date      2026-07-04
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_GTM_APP_H_
#define CDD_GTM_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "cdd_config.h"                 /* embed_sim_sys_types.h + embed_sim_compiler.h */
#include "embed_sim_dfc_controller.h"   /* DFC_Mode_T, DFC_Diag_T                       */

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/* No public macros defined in this header */

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/*
 * The control-mode selection (CddApp_CtrlMode_T: CDDAPP_CTRL_OPENLOOP /
 * CDDAPP_CTRL_CLOSEDLOOP) and the speed reference live in the central
 * CddApp_T — see cdd_app.h.  This module only executes the latched mode.
 */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/* No public global variables */

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/


extern void CddGtm_InitModule(void);



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
extern void CddGtm_InitInverter(void);

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





#endif /* CDD_GTM_APP_H_ */
