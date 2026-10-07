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
 *            ## Public API surface
 *            Only three functions are exported by this module:
 *              - `CddGtm_InitModule()`   — bring the GTM module clock up and
 *                                          release write-protection.
 *              - `CddGtm_InitInverter()` — full ATOM0 / CDTM0 / pin-mux setup
 *                                          with shadow registers pre-loaded
 *                                          to the 50 % zero vector.
 *              - `CddGtm_Start()`        — issue HOST_TRIG; PWM goes live.
 *
 *            Everything else (duty writes, open-loop ramp, DFC dispatch) is
 *            driven from the 20 kHz ATOM0_CH0 CCU1 ISR defined in the .c file.
 *
 *            ## Required call order
 *            ```
 *            CddGtm_InitModule();        // clock + write-protect off
 *            CddGtm_InitInverter();      // channels, mux, shadow pre-load
 *            CddApp_InitInverter();      // gate-driver enable (elsewhere)
 *            CddGtm_Start();             // HOST_TRIG -> PWM live
 *            // SRC SRE = 1 -> ISR armed
 *            ```
 *            Issuing HOST_TRIG before the gate driver is enabled would expose
 *            the power stage to an undefined duty; the split between
 *            `CddGtm_InitInverter()` and `CddGtm_Start()` exists precisely to
 *            let the application interleave `CddApp_InitInverter()`.
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
 *
 * Rationale: the mode is latched once at the RUN-entry edge and held for the
 * entire run.  Changing CddApp_G.CtrlMode mid-run has no effect until the
 * next fault / stop / re-entry.  This avoids a mode switch mid-PWM-period
 * (which would leave the DFC and open-loop states inconsistent) at the cost
 * of requiring an explicit stop/start to change strategy.
 */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/* No public global variables */

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Bring up the GTM module clock and disable cluster write-protection.
 * \return  void
 *
 * \details Releases the GTM module from reset (GTM_CLC.DISR = 0), waits for
 *          the clock to be running (GTM_CLC.DISS == 0), clears
 *          GTM_CTRL.RF_PROT and GTM_CCM0_PROT.CLS_PROT so subsequent register
 *          writes to the cluster are accepted, sets the cluster-0 clock
 *          divider, disables all CMU clocks, and finally programs CMU CLK0 to
 *          GTM_CMU_CLK0_FREQUENCY.
 *
 *          Must be the first GTM call in the boot sequence. Idempotency is
 *          not guaranteed — calling twice will re-run the clock-enable
 *          handshake and re-program CLK0.  Call once, before
 *          CddGtm_InitInverter().
 *
 * \note    The watchdog is briefly disabled around the GTM_CLC write so the
 *          clock-enable handshake does not trip the CPU WDT.
 */
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
 *          After this function returns, all ATOM0 channels are configured and
 *          their shadow registers hold the zero vector, but the carrier is
 *          not yet running: the outputs sit at whatever the SL bit dictates
 *          (idle state) and no CCU1 interrupt can fire.  This is the
 *          "configured but not armed" state, deliberately split from Start()
 *          so the application can bring the gate driver up first.
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
 *          HOST_TRIG is a single-shot write: after it lands, ATOM0 begins
 *          counting on the next CMU CLK0 edge and the ISR begins firing at
 *          each half-period valley.  There is no software path to un-trigger
 *          the carrier short of a hardware reset or clearing
 *          GTM_ATOM0_AGC_GLB_CTRL.HOST_TRIG back to zero.
 *
 * \warning Do not call before the gate driver is enabled and the DC link is
 *          within its valid operating window.  Once HOST_TRIG is issued, the
 *          PWM outputs are live and the power stage will switch on the next
 *          control-loop tick.
 *
 * \return  void
 */
extern void CddGtm_Start(void);


#endif /* CDD_GTM_APP_H_ */
