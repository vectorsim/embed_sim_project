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
 *              ATOM0_CH7  — ADC trigger  P00.8 (TOUT17)  duty 0.9, EVADC G0/G1/G2
 *
 *              CddGtm_SetPwmDuty(), CddGtm_RunOpenLoop(), CddGtm_RunDfc() are
 *              STATIC (internal to cdd_gtm_app.c) — dispatched by the 20 kHz
 *              ISR on the mode latched from CddApp_G.CtrlMode.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gtm_app.c
 *              - Rule 17.2 : No recursion
 *
 * \version     1.5.0
 * \date        2026-07-04
 *
 * \par v1.5.0
 *   Redesign around the central CddApp_T:
 *     - Two control options only — CDDAPP_CTRL_OPENLOOP (V/f at the
 *       slew-limited SpeedRefRpm) and CDDAPP_CTRL_CLOSEDLOOP (full sensorless
 *       DFC).  Mode and speed live in CddApp_T; set via CddApp_SetCtrlMode()
 *       / CddApp_SetSpeedRefRpm().  The mode is latched by the ISR once on
 *       the activation edge — no switching during operation.
 *     - DFC loop-option API (A/B) removed; CLOSEDLOOP always runs the full
 *       DFC sequence.  CddGtm_SetCtrlMode/GetCtrlMode/SetDfcLoopOption/
 *       GetDfcLoopOption/SetSpeedRefRpm removed (moved to cdd_app).
 *     - EVADC measurements read once per ISR tick into CddApp_G.Meas and
 *       CddApp_G.PhaseCurrents (both modes).
 *
 * \par v1.4.0
 *   Loop option simplified to a pre-start selection — no runtime switching:
 *     CddGtm_SetDfcLoopOption() is accepted only while the DFC is inactive;
 *     the ISR latches the option once on the activation edge, fixed for the
 *     entire run.  Change requires stop → set → restart.
 *   CddGtm_CtrlInit() default SpeedRefRpm corrected to 0.0F (was 1500.0F) —
 *   the host must command a speed explicitly before CddApp_Start().
 *
 * \par v1.3.0
 *   Flatness loop options wired through to the DFC (DFC_LoopOption_T v4.3.0):
 *     Option A — CddGtm_SetDfcLoopOption(DFC_LOOP_OPENLOOP):  I-f hold.
 *     Option B — CddGtm_SetDfcLoopOption(DFC_LOOP_CLOSEDLOOP): closed loop.
 *   ATOM0_CH7 ADC trigger reconfigured to a fixed 0.9 duty cycle
 *   (GTM_ADC_TRIG_DUTY in cdd_gtm_app.c).
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
#include "cdd_config.h"                 /* embed_sim_sys_types.h + embed_sim_compiler.h */
#include "embed_sim_dfc_controller.h"   /* DFC_Mode_T, DFC_Diag_T                       */

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/*
 * The control-mode selection (CddApp_CtrlMode_T: CDDAPP_CTRL_OPENLOOP /
 * CDDAPP_CTRL_CLOSEDLOOP) and the speed reference live in the central
 * CddApp_T — see cdd_app.h.  This module only executes the latched mode.
 */

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

/**********************************************************************************************************************
 * Function Prototypes — Control Loop (DFC integration)
 *********************************************************************************************************************/

/**
 * \brief   Initialises the control-loop layer: Transform_Init() + DFC_Init(),
 *          then resets the private controller state.  Command defaults
 *          (CtrlMode = CDDAPP_CTRL_OPENLOOP, SpeedRefRpm = 0) are set by
 *          CddApp_Init() in the central CddApp_T.
 *
 * \details Called by CddApp_Init() after CddGtm_Init().  The ISR never
 *          dispatches the DFC path before this has returned 0x1U.
 *
 * \return  0x1U on success; 0x0U if DFC_Init() failed.
 */
extern uint32_T CddGtm_CtrlInit(void);

/**
 * \brief   Returns the latest SMO mechanical speed estimate (closed-loop
 *          mode; returns 0 in open loop / after reset).
 * \return  Estimated speed  [RPM]
 */
extern real32_T CddGtm_GetSpeedRpm(void);

/**
 * \brief   Returns the DFC internal startup/run mode of the latest step.
 * \return  DFC_MODE_ALIGN / DFC_MODE_OPENLOOP / DFC_MODE_CLOSEDLOOP.
 */
extern DFC_Mode_T CddGtm_GetDfcMode(void);

/**
 * \brief   Copies the latest DFC diagnostic snapshot (VDq, Idq, angle, sector,
 *          TLoadHat) for CAN telemetry or debugger inspection.
 * \param[out] Diag_P  Destination (must not be NULL).
 * \return  0x1U on success, 0x0U on NULL pointer.
 */
extern uint32_T CddGtm_GetDfcDiagnostics(DFC_Diag_T * const Diag_P);

#endif /* CDD_GTM_APP_H_ */
