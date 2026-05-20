/**********************************************************************************************************************
 * \file        cdd_gtm_app.h
 * \brief       GTM ATOM0 direct 6-channel driver interface for 3-phase FOC PWM
 *              generation on the AP32541 motor control board (TC38x).
 *
 * \details     Architecture (all TOUTSEL values from TC38x UM appx1):
 *
 *              ATOM0_CH0  — Master PWM carrier, CCU1 ISR at half-period → CPU
 *                           TOUTSEL1.SEL1 = 0x02 → CDTM0_DTM4_0 → TOUT9 / P00.0
 *
 *              ATOM0_CH1  — Phase U LS   IL1  P00.2  active HIGH  SL=0  SOMP slave
 *                           TOUTSEL1.SEL3 = 0x02 → CDTM0_DTM4_1 → TOUT11
 *              ATOM0_CH2  — Phase U HS  /IH1  P00.3  active LOW   SL=0  SOMP slave
 *                           TOUTSEL1.SEL4 = 0x02 → CDTM0_DTM4_2 → TOUT12
 *
 *              ATOM0_CH3  — Phase V LS   IL2  P00.4  active HIGH  SL=0  SOMP slave
 *                           TOUTSEL1.SEL5 = 0x02 → CDTM0_DTM4_3 → TOUT13
 *              ATOM0_CH4  — Phase V HS  /IH2  P00.5  active LOW   SL=0  SOMP slave
 *                           TOUTSEL1.SEL6 = 0x02 → CDTM0_DTM5_0 → TOUT14
 *
 *              ATOM0_CH5  — Phase W LS   IL3  P00.6  active HIGH  SL=0  SOMP slave
 *                           TOUTSEL1.SEL7 = 0x02 → CDTM0_DTM5_1 → TOUT15
 *              ATOM0_CH6  — Phase W HS  /IH3  P00.7  active LOW   SL=0  SOMP slave
 *                           TOUTSEL2.SEL0 = 0x02 → CDTM0_DTM5_2 → TOUT16
 *
 *              ATOM0_CH7  — ADC trigger (valley-aligned, internal only)
 *                           CCU1 → ADCTRIG0 → EVADC G0/G1/G2 (phase currents)
 *
 *              GTM_Set_PWM_Duty() is STATIC (internal to cdd_gtm_app.c) — it takes
 *              duty cycles from CDD_App_G.DutyU/V/W.  It is not part of the public
 *              API.  The retired GTM_PWM_Duty_T struct has been removed.
 *
 *              Open-loop V/f:
 *                  GTM_OpenLoop_Set_RPM(rpm, mi) — arms open-loop at target speed
 *                  GTM_OpenLoop_Stop()           — disarms, returns to 50% duty
 *                  GTM_OpenLoop_Run()            — static, called from ISR only;
 *                                                  all state in OL_State_G (file scope
 *                                                  in cdd_gtm_app.c).  Remove this
 *                                                  function block to switch to FOC.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gtm_app.c
 *              - Rule 17.2 : No recursion
 *
 * \version     1.1.0
 * \date        2025-05-18
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
#include "cdd_config.h"

/**********************************************************************************************************************
 * Function Prototypes — Initialisation
 *********************************************************************************************************************/

/**
 * \brief   Initialises GTM CMU, ATOM0 channels CH0–CH7, CDTM0 DTM4/DTM5, and pin mux.
 *
 * \details Sequence:
 *              1. Compute Period_Ticks_G / Half_Period_Ticks_G / Sample_Time_G
 *              2. ATOM0_CH0  master SOMP carrier, CCU1 ISR armed (SRE=0 — not yet enabled)
 *              3. ATOM0_CH1–CH6  Phase U/V/W LS+HS (SL safe-off initial state)
 *              4. ATOM0_CH7  ADC valley trigger → ADCTRIG0 → EVADC G0/G1/G2
 *              5. CDTM0_DTM4/5  CLK_SEL = CMU CLK0, passthrough (zero DTV)
 *              6. TOUTSEL + GPIO_Configure_GTM_<phase>_<Pxx_y>() per pin
 *              7. Write 50% duty to CDD_App_G.DutyU/V/W → shadow registers
 *              8. Initialise OL_State_G (open-loop state, active = 0)
 *
 *          HOST_TRIG (PWM carrier start) is NOT issued here.
 *          Call Start_GTM_Module() after Initialize_Inverter() to start PWM.
 *
 * \return  None
 */
extern void Initialize_GTM_Module(void);

/**
 * \brief   Issues HOST_TRIG — transfers shadow registers to active compare registers
 *          and starts the ATOM0 carrier.  PWM is live after this call.
 *
 * \details Must be called after Initialize_GTM_Module() and Initialize_Inverter().
 *          Call GD9180_Enable_Outputs() and then arm the ISR (SRC SRE=1) after this.
 *
 * \return  None
 */
extern void Start_GTM_Module(void);

/**********************************************************************************************************************
 * Function Prototypes — Open-Loop V/f Test
 *
 * These two functions are the ONLY open-loop entry points visible outside
 * cdd_gtm_app.c.  GTM_OpenLoop_Run() is static and lives entirely inside
 * cdd_gtm_app.c together with OL_State_G and the GTM_OL_State_T typedef.
 * To switch to closed-loop FOC: remove GTM_OpenLoop_Set_RPM(), GTM_OpenLoop_Stop(),
 * and the GTM_OpenLoop_Run() block from cdd_gtm_app.c.
 *********************************************************************************************************************/

/**
 * \brief   Arms the open-loop V/f test at the requested mechanical speed and
 *          modulation index.
 *
 * \details Sets omega_e, mi, resets theta to 0, sets active=1.
 *          The next ISR tick will call GTM_OpenLoop_Run() and the motor will
 *          see sinusoidal voltages immediately.
 *
 *          For DB42S02 at 3000 RPM start: Rpm=3000, Mi=0.3F.
 *          omega_e = Rpm * pi/30 * CDD_MOTOR_POLE_PAIRS  [rad_e/s]
 *
 * \param   Rpm   Target mechanical speed  [RPM,  range 0 .. CDD_OL_MAX_RPM]
 * \param   Mi    Modulation index         [0.0 .. 1.0]
 * \return  None
 */
extern void GTM_OpenLoop_Set_RPM(uint32_T Rpm, real32_T Mi);

/**
 * \brief   Disarms the open-loop controller and returns all three phases to 50% duty.
 *
 * \details Sets active=0, zeros omega_e/mi/theta, writes 50% to CDD_App_G.DutyU/V/W,
 *          and calls GTM_Set_PWM_Duty() to flush the safe state to hardware.
 *
 * \return  None
 */
extern void GTM_OpenLoop_Stop(void);

/**********************************************************************************************************************
 * Function Prototypes — Accessors
 *********************************************************************************************************************/

/**
 * \brief   Returns the control loop carrier period in CMU CLK0 ticks.
 * \return  Period ticks  [CLK0 ticks]
 */
extern uint32_T GTM_Get_Period_Ticks(void);

/**
 * \brief   Returns the control loop sample time in seconds.
 * \return  Sample time  [s]
 */
extern real32_T GTM_Get_Sample_Time(void);

#endif /* CDD_GTM_APP_H_ */
