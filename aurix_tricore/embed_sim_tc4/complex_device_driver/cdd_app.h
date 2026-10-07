/**********************************************************************************************************************
 * \file      cdd_app.h
 * \brief     Public interface for the CddSys low-level system utility module.
 *
 * \details   Declares the CddSys function API used across the CDD layer for
 *            TriCore-specific primitives:
 *              - Atomic bit-field load-modify-store (LDMST instruction).
 *              - Binary semaphore via compare-and-swap (CMPSWAP.W instruction).
 *              - Single and multi-instruction NOP delays.
 *              - Interrupt-enable intrinsic.
 *
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCasePtr
 *              - Local variables: lowerPascalCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.0.0
 * \date      2026-10-04
 * \author   Paul Abraham
 *             EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"
/*************************************
 * ********************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
#define CTRL_ISR_PRIORITY         (20U)    /* Inverter & Control Loop */
#define CTRL_ISR_TOS              (0U)     /* CPU 0                   */
#define TIM0CH4_ISR_PRIORITY      (25U)    /*  Fast ShutOff           */
#define TIM0CH4_ISR_TOS           (0U)     /*  CPU 0                  */

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Central runtime state of the inverter control application.
 *
 * \details Single global instance (Inv_G) shared between the control-loop ISR
 *          (ControlLoop_ISR), the fast shut-off ISR (TIM0_CH4_ISR), and the
 *          application layer. Written by the application, read/written by
 *          ControlLoop_ISR, and partially written by TIM0_CH4_ISR.
 *
 * \note    All members are accessed from ISR context. Updates that must be
 *          seen by the control loop as a coherent set (for example the three
 *          duty cycles together with a FrequencyChangeRequest) should be
 *          staged with interrupts disabled, or via CddSys_Ldmst.
 */
typedef struct
{
    uint32_T   PeriodTicks;             /**< PWM period in eGTM CMU_CLK0 ticks.
                                             Derived from CmuClk0Frequency and
                                             FrequencyMode by
                                             CddEgtm_SetInverterFrequency().
                                             Programmed into the master
                                             (ATOM CH0) shadow register SR0.   */

    uint32_T   DeadTimeTicks;           /**< Dead time applied to each
                                             complementary phase output
                                             (U/V/W), in CMU_CLK0 ticks.
                                             Written to the DTM5 channel DTV
                                             registers (RELRISE / RELFALL) at
                                             init.                              */

    real32_T   CmuClk0Frequency;        /**< Frequency of the eGTM cluster-0
                                             CMU clock 0 (CMU_CLK0) in Hz.
                                             Captured once at init from
                                             CddSys_GetEgtmCmu0Frequency() and
                                             used as the time base for
                                             PeriodTicks.                       */

    real32_T   DutyCycleU;              /**< Phase U duty cycle, normalised to
                                             [0, +1.0]. Signed value: the
                                             sign selects which side of the
                                             half-bridge is active.
                                             Consumed by CAL_PHASE_PERIOD /
                                             CAL_PHASE_DUTY_CYCLE in
                                             ControlLoop_ISR.                    */

    real32_T   DutyCycleV;              /**< Phase V duty cycle, normalised to
                                             [0, +1.0]. Same convention as
                                             DutyCycleU.                        */

    real32_T   DutyCycleW;              /**< Phase W duty cycle, normalised to
                                             [0, +1.0]. Same convention as
                                             DutyCycleU.                        */

    uint64_T   Counter;                 /**< Free-running PWM cycle counter.
                                             Incremented once per control-loop
                                             iteration (every ATOM CH0 CCU0
                                             event). 64-bit so it does not
                                             wrap within any realistic
                                             mission time at 30 kHz.            */

    uint32_T   ShutOffState;            /**< Latched shut-off state.
                                             - 0x0U : inverter running normally.
                                             - 0x1U : a shut-off has been
                                                      latched by TIM0_CH4_ISR
                                                      and the CDTM5 dead-time
                                                      module is holding the
                                                      phase outputs in their
                                                      safe state.
                                             Cleared by ControlLoop_ISR once
                                             ShutOffRstRequest is set.          */

    uint32_T   ShutOffRstRequest;       /**< Application request to leave the
                                             shut-off state.
                                             - 0x0U : no request pending.
                                             - 0x1U : request pending; the
                                                      control loop will clear
                                                      ShutOffState and pulse
                                                      EGTM_CLS0_CDTM_DTM5_CTRL.
                                                      SHUT_OFF_RST on the next
                                                      CCU0 event.                */

    uint32_T   FrequencyMode;           /**< Selected PWM frequency mode.
                                             - 1U : 10 kHz
                                             - 2U : 20 kHz
                                             - 3U : 30 kHz
                                             Any other value falls back to
                                             10 kHz in
                                             CddEgtm_SetInverterFrequency().    */

    uint32_T   FrequencyChangeRequest;  /**< Application request to reprogram
                                             the master period mid-run.
                                             - 0x0U : no request pending.
                                             - 0x1U : on the next CCU1 event,
                                                      PeriodTicks is written to
                                                      ATOM CH0 SR0/SR1, a host
                                                      trigger is issued to latch
                                                      the shadow registers, and
                                                      this flag is cleared.       */
} CddInverterCtrl_T;


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
extern CddInverterCtrl_T  Inv_G;

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialises the CDD layer.
 *
 * \details Entry point for CDD-level initialisation. Currently forwards to
 *          the EGTM module initialiser, which configures the eGTM clock tree,
 *          CMU clock 0, the TIM0 channel 4 fast shut-off path, and the
 *          ATOM CLS0 PWM channels used to drive the three inverter phases.
 *
 *          Called once during start-up, before Cdd_Start() and before any
 *          ISR that depends on Inv_G has been enabled.
 *
 * \return  void
 *
 * \pre     The system clock tree is running.
 * \post    Inv_G is fully initialised and the eGTM is configured but not yet
 *          triggered (outputs remain inactive until Cdd_Start()).
 *
 * \see     CddEgtm_InitModule
 * \see     Cdd_Start
 */
extern void Cdd_Init(void);

/**
 * \brief   Starts the CDD layer.
 *
 * \details Issues the eGTM host trigger that synchronously starts all enabled
 *          ATOM channels. From this point the master channel (ATOM CH0) runs
 *          the control-loop ISR at the configured inverter frequency, and the
 *          three PWM phases (ATOM CH5/CH6/CH7) produce the complementary
 *          outputs with dead time.
 *
 * \return  void
 *
 * \pre     Cdd_Init() has completed successfully.
 * \post    PWM generation and the control-loop ISR are active.
 *
 * \see     Cdd_Init
 * \see     CddEgtm_Start
 */
extern void Cdd_Start(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_APP_H_ */
