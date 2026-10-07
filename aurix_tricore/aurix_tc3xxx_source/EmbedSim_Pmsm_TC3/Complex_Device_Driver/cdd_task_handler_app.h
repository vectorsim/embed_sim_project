/**********************************************************************************************************************
 * \file        cdd_task_handler_app.h
 * \brief       Software task scheduler driven by the STM0 1 ms system tick.
 *
 * \details     Provides the System_Tick_Handler() entry point called from the
 *              STM0 Compare-0 ISR (cdd_stm_app.c) every 1 ms, and declares
 *              the four periodic application task hooks:
 *
 *                  Task_1ms()    — called every      1 ms  (1000 Hz)
 *                  Task_10ms()   — called every     10 ms  ( 100 Hz)
 *                  Task_100ms()  — called every    100 ms  (  10 Hz)
 *                  Task_1s()     — called every  1 000 ms  (   1 Hz)
 *
 *              Task bodies are defined in cdd_task_handler_app.c.
 *              Add application logic there — do not modify the STM driver.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function, matching .c definition
 *              - Rule  8.6 : All definitions reside in cdd_task_handler_app.c
 *              - Rule 17.2 : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_TASK_HANDLER_APP_H_
#define CDD_TASK_HANDLER_APP_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"


/**********************************************************************************************************************
 * Function Prototypes — Scheduler Entry (called from STM ISR)
 *********************************************************************************************************************/

/**
 * \brief   System tick dispatcher — called from Stm_00_Cmp_00_Isr every 1 ms.
 *
 * \details Runs the rate-divider counters and calls the appropriate task
 *          functions.  Must complete within one 1 ms tick budget.
 *          Must NOT be called from any context other than the STM ISR.
 *
 * \return  None
 */
extern void System_Tick_Handler(void);

/**********************************************************************************************************************
 * Function Prototypes — Periodic Application Task Hooks
 *********************************************************************************************************************/

/**
 * \brief   1 ms task — highest rate application work.
 *
 * \details Typical use: FOC current loop, ADC readout, IPC queue poll.
 *          Budget: must complete well within 1 ms.
 *
 * \return  None
 */
extern void Task_1ms(void);

/**
 * \brief   10 ms task — medium rate application work.
 *
 * \details Typical use: speed controller, IPC message dispatch, state machine.
 *          Budget: must complete within 10 ms cumulative slot.
 *
 * \return  None
 */
extern void Task_10ms(void);

/**
 * \brief   100 ms task — low rate application work.
 *
 * \details Typical use: thermal monitoring, diagnostics, parameter update.
 *          Budget: must complete within 100 ms cumulative slot.
 *
 * \return  None
 */
extern void Task_100ms(void);

/**
 * \brief   1 s task — housekeeping and telemetry.
 *
 * \details Typical use: watchdog service, logging, heartbeat LED, IPC status.
 *          Budget: must complete within 1 s cumulative slot.
 *
 * \return  None
 */
extern void Task_1s(void);

#endif /* CDD_TASK_HANDLER_APP_H_ */
