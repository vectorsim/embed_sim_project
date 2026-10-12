/**********************************************************************************************************************
 * \file      cdd_task_handler_app.h
 * \brief     Public interface for the software task scheduler driven by the STM 1 ms tick.
 *
 * \details   Provides the System_Tick_Handler() entry point called from the
 *            STM0 Compare-0 ISR (cdd_stm_app.c) every 1 ms, and declares the
 *            four periodic application task hooks:
 *
 *              Task_1ms()    — called every      1 ms  (1000 Hz)
 *              Task_10ms()   — called every     10 ms  ( 100 Hz)
 *              Task_100ms()  — called every    100 ms  (  10 Hz)
 *              Task_1s()     — called every  1 000 ms  (   1 Hz)
 *
 *            Task bodies are defined in cdd_task_handler_app.c.
 *            Add application logic there — do not modify the STM driver.
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
 * \note      MISRA C:2012 compliance notes for this module:
 *              - Rule 8.5 / 8.6 : External declarations are provided only in this
 *                header; definitions reside in the corresponding .c file.
 *              - Rule 17.2     : No recursion.
 *
 * \version   1.0.0
 * \date      2026-10-04
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_TASK_HANDLER_APP_H_
#define CDD_TASK_HANDLER_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   System tick dispatcher — called from Stm_00_Cmp_00_Isr every 1 ms.
 *
 * \details Runs the rate-divider counters and calls the appropriate task
 *          functions.  Must complete within one 1 ms tick budget.
 *          Must NOT be called from any context other than the STM ISR.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 *
 * \see     Stm_00_Cmp_00_Isr
 * \see     Task_1ms
 * \see     Task_10ms
 * \see     Task_100ms
 * \see     Task_1s
 */
extern void System_Tick_Handler(void);

/**
 * \brief   1 ms task — highest rate application work.
 *
 * \details Typical use: FOC current loop, ADC readout, IPC queue poll.
 *          Budget: must complete well within 1 ms.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void Task_1ms(void);

/**
 * \brief   10 ms task — medium rate application work.
 *
 * \details Typical use: speed controller, IPC message dispatch, state machine.
 *          Budget: must complete within 10 ms cumulative slot.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void Task_10ms(void);

/**
 * \brief   100 ms task — low rate application work.
 *
 * \details Typical use: thermal monitoring, diagnostics, parameter update.
 *          Budget: must complete within 100 ms cumulative slot.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void Task_100ms(void);

/**
 * \brief   1 s task — housekeeping and telemetry.
 *
 * \details Typical use: watchdog service, logging, heartbeat LED, IPC status.
 *          Budget: must complete within 1 s cumulative slot.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void Task_1s(void);

#endif /* CDD_TASK_HANDLER_APP_H_ */
