/**********************************************************************************************************************
 * \file      cdd_task_handler_app.c
 * \brief     Implementation of the software task scheduler and periodic application task stubs.
 *
 * \details   System_Tick_Handler() is called from the STM0 Compare-0 ISR
 *            (cdd_stm_app.c) every 1 ms.  It maintains three rate-divider
 *            counters and explicitly calls the task functions at the
 *            appropriate rates:
 *
 *              Counter             Reload            Task
 *              ──────────────────  ────────────────  ────────────
 *              Tick_10ms_Cnt_G     TICK_10MS_RELOAD  Task_10ms()
 *              Tick_100ms_Cnt_G    TICK_100MS_RELOAD Task_100ms()
 *              Tick_1s_Cnt_G       TICK_1S_RELOAD    Task_1s()
 *
 *            Function pointer table is intentionally NOT used — explicit
 *            calls are statically traceable by TASKING and MISRA checkers.
 *
 *            Application task bodies (Task_1ms .. Task_1s) are stubbed below.
 *            Add motor control, IPC queue processing, diagnostics, etc. in
 *            the appropriate task slot.
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
 * \note      MISRA C:2012 compliance overview for this translation unit:
 *              - Rule  8.9  : File-scope counters limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit uint32_T comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
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


/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_task_handler_app.h"
#include "cdd_gpio_app.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Rate-divider reload values expressed in 1 ms ticks.
 */
#define TICK_10MS_RELOAD    (10U)       /**< 1 ms ticks per 10 ms period    */
#define TICK_100MS_RELOAD   (100U)      /**< 1 ms ticks per 100 ms period   */
#define TICK_1S_RELOAD      (1000U)     /**< 1 ms ticks per 1 s period      */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Up-counter for 10 ms rate division [1 ms ticks].
 *
 * \note    MISRA C:2012 Rule 8.9 – file-scope variable limited to this TU.
 */
static uint32_T Tick_10ms_Cnt_G;

/**
 * \brief   Up-counter for 100 ms rate division [1 ms ticks].
 *
 * \note    MISRA C:2012 Rule 8.9 – file-scope variable limited to this TU.
 */
static uint32_T Tick_100ms_Cnt_G;

/**
 * \brief   Up-counter for 1 s rate division [1 ms ticks].
 *
 * \note    MISRA C:2012 Rule 8.9 – file-scope variable limited to this TU.
 */
static uint32_T Tick_1s_Cnt_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   System tick dispatcher — called from Stm_00_Cmp_00_Isr every 1 ms.
 *
 * \details Execution order each tick:
 *            -# Task_1ms() — always.
 *            -# Increment Tick_10ms_Cnt_G; on reload call Task_10ms().
 *            -# Increment Tick_100ms_Cnt_G; on reload call Task_100ms().
 *            -# Increment Tick_1s_Cnt_G; on reload call Task_1s().
 *
 *          Must complete within one 1 ms tick budget.
 *          Must NOT be called from any context other than the STM ISR.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 14.4 – all if-conditions use explicit comparison
 *          against the reload constants.
 *          MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 *
 * \see     Stm_00_Cmp_00_Isr
 * \see     Task_1ms
 * \see     Task_10ms
 * \see     Task_100ms
 * \see     Task_1s
 */
void System_Tick_Handler(void)
{
    /* ------------------------------------------------------------------ */
    /* 1 ms — every tick                                                   */
    /* ------------------------------------------------------------------ */
    Task_1ms();

    /* ------------------------------------------------------------------ */
    /* 10 ms                                                               */
    /* ------------------------------------------------------------------ */
    Tick_10ms_Cnt_G++;
    if (Tick_10ms_Cnt_G >= TICK_10MS_RELOAD)
    {
        Tick_10ms_Cnt_G = 0U;
        Task_10ms();
    }

    /* ------------------------------------------------------------------ */
    /* 100 ms                                                              */
    /* ------------------------------------------------------------------ */
    Tick_100ms_Cnt_G++;
    if (Tick_100ms_Cnt_G >= TICK_100MS_RELOAD)
    {
        Tick_100ms_Cnt_G = 0U;
        Task_100ms();
    }

    /* ------------------------------------------------------------------ */
    /* 1 s                                                                 */
    /* ------------------------------------------------------------------ */
    Tick_1s_Cnt_G++;
    if (Tick_1s_Cnt_G >= TICK_1S_RELOAD)
    {
        Tick_1s_Cnt_G = 0U;
        Task_1s();
    }
}

/**
 * \brief   1 ms task — highest rate application work (1000 Hz).
 *
 * \details Typical use: FOC current loop, ADC readout, IPC queue poll.
 *          Budget: must complete well within 1 ms.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
void Task_1ms(void)
{
    /* TODO: add 1 ms application logic here */
}

/**
 * \brief   10 ms task — medium rate application work (100 Hz).
 *
 * \details Typical use: speed / position controller, IPC message dispatch,
 *          state machine.
 *          Budget: must complete within 10 ms cumulative slot.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
void Task_10ms(void)
{
    /* TODO: add 10 ms application logic here */
}

/**
 * \brief   100 ms task — low rate application work (10 Hz).
 *
 * \details Typical use: thermal monitoring, diagnostics, parameter update.
 *          Budget: must complete within 100 ms cumulative slot.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
void Task_100ms(void)
{
    /* TODO: add 100 ms application logic here */
}

/**
 * \brief   1 s task — housekeeping and telemetry (1 Hz).
 *
 * \details Toggles the heartbeat LED on P03.10.
 *          Typical additional use: watchdog service, logging, IPC status.
 *          Budget: must complete within 1 s cumulative slot.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 *
 * \see     CddGpio_Toggle_P03_10_Led_HeartBeat
 */
void Task_1s(void)
{
    CddGpio_Toggle_P03_10_Led_HeartBeat();
}
