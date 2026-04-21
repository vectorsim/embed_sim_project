/**********************************************************************************************************************
 * \file        cdd_task_handler_app.c
 * \brief       Implementation of cdd_task_handler_app.h — software task scheduler
 *              and periodic application task stubs.
 *
 * \details     System_Tick_Handler() is called from the STM0 Compare-0 ISR
 *              (cdd_stm_app.c) every 1 ms.  It maintains three rate-divider
 *              counters and explicitly calls the task functions at the
 *              appropriate rates:
 *
 *                  Counter             Reload          Task
 *                  ──────────────────  ──────────────  ────────────
 *                  Tick_10ms_Cnt_G     TICK_10MS_RELOAD  Task_10ms()
 *                  Tick_100ms_Cnt_G    TICK_100MS_RELOAD Task_100ms()
 *                  Tick_1s_Cnt_G       TICK_1S_RELOAD    Task_1s()
 *
 *              Function pointer table is intentionally NOT used here — explicit
 *              calls are statically traceable by TASKING and MISRA checkers.
 *
 *              Application task bodies (Task_1ms .. Task_1s) are stubbed below.
 *              Add motor control, IPC queue processing, diagnostics, etc. in
 *              the appropriate task slot.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope counters limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit uint32_T comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_task_handler_app.h"
#include "cdd_gpio_app.h"

/**********************************************************************************************************************
 * Private Macros — Rate-Divider Reload Values
 *********************************************************************************************************************/
#define TICK_10MS_RELOAD    (10U)       /**< \brief 1 ms ticks per 10 ms period    */
#define TICK_100MS_RELOAD   (100U)      /**< \brief 1 ms ticks per 100 ms period   */
#define TICK_1S_RELOAD      (1000U)     /**< \brief 1 ms ticks per 1 s period      */

/**********************************************************************************************************************
 * Private Variables — Rate-Divider Counters
 *********************************************************************************************************************/

/** \brief  Up-counter for 10 ms rate division   [1 ms ticks]                  */
static uint32_T Tick_10ms_Cnt_G;

/** \brief  Up-counter for 100 ms rate division  [1 ms ticks]                  */
static uint32_T Tick_100ms_Cnt_G;

/** \brief  Up-counter for 1 s rate division     [1 ms ticks]                  */
static uint32_T Tick_1s_Cnt_G;

/**********************************************************************************************************************
 * Public Function Implementations — Scheduler Entry
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * System_Tick_Handler  —  called from STM0 ISR every 1 ms
 *------------------------------------------------------------------------------------------------------------------*/
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

/**********************************************************************************************************************
 * Public Function Implementations — Application Task Stubs
 *
 * Add application logic below.  Each task must complete within its time budget.
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Task_1ms  — 1000 Hz   (budget: < 1 ms)
 * Typical:  FOC current loop, ADC readout, IPC queue poll
 *------------------------------------------------------------------------------------------------------------------*/
void Task_1ms(void)
{
    /* TODO: add 1 ms application logic here */
}

/*--------------------------------------------------------------------------------------------------------------------
 * Task_10ms  — 100 Hz   (budget: < 10 ms cumulative)
 * Typical:  speed / position controller, IPC message dispatch, state machine
 *------------------------------------------------------------------------------------------------------------------*/
void Task_10ms(void)
{
    /* TODO: add 10 ms application logic here */
}

/*--------------------------------------------------------------------------------------------------------------------
 * Task_100ms  — 10 Hz   (budget: < 100 ms cumulative)
 * Typical:  thermal monitoring, diagnostics, parameter update
 *------------------------------------------------------------------------------------------------------------------*/
void Task_100ms(void)
{
    /* TODO: add 100 ms application logic here */
}

/*--------------------------------------------------------------------------------------------------------------------
 * Task_1s  — 1 Hz   (budget: < 1 s cumulative)
 * HeartBeat LED
 *------------------------------------------------------------------------------------------------------------------*/
void Task_1s(void)
{
    GPIO_Toggle_LED_P33_4();
}
