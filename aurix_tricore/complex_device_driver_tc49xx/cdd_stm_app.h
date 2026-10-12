/**********************************************************************************************************************
 * \file      cdd_stm_app.h
 * \brief     Public interface for the System Timer Module (STM) driver.
 *
 * \details   Configures the STM of CPU0 Compare-0 to generate a periodic
 *            1 ms system tick interrupt.  The ISR calls System_Tick_Handler()
 *            owned by cdd_task_handler_app.c.
 *
 *            Target device family: AURIX TC4x (TriCore 1.8).
 *            On TC4x the STM register block is integrated into the CPU module
 *            and is accessed through MODULE_CPU0.STM.*, not through a
 *            standalone Ifx_STM* module.
 *
 *            Hardware note (TC4x Reference Manual, STM chapter):
 *              STM_ABS is a single 64-bit atomic counter.
 *              CddStm_GetTime() reads ABS in one access; there is no
 *              TIM0/CAP latch pair as on TC3xx.
 *
 *            Typical non-blocking timeout pattern:
 *            \code
 *                uint64_T deadline = CddStm_GetDeadline(CDD_STM_TIME_CONST_10MS);
 *                while (CddStm_IsDeadlineElapsed(deadline) == 0x0U) { ; }
 *            \endcode
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
 *              - Rule 2.1       : No dead code in the public interface.
 *
 *            MISRA C:2012 deviations:
 *              - DEV-STM-001  Rule 8.4 : Stm_00_Cmp_00_Isr has no matching extern
 *                             declaration; installed exclusively via IFX_INTERRUPT().
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

#ifndef CDD_STM_APP_H_
#define CDD_STM_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Total number of time-constant entries in CddStm_TimeTable_G.
 */
#define CDD_STM_TIMER_COUNT         (11U)

#define CDD_STM_TIMER_INDEX_10NS    (0U)    /**< Index — 10 ns time constant          */
#define CDD_STM_TIMER_INDEX_100NS   (1U)    /**< Index — 100 ns time constant         */
#define CDD_STM_TIMER_INDEX_1US     (2U)    /**< Index — 1 µs time constant           */
#define CDD_STM_TIMER_INDEX_10US    (3U)    /**< Index — 10 µs time constant          */
#define CDD_STM_TIMER_INDEX_100US   (4U)    /**< Index — 100 µs time constant         */
#define CDD_STM_TIMER_INDEX_1MS     (5U)    /**< Index — 1 ms time constant (ISR arm) */
#define CDD_STM_TIMER_INDEX_10MS    (6U)    /**< Index — 10 ms time constant          */
#define CDD_STM_TIMER_INDEX_100MS   (7U)    /**< Index — 100 ms time constant         */
#define CDD_STM_TIMER_INDEX_1S      (8U)    /**< Index — 1 s time constant            */
#define CDD_STM_TIMER_INDEX_10S     (9U)    /**< Index — 10 s time constant           */
#define CDD_STM_TIMER_INDEX_100S    (10U)   /**< Index — 100 s time constant          */

/**
 * \brief   Convenience accessors for CddStm_TimeTable_G entries [STM ticks].
 *
 * \details Allocated and populated in cdd_stm_app.c by CddStm_InitTimeTable().
 *          Downstream translation units access individual entries via these
 *          macros only.
 *
 * \note    MISRA C:2012 Directive 4.9 – function-like / object-like macros;
 *          the underlying array is a single definition in the .c file.
 */
#define CDD_STM_TIME_CONST_10NS  (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10NS])    /**< 10 ns  [STM ticks] */
#define CDD_STM_TIME_CONST_100NS (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100NS])   /**< 100 ns [STM ticks] */
#define CDD_STM_TIME_CONST_1US   (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_1US])     /**< 1 µs   [STM ticks] */
#define CDD_STM_TIME_CONST_10US  (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10US])    /**< 10 µs  [STM ticks] */
#define CDD_STM_TIME_CONST_100US (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100US])   /**< 100 µs [STM ticks] */
#define CDD_STM_TIME_CONST_1MS   (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_1MS])     /**< 1 ms   [STM ticks] */
#define CDD_STM_TIME_CONST_10MS  (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10MS])    /**< 10 ms  [STM ticks] */
#define CDD_STM_TIME_CONST_100MS (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100MS])   /**< 100 ms [STM ticks] */
#define CDD_STM_TIME_CONST_1S    (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_1S])      /**< 1 s    [STM ticks] */
#define CDD_STM_TIME_CONST_10S   (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10S])     /**< 10 s   [STM ticks] */
#define CDD_STM_TIME_CONST_100S  (CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100S])    /**< 100 s  [STM ticks] */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Pre-computed STM tick counts for each time constant [STM ticks].
 *
 * \details Populated once in CddStm_InitTimeTable() from the live fSTM
 *          frequency.  Accessed exclusively via the CDD_STM_TIME_CONST_xxx
 *          macros.
 */
extern uint64_T CddStm_TimeTable_G[CDD_STM_TIMER_COUNT];

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
 * \brief   Initialises the STM of CPU0 and arms Compare-0 for a 1 ms period.
 *
 * \details Computes the full time-constant table from the live fSTM frequency,
 *          configures CMCON / STM_ICR / SRC registers, then loads the first
 *          compare value.  Must be called once during system init before
 *          global interrupt enable.
 *
 * \return  void
 *
 * \post    Compare-0 interrupt is armed; the first ISR fires after 1 ms.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 *
 * \see     Stm_00_Cmp_00_Isr
 * \see     System_Tick_Handler
 */
extern void CddStm_Init(void);

/**
 * \brief   Returns the lower 32 bits of the STM free-running counter.
 *
 * \details On TC4x this reads MODULE_CPU0.STM.ABS.U and truncates to 32 bits.
 *          Provided for call sites that want a cheap snapshot without
 *          passing a 64-bit value around.
 *
 * \return  Current lower 32-bit STM tick count [STM ticks].
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern uint32_T CddStm_GetTimeLow(void);

/**
 * \brief   Returns the full 64-bit STM free-running counter value.
 *
 * \details On TC4x STM_ABS is a single atomic 64-bit register; no TIM0/CAP
 *          capture sequence is required (that was the TC3xx mechanism).
 *
 * \return  Current 64-bit STM tick count [STM ticks].
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 *          MISRA DEV-STM-003 (Rule 10.5): hardware register field cast is
 *          intentional; see implementation.
 */
extern uint64_T CddStm_GetTime(void);

/**
 * \brief   Computes an absolute deadline from a relative timeout.
 *
 * \param[in] TimeOut  Timeout in STM ticks.  Use CDD_STM_TIME_CONST_xxx macros.
 *
 * \return  Absolute 64-bit STM tick count at which the deadline expires.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern uint64_T CddStm_GetDeadline(uint64_T TimeOut);

/**
 * \brief   Tests whether an absolute deadline has been reached or exceeded.
 *
 * \param[in] Deadline  Absolute STM tick deadline from CddStm_GetDeadline().
 *
 * \return  0x1U if current time > Deadline, 0x0U otherwise.
 *
 * \note    MISRA C:2012 Rule 14.4 – result is encoded as explicit 0x0U / 0x1U
 *          to avoid implicit Boolean-to-integer conversion.
 *          MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern uint32_T CddStm_IsDeadlineElapsed(uint64_T Deadline);

/**
 * \brief   Busy-wait delay of the specified number of microseconds.
 *
 * \details Computes an absolute deadline as
 *              now + (Microseconds × CDD_STM_TIME_CONST_1US)
 *          and spins until CddStm_IsDeadlineElapsed() returns 0x1U.
 *
 *          Minimum resolution is one STM tick (~2 ns at 500 MHz on TC49x).
 *          Passing 0 returns immediately without blocking.
 *
 * \pre     CddStm_Init() must have been called so that CddStm_TimeTable_G
 *          is populated before this function is used.
 *
 * \param[in] Microseconds  Delay duration [µs], must be >= 0.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in this header only.
 */
extern void CddStm_Delay_Us(uint32_T Microseconds);

#endif /* CDD_STM_APP_H_ */
