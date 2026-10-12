/**********************************************************************************************************************
 * \file      cdd_stm_app.c
 * \brief     Implementation of the System Timer Module (STM) driver (AURIX TC4x).
 *
 * \details   Responsibilities:
 *              1. Build the time-constant table from the live fSTM frequency.
 *              2. Configure the CPU0 STM Compare-0 for a 1 ms periodic interrupt.
 *              3. ISR: call System_Tick_Handler() then rearm the compare register.
 *
 *            ISR rearm strategy (drift-free):
 *                elapsed = now - entry_time          (wraps correctly, uint32)
 *                periods = (elapsed / 1ms_ticks) + 1
 *                next    = entry_time + periods * 1ms
 *
 *            Target device family: AURIX TC4x (TriCore 1.8).
 *            On TC4x the STM register block is integrated into the CPU module.
 *            There is no standalone Ifx_STM module and no IfxStm_reg.h header.
 *            All access goes through the CPU0 register base: MODULE_CPU0.STM.*
 *
 *            Register map used here (all in MODULE_CPU0.STM):
 *              - ABS.U            64-bit free-running counter (atomic read)
 *              - CMP[0].B.CMPVAL  Compare-0 value
 *              - CMCON.B.MSTART0  Compare-0 start bit position
 *              - CMCON.B.MSIZE0   Compare-0 compare width
 *              - STM_ICR.B.CMP0OS Compare-0 interrupt output select
 *              - STM_ICR.B.CMP0EN Compare-0 interrupt enable
 *              - ISCR.B.CMP0IRR   Compare-0 interrupt flag clear
 *
 *            The 64-bit counter is a single register; no TIM0/CAP latch
 *            pair as on TC3xx.
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
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *
 *            MISRA C:2012 deviations:
 *              - DEV-STM-001  Rule 8.4  : Stm_00_Cmp_00_Isr has no matching extern
 *                             declaration; installed via IFX_INTERRUPT() only.
 *              - DEV-STM-003  Rule 10.5 : Hardware register fields (.U) cast to uint64_T.
 *                             PRQA S 0303 applied at each tagged site.
 *              - DEV-STM-004  Rule 10.3 : (uint32_T)CDD_STM_TIME_CONST_1MS narrows
 *                             uint64_T to 32 bits; safe for fSTM <= 4.3 GHz.
 *                             PRQA S 4342 applied.
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
#include "cdd_stm_app.h"
#include "cdd_task_handler_app.h"
#include "cdd_sys_util.h"
#include "cdd_app.h"
#include "IfxCpu_reg.h"    /* MODULE_CPU0 and the Ifx_CPU_STM_* register types */
#include "IfxSrc_reg.h"    /* SRC_STMCPU0SR2 / SRC_STMCPU0SR3 */
#include "Bsp.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Pre-computed STM tick counts for each time constant [STM ticks].
 *
 * \details Populated once in CddStm_InitTimeTable() from the live fSTM frequency.
 *          Declared extern in cdd_stm_app.h; accessed via CDD_STM_TIME_CONST_xxx macros.
 */
uint64_T CddStm_TimeTable_G[CDD_STM_TIMER_COUNT];

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Private helper that builds the STM time-constant table.
 *
 * \details Converts the live fSTM frequency into tick counts for every
 *          CDD_STM_TIMER_INDEX_xxx entry and stores them in CddStm_TimeTable_G.
 *          Called only from CddStm_Init().
 *
 * \param[in] StmFreq  STM clock frequency in Hz.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 */
static void CddStm_InitTimeTable(uint64_T StmFreq);

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Interrupt vector table entry for STM Compare-0.
 *
 * \note    MISRA DEV-STM-001 (Rule 8.4): No extern declaration for this symbol
 *          in the header.  The ISR is installed exclusively through
 *          IFX_INTERRUPT() into the .traptab section.
 */
IFX_INTERRUPT(Stm_00_Cmp_00_Isr, 0x0, STM0_CMP0_IR_SRPN);

/**
 * \brief   STM Compare-0 ISR — fires every 1 ms.
 *
 * \details Dispatches to System_Tick_Handler() then rearms the compare
 *          register using a drift-free strategy:
 *            - Snapshot entry time.
 *            - Call System_Tick_Handler().
 *            - Compute elapsed ticks (uint32 modular arithmetic handles wrap).
 *            - periods = (elapsed / 1 ms ticks) + 1
 *            - nextCmp = entryTime + periods * 1 ms ticks
 *            - Write CMPVAL and clear the interrupt flag.
 *
 * \return  void
 *
 * \note    MISRA DEV-STM-004 (Rule 10.3): (uint32_T)CDD_STM_TIME_CONST_1MS
 *          narrows uint64_T.  Safe for fSTM <= 4.3 GHz.  PRQA S 4342.
 *          MISRA C:2012 Rule 8.7 – ISR has external linkage because it is
 *          referenced by the vector table.
 *
 * \see     System_Tick_Handler
 * \see     CddStm_Init
 */
void Stm_00_Cmp_00_Isr(void)
{
    uint32_T entryTime = CddStm_GetTimeLow();   /* snapshot at ISR entry   */
    uint32_T elapsed   = 0U;
    uint32_T periods   = 0U;
    uint32_T nextCmp   = 0U;

    System_Tick_Handler();

    elapsed  = CddStm_GetTimeLow() - entryTime;                               /* wraps safely — uint32 modular arithmetic  */
    periods  = (elapsed / (uint32_T)CDD_STM_TIME_CONST_1MS) + 1U;            /* PRQA S 4342 */ /* MD_STM_4342_CDD_STM_TIME_CONST_1MS_32bit */
    nextCmp  = entryTime + (periods * (uint32_T)CDD_STM_TIME_CONST_1MS);     /* PRQA S 4342 */ /* MD_STM_4342_CDD_STM_TIME_CONST_1MS_32bit */

    MODULE_CPU0.STM.CMP[0].B.CMPVAL = nextCmp;
    MODULE_CPU0.STM.ISCR.B.CMP0IRR  = 0x1U;   /* Clear Compare-0 interrupt flag */
}

/**
 * \brief   Initialises the STM of CPU0 and arms Compare-0 for a 1 ms period.
 *
 * \details Sequential bring-up:
 *            -# Read live fSTM frequency and build the time-constant table.
 *            -# Configure Compare-0 as 32-bit width starting at bit 0
 *               (MSTART0 = 0, MSIZE0 = 31).
 *            -# Select interrupt request on rising edge of the compare event
 *               (CMP0OS = IR0).
 *            -# Arm the service-request node (SRPN, TOS = CPU0, clear pending,
 *               enable SRE).
 *            -# Disarm by writing a compare value in the past during setup.
 *            -# Enable compare output (CMP0EN) and load the first real
 *               compare value (now + 1 ms).
 *
 * \return  void
 *
 * \post    Compare-0 interrupt is armed; the first ISR fires after 1 ms.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in cdd_stm_app.h.
 *          MISRA C:2012 Rule 8.9 – local variable stmFreq is defined at the
 *          narrowest scope.
 *          MISRA DEV-STM-004 (Rule 10.3): narrowing cast of
 *          CDD_STM_TIME_CONST_1MS is intentional.  PRQA S 4342.
 *
 * \see     CddStm_InitTimeTable
 * \see     Stm_00_Cmp_00_Isr
 */
void CddStm_Init(void)
{
    uint64_T stmFreq;

    stmFreq = (uint64_T)CddSys_GetStmFrequency();
    CddStm_InitTimeTable(stmFreq);

    /* Compare-0: 32-bit width starting at bit 0 (MSTART0 = 0, MSIZE0 = 31) */
    MODULE_CPU0.STM.CMCON.B.MSTART0 = 0x0U;
    MODULE_CPU0.STM.CMCON.B.MSIZE0  = 0x1FU;

    /* Compare-0 output: interrupt request on rising edge of the compare event */
    MODULE_CPU0.STM.STM_ICR.B.CMP0OS = 0x0U;   /* select IR0 */

    /* Service request node: arm SRC, enable SR */
    SRC_STMCPU0SR2.B.SRPN = STM0_CMP0_IR_SRPN;
    SRC_STMCPU0SR2.B.TOS  = 0x0U;   /* CPU0 */
    SRC_STMCPU0SR2.B.CLRR = 0x1U;   /* Clear any pending request */
    SRC_STMCPU0SR2.B.SRE  = 0x1U;   /* Enable service request    */

    /* Disarm: write a compare value firmly in the past during setup. */
    MODULE_CPU0.STM.CMP[0].B.CMPVAL = (CddStm_GetTimeLow() >> 0x1U);

    /* Enable compare output, then load the first real compare value. */
    MODULE_CPU0.STM.STM_ICR.B.CMP0EN = 0x1U;
    MODULE_CPU0.STM.CMP[0].B.CMPVAL  = CddStm_GetTimeLow() + (uint32_T)CDD_STM_TIME_CONST_1MS; /* PRQA S 4342 */
}

/**
 * \brief   Returns the full 64-bit STM free-running counter value.
 *
 * \details On TC4x STM_ABS is a single atomic 64-bit register; no TIM0/CAP
 *          capture sequence is required.
 *
 * \return  Current 64-bit STM tick count [STM ticks].
 *
 * \note    MISRA DEV-STM-003 (Rule 10.5 / PRQA S 0303):
 *          MODULE_CPU0.STM.ABS.U is a hardware register union.  Cast to
 *          uint64_T is intentional.  On TC4x the whole 64-bit counter is
 *          returned in one atomic read.
 *          MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
uint64_T CddStm_GetTime(void)
{
    uint64_T now;

    now = (uint64_T)MODULE_CPU0.STM.ABS.U; /* PRQA S 0303 */

    return now;
}

/**
 * \brief   Returns the lower 32 bits of the STM free-running counter.
 *
 * \details On TC4x this reads MODULE_CPU0.STM.ABS.U and truncates to 32 bits.
 *
 * \return  Current lower 32-bit STM tick count [STM ticks].
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
uint32_T CddStm_GetTimeLow(void)
{
    return (uint32_T)MODULE_CPU0.STM.ABS.U;
}

/**
 * \brief   Computes an absolute deadline from a relative timeout.
 *
 * \param[in] TimeOut  Timeout in STM ticks.  Use CDD_STM_TIME_CONST_xxx macros.
 *
 * \return  Absolute 64-bit STM tick count at which the deadline expires.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
uint64_T CddStm_GetDeadline(uint64_T TimeOut)
{
    uint64_T deadline = CddStm_GetTime() + TimeOut;

    return deadline;
}

/**
 * \brief   Tests whether an absolute deadline has been reached or exceeded.
 *
 * \param[in] Deadline  Absolute STM tick deadline from CddStm_GetDeadline().
 *
 * \return  0x1U if current time > Deadline, 0x0U otherwise.
 *
 * \note    MISRA C:2012 Rule 14.4: `now > Deadline` produces an essentially
 *          Boolean result assigned to uint32_T.  Explicit 0x0U / 0x1U
 *          encoding avoids an implicit Boolean-to-integer conversion warning.
 *          MISRA C:2012 Rule 8.7 – function has external linkage.
 *          MISRA C:2012 Rule 15.5 – single exit point.
 */
uint32_T CddStm_IsDeadlineElapsed(uint64_T Deadline)
{
    uint64_T now       = CddStm_GetTime();
    uint32_T isElapsed = 0x0U;

    if (now > Deadline)
    {
        isElapsed = 0x1U;
    }

    return isElapsed;
}

/**
 * \brief   Busy-wait delay of the specified number of microseconds.
 *
 * \details Computes an absolute deadline as
 *              now + (Microseconds × CDD_STM_TIME_CONST_1US)
 *          and spins until CddStm_IsDeadlineElapsed() returns 0x1U.
 *
 *          Passing Microseconds == 0U returns immediately — the deadline
 *          equals now, and the while-condition evaluates to false on the
 *          first test.
 *
 * \pre     CddStm_Init() must have been called so that CddStm_TimeTable_G
 *          is populated.
 *
 * \param[in] Microseconds  Delay duration [µs], must be >= 0.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 14.4  : while-condition uses explicit == 0x0U.
 *          MISRA C:2012 Rule 10.3  : (uint64_T)Microseconds widens uint32_T
 *          before multiplication; no truncation.
 *          MISRA C:2012 Rule 8.7 – function has external linkage.
 */
void CddStm_Delay_Us(uint32_T Microseconds)
{
    uint64_T deadline = CddStm_GetDeadline((uint64_T)Microseconds * CDD_STM_TIME_CONST_1US);

    while (CddStm_IsDeadlineElapsed(deadline) == 0x0U)
    {
        CddSys_NopDelay(1U, 1U);
    }
}

/**
 * \brief   Builds the STM time-constant table from the live fSTM frequency.
 *
 * \details All divisions are exact integer divisions; fractional ticks are
 *          truncated toward zero.
 *
 *          At fSTM = 300 MHz (TC4x mid-range):
 *            10 ns  →  3 ticks
 *            1 ms   →  300 000 ticks   (ISR period)
 *            10 s   →  3 000 000 000 ticks
 *
 *          At fSTM = 500 MHz (TC49x max):
 *            10 ns  →  5 ticks
 *            1 ms   →  500 000 ticks
 *            10 s   →  5 000 000 000 ticks
 *
 * \param[in] StmFreq  STM clock frequency in Hz.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 7.2 / 10.3: Divisors on the right are uint32
 *          literals; stmFreq (uint64_T) is promoted in the division — no
 *          truncation in the dividend.
 *          MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 */
static void CddStm_InitTimeTable(uint64_T StmFreq)
{
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10NS]  = StmFreq / (1000000000U / 10U);    /* ÷ 100 000 000 → ~3 ticks @ 300 MHz  */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100NS] = StmFreq / (1000000000U / 100U);   /* ÷  10 000 000 → ~30 ticks           */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_1US]   = StmFreq / 1000000U;               /* ÷       1 000 → 300 ticks           */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10US]  = StmFreq / 100000U;                /* ÷         100 → 3 000 ticks         */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100US] = StmFreq / 10000U;                 /* ÷          10 → 30 000 ticks        */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_1MS]   = StmFreq / 1000U;                  /*              → 300 000 ticks ← ISR  */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10MS]  = StmFreq / 100U;                   /*              → 3 000 000 ticks      */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100MS] = StmFreq / 10U;                    /*              → 30 000 000 ticks     */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_1S]    = StmFreq;                          /*              → 300 000 000 ticks    */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_10S]   = StmFreq * 10U;                    /* × 10         → 3 000 000 000 ticks  */
    CddStm_TimeTable_G[CDD_STM_TIMER_INDEX_100S]  = StmFreq * 100U;                   /* × 100        → 30 000 000 000 ticks */
}
