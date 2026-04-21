/**********************************************************************************************************************
 * \file        cdd_stm_app.c
 * \brief       Implementation of cdd_stm_app.h — STM0 driver.
 *
 * \details     Responsibilities of this file:
 *              1. Build the time-constant table from the live fSTM frequency.
 *              2. Configure STM0 Compare-0 for a 1 ms periodic interrupt.
 *              3. ISR: call System_Tick_Handler() (owned by cdd_task_handler_app)
 *                      then rearm the compare register for the next 1 ms.
 *
 *              The software task scheduler (1 ms / 10 ms / 100 ms / 1 s) is
 *              intentionally separated into cdd_task_handler_app.c so that
 *              application tasks can be modified without touching the STM driver.
 *
 *              ISR rearm strategy:
 *                  next_cmp = Get_Lower_System_Time() + TimeConst_1ms
 *              Unsigned 32-bit wrap at 2^32 is correct — the STM compare
 *              register follows the same wrap, so no special handling needed.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit comparison
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
#include "cdd_stm_app.h"
#include "cdd_task_handler_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "ifxStm_reg.h"
#include "ifxSrc_reg.h"
#include "Bsp.h"

/**********************************************************************************************************************
 * Interrupt Vector Table Entry
 *
 * Installs Stm_00_Cmp_00_Isr into the CPU0 vector table at priority
 * STM0_CMP0_IR_SRPN using the TASKING HI:/LO: relocation syntax.
 *********************************************************************************************************************/
IFX_INTERRUPT(Stm_00_Cmp_00_Isr, 0, STM0_CMP0_IR_SRPN);    /* STM0_CMP0_IR_SRPN */

/**********************************************************************************************************************
 * Private Macros — Time-Constant Table Indices
 *********************************************************************************************************************/
#define TIMER_COUNT         (11U)   /**< \brief Total number of pre-computed time constants */

#define TIMER_INDEX_10NS    (0U)    /**< \brief Index:  10 ns  */
#define TIMER_INDEX_100NS   (1U)    /**< \brief Index: 100 ns  */
#define TIMER_INDEX_1US     (2U)    /**< \brief Index:   1 us  */
#define TIMER_INDEX_10US    (3U)    /**< \brief Index:  10 us  */
#define TIMER_INDEX_100US   (4U)    /**< \brief Index: 100 us  */
#define TIMER_INDEX_1MS     (5U)    /**< \brief Index:   1 ms  */
#define TIMER_INDEX_10MS    (6U)    /**< \brief Index:  10 ms  */
#define TIMER_INDEX_100MS   (7U)    /**< \brief Index: 100 ms  */
#define TIMER_INDEX_1S      (8U)    /**< \brief Index:   1 s   */
#define TIMER_INDEX_10S     (9U)    /**< \brief Index:  10 s   */
#define TIMER_INDEX_100S    (10U)   /**< \brief Index: 100 s   */

/** \brief  Accessor macros for the pre-computed tick table                     */
#define TimeConst_10ns      (Sys_Tick_Time_Table_G[TIMER_INDEX_10NS])
#define TimeConst_100ns     (Sys_Tick_Time_Table_G[TIMER_INDEX_100NS])
#define TimeConst_1us       (Sys_Tick_Time_Table_G[TIMER_INDEX_1US])
#define TimeConst_10us      (Sys_Tick_Time_Table_G[TIMER_INDEX_10US])
#define TimeConst_100us     (Sys_Tick_Time_Table_G[TIMER_INDEX_100US])
#define TimeConst_1ms       (Sys_Tick_Time_Table_G[TIMER_INDEX_1MS])
#define TimeConst_10ms      (Sys_Tick_Time_Table_G[TIMER_INDEX_10MS])
#define TimeConst_100ms     (Sys_Tick_Time_Table_G[TIMER_INDEX_100MS])
#define TimeConst_1s        (Sys_Tick_Time_Table_G[TIMER_INDEX_1S])
#define TimeConst_10s       (Sys_Tick_Time_Table_G[TIMER_INDEX_10S])
#define TimeConst_100s      (Sys_Tick_Time_Table_G[TIMER_INDEX_100S])

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/** \brief  Pre-computed STM tick counts for each time constant  [STM ticks]   */
static uint64_T Sys_Tick_Time_Table_G[TIMER_COUNT];

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
static void Init_Time_Table(uint64_T stm_freq);

/**********************************************************************************************************************
 * ISR
 *********************************************************************************************************************/

/**
 * \brief   STM0 Compare-0 ISR — fires every 1 ms.
 *
 * \details Dispatches to System_Tick_Handler() then rearms the compare
 *          register inside a brief critical section to prevent a race between
 *          reading TIM0 and writing CMP0.
 */
void Stm_00_Cmp_00_Isr(void)
{
    uint32_T prev_ir_state;

    /* Dispatch software task scheduler (defined in cdd_task_handler_app.c) */
    System_Tick_Handler();

    /* Rearm: next compare = now + 1 ms, atomically */
    prev_ir_state = Disable_CPU_Interrupt();
    STM0_CMP0.B.CMPVAL = Get_Lower_System_Time() + (uint32_T)TimeConst_1ms;
    Restore_CPU_Interrupt(prev_ir_state);
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Initialize_STM_Module
 *------------------------------------------------------------------------------------------------------------------*/
void Initialize_STM_Module(void)
{
    uint64_T stm_frequency;

    stm_frequency = (uint64_T)Get_STM_Frequency();

    Init_Time_Table(stm_frequency);

    /* Configure Compare Match Control Register (ds2 P.68)
     * MSIZE0  = 0x1F : CMP0 compares all 32 bits [31:0]
     * MSTART0 = 0x00 : bit 0 of STM is the lowest compared bit */
    STM0_CMCON.B.MSIZE0  = 0x1FU;
    STM0_CMCON.B.MSTART0 = 0x0U;

    /* Route compare match to interrupt output STMIR0 (ds2 P.68)             */
    STM0_ICR.B.CMP0OS = 0x0U;

    /* Configure Service Request node (TC3xx AppNote appx1 P.409)            */
    SRC_STM0SR0.B.SRPN = STM0_CMP0_IR_SRPN;    /* Priority                  */
    SRC_STM0SR0.B.TOS  = 0x0U;                  /* Type of service: CPU0     */
    SRC_STM0SR0.B.CLRR = 0x1U;                  /* Clear any pending request */
    SRC_STM0SR0.B.SRE  = 0x1U;                  /* Enable service request    */

    /* Disarm: write current_time >> 1 so CMP0 cannot fire during setup      */
    STM0_CMP0.B.CMPVAL = (Get_Lower_System_Time() >> 0x1U);

    /* Enable compare output, then load the real first compare value          */
    STM0_ICR.B.CMP0EN  = 0x1U;
    STM0_CMP0.B.CMPVAL = Get_Lower_System_Time() + (uint32_T)TimeConst_1ms;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Get_System_Time
 *------------------------------------------------------------------------------------------------------------------*/
uint64_T Get_System_Time(void)
{
    uint64_T lower_sys_time;
    uint64_T upper_sys_time;
    uint32_T prev_ir_state;

    /* Read TIM0 first — hardware latches upper bits into CAP (ds2 P.60)     */
    prev_ir_state  = Disable_CPU_Interrupt();
    lower_sys_time = (uint64_T)STM0_TIM0.U;
    upper_sys_time = (uint64_T)STM0_CAP.U;
    Restore_CPU_Interrupt(prev_ir_state);

    upper_sys_time = (upper_sys_time << 0x20U) | lower_sys_time;

    return upper_sys_time;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Get_Lower_System_Time
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T Get_Lower_System_Time(void)
{
    /* Reading TIM0 also latches CAP — must precede any Get_System_Time call  */
    return STM0_TIM0.U;
}

/**********************************************************************************************************************
 * Private Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Init_Time_Table
 *------------------------------------------------------------------------------------------------------------------*/
static void Init_Time_Table(uint64_T stm_freq)
{
    Sys_Tick_Time_Table_G[TIMER_INDEX_10NS]  = stm_freq / (1000000000U / 10U);
    Sys_Tick_Time_Table_G[TIMER_INDEX_100NS] = stm_freq / (1000000000U / 100U);
    Sys_Tick_Time_Table_G[TIMER_INDEX_1US]   = stm_freq / 1000000U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_10US]  = stm_freq / 100000U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_100US] = stm_freq / 10000U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_1MS]   = stm_freq / 1000U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_10MS]  = stm_freq / 100U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_100MS] = stm_freq / 10U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_1S]    = stm_freq;
    Sys_Tick_Time_Table_G[TIMER_INDEX_10S]   = stm_freq * 10U;
    Sys_Tick_Time_Table_G[TIMER_INDEX_100S]  = stm_freq * 100U;
}
