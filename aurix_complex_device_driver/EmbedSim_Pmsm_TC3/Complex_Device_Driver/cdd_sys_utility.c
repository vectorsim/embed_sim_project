/**********************************************************************************************************************
 * \file        cdd_sys_utility.c
 * \brief       Implementation of system-level utility interfaces for AURIX TC3xx.
 *
 * \details     All clock-tree calculations follow the TC3xx Reference Manual (ds1) register map.
 *              System PLL formula (ds1 §P.937):
 *
 *                  fPLL0 = (fOSC × (NDIV + 1)) / ((PDIV + 1) × (K2DIV + 1))
 *
 *              Watchdog EndInit sequences follow ds1 §P.974.
 *              GTM CMU register access follows TC3xx GTM User Manual (ds2).
 *
 * \note        MISRA C:2012 deviations (pragma nomisrac):
 *
 *              Rule 10.7 / 12.2 / 2.2 — Composite expressions in watchdog register writes are
 *              unavoidable due to the TC3xx hardware password protocol: all fields of
 *              WDTCPUxCON0 / WDTSCON0 must be written in a single 32-bit bus access,
 *              combining the password, ENDINIT, LCK, and REL fields simultaneously.
 *              The zero-valued (0x0U) OR terms are retained for field-by-field
 *              documentation clarity (Rule 2.2 harmless OR with zero).
 *              Each deviation site is individually bracketed with
 *              #pragma nomisrac / #pragma nomisrac restore.
 *
 *              Rule 14.3 — In CddSys_GetPll00Freq, CddSys_GetPll01Freq, and
 *              CddSys_GetPerPllK3Freq the divisor guards (p_div >= 1.0, k_div >= 1.0)
 *              are invariantly true because each value is derived from an unsigned hardware
 *              register field with +1U applied before casting.  The guards are retained as
 *              defensive programming against future refactoring; they are documented inline.
 *
 * \note        MISRA corrections applied (vs original):
 *              Rule 15.5  — CddSys_GetGtmSrcFreq, CddSys_GetGtmClusterFreq,
 *                           CddSys_GetGtmCmuGlobalFreq, CddSys_GetPerPllK3Freq
 *                           rewritten to single point of exit.
 *              Rule 14.3  — Invariant guards in GetPll00Freq / GetPll01Freq / GetPerPllK3Freq
 *                           inverted to positive-sense (>= 1.0 && >= 1.0) with deviation note.
 *              Rule 10.4  — All float literals (f-suffix) in real64_T expressions replaced
 *                           with double literals throughout clock-tree functions.
 *              Rule 2.2   — (0x0U << 0xNU) dead zero-shift expressions in all WDT writes
 *                           replaced with plain (0x0U).
 *
 * \copyright   Copyright (C) SEPL UG 2024
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_sys_utility.h"
#include "cdd_asm_functions.h"
#include "IfxScu_reg.h"
#include "IfxCpu.h"
#include "IfxGtm_reg.h"

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  Lock bit active — WDTCPUxCON0.LCK / WDTSCON0.LCK value when the register
 *          is in the locked state (a password access is required to unlock it). */
#define WDT_LOCKED              (0x1U)

/** \brief  ENDINIT asserted — protected registers are write-locked.
 *          Written to WDTCPUxCON0.ENDINIT / WDTSCON0.ENDINIT to re-engage protection. */
#define WDT_ENDINIT_SET         (0x1U)

/** \brief  ENDINIT cleared — protected registers are accessible for writing.
 *          Written to WDTCPUxCON0.ENDINIT / WDTSCON0.ENDINIT to lift protection. */
#define WDT_ENDINIT_CLEAR       (0x0U)

/** \brief  SCU_CCUCON1.PLL1DIVDIS == 0 means the PLL1 post-divider (K3) is active.
 *          Named PLL1_DIV_ACTIVE to reflect the non-inverted semantic: the divider
 *          is active (not disabled) when this field reads 0. */
#define PLL1_DIV_ACTIVE         (0x0U)

/** \brief  GTM_CMU_CLK_EN.EN_CLKx == 0x3 indicates the corresponding CMU clock
 *          is currently enabled and running (ds2 §P.184). */
#define CMU_CLK_ENABLED         (0x3U)

/** \brief  SCU_CCUCON0.GTMDIV == 1 selects the 2×fSPB GTM source bypass mode.
 *          All other non-zero values select fSOURCE0 / GTMDIV. */
#define GTM_SRC_2XSPB           (0x1U)

/**********************************************************************************************************************
 * Function Implementations — Core Identification
 *********************************************************************************************************************/

/**
 * \brief   Reads the CORE_ID SFR via MFCR and returns the hardware core index.
 */
uint32_T CddSys_GetCoreId(void)
{
    return (uint32_T)__mfcr(CPU_CORE_ID);
}

/**********************************************************************************************************************
 * Function Implementations — Busy-Wait
 *********************************************************************************************************************/

/**
 * \brief   Executes OuterLoop × InnerLoop NOP instructions as a software delay.
 *
 * \details No timer resource is consumed.  __asm("nop") prevents the inner body from
 *          being optimised away by the compiler.
 */
void CddSys_NopDelay(uint32_T InnerLoop, uint32_T OuterLoop)
{
    uint32_T i;
    uint32_T j;

    for (i = 0U; i < OuterLoop; i++)
    {
        for (j = 0U; j < InnerLoop; j++)
        {
            __asm("nop");
        }
    }
}

/**********************************************************************************************************************
 * Function Implementations — Watchdog Password Retrieval
 *********************************************************************************************************************/

/**
 * \brief   Reads SCU_WDTCPU0CON0.PW and inverts bits [7:2] (ds1 §P.975).
 *
 * \details TC3xx hardware stores the password with bits [7:2] inverted.  XORing with
 *          0x3F restores the correct password to embed in the ENDINIT access word.
 */
uint32_T CddSys_GetCpuWdt00Pwd(void)
{
    uint32_T pwd;

    pwd  = SCU_WDTCPU0CON0.B.PW;
    pwd ^= 0x3FU;   /* Undo hardware inversion of bits [7:2] (ds1 §P.975) */
    return pwd;
}

/**
 * \brief   Reads SCU_WDTCPU1CON0.PW and inverts bits [7:2] (ds1 §P.975).
 */
uint32_T CddSys_GetCpuWdt01Pwd(void)
{
    uint32_T pwd;

    pwd  = SCU_WDTCPU1CON0.B.PW;
    pwd ^= 0x3FU;   /* Undo hardware inversion of bits [7:2] (ds1 §P.975) */
    return pwd;
}

/**
 * \brief   Reads SCU_WDTSCON0.PW and inverts bits [7:2] (ds1 §P.975).
 */
uint32_T CddSys_GetSafetyWdtPwd(void)
{
    uint32_T pwd;

    pwd  = SCU_WDTSCON0.B.PW;
    pwd ^= 0x3FU;   /* Undo hardware inversion of bits [7:2] (ds1 §P.975) */
    return pwd;
}

/**********************************************************************************************************************
 * Function Implementations — CPU0 Watchdog EndInit
 *********************************************************************************************************************/

/**
 * \brief   Three-step ENDINIT clear sequence for CPU0 watchdog (ds1 §P.974).
 *
 * \details Step 1 — Password access: unlock the register if LCK == 1 by writing
 *                   ENDINIT=1, LCK=0, PW=corrected password, REL unchanged.
 *          Step 2 — Modify access: clear ENDINIT and re-lock (LCK=1).
 *          Poll  — Spin until SCU_WDTCPU0CON0.ENDINIT hardware-confirms 0.
 */
void CddSys_ClearCpuWdt00EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetCpuWdt00Pwd();

    /* Step 1: password access — unlock the register (LCK=0) while keeping ENDINIT=1 */
    if (SCU_WDTCPU0CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
        SCU_WDTCPU0CON0.U = (0x1U << 0x0U)                    |   /* ENDINIT = 1 (not cleared yet) */
                            (0x0U)                    |   /* LCK     = 0 (unlock)          */
                            (pwd  << 0x2U)                    |   /* PW      = corrected password  */
                            (SCU_WDTCPU0CON0.B.REL << 16U);       /* REL     = preserve            */
        #pragma nomisrac restore
    }

    /* Step 2: modify access — clear ENDINIT and re-lock the register (LCK=1) */
    #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
    SCU_WDTCPU0CON0.U = (0x0U)                        |   /* ENDINIT = 0 (cleared)        */
                        (0x1U << 0x1U)                        |   /* LCK     = 1 (lock)           */
                        (pwd  << 0x2U)                        |   /* PW      = corrected password */
                        (SCU_WDTCPU0CON0.B.REL << 16U);           /* REL     = preserve           */
    #pragma nomisrac restore

    /* Poll: wait until hardware confirms ENDINIT == 0 */
    while (SCU_WDTCPU0CON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

/**
 * \brief   Three-step ENDINIT set sequence for CPU0 watchdog (ds1 §P.974).
 *
 * \details Step 1 — Password access: unlock the register (LCK=0), ENDINIT stays 1.
 *          Step 2 — Modify access: assert ENDINIT=1 and re-lock (LCK=1).
 *          Poll  — Spin until SCU_WDTCPU0CON0.ENDINIT hardware-confirms 1.
 */
void CddSys_SetCpuWdt00EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetCpuWdt00Pwd();

    /* Step 1: password access — unlock the register (LCK=0) */
    if (SCU_WDTCPU0CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
        SCU_WDTCPU0CON0.U = (0x1U << 0x0U)                    |   /* ENDINIT = 1 (maintained)     */
                            (0x0U)                    |   /* LCK     = 0 (unlock)         */
                            (pwd  << 0x2U)                    |   /* PW      = corrected password */
                            (SCU_WDTCPU0CON0.B.REL << 16U);       /* REL     = preserve           */
        #pragma nomisrac restore
    }

    /* Step 2: modify access — assert ENDINIT=1 and re-lock the register */
    #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
    SCU_WDTCPU0CON0.U = (0x1U << 0x0U)                        |   /* ENDINIT = 1 (set)            */
                        (0x1U << 0x1U)                        |   /* LCK     = 1 (lock)           */
                        (pwd  << 0x2U)                        |   /* PW      = corrected password */
                        (SCU_WDTCPU0CON0.B.REL << 16U);           /* REL     = preserve           */
    #pragma nomisrac restore

    /* Poll: wait until hardware confirms ENDINIT == 1 */
    while (SCU_WDTCPU0CON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

/**
 * \brief   Disables the CPU0 watchdog timer via the DR bit (ds1 §P.980).
 *
 * \details Internally sequences: ClearCpuWdt00EndInit → DR=1 → SetCpuWdt00EndInit.
 *          Interrupts must be disabled by the caller across this call.
 */
void CddSys_DisableCpuWdt00(void)
{
    CddSys_ClearCpuWdt00EndInit();
    SCU_WDTCPU0CON1.B.DR = 0x1U;   /* DR: Disable Request — halts the watchdog timer */
    CddSys_SetCpuWdt00EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — CPU1 Watchdog EndInit
 *********************************************************************************************************************/

/**
 * \brief   Three-step ENDINIT clear sequence for CPU1 watchdog (ds1 §P.974).
 *
 * \details Identical protocol to CddSys_ClearCpuWdt00EndInit() applied to
 *          SCU_WDTCPU1CON0.  Step comments match the CPU0 variant for traceability.
 */
void CddSys_ClearCpuWdt01EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetCpuWdt01Pwd();

    /* Step 1: password access — unlock the register (LCK=0) while keeping ENDINIT=1 */
    if (SCU_WDTCPU1CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
        SCU_WDTCPU1CON0.U = (0x1U << 0x0U)                    |   /* ENDINIT = 1 (not cleared yet) */
                            (0x0U)                    |   /* LCK     = 0 (unlock)          */
                            (pwd  << 0x2U)                    |   /* PW      = corrected password  */
                            (SCU_WDTCPU1CON0.B.REL << 16U);       /* REL     = preserve            */
        #pragma nomisrac restore
    }

    /* Step 2: modify access — clear ENDINIT and re-lock the register (LCK=1) */
    #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
    SCU_WDTCPU1CON0.U = (0x0U)                        |   /* ENDINIT = 0 (cleared)        */
                        (0x1U << 0x1U)                        |   /* LCK     = 1 (lock)           */
                        (pwd  << 0x2U)                        |   /* PW      = corrected password */
                        (SCU_WDTCPU1CON0.B.REL << 16U);           /* REL     = preserve           */
    #pragma nomisrac restore

    /* Poll: wait until hardware confirms ENDINIT == 0 */
    while (SCU_WDTCPU1CON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

/**
 * \brief   Three-step ENDINIT set sequence for CPU1 watchdog (ds1 §P.974).
 *
 * \details Identical protocol to CddSys_SetCpuWdt00EndInit() applied to
 *          SCU_WDTCPU1CON0.
 */
void CddSys_SetCpuWdt01EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetCpuWdt01Pwd();

    /* Step 1: password access — unlock the register (LCK=0) */
    if (SCU_WDTCPU1CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
        SCU_WDTCPU1CON0.U = (0x1U << 0x0U)                    |   /* ENDINIT = 1 (maintained)     */
                            (0x0U)                    |   /* LCK     = 0 (unlock)         */
                            (pwd  << 0x2U)                    |   /* PW      = corrected password */
                            (SCU_WDTCPU1CON0.B.REL << 16U);       /* REL     = preserve           */
        #pragma nomisrac restore
    }

    /* Step 2: modify access — assert ENDINIT=1 and re-lock the register */
    #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
    SCU_WDTCPU1CON0.U = (0x1U << 0x0U)                        |   /* ENDINIT = 1 (set)            */
                        (0x1U << 0x1U)                        |   /* LCK     = 1 (lock)           */
                        (pwd  << 0x2U)                        |   /* PW      = corrected password */
                        (SCU_WDTCPU1CON0.B.REL << 16U);           /* REL     = preserve           */
    #pragma nomisrac restore

    /* Poll: wait until hardware confirms ENDINIT == 1 */
    while (SCU_WDTCPU1CON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

/**
 * \brief   Disables the CPU1 watchdog timer via the DR bit.
 *
 * \details Internally sequences: ClearCpuWdt01EndInit → DR=1 → SetCpuWdt01EndInit.
 *          Mirrors CddSys_DisableCpuWdt00() for CPU1.
 */
void CddSys_DisableCpuWdt01(void)
{
    CddSys_ClearCpuWdt01EndInit();
    SCU_WDTCPU1CON1.B.DR = 0x1U;   /* DR: Disable Request — halts the watchdog timer */
    CddSys_SetCpuWdt01EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — CPU Watchdog Dispatch (core-agnostic)
 *********************************************************************************************************************/

/**
 * \brief   Dispatches ENDINIT clear to the watchdog of the currently executing core.
 *
 * \details Reads CORE_ID at runtime to select CPU0 or CPU1 variant.
 *          The default case is a deliberate no-op (MISRA 16.4: default required).
 */
void CddSys_ClearCpuWdtEndInit(void)
{
    uint32_T core_id;

    core_id = CddSys_GetCoreId();
    switch (core_id)
    {
        case 0x0U:
            CddSys_ClearCpuWdt00EndInit();
            break;
        case 0x1U:
            CddSys_ClearCpuWdt01EndInit();
            break;
        default:
            /* Unknown core — no action (MISRA 16.4: default required) */
            break;
    }
}

/**
 * \brief   Dispatches ENDINIT set to the watchdog of the currently executing core.
 *
 * \details Mirrors CddSys_ClearCpuWdtEndInit(); must always be paired with it.
 */
void CddSys_SetCpuWdtEndInit(void)
{
    uint32_T core_id;

    core_id = CddSys_GetCoreId();
    switch (core_id)
    {
        case 0x0U:
            CddSys_SetCpuWdt00EndInit();
            break;
        case 0x1U:
            CddSys_SetCpuWdt01EndInit();
            break;
        default:
            /* Unknown core — no action (MISRA 16.4: default required) */
            break;
    }
}

/**********************************************************************************************************************
 * Function Implementations — Safety Watchdog EndInit
 *********************************************************************************************************************/

/**
 * \brief   Three-step Safety ENDINIT clear sequence (ds1 §P.974).
 *
 * \details Identical protocol to the CPU WDT variants, applied to SCU_WDTSCON0.
 *          Step 1 — Password access: unlock (LCK=0), ENDINIT=1.
 *          Step 2 — Modify access: ENDINIT=0, re-lock (LCK=1).
 *          Poll  — Spin until SCU_WDTSCON0.ENDINIT hardware-confirms 0.
 */
void CddSys_ClearSafetyWdtEndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetSafetyWdtPwd();

    /* Step 1: password access — unlock the register (LCK=0) while keeping ENDINIT=1 */
    if (SCU_WDTSCON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
        SCU_WDTSCON0.U = (0x1U << 0x0U)                       |   /* ENDINIT = 1 (not cleared yet) */
                         (0x0U)                       |   /* LCK     = 0 (unlock)          */
                         (pwd  << 0x2U)                       |   /* PW      = corrected password  */
                         (SCU_WDTSCON0.B.REL << 16U);             /* REL     = preserve            */
        #pragma nomisrac restore
    }

    /* Step 2: modify access — clear ENDINIT and re-lock the register (LCK=1) */
    #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
    SCU_WDTSCON0.U = (0x0U)                           |   /* ENDINIT = 0 (cleared)        */
                     (0x1U << 0x1U)                           |   /* LCK     = 1 (lock)           */
                     (pwd  << 0x2U)                           |   /* PW      = corrected password */
                     (SCU_WDTSCON0.B.REL << 16U);                 /* REL     = preserve           */
    #pragma nomisrac restore

    /* Poll: wait until hardware confirms ENDINIT == 0 */
    while (SCU_WDTSCON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

/**
 * \brief   Three-step Safety ENDINIT set sequence (ds1 §P.974).
 *
 * \details Step 1 — Password access: unlock (LCK=0), ENDINIT stays 1.
 *          Step 2 — Modify access: ENDINIT=1, re-lock (LCK=1).
 *          Poll  — Spin until SCU_WDTSCON0.ENDINIT hardware-confirms 1.
 */
void CddSys_SetSafetyWdtEndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetSafetyWdtPwd();

    /* Step 1: password access — unlock the register (LCK=0) */
    if (SCU_WDTSCON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
        SCU_WDTSCON0.U = (0x1U << 0x0U)                       |   /* ENDINIT = 1 (maintained)     */
                         (0x0U)                       |           /* LCK     = 0 (unlock)         */
                         (pwd  << 0x2U)                       |   /* PW      = corrected password */
                         (SCU_WDTSCON0.B.REL << 16U);             /* REL     = preserve           */
        #pragma nomisrac restore
    }

    /* Step 2: modify access — assert ENDINIT=1 and re-lock the register */
    #pragma nomisrac   /* Rule 10.7, 12.2, 2.2: single-access hardware password protocol; (0x0U) OR is harmless but kept for field-by-field documentation */
    SCU_WDTSCON0.U = (0x1U << 0x0U)                           |   /* ENDINIT = 1 (set)            */
                     (0x1U << 0x1U)                           |   /* LCK     = 1 (lock)           */
                     (pwd  << 0x2U)                           |   /* PW      = corrected password */
                     (SCU_WDTSCON0.B.REL << 16U);                 /* REL     = preserve           */
    #pragma nomisrac restore

    /* Poll: wait until hardware confirms ENDINIT == 1 */
    while (SCU_WDTSCON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

/**
 * \brief   Disables the Safety watchdog via WDTSCON1.DR (ds1 §P.977).
 *
 * \details Internally sequences:
 *            CddSys_ClearSafetyWdtEndInit() → DR=1 → CddSys_SetSafetyWdtEndInit().
 */
void CddSys_DisableSafetyWdt(void)
{
    CddSys_ClearSafetyWdtEndInit();
    SCU_WDTSCON1.B.DR = 0x1U;   /* DR: Disable Request — halts the Safety watchdog timer */
    CddSys_SetSafetyWdtEndInit();
}

/**********************************************************************************************************************
 * Function Implementations — Clock Tree Frequency Interrogation
 *********************************************************************************************************************/

/**
 * \brief   Returns EVR_OSC_FREQUENCY as a real64_T (compile-time constant).
 */
real64_T CddSys_GetEvrFreq(void)
{
    return (real64_T)EVR_OSC_FREQUENCY;
}

/**
 * \brief   Returns SYSCLK_OSC_FREQUENCY as a real64_T (compile-time constant).
 */
real64_T CddSys_GetSysClkFreq(void)
{
    return (real64_T)SYSCLK_OSC_FREQUENCY;
}

/**
 * \brief   Returns XTAL_OSC_FREQUENCY as a real64_T (compile-time constant).
 */
real64_T CddSys_GetExtOscFreq(void)
{
    return (real64_T)XTAL_OSC_FREQUENCY;
}

/**
 * \brief   Reads SCU_SYSPLLCON0.INSEL and returns the corresponding oscillator frequency.
 *
 * \details INSEL 0=EVR, 1=XTAL, 2=SysClk, other → 0 Hz (PLL input unknown / off).
 */
real64_T CddSys_GetPrimaryOscFreq(void)
{
    real64_T freq;

    switch (SCU_SYSPLLCON0.B.INSEL)
    {
        case 0x0U:
            freq = CddSys_GetEvrFreq();     /* EVR backup oscillator */
            break;
        case 0x1U:
            freq = CddSys_GetExtOscFreq();  /* External crystal      */
            break;
        case 0x2U:
            freq = CddSys_GetSysClkFreq();  /* System clock OSC      */
            break;
        default:
            freq = 0.0;                    /* Reserved / undefined  */
            break;
    }
    return freq;
}

/**
 * \brief   Computes fPLL0 using the integer PLL formula from ds1 §P.937.
 *
 * \details fPLL0 = (fOSC × (NDIV+1)) / ((PDIV+1) × (K2DIV+1)).
 *          p_div and k2_div are derived from uint register fields (+1 each), so they
 *          are always >= 1 and the zero-guard is defensive against future register
 *          reads returning 0xFF (all-ones on bus error).
 */
real64_T CddSys_GetPll00Freq(void)
{
    real64_T pll_freq;
    real64_T n_div;
    real64_T p_div;
    real64_T k2_div;

    n_div  = (real64_T)(SCU_SYSPLLCON0.B.NDIV  + 1U);
    p_div  = (real64_T)(SCU_SYSPLLCON0.B.PDIV  + 1U);
    k2_div = (real64_T)(SCU_SYSPLLCON1.B.K2DIV + 1U);
    pll_freq = 0.0;

    /* p_div and k2_div are always >= 1.0 by construction (unsigned hardware field + 1U).
     * Positive-sense guard retained for defensive robustness against future refactoring.
     * Rule 14.3 deviation: condition is invariantly true; justified as defensive programming. */
    if ((p_div >= 1.0) && (k2_div >= 1.0))
    {
        pll_freq = (n_div * CddSys_GetPrimaryOscFreq()) / (p_div * k2_div);
    }
    return pll_freq;
}

/**
 * \brief   Returns the PerPLL K2-path output (fSOURCE1) after the PLL1DIVDIS post-divider.
 *
 * \details The Peripheral PLL has its own N/P/K oscillator chain, independent of the
 *          System PLL (fPLL0).  The K2 output formula (ds1 §P.938):
 *
 *              fPerPLL_K2 = (fOSC × (PERPLL_NDIV+1)) / ((PERPLL_PDIV+1) × (PERPLL_K2DIV+1))
 *
 *          SCU_CCUCON1.PLL1DIVDIS == 0 (PLL1_DIV_ACTIVE) means the ÷2 post-divider on
 *          the K2 output is active → fSOURCE1 = fPerPLL_K2 / 2.
 *          PLL1DIVDIS == 1 bypasses the ÷2 → fSOURCE1 = fPerPLL_K2 directly.
 *          fSOURCE1 feeds the ADC and QSPI (when SCU_CCUCON1.CLKSELQSPI == 0x1) domains.
 *
 * \note    FIX (Bug-2): previous code incorrectly used fPLL0 (SysPLL) / K3DIV as base.
 *          Correct base is SCU_PERPLLCON0/1 registers, not SCU_SYSPLLCON0/1 (ds1 §P.938).
 */
real64_T CddSys_GetPll01Freq(void)
{
    real64_T n_div;
    real64_T p_div;
    real64_T k2_div;
    real64_T pll_freq;

    n_div  = (real64_T)(SCU_PERPLLCON0.B.NDIV  + 1U);
    p_div  = (real64_T)(SCU_PERPLLCON0.B.PDIV  + 1U);
    k2_div = (real64_T)(SCU_PERPLLCON1.B.K2DIV + 1U);
    pll_freq = 0.0;

    /* p_div and k2_div are always >= 1.0 by construction (unsigned hardware field + 1U).
     * Positive-sense guard retained for defensive robustness against future refactoring.
     * Rule 14.3 deviation: condition is invariantly true; justified as defensive programming. */
    if ((p_div >= 1.0) && (k2_div >= 1.0))
    {
        pll_freq = (n_div * CddSys_GetPrimaryOscFreq()) / (p_div * k2_div);
    }

    /* PLL1DIVDIS == 0 (PLL1_DIV_ACTIVE): the K2-output post-÷2 prescaler is active (ds1 §P.938) */
    if (SCU_CCUCON1.B.PLL1DIVDIS == PLL1_DIV_ACTIVE)
    {
        pll_freq = pll_freq / 2.0;
    }
    return pll_freq;
}

/**
 * \brief   Returns the PerPLL K3-path output (fSOURCE2) after the DIVBY fractional factor.
 *
 * \details PerPLL K3 path formula (ds1 §P.938):
 *
 *              fPerPLL_K3 = (fOSC × (NDIV+1)) / ((PDIV+1) × (K3DIV+1) × factor)
 *
 *          SCU_PERPLLCON0.DIVBY selects the K3 fractional prescaler:
 *            DIVBY == 0  →  factor = 1.6   (default fractional mode)
 *            DIVBY == 1  →  factor = 2.0   (integer divide-by-2 mode)
 *
 *          fSOURCE2 feeds the QSPI clock when SCU_CCUCON1.CLKSELQSPI == 0x2.
 */
real64_T CddSys_GetPerPllK3Freq(void)
{
    real64_T n_div;
    real64_T p_div;
    real64_T k3_div;
    real64_T divby_factor;
    real64_T k3_freq = 0.0;

    n_div  = (real64_T)(SCU_PERPLLCON0.B.NDIV  + 1U);
    p_div  = (real64_T)(SCU_PERPLLCON0.B.PDIV  + 1U);
    k3_div = (real64_T)(SCU_PERPLLCON1.B.K3DIV + 1U);

    /* PERPLLCON0.DIVBY selects the K3 fractional prescaler (ds1 §P.938):
     *   DIVBY == 0 → 1.6 factor (fractional mode)
     *   DIVBY == 1 → 2.0 factor (integer mode)                                      */
    divby_factor = (0x0U == SCU_PERPLLCON0.B.DIVBY) ? 1.6 : 2.0;

    /* p_div and k3_div are always >= 1.0 by construction (unsigned hardware field + 1U).
     * Positive-sense guard retained for defensive robustness against future refactoring.
     * Rule 14.3 deviation: condition is invariantly true; justified as defensive programming. */
    if ((p_div >= 1.0) && (k3_div >= 1.0))
    {
        k3_freq = (n_div * CddSys_GetPrimaryOscFreq()) / (p_div * k3_div * divby_factor);
    }
    return k3_freq;
}

/**
 * \brief   Returns fSOURCE0 — the primary clock source for SRI, STM, SPB, and GTM domains.
 *
 * \details SCU_CCUCON0.CLKSEL: 0x1 → fPLL0,  else → EVR backup clock.
 */
real64_T CddSys_GetSrc00Freq(void)
{
    real64_T freq;

    switch (SCU_CCUCON0.B.CLKSEL)
    {
        case 0x1U:
            freq = CddSys_GetPll00Freq();  /* PLL0 selected     */
            break;
        case 0x0U:                         /* fall-through      */
        default:
            freq = CddSys_GetEvrFreq();    /* EVR backup (safe) */
            break;
    }
    return freq;
}

/**
 * \brief   Returns fSOURCE1 — the primary clock source for ADC and QSPI domains.
 *
 * \details SCU_CCUCON0.CLKSEL: 0x1 → fPLL1,  else → EVR backup clock.
 *          In low-power mode (LPDIV >= 2), an additional ÷2 prescaler is applied by hardware
 *          to the SOURCE1 domain.  LPDIV 0 and 1 do not apply this extra factor.
 */
real64_T CddSys_GetSrc01Freq(void)
{
    real64_T freq;

    switch (SCU_CCUCON0.B.CLKSEL)
    {
        case 0x1U:
            freq = CddSys_GetPll01Freq();  /* PLL1 selected     */
            break;
        case 0x0U:                         /* fall-through      */
        default:
            freq = CddSys_GetEvrFreq();    /* EVR backup (safe) */
            break;
    }

    /* Low-power additional ÷2 for SOURCE1 domain (LPDIV >= 2 only; 0 and 1 exempt) */
    if ((SCU_CCUCON0.B.LPDIV != 0x0U) && (SCU_CCUCON0.B.LPDIV != 0x1U))
    {
        freq = freq / 2.0;
    }
    return freq;
}

/**
 * \brief   Computes fSRI from fSOURCE0 via SRIDIV (normal) or LPDIV table (low-power).
 *
 * \details Normal mode (LPDIV == 0):
 *              fSRI = fSOURCE0 / SRIDIV   (SRIDIV == 0 → clock off → 0 Hz returned)
 *          Low-power mode (LPDIV 1–4): prescaler table (÷30, ÷60, ÷120, ÷240).
 *          Unknown LPDIV → 0 Hz.
 */
real64_T CddSys_GetSriFreq(void)
{
    real64_T source_freq;
    real64_T sri_freq;

    source_freq = CddSys_GetSrc00Freq();

    switch (SCU_CCUCON0.B.LPDIV)
    {
        case 0x0U:
            sri_freq = (0x0U == SCU_CCUCON0.B.SRIDIV) ? 0.0 :
                       source_freq / (real64_T)SCU_CCUCON0.B.SRIDIV;
            break;
        case 0x1U:
            sri_freq = source_freq / 30.0;    /* Low-power ÷30  */
            break;
        case 0x2U:
            sri_freq = source_freq / 60.0;    /* Low-power ÷60  */
            break;
        case 0x3U:
            sri_freq = source_freq / 120.0;   /* Low-power ÷120 */
            break;
        case 0x4U:
            sri_freq = source_freq / 240.0;   /* Low-power ÷240 */
            break;
        default:
            sri_freq = 0.0;                   /* Reserved       */
            break;
    }
    return sri_freq;
}

/**
 * \brief   Returns fCPU0 by applying the fractional CPU0DIV to fSRI.
 *
 * \details fCPU0 = fSRI × (64 − CPU0DIV) / 64  (SCU_CCUCON6.CPU0DIV).
 *          CPU0DIV == 0 means no division (fCPU0 == fSRI).
 *          Note: this function always reads CPU0DIV regardless of the calling core.
 */
real64_T CddSys_GetCpuFreq(void)
{
    real64_T cpu_freq;
    uint32_T cpu_div;

    cpu_freq = CddSys_GetSriFreq();
    cpu_div  = SCU_CCUCON6.B.CPU0DIV;

    if (0x0U != cpu_div)
    {
        cpu_freq = cpu_freq * ((64.0 - (real64_T)cpu_div) / 64.0);
    }
    return cpu_freq;
}

/**
 * \brief   Returns fSTM = fSOURCE0 / STMDIV  (SCU_CCUCON0.STMDIV).
 *
 * \details STMDIV == 0 → STM clock gated off → returns 0.0.
 */
real64_T CddSys_GetStmFreq(void)
{
    real64_T stm_freq;
    uint32_T stm_div;

    stm_div  = SCU_CCUCON0.B.STMDIV;
    stm_freq = 0.0;

    if (0x0U != stm_div)
    {
        stm_freq = CddSys_GetSrc00Freq() / (real64_T)stm_div;
    }
    return stm_freq;
}

/**
 * \brief   Computes fSPB from fSOURCE0 via SPBDIV (normal) or LPDIV table (low-power).
 *
 * \details Normal mode (LPDIV == 0):
 *              fSPB = fSOURCE0 / SPBDIV   (SPBDIV must be >= 2 per TC3xx spec; < 2 → 0 Hz)
 *          Low-power mode (LPDIV 1–4): same prescaler table as fSRI.
 *          Unknown LPDIV → 0 Hz.
 */
real64_T CddSys_GetSpbFreq(void)
{
    real64_T source_freq;
    real64_T spb_freq;

    source_freq = CddSys_GetSrc00Freq();

    switch (SCU_CCUCON0.B.LPDIV)
    {
        case 0x0U:
            spb_freq = (SCU_CCUCON0.B.SPBDIV < 0x2U) ? 0.0 :   /* SPBDIV < 2 is illegal */
                       source_freq / (real64_T)SCU_CCUCON0.B.SPBDIV;
            break;
        case 0x1U:
            spb_freq = source_freq / 30.0;    /* Low-power ÷30  */
            break;
        case 0x2U:
            spb_freq = source_freq / 60.0;    /* Low-power ÷60  */
            break;
        case 0x3U:
            spb_freq = source_freq / 120.0;   /* Low-power ÷120 */
            break;
        case 0x4U:
            spb_freq = source_freq / 240.0;   /* Low-power ÷240 */
            break;
        default:
            spb_freq = 0.0;                   /* Reserved       */
            break;
    }
    return spb_freq;
}

/**
 * \brief   Returns the GTM clock source frequency before the GTMDIV module divider.
 *
 * \details SCU_CCUCON0.GTMDIV == 1 selects the 2×fSPB bypass source (GTM_SRC_2XSPB).
 *          All other GTMDIV values (including 0 — GTM off) return fSOURCE0 as the
 *          dormant source oscillator.  CddSys_GetGtmFreq() guards the zero-divide case.
 */
real64_T CddSys_GetGtmSrcFreq(void)
{
    real64_T gtm_src_freq;

    if (GTM_SRC_2XSPB == SCU_CCUCON0.B.GTMDIV)
    {
        gtm_src_freq = 2.0 * CddSys_GetSpbFreq();   /* 2×fSPB bypass mode (GTMDIV == 1) */
    }
    else
    {
        gtm_src_freq = CddSys_GetSrc00Freq();        /* fSOURCE0 for all other GTMDIV values */
    }
    return gtm_src_freq;
}

/**
 * \brief   Returns fGTM = fGTMSRC / GTMDIV  (SCU_CCUCON0.GTMDIV).
 *
 * \details GTMDIV == 0 → GTM module clock gated off → returns 0.0.
 */
real64_T CddSys_GetGtmFreq(void)
{
    real64_T gtm_freq;
    uint32_T gtm_div;

    gtm_div  = SCU_CCUCON0.B.GTMDIV;
    gtm_freq = 0.0;

    if (0x0U != gtm_div)
    {
        gtm_freq = CddSys_GetGtmSrcFreq() / (real64_T)gtm_div;
    }
    return gtm_freq;
}

/**
 * \brief   Returns fCLS0 = fGTM / CLS0_CLK_DIV  (GTM_CLS_CLK_CFG.CLS0_CLK_DIV).
 *
 * \details CLS0_CLK_DIV == 0 → cluster clock off → returns 0.0.
 */
real64_T CddSys_GetGtmClusterFreq(void)
{
    uint32_T cluster_div;
    real64_T cluster_freq;

    cluster_div  = GTM_CLS_CLK_CFG.B.CLS0_CLK_DIV;
    cluster_freq = 0.0;

    if (0x0U != cluster_div)
    {
        cluster_freq = CddSys_GetGtmFreq() / (real64_T)cluster_div;
    }
    return cluster_freq;
}

/**
 * \brief   Returns fGCLK = (GCLK_DEN / GCLK_NUM) × fCLS0  (ds2 §P.188).
 *
 * \details GCLK_NUM is read from an unsigned hardware register; the zero check uses
 *          == 0.0 (exact) rather than <= 0.0, since the cast from uint32_T can
 *          never produce a negative value (MISRA Rule 14.3 compliance).
 */
real64_T CddSys_GetGtmCmuGlobalFreq(void)
{
    real64_T numerator   = (real64_T)GTM_CMU_GCLK_NUM.B.GCLK_NUM;
    real64_T denominator;
    real64_T gclk_freq;

    denominator = (real64_T)GTM_CMU_GCLK_DEN.B.GCLK_DEN;
    gclk_freq   = 0.0;

    /* GCLK_NUM == 0 → divider not configured; zero-divide guard.
     * Comparison to exact 0.0 is safe: cast from uint32_T can never produce a negative
     * value (MISRA Rule 14.3 compliance note in original). */
    if (numerator != 0.0)
    {
        gclk_freq = (denominator / numerator) * CddSys_GetGtmClusterFreq();
    }
    return gclk_freq;
}

/**
 * \brief   Returns fADC, which equals fSOURCE1 on TC3xx.
 */
real64_T CddSys_GetAdcFreq(void)
{
    return CddSys_GetSrc01Freq();
}

/**
 * \brief   Returns fQSPI by selecting the PerPLL source and dividing by QSPIDIV.
 *
 * \details QSPI is clocked from the Peripheral PLL, NOT from fSOURCE0 (SysPLL) (ds1 §P.962).
 *          SCU_CCUCON1.CLKSELQSPI selects:
 *            0x1 → fSOURCE1 = PerPLL K2 path / PLL1DIVDIS post-÷2  (CddSys_GetPll01Freq)
 *            0x2 → fSOURCE2 = PerPLL K3 path / DIVBY factor         (CddSys_GetPerPllK3Freq)
 *            other → 0 Hz (clock off).
 *          QSPIDIV == 0 → clock gated → returns 0.0.
 *
 * \note    FIX (Bug-1): previous code incorrectly routed CLKSELQSPI=1 → fSOURCE0 (SysPLL)
 *          and CLKSELQSPI=2 → CddSys_GetSrc01Freq (which was also wrong).
 *          Root cause: TC3xx QSPI clock source is the Peripheral PLL domain, shared
 *          with ADC, not the System PLL domain that drives SRI/SPB/GTM.
 */
real64_T CddSys_GetQspiFreq(void)
{
    real64_T source;
    real64_T freq;
    uint32_T qspi_div;

    qspi_div = SCU_CCUCON1.B.QSPIDIV;   /* Snapshot volatile SFR once — prevents value change between guard and division */
    freq     = 0.0;                      /* Default: clock gated off (qspi_div == 0 path) */

    switch (SCU_CCUCON1.B.CLKSELQSPI)
    {
        case 0x1U:
            source = CddSys_GetPll01Freq();      /* fSOURCE1: PerPLL K2 path */
            break;
        case 0x2U:
            source = CddSys_GetPerPllK3Freq();   /* fSOURCE2: PerPLL K3 path */
            break;
        default:
            source = 0.0;                       /* Clock off                */
            break;
    }

    if (0x0U != qspi_div)
    {
        freq = source / (real64_T)qspi_div;
    }
    return freq;
}

/**********************************************************************************************************************
 * Function Implementations — GTM CMU CLK0
 *********************************************************************************************************************/

/**
 * \brief   Computes the CLK_CNT divider and writes it to GTM_CMU_CLK_0_CTRL.
 *
 * \details CLK_CNT = round(fGCLK / CmuClk00Freq) − 1  (ds2 §P.186).
 *          Write sequence — CLK_CNT may only be written while CLK0 is disabled
 *          (TC3xx GTM UM requirement):
 *            1. ClearCpuWdtEndInit
 *            2. EN_CLK0 = 0x1U  (disable CLK0)
 *            3. CLK_CNT = computed value
 *            4. SetCpuWdtEndInit
 *            5. EN_CLK0 = 0x2U  (re-enable CLK0)
 */
void CddSys_SetGtmCmuClk00Freq(real64_T CmuClk00Freq)
{
    real64_T cmu_global_freq;
    real64_T cmu_divider;
    uint32_T clk_cnt;

    cmu_global_freq = CddSys_GetGtmCmuGlobalFreq();
    cmu_divider     = (cmu_global_freq / CmuClk00Freq) - 1.0;
    clk_cnt         = (uint32_T)cmu_divider;

    /* Round to nearest integer (0.5 threshold) */
    if ((cmu_divider - (real64_T)clk_cnt) > 0.5)
    {
        clk_cnt++;
    }

    CddSys_ClearCpuWdtEndInit();
    GTM_CMU_CLK_EN.B.EN_CLK0     = 0x1U;      /* Disable CLK0 before modifying (ds2 §P.184) */
    GTM_CMU_CLK_0_CTRL.B.CLK_CNT = clk_cnt;   /* Write divider value           (ds2 §P.186) */
    CddSys_SetCpuWdtEndInit();

    GTM_CMU_CLK_EN.B.EN_CLK0     = 0x2U;   /* Re-enable CLK0 after write    (ds2 §P.184) */
}

/**
 * \brief   Back-computes fCLK0 = fGCLK / (CLK_CNT + 1) from the live register.
 *
 * \details Returns 0.0 if CLK0 is not in the enabled state (EN_CLK0 != CMU_CLK_ENABLED).
 */
real64_T CddSys_GetGtmCmuClk00Freq(void)
{
    real64_T cmu_clk_freq;

    cmu_clk_freq = 0.0;

    if (CMU_CLK_ENABLED == GTM_CMU_CLK_EN.B.EN_CLK0)
    {
        real64_T clk_div = (real64_T)GTM_CMU_CLK_0_CTRL.B.CLK_CNT + 1.0;
        cmu_clk_freq = CddSys_GetGtmCmuGlobalFreq() / clk_div;
    }
    return cmu_clk_freq;
}

/**********************************************************************************************************************
 * Function Implementations — CPU Interrupt Control
 *********************************************************************************************************************/

/**
 * \brief   Returns ICR.IE of the calling core via MFCR(CPU_ICR).
 */
uint32_T CddSys_IsIrqEnabled(void)
{
    Ifx_CPU_ICR icr_reg;

    icr_reg.U = (uint32_T)__mfcr(CPU_ICR);
    return (0x1U == icr_reg.B.IE) ? 1U : 0U;
}

/**
 * \brief   Issues a TriCore DISABLE instruction and returns the prior ICR.IE state.
 *
 * \details The prior state is the save token for nested save-restore critical sections.
 *          Only issues DISABLE if interrupts were enabled; avoids a redundant instruction
 *          if already in a disabled context.
 */
uint32_T CddSys_DisableIrq(void)
{
    uint32_T prev_state = CddSys_IsIrqEnabled();

    if (1U == prev_state)
    {
        __disable();   /* TriCore DISABLE instruction — clears ICR.IE */
    }
    return prev_state;
}

/**
 * \brief   Issues a TriCore ENABLE instruction only if the saved state was enabled.
 *
 * \details Correct nesting: an outer critical section that was already disabled is
 *          not accidentally re-enabled by an inner restore.
 */
void CddSys_RestoreIrq(uint32_T PrevState)
{
    if (1U == PrevState)
    {
        __enable();   /* TriCore ENABLE instruction — sets ICR.IE */
    }
}

/**********************************************************************************************************************
 * Function Implementations — Spinlock
 *********************************************************************************************************************/

/**
 * \brief   Performs a single CMPSWAP.W atomic try-acquire on *LockPtr.
 *
 * \details alleged_free == 0 is the expected value (lock free).
 *          CddAsm_CmpAndSwap returns the *previous* value of *LockPtr:
 *            prev == 0 → swap succeeded → lock is now held by this core → return 1.
 *            prev != 0 → lock was held by another core → no swap → return 0.
 */
uint32_T CddSys_AcquireSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr)
{
    uint32_T alleged_free;
    uint32_T prev_value;

    alleged_free = 0U;                                             /* Expected state: lock is free */
    prev_value   = CddAsm_CmpAndSwap(LockPtr, 1U, alleged_free);   /* Atomic test-and-set          */

    return (prev_value == alleged_free) ? 1U : 0U;   /* 1 = acquired, 0 = contended */
}

/**
 * \brief   Releases the spinlock by writing 0 to *LockPtr atomically.
 *
 * \details A plain store is sufficient here because only the owning core writes 0,
 *          and the TriCore memory model guarantees store visibility across cores
 *          sharing LMU.  No CMPSWAP needed for the release path.
 */
void CddSys_ReleaseSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr)
{
    *LockPtr = 0U;   /* Atomic release: restores free state, visible to all cores */
}

/**
 * \brief   Computes |Lhs − Rhs| and compares to Epsilon without using fabsf().
 *
 * \details Manual negation avoids a <math.h> dependency, which may not be permitted
 *          in the MISRA C:2012 configuration for this module.  NaN / infinity inputs
 *          are not handled; the caller must ensure finite inputs.
 */
uint32_T CddSys_AreEqual(real32_T Lhs, real32_T Rhs, real32_T Epsilon)
{
    real32_T diff   = Lhs - Rhs;
    uint32_T result = 0U;

    if (diff < 0.0f)
    {
        diff = -diff;   /* Manual absolute value — avoids fabsf() / <math.h> dependency */
    }
    result = (diff <= Epsilon) ? 1U : 0U;

    return result;
}

/**
 * \brief   Computes |Lhs − Rhs| and compares to Epsilon for single-precision operands.
 *
 * \details Explicit real32_T variant of CddSys_AreEqual.  Manual negation avoids
 *          a <math.h> dependency.  NaN / infinity inputs are not handled.
 */
uint32_T CddSys_AreEqual32(real32_T Lhs, real32_T Rhs, real32_T Epsilon)
{
    real32_T diff   = Lhs - Rhs;
    uint32_T result = 0U;

    if (diff < 0.0f)
    {
        diff = -diff;   /* Manual absolute value — avoids fabsf() / <math.h> dependency */
    }
    result = (diff <= Epsilon) ? 1U : 0U;

    return result;
}

/**
 * \brief   Computes |Lhs − Rhs| and compares to Epsilon for double-precision operands.
 *
 * \details real64_T variant for clock-frequency and high-resolution comparisons.
 *          Manual negation avoids a <math.h> dependency.  NaN / infinity inputs
 *          are not handled.
 */
uint32_T CddSys_AreEqual64(real64_T Lhs, real64_T Rhs, real64_T Epsilon)
{
    real64_T diff   = Lhs - Rhs;
    uint32_T result = 0U;

    if (diff < 0.0)
    {
        diff = -diff;   /* Manual absolute value — avoids fabs() / <math.h> dependency */
    }
    result = (diff <= Epsilon) ? 1U : 0U;

    return result;
}
