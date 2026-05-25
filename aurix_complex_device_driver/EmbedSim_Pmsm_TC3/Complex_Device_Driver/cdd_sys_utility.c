/**********************************************************************************************************************
 * \file        cdd_sys_utility.c
 * \brief       Implementation of cdd_sys_utility.h
 *
 * \details     All clock-tree calculations follow the TC3xx Reference Manual (ds1)
 *              register map and the PLL formula:
 *
 *                  fPLL = (N * fOSC) / (P * K2)
 *
 *              Watchdog EndInit sequences follow ds1 P.974.
 *
 * \note        MISRA C:2012 deviations (pragma nomisrac):
 *              - Rule 10.7 / 12.2 : Composite expressions in watchdog register
 *                writes are unavoidable due to the hardware password protocol.
 *                Each deviation site is individually bracketed with
 *                #pragma nomisrac / #pragma nomisrac restore.
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

/** \brief  Lock bit value in WDTCPUxCON0.LCK / WDTSCON0.LCK  */
#define WDT_LOCKED              (0x1U)

/** \brief  ENDINIT active (protected registers locked)          */
#define WDT_ENDINIT_SET         (0x1U)

/** \brief  ENDINIT cleared (protected registers accessible)     */
#define WDT_ENDINIT_CLEAR       (0x0U)

/** \brief  PLL1 divider disabled flag in SCU_CCUCON1            */
#define PLL1_DIV_DISABLED       (0x0U)

/** \brief  CMU CLK enabled status in GTM_CMU_CLK_EN             */
#define CMU_CLK_ENABLED         (0x3U)

/** \brief  GTM clock source is 2*fSPB when GTMDIV == 1          */
#define GTM_SRC_2XSPB           (0x1U)

/**********************************************************************************************************************
 * Function Implementations — Core Identification
 *********************************************************************************************************************/

uint32_T CddSys_GetCoreId(void)
{
    return (uint32_T)__mfcr(CPU_CORE_ID);
}

/**********************************************************************************************************************
 * Function Implementations — Busy-Wait
 *********************************************************************************************************************/

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
 * Function Implementations — Watchdog Password
 *********************************************************************************************************************/

uint32_T CddSys_GetWdt00Pwd(void)
{
    uint32_T pwd;
    /* Bits [7:2] are stored inverted in the register (ds1 P.975) */
    pwd  = SCU_WDTCPU0CON0.B.PW;
    pwd ^= 0x3FU;
    return pwd;
}

uint32_T CddSys_GetWdt01Pwd(void)
{
    uint32_T pwd;
    pwd  = SCU_WDTCPU1CON0.B.PW;
    pwd ^= 0x3FU;
    return pwd;
}

uint32_T CddSys_GetSafetyWdtPwd(void)
{
    uint32_T pwd;
    pwd  = SCU_WDTSCON0.B.PW;
    pwd ^= 0x3FU;
    return pwd;
}

/**********************************************************************************************************************
 * Function Implementations — CPU0 Watchdog EndInit
 *********************************************************************************************************************/

void CddSys_ClearWdt00EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetWdt00Pwd();

    /* Step 1: unlock the register if currently locked (ds1 P.974) */
    if (SCU_WDTCPU0CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac   /* Rule 10.7, 12.2: hardware password protocol */
        SCU_WDTCPU0CON0.U = (0x1U << 0x0U)                    |   /* ENDINIT = 1 (not cleared yet)  */
                            (0x0U << 0x1U)                    |   /* LCK = 0    (unlock)            */
                            (pwd  << 0x2U)                    |
                            (SCU_WDTCPU0CON0.B.REL << 16U);
        #pragma nomisrac restore
    }

    /* Step 2: clear ENDINIT and lock the register */
    #pragma nomisrac   /* Rule 10.7, 12.2: hardware password protocol */
    SCU_WDTCPU0CON0.U = (0x0U << 0x0U)                        |   /* ENDINIT = 0 (cleared)         */
                        (0x1U << 0x1U)                        |   /* LCK = 1    (lock)             */
                        (pwd  << 0x2U)                        |
                        (SCU_WDTCPU0CON0.B.REL << 16U);
    #pragma nomisrac restore

    while (SCU_WDTCPU0CON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

void CddSys_SetWdt00EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetWdt00Pwd();

    /* Step 1: unlock */
    if (SCU_WDTCPU0CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTCPU0CON0.U = (0x1U << 0x0U)                    |
                            (0x0U << 0x1U)                    |
                            (pwd  << 0x2U)                    |
                            (SCU_WDTCPU0CON0.B.REL << 16U);
        #pragma nomisrac restore
    }

    /* Step 2: set ENDINIT and lock */
    #pragma nomisrac
    SCU_WDTCPU0CON0.U = (0x1U << 0x0U)                        |   /* ENDINIT = 1 */
                        (0x1U << 0x1U)                        |   /* LCK = 1     */
                        (pwd  << 0x2U)                        |
                        (SCU_WDTCPU0CON0.B.REL << 16U);
    #pragma nomisrac restore

    while (SCU_WDTCPU0CON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

void CddSys_DisableWdt00(void)
{
    CddSys_ClearWdt00EndInit();
    SCU_WDTCPU0CON1.B.DR = 0x1U;   /* Disable request (ds1 P.980) */
    CddSys_SetWdt00EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — CPU1 Watchdog EndInit
 *********************************************************************************************************************/

void CddSys_ClearWdt01EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetWdt01Pwd();

    if (SCU_WDTCPU1CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTCPU1CON0.U = (0x1U << 0x0U)                    |
                            (0x0U << 0x1U)                    |
                            (pwd  << 0x2U)                    |
                            (SCU_WDTCPU1CON0.B.REL << 16U);
        #pragma nomisrac restore
    }

    #pragma nomisrac
    SCU_WDTCPU1CON0.U = (0x0U << 0x0U)                        |
                        (0x1U << 0x1U)                        |
                        (pwd  << 0x2U)                        |
                        (SCU_WDTCPU1CON0.B.REL << 16U);
    #pragma nomisrac restore

    while (SCU_WDTCPU1CON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

void CddSys_SetWdt01EndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetWdt01Pwd();

    if (SCU_WDTCPU1CON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTCPU1CON0.U = (0x1U << 0x0U)                    |
                            (0x0U << 0x1U)                    |
                            (pwd  << 0x2U)                    |
                            (SCU_WDTCPU1CON0.B.REL << 16U);
        #pragma nomisrac restore
    }

    #pragma nomisrac
    SCU_WDTCPU1CON0.U = (0x1U << 0x0U)                        |
                        (0x1U << 0x1U)                        |
                        (pwd  << 0x2U)                        |
                        (SCU_WDTCPU1CON0.B.REL << 16U);
    #pragma nomisrac restore

    while (SCU_WDTCPU1CON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

void CddSys_DisableWdt01(void)
{
    CddSys_ClearWdt01EndInit();
    SCU_WDTCPU1CON1.B.DR = 0x1U;
    CddSys_SetWdt01EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — CPU Watchdog Dispatch (core-agnostic)
 *********************************************************************************************************************/

void CddSys_ClearWdtEndInit(void)
{
    uint32_T core_id;

    core_id = CddSys_GetCoreId();
    switch (core_id)
    {
        case 0x0U:
            CddSys_ClearWdt00EndInit();
            break;
        case 0x1U:
            CddSys_ClearWdt01EndInit();
            break;
        default:
            /* Unknown core — no action (MISRA 16.4: default required) */
            break;
    }
}

void CddSys_SetWdtEndInit(void)
{
    uint32_T core_id;

    core_id = CddSys_GetCoreId();
    switch (core_id)
    {
        case 0x0U:
            CddSys_SetWdt00EndInit();
            break;
        case 0x1U:
            CddSys_SetWdt01EndInit();
            break;
        default:
            break;
    }
}

/**********************************************************************************************************************
 * Function Implementations — Safety Watchdog EndInit
 *********************************************************************************************************************/

void CddSys_ClearSafetyWdtEndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetSafetyWdtPwd();

    if (SCU_WDTSCON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTSCON0.U = (0x1U << 0x0U)                       |
                         (0x0U << 0x1U)                       |
                         (pwd  << 0x2U)                       |
                         (SCU_WDTSCON0.B.REL << 16U);
        #pragma nomisrac restore
    }

    #pragma nomisrac
    SCU_WDTSCON0.U = (0x0U << 0x0U)                           |
                     (0x1U << 0x1U)                           |
                     (pwd  << 0x2U)                           |
                     (SCU_WDTSCON0.B.REL << 16U);
    #pragma nomisrac restore

    while (SCU_WDTSCON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

void CddSys_SetSafetyWdtEndInit(void)
{
    uint32_T pwd;

    pwd = CddSys_GetSafetyWdtPwd();

    if (SCU_WDTSCON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTSCON0.U = (0x1U << 0x0U)                       |
                         (0x0U << 0x1U)                       |
                         (pwd  << 0x2U)                       |
                         (SCU_WDTSCON0.B.REL << 16U);
        #pragma nomisrac restore
    }

    #pragma nomisrac
    SCU_WDTSCON0.U = (0x1U << 0x0U)                           |
                     (0x1U << 0x1U)                           |
                     (pwd  << 0x2U)                           |
                     (SCU_WDTSCON0.B.REL << 16U);
    #pragma nomisrac restore

    while (SCU_WDTSCON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

void CddSys_DisableSafetyWdt(void)
{
    CddSys_ClearSafetyWdtEndInit();
    SCU_WDTSCON1.B.DR = 0x1U;
    CddSys_SetSafetyWdtEndInit();
}

/**********************************************************************************************************************
 * Function Implementations — Clock Tree Frequency Interrogation
 *********************************************************************************************************************/

real64_T CddSys_GetEvrFreq(void)
{
    return (real64_T)EVR_OSC_FREQUENCY;
}

real64_T CddSys_GetSysClkFreq(void)
{
    return (real64_T)SYSCLK_OSC_FREQUENCY;
}

real64_T CddSys_GetExtOscFreq(void)
{
    return (real64_T)XTAL_OSC_FREQUENCY;
}

real64_T CddSys_GetPrimaryOscFreq(void)
{
    real64_T freq;

    switch (SCU_SYSPLLCON0.B.INSEL)
    {
        case 0x0U:  freq = CddSys_GetEvrFreq();    break;
        case 0x1U:  freq = CddSys_GetExtOscFreq(); break;
        case 0x2U:  freq = CddSys_GetSysClkFreq(); break;
        default:    freq = 0.0f;                    break;
    }
    return freq;
}

real64_T CddSys_GetPll00Freq(void)
{
    real64_T pll_freq;
    real64_T n_div;
    real64_T p_div;
    real64_T k2_div;

    n_div    = (real64_T)(SCU_SYSPLLCON0.B.NDIV + 1U);
    p_div    = (real64_T)(SCU_SYSPLLCON0.B.PDIV + 1U);
    k2_div   = (real64_T)(SCU_SYSPLLCON1.B.K2DIV + 1U);

    if ((p_div <= 0.0f) || (k2_div <= 0.0f))
    {
        pll_freq = 0.0f;
    }
    else
    {
        pll_freq = (n_div * CddSys_GetPrimaryOscFreq()) / (p_div * k2_div);
    }
    return pll_freq;
}

real64_T CddSys_GetPll01Freq(void)
{
    real64_T pll_freq;
    real64_T k3_div;

    if (SCU_CCUCON1.B.PLL1DIVDIS == PLL1_DIV_DISABLED)
    {
        k3_div   = (real64_T)(SCU_PERPLLCON1.B.K3DIV + 1U);
        pll_freq = CddSys_GetPll00Freq() / k3_div;
    }
    else
    {
        pll_freq = 0.0f;
    }
    return pll_freq;
}

real64_T CddSys_GetSrc00Freq(void)
{
    real64_T freq;

    switch (SCU_CCUCON0.B.CLKSEL)
    {
        case 0x1U:  freq = CddSys_GetPll00Freq();  break;
        case 0x0U:  /* fall-through */
        default:    freq = CddSys_GetEvrFreq();     break;
    }
    return freq;
}

real64_T CddSys_GetSrc01Freq(void)
{
    real64_T freq;

    switch (SCU_CCUCON0.B.CLKSEL)
    {
        case 0x1U:  freq = CddSys_GetPll01Freq();  break;
        case 0x0U:  /* fall-through */
        default:    freq = CddSys_GetEvrFreq();     break;
    }

    if ((SCU_CCUCON0.B.LPDIV != 0x0U) && (SCU_CCUCON0.B.LPDIV != 0x1U))
    {
        freq = freq / 2.0f;
    }
    return freq;
}

real64_T CddSys_GetSriFreq(void)
{
    real64_T source_freq = CddSys_GetSrc00Freq();
    real64_T sri_freq;

    switch (SCU_CCUCON0.B.LPDIV)
    {
        case 0x0U:
            sri_freq = (0x0U == SCU_CCUCON0.B.SRIDIV) ? 0.0f :
                       source_freq / (real64_T)SCU_CCUCON0.B.SRIDIV;
            break;
        case 0x1U:  sri_freq = source_freq / 30.0f;   break;
        case 0x2U:  sri_freq = source_freq / 60.0f;   break;
        case 0x3U:  sri_freq = source_freq / 120.0f;  break;
        case 0x4U:  sri_freq = source_freq / 240.0f;  break;
        default:    sri_freq = 0.0f;                   break;
    }
    return sri_freq;
}

real64_T CddSys_GetCpuFreq(void)
{
    real64_T cpu_freq = CddSys_GetSriFreq();
    uint32_T cpu_div  = SCU_CCUCON6.B.CPU0DIV;

    if (0x0U != cpu_div)
    {
        cpu_freq = cpu_freq * ((64.0f - (real64_T)cpu_div) / 64.0f);
    }
    return cpu_freq;
}

real64_T CddSys_GetStmFreq(void)
{
    real64_T stm_freq = 0.0f;
    uint32_T stm_div  = SCU_CCUCON0.B.STMDIV;

    if (0x0U != stm_div)
    {
        stm_freq = CddSys_GetSrc00Freq() / (real64_T)stm_div;
    }
    return stm_freq;
}

real64_T CddSys_GetSpbFreq(void)
{
    real64_T source_freq = CddSys_GetSrc00Freq();
    real64_T spb_freq;

    switch (SCU_CCUCON0.B.LPDIV)
    {
        case 0x0U:
            spb_freq = (SCU_CCUCON0.B.SPBDIV < 0x2U) ? 0.0f :
                       source_freq / (real64_T)SCU_CCUCON0.B.SPBDIV;
            break;
        case 0x1U:  spb_freq = source_freq / 30.0f;   break;
        case 0x2U:  spb_freq = source_freq / 60.0f;   break;
        case 0x3U:  spb_freq = source_freq / 120.0f;  break;
        case 0x4U:  spb_freq = source_freq / 240.0f;  break;
        default:    spb_freq = 0.0f;                   break;
    }
    return spb_freq;
}

real64_T CddSys_GetGtmSrcFreq(void)
{
    if (GTM_SRC_2XSPB == SCU_CCUCON0.B.GTMDIV)
    {
        return 2.0f * CddSys_GetSpbFreq();
    }
    return CddSys_GetSrc00Freq();
}

real64_T CddSys_GetGtmFreq(void)
{
    real64_T gtm_freq = 0.0f;
    uint32_T gtm_div  = SCU_CCUCON0.B.GTMDIV;

    if (0x0U != gtm_div)
    {
        gtm_freq = CddSys_GetGtmSrcFreq() / (real64_T)gtm_div;
    }
    return gtm_freq;
}

real64_T CddSys_GetGtmClusterFreq(void)
{
    uint32_T cluster_div = GTM_CLS_CLK_CFG.B.CLS0_CLK_DIV;

    if (0x0U == cluster_div)
    {
        return 0.0f;
    }
    return CddSys_GetGtmFreq() / (real64_T)cluster_div;
}

real64_T CddSys_GetGtmCmuGlobalFreq(void)
{
    real64_T numerator   = (real64_T)GTM_CMU_GCLK_NUM.B.GCLK_NUM;
    real64_T denominator = (real64_T)GTM_CMU_GCLK_DEN.B.GCLK_DEN;

    if (numerator <= 0.0f)
    {
        return 0.0f;
    }
    return (denominator / numerator) * CddSys_GetGtmClusterFreq();
}

real64_T CddSys_GetAdcFreq(void)
{
    return CddSys_GetSrc01Freq();
}

real64_T CddSys_GetQspiFreq(void)
{
    real64_T source   = 0.0f;
    real64_T freq     = 0.0f;
    uint32_T qspi_div = SCU_CCUCON1.B.QSPIDIV;

    switch (SCU_CCUCON1.B.CLKSELQSPI)
    {
        case 0x1U:  source = CddSys_GetSrc00Freq();  break;
        case 0x2U:  source = CddSys_GetSrc01Freq();  break;
        default:    source = 0.0f;                    break;
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

void CddSys_SetGtmCmuClk00Freq(real64_T CmuClk00Freq)
{
    real64_T cmu_global_freq = CddSys_GetGtmCmuGlobalFreq();
    real64_T cmu_divider     = (cmu_global_freq / CmuClk00Freq) - 1.0f;
    uint32_T clk_cnt         = (uint32_T)cmu_divider;

    /* Round to nearest integer */
    if ((cmu_divider - (real64_T)clk_cnt) > 0.5f)
    {
        clk_cnt++;
    }

    CddSys_ClearWdtEndInit();
    GTM_CMU_CLK_0_CTRL.B.CLK_CNT = clk_cnt;   /* ds2 P.186 */
    CddSys_SetWdtEndInit();

    GTM_CMU_CLK_EN.B.EN_CLK0 = 0x2U;           /* ds2 P.184 */
}

real64_T CddSys_GetGtmCmuClk00Freq(void)
{
    real64_T cmu_clk_freq = 0.0f;

    if (CMU_CLK_ENABLED == GTM_CMU_CLK_EN.B.EN_CLK0)
    {
        real64_T clk_div = (real64_T)GTM_CMU_CLK_0_CTRL.B.CLK_CNT + 1.0f;
        cmu_clk_freq = CddSys_GetGtmCmuGlobalFreq() / clk_div;
    }
    return cmu_clk_freq;
}

/**********************************************************************************************************************
 * Function Implementations — CPU Interrupt Control
 *********************************************************************************************************************/

uint32_T CddSys_IsIrqEnabled(void)
{
    Ifx_CPU_ICR icr_reg;
    icr_reg.U = (uint32_T)__mfcr(CPU_ICR);
    return (0x1U == icr_reg.B.IE) ? 1U : 0U;
}

uint32_T CddSys_DisableIrq(void)
{
    uint32_T prev_state = CddSys_IsIrqEnabled();
    if (1U == prev_state)
    {
        __disable();
    }
    return prev_state;
}

void CddSys_RestoreIrq(uint32_T PrevState)
{
    if (1U == PrevState)
    {
        __enable();
    }
}

/**********************************************************************************************************************
 * Function Implementations — Spinlock
 *********************************************************************************************************************/

uint32_T CddSys_AcquireSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr)
{
    uint32_T alleged_free;
    uint32_T prev_value;

    alleged_free = 0U;   /* Expected: lock is free */
    prev_value   = CddAsm_CmpAndSwap(LockPtr, 1U, alleged_free);

    /* If previous value matched alleged_free, swap succeeded → locked */
    return (prev_value == alleged_free) ? 1U : 0U;
}

void CddSys_ReleaseSpinLock(CONSTP2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) LockPtr)
{
    *LockPtr = 0U;
}

uint32_T CddSys_AreEqual(real32_T Lhs, real32_T Rhs, real32_T Epsilon)
{
    real32_T diff   = Lhs - Rhs;
    uint32_T result = 0U;

    if (diff < 0.0f)
    {
        diff = -diff;
    }
    result = (diff <= Epsilon) ? 1U : 0U;

    return result;
}
