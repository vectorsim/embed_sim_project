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

/** \brief  ENDINIT active (protected registers locked)         */
#define WDT_ENDINIT_SET         (0x1U)

/** \brief  ENDINIT cleared (protected registers accessible)    */
#define WDT_ENDINIT_CLEAR       (0x0U)

/** \brief  PLL1 divider disabled flag in SCU_CCUCON1           */
#define PLL1_DIV_DISABLED       (0x0U)

/** \brief  CMU CLK enabled status in GTM_CMU_CLK_EN            */
#define CMU_CLK_ENABLED         (0x3U)

/** \brief  GTM clock source is 2*fSPB when GTMDIV == 1         */
#define GTM_SRC_2XSPB           (0x1U)

/**********************************************************************************************************************
 * Function Implementations — Core Identification
 *********************************************************************************************************************/

uint32_T Get_Core_Id(void)
{
    return (uint32_T)__mfcr(CPU_CORE_ID);
}

/**********************************************************************************************************************
 * Function Implementations — Busy-Wait
 *********************************************************************************************************************/

void Nop_Delay(uint32_T InnerLoop, uint32_T OuterLoop)
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

uint32_T Get_CPU_00_WDT_Pwd(void)
{
    uint32_T pwd;
    /* Bits [7:2] are stored inverted in the register (ds1 P.975) */
    pwd  = SCU_WDTCPU0CON0.B.PW;
    pwd ^= 0x3FU;
    return pwd;
}

uint32_T Get_CPU_01_WDT_Pwd(void)
{
    uint32_T pwd;
    pwd  = SCU_WDTCPU1CON0.B.PW;
    pwd ^= 0x3FU;
    return pwd;
}

uint32_T Get_Safety_WDT_Pwd(void)
{
    uint32_T pwd;
    pwd  = SCU_WDTSCON0.B.PW;
    pwd ^= 0x3FU;
    return pwd;
}

/**********************************************************************************************************************
 * Function Implementations — CPU0 Watchdog EndInit
 *********************************************************************************************************************/

void Clear_CPU_00_WDT_EndInit(void)
{
    uint32_T pwd;

    pwd = Get_CPU_00_WDT_Pwd();

    /* Step 1: unlock the register if it is currently locked (ds1 P.974) */
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
    SCU_WDTCPU0CON0.U = (0x0U << 0x0U)                        |   /* ENDINIT = 0 (cleared)          */
                        (0x1U << 0x1U)                        |   /* LCK = 1    (lock)              */
                        (pwd  << 0x2U)                        |
                        (SCU_WDTCPU0CON0.B.REL << 16U);
    #pragma nomisrac restore

    /* Wait until ENDINIT is confirmed cleared */
    while (SCU_WDTCPU0CON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        Nop_Delay(0x1U, 0x1U);
    }
}

void Set_CPU_00_WDT_EndInit(void)
{
    uint32_T pwd;

    pwd = Get_CPU_00_WDT_Pwd();

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
        Nop_Delay(0x1U, 0x1U);
    }
}

void Disable_CPU_00_WDT(void)
{
    Clear_CPU_00_WDT_EndInit();
    SCU_WDTCPU0CON1.B.DR = 0x1U;   /* Disable request (ds1 P.980) */
    Set_CPU_00_WDT_EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — CPU1 Watchdog EndInit
 *********************************************************************************************************************/

void Clear_CPU_01_WDT_EndInit(void)
{
    uint32_T pwd;

    pwd = Get_CPU_01_WDT_Pwd();

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
        Nop_Delay(0x1U, 0x1U);
    }
}

void Set_CPU_01_WDT_EndInit(void)
{
    uint32_T pwd;

    pwd = Get_CPU_01_WDT_Pwd();

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
        Nop_Delay(0x1U, 0x1U);
    }
}

void Disable_CPU_01_WDT(void)
{
    Clear_CPU_01_WDT_EndInit();
    SCU_WDTCPU1CON1.B.DR = 0x1U;
    Set_CPU_01_WDT_EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — CPU Watchdog Dispatch (core-agnostic)
 *********************************************************************************************************************/

void Clear_CPU_WDT_EndInit(void)
{
    uint32_T core_id;

    core_id = Get_Core_Id();
    switch (core_id)
    {
        case 0U: Clear_CPU_00_WDT_EndInit(); break;
        case 1U: Clear_CPU_01_WDT_EndInit(); break;
        default: /* no action — unsupported core */  break;
    }
}

void Set_CPU_WDT_EndInit(void)
{
    uint32_T core_id;

    core_id = Get_Core_Id();
    switch (core_id)
    {
        case 0U: Set_CPU_00_WDT_EndInit(); break;
        case 1U: Set_CPU_01_WDT_EndInit(); break;
        default: /* no action */ break;
    }
}

/**********************************************************************************************************************
 * Function Implementations — Safety Watchdog EndInit
 *********************************************************************************************************************/

void Clear_Safety_WDT_EndInit(void)
{
    uint32_T pwd;

    pwd = Get_Safety_WDT_Pwd();

    if (SCU_WDTSCON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTSCON0.U = (0x1U << 0x0U)                       |
                         (0x0U << 0x1U)                       |
                         (pwd  << 0x2U)                       |
                         (SCU_WDTSCON0.B.REL << 0x10U);
        #pragma nomisrac restore
    }

    #pragma nomisrac
    SCU_WDTSCON0.U = (0x0U << 0x0U)                           |
                     (0x1U << 0x1U)                           |
                     (pwd  << 0x2U)                           |
                     (SCU_WDTSCON0.B.REL << 0x10U);
    #pragma nomisrac restore

    while (SCU_WDTSCON0.B.ENDINIT != WDT_ENDINIT_CLEAR)
    {
        Nop_Delay(0x1U, 0x1U);
    }
}

void Set_Safety_WDT_EndInit(void)
{
    uint32_T pwd;

    pwd = Get_Safety_WDT_Pwd();

    if (SCU_WDTSCON0.B.LCK == WDT_LOCKED)
    {
        #pragma nomisrac
        SCU_WDTSCON0.U = (0x1U << 0x0U)                       |
                         (0x0U << 0x1U)                       |
                         (pwd  << 0x2U)                       |
                         (SCU_WDTSCON0.B.REL << 0x10U);
        #pragma nomisrac restore
    }

    #pragma nomisrac
    SCU_WDTSCON0.U = (0x1U << 0x0U)                           |
                     (0x1U << 0x1U)                           |
                     (pwd  << 0x2U)                           |
                     (SCU_WDTSCON0.B.REL << 0x10U);
    #pragma nomisrac restore

    while (SCU_WDTSCON0.B.ENDINIT != WDT_ENDINIT_SET)
    {
        Nop_Delay(0x1U, 0x1U);
    }
}

void Disable_Safety_WDT(void)
{
    Clear_Safety_WDT_EndInit();
    SCU_WDTSCON1.B.DR = 0x1U;   /* ds1 P.977 */
    Set_Safety_WDT_EndInit();
}

/**********************************************************************************************************************
 * Function Implementations — Clock Tree
 *********************************************************************************************************************/

real64_T Get_EVR_Frequency(void)         { return EVR_OSC_FREQUENCY;    }
real64_T Get_SysClk_Frequency(void)      { return SYSCLK_OSC_FREQUENCY; }
real64_T Get_External_OSC_Frequency(void){ return XTAL_OSC_FREQUENCY;   }

real64_T Get_Primary_OSC_Frequency(void)
{
    real64_T osc_freq;
    uint32_T osc_sel;

    osc_sel = SCU_SYSPLLCON0.B.INSEL;   /* ds1 P.1085 */
    switch (osc_sel)
    {
        case 0x0U:  osc_freq = Get_EVR_Frequency();           break;
        case 0x1U:  osc_freq = Get_External_OSC_Frequency();  break;
        case 0x2U:  osc_freq = Get_SysClk_Frequency();        break;
        default:    osc_freq = 0.0f;                          break;
    }
    return osc_freq;
}

real64_T Get_SYS_PLL_00_Frequency(void)
{
    /* fPLL0 = (N * fOSC) / (P * K2)   ds1 P.1031 */
    real64_T n = (real64_T)SCU_SYSPLLCON0.B.NDIV + 1.0f;
    real64_T p = (real64_T)SCU_SYSPLLCON0.B.PDIV + 1.0f;
    real64_T k2= (real64_T)SCU_SYSPLLCON1.B.K2DIV + 1.0f;
    return (Get_Primary_OSC_Frequency() * n) / (p * k2);
}

real64_T Get_SYS_PLL_01_Frequency(void)
{
    real64_T n = (real64_T)SCU_PERPLLCON0.B.NDIV + 1.0f;
    real64_T p = (real64_T)SCU_PERPLLCON0.B.PDIV + 1.0f;
    real64_T k2= (real64_T)SCU_PERPLLCON1.B.K2DIV + 1.0f;
    return (Get_Primary_OSC_Frequency() * n) / (p * k2);
}

real64_T Get_Source_00_Frequency(void)
{
    real64_T freq;

    switch (SCU_CCUCON0.B.CLKSEL)
    {
        case 0x0U:  freq = Get_Primary_OSC_Frequency();   break;
        case 0x1U:  freq = Get_SYS_PLL_00_Frequency();   break;
        default:    freq = 0.0f;                          break;
    }
    return freq;
}

real64_T Get_Source_01_Frequency(void)
{
    real64_T freq;

    switch (SCU_CCUCON0.B.CLKSEL)
    {
        case 0x0U:
            freq = Get_Primary_OSC_Frequency();
            break;
        case 0x1U:
            freq = Get_SYS_PLL_01_Frequency();
            if (SCU_CCUCON1.B.PLL1DIVDIS == PLL1_DIV_DISABLED)
            {
                freq = freq / 2.0f;
            }
            break;
        default:
            freq = 0.0f;
            break;
    }
    return freq;
}

real64_T Get_ADC_Frequency(void) { return Get_Source_01_Frequency(); }

real64_T Get_SRI_Frequency(void)
{
    real64_T source_freq = Get_Source_00_Frequency();
    real64_T sri_freq;

    switch (SCU_CCUCON0.B.LPDIV)   /* ds1 P.1049 */
    {
        case 0x0U:
            sri_freq = (0x0U == SCU_CCUCON0.B.SRIDIV) ? 0.0f :
                       source_freq / (real64_T)SCU_CCUCON0.B.SRIDIV;
            break;
        case 0x1U:  sri_freq = source_freq / 30.0f;  break;
        case 0x2U:  sri_freq = source_freq / 60.0f;  break;
        case 0x3U:  sri_freq = source_freq / 120.0f; break;
        case 0x4U:  sri_freq = source_freq / 240.0f; break;
        default:    sri_freq = 0.0f;                 break;
    }
    return sri_freq;
}

real64_T Get_CPU_Frequency(void)
{
    real64_T cpu_freq = Get_SRI_Frequency();
    uint32_T cpu_div  = SCU_CCUCON6.B.CPU0DIV;   /* ds1 P.1060 */

    if (0x0U != cpu_div)
    {
        cpu_freq = cpu_freq * ((64.0f - (real64_T)cpu_div) / 64.0f);
    }
    return cpu_freq;
}

real32_T Get_STM_Frequency(void)
{
    real64_T stm_freq = 0.0f;
    uint32_T stm_div  = SCU_CCUCON0.B.STMDIV;

    if (0x0U != stm_div)
    {
        stm_freq = Get_Source_00_Frequency() / (real64_T)stm_div;
    }
    return (real32_T)stm_freq;
}

real64_T Get_SPB_Frequency(void)
{
    real64_T source_freq = Get_Source_00_Frequency();
    real64_T spb_freq;

    switch (SCU_CCUCON0.B.LPDIV)
    {
        case 0x0U:
            spb_freq = (SCU_CCUCON0.B.SPBDIV < 0x2U) ? 0.0f :
                       source_freq / (real64_T)SCU_CCUCON0.B.SPBDIV;
            break;
        case 0x1U:  spb_freq = source_freq / 30.0f;  break;
        case 0x2U:  spb_freq = source_freq / 60.0f;  break;
        case 0x3U:  spb_freq = source_freq / 120.0f; break;
        case 0x4U:  spb_freq = source_freq / 240.0f; break;
        default:    spb_freq = 0.0f;                 break;
    }
    return spb_freq;
}

real64_T Get_GTM_Source_Frequency(void)
{
    if (GTM_SRC_2XSPB == SCU_CCUCON0.B.GTMDIV)
    {
        return 2.0f * Get_SPB_Frequency();
    }
    return Get_Source_00_Frequency();
}

real64_T Get_GTM_Frequency(void)
{
    real64_T gtm_freq = 0.0f;
    uint32_T gtm_div  = SCU_CCUCON0.B.GTMDIV;

    if (0x0U != gtm_div)
    {
        gtm_freq = Get_GTM_Source_Frequency() / (real64_T)gtm_div;
    }
    return gtm_freq;
}

real64_T Get_GTM_Cluster_Frequency(void)
{
    uint32_T cluster_div = GTM_CLS_CLK_CFG.B.CLS0_CLK_DIV;

    if (0x0U == cluster_div)
    {
        return 0.0f;
    }
    return Get_GTM_Frequency() / (real64_T)cluster_div;
}

real64_T Get_GTM_CMU_Global_Frequency(void)
{
    real64_T numerator   = (real64_T)GTM_CMU_GCLK_NUM.B.GCLK_NUM;
    real64_T denominator = (real64_T)GTM_CMU_GCLK_DEN.B.GCLK_DEN;

    if (numerator <= 0.0f)
    {
        return 0.0f;
    }
    return (denominator / numerator) * Get_GTM_Cluster_Frequency();
}

/**********************************************************************************************************************
 * Function Implementations — GTM CMU CLK0
 *********************************************************************************************************************/

void Set_GTM_CMU_CLK_00_Frequency(real64_T CMU_CLK_00_Frequency)
{
    real64_T cmu_global_freq = Get_GTM_CMU_Global_Frequency();
    real64_T cmu_divider     = (cmu_global_freq / CMU_CLK_00_Frequency) - 1.0f;
    uint32_T clk_cnt         = (uint32_T)cmu_divider;

    /* Round to nearest integer */
    if ((cmu_divider - (real64_T)clk_cnt) > 0.5f)
    {
        clk_cnt++;
    }

    Clear_CPU_WDT_EndInit();
    GTM_CMU_CLK_0_CTRL.B.CLK_CNT = clk_cnt;   /* ds2 P.186 */
    Set_CPU_WDT_EndInit();

    GTM_CMU_CLK_EN.B.EN_CLK0 = 0x2U;           /* ds2 P.184 */
}

real64_T Get_GTM_CMU_CLK_00_Frequency(void)
{
    real64_T cmu_clk_freq = 0.0f;

    if (CMU_CLK_ENABLED == GTM_CMU_CLK_EN.B.EN_CLK0)
    {
        real64_T clk_div = (real64_T)GTM_CMU_CLK_0_CTRL.B.CLK_CNT + 1.0f;
        cmu_clk_freq = Get_GTM_CMU_Global_Frequency() / clk_div;
    }
    return cmu_clk_freq;
}

/**********************************************************************************************************************
 * Function Implementations — CPU Interrupt Control
 *********************************************************************************************************************/

uint32_T Is_CPU_Interrupt_Enabled(void)
{
    Ifx_CPU_ICR icr_reg;
    icr_reg.U = (uint32_T)__mfcr(CPU_ICR);
    return (0x1U == icr_reg.B.IE) ? 1U : 0U;
}

uint32_T Disable_CPU_Interrupt(void)
{
    uint32_T prev_state = Is_CPU_Interrupt_Enabled();
    if (1U == prev_state)
    {
        __disable();
    }
    return prev_state;
}

void Restore_CPU_Interrupt(uint32_T Previous_State)
{
    if (1U == Previous_State)
    {
        __enable();
    }
}

/**********************************************************************************************************************
 * Function Implementations — Spinlock
 *********************************************************************************************************************/

uint32_T Acquire_Spin_Lock(uint32_T * const Lock_Ptr)
{
    uint32_T alleged_free  = 0U;   /* Expected: lock is free */
    uint32_T prev_value;

    prev_value = ASM_Cmp_And_Swap(Lock_Ptr, 1U, alleged_free);

    /* If previous value matched alleged_free, swap succeeded -> locked */
    return (prev_value == alleged_free) ? 1U : 0U;
}

void Release_Spin_Lock(uint32_T * const Lock_Ptr)
{
    *Lock_Ptr = 0U;
}
