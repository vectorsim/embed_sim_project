/**********************************************************************************************************************
 * \file      cdd_sys_util.c
 * \brief     Implementation of the CddSys low-level system utility module.
 *
 * \details   Provides the definitions for every function declared in
 *            cdd_sys_util.h. Each function is a thin wrapper over one
 *            TriCore instruction (or a small sequence of them) and is
 *            defined with external linkage so it can be called from any
 *            translation unit in the CDD layer.
 *
 *            All public functions are defined without `extern` and without
 *            `static`: external linkage is the default for a function
 *            definition, and the compatible declaration is visible via the
 *            included header (MISRA C:2012 Rule 8.4).
 *
 *            There are no file-scope variables and no static helpers in this
 *            module - every function maps directly to an assembly sequence
 *            and is called from at least one other translation unit.
 *
 *            For the PMSM application layer that consumes these helpers,
 *            see cdd_app.c. For GTM / ATOM configuration, see cdd_gtm_app.c.
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : All functions have explicit return type
 *              - Rule  8.4 : Compatible declaration visible at every definition
 *              - Rule  8.5 : One declaration per identifier
 *              - Rule  8.6 : No definitions in header files
 *              - Rule  8.7 : No static functions in this module (all are
 *                            referenced from other translation units)
 *              - Rule  8.9 : File scope variables minimised (none here)
 *              - Rule 14.4 : All controlling expressions use explicit comparison
 *              - Rule 15.5 : Single exit point per function
 *              - Rule 17.2 : No recursion
 *              - Rule 18.4 : No non-constant pointer arithmetic
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
 * \version   1.0.0
 * \date      2026-09-25
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_sys_util.h"
#include "IfxCpu_reg.h"
#include "IfxWtu_reg.h"
#include "IfxScu_reg.h"
#include "IfxCpu.h"
#include "IfxEgtm_reg.h"


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None - this module is stateless.                                                   */
/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None - every function maps directly to an instruction sequence.                    */
/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/
/* None - functions are defined bottom-up so no forward prototypes are required.      */
/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/



void CddSys_Ldmst(P2VAR(volatile uint32_T, AUTOMATIC, CDD_APPL_DATA) AddressPtr, uint32_T Mask, uint32_T Value)
{
    __extension__ uint64_T operand = ((uint64_T)Mask << 32) | (uint64_T)Value;

    __asm__ volatile ("ldmst [%0]0,%A1"
                      : /* no output */
                      : "a"(AddressPtr), "d"(operand)
                      : "memory");
}

void CddSys_NopDelay(uint32_T InnerLoop, uint32_T OuterLoop)
{
    uint32_T i;
    uint32_T j;

    for (i = 0U; i < OuterLoop; i++)
    {
        for (j = 0U; j < InnerLoop; j++)
        {
            CddSys_Nop();
        }
    }
}

void CddSys_Nop(void)
{
    __asm__ volatile ("nop" : : : "memory");
}


uint32_T CddSys_CpuCoreId(void)
{
    Ifx_CPU_CORE_ID reg;

    reg.U = CddSys_Mfcr(CPU_CORE_ID);
    return reg.B.CORE_ID;
}


uint32_T CddSys_CmpAndSwap(P2VAR(volatile uint32_T, AUTOMATIC, CDD_APPL_DATA) AddressPtr, uint32_T Value, uint32_T Condition)
{
    __extension__ uint64_T reg64 = (uint64_T)Value | ((uint64_T)Condition << 32);

    __asm__ __volatile__ ("cmpswap.w [%[addr]]0, %A[reg]"
                          : [reg] "+d" (reg64)
                          : [addr] "a" (AddressPtr)
                          : "memory");
    return (uint32_T)reg64;
}


void CddSys_IsrEnable(void)
{
    __asm__ volatile ("enable" : : : "memory");
}

void CddSys_IsrDisable(void)
{
    __asm__ volatile ("disable" : : : "memory");
}

uint16_T CddSys_GetCpuWdtPwd(void)
{
    volatile Ifx_WTU_CTRLA* ctrlAPtr;
    uint16_T       pwd;

    ctrlAPtr = &MODULE_WTU.WDTCPU[CddSys_CpuCoreId()].CTRLA;
    pwd = 0x0U;

    pwd =  ctrlAPtr->B.PW;
    pwd ^= 0x007F;

    return pwd;
}


void CddSys_DisableCpuWdt(void)
{
    uint32_T        coreId;
    uint16_T        password;
    Ifx_WTU_WDTCPU* wdtPtr;
    Ifx_WTU_CTRLA   ctrlA;
    Ifx_WTU_CTRLB   ctrlB;

    coreId   = CddSys_CpuCoreId();
    password = CddSys_GetCpuWdtPwd();
    wdtPtr = &MODULE_WTU.WDTCPU[coreId];
    ctrlA.U = wdtPtr->CTRLA.U;
    ctrlB.U = wdtPtr->CTRLB.U;

    if(ctrlA.B.LCK == 0x1U)
    {
        ctrlA.B.LCK = 0x0U;
        ctrlA.B.PW  = password;
        wdtPtr->CTRLA.U = ctrlA.U;
    }

    ctrlB.B.DR = 0x1U;
    wdtPtr->CTRLB.U = ctrlB.U;

    ctrlA.B.LCK = 0x1U;
    wdtPtr->CTRLA.U = ctrlA.U;
}


uint16_T CddSys_GetSystemWdtPwd(void)
{
    Ifx_WTU_WDTSYS* watchdogPtr;
    uint16_T        password;

    watchdogPtr = &MODULE_WTU.WDTSYS;

    password  = watchdogPtr->CTRLA.B.PW;
    password  ^= 0x007FU;

    return password;
}


void CddSys_DisableSystemWdt(void)
{
   Ifx_WTU_WDTSYS* watchdogPtr;
   Ifx_WTU_CTRLA   ctrlA;
   Ifx_WTU_CTRLB   ctrlB;
   uint16_T        password;

   watchdogPtr = &MODULE_WTU.WDTSYS;
   password    = CddSys_GetSystemWdtPwd();

   ctrlA.U = watchdogPtr->CTRLA.U;
   ctrlB.U = watchdogPtr->CTRLB.U;

   if (ctrlA.B.LCK == 0x1U)
   {
       ctrlA.B.LCK = 0U;
       ctrlA.B.PW  = password;

       watchdogPtr->CTRLA.U = ctrlA.U;
   }

   ctrlB.B.DR = 1U;
   watchdogPtr->CTRLB.U = ctrlB.U;

   ctrlA.B.LCK = 1U;
   watchdogPtr->CTRLA.U = ctrlA.U;
}

 real32_T CddSys_GetEvrFrequency(void)
 {
     return CDD_SYS_EVR_OSC_FREQUENCY;
 }


 real32_T CddSys_GetOscFrequency(void)
 {
     real32_T freq;

     switch (CLOCK_OSCCON.B.INSEL)
     {
         case 0x0U:
             freq = CddSys_GetEvrFrequency();
             break;

         case 0x1U:
             freq = CDD_XTAL_FREQUENCY;
             break;

         case 0x2U:
             freq = CDD_CFG_OSC_F_FREQ;
             break;

         default:
             freq = CddSys_GetEvrFrequency();
             break;
     }

     return freq;
 }


 real32_T CddSys_GetPllFrequency(void)
 {
     Ifx_CLOCK *clock = &MODULE_CLOCK;
     real32_T  oscFreq;
     real32_T  freq;
     uint8     preDiv[4] = {10u, 20u, 12u, 16u};

     oscFreq = CddSys_GetOscFrequency();

     freq = (oscFreq * (clock->SYSPLLCON0.B.NDIV + 1U) * 10.0F)
          / ((clock->SYSPLLCON0.B.PDIV + 1U)
          *  (clock->SYSPLLCON1.B.K2DIV + 1U)
          *  (preDiv[clock->SYSPLLCON1.B.K2PREDIV]));

     freq = freq
          * (real32_T)clock->SYSPLLSTAT.B.PLLLOCK
          * (real32_T)clock->SYSPLLSTAT.B.PWRSTAT;

     return freq;
 }

 real32_T CddSys_GetRampFrequency(void)
 {
     real32_T freq = 0.0F;

     if(CLOCK_RAMPSTAT.B.SSTAT == 0x0U)  /* idle */
     {
         switch(CLOCK_RAMPSTAT.B.FSTAT)
         {
             case 0x2U:  /* at top */
                 freq = (real32_T)CLOCK_RAMPCON0.B.UFL * 1000000.0F;
                 break;
             case 0x1U: /* ramp up */
                 freq = MHZ_100;
                 break;
             default:
                 freq = 0.0F;
                 break;
         }
     }
     else
     {
         freq = 0.0F;
     }

     return freq;
 }


 real32_T CddSys_GetSysSourceFrequency(void)
 {
     real32_T sourcefreq = 0.0F;

     switch (CLOCK_CCUSTAT.B.CLKSELS)
     {
         case 0x0U:
             /* System PLL is selected. */
             sourcefreq = CddSys_GetPllFrequency();
             break;

         case 0x1U:
             /* EVR backup clock (fBACK) is selected. */
             sourcefreq = CddSys_GetEvrFrequency();
             break;

         case 0x2U:
             /* Ramp clock is selected. */
             sourcefreq = CddSys_GetRampFrequency();
             break;

         default:
             /* Invalid or unsupported clock source selection. */
             sourcefreq = 0.0F;
             break;
     }

     return sourcefreq;
 }


 real32_T CddSys_GetEgtmFrequency(void)
 {
     uint32_T eGtmDiv;
     real32_T gtmFreq;

     real32_T clkdiv[16] =
     {
         1.0F,  1.0F,  2.0F,  3.0F,
         4.0F,  5.0F,  6.0F,  6.0F,
         8.0F,  8.0F, 10.0F, 10.0F,
        12.0F, 12.0F, 12.0F, 15.0F
     };

     gtmFreq = 0.0F;
     eGtmDiv = CLOCK_SYSCCUCON1.B.EGTMDIV;

     if (CLOCK_SYSCCUCON0.B.LPDIV == 0x0U)
     {
         if (eGtmDiv != 0x0U)
         {
             gtmFreq = CddSys_GetSysSourceFrequency() / clkdiv[eGtmDiv];
         }
     }
     else
     {
         gtmFreq = CddSys_GetSysSourceFrequency() / 120.0F;
     }

     return gtmFreq;
 }

 real32_T CddSys_GetEgtmCls0Frequency(void)
 {
     real32_T frequency;
     uint32_T clusterDivider;

     frequency      = 0.0F;
     clusterDivider = EGTM_CLS0_ARCH_CLK_CFG.B.CLS0_CLK_DIV;

     if(clusterDivider == 0x0u)
     {
         frequency = 0.0F;   /* cluster disabled */
     }
     else
     {
         frequency = CddSys_GetEgtmFrequency() / (real32_T)clusterDivider;
     }

     return frequency;
 }


 real32_T CddSys_GetEgtmCmuFrequency(void)
 {
     real32_T numerator;
     real32_T denominator;
     real32_T moduleFrequency;
     real32_T cmuFrequency;

     cmuFrequency    = 0.0F;
     denominator     = 0.0F;
     moduleFrequency = 0.0F;

     moduleFrequency = CddSys_GetEgtmCls0Frequency();
     numerator    = (real32_T)EGTM_CLS0_CMU_GCLK_NUM.B.GCLK_NUM;
     denominator  = (real32_T)EGTM_CLS0_CMU_GCLK_DEN.B.GCLK_DEN;
     cmuFrequency = (denominator / numerator) * moduleFrequency;

     return cmuFrequency;
 }

 void CddSys_SetEgtmCmuFrequency(real32_T Frequency)
 {
     real32_T bestDistance;
     real32_T fIn;
     real32_T t;
     real32_T f;
     real32_T distance;
     uint32_T z;
     uint32_T n;
     uint32_T nBest;
     uint32_T zBest;
     boolean  endLoop;

     bestDistance = Frequency;
     fIn          = CddSys_GetEgtmCls0Frequency();
     nBest        = 1U;
     zBest        = 1U;

     for (z = 1U; z < 0xFFFFFFU; z++)
     {
         endLoop = FALSE;
         t = fIn / (real32_T)z;

         for (n = z; n > 0U; n--)
         {
             f        = t * (real32_T)n;
             distance = fabsf(Frequency - f);

             if (distance < bestDistance)
             {
                 bestDistance = distance;
                 nBest        = n;
                 zBest        = z;
             }

             if (bestDistance < 0.1F)
             {
                 endLoop = TRUE;
                 break;
             }
         }

         if (endLoop == TRUE)
         {
             break;
         }
     }

     EGTM_CLS0_CMU_GCLK_NUM.B.GCLK_NUM = zBest;
     EGTM_CLS0_CMU_GCLK_NUM.B.GCLK_NUM = zBest;
     EGTM_CLS0_CMU_GCLK_DEN.B.GCLK_DEN = nBest;
 }


 void CddSys_SetEgtmCmu0Frequency(real32_T Frequency)
 {
     real32_T t   = (CddSys_GetEgtmCmuFrequency() / Frequency) - (real32_T)1.0F;
     uint32_T cnt = (uint32)t;

     if((t - (real32_T)cnt) > (real32_T)0.5F)
     {
         cnt++;
     }

     EGTM_CLS0_CMU_CLK0_CTRL.B.CLK_CNT = cnt;
 }

 real32_T CddSys_GetEgtmCmu0Frequency(void)
 {
     real32_T cmuClk0Frequency;
     real32_T cmuGlobalfrequency;
     real32_T clkDiv;

     cmuClk0Frequency   = 0.0F;
     clkDiv             = 0.0F;
     cmuGlobalfrequency = 0.0F;

     if(EGTM_CLS0_CMU_CLK_EN.B.EN_CLK0 == 0x3U)
     {
         cmuGlobalfrequency = CddSys_GetEgtmCmuFrequency();
         clkDiv = (real32_T)EGTM_CLS0_CMU_CLK0_CTRL.B.CLK_CNT + 1.0F;
         cmuClk0Frequency  = cmuGlobalfrequency / clkDiv;
     }

     return cmuClk0Frequency;
 }

 uint32_T CddSys_AreEqual32(real32_T Lhs, real32_T Rhs, real32_T Epsilon)
 {
     real32_T diff   = Lhs - Rhs;
     uint32_T result = FALSE;

     if(diff < 0.0F)
     {
         diff = -diff;
     }
     result = (diff <= Epsilon) ? TRUE : FALSE;

     return result;
 }

 void CddSys_EnableEgtmCmu0Clock(void)
 {
     EGTM_CLS0_CMU_CLK_EN.B.EN_CLK0 = 0x2U;
 }

 void CddSys_EnableEgtmClock(void)
 {
     /* Enable EGTM Module */
     EGTM_CLC.B.DISR  = 0x0u;
     EGTM_CLS0_ARCH_CTRL.B.RF_PROT  = 0x0u;
     EGTM_CLS0_CCM_PROT.B.CLS_PROT  = 0x0u;

 }

