/**********************************************************************************************************************
 * \file        cdd_gpio_app.c
 * \brief       Implementation of cdd_gpio_app.h — GPIO pad configuration for
 *              the AURIX TC3xx Motor Control Power Board (AP32541).
 *
 * \details     All writes follow TC3xx Reference Manual (ds1) port register map.
 *              Each function writes exactly two registers per pin:
 *                  IOCRx — output mode (alternate function + direction)
 *                  PDRx  — pad driver strength (automotive speed class)
 *
 *              IOCR PC field encoding (ds1 P.1011):
 *                  0x11  push-pull output alt-func 1  (GTM ATOM TOUT — O1)
 *                  0x13  push-pull output alt-func 3  (QSPI4 MTSR/SLSO3/SCLK — O3)
 *                  0x10  push-pull output general      (debug GPIO)
 *                  0x00  input, no pull device         (QSPI4 MISO)
 *                  0x02  input, pull-up device         (/ERR input)
 *
 *              PDR PD field encoding (ds1 P.1017):
 *                  0x3   automotive CMOS speed-3  (~80 pF, GTM outputs)
 *                  0x0   automotive CMOS speed-1  (debug probe)
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : No file-scope variables (stateless driver)
 *              - Rule 14.4  : No implicit Boolean conversions
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#include "cdd_gpio_app.h"
#include "cdd_sys_utility.h"   /* CddSys_ClearWdtEndInit / CddSys_SetWdtEndInit */
#include "IfxPort_reg.h"

/**********************************************************************************************************************
 * QSPI4 Pin Mux Helpers
 *********************************************************************************************************************/

void CddGpio_ConfigQspi4Mosi_P22_0(void)
{
    /* QSPI4_MTSR = O3 → PC = 0x13 (push-pull alt-func 3)  appx2 W25 */
    P22_IOCR0.B.PC0 = 0x13U;
    CddSys_ClearCpuWdtEndInit();
    P22_PDR0.B.PD0 = 0x2U;
    P22_PDR0.B.PL0 = 0x0U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigQspi4Miso_P22_1(void)
{
    /* QSPI4_MRST input — PC = 0x01 (input, pull-down)  appx2 W24
     * Pull-down defines a safe idle level between frames; tri-state (0x00)
     * leaves MISO floating when TLE9180D is not driving.                    */
    P22_IOCR0.B.PC1 = 0x01U;
    /* Input pin — PDR write not required */
}

void CddGpio_ConfigQspi4Cs_P22_2(void)
{
    /* QSPI4_SLSO3 = O3 → PC = 0x13 (push-pull alt-func 3)  appx2 Y25 */
    P22_IOCR0.B.PC2 = 0x13U;
    CddSys_ClearCpuWdtEndInit();
    P22_PDR0.B.PD2  = 0x2U;
    P22_PDR0.B.PL2  = 0x0U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigQspi4Sclk_P22_3(void)
{
    /* QSPI4_SCLK = O3 → PC = 0x13 (push-pull alt-func 3)  appx2 Y24 */
    P22_IOCR0.B.PC3 = 0x13U;
    CddSys_ClearCpuWdtEndInit();
    P22_PDR0.B.PD3  = 0x2U;
    P22_PDR0.B.PL3  = 0x0U;
    CddSys_SetCpuWdtEndInit();
}

/**********************************************************************************************************************
 * GTM ATOM Pin Mux Helpers
 *********************************************************************************************************************/

void CddGpio_ConfigGtmMaster_P00_0(void)
{
    /* GTM_TOUT9 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 M6 */
    P00_IOCR0.B.PC0 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD0  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseULs_P00_2(void)
{
    /* GTM_TOUT11 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 N6 */
    P00_IOCR0.B.PC2 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD2  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseUHs_P00_3(void)
{
    /* GTM_TOUT12 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 N5 */
    P00_IOCR0.B.PC3 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD3  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseVLs_P00_4(void)
{
    /* GTM_TOUT13 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 P6 */
    P00_IOCR4.B.PC4 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD4  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseVHs_P00_5(void)
{
    /* GTM_TOUT14 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 P5 */
    P00_IOCR4.B.PC5 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD5  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseWLs_P00_6(void)
{
    /* GTM_TOUT15 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 P4 */
    P00_IOCR4.B.PC6 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD6  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseWHs_P00_7(void)
{
    /* GTM_TOUT16 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 R6 */
    P00_IOCR4.B.PC7 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR0.B.PD7  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

void CddGpio_ConfigGtmPhaseADCTrigger_P00_8(void)
{
    /* GTM_TOUT17 = O1 → PC = 0x11 (push-pull alt-func 1)  appx2 R6 */
    P00_IOCR8.B.PC8 = 0x11U;
    CddSys_ClearCpuWdtEndInit();
    P00_PDR1.B.PD8  = 0x3U;
    CddSys_SetCpuWdtEndInit();
}

/**********************************************************************************************************************
 * ISR Timing Probe — P14.5
 *********************************************************************************************************************/

void CddGpio_ConfigIsrTiming_P14_5(void)
{
    P14_IOCR4.B.PC5 = 0x10U;   /* push-pull GP output */
    CddSys_ClearCpuWdtEndInit();
    P14_PDR0.B.PD5  = 0x0U;    /* automotive CMOS speed-1 */
    CddSys_SetCpuWdtEndInit();
    P14_OMR.B.PCL5  = 0x1U;   /* drive LOW on init */
}

void CddGpio_ToggleIsrTiming_P14_5(void)
{
    if (P14_OUT.B.P5 == 0x0U)
    {
        P14_OMR.B.PS5  = 0x1U;
    }
    else
    {
        P14_OMR.B.PCL5 = 0x1U;
    }
}

/**********************************************************************************************************************
 * TLE9180D Gate Driver Pin Mux Helpers
 *********************************************************************************************************************/

void CddGpio_ConfigGd9180Inh_P20_0(void)
{
    /* /INH — active-LOW inhibit, init LOW (gate driver sleep on boot)       */
    P20_IOCR0.B.PC0 = 0x10U;   /* push-pull GP output */
    CddSys_ClearCpuWdtEndInit();
    P20_PDR0.B.PD0  = 0x0U;    /* automotive CMOS speed-1 */
    CddSys_SetCpuWdtEndInit();
    P20_OMR.B.PCL0  = 0x1U;               /* drive LOW on init             */
}

void CddGpio_ConfigGd9180Soff_P33_10(void)
{
    /* /SOFF — active-LOW safe-off, init HIGH (normal operation)             */
    P33_IOCR8.B.PC10 = 0x10U;  /* push-pull GP output */
    CddSys_ClearCpuWdtEndInit();
    P33_PDR1.B.PD10  = 0x0U;   /* automotive CMOS speed-1 */
    CddSys_SetCpuWdtEndInit();
    P33_OMR.B.PS10   = 0x1U;              /* drive HIGH on init            */
}

void CddGpio_ConfigGd9180Ena_P33_11(void)
{
    /* ENA — active-HIGH enable, init LOW (outputs disabled on boot)         */
    P33_IOCR8.B.PC11 = 0x10U;  /* push-pull GP output */
    CddSys_ClearCpuWdtEndInit();
    P33_PDR1.B.PD11  = 0x0U;   /* automotive CMOS speed-1 */
    CddSys_SetCpuWdtEndInit();
    P33_OMR.B.PCL11  = 0x1U;             /* drive LOW on init             */
}

void CddGpio_ConfigGd9180Err_P15_2(void)
{
    /* /ERR — active-LOW error flag, input with pull-up                      */
    P15_IOCR0.B.PC2  = 0x02U;            /* input pull-up device          */
    /* Error pin is an input — PDR write not required                       */
}

/**********************************************************************************************************************
 * TLE9180D Control Output Drivers
 *********************************************************************************************************************/

void CddGpio_SetInh_P20_0(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P20_OMR.B.PS0  = 0x1U;
    }
    else
    {
        P20_OMR.B.PCL0 = 0x1U;
    }
}

void CddGpio_SetEna_P33_11(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS11  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL11 = 0x1U;
    }
}

void CddGpio_SetSoff_P33_10(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS10  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL10 = 0x1U;
    }
}

uint32_T CddGpio_GetErr_P15_2(void)
{
    return (uint32_T)P15_IN.B.P2;
}

/**********************************************************************************************************************
 * Debug LEDs — P33.4 – P33.7
 *
 * Toggle: single 32-bit write to P33_OMR.U with PSx=1 and PCLx=1 in the
 * same word — hardware XORs the output latch (true atomic toggle).
 * Bit positions: pin 4 → 0x00100010U | 5 → 0x00200020U | 6 → 0x00400040U | 7 → 0x00800080U
 *********************************************************************************************************************/

void CddGpio_InitLed_P33(void)
{
    /* IOCR: push-pull GP = 0x10  (IOCR4 covers pins 4–7) */
    P33_IOCR4.B.PC4 = 0x10U;
    P33_IOCR4.B.PC5 = 0x10U;
    P33_IOCR4.B.PC6 = 0x10U;
    P33_IOCR4.B.PC7 = 0x10U;

    /* PDR: medium automotive driver  (PDR0 covers pins 0–7, EndInit-protected) */
    CddSys_ClearCpuWdtEndInit();
    P33_PDR0.B.PD4  = 0x2U;
    P33_PDR0.B.PL4  = 0x0U;
    P33_PDR0.B.PD5  = 0x2U;
    P33_PDR0.B.PL5  = 0x0U;
    P33_PDR0.B.PD6  = 0x2U;
    P33_PDR0.B.PL6  = 0x0U;
    P33_PDR0.B.PD7  = 0x2U;
    P33_PDR0.B.PL7  = 0x0U;
    CddSys_SetCpuWdtEndInit();

    /* Drive all LOW (LED off) */
    P33_OMR.B.PCL4  = 0x1U;
    P33_OMR.B.PCL5  = 0x1U;
    P33_OMR.B.PCL6  = 0x1U;
    P33_OMR.B.PCL7  = 0x1U;
}

void CddGpio_ToggleLed_P33_4(void)
{
    /* PS4=1 and PCL4=1 in same word — hardware XORs output latch (atomic toggle) */
    P33_OMR.U = 0x00100010U;
}

void CddGpio_ToggleLed_P33_5(void)
{
    P33_OMR.U = 0x00200020U;
}

void CddGpio_ToggleLed_P33_6(void)
{
    P33_OMR.U = 0x00400040U;
}

void CddGpio_ToggleLed_P33_7(void)
{
    P33_OMR.U = 0x00800080U;
}

void CddGpio_SetLed_P33_4(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS4  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL4 = 0x1U;
    }
}

void CddGpio_SetLed_P33_5(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS5  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL5 = 0x1U;
    }
}

void CddGpio_SetLed_P33_6(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS6  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL6 = 0x1U;
    }
}

void CddGpio_SetLed_P33_7(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS7  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL7 = 0x1U;
    }
}
