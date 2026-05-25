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
 *                  0x11  push-pull output alt-func 1  (GTM ATOM TOUT)
 *                  0x12  push-pull output alt-func 2  (QSPI4 CS)
 *                  0x10  push-pull output general      (debug GPIO)
 *                  0x00  input, no pull device         (QSPI4 MISO)
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
 * Private Macros — IOCR PC field values
 *********************************************************************************************************************/

/** \brief  Push-pull output, alternate function 1  (GTM ATOM TOUT routing)  */
#define GPIO_PC_PP_ALT1         (0x11U)

/** \brief  Push-pull output, alternate function 2  (QSPI4 SLSO3 routing)    */
#define GPIO_PC_PP_ALT2         (0x12U)

/** \brief  Input, no pull device  (QSPI4 MRST / MISO)                       */
#define GPIO_PC_INPUT_NP        (0x00U)

/** \brief  Push-pull output, general-purpose  (debug / ISR timing probe)    */
#define GPIO_PC_PP_GP           (0x10U)

/**********************************************************************************************************************
 * Private Macros — PDR PD field values
 *********************************************************************************************************************/

/** \brief  Automotive CMOS speed-3  (~80 pF, GTM switching)                 */
#define GPIO_PD_SPEED3          (0x3U)

/** \brief  Automotive CMOS speed-1  (low-speed debug output)                */
#define GPIO_PD_SPEED1          (0x0U)

/**********************************************************************************************************************
 * QSPI4 Pin Mux
 *********************************************************************************************************************/

void CddGpio_ConfigQspi4Pins(void)
{
    /* IOCR (not EndInit-protected) */
    P22_IOCR0.B.PC0 = GPIO_PC_PP_ALT1;    /* MOSI: push-pull alt-func 1    */
    P22_IOCR0.B.PC1 = GPIO_PC_INPUT_NP;   /* MISO: input, no pull          */
    P22_IOCR0.B.PC2 = GPIO_PC_PP_ALT2;    /* CS  : push-pull alt-func 2    */
    P22_IOCR0.B.PC3 = GPIO_PC_PP_ALT1;    /* SCLK: push-pull alt-func 1    */

    /* PDR (EndInit-protected) */
    CddSys_ClearWdtEndInit();
    P22_PDR0.B.PD0 = GPIO_PD_SPEED3;      /* MOSI speed-3                  */
    P22_PDR0.B.PD2 = GPIO_PD_SPEED3;      /* CS   speed-3                  */
    P22_PDR0.B.PD3 = GPIO_PD_SPEED3;      /* SCLK speed-3                  */
    CddSys_SetWdtEndInit();
}

/**********************************************************************************************************************
 * GTM ATOM Pin Mux Helpers
 *********************************************************************************************************************/

void CddGpio_ConfigGtmMaster_P00_0(void)
{
    P00_IOCR0.B.PC0 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD0  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

void CddGpio_ConfigGtmPhaseULs_P00_2(void)
{
    P00_IOCR0.B.PC2 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD2  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

void CddGpio_ConfigGtmPhaseUHs_P00_3(void)
{
    P00_IOCR0.B.PC3 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD3  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

void CddGpio_ConfigGtmPhaseVLs_P00_4(void)
{
    P00_IOCR4.B.PC4 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD4  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

void CddGpio_ConfigGtmPhaseVHs_P00_5(void)
{
    P00_IOCR4.B.PC5 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD5  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

void CddGpio_ConfigGtmPhaseWLs_P00_6(void)
{
    P00_IOCR4.B.PC6 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD6  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

void CddGpio_ConfigGtmPhaseWHs_P00_7(void)
{
    P00_IOCR4.B.PC7 = GPIO_PC_PP_ALT1;
    CddSys_ClearWdtEndInit();
    P00_PDR0.B.PD7  = GPIO_PD_SPEED3;
    CddSys_SetWdtEndInit();
}

/**********************************************************************************************************************
 * ISR Timing Probe — P14.5
 *********************************************************************************************************************/

void CddGpio_ConfigIsrTiming_P14_5(void)
{
    P14_IOCR4.B.PC5 = GPIO_PC_PP_GP;
    CddSys_ClearWdtEndInit();
    P14_PDR0.B.PD5  = GPIO_PD_SPEED1;
    CddSys_SetWdtEndInit();
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
 * TLE9180D Gate Driver Control Pins
 *********************************************************************************************************************/

void CddGpio_ConfigGd9180Pins(void)
{
    /* P20.0 /INH — output, push-pull GP, speed-1, init LOW */
    P20_IOCR0.B.PC0  = GPIO_PC_PP_GP;
    CddSys_ClearWdtEndInit();
    P20_PDR0.B.PD0   = GPIO_PD_SPEED1;
    CddSys_SetWdtEndInit();
    P20_OMR.B.PCL0   = 0x1U;

    /* P33.10 /SOFF — output, push-pull GP, speed-1, init HIGH (/SOFF=HIGH = normal) */
    P33_IOCR8.B.PC10 = GPIO_PC_PP_GP;
    CddSys_ClearWdtEndInit();
    P33_PDR1.B.PD10  = GPIO_PD_SPEED1;
    CddSys_SetWdtEndInit();
    P33_OMR.B.PS10   = 0x1U;

    /* P33.11 ENA — output, push-pull GP, speed-1, init LOW */
    P33_IOCR8.B.PC11 = GPIO_PC_PP_GP;
    CddSys_ClearWdtEndInit();
    P33_PDR1.B.PD11  = GPIO_PD_SPEED1;
    CddSys_SetWdtEndInit();
    P33_OMR.B.PCL11  = 0x1U;

    /* P15.2 /ERR — input, pull-up (0x02 = input pull-up device) */
    P15_IOCR0.B.PC2  = 0x02U;
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
    P33_IOCR4.B.PC4 = GPIO_PC_PP_GP;
    P33_IOCR4.B.PC5 = GPIO_PC_PP_GP;
    P33_IOCR4.B.PC6 = GPIO_PC_PP_GP;
    P33_IOCR4.B.PC7 = GPIO_PC_PP_GP;

    /* PDR: medium automotive driver  (PDR0 covers pins 0–7, EndInit-protected) */
    CddSys_ClearWdtEndInit();
    P33_PDR0.B.PD4  = 0x2U;
    P33_PDR0.B.PL4  = 0x0U;
    P33_PDR0.B.PD5  = 0x2U;
    P33_PDR0.B.PL5  = 0x0U;
    P33_PDR0.B.PD6  = 0x2U;
    P33_PDR0.B.PL6  = 0x0U;
    P33_PDR0.B.PD7  = 0x2U;
    P33_PDR0.B.PL7  = 0x0U;
    CddSys_SetWdtEndInit();

    /* Drive all LOW (LED off) */
    P33_OMR.B.PCL4  = 0x1U;
    P33_OMR.B.PCL5  = 0x1U;
    P33_OMR.B.PCL6  = 0x1U;
    P33_OMR.B.PCL7  = 0x1U;
}

void CddGpio_ToggleLed_P33_4(void) { P33_OMR.U = 0x00100010U; }
void CddGpio_ToggleLed_P33_5(void) { P33_OMR.U = 0x00200020U; }
void CddGpio_ToggleLed_P33_6(void) { P33_OMR.U = 0x00400040U; }
void CddGpio_ToggleLed_P33_7(void) { P33_OMR.U = 0x00800080U; }

void CddGpio_SetLed_P33_4(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH) { P33_OMR.B.PS4  = 0x1U; }
    else                             { P33_OMR.B.PCL4 = 0x1U; }
}
void CddGpio_SetLed_P33_5(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH) { P33_OMR.B.PS5  = 0x1U; }
    else                             { P33_OMR.B.PCL5 = 0x1U; }
}
void CddGpio_SetLed_P33_6(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH) { P33_OMR.B.PS6  = 0x1U; }
    else                             { P33_OMR.B.PCL6 = 0x1U; }
}
void CddGpio_SetLed_P33_7(CddGpio_Level_T Level)
{
    if (Level == CDDGPIO_LEVEL_HIGH) { P33_OMR.B.PS7  = 0x1U; }
    else                             { P33_OMR.B.PCL7 = 0x1U; }
}
