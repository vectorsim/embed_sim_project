/**********************************************************************************************************************
 * \file        cdd_gpio_app.c
 * \brief       Implementation of cdd_gpio_app.h — GPIO pad configuration for
 *              the AURIX TC3xx Motor Control Power Board (AP32541).
 *
 * \details     All writes follow the TC3xx Reference Manual (ds1) port register
 *              map.  Each function writes exactly two registers per pin:
 *
 *                  IOCRx   — output mode (alternate function + direction)
 *                  PDRx    — pad driver strength (automotive speed class)
 *
 *              IOCR field encoding  (TC3xx ds1 P.1011):
 *                  PC[4:0] = 0x11  push-pull output alt-func 1  (GTM ATOM TOUT)
 *                  PC[4:0] = 0x10  push-pull output general      (debug GPIO)
 *
 *              PDR field encoding   (TC3xx ds1 P.1017):
 *                  PD[3:0] = 0x3   automotive CMOS speed-3  (~80 pF, GTM outputs)
 *                  PD[3:0] = 0x0   automotive CMOS speed-1  (debug probe)
 *
 *              IOCR registers are byte-addressed in groups of 4 pins.
 *              PDR  registers are word-addressed in groups of 8 pins.
 *
 *              Register naming convention (flat TASKING / Infineon headers):
 *                  P00_IOCR0   — P00 pins 0–3   (bits [31:24]=PC3 .. [7:0]=PC0)
 *                  P00_IOCR4   — P00 pins 4–7
 *                  P00_PDR0    — P00 pins 0–7
 *                  P14_IOCR4   — P14 pins 4–7
 *                  P14_PDR0    — P14 pins 0–7
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables: none (stateless driver)
 *              - Rule 14.4  : No implicit Boolean conversions
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_gpio_app.h"
#include "cdd_sys_utility.h"
#include "IfxPort_reg.h"       /* P00_IOCR0, P00_PDR0, P14_IOCR4, P14_PDR0   */

/**********************************************************************************************************************
 * Private Macros — IOCR PC field values
 *
 * Each PCx field occupies bits [4:0] within its byte of the IOCRx register.
 * The byte position within the 32-bit IOCRx word depends on the pin number
 * within the group of four:
 *   pin % 4 == 0 → bits [12:8]   (byte 1)
 *   pin % 4 == 1 → bits [20:16]  (byte 2)
 *   pin % 4 == 2 → bits [28:24]  (byte 3)   ← note: TASKING .B accessor names
 *   pin % 4 == 3 → bits [4:0]    (byte 0)       differ; we use .B.PCx directly
 *
 * Using the Infineon flat-register .B accessor is the cleanest MISRA-safe way.
 *********************************************************************************************************************/

/** \brief  Push-pull output, alternate function 1  (GTM ATOM TOUT routing)  */
#define GPIO_PC_PP_ALT1         (0x11U)

/** \brief  Push-pull output, general-purpose  (debug / ISR timing probe)    */
#define GPIO_PC_PP_GP           (0x10U)

/**********************************************************************************************************************
 * Private Macros — PDR PD field values
 *********************************************************************************************************************/

/** \brief  Automotive CMOS speed-3  (~80 pF, suitable for GTM switching)    */
#define GPIO_PD_SPEED3          (0x3U)

/** \brief  Automotive CMOS speed-1  (low-speed debug output)                */
#define GPIO_PD_SPEED1          (0x0U)

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
static void GPIO_Configure_GD9180_Pins(void);

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_Master_P00_0
 * P00.0 — ATOM0_CH0 master PWM / scope probe
 * IOCR0.PC0  = PP_ALT1
 * PDR0.PD0   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_Master_P00_0(void)
{
    P00_IOCR0.B.PC0 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD0  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_PhaseU_LS_P00_2
 * P00.2 — DTM0_CH0 IL1 low-side Phase U
 * IOCR0.PC2  = PP_ALT1
 * PDR0.PD2   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_PhaseU_LS_P00_2(void)
{
    P00_IOCR0.B.PC2 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD2  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_PhaseU_HS_P00_3
 * P00.3 — DTM0_CH0 /IH1 high-side Phase U
 * IOCR0.PC3  = PP_ALT1
 * PDR0.PD3   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_PhaseU_HS_P00_3(void)
{
    P00_IOCR0.B.PC3 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD3  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_PhaseV_LS_P00_4
 * P00.4 — DTM0_CH1 IL2 low-side Phase V
 * IOCR4.PC4  = PP_ALT1
 * PDR0.PD4   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_PhaseV_LS_P00_4(void)
{
    P00_IOCR4.B.PC4 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD4  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_PhaseV_HS_P00_5
 * P00.5 — DTM0_CH1 /IH2 high-side Phase V
 * IOCR4.PC5  = PP_ALT1
 * PDR0.PD5   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_PhaseV_HS_P00_5(void)
{
    P00_IOCR4.B.PC5 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD5  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_PhaseW_LS_P00_6
 * P00.6 — DTM0_CH2 IL3 low-side Phase W
 * IOCR4.PC6  = PP_ALT1
 * PDR0.PD6   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_PhaseW_LS_P00_6(void)
{
    P00_IOCR4.B.PC6 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD6  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GTM_PhaseW_HS_P00_7
 * P00.7 — DTM0_CH2 /IH3 high-side Phase W
 * IOCR4.PC7  = PP_ALT1
 * PDR0.PD7   = SPEED3
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_GTM_PhaseW_HS_P00_7(void)
{
    P00_IOCR4.B.PC7 = GPIO_PC_PP_ALT1;

    Clear_CPU_WDT_EndInit();
    P00_PDR0.B.PD7  = GPIO_PD_SPEED3;    /* P00_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_ISR_Timing_P14_5
 * P14.5 — ISR timing probe (push-pull GP output, speed-1)
 * IOCR4.PC5  = PP_GP   (P14 IOCR4 covers pins 4–7)
 * PDR0.PD5   = SPEED1
 * Initial output level: LOW
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Configure_ISR_Timing_P14_5(void)
{
    P14_IOCR4.B.PC5 = GPIO_PC_PP_GP;

    Clear_CPU_WDT_EndInit();
    P14_PDR0.B.PD5  = GPIO_PD_SPEED1;    /* P14_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();

    P14_OMR.B.PCL5  = 0x1U;              /* drive LOW on init                    */
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Toggle_ISR_Timing_P14_5
 * Toggles P14.5 by reading the current output latch (P14_OUT.B.P5) and
 * using OMR set (PS5) or clear (PCL5) — single write, no read-modify-write
 * race on the output register.
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Toggle_ISR_Timing_P14_5(void)
{
    if (P14_OUT.B.P5 == 0x0U)
    {
        P14_OMR.B.PS5  = 0x1U;        /* set HIGH                             */
    }
    else
    {
        P14_OMR.B.PCL5 = 0x1U;        /* set LOW                              */
    }
}

/**********************************************************************************************************************
 * TLE9180D Gate Driver Control Pins
 *
 * All three outputs are push-pull GP, speed-1 (low-frequency control signals).
 * Init is performed inside Initialize_GPIO_Module() via GD9180_Init_Pins().
 *
 * P20.0   /INH   — output, push-pull, IOCR0.PC0, PDR0.PD0
 * P33.11  ENA    — output, push-pull, IOCR8.PC11 (pins 8–11), PDR1.PD11 (pins 8–15)
 * P33.10  /SOFF  — output, push-pull, IOCR8.PC10, PDR1.PD10
 * P15.2   /ERR   — input,  pull-up,   IOCR0.PC2  (pins 0–3), PDR0.PD2
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Configure_GD9180_Pins  (called from Initialize_GPIO_Module)
 *------------------------------------------------------------------------------------------------------------------*/
static void GPIO_Configure_GD9180_Pins(void)
{
    /* P20.0 /INH — output, push-pull GP, speed-1, init LOW */
    P20_IOCR0.B.PC0 = GPIO_PC_PP_GP;
    Clear_CPU_WDT_EndInit();
    P20_PDR0.B.PD0  = GPIO_PD_SPEED1;    /* P20_PDR0 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
    P20_OMR.B.PCL0  = 0x1U;

    /* P33.10 /SOFF — output, push-pull GP, speed-1, init HIGH (/SOFF=HIGH = normal path) */
    P33_IOCR8.B.PC10 = GPIO_PC_PP_GP;
    Clear_CPU_WDT_EndInit();
    P33_PDR1.B.PD10  = GPIO_PD_SPEED1;   /* P33_PDR1 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
    P33_OMR.B.PS10   = 0x1U;

    /* P33.11 ENA — output, push-pull GP, speed-1, init LOW */
    P33_IOCR8.B.PC11 = GPIO_PC_PP_GP;
    Clear_CPU_WDT_EndInit();
    P33_PDR1.B.PD11  = GPIO_PD_SPEED1;   /* P33_PDR1 is EndInit-protected        */
    Set_CPU_WDT_EndInit();
    P33_OMR.B.PCL11  = 0x1U;

    /* P15.2 /ERR — input, pull-up (0x02 = input pull-up device) */
    P15_IOCR0.B.PC2 = 0x02U;             /* IOCR is not EndInit-protected        */
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Set_INH_P20_0
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Set_INH_P20_0(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P20_OMR.B.PS0  = 0x1U;
    }
    else
    {
        P20_OMR.B.PCL0 = 0x1U;
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Set_ENA_P33_11
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Set_ENA_P33_11(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS11  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL11 = 0x1U;
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Set_SOFF_P33_10
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Set_SOFF_P33_10(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS10  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL10 = 0x1U;
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Get_ERR_P15_2
 * Returns 1 if P15.2 is HIGH (/ERR = no fault), 0 if LOW (fault active).
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T GPIO_Get_ERR_P15_2(void)
{
    return (uint32_T)P15_IN.B.P2;
}

/**********************************************************************************************************************
 * Debug LEDs — P33.4 – P33.7
 *
 * Pattern follows port_utility.c:
 *   IOCR:  PC = 0x10  (push-pull GP output)
 *   PDR:   PD = 0x2U, PL = 0x0U  (medium automotive driver)
 *   Init:  PCLx = 1  (drive LOW = LED off)
 *   Toggle: single 32-bit write to OMR.U with PSx=1 and PCLx=1 set in the
 *           same word — hardware XORs the output latch (true atomic toggle).
 *           Two sequential .B field writes are NOT equivalent: the compiler
 *           emits two separate read-modify-write cycles; the PCL write always
 *           wins and the pin is driven permanently LOW.
 *   Set:   PSx for HIGH, PCLx for LOW
 *
 * P33 IOCR layout:
 *   IOCR4  pins 4–7
 * P33 PDR layout:
 *   PDR0   pins 0–7
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * GPIO_Init_LED_P33  — initialises P33.4 – P33.7 as GP outputs, all LOW
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Init_LED_P33(void)
{
    /* IOCR: push-pull GP = 0x10  (IOCR4 covers pins 4–7) */
    P33_IOCR4.B.PC4  = GPIO_PC_PP_GP;
    P33_IOCR4.B.PC5  = GPIO_PC_PP_GP;
    P33_IOCR4.B.PC6  = GPIO_PC_PP_GP;
    P33_IOCR4.B.PC7  = GPIO_PC_PP_GP;

    /* PDR: medium automotive driver  (PDR0 covers pins 0–7) */
    Clear_CPU_WDT_EndInit();
    P33_PDR0.B.PD4   = 0x2U;
    P33_PDR0.B.PL4   = 0x0U;
    P33_PDR0.B.PD5   = 0x2U;
    P33_PDR0.B.PL5   = 0x0U;
    P33_PDR0.B.PD6   = 0x2U;
    P33_PDR0.B.PL6   = 0x0U;
    P33_PDR0.B.PD7   = 0x2U;
    P33_PDR0.B.PL7   = 0x0U;
    Set_CPU_WDT_EndInit();

    /* Drive all LOW (LED off) */
    P33_OMR.B.PCL4   = 0x1U;
    P33_OMR.B.PCL5   = 0x1U;
    P33_OMR.B.PCL6   = 0x1U;
    P33_OMR.B.PCL7   = 0x1U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * Toggle helpers — single atomic 32-bit write to P33_OMR.U.
 *
 * P33_OMR register layout (TC3xx ds1):
 *   bits [15: 0]  PSx  — set   output latch bit x
 *   bits [31:16]  PCLx — clear output latch bit x  (PCLx occupies bit x+16)
 *
 * Writing PSx=1 AND PCLx=1 in the SAME 32-bit bus transaction causes the
 * port hardware to XOR the output latch → toggle.
 *
 * Two sequential .B field writes (first PS, then PCL) are two separate
 * read-modify-write cycles on the volatile register; the second write wins
 * and the pin is always driven LOW — that is the defect this replaces.
 *
 * Bit positions for P33.4 – P33.7:
 *   pin 4:  PS4  = bit  4,  PCL4 = bit 20  → 0x00100010U
 *   pin 5:  PS5  = bit  5,  PCL5 = bit 21  → 0x00200020U
 *   pin 6:  PS6  = bit  6,  PCL6 = bit 22  → 0x00400040U
 *   pin 7:  PS7  = bit  7,  PCL7 = bit 23  → 0x00800080U
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Toggle_LED_P33_4(void)
{
    P33_OMR.U = 0x00100010U;   /* PS4=1, PCL4=1 — single write, HW toggles pin */
}
void GPIO_Toggle_LED_P33_5(void)
{
    P33_OMR.U = 0x00200020U;   /* PS5=1, PCL5=1 */
}
void GPIO_Toggle_LED_P33_6(void)
{
    P33_OMR.U = 0x00400040U;   /* PS6=1, PCL6=1 */
}
void GPIO_Toggle_LED_P33_7(void)
{
    P33_OMR.U = 0x00800080U;   /* PS7=1, PCL7=1 */
}

/*--------------------------------------------------------------------------------------------------------------------
 * Set helpers — PSx for HIGH, PCLx for LOW (same pattern as set_port00_5)
 *------------------------------------------------------------------------------------------------------------------*/
void GPIO_Set_LED_P33_4(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS4  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL4 = 0x1U;
    }
}
void GPIO_Set_LED_P33_5(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS5  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL5 = 0x1U;
    }
}
void GPIO_Set_LED_P33_6(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS6  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL6 = 0x1U;
    }
}
void GPIO_Set_LED_P33_7(GPIO_Level_T Level)
{
    if (Level == GPIO_LEVEL_HIGH)
    {
        P33_OMR.B.PS7  = 0x1U;
    }
    else
    {
        P33_OMR.B.PCL7 = 0x1U;
    }
}
