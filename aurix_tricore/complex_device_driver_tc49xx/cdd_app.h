/**********************************************************************************************************************
 * \file      cdd_app.h
 * \brief     Public interface for the CddSys low-level system utility module.
 *
 * \details   Declares the CddSys function API used across the CDD layer for
 *            TriCore-specific primitives:
 *              - Atomic bit-field load-modify-store (LDMST instruction).
 *              - Binary semaphore via compare-and-swap (CMPSWAP.W instruction).
 *              - Single and multi-instruction NOP delays.
 *              - Interrupt-enable intrinsic.
 *
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
 * \version   1.0.1
 * \date      2026-10-12
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
#define CTRL_ISR_PRIORITY         (20U)    /* Inverter & Control Loop */
#define CTRL_ISR_TOS              (0U)     /* CPU 0                   */
#define TIM0CH4_ISR_PRIORITY      (25U)    /* Fast ShutOff            */
#define TIM0CH4_ISR_TOS           (0U)     /* CPU 0                   */
#define STM0_CMP0_IR_SRPN         (30U)

/* ADC phase-valid bit positions inside CddInverterCtrl_T.AdcPhaseValid.
 * The value is the shift amount (bit index), not a mask. */
#define CDD_PHASE_U_VALID         (0x0U)   /* bit 0 – Phase U fresh */
#define CDD_PHASE_V_VALID         (0x1U)   /* bit 1 – Phase V fresh */
#define CDD_PHASE_W_VALID         (0x2U)   /* bit 2 – Phase W fresh */
#define CDD_PHASE_VDC_VALID       (0x3U)   /* bit 3 – Vdc fresh     */
#define CDD_PHASE_VALID_MASK      (0xFU)   /* convenience mask for "all 4 fresh" */

/*********************************************************************************************************************/
/*---------------------------------------------PORT / PIN ASSIGNMENT MAP---------------------------------------------*/
/*********************************************************************************************************************/
/*
 * Target board : KIT_A3G_TC4D7_LITE (AURIX TC4D7 COM)
 *
 *  PORT  PIN   DIRECTION   SIGNAL                              CONFIGURED BY
 *  ----  ----  ---------   ---------------------------------   ---------------------------------------------
 *
 *  -- eGTM PWM outputs (three-phase inverter bridge) ------------------------------
 *  P02   0     OUT         ATOM0 CH0   master timing reference   CddGpio_ConfigEgtmAtom0_Ch0_P02_00_M
 *  P02   1     OUT         ATOM0 CH1   ADC trigger (EGTM_ADC_OUT0.SEL0)
 *                                                                CddGpio_ConfigEgtmAtom0_Ch1_P02_01_AT
 *  P02   5     OUT         ATOM0 CH5   Phase U high-side
 *                                                                CddGpio_ConfigEgtmAtom0_Ch5_P02_05_U_HS
 *  P00   2     OUT         ATOM0 CH5N  Phase U low-side
 *                                                                CddGpio_ConfigEgtmAtom0_Ch5_P00_02_U_LS
 *  P02   6     OUT         ATOM0 CH6   Phase V high-side
 *                                                                CddGpio_ConfigEgtmAtom0_Ch6_P02_06_V_HS
 *  P00   4     OUT         ATOM0 CH6N  Phase V low-side
 *                                                                CddGpio_ConfigEgtmAtom0_Ch6_P00_04_V_LS
 *  P02   7     OUT         ATOM0 CH7   Phase W high-side
 *                                                                CddGpio_ConfigEgtmAtom0_Ch7_P02_07_W_HS
 *  P00   6     OUT         ATOM0 CH7N  Phase W low-side
 *                                                                CddGpio_ConfigEgtmAtom0_Ch7_P00_06_W_LS
 *
 *  -- eGTM fast shut-off input ----------------------------------------------------
 *  P00   1     IN          TIM0 CH4    fast-shut-off input       CddGpio_ConfigEgtm_Tim0_Ch4_P00_01_FS
 *                          (TIM_IN4, pull-down)
 *
 *  -- Analog inputs (TMADC) ------------------------------------------------------
 *  AN1         IN (analog) TMADC0.Ch1  Phase U sense   dedicated class-D, X2.26
 *  AN16        IN (analog) TMADC1.Ch4  Phase V sense   dedicated class-D, Shield2Go #1 pin 2
 *  P00   3     IN (analog) TMADC2.Ch9  Phase W sense   SLOW pad, Arduino A2 / X301.3
 *  AN2         IN (analog) TMADC0.Ch2  Vdc sense       dedicated class-D, X2.25
 *
 *  NOTE: TC4D7 has no DSADC block (datasheet: DS Converters = 0). Vdc uses TMADC0.Ch2.
 *
 *  -- Status / debug --------------------------------------------------------------
 *  P03   10    OUT         User LED 2 (green, low-active)        CddGpio_Init_P03_10_Led_HeartBeat
 *
 *  -- On-board peripherals NOT used by this CDD layer -----------------------------
 *  P03   11    IN          User button (low-active)              – not configured
 *  P03   5     OUT         CAN transceiver STB                   – not configured
 *  P01   3     OUT         CAN01_TXD                             – not configured
 *  P01   4     IN          CAN01_RXDC                            – not configured
 *  P13   1/2   bidir       I2C0 SCL / SDA (EEPROM 24AA02E48)     – not configured
 *  P21   2/3   bidir       Ethernet MDC / MDIO                   – not configured
 *  P16   x     multi       Ethernet RMII signals                 – not configured
 *
 *  -- Reserved pins to AVOID ------------------------------------------------------
 *  P00   1     already used as TIM0_CH4 fast-shut-off input (see above).
 *              Do NOT reuse as Arduino ADC0.
 *  P00   2/4/6 already used as Phase U/V/W low-side gate outputs.
 *              Do NOT reuse as Arduino ADC1/3/5.
 *  P02   0/1   already used as eGTM master / ADC trigger outputs.
 *              Do NOT reuse.
 *  P02   5/6/7 already used as Phase U/V/W high-side gate outputs.
 *              Do NOT reuse.
 *  AN0         on-board 250 kΩ potentiometer (R32, jumper R35).
 *              Free for a spare analog input if the pot is left in place.
 */
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Central runtime state of the inverter control application.
 *
 * \details Single global instance (Inv_G) shared between the control-loop ISR
 *          (ControlLoop_ISR), the fast shut-off ISR (TIM0_CH4_ISR), and the
 *          application layer. Written by the application, read/written by
 *          ControlLoop_ISR, and partially written by TIM0_CH4_ISR.
 *
 * \note    All members are accessed from ISR context. Updates that must be
 *          seen by the control loop as a coherent set (for example the three
 *          duty cycles together with a FrequencyChangeRequest) should be
 *          staged with interrupts disabled, or via CddSys_Ldmst.
 */
typedef struct
{
    uint32_T   PeriodTicks;             /**< PWM period in eGTM CMU_CLK0 ticks.                */

    uint32_T   DeadTimeTicks;           /**< Dead time for each complementary phase, in ticks. */

    real32_T   CmuClk0Frequency;        /**< eGTM cluster-0 CMU clock 0 frequency in Hz.       */

    real32_T   DutyCyclePhaseU;         /**< Phase U duty cycle [-1.0, +1.0].                  */

    real32_T   DutyCyclePhaseV;         /**< Phase V duty cycle [-1.0, +1.0].                  */

    real32_T   DutyCyclePhaseW;         /**< Phase W duty cycle [-1.0, +1.0].                  */

    real32_T   DutyCycleAdcTr;          /**< ADC trigger duty cycle [0, +1.0].                 */

    uint64_T   Counter;                 /**< Free-running PWM cycle counter.                   */

    uint32_T   ShutOffState;            /**< Latched shut-off state (0 = running, 1 = shut).   */

    uint32_T   ShutOffRstRequest;       /**< Application request to leave shut-off state.      */

    uint32_T   FrequencyMode;           /**< 1 = 10 kHz, 2 = 20 kHz, 3 = 30 kHz.               */

    uint32_T   FrequencyChangeRequest;  /**< Mid-run period reprogram request.                 */

    uint32_T   AdcPhaseU;               /**< TMADC0.Ch1  (AN1)  raw 12-bit – Phase U.          */
    uint32_T   AdcPhaseV;               /**< TMADC1.Ch4  (AN16) raw 12-bit – Phase V.          */
    uint32_T   AdcPhaseW;               /**< TMADC2.Ch9  (P00.3) raw 12-bit – Phase W.         */
    uint32_T   AdcVdc;
    uint32_T   AdcPhaseValid;           /**< Bitmask: bit0=U, bit1=V, bit2=W, bit3=Vdc(1 = fresh).      */
} CddInverterCtrl_T;


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
extern CddInverterCtrl_T  Inv_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Initialises the CDD layer.
 * \return  void
 */
extern void Cdd_Init(void);

/**
 * \brief   Starts the CDD layer (issues the eGTM host trigger).
 * \return  void
 */
extern void Cdd_Start(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_APP_H_ */
