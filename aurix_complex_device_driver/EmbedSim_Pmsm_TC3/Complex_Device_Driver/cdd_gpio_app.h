/**********************************************************************************************************************
 * \file        cdd_gpio_app.h
 * \brief       GPIO pad configuration interface for AURIX TC3xx Motor Control
 *              Power Board (AP32541).
 *
 * \details     Owns IOCR + PDR configuration for every port pin driven by the
 *              CDD layer.  Pin groups:
 *
 *              GTM ATOM outputs (TOUTSEL 0x02 alt-function, push-pull, speed 3):
 *                  P00.0   ATOM0_CH0  → CDTM0_DTM4_0 → TOUT9   master scope probe
 *                  P00.2   ATOM0_CH1  → CDTM0_DTM4_1 → TOUT11  IL1  low-side  Phase U
 *                  P00.3   ATOM0_CH2  → CDTM0_DTM4_2 → TOUT12  /IH1 high-side Phase U
 *                  P00.4   ATOM0_CH3  → CDTM0_DTM4_3 → TOUT13  IL2  low-side  Phase V
 *                  P00.5   ATOM0_CH4  → CDTM0_DTM5_0 → TOUT14  /IH2 high-side Phase V
 *                  P00.6   ATOM0_CH5  → CDTM0_DTM5_1 → TOUT15  IL3  low-side  Phase W
 *                  P00.7   ATOM0_CH6  → CDTM0_DTM5_2 → TOUT16  /IH3 high-side Phase W
 *
 *              ISR timing / debug (push-pull, speed 1):
 *                  P14.5   ISR timing probe  (toggle in EVADC_G2_Isr)
 *
 *              Debug LEDs (push-pull GP output, medium driver):
 *                  P33.4 – P33.7   four general-purpose debug LEDs
 *
 *              Note: P02.6/7/8 (GPT12 encoder) are configured exclusively by
 *              IfxGpt12_IncrEnc_init() and are NOT touched here.
 *
 *              Note: QSPI4 pins (P22.0/1/2/3) are configured by
 *              GPIO_Configure_QSPI4_Pins() which MUST be called before
 *              CDD_Qspi_Init().  In bare-metal builds IfxQspi_SpiMaster_initModule()
 *              is NOT used, so the pin mux must be set manually here.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_gpio_app.c
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_GPIO_APP_H_
#define CDD_GPIO_APP_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"


/**********************************************************************************************************************
 * Function Prototypes — GTM ATOM Pin Mux Helpers
 *
 * Each function writes IOCR (output mode = push-pull alt-function) and
 * PDR (pad driver = automotive speed 3) for one port pin.
 * Called individually from Initialize_GTM_Module() immediately after the
 * corresponding GTM_TOUTSELx.B.SELy write so the sequencing is explicit.
 *********************************************************************************************************************/

/** \brief  P00.0 — ATOM0_CH0 → CDTM0_DTM4_0 → TOUT9  master PWM / scope probe  */
extern void GPIO_Configure_GTM_Master_P00_0(void);

/** \brief  P00.2 — ATOM0_CH1 → CDTM0_DTM4_1 → TOUT11  IL1  low-side  Phase U   */
extern void GPIO_Configure_GTM_PhaseU_LS_P00_2(void);

/** \brief  P00.3 — ATOM0_CH2 → CDTM0_DTM4_2 → TOUT12  /IH1 high-side Phase U   */
extern void GPIO_Configure_GTM_PhaseU_HS_P00_3(void);

/** \brief  P00.4 — ATOM0_CH3 → CDTM0_DTM4_3 → TOUT13  IL2  low-side  Phase V   */
extern void GPIO_Configure_GTM_PhaseV_LS_P00_4(void);

/** \brief  P00.5 — ATOM0_CH4 → CDTM0_DTM5_0 → TOUT14  /IH2 high-side Phase V   */
extern void GPIO_Configure_GTM_PhaseV_HS_P00_5(void);

/** \brief  P00.6 — ATOM0_CH5 → CDTM0_DTM5_1 → TOUT15  IL3  low-side  Phase W   */
extern void GPIO_Configure_GTM_PhaseW_LS_P00_6(void);

/** \brief  P00.7 — ATOM0_CH6 → CDTM0_DTM5_2 → TOUT16  /IH3 high-side Phase W   */
extern void GPIO_Configure_GTM_PhaseW_HS_P00_7(void);

/**********************************************************************************************************************
 * Function Prototypes — Debug / ISR Timing
 *********************************************************************************************************************/

/**
 * \brief   Configures P14.5 as a push-pull output for ISR timing measurement.
 * \details Called from Initialize_EVADC_Module() before GTM triggers are armed.
 */
extern void GPIO_Configure_ISR_Timing_P14_5(void);

/**
 * \brief   Toggles P14.5 (used as a scope trigger inside EVADC_G2_Isr).
 */
extern void GPIO_Toggle_ISR_Timing_P14_5(void);

/**********************************************************************************************************************
 * Data Types — Digital Output Level
 *********************************************************************************************************************/

/**
 * \brief   Logic level for GPIO output set functions.
 */
typedef enum
{
    GPIO_LEVEL_LOW  = 0U,   /**< \brief Drive output LOW   */
    GPIO_LEVEL_HIGH = 1U    /**< \brief Drive output HIGH  */
} GPIO_Level_T;

/**********************************************************************************************************************
 * Function Prototypes — QSPI4 Pin Mux  (P22.0 / P22.1 / P22.2 / P22.3)
 *
 * Must be called from Initialize_Inverter() BEFORE CDD_Qspi_Init().
 * In the bare-metal build, IfxQspi_SpiMaster_initModule() is not used,
 * so the QSPI4 alternate-function mux is not set automatically.
 *
 * Pin map (AP32541 / TC38x port alt-function table):
 *   P22.0   QSPI4_MTSR  MOSI  output alt-func 1
 *   P22.1   QSPI4_MRST  MISO  input  no pull
 *   P22.2   QSPI4_SLSO3 CS    output alt-func 2  (IfxQspi4_SLSO3_P22_2_OUT)
 *   P22.3   QSPI4_SCLK  SCLK  output alt-func 1
 *********************************************************************************************************************/

/** \brief  Configures P20.0 (/INH), P33.10 (/SOFF), P33.11 (ENA), P15.2 (/ERR).
 *          Called from Initialize_GPIO_Module() before CDD_Tle9180_Init().  */
extern void GPIO_Configure_GD9180_Pins(void);

/** \brief  Configures P22.0/1/2/3 for QSPI4 alternate function.  */
extern void GPIO_Configure_QSPI4_Pins(void);

/**********************************************************************************************************************
 * Function Prototypes — TLE9180D Gate Driver Control Pins
 *
 * Pin assignment  (AP32541 schematic):
 *   P20.0   /INH   active-LOW inhibit   — LOW = sleep, HIGH = active
 *   P33.11  ENA    active-HIGH enable   — HIGH = outputs enabled
 *   P33.10  /SOFF  active-LOW safe-off  — LOW = safe-off state
 *   P15.2   /ERR   active-LOW error     — input, read by GPIO_Get_ERR_P15_2()
 *********************************************************************************************************************/

/** \brief  Drives P20.0 (/INH) to the requested level.   */
extern void GPIO_Set_INH_P20_0(GPIO_Level_T Level);

/** \brief  Drives P33.11 (ENA) to the requested level.   */
extern void GPIO_Set_ENA_P33_11(GPIO_Level_T Level);

/** \brief  Drives P33.10 (/SOFF) to the requested level. */
extern void GPIO_Set_SOFF_P33_10(GPIO_Level_T Level);

/**
 * \brief   Reads the /ERR input pin P15.2.
 * \return  1 if P15.2 is HIGH (no fault), 0 if LOW (fault active)
 */
extern uint32_T GPIO_Get_ERR_P15_2(void);

/**********************************************************************************************************************
 * Function Prototypes — Debug LEDs  P33.4 – P33.7
 *
 * Four general-purpose push-pull outputs on port 33, used as debug LEDs.
 * All initialised LOW (LED off) by Initialize_GPIO_Module().
 *********************************************************************************************************************/

/** \brief  Initialises P33.4 – P33.7 as push-pull GP outputs, drive LOW.  */
extern void GPIO_Init_LED_P33(void);

/** \brief  Toggles P33.4  */
extern void GPIO_Toggle_LED_P33_4(void);
/** \brief  Toggles P33.5  */
extern void GPIO_Toggle_LED_P33_5(void);
/** \brief  Toggles P33.6  */
extern void GPIO_Toggle_LED_P33_6(void);
/** \brief  Toggles P33.7  */
extern void GPIO_Toggle_LED_P33_7(void);

/** \brief  Sets P33.4 – P33.7 to the requested level.  */
extern void GPIO_Set_LED_P33_4(GPIO_Level_T Level);
extern void GPIO_Set_LED_P33_5(GPIO_Level_T Level);
extern void GPIO_Set_LED_P33_6(GPIO_Level_T Level);
extern void GPIO_Set_LED_P33_7(GPIO_Level_T Level);

#endif /* CDD_GPIO_APP_H_ */
