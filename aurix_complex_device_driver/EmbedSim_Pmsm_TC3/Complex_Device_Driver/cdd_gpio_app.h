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
 *                  P14.5   ISR timing probe (toggle in EVADC_G2_Isr)
 *
 *              Debug LEDs (push-pull GP output, medium driver):
 *                  P33.4 – P33.7   four general-purpose debug LEDs
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

#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Data Types
 *********************************************************************************************************************/

/**
 * \brief   Logic level for GPIO output set functions.
 */
typedef enum
{
    CDDGPIO_LEVEL_LOW  = 0U,   /**< Drive output LOW   [dimensionless] */
    CDDGPIO_LEVEL_HIGH = 1U    /**< Drive output HIGH  [dimensionless] */
} CddGpio_Level_T;

/**********************************************************************************************************************
 * Function Prototypes — GTM ATOM Pin Mux Helpers
 *
 * Each function writes IOCR (output mode = push-pull alt-function) and
 * PDR (pad driver = automotive speed 3) for one port pin.
 * Called from CddGtm_Init() immediately after the corresponding GTM_TOUTSELx write.
 *********************************************************************************************************************/

/** \brief  P00.0 — ATOM0_CH0 → CDTM0_DTM4_0 → TOUT9  master PWM / scope probe  */
extern void CddGpio_ConfigGtmMaster_P00_0(void);

/** \brief  P00.2 — ATOM0_CH1 → CDTM0_DTM4_1 → TOUT11  IL1  low-side  Phase U   */
extern void CddGpio_ConfigGtmPhaseULs_P00_2(void);

/** \brief  P00.3 — ATOM0_CH2 → CDTM0_DTM4_2 → TOUT12  /IH1 high-side Phase U   */
extern void CddGpio_ConfigGtmPhaseUHs_P00_3(void);

/** \brief  P00.4 — ATOM0_CH3 → CDTM0_DTM4_3 → TOUT13  IL2  low-side  Phase V   */
extern void CddGpio_ConfigGtmPhaseVLs_P00_4(void);

/** \brief  P00.5 — ATOM0_CH4 → CDTM0_DTM5_0 → TOUT14  /IH2 high-side Phase V   */
extern void CddGpio_ConfigGtmPhaseVHs_P00_5(void);

/** \brief  P00.6 — ATOM0_CH5 → CDTM0_DTM5_1 → TOUT15  IL3  low-side  Phase W   */
extern void CddGpio_ConfigGtmPhaseWLs_P00_6(void);

/** \brief  P00.7 — ATOM0_CH6 → CDTM0_DTM5_2 → TOUT16  /IH3 high-side Phase W   */
extern void CddGpio_ConfigGtmPhaseWHs_P00_7(void);

/**********************************************************************************************************************
 * Function Prototypes — Debug / ISR Timing
 *********************************************************************************************************************/

/** \brief  Configures P14.5 as push-pull output for ISR timing measurement.
 *          Called from CddEvadc_Init() before GTM triggers are armed. */
extern void CddGpio_ConfigIsrTiming_P14_5(void);

/** \brief  Toggles P14.5 (used as a scope trigger inside EVADC_G2_Isr). */
extern void CddGpio_ToggleIsrTiming_P14_5(void);

/**********************************************************************************************************************
 * Function Prototypes — QSPI4 Pin Mux  (P22.0 / P22.1 / P22.2 / P22.3)
 *
 * Must be called from CddApp_InitInverter() BEFORE CddQspi4_Init().
 * Pin map: P22.0 MOSI alt-func 1 | P22.1 MISO input | P22.2 CS alt-func 2 | P22.3 SCLK alt-func 1
 *********************************************************************************************************************/

/** \brief  Configures P20.0 (/INH), P33.10 (/SOFF), P33.11 (ENA), P15.2 (/ERR).
 *          Call before CddTle9180_Init(). */
extern void CddGpio_ConfigGd9180Pins(void);

/** \brief  Configures P22.0/1/2/3 for QSPI4 alternate function.  */
extern void CddGpio_ConfigQspi4Pins(void);

/**********************************************************************************************************************
 * Function Prototypes — TLE9180D Gate Driver Control Pins
 *
 *   P20.0   /INH   active-LOW inhibit   — LOW = sleep, HIGH = active
 *   P33.11  ENA    active-HIGH enable   — HIGH = outputs enabled
 *   P33.10  /SOFF  active-LOW safe-off  — LOW = safe-off state
 *   P15.2   /ERR   active-LOW error     — input
 *********************************************************************************************************************/

/** \brief  Drives P20.0 (/INH) to the requested level.   */
extern void CddGpio_SetInh_P20_0(CddGpio_Level_T Level);

/** \brief  Drives P33.11 (ENA) to the requested level.   */
extern void CddGpio_SetEna_P33_11(CddGpio_Level_T Level);

/** \brief  Drives P33.10 (/SOFF) to the requested level. */
extern void CddGpio_SetSoff_P33_10(CddGpio_Level_T Level);

/**
 * \brief   Reads the /ERR input pin P15.2.
 * \return  1U if P15.2 is HIGH (no fault), 0U if LOW (fault active). [dimensionless]
 */
extern uint32_T CddGpio_GetErr_P15_2(void);

/**********************************************************************************************************************
 * Function Prototypes — Debug LEDs  P33.4 – P33.7
 *********************************************************************************************************************/

/** \brief  Initialises P33.4 – P33.7 as push-pull GP outputs, all driven LOW. */
extern void CddGpio_InitLed_P33(void);

extern void CddGpio_ToggleLed_P33_4(void);
extern void CddGpio_ToggleLed_P33_5(void);
extern void CddGpio_ToggleLed_P33_6(void);
extern void CddGpio_ToggleLed_P33_7(void);

extern void CddGpio_SetLed_P33_4(CddGpio_Level_T Level);
extern void CddGpio_SetLed_P33_5(CddGpio_Level_T Level);
extern void CddGpio_SetLed_P33_6(CddGpio_Level_T Level);
extern void CddGpio_SetLed_P33_7(CddGpio_Level_T Level);

#endif /* CDD_GPIO_APP_H_ */
