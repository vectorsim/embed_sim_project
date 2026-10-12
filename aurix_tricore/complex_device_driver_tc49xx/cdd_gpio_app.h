/**********************************************************************************************************************
 * \file      cdd_gpio_app.h
 * \brief     Public interface for the GPIO application module.
 *
 * \details   Bare-metal GPIO / alternate-output pin configuration for TC49x.
 *            Each function configures the pad driver (PADCFGx.DRVCFG) and
 *            routes the eGTM output (EGTM_TOUTSELx) for one ATOM0 PWM
 *            channel of the three-phase inverter bridge:
 *              - P02.0  ATOM0 CH0  Master PWM output.
 *              - P02.1  ATOM0 CH0  ADC Trigger PWM output.
 *              - P02.5  ATOM0 CH5  phase-U high-side output.
 *              - P00.2  ATOM0 CH5N phase-U low-side  output.
 *              - P02.6  ATOM0 CH6  phase-V high-side output.
 *              - P00.4  ATOM0 CH6N phase-V low-side  output.
 *              - P02.7  ATOM0 CH7  phase-W high-side output.
 *              - P00.6  ATOM0 CH7N phase-W low-side  output.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter -> Uppercase)
 *              - Output pointers: PascalCasePtr
 *              - Local variables: lowerPascalCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.0.0
 * \date      2026-10-07
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef COMPLEX_DEVICE_DRIVER_CDD_GPIO_APP_H_
#define COMPLEX_DEVICE_DRIVER_CDD_GPIO_APP_H_

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Configure P02.0 as ATOM0 CH0 Master channel output.
 *
 * \details Configures the P02.0 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 0 dead-time output onto the pin through
 *          EGTM_TOUTSEL0.SEL0 (ToutSel = 0x1).
 *
 * \note    Intended to be called before the per-phase output functions so the
 *          master/sync channel is in place first.
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch0_P02_00_M(void);


/**
 * \brief   Configure P02.1 as ATOM0 CH1 ADC Trigger
 *
 * \details Configures the P02.1 pad driver (MODE = 0x9, DIR = output)
 *          EGTM_TOUTSEL0.SEL1 (ToutSel = 0x12).
 *
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch1_P02_01_AT(void);

/**
 * \brief   Configure P02.5 as phase U high-side ATOM output.
 *
 * \details Configures the P02.5 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 5 output onto the pin through
 *          EGTM_TOUTSEL0.SEL5 (ToutSel = 0x12).
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch5_P02_05_U_HS(void);

/**
 * \brief   Configure P00.2 as phase U low-side ATOM output.
 *
 * \details Configures the P00.2 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 5 inverted output onto the pin through
 *          EGTM_TOUTSEL1.SEL5 (ToutSel = 0x01).
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch5_P00_02_U_LS(void);

/**
 * \brief   Configure P02.6 as phase V high-side ATOM output.
 *
 * \details Configures the P02.6 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 6 output onto the pin through
 *          EGTM_TOUTSEL1.SEL0 (ToutSel = 0x12).
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch6_P02_06_V_HS(void);

/**
 * \brief   Configure P00.4 as phase V low-side ATOM output.
 *
 * \details Configures the P00.4 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 6 inverted output onto the pin through
 *          EGTM_TOUTSEL2.SEL1 (ToutSel = 0x1).
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch6_P00_04_V_LS(void);

/**
 * \brief   Configure P02.7 as phase W high-side ATOM output.
 *
 * \details Configures the P02.7 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 7 output onto the pin through
 *          EGTM_TOUTSEL1.SEL1 (ToutSel = 0x12).
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch7_P02_07_W_HS(void);

/**
 * \brief   Configure P00.6 as phase W low-side ATOM output.
 *
 * \details Configures the P00.6 pad driver (MODE = 0x9, DIR = output) and
 *          routes the ATOM0 channel 7 inverted output onto the pin through
 *          EGTM_TOUTSEL2.SEL3 (ToutSel = 0x1).
 *
 * \note    Slice 3 sits at bits [20:16], not [19:15], because bit 15 is
 *          reserved in TOUTSELn.
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtmAtom0_Ch7_P00_06_W_LS(void);

/**
 * \brief   Configure P03.10 (BOARD_USER_LED_2 / green) as a push-pull GPIO output.
 *
 * \details Sets P03_PADCFG10_DRVCFG.MODE = 0x0 (default pad driver),
 *          .OD = 0x0 (push-pull) and .DIR = 0x1 (output). PCSRSEL is left
 *          untouched so the pin stays under normal TriCore control (GPIO).
 *
 * \note    The LED on this board is low-active.
 *
 * \return  void
 */
extern void CddGpio_Init_P03_10_Led_HeartBeat(void);


/**
 * \brief   Configure P00.01 as GPIO Fast ShutOff input
 *
 * \return  void
 */
extern void CddGpio_ConfigEgtm_Tim0_Ch4_P00_01_FS(void);

/**
 * \brief   Configure P00.3 (Arduino ADC2 / X301.3) as analog input for
 *          TMADC2 channel 9, used as Phase-W sense.
 *
 * \details P00.3 is a pad-class-SLOW pin overlaid with GPIO.  The pad is
 *          set to tristate input (high-Z) with no pull device so the analog
 *          signal is presented unloaded to the TMADC2.Ch9 sampler.
 *
 * \note    This is the ONLY phase-sense pin that requires pad configuration.
 *          AN1 (Phase U) and AN16 (Phase V) are dedicated class-D analog
 *          inputs and have no GPIO overlay.
 *
 * \return  void
 */
extern void CddGpio_ConfigAdc_Tmadc2_Ch9_P00_03_PhaseW(void);

#endif /* COMPLEX_DEVICE_DRIVER_CDD_GPIO_APP_H_ */
