/**********************************************************************************************************************
 * \file      cdd_gpio_app.c
 * \brief     Bare-metal GPIO / alternate-output pin configuration for TC49x.
 *
 * \details   Configures the pad driver (PADCFGx.DRVCFG) and the eGTM output
 *            routing (EGTM_TOUTSELx) for the ATOM0 PWM channels used by the
 *            three-phase inverter bridge:
 *              - P02.0  ATOM0 CH0  master PWM output.
 *              - P02.5  ATOM0 CH5  phase-U high-side output.
 *              - P00.2  ATOM0 CH5N phase-U low-side  output.
 *              - P02.6  ATOM0 CH6  phase-V high-side output.
 *              - P00.4  ATOM0 CH6N phase-V low-side  output.
 *              - P02.7  ATOM0 CH7  phase-W high-side output.
 *              - P00.6  ATOM0 CH7N phase-W low-side  output.
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
 * \date      2026-10-07
 * \author    Paul Abraham
 *            EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/


/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_gpio_app.h"
#include "IfxPort_reg.h"
#include "IfxEGtm_reg.h"


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/
/* None. */

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

void CddGpio_ConfigEgtmAtom0_Ch0_P02_00_M(void)
{
    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    padCfg.U  = P02_PADCFG0_DRVCFG.U;
    toutSel.U = EGTM_TOUTSEL0.U;

    padCfg.B.MODE  = 0x9u;   /* pad driver mode   */
    padCfg.B.DIR   = 0x1u;   /* direction: output */
    toutSel.B.SEL0 = 0x1U;   /* output selection  */

    P02_PADCFG0_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL0.U      = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch1_P02_01_AT(void)
{
    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    padCfg.U  = P02_PADCFG1_DRVCFG.U;
    toutSel.U = EGTM_TOUTSEL0.U;

    padCfg.B.MODE  = 0x9u;   /* pad driver mode   */
    padCfg.B.DIR   = 0x1u;   /* direction: output */
    toutSel.B.SEL1 = 0x12U;  /* output selection  */

    P02_PADCFG1_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL0.U      = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch5_P02_05_U_HS(void)
{
    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    toutSel.U = EGTM_TOUTSEL0.U;
    padCfg.U  = P02_PADCFG5_DRVCFG.U;

    padCfg.B.MODE  = 0x9U;   /* pad driver mode   */
    padCfg.B.DIR   = 0x1U;   /* direction: output */
    toutSel.B.SEL5 = 0x12U;  /* output selection  */

    P02_PADCFG5_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL0.U = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch5_P00_02_U_LS(void)
{

    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    padCfg.U = P00_PADCFG2_DRVCFG.U;
    toutSel.U = EGTM_TOUTSEL1.U;

    padCfg.B.MODE  = 0x09U;   /* pad driver mode   */
    padCfg.B.DIR   = 0x01U;   /* direction: output */
    toutSel.B.SEL5 = 0x01U;   /* tout egtm          */

    P00_PADCFG2_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL1.U = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch6_P02_06_V_HS(void)
{

    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    toutSel.U = EGTM_TOUTSEL1.U;
    padCfg.U = P02_PADCFG6_DRVCFG.U;

    padCfg.B.MODE  = 0x9U;   /* pad driver mode    */
    padCfg.B.DIR   = 0x1U;   /* direction: output  */
    toutSel.B.SEL0 = 0x12U;  /* tout egtm          */

    P02_PADCFG6_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL1.U      = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch6_P00_04_V_LS(void)
{

    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    padCfg.U = P00_PADCFG4_DRVCFG.U;
    toutSel.U = EGTM_TOUTSEL2.U;

    padCfg.B.MODE  = 0x09U;   /* pad driver mode    */
    padCfg.B.DIR   = 0x1U;    /* direction: output  */
    toutSel.B.SEL1 = 0x1U;    /* tout egtm          */

    P00_PADCFG4_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL2.U = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch7_P02_07_W_HS(void)
{

    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    padCfg.U  = P02_PADCFG7_DRVCFG.U;
    toutSel.U = EGTM_TOUTSEL1.U;

    padCfg.B.MODE  = 0x09U;   /* pad driver mode   */
    padCfg.B.DIR   = 0x1U;    /* direction: output */
    toutSel.B.SEL1 = 0x12U;   /* tout egtm          */

    P02_PADCFG7_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL1.U = toutSel.U;
}

void CddGpio_ConfigEgtmAtom0_Ch7_P00_06_W_LS(void)
{

    Ifx_P_PADCFG_DRVCFG    padCfg;
    Ifx_EGTM_TOUTSEL       toutSel;

    padCfg.U = P00_PADCFG6_DRVCFG.U;
    toutSel.U =   EGTM_TOUTSEL2.U ;

    padCfg.B.MODE  = 0x09U;   /* pad driver mode    */
    padCfg.B.DIR   = 0x01U;   /* direction: output  */
    toutSel.B.SEL3 = 0x1U;     /* tout egtm          */

    P00_PADCFG6_DRVCFG.U = padCfg.U;
    EGTM_TOUTSEL2.U = toutSel.U;
}

void CddGpio_Init_P03_10_Led_HeartBeat(void)
{
    Ifx_P_PADCFG_DRVCFG    padCfg;

    padCfg.U = P03_PADCFG10_DRVCFG.U;

    padCfg.B.MODE = 0x0U;   /* pad driver mode   */
    padCfg.B.OD   = 0x0U;   /* push-pull         */
    padCfg.B.DIR  = 0x1U;   /* direction: output */

    P03_PADCFG10_DRVCFG.U = padCfg.U;
}

void CddGpio_Toggle_P03_10_Led_HeartBeat(void)
{
    Ifx_P_OMR  omr;

    omr.U = P03_OMR.U;
    omr.B.PCL10 = 0x1U;
    omr.B.PS10 = 0x1U;
   // P03_OMR.U = (1U << 10U) | (1U << (16U + 10U));

    P03_OMR.U = omr.U;
}

void CddGpio_ConfigEgtm_Tim0_Ch4_P00_01_FS(void)
{
    Ifx_P_PADCFG_DRVCFG padCfg;

    /* 1. Route P00.1 -> TIM_IN4 : TIMINSEL0.SEL4 = 11 (0xB) */
    EGTM_TIMINSEL0.B.CH4SEL = 0xBU;

    /* 2. Configure P00.1 pad: input with pull-down */
    padCfg.U      = P00_PADCFG1_DRVCFG.U;
    padCfg.B.MODE = 0x2U;    /* input with pull-down */
    padCfg.B.OD   = 0x0U;    /* push-pull (n/a for input) */
    padCfg.B.DIR  = 0x0U;    /* direction: input */
    P00_PADCFG1_DRVCFG.U = padCfg.U;
}

void CddGpio_ConfigAdc_Tmadc2_Ch9_P00_03_PhaseW(void)
{
    Ifx_P_PADCFG_DRVCFG padCfg;

    padCfg.U      = P00_PADCFG3_DRVCFG.U;
    padCfg.B.MODE = 0x0U;    /* tristate input -> high-Z for analog */
    padCfg.B.OD   = 0x0U;    /* push-pull (n/a for input)           */
    padCfg.B.DIR  = 0x0U;    /* direction: input                    */
    padCfg.B.PD   = 0x0U;    /* driver strength (n/a for input)     */

    P00_PADCFG3_DRVCFG.U = padCfg.U;
}




