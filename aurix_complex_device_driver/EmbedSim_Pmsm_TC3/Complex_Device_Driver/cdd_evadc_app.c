/**********************************************************************************************************************
 * \file        cdd_evadc_app.c
 * \brief       Implementation of cdd_evadc_app.h — EVADC channel init and readout.
 *
 * \details     Four EVADC groups follow the same init pattern:
 *              1. Configure arbitration priority (ARBPR)
 *              2. Configure channel control (CHCTR): global class 0, result reg 0
 *              3. Add channel to queue (QINR): auto-refill, external trigger
 *              4. Configure queue trigger (QCTRL): GTM ATOM via ADCTRIG
 *              5. Enable queue trigger (QMR)
 *              6. Configure data reduction (RCR): N samples, service request
 *              7. Configure service request node (SRC): SRPN, CPU1
 *              8. Start converter (ARBCFG.ANONC = 3)
 *
 *              ADCTRIG wiring (set in cdd_gtm_app.c):
 *                  ATOM0_CH4 → ADCTRIG0 → G0/G1/G2  (phase current, XTSEL=0x8)
 *                  ATOM0_CH3 → ADCTRIG3 → G8         (DC-link, XTSEL=0xB)
 *
 * \note        MISRA C:2012: Rules 8.9, 14.4, 15.5, 17.2.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#include "cdd_evadc_app.h"
#include "cdd_sys_utility.h"   /* CddSys_ClearWdtEndInit, CddSys_SetWdtEndInit, CddSys_NopDelay */
#include "cdd_gpio_app.h"      /* CddGpio_ConfigIsrTiming_P14_5, CddGpio_ToggleIsrTiming_P14_5  */
#include "cdd_config.h"
#include "IfxEvadc_reg.h"
#include "IfxSrc_reg.h"
#include "IfxConverter_reg.h"
#include "Bsp.h"

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

#define EVADC_FULL_SCALE            (4095.0f)   /**< 12-bit full-scale value          */
#define ADC_MAX_VOLTAGE             (5.0f)       /**< Maximum ADC input voltage  [V]   */
#define EVADC_XTSEL_ADCTRIG0        (0x8U)       /**< XTSEL for ADCTRIG0 (phase curr)  */
#define EVADC_XTSEL_ADCTRIG3        (0xBU)       /**< XTSEL for ADCTRIG3 (DC-link)     */
#define EVADC_PHASE_SAMPLES         (1U)         /**< DMM data reduction sample count  */
#define EVADC_ARBPR_PRIO            (0x1U)       /**< Arbitration priority             */

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
STATIC void CddEvadc_InitConvctrl(void);
STATIC void CddEvadc_EnableClock(void);
STATIC void CddEvadc_ConfigGlobal(void);
STATIC void CddEvadc_CalibrateAllGroups(void);
STATIC void CddEvadc_ConfigG00PhaseU(void);
STATIC void CddEvadc_ConfigG01PhaseV(void);
STATIC void CddEvadc_ConfigG02PhaseW(void);
STATIC void CddEvadc_ConfigG08DcLink(void);

LOCAL_INLINE void CddEvadc_ReadPhaseU(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr);
LOCAL_INLINE void CddEvadc_ReadPhaseV(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr);
LOCAL_INLINE void CddEvadc_ReadPhaseW(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr);
LOCAL_INLINE void CddEvadc_ReadDcLink(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr);

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

void CddEvadc_Init(void)
{
    CddGpio_ConfigIsrTiming_P14_5();

    CddEvadc_InitConvctrl();
    CddEvadc_EnableClock();
    CddEvadc_ConfigGlobal();
    CddEvadc_CalibrateAllGroups();

    CddEvadc_ConfigG00PhaseU();
    CddEvadc_ConfigG01PhaseV();
    CddEvadc_ConfigG02PhaseW();
    CddEvadc_ConfigG08DcLink();
}

void CddEvadc_ReadSensorMeas(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr)
{
    CddEvadc_ReadPhaseU(MeasPtr);
    CddEvadc_ReadPhaseV(MeasPtr);
    CddEvadc_ReadPhaseW(MeasPtr);
    CddEvadc_ReadDcLink(MeasPtr);
}

/**********************************************************************************************************************
 * ISR Vector Registrations
 * SRPN literals must match cdd_config.h: 90, 91, 92, 95
 * ISR names are vector-table entries — unchanged per MISRA DEV convention.
 *********************************************************************************************************************/

IFX_INTERRUPT(EVADC_G0_Isr, 1, 90);    /* CORE_01_ADC_PHASE_U_SRPN             */
IFX_INTERRUPT(EVADC_G1_Isr, 1, 91);    /* CORE_01_ADC_PHASE_V_SRPN             */
IFX_INTERRUPT(EVADC_G2_Isr, 1, 92);    /* CORE_01_ADC_PHASE_W_SRPN             */
IFX_INTERRUPT(EVADC_G8_Isr, 1, 95);    /* CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN  */

/**********************************************************************************************************************
 * ISR Bodies
 *********************************************************************************************************************/

void EVADC_G0_Isr(void) { /* Phase U result ready — handled in control loop */ }
void EVADC_G1_Isr(void) { /* Phase V result ready — handled in control loop */ }
void EVADC_G2_Isr(void)
{
    CddGpio_ToggleIsrTiming_P14_5();
    /* Phase W result ready — handled in control loop */
    CddGpio_ToggleIsrTiming_P14_5();
}
void EVADC_G8_Isr(void) { /* DC-link result ready — handled in control loop */ }

/**********************************************************************************************************************
 * Private — Hardware Init Helpers
 *********************************************************************************************************************/

STATIC void CddEvadc_InitConvctrl(void)
{
    CddSys_ClearWdtEndInit();
    CONVCTRL_CLC.U = 0x00000000U;                  /* Enable CONVCTRL module        */
    while (CONVCTRL_CLC.B.DISS == 0x1U)
    {
        CddSys_NopDelay(1U, 1U);
    }
    CONVCTRL_CCCTRL.U = 0xB0000000U;               /* Unlock converter control regs */
    CONVCTRL_PHSCFG.U = 0x00008007U;               /* fADC=160MHz, fPHSYNC=20MHz    */
    CONVCTRL_CCCTRL.U = 0x00000000U;               /* Lock converter control regs   */
    CddSys_SetWdtEndInit();
}

STATIC void CddEvadc_EnableClock(void)
{
    CddSys_ClearWdtEndInit();
    EVADC_CLC.B.DISR = 0x0U;
    CddSys_SetWdtEndInit();
    while (EVADC_CLC.B.DISS != 0x0U)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

STATIC void CddEvadc_ConfigGlobal(void)
{
    Ifx_EVADC_GLOBCFG     globcfg;
    Ifx_EVADC_GLOB_ICLASS iclass;
    Ifx_EVADC_G_ANCFG     an_cfg;

    globcfg.U        = EVADC_GLOBCFG.U;
    globcfg.B.CPWC   = 0x1U;
    globcfg.B.SUPLEV = 0x0U;
    globcfg.B.USC    = 0x1U;
    EVADC_GLOBCFG.U  = globcfg.U;

    an_cfg.U           = 0x0U;
    an_cfg.B.RPE       = 0x0U;
    an_cfg.B.DIVA      = 0x3U;    /* 160MHz / (3+1) = 40 MHz */
    an_cfg.B.DPCAL     = 0x0U;
    an_cfg.B.CALSTC    = 0x3U;
    EVADC_G0ANCFG.U    = an_cfg.U;
    EVADC_G1ANCFG.U    = an_cfg.U;
    EVADC_G2ANCFG.U    = an_cfg.U;
    EVADC_G8ANCFG.U    = an_cfg.U;

    iclass.U            = EVADC_GLOBICLASS0.U;
    iclass.B.CMS        = 0x0U;
    iclass.B.AIPS       = 0x0U;
    iclass.B.STCS       = 0xFU;   /* 850 ns sample time (16 clocks at 40 MHz) */
    EVADC_GLOBICLASS0.U = iclass.U;
}

STATIC void CddEvadc_CalibrateAllGroups(void)
{
    EVADC_GLOBCFG.B.SUCAL = 0x1U;
    while (EVADC_G0ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
    while (EVADC_G1ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
    while (EVADC_G2ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
    while (EVADC_G8ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/**********************************************************************************************************************
 * Private — Channel Configuration Helpers
 *********************************************************************************************************************/

STATIC void CddEvadc_ConfigG00PhaseU(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U           = EVADC_G0ARBPR.U;
    arb_pr.B.PRIO0     = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0     = 0x1U;
    arb_pr.B.CSM0      = 0x0U;
    EVADC_G0ARBPR.U    = arb_pr.U;

    ch_ctrl.U          = EVADC_G0CHCTR0.U;
    ch_ctrl.B.ICLSEL   = 0x2U;
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G0CHCTR0.U   = ch_ctrl.U;

    q_qinr.U           = EVADC_G0QINR0.U;
    q_qinr.B.REQCHNR   = 0x0U;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x1U;
    EVADC_G0QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G0QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG0;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G0QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G0QMR0.U;
    q_qmr.B.ENGT       = 0x1U;
    q_qmr.B.ENTR       = 0x1U;
    EVADC_G0QMR0.U     = q_qmr.U;

    rcr.U              = EVADC_G0RCR0.U;
    rcr.B.DRCTR        = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x1U;
    EVADC_G0RCR0.U     = rcr.U;

    EVADC_G0REVNP0.U   = 0x0U;

    src_cfg.U          = SRC_VADC_G0_SR0.U;
    src_cfg.B.SRPN     = CORE_01_ADC_PHASE_U_SRPN;
    src_cfg.B.TOS      = 0x2U;
    SRC_VADC_G0_SR0.U  = src_cfg.U;
    SRC_VADC_G0_SR0.B.SRE = EVADC_ENABLE_PHASE_U_SR;

    EVADC_G0ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G0ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

STATIC void CddEvadc_ConfigG01PhaseV(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U           = EVADC_G1ARBPR.U;
    arb_pr.B.PRIO0     = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0     = 0x1U;
    EVADC_G1ARBPR.U    = arb_pr.U;

    ch_ctrl.U          = EVADC_G1CHCTR0.U;
    ch_ctrl.B.ICLSEL   = 0x2U;
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G1CHCTR0.U   = ch_ctrl.U;

    q_qinr.U           = EVADC_G1QINR0.U;
    q_qinr.B.REQCHNR   = 0x0U;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x1U;
    EVADC_G1QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G1QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG0;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G1QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G1QMR0.U;
    q_qmr.B.ENGT       = 0x1U;
    q_qmr.B.ENTR       = 0x1U;
    EVADC_G1QMR0.U     = q_qmr.U;

    rcr.U              = EVADC_G1RCR0.U;
    rcr.B.DRCTR        = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x1U;
    EVADC_G1RCR0.U     = rcr.U;

    EVADC_G1REVNP0.U   = 0x0U;

    src_cfg.U          = SRC_VADC_G1_SR0.U;
    src_cfg.B.SRPN     = CORE_01_ADC_PHASE_V_SRPN;
    src_cfg.B.TOS      = 0x2U;
    SRC_VADC_G1_SR0.U  = src_cfg.U;
    SRC_VADC_G1_SR0.B.SRE = EVADC_ENABLE_PHASE_V_SR;

    EVADC_G1ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G1ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

STATIC void CddEvadc_ConfigG02PhaseW(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U           = EVADC_G2ARBPR.U;
    arb_pr.B.PRIO0     = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0     = 0x1U;
    EVADC_G2ARBPR.U    = arb_pr.U;

    ch_ctrl.U          = EVADC_G2CHCTR0.U;
    ch_ctrl.B.ICLSEL   = 0x2U;
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G2CHCTR0.U   = ch_ctrl.U;

    q_qinr.U           = EVADC_G2QINR0.U;
    q_qinr.B.REQCHNR   = 0x0U;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x1U;
    EVADC_G2QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G2QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG0;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G2QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G2QMR0.U;
    q_qmr.B.ENGT       = 0x1U;
    q_qmr.B.ENTR       = 0x1U;
    EVADC_G2QMR0.U     = q_qmr.U;

    rcr.U              = EVADC_G2RCR0.U;
    rcr.B.DRCTR        = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x1U;
    EVADC_G2RCR0.U     = rcr.U;

    EVADC_G2REVNP0.U   = 0x0U;

    src_cfg.U          = SRC_VADC_G2_SR0.U;
    src_cfg.B.SRPN     = CORE_01_ADC_PHASE_W_SRPN;
    src_cfg.B.TOS      = 0x2U;
    SRC_VADC_G2_SR0.U  = src_cfg.U;
    SRC_VADC_G2_SR0.B.SRE = EVADC_ENABLE_PHASE_W_SR;

    EVADC_G2ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G2ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

STATIC void CddEvadc_ConfigG08DcLink(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U           = EVADC_G8ARBPR.U;
    arb_pr.B.PRIO0     = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0     = 0x1U;
    EVADC_G8ARBPR.U    = arb_pr.U;

    ch_ctrl.U          = EVADC_G8CHCTR8.U;
    ch_ctrl.B.ICLSEL   = 0x2U;
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G8CHCTR8.U   = ch_ctrl.U;

    q_qinr.U           = EVADC_G8QINR0.U;
    q_qinr.B.REQCHNR   = 0x8U;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x1U;
    EVADC_G8QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G8QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG3;
    q_ctrl.B.GTSEL     = EVADC_XTSEL_ADCTRIG3;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G8QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G8QMR0.U;
    q_qmr.B.ENGT       = 0x1U;
    q_qmr.B.ENTR       = 0x1U;
    EVADC_G8QMR0.U     = q_qmr.U;

    rcr.U              = EVADC_G8RCR0.U;
    rcr.B.DRCTR        = 0x0U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x1U;
    EVADC_G8RCR0.U     = rcr.U;

    EVADC_G8REVNP0.U   = 0x0U;

    src_cfg.U          = SRC_VADC_G8_SR0.U;
    src_cfg.B.SRPN     = CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN;
    src_cfg.B.TOS      = 0x2U;
    SRC_VADC_G8_SR0.U  = src_cfg.U;
    SRC_VADC_G8_SR0.B.SRE = EVADC_ENABLE_DC_LINK_SR;

    EVADC_G8ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G8ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/**********************************************************************************************************************
 * Private — Inline Result Readers
 *********************************************************************************************************************/

LOCAL_INLINE void CddEvadc_ReadPhaseU(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr)
{
    if (EVADC_G0RES0.B.VF == 0x1U)
    {
        MeasPtr->IPhU = ((real32_T)EVADC_G0RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}

LOCAL_INLINE void CddEvadc_ReadPhaseV(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr)
{
    if (EVADC_G1RES0.B.VF == 0x1U)
    {
        MeasPtr->IPhV = ((real32_T)EVADC_G1RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}

LOCAL_INLINE void CddEvadc_ReadPhaseW(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr)
{
    if (EVADC_G2RES0.B.VF == 0x1U)
    {
        MeasPtr->IPhW = ((real32_T)EVADC_G2RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}

LOCAL_INLINE void CddEvadc_ReadDcLink(P2VAR(volatile CddEvadc_Meas_T, AUTOMATIC, CDD_APPL_DATA) MeasPtr)
{
    if (EVADC_G8RES0.B.VF == 0x1U)
    {
        MeasPtr->UDcLink = ((real32_T)EVADC_G8RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}
