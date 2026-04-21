/**********************************************************************************************************************
 * \file        cdd_evadc_app.c
 * \brief       Implementation of cdd_evadc_app.h
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
 *              ADCTRIG wiring (set in cdd_gtm_app):
 *                  ATOM0_CH4 -> ADCTRIG0 -> G0/G1/G2  (phase current, XTSEL=0x8)
 *                  ATOM0_CH3 -> ADCTRIG3 -> G8         (DC-link,       XTSEL=0xB)
 *
 *              Resolver groups G3/G11 are not present on the DB42S02 motorkit.
 *              Position/speed uses GPT12 incremental encoder via cdd_gpt12_app.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_evadc_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "cdd_gpio_app.h"
#include "IfxEvadc_reg.h"
#include "IfxSrc_reg.h"
#include "IfxConverter_reg.h"
#include "Bsp.h"                   /* IFX_INTERRUPT macro (same as cdd_stm_app.c) */

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  EVADC 12-bit full-scale value for normalisation  */
#define EVADC_FULL_SCALE            (4095.0f)

/** \brief  Maximum ADC input voltage [V]  */
#define ADC_MAX_VOLTAGE             (5.0f)

/** \brief  XTSEL for ADCTRIG0 (phase current, ATOM0_CH4)  */
#define EVADC_XTSEL_ADCTRIG0        (0x8U)

/** \brief  XTSEL for ADCTRIG3 (DC-link, ATOM0_CH3)  */
#define EVADC_XTSEL_ADCTRIG3        (0xBU)

/** \brief  Number of samples for current phase DMM data reduction  */
#define EVADC_PHASE_SAMPLES         (1U)

/** \brief  Arbitration priority — all queues use priority 1  */
#define EVADC_ARBPR_PRIO            (0x1U)

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
static void Init_CONVCTRL(void);
static void Enable_EVADC_Clock(void);
static void Config_EVADC_Global(void);
static void Calibrate_All_Groups(void);

static void Config_G00_C00_An00_Phase_U(void);
static void Config_G01_C00_An08_Phase_V(void);
static void Config_G02_C00_An16_Phase_W(void);
static void Config_G08_C08_An40_DC_Link(void);

static inline void Read_Phase_U  (volatile EVADC_Meas_T * const Meas_Ptr);
static inline void Read_Phase_V  (volatile EVADC_Meas_T * const Meas_Ptr);
static inline void Read_Phase_W  (volatile EVADC_Meas_T * const Meas_Ptr);
static inline void Read_DC_Link  (volatile EVADC_Meas_T * const Meas_Ptr);

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

void Initialize_EVADC_Module(void)
{
    /* EVADC G2 Phase W ISR timing probe */
    GPIO_Configure_ISR_Timing_P14_5();

    Init_CONVCTRL();
    Enable_EVADC_Clock();
    Config_EVADC_Global();
    Calibrate_All_Groups();

    Config_G00_C00_An00_Phase_U();
    Config_G01_C00_An08_Phase_V();
    Config_G02_C00_An16_Phase_W();
    Config_G08_C08_An40_DC_Link();
}

void Read_EVADC_Sensor_Measurement(volatile EVADC_Meas_T * const Meas_Ptr)
{
    Read_Phase_U(Meas_Ptr);
    Read_Phase_V(Meas_Ptr);
    Read_Phase_W(Meas_Ptr);
    Read_DC_Link(Meas_Ptr);
}

/**********************************************************************************************************************
 * ISR Vector Registrations  (IFX_INTERRUPT requires literal integer SRPN — TASKING constraint)
 *
 * SRPN literals must match the #define values in cdd_config.h:
 *   CORE_01_ADC_PHASE_U_SRPN            = 90
 *   CORE_01_ADC_PHASE_V_SRPN            = 91
 *   CORE_01_ADC_PHASE_W_SRPN            = 92
 *   CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN = 95
 *********************************************************************************************************************/

IFX_INTERRUPT(EVADC_G0_Isr, 1, 90);    /* CORE_01_ADC_PHASE_U_SRPN            */
IFX_INTERRUPT(EVADC_G1_Isr, 1, 91);    /* CORE_01_ADC_PHASE_V_SRPN            */
IFX_INTERRUPT(EVADC_G2_Isr, 1, 92);    /* CORE_01_ADC_PHASE_W_SRPN            */
IFX_INTERRUPT(EVADC_G8_Isr, 1, 95);    /* CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN */

/**********************************************************************************************************************
 * ISR Bodies
 *********************************************************************************************************************/

void EVADC_G0_Isr(void) { /* Phase U result ready — handled in control loop */ }
void EVADC_G1_Isr(void) { /* Phase V result ready — handled in control loop */ }
void EVADC_G2_Isr(void)
{
    GPIO_Toggle_ISR_Timing_P14_5();
    /* Phase W result ready — handled in control loop */
    GPIO_Toggle_ISR_Timing_P14_5();
}
void EVADC_G8_Isr(void) { /* DC-link result ready — handled in control loop */ }

/**********************************************************************************************************************
 * Private — Hardware Init Helpers
 *********************************************************************************************************************/

static void Init_CONVCTRL(void)
{
    Clear_CPU_WDT_EndInit();
    CONVCTRL_CLC.U = 0x00000000U;                  /* Enable CONVCTRL module        */
    while (CONVCTRL_CLC.B.DISS == 0x1U)
    {
        Nop_Delay(1U, 1U);
    }
    CONVCTRL_CCCTRL.U = 0xB0000000U;               /* Unlock converter control regs */
    CONVCTRL_PHSCFG.U = 0x00008007U;               /* fADC=160MHz, fPHSYNC=20MHz    */
    CONVCTRL_CCCTRL.U = 0x00000000U;               /* Lock converter control regs   */
    Set_CPU_WDT_EndInit();
}

static void Enable_EVADC_Clock(void)
{
    Clear_CPU_WDT_EndInit();
    EVADC_CLC.B.DISR = 0x0U;
    Set_CPU_WDT_EndInit();
    while (EVADC_CLC.B.DISS != 0x0U)
    {
        Nop_Delay(0x1U, 0x1U);
    }
}

static void Config_EVADC_Global(void)
{
    Ifx_EVADC_GLOBCFG     globcfg;
    Ifx_EVADC_GLOB_ICLASS iclass;
    Ifx_EVADC_G_ANCFG     an_cfg;

    /* Global config: unsynchronised conversion mode */
    globcfg.U          = EVADC_GLOBCFG.U;
    globcfg.B.CPWC     = 0x1U;
    globcfg.B.SUPLEV   = 0x0U;
    globcfg.B.USC      = 0x1U;
    EVADC_GLOBCFG.U    = globcfg.U;

    /* Analog config: fADC = 40 MHz, no post-calibration */
    an_cfg.U           = 0x0U;
    an_cfg.B.RPE       = 0x0U;
    an_cfg.B.DIVA      = 0x3U;    /* 160MHz / (3+1) = 40 MHz */
    an_cfg.B.DPCAL     = 0x0U;
    an_cfg.B.CALSTC    = 0x3U;
    EVADC_G0ANCFG.U    = an_cfg.U;
    EVADC_G1ANCFG.U    = an_cfg.U;
    EVADC_G2ANCFG.U    = an_cfg.U;
    EVADC_G8ANCFG.U    = an_cfg.U;

    /* Global input class 0: standard conversion, 850 ns sample time (16 clocks at 40 MHz) */
    iclass.U           = EVADC_GLOBICLASS0.U;
    iclass.B.CMS       = 0x0U;
    iclass.B.AIPS      = 0x0U;
    iclass.B.STCS      = 0xFU;
    EVADC_GLOBICLASS0.U = iclass.U;
}

static void Calibrate_All_Groups(void)
{
    EVADC_GLOBCFG.B.SUCAL = 0x1U;   /* Start simultaneous calibration */
    while (EVADC_G0ARBCFG.B.CAL  == 0x1U) { Nop_Delay(0x1U, 0x1U); }
    while (EVADC_G1ARBCFG.B.CAL  == 0x1U) { Nop_Delay(0x1U, 0x1U); }
    while (EVADC_G2ARBCFG.B.CAL  == 0x1U) { Nop_Delay(0x1U, 0x1U); }
    while (EVADC_G8ARBCFG.B.CAL  == 0x1U) { Nop_Delay(0x1U, 0x1U); }
}

/**********************************************************************************************************************
 * Private — Channel Configuration Helpers
 *********************************************************************************************************************/

static void Config_G00_C00_An00_Phase_U(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U = EVADC_G0ARBPR.U;
    arb_pr.B.PRIO0 = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0 = 0x1U;
    arb_pr.B.CSM0  = 0x0U;
    EVADC_G0ARBPR.U = arb_pr.U;

    ch_ctrl.U = EVADC_G0CHCTR0.U;
    ch_ctrl.B.ICLSEL = 0x2U;
    ch_ctrl.B.RESREG = 0x0U;
    ch_ctrl.B.RESTGT = 0x0U;
    EVADC_G0CHCTR0.U = ch_ctrl.U;

    q_qinr.U = EVADC_G0QINR0.U;
    q_qinr.B.REQCHNR = 0x0U;
    q_qinr.B.RF      = 0x1U;
    q_qinr.B.EXTR    = 0x1U;
    EVADC_G0QINR0.U = q_qinr.U;

    q_ctrl.U = EVADC_G0QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG0;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G0QCTRL0.U = q_ctrl.U;

    q_qmr.U = EVADC_G0QMR0.U;
    q_qmr.B.ENGT = 0x1U;
    q_qmr.B.ENTR = 0x1U;
    EVADC_G0QMR0.U = q_qmr.U;

    rcr.U = EVADC_G0RCR0.U;
    rcr.B.DRCTR = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM   = 0x0U;
    rcr.B.WFR   = 0x0U;
    rcr.B.FEN   = 0x0U;
    rcr.B.SRGEN = 0x1U;
    EVADC_G0RCR0.U = rcr.U;

    EVADC_G0REVNP0.U = 0x0U;

    src_cfg.U         = SRC_VADC_G0_SR0.U;
    src_cfg.B.SRPN    = CORE_01_ADC_PHASE_U_SRPN;
    src_cfg.B.TOS     = 0x2U;
    SRC_VADC_G0_SR0.U = src_cfg.U;
    SRC_VADC_G0_SR0.B.SRE = EVADC_ENABLE_PHASE_U_SR;

    EVADC_G0ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G0ARBCFG.B.ANONS != 0x3U) { Nop_Delay(0x1U, 0x1U); }
}

static void Config_G01_C00_An08_Phase_V(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U = EVADC_G1ARBPR.U;
    arb_pr.B.PRIO0 = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0 = 0x1U;
    EVADC_G1ARBPR.U = arb_pr.U;

    ch_ctrl.U = EVADC_G1CHCTR0.U;
    ch_ctrl.B.ICLSEL = 0x2U;
    ch_ctrl.B.RESREG = 0x0U;
    ch_ctrl.B.RESTGT = 0x0U;
    EVADC_G1CHCTR0.U = ch_ctrl.U;

    q_qinr.U = EVADC_G1QINR0.U;
    q_qinr.B.REQCHNR = 0x0U;
    q_qinr.B.RF      = 0x1U;
    q_qinr.B.EXTR    = 0x1U;
    EVADC_G1QINR0.U = q_qinr.U;

    q_ctrl.U = EVADC_G1QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG0;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G1QCTRL0.U = q_ctrl.U;

    q_qmr.U = EVADC_G1QMR0.U;
    q_qmr.B.ENGT = 0x1U;
    q_qmr.B.ENTR = 0x1U;
    EVADC_G1QMR0.U = q_qmr.U;

    rcr.U = EVADC_G1RCR0.U;
    rcr.B.DRCTR = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM   = 0x0U;
    rcr.B.WFR   = 0x0U;
    rcr.B.FEN   = 0x0U;
    rcr.B.SRGEN = 0x1U;
    EVADC_G1RCR0.U = rcr.U;

    EVADC_G1REVNP0.U = 0x0U;

    src_cfg.U         = SRC_VADC_G1_SR0.U;
    src_cfg.B.SRPN    = CORE_01_ADC_PHASE_V_SRPN;
    src_cfg.B.TOS     = 0x2U;
    SRC_VADC_G1_SR0.U = src_cfg.U;
    SRC_VADC_G1_SR0.B.SRE = EVADC_ENABLE_PHASE_V_SR;

    EVADC_G1ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G1ARBCFG.B.ANONS != 0x3U) { Nop_Delay(0x1U, 0x1U); }
}

static void Config_G02_C00_An16_Phase_W(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U = EVADC_G2ARBPR.U;
    arb_pr.B.PRIO0 = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0 = 0x1U;
    EVADC_G2ARBPR.U = arb_pr.U;

    ch_ctrl.U = EVADC_G2CHCTR0.U;
    ch_ctrl.B.ICLSEL = 0x2U;
    ch_ctrl.B.RESREG = 0x0U;
    ch_ctrl.B.RESTGT = 0x0U;
    EVADC_G2CHCTR0.U = ch_ctrl.U;

    q_qinr.U = EVADC_G2QINR0.U;
    q_qinr.B.REQCHNR = 0x0U;
    q_qinr.B.RF      = 0x1U;
    q_qinr.B.EXTR    = 0x1U;
    EVADC_G2QINR0.U = q_qinr.U;

    q_ctrl.U = EVADC_G2QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG0;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G2QCTRL0.U = q_ctrl.U;

    q_qmr.U = EVADC_G2QMR0.U;
    q_qmr.B.ENGT = 0x1U;
    q_qmr.B.ENTR = 0x1U;
    EVADC_G2QMR0.U = q_qmr.U;

    rcr.U = EVADC_G2RCR0.U;
    rcr.B.DRCTR = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM   = 0x0U;
    rcr.B.WFR   = 0x0U;
    rcr.B.FEN   = 0x0U;
    rcr.B.SRGEN = 0x1U;
    EVADC_G2RCR0.U = rcr.U;

    EVADC_G2REVNP0.U = 0x0U;

    src_cfg.U         = SRC_VADC_G2_SR0.U;
    src_cfg.B.SRPN    = CORE_01_ADC_PHASE_W_SRPN;
    src_cfg.B.TOS     = 0x2U;
    SRC_VADC_G2_SR0.U = src_cfg.U;
    SRC_VADC_G2_SR0.B.SRE = EVADC_ENABLE_PHASE_W_SR;

    EVADC_G2ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G2ARBCFG.B.ANONS != 0x3U) { Nop_Delay(0x1U, 0x1U); }
}

static void Config_G08_C08_An40_DC_Link(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U = EVADC_G8ARBPR.U;
    arb_pr.B.PRIO0 = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0 = 0x1U;
    EVADC_G8ARBPR.U = arb_pr.U;

    ch_ctrl.U = EVADC_G8CHCTR8.U;
    ch_ctrl.B.ICLSEL = 0x2U;
    ch_ctrl.B.RESREG = 0x0U;
    ch_ctrl.B.RESTGT = 0x0U;
    EVADC_G8CHCTR8.U = ch_ctrl.U;

    q_qinr.U = EVADC_G8QINR0.U;
    q_qinr.B.REQCHNR = 0x8U;   /* channel 8 */
    q_qinr.B.RF      = 0x1U;
    q_qinr.B.EXTR    = 0x1U;
    EVADC_G8QINR0.U = q_qinr.U;

    q_ctrl.U = EVADC_G8QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_ADCTRIG3;
    q_ctrl.B.GTSEL     = EVADC_XTSEL_ADCTRIG3;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x2U;
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G8QCTRL0.U = q_ctrl.U;

    q_qmr.U = EVADC_G8QMR0.U;
    q_qmr.B.ENGT = 0x1U;
    q_qmr.B.ENTR = 0x1U;
    EVADC_G8QMR0.U = q_qmr.U;

    rcr.U = EVADC_G8RCR0.U;
    rcr.B.DRCTR = 0x0U;
    rcr.B.DMM   = 0x0U;
    rcr.B.WFR   = 0x0U;
    rcr.B.FEN   = 0x0U;
    rcr.B.SRGEN = 0x1U;
    EVADC_G8RCR0.U = rcr.U;

    EVADC_G8REVNP0.U = 0x0U;

    src_cfg.U         = SRC_VADC_G8_SR0.U;
    src_cfg.B.SRPN    = CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN;
    src_cfg.B.TOS     = 0x2U;
    SRC_VADC_G8_SR0.U = src_cfg.U;
    SRC_VADC_G8_SR0.B.SRE = EVADC_ENABLE_DC_LINK_SR;

    EVADC_G8ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G8ARBCFG.B.ANONS != 0x3U) { Nop_Delay(0x1U, 0x1U); }
}

/**********************************************************************************************************************
 * Private — Inline Result Readers
 *********************************************************************************************************************/

static inline void Read_Phase_U(volatile EVADC_Meas_T * const Meas_Ptr)
{
    if (EVADC_G0RES0.B.VF == 0x1U)
    {
        Meas_Ptr->IPhU = ((real32_T)EVADC_G0RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}

static inline void Read_Phase_V(volatile EVADC_Meas_T * const Meas_Ptr)
{
    if (EVADC_G1RES0.B.VF == 0x1U)
    {
        Meas_Ptr->IPhV = ((real32_T)EVADC_G1RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}

static inline void Read_Phase_W(volatile EVADC_Meas_T * const Meas_Ptr)
{
    if (EVADC_G2RES0.B.VF == 0x1U)
    {
        Meas_Ptr->IPhW = ((real32_T)EVADC_G2RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}

static inline void Read_DC_Link(volatile EVADC_Meas_T * const Meas_Ptr)
{
    if (EVADC_G8RES0.B.VF == 0x1U)
    {
        Meas_Ptr->UDcLink = ((real32_T)EVADC_G8RES0.B.RESULT / EVADC_FULL_SCALE) * ADC_MAX_VOLTAGE;
    }
}
