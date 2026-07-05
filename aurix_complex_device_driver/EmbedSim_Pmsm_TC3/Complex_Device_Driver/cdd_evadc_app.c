/**********************************************************************************************************************
 * \file        cdd_evadc_app.c
 * \brief       Implementation of cdd_evadc_app.h — EVADC channel init and readout.
 *
 * \details     Five EVADC channels on four groups follow the same init pattern:
 *              1. Configure arbitration priority (ARBPR)
 *              2. Configure channel control (CHCTR): global class 0
 *              3. Add channel(s) to queue (QINR): auto-refill; external trigger
 *                 on the first entry, back-to-back conversion for follow-ups
 *              4. Configure queue trigger (QCTRL): GTM ATOM via ADCTRIG
 *              5. Enable queue trigger (QMR)
 *              6. Configure data reduction (RCR): 1 sample, service request
 *              7. Configure service request node (SRC): SRPN, CPU1
 *              8. Start converter (ARBCFG.ANONC = 3)
 *
 *              Channel map (AP32541 v1.0 Table 12 / Table 15, AppKit TC387):
 *
 *                  Signal    AppKit pin  Analog in  Group/Ch  Result reg
 *                  VO1  (U)  T10         AN0        G0  CH0   G0RES0
 *                  VO2  (V)  W2          AN24       G3  CH0   G3RES0
 *                  VO3  (W)  W5          AN16       G2  CH0   G2RES0
 *                  VRO       W8          AN8        G1  CH0   G1RES0
 *                  VOLT_DC   W7          AN11       G1  CH3   G1RES3
 *
 *              Trigger wiring (verified, TC38x UM Appendix Table 292 and
 *              GTM_ADCTRIG0OUT0 field encoding, §26.3.9):
 *                  GTM ADC_TRIG0[x] connects to EVADC group x input REQTRI,
 *                  and REQTRI is selected by GxQCTRL0.XTSEL = 0x8 UNIFORMLY
 *                  for all groups.  ADC_TRIG0[x] is driven by
 *                  GTM_ADCTRIG0OUT0.SELx; for SEL0..SEL4 the code 0x8 selects
 *                  CDTM0_DTM5_3 = ATOM0_CH7 (dead-time passthrough, DTM5
 *                  configured passthrough in cdd_gtm_app.c Step 7).
 *
 *                  ATOM0_CH7 (duty 0.9) → ADCTRIG0OUT0.SEL0/1/2/3 →
 *                      ADC_TRIG0[0..3] → G0/G1/G2/G3 REQTRI (XTSEL 0x8)
 *
 *              ALL FIVE channels therefore convert on the same ATOM0_CH7
 *              falling edge each PWM period: G0/G3/G2 phase currents in
 *              parallel, G1 VRO + VOLT_DC back-to-back from its queue.
 *              There is no separate DC-link trigger any more — the former
 *              ADCTRIG3/ATOM0_CH3 concept is void (ATOM0_CH3 is the phase V
 *              low-side PWM output in the v1.5 GTM channel map, and
 *              ADCTRIG3OUTx was never programmed, so the old G8 DC-link
 *              channel in fact never triggered).
 *
 *              Required cdd_gtm_app.c change (Step 6):
 *                  + GTM_ADCTRIG0OUT0.B.SEL3 = 0x8U;   (ATOM0_CH7 → G3, phase V)
 *                  - GTM_ADCTRIG0OUT0.B.SEL7 = 0x8U;   (DELETE: SEL7 feeds G7,
 *                    not G8, and code 0x8 in the SEL5..7 encoding selects
 *                    ATOM5_CH5, not ATOM0_CH7 — the line was doubly wrong)
 *
 *              Zero-current reference: the AP32541 routes the TLE9180D VRO
 *              reference-buffer output to AN8 precisely so software can
 *              measure the true zero-current level instead of assuming the
 *              2.5 V nominal (device spread + temperature drift, DS §9.2).
 *              CddEvadc_ConvertPhaseCurrents() therefore subtracts the LIVE
 *              measured VRO.  Because VRO and VOx are converted by the same
 *              ADC against the same VAREF, this also cancels the ADC
 *              gain/reference error at the zero-current operating point.
 *              Residual per-CSA output offsets (±100 mV uncalibrated,
 *              ±10 mV after TLE9180 auto-cal, DS P_9.6.17/18) are removed by
 *              CddEvadc_CalibratePhaseOffsets() at standstill.
 *
 *              Result readout uses a SINGLE 32-bit read of GxRESy into a
 *              local copy: VF is clear-on-read, so checking VF and then
 *              re-reading RESULT in a second access would race against the
 *              next conversion.  VF and RESULT are taken from the same load.
 *
 * \note        Required cdd_config.h changes (SRPN literals below must match):
 *                  CORE_01_ADC_PHASE_V_SRPN  now serviced by SRC_VADC_G3_SR0
 *                  CORE_01_ADC_DC_LINK_SRPN  (rename of the old
 *                      CORE_01_ADC_G_08_CH_08_DC_LINK_SRPN) on SRC_VADC_G1_SR0
 *                  EVADC_ENABLE_DC_LINK_SR   unchanged, now applied to G1_SR0
 *
 * \note        MISRA C:2012: Rules 8.9, 8.10, 14.4, 15.5, 17.2.
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
#include <stddef.h>            /* NULL (pointer checks)                                          */

/**********************************************************************************************************************
 * Private Macros — ADC Scaling
 *********************************************************************************************************************/

#define EVADC_FULL_SCALE            (4096.0f)    /**< 12-bit LSB divisor: 1 LSB = VAREF/4096   [dimensionless] */
#define EVADC_VAREF_VOLT            (5.0f)       /**< AppKit TC387 VAREF.  AP32541 §3.3.4: CSA outputs
                                                  *   "feed ADCs with an analog range from 0 V to 5 V";
                                                  *   VRO = 2.5 V is mid-scale of this range.        [V]      */
#define EVADC_XTSEL_REQTRI          (0x8U)       /**< GxQCTRL0.XTSEL code for input REQTRI = GTM
                                                  *   ADC_TRIG0[group].  Uniform for all groups
                                                  *   (TC38x UM Appendix Table 292).                          */
#define EVADC_PHASE_SAMPLES         (1U)         /**< DMM data reduction sample count                          */
#define EVADC_ARBPR_PRIO            (0x1U)       /**< Arbitration priority                                     */
#define EVADC_G1_UDC_CHANNEL        (0x3U)       /**< G1 channel number of VOLT_DC (AN11)                      */
#define EVADC_G1_UDC_RESREG         (0x3U)       /**< G1 result register of VOLT_DC (G1RES3)                   */

/** \brief  ADC code → pin voltage  [V] */
#define EVADC_CODE_TO_VOLT(Code)    (((real32_T)(Code) / EVADC_FULL_SCALE) * EVADC_VAREF_VOLT)

/**********************************************************************************************************************
 * Private Macros — Shunt Current Conversion  (CddEvadc_ConvertPhaseCurrents)
 *
 *   I [A] = EVADC_I_SIGN * (V_ox - V_ro_measured - PhaseOffset) / (EVADC_CSA_GAIN * EVADC_SHUNT_R_OHM)
 *
 * Values below are tied to the ACTUAL hardware configuration — change them ONLY together with their source:
 *    EVADC_SHUNT_R_OHM  AP32541 BoM item 34: R17/R27/R37 = 10 mOhm 1% (WSL3637).
 *    EVADC_CSA_GAIN     TLE9180D DS Table 19 P_9.6.7, gain code 100B = 30.81 V/V typ (30.19..31.42).
 *                       MUST track OP_GAIN1/2/3 = 0x44 in the SPI startup batch (cdd_tle9180_app.c).
 *    EVADC_VRO_NOM_V    zcl = 10B in OP_OCL (cdd_tle9180_app.c) → VRO = 2.5 V nominal.  Used only as
 *                       fallback until the first VRO conversion completes; runtime uses measured VRO.
 *    EVADC_I_SIGN       +1.0f/-1.0f — flips once if the ISP/ISN shunt orientation yields VOx BELOW
 *                       VRO for current INTO the terminal (DFC convention: positive = into terminal).
 *                       Determine empirically with a DC injection through one phase.
 *
 * Full-scale check: 2.5 V / (30.81 * 0.010) = ±8.1 A around VRO.
 *********************************************************************************************************************/

#define EVADC_SHUNT_R_OHM           (0.010f)     /**< Low-side phase shunt        [Ohm]  AP32541 BoM #34   */
#define EVADC_CSA_GAIN              (30.81f)     /**< TLE9180D CSA gain, code 100B [V/V] DS P_9.6.7        */
#define EVADC_VRO_NOM_V             (2.5f)       /**< Nominal VRO (zcl=10B), fallback only  [V]            */
#define EVADC_I_SIGN                (1.0f)       /**< Polarity, see block comment  [dimensionless]         */

/** \brief  Sense-voltage → ampere conversion factor:  1/(30.81 * 0.010) = 3.246  [A/V] */
#define EVADC_V_TO_A                (1.0f / (EVADC_CSA_GAIN * EVADC_SHUNT_R_OHM))

/**********************************************************************************************************************
 * Private Macros — DC-Link Voltage Divider  (AP32541 Eq. 1)
 *
 *   VOLT_DC(pin) = VBAT * R113 / (R113 + R114)   with R113 = 5.6 kOhm, R114 = 56 kOhm
 *   → VBAT = VOLT_DC(pin) * 11.0     (12 V bus reads 1.09 V at the pin)
 *********************************************************************************************************************/

#define EVADC_UDC_R113_KOHM         (5.6f)       /**< Divider lower resistor      [kOhm]  AP32541 Fig. 10 */
#define EVADC_UDC_R114_KOHM         (56.0f)      /**< Divider upper resistor      [kOhm]  AP32541 Fig. 10 */

/** \brief  Pin voltage → DC-link bus voltage:  (5.6+56)/5.6 = 11.0  [V/V] */
#define EVADC_UDC_PIN_TO_BUS        ((EVADC_UDC_R113_KOHM + EVADC_UDC_R114_KOHM) / EVADC_UDC_R113_KOHM)

/**********************************************************************************************************************
 * Private Macros — Offset Calibration
 *********************************************************************************************************************/

#define EVADC_CAL_VF_TIMEOUT        (100000U)    /**< Poll iterations per VF wait; >> one 50 us trigger
                                                  *   period at 20 kHz              [loop iterations]     */

/**********************************************************************************************************************
 * Private Data
 *********************************************************************************************************************/

/** \brief  Last measured VRO reference-buffer voltage.  Initialised to the nominal 2.5 V so that
 *          conversions before the first G1 result remain plausible.                          [V] */
static volatile real32_T CddEvadc_VroVolt_G = EVADC_VRO_NOM_V;



/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
static void CddEvadc_InitConvctrl(void);
static void CddEvadc_EnableClock(void);
static void CddEvadc_ConfigGlobal(void);
static void CddEvadc_CalibrateAllGroups(void);
static void CddEvadc_ConfigG00PhaseU(void);
static void CddEvadc_ConfigG03PhaseV(void);
static void CddEvadc_ConfigG02PhaseW(void);
static void CddEvadc_ConfigG01VroUdc(void);

static void CddEvadc_ReadVro(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);
static void CddEvadc_ReadPhaseU(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);
static void CddEvadc_ReadPhaseV(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);
static void CddEvadc_ReadPhaseW(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);
static void CddEvadc_ReadDcLink(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr);



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
    CddEvadc_ConfigG03PhaseV();
    CddEvadc_ConfigG02PhaseW();
    CddEvadc_ConfigG01VroUdc();
}

void CddEvadc_ReadSensorMeas(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    /* VRO first, so the phase conversions of THIS cycle use the freshest reference */
    CddEvadc_ReadVro(CddAppPtr);
    CddEvadc_ReadPhaseU(CddAppPtr);
    CddEvadc_ReadPhaseV(CddAppPtr);
    CddEvadc_ReadPhaseW(CddAppPtr);
    CddEvadc_ReadDcLink(CddAppPtr);
    /* NOTE: CddEvadc_CalibratePhaseOffsets() is deliberately NOT called here.
     * Calibration is a ONE-SHOT commissioning step at standstill with /SOFF
     * asserted.  Calling it cyclically re-zeroes the offsets to the
     * instantaneous residual every cycle, which (a) made Vuo/Vvo/Vwo track
     * the live signal and (b) would force Iu/Iv/Iw to read 0 A permanently
     * once the offsets are subtracted in ConvertPhaseCurrents().             */
}

void CddEvadc_ConvertPhaseCurrents(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    /* Ix = SIGN * (Vx - VRO_measured - Offset_x) * 3.246 A/V.
     * Vr is the live AN8 VRO reading (device spread + drift of the reference
     * buffer); Vuo/Vvo/Vwo are the per-CSA residual output offsets captured
     * once at standstill by CddEvadc_CalibratePhaseOffsets().  Before the
     * first calibration the offsets are 0.0f (struct zero-init), so the
     * conversion degrades gracefully to the uncompensated value.             */
    CddAppPtr->Iu = (real32_T)(EVADC_I_SIGN
                    * ((CddAppPtr->Vu - CddAppPtr->Vr - CddAppPtr->Vuo) * EVADC_V_TO_A));
    CddAppPtr->Iv = (real32_T)(EVADC_I_SIGN
                    * ((CddAppPtr->Vv - CddAppPtr->Vr - CddAppPtr->Vvo) * EVADC_V_TO_A));
    CddAppPtr->Iw = (real32_T)(EVADC_I_SIGN
                    * ((CddAppPtr->Vw - CddAppPtr->Vr - CddAppPtr->Vwo) * EVADC_V_TO_A));

    /* Plausibility signal: |Isum| beyond noise indicates offset drift, a lost sample,
     * or the sampling instant leaving the low-side ON window at high modulation.        */
    CddAppPtr->Isum = CddAppPtr->Iu + CddAppPtr->Iv + CddAppPtr->Iw;
}

void CddEvadc_CalibratePhaseOffsets(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr, uint32_T NumSamples)
{
    /* ONE-SHOT commissioning function — task level, blocking.
     *
     * Preconditions: GTM triggers running, 20 kHz control ISR active (it
     * executes CddEvadc_ReadSensorMeas and increments ControlLoopCounter),
     * TLE9180 in NORMAL mode, bridge gates DISABLED (/SOFF asserted) so the
     * true phase currents are zero and Vx - Vr equals the pure CSA offset.
     *
     * Fresh-sample gating: each accumulated sample waits for a NEW control
     * cycle by observing the low word of ControlLoopCounter.  This avoids
     * racing the ISR on the EVADC result registers (VF is clear-on-read and
     * is consumed by the ISR readout) and guarantees NumSamples DISTINCT
     * conversion sets.  Only the low 32 bits are compared: uint64_T reads
     * are not atomic on TriCore, but a change in the low word is both
     * necessary and sufficient here.
     *
     * On timeout (ISR not running) the stored offsets are left UNCHANGED.  */
    uint32_T Sample;
    uint32_T Guard;
    uint32_T LastCount;
    real32_T SumU = 0.0f;
    real32_T SumV = 0.0f;
    real32_T SumW = 0.0f;

    if (NumSamples == 0x0U)
    {
        return;                              /* Invalid parameter — offsets unchanged */
    }

    LastCount = (uint32_T)CddAppPtr->ControlLoopCounter;

    for (Sample = 0x0U; Sample < NumSamples; Sample++)
    {
        /* Wait for the next completed control cycle (fresh conversion set) */
        Guard = 0x0U;
        while ((uint32_T)CddAppPtr->ControlLoopCounter == LastCount)
        {
            Guard++;
            if (Guard >= EVADC_CAL_VF_TIMEOUT)
            {
                return;                      /* Triggers/ISR not running — offsets unchanged */
            }
        }
        LastCount = (uint32_T)CddAppPtr->ControlLoopCounter;

        SumU += (CddAppPtr->Vu - CddAppPtr->Vr);
        SumV += (CddAppPtr->Vv - CddAppPtr->Vr);
        SumW += (CddAppPtr->Vw - CddAppPtr->Vr);
    }

    CddAppPtr->Vuo = SumU / (real32_T)NumSamples;
    CddAppPtr->Vvo = SumV / (real32_T)NumSamples;
    CddAppPtr->Vwo = SumW / (real32_T)NumSamples;
}

real32_T CddEvadc_GetVroVolt(void)
{
    return CddEvadc_VroVolt_G;
}

/**********************************************************************************************************************
 * ISR Vector Registrations
 * SRPN literals must match cdd_config.h: 90 (U/G0), 91 (V/G3), 92 (W/G2), 95 (VRO+DC-link/G1).
 * ISR names are vector-table entries — unchanged per MISRA DEV convention.
 *********************************************************************************************************************/

EMBED_SIM_INTERRUPT(EVADC_G0_Isr, 0x0u, CORE_01_ADC_PHASE_U_SRPN);    /* CORE_01_ADC_PHASE_U_SRPN   */
EMBED_SIM_INTERRUPT(EVADC_G3_Isr, 0x0u, 91);    /* CORE_01_ADC_PHASE_V_SRPN   */
EMBED_SIM_INTERRUPT(EVADC_G2_Isr, 0x0u, 92);    /* CORE_01_ADC_PHASE_W_SRPN   */
EMBED_SIM_INTERRUPT(EVADC_G1_Isr, 0x0u, 95);    /* CORE_01_ADC_DC_LINK_SRPN   */

/**********************************************************************************************************************
 * ISR Bodies
 *********************************************************************************************************************/
volatile  int paul_counter = 0;
void EVADC_G0_Isr(void)
{ /* Phase U result ready — handled in control loop        */
    paul_counter++;
    if(paul_counter>1000)
    {
        paul_counter=1;
    }

}
void EVADC_G3_Isr(void) { /* Phase V result ready — handled in control loop        */

    paul_counter++;
      if(paul_counter>1000)
      {
          paul_counter=1;
      }
}
void EVADC_G2_Isr(void)
{
    CddGpio_ToggleIsrTiming_P14_5();
    /* Phase W result ready — handled in control loop */
    CddGpio_ToggleIsrTiming_P14_5();
}
void EVADC_G1_Isr(void) {
    paul_counter++;

}



/**********************************************************************************************************************
 * Private — Hardware Init Helpers
 *********************************************************************************************************************/

static void CddEvadc_InitConvctrl(void)
{
    CddSys_ClearCpuWdtEndInit();
    CONVCTRL_CLC.U = 0x00000000U;                  /* Enable CONVCTRL module        */
    while (CONVCTRL_CLC.B.DISS == 0x1U)
    {
        CddSys_NopDelay(1U, 1U);
    }
    CONVCTRL_CCCTRL.U = 0xB0000000U;               /* Unlock converter control regs */
    CONVCTRL_PHSCFG.U = 0x00008007U;               /* fADC=160MHz, fPHSYNC=20MHz    */
    CONVCTRL_CCCTRL.U = 0x00000000U;               /* Lock converter control regs   */
    CddSys_SetCpuWdtEndInit();
}

static void CddEvadc_EnableClock(void)
{
    CddSys_ClearCpuWdtEndInit();
    EVADC_CLC.B.DISR = 0x0U;
    CddSys_SetCpuWdtEndInit();
    while (EVADC_CLC.B.DISS != 0x0U)
    {
        CddSys_NopDelay(0x1U, 0x1U);
    }
}

static void CddEvadc_ConfigGlobal(void)
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
    EVADC_G3ANCFG.U    = an_cfg.U;

    iclass.U            = EVADC_GLOBICLASS0.U;
    iclass.B.CMS        = 0x0U;
    iclass.B.AIPS       = 0x0U;
    iclass.B.STCS       = 0xFU;   /* 850 ns sample time (16 clocks at 40 MHz) */
    EVADC_GLOBICLASS0.U = iclass.U;
}

static void CddEvadc_CalibrateAllGroups(void)
{
    EVADC_GLOBCFG.B.SUCAL = 0x1U;
    while (EVADC_G0ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
    while (EVADC_G1ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
    while (EVADC_G2ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
    while (EVADC_G3ARBCFG.B.CAL == 0x1U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/**********************************************************************************************************************
 * Private — Channel Configuration Helpers
 *********************************************************************************************************************/

/** \brief  G0 CH0 = AN0 = VO1 — phase U current.  Triggered by ADCTRIG0 (ATOM0_CH7). */
static void CddEvadc_ConfigG00PhaseU(void)
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
    ch_ctrl.B.ICLSEL   = 0x2U;                     /* Global class 0                */
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G0CHCTR0.U   = ch_ctrl.U;

    q_qinr.U           = EVADC_G0QINR0.U;
    q_qinr.B.REQCHNR   = 0x0U;
    q_qinr.B.RF        = 0x1U;                     /* Auto-refill                   */
    q_qinr.B.EXTR      = 0x1U;                     /* Wait for external trigger     */
    EVADC_G0QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G0QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_REQTRI;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x1U;    /* Falling edge of ATOM0_CH7 (duty 0.9).  VERIFY on the scope
                                   * (P14.5 toggle vs. gate signal) that the sample instant sits
                                   * centred in the all-low-side-ON window; otherwise the phase
                                   * with the highest duty samples an open shunt and Isum grows
                                   * with modulation index.                                       */
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G0QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G0QMR0.U;
    q_qmr.B.ENGT       = 0x1U;                     /* Requests issued, gate ignored */
    q_qmr.B.ENTR       = 0x1U;                     /* External trigger enabled      */
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
    src_cfg.B.TOS      = 0x0U;
    SRC_VADC_G0_SR0.U  = src_cfg.U;
    SRC_VADC_G0_SR0.B.SRE = EVADC_ENABLE_PHASE_U_SR;

    EVADC_G0ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G0ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/** \brief  G3 CH0 = AN24 = VO2 — phase V current.  Triggered by ADCTRIG0 (ATOM0_CH7).
 *
 *  \note   Phase V was previously (and incorrectly) configured on G1 CH0 = AN8, which
 *          the AP32541 routes to the TLE9180D VRO reference output (Table 12): the
 *          "phase V" channel returned a 2.5 V DC level, so Iv computed as ~0 A and
 *          Isum showed -Iv(actual) — a sinusoid at electrical frequency scaling with
 *          load.  AN24 = P40.0 is the true VO2 signal on AppKit pin W2.               */
static void CddEvadc_ConfigG03PhaseV(void)
{
    Ifx_EVADC_G_ARBPR   arb_pr;
    Ifx_EVADC_G_CHCTR   ch_ctrl;
    Ifx_EVADC_G_Q_QINR  q_qinr;
    Ifx_EVADC_G_Q_QCTRL q_ctrl;
    Ifx_EVADC_G_Q_QMR   q_qmr;
    Ifx_EVADC_G_RCR     rcr;
    Ifx_SRC_SRCR        src_cfg;

    arb_pr.U           = EVADC_G3ARBPR.U;
    arb_pr.B.PRIO0     = EVADC_ARBPR_PRIO;
    arb_pr.B.ASEN0     = 0x1U;
    EVADC_G3ARBPR.U    = arb_pr.U;

    ch_ctrl.U          = EVADC_G3CHCTR0.U;
    ch_ctrl.B.ICLSEL   = 0x2U;                     /* Global class 0                */
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G3CHCTR0.U   = ch_ctrl.U;

    q_qinr.U           = EVADC_G3QINR0.U;
    q_qinr.B.REQCHNR   = 0x0U;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x1U;
    EVADC_G3QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G3QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_REQTRI;       /* ADC_TRIG0[3] = ATOM0_CH7 (verified)   */
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x1U;                     /* Falling edge — same instant as G0/G2  */
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G3QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G3QMR0.U;
    q_qmr.B.ENGT       = 0x1U;
    q_qmr.B.ENTR       = 0x1U;
    EVADC_G3QMR0.U     = q_qmr.U;

    rcr.U              = EVADC_G3RCR0.U;
    rcr.B.DRCTR        = EVADC_PHASE_SAMPLES - 1U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x1U;
    EVADC_G3RCR0.U     = rcr.U;

    EVADC_G3REVNP0.U   = 0x0U;

    src_cfg.U          = SRC_VADC_G3_SR0.U;
    src_cfg.B.SRPN     = CORE_01_ADC_PHASE_V_SRPN;
    src_cfg.B.TOS      = 0x0U;
    SRC_VADC_G3_SR0.U  = src_cfg.U;
    SRC_VADC_G3_SR0.B.SRE = EVADC_ENABLE_PHASE_V_SR;

    EVADC_G3ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G3ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/** \brief  G2 CH0 = AN16 = VO3 — phase W current.  Triggered by ADCTRIG0 (ATOM0_CH7). */
static void CddEvadc_ConfigG02PhaseW(void)
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
    ch_ctrl.B.ICLSEL   = 0x2U;                     /* Global class 0                */
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
    q_ctrl.B.XTSEL     = EVADC_XTSEL_REQTRI;
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x1U;                     /* Falling edge — same instant as G0/G3  */
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
    src_cfg.B.TOS      = 0x0U;
    SRC_VADC_G2_SR0.U  = src_cfg.U;
    SRC_VADC_G2_SR0.B.SRE = EVADC_ENABLE_PHASE_W_SR;

    EVADC_G2ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G2ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/** \brief  G1 queue 0 with two entries — VRO reference and DC-link voltage.
 *
 *              Entry 0:  CH0 = AN8  = VRO      → G1RES0  (EXTR=1, starts on ADC_TRIG0[1])
 *              Entry 1:  CH3 = AN11 = VOLT_DC  → G1RES3  (EXTR=0, converts back-to-back)
 *
 *          One ATOM0_CH7 falling edge (ADC_TRIG0[1], same instant as the
 *          phase-current groups) therefore converts both channels
 *          sequentially; both entries auto-refill.  The service request is
 *          generated on the LAST result (G1RES3 via RCR3), so a G1 SR means
 *          both values are fresh.
 *
 *  \note   The old implementation read "DC-link" on G8 CH8 = AN40, which is not
 *          connected to VOLT_DC on this board (AP32541 Table 15: VOLT_DC = AN11).  */
static void CddEvadc_ConfigG01VroUdc(void)
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

    /* CH0 — VRO → result register 0 */
    ch_ctrl.U          = EVADC_G1CHCTR0.U;
    ch_ctrl.B.ICLSEL   = 0x2U;                     /* Global class 0                */
    ch_ctrl.B.RESREG   = 0x0U;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G1CHCTR0.U   = ch_ctrl.U;

    /* CH3 — VOLT_DC → result register 3 */
    ch_ctrl.U          = EVADC_G1CHCTR3.U;
    ch_ctrl.B.ICLSEL   = 0x2U;                     /* Global class 0                */
    ch_ctrl.B.RESREG   = EVADC_G1_UDC_RESREG;
    ch_ctrl.B.RESTGT   = 0x0U;
    EVADC_G1CHCTR3.U   = ch_ctrl.U;

    /* Queue entry 0: VRO, waits for external trigger */
    q_qinr.U           = 0x0U;
    q_qinr.B.REQCHNR   = 0x0U;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x1U;
    EVADC_G1QINR0.U    = q_qinr.U;

    /* Queue entry 1: VOLT_DC, converts immediately after VRO (no own trigger) */
    q_qinr.U           = 0x0U;
    q_qinr.B.REQCHNR   = EVADC_G1_UDC_CHANNEL;
    q_qinr.B.RF        = 0x1U;
    q_qinr.B.EXTR      = 0x0U;
    EVADC_G1QINR0.U    = q_qinr.U;

    q_ctrl.U           = EVADC_G1QCTRL0.U;
    q_ctrl.B.XTWC      = 0x1U;
    q_ctrl.B.TRSEL     = 0x0U;
    q_ctrl.B.XTSEL     = EVADC_XTSEL_REQTRI;       /* ADC_TRIG0[1] = ATOM0_CH7 — same edge as
                                                    * the phase groups; VRO+VOLT_DC convert
                                                    * back-to-back at 20 kHz                  */
    q_ctrl.B.GTSEL     = 0x0U;
    q_ctrl.B.GTWC      = 0x1U;
    q_ctrl.B.XTMODE    = 0x1U;                     /* Falling edge of ATOM0_CH7             */
    q_ctrl.B.SRCRESREG = 0x0U;
    EVADC_G1QCTRL0.U   = q_ctrl.U;

    q_qmr.U            = EVADC_G1QMR0.U;
    q_qmr.B.ENGT       = 0x1U;
    q_qmr.B.ENTR       = 0x1U;
    EVADC_G1QMR0.U     = q_qmr.U;

    /* RCR0 (VRO): no service request — SR is raised by the last conversion only */
    rcr.U              = EVADC_G1RCR0.U;
    rcr.B.DRCTR        = 0x0U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x0U;
    EVADC_G1RCR0.U     = rcr.U;

    /* RCR3 (VOLT_DC): service request after the pair completes */
    rcr.U              = EVADC_G1RCR3.U;
    rcr.B.DRCTR        = 0x0U;
    rcr.B.DMM          = 0x0U;
    rcr.B.WFR          = 0x0U;
    rcr.B.FEN          = 0x0U;
    rcr.B.SRGEN        = 0x1U;
    EVADC_G1RCR3.U     = rcr.U;

    EVADC_G1REVNP0.U   = 0x0U;

   /* src_cfg.U          = SRC_VADC_G1_SR0.U;
    src_cfg.B.SRPN     = CORE_01_ADC_DC_LINK_SRPN;
    src_cfg.B.TOS      = 0x0U;
    SRC_VADC_G1_SR0.U  = src_cfg.U;
    SRC_VADC_G1_SR0.B.SRE = EVADC_ENABLE_DC_LINK_SR;
*/
    EVADC_G1ARBCFG.B.ANONC = 0x3U;
    while (EVADC_G1ARBCFG.B.ANONS != 0x3U) { CddSys_NopDelay(0x1U, 0x1U); }
}

/**********************************************************************************************************************
 * Private — Result Readout Helpers
 *
 * Pattern: one 32-bit read of the result register into a local copy.  VF is
 * clear-on-read; taking VF and RESULT from the SAME load avoids the race of
 * the old two-access pattern (VF check, then separate RESULT read).  If VF
 * is clear the previous value is retained (Rule 15.7 documented else).
 *********************************************************************************************************************/

static void CddEvadc_ReadVro(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    Ifx_EVADC_G_RES Res;

    Res.U = EVADC_G1RES0.U;
    if (Res.B.VF == 0x1U)
    {
        CddAppPtr->Vr = EVADC_CODE_TO_VOLT(Res.B.RESULT);
        /* Sanity anchor: with VAREF = 5 V and zcl = 10B this reads ~2048 counts
         * (~2.5 V).  A grossly different value means the reference assumption
         * or the channel mapping is wrong.                                     */
    }
    else
    {
        /* No fresh result — previous value retained (Rule 15.7) */
    }
}

static void CddEvadc_ReadPhaseU(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    Ifx_EVADC_G_RES Res;

    Res.U = EVADC_G0RES0.U;
    if (Res.B.VF == 0x1U)
    {
        CddAppPtr->Vu = EVADC_CODE_TO_VOLT(Res.B.RESULT);
    }
    else
    {
        /* No fresh result — previous value retained (Rule 15.7) */
    }
}

static void CddEvadc_ReadPhaseV(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    Ifx_EVADC_G_RES Res;

    Res.U = EVADC_G3RES0.U;
    if (Res.B.VF == 0x1U)
    {
        CddAppPtr->Vv = EVADC_CODE_TO_VOLT(Res.B.RESULT);
    }
    else
    {
        /* No fresh result — previous value retained (Rule 15.7) */
    }
}

static void CddEvadc_ReadPhaseW(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    Ifx_EVADC_G_RES Res;

    Res.U = EVADC_G2RES0.U;
    if (Res.B.VF == 0x1U)
    {
        CddAppPtr->Vw = EVADC_CODE_TO_VOLT(Res.B.RESULT);
    }
    else
    {
        /* No fresh result — previous value retained (Rule 15.7) */
    }
}

static void CddEvadc_ReadDcLink(P2VAR(volatile CddApp_T, AUTOMATIC, CDD_APPL_DATA) CddAppPtr)
{
    Ifx_EVADC_G_RES Res;

    Res.U = EVADC_G1RES3.U;
    if (Res.B.VF == 0x1U)
    {
        /* Vdc is the BUS voltage [V], i.e. pin voltage scaled by the on-board
         * 5.6k/56k divider inverse (x11.0, AP32541 Eq. 1): 12 V bus → 1.09 V pin.
         * Callers that previously consumed the raw pin voltage must be updated.  */
        CddAppPtr->Vdc = EVADC_CODE_TO_VOLT(Res.B.RESULT) * EVADC_UDC_PIN_TO_BUS;
    }
    else
    {
        /* No fresh result — previous value retained (Rule 15.7) */
    }
}


