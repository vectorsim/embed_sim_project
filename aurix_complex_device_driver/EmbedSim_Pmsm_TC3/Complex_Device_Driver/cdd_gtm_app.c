/**********************************************************************************************************************
 * \file        cdd_gtm_app.c
 * \brief       Implementation of cdd_gtm_app.h — GTM ATOM0 + DTM driver for
 *              3-phase FOC PWM generation on the AP32541 motor control board.
 *
 * \details     Channel assignment:
 *
 *              ATOM0_CH0  Master  — centre-aligned carrier, CCU1 ISR → CPU1
 *                                   Output: P00.0 (scope probe)
 *              ATOM0_CH3  ADC-R   — valley-aligned trigger, Resolver + DC-link
 *                                   Internal: ADCTRIG3 → EVADC G3 / G11 / G8
 *              ATOM0_CH4  ADC-I   — valley-aligned trigger, Phase currents
 *                                   Internal: ADCTRIG0 → EVADC G0 / G1 / G2
 *              ATOM0_CH5  Phase U — DTM0_CH0 → IL1 P00.2 (LS) + /IH1 P00.3 (HS)
 *              ATOM0_CH6  Phase V — DTM0_CH1 → IL2 P00.4 (LS) + /IH2 P00.5 (HS)
 *              ATOM0_CH7  Phase W — DTM0_CH2 → IL3 P00.6 (LS) + /IH3 P00.7 (HS)
 *
 *              ADC valley-sampling strategy (6-pin DTM mode):
 *                  SR1 = ADC_VALLEY_OFFSET_TICKS after CCU0 reset (valley).
 *                  All three LS FETs are ON at the carrier valley — the only
 *                  window where all three shunts carry settled phase currents
 *                  (TLE9180D DS 16.3.4 cross-talk note).
 *                  200 ns offset > CDD_GTM_DTM_DEAD_TIME_TICKS (100 ns) so
 *                  the ADC trigger never falls inside the dead-time window.
 *
 *              DTM polarity (6-pin, AP32541 TLE9180D wiring):
 *                  ILx  active HIGH → DTM low-side  output non-inverted (POL_L = 0)
 *                  /IHx active LOW  → DTM high-side output inverted      (POL_H = 1)
 *
 *              Dead-time owner: GTM DTM (CDD_GTM_DTM_DEAD_TIME_TICKS).
 *              TLE9180D Dt_hs / Dt_ls set to minimum 107 ns in SPI config.
 *
 *              Code pattern: flat Infineon EGTM_ATOM_3_Phase_Inverter_HRPWM
 *              style — all interim structs declared at the top of
 *              Initialize_GTM_Module(), shared structs accumulated across all
 *              phase blocks, written back once at the bottom.  No private
 *              sub-functions.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_gtm_app.h"
#include "cdd_gpio_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "IfxGtm_reg.h"
#include "IfxGtm_Atom.h"
#include "IfxSrc_reg.h"

/**********************************************************************************************************************
 * ISR Vector Registration  (one IFX_INTERRUPT per ISR, in the owning .c only)
 *********************************************************************************************************************/

/* CORE_01_ATOM_00_CH_00_CL_SRPN = 80 — literal required by TASKING IFX_INTERRUPT */
IFX_INTERRUPT(GTM_Atom_00_Ch_00_Isr, 1, 80);

/**********************************************************************************************************************
 * ISR Body — FOC 20 kHz control loop
 *********************************************************************************************************************/

void GTM_Atom_00_Ch_00_Isr(void)
{
    /* Clear CCU1 interrupt flag */
    GTM_ATOM0_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1U;

    /* TODO: call EmbedSim_Step() / Read_EVADC_Sensor_Measurement() here */
}

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  ATOM SOMP mode value (Signal Output Mode PWM)                       */
#define ATOM_MODE_SOMP              (0x2U)

/** \brief  TOUTSEL mux value for GTM ATOM alternate output function             */
#define TOUTSEL_GTM_ATOM            (0x02U)


/**
 * \brief  ADC trigger offset from PWM valley  [CMU CLK0 ticks]
 *
 * \details At 200 MHz CMU CLK0: 1 tick = 5 ns.  Default 40 ticks = 200 ns.
 *          Must satisfy: ADC_VALLEY_OFFSET_TICKS > CDD_GTM_DTM_DEAD_TIME_TICKS
 *          so that the ADC trigger never falls inside the dead-time window.
 */
#define ADC_VALLEY_OFFSET_TICKS     (40U)

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/** \brief  Control loop period  [CMU CLK0 ticks]                               */
static uint32_T Period_Ticks_G;

/** \brief  Control loop half-period  [CMU CLK0 ticks]                          */
static uint32_T Half_Period_Ticks_G;

/** \brief  Control loop sample time  [s]                                        */
static real32_T Sample_Time_G;

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * Initialize_GTM_Module
 *
 * Flat Infineon-pattern init — see EGTM_ATOM_3_Phase_Inverter_HRPWM.c.
 *
 * All interim structs are declared at the top.
 * Structs shared across phase blocks (dtmCtrl, glbCtrl, fupdCtrl, endisCtrl,
 * outenCtrl) are read once from hardware, accumulated across all phase
 * sections, then written back once at the bottom.
 * Per-phase structs (chCtrl, chDtv) are reused: read fresh from the relevant
 * hardware register, modified, written back, then reused for the next phase.
 *
 * Step labels follow the Infineon convention:
 *   M1..M6  — Master channel   ATOM0_CH0  (carrier + CCU1 ISR)
 *   R1..R4  — ADC-R trigger    ATOM0_CH3  (Resolver SIN/COS + DC-link)
 *   I1..I4  — ADC-I trigger    ATOM0_CH4  (Phase currents U/V/W)
 *   U1..U5  — Phase U          ATOM0_CH5 → DTM0_CH0 → P00.2 / P00.3
 *   V1..V5  — Phase V          ATOM0_CH6 → DTM0_CH1 → P00.4 / P00.5
 *   W1..W5  — Phase W          ATOM0_CH7 → DTM0_CH2 → P00.6 / P00.7
 *   D1      — DTM global       DTM0_CTRL write-back
 *   A1..A2  — AGC write-back + HOST_TRIG
 *------------------------------------------------------------------------------------------------------------------*/
void Initialize_GTM_Module(void)
{
    /* Interim structs — all declared at the top following Infineon pattern  */
    Ifx_GTM_ATOM_CH_CTRL        chCtrl;       /* reused per channel           */
    Ifx_GTM_ATOM_CH_IRQ_EN      chIrqEn;
    Ifx_SRC_SRCR                srcCfg;
    Ifx_GTM_ATOM_AGC_GLB_CTRL   glbCtrl;     /* shared — accumulated, written back once */
    Ifx_GTM_ATOM_AGC_FUPD_CTRL  fupdCtrl;    /* shared — accumulated, written back once */
    Ifx_GTM_ATOM_AGC_ENDIS_CTRL endisCtrl;   /* shared — accumulated, written back once */
    Ifx_GTM_ATOM_AGC_OUTEN_CTRL outenCtrl;   /* shared — accumulated, written back once */

    /* ------------------------------------------------------------------ */
    /* Pre-compute timing constants from live CMU CLK0 frequency           */
    /* ------------------------------------------------------------------ */
    Period_Ticks_G      = (uint32_T)((real32_T)GTM_CMU_CLK0_FREQUENCY /
                                      (real32_T)BMC_SWC3_ED_CONTROL_FREQUENCY);
    Half_Period_Ticks_G = Period_Ticks_G / 2U;
    Sample_Time_G       = 1.0f / (real32_T)BMC_SWC3_ED_CONTROL_FREQUENCY;

    /* ------------------------------------------------------------------ */
    /* Enable GTM module clock and CMU CLK0                                */
    /* CMU CLK0 frequency set upstream by Set_GTM_CMU_CLK_00_Frequency()  */
    /* ------------------------------------------------------------------ */
    Clear_CPU_WDT_EndInit();
    GTM_CLC.B.DISR = 0x0U;                /* enable GTM module clock  ds2 P.61  */
    Set_CPU_WDT_EndInit();

    while (GTM_CLC.B.DISS != 0x0U) {}     /* wait for module clock active        */

    GTM_CMU_CLK_EN.B.EN_CLK0 = 0x2U;      /* enable CMU CLK0          ds2 P.184  */

    /* ------------------------------------------------------------------ */
    /* Read shared interim structs once — modified per-channel below       */
    /* ------------------------------------------------------------------ */
    glbCtrl.U   = GTM_ATOM0_AGC_GLB_CTRL.U;
    fupdCtrl.U  = GTM_ATOM0_AGC_FUPD_CTRL.U;
    endisCtrl.U = GTM_ATOM0_AGC_ENDIS_CTRL.U;
    outenCtrl.U = GTM_ATOM0_AGC_OUTEN_CTRL.U;

    /* ================================================================== */
    /* MASTER CHANNEL — ATOM0_CH0                                          */
    /* Centre-aligned SOMP carrier.  CCU1 fires at half-period → CPU1 ISR  */
    /* Output: P00.0 scope probe                                            */
    /* ================================================================== */

    /* M1. Configure channel control */
    chCtrl.U              = GTM_ATOM0_CH0_CTRL.U;
    chCtrl.B.MODE         = ATOM_MODE_SOMP;   /* centre-aligned PWM              */
    chCtrl.B.RST_CCU0     = 0x0U;             /* master does not self-reset       */
    chCtrl.B.TRIGOUT      = 0x1U;             /* expose CCU0 event to slaves      */
    chCtrl.B.ARU_EN       = 0x0U;
    chCtrl.B.CLK_SRC_SR   = 0x0U;             /* CMU CLK0                         */
    chCtrl.B.SL           = 0x0U;
    GTM_ATOM0_CH0_CTRL.U  = chCtrl.U;

    /* M2. Shadow registers: period + CCU1 at half-period for control ISR  */
    GTM_ATOM0_CH0_SR0.B.SR0 = Period_Ticks_G;
    GTM_ATOM0_CH0_SR1.B.SR1 = Half_Period_Ticks_G;

    /* M3. Enable CCU1 interrupt, pulse mode                               */
    chIrqEn.U                 = GTM_ATOM0_CH0_IRQ_EN.U;
    chIrqEn.B.CCU0TC_IRQ_EN   = 0x0U;
    chIrqEn.B.CCU1TC_IRQ_EN   = 0x1U;         /* CCU1 fires at half-period        */
    GTM_ATOM0_CH0_IRQ_EN.U    = chIrqEn.U;
    GTM_ATOM0_CH0_IRQ_MODE.B.IRQ_MODE = 0x0U; /* pulse mode                       */

    /* M4. Service request → CPU1 */
    srcCfg.U              = SRC_GTM_ATOM0_0.U;
    srcCfg.B.SRPN         = CORE_01_ATOM_00_CH_00_CL_SRPN;
    srcCfg.B.TOS          = 0x2U;             /* CPU1                             */
    SRC_GTM_ATOM0_0.U     = srcCfg.U;
    SRC_GTM_ATOM0_0.B.SRE = 0x1U;

    /* M5. Pin mux P00.0 — master scope probe */
    GTM_TOUTSEL1.B.SEL1 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_Master_P00_0();

    /* M6. AGC: enable CH0 shadow update, channel, output */
    glbCtrl.B.UPEN_CTRL0    = 0x2U;
    fupdCtrl.B.FUPD_CTRL0   = 0x2U;
    endisCtrl.B.ENDIS_CTRL0 = 0x2U;
    outenCtrl.B.OUTEN_CTRL0 = 0x2U;

    /* ================================================================== */
    /* ADC RESOLVER TRIGGER — ATOM0_CH3                                    */
    /* Valley-aligned → EVADC G3 (SIN+ AN24), G11 (COS+ AN19), G8 (VOLT_DC AN40) */
    /* Internal trigger only — no physical pin                              */
    /* ================================================================== */

    /* R1. Configure channel control — slave, resets on master CCU0 event  */
    chCtrl.U              = GTM_ATOM0_CH3_CTRL.U;
    chCtrl.B.MODE         = ATOM_MODE_SOMP;
    chCtrl.B.RST_CCU0     = 0x1U;             /* reset synchronously with master  */
    chCtrl.B.TRIGOUT      = 0x0U;
    chCtrl.B.ARU_EN       = 0x0U;
    chCtrl.B.CLK_SRC_SR   = 0x0U;
    chCtrl.B.SL           = 0x0U;
    GTM_ATOM0_CH3_CTRL.U  = chCtrl.U;

    /* R2. Shadow registers: CCU1 fires ADC_VALLEY_OFFSET_TICKS after valley */
    GTM_ATOM0_CH3_SR1.U = ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH3_SR0.U = Period_Ticks_G;

    /* R3. Route CH3 CCU1 to EVADC G3 / G11 / G8 via ADCTRIG3  appx1 P.781 */
    GTM_ADCTRIG3OUT0.B.SEL3 = 0x1U;           /* SIN+ (AN24) → G3                 */
    GTM_ADCTRIG3OUT1.B.SEL3 = 0x1U;           /* COS+ (AN19) → G11                */
    GTM_ADCTRIG3OUT1.B.SEL0 = 0x1U;           /* VOLT_DC (AN40) → G8              */

    /* R4. AGC: enable CH3 */
    glbCtrl.B.UPEN_CTRL3    = 0x2U;
    fupdCtrl.B.FUPD_CTRL3   = 0x2U;
    endisCtrl.B.ENDIS_CTRL3 = 0x2U;
    outenCtrl.B.OUTEN_CTRL3 = 0x2U;

    /* ================================================================== */
    /* ADC PHASE CURRENT TRIGGER — ATOM0_CH4                               */
    /* Valley-aligned → EVADC G0 (Ph U AN00), G1 (Ph V AN08), G2 (Ph W AN16) */
    /* All LS FETs ON at valley = only valid 3-shunt measurement window     */
    /* Internal trigger only — no physical pin                              */
    /* ================================================================== */

    /* I1. Configure channel control — slave, resets on master CCU0 event  */
    chCtrl.U              = GTM_ATOM0_CH4_CTRL.U;
    chCtrl.B.MODE         = ATOM_MODE_SOMP;
    chCtrl.B.RST_CCU0     = 0x1U;
    chCtrl.B.TRIGOUT      = 0x0U;
    chCtrl.B.ARU_EN       = 0x0U;
    chCtrl.B.CLK_SRC_SR   = 0x0U;
    chCtrl.B.SL           = 0x0U;
    GTM_ATOM0_CH4_CTRL.U  = chCtrl.U;

    /* I2. Shadow registers: valley trigger */
    GTM_ATOM0_CH4_SR1.U = ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH4_SR0.U = Period_Ticks_G;

    /* I3. Route CH4 CCU1 to EVADC G0 / G1 / G2 via ADCTRIG0  appx1 P.769 */
    GTM_ADCTRIG0OUT0.B.SEL0 = 0x5U;           /* Phase U → G0                     */
    GTM_ADCTRIG0OUT0.B.SEL1 = 0x5U;           /* Phase V → G1                     */
    GTM_ADCTRIG0OUT0.B.SEL2 = 0x5U;           /* Phase W → G2                     */

    /* I4. AGC: enable CH4 */
    glbCtrl.B.UPEN_CTRL4    = 0x2U;
    fupdCtrl.B.FUPD_CTRL4   = 0x2U;
    endisCtrl.B.ENDIS_CTRL4 = 0x2U;
    outenCtrl.B.OUTEN_CTRL4 = 0x2U;

    /* ================================================================== */
    /* PHASE U — ATOM0_CH5 → DTM0_CH0 → IL1 P00.2 (LS) + /IH1 P00.3 (HS) */
    /* ================================================================== */

    /* U1. Configure channel control — slave, resets with master           */
    chCtrl.U              = GTM_ATOM0_CH5_CTRL.U;
    chCtrl.B.MODE         = ATOM_MODE_SOMP;
    chCtrl.B.RST_CCU0     = 0x1U;             /* reset on master CCU0 → centre-aligned */
    chCtrl.B.TRIGOUT      = 0x0U;
    chCtrl.B.ARU_EN       = 0x0U;
    chCtrl.B.CLK_SRC_SR   = 0x0U;
    chCtrl.B.SL           = 0x0U;
    GTM_ATOM0_CH5_CTRL.U  = chCtrl.U;

    /* U2. Shadow registers: period + 50% initial duty */
    GTM_ATOM0_CH5_SR0.B.SR0 = Period_Ticks_G;
    GTM_ATOM0_CH5_SR1.B.SR1 = Half_Period_Ticks_G;

    /* U3. DTM0_CH0 dead-time values
     *     iLLD 1.20.0: DTV bitfields are RELRISE (rising edge) / RELFALL (falling edge).
     *     Both edges get the same dead-time value.                           */
    GTM_CDTM0_DTM0_CH0_DTV.B.RELRISE = CDD_GTM_DTM_DEAD_TIME_TICKS;
    GTM_CDTM0_DTM0_CH0_DTV.B.RELFALL = CDD_GTM_DTM_DEAD_TIME_TICKS;

    /* U4. DTM0_CH0 polarity — iLLD 1.20.0 register map:
     *     Per-channel polarity lives in the SHARED DTM0_CH_CTRL1 register.
     *     There is no per-channel DTM0_CH0_CTRL register.
     *     CH_CTRL1 layout:  POL0_x = output 0 (low-side),  POL1_x = output 1 (high-side)
     *     Channel index x: 0=CH0, 1=CH1, 2=CH2, 3=CH3
     *     AP32541 wiring: IL1 active HIGH → POL0_0 = 0 (non-inverted)
     *                    /IH1 active LOW  → POL1_0 = 1 (inverted)           */
    GTM_CDTM0_DTM0_CH_CTRL2.B.POL0_0 = 0x0U;  /* IL1  active HIGH (LS, non-inverted) */
    GTM_CDTM0_DTM0_CH_CTRL2.B.POL1_0 = 0x1U;  /* /IH1 active LOW  (HS, inverted)     */

    /* U5. Pin mux: P00.2 = IL1 (DTM0_CH0_OUT), P00.3 = /IH1 (DTM0_CH0_OUT_N) */
    GTM_TOUTSEL1.B.SEL3 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseU_LS_P00_2();
    GTM_TOUTSEL1.B.SEL4 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseU_HS_P00_3();

    /* U6. AGC: enable CH5 */
    glbCtrl.B.UPEN_CTRL5    = 0x2U;
    fupdCtrl.B.FUPD_CTRL5   = 0x2U;
    endisCtrl.B.ENDIS_CTRL5 = 0x2U;
    outenCtrl.B.OUTEN_CTRL5 = 0x2U;

    /* ================================================================== */
    /* PHASE V — ATOM0_CH6 → DTM0_CH1 → IL2 P00.4 (LS) + /IH2 P00.5 (HS) */
    /* ================================================================== */

    /* V1. Configure channel control */
    chCtrl.U              = GTM_ATOM0_CH6_CTRL.U;
    chCtrl.B.MODE         = ATOM_MODE_SOMP;
    chCtrl.B.RST_CCU0     = 0x1U;
    chCtrl.B.TRIGOUT      = 0x0U;
    chCtrl.B.ARU_EN       = 0x0U;
    chCtrl.B.CLK_SRC_SR   = 0x0U;
    chCtrl.B.SL           = 0x0U;
    GTM_ATOM0_CH6_CTRL.U  = chCtrl.U;

    /* V2. Shadow registers */
    GTM_ATOM0_CH6_SR0.B.SR0 = Period_Ticks_G;
    GTM_ATOM0_CH6_SR1.B.SR1 = Half_Period_Ticks_G;

    /* V3. DTM0_CH1 dead-time values */
    GTM_CDTM0_DTM0_CH1_DTV.B.RELRISE = CDD_GTM_DTM_DEAD_TIME_TICKS;
    GTM_CDTM0_DTM0_CH1_DTV.B.RELFALL = CDD_GTM_DTM_DEAD_TIME_TICKS;

    /* V4. DTM0_CH1 polarity — CH_CTRL1 shared register, channel index 1
     *     IL2 active HIGH → POL0_1 = 0,  /IH2 active LOW → POL1_1 = 1      */
    GTM_CDTM0_DTM0_CH_CTRL2.B.POL0_1 = 0x0U;  /* IL2  active HIGH (LS, non-inverted) */
    GTM_CDTM0_DTM0_CH_CTRL2.B.POL1_1 = 0x1U;  /* /IH2 active LOW  (HS, inverted)     */

    /* V5. Pin mux: P00.4 = IL2 (DTM0_CH1_OUT), P00.5 = /IH2 (DTM0_CH1_OUT_N) */
    GTM_TOUTSEL1.B.SEL5 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseV_LS_P00_4();
    GTM_TOUTSEL1.B.SEL6 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseV_HS_P00_5();

    /* V6. AGC: enable CH6 */
    glbCtrl.B.UPEN_CTRL6    = 0x2U;
    fupdCtrl.B.FUPD_CTRL6   = 0x2U;
    endisCtrl.B.ENDIS_CTRL6 = 0x2U;
    outenCtrl.B.OUTEN_CTRL6 = 0x2U;

    /* ================================================================== */
    /* PHASE W — ATOM0_CH7 → DTM0_CH2 → IL3 P00.6 (LS) + /IH3 P00.7 (HS) */
    /* ================================================================== */

    /* W1. Configure channel control */
    chCtrl.U              = GTM_ATOM0_CH7_CTRL.U;
    chCtrl.B.MODE         = ATOM_MODE_SOMP;
    chCtrl.B.RST_CCU0     = 0x1U;
    chCtrl.B.TRIGOUT      = 0x0U;
    chCtrl.B.ARU_EN       = 0x0U;
    chCtrl.B.CLK_SRC_SR   = 0x0U;
    chCtrl.B.SL           = 0x0U;
    GTM_ATOM0_CH7_CTRL.U  = chCtrl.U;

    /* W2. Shadow registers */
    GTM_ATOM0_CH7_SR0.B.SR0 = Period_Ticks_G;
    GTM_ATOM0_CH7_SR1.B.SR1 = Half_Period_Ticks_G;

    /* W3. DTM0_CH2 dead-time values */
    GTM_CDTM0_DTM0_CH2_DTV.B.RELRISE = CDD_GTM_DTM_DEAD_TIME_TICKS;
    GTM_CDTM0_DTM0_CH2_DTV.B.RELFALL = CDD_GTM_DTM_DEAD_TIME_TICKS;

    /* W4. DTM0_CH2 polarity — CH_CTRL1 shared register, channel index 2
     *     IL3 active HIGH → POL0_2 = 0,  /IH3 active LOW → POL1_2 = 1      */
    GTM_CDTM0_DTM0_CH_CTRL2.B.POL0_2 = 0x0U;  /* IL3  active HIGH (LS, non-inverted) */
    GTM_CDTM0_DTM0_CH_CTRL2.B.POL1_2 = 0x1U;  /* /IH3 active LOW  (HS, inverted)     */

    /* W5. Pin mux: P00.6 = IL3 (DTM0_CH2_OUT), P00.7 = /IH3 (DTM0_CH2_OUT_N) */
    GTM_TOUTSEL1.B.SEL7 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseW_LS_P00_6();
    GTM_TOUTSEL2.B.SEL0 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseW_HS_P00_7();

    /* W6. AGC: enable CH7 */
    glbCtrl.B.UPEN_CTRL7    = 0x2U;
    fupdCtrl.B.FUPD_CTRL7   = 0x2U;
    endisCtrl.B.ENDIS_CTRL7 = 0x2U;
    outenCtrl.B.OUTEN_CTRL7 = 0x2U;

    /* ================================================================== */
    /* Write back shared interim structs to their control registers        */
    /* ================================================================== */

    /* D1. DTM0 global: CMU CLK0, no shut-off reset
     *     iLLD 1.20.0 DTM_CTRL fields: CLK_SEL, DTM_SEL, UPD_MODE, SHUT_OFF_RST */
    GTM_CDTM0_DTM0_CTRL.B.CLK_SEL      = 0x0U;   /* CMU CLK0              */
    GTM_CDTM0_DTM0_CTRL.B.SHUT_OFF_RST = 0x0U;   /* no shut-off reset     */

    /* A1. Write back accumulated AGC configurations */
    GTM_ATOM0_AGC_GLB_CTRL.U   = glbCtrl.U;
    GTM_ATOM0_AGC_FUPD_CTRL.U  = fupdCtrl.U;
    GTM_ATOM0_AGC_ENDIS_CTRL.U = endisCtrl.U;
    GTM_ATOM0_AGC_OUTEN_CTRL.U = outenCtrl.U;

    /* A2. HOST_TRIG: start all channels synchronously */
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG = 0x1U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GTM_Set_PWM_Duty
 *------------------------------------------------------------------------------------------------------------------*/
void GTM_Set_PWM_Duty(const GTM_PWM_Duty_T * const Duty_Ptr)
{
    uint32_T sr0;
    uint32_T sr1;
    real32_T dc;

    /* ADC triggers — valley-aligned, fixed, independent of duty cycle     */
    GTM_ATOM0_CH3_SR1.U = ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH3_SR0.U = Period_Ticks_G;
    GTM_ATOM0_CH4_SR1.U = ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH4_SR0.U = Period_Ticks_G;

    /* Phase U — SR1 = (1-Dc)*Half, SR0 = (1+Dc)*Half  (centre-aligned)   */
    dc = Duty_Ptr->DutyU;
    if      (dc >= 1.0f)  { sr1 = 0U;            sr0 = Period_Ticks_G + 1U; }
    else if (dc <= 0.0f)  { sr1 = Period_Ticks_G; sr0 = 0U;                 }
    else
    {
        sr1 = (uint32_T)((1.0f - dc) * (real32_T)Half_Period_Ticks_G);
        sr0 = (uint32_T)((1.0f + dc) * (real32_T)Half_Period_Ticks_G);
    }
    GTM_ATOM0_CH5_SR1.U = sr1;
    GTM_ATOM0_CH5_SR0.U = sr0;

    /* Phase V */
    dc = Duty_Ptr->DutyV;
    if      (dc >= 1.0f)  { sr1 = 0U;            sr0 = Period_Ticks_G + 1U; }
    else if (dc <= 0.0f)  { sr1 = Period_Ticks_G; sr0 = 0U;                 }
    else
    {
        sr1 = (uint32_T)((1.0f - dc) * (real32_T)Half_Period_Ticks_G);
        sr0 = (uint32_T)((1.0f + dc) * (real32_T)Half_Period_Ticks_G);
    }
    GTM_ATOM0_CH6_SR1.U = sr1;
    GTM_ATOM0_CH6_SR0.U = sr0;

    /* Phase W */
    dc = Duty_Ptr->DutyW;
    if      (dc >= 1.0f)  { sr1 = 0U;            sr0 = Period_Ticks_G + 1U; }
    else if (dc <= 0.0f)  { sr1 = Period_Ticks_G; sr0 = 0U;                 }
    else
    {
        sr1 = (uint32_T)((1.0f - dc) * (real32_T)Half_Period_Ticks_G);
        sr0 = (uint32_T)((1.0f + dc) * (real32_T)Half_Period_Ticks_G);
    }
    GTM_ATOM0_CH7_SR1.U = sr1;
    GTM_ATOM0_CH7_SR0.U = sr0;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GTM_Get_Period_Ticks
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T GTM_Get_Period_Ticks(void)
{
    return Period_Ticks_G;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GTM_Get_Sample_Time
 *------------------------------------------------------------------------------------------------------------------*/
real32_T GTM_Get_Sample_Time(void)
{
    return Sample_Time_G;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GTM_Enable_PWM_Outputs
 *------------------------------------------------------------------------------------------------------------------*/
void GTM_Enable_PWM_Outputs(void)
{
    Ifx_GTM_ATOM_AGC_ENDIS_CTRL endisCtrl;
    Ifx_GTM_ATOM_AGC_OUTEN_CTRL outenCtrl;

    endisCtrl.U = GTM_ATOM0_AGC_ENDIS_CTRL.U;
    outenCtrl.U = GTM_ATOM0_AGC_OUTEN_CTRL.U;

    endisCtrl.B.ENDIS_CTRL5 = 0x2U;   /* enable CH5 Phase U                */
    endisCtrl.B.ENDIS_CTRL6 = 0x2U;   /* enable CH6 Phase V                */
    endisCtrl.B.ENDIS_CTRL7 = 0x2U;   /* enable CH7 Phase W                */
    outenCtrl.B.OUTEN_CTRL5 = 0x2U;
    outenCtrl.B.OUTEN_CTRL6 = 0x2U;
    outenCtrl.B.OUTEN_CTRL7 = 0x2U;

    GTM_ATOM0_AGC_ENDIS_CTRL.U         = endisCtrl.U;
    GTM_ATOM0_AGC_OUTEN_CTRL.U         = outenCtrl.U;
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG = 0x1U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GTM_Disable_PWM_Outputs
 *------------------------------------------------------------------------------------------------------------------*/
void GTM_Disable_PWM_Outputs(void)
{
    /* Immediate disable — safe to call from ISR */
    GTM_ATOM0_AGC_ENDIS_CTRL.B.ENDIS_CTRL5 = 0x1U;   /* 01b = disable     */
    GTM_ATOM0_AGC_ENDIS_CTRL.B.ENDIS_CTRL6 = 0x1U;
    GTM_ATOM0_AGC_ENDIS_CTRL.B.ENDIS_CTRL7 = 0x1U;
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG     = 0x1U;
}
