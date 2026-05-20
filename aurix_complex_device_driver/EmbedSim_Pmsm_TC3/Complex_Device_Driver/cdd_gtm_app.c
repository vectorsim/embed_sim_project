/**********************************************************************************************************************
 * \file        cdd_gtm_app.c
 * \brief       GTM ATOM0 direct 6-channel driver for 3-phase FOC PWM generation
 *              on the AP32541 motor control board (TC38x).
 *
 * \details     Channel assignment (all TOUTSEL values from TC38x UM appx1):
 *
 *              ATOM0_CH0  Master  — centre-aligned SOMP carrier, CCU1 ISR → CPU
 *                                   TOUTSEL1.SEL1 = 0x02 → CDTM0_DTM4_0 → TOUT9 / P00.0
 *
 *              ATOM0_CH1  Phase U LS — IL1  P00.2  active HIGH  SL = 0
 *                                   TOUTSEL1.SEL3 = 0x02 → CDTM0_DTM4_1 → TOUT11
 *              ATOM0_CH2  Phase U HS — /IH1 P00.3  active LOW   SL = 0
 *                                   TOUTSEL1.SEL4 = 0x02 → CDTM0_DTM4_2 → TOUT12
 *
 *              ATOM0_CH3  Phase V LS — IL2  P00.4  active HIGH  SL = 0
 *                                   TOUTSEL1.SEL5 = 0x02 → CDTM0_DTM4_3 → TOUT13
 *              ATOM0_CH4  Phase V HS — /IH2 P00.5  active LOW   SL = 0
 *                                   TOUTSEL1.SEL6 = 0x02 → CDTM0_DTM5_0 → TOUT14
 *
 *              ATOM0_CH5  Phase W LS — IL3  P00.6  active HIGH  SL = 0
 *                                   TOUTSEL1.SEL7 = 0x02 → CDTM0_DTM5_1 → TOUT15
 *              ATOM0_CH6  Phase W HS — /IH3 P00.7  active LOW   SL = 0
 *                                   TOUTSEL2.SEL0 = 0x02 → CDTM0_DTM5_2 → TOUT16
 *
 *              ATOM0_CH7  ADC trigger — valley-aligned, internal only
 *                                   CCU1 → ADCTRIG0 → EVADC G0/G1/G2 (phase currents)
 *
 *              Dead-time: software, applied symmetrically both edges in GTM_Set_PWM_Duty():
 *                  sr1_hs = (1 - dc) * Half          sr0_hs = (1 + dc) * Half
 *                  sr1_ls = sr1_hs + DT              sr0_ls = sr0_hs - DT
 *
 *              GTM_Set_PWM_Duty() is STATIC — takes duty cycles from CDD_App_G.
 *              GTM_PWM_Duty_T has been eliminated.  All duty values live in CDD_APP_t.
 *
 *              Open-loop block (GTM_OpenLoop_Run / GTM_OpenLoop_Set_RPM / OL_State_G):
 *                  Self-contained in this file.  To replace with FOC:
 *                    1. Remove GTM_OpenLoop_Run() and OL_State_G / GTM_OL_State_T.
 *                    2. Replace the ISR body with EmbedSim_Step() / IPC dispatch.
 *                    3. Remove GTM_OpenLoop_Set_RPM() / GTM_OpenLoop_Stop() from header.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *
 * \version     1.1.0
 * \date        2025-05-18
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_gtm_app.h"
#include "cdd_app.h"           /* CDD_APP_t, CDD_App_G — central state hub              */
#include "cdd_gpio_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "IfxGtm_reg.h"
#include "IfxGtm_Atom.h"
#include "IfxSrc_reg.h"
#include <math.h>              /* sinf()                                                 */

/**********************************************************************************************************************
 * ISR Vector Registration
 *********************************************************************************************************************/

/* CORE_01_ATOM_00_CH_00_CL_SRPN = 80 — literal value required by TASKING IFX_INTERRUPT */
EMBED_SIM_INTERRUPT(GTM_Atom_00_Ch_00_Isr, 0x0U, CORE_01_ATOM_00_CH_00_CL_SRPN);

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  ATOM SOMP mode — master only (self-resets at CM0 = carrier period)    */
#define ATOM_MODE_SOMP              (0x2U)

/** \brief  ATOM up-count mode (CN0 counts upward)                                */
#define ATOM_UD_COUNT_MODE          (0x0U)

/**
 * \brief  SL for HS channels — controlled by CDD_GTM_HS_ACTIVE_LOW (cdd_config.h).
 *
 *         TLE9180D (/IHx active LOW, CDD_GTM_HS_ACTIVE_LOW = 1):
 *             SOMP reset → output = ~SL = 1 → /IH = HIGH → HS gate OFF ✓
 *             ATOM_HS_CH_SL = 0x0U
 *
 *         Standard IHx active HIGH (CDD_GTM_HS_ACTIVE_LOW = 0):
 *             SOMP reset → output = ~SL = 0 → IH = LOW → HS gate OFF ✓
 *             ATOM_HS_CH_SL = 0x1U
 */
#if (CDD_GTM_HS_ACTIVE_LOW != 0U)
    #define ATOM_HS_CH_SL   (0x0U)
#else
    #define ATOM_HS_CH_SL   (0x1U)
#endif

/** \brief  SL for LS channels — ILx active HIGH, SL=0: reset → ~SL=1 → IL HIGH → freewheeling ✓ */
#define ATOM_LS_CH_SL               (0x0U)

/** \brief  TOUTSEL mux value = 0x02 — ATOM0 output through CDTM0                 */
#define TOUTSEL_GTM_ATOM            (0x02U)

/** \brief  Open-loop: 2π/3 radians [rad]                                          */
#define OL_TWO_PI_OVER_THREE        (2.09439510F)

/** \brief  Open-loop: 2π radians [rad]                                            */
#define OL_TWO_PI                   (6.28318530F)

/** \brief  Open-loop: π/30 for RPM → mechanical rad/s conversion [rad/(s·RPM)]   */
#define OL_PI_OVER_30               (0.10471975F)

/**
 * \brief  Motor pole pairs for RPM → electrical rad/s.
 *         DB42S02: p = 4.  Override in cdd_config.h as CDD_MOTOR_POLE_PAIRS.
 */
#ifndef CDD_MOTOR_POLE_PAIRS
    #define CDD_MOTOR_POLE_PAIRS    (4U)
#endif

/**********************************************************************************************************************
 * Private Types
 *********************************************************************************************************************/

/*
 * ============================================================
 * OPEN-LOOP STATE — strip this typedef together with
 * OL_State_G and GTM_OpenLoop_Run() when switching to FOC.
 * ============================================================
 */
/**
 * \brief  Open-loop V/f sinusoidal modulation state.
 *
 * \note   MISRA C:2012 Rule 8.9 deviation: file scope required so both
 *         GTM_OpenLoop_Run(), GTM_OpenLoop_Set_RPM(), and GTM_OpenLoop_Stop()
 *         can access shared mutable state without a global struct in CDD_APP_t
 *         (keeping the open-loop block self-contained and strip-ready).
 */
typedef struct
{
    real32_T    omega_e;    /**< \brief Electrical angular velocity  [rad/s]       */
    real32_T    mi;         /**< \brief Modulation index             [0..1]        */
    real32_T    theta;      /**< \brief Electrical angle accumulator [0..2π rad]   */
    uint32_T    active;     /**< \brief 1 = open-loop step active    [bool]        */
} GTM_OL_State_T;

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/*
 * CDD_App_G.PeriodTicks, CDD_App_G.HalfPeriodTicks, CDD_App_G.SampleTime have moved to
 * CDD_App_G.PeriodTicks / .HalfPeriodTicks / .SampleTime (cdd_app.h).
 */

/*
 * ============================================================
 * OPEN-LOOP STATE INSTANCE — strip together with GTM_OL_State_T
 * and GTM_OpenLoop_Run() when switching to FOC.
 * ============================================================
 */
/* PRQA S 1533 1 -- MISRA C:2012 Rule 8.9: shared across open-loop API and ISR */
static GTM_OL_State_T OL_State_G;

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/

static void GTM_Set_PWM_Duty(const CDD_APP_t * const App_Ptr);

/*
 * ============================================================
 * OPEN-LOOP PROTOTYPE — strip together with implementation below
 * when switching to FOC.
 * ============================================================
 */
static void GTM_OpenLoop_Run(real32_T Omega_Rad_E);

/**********************************************************************************************************************
 * ISR Body — 20 kHz control loop
 *
 * When OL_State_G.active == 1: advances electrical angle and writes SVPWM duties.
 * When OL_State_G.active == 0: ISR is a no-op (safe during startup and after stop).
 *
 * To switch to FOC: replace the GTM_OpenLoop_Run() call with EmbedSim_Step() or
 * IPC dispatch; remove the OL_State_G.active guard.
 *********************************************************************************************************************/
void GTM_Atom_00_Ch_00_Isr(void)
{
    /* Clear CCU1 interrupt flag (half-period match) */
    GTM_ATOM0_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1U;


//    CDD_App_G.DutyU = 0.35f;
//    CDD_App_G.DutyV = 0.45f;
//    CDD_App_G.DutyW = 0.55f;
//    GTM_Set_PWM_Duty(&CDD_App_G);

    /*
     * ============================================================
     * OPEN-LOOP DISPATCH — strip this block when switching to FOC.
     * ============================================================
     */
    if (OL_State_G.active != 0U)
    {
        GTM_OpenLoop_Run(OL_State_G.omega_e);
    }
    /*
     * else: closed-loop FOC — replace with:
     *     EmbedSim_Step(&CDD_App_G);  (writes DutyU/V/W then calls GTM_Set_PWM_Duty)
     */
}

/**********************************************************************************************************************
 * Initialize_GTM_Module
 *
 * Flat Infineon-pattern init.  Shared AGC structs are read once from hardware,
 * accumulated across channel blocks, then written back once.  Per-channel chCtrl
 * is reused.
 *
 * Step labels:
 *   M1..M6  — Master channel    ATOM0_CH0  (carrier + CCU1 ISR, SRE=0 on exit)
 *   U1..U8  — Phase U           CH1 (IL1/P00.2, SL=0) + CH2 (/IH1/P00.3, SL=0)
 *   V1..V8  — Phase V           CH3 (IL2/P00.4, SL=0) + CH4 (/IH2/P00.5, SL=0)
 *   W1..W8  — Phase W           CH5 (IL3/P00.6, SL=0) + CH6 (/IH3/P00.7, SL=0)
 *   T1..T4  — ADC trigger       CH7 (valley → ADCTRIG0 → EVADC G0/G1/G2)
 *   D1..D2  — CDTM0 DTM4+DTM5  passthrough (CLK_SEL + zero DTV)
 *   A1..A3  — AGC write-back + 50% duty init + OL state init
 *********************************************************************************************************************/
void Initialize_GTM_Module(void)
{
    Ifx_GTM_ATOM_CH_CTRL        chCtrl;
    Ifx_GTM_ATOM_CH_IRQ_EN      chIrqEn;
    Ifx_SRC_SRCR                srcCfg;
    Ifx_GTM_ATOM_AGC_GLB_CTRL   glbCtrl;
    Ifx_GTM_ATOM_AGC_FUPD_CTRL  fupdCtrl;
    Ifx_GTM_ATOM_AGC_ENDIS_CTRL endisCtrl;
    Ifx_GTM_ATOM_AGC_OUTEN_CTRL outenCtrl;

    /* ------------------------------------------------------------------ */
    /* Pre-compute timing constants from CMU CLK0 frequency               */
    /* ------------------------------------------------------------------ */
    CDD_App_G.PeriodTicks      = (uint32_T)((real32_T)GTM_CMU_CLK0_FREQUENCY /
                                             (real32_T)CDD_CONTROL_LOOP_FREQUENCY);
    CDD_App_G.HalfPeriodTicks  = CDD_App_G.PeriodTicks / 2U;
    CDD_App_G.SampleTime       = 1.0F / (real32_T)CDD_CONTROL_LOOP_FREQUENCY;

    /* ------------------------------------------------------------------ */
    /* Read shared AGC structs once from hardware                          */
    /* ------------------------------------------------------------------ */
    glbCtrl.U   = GTM_ATOM0_AGC_GLB_CTRL.U;
    fupdCtrl.U  = GTM_ATOM0_AGC_FUPD_CTRL.U;
    endisCtrl.U = GTM_ATOM0_AGC_ENDIS_CTRL.U;
    outenCtrl.U = GTM_ATOM0_AGC_OUTEN_CTRL.U;

    /* ================================================================== */
    /* MASTER — ATOM0_CH0                                                  */
    /* Centre-aligned SOMP carrier.  CCU1 fires at half-period → ISR      */
    /* TOUTSEL1.SEL1 = 0x02 → CDTM0_DTM4_0 → TOUT9 / P00.0              */
    /* ================================================================== */

    /* M1. Channel control */
    chCtrl.U            = GTM_ATOM0_CH0_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x0U;   /* master: expose CCU0 via TRIGOUT to AGC bus    */
    chCtrl.B.TRIGOUT    = 0x1U;   /* CCU0 reset event → AGC slave resets           */
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;   /* CMU CLK0                                      */
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH0_CTRL.U = chCtrl.U;

    /* M2. Shadow registers: period + CCU1 ISR at half-period             */
    GTM_ATOM0_CH0_SR0.B.SR0 = CDD_App_G.PeriodTicks;
    GTM_ATOM0_CH0_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* M3. Enable CCU1 interrupt (CCU0 unused; CCU1 → 20 kHz ISR)        */
    chIrqEn.U               = GTM_ATOM0_CH0_IRQ_EN.U;
    chIrqEn.B.CCU0TC_IRQ_EN = 0x0U;
    chIrqEn.B.CCU1TC_IRQ_EN = 0x1U;
    GTM_ATOM0_CH0_IRQ_EN.U  = chIrqEn.U;
    GTM_ATOM0_CH0_IRQ_MODE.B.IRQ_MODE = 0x0U;   /* pulse mode             */

    /* M4. Service request node: SRE = 0 (armed, not yet enabled)
     *     SRE is set to 1 by Initialize_Pmsm_App() after bridge is live  */
    srcCfg.U      = SRC_GTM_ATOM0_0.U;
    srcCfg.B.SRPN = CORE_01_ATOM_00_CH_00_CL_SRPN;
    srcCfg.B.TOS  = 0x0U;   /* target CPU (matches ISR vector table)      */
    srcCfg.B.CLRR = 0x1U;   /* clear any stale pending request            */
    srcCfg.B.SRE  = 0x0U;   /* NOT enabled here — see Initialize_Pmsm_App */

    /* M5. Pin mux → TOUT9 / P00.0 (carrier probe)                        */
    GTM_TOUTSEL1.B.SEL1 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_Master_P00_0();

    /* M6. AGC: enable CH0                                                */
    glbCtrl.B.UPEN_CTRL0    = 0x2U;
    fupdCtrl.B.FUPD_CTRL0   = 0x2U;
    endisCtrl.B.ENDIS_CTRL0 = 0x2U;
    outenCtrl.B.OUTEN_CTRL0 = 0x2U;

    /* ================================================================== */
    /* PHASE U LS — ATOM0_CH1                                             */
    /* IL1 P00.2  active HIGH  SL = 0                                     */
    /* TOUTSEL1.SEL3 = 0x02 → CDTM0_DTM4_1 → TOUT11                     */
    /* ================================================================== */

    /* U1. Channel control: SOMP slave (RST_CCU0=1, locked to master TRIGOUT) */
    chCtrl.U            = GTM_ATOM0_CH1_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_LS_CH_SL;   /* ~SL=1=HIGH=IL=HIGH=gate ON (freewheeling) ✓ */
    GTM_ATOM0_CH1_CTRL.U = chCtrl.U;

    /* U2. Placeholder SR — overwritten by 50% init below (step A2) */
    GTM_ATOM0_CH1_SR0.B.SR0 = CDD_App_G.HalfPeriodTicks;
    GTM_ATOM0_CH1_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* U3. Pin mux → TOUT11 / P00.2                                       */
    GTM_TOUTSEL1.B.SEL3 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseU_LS_P00_2();

    /* U4. AGC: enable CH1                                                */
    glbCtrl.B.UPEN_CTRL1    = 0x2U;
    fupdCtrl.B.FUPD_CTRL1   = 0x2U;
    endisCtrl.B.ENDIS_CTRL1 = 0x2U;
    outenCtrl.B.OUTEN_CTRL1 = 0x2U;

    /* ================================================================== */
    /* PHASE U HS — ATOM0_CH2                                             */
    /* /IH1 P00.3  active LOW  SL = 0 (TLE9180D, CDD_GTM_HS_ACTIVE_LOW=1) */
    /* TOUTSEL1.SEL4 = 0x02 → CDTM0_DTM4_2 → TOUT12                     */
    /* ================================================================== */

    /* U5. Channel control: SOMP slave (RST_CCU0=1)                          */
    chCtrl.U            = GTM_ATOM0_CH2_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_HS_CH_SL;   /* ~SL=1=HIGH=/IH=HIGH=gate OFF at reset ✓ */
    GTM_ATOM0_CH2_CTRL.U = chCtrl.U;

    /* U6. Placeholder SR — overwritten by 50% init (step A2)             */
    GTM_ATOM0_CH2_SR0.B.SR0 = CDD_App_G.HalfPeriodTicks;
    GTM_ATOM0_CH2_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* U7. Pin mux → TOUT12 / P00.3                                       */
    GTM_TOUTSEL1.B.SEL4 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseU_HS_P00_3();

    /* U8. AGC: enable CH2                                                */
    glbCtrl.B.UPEN_CTRL2    = 0x2U;
    fupdCtrl.B.FUPD_CTRL2   = 0x2U;
    endisCtrl.B.ENDIS_CTRL2 = 0x2U;
    outenCtrl.B.OUTEN_CTRL2 = 0x2U;

    /* ================================================================== */
    /* PHASE V LS — ATOM0_CH3                                             */
    /* IL2 P00.4  active HIGH  SL = 0                                     */
    /* TOUTSEL1.SEL5 = 0x02 → CDTM0_DTM4_3 → TOUT13                     */
    /* ================================================================== */

    /* V1. Channel control: SOMP slave (RST_CCU0=1, locked to master TRIGOUT) */
    chCtrl.U            = GTM_ATOM0_CH3_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH3_CTRL.U = chCtrl.U;

    /* V2. Placeholder SR — overwritten by 50% init (step A2)             */
    GTM_ATOM0_CH3_SR0.B.SR0 = CDD_App_G.HalfPeriodTicks;
    GTM_ATOM0_CH3_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* V3. Pin mux → TOUT13 / P00.4                                       */
    GTM_TOUTSEL1.B.SEL5 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseV_LS_P00_4();

    /* V4. AGC: enable CH3                                                */
    glbCtrl.B.UPEN_CTRL3    = 0x2U;
    fupdCtrl.B.FUPD_CTRL3   = 0x2U;
    endisCtrl.B.ENDIS_CTRL3 = 0x2U;
    outenCtrl.B.OUTEN_CTRL3 = 0x2U;

    /* ================================================================== */
    /* PHASE V HS — ATOM0_CH4                                             */
    /* /IH2 P00.5  active LOW  SL = 0 (TLE9180D)                          */
    /* TOUTSEL1.SEL6 = 0x02 → CDTM0_DTM5_0 → TOUT14                     */
    /* ================================================================== */

    /* V5. Channel control: SOMP slave (RST_CCU0=1)                          */
    chCtrl.U            = GTM_ATOM0_CH4_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH4_CTRL.U = chCtrl.U;

    /* V6. Placeholder SR — overwritten by 50% init (step A2)             */
    GTM_ATOM0_CH4_SR0.B.SR0 = CDD_App_G.HalfPeriodTicks;
    GTM_ATOM0_CH4_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* V7. Pin mux → TOUT14 / P00.5                                       */
    GTM_TOUTSEL1.B.SEL6 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseV_HS_P00_5();

    /* V8. AGC: enable CH4                                                */
    glbCtrl.B.UPEN_CTRL4    = 0x2U;
    fupdCtrl.B.FUPD_CTRL4   = 0x2U;
    endisCtrl.B.ENDIS_CTRL4 = 0x2U;
    outenCtrl.B.OUTEN_CTRL4 = 0x2U;

    /* ================================================================== */
    /* PHASE W LS — ATOM0_CH5                                             */
    /* IL3 P00.6  active HIGH  SL = 0                                     */
    /* TOUTSEL1.SEL7 = 0x02 → CDTM0_DTM5_1 → TOUT15                     */
    /* ================================================================== */

    /* W1. Channel control: SOMP slave (RST_CCU0=1, locked to master TRIGOUT) */
    chCtrl.U            = GTM_ATOM0_CH5_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH5_CTRL.U = chCtrl.U;

    /* W2. Placeholder SR — overwritten by 50% init (step A2)             */
    GTM_ATOM0_CH5_SR0.B.SR0 = CDD_App_G.HalfPeriodTicks;
    GTM_ATOM0_CH5_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* W3. Pin mux → TOUT15 / P00.6                                       */
    GTM_TOUTSEL1.B.SEL7 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseW_LS_P00_6();

    /* W4. AGC: enable CH5                                                */
    glbCtrl.B.UPEN_CTRL5    = 0x2U;
    fupdCtrl.B.FUPD_CTRL5   = 0x2U;
    endisCtrl.B.ENDIS_CTRL5 = 0x2U;
    outenCtrl.B.OUTEN_CTRL5 = 0x2U;

    /* ================================================================== */
    /* PHASE W HS — ATOM0_CH6                                             */
    /* /IH3 P00.7  active LOW  SL = 0 (TLE9180D)                          */
    /* TOUTSEL2.SEL0 = 0x02 → CDTM0_DTM5_2 → TOUT16                     */
    /* ================================================================== */

    /* W5. Channel control: SOMP slave (RST_CCU0=1)                          */
    chCtrl.U            = GTM_ATOM0_CH6_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH6_CTRL.U = chCtrl.U;

    /* W6. Placeholder SR — overwritten by 50% init (step A2)             */
    GTM_ATOM0_CH6_SR0.B.SR0 = CDD_App_G.HalfPeriodTicks;
    GTM_ATOM0_CH6_SR1.B.SR1 = CDD_App_G.HalfPeriodTicks;

    /* W7. Pin mux → TOUT16 / P00.7                                       */
    GTM_TOUTSEL2.B.SEL0 = TOUTSEL_GTM_ATOM;
    GPIO_Configure_GTM_PhaseW_HS_P00_7();

    /* W8. AGC: enable CH6                                                */
    glbCtrl.B.UPEN_CTRL6    = 0x2U;
    fupdCtrl.B.FUPD_CTRL6   = 0x2U;
    endisCtrl.B.ENDIS_CTRL6 = 0x2U;
    outenCtrl.B.OUTEN_CTRL6 = 0x2U;

    /* ================================================================== */
    /* ADC TRIGGER — ATOM0_CH7                                            */
    /* Valley-aligned: CCU1 fires CDD_GTM_ADC_VALLEY_OFFSET_TICKS after  */
    /* CCU0 reset.  Internal only — no physical output pin.               */
    /* ================================================================== */

    /* T1. Channel control: SOMP slave, ADC trigger (no output pin)          */
    chCtrl.U            = GTM_ATOM0_CH7_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH7_CTRL.U = chCtrl.U;

    /* T2. CCU1 fires CDD_GTM_ADC_VALLEY_OFFSET_TICKS after valley       */
    GTM_ATOM0_CH7_SR0.B.SR0 = CDD_App_G.PeriodTicks;
    GTM_ATOM0_CH7_SR1.B.SR1 = CDD_GTM_ADC_VALLEY_OFFSET_TICKS;

    /* T3. Route CH7 CCU1 → ADCTRIG0 → EVADC G0/G1/G2
     *     SEL = 0x8 = CDTM0_DTM5_3 / ATOM0_CH7 (TC38x appx1 p.769)    */
    GTM_ADCTRIG0OUT0.B.SEL0 = 0x8U;   /* Phase U → EVADC G0             */
    GTM_ADCTRIG0OUT0.B.SEL1 = 0x8U;   /* Phase V → EVADC G1             */
    GTM_ADCTRIG0OUT0.B.SEL2 = 0x8U;   /* Phase W → EVADC G2             */

    /* T4. AGC: enable CH7                                                */
    glbCtrl.B.UPEN_CTRL7    = 0x2U;
    fupdCtrl.B.FUPD_CTRL7   = 0x2U;
    endisCtrl.B.ENDIS_CTRL7 = 0x2U;
    outenCtrl.B.OUTEN_CTRL7 = 0x2U;

    /* ================================================================== */
    /* CDTM0 DTM4 — passthrough (ATOM0 CH0–CH3 → TOUT9/11/12/13)         */
    /* ================================================================== */

    /* D1. DTM4: CMU CLK0, no shut-off reset                              */
    GTM_CDTM0_DTM4_CTRL.B.CLK_SEL      = 0x0U;
    GTM_CDTM0_DTM4_CTRL.B.SHUT_OFF_RST = 0x0U;

    /* ================================================================== */
    /* CDTM0 DTM5 — passthrough (ATOM0 CH4–CH7 → TOUT14/15/16)           */
    /* ================================================================== */

    /* D2. DTM5: CMU CLK0, no shut-off reset                              */
    GTM_CDTM0_DTM5_CTRL.B.CLK_SEL      = 0x0U;
    GTM_CDTM0_DTM5_CTRL.B.SHUT_OFF_RST = 0x0U;

    /* ================================================================== */
    /* A1. Write back accumulated AGC structs                             */
    /* ================================================================== */
    GTM_ATOM0_AGC_GLB_CTRL.U   = glbCtrl.U;
    GTM_ATOM0_AGC_FUPD_CTRL.U  = fupdCtrl.U;
    GTM_ATOM0_AGC_ENDIS_CTRL.U = endisCtrl.U;
    GTM_ATOM0_AGC_OUTEN_CTRL.U = outenCtrl.U;

    /* ================================================================== */
    /* A2. Write 50% duty via CDD_App_G before HOST_TRIG.                 */
    /*                                                                    */
    /* Duty fields were pre-initialised to 0.5F in CDD_App_G's            */
    /* definition in cdd_app.c.  Write them to the shadow registers now   */
    /* so HOST_TRIG transfers 50% (zero-voltage vector) to all channels   */
    /* simultaneously on the first carrier cycle.                         */
    /* ================================================================== */
    GTM_Set_PWM_Duty(&CDD_App_G);

    /* ================================================================== */
    /* A3. Open-loop state initialisation                                 */
    /*     (strip together with GTM_OL_State_T and GTM_OpenLoop_Run)     */
    /* ================================================================== */
    OL_State_G.omega_e = 0.0F;
    OL_State_G.mi      = 0.0F;
    OL_State_G.theta   = 0.0F;
    OL_State_G.active  = 0U;

    /* ================================================================== */
    /* A4. Write SRC node (SRE=0 — ISR enabled later by Initialize_Pmsm_App) */
    /* ================================================================== */
    SRC_GTM_ATOM0_0.U = srcCfg.U;
}

/**********************************************************************************************************************
 * Start_GTM_Module
 *
 * Issues HOST_TRIG: transfers all shadow registers (including 50% duty) to
 * active compare registers and starts the ATOM0 carrier.  PWM is live after this.
 *********************************************************************************************************************/
void Start_GTM_Module(void)
{
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG = 0x1U;
    SRC_GTM_ATOM0_0.B.SRE = 0x1U;
}

/**********************************************************************************************************************
 * GTM_Set_PWM_Duty  [STATIC — internal use only]
 *
 * Reads DutyU/V/W from App_Ptr (i.e. CDD_App_G) and writes ATOM0 CH1–CH7
 * shadow registers with symmetric software dead-time.
 *
 * HS (SL=0, TLE9180D /IHx active LOW):
 *     reset / CM0 → output = SL = 0 → /IH = LOW  → gate ON
 *     CM1         → output = ~SL= 1 → /IH = HIGH → gate OFF
 *     Gate ON  window: [0, SR1_hs] and [SR0_hs, Period]
 *     SR1_hs = (1 - dc) * Half
 *     SR0_hs = (1 + dc) * Half
 *
 * LS (SL=0, ILx active HIGH):
 *     reset / CM0 → output = SL = 0 → IL = LOW  → gate OFF
 *     CM1         → output = ~SL= 1 → IL = HIGH → gate ON (freewheeling)
 *     Gate ON window: [SR1_ls, SR0_ls]
 *     SR1_ls = SR1_hs + DT   (LS on after HS off — rising  dead-time guard)
 *     SR0_ls = SR0_hs - DT   (LS off before HS on — falling dead-time guard)
 *********************************************************************************************************************/
static void GTM_Set_PWM_Duty(const CDD_APP_t * const App_Ptr)
{
    uint32_T    sr1_hs;
    uint32_T    sr0_hs;
    uint32_T    sr1_ls;
    uint32_T    sr0_ls;
    real32_T    dc;
    const uint32_T dt = CDD_GTM_SW_DEAD_TIME_TICKS;

    /* ADC valley trigger — constant offset from carrier reset             */
    GTM_ATOM0_CH7_SR1.U = CDD_GTM_ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH7_SR0.U = CDD_App_G.PeriodTicks;

    /* ------------------------------------------------------------------ */
    /* Phase U                                                            */
    /* ------------------------------------------------------------------ */
    dc = App_Ptr->DutyU;
    if (dc >= 1.0F)
    {
        sr1_hs = 0U;
        sr0_hs = CDD_App_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CDD_App_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        sr1_hs = CDD_App_G.PeriodTicks;
        sr0_hs = CDD_App_G.PeriodTicks;
        sr1_ls = CDD_App_G.PeriodTicks;
        sr0_ls = CDD_App_G.PeriodTicks;
    }
    else
    {
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CDD_App_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CDD_App_G.HalfPeriodTicks);
        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            sr1_ls = CDD_App_G.HalfPeriodTicks;
            sr0_ls = CDD_App_G.HalfPeriodTicks;
        }
    }
    GTM_ATOM0_CH1_SR1.U = sr1_ls;   /* U_LS: CH1  IL1  / P00.2 */
    GTM_ATOM0_CH1_SR0.U = sr0_ls;
    GTM_ATOM0_CH2_SR1.U = sr1_hs;   /* U_HS: CH2  /IH1 / P00.3 */
    GTM_ATOM0_CH2_SR0.U = sr0_hs;

    /* ------------------------------------------------------------------ */
    /* Phase V                                                            */
    /* ------------------------------------------------------------------ */
    dc = App_Ptr->DutyV;
    if (dc >= 1.0F)
    {
        sr1_hs = 0U;
        sr0_hs = CDD_App_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CDD_App_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        sr1_hs = CDD_App_G.PeriodTicks;
        sr0_hs = CDD_App_G.PeriodTicks;
        sr1_ls = CDD_App_G.PeriodTicks;
        sr0_ls = CDD_App_G.PeriodTicks;
    }
    else
    {
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CDD_App_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CDD_App_G.HalfPeriodTicks);
        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            sr1_ls = CDD_App_G.HalfPeriodTicks;
            sr0_ls = CDD_App_G.HalfPeriodTicks;
        }
    }
    GTM_ATOM0_CH3_SR1.U = sr1_ls;   /* V_LS: CH3  IL2  / P00.4 */
    GTM_ATOM0_CH3_SR0.U = sr0_ls;
    GTM_ATOM0_CH4_SR1.U = sr1_hs;   /* V_HS: CH4  /IH2 / P00.5 */
    GTM_ATOM0_CH4_SR0.U = sr0_hs;

    /* ------------------------------------------------------------------ */
    /* Phase W                                                            */
    /* ------------------------------------------------------------------ */
    dc = App_Ptr->DutyW;
    if (dc >= 1.0F)
    {
        sr1_hs = 0U;
        sr0_hs = CDD_App_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CDD_App_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        sr1_hs = CDD_App_G.PeriodTicks;
        sr0_hs = CDD_App_G.PeriodTicks;
        sr1_ls = CDD_App_G.PeriodTicks;
        sr0_ls = CDD_App_G.PeriodTicks;
    }
    else
    {
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CDD_App_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CDD_App_G.HalfPeriodTicks);
        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            sr1_ls = CDD_App_G.HalfPeriodTicks;
            sr0_ls = CDD_App_G.HalfPeriodTicks;
        }
    }
    GTM_ATOM0_CH5_SR1.U = sr1_ls;   /* W_LS: CH5  IL3  / P00.6 */
    GTM_ATOM0_CH5_SR0.U = sr0_ls;
    GTM_ATOM0_CH6_SR1.U = sr1_hs;   /* W_HS: CH6  /IH3 / P00.7 */
    GTM_ATOM0_CH6_SR0.U = sr0_hs;
}

/**********************************************************************************************************************
 * GTM_OpenLoop_Run  [STATIC — called from ISR only]
 *
 * Sinusoidal modulation + SVPWM zero-sequence injection.
 * Advances OL_State_G.theta by Omega_Rad_E * Ts each ISR tick.
 * Writes computed duty cycles to CDD_App_G.DutyU/V/W then calls GTM_Set_PWM_Duty().
 *
 * STRIP THIS ENTIRE FUNCTION (and OL_State_G, GTM_OL_State_T) when replacing
 * with closed-loop FOC.
 *********************************************************************************************************************/
static void GTM_OpenLoop_Run(real32_T Omega_Rad_E)
{
    real32_T    sinU;
    real32_T    sinV;
    real32_T    sinW;
    real32_T    vmax;
    real32_T    vmin;
    real32_T    v_zero;
    real32_T    mi;

    /* I1. Advance electrical angle — wrap to [0, 2π) */
    OL_State_G.theta += Omega_Rad_E * CDD_App_G.SampleTime;
    if (OL_State_G.theta >= OL_TWO_PI)
    {
        OL_State_G.theta -= OL_TWO_PI;
    }
    if (OL_State_G.theta < 0.0F)
    {
        OL_State_G.theta += OL_TWO_PI;
    }

    mi = OL_State_G.mi;

    /* I2. Three-phase sinusoidal reference voltages (peak amplitude = mi) */
    sinU = mi * sinf(OL_State_G.theta);
    sinV = mi * sinf(OL_State_G.theta - OL_TWO_PI_OVER_THREE);
    sinW = mi * sinf(OL_State_G.theta + OL_TWO_PI_OVER_THREE);

    /* I3. SVPWM zero-sequence injection: v0 = -0.5*(max + min)
     *     Extends linear modulation range from mi=0.5 (SPWM) to mi≈0.577  */
    vmax = sinU;
    if (sinV > vmax) { vmax = sinV; }
    if (sinW > vmax) { vmax = sinW; }
    vmin = sinU;
    if (sinV < vmin) { vmin = sinV; }
    if (sinW < vmin) { vmin = sinW; }
    v_zero = -0.5F * (vmax + vmin);

    /* I4. Map [-1..+1] normalised voltage → [0..1] duty cycle.
     *     Write to central state — GTM_Set_PWM_Duty reads from here.     */
    CDD_App_G.DutyU = 0.5F + 0.5F * (sinU + v_zero);
    CDD_App_G.DutyV = 0.5F + 0.5F * (sinV + v_zero);
    CDD_App_G.DutyW = 0.5F + 0.5F * (sinW + v_zero);

    /* I5. Write shadow registers — take effect at next carrier period reset */
    GTM_Set_PWM_Duty(&CDD_App_G);
}

/**********************************************************************************************************************
 * GTM_OpenLoop_Set_RPM  [PUBLIC — strip from header when switching to FOC]
 *********************************************************************************************************************/
void GTM_OpenLoop_Set_RPM(uint32_T Rpm, real32_T Mi)
{
    /*
     * omega_mechanical = Rpm * π/30  [rad/s]
     * omega_electrical  = omega_mechanical * p  [rad_e/s]
     */
    OL_State_G.omega_e = (real32_T)Rpm * OL_PI_OVER_30 * (real32_T)CDD_MOTOR_POLE_PAIRS;
    OL_State_G.mi      = Mi;
    OL_State_G.theta   = 0.0F;
    OL_State_G.active  = 1U;
}

/**********************************************************************************************************************
 * GTM_OpenLoop_Stop  [PUBLIC — strip from header when switching to FOC]
 *********************************************************************************************************************/
void GTM_OpenLoop_Stop(void)
{
    OL_State_G.active  = 0U;
    OL_State_G.omega_e = 0.0F;
    OL_State_G.mi      = 0.0F;
    OL_State_G.theta   = 0.0F;

    /* Return all phases to 50% (zero-voltage vector) */
    CDD_App_G.DutyU = 0.5F;
    CDD_App_G.DutyV = 0.5F;
    CDD_App_G.DutyW = 0.5F;
    GTM_Set_PWM_Duty(&CDD_App_G);
}

/**********************************************************************************************************************
 * GTM_Get_Period_Ticks
 *********************************************************************************************************************/
uint32_T GTM_Get_Period_Ticks(void)
{
    return CDD_App_G.PeriodTicks;
}

/**********************************************************************************************************************
 * GTM_Get_Sample_Time
 *********************************************************************************************************************/
real32_T GTM_Get_Sample_Time(void)
{
    return CDD_App_G.SampleTime;
}
