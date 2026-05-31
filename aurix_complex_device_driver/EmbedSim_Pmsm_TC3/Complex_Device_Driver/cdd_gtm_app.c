/**********************************************************************************************************************
 * \file        cdd_gtm_app.c
 * \brief       GTM ATOM0 direct 6-channel driver for 3-phase FOC PWM generation
 *              on the AP32541 motor control board (TC38x).
 *
 * \details     Channel assignment (TOUTSEL values from TC38x UM appx1):
 *              ATOM0_CH0  Master — centre-aligned SOMP carrier, CCU1 ISR → CPU
 *              ATOM0_CH1  Phase U LS  IL1  P00.2  active HIGH  SL=0
 *              ATOM0_CH2  Phase U HS /IH1  P00.3  active LOW   SL=0
 *              ATOM0_CH3  Phase V LS  IL2  P00.4  active HIGH  SL=0
 *              ATOM0_CH4  Phase V HS /IH2  P00.5  active LOW   SL=0
 *              ATOM0_CH5  Phase W LS  IL3  P00.6  active HIGH  SL=0
 *              ATOM0_CH6  Phase W HS /IH3  P00.7  active LOW   SL=0
 *              ATOM0_CH7  ADC trigger — valley-aligned, internal only
 *
 *              Dead-time: software, applied symmetrically both edges in CddGtm_SetPwmDuty():
 *                  sr1_hs = (1 - dc) * Half     sr0_hs = (1 + dc) * Half
 *                  sr1_ls = sr1_hs + DT          sr0_ls = sr0_hs - DT
 *
 *              CddGtm_SetPwmDuty() is STATIC — reads duty from CddApp_G.
 *              Open-loop block (CddGtm_OpenLoopRun / CddGtm_OpenLoopSetRpm / OL_State_G)
 *              is self-contained.  To switch to FOC: remove CddGtm_OpenLoopRun(),
 *              OL_State_G, CddGtm_OlState_T.  Replace ISR body with EmbedSim_Step().
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All if-conditions use explicit comparison
 *              - Rule 15.5  : Single exit point per function
 *              - Rule 17.2  : No recursion
 *
 * \version     1.2.0
 * \date        2025-05-24
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
#include "cdd_app.h"           /* CddApp_T, CddApp_G — central state hub              */
#include "cdd_gpio_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "IfxGtm_reg.h"
#include "IfxGtm_Atom.h"
#include "IfxSrc_reg.h"
#include <math.h>              /* sinf()                                               */

/**********************************************************************************************************************
 * ISR Vector Registration
 *********************************************************************************************************************/

/* CORE_01_ATOM_00_CH_00_CL_SRPN = 80 — literal value required by TASKING IFX_INTERRUPT */
EMBED_SIM_INTERRUPT(GTM_Atom_00_Ch_00_Isr, 0x0U, CORE_00_ATOM_00_CH_00_CL_SRPN);

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  ATOM SOMP mode — master only (self-resets at CM0 = carrier period)    */
#define ATOM_MODE_SOMP              (0x2U)

/** \brief  ATOM up-count mode (CN0 counts upward)                                */
#define ATOM_UD_COUNT_MODE          (0x0U)

/**
 * \brief  SL for HS channels — controlled by CDD_GTM_HS_ACTIVE_LOW (cdd_config.h).
 */
#if (CDD_GTM_HS_ACTIVE_LOW != 0U)
    #define ATOM_HS_CH_SL   (0x0U)
#else
    #define ATOM_HS_CH_SL   (0x1U)
#endif

/** \brief  SL for LS channels — ILx active HIGH, SL=0: reset → ~SL=1 → IL HIGH → freewheeling */
#define ATOM_LS_CH_SL               (0x0U)

/** \brief  TOUTSEL mux value = 0x02 — ATOM0 output through CDTM0               */
#define TOUTSEL_GTM_ATOM            (0x02U)

/** \brief  Open-loop: 2π/3 radians  [rad]                                       */
#define OL_TWO_PI_OVER_THREE        (2.09439510F)

/** \brief  Open-loop: 2π radians  [rad]                                          */
#define OL_TWO_PI                   (6.28318530F)

/** \brief  Open-loop: π/30 for RPM → mechanical rad/s  [rad/(s·RPM)]            */
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
 * OL_State_G and CddGtm_OpenLoopRun() when switching to FOC.
 * ============================================================
 */
/**
 * \brief  Open-loop V/f sinusoidal modulation state.
 *
 * \note   MISRA C:2012 Rule 8.9 deviation: file scope required so that
 *         CddGtm_OpenLoopRun(), CddGtm_OpenLoopSetRpm(), and CddGtm_OpenLoopStop()
 *         can access shared mutable state without polluting CddApp_T.
 */
typedef struct
{
    real32_T    omega_e;    /**< \brief Electrical angular velocity  [rad/s]     */
    real32_T    mi;         /**< \brief Modulation index             [0..1]      */
    real32_T    theta;      /**< \brief Electrical angle accumulator [0..2π rad] */
    uint32_T    active;     /**< \brief 1 = open-loop step active    [boolean]   */
} CddGtm_OlState_T;

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/*
 * ============================================================
 * OPEN-LOOP STATE INSTANCE — strip with CddGtm_OlState_T and
 * CddGtm_OpenLoopRun() when switching to FOC.
 * ============================================================
 */
/* PRQA S 1533 1 -- MISRA C:2012 Rule 8.9: shared across open-loop API and ISR */
STATIC CddGtm_OlState_T OL_State_G;

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/

STATIC void CddGtm_SetPwmDuty(P2CONST(CddApp_T, AUTOMATIC, CDD_APPL_DATA) AppPtr);

/*
 * ============================================================
 * OPEN-LOOP PROTOTYPE — strip with implementation below
 * when switching to FOC.
 * ============================================================
 */
STATIC void CddGtm_OpenLoopRun(real32_T OmegaRadE);

/**********************************************************************************************************************
 * ISR Body — 20 kHz control loop
 *
 * When OL_State_G.active == 1: advances electrical angle and writes SVPWM duties.
 * When OL_State_G.active == 0: ISR is a no-op (safe during startup and after stop).
 *
 * To switch to FOC: replace the CddGtm_OpenLoopRun() call with EmbedSim_Step() or
 * IPC dispatch; remove the OL_State_G.active guard.
 *********************************************************************************************************************/
void GTM_Atom_00_Ch_00_Isr(void)
{
    /* Clear CCU1 interrupt flag (half-period match) */
    GTM_ATOM0_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1U;

    /*
     * ============================================================
     * OPEN-LOOP DISPATCH — strip this block when switching to FOC.
     * ============================================================
     */
    if (OL_State_G.active != 0U)
    {
        CddGtm_OpenLoopRun(OL_State_G.omega_e);
    }


    /*
     * else: closed-loop FOC — replace with:
     *     EmbedSim_Step(&CddApp_G);
     */
}

/**********************************************************************************************************************
 * CddGtm_Init
 *********************************************************************************************************************/
void CddGtm_Init(void)
{
    Ifx_GTM_ATOM_CH_CTRL        chCtrl;
    Ifx_GTM_ATOM_CH_IRQ_EN      chIrqEn;
    Ifx_SRC_SRCR                srcCfg;
    Ifx_GTM_ATOM_AGC_GLB_CTRL   glbCtrl;
    Ifx_GTM_ATOM_AGC_FUPD_CTRL  fupdCtrl;
    Ifx_GTM_ATOM_AGC_ENDIS_CTRL endisCtrl;
    Ifx_GTM_ATOM_AGC_OUTEN_CTRL outenCtrl;

    /* Pre-compute timing constants from CMU CLK0 frequency */
    CddApp_G.PeriodTicks     = (uint32_T)((real32_T)GTM_CMU_CLK0_FREQUENCY /
                                           (real32_T)CDD_CONTROL_LOOP_FREQUENCY);
    CddApp_G.HalfPeriodTicks = CddApp_G.PeriodTicks / 2U;
    CddApp_G.SampleTime      = 1.0F / (real32_T)CDD_CONTROL_LOOP_FREQUENCY;

    /* Read shared AGC structs once from hardware */
    glbCtrl.U   = GTM_ATOM0_AGC_GLB_CTRL.U;
    fupdCtrl.U  = GTM_ATOM0_AGC_FUPD_CTRL.U;
    endisCtrl.U = GTM_ATOM0_AGC_ENDIS_CTRL.U;
    outenCtrl.U = GTM_ATOM0_AGC_OUTEN_CTRL.U;

    /* ================================================================== */
    /* MASTER — ATOM0_CH0                                                  */
    /* ================================================================== */

    /* M1. Channel control */
    chCtrl.U            = GTM_ATOM0_CH0_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x0U;
    chCtrl.B.TRIGOUT    = 0x1U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH0_CTRL.U = chCtrl.U;

    /* M2. Shadow registers: period + CCU1 ISR at half-period */
    GTM_ATOM0_CH0_SR0.B.SR0 = CddApp_G.PeriodTicks;
    GTM_ATOM0_CH0_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;

    /* M3. Enable CCU1 interrupt */
    chIrqEn.U               = GTM_ATOM0_CH0_IRQ_EN.U;
    chIrqEn.B.CCU0TC_IRQ_EN = 0x0U;
    chIrqEn.B.CCU1TC_IRQ_EN = 0x1U;
    GTM_ATOM0_CH0_IRQ_EN.U  = chIrqEn.U;
    GTM_ATOM0_CH0_IRQ_MODE.B.IRQ_MODE = 0x0U;

    /* M4. Service request node: SRE=0 (not yet enabled) */
    srcCfg.U      = SRC_GTM_ATOM0_0.U;
    srcCfg.B.SRPN = CORE_00_ATOM_00_CH_00_CL_SRPN;
    srcCfg.B.TOS  = 0x0U;
    srcCfg.B.CLRR = 0x1U;
    srcCfg.B.SRE  = 0x0U;

    /* M5. Pin mux → TOUT9 / P00.0 */
    GTM_TOUTSEL1.B.SEL1 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmMaster_P00_0();

    /* M6. AGC: enable CH0 */
    glbCtrl.B.UPEN_CTRL0    = 0x2U;
    fupdCtrl.B.FUPD_CTRL0   = 0x2U;
    endisCtrl.B.ENDIS_CTRL0 = 0x2U;
    outenCtrl.B.OUTEN_CTRL0 = 0x2U;

    /* ================================================================== */
    /* PHASE U LS — ATOM0_CH1  IL1 P00.2  active HIGH  SL=0              */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH1_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH1_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH1_SR0.B.SR0 = CddApp_G.HalfPeriodTicks;
    GTM_ATOM0_CH1_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;
    GTM_TOUTSEL1.B.SEL3 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseULs_P00_2();
    glbCtrl.B.UPEN_CTRL1    = 0x2U;
    fupdCtrl.B.FUPD_CTRL1   = 0x2U;
    endisCtrl.B.ENDIS_CTRL1 = 0x2U;
    outenCtrl.B.OUTEN_CTRL1 = 0x2U;

    /* ================================================================== */
    /* PHASE U HS — ATOM0_CH2  /IH1 P00.3  active LOW  SL=0              */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH2_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH2_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH2_SR0.B.SR0 = CddApp_G.HalfPeriodTicks;
    GTM_ATOM0_CH2_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;
    GTM_TOUTSEL1.B.SEL4 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseUHs_P00_3();
    glbCtrl.B.UPEN_CTRL2    = 0x2U;
    fupdCtrl.B.FUPD_CTRL2   = 0x2U;
    endisCtrl.B.ENDIS_CTRL2 = 0x2U;
    outenCtrl.B.OUTEN_CTRL2 = 0x2U;

    /* ================================================================== */
    /* PHASE V LS — ATOM0_CH3  IL2 P00.4  active HIGH  SL=0              */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH3_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH3_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH3_SR0.B.SR0 = CddApp_G.HalfPeriodTicks;
    GTM_ATOM0_CH3_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;
    GTM_TOUTSEL1.B.SEL5 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseVLs_P00_4();
    glbCtrl.B.UPEN_CTRL3    = 0x2U;
    fupdCtrl.B.FUPD_CTRL3   = 0x2U;
    endisCtrl.B.ENDIS_CTRL3 = 0x2U;
    outenCtrl.B.OUTEN_CTRL3 = 0x2U;

    /* ================================================================== */
    /* PHASE V HS — ATOM0_CH4  /IH2 P00.5  active LOW  SL=0              */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH4_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH4_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH4_SR0.B.SR0 = CddApp_G.HalfPeriodTicks;
    GTM_ATOM0_CH4_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;
    GTM_TOUTSEL1.B.SEL6 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseVHs_P00_5();
    glbCtrl.B.UPEN_CTRL4    = 0x2U;
    fupdCtrl.B.FUPD_CTRL4   = 0x2U;
    endisCtrl.B.ENDIS_CTRL4 = 0x2U;
    outenCtrl.B.OUTEN_CTRL4 = 0x2U;

    /* ================================================================== */
    /* PHASE W LS — ATOM0_CH5  IL3 P00.6  active HIGH  SL=0              */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH5_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH5_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH5_SR0.B.SR0 = CddApp_G.HalfPeriodTicks;
    GTM_ATOM0_CH5_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;
    GTM_TOUTSEL1.B.SEL7 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseWLs_P00_6();
    glbCtrl.B.UPEN_CTRL5    = 0x2U;
    fupdCtrl.B.FUPD_CTRL5   = 0x2U;
    endisCtrl.B.ENDIS_CTRL5 = 0x2U;
    outenCtrl.B.OUTEN_CTRL5 = 0x2U;

    /* ================================================================== */
    /* PHASE W HS — ATOM0_CH6  /IH3 P00.7  active LOW  SL=0              */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH6_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH6_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH6_SR0.B.SR0 = CddApp_G.HalfPeriodTicks;
    GTM_ATOM0_CH6_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;
    GTM_TOUTSEL2.B.SEL0 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseWHs_P00_7();
    glbCtrl.B.UPEN_CTRL6    = 0x2U;
    fupdCtrl.B.FUPD_CTRL6   = 0x2U;
    endisCtrl.B.ENDIS_CTRL6 = 0x2U;
    outenCtrl.B.OUTEN_CTRL6 = 0x2U;

    /* ================================================================== */
    /* ADC TRIGGER — ATOM0_CH7  valley-aligned, internal only             */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH7_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = 0x0U;
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH7_CTRL.U = chCtrl.U;
    GTM_ATOM0_CH7_SR0.B.SR0 = CddApp_G.PeriodTicks;
    GTM_ATOM0_CH7_SR1.B.SR1 = CDD_GTM_ADC_VALLEY_OFFSET_TICKS;
    GTM_ADCTRIG0OUT0.B.SEL0  = 0x8U;
    GTM_ADCTRIG0OUT0.B.SEL1  = 0x8U;
    GTM_ADCTRIG0OUT0.B.SEL2  = 0x8U;
    glbCtrl.B.UPEN_CTRL7    = 0x2U;
    fupdCtrl.B.FUPD_CTRL7   = 0x2U;
    endisCtrl.B.ENDIS_CTRL7 = 0x2U;
    outenCtrl.B.OUTEN_CTRL7 = 0x2U;

    /* ================================================================== */
    /* CDTM0 DTM4 + DTM5 — passthrough, CMU CLK0                         */
    /* ================================================================== */
    GTM_CDTM0_DTM4_CTRL.B.CLK_SEL      = 0x0U;
    GTM_CDTM0_DTM4_CTRL.B.SHUT_OFF_RST = 0x0U;
    GTM_CDTM0_DTM5_CTRL.B.CLK_SEL      = 0x0U;
    GTM_CDTM0_DTM5_CTRL.B.SHUT_OFF_RST = 0x0U;

    /* A1. Write back accumulated AGC structs */
    GTM_ATOM0_AGC_GLB_CTRL.U   = glbCtrl.U;
    GTM_ATOM0_AGC_FUPD_CTRL.U  = fupdCtrl.U;
    GTM_ATOM0_AGC_ENDIS_CTRL.U = endisCtrl.U;
    GTM_ATOM0_AGC_OUTEN_CTRL.U = outenCtrl.U;

    /* A2. Write 50% duty via CddApp_G before HOST_TRIG */
    CddGtm_SetPwmDuty(&CddApp_G);

    /* A3. Open-loop state initialisation */
    OL_State_G.omega_e = 0.0F;
    OL_State_G.mi      = 0.0F;
    OL_State_G.theta   = 0.0F;
    OL_State_G.active  = 0U;

    /* A4. Write SRC node (SRE=0 — ISR enabled later by CddApp_Init) */
    SRC_GTM_ATOM0_0.U = srcCfg.U;
}

/**********************************************************************************************************************
 * CddGtm_Start
 *********************************************************************************************************************/
void CddGtm_Start(void)
{
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG = 0x1U;
    SRC_GTM_ATOM0_0.B.SRE              = 0x1U;
}

/**********************************************************************************************************************
 * CddGtm_SetPwmDuty  [STATIC — internal use only]
 *
 * Reads DutyU/V/W from AppPtr (i.e. CddApp_G) and writes ATOM0 CH1–CH7
 * shadow registers with symmetric software dead-time.
 *********************************************************************************************************************/
STATIC void CddGtm_SetPwmDuty(P2CONST(CddApp_T, AUTOMATIC, CDD_APPL_DATA) AppPtr)
{
    uint32_T       sr1_hs;
    uint32_T       sr0_hs;
    uint32_T       sr1_ls;
    uint32_T       sr0_ls;
    real32_T       dc;
    const uint32_T dt = CDD_GTM_SW_DEAD_TIME_TICKS;

    /* ADC valley trigger — constant offset from carrier reset */
    GTM_ATOM0_CH7_SR1.U = CDD_GTM_ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH7_SR0.U = CddApp_G.PeriodTicks;

    /* ------------------------------------------------------------------ */
    /* Phase U                                                             */
    /* ------------------------------------------------------------------ */
    dc = AppPtr->DutyU;
    if (dc >= 1.0F)
    {
        sr1_hs = 0U;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        sr1_hs = CddApp_G.PeriodTicks;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = CddApp_G.PeriodTicks;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else
    {
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            sr1_ls = CddApp_G.HalfPeriodTicks;
            sr0_ls = CddApp_G.HalfPeriodTicks;
        }
    }
    GTM_ATOM0_CH1_SR1.U = sr1_ls;
    GTM_ATOM0_CH1_SR0.U = sr0_ls;
    GTM_ATOM0_CH2_SR1.U = sr1_hs;
    GTM_ATOM0_CH2_SR0.U = sr0_hs;

    /* ------------------------------------------------------------------ */
    /* Phase V                                                             */
    /* ------------------------------------------------------------------ */
    dc = AppPtr->DutyV;
    if (dc >= 1.0F)
    {
        sr1_hs = 0U;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        sr1_hs = CddApp_G.PeriodTicks;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = CddApp_G.PeriodTicks;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else
    {
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            sr1_ls = CddApp_G.HalfPeriodTicks;
            sr0_ls = CddApp_G.HalfPeriodTicks;
        }
    }
    GTM_ATOM0_CH3_SR1.U = sr1_ls;
    GTM_ATOM0_CH3_SR0.U = sr0_ls;
    GTM_ATOM0_CH4_SR1.U = sr1_hs;
    GTM_ATOM0_CH4_SR0.U = sr0_hs;

    /* ------------------------------------------------------------------ */
    /* Phase W                                                             */
    /* ------------------------------------------------------------------ */
    dc = AppPtr->DutyW;
    if (dc >= 1.0F)
    {
        sr1_hs = 0U;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        sr1_hs = CddApp_G.PeriodTicks;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = CddApp_G.PeriodTicks;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else
    {
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            sr1_ls = CddApp_G.HalfPeriodTicks;
            sr0_ls = CddApp_G.HalfPeriodTicks;
        }
    }
    GTM_ATOM0_CH5_SR1.U = sr1_ls;
    GTM_ATOM0_CH5_SR0.U = sr0_ls;
    GTM_ATOM0_CH6_SR1.U = sr1_hs;
    GTM_ATOM0_CH6_SR0.U = sr0_hs;
}

/**********************************************************************************************************************
 * CddGtm_OpenLoopRun  [STATIC — called from ISR only]
 *********************************************************************************************************************/
STATIC void CddGtm_OpenLoopRun(real32_T OmegaRadE)
{
    real32_T sinU;
    real32_T sinV;
    real32_T sinW;
    real32_T vmax;
    real32_T vmin;
    real32_T v_zero;
    real32_T mi;

    /* I1. Advance electrical angle — wrap to [0, 2π) */
    OL_State_G.theta += OmegaRadE * CddApp_G.SampleTime;
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

    /* I3. SVPWM zero-sequence injection: v0 = -0.5*(max + min) */
    vmax = sinU;
    if (sinV > vmax) { vmax = sinV; }
    if (sinW > vmax) { vmax = sinW; }
    vmin = sinU;
    if (sinV < vmin) { vmin = sinV; }
    if (sinW < vmin) { vmin = sinW; }
    v_zero = -0.5F * (vmax + vmin);

    /* I4. Map [-1..+1] → [0..1] duty cycle */
    CddApp_G.DutyU = 0.5F + 0.5F * (sinU + v_zero);
    CddApp_G.DutyV = 0.5F + 0.5F * (sinV + v_zero);
    CddApp_G.DutyW = 0.5F + 0.5F * (sinW + v_zero);

    /* I5. Write shadow registers */
    CddGtm_SetPwmDuty(&CddApp_G);
}

/**********************************************************************************************************************
 * CddGtm_OpenLoopSetRpm
 *********************************************************************************************************************/
void CddGtm_OpenLoopSetRpm(uint32_T Rpm, real32_T Mi)
{
    OL_State_G.omega_e = (real32_T)Rpm * OL_PI_OVER_30 * (real32_T)CDD_MOTOR_POLE_PAIRS;
    OL_State_G.mi      = Mi;
    OL_State_G.theta   = 0.0F;
    OL_State_G.active  = 1U;
}

/**********************************************************************************************************************
 * CddGtm_OpenLoopStop
 *********************************************************************************************************************/
void CddGtm_OpenLoopStop(void)
{
    OL_State_G.active  = 0U;
    OL_State_G.omega_e = 0.0F;
    OL_State_G.mi      = 0.0F;
    OL_State_G.theta   = 0.0F;

    CddApp_G.DutyU = 0.5F;
    CddApp_G.DutyV = 0.5F;
    CddApp_G.DutyW = 0.5F;
    CddGtm_SetPwmDuty(&CddApp_G);
}

/**********************************************************************************************************************
 * CddGtm_GetPeriodTicks
 *********************************************************************************************************************/
uint32_T CddGtm_GetPeriodTicks(void)
{
    return CddApp_G.PeriodTicks;
}

/**********************************************************************************************************************
 * CddGtm_GetSampleTime
 *********************************************************************************************************************/
real32_T CddGtm_GetSampleTime(void)
{
    return CddApp_G.SampleTime;
}
