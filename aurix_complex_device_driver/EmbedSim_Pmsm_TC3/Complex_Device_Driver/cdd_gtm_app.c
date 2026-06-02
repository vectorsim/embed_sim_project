/**********************************************************************************************************************
 * \file        cdd_gtm_app.c
 * \brief       GTM ATOM0 direct 6-channel driver for 3-phase FOC PWM generation
 *              on the AP32541 motor control board (TC38x).
 *
 * \details
 *  Channel assignment (TOUTSEL values from TC38x UM, Appendix 1):
 *
 *  | Channel    | Signal | Pin   | Polarity    | Mode        |
 *  |------------|--------|-------|-------------|-------------|
 *  | ATOM0_CH0  | Master | P00.0 | —           | SOMP master |
 *  | ATOM0_CH1  | IL1 LS | P00.2 | active HIGH | SOMP slave  |
 *  | ATOM0_CH2  | IH1 HS | P00.3 | active LOW  | SOMP slave  |
 *  | ATOM0_CH3  | IL2 LS | P00.4 | active HIGH | SOMP slave  |
 *  | ATOM0_CH4  | IH2 HS | P00.5 | active LOW  | SOMP slave  |
 *  | ATOM0_CH5  | IL3 LS | P00.6 | active HIGH | SOMP slave  |
 *  | ATOM0_CH6  | IH3 HS | P00.7 | active LOW  | SOMP slave  |
 *  | ATOM0_CH7  | ADCTRIG| —     | internal    | SOMP slave  |
 *
 *  Software dead-time — applied symmetrically on both edges in CddGtm_SetPwmDuty():
 *
 *      sr1_hs = (1 - dc) * Half          SR1 HS : rising  edge compare
 *      sr0_hs = (1 + dc) * Half          SR0 HS : falling edge compare
 *      sr1_ls = sr1_hs + DT              SR1 LS : delayed rising  edge
 *      sr0_ls = sr0_hs - DT              SR0 LS : advanced falling edge
 *
 *  CddGtm_SetPwmDuty() is file-scope static (Rule 8.7); it reads DutyU/V/W
 *  from CddApp_G and is called exclusively from the ISR and CddGtm_Init().
 *
 *  Transition to FOC: replace the SVM call in GTM_Atom_00_Ch_00_Isr() with
 *  EmbedSim_Step() or an IPC dispatch; retain the CDDAPP_RUN_STATE guard and
 *  the CCU1 interrupt-clear at the end.
 *
 * \note    MISRA C:2012 deviations recorded in this file:
 *          - Rule  8.7 : CddGtm_SetPwmDuty — internal linkage (intentional)
 *          - Rule  8.9 : OL_State_G, angle, last_error_count — file/function
 *                        scope variables (minimised lifetime)
 *          - Rule 14.4 : All controlling expressions use explicit comparison
 *          - Rule 15.5 : Single exit point per function
 *          - Rule 17.2 : No recursion
 *
 * \version     1.3.0
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
#include "embed_sim_sv_pwm.h"

/**********************************************************************************************************************
 * ISR Vector Registration
 *********************************************************************************************************************/

/** \brief  Target OS / CPU for GTM ATOM0 CH0 ISR — CPU0 (TOS = 0)                  */
#define TOS_GTM_ISR     (0x0U)

/** \brief  Service request priority number for ATOM0 CH0 on CPU0                   */
#define SRPN_GTM_ISR    CORE_00_ATOM_00_CH_00_CL_SRPN

EMBED_SIM_INTERRUPT(GTM_Atom_00_Ch_00_Isr, TOS_GTM_ISR, SRPN_GTM_ISR);

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  ATOM SOMP mode — master only; CN0 resets at CM0 = carrier period         */
#define ATOM_MODE_SOMP              (0x2U)

/** \brief  ATOM up-count mode — CN0 counts upward only                              */
#define ATOM_UD_COUNT_MODE          (0x0U)

/** \brief  CMU CLK source index — CLK0 selected for all ATOM0 channels              */
#define ATOM_CMU_CLK                (0x0U)

/**
 * \brief   SL polarity for HS channels, controlled by CDD_GTM_HS_ACTIVE_LOW.
 *
 * \details When CDD_GTM_HS_ACTIVE_LOW != 0: SL=0, output idles LOW (safe-off for
 *          active-LOW gate driver).  When == 0: SL=1, output idles HIGH.
 */
#if (CDD_GTM_HS_ACTIVE_LOW != 0U)
    #define ATOM_HS_CH_SL   (0x0U)
#else
    #define ATOM_HS_CH_SL   (0x1U)
#endif

/**
 * \brief   SL polarity for LS channels — ILx active HIGH, SL=0.
 *
 * \details Reset → output = ~SL = 1 → ILx HIGH → freewheeling diode conducts.
 *          Safe-off state for a low-side active-HIGH gate driver.
 */
#define ATOM_LS_CH_SL               (0x0U)

/** \brief  TOUTSEL mux value routing ATOM0 outputs through CDTM0                   */
#define TOUTSEL_GTM_ATOM            (0x02U)

/**
 * \brief   Maximum consecutive SVM failures before emergency duty zeroing.
 *
 * \details At 20 kHz, 100 failures = 5 ms of consecutive SVM errors.  Beyond this
 *          threshold the ISR forces DutyU/V/W to 0.0F and transitions to
 *          CDDAPP_ERROR_STATE.
 */
#define GTM_SVM_FAIL_LIMIT          (100U)

/** \brief  Open-loop electrical angle increment per ISR tick [rad].
 *
 * \details Derived offline: target_omega_e * T_s.  Adjust for commissioning only.
 */
#define GTM_OL_ANGLE_INCREMENT      (0.00399F)

/** \brief  Open-loop modulation index at start-up (dimensionless, 0 < mi < 1)      */
#define GTM_OL_MOD_INDEX_START      (0.05F)

/**********************************************************************************************************************
 * Private Types
 *********************************************************************************************************************/
/* No private types in this translation unit.                                        */

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/
/* No file-scope private variables in this translation unit.                         */

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Writes ATOM0 CH1–CH6 shadow registers for all three phases with
 *          symmetric software dead-time insertion.
 *
 * \details Reads DutyU, DutyV, DutyW from \p AppPtr and computes SR0/SR1 compare
 *          values for LS and HS channels using the centre-aligned dead-time formula:
 *
 *              sr1_hs = (1 - dc) * Half
 *              sr0_hs = (1 + dc) * Half
 *              sr1_ls = sr1_hs + DT
 *              sr0_ls = sr0_hs - DT
 *
 *          Duty clamping: dc >= 1.0F → full ON (LS always conducting);
 *                         dc <= 0.0F → full OFF (both switches at PeriodTicks).
 *
 *          The per-phase code block is repeated deliberately (U / V / W) to
 *          avoid a helper pointer/array indirection that would introduce MISRA
 *          Rule 18.4 concerns and non-constant pointer arithmetic.
 *
 *          CH7 ADC valley trigger SR0/SR1 are also refreshed here to keep
 *          the trigger aligned after any period-tick recalculation.
 *
 * \param[in]  AppPtr   Pointer to the central CddApp_G state structure (const).
 *
 * \note    STATIC — callable only from this translation unit (Rule 8.7).
 *          Must not be called before CddGtm_Init() has populated PeriodTicks
 *          and HalfPeriodTicks.
 */
static void CddGtm_SetPwmDuty(P2CONST(CddApp_T, AUTOMATIC, CDD_APPL_DATA) AppPtr);

/**********************************************************************************************************************
 * ISR — GTM ATOM0 CH0 CCU1 (20 kHz control loop)
 *********************************************************************************************************************/

/**
 * \brief   20 kHz control-loop ISR, routed to CPU0 via SRC_GTM_ATOM0_0.
 *
 * \details Triggered by the ATOM0_CH0 CCU1 compare (half-period → valley of the
 *          centre-aligned carrier).  Execution sequence:
 *
 *          1. Increment ControlLoopCounter (diagnostics / watchdog feed).
 *          2. Advance electrical angle by GTM_OL_ANGLE_INCREMENT; wrap to [0, 2π).
 *          3. If CDDAPP_RUN_STATE:
 *               a. Compute SVPWM duty cycles via SVM_CalculateDutyCycle().
 *               b. On MATRIX_SUCCESS: copy duties to CddApp_G, call
 *                  CddGtm_SetPwmDuty(), reset consecutive-error counter.
 *               c. On failure for < GTM_SVM_FAIL_LIMIT ticks: hold previous
 *                  duties (CddGtm_SetPwmDuty() re-writes the same values).
 *               d. On failure for >= GTM_SVM_FAIL_LIMIT ticks: zero all duties,
 *                  call CddGtm_SetPwmDuty(), set CDDAPP_ERROR_STATE.
 *          4. If not CDDAPP_RUN_STATE: zero duties and call CddGtm_SetPwmDuty().
 *          5. Clear CCU1 interrupt flag (write-1-to-clear).
 *
 * \note    angle and last_error_count are declared static to preserve state
 *          across ISR re-entries without using file-scope variables (Rule 8.9).
 *
 * \note    MISRA Rule 8.4 satisfied: prototype supplied by EMBED_SIM_INTERRUPT
 *          macro expansion in the vector table.
 *
 * \note    Transition to FOC: replace step 3a–3d with EmbedSim_Step() and IPC
 *          dispatch.  Retain steps 1, 2 (for speed estimation seed), 4, and 5.
 */
void GTM_Atom_00_Ch_00_Isr(void)
{
    /* MISRA Rule 8.9: static locals preserve ISR state without file scope.          */
    static FocAngle_T  angle           = { .ThetaE = 0.0F };
    static uint32_T    last_error_count = 0U;

    SVM_DutyCycle_T    svm_dc;
    MatrixStatus_Type  status;

    /* Step 1 — diagnostics counter                                                  */
    CddApp_G.ControlLoopCounter++;

    /* Step 2 — electrical angle integration with conditional wrap (avoids fmodf)   */
    angle.ThetaE += GTM_OL_ANGLE_INCREMENT;

    if (angle.ThetaE >= RAD_360)
    {
        angle.ThetaE -= RAD_360;
    }
    else if (angle.ThetaE < 0.0F)
    {
        angle.ThetaE += RAD_360;
    }
    else
    {
        /* Angle within [0, 2π) — no wrap required (Rule 14.4: mandatory else)      */
    }

    /* Step 3 — PWM generation, active only in RUN state                            */
    if (CddApp_G.CDDAppStatus == CDDAPP_RUN_STATE)
    {
        status = SVM_CalculateDutyCycle(GTM_OL_MOD_INDEX_START, &angle, &svm_dc);

        if (status == MATRIX_SUCCESS)
        {
            /* 3a/3b — nominal path: copy duties and write shadow registers          */
            SVM_GetDutyCyclesFloat(&svm_dc,
                                   &CddApp_G.DutyU,
                                   &CddApp_G.DutyV,
                                   &CddApp_G.DutyW);
            CddGtm_SetPwmDuty(&CddApp_G);
            last_error_count = 0U;
        }
        else
        {
            last_error_count++;

            if (last_error_count >= GTM_SVM_FAIL_LIMIT)
            {
                /* 3d — sustained SVM failure: emergency shutdown                   */
                CddApp_G.DutyU     = 0.0F;
                CddApp_G.DutyV     = 0.0F;
                CddApp_G.DutyW     = 0.0F;
                CddGtm_SetPwmDuty(&CddApp_G);
                CddApp_G.CDDAppStatus = CDDAPP_ERROR_STATE;
            }
            else
            {
                /* 3c — transient failure: retain previous duty cycle               */
                CddGtm_SetPwmDuty(&CddApp_G);
            }
        }
    }
    else
    {
        /* Step 4 — safe-off in all non-RUN states                                  */
        CddApp_G.DutyU = 0.0F;
        CddApp_G.DutyV = 0.0F;
        CddApp_G.DutyW = 0.0F;
        CddGtm_SetPwmDuty(&CddApp_G);
    }

    /* Step 5 — clear CCU1 interrupt flag (write-1-to-clear, TC38x UM §24)          */
    GTM_ATOM0_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1U;
}

/**********************************************************************************************************************
 * CddGtm_SetPwmDuty  [STATIC]
 *********************************************************************************************************************/
void CddGtm_SetPwmDuty(P2CONST(CddApp_T, AUTOMATIC, CDD_APPL_DATA) AppPtr)
{
    uint32_T       sr1_hs;
    uint32_T       sr0_hs;
    uint32_T       sr1_ls;
    uint32_T       sr0_ls;
    real32_T       dc;
    const uint32_T dt = CDD_GTM_SW_DEAD_TIME_TICKS;

    /* ADC valley trigger: refresh SR0/SR1 on every duty update                     */
    GTM_ATOM0_CH7_SR1.B.SR1 = CDD_GTM_ADC_VALLEY_OFFSET_TICKS;
    GTM_ATOM0_CH7_SR0.B.SR0 = CddApp_G.PeriodTicks;

    /* ------------------------------------------------------------------
     * Phase U  —  CH1 (LS IL1 P00.2) / CH2 (HS /IH1 P00.3)
     * ------------------------------------------------------------------ */
    dc = AppPtr->DutyU;

    if (dc >= 1.0F)
    {
        /* Full ON: LS conducts entire period; HS at period (off)                   */
        sr1_hs = 0U;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = 0U;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else if (dc <= 0.0F)
    {
        /* Full OFF: both switches parked at period ticks (zero pulse width)        */
        sr1_hs = CddApp_G.PeriodTicks;
        sr0_hs = CddApp_G.PeriodTicks;
        sr1_ls = CddApp_G.PeriodTicks;
        sr0_ls = CddApp_G.PeriodTicks;
    }
    else
    {
        /* Normal range: compute centre-aligned SR values, then add dead-time       */
        sr1_hs = (uint32_T)((1.0F - dc) * (real32_T)CddApp_G.HalfPeriodTicks);
        sr0_hs = (uint32_T)((1.0F + dc) * (real32_T)CddApp_G.HalfPeriodTicks);

        if ((sr1_hs + dt) < (sr0_hs - dt))
        {
            sr1_ls = sr1_hs + dt;
            sr0_ls = sr0_hs - dt;
        }
        else
        {
            /* Dead-time exceeds pulse width: collapse to 50% (safe neutral)        */
            sr1_ls = CddApp_G.HalfPeriodTicks;
            sr0_ls = CddApp_G.HalfPeriodTicks;
        }
    }

    GTM_ATOM0_CH1_SR1.B.SR1 = sr1_ls;   /* Phase U LS rising  edge                */
    GTM_ATOM0_CH1_SR0.B.SR0 = sr0_ls;   /* Phase U LS falling edge                */
    GTM_ATOM0_CH2_SR1.B.SR1 = sr1_hs;   /* Phase U HS rising  edge                */
    GTM_ATOM0_CH2_SR0.B.SR0 = sr0_hs;   /* Phase U HS falling edge                */

    /* ------------------------------------------------------------------
     * Phase V  —  CH3 (LS IL2 P00.4) / CH4 (HS /IH2 P00.5)
     * ------------------------------------------------------------------ */
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

    GTM_ATOM0_CH3_SR1.B.SR1 = sr1_ls;   /* Phase V LS rising  edge                */
    GTM_ATOM0_CH3_SR0.B.SR0 = sr0_ls;   /* Phase V LS falling edge                */
    GTM_ATOM0_CH4_SR1.B.SR1 = sr1_hs;   /* Phase V HS rising  edge                */
    GTM_ATOM0_CH4_SR0.B.SR0 = sr0_hs;   /* Phase V HS falling edge                */

    /* ------------------------------------------------------------------
     * Phase W  —  CH5 (LS IL3 P00.6) / CH6 (HS /IH3 P00.7)
     * ------------------------------------------------------------------ */
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

    GTM_ATOM0_CH5_SR1.B.SR1 = sr1_ls;   /* Phase W LS rising  edge                */
    GTM_ATOM0_CH5_SR0.B.SR0 = sr0_ls;   /* Phase W LS falling edge                */
    GTM_ATOM0_CH6_SR1.B.SR1 = sr1_hs;   /* Phase W HS rising  edge                */
    GTM_ATOM0_CH6_SR0.B.SR0 = sr0_hs;   /* Phase W HS falling edge                */
}

/**********************************************************************************************************************
 * CddGtm_Start
 *********************************************************************************************************************/

/**
 * \brief   Finalises start-up: initialises SVM tables, sets RUN state, and issues
 *          HOST_TRIG to transfer all shadow registers to active compare registers.
 *
 * \details Call sequence:
 *              CddGtm_Init()           — hardware init, shadow regs pre-loaded
 *              CddApp_InitInverter()   — gate driver enable
 *              CddGtm_Start()          — HOST_TRIG, PWM goes live
 *              SRC SRE = 1             — arm ISR after PWM is live
 *
 *          HOST_TRIG is a single-shot write; ATOM0 begins counting on the next
 *          CMU CLK0 edge.
 *
 * \return  void
 */
void CddGtm_Start(void)
{
    SVM_Init();
    CddApp_G.CDDAppStatus          = CDDAPP_RUN_STATE;
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG = 0x1U;
}

/**********************************************************************************************************************
 * CddGtm_Init
 *********************************************************************************************************************/

/**
 * \brief   Initialises GTM CMU, ATOM0 channels CH0–CH7, CDTM0 DTM4/DTM5, pin mux,
 *          and pre-loads shadow registers with a 50 % zero-vector duty cycle.
 *
 * \details Initialisation sequence:
 *
 *          1. Compute PeriodTicks, HalfPeriodTicks, SampleTime from
 *             GTM_CMU_CLK0_FREQUENCY and CDD_CONTROL_LOOP_FREQUENCY.
 *          2. Pre-load DutyU/V/W = 0.5F (zero vector) and call CddGtm_SetPwmDuty()
 *             to populate all phase shadow registers before HOST_TRIG.
 *          3. Read AGC register images from hardware once (RMW pattern).
 *          4. Configure ATOM0_CH0: SOMP master, TRIGOUT=1, CCU1 ISR enabled
 *             (SRE=1 set here; ISR will fire after CddGtm_Start() HOST_TRIG).
 *          5. Configure ATOM0_CH1–CH6: SOMP slaves, RST_CCU0=1 (sync to master).
 *          6. Configure ATOM0_CH7: ADC valley trigger via ADCTRIG0OUT0.
 *          7. Configure CDTM0_DTM4 / DTM5: CMU CLK0, passthrough (no DTM dead-time).
 *          8. Write back accumulated AGC enable/output/update control words.
 *
 *          HOST_TRIG is NOT issued here.  Call CddGtm_Start() after
 *          CddApp_InitInverter() to start PWM simultaneously with gate-driver enable.
 *
 * \return  void
 */
void CddGtm_Init(void)
{
    Ifx_GTM_ATOM_CH_CTRL        chCtrl;
    Ifx_GTM_ATOM_CH_IRQ_EN      chIrqEn;
    Ifx_SRC_SRCR                srcCfg;
    Ifx_GTM_ATOM_AGC_GLB_CTRL   glbCtrl;
    Ifx_GTM_ATOM_AGC_FUPD_CTRL  fupdCtrl;
    Ifx_GTM_ATOM_AGC_ENDIS_CTRL endisCtrl;
    Ifx_GTM_ATOM_AGC_OUTEN_CTRL outenCtrl;

    /* Step 1 — timing constants from CMU CLK0 frequency and control loop rate      */
    CddApp_G.PeriodTicks        = (uint32_T)((real32_T)GTM_CMU_CLK0_FREQUENCY /
                                             (real32_T)CDD_CONTROL_LOOP_FREQUENCY);
    CddApp_G.HalfPeriodTicks    = CddApp_G.PeriodTicks / 2U;
    CddApp_G.SampleTime         = 1.0F / (real32_T)CDD_CONTROL_LOOP_FREQUENCY;
    CddApp_G.ControlLoopCounter = 0U;

    /* Step 2 — zero-vector pre-load: 50 % duty on all phases (no net voltage)      */
    CddApp_G.DutyU = 0.5F;
    CddApp_G.DutyV = 0.5F;
    CddApp_G.DutyW = 0.5F;
    CddGtm_SetPwmDuty(&CddApp_G);

    /* Step 3 — snapshot current AGC registers for read-modify-write                */
    glbCtrl.U   = GTM_ATOM0_AGC_GLB_CTRL.U;
    fupdCtrl.U  = GTM_ATOM0_AGC_FUPD_CTRL.U;
    endisCtrl.U = GTM_ATOM0_AGC_ENDIS_CTRL.U;
    outenCtrl.U = GTM_ATOM0_AGC_OUTEN_CTRL.U;

    /* ================================================================== */
    /* Step 4 — MASTER: ATOM0_CH0  P00.0  CCU1 ISR → CPU0                */
    /* ================================================================== */

    /* M1. Channel control: SOMP master, TRIGOUT=1 propagates sync to slaves        */
    chCtrl.U            = GTM_ATOM0_CH0_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x0U;   /* Master does not self-reset — controlled by CM0 */
    chCtrl.B.TRIGOUT    = 0x1U;   /* Propagate sync trigger to slave channels        */
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH0_CTRL.U = chCtrl.U;

    /* M2. Shadow registers: period + CCU1 compare at half-period (valley)          */
    GTM_ATOM0_CH0_SR0.B.SR0 = CddApp_G.PeriodTicks;
    GTM_ATOM0_CH0_SR1.B.SR1 = CddApp_G.HalfPeriodTicks;

    /* M3. Enable CCU1 interrupt only (CCU0 unused for master)                      */
    chIrqEn.U               = GTM_ATOM0_CH0_IRQ_EN.U;
    chIrqEn.B.CCU0TC_IRQ_EN = 0x0U;
    chIrqEn.B.CCU1TC_IRQ_EN = 0x1U;
    GTM_ATOM0_CH0_IRQ_EN.U  = chIrqEn.U;
    GTM_ATOM0_CH0_IRQ_MODE.B.IRQ_MODE = 0x0U;   /* Pulse mode                      */

    /* M4. Service request node: route to CPU0, SRPN set, clear pending, SRE=1      */
    srcCfg.U      = SRC_GTM_ATOM0_0.U;
    srcCfg.B.SRPN = CORE_01_ATOM_00_CH_00_CL_SRPN;
    srcCfg.B.TOS  = TOS_GTM_ISR;
    srcCfg.B.CLRR = 0x1U;   /* Clear pending request before arming                 */
    srcCfg.B.SRE  = 0x1U;   /* Arm: ISR will fire after HOST_TRIG in CddGtm_Start()*/
    SRC_GTM_ATOM0_0.U = srcCfg.U;

    /* M5. Pin mux: TOUT9 → P00.0                                                   */
    GTM_TOUTSEL1.B.SEL1 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmMaster_P00_0();

    /* M6. AGC: enable CH0 update, force-update, enable, output-enable              */
    glbCtrl.B.UPEN_CTRL0    = 0x2U;
    fupdCtrl.B.FUPD_CTRL0   = 0x2U;
    endisCtrl.B.ENDIS_CTRL0 = 0x2U;
    outenCtrl.B.OUTEN_CTRL0 = 0x2U;

    /* ================================================================== */
    /* Step 5a — PHASE U LS: ATOM0_CH1  IL1 P00.2  active HIGH  SL=0     */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH1_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;   /* Slave: reset CN0 on master TRIGOUT             */
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH1_CTRL.U = chCtrl.U;
    GTM_TOUTSEL1.B.SEL3  = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseULs_P00_2();
    glbCtrl.B.UPEN_CTRL1    = 0x2U;
    fupdCtrl.B.FUPD_CTRL1   = 0x2U;
    endisCtrl.B.ENDIS_CTRL1 = 0x2U;
    outenCtrl.B.OUTEN_CTRL1 = 0x2U;

    /* ================================================================== */
    /* Step 5b — PHASE U HS: ATOM0_CH2  /IH1 P00.3  active LOW   SL=0   */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH2_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH2_CTRL.U = chCtrl.U;
    GTM_TOUTSEL1.B.SEL4  = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseUHs_P00_3();
    glbCtrl.B.UPEN_CTRL2    = 0x2U;
    fupdCtrl.B.FUPD_CTRL2   = 0x2U;
    endisCtrl.B.ENDIS_CTRL2 = 0x2U;
    outenCtrl.B.OUTEN_CTRL2 = 0x2U;

    /* ================================================================== */
    /* Step 5c — PHASE V LS: ATOM0_CH3  IL2 P00.4  active HIGH  SL=0    */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH3_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH3_CTRL.U = chCtrl.U;
    GTM_TOUTSEL1.B.SEL5  = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseVLs_P00_4();
    glbCtrl.B.UPEN_CTRL3    = 0x2U;
    fupdCtrl.B.FUPD_CTRL3   = 0x2U;
    endisCtrl.B.ENDIS_CTRL3 = 0x2U;
    outenCtrl.B.OUTEN_CTRL3 = 0x2U;

    /* ================================================================== */
    /* Step 5d — PHASE V HS: ATOM0_CH4  /IH2 P00.5  active LOW   SL=0  */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH4_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH4_CTRL.U = chCtrl.U;
    GTM_TOUTSEL1.B.SEL6  = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseVHs_P00_5();
    glbCtrl.B.UPEN_CTRL4    = 0x2U;
    fupdCtrl.B.FUPD_CTRL4   = 0x2U;
    endisCtrl.B.ENDIS_CTRL4 = 0x2U;
    outenCtrl.B.OUTEN_CTRL4 = 0x2U;

    /* ================================================================== */
    /* Step 5e — PHASE W LS: ATOM0_CH5  IL3 P00.6  active HIGH  SL=0   */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH5_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = ATOM_LS_CH_SL;
    GTM_ATOM0_CH5_CTRL.U = chCtrl.U;
    GTM_TOUTSEL1.B.SEL7  = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseWLs_P00_6();
    glbCtrl.B.UPEN_CTRL5    = 0x2U;
    fupdCtrl.B.FUPD_CTRL5   = 0x2U;
    endisCtrl.B.ENDIS_CTRL5 = 0x2U;
    outenCtrl.B.OUTEN_CTRL5 = 0x2U;

    /* ================================================================== */
    /* Step 5f — PHASE W HS: ATOM0_CH6  /IH3 P00.7  active LOW   SL=0  */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH6_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = ATOM_HS_CH_SL;
    GTM_ATOM0_CH6_CTRL.U = chCtrl.U;
    GTM_TOUTSEL2.B.SEL0  = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseWHs_P00_7();
    glbCtrl.B.UPEN_CTRL6    = 0x2U;
    fupdCtrl.B.FUPD_CTRL6   = 0x2U;
    endisCtrl.B.ENDIS_CTRL6 = 0x2U;
    outenCtrl.B.OUTEN_CTRL6 = 0x2U;

    /* ================================================================== */
    /* Step 6 — ADC TRIGGER: ATOM0_CH7  valley-aligned, internal only    */
    /* ================================================================== */
    /* SR1 = valley offset from carrier reset; SR0 = full period (pulse width ~0)   */
    chCtrl.U            = GTM_ATOM0_CH7_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH7_CTRL.U    = chCtrl.U;
    GTM_ATOM0_CH7_SR0.B.SR0 = CddApp_G.PeriodTicks;
    GTM_ATOM0_CH7_SR1.B.SR1 = CDD_GTM_ADC_VALLEY_OFFSET_TICKS;

    /* Route CH7 trigger to EVADC groups G0, G1, G2 via ADCTRIG0OUT0               */
    GTM_ADCTRIG0OUT0.B.SEL0 = 0x8U;   /* ATOM0_CH7 → EVADC G0                     */
    GTM_ADCTRIG0OUT0.B.SEL1 = 0x8U;   /* ATOM0_CH7 → EVADC G1                     */
    GTM_ADCTRIG0OUT0.B.SEL2 = 0x8U;   /* ATOM0_CH7 → EVADC G2                     */

    glbCtrl.B.UPEN_CTRL7    = 0x2U;
    fupdCtrl.B.FUPD_CTRL7   = 0x2U;
    endisCtrl.B.ENDIS_CTRL7 = 0x2U;
    outenCtrl.B.OUTEN_CTRL7 = 0x2U;

    /* ================================================================== */
    /* Step 7 — CDTM0 DTM4 + DTM5: CMU CLK0, no DTM dead-time            */
    /* ================================================================== */
    GTM_CDTM0_DTM4_CTRL.B.CLK_SEL      = 0x0U;   /* Select CMU CLK0               */
    GTM_CDTM0_DTM4_CTRL.B.SHUT_OFF_RST = 0x0U;   /* No shut-off reset             */
    GTM_CDTM0_DTM5_CTRL.B.CLK_SEL      = 0x0U;
    GTM_CDTM0_DTM5_CTRL.B.SHUT_OFF_RST = 0x0U;

    /* Step 8 — write back accumulated AGC control words atomically                 */
    GTM_ATOM0_AGC_GLB_CTRL.U   = glbCtrl.U;
    GTM_ATOM0_AGC_FUPD_CTRL.U  = fupdCtrl.U;
    GTM_ATOM0_AGC_ENDIS_CTRL.U = endisCtrl.U;
    GTM_ATOM0_AGC_OUTEN_CTRL.U = outenCtrl.U;
}

/**********************************************************************************************************************
 * CddGtm_GetPeriodTicks
 *********************************************************************************************************************/

/**
 * \brief   Returns the carrier period in CMU CLK0 ticks.
 * \return  CddApp_G.PeriodTicks  [CLK0 ticks]
 */
uint32_T CddGtm_GetPeriodTicks(void)
{
    return CddApp_G.PeriodTicks;
}

/**********************************************************************************************************************
 * CddGtm_GetSampleTime
 *********************************************************************************************************************/

/**
 * \brief   Returns the control-loop sample time in seconds.
 * \return  CddApp_G.SampleTime  [s]
 */
real32_T CddGtm_GetSampleTime(void)
{
    return CddApp_G.SampleTime;
}
