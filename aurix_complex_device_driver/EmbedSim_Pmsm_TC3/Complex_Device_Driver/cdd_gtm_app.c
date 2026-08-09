/**********************************************************************************************************************
 * \file      cdd_gtm_app.c
 * \brief     GTM ATOM0 direct 6-channel driver for 3-phase FOC PWM generation
 *            on the AP32541 motor control board (TC38x).
 *
 * \details   Provides static-allocation GTM configuration targeting Infineon
 *            AURIX TC38x.  All algorithms are iterative — no recursion — and
 *            the implementation is MISRA C:2012 compliant.
 *
 *            Channel assignment (TOUTSEL values from TC38x UM, Appendix 1):
 *
 *            | Channel    | Signal | Pin   | Polarity    | Mode        |
 *            |------------|--------|-------|-------------|-------------|
 *            | ATOM0_CH0  | Master | P00.0 | —           | SOMP master |
 *            | ATOM0_CH1  | IL1 LS | P00.2 | active HIGH | SOMP slave  |
 *            | ATOM0_CH2  | IH1 HS | P00.3 | active LOW  | SOMP slave  |
 *            | ATOM0_CH3  | IL2 LS | P00.4 | active HIGH | SOMP slave  |
 *            | ATOM0_CH4  | IH2 HS | P00.5 | active LOW  | SOMP slave  |
 *            | ATOM0_CH5  | IL3 LS | P00.6 | active HIGH | SOMP slave  |
 *            | ATOM0_CH6  | IH3 HS | P00.7 | active LOW  | SOMP slave  |
 *            | ATOM0_CH7  | ADCTRIG| P00.8 | edge@centre | SOMP slave  |
 *
 *            Software dead-time — applied symmetrically on both edges:
 *
 *                sr1_hs = (1 - dc) * Half          SR1 HS : rising  edge compare
 *                sr0_hs = (1 + dc) * Half          SR0 HS : falling edge compare
 *                sr1_ls = sr1_hs + DT              SR1 LS : delayed rising  edge
 *                sr0_ls = sr0_hs - DT              SR0 LS : advanced falling edge
 *
 *            Control-path dispatch (20 kHz) — two options commanded through
 *            CddApp_T (CddApp_SetCtrlMode / CddApp_SetSpeedRefRpm):
 *                - CDDAPP_CTRL_OPENLOOP   : V/f rotating vector at slew-limited
 *                                            SpeedRefRpm, no feedback.
 *                - CDDAPP_CTRL_CLOSEDLOOP : Full sensorless Differential
 *                                            Flatness Controller.
 *            The mode is latched by the ISR once on the activation edge and is
 *            fixed for the entire run — no switching during operation.
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : All functions have explicit return type
 *              - Rule  8.5 : One declaration per identifier
 *              - Rule  8.6 : No definitions in header files
 *              - Rule  8.7 : Internal linkage for static functions
 *              - Rule  8.9 : File scope variables minimised
 *              - Rule 14.4 : All controlling expressions use explicit comparison
 *              - Rule 15.5 : Single exit point per function
 *              - Rule 17.2 : No recursion
 *              - Rule 18.4 : No non-constant pointer arithmetic
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase  (single-letter → Uppercase)
 *              - Output pointers: PascalCase_P
 *              - Local variables: lower_snake_case
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   1.6.0
 * \date      2026-07-04
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) EmbedSim Project / Paul Abraham 2024
 *            https://github.com/vectorsim/embed_sim_project
 *            SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/

#include "cdd_gtm_app.h"
#include "cdd_app.h"
#include "cdd_gpio_app.h"
#include "cdd_gpt12_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "cdd_evadc_app.h"
#include "embed_sim_control.h"
#include "IfxGtm_reg.h"
#include "IfxGtm_Atom.h"
#include "IfxSrc_reg.h"
#include <math.h>




/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief  Target OS / CPU for GTM ATOM0 CH0 ISR — CPU0 (TOS = 0)                  */
#define TOS_GTM_ISR     (0x0U)

/** \brief  Service request priority number for ATOM0 CH0 on CPU0                   */
#define SRPN_GTM_ISR    CORE_00_ATOM_00_CH_00_CL_SRPN

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
    #define ATOM_HS_CH_SL   (0x1U)
#else
    #define ATOM_HS_CH_SL   (0x0U)
#endif

/**
 * \brief   SL polarity for LS channels
 */
#define ATOM_LS_CH_SL               (0x1U)

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

/**
 * \brief   Maximum consecutive DFC_Step() failures before emergency duty zeroing.
 *
 * \details Same rationale as GTM_SVM_FAIL_LIMIT: at 20 kHz, 100 failures = 5 ms.
 *          A DFC_Step() mid-step error already forces safe 0.5 duties internally;
 *          the counter guards against a persistent fault (ADC chain down, transform
 *          error) and escalates to CDDAPP_ERROR_STATE + DFC_Reset().
 */
#define GTM_DFC_FAIL_LIMIT          (100U)

/** \brief  rad/s (mechanical) → RPM for the telemetry accessor.  [RPM*s/rad]        */
#define GTM_RADPS_TO_RPM            (60.0F / ES_MATH_2PI_F)

/**
 * \brief   Motor pole pairs — NANOTEC DB42S02 (8 poles).  [dimensionless] — VERIFY
 *
 * \details Used by the open-loop V/f path to convert the mechanical speed
 *          reference into the electrical angle increment.  Must match the
 *          value used by the DFC motor parameter set.
 */
#define GTM_MOTOR_POLE_PAIRS        (4.0F)

/**
 * \brief   Mechanical RPM → electrical rad/s conversion.  [rad*min/(s*rev)]
 */
#define GTM_RPM_TO_RADPS_E          (ES_MATH_2PI_F * GTM_MOTOR_POLE_PAIRS / 60.0F)

/**
 * \brief   Open-loop speed slew limit per ISR tick.  [RPM/tick]
 *
 * \details 0.025 RPM/tick at 20 kHz = 500 RPM/s.  An open-loop rotating
 *          vector must never step in frequency (loss of synchronism); the
 *          reference is therefore ramped inside CddGtm_RunOpenLoop()
 *          regardless of how CddApp_G.SpeedRefRpm changes.
 */
#define GTM_OL_RPM_SLEW             (0.025F)

/** \brief  Open-loop V/f law: modulation index at zero speed (boost).
 *          [dimensionless] — tune on hardware                                       */
#define GTM_OL_VF_BOOST             (0.05F)

/** \brief  Open-loop V/f law: modulation index gain per RPM.
 *          mi = BOOST + GAIN * |rpm|.  1.0E-4 → mi = 0.20 at 1500 RPM.
 *          [1/RPM] — tune on hardware                                               */
#define GTM_OL_VF_GAIN              (1.0E-4F)

/** \brief  Open-loop V/f law: modulation index ceiling.  [dimensionless]            */
#define GTM_OL_MI_MAX               (0.95F)

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/



/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/



EmbedSimCtrlInput_T  EmbedSimCtrlInput_G;
EmbedSimCtrlOutput_T EmbedSimCtrlOutput_G;

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

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


/**
 * \brief   Helper function to configure a single phase's PWM compare registers.
 *
 * \details Computes SR0 and SR1 compare values for both high-side and low-side
 *          switches of a single phase using the centre-aligned dead-time formula.
 *          This function is called once per phase from CddGtm_SetPwmDuty().
 *
 * \param[in]  Duty          Duty cycle for this phase [0.0 .. 1.0]
 * \param[in]  HalfPeriod    Half of the PWM period in ticks
 * \param[in]  DeadTime      Dead-time in ticks
 * \param[out] HsSr0Ptr      Pointer to HS SR0 register (falling edge)
 * \param[out] HsSr1Ptr      Pointer to HS SR1 register (rising edge)
 * \param[out] LsSr0Ptr      Pointer to LS SR0 register (falling edge)
 * \param[out] LsSr1Ptr      Pointer to LS SR1 register (rising edge)
 *
 * \note    STATIC — callable only from this translation unit (Rule 8.7).
 */
static void CddGtm_ConfigurePhase(real32_T Duty,
                                  uint32_T HalfPeriod,
                                  uint32_T DeadTime,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR0, AUTOMATIC, CDD_APPL_DATA) HsSr0Ptr,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR1, AUTOMATIC, CDD_APPL_DATA) HsSr1Ptr,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR0, AUTOMATIC, CDD_APPL_DATA) LsSr0Ptr,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR1, AUTOMATIC, CDD_APPL_DATA) LsSr1Ptr);

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/**********************************************************************************************************************
 * ISR — GTM ATOM0 CH0 CCU1 (20 kHz control loop)
 *********************************************************************************************************************/
EMBED_SIM_INTERRUPT(GTM_Atom_00_Ch_00_Isr, TOS_GTM_ISR, SRPN_GTM_ISR);

/**
 * \brief   20 kHz control-loop ISR, routed to CPU0 via SRC_GTM_ATOM0_0.
 *
 * \details Triggered by the ATOM0_CH0 CCU1 compare (half-period → valley of the
 *          centre-aligned carrier).  Execution sequence:
 *
 *          1. Increment ControlLoopCounter (diagnostics / watchdog feed).
 *          2. If CDDAPP_RUN_STATE:
 *               a. Measurements — single VF-gated EVADC read of the tick into
 *                  CddApp_G.Meas (phase sense voltages + DC link [V]), then
 *                  conversion into CddApp_G.PhaseCurrents [A].  Both control
 *                  modes get fresh measurements (open loop: telemetry only).
 *               b. Activation edge (cold start, RUN re-entry after fault):
 *                  latch CddApp_G.CtrlMode into ModeActive — once, fixed for
 *                  the entire run — and reset both controller states
 *                  (DFC_Reset(): ALIGN from standstill; open-loop angle and
 *                  speed ramp to zero).
 *               c. Dispatch on the LATCHED mode:
 *                  CDDAPP_CTRL_CLOSEDLOOP (and CtrlInitDone) → CddGtm_RunDfc()
 *                  — full sensorless DFC.  Otherwise → CddGtm_RunOpenLoop()
 *                  — V/f rotating vector at the slew-limited SpeedRefRpm.
 *          3. If not CDDAPP_RUN_STATE: zero duties and call CddGtm_SetPwmDuty();
 *             clear CtrlActive so the next activation re-latches the mode
 *             and restarts from a defined state.
 *          4. Clear CCU1 interrupt flag (write-1-to-clear).
 *
 * \note    MISRA Rule 8.4 satisfied: prototype supplied by EMBED_SIM_INTERRUPT
 *          macro expansion in the vector table.
 */
void GTM_Atom_00_Ch_00_Isr(void)
{

    CddApp_G.ControlLoopCounter++;
    CddApp_G.DutyAdcTrig = 0.8F;
    CddEvadc_ConvertPhaseCurrents(&CddApp_G);
    CddGpt12_Update();
    CddApp_G.RotorSpeedRpm = CddGpt12_GetSpeedRpm();
    CddApp_G.RotorPosition = CddGpt12_GetMechanicalPosition();

    if((CddApp_G.CDDAppStatus == CDDAPP_INIT_OK) || (CddApp_G.CDDAppStatus == CDDAPP_RUN_STATE))
    {

        if(CddApp_G.CDDAppStatus == CDDAPP_INIT_OK)
        {
            CddApp_G.DutyU       = 0.5F;
            CddApp_G.DutyV       = 0.5F;
            CddApp_G.DutyW       = 0.5F;
            CddEvadc_CalibrateCurrentOffset(&CddApp_G);
        }
        else
        {
            EmbedSimCtrlInput_G.AngularVelocityRefRpm   = CddApp_G.SpeedRefRpm;
            EmbedSimCtrlInput_G.Iu                      = CddApp_G.Iu;
            EmbedSimCtrlInput_G.Iv                      = CddApp_G.Iv;
            EmbedSimCtrlInput_G.Iw                      = CddApp_G.Iw;
            EmbedSimCtrlInput_G.DutyU                   = CddApp_G.DutyU;
            EmbedSimCtrlInput_G.DutyV                   = CddApp_G.DutyV;
            EmbedSimCtrlInput_G.DutyW                   = CddApp_G.DutyW;
            EmbedSimCtrlInput_G.RotorPositionSensor     = CddApp_G.RotorPosition;
            EmbedSimCtrlInput_G.RotorSpeedSensor        = CddApp_G.RotorSpeedRpm;
            EmbedSimCtrlInput_G.SampleTime              = CddApp_G.SampleTime;
            EmbedSimCtrlInput_G.Vdc                     = CddApp_G.Vdc;
            EmbedSimCtrlInput_G.Valid = 0x1U;
            EmbedSim_ControlStep(&EmbedSimCtrlInput_G, &EmbedSimCtrlOutput_G);

            if(EmbedSimCtrlOutput_G.Valid == 0x1U)
            {
                CddApp_G.DutyU = EmbedSimCtrlOutput_G.DutyU;
                CddApp_G.DutyV = EmbedSimCtrlOutput_G.DutyV;
                CddApp_G.DutyW = EmbedSimCtrlOutput_G.DutyW;
            }
            else
            {
                CddApp_G.DutyU       = 0.5F;
                CddApp_G.DutyV       = 0.5F;
                CddApp_G.DutyW       = 0.5F;
            }
        }
    }

    CddGtm_SetPwmDuty(&CddApp_G);
    GTM_ATOM0_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1U;
}



/**********************************************************************************************************************
 * CddGtm_ConfigurePhase
 *********************************************************************************************************************/

/**
 * \brief   Configures a single phase's PWM compare registers with dead-time.
 *
 * \details Computes SR0 and SR1 compare values for both high-side and low-side
 *          switches of a single phase using the centre-aligned dead-time formula.
 *          This function is called once per phase from CddGtm_SetPwmDuty().
 *
 *          The dead-time is applied symmetrically:
 *          - High-side rising edge: delayed by DeadTime ticks
 *          - Low-side falling edge: advanced by DeadTime ticks
 *
 * \param[in]  Duty          Duty cycle for this phase [0.0 .. 1.0]
 * \param[in]  HalfPeriod    Half of the PWM period in ticks
 * \param[in]  DeadTime      Dead-time in ticks
 * \param[out] HsSr0Ptr      Pointer to HS SR0 register (falling edge)
 * \param[out] HsSr1Ptr      Pointer to HS SR1 register (rising edge)
 * \param[out] LsSr0Ptr      Pointer to LS SR0 register (falling edge)
 * \param[out] LsSr1Ptr      Pointer to LS SR1 register (rising edge)
 *
 * \note    STATIC — callable only from this translation unit (Rule 8.7).
 */
void CddGtm_ConfigurePhase(real32_T Duty,
                                  uint32_T HalfPeriod,
                                  uint32_T DeadTime,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR0, AUTOMATIC, CDD_APPL_DATA) HsSr0Ptr,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR1, AUTOMATIC, CDD_APPL_DATA) HsSr1Ptr,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR0, AUTOMATIC, CDD_APPL_DATA) LsSr0Ptr,
                                  P2VAR(volatile Ifx_GTM_ATOM_CH_SR1, AUTOMATIC, CDD_APPL_DATA) LsSr1Ptr)
{
    uint32_T sr1Hs = 0U;
    uint32_T sr0Hs = 0U;
    uint32_T sr1Ls = 0U;
    uint32_T sr0Ls = 0U;

    sr1Hs = (uint32_T)((1.0F - Duty) * (real32_T)HalfPeriod);
    sr0Hs = (uint32_T)((1.0F + Duty) * (real32_T)HalfPeriod);

    sr1Ls = sr1Hs - DeadTime;
    sr0Ls = sr0Hs + DeadTime;

    LsSr1Ptr->B.SR1 = sr1Ls;
    LsSr0Ptr->B.SR0 = sr0Ls;
    HsSr1Ptr->B.SR1 = sr1Hs;
    HsSr0Ptr->B.SR0 = sr0Hs;
}

/**********************************************************************************************************************
 * CddGtm_SetPwmDuty
 *********************************************************************************************************************/

/**
 * \brief   Writes ATOM0 CH1–CH6 shadow registers for all three phases with
 *          symmetric software dead-time insertion.
 *
 * \details Reads DutyU, DutyV, DutyW from \p AppPtr and calls CddGtm_ConfigurePhase()
 *          for each phase to compute and write the compare values.
 *          Also updates the ADC trigger channel (CH7) to maintain alignment
 *          with the PWM carrier.
 *
 * \param[in]  AppPtr   Pointer to the central CddApp_G state structure (const).
 *
 * \note    STATIC — callable only from this translation unit (Rule 8.7).
 *          Must not be called before CddGtm_Init() has populated PeriodTicks
 *          and HalfPeriodTicks.
 */
static void CddGtm_SetPwmDuty(P2CONST(CddApp_T, AUTOMATIC, CDD_APPL_DATA) AppPtr)
{
    uint32_T halfPeriod;
    uint32_T deadTimeTicks;

    halfPeriod    = AppPtr->HalfPeriodTicks;
    deadTimeTicks = CDD_GTM_SW_DEAD_TIME_TICKS;


    /* Phases U,V,B */
    CddGtm_ConfigurePhase(AppPtr->DutyU, halfPeriod, deadTimeTicks,
                          &GTM_ATOM0_CH2_SR0, &GTM_ATOM0_CH2_SR1,
                          &GTM_ATOM0_CH1_SR0, &GTM_ATOM0_CH1_SR1);

    CddGtm_ConfigurePhase(AppPtr->DutyV, halfPeriod, deadTimeTicks,
                          &GTM_ATOM0_CH4_SR0, &GTM_ATOM0_CH4_SR1,
                          &GTM_ATOM0_CH3_SR0, &GTM_ATOM0_CH3_SR1);

    CddGtm_ConfigurePhase(AppPtr->DutyW, halfPeriod, deadTimeTicks,
                          &GTM_ATOM0_CH6_SR0, &GTM_ATOM0_CH6_SR1,
                          &GTM_ATOM0_CH5_SR0, &GTM_ATOM0_CH5_SR1);

    /* ADC Trigger */
    GTM_ATOM0_CH7_SR0.B.SR0 = (uint32_T)((1.0F + AppPtr->DutyAdcTrig) * (real32_T)halfPeriod);
    GTM_ATOM0_CH7_SR1.B.SR1 = (uint32_T)((1.0F - AppPtr->DutyAdcTrig) * (real32_T)halfPeriod);
}

void CddGtm_InitModule(void)
{
    CddSys_ClearCpuWdtEndInit();
    GTM_CLC.B.DISR = 0x0U;       /* request clock enable                     */
    CddSys_SetCpuWdtEndInit();
    while (GTM_CLC.B.DISS != 0x0U)
    {
        CddSys_NopDelay(1U, 1U);  /* wait for clock to be running             */
    }

    /* Disable write protection on cluster configuration registers */
    GTM_CTRL.B.RF_PROT       = 0x0U;
    GTM_CCM0_PROT.B.CLS_PROT = 0x0U;

    /* Enable cluster 0 with no additional divider (UM p.122) */
    GTM_CLS_CLK_CFG.B.CLS0_CLK_DIV = 0x1U;

    /* Disable all CMU clocks, program CLK0 = 200 MHz */
    GTM_CMU_CLK_EN.U = 0x55555555U;                     /* disable all clocks */
    CddSys_SetGtmCmuClk00Freq(GTM_CMU_CLK0_FREQUENCY);  /* set CLK0 = 200 MHz */

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
 *          4. Configure ATOM0_CH0: SOMP master, TRIGOUT=1, CCU1 ISR enabled.
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

void CddGtm_InitInverter(void)
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


    EmbedSimCtrlInput_G.CtrlAlg = SIM_CTRL_DFC;  //SIM_CTRL_OPEN_LOOP; SIM_CTRL_DFC;
    EmbedSimCtrlInput_G.SwitchToClosedLoop = 0x0U;
    EmbedSim_ControlInit();


    /* Step 2 — zero-vector pre-load: 50 % duty on all phases (no net voltage)      */
    CddApp_G.DutyU       = 0.5F;
    CddApp_G.DutyV       = 0.5F;
    CddApp_G.DutyW       = 0.5F;
    CddApp_G.DutyAdcTrig = 0.8F;
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
    /* Step 5a — PHASE U LS: ATOM0_CH1  IL1 P00.2                         */
    /* ================================================================== */
    chCtrl.U            = GTM_ATOM0_CH1_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;   /* Slave: reset CN0 on master TRIGOUT */
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
    /* Step 5b — PHASE U HS: ATOM0_CH2  /IH1 P00.3  active LOW   SL=0     */
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
    /* Step 5c — PHASE V LS: ATOM0_CH3  IL2 P00.4  active HIGH  SL=0      */
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
    /* Step 5d — PHASE V HS: ATOM0_CH4  /IH2 P00.5  active LOW   SL=0     */
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
    /* Step 5e — PHASE W LS: ATOM0_CH5  IL3 P00.6  active HIGH  SL=0      */
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
    /* Step 5f — PHASE W HS: ATOM0_CH6  /IH3 P00.7  active LOW   SL=0     */
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
    /* Step 6 — ADC TRIGGER: ATOM0_CH7  P00.8 + EVADC G0/G1/G2/G3         */
    /* ================================================================== */
    /* FALLING edge (EVADC trigger event, XTMODE=1) at
     *     SR0 = HalfPeriodTicks - GTM_ADC_TRIG_LEAD_TICKS
     * = centre of the all-low-side-ON window, minus the conversion lead.
     * Rising edge (pulse start, scope reference) at SR0 - PULSE ticks.
     * NEVER place the falling edge at PeriodTicks: (a) that is the all-
     * HIGH-side V7 window (shunts carry zero current), and (b) it races
     * the slave CN0 reset from the master TRIGOUT — swallowed CCU0
     * matches cause missed/jittering triggers.                            */
    chCtrl.U            = GTM_ATOM0_CH7_CTRL.U;
    chCtrl.B.MODE       = ATOM_MODE_SOMP;
    chCtrl.B.UDMODE     = ATOM_UD_COUNT_MODE;
    chCtrl.B.RST_CCU0   = 0x1U;
    chCtrl.B.TRIGOUT    = 0x0U;
    chCtrl.B.ARU_EN     = 0x0U;
    chCtrl.B.CLK_SRC_SR = ATOM_CMU_CLK;
    chCtrl.B.SL         = 0x0U;
    GTM_ATOM0_CH7_CTRL.U    = chCtrl.U;

    /* Route CH7 (via CDTM0_DTM5_3 dead-time output, code 0x8 — identical
     * encoding for SEL0..SEL4, TC38x UM appx Table p.26-320/321) to the
     * ADC_TRIG0[x] lines; Table 292: ADC_TRIG0[x] → Gx REQTRI one-to-one. */
    GTM_ADCTRIG0OUT0.B.SEL0 = 0x8U;   /* ATOM0_CH7 → ADC_TRIG0[0] → G0 REQTRI  (U)        */
    GTM_ADCTRIG0OUT0.B.SEL3 = 0x8U;   /* ATOM0_CH7 → ADC_TRIG0[3] → G3 REQTRI  (V)        */
    GTM_ADCTRIG0OUT0.B.SEL2 = 0x8U;   /* ATOM0_CH7 → ADC_TRIG0[2] → G2 REQTRI  (W)        */
    GTM_ADCTRIG0OUT0.B.SEL1 = 0x8U;   /* ATOM0_CH7 → ADC_TRIG0[1] → G1 REQTRI  (VRO+Vdc)  */



    /* Route CH7 additionally to P00.8 (TOUT17) — scope observation of the
     * shunt sampling instant relative to the phase PWM / LS conduction window.
     * Shunt current sensing: the EVADC must sample while the low-side
     * switches conduct; the falling edge is placed at HalfPeriodTicks - LEAD (all-LS centre). */
    GTM_TOUTSEL2.B.SEL1 = TOUTSEL_GTM_ATOM;
    CddGpio_ConfigGtmPhaseADCTrigger_P00_8();

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

    //CddApp_G.CDDAppStatus = CDDAPP_RUN_STATE;
    GTM_ATOM0_AGC_GLB_CTRL.B.HOST_TRIG = 0x1U;
}




