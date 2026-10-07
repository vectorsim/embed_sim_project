/**********************************************************************************************************************
 * \file      cdd_egtm_app.c
 * \brief     Public interface for the CddSys low-level system utility module.
 *
 * \details   Declares the CddSys function API used across the CDD layer for
 *            TriCore-specific primitives:
 *              - Atomic bit-field load-modify-store (LDMST instruction).
 *              - Binary semaphore via compare-and-swap (CMPSWAP.W instruction).
 *              - Single and multi-instruction NOP delays.
 *              - Interrupt-enable intrinsic.
 *
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
 * \date      2026-10-04
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
#include "cdd_egtm_app.h"
#include "cdd_sys_util.h"
#include "cdd_app.h"
#include "IfxEGtm_reg.h"
#include "IfxEGtm_regdef.h"
#include "cdd_gpio_app.h"

#include "Bsp.h"
#include "IfxEGtm_reg.h"
#include "IfxEGtm_regdef.h"
#include "IfxEGtm_PinMap.h"
#include "IfxAdc_reg.h"
#include <math.h>
#include "IfxScuEru.h"

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Master ATOM channel output pin (ATOM CH0, P02.0).
 *          Not connected to an inverter phase; used only to generate the
 *          control-loop timing reference (CCU0 / CCU1 events).
 */
#define MASTER_CHANNEL            &IfxEgtm_ATOM0_0_TOUT0_P02_0_OUT

/**
 * \brief   Phase U high-side gate drive output (ATOM CH5, P02.5).
 */
#define U_H                       &IfxEgtm_ATOM0_5_TOUT5_P02_5_OUT

/**
 * \brief   Phase U low-side gate drive output (ATOM CH5N, P00.2).
 */
#define U_L                       &IfxEgtm_ATOM0_5N_TOUT11_P00_2_OUT

/**
 * \brief   Phase V high-side gate drive output (ATOM CH6, P02.6).
 */
#define V_H                       &IfxEgtm_ATOM0_6_TOUT6_P02_6_OUT

/**
 * \brief   Phase V low-side gate drive output (ATOM CH6N, P00.4).
 */
#define V_L                       &IfxEgtm_ATOM0_6N_TOUT13_P00_4_OUT

/**
 * \brief   Phase W high-side gate drive output (ATOM CH7, P02.7).
 */
#define W_H                       &IfxEgtm_ATOM0_7_TOUT7_P02_7_OUT

/**
 * \brief   Phase W low-side gate drive output (ATOM CH7N, P00.6).
 */
#define W_L                       &IfxEgtm_ATOM0_7N_TOUT15_P00_6_OUT

/**
 * \brief   Compute the shadow register SR0 compare value for a phase.
 *
 * \details Applies the signed-duty-cycle mapping used by the inverter:
 *            SR0 = ((1 + DutyCycle) * PeriodTicks) / 2
 *          The result is truncated to uint32.
 *
 * \param[in] PeriodTicks   PWM period in CMU_CLK0 ticks.
 * \param[in] DutyCycle     Signed duty cycle in [-1.0, +1.0].
 *
 * \return  Compare value for the phase shadow register SR0.
 *
 * \note    The arguments are evaluated more than once.
 */
#define CAL_PHASE_PERIOD(PeriodTicks, DutyCycle)      (uint32)(((1.0F + DutyCycle) * PeriodTicks)/2.0F)   /* Formula to calculate the Period     */

/**
 * \brief   Compute the shadow register SR1 compare value for a phase.
 *
 * \details Applies the signed-duty-cycle mapping used by the inverter:
 *            SR1 = ((1 - DutyCycle) * PeriodTicks) / 2
 *          The result is truncated to uint32.
 *
 * \param[in] PeriodTicks   PWM period in CMU_CLK0 ticks.
 * \param[in] DutyCycle     Signed duty cycle in [-1.0, +1.0].
 *
 * \return  Compare value for the phase shadow register SR1.
 *
 * \note    The arguments are evaluated more than once.
 */
#define CAL_PHASE_DUTY_CYCLE(PeriodTicks, DutyCycle)  (uint32)(((1.0F - DutyCycle) * PeriodTicks)/2.0F)   /* Formula to calculate the Duty Cycle */


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/



/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/
static void CddEgtm_InitAtomCls0(void);
static void CddEgtm_InitTim0Ch4(void);
static void CddEgtm_SetInverterFrequency(void);


/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/* Macro to define the Interrupt Service Routine of Control Loop (ATOM_CH0) */
IFX_INTERRUPT(ControlLoop_ISR, 0, CTRL_ISR_PRIORITY);
IFX_INTERRUPT(TIM0_CH4_ISR, 0, TIM0CH4_ISR_PRIORITY);

/**
 * \brief   Fast shut-off ISR, triggered by TIM0 channel 4.
 *
 * \details Latches a shut-off request from the TIM input (P00.1, TIM_IN4).
 *          Sets Inv_G.ShutOffState to request that the control loop
 *          de-assert the CDTM shut-off via EGTM_CLS0_CDTM_DTM5_CTRL.
 *          SHUT_OFF_RST on the next CCU0 event. Actual recovery is deferred
 *          to ControlLoop_ISR so that register updates stay synchronous with
 *          the PWM period.
 *
 * \return  void
 *
 * \note    Must remain short — it runs at the highest CDD priority
 *          (TIM0CH4_ISR_PRIORITY) and only sets a flag.
 *
 * \see     ControlLoop_ISR
 */
void TIM0_CH4_ISR(void)
{
    Inv_G.ShutOffState = 0x1U;
}


/**
 * \brief   Control-loop ISR for the inverter (ATOM channel 0, CCU0 and CCU1).
 *
 * \details Executes once per PWM period on the master channel's CCU0/CCU1
 *          events. Responsibilities:
 *            -# Increment the free-running cycle counter Inv_G.Counter.
 *            -# Recompute Inv_G.PeriodTicks from the selected frequency mode.
 *            -# On CCU0: if a shut-off is latched (ShutOffState == 1) and the
 *               application has requested recovery (ShutOffRstRequest == 1),
 *               release the dead-time module shut-off and clear the flags.
 *            -# On CCU1: recompute the three phase compare values (SR0/SR1)
 *               from Inv_G.PeriodTicks and the three duty cycles.
 *            -# On CCU1, if FrequencyChangeRequest is set, reprogram the
 *               master period, issue a host trigger to latch the shadow
 *               registers, and clear the request.
 *
 * \return  void
 *
 * \note    Both IRQ notification flags are explicitly cleared so that the
 *          service request is de-asserted before returning.
 *
 * \note    Runs at CTRL_ISR_PRIORITY on CTRL_ISR_TOS.
 *
 * \see     CddEgtm_SetInverterFrequency
 * \see     TIM0_CH4_ISR
 */
void ControlLoop_ISR(void)
{
    Inv_G.Counter++;
    CddEgtm_SetInverterFrequency();

    if(EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU0TC != 0x0u)
    {
        if((Inv_G.ShutOffState ==  0x1U)  && (Inv_G.ShutOffRstRequest == 0x1U))
        {
            EGTM_CLS0_CDTM_DTM5_CTRL.B.SHUT_OFF_RST = 0x1u;
            Inv_G.ShutOffRstRequest = 0x0U;
            Inv_G.ShutOffState  = 0x0U;
        }
        EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU0TC = 0x1u;   /* clear the notification flag */
    }

    if(EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU1TC != 0x0u)
    {

        EGTM_CLS0_ATOM_CH5_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCycleU);
        EGTM_CLS0_ATOM_CH5_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCycleU);
        EGTM_CLS0_ATOM_CH6_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCycleV);
        EGTM_CLS0_ATOM_CH6_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCycleV);
        EGTM_CLS0_ATOM_CH7_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCycleW);
        EGTM_CLS0_ATOM_CH7_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCycleW);

        if(Inv_G.FrequencyChangeRequest==0x1U)
        {
            EGTM_CLS0_ATOM_CH0_SR0.B.SR0 = Inv_G.PeriodTicks;   /* note: Master is not connected to HRPWM */
            EGTM_CLS0_ATOM_CH0_SR1.B.SR1 = Inv_G.PeriodTicks/2u;
            EGTM_CLS0_ATOM_AGC_GLB_CTRL.B.HOST_TRIG = 0x1u;
            Inv_G.FrequencyChangeRequest = 0x0U;
        }

        EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1u;   /* clear the notification flag */
    }

}


/**
 * \brief   Initialises the eGTM module for three-phase inverter control.
 *
 * \details Performs the full eGTM bring-up for the inverter:
 *            -# Enables the eGTM clock and CMU0 global clock, and programs the
 *               CMU to the CMU0 target frequency derived from the cluster
 *               clock.
 *            -# Initialises the Inv_G global with default dead time, duty
 *               cycles, shut-off state, frequency mode, and CMU0 frequency.
 *            -# Computes Inv_G.PeriodTicks for the selected frequency mode.
 *            -# Configures the TIM0 channel 4 fast shut-off path.
 *            -# Configures the ATOM CLS0 master channel and the three
 *               complementary PWM phases with dead time.
 *
 * \return  void
 *
 * \post    eGTM is configured but outputs remain inactive until the host
 *          trigger is issued by CddEgtm_Start().
 *
 * \see     CddEgtm_InitTim0Ch4
 * \see     CddEgtm_InitAtomCls0
 * \see     CddEgtm_Start
 */
void CddEgtm_InitModule(void)
{
    real32_T frequency;

    frequency = 0.0F;


    /* Enable EGTM  & Set CMU0 Clk Frequency */
    CddSys_EnableEgtmClock();
    frequency = CddSys_GetEgtmCmuFrequency();
    CddSys_SetEgtmCmuFrequency(frequency);
    CddSys_SetEgtmCmu0Frequency(frequency);
    CddSys_EnableEgtmCmu0Clock();

    Inv_G.DeadTimeTicks    = 100;
    Inv_G.DutyCycleU       = 0.8F;
    Inv_G.DutyCycleV       = 0.6F;
    Inv_G.DutyCycleW       = 0.7F;
    Inv_G.ShutOffState     = 0x0U;
    Inv_G.ShutOffRstRequest = 0x0U;
    Inv_G.FrequencyChangeRequest = 0x0U;
    Inv_G.FrequencyMode     = 0x3U;
    Inv_G.CmuClk0Frequency  = CddSys_GetEgtmCmu0Frequency();
    CddEgtm_SetInverterFrequency();

    /* Init ShutOff */
    CddEgtm_InitTim0Ch4();
    /* Init Atom for PWM */
    CddEgtm_InitAtomCls0();

}


/**
 * \brief   Initialises eGTM TIM0 channel 4 as the fast shut-off input.
 *
 * \details Configures TIM channel 4 in TIEM mode with no adjacent-channel
 *          pairing and no LUT, routing pin P00.1 (TIM_IN4) as the input.
 *          NEWVAL events are enabled and routed to CPU0 at
 *          TIM0CH4_ISR_PRIORITY with a pulse IRQ mode. The channel is
 *          enabled at the end of the function.
 *
 *          Steps:
 *            -# Read-modify-write the shared and per-channel registers.
 *            -# Route P00.1 to TIM_IN4 via the pin map.
 *            -# Set TIEM mode, no channel pairing, GPR/DSL/ISL selection.
 *            -# Disable the LUT.
 *            -# Enable NEWVAL IRQ with pulse mode.
 *            -# Configure and enable the service request node.
 *            -# Write back and enable the channel.
 *
 * \return  void
 *
 * \note    The TIM_IN_SRC register is shared across all eight TIM channels,
 *          so the read-modify-write of MODE4/VAL4 preserves the settings of
 *          the other channels.
 *
 * \see     TIM0_CH4_ISR
 */
/* This function initializes GTM TIM0_CH4
 * - P00.1 (TIM_IN4) is used as the channel input
 * - no LUT, no channel pairing (CICTRL = 0)
 */
void CddEgtm_InitTim0Ch4(void)
{
    Ifx_EGTM_CLS_TIM_CH_CTRL  timChCtrl;
    Ifx_EGTM_CLS_TIM_CH_ECTRL timChExtCtrl;
    Ifx_EGTM_CLS_TIM_IN_SRC   timInSrc;
    Ifx_SRC_SRCR              srcCfg;

    /* 1. Read-modify-write the shared / per-channel registers */
    timChCtrl.U    = EGTM_CLS0_TIM_CH4_CTRL.U;    /* UM P.8944 */
    timChExtCtrl.U = EGTM_CLS0_TIM_CH4_ECTRL.U;   /* UM P.8957 */
    timInSrc.U     = EGTM_CLS0_TIM_IN_SRC.U;      /* UM P.8975 (shared across all 8 TIM channels) */
    srcCfg.U       =  SRC_EGTMTIM0SR4.U;

    /* 2. Route P00.1 to TIM_IN4 */
    IfxEgtm_PinMap_setTimTin(&IfxEgtm_TIM0_4_P00_1_IN, IfxPort_InputMode_pullDown);

    /* 3. Channel control: TIEM mode, no pairing with adjacent channel */
    timChCtrl.B.TIM_MODE   = 0x2u;   /* TIEM mode                              */
    timChCtrl.B.CICTRL     = 0x0u;   /* do NOT combine with adjacent channel   */
    timChCtrl.B.GPR0_SEL   = 0x3U;
    timChCtrl.B.GPR1_SEL   = 0x3U;
    timChCtrl.B.DSL   = 0x1U;
    timChCtrl.B.ISL   = 0x1U;
                                     /* -> uses TIM_IN(x) = TIM_IN4            */
    timInSrc.B.MODE4       = 0x0u;   /* not forced to constant                 */
    timInSrc.B.VAL4        = 0x0u;   /* not forced to constant                 */

    /* 4. Do not use the LUT */
    timChExtCtrl.B.USE_LUT = 0x0u;

   // EGTM_CLS0_TIM_CH4_IRQ_NOTIFY.B.NEWVAL = 0x1U;
    EGTM_CLS0_TIM_CH4_IRQ_EN.B.NEWVAL_IRQ_EN = 0x1U,
    EGTM_CLS0_TIM_CH4_IRQ_MODE.B.IRQ_MODE = 0x2u;   /* pulse */


    srcCfg.B.TOS  = TIM0CH4_ISR_TOS;
    srcCfg.B.SRPN = TIM0CH4_ISR_PRIORITY;
    SRC_EGTMTIM0SR4.U = srcCfg.U;
    SRC_EGTMTIM0SR4.B.SRE = 0x1U;

    /* 6. Write back */
    EGTM_CLS0_TIM_CH4_CTRL.U  = timChCtrl.U;
    EGTM_CLS0_TIM_CH4_ECTRL.U = timChExtCtrl.U;
    EGTM_CLS0_TIM_IN_SRC.U    = timInSrc.U;

    EGTM_CLS0_TIM_CH4_CTRL.B.TIM_EN = 0x1U;

}

/**
 * \brief   Computes the ATOM master period from the selected frequency mode.
 *
 * \details Maps Inv_G.FrequencyMode to an ATOM period in ticks:
 *            1U -> 10 kHz, 2U -> 20 kHz, 3U -> 30 kHz, default -> 10 kHz.
 *          The period is derived as CmuClk0Frequency / target_frequency and
 *          stored in Inv_G.PeriodTicks.
 *
 * \return  void
 *
 * \pre     Inv_G.CmuClk0Frequency has been set from
 *          CddSys_GetEgtmCmu0Frequency().
 *
 * \note    Any frequency mode outside {1, 2, 3} silently falls back to
 *          10 kHz; there is no error reporting.
 */
void CddEgtm_SetInverterFrequency(void)
{
    switch(Inv_G.FrequencyMode)
    {
        case 1U:
            Inv_G.PeriodTicks      = (uint32)(Inv_G.CmuClk0Frequency/  KHZ_10);
            break;
        case 2U:
            Inv_G.PeriodTicks      = (uint32)(Inv_G.CmuClk0Frequency/  KHZ_20);
            break;
        case 3U:
            Inv_G.PeriodTicks      = (uint32)(Inv_G.CmuClk0Frequency/  KHZ_30);
            break;
        default:
            Inv_G.PeriodTicks      = (uint32)(Inv_G.CmuClk0Frequency/  KHZ_10);
            break;
    }
}

/**
 * \brief   Configures ATOM cluster 0 for three-phase inverter PWM generation.
 *
 * \details Full ATOM channel setup:
 *            -# DTM4/DTM5 clock select, update mode, and shadow-register
 *               update enable.
 *            -# PSU input select and TIM select for the shut-off path.
 *            -# ATOM CH0 as master in SOMP mode: period and half-period in
 *               the shadow registers, CCU0/CCU1 IRQ enabled and routed to
 *               CPU0, output pin enabled.
 *            -# ATOM CH5/CH6/CH7 as phases U/V/W in SOMP mode, each reset by
 *               the master's CCU0 event, with per-phase dead time
 *               (DTV1 / DTV2 / DTV3) and complementary high/low outputs.
 *            -# Shut-off safe levels for each phase
 *               (OC0_x_SR = 1, OC1_x_SR = 1, SL0_x_SR_SR = 0, SL1_x_SR_SR = 0).
 *            -# Global AGC registers enabling shadow update, force update,
 *               channel enable, and output enable for all used channels.
 *
 *          Channel map:
 *            - ATOM CH0  : master timing reference (P02.0), not a gate drive.
 *            - ATOM CH5  : phase U high side (P02.5).
 *            - ATOM CH5N : phase U low  side (P00.2).
 *            - ATOM CH6  : phase V high side (P02.6).
 *            - ATOM CH6N : phase V low  side (P00.4).
 *            - ATOM CH7  : phase W high side (P02.7).
 *            - ATOM CH7N : phase W low  side (P00.6).
 *
 * \return  void
 *
 * \pre     CddSys_EnableEgtmClock(), CMU0 configuration and
 *          Inv_G.PeriodTicks / Inv_G.DeadTimeTicks have been set by
 *          CddEgtm_InitModule().
 * \post    All channels are configured but not running; a host trigger is
 *          required to start them (issued by CddEgtm_Start()).
 *
 * \note    The dead-time control word dtm5ChCtrl2 accumulates the DT0_x/DT1_x
 *          enables for all three phases and is written back once. This
 *          relies on the read at function entry having those bits clear;
 *          call this function only on an unconfigured DTM5.
 *
 * \see     CddEgtm_Start
 */
 /* This function initializes the EGTM_CLS0_ATOM
  * - configures CLS0_ATOM_CH0 (P02.00) as Master Channel and sets-up CCU0 & CCU1 interrupt event to facilitate Control-loop
  * - configures CLS0_ATOM_CH5     (P02.05) as Phase U_H
  * - configures CLS0_ATOM_CH5_N   (P00_02) as Phase U_L
  * - configures CLS0_ATOM_CH6     (P02.06) as Phase V_H
  * - configures CLS0_ATOM_CH6_N   (P00_04) as Phase V_L
  * - configures CLS0_ATOM_CH7     (P02.07) as Phase W_H
  * - configures CLS0_ATOM_CH7_N   (P00_06) as Phase W_L
*/
 void CddEgtm_InitAtomCls0(void)
 {

     Ifx_EGTM_CLS_ATOM_CH_CTRL         chCtrl;
     Ifx_EGTM_CLS_CDTM_DTM_CH_CTRL2    dtm4ChCtrl2;
     Ifx_EGTM_CLS_CDTM_DTM_CTRL        dtm4Ctrl;
     Ifx_EGTM_CLS_CDTM_DTM_CH_CTRL2    dtm5ChCtrl2;
     Ifx_EGTM_CLS_CDTM_DTM_CTRL        dtm5Ctrl;
     Ifx_EGTM_CLS_CDTM_DTM_CH_DTV      chDTV;
     Ifx_EGTM_CLS_CDTM_DTM_CH_CTRL2_SR dtm5ChCtrl2Sr;
     Ifx_EGTM_CLS_CDTM_DTM_CH_SR       dtm5CtrlChSr;
     Ifx_SRC_SRCR                      srcCfg;
     Ifx_EGTM_CLS_ATOM_CH_IRQ_EN       chIrqEn;
     Ifx_EGTM_CLS_CDTM_DTM_PS_CTRL     psCtrl;


     /* Copy Control Registers to Interim Structures */
     dtm4Ctrl.U    = EGTM_CLS0_CDTM_DTM4_CTRL.U;
     dtm4ChCtrl2.U = EGTM_CLS0_CDTM_DTM4_CH_CTRL2.U;
     dtm5Ctrl.U    = EGTM_CLS0_CDTM_DTM5_CTRL.U;
     dtm5ChCtrl2.U = EGTM_CLS0_CDTM_DTM5_CH_CTRL2.U;
     dtm5ChCtrl2Sr.U = EGTM_CLS0_CDTM_DTM5_CH_CTRL2_SR.U;
     dtm5CtrlChSr.U  = EGTM_CLS0_CDTM_DTM5_CH_SR.U;
     psCtrl.U        = EGTM_CLS0_CDTM_DTM5_PS_CTRL.U;

     /* Set Clock for DTM modules */
     /* Configure Overall Dead Time Module CDTM_DTM5 for Phases (U, V, W) */
     dtm4Ctrl.B.CLK_SEL       = 0x1U;   /* select CMU_CLk0                              */
     dtm5Ctrl.B.CLK_SEL       = 0x1U;   /* select CMU_CLk0                              */
     dtm5Ctrl.B.UPD_MODE      = 0x1U;   /* release shut off by writing the SHUT_OFF_RST */
     dtm5Ctrl.B.SR_UPD_EN     = 0x1U;   /* allow shadow register update                 */
     /* Configure PSU to send shut-off signal to the phases (U, V, W) */
     psCtrl.B.PSU_IN_SEL     = 0x0U;
     psCtrl.B.TIM_SEL        = 0x1U;   /* select TIM_CH_IN0 or select TIM_CH_IN1       */


     /* Configure ATOM_CH0 as Master  */
     /* M1. Configure Channel Control */
     chCtrl.U                  = EGTM_CLS0_ATOM_CH0_CTRL.U;
     chCtrl.B.MODE             = 0x2U;   /* set in SOMP Mode                       */
     chCtrl.B.UDMODE           = 0x0U;   /* enable                                 */
     chCtrl.B.CLK_SRC          = 0x0U;   /* select CMU_CLK0                        */
     chCtrl.B.TRIGOUT          = 0x1U;   /* expose CCU0 event to slave channels    */
     chCtrl.B.RST_CCU0         = 0x0U;   /* reset the counter on it own CCU0 event */
     chCtrl.B.SL               = 0x1U;
     EGTM_CLS0_ATOM_CH0_CTRL.U = chCtrl.U;

     /* M2. Specify Period and Duty Cycle of PWM Signal (Shadow Register) */
     EGTM_CLS0_ATOM_CH0_SR0.B.SR0 = Inv_G.PeriodTicks;   /* note: Master is not connected to HRPWM */
     EGTM_CLS0_ATOM_CH0_SR1.B.SR1 = Inv_G.PeriodTicks/2u;

     /* M3. Configure Output */
     CddGpio_ConfigEgtmAtomMasterP02_00();

     /* M4. Setup Interrupt Service Request to CPU0 */
     chIrqEn.U                              = EGTM_CLS0_ATOM_CH0_IRQ_EN.U;
     chIrqEn.B.CCU0TC_IRQ_EN                = 0x1U;   /* enable CCU0 Event */
     chIrqEn.B.CCU1TC_IRQ_EN                = 0x1U;   /* enable CCU1 Event */
     EGTM_CLS0_ATOM_CH0_IRQ_EN.U            = chIrqEn.U;
     EGTM_CLS0_ATOM_CH0_IRQ_MODE.B.IRQ_MODE = 0x2U;
     srcCfg.U                               = SRC_EGTMATOM0SR0.U;
     srcCfg.B.SRPN                          = CTRL_ISR_PRIORITY;   /* assign Service Request Priority */
     srcCfg.B.TOS                           = CTRL_ISR_TOS;
     SRC_EGTMATOM0SR0.U                     = srcCfg.U;
     SRC_EGTMATOM0SR0.B.SRE                 = 0x1U;                /* enable Service Request          */


     /* Configure ATOM_CH5 as Phase U */
     /* U1. Configure Channel Control */
     chCtrl.U                  = EGTM_CLS0_ATOM_CH5_CTRL.U;
     chCtrl.B.MODE             = 0x2U;   /* set in SOMP Mode                                      */
     chCtrl.B.UDMODE           = 0x0U;   /* enable Up Count Mode                                  */
     chCtrl.B.CLK_SRC          = 0x0U;   /* select CMU_CLK0                                       */
     chCtrl.B.TRIGOUT          = 0x0U;   /* disable own counter reset                             */
     chCtrl.B.RST_CCU0         = 0x1U;   /* reset the counter on the event CCU0 of Master Channel */
     EGTM_CLS0_ATOM_CH5_CTRL.U = chCtrl.U;

     /* U2. Specify Period and Duty Cycle of PWM Signal (Shadow Register) */
     EGTM_CLS0_ATOM_CH5_SR0.B.SR0 = Inv_G.PeriodTicks;
     EGTM_CLS0_ATOM_CH5_SR1.B.SR1 = Inv_G.PeriodTicks/2u;

     /* U3. Dead Time Configuration */
     chDTV.U         = EGTM_CLS0_CDTM_DTM5_CH_DTV1.U;
     chDTV.B.RELFALL = Inv_G.DeadTimeTicks;    /* set the dead time for falling edge    */
     chDTV.B.RELRISE = Inv_G.DeadTimeTicks;   /* set the dead time for rising  edge    */
     EGTM_CLS0_CDTM_DTM5_CH_DTV1.U = chDTV.U;
     dtm5ChCtrl2.B.DT0_1 = 0x1U;   /* enable dead time for high-switch */
     dtm5ChCtrl2.B.DT1_1 = 0x1U;   /* enable dead time for low-switch  */

     /* U4. Connect the Channel to Output Pin*/
     CddGpio_ConfigEgtmAtomUhP02_05();
     CddGpio_ConfigEgtmAtomUlP00_02();
     //IfxEgtm_PinMap_setAtomTout(U_L, IfxPort_Mode_outputPushPullGeneral, IfxPort_PadDriver_cmosAutomotiveSpeed1);

     /* U5. Configure Signal-Level  during the Shut-off in Shadow Register */
     dtm5ChCtrl2Sr.B.OC0_1_SR   = 0x1U;
     dtm5ChCtrl2Sr.B.OC1_1_SR   = 0x1U;
     dtm5CtrlChSr.B.SL0_1_SR_SR = 0x0U;
     dtm5CtrlChSr.B.SL1_1_SR_SR = 0x0U;

     /* Configure ATOM_CH6 as Phase V */
     /* V1. Configure Channel Control */
     chCtrl.U                  = EGTM_CLS0_ATOM_CH6_CTRL.U;
     chCtrl.B.MODE             = 0x2U;   /* set in SOMP Mode                                      */
     chCtrl.B.UDMODE           = 0x0U;   /* enable Up Count Mode                                  */
     chCtrl.B.CLK_SRC          = 0x0U;   /* select CMU_CLK0                                       */
     chCtrl.B.TRIGOUT          = 0x0U;   /* disable own counter reset                             */
     chCtrl.B.RST_CCU0         = 0x1U;   /* reset the counter on the event CCU0 of Master Channel */
     EGTM_CLS0_ATOM_CH6_CTRL.U = chCtrl.U;

     /* V2. Specify Period and Duty Cycle of PWM Signal (Shadow Register) */
     EGTM_CLS0_ATOM_CH6_SR0.B.SR0 = Inv_G.PeriodTicks;
     EGTM_CLS0_ATOM_CH6_SR1.B.SR1 = Inv_G.PeriodTicks/2u;

     /* V3. Dead Time Configuration */
     chDTV.U         = EGTM_CLS0_CDTM_DTM5_CH_DTV2.U;
     chDTV.B.RELFALL = Inv_G.DeadTimeTicks;    /* set the dead time for falling edge    */
     chDTV.B.RELRISE = Inv_G.DeadTimeTicks;    /* set the dead time for rising  edge    */
     EGTM_CLS0_CDTM_DTM5_CH_DTV2.U = chDTV.U;
     dtm5ChCtrl2.B.DT0_2 = 0x1U;   /* enable dead time for high-switch */
     dtm5ChCtrl2.B.DT1_2 = 0x1U;   /* enable dead time for low-switch  */

     /* V4. Connect the Channel to Output Pin*/
     CddGpio_ConfigEgtmAtomVhP02_06();
     CddGpio_ConfigEgtmAtomVlP00_04();

     /* V5. Configure Signal-Level  during the Shut-off in Shadow Register */
     dtm5ChCtrl2Sr.B.OC0_2_SR   = 0x1U;
     dtm5ChCtrl2Sr.B.OC1_2_SR   = 0x1U;
     dtm5CtrlChSr.B.SL0_2_SR_SR = 0x0U;
     dtm5CtrlChSr.B.SL1_2_SR_SR = 0x0U;

     /* Configure ATOM_CH7 as Phase W */
     /* W1. Configure Channel Control */
     chCtrl.U                  = EGTM_CLS0_ATOM_CH7_CTRL.U;
     chCtrl.B.MODE             = 0x2U;   /* set in SOMP Mode                                      */
     chCtrl.B.UDMODE           = 0x0U;   /* enable Up Count Mode                                  */
     chCtrl.B.CLK_SRC          = 0x0U;   /* select CMU_CLK0                                       */
     chCtrl.B.TRIGOUT          = 0x0U;   /* disable own counter reset                             */
     chCtrl.B.RST_CCU0         = 0x1U;   /* reset the counter on the event CCU0 of Master Channel */
     EGTM_CLS0_ATOM_CH7_CTRL.U = chCtrl.U;

     /* W2. Specify Period and Duty Cycle of PWM Signal (Shadow Register) */
     EGTM_CLS0_ATOM_CH7_SR0.B.SR0 = Inv_G.PeriodTicks;
     EGTM_CLS0_ATOM_CH7_SR1.B.SR1 = Inv_G.PeriodTicks/2u;

     /* W3. Dead Time Configuration */
     chDTV.U         = EGTM_CLS0_CDTM_DTM5_CH_DTV3.U;
     chDTV.B.RELFALL = Inv_G.DeadTimeTicks;    /* set the dead time for falling edge    */
     chDTV.B.RELRISE = Inv_G.DeadTimeTicks;   /* set the dead time for rising  edge    */
     EGTM_CLS0_CDTM_DTM5_CH_DTV3.U = chDTV.U;
     dtm5ChCtrl2.B.DT0_3 = 0x1U;   /* enable dead time for high-switch */
     dtm5ChCtrl2.B.DT1_3 = 0x1U;   /* enable dead time for low-switch  */

     /* W4. Connect the Channel to Output Pin*/
     CddGpio_ConfigEgtmAtomWhP02_07();
     CddGpio_ConfigEgtmAtomWlP00_06();

     /* W5. Configure Signal-Level  during the Shut-off in Shadow Register */
     dtm5ChCtrl2Sr.B.OC0_3_SR   = 0x1U;
     dtm5ChCtrl2Sr.B.OC1_3_SR   = 0x1U;
     dtm5CtrlChSr.B.SL0_3_SR_SR = 0x0U;
     dtm5CtrlChSr.B.SL1_3_SR_SR = 0x0U;

     /* Write back the Configurations to its corresponding Control Registers */
     EGTM_CLS0_CDTM_DTM5_PS_CTRL.U     = psCtrl.U;
     EGTM_CLS0_CDTM_DTM4_CTRL.U        = dtm4Ctrl.U;
     EGTM_CLS0_CDTM_DTM4_CH_CTRL2.U    = dtm4ChCtrl2.U;
     EGTM_CLS0_CDTM_DTM5_CTRL.U        = dtm5Ctrl.U;
     EGTM_CLS0_CDTM_DTM5_CH_CTRL2.U    = dtm5ChCtrl2.U;
     EGTM_CLS0_CDTM_DTM5_CH_CTRL2_SR.U = dtm5ChCtrl2Sr.U;
     EGTM_CLS0_CDTM_DTM5_CH_SR.U       = dtm5CtrlChSr.U;

     /* Overall Configuration */
     EGTM_CLS0_ATOM_AGC_GLB_CTRL.U   = 0xAAAA0000U;   /* enable update shadow registers of CH0, CH1, CH2,CH3,CH4, CH5, CH6, CH7 */
     EGTM_CLS0_ATOM_AGC_FUPD_CTRL.U  = 0x0000AAAAU;   /* enable force update on channels CH0, CH1, CH5, CH6, CH7   */
     EGTM_CLS0_ATOM_AGC_ENDIS_CTRL.U = 0x0000AAAAU;   /* enable channels CH0, CH1, CH5, CH6, CH7                   */
     EGTM_CLS0_ATOM_AGC_OUTEN_CTRL.U = 0x0000AAAAU;   /* enable output of CH0, CH1, CH5, CH6, CH7                  */

    // EGTM_CLS0_ATOM_AGC_GLB_CTRL.U  = 0xA8090000u;   /* disable update for master channel            */
    // EGTM_CLS0_ATOM_AGC_FUPD_CTRL.U = 0x00005405u;
 }


/**
 * \brief   Starts all enabled eGTM ATOM channels synchronously.
 *
 * \details Sets HOST_TRIG in EGTM_CLS0_ATOM_AGC_GLB_CTRL, which latches the
 *          shadow registers and starts every enabled channel at the same
 *          eGTM clock edge. From this point the master channel drives the
 *          control-loop ISR and the PWM phases drive the gate outputs.
 *
 * \return  void
 *
 * \pre     CddEgtm_InitModule() has completed.
 *
 * \see     CddEgtm_InitModule
 * \see     CddEgtm_InitAtomCls0
 */
void CddEgtm_Start(void)
{
    EGTM_CLS0_ATOM_AGC_GLB_CTRL.B.HOST_TRIG = 0x1u;  /* set the trigger request to start all channels synchronously */

}

