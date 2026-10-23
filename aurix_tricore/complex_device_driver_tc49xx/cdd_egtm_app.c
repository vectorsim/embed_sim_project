/**********************************************************************************************************************
 * \file      cdd_egtm_app.c
 * \brief     Implementation of the CddEgtm application module (eGTM inverter control).
 *
 * \details   Provides the full software path that configures and runs a
 *            three-phase inverter PWM generator on the Infineon AURIX eGTM:
 *              - CMU clock enable and frequency programming.
 *              - ATOM cluster 0 master (CH0) + phases U/V/W (CH5/CH6/CH7)
 *                with complementary outputs and hardware dead-time.
 *              - TIM0 channel 4 fast shut-off input (P00.1) that latches a
 *                shut-off request for the control-loop ISR.
 *              - Control-loop ISR that updates period, duty-cycle shadow
 *                registers and recovers from shut-off on CCU0/CCU1 events.
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
 * \note      MISRA C:2012 compliance overview for this translation unit:
 *              - Rule 2.1   : No unreachable code.
 *              - Rule 8.7   : Functions with external linkage are declared in
 *                            the corresponding header.
 *              - Rule 8.9   : Objects are defined at the narrowest scope.
 *              - Rule 10.1  / 10.3 / 10.4 : Explicit casts are used where
 *                            signed/unsigned or floating-point conversions
 *                            occur (see CAL_PHASE_* macros).
 *              - Rule 12.1  : Parentheses enforce evaluation order in macros.
 *              - Rule 13.5  : Side-effect free expressions in logical operators
 *                            (ShutOffState / FrequencyChangeRequest checks).
 *              - Rule 17.7  : Return values of void functions are not used.
 *              - Directive 4.9 : Function-like macros are documented; the
 *                            multiple-evaluation note is recorded for each.
 *              - Rule 11.4  / 11.6 : Register access is performed through the
 *                            vendor-supplied IfxEGtm_reg / regdef types; no
 *                            additional pointer casts are introduced.
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
#include "cdd_adc_app.h"

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
 *          MISRA C:2012 Directive 4.9 – function-like macro; caller must
 *          ensure that PeriodTicks and DutyCycle are free of side-effects.
 *          MISRA C:2012 Rule 10.3 / 10.4 – explicit (uint32) cast after
 *          floating-point arithmetic is intentional and documented.
 */
#define CAL_PHASE_PERIOD(PeriodTicks, DutyCycle)      (uint32)(((1.0F + DutyCycle) * PeriodTicks)/2.0F)

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
 *          MISRA C:2012 Directive 4.9 – function-like macro; caller must
 *          ensure that PeriodTicks and DutyCycle are free of side-effects.
 *          MISRA C:2012 Rule 10.3 / 10.4 – explicit (uint32) cast after
 *          floating-point arithmetic is intentional and documented.
 */
#define CAL_PHASE_DUTY_CYCLE(PeriodTicks, DutyCycle)  (uint32)(((1.0F - DutyCycle) * PeriodTicks)/2.0F)

/**
 * \brief   Compile-time switch that enables the TIM0 CH4 fast-shut-off path
 *          on port pin P00_01.
 *
 * \details When set to 1 the initialisation sequence configures TIM0 channel 4
 *          and routes the associated ISR.  When set to 0 the path is omitted.
 *
 * \note    MISRA C:2012 Rule 2.5 – macro is used for conditional compilation;
 *          the unused branch is intentionally excluded from the build.
 */
#define  CDD_TIM0_CH4_ON   (0x1U)


/*********************************************************************************************************************/
/*-------------------------------------------------Global variables--------------------------------------------------*/
/*********************************************************************************************************************/

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/



/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/
/**
 * \brief   Private helper that configures ATOM cluster 0 for inverter PWM.
 *
 * \details Sets up the master channel (CH0) and the three phase channels
 *          (CH5/CH6/CH7) including dead-time modules, pin routing and
 *          shut-off safe levels.  Called only from CddEgtm_Init().
 *
 * \note    MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 */
static void CddEgtm_Init_AtomCls0_Inv(void);

/**
 * \brief   Private helper that configures TIM0 channel 4 as fast shut-off input.
 *
 * \details Programs TIEM mode, pin map for P00.1, NEWVAL IRQ and service-request
 *          node.  Called only from CddEgtm_Init() when CDD_TIM0_CH4_ON is set.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 */
static void CddEgtm_Init_Tim0_Ch4_P00_01_FS(void);

/**
 * \brief   Private helper that maps FrequencyMode to PeriodTicks.
 *
 * \details Converts the selected frequency mode (10/20/30 kHz) into the
 *          corresponding ATOM period in CMU_CLK0 ticks and stores the result
 *          in Inv_G.PeriodTicks.  Called from the control-loop ISR and from
 *          CddEgtm_Init().
 *
 * \note    MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 */
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
 * \details Latches a shut-off request originating from the TIM input pin
 *          P00.1 (TIM_IN4).  The only action performed is to set
 *          Inv_G.ShutOffState = 1.  Actual de-assertion of the CDTM shut-off
 *          (via EGTM_CLS0_CDTM_DTM5_CTRL.SHUT_OFF_RST) is deferred to the
 *          next CCU0 event inside ControlLoop_ISR so that register updates
 *          remain synchronous with the PWM period.
 *
 * \return  void
 *
 * \note    Must remain extremely short — it executes at the highest CDD
 *          priority (TIM0CH4_ISR_PRIORITY) and must not perform any register
 *          writes other than the single flag assignment.
 *
 * \note    MISRA C:2012 Rule 8.7 – ISR has external linkage because it is
 *          referenced by the vector table generated by IFX_INTERRUPT.
 *          MISRA C:2012 Rule 17.7 – return value is void; no further action.
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
 * \details Executes once per PWM period on the master channel's CCU0 and/or
 *          CCU1 events.  Responsibilities in execution order:
 *            -# Increment the free-running cycle counter Inv_G.Counter.
 *            -# Recompute Inv_G.PeriodTicks from the currently selected
 *               frequency mode via CddEgtm_SetInverterFrequency().
 *            -# On CCU0 event: if a shut-off is latched (ShutOffState == 1)
 *               and the application has requested recovery
 *               (ShutOffRstRequest == 1), release the dead-time module
 *               shut-off by writing SHUT_OFF_RST and clear both flags.
 *            -# On CCU1 event: recompute the three phase compare values
 *               (SR0/SR1) for channels CH5/CH6/CH7 from the current period
 *               and the three duty-cycle variables.
 *            -# On CCU1 event, if FrequencyChangeRequest is set, reprogram
 *               the master period (SR0/SR1), issue a host trigger to latch
 *               the shadow registers, and clear the request flag.
 *            -# Explicitly clear the CCU0TC / CCU1TC notification flags so
 *               that the service request is de-asserted before return.
 *
 * \return  void
 *
 * \note    Both IRQ notification flags are written with 1 to clear them
 *          (hardware clear-on-write-1 behaviour of the eGTM).
 *
 * \note    Runs at CTRL_ISR_PRIORITY on the service-request node selected by
 *          CTRL_ISR_TOS.
 *
 * \note    MISRA C:2012 Rule 8.7 – ISR has external linkage because it is
 *          referenced by the vector table.
 *          MISRA C:2012 Rule 13.5 – the logical expressions testing
 *          ShutOffState and FrequencyChangeRequest contain no side-effects.
 *          MISRA C:2012 Rule 10.1 – the comparison against 0x0u / 0x1U uses
 *          the same essential type (unsigned).
 *
 * \see     CddEgtm_SetInverterFrequency
 * \see     TIM0_CH4_ISR
 */
void ControlLoop_ISR(void)
{
    Inv_G.Counter++;
    /* Set Operator Frequency */
    CddEgtm_SetInverterFrequency();
    /* Service Watchdog */
    CddSys_ServiceCpuWdt();


    if(EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU0TC != 0x0u)
    {
        if((Inv_G.ShutOffState ==  0x1U)  && (Inv_G.ShutOffRstRequest == 0x1U))
        {
            EGTM_CLS0_CDTM_DTM5_CTRL.B.SHUT_OFF_RST = 0x1U;
            Inv_G.ShutOffRstRequest = 0x0U;
            Inv_G.ShutOffState  = 0x0U;
        }
        EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU0TC = 0x1U;   /* clear the notification flag */
    }

    if(EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU1TC != 0x0U)
    {
        CddAdc_ReadSensor(&Inv_G);

        EGTM_CLS0_ATOM_CH1_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCycleAdcTr);
        EGTM_CLS0_ATOM_CH1_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCycleAdcTr);
        EGTM_CLS0_ATOM_CH5_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCyclePhaseU);
        EGTM_CLS0_ATOM_CH5_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCyclePhaseU);
        EGTM_CLS0_ATOM_CH6_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCyclePhaseV);
        EGTM_CLS0_ATOM_CH6_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCyclePhaseV);
        EGTM_CLS0_ATOM_CH7_SR0.B.SR0 = CAL_PHASE_PERIOD(Inv_G.PeriodTicks, Inv_G.DutyCyclePhaseW);
        EGTM_CLS0_ATOM_CH7_SR1.B.SR1 = CAL_PHASE_DUTY_CYCLE(Inv_G.PeriodTicks, Inv_G.DutyCyclePhaseW);

        if(Inv_G.FrequencyChangeRequest == 0x1U)
        {
            EGTM_CLS0_ATOM_CH0_SR0.B.SR0 = Inv_G.PeriodTicks;
            EGTM_CLS0_ATOM_CH0_SR1.B.SR1 = Inv_G.PeriodTicks/2u;
            EGTM_CLS0_ATOM_AGC_GLB_CTRL.B.HOST_TRIG = 0x1u;
            Inv_G.FrequencyChangeRequest = 0x0U;
        }

        EGTM_CLS0_ATOM_CH0_IRQ_NOTIFY.B.CCU1TC = 0x1U;   /* clear the notification flag */
    }
}

/**
 * \brief   Initialises the eGTM module for three-phase inverter control.
 *
 * \details Performs the full eGTM bring-up for the inverter in the following
 *          sequential steps:
 *            -# Enables the eGTM clock, reads the cluster frequency, programs
 *               the CMU global and CMU0 clocks, then enables CMU0.
 *            -# Initialises the Inv_G global with default dead-time
 *               (100 ticks), duty cycles (0.8 / 0.6 / 0.7), shut-off state
 *               flags cleared, frequency mode 3 (30 kHz) and the measured
 *               CMU0 frequency.
 *            -# Computes Inv_G.PeriodTicks for the selected frequency mode.
 *            -# Conditionally configures the TIM0 channel 4 fast shut-off
 *               path when CDD_TIM0_CH4_ON is defined as 1.
 *            -# Configures the ATOM CLS0 master channel and the three
 *               complementary PWM phases with hardware dead-time.
 *
 * \return  void
 *
 * \post    Hardware is fully configured; outputs remain inactive until
 *          CddEgtm_Start() is called.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in cdd_egtm_app.h.
 *          MISRA C:2012 Rule 8.9 – local variable frequency is defined at
 *          the narrowest scope.
 *          MISRA C:2012 Rule 2.1 – the conditional compilation of the TIM
 *          path produces no unreachable code in either configuration.
 *
 * \see     CddEgtm_Init_Tim0_Ch4_P00_01_FS
 * \see     CddEgtm_Init_AtomCls0_Inv
 * \see     CddEgtm_Start
 */
void CddEgtm_Init(void)
{
    real32_T frequency;

    frequency = 0.0F;

    /* Enable EGTM  & Set CMU0 Clk Frequency */
    CddSys_EnableEgtmClock();
    frequency = CddSys_GetEgtmCmuFrequency();
    CddSys_SetEgtmCmuFrequency(frequency);
    CddSys_SetEgtmCmu0Frequency(frequency);
    CddSys_EnableEgtmCmu0Clock();

    Inv_G.DeadTimeTicks    = 50U;
    Inv_G.DutyCyclePhaseU  = 0.8F;       /* Duty Cycle of Phase U    */
    Inv_G.DutyCyclePhaseV  = 0.6F;       /* Duty Cycle of Phase V    */
    Inv_G.DutyCyclePhaseW  = 0.3F;       /* Duty Cycle of Phase W    */
    Inv_G.DutyCycleAdcTr   = 0.9F;
    Inv_G.ShutOffState     = 0x0U;
    Inv_G.ShutOffRstRequest = 0x0U;
    Inv_G.FrequencyChangeRequest = 0x0U;
    Inv_G.FrequencyMode     = 0x3U;
    Inv_G.CmuClk0Frequency  = CddSys_GetEgtmCmu0Frequency();
    CddEgtm_SetInverterFrequency();

    /* Init ShutOff */
#if CDD_TIM0_CH4_ON == 0x1U
    CddEgtm_Init_Tim0_Ch4_P00_01_FS();
#endif
    /* Init Atom for PWM */
    CddEgtm_Init_AtomCls0_Inv();
}


void CddEgtm_Init_Tim0_Ch4_P00_01_FS(void)
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
    CddGpio_ConfigEgtm_Tim0_Ch4_P00_01_FS();

    /* 3. Channel control: TIEM mode, no pairing with adjacent channel */
    timChCtrl.B.TIM_MODE   = 0x2U;   /* TIEM mode                              */
    timChCtrl.B.CICTRL     = 0x0U;   /* do NOT combine with adjacent channel   */
    timChCtrl.B.GPR0_SEL   = 0x3U;
    timChCtrl.B.GPR1_SEL   = 0x3U;
    timChCtrl.B.DSL   = 0x1U;
    timChCtrl.B.ISL   = 0x1U;
                                     /* -> uses TIM_IN(x) = TIM_IN4            */
    timInSrc.B.MODE4       = 0x0u;   /* not forced to constant                 */
    timInSrc.B.VAL4        = 0x0u;   /* not forced to constant                 */

    /* 4. Do not use the LUT */
    timChExtCtrl.B.USE_LUT = 0x0U;

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
 * \details Maps Inv_G.FrequencyMode onto an ATOM period expressed in CMU_CLK0
 *          ticks according to the following table:
 *            | FrequencyMode | Target frequency | PeriodTicks calculation      |
 *            |---------------|------------------|------------------------------|
 *            | 1U            | 10 kHz           | CmuClk0Frequency / KHZ_10    |
 *            | 2U            | 20 kHz           | CmuClk0Frequency / KHZ_20    |
 *            | 3U            | 30 kHz           | CmuClk0Frequency / KHZ_30    |
 *            | default       | 10 kHz           | CmuClk0Frequency / KHZ_10    |
 *
 *          The computed value is stored in Inv_G.PeriodTicks and is later
 *          used by the control-loop ISR to update the master and phase
 *          shadow registers.
 *
 * \return  void
 *
 * \pre     Inv_G.CmuClk0Frequency has been initialised by a prior call to
 *          CddSys_GetEgtmCmu0Frequency().
 *
 * \note    Any FrequencyMode value outside the set {1, 2, 3} silently falls
 *          back to 10 kHz; no error is reported to the caller.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 *          MISRA C:2012 Rule 10.3 – the division result is explicitly cast to
 *          uint32, matching the type of PeriodTicks.
 *          MISRA C:2012 Rule 16.1 / 16.3 – the switch statement is well-formed
 *          and every case (including default) ends with a break.
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
 * \details Performs the complete register-level setup of the ATOM and CDTM
 *          modules required for complementary three-phase PWM with hardware
 *          dead-time and fast shut-off support.
 *
 *          Configuration blocks (executed in order):
 *            -# DTM4 / DTM5 clock select, update mode and shadow-register
 *               update enable; PSU input select and TIM select for the
 *               shut-off path.
 *            -# ATOM CH0 as master in SOMP mode: period and half-period loaded
 *               into the shadow registers, CCU0/CCU1 IRQs enabled and routed
 *               to CPU0, output pin (P02.0) enabled.
 *            -# ATOM CH5 / CH6 / CH7 as phases U / V / W in SOMP mode, each
 *               reset by the master's CCU0 event, with per-phase dead-time
 *               (DTV1 / DTV2 / DTV3) and complementary high/low outputs.
 *            -# Shut-off safe levels for every phase
 *               (OC0_x_SR = 1, OC1_x_SR = 1, SL0_x_SR_SR = 0, SL1_x_SR_SR = 0).
 *            -# Global AGC registers that enable shadow update, force update,
 *               channel enable and output enable for all used channels.
 *
 *          Channel-to-pin map:
 *            - ATOM CH0      : master timing reference (P02.0) – not a gate drive.
 *            - ATOM CH5 / CH5N : phase U high-side (P02.5) / low-side (P00.2).
 *            - ATOM CH6 / CH6N : phase V high-side (P02.6) / low-side (P00.4).
 *            - ATOM CH7 / CH7N : phase W high-side (P02.7) / low-side (P00.6).
 *
 * \return  void
 *
 * \note    All register images are first prepared in local union variables
 *          and written back only after every field has been set.  This
 *          guarantees atomicity of multi-field updates and satisfies the
 *          hardware requirement that certain control bits are written
 *          together.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has internal linkage (static).
 *          MISRA C:2012 Rule 8.9 – all local register images are defined at
 *          function scope.
 *          MISRA C:2012 Rule 11.4 – register access uses the vendor-supplied
 *          bit-field types; no additional pointer casts are introduced.
 *          MISRA C:2012 Rule 2.1 – the final AGC write sequence contains no
 *          unreachable statements.
 *
 * \see     CddEgtm_Init
 * \see     CddEgtm_Start
 */
 /* This function initializes the EGTM_CLS0_ATOM
  * - configures CLS0_ATOM_CH0     (P02.00) as Master Channel and sets-up CCU0 & CCU1 interrupt event to facilitate Control-loop
  * - configures CLS0_ATOM_CH1     (P02.01) as ADC Trigger
  * - configures CLS0_ATOM_CH5     (P02.05) as Phase U_H
  * - configures CLS0_ATOM_CH5_N   (P00_02) as Phase U_L
  * - configures CLS0_ATOM_CH6     (P02.06) as Phase V_H
  * - configures CLS0_ATOM_CH6_N   (P00_04) as Phase V_L
  * - configures CLS0_ATOM_CH7     (P02.07) as Phase W_H
  * - configures CLS0_ATOM_CH7_N   (P00_06) as Phase W_L
*/
 void CddEgtm_Init_AtomCls0_Inv(void)
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
     CddGpio_ConfigEgtmAtom0_Ch0_P02_00_M();

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

     /* Configure ATOM_CH1 as  ADC Trigger */
     /* T1. Configure Channel Control */
     chCtrl.U                  = EGTM_CLS0_ATOM_CH1_CTRL.U;
     chCtrl.B.MODE             = 0x2U;   /* set in SOMP Mode                                      */
     chCtrl.B.UDMODE           = 0x0U;   /* enable Up Count Mode                                  */
     chCtrl.B.CLK_SRC          = 0x0U;   /* select CMU_CLK0                                       */
     chCtrl.B.TRIGOUT          = 0x0U;   /* disable own counter reset                             */
     chCtrl.B.RST_CCU0         = 0x1U;   /* reset the counter on the event CCU0 of Master Channel */
     EGTM_CLS0_ATOM_CH1_CTRL.U = chCtrl.U;

     /* T2. Specify Period and Duty Cycle of PWM Signal (Shadow Register) */
     EGTM_CLS0_ATOM_CH1_SR0.B.SR0 = Inv_G.PeriodTicks;
     EGTM_CLS0_ATOM_CH1_SR1.B.SR1 = Inv_G.PeriodTicks / 2U;

     /* T3. Connect the Channel to Output Pin*/
     CddGpio_ConfigEgtmAtom0_Ch1_P02_01_AT();

     /* T4. Activate the Trigger to TMADC0 */
     EGTM_ADC_OUT0.B.SEL0 = 0x2U; /* USM P.8286, CDTM0_DTM4_1: Dead-time output of ATOM0_1 */
     /* Please be aware that the output ADC_TRIG[11:0] from eGTM are mapped to input TRIG[47:36] of ADC block. USM P.7576 */

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
     EGTM_CLS0_ATOM_CH5_SR1.B.SR1 = Inv_G.PeriodTicks / 2U;

     /* U3. Dead Time Configuration */
     chDTV.U         = EGTM_CLS0_CDTM_DTM5_CH_DTV1.U;
     chDTV.B.RELFALL = Inv_G.DeadTimeTicks;    /* set the dead time for falling edge    */
     chDTV.B.RELRISE = Inv_G.DeadTimeTicks;   /* set the dead time for rising  edge    */
     EGTM_CLS0_CDTM_DTM5_CH_DTV1.U = chDTV.U;
     dtm5ChCtrl2.B.DT0_1 = 0x1U;   /* enable dead time for high-switch */
     dtm5ChCtrl2.B.DT1_1 = 0x1U;   /* enable dead time for low-switch  */

     /* U4. Connect the Channel to Output Pin*/
     CddGpio_ConfigEgtmAtom0_Ch5_P02_05_U_HS();
     CddGpio_ConfigEgtmAtom0_Ch5_P00_02_U_LS();

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
     CddGpio_ConfigEgtmAtom0_Ch6_P02_06_V_HS();
     CddGpio_ConfigEgtmAtom0_Ch6_P00_04_V_LS();

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
     EGTM_CLS0_ATOM_CH7_SR1.B.SR1 = Inv_G.PeriodTicks / 2U;

     /* W3. Dead Time Configuration */
     chDTV.U         = EGTM_CLS0_CDTM_DTM5_CH_DTV3.U;
     chDTV.B.RELFALL = Inv_G.DeadTimeTicks;    /* set the dead time for falling edge    */
     chDTV.B.RELRISE = Inv_G.DeadTimeTicks;   /* set the dead time for rising  edge    */
     EGTM_CLS0_CDTM_DTM5_CH_DTV3.U = chDTV.U;
     dtm5ChCtrl2.B.DT0_3 = 0x1U;   /* enable dead time for high-switch */
     dtm5ChCtrl2.B.DT1_3 = 0x1U;   /* enable dead time for low-switch  */

     /* W4. Connect the Channel to Output Pin*/
     CddGpio_ConfigEgtmAtom0_Ch7_P02_07_W_HS();
     CddGpio_ConfigEgtmAtom0_Ch7_P00_06_W_LS();

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
 }


/**
 * \brief   Starts all enabled eGTM ATOM channels synchronously.
 *
 * \details Sets HOST_TRIG in EGTM_CLS0_ATOM_AGC_GLB_CTRL, which latches the
 *          shadow registers and starts every enabled channel on the same
 *          eGTM clock edge.  From this point the master channel drives the
 *          control-loop ISR and the PWM phases drive the gate outputs.
 *
 * \return  void
 *
 * \pre     CddEgtm_Init() has completed.
 *
 * \note    MISRA C:2012 Rule 8.7 – function has external linkage and is
 *          declared in cdd_egtm_app.h.
 *          MISRA C:2012 Rule 13.3 – the single assignment has no other
 *          side-effects.
 *
 * \see     CddEgtm_Init
 * \see     CddEgtm_Init_AtomCls0_Inv
 */
void CddEgtm_Start(void)
{
    EGTM_CLS0_ATOM_AGC_GLB_CTRL.B.HOST_TRIG = 0x1u;  /* set the trigger request to start all channels synchronously */
}
