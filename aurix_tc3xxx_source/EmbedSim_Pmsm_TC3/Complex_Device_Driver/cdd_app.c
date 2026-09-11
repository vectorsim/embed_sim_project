/**********************************************************************************************************************
 * \file      cdd_app.c
 * \brief     Top-level PMSM application lifecycle and central state for the
 *            AP32541 motor control board (TC38x).
 *
 * \details   Owns the CddApp_G central state object, sequences CDD sub-module
 *            initialisation (STM, EVADC, Encoder/GPT12, GTM, TLE9180D), gates
 *            the transition from "initialised" to "running", and exposes the
 *            control-mode selector and speed-reference setter.
 *
 *            Actual initialisation order executed by CddApp_Init():
 *                CPU-freq check → STM-freq check + CddStm_Init()
 *                → ADC-freq check + CddEvadc_Init()
 *                → CddEncoder_Init()
 *                → GTM-freq check + CddGtm_InitModule()
 *                  + CMU CLK0 check + CddGtm_InitInverter()
 *                → QSPI-freq check + CddApp_InitInverter()
 *
 *            GPIO is NOT initialised in this translation unit despite
 *            "cdd_gpio_app.h" being included; it is assumed to be done by
 *            the iLLD BSP before CddApp_Init() is called.
 *
 * \note      For the ATOM0 channel assignment, dead-time equations, and the
 *            centre-aligned carrier / valley-trigger discussion, see
 *            cdd_gtm_app.c and cdd_gtm_app.h.
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
 *              - Output pointers: PascalCasePtr
 *              - Local variables: lowerPascalCase
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

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_app.h"
#include "cdd_evadc_app.h"
#include "cdd_tle9180_app.h"
#include "cdd_sys_utility.h"
#include "IfxGtm_reg.h"
#include "IfxSrc_reg.h"
#include "IfxScuCcu.h"
#include "IfxConverter_reg.h"
#include "cdd_stm_app.h"
#include "cdd_gpio_app.h"
#include "cdd_gtm_app.h"
#include "cdd_evadc_app.h"
#include "cdd_encoder_app.h"

/* \note "cdd_evadc_app.h" is included twice; "cdd_tle9180_app.h" is also
 *       pulled in transitively via "cdd_app.h".  Neither is harmful but the
 *       duplication can be cleaned up. */

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/**
 * \brief  Central application state — all CDD sub-modules read/write through this.
 *
 * \details Zero-initialised by the C runtime (.bss section); CDDAPP_INIT_PENDING = 0U
 *          so the guard in CddApp_Init() is valid from reset without an explicit
 *          initialiser.
 *
 *          Shared between task context (CddApp_Init, CddApp_Start,
 *          CddApp_SetCtrlMode, CddApp_SetSpeedRefRpm) and the 20 kHz FOC ISR
 *          (which reads CtrlMode, SpeedRefRpm, DutyU/V/W, Vdc, and the offset
 *          fields).
 *
 * \note   MISRA C:2012 Rule 8.9 deviation [D-8.9]: file scope is required
 *         because the structure must persist for the module lifetime and be
 *         accessible from multiple translation units via the extern
 *         declaration in cdd_app.h.
 *
 * \warning CddApp_G is NOT declared volatile.  On TriCore, aligned 32-bit
 *          stores are atomic, so tearing is not a concern, but the compiler
 *          is free to cache field values across the ISR boundary unless the
 *          ISR's own accesses force reloads.  The memory-model contract for
 *          this object is not stated anywhere and should be reviewed.
 *
 * \warning The guard in CddApp_Init() is a one-shot latch on
 *          CDDAppStatus == CDDAPP_INIT_PENDING, not re-entrancy protection.
 */
CddApp_T   CddApp_G;

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialise the power inverter sub-system (TLE9180D gate driver).
 *
 * \details Calls CddTle9180_Startup() which executes the full three-phase sequence:
 *              1. CddTle9180_Init()        — QSPI4 master init (24-bit, ~5 MHz) + GPIO
 *                                            power-on sequence: ENA=HIGH, /INH toggle for
 *                                            forced SLEEP (1 s), then exit SLEEP → IDLE,
 *                                            /SOFF=HIGH.
 *              2. CddTle9180_Configure()   — 13-frame SPI write batch: GEN_CFG1/2,
 *                                            TL_VDH, TL_CBVCC, FM1/3/4/6, CONF_SIG lock,
 *                                            OP_GAIN1/2/3, OP_OCL.
 *              3. CddTle9180_IsNormalMode()— 2-frame STATUS pipeline read; verifies
 *                                            norm_m=1 and CONFVALID=1 in the receive header.
 *
 * \note    CddTle9180_AssertEnable() (ENA = HIGH; gate drive outputs active) is NOT
 *          called here; it is called by CddApp_Start().
 *
 * \warning The safety rationale "called by CddApp_Start() after CddGtm_Start() so
 *          that the bridge output transistors are never energised before the GTM
 *          PWM carrier is live" does NOT match the code.  CddApp_Start() calls
 *          CddTle9180_AssertEnable() and CddTle9180_DeassertSafeOff() BEFORE
 *          CddGtm_Start().  Confirm whether this ordering is intentional.
 *
 * \warning The local "error_code" written by CddTle9180_Startup() is discarded
 *          by the definition of this function; the specific failure reason is
 *          lost.  Consider propagating it into CddApp_G.DTC.
 *
 * \return  0x1U  TLE9180D reached NORMAL operating mode.
 *          0x0U  Startup failed (SPI error, or device did not confirm NORMAL mode).
 */
STATIC uint32_T CddApp_InitInverter(void);

/**********************************************************************************************************************
 * Function Implementations
 *********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation.
 *
 * \details Executes CDD sub-module init in the order shown below, gated by a
 *          local "ok" flag so that no step runs after a failure.  The function
 *          has a single exit point (no internal return statement), satisfying
 *          MISRA C:2012 Rule 15.5.
 *
 *              1. CPU-freq check (300 MHz)
 *              2. STM-freq check (100 MHz)  → CddStm_Init()
 *              3. ADC-freq check (160 MHz)  → CddEvadc_Init()
 *              4. CddEncoder_Init()
 *              5. GTM-freq check (200 MHz)  → CddGtm_InitModule()
 *                                           → CMU CLK0 check (200 MHz)
 *                                           → CddGtm_InitInverter()
 *              6. QSPI-freq check (200 MHz) → CddApp_InitInverter()
 *
 *          GPIO is not initialised here — it is assumed done by the BSP.
 *
 * \pre     CPU/STM/ADC/GTM/QSPI clocks and PLL configured; iLLD BSP initialised.
 *
 * \post    On success: CddAppStatus == CDDAPP_INIT_DONE_INV and DTC == DTC_NONE.
 *          On failure: CddAppStatus is one of the *_ERR_* values and DTC holds
 *          the corresponding code.  Note that CDDAPP_INIT_OK is written by
 *          CddApp_Start(), not by this function.
 *
 * \note    The guard on CDDAppStatus == CDDAPP_INIT_PENDING makes this function
 *          one-shot: a second call after any terminal status — success OR
 *          failure — is a silent no-op.  There is no retry path.
 *
 * \warning The status enum values do not track the execution order of this
 *          function: CDDAPP_INIT_DONE_GTM (55U) is written before
 *          CDDAPP_INIT_DONE_INV (30U).  See CddApp_Status_T in cdd_app.h.
 *
 * \warning The following CddApp_G fields are never explicitly written here and
 *          rely on .bss zero-init: OffsetVro, Isum, Vu, Vv, Vw, DutyAdcTrig,
 *          RotorSpeedRpm, RotorPosition, PeriodTicks, HalfPeriodTicks,
 *          SampleTime, ControlLoopCounter.  PeriodTicks / HalfPeriodTicks /
 *          SampleTime are timing-critical downstream; confirm they are
 *          populated before first use.
 *
 * \warning CDDAPP_INIT_DONE_CTRL, CDDAPP_INIT_ERR_CTRL, CDDAPP_CALIBRATE_OK,
 *          CDDAPP_ERROR_STATE (status enum) and CDDAPP_DTC_CTRL_INIT (DTC enum)
 *          are declared in cdd_app.h but never assigned in this file.
 */
void CddApp_Init(void)
{
    volatile uint32_T ok;

    /* Initialise duty cycles to 50% (zero-voltage vector) before anything else.
     * Control command defaults: OPENLOOP, zero speed — the host must select
     * mode and speed explicitly BEFORE CddApp_Start(); the ISR latches the
     * mode once on the activation edge (no switching during operation).      */
    CddApp_G.DTC         = CDDAPP_DTC_NONE;
    CddApp_G.Iu          = 0.0F;
    CddApp_G.Iv          = 0.0F;
    CddApp_G.Iw          = 0.0F;
    CddApp_G.Vro         = 2.5F;
    CddApp_G.Vdc         = 12.0F;
    CddApp_G.DutyU       = 0.5F;
    CddApp_G.DutyV       = 0.5F;
    CddApp_G.DutyW       = 0.5F;
    CddApp_G.OffsetIu    = 0.0F;
    CddApp_G.OffsetIv    = 0.0F;
    CddApp_G.OffsetIw    = 0.0F;

    CddApp_G.CtrlMode    =  CDDAPP_CTRL_DFC_CLOSEDLOOP;
    CddApp_G.SpeedRefRpm = 1800.0F;
    CddApp_G.SensorReadingBitField = 0x0U;

    /* Guard: proceed only from the reset state.  CDDAPP_INIT_PENDING == 0 is
     * load-bearing for the .bss zero-init assumption.                         */
    ok = ((CddApp_G.CDDAppStatus == CDDAPP_INIT_PENDING) ? 0x1U : 0x0U);

    if(ok == 0x1U)
    {
        /*  CPU Frequency Check */
        ok = (CddSys_AreEqual64(CddSys_GetCpuFreq(), MHZ_300, EPSILON_ZERO) ? 0x1U : 0x0U);
        if(ok != 0x1U)
        {
            CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_CLK;
            CddApp_G.DTC          = CDDAPP_DTC_CPU_FREQ;
        }

        /*  Check STM Frequency and Init STM Module */
        if (ok == 0x1U)
        {
            ok = (CddSys_AreEqual64(CddSys_GetStmFreq(), MHZ_100, EPSILON_ZERO) ? 0x1U : 0x0U);
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_STM;
                CddApp_G.DTC          = CDDAPP_DTC_STM_FREQ;
            }
            else
            {
                CddStm_Init();            /* STM compare-match for 20 kHz FOC ISR deadline */
                CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_STM;
                CddApp_G.DTC          = CDDAPP_DTC_NONE;
            }
        }

        /* Init ADC Module */
        if(ok == 0x1U)
        {
            ok = (CddSys_AreEqual64(CddSys_GetAdcFreq(), MHZ_160, EPSILON_ZERO) ? 0x1U : 0x0U);
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_ADC;
                CddApp_G.DTC          = CDDAPP_DTC_ADC_FREQ;
            }
            else
            {
                CddEvadc_Init();
                CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_ADC;
                CddApp_G.DTC          = CDDAPP_DTC_NONE;
            }
        }

        /* Init GPIT Module(Encoder) */
        if(ok == 0x1U)
        {
            ok = CddEncoder_Init();
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_GPT12;
                CddApp_G.DTC          = CDDAPP_DTC_ERR;
            }
            else
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_GPT12;
                CddApp_G.DTC          = CDDAPP_DTC_NONE;
            }
        }

        /* Init GTM Module  */
        if(ok == 0x1U)
        {
            ok = (CddSys_AreEqual64(CddSys_GetGtmFreq(), MHZ_200, EPSILON_ZERO) ? 0x1U : 0x0U);
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_GTM;
                CddApp_G.DTC          = CDDAPP_DTC_GTM_FREQ;
            }
            else
            {
                CddGtm_InitModule();
                /* Verify CMU CLK0 frequency */
                ok = (CddSys_AreEqual64(CddSys_GetGtmCmuClk00Freq(), MHZ_200, EPSILON_ZERO) ? 0x1U : 0x0U);
                if (ok != 0x1U)
                {
                    CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_GTM;
                    CddApp_G.DTC          = CDDAPP_DTC_GTM_CMU0_FREQ;
                }
                else
                {
                    /* ATOM0 CH0–CH5: complementary PWM pairs (UH/UL, VH/VL, WH/WL).
                     * ISR service request is configured inside the GTM layer.        */
                     CddGtm_InitInverter();
                     CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_GTM;
                     CddApp_G.DTC          = CDDAPP_DTC_NONE;
                }
            }
        }

        /* Initialise QSPI module and Tle9180 Inverter */
        if(ok == 0x1U)
        {
            ok = (CddSys_AreEqual64(CddSys_GetQspiFreq(), MHZ_200, EPSILON_ZERO) ? 0x1U : 0x0U);
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_INV;
                CddApp_G.DTC          = CDDAPP_DTC_QSPI_FREQ;
            }
            else
            {
                ok = CddApp_InitInverter();
                if (ok != 0x1U)
                {
                    CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_INV;
                    CddApp_G.DTC          = CDDAPP_DTC_INV_STARTUP;
                }
                else
                {
                    CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_INV;
                    CddApp_G.DTC          = CDDAPP_DTC_NONE;
                }
            }
        }
    }
}

/**
 * \brief   Thin wrapper around CddTle9180_Startup().
 *
 * \details CddTle9180_Startup() runs Init → Configure → IsNormalMode as one
 *          atomic operation and returns 0x1U only if the TLE9180D has
 *          confirmed norm_m=1 and CONFVALID=1 in the STATUS register pipeline
 *          read.
 *
 * \warning The local "error_code" is written by CddTle9180_Startup() but never
 *          read; the specific failure cause is discarded.  The "volatile"
 *          qualifier on a write-only local also forces a stack slot for no
 *          benefit.
 *
 * \return  0x1U  TLE9180D reached NORMAL operating mode.
 *          0x0U  Startup failed (SPI error, or device did not confirm NORMAL mode).
 */
STATIC uint32_T CddApp_InitInverter(void)
{
    volatile uint32_T error_code;

    return CddTle9180_Startup(&CddApp_G.Inverter, &error_code);
}

/**
 * \brief   Completes the startup sequence: safe-off → ENA → release safe-off
 *          → GTM carrier live.
 *
 * \details Executes, in order:
 *              1. CddTle9180_AssertSafeOff()    — /SOFF asserted, outputs held off
 *              2. CddTle9180_AssertEnable()     — ENA = HIGH
 *              3. CddTle9180_DeassertSafeOff()  — /SOFF released, outputs live
 *              4. CddGtm_Start()                — HOST_TRIG; PWM carrier live
 *              5. CDDAppStatus = CDDAPP_INIT_OK
 *
 * \pre     CddAppStatus == CDDAPP_INIT_DONE_INV (the last status written by
 *          CddApp_Init() on success).
 *
 * \post    CDDAppStatus == CDDAPP_INIT_OK; subsequent calls are no-ops.
 *
 * \warning ENA is asserted and /SOFF is released BEFORE CddGtm_Start(), so
 *          there is a window in which the bridge is un-gated and the PWM
 *          carrier is not yet running.  Confirm whether the ATOM reset state
 *          guarantees a safe (all-low) output throughout that window.  The
 *          safety rationale in cdd_gtm_app.h — "ENA after HOST_TRIG" — does
 *          not describe this code.
 *
 * \warning SRC_GTM_ATOM0_0.B.SRE = 1U does NOT appear anywhere in this file.
 *          If the FOC ISR is armed at all, it is armed inside CddGtm_Start().
 *          The function does not "arm the ISR" as the header claims.
 *
 * \note    The internal "static started" flag is never cleared, so this
 *          function is one-shot: after the first successful call, all
 *          subsequent calls are no-ops.  There is no stop → restart path in
 *          this file, despite cdd_app.h describing a stop → set → restart
 *          contract for CddApp_SetCtrlMode().
 *
 * \warning "started" is not volatile.  If CddApp_Start() can be called from
 *          more than one context this is a race.
 *
 * \return  0x1U once the sequence has ever succeeded; 0x0U otherwise.
 *          Note that a second call returns 0x1U without executing any step —
 *          the return value does NOT mean "all steps were just executed".
 */
uint32_T CddApp_Start(void)
{
    static uint32_T started = 0x0U;

    if((CddApp_G.CDDAppStatus == CDDAPP_INIT_DONE_INV) && (started != 0x1U))
    {
       CddTle9180_AssertSafeOff();
       CddTle9180_AssertEnable();
       CddTle9180_DeassertSafeOff();
       CddGtm_Start();
       CddApp_G.CDDAppStatus = CDDAPP_INIT_OK;
       started = 0x1U;
    }

    return started;
}

/**
 * \brief   Returns the current application initialisation status.
 *
 * \return  Current value of CddApp_G.CDDAppStatus (CddApp_Status_T).
 *          Single aligned 32-bit read; atomic on TriCore.
 */
CddApp_Status_T CddApp_GetInitStatus(void)
{
    return CddApp_G.CDDAppStatus;
}

/**
 * \brief   Selects the control mode — OPENLOOP or CLOSEDLOOP.
 *
 * \details Accepted only while CDDAppStatus != CDDAPP_RUN_STATE.  The 20 kHz
 *          ISR latches the mode once on the activation edge and it is fixed
 *          for the entire run.  Invalid enum values and writes during
 *          CDDAPP_RUN_STATE are silently ignored.
 *
 * \param[in]  Mode  CDDAPP_CTRL_OPENLOOP or CDDAPP_CTRL_DFC_CLOSEDLOOP.
 *
 * \warning The race-benign argument (check-then-store) depends on ISR
 *          ordering that is not visible in this file: the ISR must set
 *          CDDAPP_RUN_STATE AFTER it has latched CtrlMode.  Cross-check the
 *          ISR.
 *
 * \warning CtrlMode is not volatile.  Task-context write, ISR read; aligned
 *          32-bit store is atomic on TriCore, but the compiler's caching
 *          behaviour across the ISR boundary is not constrained by the type.
 */
void CddApp_SetCtrlMode(const CddApp_CtrlMode_T Mode)
{
    if(((Mode == CDDAPP_CTRL_OPENLOOP) || (Mode == CDDAPP_CTRL_DFC_CLOSEDLOOP)) && (CddApp_G.CDDAppStatus != CDDAPP_RUN_STATE))
    {
        CddApp_G.CtrlMode = Mode;
    }
}

/**
 * \brief   Returns the currently selected control mode.
 *
 * \return  Current value of CddApp_G.CtrlMode (CddApp_CtrlMode_T).
 */
CddApp_CtrlMode_T CddApp_GetCtrlMode(void)
{
    return CddApp_G.CtrlMode;
}

/**
 * \brief   Sets the mechanical speed reference.
 *
 * \details Single aligned 32-bit store — atomic on TriCore; live-settable from
 *          any thread context.  Both control paths bound the command
 *          internally (open loop: slew limiter; DFC: ±DFC_OMEGA_CMD_MAX clamp).
 *
 * \param[in]  SpeedRpm  Mechanical speed reference  [RPM]
 *
 * \warning No validation is performed here.  NaN, ±Inf, and arbitrarily large
 *          finite values pass through unchanged.  The claim that both control
 *          paths bound the command is a cross-module invariant not enforced
 *          at this interface.
 */
void CddApp_SetSpeedRefRpm(const real32_T SpeedRpm)
{
    CddApp_G.SpeedRefRpm = SpeedRpm;
}
