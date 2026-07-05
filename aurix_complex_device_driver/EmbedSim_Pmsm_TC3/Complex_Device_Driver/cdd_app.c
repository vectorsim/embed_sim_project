/**********************************************************************************************************************
 * \file        cdd_app.c
 * \brief       PMSM application top-level initialisation implementation for AURIX TC3xx.
 *
 * \details     Implements CddApp_Init() and CddApp_Start() — the two startup entry-points
 *              that bring up all CDD sub-modules in strict dependency order.
 *
 *              Startup sequence (actual code order):
 *                  1. GPIO            — CddGpio_InitLed_P33(): LED diagnostic outputs (Port 33)
 *                  2. STM             — CddStm_Init(): System Timer for 20 kHz FOC ISR
 *                  3. INVERTER        — CddTle9180_Startup(): QSPI4 init + TLE9180D GPIO
 *                                       power-on sequence (INH/ENA//SOFF) + 13-frame SPI
 *                                       configuration batch + NORMAL mode verification
 *                  4. GTM CLK ENABLE  — GTM_CLC.DISR=0: release GTM module clock gate
 *                  5. GTM CMU CLK0    — CddSys_SetGtmCmuClk00Freq(): CMU CLK0 = 200 MHz
 *                  6. GTM ATOM0       — CddGtm_Init(): ATOM0 CH0–CH5 complementary PWM;
 *                                       ISR service request armed but not yet firing (SRE=0)
 *                  7. CTRL LAYER      — CddGtm_CtrlInit(): Transform_Init() + DFC_Init();
 *                                       CtrlMode defaults to CDDAPP_CTRL_OPENLOOP
 *              --- CddApp_Start() completes the sequence ---
 *                  8. PWM LIVE        — CddGtm_Start(): HOST_TRIG; PWM carrier goes live
 *                  9. BRIDGE ENABLE   — CddTle9180_AssertEnable(): ENA = HIGH; gate
 *                                       drive outputs active on AP32541
 *                 10. ISR ARM         — SRC_GTM_ATOM0_0.B.SRE = 1U: 20 kHz FOC ISR fires
 *
 *              The split between CddApp_Init() and CddApp_Start() guarantees that
 *              bridge output transistors are never energised before the GTM PWM carrier
 *              is live, and the ISR never fires before the inverter has reached NORMAL
 *              operating mode.
 *
 * \note        MISRA C:2012 deviation record:
 *              [D-8.9]      CddApp_G has file scope; module-lifetime state accessed by
 *                           multiple translation units via extern in cdd_app.h.
 *              [D-15.5]     CddApp_Init() achieves a single exit point by absorbing the
 *                           re-entrant guard into ok at function entry.  All startup steps
 *                           execute inside one if(ok == 0x1U) block.  No return statement
 *                           appears inside the function body.
 *              [D-2.2-CMU]  gtm_cmu_frequency is declared volatile.  The assigned value
 *                           is not read in production code paths but is inspected live via
 *                           the AURIX debugger during CMU CLK0 bring-up verification.
 *                           The volatile qualifier prevents the assignment from being
 *                           optimised away and renders Rule 2.2 inapplicable.
 *
 * \version     1.4.0
 * \date        2026-07-04
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
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
#include "cdd_stm_app.h"        /* CddStm_Init()                                               */
#include "cdd_gpio_app.h"       /* CddGpio_InitLed_P33()                                       */
#include "cdd_gtm_app.h"        /* CddGtm_Init(), CddGtm_Start()                               */
#include "cdd_evadc_app.h"      /* CddEvadc_Meas_T, FocUvw_T (via embed_sim_foc_types.h)       */

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/*
 * Central application state — all CDD sub-modules read/write through this.
 * Zero-initialised by the C runtime (.bss section); CDDAPP_INIT_PENDING = 0U so the
 * re-entrant guard in CddApp_Init() is correct from reset without an explicit initialiser.
 *
 * MISRA C:2012 Rule 8.9 deviation [D-8.9]: file scope is required because the structure
 * must persist for the module lifetime and be accessible from multiple translation units
 * via the extern declaration in cdd_app.h.
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
 * \note    CddTle9180_AssertEnable() (ENA = HIGH; gate drive outputs active) is NOT called
 *          here.  It is called by CddApp_Start() after CddGtm_Start() so that the bridge
 *          output transistors are never energised before the GTM PWM carrier is live.
 *
 * \return  0x1U  TLE9180D reached NORMAL operating mode.
 *          0x0U  Startup failed (SPI error, or device did not confirm NORMAL mode).
 */
STATIC uint32_T CddApp_InitInverter(void);

/**********************************************************************************************************************
 * Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_Init
 *
 * Top-level PMSM application initialisation.  Full contract in cdd_app.h.
 *
 * Startup order: GPIO → STM → INVERTER → GTM CLK → GTM CMU → GTM ATOM.
 * The re-entrant guard is absorbed into ok at function entry so that all steps
 * execute inside a single if(ok == 0x1U) block with no internal return statement.
 * MISRA C:2012 Rule 15.5 is therefore satisfied — deviation [D-15.5].
 *------------------------------------------------------------------------------------------------------------------*/
void CddApp_Init(void)
{
    volatile uint32_T ok;

    /* Initialise duty cycles to 50% (zero-voltage vector) before anything else.
     * Control command defaults: OPENLOOP, zero speed — the host must select
     * mode and speed explicitly BEFORE CddApp_Start(); the ISR latches the
     * mode once on the activation edge (no switching during operation).      */
    CddApp_G.DTC         = CDDAPP_DTC_NONE;
    CddApp_G.DutyU       = 0.5F;
    CddApp_G.DutyV       = 0.5F;
    CddApp_G.DutyW       = 0.5F;
    CddApp_G.Vuo         = 0.0F;
    CddApp_G.Vvo         = 0.0F;
    CddApp_G.Vwo         = 0.0F;
    CddApp_G.CtrlMode    = CDDAPP_CTRL_OPENLOOP;
    CddApp_G.SpeedRefRpm = 600.0F;

    /* Re-entrant / repeated call guard: CDDAPP_INIT_PENDING = 0 (set by .bss at reset) */
    ok = ((CddApp_G.CDDAppStatus == CDDAPP_INIT_PENDING) ? 0x1U : 0x0U);

    if (ok == 0x1U)
    {
        /* ── Step 1: CPU frequency check ──────────────────────────────────────────────── */
        ok = (CddSys_AreEqual64(CddSys_GetCpuFreq(), MHZ_300, EPSILON_ZERO) ? 0x1U : 0x0U);
        if (ok != 0x1U)
        {
            CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_CLK;
            CddApp_G.DTC          = CDDAPP_DTC_CPU_FREQ;
        }

        /* ── Step 2: STM frequency check ──────────────────────────────────────────────── */
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
                /* ── Step 2 init: GPIO (LED diagnostics) + STM ────────────────────────── */
                CddGpio_InitLed_P33();    /* Port 33 LED outputs for diagnostic indication */
                CddStm_Init();            /* STM compare-match for 20 kHz FOC ISR deadline */
                CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_STM;
            }
        }

        /* ── Step 3: QSPI frequency check + inverter init ─────────────────────────────── */
        if (ok == 0x1U)
        {
            ok = (CddSys_AreEqual64(CddSys_GetQspiFreq(), MHZ_200, EPSILON_ZERO) ? 0x1U : 0x0U);
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_INV;
                CddApp_G.DTC          = CDDAPP_DTC_QSPI_FREQ;
            }
            else
            {
                /* CddTle9180_Startup(): Init → Configure → IsNormalMode */
                ok = CddApp_InitInverter();

                if (ok != 0x1U)
                {
                    CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_INV;
                    CddApp_G.DTC          = CDDAPP_DTC_INV_STARTUP;
                }
                else
                {
                    CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_INV;
                }
            }
        }

        /* ── Step 4: GTM frequency check ──────────────────────────────────────────────── */
        if (ok == 0x1U)
        {
            ok = (CddSys_AreEqual64(CddSys_GetGtmFreq(), MHZ_200, EPSILON_ZERO) ? 0x1U : 0x0U);
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_GTM;
                CddApp_G.DTC          = CDDAPP_DTC_GTM_FREQ;
            }
            else
            {
                /* ── Step 4 init: Release GTM module clock gate ───────────────────────── */
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

                /* ── Step 5: Disable all CMU clocks, program CLK0 = 200 MHz ─────────── */
                GTM_CMU_CLK_EN.U = 0x55555555U;                     /* disable all clocks */
                CddSys_SetGtmCmuClk00Freq(GTM_CMU_CLK0_FREQUENCY);  /* set CLK0 = 200 MHz */

                /* Verify CMU CLK0 frequency after programming */
                ok = (CddSys_AreEqual64(CddSys_GetGtmCmuClk00Freq(), MHZ_200, EPSILON_ZERO) ? 0x1U : 0x0U);
                if (ok != 0x1U)
                {
                    CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_GTM;
                    CddApp_G.DTC          = CDDAPP_DTC_GTM_CMU0_FREQ;
                }
            }
        }

        /* ── Step 6: GTM ATOM0 PWM init ───────────────────────────────────────────────── */
        if (ok == 0x1U)
        {
            /* ATOM0 CH0–CH5: complementary PWM pairs (UH/UL, VH/VL, WH/WL).
             * ISR service request is configured here but SRE=0 — ISR does not fire
             * until CddApp_Start() sets SRE=1 after the PWM carrier is live.          */
            CddEvadc_Init();
            CddGtm_Init();
            CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_GTM;
        }

        /* ── Step 7: Control-loop layer (Transform_Init + DFC_Init) ───────────────────── */
        if (ok == 0x1U)
        {
            /* Clarke/Park transform setup + DFC controller state zeroing with
             * default gains.  Until this succeeds the ISR keeps to the open-loop
             * V/f path regardless of the mode requested via CddApp_SetCtrlMode(). */
            ok = CddGtm_CtrlInit();
            if (ok != 0x1U)
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_ERR_CTRL;
                CddApp_G.DTC          = CDDAPP_DTC_CTRL_INIT;
            }
            else
            {
                CddApp_G.CDDAppStatus = CDDAPP_INIT_DONE_CTRL;
                CddApp_G.CDDAppStatus = CDDAPP_INIT_OK;
            }
        }
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_InitInverter
 *
 * Thin wrapper that calls CddTle9180_Startup() and propagates the return value.
 * CddTle9180_Startup() is the correct entry-point: it runs Init → Configure →
 * IsNormalMode as one atomic operation and returns 0x1U only if the TLE9180D
 * has confirmed norm_m=1 and CONFVALID=1 in the STATUS register pipeline read.
 *------------------------------------------------------------------------------------------------------------------*/
STATIC uint32_T CddApp_InitInverter(void)
{
    volatile uint32_T error_code;

    return CddTle9180_Startup(&CddApp_G.Inverter, &error_code);
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_Start
 *
 * Completes the startup sequence after CddApp_Init() has returned CDDAPP_INIT_OK:
 *   1. CddGtm_Start()             — HOST_TRIG; GTM ATOM0 PWM carrier goes live.
 *   2. CddTle9180_AssertEnable()  — ENA = HIGH; TLE9180D gate drive outputs become
 *                                   active.  Called after HOST_TRIG so that the bridge
 *                                   transistors are never energised into an undefined
 *                                   PWM state.
 *   3. SRC_GTM_ATOM0_0.B.SRE=1U  — Arms the GTM ATOM0 CH0 interrupt; 20 kHz FOC ISR
 *                                   begins firing on the next compare-match event.
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddApp_Start(void)
{
    static uint32_T started = 0x0U;



    if((CddApp_G.CDDAppStatus == CDDAPP_INIT_OK) && (started != 0x1U))
    {
       uint32_T clrErr = 0x0U;

       /* Step 7 — CALIBRATION WINDOW: gates hard-off while everything else
        * runs.  PowerOnSequence leaves /SOFF deasserted, so it must be
        * re-asserted HERE, before ENA and the PWM carrier, to guarantee the
        * bridge is never energised during offset calibration.                 */
       CddTle9180_AssertSafeOff();            /* gates hard-off (latched)                */
       CddTle9180_AssertEnable();             /* pre-drivers active, outputs held off    */
       CddGtm_Start();                        /* PWM carrier live, 20 kHz ISR firing,
                                               * EVADC conversions flowing (ISR reads
                                               * sensors in every state)                 */

       /* Step 8 — one-shot CSA offset calibration: true phase currents are
        * zero (/SOFF asserted), the ISR refreshes Vu/Vv/Vw/Vr each tick.
        * Blocking ~3.2 ms at 64 samples.  On timeout (ISR not ticking) the
        * offsets stay 0.0f and conversion degrades gracefully — verify
        * Vuo/Vvo/Vwo are non-zero in the watch window after startup.          */
       CddEvadc_CalibratePhaseOffsets(&CddApp_G, 64U);

       /* Step 9 — clear any faults logged while inputs toggled against
        * /SOFF (Err_indiag class), then release the gates and go live.        */
       //(void)CddTle9180_ClearFaults(&clrErr);
       CddTle9180_DeassertSafeOff();          /* bridge live                             */

       CddApp_G.CDDAppStatus = CDDAPP_RUN_STATE;
       started = 0x1U;

    }

    return started;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_GetInitStatus
 *------------------------------------------------------------------------------------------------------------------*/
CddApp_Status_T CddApp_GetInitStatus(void)
{
    return CddApp_G.CDDAppStatus;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_SetCtrlMode
 *
 * Accepted only outside CDDAPP_RUN_STATE — the ISR latches the mode exactly
 * once on the activation edge and it is fixed for the entire run.  The
 * check-then-store sequence is race-benign: if the state flips to RUN between
 * check and store, the ISR has already latched the previous mode on that
 * activation edge, and the late-written value is not consumed before the next
 * activation — exactly the stop → set → restart contract.
 *------------------------------------------------------------------------------------------------------------------*/
void CddApp_SetCtrlMode(const CddApp_CtrlMode_T Mode)
{
    if (((Mode == CDDAPP_CTRL_OPENLOOP) || (Mode == CDDAPP_CTRL_CLOSEDLOOP)) &&
        (CddApp_G.CDDAppStatus != CDDAPP_RUN_STATE))
    {
        CddApp_G.CtrlMode = Mode;
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_GetCtrlMode
 *------------------------------------------------------------------------------------------------------------------*/
CddApp_CtrlMode_T CddApp_GetCtrlMode(void)
{
    return CddApp_G.CtrlMode;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddApp_SetSpeedRefRpm
 *
 * Single aligned 32-bit store — atomic on TriCore; live-settable from any
 * thread context.  Both control paths bound the command internally (open
 * loop: slew limiter; DFC: ±DFC_OMEGA_CMD_MAX clamp).
 *------------------------------------------------------------------------------------------------------------------*/
void CddApp_SetSpeedRefRpm(const real32_T SpeedRpm)
{
    CddApp_G.SpeedRefRpm = SpeedRpm;
}
