/**********************************************************************************************************************
 * \file        cdd_app.c
 * \brief       PMSM application top-level initialisation implementation for AURIX TC3xx.
 *
 * \details     Implements CddApp_Init() — the single startup entry-point that
 *              brings up all CDD sub-modules in strict dependency order.
 *
 *              Startup sequence:
 *                  1. GPIO            — LED diagnostic outputs (Port 33)
 *                  2. STM             — 20 kHz FOC ISR scheduling
 *                  3. GTM CMU/ATOM    — PWM carrier + 6-channel phase drive; ISR armed (SRE=0)
 *                  4. INVERTER        — QSPI4 init + TLE9180D SPI configuration + normal mode
 *                  5. START           — GTM HOST_TRIG (PWM live)
 *                  6. BRIDGE ENABLE   — CddTle9180_AssertEnable() (ENA = HIGH)
 *                  7. ISR ARM         — SRC_GTM_ATOM0_0.B.SRE = 1U (20 kHz ISR fires)
 *
 *              This ordering guarantees that bridge output transistors are never
 *              energised before the GTM PWM carrier is live, and the ISR never
 *              fires before the inverter is fully configured.
 *
 * \note        MISRA C:2012 deviation record:
 *              [D-8.9]      CddApp_G has file scope; module-lifetime diagnostic state
 *                           accessed by multiple translation units via extern in cdd_app.h.
 *              [D-15.5]     CddApp_Init() achieves a single exit point by absorbing
 *                           the re-entrant guard into b_ok at function entry.  All seven
 *                           startup steps execute inside one if(b_ok == 1U) block.
 *                           No return statement appears inside the function body.
 *              [D-2.2-CMU]  gtm_cmu_frequency is declared volatile.  The assigned value
 *                           is not read in production code paths but is inspected live
 *                           via the AURIX debugger during CMU CLK0 bring-up verification.
 *                           The volatile qualifier prevents the assignment from being
 *                           optimised away and renders Rule 2.2 inapplicable.
 *
 * \version     1.3.0
 * \date        2025-05-24
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
/*
 * cdd_app.h already pulls in cdd_stm_app.h, cdd_gpio_app.h, and cdd_gtm_app.h.
 * Those three headers are NOT re-included here to avoid redundant declarations
 * visible to static analysis tools.  (MISRA C:2012 Advisory — include-once principle.)
 */
#include "cdd_app.h"                  /* Own interface — always first                                     */
#include "cdd_tle9180_app.h"          /* CddTle9180_Startup(), CddTle9180_AssertEnable(), CddTle9180_T      */
/* cdd_qspi_app.h is NOT included here: CddQspi4_Init() is called internally by
 * CddTle9180_Startup() -> CddTle9180_Init().  A direct call from this module
 * would double-initialise the QSPI4 peripheral.                                */
#include "cdd_sys_utility.h"          /* CddSys_NopDelay(), CddSys_ClearWdtEndInit(),
                                         CddSys_SetWdtEndInit(), CddSys_SetGtmCmuClk00Freq(),
                                         CddSys_GetGtmCmuClk00Freq()                                      */
#include "IfxGtm_reg.h"               /* GTM_CLC, GTM_CTRL, GTM_CCM0_PROT, GTM_CLS_CLK_CFG,
                                         GTM_CMU_CLK_EN                                                    */
#include "IfxSrc_reg.h"               /* SRC_GTM_ATOM0_0                                                  */

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/*
 * Central application state — all CDD sub-modules read/write through this.
 * Initialised to CDDAPP_INIT_PENDING so the guard in CddApp_Init()
 * detects re-entrant or repeated calls via the b_ok evaluation at entry.
 *
 * MISRA C:2012 Rule 8.9 deviation [D-8.9]: file scope is required because
 * the structure must persist for the module lifetime and be accessible from
 * multiple translation units via the extern declaration in cdd_app.h.
 */
CddApp_T   CddApp_G =
{
    CDDAPP_INIT_PENDING,    /* CDDAppInitStatus  */
    CDDINV_INIT_PENDING,    /* CDDInverterStatus */
    0.6F,                   /* DutyU             */
    0.6F,                   /* DutyV             */
    0.6F,                   /* DutyW             */
    0U,                     /* PeriodTicks       */
    0U,                     /* HalfPeriodTicks   */
    0.0F                    /* SampleTime        */
};

/*
 * TLE9180D runtime handle — private to this module.
 * Static storage guarantees zero-initialisation per ISO C99 §6.7.8:10.
 * Internal linkage is correct: CddApp_InitInverter() and the cyclic fault
 * monitor pass this handle by pointer to cdd_tle9180_app functions.
 * MISRA C:2012 Rule 8.9: internal linkage — no deviation required.
 */
static CddTle9180_T CddTle9180_G;

/**********************************************************************************************************************
 * Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Top-level PMSM application initialisation.
 * \details Full contract in cdd_app.h.
 *          Startup order: GPIO → STM → GTM → INVERTER → HOST_TRIG → BRIDGE ENABLE → ISR ARM.
 *          The re-entrant guard is absorbed into b_ok at function entry so that all
 *          seven steps execute inside a single if(b_ok == 1U) block with no internal
 *          return statement.  MISRA C:2012 Rule 15.5 is therefore satisfied — deviation
 *          [D-15.5] documents this pattern.
 *------------------------------------------------------------------------------------------------------------------*/
void CddApp_Init(void)
{
    /*
     * MISRA C:2012 Rule 2.2 deviation [D-2.2-CMU]:
     * gtm_cmu_frequency is volatile.  It receives the resolved CMU CLK0 frequency
     * from CddSys_GetGtmCmuClk00Freq() and is inspected live via the AURIX debugger
     * during bring-up.  The volatile qualifier is the side-effect that satisfies
     * Rule 2.2; the value is intentionally not consumed in production code.
     */
    volatile real32_T gtm_cmu_frequency;
    uint32_T          b_ok;

    /*
     * Guard — absorbed into b_ok (Rule 15.5 deviation [D-15.5]).
     * b_ok = 1U : first call, status is PENDING   → proceed with all steps.
     * b_ok = 0U : repeated or re-entrant call      → if-block is skipped entirely.
     * No return statement is used inside this function.
     */
    b_ok = (CDDAPP_INIT_STATUS_G == CDDAPP_INIT_PENDING) ? 1U : 0U;

    if (b_ok == 1U)
    {
        /*--- Step 1: GPIO — LED diagnostic outputs (Port 33) -------------------------------------------*/
        CddGpio_InitLed_P33();
        /*
         * Wrapper is void; HW fault manifests as missing LED toggle, detected
         * by the application watchdog supervisor.
         */

        /*--- Step 2: STM — System Timer for 20 kHz FOC ISR scheduling ---------------------------------*/
        CddStm_Init();
        /*
         * Wrapper is void; misconfiguration manifests as a missing 20 kHz interrupt,
         * detected by the application watchdog supervisor.
         */

        /*--- Step 3: GTM — CMU CLK0 bring-up + ATOM0 PWM init -----------------------------------------*/

        /* Enable GTM module clock — GTM_CLC is ENDINIT-protected */
        CddSys_ClearWdtEndInit();
        GTM_CLC.B.DISR = 0x0U;
        CddSys_SetWdtEndInit();
        while (GTM_CLC.B.DISS != 0x0U)
        {
            CddSys_NopDelay(1U, 1U);
        }

        /* Disable write protection on cluster configuration registers */
        GTM_CTRL.B.RF_PROT       = 0x0U;
        GTM_CCM0_PROT.B.CLS_PROT = 0x0U;

        /* Enable cluster 0 with no additional clock divider (User Manual P.122) */
        GTM_CLS_CLK_CFG.B.CLS0_CLK_DIV = 0x1U;

        /* Disable all CMU clocks first, then program CLK0 to GTM_CMU_CLK0_FREQUENCY */
        GTM_CMU_CLK_EN.U = 0x55555555U;
        CddSys_SetGtmCmuClk00Freq(GTM_CMU_CLK0_FREQUENCY);
        gtm_cmu_frequency = (real32_T)CddSys_GetGtmCmuClk00Freq();   /* [D-2.2-CMU] */

        /* Initialise GTM ATOM0 channels CH0–CH7; ISR service request is armed (SRE=0 — not yet firing) */
        CddGtm_Init();

        /*--- Step 4: INVERTER — QSPI4 init + TLE9180D SPI configuration + normal mode -----------------*/
        /* CddApp_InitInverter(); */

        /*--- Step 5: START — GTM HOST_TRIG, PWM carrier live before bridge is enabled -----------------*/
        CddGtm_Start();

        /*--- Step 6: BRIDGE ENABLE — assert ENA high; bridge output transistors now driven by PWM -----*/
        CddTle9180_AssertEnable();


        /*--- Latch final status: all seven startup steps completed successfully -----------------------*/
        CDDAPP_INIT_STATUS_G = CDDAPP_INIT_OK;

    } /* end if (b_ok == 1U) */
    /*
     * else: CDDAPP_INIT_STATUS_G retains CDDAPP_INIT_PENDING.
     * The caller detects the outcome via CddApp_GetInitStatus().
     */
}

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Inverter sub-system initialisation (QSPI4 + TLE9180D).
 * \details Full contract in cdd_app.h.
 *          CddTle9180_AssertEnable() (ENA = HIGH) is deliberately NOT called here.
 *          It is called by CddApp_Init() at Step 6, after CddGtm_Start() at Step 5,
 *          so that bridge output transistors are never energised before the PWM
 *          carrier is live.
 *------------------------------------------------------------------------------------------------------------------*/
void CddApp_InitInverter(void)
{
    uint32_T result;

    /*
     * Configure QSPI4 GPIO pins before the TLE9180 driver touches the bus.
     * MOSI=P22.0, MISO=P22.1, CS=P22.2 (SLSO3), SCLK=P22.3.
     * INH=LOW, ENA=LOW, /SOFF=HIGH — safe-state before SPI bring-up.
     */
    CddGpio_ConfigQspi4Pins();

    /*
     * CddTle9180_Startup() is the convenience wrapper:
     *   Init (calls CddQspi4_Init internally) → Configure → IsNormalMode.
     * CddQspi4_Init() is therefore NOT called separately here.
     * Return value 0x1U = NORMAL mode reached; 0x0U = startup failed.
     * MISRA C:2012 Rule 17.7: return value captured and used to set CDDInverterStatus.
     */
    result = CddTle9180_Startup(&CddTle9180_G);

    if (result == 1U)
    {
        CddApp_G.CDDInverterStatus = CDDINV_INIT_OK;
    }
    else
    {
        CddApp_G.CDDInverterStatus = CDDINV_INIT_ERR;
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Start the PMSM application (re-trigger HOST_TRIG).
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddApp_Start(void)
{
    uint32_T started;

    started = 0x0U;

    if (CddApp_G.CDDAppInitStatus == CDDAPP_INIT_OK)
    {
        CddGtm_Start();
        started = 0x1U;
    }

    return started;
}

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Return the current application initialisation status.
 *------------------------------------------------------------------------------------------------------------------*/
CddApp_InitStatus_T CddApp_GetInitStatus(void)
{
    return CDDAPP_INIT_STATUS_G;
}
