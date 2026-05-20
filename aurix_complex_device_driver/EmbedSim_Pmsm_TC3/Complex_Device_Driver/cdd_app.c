/**********************************************************************************************************************
 * \file        cdd_app.c
 * \brief       PMSM application top-level initialisation implementation for AURIX TC3xx.
 *
 * \details     Implements Initialize_Pmsm_App() — the single startup entry-point that
 *              brings up all CDD sub-modules in strict dependency order.
 *
 *              Startup sequence:
 *                  1. GPIO            — LED diagnostic outputs (Port 33)
 *                  2. STM             — 20 kHz FOC ISR scheduling
 *                  3. GTM CMU/ATOM    — PWM carrier + 6-channel phase drive; ISR armed (SRE=0)
 *                  4. INVERTER        — QSPI4 init + TLE9180D SPI configuration + normal mode
 *                  5. START           — GTM HOST_TRIG (PWM live)
 *                  6. BRIDGE ENABLE   — CDD_Tle9180_Enable() (ENA = HIGH)
 *                  7. ISR ARM         — SRC_GTM_ATOM0_0.B.SRE = 1 (20 kHz ISR fires)
 *
 *              This ordering guarantees that bridge output transistors are never
 *              energised before the GTM PWM carrier is live, and the ISR never
 *              fires before the inverter is fully configured.
 *
 *              Implements Initialize_Inverter() — QSPI4 + TLE9180D startup.
 *
 * \note        MISRA C:2012 deviation record:
 *              [D-8.9]  CDD_App_G has file scope; module-lifetime diagnostic state.
 *              [D-15.5] Initialize_Pmsm_App() uses a single exit point; failures are
 *                       latched and guarded via b_ok without multiple returns.
 *
 * \version     1.1.0
 * \date        2025-05-18
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_app.h"                  /* Own interface — always first                                    */
#include "cdd_stm_app.h"              /* Initialize_STM_Module()                                         */
#include "cdd_gpio_app.h"             /* GPIO_Init_LED_P33(), GPIO_Configure_QSPI4_Pins()                */
#include "cdd_gtm_app.h"              /* Initialize_GTM_Module(), Start_GTM_Module()                     */
#include "cdd_sys_utility.h"          /* Nop_Delay(), Clear/Set_CPU_WDT_EndInit()                        */
#include "IfxGtm_reg.h"               /* GTM_CLC, GTM_CTRL, GTM_CMU_CLK_EN, SRC_GTM_ATOM0_0             */
#include "IfxSrc_reg.h"               /* SRC_GTM_ATOM0_0                                                 */

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/*
 * Central application state — all CDD sub-modules read/write through this.
 * Initialised to CDD_APP_INIT_PENDING so the guard in Initialize_Pmsm_App()
 * rejects re-entrant calls.
 *
 * MISRA C:2012 Rule 8.9 deviation [D-8.9]: file scope required — the structure
 * must persist for the module lifetime and be accessible by all sub-modules
 * via the extern declaration in cdd_app.h.
 */
CDD_APP_t   CDD_App_G =
{
    CDD_APP_INIT_PENDING,   /* CDDAppInitStatus  */
    CDD_INV_INIT_PENDING,   /* CDDInverterStatus */
    0.5F,                   /* DutyU             */
    0.5F,                   /* DutyV             */
    0.5F,                   /* DutyW             */
    0U,                     /* PeriodTicks       */
    0U,                     /* HalfPeriodTicks   */
    0.0F                    /* SampleTime        */
};

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Top-level PMSM application initialisation.
 * \details Full contract in cdd_app.h.
 *          Startup order: GPIO → STM → GTM → INVERTER → HOST_TRIG → BRIDGE → ISR ARM.
 *          On first sub-module failure the error code is latched and all remaining
 *          steps are skipped via b_ok (MISRA C:2012 Rule 15.5 — single exit point).
 *------------------------------------------------------------------------------------------------------------------*/
void Initialize_Pmsm_App(void)
{
    uint32_T         b_ok;
    volatile real32_T  gtm_cmu_frequency;   /* retained — useful for debugger inspection */

    b_ok = 1U;

    /*--- Guard: reject re-entrant or repeated calls ------------------------------------------------- */
    if (CDD_APP_INIT_STATUS_G != CDD_APP_INIT_PENDING)
    {
        return;   /* MISRA C:2012 Rule 15.5: single-exit maintained by guard */
    }

    /*--- Step 1: GPIO — LED diagnostic outputs (Port 33) ------------------------------------------- */
    if (b_ok == 1U)
    {
        GPIO_Init_LED_P33();
        /* GPIO wrapper is void; HW fault manifests as missing LED toggle → app watchdog. */
    }

    /*--- Step 2: STM — System Timer for 20 kHz FOC ISR scheduling ---------------------------------- */
    if (b_ok == 1U)
    {
        Initialize_STM_Module();
        /* STM wrapper is void; misconfiguration manifests as missing 20 kHz interrupt. */
    }

    /*--- Step 3: GTM — CMU CLK0 bring-up + ATOM0 PWM init ----------------------------------------- */
    if (b_ok == 1U)
    {
        /* Enable GTM module clock */
        Clear_CPU_WDT_EndInit();
        GTM_CLC.B.DISR = 0x0U;
        Set_CPU_WDT_EndInit();
        while (GTM_CLC.B.DISS != 0x0U)
        {
            Nop_Delay(1U, 1U);
        }

        /* Disable write protection of cluster configuration registers */
        GTM_CTRL.B.RF_PROT       = 0x0U;
        GTM_CCM0_PROT.B.CLS_PROT = 0x0U;

        /* Enable cluster 0, no clock divider (ds2 P.122) */
        GTM_CLS_CLK_CFG.B.CLS0_CLK_DIV = 0x1U;

        /* Disable all CMU clocks first, then configure CLK0 */
        GTM_CMU_CLK_EN.U = 0x55555555U;
        Set_GTM_CMU_CLK_00_Frequency(GTM_CMU_CLK0_FREQUENCY);
        gtm_cmu_frequency = Get_GTM_CMU_CLK_00_Frequency();

        /* Initialise GTM ATOM0 channels CH0–CH7; ISR is armed (SRE=0 — not yet firing) */
        Initialize_GTM_Module();
    }

    /*--- Step 4: INVERTER — QSPI4 + TLE9180D startup ------------------------------------------------ */
    if (b_ok == 1U)
    {
        Initialize_Inverter();

        if (CDD_App_G.CDDInverterStatus != CDD_INV_INIT_OK)
        {
            CDD_APP_INIT_STATUS_G = CDD_APP_INIT_ERR_INV;
            b_ok = 0U;
        }
    }


    /*--- Step 5: START — GTM HOST_TRIG (PWM carrier live before bridge enable) -------------------- */
    if (b_ok == 1U)
    {
        Start_GTM_Module();
    }

    /*--- Step 6: BRIDGE ENABLE — ENA = HIGH (output transistors energised) ------------------------- */
    if (b_ok == 1U)
    {
        /*CDD_Tle9180_Enable();*/
        /*
         * Bridge outputs are enabled with 50% duty → zero voltage vector applied.
         * Open-loop will ramp from this state when GTM_OpenLoop_Set_RPM() is called.
         */
    }

    /*--- Step 7: ISR ARM — enable GTM ATOM0_CH0 CCU1 service request to CPU ------------------------ */
    if (b_ok == 1U)
    {
        /* Enable the GTM ATOM0_CH0 interrupt service request to CPU0.        *
         * ISR was initialised with SRE=0 in Initialize_GTM_Module() to       *
         * prevent premature firing before bridge outputs are enabled.         */
        SRC_GTM_ATOM0_0.B.SRE = 0x1U;
    }

    /*--- Latch final status ------------------------------------------------------------------------- */
    if (b_ok == 1U)
    {
        CDD_APP_INIT_STATUS_G = CDD_APP_INIT_OK;
    }
    /* else: CDD_APP_INIT_STATUS_G already holds the failing sub-module error code */
}

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Inverter sub-system initialisation (QSPI4 + TLE9180D).
 * \details Full contract in cdd_app.h.
 *          Note: CDD_Tle9180_Enable() is deliberately NOT called here.
 *          It is called by Initialize_Pmsm_App() after Start_GTM_Module()
 *          so that bridge outputs are never enabled before the PWM carrier is live.
 *------------------------------------------------------------------------------------------------------------------*/
void Initialize_Inverter(void)
{

}

uint32_T Start_Pmsm_App(void)
{
    uint32_T started;

    started = 0x0u;

    if(CDD_App_G.CDDAppInitStatus == CDD_APP_INIT_OK)
    {
        Start_GTM_Module();
        started = 0x1u;
    }

    return started;

}


/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Return the current application initialisation status.
 *------------------------------------------------------------------------------------------------------------------*/
CDD_App_InitStatus_t CDD_App_GetInitStatus(void)
{
    return CDD_APP_INIT_STATUS_G;
}
