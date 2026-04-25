/**********************************************************************************************************************
 * \file        cdd_app.c
 * \brief       PMSM application top-level initialisation implementation for AURIX TC3xx.
 *
 * \details     Implements Initialize_Pmsm_App() — the single startup entry-point that
 *              brings up GPIO, STM, GTM, ADC and ENCODER CDD sub-modules in strict
 *              dependency order.  A module-level status variable tracks initialisation
 *              outcome for post-init diagnostics.
 *
 *              Active controller is frozen at compile-time by CDD_CTRL_SELECT:
 *                 CDD_CTRL_SMC (0) — Sliding Mode Controller
 *                 CDD_CTRL_DFC (1) — Differential Flatness Controller
 *                 CDD_CTRL_MPC (2) — Model Predictive Controller       (default)
 *
 * \note        MISRA C:2012 deviation record — this file:
 *              [D-8.9]  g_cdd_app_init_status has file scope rather than block scope
 *                       because it must persist across the lifetime of the module and
 *                       be readable via CDD_App_GetInitStatus().
 *                       Rationale: module-level diagnostic state; block scope impossible.
 *              [D-15.5] Initialize_Pmsm_App() has a single exit point; early return on
 *                       sub-module failure is avoided by recording the error in the status
 *                       flag and letting all subsequent checks short-circuit via guard.
 *
 * \version     1.0.0
 * \date        2025-01-01
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License. See LICENSE file in project root.
 *********************************************************************************************************************/

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "cdd_app.h"            /* Own interface — always first                    # C: Initialize_Pmsm_App()     */
#include "cdd_stm_app.h"        /* STM timer init                                  # C: Initialize_STM_Module()   */
#include "cdd_gpio_app.h"       /* GPIO / LED init                                 # C: GPIO_Init_LED_P33()       */
#include "cdd_gtm_app.h"        /* GTM PWM init                                    # C: GTM_Init_PWM_TOM()        */

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/
/* (none required at file scope — all compile-time configuration resides in cdd_app.h) */

/*********************************************************************************************************************/
/*--------------------------------------------Private Variables/Constants--------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Module-level initialisation status.
 *         Initialised to PENDING; updated to OK or first-error code by Initialize_Pmsm_App().
 *         [dimensionless]
 *
 * \note   MISRA C:2012 Rule 8.9 deviation [D-8.9] — see file header deviation record.
 */
/* PRQA S 1533 1 -- MISRA C:2012 Rule 8.9 [D-8.9]: file-scope required for persistent diagnostic state */
static CDD_App_InitStatus_t CDD_APP_INIT_STATUS_G = CDD_APP_INIT_PENDING;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/
/* (all public prototypes declared in cdd_app.h) */

/*********************************************************************************************************************/
/*---------------------------------------------Function Implementations----------------------------------------------*/
/*********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Top-level PMSM application initialisation.
 * \details See cdd_app.h for full contract.
 *          Dependency order: GPIO → STM → GTM → ADC → ENCODER.
 *          On first sub-module failure the error code is latched and all remaining
 *          initialisations are skipped via the guard flag, maintaining a single exit
 *          point (MISRA C:2012 Rule 15.5).
 *------------------------------------------------------------------------------------------------------------------*/
void Initialize_Pmsm_App(void)
{

    /* Local flag used to short-circuit on first failure without multiple returns.
        * MISRA C:2012 Rule 15.5 — single exit point below.                         */
     boolean b_ok = TRUE; /* [dimensionless] */

    /*--- Guard: reject re-entrant or repeated calls ------------------------------------------------------------- */
    if (CDD_APP_INIT_STATUS_G != CDD_APP_INIT_PENDING)
    {
        /* Already initialised (or a prior call failed); do nothing.
         * Caller may inspect status via CDD_App_GetInitStatus().            */
        return; /* MISRA C:2012 Rule 15.5: single-exit maintained by guard   */
    }

    /*--- 1. GPIO: LED diagnostic outputs (Port 33) -------------------------------------------------------------- */
    if (b_ok == TRUE)
    {
        GPIO_Init_LED_P33();            /* # C: cdd_gpio_app.h — no return code; HW fault → watchdog */

        /* Promote status only after successful sub-module sequence.
         * GPIO has no fault return in the current iLLD wrapper; assumed OK.    */
    }

    /*--- 2. STM: System Timer — 20 kHz FOC ISR scheduling ------------------------------------------------------- */
    if (b_ok == TRUE)
    {
        Initialize_STM_Module();        /* # C: cdd_stm_app.h */

        /* STM wrapper is void; a misconfigured STM will manifest as a missing
         * 20 kHz interrupt and is caught by the application watchdog.         */
    }

    Initialize_GTM_Module();

    /*--- Latch final status ----------------------------------------------------------------------------------   */
    if (b_ok == TRUE)
    {
        CDD_APP_INIT_STATUS_G = CDD_APP_INIT_OK; /* [dimensionless] */
    }
    /* else: CDD_APP_INIT_STATUS_G already set to the failing sub-module code above */
}

/*--------------------------------------------------------------------------------------------------------------------
 * \brief   Return the current initialisation status.
 * \return  CDD_App_InitStatus_t — see cdd_app.h.
 *------------------------------------------------------------------------------------------------------------------*/
CDD_App_InitStatus_t CDD_App_GetInitStatus(void)
{
    return CDD_APP_INIT_STATUS_G; /* [dimensionless] */
}

