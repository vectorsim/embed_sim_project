/**********************************************************************************************************************
 * \file        cdd_app.h
 * \brief       PMSM application top-level initialisation interface for AURIX TC3xx.
 *
 * \details     This module is the single entry-point called by the OS/startup layer to bring up all
 *              Complex Device Driver (CDD) sub-modules required for PMSM Field-Oriented Control.
 *              Sub-modules initialised in dependency order:
 *                 1. GPIO  – LED diagnostic outputs (Port 33)
 *                 2. STM   – System Timer for 20 kHz FOC ISR scheduling
 *                 3. GTM   – PWM generation (TOM channels, three-phase inverter)
 *                 4. EVADC – Phase-current and DC-link voltage acquisition
 *                 5. ENCODER – Quadrature encoder position/speed interface
 *
 *              Controller build is selected at compile-time via the CDD_CTRL_SELECT macro:
 *                 CDD_CTRL_SMC  (0) – Sliding Mode Controller          # C: smc_controller.h
 *                 CDD_CTRL_DFC  (1) – Differential Flatness Controller  # C: dfc_controller.h
 *                 CDD_CTRL_MPC  (2) – Model Predictive Controller       # C: mpc_controller.h
 *
 * \note        MISRA C:2012 deviation record — this file:
 *              [D-14.4] Boolean conditions in #if directives use integer macros (CDD_CTRL_SELECT).
 *                       Rationale: compile-time controller selection; no runtime Boolean involved.
 *              [D-20.9] Use of #if is necessary for multi-controller build configuration.
 *                       Rationale: feature-selection pattern; no equivalent without directives.
 *
 * \version     1.0.0
 * \date        2025-01-01
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License. See LICENSE file in project root.
 *********************************************************************************************************************/

#ifndef CDD_APP_H
#define CDD_APP_H

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "Ifx_Types.h"          /* AURIX platform types: uint8, uint16, uint32, float32, boolean  */
#include "cdd_stm_app.h"        /* STM module interface   # C: Initialize_STM_Module()            */
#include "cdd_gpio_app.h"       /* GPIO module interface  # C: GPIO_Init_LED_P33()                */
#include "cdd_gtm_app.h"        /* GTM/PWM module interface # C: GTM_Init_PWM_TOM()               */


/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief  Module version — Major.Minor.Patch encoded as 0xMMmmPP [dimensionless] */
#define CDD_APP_VERSION         (0x010000UL)

/** \brief  Controller build-select token [dimensionless].
 *          Pass -DCDD_CTRL_SELECT=<n> on the compiler command line, or define here.
 *          0 = SMC  |  1 = DFC  |  2 = MPC                                        */
#ifndef CDD_CTRL_SELECT
    #define CDD_CTRL_SELECT     (2)     /* Default: MPC */
#endif

#define CDD_CTRL_SMC            (0)     /**< Sliding Mode Controller token          [dimensionless] */
#define CDD_CTRL_DFC            (1)     /**< Differential Flatness Controller token [dimensionless] */
#define CDD_CTRL_MPC            (2)     /**< Model Predictive Controller token      [dimensionless] */

/** \brief  Compile-time assert: CDD_CTRL_SELECT must be in {0,1,2}               */
/* MISRA C:2012 Rule 20.9 deviation — see file header deviation record [D-20.9]   */
#if ((CDD_CTRL_SELECT) != CDD_CTRL_SMC)  && \
    ((CDD_CTRL_SELECT) != CDD_CTRL_DFC)  && \
    ((CDD_CTRL_SELECT) != CDD_CTRL_MPC)
    #error "CDD_APP_H: CDD_CTRL_SELECT must be 0 (SMC), 1 (DFC), or 2 (MPC)."
#endif

/** \brief  FOC ISR period [us] — must match STM compare-match configuration       */
#define CDD_APP_FOC_PERIOD_US   (50.0F)

/** \brief  Number of sub-modules initialised by Initialize_Pmsm_App()            [dimensionless] */
#define CDD_APP_NUM_SUBMODULES  (5U)

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Initialisation status codes returned by CDD_App_GetInitStatus().
 */
typedef enum
{
    CDD_APP_INIT_OK          = 0U,  /**< All sub-modules initialised successfully  [dimensionless] */
    CDD_APP_INIT_PENDING     = 1U,  /**< Initialisation not yet called             [dimensionless] */
    CDD_APP_INIT_ERR_GPIO    = 2U,  /**< GPIO sub-module init failed               [dimensionless] */
    CDD_APP_INIT_ERR_STM     = 3U,  /**< STM sub-module init failed                [dimensionless] */
    CDD_APP_INIT_ERR_GTM     = 4U,  /**< GTM sub-module init failed                [dimensionless] */
    CDD_APP_INIT_ERR_ADC     = 5U,  /**< ADC sub-module init failed                [dimensionless] */
    CDD_APP_INIT_ERR_ENCODER = 6U   /**< Encoder sub-module init failed            [dimensionless] */
} CDD_App_InitStatus_t;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation.
 *
 * \details Initialises all CDD sub-modules in strict dependency order:
 *          GPIO → STM → GTM → ADC → ENCODER.
 *          Must be called exactly once from the OS startup hook, before any
 *          scheduler or ISR activation.  Re-entrant call behaviour is undefined.
 *
 *          Internal state: sets g_cdd_app_init_status to CDD_APP_INIT_OK on
 *          success, or to the first failing sub-module error code on failure.
 *
 * \pre     CPU clock and PLL configured; iLLD BSP initialised.
 * \post    All sub-module hardware is ready; STM ISR may fire after this returns.
 *
 * \return  void
 *
 * \note    MISRA C:2012 Rule 15.5 — single exit point maintained via status flag.
 *
 * # C: GPIO_Init_LED_P33(), Initialize_STM_Module(), GTM_Init_PWM_TOM(),
 *       ADC_Init_PhaseCurrents(), Encoder_Init_QEP()
 */
extern void Initialize_Pmsm_App(void);

/**
 * \brief   Return the initialisation status set by Initialize_Pmsm_App().
 *
 * \return  CDD_App_InitStatus_t  Status code [dimensionless].
 *          CDD_APP_INIT_PENDING if Initialize_Pmsm_App() has not been called.
 *
 * # C: g_cdd_app_init_status
 */
extern CDD_App_InitStatus_t CDD_App_GetInitStatus(void);

/**
 * \brief   Return the active controller token selected at build time.
 *
 * \return  uint8_t  CDD_CTRL_SMC (0), CDD_CTRL_DFC (1), or CDD_CTRL_MPC (2)
 *                   [dimensionless].
 */
//extern uint8_t CDD_App_GetActiveController(void);

#endif /* CDD_APP_H */
