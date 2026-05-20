/**********************************************************************************************************************
 * \file        cdd_app.h
 * \brief       PMSM application top-level interface — CDD_APP_t is the central state hub.
 *
 * \details     CDD_APP_t is the single structure shared across all CDD sub-modules.
 *              It carries:
 *                  - Initialisation status of the application and the inverter
 *                  - Three-phase PWM duty cycles (replaces the retired GTM_PWM_Duty_T)
 *
 *              Sub-modules initialised in dependency order by Initialize_Pmsm_App():
 *                  1. GPIO    — LED diagnostic outputs (Port 33)
 *                  2. STM     — System Timer for 20 kHz FOC ISR scheduling
 *                  3. GTM     — CMU CLK0 + ATOM0 PWM (CH0–CH7), ISR armed (SRE=0)
 *                  4. INVERTER— QSPI4 init + TLE9180D startup (SPI config, normal mode)
 *                  5. START   — HOST_TRIG (PWM live) → bridge enable → ISR arm (SRE=1)
 *
 *              Controller build selected at compile-time via CDD_CTRL_SELECT:
 *                  CDD_CTRL_SMC (0) — Sliding Mode Controller
 *                  CDD_CTRL_DFC (1) — Differential Flatness Controller
 *                  CDD_CTRL_MPC (2) — Model Predictive Controller (default)
 *
 * \note        MISRA C:2012 deviation record:
 *              [D-14.4] #if directives use integer macros (CDD_CTRL_SELECT).
 *              [D-20.9] #if is necessary for multi-controller build selection.
 *
 * \version     1.1.0
 * \date        2025-05-18
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef CDD_APP_H
#define CDD_APP_H

/*********************************************************************************************************************/
/*-----------------------------------------------------Includes------------------------------------------------------*/
/*********************************************************************************************************************/
#include "embed_sim_sys_types.h"   /* uint8_T, uint32_T, real32_T, boolean_T     */
#include "cdd_stm_app.h"           /* Initialize_STM_Module()                     */
#include "cdd_gpio_app.h"          /* GPIO_Init_LED_P33()                         */
#include "cdd_gtm_app.h"           /* Initialize_GTM_Module(), Start_GTM_Module() */

/*********************************************************************************************************************/
/*------------------------------------------------------Macros-------------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief  Module version [dimensionless] */
#define CDD_APP_VERSION             (0x010100UL)

/** \brief  Controller build-select token.
 *          Pass -DCDD_CTRL_SELECT=<n> on the compiler command line, or define here.
 *          0 = SMC | 1 = DFC | 2 = MPC                                             */
#ifndef CDD_CTRL_SELECT
    #define CDD_CTRL_SELECT         (2)   /* Default: MPC */
#endif

#define CDD_CTRL_SMC                (0)   /**< Sliding Mode Controller token          [dimensionless] */
#define CDD_CTRL_DFC                (1)   /**< Differential Flatness Controller token [dimensionless] */
#define CDD_CTRL_MPC                (2)   /**< Model Predictive Controller token      [dimensionless] */

/** \brief  Compile-time guard: CDD_CTRL_SELECT must be in {0, 1, 2}               */
#if ((CDD_CTRL_SELECT) != CDD_CTRL_SMC)  && \
    ((CDD_CTRL_SELECT) != CDD_CTRL_DFC)  && \
    ((CDD_CTRL_SELECT) != CDD_CTRL_MPC)
    #error "CDD_APP_H: CDD_CTRL_SELECT must be 0 (SMC), 1 (DFC), or 2 (MPC)."
#endif

/** \brief  FOC ISR period [us] — must match STM compare-match configuration       */
#define CDD_APP_FOC_PERIOD_US       (50.0F)

/** \brief  Number of sub-modules initialised by Initialize_Pmsm_App()            */
#define CDD_APP_NUM_SUBMODULES      (5U)

/**
 * \brief  Convenience macro: direct access to the application init status field.
 *
 * \details Allows legacy-style code to write/read CDD_APP_INIT_STATUS_G without
 *          knowing the struct field name.  Expands to an lvalue; safe for both
 *          read and assignment.
 */
#define CDD_APP_INIT_STATUS_G       (CDD_App_G.CDDAppInitStatus)

/*********************************************************************************************************************/
/*-------------------------------------------------Data Structures---------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief  Application-level initialisation status codes.
 */
typedef enum
{
    CDD_APP_INIT_OK          = 0U,   /**< All sub-modules initialised successfully  [dimensionless] */
    CDD_APP_INIT_PENDING     = 1U,   /**< Initialisation not yet called             [dimensionless] */
    CDD_APP_INIT_ERR_GPIO    = 2U,   /**< GPIO sub-module init failed               [dimensionless] */
    CDD_APP_INIT_ERR_STM     = 3U,   /**< STM sub-module init failed                [dimensionless] */
    CDD_APP_INIT_ERR_GTM     = 4U,   /**< GTM sub-module init failed                [dimensionless] */
    CDD_APP_INIT_ERR_ADC     = 5U,   /**< ADC sub-module init failed                [dimensionless] */
    CDD_APP_INIT_ERR_ENCODER = 6U,   /**< Encoder sub-module init failed            [dimensionless] */
    CDD_APP_INIT_ERR_INV     = 7U    /**< Inverter (TLE9180D) init failed           [dimensionless] */
} CDD_App_InitStatus_t;

/**
 * \brief  Inverter (TLE9180D) initialisation status codes.
 */
typedef enum
{
    CDD_INV_INIT_PENDING = 0U,   /**< GD9180_Startup() not yet called            [dimensionless] */
    CDD_INV_INIT_OK      = 1U,   /**< Device in normal operation mode            [dimensionless] */
    CDD_INV_INIT_ERR     = 2U    /**< Startup sequence failed (ERR or SPI fault) [dimensionless] */
} CDD_Inverter_Status_t;

/**
 * \brief  Central application state structure.
 *
 * \details All CDD sub-modules read and write through this single structure.
 *          The global instance CDD_App_G is declared below and defined in cdd_app.c.
 *
 *          Three-phase PWM duty cycles (DutyU / DutyV / DutyW) replace the
 *          retired GTM_PWM_Duty_T.  The open-loop controller and (later) the
 *          closed-loop FOC controller both write these fields; the GTM driver
 *          reads them via GTM_Set_PWM_Duty().
 *
 *          Duty cycle convention:
 *              0.0F — zero voltage (transistor fully OFF for that leg)
 *              0.5F — zero voltage vector (centre of carrier)
 *              1.0F — full voltage (transistor fully ON for that leg)
 */
typedef struct
{
    /** \brief  Application-level init status   [CDD_App_InitStatus_t] */
    CDD_App_InitStatus_t    CDDAppInitStatus;

    /** \brief  Inverter (TLE9180D) init status [CDD_Inverter_Status_t] */
    CDD_Inverter_Status_t   CDDInverterStatus;

    /** \brief  Phase U PWM duty cycle          [0.0 .. 1.0] */
    real32_T                DutyU;

    /** \brief  Phase V PWM duty cycle          [0.0 .. 1.0] */
    real32_T                DutyV;

    /** \brief  Phase W PWM duty cycle          [0.0 .. 1.0] */
    real32_T                DutyW;

    /** \brief  GTM ATOM0 carrier period in CMU CLK0 ticks
     *          = GTM_CMU_CLK0_FREQUENCY / CDD_CONTROL_LOOP_FREQUENCY  [CLK0 ticks] */
    uint32_T                PeriodTicks;

    /** \brief  Half of PeriodTicks — centre of carrier  [CLK0 ticks] */
    uint32_T                HalfPeriodTicks;

    /** \brief  Control loop sample time = 1 / CDD_CONTROL_LOOP_FREQUENCY  [s] */
    real32_T                SampleTime;

} CDD_APP_t;

/*********************************************************************************************************************/
/*------------------------------------------------Global Instance-----------------------------------------------------*/
/*********************************************************************************************************************/

/** \brief  Central application state — defined in cdd_app.c, shared across all CDDs. */
extern CDD_APP_t   CDD_App_G;

/*********************************************************************************************************************/
/*------------------------------------------------Function Prototypes------------------------------------------------*/
/*********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation.
 *
 * \details Initialises all CDD sub-modules in strict dependency order:
 *              GPIO → STM → GTM → INVERTER → START → ISR ARM
 *
 *          Must be called exactly once from the OS/startup hook before any
 *          scheduler or ISR activation.  Re-entrant call is rejected silently.
 *
 * \pre     CPU clock and PLL configured; iLLD BSP initialised.
 * \post    GTM PWM is live, TLE9180D bridge outputs enabled, ISR firing at 20 kHz.
 *
 * \return  void
 */
extern void Initialize_Pmsm_App(void);

/**
 * \brief   Initialises the power inverter sub-system (QSPI4 + TLE9180D).
 *
 * \details Sequence:
 *              1. CDD_Qspi_Init()      — QSPI4 master mode, 24-bit, ~5 MHz
 *              2. GD9180_Init_Pins()   — INH=LOW, ENA=LOW, /SOFF=HIGH (safe default)
 *              3. GD9180_Startup()     — full TLE9180D power-up + SPI config batch
 *
 *          Sets CDD_App_G.CDDInverterStatus = CDD_INV_INIT_OK on success,
 *          CDD_INV_INIT_ERR on failure.
 *
 *          Note: GD9180_Enable_Outputs() (ENA = HIGH) is NOT called here.
 *          It is called by Initialize_Pmsm_App() after Start_GTM_Module()
 *          so that the bridge outputs are never enabled before PWM is live.
 *
 * \return  void
 */
extern void Initialize_Inverter(void);


extern  uint32_T Start_Pmsm_App(void);


/**
 * \brief   Returns the current application initialisation status.
 * \return  CDD_App_InitStatus_t
 */
extern CDD_App_InitStatus_t CDD_App_GetInitStatus(void);

#endif /* CDD_APP_H */
