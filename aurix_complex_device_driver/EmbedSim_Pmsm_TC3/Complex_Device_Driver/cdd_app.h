/**********************************************************************************************************************
 * \file        cdd_app.h
 * \brief       PMSM application top-level interface — CddApp_T is the central state hub.
 *
 * \details     CddApp_T is the single structure shared across all CDD sub-modules.
 *              It carries:
 *                  - Initialisation status of the application and the inverter
 *                  - Three-phase PWM duty cycles
 *                  - GTM carrier period and sample time derived quantities
 *
 *              Sub-modules initialised in dependency order by CddApp_Init():
 *                  1. GPIO     — LED diagnostic outputs (Port 33)
 *                  2. STM      — System Timer for 20 kHz FOC ISR scheduling
 *                  3. GTM      — CMU CLK0 + ATOM0 PWM (CH0–CH7), ISR armed (SRE=0)
 *                  4. INVERTER — QSPI4 init + TLE9180D startup (SPI config, normal mode)
 *                  5. START    — HOST_TRIG (PWM carrier live)
 *                  6. BRIDGE   — CddTle9180_AssertEnable() (ENA = HIGH)
 *                  7. ISR ARM  — SRC_GTM_ATOM0_0.B.SRE = 1U (20 kHz ISR fires)
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
 * \version     1.3.0
 * \date        2025-05-24
 * \author      EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright   Copyright (C) 2025 EmbedSim — EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *              Licensed under the MIT License.
 *********************************************************************************************************************/

#ifndef CDD_APP_H
#define CDD_APP_H

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"      /* embed_sim_sys_types.h + embed_sim_compiler.h pulled in here */
#include "cdd_stm_app.h"     /* CddStm_Init()                                               */
#include "cdd_gpio_app.h"    /* CddGpio_InitLed_P33(), CddGpio_ConfigQspi4Pins()            */
#include "cdd_gtm_app.h"     /* CddGtm_Init(), CddGtm_Start()                              */

/**********************************************************************************************************************
 * Macros
 *********************************************************************************************************************/

/** \brief  Module version  [dimensionless] */
#define CDDAPP_VERSION              (0x010300UL)

/** \brief  Controller build-select token.
 *          Pass -DCDD_CTRL_SELECT=<n> on the compiler command line, or define here.
 *          0 = SMC | 1 = DFC | 2 = MPC                                              */
#ifndef CDD_CTRL_SELECT
    #define CDD_CTRL_SELECT         (2)   /* Default: MPC */
#endif

#define CDD_CTRL_SMC                (0)   /**< Sliding Mode Controller token          [dimensionless] */
#define CDD_CTRL_DFC                (1)   /**< Differential Flatness Controller token [dimensionless] */
#define CDD_CTRL_MPC                (2)   /**< Model Predictive Controller token      [dimensionless] */

/** \brief  Compile-time guard: CDD_CTRL_SELECT must be in {0, 1, 2}                */
#if ((CDD_CTRL_SELECT) != CDD_CTRL_SMC)  && \
    ((CDD_CTRL_SELECT) != CDD_CTRL_DFC)  && \
    ((CDD_CTRL_SELECT) != CDD_CTRL_MPC)
    #error "CDD_APP_H: CDD_CTRL_SELECT must be 0 (SMC), 1 (DFC), or 2 (MPC)."
#endif

/** \brief  FOC ISR period  [us] — must match STM compare-match configuration        */
#define CDDAPP_FOC_PERIOD_US        (50.0F)

/**
 * \brief  Total number of sub-modules initialised in sequence by CddApp_Init().
 *         Compile-time assertion in cdd_app.c cross-checks this constant.
 */
#define CDDAPP_NUM_SUBMODULES       (7U)

/**
 * \brief  Convenience macro: direct lvalue access to the application init status.
 *
 * \details Expands to CddApp_G.CDDAppInitStatus — usable for both read and write.
 *          Retained for compatibility with legacy ISR and startup code.
 */
#define CDDAPP_INIT_STATUS_G        (CddApp_G.CDDAppInitStatus)

/**********************************************************************************************************************
 * Compile-time consistency check
 *********************************************************************************************************************/

/*
 * Verify that CDDAPP_NUM_SUBMODULES matches the seven documented startup steps.
 * Implemented as a preprocessor #if/#error so that it is valid under --iso=99 (C99).
 * _Static_assert is C11 and is not available with the TASKING --iso=99 build flag.
 * If CDDAPP_NUM_SUBMODULES is changed without updating CddApp_Init() (or vice versa),
 * the build fails immediately with the diagnostic below.
 */
#if (CDDAPP_NUM_SUBMODULES != 7U)
    #error "CDD_APP_H: CDDAPP_NUM_SUBMODULES must equal 7 — update macro and CddApp_Init() together."
#endif

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \brief  Application-level initialisation status codes.
 */
typedef enum
{
    CDDAPP_INIT_OK          = 0U,   /**< All sub-modules initialised successfully  [dimensionless] */
    CDDAPP_INIT_PENDING     = 1U,   /**< Initialisation not yet called             [dimensionless] */
    CDDAPP_INIT_ERR_GPIO    = 2U,   /**< GPIO sub-module init failed               [dimensionless] */
    CDDAPP_INIT_ERR_STM     = 3U,   /**< STM sub-module init failed                [dimensionless] */
    CDDAPP_INIT_ERR_GTM     = 4U,   /**< GTM sub-module init failed                [dimensionless] */
    CDDAPP_INIT_ERR_ADC     = 5U,   /**< ADC sub-module init failed                [dimensionless] */
    CDDAPP_INIT_ERR_ENCODER = 6U,   /**< Encoder sub-module init failed            [dimensionless] */
    CDDAPP_INIT_ERR_INV     = 7U    /**< Inverter (TLE9180D) init failed           [dimensionless] */
} CddApp_InitStatus_T;

/**
 * \brief  Inverter (TLE9180D) initialisation status codes.
 */
typedef enum
{
    CDDINV_INIT_PENDING = 0U,   /**< CddApp_InitInverter() not yet called       [dimensionless] */
    CDDINV_INIT_OK      = 1U,   /**< Device in normal operation mode            [dimensionless] */
    CDDINV_INIT_ERR     = 2U    /**< Startup sequence failed (ERR or SPI fault) [dimensionless] */
} CddApp_InvStatus_T;

/**
 * \brief  Central application state structure.
 *
 * \details All CDD sub-modules read and write through this single structure.
 *          The global instance CddApp_G is declared below and defined in cdd_app.c.
 *
 *          Duty cycle convention:
 *              0.0F — zero voltage (transistor fully OFF for that leg)
 *              0.5F — zero voltage vector (centre of carrier)
 *              1.0F — full voltage (transistor fully ON for that leg)
 */
typedef struct
{
    /** \brief  Application-level init status   [CddApp_InitStatus_T] */
    CddApp_InitStatus_T     CDDAppInitStatus;

    /** \brief  Inverter (TLE9180D) init status [CddApp_InvStatus_T]  */
    CddApp_InvStatus_T      CDDInverterStatus;

    /** \brief  Phase U PWM duty cycle          [0.0 .. 1.0]          */
    real32_T                DutyU;

    /** \brief  Phase V PWM duty cycle          [0.0 .. 1.0]          */
    real32_T                DutyV;

    /** \brief  Phase W PWM duty cycle          [0.0 .. 1.0]          */
    real32_T                DutyW;

    /** \brief  GTM ATOM0 carrier period in CMU CLK0 ticks
     *          = GTM_CMU_CLK0_FREQUENCY / CDD_CONTROL_LOOP_FREQUENCY  [CLK0 ticks] */
    uint32_T                PeriodTicks;

    /** \brief  Half of PeriodTicks — centre of carrier                [CLK0 ticks] */
    uint32_T                HalfPeriodTicks;

    /** \brief  Control loop sample time = 1 / CDD_CONTROL_LOOP_FREQUENCY  [s]      */
    real32_T                SampleTime;

} CddApp_T;

/**********************************************************************************************************************
 * Global Instance
 *********************************************************************************************************************/

/** \brief  Central application state — defined in cdd_app.c, shared across all CDDs. */
extern CddApp_T   CddApp_G;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation.
 *
 * \details Initialises all CDD sub-modules in strict dependency order:
 *              GPIO → STM → GTM → INVERTER → HOST_TRIG → BRIDGE ENABLE → ISR ARM
 *
 *          Must be called exactly once from the OS/startup hook before any
 *          scheduler or ISR activation.  A repeated or re-entrant call is
 *          silently ignored via the b_ok guard — no return value is needed.
 *
 * \pre     CPU clock and PLL configured; iLLD BSP initialised.
 * \post    GTM PWM carrier live; TLE9180D bridge outputs enabled; ISR firing at 20 kHz.
 *          CddApp_G.CDDAppInitStatus == CDDAPP_INIT_OK.
 *
 * \return  void
 */
void CddApp_Init(void);

/**
 * \brief   Initialises the power inverter sub-system (QSPI4 + TLE9180D).
 *
 * \details Sequence:
 *              1. CddQspi4_Init()            — QSPI4 master mode, 24-bit frame, ~5 MHz
 *              2. CddGpio_ConfigQspi4Pins()  — INH=LOW, ENA=LOW, /SOFF=HIGH
 *              3. CddTle9180_Startup()       — full TLE9180D power-up + SPI config batch
 *
 *          Sets CddApp_G.CDDInverterStatus = CDDINV_INIT_OK on completion.
 *
 * \note    CddTle9180_AssertEnable() (ENA = HIGH) is NOT called here.
 *          It is called by CddApp_Init() at Step 6, after CddGtm_Start() at Step 5,
 *          so that bridge output transistors are never energised before the PWM
 *          carrier is live.
 *
 * \return  void
 */
void CddApp_InitInverter(void);

/**
 * \brief   Re-triggers GTM HOST_TRIG if the application has been fully initialised.
 *
 * \return  0x1U if CddGtm_Start() was called successfully.
 *          0x0U if CddApp_Init() has not yet completed.
 */
uint32_T CddApp_Start(void);

/**
 * \brief   Returns the current application initialisation status.
 * \return  CddApp_InitStatus_T — current value of CddApp_G.CDDAppInitStatus.
 */
CddApp_InitStatus_T CddApp_GetInitStatus(void);

#endif /* CDD_APP_H */
