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
 *                  3. INVERTER — CddTle9180_Startup(): QSPI4 init + TLE9180D GPIO
 *                                power-on sequence + SPI configuration batch +
 *                                NORMAL mode verification (IsNormalMode)
 *                  4. GTM CMU  — Module clock enable + CMU CLK0 = 200 MHz
 *                  5. GTM ATOM — ATOM0 CH0–CH5 complementary PWM init; ISR armed (SRE=0)
 *                  6. START    — CddGtm_Start(): HOST_TRIG (PWM carrier live)
 *                  7. BRIDGE   — CddTle9180_AssertEnable() (ENA = HIGH; gate drive active)
 *                  8. ISR ARM  — SRC_GTM_ATOM0_0.B.SRE = 1U (20 kHz FOC ISR fires)
 *
 *              This ordering guarantees that bridge output transistors are never
 *              energised before the GTM PWM carrier is live, and the ISR never fires
 *              before the inverter has reached NORMAL operating mode.
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
 * \version     1.4.0
 * \date        2026-07-04
 *
 * \par v1.4.0
 *   Redesign: CddApp_T is the single central object —
 *     - CtrlMode (CDDAPP_CTRL_OPENLOOP / CDDAPP_CTRL_CLOSEDLOOP) and
 *       SpeedRefRpm moved here; set via CddApp_SetCtrlMode() /
 *       CddApp_SetSpeedRefRpm() before CddApp_Start(); the mode is latched
 *       once by the ISR — no switching during operation.
 *     - EVADC measurements (Meas: raw sense voltages + DC link [V]) and the
 *       converted PhaseCurrents [A] moved here, updated every ISR tick.
 *   The DFC loop-option layer (A/B) is removed from the public API:
 *   CLOSEDLOOP always runs the full sensorless DFC sequence.
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
#include "cdd_config.h"         /* embed_sim_sys_types.h + embed_sim_compiler.h pulled in here */
#include "cdd_tle9180_app.h"    /* CddTle9180_T, CddTle9180_Startup(), CddTle9180_AssertEnable */

/**********************************************************************************************************************
 * Macros
 *********************************************************************************************************************/

/** \brief  Module version  [dimensionless] */
#define CDDAPP_VERSION              (0x010400UL)

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
 * \brief  Total number of sub-module startup steps in CddApp_Init().
 *         v1.4.0: +1 for the control-loop layer (Transform + DFC).
 *         Compile-time assertion below cross-checks this constant.
 */
#define CDDAPP_NUM_SUBMODULES       (9U)

/**********************************************************************************************************************
 * Compile-time consistency check
 *********************************************************************************************************************/

/*
 * Verify that CDDAPP_NUM_SUBMODULES matches the nine documented startup steps.
 * Implemented as a preprocessor #if/#error so that it is valid under --iso=99 (C99).
 * _Static_assert is C11 and is not available with the TASKING --iso=99 build flag.
 */
#if (CDDAPP_NUM_SUBMODULES != 9U)
    #error "CDD_APP_H: CDDAPP_NUM_SUBMODULES must equal 9 — update macro and CddApp_Init() together."
#endif


/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \brief  Application-level initialisation status codes.
 *
 * \details Values are assigned in CddApp_Init() in dependency order so that
 *          the last written value identifies exactly which step was reached
 *          before a failure — useful for debugger inspection and DTC mapping.
 */
typedef enum
{
    CDDAPP_INIT_PENDING        =    0U,    /**< Initialisation not yet called             [dimensionless] */
    CDDAPP_INIT_ERR_CLK        =    2U,    /**< CPU clock frequency check failed          [dimensionless] */
    CDDAPP_INIT_ERR_STM        =    4U,    /**< STM frequency check failed                [dimensionless] */
    CDDAPP_INIT_DONE_STM       =   10U,    /**< STM sub-modules initialised               [dimensionless] */
    CDDAPP_INIT_ERR_ADC        =   12U,    /**< ADC frequency check failed                [dimensionless] */
    CDDAPP_INIT_DONE_ADC       =   13U,    /**< ADC-modules initialised                   [dimensionless] */
    CDDAPP_INIT_ERR_GPT12      =   19U,    /**< GPT12 initialisation error                [dimensionless] */
    CDDAPP_INIT_DONE_GPT12     =   21U,    /**< GPT12 initialised                         [dimensionless] */
    CDDAPP_INIT_ERR_INV        =   25U,    /**< TLE9180D startup failed                   [dimensionless] */
    CDDAPP_INIT_DONE_INV       =   30U,    /**< Inverter (TLE9180D) reached NORMAL mode   [dimensionless] */
    CDDAPP_INIT_ERR_GTM        =   45U,    /**< GTM frequency or CMU CLK0 check failed    [dimensionless] */
    CDDAPP_INIT_DONE_GTM       =   55U,    /**< GTM CMU + ATOM0 PWM initialised           [dimensionless] */
    CDDAPP_INIT_ERR_CTRL       =   65U,    /**< Control-loop init (DFC/transform) failed  [dimensionless] */
    CDDAPP_INIT_DONE_CTRL      =   75U,    /**< Transform + DFC controller initialised    [dimensionless] */
    CDDAPP_INIT_OK             =  100U,    /**< All sub-modules initialised successfully  [dimensionless] */
    CDDAPP_CALIBRATE_OK        =  101U,    /**< All sub-modules initialised successfully  [dimensionless] */
    CDDAPP_RUN_STATE           =  105U,    /**< All sub-modules initialised successfully  [dimensionless] */
    CDDAPP_ERROR_STATE         =  110U
} CddApp_Status_T;

/**
 * \brief  Application-level Diagnostic Trouble Codes.
 */
typedef enum
{
    CDDAPP_DTC_NONE            =  0U,   /**< No fault                                   [dimensionless] */
    CDDAPP_DTC_ERR             =  1U,   /**< General Error                              [dimensionless] */
    CDDAPP_DTC_CPU_FREQ        =  5U,   /**< CPU frequency not 300 MHz                  [dimensionless] */
    CDDAPP_DTC_STM_FREQ        = 15U,   /**< STM frequency not 100 MHz                  [dimensionless] */
    CDDAPP_DTC_ADC_FREQ        = 16U,   /**< ADC frequency not 160 MHz                  [dimensionless] */
    CDDAPP_DTC_QSPI_FREQ       = 18U,   /**< fPeriph (QSPI source) not 200 MHz          [dimensionless] */
    CDDAPP_DTC_INV_STARTUP     = 22U,   /**< TLE9180D did not reach NORMAL mode         [dimensionless] */
    CDDAPP_DTC_GTM_FREQ        = 25U,   /**< GTM clock not 200 MHz                      [dimensionless] */
    CDDAPP_DTC_GTM_CMU0_FREQ   = 28U,   /**< GTM CMU CLK0 not 200 MHz after programming [dimensionless] */
    CDDAPP_DTC_CTRL_INIT       = 32U    /**< DFC_Init() failed (control-loop layer)     [dimensionless] */
} CddApp_DTC_T;

/**
 * \enum   CddApp_CtrlMode_T
 * \brief  Control mode executed by the 20 kHz ISR — exactly two options.
 *
 * \details Selected BEFORE CddApp_Start() via CddApp_SetCtrlMode(); the ISR
 *          latches the mode once on the activation edge and it is fixed for
 *          the entire run — no switching during operation.  To change the
 *          mode: stop, set, restart.  Both modes consume SpeedRefRpm.
 *
 *          CDDAPP_CTRL_OPENLOOP is the reset default (value 0, .bss safe):
 *          V/f rotating vector at the ramped speed reference, no current
 *          feedback — the commissioning / bring-up path.
 *          CDDAPP_CTRL_CLOSEDLOOP runs the full sensorless DFC (its internal
 *          ALIGN → I-f → CLOSEDLOOP startup sequence, then flatness FOC).
 */
typedef enum
{
    CDDAPP_CTRL_OPENLOOP   = 0U,   /**< V/f rotating vector at SpeedRefRpm, no feedback. */
    CDDAPP_CTRL_CLOSEDLOOP = 1U    /**< flatness FOC (DFC), full closed loop. */
} CddApp_CtrlMode_T;

/**
 * \brief  Central application state structure.
 *
 * \details All CDD sub-modules read and write through this single structure.
 *          The global instance CddApp_G is declared below and defined in cdd_app.c.
 *          Zero-initialised by C startup (.bss); CDDAPP_INIT_PENDING = 0U so the
 *          re-entrant guard in CddApp_Init() is valid from reset without an explicit
 *          initialiser.
 *
 *          Duty cycle convention:
 *              0.0F — zero voltage (leg fully OFF)
 *              0.5F — zero voltage vector (centre of symmetrical carrier)
 *              1.0F — full voltage (leg fully ON)
 */
typedef struct
{
    /** \brief  Application-level status                               [CddApp_Status_T] */
    CddApp_Status_T         CDDAppStatus;

    /** \brief  Active Diagnostic Trouble Code                         [CddApp_DTC_T]    */
    CddApp_DTC_T            DTC;

    /* ------------------------------------------------------------------
     * Control command block  (host → ISR)
     * ------------------------------------------------------------------ */

    /** \brief  Selected control mode — OPENLOOP or CLOSEDLOOP.  Written by
     *          CddApp_SetCtrlMode() before start; latched once by the ISR
     *          on the activation edge (no switching during operation).
     *          Atomic 32-bit store on TriCore.       [CddApp_CtrlMode_T]               */
    CddApp_CtrlMode_T       CtrlMode;

    /** \brief  Mechanical speed reference, consumed by BOTH control modes.
     *          Live-settable via CddApp_SetSpeedRefRpm() (atomic store);
     *          the open-loop path slew-limits it internally, the DFC clamps
     *          it to ±DFC_OMEGA_CMD_MAX.                              [RPM]            */
    real32_T                SpeedRefRpm;

    /* ------------------------------------------------------------------
     * PWM outputs and timing
     * ------------------------------------------------------------------ */

    /** \brief  Phase U PWM duty cycle                                 [0.0 .. 1.0]      */
    real32_T                DutyU;

    /** \brief  Phase V PWM duty cycle                                 [0.0 .. 1.0]      */
    real32_T                DutyV;

    /** \brief  Phase W PWM duty cycle                                 [0.0 .. 1.0]      */
    real32_T                DutyW;

    /** \brief  ADC Trig PWM duty cycle                                 [0.0 .. 1.0]     */
    real32_T                DutyAdcTrig;

    /** \brief  Current Phase U                                        [A]               */
    real32_T                Iu;

    /** \brief  Current Phase V                                        [A]               */
    real32_T                Iv;

    /** \brief  Current Phase W                                        [A]               */
    real32_T                Iw;

    /** \brief  Current Offset Phase U                                 [A]               */
    real32_T                OffsetIu;

    /** \brief  Current Offset Phase V                                 [A]               */
    real32_T                OffsetIv;

    /** \brief  Current Offset Phase V                                 [A]               */
    real32_T                OffsetIw;

    /** \brief  Voltage Offset Reference                               [V]              */
    real32_T                OffsetVro;

    /** \brief  Sum of Current Phases                                  [A]               */
    real32_T                Isum;

    /** \brief  ADC Voltage Phase U                                    [V]               */
    real32_T                Vu;

    /** \brief  ADC Voltage Phase V                                    [V]               */
    real32_T                Vv;

    /** \brief  ADC Voltage Phase W                                    [V]               */
    real32_T                Vw;

    /** \brief  Sensor Validity Bitfields                              [V]               */
    uint32_T                SensorReadingBitField;

    /** \brief  Rotor Velocity in RPM                                  [RPM]             */
    real32_T                RotorSpeedRpm;

    /** \brief  Rotor Position                                                           */
    real32_T                RotorPosition;

    /** \brief  ADC DC  Voltage                                        [V]              */
    real32_T                Vdc;

    /** \brief  ADC Reference Voltage                                  [V]              */
     real32_T               Vro;

    /** \brief  GTM ATOM0 carrier period in CMU CLK0 ticks
     *          = GTM_CMU_CLK0_FREQUENCY / CDD_CONTROL_LOOP_FREQUENCY [CLK0 ticks]      */
    uint32_T                PeriodTicks;

    /** \brief  Half of PeriodTicks — midpoint of the symmetrical carrier
     *         used as the ATOM compare value for the zero-voltage vector [CLK0 ticks]  */
    uint32_T                HalfPeriodTicks;

    /** \brief  Control loop sample time = 1 / CDD_CONTROL_LOOP_FREQUENCY  [s]          */
    real32_T                SampleTime;

    /** \brief  TLE9180D gate driver runtime handle                                     */
    CddTle9180_T            Inverter;

    /** \brief  Control loop Counter                                                    */
    uint64_T                ControlLoopCounter;

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
 *              GPIO → STM → INVERTER → GTM CMU → GTM ATOM →
 *              HOST_TRIG → BRIDGE ENABLE → ISR ARM
 *
 *          Must be called exactly once from the OS/startup hook before any
 *          scheduler or ISR activation.  A repeated or re-entrant call is
 *          silently ignored via the b_ok guard.
 *
 * \pre     CPU clock and PLL configured; iLLD BSP initialised.
 * \post    GTM PWM carrier live; TLE9180D gate drive enabled; ISR firing at 20 kHz.
 *          CddApp_G.CDDAppStatus == CDDAPP_INIT_OK.
 *
 * \return  void
 */
extern void CddApp_Init(void);

/**
 * \brief   Re-triggers GTM HOST_TRIG, asserts bridge ENA, and arms the FOC ISR.
 *
 * \details Called only after CddApp_Init() has completed successfully.
 *          Sequence:
 *              1. CddGtm_Start()               — HOST_TRIG; PWM carrier goes live
 *              2. CddTle9180_AssertEnable()    — ENA = HIGH; gate drive outputs active
 *              3. SRC_GTM_ATOM0_0.B.SRE = 1U  — 20 kHz FOC ISR begins firing
 *
 * \return  0x1U if all three steps were executed (CDDAppStatus == CDDAPP_INIT_OK).
 *          0x0U if CddApp_Init() has not yet completed.
 */
extern uint32_T CddApp_Start(void);

/**
 * \brief   Returns the current application initialisation status.
 * \return  CddApp_Status_T — current value of CddApp_G.CDDAppStatus.
 */
extern CddApp_Status_T CddApp_GetInitStatus(void);

/**
 * \brief   Selects the control mode — OPENLOOP or CLOSEDLOOP (atomic store).
 *
 * \details Accepted only while the application is NOT in CDDAPP_RUN_STATE.
 *          The 20 kHz ISR latches the mode exactly once on the activation
 *          edge; it is fixed for the entire run — no switching during
 *          operation.  To change the mode: stop, set, restart.
 *          Requests during RUN, or invalid enum values, are silently ignored.
 *
 *          Host sequence:
 *              CddApp_Init();
 *              CddApp_SetCtrlMode(CDDAPP_CTRL_CLOSEDLOOP);  // or OPENLOOP — final
 *              CddApp_SetSpeedRefRpm(1500.0F);
 *              CddApp_Start();
 *
 * \param[in]  Mode  CDDAPP_CTRL_OPENLOOP or CDDAPP_CTRL_CLOSEDLOOP.
 */
extern void CddApp_SetCtrlMode(const CddApp_CtrlMode_T Mode);

/**
 * \brief   Returns the currently selected control mode.
 * \return  CddApp_CtrlMode_T
 */
extern CddApp_CtrlMode_T CddApp_GetCtrlMode(void);

/**
 * \brief   Sets the mechanical speed reference (atomic 32-bit store).
 *
 * \details Consumed by both control modes and live-settable at any time:
 *          the open-loop path slew-limits changes internally (no step in the
 *          rotating vector frequency); the DFC clamps to ±DFC_OMEGA_CMD_MAX.
 *
 * \param[in]  RotorSpeedRpm  Mechanical speed reference  [RPM]
 */
extern void CddApp_SetSpeedRefRpm(const real32_T SpeedRpm);

#endif /* CDD_APP_H */
