/**********************************************************************************************************************
 * \file      cdd_app.h
 * \brief     Public interface for the top-level PMSM application layer.
 *
 * \details   Declares the central CddApp_T state structure, the CddApp_G
 *            global instance, and the application lifecycle functions
 *            (Init, Start, GetInitStatus), the control-mode selector, and the
 *            speed-reference setter.
 *
 *            For the ATOM0 channel assignment, dead-time equations, and the
 *            centre-aligned carrier / valley-trigger discussion, see
 *            cdd_gtm_app.h.
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

/** \brief  FOC ISR period  [us].
 *
 *  \note   Must equal 1 / CDD_CONTROL_LOOP_FREQUENCY.  The match is not
 *          enforced at compile time. */
#define CDDAPP_FOC_PERIOD_US        (50.0F)

/**********************************************************************************************************************
 * Data Structures
 *********************************************************************************************************************/

/**
 * \brief  Application-level initialisation status codes.
 *
 * \details Written in CddApp_Init() as each sub-module completes; the last
 *          value observed in a debugger identifies the step reached before a
 *          failure.
 *
 * \note   The numeric values do NOT track the execution order of CddApp_Init().
 *         The function runs STM → ADC → GPT12 → GTM → INV, whereas the enum
 *         orders STM → ADC → GPT12 → INV (30U) → GTM (55U).  DONE_INV is
 *         written after DONE_GTM.  Do not rely on value ordering to infer
 *         progress.
 *
 * \note   CDDAPP_INIT_DONE_CTRL, CDDAPP_INIT_ERR_CTRL, CDDAPP_CALIBRATE_OK and
 *         CDDAPP_ERROR_STATE are declared but never written anywhere in
 *         cdd_app.c; treat them as reserved placeholders.
 *
 * \warning CDDAPP_INIT_PENDING == 0U is load-bearing: the guard in
 *          CddApp_Init() relies on .bss zero-init at reset.  Do not change the
 *          value without auditing that function.
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
    CDDAPP_ERROR_STATE         =  110U     /**< Unrecoverable application-level error     [dimensionless] */
} CddApp_Status_T;

/**
 * \brief  Application-level Diagnostic Trouble Codes.
 *
 * \note   DTC is only meaningful when CDDAppStatus is one of the *_ERR_*
 *         values.  On every successful step in CddApp_Init(), DTC is reset
 *         to CDDAPP_DTC_NONE.
 *
 * \note   CDDAPP_DTC_CTRL_INIT is declared but never assigned in cdd_app.c.
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
 *          the entire run — no switching during operation.  Both modes consume
 *          SpeedRefRpm.
 *
 *          CDDAPP_CTRL_OPENLOOP is the reset default (value 0, .bss safe):
 *          V/f rotating vector at the ramped speed reference, no current
 *          feedback — the commissioning / bring-up path.
 *          CDDAPP_CTRL_DFC_CLOSEDLOOP runs the full sensorless DFC (its internal
 *          ALIGN → I-f → CLOSEDLOOP startup sequence, then flatness FOC).
 *
 * \warning The "stop, set, restart" cycle implied above is not supported by
 *          the current cdd_app.c: CddApp_Start() is a permanent no-op after
 *          its first success, and there is no CddApp_Stop().  "Restart" in
 *          this header is aspirational.
 */
typedef enum
{
    CDDAPP_CTRL_OPENLOOP       = 0x0U,   /**< V/f rotating vector at SpeedRefRpm, no feedback, only for Test , under 800 RPM */
    CDDAPP_CTRL_DFC_CLOSEDLOOP = 0x1U    /**< flatness FOC (DFC), full closed loop. */
} CddApp_CtrlMode_T;

/**
 * \brief  Central application state structure.
 *
 * \details All CDD sub-modules read and write through this single structure.
 *          The global instance CddApp_G is declared below and defined in
 *          cdd_app.c.  Zero-initialised by C startup (.bss); CDDAPP_INIT_PENDING
 *          = 0U so the guard in CddApp_Init() is valid from reset without an
 *          explicit initialiser.
 *
 *          Duty cycle convention:
 *              0.0F — zero voltage (leg fully OFF)
 *              0.5F — zero voltage vector (centre of symmetrical carrier)
 *              1.0F — full voltage (leg fully ON)
 *
 * \warning This structure is shared between task context (writers) and the
 *          20 kHz FOC ISR (reader) and is not declared volatile.  Aligned
 *          32-bit accesses are atomic on TriCore so field tearing is not a
 *          concern, but the compiler caching / memory-order contract is not
 *          stated here and should be reviewed before relying on it.
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

    /** \brief  ADC Trig PWM duty cycle                                [0.0 .. 1.0]
     *  \note   Never written by CddApp_Init() and not referenced in the visible
     *          cdd_app.c; confirm the producer.                                         */
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

    /** \brief  Current Offset Phase W                                 [A]               */
    real32_T                OffsetIw;

    /** \brief  Voltage Offset Reference                               [V]
     *  \note   Never written by CddApp_Init(); relies on .bss zero.                     */
    real32_T                OffsetVro;

    /** \brief  Sum of Current Phases                                  [A]               */
    real32_T                Isum;

    /** \brief  ADC Voltage Phase U                                    [V]               */
    real32_T                Vu;

    /** \brief  ADC Voltage Phase V                                    [V]               */
    real32_T                Vv;

    /** \brief  ADC Voltage Phase W                                    [V]               */
    real32_T                Vw;

    /** \brief  Sensor validity bitfield.
     *  \warning The original unit tag "[V]" is incorrect for a bitfield; removed.        */
    uint32_T                SensorReadingBitField;

    /** \brief  Rotor Velocity in RPM                                  [RPM]
     *  \note   Measured quantity.  Contrast with SpeedRefRpm, which is the command.      */
    real32_T                RotorSpeedRpm;

    /** \brief  Rotor Position                                                           */
    real32_T                RotorPosition;

    /** \brief  ADC DC  Voltage                                        [V]              */
    real32_T                Vdc;

    /** \brief  ADC Reference Voltage                                  [V]              */
     real32_T               Vro;

    /** \brief  GTM ATOM0 carrier period in CMU CLK0 ticks
     *          = GTM_CMU_CLK0_FREQUENCY / CDD_CONTROL_LOOP_FREQUENCY [CLK0 ticks]
     *  \note   Not written by CddApp_Init(); confirm it is populated by the GTM
     *          layer before any consumer reads it.                                      */
    uint32_T                PeriodTicks;

    /** \brief  Half of PeriodTicks — midpoint of the symmetrical carrier
     *         used as the ATOM compare value for the zero-voltage vector [CLK0 ticks]
     *  \note   Same producer caveat as PeriodTicks.                                      */
    uint32_T                HalfPeriodTicks;

    /** \brief  Control loop sample time = 1 / CDD_CONTROL_LOOP_FREQUENCY  [s]
     *  \note   Not written by CddApp_Init(); confirm producer before use.                */
    real32_T                SampleTime;

    /** \brief  TLE9180D gate driver runtime handle                                     */
    CddTle9180_T            Inverter;

    /** \brief  Control loop Counter
     *  \note   Not written by CddApp_Init(); incremented elsewhere (ISR or
     *          control layer).                                                           */
    uint64_T                ControlLoopCounter;

} CddApp_T;

/**********************************************************************************************************************
 * Global Instance
 *********************************************************************************************************************/

/**
 * \brief  Central application state — defined in cdd_app.c, shared across all CDDs.
 *
 * \warning Shared between task and ISR contexts; not declared volatile.  See
 *          the discussion on CddApp_T.
 */
extern CddApp_T   CddApp_G;

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Top-level PMSM application initialisation.
 *
 * \details Runs CDD sub-module init in the order shown below, gated by a local
 *          ok flag so that no step runs after a failure.  Single exit point
 *          (MISRA C:2012 Rule 15.5).
 *
 *              1. CPU-freq check (300 MHz)
 *              2. STM-freq check (100 MHz)  → CddStm_Init()
 *              3. ADC-freq check (160 MHz)  → CddEvadc_Init()
 *              4. CddEncoder_Init()
 *              5. GTM-freq check (200 MHz)  → CddGtm_InitModule()
 *                                           → CMU CLK0 check
 *                                           → CddGtm_InitInverter()
 *              6. QSPI-freq check (200 MHz) → CddApp_InitInverter()
 *
 *          GPIO is not initialised here; it is assumed done by the BSP.
 *
 * \pre     CPU/STM/ADC/GTM/QSPI clocks and PLL configured; iLLD BSP initialised.
 *
 * \post    On success: CDDAppStatus == CDDAPP_INIT_DONE_INV and DTC == DTC_NONE.
 *          Note that CDDAPP_INIT_OK is written by CddApp_Start(), not by this
 *          function; the PWM carrier is NOT live, the gate drive is NOT enabled,
 *          and the ISR is NOT armed after Init alone.
 *
 * \note    One-shot: a repeated call after any terminal status — success OR
 *          failure — is silently ignored.  There is no retry path.
 *
 * \warning Do not confuse the end state of Init with that of Start.
 */
extern void CddApp_Init(void);

/**
 * \brief   Completes the startup sequence: safe-off → ENA → release safe-off
 *          → GTM carrier live.
 *
 * \details Called only after CddApp_Init() has reached CDDAPP_INIT_DONE_INV.
 *          Sequence executed:
 *              1. CddTle9180_AssertSafeOff()
 *              2. CddTle9180_AssertEnable()
 *              3. CddTle9180_DeassertSafeOff()
 *              4. CddGtm_Start()
 *              5. CDDAppStatus = CDDAPP_INIT_OK
 *
 * \note    One-shot: the internal started flag is never cleared, so subsequent
 *          calls are no-ops.  No stop → restart path exists in this file.
 *
 * \warning The FOC ISR is NOT armed here.  SRC_GTM_ATOM0_0.B.SRE does not
 *          appear in cdd_app.c; if the ISR is armed at all, that happens inside
 *          CddGtm_Start().
 *
 * \warning ENA is asserted and /SOFF released before CddGtm_Start(); the
 *          bridge is un-gated for one call duration before the PWM carrier is
 *          live.  Confirm this is safe given the ATOM reset state.
 *
 * \return  0x1U once the sequence has ever succeeded; 0x0U otherwise.  A
 *          second call returns 0x1U without executing any step.
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
 *          operation.  Requests during RUN, or invalid enum values, are
 *          silently ignored.
 *
 *          Host sequence:
 *              CddApp_Init();
 *              CddApp_SetCtrlMode(CDDAPP_CTRL_CLOSEDLOOP);  // or OPENLOOP — final
 *              CddApp_SetSpeedRefRpm(1500.0F);
 *              CddApp_Start();
 *
 * \param[in]  Mode  CDDAPP_CTRL_OPENLOOP or CDDAPP_CTRL_DFC_CLOSEDLOOP.
 *
 * \warning The "stop, set, restart" cycle implied elsewhere is not currently
 *          supported: CddApp_Start() is one-shot and there is no CddApp_Stop().
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
 * \param[in]  SpeedRpm  Mechanical speed reference  [RPM]
 *
 * \warning No NaN / Inf validation at this interface; such values pass
 *          through unchanged.
 */
extern void CddApp_SetSpeedRefRpm(const real32_T SpeedRpm);

#endif /* CDD_APP_H */
