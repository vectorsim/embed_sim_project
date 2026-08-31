/**********************************************************************************************************************
 * \file        cdd_tle9180_app.h
 * \brief       TLE9180D-31QK gate driver CDD — public interface.
 *
 * \details     Hardware: AP32541 Motor Control Power Board
 *              GPIO:  /INH P20.0 BU7.10 | ENA P33.11 BU6.26
 *                     /SOFF P33.10 BU6.38 | /ERR P15.2 BU6.31
 *              SPI:   QSPI4 — MOSI P22.0 | MISO P22.1 | CS P22.2 | SCLK P22.3
 *
 *              Design notes:
 *              - All SPI frame CRC-3 values are computed at runtime by
 *                CddTle9180_ComputeCrc3() during table initialisation; no
 *                hardcoded CRC literals appear in the configuration tables.
 *              - All blocking waits use CddStm_Wait() (STM deadline-based)
 *                rather than raw NOP loops, giving calibrated timing independent
 *                of CPU clock frequency.
 *              - CddTle9180_Init() verifies the QSPI4 baud rate after
 *                CddQspi4_Init() and before any GPIO or SPI activity, using
 *                CddQspi4_GetBaudRate() and CddSys_AreEqual32().  This mirrors
 *                the old is_not_equal_epsilon(get_qspi0_ch12_baud_rate_frequency(),
 *                MHZ_5) guard from put_tle9180_in_normal_state().
 *                On baud rate mismatch: ErrorCode = CDD_TLE9180_ERR_BAUD_RATE,
 *                Handle->State = CDDTLE9180_STATE_FAULT, return 0x0U.
 *
 *              Return value convention (all public functions except void):
 *                  0x1U  success
 *                  0x0U  failure — detail written to the *ErrorCode out-parameter.
 *
 * \note        MISRA C:2012: Rules 8.5, 8.6, 14.4, 15.5.
 *              CddTle9180_PingSr0() is a STATIC (TU-internal) diagnostic function
 *              called from CddTle9180_Init() immediately after the baud rate check.
 *              Its six function-local static volatile debug variables are visible
 *              in the JTAG watch window under cdd_tle9180_app.c scope:
 *                  dbg_Sr0_QspiErr    CDD_QSPI_ERR_NONE(0)=OK, else RX timeout
 *                  dbg_Sr0_RawFrame   full raw 24-bit RxBuf[1].U
 *                  dbg_Sr0_Data       SR0 DATA byte [11:4]
 *                  dbg_Sr0_ConfValid  CONFVALID [21] — 0 expected before config
 *                  dbg_Sr0_SpiErr     SPIERR [20]    — 0 = frame accepted
 *                  dbg_Sr0_Error      ERROR  [23]    — mirrors /ERR pin
 *                  dbg_Sr0_Warning    WARNING [22]   — OR of all warning-class faults
 *
 *              CddTle9180_ClearFaults() sweeps error registers 0x41–0x4D (excl. 0x4A)
 *              in a single 13-frame burst.  Its 13 function-local static volatile debug
 *              variables are visible in the JTAG watch window:
 *                  dbg_FaultClr_QspiErr   QSPI exchange error code
 *                  dbg_FaultClr_ErrOver   0x41 Err_over  — Error Overview DATA byte
 *                  dbg_FaultClr_Ser       0x42 Ser       — Special Event Register
 *                  dbg_FaultClr_ErrI1     0x43 Err_i_1   — Internal Errors 1
 *                  dbg_FaultClr_ErrI2     0x44 Err_i_2   — Internal Errors 2
 *                  dbg_FaultClr_ErrE      0x45 Err_e     — External Errors
 *                  dbg_FaultClr_ErrSd     0x46 Err_sd    — Shutdown Errors
 *                  dbg_FaultClr_ErrScd    0x47 Err_scd   — Short Circuit Errors
 *                  dbg_FaultClr_ErrIndiag 0x48 Err_indiag— Input Pattern Violations
 *                  dbg_FaultClr_ErrOsf    0x49 Err_osf   — Output Stage Feedback Errors
 *                  dbg_FaultClr_ErrOp12   0x4B Err_op_12 — CSA 1 & 2 Errors
 *                  dbg_FaultClr_ErrOp3    0x4C Err_op_3  — CSA 3 Errors
 *                  dbg_FaultClr_ErrOutp   0x4D Err_outp  — Digital Output Pin Errors
 *              Note: 0x4A (Err_spiconf) is NOT swept — not cleared by read (DS §14.3).
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_TLE9180_APP_H_
#define CDD_TLE9180_APP_H_

#include "cdd_config.h"       /* embed_sim_sys_types.h + embed_sim_compiler.h */
#include "cdd_tle9180_reg.h"  /* CddTle9180_SpiTx_T, CddTle9180_SpiRx_T     */
#include "cdd_stm_app.h"      /* TimeConst_xxx, CddStm_GetDeadline, ...       */

/**********************************************************************************************************************
 * Error Codes  (written to the *ErrorCode out-parameter; return value is always 0x1U / 0x0U)
 *********************************************************************************************************************/

#define CDD_TLE9180_ERR_NONE        (0x0U)  /**< No error                              [dimensionless] */
#define CDD_TLE9180_ERR_SPI         (0x1U)  /**< QSPI exchange failed (RX timeout)     [dimensionless] */
#define CDD_TLE9180_ERR_NOT_NORMAL  (0x2U)  /**< norm_m=0 or CONFVALID=0 after config  [dimensionless] */
#define CDD_TLE9180_ERR_BAUD_RATE   (0x3U)  /**< QSPI4 baud rate != 5 MHz after Init   [dimensionless] */

/**********************************************************************************************************************
 * State Machine Type
 *********************************************************************************************************************/

/**
 * \brief   TLE9180 operating state.
 *
 * \details State transitions:
 *
 *   [POWERON] ──Init()──► [SLEEP] ──1 s──► [IDLE] ──Configure()──► [CONFIGURING]
 *       ▲                                                                │
 *       │                                                        IsNormalMode()
 *       │                                                                │
 *   ResetFault()                                                         ▼
 *       │                                                           [NORMAL]
 *   [FAULT] ◄────────── /ERR asserted (MonitorFault, 1 ms task) ─────────┘
 *
 *   Init() also transitions to [FAULT] immediately if the QSPI4 baud rate
 *   check fails (CDD_TLE9180_ERR_BAUD_RATE), before any GPIO activity.
 */
typedef enum
{
    CDDTLE9180_STATE_POWERON     = 0x0U,   /**< Power-on sequence in progress  [dimensionless] */
    CDDTLE9180_STATE_SLEEP       = 0x1U,   /**< Device in sleep (INH=LOW)      [dimensionless] */
    CDDTLE9180_STATE_IDLE        = 0x2U,   /**< Powered, not configured        [dimensionless] */
    CDDTLE9180_STATE_CONFIGURING = 0x3U,   /**< SPI startup sequence in flight [dimensionless] */
    CDDTLE9180_STATE_NORMAL      = 0x4U,   /**< Normal operating mode          [dimensionless] */
    CDDTLE9180_STATE_FAULT       = 0x5U    /**< Fault latched, /SOFF asserted  [dimensionless] */
} CddTle9180_State_T;

/**********************************************************************************************************************
 * Driver Handle
 *********************************************************************************************************************/

/**
 * \brief   TLE9180 runtime handle.
 *
 * \details Owned by the application layer; must not be accessed directly — use
 *          the public API functions only.  Lifetime must exceed the last API call.
 */
typedef struct
{
    CddTle9180_SpiRx_T  RxBuf[CDD_TLE9180_SPI_BUF_SIZE];   /**< Receive buffer              */
    uint32_T            ReadIndex;                         /**< Cyclic read index [0..12]   */
    CddTle9180_State_T  State;                             /**< Current operating state     */
    uint32_T            FaultCode;                         /**< Latched fault bits, 0=clear */
} CddTle9180_T;

/**********************************************************************************************************************
 * GPIO Control — Public Function Prototypes
 *********************************************************************************************************************/

/** \brief  /INH = LOW  → TLE9180 power-down → SLEEP state.  (P20.0) */
extern void     CddTle9180_AssertInhibit(void);

/** \brief  /INH = HIGH → releases inhibit → IDLE state.             */
extern void     CddTle9180_DeassertInhibit(void);

/** \brief  ENA = HIGH → enables TLE9180 internal pre-drivers.       */
extern void     CddTle9180_AssertEnable(void);

/** \brief  ENA = LOW  → disables pre-drivers.                        */
extern void     CddTle9180_DeassertEnable(void);

/** \brief  /SOFF = LOW → immediate hardware gate shutdown. Latched — must be
 *          explicitly deasserted after fault cleared.                 */
extern void     CddTle9180_AssertSafeOff(void);

/** \brief  /SOFF = HIGH → re-enables gate outputs.                   */
extern void     CddTle9180_DeassertSafeOff(void);

/**
 * \brief   Read /ERR pin (P15.2, open-drain, pulled up on AP32541).
 * \return  0x1U error active (/ERR=LOW), 0x0U no error (/ERR=HIGH).  [dimensionless]
 */
extern uint32_T CddTle9180_IsErrorActive(void);

/**********************************************************************************************************************
 * Utility — Public Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Compute CRC-3 for a 24-bit SPI transmit frame.
 *
 * \details Polynomial x³+x+1 (0x3), initial value 0x4.  Operates on bits [23:3]
 *          (21 data bits) MSB-first; bits [2:0] of the frame must be 0x0 on entry.
 *
 * \param[in]  Frame24Bit  Complete transmit frame with CRC field zeroed ([2:0] = 0).
 * \return     3-bit CRC to be placed in bits [2:0].  [dimensionless]
 */
extern uint32_T CddTle9180_ComputeCrc3(uint32_T Frame24Bit);

/**
 * \brief   Blocking wait using the STM hardware timer.
 *
 * \param[in]  TimeConst   Time constant from cdd_stm_app.h  (e.g. TimeConst_1ms).
 * \return  void
 */
extern void CddTle9180_Wait(uint32_T TimeConst);

/**********************************************************************************************************************
 * State Machine — Public Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialise QSPI4 hardware, verify baud rate, execute power-on GPIO sequence.
 *
 * \details Call sequence:
 *            1. CddQspi4_Init()
 *            2. CddSys_AreEqual32(CddQspi4_GetBaudRate(), CDD_QSPI4_BAUD_RATE_HZ,
 *                                  CDD_QSPI4_BAUD_RATE_EPSILON)
 *               → mismatch: ErrorCode=CDD_TLE9180_ERR_BAUD_RATE, State=FAULT, return 0x0U
 *            3. CddTle9180_PowerOnSequence() (INH/ENA/SOFF per DS §6.4)
 *            4. CddTle9180_PingSr0()         (SPI physical-layer verification)
 *
 *          On success: Handle->State == CDDTLE9180_STATE_IDLE.
 *          On failure: Handle->State == CDDTLE9180_STATE_FAULT.
 *
 * \param[out] Handle     Pointer to uninitialised handle.
 * \param[out] ErrorCode  CDD_TLE9180_ERR_BAUD_RATE — QSPI4 baud rate != 5 MHz.
 *                        CDD_TLE9180_ERR_SPI       — QSPI init or ping failure.
 *                        CDD_TLE9180_ERR_NONE      — success.  [dimensionless]
 * \return  0x1U success, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddTle9180_Init(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   Send the 13-frame startup SPI write sequence (blocking).
 *
 * \param[in,out] Handle     Pointer to initialised handle.
 * \param[out]    ErrorCode  CDD_TLE9180_ERR_SPI on exchange failure,
 *                           CDD_TLE9180_ERR_NONE on success.  [dimensionless]
 * \return  0x1U success, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddTle9180_Configure(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   2-frame pipeline read of STATUS — confirm NORMAL mode.
 *
 * \param[in,out] Handle     Pointer to initialised handle.
 * \param[out]    ErrorCode  CDD_TLE9180_ERR_SPI on exchange failure,
 *                           CDD_TLE9180_ERR_NOT_NORMAL if norm_m=0 or CONFVALID=0,
 *                           CDD_TLE9180_ERR_NONE on success.  [dimensionless]
 * \return  0x1U norm_m=1 and CONFVALID=1 (NORMAL mode), 0x0U otherwise.  [dimensionless]
 */
extern uint32_T CddTle9180_IsNormalMode(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   Convenience wrapper: Init → Configure → IsNormalMode.
 *
 * \param[in,out] Handle     Pointer to uninitialised handle.
 * \param[out]    ErrorCode  First non-zero error code from the sequence,
 *                           CDD_TLE9180_ERR_NONE on success.  [dimensionless]
 * \return  0x1U reached NORMAL mode, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddTle9180_Startup(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   Cyclic register readback — call once per slow task (10 ms).
 *
 * \param[in,out] Handle     Pointer to initialised handle.
 * \param[out]    ErrorCode  CDD_TLE9180_ERR_SPI on exchange failure,
 *                           CDD_TLE9180_ERR_NONE on success.  [dimensionless]
 * \return  0x1U frame OK, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddTle9180_ReadRegister(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   Fault monitor — call from 1 ms task.
 *
 * \details If State==NORMAL and /ERR=LOW: asserts /SOFF immediately (hardware
 *          gate shutdown), then latches State=FAULT and FaultCode=0x1U.
 *
 * \param[in,out] Handle  Pointer to initialised handle.
 * \return  void
 */
extern void CddTle9180_MonitorFault(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   Clear fault and re-run full Startup sequence.
 *
 * \param[in,out] Handle     Pointer to handle in fault state.
 * \param[out]    ErrorCode  First non-zero error from the restart sequence,
 *                           CDD_TLE9180_ERR_NONE on success.  [dimensionless]
 * \return  0x1U returned to NORMAL, 0x0U re-init failed.  [dimensionless]
 */
extern uint32_T CddTle9180_ResetFault(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   Read and clear all TLE9180 error/warning status flags.
 *
 * \param[out] ErrorCode  CDD_TLE9180_ERR_SPI on QSPI exchange failure,
 *                        CDD_TLE9180_ERR_NONE on success.  [dimensionless]
 * \return  0x1U success, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddTle9180_ClearFaults(
    P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode);

#endif /* CDD_TLE9180_APP_H_ */
