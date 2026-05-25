/**********************************************************************************************************************
 * \file        cdd_tle9180_app.h
 * \brief       TLE9180D-31QK gate driver CDD — public interface.
 *
 * \details     Hardware: AP32541 Motor Control Power Board
 *              GPIO:  /INH P20.0 BU7.10 | ENA P33.11 BU6.26
 *                     /SOFF P33.10 BU6.38 | /ERR P15.2 BU6.31
 *              SPI:   QSPI4 — MOSI P22.0 | MISO P22.1 | CS P22.2 | SCLK P22.3
 *
 * \note        MISRA C:2012: Rules 8.5, 8.6, 14.4, 15.5.
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
 * State Machine Type
 *********************************************************************************************************************/

/**
 * \brief   TLE9180 operating state.
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
 */
typedef struct
{
    CddTle9180_SpiRx_T  RxBuf[CDD_TLE9180_SPI_BUF_SIZE]; /**< Receive buffer              */
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
 * Utility — Public Function Prototype
 *********************************************************************************************************************/

/**
 * \brief   Compute CRC-3 for a 24-bit SPI transmit frame.
 *
 * \details Polynomial x³+x+1, initial value 0x7.  Operates on bits [23:3] (21 bits), MSB first.
 *
 * \param[in]  Frame24Bit  Complete transmit frame with CRC field = 0x0.
 * \return     3-bit CRC to be placed in bits [2:0].  [dimensionless]
 */
extern uint32_T CddTle9180_ComputeCrc3(uint32_T Frame24Bit);

/**********************************************************************************************************************
 * State Machine — Public Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialise QSPI4 hardware and execute the power-on GPIO sequence.
 *
 * \details Calls CddQspi4_Init() then the internal power-on sequence.
 *          On return: Handle->State == CDDTLE9180_STATE_IDLE.
 *
 * \param[out] Handle  Pointer to uninitialised handle.
 * \return  void
 */
extern void     CddTle9180_Init(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   Send the 13-frame startup SPI write sequence (blocking).
 *
 * \param[in,out] Handle  Pointer to initialised handle.
 * \return  0x1U success, 0x0U SPI error.  [dimensionless]
 */
extern uint32_T CddTle9180_Configure(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   2-frame pipeline read of STATUS — confirm NORMAL mode.
 *
 * \param[in,out] Handle  Pointer to initialised handle.
 * \return  0x1U norm_m=1 (NORMAL mode), 0x0U not in NORMAL mode.  [dimensionless]
 */
extern uint32_T CddTle9180_IsNormalMode(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   Convenience wrapper: Init → Configure → IsNormalMode.
 *
 * \param[in,out] Handle  Pointer to uninitialised handle.
 * \return  0x1U reached NORMAL mode, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddTle9180_Startup(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   Cyclic register readback — call once per slow task (10 ms).
 *
 * \param[in,out] Handle  Pointer to initialised handle.
 * \return  0x1U frame OK, 0x0U SPI error.  [dimensionless]
 */
extern uint32_T CddTle9180_ReadRegister(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   Fault monitor — call from 1 ms task.
 *
 * \details If State==NORMAL and /ERR=LOW: asserts /SOFF, sets State=FAULT.
 *
 * \param[in,out] Handle  Pointer to initialised handle.
 * \return  void
 */
extern void     CddTle9180_MonitorFault(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

/**
 * \brief   Clear fault and re-run full Startup sequence.
 *
 * \param[in,out] Handle  Pointer to handle in fault state.
 * \return  0x1U returned to NORMAL, 0x0U re-init failed.  [dimensionless]
 */
extern uint32_T CddTle9180_ResetFault(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle);

#endif /* CDD_TLE9180_APP_H_ */
