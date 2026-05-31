/**********************************************************************************************************************
 * \file        cdd_qspi_app.h
 * \brief       QSPI4 bare-metal driver — public interface.
 *
 * \details     Hardware: AP32541 Motor Control Power Board
 *                  P22.0  QSPI4_MTSR  MOSI   alt-func 1
 *                  P22.1  QSPI4_MRST  MISO   input, no pull
 *                  P22.2  QSPI4_SLSO3 CS     alt-func 2  (SLSO channel 3)
 *                  P22.3  QSPI4_SCLK  SCLK   alt-func 1
 *
 *              5 MHz, MODE 0, 24-bit MSB-first, ISR-driven (no DMA).
 *              No iLLD driver layer — direct register access via ifxQspi_reg.h.
 *
 *              Exchange model
 *              ──────────────
 *              CddQspi4_Exchange() is blocking from the caller's perspective but
 *              ISR-driven underneath.  For each 24-bit frame it writes BACON then
 *              DATA into the hardware FIFO and spin-waits on a flag set by
 *              qspi4_rx_handler (SRPN 56).  qspi4_err_handler (SRPN 57) also sets
 *              the flag so no error condition can hang the foreground.
 *
 *              Return value convention (all public functions):
 *                  0x1U  success
 *                  0x0U  failure — detail written to the *ErrorCode out-parameter.
 *
 * \note        MISRA C:2012: Rules 8.5, 8.6, 15.5.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_QSPI_APP_H_
#define CDD_QSPI_APP_H_

#include "cdd_config.h"
#include "cdd_stm_app.h"

/**********************************************************************************************************************
 * Baud Rate Constants
 *********************************************************************************************************************/

/** Required QSPI4 SCK frequency [Hz].  Verified via CddQspi4_GetBaudRate()
 *  before any SPI or GPIO activity in CddTle9180_Init().
 *  Actual fPeriph is read at runtime via CddSys_GetQspiFreq().               */
#define CDD_QSPI4_BAUD_RATE_HZ       (MHZ_1)  /**< Required QSPI4 SCK frequency [Hz]  */

/** Comparison epsilon for CddSys_AreEqual32() baud-rate check [Hz].
 *  The baud rate is derived from integer register fields — any deviation from
 *  exactly 5 000 000 Hz indicates a real misconfiguration.  1.0F satisfies the
 *  float comparison contract without masking genuine errors.                      */
#define CDD_QSPI4_BAUD_RATE_EPSILON  (1.0F)

/**********************************************************************************************************************
 * Error Codes  (written to the *ErrorCode out-parameter; return value is always 0x1U / 0x0U)
 *********************************************************************************************************************/

#define CDD_QSPI_ERR_NONE            (0x0U)   /**< No error                                    [dimensionless] */
#define CDD_QSPI_ERR_RX_TIMEOUT      (0x1U)   /**< RX spin-wait timeout — frame lost           [dimensionless] */
#define CDD_QSPI_ERR_ISR_ERROR       (0x2U)   /**< ERR ISR fired — QSPI4 protocol error        [dimensionless] */
#define CDD_QSPI_ERR_BAUD_MISMATCH   (0x3U)

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   One-time QSPI4 hardware initialisation (master, 5 MHz, MODE 0, 24-bit, ISR-driven).
 *
 * \details ENDINIT-protected CLC access handled internally via cdd_sys_utility.h.
 *          Configures and enables SRC_QSPI4TX (SRPN 55), SRC_QSPI4RX (SRPN 56),
 *          and SRC_QSPI4ERR (SRPN 57) on CPU0.
 *          Call once before the first CddQspi4_Exchange().
 *
 * \param[out] ErrorCode  Set to CDD_QSPI_ERR_NONE on success.  [dimensionless]
 * \return  0x1U success, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddQspi4_Init(
    P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode);

/**
 * \brief   Read back the actual QSPI4 SCK frequency from ECON3 and GLOBALCON.TQ.
 *
 * \details Mirrors get_qspi0_ch12_baud_rate_frequency() exactly (USM2 p1702):
 *              f = CddSys_GetQspiFreq()
 *              f = f / (TQ + 1)
 *              f = f / (Q  + 1)
 *              f = f / ((A + 1) + B + C)
 *          All arithmetic in real32_T.
 *          Used by CddTle9180_Init() with CddSys_AreEqual32() to verify 5 MHz
 *          is achieved before any GPIO or SPI activity.
 *
 * \return  Actual SCK frequency [Hz] as real32_T.
 */
extern real32_T CddQspi4_GetBaudRate(void);

/**
 * \brief   Blocking SPI exchange: transmit and receive one 24-bit frame, ISR-driven.
 *
 * \details Mirrors exchange_qspi0() from qspi_utility.c.
 *          Writes BACONENTRY twice (LAST=0 then LAST=1) then DATAENTRY0.
 *          Spin-waits on qspi4_rx_handler setting the received flag.
 *          qspi4_err_handler also sets received so no error can hang the foreground.
 *
 * \param[in]  TxWord     Pointer to 24-bit transmit word (bits [23:0]).  [—]
 * \param[out] RxWord     Pointer to receive word.                        [—]
 * \param[out] ErrorCode  CDD_QSPI_ERR_RX_TIMEOUT  — spin-wait exceeded limit.
 *                        CDD_QSPI_ERR_ISR_ERROR   — ERR ISR fired.
 *                        CDD_QSPI_ERR_NONE        — success.            [dimensionless]
 * \return  0x1U success, 0x0U failure.  [dimensionless]
 */
extern uint32_T CddQspi4_Exchange(
    P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA) TxWord,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) RxWord,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode);

#endif /* CDD_QSPI_APP_H_ */
