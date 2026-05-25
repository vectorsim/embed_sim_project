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
 *              5 MHz, MODE 0, 24-bit MSB-first, polling (no DMA, no IRQ).
 *              No iLLD driver layer — direct register access via ifxQspi_reg.h.
 *
 * \note        MISRA C:2012: Rules 8.5, 8.6, 15.5.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_QSPI_APP_H_
#define CDD_QSPI_APP_H_

#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Return Codes
 *********************************************************************************************************************/

#define CDD_QSPI_OK             (0x0U)   /**< Frame exchange succeeded        [dimensionless] */
#define CDD_QSPI_ERR_TIMEOUT    (0x1U)   /**< RX FIFO timeout — frame lost    [dimensionless] */

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   One-time QSPI4 hardware initialisation (master, 5 MHz, MODE 0, 24-bit).
 *
 * \details ENDINIT-protected CLC access handled internally via cdd_sys_utility.h.
 *          Call once before the first CddQspi4_Exchange().
 *
 * \return  void
 */
extern void CddQspi4_Init(void);

/**
 * \brief   Blocking SPI exchange: transmit and receive \p Count × 24-bit frames.
 *
 * \details Frames processed one at a time.  BACON.LAST set on the final frame
 *          to deassert CS after the transfer.  Both buffers must be at least
 *          Count words long.
 *
 * \param[in]  TxBuf   Pointer to transmit words (24-bit payload in bits [23:0]).
 * \param[out] RxBuf   Pointer to receive word buffer.
 * \param[in]  Count   Number of frames to exchange (>= 1)  [dimensionless]
 *
 * \return  CDD_QSPI_OK (0x0U) on success, CDD_QSPI_ERR_TIMEOUT (0x1U) on failure.
 */
extern uint32_T CddQspi4_Exchange(
    P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA) TxBuf,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) RxBuf,
    uint32_T Count);

#endif /* CDD_QSPI_APP_H_ */
