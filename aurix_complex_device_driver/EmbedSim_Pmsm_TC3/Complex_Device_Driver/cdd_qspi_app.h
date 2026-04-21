/**********************************************************************************************************************
 * \file        cdd_qspi_app.h
 * \brief       QSPI4 SPI exchange interface for the TLE9180D gate driver.
 *
 * \details     Bare-metal flat-register driver (no iLLD SpiMaster layer).
 *              Provides a single blocking 24-bit exchange function used by
 *              cdd_gate_driver_9180.c.
 *
 *              Hardware  (AP32541 Table 16):
 *                  SCLK P22.3, MOSI P22.0, MISO P22.1, CS P22.2
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per function
 *              - Rule  8.6 : Definitions in cdd_qspi_app.c
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_QSPI_APP_H_
#define CDD_QSPI_APP_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"
#include "IfxQspi_reg.h"

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Transmits one 24-bit SPI frame to the TLE9180D and receives the response.
 *
 * \details Blocking.  Writes BACON + DATAENTRY0, waits for RX interrupt flag,
 *          reads RXEXIT.  Uses QSPI4 channel 4 (ECON4, CS = P22.2 SLSO4.2).
 *
 * \param   Tx_Frame   24-bit MOSI word
 * \param   Rx_Frame   Pointer to receive the 24-bit MISO response
 * \return  1 if transfer completed without error, 0 on SPI error
 */
extern uint32_T QSPI_TLE9180_Exchange(uint32_T Tx_Frame, uint32_T * const Rx_Frame);

#endif /* CDD_QSPI_APP_H_ */
