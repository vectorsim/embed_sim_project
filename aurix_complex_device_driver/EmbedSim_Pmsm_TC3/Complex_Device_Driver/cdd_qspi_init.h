/**********************************************************************************************************************
 * \file        cdd_qspi_init.h
 * \brief       QSPI4 SPI master module initialisation for the AP32541 board.
 *
 * \details     Bare-metal flat-register implementation (no iLLD SpiMaster layer).
 *              Follows the same pattern as qspi_utility.c (QSPI0).
 *
 *              Hardware  (AP32541 Table 16):
 *                  Module  : QSPI4
 *                  SCLK    : P22.3
 *                  MOSI    : P22.0
 *                  MISO    : P22.1   PISEL = 0x02 (MRIS = 2)
 *                  CS/SLSO : P22.2   SSOC.OEN bit 12 → channel 12 (ECON4)
 *
 *              Frame:  24-bit, MSB first, CPOL=0 CPHA=0, ~5 MHz
 *
 *              Interrupt priorities  (cdd_config.h):
 *                  TX : CORE_00_QSPI4_TX_SRPN  (55)  CPU0
 *                  RX : CORE_00_QSPI4_RX_SRPN  (56)  CPU0
 *                  ERR: CORE_00_QSPI4_ERR_SRPN (57)  CPU0
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.5 : One declaration per object / function
 *              - Rule  8.6 : Definitions in cdd_qspi_init.c
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_QSPI_INIT_H_
#define CDD_QSPI_INIT_H_

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_config.h"
#include "IfxQspi_reg.h"
#include "IfxSrc_reg.h"

/**********************************************************************************************************************
 * Function Prototypes
 *********************************************************************************************************************/

/**
 * \brief   Initialises QSPI4 in SPI master mode for the TLE9180D.
 *
 * \details Enables module clock, configures GLOBALCON / GLOBALCON1 / ECON4,
 *          sets up pin mux for P22.0/1/2/3, configures TX/RX/ERR SRC nodes,
 *          and arms the TX FIFO.  Must be called before CDD_Qspi_Exchange().
 *
 * \return  None
 */
extern void CDD_Qspi_Init(void);

#endif /* CDD_QSPI_INIT_H_ */
