/**********************************************************************************************************************
 * \file        cdd_qspi_app.c
 * \brief       QSPI4 bare-metal driver — hardware init and polling exchange.
 *
 * \details     Direct register access via ifxQspi_reg.h / ifxSrc_reg.h.
 *              No iLLD driver layer, no DMA, no interrupts.
 *
 * \note        MISRA C:2012: Rules 8.9, 10.1, 14.4, 15.5.
 *              PRQA S 0303 — hardware-register union access via .U / .B members.
 *              PRQA S 0750 — Ifx_QSPI_BACON union .U and .B accessed within same function.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#include "cdd_qspi_app.h"
#include "cdd_sys_utility.h"   /* CddSys_ClearWdtEndInit / CddSys_SetWdtEndInit */
#include "ifxQspi_reg.h"
#include "ifxSrc_reg.h"

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  BACON.CS — selects SLSO3 on P22.2.                                */
#define QSPI4_CS_CHANNEL        (3U)

/** \brief  SSOC whole-register value: OEN bit[3]=1 enables SLSO3.            */
#define QSPI4_SSOC_VAL          (0x00000008UL)

/** \brief  STATUS.RXFIFOLEVEL mask — bits [14:12] per TC38x RM §31.7.1.      */
#define QSPI4_STATUS_RXLEVEL_MASK   (0x00007000UL)

/** \brief  BACON.DL — frame length minus 1 (24 bits → 23 = 0x17).            */
#define QSPI4_FRAME_LEN_M1      (0x17U)

/** \brief  PISEL value for MISO on P22.1.                                     */
#define QSPI4_PISEL_MRST        (0x00000002U)

/** \brief  GLOBALCON.TQ prescaler: fSPB=100 MHz, TQ=9 → fTQ=10 MHz.         */
#define QSPI4_GLOBALCON_TQ      (0x9U)

/** \brief  ECON3: Q=0, A=1, B=2, C=0, CPH=0, CPOL=0 → 5 MHz MODE 0.        */
#define QSPI4_ECON3_TIMING      (0x00000101U)

/** \brief  RX polling loop limit.                                             */
#define QSPI4_RX_POLL_TIMEOUT   (1000U)

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddQspi4_Init
 *------------------------------------------------------------------------------------------------------------------*/
void CddQspi4_Init(void)
{
    /* Enable QSPI4 clock (ENDINIT-protected) */
    CddSys_ClearWdtEndInit();
    QSPI4_CLC.B.DISR = 0x0U;
    QSPI4_CLC.B.EDIS = 0x1U;
    while (QSPI4_CLC.B.DISS != 0x0U) { ; }
    CddSys_SetWdtEndInit();

    /* GLOBALCON: master mode, TQ=9 */
    QSPI4_GLOBALCON.B.TQ     = QSPI4_GLOBALCON_TQ;
    QSPI4_GLOBALCON.B.EXPECT = 0xFU;
    QSPI4_GLOBALCON.B.MS     = 0x0U;
    QSPI4_GLOBALCON.B.AREN   = 0x0U;
    QSPI4_GLOBALCON.B.RESETS = 0x1U;
    QSPI4_GLOBALCON.B.CLKSEL = 0x1U;

    /* GLOBALCON1: TX/RX enabled, no FIFO interrupt, no DMA */
    QSPI4_GLOBALCON1.B.TXEN      = 0x1U;
    QSPI4_GLOBALCON1.B.RXEN      = 0x1U;
    QSPI4_GLOBALCON1.B.TXFIFOINT = 0x0U;
    QSPI4_GLOBALCON1.B.RXFIFOINT = 0x0U;
    QSPI4_GLOBALCON1.B.TXFM      = 0x0U;
    QSPI4_GLOBALCON1.B.RXFM      = 0x0U;

    /* MISO input: P22.1 = QSPI4 MRST */
    QSPI4_PISEL.U = QSPI4_PISEL_MRST;  /* PRQA S 0303 */

    /* ECON3: 5 MHz, MODE 0, no parity */
    QSPI4_ECON3.U       = QSPI4_ECON3_TIMING;  /* PRQA S 0303 */
    QSPI4_ECON3.B.PAREN = 0x0U;

    /* SSOC: SLSO3 active-LOW, no auto CS */
    QSPI4_SSOC.U = QSPI4_SSOC_VAL;  /* PRQA S 0303 */

    /* Disable all QSPI4 interrupt service requests (polling only) */
    SRC_QSPI4TX.B.SRE  = 0x0U;
    SRC_QSPI4RX.B.SRE  = 0x0U;
    SRC_QSPI4ERR.B.SRE = 0x0U;

    /* Enable module, disable loopback */
    QSPI4_GLOBALCON.B.EN = 0x1U;
    QSPI4_GLOBALCON.B.LB = 0x0U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddQspi4_Exchange
 *
 * Transmit and receive Count × 24-bit frames, one at a time.
 * BACON is built once; LAST is toggled on the final frame to deassert CS.
 * Write order into FIFO: BACONENTRY first, then DATAENTRY0 (TC3xx RM §31).
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddQspi4_Exchange(
    P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA) TxBuf,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) RxBuf,
    uint32_T Count)
{
    uint32_T       i;
    uint32_T       timeout;
    uint32_T       result = CDD_QSPI_OK;
    Ifx_QSPI_BACON bacon; /* PRQA S 0750 */

    bacon.U        = 0x0U;
    bacon.B.MSB    = 0x1U;
    bacon.B.CS     = QSPI4_CS_CHANNEL;
    bacon.B.DL     = QSPI4_FRAME_LEN_M1;
    bacon.B.BYTE   = 0x0U;
    bacon.B.PARTYP = 0x0U;

    for (i = 0x0U; (i < Count) && (result == CDD_QSPI_OK); i++)
    {
        /* LAST=1 on final frame — deasserts CS after transfer */
        if (i == (Count - 0x1U))
        {
            bacon.B.LAST = 0x1U;
        }
        else
        {
            bacon.B.LAST = 0x0U;
        }

        /* Load BACON then DATA — mandatory write order (TC3xx RM §31) */
        QSPI4_BACONENTRY.U = bacon.U;      /* PRQA S 0303 */
        QSPI4_DATAENTRY0.U = TxBuf[i];     /* PRQA S 0303 */

        /* Poll RX FIFO */
        timeout = QSPI4_RX_POLL_TIMEOUT;
        while (((QSPI4_STATUS.U & QSPI4_STATUS_RXLEVEL_MASK) == 0x0UL) && (timeout > 0x0U)) /* PRQA S 0303 */
        {
            timeout--;
        }

        if (timeout == 0x0U)
        {
            result = CDD_QSPI_ERR_TIMEOUT;
        }
        else
        {
            RxBuf[i] = QSPI4_RXEXIT.U;    /* PRQA S 0303 */
        }

        QSPI4_FLAGSCLEAR.B.RXC = 0x1U;
    }

    return result;
}
