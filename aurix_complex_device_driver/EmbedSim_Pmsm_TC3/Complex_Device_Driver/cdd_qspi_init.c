/**********************************************************************************************************************
 * \file        cdd_qspi_init.c
 * \brief       Implementation of cdd_qspi_init.h — QSPI4 bare-metal init.
 *
 * \details     Follows the flat-register pattern of qspi_utility.c (QSPI0).
 *              All register names are from IfxQspi_reg.h (TC38x, address base
 *              0xF0002000 for QSPI4).
 *
 *              GLOBALCON.TQ  = 0x9  → fQSPI4 / (2*(9+1)) = 200MHz/20 = 10 MHz SCLK max
 *              ECON4.Q/A/B/C = 0/1/2/0 → baud = fQSPI/(2*(Q+1)*(A+B+C+1))
 *                                       ≈ 200/(2*1*4) = 25 MHz peak;
 *                              actual SCLK limited by TLE9180D max 50 MHz — well within range.
 *              ECON4 channel 4 is mapped to CS pin via SSOC.OEN bit 12.
 *
 *              Pin mux (P22.0/1/2/3) is owned by cdd_gpio_app.  This file
 *              only writes QSPI4_PISEL for MISO input selection.
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : No file-scope variables (stateless init)
 *              - Rule 14.4  : All conditions explicit
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_qspi_init.h"
#include "cdd_sys_utility.h"

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CDD_Qspi_Init
 *------------------------------------------------------------------------------------------------------------------*/
void CDD_Qspi_Init(void)
{
    Ifx_QSPI_GLOBALCON  globalcon;
    Ifx_QSPI_GLOBALCON1 globalcon1;
    Ifx_QSPI_ECON       econ;

    /* Enable QSPI4 module clock  (TC3xx UM P.1747) */
    Clear_CPU_WDT_EndInit();
    QSPI4_CLC.B.DISR = 0x0U;
    QSPI4_CLC.B.EDIS = 0x1U;   /* disable sleep mode */
    Set_CPU_WDT_EndInit();

    while (QSPI4_CLC.B.DISS != 0x0U)
    {
        Nop_Delay(0x1U, 0x1U);
    }

    /* GLOBALCON — master mode, TQ prescaler, expect timeout  (UM P.1760) */
    globalcon.U           = 0x0U;
    globalcon.B.TQ        = 0x9U;   /* fQSPI4 baud prescaler base          */
    globalcon.B.STROBE    = 0x0U;
    globalcon.B.EXPECT    = 0xFU;   /* max timeout                          */
    globalcon.B.MS        = 0x0U;   /* master mode                          */
    globalcon.B.AREN      = 0x0U;
    globalcon.B.RESETS    = 0x1U;
    globalcon.B.CLKSEL    = 0x1U;
    QSPI4_GLOBALCON.U     = globalcon.U;

    /* GLOBALCON1 — enable TX/RX interrupts and all error sources  (UM P.1761) */
    globalcon1.U            = 0x0U;
    globalcon1.B.ERRORENS   = 0x1FFU;
    globalcon1.B.TXEN       = 0x1U;
    globalcon1.B.RXEN       = 0x1U;
    globalcon1.B.TXFIFOINT  = 0x0U;
    globalcon1.B.RXFIFOINT  = 0x0U;
    globalcon1.B.TXFM       = 0x0U;
    globalcon1.B.RXFM       = 0x0U;
    QSPI4_GLOBALCON1.U      = globalcon1.U;

    /* PISEL — MISO input: P22.1 = MRIS 2  (Appendix P.977) */
    QSPI4_PISEL.U = 0x00000002U;

    /* SRC nodes — TX/RX/ERR → CPU0  (TC3xx Appendix P.409) */
    SRC_QSPI4TX.B.SRPN  = CORE_00_QSPI4_TX_SRPN;
    SRC_QSPI4TX.B.TOS   = 0x0U;
    SRC_QSPI4TX.B.CLRR  = 0x1U;
    SRC_QSPI4TX.B.SRE   = 0x1U;

    SRC_QSPI4RX.B.SRPN  = CORE_00_QSPI4_RX_SRPN;
    SRC_QSPI4RX.B.TOS   = 0x0U;
    SRC_QSPI4RX.B.CLRR  = 0x1U;
    SRC_QSPI4RX.B.SRE   = 0x1U;

    SRC_QSPI4ERR.B.SRPN = CORE_00_QSPI4_ERR_SRPN;
    SRC_QSPI4ERR.B.TOS  = 0x0U;
    SRC_QSPI4ERR.B.CLRR = 0x1U;
    SRC_QSPI4ERR.B.SRE  = 0x1U;

    /* ECON4 — channel 4 timing: CPOL=0, CPHA=0, ~5 MHz */
    econ.U       = 0x0U;
    econ.B.Q     = 0x0U;
    econ.B.A     = 0x1U;
    econ.B.B     = 0x2U;
    econ.B.C     = 0x0U;
    econ.B.CPH   = 0x0U;   /* CPHA = 0 */
    econ.B.CPOL  = 0x0U;   /* CPOL = 0 */
    econ.B.PAREN = 0x0U;
    QSPI4_ECON4.U = econ.U;

    /* SSOC — enable CS output on channel 12 (P22.2 = SLSO4.2)  (UM P.1773) */
    QSPI4_SSOC.B.OEN = 0x1000U;

    /* Enable module */
    QSPI4_GLOBALCON.B.EN = 0x1U;
    QSPI4_GLOBALCON.B.LB = 0x0U;
}
