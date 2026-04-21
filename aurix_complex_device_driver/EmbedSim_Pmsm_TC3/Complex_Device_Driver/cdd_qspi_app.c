/**********************************************************************************************************************
 * \file        cdd_qspi_app.c
 * \brief       Implementation of cdd_qspi_app.h — QSPI4 24-bit blocking exchange
 *              for the TLE9180D gate driver on the AP32541 board.
 *
 * \details     Follows the flat-register pattern of qspi_utility.c (QSPI0).
 *
 *              Transfer flow:
 *              1. Write BACON (channel 4, 24 bits, LAST=1) to QSPI4_BACONENTRY
 *              2. Write 24-bit payload to QSPI4_DATAENTRY0
 *              3. Poll state flags set by TX/RX ISRs
 *              4. Read result from QSPI4_RXEXIT
 *              5. Reset state
 *
 *              ISR naming convention (IFX_INTERRUPT, TASKING):
 *                  QSPI4_Tx_Isr  — SRPN 55, CPU0
 *                  QSPI4_Rx_Isr  — SRPN 56, CPU0
 *                  QSPI4_Err_Isr — SRPN 57, CPU0
 *
 * \note        MISRA C:2012 compliance:
 *              - Rule  8.9  : File-scope variables limited to this TU
 *              - Rule 14.4  : All conditions use explicit comparison
 *              - Rule 17.2  : No recursion
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_qspi_app.h"
#include "cdd_qspi_init.h"
#include "cdd_sys_utility.h"
#include "Bsp.h"                   /* IFX_INTERRUPT                           */

/**********************************************************************************************************************
 * ISR Vector Registrations
 *********************************************************************************************************************/
IFX_INTERRUPT(QSPI4_Tx_Isr,  0, 55);   /* CORE_00_QSPI4_TX_SRPN  */
IFX_INTERRUPT(QSPI4_Rx_Isr,  0, 56);   /* CORE_00_QSPI4_RX_SRPN  */
IFX_INTERRUPT(QSPI4_Err_Isr, 0, 57);   /* CORE_00_QSPI4_ERR_SRPN */

/**********************************************************************************************************************
 * Private Types
 *********************************************************************************************************************/

typedef struct
{
    volatile uint32_T transition;
    volatile uint32_T transmitted;
    volatile uint32_T received;
    volatile uint32_T error;
} QSPI4_State_T;

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/** \brief  QSPI4 transfer state flags (written by ISRs, read by exchange fn) */
static QSPI4_State_T QSPI4_State_G;

/** \brief  Last received 24-bit MISO word                                    */
static volatile uint32_T QSPI4_Rx_Buffer_G;

/**********************************************************************************************************************
 * ISR Bodies
 *********************************************************************************************************************/

void QSPI4_Tx_Isr(void)
{
    QSPI4_FLAGSCLEAR.B.TXC  = 0x1U;
    QSPI4_State_G.transition   = 0x1U;
    QSPI4_State_G.transmitted  = 0x1U;
    QSPI4_State_G.received     = 0x0U;
    QSPI4_State_G.error        = 0x0U;
}

void QSPI4_Rx_Isr(void)
{
    QSPI4_FLAGSCLEAR.B.RXC  = 0x1U;
    QSPI4_Rx_Buffer_G           = QSPI4_RXEXIT.U;
    QSPI4_State_G.transition   = 0x1U;
    QSPI4_State_G.transmitted  = 0x1U;
    QSPI4_State_G.received     = 0x1U;
    QSPI4_State_G.error        = 0x0U;
}

void QSPI4_Err_Isr(void)
{
    QSPI4_FLAGSCLEAR.B.ERRORCLEARS = 0x1U;
    QSPI4_State_G.error = 0x1U;
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * QSPI_TLE9180_Exchange
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T QSPI_TLE9180_Exchange(uint32_T Tx_Frame, uint32_T * const Rx_Frame)
{
    Ifx_QSPI_BACON bacon;
    uint32_T       result;

    result = 0U;

    /* Only start a transfer when the channel is fully idle */
    if (    (QSPI4_State_G.received    == 0x0U) &&
            (QSPI4_State_G.transmitted == 0x0U) &&
            (QSPI4_State_G.transition  == 0x0U) &&
            (QSPI4_State_G.error       == 0x0U) )
    {
        /* Build BACON: channel 4, 24 bits (DL=23), MSB first, LAST=1        */
        bacon.U        = 0x0U;
        bacon.B.MSB    = 0x1U;   /* MSB first                                */
        bacon.B.CS     = 0x4U;   /* channel select 4 → ECON4, SLSO4.2       */
        bacon.B.DL     = 0x17U;  /* data length = 24 bits (DL+1)            */
        bacon.B.BYTE   = 0x0U;
        bacon.B.LAST   = 0x1U;   /* single-frame transfer                    */

        /* Arm transfer — disable interrupts around BACON+DATA write          */
        (void)Disable_CPU_Interrupt();
        QSPI4_State_G.transition = 0x1U;
        QSPI4_BACONENTRY.U       = bacon.U;
        Restore_CPU_Interrupt(0x1U);

        /* Push data — triggers SPI shift-out                                 */
        QSPI4_DATAENTRY0.U = Tx_Frame;

        /* Wait for RX complete (set by QSPI4_Rx_Isr)                        */
        while (QSPI4_State_G.received != 0x1U)
        {
            Nop_Delay(0x1U, 0x1U);
        }

        if (QSPI4_State_G.error == 0x0U)
        {
            *Rx_Frame = QSPI4_Rx_Buffer_G;
            result    = 1U;
        }

        /* Reset state for next transfer */
        QSPI4_State_G.transition   = 0x0U;
        QSPI4_State_G.transmitted  = 0x0U;
        QSPI4_State_G.received     = 0x0U;
        QSPI4_State_G.error        = 0x0U;
    }

    return result;
}
