/**********************************************************************************************************************
 * \file        cdd_tle9180_app.c
 * \brief       TLE9180D-31QK gate driver CDD — state machine, CRC-3,
 *              configuration tables, GPIO wrappers.
 *
 * \details     QSPI4 hardware is owned by cdd_qspi_app.c.
 *              This file calls CddQspi4_Init() and CddQspi4_Exchange().
 *              GPIO via cdd_gpio_app.h.  Timing via cdd_stm_app.h.
 *
 *              Private STATIC function:
 *                  CddTle9180_PowerOnSequence()   INH/ENA/SOFF GPIO sequence.
 *
 * \note        MISRA C:2012: Rules 8.9, 10.1, 14.4, 15.5, 17.2.
 *
 *              PRQA S 11.3 deviation MD_TLE9180_11.3_union_U  [Rule 11.3 Required]
 *                  CddTle9180_SpiTx_T and CddTle9180_SpiRx_T are unions whose
 *                  first (and only full-width) member is uint32_T U.  Casting a
 *                  pointer to the union to const uint32_T* / uint32_T* is safe:
 *                  alignment is guaranteed (both are 32-bit aligned on TC3xx),
 *                  and CddQspi4_Exchange reads/writes only the .U word.
 *                  Suppression applied at each of the six cast sites below.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_tle9180_app.h"
#include "cdd_qspi_app.h"       /* CddQspi4_Init, CddQspi4_Exchange          */
#include "cdd_gpio_app.h"       /* CddGpio_SetInh_P20_0, CddGpio_SetEna_P33_11   */
#include "cdd_stm_app.h"        /* TimeConst_xxx, CddStm_GetDeadline, ...    */
#include "cdd_sys_utility.h"    /* CddSys_NopDelay                           */

/**********************************************************************************************************************
 * Private Macros — STATUS register read pipeline
 *********************************************************************************************************************/

/** \brief  Frame 1: READ reg 0x40, CRC=6  */
#define STATUS_READ_CMD     (0x400006U)

/** \brief  Frame 2: NOP pipeline flush, CRC=3  */
#define STATUS_NOP_CMD      (0x320003U)

/** \brief  Bit [7] of STATUS.DATA = norm_m  */
#define STATUS_NORM_M_BIT   (0x80U)

/**********************************************************************************************************************
 * Private Macros — CRC-3
 *********************************************************************************************************************/

#define CRC3_INIT           (0x7U)
#define CRC3_POLY           (0x3U)
#define CRC3_MASK           (0x7U)
#define CRC3_MSB_SHIFT      (2U)
#define FRAME_MSB           (23U)
#define FRAME_CRC_LSB       (3U)

/**********************************************************************************************************************
 * Configuration Tables
 *
 * Startup sequence: 13 write frames (AP32541 12V, 3-shunt 10mΩ).
 * Frame 9 (reg 0x00 = 0xAC) is last — locks config, triggers NORMAL mode.
 *********************************************************************************************************************/

const CddTle9180_SpiTx_T CddTle9180_StartupConfig_G[CDD_TLE9180_STARTUP_CMD_COUNT] =
{
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_GEN_CFG1,    .B.DATA=CDD_TLE9180_VAL_GEN_CFG1,    .B.CRC=4U }, /* General Configuration 1            */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_GEN_CFG2,    .B.DATA=CDD_TLE9180_VAL_GEN_CFG2,    .B.CRC=0U }, /* General Configuration 2            */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_VDHP_LO,     .B.DATA=CDD_TLE9180_VAL_VDHP_LO,     .B.CRC=6U }, /* VDHP OV/UV Threshold low           */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_VDHP_HI,     .B.DATA=CDD_TLE9180_VAL_VDHP_HI,     .B.CRC=6U }, /* VDHP OV/UV Threshold high          */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_CP_HS_FAIL,  .B.DATA=CDD_TLE9180_VAL_CP_HS_FAIL,  .B.CRC=1U }, /* CP / HS Buffer Failure Modes       */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_UV_FAIL,     .B.DATA=CDD_TLE9180_VAL_UV_FAIL,     .B.CRC=3U }, /* Undervoltage Failure Modes         */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_OV_FAIL,     .B.DATA=CDD_TLE9180_VAL_OV_FAIL,     .B.CRC=3U }, /* Overvoltage  Failure Modes         */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_OC_FAIL,     .B.DATA=CDD_TLE9180_VAL_OC_FAIL,     .B.CRC=5U }, /* Overcurrent  Failure Modes         */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_MODE,        .B.DATA=CDD_TLE9180_VAL_MODE_LOCK,   .B.CRC=2U }, /* Config Signature → triggers NORMAL */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_CSA12_GAIN1, .B.DATA=CDD_TLE9180_VAL_CSA12_GAIN1, .B.CRC=3U }, /* CSA 1&2 Gain 1                     */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_CSA12_GAIN2, .B.DATA=CDD_TLE9180_VAL_CSA12_GAIN2, .B.CRC=7U }, /* CSA 1&2 Gain 2                     */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_CSA3_GAIN,   .B.DATA=CDD_TLE9180_VAL_CSA3_GAIN,   .B.CRC=0U }, /* CSA 3 Gain                         */
    { .B.C=1U, .B.ADDRESS=CDD_TLE9180_REG_CSA_OFFSET,  .B.DATA=CDD_TLE9180_VAL_CSA_OFFSET,  .B.CRC=0U }, /* CSA Zero-Current Offset (VRO=2.5V) */
};

/** \brief  Cyclic read command table — one entry sent per slow-task call.  */
const CddTle9180_SpiTx_T CddTle9180_ReadCmds_G[CDD_TLE9180_READ_CMD_COUNT] =
{
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_SC_LS1,        .B.DATA=0x00U, .B.CRC=7U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_SC_LS2,        .B.DATA=0x00U, .B.CRC=0U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_SC_LS3,        .B.DATA=0x00U, .B.CRC=4U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_SC_HS1,        .B.DATA=0x00U, .B.CRC=2U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_SC_HS2,        .B.DATA=0x00U, .B.CRC=6U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_SC_HS3,        .B.DATA=0x00U, .B.CRC=1U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_LIMP_HOME,     .B.DATA=0x00U, .B.CRC=5U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_PFB_GAIN,      .B.DATA=0x00U, .B.CRC=4U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_RECT_THRESH_P, .B.DATA=0x00U, .B.CRC=0U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_RECT_THRESH_A, .B.DATA=0x00U, .B.CRC=7U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_RECT_FILTER,   .B.DATA=0x00U, .B.CRC=3U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_RECT_ACCURACY, .B.DATA=0x00U, .B.CRC=0U },
    { .B.C=0U, .B.ADDRESS=CDD_TLE9180_REG_MODE,          .B.DATA=0x00U, .B.CRC=4U },
};

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/** \brief  2-frame STATUS pipeline read sequence.  */
STATIC const CddTle9180_SpiTx_T Status_ReadSeq_S[2U] =
{
    { .U = STATUS_READ_CMD },
    { .U = STATUS_NOP_CMD  },
};

/** \brief  Receive buffer for STATUS pipeline read.  */
STATIC CddTle9180_SpiRx_T Status_RxBuf_S[2U];

/**********************************************************************************************************************
 * Private Function Prototype
 *********************************************************************************************************************/
STATIC void CddTle9180_PowerOnSequence(void);

/**********************************************************************************************************************
 * GPIO Wrapper Implementations
 *********************************************************************************************************************/

void CddTle9180_AssertInhibit(void)
{
    CddGpio_SetInh_P20_0(CDDGPIO_LEVEL_LOW);
}

void CddTle9180_DeassertInhibit(void)
{
    CddGpio_SetInh_P20_0(CDDGPIO_LEVEL_HIGH);
}

void CddTle9180_AssertEnable(void)
{
    CddGpio_SetEna_P33_11(CDDGPIO_LEVEL_HIGH);
}

void CddTle9180_DeassertEnable(void)
{
    CddGpio_SetEna_P33_11(CDDGPIO_LEVEL_LOW);
}

void CddTle9180_AssertSafeOff(void)
{
    CddGpio_SetSoff_P33_10(CDDGPIO_LEVEL_LOW);
}

void CddTle9180_DeassertSafeOff(void)
{
    CddGpio_SetSoff_P33_10(CDDGPIO_LEVEL_HIGH);
}

uint32_T CddTle9180_IsErrorActive(void)
{
    uint32_T result;

    if (CddGpio_GetErr_P15_2() == 0x0U)   /* LOW = /ERR asserted = fault/sleep */
    {
        result = 0x1U;
    }
    else
    {
        result = 0x0U;
    }

    return result;
}

/**********************************************************************************************************************
 * Utility Function Implementation
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_ComputeCrc3
 *
 * CRC-3 over bits [23:3] of the 24-bit frame (21 bits), MSB first.
 * Polynomial x³+x+1 (0x3), initial value 0x7.
 *
 * Validation (startup config):
 *   Addr 0x01/0x81 → 4  |  Addr 0x02/0x0F → 0  |  Addr 0x06/0x70 → 6
 *   Addr 0x07/0x9A → 6  |  Addr 0x08/0x32 → 1  |  Addr 0x0A/0x2A → 3
 *   Addr 0x0B/0x4A → 3  |  Addr 0x13/0x2A → 5  |  Addr 0x00/0xAC → 2
 *   Addr 0x20/0x44 → 3  |  Addr 0x21/0x44 → 7  |  Addr 0x22/0x44 → 0
 *   Addr 0x23/0x9F → 0
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_ComputeCrc3(uint32_T Frame24Bit)
{
    uint32_T crc = CRC3_INIT;
    uint32_T i;
    uint32_T bit;

    for (i = FRAME_MSB; i >= FRAME_CRC_LSB; i--)
    {
        bit = (Frame24Bit >> i) & 0x1U;

        if (((crc >> CRC3_MSB_SHIFT) ^ bit) != 0x0U)
        {
            crc = ((crc << 0x1U) ^ CRC3_POLY) & CRC3_MASK;
        }
        else
        {
            crc = (crc << 0x1U) & CRC3_MASK;
        }

        if (i == 0x0U)   /* Guard against unsigned wrap on i-- */
        {
            break;
        }
    }

    return crc;
}

/**********************************************************************************************************************
 * Private Function Implementation
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_PowerOnSequence
 *
 * INH/ENA/SOFF GPIO sequence per TLE9180D datasheet p.41:
 *   1. ENA = HIGH (enable internal LDO)
 *   2. /INH = HIGH → 1 ms wait (capacitor charge)
 *   3. /INH = LOW  → 1 s  wait (enter SLEEP, /ERR goes LOW)
 *   4. /INH = HIGH → 1 ms wait (exit SLEEP, enter IDLE, /ERR goes HIGH)
 *   5. /SOFF = HIGH (release gate shutdown)
 *------------------------------------------------------------------------------------------------------------------*/
STATIC void CddTle9180_PowerOnSequence(void)
{
    uint64_T dl;

    CddTle9180_AssertEnable();

    CddTle9180_DeassertInhibit();
    dl = CddStm_GetDeadline(TimeConst_1ms);
    while (CddStm_IsDeadlineElapsed(dl) == 0x0U) { ; }

    CddTle9180_AssertInhibit();               /* → SLEEP (/ERR goes LOW)   */
    dl = CddStm_GetDeadline(TimeConst_1s);
    while (CddStm_IsDeadlineElapsed(dl) == 0x0U) { ; }

    CddTle9180_DeassertInhibit();             /* → IDLE                    */
    dl = CddStm_GetDeadline(TimeConst_1ms);
    while (CddStm_IsDeadlineElapsed(dl) == 0x0U) { ; }

    CddTle9180_DeassertSafeOff();
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_Init
 *------------------------------------------------------------------------------------------------------------------*/
void CddTle9180_Init(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    uint32_T i;

    Handle->ReadIndex = 0x0U;
    Handle->State     = CDDTLE9180_STATE_POWERON;
    Handle->FaultCode = 0x0U;

    for (i = 0x0U; i < CDD_TLE9180_SPI_BUF_SIZE; i++)
    {
        Handle->RxBuf[i].U = 0x0U;
    }

    CddQspi4_Init();
    CddTle9180_PowerOnSequence();

    Handle->State = CDDTLE9180_STATE_IDLE;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_Configure
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_Configure(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    uint32_T spi_result;
    uint32_T result;

    Handle->State = CDDTLE9180_STATE_CONFIGURING;

    CddSys_NopDelay(10000U, 10000U);   /* VDD/PLL stabilisation before first SPI frame */

    spi_result = CddQspi4_Exchange(
                     (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))CddTle9180_StartupConfig_G, /* PRQA S 11.3 */
                     (P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA))Handle->RxBuf,                /* PRQA S 11.3 */
                     CDD_TLE9180_STARTUP_CMD_COUNT);

    if (spi_result == CDD_QSPI_OK)
    {
        result = 0x1U;
    }
    else
    {
        Handle->State = CDDTLE9180_STATE_FAULT;
        result        = 0x0U;
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_IsNormalMode
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_IsNormalMode(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    uint32_T spi_result;
    uint32_T result;
    uint32_T norm_m;

    spi_result = CddQspi4_Exchange(
                     (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))Status_ReadSeq_S, /* PRQA S 11.3 */
                     (P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA))Status_RxBuf_S,     /* PRQA S 11.3 */
                     2U);

    if (spi_result == CDD_QSPI_OK)
    {
        norm_m = (uint32_T)(Status_RxBuf_S[1U].B.DATA & STATUS_NORM_M_BIT);

        if (norm_m != 0x0U)
        {
            Handle->State = CDDTLE9180_STATE_NORMAL;
            result        = 0x1U;
        }
        else
        {
            result = 0x0U;
        }
    }
    else
    {
        result = 0x0U;
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_Startup
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_Startup(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    uint32_T result;

    CddTle9180_Init(Handle);
    result = CddTle9180_Configure(Handle);

    if (result == 0x1U)
    {
        result = CddTle9180_IsNormalMode(Handle);
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_ReadRegister
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_ReadRegister(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    uint32_T spi_result;
    uint32_T result;
    uint32_T buf_idx;

    buf_idx = CDD_TLE9180_READ_BUF_OFFSET + Handle->ReadIndex;

    spi_result = CddQspi4_Exchange(
                     (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))&CddTle9180_ReadCmds_G[Handle->ReadIndex], /* PRQA S 11.3 */
                     (P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA))&Handle->RxBuf[buf_idx],                     /* PRQA S 11.3 */
                     1U);

    if (spi_result == CDD_QSPI_OK)
    {
        Handle->ReadIndex++;

        if (Handle->ReadIndex >= CDD_TLE9180_READ_CMD_COUNT)
        {
            Handle->ReadIndex = 0x0U;
        }

        result = 0x1U;
    }
    else
    {
        result = 0x0U;
    }

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_MonitorFault
 *------------------------------------------------------------------------------------------------------------------*/
void CddTle9180_MonitorFault(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    if (Handle->State == CDDTLE9180_STATE_NORMAL)
    {
        if (CddTle9180_IsErrorActive() == 0x1U)
        {
            CddTle9180_AssertSafeOff();       /* hardware shutdown first   */
            Handle->State     = CDDTLE9180_STATE_FAULT;
            Handle->FaultCode = 0x1U;
        }
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_ResetFault
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_ResetFault(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    Handle->FaultCode = 0x0U;
    Handle->State     = CDDTLE9180_STATE_POWERON;

    return CddTle9180_Startup(Handle);
}
