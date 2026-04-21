/**********************************************************************************************************************
 * \file        cdd_gate_driver_9180.c
 * \brief       Implementation of cdd_gate_driver_9180.h — TLE9180D-31QK driver.
 *
 * \details     SPI configuration register changes vs. original 3-pin code:
 *
 *              addr 0x0D  Dt_hs  Dead Time High-side
 *                  OLD: 0x19 (25 → 1000 ns) — 3-pin mode, IC generates dead-time
 *                  NEW: 0x00 (0  → 107 ns minimum) — 6-pin DTM mode, GTM DTM
 *                       generates dead-time, IC provides only the 107 ns hardware
 *                       minimum as final shoot-through guard
 *
 *              addr 0x0E  Dt_ls  Dead Time Low-side
 *                  OLD: 0x19 (1000 ns)
 *                  NEW: 0x00 (107 ns minimum) — same reason as Dt_hs
 *
 *              addr 0x08  Fm_1   Failure Mode Configuration 1
 *                  OLD: 0x32 = 0b00110010
 *                       fm_in_diag [1:0] = 10b → ARE (Auto Restart Error)
 *                  NEW: 0x30 = 0b00110000
 *                       fm_in_diag [1:0] = 00b → W (Warning only)
 *                  Reason: In 6-pin mode, ILx and /IHx are driven by the GTM
 *                  DTM with CDD_GTM_DTM_DEAD_TIME_TICKS gap between transitions.
 *                  The TLE9180D input pattern supervision sees the GTM dead-time
 *                  as a "violation" (IHx and ILx briefly both high/both low)
 *                  unless the internal Dt_hs/Dt_ls ≥ GTM dead-time.  By setting
 *                  Dt_hs/Dt_ls = min (0x00) and fm_in_diag = WARNING, the IC
 *                  flags violations in the error register but does NOT shut down
 *                  the bridge.  The GTM DTM is the authoritative dead-time source.
 *
 *              All other register values are preserved from the original
 *              gate_driver_9180.c configuration.
 *
 *              CRC3 values in the SPI frames are pre-computed for the specific
 *              address+data combinations and match the original code.  Any change
 *              to address or data requires recalculation using the TLE9180D
 *              DS CRC polynomial (x³ + x + 1, initialised with 0b110).
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              https://github.com/vectorsim/embed_sim_project
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Includes
 *********************************************************************************************************************/
#include "cdd_gate_driver_9180.h"
#include "cdd_gpio_app.h"
#include "cdd_sys_utility.h"
#include "cdd_config.h"
#include "cdd_qspi_app.h"

/**********************************************************************************************************************
 * Private Macros
 *********************************************************************************************************************/

/** \brief  Sleep/idle transition delay — wait at least 5 ms (DS tINH_Pen1)   */
#define GD9180_WAKEUP_DELAY_INNER   (1000U)
#define GD9180_WAKEUP_DELAY_OUTER   (10U)

/** \brief  Maximum /ERR polling iterations before timeout                     */
#define GD9180_ERR_POLL_MAX         (10000U)

/** \brief  SPI no-operation command — addr 0x32, read, CRC 0x03               */
#define GD9180_NOOP_CMD             (0x320003U)

/** \brief  Read operation mode register — addr 0x40, read, CRC 0x06           */
#define GD9180_READ_OPMODE_CMD      (0x400006U)

/**********************************************************************************************************************
 * Private Variables — SPI Configuration Batch
 *
 * Frame encoding:  [23]=write(1), [22:16]=addr, [15:8]=data, [7:0]=CRC3
 *
 * Register map (TLE9180D DS Chapter 15):
 *   0x00  Conf_Sig    Configuration signature         → 0xA1  (enter normal mode)
 *   0x01  Conf_Gen_1  General config 1                → 0x81
 *   0x02  Conf_Gen_2  General config 2                → 0x0F
 *   0x06  Tl_vdh      VDHP threshold                  → 0x70
 *   0x07  Tl_cbvcc    CB/VCC thresholds               → 0x9A
 *   0x08  Fm_1        Failure mode 1                  → 0x30  ← fm_in_diag=WARNING (6-pin)
 *   0x0A  Fm_3        Failure mode 3                  → 0x2A
 *   0x0B  Fm_4        Failure mode 4                  → 0x4A
 *   0x0D  Dt_hs       Dead time high-side             → 0x00  ← 107 ns min (6-pin DTM)
 *   0x0E  Dt_ls       Dead time low-side              → 0x00  ← 107 ns min (6-pin DTM)
 *   0x13  Conf_csa    CSA configuration               → 0x2A
 *   0x20  Csa_gain_1  CSA1 gain                       → 0x44  (gain=34.45, VRO=1.65V)
 *   0x21  Csa_gain_2  CSA2 gain                       → 0x44
 *   0x22  Csa_gain_3  CSA3 gain                       → 0x44
 *   0x23  Csa_cfg     CSA config / signature part     → 0x9F
 *
 * Note on CRC3:
 *   The CRC fields for unchanged registers are identical to the original code.
 *   Registers 0x08, 0x0D, 0x0E have new data bytes; their CRC3 values have been
 *   recalculated using the TLE9180D CRC3 polynomial (G(x) = x³+x+1, init=0b110).
 *   TODO: verify 0x08/0x0D/0x0E CRC3 values on bench before production use.
 *         Use the TLE9180D SPI CRC Calculator tool (Infineon application note).
 *********************************************************************************************************************/
static const uint32_T GD9180_Config_Frames_G[GD9180_CONFIGURE_CMD_SIZE] =
{
    /* addr  data   CRC3    Register            Description                          */
    0x818104U,  /* 0x01   0x81   0x04   Conf_Gen_1          unchanged               */
    0x820F00U,  /* 0x02   0x0F   0x00   Conf_Gen_2          unchanged               */
    0x867006U,  /* 0x06   0x70   0x06   Tl_vdh              unchanged               */
    0x879A06U,  /* 0x07   0x9A   0x06   Tl_cbvcc            unchanged               */
    0x883001U,  /* 0x08   0x30   0x01   Fm_1                fm_in_diag=W (6-pin)    */
    0x8A2A03U,  /* 0x0A   0x2A   0x03   Fm_3                unchanged               */
    0x8B4A03U,  /* 0x0B   0x4A   0x03   Fm_4                unchanged               */
    0x8D0003U,  /* 0x0D   0x00   0x03   Dt_hs               107 ns min (6-pin DTM)  */
    0x8E0007U,  /* 0x0E   0x00   0x07   Dt_ls               107 ns min (6-pin DTM)  */
    0x932A05U,  /* 0x13   0x2A   0x05   Conf_csa            unchanged               */
    0x80A105U,  /* 0x00   0xA1   0x05   Conf_Sig            enter normal mode        */
    0xA04403U,  /* 0x20   0x44   0x03   Csa_gain_1          gain=34.45, VRO=1.65V   */
    0xA14407U,  /* 0x21   0x44   0x07   Csa_gain_2          unchanged               */
    0xA24400U,  /* 0x22   0x44   0x00   Csa_gain_3          unchanged               */
    0xA39F00U   /* 0x23   0x9F   0x00   Csa_cfg             unchanged               */
};

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Init_Pins
 *------------------------------------------------------------------------------------------------------------------*/
void GD9180_Init_Pins(void)
{
    /* Safe default state: gate driver fully inhibited */
    GPIO_Set_INH_P20_0(GPIO_LEVEL_LOW);     /* /INH low  = sleep mode        */
    GPIO_Set_ENA_P33_11(GPIO_LEVEL_LOW);     /* ENA  low  = outputs disabled  */
    GPIO_Set_SOFF_P33_10(GPIO_LEVEL_HIGH);   /* /SOFF high = normal path      */
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Startup — recommended power-up sequence (TLE9180D DS Fig. 34)
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T GD9180_Startup(void)
{
    uint32_T     result;
    uint32_T     err_poll;
    uint32_T     configured;
    GD9180_OpMode_T op_mode;

    result = 0U;

    /* Step 1: assert sleep state */
    GPIO_Set_ENA_P33_11(GPIO_LEVEL_LOW);
    GPIO_Set_SOFF_P33_10(GPIO_LEVEL_HIGH);
    GPIO_Set_INH_P20_0(GPIO_LEVEL_LOW);
    Nop_Delay(GD9180_WAKEUP_DELAY_INNER, GD9180_WAKEUP_DELAY_OUTER);

    /* Step 2: set INH = HIGH — device powers up and enters idle mode         */
    GPIO_Set_INH_P20_0(GPIO_LEVEL_HIGH);

    /* Step 3: wait for /ERR = HIGH (device ready for configuration)          */
    /* tINH_Pen1 ≤ 5 ms, tINH_cfg ≤ 2.5 ms (DS P_5.7.26 / P_5.7.27)        */
    Nop_Delay(GD9180_WAKEUP_DELAY_INNER, GD9180_WAKEUP_DELAY_OUTER);

    err_poll = 0U;
    while ((GD9180_No_Error() == 0U) && (err_poll < GD9180_ERR_POLL_MAX))
    {
        Nop_Delay(0x1U, 0x1U);
        err_poll++;
    }

    if (err_poll >= GD9180_ERR_POLL_MAX)
    {
        /* /ERR never went high — hardware fault or power issue               */
        return result;
    }

    /* Step 4+5: send configuration batch (includes config signature 0x00)   */
    configured = GD9180_Configure();
    if (configured != 1U)
    {
        return result;
    }

    /* Brief settling time before reading back mode                           */
    Nop_Delay(GD9180_WAKEUP_DELAY_INNER, 0x1U);

    /* Step 6: read back operation mode — must be normal_m = 1               */
    GD9180_Read_Op_Mode(&op_mode);
    if (op_mode.norm_m == 1U)
    {
        result = 1U;
    }

    /* Step 7: enable output stages (caller may defer this)                  */
    /* GD9180_Enable_Outputs() is intentionally NOT called here so the        */
    /* application can start GTM PWM before enabling bridge outputs.          */

    return result;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Configure
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T GD9180_Configure(void)
{
    uint32_T frame_idx;
    uint32_T transmitted;
    uint32_T rx_frame;

    for (frame_idx = 0U; frame_idx < GD9180_CONFIGURE_CMD_SIZE; frame_idx++)
    {
        /* QSPI_TLE9180_Exchange transmits one 24-bit frame and returns the MISO frame */
        transmitted = QSPI_TLE9180_Exchange(
                          GD9180_Config_Frames_G[frame_idx],
                          &rx_frame);

        if (transmitted != 1U)
        {
            return 0U;
        }
    }

    return 1U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Read_Op_Mode
 *------------------------------------------------------------------------------------------------------------------*/
void GD9180_Read_Op_Mode(GD9180_OpMode_T * const Reg_Ptr)
{
    uint32_T tx_frames[GD9180_READ_REG_SIZE];
    uint32_T rx_frames[GD9180_READ_REG_SIZE];
    uint32_T frame_idx;
    uint32_T transmitted;
    GD9180_Rx_Frame_T miso;

    /* Two-frame read: send read command, then no-op to clock out the result  */
    tx_frames[0U] = GD9180_READ_OPMODE_CMD;
    tx_frames[1U] = GD9180_NOOP_CMD;

    /* Initialise result */
    Reg_Ptr->norm_m      = 0U;
    Reg_Ptr->rect_m      = 0U;
    Reg_Ptr->err_m       = 0U;
    Reg_Ptr->soff_m      = 0U;
    Reg_Ptr->self_test_m = 0U;
    Reg_Ptr->conf_lock   = 0U;
    Reg_Ptr->conf_m      = 0U;
    Reg_Ptr->idle_m      = 0U;

    for (frame_idx = 0U; frame_idx < GD9180_READ_REG_SIZE; frame_idx++)
    {
        transmitted = QSPI_TLE9180_Exchange(tx_frames[frame_idx], &rx_frames[frame_idx]);
        if (transmitted != 1U)
        {
            return;
        }
    }

    /* The response to the read command arrives in the second MISO frame      */
    miso.U = rx_frames[1U];

    Reg_Ptr->norm_m      = (miso.B.data & 0x80U) >> 7U;
    Reg_Ptr->rect_m      = (miso.B.data & 0x40U) >> 6U;
    Reg_Ptr->err_m       = (miso.B.data & 0x20U) >> 5U;
    Reg_Ptr->soff_m      = (miso.B.data & 0x10U) >> 4U;
    Reg_Ptr->self_test_m = (miso.B.data & 0x08U) >> 3U;
    Reg_Ptr->conf_lock   = (miso.B.data & 0x04U) >> 2U;
    Reg_Ptr->conf_m      = (miso.B.data & 0x02U) >> 1U;
    Reg_Ptr->idle_m      = (miso.B.data & 0x01U) >> 0U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_No_Error
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T GD9180_No_Error(void)
{
    /* /ERR is active-low: HIGH = no fault, LOW = fault present               */
    return GPIO_Get_ERR_P15_2();
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Safe_Off
 *------------------------------------------------------------------------------------------------------------------*/
void GD9180_Safe_Off(void)
{
    GPIO_Set_ENA_P33_11(GPIO_LEVEL_LOW);
    GPIO_Set_SOFF_P33_10(GPIO_LEVEL_LOW);   /* /SOFF low = safe state         */
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Release_Safe_Off
 *------------------------------------------------------------------------------------------------------------------*/
void GD9180_Release_Safe_Off(void)
{
    GPIO_Set_SOFF_P33_10(GPIO_LEVEL_HIGH);  /* /SOFF high = normal operation  */
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Enable_Outputs
 *------------------------------------------------------------------------------------------------------------------*/
void GD9180_Enable_Outputs(void)
{
    GPIO_Set_ENA_P33_11(GPIO_LEVEL_HIGH);
}

/*--------------------------------------------------------------------------------------------------------------------
 * GD9180_Disable_Outputs
 *------------------------------------------------------------------------------------------------------------------*/
void GD9180_Disable_Outputs(void)
{
    GPIO_Set_ENA_P33_11(GPIO_LEVEL_LOW);
}
