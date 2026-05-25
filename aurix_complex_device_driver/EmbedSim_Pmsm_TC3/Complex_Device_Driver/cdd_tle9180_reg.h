/**********************************************************************************************************************
 * \file        cdd_tle9180_reg.h
 * \brief       TLE9180D-31QK register map, SPI frame types, and table declarations.
 *
 * \details     Hardware : AP32541 Motor Control Power Board
 *              SPI      : QSPI4, 5 MHz, MODE 0, 24-bit MSB-first
 *              Pins     : P22.0 MOSI | P22.1 MISO | P22.2 CS (SLSO3) | P22.3 SCLK
 *
 * \note        MISRA C:2012:
 *              DEV-REG-001  Rule 19.2 : Union for SPI frame — accessed only via .U or .B.
 *              PRQA S 0750 applied at each union definition.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_TLE9180_REG_H_
#define CDD_TLE9180_REG_H_

#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Register Addresses
 *********************************************************************************************************************/

#define CDD_TLE9180_REG_MODE            (0x00U)
#define CDD_TLE9180_REG_GEN_CFG1        (0x01U)
#define CDD_TLE9180_REG_GEN_CFG2        (0x02U)
#define CDD_TLE9180_REG_VDHP_LO         (0x06U)
#define CDD_TLE9180_REG_VDHP_HI         (0x07U)
#define CDD_TLE9180_REG_CP_HS_FAIL      (0x08U)
#define CDD_TLE9180_REG_UV_FAIL         (0x0AU)
#define CDD_TLE9180_REG_OV_FAIL         (0x0BU)
#define CDD_TLE9180_REG_OC_FAIL         (0x13U)
#define CDD_TLE9180_REG_CSA12_GAIN1     (0x20U)
#define CDD_TLE9180_REG_CSA12_GAIN2     (0x21U)
#define CDD_TLE9180_REG_CSA3_GAIN       (0x22U)
#define CDD_TLE9180_REG_CSA_OFFSET      (0x23U)
#define CDD_TLE9180_REG_SC_LS1          (0x25U)
#define CDD_TLE9180_REG_SC_LS2          (0x26U)
#define CDD_TLE9180_REG_SC_LS3          (0x27U)
#define CDD_TLE9180_REG_SC_HS1          (0x28U)
#define CDD_TLE9180_REG_SC_HS2          (0x29U)
#define CDD_TLE9180_REG_SC_HS3          (0x2AU)
#define CDD_TLE9180_REG_LIMP_HOME       (0x2BU)
#define CDD_TLE9180_REG_PFB_GAIN        (0x2CU)
#define CDD_TLE9180_REG_RECT_THRESH_P   (0x2DU)
#define CDD_TLE9180_REG_RECT_THRESH_A   (0x2EU)
#define CDD_TLE9180_REG_RECT_FILTER     (0x2FU)
#define CDD_TLE9180_REG_RECT_ACCURACY   (0x30U)

/**********************************************************************************************************************
 * Configuration Data Values  (AP32541, 12V, 3-shunt 10mΩ, CSA ~30.81 V/V)
 *********************************************************************************************************************/

#define CDD_TLE9180_VAL_GEN_CFG1        (0x81U)
#define CDD_TLE9180_VAL_GEN_CFG2        (0x0FU)
#define CDD_TLE9180_VAL_VDHP_LO         (0x70U)
#define CDD_TLE9180_VAL_VDHP_HI         (0x9AU)
#define CDD_TLE9180_VAL_CP_HS_FAIL      (0x32U)
#define CDD_TLE9180_VAL_UV_FAIL         (0x2AU)
#define CDD_TLE9180_VAL_OV_FAIL         (0x4AU)
#define CDD_TLE9180_VAL_OC_FAIL         (0x2AU)
#define CDD_TLE9180_VAL_MODE_LOCK       (0xACU)  /**< Locks config, triggers NORMAL mode */
#define CDD_TLE9180_VAL_CSA12_GAIN1     (0x44U)
#define CDD_TLE9180_VAL_CSA12_GAIN2     (0x44U)
#define CDD_TLE9180_VAL_CSA3_GAIN       (0x44U)
#define CDD_TLE9180_VAL_CSA_OFFSET      (0x9FU)  /**< VRO = 2.5V, zero-current offset  */

/**********************************************************************************************************************
 * Buffer Sizing
 *********************************************************************************************************************/

#define CDD_TLE9180_STARTUP_CMD_COUNT   (13U)
#define CDD_TLE9180_READ_CMD_COUNT      (13U)
#define CDD_TLE9180_BUFFER_PAD          (4U)

#define CDD_TLE9180_SPI_BUF_SIZE        (CDD_TLE9180_STARTUP_CMD_COUNT + \
                                         CDD_TLE9180_BUFFER_PAD        + \
                                         CDD_TLE9180_READ_CMD_COUNT)

#define CDD_TLE9180_READ_BUF_OFFSET     (CDD_TLE9180_STARTUP_CMD_COUNT + 2U)

/**********************************************************************************************************************
 * SPI Frame Types  (24-bit MSB-first, TriCore little-endian bitfield)
 *********************************************************************************************************************/

/**
 * \brief  Transmit frame.
 *         Bit 23: C (1=write) | [22:16]: ADDRESS[6:0] | [15:8]: DATA[7:0] | [7:3]: Res | [2:0]: CRC
 */
typedef struct
{
    uint32_T CRC     :3;   /**< [2:0]   CRC-3 (poly x³+x+1, init 0x7)   */
    uint32_T         :5;   /**< [7:3]   Reserved — write 0x0             */
    uint32_T DATA    :8;   /**< [15:8]  Register payload                 */
    uint32_T ADDRESS :7;   /**< [22:16] 7-bit register address           */
    uint32_T C       :1;   /**< [23]    1=write, 0=read                  */
    uint32_T         :8;   /**< [31:24] Not transmitted (24-bit frame)   */
} CddTle9180_SpiTxBits_T;

typedef union                                        /* PRQA S 0750 */  /* DEV-REG-001 */
{
    uint32_T               U;
    CddTle9180_SpiTxBits_T B;
} CddTle9180_SpiTx_T;

/**
 * \brief  Receive frame.
 *         Bit23: ERROR | 22: WARN | 21: CONFVALID | 20: SPIERR | 19: SPLEVENT |
 *         [18:12]: ADDRESS | [11:4]: DATA | 3: Res | [2:0]: CRC
 */
typedef struct
{
    uint32_T CRC       :3;   /**< [2:0]   CRC-3 of received frame        */
    uint32_T           :1;   /**< [3]     Reserved                        */
    uint32_T DATA      :8;   /**< [11:4]  Register value (pipeline N+1)  */
    uint32_T ADDRESS   :7;   /**< [18:12] Echoed register address         */
    uint32_T SPLEVENT  :1;   /**< [19]    Special event flag             */
    uint32_T SPIERR    :1;   /**< [20]    SPI communication error         */
    uint32_T CONFVALID :1;   /**< [21]    Configuration valid             */
    uint32_T WARNING   :1;   /**< [22]    Warning flag                    */
    uint32_T ERROR     :1;   /**< [23]    Error flag (mirrors /ERR pin)   */
    uint32_T           :8;   /**< [31:24] Not received (24-bit frame)    */
} CddTle9180_SpiRxBits_T;

typedef union                                        /* PRQA S 0750 */  /* DEV-REG-001 */
{
    uint32_T               U;
    CddTle9180_SpiRxBits_T B;
} CddTle9180_SpiRx_T;

/**********************************************************************************************************************
 * Table Declarations  (defined in cdd_tle9180_app.c)
 *********************************************************************************************************************/

extern const CddTle9180_SpiTx_T CddTle9180_StartupConfig_G[CDD_TLE9180_STARTUP_CMD_COUNT];
extern const CddTle9180_SpiTx_T CddTle9180_ReadCmds_G[CDD_TLE9180_READ_CMD_COUNT];

#endif /* CDD_TLE9180_REG_H_ */
