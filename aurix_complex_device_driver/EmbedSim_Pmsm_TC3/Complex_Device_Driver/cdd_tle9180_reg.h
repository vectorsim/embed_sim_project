/**********************************************************************************************************************
 * \file        cdd_tle9180_reg.h
 * \brief       TLE9180D-31QK register map, SPI frame types, register bitfield unions,
 *              and pre-computed SPI frame constants.
 *
 * \details     Hardware : AP32541 Motor Control Power Board
 *              SPI      : QSPI4, 5 MHz, MODE 1 (CPOL=0, CPHA=1), 24-bit MSB-first
 *              Pins     : P22.0 MOSI | P22.1 MISO | P22.2 CS (SLSO3) | P22.3 SCLK
 *
 *              SPI Frame Constants (CDD_TLE9180_FRAME_WR_xxx / _RD_xxx)
 *              ──────────────────────────────────────────────────────────
 *              Every configuration and read-command frame is expressed as a
 *              preprocessor integer constant so that the const table arrays in
 *              cdd_tle9180_app.c can be initialised with { .U = FRAME_xxx }
 *              — valid as a C99 constant expression (ISO C99 §6.6).
 *
 *              Frame layout (24-bit, MSB first on wire):
 *                [23]    C       — 1=write, 0=read
 *                [22:16] ADDRESS — 7-bit register address
 *                [15:8]  DATA    — 8-bit register payload
 *                [7:3]   0       — reserved, always 0
 *                [2:0]   CRC     — CRC-3 (poly x³+x+1, init 0x4, MSB-first bits[23:3])
 *
 *              CRC-3 note: the correct initialisation value is 0x4 (0b100), not 0x7.
 *              All startup CRC values have been verified against the AP32541
 *              bring-up trace.  See CddTle9180_ComputeCrc3() in cdd_tle9180_app.c.
 *
 *              Register bitfield unions (CddTle9180_Reg<Name>_T)
 *              ──────────────────────────────────────────────────
 *              One union per writable startup register.  Field names and bit
 *              positions match the datasheet exactly (TLE9180D-31QK DS Rev 1.20).
 *              These unions are for documentation and runtime-dynamic frame
 *              construction only; the static tables use the FRAME_ constants above.
 *
 * \note        MISRA C:2012:
 *              DEV-REG-001  Rule 19.2 : Union for SPI frame — .U or .B only.
 *              DEV-REG-002  Rule 19.2 : Register unions   — .U or .B only.
 *              PRQA S 0750 applied at each union definition.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#ifndef CDD_TLE9180_REG_H_
#define CDD_TLE9180_REG_H_

#include "cdd_config.h"   /* embed_sim_sys_types.h + embed_sim_compiler.h */

/**********************************************************************************************************************
 * Register Addresses — Configuration
 *********************************************************************************************************************/

#define CDD_TLE9180_REG_CONF_SIG        (0x00U)  /**< DS §15.1.1.1  p120: Config signature / CRC8 lock byte  */
#define CDD_TLE9180_REG_GEN_CFG1        (0x01U)  /**< DS §15.1.1.2  p121: General configuration 1           */
#define CDD_TLE9180_REG_GEN_CFG2        (0x02U)  /**< DS §15.1.1.3  p122: General configuration 2           */
#define CDD_TLE9180_REG_TL_VDH          (0x06U)  /**< DS §15.1.1.7  p127: VDHP OV/UV thresholds             */
#define CDD_TLE9180_REG_TL_CBVCC        (0x07U)  /**< DS §15.1.1.8  p129: CB UV + VCC OV/UV thresholds      */
#define CDD_TLE9180_REG_FM1             (0x08U)  /**< DS §15.1.1.9  p130: CP2 overload + CB-UV + HS-UV modes */
#define CDD_TLE9180_REG_FM3             (0x0AU)  /**< DS §15.1.1.11 p132: Vs + VDHP + VCC UV failure modes  */
#define CDD_TLE9180_REG_FM4             (0x0BU)  /**< DS §15.1.1.12 p133: Vs + VDHP + VCC OV failure modes  */
#define CDD_TLE9180_REG_DT_HS           (0x0DU)  /**< DS §15.1.1.14 p135: Dead time high-side output stages */
#define CDD_TLE9180_REG_DT_LS           (0x0EU)  /**< DS §15.1.1.15 p136: Dead time low-side output stages  */
#define CDD_TLE9180_REG_FM6             (0x13U)  /**< DS §15.1.1.20 p141: CSA 1/2/3 overcurrent modes       */
#define CDD_TLE9180_REG_OP_GAIN1        (0x20U)  /**< DS §15.1.2.1  p142: CSA 1&2 gain stage 1              */
#define CDD_TLE9180_REG_OP_GAIN2        (0x21U)  /**< DS §15.1.2.2  p143: CSA 1&2 gain stage 2              */
#define CDD_TLE9180_REG_OP_GAIN3        (0x22U)  /**< DS §15.1.2.3  p144: CSA 3 gain stage 1&2              */
#define CDD_TLE9180_REG_OP_OCL          (0x23U)  /**< DS §15.1.2.4  p145: CSA zero-current offset           */
#define CDD_TLE9180_REG_LI_CTR          (0x2BU)  /**< DS §15.1.2.12 p153: Limp-home / half-bridge enable    */

/**********************************************************************************************************************
 * Register Addresses — Cyclic Read / Diagnostic
 *********************************************************************************************************************/

#define CDD_TLE9180_REG_SC_LS1          (0x25U)  /**< Short-circuit LS phase 1 status                       */
#define CDD_TLE9180_REG_SC_LS2          (0x26U)  /**< Short-circuit LS phase 2 status                       */
#define CDD_TLE9180_REG_SC_LS3          (0x27U)  /**< Short-circuit LS phase 3 status                       */
#define CDD_TLE9180_REG_SC_HS1          (0x28U)  /**< Short-circuit HS phase 1 status                       */
#define CDD_TLE9180_REG_SC_HS2          (0x29U)  /**< Short-circuit HS phase 2 status                       */
#define CDD_TLE9180_REG_SC_HS3          (0x2AU)  /**< Short-circuit HS phase 3 status                       */
#define CDD_TLE9180_REG_LIMP_HOME       (0x2BU)  /**< Limp-home mode configuration                          */
#define CDD_TLE9180_REG_PFB_GAIN        (0x2CU)  /**< Peak/flat-band current sense gain                     */
#define CDD_TLE9180_REG_RECT_THRESH_P   (0x2DU)  /**< Rectifier positive threshold                          */
#define CDD_TLE9180_REG_RECT_THRESH_A   (0x2EU)  /**< Rectifier adaptive threshold                          */
#define CDD_TLE9180_REG_RECT_FILTER     (0x2FU)  /**< Rectifier filter time constant                        */
#define CDD_TLE9180_REG_RECT_ACCURACY   (0x30U)  /**< Rectifier accuracy/hysteresis                         */

/**********************************************************************************************************************
 * Register Addresses — Error / Fault Status (DS §14, Register Map p118)
 *
 * Reading any register in this range clears the WARNING and ERROR SPI status flags
 * for the associated fault condition, provided the fault is no longer present.
 * Exception: 0x4A (Err_spiconf) bits 0,1,3,4 are NOT cleared by read — they
 * require a separate write-to-clear sequence per DS §14.3.
 *********************************************************************************************************************/

#define CDD_TLE9180_REG_ERR_OVER        (0x41U)  /**< Error Overview                                        */
#define CDD_TLE9180_REG_SER             (0x42U)  /**< Special Event Register                                */
#define CDD_TLE9180_REG_ERR_I_1         (0x43U)  /**< Internal Errors 1                                     */
#define CDD_TLE9180_REG_ERR_I_2         (0x44U)  /**< Internal Errors 2                                     */
#define CDD_TLE9180_REG_ERR_E           (0x45U)  /**< External Errors                                       */
#define CDD_TLE9180_REG_ERR_SD          (0x46U)  /**< Shutdown Errors                                       */
#define CDD_TLE9180_REG_ERR_SCD         (0x47U)  /**< Short Circuit Errors                                  */
#define CDD_TLE9180_REG_ERR_INDIAG      (0x48U)  /**< Input Pattern Violations                              */
#define CDD_TLE9180_REG_ERR_OSF         (0x49U)  /**< Output Stage Feedback Errors                          */
#define CDD_TLE9180_REG_ERR_SPICONF     (0x4AU)  /**< SPI Communication and Configuration Errors (NOT cleared by read) */
#define CDD_TLE9180_REG_ERR_OP_12       (0x4BU)  /**< Current Sense Amplifiers 1 & 2 Errors                 */
#define CDD_TLE9180_REG_ERR_OP_3        (0x4CU)  /**< Current Sense Amplifier 3 Errors                      */
#define CDD_TLE9180_REG_ERR_OUTP        (0x4DU)  /**< Digital Output Pin Errors                             */

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

/** Number of error registers swept by CddTle9180_ClearFaults().
 *  Registers 0x41–0x4D, excluding 0x4A (Err_spiconf — not cleared by read). */
#define CDD_TLE9180_ERR_REG_COUNT       (12U)

/**********************************************************************************************************************
 * Pre-computed SPI Frame Constants
 *
 * Each constant is a fully formed 24-bit frame word (stored in bits [23:0] of
 * a uint32_T) including the correct CRC-3 in bits [2:0].
 *
 * CRC-3: polynomial x³+x+1 (0x3), initialisation 0x4, MSB-first over bits [23:3].
 * All values verified against AP32541 bring-up trace.
 *
 * Frame bit layout:
 *   [23]    C=1 (write) or C=0 (read)
 *   [22:16] ADDRESS[6:0]
 *   [15:8]  DATA[7:0]   (meaningful only for write frames)
 *   [7:3]   0x00        (reserved)
 *   [2:0]   CRC[2:0]
 *********************************************************************************************************************/

/* ── Startup write frames (C=1) ──────────────────────────────────────────────
 * Field-by-field derivation of each DATA byte is documented in cdd_tle9180_app.c. */

/** GEN_CFG1 (0x01) = 0x81: tl_ot_w=100(140°C), vcc_select=1(5V)               */
#define CDD_TLE9180_FRAME_WR_GEN_CFG1      (0x818104U)  /* addr=0x01 data=0x81 CRC=4 */

/** GEN_CFG2 (0x02) = 0x0F: en_op1/2/3=1, en_vdh3=1, all OV detections enabled */
#define CDD_TLE9180_FRAME_WR_GEN_CFG2      (0x820F00U)  /* addr=0x02 data=0x0F CRC=0 */

/** TL_VDH (0x06) = 0x70: tl_ov_vdh=0111(48.18V), tl_uv_vdh=0000(18.00V)      */
#define CDD_TLE9180_FRAME_WR_TL_VDH        (0x867006U)  /* addr=0x06 data=0x70 CRC=6 */

/** TL_CBVCC (0x07) = 0x9A: tl_uv_cb=1001(9.07V), tl_ov_vcc=10(10%), tl_uv_vcc=10(10%) */
#define CDD_TLE9180_FRAME_WR_TL_CBVCC      (0x879A06U)  /* addr=0x07 data=0x9A CRC=6 */

/** FM1 (0x08) = 0x32: fm_uv_cb=00(W), Res[5]=1, fm_cp2_off=1(shutdown), fm_uv_bs=01(ERR) */
#define CDD_TLE9180_FRAME_WR_FM1           (0x883201U)  /* addr=0x08 data=0x32 CRC=1 */

/** FM3 (0x0A) = 0x2A: fm_uv_vs/vdh/vcc=10(ARE)                                 */
#define CDD_TLE9180_FRAME_WR_FM3           (0x8A2A03U)  /* addr=0x0A data=0x2A CRC=3 */

/** FM4 (0x0B) = 0x4A: fm_ov_vs=10(ARE), fm_ov_vdh=010(ARE all FETs), fm_ov_vcc=10(ARE) */
#define CDD_TLE9180_FRAME_WR_FM4           (0x8B4A03U)  /* addr=0x0B data=0x4A CRC=3 */

/** FM6 (0x13) = 0x2A: fm_oc_op1/2/3=10(ARE)                                    */
#define CDD_TLE9180_FRAME_WR_FM6           (0x932A05U)  /* addr=0x13 data=0x2A CRC=5 */

/** CONF_SIG (0x00) = 0xAC: CRC8 lock byte — commits config, enters NORMAL mode  */
#define CDD_TLE9180_FRAME_WR_CONF_SIG      (0x80AC02U)  /* addr=0x00 data=0xAC CRC=2 */

/** OP_GAIN1 (0x20) = 0x44: op1_gain1=100(30.81V/V), op2_gain1=100(30.81V/V)   */
#define CDD_TLE9180_FRAME_WR_OP_GAIN1      (0xA04403U)  /* addr=0x20 data=0x44 CRC=3 */

/** OP_GAIN2 (0x21) = 0x44: op1_gain2=100(30.81V/V), op2_gain2=100(30.81V/V)   */
#define CDD_TLE9180_FRAME_WR_OP_GAIN2      (0xA14407U)  /* addr=0x21 data=0x44 CRC=7 */

/** OP_GAIN3 (0x22) = 0x44: op3_gain2=100(30.81V/V), op3_gain1=100(30.81V/V)   */
#define CDD_TLE9180_FRAME_WR_OP_GAIN3      (0xA24400U)  /* addr=0x22 data=0x44 CRC=0 */

/** OP_OCL (0x23) = 0x9F: zcl=10(VRO=2.5V), ofs=011111(no fine adjustment)      */
#define CDD_TLE9180_FRAME_WR_OP_OCL        (0xA39F00U)  /* addr=0x23 data=0x9F CRC=0 */

/* ── Cyclic read command frames (C=0, DATA=0x00) ─────────────────────────────*/

#define CDD_TLE9180_FRAME_RD_SC_LS1          (0x250007U)  /**< addr=0x25 CRC=7 */
#define CDD_TLE9180_FRAME_RD_SC_LS2          (0x260000U)  /**< addr=0x26 CRC=0 */
#define CDD_TLE9180_FRAME_RD_SC_LS3          (0x270004U)  /**< addr=0x27 CRC=4 */
#define CDD_TLE9180_FRAME_RD_SC_HS1          (0x280002U)  /**< addr=0x28 CRC=2 */
#define CDD_TLE9180_FRAME_RD_SC_HS2          (0x290006U)  /**< addr=0x29 CRC=6 */
#define CDD_TLE9180_FRAME_RD_SC_HS3          (0x2A0001U)  /**< addr=0x2A CRC=1 */
#define CDD_TLE9180_FRAME_RD_LIMP_HOME       (0x2B0005U)  /**< addr=0x2B CRC=5 */
#define CDD_TLE9180_FRAME_RD_PFB_GAIN        (0x2C0004U)  /**< addr=0x2C CRC=4 */
#define CDD_TLE9180_FRAME_RD_RECT_THRESH_P   (0x2D0000U)  /**< addr=0x2D CRC=0 */
#define CDD_TLE9180_FRAME_RD_RECT_THRESH_A   (0x2E0007U)  /**< addr=0x2E CRC=7 */
#define CDD_TLE9180_FRAME_RD_RECT_FILTER     (0x2F0003U)  /**< addr=0x2F CRC=3 */
#define CDD_TLE9180_FRAME_RD_RECT_ACCURACY   (0x300000U)  /**< addr=0x30 CRC=0 */
#define CDD_TLE9180_FRAME_RD_CONF_SIG        (0x000004U)  /**< addr=0x00 CRC=4 */

/* ── Error register read frames (C=0, DATA=0x00) — used by CddTle9180_ClearFaults()
 *
 * Reading these registers clears the associated WARNING/ERROR SPI status flag bits
 * once the fault condition is no longer present (DS §14.3.1, p118).
 * 0x4A (Err_spiconf) is intentionally omitted — not cleared by read.
 * CRC-3 verified with poly=0x3, init=0x4, MSB-first over bits [23:3].            */

#define CDD_TLE9180_FRAME_RD_ERR_OVER        (0x410002U)  /**< addr=0x41 CRC=2 — Error Overview             */
#define CDD_TLE9180_FRAME_RD_SER             (0x420005U)  /**< addr=0x42 CRC=5 — Special Event Register     */
#define CDD_TLE9180_FRAME_RD_ERR_I_1         (0x430001U)  /**< addr=0x43 CRC=1 — Internal Errors 1          */
#define CDD_TLE9180_FRAME_RD_ERR_I_2         (0x440000U)  /**< addr=0x44 CRC=0 — Internal Errors 2          */
#define CDD_TLE9180_FRAME_RD_ERR_E           (0x450004U)  /**< addr=0x45 CRC=4 — External Errors            */
#define CDD_TLE9180_FRAME_RD_ERR_SD          (0x460003U)  /**< addr=0x46 CRC=3 — Shutdown Errors            */
#define CDD_TLE9180_FRAME_RD_ERR_SCD         (0x470007U)  /**< addr=0x47 CRC=7 — Short Circuit Errors       */
#define CDD_TLE9180_FRAME_RD_ERR_INDIAG      (0x480001U)  /**< addr=0x48 CRC=1 — Input Pattern Violations   */
#define CDD_TLE9180_FRAME_RD_ERR_OSF         (0x490005U)  /**< addr=0x49 CRC=5 — Output Stage Feedback Errs */
#define CDD_TLE9180_FRAME_RD_ERR_OP_12       (0x4B0006U)  /**< addr=0x4B CRC=6 — CSA 1 & 2 Errors          */
#define CDD_TLE9180_FRAME_RD_ERR_OP_3        (0x4C0007U)  /**< addr=0x4C CRC=7 — CSA 3 Errors              */
#define CDD_TLE9180_FRAME_RD_ERR_OUTP        (0x4D0003U)  /**< addr=0x4D CRC=3 — Digital Output Pin Errors  */

/**********************************************************************************************************************
 * SPI Frame Types  (24-bit MSB-first, TriCore little-endian bitfield)
 *********************************************************************************************************************/

/**
 * \brief  Transmit frame.
 *
 *  Bit layout (24-bit wire format, MSB first):
 *  ┌─────┬─────────────┬────────────────┬──────────┬───────┐
 *  │ 23  │  [22:16]    │   [15:8]       │  [7:3]   │ [2:0] │
 *  │  C  │ ADDRESS[6:0]│   DATA[7:0]    │ Reserved │  CRC  │
 *  │ r/w │ reg addr    │ write payload  │   0x0    │ CRC-3 │
 *  └─────┴─────────────┴────────────────┴──────────┴───────┘
 */
typedef struct
{
    uint32_T CRC      :3;   /**< [2:0]   CRC-3 (poly x³+x+1, init 0x4)  */
    uint32_T DUMMY5_3 :5;   /**< [7:3]   Reserved — write 0x0            */
    uint32_T DATA     :8;   /**< [15:8]  Register payload                */
    uint32_T ADDRESS  :7;   /**< [22:16] 7-bit register address          */
    uint32_T C        :1;   /**< [23]    1=write, 0=read                 */
    uint32_T DUMMY8_24:8;   /**< [31:24] Not transmitted (24-bit frame)  */
} CddTle9180_SpiTxBits_T;

typedef union                                        /* PRQA S 0750 */  /* DEV-REG-001 */
{
    uint32_T               U;
    CddTle9180_SpiTxBits_T B;
} CddTle9180_SpiTx_T;

/**
 * \brief  Receive frame (pipelined — response belongs to the previous command).
 *
 *  Bit layout (24-bit wire format):
 *  ┌───────┬──────┬───────────┬────────┬──────────┬─────────────┬──────────┬───┬───────┐
 *  │  23   │  22  │    21     │   20   │    19    │   [18:12]   │  [11:4]  │ 3 │ [2:0] │
 *  │ ERROR │ WARN │CONFVALID  │ SPIERR │ SPLEVENT │ ADDRESS[6:0]│ DATA[7:0]│Res│  CRC  │
 *  └───────┴──────┴───────────┴────────┴──────────┴─────────────┴──────────┴───┴───────┘
 */
typedef struct
{
    uint32_T CRC       :3;   /**< [2:0]   CRC-3 of received frame         */
    uint32_T DUMMY1_3  :1;   /**< [3]     Reserved                         */
    uint32_T DATA      :8;   /**< [11:4]  Register value (pipeline N+1)   */
    uint32_T ADDRESS   :7;   /**< [18:12] Echoed register address          */
    uint32_T SPLEVENT  :1;   /**< [19]    Special event flag               */
    uint32_T SPIERR    :1;   /**< [20]    SPI communication error          */
    uint32_T CONFVALID :1;   /**< [21]    Configuration valid              */
    uint32_T WARNING   :1;   /**< [22]    Warning flag                     */
    uint32_T ERROR     :1;   /**< [23]    Error flag (mirrors /ERR pin)    */
    uint32_T DUMMY8_24 :8;   /**< [31:24] Not received (24-bit frame)      */
} CddTle9180_SpiRxBits_T;

typedef union                                        /* PRQA S 0750 */  /* DEV-REG-001 */
{
    uint32_T               U;
    CddTle9180_SpiRxBits_T B;
} CddTle9180_SpiRx_T;

/**********************************************************************************************************************
 * Register Bitfield Unions — for documentation and runtime dynamic frame construction
 *
 * These unions are NOT used in the static const tables (which use FRAME_ constants).
 * They are available for CddTle9180_ComputeCrc3() callers that need to build frames
 * dynamically at runtime (e.g. dead-time adjustment, gain trim at commissioning).
 *********************************************************************************************************************/

/** Conf_Gen_1 — DS §15.1.1.2 p121 — Address 0x01H, reset 0x80H */
typedef struct
{
    uint32_T VCC_SELECT  :1;   /**< [0]    VCC monitoring: 1=5V, 0=3.3V (default)          */
    uint32_T VCC_SUP_OFF :1;   /**< [1]    VCC supervision: 1=off, 0=on (default)           */
    uint32_T LIMP_ACT    :1;   /**< [2]    Limp home: 1=enabled, 0=disabled (default)       */
    uint32_T SPI_WWD_ACT :1;   /**< [3]    SPI watchdog: 1=enabled, 0=disabled (default)    */
    uint32_T IN_DIAG_ACT :1;   /**< [4]    Input supervision: 1=enabled, 0=off (default)    */
    uint32_T TL_OT_W     :3;   /**< [7:5]  OT threshold: 0b100=140°C (default), 0b111=125°C */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                         */
} CddTle9180_RegGenCfg1Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegGenCfg1Bits_T B;
} CddTle9180_RegGenCfg1_T;

/** Conf_Gen_2 — DS §15.1.1.3 p122 — Address 0x02H, reset 0x03H */
typedef struct
{
    uint32_T EN_OP1        :1;   /**< [0]    CSA1+ref: 1=on (default), 0=off            */
    uint32_T EN_OP2        :1;   /**< [1]    CSA2: 1=on (default), 0=off               */
    uint32_T EN_OP3        :1;   /**< [2]    CSA3: 1=on, 0=off (default)               */
    uint32_T EN_VDH3       :1;   /**< [3]    VDH: 1=3-pin, 0=1-VDHP (default)         */
    uint32_T DIS_SD_VDH    :1;   /**< [4]    OV SD at VDHP: 1=off, 0=on (default)      */
    uint32_T DIS_OV_LD_VDH :1;   /**< [5]    OV LD at VDHP: 1=off, 0=on (default)      */
    uint32_T DIS_OV_BH     :1;   /**< [6]    OV HS caps: 1=off, 0=on (default)         */
    uint32_T TL_OC_OP      :1;   /**< [7]    OC threshold: 1=3.3V, 0=5V (default)      */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                   */
} CddTle9180_RegGenCfg2Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegGenCfg2Bits_T B;
} CddTle9180_RegGenCfg2_T;

/** Tl_vdh — DS §15.1.1.7 p127 — Address 0x06H, reset 0xA0H */
typedef struct
{
    uint32_T TL_UV_VDH :4;   /**< [3:0]  VDHP UV: 0=3.96V (default), 7=7.01V, F=39.95V */
    uint32_T TL_OV_VDH :4;   /**< [7:4]  VDHP OV: A=56.11V (default), 7=48.18V         */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                        */
} CddTle9180_RegTlVdhBits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegTlVdhBits_T B;
} CddTle9180_RegTlVdh_T;

/** Tl_cbvcc — DS §15.1.1.8 p129 — Address 0x07H, reset 0x95H */
typedef struct
{
    uint32_T TL_UV_VCC :2;   /**< [1:0]  VCC UV: 10=10%, 01=4% (default)               */
    uint32_T TL_OV_VCC :2;   /**< [3:2]  VCC OV: 10=10%, 01=4% (default)               */
    uint32_T TL_UV_CB  :4;   /**< [7:4]  CB UV: 9=9.07V (default), F=10.44V            */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                       */
} CddTle9180_RegTlCbvccBits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegTlCbvccBits_T B;
} CddTle9180_RegTlCbvcc_T;

/** Fm_1 — DS §15.1.1.9 p130 — Address 0x08H, reset 0x30H */
typedef struct
{
    uint32_T FM_UV_BS   :2;   /**< [1:0]  HS buf UV: 01=ERR, 00=W (default)             */
    uint32_T DUMMY2_3 :2;   /**< [3:2]  Reserved — write 0x0                           */
    uint32_T FM_CP2_OFF :1;   /**< [4]    CP2 OL: 1=shutdown (default)                  */
    uint32_T DUMMY1_5 :1;   /**< [5]    Reserved — MUST write 0x1 (default value)      */
    uint32_T FM_UV_CB   :2;   /**< [7:6]  CB UV: 00=W (default)                         */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                       */
} CddTle9180_RegFm1Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegFm1Bits_T B;
} CddTle9180_RegFm1_T;

/** Fm_3 — DS §15.1.1.11 p132 — Address 0x0AH, reset 0x10H */
typedef struct
{
    uint32_T FM_UV_VCC :2;   /**< [1:0]  VCC UV: 10=ARE, 01=ERR (default)              */
    uint32_T FM_UV_VDH :2;   /**< [3:2]  VDHP UV: 10=ARE, 00=W (default)               */
    uint32_T FM_UV_VS  :2;   /**< [5:4]  Vs UV: 10=ARE, 01=ERR (default)               */
    uint32_T DUMMY2_7 :2;   /**< [7:6]  Reserved — write 0x0                           */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                       */
} CddTle9180_RegFm3Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegFm3Bits_T B;
} CddTle9180_RegFm3_T;

/** Fm_4 — DS §15.1.1.12 p133 — Address 0x0BH, reset 0x20H */
typedef struct
{
    uint32_T FM_OV_VCC :2;   /**< [1:0]  VCC OV: 10=ARE, 00=W (default)                */
    uint32_T FM_OV_VDH :3;   /**< [4:2]  VDHP OV: 010=ARE all FETs, 000=W (default)    */
    uint32_T FM_OV_VS  :2;   /**< [6:5]  Vs OV: 10=ARE, 01=ERR (default)               */
    uint32_T DUMMY1_7 :1;   /**< [7]    Reserved — write 0x0                           */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                       */
} CddTle9180_RegFm4Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegFm4Bits_T B;
} CddTle9180_RegFm4_T;

/** Fm_6 — DS §15.1.1.20 p141 — Address 0x13H, reset 0x00H */
typedef struct
{
    uint32_T FM_OC_OP1 :2;   /**< [1:0]  CSA1 OC: 10=ARE, 00=W (default)               */
    uint32_T FM_OC_OP2 :2;   /**< [3:2]  CSA2 OC: 10=ARE, 00=W (default)               */
    uint32_T FM_OC_OP3 :2;   /**< [5:4]  CSA3 OC: 10=ARE, 00=W (default)               */
    uint32_T DUMMY2_7 :2;   /**< [7:6]  Reserved — write 0x0                           */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                       */
} CddTle9180_RegFm6Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegFm6Bits_T B;
} CddTle9180_RegFm6_T;

/** Dt_hs — DS §15.1.1.14 p135 — Address 0x0DH, reset 0x0EH (600 ns) */
typedef struct
{
    uint32_T DTHS :8;   /**< [7:0]  HS dead time: t_ns = value×35.7+107; 0x0E=600ns (default) */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                                   */
} CddTle9180_RegDtHsBits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegDtHsBits_T B;
} CddTle9180_RegDtHs_T;

/** Dt_ls — DS §15.1.1.15 p136 — Address 0x0EH, reset 0x0EH (600 ns) */
typedef struct
{
    uint32_T DTLS :8;   /**< [7:0]  LS dead time: t_ns = value×35.7+107; 0x0E=600ns (default) */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                                   */
} CddTle9180_RegDtLsBits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegDtLsBits_T B;
} CddTle9180_RegDtLs_T;

/** Op_gain_1 — DS §15.1.2.1 p142 — Address 0x20H, reset 0x33H */
typedef struct
{
    uint32_T OP2_GAIN1 :3;   /**< [2:0]  CSA2 gain1: 100=30.81V/V, 011=26.90 (default) */
    uint32_T DUMMY1_3 :1;   /**< [3]    Reserved — write 0x0                            */
    uint32_T OP1_GAIN1 :3;   /**< [6:4]  CSA1 gain1: 100=30.81V/V, 011=26.90 (default) */
    uint32_T DUMMY1_7 :1;   /**< [7]    Reserved — write 0x0                            */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                        */
} CddTle9180_RegOpGain1Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegOpGain1Bits_T B;
} CddTle9180_RegOpGain1_T;

/** Op_gain_2 — DS §15.1.2.2 p143 — Address 0x21H, reset 0x55H */
typedef struct
{
    uint32_T OP2_GAIN2 :3;   /**< [2:0]  CSA2 gain2: 100=30.81V/V, 101=34.45 (default) */
    uint32_T DUMMY1_3 :1;   /**< [3]    Reserved — write 0x0                            */
    uint32_T OP1_GAIN2 :3;   /**< [6:4]  CSA1 gain2: 100=30.81V/V, 101=34.45 (default) */
    uint32_T DUMMY1_7 :1;   /**< [7]    Reserved — write 0x0                            */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                        */
} CddTle9180_RegOpGain2Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegOpGain2Bits_T B;
} CddTle9180_RegOpGain2_T;

/** Op_gain_3 — DS §15.1.2.3 p144 — Address 0x22H, reset 0x53H */
typedef struct
{
    uint32_T OP3_GAIN1 :3;   /**< [2:0]  CSA3 gain1: 100=30.81V/V, 011=26.90 (default) */
    uint32_T DUMMY1_3 :1;   /**< [3]    Reserved — write 0x0                            */
    uint32_T OP3_GAIN2 :3;   /**< [6:4]  CSA3 gain2: 100=30.81V/V, 101=34.45 (default) */
    uint32_T DUMMY1_7 :1;   /**< [7]    Reserved — write 0x0                            */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                        */
} CddTle9180_RegOpGain3Bits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegOpGain3Bits_T B;
} CddTle9180_RegOpGain3_T;

/** Op_0cl — DS §15.1.2.4 p145 — Address 0x23H, reset 0x5FH */
typedef struct
{
    uint32_T OFS :6;   /**< [5:0]  Fine offset: 0x1F=none (default), 0x3F=most+, 0x00=most- */
    uint32_T ZCL :2;   /**< [7:6]  VRO: 10=2.5V, 01=1.65V (default), 00=0.5V               */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                                  */
} CddTle9180_RegOpOclBits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegOpOclBits_T B;
} CddTle9180_RegOpOcl_T;

/** Li_ctr — DS §15.1.2.12 p153 — Address 0x2BH, reset 0xE8H */
typedef struct
{
    uint32_T DIS_HB1 :1;   /**< [0]    Disable HB1: 1=off, 0=active (default)          */
    uint32_T DIS_HB2 :1;   /**< [1]    Disable HB2: 1=off, 0=active (default)          */
    uint32_T DIS_HB3 :1;   /**< [2]    Disable HB3: 1=off, 0=active (default)          */
    uint32_T EX_LIMP :1;   /**< [3]    Exit limp: 1=exit (default), 0=enter            */
    uint32_T EN_LIMP :1;   /**< [4]    Enter limp: 1=enter, 0=exit (default)           */
    uint32_T EN_HB1  :1;   /**< [5]    Enable HB1: 1=active (default), 0=off           */
    uint32_T EN_HB2  :1;   /**< [6]    Enable HB2: 1=active (default), 0=off           */
    uint32_T EN_HB3  :1;   /**< [7]    Enable HB3: 1=active (default), 0=off           */
    uint32_T DUMMY24 :24;  /**< [31:8] Not part of DATA payload                         */
} CddTle9180_RegLiCtrBits_T;
typedef union                                        /* PRQA S 0750 */  /* DEV-REG-002 */
{
    uint32_T   U;
    CddTle9180_RegLiCtrBits_T B;
} CddTle9180_RegLiCtr_T;

/**********************************************************************************************************************
 * Table Declarations  (defined in cdd_tle9180_app.c)
 *********************************************************************************************************************/

extern const CddTle9180_SpiTx_T CddTle9180_StartupConfig_G[CDD_TLE9180_STARTUP_CMD_COUNT];
extern const CddTle9180_SpiTx_T CddTle9180_ReadCmds_G[CDD_TLE9180_READ_CMD_COUNT];

#endif /* CDD_TLE9180_REG_H_ */
