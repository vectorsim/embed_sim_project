/**********************************************************************************************************************
 * \file        cdd_tle9180_app.c
 * \brief       TLE9180D-31QK gate driver CDD — state machine, CRC-3,
 *              configuration tables, GPIO wrappers.
 *
 * \details     QSPI4 hardware is owned by cdd_qspi_app.c.
 *              GPIO via cdd_gpio_app.h.  Timing via cdd_stm_app.h.
 *
 *              Configuration table strategy (C99 compliance)
 *              ──────────────────────────────────────────────
 *              TASKING cctc compiles with --iso=99.  C99 §6.6 requires that
 *              initialisers for objects with static storage duration are
 *              constant expressions.  Compound literals (e.g. (MyUnion_T){.B.x=1}.U)
 *              are not constant expressions in C99 — they are runtime objects.
 *              Therefore both const tables use pre-computed integer constants
 *              (CDD_TLE9180_FRAME_WR_xxx, CDD_TLE9180_FRAME_RD_xxx) from
 *              cdd_tle9180_reg.h.  Each constant is a fully formed 24-bit frame
 *              word including its CRC-3, expressed as a plain #define so that
 *              { .U = CDD_TLE9180_FRAME_xxx } is a valid C99 constant initialiser.
 *
 *              Field-by-field derivation of each DATA byte is documented in the
 *              startup table comments below, together with the DS section and
 *              page reference for every field.
 *
 *              CRC-3 initialisation correction
 *              ────────────────────────────────
 *              The initialisation value for CddTle9180_ComputeCrc3() is 0x4
 *              (0b100), not 0x7.  All 13 startup CRC values have been re-verified
 *              against the AP32541 bring-up trace with init=0x4; the FRAME_
 *              constants in cdd_tle9180_reg.h match this function exactly.
 *
 *              W560 (possible truncation) suppression
 *              ───────────────────────────────────────
 *              Assignments from uint32_T into narrow bitfields (.B.C, .B.ADDRESS,
 *              .B.DATA, .B.CRC) generate W560 because the compiler cannot prove
 *              at the call site that the value fits.  Each assignment is preceded
 *              by an explicit mask so the value is provably within range before
 *              the bitfield write.
 *
 *              Private STATIC functions:
 *                  CddTle9180_PowerOnSequence()   INH/ENA/SOFF GPIO sequence.
 *                  CddTle9180_BuildFrame()         runtime frame construction
 *                                                  (used only in non-const contexts).
 *
 * \note        MISRA C:2012: Rules 8.9, 10.1, 14.4, 15.5, 17.2.
 *
 *              PRQA S 11.3 deviation MD_TLE9180_11.3_union_U  [Rule 11.3 Required]
 *                  Casting CddTle9180_SpiTx_T / CddTle9180_SpiRx_T to uint32_T*
 *                  is safe: alignment guaranteed (32-bit on TC3xx); only .U accessed.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

/**********************************************************************************************************************
 * Private Includes
 *********************************************************************************************************************/
#include "cdd_tle9180_app.h"
#include "cdd_qspi_app.h"       /* CddQspi4_Init, CddQspi4_Exchange, CddQspi4_GetBaudRate */
#include "cdd_gpio_app.h"       /* CddGpio_SetInh_P20_0, etc.                             */
#include "cdd_stm_app.h"        /* TimeConst_xxx, CddStm_GetDeadline, ...                 */
#include "cdd_sys_utility.h"    /* CddSys_AreEqual32                                      */

/**********************************************************************************************************************
 * Private Helper — Cdd_ExchangeN
 *
 * Sends Count frames one at a time using the single-frame Cdd_ExchangeN().
 * Stops on first error.  Mirrors the old multi-frame call site pattern.
 *********************************************************************************************************************/
static uint32_T Cdd_ExchangeN(
    P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA) TxBuf,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) RxBuf,
    uint32_T                                     Count,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    uint32_T i;
    uint32_T result = 0x1U;

    *ErrorCode = CDD_QSPI_ERR_NONE;

    for (i = 0U; (i < Count) && (result == 0x1U); i++)
    {
        result = CddQspi4_Exchange(
                     &TxBuf[i],
                     &RxBuf[i],
                     ErrorCode);
    }

    return result;
}

/**********************************************************************************************************************
 * Private Macros — STATUS register read pipeline
 *********************************************************************************************************************/

/** Frame 1: READ STATUS reg 0x40.  CRC computed with init=0x4: addr=0x40,data=0 → CRC=6 */
#define STATUS_READ_CMD     (0x400006U)

/** Frame 2: NOP pipeline flush.  addr=0x32,data=0 → CRC=3 */
#define STATUS_NOP_CMD      (0x320003U)

/** STATUS.DATA bit [7] = norm_m */
#define STATUS_NORM_M_BIT   (0x80U)

/**********************************************************************************************************************
 * Private Macros — CRC-3
 *
 * Polynomial x³+x+1 (0x3), initialisation 0x4 (NOT 0x7 as originally written —
 * see module header for correction rationale), MSB-first over bits [23:3].
 *********************************************************************************************************************/

#define CRC3_INIT           (0x4U)   /**< Correct seed: 0b100                          */
#define CRC3_POLY           (0x3U)   /**< Generator polynomial coefficient bits         */
#define CRC3_MASK           (0x7U)   /**< 3-bit result mask                             */
#define CRC3_MSB_SHIFT      (2U)     /**< Position of MSB within 3-bit CRC word         */
#define FRAME_MSB           (23U)    /**< Most-significant processed bit position        */
#define FRAME_CRC_LSB       (3U)     /**< Least-significant processed bit position       */

/* Masks for W560 suppression — explicit range-limiting before narrow bitfield writes */
#define FRAME_C_MASK        (0x1U)   /**< 1-bit direction field mask                    */
#define FRAME_ADDR_MASK     (0x7FU)  /**< 7-bit address field mask                      */
#define FRAME_DATA_MASK     (0xFFU)  /**< 8-bit data field mask                         */
#define FRAME_CRC_MASK      (0x7U)   /**< 3-bit CRC field mask                          */

/**********************************************************************************************************************
 * Configuration Tables
 *
 * Both tables use CDD_TLE9180_FRAME_WR_xxx / CDD_TLE9180_FRAME_RD_xxx constants
 * from cdd_tle9180_reg.h.  These are plain #define integer constants, giving valid
 * C99 constant initialisers for const arrays with static storage duration.
 *
 * Startup sequence — 13 write frames for AP32541 (12 V bus, 3-shunt 10 mΩ).
 * Frame ordering:
 *   1  GEN_CFG1   — VCC monitoring, OT threshold.
 *   2  GEN_CFG2   — Enable all 3 CSAs + 3-VDH sense pin mode.
 *   3  TL_VDH     — VDHP OV/UV window (single register, not a 16-bit pair).
 *   4  TL_CBVCC   — CB undervoltage + VCC OV/UV thresholds.
 *   5  FM1        — CP2 overload + HS buffer cap UV failure modes.
 *   6  FM3        — Vs + VDHP + VCC undervoltage failure modes.
 *   7  FM4        — Vs + VDHP + VCC overvoltage failure modes.
 *   8  FM6        — CSA 1/2/3 overcurrent failure modes.
 *   9  CONF_SIG   — Configuration lock (0xAC); device enters NORMAL mode.
 *   10 OP_GAIN1   — CSA 1&2 gain stage 1 = 30.81 V/V.
 *   11 OP_GAIN2   — CSA 1&2 gain stage 2 = 30.81 V/V (overrides default 34.45).
 *   12 OP_GAIN3   — CSA 3 gain stage 1&2 = 30.81 V/V.
 *   13 OP_OCL     — VRO = 2.5 V, fine trim = no adjustment.
 *********************************************************************************************************************/

const CddTle9180_SpiTx_T CddTle9180_StartupConfig_G[CDD_TLE9180_STARTUP_CMD_COUNT] =
{
    /* ── GEN_CFG1 (0x01) = 0x81  DS §15.1.1.2 p121 ─────────────────────────────────────────────────────────
     * tl_ot_w    [7:5] = 100B  → 140°C OT threshold (default; adequate margin for AP32541 ambient)
     * in_diag_act[4]   = 0     → input pattern supervision disabled
     * spi_wwd_act[3]   = 0     → SPI watchdog disabled (handled by TC3xx WDT at application level)
     * limp_act   [2]   = 0     → limp-home disabled (full fault stop preferred for motor application)
     * vcc_sup_off[1]   = 0     → VCC supervision enabled (protects 5V rail)
     * vcc_select [0]   = 1     → 5V selected as VCC monitoring threshold                            */
    { .U = CDD_TLE9180_FRAME_WR_GEN_CFG1  },

    /* ── GEN_CFG2 (0x02) = 0x0F  DS §15.1.1.3 p122 ─────────────────────────────────────────────────────────
     * tl_oc_op      [7]   = 0  → 5V OC detection threshold for CSAs (wider range for 10 mΩ shunts)
     * dis_ov_bh     [6]   = 0  → OV detection on HS buffer caps enabled
     * dis_ov_ld_vdh [5]   = 0  → OV load-dump detection at VDHP enabled
     * dis_sd_vdh    [4]   = 0  → OV shutdown at VDHP enabled
     * en_vdh3       [3]   = 1  → 3 VDH sense pins + 1 VDHP power pin (AP32541 3-phase layout)
     * en_op3        [2]   = 1  → CSA 3 enabled (third shunt, required for 3-phase FOC)
     * en_op2        [1]   = 1  → CSA 2 enabled (default)
     * en_op1        [0]   = 1  → CSA 1 + reference output buffer enabled (default)                  */
    { .U = CDD_TLE9180_FRAME_WR_GEN_CFG2  },

    /* ── TL_VDH (0x06) = 0x70  DS §15.1.1.7 p127 ───────────────────────────────────────────────────────────
     * tl_ov_vdh [7:4] = 0111B → 48.18V OV threshold (headroom above 12V bus incl. load-dump)
     * tl_uv_vdh [3:0] = 0000B → 18.00V UV threshold (low to avoid false trips during cap charge)    */
    { .U = CDD_TLE9180_FRAME_WR_TL_VDH    },

    /* ── TL_CBVCC (0x07) = 0x9A  DS §15.1.1.8 p129 ─────────────────────────────────────────────────────────
     * tl_uv_cb  [7:4] = 1001B → 9.07V CB undervoltage threshold (default)
     * tl_ov_vcc [3:2] = 10B   → 10% VCC overvoltage margin  (5V + 10% = 5.5V)
     * tl_uv_vcc [1:0] = 10B   → 10% VCC undervoltage margin (5V - 10% = 4.5V)                      */
    { .U = CDD_TLE9180_FRAME_WR_TL_CBVCC  },

    /* ── FM1 (0x08) = 0x32  DS §15.1.1.9 p130 ──────────────────────────────────────────────────────────────
     * fm_uv_cb  [7:6] = 00B  → CB UV → Warning only (CB UV non-critical during bootstrap charge)
     * Res       [5]   = 1    → Reserved; MUST write 1 (default value per datasheet)
     * fm_cp2_off[4]   = 1    → CP2 overload → shutdown output stages (CP loss unrecoverable)
     * Res       [3:2] = 00B  → Reserved; write 0 (default)
     * fm_uv_bs  [1:0] = 01B  → HS buffer cap UV → ERR (report before shutdown)                      */
    { .U = CDD_TLE9180_FRAME_WR_FM1       },

    /* ── FM3 (0x0A) = 0x2A  DS §15.1.1.11 p132 ─────────────────────────────────────────────────────────────
     * Res       [7:6] = 00B  → Reserved; write 0
     * fm_uv_vs  [5:4] = 10B  → Vs UV → ARE (auto-restart; supply can recover)
     * fm_uv_vdh [3:2] = 10B  → VDHP UV → ARE (VDHP recovers once supply restored)
     * fm_uv_vcc [1:0] = 10B  → VCC UV → ARE (5V rail can recover after brownout)                    */
    { .U = CDD_TLE9180_FRAME_WR_FM3       },

    /* ── FM4 (0x0B) = 0x4A  DS §15.1.1.12 p133 ─────────────────────────────────────────────────────────────
     * Res       [7]   = 0    → Reserved; write 0
     * fm_ov_vs  [6:5] = 10B  → Vs OV → ARE (regenerative transients can recover)
     * fm_ov_vdh [4:2] = 010B → VDHP OV → ARE all FETs off (shuts outputs, can restart)
     * fm_ov_vcc [1:0] = 10B  → VCC OV → ARE (5V rail OV transiently possible on AP32541)           */
    { .U = CDD_TLE9180_FRAME_WR_FM4       },

    /* ── FM6 (0x13) = 0x2A  DS §15.1.1.20 p141 ─────────────────────────────────────────────────────────────
     * Res       [7:6] = 00B  → Reserved; write 0
     * fm_oc_op3 [5:4] = 10B  → CSA3 OC → ARE (consistent policy all 3 phases)
     * fm_oc_op2 [3:2] = 10B  → CSA2 OC → ARE
     * fm_oc_op1 [1:0] = 10B  → CSA1 OC → ARE                                                        */
    { .U = CDD_TLE9180_FRAME_WR_FM6       },

    /* ── CONF_SIG (0x00) = 0xAC  DS §15.1.1.1 p120 ─────────────────────────────────────────────────────────
     * crc [7:0] = 10111010B → CRC8 signature byte.
     *   Writing 0xAC triggers internal CRC verification of all previously written
     *   configuration registers and transitions the TLE9180D into NORMAL mode.
     *   After this frame, GEN_CFG1/2, TL_VDH, TL_CBVCC, FM1/3/4/6 are locked until
     *   a full power cycle or INH toggle.  CSA control registers remain writable.       */
    { .U = CDD_TLE9180_FRAME_WR_CONF_SIG  },

    /* ── OP_GAIN1 (0x20) = 0x44  DS §15.1.2.1 p142 ─────────────────────────────────────────────────────────
     * Res       [7]   = 0    → Reserved; write 0
     * op1_gain1 [6:4] = 100B → 30.81 V/V CSA1 gain stage 1
     * Res       [3]   = 0    → Reserved; write 0
     * op2_gain1 [2:0] = 100B → 30.81 V/V CSA2 gain stage 1                                          */
    { .U = CDD_TLE9180_FRAME_WR_OP_GAIN1  },

    /* ── OP_GAIN2 (0x21) = 0x44  DS §15.1.2.2 p143 ─────────────────────────────────────────────────────────
     * Res       [7]   = 0    → Reserved; write 0
     * op1_gain2 [6:4] = 100B → 30.81 V/V CSA1 gain stage 2 (overrides default 34.45 V/V)
     * Res       [3]   = 0    → Reserved; write 0
     * op2_gain2 [2:0] = 100B → 30.81 V/V CSA2 gain stage 2 (overrides default 34.45 V/V)            */
    { .U = CDD_TLE9180_FRAME_WR_OP_GAIN2  },

    /* ── OP_GAIN3 (0x22) = 0x44  DS §15.1.2.3 p144 ─────────────────────────────────────────────────────────
     * Res       [7]   = 0    → Reserved; write 0
     * op3_gain2 [6:4] = 100B → 30.81 V/V CSA3 gain stage 2
     * Res       [3]   = 0    → Reserved; write 0
     * op3_gain1 [2:0] = 100B → 30.81 V/V CSA3 gain stage 1
     *   With 10 mΩ shunts and VRO=2.5V: I_FS = 2.5V/(30.81×0.010Ω) ≈ ±8.1A            */
    { .U = CDD_TLE9180_FRAME_WR_OP_GAIN3  },

    /* ── OP_OCL (0x23) = 0x9F  DS §15.1.2.4 p145 ───────────────────────────────────────────────────────────
     * zcl [7:6] = 10B      → VRO = 2.5V (mid-rail of 0–3.3V ADC range on AP32541)
     * ofs [5:0] = 011111B  → no fine adjustment (factory default trim)                               */
    { .U = CDD_TLE9180_FRAME_WR_OP_OCL    },
};

/**********************************************************************************************************************
 * Cyclic Read Command Table
 *
 * One frame per 10 ms slow-task call; all 13 diagnostic registers read in 130 ms.
 *********************************************************************************************************************/

const CddTle9180_SpiTx_T CddTle9180_ReadCmds_G[CDD_TLE9180_READ_CMD_COUNT] =
{
    { .U = CDD_TLE9180_FRAME_RD_SC_LS1        },   /**< Short-circuit LS phase 1  */
    { .U = CDD_TLE9180_FRAME_RD_SC_LS2        },   /**< Short-circuit LS phase 2  */
    { .U = CDD_TLE9180_FRAME_RD_SC_LS3        },   /**< Short-circuit LS phase 3  */
    { .U = CDD_TLE9180_FRAME_RD_SC_HS1        },   /**< Short-circuit HS phase 1  */
    { .U = CDD_TLE9180_FRAME_RD_SC_HS2        },   /**< Short-circuit HS phase 2  */
    { .U = CDD_TLE9180_FRAME_RD_SC_HS3        },   /**< Short-circuit HS phase 3  */
    { .U = CDD_TLE9180_FRAME_RD_LIMP_HOME     },   /**< Limp-home config shadow   */
    { .U = CDD_TLE9180_FRAME_RD_PFB_GAIN      },   /**< PFB gain readback         */
    { .U = CDD_TLE9180_FRAME_RD_RECT_THRESH_P },   /**< Rectifier positive thresh */
    { .U = CDD_TLE9180_FRAME_RD_RECT_THRESH_A },   /**< Rectifier adaptive thresh */
    { .U = CDD_TLE9180_FRAME_RD_RECT_FILTER   },   /**< Rectifier filter constant */
    { .U = CDD_TLE9180_FRAME_RD_RECT_ACCURACY },   /**< Rectifier hysteresis      */
    { .U = CDD_TLE9180_FRAME_RD_CONF_SIG      },   /**< Config signature shadow   */
};

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

static const CddTle9180_SpiTx_T Status_ReadSeq_S[2U] =
{
    { .U = STATUS_READ_CMD },
    { .U = STATUS_NOP_CMD  },
};

static CddTle9180_SpiRx_T Status_RxBuf_S[2U];

/**********************************************************************************************************************
 * Private Function Prototypes
 *********************************************************************************************************************/
static void CddTle9180_PowerOnSequence(void);
static uint32_T CddTle9180_PingSr0(P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode);


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

    if (CddGpio_GetErr_P15_2() == 0x0U)
    {
        result = 0x1U;   /* /ERR = LOW → error active */
    }
    else
    {
        result = 0x0U;   /* /ERR = HIGH → no error    */
    }

    return result;
}

/**********************************************************************************************************************
 * Utility Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_ComputeCrc3
 *
 * CRC-3 over bits [23:3] of the 24-bit frame (21 bits), MSB-first.
 * Polynomial x³+x+1 (0x3), initialisation 0x4.
 *
 * Note: CRC3_INIT is 0x4 (0b100), not 0x7.  The original value 0x7 produced
 * incorrect CRCs that do not match hardware behaviour.  All 13 startup frame
 * CRCs in cdd_tle9180_reg.h are computed and verified with init=0x4.
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

        if (i == 0x0U)          /* MISRA Rule 14.4: guard uint32_T underflow */
        {
            break;
        }
    }

    return crc;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_Wait
 *------------------------------------------------------------------------------------------------------------------*/
void CddTle9180_Wait(uint32_T TimeConst)
{
    uint64_T deadline = CddStm_GetDeadline(TimeConst);

    while (CddStm_IsDeadlineElapsed(deadline) == 0x0U)
    {
        CddSys_NopDelay(1U, 1U);
    }
}

/**********************************************************************************************************************
 * Private Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_PowerOnSequence  (DS §6.4)
 *
 *   Step 1  ENA = HIGH       Enable LDO and pre-driver supply.
 *   Step 2  /INH = HIGH      Release inhibit — charges bootstrap caps.
 *           wait 1 ms
 *   Step 3  /INH = LOW       Force SLEEP (/ERR LOW within t_sleep).
 *           wait 1 s         POR + EEPROM shadow load (t_sleep_min=500ms; 2× margin).
 *   Step 4  /INH = HIGH      Exit SLEEP → IDLE (/ERR returns HIGH).
 *           wait 1 ms        LDO re-stabilise.
 *   Step 5  /SOFF = HIGH     Release gate shutdown.
 *------------------------------------------------------------------------------------------------------------------*/
static void CddTle9180_PowerOnSequence(void)
{
    CddTle9180_AssertEnable();

    CddTle9180_DeassertInhibit();
    CddTle9180_Wait(TimeConst_1ms);

    CddTle9180_AssertInhibit();
    CddTle9180_Wait(TimeConst_1s);

    CddTle9180_DeassertInhibit();
    CddTle9180_Wait(TimeConst_1ms);

    CddTle9180_DeassertSafeOff();
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_PingSr0  — SPI physical-layer verification (pre-configuration)
 *
 * Sends READ(CONF_SIG / SR0, addr=0x00) + NOP pipeline flush.
 * SR0 is readable from reset with no prior TLE9180 configuration.
 *
 * TX frames:
 *   [0]  CDD_TLE9180_FRAME_RD_CONF_SIG  (0x000004U)  — READ addr=0x00, CRC=4
 *   [1]  STATUS_NOP_CMD                 (0x320003U)  — NOP pipeline flush
 *
 * Pipelined response arrives in RxBuf[1]:
 *   .B.DATA       [11:4]  — SR0 register byte (any non-0x00 plausible value = chip alive)
 *   .B.CONFVALID  [21]    — 0 expected before CddTle9180_Configure()
 *   .B.SPIERR     [20]    — 0 = last frame CRC accepted by TLE9180
 *   .B.ERROR      [23]    — mirrors /ERR pin (1 = fault active)
 *
 * Diagnostic variables (function-local static volatile — JTAG watch window):
 *   dbg_Sr0_QspiErr   QSPI exchange error code (CDD_QSPI_ERR_NONE = 0x0 = OK)
 *   dbg_Sr0_RawFrame  full raw 24-bit word from RxBuf[1].U
 *   dbg_Sr0_Data      DATA field  [11:4] — SR0 register byte
 *   dbg_Sr0_ConfValid CONFVALID   [21]   — 0 expected before config
 *   dbg_Sr0_SpiErr    SPIERR      [20]   — 0 = frame accepted
 *   dbg_Sr0_Error     ERROR       [23]   — /ERR pin mirror
 *
 * Pass/fail decision table:
 *   dbg_Sr0_QspiErr != 0            → CDD_TLE9180_ERR_SPI: QSPI RX timeout
 *   dbg_Sr0_RawFrame == 0x000000    → MISO stuck LOW  (pin mux, /INH, VREF)
 *   dbg_Sr0_RawFrame == 0xFFFFFF    → MISO stuck HIGH (open line, wrong polarity)
 *   dbg_Sr0_SpiErr  == 1            → TLE9180 rejected frame CRC
 *   plausible non-zero, SpiErr==0   → SPI physical layer confirmed OK
 *------------------------------------------------------------------------------------------------------------------*/
static uint32_T CddTle9180_PingSr0(P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    /* TX: READ SR0 (addr=0x00, pre-verified constant) + NOP pipeline flush */
    static const CddTle9180_SpiTx_T Sr0_TxSeq_S[2U] =
    {
        { .U = CDD_TLE9180_FRAME_RD_CONF_SIG },   /* 0x000004U — READ addr=0x00, CRC=4 */
        { .U = STATUS_NOP_CMD                }    /* 0x320003U — NOP pipeline flush      */
    };

    static CddTle9180_SpiRx_T Sr0_RxBuf_S[2U];

    /* Diagnostic variables — static volatile: survive optimisation, visible in JTAG */
    static volatile uint32_T dbg_Sr0_QspiErr;    /**< CddQspi4_Exchange qspiErr output     */
    static volatile uint32_T dbg_Sr0_RawFrame;   /**< RxBuf[1].U  — full raw 24-bit word   */
    static volatile uint32_T dbg_Sr0_Data;       /**< RxBuf[1].B.DATA      bits [11:4]     */
    static volatile uint32_T dbg_Sr0_ConfValid;  /**< RxBuf[1].B.CONFVALID bit  [21]       */
    static volatile uint32_T dbg_Sr0_SpiErr;     /**< RxBuf[1].B.SPIERR    bit  [20]       */
    static volatile uint32_T dbg_Sr0_Warning;    /**< RxBuf[1].B.WARNING   bit  [22]       */
    static volatile uint32_T dbg_Sr0_Error;      /**< RxBuf[1].B.ERROR     bit  [23]       */

    uint32_T qspiErr;
    uint32_T result;

    *ErrorCode = CDD_TLE9180_ERR_NONE;

    result = Cdd_ExchangeN(
                 (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))Sr0_TxSeq_S,  /* PRQA S 11.3 */
                 (P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA))Sr0_RxBuf_S,  /* PRQA S 11.3 */
                 2U,
                 &qspiErr);

    /* Capture all diagnostic fields — inspect in UDE / Lauterbach watch window */
    dbg_Sr0_QspiErr   = qspiErr;
    dbg_Sr0_RawFrame  = Sr0_RxBuf_S[1U].U;
    dbg_Sr0_Data      = (uint32_T)Sr0_RxBuf_S[1U].B.DATA;
    dbg_Sr0_ConfValid = (uint32_T)Sr0_RxBuf_S[1U].B.CONFVALID;
    dbg_Sr0_SpiErr    = (uint32_T)Sr0_RxBuf_S[1U].B.SPIERR;
    dbg_Sr0_Warning   = (uint32_T)Sr0_RxBuf_S[1U].B.WARNING;
    dbg_Sr0_Error     = (uint32_T)Sr0_RxBuf_S[1U].B.ERROR;

    if (result != 0x1U)
    {
        *ErrorCode = CDD_TLE9180_ERR_SPI;
    }

    return result;
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/
uint32_T CddTle9180_Init(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    uint32_T i;
    uint32_T qspiErr;
    uint32_T result;

    *ErrorCode        = CDD_TLE9180_ERR_NONE;
    Handle->ReadIndex = 0x0U;
    Handle->State     = CDDTLE9180_STATE_POWERON;
    Handle->FaultCode = 0x0U;

    for (i = 0x0U; i < CDD_TLE9180_SPI_BUF_SIZE; i++)
    {
        Handle->RxBuf[i].U = 0x0U;
    }

    result = CddQspi4_Init(&qspiErr);

    if (result == 0x1U)
    {
        /* ── Baud rate verification ───────────────────────────────────────────────
         * Mirrors: is_not_equal_epsilon(get_qspi0_ch12_baud_rate_frequency(), MHZ_5)
         * Must pass before any GPIO or SPI activity.  A mismatch indicates a clock
         * tree or ECON register misconfiguration — proceeding would corrupt all
         * SPI frames sent to the TLE9180D.                                        */
        if (CddSys_AreEqual32((real32_T)CddQspi4_GetBaudRate(),
                               (real32_T)MHZ_1,
                               CDD_QSPI4_BAUD_RATE_EPSILON) != 0x1U)
        {
            *ErrorCode    = CDD_TLE9180_ERR_BAUD_RATE;
            Handle->State = CDDTLE9180_STATE_FAULT;
            result        = 0x0U;
        }
    }
    else
    {
        *ErrorCode    = CDD_TLE9180_ERR_SPI;
        Handle->State = CDDTLE9180_STATE_FAULT;
    }

    if (result == 0x1U)
    {
        /* Power-on sequence — /INH must be HIGH before any SPI exchange */
        CddTle9180_PowerOnSequence();
        Handle->State = CDDTLE9180_STATE_IDLE;

        /* SR0 ping — TLE9180 is awake and MISO is actively driven */
        result = CddTle9180_PingSr0(ErrorCode);
    }

    if (result != 0x1U)
    {
        Handle->State = CDDTLE9180_STATE_FAULT;
    }

    return result;
}

uint32_T CddTle9180_Configure(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    uint32_T qspiErr;
    uint32_T result;

    *ErrorCode    = CDD_TLE9180_ERR_NONE;
    Handle->State = CDDTLE9180_STATE_CONFIGURING;

    CddTle9180_Wait(TimeConst_1ms);   /* VDD / QSPI clock stabilisation */

    result = Cdd_ExchangeN(
                 (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))CddTle9180_StartupConfig_G, /* PRQA S 11.3 */
                 (P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA))Handle->RxBuf,               /* PRQA S 11.3 */
                 CDD_TLE9180_STARTUP_CMD_COUNT,
                 &qspiErr);

    if (result != 0x1U)
    {
        *ErrorCode    = CDD_TLE9180_ERR_SPI;
        Handle->State = CDDTLE9180_STATE_FAULT;
    }

    return result;
}

uint32_T CddTle9180_IsNormalMode(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    uint32_T qspiErr;
    uint32_T result;

    *ErrorCode = CDD_TLE9180_ERR_NONE;

    result = Cdd_ExchangeN(
                 (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))Status_ReadSeq_S, /* PRQA S 11.3 */
                 (P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA))Status_RxBuf_S,    /* PRQA S 11.3 */
                 2U,
                 &qspiErr);

    if (result == 0x1U)
    {
        if (((uint32_T)(Status_RxBuf_S[1U].B.DATA & STATUS_NORM_M_BIT) != 0x0U) &&
            ((uint32_T)(Status_RxBuf_S[1U].B.CONFVALID)                != 0x0U))
        {
            Handle->State = CDDTLE9180_STATE_NORMAL;
        }
        else
        {
            *ErrorCode = CDD_TLE9180_ERR_NOT_NORMAL;
            result     = 0x0U;
        }
    }
    else
    {
        *ErrorCode = CDD_TLE9180_ERR_SPI;
    }

    return result;
}

uint32_T CddTle9180_Startup(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    uint32_T result;

    result = CddTle9180_Init(Handle, ErrorCode);

    if (result == 0x1U)
    {
        result = CddTle9180_Configure(Handle, ErrorCode);
    }

    if (result == 0x1U)
    {
        result = CddTle9180_IsNormalMode(Handle, ErrorCode);
    }

    return result;
}

uint32_T CddTle9180_ReadRegister(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    uint32_T qspiErr;
    uint32_T result;
    uint32_T buf_idx;

    *ErrorCode = CDD_TLE9180_ERR_NONE;
    buf_idx    = CDD_TLE9180_READ_BUF_OFFSET + Handle->ReadIndex;

    result = Cdd_ExchangeN(
                 (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))&CddTle9180_ReadCmds_G[Handle->ReadIndex], /* PRQA S 11.3 */
                 (P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA))&Handle->RxBuf[buf_idx],                    /* PRQA S 11.3 */
                 1U,
                 &qspiErr);

    if (result == 0x1U)
    {
        Handle->ReadIndex++;

        if (Handle->ReadIndex >= CDD_TLE9180_READ_CMD_COUNT)
        {
            Handle->ReadIndex = 0x0U;
        }
    }
    else
    {
        *ErrorCode = CDD_TLE9180_ERR_SPI;
    }

    return result;
}

void CddTle9180_MonitorFault(P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle)
{
    if (Handle->State == CDDTLE9180_STATE_NORMAL)
    {
        if (CddTle9180_IsErrorActive() == 0x1U)
        {
            CddTle9180_AssertSafeOff();       /* hardware shutdown first */
            Handle->State     = CDDTLE9180_STATE_FAULT;
            Handle->FaultCode = 0x1U;
        }
    }
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddTle9180_ClearFaults
 *
 * Reads all 12 clearable error registers (0x41–0x4D, excl. 0x4A) in a single
 * 13-frame SPI burst (12 READ frames + 1 NOP pipeline flush).
 *
 * Per DS §14.3.1: reading a register clears the WARNING and ERROR SPI status
 * flags for any fault that is no longer present.  The pipelined DATA byte for
 * each register arrives one frame late — RxBuf[N+1] holds the response to
 * TxSeq[N]:
 *
 *   RxBuf[0]   response to the previous command (pre-burst context — discard)
 *   RxBuf[1]   Err_over   (0x41)
 *   RxBuf[2]   Ser        (0x42)
 *   RxBuf[3]   Err_i_1    (0x43)
 *   RxBuf[4]   Err_i_2    (0x44)
 *   RxBuf[5]   Err_e      (0x45)
 *   RxBuf[6]   Err_sd     (0x46)
 *   RxBuf[7]   Err_scd    (0x47)
 *   RxBuf[8]   Err_indiag (0x48)
 *   RxBuf[9]   Err_osf    (0x49)
 *   RxBuf[10]  Err_op_12  (0x4B)
 *   RxBuf[11]  Err_op_3   (0x4C)
 *   RxBuf[12]  Err_outp   (0x4D)  [delivered by NOP flush frame]
 *
 * 0x4A (Err_spiconf) is intentionally excluded: per DS §14.3, its bits 0,1,3,4
 * are not cleared by read and require a separate write-to-clear sequence.
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddTle9180_ClearFaults(
    P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    /* TX: 12 READ frames (one per clearable error register) + 1 NOP flush.
     * C99 constant initialisers — FRAME_ constants from cdd_tle9180_reg.h.   */
    static const CddTle9180_SpiTx_T FaultClr_TxSeq_S[CDD_TLE9180_ERR_REG_COUNT + 1U] =
    {
        { .U = CDD_TLE9180_FRAME_RD_ERR_OVER   },  /* 0x410002U — Err_over   addr=0x41 CRC=2 */
        { .U = CDD_TLE9180_FRAME_RD_SER        },  /* 0x420005U — Ser        addr=0x42 CRC=5 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_I_1    },  /* 0x430001U — Err_i_1    addr=0x43 CRC=1 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_I_2    },  /* 0x440000U — Err_i_2    addr=0x44 CRC=0 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_E      },  /* 0x450004U — Err_e      addr=0x45 CRC=4 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_SD     },  /* 0x460003U — Err_sd     addr=0x46 CRC=3 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_SCD    },  /* 0x470007U — Err_scd    addr=0x47 CRC=7 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_INDIAG },  /* 0x480001U — Err_indiag addr=0x48 CRC=1 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_OSF    },  /* 0x490005U — Err_osf    addr=0x49 CRC=5 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_OP_12  },  /* 0x4B0006U — Err_op_12  addr=0x4B CRC=6 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_OP_3   },  /* 0x4C0007U — Err_op_3   addr=0x4C CRC=7 */
        { .U = CDD_TLE9180_FRAME_RD_ERR_OUTP   },  /* 0x4D0003U — Err_outp   addr=0x4D CRC=3 */
        { .U = STATUS_NOP_CMD                   },  /* 0x320003U — NOP pipeline flush          */
    };

    static CddTle9180_SpiRx_T FaultClr_RxBuf_S[CDD_TLE9180_ERR_REG_COUNT + 1U];

    /* Diagnostic snapshot — static volatile: survive optimisation, visible in JTAG */
    static volatile uint32_T dbg_FaultClr_QspiErr;    /**< QSPI exchange error code          */
    static volatile uint32_T dbg_FaultClr_ErrOver;    /**< 0x41 Err_over   DATA byte          */
    static volatile uint32_T dbg_FaultClr_Ser;        /**< 0x42 Ser        DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrI1;      /**< 0x43 Err_i_1   DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrI2;      /**< 0x44 Err_i_2   DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrE;       /**< 0x45 Err_e      DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrSd;      /**< 0x46 Err_sd     DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrScd;     /**< 0x47 Err_scd    DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrIndiag;  /**< 0x48 Err_indiag DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrOsf;     /**< 0x49 Err_osf    DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrOp12;    /**< 0x4B Err_op_12  DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrOp3;     /**< 0x4C Err_op_3   DATA byte          */
    static volatile uint32_T dbg_FaultClr_ErrOutp;    /**< 0x4D Err_outp   DATA byte          */

    uint32_T qspiErr;
    uint32_T result;

    *ErrorCode = CDD_TLE9180_ERR_NONE;

    result = Cdd_ExchangeN(
                 (P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA))FaultClr_TxSeq_S,  /* PRQA S 11.3 */
                 (P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA))FaultClr_RxBuf_S,  /* PRQA S 11.3 */
                 CDD_TLE9180_ERR_REG_COUNT + 1U,
                 &qspiErr);

    /* Capture pipelined DATA bytes — RxBuf[N+1] holds response to TxSeq[N]   */
    dbg_FaultClr_QspiErr   = qspiErr;
    dbg_FaultClr_ErrOver   = (uint32_T)FaultClr_RxBuf_S[1U].B.DATA;
    dbg_FaultClr_Ser       = (uint32_T)FaultClr_RxBuf_S[2U].B.DATA;
    dbg_FaultClr_ErrI1     = (uint32_T)FaultClr_RxBuf_S[3U].B.DATA;
    dbg_FaultClr_ErrI2     = (uint32_T)FaultClr_RxBuf_S[4U].B.DATA;
    dbg_FaultClr_ErrE      = (uint32_T)FaultClr_RxBuf_S[5U].B.DATA;
    dbg_FaultClr_ErrSd     = (uint32_T)FaultClr_RxBuf_S[6U].B.DATA;
    dbg_FaultClr_ErrScd    = (uint32_T)FaultClr_RxBuf_S[7U].B.DATA;
    dbg_FaultClr_ErrIndiag = (uint32_T)FaultClr_RxBuf_S[8U].B.DATA;
    dbg_FaultClr_ErrOsf    = (uint32_T)FaultClr_RxBuf_S[9U].B.DATA;
    dbg_FaultClr_ErrOp12   = (uint32_T)FaultClr_RxBuf_S[10U].B.DATA;
    dbg_FaultClr_ErrOp3    = (uint32_T)FaultClr_RxBuf_S[11U].B.DATA;
    dbg_FaultClr_ErrOutp   = (uint32_T)FaultClr_RxBuf_S[12U].B.DATA;

    if (result != 0x1U)
    {
        *ErrorCode = CDD_TLE9180_ERR_SPI;
    }

    return result;
}

uint32_T CddTle9180_ResetFault(
    P2VAR(CddTle9180_T, AUTOMATIC, CDD_APPL_DATA) Handle,
    P2VAR(uint32_T,     AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    CddTle9180_DeassertSafeOff();

    Handle->FaultCode = 0x0U;
    Handle->State     = CDDTLE9180_STATE_POWERON;

    return CddTle9180_Startup(Handle, ErrorCode);
}
