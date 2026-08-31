/**********************************************************************************************************************
 * \file        cdd_qspi_app.c
 * \brief       QSPI4 bare-metal driver — hardware init and ISR-driven exchange.
 *
 * \details     Direct register access via ifxQspi_reg.h / ifxSrc_reg.h.
 *              No iLLD driver layer, no DMA.
 *
 *              Exchange model
 *              ──────────────
 *              CddQspi4_Exchange() is a blocking foreground call.  For each
 *              24-bit frame it:
 *                1. Loads BACONENTRY then DATAENTRY0 (mandatory TC3xx write order).
 *                2. Spin-waits on Qspi4State_S.received, which is set by the
 *                   RX ISR (qspi4_rx_handler, SRPN 56, CPU0).
 *                3. On success copies RXEXIT (captured in the ISR) into RxBuf[i].
 *                4. On RX timeout sets CDD_QSPI_ERR_RX_TIMEOUT and returns 0x0U.
 *
 *              Three ISRs are registered:
 *                qspi4_tx_handler  (SRPN 55) — clears TX flag; sets transmitted.
 *                qspi4_rx_handler  (SRPN 56) — pops RXEXIT; sets received.
 *                qspi4_err_handler (SRPN 57) — clears error flags; sets error AND
 *                                              received so the spin-wait is always
 *                                              released (no hang on error frame).
 *
 *              State guard
 *              ───────────
 *              All four flags in Qspi4State_S must be false before a frame is
 *              issued.  If the bus is not idle when CddQspi4_Exchange() is entered
 *              the function returns 0x0U immediately with CDD_QSPI_ERR_RX_TIMEOUT.
 *              This makes re-entrancy safe (one caller at a time).
 *
 *              Baud rate readback — CddQspi4_GetBaudRate()
 *              ────────────────────────────────────────────
 *              Reads QSPI4_GLOBALCON.TQ and QSPI4_ECON3.{Q,A,B,C} and computes:
 *                f_baud = f_periph / ((TQ+1) × (Q+A+B+C))
 *              where f_periph = 200 000 000 Hz (fixed on TC3xx).
 *              Called by CddTle9180_Init() via CddSys_AreEqual32() to verify
 *              the baud rate before any GPIO or SPI activity — replacing the old
 *              is_not_equal_epsilon(get_qspi0_ch12_baud_rate_frequency(), MHZ_5).
 *
 * \note        MISRA C:2012: Rules 8.9, 10.1, 13.3, 14.4, 15.4, 15.5.
 *              PRQA S 0303 — hardware-register union access via .U / .B members.
 *              PRQA S 0750 — Ifx_QSPI_BACON union .U and .B accessed within same
 *                            function scope.
 *              Rule 13.3 deviation in CddQspi4_CalcTq(): float epsilon comparison
 *              used as early-exit predicate; annotated inline.
 *
 * \copyright   Copyright (C) EmbedSim Project / Paul Abraham 2024
 *              SPDX-License-Identifier: MIT
 *********************************************************************************************************************/

#include "cdd_qspi_app.h"
#include "cdd_sys_utility.h"
#include "cdd_gpio_app.h"
#include "ifxQspi_reg.h"
#include "ifxSrc_reg.h"

/**********************************************************************************************************************
 * Private Types
 *********************************************************************************************************************/

/**
 * \brief  Per-frame ISR handshake flags.
 *
 * \note   All members are uint32_T (not boolean_T) so that ISR writes are
 *         naturally atomic on TriCore (32-bit aligned store).  volatile forces
 *         the compiler to re-read each flag on every loop iteration in the
 *         foreground spin-wait.
 */
typedef struct
{
    volatile uint32_T transition;    /**< Set when any ISR fires; cleared after each frame  */
    volatile uint32_T transmitted;   /**< Set by TX ISR; cleared after each frame            */
    volatile uint32_T received;      /**< Set by RX or ERR ISR; spin-wait exit condition     */
    volatile uint32_T error;         /**< Set by ERR ISR; checked after spin-wait            */
} CddQspi4_State_T;

/**********************************************************************************************************************
 * Private Constants
 *********************************************************************************************************************/

/* No hardcoded fPeriph constant — actual peripheral clock read via CddSys_GetQspiFreq()
 * in both CddQspi4_CalcTq() and CddQspi4_GetBaudRate().                                */

/**********************************************************************************************************************
 * Private Variables
 *********************************************************************************************************************/

/**
 * \brief  ISR handshake state — one instance for QSPI4 (single-master, single-slave).
 *
 * \note   Rule 8.9: file-scope static; no external linkage required.
 *         Initialised to all-zero (all false) at startup by C runtime.
 */
static CddQspi4_State_T Qspi4State_S;

/**
 * \brief  Receive word captured inside qspi4_rx_handler() from QSPI4_RXEXIT.
 *
 * \note   Written only by the RX ISR; read only by CddQspi4_Exchange() after
 *         the spin-wait exits.  volatile prevents load-hoisting by the compiler.
 */
static volatile uint32_T Qspi4RxWord_S;

/**********************************************************************************************************************
 * ISR Declarations
 *
 * EMBED_SIM_INTERRUPT(name, vectab, srpn) expands to:
 *   void __interrupt(srpn) __vector_table(vectab) name(void)
 *
 * All three ISRs route to CPU0 (TOS=0).  SRPNs from cdd_config.h.
 *********************************************************************************************************************/

EMBED_SIM_INTERRUPT(qspi4_tx_handler,  0, CORE_00_QSPI4_TX_SRPN);
EMBED_SIM_INTERRUPT(qspi4_rx_handler,  0, CORE_00_QSPI4_RX_SRPN);
EMBED_SIM_INTERRUPT(qspi4_err_handler, 0, CORE_00_QSPI4_ERR_SRPN);

/**********************************************************************************************************************
 * Private Function Implementations
 *********************************************************************************************************************/


/*******************************************************************************
 * \brief   Compute ECON register fields Q, A, B, C for a target baud rate.
 *
 * \details Implements the iLLD algorithm from
 *          IfxQspi_calculateExtendedConfigurationValue() verbatim, but
 *          isolated, named, typed, and annotated for MISRA C:2012 / ASIL-D.
 *
 *          Two bugs present in the original EmbedSim rewrite are corrected:
 *            1. Equal-error update guard: iLLD updates when (error <= bestError)
 *               AND bestAbc (pre-update) is even.  The prior rewrite incorrectly
 *               required curAbc to be even AND bestAbc to be odd.
 *            2. Early-exit done flag: iLLD re-evaluates bestAbc parity AFTER
 *               storing the new best, then tests error==0.  The prior rewrite
 *               used the pre-update bestEven, so done=1 was never reached on
 *               the first exact even-abc candidate (e.g. abc=8 at 1 MHz).
 *
 *          The SCK period is factored as:
 *              T_SCK = T_TQ * Q * (A+1 + B + C)
 *          where T_TQ = 1 / f_TQ (time-quanta period).
 *
 *          ECON encoding:
 *              ECON.Q  = Q - 1        (6-bit, 1..64 → 0..63)
 *              ECON.A  = (ABC/2 + ABC%2) - 1
 *              ECON.B  = min(halfBaud, maxB)    maxB = 3
 *              ECON.C  = max(halfBaud - maxB, 0)
 *
 * \param[in]  fBaud        Desired SCK frequency [Hz]  — mirrors iLLD fBaud = chConfig->baudrate.
 *                        tQspi = (TQ+1) / fQspi is derived internally from hardware,
 *                        matching IfxQspi_getTimeQuantaFrequency(qspi).
 * \param[out] OutQ       ECON.Q field value  (Q − 1)
 * \param[out] OutA       ECON.A field value  (A+1 = ABC/2, so A = ABC/2 − 1)
 * \param[out] OutB       ECON.B field value
 * \param[out] OutC       ECON.C field value
 *
 * \return  Achieved SCK frequency [Hz] for the caller to verify against epsilon.
 ******************************************************************************/
static real32_T Cdd_Qspi4_Calc_Econ(
    real32_T  fBaud,
    uint32_T *OutQ,
    uint32_T *OutA,
    uint32_T *OutB,
    uint32_T *OutC)
{
    static const int32_T KMaxB   =  3;
    static const int32_T KAbcMin =  2;
    static const int32_T KAbcMax =  8;
    static const int32_T KQMax   = 64;
    static const int32_T KQMin   =  1;

    real32_T fTarget;
    real32_T tTq;       /* = (TQ+1) / fQspi  ≡  iLLD tQspi                          */
    real32_T tBaud;
    real32_T tTmp;
    real32_T tBaudTmp;
    real32_T error;
    real32_T bestError;
    real32_T abcTotal;
    int32_T  abc;
    int32_T  q;
    int32_T  swapTmp;
    int32_T  bestAbc;
    int32_T  bestQ;
    int32_T  halfBaud;
    int32_T  diffB;
    int32_T  econQ;
    int32_T  econA;
    int32_T  econB;
    int32_T  econC;
    uint32_T leq;       /* error <= bestError                                        */
    uint32_T neq;       /* error != bestError (strictly better)                       */
    uint32_T bestEven;  /* bestAbc is even AFTER any update in this iteration         */
    uint32_T abcOdd;
    uint32_T qEven;
    uint32_T done;

    /* Derive tQspi from hardware — mirrors IfxQspi_getTimeQuantaFrequency(qspi):
     *   f_TQ = fQspi / (TQ+1)  →  tQspi = (TQ+1) / fQspi                     */
    fTarget   = (fBaud > 0.0F) ? fBaud : 1.0F;
    tTq       = (real32_T)(QSPI4_GLOBALCON.B.TQ + 1U)                /* PRQA S 0303 */
                / CddSys_GetQspiFreq();
    tBaud     = 1.0F / fTarget;
    bestError = 1.0e6F;
    bestAbc   = KAbcMax;
    bestQ     = KQMin;
    done      = 0U;

    for (abc = KAbcMax; (abc >= KAbcMin) && (done == 0U); abc--)
    {
        tTmp = tTq * (real32_T)abc;
        q    = (int32_T)((tBaud / tTmp) + 0.5F);

        if (q > KQMax)
        {
            q = KQMax;
        }
        else if ((q * abc) < 4)
        {
            q = 2;
        }
        else if (q < KQMin)
        {
            q = KQMin;
        }
        else
        {
            /* q already in range — MISRA 15.7 */
        }

        tBaudTmp = tTmp * (real32_T)q;
        error    = (tBaudTmp >= tBaud)
                   ? (tBaudTmp - tBaud)
                   : (tBaud   - tBaudTmp);

        leq = (error <= bestError) ? 1U : 0U;           /* __leqf equivalent        */
        neq = (error != bestError) ? 1U : 0U;           /* __neqf equivalent        */ /* NOLINT Rule 13.3 */

        if (leq == 1U)
        {
            /* iLLD inner guard: update if strictly better OR bestAbc is even.
             * NOTE: uses bestAbc parity, not current abc — must be evaluated
             * before the update so the pre-update bestAbc drives the decision,
             * matching IfxQspi_calculateExtendedConfigurationValue exactly.   */
            bestEven = (((uint32_T)bestAbc & 1U) == 0U) ? 1U : 0U;

            if ((neq == 1U) || (bestEven == 1U))
            {
                bestError = error;
                bestAbc   = abc;
                bestQ     = q;
            }

            /* Early-exit: re-read bestAbc parity AFTER the update, then check
             * error == 0.  Mirrors the iLLD break-on-exact-even-abc logic.    */
            bestEven = (((uint32_T)bestAbc & 1U) == 0U) ? 1U : 0U;

            if (bestEven == 1U)
            {
                if (error == 0.0F)  /* NOLINT Rule 13.3 — exact float zero from integer arithmetic */ /* PRQA S 3189 */
                {
                    done = 1U;
                }
            }
        }
    }

    abcOdd = (((uint32_T)bestAbc & 1U) != 0U) ? 1U : 0U;
    qEven  = (((uint32_T)bestQ   & 1U) == 0U) ? 1U : 0U;

    if ((bestQ <= KAbcMax) && (abcOdd == 1U) && (qEven == 1U))
    {
        swapTmp = bestQ;
        bestQ   = bestAbc;
        bestAbc = swapTmp;
    }

    halfBaud = bestAbc / 2;
    diffB    = halfBaud - KMaxB;

    econQ    = bestQ - 1;
    econA    = (halfBaud + (bestAbc % 2)) - 1;
    econB    = (diffB > 0) ? KMaxB    : halfBaud;
    econC    = (diffB > 0) ? diffB    : 0;

    *OutQ = (uint32_T)econQ;
    *OutA = (uint32_T)econA;
    *OutB = (uint32_T)econB;
    *OutC = (uint32_T)econC;

    abcTotal = (real32_T)(econA + 1)
             + (real32_T)econB
             + (real32_T)econC;

    return 1.0F / (tTq * (real32_T)(econQ + 1) * abcTotal);
}


/*--------------------------------------------------------------------------------------------------------------------
 * CddQspi4_CalcTq
 *
 * Computes the GLOBALCON.TQ prescaler value that minimises baud-rate error
 * for a given target frequency.
 *
 * Algorithm adapted from IfxQspi_calculateTimeQuantumLength() (Infineon iLLD v1.x).
 * All iLLD intrinsics replaced with ISO C99 equivalents.
 * MISRA C:2012 deviations:
 *   Rule 13.3  — float epsilon comparison used as early-exit (see inline comment).
 *   Rule 15.4  — single break inside the search loop (early exit on exact match).
 *------------------------------------------------------------------------------------------------------------------*/
/**
 * \brief   Calculates the QSPI GLOBALCON.TQ prescaler for a target baud rate.
 *
 * \param[in]  MaxBaudrate   Target QSPI clock frequency [Hz], must be > 0.
 * \return     GLOBALCON.TQ field value [0 … (QSPI4_GLOBALCON_TQ_ABCQ_MAX − 1)].
 */
static uint32_T CddQspi4_CalcTq(real32_T MaxBaudrate)
{
    static const uint32_T abcqMin      = 4U;
    static const uint32_T abcqMax      = 504U;
    static const real64_T deltaExactHz = 0.5;

    uint32_T  abcq;
    uint32_T  bestTq;
    real64_T  fQspi;
    real64_T  realTq;
    real64_T  tq;
    real64_T  achievedMax;
    real64_T  deltaMax;
    real64_T  bestDelta;
    boolean_T exactMatch;

    fQspi = CddSys_GetQspiFreq();   /* actual fPeriph — USM2 §31 */

    realTq    = fQspi / (4.0 * (real64_T)MaxBaudrate);
    bestTq    = (uint32_T)(realTq + 0.5);
    if (bestTq < 1U)
    {
        bestTq = 1U;
    }
    bestDelta  = fQspi / (real64_T)bestTq;
    bestDelta  = (bestDelta > (real64_T)MaxBaudrate) ?
                 (bestDelta - (real64_T)MaxBaudrate) :
                 ((real64_T)MaxBaudrate - bestDelta);
    exactMatch = (boolean_T)0U;

    for (abcq = abcqMin;
         (abcq <= abcqMax) && (exactMatch == (boolean_T)0U);
         abcq++)
    {
        realTq      = fQspi / ((real64_T)MaxBaudrate * (real64_T)abcq);
        tq          = realTq + 0.5;
        achievedMax = fQspi / (tq * (real64_T)abcq);
        deltaMax    = (achievedMax > (real64_T)MaxBaudrate) ?
                      (achievedMax - (real64_T)MaxBaudrate) :
                      ((real64_T)MaxBaudrate - achievedMax);

        if ((deltaMax <= bestDelta) && (tq >= 1.0))
        {
            bestDelta = deltaMax;
            bestTq    = (uint32_T)tq;
        }

        /* Rule 13.3 deviation: epsilon comparison avoids exact float==0.
         * deltaExactHz = 0.5 Hz is sub-ppm at 5 MHz — true exact match.  */
        if ((bestDelta < deltaExactHz) || (tq < 1.0))
        {
            exactMatch = (boolean_T)1U;   /* MISRA 15.4: single break via flag */
        }
    }

    return (bestTq > 0U) ? (bestTq - 1U) : 0U;
}

/**********************************************************************************************************************
 * ISR Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * qspi4_tx_handler  (SRPN 55, CPU0)
 *
 * Fires when the TX FIFO drains below the TXFIFOINT threshold (0 = empty).
 * Clears the TX service-request flag, marks the frame as transmitted.
 * Does NOT set received — the frame is still shifting out; the RX ISR
 * fires separately once the word is fully clocked in.
 *------------------------------------------------------------------------------------------------------------------*/
EMBED_SIM_INTERRUPT(qspi4_tx_handler, 0, CORE_00_QSPI4_TX_SRPN)
{
    volatile uint32_T discard;

    QSPI4_FLAGSCLEAR.B.TXC = 0x1U;       /* clear TX service-request flag    */    /* PRQA S 0303 */
    discard = QSPI4_STATUS.B.TXF;         /* read-back: flush write buffer    */    /* PRQA S 0303 */
    (void)discard;

    Qspi4State_S.transition  = 0x1U;
    Qspi4State_S.transmitted = 0x1U;
    /* received intentionally NOT set — frame still shifting */
}

/*--------------------------------------------------------------------------------------------------------------------
 * qspi4_rx_handler  (SRPN 56, CPU0)
 *
 * Fires when the RX FIFO level reaches the RXFIFOINT threshold (0 = ≥1 word).
 * Pops one word from RXEXIT, stores it in Qspi4RxWord_S, then sets received.
 * This is the frame-completion event that unblocks the foreground spin-wait.
 *
 * Read-back of FLAGSCLEAR after writing it flushes the TriCore store buffer,
 * guaranteeing the hardware sees the clear before the ISR returns.
 *------------------------------------------------------------------------------------------------------------------*/
EMBED_SIM_INTERRUPT(qspi4_rx_handler, 0, CORE_00_QSPI4_RX_SRPN)
{
    volatile uint32_T discard;

    QSPI4_FLAGSCLEAR.B.RXC = 0x1U;            /* clear RX service-request flag     */  /* PRQA S 0303 */
    discard = QSPI4_FLAGSCLEAR.B.RXC;          /* read-back: flush write buffer     */  /* PRQA S 0303 */
    (void)discard;

    Qspi4RxWord_S = QSPI4_RXEXIT.U;           /* destructive pop from RX FIFO      */  /* PRQA S 0303 */

    Qspi4State_S.transition  = 0x1U;
    Qspi4State_S.transmitted = 0x1U;
    Qspi4State_S.received    = 0x1U;           /* ← unblocks foreground spin-wait   */
    Qspi4State_S.error       = 0x0U;
}

/*--------------------------------------------------------------------------------------------------------------------
 * qspi4_err_handler  (SRPN 57, CPU0)
 *
 * Fires on any QSPI4 protocol error (parity, phase, overflow, underflow …).
 * Sets error AND received so that the foreground spin-wait always exits.
 * Without setting received, any error that suppresses the RX ISR would hang
 * the foreground loop indefinitely (defect present in the original polling
 * driver and in qspi_utility.c).
 *------------------------------------------------------------------------------------------------------------------*/
EMBED_SIM_INTERRUPT(qspi4_err_handler, 0, CORE_00_QSPI4_ERR_SRPN)
{
    volatile uint32_T discard;

    QSPI4_FLAGSCLEAR.B.ERRORCLEARS = 0x1U;     /* clear all error service-request flags */  /* PRQA S 0303 */
    discard = QSPI4_FLAGSCLEAR.B.ERRORCLEARS;   /* read-back: flush write buffer         */  /* PRQA S 0303 */
    (void)discard;

    Qspi4State_S.error    = 0x1U;
    Qspi4State_S.received = 0x1U;   /* always release spin-wait — prevents deadlock */
}

/**********************************************************************************************************************
 * Public Function Implementations
 *********************************************************************************************************************/

/*--------------------------------------------------------------------------------------------------------------------
 * CddQspi4_Init
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddQspi4_Init(P2VAR(uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    Ifx_QSPI_GLOBALCON    globalCon;
    Ifx_QSPI_GLOBALCON1   globalCon1;
    Ifx_QSPI_ECON         eCon;
    uint32_T              qVal;
    uint32_T              aVal;
    uint32_T              bVal;
    uint32_T              cVal;
    real32_T              fAchieved;
    uint32_T              result;      /* single return value — Rule 15.5 */

    *ErrorCode = CDD_QSPI_ERR_NONE;
    result     = 0x1U;

    globalCon.U  = 0x0U;
    globalCon1.U = 0x0U;
    eCon.U       = 0x0U;

    /* ── Enable QSPI4 clock ──────────────────────────────────────────────────────── */
    CddSys_ClearCpuWdtEndInit();
    QSPI4_CLC.B.DISR = 0x0U;                                               /* PRQA S 0303 */
    QSPI4_CLC.B.EDIS = 0x1U;                                               /* PRQA S 0303 */
    while (QSPI4_CLC.B.DISS != 0x0U)                                       /* PRQA S 0303 */
    {
        CddSys_NopDelay(1U, 1U);
    }
    CddSys_SetCpuWdtEndInit();

    /* ── GLOBALCON: master mode, TQ from 5 MHz target ───────────────────────────── */
    globalCon.U        = QSPI4_GLOBALCON.U;                                /* PRQA S 0303 */
    globalCon.B.MS     = 0x0U;
    globalCon.B.CLKSEL = 0x1U;
    globalCon.B.TQ     = CddQspi4_CalcTq(MHZ_50);
    globalCon.B.EXPECT = 0xFU;
    globalCon.B.LB     = 0x0U;
    globalCon.B.STROBE = 0x0U;
    QSPI4_GLOBALCON.U  = globalCon.U;                                      /* PRQA S 0303 */

    QSPI4_GLOBALCON.B.RESETS = 0x1U;                                       /* PRQA S 0303 */
    CddStm_Delay_Us(1U);

    /* ── GLOBALCON1: TX/RX enabled, no DMA ─────────────────────────────────────── */
    globalCon1.U           = QSPI4_GLOBALCON1.U;                           /* PRQA S 0303 */
    globalCon1.B.TXEN      = 0x1U;
    globalCon1.B.RXEN      = 0x1U;
    globalCon1.B.TXFIFOINT = 0x0U;
    globalCon1.B.RXFIFOINT = 0x0U;
    globalCon1.B.TXFM      = 0x0U;
    globalCon1.B.RXFM      = 0x0U;
    globalCon1.B.ERRORENS  = 0x1FFU;
    QSPI4_GLOBALCON1.U     = globalCon1.U;                                 /* PRQA S 0303 */

    /* ── Configure I/O pins ─────────────────────────────────────────────────────── */
    CddGpio_ConfigQspi4Sclk_P22_3();
    CddGpio_ConfigQspi4Cs_P22_2();
    CddGpio_ConfigQspi4Miso_P22_1();
    CddGpio_ConfigQspi4Mosi_P22_0();
    QSPI4_PISEL.B.MRIS = 0x1U;                                            /* PRQA S 0303 */

    /* ── ECON3: compute Q, A, B, C; verify baud rate before writing ────────────── */
    fAchieved = Cdd_Qspi4_Calc_Econ(CDD_QSPI4_BAUD_RATE_HZ,
                                     &qVal, &aVal, &bVal, &cVal);

    if (CddSys_AreEqual32(fAchieved, CDD_QSPI4_BAUD_RATE_HZ,
                          CDD_QSPI4_BAUD_RATE_EPSILON) == 0x0U)
    {
        *ErrorCode = CDD_QSPI_ERR_BAUD_MISMATCH;
        result     = 0x0U;
    }
    else
    {
        eCon.U        = QSPI4_ECON3.U;                                     /* PRQA S 0303 */
        eCon.B.Q      = qVal;
        eCon.B.A      = aVal;
        eCon.B.B      = bVal;
        eCon.B.C      = cVal;
        eCon.B.CPH    = 0x0U;    /* SPI Mode 0 — data captured on first edge */
        eCon.B.CPOL   = 0x0U;    /* SCK idle low                              */
        eCon.B.PAREN  = 0x0U;    /* no parity                                 */
        eCon.B.BE     = 0x0U;
        QSPI4_ECON3.U = eCon.U;                                            /* PRQA S 0303 */

        /* ── SSOC: enable SLSO3 output ──────────────────────────────────────────── */
        QSPI4_SSOC.B.OEN = 0x8U;                                           /* PRQA S 0303 */

        /* ── Wire ISR service request nodes (CPU0, SRE=1) ───────────────────────── */
        SRC_QSPI4TX.B.SRPN  = CORE_00_QSPI4_TX_SRPN;                      /* PRQA S 0303 */
        SRC_QSPI4TX.B.TOS   = 0x0U;
        SRC_QSPI4TX.B.CLRR  = 0x1U;
        SRC_QSPI4TX.B.SRE   = 0x1U;

        SRC_QSPI4RX.B.SRPN  = CORE_00_QSPI4_RX_SRPN;                      /* PRQA S 0303 */
        SRC_QSPI4RX.B.TOS   = 0x0U;
        SRC_QSPI4RX.B.CLRR  = 0x1U;
        SRC_QSPI4RX.B.SRE   = 0x1U;

        SRC_QSPI4ERR.B.SRPN = CORE_00_QSPI4_ERR_SRPN;                     /* PRQA S 0303 */
        SRC_QSPI4ERR.B.TOS  = 0x0U;
        SRC_QSPI4ERR.B.CLRR = 0x1U;
        SRC_QSPI4ERR.B.SRE  = 0x1U;

        /* ── Initialise ISR handshake state ─────────────────────────────────────── */
        Qspi4State_S.transition  = 0x0U;
        Qspi4State_S.transmitted = 0x0U;
        Qspi4State_S.received    = 0x0U;
        Qspi4State_S.error       = 0x0U;
        Qspi4RxWord_S            = 0x0U;

        /* ── Enable module ───────────────────────────────────────────────────────── */
        QSPI4_GLOBALCON.B.EN = 0x1U;                                       /* PRQA S 0303 */
    }

    return result;
}
/*--------------------------------------------------------------------------------------------------------------------
 * CddQspi4_GetBaudRate
 *
 * Reads back the actual QSPI4 SCK frequency from hardware registers.
 *
 * Mirrors get_qspi0_ch12_baud_rate_frequency() exactly (USM2 §31, p1702):
 *
 *   f  = CddSys_GetQspiFreq()
 *   f  = f / (TQ + 1)
 *   f  = f / (Q  + 1)
 *   f  = f / ((A + 1) + B + C)
 *
 * All arithmetic in real32_T, matching the original return type.
 * No integer intermediate — no precision loss before CddSys_AreEqual32().
 *------------------------------------------------------------------------------------------------------------------*/
real32_T CddQspi4_GetBaudRate(void)
{
    real32_T freq;

    freq = (real32_T)CddSys_GetQspiFreq();                                 /* fPeriph [Hz] */

    freq = freq / ((real32_T)QSPI4_GLOBALCON.B.TQ + 1.0F);                /* PRQA S 0303 */
    freq = freq / ((real32_T)QSPI4_ECON3.B.Q      + 1.0F);                /* PRQA S 0303 */
    freq = freq / (((real32_T)QSPI4_ECON3.B.A     + 1.0F) +               /* PRQA S 0303 */
                    (real32_T)QSPI4_ECON3.B.B              +               /* PRQA S 0303 */
                    (real32_T)QSPI4_ECON3.B.C);                            /* PRQA S 0303 */

    return freq;
}

/*--------------------------------------------------------------------------------------------------------------------
 * CddQspi4_Exchange
 *
 * Transmit and receive one 24-bit frame, ISR-driven.  Mirrors exchange_qspi0()
 * from qspi_utility.c exactly:
 *
 *   1. Guard: all four state flags must be false (bus idle).
 *   2. Disable interrupts; set transition flag; write BACONENTRY twice
 *      (LAST=0 pre-load, then LAST=1 final) — TC3xx FIFO requires both
 *      BACON entries before DATA.  Re-enable interrupts.
 *   3. Write DATAENTRY0 — triggers the shift clock.
 *   4. Spin-wait on Qspi4State_S.received (set by RX ISR or ERR ISR).
 *   5. On success: copy Qspi4RxWord_S to *RxWord.
 *   6. Reset all four flags — bus released.
 *------------------------------------------------------------------------------------------------------------------*/
uint32_T CddQspi4_Exchange(
    P2CONST(uint32_T, AUTOMATIC, CDD_APPL_DATA) TxWord,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) RxWord,
    P2VAR  (uint32_T, AUTOMATIC, CDD_APPL_DATA) ErrorCode)
{
    static const uint32_T csChannel     = 3U;      /* BACON.CS → SLSO3 on P22.2        */
    static const uint32_T frameLenM1    = 0x17U;   /* BACON.DL: 24 bits − 1 = 23       */
    static const uint32_T rxPollTimeout = 10000U;  /* spin-wait loop limit              */

    uint32_T       timeout;
    uint32_T       result = 0x1U;
    Ifx_QSPI_BACON bacon;                          /* PRQA S 0750                       */

    *ErrorCode = CDD_QSPI_ERR_NONE;

    /* ── Guard: bus must be idle ─────────────────────────────────────────────────── */
    if ((Qspi4State_S.received    != 0x0U) ||
        (Qspi4State_S.transmitted != 0x0U) ||
        (Qspi4State_S.transition  != 0x0U) ||
        (Qspi4State_S.error       != 0x0U))
    {
        *ErrorCode = CDD_QSPI_ERR_RX_TIMEOUT;
        result     = 0x0U;
    }
    else
    {
        /* ── Build BACON word ────────────────────────────────────────────────────── */
        bacon.U        = 0x0U;
        bacon.B.MSB    = 0x1U;
        bacon.B.CS     = csChannel;
        bacon.B.DL     = frameLenM1;
        bacon.B.BYTE   = 0x0U;
        bacon.B.LAST   = 0x0U;

        /* ── Critical section: set flag, push LAST=0 then LAST=1 into FIFO ─────────
         * Matches exchange_qspi0: two BACONENTRY writes before DATAENTRY0.
         * TC3xx RM §31: BACON must precede DATA in the write order.            */
        __disable();
        Qspi4State_S.transition = 0x1U;
        QSPI4_BACONENTRY.U = bacon.U;              /* pre-load entry  LAST=0   */  /* PRQA S 0303 */
        bacon.B.LAST       = 0x1U;
        QSPI4_BACONENTRY.U = bacon.U;              /* final entry     LAST=1   */  /* PRQA S 0303 */
        __enable();

        /* ── DATAENTRY0 — triggers the shift clock ───────────────────────────────── */
        QSPI4_DATAENTRY0.U = *TxWord;              /* PRQA S 0303 */

        /* ── Spin-wait: RX ISR or ERR ISR sets received ─────────────────────────── */
        timeout = rxPollTimeout;
        while ((Qspi4State_S.received == 0x0U) && (timeout > 0x0U))
        {
            timeout--;
        }

        /* ── Evaluate outcome ────────────────────────────────────────────────────── */
        if (timeout == 0x0U)
        {
            *ErrorCode = CDD_QSPI_ERR_RX_TIMEOUT;
            result     = 0x0U;
        }
        else if (Qspi4State_S.error != 0x0U)
        {
            *ErrorCode = CDD_QSPI_ERR_ISR_ERROR;
            result     = 0x0U;
        }
        else
        {
            *RxWord = Qspi4RxWord_S;               /* copy word captured in RX ISR     */
        }

        /* ── Reset all flags — bus released ─────────────────────────────────────── */
        Qspi4State_S.transition  = 0x0U;
        Qspi4State_S.transmitted = 0x0U;
        Qspi4State_S.received    = 0x0U;
        Qspi4State_S.error       = 0x0U;
    }

    return result;
}
