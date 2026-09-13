/**********************************************************************************************************************
 * \file      cdd_encoder_app.c
 * \brief     Incremental Encoder Driver Implementation
 *
 * \details   Implements the complete incremental encoder interface for motor control
 *            applications using the Infineon GPT12 module in Incremental Interface
 *            Mode (Mode 6).
 *
 *            ## Speed calculation
 *            Direct per-tick pulse-count mode. The signed count delta since the
 *            last 20 kHz tick is converted to rad/s with a single float multiply:
 *
 *                diff     = signed count change since the last 20 kHz tick
 *                rawSpeed = (real32_T)diff * T3SpeedConversionQuotient
 *                SpeedRad = (SPEED_LPF_ALPHA * rawSpeed)
 *                         + (SPEED_LPF_ONE_MINUS_ALPHA * SpeedRad)
 *
 *            where T3SpeedConversionQuotient = 2*pi / (EncoderResolution * UpdatePeriod).
 *            Direction is baked into `diff` itself, so no separate sign step.
 *
 *            ## Why there is no integer divide
 *            At 800 RPM with 4000 CPR at 20 kHz the true per-tick delta is
 *            2.6667 counts; the encoder dithers {3,3,2,3,3,2,...}. An earlier
 *            design accumulated N of these into an integer moving-average and
 *            divided by N in integer, which truncated every window sum (10 or
 *            11) down to 2 and locked SpeedRad at 2 * k = 600 RPM. That round
 *            buffer has been removed entirely. The current code performs no
 *            integer division on the speed path at all: `diff` is cast to
 *            real32_T and multiplied by the conversion quotient. The 800-vs-600
 *            defect is therefore structurally impossible.
 *
 *            ## Speed-domain IIR low-pass
 *            The raw per-tick speed is quantisation-noisy (dither {3,3,2,...}
 *            is visible directly), so a first-order IIR is applied in the
 *            SpeedRad domain before publication:
 *
 *                SpeedRad = alpha * rawSpeed + (1 - alpha) * SpeedRad
 *
 *            with alpha = SPEED_LPF_ALPHA (0.0589), giving approximately a
 *            600 Hz cutoff at the 20 kHz update rate. Downstream consumers
 *            receive the already-filtered SpeedRad; they are not expected to
 *            re-filter. RotorAngle is not filtered: it is derived directly
 *            from T3 and is exact to one count.
 *
 *            The former T5 / CAPREL time-diff path has been removed: on this
 *            target the T5SC / T5CLR capture bits were not producing a CAPREL
 *            update on encoder edges, so any update with diff <= threshold fell
 *            into the CAPREL branch, read 0, and held SpeedRad at 0.
 *
 *            ## 16-bit T3 is not a modulo-4000 counter
 *            T3 lives in the full 16-bit space, even with CLRT3EN = 1. Over one
 *            mechanical revolution it visits:
 *                CW  : 0x0000 .. 0x0F9F   ( 0 .. +3999 )
 *                ACW : 0xF061 .. 0xFFFF   ( -3999 .. -1 as int16_T )
 *            The delta path therefore casts T3 to int16_T and subtracts in
 *            int32_T, mirroring the Infineon iLLD GPT12 incremental-encoder
 *            driver.
 *
 *            ## Index-Reset Architecture (CLRT3EN = 1)
 *            The Z-index pulse hardware-clears T3 to zero. The ISR sets
 *            ZEventPending so the very next update skips one speed delta; that
 *            update still latches T3CounterPrev so the following tick computes
 *            its delta from the post-Z position.
 *
 *            ## Corner case - wrap-around at Z-boundary
 *            Because T3 is hardware-cleared by T4 on every Z-pulse, the raw
 *            delta between two consecutive ticks can appear huge even though
 *            the physical movement was small:
 *
 *                CW  : prev=3998, new=6    -> raw diff = 6 - 3998    = -3992
 *                ACW : prev=6,    new=3994 -> raw diff = 3994 - 6    = +3988
 *
 *            Both are corrected by comparing |diff| against half the encoder
 *            resolution and adding/subtracting ENCODER_COUNTS_PER_REV. The
 *            corrected values are +8 (CW) and -12 (ACW) respectively.
 *
 *            ## Concurrency
 *            Encoder_Index_ISR runs asynchronously to CddEncoder_Update and to
 *            the getters. The 64-bit TurnCount is updated only from the ISR and
 *            is not read by any function in this driver; any external reader
 *            MUST wrap its read in a critical section, because 64-bit accesses
 *            are not atomic on TriCore.
 *
 * \note      MISRA C:2012 compliance:
 *              - Rule  8.1 : All functions have explicit return types.
 *              - Rule  8.5 : One declaration per identifier.
 *              - Rule  8.6 : No definitions in header files.
 *              - Rule  8.7 : Internal linkage for static helper
 *                            (CddEncoder_InitHardware).
 *              - Rule  8.9 : No file-scope object with internal linkage.
 *                            EncoderState_G has external linkage by design
 *                            (declared extern in the header).
 *              - Rule 13.1 : No initializer lists with side effects.
 *              - Rule 13.2 : No persistent side effects in the RHS of the
 *                            assignment expression.
 *              - Rule 14.4 : Controlling expressions are essentially Boolean
 *                            (uint32_T compared against 0U / 1U, int32_T
 *                            compared against a computed threshold).
 *              - Rule 15.5 : Single exit point per function.
 *              - Rule 17.2 : No recursion.
 *              - Rule 18.4 : No non-constant pointer arithmetic.
 *
 *              Deviations, each flagged inline in this file with a
 *              "MISRA-DEV" comment:
 *              - Rule 10.3 : deliberate narrowing casts in the delta path
 *                            (uint16_T -> int16_T, int32_T -> real32_T).
 *              - Rule 10.4 : mixed int32_T / real32_T arithmetic at the
 *                            speed conversion.
 *              - Rule 10.5 : casts between signed and unsigned integer types.
 *
 * \note      EmbedSim naming convention:
 *              - Functions      : Pascal_Snake_Case
 *              - Parameters     : PascalCase
 *              - Output pointers: PascalCasePtr
 *              - Local variables: Lower camelCase
 *              - Struct members : PascalCase
 *              - Macros         : UPPER_SNAKE_CASE
 *              - Typedefs       : Pascal_Snake_Case_T
 *
 * \version   2.8.1
 * \date      2026-09-13
 * \author    EmbedSim / EV Light Vehicle Foundation
 *
 * \copyright Copyright (C) 2026 EmbedSim - EV Light Vehicle Foundation, Jaffna, Sri Lanka.
 *            Licensed under the MIT License.
 *********************************************************************************************************************/

#include "cdd_encoder_app.h"
#include "cdd_config.h"
#include "embed_sim_sys_types.h"
#include "IfxGpt12_reg.h"
#include "IfxGpt12_bf.h"
#include "IfxSrc_reg.h"
#include "IfxGpt12.h"
#include <stddef.h>
#include "cdd_sys_utility.h"

/*-------------------------------------------------Global variables--------------------------------------------------*/

/** \brief Global encoder state structure instance. */
CddEncoder_State_T EncoderState_G;

/*-------------------------------------------------Private functions-------------------------------------------------*/

/**
 * \brief   Internal function to initialize the GPT12 hardware.
 * \return  void
 *
 * \details Configures T3 for incremental interface mode and T4 for Z-index
 *          capture with automatic T3 clearing.
 */
static void CddEncoder_InitHardware(void)
{
    Ifx_GPT12_T3CON t3conCfg;
    Ifx_GPT12_T4CON t4conCfg;
    Ifx_GPT12_PISEL piselCfg;
    Ifx_SRC_SRCR    srcCfg;

    t3conCfg.U = GPT120_T3CON.U;
    t4conCfg.U = GPT120_T4CON.U;
    piselCfg.U = GPT120_PISEL.U;
    srcCfg.U   = SRC_GPT12_GPT120_T4.U;

    t3conCfg.B.BPS1  = 0x0U;
    t3conCfg.B.T3M   = 0x6U;
    t3conCfg.B.T3I   = 0x3U;
    t3conCfg.B.T3UDE = 0x1U;
    t3conCfg.B.T3OE  = 0x0U;
    GPT120_T3CON.U = t3conCfg.U;

    GPT120_T3.U = 0x0000U;
    GPT120_T3CON.B.T3R = 0x1U;

    t4conCfg.B.T4M     = 0x5U;
    t4conCfg.B.T4I     = 0x1U;
    t4conCfg.B.CLRT3EN = 0x1U;
    t4conCfg.B.CLRT2EN = 0x0U;
    t4conCfg.B.T4IRDIS = 0x0U;
    t4conCfg.B.T4RC    = 0x0U;
    t4conCfg.B.T4R     = 0x0U;
    GPT120_T4CON.U = t4conCfg.U;

    piselCfg.B.IST3IN  = 0x0U;
    piselCfg.B.IST3EUD = 0x0U;
    piselCfg.B.IST4IN  = 0x0U;
    GPT120_PISEL.U = piselCfg.U;

    srcCfg.B.SRPN = CORE_00_GPT12_ENCODER_ZERO_SRPN;
    srcCfg.B.TOS  = 0x0U;
    srcCfg.B.CLRR = 0x1U;
    SRC_GPT12_GPT120_T4.U = srcCfg.U;
    SRC_GPT12_GPT120_T4.B.SRE = 0x1U;
}

/*-------------------------------------------------ISR Implementations-----------------------------------------------*/

EMBED_SIM_INTERRUPT(Encoder_Index_ISR, 0x0U, CORE_00_GPT12_ENCODER_ZERO_SRPN);

/**
 * \brief   Z-index pulse ISR.
 *
 * \warning TurnCount is a 64-bit value written here without a critical
 *          section. It is not read anywhere inside this driver. Any future
 *          external reader MUST protect the read against this ISR, because
 *          64-bit accesses are not atomic on TriCore.
 *
 * \warning This ISR runs at the T4 interrupt priority configured by
 *          CORE_00_GPT12_ENCODER_ZERO_SRPN. It must remain short; it currently
 *          performs only register accesses and one 64-bit accumulate.
 */
void Encoder_Index_ISR(void)
{
    uint32_T directionNow;
    int64_T turnDelta;

    directionNow = (uint32_T)GPT120_T3CON.B.T3RDIR;
    turnDelta = (directionNow == (uint32_T)ENC_DIR_CW) ? 1LL : -1LL;

    EncoderState_G.TurnCount += turnDelta;
    EncoderState_G.Direction = directionNow;
    EncoderState_G.ZEventPending = 1U;

    SRC_GPT12_GPT120_T4.B.CLRR = 0x1U;
}

/*---------------------------------------------Function Implementations----------------------------------------------*/

/**
 * \brief   Initialize the encoder driver.
 * \return  uint32_T - initialization status.
 */
uint32_T CddEncoder_Init(void)
{
    if (EncoderState_G.Initialized == 1U)
    {
        /* Already initialized. */
    }
    else
    {
        EncoderState_G.SpeedRad = 0.0F;
        EncoderState_G.SpeedRpm = 0.0F;
        EncoderState_G.RotorAngle = 0.0F;
        EncoderState_G.T3CounterPrev = 0U;
        EncoderState_G.ZEventPending = 0U;
        EncoderState_G.TurnCount = 0LL;
        EncoderState_G.Direction = (uint32_T)ENC_DIR_CW;
        EncoderState_G.EncoderResolution = ENCODER_COUNTS_PER_REV;
        EncoderState_G.UpdatePeriod = ENCODER_UPDATE_PERIOD;
        EncoderState_G.CountsToRadians = ENCODER_COUNTS_TO_RAD;

        EncoderState_G.T3SpeedConversionQuotient =
            ES_MATH_2PI_F /
            ((real32_T)EncoderState_G.EncoderResolution * EncoderState_G.UpdatePeriod);

        /* ----- Enable the GPT12 module clock ----- */
        CddSys_ClearCpuWdtEndInit();                   /**< Disable watchdog      */
        GPT120_CLC.B.DISR = 0x0U;                      /**< Exit module reset     */
        CddSys_SetCpuWdtEndInit();                     /**< Re-enable watchdog    */

        CddEncoder_InitHardware();

        /* Synchronize the software snapshot with the actual hardware counter. */
        EncoderState_G.T3CounterPrev = (uint16_T)GPT120_T3.U;
        EncoderState_G.Initialized = 1U;
    }

    return EncoderState_G.Initialized;
}

/**
 * \brief   Update encoder state (call at 20 kHz).
 * \return  void
 *
 * \details Direct signed per-tick pulse-count speed, then a first-order IIR
 *          low-pass in the SpeedRad domain. ZEventPending suppresses the
 *          delta spanning the hardware T3 reset. A half-range fallback
 *          protects against the small ISR/update scheduling race at Z.
 *
 *          Positive speed = CW, negative speed = ACW.
 */
void CddEncoder_Update(void)
{
    uint16_T currentT3Counter;
    int32_T  newPosition;
    int32_T  prevPosition;
    int32_T  diff;
    int32_T  angleCount;
    int32_T  resetThreshold;
    real32_T rawSpeedRad;
    uint32_T zEventPending;

    currentT3Counter = (uint16_T)GPT120_T3.U;

    /* MISRA-DEV Rule 10.3: narrowing uint16_T -> int16_T is intentional; the
     * two's-complement reinterpretation is what turns the raw T3 register
     * into a signed position. */
    newPosition  = (int32_T)(int16_T)currentT3Counter;
    prevPosition = (int32_T)(int16_T)EncoderState_G.T3CounterPrev;

    zEventPending = EncoderState_G.ZEventPending;

    if (zEventPending != 0U)
    {
        /*
         * T4 has cleared T3 at Z. The reset-spanning delta is not a physical
         * speed measurement, so discard it and synchronize to post-Z T3.
         */
        EncoderState_G.ZEventPending = 0U;
        EncoderState_G.T3CounterPrev = currentT3Counter;
    }
    else
    {
        diff = newPosition - prevPosition;

        /*
         * T3 is NOT modulo 4000. This correction is only a fallback for the
         * scheduling race in which T3 was cleared by Z before the ISR set
         * ZEventPending.
         *
         * At the supported maximum speed, the physical per-tick delta is far
         * below half a revolution, so a delta beyond half a revolution is
         * necessarily the artificial Z-reset discontinuity.
         */
        resetThreshold = (int32_T)(ENCODER_COUNTS_PER_REV / 2U);

        if (diff > resetThreshold)
        {
            diff -= (int32_T)ENCODER_COUNTS_PER_REV;
        }
        else if (diff < -resetThreshold)
        {
            diff += (int32_T)ENCODER_COUNTS_PER_REV;
        }
        else
        {
            /* Normal signed delta. */
        }

        /* MISRA-DEV Rule 10.3 / 10.4: int32_T -> real32_T and mixed-mode
         * arithmetic with T3SpeedConversionQuotient (real32_T). This is the
         * single deliberate narrowing on the speed path; no integer divide
         * is performed. */
        rawSpeedRad = (real32_T)diff * EncoderState_G.T3SpeedConversionQuotient;

        /* First-order IIR low-pass in the SpeedRad domain.
         * MISRA-DEV Rule 10.4: mixed SPEED_LPF_*_F and real32_T operands. */
        EncoderState_G.SpeedRad =
            (SPEED_LPF_ALPHA * rawSpeedRad) +
            (SPEED_LPF_ONE_MINUS_ALPHA * EncoderState_G.SpeedRad);

        EncoderState_G.T3CounterPrev = currentT3Counter;
    }

    EncoderState_G.SpeedRpm = EncoderState_G.SpeedRad * ENCODER_RAD_PER_SEC_TO_RPM;

    /*
     * Convert signed T3 position to an unsigned mechanical angle:
     *
     *     CW  : 0 .. 3999
     *     ACW : -1 .. -3999 -> 3999 .. 1
     */
    angleCount = newPosition;

    if (angleCount < 0)
    {
        angleCount += (int32_T)ENCODER_COUNTS_PER_REV;
    }
    else
    {
        /* Positive T3 position is already in the required range. */
    }

    /* MISRA-DEV Rule 10.3: int32_T -> real32_T conversion at the angle output. */
    EncoderState_G.RotorAngle = (real32_T)angleCount * EncoderState_G.CountsToRadians;
}

/**
 * \brief   Reset the encoder to a known state.
 * \return  void
 */
void CddEncoder_Reset(void)
{
    if (EncoderState_G.Initialized == 0U)
    {
        /* Not initialized. */
    }
    else
    {
        GPT120_T3CON.B.T3R = 0x0U;
        GPT120_T3.U = 0x0000U;

        EncoderState_G.RotorAngle = 0.0F;
        EncoderState_G.SpeedRad = 0.0F;
        EncoderState_G.SpeedRpm = 0.0F;
        EncoderState_G.T3CounterPrev = 0U;
        EncoderState_G.ZEventPending = 0U;
        EncoderState_G.TurnCount = 0LL;
        EncoderState_G.Direction = (uint32_T)ENC_DIR_CW;

        GPT120_T3CON.B.T3R = 0x1U;
    }
}

real32_T CddEncoder_GetRotorPosition(void)
{
    real32_T angle = 0.0F;

    if (EncoderState_G.Initialized == 1U)
    {
        angle = EncoderState_G.RotorAngle;
    }
    else
    {
        /* Return zero if not initialized. */
    }

    return angle;
}

real32_T CddEncoder_GetSpeedRad(void)
{
    real32_T speedRad = 0.0F;

    if (EncoderState_G.Initialized == 1U)
    {
        speedRad = EncoderState_G.SpeedRad;
    }
    else
    {
        /* Return zero if not initialized. */
    }

    return speedRad;
}

real32_T CddEncoder_GetSpeedRpm(void)
{
    real32_T speedRpm = 0.0F;

    if (EncoderState_G.Initialized == 1U)
    {
        speedRpm = EncoderState_G.SpeedRpm;
    }
    else
    {
        /* Return zero if not initialized. */
    }

    return speedRpm;
}

uint32_T CddEncoder_GetDirection(void)
{
    uint32_T direction = (uint32_T)ENC_DIR_CW;

    if (EncoderState_G.Initialized == 1U)
    {
        direction = EncoderState_G.Direction;
    }
    else
    {
        /* Return CW as default if not initialized. */
    }

    return direction;
}

uint32_T CddEncoder_IsInitialized(void)
{
    return EncoderState_G.Initialized;
}
